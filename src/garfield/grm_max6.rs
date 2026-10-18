//! Full exact GRM-Max6 window geometry.
//!
//! The shared geometry computes `Z=P_C X`, `A=X'Z`, and
//! `H=X'(X\odot Z)`. The default q path is the exact Phase-6B mincell
//! kernel; the legacy, spectral, and binary-support subset kernels remain
//! available as explicit reference/debug alternatives.
//!
//! The production dosage channel is `fwl_product` (historical GRM-Max6).
//! `centered_product` is an EXPERIMENTAL GRM-nMax6 ablation and must remain
//! opt-in and separately labeled in callers, benchmarks, and reports.

use super::residual::NoGrmCovariateProjector;
use crate::blas::{
    cblas_dgemm_dispatch, ensure_garfield_blas_backend, CblasInt, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR,
    CBLAS_TRANS,
};
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::BoundObject;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

const DENOMINATOR_EPS: f64 = 1.0e-12;

#[derive(Clone, Debug)]
pub(crate) struct Phase1Geometry {
    pub(crate) pair_i: Vec<usize>,
    pub(crate) pair_j: Vec<usize>,
    pub(crate) z: Vec<f64>,
    pub(crate) a: Vec<f64>,
    pub(crate) h: Vec<f64>,
    pub(crate) q: Vec<f64>,
    pub(crate) mincell_q: Option<Vec<f64>>,
    pub(crate) mincell_cell: Option<Vec<u8>>,
    pub(crate) mincell_support: Option<Vec<usize>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Phase1Timing {
    pub(crate) pcx_seconds: f64,
    pub(crate) a_seconds: f64,
    pub(crate) xz_build_seconds: f64,
    pub(crate) h_gram_seconds: f64,
    pub(crate) h_seconds: f64,
    pub(crate) q_seconds: f64,
    pub(crate) geometry_seconds: f64,
    pub(crate) subset_work: Option<SubsetQWork>,
    pub(crate) subset_q_timing: Option<SubsetQTiming>,
    pub(crate) mincell_work: Option<MinCellQWork>,
    pub(crate) mincell_q_timing: Option<MinCellQTiming>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Phase1ScoreTiming {
    pub(crate) py_seconds: f64,
    pub(crate) b_seconds: f64,
    pub(crate) weighted_gram_seconds: f64,
    pub(crate) score_allocation_seconds: f64,
    pub(crate) dosage_score_seconds: f64,
    pub(crate) logic_score_seconds: f64,
    pub(crate) max6_reduce_seconds: f64,
    pub(crate) target_rank_seconds: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SubsetQWork {
    pub(crate) work_subset: f64,
    pub(crate) work_dense: f64,
    pub(crate) work_ratio: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SubsetQTiming {
    pub(crate) support_seconds: f64,
    pub(crate) psub_seconds: f64,
    pub(crate) xsub_seconds: f64,
    pub(crate) gemm_seconds: f64,
    pub(crate) reduce_seconds: f64,
    pub(crate) alloc_zero_seconds: f64,
    pub(crate) workspace_allocations: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MinCellQWork {
    pub(crate) work_mincell: f64,
    pub(crate) work_subset: f64,
    pub(crate) work_dense: f64,
    pub(crate) mincell_subset_ratio: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MinCellQTiming {
    pub(crate) validation_seconds: f64,
    pub(crate) bitset_seconds: f64,
    pub(crate) cell_seconds: f64,
    pub(crate) enumerate_seconds: f64,
    pub(crate) quadratic_seconds: f64,
    pub(crate) reconstruct_seconds: f64,
}

impl MinCellQTiming {
    fn sum(self) -> f64 {
        self.validation_seconds
            + self.bitset_seconds
            + self.cell_seconds
            + self.enumerate_seconds
            + self.quadratic_seconds
            + self.reconstruct_seconds
    }
}

impl SubsetQTiming {
    fn sum(self) -> f64 {
        self.support_seconds
            + self.psub_seconds
            + self.xsub_seconds
            + self.gemm_seconds
            + self.reduce_seconds
            + self.alloc_zero_seconds
    }
}

struct SubsetQResult {
    q: Vec<f64>,
    work_subset: f64,
    work_dense: f64,
    work_ratio: f64,
    timing: SubsetQTiming,
}

struct MinCellQResult {
    q_min: Vec<f64>,
    cell: Vec<u8>,
    support: Vec<usize>,
    work: MinCellQWork,
    timing: MinCellQTiming,
}

struct MinSupportEncoding {
    x: Vec<f64>,
    flipped: Vec<bool>,
}

#[derive(Debug, PartialEq)]
struct SampleRowWeightedCResult {
    c_upper: Vec<f64>,
    w_anchor: u64,
    w_row: u64,
    w_dense: u64,
}

#[derive(Debug)]
struct CandidatePairSet {
    pair_i: Vec<usize>,
    pair_j: Vec<usize>,
    input_to_unique: Vec<i64>,
    input_self_count: usize,
    input_duplicate_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CandidateRerankTiming {
    pub(crate) endpoint_prepare_seconds: f64,
    pub(crate) pcx_u_seconds: f64,
    pub(crate) pcx_sparse_support: bool,
    pub(crate) py_seconds: f64,
    pub(crate) candidate_geometry_seconds: f64,
    pub(crate) geometry_marker_moments_seconds: f64,
    pub(crate) geometry_pairwise_seconds: f64,
    pub(crate) geometry_au_validation_seconds: f64,
    pub(crate) geometry_au_allocation_seconds: f64,
    pub(crate) geometry_au_materialization_seconds: f64,
    pub(crate) geometry_au_gemm_seconds: f64,
    pub(crate) geometry_au_candidate_lookup_seconds: f64,
    /// Fused support traversal time; a detail within pairwise geometry time.
    pub(crate) geometry_au_fused_py_seconds: f64,
    pub(crate) geometry_au_copy_seconds: f64,
    pub(crate) geometry_au_workspace_bytes: usize,
    pub(crate) geometry_lookup_seconds: f64,
    pub(crate) geometry_interaction_py_seconds: f64,
    pub(crate) q_seconds: f64,
    pub(crate) q_validation_seconds: f64,
    pub(crate) q_bitset_seconds: f64,
    pub(crate) q_cell_seconds: f64,
    pub(crate) q_enumerate_seconds: f64,
    pub(crate) q_quadratic_seconds: f64,
    pub(crate) q_reconstruct_seconds: f64,
    pub(crate) score_seconds: f64,
    pub(crate) topk_seconds: f64,
    pub(crate) kernel_seconds: f64,
}

#[derive(Debug)]
pub(crate) struct CandidateRerankResult {
    pub(crate) pair_i: Vec<usize>,
    pub(crate) pair_j: Vec<usize>,
    pub(crate) input_to_unique: Vec<i64>,
    pub(crate) input_self_count: usize,
    pub(crate) input_duplicate_count: usize,
    pub(crate) global_pair_id: Vec<usize>,
    pub(crate) unique_markers: Vec<usize>,
    pub(crate) z: Vec<f64>,
    pub(crate) q: Vec<f64>,
    pub(crate) mincell_q: Vec<f64>,
    pub(crate) mincell_cell: Vec<u8>,
    pub(crate) mincell_support: Vec<usize>,
    pub(crate) logic: Vec<[f64; 5]>,
    pub(crate) dosage: Vec<f64>,
    pub(crate) max6: Vec<f64>,
    pub(crate) best_logic_gate: Vec<i64>,
    pub(crate) best_max6_channel: Vec<&'static str>,
    pub(crate) candidate_rank: Vec<usize>,
    pub(crate) top_order: Vec<usize>,
    pub(crate) flipped: Vec<bool>,
    pub(crate) timing: CandidateRerankTiming,
}

#[inline]
fn validate_positive_score_scale(score_scale: f64) -> Result<(), String> {
    if score_scale.is_finite() && score_scale > 0.0 {
        Ok(())
    } else {
        Err(format!(
            "GRM-Max6 score scale must be finite and positive, got {score_scale}"
        ))
    }
}

/// Apply the covariance-scale factor to all exact channel scores.
///
/// `P_C` is stored as `(K + lambda I)^-1` with the fixed-effect projection.
/// The full LMM precision is `(1 / sigma_g^2) * P_C`, so the corresponding
/// quadratic-form score needs the same positive scalar.  A positive common
/// scale leaves channel labels and ranks unchanged, but makes scores and
/// permutation thresholds comparable across phenotypes.
fn apply_score_scale(result: &mut CandidateRerankResult, score_scale: f64) {
    for logic in result.logic.iter_mut() {
        for value in logic.iter_mut() {
            *value *= score_scale;
        }
    }
    for value in result.dosage.iter_mut() {
        *value *= score_scale;
    }
    for value in result.max6.iter_mut() {
        *value *= score_scale;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DosageChannel {
    /// Production GRM-Max6 dosage channel: residualize the raw product
    /// against the two endpoint markers in the P_C metric.
    FwlProduct,
    /// EXPERIMENTAL GRM-nMax6 dosage channel: use the centered endpoint
    /// product directly in the same P_C quadratic-form statistic as the five
    /// logic channels.
    ///
    /// This option is intentionally opt-in and is not interchangeable with
    /// `FwlProduct`. Keep `centered_product` results separate from production
    /// GRM-Max6 results; do not silently pair it with the formal Max6 label.
    CenteredProduct,
}

impl DosageChannel {
    fn name(self) -> &'static str {
        match self {
            Self::FwlProduct => "fwl_product",
            Self::CenteredProduct => "centered_product",
        }
    }
}

const MAX6_CHANNEL_NAMES: [&str; 6] = ["AND00", "AND01", "AND10", "AND11", "XOR", "DOSAGE_PRODUCT"];

/// Return the winning production GRM-Max6 channel, including the sixth
/// dosage channel. Non-finite channels are ignored; exact ties retain the
/// fixed channel order above.
fn best_max6_channel_name(logic: &[f64; 5], dosage: f64) -> &'static str {
    let mut best: Option<(usize, f64)> = None;
    for (index, &score) in logic.iter().enumerate() {
        if !score.is_finite() {
            continue;
        }
        if best
            .map(|(_, best_score)| score > best_score)
            .unwrap_or(true)
        {
            best = Some((index, score));
        }
    }
    if dosage.is_finite()
        && best
            .map(|(_, best_score)| dosage > best_score)
            .unwrap_or(true)
    {
        best = Some((5, dosage));
    }
    best.map(|(index, _)| MAX6_CHANNEL_NAMES[index])
        .unwrap_or("NONE")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WeightedCKernel {
    DenseReference,
    BinarySparse,
    MinorSupport,
    SampleRowSparse,
}

#[derive(Clone, Debug)]
pub(crate) struct Phase1Scores {
    pub(crate) logic: Vec<[f64; 5]>,
    pub(crate) dosage: Vec<f64>,
    pub(crate) max6: Vec<f64>,
    pub(crate) target_index: Option<usize>,
    pub(crate) max6_rank: Option<usize>,
}

fn checked_matrix_len(rows: usize, cols: usize, label: &str) -> Result<usize, String> {
    rows.checked_mul(cols)
        .ok_or_else(|| format!("{label}: matrix size overflow for {rows} x {cols}"))
}

fn validate_genotype_inputs(x: &[f64], n_samples: usize, n_markers: usize) -> Result<(), String> {
    let x_len = checked_matrix_len(n_samples, n_markers, "genotype")?;
    if x.len() != x_len {
        return Err(format!(
            "genotype length mismatch: got {}, expected {x_len}",
            x.len()
        ));
    }
    if n_samples == 0 || n_markers < 2 {
        return Err("full GRM-Max6 requires samples > 0 and at least two markers".to_string());
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err("genotype values must be finite".to_string());
    }
    Ok(())
}

fn resolve_dosage_channel(value: Option<&str>) -> Result<DosageChannel, String> {
    // Keep the production path explicit. `centered_product` is the
    // experimental GRM-nMax6 ablation and must never become an accidental
    // default through a missing/renamed configuration value.
    match value.unwrap_or("fwl_product") {
        "fwl_product" | "fwl" | "grm_fwl" => Ok(DosageChannel::FwlProduct),
        "centered_product" | "product" | "unified_product" => Ok(DosageChannel::CenteredProduct),
        other => Err(format!(
            "unknown dosage_channel {other:?}; expected fwl_product or centered_product"
        )),
    }
}

fn validate_inputs(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: Option<&[f64]>,
) -> Result<(), String> {
    validate_genotype_inputs(x, n_samples, n_markers)?;
    let pc_len = checked_matrix_len(n_samples, n_samples, "P_C")?;
    if p_c.len() != pc_len {
        return Err(format!(
            "P_C length mismatch: got {}, expected {pc_len}",
            p_c.len()
        ));
    }
    if p_c.iter().any(|value| !value.is_finite()) {
        return Err("P_C values must be finite".to_string());
    }
    if let Some(y) = y {
        if y.len() != n_samples {
            return Err(format!(
                "phenotype length mismatch: got {}, expected {n_samples}",
                y.len()
            ));
        }
        if y.iter().any(|value| !value.is_finite()) {
            return Err("phenotype values must be finite".to_string());
        }
    }
    Ok(())
}

/// Cheap shape-only validation for an internal candidate rerank call.
///
/// The full public path uses `validate_inputs`, which scans the complete
/// genotype and projection matrix for finite values.  The Max6 -> GRM-Max6
/// pipeline has already constructed those arrays from validated inputs and
/// calls many candidate windows, so repeating that O(n*m + n^2) validation in
/// every window would turn into a significant hot-path cost.
fn validate_candidate_rerank_shapes(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: &[f64],
    py: &[f64],
) -> Result<(), String> {
    let x_len = checked_matrix_len(n_samples, n_markers, "genotype")?;
    if x.len() != x_len {
        return Err(format!(
            "genotype length mismatch: got {}, expected {x_len}",
            x.len()
        ));
    }
    if n_samples == 0 || n_markers < 2 {
        return Err("candidate rerank requires samples > 0 and at least two markers".to_string());
    }
    let pc_len = checked_matrix_len(n_samples, n_samples, "P_C")?;
    if p_c.len() != pc_len {
        return Err(format!(
            "P_C length mismatch: got {}, expected {pc_len}",
            p_c.len()
        ));
    }
    if y.len() != n_samples {
        return Err(format!(
            "phenotype length mismatch: got {}, expected {n_samples}",
            y.len()
        ));
    }
    if py.len() != n_samples || py.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "candidate rerank projected phenotype length/finite mismatch: got {}, expected {}",
            py.len(),
            n_samples
        ));
    }
    Ok(())
}

fn pair_indices(n_markers: usize) -> (Vec<usize>, Vec<usize>) {
    let n_pairs = n_markers.saturating_mul(n_markers.saturating_sub(1)) / 2;
    let mut pair_i = Vec::with_capacity(n_pairs);
    let mut pair_j = Vec::with_capacity(n_pairs);
    for i in 0..n_markers {
        for j in (i + 1)..n_markers {
            pair_i.push(i);
            pair_j.push(j);
        }
    }
    (pair_i, pair_j)
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(super) fn dgemm_row_major(
    trans_left: bool,
    trans_right: bool,
    m: usize,
    n: usize,
    k: usize,
    left: &[f64],
    right: &[f64],
    out: &mut [f64],
) -> Result<(), String> {
    let left_rows = if trans_left { k } else { m };
    let left_cols = if trans_left { m } else { k };
    let right_rows = if trans_right { n } else { k };
    let right_cols = if trans_right { k } else { n };
    let left_len = checked_matrix_len(left_rows, left_cols, "left GEMM operand")?;
    let right_len = checked_matrix_len(right_rows, right_cols, "right GEMM operand")?;
    let out_len = checked_matrix_len(m, n, "GEMM output")?;
    if left.len() != left_len || right.len() != right_len || out.len() != out_len {
        return Err(format!(
            "GEMM shape mismatch: left={} (expected {left_len}), right={} (expected {right_len}), out={} (expected {out_len})",
            left.len(), right.len(), out.len()
        ));
    }
    let m_i = CblasInt::try_from(m).map_err(|_| format!("GEMM m dimension overflow: {m}"))?;
    let n_i = CblasInt::try_from(n).map_err(|_| format!("GEMM n dimension overflow: {n}"))?;
    let k_i = CblasInt::try_from(k).map_err(|_| format!("GEMM k dimension overflow: {k}"))?;
    let lda = CblasInt::try_from(if trans_left { m } else { k }.max(1))
        .map_err(|_| "GEMM lda dimension overflow".to_string())?;
    let ldb = CblasInt::try_from(if trans_right { k } else { n }.max(1))
        .map_err(|_| "GEMM ldb dimension overflow".to_string())?;
    let ldc =
        CblasInt::try_from(n.max(1)).map_err(|_| "GEMM ldc dimension overflow".to_string())?;
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_ROW_MAJOR,
            if trans_left {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            if trans_right {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            m_i,
            n_i,
            k_i,
            1.0,
            left.as_ptr(),
            lda,
            right.as_ptr(),
            ldb,
            0.0,
            out.as_mut_ptr(),
            ldc,
        );
    }
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub(super) fn dgemm_row_major(
    trans_left: bool,
    trans_right: bool,
    m: usize,
    n: usize,
    k: usize,
    left: &[f64],
    right: &[f64],
    out: &mut [f64],
) -> Result<(), String> {
    let left_len = checked_matrix_len(
        if trans_left { k } else { m },
        if trans_left { m } else { k },
        "left GEMM operand",
    )?;
    let right_len = checked_matrix_len(
        if trans_right { n } else { k },
        if trans_right { k } else { n },
        "right GEMM operand",
    )?;
    let out_len = checked_matrix_len(m, n, "GEMM output")?;
    if left.len() != left_len || right.len() != right_len || out.len() != out_len {
        return Err("GEMM shape mismatch".to_string());
    }
    for row in 0..m {
        for col in 0..n {
            let mut value = 0.0;
            for p in 0..k {
                let a = if trans_left {
                    left[p * m + row]
                } else {
                    left[row * k + p]
                };
                let b = if trans_right {
                    right[col * k + p]
                } else {
                    right[p * n + col]
                };
                value += a * b;
            }
            out[row * n + col] = value;
        }
    }
    Ok(())
}

fn matvec_row_major(matrix: &[f64], n: usize, vector: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; n];
    for row in 0..n {
        let mut value = 0.0;
        for col in 0..n {
            value += matrix[row * n + col] * vector[col];
        }
        result[row] = value;
    }
    result
}

const GRM_INT_MAX_BATCH_CANDIDATES: usize = 128;
const GRM_INT_MAX_BATCH_WORKSPACE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GrmIntScoreDiagnostics {
    pub(crate) singular_endpoint_design: usize,
    pub(crate) negligible_residual_interaction_information: usize,
    pub(crate) nonfinite: usize,
    pub(crate) batches: usize,
    pub(crate) max_workspace_bytes: usize,
    pub(crate) elapsed_seconds: f64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct GrmIntScoreBatch {
    /// One interaction chi-square(1) score per input pair, in input order.
    /// Singular, uninformative, and non-finite tests are represented by NaN.
    pub(crate) chisq: Vec<f64>,
    pub(crate) diagnostics: GrmIntScoreDiagnostics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GrmIntInvalidScore {
    SingularEndpointDesign,
    NegligibleInteractionInformation,
    Nonfinite,
}

#[inline]
fn grm_int_fwl_score_from_geometry(
    a00: f64,
    a11: f64,
    a01: f64,
    b0: f64,
    b1: f64,
    t0: f64,
    t1: f64,
    q: f64,
    c: f64,
    interaction_reference: f64,
    score_scale: f64,
) -> Result<f64, GrmIntInvalidScore> {
    let values = [
        a00,
        a11,
        a01,
        b0,
        b1,
        t0,
        t1,
        q,
        c,
        interaction_reference,
        score_scale,
    ];
    if values.iter().any(|value| !value.is_finite()) || score_scale <= 0.0 {
        return Err(GrmIntInvalidScore::Nonfinite);
    }
    let metric_scale = a00.max(a11);
    if !(a00 > 0.0 && a11 > 0.0 && metric_scale.is_finite() && metric_scale > 0.0) {
        return Err(GrmIntInvalidScore::SingularEndpointDesign);
    }
    let d0 = a00 / metric_scale;
    let d1 = a11 / metric_scale;
    let off = a01 / metric_scale;
    let relative_det = d0 * d1 - off * off;
    if !(relative_det.is_finite() && relative_det > DENOMINATOR_EPS * d0 * d1) {
        return Err(GrmIntInvalidScore::SingularEndpointDesign);
    }

    // Solve A beta_y = G'P_Cy and A beta_h = G'P_Ch using a normalized
    // 2x2 system. Scaling first avoids determinant overflow/underflow.
    let by0 = b0 / metric_scale;
    let by1 = b1 / metric_scale;
    let th0 = t0 / metric_scale;
    let th1 = t1 / metric_scale;
    let beta_y0 = (d1 * by0 - off * by1) / relative_det;
    let beta_y1 = (-off * by0 + d0 * by1) / relative_det;
    let beta_h0 = (d1 * th0 - off * th1) / relative_det;
    let beta_h1 = (-off * th0 + d0 * th1) / relative_det;
    let numerator = c - t0 * beta_y0 - t1 * beta_y1;
    let denominator = q - t0 * beta_h0 - t1 * beta_h1;
    let denominator_scale = q
        .abs()
        .max((t0 * beta_h0).abs())
        .max((t1 * beta_h1).abs())
        .max(interaction_reference.abs());
    if ![beta_y0, beta_y1, beta_h0, beta_h1, numerator, denominator]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(GrmIntInvalidScore::Nonfinite);
    }
    if !(denominator_scale.is_finite()
        && denominator_scale > 0.0
        && denominator > DENOMINATOR_EPS * denominator_scale)
    {
        return Err(GrmIntInvalidScore::NegligibleInteractionInformation);
    }
    let chisq = score_scale * numerator * numerator / denominator;
    if chisq.is_finite() && chisq >= 0.0 {
        Ok(chisq)
    } else {
        Err(GrmIntInvalidScore::Nonfinite)
    }
}

/// Compute conditional one-df interaction scores for an already selected set
/// of packed binary marker pairs. The endpoint main effects are projected out
/// under the same P_C metric as the phenotype; no GRM-Max6 gate channels are
/// evaluated here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grm_int_interaction_scores_packed(
    bits: &[u64],
    row_words: usize,
    n_samples: usize,
    n_markers: usize,
    candidate_pairs: &[(usize, usize)],
    p_c: Option<&[f64]>,
    no_grm_projector: Option<&NoGrmCovariateProjector>,
    projected_y: &[f64],
    score_scale: f64,
) -> Result<GrmIntScoreBatch, String> {
    let started = Instant::now();
    if n_samples == 0 || n_markers < 2 || row_words < n_samples.div_ceil(64) {
        return Err(
            "GRM-INT scorer requires samples, markers, and sufficient row words".to_string(),
        );
    }
    let packed_len = n_markers
        .checked_mul(row_words)
        .ok_or_else(|| "GRM-INT packed row length overflow".to_string())?;
    if bits.len() < packed_len {
        return Err(format!(
            "GRM-INT packed rows length={} smaller than required {packed_len}",
            bits.len()
        ));
    }
    if projected_y.len() != n_samples || projected_y.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "GRM-INT projected phenotype length/finite mismatch: got {}, expected {n_samples}",
            projected_y.len()
        ));
    }
    if !(score_scale.is_finite() && score_scale > 0.0) {
        return Err(format!(
            "GRM-INT score scale must be positive and finite, got {score_scale}"
        ));
    }
    match (p_c, no_grm_projector) {
        (Some(matrix), None) => {
            let expected = checked_matrix_len(n_samples, n_samples, "GRM-INT P_C")?;
            // The full dense projection is checked once before unit scanning
            // by the pipeline. Keep only this O(1) shape guard here; this
            // scorer is called once per unit, so rescanning P_C would add an
            // avoidable O(units * n_samples^2) validation pass.
            if matrix.len() != expected {
                return Err(format!(
                    "GRM-INT P_C shape mismatch: got {} values, expected {expected}",
                    matrix.len()
                ));
            }
            ensure_garfield_blas_backend()?;
        }
        (None, Some(projector)) if projector.n_samples() == n_samples => {}
        (None, Some(projector)) => {
            return Err(format!(
                "GRM-INT no-GRM projector sample count {} != n_samples {n_samples}",
                projector.n_samples()
            ));
        }
        (Some(_), Some(_)) => {
            return Err("GRM-INT received both dense P_C and no-GRM projector".to_string());
        }
        (None, None) => {
            return Err("GRM-INT requires dense P_C or a no-GRM projector".to_string());
        }
    }
    for (pair_index, &(left, right)) in candidate_pairs.iter().enumerate() {
        if left >= n_markers || right >= n_markers || left == right {
            return Err(format!(
                "GRM-INT pair {pair_index} has invalid endpoints ({left}, {right}) for {n_markers} markers"
            ));
        }
    }

    let mut result = GrmIntScoreBatch {
        chisq: vec![f64::NAN; candidate_pairs.len()],
        diagnostics: GrmIntScoreDiagnostics::default(),
    };
    if candidate_pairs.is_empty() {
        result.diagnostics.elapsed_seconds = started.elapsed().as_secs_f64();
        return Ok(result);
    }
    let bytes_per_candidate = n_samples
        .saturating_mul(6)
        .saturating_mul(std::mem::size_of::<f64>())
        .max(1);
    let budget_batch = (GRM_INT_MAX_BATCH_WORKSPACE_BYTES / bytes_per_candidate).max(1);
    let batch_size = budget_batch.min(GRM_INT_MAX_BATCH_CANDIDATES);
    let no_grm_projected_diagonal =
        no_grm_projector.map(NoGrmCovariateProjector::projected_diagonal);

    let mut batch_start = 0usize;
    while batch_start < candidate_pairs.len() {
        let batch_end = batch_start
            .saturating_add(batch_size)
            .min(candidate_pairs.len());
        let pairs = &candidate_pairs[batch_start..batch_end];
        let mut endpoint_rows = Vec::with_capacity(pairs.len().saturating_mul(2));
        for &(left, right) in pairs {
            endpoint_rows.push(left);
            endpoint_rows.push(right);
        }
        endpoint_rows.sort_unstable();
        endpoint_rows.dedup();
        let local_pairs = pairs
            .iter()
            .map(|&(left, right)| {
                let local_left = endpoint_rows
                    .binary_search(&left)
                    .map_err(|_| "GRM-INT endpoint mapping is incomplete".to_string())?;
                let local_right = endpoint_rows
                    .binary_search(&right)
                    .map_err(|_| "GRM-INT endpoint mapping is incomplete".to_string())?;
                Ok((local_left, local_right))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let n_endpoints = endpoint_rows.len();
        let n_pairs = pairs.len();
        let endpoint_len = checked_matrix_len(n_samples, n_endpoints, "GRM-INT endpoint matrix")?;
        let interaction_len = checked_matrix_len(n_samples, n_pairs, "GRM-INT interaction matrix")?;
        let mut x_endpoints = vec![0.0; endpoint_len];
        let mut endpoint_means = vec![0.0; n_endpoints];
        for (local_endpoint, &global_endpoint) in endpoint_rows.iter().enumerate() {
            let row_offset = global_endpoint * row_words;
            let mut sum = 0.0;
            for sample in 0..n_samples {
                let value = ((bits[row_offset + sample / 64] >> (sample % 64)) & 1) as f64;
                x_endpoints[sample * n_endpoints + local_endpoint] = value;
                sum += value;
            }
            endpoint_means[local_endpoint] = sum / n_samples as f64;
        }
        let mut h = vec![0.0; interaction_len];
        for (pair_column, &(left, right)) in local_pairs.iter().enumerate() {
            let mean_left = endpoint_means[left];
            let mean_right = endpoint_means[right];
            for sample in 0..n_samples {
                h[sample * n_pairs + pair_column] = (x_endpoints[sample * n_endpoints + left]
                    - mean_left)
                    * (x_endpoints[sample * n_endpoints + right] - mean_right);
            }
        }
        let mut z_endpoints = vec![0.0; endpoint_len];
        let mut z_h = vec![0.0; interaction_len];
        if let Some(matrix) = p_c {
            dgemm_row_major(
                false,
                false,
                n_samples,
                n_endpoints,
                n_samples,
                matrix,
                &x_endpoints,
                &mut z_endpoints,
            )?;
            dgemm_row_major(
                false, false, n_samples, n_pairs, n_samples, matrix, &h, &mut z_h,
            )?;
        } else {
            let projector = no_grm_projector.expect("validated no-GRM projector");
            for column in 0..n_endpoints {
                let values = (0..n_samples)
                    .map(|sample| x_endpoints[sample * n_endpoints + column])
                    .collect::<Vec<_>>();
                let projected = projector.project(&values)?;
                for (sample, value) in projected.into_iter().enumerate() {
                    z_endpoints[sample * n_endpoints + column] = value;
                }
            }
            for column in 0..n_pairs {
                let values = (0..n_samples)
                    .map(|sample| h[sample * n_pairs + column])
                    .collect::<Vec<_>>();
                let projected = projector.project(&values)?;
                for (sample, value) in projected.into_iter().enumerate() {
                    z_h[sample * n_pairs + column] = value;
                }
            }
        }
        let workspace_bytes = endpoint_len
            .saturating_add(interaction_len)
            .saturating_add(endpoint_len)
            .saturating_add(interaction_len)
            .saturating_mul(std::mem::size_of::<f64>());
        result.diagnostics.max_workspace_bytes =
            result.diagnostics.max_workspace_bytes.max(workspace_bytes);
        result.diagnostics.batches = result.diagnostics.batches.saturating_add(1);

        for (pair_column, &(left, right)) in local_pairs.iter().enumerate() {
            let mut a00 = 0.0;
            let mut a11 = 0.0;
            let mut a01_lr = 0.0;
            let mut a01_rl = 0.0;
            let mut b0 = 0.0;
            let mut b1 = 0.0;
            let mut t0 = 0.0;
            let mut t1 = 0.0;
            let mut q = 0.0;
            let mut c = 0.0;
            let mut interaction_reference = 0.0;
            for sample in 0..n_samples {
                let x0 = x_endpoints[sample * n_endpoints + left];
                let x1 = x_endpoints[sample * n_endpoints + right];
                let z0 = z_endpoints[sample * n_endpoints + left];
                let z1 = z_endpoints[sample * n_endpoints + right];
                let interaction = h[sample * n_pairs + pair_column];
                let p_interaction = z_h[sample * n_pairs + pair_column];
                a00 += x0 * z0;
                a11 += x1 * z1;
                a01_lr += x0 * z1;
                a01_rl += x1 * z0;
                b0 += x0 * projected_y[sample];
                b1 += x1 * projected_y[sample];
                t0 += x0 * p_interaction;
                t1 += x1 * p_interaction;
                q += interaction * p_interaction;
                c += interaction * projected_y[sample];
                let metric_diagonal = if let Some(matrix) = p_c {
                    matrix[sample * n_samples + sample].abs()
                } else {
                    no_grm_projected_diagonal
                        .as_ref()
                        .expect("validated no-GRM projector")[sample]
                };
                interaction_reference += interaction * interaction * metric_diagonal;
            }
            let a01 = 0.5 * (a01_lr + a01_rl);
            let score = grm_int_fwl_score_from_geometry(
                a00,
                a11,
                a01,
                b0,
                b1,
                t0,
                t1,
                q,
                c,
                interaction_reference,
                score_scale,
            );
            let output_index = batch_start + pair_column;
            match score {
                Ok(chisq) => result.chisq[output_index] = chisq,
                Err(GrmIntInvalidScore::SingularEndpointDesign) => {
                    result.diagnostics.singular_endpoint_design = result
                        .diagnostics
                        .singular_endpoint_design
                        .saturating_add(1);
                }
                Err(GrmIntInvalidScore::NegligibleInteractionInformation) => {
                    result
                        .diagnostics
                        .negligible_residual_interaction_information = result
                        .diagnostics
                        .negligible_residual_interaction_information
                        .saturating_add(1);
                }
                Err(GrmIntInvalidScore::Nonfinite) => {
                    result.diagnostics.nonfinite = result.diagnostics.nonfinite.saturating_add(1);
                }
            }
        }
        batch_start = batch_end;
    }
    result.diagnostics.elapsed_seconds = started.elapsed().as_secs_f64();
    Ok(result)
}

/// Compute the phenotype projection shared by candidate reranks in one
/// Max6 -> GRM-Max6 pipeline invocation.
#[inline]
pub(crate) fn project_phenotype_for_candidate_rerank(
    p_c: &[f64],
    n_samples: usize,
    y: &[f64],
) -> Vec<f64> {
    matvec_row_major(p_c, n_samples, y)
}

fn old_q_kernel(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    pair_i: &[usize],
    pair_j: &[usize],
    block_pairs: usize,
) -> Result<Vec<f64>, String> {
    let block_pairs = block_pairs.max(1);
    let mut q = vec![0.0; pair_i.len()];
    let mut start = 0usize;
    while start < pair_i.len() {
        let width = block_pairs.min(pair_i.len() - start);
        let mut w = vec![0.0; n_samples * width];
        for row in 0..n_samples {
            for offset in 0..width {
                let marker_i = pair_i[start + offset];
                let marker_j = pair_j[start + offset];
                w[row * width + offset] =
                    x[row * n_markers + marker_i] * x[row * n_markers + marker_j];
            }
        }
        let mut pcw = vec![0.0; n_samples * width];
        dgemm_row_major(false, false, n_samples, width, n_samples, p_c, &w, &mut pcw)?;
        for offset in 0..width {
            let mut value = 0.0;
            for row in 0..n_samples {
                value += w[row * width + offset] * pcw[row * width + offset];
            }
            q[start + offset] = value;
        }
        start += width;
    }
    Ok(q)
}

fn validate_binary_genotype_inputs(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
) -> Result<(), String> {
    validate_genotype_inputs(x, n_samples, n_markers)?;
    if x.iter().any(|&value| value != 0.0 && value != 1.0) {
        return Err("subset q requires genotype values in {0, 1}".to_string());
    }
    Ok(())
}

/// Recode each binary marker to its minor-support representation.
///
/// Columns stay in their original order. The subset q kernel performs the
/// support-sorted anchor traversal internally, so pair indices remain in the
/// caller's original SNP order and need no second permutation step.
fn min_support_encode(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
) -> Result<MinSupportEncoding, String> {
    validate_binary_genotype_inputs(x, n_samples, n_markers)?;
    let mut support = vec![0usize; n_markers];
    for row in 0..n_samples {
        for marker in 0..n_markers {
            support[marker] += (x[row * n_markers + marker] == 1.0) as usize;
        }
    }

    let mut encoded = x.to_vec();
    let mut flipped = vec![false; n_markers];
    for marker in 0..n_markers {
        if support[marker] > n_samples / 2 {
            flipped[marker] = true;
            for row in 0..n_samples {
                let index = row * n_markers + marker;
                encoded[index] = 1.0 - encoded[index];
            }
            support[marker] = n_samples - support[marker];
        }
    }
    Ok(MinSupportEncoding {
        x: encoded,
        flipped,
    })
}

/// Map a logic gate index computed after marker complementation back to the
/// original allele encoding. Gate order is AND00, AND01, AND10, AND11, XOR.
fn remap_logic_gate_index(gate: usize, flip_i: bool, flip_j: bool) -> usize {
    if gate == 4 {
        return 4;
    }
    let new_i = gate / 2;
    let new_j = gate % 2;
    let old_i = new_i ^ usize::from(flip_i);
    let old_j = new_j ^ usize::from(flip_j);
    old_i * 2 + old_j
}

fn best_logic_gate_index(values: &[f64; 5]) -> i64 {
    let mut best_gate = -1i64;
    let mut best_value = f64::NEG_INFINITY;
    for (gate, &value) in values.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_gate = gate as i64;
        }
    }
    best_gate
}

fn canonical_q_kernel_name(name: &str) -> Option<&'static str> {
    match name {
        "subset_min_support" | "subset_minor_support" => Some("subset_min_support"),
        "mincell_exact" => Some("mincell_exact"),
        _ => None,
    }
}

fn canonical_weighted_c_kernel_name(name: &str) -> Option<&'static str> {
    match name {
        "dense" | "dense_reference" => Some("dense_reference"),
        "binary_sparse" | "binary_sparse_C" => Some("binary_sparse_C"),
        "minor_support_weighted_gram" | "minor_support_C" => Some("minor_support_weighted_gram"),
        "sample_row_sparse" | "sample_row_sparse_C" => Some("sample_row_sparse"),
        _ => None,
    }
}

fn resolve_full_exact_kernel_names<'a>(
    requested_q_kernel: Option<&'a str>,
    requested_c_kernel: Option<&'a str>,
    has_spectral_args: bool,
) -> (&'a str, &'a str) {
    let q_kernel = requested_q_kernel.unwrap_or(if has_spectral_args {
        "spectral"
    } else {
        "mincell_exact"
    });
    let default_c_kernel = if matches!(
        q_kernel,
        "subset_min_support" | "subset_minor_support" | "mincell_exact"
    ) {
        "sample_row_sparse"
    } else {
        "dense_reference"
    };
    (q_kernel, requested_c_kernel.unwrap_or(default_c_kernel))
}

/// Compute exact `q=w'P_Cw` by restricting each interaction to one binary
/// anchor's support. The returned q values remain in the caller's pair order;
/// `support_sorted` changes only the internal anchor traversal order.
#[inline(never)]
fn subset_q_kernel(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    pair_i: &[usize],
    pair_j: &[usize],
    support_sorted: bool,
) -> Result<SubsetQResult, String> {
    validate_binary_genotype_inputs(x, n_samples, n_markers)?;
    let pc_len = checked_matrix_len(n_samples, n_samples, "P_C")?;
    if p_c.len() != pc_len || p_c.iter().any(|value| !value.is_finite()) {
        return Err("P_C values must be finite and have shape n_samples x n_samples".to_string());
    }
    if pair_i.len() != pair_j.len() {
        return Err("subset q pair index lengths differ".to_string());
    }
    let expected_pairs = n_markers
        .checked_mul(n_markers.saturating_sub(1))
        .map(|value| value / 2)
        .ok_or_else(|| "subset q pair count overflow".to_string())?;
    if pair_i.len() != expected_pairs {
        return Err(format!(
            "subset q requires all marker pairs: got {}, expected {}",
            pair_i.len(),
            expected_pairs
        ));
    }

    let pair_lookup_len = checked_matrix_len(n_markers, n_markers, "pair lookup")?;
    let mut timing = SubsetQTiming::default();
    let alloc_started = Instant::now();
    let mut pair_lookup = vec![usize::MAX; pair_lookup_len];
    timing.alloc_zero_seconds += alloc_started.elapsed().as_secs_f64();

    let support_started = Instant::now();
    for (pair, (&left, &right)) in pair_i.iter().zip(pair_j.iter()).enumerate() {
        if left >= n_markers || right >= n_markers || left == right {
            return Err("subset q pair index is outside marker range".to_string());
        }
        pair_lookup[left * n_markers + right] = pair;
        pair_lookup[right * n_markers + left] = pair;
    }

    let alloc_started = Instant::now();
    let mut support_counts = vec![0usize; n_markers];
    timing.alloc_zero_seconds += alloc_started.elapsed().as_secs_f64();
    for row in 0..n_samples {
        for marker in 0..n_markers {
            support_counts[marker] += (x[row * n_markers + marker] == 1.0) as usize;
        }
    }

    let alloc_started = Instant::now();
    let mut anchors = (0..n_markers).collect::<Vec<_>>();
    timing.alloc_zero_seconds += alloc_started.elapsed().as_secs_f64();
    if support_sorted {
        anchors.sort_by_key(|&marker| (support_counts[marker], marker));
    }
    timing.support_seconds += support_started.elapsed().as_secs_f64();

    let alloc_started = Instant::now();
    let mut q = vec![0.0; pair_i.len()];
    let max_support = support_counts.iter().copied().max().unwrap_or(0);
    let max_width = n_markers.saturating_sub(1);
    let mut p_sub_buf = vec![0.0; max_support * max_support];
    let mut x_sub_buf = vec![0.0; max_support * max_width];
    let mut y_sub_buf = vec![0.0; max_support * max_width];
    timing.alloc_zero_seconds += alloc_started.elapsed().as_secs_f64();
    timing.workspace_allocations = 3;

    let mut work_subset = 0.0;
    for anchor_position in 0..n_markers.saturating_sub(1) {
        let anchor = anchors[anchor_position];
        let tail = &anchors[(anchor_position + 1)..];
        let support_started = Instant::now();
        let support_rows = (0..n_samples)
            .filter(|&row| x[row * n_markers + anchor] == 1.0)
            .collect::<Vec<_>>();
        let support = support_rows.len();
        work_subset += (support as f64) * (support as f64) * (tail.len() as f64);
        timing.support_seconds += support_started.elapsed().as_secs_f64();
        if support == 0 || tail.is_empty() {
            continue;
        }

        let p_sub = &mut p_sub_buf[..support * support];
        let psub_started = Instant::now();
        for (sub_row, &row) in support_rows.iter().enumerate() {
            for (sub_col, &col) in support_rows.iter().enumerate() {
                p_sub[sub_row * support + sub_col] = p_c[row * n_samples + col];
            }
        }
        timing.psub_seconds += psub_started.elapsed().as_secs_f64();

        let width = tail.len();
        let x_sub = &mut x_sub_buf[..support * width];
        let xsub_started = Instant::now();
        for (sub_row, &row) in support_rows.iter().enumerate() {
            for (offset, &marker) in tail.iter().enumerate() {
                x_sub[sub_row * width + offset] = x[row * n_markers + marker];
            }
        }
        timing.xsub_seconds += xsub_started.elapsed().as_secs_f64();

        let y_sub = &mut y_sub_buf[..support * width];
        let gemm_started = Instant::now();
        dgemm_row_major(false, false, support, width, support, &p_sub, &x_sub, y_sub)?;
        timing.gemm_seconds += gemm_started.elapsed().as_secs_f64();

        let reduce_started = Instant::now();
        for (offset, &marker) in tail.iter().enumerate() {
            let mut value = 0.0;
            for sub_row in 0..support {
                if x_sub[sub_row * width + offset] == 1.0 {
                    value += y_sub[sub_row * width + offset];
                }
            }
            let pair = pair_lookup[anchor * n_markers + marker];
            if pair == usize::MAX {
                return Err("subset q pair lookup is incomplete".to_string());
            }
            q[pair] = value;
        }
        timing.reduce_seconds += reduce_started.elapsed().as_secs_f64();
    }

    let work_dense = (n_samples as f64) * (n_samples as f64) * (pair_i.len() as f64);
    let work_ratio = if work_dense > 0.0 {
        work_subset / work_dense
    } else {
        0.0
    };
    Ok(SubsetQResult {
        q,
        work_subset,
        work_dense,
        work_ratio,
        timing,
    })
}

/// Recover the original `q = w_11' P_C w_11` from the quadratic form of one
/// of the four binary cells. The cell codes are ordered as 00, 01, 10, 11.
fn reconstruct_original_q(
    q_cell: f64,
    cell: u8,
    a1: f64,
    a2: f64,
    a12: f64,
    u0: f64,
    u1: f64,
) -> f64 {
    match cell {
        0 => q_cell - a1 - a2 - 2.0 * a12 + 2.0 * u0 + 2.0 * u1,
        1 => q_cell - a2 + 2.0 * u1,
        2 => q_cell - a1 + 2.0 * u0,
        3 => q_cell,
        _ => f64::NAN,
    }
}

/// Compute `q_c = h_c' P_C h_c` for the sparsest four-cell support of every
/// marker pair. This is deliberately a correctness-first reference kernel:
/// it materializes marker bitsets, chooses a cell with bit operations, and
/// enumerates that cell's set bits for the exact quadratic form. It does not
/// use the result as the final Max6 denominator; callers reconstruct the
/// original `q_11` and continue through the existing score assembly.
#[inline(never)]
fn mincell_q_kernel_for_pairs(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    pair_i: &[usize],
    pair_j: &[usize],
) -> Result<MinCellQResult, String> {
    validate_binary_genotype_inputs(x, n_samples, n_markers)?;
    let words = n_samples.div_ceil(64);
    let bitset_started = Instant::now();
    let mut bitsets = vec![0u64; n_markers * words];
    let mut supports = vec![0usize; n_markers];
    for row in 0..n_samples {
        let word = row / 64;
        let bit = 1u64 << (row % 64);
        let row_offset = row * n_markers;
        for marker in 0..n_markers {
            if x[row_offset + marker] == 1.0 {
                bitsets[marker * words + word] |= bit;
                supports[marker] += 1;
            }
        }
    }
    let bitset_seconds = bitset_started.elapsed().as_secs_f64();

    mincell_q_from_bitsets(
        &bitsets,
        words,
        n_samples,
        n_markers,
        p_c,
        pair_i,
        pair_j,
        &supports,
        bitset_seconds,
    )
}

/// Exact min-cell q kernel for marker-major packed binary rows.
///
/// The packed rows are already the representation produced by the screen
/// (`marker * row_words + sample_word`). This path performs no dense f64
/// genotype materialization and no second dense minor-support encoding.
#[inline(never)]
fn mincell_q_kernel_for_packed_pairs(
    bits: &[u64],
    row_words: usize,
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    pair_i: &[usize],
    pair_j: &[usize],
    supports: &[usize],
    bitset_seconds: f64,
) -> Result<MinCellQResult, String> {
    let words = n_samples.div_ceil(64);
    if row_words < words {
        return Err(format!(
            "packed mincell q row_words={row_words} smaller than required {words}"
        ));
    }
    let required = n_markers
        .checked_mul(row_words)
        .ok_or_else(|| "packed mincell q input length overflow".to_string())?;
    if bits.len() < required {
        return Err(format!(
            "packed mincell q input length={} smaller than required {required}",
            bits.len()
        ));
    }
    mincell_q_from_bitsets(
        bits,
        row_words,
        n_samples,
        n_markers,
        p_c,
        pair_i,
        pair_j,
        supports,
        bitset_seconds,
    )
}

#[inline(never)]
fn mincell_q_from_bitsets(
    bitsets: &[u64],
    bitset_stride: usize,
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    pair_i: &[usize],
    pair_j: &[usize],
    supports: &[usize],
    bitset_seconds: f64,
) -> Result<MinCellQResult, String> {
    let validation_started = Instant::now();
    let words = n_samples.div_ceil(64);
    if bitset_stride < words {
        return Err(format!(
            "mincell q bitset stride={bitset_stride} smaller than required {words}"
        ));
    }
    let required = n_markers
        .checked_mul(bitset_stride)
        .ok_or_else(|| "mincell q bitset length overflow".to_string())?;
    if bitsets.len() < required {
        return Err(format!(
            "mincell q bitset length={} smaller than required {required}",
            bitsets.len()
        ));
    }
    if supports.len() != n_markers || supports.iter().any(|&support| support > n_samples) {
        return Err("mincell q support counts have an invalid shape or value".to_string());
    }
    let pc_len = checked_matrix_len(n_samples, n_samples, "P_C")?;
    if p_c.len() != pc_len || p_c.iter().any(|value| !value.is_finite()) {
        return Err(
            "mincell q requires finite P_C values with shape n_samples x n_samples".to_string(),
        );
    }
    if pair_i.len() != pair_j.len() {
        return Err("mincell q pair index lengths differ".to_string());
    }
    for (&left, &right) in pair_i.iter().zip(pair_j.iter()) {
        if left >= n_markers || right >= n_markers || left >= right {
            return Err("mincell q pair indices must be canonical and in range".to_string());
        }
    }
    let validation_seconds = validation_started.elapsed().as_secs_f64();

    let last_bits = n_samples % 64;
    let last_mask = if last_bits == 0 {
        u64::MAX
    } else {
        (1u64 << last_bits) - 1
    };
    let mut sorted_supports = supports.to_vec();
    sorted_supports.sort_unstable();
    let work_subset = sorted_supports
        .iter()
        .enumerate()
        .map(|(position, &support)| {
            (support as f64) * (support as f64) * (n_markers - position - 1) as f64
        })
        .sum::<f64>();
    let work_dense = (n_samples as f64) * (n_samples as f64) * (pair_i.len() as f64);

    let mut q_min = vec![0.0; pair_i.len()];
    let mut cell = vec![0u8; pair_i.len()];
    let mut min_support = vec![0usize; pair_i.len()];
    let mut work_mincell = 0.0;
    let mut indices = Vec::<usize>::new();
    let mut cell_seconds = 0.0;
    let mut enumerate_seconds = 0.0;
    let mut quadratic_seconds = 0.0;

    for pair in 0..pair_i.len() {
        let cell_started = Instant::now();
        let left = pair_i[pair];
        let right = pair_j[pair];
        let left_support = supports[left];
        let right_support = supports[right];
        let mut n11 = 0usize;
        for word in 0..words {
            let valid = if word + 1 == words {
                last_mask
            } else {
                u64::MAX
            };
            n11 += (bitsets[left * bitset_stride + word]
                & bitsets[right * bitset_stride + word]
                & valid)
                .count_ones() as usize;
        }
        let n00 = n_samples
            .checked_add(n11)
            .and_then(|value| value.checked_sub(left_support))
            .and_then(|value| value.checked_sub(right_support))
            .ok_or_else(|| "mincell q produced an invalid 00 cell count".to_string())?;
        let counts = [n00, right_support - n11, left_support - n11, n11];
        let mut chosen = 0usize;
        for candidate in 1..4 {
            if counts[candidate] < counts[chosen] {
                chosen = candidate;
            }
        }
        cell[pair] = chosen as u8;
        min_support[pair] = counts[chosen];
        work_mincell += (counts[chosen] as f64) * (counts[chosen] as f64);
        cell_seconds += cell_started.elapsed().as_secs_f64();

        let enumerate_started = Instant::now();
        indices.clear();
        for word in 0..words {
            let valid = if word + 1 == words {
                last_mask
            } else {
                u64::MAX
            };
            let left_bits = bitsets[left * bitset_stride + word];
            let right_bits = bitsets[right * bitset_stride + word];
            let mut selected = match chosen {
                0 => !(left_bits | right_bits),
                1 => !left_bits & right_bits,
                2 => left_bits & !right_bits,
                3 => left_bits & right_bits,
                _ => unreachable!("cell selection is limited to four cells"),
            } & valid;
            while selected != 0 {
                let offset = selected.trailing_zeros() as usize;
                indices.push(word * 64 + offset);
                selected &= selected - 1;
            }
        }
        debug_assert_eq!(indices.len(), counts[chosen]);
        enumerate_seconds += enumerate_started.elapsed().as_secs_f64();

        let quadratic_started = Instant::now();
        let mut value = 0.0;
        // This is the same quadratic form as diag + 2 * upper triangle for a
        // symmetric P_C, but the row-major order is closer to the existing
        // subset-q GEMM/reduction accumulation and is useful for diagnostics.
        for &row in &indices {
            let mut row_value = 0.0;
            for &col in &indices {
                row_value += p_c[row * n_samples + col];
            }
            value += row_value;
        }
        q_min[pair] = value;
        quadratic_seconds += quadratic_started.elapsed().as_secs_f64();
    }

    let mincell_subset_ratio = if work_subset > 0.0 {
        work_mincell / work_subset
    } else {
        0.0
    };
    Ok(MinCellQResult {
        q_min,
        cell,
        support: min_support,
        work: MinCellQWork {
            work_mincell,
            work_subset,
            work_dense,
            mincell_subset_ratio,
        },
        timing: MinCellQTiming {
            validation_seconds,
            bitset_seconds,
            cell_seconds,
            enumerate_seconds,
            quadratic_seconds,
            reconstruct_seconds: 0.0,
        },
    })
}

#[inline(never)]
fn mincell_q_kernel(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    pair_i: &[usize],
    pair_j: &[usize],
) -> Result<MinCellQResult, String> {
    let expected_pairs = n_markers
        .checked_mul(n_markers.saturating_sub(1))
        .map(|value| value / 2)
        .ok_or_else(|| "mincell q pair count overflow".to_string())?;
    if pair_i.len() != expected_pairs || pair_j.len() != expected_pairs {
        return Err(format!(
            "mincell q requires all marker pairs: got {} and {}, expected {expected_pairs}",
            pair_i.len(),
            pair_j.len()
        ));
    }
    mincell_q_kernel_for_pairs(x, n_samples, n_markers, p_c, pair_i, pair_j)
}

#[inline]
fn canonical_pair_id(n_markers: usize, left: usize, right: usize) -> usize {
    debug_assert!(left < right && right < n_markers);
    left * n_markers - left * (left + 1) / 2 + (right - left - 1)
}

fn candidate_pcx_sparse_support(
    encoded_unique: &[u64],
    row_words: usize,
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
) -> Result<Vec<f64>, String> {
    let words = n_samples.div_ceil(64);
    if row_words < words {
        return Err(format!(
            "sparse candidate P_CX row_words={row_words} smaller than required {words}"
        ));
    }
    let encoded_len = checked_matrix_len(n_markers, row_words, "sparse candidate genotype")?;
    if encoded_unique.len() < encoded_len {
        return Err(format!(
            "sparse candidate genotype length={} smaller than required {encoded_len}",
            encoded_unique.len()
        ));
    }
    let pc_len = checked_matrix_len(n_samples, n_samples, "sparse candidate P_C")?;
    if p_c.len() != pc_len {
        return Err(format!(
            "sparse candidate P_C length={} does not match n_samples^2={pc_len}",
            p_c.len()
        ));
    }

    let mut offsets = Vec::with_capacity(n_markers + 1);
    let mut support_samples = Vec::new();
    offsets.push(0usize);
    let last_bits = n_samples % 64;
    let last_mask = if last_bits == 0 {
        u64::MAX
    } else {
        (1u64 << last_bits) - 1
    };
    for marker in 0..n_markers {
        for word in 0..words {
            let valid = if word + 1 == words {
                last_mask
            } else {
                u64::MAX
            };
            let mut selected = encoded_unique[marker * row_words + word] & valid;
            while selected != 0 {
                support_samples.push(word * 64 + selected.trailing_zeros() as usize);
                selected &= selected - 1;
            }
        }
        offsets.push(support_samples.len());
    }

    let output_len = checked_matrix_len(n_samples, n_markers, "sparse candidate P_CX")?;
    let mut z = vec![0.0; output_len];
    for row in 0..n_samples {
        let p_row = &p_c[row * n_samples..(row + 1) * n_samples];
        let z_row = &mut z[row * n_markers..(row + 1) * n_markers];
        for marker in 0..n_markers {
            let mut value = 0.0;
            for &sample in &support_samples[offsets[marker]..offsets[marker + 1]] {
                value += p_row[sample];
            }
            z_row[marker] = value;
        }
    }
    Ok(z)
}

fn candidate_pcx_sparse_support_from_env() -> Result<bool, String> {
    let raw = std::env::var("JX_GARFIELD_EXPERIMENTAL_PCX_KERNEL").unwrap_or_default();
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "dgemm" | "dense" => Ok(false),
        "sparse_support" | "sparse" => Ok(true),
        other => Err(format!(
            "JX_GARFIELD_EXPERIMENTAL_PCX_KERNEL must be dgemm or sparse_support; got {other}"
        )),
    }
}

fn canonicalize_candidate_pairs(
    candidate_first: &[i64],
    candidate_second: &[i64],
    n_markers: usize,
) -> Result<CandidatePairSet, String> {
    if candidate_first.len() != candidate_second.len() {
        return Err(format!(
            "candidate pair index lengths differ: {} vs {}",
            candidate_first.len(),
            candidate_second.len()
        ));
    }
    let mut pair_i = Vec::new();
    let mut pair_j = Vec::new();
    let mut input_to_unique = Vec::with_capacity(candidate_first.len());
    let mut seen = HashMap::<(usize, usize), usize>::new();
    let mut input_self_count = 0usize;
    let mut input_duplicate_count = 0usize;

    for (&raw_first, &raw_second) in candidate_first.iter().zip(candidate_second.iter()) {
        if raw_first < 0
            || raw_second < 0
            || raw_first as usize >= n_markers
            || raw_second as usize >= n_markers
        {
            return Err(format!(
                "candidate pair ({raw_first}, {raw_second}) is outside marker range 0..{n_markers}"
            ));
        }
        let first = raw_first as usize;
        let second = raw_second as usize;
        if first == second {
            input_self_count += 1;
            input_to_unique.push(-1);
            continue;
        }
        let (left, right) = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        let unique_index = if let Some(&existing) = seen.get(&(left, right)) {
            input_duplicate_count += 1;
            existing
        } else {
            let index = pair_i.len();
            pair_i.push(left);
            pair_j.push(right);
            seen.insert((left, right), index);
            index
        };
        input_to_unique.push(unique_index as i64);
    }

    Ok(CandidatePairSet {
        pair_i,
        pair_j,
        input_to_unique,
        input_self_count,
        input_duplicate_count,
    })
}

fn prepare_candidate_endpoint_mapping(
    candidate_first: &[i64],
    candidate_second: &[i64],
    n_markers: usize,
) -> Result<(CandidatePairSet, Vec<usize>, Vec<usize>, Vec<usize>), String> {
    let pair_set = canonicalize_candidate_pairs(candidate_first, candidate_second, n_markers)?;
    if pair_set.pair_i.is_empty() {
        return Err("candidate rerank requires at least one non-self pair".to_string());
    }
    let mut unique_markers = pair_set
        .pair_i
        .iter()
        .chain(pair_set.pair_j.iter())
        .copied()
        .collect::<Vec<_>>();
    unique_markers.sort_unstable();
    unique_markers.dedup();
    let mut local_pair_i = Vec::with_capacity(pair_set.pair_i.len());
    let mut local_pair_j = Vec::with_capacity(pair_set.pair_i.len());
    for (&global_i, &global_j) in pair_set.pair_i.iter().zip(pair_set.pair_j.iter()) {
        let local_i = unique_markers
            .binary_search(&global_i)
            .map_err(|_| "candidate endpoint mapping is incomplete".to_string())?;
        let local_j = unique_markers
            .binary_search(&global_j)
            .map_err(|_| "candidate endpoint mapping is incomplete".to_string())?;
        if local_i >= local_j {
            return Err("candidate local pair order is not canonical".to_string());
        }
        local_pair_i.push(local_i);
        local_pair_j.push(local_j);
    }
    Ok((pair_set, unique_markers, local_pair_i, local_pair_j))
}

/// Minor-support encode packed marker rows and decode only the unique
/// candidate endpoints into the row-major f64 layout required by BLAS.
///
/// The full selected window is never materialized as dense f64.  The packed
/// encoded rows are also returned so mincell q can consume the same bits
/// without rebuilding them from the dense endpoint matrix.
fn packed_minor_support_decode_unique(
    bits: &[u64],
    row_words: usize,
    n_samples: usize,
    n_markers: usize,
    unique_markers: &[usize],
) -> Result<(Vec<f64>, Vec<u64>, Vec<bool>, Vec<usize>, f64), String> {
    let words = n_samples.div_ceil(64);
    if row_words < words {
        return Err(format!(
            "packed candidate rerank row_words={row_words} smaller than required {words}"
        ));
    }
    let required = n_markers
        .checked_mul(row_words)
        .ok_or_else(|| "packed candidate rerank input length overflow".to_string())?;
    if bits.len() < required {
        return Err(format!(
            "packed candidate rerank input length={} smaller than required {required}",
            bits.len()
        ));
    }
    if unique_markers.iter().any(|&marker| marker >= n_markers) {
        return Err("candidate endpoint marker is outside packed input range".to_string());
    }

    let last_bits = n_samples % 64;
    let last_mask = if last_bits == 0 {
        u64::MAX
    } else {
        (1u64 << last_bits) - 1
    };
    let bitset_started = Instant::now();
    let mut supports_unique = vec![0usize; unique_markers.len()];
    for (local, &marker) in unique_markers.iter().enumerate() {
        let offset = marker * row_words;
        for word in 0..words {
            let valid = if word + 1 == words {
                last_mask
            } else {
                u64::MAX
            };
            supports_unique[local] += (bits[offset + word] & valid).count_ones() as usize;
        }
    }
    let bitset_seconds = bitset_started.elapsed().as_secs_f64();

    let mut flipped = vec![false; n_markers];
    for (local, &marker) in unique_markers.iter().enumerate() {
        flipped[marker] = supports_unique[local] > n_samples / 2;
    }

    let n_unique = unique_markers.len();
    let mut encoded_unique = vec![0u64; n_unique * row_words];
    let mut encoded_supports = vec![0usize; n_unique];
    for (local, &marker) in unique_markers.iter().enumerate() {
        let source_offset = marker * row_words;
        let target_offset = local * row_words;
        let flip_mask = if flipped[marker] { u64::MAX } else { 0 };
        for word in 0..words {
            let valid = if word + 1 == words {
                last_mask
            } else {
                u64::MAX
            };
            let encoded_word = (bits[source_offset + word] ^ flip_mask) & valid;
            encoded_unique[target_offset + word] = encoded_word;
            encoded_supports[local] += encoded_word.count_ones() as usize;
        }
    }

    let mut x_unique = vec![0.0; n_samples * n_unique];
    for row in 0..n_samples {
        let word = row / 64;
        let bit = 1u64 << (row % 64);
        for local in 0..n_unique {
            x_unique[row * n_unique + local] =
                ((encoded_unique[local * row_words + word] & bit) != 0) as u8 as f64;
        }
    }
    Ok((
        x_unique,
        encoded_unique,
        flipped,
        encoded_supports,
        bitset_seconds,
    ))
}

#[derive(Clone, Copy, Debug)]
struct CandidateAu {
    a12: f64,
    u0: f64,
    u1: f64,
}

struct CandidateAuBatch {
    values: Vec<CandidateAu>,
    workspace_bytes: usize,
    validation_seconds: f64,
    allocation_seconds: f64,
    materialization_seconds: f64,
    gemm_seconds: f64,
    lookup_seconds: f64,
    copy_seconds: f64,
    total_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateAuKernel {
    Reference,
    Support,
    SupportFusedPy,
    WindowBlas,
}

fn candidate_au_kernel_from_env_value(name: Option<&str>) -> Result<CandidateAuKernel, String> {
    // Production GARFIELD receives complete packed 0/1 rows: upstream encoding
    // applies the existing mode-imputation rule before packing. Support and
    // Reference therefore consume identical genotype semantics. The fused
    // kernel also accumulates interaction-Py while enumerating exact supports.
    match name {
        None => Ok(CandidateAuKernel::SupportFusedPy),
        Some("support") => Ok(CandidateAuKernel::Support),
        Some("reference") => Ok(CandidateAuKernel::Reference),
        Some("support_fused_py") => Ok(CandidateAuKernel::SupportFusedPy),
        Some("window_blas") => Ok(CandidateAuKernel::WindowBlas),
        Some(other) => Err(format!(
            "unknown experimental candidate A/U kernel: {other}"
        )),
    }
}

fn candidate_au_kernel_from_env() -> Result<CandidateAuKernel, String> {
    match std::env::var("JX_GARFIELD_EXPERIMENTAL_CANDIDATE_AU_KERNEL") {
        Err(std::env::VarError::NotPresent) => candidate_au_kernel_from_env_value(None),
        Err(error) => Err(format!("candidate A/U kernel environment: {error}")),
        Ok(name) => candidate_au_kernel_from_env_value(Some(name.as_str())),
    }
}

fn validate_candidate_bitset_dense_semantics(
    bits: &[u64],
    row_words: usize,
    x: &[f64],
    n_samples: usize,
    n_unique: usize,
) -> Result<(), String> {
    if bits.len() != checked_matrix_len(n_unique, row_words, "candidate bitsets")?
        || x.len() != checked_matrix_len(n_samples, n_unique, "candidate dense genotypes")?
        || row_words < n_samples.div_ceil(64)
    {
        return Err("candidate bitset/dense-X shape mismatch".to_string());
    }
    for row in 0..n_samples {
        for marker in 0..n_unique {
            let dense = x[row * n_unique + marker];
            let packed = (bits[marker * row_words + row / 64] >> (row % 64)) & 1;
            if (dense != 0.0 && dense != 1.0) || dense as u64 != packed {
                return Err(format!(
                    "candidate bitset/dense-X binary or missing mismatch at sample {row}, marker {marker}"
                ));
            }
        }
    }
    Ok(())
}

fn candidate_au_support(
    bits: &[u64],
    row_words: usize,
    n_samples: usize,
    n_unique: usize,
    x: &[f64],
    z: &[f64],
    i: usize,
    j: usize,
) -> Result<CandidateAu, String> {
    debug_assert_eq!(x.len(), n_samples * n_unique);
    debug_assert_eq!(z.len(), n_samples * n_unique);
    if i >= n_unique || j >= n_unique || bits.len() < n_unique * row_words {
        return Err("candidate support pair/bitset outside endpoint universe".to_string());
    }
    let mut out = CandidateAu {
        a12: 0.0,
        u0: 0.0,
        u1: 0.0,
    };
    for word in 0..n_samples.div_ceil(64) {
        let mut mask = bits[i * row_words + word];
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let row = word * 64 + bit;
            if row >= n_samples {
                continue;
            }
            let offset = row * n_unique;
            let zj = z[offset + j];
            out.a12 += zj;
            if (bits[j * row_words + word] & (1u64 << bit)) != 0 {
                out.u0 += z[offset + i];
                out.u1 += zj;
            }
        }
    }
    Ok(out)
}

fn candidate_au_support_fused_py(
    bits: &[u64],
    row_words: usize,
    n_samples: usize,
    n_unique: usize,
    z: &[f64],
    py: &[f64],
    i: usize,
    j: usize,
) -> Result<(CandidateAu, f64), String> {
    if i >= n_unique
        || j >= n_unique
        || i == j
        || bits.len() != checked_matrix_len(n_unique, row_words, "candidate bitsets")?
        || z.len() != checked_matrix_len(n_samples, n_unique, "candidate Z")?
        || py.len() != n_samples
        || row_words < n_samples.div_ceil(64)
    {
        return Err("candidate fused A/U/Py input shape or pair mismatch".to_string());
    }

    let mut au = CandidateAu {
        a12: 0.0,
        u0: 0.0,
        u1: 0.0,
    };
    let mut interaction_py = 0.0;
    for word in 0..n_samples.div_ceil(64) {
        let mut mask_i = bits[i * row_words + word];
        let mask_j = bits[j * row_words + word];
        while mask_i != 0 {
            let bit = mask_i.trailing_zeros() as usize;
            mask_i &= mask_i - 1;
            let row = word * 64 + bit;
            if row >= n_samples {
                continue;
            }
            let z_offset = row * n_unique;
            let z_j = z[z_offset + j];
            au.a12 += z_j;
            if (mask_j & (1u64 << bit)) != 0 {
                au.u0 += z[z_offset + i];
                au.u1 += z_j;
                interaction_py += py[row];
            }
        }
    }
    Ok((au, interaction_py))
}

fn candidate_au_window_blas(
    x: &[f64],
    z: &[f64],
    n_samples: usize,
    n_unique: usize,
    pair_i: &[usize],
    pair_j: &[usize],
) -> Result<CandidateAuBatch, String> {
    let total_started = Instant::now();
    let dense_len = checked_matrix_len(n_samples, n_unique, "candidate dense-X")?;
    let gram_len = checked_matrix_len(n_unique, n_unique, "candidate A/H")?;
    if x.len() != dense_len || z.len() != dense_len || pair_i.len() != pair_j.len() {
        return Err("candidate A/H input shape mismatch".to_string());
    }
    let workspace_bytes = candidate_au_workspace_bytes(n_samples, n_unique, pair_i.len(), true)?;
    const MAX_EXPERIMENTAL_WORKSPACE_BYTES: usize = 512 * 1024 * 1024;
    if workspace_bytes > MAX_EXPERIMENTAL_WORKSPACE_BYTES {
        return Err(format!("experimental candidate A/H workspace {workspace_bytes} bytes exceeds 512 MiB preflight limit"));
    }
    let allocation_started = Instant::now();
    let mut xz = vec![0.0; dense_len];
    let mut a = vec![0.0; gram_len];
    let mut h = vec![0.0; gram_len];
    let mut values = Vec::with_capacity(pair_i.len());
    let allocation_seconds = allocation_started.elapsed().as_secs_f64();
    let materialization_started = Instant::now();
    for row in 0..dense_len {
        xz[row] = x[row] * z[row];
    }
    let materialization_seconds = materialization_started.elapsed().as_secs_f64();
    let gemm_started = Instant::now();
    dgemm_row_major(true, false, n_unique, n_unique, n_samples, x, z, &mut a)?;
    dgemm_row_major(true, false, n_unique, n_unique, n_samples, x, &xz, &mut h)?;
    let gemm_seconds = gemm_started.elapsed().as_secs_f64();
    let lookup_started = Instant::now();
    for (&i, &j) in pair_i.iter().zip(pair_j.iter()) {
        if i >= n_unique || j >= n_unique {
            return Err("candidate A/H pair outside endpoint universe".to_string());
        }
        values.push(CandidateAu {
            a12: a[i * n_unique + j],
            u0: h[j * n_unique + i],
            u1: h[i * n_unique + j],
        });
    }
    let lookup_seconds = lookup_started.elapsed().as_secs_f64();
    Ok(CandidateAuBatch {
        values,
        workspace_bytes,
        validation_seconds: 0.0,
        allocation_seconds,
        materialization_seconds,
        gemm_seconds,
        lookup_seconds,
        copy_seconds: 0.0,
        total_seconds: total_started.elapsed().as_secs_f64(),
    })
}

fn candidate_au_microbench_once(
    kernel: CandidateAuKernel,
    bits: &[u64],
    row_words: usize,
    x: &[f64],
    z: &[f64],
    py: &[f64],
    n_samples: usize,
    n_unique: usize,
    pair_i: &[usize],
    pair_j: &[usize],
) -> Result<CandidateAuBatch, String> {
    let total_started = Instant::now();
    let allocation_started = Instant::now();
    let mut a12 = vec![0.0; pair_i.len()];
    let mut u0 = vec![0.0; pair_i.len()];
    let mut u1 = vec![0.0; pair_i.len()];
    let output_allocation_seconds = allocation_started.elapsed().as_secs_f64();
    match kernel {
        CandidateAuKernel::Reference => {
            let lookup_started = Instant::now();
            for (candidate, (&i, &j)) in pair_i.iter().zip(pair_j.iter()).enumerate() {
                if i >= n_unique || j >= n_unique {
                    return Err("candidate reference pair outside endpoint universe".to_string());
                }
                let mut value = CandidateAu {
                    a12: 0.0,
                    u0: 0.0,
                    u1: 0.0,
                };
                for row in 0..n_samples {
                    let x_i = x[row * n_unique + i];
                    let x_j = x[row * n_unique + j];
                    let w = x_i * x_j;
                    value.a12 += x_i * z[row * n_unique + j];
                    value.u0 += w * z[row * n_unique + i];
                    value.u1 += w * z[row * n_unique + j];
                }
                a12[candidate] = value.a12;
                u0[candidate] = value.u0;
                u1[candidate] = value.u1;
            }
            let lookup_seconds = lookup_started.elapsed().as_secs_f64();
            let total_seconds = total_started.elapsed().as_secs_f64();
            let values = (0..pair_i.len())
                .map(|candidate| CandidateAu {
                    a12: a12[candidate],
                    u0: u0[candidate],
                    u1: u1[candidate],
                })
                .collect();
            let workspace_bytes =
                candidate_au_workspace_bytes(n_samples, n_unique, pair_i.len(), false)?
                    .checked_add(
                        pair_i
                            .len()
                            .checked_mul(std::mem::size_of::<CandidateAu>())
                            .ok_or("candidate A/U parity workspace overflow")?,
                    )
                    .ok_or("candidate A/U parity workspace overflow")?;
            Ok(CandidateAuBatch {
                values,
                workspace_bytes,
                validation_seconds: 0.0,
                allocation_seconds: output_allocation_seconds,
                materialization_seconds: 0.0,
                gemm_seconds: 0.0,
                lookup_seconds,
                copy_seconds: 0.0,
                total_seconds,
            })
        }
        CandidateAuKernel::Support => {
            let validation_started = Instant::now();
            validate_candidate_bitset_dense_semantics(bits, row_words, x, n_samples, n_unique)?;
            let validation_seconds = validation_started.elapsed().as_secs_f64();
            let lookup_started = Instant::now();
            for (candidate, (&i, &j)) in pair_i.iter().zip(pair_j.iter()).enumerate() {
                let value = candidate_au_support(bits, row_words, n_samples, n_unique, x, z, i, j)?;
                a12[candidate] = value.a12;
                u0[candidate] = value.u0;
                u1[candidate] = value.u1;
            }
            let lookup_seconds = lookup_started.elapsed().as_secs_f64();
            let total_seconds = total_started.elapsed().as_secs_f64();
            let values = (0..pair_i.len())
                .map(|candidate| CandidateAu {
                    a12: a12[candidate],
                    u0: u0[candidate],
                    u1: u1[candidate],
                })
                .collect();
            let workspace_bytes =
                candidate_au_workspace_bytes(n_samples, n_unique, pair_i.len(), false)?
                    .checked_add(
                        pair_i
                            .len()
                            .checked_mul(std::mem::size_of::<CandidateAu>())
                            .ok_or("candidate A/U parity workspace overflow")?,
                    )
                    .ok_or("candidate A/U parity workspace overflow")?;
            Ok(CandidateAuBatch {
                values,
                workspace_bytes,
                validation_seconds,
                allocation_seconds: output_allocation_seconds,
                materialization_seconds: 0.0,
                gemm_seconds: 0.0,
                lookup_seconds,
                copy_seconds: 0.0,
                total_seconds,
            })
        }
        CandidateAuKernel::SupportFusedPy => {
            let validation_started = Instant::now();
            validate_candidate_bitset_dense_semantics(bits, row_words, x, n_samples, n_unique)?;
            let validation_seconds = validation_started.elapsed().as_secs_f64();
            let lookup_started = Instant::now();
            let mut interaction_py_checksum = 0.0;
            for (candidate, (&i, &j)) in pair_i.iter().zip(pair_j.iter()).enumerate() {
                let (value, py_value) = candidate_au_support_fused_py(
                    bits, row_words, n_samples, n_unique, z, py, i, j,
                )?;
                a12[candidate] = value.a12;
                u0[candidate] = value.u0;
                u1[candidate] = value.u1;
                interaction_py_checksum += py_value;
            }
            std::hint::black_box(interaction_py_checksum);
            let lookup_seconds = lookup_started.elapsed().as_secs_f64();
            let total_seconds = total_started.elapsed().as_secs_f64();
            let values = (0..pair_i.len())
                .map(|candidate| CandidateAu {
                    a12: a12[candidate],
                    u0: u0[candidate],
                    u1: u1[candidate],
                })
                .collect();
            let workspace_bytes =
                candidate_au_workspace_bytes(n_samples, n_unique, pair_i.len(), false)?
                    .checked_add(
                        pair_i
                            .len()
                            .checked_mul(std::mem::size_of::<CandidateAu>())
                            .ok_or("candidate fused A/U/Py workspace overflow")?,
                    )
                    .ok_or("candidate fused A/U/Py workspace overflow")?;
            Ok(CandidateAuBatch {
                values,
                workspace_bytes,
                validation_seconds,
                allocation_seconds: output_allocation_seconds,
                materialization_seconds: 0.0,
                gemm_seconds: 0.0,
                lookup_seconds,
                copy_seconds: 0.0,
                total_seconds,
            })
        }
        CandidateAuKernel::WindowBlas => {
            let mut batch = candidate_au_window_blas(x, z, n_samples, n_unique, pair_i, pair_j)?;
            let copy_started = Instant::now();
            for (idx, values) in batch.values.iter().enumerate() {
                a12[idx] = values.a12;
                u0[idx] = values.u0;
                u1[idx] = values.u1;
            }
            batch.copy_seconds = copy_started.elapsed().as_secs_f64();
            batch.allocation_seconds += output_allocation_seconds;
            batch.total_seconds = total_started.elapsed().as_secs_f64();
            Ok(batch)
        }
    }
}

fn candidate_au_microbench_if_requested(
    kernel: CandidateAuKernel,
    bits: &[u64],
    row_words: usize,
    x: &[f64],
    z: &[f64],
    py: &[f64],
    n_samples: usize,
    n_unique: usize,
    pair_i: &[usize],
    pair_j: &[usize],
    baseline: &[CandidateAu],
) -> Result<(), String> {
    let repeats = match std::env::var("JX_GARFIELD_EXPERIMENTAL_CANDIDATE_AU_BENCH_REPEATS") {
        Err(std::env::VarError::NotPresent) => return Ok(()),
        Err(error) => return Err(format!("candidate A/U benchmark environment: {error}")),
        Ok(value) => value
            .parse::<usize>()
            .map_err(|error| format!("invalid candidate A/U benchmark repeats: {error}"))?,
    };
    if !(5..=100).contains(&repeats) {
        return Err("candidate A/U microbenchmark requires 5..=100 measured repeats".to_string());
    }
    if baseline.len() != pair_i.len() {
        return Err("candidate A/U microbenchmark baseline length mismatch".to_string());
    }

    let compare = |values: &[CandidateAu]| -> (usize, f64) {
        let mut bitwise_diffs = 0usize;
        let mut max_abs: f64 = 0.0;
        for (got, want) in values.iter().zip(baseline.iter()) {
            for (g, w) in [(got.a12, want.a12), (got.u0, want.u0), (got.u1, want.u1)] {
                bitwise_diffs += (g.to_bits() != w.to_bits()) as usize;
                max_abs = max_abs.max((g - w).abs());
            }
        }
        (bitwise_diffs, max_abs)
    };
    let warmup = candidate_au_microbench_once(
        kernel, bits, row_words, x, z, py, n_samples, n_unique, pair_i, pair_j,
    )?;
    let (warmup_bitwise, warmup_error) = compare(&warmup.values);
    let mut measured = Vec::with_capacity(repeats);
    let mut samples = Vec::with_capacity(repeats);
    let mut max_bitwise_diffs = warmup_bitwise;
    let mut max_abs_error = warmup_error;
    for _ in 0..repeats {
        let batch = candidate_au_microbench_once(
            kernel, bits, row_words, x, z, py, n_samples, n_unique, pair_i, pair_j,
        )?;
        let (bitwise_diffs, abs_error) = compare(&batch.values);
        max_bitwise_diffs = max_bitwise_diffs.max(bitwise_diffs);
        max_abs_error = max_abs_error.max(abs_error);
        measured.push(batch.total_seconds);
        samples.push(batch);
    }
    let median = |mut values: Vec<f64>| -> f64 {
        values.sort_by(f64::total_cmp);
        let middle = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[middle - 1] + values[middle]) / 2.0
        } else {
            values[middle]
        }
    };
    let median_seconds = median(measured.clone());
    let mut deviations = measured
        .iter()
        .map(|&value| (value - median_seconds).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    let mad_seconds = median(deviations);
    let allocation_median = median(
        samples
            .iter()
            .map(|sample| sample.allocation_seconds)
            .collect(),
    );
    let validation_median = median(
        samples
            .iter()
            .map(|sample| sample.validation_seconds)
            .collect(),
    );
    let materialization_median = median(
        samples
            .iter()
            .map(|sample| sample.materialization_seconds)
            .collect(),
    );
    let gemm_median = median(samples.iter().map(|sample| sample.gemm_seconds).collect());
    let lookup_median = median(samples.iter().map(|sample| sample.lookup_seconds).collect());
    let copy_median = median(samples.iter().map(|sample| sample.copy_seconds).collect());
    let workspace_bytes = samples
        .iter()
        .map(|sample| sample.workspace_bytes)
        .max()
        .unwrap_or(0);
    eprintln!(
        "CANDIDATE_AU_MICROBENCH kernel={kernel:?} n={n_samples} m_u={n_unique} candidates={} warmup=1 repeats={repeats} median_seconds={median_seconds:.9} mad_seconds={mad_seconds:.9} samples_seconds={:?} median_validation={validation_median:.9} median_allocation={allocation_median:.9} median_xz={materialization_median:.9} median_AH_gemm={gemm_median:.9} median_candidate_lookup={lookup_median:.9} median_copy={copy_median:.9} max_workspace_bytes={workspace_bytes} max_repeat_bitwise_diffs={max_bitwise_diffs} max_repeat_abs_error={max_abs_error:.17e}",
        pair_i.len(), measured,
    );
    Ok(())
}

fn candidate_au_workspace_bytes(
    n_samples: usize,
    n_unique: usize,
    n_candidates: usize,
    include_window_blas: bool,
) -> Result<usize, String> {
    let output_bytes = n_candidates
        .checked_mul(3 * std::mem::size_of::<f64>())
        .ok_or("candidate A/U output workspace overflow")?;
    if !include_window_blas {
        return Ok(output_bytes);
    }
    let dense_len = checked_matrix_len(n_samples, n_unique, "candidate dense-X")?;
    let gram_len = checked_matrix_len(n_unique, n_unique, "candidate A/H")?;
    let matrix_bytes = dense_len
        .checked_add(
            gram_len
                .checked_mul(2)
                .ok_or("candidate A/H workspace overflow")?,
        )
        .and_then(|len| len.checked_mul(std::mem::size_of::<f64>()))
        .ok_or("candidate A/H workspace byte overflow")?;
    let staging_bytes = n_candidates
        .checked_mul(std::mem::size_of::<CandidateAu>())
        .ok_or("candidate A/U staging workspace overflow")?;
    matrix_bytes
        .checked_add(output_bytes)
        .and_then(|bytes| bytes.checked_add(staging_bytes))
        .ok_or_else(|| "candidate A/U total workspace byte overflow".to_string())
}

fn candidate_au_support_fused_py_workspace_bytes(
    n_samples: usize,
    n_unique: usize,
    n_candidates: usize,
) -> Result<usize, String> {
    candidate_au_workspace_bytes(n_samples, n_unique, n_candidates, false)?
        .checked_add(
            n_candidates
                .checked_mul(std::mem::size_of::<f64>())
                .ok_or("candidate fused interaction-Py workspace overflow")?,
        )
        .ok_or_else(|| "candidate fused interaction-Py workspace overflow".to_string())
}

#[inline(never)]
fn candidate_rerank_from_encoded_unique(
    x_unique: Vec<f64>,
    encoded_unique: &[u64],
    row_words: usize,
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    py: &[f64],
    pair_set: CandidatePairSet,
    unique_markers: Vec<usize>,
    local_pair_i: Vec<usize>,
    local_pair_j: Vec<usize>,
    flipped: Vec<bool>,
    mut mincell: MinCellQResult,
    top_k: usize,
    dosage_channel: DosageChannel,
    kernel_started: Instant,
    endpoint_prepare_seconds: f64,
    au_kernel: CandidateAuKernel,
    pcx_sparse_support: bool,
) -> Result<CandidateRerankResult, String> {
    let n_unique = unique_markers.len();
    let py_sum = py.iter().sum::<f64>();

    let pcx_started = Instant::now();
    let z_unique = if pcx_sparse_support {
        candidate_pcx_sparse_support(encoded_unique, row_words, n_samples, n_unique, p_c)?
    } else {
        let mut z = vec![0.0; n_samples * n_unique];
        dgemm_row_major(
            false, false, n_samples, n_unique, n_samples, p_c, &x_unique, &mut z,
        )?;
        z
    };
    let pcx_u_seconds = pcx_started.elapsed().as_secs_f64();

    let geometry_started = Instant::now();
    let marker_moments_started = Instant::now();
    let mut marker_a = vec![0.0; n_unique];
    let mut marker_b = vec![0.0; n_unique];
    let mut marker_mean = vec![0.0; n_unique];
    for marker in 0..n_unique {
        let mut a_value = 0.0;
        let mut b_value = 0.0;
        let mut mean_value = 0.0;
        for row in 0..n_samples {
            let x_value = x_unique[row * n_unique + marker];
            a_value += x_value * z_unique[row * n_unique + marker];
            b_value += x_value * py[row];
            mean_value += x_value;
        }
        marker_a[marker] = a_value;
        marker_b[marker] = b_value;
        marker_mean[marker] = mean_value / n_samples as f64;
    }
    let geometry_marker_moments_seconds = marker_moments_started.elapsed().as_secs_f64();

    let pairwise_geometry_started = Instant::now();
    let au_output_allocation_started = Instant::now();
    let mut candidate_a12 = vec![0.0; local_pair_i.len()];
    let mut candidate_u0 = vec![0.0; local_pair_i.len()];
    let mut candidate_u1 = vec![0.0; local_pair_i.len()];
    let mut au_allocation_seconds = au_output_allocation_started.elapsed().as_secs_f64();
    let mut fused_interaction_py = None;
    if au_kernel == CandidateAuKernel::SupportFusedPy {
        let fused_py_allocation_started = Instant::now();
        fused_interaction_py = Some(vec![0.0; local_pair_i.len()]);
        au_allocation_seconds += fused_py_allocation_started.elapsed().as_secs_f64();
    }
    let mut au_validation_seconds = 0.0;
    let mut au_materialization_seconds = 0.0;
    let mut au_gemm_seconds = 0.0;
    let mut au_fused_py_seconds = 0.0;
    let au_candidate_lookup_seconds;
    let mut au_copy_seconds = 0.0;
    let mut au_workspace_bytes =
        candidate_au_workspace_bytes(n_samples, n_unique, local_pair_i.len(), false)?;
    if au_kernel == CandidateAuKernel::SupportFusedPy {
        au_workspace_bytes =
            candidate_au_support_fused_py_workspace_bytes(n_samples, n_unique, local_pair_i.len())?;
    }
    match au_kernel {
        CandidateAuKernel::Reference => {
            let lookup_started = Instant::now();
            for candidate in 0..local_pair_i.len() {
                let local_i = local_pair_i[candidate];
                let local_j = local_pair_j[candidate];
                for row in 0..n_samples {
                    let x_i = x_unique[row * n_unique + local_i];
                    let x_j = x_unique[row * n_unique + local_j];
                    let w = x_i * x_j;
                    candidate_a12[candidate] += x_i * z_unique[row * n_unique + local_j];
                    candidate_u0[candidate] += w * z_unique[row * n_unique + local_i];
                    candidate_u1[candidate] += w * z_unique[row * n_unique + local_j];
                }
            }
            au_candidate_lookup_seconds = lookup_started.elapsed().as_secs_f64();
        }
        CandidateAuKernel::Support => {
            let validation_started = Instant::now();
            validate_candidate_bitset_dense_semantics(
                encoded_unique,
                row_words,
                &x_unique,
                n_samples,
                n_unique,
            )?;
            au_validation_seconds = validation_started.elapsed().as_secs_f64();
            let lookup_started = Instant::now();
            for candidate in 0..local_pair_i.len() {
                let values = candidate_au_support(
                    encoded_unique,
                    row_words,
                    n_samples,
                    n_unique,
                    &x_unique,
                    &z_unique,
                    local_pair_i[candidate],
                    local_pair_j[candidate],
                )?;
                candidate_a12[candidate] = values.a12;
                candidate_u0[candidate] = values.u0;
                candidate_u1[candidate] = values.u1;
            }
            au_candidate_lookup_seconds = lookup_started.elapsed().as_secs_f64();
        }
        CandidateAuKernel::SupportFusedPy => {
            let validation_started = Instant::now();
            validate_candidate_bitset_dense_semantics(
                encoded_unique,
                row_words,
                &x_unique,
                n_samples,
                n_unique,
            )?;
            au_validation_seconds = validation_started.elapsed().as_secs_f64();
            let fused_started = Instant::now();
            let py_values = fused_interaction_py
                .as_mut()
                .expect("fused support kernel allocates interaction-Py output");
            for candidate in 0..local_pair_i.len() {
                let (values, py_value) = candidate_au_support_fused_py(
                    encoded_unique,
                    row_words,
                    n_samples,
                    n_unique,
                    &z_unique,
                    py,
                    local_pair_i[candidate],
                    local_pair_j[candidate],
                )?;
                candidate_a12[candidate] = values.a12;
                candidate_u0[candidate] = values.u0;
                candidate_u1[candidate] = values.u1;
                py_values[candidate] = py_value;
            }
            au_fused_py_seconds = fused_started.elapsed().as_secs_f64();
            au_candidate_lookup_seconds = 0.0;
        }
        CandidateAuKernel::WindowBlas => {
            let batch = candidate_au_window_blas(
                &x_unique,
                &z_unique,
                n_samples,
                n_unique,
                &local_pair_i,
                &local_pair_j,
            )?;
            au_workspace_bytes = batch.workspace_bytes;
            au_allocation_seconds += batch.allocation_seconds;
            au_materialization_seconds = batch.materialization_seconds;
            au_gemm_seconds = batch.gemm_seconds;
            au_candidate_lookup_seconds = batch.lookup_seconds;
            au_copy_seconds = batch.copy_seconds;
            let copy_started = Instant::now();
            for (candidate, values) in batch.values.iter().enumerate() {
                candidate_a12[candidate] = values.a12;
                candidate_u0[candidate] = values.u0;
                candidate_u1[candidate] = values.u1;
            }
            au_copy_seconds += copy_started.elapsed().as_secs_f64();
        }
    }
    if std::env::var_os("JX_GARFIELD_EXPERIMENTAL_CANDIDATE_AU_BENCH_REPEATS").is_some() {
        let au_baseline = (0..local_pair_i.len())
            .map(|candidate| CandidateAu {
                a12: candidate_a12[candidate],
                u0: candidate_u0[candidate],
                u1: candidate_u1[candidate],
            })
            .collect::<Vec<_>>();
        candidate_au_microbench_if_requested(
            au_kernel,
            encoded_unique,
            row_words,
            &x_unique,
            &z_unique,
            py,
            n_samples,
            n_unique,
            &local_pair_i,
            &local_pair_j,
            &au_baseline,
        )?;
    }
    let geometry_pairwise_seconds = pairwise_geometry_started.elapsed().as_secs_f64();
    let audit_au = au_kernel != CandidateAuKernel::Reference
        && std::env::var_os("JX_GARFIELD_EXPERIMENTAL_CANDIDATE_AU_AUDIT").is_some();
    let reference_au = if audit_au {
        let mut values = Vec::with_capacity(local_pair_i.len());
        for candidate in 0..local_pair_i.len() {
            let local_i = local_pair_i[candidate];
            let local_j = local_pair_j[candidate];
            let mut value = CandidateAu {
                a12: 0.0,
                u0: 0.0,
                u1: 0.0,
            };
            for row in 0..n_samples {
                let x_i = x_unique[row * n_unique + local_i];
                let x_j = x_unique[row * n_unique + local_j];
                let w = x_i * x_j;
                value.a12 += x_i * z_unique[row * n_unique + local_j];
                value.u0 += w * z_unique[row * n_unique + local_i];
                value.u1 += w * z_unique[row * n_unique + local_j];
            }
            values.push(value);
        }
        Some(values)
    } else {
        None
    };

    // Candidate-only equivalent of sample-row sparse weighted C. The lookup
    // table is only over unique endpoints and contains no non-candidate edge.
    let (geometry_lookup_seconds, interaction_py, geometry_interaction_py_seconds) =
        if let Some(interaction_py) = fused_interaction_py {
            (0.0, interaction_py, 0.0)
        } else {
            let lookup_started = Instant::now();
            let lookup_len = checked_matrix_len(n_unique, n_unique, "candidate C lookup")?;
            let mut candidate_lookup = vec![usize::MAX; lookup_len];
            for (candidate, (&local_i, &local_j)) in
                local_pair_i.iter().zip(local_pair_j.iter()).enumerate()
            {
                candidate_lookup[local_i * n_unique + local_j] = candidate;
                candidate_lookup[local_j * n_unique + local_i] = candidate;
            }
            let geometry_lookup_seconds = lookup_started.elapsed().as_secs_f64();

            let interaction_py_started = Instant::now();
            let mut interaction_py = vec![0.0; local_pair_i.len()];
            let mut active = Vec::with_capacity(n_unique);
            for row in 0..n_samples {
                active.clear();
                for marker in 0..n_unique {
                    if x_unique[row * n_unique + marker] == 1.0 {
                        active.push(marker);
                    }
                }
                for left_position in 0..active.len() {
                    let left = active[left_position];
                    for &right in &active[(left_position + 1)..] {
                        let candidate = candidate_lookup[left * n_unique + right];
                        if candidate != usize::MAX {
                            interaction_py[candidate] += py[row];
                        }
                    }
                }
            }
            let geometry_interaction_py_seconds = interaction_py_started.elapsed().as_secs_f64();
            (
                geometry_lookup_seconds,
                interaction_py,
                geometry_interaction_py_seconds,
            )
        };
    let candidate_geometry_seconds = geometry_started.elapsed().as_secs_f64();

    let q_reconstruct_started = Instant::now();
    let mut q = vec![0.0; local_pair_i.len()];
    for candidate in 0..local_pair_i.len() {
        let local_i = local_pair_i[candidate];
        let local_j = local_pair_j[candidate];
        q[candidate] = reconstruct_original_q(
            mincell.q_min[candidate],
            mincell.cell[candidate],
            marker_a[local_i],
            marker_a[local_j],
            candidate_a12[candidate],
            candidate_u0[candidate],
            candidate_u1[candidate],
        );
    }
    mincell.timing.reconstruct_seconds = q_reconstruct_started.elapsed().as_secs_f64();
    let q_timing = mincell.timing;
    let q_seconds = q_timing.sum();

    let score_started = Instant::now();
    let mut logic = vec![[f64::NEG_INFINITY; 5]; local_pair_i.len()];
    let mut dosage = vec![f64::NEG_INFINITY; local_pair_i.len()];
    let mut max6 = vec![f64::NEG_INFINITY; local_pair_i.len()];
    let mut best_logic_gate = vec![-1i64; local_pair_i.len()];
    let mut best_max6_channel = Vec::with_capacity(local_pair_i.len());
    let mut audit_reference_max6 = audit_au.then(|| vec![f64::NEG_INFINITY; local_pair_i.len()]);
    let mut audit_max_a12: f64 = 0.0;
    let mut audit_max_u0: f64 = 0.0;
    let mut audit_max_u1: f64 = 0.0;
    let mut audit_max_q: f64 = 0.0;
    let mut audit_max_score: f64 = 0.0;
    let mut audit_bitwise_a12 = 0usize;
    let mut audit_bitwise_u0 = 0usize;
    let mut audit_bitwise_u1 = 0usize;
    let mut audit_best_channel_diff = 0usize;
    let mut audit_finite_channel_diff = 0usize;
    for candidate in 0..local_pair_i.len() {
        let local_i = local_pair_i[candidate];
        let local_j = local_pair_j[candidate];
        let (encoded_logic, encoded_dosage, encoded_max6) = candidate_score_values(
            marker_b[local_i],
            marker_b[local_j],
            marker_a[local_i],
            marker_a[local_j],
            candidate_a12[candidate],
            candidate_u0[candidate],
            candidate_u1[candidate],
            q[candidate],
            interaction_py[candidate],
            marker_mean[local_i],
            marker_mean[local_j],
            py_sum,
            dosage_channel,
        );
        let global_i = pair_set.pair_i[candidate];
        let global_j = pair_set.pair_j[candidate];
        let flip_i = flipped[global_i];
        let flip_j = flipped[global_j];
        let mut mapped_logic = [f64::NEG_INFINITY; 5];
        for (gate, &value) in encoded_logic.iter().enumerate() {
            mapped_logic[remap_logic_gate_index(gate, flip_i, flip_j)] = value;
        }
        logic[candidate] = mapped_logic;
        dosage[candidate] = encoded_dosage;
        max6[candidate] = encoded_max6;
        best_logic_gate[candidate] = best_logic_gate_index(&mapped_logic);
        best_max6_channel.push(best_max6_channel_name(&mapped_logic, encoded_dosage));
        if let (Some(reference_au), Some(reference_max6)) =
            (reference_au.as_ref(), audit_reference_max6.as_mut())
        {
            let reference_geometry = reference_au[candidate];
            audit_max_a12 =
                audit_max_a12.max((candidate_a12[candidate] - reference_geometry.a12).abs());
            audit_max_u0 =
                audit_max_u0.max((candidate_u0[candidate] - reference_geometry.u0).abs());
            audit_max_u1 =
                audit_max_u1.max((candidate_u1[candidate] - reference_geometry.u1).abs());
            audit_bitwise_a12 +=
                (candidate_a12[candidate].to_bits() != reference_geometry.a12.to_bits()) as usize;
            audit_bitwise_u0 +=
                (candidate_u0[candidate].to_bits() != reference_geometry.u0.to_bits()) as usize;
            audit_bitwise_u1 +=
                (candidate_u1[candidate].to_bits() != reference_geometry.u1.to_bits()) as usize;
            let q_reference = reconstruct_original_q(
                mincell.q_min[candidate],
                mincell.cell[candidate],
                marker_a[local_i],
                marker_a[local_j],
                reference_geometry.a12,
                reference_geometry.u0,
                reference_geometry.u1,
            );
            audit_max_q = audit_max_q.max((q[candidate] - q_reference).abs());
            let (reference_logic, reference_dosage, reference_score) = candidate_score_values(
                marker_b[local_i],
                marker_b[local_j],
                marker_a[local_i],
                marker_a[local_j],
                reference_geometry.a12,
                reference_geometry.u0,
                reference_geometry.u1,
                q_reference,
                interaction_py[candidate],
                marker_mean[local_i],
                marker_mean[local_j],
                py_sum,
                dosage_channel,
            );
            let mut mapped_reference_logic = [f64::NEG_INFINITY; 5];
            for (gate, &value) in reference_logic.iter().enumerate() {
                mapped_reference_logic[remap_logic_gate_index(gate, flip_i, flip_j)] = value;
            }
            let mut channel_error: f64 = 0.0;
            for gate in 0..5 {
                let current_finite = mapped_logic[gate].is_finite();
                let reference_finite = mapped_reference_logic[gate].is_finite();
                if current_finite != reference_finite {
                    audit_finite_channel_diff += 1;
                    channel_error = f64::INFINITY;
                } else if current_finite {
                    channel_error = channel_error
                        .max((mapped_logic[gate] - mapped_reference_logic[gate]).abs());
                }
            }
            if encoded_dosage.is_finite() != reference_dosage.is_finite() {
                audit_finite_channel_diff += 1;
                channel_error = f64::INFINITY;
            } else if encoded_dosage.is_finite() {
                channel_error = channel_error.max((encoded_dosage - reference_dosage).abs());
            }
            audit_max_score = audit_max_score.max(channel_error);
            if best_max6_channel_name(&mapped_reference_logic, reference_dosage)
                != best_max6_channel_name(&mapped_logic, encoded_dosage)
            {
                audit_best_channel_diff += 1;
            }
            reference_max6[candidate] = reference_score;
        }
    }
    let score_seconds = score_started.elapsed().as_secs_f64();

    let global_pair_id = pair_set
        .pair_i
        .iter()
        .zip(pair_set.pair_j.iter())
        .map(|(&left, &right)| canonical_pair_id(n_markers, left, right))
        .collect::<Vec<_>>();
    let topk_started = Instant::now();
    let mut order = (0..max6.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        max6[right]
            .total_cmp(&max6[left])
            .then_with(|| global_pair_id[left].cmp(&global_pair_id[right]))
    });
    let mut candidate_rank = vec![0usize; order.len()];
    for (rank, &candidate) in order.iter().enumerate() {
        candidate_rank[candidate] = rank + 1;
    }
    let top_order = order[..top_k.min(order.len())].to_vec();
    if let Some(reference_scores) = audit_reference_max6.as_ref() {
        let mut reference_order = (0..reference_scores.len()).collect::<Vec<_>>();
        reference_order.sort_by(|&left, &right| {
            reference_scores[right]
                .total_cmp(&reference_scores[left])
                .then_with(|| global_pair_id[left].cmp(&global_pair_id[right]))
        });
        let compared_top_k = top_k.min(order.len());
        let overlap = order[..compared_top_k]
            .iter()
            .filter(|candidate| reference_order[..compared_top_k].contains(candidate))
            .count();
        let mut reference_rank = vec![0usize; order.len()];
        for (rank, &candidate) in reference_order.iter().enumerate() {
            reference_rank[candidate] = rank + 1;
        }
        let max_rank_shift = candidate_rank
            .iter()
            .enumerate()
            .map(|(candidate, &rank)| rank.abs_diff(reference_rank[candidate]))
            .max()
            .unwrap_or(0);
        let cutoff_index = compared_top_k.saturating_sub(1);
        let reference_cutoff = reference_order
            .get(cutoff_index)
            .map(|&candidate| reference_scores[candidate]);
        let changed_membership = order[..compared_top_k]
            .iter()
            .filter(|candidate| !reference_order[..compared_top_k].contains(candidate))
            .chain(
                reference_order[..compared_top_k]
                    .iter()
                    .filter(|candidate| !order[..compared_top_k].contains(candidate)),
            )
            .count();
        let substantive_cutoff_changes = reference_cutoff
            .map(|cutoff| {
                order[..compared_top_k]
                    .iter()
                    .filter(|candidate| !reference_order[..compared_top_k].contains(candidate))
                    .chain(
                        reference_order[..compared_top_k]
                            .iter()
                            .filter(|candidate| !order[..compared_top_k].contains(candidate)),
                    )
                    .filter(|&&candidate| {
                        (reference_scores[candidate] - cutoff).abs() > 2.0 * audit_max_score
                    })
                    .count()
            })
            .unwrap_or(0);
        let mut top_union = order[..compared_top_k]
            .iter()
            .chain(reference_order[..compared_top_k].iter())
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        top_union.sort_unstable();
        let mut top_union_inversions = 0usize;
        let mut substantive_top_union_inversions = 0usize;
        for left_pos in 0..top_union.len() {
            for right_pos in (left_pos + 1)..top_union.len() {
                let left = top_union[left_pos];
                let right = top_union[right_pos];
                let current_cmp = candidate_rank[left].cmp(&candidate_rank[right]);
                let reference_cmp = reference_rank[left].cmp(&reference_rank[right]);
                if current_cmp != reference_cmp {
                    top_union_inversions += 1;
                    if (reference_scores[left] - reference_scores[right]).abs()
                        > 2.0 * audit_max_score
                    {
                        substantive_top_union_inversions += 1;
                    }
                }
            }
        }
        eprintln!(
            "CANDIDATE_AU_AUDIT kernel={au_kernel:?} n={n_samples} m_u={n_unique} K={} topK={compared_top_k} bitwise_diff_a12={audit_bitwise_a12} bitwise_diff_u0={audit_bitwise_u0} bitwise_diff_u1={audit_bitwise_u1} max_abs_a12={audit_max_a12:.17e} max_abs_u0={audit_max_u0:.17e} max_abs_u1={audit_max_u1:.17e} max_abs_q={audit_max_q:.17e} max_abs_score={audit_max_score:.17e} finite_channel_diff={audit_finite_channel_diff} best_channel_diff={audit_best_channel_diff} topK_overlap={overlap} max_rank_shift={max_rank_shift} cutoff_membership_changes={changed_membership} substantive_cutoff_changes={substantive_cutoff_changes} top_union_inversions={top_union_inversions} substantive_top_union_inversions={substantive_top_union_inversions}",
            order.len(),
        );
    }
    let topk_seconds = topk_started.elapsed().as_secs_f64();

    Ok(CandidateRerankResult {
        pair_i: pair_set.pair_i,
        pair_j: pair_set.pair_j,
        input_to_unique: pair_set.input_to_unique,
        input_self_count: pair_set.input_self_count,
        input_duplicate_count: pair_set.input_duplicate_count,
        global_pair_id,
        unique_markers,
        z: z_unique,
        q,
        mincell_q: mincell.q_min,
        mincell_cell: mincell.cell,
        mincell_support: mincell.support,
        logic,
        dosage,
        max6,
        best_logic_gate,
        best_max6_channel,
        candidate_rank,
        top_order,
        flipped,
        timing: CandidateRerankTiming {
            endpoint_prepare_seconds,
            pcx_u_seconds,
            pcx_sparse_support,
            py_seconds: 0.0,
            candidate_geometry_seconds,
            geometry_marker_moments_seconds,
            geometry_pairwise_seconds,
            geometry_au_validation_seconds: au_validation_seconds,
            geometry_au_allocation_seconds: au_allocation_seconds,
            geometry_au_materialization_seconds: au_materialization_seconds,
            geometry_au_gemm_seconds: au_gemm_seconds,
            geometry_au_candidate_lookup_seconds: au_candidate_lookup_seconds,
            geometry_au_fused_py_seconds: au_fused_py_seconds,
            geometry_au_copy_seconds: au_copy_seconds,
            geometry_au_workspace_bytes: au_workspace_bytes,
            geometry_lookup_seconds,
            geometry_interaction_py_seconds,
            q_seconds,
            q_validation_seconds: q_timing.validation_seconds,
            q_bitset_seconds: q_timing.bitset_seconds,
            q_cell_seconds: q_timing.cell_seconds,
            q_enumerate_seconds: q_timing.enumerate_seconds,
            q_quadratic_seconds: q_timing.quadratic_seconds,
            q_reconstruct_seconds: q_timing.reconstruct_seconds,
            score_seconds,
            topk_seconds,
            kernel_seconds: kernel_started.elapsed().as_secs_f64(),
            ..CandidateRerankTiming::default()
        },
    })
}

#[inline]
fn centered_product_score(
    b0: f64,
    b1: f64,
    a00: f64,
    a11: f64,
    a01: f64,
    u0: f64,
    u1: f64,
    q: f64,
    c: f64,
    mean0: f64,
    mean1: f64,
    py_sum: f64,
) -> f64 {
    // w_c = (x_i - mean0) * (x_j - mean1), with w=x_i*x_j.
    // P_C contains the intercept projection, so P_C*1=0.  The explicit
    // py_sum term is retained in the numerator to preserve the same result
    // when the supplied matrix is only numerically intercept-orthogonal.
    let numerator = c - mean1 * b0 - mean0 * b1 + mean0 * mean1 * py_sum;
    let denominator = q - 2.0 * mean1 * u0 - 2.0 * mean0 * u1
        + mean1 * mean1 * a00
        + mean0 * mean0 * a11
        + 2.0 * mean0 * mean1 * a01;
    score_ratio(numerator, denominator)
}

#[inline]
fn dosage_channel_score(
    channel: DosageChannel,
    b0: f64,
    b1: f64,
    a00: f64,
    a11: f64,
    a01: f64,
    u0: f64,
    u1: f64,
    q: f64,
    c: f64,
    mean0: f64,
    mean1: f64,
    py_sum: f64,
) -> f64 {
    match channel {
        DosageChannel::FwlProduct => {
            // Rank is a relative property of G'P_C G, not a property of the
            // arbitrary units of P_C. Positive diagonal entries are required
            // even when two negative diagonals would give a positive det.
            let metric_scale = a00.max(a11);
            if !(a00.is_finite() && a11.is_finite() && a01.is_finite() && a00 > 0.0 && a11 > 0.0) {
                return f64::NEG_INFINITY;
            }
            let d0 = a00 / metric_scale;
            let d1 = a11 / metric_scale;
            let off = a01 / metric_scale;
            let relative_det = d0 * d1 - off * off;
            if !(relative_det.is_finite() && relative_det > DENOMINATOR_EPS * d0 * d1) {
                return f64::NEG_INFINITY;
            }
            let det = a00 * a11 - a01 * a01;
            let (beta_y0, beta_y1, beta_w0, beta_w1) = if det.is_finite() && det > 0.0 {
                // Preserve the existing reduction order in ordinary inputs.
                (
                    (a11 * b0 - a01 * b1) / det,
                    (-a01 * b0 + a00 * b1) / det,
                    (a11 * u0 - a01 * u1) / det,
                    (-a01 * u0 + a00 * u1) / det,
                )
            } else {
                // Avoid determinant overflow/underflow in extreme units.
                let (by0, by1, uw0, uw1) = (
                    b0 / metric_scale,
                    b1 / metric_scale,
                    u0 / metric_scale,
                    u1 / metric_scale,
                );
                (
                    (d1 * by0 - off * by1) / relative_det,
                    (-off * by0 + d0 * by1) / relative_det,
                    (d1 * uw0 - off * uw1) / relative_det,
                    (-off * uw0 + d0 * uw1) / relative_det,
                )
            };
            let numerator = c - u0 * beta_y0 - u1 * beta_y1;
            let denominator = q - u0 * beta_w0 - u1 * beta_w1;
            let denominator_scale = q.abs().max((u0 * beta_w0).abs()).max((u1 * beta_w1).abs());
            if numerator.is_finite()
                && denominator.is_finite()
                && denominator > DENOMINATOR_EPS * denominator_scale
            {
                numerator * numerator / denominator
            } else {
                f64::NEG_INFINITY
            }
        }
        DosageChannel::CenteredProduct => {
            centered_product_score(b0, b1, a00, a11, a01, u0, u1, q, c, mean0, mean1, py_sum)
        }
    }
}

#[inline]
fn candidate_score_values(
    b0: f64,
    b1: f64,
    a00: f64,
    a11: f64,
    a01: f64,
    u0: f64,
    u1: f64,
    q: f64,
    c: f64,
    mean0: f64,
    mean1: f64,
    py_sum: f64,
    dosage_channel: DosageChannel,
) -> ([f64; 5], f64, f64) {
    let dosage = dosage_channel_score(
        dosage_channel,
        b0,
        b1,
        a00,
        a11,
        a01,
        u0,
        u1,
        q,
        c,
        mean0,
        mean1,
        py_sum,
    );

    let numerators = [-b0 - b1 + c, b1 - c, b0 - c, c, b0 + b1 - 2.0 * c];
    let denominators = [
        a00 + a11 + q + 2.0 * a01 - 2.0 * u0 - 2.0 * u1,
        a11 - 2.0 * u1 + q,
        a00 - 2.0 * u0 + q,
        q,
        a00 + a11 + 4.0 * q + 2.0 * a01 - 4.0 * u0 - 4.0 * u1,
    ];
    let mut logic = [f64::NEG_INFINITY; 5];
    let mut logic_best = f64::NEG_INFINITY;
    for gate in 0..5 {
        logic[gate] = score_ratio(numerators[gate], denominators[gate]);
        if logic[gate] > logic_best {
            logic_best = logic[gate];
        }
    }
    (logic, dosage, logic_best.max(dosage))
}

/// Exact candidate-only GRM-Max6 rerank.
///
/// This path intentionally computes only the unique endpoint columns and the
/// requested candidate edges. It does not construct a candidate-subset A/H/C
/// matrix. The endpoint columns are minor-support encoded just like the
/// production `mincell_exact` full-window path; the final logic labels are
/// mapped back to the original marker encoding.
#[inline(never)]
pub(crate) fn candidate_rerank_exact(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: &[f64],
    candidate_first: &[i64],
    candidate_second: &[i64],
    top_k: usize,
    dosage_channel: DosageChannel,
) -> Result<CandidateRerankResult, String> {
    let kernel_started = Instant::now();
    validate_inputs(x, n_samples, n_markers, p_c, Some(y))?;
    let py_started = Instant::now();
    let py = matvec_row_major(p_c, n_samples, y);
    let py_seconds = py_started.elapsed().as_secs_f64();
    let mut result = candidate_rerank_exact_with_py_trusted(
        x,
        n_samples,
        n_markers,
        p_c,
        y,
        &py,
        candidate_first,
        candidate_second,
        top_k,
        dosage_channel,
    )?;
    result.timing.py_seconds = py_seconds;
    result.timing.kernel_seconds = kernel_started.elapsed().as_secs_f64();
    Ok(result)
}

/// Candidate reranking with a precomputed `P_C y`.
///
/// `P_C y` is shared by every candidate window for a fixed phenotype and
/// projection.  The public convenience wrapper above retains the historical
/// API, while the full Max6 -> GRM-Max6 pipeline uses this form to avoid
/// repeating the same O(n^2) matrix-vector product once per window.
#[inline(never)]
pub(crate) fn candidate_rerank_exact_with_py(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: &[f64],
    py: &[f64],
    candidate_first: &[i64],
    candidate_second: &[i64],
    top_k: usize,
    dosage_channel: DosageChannel,
) -> Result<CandidateRerankResult, String> {
    candidate_rerank_exact_with_py_inner(
        x,
        n_samples,
        n_markers,
        p_c,
        y,
        py,
        candidate_first,
        candidate_second,
        top_k,
        dosage_channel,
        true,
    )
}

/// Internal candidate rerank entry point for arrays validated by the caller.
///
/// The production Max6 pipeline validates the BED-derived bits, phenotype and
/// projection once before entering the window loop.  This trusted variant
/// keeps shape checks but avoids rescanning every window's dense genotype and
/// the full n-by-n projection matrix for finiteness.
#[inline(never)]
pub(crate) fn candidate_rerank_exact_with_py_trusted(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: &[f64],
    py: &[f64],
    candidate_first: &[i64],
    candidate_second: &[i64],
    top_k: usize,
    dosage_channel: DosageChannel,
) -> Result<CandidateRerankResult, String> {
    candidate_rerank_exact_with_py_inner(
        x,
        n_samples,
        n_markers,
        p_c,
        y,
        py,
        candidate_first,
        candidate_second,
        top_k,
        dosage_channel,
        false,
    )
}

/// Production exact rerank wrapper with the LMM covariance-scale correction.
/// The historical trusted entry point remains unscaled for reference/debug
/// comparisons; production GRM-Max6 passes `1 / sigma_g2` here.
#[inline(never)]
pub(crate) fn candidate_rerank_exact_with_py_trusted_scaled(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: &[f64],
    py: &[f64],
    candidate_first: &[i64],
    candidate_second: &[i64],
    top_k: usize,
    dosage_channel: DosageChannel,
    score_scale: f64,
) -> Result<CandidateRerankResult, String> {
    validate_positive_score_scale(score_scale)?;
    let mut result = candidate_rerank_exact_with_py_trusted(
        x,
        n_samples,
        n_markers,
        p_c,
        y,
        py,
        candidate_first,
        candidate_second,
        top_k,
        dosage_channel,
    )?;
    apply_score_scale(&mut result, score_scale);
    Ok(result)
}

#[inline(never)]
fn candidate_rerank_exact_with_py_inner(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: &[f64],
    py: &[f64],
    candidate_first: &[i64],
    candidate_second: &[i64],
    top_k: usize,
    dosage_channel: DosageChannel,
    validate_full_inputs: bool,
) -> Result<CandidateRerankResult, String> {
    let kernel_started = Instant::now();
    if validate_full_inputs {
        validate_inputs(x, n_samples, n_markers, p_c, Some(y))?;
    } else {
        validate_candidate_rerank_shapes(x, n_samples, n_markers, p_c, y, py)?;
    }
    if py.len() != n_samples || py.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "candidate rerank projected phenotype length/finite mismatch: got {}, expected {}",
            py.len(),
            n_samples
        ));
    }
    let encoding = min_support_encode(x, n_samples, n_markers)?;
    let pair_set = canonicalize_candidate_pairs(candidate_first, candidate_second, n_markers)?;
    if pair_set.pair_i.is_empty() {
        return Err("candidate rerank requires at least one non-self pair".to_string());
    }

    let mut unique_markers = pair_set
        .pair_i
        .iter()
        .chain(pair_set.pair_j.iter())
        .copied()
        .collect::<Vec<_>>();
    unique_markers.sort_unstable();
    unique_markers.dedup();
    let n_unique = unique_markers.len();
    let mut local_pair_i = Vec::with_capacity(pair_set.pair_i.len());
    let mut local_pair_j = Vec::with_capacity(pair_set.pair_i.len());
    for (&global_i, &global_j) in pair_set.pair_i.iter().zip(pair_set.pair_j.iter()) {
        let local_i = unique_markers
            .binary_search(&global_i)
            .map_err(|_| "candidate endpoint mapping is incomplete".to_string())?;
        let local_j = unique_markers
            .binary_search(&global_j)
            .map_err(|_| "candidate endpoint mapping is incomplete".to_string())?;
        if local_i >= local_j {
            return Err("candidate local pair order is not canonical".to_string());
        }
        local_pair_i.push(local_i);
        local_pair_j.push(local_j);
    }

    let py_sum = py.iter().sum::<f64>();

    let mut x_unique = vec![0.0; n_samples * n_unique];
    for row in 0..n_samples {
        for (local, &global) in unique_markers.iter().enumerate() {
            x_unique[row * n_unique + local] = encoding.x[row * n_markers + global];
        }
    }
    let pcx_started = Instant::now();
    let mut z_unique = vec![0.0; n_samples * n_unique];
    dgemm_row_major(
        false,
        false,
        n_samples,
        n_unique,
        n_samples,
        p_c,
        &x_unique,
        &mut z_unique,
    )?;
    let pcx_u_seconds = pcx_started.elapsed().as_secs_f64();

    let geometry_started = Instant::now();
    let mut marker_a = vec![0.0; n_unique];
    let mut marker_b = vec![0.0; n_unique];
    let mut marker_mean = vec![0.0; n_unique];
    for marker in 0..n_unique {
        let mut a_value = 0.0;
        let mut b_value = 0.0;
        let mut mean_value = 0.0;
        for row in 0..n_samples {
            let x_value = x_unique[row * n_unique + marker];
            a_value += x_value * z_unique[row * n_unique + marker];
            b_value += x_value * py[row];
            mean_value += x_value;
        }
        marker_a[marker] = a_value;
        marker_b[marker] = b_value;
        marker_mean[marker] = mean_value / n_samples as f64;
    }

    let mut candidate_a12 = vec![0.0; local_pair_i.len()];
    let mut candidate_u0 = vec![0.0; local_pair_i.len()];
    let mut candidate_u1 = vec![0.0; local_pair_i.len()];
    for candidate in 0..local_pair_i.len() {
        let local_i = local_pair_i[candidate];
        let local_j = local_pair_j[candidate];
        for row in 0..n_samples {
            let x_i = x_unique[row * n_unique + local_i];
            let x_j = x_unique[row * n_unique + local_j];
            let w = x_i * x_j;
            candidate_a12[candidate] += x_i * z_unique[row * n_unique + local_j];
            candidate_u0[candidate] += w * z_unique[row * n_unique + local_i];
            candidate_u1[candidate] += w * z_unique[row * n_unique + local_j];
        }
    }

    // Candidate-only equivalent of sample-row sparse weighted C. The lookup
    // table is only over unique endpoints and contains no non-candidate edge.
    let lookup_len = checked_matrix_len(n_unique, n_unique, "candidate C lookup")?;
    let mut candidate_lookup = vec![usize::MAX; lookup_len];
    for (candidate, (&local_i, &local_j)) in
        local_pair_i.iter().zip(local_pair_j.iter()).enumerate()
    {
        candidate_lookup[local_i * n_unique + local_j] = candidate;
        candidate_lookup[local_j * n_unique + local_i] = candidate;
    }
    let mut interaction_py = vec![0.0; local_pair_i.len()];
    let mut active = Vec::with_capacity(n_unique);
    for row in 0..n_samples {
        active.clear();
        for marker in 0..n_unique {
            if x_unique[row * n_unique + marker] == 1.0 {
                active.push(marker);
            }
        }
        for left_position in 0..active.len() {
            let left = active[left_position];
            for &right in &active[(left_position + 1)..] {
                let candidate = candidate_lookup[left * n_unique + right];
                if candidate != usize::MAX {
                    interaction_py[candidate] += py[row];
                }
            }
        }
    }

    let candidate_geometry_seconds = geometry_started.elapsed().as_secs_f64();
    let q_started = Instant::now();
    let mincell = mincell_q_kernel_for_pairs(
        &x_unique,
        n_samples,
        n_unique,
        p_c,
        &local_pair_i,
        &local_pair_j,
    )?;
    let mut q = vec![0.0; local_pair_i.len()];
    for candidate in 0..local_pair_i.len() {
        let local_i = local_pair_i[candidate];
        let local_j = local_pair_j[candidate];
        q[candidate] = reconstruct_original_q(
            mincell.q_min[candidate],
            mincell.cell[candidate],
            marker_a[local_i],
            marker_a[local_j],
            candidate_a12[candidate],
            candidate_u0[candidate],
            candidate_u1[candidate],
        );
    }
    let q_seconds = q_started.elapsed().as_secs_f64();

    let score_started = Instant::now();
    let mut logic = vec![[f64::NEG_INFINITY; 5]; local_pair_i.len()];
    let mut dosage = vec![f64::NEG_INFINITY; local_pair_i.len()];
    let mut max6 = vec![f64::NEG_INFINITY; local_pair_i.len()];
    let mut best_logic_gate = vec![-1i64; local_pair_i.len()];
    let mut best_max6_channel = Vec::with_capacity(local_pair_i.len());
    for candidate in 0..local_pair_i.len() {
        let local_i = local_pair_i[candidate];
        let local_j = local_pair_j[candidate];
        let (encoded_logic, encoded_dosage, encoded_max6) = candidate_score_values(
            marker_b[local_i],
            marker_b[local_j],
            marker_a[local_i],
            marker_a[local_j],
            candidate_a12[candidate],
            candidate_u0[candidate],
            candidate_u1[candidate],
            q[candidate],
            interaction_py[candidate],
            marker_mean[local_i],
            marker_mean[local_j],
            py_sum,
            dosage_channel,
        );
        let global_i = pair_set.pair_i[candidate];
        let global_j = pair_set.pair_j[candidate];
        let flip_i = encoding.flipped[global_i];
        let flip_j = encoding.flipped[global_j];
        let mut mapped_logic = [f64::NEG_INFINITY; 5];
        for (gate, &value) in encoded_logic.iter().enumerate() {
            mapped_logic[remap_logic_gate_index(gate, flip_i, flip_j)] = value;
        }
        logic[candidate] = mapped_logic;
        dosage[candidate] = encoded_dosage;
        max6[candidate] = encoded_max6;
        best_logic_gate[candidate] = best_logic_gate_index(&mapped_logic);
        best_max6_channel.push(best_max6_channel_name(&mapped_logic, encoded_dosage));
    }
    let score_seconds = score_started.elapsed().as_secs_f64();

    let global_pair_id = pair_set
        .pair_i
        .iter()
        .zip(pair_set.pair_j.iter())
        .map(|(&left, &right)| canonical_pair_id(n_markers, left, right))
        .collect::<Vec<_>>();
    let topk_started = Instant::now();
    let mut order = (0..max6.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        max6[right]
            .total_cmp(&max6[left])
            .then_with(|| global_pair_id[left].cmp(&global_pair_id[right]))
    });
    let mut candidate_rank = vec![0usize; order.len()];
    for (rank, &candidate) in order.iter().enumerate() {
        candidate_rank[candidate] = rank + 1;
    }
    let top_order = order[..top_k.min(order.len())].to_vec();
    let topk_seconds = topk_started.elapsed().as_secs_f64();

    Ok(CandidateRerankResult {
        pair_i: pair_set.pair_i,
        pair_j: pair_set.pair_j,
        input_to_unique: pair_set.input_to_unique,
        input_self_count: pair_set.input_self_count,
        input_duplicate_count: pair_set.input_duplicate_count,
        global_pair_id,
        unique_markers,
        z: z_unique,
        q,
        mincell_q: mincell.q_min,
        mincell_cell: mincell.cell,
        mincell_support: mincell.support,
        logic,
        dosage,
        max6,
        best_logic_gate,
        best_max6_channel,
        candidate_rank,
        top_order,
        flipped: encoding.flipped,
        timing: CandidateRerankTiming {
            pcx_u_seconds,
            py_seconds: 0.0,
            candidate_geometry_seconds,
            q_seconds,
            score_seconds,
            topk_seconds,
            kernel_seconds: kernel_started.elapsed().as_secs_f64(),
            ..CandidateRerankTiming::default()
        },
    })
}

/// Exact candidate rerank directly from the packed rows emitted by the
/// Max6 screen. This is the production packed path; the dense APIs above are
/// retained as reference/debug entry points.
#[inline(never)]
pub(crate) fn candidate_rerank_exact_packed_with_py_trusted(
    bits: &[u64],
    row_words: usize,
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: &[f64],
    py: &[f64],
    candidate_first: &[i64],
    candidate_second: &[i64],
    top_k: usize,
    dosage_channel: DosageChannel,
) -> Result<CandidateRerankResult, String> {
    candidate_rerank_exact_packed_with_py_trusted_kernel(
        bits,
        row_words,
        n_samples,
        n_markers,
        p_c,
        y,
        py,
        candidate_first,
        candidate_second,
        top_k,
        dosage_channel,
        candidate_au_kernel_from_env()?,
        candidate_pcx_sparse_support_from_env()?,
    )
}

#[inline(never)]
fn candidate_rerank_exact_packed_with_py_trusted_kernel(
    bits: &[u64],
    row_words: usize,
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: &[f64],
    py: &[f64],
    candidate_first: &[i64],
    candidate_second: &[i64],
    top_k: usize,
    dosage_channel: DosageChannel,
    au_kernel: CandidateAuKernel,
    pcx_sparse_support: bool,
) -> Result<CandidateRerankResult, String> {
    let kernel_started = Instant::now();
    let words = n_samples.div_ceil(64);
    if n_samples == 0 || n_markers < 2 {
        return Err(
            "packed candidate rerank requires samples > 0 and at least two markers".to_string(),
        );
    }
    if row_words < words {
        return Err(format!(
            "packed candidate rerank row_words={row_words} smaller than required {words}"
        ));
    }
    let packed_len = checked_matrix_len(n_markers, row_words, "packed genotype")?;
    if bits.len() < packed_len {
        return Err(format!(
            "packed genotype length mismatch: got {}, expected at least {packed_len}",
            bits.len()
        ));
    }
    let pc_len = checked_matrix_len(n_samples, n_samples, "P_C")?;
    if p_c.len() != pc_len {
        return Err(format!(
            "P_C length mismatch: got {}, expected {pc_len}",
            p_c.len()
        ));
    }
    if y.len() != n_samples {
        return Err(format!(
            "phenotype length mismatch: got {}, expected {n_samples}",
            y.len()
        ));
    }
    if py.len() != n_samples || py.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "candidate rerank projected phenotype length/finite mismatch: got {}, expected {}",
            py.len(),
            n_samples
        ));
    }

    let endpoint_prepare_started = Instant::now();
    let (pair_set, unique_markers, local_pair_i, local_pair_j) =
        prepare_candidate_endpoint_mapping(candidate_first, candidate_second, n_markers)?;
    let (x_unique, encoded_unique, flipped, supports_unique, packed_bitset_seconds) =
        packed_minor_support_decode_unique(bits, row_words, n_samples, n_markers, &unique_markers)?;
    if std::env::var_os("JX_GARFIELD_EXPERIMENTAL_CANDIDATE_AU_PROFILE").is_some() {
        let mut w_a = 0u128;
        let mut intersection_sum = 0u128;
        let mut intersections = Vec::with_capacity(local_pair_i.len());
        for (&i, &j) in local_pair_i.iter().zip(local_pair_j.iter()) {
            w_a += supports_unique[i] as u128;
            let intersection = (0..words)
                .map(|word| {
                    (encoded_unique[i * row_words + word] & encoded_unique[j * row_words + word])
                        .count_ones() as usize
                })
                .sum::<usize>();
            intersection_sum += intersection as u128;
            intersections.push(intersection);
        }
        let mut support_summary = supports_unique.clone();
        support_summary.sort_unstable();
        intersections.sort_unstable();
        let quantile_index = |len: usize, q: f64| -> usize {
            if len == 0 {
                0
            } else {
                (((len - 1) as f64 * q).ceil() as usize).min(len - 1)
            }
        };
        let support_quantiles = [0.0, 0.5, 0.9, 0.99, 1.0].map(|q| {
            support_summary
                .get(quantile_index(support_summary.len(), q))
                .copied()
                .unwrap_or(0)
        });
        let intersection_quantiles = [0.0, 0.5, 0.9, 0.99, 1.0].map(|q| {
            intersections
                .get(quantile_index(intersections.len(), q))
                .copied()
                .unwrap_or(0)
        });
        let m_u = unique_markers.len();
        let k_unique = local_pair_i.len();
        let possible = m_u.saturating_mul(m_u.saturating_sub(1)) / 2;
        let density = if possible == 0 {
            0.0
        } else {
            k_unique as f64 / possible as f64
        };
        let blas_work = (n_samples as u128) * (m_u as u128) * (m_u as u128);
        let blas_workspace =
            ((n_samples as u128) * (m_u as u128) + 2 * (m_u as u128) * (m_u as u128)) * 8;
        let k_raw = pair_set.input_to_unique.len();
        let duplicate_fraction = if k_raw == 0 {
            0.0
        } else {
            pair_set.input_duplicate_count as f64 / k_raw as f64
        };
        eprintln!(
            "CANDIDATE_AU_PROFILE n={n_samples} m_u={m_u} K_raw={k_raw} K_unique={k_unique} duplicate={} duplicate_fraction={duplicate_fraction:.8} self_pairs={} density={density:.8} W_A={w_a} support_q0_q50_q90_q99_max={support_quantiles:?} intersection_sum={intersection_sum} intersection_q0_q50_q90_q99_max={intersection_quantiles:?} W_B={blas_work} B_extra_workspace_bytes={blas_workspace} marker_first={} marker_last={}",
            pair_set.input_duplicate_count,
            pair_set.input_self_count, unique_markers.first().copied().unwrap_or(0),
            unique_markers.last().copied().unwrap_or(0),
        );
    }
    let endpoint_prepare_seconds = endpoint_prepare_started.elapsed().as_secs_f64();
    let mincell = mincell_q_kernel_for_packed_pairs(
        &encoded_unique,
        row_words,
        n_samples,
        unique_markers.len(),
        p_c,
        &local_pair_i,
        &local_pair_j,
        &supports_unique,
        packed_bitset_seconds,
    )?;
    candidate_rerank_from_encoded_unique(
        x_unique,
        &encoded_unique,
        row_words,
        n_samples,
        n_markers,
        p_c,
        py,
        pair_set,
        unique_markers,
        local_pair_i,
        local_pair_j,
        flipped,
        mincell,
        top_k,
        dosage_channel,
        kernel_started,
        endpoint_prepare_seconds,
        au_kernel,
        pcx_sparse_support,
    )
}

/// Production packed exact rerank wrapper with the LMM covariance-scale
/// correction.  It preserves the reference packed kernel and only scales the
/// returned exact channel values by the supplied positive factor.
#[inline(never)]
pub(crate) fn candidate_rerank_exact_packed_with_py_trusted_scaled(
    bits: &[u64],
    row_words: usize,
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: &[f64],
    py: &[f64],
    candidate_first: &[i64],
    candidate_second: &[i64],
    top_k: usize,
    dosage_channel: DosageChannel,
    score_scale: f64,
) -> Result<CandidateRerankResult, String> {
    validate_positive_score_scale(score_scale)?;
    let mut result = candidate_rerank_exact_packed_with_py_trusted(
        bits,
        row_words,
        n_samples,
        n_markers,
        p_c,
        y,
        py,
        candidate_first,
        candidate_second,
        top_k,
        dosage_channel,
    )?;
    apply_score_scale(&mut result, score_scale);
    Ok(result)
}

/// Exact packed candidate rerank for the no-GRM boundary using a compact
/// fixed-effect projector.  This is algebraically identical to the dense
/// `P_C` path with `P_C = I - Q Q'`, but never materializes an `n x n`
/// matrix.  It is intentionally separate from the dense reference path so
/// that the two implementations can be kept in direct numerical regression.
#[inline(never)]
pub(crate) fn candidate_rerank_exact_packed_no_grm_covariate_with_py_trusted_scaled(
    bits: &[u64],
    row_words: usize,
    n_samples: usize,
    n_markers: usize,
    projector: &NoGrmCovariateProjector,
    y: &[f64],
    projected_y: &[f64],
    candidate_first: &[i64],
    candidate_second: &[i64],
    top_k: usize,
    dosage_channel: DosageChannel,
    score_scale: f64,
) -> Result<CandidateRerankResult, String> {
    validate_positive_score_scale(score_scale)?;
    let kernel_started = Instant::now();
    let words = n_samples.div_ceil(64);
    if n_samples == 0 || n_markers < 2 {
        return Err(
            "packed no-GRM candidate rerank requires samples > 0 and at least two markers"
                .to_string(),
        );
    }
    if projector.n_samples() != n_samples || projector.rank() == 0 || projector.rank() >= n_samples
    {
        return Err(format!(
            "no-GRM projector shape mismatch: n={}, rank={}, expected n={}, rank in 1..{}",
            projector.n_samples(),
            projector.rank(),
            n_samples,
            n_samples.saturating_sub(1)
        ));
    }
    if row_words < words {
        return Err(format!(
            "packed no-GRM candidate rerank row_words={row_words} smaller than required {words}"
        ));
    }
    let packed_len = checked_matrix_len(n_markers, row_words, "packed genotype")?;
    if bits.len() < packed_len {
        return Err(format!(
            "packed genotype length mismatch: got {}, expected at least {packed_len}",
            bits.len()
        ));
    }
    if y.len() != n_samples || projected_y.len() != n_samples {
        return Err(format!(
            "no-GRM candidate phenotype lengths must both equal {n_samples}, got y={} projected_y={}",
            y.len(),
            projected_y.len()
        ));
    }
    if y.iter()
        .chain(projected_y.iter())
        .any(|value| !value.is_finite())
    {
        return Err("no-GRM candidate phenotype contains a non-finite value".to_string());
    }

    let (pair_set, unique_markers, local_pair_i, local_pair_j) =
        prepare_candidate_endpoint_mapping(candidate_first, candidate_second, n_markers)?;
    let (x_unique, encoded_unique, flipped, supports_unique, _packed_bitset_seconds) =
        packed_minor_support_decode_unique(bits, row_words, n_samples, n_markers, &unique_markers)?;
    let n_unique = unique_markers.len();
    let rank = projector.rank();
    let last_bits = n_samples % 64;
    let last_mask = if last_bits == 0 {
        u64::MAX
    } else {
        (1u64 << last_bits) - 1
    };

    let projection_started = Instant::now();
    // Q'X, stored in marker-major order for cheap pairwise coordinate dots.
    let mut basis_coords_by_basis = vec![0.0_f64; rank * n_unique];
    dgemm_row_major(
        true,
        false,
        rank,
        n_unique,
        n_samples,
        projector.q(),
        &x_unique,
        &mut basis_coords_by_basis,
    )?;
    let mut marker_coords = vec![0.0_f64; n_unique * rank];
    for marker in 0..n_unique {
        for coordinate in 0..rank {
            marker_coords[marker * rank + coordinate] =
                basis_coords_by_basis[coordinate * n_unique + marker];
        }
    }
    let mut fitted_unique = vec![0.0_f64; n_samples * n_unique];
    dgemm_row_major(
        false,
        false,
        n_samples,
        n_unique,
        rank,
        projector.q(),
        &basis_coords_by_basis,
        &mut fitted_unique,
    )?;
    let mut z_unique = x_unique.clone();
    for (value, fitted) in z_unique.iter_mut().zip(fitted_unique.iter()) {
        *value -= fitted;
    }
    let pcx_u_seconds = projection_started.elapsed().as_secs_f64();

    let geometry_started = Instant::now();
    let projected_y_sum = projected_y.iter().sum::<f64>();
    let mut marker_b = vec![0.0_f64; n_unique];
    let mut marker_a = vec![0.0_f64; n_unique];
    let mut marker_mean = vec![0.0_f64; n_unique];
    for marker in 0..n_unique {
        let support = supports_unique[marker];
        marker_mean[marker] = support as f64 / n_samples as f64;
        for word in 0..words {
            let valid = if word + 1 == words {
                last_mask
            } else {
                u64::MAX
            };
            let mut selected = encoded_unique[marker * row_words + word] & valid;
            while selected != 0 {
                let row = word * 64 + selected.trailing_zeros() as usize;
                marker_b[marker] += projected_y[row];
                selected &= selected - 1;
            }
        }
        let coordinates = &marker_coords[marker * rank..(marker + 1) * rank];
        marker_a[marker] =
            support as f64 - coordinates.iter().map(|value| value * value).sum::<f64>();
    }
    let one_coordinates = (0..rank)
        .map(|coordinate| {
            (0..n_samples)
                .map(|row| projector.q()[row * rank + coordinate])
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let mut q = vec![0.0_f64; local_pair_i.len()];
    let mut mincell_q = vec![0.0_f64; local_pair_i.len()];
    let mut mincell_cell = vec![0u8; local_pair_i.len()];
    let mut mincell_support = vec![0usize; local_pair_i.len()];
    let mut logic = vec![[f64::NEG_INFINITY; 5]; local_pair_i.len()];
    let mut dosage = vec![f64::NEG_INFINITY; local_pair_i.len()];
    let mut max6 = vec![f64::NEG_INFINITY; local_pair_i.len()];
    let mut best_logic_gate = vec![-1i64; local_pair_i.len()];
    let mut best_max6_channel = Vec::with_capacity(local_pair_i.len());
    let mut work_mincell = 0.0_f64;
    let mut q_w = vec![0.0_f64; rank];

    for candidate in 0..local_pair_i.len() {
        let local_i = local_pair_i[candidate];
        let local_j = local_pair_j[candidate];
        let mut n11 = 0usize;
        let mut sum11 = 0.0_f64;
        q_w.fill(0.0);
        for word in 0..words {
            let valid = if word + 1 == words {
                last_mask
            } else {
                u64::MAX
            };
            let mut selected = (encoded_unique[local_i * row_words + word]
                & encoded_unique[local_j * row_words + word])
                & valid;
            while selected != 0 {
                let row = word * 64 + selected.trailing_zeros() as usize;
                n11 += 1;
                sum11 += projected_y[row];
                for coordinate in 0..rank {
                    q_w[coordinate] += projector.q()[row * rank + coordinate];
                }
                selected &= selected - 1;
            }
        }
        let left_coords = &marker_coords[local_i * rank..(local_i + 1) * rank];
        let right_coords = &marker_coords[local_j * rank..(local_j + 1) * rank];
        let a12 = n11 as f64
            - left_coords
                .iter()
                .zip(right_coords.iter())
                .map(|(&left, &right)| left * right)
                .sum::<f64>();
        let u0 = n11 as f64
            - q_w
                .iter()
                .zip(left_coords.iter())
                .map(|(&left, &right)| left * right)
                .sum::<f64>();
        let u1 = n11 as f64
            - q_w
                .iter()
                .zip(right_coords.iter())
                .map(|(&left, &right)| left * right)
                .sum::<f64>();
        let q_value = n11 as f64 - q_w.iter().map(|value| value * value).sum::<f64>();
        q[candidate] = q_value;

        let n00 = n_samples
            .checked_add(n11)
            .and_then(|value| value.checked_sub(supports_unique[local_i]))
            .and_then(|value| value.checked_sub(supports_unique[local_j]))
            .ok_or_else(|| "no-GRM candidate q produced an invalid 00 cell count".to_string())?;
        let counts = [
            n00,
            supports_unique[local_j] - n11,
            supports_unique[local_i] - n11,
            n11,
        ];
        let mut cell_q = [0.0_f64; 4];
        for coordinate in 0..rank {
            let q11 = q_w[coordinate];
            let q10 = right_coords[coordinate] - q11;
            let q01 = left_coords[coordinate] - q11;
            let q00 =
                one_coordinates[coordinate] - left_coords[coordinate] - right_coords[coordinate]
                    + q11;
            cell_q[0] += q00 * q00;
            cell_q[1] += q10 * q10;
            cell_q[2] += q01 * q01;
            cell_q[3] += q11 * q11;
        }
        let mut chosen = 0usize;
        for cell_index in 1..4 {
            if counts[cell_index] < counts[chosen] {
                chosen = cell_index;
            }
        }
        mincell_cell[candidate] = chosen as u8;
        mincell_support[candidate] = counts[chosen];
        work_mincell += (counts[chosen] as f64) * (counts[chosen] as f64);
        mincell_q[candidate] = counts[chosen] as f64 - cell_q[chosen];

        let c = sum11;
        let (encoded_logic, encoded_dosage, encoded_max6) = candidate_score_values(
            marker_b[local_i],
            marker_b[local_j],
            marker_a[local_i],
            marker_a[local_j],
            a12,
            u0,
            u1,
            q_value,
            c,
            marker_mean[local_i],
            marker_mean[local_j],
            projected_y_sum,
            dosage_channel,
        );
        let global_i = pair_set.pair_i[candidate];
        let global_j = pair_set.pair_j[candidate];
        let flip_i = flipped[global_i];
        let flip_j = flipped[global_j];
        let mut mapped_logic = [f64::NEG_INFINITY; 5];
        for (gate, &value) in encoded_logic.iter().enumerate() {
            mapped_logic[remap_logic_gate_index(gate, flip_i, flip_j)] = value;
        }
        logic[candidate] = mapped_logic;
        dosage[candidate] = encoded_dosage;
        max6[candidate] = encoded_max6;
        best_logic_gate[candidate] = best_logic_gate_index(&mapped_logic);
        best_max6_channel.push(best_max6_channel_name(&mapped_logic, encoded_dosage));
    }
    let candidate_geometry_seconds = geometry_started.elapsed().as_secs_f64();

    let work_subset = {
        let mut sorted = supports_unique.clone();
        sorted.sort_unstable();
        sorted
            .iter()
            .enumerate()
            .map(|(position, &support)| {
                (support as f64) * (support as f64) * (n_unique - position - 1) as f64
            })
            .sum::<f64>()
    };
    let work_dense = n_samples as f64 * n_samples as f64 * local_pair_i.len() as f64;
    let _mincell_work = MinCellQWork {
        work_mincell,
        work_subset,
        work_dense,
        mincell_subset_ratio: if work_subset > 0.0 {
            work_mincell / work_subset
        } else {
            0.0
        },
    };
    let q_seconds = candidate_geometry_seconds;
    let score_seconds = 0.0;
    let global_pair_id = pair_set
        .pair_i
        .iter()
        .zip(pair_set.pair_j.iter())
        .map(|(&left, &right)| canonical_pair_id(n_markers, left, right))
        .collect::<Vec<_>>();
    let topk_started = Instant::now();
    let mut order = (0..max6.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        max6[right]
            .total_cmp(&max6[left])
            .then_with(|| global_pair_id[left].cmp(&global_pair_id[right]))
    });
    let mut candidate_rank = vec![0usize; order.len()];
    for (rank, &candidate) in order.iter().enumerate() {
        candidate_rank[candidate] = rank + 1;
    }
    let top_order = order[..top_k.min(order.len())].to_vec();
    let topk_seconds = topk_started.elapsed().as_secs_f64();

    let mut result = CandidateRerankResult {
        pair_i: pair_set.pair_i,
        pair_j: pair_set.pair_j,
        input_to_unique: pair_set.input_to_unique,
        input_self_count: pair_set.input_self_count,
        input_duplicate_count: pair_set.input_duplicate_count,
        global_pair_id,
        unique_markers,
        z: z_unique,
        q,
        mincell_q,
        mincell_cell,
        mincell_support,
        logic,
        dosage,
        max6,
        best_logic_gate,
        best_max6_channel,
        candidate_rank,
        top_order,
        flipped,
        timing: CandidateRerankTiming {
            pcx_u_seconds,
            py_seconds: 0.0,
            candidate_geometry_seconds,
            q_seconds,
            score_seconds,
            topk_seconds,
            kernel_seconds: kernel_started.elapsed().as_secs_f64(),
            ..CandidateRerankTiming::default()
        },
    };
    apply_score_scale(&mut result, score_scale);
    Ok(result)
}

fn validate_spectral_inputs(
    n_samples: usize,
    eigenvalues: &[f64],
    eigenvectors: &[f64],
    lambda_vc: f64,
) -> Result<(), String> {
    if eigenvalues.len() != n_samples {
        return Err(format!(
            "spectral eigenvalue length mismatch: got {}, expected {n_samples}",
            eigenvalues.len()
        ));
    }
    let expected_vectors = checked_matrix_len(n_samples, n_samples, "eigenvectors")?;
    if eigenvectors.len() != expected_vectors {
        return Err(format!(
            "spectral eigenvector length mismatch: got {}, expected {expected_vectors}",
            eigenvectors.len()
        ));
    }
    if !lambda_vc.is_finite() || lambda_vc <= 0.0 {
        return Err(format!(
            "spectral lambda_vc must be finite and > 0, got {lambda_vc}"
        ));
    }
    if eigenvalues
        .iter()
        .chain(eigenvectors.iter())
        .any(|value| !value.is_finite())
    {
        return Err("spectral eigenvalues and eigenvectors must be finite".to_string());
    }
    Ok(())
}

/// Compute `q=w'P_Cw` from a full GRM eigendecomposition without materializing
/// or multiplying by `P_C` for each interaction block.
fn spectral_q_kernel(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    eigenvalues: &[f64],
    eigenvectors: &[f64],
    lambda_vc: f64,
    pair_i: &[usize],
    pair_j: &[usize],
    block_pairs: usize,
) -> Result<Vec<f64>, String> {
    validate_genotype_inputs(x, n_samples, n_markers)?;
    validate_spectral_inputs(n_samples, eigenvalues, eigenvectors, lambda_vc)?;
    if pair_i.len() != pair_j.len() {
        return Err("spectral q pair index lengths differ".to_string());
    }
    if block_pairs == 0 {
        return Err("block_pairs must be > 0".to_string());
    }

    let weights = eigenvalues
        .iter()
        .map(|&value| {
            let denominator = value + lambda_vc;
            if !denominator.is_finite() || denominator <= 0.0 {
                None
            } else {
                Some(1.0 / denominator)
            }
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "spectral eigenvalue + lambda_vc must be finite and > 0".to_string())?;
    let mut one_rot = vec![0.0; n_samples];
    for row in 0..n_samples {
        for coordinate in 0..n_samples {
            one_rot[coordinate] += eigenvectors[row * n_samples + coordinate];
        }
    }
    let weighted_one = one_rot
        .iter()
        .zip(weights.iter())
        .map(|(&value, &weight)| value * weight)
        .collect::<Vec<_>>();
    let information = one_rot
        .iter()
        .zip(weighted_one.iter())
        .map(|(&left, &right)| left * right)
        .sum::<f64>();
    if !information.is_finite() || information <= DENOMINATOR_EPS {
        return Err("spectral intercept information is not positive".to_string());
    }

    let block_pairs = block_pairs.max(1);
    let mut q = vec![0.0; pair_i.len()];
    let mut start = 0usize;
    while start < pair_i.len() {
        let width = block_pairs.min(pair_i.len() - start);
        let mut w = vec![0.0; n_samples * width];
        for row in 0..n_samples {
            for offset in 0..width {
                let marker_i = pair_i[start + offset];
                let marker_j = pair_j[start + offset];
                if marker_i >= n_markers || marker_j >= n_markers {
                    return Err("spectral q pair index is outside marker range".to_string());
                }
                w[row * width + offset] =
                    x[row * n_markers + marker_i] * x[row * n_markers + marker_j];
            }
        }
        let mut rotated = vec![0.0; n_samples * width];
        dgemm_row_major(
            true,
            false,
            n_samples,
            width,
            n_samples,
            eigenvectors,
            &w,
            &mut rotated,
        )?;
        for offset in 0..width {
            let mut intercept = 0.0;
            for coordinate in 0..n_samples {
                let value = rotated[coordinate * width + offset];
                intercept += weighted_one[coordinate] * value;
            }
            let alpha = intercept / information;
            let mut stable_q = 0.0;
            for coordinate in 0..n_samples {
                let residual = rotated[coordinate * width + offset] - alpha * one_rot[coordinate];
                stable_q += weights[coordinate] * residual * residual;
            }
            q[start + offset] = stable_q;
        }
        start += width;
    }
    Ok(q)
}

#[cfg(test)]
pub(crate) fn reference_geometry(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    block_pairs: usize,
) -> Result<Phase1Geometry, String> {
    validate_inputs(x, n_samples, n_markers, p_c, None)?;
    let (pair_i, pair_j) = pair_indices(n_markers);
    let mut z = vec![0.0; n_samples * n_markers];
    for marker in 0..n_markers {
        let mut column = vec![0.0; n_samples];
        for row in 0..n_samples {
            column[row] = x[row * n_markers + marker];
        }
        let projected = matvec_row_major(p_c, n_samples, &column);
        for row in 0..n_samples {
            z[row * n_markers + marker] = projected[row];
        }
    }
    let mut a = vec![0.0; n_markers * n_markers];
    let mut h = vec![0.0; n_markers * n_markers];
    for left in 0..n_markers {
        for right in 0..n_markers {
            let mut a_value = 0.0;
            let mut h_value = 0.0;
            for row in 0..n_samples {
                let x_left = x[row * n_markers + left];
                let x_right = x[row * n_markers + right];
                let z_right = z[row * n_markers + right];
                a_value += x_left * z_right;
                h_value += x_left * x_right * z_right;
            }
            a[left * n_markers + right] = a_value;
            h[left * n_markers + right] = h_value;
        }
    }
    let q = old_q_kernel(x, n_samples, n_markers, p_c, &pair_i, &pair_j, block_pairs)?;
    Ok(Phase1Geometry {
        pair_i,
        pair_j,
        z,
        a,
        h,
        q,
        mincell_q: None,
        mincell_cell: None,
        mincell_support: None,
    })
}

#[cfg(test)]
pub(crate) fn phase1_geometry(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    block_pairs: usize,
) -> Result<Phase1Geometry, String> {
    Ok(phase1_geometry_timed(x, n_samples, n_markers, p_c, block_pairs)?.0)
}

pub(crate) fn phase1_geometry_timed(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    block_pairs: usize,
) -> Result<(Phase1Geometry, Phase1Timing), String> {
    geometry_with_q_timed(
        x,
        n_samples,
        n_markers,
        p_c,
        block_pairs,
        |pair_i, pair_j| {
            Ok((
                old_q_kernel(x, n_samples, n_markers, p_c, pair_i, pair_j, block_pairs)?,
                None,
                None,
            ))
        },
    )
}

fn phase2_geometry_timed(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    eigenvalues: &[f64],
    eigenvectors: &[f64],
    lambda_vc: f64,
    block_pairs: usize,
) -> Result<(Phase1Geometry, Phase1Timing), String> {
    geometry_with_q_timed(
        x,
        n_samples,
        n_markers,
        p_c,
        block_pairs,
        |pair_i, pair_j| {
            Ok((
                spectral_q_kernel(
                    x,
                    n_samples,
                    n_markers,
                    eigenvalues,
                    eigenvectors,
                    lambda_vc,
                    pair_i,
                    pair_j,
                    block_pairs,
                )?,
                None,
                None,
            ))
        },
    )
}

fn phase3_geometry_timed(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    block_pairs: usize,
    support_sorted: bool,
) -> Result<(Phase1Geometry, Phase1Timing), String> {
    geometry_with_q_timed(
        x,
        n_samples,
        n_markers,
        p_c,
        block_pairs,
        |pair_i, pair_j| {
            let result =
                subset_q_kernel(x, n_samples, n_markers, p_c, pair_i, pair_j, support_sorted)?;
            Ok((
                result.q,
                Some(SubsetQWork {
                    work_subset: result.work_subset,
                    work_dense: result.work_dense,
                    work_ratio: result.work_ratio,
                }),
                Some(result.timing),
            ))
        },
    )
}

fn phase6b_geometry_timed(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    block_pairs: usize,
) -> Result<(Phase1Geometry, Phase1Timing), String> {
    let mut mincell_result = None;
    let (mut geometry, mut timing) = geometry_with_q_timed(
        x,
        n_samples,
        n_markers,
        p_c,
        block_pairs,
        |pair_i, pair_j| {
            let result = mincell_q_kernel(x, n_samples, n_markers, p_c, pair_i, pair_j)?;
            let q_min = result.q_min.clone();
            mincell_result = Some(result);
            Ok((q_min, None, None))
        },
    )?;
    let result = mincell_result.ok_or_else(|| "mincell q result was not produced".to_string())?;
    let reconstruct_started = Instant::now();
    let mut q_original = vec![0.0; geometry.q.len()];
    for pair in 0..geometry.q.len() {
        let i = geometry.pair_i[pair];
        let j = geometry.pair_j[pair];
        let a1 = geometry.a[i * n_markers + i];
        let a2 = geometry.a[j * n_markers + j];
        let a12 = geometry.a[i * n_markers + j];
        let u0 = geometry.h[j * n_markers + i];
        let u1 = geometry.h[i * n_markers + j];
        q_original[pair] =
            reconstruct_original_q(result.q_min[pair], result.cell[pair], a1, a2, a12, u0, u1);
    }
    let reconstruct_seconds = reconstruct_started.elapsed().as_secs_f64();
    let mut q_timing = result.timing;
    q_timing.reconstruct_seconds = reconstruct_seconds;
    geometry.q = q_original;
    geometry.mincell_q = Some(result.q_min);
    geometry.mincell_cell = Some(result.cell);
    geometry.mincell_support = Some(result.support);
    timing.mincell_work = Some(result.work);
    timing.q_seconds += reconstruct_seconds;
    timing.geometry_seconds += reconstruct_seconds;
    timing.mincell_q_timing = Some(q_timing);
    Ok((geometry, timing))
}

fn geometry_with_q_timed<Q>(
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    block_pairs: usize,
    q_kernel: Q,
) -> Result<(Phase1Geometry, Phase1Timing), String>
where
    Q: FnOnce(
        &[usize],
        &[usize],
    ) -> Result<(Vec<f64>, Option<SubsetQWork>, Option<SubsetQTiming>), String>,
{
    validate_inputs(x, n_samples, n_markers, p_c, None)?;
    if block_pairs == 0 {
        return Err("block_pairs must be > 0".to_string());
    }
    let geometry_started = Instant::now();
    let (pair_i, pair_j) = pair_indices(n_markers);

    let pcx_started = Instant::now();
    let mut z = vec![0.0; n_samples * n_markers];
    dgemm_row_major(
        false, false, n_samples, n_markers, n_samples, p_c, x, &mut z,
    )?;
    let pcx_seconds = pcx_started.elapsed().as_secs_f64();

    let a_started = Instant::now();
    let mut a = vec![0.0; n_markers * n_markers];
    dgemm_row_major(true, false, n_markers, n_markers, n_samples, x, &z, &mut a)?;
    let a_seconds = a_started.elapsed().as_secs_f64();

    let xz_build_started = Instant::now();
    let mut xz = vec![0.0; n_samples * n_markers];
    for (value, (&x_value, &z_value)) in xz.iter_mut().zip(x.iter().zip(z.iter())) {
        *value = x_value * z_value;
    }
    let xz_build_seconds = xz_build_started.elapsed().as_secs_f64();

    let h_gram_started = Instant::now();
    let mut h = vec![0.0; n_markers * n_markers];
    dgemm_row_major(true, false, n_markers, n_markers, n_samples, x, &xz, &mut h)?;
    let h_gram_seconds = h_gram_started.elapsed().as_secs_f64();
    let h_seconds = xz_build_seconds + h_gram_seconds;

    let q_started = Instant::now();
    let (q, subset_work, subset_q_timing) = q_kernel(&pair_i, &pair_j)?;
    let q_seconds = q_started.elapsed().as_secs_f64();

    let timing = Phase1Timing {
        pcx_seconds,
        a_seconds,
        xz_build_seconds,
        h_gram_seconds,
        h_seconds,
        q_seconds,
        geometry_seconds: geometry_started.elapsed().as_secs_f64(),
        subset_work,
        subset_q_timing,
        mincell_work: None,
        mincell_q_timing: None,
    };
    Ok((
        Phase1Geometry {
            pair_i,
            pair_j,
            z,
            a,
            h,
            q,
            mincell_q: None,
            mincell_cell: None,
            mincell_support: None,
        },
        timing,
    ))
}

fn score_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.is_finite() && denominator > DENOMINATOR_EPS {
        numerator * numerator / denominator
    } else {
        f64::NEG_INFINITY
    }
}

#[inline(never)]
fn weighted_interaction_py(
    x: &[f64],
    py: &[f64],
    n_samples: usize,
    n_markers: usize,
    pair_i: &[usize],
    pair_j: &[usize],
) -> Vec<f64> {
    let mut interaction_py = vec![0.0; pair_i.len()];
    for pair in 0..pair_i.len() {
        let i = pair_i[pair];
        let j = pair_j[pair];
        let mut value = 0.0;
        for row in 0..n_samples {
            value += x[row * n_markers + i] * x[row * n_markers + j] * py[row];
        }
        interaction_py[pair] = value;
    }
    interaction_py
}

/// Compute the exact weighted interaction Gram values by visiting only the
/// active binary supports in each sample. The returned values use the
/// canonical `(i, j)` pair order produced by [`pair_indices`].
///
/// This is intentionally a separate prototype from the dense reference
/// kernel. It does not materialize the `m x m` weighted Gram matrix; it only
/// accumulates the upper-triangle pair vector required by the score loop.
#[inline(never)]
fn weighted_interaction_py_binary_sparse(
    x: &[f64],
    py: &[f64],
    n_samples: usize,
    n_markers: usize,
    pair_i: &[usize],
    pair_j: &[usize],
) -> Result<Vec<f64>, String> {
    validate_binary_genotype_inputs(x, n_samples, n_markers)?;
    if py.len() != n_samples {
        return Err(format!(
            "weighted sparse C phenotype length mismatch: got {}, expected {n_samples}",
            py.len()
        ));
    }
    if py.iter().any(|value| !value.is_finite()) {
        return Err("weighted sparse C phenotype values must be finite".to_string());
    }
    if pair_i.len() != pair_j.len() {
        return Err("weighted sparse C pair index lengths differ".to_string());
    }
    let expected_pairs = n_markers
        .checked_mul(n_markers.saturating_sub(1))
        .map(|value| value / 2)
        .ok_or_else(|| "weighted sparse C pair count overflow".to_string())?;
    if pair_i.len() != expected_pairs {
        return Err(format!(
            "weighted sparse C requires all canonical pairs: got {}, expected {expected_pairs}",
            pair_i.len()
        ));
    }
    for i in 0..n_markers {
        for j in (i + 1)..n_markers {
            let pair = i * n_markers - i * (i + 1) / 2 + (j - i - 1);
            if pair_i[pair] != i || pair_j[pair] != j {
                return Err("weighted sparse C requires canonical pair order".to_string());
            }
        }
    }

    let mut interaction_py = vec![0.0; pair_i.len()];
    let mut active = Vec::with_capacity(n_markers);
    for row in 0..n_samples {
        active.clear();
        let row_offset = row * n_markers;
        for marker in 0..n_markers {
            if x[row_offset + marker] == 1.0 {
                active.push(marker);
            }
        }
        for left in 0..active.len() {
            let i = active[left];
            for &j in active[(left + 1)..].iter() {
                let pair = i * n_markers - i * (i + 1) / 2 + (j - i - 1);
                interaction_py[pair] += py[row];
            }
        }
    }
    Ok(interaction_py)
}

#[inline]
fn packed_upper_index(n_markers: usize, row: usize, col: usize) -> usize {
    debug_assert!(row <= col && col < n_markers);
    let row_start = row * n_markers - row * row.saturating_sub(1) / 2;
    row_start + (col - row)
}

/// Build the exact weighted interaction matrix in packed upper-triangle form
/// by visiting active marker pairs once per sample. The diagonal is filled
/// directly from `b_j = sum_i p_y[i] * x[i, j]` and is never accumulated as a
/// sample-pair contribution.
#[inline(never)]
fn sample_row_sparse_weighted_c(
    x: &[f64],
    py: &[f64],
    marker_py: &[f64],
    n_samples: usize,
    n_markers: usize,
) -> Result<SampleRowWeightedCResult, String> {
    validate_binary_genotype_inputs(x, n_samples, n_markers)?;
    if py.len() != n_samples {
        return Err(format!(
            "sample-row sparse C phenotype length mismatch: got {}, expected {n_samples}",
            py.len()
        ));
    }
    if marker_py.len() != n_markers {
        return Err(format!(
            "sample-row sparse C marker-weight length mismatch: got {}, expected {n_markers}",
            marker_py.len()
        ));
    }
    if py.iter().any(|value| !value.is_finite()) {
        return Err("sample-row sparse C phenotype values must be finite".to_string());
    }
    if marker_py.iter().any(|value| !value.is_finite()) {
        return Err("sample-row sparse C marker weights must be finite".to_string());
    }
    let packed_len = n_markers
        .checked_mul(n_markers.checked_add(1).ok_or_else(|| {
            "sample-row sparse C packed upper-triangle length overflow".to_string()
        })?)
        .map(|value| value / 2)
        .ok_or_else(|| "sample-row sparse C packed upper-triangle length overflow".to_string())?;
    let mut c_upper = vec![0.0; packed_len];
    let mut support = vec![0usize; n_markers];
    for row in 0..n_samples {
        let row_offset = row * n_markers;
        for marker in 0..n_markers {
            if x[row_offset + marker] == 1.0 {
                support[marker] += 1;
            }
        }
    }
    for marker in 0..n_markers {
        c_upper[packed_upper_index(n_markers, marker, marker)] = marker_py[marker];
    }

    let mut sorted_support = support.clone();
    sorted_support.sort_unstable();
    let w_anchor = sorted_support
        .iter()
        .enumerate()
        .map(|(position, &value)| (value as u64) * (n_markers.saturating_sub(position + 1) as u64))
        .sum::<u64>();
    let w_dense =
        (n_samples as u64) * (n_markers.saturating_mul(n_markers.saturating_sub(1)) as u64 / 2);

    let mut active = Vec::with_capacity(n_markers);
    let mut w_row = 0u64;
    for row in 0..n_samples {
        active.clear();
        let row_offset = row * n_markers;
        for marker in 0..n_markers {
            if x[row_offset + marker] == 1.0 {
                active.push(marker);
            }
        }
        w_row += (active.len() as u64) * (active.len().saturating_sub(1) as u64) / 2;
        for left in 0..active.len() {
            let marker_i = active[left];
            for &marker_j in &active[(left + 1)..] {
                let index = packed_upper_index(n_markers, marker_i, marker_j);
                c_upper[index] += py[row];
            }
        }
    }

    Ok(SampleRowWeightedCResult {
        c_upper,
        w_anchor,
        w_row,
        w_dense,
    })
}

/// Build a complete weighted interaction matrix as a packed upper triangle.
///
/// This is a reference-only diagnostic API. It does not enter the Max6 score
/// assembly or change any production C kernel. The returned `c_upper` contains
/// the diagonal and all `j < k` entries in row-packed upper-triangle order;
/// symmetry reconstructs the lower triangle.
#[pyfunction(name = "garfield_weighted_c_sample_row_sparse")]
pub fn garfield_weighted_c_sample_row_sparse_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    py_values: PyReadonlyArray1<'py, f64>,
    marker_py: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let x_shape = x.shape();
    if x_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "garfield_weighted_c_sample_row_sparse: x must be two-dimensional",
        ));
    }
    let n_samples = x_shape[0];
    let n_markers = x_shape[1];
    let x_vec = x.as_array().iter().copied().collect::<Vec<_>>();
    let py_vec = py_values.as_array().iter().copied().collect::<Vec<_>>();
    let marker_py_vec = marker_py.as_array().iter().copied().collect::<Vec<_>>();
    let started = Instant::now();
    let result =
        sample_row_sparse_weighted_c(&x_vec, &py_vec, &marker_py_vec, n_samples, n_markers)
            .map_err(PyValueError::new_err)?;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let out = PyDict::new(py);
    out.set_item(
        "c_upper",
        PyArray1::from_owned_array(py, Array1::from_vec(result.c_upper)).into_bound(),
    )?;
    out.set_item("packed_upper_triangle", true)?;
    out.set_item("n_samples", n_samples)?;
    out.set_item("n_markers", n_markers)?;
    out.set_item("w_anchor", result.w_anchor)?;
    out.set_item("w_row", result.w_row)?;
    out.set_item("w_dense", result.w_dense)?;
    out.set_item("seconds", elapsed_seconds)?;
    out.set_item(
        "ns_per_contribution",
        if result.w_row > 0 {
            Some(elapsed_seconds * 1.0e9 / result.w_row as f64)
        } else {
            None
        },
    )?;
    out.set_item(
        "ns_per_anchor_work",
        if result.w_anchor > 0 {
            Some(elapsed_seconds * 1.0e9 / result.w_anchor as f64)
        } else {
            None
        },
    )?;
    Ok(out)
}

/// Compute the exact weighted interaction Gram values in minor-support space.
///
/// Each anchor visits only its active rows and evaluates the complete tail of
/// markers from that support. This keeps the result in canonical pair order
/// while avoiding both the dense `n x m x m` traversal and the sample-level
/// active-pair updates used by `weighted_interaction_py_binary_sparse`.
#[inline(never)]
fn weighted_interaction_py_minor_support(
    x: &[f64],
    py: &[f64],
    n_samples: usize,
    n_markers: usize,
    pair_i: &[usize],
    pair_j: &[usize],
) -> Result<Vec<f64>, String> {
    validate_binary_genotype_inputs(x, n_samples, n_markers)?;
    if py.len() != n_samples {
        return Err(format!(
            "minor-support C phenotype length mismatch: got {}, expected {n_samples}",
            py.len()
        ));
    }
    if py.iter().any(|value| !value.is_finite()) {
        return Err("minor-support C phenotype values must be finite".to_string());
    }
    if pair_i.len() != pair_j.len() {
        return Err("minor-support C pair index lengths differ".to_string());
    }
    let expected_pairs = n_markers
        .checked_mul(n_markers.saturating_sub(1))
        .map(|value| value / 2)
        .ok_or_else(|| "minor-support C pair count overflow".to_string())?;
    if pair_i.len() != expected_pairs {
        return Err(format!(
            "minor-support C requires all canonical pairs: got {}, expected {expected_pairs}",
            pair_i.len()
        ));
    }
    for i in 0..n_markers {
        for j in (i + 1)..n_markers {
            let pair = i * n_markers - i * (i + 1) / 2 + (j - i - 1);
            if pair_i[pair] != i || pair_j[pair] != j {
                return Err("minor-support C requires canonical pair order".to_string());
            }
        }
    }

    let mut interaction_py = vec![0.0; pair_i.len()];
    let mut support_rows = Vec::with_capacity(n_samples);
    for anchor in 0..n_markers.saturating_sub(1) {
        support_rows.clear();
        for row in 0..n_samples {
            if x[row * n_markers + anchor] == 1.0 {
                support_rows.push(row);
            }
        }
        for marker in (anchor + 1)..n_markers {
            let mut value = 0.0;
            for &row in &support_rows {
                value += py[row] * x[row * n_markers + marker];
            }
            let pair = anchor * n_markers - anchor * (anchor + 1) / 2 + (marker - anchor - 1);
            interaction_py[pair] = value;
        }
    }
    Ok(interaction_py)
}

pub(crate) fn phase1_scores(
    geometry: &Phase1Geometry,
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: &[f64],
    target: Option<(usize, usize)>,
) -> Result<Phase1Scores, String> {
    Ok(phase1_scores_timed(geometry, x, n_samples, n_markers, p_c, y, target)?.0)
}

pub(crate) fn phase1_scores_timed(
    geometry: &Phase1Geometry,
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: &[f64],
    target: Option<(usize, usize)>,
) -> Result<(Phase1Scores, Phase1ScoreTiming), String> {
    phase1_scores_timed_with_c_kernel(
        geometry,
        x,
        n_samples,
        n_markers,
        p_c,
        y,
        target,
        WeightedCKernel::DenseReference,
    )
}

pub(crate) fn phase1_scores_timed_with_c_kernel(
    geometry: &Phase1Geometry,
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: &[f64],
    target: Option<(usize, usize)>,
    c_kernel: WeightedCKernel,
) -> Result<(Phase1Scores, Phase1ScoreTiming), String> {
    phase1_scores_timed_with_c_kernel_and_dosage(
        geometry,
        x,
        n_samples,
        n_markers,
        p_c,
        y,
        target,
        c_kernel,
        DosageChannel::FwlProduct,
    )
}

pub(crate) fn phase1_scores_timed_with_c_kernel_and_dosage(
    geometry: &Phase1Geometry,
    x: &[f64],
    n_samples: usize,
    n_markers: usize,
    p_c: &[f64],
    y: &[f64],
    target: Option<(usize, usize)>,
    c_kernel: WeightedCKernel,
    dosage_channel: DosageChannel,
) -> Result<(Phase1Scores, Phase1ScoreTiming), String> {
    validate_inputs(x, n_samples, n_markers, p_c, Some(y))?;
    if geometry.pair_i.len() != geometry.q.len()
        || geometry.pair_j.len() != geometry.q.len()
        || geometry.a.len() != n_markers * n_markers
        || geometry.h.len() != n_markers * n_markers
    {
        return Err("geometry dimensions do not match the genotype window".to_string());
    }
    let py_started = Instant::now();
    let py = matvec_row_major(p_c, n_samples, y);
    let py_seconds = py_started.elapsed().as_secs_f64();
    let py_sum = py.iter().sum::<f64>();

    let b_started = Instant::now();
    let mut marker_py = vec![0.0; n_markers];
    let mut marker_mean = vec![0.0; n_markers];
    for marker in 0..n_markers {
        let mut value = 0.0;
        let mut mean_value = 0.0;
        for row in 0..n_samples {
            value += x[row * n_markers + marker] * py[row];
            mean_value += x[row * n_markers + marker];
        }
        marker_py[marker] = value;
        marker_mean[marker] = mean_value / n_samples as f64;
    }
    let b_seconds = b_started.elapsed().as_secs_f64();

    let weighted_gram_started = Instant::now();
    let interaction_py = match c_kernel {
        WeightedCKernel::DenseReference => weighted_interaction_py(
            x,
            &py,
            n_samples,
            n_markers,
            &geometry.pair_i,
            &geometry.pair_j,
        ),
        WeightedCKernel::BinarySparse => weighted_interaction_py_binary_sparse(
            x,
            &py,
            n_samples,
            n_markers,
            &geometry.pair_i,
            &geometry.pair_j,
        )?,
        WeightedCKernel::MinorSupport => weighted_interaction_py_minor_support(
            x,
            &py,
            n_samples,
            n_markers,
            &geometry.pair_i,
            &geometry.pair_j,
        )?,
        WeightedCKernel::SampleRowSparse => {
            let packed = sample_row_sparse_weighted_c(x, &py, &marker_py, n_samples, n_markers)?;
            if geometry.pair_i.len() != geometry.pair_j.len() {
                return Err("sample-row sparse C pair index lengths differ".to_string());
            }
            geometry
                .pair_i
                .iter()
                .zip(geometry.pair_j.iter())
                .map(|(&i, &j)| {
                    if i >= j || j >= n_markers {
                        return Err("sample-row sparse C requires canonical pair order".to_string());
                    }
                    Ok(packed.c_upper[packed_upper_index(n_markers, i, j)])
                })
                .collect::<Result<Vec<_>, String>>()?
        }
    };
    let weighted_gram_seconds = weighted_gram_started.elapsed().as_secs_f64();

    let score_allocation_started = Instant::now();
    let mut logic = vec![[f64::NEG_INFINITY; 5]; geometry.q.len()];
    let mut dosage = vec![f64::NEG_INFINITY; geometry.q.len()];
    let mut max6 = vec![f64::NEG_INFINITY; geometry.q.len()];
    let score_allocation_seconds = score_allocation_started.elapsed().as_secs_f64();

    let dosage_score_started = Instant::now();
    for pair in 0..geometry.q.len() {
        let i = geometry.pair_i[pair];
        let j = geometry.pair_j[pair];
        let b0 = marker_py[i];
        let b1 = marker_py[j];
        let a00 = geometry.a[i * n_markers + i];
        let a11 = geometry.a[j * n_markers + j];
        let a01 = geometry.a[i * n_markers + j];
        let u0 = geometry.h[j * n_markers + i];
        let u1 = geometry.h[i * n_markers + j];
        let q = geometry.q[pair];
        let c = interaction_py[pair];

        dosage[pair] = dosage_channel_score(
            dosage_channel,
            b0,
            b1,
            a00,
            a11,
            a01,
            u0,
            u1,
            q,
            c,
            marker_mean[i],
            marker_mean[j],
            py_sum,
        );
    }
    let dosage_score_seconds = dosage_score_started.elapsed().as_secs_f64();

    let logic_score_started = Instant::now();
    let mut logic_best = vec![f64::NEG_INFINITY; geometry.q.len()];
    for pair in 0..geometry.q.len() {
        let i = geometry.pair_i[pair];
        let j = geometry.pair_j[pair];
        let b0 = marker_py[i];
        let b1 = marker_py[j];
        let a00 = geometry.a[i * n_markers + i];
        let a11 = geometry.a[j * n_markers + j];
        let a01 = geometry.a[i * n_markers + j];
        let u0 = geometry.h[j * n_markers + i];
        let u1 = geometry.h[i * n_markers + j];
        let q = geometry.q[pair];
        let c = interaction_py[pair];

        let numerators = [-b0 - b1 + c, b1 - c, b0 - c, c, b0 + b1 - 2.0 * c];
        let denominators = [
            a00 + a11 + q + 2.0 * a01 - 2.0 * u0 - 2.0 * u1,
            a11 - 2.0 * u1 + q,
            a00 - 2.0 * u0 + q,
            q,
            a00 + a11 + 4.0 * q + 2.0 * a01 - 4.0 * u0 - 4.0 * u1,
        ];
        let mut best_logic = f64::NEG_INFINITY;
        for gate in 0..5 {
            let value = score_ratio(numerators[gate], denominators[gate]);
            logic[pair][gate] = value;
            if value > best_logic {
                best_logic = value;
            }
        }
        logic_best[pair] = best_logic;
    }
    let logic_score_seconds = logic_score_started.elapsed().as_secs_f64();

    let max6_reduce_started = Instant::now();
    for pair in 0..geometry.q.len() {
        max6[pair] = logic_best[pair].max(dosage[pair]);
    }
    let max6_reduce_seconds = max6_reduce_started.elapsed().as_secs_f64();

    let target_rank_started = Instant::now();
    let (target_index, max6_rank) = if let Some((raw_i, raw_j)) = target {
        let (target_i, target_j) = if raw_i <= raw_j {
            (raw_i, raw_j)
        } else {
            (raw_j, raw_i)
        };
        let index = geometry
            .pair_i
            .iter()
            .zip(geometry.pair_j.iter())
            .position(|(&i, &j)| i == target_i && j == target_j);
        let rank = index.and_then(|index| {
            let value = max6[index];
            value
                .is_finite()
                .then(|| 1 + max6.iter().filter(|&&other| other > value).count())
        });
        (index, rank)
    } else {
        (None, None)
    };
    let target_rank_seconds = target_rank_started.elapsed().as_secs_f64();
    Ok((
        Phase1Scores {
            logic,
            dosage,
            max6,
            target_index,
            max6_rank,
        },
        Phase1ScoreTiming {
            py_seconds,
            b_seconds,
            weighted_gram_seconds,
            score_allocation_seconds,
            dosage_score_seconds,
            logic_score_seconds,
            max6_reduce_seconds,
            target_rank_seconds,
        },
    ))
}

fn array2_from_vec<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    values: Vec<f64>,
    label: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let array = Array2::from_shape_vec((rows, cols), values).map_err(|error| {
        PyRuntimeError::new_err(format!("{label}: cannot build output matrix: {error}"))
    })?;
    Ok(PyArray2::from_owned_array(py, array).into_bound())
}

/// Exact GRM-Max6 rerank for an explicit candidate edge list.
///
/// Unlike the full-window API, this function never constructs a candidate
/// subset `A`, `H`, or `C` matrix. It computes `P_C X_U` once for the unique
/// candidate endpoints and then evaluates only the requested candidate edges.
/// The q and C calculations remain exact `mincell_exact` and candidate-only
/// sample-row sparse calculations respectively. Passing
/// `dosage_channel="centered_product"` selects the EXPERIMENTAL GRM-nMax6
/// variant; production GRM-Max6 uses the default `fwl_product` channel.
#[pyfunction(name = "garfield_grm_max6_rerank_candidates")]
#[pyo3(signature = (x, p_c, y, candidate_first, candidate_second, top_k=500, dosage_channel=None))]
pub fn garfield_grm_max6_rerank_candidates_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    p_c: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    candidate_first: PyReadonlyArray1<'py, i64>,
    candidate_second: PyReadonlyArray1<'py, i64>,
    top_k: usize,
    dosage_channel: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_garfield_blas_backend().map_err(PyRuntimeError::new_err)?;
    let dosage_channel =
        resolve_dosage_channel(dosage_channel.as_deref()).map_err(PyValueError::new_err)?;
    let started = Instant::now();
    let x_shape = x.shape();
    if x_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "garfield_grm_max6_rerank_candidates: x must be two-dimensional",
        ));
    }
    let n_samples = x_shape[0];
    let n_markers = x_shape[1];
    let p_c_shape = p_c.shape();
    if p_c_shape.len() != 2 || p_c_shape[0] != n_samples || p_c_shape[1] != n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_grm_max6_rerank_candidates: p_c shape {:?} must be ({n_samples}, {n_samples})",
            p_c_shape
        )));
    }
    if candidate_first.shape().len() != 1 || candidate_second.shape().len() != 1 {
        return Err(PyValueError::new_err(
            "garfield_grm_max6_rerank_candidates: candidate indices must be one-dimensional",
        ));
    }

    let input_copy_started = Instant::now();
    let x_vec = x.as_array().iter().copied().collect::<Vec<_>>();
    let p_c_vec = p_c.as_array().iter().copied().collect::<Vec<_>>();
    let y_vec = y.as_array().iter().copied().collect::<Vec<_>>();
    let candidate_first_vec = candidate_first
        .as_array()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let candidate_second_vec = candidate_second
        .as_array()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let input_copy_seconds = input_copy_started.elapsed().as_secs_f64();
    let result = candidate_rerank_exact(
        &x_vec,
        n_samples,
        n_markers,
        &p_c_vec,
        &y_vec,
        &candidate_first_vec,
        &candidate_second_vec,
        top_k,
        dosage_channel,
    )
    .map_err(PyValueError::new_err)?;

    let output_started = Instant::now();
    let out = PyDict::new(py);
    out.set_item(
        "candidate_pair_i",
        PyArray1::from_owned_array(
            py,
            Array1::from_vec(result.pair_i.iter().map(|&value| value as i64).collect()),
        )
        .into_bound(),
    )?;
    out.set_item(
        "candidate_pair_j",
        PyArray1::from_owned_array(
            py,
            Array1::from_vec(result.pair_j.iter().map(|&value| value as i64).collect()),
        )
        .into_bound(),
    )?;
    out.set_item(
        "input_to_unique",
        PyArray1::from_owned_array(py, Array1::from_vec(result.input_to_unique.clone()))
            .into_bound(),
    )?;
    out.set_item(
        "unique_marker_indices",
        PyArray1::from_owned_array(
            py,
            Array1::from_vec(
                result
                    .unique_markers
                    .iter()
                    .map(|&value| value as i64)
                    .collect(),
            ),
        )
        .into_bound(),
    )?;
    out.set_item(
        "z_unique",
        array2_from_vec(
            py,
            n_samples,
            result.unique_markers.len(),
            result.z,
            "z_unique",
        )?,
    )?;
    out.set_item(
        "q",
        PyArray1::from_owned_array(py, Array1::from_vec(result.q)).into_bound(),
    )?;
    out.set_item(
        "mincell_q",
        PyArray1::from_owned_array(py, Array1::from_vec(result.mincell_q)).into_bound(),
    )?;
    out.set_item(
        "mincell_cell",
        PyArray1::from_owned_array(py, Array1::from_vec(result.mincell_cell)).into_bound(),
    )?;
    out.set_item(
        "mincell_support",
        PyArray1::from_owned_array(
            py,
            Array1::from_vec(
                result
                    .mincell_support
                    .iter()
                    .map(|&value| value as i64)
                    .collect(),
            ),
        )
        .into_bound(),
    )?;
    let mut logic_flat = Vec::with_capacity(result.logic.len() * 5);
    for values in &result.logic {
        logic_flat.extend(values);
    }
    out.set_item(
        "logic_scores",
        array2_from_vec(py, result.logic.len(), 5, logic_flat, "logic_scores")?,
    )?;
    out.set_item(
        "dosage_scores",
        PyArray1::from_owned_array(py, Array1::from_vec(result.dosage)).into_bound(),
    )?;
    out.set_item(
        "max6_scores",
        PyArray1::from_owned_array(py, Array1::from_vec(result.max6.clone())).into_bound(),
    )?;
    out.set_item(
        "best_logic_gate",
        PyArray1::from_owned_array(py, Array1::from_vec(result.best_logic_gate)).into_bound(),
    )?;
    out.set_item("best_max6_channel", result.best_max6_channel)?;
    out.set_item(
        "candidate_rank",
        PyArray1::from_owned_array(
            py,
            Array1::from_vec(
                result
                    .candidate_rank
                    .iter()
                    .map(|&value| value as i64)
                    .collect(),
            ),
        )
        .into_bound(),
    )?;
    out.set_item(
        "global_pair_id",
        PyArray1::from_owned_array(
            py,
            Array1::from_vec(
                result
                    .global_pair_id
                    .iter()
                    .map(|&value| value as i64)
                    .collect(),
            ),
        )
        .into_bound(),
    )?;
    let top_pair_i = result
        .top_order
        .iter()
        .map(|&index| result.pair_i[index] as i64)
        .collect::<Vec<_>>();
    let top_pair_j = result
        .top_order
        .iter()
        .map(|&index| result.pair_j[index] as i64)
        .collect::<Vec<_>>();
    let top_max6 = result
        .top_order
        .iter()
        .map(|&index| result.max6[index])
        .collect::<Vec<_>>();
    out.set_item(
        "top_pair_i",
        PyArray1::from_owned_array(py, Array1::from_vec(top_pair_i)).into_bound(),
    )?;
    out.set_item(
        "top_pair_j",
        PyArray1::from_owned_array(py, Array1::from_vec(top_pair_j)).into_bound(),
    )?;
    out.set_item(
        "top_max6",
        PyArray1::from_owned_array(py, Array1::from_vec(top_max6)).into_bound(),
    )?;
    out.set_item("candidate_pairs", result.pair_i.len())?;
    out.set_item("input_pairs", candidate_first_vec.len())?;
    out.set_item("input_self_count", result.input_self_count)?;
    out.set_item("input_duplicate_count", result.input_duplicate_count)?;
    out.set_item("top_k", top_k.min(result.pair_i.len()))?;
    out.set_item(
        "min_support_flipped",
        PyArray1::from_owned_array(
            py,
            Array1::from_vec(result.flipped.iter().map(|&value| value as i8).collect()),
        )
        .into_bound(),
    )?;

    let output_seconds = output_started.elapsed().as_secs_f64();
    let total_seconds = started.elapsed().as_secs_f64();
    let timing = PyDict::new(py);
    timing.set_item("input_copy_seconds", input_copy_seconds)?;
    timing.set_item("py_seconds", result.timing.py_seconds)?;
    timing.set_item("pcx_u_seconds", result.timing.pcx_u_seconds)?;
    timing.set_item(
        "candidate_geometry_seconds",
        result.timing.candidate_geometry_seconds,
    )?;
    timing.set_item("q_seconds", result.timing.q_seconds)?;
    timing.set_item("score_seconds", result.timing.score_seconds)?;
    timing.set_item("topk_seconds", result.timing.topk_seconds)?;
    timing.set_item("kernel_seconds", result.timing.kernel_seconds)?;
    timing.set_item("output_seconds", output_seconds)?;
    timing.set_item("total_seconds", total_seconds)?;
    out.set_item("timing", timing)?;
    out.set_item("q_kernel", "mincell_exact")?;
    out.set_item("c_kernel", "sample_row_sparse_candidate")?;
    out.set_item("dosage_channel", dosage_channel.name())?;
    out.set_item("rank_scope", "candidate")?;
    Ok(out)
}

/// Full exact GRM-Max6 scan with Phase-1 shared marker geometry.
///
/// The input `p_c` is the already materialized symmetric intercept-projected
/// GRM metric. This phase computes `Z=P_C X`, `A=X'Z`, and
/// `H=X'(X\odot Z)` once per window. The default uses the exact
/// `mincell_exact` q kernel and the exact minor-support weighted-Gram C
/// kernel. Supplying complete spectral arguments without an explicit
/// `q_kernel` preserves the spectral q path and dense C reference for
/// compatibility. The explicit
/// `q_kernel="subset"` and `q_kernel="subset_support_sorted"` modes use the
/// exact binary-support subset kernel and preserve the original pair order.
/// `q_kernel="subset_min_support"` additionally complements columns whose
/// support exceeds half the sample count before running the sorted subset
/// kernel; its logic-gate output is mapped back to the original encoding.
/// The former `q_kernel="subset_minor_support"` spelling remains accepted as
/// a compatibility alias. The optional `c_kernel="binary_sparse_C"` selects
/// the earlier exact row-support weighted-Gram path; `c_kernel="sample_row_sparse"`
/// selects the packed-upper-triangle sample-row kernel, which is the default
/// C path for the minor-coded exact q kernels. Both are single-threaded paths
/// and neither changes score assembly. `c_kernel="minor_support_weighted_gram"`
/// selects the exact anchor-support weighted-Gram reference kernel and requires
/// a minor-coded q path so that the C and q calculations use the same genotype.
/// `dense_reference` remains available as an explicit reference path.
/// `q_kernel="mincell_exact"` is the default exact
/// computational path: it selects the sparsest four-cell support per pair,
/// evaluates that cell's quadratic form by set-bit enumeration, reconstructs
/// the original `q_11`, and then reuses the same score assembly as
/// `subset_min_support`. It is not an approximate scanner and does not alter
/// the production rank definition.
#[pyfunction(name = "garfield_grm_max6_full_window")]
#[pyo3(signature = (
    x,
    p_c,
    y,
    top_k=500,
    block_pairs=256,
    target_first=None,
    target_second=None,
    q_eigenvalues=None,
    q_eigenvectors=None,
    q_lambda=None,
    q_kernel=None,
    c_kernel=None,
    dosage_channel=None
))]
pub fn garfield_grm_max6_full_window_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    p_c: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    top_k: usize,
    block_pairs: usize,
    target_first: Option<usize>,
    target_second: Option<usize>,
    q_eigenvalues: Option<PyReadonlyArray1<'py, f64>>,
    q_eigenvectors: Option<PyReadonlyArray2<'py, f64>>,
    q_lambda: Option<f64>,
    q_kernel: Option<String>,
    c_kernel: Option<String>,
    dosage_channel: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_garfield_blas_backend().map_err(PyRuntimeError::new_err)?;
    let dosage_channel =
        resolve_dosage_channel(dosage_channel.as_deref()).map_err(PyValueError::new_err)?;
    if block_pairs == 0 {
        return Err(PyValueError::new_err(
            "garfield_grm_max6_full_window: block_pairs must be > 0",
        ));
    }
    let x_shape = x.shape();
    if x_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "garfield_grm_max6_full_window: x must be two-dimensional",
        ));
    }
    let n_samples = x_shape[0];
    let n_markers = x_shape[1];
    let p_c_shape = p_c.shape();
    if p_c_shape.len() != 2 || p_c_shape[0] != n_samples || p_c_shape[1] != n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_grm_max6_full_window: p_c shape {:?} must be ({n_samples}, {n_samples})",
            p_c_shape
        )));
    }
    let target = match (target_first, target_second) {
        (Some(first), Some(second)) => Some((first, second)),
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "garfield_grm_max6_full_window: target_first and target_second must be supplied together",
            ))
        }
    };
    if let Some((first, second)) = target {
        if first >= n_markers || second >= n_markers || first == second {
            return Err(PyValueError::new_err(format!(
                "garfield_grm_max6_full_window: target pair ({first}, {second}) is outside marker range"
            )));
        }
    }

    let started = Instant::now();
    let input_copy_started = Instant::now();
    let x_vec = x.as_array().iter().copied().collect::<Vec<_>>();
    let p_c_vec = p_c.as_array().iter().copied().collect::<Vec<_>>();
    let y_vec = y.as_array().iter().copied().collect::<Vec<_>>();
    let input_copy_seconds = input_copy_started.elapsed().as_secs_f64();
    let any_spectral_arg =
        q_eigenvalues.is_some() || q_eigenvectors.is_some() || q_lambda.is_some();
    let (requested_q_kernel, requested_c_kernel_name) =
        resolve_full_exact_kernel_names(q_kernel.as_deref(), c_kernel.as_deref(), any_spectral_arg);
    let c_kernel_name =
        canonical_weighted_c_kernel_name(requested_c_kernel_name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "garfield_grm_max6_full_window: unknown c_kernel {requested_c_kernel_name:?}"
            ))
        })?;
    let c_kernel = match c_kernel_name {
        "dense_reference" => WeightedCKernel::DenseReference,
        "binary_sparse_C" => WeightedCKernel::BinarySparse,
        "minor_support_weighted_gram" => WeightedCKernel::MinorSupport,
        "sample_row_sparse" => WeightedCKernel::SampleRowSparse,
        _ => unreachable!("canonical weighted C kernel name must be recognized"),
    };
    if matches!(c_kernel, WeightedCKernel::MinorSupport)
        && !matches!(
            requested_q_kernel,
            "subset_min_support" | "subset_minor_support" | "mincell_exact"
        )
    {
        return Err(PyValueError::new_err(
            "minor_support_weighted_gram requires a minor-coded q kernel",
        ));
    }
    let canonical_requested_q_kernel = canonical_q_kernel_name(requested_q_kernel);
    let all_spectral_args =
        q_eigenvalues.is_some() && q_eigenvectors.is_some() && q_lambda.is_some();
    if (matches!(
        requested_q_kernel,
        "subset"
            | "subset_support_sorted"
            | "subset_min_support"
            | "subset_minor_support"
            | "mincell_exact"
    ) || canonical_requested_q_kernel.is_some())
        && any_spectral_arg
    {
        return Err(PyValueError::new_err(
            "garfield_grm_max6_full_window: subset q kernels cannot receive spectral arguments",
        ));
    }
    if requested_q_kernel == "old" && any_spectral_arg {
        return Err(PyValueError::new_err(
            "garfield_grm_max6_full_window: old q kernel cannot receive spectral arguments",
        ));
    }
    if requested_q_kernel == "spectral" && !all_spectral_args {
        return Err(PyValueError::new_err(
            "garfield_grm_max6_full_window: spectral q kernel requires all spectral arguments",
        ));
    }
    if !matches!(
        requested_q_kernel,
        "old"
            | "spectral"
            | "subset"
            | "subset_support_sorted"
            | "subset_min_support"
            | "subset_minor_support"
            | "mincell_exact"
    ) {
        return Err(PyValueError::new_err(format!(
            "garfield_grm_max6_full_window: unknown q_kernel {requested_q_kernel:?}"
        )));
    }

    let mut score_x = x_vec.clone();
    let mut min_support_flipped = None;
    let mut min_support_seconds = 0.0;
    let (geometry, timing, q_kernel_name) = match requested_q_kernel {
        "subset" | "subset_support_sorted" => {
            let support_sorted = requested_q_kernel == "subset_support_sorted";
            let (geometry, timing) = phase3_geometry_timed(
                &x_vec,
                n_samples,
                n_markers,
                &p_c_vec,
                block_pairs,
                support_sorted,
            )
            .map_err(PyValueError::new_err)?;
            let name = if support_sorted {
                "subset_support_sorted"
            } else {
                "subset_support_original"
            };
            (geometry, timing, name)
        }
        "subset_min_support" | "subset_minor_support" => {
            let min_support_started = Instant::now();
            let encoding =
                min_support_encode(&x_vec, n_samples, n_markers).map_err(PyValueError::new_err)?;
            min_support_seconds = min_support_started.elapsed().as_secs_f64();
            let (geometry, timing) = phase3_geometry_timed(
                &encoding.x,
                n_samples,
                n_markers,
                &p_c_vec,
                block_pairs,
                true,
            )
            .map_err(PyValueError::new_err)?;
            score_x = encoding.x;
            min_support_flipped = Some(encoding.flipped);
            (geometry, timing, "subset_min_support")
        }
        "mincell_exact" => {
            let min_support_started = Instant::now();
            let encoding =
                min_support_encode(&x_vec, n_samples, n_markers).map_err(PyValueError::new_err)?;
            min_support_seconds = min_support_started.elapsed().as_secs_f64();
            let (geometry, timing) = phase6b_geometry_timed(
                &encoding.x,
                n_samples,
                n_markers,
                &p_c_vec,
                block_pairs,
            )
            .map_err(PyValueError::new_err)?;
            score_x = encoding.x;
            min_support_flipped = Some(encoding.flipped);
            (geometry, timing, "mincell_exact")
        }
        _ => match (q_eigenvalues, q_eigenvectors, q_lambda) {
            (None, None, None) => {
                let (geometry, timing) = phase1_geometry_timed(
                    &x_vec,
                    n_samples,
                    n_markers,
                    &p_c_vec,
                    block_pairs,
                )
                .map_err(PyValueError::new_err)?;
                (geometry, timing, "old_blocked_pc_times_w")
            }
            (Some(eigenvalues), Some(eigenvectors), Some(lambda_vc)) => {
                let eigenvalue_shape = eigenvalues.shape();
                let eigenvector_shape = eigenvectors.shape();
                if eigenvalue_shape.len() != 1 || eigenvalue_shape[0] != n_samples {
                    return Err(PyValueError::new_err(format!(
                        "garfield_grm_max6_full_window: q_eigenvalues shape {:?} must be ({n_samples},)",
                        eigenvalue_shape
                    )));
                }
                if eigenvector_shape.len() != 2
                    || eigenvector_shape[0] != n_samples
                    || eigenvector_shape[1] != n_samples
                {
                    return Err(PyValueError::new_err(format!(
                        "garfield_grm_max6_full_window: q_eigenvectors shape {:?} must be ({n_samples}, {n_samples})",
                        eigenvector_shape
                    )));
                }
                let eigenvalues_vec = eigenvalues.as_array().iter().copied().collect::<Vec<_>>();
                let eigenvectors_vec = eigenvectors
                    .as_array()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();
                let (geometry, timing) = phase2_geometry_timed(
                    &x_vec,
                    n_samples,
                    n_markers,
                    &p_c_vec,
                    &eigenvalues_vec,
                    &eigenvectors_vec,
                    lambda_vc,
                    block_pairs,
                )
                .map_err(PyValueError::new_err)?;
                (geometry, timing, "spectral_blocked_u_transpose_w")
            }
            _ => {
                return Err(PyValueError::new_err(
                    "garfield_grm_max6_full_window: q_eigenvalues, q_eigenvectors, and q_lambda must be supplied together",
                ))
            }
        },
    };
    let score_started = Instant::now();
    let (scores, score_timing) = phase1_scores_timed_with_c_kernel_and_dosage(
        &geometry,
        &score_x,
        n_samples,
        n_markers,
        &p_c_vec,
        &y_vec,
        target,
        c_kernel,
        dosage_channel,
    )
    .map_err(PyValueError::new_err)?;
    let score_seconds = score_started.elapsed().as_secs_f64();

    let rank_started = Instant::now();
    let mut order = (0..scores.max6.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        scores.max6[right]
            .total_cmp(&scores.max6[left])
            .then_with(|| left.cmp(&right))
    });
    let keep = top_k.min(order.len());
    let top_pair_i = order[..keep]
        .iter()
        .map(|&index| geometry.pair_i[index] as i64)
        .collect::<Vec<_>>();
    let top_pair_j = order[..keep]
        .iter()
        .map(|&index| geometry.pair_j[index] as i64)
        .collect::<Vec<_>>();
    let top_max6 = order[..keep]
        .iter()
        .map(|&index| scores.max6[index])
        .collect::<Vec<_>>();
    let rank_seconds = rank_started.elapsed().as_secs_f64();
    let n_pairs = geometry.q.len();
    let logic_output_started = Instant::now();
    let mut logic_flat = Vec::with_capacity(scores.logic.len() * 5);
    let mut best_logic_gate = Vec::with_capacity(scores.logic.len());
    let mut best_max6_channel = Vec::with_capacity(scores.logic.len());
    for (pair, values) in scores.logic.iter().enumerate() {
        let (flip_i, flip_j) = min_support_flipped
            .as_ref()
            .map(|flips| (flips[geometry.pair_i[pair]], flips[geometry.pair_j[pair]]))
            .unwrap_or((false, false));
        let mut mapped = [f64::NEG_INFINITY; 5];
        for (gate, &value) in values.iter().enumerate() {
            mapped[remap_logic_gate_index(gate, flip_i, flip_j)] = value;
        }
        best_logic_gate.push(best_logic_gate_index(&mapped));
        best_max6_channel.push(best_max6_channel_name(&mapped, scores.dosage[pair]));
        logic_flat.extend(mapped);
    }
    let logic_output_seconds = logic_output_started.elapsed().as_secs_f64();

    let output_started = Instant::now();
    let out = PyDict::new(py);
    out.set_item(
        "pair_i",
        PyArray1::from_owned_array(
            py,
            Array1::from_vec(geometry.pair_i.iter().map(|&value| value as i64).collect()),
        )
        .into_bound(),
    )?;
    out.set_item(
        "pair_j",
        PyArray1::from_owned_array(
            py,
            Array1::from_vec(geometry.pair_j.iter().map(|&value| value as i64).collect()),
        )
        .into_bound(),
    )?;
    out.set_item(
        "z",
        array2_from_vec(py, n_samples, n_markers, geometry.z, "z")?,
    )?;
    out.set_item(
        "a",
        array2_from_vec(py, n_markers, n_markers, geometry.a, "a")?,
    )?;
    out.set_item(
        "h",
        array2_from_vec(py, n_markers, n_markers, geometry.h, "h")?,
    )?;
    out.set_item(
        "q",
        PyArray1::from_owned_array(py, Array1::from_vec(geometry.q)).into_bound(),
    )?;
    if let Some(mincell_q) = geometry.mincell_q {
        out.set_item(
            "mincell_q",
            PyArray1::from_owned_array(py, Array1::from_vec(mincell_q)).into_bound(),
        )?;
    }
    if let Some(mincell_cell) = geometry.mincell_cell {
        out.set_item(
            "mincell_cell",
            PyArray1::from_owned_array(py, Array1::from_vec(mincell_cell)).into_bound(),
        )?;
    }
    if let Some(mincell_support) = geometry.mincell_support {
        out.set_item(
            "mincell_support",
            PyArray1::from_owned_array(
                py,
                Array1::from_vec(
                    mincell_support
                        .into_iter()
                        .map(|value| value as i64)
                        .collect(),
                ),
            )
            .into_bound(),
        )?;
    }
    out.set_item(
        "logic_scores",
        array2_from_vec(py, scores.max6.len(), 5, logic_flat, "logic_scores")?,
    )?;
    out.set_item(
        "dosage_scores",
        PyArray1::from_owned_array(py, Array1::from_vec(scores.dosage)).into_bound(),
    )?;
    out.set_item(
        "max6_scores",
        PyArray1::from_owned_array(py, Array1::from_vec(scores.max6)).into_bound(),
    )?;
    out.set_item(
        "best_logic_gate",
        PyArray1::from_owned_array(py, Array1::from_vec(best_logic_gate)).into_bound(),
    )?;
    out.set_item("best_max6_channel", best_max6_channel)?;
    let flipped_output = min_support_flipped
        .unwrap_or_else(|| vec![false; n_markers])
        .into_iter()
        .map(|value| value as i8)
        .collect::<Vec<_>>();
    out.set_item(
        "min_support_flipped",
        PyArray1::from_owned_array(py, Array1::from_vec(flipped_output)).into_bound(),
    )?;
    out.set_item(
        "top_pair_i",
        PyArray1::from_owned_array(py, Array1::from_vec(top_pair_i)).into_bound(),
    )?;
    out.set_item(
        "top_pair_j",
        PyArray1::from_owned_array(py, Array1::from_vec(top_pair_j)).into_bound(),
    )?;
    out.set_item(
        "top_max6",
        PyArray1::from_owned_array(py, Array1::from_vec(top_max6)).into_bound(),
    )?;
    out.set_item("pairs", n_pairs)?;
    out.set_item(
        "target_index",
        scores.target_index.map(|value| value as i64),
    )?;
    out.set_item("target_rank", scores.max6_rank.map(|value| value as i64))?;

    let output_seconds = output_started.elapsed().as_secs_f64();
    let total_seconds = started.elapsed().as_secs_f64();
    let accounted_seconds = input_copy_seconds
        + timing.geometry_seconds
        + min_support_seconds
        + score_seconds
        + rank_seconds
        + logic_output_seconds
        + output_seconds;
    let misc_seconds = (total_seconds - accounted_seconds).max(0.0);
    let timing_out = PyDict::new(py);
    timing_out.set_item("input_copy_seconds", input_copy_seconds)?;
    timing_out.set_item("min_support_seconds", min_support_seconds)?;
    timing_out.set_item("pcx_seconds", timing.pcx_seconds)?;
    timing_out.set_item("a_seconds", timing.a_seconds)?;
    timing_out.set_item("xz_build_seconds", timing.xz_build_seconds)?;
    timing_out.set_item("h_gram_seconds", timing.h_gram_seconds)?;
    timing_out.set_item("h_seconds", timing.h_seconds)?;
    timing_out.set_item("q_seconds", timing.q_seconds)?;
    timing_out.set_item("geometry_seconds", timing.geometry_seconds)?;
    if let Some(work) = timing.subset_work {
        timing_out.set_item("subset_work", work.work_subset)?;
        timing_out.set_item("dense_work", work.work_dense)?;
        timing_out.set_item("work_ratio", work.work_ratio)?;
    }
    if let Some(q_timing) = timing.subset_q_timing {
        let q_stage_sum_seconds = q_timing.sum();
        timing_out.set_item("q_support_seconds", q_timing.support_seconds)?;
        timing_out.set_item("q_psub_seconds", q_timing.psub_seconds)?;
        timing_out.set_item("q_xsub_seconds", q_timing.xsub_seconds)?;
        timing_out.set_item("q_gemm_seconds", q_timing.gemm_seconds)?;
        timing_out.set_item("q_reduce_seconds", q_timing.reduce_seconds)?;
        timing_out.set_item("q_alloc_zero_seconds", q_timing.alloc_zero_seconds)?;
        timing_out.set_item("q_workspace_allocations", q_timing.workspace_allocations)?;
        timing_out.set_item("q_stage_sum_seconds", q_stage_sum_seconds)?;
        timing_out.set_item(
            "q_unattributed_seconds",
            (timing.q_seconds - q_stage_sum_seconds).max(0.0),
        )?;
    }
    if let Some(work) = timing.mincell_work {
        timing_out.set_item("mincell_work", work.work_mincell)?;
        timing_out.set_item("mincell_subset_work", work.work_subset)?;
        timing_out.set_item("mincell_dense_work", work.work_dense)?;
        timing_out.set_item("mincell_subset_work_ratio", work.mincell_subset_ratio)?;
    }
    if let Some(q_timing) = timing.mincell_q_timing {
        timing_out.set_item("mincell_q_bitset_seconds", q_timing.bitset_seconds)?;
        timing_out.set_item("mincell_q_cell_seconds", q_timing.cell_seconds)?;
        // Keep the Phase 7B field as a compatibility alias for cell selection.
        timing_out.set_item("mincell_q_pair_seconds", q_timing.cell_seconds)?;
        timing_out.set_item("mincell_q_enumerate_seconds", q_timing.enumerate_seconds)?;
        timing_out.set_item("mincell_q_quadratic_seconds", q_timing.quadratic_seconds)?;
        timing_out.set_item(
            "mincell_q_reconstruct_seconds",
            q_timing.reconstruct_seconds,
        )?;
        timing_out.set_item("mincell_q_stage_sum_seconds", q_timing.sum())?;
    }
    timing_out.set_item("py_seconds", score_timing.py_seconds)?;
    timing_out.set_item("b_seconds", score_timing.b_seconds)?;
    timing_out.set_item("weighted_gram_seconds", score_timing.weighted_gram_seconds)?;
    timing_out.set_item("c_kernel", c_kernel_name)?;
    timing_out.set_item("dosage_channel", dosage_channel.name())?;
    timing_out.set_item(
        "score_allocation_seconds",
        score_timing.score_allocation_seconds,
    )?;
    timing_out.set_item("score_dosage_seconds", score_timing.dosage_score_seconds)?;
    timing_out.set_item("score_logic_seconds", score_timing.logic_score_seconds)?;
    timing_out.set_item("max6_reduce_seconds", score_timing.max6_reduce_seconds)?;
    timing_out.set_item("target_rank_seconds", score_timing.target_rank_seconds)?;
    timing_out.set_item("finite_checks_fused_into_score", true)?;
    timing_out.set_item("score_seconds", score_seconds)?;
    timing_out.set_item("rank_topk_seconds", rank_seconds)?;
    timing_out.set_item("logic_output_seconds", logic_output_seconds)?;
    timing_out.set_item("output_seconds", output_seconds)?;
    timing_out.set_item("misc_seconds", misc_seconds)?;
    timing_out.set_item("total_seconds", total_seconds)?;
    out.set_item("timing", timing_out)?;
    out.set_item("q_kernel", q_kernel_name)?;
    out.set_item("c_kernel", c_kernel_name)?;
    out.set_item("dosage_channel", dosage_channel.name())?;
    out.set_item("geometry_kernel", "phase1_ah")?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::{
        best_max6_channel_name, candidate_au_workspace_bytes, candidate_rerank_exact,
        candidate_rerank_exact_packed_no_grm_covariate_with_py_trusted_scaled,
        candidate_rerank_exact_packed_with_py_trusted,
        candidate_rerank_exact_packed_with_py_trusted_scaled, candidate_rerank_exact_with_py,
        candidate_rerank_exact_with_py_trusted, candidate_rerank_exact_with_py_trusted_scaled,
        canonical_q_kernel_name, canonical_weighted_c_kernel_name, centered_product_score,
        grm_int_interaction_scores_packed, matvec_row_major, min_support_encode, mincell_q_kernel,
        old_q_kernel, packed_upper_index, pair_indices, phase1_geometry, phase1_scores,
        phase1_scores_timed_with_c_kernel, phase3_geometry_timed, phase6b_geometry_timed,
        reconstruct_original_q, reference_geometry, remap_logic_gate_index,
        resolve_full_exact_kernel_names, sample_row_sparse_weighted_c, spectral_q_kernel,
        subset_q_kernel, weighted_interaction_py, weighted_interaction_py_binary_sparse,
        DosageChannel, WeightedCKernel,
    };
    use crate::garfield::residual::{
        build_design_matrix, build_exact_projection_from_grm_rust, build_no_grm_covariate_projector,
    };

    fn pack_grm_int_test_markers(x: &[f64], n_samples: usize, n_markers: usize) -> Vec<u64> {
        let row_words = n_samples.div_ceil(64);
        let mut packed = vec![0u64; n_markers * row_words];
        for sample in 0..n_samples {
            for marker in 0..n_markers {
                if x[sample * n_markers + marker] != 0.0 {
                    packed[marker * row_words + sample / 64] |= 1u64 << (sample % 64);
                }
            }
        }
        packed
    }

    fn solve_small_dense_system(mut a: Vec<f64>, mut b: Vec<f64>, dim: usize) -> Vec<f64> {
        for pivot in 0..dim {
            let best = (pivot..dim)
                .max_by(|&left, &right| {
                    a[left * dim + pivot]
                        .abs()
                        .total_cmp(&a[right * dim + pivot].abs())
                })
                .unwrap();
            for column in 0..dim {
                a.swap(pivot * dim + column, best * dim + column);
            }
            b.swap(pivot, best);
            let diagonal = a[pivot * dim + pivot];
            assert!(diagonal.is_finite() && diagonal.abs() > 1.0e-12);
            for column in pivot..dim {
                a[pivot * dim + column] /= diagonal;
            }
            b[pivot] /= diagonal;
            for row in 0..dim {
                if row == pivot {
                    continue;
                }
                let factor = a[row * dim + pivot];
                for column in pivot..dim {
                    a[row * dim + column] -= factor * a[pivot * dim + column];
                }
                b[row] -= factor * b[pivot];
            }
        }
        b
    }

    fn dense_gls_interaction_chisq(
        p_c: &[f64],
        y: &[f64],
        columns: [&[f64]; 3],
        score_scale: f64,
    ) -> f64 {
        let n = y.len();
        let projected_y = matvec_row_major(p_c, n, y);
        let projected_columns = columns
            .iter()
            .map(|column| matvec_row_major(p_c, n, column))
            .collect::<Vec<_>>();
        let mut normal = vec![0.0; 9];
        let mut rhs = vec![0.0; 3];
        for left in 0..3 {
            rhs[left] = columns[left]
                .iter()
                .zip(projected_y.iter())
                .map(|(&x, &py)| x * py)
                .sum();
            for right in 0..3 {
                normal[left * 3 + right] = columns[left]
                    .iter()
                    .zip(projected_columns[right].iter())
                    .map(|(&x, &pz)| x * pz)
                    .sum();
            }
        }
        let beta = solve_small_dense_system(normal.clone(), rhs, 3);
        let mut interaction_basis = vec![0.0; 3];
        interaction_basis[2] = 1.0;
        let inverse_interaction_column = solve_small_dense_system(normal, interaction_basis, 3);
        score_scale * beta[2] * beta[2] / inverse_interaction_column[2]
    }

    fn grm_int_dense_fixture() -> (Vec<u64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = 8usize;
        let g1 = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let g2 = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let x = (0..n)
            .flat_map(|sample| [g1[sample], g2[sample]])
            .collect::<Vec<_>>();
        let packed = pack_grm_int_test_markers(&x, n, 2);
        let covariate = vec![-1.3, -0.8, -0.25, 0.1, 0.45, 0.9, 1.4, 2.0];
        let grm = (0..n)
            .flat_map(|left| (0..n).map(move |right| 0.35_f64.powi(left.abs_diff(right) as i32)))
            .collect::<Vec<_>>();
        let p_c =
            build_exact_projection_from_grm_rust(&grm, n, Some(&covariate), 1, 1, 0.7, true, false)
                .unwrap();
        let h = (0..n)
            .map(|sample| (g1[sample] - 0.5) * (g2[sample] - 0.625))
            .collect::<Vec<_>>();
        let y = (0..n)
            .map(|sample| {
                0.4 + 1.15 * g1[sample] - 0.8 * g2[sample]
                    + 2.2 * g1[sample] * g2[sample]
                    + 0.07 * (sample as f64).powi(2)
            })
            .collect::<Vec<_>>();
        (packed, p_c, y, g1, g2, h)
    }

    #[test]
    fn grm_int_fwl_matches_full_dense_gls_with_covariate_and_grm_scale() {
        let (packed, p_c, y, g1, g2, h_centered) = grm_int_dense_fixture();
        let n = y.len();
        let projected_y = matvec_row_major(&p_c, n, &y);
        let scale = 1.7;
        let scores = grm_int_interaction_scores_packed(
            &packed,
            n.div_ceil(64),
            n,
            2,
            &[(0, 1)],
            Some(&p_c),
            None,
            &projected_y,
            scale,
        )
        .unwrap();
        let expected_centered =
            dense_gls_interaction_chisq(&p_c, &y, [&g1, &g2, &h_centered], scale);
        let h_raw = g1
            .iter()
            .zip(g2.iter())
            .map(|(&left, &right)| left * right)
            .collect::<Vec<_>>();
        let expected_raw = dense_gls_interaction_chisq(&p_c, &y, [&g1, &g2, &h_raw], scale);
        assert!(scores.chisq[0].is_finite());
        assert!((scores.chisq[0] - expected_centered).abs() < 1.0e-8);
        assert!((expected_centered - expected_raw).abs() < 1.0e-8);
        let doubled = grm_int_interaction_scores_packed(
            &packed,
            n.div_ceil(64),
            n,
            2,
            &[(0, 1)],
            Some(&p_c),
            None,
            &projected_y,
            2.0 * scale,
        )
        .unwrap();
        assert!((doubled.chisq[0] - 2.0 * scores.chisq[0]).abs() < 1.0e-8);
    }

    #[test]
    fn grm_int_compact_no_grm_matches_dense_projection_and_reports_rank_failures() {
        let (packed, _, y, g1, g2, h) = grm_int_dense_fixture();
        let n = y.len();
        let covariate = vec![-1.3, -0.8, -0.25, 0.1, 0.45, 0.9, 1.4, 2.0];
        let design = build_design_matrix(Some(&covariate), n, 1, true).unwrap();
        let projector = build_no_grm_covariate_projector(&design.0, n, design.1).unwrap();
        let identity = (0..n * n)
            .map(|index| (index / n == index % n) as u8 as f64)
            .collect::<Vec<_>>();
        let p_c = build_exact_projection_from_grm_rust(
            &identity,
            n,
            Some(&covariate),
            1,
            1,
            f64::INFINITY,
            true,
            false,
        )
        .unwrap();
        let dense_py = matvec_row_major(&p_c, n, &y);
        let compact_py = projector.project(&y).unwrap();
        let dense = grm_int_interaction_scores_packed(
            &packed,
            n.div_ceil(64),
            n,
            2,
            &[(0, 1)],
            Some(&p_c),
            None,
            &dense_py,
            1.0 / 1.9,
        )
        .unwrap();
        let compact = grm_int_interaction_scores_packed(
            &packed,
            n.div_ceil(64),
            n,
            2,
            &[(0, 1)],
            None,
            Some(&projector),
            &compact_py,
            1.0 / 1.9,
        )
        .unwrap();
        let expected = dense_gls_interaction_chisq(&p_c, &y, [&g1, &g2, &h], 1.0 / 1.9);
        assert!((dense.chisq[0] - expected).abs() < 1.0e-8);
        assert!((compact.chisq[0] - expected).abs() < 1.0e-8);

        let same_marker_twice = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let duplicate_x = (0..n)
            .flat_map(|sample| [same_marker_twice[sample], same_marker_twice[sample]])
            .collect::<Vec<_>>();
        let duplicate_bits = pack_grm_int_test_markers(&duplicate_x, n, 2);
        let singular = grm_int_interaction_scores_packed(
            &duplicate_bits,
            n.div_ceil(64),
            n,
            2,
            &[(0, 1)],
            Some(&p_c),
            None,
            &dense_py,
            1.0,
        )
        .unwrap();
        assert!(singular.chisq[0].is_nan());
        assert_eq!(singular.diagnostics.singular_endpoint_design, 1);

        let h_covariate = h.clone();
        let no_interaction_p = build_exact_projection_from_grm_rust(
            &identity,
            n,
            Some(&h_covariate),
            1,
            1,
            f64::INFINITY,
            true,
            false,
        )
        .unwrap();
        let no_interaction_py = matvec_row_major(&no_interaction_p, n, &y);
        let no_interaction = grm_int_interaction_scores_packed(
            &packed,
            n.div_ceil(64),
            n,
            2,
            &[(0, 1)],
            Some(&no_interaction_p),
            None,
            &no_interaction_py,
            1.0,
        )
        .unwrap();
        assert!(no_interaction.chisq[0].is_nan());
        assert_eq!(
            no_interaction
                .diagnostics
                .negligible_residual_interaction_information,
            1
        );
    }

    #[test]
    fn candidate_au_dispatch_defaults_to_support_fused_py_and_keeps_references_available() {
        assert_eq!(
            super::candidate_au_kernel_from_env_value(None).unwrap(),
            super::CandidateAuKernel::SupportFusedPy
        );
        assert_eq!(
            super::candidate_au_kernel_from_env_value(Some("reference")).unwrap(),
            super::CandidateAuKernel::Reference
        );
        assert_eq!(
            super::candidate_au_kernel_from_env_value(Some("support")).unwrap(),
            super::CandidateAuKernel::Support
        );
        assert_eq!(
            super::candidate_au_kernel_from_env_value(Some("support_fused_py")).unwrap(),
            super::CandidateAuKernel::SupportFusedPy
        );
        assert!(super::candidate_au_kernel_from_env_value(Some("typo")).is_err());
    }

    #[test]
    fn candidate_au_support_matches_reference_bits_and_missing_semantics() {
        // Rows 0 and 1 support marker 0; rows 0 and 2 support marker 1.
        // The nonzero reference products, in ascending sample order, are
        // a12 = 7 + 11, u0 = 2, and u1 = 7.
        let x = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let z = [2.0, 7.0, 3.0, 11.0, 5.0, 13.0];
        let bits = [0b011u64, 0b101u64];
        let got = super::candidate_au_support(&bits, 1, 3, 2, &x, &z, 0, 1).unwrap();
        assert_eq!(got.a12.to_bits(), 18.0f64.to_bits());
        assert_eq!(got.u0.to_bits(), 2.0f64.to_bits());
        assert_eq!(got.u1.to_bits(), 7.0f64.to_bits());
        super::validate_candidate_bitset_dense_semantics(&bits, 1, &x, 3, 2).unwrap();
        let mut missing = x;
        missing[0] = f64::NAN;
        assert!(
            super::validate_candidate_bitset_dense_semantics(&bits, 1, &missing, 3, 2).is_err()
        );
        let mut mismatched = x;
        mismatched[0] = 0.0;
        assert!(
            super::validate_candidate_bitset_dense_semantics(&bits, 1, &mismatched, 3, 2).is_err()
        );
    }

    #[test]
    fn candidate_au_support_is_bitwise_equal_across_word_boundary() {
        let n = 131usize;
        let m = 3usize;
        let words = n.div_ceil(64);
        let mut x = vec![0.0; n * m];
        let mut z = vec![0.0; n * m];
        let mut bits = vec![0u64; m * words];
        for row in 0..n {
            for col in 0..m {
                let present = (row * 7 + col * 11) % (col + 3) != 0;
                x[row * m + col] = present as u8 as f64;
                if present {
                    bits[col * words + row / 64] |= 1u64 << (row % 64);
                }
                z[row * m + col] = ((row * 13 + col * 5) % 19) as f64 * 0.013 - 0.1;
            }
        }
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let mut want = super::CandidateAu {
                a12: 0.0,
                u0: 0.0,
                u1: 0.0,
            };
            for row in 0..n {
                let xi = x[row * m + i];
                let xj = x[row * m + j];
                let w = xi * xj;
                want.a12 += xi * z[row * m + j];
                want.u0 += w * z[row * m + i];
                want.u1 += w * z[row * m + j];
            }
            let got = super::candidate_au_support(&bits, words, n, m, &x, &z, i, j).unwrap();
            assert_eq!(got.a12.to_bits(), want.a12.to_bits());
            assert_eq!(got.u0.to_bits(), want.u0.to_bits());
            assert_eq!(got.u1.to_bits(), want.u1.to_bits());
        }
    }

    #[test]
    fn candidate_au_support_fused_py_matches_separate_exact_reductions() {
        let n = 131usize;
        let m = 3usize;
        let words = n.div_ceil(64);
        let mut x = vec![0.0; n * m];
        let mut z = vec![0.0; n * m];
        let mut bits = vec![0u64; m * words];
        let mut py = vec![0.0; n];
        for row in 0..n {
            for col in 0..m {
                let present = (row * 7 + col * 11) % (col + 3) != 0;
                x[row * m + col] = present as u8 as f64;
                if present {
                    bits[col * words + row / 64] |= 1u64 << (row % 64);
                }
                z[row * m + col] = ((row * 13 + col * 5) % 19) as f64 * 0.013 - 0.1;
            }
            py[row] = ((row * 17) % 29) as f64 * 0.017 - 0.23;
        }

        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let mut expected = super::CandidateAu {
                a12: 0.0,
                u0: 0.0,
                u1: 0.0,
            };
            let mut expected_interaction_py = 0.0;
            for row in 0..n {
                let x_i = x[row * m + i];
                let x_j = x[row * m + j];
                let w = x_i * x_j;
                expected.a12 += x_i * z[row * m + j];
                expected.u0 += w * z[row * m + i];
                expected.u1 += w * z[row * m + j];
                expected_interaction_py += w * py[row];
            }

            let (got, got_interaction_py) =
                super::candidate_au_support_fused_py(&bits, words, n, m, &z, &py, i, j).unwrap();
            assert_eq!(got.a12.to_bits(), expected.a12.to_bits());
            assert_eq!(got.u0.to_bits(), expected.u0.to_bits());
            assert_eq!(got.u1.to_bits(), expected.u1.to_bits());
            assert_eq!(
                got_interaction_py.to_bits(),
                expected_interaction_py.to_bits()
            );
        }
    }

    #[test]
    fn candidate_au_blas_preserves_h_direction_on_asymmetric_fixture() {
        let x = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let z = [2.0, 7.0, 3.0, 11.0, 5.0, 13.0];
        let got = super::candidate_au_window_blas(&x, &z, 3, 2, &[0], &[1]).unwrap();
        assert_eq!(got.values[0].a12, 18.0);
        assert_eq!(got.values[0].u0, 2.0); // H[j,i], not H[i,j]
        assert_eq!(got.values[0].u1, 7.0); // H[i,j], not H[j,i]
    }

    #[test]
    fn candidate_pcx_sparse_support_matches_dense_gemm_orientation() {
        let n_samples = 5usize;
        let n_markers = 3usize;
        let row_words = 1usize;
        let supports = [0b1_0101u64, 0b0_1010u64, 0b1_1001u64];
        let p_c = (0..n_samples)
            .flat_map(|row| (0..n_samples).map(move |col| (row * 7 + col * 3 + 1) as f64))
            .collect::<Vec<_>>();
        let mut dense_x = vec![0.0; n_samples * n_markers];
        for marker in 0..n_markers {
            for row in 0..n_samples {
                dense_x[row * n_markers + marker] = ((supports[marker] >> row) & 1) as f64;
            }
        }
        let mut expected = vec![0.0; n_samples * n_markers];
        super::dgemm_row_major(
            false,
            false,
            n_samples,
            n_markers,
            n_samples,
            &p_c,
            &dense_x,
            &mut expected,
        )
        .unwrap();
        let got =
            super::candidate_pcx_sparse_support(&supports, row_words, n_samples, n_markers, &p_c)
                .unwrap();

        assert_eq!(got.len(), expected.len());
        let max_abs_error = got
            .iter()
            .zip(expected.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_abs_error <= 1.0e-12, "max_abs_error={max_abs_error}");
    }

    #[test]
    fn candidate_au_workspace_preflight_counts_matrices_and_candidate_staging() {
        let n = 3usize;
        let m = 2usize;
        let k = 4usize;
        let output = k * 3 * std::mem::size_of::<f64>();
        let staging = k * std::mem::size_of::<super::CandidateAu>();
        let matrices = (n * m + 2 * m * m) * std::mem::size_of::<f64>();
        assert_eq!(
            candidate_au_workspace_bytes(n, m, k, false).unwrap(),
            output
        );
        assert_eq!(
            candidate_au_workspace_bytes(n, m, k, true).unwrap(),
            output + staging + matrices
        );
        assert_eq!(
            super::candidate_au_support_fused_py_workspace_bytes(n, m, k).unwrap(),
            output + k * std::mem::size_of::<f64>()
        );
    }

    #[test]
    fn candidate_au_experimental_kernels_preserve_packed_rerank_outputs() {
        let n_samples = 8;
        let n_markers = 5;
        let x = [
            1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0,
            1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
        ];
        let mut bits = vec![0u64; n_markers];
        for row in 0..n_samples {
            for marker in 0..n_markers {
                if x[row * n_markers + marker] == 1 {
                    bits[marker] |= 1 << row;
                }
            }
        }
        let mut p_c = vec![0.0; n_samples * n_samples];
        for row in 0..n_samples {
            for col in 0..n_samples {
                p_c[row * n_samples + col] = if row == col {
                    1.5
                } else {
                    (row + col + 1) as f64 * 0.005
                };
            }
        }
        let y = [0.3, -0.8, 1.1, 0.2, -0.6, 1.7, -0.1, 0.4];
        let py = super::matvec_row_major(&p_c, n_samples, &y);
        let first = [0, 0, 1, 2, 3, 4];
        let second = [1, 4, 3, 4, 4, 3];
        let call = |kernel, pcx_sparse_support| {
            super::candidate_rerank_exact_packed_with_py_trusted_kernel(
                &bits,
                1,
                n_samples,
                n_markers,
                &p_c,
                &y,
                &py,
                &first,
                &second,
                5,
                super::DosageChannel::FwlProduct,
                kernel,
                pcx_sparse_support,
            )
            .unwrap()
        };
        let reference = call(super::CandidateAuKernel::Reference, false);
        for kernel in [
            super::CandidateAuKernel::Support,
            super::CandidateAuKernel::SupportFusedPy,
            super::CandidateAuKernel::WindowBlas,
        ] {
            let got = call(kernel, false);
            assert_eq!(got.pair_i, reference.pair_i);
            assert_eq!(got.pair_j, reference.pair_j);
            assert_eq!(got.top_order, reference.top_order);
            assert_eq!(got.candidate_rank, reference.candidate_rank);
            assert_eq!(got.best_max6_channel, reference.best_max6_channel);
            for (a, b) in got.q.iter().zip(&reference.q) {
                assert!((a - b).abs() < 1e-10);
            }
            for (a, b) in got.max6.iter().zip(&reference.max6) {
                assert!((a - b).abs() < 1e-10);
            }
            for (a, b) in got.dosage.iter().zip(&reference.dosage) {
                assert!(a == b || (a - b).abs() < 1e-10, "dosage {a} vs {b}");
            }
            for (a, b) in got.logic.iter().zip(&reference.logic) {
                for (&av, &bv) in a.iter().zip(b) {
                    assert!(av == bv || (av - bv).abs() < 1e-10, "{av} vs {bv}");
                }
            }
        }

        let sparse_pcx = call(super::CandidateAuKernel::Reference, true);
        assert!(sparse_pcx.timing.pcx_sparse_support);
        assert_eq!(sparse_pcx.pair_i, reference.pair_i);
        assert_eq!(sparse_pcx.pair_j, reference.pair_j);
        assert_eq!(sparse_pcx.candidate_rank, reference.candidate_rank);
        assert_eq!(sparse_pcx.top_order, reference.top_order);
        assert_eq!(sparse_pcx.best_max6_channel, reference.best_max6_channel);
        for (a, b) in sparse_pcx.z.iter().zip(&reference.z) {
            assert!((a - b).abs() < 1.0e-12, "PCX {a} vs {b}");
        }
        for (a, b) in sparse_pcx.q.iter().zip(&reference.q) {
            assert!((a - b).abs() < 1.0e-10);
        }
        for (a, b) in sparse_pcx.max6.iter().zip(&reference.max6) {
            assert!((a - b).abs() < 1.0e-10);
        }
    }

    #[test]
    fn candidate_rerank_matches_full_window_for_selected_pairs() {
        let x = vec![
            1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ];
        let n_samples = 6usize;
        let n_markers = 5usize;
        let mut p_c = vec![0.0; n_samples * n_samples];
        for row in 0..n_samples {
            for col in 0..n_samples {
                p_c[row * n_samples + col] = if row == col {
                    1.0
                } else {
                    0.03 * ((row + col + 1) as f64)
                };
            }
        }
        let y = vec![0.25, -1.5, 2.75, -0.125, 0.7, -0.4];
        let (candidate_first, candidate_second) = (vec![3_i64, 0, 3, 4], vec![1_i64, 4, 1, 2]);
        let candidate = candidate_rerank_exact(
            &x,
            n_samples,
            n_markers,
            &p_c,
            &y,
            &candidate_first,
            &candidate_second,
            4,
            DosageChannel::FwlProduct,
        )
        .unwrap();
        let encoding = min_support_encode(&x, n_samples, n_markers).unwrap();
        let full = phase6b_geometry_timed(&encoding.x, n_samples, n_markers, &p_c, 2)
            .unwrap()
            .0;

        assert_eq!(candidate.pair_i.len(), 3);
        assert_eq!(candidate.input_to_unique, vec![0, 1, 0, 2]);
        for (candidate_index, (&left, &right)) in candidate
            .pair_i
            .iter()
            .zip(candidate.pair_j.iter())
            .enumerate()
        {
            let full_index = full
                .pair_i
                .iter()
                .zip(full.pair_j.iter())
                .position(|(&i, &j)| i == left && j == right)
                .expect("candidate pair must be present in full scan");
            assert!(
                (candidate.q[candidate_index] - full.q[full_index]).abs() < 1.0e-10,
                "q mismatch pair=({}, {}): candidate={} full={} mincell={} cell={} support={}",
                left,
                right,
                candidate.q[candidate_index],
                full.q[full_index],
                candidate.mincell_q[candidate_index],
                candidate.mincell_cell[candidate_index],
                candidate.mincell_support[candidate_index]
            );
            assert!(
                (candidate.max6[candidate_index]
                    - phase1_scores(&full, &encoding.x, n_samples, n_markers, &p_c, &y, None,)
                        .unwrap()
                        .max6[full_index])
                    .abs()
                    < 1.0e-9
            );
        }
    }

    #[test]
    fn trusted_candidate_rerank_matches_validated_path() {
        let x = vec![
            1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ];
        let n_samples = 6usize;
        let n_markers = 5usize;
        let mut p_c = vec![0.0; n_samples * n_samples];
        for row in 0..n_samples {
            for col in 0..n_samples {
                p_c[row * n_samples + col] = if row == col {
                    1.0
                } else {
                    0.03 * ((row + col + 1) as f64)
                };
            }
        }
        let y = vec![0.25, -1.5, 2.75, -0.125, 0.7, -0.4];
        let py = matvec_row_major(&p_c, n_samples, &y);
        let candidate_first = vec![3_i64, 0, 3, 4];
        let candidate_second = vec![1_i64, 4, 1, 2];
        let validated = candidate_rerank_exact_with_py(
            &x,
            n_samples,
            n_markers,
            &p_c,
            &y,
            &py,
            &candidate_first,
            &candidate_second,
            4,
            DosageChannel::FwlProduct,
        )
        .unwrap();
        let trusted = candidate_rerank_exact_with_py_trusted(
            &x,
            n_samples,
            n_markers,
            &p_c,
            &y,
            &py,
            &candidate_first,
            &candidate_second,
            4,
            DosageChannel::FwlProduct,
        )
        .unwrap();
        assert_eq!(trusted.pair_i, validated.pair_i);
        assert_eq!(trusted.pair_j, validated.pair_j);
        assert_eq!(trusted.top_order, validated.top_order);
        assert_eq!(trusted.best_max6_channel, validated.best_max6_channel);
        for (got, want) in trusted.max6.iter().zip(validated.max6.iter()) {
            assert!((got - want).abs() < 1.0e-12);
        }

        let scale = 2.5;
        let scaled = candidate_rerank_exact_with_py_trusted_scaled(
            &x,
            n_samples,
            n_markers,
            &p_c,
            &y,
            &py,
            &candidate_first,
            &candidate_second,
            4,
            DosageChannel::FwlProduct,
            scale,
        )
        .unwrap();
        assert_eq!(scaled.top_order, trusted.top_order);
        assert_eq!(scaled.best_max6_channel, trusted.best_max6_channel);
        for (got, want) in scaled.max6.iter().zip(trusted.max6.iter()) {
            assert!((got - scale * want).abs() < 1.0e-12);
        }
    }

    #[test]
    fn packed_candidate_rerank_matches_dense_reference() {
        let x = vec![
            1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ];
        let n_samples = 6usize;
        let n_markers = 5usize;
        let row_words = n_samples.div_ceil(64);
        let mut packed = vec![0u64; n_markers * row_words];
        for row in 0..n_samples {
            for marker in 0..n_markers {
                if x[row * n_markers + marker] == 1.0 {
                    packed[marker * row_words + row / 64] |= 1u64 << (row % 64);
                }
            }
        }
        let mut p_c = vec![0.0; n_samples * n_samples];
        for row in 0..n_samples {
            for col in 0..n_samples {
                p_c[row * n_samples + col] = if row == col {
                    1.0
                } else {
                    0.03 * ((row + col + 1) as f64)
                };
            }
        }
        let y = vec![0.25, -1.5, 2.75, -0.125, 0.7, -0.4];
        let py = matvec_row_major(&p_c, n_samples, &y);
        let candidate_first = vec![3_i64, 0, 3, 4];
        let candidate_second = vec![1_i64, 4, 1, 2];
        let dense = candidate_rerank_exact_with_py_trusted(
            &x,
            n_samples,
            n_markers,
            &p_c,
            &y,
            &py,
            &candidate_first,
            &candidate_second,
            4,
            DosageChannel::FwlProduct,
        )
        .unwrap();
        let packed_result = candidate_rerank_exact_packed_with_py_trusted(
            &packed,
            row_words,
            n_samples,
            n_markers,
            &p_c,
            &y,
            &py,
            &candidate_first,
            &candidate_second,
            4,
            DosageChannel::FwlProduct,
        )
        .unwrap();

        assert_eq!(packed_result.pair_i, dense.pair_i);
        assert_eq!(packed_result.pair_j, dense.pair_j);
        assert_eq!(packed_result.input_to_unique, dense.input_to_unique);
        assert_eq!(packed_result.unique_markers, dense.unique_markers);
        assert_eq!(packed_result.flipped, dense.flipped);
        assert_eq!(packed_result.mincell_cell, dense.mincell_cell);
        assert_eq!(packed_result.mincell_support, dense.mincell_support);
        assert_eq!(packed_result.top_order, dense.top_order);
        assert_eq!(packed_result.candidate_rank, dense.candidate_rank);
        assert_eq!(packed_result.best_logic_gate, dense.best_logic_gate);
        assert_eq!(packed_result.best_max6_channel, dense.best_max6_channel);
        for (got, want) in packed_result
            .z
            .iter()
            .zip(dense.z.iter())
            .chain(packed_result.q.iter().zip(dense.q.iter()))
            .chain(packed_result.mincell_q.iter().zip(dense.mincell_q.iter()))
            .chain(packed_result.dosage.iter().zip(dense.dosage.iter()))
            .chain(packed_result.max6.iter().zip(dense.max6.iter()))
        {
            if got.is_infinite() || want.is_infinite() {
                assert_eq!(got, want, "packed={got}, dense={want}");
            } else {
                assert!((got - want).abs() < 1.0e-12, "packed={got}, dense={want}");
            }
        }
        for (got, want) in packed_result.logic.iter().zip(dense.logic.iter()) {
            for (got, want) in got.iter().zip(want.iter()) {
                if got.is_infinite() || want.is_infinite() {
                    assert_eq!(got, want, "packed={got}, dense={want}");
                } else {
                    assert!((got - want).abs() < 1.0e-12, "packed={got}, dense={want}");
                }
            }
        }
    }

    #[test]
    fn no_grm_covariate_candidate_rerank_matches_dense_projection() {
        let x = vec![
            1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ];
        let n_samples = 6usize;
        let n_markers = 5usize;
        let row_words = n_samples.div_ceil(64);
        let mut packed = vec![0u64; n_markers * row_words];
        for row in 0..n_samples {
            for marker in 0..n_markers {
                if x[row * n_markers + marker] == 1.0 {
                    packed[marker * row_words + row / 64] |= 1u64 << (row % 64);
                }
            }
        }
        let covariate = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let design = build_design_matrix(Some(&covariate), n_samples, 1, true).unwrap();
        let projector = build_no_grm_covariate_projector(&design.0, n_samples, design.1).unwrap();
        let identity = (0..n_samples * n_samples)
            .map(|index| {
                if index / n_samples == index % n_samples {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let p_c = build_exact_projection_from_grm_rust(
            &identity,
            n_samples,
            Some(&covariate),
            1,
            1,
            f64::INFINITY,
            true,
            false,
        )
        .unwrap();
        let y = vec![0.25, -1.5, 2.75, -0.125, 0.7, -0.4];
        let projected_y = matvec_row_major(&p_c, n_samples, &y);
        let compact_projected_y = projector.project(&y).unwrap();
        for (got, want) in compact_projected_y.iter().zip(projected_y.iter()) {
            assert!(
                (got - want).abs() < 1.0e-9,
                "compact Py={got}, dense Py={want}"
            );
        }
        let candidate_first = vec![3_i64, 0, 3, 4];
        let candidate_second = vec![1_i64, 4, 1, 2];
        let scale = 1.7;
        let dense = candidate_rerank_exact_packed_with_py_trusted_scaled(
            &packed,
            row_words,
            n_samples,
            n_markers,
            &p_c,
            &y,
            &projected_y,
            &candidate_first,
            &candidate_second,
            4,
            DosageChannel::FwlProduct,
            scale,
        )
        .unwrap();
        let compact = candidate_rerank_exact_packed_no_grm_covariate_with_py_trusted_scaled(
            &packed,
            row_words,
            n_samples,
            n_markers,
            &projector,
            &y,
            &projected_y,
            &candidate_first,
            &candidate_second,
            4,
            DosageChannel::FwlProduct,
            scale,
        )
        .unwrap();
        assert_eq!(compact.pair_i, dense.pair_i);
        assert_eq!(compact.pair_j, dense.pair_j);
        assert_eq!(compact.top_order, dense.top_order);
        assert_eq!(compact.candidate_rank, dense.candidate_rank);
        assert_eq!(compact.best_max6_channel, dense.best_max6_channel);
        for (got, want) in compact.z.iter().zip(dense.z.iter()) {
            assert!(
                (got - want).abs() < 1.0e-9,
                "compact z={got}, dense z={want}"
            );
        }
        for (index, (got, want)) in compact.q.iter().zip(dense.q.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1.0e-9,
                "q candidate={index}: compact={got}, dense={want}"
            );
        }
        for (index, (got, want)) in compact
            .mincell_q
            .iter()
            .zip(dense.mincell_q.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1.0e-9,
                "mincell q candidate={index}: compact={got}, dense={want}"
            );
        }
        for (index, (got, want)) in compact.max6.iter().zip(dense.max6.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1.0e-9,
                "max6 candidate={index}: compact={got}, dense={want}"
            );
        }
        for (index, (got, want)) in compact.dosage.iter().zip(dense.dosage.iter()).enumerate() {
            if got.is_finite() || want.is_finite() {
                assert!(
                    (got - want).abs() < 1.0e-9,
                    "dosage candidate={index}: compact={got}, dense={want}"
                );
            } else {
                assert!(got.is_infinite() && want.is_infinite());
            }
        }
        for (got, want) in compact.logic.iter().zip(dense.logic.iter()) {
            for (got, want) in got.iter().zip(want.iter()) {
                if got.is_finite() || want.is_finite() {
                    assert!((got - want).abs() < 1.0e-9, "compact={got}, dense={want}");
                } else {
                    assert!(got.is_infinite() && want.is_infinite());
                }
            }
        }
    }

    #[test]
    #[ignore = "diagnostic benchmark; run explicitly"]
    fn no_grm_covariate_candidate_rerank_speed_pilot() {
        let n_samples = 1024usize;
        let n_markers = 384usize;
        let row_words = n_samples.div_ceil(64);
        let mut packed = vec![0u64; n_markers * row_words];
        for marker in 0..n_markers {
            for row in 0..n_samples {
                let hash = row
                    .wrapping_mul(1_664_525)
                    .wrapping_add(marker.wrapping_mul(1_013_904_223))
                    .wrapping_add(0x9E37_79B9);
                if hash % 7 == 0 {
                    packed[marker * row_words + row / 64] |= 1u64 << (row % 64);
                }
            }
        }
        let inverse_n = 1.0 / n_samples as f64;
        let mut p_c = vec![0.0_f64; n_samples * n_samples];
        for row in 0..n_samples {
            for col in 0..n_samples {
                p_c[row * n_samples + col] = if row == col {
                    1.0 - inverse_n
                } else {
                    -inverse_n
                };
            }
        }
        let design = build_design_matrix(None, n_samples, 0, true).unwrap();
        let projector = build_no_grm_covariate_projector(&design.0, n_samples, design.1).unwrap();
        let y = (0..n_samples)
            .map(|row| ((row as f64) * 0.017).sin() + 0.2 * ((row % 11) as f64))
            .collect::<Vec<_>>();
        let projected_y = matvec_row_major(&p_c, n_samples, &y);
        let compact_projected_y = projector.project(&y).unwrap();
        let n_candidates = 5_000usize;
        let mut candidate_first = Vec::with_capacity(n_candidates);
        let mut candidate_second = Vec::with_capacity(n_candidates);
        for index in 0..n_candidates {
            let first = (index.wrapping_mul(37)) % (n_markers - 1);
            let offset = 1 + (index.wrapping_mul(97)) % (n_markers - 1);
            let second = (first + offset) % n_markers;
            candidate_first.push(first as i64);
            candidate_second.push(second as i64);
        }

        let _ = candidate_rerank_exact_packed_with_py_trusted_scaled(
            &packed,
            row_words,
            n_samples,
            n_markers,
            &p_c,
            &y,
            &projected_y,
            &candidate_first,
            &candidate_second,
            500,
            DosageChannel::FwlProduct,
            1.0,
        )
        .unwrap();
        let _ = candidate_rerank_exact_packed_no_grm_covariate_with_py_trusted_scaled(
            &packed,
            row_words,
            n_samples,
            n_markers,
            &projector,
            &y,
            &compact_projected_y,
            &candidate_first,
            &candidate_second,
            500,
            DosageChannel::FwlProduct,
            1.0,
        )
        .unwrap();

        let run_dense = || {
            candidate_rerank_exact_packed_with_py_trusted_scaled(
                &packed,
                row_words,
                n_samples,
                n_markers,
                &p_c,
                &y,
                &projected_y,
                &candidate_first,
                &candidate_second,
                500,
                DosageChannel::FwlProduct,
                1.0,
            )
            .unwrap()
        };
        let run_compact = || {
            candidate_rerank_exact_packed_no_grm_covariate_with_py_trusted_scaled(
                &packed,
                row_words,
                n_samples,
                n_markers,
                &projector,
                &y,
                &compact_projected_y,
                &candidate_first,
                &candidate_second,
                500,
                DosageChannel::FwlProduct,
                1.0,
            )
            .unwrap()
        };
        let mut dense_times = Vec::with_capacity(5);
        let mut dense = None;
        let mut compact_times = Vec::with_capacity(5);
        let mut compact = None;
        for _ in 0..5 {
            let dense_started = Instant::now();
            dense = Some(run_dense());
            dense_times.push(dense_started.elapsed().as_secs_f64());
            let compact_started = Instant::now();
            compact = Some(run_compact());
            compact_times.push(compact_started.elapsed().as_secs_f64());
        }
        let dense = dense.unwrap();
        let compact = compact.unwrap();
        dense_times.sort_by(f64::total_cmp);
        compact_times.sort_by(f64::total_cmp);
        let dense_seconds = dense_times[dense_times.len() / 2];
        let compact_seconds = compact_times[compact_times.len() / 2];
        assert_eq!(compact.top_order, dense.top_order);
        eprintln!(
            "no-GRM candidate speed pilot: dense={dense_seconds:.6}s compact={compact_seconds:.6}s speedup={:.2}x",
            dense_seconds / compact_seconds
        );
    }

    #[test]
    fn centered_product_channel_matches_direct_pc_quadratic_form() {
        let n = 5usize;
        let xi = [0.0, 1.0, 1.0, 0.0, 1.0];
        let xj = [1.0, 0.0, 1.0, 1.0, 0.0];
        let y = [0.25, -1.5, 2.75, -0.125, 0.7];
        let mut p_c = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..n {
                p_c[row * n + col] = if row == col {
                    1.0 - 1.0 / n as f64
                } else {
                    -1.0 / n as f64
                };
            }
        }
        let py = matvec_row_major(&p_c, n, &y);
        let raw_w = xi
            .iter()
            .zip(xj.iter())
            .map(|(&left, &right)| left * right)
            .collect::<Vec<_>>();
        let mean0 = xi.iter().sum::<f64>() / n as f64;
        let mean1 = xj.iter().sum::<f64>() / n as f64;
        let b0 = xi.iter().zip(py.iter()).map(|(&x, &v)| x * v).sum();
        let b1 = xj.iter().zip(py.iter()).map(|(&x, &v)| x * v).sum();
        let c = raw_w.iter().zip(py.iter()).map(|(&w, &v)| w * v).sum();
        let p_xi = matvec_row_major(&p_c, n, &xi);
        let p_xj = matvec_row_major(&p_c, n, &xj);
        let p_w = matvec_row_major(&p_c, n, &raw_w);
        let a00 = xi.iter().zip(p_xi.iter()).map(|(&x, &v)| x * v).sum();
        let a11 = xj.iter().zip(p_xj.iter()).map(|(&x, &v)| x * v).sum();
        let a01 = xi.iter().zip(p_xj.iter()).map(|(&x, &v)| x * v).sum();
        let u0 = raw_w.iter().zip(p_xi.iter()).map(|(&w, &v)| w * v).sum();
        let u1 = raw_w.iter().zip(p_xj.iter()).map(|(&w, &v)| w * v).sum();
        let q = raw_w.iter().zip(p_w.iter()).map(|(&w, &v)| w * v).sum();

        let centered = xi
            .iter()
            .zip(xj.iter())
            .map(|(&left, &right)| (left - mean0) * (right - mean1))
            .collect::<Vec<_>>();
        let expected_num = centered
            .iter()
            .zip(py.iter())
            .map(|(&w, &v)| w * v)
            .sum::<f64>();
        let p_centered = matvec_row_major(&p_c, n, &centered);
        let expected_den = centered
            .iter()
            .zip(p_centered.iter())
            .map(|(&w, &v)| w * v)
            .sum::<f64>();
        let expected = expected_num * expected_num / expected_den;
        let got = centered_product_score(
            b0,
            b1,
            a00,
            a11,
            a01,
            u0,
            u1,
            q,
            c,
            mean0,
            mean1,
            py.iter().sum(),
        );
        assert!(
            (got - expected).abs() < 1.0e-12,
            "got={got}, expected={expected}"
        );
    }

    #[test]
    fn mincell_reconstructs_original_q_for_each_cell_choice() {
        let a1 = 3.0;
        let a2 = 5.0;
        let a12 = 1.0;
        let u0 = 2.0;
        let u1 = 4.0;
        let q11 = 7.0;
        let q10 = a1 + q11 - 2.0 * u0;
        let q01 = a2 + q11 - 2.0 * u1;
        let q00 = a1 + a2 + q11 + 2.0 * a12 - 2.0 * u0 - 2.0 * u1;

        assert_eq!(reconstruct_original_q(q11, 3, a1, a2, a12, u0, u1), q11);
        assert_eq!(reconstruct_original_q(q10, 2, a1, a2, a12, u0, u1), q11);
        assert_eq!(reconstruct_original_q(q01, 1, a1, a2, a12, u0, u1), q11);
        assert_eq!(reconstruct_original_q(q00, 0, a1, a2, a12, u0, u1), q11);
    }

    #[test]
    fn mincell_bitset_allows_empty_cell_and_reconstructs_q() {
        let x = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let n_samples = 4usize;
        let n_markers = 2usize;
        let p_c = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let (pair_i, pair_j) = pair_indices(n_markers);
        let result = mincell_q_kernel(&x, n_samples, n_markers, &p_c, &pair_i, &pair_j).unwrap();

        assert_eq!(result.cell, vec![1]);
        assert_eq!(result.support, vec![0]);
        assert_eq!(result.q_min, vec![0.0]);

        let a1 = 2.0;
        let a2 = 1.0;
        let a12 = 1.0;
        let u0 = 1.0;
        let u1 = 1.0;
        let q11 = reconstruct_original_q(result.q_min[0], result.cell[0], a1, a2, a12, u0, u1);
        assert_eq!(q11, 1.0);
    }

    #[test]
    fn mincell_bitset_uses_exact_quadratic_sum_for_nonempty_cell() {
        let x = vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let n_samples = 5usize;
        let n_markers = 2usize;
        let mut p_c = vec![0.0; n_samples * n_samples];
        for row in 0..n_samples {
            p_c[row * n_samples + row] = (row + 1) as f64;
        }
        let (pair_i, pair_j) = pair_indices(n_markers);
        let result = mincell_q_kernel(&x, n_samples, n_markers, &p_c, &pair_i, &pair_j).unwrap();

        // Cell counts are 00=1, 01=1, 10=1, 11=2; the stable minimum is 00.
        assert_eq!(result.cell, vec![0]);
        assert_eq!(result.support, vec![1]);
        assert_eq!(result.q_min, vec![1.0]);
    }

    #[test]
    fn mincell_reports_cell_enumerate_quadratic_and_reconstruct_timing() {
        let x = vec![
            1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        ];
        let n_samples = 4usize;
        let n_markers = 4usize;
        let mut p_c = vec![0.0; n_samples * n_samples];
        for row in 0..n_samples {
            p_c[row * n_samples + row] = 1.0;
        }

        let (_, timing) = phase6b_geometry_timed(&x, n_samples, n_markers, &p_c, 2).unwrap();
        let q_timing = timing
            .mincell_q_timing
            .expect("mincell q timing should be present");
        assert!(q_timing.bitset_seconds.is_finite() && q_timing.bitset_seconds >= 0.0);
        assert!(q_timing.cell_seconds.is_finite() && q_timing.cell_seconds >= 0.0);
        assert!(q_timing.enumerate_seconds.is_finite() && q_timing.enumerate_seconds >= 0.0);
        assert!(q_timing.quadratic_seconds.is_finite() && q_timing.quadratic_seconds >= 0.0);
        assert!(q_timing.reconstruct_seconds.is_finite() && q_timing.reconstruct_seconds >= 0.0);
        assert!(q_timing.sum().is_finite() && q_timing.sum() >= 0.0);
    }

    #[test]
    fn mincell_geometry_reuses_existing_score_assembly() {
        let x = vec![
            1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let n = 8usize;
        let m = 4usize;
        let mut p_c = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..n {
                p_c[row * n + col] = if row == col { 7.0 / 8.0 } else { -1.0 / 8.0 };
            }
        }
        let y = vec![0.2, -1.0, 0.5, 1.7, -0.3, 0.9, -0.4, 0.6];
        let baseline = phase3_geometry_timed(&x, n, m, &p_c, 2, true).unwrap().0;
        let candidate = phase6b_geometry_timed(&x, n, m, &p_c, 2).unwrap().0;

        for (expected, got) in baseline.q.iter().zip(candidate.q.iter()) {
            assert!(
                (expected - got).abs() < 1.0e-10,
                "q mismatch: {expected} vs {got}"
            );
        }
        let baseline_scores = phase1_scores_timed_with_c_kernel(
            &baseline,
            &x,
            n,
            m,
            &p_c,
            &y,
            None,
            WeightedCKernel::DenseReference,
        )
        .unwrap()
        .0;
        let candidate_scores = phase1_scores_timed_with_c_kernel(
            &candidate,
            &x,
            n,
            m,
            &p_c,
            &y,
            None,
            WeightedCKernel::DenseReference,
        )
        .unwrap()
        .0;
        for (expected, got) in baseline_scores
            .max6
            .iter()
            .zip(candidate_scores.max6.iter())
        {
            assert!(
                (expected - got).abs() < 1.0e-9,
                "Max6 mismatch: {expected} vs {got}"
            );
        }
    }

    #[test]
    fn binary_sparse_c_matches_dense_c_and_preserves_pair_order() {
        let x = vec![
            1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 1.0,
        ];
        let py = vec![0.25, -1.5, 0.0, 2.75, -0.125];
        let n_samples = 5usize;
        let n_markers = 4usize;
        let (pair_i, pair_j) = pair_indices(n_markers);

        let dense = weighted_interaction_py(&x, &py, n_samples, n_markers, &pair_i, &pair_j);
        let sparse =
            weighted_interaction_py_binary_sparse(&x, &py, n_samples, n_markers, &pair_i, &pair_j)
                .unwrap();

        assert_eq!(sparse, dense);
    }

    #[test]
    fn binary_sparse_c_preserves_all_scores() {
        let x = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let n_samples = 4usize;
        let n_markers = 3usize;
        let p_c = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![0.25, -1.5, 2.75, -0.125];
        let geometry = phase1_geometry(&x, n_samples, n_markers, &p_c, 2).unwrap();

        let dense = phase1_scores_timed_with_c_kernel(
            &geometry,
            &x,
            n_samples,
            n_markers,
            &p_c,
            &y,
            Some((0, 2)),
            WeightedCKernel::DenseReference,
        )
        .unwrap()
        .0;
        let sparse = phase1_scores_timed_with_c_kernel(
            &geometry,
            &x,
            n_samples,
            n_markers,
            &p_c,
            &y,
            Some((0, 2)),
            WeightedCKernel::BinarySparse,
        )
        .unwrap()
        .0;

        assert_eq!(dense.target_index, sparse.target_index);
        assert_eq!(dense.max6_rank, sparse.max6_rank);
        assert_eq!(dense.logic, sparse.logic);
        assert_eq!(dense.dosage, sparse.dosage);
        assert_eq!(dense.max6, sparse.max6);
    }

    #[test]
    fn sample_row_sparse_c_preserves_all_channels_and_rank() {
        let x = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let n_samples = 4usize;
        let n_markers = 3usize;
        let p_c = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![0.25, -1.5, 2.75, -0.125];
        let geometry = phase1_geometry(&x, n_samples, n_markers, &p_c, 2).unwrap();

        let dense = phase1_scores_timed_with_c_kernel(
            &geometry,
            &x,
            n_samples,
            n_markers,
            &p_c,
            &y,
            Some((0, 2)),
            WeightedCKernel::DenseReference,
        )
        .unwrap()
        .0;
        let sample_row = phase1_scores_timed_with_c_kernel(
            &geometry,
            &x,
            n_samples,
            n_markers,
            &p_c,
            &y,
            Some((0, 2)),
            WeightedCKernel::SampleRowSparse,
        )
        .unwrap()
        .0;

        assert_eq!(dense.target_index, sample_row.target_index);
        assert_eq!(dense.max6_rank, sample_row.max6_rank);
        for (expected, got) in dense.max6.iter().zip(sample_row.max6.iter()) {
            assert!(
                (expected - got).abs() < 1.0e-12,
                "Max6 mismatch: {expected} vs {got}"
            );
        }
        for (expected, got) in dense.dosage.iter().zip(sample_row.dosage.iter()) {
            assert!(
                (expected - got).abs() < 1.0e-12,
                "dosage mismatch: {expected} vs {got}"
            );
        }
        for (expected, got) in dense.logic.iter().zip(sample_row.logic.iter()) {
            for (expected, got) in expected.iter().zip(got.iter()) {
                assert!(
                    (expected - got).abs() < 1.0e-12,
                    "logic mismatch: {expected} vs {got}"
                );
            }
        }
        assert_eq!(
            canonical_weighted_c_kernel_name("sample_row_sparse"),
            Some("sample_row_sparse")
        );
    }

    #[test]
    fn binary_sparse_c_rejects_nonbinary_genotypes() {
        let x = vec![1.0, 0.5, 1.0, 0.0, 1.0, 0.0];
        let py = vec![1.0, 2.0];
        let (pair_i, pair_j) = pair_indices(3);
        let error =
            weighted_interaction_py_binary_sparse(&x, &py, 2, 3, &pair_i, &pair_j).unwrap_err();
        assert!(error.contains("genotype values in {0, 1}"));
    }

    #[test]
    fn sample_row_sparse_c_builds_packed_upper_triangle_and_work_metrics() {
        let x = vec![
            1.0, 0.0, 1.0, // row 0
            1.0, 1.0, 0.0, // row 1
            0.0, 1.0, 1.0, // row 2
            0.0, 0.0, 1.0, // row 3
        ];
        let py = vec![0.5, -1.0, 2.0, 3.0];

        let marker_py = vec![-0.5, 1.0, 5.5];
        let result = sample_row_sparse_weighted_c(&x, &py, &marker_py, 4, 3).unwrap();

        assert_eq!(result.c_upper, vec![-0.5, -1.0, 0.5, 1.0, 2.0, 5.5]);
        assert_eq!(packed_upper_index(3, 0, 0), 0);
        assert_eq!(packed_upper_index(3, 0, 2), 2);
        assert_eq!(packed_upper_index(3, 1, 1), 3);
        assert_eq!(packed_upper_index(3, 1, 2), 4);
        assert_eq!(packed_upper_index(3, 2, 2), 5);
        assert_eq!(result.w_anchor, 6);
        assert_eq!(result.w_row, 3);
        assert_eq!(result.w_dense, 12);
    }

    #[test]
    fn subset_q_reports_internal_stage_timing() {
        let x = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let n_samples = 4usize;
        let n_markers = 3usize;
        let mut p_c = vec![0.0; n_samples * n_samples];
        for row in 0..n_samples {
            p_c[row * n_samples + row] = 1.0;
        }

        let (_, timing) = phase3_geometry_timed(&x, n_samples, n_markers, &p_c, 2, true).unwrap();
        let q_timing = timing
            .subset_q_timing
            .expect("subset q timing should be present");
        let stages = [
            q_timing.support_seconds,
            q_timing.psub_seconds,
            q_timing.xsub_seconds,
            q_timing.gemm_seconds,
            q_timing.reduce_seconds,
            q_timing.alloc_zero_seconds,
        ];
        assert!(stages
            .iter()
            .all(|&seconds| seconds.is_finite() && seconds >= 0.0));
        assert!(timing.q_seconds.is_finite() && timing.q_seconds >= 0.0);
        assert_eq!(q_timing.workspace_allocations, 3);
    }

    #[test]
    fn phase1_ah_and_scores_match_pairwise_reference() {
        let x = vec![
            0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        let n = 5usize;
        let m = 4usize;
        let pc = vec![
            1.2, 0.1, 0.0, 0.0, 0.0, 0.1, 1.1, 0.2, 0.0, 0.0, 0.0, 0.2, 1.3, 0.1, 0.0, 0.0, 0.0,
            0.1, 1.4, 0.2, 0.0, 0.0, 0.0, 0.2, 1.5,
        ];
        let y = vec![0.2, -1.0, 0.5, 1.7, -0.3];

        let old = reference_geometry(&x, n, m, &pc, 2).unwrap();
        let new = phase1_geometry(&x, n, m, &pc, 2).unwrap();

        assert_eq!(old.pair_i, new.pair_i);
        assert_eq!(old.pair_j, new.pair_j);
        assert_eq!(old.q, new.q);
        for (lhs, rhs) in old.a.iter().zip(new.a.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-12, "A mismatch: {lhs} vs {rhs}");
        }
        for (lhs, rhs) in old.h.iter().zip(new.h.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-12, "H mismatch: {lhs} vs {rhs}");
        }

        let old_scores = phase1_scores(&old, &x, n, m, &pc, &y, Some((1, 2))).unwrap();
        let new_scores = phase1_scores(&new, &x, n, m, &pc, &y, Some((1, 2))).unwrap();
        assert_eq!(old_scores.max6_rank, new_scores.max6_rank);
        for (lhs, rhs) in old_scores.max6.iter().zip(new_scores.max6.iter()) {
            assert!(
                (lhs - rhs).abs() < 1.0e-12,
                "score mismatch: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn subset_q_matches_old_q_and_preserves_pair_order_after_support_sort() {
        let x = vec![
            1.0, 1.0, 1.0, 1.0, // support: 3, 4, 4, 3
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
        ];
        let n = 6usize;
        let m = 4usize;
        let mut p_c = vec![0.0; n * n];
        for row in 0..n {
            p_c[row * n + row] = 1.0 + 0.1 * row as f64;
            if row + 1 < n {
                p_c[row * n + row + 1] = 0.03;
                p_c[(row + 1) * n + row] = 0.03;
            }
        }
        let (pair_i, pair_j) = pair_indices(m);
        let old_q = old_q_kernel(&x, n, m, &p_c, &pair_i, &pair_j, 2).unwrap();
        let original = subset_q_kernel(&x, n, m, &p_c, &pair_i, &pair_j, false).unwrap();
        let sorted = subset_q_kernel(&x, n, m, &p_c, &pair_i, &pair_j, true).unwrap();

        for ((expected, original), sorted) in
            old_q.iter().zip(original.q.iter()).zip(sorted.q.iter())
        {
            assert!(
                (expected - original).abs() < 1.0e-12,
                "original subset q mismatch"
            );
            assert!(
                (expected - sorted).abs() < 1.0e-12,
                "sorted subset q mismatch"
            );
        }
        assert!((original.work_subset - 75.0).abs() < 1.0e-12);
        assert!((sorted.work_subset - 61.0).abs() < 1.0e-12);
        assert!(sorted.work_subset < original.work_subset);
        assert!((original.work_dense - (n * n * old_q.len()) as f64).abs() < 1.0e-12);
        assert!((sorted.work_ratio - 61.0 / 216.0).abs() < 1.0e-12);
    }

    #[test]
    fn spectral_q_matches_old_q_and_max6_scores() {
        let x = vec![
            0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        let n = 5usize;
        let m = 4usize;
        let eigenvalues = vec![0.2, 0.8, 1.7, 3.1, 5.0];
        let mut eigenvectors = vec![0.0; n * n];
        for index in 0..n {
            eigenvectors[index * n + index] = 1.0;
        }
        let cosine = 0.8;
        let sine = 0.6;
        eigenvectors[0] = cosine;
        eigenvectors[1] = -sine;
        eigenvectors[n] = sine;
        eigenvectors[n + 1] = cosine;
        let lambda_vc = 0.7;
        let weights: Vec<f64> = eigenvalues
            .iter()
            .map(|&value| 1.0 / (value + lambda_vc))
            .collect();
        let mut one_rot = vec![0.0; n];
        for row in 0..n {
            for coordinate in 0..n {
                one_rot[coordinate] += eigenvectors[row * n + coordinate];
            }
        }
        let weighted_one: Vec<f64> = one_rot
            .iter()
            .zip(weights.iter())
            .map(|(&value, &weight)| value * weight)
            .collect();
        let information: f64 = one_rot
            .iter()
            .zip(weighted_one.iter())
            .map(|(&left, &right)| left * right)
            .sum();
        let mut pc_one = vec![0.0; n];
        for row in 0..n {
            for coordinate in 0..n {
                pc_one[row] += eigenvectors[row * n + coordinate] * weighted_one[coordinate];
            }
        }
        let mut p_c = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..n {
                for coordinate in 0..n {
                    p_c[row * n + col] += eigenvectors[row * n + coordinate]
                        * weights[coordinate]
                        * eigenvectors[col * n + coordinate];
                }
                p_c[row * n + col] -= pc_one[row] * pc_one[col] / information;
            }
        }
        let y = vec![0.2, -1.0, 0.5, 1.7, -0.3];

        let old = phase1_geometry(&x, n, m, &p_c, 2).unwrap();
        let spectral_q = spectral_q_kernel(
            &x,
            n,
            m,
            &eigenvalues,
            &eigenvectors,
            lambda_vc,
            &old.pair_i,
            &old.pair_j,
            2,
        )
        .unwrap();
        for (lhs, rhs) in old.q.iter().zip(spectral_q.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-12, "q mismatch: {lhs} vs {rhs}");
        }

        let spectral = super::Phase1Geometry {
            q: spectral_q,
            ..old.clone()
        };
        let old_scores = phase1_scores(&old, &x, n, m, &p_c, &y, Some((1, 2))).unwrap();
        let spectral_scores = phase1_scores(&spectral, &x, n, m, &p_c, &y, Some((1, 2))).unwrap();
        assert_eq!(old_scores.max6_rank, spectral_scores.max6_rank);
        for (lhs, rhs) in old_scores.max6.iter().zip(spectral_scores.max6.iter()) {
            assert!(
                (lhs - rhs).abs() < 1.0e-12,
                "score mismatch: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn min_support_encoding_flips_only_columns_above_half_support() {
        let x = vec![
            1.0, 1.0, 1.0, 0.0, // support: 5, 3, 5, 4
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let encoded = min_support_encode(&x, 8, 4).unwrap();
        assert_eq!(encoded.flipped, vec![true, false, true, false]);
        let transformed_support = (0..4)
            .map(|marker| {
                (0..8)
                    .filter(|&row| encoded.x[row * 4 + marker] == 1.0)
                    .count()
            })
            .collect::<Vec<_>>();
        assert_eq!(transformed_support, vec![3, 3, 3, 4]);
        for value in &encoded.x {
            assert!(*value == 0.0 || *value == 1.0);
        }
    }

    #[test]
    fn min_support_logic_mapping_returns_original_cell_labels() {
        assert_eq!(
            (0..5)
                .map(|gate| remap_logic_gate_index(gate, false, false))
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        assert_eq!(
            (0..5)
                .map(|gate| remap_logic_gate_index(gate, true, false))
                .collect::<Vec<_>>(),
            vec![2, 3, 0, 1, 4]
        );
        assert_eq!(
            (0..5)
                .map(|gate| remap_logic_gate_index(gate, false, true))
                .collect::<Vec<_>>(),
            vec![1, 0, 3, 2, 4]
        );
        assert_eq!(
            (0..5)
                .map(|gate| remap_logic_gate_index(gate, true, true))
                .collect::<Vec<_>>(),
            vec![3, 2, 1, 0, 4]
        );
    }

    #[test]
    fn min_support_recode_preserves_max6_scores_with_intercept_projection() {
        let x = vec![
            1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let n = 8usize;
        let m = 4usize;
        let mut p_c = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..n {
                p_c[row * n + col] = if row == col { 7.0 / 8.0 } else { -1.0 / 8.0 };
            }
        }
        let y = vec![0.2, -1.0, 0.5, 1.7, -0.3, 0.9, -0.4, 0.6];
        let encoded = min_support_encode(&x, n, m).unwrap();
        let original_geometry = phase1_geometry(&x, n, m, &p_c, 2).unwrap();
        let encoded_geometry = phase1_geometry(&encoded.x, n, m, &p_c, 2).unwrap();
        let original = phase1_scores(&original_geometry, &x, n, m, &p_c, &y, None).unwrap();
        let encoded_scores =
            phase1_scores(&encoded_geometry, &encoded.x, n, m, &p_c, &y, None).unwrap();

        for (lhs, rhs) in original.max6.iter().zip(encoded_scores.max6.iter()) {
            if lhs.is_finite() || rhs.is_finite() {
                assert!((lhs - rhs).abs() < 1.0e-10, "Max6 mismatch: {lhs} vs {rhs}");
            } else {
                assert_eq!(*lhs, *rhs);
            }
        }
        for (lhs, rhs) in original.dosage.iter().zip(encoded_scores.dosage.iter()) {
            if lhs.is_finite() || rhs.is_finite() {
                assert!(
                    (lhs - rhs).abs() < 1.0e-10,
                    "dosage mismatch: {lhs} vs {rhs}"
                );
            } else {
                assert_eq!(*lhs, *rhs);
            }
        }
        for (pair, (original_logic, encoded_logic)) in original
            .logic
            .iter()
            .zip(encoded_scores.logic.iter())
            .enumerate()
        {
            let i = original_geometry.pair_i[pair];
            let j = original_geometry.pair_j[pair];
            for gate in 0..5 {
                let encoded_gate =
                    remap_logic_gate_index(gate, encoded.flipped[i], encoded.flipped[j]);
                let lhs = original_logic[encoded_gate];
                let rhs = encoded_logic[gate];
                if lhs.is_finite() || rhs.is_finite() {
                    assert!(
                        (lhs - rhs).abs() < 1.0e-10,
                        "logic mismatch pair={pair} gate={gate}: {lhs} vs {rhs}"
                    );
                } else {
                    assert_eq!(lhs, rhs);
                }
            }
        }
    }

    #[test]
    fn subset_min_support_is_canonical_with_legacy_alias() {
        assert_eq!(
            canonical_q_kernel_name("subset_min_support"),
            Some("subset_min_support")
        );
        assert_eq!(
            canonical_q_kernel_name("subset_minor_support"),
            Some("subset_min_support")
        );
        assert_eq!(canonical_q_kernel_name("subset_support_sorted"), None);
    }

    #[test]
    fn ordinary_full_exact_default_is_mincell_and_spectral_is_explicit_compatibility() {
        assert_eq!(
            resolve_full_exact_kernel_names(None, None, false),
            ("mincell_exact", "sample_row_sparse")
        );
        assert_eq!(
            resolve_full_exact_kernel_names(None, None, true),
            ("spectral", "dense_reference")
        );
    }

    #[test]
    fn max6_channel_label_includes_dosage_after_logic_channels() {
        let logic = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(best_max6_channel_name(&logic, 6.0), "DOSAGE_PRODUCT");
        assert_eq!(best_max6_channel_name(&logic, 5.0), "XOR");
        assert_eq!(
            best_max6_channel_name(&[f64::NAN, 3.0, 3.0, 3.0, 3.0], 2.0),
            "AND01"
        );
        assert_eq!(
            best_max6_channel_name(&[f64::NEG_INFINITY; 5], f64::NEG_INFINITY),
            "NONE"
        );
    }
}
