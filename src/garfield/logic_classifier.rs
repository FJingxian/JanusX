//! Post-processing summaries for binary Logic5 pair candidates.
//!
//! Pair discovery remains owned by `all_pair.rs`.  This module deliberately
//! runs only after a bounded Top-K has been selected: it summarizes the four
//! binary genotype cells and classifies their phenotype means against the
//! weighted, centered templates for the four AND orientations and XOR.
//!
//! The cell order is always `[00, 01, 10, 11]`.  A 0/2 marker is mapped to
//! binary states 0/1, while heterozygotes and missing values are excluded by
//! the strict valid mask rather than imputed.

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const N_CELLS: usize = 4;
const N_TEMPLATES: usize = 5;
const CLASSIFICATION_EPS: f64 = 1.0e-12;
const AMBIGUITY_MARGIN: f64 = 0.05;
const VALUE_TOL: f64 = 1.0e-8;

/// How a marker's binary state is encoded before assigning a sample to a
/// 2-by-2 cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BinaryEncoding {
    /// Values 0 and 1 represent the two states.
    ZeroOne,
    /// Dosage values 0 and 2 represent the two homozygous states; value 1 is
    /// a heterozygote and is therefore invalid for this strict classifier.
    ZeroTwo,
    /// Infer independently for each marker.  A finite observed value 2 makes
    /// the marker 0/2; otherwise 0/1 is used.  Ambiguous all-zero markers can
    /// be handled explicitly with `ZeroTwo` by callers that need that policy.
    Auto,
}

impl BinaryEncoding {
    pub(crate) fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "01" | "0/1" | "binary" => Ok(Self::ZeroOne),
            "02" | "0/2" | "homozygous" => Ok(Self::ZeroTwo),
            other => Err(format!(
                "encoding must be one of auto, 01, or 02; got {other}"
            )),
        }
    }
}

/// The architecture template selected from the four cell means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TemplateLabel {
    And00,
    And01,
    And10,
    And11,
    Xor,
    Unresolved,
}

impl TemplateLabel {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::And00 => "AND_00",
            Self::And01 => "AND_01",
            Self::And10 => "AND_10",
            Self::And11 => "AND_11",
            Self::Xor => "XOR",
            Self::Unresolved => "UNRESOLVED",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EffectDirection {
    Positive,
    Negative,
    Neutral,
}

impl EffectDirection {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Positive => "positive",
            Self::Negative => "negative",
            Self::Neutral => "neutral",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClassificationStatus {
    Classified,
    Ambiguous,
    InsufficientSupport,
    Unresolved,
}

impl ClassificationStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Classified => "classified",
            Self::Ambiguous => "ambiguous",
            Self::InsufficientSupport => "insufficient_support",
            Self::Unresolved => "unresolved",
        }
    }
}

/// Four-cell summary and template classification for one pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PairCellSummary {
    /// Counts and means use the fixed order `[00, 01, 10, 11]`.
    pub(crate) counts: [usize; N_CELLS],
    pub(crate) means: [f64; N_CELLS],
    pub(crate) template_label: TemplateLabel,
    /// Signed weighted-centered correlation with the winning template.
    pub(crate) template_score: f64,
    /// Difference between the two largest signed template correlations.
    pub(crate) template_margin: f64,
    pub(crate) effect_direction: EffectDirection,
    pub(crate) classification_status: ClassificationStatus,
    pub(crate) n_valid: usize,
}

const TEMPLATE_LABELS: [TemplateLabel; N_TEMPLATES] = [
    TemplateLabel::And00,
    TemplateLabel::And01,
    TemplateLabel::And10,
    TemplateLabel::And11,
    TemplateLabel::Xor,
];

const TEMPLATES: [[f64; N_CELLS]; N_TEMPLATES] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 0.0],
];

#[inline]
fn close_to(value: f64, target: f64) -> bool {
    (value - target).abs() <= VALUE_TOL
}

fn infer_encoding(values: &[f64], valid_mask: Option<&[bool]>) -> BinaryEncoding {
    let has_two = values.iter().enumerate().any(|(idx, value)| {
        valid_mask.map(|mask| mask[idx]).unwrap_or(true)
            && value.is_finite()
            && close_to(*value, 2.0)
    });
    if has_two {
        BinaryEncoding::ZeroTwo
    } else {
        BinaryEncoding::ZeroOne
    }
}

#[inline]
fn encode_state(value: f64, encoding: BinaryEncoding) -> Option<u8> {
    if !value.is_finite() {
        return None;
    }
    match encoding {
        BinaryEncoding::ZeroOne => {
            if close_to(value, 0.0) {
                Some(0)
            } else if close_to(value, 1.0) {
                Some(1)
            } else {
                None
            }
        }
        BinaryEncoding::ZeroTwo => {
            if close_to(value, 0.0) {
                Some(0)
            } else if close_to(value, 2.0) {
                Some(1)
            } else {
                None
            }
        }
        BinaryEncoding::Auto => unreachable!("auto encoding must be resolved per marker"),
    }
}

#[inline]
fn weighted_centered_correlation(
    means: &[f64; N_CELLS],
    counts: &[usize; N_CELLS],
    template: &[f64; N_CELLS],
) -> Option<f64> {
    let total_weight = counts.iter().sum::<usize>() as f64;
    if total_weight <= 0.0 {
        return None;
    }
    let observed_center = counts
        .iter()
        .zip(means.iter())
        .map(|(&weight, &mean)| weight as f64 * mean)
        .sum::<f64>()
        / total_weight;
    let template_center = counts
        .iter()
        .zip(template.iter())
        .map(|(&weight, &value)| weight as f64 * value)
        .sum::<f64>()
        / total_weight;
    let mut numerator = 0.0;
    let mut observed_ss = 0.0;
    let mut template_ss = 0.0;
    for ((&count, &mean), &value) in counts.iter().zip(means.iter()).zip(template.iter()) {
        let weight = count as f64;
        let observed = mean - observed_center;
        let expected = value - template_center;
        numerator += weight * observed * expected;
        observed_ss += weight * observed * observed;
        template_ss += weight * expected * expected;
    }
    let denominator = (observed_ss * template_ss).sqrt();
    if !denominator.is_finite() || denominator <= CLASSIFICATION_EPS {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn classify_cell_means(
    counts: [usize; N_CELLS],
    means: [f64; N_CELLS],
    n_valid: usize,
    min_cell_count: usize,
) -> PairCellSummary {
    if min_cell_count > 0 && counts.iter().any(|&count| count < min_cell_count) {
        return PairCellSummary {
            counts,
            means,
            template_label: TemplateLabel::Unresolved,
            template_score: f64::NAN,
            template_margin: f64::NAN,
            effect_direction: EffectDirection::Neutral,
            classification_status: ClassificationStatus::InsufficientSupport,
            n_valid,
        };
    }

    let mut correlations = [f64::NAN; N_TEMPLATES];
    for (idx, template) in TEMPLATES.iter().enumerate() {
        correlations[idx] =
            weighted_centered_correlation(&means, &counts, template).unwrap_or(f64::NAN);
    }
    // Logic5 discovers a positive high-residual carrier set.  Keep the
    // interpretation layer consistent with that validated contract: choose
    // the template with the largest *signed* centered correlation.  Using
    // absolute correlation here would let a negative, unrelated template
    // (for example an inverted XOR pattern) displace a positive AND pattern.
    let mut best_idx = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut second_score = f64::NEG_INFINITY;
    for (idx, &score) in correlations.iter().enumerate() {
        if !score.is_finite() {
            continue;
        }
        if score > best_score {
            second_score = best_score;
            best_score = score;
            best_idx = Some(idx);
        } else if score > second_score {
            second_score = score;
        }
    }
    let Some(best_idx) = best_idx else {
        return PairCellSummary {
            counts,
            means,
            template_label: TemplateLabel::Unresolved,
            template_score: f64::NAN,
            template_margin: f64::NAN,
            effect_direction: EffectDirection::Neutral,
            classification_status: ClassificationStatus::Unresolved,
            n_valid,
        };
    };
    let template_score = correlations[best_idx];
    // With a complete four-cell table every template is finite.  Keep the
    // diagnostic well-defined even for a caller that deliberately allows an
    // incomplete table (`min_cell_count=0`): a missing runner-up contributes
    // zero rather than propagating an artificial infinity.
    let second_score = if second_score.is_finite() {
        second_score
    } else {
        0.0
    };
    let template_margin = (best_score - second_score).max(0.0);
    let effect_direction = if template_score > CLASSIFICATION_EPS {
        EffectDirection::Positive
    } else if template_score < -CLASSIFICATION_EPS {
        EffectDirection::Negative
    } else {
        EffectDirection::Neutral
    };
    let classification_status = if best_score.abs() <= CLASSIFICATION_EPS {
        ClassificationStatus::Unresolved
    } else if template_margin < AMBIGUITY_MARGIN {
        ClassificationStatus::Ambiguous
    } else {
        ClassificationStatus::Classified
    };
    PairCellSummary {
        counts,
        means,
        template_label: if classification_status == ClassificationStatus::Unresolved {
            TemplateLabel::Unresolved
        } else {
            TEMPLATE_LABELS[best_idx]
        },
        template_score,
        template_margin,
        effect_direction,
        classification_status,
        n_valid,
    }
}

fn validate_pair_inputs(
    g1: &[f64],
    g2: &[f64],
    y: &[f64],
    valid_mask: Option<&[bool]>,
) -> Result<(), String> {
    if g1.len() != g2.len() || g1.len() != y.len() {
        return Err(format!(
            "pair cell summary length mismatch: g1={}, g2={}, y={}",
            g1.len(),
            g2.len(),
            y.len()
        ));
    }
    if let Some(mask) = valid_mask {
        if mask.len() != y.len() {
            return Err(format!(
                "pair cell summary valid mask length mismatch: mask={}, expected={}",
                mask.len(),
                y.len()
            ));
        }
    }
    Ok(())
}

/// Summarize one binary pair. Invalid genotype values (including 1 under the
/// explicit 0/2 encoding), non-finite phenotype values, and false mask values
/// are all excluded from the same strict valid sample set.
pub(crate) fn summarize_binary_pair(
    g1: &[f64],
    g2: &[f64],
    y: &[f64],
    valid_mask: Option<&[bool]>,
    encoding_g1: BinaryEncoding,
    encoding_g2: BinaryEncoding,
    min_cell_count: usize,
) -> Result<PairCellSummary, String> {
    validate_pair_inputs(g1, g2, y, valid_mask)?;
    let resolved_g1 = if encoding_g1 == BinaryEncoding::Auto {
        infer_encoding(g1, valid_mask)
    } else {
        encoding_g1
    };
    let resolved_g2 = if encoding_g2 == BinaryEncoding::Auto {
        infer_encoding(g2, valid_mask)
    } else {
        encoding_g2
    };
    let mut counts = [0usize; N_CELLS];
    let mut sums = [0.0f64; N_CELLS];
    let mut n_valid = 0usize;
    for idx in 0..y.len() {
        if valid_mask.map(|mask| !mask[idx]).unwrap_or(false) {
            continue;
        }
        let Some(state_g1) = encode_state(g1[idx], resolved_g1) else {
            continue;
        };
        let Some(state_g2) = encode_state(g2[idx], resolved_g2) else {
            continue;
        };
        let response = y[idx];
        if !response.is_finite() {
            continue;
        }
        let cell = match (state_g1, state_g2) {
            (0, 0) => 0,
            (0, 1) => 1,
            (1, 0) => 2,
            (1, 1) => 3,
            _ => unreachable!("binary state must be 0 or 1"),
        };
        counts[cell] = counts[cell].saturating_add(1);
        sums[cell] += response;
        n_valid = n_valid.saturating_add(1);
    }
    let means = std::array::from_fn(|cell| {
        if counts[cell] == 0 {
            f64::NAN
        } else {
            sums[cell] / counts[cell] as f64
        }
    });
    Ok(classify_cell_means(counts, means, n_valid, min_cell_count))
}

/// Batch post-processing for marker-major genotype matrices.  It is intended
/// for the bounded Top-K output of a pair discovery backend, not for the hot
/// pair scan itself.
pub(crate) fn summarize_binary_pairs_marker_major(
    genotypes: &[f64],
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    pairs: &[(usize, usize)],
    valid_mask: Option<&[bool]>,
    encoding: BinaryEncoding,
    min_cell_count: usize,
) -> Result<Vec<PairCellSummary>, String> {
    if y.len() != n_samples {
        return Err(format!(
            "pair cell summary y length={} expected n_samples={n_samples}",
            y.len()
        ));
    }
    let expected = n_markers
        .checked_mul(n_samples)
        .ok_or_else(|| "pair cell summary genotype matrix size overflow".to_string())?;
    if genotypes.len() != expected {
        return Err(format!(
            "pair cell summary genotype length={} expected={expected} ({n_markers}x{n_samples})",
            genotypes.len()
        ));
    }
    if let Some(mask) = valid_mask {
        if mask.len() != expected {
            return Err(format!(
                "pair cell summary valid mask length={} expected={expected}",
                mask.len()
            ));
        }
    }
    let mut summaries = Vec::with_capacity(pairs.len());
    for &(first, second) in pairs.iter() {
        if first >= n_markers || second >= n_markers || first == second {
            return Err(format!(
                "pair cell summary pair ({first},{second}) outside marker range 0..{n_markers}"
            ));
        }
        let first_start = first * n_samples;
        let second_start = second * n_samples;
        let g1 = &genotypes[first_start..first_start + n_samples];
        let g2 = &genotypes[second_start..second_start + n_samples];
        let pair_mask = valid_mask.map(|mask| {
            (0..n_samples)
                .map(|idx| mask[first_start + idx] && mask[second_start + idx])
                .collect::<Vec<_>>()
        });
        summaries.push(summarize_binary_pair(
            g1,
            g2,
            y,
            pair_mask.as_deref(),
            encoding,
            encoding,
            min_cell_count,
        )?);
    }
    Ok(summaries)
}

pub(crate) fn array2_to_vec(array: &PyReadonlyArray2<'_, f64>) -> Vec<f64> {
    let view = array.as_array();
    let shape = view.shape();
    let mut out = Vec::with_capacity(view.len());
    for row in 0..shape[0] {
        for column in 0..shape[1] {
            out.push(view[[row, column]]);
        }
    }
    out
}

pub(crate) fn bool_array2_to_vec(array: &PyReadonlyArray2<'_, bool>) -> Vec<bool> {
    let view = array.as_array();
    let shape = view.shape();
    let mut out = Vec::with_capacity(view.len());
    for row in 0..shape[0] {
        for column in 0..shape[1] {
            out.push(view[[row, column]]);
        }
    }
    out
}

pub(crate) fn set_summary_columns<'py>(
    out: &Bound<'py, PyDict>,
    summaries: &[PairCellSummary],
) -> PyResult<()> {
    out.set_item(
        "n00",
        summaries
            .iter()
            .map(|summary| summary.counts[0])
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "n01",
        summaries
            .iter()
            .map(|summary| summary.counts[1])
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "n10",
        summaries
            .iter()
            .map(|summary| summary.counts[2])
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "n11",
        summaries
            .iter()
            .map(|summary| summary.counts[3])
            .collect::<Vec<_>>(),
    )?;
    for (name, cell) in [
        ("mean00", 0usize),
        ("mean01", 1),
        ("mean10", 2),
        ("mean11", 3),
    ] {
        out.set_item(
            name,
            summaries
                .iter()
                .map(|summary| summary.means[cell])
                .collect::<Vec<_>>(),
        )?;
    }
    out.set_item(
        "template_label",
        summaries
            .iter()
            .map(|summary| summary.template_label.as_str())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "template_score",
        summaries
            .iter()
            .map(|summary| summary.template_score)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "template_margin",
        summaries
            .iter()
            .map(|summary| summary.template_margin)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "effect_direction",
        summaries
            .iter()
            .map(|summary| summary.effect_direction.as_str())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "classification_status",
        summaries
            .iter()
            .map(|summary| summary.classification_status.as_str())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "n_valid",
        summaries
            .iter()
            .map(|summary| summary.n_valid)
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}

pub(crate) fn summaries_to_py_dict<'py>(
    py: Python<'py>,
    pairs: &[(usize, usize)],
    summaries: &[PairCellSummary],
) -> PyResult<Bound<'py, PyDict>> {
    if pairs.len() != summaries.len() {
        return Err(PyValueError::new_err(format!(
            "pair cell summary output mismatch: pairs={}, summaries={}",
            pairs.len(),
            summaries.len()
        )));
    }
    let out = PyDict::new(py);
    out.set_item("first", pairs.iter().map(|pair| pair.0).collect::<Vec<_>>())?;
    out.set_item(
        "second",
        pairs.iter().map(|pair| pair.1).collect::<Vec<_>>(),
    )?;
    set_summary_columns(&out, summaries)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        summarize_binary_pair, BinaryEncoding, ClassificationStatus, EffectDirection, TemplateLabel,
    };

    #[test]
    fn classifies_positive_and00_with_weighted_centering() {
        // Cell order is explicitly [00, 01, 10, 11].
        let g1 = [0.0, 0.0, 1.0, 1.0];
        let g2 = [0.0, 1.0, 0.0, 1.0];
        let y = [10.0, 0.0, 0.0, 0.0];
        let summary = summarize_binary_pair(
            &g1,
            &g2,
            &y,
            None,
            BinaryEncoding::ZeroOne,
            BinaryEncoding::ZeroOne,
            1,
        )
        .unwrap();
        assert_eq!(summary.counts, [1, 1, 1, 1]);
        assert_eq!(summary.means, [10.0, 0.0, 0.0, 0.0]);
        assert_eq!(summary.template_label, TemplateLabel::And00);
        assert_eq!(summary.effect_direction, EffectDirection::Positive);
        assert_eq!(
            summary.classification_status,
            ClassificationStatus::Classified
        );
        assert!((summary.template_score - 1.0).abs() < 1e-12);
        assert!(summary.template_margin > 0.3);
    }

    #[test]
    fn classifies_all_and_orientations_from_cell_means() {
        let g1 = [0.0, 0.0, 1.0, 1.0];
        let g2 = [0.0, 1.0, 0.0, 1.0];
        let cases = [
            ([10.0, 0.0, 0.0, 0.0], TemplateLabel::And00),
            ([0.0, 10.0, 0.0, 0.0], TemplateLabel::And01),
            ([0.0, 0.0, 10.0, 0.0], TemplateLabel::And10),
            ([0.0, 0.0, 0.0, 10.0], TemplateLabel::And11),
        ];
        for (y, expected) in cases {
            let summary = summarize_binary_pair(
                &g1,
                &g2,
                &y,
                None,
                BinaryEncoding::ZeroOne,
                BinaryEncoding::ZeroOne,
                1,
            )
            .unwrap();
            assert_eq!(summary.template_label, expected);
            assert_eq!(
                summary.classification_status,
                ClassificationStatus::Classified
            );
            assert!(summary.template_score > 0.99);
        }
    }

    #[test]
    fn classifies_xor_with_signed_template_score() {
        let g1 = [0.0, 0.0, 1.0, 1.0];
        let g2 = [0.0, 1.0, 0.0, 1.0];
        let y = [0.0, 10.0, 10.0, 0.0];
        let summary = summarize_binary_pair(
            &g1,
            &g2,
            &y,
            None,
            BinaryEncoding::ZeroOne,
            BinaryEncoding::ZeroOne,
            1,
        )
        .unwrap();
        assert_eq!(summary.template_label, TemplateLabel::Xor);
        assert_eq!(summary.effect_direction, EffectDirection::Positive);
        assert_eq!(
            summary.classification_status,
            ClassificationStatus::Classified
        );
        assert!((summary.template_score - 1.0).abs() < 1e-12);

        // The discovery path is positive-gain by definition.  The summary
        // therefore uses the signed (high-cell) template score; an inverted
        // phenotype is intentionally interpreted as its high-cell pattern,
        // rather than letting a negative unrelated template win by absolute
        // magnitude.
    }

    #[test]
    fn strict_zero_two_encoding_excludes_heterozygote_and_missing() {
        let g1 = [0.0, 0.0, 1.0, 2.0, f64::NAN];
        let g2 = [0.0, 2.0, 0.0, 2.0, 0.0];
        let y = [1.0, 2.0, 3.0, 4.0, 99.0];
        let summary = summarize_binary_pair(
            &g1,
            &g2,
            &y,
            None,
            BinaryEncoding::ZeroTwo,
            BinaryEncoding::ZeroTwo,
            1,
        )
        .unwrap();
        assert_eq!(summary.counts, [1, 1, 0, 1]);
        assert_eq!(summary.n_valid, 3);
        assert_eq!(
            summary.classification_status,
            ClassificationStatus::InsufficientSupport
        );
        assert_eq!(summary.template_label, TemplateLabel::Unresolved);
    }

    #[test]
    fn explicit_valid_mask_is_combined_with_genotype_validity() {
        let g1 = [0.0, 0.0, 1.0, 1.0];
        let g2 = [0.0, 1.0, 0.0, 1.0];
        let y = [10.0, 0.0, 0.0, 0.0];
        let valid = [true, false, true, true];
        let summary = summarize_binary_pair(
            &g1,
            &g2,
            &y,
            Some(&valid),
            BinaryEncoding::ZeroOne,
            BinaryEncoding::ZeroOne,
            1,
        )
        .unwrap();
        assert_eq!(summary.n_valid, 3);
        assert_eq!(summary.counts, [1, 0, 1, 1]);
        assert_eq!(
            summary.classification_status,
            ClassificationStatus::InsufficientSupport
        );
    }

    #[test]
    fn flat_cell_means_are_unresolved_not_a_random_template() {
        let g1 = [0.0, 0.0, 1.0, 1.0];
        let g2 = [0.0, 1.0, 0.0, 1.0];
        let y = [3.0, 3.0, 3.0, 3.0];
        let summary = summarize_binary_pair(
            &g1,
            &g2,
            &y,
            None,
            BinaryEncoding::ZeroOne,
            BinaryEncoding::ZeroOne,
            1,
        )
        .unwrap();
        assert_eq!(summary.template_label, TemplateLabel::Unresolved);
        assert_eq!(
            summary.classification_status,
            ClassificationStatus::Unresolved
        );
    }

    #[test]
    fn marker_major_batch_preserves_pair_order_and_strict_masks() {
        let genotypes = vec![
            0.0, 0.0, 1.0, 1.0, // marker 0
            0.0, 1.0, 0.0, 1.0, // marker 1
            0.0, 2.0, 0.0, 2.0, // marker 2 (0/2)
        ];
        let y = [10.0, 0.0, 0.0, 0.0];
        let pairs = [(0, 1), (0, 2)];
        let summaries = super::summarize_binary_pairs_marker_major(
            &genotypes,
            3,
            4,
            &y,
            &pairs,
            None,
            BinaryEncoding::Auto,
            1,
        )
        .unwrap();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].template_label, TemplateLabel::And00);
        assert_eq!(summaries[0].counts, [1, 1, 1, 1]);
        // Auto resolves marker 2 as 0/2 and does not mistake the 2 dosage
        // for a third cell state.
        assert_eq!(summaries[1].counts, [1, 1, 1, 1]);
    }

    #[test]
    fn template_score_uses_cell_counts_as_weights() {
        let means = [4.0, 1.0, 2.0, 3.0];
        let counts = [1, 2, 3, 4];
        let weighted =
            super::weighted_centered_correlation(&means, &counts, &super::TEMPLATES[0]).unwrap();
        assert!((weighted - 0.5819143739626463).abs() < 1e-12);
        // The unweighted correlation would be 0.774596..., so this assertion
        // guards the cell-count weighting rather than only the winning label.
        let unweighted =
            super::weighted_centered_correlation(&means, &[1, 1, 1, 1], &super::TEMPLATES[0])
                .unwrap();
        assert!((unweighted - weighted).abs() > 0.1);
    }
}
