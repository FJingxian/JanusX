//! Shared four-cell sufficient statistics for the fixed Gate5/Gate6d screen.
//!
//! Cell order is always `[00, 01, 10, 11]`.  Gate5 uses the four AND gates
//! and XOR.  Gate6d adds the saturated one-df interaction contrast from the
//! same table; it is a residual-screen ranking score, not a standalone GRM
//! inference backend.

/// Cell order is always `[00, 01, 10, 11]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FourCellSufficientStats {
    pub(crate) counts: [usize; 4],
    pub(crate) sums: [f64; 4],
    pub(crate) n_valid: usize,
}

/// One residual-screen gate derived from four-cell sufficient statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FourCellGateScore {
    pub(crate) gate: FourCellGate,
    pub(crate) score: f64,
    pub(crate) signed_score: f64,
    pub(crate) n_hit: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FourCellGate {
    And00,
    And01,
    And10,
    And11,
    Xor,
    /// The standardized saturated 0/1-by-0/1 interaction contrast.
    ///
    /// This is a sixth proposal gate only.  It is not the exact GRM-aware
    /// dosage test, which remains a downstream inference step.
    DosageInteraction,
}

impl FourCellGate {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::And00 => "AND_00",
            Self::And01 => "AND_01",
            Self::And10 => "AND_10",
            Self::And11 => "AND_11",
            Self::Xor => "XOR",
            Self::DosageInteraction => "DOSAGE_INTERACTION",
        }
    }
}

/// Scores derived from one four-cell table for the residual-screen proposal
/// policies.  The Gate6d fields are `None` when a pair has an empty cell;
/// the product fields are populated only by the `nmax6` proposal path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gate5Gate6dScores {
    pub(crate) logic5: [FourCellGateScore; 5],
    pub(crate) dosage_interaction_contrast: Option<f64>,
    pub(crate) dosage_interaction_standardized_score: Option<f64>,
    /// EXPERIMENTAL nMax6-only sixth-channel screen score for
    /// `w=(x1-mean(x1))*(x2-mean(x2))` under intercept projection.
    ///
    /// This field is not part of the production Max6 definition. Do not mix
    /// nMax6 proposal results with the formal Max6 benchmark summaries.
    pub(crate) dosage_product_score: Option<f64>,
    pub(crate) dosage_product_signed_score: Option<f64>,
}

#[inline]
pub(crate) fn derive_gate5_gate6d_scores(stats: FourCellSufficientStats) -> Gate5Gate6dScores {
    Gate5Gate6dScores {
        logic5: derive_logic5_scores(stats),
        dosage_interaction_contrast: derive_dosage_interaction_contrast(stats),
        dosage_interaction_standardized_score: derive_standardized_dosage_interaction_score(stats),
        dosage_product_score: None,
        dosage_product_signed_score: None,
    }
}

#[inline]
pub(crate) fn derive_logic5_scores(stats: FourCellSufficientStats) -> [FourCellGateScore; 5] {
    let total = stats.sums.iter().sum::<f64>();
    let n_total = stats.n_valid as f64;
    let gate_data = [
        (FourCellGate::And00, stats.counts[0], stats.sums[0]),
        (FourCellGate::And01, stats.counts[1], stats.sums[1]),
        (FourCellGate::And10, stats.counts[2], stats.sums[2]),
        (FourCellGate::And11, stats.counts[3], stats.sums[3]),
        (
            FourCellGate::Xor,
            stats.counts[1].saturating_add(stats.counts[2]),
            stats.sums[1] + stats.sums[2],
        ),
    ];
    std::array::from_fn(|idx| {
        let (gate, n_hit, sum_hit) = gate_data[idx];
        if stats.n_valid == 0 || n_hit == 0 || n_hit >= stats.n_valid {
            return FourCellGateScore {
                gate,
                score: f64::NAN,
                signed_score: f64::NAN,
                n_hit,
            };
        }
        let n_miss = stats.n_valid - n_hit;
        let hit_f = n_hit as f64;
        let miss_f = n_miss as f64;
        let denominator = hit_f * miss_f / n_total;
        if !denominator.is_finite() || denominator <= 0.0 {
            return FourCellGateScore {
                gate,
                score: f64::NAN,
                signed_score: f64::NAN,
                n_hit,
            };
        }
        // Center within the pair-valid samples.  For an intercept-containing
        // residual vector this is exactly g' r; the centering also keeps the
        // masked path consistent when the valid subset has a nonzero sum.
        let centered_sum = sum_hit - (n_hit as f64 / n_total) * total;
        let signed_score = centered_sum / denominator.sqrt();
        // Keep the historical multiplication order for bitwise-compatible
        // Logic5 scores on the missing-free path.
        let score = n_total * centered_sum * centered_sum / (hit_f * miss_f);
        FourCellGateScore {
            gate,
            score,
            signed_score,
            n_hit,
        }
    })
}

/// Saturated 0/1-by-0/1 interaction contrast derived from the four cell
/// means.  With `d=(1,-1,-1,1)` and cell order `[00,01,10,11]`, this is
/// `delta = d' mu = mu11 - mu10 - mu01 + mu00`.
#[inline]
pub(crate) fn derive_dosage_interaction_contrast(stats: FourCellSufficientStats) -> Option<f64> {
    if stats.n_valid == 0 || stats.counts.iter().any(|&count| count == 0) {
        return None;
    }
    let means: [f64; 4] = std::array::from_fn(|idx| stats.sums[idx] / stats.counts[idx] as f64);
    let contrast = means[3] - means[2] - means[1] + means[0];
    contrast.is_finite().then_some(contrast)
}

/// Standardized one-df Gate6d proposal score.
///
/// Under a common residual variance, the variance multiplier of the contrast
/// is `sum(1 / n_cell)`.  The common residual variance scale is intentionally
/// omitted from this ranking score and belongs to formal inference.
#[inline]
pub(crate) fn derive_standardized_dosage_interaction_score(
    stats: FourCellSufficientStats,
) -> Option<f64> {
    let contrast = derive_dosage_interaction_contrast(stats)?;
    let inverse_count_sum = stats
        .counts
        .iter()
        .map(|&count| 1.0 / count as f64)
        .sum::<f64>();
    if !inverse_count_sum.is_finite() || inverse_count_sum <= 0.0 {
        return None;
    }
    let score = contrast * contrast / inverse_count_sum;
    score.is_finite().then_some(score)
}

/// Exact residual-screen score for the centered endpoint product.
///
/// The screen has already received a residual phenotype, so its common
/// intercept-only metric is `M0 = I - 11'/n`.  The cell table is sufficient to
/// evaluate `w=(x1-mean(x1))*(x2-mean(x2))`, its intercept projection, and the
/// signed statistic without reconstructing the individual-level vector.
/// Unlike the saturated Gate6d contrast, this score remains defined whenever
/// the centered product has non-zero variance, even if one observed cell is
/// empty.
#[inline]
pub(crate) fn derive_centered_product_screen_score(
    stats: FourCellSufficientStats,
) -> Option<(f64, f64)> {
    if stats.n_valid < 2 || stats.counts.iter().sum::<usize>() != stats.n_valid {
        return None;
    }
    let n = stats.n_valid as f64;
    let mean_first = (stats.counts[2] + stats.counts[3]) as f64 / n;
    let mean_second = (stats.counts[1] + stats.counts[3]) as f64 / n;
    let cell_weights = [
        mean_first * mean_second,
        -mean_first * (1.0 - mean_second),
        -(1.0 - mean_first) * mean_second,
        (1.0 - mean_first) * (1.0 - mean_second),
    ];
    let mean_weight = stats
        .counts
        .iter()
        .zip(cell_weights.iter())
        .map(|(&count, &weight)| count as f64 * weight)
        .sum::<f64>()
        / n;
    let total_y = stats.sums.iter().sum::<f64>();
    let numerator = stats
        .sums
        .iter()
        .zip(cell_weights.iter())
        .map(|(&sum, &weight)| sum * weight)
        .sum::<f64>()
        - mean_weight * total_y;
    let denominator = stats
        .counts
        .iter()
        .zip(cell_weights.iter())
        .map(|(&count, &weight)| {
            let centered = weight - mean_weight;
            count as f64 * centered * centered
        })
        .sum::<f64>();
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let signed_score = numerator / denominator.sqrt();
    let score = signed_score * signed_score;
    (score.is_finite() && signed_score.is_finite()).then_some((score, signed_score))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate5_and_gate6d_use_one_four_cell_table() {
        let stats = FourCellSufficientStats {
            counts: [2, 3, 4, 5],
            sums: [2.0, 6.0, 12.0, 35.0],
            n_valid: 14,
        };
        let scores = derive_gate5_gate6d_scores(stats);
        assert_eq!(scores.logic5, derive_logic5_scores(stats));
        assert_eq!(
            scores.dosage_interaction_contrast,
            derive_dosage_interaction_contrast(stats)
        );
        assert_eq!(
            scores.dosage_interaction_standardized_score,
            derive_standardized_dosage_interaction_score(stats)
        );
    }

    #[test]
    fn standardized_gate6d_score_matches_d_transpose_sigma_d_without_common_scale() {
        let stats = FourCellSufficientStats {
            counts: [2, 3, 4, 5],
            sums: [2.0, 6.0, 12.0, 35.0],
            n_valid: 14,
        };
        // mu = [1, 2, 3, 7], so delta = 7 - 3 - 2 + 1 = 3.
        // d' Sigma d / sigma_e^2 = 1/2 + 1/3 + 1/4 + 1/5.
        let expected = 9.0 / (0.5 + 1.0 / 3.0 + 0.25 + 0.2);
        let score = derive_standardized_dosage_interaction_score(stats).unwrap();
        assert!((score - expected).abs() < 1.0e-12);
    }

    #[test]
    fn dosage_interaction_score_is_zero_for_additive_cell_means_with_main_effects() {
        // Cell means follow y = 3 - g1 + 2*g2: both endpoint effects are
        // nonzero, but there is no four-cell interaction contrast.
        let stats = FourCellSufficientStats {
            counts: [2, 5, 3, 7],
            sums: [6.0, 25.0, 6.0, 28.0],
            n_valid: 17,
        };
        let means: [f64; 4] =
            std::array::from_fn(|index| stats.sums[index] / stats.counts[index] as f64);
        assert_eq!(means, [3.0, 5.0, 2.0, 4.0]);
        assert_eq!(derive_dosage_interaction_contrast(stats), Some(0.0));
        assert_eq!(
            derive_standardized_dosage_interaction_score(stats),
            Some(0.0)
        );
    }

    #[test]
    fn standardized_gate6d_score_requires_all_cells() {
        let stats = FourCellSufficientStats {
            counts: [2, 0, 4, 5],
            sums: [2.0, 0.0, 12.0, 35.0],
            n_valid: 11,
        };
        assert!(derive_standardized_dosage_interaction_score(stats).is_none());
    }

    #[test]
    fn centered_product_screen_score_matches_intercept_projected_product() {
        let stats = FourCellSufficientStats {
            counts: [2, 3, 4, 5],
            sums: [2.0, 6.0, 12.0, 35.0],
            n_valid: 14,
        };
        let (score, signed_score) = derive_centered_product_screen_score(stats)
            .expect("the centered product must vary for this table");

        // This is the direct four-cell evaluation of
        // w=(x1-mean(x1))*(x2-mean(x2)) and M0=I-11'/n.
        assert!((score - 8.344459279038722).abs() < 1.0e-12);
        assert!((signed_score - 2.8886777734871574).abs() < 1.0e-12);
    }
}
