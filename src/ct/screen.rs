#[cfg(test)]
use crate::ct::conditional::project_off;
use crate::ct::input::GenotypeCall;
use crate::ct::packed::{EncodedPair, FilledPair, PackedGenotypes, PairCounts, PairPhaseMasks};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SnpQcThresholds {
    pub(crate) min_maf: f64,
    pub(crate) max_geno: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SnpQc {
    pub(crate) n_called: usize,
    pub(crate) n_missing: usize,
    pub(crate) maf: f64,
    pub(crate) geno: f64,
}

pub(crate) fn summarize_snp(calls: &[GenotypeCall]) -> Result<SnpQc, String> {
    if calls.is_empty() {
        return Err("cannot calculate SNP QC with zero samples".to_string());
    }
    let mut n_called = 0usize;
    let mut alt_sum = 0u64;
    for call in calls {
        match call.dosage {
            Some(dosage @ 0..=2) => {
                n_called += 1;
                alt_sum += dosage as u64;
            }
            Some(dosage) => return Err(format!("invalid SNP dosage {dosage}; expected 0, 1, 2")),
            None => {}
        }
    }
    let n_missing = calls.len() - n_called;
    let allele_frequency = if n_called > 0 {
        alt_sum as f64 / (2.0 * n_called as f64)
    } else {
        0.0
    };
    Ok(SnpQc {
        n_called,
        n_missing,
        maf: allele_frequency.min(1.0 - allele_frequency),
        geno: n_missing as f64 / calls.len() as f64,
    })
}

pub(crate) fn summarize_snp_packed(packed: &PackedGenotypes) -> Result<SnpQc, String> {
    let n_samples = packed.n_samples();
    if n_samples == 0 {
        return Err("cannot calculate SNP QC with zero samples".to_string());
    }
    if packed.has_invalid_dosage() {
        return Err("invalid SNP dosage in packed genotype; expected 0, 1, 2".to_string());
    }
    let n_called = packed.dosage_count(0) + packed.dosage_count(1) + packed.dosage_count(2);
    let n_missing = n_samples - n_called;
    let alt_sum = packed.dosage_count(1) as u64 + 2 * packed.dosage_count(2) as u64;
    let allele_frequency = if n_called > 0 {
        alt_sum as f64 / (2.0 * n_called as f64)
    } else {
        0.0
    };
    Ok(SnpQc {
        n_called,
        n_missing,
        maf: allele_frequency.min(1.0 - allele_frequency),
        geno: n_missing as f64 / n_samples as f64,
    })
}

pub(crate) fn passes_snp_qc(qc: &SnpQc, thresholds: SnpQcThresholds) -> bool {
    thresholds.min_maf.is_finite()
        && (0.0..=0.5).contains(&thresholds.min_maf)
        && thresholds.max_geno.is_finite()
        && (0.0..=1.0).contains(&thresholds.max_geno)
        && qc.n_called > 0
        && qc.maf >= thresholds.min_maf
        && qc.geno <= thresholds.max_geno
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VariantMeta {
    pub(crate) index: usize,
    pub(crate) chrom: String,
    pub(crate) pos: i64,
    pub(crate) id: String,
    pub(crate) allele0: String,
    pub(crate) allele1: String,
    pub(crate) snp_maf: f64,
    pub(crate) snp_geno: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PairQc {
    pub(crate) phase_minor_freq: f64,
    pub(crate) pair_geno: f64,
    pub(crate) n_cis_observed: usize,
    pub(crate) n_trans_observed: usize,
    pub(crate) n_other_observed: usize,
    pub(crate) n_unknown_observed: usize,
    pub(crate) genotype_missing: usize,
    pub(crate) phase_unknown: usize,
}

impl PairQc {
    pub(crate) fn from_counts(counts: &PairCounts) -> Self {
        let n_informative = counts.n_informative();
        let phase_minor_freq = if n_informative > 0 {
            counts.cis_observed().min(counts.trans_observed()) as f64 / n_informative as f64
        } else {
            0.0
        };
        let n_unknown_observed = counts.genotype_missing + counts.phase_unknown;
        let pair_geno = if counts.n_total > 0 {
            n_unknown_observed as f64 / counts.n_total as f64
        } else {
            1.0
        };
        Self {
            phase_minor_freq,
            pair_geno,
            n_cis_observed: counts.cis_observed(),
            n_trans_observed: counts.trans_observed(),
            n_other_observed: counts.other_observed(),
            n_unknown_observed,
            genotype_missing: counts.genotype_missing,
            phase_unknown: counts.phase_unknown,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ScreenCandidate {
    pub(crate) left: VariantMeta,
    pub(crate) right: VariantMeta,
    pub(crate) distance_bp: i64,
    pub(crate) raw_counts: PairCounts,
    pub(crate) pair_qc: PairQc,
    pub(crate) filled: FilledPair,
    pub(crate) screen_score: f64,
    pub(crate) screen_rank: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RankedCandidate {
    pub(crate) left: VariantMeta,
    pub(crate) right: VariantMeta,
    pub(crate) distance_bp: i64,
    pub(crate) raw_counts: PairCounts,
    pub(crate) pair_qc: PairQc,
    pub(crate) encoded: EncodedPair,
    pub(crate) screen_score: f64,
    pub(crate) screen_rank: usize,
}

impl RankedCandidate {
    pub(crate) fn materialize(self) -> ScreenCandidate {
        ScreenCandidate {
            left: self.left,
            right: self.right,
            distance_bp: self.distance_bp,
            raw_counts: self.raw_counts,
            pair_qc: self.pair_qc,
            filled: self.encoded.materialize(),
            screen_score: self.screen_score,
            screen_rank: self.screen_rank,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CoreCandidates {
    pub(crate) chrom: String,
    pub(crate) start: i64,
    pub(crate) end: i64,
    pub(crate) candidates: Vec<ScreenCandidate>,
}

#[cfg(test)]
pub(crate) fn score_pair(filled: &FilledPair, residuals: &[f64]) -> Option<f64> {
    if filled.z.len() != residuals.len() {
        return None;
    }
    let mut cis_sum = 0.0;
    let mut trans_sum = 0.0;
    let mut n_cis = 0usize;
    let mut n_trans = 0usize;
    for (&z, &residual) in filled.z.iter().zip(residuals) {
        match z {
            1 => {
                cis_sum += residual;
                n_cis += 1;
            }
            -1 => {
                trans_sum += residual;
                n_trans += 1;
            }
            _ => {}
        }
    }
    if n_cis == 0 || n_trans == 0 {
        return None;
    }
    let delta = cis_sum / n_cis as f64 - trans_sum / n_trans as f64;
    let score = delta * delta / (1.0 / n_cis as f64 + 1.0 / n_trans as f64);
    score.is_finite().then_some(score)
}

/// Four-bit subset lookup for sums over sample bitsets. At each 64-sample
/// word, 16 nibble tables map a four-bit sample mask to the corresponding sum
/// of values. Sparse masks can also sum their set bits directly. This avoids
/// materializing one z value per individual during pair screening.
#[derive(Clone, Debug)]
pub(crate) struct PackedValueLookup {
    n_samples: usize,
    word_tables: Vec<[[f64; 16]; 16]>,
}

impl PackedValueLookup {
    pub(crate) fn build(values: &[f64], n_samples: usize) -> Result<Self, String> {
        if values.len() != n_samples {
            return Err(format!(
                "packed value lookup has {} values, expected {n_samples}",
                values.len()
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("packed value lookup input contains non-finite values".to_string());
        }
        let n_words = n_samples.div_ceil(u64::BITS as usize);
        let word_tables = (0..n_words)
            .map(|word_index| {
                std::array::from_fn(|nibble_index| {
                    let first_sample = word_index * u64::BITS as usize + nibble_index * 4;
                    let mut sums = [0.0; 16];
                    for subset in 1_usize..16 {
                        let bit = subset.trailing_zeros() as usize;
                        let sample = first_sample + bit;
                        let value = if sample < n_samples {
                            values[sample]
                        } else {
                            0.0
                        };
                        sums[subset] = sums[subset & (subset - 1)] + value;
                    }
                    sums
                })
            })
            .collect();
        Ok(Self {
            n_samples,
            word_tables,
        })
    }

    fn sum_mask(&self, mask: &[u64]) -> Option<f64> {
        if mask.len() != self.word_tables.len() {
            return None;
        }
        let mut sum = 0.0;
        for (word_index, (&word, table)) in mask.iter().zip(&self.word_tables).enumerate() {
            for nibble_index in 0..16 {
                let subset = ((word >> (nibble_index * 4)) & 0xf) as usize;
                let part = table[nibble_index][subset];
                if !part.is_finite() {
                    return None;
                }
                sum += part;
            }
            if word_index * u64::BITS as usize >= self.n_samples {
                return None;
            }
        }
        sum.is_finite().then_some(sum)
    }

    fn sum_mask_sparse(&self, mask: &[u64], values: &[f64]) -> Option<f64> {
        if mask.len() != self.word_tables.len() || values.len() != self.n_samples {
            return None;
        }
        let mut sum = 0.0;
        for (word_index, &word) in mask.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let sample = word_index * u64::BITS as usize + bit;
                if sample < self.n_samples {
                    sum += values[sample];
                }
                remaining &= remaining - 1;
            }
        }
        sum.is_finite().then_some(sum)
    }

    fn sum_mask_adaptive(&self, mask: &[u64], n_set: usize, values: &[f64]) -> Option<f64> {
        // Sparse bit iteration helps when only a few samples contribute. Once
        // support approaches a small fraction of the full mask, the regular
        // nibble lookup has less per-sample work and is faster.
        let sparse_limit = self.word_tables.len().saturating_mul(4);
        if n_set <= sparse_limit {
            self.sum_mask_sparse(mask, values)
        } else {
            self.sum_mask(mask)
        }
    }
}

#[cfg(test)]
pub(crate) fn score_pair_encoded(
    pair: &EncodedPair,
    residual_lookup: &PackedValueLookup,
) -> Option<f64> {
    if residual_lookup.n_samples != pair.n_samples() {
        return None;
    }
    let n_cis = pair.n_cis();
    let n_trans = pair.n_trans();
    if n_cis == 0 || n_trans == 0 {
        return None;
    }
    let cis_sum = residual_lookup.sum_mask(pair.cis_mask())?;
    let trans_sum = residual_lookup.sum_mask(pair.trans_mask())?;
    let delta = cis_sum / n_cis as f64 - trans_sum / n_trans as f64;
    let score = delta * delta / (1.0 / n_cis as f64 + 1.0 / n_trans as f64);
    score.is_finite().then_some(score)
}

/// Precomputed sufficient statistics for bitset H/Z screening. The two
/// projection passes mirror `project_off`, while the Gram matrix preserves
/// the result even when the supplied basis is only approximately orthogonal.
#[derive(Clone, Debug)]
pub(crate) struct PairRankContext {
    sigma_e2: f64,
    projected_y_lookup: PackedValueLookup,
    projected_precision_y: Vec<f64>,
    projected_y_sum: f64,
    basis_lookups: Vec<PackedValueLookup>,
    basis_gram: Vec<Vec<f64>>,
    basis_projected_y: Vec<f64>,
    intercept_only: bool,
}

impl PairRankContext {
    pub(crate) fn new(
        projected_precision_y: &[f64],
        fixed_effect_basis: &[Vec<f64>],
        sigma_e2: f64,
    ) -> Result<Self, String> {
        if !(sigma_e2.is_finite() && sigma_e2 > 0.0) {
            return Err("CT rank scan requires positive finite sigma_e2".to_string());
        }
        let n_samples = projected_precision_y.len();
        let projected_y_lookup = PackedValueLookup::build(projected_precision_y, n_samples)?;
        for (index, column) in fixed_effect_basis.iter().enumerate() {
            if column.len() != n_samples || column.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "CT fixed-effect basis column {index} has invalid dimensions or values"
                ));
            }
        }
        let intercept_only = fixed_effect_basis.len() == 1
            && fixed_effect_basis[0].first().is_some_and(|&first| {
                first != 0.0 && fixed_effect_basis[0].iter().all(|&v| v == first)
            });
        let basis_lookups = fixed_effect_basis
            .iter()
            .filter(|_| !intercept_only)
            .map(|column| PackedValueLookup::build(column, n_samples))
            .collect::<Result<Vec<_>, _>>()?;
        let basis_gram = if intercept_only {
            Vec::new()
        } else {
            fixed_effect_basis
                .iter()
                .map(|left| {
                    fixed_effect_basis
                        .iter()
                        .map(|right| dot(left, right))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let basis_projected_y = if intercept_only {
            Vec::new()
        } else {
            fixed_effect_basis
                .iter()
                .map(|column| dot(column, projected_precision_y))
                .collect()
        };
        let projected_y_sum = projected_precision_y.iter().sum();
        Ok(Self {
            sigma_e2,
            projected_y_lookup,
            projected_precision_y: projected_precision_y.to_vec(),
            projected_y_sum,
            basis_lookups,
            basis_gram,
            basis_projected_y,
            intercept_only,
        })
    }

    fn projection_coefficients(&self, basis_dots: &[f64]) -> Vec<f64> {
        let mut first = vec![0.0; basis_dots.len()];
        for i in 0..first.len() {
            first[i] = basis_dots[i];
            for j in 0..i {
                first[i] -= self.basis_gram[i][j] * first[j];
            }
        }
        let mut second = vec![0.0; first.len()];
        for i in 0..second.len() {
            second[i] = basis_dots[i];
            for j in 0..first.len() {
                second[i] -= self.basis_gram[i][j] * first[j];
            }
            for j in 0..i {
                second[i] -= self.basis_gram[i][j] * second[j];
            }
        }
        first.iter().zip(second).map(|(a, b)| a + b).collect()
    }

    fn residual_cross(
        &self,
        raw: f64,
        left_dots: &[f64],
        right_dots: &[f64],
        left_coefficients: &[f64],
        right_coefficients: &[f64],
    ) -> f64 {
        let mut value = raw;
        for i in 0..left_coefficients.len() {
            value -= left_coefficients[i] * right_dots[i] + right_coefficients[i] * left_dots[i];
            for j in 0..right_coefficients.len() {
                value += left_coefficients[i] * self.basis_gram[i][j] * right_coefficients[j];
            }
        }
        value
    }

    fn residual_y_dot(&self, raw: f64, coefficients: &[f64]) -> f64 {
        raw - coefficients
            .iter()
            .zip(&self.basis_projected_y)
            .map(|(coefficient, basis_y)| coefficient * basis_y)
            .sum::<f64>()
    }
}

#[cfg(test)]
pub(crate) fn joint_rank_score_encoded(
    pair: &EncodedPair,
    context: &PairRankContext,
) -> Option<f64> {
    joint_rank_score_masks(
        pair.n_samples(),
        pair.n_cis(),
        pair.n_trans(),
        pair.cis_mask(),
        pair.trans_mask(),
        context,
    )
}

pub(crate) fn joint_rank_score_phase_masks(
    pair: &PairPhaseMasks,
    context: &PairRankContext,
) -> Option<f64> {
    joint_rank_score_masks(
        pair.n_samples(),
        pair.n_cis(),
        pair.n_trans(),
        pair.cis_mask(),
        pair.trans_mask(),
        context,
    )
}

/// Rank the no-phase D×D term H alone after projection on the global null
/// fixed-effect basis. The pair-specific single-site terms are fitted only
/// for TopK candidates, as in the existing CT screen.
pub(crate) fn double_dominance_rank_score_mask(
    n_samples: usize,
    double_het_mask: &[u64],
    n_double_het: usize,
    context: &PairRankContext,
) -> Option<f64> {
    if context.projected_y_lookup.n_samples != n_samples
        || double_het_mask.len() != n_samples.div_ceil(u64::BITS as usize)
        || n_double_het == 0
    {
        return None;
    }
    let raw_information = n_double_het as f64;
    let h_dot_y = context.projected_y_lookup.sum_mask_adaptive(
        double_het_mask,
        n_double_het,
        &context.projected_precision_y,
    )?;
    let (information, score) = if context.intercept_only {
        let n = n_samples as f64;
        let h_sum = raw_information;
        (
            h_sum - h_sum * h_sum / n,
            h_dot_y - h_sum * context.projected_y_sum / n,
        )
    } else {
        let basis_dots = context
            .basis_lookups
            .iter()
            .map(|lookup| lookup.sum_mask(double_het_mask))
            .collect::<Option<Vec<_>>>()?;
        let coefficients = context.projection_coefficients(&basis_dots);
        let information = context.residual_cross(
            raw_information,
            &basis_dots,
            &basis_dots,
            &coefficients,
            &coefficients,
        );
        let score = context.residual_y_dot(h_dot_y, &coefficients);
        (information, score)
    };
    if !(information.is_finite() && information > 1.0e-12 * raw_information && score.is_finite()) {
        return None;
    }
    let value = score * score / (information * context.sigma_e2);
    value.is_finite().then_some(value)
}

fn joint_rank_score_masks(
    n_samples: usize,
    n_cis: usize,
    n_trans: usize,
    cis_mask: &[u64],
    trans_mask: &[u64],
    context: &PairRankContext,
) -> Option<f64> {
    if context.projected_y_lookup.n_samples != n_samples {
        return None;
    }
    let raw_information = (n_cis + n_trans) as f64;
    if !(raw_information.is_finite() && raw_information > 0.0) {
        return None;
    }
    if context.intercept_only {
        let cis_y = context.projected_y_lookup.sum_mask_adaptive(
            cis_mask,
            n_cis,
            &context.projected_precision_y,
        )?;
        let trans_y = context.projected_y_lookup.sum_mask_adaptive(
            trans_mask,
            n_trans,
            &context.projected_precision_y,
        )?;
        let n = n_samples as f64;
        let n_phase = raw_information;
        let cis_minus_trans = n_cis as f64 - n_trans as f64;
        // With an intercept as the sole basis column, projecting H and Z
        // amounts to subtracting their sample means. These closed forms avoid
        // two additional bitset lookups for the constant basis column.
        let information_zz = n_phase - cis_minus_trans * cis_minus_trans / n;
        let information_hh = n_phase - n_phase * n_phase / n;
        let information_zh = cis_minus_trans - cis_minus_trans * n_phase / n;
        let score_z = cis_y - trans_y - cis_minus_trans * context.projected_y_sum / n;
        let score_h = cis_y + trans_y - n_phase * context.projected_y_sum / n;
        return joint_score_from_sufficient_statistics(
            information_zz,
            information_hh,
            information_zh,
            score_z,
            score_h,
            raw_information,
            context.sigma_e2,
        );
    }
    let cis_y = context.projected_y_lookup.sum_mask(cis_mask)?;
    let trans_y = context.projected_y_lookup.sum_mask(trans_mask)?;
    let z_dot_y = cis_y - trans_y;
    let h_dot_y = cis_y + trans_y;
    let basis_sums = context
        .basis_lookups
        .iter()
        .map(|lookup| {
            let cis = lookup.sum_mask(cis_mask)?;
            let trans = lookup.sum_mask(trans_mask)?;
            Some((cis - trans, cis + trans))
        })
        .collect::<Option<Vec<_>>>()?;
    let (z_dot_basis, h_dot_basis): (Vec<_>, Vec<_>) = basis_sums.into_iter().unzip();

    let z_coefficients = context.projection_coefficients(&z_dot_basis);
    let h_coefficients = context.projection_coefficients(&h_dot_basis);
    let information_zz = context.residual_cross(
        raw_information,
        &z_dot_basis,
        &z_dot_basis,
        &z_coefficients,
        &z_coefficients,
    );
    let information_hh = context.residual_cross(
        raw_information,
        &h_dot_basis,
        &h_dot_basis,
        &h_coefficients,
        &h_coefficients,
    );
    let raw_cross = n_cis as f64 - n_trans as f64;
    let information_zh = context.residual_cross(
        raw_cross,
        &z_dot_basis,
        &h_dot_basis,
        &z_coefficients,
        &h_coefficients,
    );
    let score_z = context.residual_y_dot(z_dot_y, &z_coefficients);
    let score_h = context.residual_y_dot(h_dot_y, &h_coefficients);
    joint_score_from_sufficient_statistics(
        information_zz,
        information_hh,
        information_zh,
        score_z,
        score_h,
        raw_information,
        context.sigma_e2,
    )
}

/// Scores the column space of the residualized H/Z pair, dropping aliased
/// columns. In particular, H is often the intercept when every sample is a
/// double heterozygote; that must not suppress an otherwise estimable Z score.
fn joint_score_from_sufficient_statistics(
    information_zz: f64,
    information_hh: f64,
    information_zh: f64,
    score_z: f64,
    score_h: f64,
    raw_information: f64,
    sigma_e2: f64,
) -> Option<f64> {
    if !(raw_information.is_finite()
        && raw_information > 0.0
        && sigma_e2.is_finite()
        && sigma_e2 > 0.0
        && information_zz.is_finite()
        && information_hh.is_finite()
        && information_zh.is_finite()
        && score_z.is_finite()
        && score_h.is_finite())
    {
        return None;
    }
    let tolerance = 1.0e-12 * raw_information;
    let z_estimable = information_zz > tolerance;
    let h_estimable = information_hh > tolerance;
    if !z_estimable && !h_estimable {
        return None;
    }

    let (primary_information, primary_score, secondary_information, secondary_score) =
        if z_estimable && (!h_estimable || information_zz >= information_hh) {
            (information_zz, score_z, information_hh, score_h)
        } else {
            (information_hh, score_h, information_zz, score_z)
        };
    let primary_cross = information_zh;
    let mut quadratic = primary_score * primary_score / primary_information;
    if z_estimable && h_estimable {
        let residual_information =
            secondary_information - primary_cross * primary_cross / primary_information;
        if residual_information.is_finite() && residual_information > tolerance {
            let residual_score =
                secondary_score - primary_cross * primary_score / primary_information;
            quadratic += residual_score * residual_score / residual_information;
        }
    }
    let score = quadratic.max(0.0) / sigma_e2;
    score.is_finite().then_some(score)
}

/// Materialized reference implementation of the joint H/Z score.
///
/// `projected_precision_y` is the screening response. LM supplies residualized
/// y and an orthonormal global fixed-effect basis; FvLMM supplies sigma_e^2 P_C y.
/// FvLMM pair-vector projection is Euclidean and is used for ranking only.
#[cfg(test)]
pub(crate) fn joint_rank_score(
    filled: &FilledPair,
    projected_precision_y: &[f64],
    fixed_effect_basis: &[Vec<f64>],
    sigma_e2: f64,
) -> Option<f64> {
    if filled.z.len() != projected_precision_y.len()
        || !(sigma_e2.is_finite() && sigma_e2 > 0.0)
        || projected_precision_y.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let z = filled
        .z
        .iter()
        .map(|&value| value as f64)
        .collect::<Vec<_>>();
    let h = z.iter().map(|z| z.abs()).collect::<Vec<_>>();
    let z_residualized = project_off(&z, fixed_effect_basis);
    let h_residualized = project_off(&h, fixed_effect_basis);
    let information_zz = dot(&z_residualized, &z_residualized);
    let information_hh = dot(&h_residualized, &h_residualized);
    let information_zh = dot(&z_residualized, &h_residualized);
    let score_z = dot(&z_residualized, projected_precision_y);
    let score_h = dot(&h_residualized, projected_precision_y);
    let raw_information = dot(&z, &z);
    joint_score_from_sufficient_statistics(
        information_zz,
        information_hh,
        information_zh,
        score_z,
        score_h,
        raw_information,
        sigma_e2,
    )
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::{
        joint_rank_score, joint_rank_score_encoded, joint_rank_score_phase_masks, passes_snp_qc,
        score_pair, score_pair_encoded, summarize_snp, PackedValueLookup, PairRankContext,
        SnpQcThresholds,
    };
    use crate::ct::input::{GenotypeCall, PhaseSet};
    use crate::ct::packed::{
        prepare_pair, prepare_pair_encoded, prepare_pair_for_screen_optional,
        EncodedPairPreparation, FilledPair, PackedGenotypes, PairPreparation, PairQcThresholds,
        PairScreenPreparation,
    };

    #[test]
    fn joint_rank_score_detects_shared_double_heterozygote_effects() {
        let filled = FilledPair {
            z: vec![1, 1, 1, -1, 0, 0, 0, 0],
            g_i: vec![1, 1, 1, 1, 0, 0, 2, 2],
            g_j: vec![1, 1, 1, 1, 0, 2, 0, 2],
            imputed: vec![false; 8],
            n_imputed: 0,
        };
        let y = vec![1.2, 0.8, 1.0, -1.0, 0.1, -0.1, 0.2, -0.2];
        let shifted = y
            .iter()
            .zip(&filled.z)
            .map(|(&y, &z)| y + if z != 0 { 7.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let basis = vec![vec![1.0 / 8.0_f64.sqrt(); 8]];
        let first = joint_rank_score(&filled, &y, &basis, 1.0).unwrap();
        let second = joint_rank_score(&filled, &shifted, &basis, 1.0).unwrap();
        assert!(
            second > first,
            "shared phase signal was not ranked: {first} {second}"
        );
        let left = filled
            .g_i
            .iter()
            .map(|&dosage| GenotypeCall {
                dosage: Some(dosage),
                alt_on_hap0: (dosage == 1).then_some(true),
                phase_set: PhaseSet::Implicit,
            })
            .collect::<Vec<_>>();
        let right = filled
            .g_j
            .iter()
            .zip(&filled.z)
            .map(|(&dosage, &z)| GenotypeCall {
                dosage: Some(dosage),
                alt_on_hap0: (dosage == 1).then_some(z == 1),
                phase_set: PhaseSet::Implicit,
            })
            .collect::<Vec<_>>();
        let thresholds = PairQcThresholds {
            min_phase_minor_freq: 0.0,
            max_unknown_rate: 0.0,
            min_cis: 1,
            min_trans: 1,
        };
        let EncodedPairPreparation::Ready { pair, .. } = prepare_pair_encoded(
            &left,
            &right,
            &PackedGenotypes::from_calls(&left),
            &PackedGenotypes::from_calls(&right),
            thresholds,
        )
        .unwrap() else {
            panic!("expected a testable pair");
        };
        for values in [&y, &shifted] {
            let context = PairRankContext::new(values, &basis, 1.0).unwrap();
            let observed = joint_rank_score_encoded(&pair, &context).unwrap();
            let expected = joint_rank_score(&filled, values, &basis, 1.0).unwrap();
            assert!((observed - expected).abs() < 1e-10);
        }
    }

    fn call(dosage: Option<u8>) -> GenotypeCall {
        GenotypeCall {
            dosage,
            alt_on_hap0: None,
            phase_set: PhaseSet::Unknown,
        }
    }

    #[test]
    fn snp_qc_counts_heterozygotes_and_uses_raw_genotypes() {
        let calls = vec![
            call(Some(0)),
            call(Some(1)),
            call(Some(2)),
            call(Some(1)),
            call(None),
        ];
        let qc = summarize_snp(&calls).unwrap();
        assert_eq!(qc.n_called, 4);
        assert_eq!(qc.n_missing, 1);
        assert_eq!(qc.maf, 0.5);
        assert_eq!(qc.geno, 0.2);
        assert!(passes_snp_qc(
            &qc,
            SnpQcThresholds {
                min_maf: 0.5,
                max_geno: 0.2,
            }
        ));
        assert!(!passes_snp_qc(
            &qc,
            SnpQcThresholds {
                min_maf: 0.500_001,
                max_geno: 0.2,
            }
        ));
    }

    #[test]
    fn snp_with_no_called_genotypes_fails_qc_and_invalid_dosage_is_an_error() {
        let qc = summarize_snp(&[call(None), call(None)]).unwrap();
        assert_eq!(qc.n_called, 0);
        assert!(!passes_snp_qc(
            &qc,
            SnpQcThresholds {
                min_maf: 0.0,
                max_geno: 1.0,
            }
        ));
        assert!(summarize_snp(&[call(Some(3))]).is_err());
    }

    #[test]
    fn joint_rank_score_uses_both_phase_vectors_after_covariate_projection() {
        let pair = FilledPair {
            z: vec![1, -1, 1, 0],
            g_i: vec![0; 4],
            g_j: vec![0; 4],
            imputed: vec![false; 4],
            n_imputed: 0,
        };
        let intercept = vec![vec![0.5; 4]];
        let projected_precision_y = [1.0, -1.0, 2.0, -2.0];

        let score = joint_rank_score(&pair, &projected_precision_y, &intercept, 2.0).unwrap();

        // The two-column H/Z score after intercept projection is 9.5; sigma_e2=2.
        assert!((score - 4.75).abs() < 1e-12);
    }

    #[test]
    fn joint_rank_score_keeps_estimable_contrast_when_shared_vector_is_absorbed() {
        let pair = FilledPair {
            z: vec![1, 1, -1, -1],
            g_i: vec![1; 4],
            g_j: vec![1; 4],
            imputed: vec![false; 4],
            n_imputed: 0,
        };
        let intercept = vec![vec![0.5; 4]];
        let y = [3.0, 2.0, -1.0, 0.0];

        let score = joint_rank_score(&pair, &y, &intercept, 1.0)
            .expect("estimable cis/trans contrast should remain rankable");

        // H is the intercept and is aliased; Z remains orthogonal to it.
        assert!((score - 9.0).abs() < 1e-12);
    }

    #[test]
    fn bitset_lm_and_fvlmm_scores_match_materialized_reference() {
        let left = vec![
            GenotypeCall {
                dosage: Some(1),
                alt_on_hap0: Some(true),
                phase_set: PhaseSet::Implicit,
            },
            GenotypeCall {
                dosage: Some(1),
                alt_on_hap0: Some(true),
                phase_set: PhaseSet::Implicit,
            },
            GenotypeCall {
                dosage: Some(1),
                alt_on_hap0: Some(false),
                phase_set: PhaseSet::Implicit,
            },
            GenotypeCall {
                dosage: Some(1),
                alt_on_hap0: Some(false),
                phase_set: PhaseSet::Implicit,
            },
            call(None),
            GenotypeCall {
                dosage: Some(1),
                alt_on_hap0: Some(true),
                phase_set: PhaseSet::Unknown,
            },
        ];
        let right = vec![
            GenotypeCall {
                dosage: Some(1),
                alt_on_hap0: Some(true),
                phase_set: PhaseSet::Implicit,
            },
            GenotypeCall {
                dosage: Some(1),
                alt_on_hap0: Some(true),
                phase_set: PhaseSet::Implicit,
            },
            GenotypeCall {
                dosage: Some(1),
                alt_on_hap0: Some(true),
                phase_set: PhaseSet::Implicit,
            },
            GenotypeCall {
                dosage: Some(1),
                alt_on_hap0: Some(false),
                phase_set: PhaseSet::Implicit,
            },
            GenotypeCall {
                dosage: Some(1),
                alt_on_hap0: None,
                phase_set: PhaseSet::Unknown,
            },
            GenotypeCall {
                dosage: Some(1),
                alt_on_hap0: Some(false),
                phase_set: PhaseSet::Implicit,
            },
        ];
        let thresholds = PairQcThresholds {
            min_phase_minor_freq: 0.2,
            max_unknown_rate: 0.5,
            min_cis: 1,
            min_trans: 1,
        };
        let expected = prepare_pair(&left, &right, thresholds).unwrap();
        let left_packed = PackedGenotypes::from_calls(&left);
        let right_packed = PackedGenotypes::from_calls(&right);
        let encoded =
            prepare_pair_encoded(&left, &right, &left_packed, &right_packed, thresholds).unwrap();
        let (FilledPair { .. }, pair) = match (expected, encoded) {
            (PairPreparation::Ready { filled, .. }, EncodedPairPreparation::Ready { pair, .. }) => {
                (filled, pair)
            }
            (expected, observed) => panic!("preparation mismatch: {expected:?} vs {observed:?}"),
        };

        let filled = pair.materialize();
        let residuals = [1.0, -1.0, 0.5, -0.2, 1.5, -0.8];
        let lookup = PackedValueLookup::build(&residuals, residuals.len()).unwrap();
        let expected_lm = score_pair(&filled, &residuals).unwrap();
        let observed_lm = score_pair_encoded(&pair, &lookup).unwrap();
        assert!((expected_lm - observed_lm).abs() < 1.0e-12);

        let projected_precision_y = [0.7, -0.2, 1.3, 0.1, -1.0, 0.8];
        let basis = vec![
            vec![1.0 / 6.0_f64.sqrt(); 6],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ];
        let expected_fvlmm = joint_rank_score(&filled, &projected_precision_y, &basis, 1.7)
            .expect("reference FvLMM score");
        let context = PairRankContext::new(&projected_precision_y, &basis, 1.7).unwrap();
        let observed_fvlmm = joint_rank_score_encoded(&pair, &context).expect("bitset FvLMM score");
        let PairScreenPreparation::Ready { phase_masks, .. } = prepare_pair_for_screen_optional(
            Some(&left),
            Some(&right),
            &left_packed,
            &right_packed,
            thresholds,
        )
        .unwrap() else {
            panic!("expected fast-screen pair masks");
        };
        let fast_fvlmm =
            joint_rank_score_phase_masks(&phase_masks, &context).expect("phase-mask FvLMM score");
        assert!(
            (expected_fvlmm - observed_fvlmm).abs() < 1.0e-9,
            "reference={expected_fvlmm}, encoded={observed_fvlmm}"
        );
        assert!(
            (observed_fvlmm - fast_fvlmm).abs() < 1.0e-12,
            "encoded={observed_fvlmm}, phase masks={fast_fvlmm}"
        );
    }

    #[test]
    fn sparse_mask_sum_matches_nibble_lookup_including_partial_final_word() {
        let values = (0..79)
            .map(|sample| (sample as f64 - 31.5) / 7.0)
            .collect::<Vec<_>>();
        let lookup = PackedValueLookup::build(&values, values.len()).unwrap();
        let mut mask = vec![0_u64; values.len().div_ceil(u64::BITS as usize)];
        for sample in (2..values.len()).step_by(9) {
            mask[sample / 64] |= 1_u64 << (sample % 64);
        }
        // Bits past n_samples are ignored by the existing nibble lookup and
        // must be ignored by the sparse path as well.
        mask[1] |= u64::MAX << (values.len() - 64);
        let sparse = lookup.sum_mask_sparse(&mask, &values).unwrap();
        let nibble = lookup.sum_mask(&mask).unwrap();
        assert!((sparse - nibble).abs() < 1.0e-12);
        let sparse_adaptive = lookup.sum_mask_adaptive(&mask, 1, &values).unwrap();
        assert!((sparse_adaptive - nibble).abs() < 1.0e-12);

        let dense = vec![u64::MAX; mask.len()];
        let dense_adaptive = lookup
            .sum_mask_adaptive(&dense, values.len(), &values)
            .unwrap();
        let dense_nibble = lookup.sum_mask(&dense).unwrap();
        assert!((dense_adaptive - dense_nibble).abs() < 1.0e-12);
    }

    #[test]
    fn intercept_only_pair_rank_score_matches_materialized_reference() {
        let n_samples = 73;
        let left = (0..n_samples)
            .map(|sample| GenotypeCall {
                dosage: Some(1),
                alt_on_hap0: Some(sample % 2 == 0),
                phase_set: PhaseSet::Implicit,
            })
            .collect::<Vec<_>>();
        let right = (0..n_samples)
            .map(|sample| GenotypeCall {
                dosage: Some(1),
                alt_on_hap0: Some(sample % 3 != 0),
                phase_set: PhaseSet::Implicit,
            })
            .collect::<Vec<_>>();
        let thresholds = PairQcThresholds {
            min_phase_minor_freq: 0.0,
            max_unknown_rate: 0.0,
            min_cis: 1,
            min_trans: 1,
        };
        let left_packed = PackedGenotypes::from_calls(&left);
        let right_packed = PackedGenotypes::from_calls(&right);
        let EncodedPairPreparation::Ready { pair, .. } =
            prepare_pair_encoded(&left, &right, &left_packed, &right_packed, thresholds).unwrap()
        else {
            panic!("expected testable cis/trans pair");
        };
        let y = (0..n_samples)
            .map(|sample| ((sample * 17 % 41) as f64 - 20.0) / 13.0)
            .collect::<Vec<_>>();
        let intercept = vec![vec![1.0 / (n_samples as f64).sqrt(); n_samples]];
        let context = PairRankContext::new(&y, &intercept, 0.73).unwrap();
        assert!(context.intercept_only);

        let expected = joint_rank_score(&pair.materialize(), &y, &intercept, 0.73).unwrap();
        let observed = joint_rank_score_encoded(&pair, &context).unwrap();
        assert!((observed - expected).abs() < 1.0e-10);
    }

    #[test]
    fn double_dominance_rank_score_matches_materialized_h_only_projection() {
        let n = 8;
        let h = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let y = [0.2, 1.3, -0.7, 0.9, 0.1, -0.4, 1.1, 0.2];
        let x = [-3.0, -1.0, 1.0, 3.0, -3.0, -1.0, 1.0, 3.0];
        let intercept = vec![1.0 / (n as f64).sqrt(); n];
        let covariate = x
            .iter()
            .map(|value| value / 40.0_f64.sqrt())
            .collect::<Vec<_>>();
        let masks = [0b1001_0110_u64];
        let sigma_e2 = 0.73;

        for basis in [vec![intercept.clone()], vec![intercept.clone(), covariate]] {
            let context = PairRankContext::new(&y, &basis, sigma_e2).unwrap();
            let observed = super::double_dominance_rank_score_mask(n, &masks, 4, &context)
                .expect("estimable H-only screen score");
            let mut h_residual = h.to_vec();
            let mut y_residual = y.to_vec();
            for column in &basis {
                let h_loading = h_residual
                    .iter()
                    .zip(column)
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                let y_loading = y_residual
                    .iter()
                    .zip(column)
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                for row in 0..n {
                    h_residual[row] -= h_loading * column[row];
                    y_residual[row] -= y_loading * column[row];
                }
            }
            let score = h_residual
                .iter()
                .zip(&y_residual)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let information = h_residual.iter().map(|value| value * value).sum::<f64>();
            let expected = score * score / (information * sigma_e2);
            assert!(
                (observed - expected).abs() < 1.0e-10,
                "observed={observed}, expected={expected}"
            );
        }

        let all_h = [(1_u64 << n) - 1];
        let intercept_context = PairRankContext::new(&y, &[intercept], sigma_e2).unwrap();
        assert!(
            super::double_dominance_rank_score_mask(n, &all_h, n, &intercept_context).is_none()
        );
    }
}
