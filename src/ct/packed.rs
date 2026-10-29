use crate::ct::input::{GenotypeCall, PhaseSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PairCategory {
    D00,
    D01,
    D02,
    D10,
    Cis,
    Trans,
    D12,
    D20,
    D21,
    D22,
}

impl PairCategory {
    const ALL: [Self; 10] = [
        Self::D00,
        Self::D01,
        Self::D02,
        Self::D10,
        Self::Cis,
        Self::Trans,
        Self::D12,
        Self::D20,
        Self::D21,
        Self::D22,
    ];

    pub(crate) const fn dosages(self) -> (u8, u8) {
        match self {
            Self::D00 => (0, 0),
            Self::D01 => (0, 1),
            Self::D02 => (0, 2),
            Self::D10 => (1, 0),
            Self::Cis | Self::Trans => (1, 1),
            Self::D12 => (1, 2),
            Self::D20 => (2, 0),
            Self::D21 => (2, 1),
            Self::D22 => (2, 2),
        }
    }

    const fn z(self) -> i8 {
        match self {
            Self::Cis => 1,
            Self::Trans => -1,
            _ => 0,
        }
    }

    fn index(self) -> usize {
        match self {
            Self::D00 => 0,
            Self::D01 => 1,
            Self::D02 => 2,
            Self::D10 => 3,
            Self::Cis => 4,
            Self::Trans => 5,
            Self::D12 => 6,
            Self::D20 => 7,
            Self::D21 => 8,
            Self::D22 => 9,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PairObservation {
    Observed(PairCategory),
    GenotypeMissing,
    PhaseUnknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PairCounts {
    pub(crate) category_counts: [usize; 10],
    pub(crate) genotype_missing: usize,
    pub(crate) phase_unknown: usize,
    pub(crate) n_total: usize,
}

/// Per-site dosage and phase masks used by the streaming pair scan.
///
/// The three dosage masks let pair QC count the eight non-double-heterozygote
/// categories a machine word at a time. Double heterozygotes are handled
/// separately because their cis/trans label depends on phase-set comparability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PackedGenotypes {
    n_samples: usize,
    dosage: [Vec<u64>; 3],
    missing: Vec<u64>,
    invalid: Vec<u64>,
    implicit_phase: Vec<u64>,
    alt_on_hap0: Vec<u64>,
    contains_invalid_dosage: bool,
    has_explicit_phase: bool,
}

impl PackedGenotypes {
    pub(crate) fn from_calls(calls: &[GenotypeCall]) -> Self {
        let mut builder = PackedGenotypesBuilder::new(calls.len());
        for call in calls {
            builder.push(call.dosage, call.alt_on_hap0, &call.phase_set);
        }
        builder.finish()
    }

    pub(crate) fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub(crate) fn estimated_storage_bytes_for_samples(n_samples: usize) -> usize {
        let n_words = n_samples.div_ceil(u64::BITS as usize);
        std::mem::size_of::<Self>().saturating_add(
            n_words
                .saturating_mul(7)
                .saturating_mul(std::mem::size_of::<u64>()),
        )
    }

    pub(crate) fn dosage_count(&self, dosage: u8) -> usize {
        self.dosage[dosage as usize]
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    pub(crate) fn has_invalid_dosage(&self) -> bool {
        self.contains_invalid_dosage
    }
}

/// Incremental bitset builder used by the dual-parent BED decoder so it can
/// pack each offspring call as it is decoded, without first allocating a
/// `GenotypeCall` for every sample.
pub(crate) struct PackedGenotypesBuilder {
    packed: PackedGenotypes,
    next_sample: usize,
}

impl PackedGenotypesBuilder {
    pub(crate) fn new(n_samples: usize) -> Self {
        let n_words = n_samples.div_ceil(u64::BITS as usize);
        Self {
            packed: PackedGenotypes {
                n_samples,
                dosage: std::array::from_fn(|_| vec![0; n_words]),
                missing: vec![0; n_words],
                invalid: vec![0; n_words],
                implicit_phase: vec![0; n_words],
                alt_on_hap0: vec![0; n_words],
                contains_invalid_dosage: false,
                has_explicit_phase: false,
            },
            next_sample: 0,
        }
    }

    pub(crate) fn push(
        &mut self,
        dosage: Option<u8>,
        alt_on_hap0: Option<bool>,
        phase_set: &PhaseSet,
    ) {
        debug_assert!(self.next_sample < self.packed.n_samples);
        let sample = self.next_sample;
        self.next_sample += 1;
        let word = sample / u64::BITS as usize;
        let bit = 1_u64 << (sample % u64::BITS as usize);
        match dosage {
            None => self.packed.missing[word] |= bit,
            Some(dose @ 0..=2) => {
                self.packed.dosage[dose as usize][word] |= bit;
                if dose == 1 {
                    match phase_set {
                        PhaseSet::Implicit if alt_on_hap0.is_some() => {
                            self.packed.implicit_phase[word] |= bit;
                        }
                        PhaseSet::Explicit(_) => self.packed.has_explicit_phase = true,
                        PhaseSet::Implicit | PhaseSet::Unknown => {}
                    }
                    if alt_on_hap0 == Some(true) {
                        self.packed.alt_on_hap0[word] |= bit;
                    }
                }
            }
            Some(_) => {
                self.packed.invalid[word] |= bit;
                self.packed.contains_invalid_dosage = true;
            }
        }
    }

    /// Append a call derived from two homozygous parents. Such calls are
    /// either missing or have a known parental phase when heterozygous, so we
    /// can update the masks directly without constructing a `PhaseSet`.
    pub(crate) fn push_parent_call(&mut self, dosage: Option<u8>, alt_on_hap0: bool) {
        debug_assert!(self.next_sample < self.packed.n_samples);
        let sample = self.next_sample;
        self.next_sample += 1;
        let word = sample / u64::BITS as usize;
        let bit = 1_u64 << (sample % u64::BITS as usize);
        match dosage {
            None => self.packed.missing[word] |= bit,
            Some(dose @ 0..=2) => {
                self.packed.dosage[dose as usize][word] |= bit;
                if dose == 1 {
                    self.packed.implicit_phase[word] |= bit;
                    if alt_on_hap0 {
                        self.packed.alt_on_hap0[word] |= bit;
                    }
                }
            }
            Some(_) => {
                self.packed.invalid[word] |= bit;
                self.packed.contains_invalid_dosage = true;
            }
        }
    }

    pub(crate) fn finish(self) -> PackedGenotypes {
        debug_assert_eq!(self.next_sample, self.packed.n_samples);
        self.packed
    }
}

impl PairCounts {
    pub(crate) fn cis_observed(&self) -> usize {
        self.category_counts[PairCategory::Cis.index()]
    }

    pub(crate) fn trans_observed(&self) -> usize {
        self.category_counts[PairCategory::Trans.index()]
    }

    pub(crate) fn other_observed(&self) -> usize {
        self.category_counts
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != PairCategory::Cis.index() && *i != PairCategory::Trans.index())
            .map(|(_, n)| *n)
            .sum()
    }

    pub(crate) fn n_informative(&self) -> usize {
        self.cis_observed() + self.trans_observed()
    }

    fn n_unknown(&self) -> usize {
        self.genotype_missing + self.phase_unknown
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PairQcThresholds {
    pub(crate) min_phase_minor_freq: f64,
    pub(crate) max_unknown_rate: f64,
    pub(crate) min_cis: usize,
    pub(crate) min_trans: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FilledPair {
    pub(crate) z: Vec<i8>,
    pub(crate) g_i: Vec<u8>,
    pub(crate) g_j: Vec<u8>,
    pub(crate) imputed: Vec<bool>,
    pub(crate) n_imputed: usize,
}

/// Filled genotype data kept in bitset form while a pair competes for TopK.
/// The final `FilledPair` vectors are created only after a core's TopK is
/// finalized.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct EncodedPair {
    n_samples: usize,
    category_masks: [Vec<u64>; 10],
    imputed_mask: Vec<u64>,
    pub(crate) n_imputed: usize,
    phase_independent: bool,
}

impl EncodedPair {
    pub(crate) fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub(crate) fn materialize(&self) -> FilledPair {
        let mut z = vec![0_i8; self.n_samples];
        let mut g_i = vec![0_u8; self.n_samples];
        let mut g_j = vec![0_u8; self.n_samples];
        let mut imputed = vec![false; self.n_samples];
        for category in PairCategory::ALL {
            let (dose_i, dose_j) = category.dosages();
            for (word_index, word) in self.category_masks[category.index()].iter().enumerate() {
                let mut bits = *word;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let sample = word_index * u64::BITS as usize + bit;
                    if sample < self.n_samples {
                        z[sample] = category.z();
                        g_i[sample] = dose_i;
                        g_j[sample] = dose_j;
                    }
                    bits &= bits - 1;
                }
            }
        }
        for (word_index, word) in self.imputed_mask.iter().enumerate() {
            let mut bits = *word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let sample = word_index * u64::BITS as usize + bit;
                if sample < self.n_samples {
                    imputed[sample] = true;
                }
                bits &= bits - 1;
            }
        }
        if self.phase_independent {
            z.fill(0);
        }
        FilledPair {
            z,
            g_i,
            g_j,
            imputed,
            n_imputed: self.n_imputed,
        }
    }

    pub(crate) fn cis_mask(&self) -> &[u64] {
        &self.category_masks[PairCategory::Cis.index()]
    }

    pub(crate) fn trans_mask(&self) -> &[u64] {
        &self.category_masks[PairCategory::Trans.index()]
    }

    pub(crate) fn n_cis(&self) -> usize {
        self.cis_mask()
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    pub(crate) fn n_trans(&self) -> usize {
        self.trans_mask()
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg(test)]
pub(crate) enum PairPreparation {
    QcRejected(PairCounts),
    NoCompatibleDonor(PairCounts),
    Ready {
        counts: PairCounts,
        filled: FilledPair,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum EncodedPairPreparation {
    QcRejected(PairCounts),
    NoCompatibleDonor(PairCounts),
    Ready {
        counts: PairCounts,
        pair: EncodedPair,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct PairPhaseMasks {
    n_samples: usize,
    cis: Vec<u64>,
    trans: Vec<u64>,
    #[cfg(test)]
    unknown: Vec<u64>,
    // Count once after imputation so every ranked pair can reuse its support.
    n_cis: usize,
    n_trans: usize,
}

impl PairPhaseMasks {
    pub(crate) fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub(crate) fn cis_mask(&self) -> &[u64] {
        &self.cis
    }

    pub(crate) fn trans_mask(&self) -> &[u64] {
        &self.trans
    }

    #[cfg(test)]
    fn unknown_mask(&self) -> &[u64] {
        &self.unknown
    }

    pub(crate) fn n_cis(&self) -> usize {
        self.n_cis
    }

    pub(crate) fn n_trans(&self) -> usize {
        self.n_trans
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PairDxdMasks {
    /// Filled H mask used by both the fast score and final conditional model.
    double_het: Vec<u64>,
    /// Number of set bits in `double_het`, including H calls imputed to (1,1).
    n_double_het: usize,
}

impl PairDxdMasks {
    pub(crate) fn double_het_mask(&self) -> &[u64] {
        &self.double_het
    }

    pub(crate) fn n_double_het(&self) -> usize {
        self.n_double_het
    }
}

#[derive(Default)]
pub(crate) struct PairScreenScratch {
    phase_masks: PairPhaseMasks,
    double_het_mask: Vec<u64>,
}

impl PairScreenScratch {
    pub(crate) fn recycle_phase_masks(&mut self, phase_masks: PairPhaseMasks) {
        self.phase_masks = phase_masks;
    }

    pub(crate) fn recycle_dxd_masks(&mut self, masks: PairDxdMasks) {
        self.double_het_mask = masks.double_het;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PairScreenPreparation {
    QcRejected,
    NoCompatibleDonor,
    Ready {
        counts: PairCounts,
        phase_masks: PairPhaseMasks,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PairDxdScreenPreparation {
    QcRejected,
    NoCompatibleDonor,
    Ready {
        counts: PairCounts,
        masks: PairDxdMasks,
    },
}

#[cfg(test)]
pub(crate) fn count_pair(
    left: &[GenotypeCall],
    right: &[GenotypeCall],
) -> Result<PairCounts, String> {
    if left.len() != right.len() {
        return Err(format!(
            "pair endpoint sample count mismatch: left={}, right={}",
            left.len(),
            right.len()
        ));
    }
    let mut counts = PairCounts {
        category_counts: [0; 10],
        genotype_missing: 0,
        phase_unknown: 0,
        n_total: left.len(),
    };
    for (a, b) in left.iter().zip(right) {
        match pair_observation(a, b)? {
            PairObservation::Observed(category) => counts.category_counts[category.index()] += 1,
            PairObservation::GenotypeMissing => counts.genotype_missing += 1,
            PairObservation::PhaseUnknown => counts.phase_unknown += 1,
        }
    }
    Ok(counts)
}

fn count_pair_packed_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
) -> Result<PairCounts, String> {
    if left_packed.n_samples != right_packed.n_samples {
        return Err(format!(
            "pair endpoint sample count mismatch: left={}, right={}",
            left_packed.n_samples, right_packed.n_samples
        ));
    }
    for calls in [left, right].into_iter().flatten() {
        if calls.len() != left_packed.n_samples {
            return Err("packed genotype sample count does not match pair endpoints".to_string());
        }
    }
    if left.is_some() != right.is_some() {
        return Err("pair phase calls must be supplied for both endpoints".to_string());
    }

    let mut counts = PairCounts {
        category_counts: [0; 10],
        genotype_missing: 0,
        phase_unknown: 0,
        n_total: left_packed.n_samples,
    };
    let n_words = left_packed.missing.len();

    // Preserve the scalar path's error behavior for malformed internal calls:
    // an invalid dosage only errors when the other endpoint is also observed.
    for word in 0..n_words {
        let invalid_pair = (left_packed.invalid[word] & !right_packed.missing[word])
            | (right_packed.invalid[word] & !left_packed.missing[word]);
        if invalid_pair != 0 {
            if let (Some(left), Some(right)) = (left, right) {
                let bit = invalid_pair.trailing_zeros() as usize;
                let sample = word * u64::BITS as usize + bit;
                let dose_i = left[sample].dosage.unwrap_or(0);
                let dose_j = right[sample].dosage.unwrap_or(0);
                return Err(format!("invalid diploid dosage pair ({dose_i}, {dose_j})"));
            }
            return Err("invalid diploid dosage in packed pair".to_string());
        }
    }

    for dose_i in 0..=2_u8 {
        for dose_j in 0..=2_u8 {
            if dose_i == 1 && dose_j == 1 {
                continue;
            }
            let category = category_from_dosages(dose_i, dose_j)?;
            let mut observed = 0usize;
            for word in 0..n_words {
                observed += (left_packed.dosage[dose_i as usize][word]
                    & right_packed.dosage[dose_j as usize][word])
                    .count_ones() as usize;
            }
            counts.category_counts[category.index()] = observed;
        }
    }

    let mut double_heterozygotes = 0usize;
    if !left_packed.has_explicit_phase && !right_packed.has_explicit_phase {
        for word in 0..n_words {
            let both_het = left_packed.dosage[1][word] & right_packed.dosage[1][word];
            double_heterozygotes += both_het.count_ones() as usize;
            let comparable =
                both_het & left_packed.implicit_phase[word] & right_packed.implicit_phase[word];
            let phase_discordant = left_packed.alt_on_hap0[word] ^ right_packed.alt_on_hap0[word];
            let cis = comparable & !phase_discordant;
            let trans = comparable & phase_discordant;
            counts.category_counts[PairCategory::Cis.index()] += cis.count_ones() as usize;
            counts.category_counts[PairCategory::Trans.index()] += trans.count_ones() as usize;
        }
        counts.phase_unknown =
            double_heterozygotes - counts.cis_observed() - counts.trans_observed();
    } else {
        let (Some(left), Some(right)) = (left, right) else {
            return Err("explicit VCF phase requires calls for both pair endpoints".to_string());
        };
        // Explicit VCF phase-set IDs may differ by sample. Iterate only the
        // double-heterozygote bitset and retain the exact string comparison.
        for word in 0..n_words {
            let mut both_het = left_packed.dosage[1][word] & right_packed.dosage[1][word];
            double_heterozygotes += both_het.count_ones() as usize;
            while both_het != 0 {
                let bit = both_het.trailing_zeros() as usize;
                let sample = word * u64::BITS as usize + bit;
                match pair_observation(&left[sample], &right[sample])? {
                    PairObservation::Observed(PairCategory::Cis) => {
                        counts.category_counts[PairCategory::Cis.index()] += 1;
                    }
                    PairObservation::Observed(PairCategory::Trans) => {
                        counts.category_counts[PairCategory::Trans.index()] += 1;
                    }
                    PairObservation::PhaseUnknown => counts.phase_unknown += 1,
                    PairObservation::Observed(_) | PairObservation::GenotypeMissing => {
                        return Err("internal packed double-heterozygote mismatch".to_string());
                    }
                }
                both_het &= both_het - 1;
            }
        }
    }

    for word in 0..n_words {
        counts.genotype_missing +=
            (left_packed.missing[word] | right_packed.missing[word]).count_ones() as usize;
    }
    debug_assert_eq!(
        double_heterozygotes,
        counts.cis_observed() + counts.trans_observed() + counts.phase_unknown
    );
    Ok(counts)
}

/// Count the eight non-phase genotype categories after early phase QC has
/// accepted the pair. Reuse that pass's cis/trans, missing, and unknown counts.
fn count_pair_for_screen_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    phase_counts: &PairCounts,
) -> Result<PairCounts, String> {
    // The early phase-support pass already validated the invalid masks. Keep
    // the lightweight shape checks here, then count all categories per word.
    validate_packed_pair_shapes_optional(left, right, left_packed, right_packed)?;
    let mut counts = phase_counts.clone();
    let n_words = left_packed.missing.len();

    for word in 0..n_words {
        let left_dosage = [
            left_packed.dosage[0][word],
            left_packed.dosage[1][word],
            left_packed.dosage[2][word],
        ];
        let right_dosage = [
            right_packed.dosage[0][word],
            right_packed.dosage[1][word],
            right_packed.dosage[2][word],
        ];
        counts.category_counts[PairCategory::D00.index()] +=
            (left_dosage[0] & right_dosage[0]).count_ones() as usize;
        counts.category_counts[PairCategory::D01.index()] +=
            (left_dosage[0] & right_dosage[1]).count_ones() as usize;
        counts.category_counts[PairCategory::D02.index()] +=
            (left_dosage[0] & right_dosage[2]).count_ones() as usize;
        counts.category_counts[PairCategory::D10.index()] +=
            (left_dosage[1] & right_dosage[0]).count_ones() as usize;
        counts.category_counts[PairCategory::D12.index()] +=
            (left_dosage[1] & right_dosage[2]).count_ones() as usize;
        counts.category_counts[PairCategory::D20.index()] +=
            (left_dosage[2] & right_dosage[0]).count_ones() as usize;
        counts.category_counts[PairCategory::D21.index()] +=
            (left_dosage[2] & right_dosage[1]).count_ones() as usize;
        counts.category_counts[PairCategory::D22.index()] +=
            (left_dosage[2] & right_dosage[2]).count_ones() as usize;
    }
    Ok(counts)
}

fn validate_packed_pair_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
) -> Result<(), String> {
    validate_packed_pair_shapes_optional(left, right, left_packed, right_packed)?;
    validate_packed_pair_invalid_masks_optional(left, right, left_packed, right_packed)
}

fn validate_packed_pair_invalid_masks_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
) -> Result<(), String> {
    for word in 0..left_packed.missing.len() {
        validate_packed_pair_invalid_word(left, right, left_packed, right_packed, word)?;
    }
    Ok(())
}

fn validate_packed_pair_invalid_word(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    word: usize,
) -> Result<(), String> {
    let invalid_pair = (left_packed.invalid[word] & !right_packed.missing[word])
        | (right_packed.invalid[word] & !left_packed.missing[word]);
    if invalid_pair == 0 {
        return Ok(());
    }
    Err(packed_pair_invalid_error(left, right, word, invalid_pair))
}

fn packed_pair_invalid_error(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    word: usize,
    invalid_pair: u64,
) -> String {
    if let (Some(left), Some(right)) = (left, right) {
        let bit = invalid_pair.trailing_zeros() as usize;
        let sample = word * u64::BITS as usize + bit;
        let dose_i = left[sample].dosage.unwrap_or(0);
        let dose_j = right[sample].dosage.unwrap_or(0);
        return format!("invalid diploid dosage pair ({dose_i}, {dose_j})");
    }
    "invalid diploid dosage in packed pair".to_string()
}

fn validate_packed_pair_shapes_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
) -> Result<(), String> {
    if left_packed.n_samples != right_packed.n_samples {
        return Err(format!(
            "pair endpoint sample count mismatch: left={}, right={}",
            left_packed.n_samples, right_packed.n_samples
        ));
    }
    for calls in [left, right].into_iter().flatten() {
        if calls.len() != left_packed.n_samples {
            return Err("packed genotype sample count does not match pair endpoints".to_string());
        }
    }
    if left.is_some() != right.is_some() {
        return Err("pair phase calls must be supplied for both endpoints".to_string());
    }
    Ok(())
}

fn count_pair_phase_support_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
) -> Result<PairCounts, String> {
    validate_packed_pair_shapes_optional(left, right, left_packed, right_packed)?;
    // Invalid dosage masks are normally all zero for packed BED data. Cache
    // that fact per site so valid endpoint pairs avoid an otherwise redundant
    // full word traversal during early QC.
    if left_packed.has_invalid_dosage() || right_packed.has_invalid_dosage() {
        validate_packed_pair_invalid_masks_optional(left, right, left_packed, right_packed)?;
    }
    let mut counts = PairCounts {
        category_counts: [0; 10],
        genotype_missing: 0,
        phase_unknown: 0,
        n_total: left_packed.n_samples,
    };
    let explicit_phase = left_packed.has_explicit_phase || right_packed.has_explicit_phase;
    for word in 0..left_packed.missing.len() {
        counts.genotype_missing +=
            (left_packed.missing[word] | right_packed.missing[word]).count_ones() as usize;
        let mut both_het = left_packed.dosage[1][word] & right_packed.dosage[1][word];
        if !explicit_phase {
            let comparable =
                both_het & left_packed.implicit_phase[word] & right_packed.implicit_phase[word];
            let phase_discordant = left_packed.alt_on_hap0[word] ^ right_packed.alt_on_hap0[word];
            let cis = comparable & !phase_discordant;
            let trans = comparable & phase_discordant;
            counts.category_counts[PairCategory::Cis.index()] += cis.count_ones() as usize;
            counts.category_counts[PairCategory::Trans.index()] += trans.count_ones() as usize;
            counts.phase_unknown += (both_het & !(cis | trans)).count_ones() as usize;
            continue;
        }
        let (Some(left), Some(right)) = (left, right) else {
            return Err("explicit VCF phase requires calls for both pair endpoints".to_string());
        };
        while both_het != 0 {
            let bit = both_het.trailing_zeros() as usize;
            let sample = word * u64::BITS as usize + bit;
            match pair_observation(&left[sample], &right[sample])? {
                PairObservation::Observed(PairCategory::Cis) => {
                    counts.category_counts[PairCategory::Cis.index()] += 1;
                }
                PairObservation::Observed(PairCategory::Trans) => {
                    counts.category_counts[PairCategory::Trans.index()] += 1;
                }
                PairObservation::PhaseUnknown => counts.phase_unknown += 1,
                PairObservation::Observed(_) | PairObservation::GenotypeMissing => {
                    return Err("internal packed double-heterozygote mismatch".to_string());
                }
            }
            both_het &= both_het - 1;
        }
    }
    Ok(counts)
}

#[cfg(test)]
fn build_pair_phase_masks_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    counts: &PairCounts,
) -> Result<PairPhaseMasks, String> {
    validate_packed_pair_optional(left, right, left_packed, right_packed)?;
    let n_words = left_packed.missing.len();
    let mut masks = PairPhaseMasks {
        n_samples: left_packed.n_samples,
        cis: vec![0; n_words],
        trans: vec![0; n_words],
        #[cfg(test)]
        unknown: vec![0; n_words],
        n_cis: 0,
        n_trans: 0,
    };
    let explicit_phase = left_packed.has_explicit_phase || right_packed.has_explicit_phase;
    let modes = compatible_mode_table(counts);
    for word in 0..n_words {
        let both_het = left_packed.dosage[1][word] & right_packed.dosage[1][word];
        let (cis, trans, phase_unknown) = if !explicit_phase {
            let comparable =
                both_het & left_packed.implicit_phase[word] & right_packed.implicit_phase[word];
            let phase_discordant = left_packed.alt_on_hap0[word] ^ right_packed.alt_on_hap0[word];
            let cis = comparable & !phase_discordant;
            let trans = comparable & phase_discordant;
            (cis, trans, both_het & !(cis | trans))
        } else {
            let (Some(left), Some(right)) = (left, right) else {
                return Err("explicit VCF phase requires calls for both pair endpoints".to_string());
            };
            let mut cis = 0_u64;
            let mut trans = 0_u64;
            let mut phase_unknown = 0_u64;
            let mut bits = both_het;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let sample = word * u64::BITS as usize + bit;
                match pair_observation(&left[sample], &right[sample])? {
                    PairObservation::Observed(PairCategory::Cis) => cis |= 1_u64 << bit,
                    PairObservation::Observed(PairCategory::Trans) => trans |= 1_u64 << bit,
                    PairObservation::PhaseUnknown => phase_unknown |= 1_u64 << bit,
                    PairObservation::Observed(_) | PairObservation::GenotypeMissing => {
                        return Err("internal packed double-heterozygote mismatch".to_string());
                    }
                }
                bits &= bits - 1;
            }
            (cis, trans, phase_unknown)
        };
        masks.cis[word] = cis;
        masks.trans[word] = trans;
        masks.unknown[word] =
            left_packed.missing[word] | right_packed.missing[word] | phase_unknown;

        let left_patterns = [
            left_packed.dosage[0][word],
            left_packed.dosage[1][word],
            left_packed.dosage[2][word],
            left_packed.missing[word] | left_packed.invalid[word],
        ];
        let right_patterns = [
            right_packed.dosage[0][word],
            right_packed.dosage[1][word],
            right_packed.dosage[2][word],
            right_packed.missing[word] | right_packed.invalid[word],
        ];
        for (left_pattern, &left_mask) in left_patterns.iter().enumerate() {
            for (right_pattern, &right_mask) in right_patterns.iter().enumerate() {
                let imputed = masks.unknown[word] & left_mask & right_mask;
                if imputed == 0 {
                    continue;
                }
                match modes[left_pattern][right_pattern] {
                    Some(PairCategory::Cis) => masks.cis[word] |= imputed,
                    Some(PairCategory::Trans) => masks.trans[word] |= imputed,
                    Some(_) | None => {}
                }
            }
        }
        masks.n_cis += masks.cis[word].count_ones() as usize;
        masks.n_trans += masks.trans[word].count_ones() as usize;
    }
    Ok(masks)
}

#[cfg(test)]
fn has_compatible_imputation(
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    counts: &PairCounts,
    phase_masks: &PairPhaseMasks,
) -> bool {
    let modes = compatible_mode_table(counts);
    let unknown_masks = phase_masks.unknown_mask();
    for (word, &unknown) in unknown_masks.iter().enumerate() {
        if unknown == 0 {
            continue;
        }
        let left_patterns = [
            left_packed.dosage[0][word],
            left_packed.dosage[1][word],
            left_packed.dosage[2][word],
            left_packed.missing[word] | left_packed.invalid[word],
        ];
        let right_patterns = [
            right_packed.dosage[0][word],
            right_packed.dosage[1][word],
            right_packed.dosage[2][word],
            right_packed.missing[word] | right_packed.invalid[word],
        ];
        for (left_pattern, left_mask) in left_patterns.iter().enumerate() {
            for (right_pattern, right_mask) in right_patterns.iter().enumerate() {
                if unknown & left_mask & right_mask != 0
                    && modes[left_pattern][right_pattern].is_none()
                {
                    return false;
                }
            }
        }
    }
    true
}

/// Build the screen's phase masks and test all imputed observations against
/// the same compatibility table in one pass. A missing donor makes the pair
/// untestable immediately, so no later scan over `unknown` words is needed.
fn build_compatible_phase_masks_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    counts: &PairCounts,
    masks: &mut PairPhaseMasks,
) -> Result<bool, String> {
    // The phase-support QC pass already checked invalid dosage combinations.
    validate_packed_pair_shapes_optional(left, right, left_packed, right_packed)?;
    let n_words = left_packed.missing.len();
    masks.n_samples = left_packed.n_samples;
    masks.cis.resize(n_words, 0);
    masks.trans.resize(n_words, 0);
    masks.cis.fill(0);
    masks.trans.fill(0);
    #[cfg(test)]
    {
        masks.unknown.resize(n_words, 0);
        masks.unknown.fill(0);
    }
    masks.n_cis = 0;
    masks.n_trans = 0;
    let explicit_phase = left_packed.has_explicit_phase || right_packed.has_explicit_phase;
    let mut modes = None;

    for word in 0..n_words {
        let both_het = left_packed.dosage[1][word] & right_packed.dosage[1][word];
        let (cis, trans, phase_unknown) = if !explicit_phase {
            let comparable =
                both_het & left_packed.implicit_phase[word] & right_packed.implicit_phase[word];
            let phase_discordant = left_packed.alt_on_hap0[word] ^ right_packed.alt_on_hap0[word];
            let cis = comparable & !phase_discordant;
            let trans = comparable & phase_discordant;
            (cis, trans, both_het & !(cis | trans))
        } else {
            let (Some(left), Some(right)) = (left, right) else {
                return Err("explicit VCF phase requires calls for both pair endpoints".to_string());
            };
            let mut cis = 0_u64;
            let mut trans = 0_u64;
            let mut phase_unknown = 0_u64;
            let mut bits = both_het;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let sample = word * u64::BITS as usize + bit;
                match pair_observation(&left[sample], &right[sample])? {
                    PairObservation::Observed(PairCategory::Cis) => cis |= 1_u64 << bit,
                    PairObservation::Observed(PairCategory::Trans) => trans |= 1_u64 << bit,
                    PairObservation::PhaseUnknown => phase_unknown |= 1_u64 << bit,
                    PairObservation::Observed(_) | PairObservation::GenotypeMissing => {
                        return Err("internal packed double-heterozygote mismatch".to_string());
                    }
                }
                bits &= bits - 1;
            }
            (cis, trans, phase_unknown)
        };
        masks.cis[word] = cis;
        masks.trans[word] = trans;
        let unknown = left_packed.missing[word] | right_packed.missing[word] | phase_unknown;
        #[cfg(test)]
        {
            masks.unknown[word] = unknown;
        }
        if unknown == 0 {
            masks.n_cis += masks.cis[word].count_ones() as usize;
            masks.n_trans += masks.trans[word].count_ones() as usize;
            continue;
        }
        let left_patterns = [
            left_packed.dosage[0][word],
            left_packed.dosage[1][word],
            left_packed.dosage[2][word],
            left_packed.missing[word] | left_packed.invalid[word],
        ];
        let right_patterns = [
            right_packed.dosage[0][word],
            right_packed.dosage[1][word],
            right_packed.dosage[2][word],
            right_packed.missing[word] | right_packed.invalid[word],
        ];
        for (left_pattern, &left_mask) in left_patterns.iter().enumerate() {
            for (right_pattern, &right_mask) in right_patterns.iter().enumerate() {
                let imputed = unknown & left_mask & right_mask;
                if imputed == 0 {
                    continue;
                }
                let modes = modes.get_or_insert_with(|| compatible_mode_table(counts));
                match modes[left_pattern][right_pattern] {
                    Some(PairCategory::Cis) => masks.cis[word] |= imputed,
                    Some(PairCategory::Trans) => masks.trans[word] |= imputed,
                    Some(_) => {}
                    None => return Ok(false),
                }
            }
        }
        masks.n_cis += masks.cis[word].count_ones() as usize;
        masks.n_trans += masks.trans[word].count_ones() as usize;
    }
    Ok(true)
}

#[cfg(test)]
pub(crate) fn prepare_pair_encoded(
    left: &[GenotypeCall],
    right: &[GenotypeCall],
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    thresholds: PairQcThresholds,
) -> Result<EncodedPairPreparation, String> {
    prepare_pair_encoded_optional(
        Some(left),
        Some(right),
        left_packed,
        right_packed,
        thresholds,
    )
}

pub(crate) fn prepare_pair_encoded_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    thresholds: PairQcThresholds,
) -> Result<EncodedPairPreparation, String> {
    let counts = count_pair_packed_optional(left, right, left_packed, right_packed)?;
    if !passes_pair_qc(&counts, thresholds) {
        return Ok(EncodedPairPreparation::QcRejected(counts));
    }
    match encode_pair(left, right, left_packed, right_packed, &counts) {
        Some(pair) => Ok(EncodedPairPreparation::Ready { counts, pair }),
        None => Ok(EncodedPairPreparation::NoCompatibleDonor(counts)),
    }
}

/// Fast preparation for ranking: reject pairs from cis/trans support and
/// missingness first, then retain only the H/Z masks needed by the screen.
/// The full ten-category genotype vectors are created only after TopK chooses
/// the pair.
pub(crate) fn prepare_pair_for_screen_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    thresholds: PairQcThresholds,
) -> Result<PairScreenPreparation, String> {
    let mut scratch = PairScreenScratch::default();
    prepare_pair_for_screen_with_scratch_optional(
        left,
        right,
        left_packed,
        right_packed,
        thresholds,
        &mut scratch,
    )
}

pub(crate) fn prepare_pair_for_screen_with_scratch_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    thresholds: PairQcThresholds,
    scratch: &mut PairScreenScratch,
) -> Result<PairScreenPreparation, String> {
    let phase_counts = count_pair_phase_support_optional(left, right, left_packed, right_packed)?;
    if !passes_pair_qc(&phase_counts, thresholds) {
        return Ok(PairScreenPreparation::QcRejected);
    }

    let counts =
        count_pair_for_screen_optional(left, right, left_packed, right_packed, &phase_counts)?;
    if !passes_pair_qc(&counts, thresholds) {
        return Err("pair QC support changed between packed passes".to_string());
    }
    if !build_compatible_phase_masks_optional(
        left,
        right,
        left_packed,
        right_packed,
        &counts,
        &mut scratch.phase_masks,
    )? {
        return Ok(PairScreenPreparation::NoCompatibleDonor);
    }
    let phase_masks = std::mem::take(&mut scratch.phase_masks);
    Ok(PairScreenPreparation::Ready {
        counts,
        phase_masks,
    })
}

/// Phase-independent pair preparation for the D×D genome scan. The H mask
/// includes observed double heterozygotes, regardless of phase, and missing
/// pairs imputed to dosage pair (1,1), matching the final conditional design.
pub(crate) fn prepare_pair_for_dxd_screen_with_scratch_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    thresholds: PairQcThresholds,
    scratch: &mut PairScreenScratch,
) -> Result<PairDxdScreenPreparation, String> {
    validate_packed_pair_shapes_optional(left, right, left_packed, right_packed)?;
    if left_packed.has_invalid_dosage() || right_packed.has_invalid_dosage() {
        validate_packed_pair_invalid_masks_optional(left, right, left_packed, right_packed)?;
    }
    let phase_agnostic_counts = count_pair_dxd_packed(left_packed, right_packed);
    if !passes_pair_dxd_qc(&phase_agnostic_counts, thresholds) {
        return Ok(PairDxdScreenPreparation::QcRejected);
    }
    let counts = count_pair_for_screen_optional(
        left,
        right,
        left_packed,
        right_packed,
        &phase_agnostic_counts,
    )?;
    scratch.double_het_mask.resize(left_packed.missing.len(), 0);
    if !build_dxd_double_het_mask(
        left_packed,
        right_packed,
        &counts,
        &mut scratch.double_het_mask,
    ) {
        return Ok(PairDxdScreenPreparation::NoCompatibleDonor);
    }
    let mut n_double_het = 0usize;
    for mask in &scratch.double_het_mask {
        n_double_het += mask.count_ones() as usize;
    }
    let masks = PairDxdMasks {
        double_het: std::mem::take(&mut scratch.double_het_mask),
        n_double_het,
    };
    Ok(PairDxdScreenPreparation::Ready { counts, masks })
}

pub(crate) fn materialize_encoded_pair_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    counts: &PairCounts,
) -> Option<EncodedPair> {
    encode_pair(left, right, left_packed, right_packed, counts)
}

/// Encode a pair for the reduced no-phase model. Cis and trans are collapsed
/// into the same `(1,1)` genotype category for mode imputation; the materialized
/// phase vector is all zero because this candidate is not a phase test.
pub(crate) fn materialize_encoded_pair_dxd_optional(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    counts: &PairCounts,
) -> Option<EncodedPair> {
    validate_packed_pair_shapes_optional(left, right, left_packed, right_packed).ok()?;
    if left_packed.has_invalid_dosage() || right_packed.has_invalid_dosage() {
        validate_packed_pair_invalid_masks_optional(left, right, left_packed, right_packed).ok()?;
    }
    encode_pair_dxd(left_packed, right_packed, counts)
}

fn encode_pair(
    left: Option<&[GenotypeCall]>,
    right: Option<&[GenotypeCall]>,
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    counts: &PairCounts,
) -> Option<EncodedPair> {
    let n_samples = left_packed.n_samples;
    let n_words = left_packed.missing.len();
    let mut category_masks = std::array::from_fn(|_| vec![0_u64; n_words]);
    let mut imputed_mask = vec![0_u64; n_words];

    for dose_i in 0..=2_usize {
        for dose_j in 0..=2_usize {
            if dose_i == 1 && dose_j == 1 {
                continue;
            }
            let category = category_from_dosages(dose_i as u8, dose_j as u8).ok()?;
            for word in 0..n_words {
                category_masks[category.index()][word] =
                    left_packed.dosage[dose_i][word] & right_packed.dosage[dose_j][word];
            }
        }
    }

    let explicit_phase = left_packed.has_explicit_phase || right_packed.has_explicit_phase;
    let phase_calls = if explicit_phase {
        Some((left?, right?))
    } else {
        None
    };
    let mut modes = None;
    let mut n_imputed = 0usize;
    for word in 0..n_words {
        let both_het = left_packed.dosage[1][word] & right_packed.dosage[1][word];
        let (cis, trans, phase_unknown) = if !explicit_phase {
            let comparable =
                both_het & left_packed.implicit_phase[word] & right_packed.implicit_phase[word];
            let phase_discordant = left_packed.alt_on_hap0[word] ^ right_packed.alt_on_hap0[word];
            let cis = comparable & !phase_discordant;
            let trans = comparable & phase_discordant;
            (cis, trans, both_het & !(cis | trans))
        } else {
            let (left, right) = phase_calls.expect("explicit phase calls validated");
            let mut cis = 0_u64;
            let mut trans = 0_u64;
            let mut phase_unknown = 0_u64;
            let mut bits = both_het;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let sample = word * u64::BITS as usize + bit;
                match pair_observation(&left[sample], &right[sample]).ok()? {
                    PairObservation::Observed(PairCategory::Cis) => cis |= 1_u64 << bit,
                    PairObservation::Observed(PairCategory::Trans) => trans |= 1_u64 << bit,
                    PairObservation::PhaseUnknown => phase_unknown |= 1_u64 << bit,
                    PairObservation::Observed(_) | PairObservation::GenotypeMissing => return None,
                }
                bits &= bits - 1;
            }
            (cis, trans, phase_unknown)
        };
        category_masks[PairCategory::Cis.index()][word] = cis;
        category_masks[PairCategory::Trans.index()][word] = trans;

        let unknown = left_packed.missing[word] | right_packed.missing[word] | phase_unknown;
        if unknown == 0 {
            continue;
        }
        let left_patterns = [
            left_packed.dosage[0][word],
            left_packed.dosage[1][word],
            left_packed.dosage[2][word],
            left_packed.missing[word] | left_packed.invalid[word],
        ];
        let right_patterns = [
            right_packed.dosage[0][word],
            right_packed.dosage[1][word],
            right_packed.dosage[2][word],
            right_packed.missing[word] | right_packed.invalid[word],
        ];
        for (left_pattern, left_mask) in left_patterns.iter().enumerate() {
            for (right_pattern, right_mask) in right_patterns.iter().enumerate() {
                let imputed = unknown & left_mask & right_mask;
                if imputed == 0 {
                    continue;
                }
                let modes = modes.get_or_insert_with(|| compatible_mode_table(counts));
                let category = modes[left_pattern][right_pattern]?;
                category_masks[category.index()][word] |= imputed;
                imputed_mask[word] |= imputed;
                n_imputed += imputed.count_ones() as usize;
            }
        }
    }

    let expected = sample_tail_mask(n_samples);
    for word in 0..n_words {
        let covered = category_masks
            .iter()
            .fold(0_u64, |union, mask| union | mask[word]);
        let target = if word + 1 == n_words {
            expected
        } else {
            u64::MAX
        };
        debug_assert_eq!(covered, target, "encoded pair must assign every sample");
    }

    Some(EncodedPair {
        n_samples,
        category_masks,
        imputed_mask,
        n_imputed,
        phase_independent: false,
    })
}

fn encode_pair_dxd(
    left_packed: &PackedGenotypes,
    right_packed: &PackedGenotypes,
    counts: &PairCounts,
) -> Option<EncodedPair> {
    let n_samples = left_packed.n_samples;
    let n_words = left_packed.missing.len();
    let mut category_masks = std::array::from_fn(|_| vec![0_u64; n_words]);
    let mut imputed_mask = vec![0_u64; n_words];

    for dose_i in 0..=2_usize {
        for dose_j in 0..=2_usize {
            let category = if dose_i == 1 && dose_j == 1 {
                PairCategory::Cis
            } else {
                category_from_dosages(dose_i as u8, dose_j as u8).ok()?
            };
            for word in 0..n_words {
                category_masks[category.index()][word] |=
                    left_packed.dosage[dose_i][word] & right_packed.dosage[dose_j][word];
            }
        }
    }

    let modes = unphased_mode_table(counts);
    let mut n_imputed = 0usize;
    for word in 0..n_words {
        let unknown = left_packed.missing[word]
            | right_packed.missing[word]
            | left_packed.invalid[word]
            | right_packed.invalid[word];
        if unknown == 0 {
            continue;
        }
        let left_patterns = [
            left_packed.dosage[0][word],
            left_packed.dosage[1][word],
            left_packed.dosage[2][word],
            left_packed.missing[word] | left_packed.invalid[word],
        ];
        let right_patterns = [
            right_packed.dosage[0][word],
            right_packed.dosage[1][word],
            right_packed.dosage[2][word],
            right_packed.missing[word] | right_packed.invalid[word],
        ];
        for (left_pattern, left_mask) in left_patterns.iter().enumerate() {
            for (right_pattern, right_mask) in right_patterns.iter().enumerate() {
                let imputed = unknown & left_mask & right_mask;
                if imputed == 0 {
                    continue;
                }
                let (dose_i, dose_j) = modes[left_pattern][right_pattern]?;
                let category = if dose_i == 1 && dose_j == 1 {
                    PairCategory::Cis
                } else {
                    category_from_dosages(dose_i, dose_j).ok()?
                };
                category_masks[category.index()][word] |= imputed;
                imputed_mask[word] |= imputed;
                n_imputed += imputed.count_ones() as usize;
            }
        }
    }

    let expected = sample_tail_mask(n_samples);
    for word in 0..n_words {
        let covered = category_masks
            .iter()
            .fold(0_u64, |union, mask| union | mask[word]);
        let target = if word + 1 == n_words {
            expected
        } else {
            u64::MAX
        };
        debug_assert_eq!(covered, target, "encoded D×D pair must assign every sample");
    }

    Some(EncodedPair {
        n_samples,
        category_masks,
        imputed_mask,
        n_imputed,
        phase_independent: true,
    })
}

fn sample_tail_mask(n_samples: usize) -> u64 {
    let remainder = n_samples % u64::BITS as usize;
    if remainder == 0 {
        u64::MAX
    } else {
        (1_u64 << remainder) - 1
    }
}

fn pair_observation(left: &GenotypeCall, right: &GenotypeCall) -> Result<PairObservation, String> {
    let (Some(g_i), Some(g_j)) = (left.dosage, right.dosage) else {
        return Ok(PairObservation::GenotypeMissing);
    };
    if g_i > 2 || g_j > 2 {
        return Err(format!("invalid diploid dosage pair ({g_i}, {g_j})"));
    }
    if g_i != 1 || g_j != 1 {
        return Ok(PairObservation::Observed(category_from_dosages(g_i, g_j)?));
    }
    if !phase_sets_comparable(&left.phase_set, &right.phase_set) {
        return Ok(PairObservation::PhaseUnknown);
    }
    let (Some(left_alt_hap0), Some(right_alt_hap0)) = (left.alt_on_hap0, right.alt_on_hap0) else {
        return Ok(PairObservation::PhaseUnknown);
    };
    let category = if left_alt_hap0 == right_alt_hap0 {
        PairCategory::Cis
    } else {
        PairCategory::Trans
    };
    Ok(PairObservation::Observed(category))
}

fn phase_sets_comparable(left: &PhaseSet, right: &PhaseSet) -> bool {
    match (left, right) {
        (PhaseSet::Implicit, PhaseSet::Implicit) => true,
        (PhaseSet::Explicit(a), PhaseSet::Explicit(b)) => a == b,
        _ => false,
    }
}

fn category_from_dosages(g_i: u8, g_j: u8) -> Result<PairCategory, String> {
    match (g_i, g_j) {
        (0, 0) => Ok(PairCategory::D00),
        (0, 1) => Ok(PairCategory::D01),
        (0, 2) => Ok(PairCategory::D02),
        (1, 0) => Ok(PairCategory::D10),
        (1, 1) => Err("double heterozygote requires a phase category".to_string()),
        (1, 2) => Ok(PairCategory::D12),
        (2, 0) => Ok(PairCategory::D20),
        (2, 1) => Ok(PairCategory::D21),
        (2, 2) => Ok(PairCategory::D22),
        _ => Err(format!("invalid diploid dosage pair ({g_i}, {g_j})")),
    }
}

pub(crate) fn passes_pair_qc(counts: &PairCounts, thresholds: PairQcThresholds) -> bool {
    if !thresholds.min_phase_minor_freq.is_finite()
        || !(0.0..=0.5).contains(&thresholds.min_phase_minor_freq)
        || !thresholds.max_unknown_rate.is_finite()
        || !(0.0..=1.0).contains(&thresholds.max_unknown_rate)
        || counts.n_total == 0
        || counts.n_informative() == 0
    {
        return false;
    }
    let n_cis = counts.cis_observed();
    let n_trans = counts.trans_observed();
    if n_cis == 0 || n_trans == 0 {
        return false;
    }
    let n_informative = counts.n_informative();
    let phase_minor_freq = n_cis.min(n_trans) as f64 / n_informative as f64;
    let unknown_rate = counts.n_unknown() as f64 / counts.n_total as f64;
    phase_minor_freq >= thresholds.min_phase_minor_freq
        && unknown_rate <= thresholds.max_unknown_rate
        && n_cis >= thresholds.min_cis
        && n_trans >= thresholds.min_trans
}

fn count_pair_dxd_packed(left: &PackedGenotypes, right: &PackedGenotypes) -> PairCounts {
    let mut counts = PairCounts {
        category_counts: [0; 10],
        genotype_missing: 0,
        // In this phase-independent view, this field carries the full dosage
        // (1,1) count, including calls that happened to have known phase.
        phase_unknown: 0,
        n_total: left.n_samples,
    };
    for word in 0..left.missing.len() {
        counts.genotype_missing += (left.missing[word] | right.missing[word]).count_ones() as usize;
        counts.phase_unknown +=
            (left.dosage[1][word] & right.dosage[1][word]).count_ones() as usize;
    }
    counts
}

fn passes_pair_dxd_qc(counts: &PairCounts, thresholds: PairQcThresholds) -> bool {
    thresholds.max_unknown_rate.is_finite()
        && (0.0..=1.0).contains(&thresholds.max_unknown_rate)
        && counts.n_total > 0
        && counts.genotype_missing < counts.n_total
        && counts.phase_unknown > 0
        && counts.genotype_missing as f64 / counts.n_total as f64 <= thresholds.max_unknown_rate
}

fn unphased_genotype_pair_count(counts: &PairCounts, left: u8, right: u8) -> usize {
    match (left, right) {
        (1, 1) => counts.cis_observed() + counts.trans_observed() + counts.phase_unknown,
        (1, 0) => counts.category_counts[PairCategory::D10.index()],
        _ => category_from_dosages(left, right)
            .map(|category| counts.category_counts[category.index()])
            .unwrap_or(0),
    }
}

/// Deterministic modal donors for four genotype states per site: 0, 1, 2,
/// missing. Iterating dosage pairs in row-major order makes ties stable and
/// folds all double heterozygotes into one `(1,1)` donor count.
fn unphased_mode_table(counts: &PairCounts) -> [[Option<(u8, u8)>; 4]; 4] {
    let mut table = [[None; 4]; 4];
    for (left_pattern, row) in table.iter_mut().enumerate() {
        for (right_pattern, slot) in row.iter_mut().enumerate() {
            let mut best: Option<((u8, u8), usize)> = None;
            for left in 0..=2_u8 {
                if left_pattern != 3 && left_pattern != left as usize {
                    continue;
                }
                for right in 0..=2_u8 {
                    if right_pattern != 3 && right_pattern != right as usize {
                        continue;
                    }
                    let count = unphased_genotype_pair_count(counts, left, right);
                    if count > 0 && best.is_none_or(|(_, best_count)| count > best_count) {
                        best = Some(((left, right), count));
                    }
                }
            }
            *slot = best.map(|(pair, _)| pair);
        }
    }
    table
}

fn build_dxd_double_het_mask(
    left: &PackedGenotypes,
    right: &PackedGenotypes,
    counts: &PairCounts,
    double_het_mask: &mut Vec<u64>,
) -> bool {
    let modes = unphased_mode_table(counts);
    double_het_mask.resize(left.missing.len(), 0);
    for word in 0..left.missing.len() {
        let mut h_mask = left.dosage[1][word] & right.dosage[1][word];
        let unknown =
            left.missing[word] | right.missing[word] | left.invalid[word] | right.invalid[word];
        let left_patterns = [
            left.dosage[0][word],
            left.dosage[1][word],
            left.dosage[2][word],
            left.missing[word] | left.invalid[word],
        ];
        let right_patterns = [
            right.dosage[0][word],
            right.dosage[1][word],
            right.dosage[2][word],
            right.missing[word] | right.invalid[word],
        ];
        for (left_pattern, left_mask) in left_patterns.iter().enumerate() {
            for (right_pattern, right_mask) in right_patterns.iter().enumerate() {
                let imputed = unknown & left_mask & right_mask;
                if imputed != 0 {
                    match modes[left_pattern][right_pattern] {
                        Some((1, 1)) => h_mask |= imputed,
                        Some(_) => {}
                        None => return false,
                    }
                }
            }
        }
        double_het_mask[word] = h_mask;
    }
    true
}

#[cfg(test)]
pub(crate) fn prepare_pair(
    left: &[GenotypeCall],
    right: &[GenotypeCall],
    thresholds: PairQcThresholds,
) -> Result<PairPreparation, String> {
    let counts = count_pair(left, right)?;
    if !passes_pair_qc(&counts, thresholds) {
        return Ok(PairPreparation::QcRejected(counts));
    }
    match mode_fill_pair(left, right, &counts) {
        Some(filled) => Ok(PairPreparation::Ready { counts, filled }),
        None => Ok(PairPreparation::NoCompatibleDonor(counts)),
    }
}

#[cfg(test)]
pub(crate) fn mode_fill_pair(
    left: &[GenotypeCall],
    right: &[GenotypeCall],
    counts: &PairCounts,
) -> Option<FilledPair> {
    if left.len() != right.len() || left.len() != counts.n_total {
        return None;
    }
    let mut z = Vec::with_capacity(left.len());
    let mut g_i = Vec::with_capacity(left.len());
    let mut g_j = Vec::with_capacity(left.len());
    let mut imputed = Vec::with_capacity(left.len());
    let mut n_imputed = 0usize;
    let mut compatibility_modes = None;

    for (a, b) in left.iter().zip(right) {
        let observed = pair_observation(a, b).ok()?;
        let (category, was_imputed) = match observed {
            PairObservation::Observed(category) => (category, false),
            PairObservation::GenotypeMissing | PairObservation::PhaseUnknown => {
                let modes =
                    compatibility_modes.get_or_insert_with(|| compatible_mode_table(counts));
                let category = modes[dose_pattern(a.dosage)][dose_pattern(b.dosage)]?;
                n_imputed += 1;
                (category, true)
            }
        };
        let (dose_i, dose_j) = category.dosages();
        z.push(category.z());
        g_i.push(dose_i);
        g_j.push(dose_j);
        imputed.push(was_imputed);
    }

    Some(FilledPair {
        z,
        g_i,
        g_j,
        imputed,
        n_imputed,
    })
}

#[cfg(test)]
fn dose_pattern(dosage: Option<u8>) -> usize {
    dosage.map_or(3, |dose| dose as usize)
}

fn compatible_mode_table(counts: &PairCounts) -> [[Option<PairCategory>; 4]; 4] {
    #[cfg(test)]
    COMPATIBLE_MODE_TABLE_BUILDS.with(|builds| builds.set(builds.get() + 1));
    std::array::from_fn(|left_pattern| {
        std::array::from_fn(|right_pattern| {
            let left_dose = (left_pattern < 3).then_some(left_pattern as u8);
            let right_dose = (right_pattern < 3).then_some(right_pattern as u8);
            compatible_mode_for_dosages(left_dose, right_dose, counts)
        })
    })
}

#[cfg(test)]
std::thread_local! {
    static COMPATIBLE_MODE_TABLE_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn compatible_mode_for_dosages(
    left_dosage: Option<u8>,
    right_dosage: Option<u8>,
    counts: &PairCounts,
) -> Option<PairCategory> {
    let mut best: Option<(PairCategory, usize)> = None;
    for category in PairCategory::ALL {
        let (g_i, g_j) = category.dosages();
        let compatible = match (left_dosage, right_dosage) {
            (Some(a), Some(b)) if a == 1 && b == 1 => {
                matches!(category, PairCategory::Cis | PairCategory::Trans)
            }
            (Some(a), Some(b)) => (a, b) == (g_i, g_j),
            (Some(a), None) => a == g_i,
            (None, Some(b)) => b == g_j,
            (None, None) => true,
        };
        if !compatible {
            continue;
        }
        let count = counts.category_counts[category.index()];
        if count == 0 {
            continue;
        }
        if best.is_none_or(|(_, best_count)| count > best_count) {
            best = Some((category, count));
        }
    }
    best.map(|(category, _)| category)
}

#[cfg(test)]
mod tests {
    use super::{
        build_pair_phase_masks_optional, count_pair, count_pair_for_screen_optional,
        count_pair_packed_optional, count_pair_phase_support_optional, has_compatible_imputation,
        materialize_encoded_pair_dxd_optional, materialize_encoded_pair_optional, mode_fill_pair,
        passes_pair_qc, prepare_pair, prepare_pair_encoded,
        prepare_pair_for_dxd_screen_with_scratch_optional, prepare_pair_for_screen_optional,
        prepare_pair_for_screen_with_scratch_optional, EncodedPairPreparation, PackedGenotypes,
        PairCategory, PairCounts, PairDxdScreenPreparation, PairPreparation, PairQcThresholds,
        PairScreenPreparation, PairScreenScratch, COMPATIBLE_MODE_TABLE_BUILDS,
    };
    use crate::ct::input::{GenotypeCall, PhaseSet};

    fn call(dosage: Option<u8>, alt_on_hap0: Option<bool>, phase_set: PhaseSet) -> GenotypeCall {
        GenotypeCall {
            dosage,
            alt_on_hap0,
            phase_set,
        }
    }

    fn known(dosage: u8) -> GenotypeCall {
        call(Some(dosage), None, PhaseSet::Unknown)
    }

    fn phased(alt_on_hap0: bool, ps: PhaseSet) -> GenotypeCall {
        call(Some(1), Some(alt_on_hap0), ps)
    }

    #[test]
    fn packed_encoded_fill_matches_scalar_for_phase_unknown_and_missing_calls() {
        let left = vec![
            phased(true, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
            call(None, None, PhaseSet::Unknown),
            phased(true, PhaseSet::Unknown),
        ];
        let right = vec![
            phased(true, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
            known(1),
            phased(false, PhaseSet::Implicit),
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
        let observed =
            prepare_pair_encoded(&left, &right, &left_packed, &right_packed, thresholds).unwrap();

        match (expected, observed) {
            (
                PairPreparation::Ready {
                    counts: expected_counts,
                    filled: expected_filled,
                },
                EncodedPairPreparation::Ready { counts, pair },
            ) => {
                assert_eq!(counts, expected_counts);
                assert_eq!(pair.materialize(), expected_filled);
                assert_eq!(pair.n_imputed, 2);
            }
            (expected, observed) => panic!("preparation mismatch: {expected:?} vs {observed:?}"),
        }
    }

    #[test]
    fn screen_preparation_keeps_imputed_phase_masks_and_topk_materialization() {
        let left = vec![
            phased(true, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
            known(2),
            call(None, None, PhaseSet::Unknown),
            known(0),
            known(1),
            phased(true, PhaseSet::Unknown),
            known(0),
        ];
        let right = vec![
            phased(true, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
            known(0),
            known(0),
            call(None, None, PhaseSet::Unknown),
            known(1),
            phased(false, PhaseSet::Implicit),
            known(2),
        ];
        let thresholds = PairQcThresholds {
            min_phase_minor_freq: 0.2,
            max_unknown_rate: 0.5,
            min_cis: 1,
            min_trans: 1,
        };
        let left_packed = PackedGenotypes::from_calls(&left);
        let right_packed = PackedGenotypes::from_calls(&right);
        let reference = prepare_pair(&left, &right, thresholds).unwrap();
        let screened = prepare_pair_for_screen_optional(
            Some(&left),
            Some(&right),
            &left_packed,
            &right_packed,
            thresholds,
        )
        .unwrap();

        let (reference_counts, reference_filled) = match reference {
            PairPreparation::Ready { counts, filled } => (counts, filled),
            other => panic!("expected a testable reference pair, got {other:?}"),
        };
        match screened {
            PairScreenPreparation::Ready {
                counts,
                phase_masks,
            } => {
                assert_eq!(counts, reference_counts);
                let encoded = materialize_encoded_pair_optional(
                    Some(&left),
                    Some(&right),
                    &left_packed,
                    &right_packed,
                    &counts,
                )
                .expect("TopK pair must be materializable");
                assert_eq!(encoded.materialize(), reference_filled);
                assert_eq!(phase_masks.cis_mask(), encoded.cis_mask());
                assert_eq!(phase_masks.trans_mask(), encoded.trans_mask());
                assert_eq!(phase_masks.n_cis(), encoded.n_cis());
                assert_eq!(phase_masks.n_trans(), encoded.n_trans());
            }
            other => panic!("expected a screenable pair, got {other:?}"),
        }
    }

    #[test]
    fn dxd_prep_ignores_phase_support_and_imputes_over_nine_unphased_categories() {
        let left = vec![
            phased(true, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
            known(1),
            known(0),
            known(2),
            known(0),
            call(None, None, PhaseSet::Unknown),
            known(0),
        ];
        let right = vec![
            phased(true, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
            known(1),
            known(0),
            known(0),
            known(2),
            call(None, None, PhaseSet::Unknown),
            call(None, None, PhaseSet::Unknown),
        ];
        let thresholds = PairQcThresholds {
            min_phase_minor_freq: 0.5,
            max_unknown_rate: 0.25,
            min_cis: 1,
            min_trans: 1,
        };
        let left_packed = PackedGenotypes::from_calls(&left);
        let right_packed = PackedGenotypes::from_calls(&right);
        let mut scratch = PairScreenScratch::default();
        let prepared = prepare_pair_for_dxd_screen_with_scratch_optional(
            Some(&left),
            Some(&right),
            &left_packed,
            &right_packed,
            thresholds,
            &mut scratch,
        )
        .unwrap();
        let (counts, masks) = match prepared {
            PairDxdScreenPreparation::Ready { counts, masks } => (counts, masks),
            other => panic!("phase-independent pair should pass D×D QC: {other:?}"),
        };
        assert_eq!(counts.cis_observed(), 0);
        assert_eq!(counts.trans_observed(), 0);
        assert_eq!(counts.phase_unknown, 3);
        assert_eq!(counts.genotype_missing, 2);
        // The screen mask includes the one missing pair imputed to (1,1),
        // matching the H column materialized for the conditional model.
        assert_eq!(masks.n_double_het(), 4);
        assert_eq!(masks.double_het_mask()[0] & 0xff, 0b0100_0111);

        let encoded = materialize_encoded_pair_dxd_optional(
            Some(&left),
            Some(&right),
            &left_packed,
            &right_packed,
            &counts,
        )
        .expect("unphased dosage-pair donors exist");
        let filled = encoded.materialize();
        assert_eq!(filled.n_imputed, 2);
        assert_eq!(filled.g_i, vec![1, 1, 1, 0, 2, 0, 1, 0]);
        assert_eq!(filled.g_j, vec![1, 1, 1, 0, 0, 2, 1, 0]);
        assert_eq!(filled.z, vec![0; 8]);
        for sample in 0..filled.g_i.len() {
            let mask_has_h = (masks.double_het_mask()[sample / 64] & (1_u64 << (sample % 64))) != 0;
            assert_eq!(
                mask_has_h,
                filled.g_i[sample] == 1 && filled.g_j[sample] == 1,
                "D×D screen H differs from filled conditional H at sample {sample}"
            );
        }
    }

    fn reference_screen_preparation(
        left: Option<&[GenotypeCall]>,
        right: Option<&[GenotypeCall]>,
        left_packed: &PackedGenotypes,
        right_packed: &PackedGenotypes,
        thresholds: PairQcThresholds,
    ) -> Result<PairScreenPreparation, String> {
        let phase_counts =
            count_pair_phase_support_optional(left, right, left_packed, right_packed)?;
        if !passes_pair_qc(&phase_counts, thresholds) {
            return Ok(PairScreenPreparation::QcRejected);
        }
        let counts = count_pair_packed_optional(left, right, left_packed, right_packed)?;
        if !passes_pair_qc(&counts, thresholds) {
            return Err("pair QC support changed between packed passes".to_string());
        }
        let phase_masks =
            build_pair_phase_masks_optional(left, right, left_packed, right_packed, &counts)?;
        if !has_compatible_imputation(left_packed, right_packed, &counts, &phase_masks) {
            return Ok(PairScreenPreparation::NoCompatibleDonor);
        }
        Ok(PairScreenPreparation::Ready {
            counts,
            phase_masks,
        })
    }

    #[test]
    fn optimized_screen_preparation_matches_reference_for_randomized_phase_and_missingness() {
        let thresholds = PairQcThresholds {
            min_phase_minor_freq: 0.0,
            max_unknown_rate: 1.0,
            min_cis: 0,
            min_trans: 0,
        };
        for explicit in [false, true] {
            for replicate in 0..32 {
                let mut state = 0x9e3779b97f4a7c15_u64 ^ (replicate as u64 * 0x100000001b3);
                let mut next = || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state
                };
                let make_calls = |side: usize, next: &mut dyn FnMut() -> u64| {
                    (0..193)
                        .map(|sample| {
                            let random = next();
                            let dosage = if random % 19 == 0 {
                                None
                            } else {
                                Some(((random >> 9) % 3) as u8)
                            };
                            let (alt_on_hap0, phase_set) = if dosage == Some(1) {
                                let alt_on_hap0 = Some((random >> 17) & 1 != 0);
                                let phase_set = if explicit {
                                    PhaseSet::Explicit(format!(
                                        "ps-{}",
                                        (sample + side + replicate) % 5
                                    ))
                                } else if random & 0x10000 == 0 {
                                    PhaseSet::Implicit
                                } else {
                                    PhaseSet::Unknown
                                };
                                (alt_on_hap0, phase_set)
                            } else {
                                (None, PhaseSet::Unknown)
                            };
                            call(dosage, alt_on_hap0, phase_set)
                        })
                        .collect::<Vec<_>>()
                };
                let left = make_calls(0, &mut next);
                let right = make_calls(1, &mut next);
                let left_packed = PackedGenotypes::from_calls(&left);
                let right_packed = PackedGenotypes::from_calls(&right);
                let call_slices = if explicit {
                    (Some(left.as_slice()), Some(right.as_slice()))
                } else {
                    (None, None)
                };
                let expected = reference_screen_preparation(
                    call_slices.0,
                    call_slices.1,
                    &left_packed,
                    &right_packed,
                    thresholds,
                )
                .unwrap();
                let observed = prepare_pair_for_screen_optional(
                    call_slices.0,
                    call_slices.1,
                    &left_packed,
                    &right_packed,
                    thresholds,
                )
                .unwrap();
                assert_eq!(
                    observed, expected,
                    "explicit={explicit}, replicate={replicate}"
                );
            }
        }
    }

    #[test]
    fn screen_phase_mask_scratch_reuses_storage_and_clears_word_tail_on_resize() {
        let thresholds = PairQcThresholds {
            min_phase_minor_freq: 0.0,
            max_unknown_rate: 1.0,
            min_cis: 0,
            min_trans: 0,
        };
        let mut scratch = PairScreenScratch::default();
        let mut previous_cis_ptr: *const u64 = std::ptr::null();
        let mut previous_trans_ptr: *const u64 = std::ptr::null();

        for (n_samples, pattern) in [(129, 0), (65, 1)] {
            let left = (0..n_samples)
                .map(|_| phased(true, PhaseSet::Implicit))
                .collect::<Vec<_>>();
            let right = (0..n_samples)
                .map(|sample| {
                    let is_cis = if pattern == 0 {
                        sample < 64
                    } else {
                        sample == 64
                    };
                    phased(is_cis, PhaseSet::Implicit)
                })
                .collect::<Vec<_>>();
            let left_packed = PackedGenotypes::from_calls(&left);
            let right_packed = PackedGenotypes::from_calls(&right);
            let expected = reference_screen_preparation(
                Some(&left),
                Some(&right),
                &left_packed,
                &right_packed,
                thresholds,
            )
            .unwrap();
            let observed = prepare_pair_for_screen_with_scratch_optional(
                Some(&left),
                Some(&right),
                &left_packed,
                &right_packed,
                thresholds,
                &mut scratch,
            )
            .unwrap();
            assert_eq!(observed, expected);

            let phase_masks = match observed {
                PairScreenPreparation::Ready { phase_masks, .. } => phase_masks,
                other => panic!("expected a screenable pair, got {other:?}"),
            };
            assert_eq!(phase_masks.n_samples(), n_samples);
            assert_eq!(phase_masks.n_cis(), if n_samples == 129 { 64 } else { 1 });
            assert_eq!(phase_masks.n_trans(), n_samples - phase_masks.n_cis());
            if !previous_cis_ptr.is_null() {
                assert_eq!(phase_masks.cis.as_ptr(), previous_cis_ptr);
                assert_eq!(phase_masks.trans.as_ptr(), previous_trans_ptr);
            }
            previous_cis_ptr = phase_masks.cis.as_ptr();
            previous_trans_ptr = phase_masks.trans.as_ptr();
            scratch.recycle_phase_masks(phase_masks);
        }
    }

    #[test]
    fn screen_skips_mode_table_without_unknowns_and_preserves_donor_rejection() {
        let thresholds = PairQcThresholds {
            min_phase_minor_freq: 0.0,
            max_unknown_rate: 1.0,
            min_cis: 0,
            min_trans: 0,
        };
        let mut scratch = PairScreenScratch::default();

        let complete_left = vec![
            known(0),
            phased(true, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
        ];
        let complete_right = vec![
            known(0),
            phased(true, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
        ];
        let complete_left_packed = PackedGenotypes::from_calls(&complete_left);
        let complete_right_packed = PackedGenotypes::from_calls(&complete_right);
        let expected_complete = reference_screen_preparation(
            Some(&complete_left),
            Some(&complete_right),
            &complete_left_packed,
            &complete_right_packed,
            thresholds,
        )
        .unwrap();
        let builds_before_complete = COMPATIBLE_MODE_TABLE_BUILDS.with(|builds| builds.get());
        let observed_complete = prepare_pair_for_screen_with_scratch_optional(
            Some(&complete_left),
            Some(&complete_right),
            &complete_left_packed,
            &complete_right_packed,
            thresholds,
            &mut scratch,
        )
        .unwrap();
        let builds_after_complete = COMPATIBLE_MODE_TABLE_BUILDS.with(|builds| builds.get());
        assert_eq!(observed_complete, expected_complete);
        assert_eq!(builds_after_complete - builds_before_complete, 0);
        if let PairScreenPreparation::Ready { phase_masks, .. } = observed_complete {
            scratch.recycle_phase_masks(phase_masks);
        } else {
            panic!("complete pair unexpectedly failed screen QC");
        }

        let missing_left = vec![
            known(0),
            phased(true, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
            call(None, None, PhaseSet::Unknown),
        ];
        let missing_right = vec![
            known(0),
            phased(true, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
            known(0),
        ];
        let missing_left_packed = PackedGenotypes::from_calls(&missing_left);
        let missing_right_packed = PackedGenotypes::from_calls(&missing_right);
        let expected_missing = reference_screen_preparation(
            Some(&missing_left),
            Some(&missing_right),
            &missing_left_packed,
            &missing_right_packed,
            thresholds,
        )
        .unwrap();
        let builds_before_missing = COMPATIBLE_MODE_TABLE_BUILDS.with(|builds| builds.get());
        let observed_missing = prepare_pair_for_screen_with_scratch_optional(
            Some(&missing_left),
            Some(&missing_right),
            &missing_left_packed,
            &missing_right_packed,
            thresholds,
            &mut scratch,
        )
        .unwrap();
        let builds_after_missing = COMPATIBLE_MODE_TABLE_BUILDS.with(|builds| builds.get());
        assert_eq!(observed_missing, expected_missing);
        assert_eq!(builds_after_missing - builds_before_missing, 1);
        if let PairScreenPreparation::Ready { phase_masks, .. } = observed_missing {
            scratch.recycle_phase_masks(phase_masks);
        } else {
            panic!("pair with a compatible missing donor was rejected");
        }

        let no_donor_left = vec![
            phased(true, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
            call(None, None, PhaseSet::Unknown),
        ];
        let no_donor_right = vec![
            phased(true, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
            known(0),
        ];
        let no_donor_left_packed = PackedGenotypes::from_calls(&no_donor_left);
        let no_donor_right_packed = PackedGenotypes::from_calls(&no_donor_right);
        let expected_no_donor = reference_screen_preparation(
            Some(&no_donor_left),
            Some(&no_donor_right),
            &no_donor_left_packed,
            &no_donor_right_packed,
            thresholds,
        )
        .unwrap();
        let builds_before_no_donor = COMPATIBLE_MODE_TABLE_BUILDS.with(|builds| builds.get());
        let observed_no_donor = prepare_pair_for_screen_with_scratch_optional(
            Some(&no_donor_left),
            Some(&no_donor_right),
            &no_donor_left_packed,
            &no_donor_right_packed,
            thresholds,
            &mut scratch,
        )
        .unwrap();
        let builds_after_no_donor = COMPATIBLE_MODE_TABLE_BUILDS.with(|builds| builds.get());
        assert_eq!(observed_no_donor, expected_no_donor);
        assert_eq!(builds_after_no_donor - builds_before_no_donor, 1);
        assert!(matches!(
            observed_no_donor,
            PairScreenPreparation::NoCompatibleDonor
        ));
    }

    #[test]
    fn full_screen_counts_reuse_phase_support_without_double_counting() {
        let left = vec![
            known(0),
            known(0),
            known(0),
            known(1),
            phased(true, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
            known(1),
            known(2),
            known(2),
            known(2),
            call(None, None, PhaseSet::Unknown),
            phased(true, PhaseSet::Unknown),
        ];
        let right = vec![
            known(0),
            known(1),
            known(2),
            known(0),
            phased(true, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
            known(2),
            known(0),
            known(1),
            known(2),
            known(0),
            phased(false, PhaseSet::Unknown),
        ];
        let left_packed = PackedGenotypes::from_calls(&left);
        let right_packed = PackedGenotypes::from_calls(&right);
        let phase_counts = count_pair_phase_support_optional(
            Some(&left),
            Some(&right),
            &left_packed,
            &right_packed,
        )
        .unwrap();
        let counts = count_pair_for_screen_optional(
            Some(&left),
            Some(&right),
            &left_packed,
            &right_packed,
            &phase_counts,
        )
        .unwrap();

        assert_eq!(
            counts,
            PairCounts {
                category_counts: [1; 10],
                genotype_missing: 1,
                phase_unknown: 1,
                n_total: 12,
            }
        );
    }

    #[test]
    fn early_phase_support_checks_invalid_dosage_in_the_support_pass() {
        let invalid = vec![known(3)];
        let observed = vec![known(0)];
        let invalid_packed = PackedGenotypes::from_calls(&invalid);
        let observed_packed = PackedGenotypes::from_calls(&observed);
        assert!(invalid_packed.has_invalid_dosage());
        assert!(!observed_packed.has_invalid_dosage());
        assert_eq!(
            count_pair_phase_support_optional(
                Some(&invalid),
                Some(&observed),
                &invalid_packed,
                &observed_packed,
            )
            .unwrap_err(),
            "invalid diploid dosage pair (3, 0)"
        );

        let missing = vec![call(None, None, PhaseSet::Unknown)];
        let missing_packed = PackedGenotypes::from_calls(&missing);
        assert!(!missing_packed.has_invalid_dosage());
        assert!(count_pair_phase_support_optional(
            Some(&invalid),
            Some(&missing),
            &invalid_packed,
            &missing_packed,
        )
        .is_ok());
    }

    #[test]
    fn packed_pair_masks_match_scalar_across_word_boundaries_and_explicit_phase_sets() {
        let thresholds = PairQcThresholds {
            min_phase_minor_freq: 0.0,
            max_unknown_rate: 1.0,
            min_cis: 0,
            min_trans: 0,
        };
        for n_samples in [1, 7, 63, 64, 65, 127, 129] {
            for replicate in 0..4 {
                let make_calls = |offset: usize| {
                    (0..n_samples)
                        .map(|sample| {
                            let state =
                                (sample * (7 + offset) + replicate * (5 + offset) + offset) % 9;
                            let dosage = if state == 0 {
                                None
                            } else {
                                Some(((state + sample + offset) % 3) as u8)
                            };
                            let (alt_on_hap0, phase_set) = if dosage == Some(1) {
                                let phase = (sample + replicate + offset) % 4;
                                let phase_set = match phase {
                                    0 => PhaseSet::Implicit,
                                    1 => PhaseSet::Unknown,
                                    2 => PhaseSet::Explicit(format!("ps-{}", sample % 3)),
                                    _ => PhaseSet::Explicit(format!("other-{}", sample % 2)),
                                };
                                (Some((sample + offset) % 2 == 0), phase_set)
                            } else {
                                (None, PhaseSet::Unknown)
                            };
                            call(dosage, alt_on_hap0, phase_set)
                        })
                        .collect::<Vec<_>>()
                };
                let left = make_calls(1);
                let right = make_calls(3);
                let expected = prepare_pair(&left, &right, thresholds).unwrap();
                let left_packed = PackedGenotypes::from_calls(&left);
                let right_packed = PackedGenotypes::from_calls(&right);
                let observed =
                    prepare_pair_encoded(&left, &right, &left_packed, &right_packed, thresholds)
                        .unwrap();
                match (expected, observed) {
                    (
                        PairPreparation::Ready {
                            counts: expected_counts,
                            filled: expected_filled,
                        },
                        EncodedPairPreparation::Ready { counts, pair },
                    ) => {
                        assert_eq!(counts, expected_counts);
                        assert_eq!(pair.materialize(), expected_filled);
                    }
                    (
                        PairPreparation::QcRejected(expected_counts),
                        EncodedPairPreparation::QcRejected(counts),
                    )
                    | (
                        PairPreparation::NoCompatibleDonor(expected_counts),
                        EncodedPairPreparation::NoCompatibleDonor(counts),
                    ) => assert_eq!(counts, expected_counts),
                    (expected, observed) => {
                        panic!("n={n_samples}, replicate={replicate}: {expected:?} vs {observed:?}")
                    }
                }
            }
        }
    }

    #[test]
    fn counts_all_ten_pair_categories_and_separates_other_from_unknown() {
        let left = vec![
            known(0),
            known(0),
            known(0),
            known(1),
            phased(true, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
            known(1),
            known(2),
            known(2),
            known(2),
            call(None, None, PhaseSet::Unknown),
            phased(false, PhaseSet::Unknown),
        ];
        let right = vec![
            known(0),
            phased(false, PhaseSet::Unknown),
            known(2),
            known(0),
            phased(true, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
            known(2),
            known(0),
            known(1),
            known(2),
            known(2),
            phased(true, PhaseSet::Implicit),
        ];

        let counts = count_pair(&left, &right).unwrap();
        assert_eq!(counts.category_counts, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(counts.other_observed(), 8);
        assert_eq!(counts.cis_observed(), 1);
        assert_eq!(counts.trans_observed(), 1);
        assert_eq!(counts.genotype_missing, 1);
        assert_eq!(counts.phase_unknown, 1);
        assert_eq!(counts.n_total, 12);
        assert_eq!(
            PairCategory::Cis.dosages(),
            (1, 1),
            "cis and trans are phase subclasses of the same unphased dosage pair"
        );
    }

    #[test]
    fn raw_qc_uses_cis_trans_support_and_unknowns_before_imputation() {
        let mut left = vec![
            phased(true, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
        ];
        let mut right = vec![
            phased(true, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
        ];
        for _ in 0..8 {
            left.push(call(None, None, PhaseSet::Unknown));
            right.push(known(1));
        }
        let counts = count_pair(&left, &right).unwrap();
        let thresholds = PairQcThresholds {
            min_phase_minor_freq: 0.0,
            max_unknown_rate: 0.05,
            min_cis: 1,
            min_trans: 1,
        };
        assert!(!passes_pair_qc(&counts, thresholds));
        assert!(matches!(
            prepare_pair(&left, &right, thresholds).unwrap(),
            PairPreparation::QcRejected(_)
        ));
    }

    #[test]
    fn all_other_pairs_have_no_phase_frequency_even_if_dosages_are_common() {
        let left = vec![known(0); 20];
        let right = vec![known(2); 20];
        let counts = count_pair(&left, &right).unwrap();
        assert_eq!(counts.other_observed(), 20);
        assert_eq!(counts.n_informative(), 0);
        let thresholds = PairQcThresholds {
            min_phase_minor_freq: 0.0,
            max_unknown_rate: 1.0,
            min_cis: 0,
            min_trans: 0,
        };
        assert!(!passes_pair_qc(&counts, thresholds));
    }

    #[test]
    fn compatible_mode_preserves_known_dosage_and_fills_phase_unknown_between_cis_trans() {
        let mut left = vec![phased(true, PhaseSet::Implicit); 3];
        let mut right = vec![phased(true, PhaseSet::Implicit); 3];
        left.push(phased(true, PhaseSet::Implicit));
        right.push(phased(false, PhaseSet::Implicit));
        for _ in 0..4 {
            left.push(known(2));
            right.push(known(0));
        }
        left.push(phased(false, PhaseSet::Unknown));
        right.push(phased(true, PhaseSet::Implicit));

        let counts = count_pair(&left, &right).unwrap();
        let filled = mode_fill_pair(&left, &right, &counts).unwrap();
        assert_eq!(filled.z[filled.z.len() - 1], 1);
        assert_eq!(filled.g_i[filled.g_i.len() - 1], 1);
        assert_eq!(filled.g_j[filled.g_j.len() - 1], 1);
        assert_eq!(filled.n_imputed, 1);
    }

    #[test]
    fn mode_ties_use_cis_before_trans_and_no_compatible_donor_skips_pair() {
        let left = vec![
            phased(true, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
            call(None, None, PhaseSet::Unknown),
        ];
        let right = vec![
            phased(true, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
            phased(true, PhaseSet::Implicit),
        ];
        let counts = count_pair(&left, &right).unwrap();
        let filled = mode_fill_pair(&left, &right, &counts).unwrap();
        assert_eq!(filled.z[2], 1);

        let left = vec![
            phased(true, PhaseSet::Implicit),
            call(None, None, PhaseSet::Unknown),
        ];
        let right = vec![phased(true, PhaseSet::Implicit), known(2)];
        let counts = count_pair(&left, &right).unwrap();
        assert!(mode_fill_pair(&left, &right, &counts).is_none());
    }

    #[test]
    fn pair_qc_threshold_boundaries_are_inclusive() {
        let left = vec![phased(true, PhaseSet::Implicit); 2]
            .into_iter()
            .chain(std::iter::repeat_n(known(2), 6))
            .chain(std::iter::repeat_n(call(None, None, PhaseSet::Unknown), 2))
            .collect::<Vec<_>>();
        let right = vec![
            phased(true, PhaseSet::Implicit),
            phased(false, PhaseSet::Implicit),
        ]
        .into_iter()
        .chain(std::iter::repeat_n(known(0), 6))
        .chain(std::iter::repeat_n(known(1), 2))
        .collect::<Vec<_>>();
        let counts = count_pair(&left, &right).unwrap();
        let thresholds = PairQcThresholds {
            min_phase_minor_freq: 0.5,
            max_unknown_rate: 0.2,
            min_cis: 1,
            min_trans: 1,
        };
        assert!(passes_pair_qc(&counts, thresholds));
    }
}
