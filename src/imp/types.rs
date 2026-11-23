use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Vcf,
    Pgen,
}

/// A diploid genotype represented by allele indexes rather than dosage.
/// `phased` is meaningful for heterozygotes; homozygotes are unaffected by
/// the separator used in the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiploidCall {
    pub alleles: [Option<u16>; 2],
    pub phased: bool,
}

impl DiploidCall {
    pub const fn missing(phased: bool) -> Self {
        Self {
            alleles: [None, None],
            phased,
        }
    }

    pub const fn is_fully_missing(self) -> bool {
        self.alleles[0].is_none() && self.alleles[1].is_none()
    }

    pub const fn is_partially_missing(self) -> bool {
        self.alleles[0].is_some() != self.alleles[1].is_some()
    }

    pub const fn is_complete(self) -> bool {
        self.alleles[0].is_some() && self.alleles[1].is_some()
    }

    pub const fn is_heterozygous(self) -> bool {
        match self.alleles {
            [Some(a), Some(b)] => a != b,
            _ => false,
        }
    }

    pub fn validate_alleles(self, allele_count: usize) -> Result<(), String> {
        if allele_count == 0 {
            return Err("a variant must have at least one allele".to_owned());
        }
        for allele in self.alleles.into_iter().flatten() {
            if usize::from(allele) >= allele_count {
                return Err(format!(
                    "genotype allele index {allele} is outside a variant with {allele_count} alleles"
                ));
            }
        }
        Ok(())
    }
}

/// Compact per-variant diploid calls. Common biallelic calls use four bits per
/// sample; partial-missing and multiallelic calls are kept in a sparse side
/// list so dense windows do not pay `size_of::<DiploidCall>()` per genotype.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DiploidCalls {
    len: usize,
    packed: Vec<u8>,
    complex: Vec<(usize, DiploidCall)>,
    #[serde(default)]
    phase_sets: Option<Vec<Option<PhaseSetId>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct PhaseSetId(u32);

impl PhaseSetId {
    pub(crate) const fn new(value: u32) -> Self {
        Self(value)
    }
}

impl DiploidCalls {
    /// Create a compact call row that can be filled in sample order.
    pub fn empty(len: usize) -> Self {
        Self {
            len,
            packed: vec![0u8; len.div_ceil(2)],
            complex: Vec::new(),
            phase_sets: None,
        }
    }

    pub fn from_calls(calls: Vec<DiploidCall>) -> Self {
        Self::from_iter(calls.into_iter())
    }

    /// Packs an exact-size call iterator directly, avoiding an intermediate
    /// `Vec<DiploidCall>` when constructing large marker rows.
    pub fn from_iter(calls: impl ExactSizeIterator<Item = DiploidCall>) -> Self {
        let len = calls.len();
        let mut packed = vec![0u8; len.div_ceil(2)];
        let mut complex = Vec::new();
        for (sample, call) in calls.enumerate() {
            let code = encode_call(call);
            if let Some(code) = code {
                set_nibble(&mut packed, sample, code);
            } else {
                set_nibble(&mut packed, sample, 6);
                complex.push((sample, call));
            }
        }
        Self {
            len,
            packed,
            complex,
            phase_sets: None,
        }
    }

    /// Convert a BED-compatible biallelic row directly into the internal
    /// compact call representation. `phases` contains orientation only for
    /// heterozygous calls with phase information.
    pub(crate) fn from_packed_biallelic_row(
        row: &[u8],
        phases: Option<&[Option<bool>]>,
        len: usize,
    ) -> Result<Self, String> {
        let required_row_bytes = len.div_ceil(4);
        if row.len() < required_row_bytes {
            return Err(format!(
                "packed genotype row has {} bytes; expected at least {required_row_bytes}",
                row.len()
            ));
        }
        if phases.is_some_and(|values| values.len() < len) {
            return Err(format!(
                "packed genotype phase row has {} samples; expected at least {len}",
                phases.map_or(0, |values| values.len())
            ));
        }

        let mut packed = vec![0u8; len.div_ceil(2)];
        for (pair_index, output_byte) in packed.iter_mut().enumerate() {
            let sample0 = pair_index * 2;
            let sample1 = sample0 + 1;
            let shift = (sample0 & 3) * 2;
            let mut genotype_pair = (row[sample0 >> 2] >> shift) & 0x0f;
            if sample1 >= len {
                genotype_pair &= 0x03;
            }
            let phase0 = packed_call_phase_code(phases, sample0);
            let phase1 = packed_call_phase_code(phases, sample1);
            let table_index = usize::from(genotype_pair)
                | (usize::from(phase0) << 4)
                | (usize::from(phase1) << 6);
            *output_byte = PACKED_BIALLELIC_CALL_PAIRS[table_index];
        }
        Ok(Self {
            len,
            packed,
            complex: Vec::new(),
            phase_sets: None,
        })
    }

    pub(crate) fn from_packed_biallelic_phase_bits(
        row: &[u8],
        phase_present: &[u64],
        phase_orientation: &[u64],
        len: usize,
    ) -> Result<Self, String> {
        let required_row_bytes = len.div_ceil(4);
        let phase_words = len.div_ceil(64);
        if row.len() < required_row_bytes {
            return Err(format!(
                "packed genotype row has {} bytes; expected at least {required_row_bytes}",
                row.len()
            ));
        }
        if phase_present.len() < phase_words || phase_orientation.len() < phase_words {
            return Err(format!(
                "packed phase tracks have {}/{} words; expected at least {phase_words}",
                phase_present.len(),
                phase_orientation.len()
            ));
        }

        let mut packed = vec![0u8; len.div_ceil(2)];
        for (pair_index, output_byte) in packed.iter_mut().enumerate() {
            let sample0 = pair_index * 2;
            let sample1 = sample0 + 1;
            let shift = (sample0 & 3) * 2;
            let mut genotype_pair = (row[sample0 >> 2] >> shift) & 0x0f;
            if sample1 >= len {
                genotype_pair &= 0x03;
            }
            let phase0 = packed_phase_bits_code(phase_present, phase_orientation, sample0);
            let phase1 = if sample1 < len {
                packed_phase_bits_code(phase_present, phase_orientation, sample1)
            } else {
                0
            };
            let table_index = usize::from(genotype_pair)
                | (usize::from(phase0) << 4)
                | (usize::from(phase1) << 6);
            *output_byte = PACKED_BIALLELIC_CALL_PAIRS[table_index];
        }
        Ok(Self {
            len,
            packed,
            complex: Vec::new(),
            phase_sets: None,
        })
    }

    /// Update a sample call without rebuilding or widening the packed row.
    pub fn set_call(&mut self, sample: usize, call: DiploidCall) -> Result<(), String> {
        if sample >= self.len {
            return Err(format!(
                "sample index {sample} is outside a genotype row with {} samples",
                self.len
            ));
        }
        self.set_call_unchecked(sample, call);
        Ok(())
    }

    /// Set a call at an index already bounded by the caller. Streaming readers
    /// use this to pack each decoded sample directly, without a temporary
    /// `Vec<DiploidCall>` per marker.
    #[inline(always)]
    pub(crate) fn set_call_unchecked(&mut self, sample: usize, call: DiploidCall) {
        debug_assert!(sample < self.len);
        let previous_code = get_nibble(&self.packed, sample);
        let code = encode_call(call);
        if previous_code == 6 {
            if let Ok(index) = self
                .complex
                .binary_search_by_key(&sample, |(index, _)| *index)
            {
                self.complex.remove(index);
            }
        }
        if let Some(code) = code {
            set_nibble(&mut self.packed, sample, code);
        } else {
            set_nibble(&mut self.packed, sample, 6);
            match self
                .complex
                .binary_search_by_key(&sample, |(index, _)| *index)
            {
                Ok(index) => self.complex[index].1 = call,
                Err(index) => self.complex.insert(index, (sample, call)),
            }
        }
    }

    /// Set a commonly encoded biallelic call directly from its four-bit code.
    /// Callers must use codes 0..=5 on a freshly allocated row or replace the
    /// complex call at the same index separately.
    #[inline(always)]
    pub(crate) fn set_common_call_code_unchecked(&mut self, sample: usize, code: u8) {
        debug_assert!(sample < self.len);
        debug_assert!(code <= 5);
        set_nibble(&mut self.packed, sample, code);
    }

    /// Set two adjacent common calls with one packed-byte store.
    #[inline(always)]
    pub(crate) fn set_common_call_pair_unchecked(&mut self, sample0: usize, code0: u8, code1: u8) {
        debug_assert_eq!(sample0 & 1, 0);
        debug_assert!(sample0 + 1 < self.len);
        debug_assert!(code0 <= 5 && code1 <= 5);
        self.packed[sample0 >> 1] = code0 | (code1 << 4);
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn first_invalid_complete_phased_biallelic(&self) -> Option<usize> {
        for (byte_index, &byte) in self.packed.iter().enumerate() {
            if COMPLETE_PHASED_BIALLELIC_PAIR[byte as usize] {
                continue;
            }
            let first_sample = byte_index * 2;
            if !is_complete_phased_biallelic_code(byte & 0x0f) {
                return Some(first_sample);
            }
            if first_sample + 1 < self.len && !is_complete_phased_biallelic_code(byte >> 4) {
                return Some(first_sample + 1);
            }
        }
        None
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn set_phase_set(
        &mut self,
        sample: usize,
        phase_set: Option<PhaseSetId>,
    ) -> Result<(), String> {
        if sample >= self.len {
            return Err(format!(
                "sample index {sample} is outside a genotype row with {} samples",
                self.len
            ));
        }
        let values = self.phase_sets.get_or_insert_with(|| vec![None; self.len]);
        values[sample] = phase_set;
        Ok(())
    }

    pub(crate) fn phase_set(&self, sample: usize) -> Option<PhaseSetId> {
        self.phase_sets
            .as_ref()
            .and_then(|values| values.get(sample))
            .copied()
            .flatten()
    }

    pub(crate) fn has_phase_set_track(&self) -> bool {
        self.phase_sets.is_some()
    }

    pub fn get(&self, sample: usize) -> Option<DiploidCall> {
        if sample >= self.len {
            return None;
        }
        let code = get_nibble(&self.packed, sample);
        Some(match code {
            0 => DiploidCall {
                alleles: [Some(0), Some(0)],
                phased: false,
            },
            1 => DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: false,
            },
            2 => DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: true,
            },
            3 => DiploidCall {
                alleles: [Some(1), Some(0)],
                phased: true,
            },
            4 => DiploidCall {
                alleles: [Some(1), Some(1)],
                phased: false,
            },
            5 => DiploidCall::missing(false),
            6 => self
                .complex
                .binary_search_by_key(&sample, |(index, _)| *index)
                .ok()
                .map(|index| self.complex[index].1)
                .unwrap_or_else(|| DiploidCall::missing(false)),
            _ => DiploidCall::missing(false),
        })
    }

    #[inline(always)]
    pub(crate) fn common_biallelic_code(&self, sample: usize) -> Option<u8> {
        if sample >= self.len {
            return None;
        }
        let code = get_nibble(&self.packed, sample);
        (code <= 5).then_some(code)
    }

    /// Read one phased haplotype allele without materializing a full call.
    /// Missing and out-of-range sample/haplotype indexes both return `None`.
    #[inline(always)]
    pub fn get_allele(&self, sample: usize, haplotype: usize) -> Option<u16> {
        if sample >= self.len || haplotype >= 2 {
            return None;
        }
        match get_nibble(&self.packed, sample) {
            0 => Some(0),
            1 | 2 => Some(haplotype as u16),
            3 => Some((1 - haplotype) as u16),
            4 => Some(1),
            5 => None,
            6 => self
                .complex
                .binary_search_by_key(&sample, |(index, _)| *index)
                .ok()
                .and_then(|index| self.complex[index].1.alleles[haplotype]),
            _ => None,
        }
    }

    /// Read a haplotype allele using a donor index that is already known to
    /// be in `0..2 * len`. Hot PBWT projection loops use this after validating
    /// the donor list once, avoiding two bounds checks for every state.
    #[inline(always)]
    pub(crate) fn get_allele_unchecked(&self, haplotype_index: usize) -> Option<u16> {
        debug_assert!(haplotype_index < self.len * 2);
        let sample = haplotype_index >> 1;
        let haplotype = haplotype_index & 1;
        match get_nibble(&self.packed, sample) {
            0 => Some(0),
            1 | 2 => Some(haplotype as u16),
            3 => Some((1 - haplotype) as u16),
            4 => Some(1),
            5 => None,
            6 => self
                .complex
                .binary_search_by_key(&sample, |(index, _)| *index)
                .ok()
                .and_then(|index| self.complex[index].1.alleles[haplotype]),
            _ => None,
        }
    }

    /// Decode up to 64 complete biallelic haplotypes into one allele bitplane
    /// word. The input already stores two 4-bit genotype codes per byte, so a
    /// byte lookup expands four haplotype alleles at once. `None` indicates a
    /// missing, partial, or non-biallelic call and lets callers use their
    /// detailed validation path.
    #[inline]
    pub(crate) fn biallelic_haplotype_word(
        &self,
        word_index: usize,
        swap_alleles: bool,
    ) -> Option<u64> {
        let haplotype_start = word_index.checked_mul(64)?;
        let haplotype_count = self.len.checked_mul(2)?;
        if haplotype_start >= haplotype_count {
            return None;
        }
        let byte_start = word_index.checked_mul(16)?;
        let byte_end = byte_start.saturating_add(16).min(self.packed.len());
        let mut word = 0u64;
        for (offset, &packed) in self.packed[byte_start..byte_end].iter().enumerate() {
            let decoded = BYTE_TO_BIALLELIC_HAP_MASK[packed as usize];
            if decoded == u8::MAX {
                return None;
            }
            word |= u64::from(decoded) << (offset * 4);
        }
        let valid_haplotypes = (haplotype_count - haplotype_start).min(64);
        let valid_mask = if valid_haplotypes == 64 {
            u64::MAX
        } else {
            (1u64 << valid_haplotypes) - 1
        };
        Some(if swap_alleles {
            !word & valid_mask
        } else {
            word & valid_mask
        })
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = DiploidCall> + '_ {
        (0..self.len).map(|sample| self.get(sample).unwrap())
    }

    pub fn bytes(&self) -> usize {
        self.packed.capacity()
            + self.complex.capacity() * std::mem::size_of::<(usize, DiploidCall)>()
            + self.phase_sets.as_ref().map_or(0, |values| {
                values.capacity() * std::mem::size_of::<Option<PhaseSetId>>()
            })
    }
}

impl From<Vec<DiploidCall>> for DiploidCalls {
    fn from(value: Vec<DiploidCall>) -> Self {
        Self::from_calls(value)
    }
}

fn set_nibble(bytes: &mut [u8], index: usize, value: u8) {
    let byte = &mut bytes[index >> 1];
    if index & 1 == 0 {
        *byte = (*byte & 0xf0) | (value & 0x0f);
    } else {
        *byte = (*byte & 0x0f) | ((value & 0x0f) << 4);
    }
}

fn encode_call(call: DiploidCall) -> Option<u8> {
    match call.alleles {
        [Some(0), Some(0)] => Some(0),
        [Some(0), Some(1)] if call.phased => Some(2),
        [Some(1), Some(0)] if call.phased => Some(3),
        [Some(0), Some(1)] | [Some(1), Some(0)] => Some(1),
        [Some(1), Some(1)] => Some(4),
        [None, None] => Some(5),
        _ => None,
    }
}

#[inline]
fn packed_call_phase_code(phases: Option<&[Option<bool>]>, sample: usize) -> u8 {
    match phases
        .and_then(|values| values.get(sample))
        .copied()
        .flatten()
    {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    }
}

#[inline]
fn packed_phase_bits_code(present: &[u64], orientation: &[u64], sample: usize) -> u8 {
    let word = sample >> 6;
    let bit = 1u64 << (sample & 63);
    if present[word] & bit == 0 {
        0
    } else if orientation[word] & bit == 0 {
        1
    } else {
        2
    }
}

const fn encode_packed_biallelic_call(code: u8, phase: u8) -> u8 {
    match code {
        0 => 0,
        1 => 5,
        2 => match phase {
            1 => 2,
            2 => 3,
            _ => 1,
        },
        3 => 4,
        _ => 5,
    }
}

const fn build_packed_biallelic_call_pairs() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut index = 0usize;
    while index < table.len() {
        let key = index as u8;
        let genotype0 = key & 0x03;
        let genotype1 = (key >> 2) & 0x03;
        let phase0 = (key >> 4) & 0x03;
        let phase1 = (key >> 6) & 0x03;
        let call0 = encode_packed_biallelic_call(genotype0, phase0);
        let call1 = encode_packed_biallelic_call(genotype1, phase1);
        table[index] = call0 | (call1 << 4);
        index += 1;
    }
    table
}

const PACKED_BIALLELIC_CALL_PAIRS: [u8; 256] = build_packed_biallelic_call_pairs();

const fn is_complete_phased_biallelic_code(code: u8) -> bool {
    matches!(code, 0 | 2 | 3 | 4)
}

const fn build_complete_phased_biallelic_pair_table() -> [bool; 256] {
    let mut table = [false; 256];
    let mut index = 0usize;
    while index < table.len() {
        table[index] = is_complete_phased_biallelic_code((index & 0x0f) as u8)
            && is_complete_phased_biallelic_code((index >> 4) as u8);
        index += 1;
    }
    table
}

const COMPLETE_PHASED_BIALLELIC_PAIR: [bool; 256] = build_complete_phased_biallelic_pair_table();

const fn biallelic_haplotype_mask(code: u8) -> Option<u8> {
    match code {
        0 => Some(0b00),     // 0|0
        1 | 2 => Some(0b10), // unphased or phased 0|1
        3 => Some(0b01),     // 1|0
        4 => Some(0b11),     // 1|1
        _ => None,
    }
}

const fn build_biallelic_byte_lookup() -> [u8; 256] {
    let mut lookup = [u8::MAX; 256];
    let mut packed = 0usize;
    while packed < lookup.len() {
        let low = biallelic_haplotype_mask((packed & 0x0f) as u8);
        let high = biallelic_haplotype_mask((packed >> 4) as u8);
        if let (Some(low), Some(high)) = (low, high) {
            lookup[packed] = low | (high << 2);
        }
        packed += 1;
    }
    lookup
}

const BYTE_TO_BIALLELIC_HAP_MASK: [u8; 256] = build_biallelic_byte_lookup();

fn get_nibble(bytes: &[u8], index: usize) -> u8 {
    let byte = bytes[index >> 1];
    if index & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariantMeta {
    pub chrom: String,
    pub pos: u64,
    pub id: String,
    pub ref_allele: String,
    pub alt_alleles: Vec<String>,
}

impl VariantMeta {
    pub fn allele_count(&self) -> usize {
        1 + self.alt_alleles.len()
    }

    pub fn key(&self) -> VariantKey {
        VariantKey {
            chrom: self.chrom.clone(),
            pos: self.pos,
            ref_allele: self.ref_allele.clone(),
            alt_alleles: self.alt_alleles.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariantRow {
    pub meta: VariantMeta,
    /// One call per sample, in the exact sample order from the input header.
    pub calls: DiploidCalls,
}

impl VariantRow {
    pub fn validate(&self, expected_samples: usize) -> Result<(), String> {
        self.validate_shape(expected_samples)?;
        for (sample_index, call) in self.calls.iter().enumerate() {
            call.validate_alleles(self.meta.allele_count())
                .map_err(|err| {
                    format!(
                        "invalid genotype at {}:{} for sample index {}: {err}",
                        self.meta.chrom,
                        self.meta.pos,
                        sample_index + 1
                    )
                })?;
        }
        Ok(())
    }

    pub(crate) fn validate_shape(&self, expected_samples: usize) -> Result<(), String> {
        if self.meta.chrom.is_empty() {
            return Err("variant chromosome must not be empty".to_owned());
        }
        if self.meta.pos == 0 {
            return Err(format!(
                "variant {}:{} has an invalid zero-based position; VCF positions are one-based",
                self.meta.chrom, self.meta.id
            ));
        }
        if self.meta.ref_allele.is_empty()
            || self.meta.alt_alleles.is_empty()
            || self.meta.alt_alleles.iter().any(String::is_empty)
        {
            return Err(format!(
                "variant {}:{} has an empty REF or ALT allele",
                self.meta.chrom, self.meta.pos
            ));
        }
        if self.calls.len() != expected_samples {
            return Err(format!(
                "variant {}:{} has {} genotype calls; expected {expected_samples}",
                self.meta.chrom,
                self.meta.pos,
                self.calls.len()
            ));
        }
        Ok(())
    }
}

/// Validate the reference-only constraints while walking genotype calls once.
/// The caller must have checked the row shape and confirmed one ALT allele.
pub(crate) fn validate_biallelic_reference_calls(row: &VariantRow) -> Result<(), String> {
    validate_reference_call_set(&row.meta, &row.calls)
}

/// Validate the same reference-only constraints after a row has entered a
/// phased window, before it can be used to construct any HMM panel.
pub(crate) fn validate_phased_reference_calls(row: &PhasedRow) -> Result<(), String> {
    validate_reference_call_set(&row.meta, &row.calls)
}

fn validate_reference_call_set(meta: &VariantMeta, calls: &DiploidCalls) -> Result<(), String> {
    if let Some(sample) = calls.first_invalid_complete_phased_biallelic() {
        return Err(format!(
            "reference GT must be phased and nonmissing at {}:{} sample {}",
            meta.chrom,
            meta.pos,
            sample + 1
        ));
    }
    Ok(())
}

/// A complete active cM window. `incoming_overlap_len` is the prefix of the
/// incoming marker overlap whose phase is fixed from the preceding window.
/// `emit_start..emit_end` identifies the rows owned by this window, split at
/// the midpoint of each adjacent window overlap.
#[derive(Debug, Clone)]
pub struct PhaseWindow {
    pub chrom: String,
    pub index: u64,
    pub rows: Vec<VariantRow>,
    pub cm_positions: Vec<f64>,
    pub incoming_overlap_len: usize,
    pub emit_start: usize,
    pub emit_end: usize,
}

impl PhaseWindow {
    pub fn validate(&self, expected_samples: usize) -> Result<(), String> {
        if self.rows.is_empty() {
            return Err("phase window must contain at least one variant".to_owned());
        }
        if self.rows.len() != self.cm_positions.len() {
            return Err(format!(
                "phase window has {} records but {} genetic positions",
                self.rows.len(),
                self.cm_positions.len()
            ));
        }
        if self.emit_start > self.emit_end || self.emit_end > self.rows.len() {
            return Err(format!(
                "invalid emit range {}..{} for a {}-variant window",
                self.emit_start,
                self.emit_end,
                self.rows.len()
            ));
        }
        if self.incoming_overlap_len > self.rows.len() {
            return Err(format!(
                "incoming overlap has {} rows for a {}-variant window",
                self.incoming_overlap_len,
                self.rows.len()
            ));
        }
        let mut prev_cm = f64::NEG_INFINITY;
        for (i, (row, cm)) in self.rows.iter().zip(&self.cm_positions).enumerate() {
            row.validate(expected_samples)?;
            if row.meta.chrom != self.chrom {
                return Err(format!(
                    "phase window {} contains chromosome {} at row {}",
                    self.chrom,
                    row.meta.chrom,
                    i + 1
                ));
            }
            if !cm.is_finite() || *cm < prev_cm {
                return Err(format!(
                    "genetic positions must be finite and nondecreasing (row {})",
                    i + 1
                ));
            }
            prev_cm = *cm;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PhasedRow {
    pub meta: VariantMeta,
    pub calls: DiploidCalls,
    pub dosage: Option<Vec<f32>>,
    pub ap1: Option<Vec<f32>>,
    pub ap2: Option<Vec<f32>>,
    pub gp: Option<Vec<[f32; 3]>>,
    pub info_af: Option<f32>,
    pub info_dr2: Option<f32>,
    pub imputed: bool,
}

#[derive(Debug, Clone)]
pub struct PhasedWindow {
    pub chrom: String,
    pub index: u64,
    pub rows: Vec<PhasedRow>,
    pub emit_start: usize,
    pub emit_end: usize,
}

/// Marker-major bit-packed haplotypes. Each marker has a compact allele-index
/// plane and a separate missingness plane. Biallelic data uses one bit per
/// haplotype; multiallelic VCF records can use up to 16 bits per allele.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackedLayout {
    MarkerMajor,
    StateMajor,
}

#[derive(Debug, Clone)]
pub struct PackedHaplotypes {
    pub n_haplotypes: usize,
    n_haplotypes_capacity: usize,
    pub n_markers: usize,
    pub bits_per_allele: u8,
    layout: PackedLayout,
    pub allele_bits: Vec<u64>,
    pub missing_bits: Vec<u64>,
    complete: bool,
}

impl PackedHaplotypes {
    pub fn empty(
        n_haplotypes: usize,
        n_markers: usize,
        bits_per_allele: u8,
    ) -> Result<Self, String> {
        if n_haplotypes == 0 || n_markers == 0 {
            return Err("packed haplotypes require positive dimensions".to_owned());
        }
        if !matches!(bits_per_allele, 1 | 2 | 4 | 8 | 16) {
            return Err("packed allele width must be 1, 2, 4, 8, or 16 bits".to_owned());
        }
        let values = n_haplotypes
            .checked_mul(n_markers)
            .ok_or_else(|| "packed haplotype dimensions overflow".to_owned())?;
        let allele_bit_len = values
            .checked_mul(bits_per_allele as usize)
            .ok_or_else(|| "packed allele bit length overflow".to_owned())?;
        let missing_bit_len = values;
        let mut missing_bits = vec![u64::MAX; missing_bit_len.div_ceil(64)];
        if let Some(last) = missing_bits.last_mut() {
            let used = missing_bit_len & 63;
            if used != 0 {
                *last &= (1u64 << used) - 1;
            }
        }
        Ok(Self {
            n_haplotypes,
            n_haplotypes_capacity: n_haplotypes,
            n_markers,
            bits_per_allele,
            layout: PackedLayout::MarkerMajor,
            allele_bits: vec![0; allele_bit_len.div_ceil(64)],
            missing_bits,
            complete: false,
        })
    }

    pub(crate) fn empty_state_major(
        n_haplotypes: usize,
        n_markers: usize,
        bits_per_allele: u8,
    ) -> Result<Self, String> {
        let mut packed = Self::empty(n_haplotypes, n_markers, bits_per_allele)?;
        packed.layout = PackedLayout::StateMajor;
        Ok(packed)
    }

    /// Allocate a complete state-major biallelic bitplane without a redundant
    /// zero-filled `missing_bits` vector, halving memory for large target and
    /// reference snapshots.
    pub(crate) fn empty_complete_biallelic_state_major(
        n_haplotypes: usize,
        n_markers: usize,
    ) -> Result<Self, String> {
        if n_haplotypes == 0 || n_markers == 0 {
            return Err("packed haplotypes require positive dimensions".to_owned());
        }
        let values = n_haplotypes
            .checked_mul(n_markers)
            .ok_or_else(|| "packed haplotype dimensions overflow".to_owned())?;
        Ok(Self {
            n_haplotypes,
            n_haplotypes_capacity: n_haplotypes,
            n_markers,
            bits_per_allele: 1,
            layout: PackedLayout::StateMajor,
            allele_bits: vec![0; values.div_ceil(64)],
            missing_bits: Vec::new(),
            complete: true,
        })
    }

    pub(crate) fn complete_biallelic_state_major(
        window: &PhasedWindow,
    ) -> Result<Option<Self>, String> {
        let Some(first_row) = window.rows.first() else {
            return Ok(None);
        };
        let n_samples = first_row.calls.len();
        if n_samples == 0
            || window
                .rows
                .iter()
                .any(|row| row.calls.len() != n_samples || row.meta.allele_count() != 2)
        {
            return Ok(None);
        }
        let n_markers = window.rows.len();
        let n_haplotypes = n_samples
            .checked_mul(2)
            .ok_or_else(|| "phase snapshot haplotype count overflow".to_owned())?;
        let mut packed = Self::empty_complete_biallelic_state_major(n_haplotypes, n_markers)?;
        for (marker, row) in window.rows.iter().enumerate() {
            for (sample, call) in row.calls.iter().enumerate() {
                let [Some(first), Some(second)] = call.alleles else {
                    return Ok(None);
                };
                if first > 1 || second > 1 {
                    return Ok(None);
                }
                let first_bit = (sample * 2) * n_markers + marker;
                let second_bit = first_bit + n_markers;
                if first == 1 {
                    packed.allele_bits[first_bit >> 6] |= 1u64 << (first_bit & 63);
                }
                if second == 1 {
                    packed.allele_bits[second_bit >> 6] |= 1u64 << (second_bit & 63);
                }
            }
        }
        Ok(Some(packed))
    }

    /// Refresh a complete biallelic state-major matrix from sample-major hard
    /// calls. Each haplotype track is packed in marker chunks so destination
    /// writes stay contiguous even when the marker count is not word aligned.
    pub(crate) fn overwrite_complete_biallelic_state_major_from_calls<F>(
        &mut self,
        n_samples: usize,
        n_markers: usize,
        mut call_at: F,
    ) -> Result<bool, String>
    where
        F: FnMut(usize, usize) -> DiploidCall,
    {
        if n_samples == 0 || n_markers == 0 {
            return Err("packed haplotypes require positive dimensions".to_owned());
        }
        let n_haplotypes = n_samples
            .checked_mul(2)
            .ok_or_else(|| "phase snapshot haplotype count overflow".to_owned())?;
        if self.n_haplotypes != n_haplotypes || self.n_markers != n_markers {
            self.reset_state_major(n_haplotypes, n_markers, 1)?;
        } else if !self.is_state_major() || self.bits_per_allele != 1 {
            return Err("packed phase snapshot is not state-major biallelic".to_owned());
        }
        self.complete = false;
        self.missing_bits.fill(0);
        for sample in 0..n_samples {
            let first_state_start = sample * 2 * n_markers;
            let second_state_start = first_state_start + n_markers;
            let mut marker_start = 0usize;
            while marker_start < n_markers {
                let count = (n_markers - marker_start).min(64);
                let mut first_bits = 0u64;
                let mut second_bits = 0u64;
                for offset in 0..count {
                    let call = call_at(marker_start + offset, sample);
                    let [Some(first), Some(second)] = call.alleles else {
                        return Ok(false);
                    };
                    if first > 1 || second > 1 {
                        return Ok(false);
                    }
                    if first == 1 {
                        first_bits |= 1u64 << offset;
                    }
                    if second == 1 {
                        second_bits |= 1u64 << offset;
                    }
                }
                write_bit_chunk(
                    &mut self.allele_bits,
                    first_state_start + marker_start,
                    count,
                    first_bits,
                );
                write_bit_chunk(
                    &mut self.allele_bits,
                    second_state_start + marker_start,
                    count,
                    second_bits,
                );
                marker_start += count;
            }
        }
        let bit_len = n_haplotypes * n_markers;
        if let Some(last) = self.allele_bits.last_mut() {
            let used = bit_len & 63;
            if used != 0 {
                *last &= (1u64 << used) - 1;
            }
        }
        self.complete = true;
        Ok(true)
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.complete
    }

    pub(crate) fn mark_complete(&mut self) {
        self.complete = true;
    }

    pub(crate) fn is_state_major(&self) -> bool {
        self.layout == PackedLayout::StateMajor
    }

    #[inline]
    pub(crate) fn state_major_capacity_haplotypes(&self) -> Option<usize> {
        (self.layout == PackedLayout::StateMajor).then_some(self.n_haplotypes_capacity)
    }

    pub(crate) fn reset_state_major(
        &mut self,
        n_haplotypes: usize,
        n_markers: usize,
        bits_per_allele: u8,
    ) -> Result<(), String> {
        if n_haplotypes == 0 || n_markers == 0 {
            return Err("packed haplotypes require positive dimensions".to_owned());
        }
        if !matches!(bits_per_allele, 1 | 2 | 4 | 8 | 16) {
            return Err("packed allele width must be 1, 2, 4, 8, or 16 bits".to_owned());
        }
        let values = n_haplotypes
            .checked_mul(n_markers)
            .ok_or_else(|| "packed haplotype dimensions overflow".to_owned())?;
        let allele_bit_len = values
            .checked_mul(bits_per_allele as usize)
            .ok_or_else(|| "packed allele bit length overflow".to_owned())?;
        let allele_words = allele_bit_len.div_ceil(64);
        let missing_words = values.div_ceil(64);
        self.n_haplotypes = n_haplotypes;
        self.n_haplotypes_capacity = n_haplotypes;
        self.n_markers = n_markers;
        self.bits_per_allele = bits_per_allele;
        self.layout = PackedLayout::StateMajor;
        self.complete = false;
        self.allele_bits.clear();
        self.allele_bits.resize(allele_words, 0);
        self.missing_bits.clear();
        self.missing_bits.resize(missing_words, u64::MAX);
        if let Some(last) = self.missing_bits.last_mut() {
            let used = values & 63;
            if used != 0 {
                *last &= (1u64 << used) - 1;
            }
        }
        Ok(())
    }

    /// Resize a state-major biallelic panel for a caller that will overwrite
    /// every state and marker bit. New storage starts with cleared missingness
    /// because complete donor mosaics overwrite every visited allele and never
    /// need the generic all-missing initialization.
    pub(crate) fn resize_state_major_biallelic_for_overwrite(
        &mut self,
        n_haplotypes: usize,
        n_markers: usize,
    ) -> Result<(), String> {
        if n_haplotypes == 0 || n_markers == 0 {
            return Err("packed haplotypes require positive dimensions".to_owned());
        }
        let storage_reusable = self.layout == PackedLayout::StateMajor
            && self.bits_per_allele == 1
            && self.n_markers == n_markers
            && self.n_haplotypes_capacity >= n_haplotypes;
        self.n_haplotypes = n_haplotypes;
        if !storage_reusable {
            let values = n_haplotypes
                .checked_mul(n_markers)
                .ok_or_else(|| "packed haplotype dimensions overflow".to_owned())?;
            let allele_words = values.div_ceil(64);
            let missing_words = values.div_ceil(64);
            self.n_haplotypes_capacity = n_haplotypes;
            self.n_markers = n_markers;
            self.bits_per_allele = 1;
            self.layout = PackedLayout::StateMajor;
            self.complete = false;
            self.allele_bits.clear();
            self.allele_bits.resize(allele_words, 0);
            self.missing_bits.clear();
            self.missing_bits.resize(missing_words, 0);
        }
        Ok(())
    }

    pub(crate) fn truncate_state_major_biallelic_for_overwrite(
        &mut self,
        n_haplotypes: usize,
    ) -> Result<(), String> {
        if n_haplotypes == 0
            || n_haplotypes > self.n_haplotypes_capacity
            || self.layout != PackedLayout::StateMajor
            || self.bits_per_allele != 1
        {
            return Err("packed state-major truncation dimensions are invalid".to_owned());
        }
        self.n_haplotypes = n_haplotypes;
        Ok(())
    }

    #[inline]
    fn flat_index(&self, marker: usize, haplotype: usize) -> Option<usize> {
        if marker >= self.n_markers || haplotype >= self.n_haplotypes {
            return None;
        }
        match self.layout {
            PackedLayout::MarkerMajor => marker
                .checked_mul(self.n_haplotypes)?
                .checked_add(haplotype),
            PackedLayout::StateMajor => haplotype.checked_mul(self.n_markers)?.checked_add(marker),
        }
    }

    pub fn allele(&self, marker: usize, haplotype: usize) -> Option<u16> {
        let index = self.flat_index(marker, haplotype)?;
        if self.get_missing(index) {
            return None;
        }
        let bit_offset = index * self.bits_per_allele as usize;
        let width = self.bits_per_allele as usize;
        let word = bit_offset >> 6;
        let shift = bit_offset & 63;
        let mut value = *self.allele_bits.get(word)? >> shift;
        if shift + width > 64 {
            value |= *self.allele_bits.get(word + 1)? << (64 - shift);
        }
        Some((value & ((1u64 << width) - 1)) as u16)
    }

    pub fn set_allele(
        &mut self,
        marker: usize,
        haplotype: usize,
        allele: Option<u16>,
    ) -> Result<(), String> {
        if marker >= self.n_markers || haplotype >= self.n_haplotypes {
            return Err("packed haplotype index is out of bounds".to_owned());
        }
        let index = self
            .flat_index(marker, haplotype)
            .ok_or_else(|| "packed haplotype index is out of bounds".to_owned())?;
        let bit_offset = index * self.bits_per_allele as usize;
        let width = self.bits_per_allele as usize;
        let word = bit_offset >> 6;
        let shift = bit_offset & 63;
        let value_mask = (1u64 << width) - 1;
        let value = allele.unwrap_or(0) as u64;
        if value > value_mask {
            return Err(format!(
                "allele index {} does not fit in {} bits",
                value, self.bits_per_allele
            ));
        }
        self.allele_bits[word] &= !(value_mask << shift);
        self.allele_bits[word] |= value << shift;
        if shift + width > 64 {
            let high_width = shift + width - 64;
            let high_mask = (1u64 << high_width) - 1;
            self.allele_bits[word + 1] &= !high_mask;
            self.allele_bits[word + 1] |= value >> (64 - shift);
        }
        if !self.missing_bits.is_empty() {
            let missing_word = index >> 6;
            let missing_mask = 1u64 << (index & 63);
            if allele.is_some() {
                self.missing_bits[missing_word] &= !missing_mask;
            } else {
                self.missing_bits[missing_word] |= missing_mask;
            }
        } else if allele.is_none() {
            let values = self.n_haplotypes * self.n_markers;
            self.missing_bits.resize(values.div_ceil(64), 0);
            self.missing_bits[index >> 6] |= 1u64 << (index & 63);
            self.complete = false;
        }
        Ok(())
    }

    /// Set a biallelic state-major entry after its dimensions and allele value
    /// have been validated by the caller.
    #[inline(always)]
    pub(crate) unsafe fn set_state_major_biallelic_unchecked(
        &mut self,
        marker: usize,
        state: usize,
        allele: Option<u16>,
    ) {
        debug_assert!(marker < self.n_markers);
        debug_assert!(state < self.n_haplotypes);
        debug_assert_eq!(self.layout, PackedLayout::StateMajor);
        debug_assert_eq!(self.bits_per_allele, 1);
        debug_assert!(allele.is_none_or(|value| value <= 1));
        let index = state * self.n_markers + marker;
        let word = index >> 6;
        let shift = index & 63;
        let mask = 1u64 << shift;
        if allele == Some(1) {
            *self.allele_bits.get_unchecked_mut(word) |= mask;
        } else {
            *self.allele_bits.get_unchecked_mut(word) &= !mask;
        }
        if !self.missing_bits.is_empty() {
            let missing_word = index >> 6;
            let missing_mask = 1u64 << (index & 63);
            if allele.is_some() {
                *self.missing_bits.get_unchecked_mut(missing_word) &= !missing_mask;
            } else {
                *self.missing_bits.get_unchecked_mut(missing_word) |= missing_mask;
            }
        } else {
            debug_assert!(allele.is_some());
        }
    }

    /// Populate one marker column of a freshly allocated (zero-initialized
    /// allele plane) state-major biallelic `PackedHaplotypes` directly from
    /// 64-bit marker-major haplotype words.
    ///
    /// Iterating only set bits via `trailing_zeros` skips all reference-allele
    /// haplotypes and avoids per-haplotype bounds checks.
    pub(crate) fn write_fresh_state_major_biallelic_marker_from_words(
        &mut self,
        marker: usize,
        allele_words: &[u64],
        swap: bool,
    ) -> Result<(), String> {
        let required_words = self.n_haplotypes.div_ceil(64);
        if marker >= self.n_markers
            || self.layout != PackedLayout::StateMajor
            || self.bits_per_allele != 1
            || allele_words.len() < required_words
        {
            return Err("state-major biallelic marker word write dimensions are invalid".to_owned());
        }
        let n_markers = self.n_markers;
        let n_haplotypes = self.n_haplotypes;
        if !self.missing_bits.is_empty() {
            let mut index = marker;
            for _ in 0..n_haplotypes {
                // SAFETY: `index = state * n_markers + marker < n_haplotypes * n_markers`,
                // and `missing_bits` has `(n_haplotypes * n_markers).div_ceil(64)` words.
                unsafe {
                    *self.missing_bits.get_unchecked_mut(index >> 6) &= !(1u64 << (index & 63));
                }
                index += n_markers;
            }
        }
        let swap_word = if swap { u64::MAX } else { 0 };
        let full_words = n_haplotypes / 64;
        let rem_bits = n_haplotypes & 63;
        let tail_mask = if rem_bits == 0 {
            u64::MAX
        } else {
            (1u64 << rem_bits) - 1
        };
        for (word_idx, &raw_word) in allele_words[..required_words].iter().enumerate() {
            let mut bits = raw_word ^ swap_word;
            if word_idx == full_words && rem_bits != 0 {
                bits &= tail_mask;
            }
            let base_hap = word_idx * 64;
            while bits != 0 {
                let tz = bits.trailing_zeros() as usize;
                let index = (base_hap + tz) * n_markers + marker;
                // SAFETY: `base_hap + tz < n_haplotypes` and `marker < n_markers`.
                unsafe {
                    *self.allele_bits.get_unchecked_mut(index >> 6) |= 1u64 << (index & 63);
                }
                bits &= bits - 1;
            }
        }
        Ok(())
    }

    /// Write a biallelic marker row in packed 64-state words.
    ///
    /// When the state count is not word aligned, this uses the scalar writer
    /// to preserve marker-boundary bit offsets exactly.
    #[inline]
    pub fn set_biallelic_marker_with(
        &mut self,
        marker: usize,
        mut allele_at: impl FnMut(usize) -> Option<u16>,
    ) -> Result<(), String> {
        if marker >= self.n_markers {
            return Err("packed haplotype marker index is out of bounds".to_owned());
        }
        if self.bits_per_allele != 1 {
            return Err("bulk biallelic marker writing requires one-bit alleles".to_owned());
        }
        if self.layout != PackedLayout::MarkerMajor {
            return Err("bulk marker writing requires marker-major haplotypes".to_owned());
        }

        let marker_start = marker * self.n_haplotypes;
        for state_start in (0..self.n_haplotypes).step_by(64) {
            let state_count = (self.n_haplotypes - state_start).min(64);
            let mut allele_word = 0u64;
            let mut missing_word = 0u64;
            for bit in 0..state_count {
                match allele_at(state_start + bit) {
                    Some(0) => {}
                    Some(1) => allele_word |= 1u64 << bit,
                    Some(value) => {
                        return Err(format!(
                            "biallelic allele index {value} is outside the range 0..=1"
                        ));
                    }
                    None => missing_word |= 1u64 << bit,
                }
            }
            let bit_start = marker_start + state_start;
            write_bit_chunk(&mut self.allele_bits, bit_start, state_count, allele_word);
            write_bit_chunk(&mut self.missing_bits, bit_start, state_count, missing_word);
        }
        Ok(())
    }

    /// Write a complete biallelic marker from its already packed allele words.
    /// This avoids decoding one `Option<u16>` per state when a donor mosaic has
    /// gathered a row directly from a reference bitplane.
    #[inline]
    pub(crate) fn set_biallelic_marker_words(
        &mut self,
        marker: usize,
        allele_words: &[u64],
    ) -> Result<(), String> {
        if marker >= self.n_markers {
            return Err("packed haplotype marker index is out of bounds".to_owned());
        }
        if self.bits_per_allele != 1 {
            return Err("bulk biallelic marker writing requires one-bit alleles".to_owned());
        }
        if self.layout != PackedLayout::MarkerMajor {
            return Err("bulk marker writing requires marker-major haplotypes".to_owned());
        }
        let required_words = self.n_haplotypes.div_ceil(64);
        if allele_words.len() != required_words {
            return Err("packed biallelic marker word count differs from haplotypes".to_owned());
        }

        let marker_start = marker * self.n_haplotypes;
        for (word_index, state_start) in (0..self.n_haplotypes).step_by(64).enumerate() {
            let state_count = (self.n_haplotypes - state_start).min(64);
            let state_mask = if state_count == 64 {
                u64::MAX
            } else {
                (1u64 << state_count) - 1
            };
            let bit_start = marker_start + state_start;
            write_bit_chunk(
                &mut self.allele_bits,
                bit_start,
                state_count,
                allele_words[word_index] & state_mask,
            );
            write_bit_chunk(&mut self.missing_bits, bit_start, state_count, 0);
        }
        Ok(())
    }

    /// Borrow the allele and missingness words for one biallelic marker when
    /// each marker starts on a 64-state boundary.
    pub(crate) fn biallelic_marker_planes(&self, marker: usize) -> Option<(&[u64], &[u64])> {
        if marker >= self.n_markers
            || self.bits_per_allele != 1
            || self.layout != PackedLayout::MarkerMajor
            || self.n_haplotypes % 64 != 0
        {
            return None;
        }
        let words_per_marker = self.n_haplotypes / 64;
        let start = marker.checked_mul(words_per_marker)?;
        let end = start.checked_add(words_per_marker)?;
        Some((
            self.allele_bits.get(start..end)?,
            self.missing_bits.get(start..end)?,
        ))
    }

    /// Extract a zero-padded 64-state chunk from one biallelic marker row.
    /// Unlike `biallelic_marker_planes`, this supports marker rows whose starts
    /// are not aligned to a machine word.
    pub(crate) fn biallelic_marker_word(
        &self,
        marker: usize,
        first_state: usize,
    ) -> Option<(u64, u64)> {
        if marker >= self.n_markers
            || self.bits_per_allele != 1
            || self.layout != PackedLayout::MarkerMajor
            || first_state >= self.n_haplotypes
            || first_state % 64 != 0
        {
            return None;
        }
        let state_count = (self.n_haplotypes - first_state).min(64);
        let bit_index = marker
            .checked_mul(self.n_haplotypes)?
            .checked_add(first_state)?;
        let word_index = bit_index >> 6;
        let shift = bit_index & 63;
        let mask = if state_count == 64 {
            u64::MAX
        } else {
            (1u64 << state_count) - 1
        };
        let extract = |bits: &[u64]| {
            let low = *bits.get(word_index)?;
            let high = bits.get(word_index + 1).copied().unwrap_or(0);
            let value = if shift == 0 {
                low
            } else {
                (low >> shift) | (high << (64 - shift))
            };
            Some(value & mask)
        };
        Some((extract(&self.allele_bits)?, extract(&self.missing_bits)?))
    }

    #[inline]
    pub(crate) fn state_major_biallelic_words(
        &self,
        state: usize,
    ) -> Option<(usize, &[u64], &[u64])> {
        if state >= self.n_haplotypes
            || self.bits_per_allele != 1
            || self.layout != PackedLayout::StateMajor
        {
            return None;
        }
        Some((
            state.checked_mul(self.n_markers)?,
            &self.allele_bits,
            &self.missing_bits,
        ))
    }

    pub(crate) fn project_state_major_biallelic(
        &self,
        marker_indices: &[usize],
    ) -> Result<Self, String> {
        if self.layout != PackedLayout::StateMajor
            || self.bits_per_allele != 1
            || marker_indices.is_empty()
            || marker_indices
                .iter()
                .any(|&marker| marker >= self.n_markers)
            || marker_indices.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err("packed haplotype projection requires ordered unique markers and state-major biallelic data".to_owned());
        }
        let mut projected = if self.complete {
            Self::empty_complete_biallelic_state_major(self.n_haplotypes, marker_indices.len())?
        } else {
            Self::empty_state_major(self.n_haplotypes, marker_indices.len(), 1)?
        };
        for (target_marker, &source_marker) in marker_indices.iter().enumerate() {
            for state in 0..self.n_haplotypes {
                projected.set_allele(target_marker, state, self.allele(source_marker, state))?;
            }
        }
        Ok(projected)
    }

    #[inline]
    pub(crate) fn state_major_biallelic_allele(&self, marker: usize, state: usize) -> Option<u16> {
        if marker >= self.n_markers
            || state >= self.n_haplotypes
            || self.bits_per_allele != 1
            || self.layout != PackedLayout::StateMajor
        {
            return None;
        }
        let index = state * self.n_markers + marker;
        if self.get_missing(index) {
            return None;
        }
        Some(((self.allele_bits[index >> 6] >> (index & 63)) & 1) as u16)
    }

    /// Read a complete state-major biallelic entry after dimensions and the
    /// zero missingness plane have been validated by the caller.
    #[inline(always)]
    pub(crate) unsafe fn state_major_biallelic_allele_unchecked(
        &self,
        marker: usize,
        state: usize,
    ) -> u16 {
        debug_assert!(marker < self.n_markers);
        debug_assert!(state < self.n_haplotypes);
        debug_assert_eq!(self.bits_per_allele, 1);
        debug_assert!(self.is_state_major());
        let index = state * self.n_markers + marker;
        unsafe { ((*self.allele_bits.get_unchecked(index >> 6) >> (index & 63)) & 1) as u16 }
    }

    #[inline]
    pub(crate) fn state_major_biallelic_marker(
        &self,
        marker: usize,
    ) -> Option<(usize, &[u64], &[u64])> {
        if marker >= self.n_markers
            || self.bits_per_allele != 1
            || self.layout != PackedLayout::StateMajor
        {
            return None;
        }
        Some((self.n_markers, &self.allele_bits, &self.missing_bits))
    }

    /// Overwrite one complete biallelic diploid call in an existing
    /// state-major snapshot. Returns false without writing when the call is
    /// missing or outside the binary allele model.
    #[inline(always)]
    pub(crate) fn overwrite_complete_biallelic_call(
        &mut self,
        marker: usize,
        sample: usize,
        call: DiploidCall,
    ) -> bool {
        if marker >= self.n_markers
            || self.bits_per_allele != 1
            || self.layout != PackedLayout::StateMajor
            || sample
                .checked_mul(2)
                .and_then(|state| state.checked_add(1))
                .is_none_or(|state| state >= self.n_haplotypes)
        {
            return false;
        }
        let [Some(first), Some(second)] = call.alleles else {
            return false;
        };
        if first > 1 || second > 1 {
            return false;
        }
        let first_index = sample * 2 * self.n_markers + marker;
        let second_index = first_index + self.n_markers;
        // SAFETY: dimensions were validated above and indexing uses the
        // state-major biallelic layout.
        unsafe {
            overwrite_bit(&mut self.allele_bits, first_index, first == 1);
            overwrite_bit(&mut self.allele_bits, second_index, second == 1);
            if !self.missing_bits.is_empty() {
                clear_bit(&mut self.missing_bits, first_index);
                clear_bit(&mut self.missing_bits, second_index);
            }
        }
        true
    }

    pub(crate) fn copy_biallelic_state_segment_from(
        &mut self,
        state: usize,
        start_marker: usize,
        end_marker: usize,
        source: &Self,
        source_state: usize,
        copy_missing: bool,
    ) -> Result<(), String> {
        if self.layout != PackedLayout::StateMajor
            || source.layout != PackedLayout::StateMajor
            || self.bits_per_allele != 1
            || source.bits_per_allele != 1
            || self.n_markers != source.n_markers
            || state >= self.n_haplotypes
            || source_state >= source.n_haplotypes
            || start_marker > end_marker
            || end_marker > self.n_markers
        {
            return Err("state-major biallelic segment dimensions are invalid".to_owned());
        }
        let source_start = source_state
            .checked_mul(source.n_markers)
            .and_then(|value| value.checked_add(start_marker))
            .ok_or_else(|| "state-major source segment offset overflow".to_owned())?;
        let destination_start = state
            .checked_mul(self.n_markers)
            .and_then(|value| value.checked_add(start_marker))
            .ok_or_else(|| "state-major destination segment offset overflow".to_owned())?;
        let bit_count = end_marker - start_marker;
        copy_bit_range(
            &mut self.allele_bits,
            destination_start,
            &source.allele_bits,
            source_start,
            bit_count,
        );
        if copy_missing {
            copy_bit_range(
                &mut self.missing_bits,
                destination_start,
                &source.missing_bits,
                source_start,
                bit_count,
            );
        }
        Ok(())
    }

    pub(crate) fn reset_all_missing(&mut self) {
        self.allele_bits.fill(0);
        self.missing_bits.fill(u64::MAX);
        let bit_len = self.n_haplotypes * self.n_markers;
        if let Some(last) = self.missing_bits.last_mut() {
            let used = bit_len & 63;
            if used != 0 {
                *last &= (1u64 << used) - 1;
            }
        }
    }

    fn get_missing(&self, index: usize) -> bool {
        self.missing_bits
            .get(index >> 6)
            .is_some_and(|word| (word >> (index & 63)) & 1 != 0)
    }

    pub fn bytes(&self) -> usize {
        (self.allele_bits.capacity() + self.missing_bits.capacity()) * std::mem::size_of::<u64>()
    }
}

#[inline(always)]
fn write_bit_chunk(words: &mut [u64], bit_start: usize, bit_count: usize, value: u64) {
    let word = bit_start >> 6;
    let shift = bit_start & 63;
    let low_count = bit_count.min(64 - shift);
    let low_mask = if low_count == 64 {
        u64::MAX
    } else {
        (1u64 << low_count) - 1
    };
    let shifted_low_mask = low_mask << shift;
    words[word] = (words[word] & !shifted_low_mask) | ((value & low_mask) << shift);

    let high_count = bit_count - low_count;
    if high_count != 0 {
        let high_mask = (1u64 << high_count) - 1;
        words[word + 1] = (words[word + 1] & !high_mask) | ((value >> low_count) & high_mask);
    }
}

#[inline(always)]
unsafe fn overwrite_bit(words: &mut [u64], bit: usize, value: bool) {
    let word = unsafe { words.get_unchecked_mut(bit >> 6) };
    let mask = 1u64 << (bit & 63);
    if value {
        *word |= mask;
    } else {
        *word &= !mask;
    }
}

#[inline(always)]
unsafe fn clear_bit(words: &mut [u64], bit: usize) {
    let word = unsafe { words.get_unchecked_mut(bit >> 6) };
    *word &= !(1u64 << (bit & 63));
}

#[inline(always)]
pub(crate) fn read_bit_chunk(words: &[u64], bit_start: usize, bit_count: usize) -> u64 {
    debug_assert!(bit_count <= 64);
    if bit_count == 0 {
        return 0;
    }
    let word = bit_start >> 6;
    let shift = bit_start & 63;
    let low = words.get(word).copied().unwrap_or(0) >> shift;
    let high = if shift + bit_count > 64 {
        words.get(word + 1).copied().unwrap_or(0) << (64 - shift)
    } else {
        0
    };
    let mask = if bit_count == 64 {
        u64::MAX
    } else {
        (1u64 << bit_count) - 1
    };
    (low | high) & mask
}

fn copy_bit_range(
    destination: &mut [u64],
    mut destination_start: usize,
    source: &[u64],
    mut source_start: usize,
    mut bit_count: usize,
) {
    while bit_count > 0 {
        let chunk = bit_count.min(64);
        let value = read_bit_chunk(source, source_start, chunk);
        write_bit_chunk(destination, destination_start, chunk, value);
        destination_start += chunk;
        source_start += chunk;
        bit_count -= chunk;
    }
}

#[derive(Debug, Clone)]
pub struct CompositePanel {
    pub haplotypes: PackedHaplotypes,
    pub source_samples: Vec<usize>,
    pub marker_indices: Vec<usize>,
    pub allele_counts: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct PosteriorWindow {
    /// Marker-major concatenation of allele marginals.
    pub allele_probabilities: Vec<f32>,
    /// Allele marginals with the current marker's observation removed.
    /// These support phase updates without letting a tentative phase call
    /// reinforce itself through its own emission.
    pub leave_one_out_allele_probabilities: Vec<f32>,
    /// Half-open offsets into `allele_probabilities`, with one extra end entry.
    pub allele_offsets: Vec<usize>,
    pub state_probabilities: Vec<Vec<f32>>,
    /// Expected latent mismatch and state-transition counts for Li-Stephens
    /// parameter updates.
    pub parameter_stats: HMMParameterStats,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HMMParameterStats {
    pub mismatch_probability_sum: f64,
    pub mismatch_marker_count: usize,
    pub switch_probability_sum: f64,
    pub switch_distance_cm_sum: f64,
}

impl HMMParameterStats {
    pub fn add_assign(&mut self, other: Self) {
        self.mismatch_probability_sum += other.mismatch_probability_sum;
        self.mismatch_marker_count += other.mismatch_marker_count;
        self.switch_probability_sum += other.switch_probability_sum;
        self.switch_distance_cm_sum += other.switch_distance_cm_sum;
    }
}

impl PosteriorWindow {
    pub fn allele_probabilities_at(&self, marker: usize) -> Option<&[f32]> {
        let start = *self.allele_offsets.get(marker)?;
        let end = *self.allele_offsets.get(marker + 1)?;
        self.allele_probabilities.get(start..end)
    }

    pub fn leave_one_out_allele_probabilities_at(&self, marker: usize) -> Option<&[f32]> {
        let start = *self.allele_offsets.get(marker)?;
        let end = *self.allele_offsets.get(marker + 1)?;
        self.leave_one_out_allele_probabilities.get(start..end)
    }
}

#[derive(Debug, Clone)]
pub struct ImputedWindow {
    pub rows: Vec<PhasedRow>,
    /// Row indexes owned by this target window after overlap stitching.
    pub emit_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VariantKey {
    pub chrom: String,
    pub pos: u64,
    pub ref_allele: String,
    pub alt_alleles: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ibd2Interval {
    pub source_sample: usize,
    pub target_sample: usize,
    pub start_marker: usize,
    pub end_marker: usize,
}

#[derive(Debug, Clone, Default)]
pub struct MemoryEstimate {
    pub input_window_bytes: usize,
    pub phase_snapshot_bytes: usize,
    pub phase_initial_pbwt_bytes: usize,
    pub phase_stage1_pbwt_donors_bytes: usize,
    pub phase_stage2_pbwt_donors_bytes: usize,
    pub composite_panel_bytes: usize,
    pub checkpoint_bytes_per_worker: usize,
    pub phase_event_workspace_bytes_per_worker: usize,
    pub scratch_bytes_per_worker: usize,
    pub phase_stage2_checkpoint_bytes_per_worker: usize,
    pub phase_stage2_scratch_bytes_per_worker: usize,
    pub output_buffer_bytes: usize,
    pub imputation_fixed_bytes: usize,
    pub imputation_checkpoint_bytes_per_worker: usize,
    pub imputation_scratch_bytes_per_worker: usize,
    pub output_scratch_bytes_per_worker: usize,
    pub phase_init_pbwt_stage_bytes: usize,
    pub phase_stage1_bytes: usize,
    pub phase_stage2_bytes: usize,
    pub phase_stage_bytes: usize,
    pub imputation_stage_bytes: usize,
    pub output_stage_bytes: usize,
    pub worker_count: usize,
    pub total_bytes: usize,
}

#[derive(Debug, Clone, Default)]
pub struct HMMWorkspace {
    pub forward: Vec<f32>,
    pub forward_values: Vec<f32>,
    pub backward: Vec<f32>,
    pub state_posterior: Vec<f32>,
    pub emission_block: Vec<f32>,
    pub transition_probabilities: Vec<f32>,
    pub checkpoint_values: Vec<f32>,
    pub checkpoint_markers: Vec<usize>,
    pub selected_forward_values: Vec<f32>,
    pub previous_forward: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VcfOutputFields {
    pub dosage: bool,
    pub allele_probabilities: bool,
    pub genotype_probabilities: bool,
}

impl Default for VcfOutputFields {
    fn default() -> Self {
        Self {
            dosage: true,
            allele_probabilities: false,
            genotype_probabilities: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImpConfig {
    #[serde(alias = "gt")]
    pub target: PathBuf,
    #[serde(alias = "ref")]
    pub reference: Option<PathBuf>,
    pub map: Option<PathBuf>,
    pub region: Option<RegionSpec>,
    #[serde(default)]
    pub regions: Vec<RegionSpec>,
    #[serde(alias = "out")]
    pub output: PathBuf,
    #[serde(alias = "fmt")]
    pub format: OutputFormat,
    #[serde(alias = "nthreads", alias = "n_threads")]
    pub threads: usize,
    pub seed: i64,
    #[serde(alias = "window_cm")]
    pub window: f64,
    #[serde(alias = "overlap_cm")]
    pub overlap: f64,
    #[serde(alias = "window_markers")]
    pub window_markers: usize,
    #[serde(alias = "phase_states")]
    pub phase_states: usize,
    #[serde(alias = "imp_states")]
    pub imp_states: usize,
    #[serde(alias = "adaptive_states")]
    pub adaptive_states: bool,
    pub burnin: usize,
    pub iterations: usize,
    #[serde(alias = "initial-lr")]
    pub initial_lr: f64,
    pub ne: u64,
    pub err: Option<f64>,
    pub em: bool,
    pub impute: bool,
    #[serde(alias = "imp_segment")]
    pub imp_segment: f64,
    #[serde(alias = "imp_step")]
    pub imp_step: f64,
    #[serde(alias = "imp_nsteps")]
    pub imp_nsteps: usize,
    pub cluster: f64,
    pub ap: bool,
    pub gp: bool,
    pub assume_global_phase: bool,
    pub diagnostic_site: Option<ImpDiagnosticSite>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionSpec {
    pub chrom: String,
    pub start: Option<u64>,
    pub end: Option<u64>,
}

impl RegionSpec {
    pub fn contains_chromosome(&self, chrom: &str) -> bool {
        normalize_region_chromosome(chrom) == normalize_region_chromosome(&self.chrom)
    }

    pub fn contains(&self, chrom: &str, pos: u64) -> bool {
        self.contains_chromosome(chrom)
            && self.start.is_none_or(|start| pos >= start)
            && self.end.is_none_or(|end| pos <= end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionFilter {
    by_chrom: std::collections::HashMap<String, Vec<(Option<u64>, Option<u64>)>>,
}

impl RegionFilter {
    pub fn from_specs(specs: &[RegionSpec]) -> Option<Self> {
        if specs.is_empty() {
            return None;
        }
        let mut grouped: std::collections::HashMap<String, Vec<(Option<u64>, Option<u64>)>> =
            std::collections::HashMap::new();
        for spec in specs {
            let key = normalize_region_chromosome(&spec.chrom);
            grouped.entry(key).or_default().push((spec.start, spec.end));
        }
        let mut by_chrom = std::collections::HashMap::with_capacity(grouped.len());
        for (chrom, intervals) in grouped {
            if intervals
                .iter()
                .any(|(start, end)| start.is_none() && end.is_none())
            {
                by_chrom.insert(chrom, vec![(None, None)]);
                continue;
            }
            let mut concrete = intervals
                .into_iter()
                .map(|(start, end)| (start.unwrap_or(0), end.unwrap_or(u64::MAX)))
                .collect::<Vec<_>>();
            concrete.sort_unstable();
            let mut merged: Vec<(Option<u64>, Option<u64>)> = Vec::with_capacity(concrete.len());
            for (start, end) in concrete {
                if let Some((_, Some(last_end))) = merged.last_mut() {
                    if start <= last_end.saturating_add(1) {
                        *last_end = (*last_end).max(end);
                        continue;
                    }
                }
                merged.push((Some(start), Some(end)));
            }
            by_chrom.insert(chrom, merged);
        }
        Some(Self { by_chrom })
    }

    pub fn chrom_count(&self) -> usize {
        self.by_chrom.len()
    }

    pub fn contains_chromosome(&self, chrom: &str) -> bool {
        self.by_chrom
            .contains_key(&normalize_region_chromosome(chrom))
    }

    pub fn interval_for(&self, chrom: &str, pos: u64) -> Option<(usize, Option<u64>, Option<u64>)> {
        let intervals = self.by_chrom.get(&normalize_region_chromosome(chrom))?;
        for (index, &(start, end)) in intervals.iter().enumerate() {
            if start.is_none_or(|s| pos >= s) && end.is_none_or(|e| pos <= e) {
                return Some((index, start, end));
            }
        }
        None
    }

    pub fn contains(&self, chrom: &str, pos: u64) -> bool {
        self.interval_for(chrom, pos).is_some()
    }

    pub fn is_past_chrom_bounds(&self, chrom: &str, pos: u64) -> bool {
        self.by_chrom
            .get(&normalize_region_chromosome(chrom))
            .and_then(|intervals| intervals.last())
            .is_some_and(|(_, end)| end.is_some_and(|e| pos > e))
    }

    pub fn same_interval(&self, chrom_a: &str, pos_a: u64, chrom_b: &str, pos_b: u64) -> bool {
        normalize_region_chromosome(chrom_a) == normalize_region_chromosome(chrom_b)
            && self.interval_for(chrom_a, pos_a).map(|(idx, _, _)| idx)
                == self.interval_for(chrom_b, pos_b).map(|(idx, _, _)| idx)
    }
}

pub(crate) fn normalize_region_chromosome(value: &str) -> String {
    let trimmed = value.trim();
    let without_prefix = if trimmed
        .get(..3)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("chr"))
    {
        &trimmed[3..]
    } else {
        trimmed
    };
    without_prefix.to_ascii_uppercase()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpDiagnosticSite {
    pub chrom: String,
    pub position: u64,
    pub sample: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedDiagnosticSite {
    pub chrom: String,
    pub position: u64,
    pub sample: String,
    pub sample_index: usize,
}

impl Default for ImpConfig {
    fn default() -> Self {
        Self {
            target: PathBuf::new(),
            reference: None,
            map: None,
            region: None,
            regions: Vec::new(),
            output: PathBuf::new(),
            format: OutputFormat::Vcf,
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            seed: -99_999,
            window: 40.0,
            overlap: 2.0,
            window_markers: 4_000_000,
            phase_states: 280,
            imp_states: 1_600,
            adaptive_states: true,
            burnin: 3,
            iterations: 12,
            initial_lr: 100_000.0,
            ne: 100_000,
            err: None,
            em: true,
            impute: true,
            imp_segment: 6.0,
            imp_step: 0.1,
            imp_nsteps: 7,
            cluster: 0.005,
            ap: false,
            gp: false,
            assume_global_phase: false,
            diagnostic_site: None,
        }
    }
}

impl ImpConfig {
    pub fn effective_regions(&self) -> Vec<RegionSpec> {
        let mut out = self.regions.clone();
        if let Some(region) = self.region.as_ref() {
            if !out.contains(region) {
                out.push(region.clone());
            }
        }
        out
    }

    pub fn region_filter(&self) -> Option<RegionFilter> {
        RegionFilter::from_specs(&self.effective_regions())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.target.as_os_str().is_empty() {
            return Err("missing required target genotype path (-gt/--target)".to_owned());
        }
        if self.output.as_os_str().is_empty() {
            return Err("missing required output path (-out/--out)".to_owned());
        }
        for region in self.effective_regions() {
            if normalize_region_chromosome(&region.chrom).is_empty() {
                return Err("region chromosome must not be empty".to_owned());
            }
            if region.start.is_some() != region.end.is_some() {
                return Err("region must specify both physical bounds or neither".to_owned());
            }
            if region
                .start
                .zip(region.end)
                .is_some_and(|(start, end)| start > end)
            {
                return Err("region start must not exceed region end".to_owned());
            }
        }
        if self.threads == 0 {
            return Err("threads must be a positive integer".to_owned());
        }
        if !self.window.is_finite() || self.window <= 0.0 {
            return Err("window must be a finite positive cM value".to_owned());
        }
        if !self.overlap.is_finite() || self.overlap <= 0.0 {
            return Err("overlap must be a finite positive cM value".to_owned());
        }
        if self.window < 1.1 * self.overlap {
            return Err("window must be at least 1.1 times the overlap".to_owned());
        }
        if self.window_markers < 2 {
            return Err("window-markers must be at least 2 to preserve window overlap".to_owned());
        }
        if self.phase_states == 0 || self.imp_states == 0 {
            return Err("phase-states and imp-states must be positive integers".to_owned());
        }
        if self.burnin == 0 || self.iterations == 0 {
            return Err("burnin and iterations must be positive integers".to_owned());
        }
        if !self.initial_lr.is_finite() || self.initial_lr < 1.0 {
            return Err("initial-lr must be finite and at least 1".to_owned());
        }
        if self.ne == 0 {
            return Err("ne must be a positive integer".to_owned());
        }
        if let Some(err) = self.err {
            if !err.is_finite() || err < 0.0 || err >= 0.5 {
                return Err("err must be finite and in the range [0, 0.5)".to_owned());
            }
        }
        if !self.imp_segment.is_finite() || self.imp_segment <= 0.0 {
            return Err("imp-segment must be a finite positive cM value".to_owned());
        }
        if !self.imp_step.is_finite() || self.imp_step <= 0.0 {
            return Err("imp-step must be a finite positive cM value".to_owned());
        }
        if self.imp_nsteps < 2 {
            return Err("imp-nsteps must be at least 2".to_owned());
        }
        if !self.cluster.is_finite() || self.cluster < 0.0 {
            return Err("cluster must be a finite nonnegative cM value".to_owned());
        }
        if self.imp_step != 0.1 || self.imp_nsteps != 7 {
            return Err(
                "imp-step/imp-nsteps currently only accept their Beagle-compatible defaults (0.1/7); these matching controls are not yet implemented".to_owned(),
            );
        }
        if self.diagnostic_site.as_ref().is_some_and(|site| {
            site.chrom.is_empty() || site.position == 0 || site.sample.is_empty()
        }) {
            return Err(
                "trace-site requires nonempty CHROM and SAMPLE and a positive POS".to_owned(),
            );
        }
        if (self.reference.is_none() || !self.impute) && (self.ap || self.gp) {
            return Err("--ap and --gp require reference imputation to be enabled".to_owned());
        }
        if self.format == OutputFormat::Pgen && (self.ap || self.gp) {
            return Err("PGEN output cannot store VCF AP/GP fields; choose VCF output".to_owned());
        }
        Ok(())
    }

    pub fn master_seed(&self) -> u64 {
        self.seed as u64
    }

    pub fn hmm_config(&self, haplotype_count: usize) -> Result<HMMConfig, String> {
        self.validate()?;
        let err = match self.err {
            Some(value) => value,
            None => default_err(haplotype_count)?,
        };
        Ok(HMMConfig {
            ne: self.ne,
            recombination_intensity: (0.04_f32 * self.ne as f32) / haplotype_count as f32,
            err: f64::from(err as f32),
            em: self.em,
            burnin: self.burnin,
            iterations: self.iterations,
            initial_lr: self.initial_lr,
            phase_states: self.phase_states,
            haplotype_count,
        })
    }

    pub fn phase_config(&self, haplotype_count: usize) -> Result<PhaseConfig, String> {
        const DEFAULT_PACKED_CACHE_BUDGET_BYTES: usize = 256 * 1024 * 1024;
        let hmm = self.hmm_config(haplotype_count)?;
        Ok(PhaseConfig {
            window_cm: self.window,
            overlap_cm: self.overlap,
            window_markers: self.window_markers,
            seed: self.master_seed(),
            hmm,
            packed_cache_budget_bytes: DEFAULT_PACKED_CACHE_BUDGET_BYTES,
        })
    }

    pub fn impute_config(&self) -> ImputeConfig {
        ImputeConfig {
            states: self.imp_states,
            window_markers: self.window_markers,
            segment_cm: self.imp_segment,
            step_cm: self.imp_step,
            nsteps: self.imp_nsteps,
            cluster_cm: self.cluster,
            output: VcfOutputFields {
                dosage: self.format == OutputFormat::Vcf,
                allele_probabilities: self.ap,
                genotype_probabilities: self.gp,
            },
        }
    }
}

pub fn default_err(haplotype_count: usize) -> Result<f64, String> {
    if haplotype_count == 0 {
        return Err("cannot derive default err without any haplotypes".to_owned());
    }
    let h = haplotype_count as f64;
    let theta = 1.0 / (0.5 + h.ln());
    Ok(theta / (2.0 * (theta + h)))
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HMMConfig {
    pub ne: u64,
    /// Beagle stores and updates this float directly; `ne` is only its
    /// rounded reporting value and must not be used to reconstruct transitions.
    pub recombination_intensity: f32,
    pub err: f64,
    pub em: bool,
    pub burnin: usize,
    pub iterations: usize,
    pub initial_lr: f64,
    pub phase_states: usize,
    /// Total haplotypes in the population used to calibrate recombination
    /// intensity. This is distinct from the bounded number of HMM states.
    pub haplotype_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhaseConfig {
    pub window_cm: f64,
    pub overlap_cm: f64,
    pub window_markers: usize,
    pub seed: u64,
    pub hmm: HMMConfig,
    pub(crate) packed_cache_budget_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ImputeConfig {
    pub states: usize,
    pub window_markers: usize,
    pub segment_cm: f64,
    pub step_cm: f64,
    pub nsteps: usize,
    pub cluster_cm: f64,
    pub output: VcfOutputFields,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ImpSummary {
    pub variants_written: u64,
    pub samples_written: usize,
    pub peak_window_bytes: u64,
    pub region: Option<RegionSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<RegionSpec>,
    pub total_target_windows: u64,
    pub direct_imputation_windows: u64,
    pub normal_phasing_windows: u64,
    pub reference_phasing_windows: u64,
    pub imputation_windows: u64,
    pub direct_fallback_reasons: BTreeMap<String, u64>,
    pub direct_bypass_fraction: f64,
    pub parameter_trace: Vec<HMMParameterTrace>,
    pub hmm_trace: Vec<ImpHmmTrace>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ImpProgress {
    pub stage: String,
    pub chrom: String,
    pub window_index: u64,
    pub markers: usize,
    pub variants_written: u64,
    pub samples: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ImpConversionSummary {
    pub samples: usize,
    pub variants: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HMMParameterTrace {
    pub chrom: String,
    pub window_index: u64,
    pub stage: String,
    pub round: usize,
    pub ne: u64,
    pub err: f64,
}

impl Default for ImpSummary {
    fn default() -> Self {
        Self {
            variants_written: 0,
            samples_written: 0,
            peak_window_bytes: 0,
            region: None,
            regions: Vec::new(),
            total_target_windows: 0,
            direct_imputation_windows: 0,
            normal_phasing_windows: 0,
            reference_phasing_windows: 0,
            imputation_windows: 0,
            direct_fallback_reasons: BTreeMap::new(),
            direct_bypass_fraction: 0.0,
            parameter_trace: Vec::new(),
            hmm_trace: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImpHmmTrace {
    pub chrom: String,
    pub position: u64,
    pub sample: String,
    pub window_index: u64,
    pub round: usize,
    /// Beagle event type that owned the traced marker during this phase round.
    pub cluster_kind: Option<String>,
    pub transition_probability: f32,
    /// Alleles carried by each HMM track's reference states at the traced marker.
    pub state_alleles: Vec<Vec<Option<u16>>>,
    /// Emission probability by state for genotype, haplotype 0, haplotype 1.
    pub emission_probabilities: Vec<Vec<f32>>,
    /// Filtered forward probability by state for each HMM track.
    pub forward_probabilities: Vec<Vec<f32>>,
    /// Backward probability by state for each HMM track.
    pub backward_probabilities: Vec<Vec<f32>>,
    /// Sum of forward[k] * backward[k] for each HMM track.
    pub posterior_normalization: Vec<f64>,
    pub phase_swap: Option<bool>,
    pub phase_log_odds: Option<f64>,
    pub phase_threshold: f64,
    pub selected_alleles: [Option<u16>; 2],
}

#[cfg(test)]
mod packed_mosaic_tests {
    use super::{
        DiploidCall, DiploidCalls, ImpConfig, PackedHaplotypes, PhaseSetId, PhasedRow,
        PhasedWindow, VariantMeta,
    };

    #[test]
    fn adaptive_imputation_states_are_enabled_by_default() {
        assert!(ImpConfig::default().adaptive_states);
    }

    #[test]
    fn packed_phase_cache_budget_is_fixed() {
        let config = ImpConfig {
            target: "target.vcf.gz".into(),
            output: "phased.vcf.gz".into(),
            ..ImpConfig::default()
        };
        assert_eq!(
            config
                .phase_config(1_000)
                .unwrap()
                .packed_cache_budget_bytes,
            256 * 1024 * 1024
        );
    }

    #[test]
    fn complete_window_packing_matches_general_state_major_writer() {
        let rows = (0..3)
            .map(|marker| PhasedRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("m{marker}"),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(
                    (0..4)
                        .map(|sample| {
                            let first = ((marker + sample) % 2) as u16;
                            let second = ((marker + sample + 1) % 2) as u16;
                            DiploidCall {
                                alleles: [Some(first), Some(second)],
                                phased: true,
                            }
                        })
                        .collect(),
                ),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            })
            .collect::<Vec<_>>();
        let window = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            emit_start: 0,
            emit_end: 3,
        };
        let observed = PackedHaplotypes::complete_biallelic_state_major(&window)
            .unwrap()
            .unwrap();
        let mut expected = PackedHaplotypes::empty_state_major(8, 3, 1).unwrap();
        for marker in 0..3 {
            for haplotype in 0..8 {
                expected
                    .set_allele(
                        marker,
                        haplotype,
                        window.rows[marker].calls.get_allele_unchecked(haplotype),
                    )
                    .unwrap();
            }
        }
        assert_eq!(observed.allele_bits, expected.allele_bits);
        assert!(observed.missing_bits.iter().all(|&word| word == 0));
    }

    #[test]
    fn persistent_state_major_refresh_matches_full_window_packing() {
        let rows = (0..70)
            .map(|marker| PhasedRow {
                meta: VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("m{marker}"),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: DiploidCalls::from_calls(
                    (0..3)
                        .map(|sample| DiploidCall {
                            alleles: [
                                Some(((marker + sample) & 1) as u16),
                                Some(((marker * 3 + sample) & 1) as u16),
                            ],
                            phased: true,
                        })
                        .collect(),
                ),
                dosage: None,
                ap1: None,
                ap2: None,
                gp: None,
                info_af: None,
                info_dr2: None,
                imputed: false,
            })
            .collect::<Vec<_>>();
        let window = PhasedWindow {
            chrom: "1".to_owned(),
            index: 0,
            rows,
            emit_start: 0,
            emit_end: 70,
        };
        let expected = PackedHaplotypes::complete_biallelic_state_major(&window)
            .unwrap()
            .unwrap();
        let mut refreshed = expected.clone();
        refreshed.allele_bits.fill(u64::MAX);
        assert!(refreshed
            .overwrite_complete_biallelic_state_major_from_calls(3, 70, |marker, sample| {
                window.rows[marker].calls.get(sample).unwrap()
            })
            .unwrap());
        assert_eq!(refreshed.allele_bits, expected.allele_bits);
        assert_eq!(refreshed.missing_bits, expected.missing_bits);
    }

    #[test]
    fn pgen_call_row_packing_matches_scalar_conversion_for_byte_values_and_tails() {
        for packed_byte in 0u16..=u8::MAX as u16 {
            let packed_byte = packed_byte as u8;
            let packed_row = [packed_byte; 4];
            let phases = (0..16)
                .map(|sample| match (packed_byte as usize + sample) % 3 {
                    0 => None,
                    1 => Some(false),
                    _ => Some(true),
                })
                .collect::<Vec<_>>();
            for sample_count in 0..=13 {
                let actual = DiploidCalls::from_packed_biallelic_row(
                    &packed_row,
                    Some(&phases),
                    sample_count,
                )
                .unwrap();
                let expected = DiploidCalls::from_iter((0..sample_count).map(|sample| {
                    let code = (packed_row[sample >> 2] >> ((sample & 3) * 2)) & 0x03;
                    match (code, phases[sample]) {
                        (0, _) => DiploidCall {
                            alleles: [Some(0), Some(0)],
                            phased: false,
                        },
                        (1, _) => DiploidCall::missing(false),
                        (2, Some(true)) => DiploidCall {
                            alleles: [Some(1), Some(0)],
                            phased: true,
                        },
                        (2, Some(false)) => DiploidCall {
                            alleles: [Some(0), Some(1)],
                            phased: true,
                        },
                        (2, None) => DiploidCall {
                            alleles: [Some(0), Some(1)],
                            phased: false,
                        },
                        (3, _) => DiploidCall {
                            alleles: [Some(1), Some(1)],
                            phased: false,
                        },
                        _ => unreachable!(),
                    }
                }));
                assert_eq!(
                    actual, expected,
                    "byte {packed_byte:#04x}, samples {sample_count}"
                );
            }
        }
    }

    #[test]
    fn packed_phase_bits_match_optional_phase_conversion() {
        let codes = [0u8, 1, 2, 3, 2, 0, 1, 2];
        let phases = [
            None,
            None,
            Some(true),
            None,
            Some(false),
            None,
            None,
            Some(false),
        ];
        let mut row = [0u8; 2];
        for (sample, &code) in codes.iter().enumerate() {
            row[sample >> 2] |= code << ((sample & 3) * 2);
        }

        let mut present = [0u64; 1];
        let mut orientation = [0u64; 1];
        for sample in 0..codes.len() {
            if let Some(phase) = phases[sample] {
                present[0] |= 1u64 << sample;
                if phase {
                    orientation[0] |= 1u64 << sample;
                }
            }
        }

        let expected =
            DiploidCalls::from_packed_biallelic_row(&row, Some(&phases), codes.len()).unwrap();
        let observed = DiploidCalls::from_packed_biallelic_phase_bits(
            &row,
            &present,
            &orientation,
            codes.len(),
        )
        .unwrap();
        assert_eq!(observed, expected);
    }

    #[test]
    fn biallelic_call_bytes_expand_to_haplotype_words_with_correct_swap_and_padding() {
        let calls = (0..70)
            .map(|sample| match sample % 5 {
                0 => DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: false,
                },
                1 => DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: false,
                },
                2 => DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: true,
                },
                3 => DiploidCall {
                    alleles: [Some(1), Some(0)],
                    phased: true,
                },
                _ => DiploidCall {
                    alleles: [Some(1), Some(1)],
                    phased: false,
                },
            })
            .collect::<Vec<_>>();
        let packed = DiploidCalls::from_calls(calls);
        let haplotype_count = packed.len() * 2;

        for swap in [false, true] {
            for word_index in 0..haplotype_count.div_ceil(64) {
                let observed = packed.biallelic_haplotype_word(word_index, swap).unwrap();
                for bit in 0..64 {
                    let haplotype = word_index * 64 + bit;
                    let expected = if haplotype < haplotype_count {
                        packed.get_allele_unchecked(haplotype).unwrap() ^ u16::from(swap)
                    } else {
                        0
                    };
                    assert_eq!((observed >> bit) & 1, u64::from(expected));
                }
            }
        }

        let incomplete = DiploidCalls::from_calls(vec![DiploidCall {
            alleles: [Some(1), None],
            phased: false,
        }]);
        assert_eq!(incomplete.biallelic_haplotype_word(0, false), None);
    }

    #[test]
    fn biallelic_marker_planes_expose_word_aligned_alleles_and_missingness() {
        let mut packed = PackedHaplotypes::empty(128, 2, 1).unwrap();
        for marker in 0..2 {
            for state in 0..128 {
                let allele = match (marker * 3 + state) % 11 {
                    0 => None,
                    1 | 2 | 3 => Some(1),
                    _ => Some(0),
                };
                packed.set_allele(marker, state, allele).unwrap();
            }
        }

        for marker in 0..2 {
            let (allele_words, missing_words) = packed.biallelic_marker_planes(marker).unwrap();
            assert_eq!(allele_words.len(), 2);
            assert_eq!(missing_words.len(), 2);
            for word in 0..2 {
                let mut expected_alleles = 0u64;
                let mut expected_missing = 0u64;
                for bit in 0..64 {
                    match (marker * 3 + word * 64 + bit) % 11 {
                        0 => expected_missing |= 1u64 << bit,
                        1 | 2 | 3 => expected_alleles |= 1u64 << bit,
                        _ => {}
                    }
                }
                assert_eq!(allele_words[word], expected_alleles);
                assert_eq!(missing_words[word], expected_missing);
            }
        }

        assert!(packed.biallelic_marker_planes(2).is_none());
        assert!(PackedHaplotypes::empty(70, 1, 1)
            .unwrap()
            .biallelic_marker_planes(0)
            .is_none());
        assert!(PackedHaplotypes::empty(64, 1, 2)
            .unwrap()
            .biallelic_marker_planes(0)
            .is_none());
    }

    #[test]
    fn biallelic_marker_word_extracts_unaligned_rows_across_word_boundaries() {
        let mut packed = PackedHaplotypes::empty(70, 2, 1).unwrap();
        for marker in 0..2 {
            for state in 0..70 {
                let missing = (marker == 0 && matches!(state, 3 | 67))
                    || (marker == 1 && matches!(state, 4 | 65));
                let allele = if missing {
                    None
                } else {
                    Some((state % 2) as u16)
                };
                packed.set_allele(marker, state, allele).unwrap();
            }
        }

        assert_eq!(
            packed.biallelic_marker_word(0, 0),
            Some((0xAAAAAAAAAAAAAAA2, 0x8))
        );
        assert_eq!(packed.biallelic_marker_word(0, 64), Some((0x22, 0x8)));
        assert_eq!(
            packed.biallelic_marker_word(1, 0),
            Some((0xAAAAAAAAAAAAAAAA, 0x10))
        );
        assert_eq!(packed.biallelic_marker_word(1, 64), Some((0x28, 0x2)));
        assert!(packed.biallelic_marker_word(0, 1).is_none());
        assert!(packed.biallelic_marker_word(2, 0).is_none());
        assert!(PackedHaplotypes::empty(64, 1, 2)
            .unwrap()
            .biallelic_marker_word(0, 0)
            .is_none());
    }

    #[test]
    fn direct_haplotype_allele_access_matches_packed_and_complex_calls() {
        let calls = DiploidCalls::from_calls(vec![
            DiploidCall {
                alleles: [Some(0), Some(0)],
                phased: false,
            },
            DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: false,
            },
            DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: true,
            },
            DiploidCall {
                alleles: [Some(1), Some(0)],
                phased: true,
            },
            DiploidCall {
                alleles: [Some(1), Some(1)],
                phased: false,
            },
            DiploidCall::missing(false),
            DiploidCall {
                alleles: [Some(1), None],
                phased: true,
            },
            DiploidCall {
                alleles: [Some(2), Some(1)],
                phased: true,
            },
        ]);
        let expected = [
            [Some(0), Some(0)],
            [Some(0), Some(1)],
            [Some(0), Some(1)],
            [Some(1), Some(0)],
            [Some(1), Some(1)],
            [None, None],
            [Some(1), None],
            [Some(2), Some(1)],
        ];

        for (sample, alleles) in expected.into_iter().enumerate() {
            for (haplotype, allele) in alleles.into_iter().enumerate() {
                assert_eq!(calls.get_allele(sample, haplotype), allele);
            }
        }
        assert_eq!(calls.get_allele(calls.len(), 0), None);
        assert_eq!(calls.get_allele(0, 2), None);
    }

    #[test]
    fn unchecked_haplotype_allele_access_matches_checked_access() {
        let calls = DiploidCalls::from_calls(vec![
            DiploidCall {
                alleles: [Some(0), Some(0)],
                phased: false,
            },
            DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: false,
            },
            DiploidCall {
                alleles: [Some(1), Some(0)],
                phased: true,
            },
            DiploidCall::missing(false),
            DiploidCall {
                alleles: [Some(2), Some(1)],
                phased: true,
            },
        ]);

        for sample in 0..calls.len() {
            for haplotype in 0..2 {
                assert_eq!(
                    calls.get_allele_unchecked(sample * 2 + haplotype),
                    calls.get_allele(sample, haplotype)
                );
            }
        }
    }

    #[test]
    fn direct_call_packing_matches_batch_construction() {
        let values = [
            DiploidCall {
                alleles: [Some(0), Some(0)],
                phased: true,
            },
            DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: true,
            },
            DiploidCall {
                alleles: [Some(1), Some(0)],
                phased: true,
            },
            DiploidCall {
                alleles: [Some(1), None],
                phased: false,
            },
            DiploidCall::missing(true),
            DiploidCall {
                alleles: [Some(2), Some(1)],
                phased: true,
            },
        ];
        let expected = DiploidCalls::from_calls(values.to_vec());
        let mut observed = DiploidCalls::empty(values.len());
        for (sample, &call) in values.iter().enumerate() {
            observed.set_call_unchecked(sample, call);
        }
        assert_eq!(observed, expected);
    }

    #[test]
    fn biallelic_bulk_marker_writer_matches_scalar_for_aligned_state_count() {
        let values = (0..3)
            .map(|marker| {
                (0..128)
                    .map(|state| match (marker + state) % 5 {
                        0 => None,
                        1 | 2 => Some(1),
                        _ => Some(0),
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut actual = PackedHaplotypes::empty(128, values.len(), 1).unwrap();
        let mut expected = PackedHaplotypes::empty(128, values.len(), 1).unwrap();

        for (marker, alleles) in values.iter().enumerate() {
            actual
                .set_biallelic_marker_with(marker, |state| alleles[state])
                .unwrap();
            for (state, &allele) in alleles.iter().enumerate() {
                expected.set_allele(marker, state, allele).unwrap();
            }
        }

        assert_eq!(actual.allele_bits, expected.allele_bits);
        assert_eq!(actual.missing_bits, expected.missing_bits);
    }

    #[test]
    fn biallelic_marker_word_writer_matches_callback_for_aligned_and_unaligned_rows() {
        for state_count in [70usize, 128, 1600] {
            let mut scalar = PackedHaplotypes::empty(state_count, 3, 1).unwrap();
            let mut bulk = PackedHaplotypes::empty(state_count, 3, 1).unwrap();
            for marker in 0..3 {
                let mut words = vec![0u64; state_count.div_ceil(64)];
                for state in 0..state_count {
                    if (state * 13 + marker * 17) % 19 < 7 {
                        words[state / 64] |= 1u64 << (state % 64);
                    }
                }
                scalar
                    .set_biallelic_marker_with(marker, |state| {
                        Some(((words[state / 64] >> (state % 64)) & 1) as u16)
                    })
                    .unwrap();
                bulk.set_biallelic_marker_words(marker, &words).unwrap();
            }
            assert_eq!(bulk.allele_bits, scalar.allele_bits);
            assert_eq!(bulk.missing_bits, scalar.missing_bits);
        }
    }

    #[test]
    fn biallelic_bulk_marker_writer_handles_unaligned_state_count() {
        let haplotype_count = 280;
        let marker_count = 5;
        let mut actual = PackedHaplotypes::empty(haplotype_count, marker_count, 1).unwrap();
        let mut expected = PackedHaplotypes::empty(haplotype_count, marker_count, 1).unwrap();

        for marker in 0..marker_count {
            actual
                .set_biallelic_marker_with(marker, |state| match (state + marker * 3) % 7 {
                    0 | 1 => None,
                    2 | 3 => Some(1),
                    _ => Some(0),
                })
                .unwrap();
            for state in 0..haplotype_count {
                let allele = match (state + marker * 3) % 7 {
                    0 | 1 => None,
                    2 | 3 => Some(1),
                    _ => Some(0),
                };
                expected.set_allele(marker, state, allele).unwrap();
            }
        }

        assert_eq!(actual.allele_bits, expected.allele_bits);
        assert_eq!(actual.missing_bits, expected.missing_bits);
    }

    #[test]
    fn biallelic_bulk_marker_writer_rejects_nonbinary_alleles() {
        let mut packed = PackedHaplotypes::empty(64, 1, 1).unwrap();
        let result = packed.set_biallelic_marker_with(0, |_| Some(2));
        assert!(result.is_err());
    }

    #[test]
    fn complete_phased_biallelic_validation_matches_call_invariants() {
        let mut calls = DiploidCalls::from_calls(vec![
            DiploidCall {
                alleles: [Some(0), Some(0)],
                phased: false,
            },
            DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: true,
            },
            DiploidCall {
                alleles: [Some(1), Some(0)],
                phased: true,
            },
            DiploidCall {
                alleles: [Some(1), Some(1)],
                phased: false,
            },
        ]);
        assert_eq!(calls.first_invalid_complete_phased_biallelic(), None);

        let invalid_calls = [
            DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: false,
            },
            DiploidCall::missing(false),
            DiploidCall {
                alleles: [Some(0), None],
                phased: false,
            },
            DiploidCall {
                alleles: [Some(2), Some(2)],
                phased: false,
            },
        ];
        for (sample, invalid) in invalid_calls.into_iter().enumerate() {
            calls.set_call(sample, invalid).unwrap();
            assert_eq!(
                calls.first_invalid_complete_phased_biallelic(),
                Some(sample)
            );
            calls
                .set_call(
                    sample,
                    DiploidCall {
                        alleles: [Some(sample as u16 & 1), Some(sample as u16 & 1)],
                        phased: false,
                    },
                )
                .unwrap();
        }
    }

    #[test]
    fn phase_set_storage_is_optional_and_counted_in_memory() {
        let mut calls = DiploidCalls::empty(2);
        assert!(!calls.has_phase_set_track());
        assert_eq!(calls.phase_set(0), None);
        let before = calls.bytes();
        calls.set_phase_set(0, Some(PhaseSetId::new(7))).unwrap();
        calls.set_phase_set(1, Some(PhaseSetId::new(7))).unwrap();
        assert!(calls.has_phase_set_track());
        assert_eq!(calls.phase_set(0), Some(PhaseSetId::new(7)));
        assert_eq!(calls.phase_set(1), Some(PhaseSetId::new(7)));
        assert!(calls.bytes() > before);
    }
}
