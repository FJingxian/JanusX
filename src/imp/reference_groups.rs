//! Sequence compression block identity for Beagle-compatible imputation.
//!
//! Only the partition and block identity are needed here, not compressed
//! records. Keep O(reference haplotypes) state across all reference windows.

use super::types::DiploidCalls;

pub(crate) struct ReferenceSequenceGroups {
    chromosome: String,
    group: u64,
    hap_to_sequence: Vec<usize>,
    sequence_counts: Vec<usize>,
    minor_counts: Vec<usize>,
    minor_sequences: Vec<usize>,
    allele_words: Vec<u64>,
    max_sequences: usize,
    max_major_count: usize,
}

impl ReferenceSequenceGroups {
    pub(crate) fn new(samples: usize) -> Self {
        // SeqCoder3.defaultMaxNSeq and RefIt.maxSeqCodingMajorCnt. The latter
        // deliberately multiplies and subtracts in float32 before flooring.
        let max_sequences = if samples == 1 {
            3
        } else {
            2.0f64.powf(2.0 * (samples as f64).log10() + 1.0).floor() as usize
        }
        .min(65_535);
        let haplotypes = samples * 2;
        Self {
            chromosome: String::new(),
            group: 0,
            hap_to_sequence: vec![0; haplotypes],
            sequence_counts: vec![haplotypes],
            minor_counts: Vec::new(),
            minor_sequences: Vec::new(),
            allele_words: vec![0; haplotypes.div_ceil(64)],
            max_sequences,
            max_major_count: ((haplotypes as f32 * 0.995f32 - 1.0f32).floor().max(0.0)) as usize,
        }
    }

    fn reset(&mut self) {
        self.hap_to_sequence.fill(0);
        self.sequence_counts.clear();
        self.sequence_counts.push(self.hap_to_sequence.len());
        self.group += 1;
    }

    /// `None` denotes an allele-coded (very rare or monomorphic) record. Such
    /// records neither refine the sequence partition nor reset its identity.
    pub(crate) fn push(
        &mut self,
        chromosome: &str,
        calls: &DiploidCalls,
    ) -> Result<Option<u64>, String> {
        if self.chromosome != chromosome {
            self.chromosome.clear();
            self.chromosome.push_str(chromosome);
            self.reset();
        }
        if calls.len() * 2 != self.hap_to_sequence.len() {
            return Err("reference sequence grouping sample count changed".to_owned());
        }
        let mut alt_count = 0usize;
        for (word, output) in self.allele_words.iter_mut().enumerate() {
            *output = calls.biallelic_haplotype_word(word, false).ok_or_else(|| {
                "reference sequence grouping requires complete biallelic GT".to_owned()
            })?;
            alt_count += output.count_ones() as usize;
        }
        let haplotypes = self.hap_to_sequence.len();
        let major_is_alt = alt_count > haplotypes / 2;
        let major_count = alt_count.max(haplotypes - alt_count);
        if major_count > self.max_major_count {
            return Ok(None);
        }
        if major_is_alt {
            for (word, bits) in self.allele_words.iter_mut().enumerate() {
                let remaining = (haplotypes - word * 64).min(64);
                *bits = !*bits & (u64::MAX >> (64 - remaining));
            }
        }
        // A partition split is independent of sequence numbering. Keeping
        // the major subpartition's old ID lets us update only minority haps.
        if !self.refine() {
            self.reset();
            if !self.refine() {
                return Err(
                    "reference sequence grouping cannot encode a biallelic marker".to_owned(),
                );
            }
        }
        Ok(Some(self.group))
    }

    fn refine(&mut self) -> bool {
        let previous_count = self.sequence_counts.len();
        self.minor_counts.clear();
        self.minor_counts.resize(previous_count, 0);
        for (word, &value) in self.allele_words.iter().enumerate() {
            let mut bits = value;
            while bits != 0 {
                let hap = word * 64 + bits.trailing_zeros() as usize;
                self.minor_counts[self.hap_to_sequence[hap]] += 1;
                bits &= bits - 1;
            }
        }
        let splits = self
            .minor_counts
            .iter()
            .zip(&self.sequence_counts)
            .filter(|&(minor, total)| *minor != 0 && minor < total)
            .count();
        if previous_count + splits > self.max_sequences {
            return false;
        }
        self.minor_sequences.clear();
        for sequence in 0..previous_count {
            let minor = self.minor_counts[sequence];
            let total = self.sequence_counts[sequence];
            if minor != 0 && minor < total {
                self.minor_sequences.push(self.sequence_counts.len());
                self.sequence_counts[sequence] -= minor;
                self.sequence_counts.push(minor);
            } else {
                self.minor_sequences.push(sequence);
            }
        }
        for (word, &value) in self.allele_words.iter().enumerate() {
            let mut bits = value;
            while bits != 0 {
                let hap = word * 64 + bits.trailing_zeros() as usize;
                self.hap_to_sequence[hap] = self.minor_sequences[self.hap_to_sequence[hap]];
                bits &= bits - 1;
            }
        }
        true
    }
}
