//! Core types, configuration, genetic map interpolation, and compact 2-bit
//! whole-chromosome haplotype storage for `src/haploid`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Controls whether samples are treated as pseudo-haploid inbred/DH lines or
/// diploid individuals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PloidyMode {
    /// Auto-detect per sample using `inbred_het_threshold`.
    #[default]
    Auto,
    /// Force every sample to be treated as a single pseudo-haplotype (inbred / DH).
    Inbred,
    /// Force every sample to be treated as diploid (2 haplotypes).
    Diploid,
}

/// Operating mode selected automatically by whether `--ref` is provided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HaploidIbdMode {
    PairwiseIbd,
    MultiParentAncestry,
}

fn default_threads() -> usize {
    1
}
fn default_inbred_het_threshold() -> f64 {
    0.02
}
fn default_min_seed_cm() -> f64 {
    1.0
}
fn default_min_output_cm() -> f64 {
    2.0
}
fn default_max_gap_cm() -> f64 {
    0.5
}
fn default_max_mismatch_rate() -> f64 {
    0.005
}
fn default_min_markers() -> usize {
    30
}
fn default_ancestry_switch_penalty() -> f32 {
    12.0
}
fn default_ancestry_top_donors() -> usize {
    64
}
fn default_ancestry_step_cm() -> f64 {
    0.05
}
fn default_cm_per_mb() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HaploidIbdConfig {
    #[serde(alias = "gt")]
    pub target: PathBuf,
    #[serde(alias = "ref")]
    pub reference: Option<PathBuf>,
    pub map: Option<PathBuf>,
    #[serde(alias = "out")]
    pub output: PathBuf,
    pub threads: usize,
    pub ploidy_mode: PloidyMode,
    pub inbred_het_threshold: f64,
    pub min_seed_cm: f64,
    pub min_output_cm: f64,
    pub max_gap_cm: f64,
    pub max_mismatch_rate: f64,
    pub min_markers: usize,
    pub ancestry_switch_penalty: f32,
    pub ancestry_top_donors: usize,
    pub ancestry_step_cm: f64,
    pub cm_per_mb: f64,
    pub write_tsv: bool,
}

impl Default for HaploidIbdConfig {
    fn default() -> Self {
        Self {
            target: PathBuf::new(),
            reference: None,
            map: None,
            output: PathBuf::new(),
            threads: default_threads(),
            ploidy_mode: PloidyMode::Auto,
            inbred_het_threshold: default_inbred_het_threshold(),
            min_seed_cm: default_min_seed_cm(),
            min_output_cm: default_min_output_cm(),
            max_gap_cm: default_max_gap_cm(),
            max_mismatch_rate: default_max_mismatch_rate(),
            min_markers: default_min_markers(),
            ancestry_switch_penalty: default_ancestry_switch_penalty(),
            ancestry_top_donors: default_ancestry_top_donors(),
            ancestry_step_cm: default_ancestry_step_cm(),
            cm_per_mb: default_cm_per_mb(),
            write_tsv: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HaploidIbdSummary {
    pub mode: HaploidIbdMode,
    pub chromosomes: usize,
    pub markers_processed: u64,
    pub target_samples: usize,
    pub reference_samples: usize,
    pub inbred_samples: usize,
    pub diploid_samples: usize,
    pub segments_written: u64,
    pub sparse_bins_written: u64,
    pub sparse_nnz: u64,
    pub output_path: String,
    pub tsv_path: Option<String>,
    pub elapsed_s: f64,
}

/// Pairwise IBD segment emitted when `--ref` is not provided.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IbdSegment {
    pub sample1: u32,
    /// Primary supporting haplotype index on `sample1` (`0` or `1`).
    pub hap1: u8,
    pub sample2: u32,
    /// Primary supporting haplotype index on `sample2` (`0` or `1`).
    pub hap2: u8,
    /// For diploid IBD2 segments (`ibd_type == 2`), the complementary second
    /// supporting haplotype pair `(hap1_second, hap2_second)` — e.g. `Some((1, 1))`
    /// for parallel `(0,0)+(1,1)` IBD2 or `Some((1, 0))` for crossed `(0,1)+(1,0)`
    /// IBD2. `None` for IBD1 and inbred/DH single-haplotype IBD2.
    #[serde(default)]
    pub track2: Option<(u8, u8)>,
    pub chrom_idx: u16,
    pub start_bp: u64,
    pub end_bp: u64,
    pub start_cm: f32,
    pub end_cm: f32,
    /// `1` for IBD1 (single shared haplotype), `2` for IBD2 (both haplotypes / inbred line match).
    pub ibd_type: u8,
    pub n_markers: u32,
    pub mismatch_rate: f32,
}

/// Contiguous multi-parent ancestry block emitted when `--ref` is provided.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AncestrySegment {
    pub target_sample: u32,
    pub hap: u8,
    pub chrom_idx: u16,
    pub start_bp: u64,
    pub end_bp: u64,
    pub start_cm: f32,
    pub end_cm: f32,
    /// Index into reference sample list (parent ID).
    pub parent_sample: u32,
    /// Reference haplotype index (`0` or `1`, `0` for inbred parent).
    pub parent_hap: u8,
    /// Local posterior confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Number of parents tied within epsilon of the winning Viterbi score.
    pub ambiguous_parents: u16,
    pub n_markers: u32,
}

/// Marker-major compact 2-bit haplotype storage for a single chromosome.
/// Stores one `allele_bit` (`0` = ref, `1` = alt) and one `missing_bit`
/// (`1` = missing or unphased-het wildcard) per `(marker, haplotype)`.
#[derive(Debug, Clone)]
pub struct BitHaplotypeMatrix {
    pub n_haps: usize,
    pub n_markers: usize,
    pub words_per_marker: usize,
    pub allele_bits: Vec<u64>,
    pub missing_bits: Vec<u64>,
}

impl BitHaplotypeMatrix {
    pub fn new(n_haps: usize, n_markers: usize) -> Self {
        let words_per_marker = n_haps.div_ceil(64);
        let total_words = n_markers.saturating_mul(words_per_marker);
        Self {
            n_haps,
            n_markers,
            words_per_marker,
            allele_bits: vec![0u64; total_words],
            missing_bits: vec![0u64; total_words],
        }
    }

    pub fn with_capacity(n_haps: usize, marker_capacity: usize) -> Self {
        let words_per_marker = n_haps.div_ceil(64);
        let word_cap = marker_capacity.saturating_mul(words_per_marker);
        Self {
            n_haps,
            n_markers: 0,
            words_per_marker,
            allele_bits: Vec::with_capacity(word_cap),
            missing_bits: Vec::with_capacity(word_cap),
        }
    }

    pub fn push_marker_row(&mut self, alleles: &[Option<u8>]) {
        debug_assert_eq!(alleles.len(), self.n_haps);
        let base = self.n_markers * self.words_per_marker;
        self.allele_bits.resize(base + self.words_per_marker, 0);
        self.missing_bits.resize(base + self.words_per_marker, 0);
        for (hap, &allele) in alleles.iter().enumerate() {
            let word = base + (hap >> 6);
            let mask = 1u64 << (hap & 63);
            match allele {
                Some(0) => {}
                Some(_) => self.allele_bits[word] |= mask,
                None => self.missing_bits[word] |= mask,
            }
        }
        self.n_markers += 1;
    }

    #[inline]
    pub fn set_allele(&mut self, marker: usize, hap: usize, allele: Option<u8>) {
        let word = marker * self.words_per_marker + (hap >> 6);
        let mask = 1u64 << (hap & 63);
        match allele {
            Some(0) => {
                self.allele_bits[word] &= !mask;
                self.missing_bits[word] &= !mask;
            }
            Some(_) => {
                self.allele_bits[word] |= mask;
                self.missing_bits[word] &= !mask;
            }
            None => {
                self.allele_bits[word] &= !mask;
                self.missing_bits[word] |= mask;
            }
        }
    }

    #[inline]
    pub fn allele(&self, marker: usize, hap: usize) -> Option<u8> {
        let word = marker * self.words_per_marker + (hap >> 6);
        let shift = hap & 63;
        if ((self.missing_bits[word] >> shift) & 1) != 0 {
            None
        } else {
            Some(((self.allele_bits[word] >> shift) & 1) as u8)
        }
    }

    /// Compare two haplotypes across `[start_marker, end_marker)`, returning
    /// `(comparable_markers, mismatched_markers)`.
    #[inline]
    pub fn count_mismatches(
        &self,
        hap_a: usize,
        hap_b: usize,
        start_marker: usize,
        end_marker: usize,
    ) -> (u32, u32) {
        let word_a = hap_a >> 6;
        let shift_a = hap_a & 63;
        let word_b = hap_b >> 6;
        let shift_b = hap_b & 63;
        let stride = self.words_per_marker;
        let mut comparable = 0u32;
        let mut mismatches = 0u32;
        for marker in start_marker..end_marker {
            let base = marker * stride;
            let miss_a = (self.missing_bits[base + word_a] >> shift_a) & 1;
            let miss_b = (self.missing_bits[base + word_b] >> shift_b) & 1;
            if (miss_a | miss_b) == 0 {
                comparable += 1;
                let a = (self.allele_bits[base + word_a] >> shift_a) & 1;
                let b = (self.allele_bits[base + word_b] >> shift_b) & 1;
                mismatches += (a ^ b) as u32;
            }
        }
        (comparable, mismatches)
    }
}

/// Lightweight PLINK genetic map reader (`chrom id cm bp`) with linear
/// interpolation and constant-rate fallback.
#[derive(Debug, Clone)]
pub struct GeneticMap {
    chrom_points: HashMap<String, Vec<(u64, f64)>>,
    cm_per_mb: f64,
}

impl GeneticMap {
    pub fn constant(cm_per_mb: f64) -> Self {
        Self {
            chrom_points: HashMap::new(),
            cm_per_mb: if cm_per_mb.is_finite() && cm_per_mb > 0.0 {
                cm_per_mb
            } else {
                1.0
            },
        }
    }

    pub fn from_plink_file(path: &Path, cm_per_mb: f64) -> Result<Self, String> {
        let file = File::open(path)
            .map_err(|err| format!("open genetic map {}: {err}", path.display()))?;
        let reader = BufReader::new(file);
        let mut chrom_points: HashMap<String, Vec<(u64, f64)>> = HashMap::new();
        for (line_no, line_res) in reader.lines().enumerate() {
            let line = line_res
                .map_err(|err| format!("read genetic map {} line {}: {err}", path.display(), line_no + 1))?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = trimmed.split_whitespace().collect();
            if cols.len() < 4 {
                continue;
            }
            let Ok(cm) = cols[2].parse::<f64>() else {
                continue;
            };
            let Ok(bp) = cols[3].parse::<u64>() else {
                continue;
            };
            if cm.is_finite() {
                chrom_points
                    .entry(cols[0].to_owned())
                    .or_default()
                    .push((bp, cm));
            }
        }
        for points in chrom_points.values_mut() {
            points.sort_by_key(|&(bp, _)| bp);
            points.dedup_by_key(|(bp, _)| *bp);
        }
        Ok(Self {
            chrom_points,
            cm_per_mb: if cm_per_mb.is_finite() && cm_per_mb > 0.0 {
                cm_per_mb
            } else {
                1.0
            },
        })
    }

    pub fn interpolate_cm(&self, chrom: &str, bp: u64) -> f64 {
        let Some(points) = self.chrom_points.get(chrom) else {
            return (bp as f64 / 1_000_000.0) * self.cm_per_mb;
        };
        if points.is_empty() {
            return (bp as f64 / 1_000_000.0) * self.cm_per_mb;
        }
        match points.binary_search_by_key(&bp, |&(p_bp, _)| p_bp) {
            Ok(idx) => points[idx].1,
            Err(0) => {
                let (bp0, cm0) = points[0];
                (cm0 - ((bp0.saturating_sub(bp)) as f64 / 1_000_000.0) * self.cm_per_mb).max(0.0)
            }
            Err(idx) if idx >= points.len() => {
                let (bp_last, cm_last) = points[points.len() - 1];
                cm_last + ((bp.saturating_sub(bp_last)) as f64 / 1_000_000.0) * self.cm_per_mb
            }
            Err(idx) => {
                let (bp0, cm0) = points[idx - 1];
                let (bp1, cm1) = points[idx];
                if bp1 <= bp0 {
                    cm0
                } else {
                    let frac = (bp - bp0) as f64 / (bp1 - bp0) as f64;
                    cm0 + frac * (cm1 - cm0)
                }
            }
        }
    }
}
