//! Compact 1-bit reference panel storage and optional disk spill (P1-1a / P1-1b).
//!
//! Reference panels for phasing and imputation are validated as complete,
//! phased, biallelic genotypes as each record is read. Storing them directly
//! in 1-bit haplotype words halves reference genotype memory compared with
//! 4-bit `DiploidCalls`, eliminates the 4-bit + 1-bit coexistence spike when
//! entering imputation, and allows the reference allele bitplane to spill to a
//! sequential temporary file during phasing (zero reference genotype RAM while
//! PBWT/PhaseBaum workspaces are live).

use super::types::{DiploidCalls, PhasedRow, PhasedWindow, VariantMeta, VariantRow};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_REF_SPILL: AtomicU64 = AtomicU64::new(0);

/// Default auto-spill threshold for P1-1b (16 MiB of 1-bit reference alleles).
const DEFAULT_REF_SPILL_THRESHOLD_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReferenceSpillPolicy {
    Auto(usize),
    Always,
    Never,
}

impl ReferenceSpillPolicy {
    pub(crate) fn from_env() -> Self {
        let Some(raw) = std::env::var("JX_IMP_REF_SPILL").ok() else {
            return Self::Auto(DEFAULT_REF_SPILL_THRESHOLD_BYTES);
        };
        let trimmed = raw.trim().to_ascii_lowercase();
        match trimmed.as_str() {
            "" | "auto" => Self::Auto(DEFAULT_REF_SPILL_THRESHOLD_BYTES),
            "0" | "false" | "no" | "off" => Self::Never,
            "1" | "true" | "yes" | "on" => Self::Always,
            _ => {
                if let Ok(mb) = trimmed.parse::<usize>() {
                    if mb == 0 {
                        Self::Never
                    } else {
                        Self::Auto(mb.saturating_mul(1024 * 1024))
                    }
                } else {
                    Self::Auto(DEFAULT_REF_SPILL_THRESHOLD_BYTES)
                }
            }
        }
    }

    #[inline]
    pub(crate) fn should_spill(self, allele_bytes: usize) -> bool {
        match self {
            Self::Always => allele_bytes > 0,
            Self::Never => false,
            Self::Auto(threshold) => allele_bytes >= threshold,
        }
    }
}

/// One validated complete, phased, biallelic reference marker packed at 1 bit
/// per haplotype (`2 * n_samples` bits).
#[derive(Debug, Clone)]
pub(crate) struct CompactReferenceRow {
    pub(crate) meta: VariantMeta,
    pub(crate) n_samples: usize,
    pub(crate) allele_words: Vec<u64>,
}

impl CompactReferenceRow {
    /// Pack a validated complete, phased, biallelic `VariantRow` into 1-bit
    /// haplotype words and immediately drop its 4-bit `DiploidCalls`.
    pub(crate) fn from_validated_variant_row(row: VariantRow) -> Result<Self, String> {
        let n_samples = row.calls.len();
        let n_haplotypes = n_samples
            .checked_mul(2)
            .ok_or_else(|| "compact reference haplotype count overflow".to_owned())?;
        let words_per_marker = n_haplotypes.div_ceil(64);
        let mut allele_words = vec![0u64; words_per_marker];
        for (word_index, word) in allele_words.iter_mut().enumerate() {
            let Some(packed) = row.calls.biallelic_haplotype_word(word_index, false) else {
                return Err(format!(
                    "reference GT must be complete and biallelic at {}:{}",
                    row.meta.chrom, row.meta.pos
                ));
            };
            *word = packed;
        }
        Ok(Self {
            meta: row.meta,
            n_samples,
            allele_words,
        })
    }

    /// Reconstruct a 4-bit `PhasedRow` for phasing alignment on shared markers.
    pub(crate) fn to_phased_row(&self) -> PhasedRow {
        let calls = diploid_calls_from_haplotype_words(&self.allele_words, self.n_samples);
        PhasedRow {
            meta: self.meta.clone(),
            calls,
            dosage: None,
            ap1: None,
            ap2: None,
            gp: None,
            info_af: None,
            info_dr2: None,
            imputed: false,
        }
    }
}

/// Reconstruct complete, phased, biallelic `DiploidCalls` from 1-bit haplotype
/// words (`2 * n_samples` bits, two bits per sample).
pub(crate) fn diploid_calls_from_haplotype_words(
    allele_words: &[u64],
    n_samples: usize,
) -> DiploidCalls {
    // 2-bit haplotype pair (h0 | (h1 << 1)) -> DiploidCalls 4-bit code:
    // 0b00 -> 0 (0|0), 0b01 -> 3 (1|0), 0b10 -> 2 (0|1), 0b11 -> 4 (1|1).
    const PAIR_TO_CODE: [u8; 4] = [0, 3, 2, 4];

    let mut calls = DiploidCalls::empty(n_samples);
    let mut sample = 0usize;
    while sample + 1 < n_samples {
        let hap0 = sample * 2;
        let word0 = allele_words[hap0 >> 6];
        let shift0 = hap0 & 63;
        let pair0 = ((word0 >> shift0) & 0x3) as usize;
        let hap1 = hap0 + 2;
        let word1 = allele_words[hap1 >> 6];
        let shift1 = hap1 & 63;
        let pair1 = ((word1 >> shift1) & 0x3) as usize;
        calls.set_common_call_pair_unchecked(sample, PAIR_TO_CODE[pair0], PAIR_TO_CODE[pair1]);
        sample += 2;
    }
    if sample < n_samples {
        let hap = sample * 2;
        let pair = ((allele_words[hap >> 6] >> (hap & 63)) & 0x3) as usize;
        calls.set_common_call_code_unchecked(sample, PAIR_TO_CODE[pair]);
    }
    calls
}

/// Sequential temporary file holding a window's 1-bit reference allele words
/// during phasing. Deleted automatically on `Drop`.
#[derive(Debug)]
pub(crate) struct ReferenceAlleleSpill {
    file: Option<File>,
    path: PathBuf,
    n_markers: usize,
    words_per_marker: usize,
}

impl ReferenceAlleleSpill {
    pub(crate) fn create_from_words(
        words: &[u64],
        n_markers: usize,
        words_per_marker: usize,
    ) -> Result<Self, String> {
        let expected_words = n_markers
            .checked_mul(words_per_marker)
            .ok_or_else(|| "reference spill word count overflow".to_owned())?;
        if words.len() != expected_words {
            return Err("reference spill buffer length does not match dimensions".to_owned());
        }
        let root = std::env::var_os("JANUSX_RUNTIME_CACHE_DIR")
            .or_else(|| std::env::var_os("JANUSX_CACHE_DIR"))
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&root)
            .map_err(|e| format!("create IMP reference cache {}: {e}", root.display()))?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| format!("name IMP reference cache: {e}"))?
            .as_nanos();
        let mut created = None;
        for _ in 0..100 {
            let id = NEXT_REF_SPILL.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(
                "jx-imp-ref-{}-{nonce}-{id}.bin",
                std::process::id()
            ));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => {
                    created = Some((file, path));
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => {
                    return Err(format!(
                        "create IMP reference cache {}: {e}",
                        path.display()
                    ));
                }
            }
        }
        let (file, path) =
            created.ok_or_else(|| "cannot allocate a unique IMP reference cache".to_owned())?;
        let mut spill = Self {
            file: Some(file),
            path,
            n_markers,
            words_per_marker,
        };
        {
            let file_ref = spill.file.as_mut().unwrap();
            let mut writer = BufWriter::with_capacity(1024 * 1024, file_ref);
            // SAFETY: initialized native u64 slice copied losslessly to temp file
            // and read back in the same process.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    words.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(words),
                )
            };
            writer
                .write_all(bytes)
                .map_err(|e| format!("write IMP reference cache {}: {e}", spill.path.display()))?;
            writer
                .flush()
                .map_err(|e| format!("flush IMP reference cache {}: {e}", spill.path.display()))?;
        }
        Ok(spill)
    }

    pub(crate) fn open_reader(&mut self) -> Result<BufReader<&mut File>, String> {
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| "IMP reference cache file already closed".to_owned())?;
        file.seek(SeekFrom::Start(0))
            .map_err(|e| format!("seek IMP reference cache {}: {e}", self.path.display()))?;
        Ok(BufReader::with_capacity(1024 * 1024, file))
    }

    #[inline]
    pub(crate) fn n_markers(&self) -> usize {
        self.n_markers
    }

    #[inline]
    pub(crate) fn words_per_marker(&self) -> usize {
        self.words_per_marker
    }

    #[inline]
    pub(crate) fn path_display(&self) -> std::path::Display<'_> {
        self.path.display()
    }
}

impl Drop for ReferenceAlleleSpill {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Backing storage for a window's 1-bit reference haplotype plane.
#[derive(Debug)]
pub(crate) enum ReferenceWindowAlleles {
    Resident(Vec<u64>),
    Spilled(ReferenceAlleleSpill),
}

/// Compact 1-bit representation of one imputation reference window.
#[derive(Debug)]
pub(crate) struct CompactReferenceWindow {
    pub(crate) chrom: String,
    pub(crate) index: u64,
    pub(crate) n_samples: usize,
    pub(crate) words_per_marker: usize,
    #[allow(dead_code)]
    pub(crate) emit_start: usize,
    #[allow(dead_code)]
    pub(crate) emit_end: usize,
    pub(crate) metas: Vec<VariantMeta>,
    pub(crate) cm_positions: Vec<f64>,
    pub(crate) groups: Vec<Option<u64>>,
    pub(crate) alleles: ReferenceWindowAlleles,
}

/// Read-only reference panel handle shared by alignment, rare-carrier
/// classification, IBD2 detection, and reference-assisted phasing.
#[derive(Clone, Copy)]
pub(crate) enum PhaseReference<'a> {
    Phased(&'a PhasedWindow),
    Compact(&'a CompactReferenceWindow),
}

impl<'a> PhaseReference<'a> {
    #[inline]
    pub(crate) fn n_samples(self) -> usize {
        match self {
            Self::Phased(window) => window.rows.first().map_or(0, |row| row.calls.len()),
            Self::Compact(window) => window.n_samples,
        }
    }
}

impl CompactReferenceWindow {
    pub(crate) fn from_compact_rows(
        chrom: String,
        index: u64,
        emit_start: usize,
        emit_end: usize,
        rows: Vec<(CompactReferenceRow, f64, Option<u64>)>,
    ) -> Result<Self, String> {
        let Some((first, _, _)) = rows.first() else {
            return Err(format!("reference window on chromosome {chrom} is empty"));
        };
        let n_samples = first.n_samples;
        let words_per_marker = first.allele_words.len();
        let n_markers = rows.len();
        let total_words = n_markers
            .checked_mul(words_per_marker)
            .ok_or_else(|| "compact reference window size overflow".to_owned())?;
        let mut metas = Vec::with_capacity(n_markers);
        let mut cm_positions = Vec::with_capacity(n_markers);
        let mut groups = Vec::with_capacity(n_markers);
        let mut flat_words = Vec::with_capacity(total_words);
        for (row, cm, group) in rows {
            if row.n_samples != n_samples || row.allele_words.len() != words_per_marker {
                return Err("compact reference rows have inconsistent sample counts".to_owned());
            }
            metas.push(row.meta);
            cm_positions.push(cm);
            groups.push(group);
            flat_words.extend_from_slice(&row.allele_words);
        }
        Ok(Self {
            chrom,
            index,
            n_samples,
            words_per_marker,
            emit_start,
            emit_end,
            metas,
            cm_positions,
            groups,
            alleles: ReferenceWindowAlleles::Resident(flat_words),
        })
    }

    #[inline]
    pub(crate) fn n_markers(&self) -> usize {
        self.metas.len()
    }

    #[inline]
    pub(crate) fn allele_bytes(&self) -> usize {
        self.metas
            .len()
            .saturating_mul(self.words_per_marker)
            .saturating_mul(std::mem::size_of::<u64>())
    }

    #[inline]
    pub(crate) fn is_spilled(&self) -> bool {
        matches!(self.alleles, ReferenceWindowAlleles::Spilled(_))
    }

    /// Borrow the 1-bit haplotype words for one reference marker while the
    /// window is resident in RAM (before P1-1b disk spill).
    #[inline]
    pub(crate) fn resident_marker_words(&self, marker: usize) -> Result<&[u64], String> {
        let ReferenceWindowAlleles::Resident(ref words) = self.alleles else {
            return Err("cannot read resident reference alleles after disk spill".to_owned());
        };
        let start = marker
            .checked_mul(self.words_per_marker)
            .ok_or_else(|| "compact reference marker offset overflow".to_owned())?;
        let end = start
            .checked_add(self.words_per_marker)
            .ok_or_else(|| "compact reference marker offset overflow".to_owned())?;
        words.get(start..end).ok_or_else(|| {
            format!("compact reference marker index {marker} is out of bounds")
        })
    }

    /// Spill the resident 1-bit allele bitplane to a sequential temporary file
    /// and release its heap allocation immediately.
    pub(crate) fn spill_to_disk(&mut self) -> Result<(), String> {
        let ReferenceWindowAlleles::Resident(ref words) = self.alleles else {
            return Ok(());
        };
        let spill = ReferenceAlleleSpill::create_from_words(
            words,
            self.metas.len(),
            self.words_per_marker,
        )?;
        self.alleles = ReferenceWindowAlleles::Spilled(spill);
        Ok(())
    }

    /// Spill to disk if permitted by `JX_IMP_REF_SPILL` / default threshold.
    pub(crate) fn maybe_spill_to_disk(&mut self) -> Result<bool, String> {
        let policy = ReferenceSpillPolicy::from_env();
        if policy.should_spill(self.allele_bytes()) {
            self.spill_to_disk()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Read exact `u64` words from a buffered reader into `dest`.
pub(crate) fn read_exact_u64_words<R: Read>(
    reader: &mut R,
    dest: &mut [u64],
    label: &str,
) -> Result<(), String> {
    // SAFETY: all u64 bit patterns are valid; bytes come from native u64 writes
    // in the same process.
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(dest))
    };
    reader
        .read_exact(bytes)
        .map_err(|e| format!("read IMP reference cache {label}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imp::types::DiploidCall;

    fn sample_meta(pos: u64) -> VariantMeta {
        VariantMeta {
            chrom: "1".to_owned(),
            pos,
            id: format!("v{pos}"),
            ref_allele: "A".to_owned(),
            alt_alleles: vec!["G".to_owned()],
        }
    }

    #[test]
    fn compact_row_round_trips_all_complete_phased_biallelic_genotypes() {
        for n_samples in [1usize, 2, 3, 31, 32, 33, 64, 65, 131] {
            let calls = DiploidCalls::from_calls(
                (0..n_samples)
                    .map(|sample| match sample % 4 {
                        0 => DiploidCall {
                            alleles: [Some(0), Some(0)],
                            phased: false,
                        },
                        1 => DiploidCall {
                            alleles: [Some(0), Some(1)],
                            phased: true,
                        },
                        2 => DiploidCall {
                            alleles: [Some(1), Some(0)],
                            phased: true,
                        },
                        _ => DiploidCall {
                            alleles: [Some(1), Some(1)],
                            phased: false,
                        },
                    })
                    .collect(),
            );
            let row = VariantRow {
                meta: sample_meta(100),
                calls: calls.clone(),
            };
            let compact = CompactReferenceRow::from_validated_variant_row(row).unwrap();
            assert_eq!(compact.allele_words.len(), (n_samples * 2).div_ceil(64));
            let restored = compact.to_phased_row();
            assert_eq!(restored.calls, calls);
        }
    }

    #[test]
    fn shared_phase_window_and_disk_spill_preserve_exact_words() {
        let n_samples = 35;
        let make_row = |pos: u64| {
            let calls = DiploidCalls::from_calls(
                (0..n_samples)
                    .map(|s| DiploidCall {
                        alleles: [
                            Some(((pos as usize + s) & 1) as u16),
                            Some(((pos as usize * 3 + s) & 1) as u16),
                        ],
                        phased: true,
                    })
                    .collect(),
            );
            CompactReferenceRow::from_validated_variant_row(VariantRow {
                meta: sample_meta(pos),
                calls,
            })
            .unwrap()
        };
        let rows = vec![
            (make_row(100), 0.1, Some(1)),
            (make_row(200), 0.2, Some(1)),
            (make_row(300), 0.3, Some(2)),
        ];
        let expected_words = rows
            .iter()
            .flat_map(|(r, _, _)| r.allele_words.iter().copied())
            .collect::<Vec<_>>();
        let mut window =
            CompactReferenceWindow::from_compact_rows("1".to_owned(), 0, 0, 3, rows).unwrap();

        let mid_words = window.resident_marker_words(1).unwrap();
        assert_eq!(mid_words, &expected_words[window.words_per_marker..2 * window.words_per_marker]);
        assert!(ReferenceSpillPolicy::Auto(DEFAULT_REF_SPILL_THRESHOLD_BYTES).should_spill(16 * 1024 * 1024));
        assert!(!ReferenceSpillPolicy::Auto(DEFAULT_REF_SPILL_THRESHOLD_BYTES).should_spill(16 * 1024 * 1024 - 1));

        window.spill_to_disk().unwrap();
        assert!(window.is_spilled());
        assert!(window.resident_marker_words(0).is_err());
        let ReferenceWindowAlleles::Spilled(ref mut spill) = window.alleles else {
            panic!("expected spilled storage");
        };
        let label = spill.path_display().to_string();
        let mut reader = spill.open_reader().unwrap();
        let mut restored = vec![0u64; expected_words.len()];
        read_exact_u64_words(&mut reader, &mut restored, &label).unwrap();
        assert_eq!(restored, expected_words);
    }
}
