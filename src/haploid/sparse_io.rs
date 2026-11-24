//! Rust-native compressed sparse binary archive (`JHIBD01`) designed for
//! direct zero-loss loading by `src/hgwas`, plus optional `.tsv.gz` export.

use super::ancestry::ChromosomeAncestryBinsSparse;
use super::ibd::SparseIbdKinshipCsr;
use super::types::{AncestrySegment, HaploidIbdMode, IbdSegment};
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const JHIBD_MAGIC: &[u8; 8] = b"JHIBD01\0";
static TMP_WRITE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn sibling_temp_path(target: &Path) -> PathBuf {
    let id = TMP_WRITE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = target
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("haploid_out");
    let tmp_name = format!(".{file_name}.tmp.{}.{id}", std::process::id());
    target.with_file_name(tmp_name)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HaploidSparsePayload {
    PairwiseIbd {
        segments: Vec<IbdSegment>,
        kinship_csr: SparseIbdKinshipCsr,
    },
    MultiParentAncestry {
        segments: Vec<AncestrySegment>,
        chrom_bins: Vec<ChromosomeAncestryBinsSparse>,
    },
}

/// Top-level Rust-native sparse archive written by `--only-ibd` and consumed by
/// `src/hgwas`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HaploidSparseArchive {
    pub version: u32,
    pub mode: HaploidIbdMode,
    pub target_sample_ids: Vec<String>,
    pub reference_sample_ids: Vec<String>,
    pub chrom_names: Vec<String>,
    pub payload: HaploidSparsePayload,
}

/// Write a `HaploidSparseArchive` atomically to `path` via a same-directory
/// temporary file using zstd-compressed bincode with a fixed 8-byte `JHIBD01\0`
/// magic header, so an abrupt signal (`os._exit`) never leaves a truncated archive
/// at `path`.
pub fn write_sparse_archive(path: &Path, archive: &HaploidSparseArchive) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|err| {
                format!("create output directory {}: {err}", parent.display())
            })?;
        }
    }
    let tmp_path = sibling_temp_path(path);
    let write_res = (|| -> Result<(), String> {
        let file = File::create(&tmp_path)
            .map_err(|err| format!("create sparse IBD temp output {}: {err}", tmp_path.display()))?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(JHIBD_MAGIC)
            .map_err(|err| format!("write magic header to {}: {err}", tmp_path.display()))?;
        let mut encoder = zstd::stream::Encoder::new(writer, 3)
            .map_err(|err| format!("init zstd encoder for {}: {err}", tmp_path.display()))?;
        bincode::serialize_into(&mut encoder, archive)
            .map_err(|err| format!("serialize sparse IBD archive {}: {err}", tmp_path.display()))?;
        let mut inner = encoder
            .finish()
            .map_err(|err| format!("finish zstd stream {}: {err}", tmp_path.display()))?;
        inner
            .flush()
            .map_err(|err| format!("flush sparse IBD output {}: {err}", tmp_path.display()))?;
        Ok(())
    })();
    if let Err(err) = write_res {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err);
    }
    std::fs::rename(&tmp_path, path).map_err(|err| {
        let _ = std::fs::remove_file(&tmp_path);
        format!(
            "finalize sparse IBD archive {} -> {}: {err}",
            tmp_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

/// Load a `HaploidSparseArchive` from `path` (primary entry point for `src/hgwas`).
pub fn read_sparse_archive(path: &Path) -> Result<HaploidSparseArchive, String> {
    let file = File::open(path)
        .map_err(|err| format!("open sparse IBD archive {}: {err}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|err| format!("read magic header from {}: {err}", path.display()))?;
    if &magic != JHIBD_MAGIC {
        return Err(format!(
            "invalid JHIBD magic header in {}: expected {:?}, got {:?}",
            path.display(),
            JHIBD_MAGIC,
            magic
        ));
    }
    let decoder = zstd::stream::Decoder::new(reader)
        .map_err(|err| format!("init zstd decoder for {}: {err}", path.display()))?;
    bincode::deserialize_from(decoder)
        .map_err(|err| format!("deserialize sparse IBD archive {}: {err}", path.display()))
}

/// Export segments in human-readable gzipped TSV alongside the binary archive
/// when `write_tsv = true`.
pub fn write_segments_tsv_gz(path: &Path, archive: &HaploidSparseArchive) -> Result<(), String> {
    write_segments_tsv_gz_cancellable(path, archive, None)
}

#[inline]
fn is_cancelled(cancelled: Option<&std::sync::atomic::AtomicBool>) -> bool {
    cancelled
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(false)
}

/// Cancellable atomic gzipped TSV export checking `cancelled` every 4,096 rows
/// and renaming from a same-directory temp file only upon full completion.
pub fn write_segments_tsv_gz_cancellable(
    path: &Path,
    archive: &HaploidSparseArchive,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|err| {
                format!("create TSV output directory {}: {err}", parent.display())
            })?;
        }
    }
    let tmp_path = sibling_temp_path(path);
    let write_res = (|| -> Result<(), String> {
        let file = File::create(&tmp_path)
            .map_err(|err| format!("create TSV temp output {}: {err}", tmp_path.display()))?;
        let mut gz = GzEncoder::new(BufWriter::new(file), Compression::fast());
        match &archive.payload {
            HaploidSparsePayload::PairwiseIbd { segments, .. } => {
                writeln!(
                    gz,
                    "sample1\thap1\tsample2\thap2\tibd2_hap1\tibd2_hap2\tchrom\tstart_bp\tend_bp\tstart_cm\tend_cm\tibd_type\tn_markers\tmismatch_rate"
                )
                .map_err(|err| format!("write IBD TSV header: {err}"))?;
                for (idx, seg) in segments.iter().enumerate() {
                    if (idx & 4095) == 0 && is_cancelled(cancelled) {
                        return Err("Interrupted".to_owned());
                    }
                    let s1 = &archive.target_sample_ids[seg.sample1 as usize];
                    let s2 = &archive.target_sample_ids[seg.sample2 as usize];
                    let chrom = &archive.chrom_names[seg.chrom_idx as usize];
                    let (t2_h1, t2_h2) = match seg.track2 {
                        Some((a, b)) => (a.to_string(), b.to_string()),
                        None => (".".to_owned(), ".".to_owned()),
                    };
                    writeln!(
                        gz,
                        "{s1}\t{}\t{s2}\t{}\t{t2_h1}\t{t2_h2}\t{chrom}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{:.6}",
                        seg.hap1,
                        seg.hap2,
                        seg.start_bp,
                        seg.end_bp,
                        seg.start_cm,
                        seg.end_cm,
                        seg.ibd_type,
                        seg.n_markers,
                        seg.mismatch_rate,
                    )
                    .map_err(|err| format!("write IBD TSV row: {err}"))?;
                }
            }
            HaploidSparsePayload::MultiParentAncestry { segments, .. } => {
                writeln!(
                    gz,
                    "target\thap\tchrom\tstart_bp\tend_bp\tstart_cm\tend_cm\tparent\tparent_hap\tconfidence\tambiguous_parents\tn_markers"
                )
                .map_err(|err| format!("write ancestry TSV header: {err}"))?;
                for (idx, seg) in segments.iter().enumerate() {
                    if (idx & 4095) == 0 && is_cancelled(cancelled) {
                        return Err("Interrupted".to_owned());
                    }
                    let target = &archive.target_sample_ids[seg.target_sample as usize];
                    let parent = &archive.reference_sample_ids[seg.parent_sample as usize];
                    let chrom = &archive.chrom_names[seg.chrom_idx as usize];
                    writeln!(
                        gz,
                        "{target}\t{}\t{chrom}\t{}\t{}\t{:.4}\t{:.4}\t{parent}\t{}\t{:.4}\t{}\t{}",
                        seg.hap,
                        seg.start_bp,
                        seg.end_bp,
                        seg.start_cm,
                        seg.end_cm,
                        seg.parent_hap,
                        seg.confidence,
                        seg.ambiguous_parents,
                        seg.n_markers,
                    )
                    .map_err(|err| format!("write ancestry TSV row: {err}"))?;
                }
            }
        }
        let mut inner = gz
            .finish()
            .map_err(|err| format!("finish gzip TSV stream {}: {err}", tmp_path.display()))?;
        inner
            .flush()
            .map_err(|err| format!("flush gzip TSV stream {}: {err}", tmp_path.display()))?;
        Ok(())
    })();
    if let Err(err) = write_res {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err);
    }
    std::fs::rename(&tmp_path, path).map_err(|err| {
        let _ = std::fs::remove_file(&tmp_path);
        format!(
            "finalize TSV output {} -> {}: {err}",
            tmp_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        read_sparse_archive, write_sparse_archive, HaploidSparseArchive, HaploidSparsePayload,
    };
    use crate::haploid::ibd::SparseIbdKinshipCsr;
    use crate::haploid::types::{HaploidIbdMode, IbdSegment};

    #[test]
    fn sparse_archive_round_trips_losslessly() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("janusx_haploid_test_{}.jhibd", std::process::id()));
        let archive = HaploidSparseArchive {
            version: 1,
            mode: HaploidIbdMode::PairwiseIbd,
            target_sample_ids: vec!["L1".to_owned(), "L2".to_owned()],
            reference_sample_ids: Vec::new(),
            chrom_names: vec!["chr1".to_owned()],
            payload: HaploidSparsePayload::PairwiseIbd {
                segments: vec![IbdSegment {
                    sample1: 0,
                    hap1: 0,
                    sample2: 1,
                    hap2: 0,
                    track2: None,
                    chrom_idx: 0,
                    start_bp: 1000,
                    end_bp: 500_000,
                    start_cm: 0.1,
                    end_cm: 5.0,
                    ibd_type: 2,
                    n_markers: 250,
                    mismatch_rate: 0.002,
                }],
                kinship_csr: SparseIbdKinshipCsr {
                    n_samples: 2,
                    total_genome_cm: 10.0,
                    row_ptr: vec![0, 1, 1],
                    col_idx: vec![1],
                    ibd1_frac: vec![0.0],
                    ibd2_frac: vec![0.49],
                    kinship: vec![0.245],
                },
            },
        };
        write_sparse_archive(&path, &archive).unwrap();
        let loaded = read_sparse_archive(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.mode, HaploidIbdMode::PairwiseIbd);
        assert_eq!(loaded.target_sample_ids, archive.target_sample_ids);
        match loaded.payload {
            HaploidSparsePayload::PairwiseIbd {
                segments,
                kinship_csr,
            } => {
                assert_eq!(segments.len(), 1);
                assert_eq!(segments[0].end_bp, 500_000);
                assert_eq!(kinship_csr.col_idx, vec![1]);
            }
            _ => panic!("unexpected payload variant"),
        }
    }
}
