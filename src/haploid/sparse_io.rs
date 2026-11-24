//! Rust-native compressed sparse binary archive (`JHIBD01`) designed for
//! direct zero-loss loading by `src/hgwas`, plus optional `.tsv.gz` export.

use super::ancestry::ChromosomeAncestryBinsSparse;
use super::ibd::SparseIbdKinshipCsr;
use super::types::{AncestrySegment, HaploidIbdMode, IbdSegment};
use flate2::write::GzEncoder;
use flate2::Compression;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub const JHIBD_MAGIC: &[u8; 8] = b"JHIBD01\0";
static TMP_WRITE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Raw chunk size (1 MiB) per independent zstd frame when multi-threaded
/// archive compression is active.
const ZSTD_PARALLEL_FRAME_BYTES: usize = 1 << 20;
/// Number of segments formatted & gzip-compressed per parallel chunk.
const TSV_PARALLEL_CHUNK_ROWS: usize = 32_768;

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

/// Bounded-memory chunked parallel zstd writer that emits standard RFC 8878
/// concatenated zstd frames (transparently decoded by `zstd::stream::Decoder`).
///
/// Memory footprint: buffers up to `2 * threads` raw 1 MiB chunks before
/// dispatching a parallel batch; each raw chunk is dropped immediately after
/// its frame finishes compressing, so peak staging is roughly `2 * threads MiB`
/// for compressible data (up to `~4 * threads MiB` worst-case on incompressible
/// input), plus the 1 MiB `BufWriter` and per-worker zstd compression contexts.
struct ParallelZstdFrameWriter<W: Write> {
    inner: W,
    level: i32,
    batch_cap: usize,
    current: Vec<u8>,
    pending: Vec<Vec<u8>>,
}

impl<W: Write> ParallelZstdFrameWriter<W> {
    fn new(inner: W, level: i32, threads: usize) -> Self {
        let batch_cap = (threads.max(1) * 2).max(2);
        Self {
            inner,
            level,
            batch_cap,
            current: Vec::with_capacity(ZSTD_PARALLEL_FRAME_BYTES),
            pending: Vec::with_capacity(batch_cap),
        }
    }

    fn flush_pending_batch(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let level = self.level;
        let batch = std::mem::replace(&mut self.pending, Vec::with_capacity(self.batch_cap));
        let frames: Vec<io::Result<Vec<u8>>> = batch
            .into_par_iter()
            .map(|raw| {
                let compressed = zstd::bulk::compress(&raw, level);
                drop(raw);
                compressed
            })
            .collect();
        for frame_res in frames {
            let frame = frame_res?;
            self.inner.write_all(&frame)?;
        }
        Ok(())
    }

    fn finish(mut self) -> io::Result<W> {
        if !self.current.is_empty() {
            let tail = std::mem::take(&mut self.current);
            self.pending.push(tail);
        }
        self.flush_pending_batch()?;
        self.inner.flush()?;
        Ok(self.inner)
    }
}

impl<W: Write> Write for ParallelZstdFrameWriter<W> {
    fn write(&mut self, mut buf: &[u8]) -> io::Result<usize> {
        let total = buf.len();
        while !buf.is_empty() {
            let space = ZSTD_PARALLEL_FRAME_BYTES.saturating_sub(self.current.len());
            let take = buf.len().min(space);
            self.current.extend_from_slice(&buf[..take]);
            buf = &buf[take..];
            if self.current.len() >= ZSTD_PARALLEL_FRAME_BYTES {
                let full = std::mem::replace(
                    &mut self.current,
                    Vec::with_capacity(ZSTD_PARALLEL_FRAME_BYTES),
                );
                self.pending.push(full);
                if self.pending.len() >= self.batch_cap {
                    self.flush_pending_batch()?;
                }
            }
        }
        Ok(total)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_pending_batch()?;
        self.inner.flush()
    }
}

/// Write a `HaploidSparseArchive` atomically to `path` via a same-directory
/// temporary file using zstd-compressed bincode with a fixed 8-byte `JHIBD01\0`
/// magic header, so an abrupt signal (`os._exit`) never leaves a truncated archive
/// at `path`.
pub fn write_sparse_archive(path: &Path, archive: &HaploidSparseArchive) -> Result<(), String> {
    write_sparse_archive_with_threads(path, archive, 1)
}

/// Multi-threaded variant of `write_sparse_archive`. When `threads > 1`,
/// compresses 1 MiB chunks into standard concatenated zstd frames in parallel
/// on the active Rayon thread pool.
pub fn write_sparse_archive_with_threads(
    path: &Path,
    archive: &HaploidSparseArchive,
    threads: usize,
) -> Result<(), String> {
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
        let mut writer = BufWriter::with_capacity(1 << 20, file);
        writer
            .write_all(JHIBD_MAGIC)
            .map_err(|err| format!("write magic header to {}: {err}", tmp_path.display()))?;
        if threads > 1 {
            let mut encoder = ParallelZstdFrameWriter::new(writer, 3, threads);
            bincode::serialize_into(&mut encoder, archive).map_err(|err| {
                format!("serialize sparse IBD archive {}: {err}", tmp_path.display())
            })?;
            encoder
                .finish()
                .map_err(|err| format!("finish parallel zstd stream {}: {err}", tmp_path.display()))?;
        } else {
            let mut encoder = zstd::stream::Encoder::new(writer, 3)
                .map_err(|err| format!("init zstd encoder for {}: {err}", tmp_path.display()))?;
            bincode::serialize_into(&mut encoder, archive).map_err(|err| {
                format!("serialize sparse IBD archive {}: {err}", tmp_path.display())
            })?;
            let mut inner = encoder
                .finish()
                .map_err(|err| format!("finish zstd stream {}: {err}", tmp_path.display()))?;
            inner
                .flush()
                .map_err(|err| format!("flush sparse IBD output {}: {err}", tmp_path.display()))?;
        }
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
fn is_cancelled(cancelled: Option<&AtomicBool>) -> bool {
    cancelled
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(false)
}

fn gzip_bytes(raw: &[u8]) -> Result<Vec<u8>, String> {
    let mut gz = GzEncoder::new(Vec::with_capacity(raw.len() / 4 + 64), Compression::fast());
    gz.write_all(raw)
        .map_err(|err| format!("compress gzip TSV chunk: {err}"))?;
    gz.finish()
        .map_err(|err| format!("finish gzip TSV chunk: {err}"))
}

fn format_ibd_chunk(
    chunk: &[IbdSegment],
    archive: &HaploidSparseArchive,
    cancelled: Option<&AtomicBool>,
) -> Result<Vec<u8>, String> {
    let mut text = String::with_capacity(chunk.len() * 88);
    for (idx, seg) in chunk.iter().enumerate() {
        if (idx & 4095) == 0 && is_cancelled(cancelled) {
            return Err("Interrupted".to_owned());
        }
        let s1 = &archive.target_sample_ids[seg.sample1 as usize];
        let s2 = &archive.target_sample_ids[seg.sample2 as usize];
        let chrom = &archive.chrom_names[seg.chrom_idx as usize];
        let _ = match seg.track2 {
            Some((a, b)) => writeln!(
                text,
                "{s1}\t{}\t{s2}\t{}\t{a}\t{b}\t{chrom}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{:.6}",
                seg.hap1,
                seg.hap2,
                seg.start_bp,
                seg.end_bp,
                seg.start_cm,
                seg.end_cm,
                seg.ibd_type,
                seg.n_markers,
                seg.mismatch_rate,
            ),
            None => writeln!(
                text,
                "{s1}\t{}\t{s2}\t{}\t.\t.\t{chrom}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{:.6}",
                seg.hap1,
                seg.hap2,
                seg.start_bp,
                seg.end_bp,
                seg.start_cm,
                seg.end_cm,
                seg.ibd_type,
                seg.n_markers,
                seg.mismatch_rate,
            ),
        };
    }
    gzip_bytes(text.as_bytes())
}

fn format_ancestry_chunk(
    chunk: &[AncestrySegment],
    archive: &HaploidSparseArchive,
    cancelled: Option<&AtomicBool>,
) -> Result<Vec<u8>, String> {
    let mut text = String::with_capacity(chunk.len() * 80);
    for (idx, seg) in chunk.iter().enumerate() {
        if (idx & 4095) == 0 && is_cancelled(cancelled) {
            return Err("Interrupted".to_owned());
        }
        let target = &archive.target_sample_ids[seg.target_sample as usize];
        let parent = &archive.reference_sample_ids[seg.parent_sample as usize];
        let chrom = &archive.chrom_names[seg.chrom_idx as usize];
        let _ = writeln!(
            text,
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
        );
    }
    gzip_bytes(text.as_bytes())
}

/// Cancellable atomic single-member gzipped TSV export checking `cancelled`
/// every 4,096 rows and renaming from a same-directory temp file only upon full
/// completion.
pub fn write_segments_tsv_gz_cancellable(
    path: &Path,
    archive: &HaploidSparseArchive,
    cancelled: Option<&AtomicBool>,
) -> Result<(), String> {
    write_segments_tsv_gz_parallel(path, archive, 1, cancelled)
}

fn write_segments_tsv_gz_single_member(
    file: File,
    archive: &HaploidSparseArchive,
    tmp_path: &Path,
    cancelled: Option<&AtomicBool>,
) -> Result<(), String> {
    let mut gz = GzEncoder::new(BufWriter::with_capacity(1 << 20, file), Compression::fast());
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
                match seg.track2 {
                    Some((a, b)) => writeln!(
                        gz,
                        "{s1}\t{}\t{s2}\t{}\t{a}\t{b}\t{chrom}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{:.6}",
                        seg.hap1,
                        seg.hap2,
                        seg.start_bp,
                        seg.end_bp,
                        seg.start_cm,
                        seg.end_cm,
                        seg.ibd_type,
                        seg.n_markers,
                        seg.mismatch_rate,
                    ),
                    None => writeln!(
                        gz,
                        "{s1}\t{}\t{s2}\t{}\t.\t.\t{chrom}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{:.6}",
                        seg.hap1,
                        seg.hap2,
                        seg.start_bp,
                        seg.end_bp,
                        seg.start_cm,
                        seg.end_cm,
                        seg.ibd_type,
                        seg.n_markers,
                        seg.mismatch_rate,
                    ),
                }
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
}

/// Gzipped TSV export supporting single-threaded single-member output (`threads <= 1`)
/// and multi-threaded chunked RFC 1952 concatenated-member output (`threads > 1`).
///
/// Note on readers: when `threads > 1`, each formatted chunk is compressed as an
/// independent RFC 1952 gzip member (like `pigz` / `bgzip`), so consumers must use
/// a multi-member-aware gzip reader (`gunzip`, `zcat`, Python `gzip`, or
/// `flate2::read::MultiGzDecoder`) rather than single-member raw `zlib.decompress`.
/// When `threads <= 1`, a single standard gzip member is emitted for full
/// compatibility with single-member decoders.
pub fn write_segments_tsv_gz_parallel(
    path: &Path,
    archive: &HaploidSparseArchive,
    threads: usize,
    cancelled: Option<&AtomicBool>,
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
        if threads <= 1 {
            return write_segments_tsv_gz_single_member(file, archive, &tmp_path, cancelled);
        }
        let wave_chunks = (threads * 2).max(2);
        let mut writer = BufWriter::with_capacity(1 << 20, file);
        match &archive.payload {
            HaploidSparsePayload::PairwiseIbd { segments, .. } => {
                let header = b"sample1\thap1\tsample2\thap2\tibd2_hap1\tibd2_hap2\tchrom\tstart_bp\tend_bp\tstart_cm\tend_cm\tibd_type\tn_markers\tmismatch_rate\n";
                let hdr_gz = gzip_bytes(header)?;
                writer
                    .write_all(&hdr_gz)
                    .map_err(|err| format!("write IBD TSV header: {err}"))?;
                let chunks: Vec<&[IbdSegment]> =
                    segments.chunks(TSV_PARALLEL_CHUNK_ROWS).collect();
                for wave in chunks.chunks(wave_chunks) {
                    if is_cancelled(cancelled) {
                        return Err("Interrupted".to_owned());
                    }
                    let compressed_wave: Result<Vec<Vec<u8>>, String> = wave
                        .par_iter()
                        .map(|chunk| format_ibd_chunk(chunk, archive, cancelled))
                        .collect();
                    for block in compressed_wave? {
                        writer
                            .write_all(&block)
                            .map_err(|err| format!("write IBD TSV block: {err}"))?;
                    }
                }
            }
            HaploidSparsePayload::MultiParentAncestry { segments, .. } => {
                let header = b"target\thap\tchrom\tstart_bp\tend_bp\tstart_cm\tend_cm\tparent\tparent_hap\tconfidence\tambiguous_parents\tn_markers\n";
                let hdr_gz = gzip_bytes(header)?;
                writer
                    .write_all(&hdr_gz)
                    .map_err(|err| format!("write ancestry TSV header: {err}"))?;
                let chunks: Vec<&[AncestrySegment]> =
                    segments.chunks(TSV_PARALLEL_CHUNK_ROWS).collect();
                for wave in chunks.chunks(wave_chunks) {
                    if is_cancelled(cancelled) {
                        return Err("Interrupted".to_owned());
                    }
                    let compressed_wave: Result<Vec<Vec<u8>>, String> = wave
                        .par_iter()
                        .map(|chunk| format_ancestry_chunk(chunk, archive, cancelled))
                        .collect();
                    for block in compressed_wave? {
                        writer
                            .write_all(&block)
                            .map_err(|err| format!("write ancestry TSV block: {err}"))?;
                    }
                }
            }
        }
        writer
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
        read_sparse_archive, write_segments_tsv_gz_parallel, write_sparse_archive,
        write_sparse_archive_with_threads, HaploidSparseArchive, HaploidSparsePayload,
    };
    use crate::haploid::ibd::SparseIbdKinshipCsr;
    use crate::haploid::types::{AncestrySegment, HaploidIbdMode, IbdSegment};
    use flate2::read::{GzDecoder, MultiGzDecoder};
    use std::fs::File;
    use std::io::Read;

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

    #[test]
    fn sparse_archive_multithreaded_multi_frame_and_parallel_tsv_round_trip() {
        // Build > 2.5 MiB of bincode payload (~70,000 ancestry segments) so
        // `write_sparse_archive_with_threads(..., 4)` emits multiple concatenated
        // zstd frames and `write_segments_tsv_gz_parallel(..., 4)` emits multiple
        // concatenated gzip members, verifying lossless round-trip and exact order.
        let n_segs = 70_000usize;
        let segments: Vec<AncestrySegment> = (0..n_segs)
            .map(|i| AncestrySegment {
                target_sample: (i % 4) as u32,
                hap: 0,
                chrom_idx: 0,
                start_bp: (i as u64) * 100 + 1,
                end_bp: (i as u64) * 100 + 100,
                start_cm: (i as f32) * 0.001,
                end_cm: ((i + 1) as f32) * 0.001,
                parent_sample: (i % 3) as u32,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 25,
            })
            .collect();
        let archive = HaploidSparseArchive {
            version: 1,
            mode: HaploidIbdMode::MultiParentAncestry,
            target_sample_ids: vec![
                "T0".to_owned(),
                "T1".to_owned(),
                "T2".to_owned(),
                "T3".to_owned(),
            ],
            reference_sample_ids: vec!["P0".to_owned(), "P1".to_owned(), "P2".to_owned()],
            chrom_names: vec!["chr1".to_owned()],
            payload: HaploidSparsePayload::MultiParentAncestry {
                segments,
                chrom_bins: Vec::new(),
            },
        };

        let dir = std::env::temp_dir();
        let jhibd_path = dir.join(format!(
            "janusx_haploid_mt_test_{}.jhibd",
            std::process::id()
        ));
        let tsv_path = dir.join(format!(
            "janusx_haploid_mt_test_{}.jhibd.tsv.gz",
            std::process::id()
        ));

        write_sparse_archive_with_threads(&jhibd_path, &archive, 4).unwrap();
        let loaded = read_sparse_archive(&jhibd_path).unwrap();
        let _ = std::fs::remove_file(&jhibd_path);

        match loaded.payload {
            HaploidSparsePayload::MultiParentAncestry { segments: out, .. } => {
                assert_eq!(out.len(), n_segs);
                assert_eq!(out[0].start_bp, 1);
                assert_eq!(out[n_segs - 1].start_bp, ((n_segs - 1) as u64) * 100 + 1);
            }
            _ => panic!("unexpected payload variant"),
        }

        write_segments_tsv_gz_parallel(&tsv_path, &archive, 4, None).unwrap();
        let mut decoder = MultiGzDecoder::new(File::open(&tsv_path).unwrap());
        let mut text = String::new();
        decoder.read_to_string(&mut text).unwrap();
        let _ = std::fs::remove_file(&tsv_path);

        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), n_segs + 1);
        assert!(lines[0].starts_with("target\thap\tchrom\t"));
        assert!(lines[1].starts_with("T0\t0\tchr1\t1\t100\t"));
        assert!(lines[n_segs].starts_with(&format!(
            "T{}\t0\tchr1\t{}\t{}\t",
            (n_segs - 1) % 4,
            ((n_segs - 1) as u64) * 100 + 1,
            ((n_segs - 1) as u64) * 100 + 100
        )));
    }

    #[test]
    fn tsv_gz_single_thread_writes_single_gzip_member() {
        let segments: Vec<AncestrySegment> = (0..16)
            .map(|i| AncestrySegment {
                target_sample: 0,
                hap: 0,
                chrom_idx: 0,
                start_bp: (i as u64) * 100 + 1,
                end_bp: (i as u64) * 100 + 100,
                start_cm: (i as f32) * 0.01,
                end_cm: ((i + 1) as f32) * 0.01,
                parent_sample: 0,
                parent_hap: 0,
                confidence: 0.99,
                ambiguous_parents: 1,
                n_markers: 20,
            })
            .collect();
        let archive = HaploidSparseArchive {
            version: 1,
            mode: HaploidIbdMode::MultiParentAncestry,
            target_sample_ids: vec!["T0".to_owned()],
            reference_sample_ids: vec!["P0".to_owned()],
            chrom_names: vec!["chr1".to_owned()],
            payload: HaploidSparsePayload::MultiParentAncestry {
                segments,
                chrom_bins: Vec::new(),
            },
        };
        let dir = std::env::temp_dir();
        let tsv_path = dir.join(format!(
            "janusx_haploid_st_test_{}.jhibd.tsv.gz",
            std::process::id()
        ));
        write_segments_tsv_gz_parallel(&tsv_path, &archive, 1, None).unwrap();

        // Single-member `GzDecoder` (analogous to `zlib.decompress(..., wbits=31)`)
        // must read the header AND all 16 data rows (17 lines total).
        let mut decoder = GzDecoder::new(File::open(&tsv_path).unwrap());
        let mut text = String::new();
        decoder.read_to_string(&mut text).unwrap();
        let _ = std::fs::remove_file(&tsv_path);

        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 17);
        assert!(lines[0].starts_with("target\thap\tchrom\t"));
        assert!(lines[16].starts_with("T0\t0\tchr1\t1501\t1600\t"));
    }
}
