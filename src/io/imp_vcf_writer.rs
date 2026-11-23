//! Streaming phased VCF writer with standards-compatible BGZF blocks.

use crate::imp::types::{DiploidCall, PhasedRow, VcfOutputFields};
use flate2::write::DeflateEncoder;
use flate2::{Compression, Crc};
use rayon::prelude::*;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::OnceLock;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const BGZF_UNCOMPRESSED_BLOCK: usize = 60_000;
const BGZF_QUEUED_BLOCKS_PER_THREAD: usize = 2;
const BGZF_MIN_QUEUE_DEPTH: usize = 4;
const BGZF_MAX_QUEUE_DEPTH: usize = 64;
const BGZF_MAX_COMPRESSION_THREADS: usize = 32;
const BGZF_THREADS_ENV: &str = "JX_IMP_BGZF_THREADS";
const VCF_PROFILE_ENV: &str = "JANUSX_IMP_VCF_PROFILE";
const PARALLEL_ROW_CHUNK: usize = 32;
const PARALLEL_ROWS_PER_GROUP: usize = 4_096;
const BGZF_EOF_BLOCK: [u8; 28] = [
    0x1f, 0x8b, 0x08, 0x04, 0, 0, 0, 0, 0, 0xff, 0x06, 0, 0x42, 0x43, 0x02, 0, 0x1b, 0, 0x03, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
];

enum VcfStream {
    Plain(BufWriter<File>),
    Bgzip(BgzfWriter),
}

pub(crate) struct ImpVcfWriter {
    path: PathBuf,
    stream: VcfStream,
    samples: Vec<String>,
    fields: VcfOutputFields,
    row_buffer: String,
    rows_written: u64,
    profile_enabled: bool,
    profile: VcfWriteProfile,
}

struct BgzfWriter {
    path: PathBuf,
    sender: Option<SyncSender<Vec<u8>>>,
    handle: Option<JoinHandle<Result<(), String>>>,
    buffer: Vec<u8>,
    profile_enabled: bool,
    profile_submit_elapsed: Duration,
}

struct CompressedBgzfBlock {
    raw_len: usize,
    encoded: Vec<u8>,
}

#[derive(Default)]
struct VcfWriteProfile {
    format_elapsed: Duration,
    format_bytes: u64,
    submit_elapsed: Duration,
    submit_bytes: u64,
}

#[derive(Default)]
struct BgzfSinkProfile {
    blocks: u64,
    raw_bytes: u64,
    encoded_bytes: u64,
    compress_elapsed: Duration,
    output_elapsed: Duration,
}

impl BgzfWriter {
    fn create(path: &Path, compression_threads: usize) -> Result<Self, String> {
        let path = path.to_path_buf();
        let output = BufWriter::new(
            File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?,
        );
        let compression_threads = compression_threads.clamp(1, BGZF_MAX_COMPRESSION_THREADS);
        let queue_depth = compression_threads
            .saturating_mul(BGZF_QUEUED_BLOCKS_PER_THREAD)
            .clamp(BGZF_MIN_QUEUE_DEPTH, BGZF_MAX_QUEUE_DEPTH);
        let (sender, receiver) = sync_channel::<Vec<u8>>(queue_depth);
        let writer_path = path.clone();
        let profile_enabled = std::env::var_os(VCF_PROFILE_ENV).is_some();
        let handle = thread::Builder::new()
            .name("imp-bgzf-writer".to_owned())
            .spawn(move || {
                run_bgzf_writer(
                    writer_path,
                    output,
                    receiver,
                    compression_threads,
                    queue_depth,
                    profile_enabled,
                )
            })
            .map_err(|err| format!("spawn BGZF writer for {}: {err}", path.display()))?;
        Ok(Self {
            path,
            sender: Some(sender),
            handle: Some(handle),
            buffer: Vec::with_capacity(BGZF_UNCOMPRESSED_BLOCK),
            profile_enabled,
            profile_submit_elapsed: Duration::ZERO,
        })
    }

    fn write_bytes(
        &mut self,
        mut bytes: &[u8],
        _worker_pool: Option<&rayon::ThreadPool>,
    ) -> Result<(), String> {
        while !bytes.is_empty() {
            let available = BGZF_UNCOMPRESSED_BLOCK - self.buffer.len();
            let take = available.min(bytes.len());
            self.buffer.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.buffer.len() == BGZF_UNCOMPRESSED_BLOCK {
                self.send_full_block()?;
            }
        }
        Ok(())
    }

    fn send_full_block(&mut self) -> Result<(), String> {
        let full_block = std::mem::replace(
            &mut self.buffer,
            Vec::with_capacity(BGZF_UNCOMPRESSED_BLOCK),
        );
        let started = self.profile_enabled.then(Instant::now);
        let result = self
            .sender
            .as_ref()
            .ok_or_else(|| format!("BGZF writer already closed for {}", self.path.display()))?
            .send(full_block)
            .map_err(|_| format!("BGZF writer closed while writing {}", self.path.display()));
        if let Some(started) = started {
            self.profile_submit_elapsed += started.elapsed();
        }
        result
    }

    fn finish(mut self, _worker_pool: Option<&rayon::ThreadPool>) -> Result<(), String> {
        if !self.buffer.is_empty() {
            let partial_block = std::mem::take(&mut self.buffer);
            self.sender
                .as_ref()
                .ok_or_else(|| format!("BGZF writer already closed for {}", self.path.display()))?
                .send(partial_block)
                .map_err(|_| {
                    format!("BGZF writer closed while finishing {}", self.path.display())
                })?;
        }
        self.sender.take();
        let result = join_bgzf_writer(&self.path, self.handle.take());
        if self.profile_enabled {
            eprintln!(
                "JANUSX_IMP_VCF_BGZF_QUEUE_PROFILE queue_submit_s={:.6}",
                self.profile_submit_elapsed.as_secs_f64(),
            );
        }
        result
    }
}

fn run_bgzf_writer(
    path: PathBuf,
    mut output: BufWriter<File>,
    receiver: std::sync::mpsc::Receiver<Vec<u8>>,
    compression_threads: usize,
    batch_blocks: usize,
    profile_enabled: bool,
) -> Result<(), String> {
    let started = profile_enabled.then(Instant::now);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(compression_threads)
        .thread_name(|index| format!("imp-bgzf-compress-{index}"))
        .build()
        .map_err(|err| format!("create BGZF compression pool: {err}"))?;
    let mut pending_blocks = Vec::with_capacity(batch_blocks);
    let mut profile = BgzfSinkProfile::default();
    for raw in receiver {
        pending_blocks.push(raw);
        if pending_blocks.len() >= batch_blocks {
            flush_bgzf_batch(&path, &mut output, &mut pending_blocks, &pool, &mut profile)?;
        }
    }
    flush_bgzf_batch(&path, &mut output, &mut pending_blocks, &pool, &mut profile)?;
    output
        .write_all(&BGZF_EOF_BLOCK)
        .and_then(|_| output.flush())
        .map_err(|err| format!("finish BGZF output {}: {err}", path.display()))?;
    if let Some(started) = started {
        eprintln!(
            "JANUSX_IMP_VCF_BGZF_SINK_PROFILE blocks={} raw_bytes={} encoded_bytes={} compress_s={:.6} output_s={:.6} span_s={:.6}",
            profile.blocks,
            profile.raw_bytes,
            profile.encoded_bytes,
            profile.compress_elapsed.as_secs_f64(),
            profile.output_elapsed.as_secs_f64(),
            started.elapsed().as_secs_f64(),
        );
    }
    Ok(())
}

fn flush_bgzf_batch(
    path: &Path,
    output: &mut BufWriter<File>,
    pending_blocks: &mut Vec<Vec<u8>>,
    pool: &rayon::ThreadPool,
    profile: &mut BgzfSinkProfile,
) -> Result<(), String> {
    if pending_blocks.is_empty() {
        return Ok(());
    }
    let started = Instant::now();
    let compressed = pool.install(|| {
        pending_blocks
            .par_drain(..)
            .map(compress_bgzf_block)
            .collect::<Result<Vec<_>, _>>()
    })?;
    profile.compress_elapsed += started.elapsed();
    let started = Instant::now();
    for block in compressed {
        output
            .write_all(&block.encoded)
            .map_err(|err| format!("write BGZF output {}: {err}", path.display()))?;
        profile.blocks = profile.blocks.saturating_add(1);
        profile.raw_bytes = profile.raw_bytes.saturating_add(block.raw_len as u64);
        profile.encoded_bytes = profile
            .encoded_bytes
            .saturating_add(block.encoded.len() as u64);
    }
    profile.output_elapsed += started.elapsed();
    Ok(())
}

fn join_bgzf_writer(
    path: &Path,
    handle: Option<JoinHandle<Result<(), String>>>,
) -> Result<(), String> {
    match handle {
        Some(handle) => handle
            .join()
            .map_err(|_| format!("BGZF writer thread panicked for {}", path.display()))?,
        None => Ok(()),
    }
}

impl Drop for BgzfWriter {
    fn drop(&mut self) {
        self.sender.take();
        let _ = join_bgzf_writer(&self.path, self.handle.take());
    }
}

fn compress_bgzf_block(raw: Vec<u8>) -> Result<CompressedBgzfBlock, String> {
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(&raw)
        .map_err(|e| format!("compress BGZF block: {e}"))?;
    let compressed = encoder
        .finish()
        .map_err(|e| format!("finish BGZF block: {e}"))?;
    let block_size = 18usize
        .checked_add(compressed.len())
        .and_then(|value| value.checked_add(8))
        .ok_or_else(|| "BGZF compressed block size overflow".to_owned())?;
    if block_size > 65_536 {
        return Err(format!(
            "BGZF block compressed to {block_size} bytes; internal block limit was exceeded"
        ));
    }
    let bsize = (block_size - 1) as u16;
    let mut header = [
        0x1f, 0x8b, 0x08, 0x04, 0, 0, 0, 0, 0, 0xff, 0x06, 0, 0x42, 0x43, 0x02, 0, 0, 0,
    ];
    header[16..18].copy_from_slice(&bsize.to_le_bytes());
    let mut crc = Crc::new();
    crc.update(&raw);
    let mut encoded = Vec::with_capacity(block_size);
    encoded.extend_from_slice(&header);
    encoded.extend_from_slice(&compressed);
    encoded.extend_from_slice(&crc.sum().to_le_bytes());
    encoded.extend_from_slice(&(raw.len() as u32).to_le_bytes());
    Ok(CompressedBgzfBlock {
        raw_len: raw.len(),
        encoded,
    })
}

impl ImpVcfWriter {
    #[cfg(test)]
    pub(crate) fn create(
        path: &Path,
        samples: &[String],
        fields: VcfOutputFields,
    ) -> Result<Self, String> {
        Self::create_with_threads(path, samples, fields, 1)
    }

    pub(crate) fn create_with_threads(
        path: &Path,
        samples: &[String],
        fields: VcfOutputFields,
        compression_threads: usize,
    ) -> Result<Self, String> {
        if samples.is_empty() {
            return Err("VCF output requires at least one sample".to_owned());
        }
        let compression_override = std::env::var(BGZF_THREADS_ENV).ok();
        let compression_threads =
            resolve_bgzf_threads(compression_threads, compression_override.as_deref());
        let is_bgzf = path
            .to_string_lossy()
            .to_ascii_lowercase()
            .ends_with(".vcf.gz");
        let stream = if is_bgzf {
            VcfStream::Bgzip(BgzfWriter::create(path, compression_threads)?)
        } else {
            VcfStream::Plain(BufWriter::new(
                File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?,
            ))
        };
        let row_capacity = samples.len().saturating_mul(24).saturating_add(256);
        let mut out = Self {
            path: path.to_path_buf(),
            stream,
            samples: samples.to_vec(),
            fields,
            row_buffer: String::with_capacity(row_capacity),
            rows_written: 0,
            profile_enabled: std::env::var_os(VCF_PROFILE_ENV).is_some(),
            profile: VcfWriteProfile::default(),
        };
        out.write_header()?;
        Ok(out)
    }

    #[cfg(test)]
    pub(crate) fn write_row(
        &mut self,
        row: &PhasedRow,
        worker_pool: &rayon::ThreadPool,
    ) -> Result<(), String> {
        let mut buffer = std::mem::take(&mut self.row_buffer);
        buffer.clear();
        let format_started = self.profile_enabled.then(Instant::now);
        if let Err(error) = self.format_row(row, &mut buffer) {
            self.row_buffer = buffer;
            return Err(error);
        }
        if let Some(started) = format_started {
            self.profile.format_elapsed += started.elapsed();
            self.profile.format_bytes = self
                .profile
                .format_bytes
                .saturating_add(buffer.len() as u64);
        }
        let submit_started = self.profile_enabled.then(Instant::now);
        let result = self.write_encoded(buffer.as_bytes(), worker_pool);
        if let Some(started) = submit_started {
            self.profile.submit_elapsed += started.elapsed();
            self.profile.submit_bytes = self
                .profile
                .submit_bytes
                .saturating_add(buffer.len() as u64);
        }
        self.row_buffer = buffer;
        result?;
        self.rows_written = self.rows_written.saturating_add(1);
        Ok(())
    }

    fn write_header(&mut self) -> Result<(), String> {
        self.write_text("##fileformat=VCFv4.3\n")?;
        self.write_text("##source=JanusX-jx-imp\n")?;
        self.write_text("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Phased genotype\">\n")?;
        if self.fields.dosage {
            self.write_text("##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"Imputed alternate allele dosage\">\n")?;
            self.write_text("##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated alternate allele frequency\">\n")?;
            self.write_text(
                "##INFO=<ID=DR2,Number=A,Type=Float,Description=\"Estimated dosage r-squared\">\n",
            )?;
            self.write_text("##INFO=<ID=IMP,Number=0,Type=Flag,Description=\"Variant absent from the target input\">\n")?;
        }
        if self.fields.allele_probabilities {
            self.write_text("##FORMAT=<ID=AP1,Number=A,Type=Float,Description=\"Haplotype 1 allele probability\">\n")?;
            self.write_text("##FORMAT=<ID=AP2,Number=A,Type=Float,Description=\"Haplotype 2 allele probability\">\n")?;
        }
        if self.fields.genotype_probabilities {
            self.write_text(
                "##FORMAT=<ID=GP,Number=G,Type=Float,Description=\"Genotype probabilities\">\n",
            )?;
        }
        self.write_text("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
        let samples = self.samples.clone();
        for sample in samples {
            self.write_text("\t")?;
            self.write_text(&sample)?;
        }
        self.write_text("\n")
    }

    pub(crate) fn write_rows(
        &mut self,
        rows: &[&PhasedRow],
        worker_pool: &rayon::ThreadPool,
    ) -> Result<(), String> {
        if rows.is_empty() {
            return Ok(());
        }
        let row_capacity = self.row_buffer.capacity();
        for row_group in rows.chunks(PARALLEL_ROWS_PER_GROUP) {
            let format_started = self.profile_enabled.then(Instant::now);
            let encoded_chunks = worker_pool.install(|| {
                row_group
                    .par_chunks(PARALLEL_ROW_CHUNK)
                    .map(|chunk| {
                        let mut encoded =
                            String::with_capacity(row_capacity.saturating_mul(chunk.len()));
                        for row in chunk {
                            self.format_row(row, &mut encoded)?;
                        }
                        Ok(encoded.into_bytes())
                    })
                    .collect::<Result<Vec<_>, String>>()
            })?;
            if let Some(started) = format_started {
                self.profile.format_elapsed += started.elapsed();
                self.profile.format_bytes = self
                    .profile
                    .format_bytes
                    .saturating_add(encoded_chunks.iter().map(Vec::len).sum::<usize>() as u64);
            }
            let submit_started = self.profile_enabled.then(Instant::now);
            for encoded in encoded_chunks {
                self.write_encoded(&encoded, worker_pool)?;
                if self.profile_enabled {
                    self.profile.submit_bytes = self
                        .profile
                        .submit_bytes
                        .saturating_add(encoded.len() as u64);
                }
            }
            if let Some(started) = submit_started {
                self.profile.submit_elapsed += started.elapsed();
            }
        }
        self.rows_written = self.rows_written.saturating_add(rows.len() as u64);
        Ok(())
    }

    fn format_row(&self, row: &PhasedRow, output: &mut String) -> Result<(), String> {
        if row.calls.len() != self.samples.len() {
            return Err(format!(
                "{}: variant {} has {} calls but output has {} samples",
                self.path.display(),
                row.meta.id,
                row.calls.len(),
                self.samples.len()
            ));
        }
        if row.meta.ref_allele.contains(',')
            || row.meta.alt_alleles.iter().any(|alt| alt.contains(','))
        {
            // ALT is already a list; commas in an allele would make the VCF
            // record ambiguous and are not emitted by the supported readers.
            return Err(format!(
                "{}:{} has an allele containing a comma",
                row.meta.chrom, row.meta.pos
            ));
        }
        output.push_str(&row.meta.chrom);
        output.push('\t');
        write!(output, "{}\t", row.meta.pos).expect("formatting into a String cannot fail");
        output.push_str(if row.meta.id.is_empty() {
            "."
        } else {
            &row.meta.id
        });
        output.push('\t');
        output.push_str(&row.meta.ref_allele);
        output.push('\t');
        if row.meta.alt_alleles.is_empty() {
            output.push('.');
        } else {
            for (index, alt) in row.meta.alt_alleles.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(alt);
            }
        }
        output.push_str("\t.\tPASS\t");
        let mut has_info = false;
        if let Some(af) = row.info_af {
            write!(output, "AF={af:.4}").expect("formatting into a String cannot fail");
            has_info = true;
        }
        if let Some(dr2) = row.info_dr2 {
            if has_info {
                output.push(';');
            }
            write!(output, "DR2={dr2:.2}").expect("formatting into a String cannot fail");
            has_info = true;
        }
        if row.imputed {
            if has_info {
                output.push(';');
            }
            output.push_str("IMP");
            has_info = true;
        }
        if !has_info {
            output.push('.');
        }
        output.push_str("\tGT");
        if self.fields.dosage {
            output.push_str(":DS");
        }
        if self.fields.allele_probabilities {
            output.push_str(":AP1:AP2");
        }
        if self.fields.genotype_probabilities {
            output.push_str(":GP");
        }
        for sample in 0..self.samples.len() {
            let call = row.calls.get(sample).unwrap();
            output.push('\t');
            append_gt(output, call);
            if self.fields.dosage {
                output.push(':');
                append_optional_dosage(output, row.dosage.as_deref(), sample);
            }
            if self.fields.allele_probabilities {
                output.push(':');
                append_optional_sample_value(output, row.ap1.as_deref(), sample, 6);
                output.push(':');
                append_optional_sample_value(output, row.ap2.as_deref(), sample, 6);
            }
            if self.fields.genotype_probabilities {
                output.push(':');
                append_gp(output, row.gp.as_deref(), sample);
            }
        }
        output.push('\n');
        Ok(())
    }

    fn write_encoded(
        &mut self,
        bytes: &[u8],
        worker_pool: &rayon::ThreadPool,
    ) -> Result<(), String> {
        match &mut self.stream {
            VcfStream::Plain(writer) => writer
                .write_all(bytes)
                .map_err(|e| format!("write {}: {e}", self.path.display()))?,
            VcfStream::Bgzip(writer) => writer.write_bytes(bytes, Some(worker_pool))?,
        }
        Ok(())
    }

    pub(crate) fn finish(self, worker_pool: Option<&rayon::ThreadPool>) -> Result<(), String> {
        let Self {
            path,
            stream,
            row_buffer,
            rows_written,
            profile_enabled,
            profile,
            ..
        } = self;
        let row_buffer_bytes = row_buffer.len();
        let started = profile_enabled.then(Instant::now);
        let result = match stream {
            VcfStream::Plain(mut writer) => writer
                .flush()
                .map_err(|e| format!("flush {}: {e}", path.display())),
            VcfStream::Bgzip(writer) => writer.finish(worker_pool),
        };
        if profile_enabled {
            eprintln!(
                "JANUSX_IMP_VCF_PROFILE rows={} format_s={:.6} format_bytes={} submit_s={:.6} submit_bytes={} buffered_bytes={} finish_s={:.6} total_s={:.6}",
                rows_written,
                profile.format_elapsed.as_secs_f64(),
                profile.format_bytes,
                profile.submit_elapsed.as_secs_f64(),
                profile.submit_bytes,
                row_buffer_bytes,
                started.map_or(0.0, |started| started.elapsed().as_secs_f64()),
                profile
                    .format_elapsed
                    .saturating_add(profile.submit_elapsed)
                    .saturating_add(
                        started.map_or(Duration::ZERO, |started| started.elapsed()),
                    )
                    .as_secs_f64(),
            );
        }
        result
    }

    fn write_text(&mut self, text: &str) -> Result<(), String> {
        match &mut self.stream {
            VcfStream::Plain(writer) => writer
                .write_all(text.as_bytes())
                .map_err(|e| format!("write {}: {e}", self.path.display())),
            VcfStream::Bgzip(writer) => writer.write_bytes(text.as_bytes(), None),
        }
    }
}

fn resolve_bgzf_threads(default_threads: usize, override_value: Option<&str>) -> usize {
    override_value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|threads| *threads > 0)
        .unwrap_or(default_threads)
        .clamp(1, BGZF_MAX_COMPRESSION_THREADS)
}

fn append_gt(output: &mut String, call: DiploidCall) {
    for (index, allele) in call.alleles.into_iter().enumerate() {
        if index > 0 {
            output.push('|');
        }
        if let Some(allele) = allele {
            match allele {
                0 => output.push('0'),
                1 => output.push('1'),
                2 => output.push('2'),
                3 => output.push('3'),
                4 => output.push('4'),
                5 => output.push('5'),
                6 => output.push('6'),
                7 => output.push('7'),
                8 => output.push('8'),
                9 => output.push('9'),
                _ => write!(output, "{allele}").expect("formatting into a String cannot fail"),
            }
        } else {
            output.push('.');
        }
    }
}

fn append_optional_dosage(output: &mut String, values: Option<&[f32]>, sample: usize) {
    if let Some(value) = values
        .and_then(|values| values.get(sample))
        .filter(|value| value.is_finite())
    {
        append_dosage(output, *value);
    } else {
        output.push('.');
    }
}

fn append_dosage(output: &mut String, value: f32) {
    let index = (f64::from(value) * 100.0).round();
    if (0.0..=200.0).contains(&index) {
        output.push_str(&dosage_labels()[index as usize]);
    } else {
        append_trimmed_fixed(output, value, 2);
    }
}

fn dosage_labels() -> &'static [String; 201] {
    static LABELS: OnceLock<[String; 201]> = OnceLock::new();
    LABELS.get_or_init(|| {
        std::array::from_fn(|index| {
            let mut output = String::new();
            append_trimmed_fixed(&mut output, index as f32 / 100.0, 2);
            output
        })
    })
}

fn append_optional_sample_value(
    output: &mut String,
    values: Option<&[f32]>,
    sample: usize,
    precision: usize,
) {
    if let Some(value) = values
        .and_then(|values| values.get(sample))
        .filter(|value| value.is_finite())
    {
        append_trimmed_fixed(output, value, precision);
    } else {
        output.push('.');
    }
}

fn append_gp(output: &mut String, values: Option<&[[f32; 3]]>, sample: usize) {
    if let Some([first, second, third]) = values.and_then(|values| values.get(sample)) {
        append_trimmed_fixed(output, first, 6);
        output.push(',');
        append_trimmed_fixed(output, second, 6);
        output.push(',');
        append_trimmed_fixed(output, third, 6);
    } else {
        output.push('.');
    }
}

fn append_trimmed_fixed<T: std::fmt::Display>(output: &mut String, value: T, precision: usize) {
    let start = output.len();
    write!(output, "{value:.precision$}").expect("formatting into a String cannot fail");
    let mut end = output.len();
    while end > start && output.as_bytes()[end - 1] == b'0' {
        end -= 1;
    }
    if end > start && output.as_bytes()[end - 1] == b'.' {
        end -= 1;
    }
    output.truncate(end);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imp::types::DiploidCalls;
    use flate2::read::MultiGzDecoder;
    use std::io::Read;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn fixed_float_output_omits_only_fractional_trailing_zeroes() {
        let mut output = String::new();
        for (value, expected) in [
            (0.0f32, "0"),
            (2.0f32, "2"),
            (0.120_000f32, "0.12"),
            (0.123_456_7f32, "0.123457"),
            (-0.0f32, "-0"),
        ] {
            output.clear();
            append_trimmed_fixed(&mut output, value, 6);
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn bgzf_thread_override_accepts_only_positive_integers() {
        assert_eq!(resolve_bgzf_threads(8, Some("3")), 3);
        assert_eq!(resolve_bgzf_threads(8, Some("64")), 32);
        assert_eq!(resolve_bgzf_threads(8, Some("0")), 8);
        assert_eq!(resolve_bgzf_threads(8, Some("not-a-number")), 8);
        assert_eq!(resolve_bgzf_threads(8, None), 8);
    }

    #[test]
    fn dosage_lookup_matches_two_decimal_trimmed_output() {
        for (value, expected) in [
            (0.0f32, "0"),
            (0.004f32, "0"),
            (0.005f32, "0"),
            (0.006f32, "0.01"),
            (0.124f32, "0.12"),
            (0.126f32, "0.13"),
            (0.5f32, "0.5"),
            (1.0f32, "1"),
            (1.456f32, "1.46"),
            (2.0f32, "2"),
        ] {
            let mut output = String::new();
            append_dosage(&mut output, value);
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn parallel_bgzf_batches_preserve_payload_order_across_partial_blocks() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "janusx-imp-vcf-bgzf-parallel-{}-{unique}.vcf.gz",
            std::process::id()
        ));
        let expected = (0..(BGZF_UNCOMPRESSED_BLOCK * 3 + 371))
            .map(|index| (index.wrapping_mul(37) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut writer = BgzfWriter::create(&path, 3).unwrap();
        let worker_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        for chunk in expected.chunks(7_919) {
            writer.write_bytes(chunk, Some(&worker_pool)).unwrap();
        }
        writer.finish(Some(&worker_pool)).unwrap();

        let mut observed = Vec::new();
        MultiGzDecoder::new(File::open(&path).unwrap())
            .read_to_end(&mut observed)
            .unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(observed, expected);
    }

    #[test]
    fn bgzf_row_preserves_gt_dosage_and_posterior_formatting() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "janusx-imp-vcf-writer-{}-{unique}.vcf.gz",
            std::process::id()
        ));
        let fields = VcfOutputFields {
            dosage: true,
            allele_probabilities: true,
            genotype_probabilities: true,
        };
        let mut writer =
            ImpVcfWriter::create(&path, &["sample-a".into(), "sample-b".into()], fields).unwrap();
        let worker_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        writer
            .write_row(
                &PhasedRow {
                    meta: crate::imp::types::VariantMeta {
                        chrom: "1".into(),
                        pos: 42,
                        id: "rs42".into(),
                        ref_allele: "A".into(),
                        alt_alleles: vec!["G".into()],
                    },
                    calls: vec![
                        DiploidCall {
                            alleles: [Some(0), Some(1)],
                            phased: true,
                        },
                        DiploidCall::missing(true),
                    ]
                    .into(),
                    dosage: Some(vec![0.123_456_7, f32::NAN]),
                    ap1: Some(vec![0.1, 0.2]),
                    ap2: Some(vec![0.9, 0.8]),
                    gp: Some(vec![[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]]),
                    info_af: Some(0.25),
                    info_dr2: Some(0.75),
                    imputed: true,
                },
                &worker_pool,
            )
            .unwrap();
        writer.finish(Some(&worker_pool)).unwrap();

        let mut decoded = String::new();
        MultiGzDecoder::new(File::open(&path).unwrap())
            .read_to_string(&mut decoded)
            .unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(decoded.ends_with(
            "1\t42\trs42\tA\tG\t.\tPASS\tAF=0.2500;DR2=0.75;IMP\tGT:DS:AP1:AP2:GP\t0|1:0.12:0.1:0.9:0.2,0.3,0.5\t.|.:.:0.2:0.8:0.1,0.4,0.5\n"
        ));
    }

    #[test]
    fn parallel_row_batch_preserves_serial_vcf_output() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let serial_path = std::env::temp_dir().join(format!(
            "janusx-imp-vcf-serial-{}-{unique}.vcf.gz",
            std::process::id()
        ));
        let parallel_path = std::env::temp_dir().join(format!(
            "janusx-imp-vcf-parallel-{}-{unique}.vcf.gz",
            std::process::id()
        ));
        let fields = VcfOutputFields {
            dosage: true,
            allele_probabilities: true,
            genotype_probabilities: true,
        };
        let samples = vec!["sample-a".to_owned(), "sample-b".to_owned()];
        let rows = (0..PARALLEL_ROWS_PER_GROUP + 257)
            .map(|marker| {
                let calls = DiploidCalls::from_calls(vec![
                    DiploidCall {
                        alleles: [Some((marker % 2) as u16), Some(((marker + 1) % 2) as u16)],
                        phased: true,
                    },
                    DiploidCall {
                        alleles: [Some(((marker + 2) % 2) as u16), Some(0)],
                        phased: true,
                    },
                ]);
                PhasedRow {
                    meta: crate::imp::types::VariantMeta {
                        chrom: "1".to_owned(),
                        pos: marker as u64 + 1,
                        id: format!("v{marker}"),
                        ref_allele: "A".to_owned(),
                        alt_alleles: vec!["G".to_owned()],
                    },
                    calls,
                    dosage: Some(
                        (0..2)
                            .map(|sample| ((marker + sample) % 3) as f32 * 0.5)
                            .collect(),
                    ),
                    ap1: Some(
                        (0..2)
                            .map(|sample| ((marker + sample + 1) % 3) as f32 * 0.25)
                            .collect(),
                    ),
                    ap2: Some(
                        (0..2)
                            .map(|sample| ((marker + sample + 2) % 3) as f32 * 0.25)
                            .collect(),
                    ),
                    gp: Some((0..2).map(|_| [0.25, 0.5, 0.25]).collect()),
                    info_af: Some(0.25),
                    info_dr2: Some(0.75),
                    imputed: marker % 2 == 0,
                }
            })
            .collect::<Vec<_>>();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        let mut serial = ImpVcfWriter::create(&serial_path, &samples, fields).unwrap();
        for row in &rows {
            serial.write_row(row, &pool).unwrap();
        }
        serial.finish(Some(&pool)).unwrap();

        let row_refs = rows.iter().collect::<Vec<_>>();
        let mut parallel = ImpVcfWriter::create(&parallel_path, &samples, fields).unwrap();
        parallel.write_rows(&row_refs, &pool).unwrap();
        parallel.finish(Some(&pool)).unwrap();

        let mut serial_text = Vec::new();
        MultiGzDecoder::new(File::open(&serial_path).unwrap())
            .read_to_end(&mut serial_text)
            .unwrap();
        let mut parallel_text = Vec::new();
        MultiGzDecoder::new(File::open(&parallel_path).unwrap())
            .read_to_end(&mut parallel_text)
            .unwrap();
        assert_eq!(parallel_text, serial_text);

        let _ = std::fs::remove_file(serial_path);
        let _ = std::fs::remove_file(parallel_path);
    }

    #[test]
    fn bounded_async_bgzf_writer_drains_queued_batches_in_order() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "janusx-imp-vcf-async-{}-{unique}.vcf.gz",
            std::process::id()
        ));
        let fields = VcfOutputFields {
            dosage: true,
            allele_probabilities: false,
            genotype_probabilities: false,
        };
        let samples = (0..16).map(|index| format!("s{index}")).collect::<Vec<_>>();
        let rows = (0..12_000)
            .map(|marker| PhasedRow {
                meta: crate::imp::types::VariantMeta {
                    chrom: "1".to_owned(),
                    pos: marker as u64 + 1,
                    id: format!("v{marker}"),
                    ref_allele: "A".to_owned(),
                    alt_alleles: vec!["G".to_owned()],
                },
                calls: (0..samples.len())
                    .map(|sample| DiploidCall {
                        alleles: [Some(((marker + sample) & 1) as u16), Some(0)],
                        phased: true,
                    })
                    .collect::<Vec<_>>()
                    .into(),
                dosage: Some(
                    (0..samples.len())
                        .map(|sample| ((marker + sample) % 3) as f32 * 0.5)
                        .collect(),
                ),
                ap1: None,
                ap2: None,
                gp: None,
                info_af: Some(0.25),
                info_dr2: Some(0.75),
                imputed: false,
            })
            .collect::<Vec<_>>();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let mut writer = ImpVcfWriter::create_with_threads(&path, &samples, fields, 3).unwrap();
        for rows in rows.chunks(257) {
            let refs = rows.iter().collect::<Vec<_>>();
            writer.write_rows(&refs, &pool).unwrap();
        }
        writer.finish(Some(&pool)).unwrap();

        let mut decoded = String::new();
        MultiGzDecoder::new(File::open(&path).unwrap())
            .read_to_string(&mut decoded)
            .unwrap();
        let _ = std::fs::remove_file(path);
        let records = decoded
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 12_000);
        assert!(records[0].starts_with("1\t1\tv0\t"));
        assert!(records[8_193].starts_with("1\t8194\tv8193\t"));
        assert!(records[11_999].starts_with("1\t12000\tv11999\t"));
    }
}
