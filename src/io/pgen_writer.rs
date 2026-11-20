//! Streaming writer for biallelic hard calls and deterministic presence rows.
//!
//! Hard-call records store the 2-bit genotype track plus phase information.
//! Presence rows use PGEN's constant or 1-bit encodings. Record lengths and
//! block offsets are written only at finish, after the input row count is
//! known; payloads are streamed to sibling scratch files. PVAR may be emitted
//! as plain text or zstd-compressed text.

use crate::imp::types::{DiploidCalls, VariantMeta};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const PGEN_BLOCK_VARIANTS: usize = 1 << 16;

struct TempPathGuard {
    paths: Vec<PathBuf>,
    armed: bool,
}

impl Drop for TempPathGuard {
    fn drop(&mut self) {
        if self.armed {
            for path in &self.paths {
                let _ = fs::remove_file(path);
            }
        }
    }
}

pub(crate) struct PgenWriter {
    pgen_path: PathBuf,
    pvar_path: PathBuf,
    psam_path: PathBuf,
    temp_pgen: PathBuf,
    temp_pvar: PathBuf,
    temp_psam: PathBuf,
    temp_data: PathBuf,
    temp_types: PathBuf,
    temp_lengths: PathBuf,
    data: BufWriter<File>,
    types: BufWriter<File>,
    lengths: BufWriter<File>,
    pvar: Option<PvarSink>,
    psam: BufWriter<File>,
    n_samples: usize,
    variant_count: usize,
    max_record_len: u32,
    block_bytes: Vec<u64>,
    overwrite_existing: bool,
    finished: bool,
}

enum PvarSink {
    Plain(BufWriter<File>),
    Zstd(BufWriter<zstd::stream::write::Encoder<'static, BufWriter<File>>>),
}

impl Write for PvarSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(writer) => writer.write(buf),
            Self::Zstd(writer) => writer.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Plain(writer) => writer.flush(),
            Self::Zstd(writer) => writer.flush(),
        }
    }
}

impl PvarSink {
    fn finish(self) -> std::io::Result<()> {
        match self {
            Self::Plain(mut writer) => writer.flush(),
            Self::Zstd(writer) => {
                let encoder = writer
                    .into_inner()
                    .map_err(std::io::IntoInnerError::into_error)?;
                let mut writer = encoder.finish()?;
                writer.flush()
            }
        }
    }
}

impl PgenWriter {
    pub(crate) fn create(prefix: &Path, samples: &[String]) -> Result<Self, String> {
        Self::create_with_options(prefix, samples, false, false)
    }

    pub(crate) fn create_with_pvar_compression(
        prefix: &Path,
        samples: &[String],
        compress_pvar: bool,
    ) -> Result<Self, String> {
        Self::create_with_options(prefix, samples, compress_pvar, false)
    }

    /// Create a native PGEN writer whose completed fileset replaces existing
    /// output members. This preserves the overwrite behavior of the generic
    /// `gformat` conversion path; kmerge and imputation continue to use the
    /// safer no-overwrite constructor above.
    pub(crate) fn create_overwrite(prefix: &Path, samples: &[String]) -> Result<Self, String> {
        Self::create_with_options(prefix, samples, false, true)
    }

    fn create_with_options(
        prefix: &Path,
        samples: &[String],
        compress_pvar: bool,
        overwrite_existing: bool,
    ) -> Result<Self, String> {
        if samples.is_empty() {
            return Err("PGEN output requires at least one sample".to_owned());
        }
        if samples.len() > u32::MAX as usize {
            return Err("PGEN output sample count exceeds the format limit".to_owned());
        }
        for sample in samples {
            if sample.is_empty() || sample.chars().any(char::is_whitespace) {
                return Err(format!(
                    "PGEN sample IDs must be nonempty and contain no whitespace: {sample:?}"
                ));
            }
        }
        let prefix = normalize_prefix(prefix);
        let pgen_path = append_suffix(&prefix, ".pgen");
        let pvar_path = append_suffix(&prefix, if compress_pvar { ".pvar.zst" } else { ".pvar" });
        let psam_path = append_suffix(&prefix, ".psam");
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|err| format!("system clock cannot name temporary PGEN files: {err}"))?
            .as_nanos();
        let temp_stem = format!(".tmp.{}.{nonce}", std::process::id());
        let temp_pgen = append_suffix(&prefix, &format!("{temp_stem}.pgen"));
        let temp_pvar_suffix = if compress_pvar {
            format!("{temp_stem}.pvar.zst")
        } else {
            format!("{temp_stem}.pvar")
        };
        let temp_pvar = append_suffix(&prefix, &temp_pvar_suffix);
        let temp_psam = append_suffix(&prefix, &format!("{temp_stem}.psam"));
        let temp_data = append_suffix(&prefix, &format!("{temp_stem}.records"));
        let temp_types = append_suffix(&prefix, &format!("{temp_stem}.types"));
        let temp_lengths = append_suffix(&prefix, &format!("{temp_stem}.lengths"));
        let mut temp_guard = TempPathGuard {
            paths: vec![
                temp_pgen.clone(),
                temp_pvar.clone(),
                temp_psam.clone(),
                temp_data.clone(),
                temp_types.clone(),
                temp_lengths.clone(),
            ],
            armed: true,
        };
        let pvar_file = create_temp_file(&temp_pvar)
            .map_err(|err| format!("create {}: {err}", temp_pvar.display()))?;
        let mut pvar = if compress_pvar {
            PvarSink::Zstd(BufWriter::with_capacity(
                1 << 20,
                zstd::stream::write::Encoder::new(BufWriter::new(pvar_file), 3)
                    .map_err(|err| format!("create zstd encoder {}: {err}", temp_pvar.display()))?,
            ))
        } else {
            PvarSink::Plain(BufWriter::new(pvar_file))
        };
        writeln!(
            pvar,
            "##INFO=<ID=PR,Number=0,Type=Flag,Description=\"Provisional reference allele\">"
        )
        .and_then(|_| writeln!(pvar, "#CHROM\tPOS\tID\tREF\tALT\tINFO"))
        .map_err(|err| format!("write {}: {err}", temp_pvar.display()))?;
        let mut psam = BufWriter::new(
            create_temp_file(&temp_psam)
                .map_err(|err| format!("create {}: {err}", temp_psam.display()))?,
        );
        writeln!(psam, "#FID\tIID")
            .map_err(|err| format!("write {}: {err}", temp_psam.display()))?;
        for sample in samples {
            writeln!(psam, "0\t{sample}")
                .map_err(|err| format!("write {}: {err}", temp_psam.display()))?;
        }
        let data = BufWriter::new(
            create_temp_file(&temp_data)
                .map_err(|err| format!("create {}: {err}", temp_data.display()))?,
        );
        let types = BufWriter::new(
            create_temp_file(&temp_types)
                .map_err(|err| format!("create {}: {err}", temp_types.display()))?,
        );
        let lengths = BufWriter::new(
            create_temp_file(&temp_lengths)
                .map_err(|err| format!("create {}: {err}", temp_lengths.display()))?,
        );
        let writer = Self {
            pgen_path,
            pvar_path,
            psam_path,
            temp_pgen,
            temp_pvar,
            temp_psam,
            temp_data,
            temp_types,
            temp_lengths,
            data,
            types,
            lengths,
            pvar: Some(pvar),
            psam,
            n_samples: samples.len(),
            variant_count: 0,
            max_record_len: 0,
            block_bytes: Vec::new(),
            overwrite_existing,
            finished: false,
        };
        temp_guard.armed = false;
        Ok(writer)
    }

    pub(crate) fn write_row(
        &mut self,
        meta: &VariantMeta,
        calls: &DiploidCalls,
        dosage: Option<&[f32]>,
    ) -> Result<(), String> {
        if calls.len() != self.n_samples {
            return Err(format!(
                "{}:{} has {} calls; PGEN output expects {}",
                meta.chrom,
                meta.pos,
                calls.len(),
                self.n_samples
            ));
        }
        if dosage.is_some() {
            return Err(format!(
                "{}:{} has posterior dosage values; this PGEN writer currently stores phased hard calls only",
                meta.chrom, meta.pos
            ));
        }
        if meta.alt_alleles.len() != 1 {
            return Err(format!(
                "PGEN output supports biallelic variants only; {}:{} has {} ALT alleles",
                meta.chrom,
                meta.pos,
                meta.alt_alleles.len()
            ));
        }
        for (name, value) in [
            ("chromosome", meta.chrom.as_str()),
            ("variant ID", meta.id.as_str()),
            ("REF allele", meta.ref_allele.as_str()),
            ("ALT allele", meta.alt_alleles[0].as_str()),
        ] {
            if value.is_empty() || value.chars().any(char::is_whitespace) {
                return Err(format!(
                    "PGEN {name} at {}:{} must be nonempty and contain no whitespace",
                    meta.chrom, meta.pos
                ));
            }
        }
        if meta.pos == 0 || meta.pos > i32::MAX as u64 {
            return Err(format!(
                "PGEN position {}:{} is outside the supported positive 32-bit range",
                meta.chrom, meta.pos
            ));
        }

        let mut hardcalls = vec![0u8; self.n_samples.div_ceil(4)];
        let phase_info = encode_phase_track(calls)?;
        for (sample, call) in calls.iter().enumerate() {
            call.validate_alleles(meta.allele_count())?;
            let category = match call.alleles {
                [Some(0), Some(0)] => 0,
                [Some(0), Some(1)] | [Some(1), Some(0)] => 1,
                [Some(1), Some(1)] => 2,
                [None, None] => 3,
                _ => {
                    return Err(format!(
                        "PGEN output requires complete biallelic hard calls at {}:{} sample {}",
                        meta.chrom,
                        meta.pos,
                        sample + 1
                    ));
                }
            };
            let bit_offset = (sample & 3) * 2;
            hardcalls[sample >> 2] |= category << bit_offset;
        }

        // Aux2 header bit 0 clear means phase information is present for every
        // heterozygote; remaining bits encode the 1|0 orientation.
        let mut record = hardcalls;
        record.extend_from_slice(&phase_info);
        self.write_encoded_record(0x10, &record, meta)?;
        Ok(())
    }

    /// Write an unphased biallelic hard-call row from JanusX's compact
    /// genotype coding (0, 1, 2, and a negative value for missing).
    ///
    /// This is the streaming counterpart used by the generic Python
    /// genotype writer.  It deliberately emits the same native PGEN 0x10
    /// record layout as `write_row`, without constructing a `DiploidCalls`
    /// object for every marker.
    pub(crate) fn write_hardcall_row(
        &mut self,
        chrom: &str,
        pos: u64,
        id: &str,
        ref_allele: &str,
        alt_allele: &str,
        row: &[i8],
    ) -> Result<(), String> {
        if row.len() != self.n_samples {
            return Err(format!(
                "{chrom}:{pos} has {} calls; PGEN output expects {}",
                row.len(),
                self.n_samples
            ));
        }
        if chrom.is_empty() || chrom.chars().any(char::is_whitespace) {
            return Err(format!(
                "PGEN chromosome at {chrom}:{pos} must be nonempty and contain no whitespace"
            ));
        }
        if pos == 0 || pos > i32::MAX as u64 {
            return Err(format!(
                "PGEN position {chrom}:{pos} is outside the supported positive 32-bit range"
            ));
        }
        for (name, value) in [
            ("variant ID", id),
            ("REF allele", ref_allele),
            ("ALT allele", alt_allele),
        ] {
            if value.is_empty() || value.chars().any(char::is_whitespace) {
                return Err(format!(
                    "PGEN {name} at {chrom}:{pos} must be nonempty and contain no whitespace"
                ));
            }
        }

        // Native PGEN hard-call categories are 0=hom-ref, 1=het,
        // 2=hom-alt, and 3=missing.  This is intentionally different from
        // the PLINK BED two-bit ordering used by mode 0x01.
        let mut record = vec![0u8; self.n_samples.div_ceil(4) + 1];
        for (sample, &genotype) in row.iter().enumerate() {
            let category = match genotype {
                0 => 0u8,
                1 => 1u8,
                2 => 2u8,
                _ => 3u8,
            };
            record[sample >> 2] |= category << ((sample & 3) * 2);
        }
        // The final byte is the empty phase track used for unphased calls.
        record[self.n_samples.div_ceil(4)] = 0;

        self.write_record_payload(0x10, &record)?;
        self.write_pvar_row(
            chrom.as_bytes(),
            pos,
            id.as_bytes(),
            ref_allele.as_bytes(),
            alt_allele.as_bytes(),
        )
    }

    /// Flush streaming buffers while keeping the temporary PGEN files
    /// unpublished. `finish` performs the final header/index assembly and
    /// publication.
    pub(crate) fn flush(&mut self) -> Result<(), String> {
        self.data
            .flush()
            .map_err(|err| format!("flush {}: {err}", self.temp_data.display()))?;
        self.types
            .flush()
            .map_err(|err| format!("flush {}: {err}", self.temp_types.display()))?;
        self.lengths
            .flush()
            .map_err(|err| format!("flush {}: {err}", self.temp_lengths.display()))?;
        self.pvar
            .as_mut()
            .ok_or_else(|| "PGEN PVAR writer is unavailable".to_owned())?
            .flush()
            .map_err(|err| format!("flush {}: {err}", self.temp_pvar.display()))?;
        self.psam
            .flush()
            .map_err(|err| format!("flush {}: {err}", self.temp_psam.display()))?;
        Ok(())
    }

    pub(crate) fn write_presence_row(
        &mut self,
        chrom: &str,
        pos: u64,
        id: &[u8],
        present: &[u8],
    ) -> Result<(), String> {
        let expected = self.n_samples.div_ceil(8);
        if present.len() != expected {
            return Err(format!(
                "presence bitset length mismatch: got {}, expected {}",
                present.len(),
                expected
            ));
        }
        if pos == 0 || pos > i32::MAX as u64 {
            return Err(format!(
                "PGEN position {pos} is outside the supported positive 32-bit range"
            ));
        }
        if chrom.is_empty() || chrom.chars().any(char::is_whitespace) {
            return Err("PGEN chromosome must be nonempty and contain no whitespace".to_owned());
        }
        if id.is_empty() || id.iter().any(u8::is_ascii_whitespace) {
            return Err("PGEN variant ID must be nonempty and contain no whitespace".to_owned());
        }
        let byte_len = self.n_samples.div_ceil(8);
        let used_bits = self.n_samples & 7;
        let last_mask = if used_bits == 0 {
            u8::MAX
        } else {
            (1u8 << used_bits) - 1
        };
        let present_count: usize = present[..byte_len]
            .iter()
            .enumerate()
            .map(|(idx, byte)| {
                let value = if idx + 1 == byte_len {
                    byte & last_mask
                } else {
                    *byte
                };
                value.count_ones() as usize
            })
            .sum();
        if present_count == 0 {
            self.write_record_payload(4, &[0])?;
        } else if present_count == self.n_samples {
            self.write_record_payload(6, &[0])?;
        } else {
            let record_len = byte_len
                .checked_add(2)
                .ok_or_else(|| "PGEN presence record length overflow".to_owned())?;
            self.start_record(1, record_len)?;
            self.data
                .write_all(&[2])
                .and_then(|_| {
                    if byte_len > 1 {
                        self.data.write_all(&present[..byte_len - 1])
                    } else {
                        Ok(())
                    }
                })
                .and_then(|_| self.data.write_all(&[present[byte_len - 1] & last_mask]))
                .and_then(|_| self.data.write_all(&[0]))
                .map_err(|err| format!("write {}: {err}", self.temp_data.display()))?;
        }
        self.write_pvar_row(chrom.as_bytes(), pos, id, b"A", b"G")
    }

    fn write_encoded_record(
        &mut self,
        record_type: u8,
        record: &[u8],
        meta: &VariantMeta,
    ) -> Result<(), String> {
        self.write_record_payload(record_type, record)?;
        let id = if meta.id.is_empty() {
            b".".as_slice()
        } else {
            meta.id.as_bytes()
        };
        self.write_pvar_row(
            meta.chrom.as_bytes(),
            meta.pos,
            id,
            meta.ref_allele.as_bytes(),
            meta.alt_alleles[0].as_bytes(),
        )
    }

    fn write_record_payload(&mut self, record_type: u8, record: &[u8]) -> Result<(), String> {
        let record_len = record.len();
        self.start_record(record_type, record_len)?;
        self.data
            .write_all(record)
            .map_err(|err| format!("write {}: {err}", self.temp_data.display()))
    }

    fn start_record(&mut self, record_type: u8, record_len: usize) -> Result<(), String> {
        if record_len > u32::MAX as usize {
            return Err("PGEN record exceeds the 32-bit record-length limit".to_owned());
        }
        let encoded_len = record_len as u32;
        self.types
            .write_all(&[record_type])
            .map_err(|err| format!("write {}: {err}", self.temp_types.display()))?;
        self.lengths
            .write_all(&encoded_len.to_le_bytes())
            .map_err(|err| format!("write {}: {err}", self.temp_lengths.display()))?;
        self.variant_count += 1;
        self.max_record_len = self.max_record_len.max(encoded_len);
        if self.variant_count % PGEN_BLOCK_VARIANTS == 1 {
            self.block_bytes.push(0);
        }
        if let Some(block) = self.block_bytes.last_mut() {
            *block = block.saturating_add(record_len as u64);
        }
        Ok(())
    }

    fn write_pvar_row(
        &mut self,
        chrom: &[u8],
        pos: u64,
        id: &[u8],
        ref_allele: &[u8],
        alt_allele: &[u8],
    ) -> Result<(), String> {
        let pvar = self
            .pvar
            .as_mut()
            .ok_or_else(|| "PGEN PVAR writer is unavailable".to_owned())?;
        pvar.write_all(chrom)
            .and_then(|_| write!(pvar, "\t{pos}\t"))
            .and_then(|_| pvar.write_all(id))
            .and_then(|_| pvar.write_all(b"\t"))
            .and_then(|_| pvar.write_all(ref_allele))
            .and_then(|_| pvar.write_all(b"\t"))
            .and_then(|_| pvar.write_all(alt_allele))
            .and_then(|_| pvar.write_all(b"\tPR\n"))
            .map_err(|err| format!("write {}: {err}", self.temp_pvar.display()))
    }

    pub(crate) fn finish(mut self) -> Result<(), String> {
        self.flush()?;
        let pvar = self
            .pvar
            .take()
            .ok_or_else(|| "PGEN PVAR writer is unavailable".to_owned())?;
        pvar.finish()
            .map_err(|err| format!("finish {}: {err}", self.temp_pvar.display()))?;
        if self.variant_count == 0 {
            return Err("PGEN output contains zero variants".to_owned());
        }
        if self.variant_count > u32::MAX as usize {
            return Err("PGEN output variant count exceeds the format limit".to_owned());
        }

        let length_bytes = if self.max_record_len <= u8::MAX as u32 {
            1usize
        } else if self.max_record_len <= u16::MAX as u32 {
            2
        } else if self.max_record_len <= 0x00ff_ffff {
            3
        } else {
            4
        };
        let format = 4 + length_bytes - 1;
        let block_count = self.variant_count.div_ceil(PGEN_BLOCK_VARIANTS);
        let offsets_end = 12usize
            .checked_add(
                block_count
                    .checked_mul(8)
                    .ok_or_else(|| "PGEN offset table overflow".to_owned())?,
            )
            .ok_or_else(|| "PGEN header size overflow".to_owned())?;
        let length_table_bytes = self
            .variant_count
            .checked_mul(length_bytes)
            .ok_or_else(|| "PGEN record-length table overflow".to_owned())?;
        let data_start = offsets_end
            .checked_add(self.variant_count)
            .and_then(|value| value.checked_add(length_table_bytes))
            .ok_or_else(|| "PGEN data offset overflow".to_owned())? as u64;
        let mut block_offsets = Vec::with_capacity(block_count);
        let mut offset = data_start;
        for bytes in &self.block_bytes {
            block_offsets.push(offset);
            offset = offset
                .checked_add(*bytes)
                .ok_or_else(|| "PGEN data offset overflow".to_owned())?;
        }

        let mut pgen = BufWriter::new(
            create_temp_file(&self.temp_pgen)
                .map_err(|err| format!("create {}: {err}", self.temp_pgen.display()))?,
        );
        pgen.write_all(&[0x6c, 0x1b, 0x10])
            .and_then(|_| pgen.write_all(&(self.variant_count as u32).to_le_bytes()))
            .and_then(|_| pgen.write_all(&(self.n_samples as u32).to_le_bytes()))
            .and_then(|_| pgen.write_all(&[format as u8]))
            .map_err(|err| format!("write {} header: {err}", self.temp_pgen.display()))?;
        for block_offset in &block_offsets {
            pgen.write_all(&block_offset.to_le_bytes())
                .map_err(|err| format!("write {} offsets: {err}", self.temp_pgen.display()))?;
        }
        let mut lengths = BufReader::new(
            File::open(&self.temp_lengths)
                .map_err(|err| format!("open {}: {err}", self.temp_lengths.display()))?,
        );
        let mut types = BufReader::new(
            File::open(&self.temp_types)
                .map_err(|err| format!("open {}: {err}", self.temp_types.display()))?,
        );
        let mut row_start = 0usize;
        for _ in 0..block_count {
            let rows = (self.variant_count - row_start).min(PGEN_BLOCK_VARIANTS);
            let mut block_types = vec![0u8; rows];
            types
                .read_exact(&mut block_types)
                .map_err(|err| format!("read {}: {err}", self.temp_types.display()))?;
            pgen.write_all(&block_types)
                .map_err(|err| format!("write {} record types: {err}", self.temp_pgen.display()))?;
            for _ in 0..rows {
                let mut encoded_len = [0u8; 4];
                lengths
                    .read_exact(&mut encoded_len)
                    .map_err(|err| format!("read {}: {err}", self.temp_lengths.display()))?;
                let bytes = u32::from_le_bytes(encoded_len).to_le_bytes();
                pgen.write_all(&bytes[..length_bytes]).map_err(|err| {
                    format!("write {} record lengths: {err}", self.temp_pgen.display())
                })?;
            }
            row_start += rows;
        }
        let mut records = File::open(&self.temp_data)
            .map_err(|err| format!("open {}: {err}", self.temp_data.display()))?;
        std::io::copy(&mut records, &mut pgen)
            .map_err(|err| format!("copy PGEN records: {err}"))?;
        pgen.flush()
            .map_err(|err| format!("flush {}: {err}", self.temp_pgen.display()))?;
        drop(pgen);

        // Publish each completed temporary file. The no-overwrite mode uses
        // hard links or create-new copies; the generic gformat mode preserves
        // its historical replacement behavior.
        let publications = [
            (&self.temp_pvar, &self.pvar_path),
            (&self.temp_psam, &self.psam_path),
            (&self.temp_pgen, &self.pgen_path),
        ];
        let mut published = Vec::with_capacity(publications.len());
        for (source, destination) in publications {
            let result = if self.overwrite_existing {
                publish_replace(source, destination)
            } else {
                publish_without_overwrite(source, destination)
            };
            if let Err(err) = result {
                for path in &published {
                    let _ = fs::remove_file(path);
                }
                return Err(format!(
                    "publish PGEN fileset member {} failed: {err}",
                    destination.display()
                ));
            }
            published.push(destination);
        }
        self.finished = true;
        for path in [
            &self.temp_pgen,
            &self.temp_pvar,
            &self.temp_psam,
            &self.temp_data,
            &self.temp_types,
            &self.temp_lengths,
        ] {
            let _ = fs::remove_file(path);
        }
        Ok(())
    }
}

fn publish_without_overwrite(source: &Path, destination: &Path) -> std::io::Result<()> {
    match fs::hard_link(source, destination) {
        Ok(()) => Ok(()),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::Unsupported | std::io::ErrorKind::InvalidInput
            ) || error.raw_os_error() == Some(45) =>
        {
            copy_without_overwrite(source, destination)
        }
        Err(error) => Err(error),
    }
}

fn publish_replace(source: &Path, destination: &Path) -> std::io::Result<()> {
    match fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            fs::remove_file(destination)?;
            fs::rename(source, destination)
        }
        Err(error) => Err(error),
    }
}

fn copy_without_overwrite(source: &Path, destination: &Path) -> std::io::Result<()> {
    let mut input = File::open(source)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    let result = std::io::copy(&mut input, &mut output).and_then(|_| output.flush());
    if let Err(error) = result {
        drop(output);
        let _ = fs::remove_file(destination);
        return Err(error);
    }
    Ok(())
}

impl Drop for PgenWriter {
    fn drop(&mut self) {
        if !self.finished {
            for path in [
                &self.temp_pgen,
                &self.temp_pvar,
                &self.temp_psam,
                &self.temp_data,
                &self.temp_types,
                &self.temp_lengths,
            ] {
                let _ = fs::remove_file(path);
            }
        }
    }
}

fn normalize_prefix(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    for suffix in [".pvar.zst", ".pgen", ".pvar", ".psam"] {
        if text.to_ascii_lowercase().ends_with(suffix) {
            return PathBuf::from(&text[..text.len() - suffix.len()]);
        }
    }
    path.to_path_buf()
}

fn append_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut output = prefix.as_os_str().to_os_string();
    output.push(suffix);
    PathBuf::from(output)
}

fn create_temp_file(path: &Path) -> Result<File, std::io::Error> {
    OpenOptions::new().write(true).create_new(true).open(path)
}

fn encode_phase_track(calls: &DiploidCalls) -> Result<Vec<u8>, String> {
    let heterozygote_count = calls.iter().filter(|call| call.is_heterozygous()).count();
    if heterozygote_count == 0 {
        return Ok(vec![0]);
    }
    let phased_count = calls
        .iter()
        .filter(|call| call.is_heterozygous() && call.phased)
        .count();
    let all_phased = phased_count == heterozygote_count;
    if all_phased {
        let mut bytes = vec![0u8; (heterozygote_count + 1).div_ceil(8)];
        for (index, call) in calls
            .iter()
            .filter(|call| call.is_heterozygous())
            .enumerate()
        {
            if call.alleles == [Some(1), Some(0)] {
                set_bit(&mut bytes, index + 1);
            }
        }
        return Ok(bytes);
    }

    let presence_bytes = (heterozygote_count + 1).div_ceil(8);
    let orientation_bytes = phased_count.div_ceil(8);
    let mut bytes = vec![0u8; presence_bytes + orientation_bytes];
    bytes[0] |= 1;
    let mut phase_index = 0usize;
    for (het_index, call) in calls
        .iter()
        .filter(|call| call.is_heterozygous())
        .enumerate()
    {
        if !call.phased {
            continue;
        }
        set_bit(&mut bytes, het_index + 1);
        if call.alleles == [Some(1), Some(0)] {
            set_bit(&mut bytes[presence_bytes..], phase_index);
        }
        phase_index += 1;
    }
    Ok(bytes)
}

fn set_bit(bytes: &mut [u8], bit: usize) {
    if let Some(byte) = bytes.get_mut(bit >> 3) {
        *byte |= 1 << (bit & 7);
    }
}

#[cfg(test)]
mod tests {
    use super::encode_phase_track;
    use crate::imp::types::{DiploidCall, DiploidCalls};

    #[test]
    fn phase_track_preserves_mixed_phased_and_unphased_heterozygotes() {
        let calls = DiploidCalls::from_calls(vec![
            DiploidCall {
                alleles: [Some(0), Some(0)],
                phased: false,
            },
            DiploidCall {
                alleles: [Some(1), Some(0)],
                phased: true,
            },
            DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: false,
            },
            DiploidCall::missing(false),
        ]);

        assert_eq!(
            encode_phase_track(&calls).unwrap(),
            vec![0b0000_0011, 0b0000_0001]
        );
    }

    #[test]
    fn phase_track_uses_compact_orientation_bits_when_all_hets_are_phased() {
        let calls = DiploidCalls::from_calls(vec![
            DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: true,
            },
            DiploidCall {
                alleles: [Some(1), Some(0)],
                phased: true,
            },
        ]);

        assert_eq!(encode_phase_track(&calls).unwrap(), vec![0b0000_0100]);
    }
}
