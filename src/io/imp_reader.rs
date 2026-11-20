//! Streaming, phase-preserving genotype readers for `jx imp`.

use crate::gfcore::{self, BimChunkReader, ImpNumericReader, SiteInfo};
use crate::imp::types::{DiploidCall, DiploidCalls, PhaseSetId, VariantMeta, VariantRow};
use crate::pgen::{self, PgenReader};
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fs::File;
use std::io::{BufRead, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const PGEN_READ_CHUNK_VARIANTS: usize = 1 << 12;
const VCF_PARSE_BLOCK_ROWS: usize = 256;
const MAX_VCF_PARSE_THREADS: usize = 8;
const DEFAULT_REORDER_BUFFER_BYTES: usize = 16 << 20;
const MAX_AUTO_REORDER_BUFFER_ROWS: usize = 1 << 10;
const MIN_AUTO_REORDER_BUFFER_ROWS: usize = 32;
const REORDER_BUFFER_ENV: &str = "JX_IMP_REORDER_MARKERS";

pub(crate) struct ImpReader {
    samples: Vec<String>,
    inner: ReaderKind,
    row_index: usize,
    phase_provenance: PhaseProvenance,
    phase_set_interner: PhaseSetInterner,
    reorder: VariantReorderBuffer,
    vcf_parse_pool: Option<rayon::ThreadPool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PhaseProvenance {
    Vcf,
    Pgen,
    Other,
}

#[derive(Debug, Default)]
pub(crate) struct PhaseSetInterner {
    ids: HashMap<String, PhaseSetId>,
    next: u32,
}

impl PhaseSetInterner {
    fn intern(&mut self, value: &str) -> Result<PhaseSetId, String> {
        if let Some(id) = self.ids.get(value) {
            return Ok(*id);
        }
        let next = self
            .next
            .checked_add(1)
            .ok_or_else(|| "VCF PS identifier space exhausted".to_owned())?;
        self.next = next;
        let id = PhaseSetId::new(next);
        self.ids.insert(value.to_owned(), id);
        Ok(id)
    }
}

enum ReaderKind {
    Vcf {
        path: PathBuf,
        reader: Box<dyn BufRead + Send + Sync>,
        line_no: usize,
        line: String,
        pending: VecDeque<VariantRow>,
    },
    HapMap {
        path: PathBuf,
        reader: Box<dyn BufRead + Send + Sync>,
        line_no: usize,
    },
    Plink(PlinkReader),
    Numeric(ImpNumericReader),
}

enum PlinkGenotypes {
    Bed {
        file: File,
        bytes_per_variant: usize,
        packed_rows: Vec<u8>,
        chunk_start: usize,
        chunk_len: usize,
        calls: Vec<Option<DiploidCalls>>,
    },
    Pgen {
        reader: PgenReader,
        packed_rows: Vec<u8>,
        chunk_start: usize,
        chunk_len: usize,
        phase_present: Vec<u64>,
        phase_orientation: Vec<u64>,
        calls: Vec<Option<DiploidCalls>>,
    },
}

struct PlinkReader {
    path: PathBuf,
    sites: BimChunkReader,
    genotypes: PlinkGenotypes,
    n_variants: usize,
    n_samples: usize,
    next_variant: usize,
    sites_checked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ReorderKey {
    chrom_rank: usize,
    position: u64,
    source_index: u64,
}

struct BufferedVariantRow {
    key: ReorderKey,
    row: VariantRow,
}

impl PartialEq for BufferedVariantRow {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for BufferedVariantRow {}

impl PartialOrd for BufferedVariantRow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BufferedVariantRow {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

struct VariantReorderBuffer {
    capacity: usize,
    chrom_ranks: HashMap<String, usize>,
    next_source_index: u64,
    heap: BinaryHeap<Reverse<BufferedVariantRow>>,
    input_exhausted: bool,
    last_emitted: Option<ReorderKey>,
}

impl VariantReorderBuffer {
    fn new(samples: usize) -> Result<Self, String> {
        let capacity = match std::env::var(REORDER_BUFFER_ENV) {
            Ok(value) if !value.trim().is_empty() => value
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    format!("{REORDER_BUFFER_ENV} must be a positive integer marker count")
                })?,
            _ => {
                // Packed biallelic calls use two bits per sample. PS metadata
                // and occasional complex calls add memory, so budget
                // conservatively while keeping a row cap for wide panels.
                let bytes_per_row = samples.saturating_mul(6).saturating_add(512);
                (DEFAULT_REORDER_BUFFER_BYTES / bytes_per_row.max(1))
                    .clamp(MIN_AUTO_REORDER_BUFFER_ROWS, MAX_AUTO_REORDER_BUFFER_ROWS)
            }
        };
        Ok(Self {
            capacity,
            chrom_ranks: HashMap::new(),
            next_source_index: 0,
            heap: BinaryHeap::new(),
            input_exhausted: false,
            last_emitted: None,
        })
    }

    fn push(&mut self, row: VariantRow) {
        let next_rank = self.chrom_ranks.len();
        let chrom_rank = *self
            .chrom_ranks
            .entry(row.meta.chrom.clone())
            .or_insert(next_rank);
        let key = ReorderKey {
            chrom_rank,
            position: row.meta.pos,
            source_index: self.next_source_index,
        };
        self.next_source_index = self.next_source_index.saturating_add(1);
        self.heap.push(Reverse(BufferedVariantRow { key, row }));
    }

    fn pop(&mut self) -> Result<Option<VariantRow>, String> {
        let Some(Reverse(buffered)) = self.heap.pop() else {
            return Ok(None);
        };
        if let Some(previous) = self.last_emitted {
            if buffered.key < previous {
                return Err(format!(
                    "input positions are too disordered for the {}-marker reorder buffer: \
                     chromosome-order rank {} position {} follows rank {} position {}; \
                     sort the input or increase {REORDER_BUFFER_ENV}",
                    self.capacity,
                    buffered.key.chrom_rank,
                    buffered.key.position,
                    previous.chrom_rank,
                    previous.position,
                ));
            }
        }
        self.last_emitted = Some(buffered.key);
        Ok(Some(buffered.row))
    }

    fn has_capacity(&self) -> bool {
        self.heap.len() < self.capacity
    }
}

impl ImpReader {
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        if !path.exists() && !looks_like_genotype_prefix(path) {
            return Err(format!("genotype input not found: {}", path.display()));
        }
        let lower = path.to_string_lossy().to_ascii_lowercase();
        let ext = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if lower.ends_with(".vcf") || lower.ends_with(".vcf.gz") {
            return Self::open_vcf(path);
        }
        if lower.ends_with(".hmp")
            || lower.ends_with(".hmp.txt")
            || lower.ends_with(".hmp.gz")
            || lower.ends_with(".hmp.txt.gz")
            || ext == "hmp"
        {
            return Self::open_hapmap(path);
        }
        if pgen::path_uses_pgen(&path.to_string_lossy()) {
            return Self::open_plink(path, true);
        }
        if matches!(ext.as_str(), "bed" | "bim" | "fam") || looks_like_genotype_prefix(path) {
            return Self::open_plink(path, false);
        }
        let numeric = ImpNumericReader::open(path)?;
        let samples = numeric.samples().to_vec();
        let reorder = VariantReorderBuffer::new(samples.len())?;
        Ok(Self {
            samples,
            inner: ReaderKind::Numeric(numeric),
            row_index: 0,
            phase_provenance: PhaseProvenance::Other,
            phase_set_interner: PhaseSetInterner::default(),
            reorder,
            vcf_parse_pool: None,
        })
    }

    pub(crate) fn sample_ids(&self) -> &[String] {
        &self.samples
    }

    pub(crate) fn phase_provenance(&self) -> PhaseProvenance {
        self.phase_provenance
    }

    pub(crate) fn set_vcf_parse_threads(&mut self, workers: usize) -> Result<(), String> {
        if !matches!(self.inner, ReaderKind::Vcf { .. }) {
            return Ok(());
        }
        let requested = std::env::var("JX_IMP_VCF_PARSE_THREADS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or_else(|| workers.min(MAX_VCF_PARSE_THREADS));
        let threads = requested.min(MAX_VCF_PARSE_THREADS);
        self.vcf_parse_pool = if threads <= 1 {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|index| format!("imp-vcf-parse-{index}"))
                    .build()
                    .map_err(|error| format!("cannot create VCF parsing thread pool: {error}"))?,
            )
        };
        Ok(())
    }

    pub(crate) fn next_row(&mut self) -> Result<Option<VariantRow>, String> {
        self.next_ordered_row(true)
    }

    /// Read a row with validated metadata and sample count, leaving allele
    /// validation to the reference stream's combined phased-call check.
    pub(crate) fn next_biallelic_reference_row(&mut self) -> Result<Option<VariantRow>, String> {
        self.next_ordered_row(false)
    }

    fn next_ordered_row(&mut self, validate_calls: bool) -> Result<Option<VariantRow>, String> {
        while !self.reorder.input_exhausted && self.reorder.has_capacity() {
            match self.next_row_raw(validate_calls)? {
                Some(row) => self.reorder.push(row),
                None => self.reorder.input_exhausted = true,
            }
        }
        self.reorder.pop()
    }

    fn next_row_raw(&mut self, validate_calls: bool) -> Result<Option<VariantRow>, String> {
        let row = match &mut self.inner {
            ReaderKind::Vcf {
                path,
                reader,
                line_no,
                line,
                pending,
            } => next_vcf_row(
                path,
                reader.as_mut(),
                line_no,
                line,
                pending,
                self.vcf_parse_pool.as_ref(),
                self.samples.len(),
                &mut self.phase_set_interner,
            )?,
            ReaderKind::HapMap {
                path,
                reader,
                line_no,
            } => next_hmp_row(path, reader.as_mut(), line_no, self.samples.len())?,
            ReaderKind::Plink(plink) => plink.next_row()?,
            ReaderKind::Numeric(numeric) => match numeric.next_row()? {
                Some((values, site)) => Some(numeric_row(values, site)?),
                None => None,
            },
        };
        if let Some(row) = row {
            let validation = if validate_calls {
                row.validate(self.samples.len())
            } else {
                row.validate_shape(self.samples.len())
            };
            validation.map_err(|err| format!("variant {}: {err}", self.row_index + 1))?;
            self.row_index += 1;
            Ok(Some(row))
        } else {
            Ok(None)
        }
    }

    fn open_vcf(path: &Path) -> Result<Self, String> {
        let mut reader = gfcore::open_text_maybe_gz(path)?;
        let mut line = String::new();
        let mut header_lines = 0usize;
        let samples = loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if n == 0 {
                return Err(format!("{}: no #CHROM header found in VCF", path.display()));
            }
            header_lines += 1;
            if line.starts_with("#CHROM") {
                let fields = line.trim_end().split('\t').collect::<Vec<_>>();
                if fields.len() < 10 || fields[8] != "FORMAT" {
                    return Err(format!(
                        "{}: VCF #CHROM header must contain FORMAT and at least one sample",
                        path.display()
                    ));
                }
                let samples = fields[9..]
                    .iter()
                    .map(|value| (*value).to_owned())
                    .collect::<Vec<_>>();
                if samples.iter().any(String::is_empty) {
                    return Err(format!(
                        "{}: VCF contains an empty sample ID",
                        path.display()
                    ));
                }
                break samples;
            }
            if !line.starts_with('#') {
                return Err(format!(
                    "{}: VCF data appears before the #CHROM header",
                    path.display()
                ));
            }
        };
        if samples.is_empty() {
            return Err(format!("{}: VCF has no samples", path.display()));
        }
        let reorder = VariantReorderBuffer::new(samples.len())?;
        Ok(Self {
            samples,
            inner: ReaderKind::Vcf {
                path: path.to_path_buf(),
                reader,
                line_no: header_lines,
                line,
                pending: VecDeque::new(),
            },
            row_index: 0,
            phase_provenance: PhaseProvenance::Vcf,
            phase_set_interner: PhaseSetInterner::default(),
            reorder,
            vcf_parse_pool: None,
        })
    }

    fn open_hapmap(path: &Path) -> Result<Self, String> {
        let mut reader = gfcore::open_text_maybe_gz(path)?;
        let mut line = String::new();
        let mut header_lines = 0usize;
        let samples = loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if n == 0 {
                return Err(format!("{}: no HapMap header found", path.display()));
            }
            header_lines += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let fields = split_hmp(trimmed);
            if fields.len() < 12 {
                return Err(format!(
                    "{}: HapMap header requires at least 12 columns",
                    path.display()
                ));
            }
            break fields[11..]
                .iter()
                .map(|value| (*value).to_owned())
                .collect::<Vec<_>>();
        };
        if samples.is_empty() {
            return Err(format!("{}: HapMap has no sample columns", path.display()));
        }
        let reorder = VariantReorderBuffer::new(samples.len())?;
        Ok(Self {
            samples,
            inner: ReaderKind::HapMap {
                path: path.to_path_buf(),
                reader,
                line_no: header_lines,
            },
            row_index: 0,
            phase_provenance: PhaseProvenance::Other,
            phase_set_interner: PhaseSetInterner::default(),
            reorder,
            vcf_parse_pool: None,
        })
    }

    fn open_plink(path: &Path, is_pgen: bool) -> Result<Self, String> {
        let path_string = path.to_string_lossy();
        let samples = gfcore::read_fam(&path_string)?;
        if samples.is_empty() {
            return Err(format!("{}: PLINK sample file is empty", path.display()));
        }
        let sites = BimChunkReader::open(&path_string)?;
        let n_samples = samples.len();
        let (n_variants, genotypes, source_path) = if is_pgen {
            let prefix = pgen::normalize_pgen_prefix(&path_string);
            let n_variants = pgen::count_pgen_variants(&prefix, n_samples)?;
            let pgen_path = PathBuf::from(format!("{prefix}.pgen"));
            let reader = PgenReader::open(&pgen_path, n_samples, n_variants)?;
            let packed_chunk_bytes = n_samples
                .div_ceil(4)
                .checked_mul(PGEN_READ_CHUNK_VARIANTS)
                .ok_or_else(|| "PGEN genotype chunk size overflow".to_owned())?;
            (
                n_variants,
                PlinkGenotypes::Pgen {
                    reader,
                    packed_rows: vec![0; packed_chunk_bytes],
                    chunk_start: 0,
                    chunk_len: 0,
                    phase_present: Vec::new(),
                    phase_orientation: Vec::new(),
                    calls: Vec::new(),
                },
                pgen_path,
            )
        } else {
            let prefix = strip_plink_suffix(path);
            let bed_path = PathBuf::from(format!("{}.bed", prefix.to_string_lossy()));
            let mut file =
                File::open(&bed_path).map_err(|e| format!("{}: {e}", bed_path.display()))?;
            let mut header = [0u8; 3];
            file.read_exact(&mut header)
                .map_err(|e| format!("{}: read BED header: {e}", bed_path.display()))?;
            if header != [0x6c, 0x1b, 0x01] {
                return Err(format!(
                    "{}: expected variant-major PLINK BED header 6c 1b 01",
                    bed_path.display()
                ));
            }
            let bytes_per_variant = n_samples.div_ceil(4);
            let payload_bytes = file
                .metadata()
                .map_err(|e| e.to_string())?
                .len()
                .checked_sub(3)
                .ok_or_else(|| format!("{}: truncated BED header", bed_path.display()))?;
            if bytes_per_variant == 0 || payload_bytes % bytes_per_variant as u64 != 0 {
                return Err(format!(
                    "{}: genotype payload size is not divisible by bytes per variant ({bytes_per_variant})",
                    bed_path.display()
                ));
            }
            let n_variants = usize::try_from(payload_bytes / bytes_per_variant as u64)
                .map_err(|_| "BED variant count is too large for this platform".to_string())?;
            if n_variants == 0 {
                return Err(format!(
                    "{}: BED contains zero variants",
                    bed_path.display()
                ));
            }
            file.seek(SeekFrom::Start(3)).map_err(|e| e.to_string())?;
            (
                n_variants,
                PlinkGenotypes::Bed {
                    file,
                    bytes_per_variant,
                    packed_rows: vec![
                        0;
                        bytes_per_variant
                            .checked_mul(PGEN_READ_CHUNK_VARIANTS)
                            .ok_or_else(
                                || "BED genotype chunk size overflow".to_owned()
                            )?
                    ],
                    chunk_start: 0,
                    chunk_len: 0,
                    calls: Vec::new(),
                },
                bed_path,
            )
        };
        let reorder = VariantReorderBuffer::new(samples.len())?;
        Ok(Self {
            samples,
            inner: ReaderKind::Plink(PlinkReader {
                path: source_path,
                sites,
                genotypes,
                n_variants,
                n_samples,
                next_variant: 0,
                sites_checked: false,
            }),
            row_index: 0,
            phase_provenance: if is_pgen {
                PhaseProvenance::Pgen
            } else {
                PhaseProvenance::Other
            },
            phase_set_interner: PhaseSetInterner::default(),
            reorder,
            vcf_parse_pool: None,
        })
    }
}

impl PlinkReader {
    fn next_row(&mut self) -> Result<Option<VariantRow>, String> {
        if self.next_variant >= self.n_variants {
            if !self.sites_checked {
                self.sites_checked = true;
                if self.sites.next_site()?.is_some() {
                    return Err(format!(
                        "{}: variant metadata contains more rows than genotype data",
                        self.path.display()
                    ));
                }
            }
            return Ok(None);
        }
        let idx = self.next_variant;
        let site = self.sites.next_site()?.ok_or_else(|| {
            format!(
                "{}: missing metadata for variant {}",
                self.path.display(),
                idx + 1
            )
        })?;
        let n_samples = self.n_samples;
        let n_variants = self.n_variants;
        let calls = match &mut self.genotypes {
            PlinkGenotypes::Bed {
                file,
                bytes_per_variant,
                packed_rows,
                chunk_start,
                chunk_len,
                calls,
            } => {
                if idx < *chunk_start || idx >= chunk_start.saturating_add(*chunk_len) {
                    *chunk_start = idx;
                    *chunk_len = (n_variants - idx).min(PGEN_READ_CHUNK_VARIANTS);
                    let byte_len = bytes_per_variant
                        .checked_mul(*chunk_len)
                        .ok_or_else(|| "BED genotype chunk size overflow".to_owned())?;
                    file.read_exact(&mut packed_rows[..byte_len]).map_err(|e| {
                        format!(
                            "{}: variants {}-{}: {e}",
                            self.path.display(),
                            idx + 1,
                            idx + *chunk_len
                        )
                    })?;
                    calls.resize_with(*chunk_len, || None);
                    decode_packed_bed_rows(
                        &packed_rows[..byte_len],
                        *bytes_per_variant,
                        *chunk_len,
                        n_samples,
                        calls,
                    )?;
                }
                let local_variant = idx - *chunk_start;
                calls
                    .get_mut(local_variant)
                    .and_then(Option::take)
                    .ok_or_else(|| {
                        format!(
                            "{}: missing decoded BED genotype call for variant {}",
                            self.path.display(),
                            idx + 1
                        )
                    })?
            }
            PlinkGenotypes::Pgen {
                reader,
                packed_rows,
                chunk_start,
                chunk_len,
                phase_present,
                phase_orientation,
                calls,
            } => {
                if idx < *chunk_start || idx >= chunk_start.saturating_add(*chunk_len) {
                    *chunk_start = idx;
                    *chunk_len = reader.read_source_block(
                        idx,
                        (n_variants - idx).min(PGEN_READ_CHUNK_VARIANTS),
                        packed_rows,
                    )?;
                    let bytes_per_variant = n_samples.div_ceil(4);
                    let phase_words = n_samples.div_ceil(64);
                    phase_present.resize(*chunk_len * phase_words, 0);
                    phase_orientation.resize(*chunk_len * phase_words, 0);
                    calls.resize_with(*chunk_len, || None);
                    reader.read_source_block_phase_bits(
                        idx,
                        *chunk_len,
                        &packed_rows[..*chunk_len * bytes_per_variant],
                        phase_present,
                        phase_orientation,
                    )?;
                    calls.par_iter_mut().enumerate().try_for_each(
                        |(row, output)| -> Result<(), String> {
                            let packed_start = row * bytes_per_variant;
                            let packed_end = packed_start + bytes_per_variant;
                            let phase_start = row * phase_words;
                            let phase_end = phase_start + phase_words;
                            *output = Some(DiploidCalls::from_packed_biallelic_phase_bits(
                                &packed_rows[packed_start..packed_end],
                                &phase_present[phase_start..phase_end],
                                &phase_orientation[phase_start..phase_end],
                                n_samples,
                            )?);
                            Ok(())
                        },
                    )?;
                }
                let local_variant = idx - *chunk_start;
                calls
                    .get_mut(local_variant)
                    .and_then(Option::take)
                    .ok_or_else(|| {
                        format!(
                            "{}: missing decoded genotype call for variant {}",
                            self.path.display(),
                            idx + 1
                        )
                    })?
            }
        };
        self.next_variant += 1;
        Ok(Some(VariantRow {
            meta: site_to_meta(site),
            calls,
        }))
    }
}

fn decode_packed_bed_rows(
    packed_rows: &[u8],
    bytes_per_variant: usize,
    variant_count: usize,
    n_samples: usize,
    calls: &mut [Option<DiploidCalls>],
) -> Result<(), String> {
    let expected_bytes = bytes_per_variant
        .checked_mul(variant_count)
        .ok_or_else(|| "BED genotype chunk size overflow".to_owned())?;
    if packed_rows.len() < expected_bytes || calls.len() < variant_count {
        return Err("BED genotype chunk dimensions are inconsistent".to_owned());
    }
    calls[..variant_count]
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(variant, output)| -> Result<(), String> {
            let start = variant * bytes_per_variant;
            let end = start + bytes_per_variant;
            *output = Some(DiploidCalls::from_packed_biallelic_row(
                &packed_rows[start..end],
                None,
                n_samples,
            )?);
            Ok(())
        })
}

struct ParsedVcfRow {
    row: VariantRow,
    phase_sets: Option<Vec<Option<String>>>,
}

fn next_vcf_row(
    path: &Path,
    reader: &mut dyn BufRead,
    line_no: &mut usize,
    line: &mut String,
    pending: &mut VecDeque<VariantRow>,
    parse_pool: Option<&rayon::ThreadPool>,
    n_samples: usize,
    phase_set_interner: &mut PhaseSetInterner,
) -> Result<Option<VariantRow>, String> {
    if let Some(row) = pending.pop_front() {
        return Ok(Some(row));
    }
    let Some(parsed_rows) = read_vcf_block(reader, path, line_no, line, n_samples, parse_pool)?
    else {
        return Ok(None);
    };
    for parsed_row in parsed_rows {
        pending.push_back(intern_vcf_phase_sets(parsed_row, phase_set_interner)?);
    }
    Ok(pending.pop_front())
}

fn read_vcf_block(
    reader: &mut dyn BufRead,
    path: &Path,
    line_no: &mut usize,
    line: &mut String,
    n_samples: usize,
    parse_pool: Option<&rayon::ThreadPool>,
) -> Result<Option<Vec<ParsedVcfRow>>, String> {
    let mut block = Vec::with_capacity(VCF_PARSE_BLOCK_ROWS);
    while block.len() < VCF_PARSE_BLOCK_ROWS {
        line.clear();
        let n = reader
            .read_line(line)
            .map_err(|e| format!("{}:{}: {e}", path.display(), *line_no + 1))?;
        if n == 0 {
            break;
        }
        *line_no += 1;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        block.push((*line_no, std::mem::take(line)));
    }
    if block.is_empty() {
        return Ok(None);
    }
    let parsed = if let Some(pool) = parse_pool {
        pool.install(|| {
            block
                .into_par_iter()
                .map(|(source_line, record)| {
                    parse_vcf_record(path, source_line, &record, n_samples)
                })
                .collect::<Vec<_>>()
        })
    } else {
        block
            .into_iter()
            .map(|(source_line, record)| parse_vcf_record(path, source_line, &record, n_samples))
            .collect::<Vec<_>>()
    };
    parsed
        .into_iter()
        .collect::<Result<Vec<_>, String>>()
        .map(Some)
}

fn intern_vcf_phase_sets(
    parsed_row: ParsedVcfRow,
    phase_set_interner: &mut PhaseSetInterner,
) -> Result<VariantRow, String> {
    let ParsedVcfRow {
        mut row,
        phase_sets,
    } = parsed_row;
    if let Some(raw_phase_sets) = phase_sets {
        for (sample, phase_set) in raw_phase_sets.into_iter().enumerate() {
            let phase_set = phase_set
                .as_deref()
                .map(|value| phase_set_interner.intern(value))
                .transpose()?;
            row.calls.set_phase_set(sample, phase_set)?;
        }
    }
    Ok(row)
}

fn parse_vcf_record(
    path: &Path,
    line_no: usize,
    line: &str,
    n_samples: usize,
) -> Result<ParsedVcfRow, String> {
    let mut fields = line.trim_end().split('\t');
    let (
        Some(chrom),
        Some(pos_text),
        Some(id),
        Some(ref_allele),
        Some(alt),
        Some(_qual),
        Some(_filter),
        Some(_info),
        Some(format),
    ) = (
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
    )
    else {
        return Err(vcf_column_count_error(path, line_no, line, n_samples));
    };
    let pos = pos_text.parse::<u64>().map_err(|_| {
        format!(
            "{}:{}: invalid VCF position {}",
            path.display(),
            line_no,
            pos_text
        )
    })?;
    if alt.is_empty() || alt == "." || alt.split(',').any(str::is_empty) {
        return Err(format!(
            "{}:{}: VCF record has an empty or missing ALT allele",
            path.display(),
            line_no
        ));
    }
    let gt_idx = format
        .split(':')
        .position(|field| field == "GT")
        .ok_or_else(|| format!("{}:{}: FORMAT does not contain GT", path.display(), line_no))?;
    let ps_idx = format.split(':').position(|field| field == "PS");
    let gt_is_first_field = gt_idx == 0;
    let mut calls = DiploidCalls::empty(n_samples);
    let mut phase_sets = ps_idx.map(|_| vec![None; n_samples]);
    if format == "GT" {
        let mut sample_idx = 0usize;
        while sample_idx < n_samples {
            let first = fields
                .next()
                .ok_or_else(|| vcf_column_count_error(path, line_no, line, n_samples))?;
            let first_code = simple_vcf_gt_code(first);
            if sample_idx + 1 < n_samples {
                let second = fields
                    .next()
                    .ok_or_else(|| vcf_column_count_error(path, line_no, line, n_samples))?;
                let second_code = simple_vcf_gt_code(second);
                if let (Some(first_code), Some(second_code)) = (first_code, second_code) {
                    calls.set_common_call_pair_unchecked(sample_idx, first_code, second_code);
                } else {
                    let first_call = parse_vcf_gt(first).map_err(|err| {
                        format!(
                            "{}:{} sample {}: {err}",
                            path.display(),
                            line_no,
                            sample_idx + 1
                        )
                    })?;
                    let second_call = parse_vcf_gt(second).map_err(|err| {
                        format!(
                            "{}:{} sample {}: {err}",
                            path.display(),
                            line_no,
                            sample_idx + 2
                        )
                    })?;
                    calls.set_call_unchecked(sample_idx, first_call);
                    calls.set_call_unchecked(sample_idx + 1, second_call);
                }
            } else {
                let call = parse_vcf_gt(first).map_err(|err| {
                    format!(
                        "{}:{} sample {}: {err}",
                        path.display(),
                        line_no,
                        sample_idx + 1
                    )
                })?;
                if let Some(code) = first_code {
                    calls.set_common_call_code_unchecked(sample_idx, code);
                } else {
                    calls.set_call_unchecked(sample_idx, call);
                }
            }
            sample_idx += 2;
        }
    } else {
        for sample_idx in 0..n_samples {
            let Some(sample_fields) = fields.next() else {
                return Err(vcf_column_count_error(path, line_no, line, n_samples));
            };
            let gt = if gt_is_first_field {
                sample_fields
                    .split_once(':')
                    .map_or(sample_fields, |(gt, _)| gt)
            } else {
                nth_colon_field(sample_fields, gt_idx).unwrap_or(".")
            };
            let call = parse_vcf_gt(gt).map_err(|err| {
                format!(
                    "{}:{} sample {}: {err}",
                    path.display(),
                    line_no,
                    sample_idx + 1
                )
            })?;
            calls.set_call_unchecked(sample_idx, call);
            if let (Some(ps_idx), Some(phase_sets)) = (ps_idx, phase_sets.as_mut()) {
                phase_sets[sample_idx] = nth_colon_field(sample_fields, ps_idx)
                    .map(str::trim)
                    .filter(|value| !value.is_empty() && *value != ".")
                    .map(str::to_owned);
            }
        }
    }
    if fields.next().is_some() {
        return Err(vcf_column_count_error(path, line_no, line, n_samples));
    }
    let alt_alleles = if alt.contains(',') {
        alt.split(',').map(str::to_owned).collect()
    } else {
        vec![alt.to_owned()]
    };
    let id = if id == "." || id.is_empty() {
        format!("{chrom}_{pos}")
    } else {
        id.to_owned()
    };
    Ok(ParsedVcfRow {
        row: VariantRow {
            meta: VariantMeta {
                chrom: chrom.to_owned(),
                pos,
                id,
                ref_allele: ref_allele.to_owned(),
                alt_alleles,
            },
            calls,
        },
        phase_sets,
    })
}

fn vcf_column_count_error(path: &Path, line_no: usize, line: &str, n_samples: usize) -> String {
    format!(
        "{}:{}: VCF record has {} columns; expected {}",
        path.display(),
        line_no,
        line.trim_end().split('\t').count(),
        9 + n_samples
    )
}

fn nth_colon_field(value: &str, index: usize) -> Option<&str> {
    value.split(':').nth(index)
}

#[inline(always)]
fn simple_vcf_gt_code(gt: &str) -> Option<u8> {
    let bytes = gt.as_bytes();
    if bytes.len() != 3 {
        return None;
    }
    match (bytes[0], bytes[1], bytes[2]) {
        (b'0', b'|', b'0') | (b'0', b'/', b'0') => Some(0),
        (b'0', b'/', b'1') | (b'1', b'/', b'0') => Some(1),
        (b'0', b'|', b'1') => Some(2),
        (b'1', b'|', b'0') => Some(3),
        (b'1', b'|', b'1') | (b'1', b'/', b'1') => Some(4),
        (b'.', b'|', b'.') | (b'.', b'/', b'.') => Some(5),
        _ => None,
    }
}

fn parse_vcf_gt(gt: &str) -> Result<DiploidCall, String> {
    let bytes = gt.as_bytes();
    if bytes.len() == 3 && matches!(bytes[1], b'|' | b'/') {
        let parse_simple_allele = |allele| match allele {
            b'0' => Some(Some(0)),
            b'1' => Some(Some(1)),
            b'.' => Some(None),
            _ => None,
        };
        if let (Some(allele0), Some(allele1)) =
            (parse_simple_allele(bytes[0]), parse_simple_allele(bytes[2]))
        {
            return Ok(DiploidCall {
                alleles: [allele0, allele1],
                phased: bytes[1] == b'|',
            });
        }
    }
    let Some(separator_index) = bytes.iter().position(|byte| matches!(*byte, b'|' | b'/')) else {
        return Err(format!(
            "expected diploid GT with / or | separator, got {gt:?}"
        ));
    };
    if bytes[separator_index + 1..]
        .iter()
        .any(|byte| matches!(*byte, b'|' | b'/'))
    {
        return Err(format!("expected diploid GT, got {gt:?}"));
    }
    let parse_allele = |allele: &str| -> Result<Option<u16>, String> {
        if allele == "." {
            return Ok(None);
        }
        allele
            .parse::<u16>()
            .map(Some)
            .map_err(|_| format!("invalid VCF allele index {allele:?}"))
    };
    Ok(DiploidCall {
        alleles: [
            parse_allele(&gt[..separator_index])?,
            parse_allele(&gt[separator_index + 1..])?,
        ],
        phased: bytes[separator_index] == b'|',
    })
}

fn next_hmp_row(
    path: &Path,
    reader: &mut dyn BufRead,
    line_no: &mut usize,
    n_samples: usize,
) -> Result<Option<VariantRow>, String> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("{}:{}: {e}", path.display(), *line_no + 1))?;
        if n == 0 {
            return Ok(None);
        }
        *line_no += 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields = split_hmp(trimmed);
        if fields.len() != 11 + n_samples {
            return Err(format!(
                "{}:{}: HapMap row has {} fields; expected {}",
                path.display(),
                *line_no,
                fields.len(),
                11 + n_samples
            ));
        }
        let (ref_allele, alt_allele) = parse_hmp_alleles(fields[1]).ok_or_else(|| {
            format!(
                "{}:{}: could not determine a biallelic SNP allele pair from {:?}",
                path.display(),
                *line_no,
                fields[1]
            )
        })?;
        let pos = fields[3].parse::<u64>().map_err(|_| {
            format!(
                "{}:{}: invalid HapMap position {}",
                path.display(),
                *line_no,
                fields[3]
            )
        })?;
        let calls = fields[11..]
            .iter()
            .enumerate()
            .map(|(sample_idx, gt)| {
                parse_hmp_gt(gt, ref_allele, alt_allele).map_err(|err| {
                    format!(
                        "{}:{} sample {}: {err}",
                        path.display(),
                        *line_no,
                        sample_idx + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Some(VariantRow {
            meta: VariantMeta {
                chrom: fields[2].to_owned(),
                pos,
                id: if fields[0].is_empty() || fields[0] == "." {
                    format!("{}_{}", fields[2], fields[3])
                } else {
                    fields[0].to_owned()
                },
                ref_allele: ref_allele.to_string(),
                alt_alleles: vec![alt_allele.to_string()],
            },
            calls: calls.into(),
        }));
    }
}

fn numeric_row(values: Vec<f32>, site: SiteInfo) -> Result<VariantRow, String> {
    if site.alt_allele.contains(',') {
        return Err("numeric hard-call input requires biallelic site metadata".to_owned());
    }
    let calls = values
        .into_iter()
        .enumerate()
        .map(|(sample_idx, value)| {
            if value == -9.0 || value.is_nan() {
                Ok(DiploidCall::missing(false))
            } else if value == 0.0 {
                Ok(DiploidCall { alleles: [Some(0), Some(0)], phased: false })
            } else if value == 1.0 {
                Ok(DiploidCall { alleles: [Some(0), Some(1)], phased: false })
            } else if value == 2.0 {
                Ok(DiploidCall { alleles: [Some(1), Some(1)], phased: false })
            } else {
                Err(format!(
                    "numeric hard-call input contains {value} for sample {}; expected exactly 0, 1, 2, or a missing value",
                    sample_idx + 1
                ))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(VariantRow {
        meta: site_to_meta(site),
        calls: DiploidCalls::from(calls),
    })
}

fn parse_hmp_alleles(field: &str) -> Option<(char, char)> {
    let mut bases = Vec::new();
    for part in field.split(|ch: char| matches!(ch, '/' | '|' | ',' | ';')) {
        for base in part
            .chars()
            .filter(|base| matches!(base.to_ascii_uppercase(), 'A' | 'C' | 'G' | 'T'))
        {
            let base = base.to_ascii_uppercase();
            if !bases.contains(&base) {
                bases.push(base);
            }
        }
    }
    if bases.len() == 2 {
        Some((bases[0], bases[1]))
    } else {
        None
    }
}

fn parse_hmp_gt(token: &str, ref_allele: char, alt_allele: char) -> Result<DiploidCall, String> {
    let value = token.trim().to_ascii_uppercase();
    if value.is_empty() || matches!(value.as_str(), "N" | "NN" | "--" | "??" | "." | "./.") {
        return Ok(DiploidCall::missing(false));
    }
    let alleles = if value.len() == 1 {
        let base = value.chars().next().unwrap();
        iupac_pair(base).unwrap_or((base, base))
    } else if value.len() == 2 {
        let mut chars = value.chars();
        (chars.next().unwrap(), chars.next().unwrap())
    } else {
        return Err(format!("unsupported HapMap genotype token {token:?}"));
    };
    let idx = |base| {
        if base == ref_allele {
            Some(0)
        } else if base == alt_allele {
            Some(1)
        } else {
            None
        }
    };
    match (idx(alleles.0), idx(alleles.1)) {
        (Some(a), Some(b)) => Ok(DiploidCall {
            alleles: [Some(a), Some(b)],
            phased: false,
        }),
        _ => Ok(DiploidCall::missing(false)),
    }
}

fn iupac_pair(code: char) -> Option<(char, char)> {
    match code {
        'R' => Some(('A', 'G')),
        'Y' => Some(('C', 'T')),
        'S' => Some(('G', 'C')),
        'W' => Some(('A', 'T')),
        'K' => Some(('G', 'T')),
        'M' => Some(('A', 'C')),
        _ => None,
    }
}

fn split_hmp(line: &str) -> Vec<&str> {
    let tab = line.split('\t').collect::<Vec<_>>();
    if tab.len() >= 12 {
        tab
    } else {
        line.split_whitespace().collect()
    }
}

fn site_to_meta(site: SiteInfo) -> VariantMeta {
    VariantMeta {
        chrom: site.chrom,
        pos: site.pos.max(0) as u64,
        id: site.snp,
        ref_allele: site.ref_allele,
        alt_alleles: site.alt_allele.split(',').map(str::to_owned).collect(),
    }
}

fn strip_plink_suffix(path: &Path) -> PathBuf {
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    if matches!(ext.to_ascii_lowercase().as_str(), "bed" | "bim" | "fam") {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

fn looks_like_genotype_prefix(path: &Path) -> bool {
    if crate::pgen::path_uses_pgen(&path.to_string_lossy()) {
        return true;
    }
    let path_text = path.to_string_lossy();
    ["bed", "bim", "fam"].iter().any(|extension| {
        path.with_extension(extension).is_file()
            || PathBuf::from(format!("{path_text}.{extension}")).is_file()
    })
}

#[cfg(test)]
mod vcf_parser_tests {
    use super::{
        decode_packed_bed_rows, intern_vcf_phase_sets, nth_colon_field, parse_vcf_gt,
        read_vcf_block, DiploidCall, ImpReader, PhaseProvenance, PhaseSetInterner,
    };
    use std::fmt::Write as _;
    use std::io::Cursor;
    use std::path::Path;

    #[test]
    fn vcf_parser_finds_gt_after_other_fields() {
        let input = b"1\t42\trs42\tA\tG\t.\tPASS\t.\tDP:GT:DS\t10:0|1:0.5\t2:./1:.\n1\t43\trs43\tG\tA\t.\tPASS\t.\tGT:DP\t1|0:10\t0|0:2\n";
        let mut reader = Cursor::new(input);
        let mut line_no = 0;
        let mut line = String::new();
        let mut phase_sets = PhaseSetInterner::default();

        let parsed = read_vcf_block(
            &mut reader,
            Path::new("memory.vcf"),
            &mut line_no,
            &mut line,
            2,
            None,
        )
        .unwrap()
        .unwrap();
        let mut parsed = parsed.into_iter();
        let first = intern_vcf_phase_sets(parsed.next().unwrap(), &mut phase_sets).unwrap();
        let second = intern_vcf_phase_sets(parsed.next().unwrap(), &mut phase_sets).unwrap();

        assert_eq!(first.meta.pos, 42);
        assert_eq!(
            first.calls.get(0),
            Some(DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: true,
            })
        );
        assert_eq!(
            first.calls.get(1),
            Some(DiploidCall {
                alleles: [None, Some(1)],
                phased: false,
            })
        );
        assert_eq!(second.meta.pos, 43);
        assert_eq!(nth_colon_field("DP:GT:DS", 1), Some("GT"));
        assert_eq!(nth_colon_field("DP:GT:DS", 3), None);
        assert_eq!(
            parse_vcf_gt("12|1"),
            Ok(DiploidCall {
                alleles: [Some(12), Some(1)],
                phased: true,
            })
        );
        for (gt, expected) in [
            (
                "0|0",
                DiploidCall {
                    alleles: [Some(0), Some(0)],
                    phased: true,
                },
            ),
            (
                "0|1",
                DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: true,
                },
            ),
            (
                "1|0",
                DiploidCall {
                    alleles: [Some(1), Some(0)],
                    phased: true,
                },
            ),
            (
                "1|1",
                DiploidCall {
                    alleles: [Some(1), Some(1)],
                    phased: true,
                },
            ),
            (
                "0/1",
                DiploidCall {
                    alleles: [Some(0), Some(1)],
                    phased: false,
                },
            ),
            ("./.", DiploidCall::missing(false)),
            (
                "./1",
                DiploidCall {
                    alleles: [None, Some(1)],
                    phased: false,
                },
            ),
        ] {
            assert_eq!(parse_vcf_gt(gt), Ok(expected), "GT {gt}");
        }
    }

    #[test]
    fn vcf_parser_records_phase_sets() {
        let input = b"1\t42\trs42\tA\tG\t.\tPASS\t.\tDP:GT:PS\t10:0|1:blockA\t12:1|0:blockA\n1\t43\trs43\tA\tG\t.\tPASS\t.\tDP:GT:PS\t10:0|1:blockB\t12:1|0:blockB\n";
        let mut reader = Cursor::new(input);
        let mut line_no = 0;
        let mut line = String::new();
        let mut interner = PhaseSetInterner::default();

        let parsed = read_vcf_block(
            &mut reader,
            Path::new("memory.vcf"),
            &mut line_no,
            &mut line,
            2,
            None,
        )
        .unwrap()
        .unwrap();
        let mut parsed = parsed.into_iter();
        let first = intern_vcf_phase_sets(parsed.next().unwrap(), &mut interner).unwrap();
        let second = intern_vcf_phase_sets(parsed.next().unwrap(), &mut interner).unwrap();

        assert_eq!(first.calls.phase_set(0), first.calls.phase_set(1));
        assert_eq!(second.calls.phase_set(0), second.calls.phase_set(1));
        assert_ne!(first.calls.phase_set(0), second.calls.phase_set(0));
        assert!(first.calls.has_phase_set_track());
        assert!(second.calls.has_phase_set_track());
    }

    #[test]
    fn vcf_gt_only_fast_path_preserves_common_and_complex_calls() {
        let input = b"1\t42\trs42\tA\tG\t.\tPASS\t.\tGT\t0|0\t0|1\t1|0\t1|1\t./.\t0/.\n";
        let mut reader = Cursor::new(input);
        let mut line_no = 0;
        let mut line = String::new();
        let parsed = read_vcf_block(
            &mut reader,
            Path::new("memory.vcf"),
            &mut line_no,
            &mut line,
            6,
            None,
        )
        .unwrap()
        .unwrap();
        let calls = &parsed[0].row.calls;

        assert_eq!(calls.get(0).unwrap().alleles, [Some(0), Some(0)]);
        assert_eq!(calls.get(1).unwrap().alleles, [Some(0), Some(1)]);
        assert!(calls.get(1).unwrap().phased);
        assert_eq!(calls.get(2).unwrap().alleles, [Some(1), Some(0)]);
        assert!(calls.get(2).unwrap().phased);
        assert_eq!(calls.get(3).unwrap().alleles, [Some(1), Some(1)]);
        assert_eq!(calls.get(4).unwrap().alleles, [None, None]);
        assert_eq!(calls.get(5).unwrap().alleles, [Some(0), None]);
    }

    #[test]
    fn vcf_blocks_preserve_order_and_shared_phase_set_ids() {
        let mut input = String::new();
        for position in 1..=257 {
            writeln!(
                input,
                "1\t{position}\trs{position}\tA\tG\t.\tPASS\t.\tGT:PS\t0|1:blockA\t1|0:blockA"
            )
            .unwrap();
        }
        let mut reader = Cursor::new(input);
        let mut line_no = 0;
        let mut line = String::new();
        let mut interner = PhaseSetInterner::default();

        let first_block = read_vcf_block(
            &mut reader,
            Path::new("memory.vcf"),
            &mut line_no,
            &mut line,
            2,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(first_block.len(), 256);
        let mut first_block = first_block.into_iter();
        let first = intern_vcf_phase_sets(first_block.next().unwrap(), &mut interner).unwrap();
        let last_of_first_block =
            intern_vcf_phase_sets(first_block.last().unwrap(), &mut interner).unwrap();

        let mut second_block = read_vcf_block(
            &mut reader,
            Path::new("memory.vcf"),
            &mut line_no,
            &mut line,
            2,
            None,
        )
        .unwrap()
        .unwrap()
        .into_iter();
        let first_of_second_block =
            intern_vcf_phase_sets(second_block.next().unwrap(), &mut interner).unwrap();

        assert_eq!(first.meta.pos, 1);
        assert_eq!(last_of_first_block.meta.pos, 256);
        assert_eq!(first_of_second_block.meta.pos, 257);
        assert_eq!(
            first.calls.phase_set(0),
            last_of_first_block.calls.phase_set(0)
        );
        assert_eq!(
            first.calls.phase_set(0),
            first_of_second_block.calls.phase_set(0)
        );
    }

    #[test]
    fn vcf_block_preserves_source_line_number_for_later_block_error() {
        let mut input = String::new();
        for position in 1..=256 {
            writeln!(
                input,
                "1\t{position}\trs{position}\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0"
            )
            .unwrap();
        }
        input.push_str("malformed\n");
        let mut reader = Cursor::new(input);
        let mut line_no = 0;
        let mut line = String::new();

        read_vcf_block(
            &mut reader,
            Path::new("memory.vcf"),
            &mut line_no,
            &mut line,
            2,
            None,
        )
        .unwrap()
        .unwrap();
        let error = match read_vcf_block(
            &mut reader,
            Path::new("memory.vcf"),
            &mut line_no,
            &mut line,
            2,
            None,
        ) {
            Err(error) => error,
            Ok(_) => panic!("malformed VCF row was accepted"),
        };

        assert!(error.contains("memory.vcf:257:"), "{error}");
    }

    #[test]
    fn reader_reports_vcf_phase_provenance() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "janusx-imp-provenance-{}-{unique}.vcf",
            std::process::id()
        ));
        std::fs::write(
            &path,
            concat!(
                "##fileformat=VCFv4.3\n",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n",
                "1\t1\tv1\tA\tG\t.\tPASS\t.\tGT\t0|1\n"
            ),
        )
        .unwrap();
        let reader = ImpReader::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(reader.phase_provenance(), PhaseProvenance::Vcf);
    }

    #[test]
    fn packed_bed_chunk_decodes_multiple_rows_with_one_bounded_read() {
        let packed_rows = [0xe4, 0x1b];
        let mut calls = [None, None];
        decode_packed_bed_rows(&packed_rows, 1, 2, 4, &mut calls).unwrap();

        assert_eq!(
            calls[0].as_ref().unwrap().get(2),
            Some(DiploidCall {
                alleles: [Some(0), Some(1)],
                phased: false,
            })
        );
        assert_eq!(
            calls[1].as_ref().unwrap().get(0),
            Some(DiploidCall {
                alleles: [Some(1), Some(1)],
                phased: false,
            })
        );
    }
}
