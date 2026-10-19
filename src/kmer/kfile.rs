use crate::breader::parse_bsite_header;
use crate::kmer::format::{KmergeMeta, SampleEntry, BSITE_HEADER_SIZE};
use crate::kmer::writer::read_idv_file;
use anyhow::{bail, Context, Result};
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::BoundObject;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Instant;

#[derive(Clone, Debug)]
struct KfileLayout {
    meta_path: PathBuf,
    bsite_path: PathBuf,
    idv_path: PathBuf,
    bim_path: Option<PathBuf>,
    meta: KmergeMeta,
    samples: Vec<SampleEntry>,
}

fn resolve_meta_path(prefix_or_meta: &Path) -> PathBuf {
    let raw = prefix_or_meta.to_string_lossy();
    if raw.ends_with(".meta.json") {
        prefix_or_meta.to_path_buf()
    } else {
        PathBuf::from(format!("{raw}.meta.json"))
    }
}

fn validate_meta(meta: &KmergeMeta, meta_path: &Path) -> Result<()> {
    if meta.format.trim() != "janusx-kmer-bitmatrix-v1" {
        bail!("unsupported kmerge format in {}", meta_path.display());
    }
    if meta.n_samples == 0 {
        bail!("kmerge metadata has zero samples: {}", meta_path.display());
    }
    if meta.n_kmers == 0 {
        bail!("kmerge metadata has zero k-mers: {}", meta_path.display());
    }
    if meta.matrix_layout.trim() != "column_major_bitset" {
        bail!(
            "unsupported matrix layout in {}: {}",
            meta_path.display(),
            meta.matrix_layout
        );
    }
    if meta.bit_order.trim() != "little_bit_order" {
        bail!(
            "unsupported bit order in {}: {}",
            meta_path.display(),
            meta.bit_order
        );
    }
    if meta.compression.trim() != "none" {
        bail!(
            "compressed bitmatrices are not supported in {}: {}",
            meta_path.display(),
            meta.compression
        );
    }
    let expected_bytes = meta
        .n_samples
        .checked_add(7)
        .ok_or_else(|| anyhow::anyhow!("bytes_per_col overflow in {}", meta_path.display()))?
        / 8;
    if meta.bytes_per_col != expected_bytes {
        bail!(
            "bytes_per_col mismatch in {}: metadata={} expected={}",
            meta_path.display(),
            meta.bytes_per_col,
            expected_bytes
        );
    }
    Ok(())
}

fn sibling_bim_path(meta_path: &Path) -> Option<PathBuf> {
    let name = meta_path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".meta.json")?;
    let candidate = meta_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{stem}.bim"));
    candidate.is_file().then_some(candidate)
}

fn count_bim_rows(path: &Path) -> Result<usize> {
    let reader = BufReader::new(
        File::open(path)
            .with_context(|| format!("failed to open BIM header: {}", path.display()))?,
    );
    let mut count = 0usize;
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read BIM header {}:{}",
                path.display(),
                line_no + 1
            )
        })?;
        if !line.trim().is_empty() && !line.trim_start().starts_with('#') {
            if line.split_whitespace().count() < 4 {
                bail!(
                    "malformed BIM header row at {}:{}",
                    path.display(),
                    line_no + 1
                );
            }
            count = count.saturating_add(1);
        }
    }
    Ok(count)
}

fn load_layout(prefix_or_meta: &Path) -> Result<KfileLayout> {
    let meta_path = resolve_meta_path(prefix_or_meta);
    let meta_file = File::open(&meta_path)
        .with_context(|| format!("failed to open kmerge meta file: {}", meta_path.display()))?;
    let meta: KmergeMeta = serde_json::from_reader(meta_file)
        .with_context(|| format!("failed to parse kmerge meta file: {}", meta_path.display()))?;
    validate_meta(&meta, &meta_path)?;

    let base_dir = meta_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let bsite_path = base_dir.join(&meta.bsite_file);
    let idv_path = base_dir.join(&meta.idv_file);
    let bkmer_path = if meta.bkmer_file.trim().is_empty() {
        None
    } else {
        Some(base_dir.join(&meta.bkmer_file))
    };
    let has_bkmer = bkmer_path.as_ref().is_some_and(|path| path.is_file());
    let bim_path = if has_bkmer {
        None
    } else {
        sibling_bim_path(&meta_path)
    };
    if let Some(path) = bim_path.as_ref() {
        let rows = count_bim_rows(path)?;
        if rows != usize::try_from(meta.n_kmers).context("n_kmers does not fit usize")? {
            bail!(
                "BIM header row count mismatch for {}: rows={} expected={}",
                path.display(),
                rows,
                meta.n_kmers
            );
        }
    }
    let mut bsite = File::open(&bsite_path)
        .with_context(|| format!("failed to open bsite file: {}", bsite_path.display()))?;
    let expected_payload = meta
        .n_kmers
        .checked_mul(meta.bytes_per_col)
        .ok_or_else(|| anyhow::anyhow!("bsite payload size overflow"))?;
    let expected_len = (BSITE_HEADER_SIZE as u64)
        .checked_add(expected_payload)
        .ok_or_else(|| anyhow::anyhow!("bsite file size overflow"))?;
    let actual_len = bsite
        .metadata()
        .with_context(|| format!("failed to stat bsite file: {}", bsite_path.display()))?
        .len();
    if actual_len != expected_len {
        bail!(
            "bsite payload length mismatch for {}: actual={} expected={}",
            bsite_path.display(),
            actual_len,
            expected_len
        );
    }
    let mut header = [0u8; BSITE_HEADER_SIZE];
    bsite
        .read_exact(&mut header)
        .with_context(|| format!("failed to read bsite header: {}", bsite_path.display()))?;
    let parsed = parse_bsite_header(&header, &bsite_path.display().to_string())
        .map_err(anyhow::Error::msg)?;
    if parsed.n_samples != meta.n_samples
        || parsed.n_kmers != meta.n_kmers
        || parsed.bytes_per_col != meta.bytes_per_col
    {
        bail!(
            "bsite header does not match metadata: {}",
            bsite_path.display()
        );
    }

    let samples = read_idv_file(&idv_path)?;
    if samples.len() != usize::try_from(meta.n_samples).context("n_samples does not fit usize")? {
        bail!(
            "idv sample count does not match metadata: {}",
            idv_path.display()
        );
    }
    Ok(KfileLayout {
        meta_path,
        bsite_path,
        idv_path,
        bim_path,
        meta,
        samples,
    })
}

fn build_sample_selection(
    samples: &[SampleEntry],
    sample_indices: Option<Vec<usize>>,
) -> Result<(Vec<usize>, Vec<String>)> {
    let indices = sample_indices.unwrap_or_else(|| (0..samples.len()).collect());
    let mut seen = HashSet::with_capacity(indices.len());
    let mut ids = Vec::with_capacity(indices.len());
    for &idx in &indices {
        let sample = samples
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("sample index out of range: {idx}"))?;
        if !seen.insert(idx) {
            bail!("duplicate sample index: {idx}");
        }
        ids.push(sample.sample_id.clone());
    }
    if indices.is_empty() {
        bail!("sample selection is empty");
    }
    Ok((indices, ids))
}

fn decode_bsite_chunk(
    bytes: &[u8],
    n_samples: usize,
    n_rows: usize,
    _row_offset: u64,
    selected_indices: &[usize],
) -> Result<(Vec<f32>, Vec<u64>)> {
    let bytes_per_col = n_samples
        .checked_add(7)
        .ok_or_else(|| anyhow::anyhow!("sample count overflow"))?
        / 8;
    let expected_len = n_rows
        .checked_mul(bytes_per_col)
        .ok_or_else(|| anyhow::anyhow!("chunk payload size overflow"))?;
    if bytes.len() != expected_len {
        bail!(
            "bsite chunk length mismatch: actual={} expected={}",
            bytes.len(),
            expected_len
        );
    }
    let indices: Vec<usize> = if selected_indices.is_empty() {
        (0..n_samples).collect()
    } else {
        selected_indices.to_vec()
    };
    for &idx in &indices {
        if idx >= n_samples {
            bail!("selected sample index out of range: {idx}");
        }
    }
    let mut dosage = vec![0.0f32; n_rows * indices.len()];
    let mut presence = vec![0u64; n_rows];
    for row in 0..n_rows {
        let col = &bytes[row * bytes_per_col..(row + 1) * bytes_per_col];
        let mut row_presence = 0u64;
        for (out_idx, &sample_idx) in indices.iter().enumerate() {
            let present = ((col[sample_idx >> 3] >> (sample_idx & 7)) & 1) != 0;
            if present {
                dosage[row * indices.len() + out_idx] = 2.0;
                row_presence += 1;
            }
        }
        presence[row] = row_presence;
    }
    Ok((dosage, presence))
}

#[inline]
fn maf_from_presence(presence: u64, n_samples: usize) -> f32 {
    if n_samples == 0 {
        return f32::NAN;
    }
    let p = presence as f32 / n_samples as f32;
    p.min(1.0 - p)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct KfileGrmStageTiming {
    pub read_ns: u64,
    pub filter_ns: u64,
    pub decode_ns: u64,
}

impl KfileGrmStageTiming {
    fn add_read(&mut self, elapsed: std::time::Duration) {
        self.read_ns = self
            .read_ns
            .saturating_add(elapsed.as_nanos().min(u64::MAX as u128) as u64);
    }

    fn add_filter(&mut self, elapsed: std::time::Duration) {
        self.filter_ns = self
            .filter_ns
            .saturating_add(elapsed.as_nanos().min(u64::MAX as u128) as u64);
    }

    fn add_decode(&mut self, elapsed: std::time::Duration) {
        self.decode_ns = self
            .decode_ns
            .saturating_add(elapsed.as_nanos().min(u64::MAX as u128) as u64);
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct KfileGrmPreparedBlock {
    packed: Vec<u8>,
    retained_row_offsets: Vec<usize>,
    row_center: Vec<f32>,
    row_scale: Vec<f32>,
    pub scanned_rows: usize,
    pub denominator: f64,
}

#[derive(Clone, Copy, Debug)]
struct KfileBitsetDecodeEntry {
    bit: usize,
    column: usize,
}

#[derive(Clone, Debug)]
struct KfileBitsetDecodeGroup {
    byte: usize,
    mask: u8,
    entries: Vec<KfileBitsetDecodeEntry>,
}

/// Precomputed bit positions for one decoded sample stripe.
///
/// The plan groups requested samples by source byte. This removes repeated
/// division/masking from the hot row loop while retaining the caller's sample
/// order in the output columns.
#[derive(Clone, Debug)]
pub(crate) struct KfileBitsetDecodePlan {
    width: usize,
    full_identity: bool,
    groups: Vec<KfileBitsetDecodeGroup>,
}

impl KfileBitsetDecodePlan {
    pub(crate) fn new(sample_indices: &[usize], n_samples_full: usize) -> Result<Self> {
        validate_grm_selection(sample_indices, n_samples_full)?;
        let bytes_per_col = n_samples_full
            .checked_add(7)
            .ok_or_else(|| anyhow::anyhow!("sample count overflow"))?
            / 8;
        let mut grouped = (0..bytes_per_col)
            .map(|_| Vec::<KfileBitsetDecodeEntry>::new())
            .collect::<Vec<_>>();
        for (column, &sample_index) in sample_indices.iter().enumerate() {
            grouped[sample_index >> 3].push(KfileBitsetDecodeEntry {
                bit: sample_index & 7,
                column,
            });
        }
        let groups = grouped
            .into_iter()
            .enumerate()
            .filter_map(|(byte, entries)| {
                if entries.is_empty() {
                    return None;
                }
                let mask = entries
                    .iter()
                    .fold(0u8, |mask, entry| mask | (1u8 << entry.bit));
                Some(KfileBitsetDecodeGroup {
                    byte,
                    mask,
                    entries,
                })
            })
            .collect::<Vec<_>>();
        let full_identity = sample_indices.len() == n_samples_full
            && sample_indices
                .iter()
                .enumerate()
                .all(|(expected, &actual)| expected == actual);
        Ok(Self {
            width: sample_indices.len(),
            full_identity,
            groups,
        })
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }
}

fn bitset_dosage_lut() -> &'static [[f32; 8]; 256] {
    static LUT: OnceLock<[[f32; 8]; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [[0.0_f32; 8]; 256];
        for (value, row) in table.iter_mut().enumerate() {
            for (bit, dosage) in row.iter_mut().enumerate() {
                if (value & (1usize << bit)) != 0 {
                    *dosage = 2.0;
                }
            }
        }
        table
    })
}

#[inline]
fn full_bitset_presence_scalar(row: &[u8], n_samples: usize) -> usize {
    let full_bytes = n_samples / 8;
    let mut count = 0u32;
    let whole_bytes = full_bytes - (full_bytes % 8);
    for chunk in row[..whole_bytes].chunks_exact(8) {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(chunk);
        count += u64::from_le_bytes(bytes).count_ones();
    }
    for &value in &row[whole_bytes..full_bytes] {
        count += value.count_ones();
    }
    let tail = n_samples % 8;
    if tail != 0 {
        count += (row[full_bytes] & ((1u8 << tail) - 1)).count_ones();
    }
    count as usize
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn full_bitset_presence_neon(row: &[u8], n_samples: usize) -> usize {
    use std::arch::aarch64::*;

    let full_bytes = n_samples / 8;
    let mut count = 0usize;
    let mut offset = 0usize;
    while offset + 16 <= full_bytes {
        let values = vld1q_u8(row.as_ptr().add(offset));
        let bits = vcntq_u8(values);
        let sum16 = vpaddlq_u8(bits);
        let sum32 = vpaddlq_u16(sum16);
        let sum64 = vpaddlq_u32(sum32);
        count += vgetq_lane_u64(sum64, 0) as usize;
        count += vgetq_lane_u64(sum64, 1) as usize;
        offset += 16;
    }
    count += row[offset..full_bytes]
        .iter()
        .map(|value| value.count_ones() as usize)
        .sum::<usize>();
    let tail = n_samples % 8;
    if tail != 0 {
        count += (row[full_bytes] & ((1u8 << tail) - 1)).count_ones() as usize;
    }
    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn full_bitset_presence_avx2(row: &[u8], n_samples: usize) -> usize {
    use std::arch::x86_64::*;

    let full_bytes = n_samples / 8;
    let lut4 = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f_i8);
    let zero = _mm256_setzero_si256();
    let mut count = 0usize;
    let mut offset = 0usize;
    while offset + 32 <= full_bytes {
        let values = _mm256_loadu_si256(row.as_ptr().add(offset) as *const __m256i);
        let low = _mm256_and_si256(values, low_mask);
        let high = _mm256_and_si256(_mm256_srli_epi16(values, 4), low_mask);
        let bits = _mm256_add_epi8(
            _mm256_shuffle_epi8(lut4, low),
            _mm256_shuffle_epi8(lut4, high),
        );
        let sums = _mm256_sad_epu8(bits, zero);
        let mut lanes = [0u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, sums);
        count += lanes.iter().map(|value| *value as usize).sum::<usize>();
        offset += 32;
    }
    count += row[offset..full_bytes]
        .iter()
        .map(|value| value.count_ones() as usize)
        .sum::<usize>();
    let tail = n_samples % 8;
    if tail != 0 {
        count += (row[full_bytes] & ((1u8 << tail) - 1)).count_ones() as usize;
    }
    count
}

/// Count full-sample presence bits using a runtime-dispatched SIMD popcount.
///
/// The bitset format is little-bit ordered, so the population count is
/// independent of the dosage expansion.  NEON/AVX2 is used only when the
/// platform advertises it; short rows and non-SIMD targets use the scalar
/// fallback.  The caller still gets exactly the same count for a tail byte.
#[inline]
fn full_bitset_presence(row: &[u8], n_samples: usize) -> usize {
    let full_bytes = n_samples / 8;
    #[cfg(target_arch = "aarch64")]
    if full_bytes >= 16 && std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { full_bitset_presence_neon(row, n_samples) };
    }
    #[cfg(target_arch = "x86_64")]
    if full_bytes >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { full_bitset_presence_avx2(row, n_samples) };
    }
    full_bitset_presence_scalar(row, n_samples)
}

/// Decode one full-identity bitset row to centered 0/2 dosage.
///
/// The hot path uses SIMD population count and a byte-to-eight-dosage LUT;
/// arbitrary sample subsets continue to use the grouped scalar decoder.  The
/// function is kept separate so tests and the GRM decoder can verify the fast
/// path against the scalar reference directly.
#[cfg(test)]
pub(crate) fn decode_centered_row_full_identity_simd(
    packed_row: &[u8],
    n_samples: usize,
    out_row: &mut [f32],
) -> Result<f32> {
    if n_samples == 0 {
        bail!("full-identity k-file row has zero samples");
    }
    if out_row.len() != n_samples {
        bail!(
            "full-identity k-file row buffer has wrong length: got {}, expected {}",
            out_row.len(),
            n_samples
        );
    }
    let bytes_per_col = n_samples.div_ceil(8);
    if packed_row.len() < bytes_per_col {
        bail!(
            "full-identity k-file row is too short: got {}, expected at least {}",
            packed_row.len(),
            bytes_per_col
        );
    }
    let presence = full_bitset_presence(packed_row, n_samples);
    let p = presence as f32 / n_samples as f32;
    let lut = bitset_dosage_lut();
    for byte_idx in 0..bytes_per_col {
        let start = byte_idx * 8;
        let width = (n_samples - start).min(8);
        let expanded = &lut[packed_row[byte_idx] as usize];
        out_row[start..start + width].copy_from_slice(&expanded[..width]);
    }
    let center = 2.0 * p;
    for value in out_row.iter_mut() {
        *value -= center;
    }
    Ok(p.min(1.0 - p))
}

impl KfileGrmPreparedBlock {
    pub(crate) fn retained_rows(&self) -> usize {
        self.retained_row_offsets.len()
    }

    pub(crate) fn packed_len(&self) -> usize {
        self.packed.len()
    }
}

fn validate_grm_parameters(method: usize, maf_threshold: f32) -> Result<()> {
    if method != 1 && method != 2 {
        bail!("GRM method must be 1 or 2, got {method}");
    }
    if !maf_threshold.is_finite() || !(0.0..=0.5).contains(&maf_threshold) {
        bail!("GRM MAF threshold must be finite and within 0..=0.5, got {maf_threshold}");
    }
    Ok(())
}

fn validate_grm_value_type(meta: &KmergeMeta, meta_path: &Path) -> Result<()> {
    if meta.value_type.trim() != "binary_presence" {
        bail!(
            "unsupported GRM value type in {}: {}",
            meta_path.display(),
            meta.value_type
        );
    }
    Ok(())
}

fn validate_grm_selection(sample_indices: &[usize], n_samples_full: usize) -> Result<()> {
    if sample_indices.is_empty() {
        bail!("sample selection is empty");
    }
    let mut seen = HashSet::with_capacity(sample_indices.len());
    for &index in sample_indices {
        if index >= n_samples_full {
            bail!("sample index out of range: {index}");
        }
        if !seen.insert(index) {
            bail!("duplicate sample index: {index}");
        }
    }
    Ok(())
}

pub(crate) fn prepare_grm_bsite_block(
    bytes: &[u8],
    n_samples_full: usize,
    n_rows: usize,
    sample_indices: &[usize],
    method: usize,
    maf_threshold: f32,
) -> Result<KfileGrmPreparedBlock> {
    validate_grm_parameters(method, maf_threshold)?;
    validate_grm_selection(sample_indices, n_samples_full)?;
    let bytes_per_col = n_samples_full
        .checked_add(7)
        .ok_or_else(|| anyhow::anyhow!("sample count overflow"))?
        / 8;
    let expected_len = n_rows
        .checked_mul(bytes_per_col)
        .ok_or_else(|| anyhow::anyhow!("GRM block payload size overflow"))?;
    if bytes.len() != expected_len {
        bail!(
            "GRM block length mismatch: actual={} expected={}",
            bytes.len(),
            expected_len
        );
    }

    let selected_n = sample_indices.len();
    let decode_plan = KfileBitsetDecodePlan::new(sample_indices, n_samples_full)?;
    let mut retained_row_offsets = Vec::new();
    let mut row_center = Vec::new();
    let mut row_scale = Vec::new();
    let mut denominator = 0.0f64;
    let full_identity = decode_plan.full_identity;
    for row in 0..n_rows {
        let packed_row = &bytes[row * bytes_per_col..(row + 1) * bytes_per_col];
        let presence = if full_identity {
            full_bitset_presence(packed_row, n_samples_full)
        } else {
            decode_plan
                .groups
                .iter()
                .map(|group| (packed_row[group.byte] & group.mask).count_ones() as usize)
                .sum()
        };
        let p = presence as f64 / selected_n as f64;
        let maf = p.min(1.0 - p);
        let variance = 2.0 * p * (1.0 - p);
        if maf >= maf_threshold as f64 && variance > 1e-12 {
            retained_row_offsets.push(row);
            row_center.push((2.0 * p) as f32);
            if method == 1 {
                row_scale.push(1.0);
                denominator += variance;
            } else {
                row_scale.push((1.0 / variance.sqrt()) as f32);
                denominator += 1.0;
            }
        }
    }
    Ok(KfileGrmPreparedBlock {
        packed: bytes.to_vec(),
        retained_row_offsets,
        row_center,
        row_scale,
        scanned_rows: n_rows,
        denominator,
    })
}

#[cfg(test)]
pub(crate) fn decode_grm_prepared_block_into(
    block: &KfileGrmPreparedBlock,
    sample_indices: &[usize],
    out: &mut [f32],
) -> Result<()> {
    if sample_indices.is_empty() {
        return Ok(());
    }
    let n_samples_full = sample_indices
        .iter()
        .copied()
        .max()
        .and_then(|index| index.checked_add(1))
        .unwrap_or(0);
    let plan = KfileBitsetDecodePlan::new(sample_indices, n_samples_full)?;
    decode_grm_prepared_block_into_with_plan(block, &plan, out)
}

pub(crate) fn decode_grm_prepared_block_into_with_plan(
    block: &KfileGrmPreparedBlock,
    plan: &KfileBitsetDecodePlan,
    out: &mut [f32],
) -> Result<()> {
    let required_len = block
        .retained_rows()
        .checked_mul(plan.width())
        .ok_or_else(|| anyhow::anyhow!("GRM decoded output size overflow"))?;
    if out.len() < required_len {
        bail!(
            "GRM decode output is too small: actual={} required={}",
            out.len(),
            required_len
        );
    }
    if block.scanned_rows == 0 {
        if !block.packed.is_empty() {
            bail!("GRM prepared block has bytes but zero scanned rows");
        }
        return Ok(());
    }
    if block.packed.len() % block.scanned_rows != 0 {
        bail!("GRM prepared block has an invalid packed layout");
    }
    let bytes_per_col = block.packed.len() / block.scanned_rows;
    let lut = bitset_dosage_lut();
    for (decoded_row, &packed_row) in block.retained_row_offsets.iter().enumerate() {
        if packed_row >= block.scanned_rows {
            bail!("GRM prepared block has an invalid retained row offset");
        }
        let row_bytes = &block.packed[packed_row * bytes_per_col..(packed_row + 1) * bytes_per_col];
        let output_row = &mut out[decoded_row * plan.width()..(decoded_row + 1) * plan.width()];
        if plan.full_identity {
            for group in &plan.groups {
                if group.byte >= bytes_per_col {
                    bail!(
                        "bitset decode plan byte out of range: {} >= {}",
                        group.byte,
                        bytes_per_col
                    );
                }
                let expanded = &lut[row_bytes[group.byte] as usize];
                let output_start = group.byte * 8;
                for (offset, &genotype) in expanded.iter().take(group.entries.len()).enumerate() {
                    output_row[output_start + offset] =
                        (genotype - block.row_center[decoded_row]) * block.row_scale[decoded_row];
                }
            }
        } else {
            for group in &plan.groups {
                if group.byte >= bytes_per_col {
                    bail!(
                        "bitset decode plan byte out of range: {} >= {}",
                        group.byte,
                        bytes_per_col
                    );
                }
                let expanded = &lut[row_bytes[group.byte] as usize];
                for entry in &group.entries {
                    output_row[entry.column] = (expanded[entry.bit]
                        - block.row_center[decoded_row])
                        * block.row_scale[decoded_row];
                }
            }
        }
    }
    Ok(())
}

pub(crate) struct KfileGrmSource {
    layout: KfileLayout,
    bsite: File,
    sample_indices: Vec<usize>,
    sample_ids: Vec<String>,
    method: usize,
    maf_threshold: f32,
    next_row: u64,
    effective_rows: usize,
    denominator: f64,
    stage_timing: KfileGrmStageTiming,
    full_decode_plan: KfileBitsetDecodePlan,
}

/// Native streaming source for association scans over a k-file bitset.
///
/// Unlike [`KfileChunkReader`], this source writes directly into caller-owned
/// buffers and centers each 0/2 dosage row on the selected-sample mean.  The
/// association kernels can therefore consume one bounded block without a
/// Python allocation or a second k-file pass.
pub(crate) struct KfileAssocSource {
    layout: KfileLayout,
    bsite: File,
    sample_indices: Vec<usize>,
    decode_plan: KfileBitsetDecodePlan,
    packed_block: Vec<u8>,
    next_row: u64,
}

/// A reusable decoded block exchanged by the persistent k-file prefetcher.
///
/// The consumer returns ordinary blocks with [`KfileAssocPrefetch::recycle`]
/// after copying/using their contents.  Keeping ownership of the two buffers
/// in the channel avoids a per-block allocation while the producer reads and
/// decodes the next block in parallel with the sparse solve.
pub(crate) struct KfileAssocPrefetchBlock {
    dosage: Vec<f32>,
    maf: Vec<f32>,
    n_samples: usize,
    row_start: usize,
    rows: usize,
    error: Option<String>,
}

impl KfileAssocPrefetchBlock {
    fn new(values_len: usize, block_rows: usize, n_samples: usize) -> Self {
        Self {
            dosage: vec![0.0; values_len],
            maf: vec![0.0; block_rows],
            n_samples,
            row_start: 0,
            rows: 0,
            error: None,
        }
    }

    pub(crate) fn row_start(&self) -> usize {
        self.row_start
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn dosage(&self) -> &[f32] {
        &self.dosage[..self.rows.saturating_mul(self.n_samples)]
    }

    pub(crate) fn maf(&self) -> &[f32] {
        &self.maf[..self.rows]
    }
}

/// Persistent two-buffer decoder for consumers whose scan loop owns its
/// output slice (for example the generic SparseLMM `GenotypeMatrix` trait).
pub(crate) struct KfileAssocPrefetch {
    ready_rx: Option<Mutex<Receiver<KfileAssocPrefetchBlock>>>,
    free_tx: Option<SyncSender<KfileAssocPrefetchBlock>>,
    join: Option<JoinHandle<()>>,
}

impl KfileAssocPrefetch {
    fn new(source: KfileAssocSource, block_rows: usize) -> Result<Self> {
        if block_rows == 0 {
            bail!("k-file association prefetch block_rows must be > 0");
        }
        let n_samples = source.n_samples();
        let values_len = block_rows
            .checked_mul(n_samples)
            .ok_or_else(|| anyhow::anyhow!("k-file association prefetch buffer overflow"))?;
        let (free_tx, free_rx) = sync_channel::<KfileAssocPrefetchBlock>(2);
        let (ready_tx, ready_rx) = sync_channel::<KfileAssocPrefetchBlock>(2);
        for _ in 0..2 {
            free_tx
                .send(KfileAssocPrefetchBlock::new(
                    values_len, block_rows, n_samples,
                ))
                .expect("k-file prefetch seed free channel");
        }
        let marker_count = source.n_kmers();
        let join = std::thread::spawn(move || {
            let mut source = source;
            while let Ok(mut block) = free_rx.recv() {
                let row_start = source.next_row as usize;
                match source.next_centered_block(
                    block_rows,
                    block.dosage.as_mut_slice(),
                    block.maf.as_mut_slice(),
                ) {
                    Ok(Some(rows)) => {
                        block.row_start = row_start;
                        block.rows = rows;
                        let more = source.next_row < source.layout.meta.n_kmers;
                        if ready_tx.send(block).is_err() || !more {
                            break;
                        }
                    }
                    Ok(None) => {
                        block.row_start = marker_count;
                        block.rows = 0;
                        let _ = ready_tx.send(block);
                        break;
                    }
                    Err(error) => {
                        block.row_start = usize::MAX;
                        block.rows = 0;
                        block.error = Some(error.to_string());
                        let _ = ready_tx.send(block);
                        break;
                    }
                }
            }
        });
        Ok(Self {
            ready_rx: Some(Mutex::new(ready_rx)),
            free_tx: Some(free_tx),
            join: Some(join),
        })
    }

    pub(crate) fn next(&mut self) -> Result<Option<KfileAssocPrefetchBlock>, String> {
        let receiver = self
            .ready_rx
            .as_ref()
            .ok_or_else(|| "k-file association prefetch has been closed".to_string())?;
        let block = receiver
            .lock()
            .map_err(|_| "k-file association prefetch receiver lock poisoned".to_string())?
            .recv()
            .map_err(|_| "k-file association prefetch producer stopped unexpectedly".to_string())?;
        if let Some(error) = block.error {
            return Err(error);
        }
        if block.rows == 0 {
            return Ok(None);
        }
        Ok(Some(block))
    }

    pub(crate) fn recycle(&self, block: KfileAssocPrefetchBlock) -> Result<(), String> {
        if let Some(sender) = self.free_tx.as_ref() {
            // The producer intentionally exits after sending the final block,
            // so its receive end may already be gone when the consumer
            // recycles that last buffer.  The buffer is no longer needed in
            // that case; treat the failed recycle as a normal EOF condition.
            let _ = sender.send(block);
            Ok(())
        } else {
            Err("k-file association prefetch has been closed".to_string())
        }
    }
}

impl Drop for KfileAssocPrefetch {
    fn drop(&mut self) {
        self.ready_rx.take();
        self.free_tx.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl KfileAssocSource {
    pub(crate) fn open(prefix: &Path, sample_indices: &[usize]) -> Result<Self> {
        let layout = load_layout(prefix)?;
        validate_grm_value_type(&layout.meta, &layout.meta_path)?;
        let (selected, _) = build_sample_selection(&layout.samples, Some(sample_indices.to_vec()))?;
        let n_samples_full =
            usize::try_from(layout.meta.n_samples).context("n_samples does not fit usize")?;
        let decode_plan = KfileBitsetDecodePlan::new(&selected, n_samples_full)?;
        let bsite = File::open(&layout.bsite_path).with_context(|| {
            format!("failed to open bsite file: {}", layout.bsite_path.display())
        })?;
        Ok(Self {
            layout,
            bsite,
            sample_indices: selected,
            decode_plan,
            packed_block: Vec::new(),
            next_row: 0,
        })
    }

    pub(crate) fn n_samples(&self) -> usize {
        self.sample_indices.len()
    }

    pub(crate) fn n_samples_full(&self) -> usize {
        self.layout.meta.n_samples as usize
    }

    pub(crate) fn n_kmers(&self) -> usize {
        self.layout.meta.n_kmers as usize
    }

    pub(crate) fn into_prefetch(self, block_rows: usize) -> Result<KfileAssocPrefetch> {
        KfileAssocPrefetch::new(self, block_rows)
    }

    /// Read a small set of rows without changing the sequential cursor.
    ///
    /// Approximate SparseLMM uses this only for its r-hat marker sample.  The
    /// main association scan remains a single sequential pass over `bsite`.
    pub(crate) fn read_centered_rows_at(
        &self,
        row_indices: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n_selected = self.sample_indices.len();
        if row_indices.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let bytes_per_col = usize::try_from(self.layout.meta.bytes_per_col)
            .context("bytes_per_col does not fit usize")?;
        let mut bsite = self
            .bsite
            .try_clone()
            .with_context(|| format!("failed to clone {}", self.layout.bsite_path.display()))?;
        let mut packed = vec![0u8; bytes_per_col];
        let mut dosage = vec![0.0_f32; row_indices.len().saturating_mul(n_selected)];
        let mut maf = vec![0.0_f32; row_indices.len()];
        for (out_row_idx, &row_idx) in row_indices.iter().enumerate() {
            if row_idx >= self.n_kmers() {
                bail!("k-file association row index out of range: {row_idx}");
            }
            let offset = (BSITE_HEADER_SIZE as u64)
                .checked_add(
                    (row_idx as u64)
                        .checked_mul(self.layout.meta.bytes_per_col)
                        .ok_or_else(|| anyhow::anyhow!("k-file association offset overflow"))?,
                )
                .ok_or_else(|| anyhow::anyhow!("k-file association offset overflow"))?;
            bsite
                .seek(SeekFrom::Start(offset))
                .with_context(|| format!("failed to seek {}", self.layout.bsite_path.display()))?;
            bsite
                .read_exact(&mut packed)
                .with_context(|| format!("failed to read {}", self.layout.bsite_path.display()))?;
            let out = &mut dosage[out_row_idx * n_selected..(out_row_idx + 1) * n_selected];
            maf[out_row_idx] = self.decode_centered_row(&packed, out)?;
        }
        Ok((dosage, maf))
    }

    /// Read selected rows as raw 0/2 dosage values without changing the
    /// sequential association cursor.  This is used for k-file site
    /// covariates, where centering belongs to the model design matrix rather
    /// than to the genotype decoder.
    pub(crate) fn read_dosage_rows_at(&self, row_indices: &[usize]) -> Result<Vec<f32>> {
        let n_selected = self.sample_indices.len();
        if row_indices.is_empty() {
            return Ok(Vec::new());
        }
        let bytes_per_col = usize::try_from(self.layout.meta.bytes_per_col)
            .context("bytes_per_col does not fit usize")?;
        let mut bsite = self
            .bsite
            .try_clone()
            .with_context(|| format!("failed to clone {}", self.layout.bsite_path.display()))?;
        let mut packed = vec![0u8; bytes_per_col];
        let mut dosage = vec![0.0_f32; row_indices.len().saturating_mul(n_selected)];
        for (out_row_idx, &row_idx) in row_indices.iter().enumerate() {
            if row_idx >= self.n_kmers() {
                bail!("k-file association row index out of range: {row_idx}");
            }
            let offset = (BSITE_HEADER_SIZE as u64)
                .checked_add(
                    (row_idx as u64)
                        .checked_mul(self.layout.meta.bytes_per_col)
                        .ok_or_else(|| anyhow::anyhow!("k-file association offset overflow"))?,
                )
                .ok_or_else(|| anyhow::anyhow!("k-file association offset overflow"))?;
            bsite
                .seek(SeekFrom::Start(offset))
                .with_context(|| format!("failed to seek {}", self.layout.bsite_path.display()))?;
            bsite
                .read_exact(&mut packed)
                .with_context(|| format!("failed to read {}", self.layout.bsite_path.display()))?;
            let out = &mut dosage[out_row_idx * n_selected..(out_row_idx + 1) * n_selected];
            self.decode_dosage_row(&packed, out)?;
        }
        Ok(dosage)
    }

    fn decode_centered_row(&self, packed_row: &[u8], out_row: &mut [f32]) -> Result<f32> {
        let presence = self.decode_dosage_row(packed_row, out_row)?;
        let p = presence as f32 / self.sample_indices.len() as f32;
        let center = 2.0 * p;
        for value in out_row.iter_mut() {
            *value -= center;
        }
        Ok(p.min(1.0 - p))
    }

    fn decode_dosage_row(&self, packed_row: &[u8], out_row: &mut [f32]) -> Result<usize> {
        let n_selected = self.sample_indices.len();
        if out_row.len() != n_selected {
            bail!("k-file association row buffer has the wrong number of samples");
        }
        let n_samples_full = self.n_samples_full();
        let full_identity = self.decode_plan.full_identity;
        if full_identity {
            if n_samples_full == 0 {
                bail!("full-identity k-file row has zero samples");
            }
            let bytes_per_col = n_samples_full.div_ceil(8);
            if packed_row.len() < bytes_per_col {
                bail!(
                    "full-identity k-file row is too short: got {}, expected at least {}",
                    packed_row.len(),
                    bytes_per_col
                );
            }
            let presence = full_bitset_presence(packed_row, n_samples_full);
            let lut = bitset_dosage_lut();
            for byte_idx in 0..bytes_per_col {
                let start = byte_idx * 8;
                let width = (n_samples_full - start).min(8);
                let expanded = &lut[packed_row[byte_idx] as usize];
                out_row[start..start + width].copy_from_slice(&expanded[..width]);
            }
            return Ok(presence);
        }
        let lut = bitset_dosage_lut();
        let mut presence = 0usize;
        for group in &self.decode_plan.groups {
            let expanded = &lut[packed_row[group.byte] as usize];
            for entry in &group.entries {
                let value = expanded[entry.bit];
                out_row[entry.column] = value;
                if value > 0.0 {
                    presence += 1;
                }
            }
        }
        Ok(presence)
    }

    /// Decode at most `max_rows` rows into centered 0/2 dosage values.
    ///
    /// `dosage` must have room for `max_rows * n_samples()` values and `maf`
    /// for `max_rows` values.  The returned count is the number of rows
    /// decoded; `None` means the source is exhausted.
    pub(crate) fn next_centered_block(
        &mut self,
        max_rows: usize,
        dosage: &mut [f32],
        maf: &mut [f32],
    ) -> Result<Option<usize>> {
        if max_rows == 0 {
            bail!("k-file association block_rows must be > 0");
        }
        let n_selected = self.sample_indices.len();
        if dosage.len() < max_rows.saturating_mul(n_selected) {
            bail!("k-file association dosage buffer is too small");
        }
        if maf.len() < max_rows {
            bail!("k-file association MAF buffer is too small");
        }
        if self.next_row >= self.layout.meta.n_kmers {
            return Ok(None);
        }

        let rows_left = self.layout.meta.n_kmers - self.next_row;
        let rows = usize::try_from(rows_left.min(max_rows as u64))
            .context("k-file association row count does not fit usize")?;
        let bytes_per_col = usize::try_from(self.layout.meta.bytes_per_col)
            .context("bytes_per_col does not fit usize")?;
        let byte_len = rows
            .checked_mul(bytes_per_col)
            .ok_or_else(|| anyhow::anyhow!("k-file association block size overflow"))?;
        let offset = (BSITE_HEADER_SIZE as u64)
            .checked_add(
                self.next_row
                    .checked_mul(self.layout.meta.bytes_per_col)
                    .ok_or_else(|| anyhow::anyhow!("k-file association offset overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("k-file association offset overflow"))?;
        self.bsite
            .seek(SeekFrom::Start(offset))
            .with_context(|| format!("failed to seek {}", self.layout.bsite_path.display()))?;
        // Reuse the bounded packed-byte buffer across sequential blocks.  A
        // k-file scan can contain millions of blocks; allocating a fresh
        // Vec for every block otherwise becomes visible in profiles even
        // though the resident memory remains bounded.
        self.packed_block.resize(byte_len, 0);
        self.bsite
            .read_exact(&mut self.packed_block)
            .with_context(|| format!("failed to read {}", self.layout.bsite_path.display()))?;

        for row in 0..rows {
            let packed_row = &self.packed_block[row * bytes_per_col..(row + 1) * bytes_per_col];
            let out_row = &mut dosage[row * n_selected..(row + 1) * n_selected];
            maf[row] = self.decode_centered_row(packed_row, out_row)?;
        }
        self.next_row += rows as u64;
        Ok(Some(rows))
    }

    /// Consume the sequential source with a bounded two-buffer pipeline.
    ///
    /// The producer owns the file cursor and fills two reusable centered
    /// dosage/MAF buffers while the caller performs projection, association,
    /// or output work in `consume`.  No marker-sized allocation is made and
    /// the source is scanned exactly once.  The sequential method remains
    /// available for adapters that require a caller-owned output slice.
    pub(crate) fn for_each_centered_block<F>(
        self,
        block_rows: usize,
        overlap: bool,
        mut consume: F,
    ) -> Result<usize, String>
    where
        F: FnMut(usize, usize, &[f32], &[f32]) -> Result<(), String>,
    {
        if block_rows == 0 {
            return Err("k-file association block_rows must be > 0".to_string());
        }
        let n = self.n_samples();
        let marker_count = self.n_kmers();
        let block_values = block_rows
            .checked_mul(n)
            .ok_or_else(|| "k-file association double-buffer size overflow".to_string())?;

        struct AssocChunk {
            dosage: Vec<f32>,
            maf: Vec<f32>,
            row_start: usize,
            rows: usize,
            producer_failed: bool,
        }

        impl AssocChunk {
            fn new(block_values: usize, block_rows: usize) -> Self {
                Self {
                    dosage: vec![0.0; block_values],
                    maf: vec![0.0; block_rows],
                    row_start: 0,
                    rows: 0,
                    producer_failed: false,
                }
            }
        }

        if !overlap || marker_count <= block_rows {
            let mut source = self;
            let mut dosage = vec![0.0_f32; block_values];
            let mut maf = vec![0.0_f32; block_rows];
            let mut done = 0usize;
            while let Some(rows) = source
                .next_centered_block(block_rows, dosage.as_mut_slice(), maf.as_mut_slice())
                .map_err(|error| error.to_string())?
            {
                consume(done, rows, &dosage[..rows * n], &maf[..rows])?;
                done = done.saturating_add(rows);
            }
            return Ok(done);
        }

        let producer_error = Arc::new(Mutex::new(None::<String>));
        let producer_error_slot = Arc::clone(&producer_error);
        let mut source = self;
        let producer = move |chunk: &mut AssocChunk| -> bool {
            let row_start = source.next_row as usize;
            match source.next_centered_block(
                block_rows,
                chunk.dosage.as_mut_slice(),
                chunk.maf.as_mut_slice(),
            ) {
                Ok(Some(rows)) => {
                    chunk.row_start = row_start;
                    chunk.rows = rows;
                    chunk.producer_failed = false;
                    source.next_row < source.layout.meta.n_kmers
                }
                Ok(None) => {
                    chunk.row_start = marker_count;
                    chunk.rows = 0;
                    chunk.producer_failed = false;
                    false
                }
                Err(error) => {
                    if let Ok(mut slot) = producer_error_slot.lock() {
                        *slot = Some(error.to_string());
                    }
                    chunk.row_start = usize::MAX;
                    chunk.rows = 0;
                    chunk.producer_failed = true;
                    false
                }
            }
        };
        let consumer_error = Arc::clone(&producer_error);
        let mut done = 0usize;
        let consumer = |chunk: &mut AssocChunk| -> Result<(), String> {
            if chunk.producer_failed || chunk.row_start == usize::MAX {
                return Err(consumer_error
                    .lock()
                    .map_err(|_| "k-file association producer error lock poisoned".to_string())?
                    .take()
                    .unwrap_or_else(|| "k-file association producer failed".to_string()));
            }
            if chunk.rows == 0 {
                return Ok(());
            }
            consume(
                chunk.row_start,
                chunk.rows,
                &chunk.dosage[..chunk.rows * n],
                &chunk.maf[..chunk.rows],
            )?;
            done = chunk.row_start.saturating_add(chunk.rows);
            Ok(())
        };

        crate::pipeline::run_double_buffer(
            2,
            || AssocChunk::new(block_values, block_rows),
            producer,
            consumer,
        )?;
        if let Some(error) = producer_error
            .lock()
            .map_err(|_| "k-file association producer error lock poisoned".to_string())?
            .take()
        {
            return Err(error);
        }
        Ok(done)
    }
}

#[allow(dead_code)]
impl KfileGrmSource {
    pub(crate) fn open(
        prefix: &Path,
        sample_indices: Option<&[usize]>,
        method: usize,
        maf_threshold: f32,
    ) -> Result<Self> {
        validate_grm_parameters(method, maf_threshold)?;
        let layout = load_layout(prefix)?;
        validate_grm_value_type(&layout.meta, &layout.meta_path)?;
        let requested_indices = sample_indices.map(|indices| indices.to_vec());
        let (sample_indices, sample_ids) =
            build_sample_selection(&layout.samples, requested_indices)?;
        let n_samples_full =
            usize::try_from(layout.meta.n_samples).context("n_samples does not fit usize")?;
        let full_decode_plan = KfileBitsetDecodePlan::new(&sample_indices, n_samples_full)?;
        let bsite = File::open(&layout.bsite_path).with_context(|| {
            format!("failed to open bsite file: {}", layout.bsite_path.display())
        })?;
        Ok(Self {
            layout,
            bsite,
            sample_indices,
            sample_ids,
            method,
            maf_threshold,
            next_row: 0,
            effective_rows: 0,
            denominator: 0.0,
            stage_timing: KfileGrmStageTiming::default(),
            full_decode_plan,
        })
    }

    pub(crate) fn next_prepared_block(
        &mut self,
        block_rows: usize,
    ) -> Result<Option<KfileGrmPreparedBlock>> {
        if block_rows == 0 {
            bail!("GRM block_rows must be > 0");
        }
        if self.next_row >= self.layout.meta.n_kmers {
            return Ok(None);
        }
        let rows_left = self.layout.meta.n_kmers - self.next_row;
        let n_rows = usize::try_from(rows_left.min(block_rows as u64))
            .context("GRM block row count does not fit usize")?;
        let bytes_per_col = usize::try_from(self.layout.meta.bytes_per_col)
            .context("bytes_per_col does not fit usize")?;
        let byte_len = n_rows
            .checked_mul(bytes_per_col)
            .ok_or_else(|| anyhow::anyhow!("GRM block payload size overflow"))?;
        let offset = (BSITE_HEADER_SIZE as u64)
            .checked_add(
                self.next_row
                    .checked_mul(self.layout.meta.bytes_per_col)
                    .ok_or_else(|| anyhow::anyhow!("GRM bsite offset overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("GRM bsite offset overflow"))?;
        self.bsite
            .seek(SeekFrom::Start(offset))
            .with_context(|| format!("failed to seek {}", self.layout.bsite_path.display()))?;
        let mut bytes = vec![0u8; byte_len];
        let read_started = Instant::now();
        self.bsite
            .read_exact(&mut bytes)
            .with_context(|| format!("failed to read {}", self.layout.bsite_path.display()))?;
        self.stage_timing.add_read(read_started.elapsed());

        let filter_started = Instant::now();
        let block = prepare_grm_bsite_block(
            &bytes,
            self.n_samples_full(),
            n_rows,
            &self.sample_indices,
            self.method,
            self.maf_threshold,
        )?;
        self.stage_timing.add_filter(filter_started.elapsed());
        self.next_row += n_rows as u64;
        self.effective_rows = self
            .effective_rows
            .checked_add(block.retained_rows())
            .ok_or_else(|| anyhow::anyhow!("effective GRM row count overflow"))?;
        self.denominator += block.denominator;
        Ok(Some(block))
    }

    pub(crate) fn decode_block_into(
        &mut self,
        block: &KfileGrmPreparedBlock,
        selected_positions: Range<usize>,
        out: &mut [f32],
    ) -> Result<()> {
        if selected_positions.start > selected_positions.end
            || selected_positions.end > self.sample_indices.len()
        {
            bail!("GRM selected sample positions are out of range");
        }
        let decode_started = Instant::now();
        if selected_positions.start == 0 && selected_positions.end == self.sample_indices.len() {
            decode_grm_prepared_block_into_with_plan(block, &self.full_decode_plan, out)?;
        } else {
            let selected = &self.sample_indices[selected_positions.clone()];
            let plan = KfileBitsetDecodePlan::new(selected, self.n_samples_full())?;
            decode_grm_prepared_block_into_with_plan(block, &plan, out)?;
        }
        self.stage_timing.add_decode(decode_started.elapsed());
        Ok(())
    }

    pub(crate) fn decode_block_into_with_plan(
        &mut self,
        block: &KfileGrmPreparedBlock,
        plan: &KfileBitsetDecodePlan,
        out: &mut [f32],
    ) -> Result<()> {
        let decode_started = Instant::now();
        decode_grm_prepared_block_into_with_plan(block, plan, out)?;
        self.stage_timing.add_decode(decode_started.elapsed());
        Ok(())
    }

    pub(crate) fn reset_scan(&mut self) -> Result<()> {
        self.bsite
            .seek(SeekFrom::Start(BSITE_HEADER_SIZE as u64))
            .with_context(|| format!("failed to reset {}", self.layout.bsite_path.display()))?;
        self.next_row = 0;
        self.effective_rows = 0;
        self.denominator = 0.0;
        Ok(())
    }

    pub(crate) fn reset_timing(&mut self) {
        self.stage_timing = KfileGrmStageTiming::default();
    }

    pub(crate) fn n_samples(&self) -> usize {
        self.sample_indices.len()
    }

    pub(crate) fn n_samples_full(&self) -> usize {
        self.layout.meta.n_samples as usize
    }

    pub(crate) fn n_kmers(&self) -> u64 {
        self.layout.meta.n_kmers
    }

    pub(crate) fn sample_indices(&self) -> &[usize] {
        &self.sample_indices
    }

    pub(crate) fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    pub(crate) fn scanned_rows(&self) -> usize {
        self.next_row as usize
    }

    pub(crate) fn effective_rows(&self) -> usize {
        self.effective_rows
    }

    pub(crate) fn denominator(&self) -> f64 {
        self.denominator
    }

    pub(crate) fn stage_timing(&self) -> KfileGrmStageTiming {
        self.stage_timing
    }
}

#[pyclass]
pub struct KfileChunkReader {
    bsite: File,
    bsite_path: PathBuf,
    n_samples_full: usize,
    n_kmers: u64,
    bytes_per_col: usize,
    sample_indices: Vec<usize>,
    sample_ids: Vec<String>,
    next_row: u64,
}

#[pymethods]
impl KfileChunkReader {
    #[new]
    #[pyo3(signature = (prefix, sample_indices=None, mmap_window_mb=None))]
    fn new(
        prefix: String,
        sample_indices: Option<Vec<usize>>,
        mmap_window_mb: Option<usize>,
    ) -> PyResult<Self> {
        let _ = mmap_window_mb;
        let layout =
            load_layout(Path::new(&prefix)).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let n_samples_full = usize::try_from(layout.meta.n_samples)
            .map_err(|_| PyRuntimeError::new_err("n_samples does not fit usize"))?;
        let (sample_indices, sample_ids) = build_sample_selection(&layout.samples, sample_indices)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let bsite = File::open(&layout.bsite_path)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to open bsite file: {e}")))?;
        Ok(Self {
            bsite,
            bsite_path: layout.bsite_path,
            n_samples_full,
            n_kmers: layout.meta.n_kmers,
            bytes_per_col: usize::try_from(layout.meta.bytes_per_col)
                .map_err(|_| PyRuntimeError::new_err("bytes_per_col does not fit usize"))?,
            sample_indices,
            sample_ids,
            next_row: 0,
        })
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.sample_indices.len()
    }

    #[getter]
    fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }

    #[getter]
    fn n_kmers(&self) -> u64 {
        self.n_kmers
    }

    #[getter]
    fn bytes_per_col(&self) -> usize {
        self.bytes_per_col
    }

    #[getter]
    fn sample_ids(&self) -> Vec<String> {
        self.sample_ids.clone()
    }

    #[getter]
    fn next_row(&self) -> u64 {
        self.next_row
    }

    #[pyo3(signature = (chunk_size))]
    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk_size: usize,
    ) -> PyResult<
        Option<(
            Bound<'py, PyArray2<f32>>,
            Bound<'py, PyArray1<f32>>,
            Bound<'py, PyArray1<i64>>,
        )>,
    > {
        if chunk_size == 0 {
            return Err(PyValueError::new_err("chunk_size must be > 0"));
        }
        if self.next_row >= self.n_kmers {
            return Ok(None);
        }
        let rows_left = self.n_kmers - self.next_row;
        let n_rows = usize::try_from(rows_left.min(chunk_size as u64))
            .map_err(|_| PyRuntimeError::new_err("chunk row count does not fit usize"))?;
        let byte_len = n_rows
            .checked_mul(self.bytes_per_col)
            .ok_or_else(|| PyRuntimeError::new_err("chunk payload size overflow"))?;
        let offset = (BSITE_HEADER_SIZE as u64)
            .checked_add(
                self.next_row
                    .checked_mul(self.bytes_per_col as u64)
                    .ok_or_else(|| PyRuntimeError::new_err("bsite offset overflow"))?,
            )
            .ok_or_else(|| PyRuntimeError::new_err("bsite offset overflow"))?;
        self.bsite.seek(SeekFrom::Start(offset)).map_err(|e| {
            PyRuntimeError::new_err(format!("failed to seek {}: {e}", self.bsite_path.display()))
        })?;
        let mut bytes = vec![0u8; byte_len];
        self.bsite.read_exact(&mut bytes).map_err(|e| {
            PyRuntimeError::new_err(format!("failed to read {}: {e}", self.bsite_path.display()))
        })?;
        let (dosage, presence) = decode_bsite_chunk(
            &bytes,
            self.n_samples_full,
            n_rows,
            self.next_row,
            &self.sample_indices,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let n_selected = self.sample_indices.len();
        let maf = presence
            .iter()
            .map(|&count| maf_from_presence(count, n_selected))
            .collect::<Vec<_>>();
        let row_indices = (self.next_row..self.next_row + n_rows as u64)
            .map(|idx| idx as i64)
            .collect::<Vec<_>>();
        self.next_row += n_rows as u64;
        let geno = Array2::from_shape_vec((n_rows, n_selected), dosage)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to shape dosage chunk: {e}")))?;
        Ok(Some((
            PyArray2::from_owned_array(py, geno).into_bound(),
            PyArray1::from_owned_array(py, Array1::from_vec(maf)).into_bound(),
            PyArray1::from_owned_array(py, Array1::from_vec(row_indices)).into_bound(),
        )))
    }
}

/// Decode selected k-file rows to raw 0/2 dosage values.
///
/// This narrow helper is intentionally separate from the sequential chunk
/// reader: GWAS site covariates need a few BIM-addressed rows while preserving
/// the native scanner's one-pass cursor and bounded memory behavior.
#[pyfunction(name = "kfile_read_rows_f32")]
#[pyo3(signature = (prefix, sample_indices, row_indices))]
pub fn kfile_read_rows_f32_py<'py>(
    py: Python<'py>,
    prefix: String,
    sample_indices: Vec<i64>,
    row_indices: Vec<i64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if sample_indices.is_empty() {
        return Err(PyValueError::new_err("sample_indices must not be empty"));
    }
    if row_indices.is_empty() {
        return Err(PyValueError::new_err("row_indices must not be empty"));
    }
    let samples = sample_indices
        .iter()
        .enumerate()
        .map(|(i, &raw)| {
            usize::try_from(raw).map_err(|_| {
                PyValueError::new_err(format!("sample_indices[{i}] must be non-negative"))
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let rows = row_indices
        .iter()
        .enumerate()
        .map(|(i, &raw)| {
            usize::try_from(raw).map_err(|_| {
                PyValueError::new_err(format!("row_indices[{i}] must be non-negative"))
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let source = KfileAssocSource::open(Path::new(&prefix), &samples)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let dosage = source
        .read_dosage_rows_at(&rows)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let array = Array2::from_shape_vec((rows.len(), samples.len()), dosage)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to shape k-file dosage: {e}")))?;
    Ok(PyArray2::from_owned_array(py, array).into_bound())
}

#[pyfunction(name = "kfile_inspect")]
pub fn kfile_inspect_py<'py>(py: Python<'py>, prefix: String) -> PyResult<Bound<'py, PyDict>> {
    let layout =
        load_layout(Path::new(&prefix)).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = PyDict::new(py);
    out.set_item("prefix", prefix)?;
    out.set_item("meta", layout.meta_path.to_string_lossy().as_ref())?;
    out.set_item("bsite", layout.bsite_path.to_string_lossy().as_ref())?;
    out.set_item("idv", layout.idv_path.to_string_lossy().as_ref())?;
    out.set_item(
        "bim",
        layout
            .bim_path
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned()),
    )?;
    out.set_item("k", layout.meta.k)?;
    out.set_item("n_samples", layout.meta.n_samples)?;
    out.set_item("n_kmers", layout.meta.n_kmers)?;
    out.set_item("bytes_per_col", layout.meta.bytes_per_col)?;
    out.set_item(
        "sample_ids",
        layout
            .samples
            .iter()
            .map(|sample| sample.sample_id.clone())
            .collect::<Vec<_>>(),
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        decode_bsite_chunk, decode_centered_row_full_identity_simd, decode_grm_prepared_block_into,
        decode_grm_prepared_block_into_with_plan, load_layout, maf_from_presence,
        prepare_grm_bsite_block, KfileAssocSource, KfileBitsetDecodePlan, KfileChunkReader,
        KfileGrmSource,
    };
    use crate::kmer::format::{BsiteHeader, KmergeMeta};
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FIXTURE_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn grm_fixture_dir() -> PathBuf {
        let unique = FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "janusx-kfile-grm-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos(),
            unique
        ));
        fs::create_dir(&path).expect("create fixture directory");
        path
    }

    fn write_grm_fixture(dir: &Path, meta: &KmergeMeta, idv_rows: &str, payload: &[u8]) {
        let prefix = dir.join("fixture");
        fs::write(
            PathBuf::from(format!("{}.meta.json", prefix.display())),
            serde_json::to_vec(meta).expect("serialize metadata"),
        )
        .expect("write metadata");
        fs::write(dir.join(&meta.idv_file), idv_rows).expect("write idv");
        let mut bsite = File::create(dir.join(&meta.bsite_file)).expect("create bsite");
        BsiteHeader {
            n_samples: meta.n_samples,
            n_kmers: meta.n_kmers,
            bytes_per_col: meta.bytes_per_col,
        }
        .write_to(&mut bsite)
        .expect("write bsite header");
        bsite.write_all(payload).expect("write bsite payload");
    }

    fn valid_grm_meta() -> KmergeMeta {
        KmergeMeta {
            format: "janusx-kmer-bitmatrix-v1".to_string(),
            k: 31,
            n_samples: 4,
            n_kmers: 3,
            bytes_per_col: 1,
            encoding: "acgt_2bit".to_string(),
            canonical: true,
            matrix_layout: "column_major_bitset".to_string(),
            value_type: "binary_presence".to_string(),
            bit_order: "little_bit_order".to_string(),
            bkmer_file: "fixture.bkmer".to_string(),
            bsite_file: "fixture.bsite".to_string(),
            idv_file: "fixture.idv".to_string(),
            min_count: 1,
            min_presence_rate: 0.0,
            max_presence_rate: 1.0,
            bucket_bits: 8,
            compression: "none".to_string(),
        }
    }

    #[test]
    fn marker_kfile_uses_sibling_bim_when_bkmer_is_absent() {
        let dir = grm_fixture_dir();
        let mut meta = valid_grm_meta();
        meta.k = 0;
        meta.bkmer_file.clear();
        let compact_idv = "s0\ns1\ns2\ns3\n";
        write_grm_fixture(&dir, &meta, compact_idv, &[3, 15, 1]);
        let bim_path = dir.join("fixture.bim");
        fs::write(
            &bim_path,
            "1\trs1\t0\t1\tA\tG\n1\trs2\t0\t2\tA\tG\n1\trs3\t0\t3\tA\tG\n",
        )
        .expect("write BIM header");
        let layout = load_layout(&dir.join("fixture")).expect("load marker kfile");
        assert_eq!(layout.bim_path.as_deref(), Some(bim_path.as_path()));
        assert_eq!(
            layout
                .samples
                .iter()
                .map(|sample| sample.sample_id.as_str())
                .collect::<Vec<_>>(),
            vec!["s0", "s1", "s2", "s3"]
        );
        fs::remove_dir_all(dir).expect("remove fixture directory");
    }

    const VALID_IDV: &str =
        "#idx\tsample_id\tkmc_prefix\n0\ts0\tp0\n1\ts1\tp1\n2\ts2\tp2\n3\ts3\tp3\n";

    #[test]
    fn grm_block_filters_on_selected_sample_maf_and_centers_zero_two_dosage() {
        // rows: 0011, 1111, 0001 over four selected samples
        let bytes = [0b0000_0011, 0b0000_1111, 0b0000_0001];
        let block = prepare_grm_bsite_block(&bytes, 4, 3, &[0, 1, 2, 3], 1, 0.25).unwrap();
        let mut values = vec![0.0; block.retained_rows() * 4];
        decode_grm_prepared_block_into(&block, &[0, 1, 2, 3], &mut values).unwrap();
        assert_eq!(block.retained_rows(), 2);
        assert_eq!(block.scanned_rows, 3);
        assert_eq!(values, vec![1.0, 1.0, -1.0, -1.0, 1.5, -0.5, -0.5, -0.5]);
        assert!((block.denominator - 0.875).abs() < 1e-12);
    }

    #[test]
    fn grm_block_method_two_excludes_monomorphic_rows_and_standardizes() {
        let bytes = [0b0000_0011, 0b0000_1111];
        let block = prepare_grm_bsite_block(&bytes, 4, 2, &[3, 1, 0, 2], 2, 0.0).unwrap();
        let mut values = vec![0.0; block.retained_rows() * 4];
        decode_grm_prepared_block_into(&block, &[3, 1, 0, 2], &mut values).unwrap();
        assert_eq!(block.retained_rows(), 1);
        assert_eq!(block.denominator, 1.0);
        assert_eq!(values.len(), 4);
        assert!((values.iter().map(|x| x * x).sum::<f32>() - 8.0).abs() < 1e-5);
    }

    #[test]
    fn grm_bitset_lut_plan_matches_scalar_for_all_bytes_and_reordered_subset() {
        let n_samples = 13usize;
        let bytes_per_col = n_samples.div_ceil(8);
        let mut bytes = Vec::<u8>::new();
        for value in 1u8..=254u8 {
            bytes.push(value);
            bytes.push((!value) & 0x1f);
        }
        let n_rows = bytes.len() / bytes_per_col;
        let selections = vec![
            (0..n_samples).collect::<Vec<_>>(),
            vec![12usize, 0, 7, 3, 10],
        ];
        for selected in selections {
            let block = prepare_grm_bsite_block(
                bytes.as_slice(),
                n_samples,
                n_rows,
                selected.as_slice(),
                1,
                0.0,
            )
            .expect("prepare bitset block");
            let plan = KfileBitsetDecodePlan::new(selected.as_slice(), n_samples)
                .expect("build bitset decode plan");
            let mut got = vec![0.0_f32; block.retained_rows() * selected.len()];
            decode_grm_prepared_block_into_with_plan(&block, &plan, &mut got)
                .expect("decode with bitset plan");
            let mut expected = vec![0.0_f32; got.len()];
            for (decoded_row, &raw_row) in block.retained_row_offsets.iter().enumerate() {
                let row_bytes = &bytes[raw_row * bytes_per_col..(raw_row + 1) * bytes_per_col];
                for (column, &sample_index) in selected.iter().enumerate() {
                    let genotype =
                        if ((row_bytes[sample_index >> 3] >> (sample_index & 7)) & 1) != 0 {
                            2.0_f32
                        } else {
                            0.0_f32
                        };
                    expected[decoded_row * selected.len() + column] =
                        genotype - block.row_center[decoded_row];
                }
            }
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn full_identity_simd_decode_matches_scalar_reference() {
        let n_samples = 137usize;
        let bytes_per_col = n_samples.div_ceil(8);
        let mut packed = vec![0u8; bytes_per_col];
        for (idx, byte) in packed.iter_mut().enumerate() {
            *byte = (idx as u8).wrapping_mul(37).rotate_left(1);
        }
        packed[bytes_per_col - 1] &= (1u8 << (n_samples % 8)) - 1;
        let mut got = vec![0.0_f32; n_samples];
        let maf = decode_centered_row_full_identity_simd(
            packed.as_slice(),
            n_samples,
            got.as_mut_slice(),
        )
        .expect("SIMD full-identity decode");

        let presence = packed
            .iter()
            .enumerate()
            .map(|(byte_idx, value)| {
                let valid_bits = if byte_idx + 1 == bytes_per_col {
                    n_samples - byte_idx * 8
                } else {
                    8
                };
                (value
                    & if valid_bits == 8 {
                        u8::MAX
                    } else {
                        (1u8 << valid_bits) - 1
                    })
                .count_ones() as usize
            })
            .sum::<usize>();
        let p = presence as f32 / n_samples as f32;
        assert!((maf - p.min(1.0 - p)).abs() < 1e-7);
        for (sample, value) in got.iter().enumerate() {
            let raw = if ((packed[sample >> 3] >> (sample & 7)) & 1) != 0 {
                2.0_f32
            } else {
                0.0_f32
            };
            assert!((*value - (raw - 2.0 * p)).abs() < 1e-6);
        }
    }

    #[test]
    fn assoc_double_buffer_preserves_order_and_values() {
        let dir = grm_fixture_dir();
        let meta = valid_grm_meta();
        write_grm_fixture(
            &dir,
            &meta,
            VALID_IDV,
            &[0b0000_0011, 0b0000_1111, 0b0000_0001],
        );
        let prefix = dir.join("fixture");
        let source = KfileAssocSource::open(&prefix, &[0, 1, 2, 3]).unwrap();
        let mut observed = Vec::<(usize, Vec<f32>, f32)>::new();
        let rows = source
            .for_each_centered_block(1, true, |row_start, rows, dosage, maf| {
                observed.push((row_start, dosage[..rows * 4].to_vec(), maf[0]));
                Ok::<(), String>(())
            })
            .expect("double-buffered association scan");
        assert_eq!(rows, 3);
        assert_eq!(observed.len(), 3);
        assert_eq!(
            observed.iter().map(|entry| entry.0).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(observed[0].1, vec![1.0, 1.0, -1.0, -1.0]);
        assert_eq!(observed[1].1, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(observed[2].1, vec![1.5, -0.5, -0.5, -0.5]);
        fs::remove_dir_all(dir).expect("remove fixture directory");
    }

    #[test]
    fn assoc_prefetch_reuses_two_buffers_without_reordering() {
        let dir = grm_fixture_dir();
        let meta = valid_grm_meta();
        write_grm_fixture(
            &dir,
            &meta,
            VALID_IDV,
            &[0b0000_0011, 0b0000_1111, 0b0000_0001],
        );
        let prefix = dir.join("fixture");
        let source = KfileAssocSource::open(&prefix, &[0, 1, 2, 3]).unwrap();
        let mut prefetch = source.into_prefetch(1).expect("start k-file prefetch");
        let mut starts = Vec::new();
        for _ in 0..3 {
            let block = prefetch
                .next()
                .expect("receive prefetch block")
                .expect("prefetch block present");
            starts.push(block.row_start());
            assert_eq!(block.rows(), 1);
            assert_eq!(block.dosage().len(), 4);
            prefetch.recycle(block).expect("recycle prefetch buffer");
        }
        assert_eq!(starts, vec![0, 1, 2]);
        drop(prefetch);
        fs::remove_dir_all(dir).expect("remove fixture directory");
    }

    #[test]
    fn decodes_little_endian_presence_bits_to_dosage() {
        let (dosage, presence) =
            decode_bsite_chunk(&[0b0101_0011], 6, 1, 0, &[]).expect("decode chunk");
        assert_eq!(dosage, vec![2.0, 2.0, 0.0, 0.0, 2.0, 0.0]);
        assert_eq!(presence, vec![3]);
    }

    #[test]
    fn decodes_selected_samples_in_requested_order() {
        let (dosage, presence) =
            decode_bsite_chunk(&[0b0101_0011, 0b0000_1000], 6, 2, 0, &[5, 0, 3])
                .expect("decode selected chunk");
        assert_eq!(dosage, vec![0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
        assert_eq!(presence, vec![1, 1]);
    }

    #[test]
    fn assoc_source_decodes_centered_zero_two_rows_in_selected_order() {
        let dir = grm_fixture_dir();
        let meta = valid_grm_meta();
        write_grm_fixture(
            &dir,
            &meta,
            VALID_IDV,
            &[0b0000_0011, 0b0000_1111, 0b0000_0001],
        );
        let prefix = dir.join("fixture");
        let mut source = KfileAssocSource::open(&prefix, &[3, 1, 0]).unwrap();
        let mut dosage = vec![0.0_f32; 2 * 3];
        let mut maf = vec![0.0_f32; 2];
        let rows = source
            .next_centered_block(2, &mut dosage, &mut maf)
            .unwrap()
            .unwrap();
        assert_eq!(rows, 2);
        assert!((maf[0] - (1.0 / 3.0)).abs() < 1e-6);
        assert_eq!(maf[1], 0.0);
        let expected = [-4.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.0, 0.0, 0.0];
        for (got, want) in dosage.iter().zip(expected) {
            assert!((*got as f64 - want).abs() < 1e-6, "got={got} want={want}");
        }

        let mut dosage_tail = vec![0.0_f32; 3];
        let mut maf_tail = vec![0.0_f32; 1];
        assert_eq!(
            source
                .next_centered_block(1, &mut dosage_tail, &mut maf_tail)
                .unwrap()
                .unwrap(),
            1
        );
        assert!((maf_tail[0] - (1.0 / 3.0)).abs() < 1e-6);
        let expected_tail = [-2.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0];
        for (got, want) in dosage_tail.iter().zip(expected_tail) {
            assert!((*got as f64 - want).abs() < 1e-6, "got={got} want={want}");
        }
        assert!(source
            .next_centered_block(1, &mut dosage_tail, &mut maf_tail)
            .unwrap()
            .is_none());
        fs::remove_dir_all(dir).expect("remove fixture directory");
    }

    #[test]
    fn assoc_source_random_rows_do_not_advance_sequential_cursor() {
        let dir = grm_fixture_dir();
        let meta = valid_grm_meta();
        write_grm_fixture(
            &dir,
            &meta,
            VALID_IDV,
            &[0b0000_0011, 0b0000_1111, 0b0000_0001],
        );
        let prefix = dir.join("fixture");
        let mut source = KfileAssocSource::open(&prefix, &[3, 1, 0, 2]).unwrap();
        let (random, random_maf) = source.read_centered_rows_at(&[2, 0]).unwrap();
        assert_eq!(random.len(), 8);
        assert_eq!(random_maf.len(), 2);
        let expected_random = [-0.5_f32, -0.5, 1.5, -0.5, -1.0, 1.0, 1.0, -1.0];
        for (got, want) in random.iter().zip(expected_random) {
            assert!((*got - want).abs() < 1e-6, "got={got} want={want}");
        }
        assert!((random_maf[0] - 0.25).abs() < 1e-6);
        assert_eq!(random_maf[1], 0.5);
        assert_eq!(source.next_row, 0);

        let mut dosage = vec![0.0_f32; 4];
        let mut maf = vec![0.0_f32; 1];
        assert_eq!(
            source
                .next_centered_block(1, &mut dosage, &mut maf)
                .unwrap(),
            Some(1)
        );
        let expected_first = [-1.0_f32, 1.0, 1.0, -1.0];
        for (got, want) in dosage.iter().zip(expected_first) {
            assert!((*got - want).abs() < 1e-6, "got={got} want={want}");
        }
        fs::remove_dir_all(dir).expect("remove fixture directory");
    }

    #[test]
    fn maf_uses_presence_frequency_without_filtering() {
        assert_eq!(maf_from_presence(0, 4), 0.0);
        assert_eq!(maf_from_presence(1, 4), 0.25);
        assert_eq!(maf_from_presence(3, 4), 0.25);
        assert_eq!(maf_from_presence(4, 4), 0.0);
    }

    #[test]
    fn chunk_row_offset_is_preserved() {
        let (dosage, presence) =
            decode_bsite_chunk(&[0b0000_0001, 0b0000_0010], 2, 2, 7, &[]).expect("decode rows");
        assert_eq!(dosage, vec![2.0, 0.0, 0.0, 2.0]);
        assert_eq!(presence, vec![1, 1]);
    }

    #[test]
    fn grm_source_accepts_prefix_or_metadata_and_resets_scan_without_resetting_timing() {
        let dir = grm_fixture_dir();
        let meta = valid_grm_meta();
        write_grm_fixture(
            &dir,
            &meta,
            VALID_IDV,
            &[0b0000_0011, 0b0000_1111, 0b0000_0001],
        );
        let prefix = dir.join("fixture");
        let meta_path = PathBuf::from(format!("{}.meta.json", prefix.display()));
        let mut from_prefix = KfileGrmSource::open(&prefix, None, 1, 0.25).unwrap();
        let from_meta = KfileGrmSource::open(&meta_path, None, 1, 0.25).unwrap();
        assert_eq!(from_prefix.sample_ids(), from_meta.sample_ids());
        assert_eq!(from_prefix.n_samples(), from_meta.n_samples());

        let first = from_prefix.next_prepared_block(3).unwrap().unwrap();
        let mut first_values = vec![0.0; first.retained_rows() * 4];
        from_prefix
            .decode_block_into(&first, 0..4, &mut first_values)
            .unwrap();
        let timing_before_reset = from_prefix.stage_timing();
        from_prefix.reset_scan().unwrap();
        assert_eq!(from_prefix.stage_timing(), timing_before_reset);
        assert_eq!(from_prefix.scanned_rows(), 0);
        assert_eq!(from_prefix.effective_rows(), 0);
        assert_eq!(from_prefix.denominator(), 0.0);
        let repeated = from_prefix.next_prepared_block(3).unwrap().unwrap();
        let mut repeated_values = vec![0.0; repeated.retained_rows() * 4];
        from_prefix
            .decode_block_into(&repeated, 0..4, &mut repeated_values)
            .unwrap();
        assert_eq!(first_values, repeated_values);
        assert_eq!(first.denominator, repeated.denominator);
        fs::remove_dir_all(dir).expect("remove fixture directory");
    }

    #[test]
    fn grm_source_decodes_selected_position_stripes_in_grm_sample_order() {
        let dir = grm_fixture_dir();
        let meta = valid_grm_meta();
        write_grm_fixture(
            &dir,
            &meta,
            VALID_IDV,
            &[0b0000_0011, 0b0000_0001, 0b0000_1110],
        );
        let prefix = dir.join("fixture");
        let mut source = KfileGrmSource::open(&prefix, Some(&[3, 1, 0, 2]), 1, 0.0).unwrap();
        let block = source.next_prepared_block(3).unwrap().unwrap();
        let rows = block.retained_rows();
        let mut full = vec![0.0; rows * 4];
        let mut left = vec![0.0; rows * 2];
        let mut right = vec![0.0; rows * 2];
        source.decode_block_into(&block, 0..4, &mut full).unwrap();
        source.decode_block_into(&block, 0..2, &mut left).unwrap();
        source.decode_block_into(&block, 2..4, &mut right).unwrap();
        let mut combined = vec![0.0; rows * 4];
        for row in 0..rows {
            combined[row * 4..row * 4 + 2].copy_from_slice(&left[row * 2..row * 2 + 2]);
            combined[row * 4 + 2..row * 4 + 4].copy_from_slice(&right[row * 2..row * 2 + 2]);
        }
        assert_eq!(combined, full);
        fs::remove_dir_all(dir).expect("remove fixture directory");
    }

    #[test]
    fn grm_source_requires_binary_presence_without_changing_chunk_reader() {
        let dir = grm_fixture_dir();
        let mut meta = valid_grm_meta();
        meta.value_type = "dosage".to_string();
        write_grm_fixture(&dir, &meta, VALID_IDV, &[3, 15, 1]);
        let prefix = dir.join("fixture");
        assert!(KfileChunkReader::new(prefix.to_string_lossy().into_owned(), None, None).is_ok());
        assert!(KfileGrmSource::open(&prefix, None, 1, 0.0).is_err());
        fs::remove_dir_all(dir).expect("remove fixture directory");
    }

    #[test]
    fn grm_source_rejects_invalid_metadata_and_selection() {
        let idv_cases = [
            (
                "duplicate idv sample",
                "#idx\tsample_id\tkmc_prefix\n0\ts0\tp0\n1\ts0\tp1\n2\ts2\tp2\n3\ts3\tp3\n",
            ),
            (
                "non-contiguous idv index",
                "#idx\tsample_id\tkmc_prefix\n0\ts0\tp0\n2\ts1\tp1\n3\ts2\tp2\n4\ts3\tp3\n",
            ),
        ];
        for (_name, idv_rows) in idv_cases {
            let dir = grm_fixture_dir();
            let meta = valid_grm_meta();
            write_grm_fixture(&dir, &meta, idv_rows, &[3, 15, 1]);
            assert!(KfileGrmSource::open(&dir.join("fixture"), None, 1, 0.0).is_err());
            fs::remove_dir_all(dir).expect("remove fixture directory");
        }

        for (name, field, value) in [
            ("non-binary value type", "value_type", "dosage"),
            ("unsupported bit order", "bit_order", "big_bit_order"),
            ("unsupported compression", "compression", "zstd"),
        ] {
            let dir = grm_fixture_dir();
            let mut meta = valid_grm_meta();
            match field {
                "value_type" => meta.value_type = value.to_string(),
                "bit_order" => meta.bit_order = value.to_string(),
                "compression" => meta.compression = value.to_string(),
                _ => unreachable!("unknown metadata field"),
            }
            write_grm_fixture(&dir, &meta, VALID_IDV, &[3, 15, 1]);
            assert!(
                KfileGrmSource::open(&dir.join("fixture"), None, 1, 0.0).is_err(),
                "{name} must fail"
            );
            fs::remove_dir_all(dir).expect("remove fixture directory");
        }

        let dir = grm_fixture_dir();
        let meta = valid_grm_meta();
        write_grm_fixture(&dir, &meta, VALID_IDV, &[3, 15, 1]);
        let prefix = dir.join("fixture");
        assert!(KfileGrmSource::open(&prefix, None, 3, 0.0).is_err());
        assert!(KfileGrmSource::open(&prefix, None, 1, -0.01).is_err());
        assert!(KfileGrmSource::open(&prefix, None, 1, 0.51).is_err());
        assert!(KfileGrmSource::open(&prefix, Some(&[0, 0]), 1, 0.0).is_err());
        assert!(KfileGrmSource::open(&prefix, Some(&[4]), 1, 0.0).is_err());
        fs::remove_file(dir.join("fixture.bsite")).expect("remove bsite");
        write_grm_fixture(&dir, &meta, VALID_IDV, &[3, 15]);
        assert!(KfileGrmSource::open(&prefix, None, 1, 0.0).is_err());
        fs::remove_dir_all(dir).expect("remove fixture directory");
    }
}
