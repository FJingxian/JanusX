use crate::kmer::format::{KmergeMeta, SampleEntry, BSITE_HEADER_SIZE};
use crate::kmer::progress::ProgressFn;
use anyhow::{bail, Context, Result};
use memmap2::MmapOptions;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
#[cfg(target_arch = "aarch64")]
use std::sync::OnceLock;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::sync::OnceLock;

#[cfg(target_arch = "x86")]
use std::arch::x86 as x86_simd;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as x86_simd;

#[derive(Clone, Debug)]
pub(crate) struct CompareGroupInfo {
    pub name: String,
    pub sample_ids: Vec<String>,
}

#[derive(Clone, Debug)]
struct CompareGroupMask {
    pub name: String,
    pub sample_ids: Vec<String>,
    pub mask_u64: Option<u64>,
    pub byte_mask: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(crate) struct KbinComparePlan {
    pub meta_path: PathBuf,
    pub idv_path: PathBuf,
    pub bsite_path: PathBuf,
    pub k: u32,
    pub n_samples: usize,
    pub n_kmers: u64,
    pub bytes_per_col: usize,
    pub scan_mode: String,
    pub groups: Vec<CompareGroupInfo>,
    group_masks: Vec<CompareGroupMask>,
}

#[derive(Clone, Debug)]
pub(crate) struct KbinPatternScan {
    pub pattern_counts: Vec<u64>,
    pub scan_bytes: u64,
    pub scan_mode: String,
}

#[derive(Clone, Debug)]
pub(crate) struct KbinCompareResult {
    pub groups_path: PathBuf,
    pub patterns_path: PathBuf,
    pub pairs_path: PathBuf,
}

pub(crate) fn inspect_kbin_compare(
    prefix_or_meta: &Path,
    raw_compare_groups: &[String],
) -> Result<KbinComparePlan> {
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
    let idv_path = base_dir.join(&meta.idv_file);
    let bsite_path = base_dir.join(&meta.bsite_file);
    if !idv_path.is_file() {
        bail!("idv file not found: {}", idv_path.display());
    }
    if !bsite_path.is_file() {
        bail!("bsite file not found: {}", bsite_path.display());
    }

    let samples = read_idv_file(&idv_path)?;
    let n_samples = usize::try_from(meta.n_samples).context("n_samples does not fit usize")?;
    if samples.len() != n_samples {
        bail!(
            "meta/idv mismatch: meta n_samples={}, idv rows={}",
            meta.n_samples,
            samples.len()
        );
    }
    let bytes_per_col =
        usize::try_from(meta.bytes_per_col).context("bytes_per_col does not fit usize")?;
    let expected_bytes = n_samples.div_ceil(8);
    if bytes_per_col != expected_bytes {
        bail!(
            "invalid bytes_per_col in meta: {} (expected {})",
            meta.bytes_per_col,
            expected_bytes
        );
    }

    let group_masks = parse_compare_groups(raw_compare_groups, &samples, bytes_per_col)?;
    if group_masks.len() < 2 {
        bail!("`-compare` requires at least 2 groups");
    }
    if group_masks.len() > 16 {
        bail!("`-compare` supports at most 16 groups");
    }

    let groups = group_masks
        .iter()
        .map(|group| CompareGroupInfo {
            name: group.name.clone(),
            sample_ids: group.sample_ids.clone(),
        })
        .collect::<Vec<_>>();
    let scan_mode = detect_scan_mode(n_samples, bytes_per_col);

    Ok(KbinComparePlan {
        meta_path,
        idv_path,
        bsite_path,
        k: meta.k,
        n_samples,
        n_kmers: meta.n_kmers,
        bytes_per_col,
        scan_mode,
        groups,
        group_masks,
    })
}

pub(crate) fn planned_kbin_outputs(out_dir: &Path, prefix: &str) -> Vec<PathBuf> {
    vec![
        out_dir.join(format!("{prefix}.compare.groups.tsv")),
        out_dir.join(format!("{prefix}.compare.patterns.tsv")),
        out_dir.join(format!("{prefix}.compare.pairs.tsv")),
    ]
}

pub(crate) fn scan_kbin_compare(
    plan: &KbinComparePlan,
    threads: usize,
    progress_callback: Option<ProgressFn>,
) -> Result<KbinPatternScan> {
    let file = File::open(&plan.bsite_path)
        .with_context(|| format!("failed to open bsite file: {}", plan.bsite_path.display()))?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .with_context(|| format!("failed to mmap bsite file: {}", plan.bsite_path.display()))?
    };
    if mmap.len() < BSITE_HEADER_SIZE {
        bail!("truncated bsite file: {}", plan.bsite_path.display());
    }
    let header = parse_bsite_header(&mmap[..BSITE_HEADER_SIZE], &plan.bsite_path)?;
    if header.n_samples != plan.n_samples as u64
        || header.n_kmers != plan.n_kmers
        || header.bytes_per_col != plan.bytes_per_col as u64
    {
        bail!(
            "bsite header does not match meta for {}",
            plan.bsite_path.display()
        );
    }

    let scan_bytes = plan
        .n_kmers
        .checked_mul(plan.bytes_per_col as u64)
        .ok_or_else(|| anyhow::anyhow!("scan byte size overflow"))?;
    let body_len = usize::try_from(scan_bytes).context("scan byte size does not fit usize")?;
    if mmap.len() < BSITE_HEADER_SIZE.saturating_add(body_len) {
        bail!("truncated bsite payload: {}", plan.bsite_path.display());
    }
    let body = &mmap[BSITE_HEADER_SIZE..BSITE_HEADER_SIZE + body_len];

    let chunk_rows = chunk_rows_for_scan(plan.bytes_per_col);
    let ranges = build_row_ranges(
        usize::try_from(plan.n_kmers).context("n_kmers does not fit usize")?,
        chunk_rows,
    );
    let pattern_space = 1usize << plan.group_masks.len();
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .context("failed to build kstats kbin thread pool")?;
    let progress_done = Arc::new(AtomicU64::new(0));
    let callback = progress_callback.clone();

    let (partials, scan_mode) = match typed_body_view(body, plan.bytes_per_col) {
        TypedBodyView::U8(values) => (
            pool.install(|| {
                ranges
                    .into_par_iter()
                    .map(|(start, end)| {
                        let local = scan_range_packed_u8(
                            values,
                            start,
                            end,
                            &plan.group_masks,
                            pattern_space,
                        );
                        report_chunk_progress(
                            callback.as_ref(),
                            progress_done.as_ref(),
                            (end - start) as u64,
                            plan.n_kmers,
                        )?;
                        Ok(local)
                    })
                    .collect::<Result<Vec<_>>>()
            })?,
            "packed-u8-scalar-rayon".to_string(),
        ),
        TypedBodyView::U16(values) => (
            pool.install(|| {
                ranges
                    .into_par_iter()
                    .map(|(start, end)| {
                        let local = scan_range_packed_u16(
                            values,
                            start,
                            end,
                            &plan.group_masks,
                            pattern_space,
                        );
                        report_chunk_progress(
                            callback.as_ref(),
                            progress_done.as_ref(),
                            (end - start) as u64,
                            plan.n_kmers,
                        )?;
                        Ok(local)
                    })
                    .collect::<Result<Vec<_>>>()
            })?,
            "packed-u16-scalar-rayon".to_string(),
        ),
        TypedBodyView::U32(values) => {
            let kernel = detect_packed_u32_kernel();
            let mode = packed_u32_kernel_name(kernel).to_string();
            (
                pool.install(|| {
                    ranges
                        .into_par_iter()
                        .map(|(start, end)| {
                            let local = match kernel {
                                PackedU32Kernel::Scalar => scan_range_packed_u32_scalar(
                                    values,
                                    start,
                                    end,
                                    &plan.group_masks,
                                    pattern_space,
                                ),
                                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                                PackedU32Kernel::Avx2 => {
                                    // SAFETY: gated by runtime AVX2 detection.
                                    unsafe {
                                        scan_range_packed_u32_avx2(
                                            values,
                                            start,
                                            end,
                                            &plan.group_masks,
                                            pattern_space,
                                        )
                                    }
                                }
                                #[cfg(target_arch = "aarch64")]
                                PackedU32Kernel::Neon => {
                                    // SAFETY: gated by runtime NEON detection.
                                    unsafe {
                                        scan_range_packed_u32_neon(
                                            values,
                                            start,
                                            end,
                                            &plan.group_masks,
                                            pattern_space,
                                        )
                                    }
                                }
                            };
                            report_chunk_progress(
                                callback.as_ref(),
                                progress_done.as_ref(),
                                (end - start) as u64,
                                plan.n_kmers,
                            )?;
                            Ok(local)
                        })
                        .collect::<Result<Vec<_>>>()
                })?,
                mode,
            )
        }
        TypedBodyView::U64(values) => (
            pool.install(|| {
                ranges
                    .into_par_iter()
                    .map(|(start, end)| {
                        let local = scan_range_packed_u64(
                            values,
                            start,
                            end,
                            &plan.group_masks,
                            pattern_space,
                        );
                        report_chunk_progress(
                            callback.as_ref(),
                            progress_done.as_ref(),
                            (end - start) as u64,
                            plan.n_kmers,
                        )?;
                        Ok(local)
                    })
                    .collect::<Result<Vec<_>>>()
            })?,
            "packed-u64-scalar-rayon".to_string(),
        ),
        TypedBodyView::Bytes => (
            pool.install(|| {
                ranges
                    .into_par_iter()
                    .map(|(start, end)| {
                        let local = scan_range_generic_bytes(
                            body,
                            plan.bytes_per_col,
                            start,
                            end,
                            &plan.group_masks,
                            pattern_space,
                        );
                        report_chunk_progress(
                            callback.as_ref(),
                            progress_done.as_ref(),
                            (end - start) as u64,
                            plan.n_kmers,
                        )?;
                        Ok(local)
                    })
                    .collect::<Result<Vec<_>>>()
            })?,
            "generic-bytes-scalar-rayon".to_string(),
        ),
    };

    let mut pattern_counts = vec![0u64; pattern_space];
    for local in partials {
        for (dst, src) in pattern_counts.iter_mut().zip(local.into_iter()) {
            *dst = dst.saturating_add(src);
        }
    }
    Ok(KbinPatternScan {
        pattern_counts,
        scan_bytes,
        scan_mode,
    })
}

pub(crate) fn write_kbin_compare_outputs(
    out_dir: &Path,
    prefix: &str,
    plan: &KbinComparePlan,
    scan: &KbinPatternScan,
    progress_callback: Option<ProgressFn>,
) -> Result<KbinCompareResult> {
    let groups_path = out_dir.join(format!("{prefix}.compare.groups.tsv"));
    let patterns_path = out_dir.join(format!("{prefix}.compare.patterns.tsv"));
    let pairs_path = out_dir.join(format!("{prefix}.compare.pairs.tsv"));

    let observed = observed_pattern_rows(&scan.pattern_counts, &plan.groups);
    let pair_rows = pair_summary_rows(&scan.pattern_counts, &plan.groups);
    let total_rows = plan
        .groups
        .len()
        .saturating_add(observed.len())
        .saturating_add(pair_rows.len())
        .max(1) as u64;
    if let Some(cb) = progress_callback.as_ref() {
        cb(0, total_rows)?;
    }
    let mut done = 0u64;

    write_group_defs(&groups_path, &plan.groups)?;
    advance_write_progress(
        progress_callback.as_ref(),
        &mut done,
        total_rows,
        plan.groups.len() as u64,
    )?;

    write_pattern_rows(&patterns_path, &plan.groups, &observed)?;
    advance_write_progress(
        progress_callback.as_ref(),
        &mut done,
        total_rows,
        observed.len() as u64,
    )?;

    write_pair_rows(&pairs_path, &pair_rows)?;
    advance_write_progress(
        progress_callback.as_ref(),
        &mut done,
        total_rows,
        pair_rows.len() as u64,
    )?;

    Ok(KbinCompareResult {
        groups_path,
        patterns_path,
        pairs_path,
    })
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
    Ok(())
}

fn read_idv_file(path: &Path) -> Result<Vec<SampleEntry>> {
    let mut rows = Vec::new();
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .with_context(|| format!("failed to open idv file: {}", path.display()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("failed to read idv header: {}", path.display()))?
        .clone();
    if headers.len() < 3
        || headers.get(0) != Some("#idx")
        || headers.get(1) != Some("sample_id")
        || headers.get(2) != Some("kmc_prefix")
    {
        bail!("invalid idv header in {}", path.display());
    }
    for rec in reader.records() {
        let rec = rec.with_context(|| format!("failed to parse idv row: {}", path.display()))?;
        let index = rec
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("missing idx in {}", path.display()))?
            .parse::<u32>()
            .with_context(|| format!("invalid idx in {}", path.display()))?;
        let sample_id = rec
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("missing sample_id in {}", path.display()))?
            .to_string();
        let kmc_prefix = rec
            .get(2)
            .ok_or_else(|| anyhow::anyhow!("missing kmc_prefix in {}", path.display()))?
            .to_string();
        rows.push(SampleEntry {
            index,
            sample_id,
            kmc_prefix,
        });
    }
    rows.sort_by_key(|row| row.index);
    for (expected, row) in rows.iter().enumerate() {
        if row.index as usize != expected {
            bail!(
                "idv indices must be contiguous from 0 in {} (saw {} at row {})",
                path.display(),
                row.index,
                expected
            );
        }
    }
    Ok(rows)
}

fn parse_compare_groups(
    raw_compare_groups: &[String],
    samples: &[SampleEntry],
    bytes_per_col: usize,
) -> Result<Vec<CompareGroupMask>> {
    if raw_compare_groups.is_empty() {
        bail!("`-compare` requires at least 2 group definitions");
    }
    let sample_index = samples
        .iter()
        .map(|sample| (sample.sample_id.as_str(), sample.index as usize))
        .collect::<std::collections::HashMap<_, _>>();
    let mut groups = Vec::with_capacity(raw_compare_groups.len());
    for (group_idx, raw_group) in raw_compare_groups.iter().enumerate() {
        let (name, members_raw) = if let Some((name, members)) = raw_group.split_once('=') {
            (name.trim().to_string(), members)
        } else {
            (format!("G{}", group_idx + 1), raw_group.as_str())
        };
        if name.trim().is_empty() {
            bail!("invalid compare group: `{raw_group}`");
        }
        let mut sample_ids = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for token in members_raw.split(',') {
            let sample_id = token.trim();
            if sample_id.is_empty() || !seen.insert(sample_id.to_string()) {
                continue;
            }
            if !sample_index.contains_key(sample_id) {
                bail!("unknown sample in `-compare`: `{sample_id}`");
            }
            sample_ids.push(sample_id.to_string());
        }
        if sample_ids.is_empty() {
            bail!("compare group `{name}` has no samples");
        }
        let mut byte_mask = vec![0u8; bytes_per_col];
        let mut mask_u64 = 0u64;
        for sample_id in &sample_ids {
            let idx = *sample_index
                .get(sample_id.as_str())
                .ok_or_else(|| anyhow::anyhow!("unknown sample: {sample_id}"))?;
            byte_mask[idx >> 3] |= 1u8 << (idx & 7);
            if samples.len() <= 64 {
                mask_u64 |= 1u64 << idx;
            }
        }
        groups.push(CompareGroupMask {
            name,
            sample_ids,
            mask_u64: (samples.len() <= 64).then_some(mask_u64),
            byte_mask,
        });
    }
    Ok(groups)
}

fn detect_scan_mode(n_samples: usize, bytes_per_col: usize) -> String {
    match (n_samples, bytes_per_col) {
        (0..=8, 1) => "packed-u8".to_string(),
        (9..=16, 2) => "packed-u16".to_string(),
        (17..=32, 4) => "packed-u32-runtime".to_string(),
        (33..=64, 8) => "packed-u64".to_string(),
        _ => "generic-bytes".to_string(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedU32Kernel {
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

fn packed_u32_kernel_name(kernel: PackedU32Kernel) -> &'static str {
    match kernel {
        PackedU32Kernel::Scalar => "packed-u32-scalar-rayon",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PackedU32Kernel::Avx2 => "packed-u32-avx2-rayon",
        #[cfg(target_arch = "aarch64")]
        PackedU32Kernel::Neon => "packed-u32-neon-rayon",
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn avx2_runtime_available() -> bool {
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

#[cfg(target_arch = "aarch64")]
fn neon_runtime_available() -> bool {
    static NEON: OnceLock<bool> = OnceLock::new();
    *NEON.get_or_init(|| std::arch::is_aarch64_feature_detected!("neon"))
}

fn detect_packed_u32_kernel() -> PackedU32Kernel {
    #[cfg(target_arch = "aarch64")]
    if neon_runtime_available() {
        return PackedU32Kernel::Neon;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if avx2_runtime_available() {
        return PackedU32Kernel::Avx2;
    }
    PackedU32Kernel::Scalar
}

#[derive(Clone, Copy, Debug)]
struct ParsedBsiteHeader {
    n_samples: u64,
    n_kmers: u64,
    bytes_per_col: u64,
}

fn parse_bsite_header(header: &[u8], path: &Path) -> Result<ParsedBsiteHeader> {
    if header.len() != BSITE_HEADER_SIZE {
        bail!("invalid bsite header size in {}", path.display());
    }
    if &header[..8] != b"JXBSIT1\0" {
        bail!("invalid bsite magic in {}", path.display());
    }
    let version = u32::from_le_bytes(header[8..12].try_into().expect("version slice"));
    let layout = u32::from_le_bytes(header[12..16].try_into().expect("layout slice"));
    if version != 1 {
        bail!(
            "unsupported bsite version in {}: {}",
            path.display(),
            version
        );
    }
    if layout != 1 {
        bail!("unsupported bsite layout in {}: {}", path.display(), layout);
    }
    Ok(ParsedBsiteHeader {
        n_samples: u64::from_le_bytes(header[16..24].try_into().expect("n_samples slice")),
        n_kmers: u64::from_le_bytes(header[24..32].try_into().expect("n_kmers slice")),
        bytes_per_col: u64::from_le_bytes(header[32..40].try_into().expect("bytes_per_col slice")),
    })
}

fn chunk_rows_for_scan(bytes_per_col: usize) -> usize {
    let target_bytes = 64usize * 1024 * 1024;
    (target_bytes / bytes_per_col.max(1)).max(1)
}

fn build_row_ranges(total_rows: usize, chunk_rows: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < total_rows {
        let end = (start + chunk_rows).min(total_rows);
        out.push((start, end));
        start = end;
    }
    if out.is_empty() {
        out.push((0, 0));
    }
    out
}

enum TypedBodyView<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
    Bytes,
}

fn typed_body_view<'a>(body: &'a [u8], bytes_per_col: usize) -> TypedBodyView<'a> {
    match bytes_per_col {
        1 => TypedBodyView::U8(body),
        2 => {
            let (prefix, values, suffix) = unsafe { body.align_to::<u16>() };
            if prefix.is_empty() && suffix.is_empty() {
                TypedBodyView::U16(values)
            } else {
                TypedBodyView::Bytes
            }
        }
        4 => {
            let (prefix, values, suffix) = unsafe { body.align_to::<u32>() };
            if prefix.is_empty() && suffix.is_empty() {
                TypedBodyView::U32(values)
            } else {
                TypedBodyView::Bytes
            }
        }
        8 => {
            let (prefix, values, suffix) = unsafe { body.align_to::<u64>() };
            if prefix.is_empty() && suffix.is_empty() {
                TypedBodyView::U64(values)
            } else {
                TypedBodyView::Bytes
            }
        }
        _ => TypedBodyView::Bytes,
    }
}

fn scan_range_packed_u8(
    values: &[u8],
    start: usize,
    end: usize,
    groups: &[CompareGroupMask],
    pattern_space: usize,
) -> Vec<u64> {
    let mut local = vec![0u64; pattern_space];
    for &value in &values[start..end] {
        let pattern = pattern_from_u64(value as u64, groups);
        local[pattern as usize] += 1;
    }
    local
}

fn scan_range_packed_u16(
    values: &[u16],
    start: usize,
    end: usize,
    groups: &[CompareGroupMask],
    pattern_space: usize,
) -> Vec<u64> {
    let mut local = vec![0u64; pattern_space];
    for &value in &values[start..end] {
        let pattern = pattern_from_u64(value as u64, groups);
        local[pattern as usize] += 1;
    }
    local
}

fn scan_range_packed_u32_scalar(
    values: &[u32],
    start: usize,
    end: usize,
    groups: &[CompareGroupMask],
    pattern_space: usize,
) -> Vec<u64> {
    let mut local = vec![0u64; pattern_space];
    for &value in &values[start..end] {
        let pattern = pattern_from_u64(value as u64, groups);
        local[pattern as usize] += 1;
    }
    local
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn scan_range_packed_u32_avx2(
    values: &[u32],
    start: usize,
    end: usize,
    groups: &[CompareGroupMask],
    pattern_space: usize,
) -> Vec<u64> {
    use x86_simd::*;

    let mut local = vec![0u64; pattern_space];
    let mut idx = start;
    let zero = _mm256_setzero_si256();
    let group_masks = groups
        .iter()
        .map(|group| group.mask_u64.expect("u32 fast path requires bit masks") as u32)
        .collect::<Vec<_>>();

    while idx + 8 <= end {
        let vals = _mm256_loadu_si256(values.as_ptr().add(idx) as *const __m256i);
        let mut lane_masks = [0u8; 16];
        for (bit_idx, &mask) in group_masks.iter().enumerate() {
            let mask_vec = _mm256_set1_epi32(mask as i32);
            let anded = _mm256_and_si256(vals, mask_vec);
            let is_zero = _mm256_cmpeq_epi32(anded, zero);
            let zero_mask = _mm256_movemask_ps(_mm256_castsi256_ps(is_zero)) as u32;
            lane_masks[bit_idx] = ((!zero_mask) & 0xFF) as u8;
        }
        let patterns = lane_patterns_from_group_masks(&lane_masks[..group_masks.len()]);
        for &pattern in &patterns {
            local[pattern as usize] += 1;
        }
        idx += 8;
    }

    for &value in &values[idx..end] {
        let pattern = pattern_from_u64(value as u64, groups);
        local[pattern as usize] += 1;
    }
    local
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scan_range_packed_u32_neon(
    values: &[u32],
    start: usize,
    end: usize,
    groups: &[CompareGroupMask],
    pattern_space: usize,
) -> Vec<u64> {
    use std::arch::aarch64::*;

    let mut local = vec![0u64; pattern_space];
    let mut idx = start;
    let zero = vdupq_n_u32(0);
    let group_masks = groups
        .iter()
        .map(|group| group.mask_u64.expect("u32 fast path requires bit masks") as u32)
        .collect::<Vec<_>>();

    while idx + 4 <= end {
        let vals = vld1q_u32(values.as_ptr().add(idx));
        let mut lane_masks = [0u8; 16];
        for (bit_idx, &mask) in group_masks.iter().enumerate() {
            let mask_vec = vdupq_n_u32(mask);
            let anded = vandq_u32(vals, mask_vec);
            let present = vcgtq_u32(anded, zero);
            let mut lanes = [0u32; 4];
            vst1q_u32(lanes.as_mut_ptr(), present);
            lane_masks[bit_idx] = (lanes[0] != 0) as u8
                | (((lanes[1] != 0) as u8) << 1)
                | (((lanes[2] != 0) as u8) << 2)
                | (((lanes[3] != 0) as u8) << 3);
        }
        let patterns = lane_patterns_from_group_masks(&lane_masks[..group_masks.len()]);
        for &pattern in &patterns[..4] {
            local[pattern as usize] += 1;
        }
        idx += 4;
    }

    for &value in &values[idx..end] {
        let pattern = pattern_from_u64(value as u64, groups);
        local[pattern as usize] += 1;
    }
    local
}

#[inline(always)]
fn lane_patterns_from_group_masks(group_lane_masks: &[u8]) -> [u16; 8] {
    let mut rows_lo = [0u8; 8];
    let mut rows_hi = [0u8; 8];
    for (idx, &mask) in group_lane_masks.iter().enumerate() {
        if idx < 8 {
            rows_lo[idx] = mask;
        } else if idx < 16 {
            rows_hi[idx - 8] = mask;
        } else {
            break;
        }
    }

    let cols_lo = bit_transpose_8x8(u64::from_le_bytes(rows_lo)).to_le_bytes();
    if group_lane_masks.len() <= 8 {
        let mut out = [0u16; 8];
        for lane in 0..8 {
            out[lane] = cols_lo[lane] as u16;
        }
        return out;
    }

    let cols_hi = bit_transpose_8x8(u64::from_le_bytes(rows_hi)).to_le_bytes();
    let mut out = [0u16; 8];
    for lane in 0..8 {
        out[lane] = cols_lo[lane] as u16 | ((cols_hi[lane] as u16) << 8);
    }
    out
}

#[inline(always)]
fn bit_transpose_8x8(mut x: u64) -> u64 {
    let mut t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AA;
    x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCC;
    x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0;
    x ^ (t ^ (t << 28))
}

fn scan_range_packed_u64(
    values: &[u64],
    start: usize,
    end: usize,
    groups: &[CompareGroupMask],
    pattern_space: usize,
) -> Vec<u64> {
    let mut local = vec![0u64; pattern_space];
    for &value in &values[start..end] {
        let pattern = pattern_from_u64(value, groups);
        local[pattern as usize] += 1;
    }
    local
}

fn scan_range_generic_bytes(
    body: &[u8],
    bytes_per_col: usize,
    start: usize,
    end: usize,
    groups: &[CompareGroupMask],
    pattern_space: usize,
) -> Vec<u64> {
    let mut local = vec![0u64; pattern_space];
    for row_idx in start..end {
        let offset = row_idx * bytes_per_col;
        let row = &body[offset..offset + bytes_per_col];
        let pattern = pattern_from_bytes(row, groups);
        local[pattern as usize] += 1;
    }
    local
}

fn pattern_from_u64(value: u64, groups: &[CompareGroupMask]) -> u16 {
    let mut pattern = 0u16;
    for (bit_idx, group) in groups.iter().enumerate() {
        if let Some(mask) = group.mask_u64 {
            if value & mask != 0 {
                pattern |= 1u16 << bit_idx;
            }
        } else {
            unreachable!("u64 fast path requires <=64 samples");
        }
    }
    pattern
}

fn pattern_from_bytes(row: &[u8], groups: &[CompareGroupMask]) -> u16 {
    let mut pattern = 0u16;
    for (bit_idx, group) in groups.iter().enumerate() {
        let mut any = false;
        for (a, b) in row.iter().zip(group.byte_mask.iter()) {
            if (*a & *b) != 0 {
                any = true;
                break;
            }
        }
        if any {
            pattern |= 1u16 << bit_idx;
        }
    }
    pattern
}

fn report_chunk_progress(
    callback: Option<&ProgressFn>,
    progress_done: &AtomicU64,
    advance: u64,
    total: u64,
) -> Result<()> {
    if let Some(cb) = callback {
        let done = progress_done
            .fetch_add(advance, AtomicOrdering::Relaxed)
            .saturating_add(advance)
            .min(total.max(1));
        cb(done, total.max(1))?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ObservedPatternRow {
    pattern_bits: String,
    group_names: String,
    bits: Vec<u8>,
    count: u64,
}

fn observed_pattern_rows(
    pattern_counts: &[u64],
    groups: &[CompareGroupInfo],
) -> Vec<ObservedPatternRow> {
    let mut rows = Vec::new();
    for (pattern, &count) in pattern_counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let mut bits = Vec::with_capacity(groups.len());
        let mut names = Vec::new();
        let mut text = String::with_capacity(groups.len());
        for (idx, group) in groups.iter().enumerate() {
            let present = ((pattern >> idx) & 1) as u8;
            bits.push(present);
            text.push(if present == 1 { '1' } else { '0' });
            if present == 1 {
                names.push(group.name.as_str());
            }
        }
        rows.push(ObservedPatternRow {
            pattern_bits: text,
            group_names: if names.is_empty() {
                ".".to_string()
            } else {
                names.join(",")
            },
            bits,
            count,
        });
    }
    rows.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| {
                let a_present = a.bits.iter().map(|&x| x as usize).sum::<usize>();
                let b_present = b.bits.iter().map(|&x| x as usize).sum::<usize>();
                b_present.cmp(&a_present)
            })
            .then_with(|| a.pattern_bits.cmp(&b.pattern_bits))
    });
    rows
}

#[derive(Clone, Debug)]
struct PairSummaryRow {
    group_a: String,
    group_b: String,
    unique_a: u64,
    unique_b: u64,
    intersection: u64,
    union: u64,
    only_a: u64,
    only_b: u64,
    both: u64,
    symmetric_diff: u64,
    jaccard: f64,
}

fn pair_summary_rows(pattern_counts: &[u64], groups: &[CompareGroupInfo]) -> Vec<PairSummaryRow> {
    let mut rows = Vec::new();
    for i in 0..groups.len() {
        for j in (i + 1)..groups.len() {
            let mut unique_a = 0u64;
            let mut unique_b = 0u64;
            let mut inter = 0u64;
            for (pattern, &count) in pattern_counts.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let a = ((pattern >> i) & 1) == 1;
                let b = ((pattern >> j) & 1) == 1;
                if a {
                    unique_a = unique_a.saturating_add(count);
                }
                if b {
                    unique_b = unique_b.saturating_add(count);
                }
                if a && b {
                    inter = inter.saturating_add(count);
                }
            }
            let union = unique_a.saturating_add(unique_b).saturating_sub(inter);
            let only_a = unique_a.saturating_sub(inter);
            let only_b = unique_b.saturating_sub(inter);
            let both = inter;
            let symmetric_diff = only_a.saturating_add(only_b);
            let jaccard = if union == 0 {
                0.0
            } else {
                inter as f64 / union as f64
            };
            rows.push(PairSummaryRow {
                group_a: groups[i].name.clone(),
                group_b: groups[j].name.clone(),
                unique_a,
                unique_b,
                intersection: inter,
                union,
                only_a,
                only_b,
                both,
                symmetric_diff,
                jaccard,
            });
        }
    }
    rows
}

fn write_group_defs(path: &Path, groups: &[CompareGroupInfo]) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create compare group file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(b"group_index\tgroup_name\tn_samples\tsample_ids\n")?;
    for (idx, group) in groups.iter().enumerate() {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}",
            idx + 1,
            group.name,
            group.sample_ids.len(),
            group.sample_ids.join(",")
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn write_pattern_rows(
    path: &Path,
    groups: &[CompareGroupInfo],
    rows: &[ObservedPatternRow],
) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create compare pattern file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(b"pattern_id\tcount\tn_present\tpattern_bits\tgroup_names")?;
    for group in groups {
        writer.write_all(b"\t")?;
        writer.write_all(group.name.as_bytes())?;
    }
    writer.write_all(b"\n")?;
    for (idx, row) in rows.iter().enumerate() {
        let n_present = row.bits.iter().map(|&x| x as u64).sum::<u64>();
        write!(
            writer,
            "{}\t{}\t{}\t{}\t{}",
            idx + 1,
            row.count,
            n_present,
            row.pattern_bits,
            row.group_names
        )?;
        for bit in &row.bits {
            write!(writer, "\t{bit}")?;
        }
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_pair_rows(path: &Path, rows: &[PairSummaryRow]) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create compare pair file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(
        b"group_a\tgroup_b\tunique_a\tunique_b\tintersection\tunion\tonly_a\tonly_b\tboth\tsymmetric_diff\tjaccard\n",
    )?;
    for row in rows {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.10}",
            row.group_a,
            row.group_b,
            row.unique_a,
            row.unique_b,
            row.intersection,
            row.union,
            row.only_a,
            row.only_b,
            row.both,
            row.symmetric_diff,
            row.jaccard
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn advance_write_progress(
    callback: Option<&ProgressFn>,
    done: &mut u64,
    total: u64,
    step: u64,
) -> Result<()> {
    *done = done.saturating_add(step).min(total.max(1));
    if let Some(cb) = callback {
        cb(*done, total.max(1))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        inspect_kbin_compare, planned_kbin_outputs, scan_kbin_compare, write_kbin_compare_outputs,
    };
    use crate::kmer::format::{BsiteHeader, KmergeMeta, SampleEntry};
    use crate::kmer::writer::{write_idv_file, write_meta_json};
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("janusx_{name}_{}_{}", std::process::id(), nanos))
    }

    #[test]
    fn kbin_compare_small_matrix_matches_expected_counts() {
        let dir = temp_dir("kbin_compare");
        fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("kmerge");
        let meta_path = dir.join("kmerge.meta.json");
        let idv_path = dir.join("kmerge.idv");
        let bsite_path = dir.join("kmerge.bsite");

        let samples = vec![
            SampleEntry {
                index: 0,
                sample_id: "sample_A".to_string(),
                kmc_prefix: "./sample_A".to_string(),
            },
            SampleEntry {
                index: 1,
                sample_id: "sample_B".to_string(),
                kmc_prefix: "./sample_B".to_string(),
            },
        ];
        write_idv_file(&idv_path, &samples).unwrap();

        let meta = KmergeMeta {
            format: "janusx-kmer-bitmatrix-v1".to_string(),
            k: 5,
            n_samples: 2,
            n_kmers: 5,
            bytes_per_col: 1,
            encoding: "ACGT_2BIT_A00_C01_G10_T11".to_string(),
            canonical: true,
            matrix_layout: "column_major_bitset".to_string(),
            value_type: "binary_presence".to_string(),
            bit_order: "little_bit_order".to_string(),
            bkmer_file: "kmerge.bkmer".to_string(),
            bsite_file: "kmerge.bsite".to_string(),
            idv_file: "kmerge.idv".to_string(),
            min_count: 1,
            min_presence_rate: 0.0,
            max_presence_rate: 1.0,
            bucket_bits: 8,
            compression: "none".to_string(),
        };
        write_meta_json(&meta_path, &meta).unwrap();

        let mut writer = File::create(&bsite_path).unwrap();
        BsiteHeader {
            n_samples: 2,
            n_kmers: 5,
            bytes_per_col: 1,
        }
        .write_to(&mut writer)
        .unwrap();
        writer
            .write_all(&[
                0b0000_0001,
                0b0000_0010,
                0b0000_0010,
                0b0000_0001,
                0b0000_0001,
            ])
            .unwrap();
        writer.flush().unwrap();

        let plan = inspect_kbin_compare(
            &prefix,
            &["A=sample_A".to_string(), "B=sample_B".to_string()],
        )
        .unwrap();
        assert_eq!(plan.scan_mode, "packed-u8");
        let scan = scan_kbin_compare(&plan, 2, None).unwrap();
        assert_eq!(scan.pattern_counts[0], 0);
        assert_eq!(scan.pattern_counts[1], 3);
        assert_eq!(scan.pattern_counts[2], 2);
        assert_eq!(scan.pattern_counts[3], 0);

        let outputs = write_kbin_compare_outputs(&dir, "cmp", &plan, &scan, None).unwrap();
        let pair_text = fs::read_to_string(outputs.pairs_path).unwrap();
        assert!(pair_text.contains("A\tB\t3\t2\t0\t5\t3\t2\t0\t5\t0.0000000000"));
        let pattern_text = fs::read_to_string(outputs.patterns_path).unwrap();
        assert!(pattern_text.contains("10\tA"));
        assert!(pattern_text.contains("01\tB"));

        let planned = planned_kbin_outputs(&dir, "cmp");
        assert_eq!(planned.len(), 3);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn lane_pattern_transpose_matches_expected_bit_order() {
        let patterns = super::lane_patterns_from_group_masks(&[0b1001, 0b0110, 0, 0]);
        assert_eq!(patterns[0], 0b0001);
        assert_eq!(patterns[1], 0b0010);
        assert_eq!(patterns[2], 0b0010);
        assert_eq!(patterns[3], 0b0001);
    }
}
