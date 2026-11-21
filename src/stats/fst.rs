//! Bitwise population differentiation statistics for PLINK SNP-major BED files.
//!
//! The implementation deliberately keeps the genotype matrix packed.  Each BED
//! row is transposed into three sample-bit planes and all population counts are
//! obtained with the SIMD-aware bitwise popcount kernels.

use crate::bitwise::and_popcount;
use crate::gfcore::SiteInfo;
use crate::stats_common::{
    emit_progress_callback, get_cached_pool, map_err_string_to_py, progress_step,
};
use memmap2::Mmap;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const BED_HEADER_LEN: usize = 3;
const FST_BLOCK_SITES: usize = 4096;
const FST_SITE_TASK_SITES: usize = 256;
const FST_OUTPUT_BUFFER_SIZE: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupStats {
    /// Number of diploid samples with a non-missing call.
    pub(crate) n: usize,
    /// Number of alternate alleles across those samples.
    pub(crate) alt_alleles: usize,
    /// Number of heterozygous diploid calls.
    pub(crate) het: usize,
}

impl GroupStats {
    #[cfg(test)]
    fn from_counts(n: usize, alt_alleles: usize, het: usize) -> Self {
        Self {
            n,
            alt_alleles,
            het,
        }
    }
}

#[derive(Clone, Debug)]
struct GroupMask {
    name: String,
    words: Vec<u64>,
    n_samples: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct BitPlanes {
    ref_bits: u64,
    het_bits: u64,
    alt_bits: u64,
}

#[derive(Clone, Debug, Default)]
struct BitPlaneWords {
    ref_bits: Vec<u64>,
    het_bits: Vec<u64>,
    alt_bits: Vec<u64>,
}

struct FstScratch {
    planes: BitPlaneWords,
    stats: Vec<GroupStats>,
}

impl FstScratch {
    fn new(n_samples: usize, n_groups: usize) -> Self {
        let n_words = words_for_samples(n_samples);
        Self {
            planes: BitPlaneWords {
                ref_bits: vec![0u64; n_words],
                het_bits: vec![0u64; n_words],
                alt_bits: vec![0u64; n_words],
            },
            stats: vec![GroupStats::default(); n_groups],
        }
    }
}

#[derive(Clone, Debug)]
struct FstOutput {
    sites: Vec<SiteInfo>,
    values: Vec<f64>,
    nmiss_left: Vec<usize>,
    nmiss_right: Vec<usize>,
    pair_names: Vec<(String, String)>,
}

struct FstInput {
    mmap: Mmap,
    n_samples: usize,
    n_sites: usize,
    bytes_per_snp: usize,
    masks: Vec<GroupMask>,
    pair_names: Vec<(String, String)>,
    method: FstMethod,
}

/// Windowed FST input keeps only a file handle and fixed-size decode buffers.
/// The site-level API still uses `FstInput`/mmap because it returns the full
/// per-site matrix to Python; window scans do not need that materialization.
struct FstWindowInput {
    bed: File,
    n_samples: usize,
    n_sites: usize,
    bytes_per_snp: usize,
    masks: Vec<GroupMask>,
}

const FST_WINDOW_BLOCK_SITES: usize = 4096;
const FST_WINDOW_HEADER: &str = "chrom\tstart\tend\tnsnp\tweightedFst\tmeanFst";
const FST_MATRIX_WINDOW_HEADER: &str = "pop1\tpop2\tchrom\tstart\tend\tnsnp\tweightedFst\tmeanFst";

#[derive(Clone, Debug)]
struct WindowSiteStats {
    pos: i64,
    stats: Vec<GroupStats>,
}

#[derive(Clone, Debug)]
struct FstWindowEmission {
    start: i64,
    stop: i64,
    summaries: Vec<Option<FstWindowSummary>>,
}

/// Bounded-memory rolling window state for the windowed FST path.
///
/// BED/BIM rows are consumed in coordinate order.  Only sites that can still
/// contribute to the next overlapping window remain in `pending`; empty
/// genomic gaps are skipped arithmetically rather than materialized as jobs.
struct FstWindowAccumulator {
    next_start: i64,
    pending: VecDeque<WindowSiteStats>,
}

impl FstWindowAccumulator {
    fn new(_chrom: &str) -> Self {
        Self {
            next_start: 1,
            pending: VecDeque::new(),
        }
    }

    fn push(
        &mut self,
        site: WindowSiteStats,
        window: i64,
        step: i64,
        method: FstMethod,
        pairs: &[(usize, usize)],
    ) -> Vec<FstWindowEmission> {
        let mut emitted = Vec::new();
        while self.next_start <= site.pos {
            let stop = self.next_start.saturating_add(window - 1);
            if stop >= site.pos {
                break;
            }
            if self.pending.is_empty() {
                self.next_start = first_window_start_covering(site.pos, window, step);
                continue;
            }
            if let Some(summary) = summarize_window_sites(&self.pending, stop, method, pairs) {
                emitted.push(FstWindowEmission {
                    start: self.next_start,
                    stop,
                    summaries: summary,
                });
            }
            self.advance(step);
        }
        if site.pos >= self.next_start && site.pos <= self.next_start.saturating_add(window - 1) {
            self.pending.push_back(site);
        }
        emitted
    }

    fn finish(
        &mut self,
        max_pos: i64,
        window: i64,
        step: i64,
        method: FstMethod,
        pairs: &[(usize, usize)],
    ) -> Vec<FstWindowEmission> {
        let mut emitted = Vec::new();
        while self.next_start <= max_pos {
            let stop = self.next_start.saturating_add(window - 1);
            if self.pending.is_empty() {
                break;
            }
            if let Some(summary) = summarize_window_sites(&self.pending, stop, method, pairs) {
                emitted.push(FstWindowEmission {
                    start: self.next_start,
                    stop,
                    summaries: summary,
                });
            }
            self.advance(step);
        }
        emitted
    }

    fn advance(&mut self, step: i64) {
        self.next_start = self.next_start.saturating_add(step);
        while self
            .pending
            .front()
            .is_some_and(|site| site.pos < self.next_start)
        {
            self.pending.pop_front();
        }
    }
}

fn first_window_start_covering(pos: i64, window: i64, step: i64) -> i64 {
    let lower = pos.saturating_sub(window - 1).max(1);
    if lower <= 1 {
        return 1;
    }
    let offset = lower - 1;
    let jumps = (offset - 1) / step + 1;
    1i64.saturating_add(jumps.saturating_mul(step))
}

struct FstBimReader {
    path: String,
    reader: BufReader<File>,
    line_no: usize,
}

impl FstBimReader {
    fn open(prefix: &str) -> Result<Self, String> {
        let path = format!("{prefix}.bim");
        let file = File::open(&path).map_err(|e| format!("{path}: {e}"))?;
        Ok(Self {
            path,
            reader: BufReader::new(file),
            line_no: 0,
        })
    }

    fn next_site(&mut self) -> Result<Option<SiteInfo>, String> {
        let mut line = String::new();
        loop {
            line.clear();
            if self
                .reader
                .read_line(&mut line)
                .map_err(|e| format!("{}: {e}", self.path))?
                == 0
            {
                return Ok(None);
            }
            self.line_no += 1;
            if !line.trim().is_empty() {
                break;
            }
        }
        let raw = line.trim_end();
        let columns = raw.split_whitespace().collect::<Vec<_>>();
        if columns.len() < 6 {
            return Err(format!(
                "malformed BIM line at {}:{}; expected 6 columns: {raw}",
                self.path, self.line_no
            ));
        }
        let pos = columns[3].parse::<i32>().map_err(|e| {
            format!(
                "invalid BIM position at {}:{}: {e}: {raw}",
                self.path, self.line_no
            )
        })?;
        Ok(Some(SiteInfo {
            chrom: columns[0].to_string(),
            pos,
            snp: columns[1].to_string(),
            ref_allele: columns[4].to_string(),
            alt_allele: columns[5].to_string(),
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FstMethod {
    Wc,
    Hudson,
}

impl FstMethod {
    fn parse(method: &str) -> Result<Self, String> {
        match method.trim().to_ascii_lowercase().as_str() {
            "wc" | "weir-cockerham" | "weir_cockerham" => Ok(Self::Wc),
            "hudson" => Ok(Self::Hudson),
            other => Err(format!(
                "unsupported FST method {other:?}; expected 'wc' or 'hudson'"
            )),
        }
    }
}

#[inline]
fn words_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(64)
}

#[inline]
fn set_bit(words: &mut [u64], sample_idx: usize) {
    words[sample_idx >> 6] |= 1u64 << (sample_idx & 63);
}

const fn build_byte_plane_lut(code: u8) -> [u8; 256] {
    let mut lut = [0u8; 256];
    let mut packed = 0usize;
    while packed < 256 {
        let mut lane = 0usize;
        let mut bits = 0u8;
        while lane < 4 {
            if ((packed as u8 >> (lane * 2)) & 0b11) == code {
                bits |= 1u8 << lane;
            }
            lane += 1;
        }
        lut[packed] = bits;
        packed += 1;
    }
    lut
}

// One BED byte stores four 2-bit calls.  These tables compact the selected
// genotype code into four one-bit sample lanes, avoiding scalar genotype
// decoding in the hot path.
const REF_BYTE_PLANE_LUT: [u8; 256] = build_byte_plane_lut(0b00);
const HET_BYTE_PLANE_LUT: [u8; 256] = build_byte_plane_lut(0b10);
const ALT_BYTE_PLANE_LUT: [u8; 256] = build_byte_plane_lut(0b11);

/// Transpose up to 64 BED genotype calls into sample-bit planes.
///
/// PLINK uses 00/10/11 for 0/1/2 and 01 for missing. Padding calls in the
/// last BED byte never enter a count. A BED byte is expanded with a small
/// compile-time lookup table rather than decoding each genotype to a numeric
/// dosage.
#[inline]
fn transpose_bed_word(row: &[u8], sample_start: usize, n_samples: usize) -> BitPlanes {
    let end = (sample_start + 64).min(n_samples);
    let valid = end - sample_start;
    let byte_count = valid.div_ceil(4);
    let byte_start = sample_start / 4;
    let mut out = BitPlanes::default();
    for byte_idx in 0..byte_count {
        let packed = row[byte_start + byte_idx] as usize;
        let shift = byte_idx * 4;
        out.ref_bits |= u64::from(REF_BYTE_PLANE_LUT[packed]) << shift;
        out.het_bits |= u64::from(HET_BYTE_PLANE_LUT[packed]) << shift;
        out.alt_bits |= u64::from(ALT_BYTE_PLANE_LUT[packed]) << shift;
    }
    if valid < 64 {
        let valid_mask = (1u64 << valid) - 1;
        out.ref_bits &= valid_mask;
        out.het_bits &= valid_mask;
        out.alt_bits &= valid_mask;
    }
    out
}

fn build_group_masks(
    group_by_sample: &[Option<String>],
    n_samples: usize,
) -> Result<Vec<GroupMask>, String> {
    if group_by_sample.len() != n_samples {
        return Err(format!(
            "group assignment length mismatch: got {}, expected {n_samples}",
            group_by_sample.len()
        ));
    }
    let mut group_names = Vec::<String>::new();
    let mut group_lookup = HashMap::<String, usize>::new();
    let mut sample_group_idx = vec![None; n_samples];
    for (sample_idx, group) in group_by_sample.iter().enumerate() {
        let Some(group) = group else { continue };
        let group = group.trim();
        if group.is_empty() {
            continue;
        }
        let group_idx = if let Some(&idx) = group_lookup.get(group) {
            idx
        } else {
            let idx = group_names.len();
            group_names.push(group.to_string());
            group_lookup.insert(group.to_string(), idx);
            idx
        };
        sample_group_idx[sample_idx] = Some(group_idx);
    }
    if group_names.len() < 2 {
        return Err("FST requires at least two non-empty groups".to_string());
    }

    let words = words_for_samples(n_samples);
    let mut masks = group_names
        .into_iter()
        .map(|name| GroupMask {
            name,
            words: vec![0u64; words],
            n_samples: 0,
        })
        .collect::<Vec<_>>();
    for (sample_idx, group_idx) in sample_group_idx.into_iter().enumerate() {
        if let Some(group_idx) = group_idx {
            set_bit(&mut masks[group_idx].words, sample_idx);
            masks[group_idx].n_samples += 1;
        }
    }
    Ok(masks)
}

fn read_fam_keys(fam_path: &str) -> Result<Vec<(String, String)>, String> {
    let file = File::open(fam_path).map_err(|e| format!("{fam_path}: {e}"))?;
    let mut fam_ids = Vec::new();
    let mut seen = HashSet::<(String, String)>::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{fam_path}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let mut cols = line.split_whitespace();
        let fid = cols
            .next()
            .ok_or_else(|| format!("malformed FAM line at {fam_path}:{}: {line}", line_no + 1))?;
        let iid = cols
            .next()
            .ok_or_else(|| format!("malformed FAM line at {fam_path}:{}: {line}", line_no + 1))?;
        let key = (fid.to_string(), iid.to_string());
        if !seen.insert(key.clone()) {
            return Err(format!(
                "duplicate FID/IID in FAM at {fam_path}:{}: {} {}",
                line_no + 1,
                key.0,
                key.1
            ));
        }
        fam_ids.push(key);
    }
    if fam_ids.is_empty() {
        return Err(format!("FAM contains no samples: {fam_path}"));
    }
    Ok(fam_ids)
}

fn read_within_groups(
    within_path: &str,
    fam_ids: &[(String, String)],
) -> Result<Vec<Option<String>>, String> {
    let file = File::open(within_path).map_err(|e| format!("{within_path}: {e}"))?;
    let reader = BufReader::new(file);

    let mut by_key = HashMap::<(String, String), String>::new();
    let mut seen_keys = HashSet::<(String, String)>::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("{within_path}:{}: {e}", line_no + 1))?;
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let cols = line.split_whitespace().collect::<Vec<_>>();
        if cols.len() < 3 {
            return Err(format!(
                "malformed --within line at {within_path}:{}; expected FID IID GROUP",
                line_no + 1
            ));
        }
        let key = (cols[0].to_string(), cols[1].to_string());
        if !seen_keys.insert(key.clone()) {
            return Err(format!(
                "duplicate sample assignment in {within_path}:{} for {} {}",
                line_no + 1,
                cols[0],
                cols[1]
            ));
        }
        let group = cols[2].to_string();
        by_key.insert(key, group);
    }

    let mut out = Vec::with_capacity(fam_ids.len());
    let mut matched = HashSet::<(String, String)>::new();
    for key in fam_ids {
        if let Some(group) = by_key.get(&key) {
            matched.insert(key.clone());
            out.push(Some(group.clone()));
        } else {
            out.push(None);
        }
    }
    if let Some((key, _)) = by_key.iter().find(|(key, _)| !matched.contains(key)) {
        return Err(format!(
            "--within contains sample {} {} not present in FAM; FID/IID must match exactly",
            key.0, key.1
        ));
    }
    Ok(out)
}

fn open_bed(prefix: &str) -> Result<(Mmap, usize, usize, usize), String> {
    let fam_path = format!("{prefix}.fam");
    let n_samples = read_fam_keys(&fam_path)?.len();
    let path = format!("{prefix}.bed");
    let file = File::open(&path).map_err(|e| format!("{path}: {e}"))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("{path}: {e}"))?;
    if mmap.len() < BED_HEADER_LEN || mmap[0] != 0x6c || mmap[1] != 0x1b || mmap[2] != 0x01 {
        return Err(format!(
            "{path}: expected SNP-major PLINK BED header 0x6c 0x1b 0x01"
        ));
    }
    let bytes_per_snp = (n_samples + 3) / 4;
    let payload = mmap.len() - BED_HEADER_LEN;
    if bytes_per_snp == 0 || payload == 0 || payload % bytes_per_snp != 0 {
        return Err(format!(
            "{path}: invalid payload length {payload} for {bytes_per_snp} bytes/SNP"
        ));
    }
    Ok((mmap, n_samples, payload / bytes_per_snp, bytes_per_snp))
}

#[inline]
fn group_stats_from_planes_bitwise(planes: &BitPlaneWords, mask: &GroupMask) -> GroupStats {
    let ref_count = and_popcount(mask.words.as_slice(), &planes.ref_bits) as usize;
    let het_count = and_popcount(mask.words.as_slice(), &planes.het_bits) as usize;
    let alt_count = and_popcount(mask.words.as_slice(), &planes.alt_bits) as usize;
    GroupStats {
        n: ref_count + het_count + alt_count,
        alt_alleles: het_count + 2 * alt_count,
        het: het_count,
    }
}

fn transpose_row_into(row: &[u8], n_samples: usize, out: &mut BitPlaneWords) {
    for (word_idx, sample_start) in (0..n_samples).step_by(64).enumerate() {
        let p = transpose_bed_word(row, sample_start, n_samples);
        out.ref_bits[word_idx] = p.ref_bits;
        out.het_bits[word_idx] = p.het_bits;
        out.alt_bits[word_idx] = p.alt_bits;
    }
}

fn stats_for_row_into(row: &[u8], n_samples: usize, masks: &[GroupMask], scratch: &mut FstScratch) {
    transpose_row_into(row, n_samples, &mut scratch.planes);
    for (stats, mask) in scratch.stats.iter_mut().zip(masks) {
        *stats = group_stats_from_planes_bitwise(&scratch.planes, mask);
    }
}

#[cfg(test)]
fn stats_for_row(row: &[u8], n_samples: usize, masks: &[GroupMask]) -> Vec<GroupStats> {
    let mut scratch = FstScratch::new(n_samples, masks.len());
    stats_for_row_into(row, n_samples, masks, &mut scratch);
    scratch.stats
}

#[inline]
fn allele_frequency(stats: &GroupStats) -> Option<f64> {
    (stats.n > 0).then(|| stats.alt_alleles as f64 / (2.0 * stats.n as f64))
}

/// Weir--Cockerham's per-site theta estimator.
pub(crate) fn wc_fst(groups: &[GroupStats]) -> Option<f64> {
    wc_components(groups).map(|(a, denominator)| a / denominator)
}

/// Return the Weir--Cockerham numerator and denominator used by the
/// ratio-of-sums window estimator.
fn wc_components(groups: &[GroupStats]) -> Option<(f64, f64)> {
    if groups.len() < 2 || groups.iter().any(|g| g.n < 2) {
        return None;
    }
    let r = groups.len() as f64;
    let total_n: usize = groups.iter().map(|g| g.n).sum();
    let n_bar = total_n as f64 / r;
    if n_bar <= 1.0 {
        return None;
    }
    let p_bar = groups
        .iter()
        .filter_map(|g| allele_frequency(g).map(|p| g.n as f64 * p))
        .sum::<f64>()
        / (r * n_bar);
    let s_squared = groups
        .iter()
        .filter_map(|g| allele_frequency(g).map(|p| g.n as f64 * (p - p_bar).powi(2)))
        .sum::<f64>()
        / ((r - 1.0) * n_bar);
    let h_bar = groups.iter().map(|g| g.het as f64).sum::<f64>() / (r * n_bar);
    let n_c = (r * n_bar - groups.iter().map(|g| (g.n as f64).powi(2)).sum::<f64>() / (r * n_bar))
        / (r - 1.0);
    if n_c <= 0.0 {
        return None;
    }
    let q = p_bar * (1.0 - p_bar) - ((r - 1.0) / r) * s_squared;
    let a = (n_bar / n_c) * (s_squared - (q - h_bar / 4.0) / (n_bar - 1.0));
    let b = (n_bar / (n_bar - 1.0)) * (q - ((2.0 * n_bar - 1.0) / (4.0 * n_bar)) * h_bar);
    let c = h_bar / 2.0;
    let denominator = a + b + c;
    if !denominator.is_finite() || denominator == 0.0 {
        None
    } else {
        Some((a, denominator))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FstWindowSummary {
    pub(crate) weighted: f64,
    pub(crate) mean: f64,
    pub(crate) n_variants: usize,
}

/// Aggregate per-site WC estimates as both a weighted ratio and an arithmetic
/// mean.  The weighted value is the standard ratio of summed WC components;
/// retaining the mean is useful for comparison with tools that report both
/// window estimators.
#[cfg(test)]
pub(crate) fn wc_window_summary(sites: &[Vec<GroupStats>]) -> Option<FstWindowSummary> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut mean_sum = 0.0;
    let mut n_variants = 0usize;
    for groups in sites {
        let Some((a, d)) = wc_components(groups) else {
            continue;
        };
        numerator += a;
        denominator += d;
        mean_sum += a / d;
        n_variants += 1;
    }
    if n_variants == 0 || denominator == 0.0 || !denominator.is_finite() {
        None
    } else {
        Some(FstWindowSummary {
            weighted: numerator / denominator,
            mean: mean_sum / n_variants as f64,
            n_variants,
        })
    }
}

/// Hudson's unbiased two-population per-site estimator.
pub(crate) fn hudson_fst(lhs: &GroupStats, rhs: &GroupStats) -> Option<f64> {
    hudson_components(lhs, rhs).map(|(numerator, denominator)| numerator / denominator)
}

fn hudson_components(lhs: &GroupStats, rhs: &GroupStats) -> Option<(f64, f64)> {
    if lhs.n < 2 || rhs.n < 2 {
        return None;
    }
    let p_lhs = allele_frequency(lhs)?;
    let p_rhs = allele_frequency(rhs)?;
    let between = p_lhs * (1.0 - p_rhs) + p_rhs * (1.0 - p_lhs);
    if !between.is_finite() || between <= 0.0 {
        return None;
    }
    let within_lhs = 2.0 * (lhs.alt_alleles as f64) * ((2 * lhs.n - lhs.alt_alleles) as f64)
        / ((2 * lhs.n) as f64 * (2 * lhs.n - 1) as f64);
    let within_rhs = 2.0 * (rhs.alt_alleles as f64) * ((2 * rhs.n - rhs.alt_alleles) as f64)
        / ((2 * rhs.n) as f64 * (2 * rhs.n - 1) as f64);
    let numerator = between - 0.5 * (within_lhs + within_rhs);
    Some((numerator, between))
}

fn normalize_plink_prefix(prefix: &str) -> String {
    let trimmed = prefix.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        trimmed[..trimmed.len() - 4].to_string()
    } else {
        trimmed.to_string()
    }
}

fn pair_names(masks: &[GroupMask], method: FstMethod) -> Vec<(String, String)> {
    match method {
        FstMethod::Wc => vec![("ALL".to_string(), "ALL".to_string())],
        FstMethod::Hudson => {
            let mut out = Vec::new();
            for lhs in 0..masks.len() {
                for rhs in (lhs + 1)..masks.len() {
                    out.push((masks[lhs].name.clone(), masks[rhs].name.clone()));
                }
            }
            out
        }
    }
}

fn open_fst_input(prefix: &str, within_path: &str, method_name: &str) -> Result<FstInput, String> {
    let method = FstMethod::parse(method_name)?;
    let bed_prefix = normalize_plink_prefix(prefix);
    let fam_ids = read_fam_keys(&format!("{bed_prefix}.fam"))?;
    let n_samples = fam_ids.len();
    let groups_by_sample = read_within_groups(within_path, &fam_ids)?;
    let masks = build_group_masks(&groups_by_sample, n_samples)?;
    for mask in &masks {
        if mask.n_samples < 2 {
            return Err(format!(
                "group {:?} has only {} assigned samples; at least 2 are required",
                mask.name, mask.n_samples
            ));
        }
    }

    let (mmap, bed_n_samples, n_sites, bytes_per_snp) = open_bed(&bed_prefix)?;
    if bed_n_samples != n_samples {
        return Err(format!(
            "BED/FAM sample count mismatch: BED/FAM={bed_n_samples}, FAM metadata={n_samples}"
        ));
    }
    let pair_names = pair_names(&masks, method);
    Ok(FstInput {
        mmap,
        n_samples,
        n_sites,
        bytes_per_snp,
        masks,
        pair_names,
        method,
    })
}

fn compute_fst_block(
    input: &FstInput,
    start: usize,
    end: usize,
    threads: usize,
) -> Result<FstOutput, String> {
    if start > end || end > input.n_sites {
        return Err(format!(
            "invalid FST block range [{start}, {end}) for {} sites",
            input.n_sites
        ));
    }
    let n_sites = end - start;
    let pair_names = input.pair_names.clone();
    let pair_count = pair_names.len();
    let mut values = vec![f64::NAN; n_sites * pair_count];
    let mut nmiss_left = vec![0usize; n_sites * pair_count];
    let mut nmiss_right = vec![0usize; n_sites * pair_count];
    let payload = &input.mmap[BED_HEADER_LEN..];

    let mut run = || {
        values
            .par_chunks_mut(FST_SITE_TASK_SITES * pair_count)
            .zip(nmiss_left.par_chunks_mut(FST_SITE_TASK_SITES * pair_count))
            .zip(nmiss_right.par_chunks_mut(FST_SITE_TASK_SITES * pair_count))
            .enumerate()
            .for_each_init(
                || FstScratch::new(input.n_samples, input.masks.len()),
                |scratch, (chunk_idx, ((values_chunk, nmiss_left_chunk), nmiss_right_chunk))| {
                    let chunk_start = chunk_idx * FST_SITE_TASK_SITES;
                    let chunk_sites = values_chunk.len() / pair_count;
                    for local_site_idx in 0..chunk_sites {
                        let row_start = local_site_idx * pair_count;
                        let row_end = row_start + pair_count;
                        let value_row = &mut values_chunk[row_start..row_end];
                        let left_row = &mut nmiss_left_chunk[row_start..row_end];
                        let right_row = &mut nmiss_right_chunk[row_start..row_end];
                        let source_site_idx = start + chunk_start + local_site_idx;
                        let row = &payload[source_site_idx * input.bytes_per_snp
                            ..(source_site_idx + 1) * input.bytes_per_snp];
                        stats_for_row_into(row, input.n_samples, &input.masks, scratch);
                        let stats = &scratch.stats;
                        match input.method {
                            FstMethod::Wc => {
                                left_row[0] = stats.iter().map(|group| group.n).sum();
                                value_row[0] = wc_fst(&stats).unwrap_or(f64::NAN);
                            }
                            FstMethod::Hudson => {
                                let mut pair_idx = 0usize;
                                for lhs in 0..stats.len() {
                                    for rhs in (lhs + 1)..stats.len() {
                                        left_row[pair_idx] = stats[lhs].n;
                                        right_row[pair_idx] = stats[rhs].n;
                                        value_row[pair_idx] = hudson_fst(&stats[lhs], &stats[rhs])
                                            .unwrap_or(f64::NAN);
                                        pair_idx += 1;
                                    }
                                }
                            }
                        }
                    }
                },
            );
    };
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    if let Some(pool) = pool {
        pool.install(&mut run);
    } else {
        run();
    }

    // Keep the row-major [site, population-pair] layout.  This is both the
    // natural output order and the shape exposed to Python.  The mmap remains
    // borrowed by `payload` until this point, so all worker reads are complete.
    Ok(FstOutput {
        sites: Vec::new(),
        values,
        nmiss_left,
        nmiss_right,
        pair_names,
    })
}

fn compute_fst_core(
    prefix: &str,
    within_path: &str,
    method_name: &str,
    threads: usize,
) -> Result<FstOutput, String> {
    let input = open_fst_input(prefix, within_path, method_name)?;
    let mut output = compute_fst_block(&input, 0, input.n_sites, threads)?;
    let mut bim_reader = FstBimReader::open(&normalize_plink_prefix(prefix))?;
    output.sites = read_fst_sites(&mut bim_reader, input.n_sites)?;
    if bim_reader.next_site()?.is_some() {
        return Err("BIM contains more rows than BED while loading FST".to_string());
    }
    Ok(output)
}

fn write_fst_header<W: Write>(writer: &mut W, method: FstMethod, path: &str) -> Result<(), String> {
    match method {
        FstMethod::Wc => {
            writeln!(writer, "CHR\tSNP\tPOS\tNMISS\tFST").map_err(|e| format!("{path}: {e}"))?;
        }
        FstMethod::Hudson => {
            writeln!(writer, "POP1\tPOP2\tCHR\tSNP\tPOS\tNMISS1\tNMISS2\tFST")
                .map_err(|e| format!("{path}: {e}"))?;
        }
    }
    Ok(())
}

fn write_fst_rows<W: Write>(
    writer: &mut W,
    output: &FstOutput,
    method: FstMethod,
    path: &str,
) -> Result<(), String> {
    match method {
        FstMethod::Wc => {
            for (site_idx, site) in output.sites.iter().enumerate() {
                writeln!(
                    writer,
                    "{}\t{}\t{}\t{}\t{}",
                    site.chrom,
                    site.snp,
                    site.pos,
                    output.nmiss_left[site_idx],
                    output.values[site_idx]
                )
                .map_err(|e| format!("{path}: {e}"))?;
            }
        }
        FstMethod::Hudson => {
            let pair_count = output.pair_names.len();
            for (site_idx, site) in output.sites.iter().enumerate() {
                for (pair_idx, (pop1, pop2)) in output.pair_names.iter().enumerate() {
                    let offset = site_idx * pair_count + pair_idx;
                    writeln!(
                        writer,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        pop1,
                        pop2,
                        site.chrom,
                        site.snp,
                        site.pos,
                        output.nmiss_left[offset],
                        output.nmiss_right[offset],
                        output.values[offset]
                    )
                    .map_err(|e| format!("{path}: {e}"))?;
                }
            }
        }
    }
    Ok(())
}

fn read_fst_sites(reader: &mut FstBimReader, count: usize) -> Result<Vec<SiteInfo>, String> {
    let mut sites = Vec::with_capacity(count);
    for _ in 0..count {
        sites.push(
            reader
                .next_site()?
                .ok_or_else(|| "BIM ended while loading FST site metadata".to_string())?,
        );
    }
    Ok(sites)
}

fn temporary_fst_output_path(output_path: &str) -> PathBuf {
    let output = Path::new(output_path);
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("fst.tsv");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    output.with_file_name(format!(
        ".{file_name}.janusx-fst-{}-{stamp}.tmp",
        std::process::id()
    ))
}

fn write_fst_tsv_streaming(
    prefix: &str,
    within_path: &str,
    output_path: &str,
    method_name: &str,
    threads: usize,
) -> Result<(usize, usize), String> {
    let input = open_fst_input(prefix, within_path, method_name)?;
    let mut bim_reader = FstBimReader::open(&normalize_plink_prefix(prefix))?;
    let method = input.method;
    let temporary_path = temporary_fst_output_path(output_path);
    let mut temporary_created = false;
    let result = (|| {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
            .map_err(|e| {
                format!(
                    "{output_path}: unable to create temporary output {}: {e}",
                    temporary_path.display()
                )
            })?;
        temporary_created = true;
        let mut writer = BufWriter::with_capacity(FST_OUTPUT_BUFFER_SIZE, file);
        write_fst_header(&mut writer, method, output_path)?;

        let mut start = 0usize;
        while start < input.n_sites {
            let end = (start + FST_BLOCK_SITES).min(input.n_sites);
            let mut block = compute_fst_block(&input, start, end, threads)?;
            block.sites = read_fst_sites(&mut bim_reader, end - start)?;
            write_fst_rows(&mut writer, &block, method, output_path)?;
            start = end;
        }
        if bim_reader.next_site()?.is_some() {
            return Err(format!(
                "BED/BIM variant count mismatch while streaming: BED={}",
                input.n_sites
            ));
        }
        writer.flush().map_err(|e| format!("{output_path}: {e}"))?;
        drop(writer);
        std::fs::rename(&temporary_path, output_path).map_err(|e| {
            format!(
                "{output_path}: unable to atomically replace output from {}: {e}",
                temporary_path.display()
            )
        })?;
        Ok((input.n_sites, input.pair_names.len()))
    })();

    if result.is_err() && temporary_created {
        if let Err(cleanup_err) = std::fs::remove_file(&temporary_path) {
            if cleanup_err.kind() != std::io::ErrorKind::NotFound {
                eprintln!(
                    "Warning: failed to remove temporary FST output {} after an earlier error: {cleanup_err}",
                    temporary_path.display()
                );
            }
        }
    }
    result
}

fn read_population_indices(path: &str, fam_ids: &[(String, String)]) -> Result<Vec<usize>, String> {
    let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut by_key = HashMap::<(String, String), usize>::new();
    let mut by_iid = HashMap::<String, Vec<usize>>::new();
    for (idx, (fid, iid)) in fam_ids.iter().enumerate() {
        by_key.insert((fid.clone(), iid.clone()), idx);
        by_iid.entry(iid.clone()).or_default().push(idx);
    }
    let mut selected = Vec::new();
    let mut seen = HashSet::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{path}: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let cols = trimmed.split_whitespace().collect::<Vec<_>>();
        let sample_idx = if cols.len() >= 2 {
            by_key
                .get(&(cols[0].to_string(), cols[1].to_string()))
                .copied()
                .ok_or_else(|| {
                    format!(
                        "{path}: FID/IID {} {} at line {} is absent from FAM",
                        cols[0],
                        cols[1],
                        line_no + 1
                    )
                })?
        } else {
            let matches = by_iid.get(cols[0]).ok_or_else(|| {
                format!(
                    "{path}: IID {} at line {} is absent from FAM",
                    cols[0],
                    line_no + 1
                )
            })?;
            if matches.len() != 1 {
                return Err(format!(
                    "{path}: IID {} at line {} is ambiguous; provide FID IID",
                    cols[0],
                    line_no + 1
                ));
            }
            matches[0]
        };
        if !seen.insert(sample_idx) {
            return Err(format!("{path}: duplicate sample at line {}", line_no + 1));
        }
        selected.push(sample_idx);
    }
    if selected.is_empty() {
        return Err(format!("{path}: no samples found"));
    }
    Ok(selected)
}

fn build_named_population_masks(
    groups: &[(String, Vec<usize>)],
    n_samples: usize,
) -> Result<Vec<GroupMask>, String> {
    if groups.len() < 2 {
        return Err("FST requires at least two populations".to_string());
    }
    let words = words_for_samples(n_samples);
    let mut used = HashSet::new();
    let mut masks = Vec::with_capacity(groups.len());
    for (name, sample_indices) in groups {
        if name.trim().is_empty() {
            return Err("population names must not be empty".to_string());
        }
        if sample_indices.len() < 2 {
            return Err(format!(
                "population {name:?} has only {} assigned samples; at least 2 are required",
                sample_indices.len()
            ));
        }
        let mut mask = GroupMask {
            name: name.clone(),
            words: vec![0u64; words],
            n_samples: 0,
        };
        for &sample_idx in sample_indices {
            if sample_idx >= n_samples {
                return Err(format!(
                    "population {name:?} contains sample index {sample_idx} >= {n_samples}"
                ));
            }
            if !used.insert(sample_idx) {
                return Err(format!(
                    "sample index {sample_idx} occurs in more than one population"
                ));
            }
            set_bit(&mut mask.words, sample_idx);
            mask.n_samples += 1;
        }
        masks.push(mask);
    }
    Ok(masks)
}

fn read_window_fst_input(
    prefix: &str,
    pop1_path: &str,
    pop2_path: &str,
    within_path: &str,
    matrix: bool,
) -> Result<FstWindowInput, String> {
    let bed_prefix = normalize_plink_prefix(prefix);
    let fam_ids = read_fam_keys(&format!("{bed_prefix}.fam"))?;
    let n_samples = fam_ids.len();
    let groups = if matrix {
        if within_path.trim().is_empty() {
            return Err("-matrix requires -within groups.tsv".to_string());
        }
        let assignments = read_within_groups(within_path, &fam_ids)?;
        let mut names = Vec::new();
        let mut indices_by_name = HashMap::<String, Vec<usize>>::new();
        for (idx, assignment) in assignments.into_iter().enumerate() {
            if let Some(name) = assignment {
                if !names.contains(&name) {
                    names.push(name.clone());
                }
                indices_by_name.entry(name).or_default().push(idx);
            }
        }
        names
            .into_iter()
            .map(|name| {
                let indices = indices_by_name.remove(&name).unwrap_or_default();
                (name, indices)
            })
            .collect::<Vec<_>>()
    } else {
        if pop1_path.trim().is_empty() || pop2_path.trim().is_empty() {
            return Err("FST requires -p1 and -p2 unless -matrix is used".to_string());
        }
        vec![
            (
                "pop1".to_string(),
                read_population_indices(pop1_path, &fam_ids)?,
            ),
            (
                "pop2".to_string(),
                read_population_indices(pop2_path, &fam_ids)?,
            ),
        ]
    };
    let masks = build_named_population_masks(&groups, n_samples)?;
    let bed_path = format!("{bed_prefix}.bed");
    let bed = File::open(&bed_path).map_err(|e| format!("{bed_path}: {e}"))?;
    let metadata = bed.metadata().map_err(|e| format!("{bed_path}: {e}"))?;
    if metadata.len() < BED_HEADER_LEN as u64 {
        return Err(format!("{bed_path}: BED file is shorter than its header"));
    }
    let mut header = [0u8; BED_HEADER_LEN];
    let mut header_reader = bed.try_clone().map_err(|e| format!("{bed_path}: {e}"))?;
    header_reader
        .read_exact(&mut header)
        .map_err(|e| format!("{bed_path}: {e}"))?;
    if header != [0x6c, 0x1b, 0x01] {
        return Err(format!(
            "{bed_path}: expected SNP-major PLINK BED header 0x6c 0x1b 0x01"
        ));
    }
    let bytes_per_snp = n_samples.div_ceil(4);
    let payload = metadata.len() as usize - BED_HEADER_LEN;
    if bytes_per_snp == 0 || payload == 0 || payload % bytes_per_snp != 0 {
        return Err(format!(
            "{bed_path}: invalid payload length {payload} for {bytes_per_snp} bytes/SNP"
        ));
    }
    let n_sites = payload / bytes_per_snp;
    Ok(FstWindowInput {
        bed,
        n_samples,
        n_sites,
        bytes_per_snp,
        masks,
    })
}

#[cfg(test)]
fn window_summary_for_pair(
    site_stats: &[Vec<GroupStats>],
    indices: &[usize],
    method: FstMethod,
    lhs: usize,
    rhs: usize,
) -> Option<FstWindowSummary> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut mean_sum = 0.0;
    let mut n_variants = 0usize;
    for &site_idx in indices {
        let stats = &site_stats[site_idx];
        let components = match method {
            FstMethod::Wc => wc_components(&[stats[lhs], stats[rhs]]),
            FstMethod::Hudson => hudson_components(&stats[lhs], &stats[rhs]),
        };
        let Some(components) = components else {
            continue;
        };
        numerator += components.0;
        denominator += components.1;
        mean_sum += components.0 / components.1;
        n_variants += 1;
    }
    if n_variants == 0 || denominator == 0.0 || !denominator.is_finite() {
        None
    } else {
        Some(FstWindowSummary {
            weighted: numerator / denominator,
            mean: mean_sum / n_variants as f64,
            n_variants,
        })
    }
}

#[cfg(test)]
fn window_summaries_for_pairs(
    site_stats: &[Vec<GroupStats>],
    indices: &[usize],
    method: FstMethod,
    pairs: &[(usize, usize)],
) -> Vec<Option<FstWindowSummary>> {
    let mut accumulators = pairs
        .iter()
        .map(|_| (0.0, 0.0, 0.0, 0usize))
        .collect::<Vec<_>>();
    for &site_idx in indices {
        let stats = &site_stats[site_idx];
        for ((lhs, rhs), accumulator) in pairs.iter().zip(accumulators.iter_mut()) {
            let components = match method {
                FstMethod::Wc => wc_components(&[stats[*lhs], stats[*rhs]]),
                FstMethod::Hudson => hudson_components(&stats[*lhs], &stats[*rhs]),
            };
            let Some((numerator, denominator)) = components else {
                continue;
            };
            accumulator.0 += numerator;
            accumulator.1 += denominator;
            accumulator.2 += numerator / denominator;
            accumulator.3 += 1;
        }
    }
    accumulators
        .into_iter()
        .map(|(numerator, denominator, mean_sum, n_variants)| {
            if n_variants == 0 || denominator == 0.0 || !denominator.is_finite() {
                None
            } else {
                Some(FstWindowSummary {
                    weighted: numerator / denominator,
                    mean: mean_sum / n_variants as f64,
                    n_variants,
                })
            }
        })
        .collect()
}

fn summarize_window_sites(
    sites: &VecDeque<WindowSiteStats>,
    stop: i64,
    method: FstMethod,
    pairs: &[(usize, usize)],
) -> Option<Vec<Option<FstWindowSummary>>> {
    let mut accumulators = pairs
        .iter()
        .map(|_| (0.0, 0.0, 0.0, 0usize))
        .collect::<Vec<_>>();
    for site in sites.iter().take_while(|site| site.pos <= stop) {
        for ((lhs, rhs), accumulator) in pairs.iter().zip(accumulators.iter_mut()) {
            let components = match method {
                FstMethod::Wc => wc_components(&[site.stats[*lhs], site.stats[*rhs]]),
                FstMethod::Hudson => hudson_components(&site.stats[*lhs], &site.stats[*rhs]),
            };
            let Some((numerator, denominator)) = components else {
                continue;
            };
            accumulator.0 += numerator;
            accumulator.1 += denominator;
            accumulator.2 += numerator / denominator;
            accumulator.3 += 1;
        }
    }
    let summaries = accumulators
        .into_iter()
        .map(|(numerator, denominator, mean_sum, n_variants)| {
            if n_variants == 0 || denominator == 0.0 || !denominator.is_finite() {
                None
            } else {
                Some(FstWindowSummary {
                    weighted: numerator / denominator,
                    mean: mean_sum / n_variants as f64,
                    n_variants,
                })
            }
        })
        .collect::<Vec<_>>();
    summaries.iter().any(Option::is_some).then_some(summaries)
}

fn write_fst_window_emission<W: Write>(
    writer: &mut W,
    chrom: &str,
    emission: &FstWindowEmission,
    pairs: &[(usize, usize)],
    masks: &[GroupMask],
    matrix: bool,
    path: &str,
) -> Result<usize, String> {
    let mut written = 0usize;
    for (pair_idx, summary) in emission.summaries.iter().enumerate() {
        let Some(summary) = summary else { continue };
        let (lhs, rhs) = pairs[pair_idx];
        let weighted = format_fst_window_value(summary.weighted);
        let mean = format_fst_window_value(summary.mean);
        if matrix {
            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                masks[lhs].name,
                masks[rhs].name,
                chrom,
                emission.start,
                emission.stop,
                summary.n_variants,
                weighted,
                mean
            )
        } else {
            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{}\t{}",
                chrom, emission.start, emission.stop, summary.n_variants, weighted, mean
            )
        }
        .map_err(|e| format!("{path}: {e}"))?;
        written += 1;
    }
    Ok(written)
}

fn format_fst_window_value(value: f64) -> String {
    if value.is_finite() && value.abs() < 0.0001 {
        format!("{value:.4e}")
    } else {
        format!("{value:.4}")
    }
}

fn run_windowed_fst(
    prefix: &str,
    pop1_path: &str,
    pop2_path: &str,
    within_path: &str,
    output_path: &str,
    window: i64,
    step: i64,
    method_name: &str,
    matrix: bool,
    chrom_filter: Option<&str>,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
) -> Result<(usize, usize), String> {
    if window <= 0 || step <= 0 {
        return Err("window and step must be positive".to_string());
    }
    let method = FstMethod::parse(method_name)?;
    let input = read_window_fst_input(prefix, pop1_path, pop2_path, within_path, matrix)?;
    let pair_indices = if matrix {
        let mut pairs = Vec::new();
        for lhs in 0..input.masks.len() {
            for rhs in (lhs + 1)..input.masks.len() {
                pairs.push((lhs, rhs));
            }
        }
        pairs.sort_by(|lhs, rhs| {
            input.masks[lhs.0]
                .name
                .cmp(&input.masks[rhs.0].name)
                .then(input.masks[lhs.1].name.cmp(&input.masks[rhs.1].name))
        });
        pairs
    } else {
        vec![(0, 1)]
    };
    let output = Path::new(output_path);
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let file = File::create(output_path).map_err(|e| format!("{output_path}: {e}"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "{}",
        if matrix {
            FST_MATRIX_WINDOW_HEADER
        } else {
            FST_WINDOW_HEADER
        }
    )
    .map_err(|e| format!("{output_path}: {e}"))?;

    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    let progress_step = progress_step(input.n_sites, progress_every);
    let mut next_progress = progress_step;
    if progress_callback.is_some() {
        emit_progress_callback(progress_callback, 0, 0, input.n_sites)?;
    }
    let mut bed = input.bed;
    bed.seek(SeekFrom::Start(BED_HEADER_LEN as u64))
        .map_err(|e| format!("{prefix}.bed: {e}"))?;
    let mut bim_reader = FstBimReader::open(&normalize_plink_prefix(prefix))?;
    let mut site_idx = 0usize;
    let mut total_rows = 0usize;
    let mut current_chrom = None::<String>;
    let mut completed_chroms = HashSet::<String>::new();
    let mut accumulator = None::<FstWindowAccumulator>;
    let mut max_pos = 0i64;
    let mut last_pos = None::<i64>;

    while site_idx < input.n_sites {
        let block_start = site_idx;
        let block_len = (input.n_sites - block_start).min(FST_WINDOW_BLOCK_SITES);
        let mut block_payload = vec![0u8; block_len * input.bytes_per_snp];
        bed.read_exact(&mut block_payload)
            .map_err(|e| format!("{prefix}.bed: {e}"))?;
        let mut block_sites = Vec::with_capacity(block_len);
        for _ in 0..block_len {
            block_sites.push(
                bim_reader
                    .next_site()?
                    .ok_or_else(|| "BIM ended while streaming windowed FST".to_string())?,
            );
        }
        let mut block_stats = (0..block_len)
            .map(|_| vec![GroupStats::default(); input.masks.len()])
            .collect::<Vec<_>>();
        let mut compute_stats = || {
            block_stats.par_iter_mut().enumerate().for_each_init(
                || FstScratch::new(input.n_samples, input.masks.len()),
                |scratch, (offset, stats)| {
                    let row = &block_payload
                        [offset * input.bytes_per_snp..(offset + 1) * input.bytes_per_snp];
                    stats_for_row_into(row, input.n_samples, &input.masks, scratch);
                    stats.copy_from_slice(&scratch.stats);
                },
            );
        };
        if let Some(pool) = pool.as_ref() {
            pool.install(&mut compute_stats);
        } else {
            compute_stats();
        }

        for (site, stats) in block_sites.into_iter().zip(block_stats) {
            site_idx += 1;
            if chrom_filter.is_some_and(|chrom| chrom != site.chrom) {
                continue;
            }
            if current_chrom.as_deref() != Some(site.chrom.as_str()) {
                if let (Some(chrom), Some(mut previous)) =
                    (current_chrom.take(), accumulator.take())
                {
                    for emission in previous.finish(max_pos, window, step, method, &pair_indices) {
                        total_rows += write_fst_window_emission(
                            &mut writer,
                            &chrom,
                            &emission,
                            &pair_indices,
                            &input.masks,
                            matrix,
                            output_path,
                        )?;
                    }
                    completed_chroms.insert(chrom);
                }
                if completed_chroms.contains(&site.chrom) {
                    return Err(format!(
                        "BIM chromosome {} reappears after a later chromosome; windowed FST requires chromosome-sorted BIM",
                        site.chrom
                    ));
                }
                current_chrom = Some(site.chrom.clone());
                accumulator = Some(FstWindowAccumulator::new(&site.chrom));
                max_pos = 0;
                last_pos = None;
            }
            if last_pos.is_some_and(|previous| i64::from(site.pos) < previous) {
                return Err(format!(
                    "BIM positions are not sorted within chromosome {}",
                    site.chrom
                ));
            }
            last_pos = Some(i64::from(site.pos));
            max_pos = max_pos.max(i64::from(site.pos));
            let emissions = accumulator.as_mut().unwrap().push(
                WindowSiteStats {
                    pos: i64::from(site.pos),
                    stats,
                },
                window,
                step,
                method,
                &pair_indices,
            );
            for emission in emissions {
                total_rows += write_fst_window_emission(
                    &mut writer,
                    current_chrom.as_deref().unwrap(),
                    &emission,
                    &pair_indices,
                    &input.masks,
                    matrix,
                    output_path,
                )?;
            }
        }
        if progress_callback.is_some() && (site_idx >= next_progress || site_idx == input.n_sites) {
            emit_progress_callback(progress_callback, 0, site_idx, input.n_sites)?;
            while next_progress <= site_idx {
                next_progress = next_progress.saturating_add(progress_step);
            }
        }
    }
    if bim_reader.next_site()?.is_some() {
        return Err(format!(
            "BED/BIM variant count mismatch while streaming windowed FST: BED={}",
            input.n_sites
        ));
    }
    if let (Some(chrom), Some(mut previous)) = (current_chrom, accumulator) {
        for emission in previous.finish(max_pos, window, step, method, &pair_indices) {
            total_rows += write_fst_window_emission(
                &mut writer,
                &chrom,
                &emission,
                &pair_indices,
                &input.masks,
                matrix,
                output_path,
            )?;
        }
    }
    writer.flush().map_err(|e| format!("{output_path}: {e}"))?;
    Ok((total_rows, pair_indices.len()))
}

#[pyfunction]
#[pyo3(signature = (prefix, pop1, pop2, within, output, window=50000, step=50000, method="wc", matrix=false, chrom=None, threads=0, progress_callback=None, progress_every=0))]
pub fn fst_bed_window_to_tsv(
    py: Python<'_>,
    prefix: String,
    pop1: String,
    pop2: String,
    within: String,
    output: String,
    window: i64,
    step: i64,
    method: &str,
    matrix: bool,
    chrom: Option<String>,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(usize, usize)> {
    let method_owned = method.to_string();
    py.detach(move || {
        run_windowed_fst(
            &prefix,
            &pop1,
            &pop2,
            &within,
            &output,
            window,
            step,
            &method_owned,
            matrix,
            chrom.as_deref(),
            threads,
            progress_callback.as_ref(),
            progress_every,
        )
    })
    .map_err(map_err_string_to_py)
}

#[cfg(test)]
mod tests {
    use super::{
        build_group_masks, hudson_fst, read_within_groups, stats_for_row, wc_fst,
        wc_window_summary, window_summaries_for_pairs, write_fst_tsv_streaming,
        write_fst_window_emission, FstMethod, FstWindowAccumulator, FstWindowEmission,
        FstWindowSummary, GroupMask, GroupStats, WindowSiteStats,
    };

    fn fst_fixture_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "janusx_fst_{label}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn write_fst_fixture(prefix: &std::path::Path, n_bed_sites: usize, n_bim_sites: usize) {
        std::fs::write(
            prefix.with_extension("fam"),
            "F1 S1 0 0 0 -9\nF1 S2 0 0 0 -9\nF1 S3 0 0 0 -9\nF1 S4 0 0 0 -9\n",
        )
        .unwrap();
        std::fs::write(
            prefix.with_extension("within"),
            "F1 S1 A\nF1 S2 A\nF1 S3 B\nF1 S4 B\n",
        )
        .unwrap();
        let mut bed = vec![0x6c, 0x1b, 0x01];
        bed.extend(std::iter::repeat(0u8).take(n_bed_sites));
        std::fs::write(prefix.with_extension("bed"), bed).unwrap();
        let bim = (1..=n_bim_sites)
            .map(|site| format!("1 rs{site} 0 {site} A G\n"))
            .collect::<String>();
        std::fs::write(prefix.with_extension("bim"), bim).unwrap();
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual={actual:.15e}, expected={expected:.15e}"
        );
    }

    #[test]
    fn wc_matches_two_population_reference_case() {
        let groups = [
            GroupStats::from_counts(4, 0, 0),
            GroupStats::from_counts(4, 8, 0),
        ];
        close(wc_fst(&groups).unwrap(), 1.0);
    }

    #[test]
    fn wc_supports_unequal_multi_population_sample_sizes() {
        let groups = [
            GroupStats::from_counts(4, 0, 0),
            GroupStats::from_counts(4, 4, 0),
            GroupStats::from_counts(4, 8, 0),
        ];
        let value = wc_fst(&groups).unwrap();
        assert!(value.is_finite());
        close(value, 2.0 / 3.0);
    }

    #[test]
    fn wc_matches_plink_with_heterozygotes() {
        let groups = [
            GroupStats::from_counts(2, 4, 0),
            GroupStats::from_counts(2, 1, 1),
            GroupStats::from_counts(2, 2, 0),
        ];
        close(wc_fst(&groups).unwrap(), 1.0 / 7.0);
    }

    #[test]
    fn hudson_uses_unbiased_within_population_diversity() {
        let groups = [
            GroupStats::from_counts(2, 0, 0),
            GroupStats::from_counts(2, 4, 0),
        ];
        close(hudson_fst(&groups[0], &groups[1]).unwrap(), 1.0);
    }

    #[test]
    fn hudson_returns_none_when_between_diversity_is_zero() {
        let group = GroupStats::from_counts(4, 0, 0);
        assert!(hudson_fst(&group, &group).is_none());
    }

    #[test]
    fn bitwise_bed_decode_ignores_missing_and_padding() {
        // Calls are: 0/0, 0/1, 1/1, missing, 0/0.  The final four BED
        // lanes are padding and must not contribute to any count.
        let row = [0b01_11_10_00u8, 0b00u8];
        let masks = build_group_masks(
            &[
                Some("A".to_string()),
                Some("A".to_string()),
                Some("B".to_string()),
                Some("B".to_string()),
                Some("B".to_string()),
            ],
            5,
        )
        .unwrap();
        let stats = stats_for_row(&row, 5, &masks);
        assert_eq!(
            stats,
            vec![
                GroupStats::from_counts(2, 1, 1),
                GroupStats::from_counts(2, 2, 0),
            ]
        );
    }

    #[test]
    fn bitwise_bed_decode_handles_multiple_sample_words() {
        let n_samples = 65;
        let mut row = vec![0u8; (n_samples + 3) / 4];
        // Sample 0: hom-ref; sample 63: het; sample 64: hom-alt.
        row[0] = 0b00;
        row[15] |= 0b10 << 6;
        row[16] |= 0b11;
        let mut assignments = vec![Some("B".to_string()); n_samples];
        assignments[0] = Some("A".to_string());
        assignments[64] = Some("C".to_string());
        let masks = build_group_masks(&assignments, n_samples).unwrap();
        let stats = stats_for_row(&row, n_samples, &masks);
        assert_eq!(stats[0], GroupStats::from_counts(1, 0, 0));
        assert_eq!(stats[1], GroupStats::from_counts(63, 1, 1));
        assert_eq!(stats[2], GroupStats::from_counts(1, 2, 0));
    }

    #[test]
    fn within_requires_exact_fid_iid_matches() {
        let path = std::env::temp_dir().join(format!(
            "janusx_fst_within_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, "F2 S1 A\nF1 S2 B\n").unwrap();
        let fam = [
            ("F1".to_string(), "S1".to_string()),
            ("F1".to_string(), "S2".to_string()),
        ];
        let err = read_within_groups(path.to_str().unwrap(), &fam).unwrap_err();
        assert!(err.contains("F2 S1"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn wc_window_summary_has_weighted_and_mean_estimators() {
        let sites = vec![
            vec![
                GroupStats::from_counts(4, 0, 0),
                GroupStats::from_counts(4, 8, 0),
            ],
            vec![
                GroupStats::from_counts(4, 2, 0),
                GroupStats::from_counts(4, 6, 0),
            ],
        ];
        let summary = wc_window_summary(&sites).unwrap();
        assert!((summary.weighted - (9.0 / 13.0)).abs() < 1e-12);
        assert!((summary.mean - 0.6).abs() < 1e-12);
        assert_eq!(summary.n_variants, 2);
    }

    #[test]
    fn matrix_window_batch_matches_individual_pair_summaries() {
        let site_stats = vec![
            vec![
                GroupStats::from_counts(4, 0, 0),
                GroupStats::from_counts(4, 8, 0),
                GroupStats::from_counts(4, 4, 2),
            ],
            vec![
                GroupStats::from_counts(4, 2, 0),
                GroupStats::from_counts(4, 6, 0),
                GroupStats::from_counts(4, 5, 2),
            ],
        ];
        let indices = [0usize, 1];
        let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
        let batched = window_summaries_for_pairs(&site_stats, &indices, FstMethod::Wc, &pairs);
        let individual = pairs
            .iter()
            .map(|&(lhs, rhs)| {
                super::window_summary_for_pair(&site_stats, &indices, FstMethod::Wc, lhs, rhs)
            })
            .collect::<Vec<_>>();
        assert_eq!(batched, individual);
    }

    #[test]
    fn streaming_window_accumulator_matches_overlapping_window_contract() {
        let mut accumulator = FstWindowAccumulator::new("chr1");
        let pairs = [(0usize, 1usize)];
        let make_site = |pos, lhs_alt, rhs_alt| WindowSiteStats {
            pos,
            stats: vec![
                GroupStats::from_counts(4, lhs_alt, 0),
                GroupStats::from_counts(4, rhs_alt, 0),
            ],
        };

        let mut emitted = Vec::new();
        emitted.extend(accumulator.push(make_site(10, 0, 8), 20, 10, FstMethod::Wc, &pairs));
        emitted.extend(accumulator.push(make_site(25, 2, 6), 20, 10, FstMethod::Wc, &pairs));
        emitted.extend(accumulator.finish(25, 20, 10, FstMethod::Wc, &pairs));

        assert_eq!(
            emitted
                .iter()
                .map(|window| (
                    window.start,
                    window.stop,
                    window.summaries[0].unwrap().n_variants
                ))
                .collect::<Vec<_>>(),
            vec![(1, 20, 1), (11, 30, 1), (21, 40, 1)],
        );
    }

    #[test]
    fn window_fst_rows_use_compact_header_value_precision() {
        let emission = FstWindowEmission {
            start: 1,
            stop: 100,
            summaries: vec![Some(FstWindowSummary {
                weighted: 0.123456,
                mean: 0.00001234,
                n_variants: 3,
            })],
        };
        let pairs = [(0usize, 1usize)];
        let masks = vec![
            GroupMask {
                name: "A".to_string(),
                words: Vec::new(),
                n_samples: 0,
            },
            GroupMask {
                name: "B".to_string(),
                words: Vec::new(),
                n_samples: 0,
            },
        ];
        let mut output = Vec::new();
        write_fst_window_emission(&mut output, "1", &emission, &pairs, &masks, false, "test")
            .unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "1\t1\t100\t3\t0.1235\t1.2340e-5\n"
        );
    }

    #[test]
    fn window_fst_headers_use_lowercase_analysis_names() {
        assert_eq!(
            super::FST_WINDOW_HEADER,
            "chrom\tstart\tend\tnsnp\tweightedFst\tmeanFst"
        );
        assert_eq!(
            super::FST_MATRIX_WINDOW_HEADER,
            "pop1\tpop2\tchrom\tstart\tend\tnsnp\tweightedFst\tmeanFst"
        );
    }

    #[test]
    fn window_fst_value_switches_to_scientific_notation_below_threshold() {
        assert_eq!(super::format_fst_window_value(0.0001), "0.0001");
        assert_eq!(super::format_fst_window_value(-0.000099), "-9.9000e-5");
        assert_eq!(super::format_fst_window_value(1.0), "1.0000");
    }

    #[test]
    fn streaming_rejects_bed_bim_mismatch_after_metadata_consumption() {
        let prefix = fst_fixture_path("mismatch");
        let output = fst_fixture_path("mismatch_output");
        write_fst_fixture(&prefix, 1, 2);

        let err = write_fst_tsv_streaming(
            prefix.to_str().unwrap(),
            prefix.with_extension("within").to_str().unwrap(),
            output.to_str().unwrap(),
            "wc",
            1,
        )
        .unwrap_err();

        assert!(
            err.contains("variant count mismatch while streaming"),
            "{err}"
        );
        for path in [
            prefix.with_extension("bed"),
            prefix.with_extension("bim"),
            prefix.with_extension("fam"),
            prefix.with_extension("within"),
            output,
        ] {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn streaming_preserves_existing_output_on_bim_mismatch() {
        let prefix = fst_fixture_path("preserve_output");
        let output = fst_fixture_path("preserve_output_result");
        write_fst_fixture(&prefix, 1, 2);
        std::fs::write(&output, "previous result\n").unwrap();

        let err = write_fst_tsv_streaming(
            prefix.to_str().unwrap(),
            prefix.with_extension("within").to_str().unwrap(),
            output.to_str().unwrap(),
            "wc",
            1,
        )
        .unwrap_err();

        assert!(
            err.contains("variant count mismatch while streaming"),
            "{err}"
        );
        assert_eq!(
            std::fs::read_to_string(&output).unwrap(),
            "previous result\n"
        );

        for path in [
            prefix.with_extension("bed"),
            prefix.with_extension("bim"),
            prefix.with_extension("fam"),
            prefix.with_extension("within"),
            output,
        ] {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn streaming_keeps_compact_rows_contiguous_across_block_boundary() {
        let prefix = fst_fixture_path("boundary");
        let output = fst_fixture_path("boundary_output");
        write_fst_fixture(&prefix, 4097, 4097);

        let result = write_fst_tsv_streaming(
            prefix.to_str().unwrap(),
            prefix.with_extension("within").to_str().unwrap(),
            output.to_str().unwrap(),
            "wc",
            1,
        )
        .unwrap();
        let lines = std::fs::read_to_string(&output)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>();

        assert_eq!(result, (4097, 1));
        assert_eq!(lines.len(), 4098);
        assert_eq!(lines[0], "CHR\tSNP\tPOS\tNMISS\tFST");
        assert_eq!(lines[4096], "1\trs4096\t4096\t4\tNaN");
        assert_eq!(lines[4097], "1\trs4097\t4097\t4\tNaN");

        for path in [
            prefix.with_extension("bed"),
            prefix.with_extension("bim"),
            prefix.with_extension("fam"),
            prefix.with_extension("within"),
            output,
        ] {
            let _ = std::fs::remove_file(path);
        }
    }
}
