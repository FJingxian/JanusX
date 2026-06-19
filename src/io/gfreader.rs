use pyo3::exceptions::*;
use pyo3::prelude::*;
use pyo3::BoundObject;

#[cfg(unix)]
use memmap2::Advice;
use memmap2::Mmap;
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;
#[cfg(target_arch = "x86")]
use std::arch::x86 as x86_avx2;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as x86_avx2;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
use std::sync::OnceLock;
use std::time::Instant;

use crate::bitwise::and_popcount;
use crate::gfcore as core;
use crate::gfcore::{BedSnpIter, HmpSnpIter, TxtSnpIter, VcfSnpIter};
use crate::gload::load_file_owned_range_exact;
use crate::vcfout::VcfOut;

// -------- Py-exposed SiteInfo (wrapper) --------
#[pyclass]
#[derive(Clone)]
pub struct SiteInfo {
    #[pyo3(get)]
    pub chrom: String,
    #[pyo3(get)]
    pub pos: i32,
    #[pyo3(get)]
    pub snp: String,
    #[pyo3(get)]
    pub ref_allele: String,
    #[pyo3(get)]
    pub alt_allele: String,
}

impl From<core::SiteInfo> for SiteInfo {
    fn from(s: core::SiteInfo) -> Self {
        Self {
            chrom: s.chrom,
            pos: s.pos,
            snp: s.snp,
            ref_allele: s.ref_allele,
            alt_allele: s.alt_allele,
        }
    }
}

#[pymethods]
impl SiteInfo {
    #[new]
    fn new(chrom: String, pos: i32, snp: String, ref_allele: String, alt_allele: String) -> Self {
        SiteInfo {
            chrom,
            pos,
            snp,
            ref_allele,
            alt_allele,
        }
    }
}

pub(crate) fn build_sample_selection(
    samples: &[String],
    sample_ids: Option<Vec<String>>,
    sample_indices: Option<Vec<usize>>,
) -> Result<(Vec<usize>, Vec<String>), String> {
    if sample_ids.is_some() && sample_indices.is_some() {
        return Err("Provide only one of sample_ids or sample_indices".into());
    }
    if let Some(ids) = sample_ids {
        if ids.is_empty() {
            return Err("sample_ids is empty".into());
        }
        let mut map: HashMap<&str, usize> = HashMap::with_capacity(samples.len());
        for (i, sid) in samples.iter().enumerate() {
            map.insert(sid.as_str(), i);
        }
        let mut seen: HashSet<usize> = HashSet::with_capacity(ids.len());
        let mut indices: Vec<usize> = Vec::with_capacity(ids.len());
        for sid in ids.iter() {
            let idx = *map
                .get(sid.as_str())
                .ok_or_else(|| format!("sample id not found: {sid}"))?;
            if !seen.insert(idx) {
                return Err(format!("duplicate sample id: {sid}"));
            }
            indices.push(idx);
        }
        return Ok((indices, ids));
    }
    if let Some(idxs) = sample_indices {
        if idxs.is_empty() {
            return Err("sample_indices is empty".into());
        }
        let mut seen: HashSet<usize> = HashSet::with_capacity(idxs.len());
        for &idx in idxs.iter() {
            if idx >= samples.len() {
                return Err(format!("sample index out of range: {idx}"));
            }
            if !seen.insert(idx) {
                return Err(format!("duplicate sample index: {idx}"));
            }
        }
        let ids: Vec<String> = idxs.iter().map(|&i| samples[i].clone()).collect();
        return Ok((idxs, ids));
    }

    let indices: Vec<usize> = (0..samples.len()).collect();
    Ok((indices, samples.to_vec()))
}

pub(crate) fn build_snp_indices(
    sites: &[core::SiteInfo],
    snp_range: Option<(usize, usize)>,
    snp_indices: Option<Vec<usize>>,
    bim_range: Option<(String, i32, i32)>,
    snp_sites: Option<Vec<(String, i32)>>,
) -> Result<Option<Vec<usize>>, String> {
    let mut count = 0;
    if snp_range.is_some() {
        count += 1;
    }
    if snp_indices.is_some() {
        count += 1;
    }
    if bim_range.is_some() {
        count += 1;
    }
    if snp_sites.is_some() {
        count += 1;
    }
    if count > 1 {
        return Err("Provide only one of snp_range, snp_indices, bim_range, or snp_sites".into());
    }

    if let Some((start, end)) = snp_range {
        let n = sites.len();
        if start >= end || end > n {
            return Err(format!("invalid snp_range: ({start}, {end})"));
        }
        let indices: Vec<usize> = (start..end).collect();
        return Ok(Some(indices));
    }

    if let Some(idxs) = snp_indices {
        if idxs.is_empty() {
            return Err("snp_indices is empty".into());
        }
        let mut seen: HashSet<usize> = HashSet::with_capacity(idxs.len());
        for &idx in idxs.iter() {
            if idx >= sites.len() {
                return Err(format!("snp index out of range: {idx}"));
            }
            if !seen.insert(idx) {
                return Err(format!("duplicate snp index: {idx}"));
            }
        }
        return Ok(Some(idxs));
    }

    if let Some((chrom, start, end)) = bim_range {
        if start > end {
            return Err("bim_range start > end".into());
        }
        let mut indices: Vec<usize> = Vec::new();
        for (i, site) in sites.iter().enumerate() {
            if site.chrom == chrom && site.pos >= start && site.pos <= end {
                indices.push(i);
            }
        }
        return Ok(Some(indices));
    }

    if let Some(site_keys) = snp_sites {
        if site_keys.is_empty() {
            return Err("snp_sites is empty".into());
        }
        let mut site_map: HashMap<(String, i32), Vec<usize>> = HashMap::new();
        for (i, site) in sites.iter().enumerate() {
            site_map
                .entry((site.chrom.clone(), site.pos))
                .or_default()
                .push(i);
        }

        let mut indices: Vec<usize> = Vec::new();
        for (chrom, pos) in site_keys.into_iter() {
            let key = (chrom.clone(), pos);
            let matched = site_map
                .get(&key)
                .ok_or_else(|| format!("snp site not found: ({chrom}, {pos})"))?;
            indices.extend(matched.iter().copied());
        }
        if indices.is_empty() {
            return Err("no SNPs matched from snp_sites".into());
        }
        return Ok(Some(indices));
    }

    Ok(None)
}

#[inline]
fn normalize_plink_prefix_local(p: &str) -> String {
    let s = p.trim();
    let low = s.to_ascii_lowercase();
    if low.ends_with(".bed") || low.ends_with(".bim") || low.ends_with(".fam") {
        return s[..s.len() - 4].to_string();
    }
    s.to_string()
}

#[inline]
fn normalize_chr_key_local(chrom: &str) -> String {
    let mut s = chrom.trim().to_string();
    let low = s.to_ascii_lowercase();
    if low.starts_with("chr") {
        s = s[3..].to_string();
    }
    s.trim().to_ascii_uppercase()
}

#[inline]
fn env_truthy_local(name: &str) -> bool {
    match std::env::var(name) {
        Ok(raw) => {
            let key = raw.trim().to_ascii_lowercase();
            matches!(key.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

#[inline]
fn gfreader_rss_debug_enabled() -> bool {
    env_truthy_local("JX_GFREADER_RSS_DEBUG")
        || env_truthy_local("JX_PACKED_IO_DEBUG")
        || env_truthy_local("JX_GS_DEBUG_STAGE")
}

#[inline]
fn bed_logic_meta_stage_timing_enabled() -> bool {
    env_truthy_local("JX_BED_LOGIC_META_TIMING")
        || env_truthy_local("JX_SPLMM_PREPARE_STAGE_TIMING")
        || env_truthy_local("JX_SPLMM_PACKED_STAGE_TIMING")
}

#[inline]
fn timing_pct_local(part_secs: f64, total_secs: f64) -> f64 {
    if total_secs > 0.0_f64 {
        (part_secs / total_secs) * 100.0_f64
    } else {
        0.0_f64
    }
}

fn emit_bed_logic_meta_timing(
    stage: &str,
    read_fam_secs: f64,
    read_bim_secs: f64,
    mmap_secs: f64,
    row_stats_secs: f64,
    site_keep_secs: f64,
    pack_kept_secs: f64,
    total_secs: f64,
    n_samples: usize,
    stats_n_samples: usize,
    n_snps: usize,
    kept_n: usize,
    stats_only: bool,
) {
    if !bed_logic_meta_stage_timing_enabled() {
        return;
    }
    let accounted = read_fam_secs
        + read_bim_secs
        + mmap_secs
        + row_stats_secs
        + site_keep_secs
        + pack_kept_secs;
    let other_secs = (total_secs - accounted).max(0.0_f64);
    eprintln!(
        "BED logic meta timing stage={stage}: read_fam={:.3}s ({:.1}%), read_bim={:.3}s ({:.1}%), mmap={:.3}s ({:.1}%), row_stats={:.3}s ({:.1}%), site_keep={:.3}s ({:.1}%), pack_kept={:.3}s ({:.1}%), other={:.3}s ({:.1}%), total={:.3}s, n_samples={}, stats_n_samples={}, n_snps={}, kept_n={}, stats_only={}",
        read_fam_secs,
        timing_pct_local(read_fam_secs, total_secs),
        read_bim_secs,
        timing_pct_local(read_bim_secs, total_secs),
        mmap_secs,
        timing_pct_local(mmap_secs, total_secs),
        row_stats_secs,
        timing_pct_local(row_stats_secs, total_secs),
        site_keep_secs,
        timing_pct_local(site_keep_secs, total_secs),
        pack_kept_secs,
        timing_pct_local(pack_kept_secs, total_secs),
        other_secs,
        timing_pct_local(other_secs, total_secs),
        total_secs,
        n_samples,
        stats_n_samples,
        n_snps,
        kept_n,
        stats_only,
    );
}

fn emit_bed_logic_meta_py_timing(
    stage: &str,
    rust_core_secs: f64,
    py_arrays_secs: f64,
    total_secs: f64,
    n_samples_full: usize,
    n_snps_total: usize,
    kept_n: usize,
) {
    if !bed_logic_meta_stage_timing_enabled() {
        return;
    }
    let other_secs = (total_secs - rust_core_secs - py_arrays_secs).max(0.0_f64);
    eprintln!(
        "BED logic meta timing stage={stage}: rust_core={:.3}s ({:.1}%), py_arrays={:.3}s ({:.1}%), other={:.3}s ({:.1}%), total={:.3}s, n_samples_full={}, n_snps_total={}, kept_n={}",
        rust_core_secs,
        timing_pct_local(rust_core_secs, total_secs),
        py_arrays_secs,
        timing_pct_local(py_arrays_secs, total_secs),
        other_secs,
        timing_pct_local(other_secs, total_secs),
        total_secs,
        n_samples_full,
        n_snps_total,
        kept_n,
    );
}

#[inline]
fn format_debug_bytes_local(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.1} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(target_os = "linux")]
#[inline]
fn process_rss_bytes_local() -> Option<(u64, &'static str)> {
    let text = std::fs::read_to_string("/proc/self/statm").ok()?;
    let mut fields = text.split_whitespace();
    let _size_pages = fields.next()?;
    let rss_pages: u64 = fields.next()?.parse().ok()?;
    // SAFETY: sysconf is thread-safe for _SC_PAGESIZE and has no side effects.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return None;
    }
    Some((rss_pages.saturating_mul(page_size as u64), "current"))
}

#[cfg(target_os = "macos")]
#[inline]
fn process_rss_bytes_local() -> Option<(u64, &'static str)> {
    let mut ru = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: ru points to valid writable storage for getrusage.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: getrusage succeeded and initialized ru.
    let ru = unsafe { ru.assume_init() };
    Some((ru.ru_maxrss as u64, "peak"))
}

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
#[inline]
fn process_rss_bytes_local() -> Option<(u64, &'static str)> {
    let mut ru = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: ru points to valid writable storage for getrusage.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: getrusage succeeded and initialized ru.
    let ru = unsafe { ru.assume_init() };
    Some(((ru.ru_maxrss as u64).saturating_mul(1024), "peak"))
}

#[cfg(not(unix))]
#[inline]
fn process_rss_bytes_local() -> Option<(u64, &'static str)> {
    None
}

fn emit_gfreader_rss_debug(stage: &str, detail: &str) {
    if !gfreader_rss_debug_enabled() {
        return;
    }
    match process_rss_bytes_local() {
        Some((rss_bytes, rss_kind)) => {
            println!(
                "[GFREADER-DEBUG] {stage} rss={} rss_kind={} {detail}",
                format_debug_bytes_local(rss_bytes),
                rss_kind,
            );
        }
        None => {
            println!("[GFREADER-DEBUG] {stage} rss=NA rss_kind=unavailable {detail}");
        }
    }
    let _ = std::io::stdout().flush();
}

fn parse_npy_shape_local(header: &str) -> Result<(usize, usize), String> {
    let shape_key_pos = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| "NPY header missing shape field".to_string())?;
    let after = &header[shape_key_pos..];
    let open = after
        .find('(')
        .ok_or_else(|| "NPY header has malformed shape tuple".to_string())?;
    let close = after[open + 1..]
        .find(')')
        .ok_or_else(|| "NPY header has malformed shape tuple".to_string())?;
    let inside = &after[open + 1..open + 1 + close];

    let dims: Vec<usize> = inside
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| format!("invalid NPY shape dimension: {s}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    match dims.as_slice() {
        [rows] => Ok((*rows, 1)),
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(format!("unsupported NPY shape rank: {:?}", dims)),
    }
}

fn parse_npy_f32_header_local(bytes: &[u8]) -> Result<(usize, usize, usize), String> {
    if bytes.len() < 10 {
        return Err("NPY file too small".into());
    }
    if &bytes[0..6] != b"\x93NUMPY" {
        return Err("invalid NPY magic".into());
    }

    let major = bytes[6];
    let minor = bytes[7];
    let (header_len, header_start) = match major {
        1 => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10usize)
        }
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("NPY file too small for v2/v3 header".into());
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12usize)
        }
        _ => return Err(format!("unsupported NPY version: {major}.{minor}")),
    };

    let header_end = header_start
        .checked_add(header_len)
        .ok_or_else(|| "NPY header overflow".to_string())?;
    if header_end > bytes.len() {
        return Err("NPY header exceeds file size".into());
    }

    let header =
        std::str::from_utf8(&bytes[header_start..header_end]).map_err(|e| e.to_string())?;
    if !header.contains("descr': '<f4'")
        && !header.contains("descr': '|f4'")
        && !header.contains("descr\": \"<f4\"")
        && !header.contains("descr\": \"|f4\"")
    {
        return Err("NPY dtype is not float32".into());
    }
    if header.contains("fortran_order': True") || header.contains("fortran_order\": true") {
        return Err("fortran_order=True NPY is not supported".into());
    }

    let (rows, cols) = parse_npy_shape_local(header)?;
    let data_offset = header_end;
    let data_bytes = rows
        .checked_mul(cols)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| "NPY data size overflow".to_string())?;
    let expected_end = data_offset
        .checked_add(data_bytes)
        .ok_or_else(|| "NPY file size overflow".to_string())?;
    if expected_end > bytes.len() {
        return Err("NPY data truncated".into());
    }

    Ok((rows, cols, data_offset))
}

#[derive(Default, Clone)]
struct SiteFilterExpr {
    site_set: Option<HashSet<(String, i32)>>,
    bim_range: Option<(String, i32, i32)>,
    chr_set: Option<HashSet<String>>,
    bp_min: Option<i32>,
    bp_max: Option<i32>,
    ranges: Option<Vec<(String, i32, i32)>>,
}

impl SiteFilterExpr {
    fn from_parts(
        snp_sites: Option<Vec<(String, i32)>>,
        bim_range: Option<(String, i32, i32)>,
        chr_keys: Option<Vec<String>>,
        bp_min: Option<i32>,
        bp_max: Option<i32>,
        ranges: Option<Vec<(String, i32, i32)>>,
    ) -> Result<Self, String> {
        if let (Some(lo), Some(hi)) = (bp_min, bp_max) {
            if lo > hi {
                return Err("bp_min cannot be greater than bp_max".to_string());
            }
        }

        let site_set: Option<HashSet<(String, i32)>> = snp_sites.and_then(|v| {
            let mut s: HashSet<(String, i32)> = HashSet::new();
            for (c, p) in v.into_iter() {
                s.insert((normalize_chr_key_local(&c), p));
            }
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        });

        let bim_range: Option<(String, i32, i32)> = if let Some((c, s, e)) = bim_range {
            if s > e {
                return Err("bim_range start cannot be greater than end".to_string());
            }
            Some((normalize_chr_key_local(&c), s, e))
        } else {
            None
        };

        let chr_set: Option<HashSet<String>> = chr_keys.and_then(|v| {
            let mut s: HashSet<String> = HashSet::new();
            for c in v.into_iter() {
                let k = normalize_chr_key_local(&c);
                if !k.is_empty() {
                    s.insert(k);
                }
            }
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        });

        let ranges: Option<Vec<(String, i32, i32)>> = if let Some(v) = ranges {
            if v.is_empty() {
                None
            } else {
                let mut out: Vec<(String, i32, i32)> = Vec::with_capacity(v.len());
                for (c, s, e) in v.into_iter() {
                    if s > e {
                        return Err("One range has start > end".to_string());
                    }
                    out.push((normalize_chr_key_local(&c), s, e));
                }
                Some(out)
            }
        } else {
            None
        };

        Ok(Self {
            site_set,
            bim_range,
            chr_set,
            bp_min,
            bp_max,
            ranges,
        })
    }

    #[inline]
    fn keep_site(&self, site: &core::SiteInfo) -> bool {
        let c = normalize_chr_key_local(&site.chrom);
        let p = site.pos;

        if let Some(ref st) = self.site_set {
            if !st.contains(&(c.clone(), p)) {
                return false;
            }
        }
        if let Some((ref bc, bs, be)) = self.bim_range {
            if !(c == *bc && p >= bs && p <= be) {
                return false;
            }
        }
        if let Some(ref cs) = self.chr_set {
            if !cs.contains(&c) {
                return false;
            }
        }
        if let Some(lo) = self.bp_min {
            if p < lo {
                return false;
            }
        }
        if let Some(hi) = self.bp_max {
            if p > hi {
                return false;
            }
        }
        if let Some(ref rr) = self.ranges {
            let mut hit = false;
            for (rc, rs, re) in rr.iter() {
                if c == *rc && p >= *rs && p <= *re {
                    hit = true;
                    break;
                }
            }
            if !hit {
                return false;
            }
        }

        true
    }

    #[inline]
    fn active(&self) -> bool {
        self.site_set.is_some()
            || self.bim_range.is_some()
            || self.chr_set.is_some()
            || self.bp_min.is_some()
            || self.bp_max.is_some()
            || self.ranges.is_some()
    }
}

#[inline]
pub(crate) fn sample_indices_are_identity(indices: &[usize]) -> bool {
    indices.iter().enumerate().all(|(i, &idx)| idx == i)
}

#[derive(Clone, Debug)]
pub(crate) struct SampleSubsetPlan {
    n_samples_full: usize,
    selected: Option<Vec<usize>>,
    excluded: Option<Vec<usize>>,
    identity: bool,
}

impl SampleSubsetPlan {
    pub(crate) fn from_optional_indices(
        n_samples_full: usize,
        sample_indices: Option<&[usize]>,
    ) -> Self {
        match sample_indices {
            Some(indices)
                if !(indices.is_empty()
                    || (indices.len() == n_samples_full
                        && sample_indices_are_identity(indices))) =>
            {
                let selected = indices.to_vec();
                let excluded = precompute_excluded_sample_indices(n_samples_full, indices);
                Self {
                    n_samples_full,
                    selected: Some(selected),
                    excluded,
                    identity: false,
                }
            }
            _ => Self {
                n_samples_full,
                selected: None,
                excluded: None,
                identity: true,
            },
        }
    }

    #[inline]
    pub(crate) fn is_identity(&self) -> bool {
        self.identity
    }

    #[inline]
    pub(crate) fn n_selected(&self) -> usize {
        self.selected
            .as_ref()
            .map(|indices| indices.len())
            .unwrap_or(self.n_samples_full)
    }

    #[inline]
    pub(crate) fn selected(&self) -> Option<&[usize]> {
        self.selected.as_deref()
    }

    #[inline]
    pub(crate) fn excluded(&self) -> Option<&[usize]> {
        self.excluded.as_deref()
    }
}

#[inline]
fn write_bim_site_line(w: &mut BufWriter<File>, site: &core::SiteInfo) -> Result<(), String> {
    write_bim_site_line_with_flip(w, site, false)
}

#[inline]
fn write_bim_site_line_with_flip(
    w: &mut BufWriter<File>,
    site: &core::SiteInfo,
    flip: bool,
) -> Result<(), String> {
    let (ref_allele, alt_allele) = if flip {
        (&site.alt_allele, &site.ref_allele)
    } else {
        (&site.ref_allele, &site.alt_allele)
    };
    writeln!(
        w,
        "{}\t{}_{}\t0\t{}\t{}\t{}",
        site.chrom, site.chrom, site.pos, site.pos, ref_allele, alt_allele
    )
    .map_err(|e| e.to_string())
}

#[inline]
fn flip_bed_byte_fast(byte: u8) -> u8 {
    let lo = byte & 0x55u8;
    let hi = (byte >> 1) & 0x55u8;
    let eq = (!(lo ^ hi)) & 0x55u8;
    byte ^ (eq | (eq << 1))
}

#[inline]
fn flip_bed_word64(word: u64) -> u64 {
    const M55: u64 = 0x5555_5555_5555_5555_u64;
    let lo = word & M55;
    let hi = (word >> 1) & M55;
    let eq = (!(lo ^ hi)) & M55;
    word ^ (eq | (eq << 1))
}

#[inline]
fn flip_bed_bytes_into(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    let mut i = 0usize;
    let n = src.len();
    while i + 8 <= n {
        // SAFETY: i + 8 <= n ensures the unaligned u64 load stays within src bounds.
        let word = unsafe { std::ptr::read_unaligned(src.as_ptr().add(i) as *const u64) };
        let flipped = flip_bed_word64(u64::from_le(word));
        // SAFETY: i + 8 <= n and src.len() == dst.len() ensure destination is in-bounds.
        unsafe {
            std::ptr::write_unaligned(dst.as_mut_ptr().add(i) as *mut u64, flipped.to_le());
        }
        i += 8;
    }
    while i < n {
        dst[i] = flip_bed_byte_fast(src[i]);
        i += 1;
    }
}

#[inline]
fn flip_bed_bytes_in_place(bytes: &mut [u8]) {
    let mut i = 0usize;
    let n = bytes.len();
    while i + 8 <= n {
        // SAFETY: i + 8 <= n ensures the unaligned u64 load/store stays within bounds.
        let ptr = unsafe { bytes.as_mut_ptr().add(i) };
        // SAFETY: ptr points to a valid contiguous 8-byte region of bytes.
        let word = unsafe { std::ptr::read_unaligned(ptr as *const u64) };
        let flipped = flip_bed_word64(u64::from_le(word));
        // SAFETY: ptr points to the same valid contiguous 8-byte region.
        unsafe {
            std::ptr::write_unaligned(ptr as *mut u64, flipped.to_le());
        }
        i += 8;
    }
    while i < n {
        bytes[i] = flip_bed_byte_fast(bytes[i]);
        i += 1;
    }
}

fn build_bed_low_high_nibble_tables() -> ([u8; 256], [u8; 256]) {
    let mut low = [0u8; 256];
    let mut high = [0u8; 256];
    for b in 0u16..=255u16 {
        let x = b as u8;
        let lo = (x & 0b0000_0001)
            | ((x & 0b0000_0100) >> 1)
            | ((x & 0b0001_0000) >> 2)
            | ((x & 0b0100_0000) >> 3);
        let hi = ((x & 0b0000_0010) >> 1)
            | ((x & 0b0000_1000) >> 2)
            | ((x & 0b0010_0000) >> 3)
            | ((x & 0b1000_0000) >> 4);
        low[b as usize] = lo;
        high[b as usize] = hi;
    }
    (low, high)
}

#[inline]
fn count_packed_row_full_bytes_popcnt(row_full: &[u8]) -> (usize, usize, usize) {
    let mut missing = 0usize;
    let mut het = 0usize;
    let mut hom_alt = 0usize;
    let m55 = 0x5555_5555_5555_5555_u64;

    let mut chunks = row_full.chunks_exact(8);
    for chunk in &mut chunks {
        // SAFETY: chunks_exact(8) guarantees chunk length is exactly 8 bytes.
        let word = unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const u64) };
        let word = u64::from_le(word);
        let odd = (word >> 1) & m55;
        let even = word & m55;
        missing = missing.saturating_add(((!odd) & even).count_ones() as usize);
        het = het.saturating_add((odd & (!even)).count_ones() as usize);
        hom_alt = hom_alt.saturating_add((odd & even).count_ones() as usize);
    }
    for &b in chunks.remainder().iter() {
        let word = b as u64;
        let odd = (word >> 1) & 0x55_u64;
        let even = word & 0x55_u64;
        missing = missing.saturating_add(((!odd) & even).count_ones() as usize);
        het = het.saturating_add((odd & (!even)).count_ones() as usize);
        hom_alt = hom_alt.saturating_add((odd & even).count_ones() as usize);
    }
    (missing, het, hom_alt)
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
const BED_COUNT_SIMD_MIN_BYTES: usize = 64;

#[cfg(target_arch = "aarch64")]
#[inline]
fn bed_neon_runtime_available() -> bool {
    static NEON: OnceLock<bool> = OnceLock::new();
    *NEON.get_or_init(|| std::arch::is_aarch64_feature_detected!("neon"))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn popcount_u8x16_neon(v: std::arch::aarch64::uint8x16_t) -> u64 {
    use std::arch::aarch64::*;
    let cnt8 = vcntq_u8(v);
    let sum16 = vpaddlq_u8(cnt8);
    let sum32 = vpaddlq_u16(sum16);
    let sum64 = vpaddlq_u32(sum32);
    vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn count_packed_row_full_bytes_neon(row_full: &[u8]) -> (usize, usize, usize) {
    use std::arch::aarch64::*;
    let even_mask = vdupq_n_u8(0x55u8);
    let mut lo_sum = 0u64;
    let mut hi_sum = 0u64;
    let mut ha_sum = 0u64;
    let mut i = 0usize;
    let n = row_full.len();

    while i + 64 <= n {
        let v0 = vld1q_u8(row_full.as_ptr().add(i));
        let lo0 = vandq_u8(v0, even_mask);
        let hi0 = vandq_u8(vshrq_n_u8(v0, 1), even_mask);
        let ha0 = vandq_u8(lo0, hi0);
        lo_sum += popcount_u8x16_neon(lo0);
        hi_sum += popcount_u8x16_neon(hi0);
        ha_sum += popcount_u8x16_neon(ha0);

        let v1 = vld1q_u8(row_full.as_ptr().add(i + 16));
        let lo1 = vandq_u8(v1, even_mask);
        let hi1 = vandq_u8(vshrq_n_u8(v1, 1), even_mask);
        let ha1 = vandq_u8(lo1, hi1);
        lo_sum += popcount_u8x16_neon(lo1);
        hi_sum += popcount_u8x16_neon(hi1);
        ha_sum += popcount_u8x16_neon(ha1);

        let v2 = vld1q_u8(row_full.as_ptr().add(i + 32));
        let lo2 = vandq_u8(v2, even_mask);
        let hi2 = vandq_u8(vshrq_n_u8(v2, 1), even_mask);
        let ha2 = vandq_u8(lo2, hi2);
        lo_sum += popcount_u8x16_neon(lo2);
        hi_sum += popcount_u8x16_neon(hi2);
        ha_sum += popcount_u8x16_neon(ha2);

        let v3 = vld1q_u8(row_full.as_ptr().add(i + 48));
        let lo3 = vandq_u8(v3, even_mask);
        let hi3 = vandq_u8(vshrq_n_u8(v3, 1), even_mask);
        let ha3 = vandq_u8(lo3, hi3);
        lo_sum += popcount_u8x16_neon(lo3);
        hi_sum += popcount_u8x16_neon(hi3);
        ha_sum += popcount_u8x16_neon(ha3);

        i += 64;
    }

    while i + 16 <= n {
        let v = vld1q_u8(row_full.as_ptr().add(i));
        let lo = vandq_u8(v, even_mask);
        let hi = vandq_u8(vshrq_n_u8(v, 1), even_mask);
        let ha = vandq_u8(lo, hi);
        lo_sum += popcount_u8x16_neon(lo);
        hi_sum += popcount_u8x16_neon(hi);
        ha_sum += popcount_u8x16_neon(ha);
        i += 16;
    }

    let (sm, sh, sha) = count_packed_row_full_bytes_popcnt(&row_full[i..]);
    let hom_alt = (ha_sum as usize).saturating_add(sha);
    let missing = (lo_sum.saturating_sub(ha_sum) as usize).saturating_add(sm);
    let het = (hi_sum.saturating_sub(ha_sum) as usize).saturating_add(sh);
    (missing, het, hom_alt)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
type BedX86M256i = x86_avx2::__m256i;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn bed_avx2_runtime_available() -> bool {
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum BedAvx2Mode {
    Baseline,
    Aggressive,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn bed_avx2_runtime_mode() -> BedAvx2Mode {
    static MODE: OnceLock<BedAvx2Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let raw = std::env::var("JANUSX_BED_AVX2_MODE").unwrap_or_default();
        let key = raw.trim().to_ascii_lowercase();
        if key == "baseline" || key == "base" || key == "0" {
            BedAvx2Mode::Baseline
        } else {
            BedAvx2Mode::Aggressive
        }
    })
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn popcount_u8x32_avx2(v: BedX86M256i) -> u64 {
    use x86_avx2::*;
    let lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0F_i8);
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    let cnt_lo = _mm256_shuffle_epi8(lut, lo);
    let cnt_hi = _mm256_shuffle_epi8(lut, hi);
    let cnt = _mm256_add_epi8(cnt_lo, cnt_hi);
    let sad = _mm256_sad_epu8(cnt, _mm256_setzero_si256());
    let mut sums = [0u64; 4];
    _mm256_storeu_si256(sums.as_mut_ptr() as *mut BedX86M256i, sad);
    sums[0] + sums[1] + sums[2] + sums[3]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn bed_avx2_accumulate_block(
    v: BedX86M256i,
    even_mask: BedX86M256i,
    lo_sum: &mut u64,
    hi_sum: &mut u64,
    ha_sum: &mut u64,
) {
    use x86_avx2::*;
    let lo = _mm256_and_si256(v, even_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 1), even_mask);
    let ha = _mm256_and_si256(lo, hi);
    *lo_sum += popcount_u8x32_avx2(lo);
    *hi_sum += popcount_u8x32_avx2(hi);
    *ha_sum += popcount_u8x32_avx2(ha);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn count_packed_row_full_bytes_avx2_baseline(row_full: &[u8]) -> (usize, usize, usize) {
    use x86_avx2::*;
    let even_mask = _mm256_set1_epi8(0x55_i8);
    let mut lo_sum = 0u64;
    let mut hi_sum = 0u64;
    let mut ha_sum = 0u64;
    let mut i = 0usize;
    let n = row_full.len();

    while i + 128 <= n {
        let v0 = _mm256_loadu_si256(row_full.as_ptr().add(i) as *const BedX86M256i);
        bed_avx2_accumulate_block(v0, even_mask, &mut lo_sum, &mut hi_sum, &mut ha_sum);

        let v1 = _mm256_loadu_si256(row_full.as_ptr().add(i + 32) as *const BedX86M256i);
        bed_avx2_accumulate_block(v1, even_mask, &mut lo_sum, &mut hi_sum, &mut ha_sum);

        let v2 = _mm256_loadu_si256(row_full.as_ptr().add(i + 64) as *const BedX86M256i);
        bed_avx2_accumulate_block(v2, even_mask, &mut lo_sum, &mut hi_sum, &mut ha_sum);

        let v3 = _mm256_loadu_si256(row_full.as_ptr().add(i + 96) as *const BedX86M256i);
        bed_avx2_accumulate_block(v3, even_mask, &mut lo_sum, &mut hi_sum, &mut ha_sum);

        i += 128;
    }

    while i + 32 <= n {
        let v = _mm256_loadu_si256(row_full.as_ptr().add(i) as *const BedX86M256i);
        bed_avx2_accumulate_block(v, even_mask, &mut lo_sum, &mut hi_sum, &mut ha_sum);
        i += 32;
    }

    let (sm, sh, sha) = count_packed_row_full_bytes_popcnt(&row_full[i..]);
    let hom_alt = (ha_sum as usize).saturating_add(sha);
    let missing = (lo_sum.saturating_sub(ha_sum) as usize).saturating_add(sm);
    let het = (hi_sum.saturating_sub(ha_sum) as usize).saturating_add(sh);
    (missing, het, hom_alt)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const BED_AVX2_PREFETCH_DISTANCE_BYTES: usize = 1024;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn count_packed_row_full_bytes_avx2_aggressive(row_full: &[u8]) -> (usize, usize, usize) {
    use x86_avx2::*;
    let even_mask = _mm256_set1_epi8(0x55_i8);
    let mut lo_sum = 0u64;
    let mut hi_sum = 0u64;
    let mut ha_sum = 0u64;
    let mut i = 0usize;
    let n = row_full.len();

    while i + 256 <= n {
        if let Some(pf_idx) = i.checked_add(BED_AVX2_PREFETCH_DISTANCE_BYTES) {
            if pf_idx < n {
                _mm_prefetch(row_full.as_ptr().add(pf_idx) as *const i8, _MM_HINT_T0);
            }
        }

        let v0 = _mm256_loadu_si256(row_full.as_ptr().add(i) as *const BedX86M256i);
        bed_avx2_accumulate_block(v0, even_mask, &mut lo_sum, &mut hi_sum, &mut ha_sum);
        let v1 = _mm256_loadu_si256(row_full.as_ptr().add(i + 32) as *const BedX86M256i);
        bed_avx2_accumulate_block(v1, even_mask, &mut lo_sum, &mut hi_sum, &mut ha_sum);
        let v2 = _mm256_loadu_si256(row_full.as_ptr().add(i + 64) as *const BedX86M256i);
        bed_avx2_accumulate_block(v2, even_mask, &mut lo_sum, &mut hi_sum, &mut ha_sum);
        let v3 = _mm256_loadu_si256(row_full.as_ptr().add(i + 96) as *const BedX86M256i);
        bed_avx2_accumulate_block(v3, even_mask, &mut lo_sum, &mut hi_sum, &mut ha_sum);
        let v4 = _mm256_loadu_si256(row_full.as_ptr().add(i + 128) as *const BedX86M256i);
        bed_avx2_accumulate_block(v4, even_mask, &mut lo_sum, &mut hi_sum, &mut ha_sum);
        let v5 = _mm256_loadu_si256(row_full.as_ptr().add(i + 160) as *const BedX86M256i);
        bed_avx2_accumulate_block(v5, even_mask, &mut lo_sum, &mut hi_sum, &mut ha_sum);
        let v6 = _mm256_loadu_si256(row_full.as_ptr().add(i + 192) as *const BedX86M256i);
        bed_avx2_accumulate_block(v6, even_mask, &mut lo_sum, &mut hi_sum, &mut ha_sum);
        let v7 = _mm256_loadu_si256(row_full.as_ptr().add(i + 224) as *const BedX86M256i);
        bed_avx2_accumulate_block(v7, even_mask, &mut lo_sum, &mut hi_sum, &mut ha_sum);

        i += 256;
    }

    let (sm, sh, sha) = count_packed_row_full_bytes_avx2_baseline(&row_full[i..]);
    let hom_alt = (ha_sum as usize).saturating_add(sha);
    let missing = (lo_sum.saturating_sub(ha_sum) as usize).saturating_add(sm);
    let het = (hi_sum.saturating_sub(ha_sum) as usize).saturating_add(sh);
    (missing, het, hom_alt)
}

#[inline]
fn count_packed_row_full_bytes_dispatch(row_full: &[u8]) -> (usize, usize, usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if row_full.len() >= BED_COUNT_SIMD_MIN_BYTES && bed_avx2_runtime_available() {
            // SAFETY: runtime-gated by AVX2 feature detection.
            return unsafe {
                match bed_avx2_runtime_mode() {
                    BedAvx2Mode::Baseline => count_packed_row_full_bytes_avx2_baseline(row_full),
                    BedAvx2Mode::Aggressive => {
                        count_packed_row_full_bytes_avx2_aggressive(row_full)
                    }
                }
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if row_full.len() >= BED_COUNT_SIMD_MIN_BYTES && bed_neon_runtime_available() {
            // SAFETY: runtime-gated by NEON feature detection.
            return unsafe { count_packed_row_full_bytes_neon(row_full) };
        }
    }
    count_packed_row_full_bytes_popcnt(row_full)
}

#[inline]
pub(crate) fn count_packed_row_counts(row: &[u8], n_samples: usize) -> (usize, usize, usize) {
    let full_bytes = n_samples / 4;
    let rem_pairs = n_samples & 3;
    let (mut missing, mut het, mut hom_alt) =
        count_packed_row_full_bytes_dispatch(&row[..full_bytes]);

    if rem_pairs > 0 {
        let b = row[full_bytes];
        let mask = (1u8 << (rem_pairs * 2)) - 1u8;
        let word = (b & mask) as u64;
        let odd = (word >> 1) & 0x55_u64;
        let even = word & 0x55_u64;
        missing = missing.saturating_add(((!odd) & even).count_ones() as usize);
        het = het.saturating_add((odd & (!even)).count_ones() as usize);
        hom_alt = hom_alt.saturating_add((odd & even).count_ones() as usize);
    }
    (missing, het, hom_alt)
}

#[inline]
fn count_packed_row_counts_scalar_indices(
    row: &[u8],
    sample_indices: &[usize],
) -> (usize, usize, usize) {
    let mut missing = 0usize;
    let mut het = 0usize;
    let mut hom_alt = 0usize;
    for &sid in sample_indices.iter() {
        let code = (row[sid >> 2] >> ((sid & 3) * 2)) & 0b11;
        match code {
            0b01 => missing += 1,
            0b10 => het += 1,
            0b11 => hom_alt += 1,
            _ => {}
        }
    }
    (missing, het, hom_alt)
}

#[inline]
pub(crate) fn precompute_excluded_sample_indices(
    n_samples: usize,
    sample_indices: &[usize],
) -> Option<Vec<usize>> {
    if sample_indices.is_empty() || sample_indices.len() >= n_samples {
        return None;
    }
    let mut selected_mask = vec![false; n_samples];
    let mut unique_selected = 0usize;
    for &sid in sample_indices {
        if sid >= n_samples {
            return None;
        }
        if !selected_mask[sid] {
            selected_mask[sid] = true;
            unique_selected += 1;
        }
    }
    if unique_selected != sample_indices.len() {
        return None;
    }
    let excluded_n = n_samples.saturating_sub(unique_selected);
    if excluded_n == 0 || excluded_n.saturating_mul(4) > unique_selected {
        return None;
    }
    let mut excluded = Vec::with_capacity(excluded_n);
    for (sid, &selected) in selected_mask.iter().enumerate() {
        if !selected {
            excluded.push(sid);
        }
    }
    Some(excluded)
}

#[inline]
pub(crate) fn count_packed_row_counts_selected_with_excluded(
    row: &[u8],
    n_samples: usize,
    sample_indices: &[usize],
    excluded_sample_indices: Option<&[usize]>,
) -> (usize, usize, usize) {
    if sample_indices_are_identity(sample_indices) && sample_indices.len() == n_samples {
        return count_packed_row_counts(row, n_samples);
    }
    if let Some(excluded_sample_indices) = excluded_sample_indices {
        let (missing, het, hom_alt) = count_packed_row_counts(row, n_samples);
        let (missing_ex, het_ex, hom_alt_ex) =
            count_packed_row_counts_scalar_indices(row, excluded_sample_indices);
        return (
            missing.saturating_sub(missing_ex),
            het.saturating_sub(het_ex),
            hom_alt.saturating_sub(hom_alt_ex),
        );
    }
    count_packed_row_counts_scalar_indices(row, sample_indices)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn count_packed_row_counts_selected(
    row: &[u8],
    n_samples: usize,
    sample_indices: &[usize],
) -> (usize, usize, usize) {
    count_packed_row_counts_selected_with_excluded(row, n_samples, sample_indices, None)
}

#[inline]
pub(crate) fn count_packed_row_pure_line_counts(row: &[u8], n_samples: usize) -> (usize, usize) {
    let full_bytes = n_samples / 4;
    let rem_pairs = n_samples & 3;
    let mut logic_missing = 0usize;
    let mut hom_alt = 0usize;

    for &b in row.iter().take(full_bytes) {
        let word = b as u64;
        let odd = (word >> 1) & 0x55_u64;
        let even = word & 0x55_u64;
        logic_missing = logic_missing.saturating_add((odd ^ even).count_ones() as usize);
        hom_alt = hom_alt.saturating_add((odd & even).count_ones() as usize);
    }

    if rem_pairs > 0 {
        let b = row[full_bytes];
        let mask = (1u8 << (rem_pairs * 2)) - 1u8;
        let word = (b & mask) as u64;
        let odd = (word >> 1) & 0x55_u64;
        let even = word & 0x55_u64;
        logic_missing = logic_missing.saturating_add((odd ^ even).count_ones() as usize);
        hom_alt = hom_alt.saturating_add((odd & even).count_ones() as usize);
    }
    (logic_missing, hom_alt)
}

#[inline]
fn count_packed_row_pure_line_counts_scalar_indices(
    row: &[u8],
    sample_indices: &[usize],
) -> (usize, usize) {
    let mut logic_missing = 0usize;
    let mut hom_alt = 0usize;
    for &sid in sample_indices.iter() {
        let code = (row[sid >> 2] >> ((sid & 3) * 2)) & 0b11;
        match code {
            0b01 | 0b10 => logic_missing += 1,
            0b11 => hom_alt += 1,
            _ => {}
        }
    }
    (logic_missing, hom_alt)
}

#[inline]
pub(crate) fn count_packed_row_pure_line_counts_selected_with_excluded(
    row: &[u8],
    n_samples: usize,
    sample_indices: &[usize],
    excluded_sample_indices: Option<&[usize]>,
) -> (usize, usize) {
    if sample_indices_are_identity(sample_indices) && sample_indices.len() == n_samples {
        return count_packed_row_pure_line_counts(row, n_samples);
    }
    if let Some(excluded_sample_indices) = excluded_sample_indices {
        let (logic_missing, hom_alt) = count_packed_row_pure_line_counts(row, n_samples);
        let (logic_missing_ex, hom_alt_ex) =
            count_packed_row_pure_line_counts_scalar_indices(row, excluded_sample_indices);
        return (
            logic_missing.saturating_sub(logic_missing_ex),
            hom_alt.saturating_sub(hom_alt_ex),
        );
    }
    count_packed_row_pure_line_counts_scalar_indices(row, sample_indices)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn count_packed_row_pure_line_counts_selected(
    row: &[u8],
    n_samples: usize,
    sample_indices: &[usize],
) -> (usize, usize) {
    count_packed_row_pure_line_counts_selected_with_excluded(row, n_samples, sample_indices, None)
}

#[inline]
fn load_u64_le_partial(bytes: &[u8], offset: usize) -> u64 {
    let mut v = 0u64;
    if offset >= bytes.len() {
        return v;
    }
    let end = std::cmp::min(bytes.len(), offset.saturating_add(8));
    for (i, &b) in bytes[offset..end].iter().enumerate() {
        v |= (b as u64) << (i * 8);
    }
    v
}

#[inline]
fn store_u64_le_partial(dst: &mut [u8], offset: usize, word: u64) {
    if offset >= dst.len() {
        return;
    }
    let bytes = word.to_le_bytes();
    let end = std::cmp::min(dst.len(), offset.saturating_add(8));
    dst[offset..end].copy_from_slice(&bytes[..(end - offset)]);
}

fn build_bed_spread4_table() -> [u8; 16] {
    let mut table = [0u8; 16];
    for x in 0u8..16u8 {
        let mut out = 0u8;
        out |= (x & 0b0001) << 0;
        out |= (x & 0b0010) << 1;
        out |= (x & 0b0100) << 2;
        out |= (x & 0b1000) << 3;
        table[x as usize] = out;
    }
    table
}

#[inline]
fn and_and_mask_popcount(lhs: &[u64], rhs: &[u64], mask: &[u64]) -> u64 {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert_eq!(lhs.len(), mask.len());
    let mut a0 = 0u64;
    let mut a1 = 0u64;
    let mut a2 = 0u64;
    let mut a3 = 0u64;
    let mut i = 0usize;
    let n = lhs.len();

    while i + 4 <= n {
        a0 += (lhs[i] & rhs[i] & mask[i]).count_ones() as u64;
        a1 += (lhs[i + 1] & rhs[i + 1] & mask[i + 1]).count_ones() as u64;
        a2 += (lhs[i + 2] & rhs[i + 2] & mask[i + 2]).count_ones() as u64;
        a3 += (lhs[i + 3] & rhs[i + 3] & mask[i + 3]).count_ones() as u64;
        i += 4;
    }
    while i < n {
        a0 += (lhs[i] & rhs[i] & mask[i]).count_ones() as u64;
        i += 1;
    }
    a0 + a1 + a2 + a3
}

#[derive(Clone, Copy)]
struct SubsetPackPlanByte {
    word_idx: [u32; 4],
    bit_idx: [u8; 4],
    n_pairs: u8,
}

#[derive(Clone, Copy)]
struct SubsetPackRunOp {
    raw_word_idx: u32,
    start_pair: u8,
    run_len: u8,
}

fn build_subset_run_plan(include_words: &[u64]) -> Vec<SubsetPackRunOp> {
    let mut plan: Vec<SubsetPackRunOp> = Vec::new();
    let mut raw_word_idx = 0u32;

    for &include_word in include_words.iter() {
        let mut include_half = include_word as u32;
        for half_idx in 0..2usize {
            if include_half != 0 {
                let mut run_mask = include_half as u64;
                while run_mask != 0 {
                    let start = run_mask.trailing_zeros();
                    let inv_shifted = (!run_mask) >> start;
                    let mut run_len = inv_shifted.trailing_zeros();
                    let max_len = 32u32.saturating_sub(start);
                    if run_len > max_len {
                        run_len = max_len;
                    }

                    plan.push(SubsetPackRunOp {
                        raw_word_idx,
                        start_pair: start as u8,
                        run_len: run_len as u8,
                    });

                    let clear_through = start.saturating_add(run_len) as usize;
                    if clear_through >= 64 {
                        run_mask = 0u64;
                    } else {
                        run_mask &= !((1u64 << clear_through) - 1u64);
                    }
                }
            }

            raw_word_idx = raw_word_idx.saturating_add(1);
            if half_idx == 0 {
                include_half = (include_word >> 32) as u32;
            }
        }
    }

    plan
}

fn collapse_bed_row_sorted_subset_with_plan(
    src_row: &[u8],
    plan: &[SubsetPackRunOp],
    out_row: &mut [u8],
) -> Result<(), String> {
    let mut cur_output_word = 0u64;
    let mut word_write_halfshift: u32 = 0;
    let mut out_word_idx = 0usize;

    for op in plan.iter().copied() {
        let raw_word = load_u64_le_partial(src_row, (op.raw_word_idx as usize).saturating_mul(8));
        let raw_shifted = raw_word >> ((op.start_pair as u32) * 2);
        let run_len = op.run_len as u32;
        let block_limit = 32u32.saturating_sub(word_write_halfshift);
        cur_output_word |= raw_shifted << (word_write_halfshift * 2);

        if run_len < block_limit {
            word_write_halfshift += run_len;
            if word_write_halfshift < 32 {
                cur_output_word &= (1u64 << (word_write_halfshift * 2)) - 1u64;
            }
        } else {
            store_u64_le_partial(out_row, out_word_idx.saturating_mul(8), cur_output_word);
            out_word_idx = out_word_idx.saturating_add(1);
            word_write_halfshift = run_len - block_limit;
            if word_write_halfshift > 0 {
                cur_output_word = (raw_shifted >> (block_limit * 2))
                    & ((1u64 << (word_write_halfshift * 2)) - 1u64);
            } else {
                cur_output_word = 0u64;
            }
        }
    }

    if word_write_halfshift > 0 {
        store_u64_le_partial(out_row, out_word_idx.saturating_mul(8), cur_output_word);
        out_word_idx = out_word_idx.saturating_add(1);
    }

    let expected_word_ct = out_row.len().div_ceil(8);
    if out_word_idx != expected_word_ct {
        return Err(format!(
            "subset collapse output words mismatch: got {out_word_idx}, expected {expected_word_ct}"
        ));
    }
    Ok(())
}

#[inline]
pub(crate) fn evaluate_packed_row_keep_and_flip(
    n_samples: usize,
    non_missing: usize,
    alt_sum: usize,
    het_count: usize,
    maf_threshold: f32,
    max_missing_rate: f32,
    apply_het_filter: bool,
    het_threshold: f32,
) -> (bool, bool) {
    if n_samples == 0 {
        return (false, false);
    }
    let n_samples_f = n_samples as f64;
    let non_missing_f = non_missing as f64;
    let missing_rate = 1.0 - (non_missing_f / n_samples_f);
    if missing_rate > (max_missing_rate as f64) {
        return (false, false);
    }

    if non_missing == 0 {
        return (maf_threshold <= 0.0, false);
    }

    if apply_het_filter {
        let het_rate = (het_count as f64) / non_missing_f;
        let max_het = het_threshold as f64;
        if het_rate > max_het {
            return (false, false);
        }
    }

    let mut alt_freq = (alt_sum as f64) / (2.0 * non_missing_f);
    let flip = alt_freq > 0.5;
    if flip {
        alt_freq = 1.0 - alt_freq;
    }
    let maf = alt_freq.min(1.0 - alt_freq);
    if maf < (maf_threshold as f64) {
        return (false, false);
    }
    (true, flip)
}

#[allow(dead_code)]
#[inline]
fn evaluate_packed_row_keep_and_flip_pure_line(
    n_samples: usize,
    logic_missing: usize,
    hom_alt: usize,
    maf_threshold: f32,
    max_missing_rate: f32,
) -> (bool, bool) {
    if n_samples == 0 {
        return (false, false);
    }
    let logic_missing = logic_missing.min(n_samples);
    let missing_rate = (logic_missing as f64) / (n_samples as f64);
    if missing_rate > (max_missing_rate as f64) {
        return (false, false);
    }

    let usable_homo = n_samples.saturating_sub(logic_missing);
    if usable_homo == 0 {
        return (false, false);
    }

    let mut alt_freq = (hom_alt as f64) / (usable_homo as f64);
    let flip = alt_freq > 0.5;
    if flip {
        alt_freq = 1.0 - alt_freq;
    }
    let maf = alt_freq.min(1.0 - alt_freq);
    if maf < (maf_threshold as f64) {
        return (false, false);
    }
    (true, flip)
}

#[inline]
pub(crate) fn packed_row_stats_from_counts(
    n_samples: usize,
    non_missing: usize,
    alt_sum: usize,
) -> (f32, f32, f32) {
    let miss = if n_samples > 0 {
        (n_samples.saturating_sub(non_missing) as f32) / (n_samples as f32)
    } else {
        0.0_f32
    };
    if non_missing == 0 {
        return (miss, 0.0_f32, 0.0_f32);
    }
    let p = alt_sum as f64 / (2.0_f64 * non_missing as f64);
    let maf = p.min(1.0_f64 - p) as f32;
    let d = (2.0_f64 * p * (1.0_f64 - p)).sqrt() as f32;
    let std = if d.is_finite() { d } else { 0.0_f32 };
    (miss, maf, std)
}

#[inline]
fn packed_row_stats_from_counts_pure_line(
    n_samples: usize,
    logic_missing: usize,
    hom_alt: usize,
) -> (f32, f32, f32) {
    let logic_missing = logic_missing.min(n_samples);
    let miss = if n_samples > 0 {
        (logic_missing as f32) / (n_samples as f32)
    } else {
        0.0_f32
    };
    let usable_homo = n_samples.saturating_sub(logic_missing);
    if usable_homo == 0 {
        return (miss, 0.0_f32, 0.0_f32);
    }
    let p = (hom_alt as f64) / (usable_homo as f64);
    let maf = p.min(1.0_f64 - p) as f32;
    let d = (2.0_f64 * p * (1.0_f64 - p)).sqrt() as f32;
    let std = if d.is_finite() { d } else { 0.0_f32 };
    (miss, maf, std)
}

fn write_plink_subset_filtered_packed(
    src_prefix: &str,
    out_prefix: &str,
    out_sample_ids: &[String],
    sites: &[core::SiteInfo],
    selected_snp_indices: &[usize],
    sample_indices: &[usize],
    n_source_samples: usize,
    maf_threshold: f32,
    max_missing_rate: f32,
    apply_het_filter: bool,
    het_threshold: f32,
    max_output_sites: Option<usize>,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
) -> Result<usize, String> {
    if out_sample_ids.is_empty() {
        return Err("No samples selected.".to_string());
    }
    if sample_indices.is_empty() {
        return Err("No samples selected.".to_string());
    }
    if out_sample_ids.len() != sample_indices.len() {
        return Err("sample id/index size mismatch".to_string());
    }
    if n_source_samples == 0 {
        return Err("source contains no samples".to_string());
    }

    let src_bed = format!("{src_prefix}.bed");
    let out_bed = format!("{out_prefix}.bed");
    let out_bim = format!("{out_prefix}.bim");
    let out_fam = format!("{out_prefix}.fam");

    if let Some(parent) = Path::new(&out_bed).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }

    write_fam_simple(Path::new(&out_fam), out_sample_ids, None)?;

    let mut rbed = File::open(&src_bed).map_err(|e| format!("{src_bed}: {e}"))?;
    let mut header = [0u8; 3];
    rbed.read_exact(&mut header)
        .map_err(|e| format!("{src_bed}: {e}"))?;
    if header != [0x6C, 0x1B, 0x01] {
        return Err(format!(
            "{src_bed}: unsupported BED header (expect SNP-major 0x6C 0x1B 0x01)"
        ));
    }

    let n_out_samples = out_sample_ids.len();
    let src_bytes_per_snp = (n_source_samples + 3) / 4;
    let out_bytes_per_snp = (n_out_samples + 3) / 4;
    let expected_payload = sites.len().saturating_mul(src_bytes_per_snp);
    let bed_size_u64 = rbed
        .metadata()
        .map_err(|e| format!("{src_bed}: {e}"))?
        .len();
    let bed_size = usize::try_from(bed_size_u64)
        .map_err(|_| format!("{src_bed}: file too large for this platform"))?;
    if bed_size < 3 {
        return Err(format!("{src_bed}: invalid BED header"));
    }
    let payload = bed_size - 3;
    if payload != expected_payload {
        return Err(format!(
            "{src_bed}: payload size mismatch, got {payload}, expected {expected_payload} (n_snps={}, n_samples={n_source_samples})",
            sites.len()
        ));
    }

    let mut wbed = BufWriter::with_capacity(
        8 * 1024 * 1024,
        File::create(&out_bed).map_err(|e| format!("{out_bed}: {e}"))?,
    );
    wbed.write_all(&header)
        .map_err(|e| format!("{out_bed}: {e}"))?;
    let mut wbim = BufWriter::with_capacity(
        4 * 1024 * 1024,
        File::create(&out_bim).map_err(|e| format!("{out_bim}: {e}"))?,
    );

    if selected_snp_indices.iter().any(|&idx| idx >= sites.len()) {
        return Err("selected SNP index out of range".to_string());
    }
    if selected_snp_indices.windows(2).any(|w| w[0] >= w[1]) {
        return Err("selected SNP indices must be strictly increasing".to_string());
    }
    if sample_indices.iter().any(|&idx| idx >= n_source_samples) {
        return Err("sample index out of range".to_string());
    }

    let prefix_identity = sample_indices_are_identity(sample_indices);
    let full_identity = n_out_samples == n_source_samples && prefix_identity;
    let tail_pairs = n_out_samples & 3;
    let full_bytes_out = if tail_pairs == 0 {
        out_bytes_per_snp
    } else {
        out_bytes_per_snp.saturating_sub(1)
    };
    let tail_mask: u8 = if tail_pairs == 0 {
        0xFF
    } else {
        ((1u16 << (tail_pairs * 2)) - 1) as u8
    };

    let mut kept = 0usize;
    let mut written = 0usize;
    let mut out_row = vec![0u8; out_bytes_per_snp];
    let total_selected = selected_snp_indices.len();
    let notify_step = if progress_every == 0 {
        (total_selected / 200).max(1)
    } else {
        progress_every.max(1)
    };
    let mut last_notified = 0usize;
    if let Some(cb) = progress_callback {
        Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            cb.call1(py2, (0usize, total_selected))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }
    let mut notify_progress = |done: usize| -> Result<(), String> {
        let done_clamped = done.min(total_selected);
        if done_clamped < last_notified.saturating_add(notify_step)
            && done_clamped != total_selected
        {
            return Ok(());
        }
        last_notified = done_clamped;
        Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            if let Some(cb) = progress_callback {
                cb.call1(py2, (done_clamped, total_selected))?;
            }
            Ok(())
        })
        .map_err(|e| e.to_string())
    };
    let mut processed = 0usize;

    if prefix_identity {
        let block_target_bytes = 8 * 1024 * 1024usize;
        let block_rows = std::cmp::max(
            64usize,
            std::cmp::min(
                4096usize,
                std::cmp::max(
                    1usize,
                    block_target_bytes / std::cmp::max(1usize, src_bytes_per_snp),
                ),
            ),
        );

        let mut prev_idx_opt: Option<usize> = None;
        for snp_chunk in selected_snp_indices.chunks(block_rows) {
            let rows_n = snp_chunk.len();
            let mut block = vec![0u8; rows_n.saturating_mul(src_bytes_per_snp)];

            for (r, &snp_idx) in snp_chunk.iter().enumerate() {
                match prev_idx_opt {
                    None => {
                        let off = 3usize
                            .checked_add(
                                snp_idx
                                    .checked_mul(src_bytes_per_snp)
                                    .ok_or_else(|| "BED row offset overflow".to_string())?,
                            )
                            .ok_or_else(|| "BED row offset overflow".to_string())?;
                        rbed.seek(SeekFrom::Start(off as u64))
                            .map_err(|e| format!("{src_bed}: {e}"))?;
                    }
                    Some(prev) => {
                        if snp_idx <= prev {
                            return Err(
                                "selected SNP indices must be strictly increasing".to_string()
                            );
                        }
                        let skip_rows = snp_idx - prev - 1;
                        if skip_rows > 0 {
                            let skip_bytes = skip_rows
                                .checked_mul(src_bytes_per_snp)
                                .ok_or_else(|| "BED seek offset overflow".to_string())?;
                            rbed.seek(SeekFrom::Current(skip_bytes as i64))
                                .map_err(|e| format!("{src_bed}: {e}"))?;
                        }
                    }
                }
                let st = r.saturating_mul(src_bytes_per_snp);
                let ed = st.saturating_add(src_bytes_per_snp);
                rbed.read_exact(&mut block[st..ed])
                    .map_err(|e| format!("{src_bed}: {e}"))?;
                prev_idx_opt = Some(snp_idx);
            }

            let decisions: Vec<(bool, bool)> = block
                .par_chunks(src_bytes_per_snp)
                .map(|row| {
                    let (missing, het, hom_alt) = count_packed_row_counts(row, n_out_samples);
                    let non_missing = n_out_samples.saturating_sub(missing);
                    let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                    let het_count = if apply_het_filter { het } else { 0usize };
                    evaluate_packed_row_keep_and_flip(
                        n_out_samples,
                        non_missing,
                        alt_sum,
                        het_count,
                        maf_threshold,
                        max_missing_rate,
                        apply_het_filter,
                        het_threshold,
                    )
                })
                .collect();

            for ((&snp_idx, &(keep, flip)), row) in snp_chunk
                .iter()
                .zip(decisions.iter())
                .zip(block.chunks(src_bytes_per_snp))
            {
                if !keep {
                    continue;
                }
                kept = kept.saturating_add(1);
                let site = &sites[snp_idx];
                let write_this = max_output_sites.map(|lim| written < lim).unwrap_or(true);
                if write_this {
                    write_bim_site_line_with_flip(&mut wbim, site, flip)?;

                    if !flip {
                        if tail_pairs == 0 {
                            if full_identity {
                                wbed.write_all(row).map_err(|e| format!("{out_bed}: {e}"))?;
                            } else {
                                wbed.write_all(&row[..out_bytes_per_snp])
                                    .map_err(|e| format!("{out_bed}: {e}"))?;
                            }
                        } else {
                            if full_bytes_out > 0 {
                                wbed.write_all(&row[..full_bytes_out])
                                    .map_err(|e| format!("{out_bed}: {e}"))?;
                            }
                            let last = row[full_bytes_out] & tail_mask;
                            wbed.write_all(&[last])
                                .map_err(|e| format!("{out_bed}: {e}"))?;
                        }
                    } else {
                        if full_bytes_out > 0 {
                            flip_bed_bytes_into(
                                &row[..full_bytes_out],
                                &mut out_row[..full_bytes_out],
                            );
                            wbed.write_all(&out_row[..full_bytes_out])
                                .map_err(|e| format!("{out_bed}: {e}"))?;
                        }
                        if tail_pairs > 0 {
                            let last = flip_bed_byte_fast(row[full_bytes_out]) & tail_mask;
                            wbed.write_all(&[last])
                                .map_err(|e| format!("{out_bed}: {e}"))?;
                        }
                    }
                    written = written.saturating_add(1);
                }
            }
            processed = processed.saturating_add(rows_n);
            notify_progress(processed)?;
        }
    } else {
        let n_words_src = n_source_samples.div_ceil(64);
        let last_word_src = n_words_src.saturating_sub(1);
        let tail_bits_src_word = n_source_samples & 63;
        let tail_mask_src_word: u64 = if tail_bits_src_word == 0 {
            u64::MAX
        } else {
            (1u64 << tail_bits_src_word) - 1u64
        };
        let sample_indices_strictly_increasing = sample_indices.windows(2).all(|w| w[0] < w[1]);
        let low_high_lut = if sample_indices_strictly_increasing {
            None
        } else {
            Some(build_bed_low_high_nibble_tables())
        };
        let spread4 = if sample_indices_strictly_increasing {
            None
        } else {
            Some(build_bed_spread4_table())
        };

        let mut selected_mask_words = vec![0u64; n_words_src];
        for &sidx in sample_indices.iter() {
            let w = sidx >> 6;
            let b = sidx & 63;
            selected_mask_words[w] |= 1u64 << b;
        }

        // Fast path for strictly-increasing sample subsets:
        // read BED rows in blocks, collapse/evaluate rows in parallel, then
        // serialize writes in SNP order.
        if sample_indices_strictly_increasing {
            let block_target_bytes = 8 * 1024 * 1024usize;
            let block_rows = std::cmp::max(
                64usize,
                std::cmp::min(
                    4096usize,
                    std::cmp::max(
                        1usize,
                        block_target_bytes / std::cmp::max(1usize, src_bytes_per_snp),
                    ),
                ),
            );
            let subset_run_plan = build_subset_run_plan(selected_mask_words.as_slice());
            let mut prev_idx_opt: Option<usize> = None;

            for snp_chunk in selected_snp_indices.chunks(block_rows) {
                let rows_n = snp_chunk.len();
                let mut block = vec![0u8; rows_n.saturating_mul(src_bytes_per_snp)];

                for (r, &snp_idx) in snp_chunk.iter().enumerate() {
                    match prev_idx_opt {
                        None => {
                            let off = 3usize
                                .checked_add(
                                    snp_idx
                                        .checked_mul(src_bytes_per_snp)
                                        .ok_or_else(|| "BED row offset overflow".to_string())?,
                                )
                                .ok_or_else(|| "BED row offset overflow".to_string())?;
                            rbed.seek(SeekFrom::Start(off as u64))
                                .map_err(|e| format!("{src_bed}: {e}"))?;
                        }
                        Some(prev) => {
                            if snp_idx <= prev {
                                return Err(
                                    "selected SNP indices must be strictly increasing".to_string()
                                );
                            }
                            let skip_rows = snp_idx - prev - 1;
                            if skip_rows > 0 {
                                let skip_bytes = skip_rows
                                    .checked_mul(src_bytes_per_snp)
                                    .ok_or_else(|| "BED seek offset overflow".to_string())?;
                                rbed.seek(SeekFrom::Current(skip_bytes as i64))
                                    .map_err(|e| format!("{src_bed}: {e}"))?;
                            }
                        }
                    }
                    let st = r.saturating_mul(src_bytes_per_snp);
                    let ed = st.saturating_add(src_bytes_per_snp);
                    rbed.read_exact(&mut block[st..ed])
                        .map_err(|e| format!("{src_bed}: {e}"))?;
                    prev_idx_opt = Some(snp_idx);
                }

                let decisions: Vec<Result<(bool, bool, Vec<u8>), String>> = block
                    .par_chunks(src_bytes_per_snp)
                    .map(|src_row| {
                        let mut collapsed = vec![0u8; out_bytes_per_snp];
                        collapse_bed_row_sorted_subset_with_plan(
                            src_row,
                            subset_run_plan.as_slice(),
                            collapsed.as_mut_slice(),
                        )?;
                        let (missing, het, hom_alt) =
                            count_packed_row_counts(collapsed.as_slice(), n_out_samples);
                        let non_missing = n_out_samples.saturating_sub(missing);
                        let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                        let het_count = if apply_het_filter { het } else { 0usize };
                        let (keep, flip) = evaluate_packed_row_keep_and_flip(
                            n_out_samples,
                            non_missing,
                            alt_sum,
                            het_count,
                            maf_threshold,
                            max_missing_rate,
                            apply_het_filter,
                            het_threshold,
                        );
                        Ok((keep, flip, collapsed))
                    })
                    .collect();

                for (&snp_idx, decision) in snp_chunk.iter().zip(decisions.into_iter()) {
                    let (keep, flip, mut collapsed) = decision?;
                    if !keep {
                        continue;
                    }
                    kept = kept.saturating_add(1);
                    let write_this = max_output_sites.map(|lim| written < lim).unwrap_or(true);
                    if !write_this {
                        continue;
                    }
                    let site = &sites[snp_idx];
                    write_bim_site_line_with_flip(&mut wbim, site, flip)?;

                    if flip {
                        if full_bytes_out > 0 {
                            flip_bed_bytes_in_place(&mut collapsed[..full_bytes_out]);
                        }
                        if tail_pairs > 0 {
                            collapsed[full_bytes_out] =
                                flip_bed_byte_fast(collapsed[full_bytes_out]);
                        }
                    }
                    if tail_pairs > 0 {
                        collapsed[full_bytes_out] &= tail_mask;
                    }
                    wbed.write_all(&collapsed)
                        .map_err(|e| format!("{out_bed}: {e}"))?;
                    written = written.saturating_add(1);
                }
                processed = processed.saturating_add(rows_n);
                notify_progress(processed)?;
            }
            wbed.flush().map_err(|e| format!("{out_bed}: {e}"))?;
            wbim.flush().map_err(|e| format!("{out_bim}: {e}"))?;
            return Ok(kept);
        }

        let mut pack_plan: Vec<SubsetPackPlanByte> = vec![
            SubsetPackPlanByte {
                word_idx: [0u32; 4],
                bit_idx: [0u8; 4],
                n_pairs: 0u8,
            };
            out_bytes_per_snp
        ];
        for (out_i, &sidx) in sample_indices.iter().enumerate() {
            let out_b = out_i >> 2;
            let lane = out_i & 3;
            let plan = &mut pack_plan[out_b];
            plan.word_idx[lane] = (sidx >> 6) as u32;
            plan.bit_idx[lane] = (sidx & 63) as u8;
            let lane_pairs = (lane as u8) + 1;
            if plan.n_pairs < lane_pairs {
                plan.n_pairs = lane_pairs;
            }
        }

        let mut src_row = vec![0u8; src_bytes_per_snp];
        let mut low_words = vec![0u64; n_words_src];
        let mut high_words = vec![0u64; n_words_src];

        for (k, &snp_idx) in selected_snp_indices.iter().enumerate() {
            if k == 0 {
                let off = 3usize
                    .checked_add(
                        snp_idx
                            .checked_mul(src_bytes_per_snp)
                            .ok_or_else(|| "BED row offset overflow".to_string())?,
                    )
                    .ok_or_else(|| "BED row offset overflow".to_string())?;
                rbed.seek(SeekFrom::Start(off as u64))
                    .map_err(|e| format!("{src_bed}: {e}"))?;
            } else {
                let prev = selected_snp_indices[k - 1];
                if snp_idx <= prev {
                    return Err("selected SNP indices must be strictly increasing".to_string());
                }
                let skip_rows = snp_idx - prev - 1;
                if skip_rows > 0 {
                    let skip_bytes = skip_rows
                        .checked_mul(src_bytes_per_snp)
                        .ok_or_else(|| "BED seek offset overflow".to_string())?;
                    rbed.seek(SeekFrom::Current(skip_bytes as i64))
                        .map_err(|e| format!("{src_bed}: {e}"))?;
                }
            }
            rbed.read_exact(&mut src_row)
                .map_err(|e| format!("{src_bed}: {e}"))?;

            let (low_lut, high_lut) = low_high_lut
                .as_ref()
                .ok_or_else(|| "subset low/high LUT missing".to_string())?;
            for w in 0..n_words_src {
                let base = w.saturating_mul(16);
                let mut lo_w = 0u64;
                let mut hi_w = 0u64;
                for j in 0..16usize {
                    let bi = base + j;
                    if bi >= src_bytes_per_snp {
                        break;
                    }
                    let b = src_row[bi] as usize;
                    lo_w |= (low_lut[b] as u64) << (j * 4);
                    hi_w |= (high_lut[b] as u64) << (j * 4);
                }
                low_words[w] = lo_w;
                high_words[w] = hi_w;
            }
            if n_words_src > 0 && tail_bits_src_word != 0 {
                low_words[last_word_src] &= tail_mask_src_word;
                high_words[last_word_src] &= tail_mask_src_word;
            }

            let low_pop =
                and_popcount(low_words.as_slice(), selected_mask_words.as_slice()) as usize;
            let high_pop =
                and_popcount(high_words.as_slice(), selected_mask_words.as_slice()) as usize;
            let hom_alt = and_and_mask_popcount(
                low_words.as_slice(),
                high_words.as_slice(),
                selected_mask_words.as_slice(),
            ) as usize;
            let missing = low_pop.saturating_sub(hom_alt);
            let het = high_pop.saturating_sub(hom_alt);
            let non_missing = n_out_samples.saturating_sub(missing);
            let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
            let het_count = if apply_het_filter { het } else { 0usize };
            let (keep, flip) = evaluate_packed_row_keep_and_flip(
                n_out_samples,
                non_missing,
                alt_sum,
                het_count,
                maf_threshold,
                max_missing_rate,
                apply_het_filter,
                het_threshold,
            );
            processed = processed.saturating_add(1);
            if !keep {
                notify_progress(processed)?;
                continue;
            }

            kept = kept.saturating_add(1);
            let write_this = max_output_sites.map(|lim| written < lim).unwrap_or(true);
            if !write_this {
                notify_progress(processed)?;
                continue;
            }

            let site = &sites[snp_idx];
            write_bim_site_line_with_flip(&mut wbim, site, flip)?;

            let spread4 = spread4
                .as_ref()
                .ok_or_else(|| "subset spread table missing".to_string())?;
            for (out_b, plan) in pack_plan.iter().enumerate() {
                let mut lo_nib = 0u8;
                let mut hi_nib = 0u8;
                for lane in 0..(plan.n_pairs as usize) {
                    let w = plan.word_idx[lane] as usize;
                    let bit = plan.bit_idx[lane];
                    lo_nib |= (((low_words[w] >> bit) & 1u64) as u8) << lane;
                    hi_nib |= (((high_words[w] >> bit) & 1u64) as u8) << lane;
                }
                let mut packed =
                    spread4[lo_nib as usize] | (spread4[hi_nib as usize].wrapping_shl(1u32));
                if flip {
                    packed = flip_bed_byte_fast(packed);
                }
                out_row[out_b] = packed;
            }
            if tail_pairs > 0 {
                out_row[full_bytes_out] &= tail_mask;
            }
            wbed.write_all(&out_row)
                .map_err(|e| format!("{out_bed}: {e}"))?;
            written = written.saturating_add(1);
            notify_progress(processed)?;
        }
    }

    wbed.flush().map_err(|e| format!("{out_bed}: {e}"))?;
    wbim.flush().map_err(|e| format!("{out_bim}: {e}"))?;
    Ok(kept)
}

#[inline]
fn plink2bits_from_g_f32(g: f32) -> u8 {
    if !g.is_finite() || g < 0.0 {
        return 0b01;
    }
    match g.round().clamp(0.0, 2.0) as i32 {
        0 => 0b00,
        1 => 0b10,
        _ => 0b11,
    }
}

#[pyfunction]
#[pyo3(signature = (
    src_prefix,
    out_prefix,
    maf_threshold=0.0,
    max_missing_rate=1.0,
    model=None,
    het_threshold=None,
))]
pub fn bed_filter_stream_to_plink_rust(
    py: Python<'_>,
    src_prefix: String,
    out_prefix: String,
    maf_threshold: f32,
    max_missing_rate: f32,
    model: Option<String>,
    het_threshold: Option<f32>,
) -> PyResult<(usize, usize, usize)> {
    let src = normalize_plink_prefix_local(&src_prefix);
    let out = normalize_plink_prefix_local(&out_prefix);
    if src.is_empty() || out.is_empty() {
        return Err(PyValueError::new_err(
            "src_prefix/out_prefix must not be empty",
        ));
    }
    if !(0.0..=0.5).contains(&maf_threshold) {
        return Err(PyValueError::new_err(
            "maf_threshold must be within [0, 0.5]",
        ));
    }
    if !(0.0..=1.0).contains(&max_missing_rate) {
        return Err(PyValueError::new_err(
            "max_missing_rate must be within [0, 1.0]",
        ));
    }

    let model_key = model
        .unwrap_or_else(|| "add".to_string())
        .to_ascii_lowercase();
    if !matches!(model_key.as_str(), "add" | "dom" | "rec" | "het") {
        return Err(PyValueError::new_err(
            "model must be one of: add, dom, rec, het",
        ));
    }
    let het = het_threshold.unwrap_or(0.02);
    if !(0.0..=1.0).contains(&het) {
        return Err(PyValueError::new_err(
            "het_threshold must be within [0, 1.0]",
        ));
    }
    let apply_het_filter = model_key != "add";

    let outv = py
        .detach(move || -> Result<(usize, usize, usize), String> {
            let mut it = BedSnpIter::new_with_fill(&src, 0.0, 1.0, false, false, het)?;
            let n_samples = it.n_samples();
            if n_samples == 0 {
                return Err("source contains no samples".to_string());
            }

            let out_bed = format!("{out}.bed");
            let out_bim = format!("{out}.bim");
            let out_fam = format!("{out}.fam");
            if let Some(parent) = Path::new(&out_bed).parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
            }

            write_fam_simple(Path::new(&out_fam), it.samples.as_slice(), None)?;

            let mut wbed = BufWriter::with_capacity(
                8 * 1024 * 1024,
                File::create(&out_bed).map_err(|e| format!("{out_bed}: {e}"))?,
            );
            wbed.write_all(&[0x6C, 0x1B, 0x01])
                .map_err(|e| format!("{out_bed}: {e}"))?;
            let mut wbim = BufWriter::with_capacity(
                4 * 1024 * 1024,
                File::create(&out_bim).map_err(|e| format!("{out_bim}: {e}"))?,
            );

            let bytes_per_snp = n_samples.div_ceil(4);
            let mut row_buf = vec![0u8; bytes_per_snp];
            let mut n_scanned = 0usize;
            let mut n_kept = 0usize;

            while let Some((mut row, mut site)) = it.next_snp_raw() {
                n_scanned = n_scanned.saturating_add(1);
                let keep = core::process_snp_row(
                    &mut row,
                    &mut site.ref_allele,
                    &mut site.alt_allele,
                    maf_threshold,
                    max_missing_rate,
                    false,
                    apply_het_filter,
                    het,
                );
                if !keep {
                    continue;
                }

                write_bim_site_line(&mut wbim, &site)?;
                let mut i = 0usize;
                for b in 0..bytes_per_snp {
                    let mut packed = 0u8;
                    for lane in 0..4usize {
                        let si = i + lane;
                        let code = if si < n_samples {
                            plink2bits_from_g_f32(row[si])
                        } else {
                            0b01
                        };
                        packed |= code << (lane * 2);
                    }
                    row_buf[b] = packed;
                    i += 4;
                }
                wbed.write_all(row_buf.as_slice())
                    .map_err(|e| format!("{out_bed}: {e}"))?;
                n_kept = n_kept.saturating_add(1);
            }

            wbed.flush().map_err(|e| format!("{out_bed}: {e}"))?;
            wbim.flush().map_err(|e| format!("{out_bim}: {e}"))?;
            Ok((n_kept, n_scanned, n_samples))
        })
        .map_err(PyRuntimeError::new_err)?;

    Ok(outv)
}

#[pyfunction]
#[pyo3(signature = (
    src_prefix,
    out_prefix,
    maf_threshold=0.0,
    max_missing_rate=1.0,
    het_threshold=0.0,
    block_rows=8192,
    parallel=true,
))]
pub fn bed_mmap_filter_to_plink_rust(
    py: Python<'_>,
    src_prefix: String,
    out_prefix: String,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    block_rows: usize,
    parallel: bool,
) -> PyResult<(usize, usize, usize, usize)> {
    let src = normalize_plink_prefix_local(&src_prefix);
    let out = normalize_plink_prefix_local(&out_prefix);
    if src.is_empty() || out.is_empty() {
        return Err(PyValueError::new_err(
            "src_prefix/out_prefix must not be empty",
        ));
    }
    if !(0.0..=0.5).contains(&maf_threshold) {
        return Err(PyValueError::new_err(
            "maf_threshold must be within [0, 0.5]",
        ));
    }
    if !(0.0..=1.0).contains(&max_missing_rate) {
        return Err(PyValueError::new_err(
            "max_missing_rate must be within [0, 1.0]",
        ));
    }
    if !(0.0..=1.0).contains(&het_threshold) {
        return Err(PyValueError::new_err(
            "het_threshold must be within [0, 1.0]",
        ));
    }

    let outv = py
        .detach(move || -> Result<(usize, usize, usize, usize), String> {
            let engine = BedMmapEngine::open(&src)?;
            let sites = core::read_bim(&src)?;
            if sites.len() != engine.n_snps {
                return Err(format!(
                    "BED/BIM SNP count mismatch: bed={}, bim={}",
                    engine.n_snps,
                    sites.len()
                ));
            }
            let sample_ids = core::read_fam(&src)?;
            if sample_ids.len() != engine.n_samples {
                return Err(format!(
                    "BED/FAM sample count mismatch: bed={}, fam={}",
                    engine.n_samples,
                    sample_ids.len()
                ));
            }

            let out_bed = format!("{out}.bed");
            let out_bim = format!("{out}.bim");
            let out_fam = format!("{out}.fam");
            if let Some(parent) = Path::new(&out_bed).parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
            }
            write_fam_simple(Path::new(&out_fam), sample_ids.as_slice(), None)?;

            let mut wbed = BufWriter::with_capacity(
                8 * 1024 * 1024,
                File::create(&out_bed).map_err(|e| format!("{out_bed}: {e}"))?,
            );
            wbed.write_all(&[0x6C, 0x1B, 0x01])
                .map_err(|e| format!("{out_bed}: {e}"))?;
            let mut wbim = BufWriter::with_capacity(
                4 * 1024 * 1024,
                File::create(&out_bim).map_err(|e| format!("{out_bim}: {e}"))?,
            );

            let n_samples = engine.n_samples;
            let bytes_per_snp = engine.bytes_per_snp;
            let tail_pairs = n_samples & 3;
            let full_bytes_out = if tail_pairs == 0 {
                bytes_per_snp
            } else {
                bytes_per_snp.saturating_sub(1)
            };
            let tail_mask: u8 = if tail_pairs == 0 {
                0xFF
            } else {
                ((1u16 << (tail_pairs * 2)) - 1) as u8
            };
            let apply_het_filter = het_threshold > 0.0_f32;
            let mut out_row = vec![0u8; bytes_per_snp];

            let blk = std::cmp::max(1usize, block_rows);
            let mut n_scanned = 0usize;
            let mut n_blocks = 0usize;
            let mut n_kept = 0usize;
            let rows_per_task = 512usize;
            let pretouch_pages = std::env::var("JANUSX_BED_MMAP_PRETOUCH")
                .ok()
                .map(|v| {
                    let k = v.trim().to_ascii_lowercase();
                    matches!(k.as_str(), "1" | "true" | "yes" | "on")
                })
                .unwrap_or(false);

            while n_scanned < engine.n_snps {
                let take = std::cmp::min(blk, engine.n_snps - n_scanned);
                let (block_bytes, got_rows) = engine.get_rows_slice(n_scanned, take)?;
                if got_rows != take {
                    return Err(format!(
                        "BED scan rows mismatch: got_rows={got_rows}, expected={take}"
                    ));
                }

                if pretouch_pages && parallel {
                    // Optional dummy read to fault pages in before parallel scan.
                    // This can help some kernels/filesystems but may hurt others.
                    let mut dummy = 0u8;
                    let mut pg = 0usize;
                    while pg < block_bytes.len() {
                        dummy ^= block_bytes[pg];
                        pg = pg.saturating_add(4096);
                    }
                    std::hint::black_box(dummy);
                }

                let eval_row = |row: &[u8]| {
                    let (missing, het, hom_alt) = count_packed_row_counts(row, n_samples);
                    let non_missing = n_samples.saturating_sub(missing);
                    let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                    let het_count = if apply_het_filter { het } else { 0usize };
                    evaluate_packed_row_keep_and_flip(
                        n_samples,
                        non_missing,
                        alt_sum,
                        het_count,
                        maf_threshold,
                        max_missing_rate,
                        apply_het_filter,
                        het_threshold,
                    )
                };

                let mut decisions: Vec<(bool, bool)> = vec![(false, false); take];
                if parallel && take >= 128 {
                    decisions
                        .par_chunks_mut(rows_per_task)
                        .enumerate()
                        .for_each(|(chunk_idx, out_chunk)| {
                            let row_start = chunk_idx.saturating_mul(rows_per_task);
                            let byte_start = row_start.saturating_mul(bytes_per_snp);
                            let byte_end = byte_start
                                .saturating_add(out_chunk.len().saturating_mul(bytes_per_snp));
                            let super_row = &block_bytes[byte_start..byte_end];
                            for (dst, row) in
                                out_chunk.iter_mut().zip(super_row.chunks(bytes_per_snp))
                            {
                                *dst = eval_row(row);
                            }
                        });
                } else {
                    for (dst, row) in decisions.iter_mut().zip(block_bytes.chunks(bytes_per_snp)) {
                        *dst = eval_row(row);
                    }
                }
                if decisions.len() != take {
                    return Err(format!(
                        "internal decision row mismatch: got {}, expected {}",
                        decisions.len(),
                        take
                    ));
                }

                for row_idx in 0..take {
                    let snp_idx = n_scanned + row_idx;
                    let row_start = row_idx * bytes_per_snp;
                    let row_end = row_start + bytes_per_snp;
                    let row = &block_bytes[row_start..row_end];
                    let (keep, flip) = decisions[row_idx];
                    if !keep {
                        continue;
                    }

                    let site = &sites[snp_idx];
                    write_bim_site_line_with_flip(&mut wbim, site, flip)?;

                    if !flip {
                        if tail_pairs == 0 {
                            wbed.write_all(row).map_err(|e| format!("{out_bed}: {e}"))?;
                        } else {
                            if full_bytes_out > 0 {
                                out_row[..full_bytes_out].copy_from_slice(&row[..full_bytes_out]);
                            }
                            out_row[full_bytes_out] = row[full_bytes_out] & tail_mask;
                            wbed.write_all(&out_row)
                                .map_err(|e| format!("{out_bed}: {e}"))?;
                        }
                    } else {
                        if full_bytes_out > 0 {
                            flip_bed_bytes_into(
                                &row[..full_bytes_out],
                                &mut out_row[..full_bytes_out],
                            );
                        }
                        if tail_pairs > 0 {
                            out_row[full_bytes_out] =
                                flip_bed_byte_fast(row[full_bytes_out]) & tail_mask;
                            wbed.write_all(&out_row)
                                .map_err(|e| format!("{out_bed}: {e}"))?;
                        } else {
                            wbed.write_all(&out_row[..full_bytes_out])
                                .map_err(|e| format!("{out_bed}: {e}"))?;
                        }
                    }
                    n_kept = n_kept.saturating_add(1);
                }

                n_scanned = n_scanned.saturating_add(take);
                n_blocks = n_blocks.saturating_add(1);
            }

            wbed.flush().map_err(|e| format!("{out_bed}: {e}"))?;
            wbim.flush().map_err(|e| format!("{out_bim}: {e}"))?;
            Ok((n_kept, n_scanned, n_samples, n_blocks))
        })
        .map_err(PyRuntimeError::new_err)?;

    Ok(outv)
}

#[pyfunction]
#[pyo3(signature = (
    src_prefix,
    out_prefix,
    maf_threshold=0.0,
    max_missing_rate=1.0,
    fill_missing=false,
    model=None,
    het_threshold=None,
    sample_ids=None,
    snp_sites=None,
    bim_range=None,
    chr_keys=None,
    bp_min=None,
    bp_max=None,
    ranges=None,
    progress_callback=None,
    progress_every=0,
    max_output_sites=None,
))]
pub fn bed_filter_to_plink_rust(
    py: Python<'_>,
    src_prefix: String,
    out_prefix: String,
    maf_threshold: f32,
    max_missing_rate: f32,
    fill_missing: bool,
    model: Option<String>,
    het_threshold: Option<f32>,
    sample_ids: Option<Vec<String>>,
    snp_sites: Option<Vec<(String, i32)>>,
    bim_range: Option<(String, i32, i32)>,
    chr_keys: Option<Vec<String>>,
    bp_min: Option<i32>,
    bp_max: Option<i32>,
    ranges: Option<Vec<(String, i32, i32)>>,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    max_output_sites: Option<usize>,
) -> PyResult<(usize, usize, usize)> {
    let src = normalize_plink_prefix_local(&src_prefix);
    let out = normalize_plink_prefix_local(&out_prefix);
    if src.is_empty() || out.is_empty() {
        return Err(PyValueError::new_err(
            "src_prefix/out_prefix must not be empty",
        ));
    }
    if !(0.0..=0.5).contains(&maf_threshold) {
        return Err(PyValueError::new_err(
            "maf_threshold must be within [0, 0.5]",
        ));
    }
    if !(0.0..=1.0).contains(&max_missing_rate) {
        return Err(PyValueError::new_err(
            "max_missing_rate must be within [0, 1.0]",
        ));
    }

    let model_key = model
        .unwrap_or_else(|| "add".to_string())
        .to_ascii_lowercase();
    if !matches!(model_key.as_str(), "add" | "dom" | "rec" | "het") {
        return Err(PyValueError::new_err(
            "model must be one of: add, dom, rec, het",
        ));
    }
    let het = het_threshold.unwrap_or(0.02);
    if !(0.0..=1.0).contains(&het) {
        return Err(PyValueError::new_err(
            "het_threshold must be within [0, 1.0]",
        ));
    }
    if fill_missing {
        return Err(PyValueError::new_err(
            "bed_filter_to_plink_rust is a 2-bit native filter engine and does not support fill_missing=true",
        ));
    }
    let apply_het_filter = model_key != "add";
    let outv = py
        .detach(move || -> Result<(usize, usize, usize), String> {
            let it = BedSnpIter::new_with_fill(&src, 0.0, 1.0, false, false, het)?;
            let (sample_indices, out_sample_ids) =
                build_sample_selection(&it.samples, sample_ids, None)?;
            if out_sample_ids.is_empty() {
                return Err("No samples selected.".to_string());
            }
            let n_out_samples = out_sample_ids.len();
            let site_filter =
                SiteFilterExpr::from_parts(snp_sites, bim_range, chr_keys, bp_min, bp_max, ranges)?;
            let mut selected_snp_indices: Vec<usize> = Vec::new();
            if site_filter.active() {
                selected_snp_indices.reserve(it.sites.len());
                for (snp_idx, site) in it.sites.iter().enumerate() {
                    if site_filter.keep_site(site) {
                        selected_snp_indices.push(snp_idx);
                    }
                }
            } else {
                selected_snp_indices.extend(0..it.sites.len());
            }
            let n_scanned: usize = selected_snp_indices.len();
            let n_kept = write_plink_subset_filtered_packed(
                &src,
                &out,
                &out_sample_ids,
                &it.sites,
                &selected_snp_indices,
                &sample_indices,
                it.n_samples(),
                maf_threshold,
                max_missing_rate,
                apply_het_filter,
                het,
                max_output_sites,
                progress_callback.as_ref(),
                progress_every,
            )?;
            Ok((n_kept, n_scanned, n_out_samples))
        })
        .map_err(PyRuntimeError::new_err)?;

    Ok(outv)
}

// -------- BedChunkReader --------
#[pyclass]
pub struct BedChunkReader {
    it: BedSnpIter,
    snp_indices: Option<Vec<usize>>,
    snp_pos: usize,
    sample_indices: Vec<usize>,
    sample_ids: Vec<String>,
    maf: f32,
    miss: f32,
    fill_missing: bool,
    apply_het_filter: bool,
    het_threshold: f32,
}

#[derive(Clone, Copy)]
enum BedPreparedCodingMode {
    Add,
    Dom,
    Rec,
    Het,
}

#[inline]
fn bed_apply_coding_value(v: f32, mode: BedPreparedCodingMode) -> f32 {
    match mode {
        BedPreparedCodingMode::Add => v,
        BedPreparedCodingMode::Dom => {
            if (v - 1.0_f32).abs() <= 1e-6_f32 || (v - 2.0_f32).abs() <= 1e-6_f32 {
                1.0_f32
            } else {
                0.0_f32
            }
        }
        BedPreparedCodingMode::Rec => {
            if (v - 2.0_f32).abs() <= 1e-6_f32 {
                1.0_f32
            } else {
                0.0_f32
            }
        }
        BedPreparedCodingMode::Het => {
            if (v - 1.0_f32).abs() <= 1e-6_f32 {
                1.0_f32
            } else {
                0.0_f32
            }
        }
    }
}

#[pymethods]
impl BedChunkReader {
    #[new]
    #[pyo3(signature = (
        prefix,
        maf_threshold=None,
        max_missing_rate=None,
        fill_missing=None,
        snp_range=None,
        snp_indices=None,
        bim_range=None,
        snp_sites=None,
        sample_ids=None,
        sample_indices=None,
        mmap_window_mb=None,
        chr_keys=None,
        bp_min=None,
        bp_max=None,
        ranges=None,
        model=None,
        het_threshold=None,
    ))]
    fn new(
        prefix: String,
        maf_threshold: Option<f32>,
        max_missing_rate: Option<f32>,
        fill_missing: Option<bool>,
        snp_range: Option<(usize, usize)>,
        snp_indices: Option<Vec<usize>>,
        bim_range: Option<(String, i32, i32)>,
        snp_sites: Option<Vec<(String, i32)>>,
        sample_ids: Option<Vec<String>>,
        sample_indices: Option<Vec<usize>>,
        mmap_window_mb: Option<usize>,
        chr_keys: Option<Vec<String>>,
        bp_min: Option<i32>,
        bp_max: Option<i32>,
        ranges: Option<Vec<(String, i32, i32)>>,
        model: Option<String>,
        het_threshold: Option<f32>,
    ) -> PyResult<Self> {
        let maf = maf_threshold.unwrap_or(0.0);
        let miss = max_missing_rate.unwrap_or(1.0);
        let fill = fill_missing.unwrap_or(true);
        let model_key = model.as_deref().unwrap_or("add").to_ascii_lowercase();
        if !matches!(model_key.as_str(), "add" | "dom" | "rec" | "het") {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "model must be one of: add, dom, rec, het",
            ));
        }
        let het = het_threshold.unwrap_or(0.02);
        if !(0.0..=1.0).contains(&het) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "het_threshold must be within [0, 1.0]",
            ));
        }
        let apply_het_filter = model_key != "add";
        if mmap_window_mb.is_some()
            && (snp_range.is_some() || snp_indices.is_some() || bim_range.is_some())
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mmap_window_mb does not support snp_range/snp_indices/bim_range",
            ));
        }
        let it = if let Some(window_mb) = mmap_window_mb {
            BedSnpIter::new_with_fill_window(&prefix, 0.0, 1.0, false, false, het, window_mb)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?
        } else {
            BedSnpIter::new_with_fill(&prefix, 0.0, 1.0, false, false, het)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?
        };
        let (sample_indices, sample_ids) =
            build_sample_selection(&it.samples, sample_ids, sample_indices)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let mut snp_indices =
            build_snp_indices(&it.sites, snp_range, snp_indices, bim_range, snp_sites)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let site_filter = SiteFilterExpr::from_parts(None, None, chr_keys, bp_min, bp_max, ranges)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        if site_filter.active() {
            if let Some(ref mut idx) = snp_indices {
                idx.retain(|&snp_idx| {
                    it.sites
                        .get(snp_idx)
                        .map(|s| site_filter.keep_site(s))
                        .unwrap_or(false)
                });
            } else {
                let mut idx: Vec<usize> = Vec::with_capacity(it.sites.len());
                for (i, s) in it.sites.iter().enumerate() {
                    if site_filter.keep_site(s) {
                        idx.push(i);
                    }
                }
                snp_indices = Some(idx);
            }
        }

        Ok(Self {
            it,
            snp_indices,
            snp_pos: 0,
            sample_indices,
            sample_ids,
            maf,
            miss,
            fill_missing: fill,
            apply_het_filter,
            het_threshold: het,
        })
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.sample_indices.len()
    }

    #[getter]
    fn n_snps(&self) -> usize {
        self.snp_indices
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(self.it.sites.len())
    }

    #[getter]
    fn sample_ids(&self) -> Vec<String> {
        self.sample_ids.clone()
    }

    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk_size: usize,
    ) -> PyResult<Option<(Bound<'py, PyArray2<f32>>, Vec<SiteInfo>)>> {
        if chunk_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chunk_size must be > 0",
            ));
        }

        let n = self.sample_indices.len();
        if n == 0 {
            return Ok(None);
        }
        let mut data: Vec<f32> = Vec::with_capacity(chunk_size * n);
        let mut sites: Vec<SiteInfo> = Vec::with_capacity(chunk_size);
        let full_samples = self.sample_indices.len() == self.it.n_samples();
        let windowed_mode = self.it.is_windowed();
        let can_parallel_random = !windowed_mode;
        let mut m: usize = 0;

        if let Some(ref snp_indices) = self.snp_indices {
            while m < chunk_size && self.snp_pos < snp_indices.len() {
                let need = chunk_size - m;
                let end = (self.snp_pos + need).min(snp_indices.len());
                let batch = &snp_indices[self.snp_pos..end];
                self.snp_pos = end;

                if can_parallel_random {
                    let it = &self.it;
                    let sample_indices = &self.sample_indices;
                    let maf = self.maf;
                    let miss = self.miss;
                    let fill_missing = self.fill_missing;
                    let apply_het_filter = self.apply_het_filter;
                    let het_threshold = self.het_threshold;
                    let decoded: Vec<Option<(Vec<f32>, SiteInfo)>> = batch
                        .par_iter()
                        .map(|&snp_idx| {
                            let (mut row_sub, mut site) = if full_samples {
                                it.get_snp_row_raw(snp_idx)?
                            } else {
                                it.get_snp_row_selected_raw(snp_idx, sample_indices)?
                            };
                            let keep = core::process_snp_row(
                                &mut row_sub,
                                &mut site.ref_allele,
                                &mut site.alt_allele,
                                maf,
                                miss,
                                fill_missing,
                                apply_het_filter,
                                het_threshold,
                            );
                            if keep {
                                Some((row_sub, site.into()))
                            } else {
                                None
                            }
                        })
                        .collect();
                    for item in decoded.into_iter().flatten() {
                        let (row_sub, site) = item;
                        data.extend_from_slice(&row_sub);
                        sites.push(site);
                        m += 1;
                    }
                } else {
                    for &snp_idx in batch.iter() {
                        let maybe = if full_samples {
                            self.it.get_snp_row_raw(snp_idx)
                        } else {
                            self.it
                                .get_snp_row_selected_raw(snp_idx, &self.sample_indices)
                        };
                        if let Some((mut row_sub, mut site)) = maybe {
                            let keep = core::process_snp_row(
                                &mut row_sub,
                                &mut site.ref_allele,
                                &mut site.alt_allele,
                                self.maf,
                                self.miss,
                                self.fill_missing,
                                self.apply_het_filter,
                                self.het_threshold,
                            );
                            if keep {
                                data.extend_from_slice(&row_sub);
                                sites.push(site.into());
                                m += 1;
                            }
                        }
                    }
                }
            }
        } else {
            if can_parallel_random {
                while m < chunk_size && self.it.cursor() < self.it.n_snps() {
                    let need = chunk_size - m;
                    let start = self.it.cursor();
                    let end = (start + need).min(self.it.n_snps());
                    self.it.set_cursor(end);
                    let it = &self.it;
                    let sample_indices = &self.sample_indices;
                    let maf = self.maf;
                    let miss = self.miss;
                    let fill_missing = self.fill_missing;
                    let apply_het_filter = self.apply_het_filter;
                    let het_threshold = self.het_threshold;
                    let decoded: Vec<Option<(Vec<f32>, SiteInfo)>> = (start..end)
                        .into_par_iter()
                        .map(|snp_idx| {
                            let (mut row_sub, mut site) = if full_samples {
                                it.get_snp_row_raw(snp_idx)?
                            } else {
                                it.get_snp_row_selected_raw(snp_idx, sample_indices)?
                            };
                            let keep = core::process_snp_row(
                                &mut row_sub,
                                &mut site.ref_allele,
                                &mut site.alt_allele,
                                maf,
                                miss,
                                fill_missing,
                                apply_het_filter,
                                het_threshold,
                            );
                            if keep {
                                Some((row_sub, site.into()))
                            } else {
                                None
                            }
                        })
                        .collect();
                    for item in decoded.into_iter().flatten() {
                        let (row_sub, site) = item;
                        data.extend_from_slice(&row_sub);
                        sites.push(site);
                        m += 1;
                    }
                }
            } else if windowed_mode {
                while m < chunk_size {
                    let need = chunk_size - m;
                    let mut raw_rows: Vec<(Vec<f32>, core::SiteInfo)> = Vec::with_capacity(need);
                    for _ in 0..need {
                        let maybe = if full_samples {
                            self.it.next_snp_raw()
                        } else {
                            self.it.next_snp_selected_raw(&self.sample_indices)
                        };
                        if let Some(v) = maybe {
                            raw_rows.push(v);
                        } else {
                            break;
                        }
                    }
                    if raw_rows.is_empty() {
                        break;
                    }

                    let maf = self.maf;
                    let miss = self.miss;
                    let fill_missing = self.fill_missing;
                    let apply_het_filter = self.apply_het_filter;
                    let het_threshold = self.het_threshold;

                    let decoded: Vec<Option<(Vec<f32>, SiteInfo)>> = if raw_rows.len() >= 64 {
                        raw_rows
                            .into_par_iter()
                            .map(|(mut row_sub, mut site)| {
                                let keep = core::process_snp_row(
                                    &mut row_sub,
                                    &mut site.ref_allele,
                                    &mut site.alt_allele,
                                    maf,
                                    miss,
                                    fill_missing,
                                    apply_het_filter,
                                    het_threshold,
                                );
                                if keep {
                                    Some((row_sub, site.into()))
                                } else {
                                    None
                                }
                            })
                            .collect()
                    } else {
                        raw_rows
                            .into_iter()
                            .map(|(mut row_sub, mut site)| {
                                let keep = core::process_snp_row(
                                    &mut row_sub,
                                    &mut site.ref_allele,
                                    &mut site.alt_allele,
                                    maf,
                                    miss,
                                    fill_missing,
                                    apply_het_filter,
                                    het_threshold,
                                );
                                if keep {
                                    Some((row_sub, site.into()))
                                } else {
                                    None
                                }
                            })
                            .collect()
                    };

                    for item in decoded.into_iter().flatten() {
                        let (row_sub, site) = item;
                        data.extend_from_slice(&row_sub);
                        sites.push(site);
                        m += 1;
                    }
                }
            } else {
                while m < chunk_size {
                    let maybe = if full_samples {
                        self.it.next_snp_raw()
                    } else {
                        self.it.next_snp_selected_raw(&self.sample_indices)
                    };
                    match maybe {
                        Some((mut row_sub, mut site)) => {
                            let keep = core::process_snp_row(
                                &mut row_sub,
                                &mut site.ref_allele,
                                &mut site.alt_allele,
                                self.maf,
                                self.miss,
                                self.fill_missing,
                                self.apply_het_filter,
                                self.het_threshold,
                            );
                            if keep {
                                data.extend_from_slice(&row_sub);
                                sites.push(site.into());
                                m += 1;
                            }
                        }
                        None => break,
                    }
                }
            }
        }
        if m == 0 {
            return Ok(None);
        }

        let mat = Array2::from_shape_vec((m, n), data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        #[allow(deprecated)]
        let py_mat = PyArray2::from_owned_array(py, mat).into_bound();
        Ok(Some((py_mat, sites)))
    }

    /// Return chunk with model-coding + row-centering prepared in Rust.
    ///
    /// Output tuple:
    ///   (geno_centered, sites, af, miss)
    /// where `af` is:
    ///   - additive: mean(additive dosage)/2 with ALT kept as dosage 1
    ///   - dom/rec/het: mean(coded value)
    /// and `miss` is the per-site missing genotype count on the selected samples.
    #[pyo3(signature = (chunk_size, coding=None, snps_only=false))]
    fn next_chunk_prepared<'py>(
        &mut self,
        py: Python<'py>,
        chunk_size: usize,
        coding: Option<String>,
        snps_only: bool,
    ) -> PyResult<
        Option<(
            Bound<'py, PyArray2<f32>>,
            Vec<SiteInfo>,
            Bound<'py, PyArray1<f32>>,
            Bound<'py, PyArray1<f32>>,
        )>,
    > {
        if chunk_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chunk_size must be > 0",
            ));
        }
        let coding_key = coding
            .unwrap_or_else(|| "add".to_string())
            .trim()
            .to_ascii_lowercase();
        let coding_mode = match coding_key.as_str() {
            "add" => BedPreparedCodingMode::Add,
            "dom" => BedPreparedCodingMode::Dom,
            "rec" => BedPreparedCodingMode::Rec,
            "het" => BedPreparedCodingMode::Het,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "coding must be one of: add, dom, rec, het",
                ))
            }
        };
        let additive_mode = matches!(coding_mode, BedPreparedCodingMode::Add);

        let n = self.sample_indices.len();
        if n == 0 {
            return Ok(None);
        }

        let full_samples = self.sample_indices.len() == self.it.n_samples();
        let windowed_mode = self.it.is_windowed();
        let can_parallel_random = !windowed_mode;

        let mut out: Vec<f32> = Vec::with_capacity(chunk_size * n);
        let mut sites: Vec<SiteInfo> = Vec::with_capacity(chunk_size);
        let mut af: Vec<f32> = Vec::with_capacity(chunk_size);
        let mut miss: Vec<f32> = Vec::with_capacity(chunk_size);
        let mut m = 0usize;
        let maf_thr = self.maf;
        let miss_thr = self.miss;
        let fill_missing = self.fill_missing;
        let apply_het_filter = self.apply_het_filter;
        let het_thr = self.het_threshold;

        let prepare_one = |mut row_sub: Vec<f32>, mut site: core::SiteInfo| {
            let stats = core::process_snp_row_with_stats_preserve_alt(
                &mut row_sub,
                &mut site.ref_allele,
                &mut site.alt_allele,
                maf_thr,
                miss_thr,
                fill_missing,
                apply_het_filter,
                het_thr,
            );
            let Some(stats) = stats else {
                return None;
            };
            if snps_only
                && (!is_simple_snp_allele(&site.ref_allele)
                    || !is_simple_snp_allele(&site.alt_allele))
            {
                return None;
            }

            let mut sum = 0.0_f64;
            for v in row_sub.iter_mut() {
                let mv = bed_apply_coding_value(*v, coding_mode);
                *v = mv;
                sum += mv as f64;
            }
            let coded_mean = (sum / n as f64) as f32;
            let mean = coded_mean;
            for v in row_sub.iter_mut() {
                *v -= mean;
            }
            let af_v = if additive_mode {
                coded_mean * 0.5_f32
            } else {
                coded_mean
            };
            Some((row_sub, site.into(), af_v, stats.missing_count as f32))
        };

        if let Some(ref snp_indices) = self.snp_indices {
            if windowed_mode {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "windowed mmap mode does not support explicit snp index selection",
                ));
            }
            while m < chunk_size && self.snp_pos < snp_indices.len() {
                let need = chunk_size - m;
                let end = (self.snp_pos + need).min(snp_indices.len());
                let batch = &snp_indices[self.snp_pos..end];
                self.snp_pos = end;

                let decoded: Vec<Option<(Vec<f32>, SiteInfo, f32, f32)>> = if can_parallel_random {
                    let it = &self.it;
                    let sample_indices = &self.sample_indices;
                    batch
                        .par_iter()
                        .map(|&snp_idx| {
                            let maybe = if full_samples {
                                it.get_snp_row_raw(snp_idx)
                            } else {
                                it.get_snp_row_selected_raw(snp_idx, sample_indices)
                            };
                            maybe.and_then(|(row_sub, site)| prepare_one(row_sub, site))
                        })
                        .collect()
                } else {
                    let mut tmp: Vec<Option<(Vec<f32>, SiteInfo, f32, f32)>> =
                        Vec::with_capacity(batch.len());
                    for &snp_idx in batch.iter() {
                        let maybe = if full_samples {
                            self.it.get_snp_row_raw(snp_idx)
                        } else {
                            self.it
                                .get_snp_row_selected_raw(snp_idx, &self.sample_indices)
                        };
                        tmp.push(maybe.and_then(|(row_sub, site)| prepare_one(row_sub, site)));
                    }
                    tmp
                };

                for item in decoded.into_iter().flatten() {
                    let (row_sub, site, af_v, miss_v) = item;
                    out.extend_from_slice(&row_sub);
                    sites.push(site);
                    af.push(af_v);
                    miss.push(miss_v);
                    m += 1;
                }
            }
        } else if can_parallel_random {
            while m < chunk_size && self.it.cursor() < self.it.n_snps() {
                let need = chunk_size - m;
                let start = self.it.cursor();
                let end = (start + need).min(self.it.n_snps());
                self.it.set_cursor(end);
                let it = &self.it;
                let sample_indices = &self.sample_indices;
                let decoded: Vec<Option<(Vec<f32>, SiteInfo, f32, f32)>> = (start..end)
                    .into_par_iter()
                    .map(|snp_idx| {
                        let maybe = if full_samples {
                            it.get_snp_row_raw(snp_idx)
                        } else {
                            it.get_snp_row_selected_raw(snp_idx, sample_indices)
                        };
                        maybe.and_then(|(row_sub, site)| prepare_one(row_sub, site))
                    })
                    .collect();
                for item in decoded.into_iter().flatten() {
                    let (row_sub, site, af_v, miss_v) = item;
                    out.extend_from_slice(&row_sub);
                    sites.push(site);
                    af.push(af_v);
                    miss.push(miss_v);
                    m += 1;
                }
            }
        } else if windowed_mode {
            while m < chunk_size {
                let need = chunk_size - m;
                let mut raw_rows: Vec<(Vec<f32>, core::SiteInfo)> = Vec::with_capacity(need);
                for _ in 0..need {
                    let maybe = if full_samples {
                        self.it.next_snp_raw()
                    } else {
                        self.it.next_snp_selected_raw(&self.sample_indices)
                    };
                    if let Some(v) = maybe {
                        raw_rows.push(v);
                    } else {
                        break;
                    }
                }
                if raw_rows.is_empty() {
                    break;
                }
                let decoded: Vec<Option<(Vec<f32>, SiteInfo, f32, f32)>> = if raw_rows.len() >= 64 {
                    raw_rows
                        .into_par_iter()
                        .map(|(row_sub, site)| prepare_one(row_sub, site))
                        .collect()
                } else {
                    raw_rows
                        .into_iter()
                        .map(|(row_sub, site)| prepare_one(row_sub, site))
                        .collect()
                };
                for item in decoded.into_iter().flatten() {
                    let (row_sub, site, af_v, miss_v) = item;
                    out.extend_from_slice(&row_sub);
                    sites.push(site);
                    af.push(af_v);
                    miss.push(miss_v);
                    m += 1;
                }
            }
        } else {
            while m < chunk_size {
                let maybe = if full_samples {
                    self.it.next_snp_raw()
                } else {
                    self.it.next_snp_selected_raw(&self.sample_indices)
                };
                if let Some((row_sub, site)) = maybe {
                    if let Some((row_sub2, site2, af_v, miss_v)) = prepare_one(row_sub, site) {
                        out.extend_from_slice(&row_sub2);
                        sites.push(site2);
                        af.push(af_v);
                        miss.push(miss_v);
                        m += 1;
                    }
                } else {
                    break;
                }
            }
        }

        if m == 0 {
            return Ok(None);
        }

        let mat = Array2::from_shape_vec((m, n), out)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        #[allow(deprecated)]
        let py_mat = PyArray2::from_owned_array(py, mat).into_bound();
        #[allow(deprecated)]
        let maf_arr = PyArray1::from_owned_array(py, Array1::from_vec(af)).into_bound();
        #[allow(deprecated)]
        let miss_arr = PyArray1::from_owned_array(py, Array1::from_vec(miss)).into_bound();
        Ok(Some((py_mat, sites, maf_arr, miss_arr)))
    }
}

// -------- VcfChunkReader --------
#[pyclass]
pub struct VcfChunkReader {
    it: VcfSnpIter,
    sample_indices: Vec<usize>,
    sample_ids: Vec<String>,
    site_filter: SiteFilterExpr,
    maf: f32,
    miss: f32,
    fill_missing: bool,
    apply_het_filter: bool,
    het_threshold: f32,
}

#[pymethods]
impl VcfChunkReader {
    #[new]
    #[pyo3(signature = (
        path,
        maf_threshold=None,
        max_missing_rate=None,
        fill_missing=None,
        sample_ids=None,
        sample_indices=None,
        model=None,
        het_threshold=None,
        snp_sites=None,
        bim_range=None,
        chr_keys=None,
        bp_min=None,
        bp_max=None,
        ranges=None,
    ))]
    fn new(
        path: String,
        maf_threshold: Option<f32>,
        max_missing_rate: Option<f32>,
        fill_missing: Option<bool>,
        sample_ids: Option<Vec<String>>,
        sample_indices: Option<Vec<usize>>,
        model: Option<String>,
        het_threshold: Option<f32>,
        snp_sites: Option<Vec<(String, i32)>>,
        bim_range: Option<(String, i32, i32)>,
        chr_keys: Option<Vec<String>>,
        bp_min: Option<i32>,
        bp_max: Option<i32>,
        ranges: Option<Vec<(String, i32, i32)>>,
    ) -> PyResult<Self> {
        let maf = maf_threshold.unwrap_or(0.0);
        let miss = max_missing_rate.unwrap_or(1.0);
        let fill = fill_missing.unwrap_or(true);
        let model_key = model.as_deref().unwrap_or("add").to_ascii_lowercase();
        if !matches!(model_key.as_str(), "add" | "dom" | "rec" | "het") {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "model must be one of: add, dom, rec, het",
            ));
        }
        let het = het_threshold.unwrap_or(0.02);
        if !(0.0..=1.0).contains(&het) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "het_threshold must be within [0, 1.0]",
            ));
        }
        let apply_het_filter = model_key != "add";
        let it = VcfSnpIter::new_with_fill(&path, 0.0, 1.0, false, false, het)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let (sample_indices, sample_ids) =
            build_sample_selection(&it.samples, sample_ids, sample_indices)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let site_filter =
            SiteFilterExpr::from_parts(snp_sites, bim_range, chr_keys, bp_min, bp_max, ranges)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(Self {
            it,
            sample_indices,
            sample_ids,
            site_filter,
            maf,
            miss,
            fill_missing: fill,
            apply_het_filter,
            het_threshold: het,
        })
    }

    #[getter]
    fn sample_ids(&self) -> Vec<String> {
        self.sample_ids.clone()
    }

    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk_size: usize,
    ) -> PyResult<Option<(Bound<'py, PyArray2<f32>>, Vec<SiteInfo>)>> {
        if chunk_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chunk_size must be > 0",
            ));
        }

        let n = self.sample_indices.len();
        let full_samples = n == self.it.n_samples();
        let mut data: Vec<f32> = Vec::with_capacity(chunk_size * n);
        let mut sites: Vec<SiteInfo> = Vec::with_capacity(chunk_size);
        let mut m: usize = 0;

        while m < chunk_size {
            match self.it.next_snp_raw() {
                Some((row, mut site)) => {
                    if !self.site_filter.keep_site(&site) {
                        continue;
                    }
                    let mut row_sub = if full_samples {
                        row
                    } else {
                        self.sample_indices.iter().map(|&i| row[i]).collect()
                    };
                    let keep = core::process_snp_row(
                        &mut row_sub,
                        &mut site.ref_allele,
                        &mut site.alt_allele,
                        self.maf,
                        self.miss,
                        self.fill_missing,
                        self.apply_het_filter,
                        self.het_threshold,
                    );
                    if keep {
                        data.extend_from_slice(&row_sub);
                        sites.push(site.into());
                        m += 1;
                    }
                }
                None => break,
            }
        }
        if m == 0 {
            return Ok(None);
        }

        let mat = Array2::from_shape_vec((m, n), data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        #[allow(deprecated)]
        let py_mat = PyArray2::from_owned_array(py, mat).into_bound();
        Ok(Some((py_mat, sites)))
    }
}

// -------- HmpChunkReader --------
#[pyclass]
pub struct HmpChunkReader {
    it: HmpSnpIter,
    sample_indices: Vec<usize>,
    sample_ids: Vec<String>,
    site_filter: SiteFilterExpr,
    maf: f32,
    miss: f32,
    fill_missing: bool,
    apply_het_filter: bool,
    het_threshold: f32,
}

#[pymethods]
impl HmpChunkReader {
    #[new]
    #[pyo3(signature = (
        path,
        maf_threshold=None,
        max_missing_rate=None,
        fill_missing=None,
        sample_ids=None,
        sample_indices=None,
        model=None,
        het_threshold=None,
        snp_sites=None,
        bim_range=None,
        chr_keys=None,
        bp_min=None,
        bp_max=None,
        ranges=None,
    ))]
    fn new(
        path: String,
        maf_threshold: Option<f32>,
        max_missing_rate: Option<f32>,
        fill_missing: Option<bool>,
        sample_ids: Option<Vec<String>>,
        sample_indices: Option<Vec<usize>>,
        model: Option<String>,
        het_threshold: Option<f32>,
        snp_sites: Option<Vec<(String, i32)>>,
        bim_range: Option<(String, i32, i32)>,
        chr_keys: Option<Vec<String>>,
        bp_min: Option<i32>,
        bp_max: Option<i32>,
        ranges: Option<Vec<(String, i32, i32)>>,
    ) -> PyResult<Self> {
        let maf = maf_threshold.unwrap_or(0.0);
        let miss = max_missing_rate.unwrap_or(1.0);
        let fill = fill_missing.unwrap_or(true);
        let model_key = model.as_deref().unwrap_or("add").to_ascii_lowercase();
        if !matches!(model_key.as_str(), "add" | "dom" | "rec" | "het") {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "model must be one of: add, dom, rec, het",
            ));
        }
        let het = het_threshold.unwrap_or(0.02);
        if !(0.0..=1.0).contains(&het) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "het_threshold must be within [0, 1.0]",
            ));
        }
        let apply_het_filter = model_key != "add";
        let it = HmpSnpIter::new_with_fill(&path, 0.0, 1.0, false, false, het)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let (sample_indices, sample_ids) =
            build_sample_selection(&it.samples, sample_ids, sample_indices)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let site_filter =
            SiteFilterExpr::from_parts(snp_sites, bim_range, chr_keys, bp_min, bp_max, ranges)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(Self {
            it,
            sample_indices,
            sample_ids,
            site_filter,
            maf,
            miss,
            fill_missing: fill,
            apply_het_filter,
            het_threshold: het,
        })
    }

    #[getter]
    fn sample_ids(&self) -> Vec<String> {
        self.sample_ids.clone()
    }

    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk_size: usize,
    ) -> PyResult<Option<(Bound<'py, PyArray2<f32>>, Vec<SiteInfo>)>> {
        if chunk_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chunk_size must be > 0",
            ));
        }

        let n = self.sample_indices.len();
        let full_samples = n == self.it.n_samples();
        let mut data: Vec<f32> = Vec::with_capacity(chunk_size * n);
        let mut sites: Vec<SiteInfo> = Vec::with_capacity(chunk_size);
        let mut m: usize = 0;

        while m < chunk_size {
            match self.it.next_snp_raw() {
                Some((row, mut site)) => {
                    if !self.site_filter.keep_site(&site) {
                        continue;
                    }
                    let mut row_sub = if full_samples {
                        row
                    } else {
                        self.sample_indices.iter().map(|&i| row[i]).collect()
                    };
                    let keep = core::process_snp_row(
                        &mut row_sub,
                        &mut site.ref_allele,
                        &mut site.alt_allele,
                        self.maf,
                        self.miss,
                        self.fill_missing,
                        self.apply_het_filter,
                        self.het_threshold,
                    );
                    if keep {
                        data.extend_from_slice(&row_sub);
                        sites.push(site.into());
                        m += 1;
                    }
                }
                None => break,
            }
        }
        if m == 0 {
            return Ok(None);
        }

        let mat = Array2::from_shape_vec((m, n), data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        #[allow(deprecated)]
        let py_mat = PyArray2::from_owned_array(py, mat).into_bound();
        Ok(Some((py_mat, sites)))
    }
}

// -------- TxtChunkReader --------
#[pyclass]
pub struct TxtChunkReader {
    it: TxtSnpIter,
    snp_indices: Option<Vec<usize>>,
    snp_pos: usize,
    sample_indices: Vec<usize>,
    sample_ids: Vec<String>,
    site_filter: SiteFilterExpr,
    maf: f32,
    miss: f32,
    fill_missing: bool,
    apply_het_filter: bool,
    het_threshold: f32,
    passthrough_raw: bool,
}

#[inline]
fn impute_missing_with_row_mean(row: &mut [f32]) {
    let mut sum: f64 = 0.0;
    let mut n_obs: usize = 0;
    for &v in row.iter() {
        if v >= 0.0 {
            sum += v as f64;
            n_obs += 1;
        }
    }
    let fill = if n_obs > 0 {
        (sum / n_obs as f64) as f32
    } else {
        0.0
    };
    for v in row.iter_mut() {
        if *v < 0.0 {
            *v = fill;
        }
    }
}

#[pymethods]
impl TxtChunkReader {
    #[new]
    #[pyo3(signature = (
        path,
        delimiter=None,
        snp_range=None,
        snp_indices=None,
        bim_range=None,
        snp_sites=None,
        sample_ids=None,
        sample_indices=None,
        maf_threshold=None,
        max_missing_rate=None,
        fill_missing=None,
        model=None,
        het_threshold=None,
        chr_keys=None,
        bp_min=None,
        bp_max=None,
        ranges=None,
    ))]
    fn new(
        path: String,
        delimiter: Option<String>,
        snp_range: Option<(usize, usize)>,
        snp_indices: Option<Vec<usize>>,
        bim_range: Option<(String, i32, i32)>,
        snp_sites: Option<Vec<(String, i32)>>,
        sample_ids: Option<Vec<String>>,
        sample_indices: Option<Vec<usize>>,
        maf_threshold: Option<f32>,
        max_missing_rate: Option<f32>,
        fill_missing: Option<bool>,
        model: Option<String>,
        het_threshold: Option<f32>,
        chr_keys: Option<Vec<String>>,
        bp_min: Option<i32>,
        bp_max: Option<i32>,
        ranges: Option<Vec<(String, i32, i32)>>,
    ) -> PyResult<Self> {
        let maf = maf_threshold.unwrap_or(0.0);
        let miss = max_missing_rate.unwrap_or(1.0);
        let fill = fill_missing.unwrap_or(true);
        let model_key = model.as_deref().unwrap_or("add").to_ascii_lowercase();
        if !matches!(model_key.as_str(), "add" | "dom" | "rec" | "het") {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "model must be one of: add, dom, rec, het",
            ));
        }
        let het = het_threshold.unwrap_or(0.02);
        if !(0.0..=1.0).contains(&het) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "het_threshold must be within [0, 1.0]",
            ));
        }
        let apply_het_filter = model_key != "add";
        // When no genotype-QC is requested in additive mode, treat numeric
        // TXT/NPY/BIN matrices as raw values instead of genotype dosage.
        // This avoids accidental row dropping for non-0/1/2 matrices.
        let passthrough_raw = maf <= 0.0 && miss >= 1.0 && !apply_het_filter;

        let it = TxtSnpIter::new(&path, delimiter.as_deref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        let (sample_indices, sample_ids) =
            build_sample_selection(&it.samples, sample_ids, sample_indices)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        let snp_indices =
            build_snp_indices(&it.sites, snp_range, snp_indices, bim_range, snp_sites)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let site_filter = SiteFilterExpr::from_parts(None, None, chr_keys, bp_min, bp_max, ranges)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        Ok(Self {
            it,
            snp_indices,
            snp_pos: 0,
            sample_indices,
            sample_ids,
            site_filter,
            maf,
            miss,
            fill_missing: fill,
            apply_het_filter,
            het_threshold: het,
            passthrough_raw,
        })
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.sample_indices.len()
    }

    #[getter]
    fn n_snps(&self) -> usize {
        self.snp_indices
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(self.it.sites.len())
    }

    #[getter]
    fn sample_ids(&self) -> Vec<String> {
        self.sample_ids.clone()
    }

    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk_size: usize,
    ) -> PyResult<Option<(Bound<'py, PyArray2<f32>>, Vec<SiteInfo>)>> {
        if chunk_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chunk_size must be > 0",
            ));
        }

        let n = self.sample_indices.len();
        if n == 0 {
            return Ok(None);
        }
        let full_samples = n == self.it.n_samples();
        let mut data: Vec<f32> = Vec::with_capacity(chunk_size * n);
        let mut sites: Vec<SiteInfo> = Vec::with_capacity(chunk_size);
        let mut m: usize = 0;

        if let Some(ref snp_indices) = self.snp_indices {
            while m < chunk_size && self.snp_pos < snp_indices.len() {
                let snp_idx = snp_indices[self.snp_pos];
                self.snp_pos += 1;
                if let Some((row, mut site)) = self.it.get_snp_row(snp_idx) {
                    if !self.site_filter.keep_site(&site) {
                        continue;
                    }
                    let mut row_sub = if full_samples {
                        row
                    } else {
                        self.sample_indices.iter().map(|&i| row[i]).collect()
                    };
                    if self.passthrough_raw {
                        if self.fill_missing {
                            impute_missing_with_row_mean(&mut row_sub);
                        }
                        data.extend_from_slice(&row_sub);
                        sites.push(site.into());
                        m += 1;
                    } else {
                        let keep = core::process_snp_row(
                            &mut row_sub,
                            &mut site.ref_allele,
                            &mut site.alt_allele,
                            self.maf,
                            self.miss,
                            self.fill_missing,
                            self.apply_het_filter,
                            self.het_threshold,
                        );
                        if keep {
                            data.extend_from_slice(&row_sub);
                            sites.push(site.into());
                            m += 1;
                        }
                    }
                }
            }
        } else {
            while m < chunk_size {
                match self.it.next_snp() {
                    Some((row, mut site)) => {
                        if !self.site_filter.keep_site(&site) {
                            continue;
                        }
                        let mut row_sub = if full_samples {
                            row
                        } else {
                            self.sample_indices.iter().map(|&i| row[i]).collect()
                        };
                        if self.passthrough_raw {
                            if self.fill_missing {
                                impute_missing_with_row_mean(&mut row_sub);
                            }
                            data.extend_from_slice(&row_sub);
                            sites.push(site.into());
                            m += 1;
                        } else {
                            let keep = core::process_snp_row(
                                &mut row_sub,
                                &mut site.ref_allele,
                                &mut site.alt_allele,
                                self.maf,
                                self.miss,
                                self.fill_missing,
                                self.apply_het_filter,
                                self.het_threshold,
                            );
                            if keep {
                                data.extend_from_slice(&row_sub);
                                sites.push(site.into());
                                m += 1;
                            }
                        }
                    }
                    None => break,
                }
            }
        }

        if m == 0 {
            return Ok(None);
        }

        let mat = Array2::from_shape_vec((m, n), data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        #[allow(deprecated)]
        let py_mat = PyArray2::from_owned_array(py, mat).into_bound();
        Ok(Some((py_mat, sites)))
    }
}

// -------- count_vcf_snps (Py function) --------
#[pyfunction]
pub fn load_bed_2bit_packed<'py>(
    py: Python<'py>,
    prefix: String,
) -> PyResult<(
    Bound<'py, PyArray2<u8>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    usize,
)> {
    let mut bed_prefix = prefix;
    let lower = bed_prefix.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        bed_prefix.truncate(bed_prefix.len() - 4);
    }
    let (packed, missing_rate, maf, std_denom, n_samples, n_snps, bytes_per_snp) = py
        .detach(move || -> Result<
            (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>, usize, usize, usize),
            String,
        > {
            let samples = core::read_fam(&bed_prefix).map_err(|e| e.to_string())?;
            let n_samples = samples.len();
            if n_samples == 0 {
                return Err("no samples found in PLINK input".to_string());
            }
            emit_gfreader_rss_debug(
                "load_bed_2bit_packed/read_fam",
                &format!("prefix={bed_prefix} n_samples={n_samples}"),
            );

            let bed_path = format!("{bed_prefix}.bed");
            let bed_file = File::open(&bed_path)
                .map_err(|e| format!("failed to open {bed_path}: {e}"))?;
            let mmap = unsafe { Mmap::map(&bed_file) }
                .map_err(|e| format!("failed to mmap {bed_path}: {e}"))?;
            if mmap.len() < 3 {
                return Err("BED too small".to_string());
            }
            if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
                return Err("Only SNP-major BED supported".to_string());
            }

            let bytes_per_snp = (n_samples + 3) / 4;
            let data_len = mmap.len() - 3;
            if bytes_per_snp == 0 || data_len % bytes_per_snp != 0 {
                return Err(format!(
                    "invalid BED payload length: data_len={data_len}, bytes_per_snp={bytes_per_snp}"
                ));
            }
            let n_snps = data_len / bytes_per_snp;
            let packed_src = &mmap[3..];
            emit_gfreader_rss_debug(
                "load_bed_2bit_packed/mmap_ready",
                &format!(
                    "prefix={bed_prefix} n_samples={n_samples} n_snps={n_snps} bytes_per_snp={bytes_per_snp} payload={}",
                    format_debug_bytes_local(packed_src.len() as u64),
                ),
            );

            let mut missing_rate: Vec<f32> = vec![0.0; n_snps];
            let mut maf: Vec<f32> = vec![0.0; n_snps];
            let mut std_denom: Vec<f32> = vec![0.0; n_snps];
            missing_rate
                .par_iter_mut()
                .zip(maf.par_iter_mut())
                .zip(std_denom.par_iter_mut())
                .enumerate()
                .for_each(|(snp_idx, ((miss_dst, maf_dst), std_dst))| {
                    let row = &packed_src[snp_idx * bytes_per_snp..(snp_idx + 1) * bytes_per_snp];
                    let (missing, het, hom_alt) = count_packed_row_counts(row, n_samples);
                    let non_missing = n_samples.saturating_sub(missing);
                    let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));

                    *miss_dst = (missing as f32) / (n_samples as f32);
                    if non_missing > 0 {
                        let p = alt_sum as f32 / (2.0_f32 * non_missing as f32);
                        let maf_v = p.min(1.0_f32 - p);
                        *maf_dst = maf_v;
                        let d = (2.0_f32 * p * (1.0_f32 - p)).sqrt();
                        *std_dst = if d.is_finite() { d } else { 0.0_f32 };
                    } else {
                        *maf_dst = 0.0_f32;
                        *std_dst = 0.0_f32;
                    }
                });
            let stats_bytes = ((missing_rate.len() + maf.len() + std_denom.len())
                * std::mem::size_of::<f32>()) as u64;
            emit_gfreader_rss_debug(
                "load_bed_2bit_packed/row_stats_ready",
                &format!(
                    "n_snps={n_snps} stats_bytes={} arrays=3xf32",
                    format_debug_bytes_local(stats_bytes),
                ),
            );
            let packed = load_file_owned_range_exact(Path::new(&bed_path), 3, packed_src.len())?;
            emit_gfreader_rss_debug(
                "load_bed_2bit_packed/packed_copy_done",
                &format!(
                    "packed_bytes={} n_snps={n_snps} bytes_per_snp={bytes_per_snp}",
                    format_debug_bytes_local(packed.len() as u64),
                ),
            );
            Ok((
                packed,
                missing_rate,
                maf,
                std_denom,
                n_samples,
                n_snps,
                bytes_per_snp,
            ))
        })
        .map_err(PyRuntimeError::new_err)?;

    let packed_mat = Array2::from_shape_vec((n_snps, bytes_per_snp), packed)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    #[allow(deprecated)]
    let packed_arr = PyArray2::from_owned_array(py, packed_mat).into_bound();
    #[allow(deprecated)]
    let miss_arr = PyArray1::from_owned_array(py, Array1::from_vec(missing_rate)).into_bound();
    #[allow(deprecated)]
    let maf_arr = PyArray1::from_owned_array(py, Array1::from_vec(maf)).into_bound();
    #[allow(deprecated)]
    let denom_arr = PyArray1::from_owned_array(py, Array1::from_vec(std_denom)).into_bound();
    Ok((packed_arr, miss_arr, maf_arr, denom_arr, n_samples))
}

pub(crate) struct PackedBedSubsetOwned {
    pub packed: Vec<u8>,
    pub missing_rate: Vec<f32>,
    pub maf: Vec<f32>,
    pub std_denom: Vec<f32>,
    pub row_flip: Vec<bool>,
    pub n_samples: usize,
    pub bytes_per_snp: usize,
}

pub(crate) struct PreparedBedLogicMetaOwned {
    pub site_keep: Vec<bool>,
    pub row_flip: Vec<bool>,
    pub row_source_indices: Vec<usize>,
    pub missing_rate: Vec<f32>,
    pub maf: Vec<f32>,
    pub sites: Vec<core::SiteInfo>,
    pub n_samples: usize,
    pub n_snps_total: usize,
    pub bytes_per_snp: usize,
}

pub(crate) struct PreparedBedPackedOwned {
    pub packed: Vec<u8>,
    pub missing_rate: Vec<f32>,
    pub maf: Vec<f32>,
    pub std_denom: Vec<f32>,
    pub row_flip: Vec<bool>,
    pub site_keep: Vec<bool>,
    #[allow(dead_code)]
    pub sites: Vec<core::SiteInfo>,
    pub n_samples: usize,
    pub n_snps_total: usize,
    pub bytes_per_snp: usize,
}

pub(crate) fn prepare_bed_logic_meta_owned_for_stats_samples_sparse_windowed(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    stats_sample_indices: Option<&[usize]>,
    mmap_window_mb: usize,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
    progress_done_offset: usize,
    progress_every: usize,
) -> Result<PreparedBedLogicMetaOwned, String> {
    if mmap_window_mb == 0 {
        return Err("mmap_window_mb must be > 0".to_string());
    }

    let total_t0 = Instant::now();
    let mut bed_prefix = prefix.to_string();
    let lower = bed_prefix.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        bed_prefix.truncate(bed_prefix.len() - 4);
    }

    let open_iter_t0 = Instant::now();
    let mut it = BedSnpIter::new_for_grm_window(&bed_prefix, mmap_window_mb)?;
    let open_iter_secs = open_iter_t0.elapsed().as_secs_f64();
    let n_samples = it.n_samples();
    if n_samples == 0 {
        return Err("no samples found in PLINK input".to_string());
    }

    let stats_sample_indices = stats_sample_indices.unwrap_or(&[]);
    if !stats_sample_indices.is_empty() && stats_sample_indices.iter().any(|&idx| idx >= n_samples)
    {
        return Err("selected sample index out of range for BED logic preparation".to_string());
    }
    let stats_identity = stats_sample_indices.is_empty()
        || (stats_sample_indices.len() == n_samples
            && sample_indices_are_identity(stats_sample_indices));
    let stats_n_samples = if stats_identity {
        n_samples
    } else {
        stats_sample_indices.len()
    };
    if stats_n_samples == 0 {
        return Err("selected sample set for BED logic preparation is empty".to_string());
    }

    let n_snps = it.n_snps();
    let bytes_per_snp = n_samples.div_ceil(4);
    emit_gfreader_rss_debug(
        "prepare_bed_logic_meta/windowed_init",
        &format!(
            "prefix={bed_prefix} n_samples={n_samples} stats_n_samples={stats_n_samples} n_snps={n_snps} bytes_per_snp={bytes_per_snp} mmap_window_mb={mmap_window_mb}"
        ),
    );

    let apply_het_filter = het_threshold > 0.0_f32;
    let row_stats_t0 = Instant::now();
    let mut site_keep = Vec::<bool>::with_capacity(n_snps);
    let mut row_flip_keep = Vec::<bool>::new();
    let mut row_source_indices = Vec::<usize>::new();
    let mut missing_rate_keep = Vec::<f32>::new();
    let mut maf_keep = Vec::<f32>::new();
    let prep_threads = threads.max(1);
    let pool = if prep_threads > 1 {
        crate::stats_common::get_cached_pool(prep_threads).map_err(|e| e.to_string())?
    } else {
        None
    };
    let notify_step = if progress_every == 0 {
        (n_snps / 200).max(1)
    } else {
        progress_every.max(1)
    };
    let mut last_notified = progress_done_offset.min(progress_done_offset.saturating_add(n_snps));
    let progress_total_hint = progress_done_offset
        .saturating_add(n_snps.saturating_mul(2))
        .max(1);
    if let Some(cb) = progress_callback {
        Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            cb.call1(py2, (progress_done_offset, progress_total_hint))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    }

    let mut keep_flags = Vec::<u8>::new();
    let mut missing_batch = Vec::<f32>::new();
    let mut maf_batch = Vec::<f32>::new();
    while it.cursor() < n_snps {
        let base = it.cursor();
        it.ensure_window_for_snp(base)?;
        let scan_rows = it.mapped_contiguous_snps_from(base);
        if scan_rows == 0 {
            return Err("windowed BED pre-stat scan reached empty window".to_string());
        }

        keep_flags.resize(scan_rows, 0u8);
        keep_flags[..scan_rows].fill(0u8);
        missing_batch.resize(scan_rows, 0.0_f32);
        maf_batch.resize(scan_rows, 0.0_f32);

        let mut run = || {
            let it_ref: &BedSnpIter = &it;
            keep_flags[..scan_rows]
                .par_iter_mut()
                .zip(missing_batch[..scan_rows].par_iter_mut())
                .zip(maf_batch[..scan_rows].par_iter_mut())
                .enumerate()
                .for_each(|(off, ((keep_dst, missing_dst), maf_dst))| {
                    let snp_idx = base + off;
                    let counts = if stats_identity {
                        it_ref
                            .decode_snp_counts_only_at(snp_idx)
                            .expect("windowed pre-stat SNP index out of range")
                    } else {
                        it_ref
                            .decode_snp_selected_counts_only_at(snp_idx, stats_sample_indices)
                            .expect("windowed pre-stat selected SNP index out of range")
                    };
                    let non_missing = counts.non_missing;
                    let alt_sum = counts.alt_sum as usize;
                    let het_count = counts.het_count;
                    let (missing_rate, _maf, _std) =
                        packed_row_stats_from_counts(stats_n_samples, non_missing, alt_sum);
                    let alt_freq = if non_missing > 0 {
                        (counts.alt_sum as f32) / (2.0_f32 * non_missing as f32)
                    } else {
                        0.0_f32
                    };
                    let keep = if missing_rate > max_missing_rate {
                        false
                    } else if non_missing == 0 {
                        maf_threshold <= 0.0_f32
                    } else if apply_het_filter
                        && ((het_count as f64) / (non_missing as f64)) > (het_threshold as f64)
                    {
                        false
                    } else {
                        alt_freq.min(1.0_f32 - alt_freq) >= maf_threshold
                    };
                    *keep_dst = if keep { 1 } else { 0 };
                    *missing_dst = missing_rate;
                    *maf_dst = alt_freq;
                });
        };
        if let Some(tp) = pool.as_ref() {
            tp.install(&mut run);
        } else {
            run();
        }

        for off in 0..scan_rows {
            let keep = keep_flags[off] != 0;
            site_keep.push(keep);
            if keep {
                row_flip_keep.push(false);
                row_source_indices.push(base + off);
                missing_rate_keep.push(missing_batch[off]);
                maf_keep.push(maf_batch[off]);
            }
        }
        it.set_cursor(base + scan_rows);

        let done = progress_done_offset.saturating_add((base + scan_rows).min(n_snps));
        let total = progress_total_hint;
        if done >= last_notified.saturating_add(notify_step) || base + scan_rows >= n_snps {
            last_notified = done;
            Python::attach(|py2| -> PyResult<()> {
                py2.check_signals()?;
                if let Some(cb) = progress_callback {
                    cb.call1(py2, (done.min(total), total))?;
                }
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        }
    }
    let row_stats_secs = row_stats_t0.elapsed().as_secs_f64();

    if site_keep.len() != n_snps {
        return Err(format!(
            "internal error: site_keep rows {} != n_snps {n_snps}",
            site_keep.len()
        ));
    }

    let kept_n = row_source_indices.len();
    if kept_n == 0 {
        return Err(
            "No SNPs left after windowed BED filtering. Please relax thresholds.".to_string(),
        );
    }
    emit_gfreader_rss_debug(
        "prepare_bed_logic_meta/windowed_done",
        &format!(
            "n_snps={n_snps} kept_n={kept_n} dropped={} keep_ratio={:.6}",
            n_snps.saturating_sub(kept_n),
            (kept_n as f64) / (n_snps as f64),
        ),
    );

    emit_bed_logic_meta_timing(
        "prepare_bed_logic_meta_owned_for_stats_samples_windowed",
        open_iter_secs,
        0.0,
        0.0,
        row_stats_secs,
        0.0,
        0.0,
        total_t0.elapsed().as_secs_f64(),
        n_samples,
        stats_n_samples,
        n_snps,
        kept_n,
        true,
    );

    Ok(PreparedBedLogicMetaOwned {
        site_keep,
        row_flip: row_flip_keep,
        row_source_indices,
        missing_rate: missing_rate_keep,
        maf: maf_keep,
        sites: Vec::new(),
        n_samples,
        n_snps_total: n_snps,
        bytes_per_snp,
    })
}

pub(crate) fn prepare_bed_logic_meta_owned_for_stats_samples(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    stats_sample_indices: Option<&[usize]>,
    stats_only: bool,
) -> Result<PreparedBedLogicMetaOwned, String> {
    prepare_bed_logic_meta_owned_for_stats_samples_with_mmap_window(
        prefix,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
        stats_sample_indices,
        stats_only,
        None,
    )
}

pub(crate) fn prepare_bed_logic_meta_owned_for_stats_samples_with_mmap_window(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    stats_sample_indices: Option<&[usize]>,
    stats_only: bool,
    mmap_window_mb: Option<usize>,
) -> Result<PreparedBedLogicMetaOwned, String> {
    let total_t0 = Instant::now();
    let mut read_bim_secs = 0.0_f64;
    if !(0.0..=0.5).contains(&maf_threshold) {
        return Err("maf_threshold must be within [0, 0.5]".to_string());
    }
    if !(0.0..=1.0).contains(&max_missing_rate) {
        return Err("max_missing_rate must be within [0, 1.0]".to_string());
    }
    if !(0.0..=1.0).contains(&het_threshold) {
        return Err("het_threshold must be within [0, 1.0]".to_string());
    }

    let mut bed_prefix = prefix.to_string();
    let lower = bed_prefix.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        bed_prefix.truncate(bed_prefix.len() - 4);
    }

    if let Some(window_mb) = mmap_window_mb {
        if stats_only && !snps_only {
            return prepare_bed_logic_meta_owned_for_stats_samples_sparse_windowed(
                &bed_prefix,
                maf_threshold,
                max_missing_rate,
                het_threshold,
                stats_sample_indices,
                window_mb,
                1usize,
                None,
                0usize,
                0usize,
            );
        }
    }

    let read_fam_t0 = Instant::now();
    let samples = core::read_fam(&bed_prefix).map_err(|e| e.to_string())?;
    let read_fam_secs = read_fam_t0.elapsed().as_secs_f64();
    let n_samples = samples.len();
    if n_samples == 0 {
        return Err("no samples found in PLINK input".to_string());
    }
    let stats_sample_indices = stats_sample_indices.unwrap_or(&[]);
    if !stats_sample_indices.is_empty() && stats_sample_indices.iter().any(|&idx| idx >= n_samples)
    {
        return Err("selected sample index out of range for BED logic preparation".to_string());
    }
    let stats_identity = stats_sample_indices.is_empty()
        || (stats_sample_indices.len() == n_samples
            && sample_indices_are_identity(stats_sample_indices));
    let stats_n_samples = if stats_identity {
        n_samples
    } else {
        stats_sample_indices.len()
    };
    let stats_excluded_sample_indices = if stats_identity {
        None
    } else {
        precompute_excluded_sample_indices(n_samples, stats_sample_indices)
    };
    if stats_n_samples == 0 {
        return Err("selected sample set for BED logic preparation is empty".to_string());
    }
    emit_gfreader_rss_debug(
        "prepare_bed_logic_meta/read_fam",
        &format!("prefix={bed_prefix} n_samples={n_samples} stats_n_samples={stats_n_samples}"),
    );

    let mmap_t0 = Instant::now();
    let bed_path = format!("{bed_prefix}.bed");
    let bed_file = File::open(&bed_path).map_err(|e| format!("failed to open {bed_path}: {e}"))?;
    let mmap =
        unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("failed to mmap {bed_path}: {e}"))?;
    let mmap_secs = mmap_t0.elapsed().as_secs_f64();
    if mmap.len() < 3 {
        return Err("BED too small".to_string());
    }
    if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
        return Err("Only SNP-major BED supported".to_string());
    }

    let bytes_per_snp = (n_samples + 3) / 4;
    let data_len = mmap.len() - 3;
    if bytes_per_snp == 0 || data_len % bytes_per_snp != 0 {
        return Err(format!(
            "invalid BED payload length: data_len={data_len}, bytes_per_snp={bytes_per_snp}"
        ));
    }
    let n_snps_bed = data_len / bytes_per_snp;
    if n_snps_bed == 0 {
        return Err("no SNP sites found in PLINK BED input".to_string());
    }

    let load_sites_all = snps_only || !stats_only;
    let sites_all = if load_sites_all {
        let read_bim_t0 = Instant::now();
        let sites_all = core::read_bim(&bed_prefix).map_err(|e| e.to_string())?;
        read_bim_secs = read_bim_t0.elapsed().as_secs_f64();
        let n_snps_bim = sites_all.len();
        if n_snps_bim == 0 {
            return Err("no SNP sites found in PLINK BIM input".to_string());
        }
        if n_snps_bed != n_snps_bim {
            return Err(format!(
                "BED/BIM SNP count mismatch: bed={n_snps_bed}, bim={n_snps_bim}"
            ));
        }
        emit_gfreader_rss_debug(
            "prepare_bed_logic_meta/read_bim",
            &format!("prefix={bed_prefix} n_snps={n_snps_bim}"),
        );
        Some(sites_all)
    } else {
        None
    };
    let n_snps = n_snps_bed;

    let packed_full = &mmap[3..];
    emit_gfreader_rss_debug(
        "prepare_bed_logic_meta/mmap_ready",
        &format!(
            "prefix={bed_prefix} n_samples={n_samples} n_snps={n_snps} bytes_per_snp={bytes_per_snp} payload={}",
            format_debug_bytes_local(packed_full.len() as u64),
        ),
    );

    let apply_het_filter = het_threshold > 0.0_f32;
    let row_stats_t0 = Instant::now();
    let keep_flip_stats: Vec<(bool, f32, f32)> = packed_full
        .par_chunks(bytes_per_snp)
        .enumerate()
        .map(|(i, row)| {
            let (missing, het, hom_alt) = if stats_identity {
                count_packed_row_counts(row, n_samples)
            } else {
                count_packed_row_counts_selected_with_excluded(
                    row,
                    n_samples,
                    stats_sample_indices,
                    stats_excluded_sample_indices.as_deref(),
                )
            };
            let non_missing = stats_n_samples.saturating_sub(missing);
            let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
            let het_count = if apply_het_filter { het } else { 0usize };
            let (missing_rate, _maf, _std) =
                packed_row_stats_from_counts(stats_n_samples, non_missing, alt_sum);
            let alt_freq = if non_missing > 0 {
                (alt_sum as f32) / (2.0_f32 * non_missing as f32)
            } else {
                0.0_f32
            };
            let pass_num = if missing_rate > max_missing_rate {
                false
            } else if non_missing == 0 {
                maf_threshold <= 0.0_f32
            } else if apply_het_filter
                && ((het_count as f64) / (non_missing as f64)) > (het_threshold as f64)
            {
                false
            } else {
                alt_freq.min(1.0_f32 - alt_freq) >= maf_threshold
            };
            let pass_snp = if snps_only {
                let site = &sites_all
                    .as_ref()
                    .expect("snps_only requires BIM metadata for allele filtering")[i];
                is_simple_snp_allele(&site.ref_allele) && is_simple_snp_allele(&site.alt_allele)
            } else {
                true
            };
            let keep = pass_num && pass_snp;
            (keep, missing_rate, alt_freq)
        })
        .collect();
    let row_stats_secs = row_stats_t0.elapsed().as_secs_f64();
    let site_keep_t0 = Instant::now();
    let site_keep: Vec<bool> = keep_flip_stats.iter().map(|(keep, _, _)| *keep).collect();
    if site_keep.len() != n_snps {
        return Err(format!(
            "internal error: site_keep rows {} != n_snps {n_snps}",
            site_keep.len()
        ));
    }

    let kept_n = site_keep.iter().filter(|&&x| x).count();
    let site_keep_secs = site_keep_t0.elapsed().as_secs_f64();
    if kept_n == 0 {
        return Err(
            "No SNPs left after packed BED filtering. Please relax thresholds.".to_string(),
        );
    }
    emit_gfreader_rss_debug(
        "prepare_bed_logic_meta/site_keep_ready",
        &format!(
            "n_snps={n_snps} kept_n={kept_n} dropped={} keep_ratio={:.6}",
            n_snps.saturating_sub(kept_n),
            (kept_n as f64) / (n_snps as f64),
        ),
    );

    let pack_kept_t0 = Instant::now();
    let mut sites_keep: Vec<core::SiteInfo> = if stats_only {
        Vec::new()
    } else {
        Vec::with_capacity(kept_n)
    };
    let mut row_flip_keep = Vec::<bool>::with_capacity(kept_n);
    let mut row_source_indices = Vec::<usize>::with_capacity(kept_n);
    let mut missing_rate_keep = Vec::<f32>::with_capacity(kept_n);
    let mut maf_keep = Vec::<f32>::with_capacity(kept_n);
    if let Some(sites_all) = sites_all {
        for (i, site) in sites_all.into_iter().enumerate() {
            if site_keep[i] {
                let (_keep, missing_rate, alt_freq) = keep_flip_stats[i];
                if !stats_only {
                    sites_keep.push(site);
                }
                row_flip_keep.push(false);
                row_source_indices.push(i);
                missing_rate_keep.push(missing_rate);
                maf_keep.push(alt_freq);
            }
        }
    } else {
        for (i, (keep, missing_rate, alt_freq)) in keep_flip_stats.iter().copied().enumerate() {
            if keep {
                row_flip_keep.push(false);
                row_source_indices.push(i);
                missing_rate_keep.push(missing_rate);
                maf_keep.push(alt_freq);
            }
        }
    }
    let pack_kept_secs = pack_kept_t0.elapsed().as_secs_f64();
    emit_bed_logic_meta_timing(
        "prepare_bed_logic_meta_owned_for_stats_samples",
        read_fam_secs,
        read_bim_secs,
        mmap_secs,
        row_stats_secs,
        site_keep_secs,
        pack_kept_secs,
        total_t0.elapsed().as_secs_f64(),
        n_samples,
        stats_n_samples,
        n_snps,
        kept_n,
        stats_only,
    );

    Ok(PreparedBedLogicMetaOwned {
        site_keep,
        row_flip: row_flip_keep,
        row_source_indices,
        missing_rate: missing_rate_keep,
        maf: maf_keep,
        sites: sites_keep,
        n_samples,
        n_snps_total: n_snps,
        bytes_per_snp,
    })
}

pub(crate) fn prepare_bed_logic_meta_owned(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
) -> Result<PreparedBedLogicMetaOwned, String> {
    prepare_bed_logic_meta_owned_for_stats_samples(
        prefix,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
        None,
        false,
    )
}

pub(crate) fn prepare_bed_logic_meta_owned_for_stats_samples_pure_line(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    snps_only: bool,
    stats_sample_indices: Option<&[usize]>,
) -> Result<PreparedBedLogicMetaOwned, String> {
    if !(0.0..=0.5).contains(&maf_threshold) {
        return Err("maf_threshold must be within [0, 0.5]".to_string());
    }
    if !(0.0..=1.0).contains(&max_missing_rate) {
        return Err("max_missing_rate must be within [0, 1.0]".to_string());
    }

    let mut bed_prefix = prefix.to_string();
    let lower = bed_prefix.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        bed_prefix.truncate(bed_prefix.len() - 4);
    }

    let samples = core::read_fam(&bed_prefix).map_err(|e| e.to_string())?;
    let n_samples = samples.len();
    if n_samples == 0 {
        return Err("no samples found in PLINK input".to_string());
    }
    let stats_sample_indices = stats_sample_indices.unwrap_or(&[]);
    if !stats_sample_indices.is_empty() && stats_sample_indices.iter().any(|&idx| idx >= n_samples)
    {
        return Err(
            "selected sample index out of range for BED pure-line logic preparation".to_string(),
        );
    }
    let stats_identity = stats_sample_indices.is_empty()
        || (stats_sample_indices.len() == n_samples
            && sample_indices_are_identity(stats_sample_indices));
    let stats_n_samples = if stats_identity {
        n_samples
    } else {
        stats_sample_indices.len()
    };
    let stats_excluded_sample_indices = if stats_identity {
        None
    } else {
        precompute_excluded_sample_indices(n_samples, stats_sample_indices)
    };
    if stats_n_samples == 0 {
        return Err("selected sample set for BED pure-line logic preparation is empty".to_string());
    }

    let sites_all = core::read_bim(&bed_prefix).map_err(|e| e.to_string())?;
    let n_snps = sites_all.len();
    if n_snps == 0 {
        return Err("no SNP sites found in PLINK BIM input".to_string());
    }

    let bed_path = format!("{bed_prefix}.bed");
    let bed_file = File::open(&bed_path).map_err(|e| format!("failed to open {bed_path}: {e}"))?;
    let mmap =
        unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("failed to mmap {bed_path}: {e}"))?;
    if mmap.len() < 3 {
        return Err("BED too small".to_string());
    }
    if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
        return Err("Only SNP-major BED supported".to_string());
    }

    let bytes_per_snp = (n_samples + 3) / 4;
    let data_len = mmap.len() - 3;
    if bytes_per_snp == 0 || data_len % bytes_per_snp != 0 {
        return Err(format!(
            "invalid BED payload length: data_len={data_len}, bytes_per_snp={bytes_per_snp}"
        ));
    }
    let n_snps_bed = data_len / bytes_per_snp;
    if n_snps_bed != n_snps {
        return Err(format!(
            "BED/BIM SNP count mismatch: bed={n_snps_bed}, bim={n_snps}"
        ));
    }

    let packed_full = &mmap[3..];
    let keep_flip_stats: Vec<(bool, f32, f32)> = packed_full
        .par_chunks(bytes_per_snp)
        .enumerate()
        .map(|(i, row)| {
            let (logic_missing, hom_alt) = if stats_identity {
                count_packed_row_pure_line_counts(row, n_samples)
            } else {
                count_packed_row_pure_line_counts_selected_with_excluded(
                    row,
                    n_samples,
                    stats_sample_indices,
                    stats_excluded_sample_indices.as_deref(),
                )
            };
            let (missing_rate, _maf, _std) =
                packed_row_stats_from_counts_pure_line(stats_n_samples, logic_missing, hom_alt);
            let usable_homo = stats_n_samples.saturating_sub(logic_missing.min(stats_n_samples));
            let alt_freq = if usable_homo > 0 {
                (hom_alt as f32) / (usable_homo as f32)
            } else {
                0.0_f32
            };
            let pass_num = if missing_rate > max_missing_rate {
                false
            } else if usable_homo == 0 {
                false
            } else {
                alt_freq.min(1.0_f32 - alt_freq) >= maf_threshold
            };
            let pass_snp = if snps_only {
                is_simple_snp_allele(&sites_all[i].ref_allele)
                    && is_simple_snp_allele(&sites_all[i].alt_allele)
            } else {
                true
            };
            let keep = pass_num && pass_snp;
            (keep, missing_rate, alt_freq)
        })
        .collect();
    let site_keep: Vec<bool> = keep_flip_stats.iter().map(|(keep, _, _)| *keep).collect();
    if site_keep.len() != n_snps {
        return Err(format!(
            "internal error: site_keep rows {} != n_snps {n_snps}",
            site_keep.len()
        ));
    }

    let kept_n = site_keep.iter().filter(|&&x| x).count();
    if kept_n == 0 {
        return Err(
            "No SNPs left after pure-line BED filtering. Please relax thresholds.".to_string(),
        );
    }

    let mut sites_keep = Vec::<core::SiteInfo>::with_capacity(kept_n);
    let mut row_flip_keep = Vec::<bool>::with_capacity(kept_n);
    let mut row_source_indices = Vec::<usize>::with_capacity(kept_n);
    let mut missing_rate_keep = Vec::<f32>::with_capacity(kept_n);
    let mut maf_keep = Vec::<f32>::with_capacity(kept_n);
    for (i, site) in sites_all.into_iter().enumerate() {
        if site_keep[i] {
            let (_keep, missing_rate, alt_freq) = keep_flip_stats[i];
            sites_keep.push(site);
            row_flip_keep.push(false);
            row_source_indices.push(i);
            missing_rate_keep.push(missing_rate);
            maf_keep.push(alt_freq);
        }
    }

    Ok(PreparedBedLogicMetaOwned {
        site_keep,
        row_flip: row_flip_keep,
        row_source_indices,
        missing_rate: missing_rate_keep,
        maf: maf_keep,
        sites: sites_keep,
        n_samples,
        n_snps_total: n_snps,
        bytes_per_snp,
    })
}

#[allow(dead_code)]
pub(crate) fn prepare_bed_logic_meta_owned_pure_line(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    snps_only: bool,
) -> Result<PreparedBedLogicMetaOwned, String> {
    prepare_bed_logic_meta_owned_for_stats_samples_pure_line(
        prefix,
        maf_threshold,
        max_missing_rate,
        snps_only,
        None,
    )
}

pub(crate) fn load_bed_2bit_packed_subset_owned_for_stats_samples(
    prefix: &str,
    site_keep: &[bool],
    stats_sample_indices: Option<&[usize]>,
) -> Result<PackedBedSubsetOwned, String> {
    let mut bed_prefix = prefix.to_string();
    let lower = bed_prefix.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        bed_prefix.truncate(bed_prefix.len() - 4);
    }

    let samples = core::read_fam(&bed_prefix).map_err(|e| e.to_string())?;
    let n_samples = samples.len();
    if n_samples == 0 {
        return Err("no samples found in PLINK input".to_string());
    }
    let stats_sample_indices = stats_sample_indices.unwrap_or(&[]);
    if !stats_sample_indices.is_empty() && stats_sample_indices.iter().any(|&idx| idx >= n_samples)
    {
        return Err("selected sample index out of range for packed BED subset stats".to_string());
    }
    let stats_identity = stats_sample_indices.is_empty()
        || (stats_sample_indices.len() == n_samples
            && sample_indices_are_identity(stats_sample_indices));
    let stats_n_samples = if stats_identity {
        n_samples
    } else {
        stats_sample_indices.len()
    };
    let stats_excluded_sample_indices = if stats_identity {
        None
    } else {
        precompute_excluded_sample_indices(n_samples, stats_sample_indices)
    };
    if stats_n_samples == 0 {
        return Err("selected sample set for packed BED subset stats is empty".to_string());
    }
    emit_gfreader_rss_debug(
        "load_bed_2bit_packed_subset_owned/read_fam",
        &format!("prefix={bed_prefix} n_samples={n_samples}"),
    );

    let bed_path = format!("{bed_prefix}.bed");
    let bed_file = File::open(&bed_path).map_err(|e| format!("failed to open {bed_path}: {e}"))?;
    let mmap =
        unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("failed to mmap {bed_path}: {e}"))?;
    if mmap.len() < 3 {
        return Err("BED too small".to_string());
    }
    if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
        return Err("Only SNP-major BED supported".to_string());
    }

    let bytes_per_snp = (n_samples + 3) / 4;
    let data_len = mmap.len() - 3;
    if bytes_per_snp == 0 || data_len % bytes_per_snp != 0 {
        return Err(format!(
            "invalid BED payload length: data_len={data_len}, bytes_per_snp={bytes_per_snp}"
        ));
    }
    let n_snps_total = data_len / bytes_per_snp;
    if site_keep.len() != n_snps_total {
        return Err(format!(
            "site_keep length mismatch: got {}, expected {n_snps_total}",
            site_keep.len()
        ));
    }

    let keep_idx: Vec<usize> = site_keep
        .iter()
        .enumerate()
        .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
        .collect();
    if keep_idx.is_empty() {
        return Err("No SNPs remained after applying site_keep mask.".to_string());
    }

    let packed_src = &mmap[3..];
    emit_gfreader_rss_debug(
        "load_bed_2bit_packed_subset_owned/mmap_ready",
        &format!(
            "prefix={bed_prefix} n_samples={n_samples} n_snps={n_snps_total} kept_n={} bytes_per_snp={bytes_per_snp} payload={}",
            keep_idx.len(),
            format_debug_bytes_local(packed_src.len() as u64),
        ),
    );

    let kept_n = keep_idx.len();
    let mut packed_keep = vec![0u8; kept_n * bytes_per_snp];
    let mut miss_keep = vec![0.0_f32; kept_n];
    let mut maf_keep = vec![0.0_f32; kept_n];
    let mut std_keep = vec![0.0_f32; kept_n];
    let mut row_flip_keep = vec![false; kept_n];

    if kept_n == n_snps_total {
        packed_keep = load_file_owned_range_exact(Path::new(&bed_path), 3, packed_src.len())?;
        packed_keep
            .par_chunks(bytes_per_snp)
            .zip(miss_keep.par_iter_mut())
            .zip(maf_keep.par_iter_mut())
            .zip(std_keep.par_iter_mut())
            .zip(row_flip_keep.par_iter_mut())
            .for_each(|((((row, miss_v), maf_v), std_v), row_flip_v)| {
                let (missing, het, hom_alt) = if stats_identity {
                    count_packed_row_counts(row, n_samples)
                } else {
                    count_packed_row_counts_selected_with_excluded(
                        row,
                        n_samples,
                        stats_sample_indices,
                        stats_excluded_sample_indices.as_deref(),
                    )
                };
                let non_missing = stats_n_samples.saturating_sub(missing);
                let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                let (miss, _maf, std) =
                    packed_row_stats_from_counts(stats_n_samples, non_missing, alt_sum);
                let af = if non_missing > 0 {
                    (alt_sum as f32) / (2.0_f32 * non_missing as f32)
                } else {
                    0.0_f32
                };
                *miss_v = miss;
                *maf_v = af.clamp(0.0, 1.0);
                *std_v = std;
                *row_flip_v = false;
            });
        emit_gfreader_rss_debug(
            "load_bed_2bit_packed_subset_owned/full_payload_loaded",
            &format!(
                "kept_n={kept_n} packed_bytes={} full_keep_fastpath=1",
                format_debug_bytes_local(packed_keep.len() as u64),
            ),
        );
        return Ok(PackedBedSubsetOwned {
            packed: packed_keep,
            missing_rate: miss_keep,
            maf: maf_keep,
            std_denom: std_keep,
            row_flip: row_flip_keep,
            n_samples,
            bytes_per_snp,
        });
    }

    packed_keep
        .par_chunks_mut(bytes_per_snp)
        .zip(miss_keep.par_iter_mut())
        .zip(maf_keep.par_iter_mut())
        .zip(std_keep.par_iter_mut())
        .zip(row_flip_keep.par_iter_mut())
        .zip(keep_idx.par_iter())
        .for_each(
            |(((((dst_row, miss_v), maf_v), std_v), row_flip_v), &src_row)| {
                let src_off = src_row * bytes_per_snp;
                let row = &packed_src[src_off..src_off + bytes_per_snp];
                dst_row.copy_from_slice(row);

                let (missing, het, hom_alt) = if stats_identity {
                    count_packed_row_counts(row, n_samples)
                } else {
                    count_packed_row_counts_selected_with_excluded(
                        row,
                        n_samples,
                        stats_sample_indices,
                        stats_excluded_sample_indices.as_deref(),
                    )
                };
                let non_missing = stats_n_samples.saturating_sub(missing);
                let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                let (miss, _maf, std) =
                    packed_row_stats_from_counts(stats_n_samples, non_missing, alt_sum);
                let af = if non_missing > 0 {
                    (alt_sum as f32) / (2.0_f32 * non_missing as f32)
                } else {
                    0.0_f32
                };
                *miss_v = miss;
                *maf_v = af.clamp(0.0, 1.0);
                *std_v = std;
                *row_flip_v = false;
            },
        );

    emit_gfreader_rss_debug(
        "load_bed_2bit_packed_subset_owned/subset_copy_done",
        &format!(
            "kept_n={kept_n} packed_bytes={} maf_bytes={} row_flip_bytes={}",
            format_debug_bytes_local(packed_keep.len() as u64),
            format_debug_bytes_local((maf_keep.len() * std::mem::size_of::<f32>()) as u64),
            format_debug_bytes_local((row_flip_keep.len() * std::mem::size_of::<bool>()) as u64),
        ),
    );

    Ok(PackedBedSubsetOwned {
        packed: packed_keep,
        missing_rate: miss_keep,
        maf: maf_keep,
        std_denom: std_keep,
        row_flip: row_flip_keep,
        n_samples,
        bytes_per_snp,
    })
}

pub(crate) fn load_bed_2bit_packed_subset_owned(
    prefix: &str,
    site_keep: &[bool],
) -> Result<PackedBedSubsetOwned, String> {
    load_bed_2bit_packed_subset_owned_for_stats_samples(prefix, site_keep, None)
}

pub(crate) fn load_bed_2bit_packed_subset_owned_for_stats_samples_pure_line(
    prefix: &str,
    site_keep: &[bool],
    stats_sample_indices: Option<&[usize]>,
) -> Result<PackedBedSubsetOwned, String> {
    let mut bed_prefix = prefix.to_string();
    let lower = bed_prefix.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        bed_prefix.truncate(bed_prefix.len() - 4);
    }

    let samples = core::read_fam(&bed_prefix).map_err(|e| e.to_string())?;
    let n_samples = samples.len();
    if n_samples == 0 {
        return Err("no samples found in PLINK input".to_string());
    }
    let stats_sample_indices = stats_sample_indices.unwrap_or(&[]);
    if !stats_sample_indices.is_empty() && stats_sample_indices.iter().any(|&idx| idx >= n_samples)
    {
        return Err("selected sample index out of range for packed BED subset stats".to_string());
    }
    let stats_identity = stats_sample_indices.is_empty()
        || (stats_sample_indices.len() == n_samples
            && sample_indices_are_identity(stats_sample_indices));
    let stats_n_samples = if stats_identity {
        n_samples
    } else {
        stats_sample_indices.len()
    };
    let stats_excluded_sample_indices = if stats_identity {
        None
    } else {
        precompute_excluded_sample_indices(n_samples, stats_sample_indices)
    };
    if stats_n_samples == 0 {
        return Err("selected sample set for packed BED subset stats is empty".to_string());
    }

    let bed_path = format!("{bed_prefix}.bed");
    let bed_file = File::open(&bed_path).map_err(|e| format!("failed to open {bed_path}: {e}"))?;
    let mmap =
        unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("failed to mmap {bed_path}: {e}"))?;
    if mmap.len() < 3 {
        return Err("BED too small".to_string());
    }
    if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
        return Err("Only SNP-major BED supported".to_string());
    }

    let bytes_per_snp = (n_samples + 3) / 4;
    let data_len = mmap.len() - 3;
    if bytes_per_snp == 0 || data_len % bytes_per_snp != 0 {
        return Err(format!(
            "invalid BED payload length: data_len={data_len}, bytes_per_snp={bytes_per_snp}"
        ));
    }
    let n_snps_total = data_len / bytes_per_snp;
    if site_keep.len() != n_snps_total {
        return Err(format!(
            "site_keep length mismatch: got {}, expected {n_snps_total}",
            site_keep.len()
        ));
    }

    let keep_idx: Vec<usize> = site_keep
        .iter()
        .enumerate()
        .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
        .collect();
    if keep_idx.is_empty() {
        return Err("No SNPs remained after applying site_keep mask.".to_string());
    }

    let packed_src = &mmap[3..];
    let kept_n = keep_idx.len();
    let mut packed_keep = vec![0u8; kept_n * bytes_per_snp];
    let mut miss_keep = vec![0.0_f32; kept_n];
    let mut maf_keep = vec![0.0_f32; kept_n];
    let mut std_keep = vec![0.0_f32; kept_n];
    let mut row_flip_keep = vec![false; kept_n];

    if kept_n == n_snps_total {
        packed_keep = load_file_owned_range_exact(Path::new(&bed_path), 3, packed_src.len())?;
        packed_keep
            .par_chunks(bytes_per_snp)
            .zip(miss_keep.par_iter_mut())
            .zip(maf_keep.par_iter_mut())
            .zip(std_keep.par_iter_mut())
            .zip(row_flip_keep.par_iter_mut())
            .for_each(|((((row, miss_v), maf_v), std_v), row_flip_v)| {
                let (logic_missing, hom_alt) = if stats_identity {
                    count_packed_row_pure_line_counts(row, n_samples)
                } else {
                    count_packed_row_pure_line_counts_selected_with_excluded(
                        row,
                        n_samples,
                        stats_sample_indices,
                        stats_excluded_sample_indices.as_deref(),
                    )
                };
                let (miss, _maf, std) =
                    packed_row_stats_from_counts_pure_line(stats_n_samples, logic_missing, hom_alt);
                let usable_homo =
                    stats_n_samples.saturating_sub(logic_missing.min(stats_n_samples));
                let af = if usable_homo > 0 {
                    (hom_alt as f32) / (usable_homo as f32)
                } else {
                    0.0_f32
                };
                *miss_v = miss;
                *maf_v = af.clamp(0.0, 1.0);
                *std_v = std;
                *row_flip_v = false;
            });
        emit_gfreader_rss_debug(
            "load_bed_2bit_packed_subset_owned_pure_line/full_payload_loaded",
            &format!(
                "kept_n={kept_n} packed_bytes={} full_keep_fastpath=1",
                format_debug_bytes_local(packed_keep.len() as u64),
            ),
        );
        return Ok(PackedBedSubsetOwned {
            packed: packed_keep,
            missing_rate: miss_keep,
            maf: maf_keep,
            std_denom: std_keep,
            row_flip: row_flip_keep,
            n_samples,
            bytes_per_snp,
        });
    }

    packed_keep
        .par_chunks_mut(bytes_per_snp)
        .zip(miss_keep.par_iter_mut())
        .zip(maf_keep.par_iter_mut())
        .zip(std_keep.par_iter_mut())
        .zip(row_flip_keep.par_iter_mut())
        .zip(keep_idx.par_iter())
        .for_each(
            |(((((dst_row, miss_v), maf_v), std_v), row_flip_v), &src_row)| {
                let src_off = src_row * bytes_per_snp;
                let row = &packed_src[src_off..src_off + bytes_per_snp];
                dst_row.copy_from_slice(row);

                let (logic_missing, hom_alt) = if stats_identity {
                    count_packed_row_pure_line_counts(row, n_samples)
                } else {
                    count_packed_row_pure_line_counts_selected_with_excluded(
                        row,
                        n_samples,
                        stats_sample_indices,
                        stats_excluded_sample_indices.as_deref(),
                    )
                };
                let (miss, _maf, std) =
                    packed_row_stats_from_counts_pure_line(stats_n_samples, logic_missing, hom_alt);
                let usable_homo =
                    stats_n_samples.saturating_sub(logic_missing.min(stats_n_samples));
                let af = if usable_homo > 0 {
                    (hom_alt as f32) / (usable_homo as f32)
                } else {
                    0.0_f32
                };
                *miss_v = miss;
                *maf_v = af.clamp(0.0, 1.0);
                *std_v = std;
                *row_flip_v = false;
            },
        );

    Ok(PackedBedSubsetOwned {
        packed: packed_keep,
        missing_rate: miss_keep,
        maf: maf_keep,
        std_denom: std_keep,
        row_flip: row_flip_keep,
        n_samples,
        bytes_per_snp,
    })
}

#[allow(dead_code)]
pub(crate) fn load_bed_2bit_packed_subset_owned_pure_line(
    prefix: &str,
    site_keep: &[bool],
) -> Result<PackedBedSubsetOwned, String> {
    load_bed_2bit_packed_subset_owned_for_stats_samples_pure_line(prefix, site_keep, None)
}

pub(crate) fn prepare_bed_2bit_packed_owned(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
) -> Result<PreparedBedPackedOwned, String> {
    let scanned = prepare_bed_logic_meta_owned(
        prefix,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
    )?;
    let subset = load_bed_2bit_packed_subset_owned(prefix, scanned.site_keep.as_slice())?;
    Ok(PreparedBedPackedOwned {
        packed: subset.packed,
        missing_rate: subset.missing_rate,
        maf: subset.maf,
        std_denom: subset.std_denom,
        row_flip: subset.row_flip,
        site_keep: scanned.site_keep,
        sites: scanned.sites,
        n_samples: scanned.n_samples,
        n_snps_total: scanned.n_snps_total,
        bytes_per_snp: scanned.bytes_per_snp,
    })
}

#[allow(dead_code)]
pub(crate) fn prepare_bed_2bit_packed_owned_pure_line(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    snps_only: bool,
) -> Result<PreparedBedPackedOwned, String> {
    let scanned =
        prepare_bed_logic_meta_owned_pure_line(prefix, maf_threshold, max_missing_rate, snps_only)?;
    let subset = load_bed_2bit_packed_subset_owned_pure_line(prefix, scanned.site_keep.as_slice())?;
    Ok(PreparedBedPackedOwned {
        packed: subset.packed,
        missing_rate: subset.missing_rate,
        maf: subset.maf,
        std_denom: subset.std_denom,
        row_flip: subset.row_flip,
        site_keep: scanned.site_keep,
        sites: scanned.sites,
        n_samples: scanned.n_samples,
        n_snps_total: scanned.n_snps_total,
        bytes_per_snp: scanned.bytes_per_snp,
    })
}

#[allow(dead_code)]
pub(crate) fn prepare_bed_2bit_packed_owned_for_stats_samples(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    stats_sample_indices: Option<&[usize]>,
) -> Result<PreparedBedPackedOwned, String> {
    let scanned = prepare_bed_logic_meta_owned_for_stats_samples(
        prefix,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
        stats_sample_indices,
        false,
    )?;
    let subset = load_bed_2bit_packed_subset_owned_for_stats_samples(
        prefix,
        scanned.site_keep.as_slice(),
        stats_sample_indices,
    )?;
    Ok(PreparedBedPackedOwned {
        packed: subset.packed,
        missing_rate: subset.missing_rate,
        maf: subset.maf,
        std_denom: subset.std_denom,
        row_flip: subset.row_flip,
        site_keep: scanned.site_keep,
        sites: scanned.sites,
        n_samples: scanned.n_samples,
        n_snps_total: scanned.n_snps_total,
        bytes_per_snp: scanned.bytes_per_snp,
    })
}

#[allow(dead_code)]
pub(crate) fn prepare_bed_2bit_packed_owned_for_stats_samples_pure_line(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    snps_only: bool,
    stats_sample_indices: Option<&[usize]>,
) -> Result<PreparedBedPackedOwned, String> {
    let scanned = prepare_bed_logic_meta_owned_for_stats_samples_pure_line(
        prefix,
        maf_threshold,
        max_missing_rate,
        snps_only,
        stats_sample_indices,
    )?;
    let subset = load_bed_2bit_packed_subset_owned_for_stats_samples_pure_line(
        prefix,
        scanned.site_keep.as_slice(),
        stats_sample_indices,
    )?;
    Ok(PreparedBedPackedOwned {
        packed: subset.packed,
        missing_rate: subset.missing_rate,
        maf: subset.maf,
        std_denom: subset.std_denom,
        row_flip: subset.row_flip,
        site_keep: scanned.site_keep,
        sites: scanned.sites,
        n_samples: scanned.n_samples,
        n_snps_total: scanned.n_snps_total,
        bytes_per_snp: scanned.bytes_per_snp,
    })
}

#[pyfunction]
pub fn scan_bed_2bit_packed_stats<'py>(
    py: Python<'py>,
    prefix: String,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<f32>>,
    usize,
)> {
    let mut bed_prefix = prefix;
    let lower = bed_prefix.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        bed_prefix.truncate(bed_prefix.len() - 4);
    }
    let (missing_rate, maf, std_denom, row_flip, het_rate, n_samples, n_snps) = py
        .detach(move || -> Result<
            (
                Vec<f32>,
                Vec<f32>,
                Vec<f32>,
                Vec<bool>,
                Vec<f32>,
                usize,
                usize,
            ),
            String,
        > {
            let samples = core::read_fam(&bed_prefix).map_err(|e| e.to_string())?;
            let n_samples = samples.len();
            if n_samples == 0 {
                return Err("no samples found in PLINK input".to_string());
            }
            emit_gfreader_rss_debug(
                "scan_bed_2bit_packed_stats/read_fam",
                &format!("prefix={bed_prefix} n_samples={n_samples}"),
            );

            let bed_path = format!("{bed_prefix}.bed");
            let bed_file = File::open(&bed_path)
                .map_err(|e| format!("failed to open {bed_path}: {e}"))?;
            let mmap = unsafe { Mmap::map(&bed_file) }
                .map_err(|e| format!("failed to mmap {bed_path}: {e}"))?;
            if mmap.len() < 3 {
                return Err("BED too small".to_string());
            }
            if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
                return Err("Only SNP-major BED supported".to_string());
            }

            let bytes_per_snp = (n_samples + 3) / 4;
            let data_len = mmap.len() - 3;
            if bytes_per_snp == 0 || data_len % bytes_per_snp != 0 {
                return Err(format!(
                    "invalid BED payload length: data_len={data_len}, bytes_per_snp={bytes_per_snp}"
                ));
            }
            let n_snps = data_len / bytes_per_snp;
            let packed_src = &mmap[3..];
            emit_gfreader_rss_debug(
                "scan_bed_2bit_packed_stats/mmap_ready",
                &format!(
                    "prefix={bed_prefix} n_samples={n_samples} n_snps={n_snps} bytes_per_snp={bytes_per_snp} payload={}",
                    format_debug_bytes_local(packed_src.len() as u64),
                ),
            );

            let mut missing_rate: Vec<f32> = vec![0.0; n_snps];
            let mut maf: Vec<f32> = vec![0.0; n_snps];
            let mut std_denom: Vec<f32> = vec![0.0; n_snps];
            let mut row_flip: Vec<bool> = vec![false; n_snps];
            let mut het_rate: Vec<f32> = vec![0.0; n_snps];

            missing_rate
                .par_iter_mut()
                .zip(maf.par_iter_mut())
                .zip(std_denom.par_iter_mut())
                .zip(row_flip.par_iter_mut())
                .zip(het_rate.par_iter_mut())
                .enumerate()
                .for_each(
                    |(snp_idx, ((((miss_dst, maf_dst), std_dst), row_flip_dst), het_dst))| {
                        let row =
                            &packed_src[snp_idx * bytes_per_snp..(snp_idx + 1) * bytes_per_snp];
                        let (missing, het, hom_alt) = count_packed_row_counts(row, n_samples);
                        let non_missing = n_samples.saturating_sub(missing);
                        let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));

                        *miss_dst = (missing as f32) / (n_samples as f32);
                        if non_missing > 0 {
                            let p_alt = alt_sum as f32 / (2.0_f32 * non_missing as f32);
                            let maf_v = p_alt.min(1.0_f32 - p_alt);
                            *maf_dst = maf_v;
                            let d = (2.0_f32 * p_alt * (1.0_f32 - p_alt)).sqrt();
                            *std_dst = if d.is_finite() { d } else { 0.0_f32 };
                            *row_flip_dst = p_alt > 0.5_f32;
                            *het_dst = (het as f32) / (non_missing as f32);
                        } else {
                            *maf_dst = 0.0_f32;
                            *std_dst = 0.0_f32;
                            *row_flip_dst = false;
                            *het_dst = 0.0_f32;
                        }
                    },
                );

            let stats_bytes = ((missing_rate.len() + maf.len() + std_denom.len() + het_rate.len())
                * std::mem::size_of::<f32>()
                + row_flip.len() * std::mem::size_of::<bool>()) as u64;
            emit_gfreader_rss_debug(
                "scan_bed_2bit_packed_stats/row_stats_ready",
                &format!(
                    "n_snps={n_snps} stats_bytes={} arrays=4xf32+1xbool",
                    format_debug_bytes_local(stats_bytes),
                ),
            );

            Ok((
                missing_rate,
                maf,
                std_denom,
                row_flip,
                het_rate,
                n_samples,
                n_snps,
            ))
        })
        .map_err(PyRuntimeError::new_err)?;

    if maf.len() != n_snps {
        return Err(PyRuntimeError::new_err(
            "internal error: stats rows mismatch after BED scan",
        ));
    }
    #[allow(deprecated)]
    let miss_arr = PyArray1::from_owned_array(py, Array1::from_vec(missing_rate)).into_bound();
    #[allow(deprecated)]
    let maf_arr = PyArray1::from_owned_array(py, Array1::from_vec(maf)).into_bound();
    #[allow(deprecated)]
    let denom_arr = PyArray1::from_owned_array(py, Array1::from_vec(std_denom)).into_bound();
    #[allow(deprecated)]
    let row_flip_arr = PyArray1::from_owned_array(py, Array1::from_vec(row_flip)).into_bound();
    #[allow(deprecated)]
    let het_arr = PyArray1::from_owned_array(py, Array1::from_vec(het_rate)).into_bound();
    Ok((
        miss_arr,
        maf_arr,
        denom_arr,
        row_flip_arr,
        het_arr,
        n_samples,
    ))
}

#[inline]
pub(crate) fn is_simple_snp_allele(a: &str) -> bool {
    let t = a.trim().to_ascii_uppercase();
    if t.len() != 1 {
        return false;
    }
    matches!(t.as_bytes()[0], b'A' | b'C' | b'G' | b'T')
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    maf_threshold=0.0,
    max_missing_rate=1.0,
    het_threshold=0.0,
    snps_only=false,
))]
pub fn prepare_bed_2bit_packed<'py>(
    py: Python<'py>,
    prefix: String,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
) -> PyResult<(
    Bound<'py, PyArray2<u8>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<bool>>,
    usize,
    usize,
)> {
    if !(0.0..=0.5).contains(&maf_threshold) {
        return Err(PyValueError::new_err(
            "maf_threshold must be within [0, 0.5]",
        ));
    }
    if !(0.0..=1.0).contains(&max_missing_rate) {
        return Err(PyValueError::new_err(
            "max_missing_rate must be within [0, 1.0]",
        ));
    }
    if !(0.0..=1.0).contains(&het_threshold) {
        return Err(PyValueError::new_err(
            "het_threshold must be within [0, 1.0]",
        ));
    }
    let prepared = py
        .detach(move || {
            prepare_bed_2bit_packed_owned(
                &prefix,
                maf_threshold,
                max_missing_rate,
                het_threshold,
                snps_only,
            )
        })
        .map_err(PyRuntimeError::new_err)?;
    let packed_keep = prepared.packed;
    let miss_keep = prepared.missing_rate;
    let maf_keep = prepared.maf;
    let std_keep = prepared.std_denom;
    let row_flip_keep = prepared.row_flip;
    let site_keep = prepared.site_keep;
    let n_samples = prepared.n_samples;
    let n_snps = prepared.n_snps_total;
    let bytes_per_snp = prepared.bytes_per_snp;

    let packed_mat = Array2::from_shape_vec((maf_keep.len(), bytes_per_snp), packed_keep)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    #[allow(deprecated)]
    let packed_arr = PyArray2::from_owned_array(py, packed_mat).into_bound();
    #[allow(deprecated)]
    let miss_arr = PyArray1::from_owned_array(py, Array1::from_vec(miss_keep)).into_bound();
    #[allow(deprecated)]
    let maf_arr = PyArray1::from_owned_array(py, Array1::from_vec(maf_keep)).into_bound();
    #[allow(deprecated)]
    let denom_arr = PyArray1::from_owned_array(py, Array1::from_vec(std_keep)).into_bound();
    #[allow(deprecated)]
    let row_flip_arr = PyArray1::from_owned_array(py, Array1::from_vec(row_flip_keep)).into_bound();
    #[allow(deprecated)]
    let site_keep_arr = PyArray1::from_owned_array(py, Array1::from_vec(site_keep)).into_bound();
    Ok((
        packed_arr,
        miss_arr,
        maf_arr,
        denom_arr,
        row_flip_arr,
        site_keep_arr,
        n_samples,
        n_snps,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    sample_indices=None,
    maf_threshold=0.0,
    max_missing_rate=1.0,
    het_threshold=0.0,
    snps_only=false,
    mmap_window_mb=None,
))]
pub fn prepare_bed_logic_meta_selected<'py>(
    py: Python<'py>,
    prefix: String,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    mmap_window_mb: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<bool>>,
    usize,
    usize,
)> {
    if !(0.0..=0.5).contains(&maf_threshold) {
        return Err(PyValueError::new_err(
            "maf_threshold must be within [0, 0.5]",
        ));
    }
    if !(0.0..=1.0).contains(&max_missing_rate) {
        return Err(PyValueError::new_err(
            "max_missing_rate must be within [0, 1.0]",
        ));
    }
    if !(0.0..=1.0).contains(&het_threshold) {
        return Err(PyValueError::new_err(
            "het_threshold must be within [0, 1.0]",
        ));
    }
    let total_t0 = Instant::now();

    let bed_prefix = normalize_plink_prefix_local(&prefix);
    let n_samples_full = core::read_fam(&bed_prefix)
        .map_err(PyRuntimeError::new_err)?
        .len();
    if n_samples_full == 0 {
        return Err(PyRuntimeError::new_err("no samples found in PLINK input"));
    }
    let sample_idx: Option<Vec<usize>> = if let Some(sample_indices) = sample_indices {
        let idx64 = sample_indices
            .as_slice()
            .map_err(|_| PyRuntimeError::new_err("sample_indices must be contiguous int64"))?;
        let mut out = Vec::with_capacity(idx64.len());
        for &sid in idx64 {
            if sid < 0 || (sid as usize) >= n_samples_full {
                return Err(PyValueError::new_err(format!(
                    "sample index out of range: {sid} for n_samples={n_samples_full}"
                )));
            }
            out.push(sid as usize);
        }
        Some(out)
    } else {
        None
    };

    let rust_core_t0 = Instant::now();
    let prepared = py
        .detach(move || {
            prepare_bed_logic_meta_owned_for_stats_samples_with_mmap_window(
                &bed_prefix,
                maf_threshold,
                max_missing_rate,
                het_threshold,
                snps_only,
                sample_idx.as_deref(),
                true, // stats_only: skip Vec<SiteInfo> allocation
                mmap_window_mb.filter(|&v| v > 0),
            )
        })
        .map_err(PyRuntimeError::new_err)?;
    let rust_core_secs = rust_core_t0.elapsed().as_secs_f64();

    let py_arrays_t0 = Instant::now();
    let row_idx: Vec<i64> = prepared
        .row_source_indices
        .iter()
        .map(|&v| v as i64)
        .collect();
    #[allow(deprecated)]
    let row_idx_arr = PyArray1::from_owned_array(py, Array1::from_vec(row_idx)).into_bound();
    #[allow(deprecated)]
    let miss_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(prepared.missing_rate)).into_bound();
    #[allow(deprecated)]
    let maf_arr = PyArray1::from_owned_array(py, Array1::from_vec(prepared.maf)).into_bound();
    #[allow(deprecated)]
    let row_flip_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(prepared.row_flip)).into_bound();
    #[allow(deprecated)]
    let site_keep_arr =
        PyArray1::from_owned_array(py, Array1::from_vec(prepared.site_keep)).into_bound();
    let py_arrays_secs = py_arrays_t0.elapsed().as_secs_f64();
    emit_bed_logic_meta_py_timing(
        "prepare_bed_logic_meta_selected_py",
        rust_core_secs,
        py_arrays_secs,
        total_t0.elapsed().as_secs_f64(),
        n_samples_full,
        prepared.n_snps_total,
        prepared.row_source_indices.len(),
    );
    Ok((
        row_idx_arr,
        miss_arr,
        maf_arr,
        row_flip_arr,
        site_keep_arr,
        prepared.n_samples,
        prepared.n_snps_total,
    ))
}

#[pyfunction]
#[pyo3(signature = (prefix, maf_threshold=None, max_missing_rate=None, fill_missing=None))]
pub fn load_bed_u8_matrix<'py>(
    py: Python<'py>,
    prefix: String,
    maf_threshold: Option<f32>,
    max_missing_rate: Option<f32>,
    fill_missing: Option<bool>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let mut bed_prefix = prefix;
    let lower = bed_prefix.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        bed_prefix.truncate(bed_prefix.len() - 4);
    }

    let maf = maf_threshold.unwrap_or(0.0);
    let miss = max_missing_rate.unwrap_or(1.0);
    let fill = fill_missing.unwrap_or(false);

    let mut it = BedSnpIter::new_with_fill(&bed_prefix, 0.0, 1.0, false, false, 0.02)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let n = it.n_samples();
    if n == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "no samples found in PLINK input",
        ));
    }

    let hint = it.sites.len().max(1);
    let mut data: Vec<u8> = Vec::with_capacity(hint.saturating_mul(n));
    let mut kept = 0usize;
    while let Some((mut row, mut site)) = it.next_snp_raw() {
        let keep = core::process_snp_row(
            &mut row,
            &mut site.ref_allele,
            &mut site.alt_allele,
            maf,
            miss,
            fill,
            false,
            0.02,
        );
        if !keep {
            continue;
        }
        for &g in row.iter() {
            let v = if !g.is_finite() || g < 0.0 {
                3_u8
            } else {
                g.round().clamp(0.0, 2.0) as u8
            };
            data.push(v);
        }
        kept += 1;
    }

    let mat = Array2::from_shape_vec((kept, n), data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    #[allow(deprecated)]
    Ok(PyArray2::from_owned_array(py, mat).into_bound())
}

struct BedMmapEngine {
    prefix: String,
    mmap: Mmap,
    n_samples: usize,
    n_snps: usize,
    bytes_per_snp: usize,
    cursor: usize,
}

impl BedMmapEngine {
    fn open(prefix: &str) -> Result<Self, String> {
        let bed_prefix = normalize_plink_prefix_local(prefix);
        if bed_prefix.is_empty() {
            return Err("prefix must not be empty".to_string());
        }

        let n_samples = core::read_fam(&bed_prefix)
            .map_err(|e| e.to_string())?
            .len();
        if n_samples == 0 {
            return Err("no samples found in PLINK FAM input".to_string());
        }
        let n_snps_bim = core::read_bim(&bed_prefix)
            .map_err(|e| e.to_string())?
            .len();

        let bed_path = format!("{bed_prefix}.bed");
        let file = File::open(&bed_path).map_err(|e| format!("failed to open {bed_path}: {e}"))?;
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| format!("failed to mmap {bed_path}: {e}"))?;
        #[cfg(unix)]
        let _ = mmap.advise(Advice::Sequential);
        if mmap.len() < 3 {
            return Err(format!("{bed_path}: file too small for BED header"));
        }
        if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
            return Err(format!(
                "{bed_path}: only SNP-major BED (0x6C 0x1B 0x01) is supported"
            ));
        }

        let bytes_per_snp = n_samples.div_ceil(4);
        if bytes_per_snp == 0 {
            return Err("invalid BED: bytes_per_snp is zero".to_string());
        }
        let payload = mmap.len() - 3;
        if payload % bytes_per_snp != 0 {
            return Err(format!(
                "{bed_path}: invalid BED payload length {}, bytes_per_snp={}",
                payload, bytes_per_snp
            ));
        }
        let n_snps = payload / bytes_per_snp;
        if n_snps != n_snps_bim {
            return Err(format!(
                "{bed_path}: BED/BIM SNP count mismatch: bed={n_snps}, bim={n_snps_bim}"
            ));
        }

        Ok(Self {
            prefix: bed_prefix,
            mmap,
            n_samples,
            n_snps,
            bytes_per_snp,
            cursor: 0,
        })
    }

    #[inline]
    fn wrapping_checksum(bytes: &[u8]) -> u64 {
        let mut acc = 0u64;
        let mut i = 0usize;
        while i + 8 <= bytes.len() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[i..i + 8]);
            acc = acc.wrapping_add(u64::from_le_bytes(b));
            i += 8;
        }
        if i < bytes.len() {
            acc = acc.wrapping_add(load_u64_le_partial(bytes, i));
        }
        acc
    }

    #[inline]
    fn reset(&mut self) {
        self.cursor = 0;
    }

    fn seek(&mut self, snp_index: usize) -> Result<(), String> {
        if snp_index > self.n_snps {
            return Err(format!(
                "snp_index out of range: {snp_index} > n_snps={}",
                self.n_snps
            ));
        }
        self.cursor = snp_index;
        Ok(())
    }

    fn row_bounds(&self, snp_idx: usize) -> Result<(usize, usize), String> {
        if snp_idx >= self.n_snps {
            return Err(format!(
                "snp_index out of range: {snp_idx} >= n_snps={}",
                self.n_snps
            ));
        }
        let start = 3usize
            .checked_add(
                snp_idx
                    .checked_mul(self.bytes_per_snp)
                    .ok_or_else(|| "BED row offset overflow".to_string())?,
            )
            .ok_or_else(|| "BED row offset overflow".to_string())?;
        let end = start
            .checked_add(self.bytes_per_snp)
            .ok_or_else(|| "BED row end overflow".to_string())?;
        if end > self.mmap.len() {
            return Err(format!(
                "BED row exceeds mmap length: end={}, mmap_len={}",
                end,
                self.mmap.len()
            ));
        }
        Ok((start, end))
    }

    fn row_slice(&self, snp_idx: usize) -> Result<&[u8], String> {
        let (start, end) = self.row_bounds(snp_idx)?;
        Ok(&self.mmap[start..end])
    }

    fn rows_range(
        &self,
        start_snp: usize,
        max_rows: usize,
    ) -> Result<(usize, usize, usize), String> {
        if start_snp > self.n_snps {
            return Err(format!(
                "start_snp out of range: {start_snp} > n_snps={}",
                self.n_snps
            ));
        }
        if start_snp == self.n_snps || max_rows == 0 {
            return Ok((3usize, 3usize, 0usize));
        }

        let end_snp = std::cmp::min(self.n_snps, start_snp.saturating_add(max_rows));
        let rows = end_snp.saturating_sub(start_snp);
        let start_byte = 3usize
            .checked_add(
                start_snp
                    .checked_mul(self.bytes_per_snp)
                    .ok_or_else(|| "BED byte offset overflow".to_string())?,
            )
            .ok_or_else(|| "BED byte offset overflow".to_string())?;
        let span = rows
            .checked_mul(self.bytes_per_snp)
            .ok_or_else(|| "BED row span overflow".to_string())?;
        let end_byte = start_byte
            .checked_add(span)
            .ok_or_else(|| "BED end byte overflow".to_string())?;
        if end_byte > self.mmap.len() {
            return Err(format!(
                "BED slice exceeds mmap length: end_byte={}, mmap_len={}",
                end_byte,
                self.mmap.len()
            ));
        }
        Ok((start_byte, end_byte, rows))
    }

    /// Internal zero-copy slice API (engine-only).
    fn get_rows_slice(&self, start_snp: usize, max_rows: usize) -> Result<(&[u8], usize), String> {
        let (start_byte, end_byte, rows) = self.rows_range(start_snp, max_rows)?;
        if rows == 0 {
            return Ok((&self.mmap[3..3], 0usize));
        }
        Ok((&self.mmap[start_byte..end_byte], rows))
    }

    fn next_rows_slice(&mut self, max_rows: usize) -> Result<(&[u8], usize), String> {
        let start = self.cursor;
        let (start_byte, end_byte, rows) = self.rows_range(start, max_rows)?;
        self.cursor = start.saturating_add(rows);
        if rows == 0 {
            return Ok((&self.mmap[3..3], 0usize));
        }
        Ok((&self.mmap[start_byte..end_byte], rows))
    }

    fn scan_rows_checksum(
        &mut self,
        max_rows: usize,
        block_rows: usize,
    ) -> Result<(u64, usize, usize), String> {
        let remaining = self.n_snps.saturating_sub(self.cursor);
        if remaining == 0 {
            return Ok((0u64, 0usize, 0usize));
        }
        let target = if max_rows == 0 {
            remaining
        } else {
            std::cmp::min(max_rows, remaining)
        };
        if target == 0 {
            return Ok((0u64, 0usize, 0usize));
        }

        let blk = std::cmp::max(1usize, block_rows);
        let mut scanned = 0usize;
        let mut blocks = 0usize;
        let mut checksum = 0u64;

        while scanned < target {
            let take = std::cmp::min(blk, target - scanned);
            let start_snp = self.cursor + scanned;
            let (slice, got_rows) = self.get_rows_slice(start_snp, take)?;
            if got_rows != take {
                return Err(format!(
                    "BED scan rows mismatch: got_rows={got_rows}, expected={take}"
                ));
            }
            checksum = checksum.wrapping_add(Self::wrapping_checksum(slice));
            scanned += take;
            blocks += 1;
        }

        self.cursor = self.cursor.saturating_add(scanned);
        Ok((checksum, scanned, blocks))
    }

    fn random_rows_checksum(&self, snp_indices: Vec<usize>) -> Result<u64, String> {
        let mut checksum = 0u64;
        for snp_idx in snp_indices.into_iter() {
            let row = self.row_slice(snp_idx)?;
            checksum = checksum.wrapping_add(Self::wrapping_checksum(row));
        }
        Ok(checksum)
    }

    fn scan_rows_qc_counts(
        &mut self,
        max_rows: usize,
        block_rows: usize,
        parallel: bool,
    ) -> Result<(u64, u64, u64, usize, usize), String> {
        let remaining = self.n_snps.saturating_sub(self.cursor);
        if remaining == 0 {
            return Ok((0u64, 0u64, 0u64, 0usize, 0usize));
        }
        let target = if max_rows == 0 {
            remaining
        } else {
            std::cmp::min(max_rows, remaining)
        };
        if target == 0 {
            return Ok((0u64, 0u64, 0u64, 0usize, 0usize));
        }

        let blk = std::cmp::max(1usize, block_rows);
        let mut scanned = 0usize;
        let mut blocks = 0usize;
        let mut missing_total = 0u64;
        let mut het_total = 0u64;
        let mut hom_alt_total = 0u64;
        let rows_per_task = 512usize;
        let task_bytes = std::cmp::max(
            self.bytes_per_snp,
            self.bytes_per_snp.saturating_mul(rows_per_task),
        );

        while scanned < target {
            let take = std::cmp::min(blk, target - scanned);
            let start_snp = self.cursor + scanned;
            let (block_bytes, got_rows) = self.get_rows_slice(start_snp, take)?;
            if got_rows != take {
                return Err(format!(
                    "BED scan rows mismatch: got_rows={got_rows}, expected={take}"
                ));
            }

            let (m, h, ha) = if parallel && take >= 128 {
                block_bytes
                    .par_chunks(task_bytes)
                    .map(|super_row| {
                        let mut m = 0u64;
                        let mut h = 0u64;
                        let mut ha = 0u64;
                        for row in super_row.chunks(self.bytes_per_snp) {
                            let (xm, xh, xha) = count_packed_row_counts(row, self.n_samples);
                            m = m.saturating_add(xm as u64);
                            h = h.saturating_add(xh as u64);
                            ha = ha.saturating_add(xha as u64);
                        }
                        (m, h, ha)
                    })
                    .reduce(
                        || (0u64, 0u64, 0u64),
                        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
                    )
            } else {
                let mut m = 0u64;
                let mut h = 0u64;
                let mut ha = 0u64;
                for row in block_bytes.chunks(self.bytes_per_snp) {
                    let (xm, xh, xha) = count_packed_row_counts(row, self.n_samples);
                    m = m.saturating_add(xm as u64);
                    h = h.saturating_add(xh as u64);
                    ha = ha.saturating_add(xha as u64);
                }
                (m, h, ha)
            };
            missing_total = missing_total.saturating_add(m);
            het_total = het_total.saturating_add(h);
            hom_alt_total = hom_alt_total.saturating_add(ha);
            scanned += take;
            blocks += 1;
        }

        self.cursor = self.cursor.saturating_add(scanned);
        Ok((missing_total, het_total, hom_alt_total, scanned, blocks))
    }

    fn scan_rows_filter_counts(
        &mut self,
        maf_threshold: f32,
        max_missing_rate: f32,
        het_threshold: f32,
        max_rows: usize,
        block_rows: usize,
        parallel: bool,
    ) -> Result<(u64, usize, usize), String> {
        if !(0.0..=0.5).contains(&maf_threshold) {
            return Err("maf_threshold must be within [0, 0.5]".to_string());
        }
        if !(0.0..=1.0).contains(&max_missing_rate) {
            return Err("max_missing_rate must be within [0, 1.0]".to_string());
        }
        if !(0.0..=1.0).contains(&het_threshold) {
            return Err("het_threshold must be within [0, 1.0]".to_string());
        }

        let remaining = self.n_snps.saturating_sub(self.cursor);
        if remaining == 0 {
            return Ok((0u64, 0usize, 0usize));
        }
        let target = if max_rows == 0 {
            remaining
        } else {
            std::cmp::min(max_rows, remaining)
        };
        if target == 0 {
            return Ok((0u64, 0usize, 0usize));
        }

        let blk = std::cmp::max(1usize, block_rows);
        let mut scanned = 0usize;
        let mut blocks = 0usize;
        let mut kept_total = 0u64;
        let apply_het_filter = het_threshold > 0.0_f32;
        let rows_per_task = 512usize;
        let task_bytes = std::cmp::max(
            self.bytes_per_snp,
            self.bytes_per_snp.saturating_mul(rows_per_task),
        );

        while scanned < target {
            let take = std::cmp::min(blk, target - scanned);
            let start_snp = self.cursor + scanned;
            let (block_bytes, got_rows) = self.get_rows_slice(start_snp, take)?;
            if got_rows != take {
                return Err(format!(
                    "BED scan rows mismatch: got_rows={got_rows}, expected={take}"
                ));
            }

            let kept_blk = if parallel && take >= 128 {
                block_bytes
                    .par_chunks(task_bytes)
                    .map(|super_row| {
                        let mut kept = 0u64;
                        for row in super_row.chunks(self.bytes_per_snp) {
                            let (missing, het, hom_alt) =
                                count_packed_row_counts(row, self.n_samples);
                            let non_missing = self.n_samples.saturating_sub(missing);
                            let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                            let het_count = if apply_het_filter { het } else { 0usize };
                            let (keep, _flip) = evaluate_packed_row_keep_and_flip(
                                self.n_samples,
                                non_missing,
                                alt_sum,
                                het_count,
                                maf_threshold,
                                max_missing_rate,
                                apply_het_filter,
                                het_threshold,
                            );
                            if keep {
                                kept = kept.saturating_add(1);
                            }
                        }
                        kept
                    })
                    .sum::<u64>()
            } else {
                let mut kept = 0u64;
                for row in block_bytes.chunks(self.bytes_per_snp) {
                    let (missing, het, hom_alt) = count_packed_row_counts(row, self.n_samples);
                    let non_missing = self.n_samples.saturating_sub(missing);
                    let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                    let het_count = if apply_het_filter { het } else { 0usize };
                    let (keep, _flip) = evaluate_packed_row_keep_and_flip(
                        self.n_samples,
                        non_missing,
                        alt_sum,
                        het_count,
                        maf_threshold,
                        max_missing_rate,
                        apply_het_filter,
                        het_threshold,
                    );
                    if keep {
                        kept = kept.saturating_add(1);
                    }
                }
                kept
            };

            kept_total = kept_total.saturating_add(kept_blk);
            scanned += take;
            blocks += 1;
        }

        self.cursor = self.cursor.saturating_add(scanned);
        Ok((kept_total, scanned, blocks))
    }

    fn random_rows_qc_counts(
        &self,
        snp_indices: Vec<usize>,
        parallel: bool,
    ) -> Result<(u64, u64, u64, usize), String> {
        if snp_indices.is_empty() {
            return Ok((0u64, 0u64, 0u64, 0usize));
        }
        for &idx in snp_indices.iter() {
            if idx >= self.n_snps {
                return Err(format!(
                    "snp_index out of range: {idx} >= n_snps={}",
                    self.n_snps
                ));
            }
        }
        let rows_per_task = 512usize;
        let (m, h, ha) = if parallel && snp_indices.len() >= 256 {
            snp_indices
                .par_chunks(rows_per_task)
                .map(|idx_chunk| {
                    let mut m = 0u64;
                    let mut h = 0u64;
                    let mut ha = 0u64;
                    for &idx in idx_chunk.iter() {
                        let start = 3usize + idx * self.bytes_per_snp;
                        let end = start + self.bytes_per_snp;
                        let row = &self.mmap[start..end];
                        let (xm, xh, xha) = count_packed_row_counts(row, self.n_samples);
                        m = m.saturating_add(xm as u64);
                        h = h.saturating_add(xh as u64);
                        ha = ha.saturating_add(xha as u64);
                    }
                    (m, h, ha)
                })
                .reduce(
                    || (0u64, 0u64, 0u64),
                    |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
                )
        } else {
            let mut m = 0u64;
            let mut h = 0u64;
            let mut ha = 0u64;
            for idx in snp_indices.iter().copied() {
                let start = 3usize + idx * self.bytes_per_snp;
                let end = start + self.bytes_per_snp;
                let row = &self.mmap[start..end];
                let (xm, xh, xha) = count_packed_row_counts(row, self.n_samples);
                m = m.saturating_add(xm as u64);
                h = h.saturating_add(xh as u64);
                ha = ha.saturating_add(xha as u64);
            }
            (m, h, ha)
        };
        Ok((m, h, ha, snp_indices.len()))
    }
}

#[pyclass]
pub struct BedMmapReader {
    engine: BedMmapEngine,
}

#[pymethods]
impl BedMmapReader {
    #[new]
    fn new(prefix: String) -> PyResult<Self> {
        let engine = BedMmapEngine::open(prefix.as_str()).map_err(PyRuntimeError::new_err)?;
        Ok(Self { engine })
    }

    fn reset(&mut self) {
        self.engine.reset();
    }

    fn seek(&mut self, snp_index: usize) -> PyResult<()> {
        self.engine.seek(snp_index).map_err(PyIndexError::new_err)
    }

    fn next_rows_packed<'py>(
        &mut self,
        py: Python<'py>,
        max_rows: usize,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let bps = self.engine.bytes_per_snp;
        let (packed, rows) = self
            .engine
            .next_rows_slice(max_rows)
            .map_err(PyRuntimeError::new_err)?;
        let mat = Array2::from_shape_vec((rows, bps), packed.to_vec())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        #[allow(deprecated)]
        Ok(PyArray2::from_owned_array(py, mat).into_bound())
    }

    fn read_rows_packed<'py>(
        &self,
        py: Python<'py>,
        start_snp: usize,
        max_rows: usize,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let bps = self.engine.bytes_per_snp;
        let (packed, rows) = self
            .engine
            .get_rows_slice(start_snp, max_rows)
            .map_err(PyRuntimeError::new_err)?;
        let mat = Array2::from_shape_vec((rows, bps), packed.to_vec())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        #[allow(deprecated)]
        Ok(PyArray2::from_owned_array(py, mat).into_bound())
    }

    fn get_row_packed<'py>(
        &self,
        py: Python<'py>,
        snp_index: usize,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let row = self
            .engine
            .row_slice(snp_index)
            .map_err(PyIndexError::new_err)?
            .to_vec();
        #[allow(deprecated)]
        Ok(PyArray1::from_owned_array(py, Array1::from_vec(row)).into_bound())
    }

    #[pyo3(signature = (max_rows=0, block_rows=8192))]
    fn scan_rows_checksum(
        &mut self,
        max_rows: usize,
        block_rows: usize,
    ) -> PyResult<(u64, usize, usize)> {
        self.engine
            .scan_rows_checksum(max_rows, block_rows)
            .map_err(PyRuntimeError::new_err)
    }

    fn random_rows_checksum(&self, snp_indices: Vec<usize>) -> PyResult<u64> {
        self.engine
            .random_rows_checksum(snp_indices)
            .map_err(PyIndexError::new_err)
    }

    #[pyo3(signature = (max_rows=0, block_rows=8192, parallel=true))]
    fn scan_rows_qc_counts(
        &mut self,
        max_rows: usize,
        block_rows: usize,
        parallel: bool,
    ) -> PyResult<(u64, u64, u64, usize, usize)> {
        self.engine
            .scan_rows_qc_counts(max_rows, block_rows, parallel)
            .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (
        maf_threshold=0.0,
        max_missing_rate=1.0,
        het_threshold=0.0,
        max_rows=0,
        block_rows=8192,
        parallel=true,
    ))]
    fn scan_rows_filter_counts(
        &mut self,
        maf_threshold: f32,
        max_missing_rate: f32,
        het_threshold: f32,
        max_rows: usize,
        block_rows: usize,
        parallel: bool,
    ) -> PyResult<(u64, usize, usize)> {
        self.engine
            .scan_rows_filter_counts(
                maf_threshold,
                max_missing_rate,
                het_threshold,
                max_rows,
                block_rows,
                parallel,
            )
            .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (snp_indices, parallel=true))]
    fn random_rows_qc_counts(
        &self,
        snp_indices: Vec<usize>,
        parallel: bool,
    ) -> PyResult<(u64, u64, u64, usize)> {
        self.engine
            .random_rows_qc_counts(snp_indices, parallel)
            .map_err(PyIndexError::new_err)
    }

    #[getter]
    fn prefix(&self) -> String {
        self.engine.prefix.clone()
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.engine.n_samples
    }

    #[getter]
    fn n_snps(&self) -> usize {
        self.engine.n_snps
    }

    #[getter]
    fn bytes_per_snp(&self) -> usize {
        self.engine.bytes_per_snp
    }

    #[getter]
    fn cursor(&self) -> usize {
        self.engine.cursor
    }
}

#[pyclass]
pub struct NpyMmapReader {
    path: String,
    mmap: Mmap,
    n_rows: usize,
    n_cols: usize,
    data_offset: usize,
    row_bytes: usize,
    cursor: usize,
}

impl NpyMmapReader {
    fn copy_rows_f32(
        &self,
        start_row: usize,
        max_rows: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        if start_row > self.n_rows {
            return Err(format!(
                "start_row out of range: {start_row} > n_rows={}",
                self.n_rows
            ));
        }
        if start_row == self.n_rows || max_rows == 0 {
            return Ok((Vec::new(), 0));
        }

        let end_row = std::cmp::min(self.n_rows, start_row.saturating_add(max_rows));
        let rows = end_row.saturating_sub(start_row);
        let start_byte = self
            .data_offset
            .checked_add(
                start_row
                    .checked_mul(self.row_bytes)
                    .ok_or_else(|| "NPY byte offset overflow".to_string())?,
            )
            .ok_or_else(|| "NPY byte offset overflow".to_string())?;
        let span = rows
            .checked_mul(self.row_bytes)
            .ok_or_else(|| "NPY row span overflow".to_string())?;
        let end_byte = start_byte
            .checked_add(span)
            .ok_or_else(|| "NPY end byte overflow".to_string())?;
        if end_byte > self.mmap.len() {
            return Err(format!(
                "NPY slice exceeds mmap length: end_byte={}, mmap_len={}",
                end_byte,
                self.mmap.len()
            ));
        }

        let bytes = &self.mmap[start_byte..end_byte];
        let total = rows
            .checked_mul(self.n_cols)
            .ok_or_else(|| "NPY output size overflow".to_string())?;
        let mut out = Vec::<f32>::with_capacity(total);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        if out.len() != total {
            return Err(format!(
                "NPY decode size mismatch: got {}, expected {}",
                out.len(),
                total
            ));
        }
        Ok((out, rows))
    }

    fn row_bounds(&self, row_idx: usize) -> Result<(usize, usize), String> {
        if row_idx >= self.n_rows {
            return Err(format!(
                "row_index out of range: {row_idx} >= n_rows={}",
                self.n_rows
            ));
        }
        let start = self
            .data_offset
            .checked_add(
                row_idx
                    .checked_mul(self.row_bytes)
                    .ok_or_else(|| "NPY row offset overflow".to_string())?,
            )
            .ok_or_else(|| "NPY row offset overflow".to_string())?;
        let end = start
            .checked_add(self.row_bytes)
            .ok_or_else(|| "NPY row end overflow".to_string())?;
        if end > self.mmap.len() {
            return Err(format!(
                "NPY row exceeds mmap length: end={}, mmap_len={}",
                end,
                self.mmap.len()
            ));
        }
        Ok((start, end))
    }
}

#[pymethods]
impl NpyMmapReader {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let file = File::open(&path)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to open {path}: {e}")))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| PyRuntimeError::new_err(format!("failed to mmap {path}: {e}")))?;
        let (n_rows, n_cols, data_offset) =
            parse_npy_f32_header_local(&mmap[..]).map_err(PyRuntimeError::new_err)?;
        let row_bytes = n_cols
            .checked_mul(4)
            .ok_or_else(|| PyRuntimeError::new_err("NPY row byte size overflow"))?;

        Ok(Self {
            path,
            mmap,
            n_rows,
            n_cols,
            data_offset,
            row_bytes,
            cursor: 0,
        })
    }

    fn reset(&mut self) {
        self.cursor = 0;
    }

    fn seek(&mut self, row_index: usize) -> PyResult<()> {
        if row_index > self.n_rows {
            return Err(PyIndexError::new_err(format!(
                "row_index out of range: {row_index} > n_rows={}",
                self.n_rows
            )));
        }
        self.cursor = row_index;
        Ok(())
    }

    fn next_rows<'py>(
        &mut self,
        py: Python<'py>,
        max_rows: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let start = self.cursor;
        let (data, rows) = self
            .copy_rows_f32(start, max_rows)
            .map_err(PyRuntimeError::new_err)?;
        self.cursor = start.saturating_add(rows);
        let mat = Array2::from_shape_vec((rows, self.n_cols), data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        #[allow(deprecated)]
        Ok(PyArray2::from_owned_array(py, mat).into_bound())
    }

    fn read_rows<'py>(
        &self,
        py: Python<'py>,
        start_row: usize,
        max_rows: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (data, rows) = self
            .copy_rows_f32(start_row, max_rows)
            .map_err(PyRuntimeError::new_err)?;
        let mat = Array2::from_shape_vec((rows, self.n_cols), data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        #[allow(deprecated)]
        Ok(PyArray2::from_owned_array(py, mat).into_bound())
    }

    fn get_row<'py>(
        &self,
        py: Python<'py>,
        row_index: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let (start, end) = self.row_bounds(row_index).map_err(PyIndexError::new_err)?;
        let bytes = &self.mmap[start..end];
        let mut row = Vec::<f32>::with_capacity(self.n_cols);
        for chunk in bytes.chunks_exact(4) {
            row.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        if row.len() != self.n_cols {
            return Err(PyRuntimeError::new_err(format!(
                "NPY row decode size mismatch: got {}, expected {}",
                row.len(),
                self.n_cols
            )));
        }
        #[allow(deprecated)]
        Ok(PyArray1::from_owned_array(py, Array1::from_vec(row)).into_bound())
    }

    #[getter]
    fn path(&self) -> String {
        self.path.clone()
    }

    #[getter]
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    #[getter]
    fn n_cols(&self) -> usize {
        self.n_cols
    }

    #[getter]
    fn cursor(&self) -> usize {
        self.cursor
    }
}

#[pyfunction]
pub fn gfd_packbits_from_dosage_block<'py>(
    py: Python<'py>,
    block: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let x = block.as_array();
    if x.ndim() != 2 {
        return Err(PyValueError::new_err(
            "block must be 2D (n_sites, n_samples)",
        ));
    }
    let n_rows = x.shape()[0];
    let n_samples = x.shape()[1];
    if n_samples == 0 {
        let out = Array2::<u8>::zeros((0, 0));
        #[allow(deprecated)]
        return Ok(PyArray2::from_owned_array(py, out).into_bound());
    }

    let packed_cols = (n_samples + 7) / 8;
    let out_rows = n_rows.saturating_mul(2);
    let total = out_rows.saturating_mul(packed_cols);
    let mut out: Vec<u8> = vec![0u8; total];

    // Prefer zero-copy for contiguous blocks; otherwise make one contiguous copy.
    let input_owned;
    let input: &[f32] = if let Some(s) = x.as_slice_memory_order() {
        s
    } else {
        let (raw, offset) = x.to_owned().into_raw_vec_and_offset();
        let total = n_rows.saturating_mul(n_samples);
        let start = offset.unwrap_or(0);
        let end = start.saturating_add(total);
        if end > raw.len() {
            return Err(PyValueError::new_err(format!(
                "block copy layout error: start={}, total={}, raw_len={}",
                start,
                total,
                raw.len()
            )));
        }
        input_owned = raw[start..end].to_vec();
        &input_owned
    };

    #[inline]
    fn normalize_g(v: f32) -> Option<u8> {
        if !v.is_finite() || v < 0.0 {
            None
        } else {
            let r = v.round();
            if r <= 0.0 {
                Some(0)
            } else if r >= 2.0 {
                Some(2)
            } else {
                Some(1)
            }
        }
    }

    py.detach(|| {
        out.par_chunks_mut(2 * packed_cols)
            .enumerate()
            .for_each(|(r, out_rows2)| {
                let row = &input[r * n_samples..(r + 1) * n_samples];

                let mut c0: usize = 0;
                let mut c1: usize = 0;
                let mut c2: usize = 0;
                for &v in row.iter() {
                    if let Some(g) = normalize_g(v) {
                        match g {
                            0 => c0 += 1,
                            1 => c1 += 1,
                            _ => c2 += 1,
                        }
                    }
                }
                // Tie order follows numpy argmax([c0,c1,c2]) => 0 > 1 > 2 on ties.
                let mode: u8 = if c0 >= c1 && c0 >= c2 {
                    0
                } else if c1 >= c2 {
                    1
                } else {
                    2
                };

                let (left, right) = out_rows2.split_at_mut(packed_cols);
                for b in 0..packed_cols {
                    let base = b * 8;
                    let mut lb: u8 = 0;
                    let mut rb: u8 = 0;
                    for bit in 0..8 {
                        let i = base + bit;
                        if i >= n_samples {
                            break;
                        }
                        let g = normalize_g(row[i]).unwrap_or(mode);
                        let mask = 1u8 << (bit as u8);
                        if g != 0 {
                            lb |= mask;
                        }
                        if g != 2 {
                            rb |= mask;
                        }
                    }
                    left[b] = lb;
                    right[b] = rb;
                }
            });
    });

    let mat = Array2::from_shape_vec((out_rows, packed_cols), out)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    #[allow(deprecated)]
    Ok(PyArray2::from_owned_array(py, mat).into_bound())
}

#[pyfunction]
#[pyo3(signature = (path_or_prefix, delimiter=None))]
pub fn load_site_info(
    path_or_prefix: String,
    delimiter: Option<String>,
) -> PyResult<Vec<SiteInfo>> {
    let p = path_or_prefix.trim().to_string();
    if p.is_empty() {
        return Err(PyValueError::new_err("path_or_prefix must not be empty"));
    }

    let lower = p.to_ascii_lowercase();
    let is_plink_explicit =
        lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam");
    let is_plink_prefix = Path::new(&(p.clone() + ".bed")).exists()
        && Path::new(&(p.clone() + ".bim")).exists()
        && Path::new(&(p.clone() + ".fam")).exists();
    if is_plink_explicit || is_plink_prefix {
        let prefix = normalize_plink_prefix_local(&p);
        let sites = core::read_bim(&prefix).map_err(PyRuntimeError::new_err)?;
        return Ok(sites.into_iter().map(Into::into).collect());
    }

    let it = TxtSnpIter::new(&p, delimiter.as_deref()).map_err(PyRuntimeError::new_err)?;
    Ok(it.sites.into_iter().map(Into::into).collect())
}

#[pyfunction]
#[pyo3(signature = (path_or_prefix, row_indices=None))]
pub fn load_bim_columns<'py>(
    path_or_prefix: String,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<(Vec<String>, Vec<i64>, Vec<String>, Vec<String>, Vec<String>)> {
    let p = path_or_prefix.trim().to_string();
    if p.is_empty() {
        return Err(PyValueError::new_err("path_or_prefix must not be empty"));
    }

    let lower = p.to_ascii_lowercase();
    let is_plink_explicit =
        lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam");
    let is_plink_prefix = Path::new(&(p.clone() + ".bed")).exists()
        && Path::new(&(p.clone() + ".bim")).exists()
        && Path::new(&(p.clone() + ".fam")).exists();
    if !(is_plink_explicit || is_plink_prefix) {
        return Err(PyValueError::new_err(
            "load_bim_columns requires a PLINK BED/BIM/FAM prefix or explicit PLINK file path",
        ));
    }

    let prefix = normalize_plink_prefix_local(&p);
    let selected_rows: Option<Vec<usize>> = if let Some(row_indices) = row_indices {
        let slice = row_indices.as_slice()?;
        let mut out = Vec::with_capacity(slice.len());
        for &raw in slice {
            let idx = usize::try_from(raw).map_err(|_| {
                PyValueError::new_err(format!("row_indices must be non-negative, got {raw}"))
            })?;
            out.push(idx);
        }
        Some(out)
    } else {
        None
    };

    let (chrom, pos, snp, allele0, allele1) =
        core::read_bim_columns(&prefix, selected_rows.as_deref())
            .map_err(PyRuntimeError::new_err)?;
    Ok((
        chrom,
        pos.into_iter().map(|v| v as i64).collect(),
        snp,
        allele0,
        allele1,
    ))
}

// -------- count_vcf_snps (Py function) --------
#[pyfunction]
pub fn count_vcf_snps(path: String) -> PyResult<usize> {
    let p = Path::new(&path);
    let mut reader =
        core::open_text_maybe_gz(p).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let mut n: usize = 0;
    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        if bytes == 0 {
            break;
        }
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        n += 1;
    }
    Ok(n)
}

#[pyfunction]
pub fn count_hmp_snps(path: String) -> PyResult<usize> {
    let p = Path::new(&path);
    let mut reader =
        core::open_text_maybe_gz(p).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let mut n: usize = 0;
    let mut line = String::new();
    let mut header_seen = false;
    loop {
        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        if bytes == 0 {
            break;
        }
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        if !header_seen {
            header_seen = true;
            continue;
        }
        n += 1;
    }
    if !header_seen {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "No HapMap header found in file",
        ));
    }
    Ok(n)
}

// ============================================================
// Streaming PLINK writer: write .fam once, then stream .bim and .bed
// ============================================================

fn write_fam_simple(
    path: &Path,
    sample_ids: &[String],
    phenotype: Option<&[f64]>,
) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    for (i, sid) in sample_ids.iter().enumerate() {
        let ph = phenotype.map(|p| p[i]).unwrap_or(-9.0);
        // FID IID PID MID SEX PHENO
        writeln!(w, "{0}\t{0}\t0\t0\t1\t{1}", sid, ph).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[inline]
fn plink2bits_from_g_i8(g: i8) -> u8 {
    // PLINK .bed 2-bit encoding:
    // 00 = homozygous A1/A1
    // 10 = heterozygous
    // 11 = homozygous A2/A2
    // 01 = missing
    match g {
        0 => 0b00,
        1 => 0b10,
        2 => 0b11,
        _ => 0b01,
    }
}

#[pyclass]
pub struct PlinkStreamWriter {
    n_samples: usize,
    bed: BufWriter<File>,
    bim: BufWriter<File>,
    written_snps: usize,
}

#[pymethods]
impl PlinkStreamWriter {
    /// PlinkStreamWriter(prefix, sample_ids, phenotype=None)
    ///
    /// Creates the following files:
    ///   - {prefix}.fam  (written once)
    ///   - {prefix}.bed  (header written once, then streamed)
    ///   - {prefix}.bim  (streamed line-by-line)
    #[new]
    #[pyo3(signature = (prefix, sample_ids, phenotype))]
    fn new(prefix: String, sample_ids: Vec<String>, phenotype: Option<Vec<f64>>) -> PyResult<Self> {
        if sample_ids.is_empty() {
            return Err(PyValueError::new_err("sample_ids is empty"));
        }
        if let Some(ref p) = phenotype {
            if p.len() != sample_ids.len() {
                return Err(PyValueError::new_err(format!(
                    "phenotype length mismatch: phenotype={}, n_samples={}",
                    p.len(),
                    sample_ids.len()
                )));
            }
        }

        let bed_path = format!("{prefix}.bed");
        let bim_path = format!("{prefix}.bim");
        let fam_path = format!("{prefix}.fam");

        // 1) write .fam once
        let ph_ref = phenotype.as_ref().map(|v| v.as_slice());
        write_fam_simple(Path::new(&fam_path), &sample_ids, ph_ref)
            .map_err(|e| PyErr::new::<PyIOError, _>(e))?;

        // 2) open .bed and write header once (SNP-major)
        let mut bed = BufWriter::with_capacity(
            8 * 1024 * 1024,
            File::create(&bed_path).map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?,
        );
        bed.write_all(&[0x6C, 0x1B, 0x01])
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

        // 3) open .bim
        let bim = BufWriter::with_capacity(
            4 * 1024 * 1024,
            File::create(&bim_path).map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?,
        );

        Ok(Self {
            n_samples: sample_ids.len(),
            bed,
            bim,
            written_snps: 0,
        })
    }

    /// write_chunk(geno_chunk, sites)
    ///
    /// geno_chunk: ndarray[int8], shape (m_chunk, n_samples), SNP-major, -9 for missing
    /// sites: Vec<SiteInfo>, length m_chunk
    fn write_chunk(
        &mut self,
        geno_chunk: PyReadonlyArray2<i8>,
        sites: Vec<SiteInfo>,
    ) -> PyResult<()> {
        let arr = geno_chunk.as_array();
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err(
                "geno_chunk must be 2D (m_chunk, n_samples)",
            ));
        }

        let m_chunk = shape[0];
        let n_samples = shape[1];

        if n_samples != self.n_samples {
            return Err(PyValueError::new_err(format!(
                "n_samples mismatch: writer expects {}, got {}",
                self.n_samples, n_samples
            )));
        }
        if sites.len() != m_chunk {
            return Err(PyValueError::new_err(format!(
                "sites length mismatch: sites={}, m_chunk={}",
                sites.len(),
                m_chunk
            )));
        }

        // Write BIM lines for this chunk
        for s in sites.iter() {
            writeln!(
                self.bim,
                "{}\t{}_{}\t0\t{}\t{}\t{}",
                s.chrom, s.chrom, s.pos, s.pos, s.ref_allele, s.alt_allele
            )
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        }

        // Write BED bytes SNP-by-SNP (packed 2-bit)
        let bytes_per_snp = (self.n_samples + 3) / 4;

        // ndarray strides are in number of elements
        let strides = arr.strides();
        let s0 = strides[0] as isize; // row stride
        let s1 = strides[1] as isize; // col stride
        let base = arr.as_ptr(); // *const i8

        let mut row_buf = vec![0u8; bytes_per_snp];
        unsafe {
            for snp in 0..m_chunk {
                let snp_off = (snp as isize) * s0;
                let mut i = 0usize;
                for out_b in 0..bytes_per_snp {
                    let mut byte: u8 = 0;
                    for k in 0..4 {
                        let si = i + k;
                        let two = if si < self.n_samples {
                            let off = snp_off + (si as isize) * s1;
                            let g = *base.offset(off);
                            plink2bits_from_g_i8(g)
                        } else {
                            0b01 // padding missing
                        };
                        byte |= two << (k * 2);
                    }
                    row_buf[out_b] = byte;
                    i += 4;
                }
                self.bed
                    .write_all(&row_buf)
                    .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
                self.written_snps += 1;
            }
        }

        Ok(())
    }

    /// Flush buffered bytes to disk.
    fn flush(&mut self) -> PyResult<()> {
        self.bed
            .flush()
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        self.bim
            .flush()
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        Ok(())
    }

    /// Close writer (flush only; file handles will be dropped by Rust).
    fn close(&mut self) -> PyResult<()> {
        self.flush()?;
        Ok(())
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[getter]
    fn written_snps(&self) -> usize {
        self.written_snps
    }
}

#[cfg(test)]
mod tests {
    use super::{
        count_packed_row_counts, count_packed_row_counts_selected,
        count_packed_row_counts_selected_with_excluded, count_packed_row_pure_line_counts,
        count_packed_row_pure_line_counts_selected_with_excluded,
        evaluate_packed_row_keep_and_flip, precompute_excluded_sample_indices,
    };

    fn pack_plink_codes(codes: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; (codes.len() + 3) / 4];
        for (i, &code) in codes.iter().enumerate() {
            out[i >> 2] |= (code & 0b11) << ((i & 3) * 2);
        }
        out
    }

    #[test]
    fn selected_counts_follow_subset_indices() {
        let row = pack_plink_codes(&[0b00, 0b11, 0b10, 0b01, 0b11, 0b00]);
        let full = count_packed_row_counts(&row, 6);
        assert_eq!(full, (1, 1, 2));

        let selected = count_packed_row_counts_selected(&row, 6, &[1, 2, 4]);
        assert_eq!(selected, (0, 1, 2));

        let identity = count_packed_row_counts_selected(&row, 6, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(identity, full);
    }

    #[test]
    fn selected_counts_with_excluded_match_direct_subset_counts() {
        let row = pack_plink_codes(&[0b00, 0b11, 0b10, 0b01, 0b11, 0b00]);
        let sample_indices = [0usize, 1, 2, 4, 5];
        let excluded = precompute_excluded_sample_indices(6, &sample_indices)
            .expect("near-full subset should precompute excluded sample indices");
        let direct = count_packed_row_counts_selected(&row, 6, &sample_indices);
        let with_excluded = count_packed_row_counts_selected_with_excluded(
            &row,
            6,
            &sample_indices,
            Some(excluded.as_slice()),
        );
        assert_eq!(with_excluded, direct);
    }

    #[test]
    fn pure_line_selected_counts_with_excluded_match_direct_subset_counts() {
        let row = pack_plink_codes(&[0b00, 0b11, 0b10, 0b01, 0b11, 0b00]);
        let sample_indices = [0usize, 1, 2, 4, 5];
        let excluded = precompute_excluded_sample_indices(6, &sample_indices)
            .expect("near-full subset should precompute excluded sample indices");
        let direct = count_packed_row_pure_line_counts_selected_with_excluded(
            &row,
            6,
            &sample_indices,
            None,
        );
        let with_excluded = count_packed_row_pure_line_counts_selected_with_excluded(
            &row,
            6,
            &sample_indices,
            Some(excluded.as_slice()),
        );
        assert_eq!(with_excluded, direct);
        assert_eq!(count_packed_row_pure_line_counts(&row, 6), (2, 2));
    }

    #[test]
    fn subset_counts_can_change_flip_direction() {
        let row = pack_plink_codes(&[0b00, 0b00, 0b00, 0b11]);

        let (full_missing, full_het, full_hom_alt) = count_packed_row_counts(&row, 4);
        let full_non_missing = 4usize.saturating_sub(full_missing);
        let full_alt_sum = full_het.saturating_add(full_hom_alt.saturating_mul(2));
        let (full_keep, full_flip) = evaluate_packed_row_keep_and_flip(
            4,
            full_non_missing,
            full_alt_sum,
            full_het,
            0.0,
            1.0,
            false,
            0.0,
        );
        assert!(full_keep);
        assert!(!full_flip);

        let (subset_missing, subset_het, subset_hom_alt) =
            count_packed_row_counts_selected(&row, 4, &[3]);
        let subset_non_missing = 1usize.saturating_sub(subset_missing);
        let subset_alt_sum = subset_het.saturating_add(subset_hom_alt.saturating_mul(2));
        let (subset_keep, subset_flip) = evaluate_packed_row_keep_and_flip(
            1,
            subset_non_missing,
            subset_alt_sum,
            subset_het,
            0.0,
            1.0,
            false,
            0.0,
        );
        assert!(subset_keep);
        assert!(subset_flip);
    }
}

// ============================================================
// Streaming VCF writer: plain .vcf OR gzip .vcf.gz (auto)
// ============================================================

#[inline]
fn vcf_gt_from_g_i8(g: i8) -> &'static str {
    match g {
        0 => "0/0",
        1 => "0/1",
        2 => "1/1",
        _ => "./.",
    }
}

#[pyclass]
pub struct VcfStreamWriter {
    n_samples: usize,
    out: Option<VcfOut>, // Option allows take() on close
    written_snps: usize,
}

#[pymethods]
impl VcfStreamWriter {
    /// VcfStreamWriter(path, sample_ids)
    ///
    /// If `path` ends with ".gz", output is gzip-compressed (VCF.gz),
    /// otherwise it is a plain text VCF.
    ///
    /// This is "true streaming": variants are written chunk-by-chunk,
    /// without accumulating in memory.
    #[new]
    fn new(path: String, sample_ids: Vec<String>) -> PyResult<Self> {
        if sample_ids.is_empty() {
            return Err(PyValueError::new_err("sample_ids is empty"));
        }

        let mut out =
            VcfOut::from_path(&path).map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

        // Write VCF headers once
        out.write_all(b"##fileformat=VCFv4.2\n")
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        out.write_all(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

        // Write header row
        out.write_all(b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        for sid in sample_ids.iter() {
            out.write_all(b"\t")
                .and_then(|_| out.write_all(sid.as_bytes()))
                .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        }
        out.write_all(b"\n")
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

        Ok(Self {
            n_samples: sample_ids.len(),
            out: Some(out),
            written_snps: 0,
        })
    }

    /// write_chunk(geno_chunk, sites)
    ///
    /// geno_chunk: ndarray[int8] shape (m_chunk, n_samples), SNP-major, -9 missing
    /// sites: Vec[SiteInfo] length m_chunk
    fn write_chunk(
        &mut self,
        geno_chunk: PyReadonlyArray2<i8>,
        sites: Vec<SiteInfo>,
    ) -> PyResult<()> {
        let out = self
            .out
            .as_mut()
            .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("writer is closed"))?;

        let arr = geno_chunk.as_array();
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err(
                "geno_chunk must be 2D (m_chunk, n_samples)",
            ));
        }
        let m_chunk = shape[0];
        let n_samples = shape[1];

        if n_samples != self.n_samples {
            return Err(PyValueError::new_err(format!(
                "n_samples mismatch: writer expects {}, got {}",
                self.n_samples, n_samples
            )));
        }
        if sites.len() != m_chunk {
            return Err(PyValueError::new_err(format!(
                "sites length mismatch: sites={}, m_chunk={}",
                sites.len(),
                m_chunk
            )));
        }

        // ndarray strides in number of elements
        let strides = arr.strides();
        let s0 = strides[0] as isize; // row stride
        let s1 = strides[1] as isize; // col stride
        let base = arr.as_ptr();

        unsafe {
            for snp in 0..m_chunk {
                let s = &sites[snp];
                let snp_id = format!("{}_{}", s.chrom, s.pos);

                // Fixed 9 columns
                let prefix = format!(
                    "{}\t{}\t{}\t{}\t{}\t.\tPASS\t.\tGT",
                    s.chrom, s.pos, snp_id, s.ref_allele, s.alt_allele
                );
                out.write_all(prefix.as_bytes())
                    .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

                // Append sample GTs
                let snp_off = (snp as isize) * s0;
                for i in 0..self.n_samples {
                    let off = snp_off + (i as isize) * s1;
                    let g = *base.offset(off);
                    out.write_all(b"\t")
                        .and_then(|_| out.write_all(vcf_gt_from_g_i8(g).as_bytes()))
                        .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
                }

                out.write_all(b"\n")
                    .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

                self.written_snps += 1;
            }
        }

        Ok(())
    }

    /// Flush buffered bytes.
    /// For gzip output, this flushes encoder buffers but does not finalize gzip trailer.
    fn flush(&mut self) -> PyResult<()> {
        let out = self
            .out
            .as_mut()
            .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("writer is closed"))?;
        out.flush()
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        Ok(())
    }

    /// Close writer. For gzip output, this finalizes the gzip stream (writes trailer).
    fn close(&mut self) -> PyResult<()> {
        if let Some(out) = self.out.take() {
            out.finish()
                .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        }
        Ok(())
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[getter]
    fn written_snps(&self) -> usize {
        self.written_snps
    }
}

#[inline]
fn hmp_base_byte(s: &str) -> u8 {
    let c = s.chars().next().unwrap_or('N').to_ascii_uppercase();
    match c {
        'A' | 'C' | 'G' | 'T' => c as u8,
        _ => b'N',
    }
}

#[inline]
fn hmp_gt_bytes_from_g_i8(g: i8, ref_b: u8, alt_b: u8) -> [u8; 2] {
    match g {
        0 => [ref_b, ref_b],
        1 => [ref_b, alt_b],
        2 => [alt_b, alt_b],
        _ => [b'N', b'N'],
    }
}

// ============================================================
// Streaming HMP writer: plain .hmp OR gzip .hmp.gz (auto)
// ============================================================

#[pyclass]
pub struct HmpStreamWriter {
    n_samples: usize,
    out: Option<VcfOut>,
    written_snps: usize,
}

#[pymethods]
impl HmpStreamWriter {
    #[new]
    fn new(path: String, sample_ids: Vec<String>) -> PyResult<Self> {
        if sample_ids.is_empty() {
            return Err(PyValueError::new_err("sample_ids is empty"));
        }
        let mut out =
            VcfOut::from_path(&path).map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

        out.write_all(
            b"rs#\talleles\tchrom\tpos\tstrand\tassembly#\tcenter\tprotLSID\tassayLSID\tpanelLSID\tQCcode",
        )
        .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        for sid in sample_ids.iter() {
            out.write_all(b"\t")
                .and_then(|_| out.write_all(sid.as_bytes()))
                .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        }
        out.write_all(b"\n")
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

        Ok(Self {
            n_samples: sample_ids.len(),
            out: Some(out),
            written_snps: 0,
        })
    }

    fn write_chunk(
        &mut self,
        geno_chunk: PyReadonlyArray2<i8>,
        sites: Vec<SiteInfo>,
    ) -> PyResult<()> {
        let out = self
            .out
            .as_mut()
            .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("writer is closed"))?;

        let arr = geno_chunk.as_array();
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err(
                "geno_chunk must be 2D (m_chunk, n_samples)",
            ));
        }
        let m_chunk = shape[0];
        let n_samples = shape[1];
        if n_samples != self.n_samples {
            return Err(PyValueError::new_err(format!(
                "n_samples mismatch: writer expects {}, got {}",
                self.n_samples, n_samples
            )));
        }
        if sites.len() != m_chunk {
            return Err(PyValueError::new_err(format!(
                "sites length mismatch: sites={}, m_chunk={}",
                sites.len(),
                m_chunk
            )));
        }

        let strides = arr.strides();
        let s0 = strides[0] as isize;
        let s1 = strides[1] as isize;
        let base = arr.as_ptr();

        unsafe {
            for snp in 0..m_chunk {
                let s = &sites[snp];
                let snp_id = format!("{}_{}", s.chrom, s.pos);
                let ref_b = hmp_base_byte(&s.ref_allele);
                let mut alt_b = hmp_base_byte(&s.alt_allele);
                if alt_b == ref_b {
                    alt_b = if ref_b == b'A' { b'C' } else { b'A' };
                }
                let alleles = format!("{}/{}", ref_b as char, alt_b as char);
                let prefix = format!(
                    "{}\t{}\t{}\t{}\t+\t.\t.\t.\t.\t.\t.",
                    snp_id, alleles, s.chrom, s.pos
                );
                out.write_all(prefix.as_bytes())
                    .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

                let snp_off = (snp as isize) * s0;
                for i in 0..self.n_samples {
                    let off = snp_off + (i as isize) * s1;
                    let g = *base.offset(off);
                    let gt = hmp_gt_bytes_from_g_i8(g, ref_b, alt_b);
                    out.write_all(b"\t")
                        .and_then(|_| out.write_all(&gt))
                        .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
                }
                out.write_all(b"\n")
                    .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
                self.written_snps += 1;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> PyResult<()> {
        let out = self
            .out
            .as_mut()
            .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("writer is closed"))?;
        out.flush()
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        Ok(())
    }

    fn close(&mut self) -> PyResult<()> {
        if let Some(out) = self.out.take() {
            out.finish()
                .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        }
        Ok(())
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[getter]
    fn written_snps(&self) -> usize {
        self.written_snps
    }
}
