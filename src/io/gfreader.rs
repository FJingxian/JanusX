use pyo3::exceptions::*;
use pyo3::prelude::*;
use pyo3::BoundObject;

use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::bitwise::and_popcount;
use crate::gfcore as core;
use crate::gfcore::{BedSnpIter, HmpSnpIter, TxtSnpIter, VcfSnpIter};
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
    pub ref_allele: String,
    #[pyo3(get)]
    pub alt_allele: String,
}

impl From<core::SiteInfo> for SiteInfo {
    fn from(s: core::SiteInfo) -> Self {
        Self {
            chrom: s.chrom,
            pos: s.pos,
            ref_allele: s.ref_allele,
            alt_allele: s.alt_allele,
        }
    }
}

#[pymethods]
impl SiteInfo {
    #[new]
    fn new(chrom: String, pos: i32, ref_allele: String, alt_allele: String) -> Self {
        SiteInfo {
            chrom,
            pos,
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
fn sample_indices_are_identity(indices: &[usize]) -> bool {
    indices.iter().enumerate().all(|(i, &idx)| idx == i)
}

#[inline]
fn write_bim_site_line(w: &mut BufWriter<File>, site: &core::SiteInfo) -> Result<(), String> {
    let snp_id = format!("{}_{}", site.chrom, site.pos);
    writeln!(
        w,
        "{}\t{}\t0\t{}\t{}\t{}",
        site.chrom, snp_id, site.pos, site.ref_allele, site.alt_allele
    )
    .map_err(|e| e.to_string())
}

#[inline]
fn flip_bed_code(code: u8) -> u8 {
    match code & 0b11 {
        0b00 => 0b11,
        0b11 => 0b00,
        v => v,
    }
}

fn build_bed_flip_byte_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    for b in 0u16..=255u16 {
        let byte = b as u8;
        let mut out = 0u8;
        for k in 0..4usize {
            let code = (byte >> (k * 2)) & 0b11;
            let flip = flip_bed_code(code);
            out |= flip << (k * 2);
        }
        table[b as usize] = out;
    }
    table
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

fn build_bed_count_byte_table() -> [(u8, u8, u8); 256] {
    // For each BED byte (4 genotypes), return:
    //   (missing_count, het_count, hom_alt_count)
    let mut table = [(0u8, 0u8, 0u8); 256];
    for b in 0u16..=255u16 {
        let byte = b as u8;
        let mut missing = 0u8;
        let mut het = 0u8;
        let mut hom_alt = 0u8;
        for k in 0..4usize {
            let code = (byte >> (k * 2)) & 0b11;
            match code {
                0b01 => missing = missing.saturating_add(1),
                0b10 => het = het.saturating_add(1),
                0b11 => hom_alt = hom_alt.saturating_add(1),
                _ => {}
            }
        }
        table[b as usize] = (missing, het, hom_alt);
    }
    table
}

#[inline]
fn count_packed_row_counts(
    row: &[u8],
    n_samples: usize,
    byte_counts: &[(u8, u8, u8); 256],
) -> (usize, usize, usize) {
    let full_bytes = n_samples / 4;
    let rem_pairs = n_samples & 3;
    let mut missing = 0usize;
    let mut het = 0usize;
    let mut hom_alt = 0usize;

    for &b in row.iter().take(full_bytes) {
        let (m, h, ha) = byte_counts[b as usize];
        missing = missing.saturating_add(m as usize);
        het = het.saturating_add(h as usize);
        hom_alt = hom_alt.saturating_add(ha as usize);
    }
    if rem_pairs > 0 {
        let b = row[full_bytes];
        for k in 0..rem_pairs {
            let code = (b >> (k * 2)) & 0b11;
            match code {
                0b01 => missing = missing.saturating_add(1),
                0b10 => het = het.saturating_add(1),
                0b11 => hom_alt = hom_alt.saturating_add(1),
                _ => {}
            }
        }
    }
    (missing, het, hom_alt)
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

fn collapse_bed_row_sorted_subset(
    src_row: &[u8],
    include_words: &[u64],
    out_row: &mut [u8],
) -> Result<(), String> {
    let mut cur_output_word = 0u64;
    let mut word_write_halfshift: u32 = 0;
    let mut out_word_idx = 0usize;
    let mut raw_word_idx = 0usize;

    for &include_word in include_words.iter() {
        let mut include_half = include_word as u32;
        for half_idx in 0..2usize {
            if include_half != 0 {
                let raw_word = load_u64_le_partial(src_row, raw_word_idx.saturating_mul(8));
                let mut run_mask = include_half as u64;
                while run_mask != 0 {
                    let start = run_mask.trailing_zeros();
                    let inv_shifted = (!run_mask) >> start;
                    let mut run_len = inv_shifted.trailing_zeros();
                    let max_len = 32u32.saturating_sub(start);
                    if run_len > max_len {
                        run_len = max_len;
                    }

                    let raw_shifted = raw_word >> (start * 2);
                    let block_limit = 32u32.saturating_sub(word_write_halfshift);
                    cur_output_word |= raw_shifted << (word_write_halfshift * 2);

                    if run_len < block_limit {
                        word_write_halfshift += run_len;
                        if word_write_halfshift < 32 {
                            cur_output_word &= (1u64 << (word_write_halfshift * 2)) - 1u64;
                        }
                    } else {
                        store_u64_le_partial(
                            out_row,
                            out_word_idx.saturating_mul(8),
                            cur_output_word,
                        );
                        out_word_idx = out_word_idx.saturating_add(1);
                        word_write_halfshift = run_len - block_limit;
                        if word_write_halfshift > 0 {
                            cur_output_word = (raw_shifted >> (block_limit * 2))
                                & ((1u64 << (word_write_halfshift * 2)) - 1u64);
                        } else {
                            cur_output_word = 0u64;
                        }
                    }

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
fn evaluate_packed_row_keep_and_flip(
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
        let low = het_threshold as f64;
        let high = 1.0 - low;
        if het_rate < low || het_rate > high {
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

    let mut wbed = BufWriter::new(File::create(&out_bed).map_err(|e| format!("{out_bed}: {e}"))?);
    wbed.write_all(&header)
        .map_err(|e| format!("{out_bed}: {e}"))?;
    let mut wbim = BufWriter::new(File::create(&out_bim).map_err(|e| format!("{out_bim}: {e}"))?);

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

    let flip_table = build_bed_flip_byte_table();
    let byte_counts = build_bed_count_byte_table();

    let mut kept = 0usize;
    let mut out_row = vec![0u8; out_bytes_per_snp];

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
                    let (missing, het, hom_alt) =
                        count_packed_row_counts(row, n_out_samples, &byte_counts);
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
                let mut site = sites[snp_idx].clone();
                if flip {
                    std::mem::swap(&mut site.ref_allele, &mut site.alt_allele);
                }
                write_bim_site_line(&mut wbim, &site)?;

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
                        for i in 0..full_bytes_out {
                            out_row[i] = flip_table[row[i] as usize];
                        }
                        wbed.write_all(&out_row[..full_bytes_out])
                            .map_err(|e| format!("{out_bed}: {e}"))?;
                    }
                    if tail_pairs > 0 {
                        let last = flip_table[row[full_bytes_out] as usize] & tail_mask;
                        wbed.write_all(&[last])
                            .map_err(|e| format!("{out_bed}: {e}"))?;
                    }
                }
                kept = kept.saturating_add(1);
            }
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

        let mut pack_plan: Vec<SubsetPackPlanByte> = Vec::new();
        if !sample_indices_strictly_increasing {
            pack_plan = vec![
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
        }

        let mut src_row = vec![0u8; src_bytes_per_snp];
        let mut low_words = if sample_indices_strictly_increasing {
            Vec::new()
        } else {
            vec![0u64; n_words_src]
        };
        let mut high_words = if sample_indices_strictly_increasing {
            Vec::new()
        } else {
            vec![0u64; n_words_src]
        };

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

            let (keep, flip) = if sample_indices_strictly_increasing {
                collapse_bed_row_sorted_subset(
                    src_row.as_slice(),
                    selected_mask_words.as_slice(),
                    out_row.as_mut_slice(),
                )?;
                let (missing, het, hom_alt) =
                    count_packed_row_counts(out_row.as_slice(), n_out_samples, &byte_counts);
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
            } else {
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
            };
            if !keep {
                continue;
            }

            let mut site = sites[snp_idx].clone();
            if flip {
                std::mem::swap(&mut site.ref_allele, &mut site.alt_allele);
            }
            write_bim_site_line(&mut wbim, &site)?;

            if sample_indices_strictly_increasing {
                if flip {
                    for b in out_row.iter_mut() {
                        *b = flip_table[*b as usize];
                    }
                }
            } else {
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
                        packed = flip_table[packed as usize];
                    }
                    out_row[out_b] = packed;
                }
            }
            if tail_pairs > 0 {
                out_row[full_bytes_out] &= tail_mask;
            }
            wbed.write_all(&out_row)
                .map_err(|e| format!("{out_bed}: {e}"))?;
            kept = kept.saturating_add(1);
        }
    }

    wbed.flush().map_err(|e| format!("{out_bed}: {e}"))?;
    wbim.flush().map_err(|e| format!("{out_bim}: {e}"))?;
    Ok(kept)
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
    if !(0.0..=0.5).contains(&het) {
        return Err(PyValueError::new_err(
            "het_threshold must be within [0, 0.5]",
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
            let full_samples = sample_indices.len() == it.n_samples();
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

            if !fill_missing {
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
                )?;
                return Ok((n_kept, n_scanned, n_out_samples));
            }

            let bed_path = format!("{out}.bed");
            let bim_path = format!("{out}.bim");
            let fam_path = format!("{out}.fam");

            write_fam_simple(Path::new(&fam_path), &out_sample_ids, None)?;
            let mut bed = BufWriter::new(File::create(&bed_path).map_err(|e| e.to_string())?);
            bed.write_all(&[0x6C, 0x1B, 0x01])
                .map_err(|e| e.to_string())?;
            let mut bim = BufWriter::new(File::create(&bim_path).map_err(|e| e.to_string())?);

            let bytes_per_snp = (n_out_samples + 3) / 4;
            let mut n_kept: usize = 0;

            for &snp_idx in selected_snp_indices.iter() {
                let maybe = if full_samples {
                    it.get_snp_row_raw(snp_idx)
                } else {
                    it.get_snp_row_selected_raw(snp_idx, &sample_indices)
                };
                let (mut row, mut site) = match maybe {
                    Some(v) => v,
                    None => continue,
                };

                let keep = core::process_snp_row(
                    &mut row,
                    &mut site.ref_allele,
                    &mut site.alt_allele,
                    maf_threshold,
                    max_missing_rate,
                    fill_missing,
                    apply_het_filter,
                    het,
                );
                if !keep {
                    continue;
                }

                let snp_id = format!("{}_{}", site.chrom, site.pos);
                writeln!(
                    bim,
                    "{}\t{}\t0\t{}\t{}\t{}",
                    site.chrom, snp_id, site.pos, site.ref_allele, site.alt_allele
                )
                .map_err(|e| e.to_string())?;

                let mut si = 0usize;
                for _ in 0..bytes_per_snp {
                    let mut byte: u8 = 0;
                    for k in 0..4usize {
                        let bits = if si < row.len() {
                            let g = row[si];
                            let gi: i8 = if !g.is_finite() || g < 0.0 {
                                -9_i8
                            } else {
                                g.round().clamp(0.0, 2.0) as i8
                            };
                            plink2bits_from_g_i8(gi)
                        } else {
                            0b00
                        };
                        byte |= bits << (k * 2);
                        si += 1;
                    }
                    bed.write_all(&[byte]).map_err(|e| e.to_string())?;
                }

                n_kept = n_kept.saturating_add(1);
            }

            bed.flush().map_err(|e| e.to_string())?;
            bim.flush().map_err(|e| e.to_string())?;
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
        if !(0.0..=0.5).contains(&het) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "het_threshold must be within [0, 0.5]",
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
        let can_parallel = !self.it.is_windowed();
        let mut m: usize = 0;

        if let Some(ref snp_indices) = self.snp_indices {
            while m < chunk_size && self.snp_pos < snp_indices.len() {
                let need = chunk_size - m;
                let end = (self.snp_pos + need).min(snp_indices.len());
                let batch = &snp_indices[self.snp_pos..end];
                self.snp_pos = end;

                if can_parallel {
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
            if can_parallel {
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
        if !(0.0..=0.5).contains(&het) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "het_threshold must be within [0, 0.5]",
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
        if !(0.0..=0.5).contains(&het) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "het_threshold must be within [0, 0.5]",
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
        if !(0.0..=0.5).contains(&het) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "het_threshold must be within [0, 0.5]",
            ));
        }
        let apply_het_filter = model_key != "add";

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

            let bed_path = format!("{bed_prefix}.bed");
            let mut bed_file = File::open(&bed_path)
                .map_err(|e| format!("failed to open {bed_path}: {e}"))?;
            let bed_len = bed_file
                .metadata()
                .map_err(|e| format!("failed to stat {bed_path}: {e}"))?
                .len() as usize;
            if bed_len < 3 {
                return Err("BED too small".to_string());
            }

            let mut header = [0u8; 3];
            bed_file
                .read_exact(&mut header)
                .map_err(|e| format!("failed to read BED header: {e}"))?;
            if header[0] != 0x6C || header[1] != 0x1B || header[2] != 0x01 {
                return Err("Only SNP-major BED supported".to_string());
            }

            let bytes_per_snp = (n_samples + 3) / 4;
            let data_len = bed_len - 3;
            if bytes_per_snp == 0 || data_len % bytes_per_snp != 0 {
                return Err(format!(
                    "invalid BED payload length: data_len={data_len}, bytes_per_snp={bytes_per_snp}"
                ));
            }
            let n_snps = data_len / bytes_per_snp;

            let mut packed: Vec<u8> = vec![0u8; data_len];
            bed_file
                .read_exact(&mut packed)
                .map_err(|e| format!("failed to read BED payload: {e}"))?;

            let mut missing_rate: Vec<f32> = vec![0.0; n_snps];
            let mut maf: Vec<f32> = vec![0.0; n_snps];
            let mut std_denom: Vec<f32> = vec![0.0; n_snps];
            let byte_counts = build_bed_count_byte_table();
            missing_rate
                .par_iter_mut()
                .zip(maf.par_iter_mut())
                .zip(std_denom.par_iter_mut())
                .enumerate()
                .for_each(|(snp_idx, ((miss_dst, maf_dst), std_dst))| {
                    let row = &packed[snp_idx * bytes_per_snp..(snp_idx + 1) * bytes_per_snp];
                    let (missing, het, hom_alt) =
                        count_packed_row_counts(row, n_samples, &byte_counts);
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

#[inline]
fn _is_simple_snp_allele(a: &str) -> bool {
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
    if !(0.0..=0.5).contains(&het_threshold) {
        return Err(PyValueError::new_err(
            "het_threshold must be within [0, 0.5]",
        ));
    }

    let mut bed_prefix = prefix;
    let lower = bed_prefix.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        bed_prefix.truncate(bed_prefix.len() - 4);
    }

    let (packed_keep, miss_keep, maf_keep, std_keep, row_flip_keep, site_keep, n_samples, n_snps, bytes_per_snp) = py
        .detach(move || -> Result<
            (
                Vec<u8>,
                Vec<f32>,
                Vec<f32>,
                Vec<f32>,
                Vec<bool>,
                Vec<bool>,
                usize,
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
            let sites = core::read_bim(&bed_prefix).map_err(|e| e.to_string())?;
            let n_snps = sites.len();
            if n_snps == 0 {
                return Err("no SNP sites found in PLINK BIM input".to_string());
            }

            let bed_path = format!("{bed_prefix}.bed");
            let mut bed_file = File::open(&bed_path)
                .map_err(|e| format!("failed to open {bed_path}: {e}"))?;
            let bed_len = bed_file
                .metadata()
                .map_err(|e| format!("failed to stat {bed_path}: {e}"))?
                .len() as usize;
            if bed_len < 3 {
                return Err("BED too small".to_string());
            }

            let mut header = [0u8; 3];
            bed_file
                .read_exact(&mut header)
                .map_err(|e| format!("failed to read BED header: {e}"))?;
            if header[0] != 0x6C || header[1] != 0x1B || header[2] != 0x01 {
                return Err("Only SNP-major BED supported".to_string());
            }

            let bytes_per_snp = (n_samples + 3) / 4;
            let data_len = bed_len - 3;
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

            let mut packed_full: Vec<u8> = vec![0u8; data_len];
            bed_file
                .read_exact(&mut packed_full)
                .map_err(|e| format!("failed to read BED payload: {e}"))?;

            let byte_counts = build_bed_count_byte_table();
            let row_stats: Vec<(f32, f32, f32, f32, bool)> = packed_full
                .par_chunks(bytes_per_snp)
                .map(|row| {
                    let (missing, het, hom_alt) =
                        count_packed_row_counts(row, n_samples, &byte_counts);
                    let non_missing = n_samples.saturating_sub(missing);
                    let alt_sum = het.saturating_add(hom_alt.saturating_mul(2));
                    let miss = (missing as f32) / (n_samples as f32);
                    if non_missing > 0 {
                        let p = alt_sum as f64 / (2.0_f64 * non_missing as f64);
                        let flip = p > 0.5_f64;
                        let maf_v = p.min(1.0_f64 - p) as f32;
                        let het_rate = het as f32 / non_missing as f32;
                        let d = (2.0_f64 * p * (1.0_f64 - p)).sqrt() as f32;
                        let std = if d.is_finite() { d } else { 0.0_f32 };
                        (miss, maf_v, std, het_rate, flip)
                    } else {
                        (miss, 0.0_f32, 0.0_f32, 0.0_f32, false)
                    }
                })
                .collect();
            if row_stats.len() != n_snps {
                return Err(format!(
                    "internal error: stats rows {} != n_snps {n_snps}",
                    row_stats.len()
                ));
            }

            let mut missing_rate: Vec<f32> = vec![0.0; n_snps];
            let mut maf: Vec<f32> = vec![0.0; n_snps];
            let mut std_denom: Vec<f32> = vec![0.0; n_snps];
            let mut het_rate: Vec<f32> = vec![0.0; n_snps];
            let mut row_flip: Vec<bool> = vec![false; n_snps];
            for (i, (miss, maf_v, std, het, flip)) in row_stats.iter().copied().enumerate() {
                missing_rate[i] = miss;
                maf[i] = maf_v;
                std_denom[i] = std;
                het_rate[i] = het;
                row_flip[i] = flip;
            }

            let mut keep: Vec<bool> = vec![true; n_snps];
            for i in 0..n_snps {
                let maf_i = maf[i];
                let miss_i = missing_rate[i];
                let pass_num = (maf_i >= maf_threshold)
                    && (maf_i <= (1.0_f32 - maf_threshold))
                    && (miss_i <= max_missing_rate)
                    && (if het_threshold > 0.0_f32 {
                        let h = het_rate[i];
                        h >= het_threshold && h <= (1.0_f32 - het_threshold)
                    } else {
                        true
                    });
                let pass_snp = if snps_only {
                    _is_simple_snp_allele(&sites[i].ref_allele)
                        && _is_simple_snp_allele(&sites[i].alt_allele)
                } else {
                    true
                };
                keep[i] = pass_num && pass_snp;
            }

            let kept_n = keep.iter().filter(|&&x| x).count();
            if kept_n == 0 {
                return Err(
                    "No SNPs left after packed BED filtering. Please relax thresholds."
                        .to_string(),
                );
            }

            let (packed_keep, miss_keep, maf_keep, std_keep, row_flip_keep) = if kept_n == n_snps {
                (packed_full, missing_rate, maf, std_denom, row_flip)
            } else {
                let mut packed_keep = vec![0u8; kept_n * bytes_per_snp];
                let mut miss_keep = Vec::<f32>::with_capacity(kept_n);
                let mut maf_keep = Vec::<f32>::with_capacity(kept_n);
                let mut std_keep = Vec::<f32>::with_capacity(kept_n);
                let mut row_flip_keep = Vec::<bool>::with_capacity(kept_n);
                let mut dst = 0usize;
                for i in 0..n_snps {
                    if !keep[i] {
                        continue;
                    }
                    let src_off = i * bytes_per_snp;
                    let dst_off = dst * bytes_per_snp;
                    packed_keep[dst_off..dst_off + bytes_per_snp]
                        .copy_from_slice(&packed_full[src_off..src_off + bytes_per_snp]);
                    miss_keep.push(missing_rate[i]);
                    maf_keep.push(maf[i]);
                    std_keep.push(std_denom[i]);
                    row_flip_keep.push(row_flip[i]);
                    dst = dst.saturating_add(1);
                }
                debug_assert_eq!(dst, kept_n);
                (packed_keep, miss_keep, maf_keep, std_keep, row_flip_keep)
            };

            Ok((
                packed_keep,
                miss_keep,
                maf_keep,
                std_keep,
                row_flip_keep,
                keep,
                n_samples,
                n_snps,
                bytes_per_snp,
            ))
        })
        .map_err(PyRuntimeError::new_err)?;

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
        let mut bed = BufWriter::new(
            File::create(&bed_path).map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?,
        );
        bed.write_all(&[0x6C, 0x1B, 0x01])
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

        // 3) open .bim
        let bim = BufWriter::new(
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
            let snp_id = format!("{}_{}", s.chrom, s.pos);
            writeln!(
                self.bim,
                "{}\t{}\t0\t{}\t{}\t{}",
                s.chrom, snp_id, s.pos, s.ref_allele, s.alt_allele
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

        unsafe {
            for snp in 0..m_chunk {
                let snp_off = (snp as isize) * s0;

                let mut i = 0usize;
                for _ in 0..bytes_per_snp {
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
                    self.bed
                        .write_all(&[byte])
                        .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
                    i += 4;
                }

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
