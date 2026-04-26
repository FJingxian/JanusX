use pyo3::exceptions::*;
use pyo3::prelude::*;
use pyo3::BoundObject;

use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

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

            if let (Some(lo), Some(hi)) = (bp_min, bp_max) {
                if lo > hi {
                    return Err("bp_min cannot be greater than bp_max".to_string());
                }
            }

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

            let bim_range_norm: Option<(String, i32, i32)> = if let Some((c, s, e)) = bim_range {
                if s > e {
                    return Err("bim_range start cannot be greater than end".to_string());
                }
                Some((normalize_chr_key_local(&c), s, e))
            } else {
                None
            };

            let ranges_norm: Option<Vec<(String, i32, i32)>> = if let Some(v) = ranges {
                if v.is_empty() {
                    None
                } else {
                    let mut out_r: Vec<(String, i32, i32)> = Vec::with_capacity(v.len());
                    for (c, s, e) in v.into_iter() {
                        if s > e {
                            return Err("One range has start > end".to_string());
                        }
                        out_r.push((normalize_chr_key_local(&c), s, e));
                    }
                    Some(out_r)
                }
            } else {
                None
            };

            let bed_path = format!("{out}.bed");
            let bim_path = format!("{out}.bim");
            let fam_path = format!("{out}.fam");

            write_fam_simple(Path::new(&fam_path), &out_sample_ids, None)?;
            let mut bed = BufWriter::new(File::create(&bed_path).map_err(|e| e.to_string())?);
            bed.write_all(&[0x6C, 0x1B, 0x01])
                .map_err(|e| e.to_string())?;
            let mut bim = BufWriter::new(File::create(&bim_path).map_err(|e| e.to_string())?);

            let bytes_per_snp = (n_out_samples + 3) / 4;
            let mut n_scanned: usize = 0;
            let mut n_kept: usize = 0;

            for snp_idx in 0..it.sites.len() {
                let site0 = &it.sites[snp_idx];
                let c = normalize_chr_key_local(&site0.chrom);
                let p = site0.pos;

                if let Some(ref st) = site_set {
                    if !st.contains(&(c.clone(), p)) {
                        continue;
                    }
                }
                if let Some((ref bc, bs, be)) = bim_range_norm {
                    if !(c == *bc && p >= bs && p <= be) {
                        continue;
                    }
                }
                if let Some(ref cs) = chr_set {
                    if !cs.contains(&c) {
                        continue;
                    }
                }
                if let Some(lo) = bp_min {
                    if p < lo {
                        continue;
                    }
                }
                if let Some(hi) = bp_max {
                    if p > hi {
                        continue;
                    }
                }
                if let Some(ref rr) = ranges_norm {
                    let mut hit = false;
                    for (rc, rs, re) in rr.iter() {
                        if c == *rc && p >= *rs && p <= *re {
                            hit = true;
                            break;
                        }
                    }
                    if !hit {
                        continue;
                    }
                }

                n_scanned = n_scanned.saturating_add(1);
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
        let snp_indices =
            build_snp_indices(&it.sites, snp_range, snp_indices, bim_range, snp_sites)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

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
            let n_samples_f = n_samples as f32;
            let full_bytes = n_samples / 4;
            let rem = n_samples % 4;
            missing_rate
                .par_iter_mut()
                .zip(maf.par_iter_mut())
                .zip(std_denom.par_iter_mut())
                .enumerate()
                .for_each(|(snp_idx, ((miss_dst, maf_dst), std_dst))| {
                    let row = &packed[snp_idx * bytes_per_snp..(snp_idx + 1) * bytes_per_snp];
                    let mut non_missing: usize = 0;
                    let mut alt_sum: usize = 0;

                    for &byte in row.iter().take(full_bytes) {
                        for within in 0..4 {
                            let code = (byte >> (within * 2)) & 0b11;
                            match code {
                                0b00 => {
                                    non_missing += 1;
                                }
                                0b10 => {
                                    non_missing += 1;
                                    alt_sum += 1;
                                }
                                0b11 => {
                                    non_missing += 1;
                                    alt_sum += 2;
                                }
                                _ => {}
                            }
                        }
                    }
                    if rem > 0 {
                        let byte = row[full_bytes];
                        for within in 0..rem {
                            let code = (byte >> (within * 2)) & 0b11;
                            match code {
                                0b00 => {
                                    non_missing += 1;
                                }
                                0b10 => {
                                    non_missing += 1;
                                    alt_sum += 1;
                                }
                                0b11 => {
                                    non_missing += 1;
                                    alt_sum += 2;
                                }
                                _ => {}
                            }
                        }
                    }

                    *miss_dst = 1.0_f32 - (non_missing as f32 / n_samples_f);
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
        input_owned = x.to_owned().into_raw_vec();
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

    py.allow_threads(|| {
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
