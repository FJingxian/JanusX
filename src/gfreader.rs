use pyo3::BoundObject;
use pyo3::prelude::*;
use pyo3::exceptions::*;

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use flate2::write::GzEncoder;
use flate2::Compression;

use crate::gfcore::{BedSnpIter, VcfSnpIter};
use crate::gfcore as core;

// -------- Py-exposed SiteInfo (wrapper) --------
#[pyclass]
#[derive(Clone)]
pub struct SiteInfo {
    #[pyo3(get)] pub chrom: String,
    #[pyo3(get)] pub pos: i32,
    #[pyo3(get)] pub ref_allele: String,
    #[pyo3(get)] pub alt_allele: String,
}

impl From<core::SiteInfo> for SiteInfo {
    fn from(s: core::SiteInfo) -> Self {
        Self { chrom: s.chrom, pos: s.pos, ref_allele: s.ref_allele, alt_allele: s.alt_allele }
    }
}

#[pymethods]
impl SiteInfo {
    #[new]
    fn new(chrom: String, pos: i32, ref_allele: String, alt_allele: String) -> Self {
        SiteInfo { chrom, pos, ref_allele, alt_allele }
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
) -> Result<Option<Vec<usize>>, String> {
    let mut count = 0;
    if snp_range.is_some() { count += 1; }
    if snp_indices.is_some() { count += 1; }
    if bim_range.is_some() { count += 1; }
    if count > 1 {
        return Err("Provide only one of snp_range, snp_indices, or bim_range".into());
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

    Ok(None)
}

// -------- BedChunkReader --------
#[pyclass]
pub struct BedChunkReader {
    it: BedSnpIter,
    snp_indices: Option<Vec<usize>>,
    snp_pos: usize,
    sample_indices: Vec<usize>,
    sample_ids: Vec<String>,
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
        sample_ids=None,
        sample_indices=None,
        mmap_window_mb=None,
    ))]
    fn new(
        prefix: String,
        maf_threshold: Option<f32>,
        max_missing_rate: Option<f32>,
        fill_missing: Option<bool>,
        snp_range: Option<(usize, usize)>,
        snp_indices: Option<Vec<usize>>,
        bim_range: Option<(String, i32, i32)>,
        sample_ids: Option<Vec<String>>,
        sample_indices: Option<Vec<usize>>,
        mmap_window_mb: Option<usize>,
    ) -> PyResult<Self> {
        let maf = maf_threshold.unwrap_or(0.0);
        let miss = max_missing_rate.unwrap_or(1.0);
        let fill = fill_missing.unwrap_or(true);
        if mmap_window_mb.is_some()
            && (snp_range.is_some() || snp_indices.is_some() || bim_range.is_some())
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mmap_window_mb does not support snp_range/snp_indices/bim_range",
            ));
        }
        let it = if let Some(window_mb) = mmap_window_mb {
            BedSnpIter::new_with_fill_window(&prefix, maf, miss, fill, window_mb)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?
        } else {
            BedSnpIter::new_with_fill(&prefix, maf, miss, fill)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?
        };
        let (sample_indices, sample_ids) = build_sample_selection(
            &it.samples,
            sample_ids,
            sample_indices,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let snp_indices = build_snp_indices(
            &it.sites,
            snp_range,
            snp_indices,
            bim_range,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(Self {
            it,
            snp_indices,
            snp_pos: 0,
            sample_indices,
            sample_ids,
        })
    }

    #[getter]
    fn n_samples(&self) -> usize { self.sample_indices.len() }

    #[getter]
    fn n_snps(&self) -> usize {
        self.snp_indices.as_ref().map(|v| v.len()).unwrap_or(self.it.sites.len())
    }

    #[getter]
    fn sample_ids(&self) -> Vec<String> { self.sample_ids.clone() }

    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk_size: usize,
    ) -> PyResult<Option<(Bound<'py, PyArray2<f32>>, Vec<SiteInfo>)>> {
        if chunk_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("chunk_size must be > 0"));
        }

        let n = self.sample_indices.len();
        if n == 0 {
            return Ok(None);
        }
        let mut data: Vec<f32> = Vec::with_capacity(chunk_size * n);
        let mut sites: Vec<SiteInfo> = Vec::with_capacity(chunk_size);
        let full_samples = self.sample_indices.len() == self.it.n_samples();
        let mut m: usize = 0;

        if let Some(ref snp_indices) = self.snp_indices {
            while m < chunk_size && self.snp_pos < snp_indices.len() {
                let snp_idx = snp_indices[self.snp_pos];
                self.snp_pos += 1;
                if let Some((row, site)) = self.it.get_snp_row(snp_idx) {
                    if full_samples {
                        data.extend_from_slice(&row);
                    } else {
                        data.extend(self.sample_indices.iter().map(|&i| row[i]));
                    }
                    sites.push(site.into());
                    m += 1;
                }
            }
        } else {
            while m < chunk_size {
                match self.it.next_snp() {
                    Some((row, site)) => {
                        if full_samples {
                            data.extend_from_slice(&row);
                        } else {
                            data.extend(self.sample_indices.iter().map(|&i| row[i]));
                        }
                        sites.push(site.into());
                        m += 1;
                    }
                    None => break,
                }
            }
        }
        if m == 0 { return Ok(None); }

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
    ))]
    fn new(
        path: String,
        maf_threshold: Option<f32>,
        max_missing_rate: Option<f32>,
        fill_missing: Option<bool>,
        sample_ids: Option<Vec<String>>,
        sample_indices: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let maf = maf_threshold.unwrap_or(0.0);
        let miss = max_missing_rate.unwrap_or(1.0);
        let fill = fill_missing.unwrap_or(true);
        let it = VcfSnpIter::new_with_fill(&path, maf, miss, fill)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let (sample_indices, sample_ids) = build_sample_selection(
            &it.samples,
            sample_ids,
            sample_indices,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(Self { it, sample_indices, sample_ids })
    }

    #[getter]
    fn sample_ids(&self) -> Vec<String> { self.sample_ids.clone() }

    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk_size: usize,
    ) -> PyResult<Option<(Bound<'py, PyArray2<f32>>, Vec<SiteInfo>)>> {
        if chunk_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("chunk_size must be > 0"));
        }

        let n = self.sample_indices.len();
        let full_samples = n == self.it.n_samples();
        let mut data: Vec<f32> = Vec::with_capacity(chunk_size * n);
        let mut sites: Vec<SiteInfo> = Vec::with_capacity(chunk_size);
        let mut m: usize = 0;

        while m < chunk_size {
            match self.it.next_snp() {
                Some((row, site)) => {
                    if full_samples {
                        data.extend_from_slice(&row);
                    } else {
                        data.extend(self.sample_indices.iter().map(|&i| row[i]));
                    }
                    sites.push(site.into());
                    m += 1;
                }
                None => break,
            }
        }
        if m == 0 { return Ok(None); }

        let mat = Array2::from_shape_vec((m, n), data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        #[allow(deprecated)]
        let py_mat = PyArray2::from_owned_array(py, mat).into_bound();
        Ok(Some((py_mat, sites)))
    }
}

// -------- count_vcf_snps (Py function) --------
#[pyfunction]
pub fn count_vcf_snps(path: String) -> PyResult<usize> {
    let p = Path::new(&path);
    let mut reader = core::open_text_maybe_gz(p)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let mut n: usize = 0;
    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader.read_line(&mut line)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        if bytes == 0 { break; }
        if line.starts_with('#') || line.trim().is_empty() { continue; }
        n += 1;
    }
    Ok(n)
}

// ============================================================
// Streaming PLINK writer: write .fam once, then stream .bim and .bed
// ============================================================

fn write_fam_simple(path: &Path, sample_ids: &[String], phenotype: Option<&[f64]>) -> Result<(), String> {
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
    fn write_chunk(&mut self, geno_chunk: PyReadonlyArray2<i8>, sites: Vec<SiteInfo>) -> PyResult<()> {
        let arr = geno_chunk.as_array();
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err("geno_chunk must be 2D (m_chunk, n_samples)"));
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
        let base = arr.as_ptr();      // *const i8

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
        self.bed.flush().map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        self.bim.flush().map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
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

/// Internal writer wrapper: either plain text or gzip-compressed text.
enum VcfOut {
    Plain(BufWriter<File>),
    Gzip(BufWriter<GzEncoder<File>>),
}

impl VcfOut {
    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        match self {
            VcfOut::Plain(w) => w.write_all(buf),
            VcfOut::Gzip(w) => w.write_all(buf),
        }
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            VcfOut::Plain(w) => w.flush(),
            VcfOut::Gzip(w) => w.flush(),
        }
    }

    /// Finish the writer. For gzip, this is critical to write the gzip trailer (CRC/footer).
    #[inline]
    fn finish(mut self) -> std::io::Result<()> {
        self.flush()?;
        match self {
            VcfOut::Plain(_w) => Ok(()),
            VcfOut::Gzip(w) => {
                let enc = w.into_inner()?; // BufWriter -> GzEncoder
                let _file = enc.finish()?; // finalize gzip stream and return File
                Ok(())
            }
        }
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

        let file = File::create(&path).map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        let mut out = if path.ends_with(".gz") {
            let enc = GzEncoder::new(file, Compression::default());
            VcfOut::Gzip(BufWriter::new(enc))
        } else {
            VcfOut::Plain(BufWriter::new(file))
        };

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
    fn write_chunk(&mut self, geno_chunk: PyReadonlyArray2<i8>, sites: Vec<SiteInfo>) -> PyResult<()> {
        let out = self
            .out
            .as_mut()
            .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("writer is closed"))?;

        let arr = geno_chunk.as_array();
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err("geno_chunk must be 2D (m_chunk, n_samples)"));
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
        out.flush().map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        Ok(())
    }

    /// Close writer. For gzip output, this finalizes the gzip stream (writes trailer).
    fn close(&mut self) -> PyResult<()> {
        if let Some(out) = self.out.take() {
            out.finish().map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
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
