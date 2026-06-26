use numpy::ndarray::ArrayView2;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::gfcore::SiteInfo as CoreSiteInfo;
use crate::gfreader::SiteInfo as PySiteInfo;
use crate::vcfout::VcfOut;

pub(crate) trait SiteRecord {
    fn chrom(&self) -> &str;
    fn pos(&self) -> i32;
    fn ref_allele(&self) -> &str;
    fn alt_allele(&self) -> &str;
}

impl SiteRecord for CoreSiteInfo {
    #[inline]
    fn chrom(&self) -> &str {
        &self.chrom
    }

    #[inline]
    fn pos(&self) -> i32 {
        self.pos
    }

    #[inline]
    fn ref_allele(&self) -> &str {
        &self.ref_allele
    }

    #[inline]
    fn alt_allele(&self) -> &str {
        &self.alt_allele
    }
}

impl SiteRecord for PySiteInfo {
    #[inline]
    fn chrom(&self) -> &str {
        &self.chrom
    }

    #[inline]
    fn pos(&self) -> i32 {
        self.pos
    }

    #[inline]
    fn ref_allele(&self) -> &str {
        &self.ref_allele
    }

    #[inline]
    fn alt_allele(&self) -> &str {
        &self.alt_allele
    }
}

pub(crate) trait SampleRecord {
    fn fid(&self) -> &str;
    fn iid(&self) -> &str;
}

struct SampleIdRecord {
    id: String,
}

impl SampleRecord for SampleIdRecord {
    #[inline]
    fn fid(&self) -> &str {
        &self.id
    }

    #[inline]
    fn iid(&self) -> &str {
        &self.id
    }
}

#[inline]
pub(crate) fn write_fam_simple(
    path: &Path,
    sample_ids: &[String],
    phenotype: Option<&[f64]>,
) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    for (i, sid) in sample_ids.iter().enumerate() {
        let ph = phenotype.map(|p| p[i]).unwrap_or(-9.0);
        writeln!(w, "{0}\t{0}\t0\t0\t1\t{1}", sid, ph).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[inline]
fn site_row_id<S: SiteRecord>(site: &S) -> String {
    format!("{}_{}", site.chrom(), site.pos())
}

#[inline]
fn vcf_gt_bytes_from_g_i8(g: i8) -> &'static [u8] {
    match g {
        0 => b"0/0",
        1 => b"0/1",
        2 => b"1/1",
        _ => b"./.",
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

#[inline]
fn plink2bits_from_g_i8(g: i8) -> u8 {
    match g {
        0 => 0b00,
        1 => 0b10,
        2 => 0b11,
        _ => 0b01,
    }
}

pub(crate) struct VcfWriter {
    out: VcfOut,
    n_samples: usize,
    written_sites: usize,
    line_buf: Vec<u8>,
}

impl VcfWriter {
    pub(crate) fn new(
        out: &str,
        sample_ids: &[String],
        source_tag: Option<&str>,
    ) -> Result<Self, String> {
        let mut out = VcfOut::from_path(out).map_err(|e| e.to_string())?;

        out.write_all(b"##fileformat=VCFv4.2\n")
            .map_err(|e| e.to_string())?;
        if let Some(tag) = source_tag {
            writeln!(&mut out, "##source={tag}").map_err(|e| e.to_string())?;
        }
        out.write_all(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
            .map_err(|e| e.to_string())?;

        out.write_all(b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
            .map_err(|e| e.to_string())?;
        for sid in sample_ids.iter() {
            out.write_all(b"\t").map_err(|e| e.to_string())?;
            out.write_all(sid.as_bytes()).map_err(|e| e.to_string())?;
        }
        out.write_all(b"\n").map_err(|e| e.to_string())?;

        Ok(Self {
            out,
            n_samples: sample_ids.len(),
            written_sites: 0,
            line_buf: Vec::with_capacity(256 + sample_ids.len().saturating_mul(4)),
        })
    }

    #[inline]
    pub(crate) fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[inline]
    pub(crate) fn written_sites(&self) -> usize {
        self.written_sites
    }

    pub(crate) fn write_site<S: SiteRecord>(
        &mut self,
        site: &S,
        gref: &str,
        galt: &str,
        row_i8: &[i8],
    ) -> Result<(), String> {
        if row_i8.len() != self.n_samples {
            return Err("Internal error: VCF row length mismatch".into());
        }

        self.line_buf.clear();
        write!(
            &mut self.line_buf,
            "{}\t{}\t{}\t{}\t{}\t.\tPASS\t.\tGT",
            site.chrom(),
            site.pos(),
            site_row_id(site),
            gref,
            galt
        )
        .map_err(|e| e.to_string())?;
        for &g in row_i8 {
            self.line_buf.push(b'\t');
            self.line_buf.extend_from_slice(vcf_gt_bytes_from_g_i8(g));
        }
        self.line_buf.push(b'\n');
        self.out
            .write_all(&self.line_buf)
            .map_err(|e| e.to_string())?;
        self.written_sites = self.written_sites.saturating_add(1);
        Ok(())
    }

    pub(crate) fn write_chunk<S: SiteRecord>(
        &mut self,
        arr: ArrayView2<'_, i8>,
        sites: &[S],
    ) -> Result<(), String> {
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err("geno_chunk must be 2D (m_chunk, n_samples)".into());
        }
        let m_chunk = shape[0];
        let n_samples = shape[1];
        if n_samples != self.n_samples {
            return Err(format!(
                "n_samples mismatch: writer expects {}, got {}",
                self.n_samples, n_samples
            ));
        }
        if sites.len() != m_chunk {
            return Err(format!(
                "sites length mismatch: sites={}, m_chunk={}",
                sites.len(),
                m_chunk
            ));
        }

        let strides = arr.strides();
        let s0 = strides[0] as isize;
        let s1 = strides[1] as isize;
        let base = arr.as_ptr();

        unsafe {
            for snp in 0..m_chunk {
                let site = &sites[snp];
                self.line_buf.clear();
                write!(
                    &mut self.line_buf,
                    "{}\t{}\t{}\t{}\t{}\t.\tPASS\t.\tGT",
                    site.chrom(),
                    site.pos(),
                    site_row_id(site),
                    site.ref_allele(),
                    site.alt_allele()
                )
                .map_err(|e| e.to_string())?;

                let snp_off = (snp as isize) * s0;
                for i in 0..self.n_samples {
                    let off = snp_off + (i as isize) * s1;
                    self.line_buf.push(b'\t');
                    self.line_buf
                        .extend_from_slice(vcf_gt_bytes_from_g_i8(*base.offset(off)));
                }
                self.line_buf.push(b'\n');
                self.out
                    .write_all(&self.line_buf)
                    .map_err(|e| e.to_string())?;
                self.written_sites = self.written_sites.saturating_add(1);
            }
        }

        Ok(())
    }

    pub(crate) fn flush(&mut self) -> Result<(), String> {
        self.out.flush().map_err(|e| e.to_string())
    }

    pub(crate) fn finish(self) -> Result<(), String> {
        self.out.finish().map_err(|e| e.to_string())
    }
}

pub(crate) struct HmpWriter {
    out: VcfOut,
    n_samples: usize,
    written_sites: usize,
    line_buf: Vec<u8>,
}

impl HmpWriter {
    pub(crate) fn new(path: &str, sample_ids: &[String]) -> Result<Self, String> {
        let mut out = VcfOut::from_path(path).map_err(|e| e.to_string())?;
        out.write_all(
            b"rs#\talleles\tchrom\tpos\tstrand\tassembly#\tcenter\tprotLSID\tassayLSID\tpanelLSID\tQCcode",
        )
        .map_err(|e| e.to_string())?;
        for sid in sample_ids.iter() {
            out.write_all(b"\t").map_err(|e| e.to_string())?;
            out.write_all(sid.as_bytes()).map_err(|e| e.to_string())?;
        }
        out.write_all(b"\n").map_err(|e| e.to_string())?;
        Ok(Self {
            out,
            n_samples: sample_ids.len(),
            written_sites: 0,
            line_buf: Vec::with_capacity(256 + sample_ids.len().saturating_mul(3)),
        })
    }

    #[inline]
    pub(crate) fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[inline]
    pub(crate) fn written_sites(&self) -> usize {
        self.written_sites
    }

    pub(crate) fn write_site<S: SiteRecord>(
        &mut self,
        site: &S,
        gref: &str,
        galt: &str,
        row_i8: &[i8],
    ) -> Result<(), String> {
        if row_i8.len() != self.n_samples {
            return Err("Internal error: HMP row length mismatch".into());
        }

        let ref_b = hmp_base_byte(gref);
        let mut alt_b = hmp_base_byte(galt);
        if alt_b == ref_b {
            alt_b = if ref_b == b'A' { b'C' } else { b'A' };
        }

        self.line_buf.clear();
        write!(
            &mut self.line_buf,
            "{}\t{}/{}\t{}\t{}\t+\t.\t.\t.\t.\t.\t.",
            site_row_id(site),
            ref_b as char,
            alt_b as char,
            site.chrom(),
            site.pos()
        )
        .map_err(|e| e.to_string())?;
        for &g in row_i8 {
            let gt = hmp_gt_bytes_from_g_i8(g, ref_b, alt_b);
            self.line_buf.push(b'\t');
            self.line_buf.extend_from_slice(&gt);
        }
        self.line_buf.push(b'\n');
        self.out
            .write_all(&self.line_buf)
            .map_err(|e| e.to_string())?;
        self.written_sites = self.written_sites.saturating_add(1);
        Ok(())
    }

    pub(crate) fn write_chunk<S: SiteRecord>(
        &mut self,
        arr: ArrayView2<'_, i8>,
        sites: &[S],
    ) -> Result<(), String> {
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err("geno_chunk must be 2D (m_chunk, n_samples)".into());
        }
        let m_chunk = shape[0];
        let n_samples = shape[1];
        if n_samples != self.n_samples {
            return Err(format!(
                "n_samples mismatch: writer expects {}, got {}",
                self.n_samples, n_samples
            ));
        }
        if sites.len() != m_chunk {
            return Err(format!(
                "sites length mismatch: sites={}, m_chunk={}",
                sites.len(),
                m_chunk
            ));
        }

        let strides = arr.strides();
        let s0 = strides[0] as isize;
        let s1 = strides[1] as isize;
        let base = arr.as_ptr();

        unsafe {
            for snp in 0..m_chunk {
                let site = &sites[snp];
                let ref_b = hmp_base_byte(site.ref_allele());
                let mut alt_b = hmp_base_byte(site.alt_allele());
                if alt_b == ref_b {
                    alt_b = if ref_b == b'A' { b'C' } else { b'A' };
                }

                self.line_buf.clear();
                write!(
                    &mut self.line_buf,
                    "{}\t{}/{}\t{}\t{}\t+\t.\t.\t.\t.\t.\t.",
                    site_row_id(site),
                    ref_b as char,
                    alt_b as char,
                    site.chrom(),
                    site.pos()
                )
                .map_err(|e| e.to_string())?;

                let snp_off = (snp as isize) * s0;
                for i in 0..self.n_samples {
                    let off = snp_off + (i as isize) * s1;
                    let gt = hmp_gt_bytes_from_g_i8(*base.offset(off), ref_b, alt_b);
                    self.line_buf.push(b'\t');
                    self.line_buf.extend_from_slice(&gt);
                }
                self.line_buf.push(b'\n');
                self.out
                    .write_all(&self.line_buf)
                    .map_err(|e| e.to_string())?;
                self.written_sites = self.written_sites.saturating_add(1);
            }
        }

        Ok(())
    }

    pub(crate) fn flush(&mut self) -> Result<(), String> {
        self.out.flush().map_err(|e| e.to_string())
    }

    pub(crate) fn finish(self) -> Result<(), String> {
        self.out.finish().map_err(|e| e.to_string())
    }
}

pub(crate) struct PlinkBfileWriter {
    fam: BufWriter<File>,
    bim: BufWriter<File>,
    bed: BufWriter<File>,
    n_samples: usize,
    bytes_per_snp: usize,
    row_buf: Vec<u8>,
    line_buf: Vec<u8>,
    written_sites: usize,
}

impl PlinkBfileWriter {
    pub(crate) fn new<S: SampleRecord>(
        prefix: &str,
        samples: &[S],
        phenotype: Option<&[f64]>,
    ) -> Result<Self, String> {
        if let Some(ph) = phenotype {
            if ph.len() != samples.len() {
                return Err(format!(
                    "phenotype length mismatch: phenotype={}, n_samples={}",
                    ph.len(),
                    samples.len()
                ));
            }
        }

        let mut fam = BufWriter::with_capacity(
            4 * 1024 * 1024,
            File::create(format!("{prefix}.fam")).map_err(|e| e.to_string())?,
        );
        for (i, sample) in samples.iter().enumerate() {
            let ph = phenotype.map(|p| p[i]).unwrap_or(-9.0);
            writeln!(fam, "{}\t{}\t0\t0\t1\t{}", sample.fid(), sample.iid(), ph)
                .map_err(|e| e.to_string())?;
        }
        fam.flush().map_err(|e| e.to_string())?;

        let bim = BufWriter::with_capacity(
            4 * 1024 * 1024,
            File::create(format!("{prefix}.bim")).map_err(|e| e.to_string())?,
        );
        let mut bed = BufWriter::with_capacity(
            8 * 1024 * 1024,
            File::create(format!("{prefix}.bed")).map_err(|e| e.to_string())?,
        );
        bed.write_all(&[0x6C, 0x1B, 0x01])
            .map_err(|e| e.to_string())?;

        let n_samples = samples.len();
        let bytes_per_snp = (n_samples + 3) / 4;
        Ok(Self {
            fam,
            bim,
            bed,
            n_samples,
            bytes_per_snp,
            row_buf: vec![0u8; bytes_per_snp],
            line_buf: Vec::with_capacity(128),
            written_sites: 0,
        })
    }

    #[inline]
    pub(crate) fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[inline]
    pub(crate) fn written_sites(&self) -> usize {
        self.written_sites
    }

    #[inline]
    fn encode_row_i8_into(&mut self, row: &[i8]) {
        for byte_idx in 0..self.bytes_per_snp {
            let mut b: u8 = 0;
            for within in 0..4 {
                let i = byte_idx * 4 + within;
                let code: u8 = if i >= self.n_samples {
                    0b01
                } else {
                    plink2bits_from_g_i8(row[i])
                };
                b |= code << (within * 2);
            }
            self.row_buf[byte_idx] = b;
        }
    }

    pub(crate) fn write_site<S: SiteRecord>(
        &mut self,
        site: &S,
        gref: &str,
        galt: &str,
        row_i8: &[i8],
    ) -> Result<(), String> {
        if row_i8.len() != self.n_samples {
            return Err("Internal error: PLINK row length mismatch".into());
        }

        self.line_buf.clear();
        writeln!(
            &mut self.line_buf,
            "{}\t{}\t0\t{}\t{}\t{}",
            site.chrom(),
            site_row_id(site),
            site.pos(),
            gref,
            galt
        )
        .map_err(|e| e.to_string())?;
        self.bim
            .write_all(&self.line_buf)
            .map_err(|e| e.to_string())?;

        self.encode_row_i8_into(row_i8);
        self.bed
            .write_all(&self.row_buf)
            .map_err(|e| e.to_string())?;
        self.written_sites = self.written_sites.saturating_add(1);
        Ok(())
    }

    pub(crate) fn write_site_encoded<S: SiteRecord>(
        &mut self,
        site: &S,
        gref: &str,
        galt: &str,
        row_bed: &[u8],
    ) -> Result<(), String> {
        if row_bed.len() != self.bytes_per_snp {
            return Err(format!(
                "Internal error: PLINK encoded row length mismatch: expected {}, got {}",
                self.bytes_per_snp,
                row_bed.len()
            ));
        }

        self.line_buf.clear();
        writeln!(
            &mut self.line_buf,
            "{}\t{}\t0\t{}\t{}\t{}",
            site.chrom(),
            site_row_id(site),
            site.pos(),
            gref,
            galt
        )
        .map_err(|e| e.to_string())?;
        self.bim
            .write_all(&self.line_buf)
            .map_err(|e| e.to_string())?;
        self.bed.write_all(row_bed).map_err(|e| e.to_string())?;
        self.written_sites = self.written_sites.saturating_add(1);
        Ok(())
    }

    pub(crate) fn write_chunk<S: SiteRecord>(
        &mut self,
        arr: ArrayView2<'_, i8>,
        sites: &[S],
    ) -> Result<(), String> {
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err("geno_chunk must be 2D (m_chunk, n_samples)".into());
        }
        let m_chunk = shape[0];
        let n_samples = shape[1];
        if n_samples != self.n_samples {
            return Err(format!(
                "n_samples mismatch: writer expects {}, got {}",
                self.n_samples, n_samples
            ));
        }
        if sites.len() != m_chunk {
            return Err(format!(
                "sites length mismatch: sites={}, m_chunk={}",
                sites.len(),
                m_chunk
            ));
        }

        let strides = arr.strides();
        let s0 = strides[0] as isize;
        let s1 = strides[1] as isize;
        let base = arr.as_ptr();

        for site in sites.iter() {
            self.line_buf.clear();
            writeln!(
                &mut self.line_buf,
                "{}\t{}\t0\t{}\t{}\t{}",
                site.chrom(),
                site_row_id(site),
                site.pos(),
                site.ref_allele(),
                site.alt_allele()
            )
            .map_err(|e| e.to_string())?;
            self.bim
                .write_all(&self.line_buf)
                .map_err(|e| e.to_string())?;
        }

        unsafe {
            for snp in 0..m_chunk {
                let snp_off = (snp as isize) * s0;
                let mut i = 0usize;
                for out_b in 0..self.bytes_per_snp {
                    let mut byte: u8 = 0;
                    for k in 0..4 {
                        let si = i + k;
                        let two = if si < self.n_samples {
                            let off = snp_off + (si as isize) * s1;
                            plink2bits_from_g_i8(*base.offset(off))
                        } else {
                            0b01
                        };
                        byte |= two << (k * 2);
                    }
                    self.row_buf[out_b] = byte;
                    i += 4;
                }
                self.bed
                    .write_all(&self.row_buf)
                    .map_err(|e| e.to_string())?;
                self.written_sites = self.written_sites.saturating_add(1);
            }
        }

        Ok(())
    }

    pub(crate) fn flush(&mut self) -> Result<(), String> {
        self.fam.flush().map_err(|e| e.to_string())?;
        self.bim.flush().map_err(|e| e.to_string())?;
        self.bed.flush().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<(), String> {
        self.flush()
    }
}

#[pyclass]
pub struct PlinkStreamWriter {
    inner: PlinkBfileWriter,
}

#[pymethods]
impl PlinkStreamWriter {
    #[new]
    #[pyo3(signature = (prefix, sample_ids, phenotype))]
    fn new(prefix: String, sample_ids: Vec<String>, phenotype: Option<Vec<f64>>) -> PyResult<Self> {
        if sample_ids.is_empty() {
            return Err(PyValueError::new_err("sample_ids is empty"));
        }
        let samples: Vec<SampleIdRecord> = sample_ids
            .into_iter()
            .map(|id| SampleIdRecord { id })
            .collect();
        let inner = PlinkBfileWriter::new(&prefix, &samples, phenotype.as_deref())
            .map_err(PyIOError::new_err)?;
        Ok(Self { inner })
    }

    fn write_chunk(
        &mut self,
        geno_chunk: PyReadonlyArray2<i8>,
        sites: Vec<PySiteInfo>,
    ) -> PyResult<()> {
        self.inner
            .write_chunk(geno_chunk.as_array(), &sites)
            .map_err(PyValueError::new_err)
    }

    fn flush(&mut self) -> PyResult<()> {
        self.inner.flush().map_err(PyIOError::new_err)
    }

    fn close(&mut self) -> PyResult<()> {
        self.flush()
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }

    #[getter]
    fn written_snps(&self) -> usize {
        self.inner.written_sites()
    }
}

#[pyclass]
pub struct VcfStreamWriter {
    inner: Option<VcfWriter>,
}

#[pymethods]
impl VcfStreamWriter {
    #[new]
    fn new(path: String, sample_ids: Vec<String>) -> PyResult<Self> {
        if sample_ids.is_empty() {
            return Err(PyValueError::new_err("sample_ids is empty"));
        }
        let inner = VcfWriter::new(&path, &sample_ids, None).map_err(PyIOError::new_err)?;
        Ok(Self { inner: Some(inner) })
    }

    fn write_chunk(
        &mut self,
        geno_chunk: PyReadonlyArray2<i8>,
        sites: Vec<PySiteInfo>,
    ) -> PyResult<()> {
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("writer is closed"))?;
        inner
            .write_chunk(geno_chunk.as_array(), &sites)
            .map_err(PyValueError::new_err)
    }

    fn flush(&mut self) -> PyResult<()> {
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("writer is closed"))?;
        inner.flush().map_err(PyIOError::new_err)
    }

    fn close(&mut self) -> PyResult<()> {
        if let Some(inner) = self.inner.take() {
            inner.finish().map_err(PyIOError::new_err)?;
        }
        Ok(())
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.as_ref().map(|w| w.n_samples()).unwrap_or(0)
    }

    #[getter]
    fn written_snps(&self) -> usize {
        self.inner.as_ref().map(|w| w.written_sites()).unwrap_or(0)
    }
}

#[pyclass]
pub struct HmpStreamWriter {
    inner: Option<HmpWriter>,
}

#[pymethods]
impl HmpStreamWriter {
    #[new]
    fn new(path: String, sample_ids: Vec<String>) -> PyResult<Self> {
        if sample_ids.is_empty() {
            return Err(PyValueError::new_err("sample_ids is empty"));
        }
        let inner = HmpWriter::new(&path, &sample_ids).map_err(PyIOError::new_err)?;
        Ok(Self { inner: Some(inner) })
    }

    fn write_chunk(
        &mut self,
        geno_chunk: PyReadonlyArray2<i8>,
        sites: Vec<PySiteInfo>,
    ) -> PyResult<()> {
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("writer is closed"))?;
        inner
            .write_chunk(geno_chunk.as_array(), &sites)
            .map_err(PyValueError::new_err)
    }

    fn flush(&mut self) -> PyResult<()> {
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("writer is closed"))?;
        inner.flush().map_err(PyIOError::new_err)
    }

    fn close(&mut self) -> PyResult<()> {
        if let Some(inner) = self.inner.take() {
            inner.finish().map_err(PyIOError::new_err)?;
        }
        Ok(())
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.as_ref().map(|w| w.n_samples()).unwrap_or(0)
    }

    #[getter]
    fn written_snps(&self) -> usize {
        self.inner.as_ref().map(|w| w.written_sites()).unwrap_or(0)
    }
}
