use crate::bincore::{bin01_site_sidecar_path, write_bin01_header};
use crate::binsidecar::write_bin_site_header;
use crate::gfcore::SiteInfo as CoreSiteInfo;
use numpy::ndarray::{s, ArrayView2};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bin01SiteMode {
    None,
    LegacyKmerBinary,
    TextTsv,
}

impl Bin01SiteMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        let mode = raw.trim().to_ascii_lowercase();
        match mode.as_str() {
            "" | "none" => Ok(Self::None),
            "kmer" | "legacy-kmer" | "legacy_kmer" | "binary-kmer" | "binary_kmer" => {
                Ok(Self::LegacyKmerBinary)
            }
            "text" | "text-tsv" | "text_tsv" | "tsv" => Ok(Self::TextTsv),
            _ => Err(format!(
                "unsupported BIN01 site mode '{raw}'; expected one of: none, kmer, text"
            )),
        }
    }

    #[inline]
    pub fn requires_site_metadata(self) -> bool {
        !matches!(self, Self::None)
    }

    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LegacyKmerBinary => "kmer",
            Self::TextTsv => "text",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Bin01SiteRecordOwned {
    pub chrom: String,
    pub pos: i32,
    pub ref_allele: String,
    pub alt_allele: String,
}

#[derive(Clone, Copy, Debug)]
pub struct Bin01SiteRecordRef<'a> {
    pub chrom: &'a str,
    pub pos: i32,
    pub ref_allele: &'a str,
    pub alt_allele: &'a str,
}

impl<'a> From<&'a CoreSiteInfo> for Bin01SiteRecordRef<'a> {
    fn from(site: &'a CoreSiteInfo) -> Self {
        Self {
            chrom: site.chrom.as_str(),
            pos: site.pos,
            ref_allele: site.ref_allele.as_str(),
            alt_allele: site.alt_allele.as_str(),
        }
    }
}

impl<'a> From<&'a Bin01SiteRecordOwned> for Bin01SiteRecordRef<'a> {
    fn from(site: &'a Bin01SiteRecordOwned) -> Self {
        Self {
            chrom: site.chrom.as_str(),
            pos: site.pos,
            ref_allele: site.ref_allele.as_str(),
            alt_allele: site.alt_allele.as_str(),
        }
    }
}

enum Bin01SidecarWriter {
    None,
    LegacyKmerBinary(BufWriter<File>),
    TextTsv(BufWriter<File>),
}

pub struct Bin01Writer {
    n_samples: usize,
    row_bytes: usize,
    n_rows_written: usize,
    payload: BufWriter<File>,
    sidecar: Bin01SidecarWriter,
    site_mode: Bin01SiteMode,
}

impl Bin01Writer {
    pub fn new(
        out_bin_path: &str,
        n_samples: usize,
        site_mode: Bin01SiteMode,
    ) -> Result<Self, String> {
        if n_samples == 0 {
            return Err("BIN01 writer requires n_samples > 0".to_string());
        }
        let out_path = Path::new(out_bin_path);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create parent dir {}: {e}", parent.display()))?;
        }

        let mut payload = BufWriter::with_capacity(
            8 * 1024 * 1024,
            File::create(out_path).map_err(|e| format!("create {}: {e}", out_path.display()))?,
        );
        write_bin01_header(&mut payload, 0, n_samples, "binwriter init")?;

        let sidecar = match site_mode {
            Bin01SiteMode::None => Bin01SidecarWriter::None,
            Bin01SiteMode::LegacyKmerBinary => {
                let site_path = bin01_site_sidecar_path(out_bin_path);
                if let Some(parent) = site_path.parent() {
                    fs::create_dir_all(parent)
                        .map_err(|e| format!("create parent dir {}: {e}", parent.display()))?;
                }
                let mut fw = BufWriter::with_capacity(
                    4 * 1024 * 1024,
                    File::create(&site_path)
                        .map_err(|e| format!("create {}: {e}", site_path.display()))?,
                );
                write_bin_site_header(&mut fw, 0, "binwriter init site")?;
                Bin01SidecarWriter::LegacyKmerBinary(fw)
            }
            Bin01SiteMode::TextTsv => {
                let site_path = bin01_site_sidecar_path(out_bin_path);
                if let Some(parent) = site_path.parent() {
                    fs::create_dir_all(parent)
                        .map_err(|e| format!("create parent dir {}: {e}", parent.display()))?;
                }
                let fw = BufWriter::with_capacity(
                    4 * 1024 * 1024,
                    File::create(&site_path)
                        .map_err(|e| format!("create {}: {e}", site_path.display()))?,
                );
                Bin01SidecarWriter::TextTsv(fw)
            }
        };

        Ok(Self {
            n_samples,
            row_bytes: n_samples.div_ceil(8),
            n_rows_written: 0,
            payload,
            sidecar,
            site_mode,
        })
    }

    #[inline]
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    #[inline]
    pub fn written_rows(&self) -> usize {
        self.n_rows_written
    }

    pub fn flush(&mut self) -> Result<(), String> {
        self.payload
            .flush()
            .map_err(|e| format!("flush BIN01 payload: {e}"))?;
        match &mut self.sidecar {
            Bin01SidecarWriter::None => {}
            Bin01SidecarWriter::LegacyKmerBinary(fw) => {
                fw.flush()
                    .map_err(|e| format!("flush BIN01 legacy site sidecar: {e}"))?;
            }
            Bin01SidecarWriter::TextTsv(fw) => {
                fw.flush()
                    .map_err(|e| format!("flush BIN01 text site sidecar: {e}"))?;
            }
        }
        Ok(())
    }

    pub fn write_bitrow(
        &mut self,
        row_bits: &[u8],
        site: Option<Bin01SiteRecordRef<'_>>,
    ) -> Result<(), String> {
        if row_bits.len() != self.row_bytes {
            return Err(format!(
                "BIN01 row byte mismatch: got {}, expected {}",
                row_bits.len(),
                self.row_bytes
            ));
        }
        self.payload
            .write_all(row_bits)
            .map_err(|e| format!("write BIN01 row: {e}"))?;
        self.write_site_record(site)?;
        self.n_rows_written = self.n_rows_written.saturating_add(1);
        Ok(())
    }

    pub fn write_f32_block(
        &mut self,
        block: ArrayView2<'_, f32>,
        sites: Option<&[Bin01SiteRecordOwned]>,
    ) -> Result<usize, String> {
        let shape = block.shape();
        if shape.len() != 2 {
            return Err("BIN01 chunk must be 2D (n_rows, n_samples)".to_string());
        }
        let n_rows = shape[0];
        let n_cols = shape[1];
        if n_cols != self.n_samples {
            return Err(format!(
                "BIN01 chunk sample mismatch: got {}, expected {}",
                n_cols, self.n_samples
            ));
        }
        validate_site_records(self.site_mode, n_rows, sites)?;

        let mut row_buf = vec![0u8; self.row_bytes];
        for row_idx in 0..n_rows {
            row_buf.fill(0);
            let row = block.slice(s![row_idx, ..]);
            for (col_idx, value) in row.iter().enumerate() {
                if *value > 0.0 {
                    row_buf[col_idx >> 3] |= 1u8 << (col_idx & 7);
                }
            }
            let site = sites.map(|items| Bin01SiteRecordRef::from(&items[row_idx]));
            self.write_bitrow(row_buf.as_slice(), site)?;
        }
        Ok(n_rows)
    }

    pub fn write_packed_block(
        &mut self,
        packed: ArrayView2<'_, u8>,
        sites: Option<&[Bin01SiteRecordOwned]>,
    ) -> Result<usize, String> {
        let shape = packed.shape();
        if shape.len() != 2 {
            return Err("BIN01 packed chunk must be 2D (n_rows, row_bytes)".to_string());
        }
        let n_rows = shape[0];
        let n_cols = shape[1];
        if n_cols != self.row_bytes {
            return Err(format!(
                "BIN01 packed chunk byte mismatch: got {}, expected {}",
                n_cols, self.row_bytes
            ));
        }
        validate_site_records(self.site_mode, n_rows, sites)?;

        let mut row_buf = vec![0u8; self.row_bytes];
        for row_idx in 0..n_rows {
            let row = packed.slice(s![row_idx, ..]);
            let row_slice = if let Some(slice) = row.as_slice() {
                slice
            } else {
                for (dst, src) in row_buf.iter_mut().zip(row.iter()) {
                    *dst = *src;
                }
                row_buf.as_slice()
            };
            let site = sites.map(|items| Bin01SiteRecordRef::from(&items[row_idx]));
            self.write_bitrow(row_slice, site)?;
        }
        Ok(n_rows)
    }

    pub fn finish(mut self) -> Result<usize, String> {
        self.flush()?;

        {
            let payload = self.payload.get_mut();
            payload
                .seek(SeekFrom::Start(0))
                .map_err(|e| format!("seek BIN01 header rewrite: {e}"))?;
            write_bin01_header(
                payload,
                self.n_rows_written as u64,
                self.n_samples,
                "binwriter finalize",
            )?;
            payload
                .flush()
                .map_err(|e| format!("flush BIN01 header rewrite: {e}"))?;
        }

        if let Bin01SidecarWriter::LegacyKmerBinary(fw) = &mut self.sidecar {
            let site_file = fw.get_mut();
            site_file
                .seek(SeekFrom::Start(0))
                .map_err(|e| format!("seek BIN01 site header rewrite: {e}"))?;
            write_bin_site_header(
                site_file,
                self.n_rows_written as u64,
                "binwriter finalize site",
            )?;
            site_file
                .flush()
                .map_err(|e| format!("flush BIN01 site header rewrite: {e}"))?;
        }

        Ok(self.n_rows_written)
    }

    fn write_site_record(&mut self, site: Option<Bin01SiteRecordRef<'_>>) -> Result<(), String> {
        match &mut self.sidecar {
            Bin01SidecarWriter::None => Ok(()),
            Bin01SidecarWriter::LegacyKmerBinary(fw) => {
                let site = site.ok_or_else(|| {
                    "missing site metadata for BIN01 legacy kmer sidecar".to_string()
                })?;
                write_legacy_bin_site_record(fw, site)
            }
            Bin01SidecarWriter::TextTsv(fw) => {
                let site =
                    site.ok_or_else(|| "missing site metadata for BIN01 text sidecar".to_string())?;
                writeln!(
                    fw,
                    "{}\t{}\t{}\t{}",
                    site.chrom, site.pos, site.ref_allele, site.alt_allele
                )
                .map_err(|e| format!("write BIN01 text site row: {e}"))
            }
        }
    }
}

fn validate_site_records(
    site_mode: Bin01SiteMode,
    n_rows: usize,
    sites: Option<&[Bin01SiteRecordOwned]>,
) -> Result<(), String> {
    if site_mode.requires_site_metadata() {
        let items = sites.ok_or_else(|| {
            format!(
                "BIN01 site mode '{}' requires site metadata for every row",
                site_mode.as_str()
            )
        })?;
        if items.len() != n_rows {
            return Err(format!(
                "BIN01 site metadata length mismatch: sites={}, n_rows={}",
                items.len(),
                n_rows
            ));
        }
    }
    Ok(())
}

fn write_legacy_bin_site_record<W: Write>(
    writer: &mut W,
    site: Bin01SiteRecordRef<'_>,
) -> Result<(), String> {
    let kmer = site.alt_allele.trim();
    if kmer.is_empty() {
        return Err("empty k-mer is not allowed for BIN01 legacy site sidecar".to_string());
    }
    if kmer.len() > u16::MAX as usize {
        return Err(format!("k-mer length exceeds u16 limit: {}", kmer.len()));
    }
    let packed = encode_kmer_2bit(kmer)?;
    writer
        .write_all(&(kmer.len() as u16).to_le_bytes())
        .map_err(|e| format!("write BIN01 legacy site kmer length: {e}"))?;
    writer
        .write_all(&packed)
        .map_err(|e| format!("write BIN01 legacy site kmer payload: {e}"))
}

fn encode_kmer_2bit(seq: &str) -> Result<Vec<u8>, String> {
    let raw = seq.as_bytes();
    let mut out = vec![0u8; raw.len().div_ceil(4)];
    for (idx, ch) in raw.iter().enumerate() {
        let code = match *ch {
            b'A' | b'a' => 0u8,
            b'T' | b't' => 1u8,
            b'C' | b'c' => 2u8,
            b'G' | b'g' => 3u8,
            _ => {
                return Err(format!(
                    "unsupported base in BIN01 legacy k-mer sidecar: {:?}",
                    char::from(*ch)
                ))
            }
        };
        out[idx >> 2] |= (code & 0b11) << ((idx & 0b11) * 2);
    }
    Ok(out)
}

#[pyclass]
pub struct Bin01StreamWriter {
    inner: Option<Bin01Writer>,
    n_samples: usize,
    row_bytes: usize,
    written_rows: usize,
    site_mode: Bin01SiteMode,
}

#[pymethods]
impl Bin01StreamWriter {
    #[new]
    #[pyo3(signature = (path, n_samples, site_mode = "none"))]
    fn new(path: String, n_samples: usize, site_mode: &str) -> PyResult<Self> {
        let parsed_mode = Bin01SiteMode::parse(site_mode).map_err(PyValueError::new_err)?;
        let inner = Bin01Writer::new(&path, n_samples, parsed_mode)
            .map_err(|e| PyErr::new::<PyIOError, _>(e))?;
        Ok(Self {
            row_bytes: inner.row_bytes(),
            inner: Some(inner),
            n_samples,
            written_rows: 0,
            site_mode: parsed_mode,
        })
    }

    #[pyo3(signature = (geno_chunk, chrom = None, pos = None, ref_allele = None, alt_allele = None))]
    fn write_chunk_f32(
        &mut self,
        geno_chunk: PyReadonlyArray2<'_, f32>,
        chrom: Option<Vec<String>>,
        pos: Option<Vec<i32>>,
        ref_allele: Option<Vec<String>>,
        alt_allele: Option<Vec<String>>,
    ) -> PyResult<usize> {
        let view = geno_chunk.as_array();
        let records = collect_site_records(
            self.site_mode,
            view.shape()[0],
            chrom,
            pos,
            ref_allele,
            alt_allele,
        )?;
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("BIN01 writer is closed"))?;
        let written = inner
            .write_f32_block(view, records.as_deref())
            .map_err(PyRuntimeError::new_err)?;
        self.written_rows = inner.written_rows();
        Ok(written)
    }

    #[pyo3(signature = (packed_chunk, chrom = None, pos = None, ref_allele = None, alt_allele = None))]
    fn write_chunk_packed(
        &mut self,
        packed_chunk: PyReadonlyArray2<'_, u8>,
        chrom: Option<Vec<String>>,
        pos: Option<Vec<i32>>,
        ref_allele: Option<Vec<String>>,
        alt_allele: Option<Vec<String>>,
    ) -> PyResult<usize> {
        let view = packed_chunk.as_array();
        let records = collect_site_records(
            self.site_mode,
            view.shape()[0],
            chrom,
            pos,
            ref_allele,
            alt_allele,
        )?;
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("BIN01 writer is closed"))?;
        let written = inner
            .write_packed_block(view, records.as_deref())
            .map_err(PyRuntimeError::new_err)?;
        self.written_rows = inner.written_rows();
        Ok(written)
    }

    fn flush(&mut self) -> PyResult<()> {
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("BIN01 writer is closed"))?;
        inner.flush().map_err(PyRuntimeError::new_err)
    }

    fn close(&mut self) -> PyResult<usize> {
        let Some(inner) = self.inner.take() else {
            return Ok(self.written_rows);
        };
        let written = inner.finish().map_err(PyRuntimeError::new_err)?;
        self.written_rows = written;
        Ok(written)
    }

    fn finish(&mut self) -> PyResult<usize> {
        self.close()
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[getter]
    fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    #[getter]
    fn written_rows(&self) -> usize {
        self.written_rows
    }

    #[getter]
    fn site_mode(&self) -> &'static str {
        self.site_mode.as_str()
    }
}

fn collect_site_records(
    site_mode: Bin01SiteMode,
    n_rows: usize,
    chrom: Option<Vec<String>>,
    pos: Option<Vec<i32>>,
    ref_allele: Option<Vec<String>>,
    alt_allele: Option<Vec<String>>,
) -> PyResult<Option<Vec<Bin01SiteRecordOwned>>> {
    if !site_mode.requires_site_metadata() {
        return Ok(None);
    }
    let chrom =
        chrom.ok_or_else(|| PyValueError::new_err("BIN01 writer missing chrom metadata"))?;
    let pos = pos.ok_or_else(|| PyValueError::new_err("BIN01 writer missing pos metadata"))?;
    let ref_allele =
        ref_allele.ok_or_else(|| PyValueError::new_err("BIN01 writer missing ref metadata"))?;
    let alt_allele =
        alt_allele.ok_or_else(|| PyValueError::new_err("BIN01 writer missing alt metadata"))?;

    if chrom.len() != n_rows
        || pos.len() != n_rows
        || ref_allele.len() != n_rows
        || alt_allele.len() != n_rows
    {
        return Err(PyValueError::new_err(format!(
            "BIN01 site metadata length mismatch: chrom={}, pos={}, ref={}, alt={}, n_rows={}",
            chrom.len(),
            pos.len(),
            ref_allele.len(),
            alt_allele.len(),
            n_rows
        )));
    }

    Ok(Some(
        chrom
            .into_iter()
            .zip(pos)
            .zip(ref_allele)
            .zip(alt_allele)
            .map(
                |(((chrom, pos), ref_allele), alt_allele)| Bin01SiteRecordOwned {
                    chrom,
                    pos,
                    ref_allele,
                    alt_allele,
                },
            )
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{Bin01SiteMode, Bin01SiteRecordOwned, Bin01Writer};
    use crate::gfcore::TxtSnpIter;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_dir(prefix: &str) -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_{prefix}_{stamp}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn writes_legacy_kmer_bin01_roundtrip() {
        let dir = make_temp_dir("binwriter_kmer");
        let bin_path = dir.join("toy.bin");
        let bin_str = bin_path.to_str().unwrap();

        let mut writer = Bin01Writer::new(bin_str, 5, Bin01SiteMode::LegacyKmerBinary).unwrap();
        let packed = [0b0001_0101u8, 0b0000_1010u8];
        let sites = vec![
            Bin01SiteRecordOwned {
                chrom: "KMER".to_string(),
                pos: 1,
                ref_allele: "A".to_string(),
                alt_allele: "ATCG".to_string(),
            },
            Bin01SiteRecordOwned {
                chrom: "KMER".to_string(),
                pos: 2,
                ref_allele: "A".to_string(),
                alt_allele: "TGCA".to_string(),
            },
        ];
        writer
            .write_packed_rows_with_sites_for_test(&packed, &sites)
            .unwrap();
        writer.finish().unwrap();

        let mut it = TxtSnpIter::new(bin_str, None).unwrap();
        assert_eq!(it.n_samples(), 5);
        assert_eq!(it.sites.len(), 2);
        assert_eq!(it.sites[0].alt_allele, "ATCG");
        assert_eq!(it.sites[1].alt_allele, "TGCA");
        let (row0, _) = it.next_snp().unwrap();
        assert_eq!(row0, vec![1.0, 0.0, 1.0, 0.0, 1.0]);
        let (row1, _) = it.next_snp().unwrap();
        assert_eq!(row1, vec![0.0, 1.0, 0.0, 1.0, 0.0]);

        let _ = fs::remove_dir_all(&dir);
    }

    impl Bin01Writer {
        fn write_packed_rows_with_sites_for_test(
            &mut self,
            packed_rows: &[u8],
            sites: &[Bin01SiteRecordOwned],
        ) -> Result<(), String> {
            if packed_rows.len() != sites.len() * self.row_bytes {
                return Err("test packed rows/site mismatch".to_string());
            }
            for (row_idx, site) in sites.iter().enumerate() {
                let start = row_idx * self.row_bytes;
                let end = start + self.row_bytes;
                self.write_bitrow(
                    &packed_rows[start..end],
                    Some(super::Bin01SiteRecordRef::from(site)),
                )?;
            }
            Ok(())
        }
    }
}
