use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use flate2::read::MultiGzDecoder;
use memmap2::Mmap;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::Bound;

/// 位点信息：chrom, pos, ref, alt
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

/// 读取 .fam，返回样本 ID 列表（第二列 IID）
fn read_fam(prefix: &str) -> Result<Vec<String>, String> {
    let fam_path = format!("{prefix}.fam");
    let file = File::open(&fam_path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);

    let mut samples = Vec::new();
    for line in reader.lines() {
        let l = line.map_err(|e| e.to_string())?;
        let mut it = l.split_whitespace();
        it.next(); // FID
        if let Some(iid) = it.next() {
            samples.push(iid.to_string());
        } else {
            return Err(format!("Malformed FAM line: {l}"));
        }
    }
    Ok(samples)
}

/// 读取 .bim，返回 SiteInfo 列表
fn read_bim(prefix: &str) -> Result<Vec<SiteInfo>, String> {
    let bim_path = format!("{prefix}.bim");
    let file = File::open(&bim_path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);

    let mut sites = Vec::new();
    for line in reader.lines() {
        let l = line.map_err(|e| e.to_string())?;
        let cols: Vec<&str> = l.split_whitespace().collect();
        if cols.len() < 6 {
            return Err(format!("Malformed BIM line: {l}"));
        }
        let chrom = cols[0].to_string();
        let pos: i32 = cols[3].parse().unwrap_or(0);
        let a1 = cols[4].to_string();
        let a2 = cols[5].to_string();

        sites.push(SiteInfo {
            chrom,
            pos,
            ref_allele: a1,
            alt_allele: a2,
        });
    }
    Ok(sites)
}

/// 对单个 SNP 行做：
/// - 按缺失率过滤
/// - 按 MAF 过滤
/// - MAF > 0.5 时翻转 ref/alt 且 g = 2 - g
/// - 对缺失值进行均值填充（可能有小数）
///
/// 返回：是否保留该 SNP（true=保留，false=过滤）
fn process_snp_row(
    row: &mut [f32],
    ref_allele: &mut String,
    alt_allele: &mut String,
    maf_threshold: f32,
    max_missing_rate: f32,
) -> bool {
    let mut alt_sum: f64 = 0.0;
    let mut non_missing: i64 = 0;

    for &g in row.iter() {
        if g >= 0.0 {
            alt_sum += g as f64;
            non_missing += 1;
        }
    }

    let n_samples = row.len() as f64;
    if n_samples == 0.0 {
        return false;
    }

    // 缺失率过滤
    let missing_rate = 1.0 - (non_missing as f64 / n_samples);
    if missing_rate > max_missing_rate as f64 {
        return false;
    }

    if non_missing == 0 {
        // 全缺失但缺失率未超过阈值：此时 ALT 频率和 MAF 都是 0
        // 如果 maf_threshold > 0，则过滤；否则可以填 0.0 后保留
        if maf_threshold > 0.0 {
            return false;
        } else {
            for g in row.iter_mut() {
                *g = 0.0;
            }
            return true;
        }
    }

    // ALT 频率
    let mut alt_freq = alt_sum / (2.0 * non_missing as f64);

    // ALT 频率 > 0.5：翻转基因型并交换 ref/alt
    if alt_freq > 0.5 {
        for g in row.iter_mut() {
            if *g >= 0.0 {
                *g = 2.0 - *g;
            }
        }
        std::mem::swap(ref_allele, alt_allele);
        // 翻转后 ALT 拷贝数重新计算
        alt_sum = 2.0 * non_missing as f64 - alt_sum;
        alt_freq = alt_sum / (2.0 * non_missing as f64);
    }

    let maf = alt_freq.min(1.0 - alt_freq);
    if maf < maf_threshold as f64 {
        return false;
    }

    // 均值填充（基于当前 ALT 计数，不再取整）
    let mean_g = alt_sum / non_missing as f64;
    let imputed: f32 = mean_g as f32;

    for g in row.iter_mut() {
        if *g < 0.0 {
            *g = imputed;
        }
    }

    true
}

/// =======================
///   BED 分块读取器
/// =======================

#[pyclass]
pub struct BedChunkReader {
    prefix: String,
    samples: Vec<String>,
    sites: Vec<SiteInfo>,
    mmap: Mmap,
    n_samples: usize,
    n_snps: usize,
    bytes_per_snp: usize,
    current_snp: usize,
    maf_threshold: f32,
    max_missing_rate: f32,
}

#[pymethods]
impl BedChunkReader {
    /// BedChunkReader("QC", maf_threshold=None, max_missing_rate=None)
    /// 对应 QC.bed/QC.bim/QC.fam
    #[new]
    fn new(
        prefix: String,
        maf_threshold: Option<f32>,
        max_missing_rate: Option<f32>,
    ) -> PyResult<Self> {
        let samples = read_fam(&prefix)
            .map_err(|msg| PyErr::new::<PyRuntimeError, _>(msg))?;
        let sites = read_bim(&prefix)
            .map_err(|msg| PyErr::new::<PyRuntimeError, _>(msg))?;

        let n_samples = samples.len();
        let n_snps = sites.len();

        let bed_path = format!("{prefix}.bed");
        let file = File::open(&bed_path)
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

        if mmap.len() < 3 {
            return Err(PyErr::new::<PyRuntimeError, _>(
                "BED file too small (no header)".to_string(),
            ));
        }
        if mmap[0] != 0x6C || mmap[1] != 0x1B {
            return Err(PyErr::new::<PyRuntimeError, _>(
                "Malformed BED header (expected 0x6C 0x1B)".to_string(),
            ));
        }
        if mmap[2] != 0x01 {
            return Err(PyErr::new::<PyRuntimeError, _>(
                "Only SNP-major BED is supported (header[2] must be 0x01)".to_string(),
            ));
        }

        let bytes_per_snp = (n_samples + 3) / 4;

        Ok(BedChunkReader {
            prefix,
            samples,
            sites,
            mmap,
            n_samples,
            n_snps,
            bytes_per_snp,
            current_snp: 0,
            maf_threshold: maf_threshold.unwrap_or(0.0),        // 默认不过滤 MAF
            max_missing_rate: max_missing_rate.unwrap_or(1.0),  // 默认不过滤缺失率
        })
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[getter]
    fn n_snps(&self) -> usize {
        self.n_snps
    }

    #[getter]
    fn sample_ids(&self) -> Vec<String> {
        self.samples.clone()
    }

    /// 获取下一块：
    /// 返回 (genos: numpy.ndarray[float32], sites: List[SiteInfo]) 或 None
    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk_size: usize,
    ) -> PyResult<Option<(&'py PyArray2<f32>, Vec<SiteInfo>)>> {
        if chunk_size == 0 {
            return Err(PyErr::new::<PyValueError, _>(
                "chunk_size must be > 0",
            ));
        }

        if self.current_snp >= self.n_snps {
            return Ok(None);
        }

        let mut geno_rows: Vec<Vec<f32>> = Vec::new();
        let mut out_sites: Vec<SiteInfo> = Vec::new();
        let data = &self.mmap[3..]; // 跳过 BED header

        while self.current_snp < self.n_snps && geno_rows.len() < chunk_size {
            let snp_idx = self.current_snp;
            let offset = snp_idx * self.bytes_per_snp;
            let snp_bytes = &data[offset..offset + self.bytes_per_snp];

            // 解码到临时行
            let mut row: Vec<f32> = vec![-9.0; self.n_samples];
            for (byte_idx, byte) in snp_bytes.iter().enumerate() {
                for within in 0..4 {
                    let samp_idx = byte_idx * 4 + within;
                    if samp_idx >= self.n_samples {
                        break;
                    }
                    let code = (byte >> (within * 2)) & 0b11;
                    let g = match code {
                        0b00 => 0.0_f32,
                        0b10 => 1.0_f32,
                        0b11 => 2.0_f32,
                        0b01 => -9.0_f32,
                        _ => -9.0_f32,
                    };
                    row[samp_idx] = g;
                }
            }

            let mut site = self.sites[snp_idx].clone();
            let keep = process_snp_row(
                &mut row,
                &mut site.ref_allele,
                &mut site.alt_allele,
                self.maf_threshold,
                self.max_missing_rate,
            );

            self.current_snp += 1;

            if keep {
                geno_rows.push(row);
                out_sites.push(site);
            }
        }

        if geno_rows.is_empty() {
            // 没有任何 SNP 通过过滤且已经读到文件末尾
            if self.current_snp >= self.n_snps {
                return Ok(None);
            } else {
                // 理论上不会进这个分支（while 会继续读直到 EOF 或有 SNP 通过过滤）
                return Ok(None);
            }
        }

        let rows = geno_rows.len();
        let cols = self.n_samples;
        let mut mat = Array2::<f32>::zeros((rows, cols));
        for (i, row) in geno_rows.into_iter().enumerate() {
            for (j, g) in row.into_iter().enumerate() {
                mat[[i, j]] = g;
            }
        }

        #[allow(deprecated)]
        let py_mat = mat.into_pyarray(py);
        Ok(Some((py_mat, out_sites)))
    }
}

/// =======================
///   VCF 分块读取器
/// =======================

fn open_text_maybe_gz(path: &Path) -> Result<Box<dyn BufRead + Send>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader: Box<dyn BufRead + Send> = if path
        .extension()
        .map(|e| e == "gz")
        .unwrap_or(false)
    {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    Ok(reader)
}

#[pyclass]
pub struct VcfChunkReader {
    samples: Vec<String>,
    reader: Box<dyn BufRead + Send>,
    finished: bool,
    maf_threshold: f32,
    max_missing_rate: f32,
}

#[pymethods]
impl VcfChunkReader {
    /// VcfChunkReader("QC.vcf.gz", maf_threshold=None, max_missing_rate=None)
    #[new]
    fn new(
        path: String,
        maf_threshold: Option<f32>,
        max_missing_rate: Option<f32>,
    ) -> PyResult<Self> {
        let p = Path::new(&path);
        let mut reader = open_text_maybe_gz(p)
            .map_err(|msg| PyErr::new::<PyRuntimeError, _>(msg))?;

        let mut header_line = String::new();
        let samples: Vec<String>;

        loop {
            header_line.clear();
            let n = reader
                .read_line(&mut header_line)
                .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
            if n == 0 {
                return Err(PyErr::new::<PyRuntimeError, _>(
                    "No #CHROM header found in VCF".to_string(),
                ));
            }
            if header_line.starts_with("#CHROM") {
                let parts: Vec<_> = header_line.trim_end().split('\t').collect();
                if parts.len() < 10 {
                    return Err(PyErr::new::<PyRuntimeError, _>(
                        "#CHROM header too short".to_string(),
                    ));
                }
                samples = parts[9..].iter().map(|s| s.to_string()).collect();
                break;
            }
        }

        Ok(VcfChunkReader {
            samples,
            reader,
            finished: false,
            maf_threshold: maf_threshold.unwrap_or(0.0),
            max_missing_rate: max_missing_rate.unwrap_or(1.0),
        })
    }

    #[getter]
    fn sample_ids(&self) -> Vec<String> {
        self.samples.clone()
    }

    /// 获取下一块 VCF：返回 (genos: numpy.ndarray[float32], sites) 或 None
    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk_size: usize,
    ) -> PyResult<Option<(&'py PyArray2<f32>, Vec<SiteInfo>)>> {
        if chunk_size == 0 {
            return Err(PyErr::new::<PyValueError, _>(
                "chunk_size must be > 0",
            ));
        }

        if self.finished {
            return Ok(None);
        }

        let mut geno_rows: Vec<Vec<f32>> = Vec::new();
        let mut sites: Vec<SiteInfo> = Vec::new();

        let mut line = String::new();

        while geno_rows.len() < chunk_size {
            line.clear();
            let n = self
                .reader
                .read_line(&mut line)
                .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
            if n == 0 {
                self.finished = true;
                break;
            }
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }

            let parts: Vec<_> = line.trim_end().split('\t').collect();
            if parts.len() < 9 {
                continue;
            }

            let mut site = SiteInfo {
                chrom: parts[0].to_string(),
                pos: parts[1].parse().unwrap_or(0),
                ref_allele: parts[3].to_string(),
                alt_allele: parts[4].to_string(),
            };

            let format = parts[8];
            if !format.split(':').any(|f| f == "GT") {
                continue;
            }

            let mut row: Vec<f32> = Vec::with_capacity(self.samples.len());
            for s in 9..parts.len() {
                let gt_field = parts[s];
                let gt = gt_field.split(':').next().unwrap_or(".");
                let g = match gt {
                    "0/0" | "0|0" => 0.0_f32,
                    "0/1" | "1/0" | "0|1" | "1|0" => 1.0_f32,
                    "1/1" | "1|1" => 2.0_f32,
                    "./." | ".|." => -9.0_f32,
                    _ => -9.0_f32,
                };
                row.push(g);
            }

            let keep = process_snp_row(
                &mut row,
                &mut site.ref_allele,
                &mut site.alt_allele,
                self.maf_threshold,
                self.max_missing_rate,
            );

            if keep {
                geno_rows.push(row);
                sites.push(site);
            }
        }

        if geno_rows.is_empty() {
            return Ok(None);
        }

        let rows = geno_rows.len();
        let cols = self.samples.len();
        let mut mat = Array2::<f32>::zeros((rows, cols));
        for (i, row) in geno_rows.into_iter().enumerate() {
            for (j, g) in row.into_iter().enumerate() {
                mat[[i, j]] = g;
            }
        }

        #[allow(deprecated)]
        let py_mat = mat.into_pyarray(py);
        Ok(Some((py_mat, sites)))
    }
}

/// =======================
///   写出：PLINK / VCF（流式，int8，-9 缺失）
///   geno shape: (n_samples, n_snps)
/// =======================

#[inline]
fn plink2bits_from_g_i8(g: i8) -> u8 {
    // PLINK bed 2-bit:
    // 00=A1/A1, 10=het, 11=A2/A2, 01=missing
    match g {
        0 => 0b00,
        1 => 0b10,
        2 => 0b11,
        _ => 0b01,
    }
}

fn write_bed_from_samples_major_stream_i8(
    path: &Path,
    geno: &PyReadonlyArray2<i8>, // (n_samples, n_snps)
) -> Result<(), String> {
    let arr = geno.as_array();
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err("geno must be 2D (n_samples, n_snps)".to_string());
    }
    let n_samples = shape[0];
    let n_snps = shape[1];

    // ndarray strides are in #elements
    let strides = arr.strides();
    let s0 = strides[0] as isize;
    let s1 = strides[1] as isize;
    let base = arr.as_ptr(); // *const i8

    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);

    // PLINK bed header: 6C 1B 01 (SNP-major)
    w.write_all(&[0x6C, 0x1B, 0x01]).map_err(|e| e.to_string())?;

    unsafe {
        for snp in 0..n_snps {
            let snp_off = (snp as isize) * s1;
            let mut i = 0usize;
            while i < n_samples {
                let mut byte: u8 = 0;
                for k in 0..4 {
                    let si = i + k;
                    let two = if si < n_samples {
                        let off = (si as isize) * s0 + snp_off;
                        let g = *base.offset(off);
                        plink2bits_from_g_i8(g)
                    } else {
                        0b01 // padding missing
                    };
                    byte |= two << (k * 2);
                }
                w.write_all(&[byte]).map_err(|e| e.to_string())?;
                i += 4;
            }
        }
    }

    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn write_bim_from_sites(
    path: &Path,
    sites: &[SiteInfo],
) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    for (_i, s) in sites.iter().enumerate() {
        // BIM: CHR SNPID GD BP A1 A2
        // 这里 SNPID 用 chrom:pos（你也可以改成 rsid 列表输入）
        let snp_id = format!("{}:{}", s.chrom, s.pos);
        writeln!(
            w,
            "{}\t{}\t0\t{}\t{}\t{}",
            s.chrom, snp_id, s.pos, s.ref_allele, s.alt_allele
        ).map_err(|e| e.to_string())?;
    }
    Ok(())
}

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
fn vcf_gt_from_g_i8(g: i8) -> &'static str {
    match g {
        0 => "0/0",
        1 => "0/1",
        2 => "1/1",
        _ => "./.",
    }
}

fn write_vcf_from_samples_major_stream_i8(
    path: &Path,
    geno: &PyReadonlyArray2<i8>, // (n_samples, n_snps)
    sites: &[SiteInfo],
    sample_ids: &[String],
) -> Result<(), String> {
    let arr = geno.as_array();
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err("geno must be 2D (n_samples, n_snps)".to_string());
    }
    let n_samples = shape[0];
    let n_snps = shape[1];

    if sites.len() != n_snps {
        return Err(format!("sites length mismatch: sites={}, n_snps={}", sites.len(), n_snps));
    }
    if sample_ids.len() != n_samples {
        return Err(format!("sample_ids length mismatch: sample_ids={}, n_samples={}", sample_ids.len(), n_samples));
    }

    let strides = arr.strides();
    let s0 = strides[0] as isize;
    let s1 = strides[1] as isize;
    let base = arr.as_ptr(); // *const i8

    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);

    writeln!(w, "##fileformat=VCFv4.2").map_err(|e| e.to_string())?;
    writeln!(w, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")
        .map_err(|e| e.to_string())?;

    write!(w, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
        .map_err(|e| e.to_string())?;
    for sid in sample_ids {
        write!(w, "\t{}", sid).map_err(|e| e.to_string())?;
    }
    writeln!(w).map_err(|e| e.to_string())?;

    unsafe {
        for snp in 0..n_snps {
            let s = &sites[snp];
            let snp_id = format!("{}:{}", s.chrom, s.pos);
            write!(
                w,
                "{}\t{}\t{}\t{}\t{}\t.\tPASS\t.\tGT",
                s.chrom, s.pos, snp_id, s.ref_allele, s.alt_allele
            ).map_err(|e| e.to_string())?;

            let snp_off = (snp as isize) * s1;
            for i in 0..n_samples {
                let off = (i as isize) * s0 + snp_off;
                let g = *base.offset(off);
                write!(w, "\t{}", vcf_gt_from_g_i8(g)).map_err(|e| e.to_string())?;
            }
            writeln!(w).map_err(|e| e.to_string())?;
        }
    }

    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Python: save_plink_stream(prefix, geno, sites, sample_ids, phenotype=None)
#[pyfunction]
fn save_plink_stream(
    prefix: String,
    geno: PyReadonlyArray2<i8>,   // (n_samples, n_snps), int8, -9 missing
    sites: Vec<SiteInfo>,         // len = n_snps
    sample_ids: Vec<String>,      // len = n_samples
    phenotype: Option<Vec<f64>>,  // len = n_samples
) -> PyResult<()> {
    let arr = geno.as_array();
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(PyErr::new::<PyValueError, _>("geno must be 2D (n_samples, n_snps)"));
    }
    let n_samples = shape[0];
    let n_snps = shape[1];

    if sites.len() != n_snps {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "sites length mismatch: sites={}, n_snps={}", sites.len(), n_snps
        )));
    }
    if sample_ids.len() != n_samples {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "sample_ids length mismatch: sample_ids={}, n_samples={}", sample_ids.len(), n_samples
        )));
    }
    if let Some(ref p) = phenotype {
        if p.len() != n_samples {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "phenotype length mismatch: phenotype={}, n_samples={}", p.len(), n_samples
            )));
        }
    }

    let bed_path = format!("{prefix}.bed");
    let bim_path = format!("{prefix}.bim");
    let fam_path = format!("{prefix}.fam");

    let bed = Path::new(&bed_path);
    let bim = Path::new(&bim_path);
    let fam = Path::new(&fam_path);

    write_bed_from_samples_major_stream_i8(bed, &geno)
        .map_err(|e| PyErr::new::<PyIOError, _>(e))?;
    write_bim_from_sites(bim, &sites)
        .map_err(|e| PyErr::new::<PyIOError, _>(e))?;
    let ph_ref = phenotype.as_ref().map(|v| v.as_slice());
    write_fam_simple(fam, &sample_ids, ph_ref)
        .map_err(|e| PyErr::new::<PyIOError, _>(e))?;

    Ok(())
}

/// Python: save_vcf_stream(path, geno, sites, sample_ids)
#[pyfunction]
fn save_vcf_stream(
    path: String,
    geno: PyReadonlyArray2<i8>,   // (n_samples, n_snps), int8, -9 missing
    sites: Vec<SiteInfo>,         // len = n_snps
    sample_ids: Vec<String>,      // len = n_samples
) -> PyResult<()> {
    let arr = geno.as_array();
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(PyErr::new::<PyValueError, _>("geno must be 2D (n_samples, n_snps)"));
    }

    write_vcf_from_samples_major_stream_i8(Path::new(&path), &geno, &sites, &sample_ids)
        .map_err(|e| PyErr::new::<PyIOError, _>(e))?;

    Ok(())
}

/// Count variant records (non-header lines) in a VCF/VCF.GZ file.
///
/// Header lines start with '#', variant records do not.
/// This makes one linear pass over the file and is usually cheap
/// compared to downstream GWAS computations.
#[pyfunction]
fn count_vcf_snps(path: String) -> PyResult<usize> {

    let p = Path::new(&path);
    let mut reader = open_text_maybe_gz(p)
        .map_err(|msg| PyErr::new::<PyRuntimeError, _>(msg))?;

    let mut n: usize = 0;
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        if bytes_read == 0 {
            break;
        }
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        n += 1;
    }

    Ok(n)
}

/// PyO3 模块导出
#[pymodule]
fn gfreader_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<SiteInfo>()?;
    m.add_class::<BedChunkReader>()?;
    m.add_class::<VcfChunkReader>()?;
    m.add_function(wrap_pyfunction!(count_vcf_snps, m)?)?;
    m.add_function(wrap_pyfunction!(save_plink_stream, m)?)?;
    m.add_function(wrap_pyfunction!(save_vcf_stream, m)?)?;
    Ok(())
}