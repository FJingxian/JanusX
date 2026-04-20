// src/gmerge.rs
use pyo3::types::PyDict;
use pyo3::{prelude::*, BoundObject};

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::gfcore::{open_text_maybe_gz, process_snp_row, read_bim, read_fam, HmpSnpIter, SiteInfo};
use crate::gfreader::build_sample_selection;
use crate::vcfout::VcfOut;

// ============================================================
// Output format
// ============================================================

#[derive(Clone, Copy, Debug)]
enum OutFmt {
    Vcf,
    Plink,
}

fn is_vcf_path(s: &str) -> bool {
    let x = s.to_ascii_lowercase();
    x.ends_with(".vcf") || x.ends_with(".vcf.gz")
}

fn is_hmp_path(s: &str) -> bool {
    let x = s.to_ascii_lowercase();
    x.ends_with(".hmp") || x.ends_with(".hmp.gz")
}

fn has_dataset_prefix(name: &str) -> bool {
    let b = name.as_bytes();
    if b.len() < 3 || b[0] != b'D' {
        return false;
    }
    let mut i = 1usize;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    i > 1 && i < b.len() && b[i] == b'_'
}

fn infer_out_fmt(out: &str, out_fmt: &str) -> Result<OutFmt, String> {
    let f = out_fmt.to_ascii_lowercase();
    if f == "auto" {
        if is_vcf_path(out) {
            Ok(OutFmt::Vcf)
        } else {
            Ok(OutFmt::Plink)
        }
    } else if f == "vcf" {
        Ok(OutFmt::Vcf)
    } else if f == "plink" || f == "bfile" || f == "bed" {
        Ok(OutFmt::Plink)
    } else {
        Err("out_fmt must be one of: auto, vcf, plink".to_string())
    }
}

// ============================================================
// Sample identity: (FID, IID)
// ============================================================

#[derive(Clone, Debug)]
struct SampleKey {
    fid: String,
    iid: String,
}

impl SampleKey {
    /// VCF sample name.
    fn vcf_name(&self) -> String {
        self.iid.clone()
    }
}

// ============================================================
// Stats (PyO3-facing)
// ============================================================

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyMergeStats {
    #[pyo3(get)]
    pub n_inputs: usize,
    #[pyo3(get)]
    pub out_fmt: String,
    #[pyo3(get)]
    pub out: String,

    #[pyo3(get)]
    pub sample_counts: Vec<usize>,
    #[pyo3(get)]
    pub n_samples_total: usize,

    #[pyo3(get)]
    pub n_sites_written: u64,
    #[pyo3(get)]
    pub n_sites_union_seen: u64,

    #[pyo3(get)]
    pub n_sites_dropped_multiallelic: u64,
    #[pyo3(get)]
    pub n_sites_dropped_non_snp: u64,
    #[pyo3(get)]
    pub n_sites_dropped_maf: u64,
    #[pyo3(get)]
    pub n_sites_dropped_geno: u64,

    #[pyo3(get)]
    pub per_input_present_sites: Vec<u64>,
    #[pyo3(get)]
    pub per_input_unaligned_sites: Vec<u64>,
    #[pyo3(get)]
    pub per_input_absent_sites: Vec<u64>,
}

#[pymethods]
impl PyMergeStats {
    pub fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py).into_bound();
        d.set_item("n_inputs", self.n_inputs)?;
        d.set_item("out_fmt", &self.out_fmt)?;
        d.set_item("out", &self.out)?;
        d.set_item("sample_counts", self.sample_counts.clone())?;
        d.set_item("n_samples_total", self.n_samples_total)?;
        d.set_item("n_sites_written", self.n_sites_written)?;
        d.set_item("n_sites_union_seen", self.n_sites_union_seen)?;
        d.set_item(
            "n_sites_dropped_multiallelic",
            self.n_sites_dropped_multiallelic,
        )?;
        d.set_item("n_sites_dropped_non_snp", self.n_sites_dropped_non_snp)?;
        d.set_item("n_sites_dropped_maf", self.n_sites_dropped_maf)?;
        d.set_item("n_sites_dropped_geno", self.n_sites_dropped_geno)?;
        d.set_item(
            "per_input_present_sites",
            self.per_input_present_sites.clone(),
        )?;
        d.set_item(
            "per_input_unaligned_sites",
            self.per_input_unaligned_sites.clone(),
        )?;
        d.set_item(
            "per_input_absent_sites",
            self.per_input_absent_sites.clone(),
        )?;
        Ok(d)
    }
}

// ============================================================
// Conversion stats (PyO3-facing)
// ============================================================

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyConvertStats {
    #[pyo3(get)]
    pub input: String,
    #[pyo3(get)]
    pub out_fmt: String,
    #[pyo3(get)]
    pub out: String,

    #[pyo3(get)]
    pub n_samples: usize,
    #[pyo3(get)]
    pub n_sites_seen: u64,
    #[pyo3(get)]
    pub n_sites_written: u64,
    #[pyo3(get)]
    pub n_sites_dropped_multiallelic: u64,
    #[pyo3(get)]
    pub n_sites_dropped_non_snp: u64,
}

#[pymethods]
impl PyConvertStats {
    pub fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py).into_bound();
        d.set_item("input", &self.input)?;
        d.set_item("out_fmt", &self.out_fmt)?;
        d.set_item("out", &self.out)?;
        d.set_item("n_samples", self.n_samples)?;
        d.set_item("n_sites_seen", self.n_sites_seen)?;
        d.set_item("n_sites_written", self.n_sites_written)?;
        d.set_item(
            "n_sites_dropped_multiallelic",
            self.n_sites_dropped_multiallelic,
        )?;
        d.set_item("n_sites_dropped_non_snp", self.n_sites_dropped_non_snp)?;
        Ok(d)
    }
}

// ============================================================
// Allele helpers (biallelic A/C/G/T only; allow swap + strand complement)
// ============================================================

#[inline]
fn is_acgt(a: &str) -> bool {
    matches!(a, "A" | "C" | "G" | "T")
}

#[inline]
fn complement_base(a: &str) -> Option<&'static str> {
    match a {
        "A" => Some("T"),
        "T" => Some("A"),
        "C" => Some("G"),
        "G" => Some("C"),
        _ => None,
    }
}

/// Return Some((REF,ALT)) if it is a clean biallelic SNP A/C/G/T.
/// Return None for multiallelic (contains comma), indel/SV (len != 1), invalid.
#[inline]
fn normalize_biallelic_snp(ref_a: &str, alt_a: &str) -> Option<(String, String)> {
    if ref_a.contains(',') || alt_a.contains(',') {
        return None;
    }
    if ref_a.len() != 1 || alt_a.len() != 1 {
        return None;
    }
    let r = ref_a.to_ascii_uppercase();
    let a = alt_a.to_ascii_uppercase();
    if !is_acgt(&r) || !is_acgt(&a) || r == a {
        return None;
    }
    Some((r, a))
}

/// Return Some((REF,ALT)) for a biallelic variant.
/// - snps_only=true : keep only SNP A/C/G/T.
/// - snps_only=false: keep non-SNP biallelic variants as well (e.g. indels),
///   while still dropping multiallelic/invalid rows.
/// - REF==ALT monomorphic rows are retained and can be filtered downstream
///   by MAF/missingness thresholds.
#[inline]
fn normalize_biallelic_variant(
    ref_a: &str,
    alt_a: &str,
    snps_only: bool,
) -> Option<(String, String)> {
    if ref_a.contains(',') || alt_a.contains(',') {
        return None;
    }
    let r = ref_a.trim().to_ascii_uppercase();
    let a = alt_a.trim().to_ascii_uppercase();
    if r.is_empty() || a.is_empty() || r == "." || a == "." {
        return None;
    }
    if snps_only {
        if r.len() != 1 || a.len() != 1 {
            return None;
        }
        if !is_acgt(&r) || !is_acgt(&a) {
            return None;
        }
    }
    Some((r, a))
}

#[derive(Clone, Copy, Debug)]
enum AlleleMap {
    Identity,
    Swap,
    StrandIdentity,
    StrandSwap,
}

fn infer_map(dref: &str, dalt: &str, gref: &str, galt: &str) -> Option<AlleleMap> {
    if dref == gref && dalt == galt {
        return Some(AlleleMap::Identity);
    }
    if dref == galt && dalt == gref {
        return Some(AlleleMap::Swap);
    }

    let cr = complement_base(dref)?;
    let ca = complement_base(dalt)?;
    if cr == gref && ca == galt {
        return Some(AlleleMap::StrandIdentity);
    }
    if cr == galt && ca == gref {
        return Some(AlleleMap::StrandSwap);
    }
    None
}

#[inline]
fn apply_map_dosage(g: f32, amap: AlleleMap) -> f32 {
    if g < 0.0 {
        return g;
    }
    match amap {
        AlleleMap::Identity | AlleleMap::StrandIdentity => g,
        AlleleMap::Swap | AlleleMap::StrandSwap => 2.0 - g,
    }
}

#[inline]
fn gt_to_dosage(gt: &str) -> f32 {
    match gt {
        "0/0" | "0|0" => 0.0,
        "0/1" | "1/0" | "0|1" | "1|0" => 1.0,
        "1/1" | "1|1" => 2.0,
        "./." | ".|." => -9.0,
        _ => -9.0,
    }
}

#[inline]
fn dosage_to_i8_rounded(g: f32) -> i8 {
    if g < 0.0 {
        return -9;
    }
    let mut v = g.round() as i32;
    if v < 0 {
        v = 0;
    } else if v > 2 {
        v = 2;
    }
    match v {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => -9,
    }
}

#[inline]
fn i8_to_vcf_gt(g: i8) -> &'static str {
    match g {
        0 => "0/0",
        1 => "0/1",
        2 => "1/1",
        _ => "./.",
    }
}

// ============================================================
// Chrom/pos ordering for k-way merge
// ============================================================

fn chrom_ord_key(chrom: &str) -> (u8, i32, String) {
    let c = chrom.trim();
    let lower = c.trim_start_matches("chr").trim_start_matches("Chr");
    if let Ok(v) = lower.parse::<i32>() {
        (0, v, String::new())
    } else {
        (1, i32::MAX, c.to_string())
    }
}

fn cmp_site(a: &SiteInfo, b: &SiteInfo) -> Ordering {
    let ka = chrom_ord_key(&a.chrom);
    let kb = chrom_ord_key(&b.chrom);
    match ka.cmp(&kb) {
        Ordering::Equal => a.pos.cmp(&b.pos),
        other => other,
    }
}

// ============================================================
// Raw iterators (NO process_snp_row flipping/imputation)
// ============================================================

struct BedSnpIterRaw {
    samples: Vec<String>,
    sites: Vec<SiteInfo>,
    mmap: Mmap,
    n_samples: usize,
    bytes_per_snp: usize,
    cur: usize,
}

impl BedSnpIterRaw {
    fn new(prefix: &str) -> Result<Self, String> {
        let samples = read_fam(prefix)?;
        let sites = read_bim(prefix)?;
        let n_samples = samples.len();

        let bed_path = format!("{prefix}.bed");
        let file = File::open(&bed_path).map_err(|e| e.to_string())?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;

        if mmap.len() < 3 {
            return Err("BED too small".into());
        }
        if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
            return Err("Only SNP-major BED supported".into());
        }

        let bytes_per_snp = (n_samples + 3) / 4;
        Ok(Self {
            samples,
            sites,
            mmap,
            n_samples,
            bytes_per_snp,
            cur: 0,
        })
    }

    fn sample_ids(&self) -> &[String] {
        &self.samples
    }
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn decode_row(&self, snp_idx: usize) -> Vec<f32> {
        let data = &self.mmap[3..];
        let offset = snp_idx * self.bytes_per_snp;
        let snp_bytes = &data[offset..offset + self.bytes_per_snp];

        let mut row: Vec<f32> = vec![-9.0; self.n_samples];
        for (byte_idx, byte) in snp_bytes.iter().enumerate() {
            for within in 0..4 {
                let samp_idx = byte_idx * 4 + within;
                if samp_idx >= self.n_samples {
                    break;
                }
                let code = (byte >> (within * 2)) & 0b11;
                row[samp_idx] = match code {
                    0b00 => 0.0,
                    0b10 => 1.0,
                    0b11 => 2.0,
                    0b01 => -9.0,
                    _ => -9.0,
                };
            }
        }
        row
    }

    fn decode_row_selected(&self, snp_idx: usize, sample_indices: &[usize]) -> Vec<f32> {
        if sample_indices.len() == self.n_samples {
            return self.decode_row(snp_idx);
        }
        let data = &self.mmap[3..];
        let offset = snp_idx * self.bytes_per_snp;
        let snp_bytes = &data[offset..offset + self.bytes_per_snp];
        let mut row: Vec<f32> = vec![-9.0; sample_indices.len()];
        for (out_i, &samp_idx) in sample_indices.iter().enumerate() {
            if samp_idx >= self.n_samples {
                continue;
            }
            let byte = snp_bytes[samp_idx >> 2];
            let code = (byte >> ((samp_idx & 3) * 2)) & 0b11;
            row[out_i] = match code {
                0b00 => 0.0,
                0b10 => 1.0,
                0b11 => 2.0,
                0b01 => -9.0,
                _ => -9.0,
            };
        }
        row
    }

    fn next_snp(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        if self.cur >= self.sites.len() {
            return None;
        }
        let idx = self.cur;
        self.cur += 1;
        let site = self.sites[idx].clone();
        let row = self.decode_row(idx);
        Some((row, site))
    }
}

struct VcfSnpIterRaw {
    samples: Vec<String>,
    reader: Box<dyn BufRead + Send>,
    finished: bool,
}

impl VcfSnpIterRaw {
    fn new(path: &str) -> Result<Self, String> {
        let p = Path::new(path);
        let mut reader = open_text_maybe_gz(p)?;

        // parse header
        let mut header_line = String::new();
        let samples: Vec<String>;
        loop {
            header_line.clear();
            let n = reader
                .read_line(&mut header_line)
                .map_err(|e| e.to_string())?;
            if n == 0 {
                return Err("No #CHROM header found in VCF".into());
            }
            if header_line.starts_with("#CHROM") {
                let parts: Vec<_> = header_line.trim_end().split('\t').collect();
                if parts.len() < 10 {
                    return Err("#CHROM header too short".into());
                }
                samples = parts[9..].iter().map(|s| s.to_string()).collect();
                break;
            }
        }

        Ok(Self {
            samples,
            reader,
            finished: false,
        })
    }

    fn sample_ids(&self) -> &[String] {
        &self.samples
    }
    fn n_samples(&self) -> usize {
        self.samples.len()
    }

    fn next_snp(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        if self.finished {
            return None;
        }

        let mut line = String::new();
        loop {
            line.clear();
            let n = self.reader.read_line(&mut line).ok()?;
            if n == 0 {
                self.finished = true;
                return None;
            }
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }

            let parts: Vec<_> = line.trim_end().split('\t').collect();
            if parts.len() < 10 {
                continue;
            }

            // require GT
            let format = parts[8];
            if !format.split(':').any(|f| f == "GT") {
                continue;
            }

            let site = SiteInfo {
                chrom: parts[0].to_string(),
                pos: parts[1].parse().unwrap_or(0),
                ref_allele: parts[3].to_string(),
                alt_allele: parts[4].to_string(),
            };

            let mut row: Vec<f32> = Vec::with_capacity(self.samples.len());
            for s in 9..parts.len() {
                let gt = parts[s].split(':').next().unwrap_or(".");
                row.push(gt_to_dosage(gt));
            }
            return Some((row, site));
        }
    }
}

struct HmpSnpIterRaw {
    samples: Vec<String>,
    inner: HmpSnpIter,
}

impl HmpSnpIterRaw {
    fn new(path: &str) -> Result<Self, String> {
        let inner = HmpSnpIter::new_with_fill(path, 0.0, 1.0, false, false, 0.02)?;
        let samples = inner.samples.clone();
        Ok(Self { samples, inner })
    }

    fn sample_ids(&self) -> &[String] {
        &self.samples
    }

    fn n_samples(&self) -> usize {
        self.samples.len()
    }

    fn next_snp(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        self.inner.next_snp_raw()
    }
}

enum InputIter {
    Bed(BedSnpIterRaw),
    Vcf(VcfSnpIterRaw),
    Hmp(HmpSnpIterRaw),
}

impl InputIter {
    fn new(path_or_prefix: &str) -> Result<Self, String> {
        if is_vcf_path(path_or_prefix) {
            Ok(InputIter::Vcf(VcfSnpIterRaw::new(path_or_prefix)?))
        } else if is_hmp_path(path_or_prefix) {
            Ok(InputIter::Hmp(HmpSnpIterRaw::new(path_or_prefix)?))
        } else {
            // Treat input as a PLINK prefix, not a filename stem extension.
            // This preserves prefixes containing dots, e.g. "geno/DH.10k".
            let bed = format!("{path_or_prefix}.bed");
            let bim = format!("{path_or_prefix}.bim");
            let fam = format!("{path_or_prefix}.fam");
            if !(Path::new(&bed).exists() && Path::new(&bim).exists() && Path::new(&fam).exists()) {
                return Err(format!(
                    "PLINK prefix not found or missing .bed/.bim/.fam: {path_or_prefix}"
                ));
            }
            Ok(InputIter::Bed(BedSnpIterRaw::new(path_or_prefix)?))
        }
    }

    fn sample_ids(&self) -> &[String] {
        match self {
            InputIter::Bed(it) => it.sample_ids(),
            InputIter::Vcf(it) => it.sample_ids(),
            InputIter::Hmp(it) => it.sample_ids(),
        }
    }

    fn n_samples(&self) -> usize {
        match self {
            InputIter::Bed(it) => it.n_samples(),
            InputIter::Vcf(it) => it.n_samples(),
            InputIter::Hmp(it) => it.n_samples(),
        }
    }

    fn next_snp(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        match self {
            InputIter::Bed(it) => it.next_snp(),
            InputIter::Vcf(it) => it.next_snp(),
            InputIter::Hmp(it) => it.next_snp(),
        }
    }
}

// ============================================================
// Output writers
// ============================================================

struct VcfWriter {
    out: VcfOut,
    n_samples_total: usize,
}

impl VcfWriter {
    fn new(out: &str, sample_ids: &[String]) -> Result<Self, String> {
        let mut out = VcfOut::from_path(out).map_err(|e| e.to_string())?;

        out.write_all(b"##fileformat=VCFv4.2\n")
            .map_err(|e| e.to_string())?;
        out.write_all(b"##source=JanusX-merge\n")
            .map_err(|e| e.to_string())?;
        out.write_all(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
            .map_err(|e| e.to_string())?;

        out.write_all(b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
            .map_err(|e| e.to_string())?;
        for s in sample_ids {
            out.write_all(b"\t").map_err(|e| e.to_string())?;
            out.write_all(s.as_bytes()).map_err(|e| e.to_string())?;
        }
        out.write_all(b"\n").map_err(|e| e.to_string())?;

        Ok(Self {
            out,
            n_samples_total: sample_ids.len(),
        })
    }

    fn write_site(
        &mut self,
        site: &SiteInfo,
        gref: &str,
        galt: &str,
        row_i8: &[i8],
    ) -> Result<(), String> {
        let vid = format!("{}_{}", site.chrom, site.pos); // keep consistent with BIM style
        let prefix = format!(
            "{}\t{}\t{}\t{}\t{}\t.\tPASS\t.\tGT",
            site.chrom, site.pos, vid, gref, galt
        );
        self.out
            .write_all(prefix.as_bytes())
            .map_err(|e| e.to_string())?;

        if row_i8.len() != self.n_samples_total {
            return Err("Internal error: VCF row length mismatch".into());
        }
        for &g in row_i8 {
            self.out.write_all(b"\t").map_err(|e| e.to_string())?;
            self.out
                .write_all(i8_to_vcf_gt(g).as_bytes())
                .map_err(|e| e.to_string())?;
        }
        self.out.write_all(b"\n").map_err(|e| e.to_string())?;
        Ok(())
    }

    fn finish(self) -> Result<(), String> {
        self.out.finish().map_err(|e| e.to_string())?;
        Ok(())
    }
}

enum RowOutcome {
    Keep {
        site: SiteInfo,
        gref: String,
        galt: String,
        row_i8: Vec<i8>,
    },
    DropMulti,
    DropNonSnp,
    DropFiltered,
}

struct PlinkBfileWriter {
    fam: BufWriter<File>,
    bim: BufWriter<File>,
    bed: BufWriter<File>,
    n_samples: usize,
    bytes_per_snp: usize,
}

impl PlinkBfileWriter {
    fn new(prefix: &str, samples: &[SampleKey]) -> Result<Self, String> {
        let fam = BufWriter::new(File::create(format!("{prefix}.fam")).map_err(|e| e.to_string())?);
        let bim = BufWriter::new(File::create(format!("{prefix}.bim")).map_err(|e| e.to_string())?);
        let mut bed =
            BufWriter::new(File::create(format!("{prefix}.bed")).map_err(|e| e.to_string())?);

        // SNP-major BED header
        bed.write_all(&[0x6C, 0x1B, 0x01])
            .map_err(|e| e.to_string())?;

        // FAM: FID IID PID MID SEX PHENO
        {
            let mut famw = fam;
            for s in samples {
                writeln!(famw, "{}\t{}\t0\t0\t1\t-9", s.fid, s.iid).map_err(|e| e.to_string())?;
            }
            famw.flush().map_err(|e| e.to_string())?;
            let fam = famw;

            let n_samples = samples.len();
            let bytes_per_snp = (n_samples + 3) / 4;

            Ok(Self {
                fam,
                bim,
                bed,
                n_samples,
                bytes_per_snp,
            })
        }
    }

    #[inline]
    fn encode_row_i8(&self, row: &[i8]) -> Vec<u8> {
        // PLINK 2-bit:
        // 00 -> homo A1 (0)
        // 10 -> het    (1)
        // 11 -> homo A2 (2)
        // 01 -> missing
        let mut out = vec![0u8; self.bytes_per_snp];
        for byte_idx in 0..self.bytes_per_snp {
            let mut b: u8 = 0;
            for within in 0..4 {
                let i = byte_idx * 4 + within;
                let code: u8 = if i >= self.n_samples {
                    0b01
                } else {
                    match row[i] {
                        0 => 0b00,
                        1 => 0b10,
                        2 => 0b11,
                        _ => 0b01,
                    }
                };
                b |= code << (within * 2);
            }
            out[byte_idx] = b;
        }
        out
    }

    fn write_site_and_row(
        &mut self,
        site: &SiteInfo,
        gref: &str,
        galt: &str,
        row_i8: &[i8],
    ) -> Result<(), String> {
        if row_i8.len() != self.n_samples {
            return Err("Internal error: PLINK row length mismatch".into());
        }

        // BIM: CHR SNP CM BP A1 A2
        // SNP id: chr_pos (as requested)
        let snp_id = format!("{}_{}", site.chrom, site.pos);
        writeln!(
            self.bim,
            "{}\t{}\t0\t{}\t{}\t{}",
            site.chrom, snp_id, site.pos, gref, galt
        )
        .map_err(|e| e.to_string())?;

        let bytes = self.encode_row_i8(row_i8);
        self.bed.write_all(&bytes).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn finish(mut self) -> Result<(), String> {
        self.fam.flush().map_err(|e| e.to_string())?;
        self.bim.flush().map_err(|e| e.to_string())?;
        self.bed.flush().map_err(|e| e.to_string())?;
        Ok(())
    }
}

// ============================================================
// Main merge API (PyO3)
// ============================================================

#[pyfunction(signature = (inputs, out, out_fmt=None, sample_prefix=false, maf=0.0, geno=1.0))]
pub fn merge_genotypes(
    inputs: Vec<String>,
    out: String,
    out_fmt: Option<String>,
    sample_prefix: bool,
    maf: f64,
    geno: f64,
) -> PyResult<PyMergeStats> {
    if inputs.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inputs must contain at least 2 items",
        ));
    }

    let fmt = infer_out_fmt(&out, out_fmt.as_deref().unwrap_or("auto"))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    if !(0.0..=0.5).contains(&maf) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "maf must be within [0, 0.5]",
        ));
    }
    if !(0.0..=1.0).contains(&geno) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "geno must be within [0, 1]",
        ));
    }

    // open inputs
    let mut iters: Vec<InputIter> = Vec::with_capacity(inputs.len());
    for p in inputs.iter() {
        let it = InputIter::new(p).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        iters.push(it);
    }

    // merged samples
    let mut merged_samples: Vec<SampleKey> = Vec::new();
    let mut seen_output_names: HashSet<String> = HashSet::new();
    let mut sample_counts: Vec<usize> = Vec::with_capacity(iters.len());

    for (i, it) in iters.iter().enumerate() {
        let fid = "0".to_string();
        sample_counts.push(it.n_samples());
        for sid in it.sample_ids().iter() {
            let base_name = if sample_prefix {
                if has_dataset_prefix(sid) {
                    sid.clone()
                } else {
                    format!("D{}_{}", i + 1, sid)
                }
            } else {
                sid.clone()
            };
            let mut out_name = base_name.clone();
            if !seen_output_names.insert(out_name.clone()) {
                let mut k: usize = 2;
                loop {
                    let cand = format!("{}__dup{}", base_name, k);
                    if seen_output_names.insert(cand.clone()) {
                        out_name = cand;
                        break;
                    }
                    k += 1;
                }
            }
            merged_samples.push(SampleKey {
                fid: fid.clone(),
                iid: out_name,
            });
        }
    }

    let merged_sample_names_vcf: Vec<String> =
        merged_samples.iter().map(|s| s.vcf_name()).collect();

    let mut stats = PyMergeStats {
        n_inputs: iters.len(),
        out_fmt: match fmt {
            OutFmt::Vcf => "vcf".into(),
            OutFmt::Plink => "plink".into(),
        },
        out: out.clone(),
        sample_counts: sample_counts.clone(),
        n_samples_total: merged_samples.len(),
        n_sites_written: 0,
        n_sites_union_seen: 0,
        n_sites_dropped_multiallelic: 0,
        n_sites_dropped_non_snp: 0,
        n_sites_dropped_maf: 0,
        n_sites_dropped_geno: 0,
        per_input_present_sites: vec![0; iters.len()],
        per_input_unaligned_sites: vec![0; iters.len()],
        per_input_absent_sites: vec![0; iters.len()],
    };

    // Keep output unique by genomic coordinate during merge.
    // Only mark a site as seen after a successful write, so a later valid
    // record at the same coordinate can still be written if earlier one was dropped.
    let mut written_site_keys: HashSet<(String, i32)> = HashSet::new();

    // writers
    let mut vcf_w: Option<VcfWriter> = None;
    let mut plink_w: Option<PlinkBfileWriter> = None;
    match fmt {
        OutFmt::Vcf => {
            vcf_w = Some(
                VcfWriter::new(&out, &merged_sample_names_vcf)
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?,
            );
        }
        OutFmt::Plink => {
            plink_w = Some(
                PlinkBfileWriter::new(&out, &merged_samples)
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?,
            );
        }
    }

    // pull first record from each input
    let mut cur: Vec<Option<(Vec<f32>, SiteInfo)>> =
        iters.iter_mut().map(|it| it.next_snp()).collect();

    // k-way merge loop
    loop {
        // find min site
        let mut min_site: Option<SiteInfo> = None;
        for item in cur.iter() {
            if let Some((_row, site)) = item {
                min_site = match min_site {
                    None => Some(site.clone()),
                    Some(ref s0) => {
                        if cmp_site(site, s0) == Ordering::Less {
                            Some(site.clone())
                        } else {
                            Some(s0.clone())
                        }
                    }
                };
            }
        }
        let min_site = match min_site {
            Some(s) => s,
            None => break,
        };
        stats.n_sites_union_seen += 1;

        // inputs present at this position
        let mut idxs: Vec<usize> = Vec::new();
        for (i, item) in cur.iter().enumerate() {
            if let Some((_row, site)) = item {
                if site.chrom == min_site.chrom && site.pos == min_site.pos {
                    idxs.push(i);
                }
            }
        }

        // choose candidate allele from first biallelic SNP
        let mut cand_ref: Option<String> = None;
        let mut cand_alt: Option<String> = None;
        for &i in idxs.iter() {
            if let Some((_row, site)) = &cur[i] {
                match normalize_biallelic_snp(&site.ref_allele, &site.alt_allele) {
                    Some((r, a)) => {
                        cand_ref = Some(r);
                        cand_alt = Some(a);
                        break;
                    }
                    None => {
                        if site.ref_allele.contains(',') || site.alt_allele.contains(',') {
                            stats.n_sites_dropped_multiallelic += 1;
                        } else {
                            stats.n_sites_dropped_non_snp += 1;
                        }
                    }
                }
            }
        }
        let (cand_ref, cand_alt) = match (cand_ref, cand_alt) {
            (Some(r), Some(a)) => (r, a),
            _ => {
                // advance all idxs
                for &i in idxs.iter() {
                    cur[i] = iters[i].next_snp();
                }
                continue;
            }
        };

        // infer mapping per input (for those present)
        let mut maps: Vec<Option<AlleleMap>> = vec![None; iters.len()];
        for &i in idxs.iter() {
            if let Some((_row, site)) = &cur[i] {
                if let Some((r, a)) = normalize_biallelic_snp(&site.ref_allele, &site.alt_allele) {
                    maps[i] = infer_map(&r, &a, &cand_ref, &cand_alt);
                }
            }
        }

        // compute global ALT freq under candidate definition
        let mut alt_sum: f64 = 0.0;
        let mut non_missing: f64 = 0.0;
        for &i in idxs.iter() {
            if let (Some((row, _)), Some(amap)) = (&cur[i], maps[i]) {
                for &g0 in row.iter() {
                    let g = apply_map_dosage(g0, amap);
                    if g >= 0.0 {
                        alt_sum += g as f64;
                        non_missing += 1.0;
                    }
                }
            }
        }

        // global MAF reorder: if ALT freq > 0.5 then swap global ref/alt
        let mut gref = cand_ref.clone();
        let mut galt = cand_alt.clone();
        let mut global_swap = false;
        if non_missing > 0.0 {
            let alt_freq = alt_sum / (2.0 * non_missing);
            if alt_freq > 0.5 {
                std::mem::swap(&mut gref, &mut galt);
                global_swap = true;
            }
        }

        // build merged row in i8 (0/1/2/-9) for all samples (dataset concatenation)
        let mut merged_row: Vec<i8> = Vec::with_capacity(stats.n_samples_total);

        for ds_i in 0..iters.len() {
            let ns = iters[ds_i].n_samples();

            if idxs.contains(&ds_i) {
                stats.per_input_present_sites[ds_i] += 1;

                if let (Some((row, _site)), Some(amap)) = (&cur[ds_i], maps[ds_i]) {
                    for &g0 in row.iter() {
                        let mut g = apply_map_dosage(g0, amap);
                        if global_swap && g >= 0.0 {
                            g = 2.0 - g;
                        }
                        let gi8 = if g < 0.0 {
                            -9
                        } else if (g - 0.0).abs() < 1e-6 {
                            0
                        } else if (g - 1.0).abs() < 1e-6 {
                            1
                        } else if (g - 2.0).abs() < 1e-6 {
                            2
                        } else {
                            -9
                        };
                        merged_row.push(gi8);
                    }
                } else {
                    stats.per_input_unaligned_sites[ds_i] += 1;
                    for _ in 0..ns {
                        merged_row.push(-9);
                    }
                }
            } else {
                stats.per_input_absent_sites[ds_i] += 1;
                for _ in 0..ns {
                    merged_row.push(-9);
                }
            }
        }

        // QC on merged row (missingness and MAF)
        let mut called: usize = 0;
        let mut alt_dosage_sum: usize = 0;
        for &g in merged_row.iter() {
            match g {
                0 => {
                    called += 1;
                }
                1 => {
                    called += 1;
                    alt_dosage_sum += 1;
                }
                2 => {
                    called += 1;
                    alt_dosage_sum += 2;
                }
                _ => {}
            }
        }
        let missing_rate = if merged_row.is_empty() {
            1.0
        } else {
            1.0 - (called as f64 / merged_row.len() as f64)
        };
        if missing_rate > geno {
            stats.n_sites_dropped_geno += 1;
            for &i in idxs.iter() {
                cur[i] = iters[i].next_snp();
            }
            continue;
        }
        if maf > 0.0 {
            let maf_obs = if called == 0 {
                0.0
            } else {
                let af = alt_dosage_sum as f64 / (2.0 * called as f64);
                if af <= 0.5 {
                    af
                } else {
                    1.0 - af
                }
            };
            if maf_obs < maf {
                stats.n_sites_dropped_maf += 1;
                for &i in idxs.iter() {
                    cur[i] = iters[i].next_snp();
                }
                continue;
            }
        }

        // write
        let site_key = (min_site.chrom.clone(), min_site.pos);
        if written_site_keys.contains(&site_key) {
            // duplicated coordinate, skip writing this record
            for &i in idxs.iter() {
                cur[i] = iters[i].next_snp();
            }
            continue;
        }
        match fmt {
            OutFmt::Vcf => {
                vcf_w
                    .as_mut()
                    .unwrap()
                    .write_site(&min_site, &gref, &galt, &merged_row)
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            }
            OutFmt::Plink => {
                plink_w
                    .as_mut()
                    .unwrap()
                    .write_site_and_row(&min_site, &gref, &galt, &merged_row)
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            }
        }
        written_site_keys.insert(site_key);

        stats.n_sites_written += 1;

        // advance participating inputs
        for &i in idxs.iter() {
            cur[i] = iters[i].next_snp();
        }
    }

    // finalize writers
    match fmt {
        OutFmt::Vcf => {
            vcf_w
                .take()
                .unwrap()
                .finish()
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        }
        OutFmt::Plink => {
            plink_w
                .take()
                .unwrap()
                .finish()
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        }
    }

    Ok(stats)
}

// ============================================================
// Single-input conversion (PyO3)
// ============================================================

#[pyfunction(signature = (
    input,
    out,
    out_fmt=None,
    progress_callback=None,
    progress_every=10000,
    threads=0,
    snps_only=true,
    maf=0.0,
    geno=1.0,
    impute=false,
    model="add",
    het=0.02,
    sample_ids=None,
    sample_indices=None
))]
pub fn convert_genotypes(
    py: Python<'_>,
    input: String,
    out: String,
    out_fmt: Option<String>,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    threads: usize,
    snps_only: bool,
    maf: f64,
    geno: f64,
    impute: bool,
    model: &str,
    het: f64,
    sample_ids: Option<Vec<String>>,
    sample_indices: Option<Vec<usize>>,
) -> PyResult<PyConvertStats> {
    if !(0.0..=0.5).contains(&maf) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "maf must be within [0, 0.5]",
        ));
    }
    if !(0.0..=1.0).contains(&geno) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "geno must be within [0, 1]",
        ));
    }
    if !(0.0..=0.5).contains(&het) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "het must be within [0, 0.5]",
        ));
    }
    let model_key = model.trim().to_ascii_lowercase();
    if !matches!(model_key.as_str(), "add" | "dom" | "rec" | "het") {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "model must be one of: add, dom, rec, het",
        ));
    }
    let apply_het_filter = model_key != "add";
    let maf_f32 = maf as f32;
    let geno_f32 = geno as f32;
    let het_f32 = het as f32;

    let fmt = infer_out_fmt(&out, out_fmt.as_deref().unwrap_or("auto"))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let mut it = InputIter::new(&input).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let (selected_indices, selected_sample_ids) = build_sample_selection(
        it.sample_ids(),
        sample_ids,
        sample_indices,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let n_samples = selected_indices.len();
    let full_sample_selection = n_samples == it.n_samples();

    let mut stats = PyConvertStats {
        input: input.clone(),
        out_fmt: match fmt {
            OutFmt::Vcf => "vcf".into(),
            OutFmt::Plink => "plink".into(),
        },
        out: out.clone(),
        n_samples,
        n_sites_seen: 0,
        n_sites_written: 0,
        n_sites_dropped_multiallelic: 0,
        n_sites_dropped_non_snp: 0,
    };

    let mut vcf_w: Option<VcfWriter> = None;
    let mut plink_w: Option<PlinkBfileWriter> = None;
    match fmt {
        OutFmt::Vcf => {
            vcf_w = Some(
                VcfWriter::new(&out, &selected_sample_ids)
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?,
            );
        }
        OutFmt::Plink => {
            let samples: Vec<SampleKey> = selected_sample_ids
                .iter()
                .map(|sid| SampleKey {
                    fid: sid.clone(),
                    iid: sid.clone(),
                })
                .collect();
            plink_w = Some(
                PlinkBfileWriter::new(&out, &samples)
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?,
            );
        }
    }

    let report_progress = progress_callback.is_some() && progress_every > 0;
    let total_sites = if report_progress {
        match &it {
            InputIter::Bed(bed) => bed.sites.len() as u64,
            _ => 0,
        }
    } else {
        0
    };
    let mut last_report: u64 = 0;

    let use_parallel = threads > 1 && matches!(&it, InputIter::Bed(_));
    if use_parallel {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let InputIter::Bed(bed) = it else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "internal error: expected BED iterator",
            ));
        };
        let n_sites = bed.sites.len();
        let sites = &bed.sites;
        let selected_indices_ref = &selected_indices;
        let target_bytes = 16usize * 1024 * 1024;
        let mut chunk_size = target_bytes / n_samples.max(1);
        if chunk_size == 0 {
            chunk_size = 1;
        }
        if chunk_size > 4096 {
            chunk_size = 4096;
        }

        for start in (0..n_sites).step_by(chunk_size) {
            let end = (start + chunk_size).min(n_sites);
            let rows: Vec<RowOutcome> = py.detach(|| {
                pool.install(|| {
                    (start..end)
                        .into_par_iter()
                        .map(|idx| {
                            let site = sites[idx].clone();
                            let (mut gref, mut galt) = match normalize_biallelic_variant(
                                &site.ref_allele,
                                &site.alt_allele,
                                snps_only,
                            ) {
                                Some(x) => x,
                                None => {
                                    if site.ref_allele.contains(',')
                                        || site.alt_allele.contains(',')
                                    {
                                        return RowOutcome::DropMulti;
                                    }
                                    return RowOutcome::DropNonSnp;
                                }
                            };

                            let mut row_f32 = if full_sample_selection {
                                bed.decode_row(idx)
                            } else {
                                bed.decode_row_selected(idx, selected_indices_ref)
                            };
                            let keep = process_snp_row(
                                &mut row_f32,
                                &mut gref,
                                &mut galt,
                                maf_f32,
                                geno_f32,
                                impute,
                                apply_het_filter,
                                het_f32,
                            );
                            if !keep {
                                return RowOutcome::DropFiltered;
                            }

                            let mut row_i8: Vec<i8> = Vec::with_capacity(row_f32.len());
                            row_i8.extend(row_f32.iter().map(|&g| dosage_to_i8_rounded(g)));
                            RowOutcome::Keep {
                                site,
                                gref,
                                galt,
                                row_i8,
                            }
                        })
                        .collect()
                })
            });

            for row in rows {
                stats.n_sites_seen += 1;
                if report_progress && stats.n_sites_seen - last_report >= progress_every as u64 {
                    if let Some(cb) = progress_callback.as_ref() {
                        cb.call1(py, (stats.n_sites_seen, total_sites))?;
                    }
                    last_report = stats.n_sites_seen;
                }

                match row {
                    RowOutcome::Keep {
                        site,
                        gref,
                        galt,
                        row_i8,
                    } => {
                        match fmt {
                            OutFmt::Vcf => {
                                vcf_w
                                    .as_mut()
                                    .unwrap()
                                    .write_site(&site, &gref, &galt, &row_i8)
                                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
                            }
                            OutFmt::Plink => {
                                plink_w
                                    .as_mut()
                                    .unwrap()
                                    .write_site_and_row(&site, &gref, &galt, &row_i8)
                                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
                            }
                        }
                        stats.n_sites_written += 1;
                    }
                    RowOutcome::DropMulti => {
                        stats.n_sites_dropped_multiallelic += 1;
                    }
                    RowOutcome::DropNonSnp => {
                        stats.n_sites_dropped_non_snp += 1;
                    }
                    RowOutcome::DropFiltered => {}
                }
            }
        }
    } else {
        let mut row_i8: Vec<i8> = Vec::with_capacity(n_samples.max(1));
        while let Some((row, site)) = it.next_snp() {
            stats.n_sites_seen += 1;
            if report_progress && stats.n_sites_seen - last_report >= progress_every as u64 {
                if let Some(cb) = progress_callback.as_ref() {
                    cb.call1(py, (stats.n_sites_seen, total_sites))?;
                }
                last_report = stats.n_sites_seen;
            }

            let (mut gref, mut galt) =
                match normalize_biallelic_variant(&site.ref_allele, &site.alt_allele, snps_only) {
                    Some(x) => x,
                    None => {
                        if site.ref_allele.contains(',') || site.alt_allele.contains(',') {
                            stats.n_sites_dropped_multiallelic += 1;
                        } else {
                            stats.n_sites_dropped_non_snp += 1;
                        }
                        continue;
                    }
                };

            let mut row_sel = if full_sample_selection {
                row
            } else {
                let mut v = Vec::with_capacity(selected_indices.len());
                for &idx in selected_indices.iter() {
                    if idx >= row.len() {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "sample index out of range while converting row",
                        ));
                    }
                    v.push(row[idx]);
                }
                v
            };

            let keep = process_snp_row(
                &mut row_sel,
                &mut gref,
                &mut galt,
                maf_f32,
                geno_f32,
                impute,
                apply_het_filter,
                het_f32,
            );
            if !keep {
                continue;
            }

            row_i8.clear();
            row_i8.extend(row_sel.iter().map(|&g| dosage_to_i8_rounded(g)));

            match fmt {
                OutFmt::Vcf => {
                    vcf_w
                        .as_mut()
                        .unwrap()
                        .write_site(&site, &gref, &galt, &row_i8)
                        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
                }
                OutFmt::Plink => {
                    plink_w
                        .as_mut()
                        .unwrap()
                        .write_site_and_row(&site, &gref, &galt, &row_i8)
                        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
                }
            }
            stats.n_sites_written += 1;
        }
    }

    if report_progress {
        if let Some(cb) = progress_callback.as_ref() {
            cb.call1(py, (stats.n_sites_seen, total_sites))?;
        }
    }

    match fmt {
        OutFmt::Vcf => {
            vcf_w
                .take()
                .unwrap()
                .finish()
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        }
        OutFmt::Plink => {
            plink_w
                .take()
                .unwrap()
                .finish()
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        }
    }

    Ok(stats)
}
