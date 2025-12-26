// src/gfcore.rs
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use flate2::read::MultiGzDecoder;
use memmap2::Mmap;

// ---------------------------
// Variant metadata
// ---------------------------
#[derive(Clone, Debug)]
pub struct SiteInfo {
    pub chrom: String,
    pub pos: i32,
    pub ref_allele: String,
    pub alt_allele: String,
}

// ---------------------------
// PLINK helpers
// ---------------------------
pub fn read_fam(prefix: &str) -> Result<Vec<String>, String> {
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

pub fn read_bim(prefix: &str) -> Result<Vec<SiteInfo>, String> {
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

// ---------------------------
// VCF open helper
// ---------------------------
pub fn open_text_maybe_gz(path: &Path) -> Result<Box<dyn BufRead + Send>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    if path.extension().map(|e| e == "gz").unwrap_or(false) {
        Ok(Box::new(BufReader::new(MultiGzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

// ---------------------------
// SNP row processing
// ---------------------------
pub fn process_snp_row(
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

    let missing_rate = 1.0 - (non_missing as f64 / n_samples);
    if missing_rate > max_missing_rate as f64 {
        return false;
    }

    if non_missing == 0 {
        if maf_threshold > 0.0 {
            return false;
        } else {
            row.fill(0.0);
            return true;
        }
    }

    let mut alt_freq = alt_sum / (2.0 * non_missing as f64);

    if alt_freq > 0.5 {
        for g in row.iter_mut() {
            if *g >= 0.0 {
                *g = 2.0 - *g;
            }
        }
        std::mem::swap(ref_allele, alt_allele);
        alt_sum = 2.0 * non_missing as f64 - alt_sum;
        alt_freq = alt_sum / (2.0 * non_missing as f64);
    }

    let maf = alt_freq.min(1.0 - alt_freq);
    if maf < maf_threshold as f64 {
        return false;
    }

    let mean_g = alt_sum / non_missing as f64;
    let imputed: f32 = mean_g as f32;
    for g in row.iter_mut() {
        if *g < 0.0 {
            *g = imputed;
        }
    }

    true
}

// ======================================================================
// BED SNP iterator (single SNP each time): returns Vec<f32> (len n)
// ======================================================================
pub struct BedSnpIter {
    pub prefix: String,
    pub samples: Vec<String>,
    pub sites: Vec<SiteInfo>,
    mmap: Mmap,
    n_samples: usize,
    n_snps: usize,
    bytes_per_snp: usize,
    cur: usize,
    maf: f32,
    miss: f32,
}

impl BedSnpIter {
    pub fn new(prefix: &str, maf: f32, miss: f32) -> Result<Self, String> {
        let samples = read_fam(prefix)?;
        let sites = read_bim(prefix)?;
        let n_samples = samples.len();
        let n_snps = sites.len();

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
            prefix: prefix.to_string(),
            samples,
            sites,
            mmap,
            n_samples,
            n_snps,
            bytes_per_snp,
            cur: 0,
            maf,
            miss,
        })
    }

    pub fn n_samples(&self) -> usize { self.n_samples }

    pub fn next_snp(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        let data = &self.mmap[3..];

        while self.cur < self.n_snps {
            let snp_idx = self.cur;
            self.cur += 1;

            let offset = snp_idx * self.bytes_per_snp;
            let snp_bytes = &data[offset..offset + self.bytes_per_snp];

            let mut row: Vec<f32> = vec![-9.0; self.n_samples];

            for (byte_idx, byte) in snp_bytes.iter().enumerate() {
                for within in 0..4 {
                    let samp_idx = byte_idx * 4 + within;
                    if samp_idx >= self.n_samples { break; }
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

            let mut site = self.sites[snp_idx].clone();
            let keep = process_snp_row(&mut row, &mut site.ref_allele, &mut site.alt_allele, self.maf, self.miss);
            if keep {
                return Some((row, site));
            }
        }
        None
    }
}

// ======================================================================
// VCF SNP iterator (single SNP each time)
// ======================================================================
pub struct VcfSnpIter {
    pub samples: Vec<String>,
    reader: Box<dyn BufRead + Send>,
    maf: f32,
    miss: f32,
    finished: bool,
}

impl VcfSnpIter {
    pub fn new(path: &str, maf: f32, miss: f32) -> Result<Self, String> {
        let p = Path::new(path);
        let mut reader = open_text_maybe_gz(p)?;

        // parse header to get samples
        let mut header_line = String::new();
        let samples: Vec<String>;
        loop {
            header_line.clear();
            let n = reader.read_line(&mut header_line).map_err(|e| e.to_string())?;
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

        Ok(Self { samples, reader, maf, miss, finished: false })
    }

    pub fn n_samples(&self) -> usize { self.samples.len() }

    pub fn next_snp(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        if self.finished { return None; }

        let mut line = String::new();
        loop {
            line.clear();
            let n = self.reader.read_line(&mut line).ok()?;
            if n == 0 {
                self.finished = true;
                return None;
            }
            if line.starts_with('#') || line.trim().is_empty() { continue; }

            let parts: Vec<_> = line.trim_end().split('\t').collect();
            if parts.len() < 10 { continue; }

            let format = parts[8];
            if !format.split(':').any(|f| f == "GT") { continue; }

            let mut site = SiteInfo {
                chrom: parts[0].to_string(),
                pos: parts[1].parse().unwrap_or(0),
                ref_allele: parts[3].to_string(),
                alt_allele: parts[4].to_string(),
            };

            let mut row: Vec<f32> = Vec::with_capacity(self.samples.len());
            for s in 9..parts.len() {
                let gt = parts[s].split(':').next().unwrap_or(".");
                let g = match gt {
                    "0/0" | "0|0" => 0.0,
                    "0/1" | "1/0" | "0|1" | "1|0" => 1.0,
                    "1/1" | "1|1" => 2.0,
                    "./." | ".|." => -9.0,
                    _ => -9.0,
                };
                row.push(g);
            }

            let keep = process_snp_row(&mut row, &mut site.ref_allele, &mut site.alt_allele, self.maf, self.miss);
            if keep {
                return Some((row, site));
            }
        }
    }
}