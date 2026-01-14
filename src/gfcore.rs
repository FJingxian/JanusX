// src/gfcore.rs
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use flate2::read::MultiGzDecoder;
use memmap2::{Mmap, MmapOptions};

const BED_HEADER_LEN: usize = 3;

#[inline]
fn system_page_size() -> usize {
    #[cfg(unix)]
    {
        let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if ps > 0 {
            ps as usize
        } else {
            4096
        }
    }
    #[cfg(not(unix))]
    {
        4096
    }
}

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
    fill_missing: bool,
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
            if fill_missing {
                row.fill(0.0);
            }
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

    if fill_missing {
        let mean_g = alt_sum / non_missing as f64;
        let imputed: f32 = mean_g as f32;
        for g in row.iter_mut() {
            if *g < 0.0 {
                *g = imputed;
            }
        }
    }

    true
}

// ======================================================================
// BED SNP iterator (single SNP each time): returns Vec<f32> (len n)
// ======================================================================
pub struct BedSnpIter {
    #[allow(dead_code)]
    pub prefix: String,
    pub samples: Vec<String>,
    pub sites: Vec<SiteInfo>,
    mmap: Mmap,
    mmap_offset: usize,
    window_snps: Option<usize>,
    window_start_snp: usize,
    window_len_snps: usize,
    bed_len: usize,
    bed_file: Option<File>,
    n_samples: usize,
    n_snps: usize,
    bytes_per_snp: usize,
    cur: usize,
    maf: f32,
    miss: f32,
    fill_missing: bool,
}

impl BedSnpIter {
    pub fn new_with_fill(prefix: &str, maf: f32, miss: f32, fill_missing: bool) -> Result<Self, String> {
        let samples = read_fam(prefix)?;
        let sites = read_bim(prefix)?;
        let n_samples = samples.len();
        let n_snps = sites.len();

        let bed_path = format!("{prefix}.bed");
        let file = File::open(&bed_path).map_err(|e| e.to_string())?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
        let bed_len = mmap.len();

        if bed_len < BED_HEADER_LEN {
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
            mmap_offset: 0,
            window_snps: None,
            window_start_snp: 0,
            window_len_snps: n_snps,
            bed_len,
            bed_file: None,
            n_samples,
            n_snps,
            bytes_per_snp,
            cur: 0,
            maf,
            miss,
            fill_missing,
        })
    }

    pub fn new_with_fill_window(
        prefix: &str,
        maf: f32,
        miss: f32,
        fill_missing: bool,
        mmap_window_mb: usize,
    ) -> Result<Self, String> {
        if mmap_window_mb == 0 {
            return Err("mmap_window_mb must be > 0".into());
        }
        let samples = read_fam(prefix)?;
        let sites = read_bim(prefix)?;
        let n_samples = samples.len();
        let n_snps = sites.len();

        let bed_path = format!("{prefix}.bed");
        let mut file = File::open(&bed_path).map_err(|e| e.to_string())?;
        let bed_len = file.metadata().map_err(|e| e.to_string())?.len() as usize;

        let mut header = [0u8; BED_HEADER_LEN];
        file.read_exact(&mut header).map_err(|e| e.to_string())?;
        if header[0] != 0x6C || header[1] != 0x1B || header[2] != 0x01 {
            return Err("Only SNP-major BED supported".into());
        }

        let bytes_per_snp = (n_samples + 3) / 4;
        let window_bytes = mmap_window_mb.saturating_mul(1024 * 1024);
        let window_snps = std::cmp::max(1, window_bytes / bytes_per_snp);
        let (mmap, mmap_offset, window_len_snps) = Self::map_window(
            &file,
            bed_len,
            0,
            window_snps,
            bytes_per_snp,
        )?;

        Ok(Self {
            prefix: prefix.to_string(),
            samples,
            sites,
            mmap,
            mmap_offset,
            window_snps: Some(window_snps),
            window_start_snp: 0,
            window_len_snps,
            bed_len,
            bed_file: Some(file),
            n_samples,
            n_snps,
            bytes_per_snp,
            cur: 0,
            maf,
            miss,
            fill_missing,
        })
    }

    fn map_window(
        file: &File,
        bed_len: usize,
        start_snp: usize,
        window_snps: usize,
        bytes_per_snp: usize,
    ) -> Result<(Mmap, usize, usize), String> {
        let data_len = bed_len.checked_sub(BED_HEADER_LEN)
            .ok_or_else(|| "BED too small".to_string())?;
        let max_snps = data_len / bytes_per_snp;
        if start_snp >= max_snps {
            return Err("window start SNP out of range".into());
        }
        let remaining_snps = max_snps - start_snp;
        let map_snps = remaining_snps.min(window_snps);

        let desired_offset = BED_HEADER_LEN + start_snp * bytes_per_snp;
        let page_size = system_page_size();
        let aligned_offset = desired_offset / page_size * page_size;
        let leading = desired_offset - aligned_offset;
        let desired_len = leading + map_snps * bytes_per_snp;
        let max_len = bed_len - aligned_offset;
        let map_len = desired_len.min(max_len);

        let mmap = unsafe {
            MmapOptions::new()
                .offset(aligned_offset as u64)
                .len(map_len)
                .map(file)
                .map_err(|e| e.to_string())?
        };
        Ok((mmap, aligned_offset, map_snps))
    }

    fn remap_window(&mut self, start_snp: usize) -> Result<(), String> {
        let window_snps = self.window_snps.ok_or_else(|| "window_snps not set".to_string())?;
        let file = self.bed_file.as_ref().ok_or_else(|| "bed_file not set".to_string())?;
        let (mmap, mmap_offset, window_len_snps) = Self::map_window(
            file,
            self.bed_len,
            start_snp,
            window_snps,
            self.bytes_per_snp,
        )?;
        self.mmap = mmap;
        self.mmap_offset = mmap_offset;
        self.window_start_snp = start_snp;
        self.window_len_snps = window_len_snps;
        Ok(())
    }

    fn snp_bytes(&self, snp_idx: usize) -> Option<&[u8]> {
        let offset = BED_HEADER_LEN + snp_idx * self.bytes_per_snp;
        if let Some(_window_snps) = self.window_snps {
            let window_end = self.window_start_snp + self.window_len_snps;
            if snp_idx < self.window_start_snp || snp_idx >= window_end {
                return None;
            }
        }
        let rel = offset.checked_sub(self.mmap_offset)?;
        let end = rel + self.bytes_per_snp;
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[rel..end])
    }

    /// 随机访问解码某个 SNP（用于并行）
    pub fn get_snp_row(&self, snp_idx: usize) -> Option<(Vec<f32>, SiteInfo)> {
        if snp_idx >= self.n_snps { return None; }
        if self.window_snps.is_some() {
            panic!("windowed mmap does not support random SNP access");
        }

        let snp_bytes = self.snp_bytes(snp_idx)?;

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
        let keep = crate::gfcore::process_snp_row(
            &mut row,
            &mut site.ref_allele,
            &mut site.alt_allele,
            self.maf,
            self.miss,
            self.fill_missing,
        );
        if keep { Some((row, site)) } else { None }
    }

    pub fn n_samples(&self) -> usize { self.n_samples }

    pub fn next_snp(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        while self.cur < self.n_snps {
            let snp_idx = self.cur;
            self.cur += 1;

            if self.window_snps.is_some() {
                let window_end = self.window_start_snp + self.window_len_snps;
                if snp_idx < self.window_start_snp || snp_idx >= window_end {
                    if let Err(e) = self.remap_window(snp_idx) {
                        panic!("failed to remap BED window: {e}");
                    }
                }
            }

            let snp_bytes = match self.snp_bytes(snp_idx) {
                Some(bytes) => bytes,
                None => return None,
            };

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
            let keep = process_snp_row(
                &mut row,
                &mut site.ref_allele,
                &mut site.alt_allele,
                self.maf,
                self.miss,
                self.fill_missing,
            );
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
    fill_missing: bool,
    finished: bool,
}

impl VcfSnpIter {
    pub fn new_with_fill(path: &str, maf: f32, miss: f32, fill_missing: bool) -> Result<Self, String> {
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

        Ok(Self {
            samples,
            reader,
            maf,
            miss,
            fill_missing,
            finished: false,
        })
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

            let keep = process_snp_row(
                &mut row,
                &mut site.ref_allele,
                &mut site.alt_allele,
                self.maf,
                self.miss,
                self.fill_missing,
            );
            if keep {
                return Some((row, site));
            }
        }
    }
}
