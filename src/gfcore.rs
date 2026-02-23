// src/gfcore.rs
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

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
pub fn open_text_maybe_gz(path: &Path) -> Result<Box<dyn BufRead + Send + Sync>, String> {
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

fn parse_delimiter_char(delimiter: Option<&str>) -> Result<Option<char>, String> {
    match delimiter {
        None => Ok(None),
        Some(s) => {
            let d = s.trim();
            if d.is_empty() {
                return Ok(None);
            }
            if d == "\\t" {
                return Ok(Some('\t'));
            }
            let mut chars = d.chars();
            let ch = chars
                .next()
                .ok_or_else(|| "delimiter is empty".to_string())?;
            if chars.next().is_some() {
                return Err(format!(
                    "delimiter must be a single character or \"\\\\t\", got: {d}"
                ));
            }
            Ok(Some(ch))
        }
    }
}

fn split_line_tokens<'a>(line: &'a str, delimiter: Option<char>) -> Vec<&'a str> {
    if let Some(delim) = delimiter {
        if delim.is_ascii_whitespace() {
            line.split_whitespace().collect()
        } else {
            line.split(|c: char| c == delim || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .collect()
        }
    } else {
        line.split(|c: char| c.is_whitespace() || c == ',' || c == ';')
            .filter(|s| !s.is_empty())
            .collect()
    }
}

fn read_txt_header_samples(path: &Path, delimiter: Option<char>) -> Result<Vec<String>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);

    for (line_no, line) in reader.lines().enumerate() {
        let l = line.map_err(|e| format!("{}:{}: {}", path.display(), line_no + 1, e))?;
        let trimmed = l.trim();
        if trimmed.is_empty() {
            continue;
        }
        let toks = split_line_tokens(trimmed, delimiter);
        if toks.is_empty() {
            continue;
        }
        return Ok(toks.iter().map(|s| s.to_string()).collect());
    }
    Err(format!(
        "no header row (sample IDs) found in {}",
        path.display()
    ))
}

fn default_sites(n_rows: usize) -> Vec<SiteInfo> {
    let mut sites = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let pos = i32::try_from(i + 1).unwrap_or(i32::MAX);
        sites.push(SiteInfo {
            chrom: "N".to_string(),
            pos,
            ref_allele: "N".to_string(),
            alt_allele: "N".to_string(),
        });
    }
    sites
}

fn write_npy_f32_header(w: &mut File, rows: usize, cols: usize) -> Result<usize, String> {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    let preamble_len = 10usize; // magic(6)+version(2)+header_len(2)
    let pad_len = (16 - ((preamble_len + header.len() + 1) % 16)) % 16;
    if pad_len > 0 {
        header.push_str(&" ".repeat(pad_len));
    }
    header.push('\n');

    let header_len = header.len();
    if header_len > u16::MAX as usize {
        return Err("NPY header too large for v1.0".into());
    }

    w.write_all(b"\x93NUMPY").map_err(|e| e.to_string())?;
    w.write_all(&[1u8, 0u8]).map_err(|e| e.to_string())?;
    w.write_all(&(header_len as u16).to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(header.as_bytes()).map_err(|e| e.to_string())?;

    Ok(preamble_len + header_len)
}

fn parse_npy_shape(header: &str) -> Result<(usize, usize), String> {
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

fn parse_npy_f32_header(bytes: &[u8]) -> Result<(usize, usize, usize), String> {
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

    let (rows, cols) = parse_npy_shape(header)?;
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

fn convert_text_matrix_to_npy(
    txt_path: &Path,
    npy_path: &Path,
    delimiter: Option<char>,
    expected_cols: usize,
) -> Result<(usize, usize), String> {
    let input = File::open(txt_path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(input);

    let raw_tmp_path = PathBuf::from(format!("{}.rawtmp", npy_path.to_string_lossy()));
    let raw_tmp = File::create(&raw_tmp_path).map_err(|e| e.to_string())?;
    let mut raw_writer = BufWriter::new(raw_tmp);

    let mut n_rows: usize = 0;
    let mut seen_header = false;

    for (line_no, line) in reader.lines().enumerate() {
        let l = line.map_err(|e| format!("{}:{}: {}", txt_path.display(), line_no + 1, e))?;
        let trimmed = l.trim();
        if trimmed.is_empty() {
            continue;
        }
        let toks = split_line_tokens(trimmed, delimiter);
        if toks.is_empty() {
            continue;
        }

        if !seen_header {
            seen_header = true;
            if toks.len() != expected_cols {
                return Err(format!(
                    "header column count mismatch at {}:{}: expected {}, got {}",
                    txt_path.display(),
                    line_no + 1,
                    expected_cols,
                    toks.len()
                ));
            }
            continue;
        }

        if toks.len() != expected_cols {
            return Err(format!(
                "column count mismatch at {}:{}: expected {}, got {}",
                txt_path.display(),
                line_no + 1,
                expected_cols,
                toks.len()
            ));
        }

        for tok in toks {
            let v: f32 = tok.parse().map_err(|_| {
                format!(
                    "invalid float at {}:{} -> {}",
                    txt_path.display(),
                    line_no + 1,
                    tok
                )
            })?;
            raw_writer
                .write_all(&v.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
        n_rows += 1;
    }
    raw_writer.flush().map_err(|e| e.to_string())?;

    if !seen_header {
        let _ = fs::remove_file(&raw_tmp_path);
        return Err(format!(
            "no header row (sample IDs) found in {}",
            txt_path.display()
        ));
    }
    if n_rows == 0 {
        let _ = fs::remove_file(&raw_tmp_path);
        return Err(format!(
            "no numeric matrix rows found after header in {}",
            txt_path.display()
        ));
    }

    let mut npy_file = File::create(npy_path).map_err(|e| e.to_string())?;
    write_npy_f32_header(&mut npy_file, n_rows, expected_cols)?;
    let mut raw_reader = File::open(&raw_tmp_path).map_err(|e| e.to_string())?;
    std::io::copy(&mut raw_reader, &mut npy_file).map_err(|e| e.to_string())?;
    npy_file.sync_all().map_err(|e| e.to_string())?;
    let _ = fs::remove_file(&raw_tmp_path);
    Ok((n_rows, expected_cols))
}

fn ensure_cached_text_npy(
    txt_path: &Path,
    npy_path: &Path,
    delimiter: Option<char>,
    expected_cols: usize,
) -> Result<(usize, usize), String> {
    let txt_mtime = fs::metadata(txt_path).and_then(|m| m.modified()).ok();
    let npy_mtime = fs::metadata(npy_path).and_then(|m| m.modified()).ok();
    let use_cache =
        npy_path.exists() && matches!((txt_mtime, npy_mtime), (Some(t), Some(n)) if n >= t);

    if use_cache {
        let file = File::open(npy_path).map_err(|e| e.to_string())?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
        if let Ok((rows, cols, _)) = parse_npy_f32_header(&mmap[..]) {
            return Ok((rows, cols));
        }
    }

    convert_text_matrix_to_npy(txt_path, npy_path, delimiter, expected_cols)
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
    pub fn new_with_fill(
        prefix: &str,
        maf: f32,
        miss: f32,
        fill_missing: bool,
    ) -> Result<Self, String> {
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
        let (mmap, mmap_offset, window_len_snps) =
            Self::map_window(&file, bed_len, 0, window_snps, bytes_per_snp)?;

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
        let data_len = bed_len
            .checked_sub(BED_HEADER_LEN)
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
        let window_snps = self
            .window_snps
            .ok_or_else(|| "window_snps not set".to_string())?;
        let file = self
            .bed_file
            .as_ref()
            .ok_or_else(|| "bed_file not set".to_string())?;
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
        if snp_idx >= self.n_snps {
            return None;
        }
        if self.window_snps.is_some() {
            panic!("windowed mmap does not support random SNP access");
        }

        let snp_bytes = self.snp_bytes(snp_idx)?;

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

        let mut site = self.sites[snp_idx].clone();
        let keep = crate::gfcore::process_snp_row(
            &mut row,
            &mut site.ref_allele,
            &mut site.alt_allele,
            self.maf,
            self.miss,
            self.fill_missing,
        );
        if keep {
            Some((row, site))
        } else {
            None
        }
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

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
    reader: Box<dyn BufRead + Send + Sync + 'static>,
    maf: f32,
    miss: f32,
    fill_missing: bool,
    finished: bool,
}

impl VcfSnpIter {
    pub fn new_with_fill(
        path: &str,
        maf: f32,
        miss: f32,
        fill_missing: bool,
    ) -> Result<Self, String> {
        let p = Path::new(path);
        let mut reader = open_text_maybe_gz(p)?;

        // parse header to get samples
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
            maf,
            miss,
            fill_missing,
            finished: false,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn next_snp(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
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

            let format = parts[8];
            if !format.split(':').any(|f| f == "GT") {
                continue;
            }

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

// ======================================================================
// TXT float32 matrix iterator:
// - read numeric text matrix in chunks
// - convert/cache as .npy (float32, C-order)
// - mmap .npy for row-wise streaming
// ======================================================================
pub struct TxtSnpIter {
    #[allow(dead_code)]
    pub path: String,
    #[allow(dead_code)]
    pub npy_path: String,
    pub samples: Vec<String>,
    pub sites: Vec<SiteInfo>,
    mmap: Mmap,
    data_offset: usize,
    n_samples: usize,
    n_snps: usize,
    row_bytes: usize,
    cur: usize,
}

impl TxtSnpIter {
    pub fn new(path: &str, delimiter: Option<&str>) -> Result<Self, String> {
        let txt_path = Path::new(path);
        if !txt_path.exists() {
            return Err(format!(
                "text matrix file not found: {}",
                txt_path.display()
            ));
        }

        let delimiter = parse_delimiter_char(delimiter)?;
        let samples = read_txt_header_samples(txt_path, delimiter)?;
        let n_samples = samples.len();
        if n_samples == 0 {
            return Err("header sample IDs are empty".into());
        }
        {
            let mut seen = std::collections::HashSet::with_capacity(n_samples);
            for sid in samples.iter() {
                if !seen.insert(sid.as_str()) {
                    return Err(format!("duplicate sample ID in header: {sid}"));
                }
            }
        }

        let npy_path = PathBuf::from(format!("{path}.f32.npy"));
        let (n_snps, n_cols) = ensure_cached_text_npy(txt_path, &npy_path, delimiter, n_samples)?;
        if n_cols != n_samples {
            return Err(format!(
                "header sample count mismatch: header={}, matrix columns={}",
                n_samples, n_cols
            ));
        }
        let sites = default_sites(n_snps);

        let file = File::open(&npy_path).map_err(|e| e.to_string())?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
        let (rows, cols, data_offset) = parse_npy_f32_header(&mmap[..])?;
        if rows != n_snps || cols != n_samples {
            return Err(format!(
                "cached NPY shape mismatch for {}: expected ({}, {}), got ({}, {})",
                npy_path.display(),
                n_snps,
                n_samples,
                rows,
                cols
            ));
        }
        let row_bytes = n_samples
            .checked_mul(4)
            .ok_or_else(|| "row byte size overflow".to_string())?;

        Ok(Self {
            path: path.to_string(),
            npy_path: npy_path.to_string_lossy().to_string(),
            samples,
            sites,
            mmap,
            data_offset,
            n_samples,
            n_snps,
            row_bytes,
            cur: 0,
        })
    }

    fn row_bytes(&self, snp_idx: usize) -> Option<&[u8]> {
        if snp_idx >= self.n_snps {
            return None;
        }
        let start = self
            .data_offset
            .checked_add(snp_idx.checked_mul(self.row_bytes)?)?;
        let end = start.checked_add(self.row_bytes)?;
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[start..end])
    }

    pub fn get_snp_row(&self, snp_idx: usize) -> Option<(Vec<f32>, SiteInfo)> {
        let bytes = self.row_bytes(snp_idx)?;
        let mut row = Vec::with_capacity(self.n_samples);
        for chunk in bytes.chunks_exact(4) {
            row.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        if row.len() != self.n_samples {
            return None;
        }
        Some((row, self.sites[snp_idx].clone()))
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn next_snp(&mut self) -> Option<(Vec<f32>, SiteInfo)> {
        if self.cur >= self.n_snps {
            return None;
        }
        let idx = self.cur;
        self.cur += 1;
        self.get_snp_row(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::TxtSnpIter;
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
    fn txt_iter_defaults_work() {
        let dir = make_temp_dir("txt_default");
        let txt_path = dir.join("geno.txt");

        fs::write(&txt_path, "s1 s2 s3\n0.1 -0.2 0.3\n-0.4 0.5 -0.6\n").unwrap();

        let mut it = TxtSnpIter::new(txt_path.to_str().unwrap(), None).unwrap();
        assert_eq!(it.n_samples(), 3);
        assert_eq!(it.sites.len(), 2);
        assert_eq!(it.sites[0].chrom, "N");
        assert_eq!(it.sites[0].pos, 1);
        assert_eq!(it.sites[1].chrom, "N");
        assert_eq!(it.sites[1].pos, 2);

        let (row0, site0) = it.next_snp().unwrap();
        assert_eq!(site0.pos, 1);
        assert!((row0[0] - 0.1).abs() < 1e-6);
        assert!((row0[1] + 0.2).abs() < 1e-6);
        assert!((row0[2] - 0.3).abs() < 1e-6);

        let (row1, site1) = it.next_snp().unwrap();
        assert_eq!(site1.pos, 2);
        assert!((row1[0] + 0.4).abs() < 1e-6);
        assert!((row1[1] - 0.5).abs() < 1e-6);
        assert!((row1[2] + 0.6).abs() < 1e-6);
        assert!(it.next_snp().is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn txt_iter_reads_header_delim() {
        let dir = make_temp_dir("txt_header_delim");
        let txt_path = dir.join("geno.txt");

        fs::write(&txt_path, "a,b,c\n1,2,3\n4,5,6\n").unwrap();

        let it = TxtSnpIter::new(txt_path.to_str().unwrap(), Some(",")).unwrap();
        assert_eq!(it.n_samples(), 3);
        assert_eq!(it.sites.len(), 2);
        assert_eq!(it.samples[0], "a");
        assert_eq!(it.samples[1], "b");
        assert_eq!(it.samples[2], "c");
        assert_eq!(it.sites[0].chrom, "N");
        assert_eq!(it.sites[0].pos, 1);
        assert_eq!(it.sites[0].ref_allele, "N");
        assert_eq!(it.sites[0].alt_allele, "N");

        let _ = fs::remove_dir_all(&dir);
    }
}
