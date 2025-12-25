use anyhow::{anyhow, Result};
use std::{fs::File, io::{BufRead, BufReader, Read}, path::Path};

use crate::core::Site;

pub struct PlinkStream {
    bed: BufReader<File>,
    sites: Vec<Site>,
    samples: Vec<String>,
    n: usize,
    cur: usize,
    bytes_per_snp: usize,
}

impl PlinkStream {
    pub fn open(bed: &Path, bim: &Path, fam: &Path) -> Result<Self> {
        let samples = read_fam(fam)?;
        let sites = read_bim_as_sites(bim)?;
        let n = samples.len();
        if n == 0 { return Err(anyhow!("No samples in FAM")); }

        let mut bedf = BufReader::new(File::open(bed)?);

        // read header
        let mut header = [0u8; 3];
        bedf.read_exact(&mut header)?;
        if header[0] != 0x6C || header[1] != 0x1B {
            return Err(anyhow!("Not a PLINK .bed file (bad magic)"));
        }
        if header[2] != 0x01 {
            return Err(anyhow!("BED is not SNP-major (mode byte != 0x01)"));
        }

        let bytes_per_snp = (n + 3) / 4; // 4 samples per byte (2 bits each)

        Ok(Self {
            bed: bedf,
            sites,
            samples,
            n,
            cur: 0,
            bytes_per_snp,
        })
    }

    pub fn samples(&self) -> &[String] { &self.samples }

    pub fn next_site(&mut self) -> Result<Option<(Site, Vec<i8>)>> {
        if self.cur >= self.sites.len() {
            return Ok(None);
        }
        let site = self.sites[self.cur].clone();
        self.cur += 1;

        let mut buf = vec![0u8; self.bytes_per_snp];
        self.bed.read_exact(&mut buf)?;

        let mut genos = vec![-9i8; self.n];
        let mut idx = 0usize;

        for b in buf {
            for shift in (0..8).step_by(2) {
                if idx >= self.n { break; }
                let code = (b >> shift) & 0b11;
                // PLINK1:
                // 00 homo A1
                // 10 het
                // 11 homo A2
                // 01 missing
                genos[idx] = match code {
                    0b00 => 0,
                    0b10 => 1,
                    0b11 => 2,
                    0b01 => -9,
                    _ => -9,
                };
                idx += 1;
            }
        }

        Ok(Some((site, genos)))
    }
}

fn read_fam(path: &Path) -> Result<Vec<String>> {
    let r = BufReader::new(File::open(path)?);
    let mut v = Vec::new();
    for line in r.lines() {
        let t = line?;
        if t.trim().is_empty() { continue; }
        let parts: Vec<&str> = t.split_whitespace().collect();
        if parts.len() < 2 { continue; }
        // FID IID ...
        v.push(parts[1].to_string());
    }
    Ok(v)
}

fn read_bim_as_sites(path: &Path) -> Result<Vec<Site>> {
    let r = BufReader::new(File::open(path)?);
    let mut v = Vec::new();
    for (ln, line) in r.lines().enumerate() {
        let t = line?;
        if t.trim().is_empty() { continue; }
        let parts: Vec<&str> = t.split_whitespace().collect();
        // CHR SNPID CM POS A1 A2
        if parts.len() < 6 {
            return Err(anyhow!("Bad BIM line {}: {}", ln + 1, t));
        }
        let chr = parts[0].to_string();
        let pos: u64 = parts[3].parse()?;
        let a1 = parts[4].as_bytes().get(0).copied().unwrap_or(b'N').to_ascii_uppercase();
        let a2 = parts[5].as_bytes().get(0).copied().unwrap_or(b'N').to_ascii_uppercase();

        // REF=A1 ALT=A2（你后续想切换，只要在这里换一下即可）
        v.push(Site { chr, pos, ref_base: a1, alt_base: a2 });
    }
    Ok(v)
}