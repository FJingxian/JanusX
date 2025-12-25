use anyhow::{anyhow, Result};
use flate2::read::MultiGzDecoder;
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

use crate::core::Site;

/// 直接按文本解析 VCF/VCF.GZ，支持 0/0, 0/1, 1/1, ./.
pub struct VcfStream {
    rdr: BufReader<Box<dyn Read>>,
    samples: Vec<String>,
}

impl VcfStream {
    /// 打开 VCF 文件，自动判断是否为 .gz，并读取 header 中的样本名
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader: Box<dyn Read> = match path.extension().and_then(|e| e.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("gz") => {
                Box::new(MultiGzDecoder::new(file))
            }
            _ => Box::new(file),
        };
        let mut rdr = BufReader::new(reader);

        let mut line = String::new();
        let mut samples = Vec::new();

        // 读 header，拿到 #CHROM 这一行里的样本名
        loop {
            line.clear();
            let n = rdr.read_line(&mut line)?;
            if n == 0 {
                break;
            }
            let t = line.trim_end();
            if t.starts_with("##") {
                continue;
            }
            if t.starts_with("#CHROM") {
                let fields: Vec<&str> = t.split('\t').collect();
                if fields.len() < 10 {
                    return Err(anyhow!(
                        "#CHROM line has < 10 columns, cannot find samples"
                    ));
                }
                samples = fields[9..].iter().map(|s| s.to_string()).collect();
                break;
            }
        }

        if samples.is_empty() {
            return Err(anyhow!(
                "No samples found in VCF header (#CHROM line missing or malformed)"
            ));
        }

        Ok(Self { rdr, samples })
    }

    pub fn samples(&self) -> &[String] {
        &self.samples
    }

    /// 从 VCF 中依次读取下一个 biallelic SNP 位点，返回 (Site, genotypes)
    /// genotypes 为 i8：-9(缺失), 0(0/0), 1(0/1,1/0), 2(1/1)
    pub fn next_site(&mut self) -> Result<Option<(Site, Vec<i8>)>> {
        let mut line = String::new();

        loop {
            line.clear();
            let n = self.rdr.read_line(&mut line)?;
            if n == 0 {
                // EOF
                return Ok(None);
            }

            let t = line.trim_end();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }

            let cols: Vec<&str> = t.split('\t').collect();
            if cols.len() < 10 {
                // 非法行，跳过
                continue;
            }

            let chr = cols[0].to_string();
            let pos: u64 = match cols[1].parse() {
                Ok(p) => p,
                Err(_) => continue,
            };
            let ref_allele = cols[3];
            let alt_allele = cols[4];

            // 只保留单一 ALT、单碱基 SNP
            if alt_allele.contains(',') {
                continue;
            }
            if ref_allele.len() != 1 || alt_allele.len() != 1 {
                continue;
            }

            let rb = ref_allele.as_bytes()[0].to_ascii_uppercase();
            let ab = alt_allele.as_bytes()[0].to_ascii_uppercase();
            if !matches!(rb, b'A' | b'C' | b'G' | b'T') {
                continue;
            }
            if !matches!(ab, b'A' | b'C' | b'G' | b'T') {
                continue;
            }

            let site = Site {
                chr,
                pos,
                ref_base: rb,
                alt_base: ab,
            };

            // 解析每个样本的 GT 字段
            let ns = self.samples.len();
            if cols.len() < 9 + ns {
                // 列数不够，跳过
                continue;
            }

            let mut genos = Vec::with_capacity(ns);

            for i in 0..ns {
                let field = cols[9 + i]; // 第一个样本在第 10 列（索引 9）
                let gt_part = field.split(':').next().unwrap_or(".");

                // 缺失：./.  .|.  . 或者空
                if gt_part == "./."
                    || gt_part == ".|."
                    || gt_part == "."
                    || gt_part.is_empty()
                {
                    genos.push(-9);
                    continue;
                }

                // 期待形如 0/0, 0/1, 1/0, 1/1 或 0|1 等
                let sep = if gt_part.contains('/') {
                    '/'
                } else if gt_part.contains('|') {
                    '|'
                } else {
                    // 其他奇怪格式，直接当缺失
                    genos.push(-9);
                    continue;
                };

                let parts: Vec<&str> = gt_part.split(sep).collect();
                if parts.len() != 2 {
                    genos.push(-9);
                    continue;
                }

                let a0 = parts[0];
                let a1 = parts[1];

                // 只接受 0 和 1，其他都视为缺失或多等位
                if !(a0 == "0" || a0 == "1") || !(a1 == "0" || a1 == "1") {
                    genos.push(-9);
                    continue;
                }

                let i0 = if a0 == "0" { 0i8 } else { 1i8 };
                let i1 = if a1 == "0" { 0i8 } else { 1i8 };

                let sum = i0 + i1;
                let g = match sum {
                    0 => 0, // 0/0
                    1 => 1, // 0/1 或 1/0
                    2 => 2, // 1/1
                    _ => -9,
                };
                genos.push(g);
            }

            return Ok(Some((site, genos)));
        }
    }
}