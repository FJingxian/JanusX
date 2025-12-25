use anyhow::{anyhow, Context, Result};
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};
use zip::ZipArchive;

use crate::core::Site;

/// 读取 .idv：每行一个样本名
pub fn load_idv(path: &Path) -> Result<Vec<String>> {
    let r = BufReader::new(File::open(path)?);
    let mut v = Vec::new();
    for line in r.lines() {
        let s = line?.trim().to_string();
        if !s.is_empty() {
            v.push(s);
        }
    }
    Ok(v)
}

/// 读取 .snp：chr pos ref alt
pub fn load_snp(path: &Path) -> Result<Vec<Site>> {
    let r = BufReader::new(File::open(path)?);
    let mut v = Vec::new();
    for (ln, line) in r.lines().enumerate() {
        let line = line?;
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = t.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(anyhow!("Bad .snp line {}: {}", ln + 1, t));
        }
        let chr = parts[0].to_string();
        let pos: u64 = parts[1].parse().context("POS parse")?;
        let refb = parts[2]
            .as_bytes()
            .get(0)
            .copied()
            .unwrap_or(b'N')
            .to_ascii_uppercase();
        let altb = parts[3]
            .as_bytes()
            .get(0)
            .copied()
            .unwrap_or(b'N')
            .to_ascii_uppercase();
        v.push(Site {
            chr,
            pos,
            ref_base: refb,
            alt_base: altb,
        });
    }
    Ok(v)
}

/// 从 .npz 读取 int8 矩阵，尝试 "arr_0.npy"，失败再尝试 "M.npy"
pub fn load_npz_i8_matrix(path: &Path) -> Result<Array2<i8>> {
    let mut buf = Vec::new();

    // 第一次尝试：arr_0.npy
    if let Ok(file) = File::open(path) {
        if let Ok(mut zip) = ZipArchive::new(file) {
            if let Ok(mut npy) = zip.by_name("arr_0.npy") {
                npy.read_to_end(&mut buf)?;
            }
        }
    }

    // 如果上面没读到任何内容，再尝试 M.npy
    if buf.is_empty() {
        let file = File::open(path)?;
        let mut zip = ZipArchive::new(file).context("Open NPZ as zip")?;
        let mut npy = zip
            .by_name("M.npy")
            .context("Cannot find arr_0.npy or M.npy in npz")?;
        npy.read_to_end(&mut buf)?;
    }

    let cursor = std::io::Cursor::new(buf);
    let arr: Array2<i8> =
        ReadNpyExt::read_npy(cursor).context("Read npy array")?;
    Ok(arr)
}