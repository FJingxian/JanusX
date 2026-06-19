use crate::gfcore::SiteInfo;
use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const BIN01_MAGIC: &[u8; 8] = b"JXBIN001";
pub const BIN01_HEADER_LEN: usize = 32;

#[derive(Clone, Debug)]
pub struct BinWindow {
    pub chrom: String,
    pub bp_start: i32,
    pub bp_end: i32,
    pub indices: Vec<usize>,
}

#[inline]
pub fn append_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut os: OsString = prefix.as_os_str().to_os_string();
    os.push(suffix);
    PathBuf::from(os)
}

pub fn bin01_header_bytes(n_rows: u64, n_samples: usize) -> [u8; BIN01_HEADER_LEN] {
    let mut header = [0u8; BIN01_HEADER_LEN];
    header[0..8].copy_from_slice(BIN01_MAGIC);
    header[8..16].copy_from_slice(&n_rows.to_le_bytes());
    header[16..24].copy_from_slice(&(n_samples as u64).to_le_bytes());
    header[24..32].copy_from_slice(&(0u64).to_le_bytes());
    header
}

pub fn write_bin01_header<W: Write>(
    writer: &mut W,
    n_rows: u64,
    n_samples: usize,
    ctx: &str,
) -> Result<(), String> {
    writer
        .write_all(&bin01_header_bytes(n_rows, n_samples))
        .map_err(|e| format!("{ctx}: write BIN01 header: {e}"))
}

pub fn parse_bin01_header(bytes: &[u8], ctx: &str) -> Result<(usize, usize, usize, usize), String> {
    if bytes.len() < BIN01_HEADER_LEN {
        return Err(format!("{ctx}: BIN file too small"));
    }
    if &bytes[0..8] != BIN01_MAGIC {
        return Err(format!("{ctx}: invalid BIN magic (expected JXBIN001)"));
    }

    let n_rows_u64 = u64::from_le_bytes(
        bytes[8..16]
            .try_into()
            .map_err(|_| format!("{ctx}: malformed BIN header n_rows"))?,
    );
    let n_samples_u64 = u64::from_le_bytes(
        bytes[16..24]
            .try_into()
            .map_err(|_| format!("{ctx}: malformed BIN header n_samples"))?,
    );

    let n_rows = usize::try_from(n_rows_u64)
        .map_err(|_| format!("{ctx}: n_rows too large for this platform"))?;
    let n_samples = usize::try_from(n_samples_u64)
        .map_err(|_| format!("{ctx}: n_samples too large for this platform"))?;
    if n_samples == 0 {
        return Err(format!("{ctx}: n_samples is zero"));
    }

    let row_bytes = n_samples.div_ceil(8);
    let data_bytes = n_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| format!("{ctx}: BIN payload size overflow"))?;
    let expected = BIN01_HEADER_LEN
        .checked_add(data_bytes)
        .ok_or_else(|| format!("{ctx}: BIN file size overflow"))?;
    if bytes.len() < expected {
        return Err(format!(
            "{ctx}: BIN payload truncated (have {}, need at least {})",
            bytes.len(),
            expected
        ));
    }

    Ok((n_rows, n_samples, row_bytes, BIN01_HEADER_LEN))
}

pub fn resolve_bin01_path(path: &str) -> Result<PathBuf, String> {
    let raw = path.trim();
    if raw.is_empty() {
        return Err("BIN path must not be empty".to_string());
    }
    let p = Path::new(raw);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    let mut os = p.as_os_str().to_os_string();
    os.push(".bin");
    let with_bin = PathBuf::from(os);
    if with_bin.exists() {
        return Ok(with_bin);
    }
    Err(format!("BIN file not found: {raw}"))
}

pub fn bin_prefix(path: &str) -> PathBuf {
    let p = Path::new(path);
    let mut out = p.to_path_buf();
    let is_bin = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("bin"))
        .unwrap_or(false);
    if is_bin {
        out.set_extension("");
    }
    out
}

#[inline]
pub fn bin01_id_sidecar_path(path: &str) -> PathBuf {
    append_suffix(&bin_prefix(path), ".bin.id")
}

#[inline]
pub fn bin01_site_sidecar_path(path: &str) -> PathBuf {
    append_suffix(&bin_prefix(path), ".bin.site")
}

pub fn discover_site_sidecar(path: &str) -> Option<PathBuf> {
    let prefix = bin_prefix(path);
    let candidates = [
        append_suffix(&prefix, ".bsite"),
        append_suffix(&prefix, ".site"),
        append_suffix(&prefix, ".site.tsv"),
        append_suffix(&prefix, ".site.txt"),
        append_suffix(&prefix, ".site.csv"),
        append_suffix(&prefix, ".sites.tsv"),
        append_suffix(&prefix, ".sites.txt"),
        append_suffix(&prefix, ".sites.csv"),
        append_suffix(&prefix, ".bin.site"),
        append_suffix(&prefix, ".bim"),
    ];
    candidates.into_iter().find(|cand| cand.exists())
}

pub fn discover_id_sidecar(path: &str) -> Option<PathBuf> {
    let prefix = bin_prefix(path);
    let candidates = [
        append_suffix(&prefix, ".bin.id"),
        append_suffix(&prefix, ".id"),
        append_suffix(&prefix, ".fam"),
    ];
    candidates.into_iter().find(|cand| cand.exists())
}

fn global_window(n_rows: usize) -> Vec<BinWindow> {
    vec![BinWindow {
        chrom: "ALL".to_string(),
        bp_start: 0,
        bp_end: 0,
        indices: (0..n_rows).collect(),
    }]
}

pub fn build_windows_from_sites<F>(
    sites: &[SiteInfo],
    n_rows: usize,
    extension: usize,
    step: usize,
    normalize_chrom: F,
) -> Vec<BinWindow>
where
    F: Fn(&str) -> String,
{
    if n_rows == 0 {
        return Vec::new();
    }
    if extension == 0 || step == 0 || sites.is_empty() {
        return global_window(n_rows);
    }

    let mut groups: HashMap<String, Vec<(i32, usize)>> = HashMap::new();
    let mut chrom_order: Vec<String> = Vec::new();
    for (idx, site) in sites.iter().enumerate().take(n_rows) {
        let chrom = normalize_chrom(&site.chrom);
        if !groups.contains_key(&chrom) {
            groups.insert(chrom.clone(), Vec::new());
            chrom_order.push(chrom.clone());
        }
        if let Some(v) = groups.get_mut(&chrom) {
            v.push((site.pos, idx));
        }
    }
    if groups.is_empty() {
        return global_window(n_rows);
    }

    let mut windows: Vec<BinWindow> = Vec::new();
    for chrom in chrom_order {
        let Some(mut pairs) = groups.remove(&chrom) else {
            continue;
        };
        if pairs.is_empty() {
            continue;
        }
        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let n = pairs.len();
        let bps: Vec<i32> = pairs.iter().map(|(bp, _)| *bp).collect();
        let idxs: Vec<usize> = pairs.iter().map(|(_, i)| *i).collect();
        let min_bp = bps[0];
        let max_bp = bps[n - 1];
        let mut l = 0usize;
        let mut r = 0usize;
        let mut center = min_bp;
        let mut prev_sig: Option<(usize, usize, usize)> = None;

        loop {
            let left_i64 = (center as i64)
                .saturating_sub(extension as i64)
                .max(min_bp as i64);
            let right_i64 = (center as i64)
                .saturating_add(extension as i64)
                .min(max_bp as i64);

            while l < n && (bps[l] as i64) < left_i64 {
                l += 1;
            }
            if r < l {
                r = l;
            }
            while r < n && (bps[r] as i64) <= right_i64 {
                r += 1;
            }

            if r > l {
                let chunk = idxs[l..r].to_vec();
                let sig = (chunk[0], chunk[chunk.len() - 1], chunk.len());
                if prev_sig != Some(sig) {
                    let bp_start = if left_i64 > i32::MAX as i64 {
                        i32::MAX
                    } else if left_i64 < i32::MIN as i64 {
                        i32::MIN
                    } else {
                        left_i64 as i32
                    };
                    let bp_end = if right_i64 > i32::MAX as i64 {
                        i32::MAX
                    } else if right_i64 < i32::MIN as i64 {
                        i32::MIN
                    } else {
                        right_i64 as i32
                    };
                    windows.push(BinWindow {
                        chrom: chrom.clone(),
                        bp_start,
                        bp_end,
                        indices: chunk,
                    });
                    prev_sig = Some(sig);
                }
            }

            if center >= max_bp {
                break;
            }
            let next = (center as i64).saturating_add(step as i64);
            if next > i32::MAX as i64 {
                break;
            }
            center = next as i32;
        }

        let has_chrom_window = windows.last().map(|w| w.chrom == chrom).unwrap_or(false);
        if !has_chrom_window {
            windows.push(BinWindow {
                chrom: chrom.clone(),
                bp_start: min_bp,
                bp_end: max_bp.saturating_add(1),
                indices: idxs,
            });
        }
    }

    if windows.is_empty() {
        global_window(n_rows)
    } else {
        windows
    }
}
