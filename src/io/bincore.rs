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
#[cfg(test)]
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
