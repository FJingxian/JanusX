use std::io::Write;

pub const BIN_SITE_MAGIC: &[u8; 8] = b"JXBSITE1";
pub const BIN_SITE_HEADER_LEN: usize = 24;

pub const LEGACY_BSITE_MAGIC: &[u8; 8] = b"JXBSIT02";
pub const LEGACY_BSITE_HEADER_LEN: usize = 36;
pub const LEGACY_BSITE_VERSION: u16 = 1;

pub fn bin_site_header_bytes(n_sites: u64) -> [u8; BIN_SITE_HEADER_LEN] {
    let mut header = [0u8; BIN_SITE_HEADER_LEN];
    header[0..8].copy_from_slice(BIN_SITE_MAGIC);
    header[8..16].copy_from_slice(&n_sites.to_le_bytes());
    header[16..24].copy_from_slice(&(0u64).to_le_bytes());
    header
}

pub fn write_bin_site_header<W: Write>(
    writer: &mut W,
    n_sites: u64,
    ctx: &str,
) -> Result<(), String> {
    writer
        .write_all(&bin_site_header_bytes(n_sites))
        .map_err(|e| format!("{ctx}: write BIN site header: {e}"))
}
