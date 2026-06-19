use crate::bincore::parse_bin01_header;
use crate::bstats::{apply_tail_mask, tail_mask, words_for_samples};
use crate::kmer::format::{
    BIT_ORDER_LITTLE, BKMER_HEADER_SIZE, BKMER_MAGIC, BKMER_VERSION, BSITE_HEADER_SIZE,
    BSITE_MAGIC, BSITE_VERSION, COMPRESSION_NONE, ENCODING_ACGT_2BIT,
    MATRIX_LAYOUT_COLUMN_MAJOR_BITSET,
};
use memmap2::{Mmap, MmapOptions};
use std::fs::{self, File};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KmerBkmerHeaderView {
    pub k: u32,
    pub n_kmers: u64,
    pub encoding: u32,
    pub canonical: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KmerBsiteHeaderView {
    pub version: u32,
    pub layout: u32,
    pub n_samples: u64,
    pub n_kmers: u64,
    pub bytes_per_col: u64,
    pub bit_order: u32,
    pub compression: u32,
}

pub struct BkmerMmapView {
    pub mmap: Mmap,
    pub header: KmerBkmerHeaderView,
    label: String,
}

pub struct BsiteMmapView {
    pub mmap: Mmap,
    pub header: KmerBsiteHeaderView,
    label: String,
}

pub enum TypedBodyView<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
    Bytes,
}

#[inline]
pub fn bkmer_body_offset() -> u64 {
    BKMER_HEADER_SIZE as u64
}

#[inline]
pub fn bsite_body_offset() -> u64 {
    BSITE_HEADER_SIZE as u64
}

#[inline]
pub fn bsite_column_offset(col_idx: u64, bytes_per_col: u64) -> u64 {
    bsite_body_offset() + col_idx.saturating_mul(bytes_per_col)
}

pub fn parse_bkmer_header(header: &[u8], ctx: &str) -> Result<KmerBkmerHeaderView, String> {
    if header.len() != BKMER_HEADER_SIZE {
        return Err(format!("{ctx}: invalid bkmer header size {}", header.len()));
    }
    if header[..8] != BKMER_MAGIC {
        return Err(format!("{ctx}: invalid bkmer magic"));
    }
    let version = u32::from_le_bytes(
        header[8..12]
            .try_into()
            .map_err(|_| format!("{ctx}: malformed bkmer version"))?,
    );
    if version != BKMER_VERSION {
        return Err(format!(
            "{ctx}: unsupported bkmer version {} (expected {})",
            version, BKMER_VERSION
        ));
    }
    let encoding = u32::from_le_bytes(
        header[24..28]
            .try_into()
            .map_err(|_| format!("{ctx}: malformed bkmer encoding"))?,
    );
    if encoding != ENCODING_ACGT_2BIT {
        return Err(format!(
            "{ctx}: unsupported bkmer encoding {} (expected {})",
            encoding, ENCODING_ACGT_2BIT
        ));
    }
    Ok(KmerBkmerHeaderView {
        k: u32::from_le_bytes(
            header[12..16]
                .try_into()
                .map_err(|_| format!("{ctx}: malformed bkmer k"))?,
        ),
        n_kmers: u64::from_le_bytes(
            header[16..24]
                .try_into()
                .map_err(|_| format!("{ctx}: malformed bkmer n_kmers"))?,
        ),
        encoding,
        canonical: u32::from_le_bytes(
            header[28..32]
                .try_into()
                .map_err(|_| format!("{ctx}: malformed bkmer canonical"))?,
        ),
    })
}

pub fn parse_bsite_header(header: &[u8], ctx: &str) -> Result<KmerBsiteHeaderView, String> {
    if header.len() != BSITE_HEADER_SIZE {
        return Err(format!("{ctx}: invalid bsite header size {}", header.len()));
    }
    if header[..8] != BSITE_MAGIC {
        return Err(format!("{ctx}: invalid bsite magic"));
    }
    let version = u32::from_le_bytes(
        header[8..12]
            .try_into()
            .map_err(|_| format!("{ctx}: malformed bsite version"))?,
    );
    if version != BSITE_VERSION {
        return Err(format!(
            "{ctx}: unsupported bsite version {} (expected {})",
            version, BSITE_VERSION
        ));
    }
    let layout = u32::from_le_bytes(
        header[12..16]
            .try_into()
            .map_err(|_| format!("{ctx}: malformed bsite layout"))?,
    );
    if layout != MATRIX_LAYOUT_COLUMN_MAJOR_BITSET {
        return Err(format!(
            "{ctx}: unsupported bsite layout {} (expected {})",
            layout, MATRIX_LAYOUT_COLUMN_MAJOR_BITSET
        ));
    }
    let bit_order = u32::from_le_bytes(
        header[40..44]
            .try_into()
            .map_err(|_| format!("{ctx}: malformed bsite bit order"))?,
    );
    if bit_order != BIT_ORDER_LITTLE {
        return Err(format!(
            "{ctx}: unsupported bsite bit order {} (expected {})",
            bit_order, BIT_ORDER_LITTLE
        ));
    }
    let compression = u32::from_le_bytes(
        header[44..48]
            .try_into()
            .map_err(|_| format!("{ctx}: malformed bsite compression"))?,
    );
    if compression != COMPRESSION_NONE {
        return Err(format!(
            "{ctx}: unsupported bsite compression {} (expected {})",
            compression, COMPRESSION_NONE
        ));
    }
    Ok(KmerBsiteHeaderView {
        version,
        layout,
        n_samples: u64::from_le_bytes(
            header[16..24]
                .try_into()
                .map_err(|_| format!("{ctx}: malformed bsite n_samples"))?,
        ),
        n_kmers: u64::from_le_bytes(
            header[24..32]
                .try_into()
                .map_err(|_| format!("{ctx}: malformed bsite n_kmers"))?,
        ),
        bytes_per_col: u64::from_le_bytes(
            header[32..40]
                .try_into()
                .map_err(|_| format!("{ctx}: malformed bsite bytes_per_col"))?,
        ),
        bit_order,
        compression,
    })
}

impl BkmerMmapView {
    pub fn body_prefix_for_rows(&self, n_rows: u64) -> Result<&[u8], String> {
        let body_len = n_rows
            .checked_mul(std::mem::size_of::<u64>() as u64)
            .ok_or_else(|| format!("{}: bkmer payload size overflow", self.label))?;
        let start = usize::try_from(bkmer_body_offset())
            .map_err(|_| format!("{}: bkmer body offset does not fit usize", self.label))?;
        let end =
            start
                .checked_add(usize::try_from(body_len).map_err(|_| {
                    format!("{}: bkmer payload size does not fit usize", self.label)
                })?)
                .ok_or_else(|| format!("{}: bkmer payload end overflow", self.label))?;
        if self.mmap.len() < end {
            return Err(format!("{}: truncated bkmer payload", self.label));
        }
        Ok(&self.mmap[start..end])
    }
}

impl BsiteMmapView {
    pub fn body_prefix_for_rows(&self, n_rows: u64) -> Result<&[u8], String> {
        let start = usize::try_from(bsite_body_offset())
            .map_err(|_| format!("{}: bsite body offset does not fit usize", self.label))?;
        let end = usize::try_from(bsite_column_offset(n_rows, self.header.bytes_per_col))
            .map_err(|_| format!("{}: bsite payload end does not fit usize", self.label))?;
        if end < start {
            return Err(format!("{}: bsite payload end overflow", self.label));
        }
        if self.mmap.len() < end {
            return Err(format!("{}: truncated bsite payload", self.label));
        }
        Ok(&self.mmap[start..end])
    }

    pub fn body(&self) -> Result<&[u8], String> {
        self.body_prefix_for_rows(self.header.n_kmers)
    }
}

pub fn open_bkmer_mmap(path: &Path, ctx: &str) -> Result<BkmerMmapView, String> {
    let file =
        File::open(path).map_err(|e| format!("{ctx}: failed to open {}: {e}", path.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .map_err(|e| format!("{ctx}: failed to mmap {}: {e}", path.display()))?;
    if mmap.len() < BKMER_HEADER_SIZE {
        return Err(format!("{ctx}: truncated bkmer file: {}", path.display()));
    }
    let header = parse_bkmer_header(&mmap[..BKMER_HEADER_SIZE], ctx)?;
    Ok(BkmerMmapView {
        mmap,
        header,
        label: path.display().to_string(),
    })
}

pub fn open_bsite_mmap(path: &Path, ctx: &str) -> Result<BsiteMmapView, String> {
    let file =
        File::open(path).map_err(|e| format!("{ctx}: failed to open {}: {e}", path.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .map_err(|e| format!("{ctx}: failed to mmap {}: {e}", path.display()))?;
    if mmap.len() < BSITE_HEADER_SIZE {
        return Err(format!("{ctx}: truncated bsite file: {}", path.display()));
    }
    let header = parse_bsite_header(&mmap[..BSITE_HEADER_SIZE], ctx)?;
    Ok(BsiteMmapView {
        mmap,
        header,
        label: path.display().to_string(),
    })
}

pub fn typed_body_view<'a>(body: &'a [u8], bytes_per_col: usize) -> TypedBodyView<'a> {
    match bytes_per_col {
        1 => TypedBodyView::U8(body),
        2 => {
            let (prefix, values, suffix) = unsafe { body.align_to::<u16>() };
            if prefix.is_empty() && suffix.is_empty() {
                TypedBodyView::U16(values)
            } else {
                TypedBodyView::Bytes
            }
        }
        4 => {
            let (prefix, values, suffix) = unsafe { body.align_to::<u32>() };
            if prefix.is_empty() && suffix.is_empty() {
                TypedBodyView::U32(values)
            } else {
                TypedBodyView::Bytes
            }
        }
        8 => {
            let (prefix, values, suffix) = unsafe { body.align_to::<u64>() };
            if prefix.is_empty() && suffix.is_empty() {
                TypedBodyView::U64(values)
            } else {
                TypedBodyView::Bytes
            }
        }
        _ => TypedBodyView::Bytes,
    }
}

pub fn load_bin01_as_u64_words(
    path: &str,
    ctx: &str,
) -> Result<(Vec<u64>, usize, usize, usize), String> {
    let file = File::open(path).map_err(|e| format!("{ctx}: failed to open {path}: {e}"))?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| format!("{ctx}: failed to mmap {}: {}", path, e))?;
    let (n_rows, n_samples, row_bytes, data_offset) = parse_bin01_header(&mmap[..], ctx)?;

    let row_words = words_for_samples(n_samples);
    let total_words = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: n_rows * row_words overflow"))?;
    let mut bits_flat = vec![0u64; total_words];

    let mask = tail_mask(n_samples);
    for r in 0..n_rows {
        let src_start = data_offset
            .checked_add(
                r.checked_mul(row_bytes)
                    .ok_or_else(|| format!("{ctx}: row byte offset overflow"))?,
            )
            .ok_or_else(|| format!("{ctx}: row byte offset overflow"))?;
        let src_end = src_start
            .checked_add(row_bytes)
            .ok_or_else(|| format!("{ctx}: row byte end overflow"))?;
        let src = &mmap[src_start..src_end];

        let dst_start = r
            .checked_mul(row_words)
            .ok_or_else(|| format!("{ctx}: row word offset overflow"))?;
        let dst = &mut bits_flat[dst_start..dst_start + row_words];

        for w in 0..row_words {
            let b0 = w * 8;
            if b0 >= row_bytes {
                break;
            }
            let b1 = (b0 + 8).min(row_bytes);
            let mut buf = [0u8; 8];
            let n = b1 - b0;
            buf[..n].copy_from_slice(&src[b0..b1]);
            dst[w] = u64::from_le_bytes(buf);
        }
        apply_tail_mask(dst, mask);
    }

    Ok((bits_flat, row_words, n_rows, n_samples))
}

pub fn load_bin01_selected_rows_as_u64_words(
    path: &str,
    row_indices: &[usize],
    ctx: &str,
) -> Result<(Vec<u64>, usize, usize, usize), String> {
    if row_indices.is_empty() {
        return Err(format!("{ctx}: row_indices is empty"));
    }
    let file = File::open(path).map_err(|e| format!("{ctx}: failed to open {path}: {e}"))?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| format!("{ctx}: failed to mmap {}: {}", path, e))?;
    let (n_rows_all, n_samples, row_bytes, data_offset) = parse_bin01_header(&mmap[..], ctx)?;

    let row_words = words_for_samples(n_samples);
    let n_rows = row_indices.len();
    let total_words = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: n_rows * row_words overflow"))?;
    let mut bits_flat = vec![0u64; total_words];

    let mask = tail_mask(n_samples);
    for (ri, &src_row_idx) in row_indices.iter().enumerate() {
        if src_row_idx >= n_rows_all {
            return Err(format!(
                "{ctx}: row index out of range: {} (n_rows={})",
                src_row_idx, n_rows_all
            ));
        }
        let src_start = data_offset
            .checked_add(
                src_row_idx
                    .checked_mul(row_bytes)
                    .ok_or_else(|| format!("{ctx}: row byte offset overflow"))?,
            )
            .ok_or_else(|| format!("{ctx}: row byte offset overflow"))?;
        let src_end = src_start
            .checked_add(row_bytes)
            .ok_or_else(|| format!("{ctx}: row byte end overflow"))?;
        let src = &mmap[src_start..src_end];

        let dst_start = ri
            .checked_mul(row_words)
            .ok_or_else(|| format!("{ctx}: row word offset overflow"))?;
        let dst = &mut bits_flat[dst_start..dst_start + row_words];

        for w in 0..row_words {
            let b0 = w * 8;
            if b0 >= row_bytes {
                break;
            }
            let b1 = (b0 + 8).min(row_bytes);
            let mut buf = [0u8; 8];
            let n = b1 - b0;
            buf[..n].copy_from_slice(&src[b0..b1]);
            dst[w] = u64::from_le_bytes(buf);
        }
        apply_tail_mask(dst, mask);
    }

    Ok((bits_flat, row_words, n_rows, n_samples))
}

pub fn gather_rows_by_indices(
    bits_flat: &[u64],
    row_words: usize,
    row_indices: &[usize],
    ctx: &str,
) -> Result<Vec<u64>, String> {
    if row_words == 0 {
        return Err(format!("{ctx}: row_words must be > 0"));
    }
    if row_indices.is_empty() {
        return Ok(Vec::new());
    }
    let n_rows_all = bits_flat.len() / row_words;
    let total_words = row_indices
        .len()
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: output size overflow"))?;
    let mut out = vec![0u64; total_words];
    for (dst_r, &src_r) in row_indices.iter().enumerate() {
        if src_r >= n_rows_all {
            return Err(format!(
                "{ctx}: row index out of range: {} (n_rows={})",
                src_r, n_rows_all
            ));
        }
        let src_start = src_r
            .checked_mul(row_words)
            .ok_or_else(|| format!("{ctx}: source offset overflow"))?;
        let dst_start = dst_r
            .checked_mul(row_words)
            .ok_or_else(|| format!("{ctx}: target offset overflow"))?;
        out[dst_start..dst_start + row_words]
            .copy_from_slice(&bits_flat[src_start..src_start + row_words]);
    }
    Ok(out)
}

pub fn gather_rows_by_range(
    bits_flat: &[u64],
    row_words: usize,
    row_start: usize,
    row_end: usize,
    ctx: &str,
) -> Result<Vec<u64>, String> {
    if row_end <= row_start {
        return Ok(Vec::new());
    }
    let n_rows_all = bits_flat.len() / row_words;
    if row_end > n_rows_all {
        return Err(format!(
            "{ctx}: row range out of bounds: [{row_start}, {row_end}) vs n_rows={n_rows_all}"
        ));
    }
    let st = row_start
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: row range start overflow"))?;
    let ed = row_end
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: row range end overflow"))?;
    Ok(bits_flat[st..ed].to_vec())
}

pub fn load_bin01_packed_payload_owned(
    path: &str,
    ctx: &str,
) -> Result<(Vec<u8>, usize, usize, usize), String> {
    let bytes = fs::read(path).map_err(|e| format!("{ctx}: failed to read {path}: {e}"))?;
    let (n_rows, n_samples, row_bytes, data_offset) = parse_bin01_header(&bytes, ctx)?;
    let payload_len = n_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| format!("{ctx}: packed payload size overflow"))?;
    let payload_end = data_offset
        .checked_add(payload_len)
        .ok_or_else(|| format!("{ctx}: packed payload end overflow"))?;
    Ok((
        bytes[data_offset..payload_end].to_vec(),
        n_rows,
        n_samples,
        row_bytes,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        bkmer_body_offset, bsite_body_offset, bsite_column_offset, open_bkmer_mmap,
        parse_bkmer_header, parse_bsite_header, typed_body_view, TypedBodyView,
    };
    use crate::kmer::format::{BkmerHeader, BsiteHeader};
    use std::fs;

    #[test]
    fn kmer_body_offsets_are_stable() {
        assert_eq!(bkmer_body_offset(), 64);
        assert_eq!(bsite_body_offset(), 80);
        assert_eq!(bsite_column_offset(3, 5), 95);
    }

    #[test]
    fn parse_kmer_headers_roundtrip() {
        let mut bkmer = Vec::new();
        BkmerHeader {
            k: 31,
            n_kmers: 12,
            canonical: 1,
        }
        .write_to(&mut bkmer)
        .expect("write bkmer");
        let bk = parse_bkmer_header(&bkmer, "breader::bkmer").expect("parse bkmer");
        assert_eq!(bk.k, 31);
        assert_eq!(bk.n_kmers, 12);
        assert_eq!(bk.canonical, 1);

        let mut bsite = Vec::new();
        BsiteHeader {
            n_samples: 9,
            n_kmers: 12,
            bytes_per_col: 2,
        }
        .write_to(&mut bsite)
        .expect("write bsite");
        let bs = parse_bsite_header(&bsite, "breader::bsite").expect("parse bsite");
        assert_eq!(bs.n_samples, 9);
        assert_eq!(bs.n_kmers, 12);
        assert_eq!(bs.bytes_per_col, 2);
    }

    #[test]
    fn bkmer_mmap_view_reads_payload() {
        let dir = std::env::temp_dir().join(format!(
            "janusx_breader_bkmer_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("t")
        ));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test.bkmer");
        let mut bytes = Vec::new();
        BkmerHeader {
            k: 21,
            n_kmers: 3,
            canonical: 1,
        }
        .write_to(&mut bytes)
        .expect("write bkmer header");
        bytes.extend_from_slice(&11u64.to_le_bytes());
        bytes.extend_from_slice(&22u64.to_le_bytes());
        bytes.extend_from_slice(&33u64.to_le_bytes());
        fs::write(&path, bytes).expect("write bkmer file");

        let view = open_bkmer_mmap(&path, "breader::bkmer_mmap").expect("open bkmer");
        assert_eq!(view.header.k, 21);
        let body = view.body_prefix_for_rows(3).expect("read bkmer body");
        assert_eq!(body.len(), 24);

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn typed_body_view_matches_word_width() {
        let body_u8 = vec![1u8, 2, 3];
        assert!(matches!(typed_body_view(&body_u8, 1), TypedBodyView::U8(_)));

        let body_u16 = vec![1u8, 0, 2, 0];
        assert!(matches!(
            typed_body_view(&body_u16, 2),
            TypedBodyView::U16(_)
        ));

        let body_u32 = vec![1u8, 0, 0, 0, 2, 0, 0, 0];
        assert!(matches!(
            typed_body_view(&body_u32, 4),
            TypedBodyView::U32(_)
        ));

        let body_u64 = vec![1u8, 0, 0, 0, 0, 0, 0, 0];
        assert!(matches!(
            typed_body_view(&body_u64, 8),
            TypedBodyView::U64(_)
        ));
    }
}
