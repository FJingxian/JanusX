use serde::{Deserialize, Serialize};
use std::io::{self, Write};

pub const BKMER_MAGIC: [u8; 8] = *b"JXBKMR1\0";
pub const BSITE_MAGIC: [u8; 8] = *b"JXBSIT1\0";
pub const BKMER_VERSION: u32 = 1;
pub const BSITE_VERSION: u32 = 1;
pub const BKMER_HEADER_SIZE: usize = 64;
pub const BSITE_HEADER_SIZE: usize = 80;
pub const ENCODING_ACGT_2BIT: u32 = 0;
pub const MATRIX_LAYOUT_COLUMN_MAJOR_BITSET: u32 = 1;
pub const BIT_ORDER_LITTLE: u32 = 0;
pub const COMPRESSION_NONE: u32 = 0;

#[derive(Clone, Debug)]
pub struct SampleEntry {
    pub index: u32,
    pub sample_id: String,
    pub kmc_prefix: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BucketPartSummary {
    pub bucket_id: usize,
    pub n_kmers: u64,
    pub bkmer_part: String,
    pub bsite_part: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KmergeMeta {
    pub format: String,
    pub k: u32,
    pub n_samples: u64,
    pub n_kmers: u64,
    pub bytes_per_col: u64,
    pub encoding: String,
    pub canonical: bool,
    pub matrix_layout: String,
    pub value_type: String,
    pub bit_order: String,
    pub bkmer_file: String,
    pub bsite_file: String,
    pub idv_file: String,
    pub min_count: u32,
    pub min_presence: u32,
    pub min_presence_rate: f64,
    pub max_presence_rate: f64,
    pub bucket_bits: u8,
    pub compression: String,
}

#[derive(Clone, Debug)]
pub struct BkmerHeader {
    pub k: u32,
    pub n_kmers: u64,
    pub canonical: u32,
}

impl BkmerHeader {
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&BKMER_MAGIC)?;
        writer.write_all(&BKMER_VERSION.to_le_bytes())?;
        writer.write_all(&self.k.to_le_bytes())?;
        writer.write_all(&self.n_kmers.to_le_bytes())?;
        writer.write_all(&ENCODING_ACGT_2BIT.to_le_bytes())?;
        writer.write_all(&self.canonical.to_le_bytes())?;
        writer.write_all(&[0u8; 32])?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct BsiteHeader {
    pub n_samples: u64,
    pub n_kmers: u64,
    pub bytes_per_col: u64,
}

impl BsiteHeader {
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&BSITE_MAGIC)?;
        writer.write_all(&BSITE_VERSION.to_le_bytes())?;
        writer.write_all(&MATRIX_LAYOUT_COLUMN_MAJOR_BITSET.to_le_bytes())?;
        writer.write_all(&self.n_samples.to_le_bytes())?;
        writer.write_all(&self.n_kmers.to_le_bytes())?;
        writer.write_all(&self.bytes_per_col.to_le_bytes())?;
        writer.write_all(&BIT_ORDER_LITTLE.to_le_bytes())?;
        writer.write_all(&COMPRESSION_NONE.to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?;
        writer.write_all(&[0u8; 28])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{BkmerHeader, BsiteHeader, BKMER_HEADER_SIZE, BSITE_HEADER_SIZE};

    #[test]
    fn header_sizes_match_spec() {
        let mut bkmer = Vec::new();
        BkmerHeader {
            k: 31,
            n_kmers: 10,
            canonical: 1,
        }
        .write_to(&mut bkmer)
        .expect("write");
        assert_eq!(bkmer.len(), BKMER_HEADER_SIZE);

        let mut bsite = Vec::new();
        BsiteHeader {
            n_samples: 8,
            n_kmers: 10,
            bytes_per_col: 1,
        }
        .write_to(&mut bsite)
        .expect("write");
        assert_eq!(bsite.len(), BSITE_HEADER_SIZE);
    }
}
