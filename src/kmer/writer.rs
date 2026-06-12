use crate::kmer::format::{KmergeMeta, SampleEntry};
use anyhow::{Context, Result};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub fn bytes_per_col(n_samples: usize) -> usize {
    n_samples.div_ceil(8)
}

pub fn bitset_column(n_samples: usize, present_sample_ids: &[u32]) -> Vec<u8> {
    let mut buf = vec![0u8; bytes_per_col(n_samples)];
    for &sample_id in present_sample_ids {
        let idx = sample_id as usize;
        let byte = idx >> 3;
        let bit = idx & 7;
        if let Some(slot) = buf.get_mut(byte) {
            *slot |= 1u8 << bit;
        }
    }
    buf
}

pub fn write_idv_file(path: &Path, samples: &[SampleEntry]) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create idv file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(b"#idx\tsample_id\tkmc_prefix\n")?;
    for sample in samples {
        writeln!(
            writer,
            "{}\t{}\t{}",
            sample.index, sample.sample_id, sample.kmc_prefix
        )?;
    }
    writer.flush()?;
    Ok(())
}

pub fn append_path_to_writer<W: Write>(writer: &mut W, path: &Path) -> Result<u64> {
    let mut reader = File::open(path)
        .with_context(|| format!("failed to open part file: {}", path.display()))?;
    let copied = io::copy(&mut reader, writer)
        .with_context(|| format!("failed appending part file: {}", path.display()))?;
    Ok(copied)
}

pub fn write_meta_json(path: &Path, meta: &KmergeMeta) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create meta file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, meta)
        .with_context(|| format!("failed to serialize meta json: {}", path.display()))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
