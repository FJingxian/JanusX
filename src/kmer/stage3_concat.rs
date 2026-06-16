use crate::kmer::encode::ENCODING_NAME;
use crate::kmer::format::{BkmerHeader, BsiteHeader, BucketPartSummary, KmergeMeta, SampleEntry};
use crate::kmer::writer::{append_path_to_writer, bytes_per_col, write_idv_file, write_meta_json};
use anyhow::{bail, Context, Result};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub struct Stage3Config<'a> {
    pub out_dir: &'a Path,
    pub prefix: &'a str,
    pub tmp_dir: &'a Path,
    pub samples: &'a [SampleEntry],
    pub parts: &'a [BucketPartSummary],
    pub k: u32,
    pub min_count: u32,
    pub freq: f64,
    pub bucket_bits: u8,
}

#[derive(Clone, Debug)]
pub struct Stage3Summary {
    pub idv_path: PathBuf,
    pub bkmer_path: PathBuf,
    pub bsite_path: PathBuf,
    pub meta_path: PathBuf,
    pub n_samples: u64,
    pub n_kmers: u64,
    pub bytes_per_col: u64,
    pub k: u32,
}

pub fn run_stage3(config: &Stage3Config<'_>) -> Result<Stage3Summary> {
    let total_kmers = config.parts.iter().map(|part| part.n_kmers).sum::<u64>();
    if total_kmers == 0 {
        bail!("no k-mers survived filters; final matrix would be empty");
    }

    fs::create_dir_all(config.out_dir)
        .with_context(|| format!("failed to create output dir: {}", config.out_dir.display()))?;

    let idv_path = config.out_dir.join(format!("{}.idv", config.prefix));
    let bkmer_path = config.out_dir.join(format!("{}.bkmer", config.prefix));
    let bsite_path = config.out_dir.join(format!("{}.bsite", config.prefix));
    let meta_path = config.out_dir.join(format!("{}.meta.json", config.prefix));

    write_idv_file(&idv_path, config.samples)?;

    let n_samples = config.samples.len() as u64;
    let bytes_per_col_u64 = bytes_per_col(config.samples.len()) as u64;

    let bkmer_file = File::create(&bkmer_path)
        .with_context(|| format!("failed to create bkmer file: {}", bkmer_path.display()))?;
    let mut bkmer_writer = BufWriter::new(bkmer_file);
    BkmerHeader {
        k: config.k,
        n_kmers: total_kmers,
        canonical: 1,
    }
    .write_to(&mut bkmer_writer)?;
    for part in config.parts.iter() {
        append_path_to_writer(&mut bkmer_writer, Path::new(&part.bkmer_part))?;
    }
    bkmer_writer.flush()?;

    let bsite_file = File::create(&bsite_path)
        .with_context(|| format!("failed to create bsite file: {}", bsite_path.display()))?;
    let mut bsite_writer = BufWriter::new(bsite_file);
    BsiteHeader {
        n_samples,
        n_kmers: total_kmers,
        bytes_per_col: bytes_per_col_u64,
    }
    .write_to(&mut bsite_writer)?;
    for part in config.parts.iter() {
        append_path_to_writer(&mut bsite_writer, Path::new(&part.bsite_part))?;
    }
    bsite_writer.flush()?;

    let meta = KmergeMeta {
        format: "janusx-kmer-bitmatrix-v1".to_string(),
        k: config.k,
        n_samples,
        n_kmers: total_kmers,
        bytes_per_col: bytes_per_col_u64,
        encoding: ENCODING_NAME.to_string(),
        canonical: true,
        matrix_layout: "column_major_bitset".to_string(),
        value_type: "binary_presence".to_string(),
        bit_order: "little_bit_order".to_string(),
        bkmer_file: bkmer_path
            .file_name()
            .and_then(|x| x.to_str())
            .unwrap_or_default()
            .to_string(),
        bsite_file: bsite_path
            .file_name()
            .and_then(|x| x.to_str())
            .unwrap_or_default()
            .to_string(),
        idv_file: idv_path
            .file_name()
            .and_then(|x| x.to_str())
            .unwrap_or_default()
            .to_string(),
        min_count: config.min_count,
        min_presence_rate: config.freq,
        max_presence_rate: 1.0 - config.freq,
        bucket_bits: config.bucket_bits,
        compression: "none".to_string(),
    };
    write_meta_json(&meta_path, &meta)?;
    fs::write(config.tmp_dir.join("stage3.done"), b"done\n")
        .context("failed to write stage3 marker")?;

    Ok(Stage3Summary {
        idv_path,
        bkmer_path,
        bsite_path,
        meta_path,
        n_samples,
        n_kmers: total_kmers,
        bytes_per_col: bytes_per_col_u64,
        k: config.k,
    })
}
