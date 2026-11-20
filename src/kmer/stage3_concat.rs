use crate::kmer::encode::ENCODING_NAME;
use crate::kmer::format::{BkmerHeader, BsiteHeader, BucketPartSummary, KmergeMeta, SampleEntry};
use crate::kmer::progress::ProgressFn;
use crate::kmer::writer::{append_path_to_writer, bytes_per_col, write_idv_file, write_meta_json};
use crate::pgen_writer::PgenWriter;
use anyhow::{bail, Context, Result};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
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
    pub out_format: &'a str,
    pub progress_callback: Option<ProgressFn>,
    pub progress_total: u64,
}

#[derive(Clone, Debug)]
pub struct Stage3Summary {
    pub output_format: String,
    pub idv_path: Option<PathBuf>,
    pub bkmer_path: Option<PathBuf>,
    pub bsite_path: Option<PathBuf>,
    pub meta_path: Option<PathBuf>,
    pub pgen_path: Option<PathBuf>,
    pub pvar_path: Option<PathBuf>,
    pub psam_path: Option<PathBuf>,
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

    match config.out_format.trim().to_ascii_lowercase().as_str() {
        "kfile" => write_kfile_stage3(config, total_kmers),
        "pfile" => write_pfile_stage3(config, total_kmers),
        other => bail!("unsupported kmerge output format: {other}"),
    }
}

fn write_kfile_stage3(config: &Stage3Config<'_>, total_kmers: u64) -> Result<Stage3Summary> {
    let idv_path = config.out_dir.join(format!("{}.idv", config.prefix));
    let bkmer_path = config.out_dir.join(format!("{}.bkmer", config.prefix));
    let bsite_path = config.out_dir.join(format!("{}.bsite", config.prefix));
    let meta_path = config.out_dir.join(format!("{}.meta.json", config.prefix));
    let mut progress_done = 0u64;

    write_idv_file(&idv_path, config.samples)?;
    emit_progress_tick(config, &mut progress_done, 1)?;

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
        emit_progress_tick(config, &mut progress_done, 1)?;
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
        emit_progress_tick(config, &mut progress_done, 1)?;
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
    emit_progress_tick(config, &mut progress_done, 1)?;
    fs::write(config.tmp_dir.join("stage3.done"), b"done\n")
        .context("failed to write stage3 marker")?;

    Ok(Stage3Summary {
        output_format: "kfile".to_string(),
        idv_path: Some(idv_path),
        bkmer_path: Some(bkmer_path),
        bsite_path: Some(bsite_path),
        meta_path: Some(meta_path),
        pgen_path: None,
        pvar_path: None,
        psam_path: None,
        n_samples,
        n_kmers: total_kmers,
        bytes_per_col: bytes_per_col_u64,
        k: config.k,
    })
}

fn write_pfile_stage3(config: &Stage3Config<'_>, total_kmers: u64) -> Result<Stage3Summary> {
    let prefix = config.out_dir.join(config.prefix);
    let sample_ids = config
        .samples
        .iter()
        .map(|sample| sample.sample_id.clone())
        .collect::<Vec<_>>();
    let mut writer = PgenWriter::create_with_pvar_compression(&prefix, &sample_ids, true)
        .map_err(anyhow::Error::msg)?;
    let bytes_per_col = bytes_per_col(config.samples.len());
    let mut progress_done = 0u64;
    let mut row_idx = 0u64;
    let mut id_buf = Vec::with_capacity(40);

    for part in config.parts.iter() {
        let mut bkmer = BufReader::new(
            File::open(&part.bkmer_part)
                .with_context(|| format!("failed to open {}", part.bkmer_part))?,
        );
        let mut bsite = BufReader::new(
            File::open(&part.bsite_part)
                .with_context(|| format!("failed to open {}", part.bsite_part))?,
        );
        let mut code_bytes = [0u8; 8];
        let mut bitset = vec![0u8; bytes_per_col];
        for _ in 0..part.n_kmers {
            bkmer.read_exact(&mut code_bytes)?;
            bsite.read_exact(&mut bitset)?;
            let code = u64::from_le_bytes(code_bytes);
            append_compact_kmer_id(&mut id_buf, config.k, code);
            writer
                .write_presence_row("0", row_idx + 1, &id_buf, &bitset)
                .map_err(anyhow::Error::msg)?;
            row_idx += 1;
        }
        emit_progress_tick(config, &mut progress_done, 1)?;
    }
    writer.finish().map_err(anyhow::Error::msg)?;
    emit_progress_tick(config, &mut progress_done, 1)?;
    if row_idx != total_kmers {
        bail!(
            "pfile row count mismatch: wrote {}, expected {}",
            row_idx,
            total_kmers
        );
    }
    fs::write(config.tmp_dir.join("stage3.done"), b"done\n")
        .context("failed to write stage3 marker")?;
    let pgen_path = config.out_dir.join(format!("{}.pgen", config.prefix));
    let pvar_path = config.out_dir.join(format!("{}.pvar.zst", config.prefix));
    let psam_path = config.out_dir.join(format!("{}.psam", config.prefix));
    Ok(Stage3Summary {
        output_format: "pfile".to_string(),
        idv_path: None,
        bkmer_path: None,
        bsite_path: None,
        meta_path: None,
        pgen_path: Some(pgen_path),
        pvar_path: Some(pvar_path),
        psam_path: Some(psam_path),
        n_samples: config.samples.len() as u64,
        n_kmers: total_kmers,
        bytes_per_col: bytes_per_col as u64,
        k: config.k,
    })
}

fn append_compact_kmer_id(out: &mut Vec<u8>, k: u32, code: u64) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.clear();
    out.extend_from_slice(b"k");
    if k >= 10 {
        out.push(b'0' + (k / 10) as u8);
    }
    out.push(b'0' + (k % 10) as u8);
    out.push(b'h');
    let width = (2 * k as usize).div_ceil(4);
    for idx in (0..width).rev() {
        out.push(HEX[((code >> (idx * 4)) & 0x0f) as usize]);
    }
}

fn emit_progress_tick(config: &Stage3Config<'_>, done: &mut u64, step: u64) -> Result<()> {
    *done = done.saturating_add(step).min(config.progress_total);
    if let Some(cb) = &config.progress_callback {
        cb(*done, config.progress_total)?;
    }
    Ok(())
}
