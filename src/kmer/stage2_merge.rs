use crate::kmer::format::BucketPartSummary;
use crate::kmer::progress::KmergeProgressBar;
use crate::kmer::record::{read_rec_opt, KmerPresenceRec};
use crate::kmer::writer::{bitset_column, bytes_per_col};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

pub struct Stage2Config<'a> {
    pub tmp_dir: &'a Path,
    pub n_samples: usize,
    pub freq: f64,
    pub bucket_bits: u8,
    pub threads: usize,
    pub progress_bar: KmergeProgressBar,
}

pub fn run_stage2(config: &Stage2Config<'_>) -> Result<Vec<BucketPartSummary>> {
    if !(0.0..=0.5).contains(&config.freq) {
        bail!("freq must be within [0, 0.5]");
    }

    let parts_dir = config.tmp_dir.join("parts");
    fs::create_dir_all(&parts_dir)
        .with_context(|| format!("failed to create parts dir: {}", parts_dir.display()))?;

    let n_buckets = 1usize << config.bucket_bits;
    let bucket_ids = (0..n_buckets).collect::<Vec<_>>();
    let pool = ThreadPoolBuilder::new()
        .num_threads(config.threads.max(1))
        .build()
        .context("failed to build stage2 thread pool")?;

    let mut parts = pool.install(|| {
        bucket_ids
            .into_par_iter()
            .map(|bucket_id| {
                let part = merge_bucket(bucket_id, &parts_dir, config)?;
                config.progress_bar.inc(1);
                Ok(part)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    parts.retain(|part| part.n_kmers > 0);
    parts.sort_by_key(|part| part.bucket_id);

    let manifest_path = config.tmp_dir.join("stage2.parts.json");
    let manifest_file = File::create(&manifest_path).with_context(|| {
        format!(
            "failed to create manifest file: {}",
            manifest_path.display()
        )
    })?;
    serde_json::to_writer_pretty(manifest_file, &parts)
        .with_context(|| format!("failed to write manifest file: {}", manifest_path.display()))?;
    fs::write(config.tmp_dir.join("stage2.done"), b"done\n")
        .context("failed to write stage2 marker")?;
    Ok(parts)
}

pub fn load_stage2_manifest(tmp_dir: &Path) -> Result<Vec<BucketPartSummary>> {
    let manifest_path = tmp_dir.join("stage2.parts.json");
    let file = File::open(&manifest_path).with_context(|| {
        format!(
            "failed to open stage2 manifest: {}",
            manifest_path.display()
        )
    })?;
    let parts = serde_json::from_reader(file).with_context(|| {
        format!(
            "failed to parse stage2 manifest: {}",
            manifest_path.display()
        )
    })?;
    Ok(parts)
}

#[derive(Debug)]
struct RunReader {
    reader: BufReader<File>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HeapItem {
    rec: KmerPresenceRec,
    run_idx: usize,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rec
            .cmp(&self.rec)
            .then_with(|| other.run_idx.cmp(&self.run_idx))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn merge_bucket(
    bucket_id: usize,
    parts_dir: &Path,
    config: &Stage2Config<'_>,
) -> Result<BucketPartSummary> {
    let bucket_dir = config.tmp_dir.join(format!("bucket_{bucket_id:04}"));
    let run_paths = list_run_files(&bucket_dir)?;
    let bkmer_part = parts_dir.join(format!("bucket_{bucket_id:04}.bkmer.part"));
    let bsite_part = parts_dir.join(format!("bucket_{bucket_id:04}.bsite.part"));

    if run_paths.is_empty() {
        return Ok(BucketPartSummary {
            bucket_id,
            n_kmers: 0,
            bkmer_part: bkmer_part.to_string_lossy().into_owned(),
            bsite_part: bsite_part.to_string_lossy().into_owned(),
        });
    }

    let mut runs = Vec::with_capacity(run_paths.len());
    let mut heap = BinaryHeap::new();
    for (run_idx, path) in run_paths.iter().enumerate() {
        let file = File::open(path)
            .with_context(|| format!("failed to open run file: {}", path.display()))?;
        let mut run = RunReader {
            reader: BufReader::new(file),
        };
        if let Some(rec) = read_rec_opt(&mut run.reader)? {
            heap.push(HeapItem { rec, run_idx });
        }
        runs.push(run);
    }

    let mut bkmer_writer = BufWriter::new(
        File::create(&bkmer_part)
            .with_context(|| format!("failed to create part file: {}", bkmer_part.display()))?,
    );
    let mut bsite_writer = BufWriter::new(
        File::create(&bsite_part)
            .with_context(|| format!("failed to create part file: {}", bsite_part.display()))?,
    );
    let col_nbytes = bytes_per_col(config.n_samples);
    let mut present_samples = Vec::<u32>::new();
    let mut written = 0u64;

    while let Some(item) = heap.pop() {
        let current_kmer = item.rec.kmer;
        present_samples.clear();
        let mut last_sample = None::<u32>;
        consume_item(
            item,
            &mut runs,
            &mut heap,
            &mut present_samples,
            &mut last_sample,
        )?;

        while let Some(next) = heap.peek().copied() {
            if next.rec.kmer != current_kmer {
                break;
            }
            let next = heap.pop().expect("heap item should exist");
            consume_item(
                next,
                &mut runs,
                &mut heap,
                &mut present_samples,
                &mut last_sample,
            )?;
        }

        let presence = present_samples.len() as u32;
        let rate = presence as f64 / config.n_samples as f64;
        if rate < config.freq || rate > 1.0 - config.freq {
            continue;
        }

        bkmer_writer.write_all(&current_kmer.to_le_bytes())?;
        let col = bitset_column(config.n_samples, &present_samples);
        debug_assert_eq!(col.len(), col_nbytes);
        bsite_writer.write_all(&col)?;
        written += 1;
    }

    bkmer_writer.flush()?;
    bsite_writer.flush()?;

    Ok(BucketPartSummary {
        bucket_id,
        n_kmers: written,
        bkmer_part: bkmer_part.to_string_lossy().into_owned(),
        bsite_part: bsite_part.to_string_lossy().into_owned(),
    })
}

fn consume_item(
    item: HeapItem,
    runs: &mut [RunReader],
    heap: &mut BinaryHeap<HeapItem>,
    present_samples: &mut Vec<u32>,
    last_sample: &mut Option<u32>,
) -> Result<()> {
    if *last_sample != Some(item.rec.sample_id) {
        present_samples.push(item.rec.sample_id);
        *last_sample = Some(item.rec.sample_id);
    }
    if let Some(next_rec) = read_rec_opt(&mut runs[item.run_idx].reader)? {
        heap.push(HeapItem {
            rec: next_rec,
            run_idx: item.run_idx,
        });
    }
    Ok(())
}

fn list_run_files(bucket_dir: &Path) -> Result<Vec<PathBuf>> {
    if !bucket_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths = fs::read_dir(bucket_dir)
        .with_context(|| format!("failed to read bucket dir: {}", bucket_dir.display()))?
        .filter_map(|entry| entry.ok().map(|x| x.path()))
        .filter(|path| path.extension().and_then(|x| x.to_str()) == Some("bin"))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}
