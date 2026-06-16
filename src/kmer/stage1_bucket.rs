use crate::kmer::encode::{bucket_id_from_code, canonical_code};
use crate::kmer::ffi::kmc_reader::KmcReader;
use crate::kmer::format::SampleEntry;
use crate::kmer::progress::KmergeProgressBar;
use crate::kmer::record::{write_rec, KmerPresenceRec};
use anyhow::{Context, Result};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct Stage1Config<'a> {
    pub tmp_dir: &'a Path,
    pub samples: &'a [SampleEntry],
    pub k: u32,
    pub min_count: u32,
    pub batch_size: usize,
    pub bucket_bits: u8,
    pub max_records_per_flush: usize,
    pub threads: usize,
    pub progress_bar: KmergeProgressBar,
}

pub fn run_stage1(config: &Stage1Config<'_>) -> Result<()> {
    let n_buckets = 1usize << config.bucket_bits;
    let run_counters = Arc::new(
        (0..n_buckets)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>(),
    );

    let pool = ThreadPoolBuilder::new()
        .num_threads(config.threads.max(1))
        .build()
        .context("failed to build stage1 thread pool")?;

    pool.install(|| {
        config
            .samples
            .par_iter()
            .try_for_each(|sample| process_sample(sample, config, &run_counters))
    })?;

    write_marker(&config.tmp_dir.join("stage1.done"))?;
    Ok(())
}

fn process_sample(
    sample: &SampleEntry,
    config: &Stage1Config<'_>,
    run_counters: &Arc<Vec<AtomicUsize>>,
) -> Result<()> {
    let mut reader = KmcReader::open(&sample.kmc_prefix, config.k)
        .with_context(|| format!("failed to open sample {}", sample.sample_id))?;
    let n_buckets = 1usize << config.bucket_bits;
    let mut bucket_buffers = (0..n_buckets)
        .map(|_| Vec::new())
        .collect::<Vec<Vec<KmerPresenceRec>>>();
    let mut total_buffered = 0usize;
    let per_bucket_limit = config.max_records_per_flush.max(1);

    let mut kmer_buf = vec![0u64; config.batch_size.max(1)];
    let mut count_buf = vec![0u32; config.batch_size.max(1)];
    let db_is_canonical = reader.kmers_are_canonical();

    loop {
        let n = reader.read_batch(&mut kmer_buf, &mut count_buf)?;
        if n == 0 {
            break;
        }
        config.progress_bar.inc(n as u64);

        if db_is_canonical {
            for idx in 0..n {
                if count_buf[idx] < config.min_count {
                    continue;
                }
                buffer_code(
                    kmer_buf[idx],
                    sample.index,
                    config,
                    &mut bucket_buffers,
                    &mut total_buffered,
                    per_bucket_limit,
                    run_counters,
                )?;
            }
        } else {
            for idx in 0..n {
                if count_buf[idx] < config.min_count {
                    continue;
                }
                buffer_code(
                    canonical_code(kmer_buf[idx], config.k),
                    sample.index,
                    config,
                    &mut bucket_buffers,
                    &mut total_buffered,
                    per_bucket_limit,
                    run_counters,
                )?;
            }
        }

        if total_buffered >= config.max_records_per_flush {
            total_buffered = flush_all(config.tmp_dir, &mut bucket_buffers, run_counters)?;
        }
    }

    let _ = flush_all(config.tmp_dir, &mut bucket_buffers, run_counters)?;
    Ok(())
}

fn buffer_code(
    code: u64,
    sample_index: u32,
    config: &Stage1Config<'_>,
    bucket_buffers: &mut [Vec<KmerPresenceRec>],
    total_buffered: &mut usize,
    per_bucket_limit: usize,
    run_counters: &Arc<Vec<AtomicUsize>>,
) -> Result<()> {
    let bucket_id = bucket_id_from_code(code, config.k, config.bucket_bits);
    bucket_buffers[bucket_id].push(KmerPresenceRec {
        kmer: code,
        sample_id: sample_index,
        pad: 0,
    });
    *total_buffered += 1;

    if bucket_buffers[bucket_id].len() >= per_bucket_limit {
        *total_buffered -= flush_bucket(
            config.tmp_dir,
            bucket_id,
            &mut bucket_buffers[bucket_id],
            run_counters,
        )?;
    }
    Ok(())
}

fn flush_all(
    tmp_dir: &Path,
    bucket_buffers: &mut [Vec<KmerPresenceRec>],
    run_counters: &Arc<Vec<AtomicUsize>>,
) -> Result<usize> {
    let mut remaining = 0usize;
    for (bucket_id, recs) in bucket_buffers.iter_mut().enumerate() {
        if recs.is_empty() {
            continue;
        }
        let _ = flush_bucket(tmp_dir, bucket_id, recs, run_counters)?;
    }
    for recs in bucket_buffers.iter() {
        remaining += recs.len();
    }
    Ok(remaining)
}

fn flush_bucket(
    tmp_dir: &Path,
    bucket_id: usize,
    records: &mut Vec<KmerPresenceRec>,
    run_counters: &Arc<Vec<AtomicUsize>>,
) -> Result<usize> {
    if records.is_empty() {
        return Ok(0);
    }
    records.sort_unstable();

    let bucket_dir = tmp_dir.join(format!("bucket_{bucket_id:04}"));
    fs::create_dir_all(&bucket_dir)
        .with_context(|| format!("failed to create bucket dir: {}", bucket_dir.display()))?;
    let run_id = run_counters[bucket_id].fetch_add(1, Ordering::SeqCst);
    let run_path = bucket_dir.join(format!("run_{run_id:06}.bin"));
    let file = File::create(&run_path)
        .with_context(|| format!("failed to create run file: {}", run_path.display()))?;
    let mut writer = BufWriter::new(file);
    for &rec in records.iter() {
        write_rec(&mut writer, rec)?;
    }
    writer.flush()?;
    let flushed = records.len();
    records.clear();
    Ok(flushed)
}

fn write_marker(path: &PathBuf) -> Result<()> {
    fs::write(path, b"done\n")
        .with_context(|| format!("failed to write stage marker: {}", path.display()))?;
    Ok(())
}
