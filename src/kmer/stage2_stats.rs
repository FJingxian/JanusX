use crate::kmer::progress::ProgressFn;
use crate::kmer::record::{read_rec_opt, KmerPresenceRec};
use anyhow::{Context, Result};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

pub struct Stage2StatsConfig<'a> {
    pub tmp_dir: &'a Path,
    pub n_samples: usize,
    pub bucket_bits: u8,
    pub threads: usize,
    pub collect_patterns: bool,
    pub progress_callback: Option<ProgressFn>,
    pub progress_total: u64,
}

#[derive(Clone, Debug)]
pub struct VennPatternCount {
    pub bit_words: Vec<u64>,
    pub count: u64,
}

#[derive(Clone, Debug)]
pub struct Stage2StatsResult {
    pub pair_counts: Vec<u64>,
    pub venn_patterns: Option<Vec<VennPatternCount>>,
}

pub fn run_stage2_stats(config: &Stage2StatsConfig<'_>) -> Result<Stage2StatsResult> {
    let counts = Arc::new(
        (0..lower_tri_len(config.n_samples)?)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>(),
    );
    let pattern_counts = if config.collect_patterns {
        Some(Arc::new(Mutex::new(HashMap::<Vec<u64>, u64>::new())))
    } else {
        None
    };
    let n_buckets = 1usize << config.bucket_bits;
    let bucket_ids = (0..n_buckets).collect::<Vec<_>>();
    let pool = ThreadPoolBuilder::new()
        .num_threads(config.threads.max(1))
        .build()
        .context("failed to build kstats stage2 thread pool")?;
    let progress_done = Arc::new(AtomicU64::new(0));

    pool.install(|| {
        bucket_ids
            .into_par_iter()
            .map(|bucket_id| {
                let local_patterns = process_bucket(bucket_id, config, counts.as_ref())?;
                if let (Some(global), Some(local)) = (&pattern_counts, local_patterns) {
                    let mut guard = global
                        .lock()
                        .map_err(|_| anyhow::anyhow!("failed to lock venn pattern accumulator"))?;
                    for (bit_words, count) in local {
                        *guard.entry(bit_words).or_insert(0) += count;
                    }
                }
                if let Some(cb) = &config.progress_callback {
                    let done = progress_done.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                    cb(done.min(config.progress_total), config.progress_total)?;
                }
                Ok(())
            })
            .collect::<Result<Vec<_>>>()
    })?;

    let pair_counts = counts
        .iter()
        .map(|x| x.load(AtomicOrdering::Relaxed))
        .collect::<Vec<_>>();
    let venn_patterns = if let Some(global) = pattern_counts {
        let guard = global
            .lock()
            .map_err(|_| anyhow::anyhow!("failed to finalize venn pattern accumulator"))?;
        let mut rows = guard
            .iter()
            .map(|(bit_words, count)| VennPatternCount {
                bit_words: bit_words.clone(),
                count: *count,
            })
            .collect::<Vec<_>>();
        rows.sort_by(|a, b| {
            let a_degree = bit_count(&a.bit_words);
            let b_degree = bit_count(&b.bit_words);
            b.count
                .cmp(&a.count)
                .then_with(|| b_degree.cmp(&a_degree))
                .then_with(|| a.bit_words.cmp(&b.bit_words))
        });
        Some(rows)
    } else {
        None
    };
    Ok(Stage2StatsResult {
        pair_counts,
        venn_patterns,
    })
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

fn process_bucket(
    bucket_id: usize,
    config: &Stage2StatsConfig<'_>,
    counts: &[AtomicU64],
) -> Result<Option<HashMap<Vec<u64>, u64>>> {
    let bucket_dir = config.tmp_dir.join(format!("bucket_{bucket_id:04}"));
    let run_paths = list_run_files(&bucket_dir)?;
    if run_paths.is_empty() {
        return Ok(config.collect_patterns.then(HashMap::new));
    }

    let mut runs = Vec::with_capacity(run_paths.len());
    let mut heap = BinaryHeap::new();
    let mut pattern_counts = config.collect_patterns.then(HashMap::new);
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

    let mut present_samples = Vec::<u32>::new();
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

        for &sample_id in &present_samples {
            counts[tri_index(sample_id as usize, sample_id as usize)]
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
        for hi_pos in 1..present_samples.len() {
            let hi = present_samples[hi_pos] as usize;
            for &lo_u32 in &present_samples[..hi_pos] {
                let lo = lo_u32 as usize;
                counts[tri_index(hi, lo)].fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
        if let Some(local) = pattern_counts.as_mut() {
            let bit_words = build_presence_words(config.n_samples, &present_samples);
            *local.entry(bit_words).or_insert(0) += 1;
        }
    }
    Ok(pattern_counts)
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

fn lower_tri_len(n: usize) -> Result<usize> {
    n.checked_mul(n + 1)
        .and_then(|x| x.checked_div(2))
        .ok_or_else(|| anyhow::anyhow!("pairwise matrix size overflow"))
}

fn tri_index(i: usize, j: usize) -> usize {
    let (hi, lo) = if i >= j { (i, j) } else { (j, i) };
    hi * (hi + 1) / 2 + lo
}

fn build_presence_words(n_samples: usize, present_samples: &[u32]) -> Vec<u64> {
    let mut words = vec![0u64; n_samples.div_ceil(64)];
    for &sample_id in present_samples {
        let idx = sample_id as usize;
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        words[word_idx] |= 1u64 << bit_idx;
    }
    words
}

fn bit_count(words: &[u64]) -> u32 {
    words.iter().map(|word| word.count_ones()).sum::<u32>()
}
