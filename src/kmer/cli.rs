use crate::kmer::format::{KmergeMeta, SampleEntry};
use crate::kmer::inputs::{inspect_kmc_inputs, resolve_samples};
use crate::kmer::progress::ProgressFn;
use crate::kmer::stage1_bucket::{run_stage1, Stage1Config};
use crate::kmer::stage2_merge::{load_stage2_manifest, run_stage2, Stage2Config};
use crate::kmer::stage3_concat::{run_stage3, Stage3Config, Stage3Summary};
use anyhow::{bail, Context, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug)]
pub struct KmergeArgs {
    pub db_inputs: Vec<String>,
    pub sample_ids: Option<Vec<String>>,
    pub out_dir: PathBuf,
    pub prefix: String,
    pub threads: usize,
    pub memory_mb: usize,
    pub freq: f64,
    pub tmp_dir: Option<PathBuf>,
    pub max_run_size_mb: usize,
    pub bucket_bits: u8,
    pub batch_size: usize,
    pub min_count: u32,
    pub resume: bool,
    pub keep_tmp: bool,
    pub force: bool,
    pub log_path: Option<PathBuf>,
    pub progress_callback: Option<Arc<Py<PyAny>>>,
}

#[pyfunction(
    name = "kmerge_run",
    signature = (
        db_inputs,
        sample_ids = None,
        out = ".",
        prefix = "kmerge",
        thread = 8,
        memory = 2048,
        freq = 0.02,
        tmp_dir = None,
        max_run_size = 2048,
        bucket_bits = 10,
        batch_size = 1_048_576,
        min_count = 1,
        resume = false,
        keep_tmp = false,
        force = false,
        log_path = None,
        progress_callback = None
    )
)]
pub fn kmerge_run_py(
    py: Python<'_>,
    db_inputs: Vec<String>,
    sample_ids: Option<Vec<String>>,
    out: &str,
    prefix: &str,
    thread: usize,
    memory: usize,
    freq: f64,
    tmp_dir: Option<String>,
    max_run_size: usize,
    bucket_bits: u8,
    batch_size: usize,
    min_count: u32,
    resume: bool,
    keep_tmp: bool,
    force: bool,
    log_path: Option<String>,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<Py<PyDict>> {
    let args = KmergeArgs {
        db_inputs,
        sample_ids,
        out_dir: PathBuf::from(out),
        prefix: prefix.to_string(),
        threads: thread.max(1),
        memory_mb: memory.max(1),
        freq,
        tmp_dir: tmp_dir.map(PathBuf::from),
        max_run_size_mb: max_run_size.max(1),
        bucket_bits,
        batch_size: batch_size.max(1),
        min_count: min_count.max(1),
        resume,
        keep_tmp,
        force,
        log_path: log_path.map(PathBuf::from),
        progress_callback: progress_callback.map(Arc::new),
    };

    let summary = py
        .detach(|| run_kmerge(args))
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    let out = PyDict::new(py);
    out.set_item("idv", summary.idv_path.to_string_lossy().as_ref())?;
    out.set_item("bkmer", summary.bkmer_path.to_string_lossy().as_ref())?;
    out.set_item("bsite", summary.bsite_path.to_string_lossy().as_ref())?;
    out.set_item("meta", summary.meta_path.to_string_lossy().as_ref())?;
    out.set_item("n_samples", summary.n_samples)?;
    out.set_item("n_kmers", summary.n_kmers)?;
    out.set_item("bytes_per_col", summary.bytes_per_col)?;
    out.set_item("k", summary.k)?;
    Ok(out.unbind())
}

pub fn run_kmerge(args: KmergeArgs) -> Result<Stage3Summary> {
    if args.db_inputs.is_empty() {
        bail!("no input KMC database provided");
    }
    if args.prefix.trim().is_empty() {
        bail!("prefix cannot be empty");
    }
    if !(0.0..=0.5).contains(&args.freq) {
        bail!("freq must be within [0, 0.5]");
    }
    if args.bucket_bits == 0 {
        bail!("bucket_bits must be > 0");
    }

    let out_dir = args
        .out_dir
        .expanduser()
        .context("failed to normalize output dir")?;
    let tmp_dir = args
        .tmp_dir
        .clone()
        .unwrap_or_else(|| out_dir.join(format!("{}.tmp", args.prefix)));
    let samples = resolve_samples(&args.db_inputs, args.sample_ids.as_ref())?;
    if samples.len() == 1 && args.freq > 0.0 {
        bail!(
            "single-sample kmerge with freq > 0 filters out all present k-mers; use freq=0.0 or provide multiple samples"
        );
    }
    let inspect = inspect_kmc_inputs(&samples)?;
    let k = inspect.k;
    let log_path = args.log_path.as_deref();
    let progress_callback = args.progress_callback.as_deref();
    let stage1_progress = build_progress_callback(args.progress_callback.as_ref(), 1);
    let stage2_progress = build_progress_callback(args.progress_callback.as_ref(), 2);
    let stage3_progress = build_progress_callback(args.progress_callback.as_ref(), 3);

    prepare_output_space(&out_dir, &tmp_dir, &args, &samples)?;

    if args.resume
        && is_stage_done(&tmp_dir.join("stage3.done"))
        && out_dir.join(format!("{}.meta.json", args.prefix)).is_file()
    {
        emit_run_line(
            log_path,
            "Stage 3/3: resume hit, reusing existing final outputs",
        )?;
        return load_existing_summary(&out_dir, &args.prefix);
    }

    let max_records_per_flush =
        max_records_per_flush(args.memory_mb, args.threads, args.max_run_size_mb)?;

    emit_run_line(
        log_path,
        &format!(
            "Rust kmerge: samples={}, k={}, threads={}, memory={}MB, bucket_bits={}, batch_size={}",
            samples.len(),
            k,
            args.threads,
            args.memory_mb,
            args.bucket_bits,
            args.batch_size
        ),
    )?;

    if !(args.resume && is_stage_done(&tmp_dir.join("stage1.done"))) {
        emit_run_line(log_path, "Stage 1/3: KMC stream -> bucketed sorted runs")?;
        emit_progress_callback(
            progress_callback,
            1,
            0,
            inspect.total_records.max(1) as usize,
        )?;
        run_stage1(&Stage1Config {
            tmp_dir: &tmp_dir,
            samples: &samples,
            k,
            min_count: args.min_count,
            batch_size: args.batch_size,
            bucket_bits: args.bucket_bits,
            max_records_per_flush,
            threads: args.threads,
            progress_callback: stage1_progress.clone(),
            progress_total: inspect.total_records.max(1),
        })?;
        emit_progress_callback(
            progress_callback,
            1,
            inspect.total_records.max(1) as usize,
            inspect.total_records.max(1) as usize,
        )?;
    } else {
        emit_run_line(
            log_path,
            "Stage 1/3: resume hit, reusing existing sorted runs",
        )?;
    }

    let parts = if args.resume && is_stage_done(&tmp_dir.join("stage2.done")) {
        emit_run_line(
            log_path,
            "Stage 2/3: resume hit, reusing existing bucket parts",
        )?;
        load_stage2_manifest(&tmp_dir)?
    } else {
        emit_run_line(
            log_path,
            "Stage 2/3: bucket merge -> .bkmer.part / .bsite.part",
        )?;
        let stage2_total = 1u64 << args.bucket_bits;
        emit_progress_callback(progress_callback, 2, 0, stage2_total as usize)?;
        let parts = run_stage2(&Stage2Config {
            tmp_dir: &tmp_dir,
            n_samples: samples.len(),
            freq: args.freq,
            bucket_bits: args.bucket_bits,
            threads: args.threads,
            progress_callback: stage2_progress.clone(),
            progress_total: stage2_total.max(1),
        })?;
        emit_progress_callback(
            progress_callback,
            2,
            stage2_total as usize,
            stage2_total as usize,
        )?;
        parts
    };

    if parts.iter().map(|part| part.n_kmers).sum::<u64>() == 0 {
        bail!("no k-mers survived filters across all buckets");
    }

    emit_run_line(log_path, "Stage 3/3: concat parts -> final matrix")?;
    let stage3_total = (parts.len() as u64).saturating_mul(2).saturating_add(2);
    emit_progress_callback(progress_callback, 3, 0, stage3_total as usize)?;
    let summary = run_stage3(&Stage3Config {
        out_dir: &out_dir,
        prefix: &args.prefix,
        tmp_dir: &tmp_dir,
        samples: &samples,
        parts: &parts,
        k,
        min_count: args.min_count,
        freq: args.freq,
        bucket_bits: args.bucket_bits,
        progress_callback: stage3_progress.clone(),
        progress_total: stage3_total.max(1),
    })?;
    emit_progress_callback(
        progress_callback,
        3,
        stage3_total as usize,
        stage3_total as usize,
    )?;

    if !args.keep_tmp {
        fs::remove_dir_all(&tmp_dir)
            .with_context(|| format!("failed to remove tmp dir: {}", tmp_dir.display()))?;
    }

    Ok(summary)
}

fn emit_run_line(log_path: Option<&Path>, line: &str) -> Result<()> {
    println!("{line}");
    if let Some(path) = log_path {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("failed to open kmerge log file: {}", path.display()))?;
        writeln!(file, "{line}")
            .with_context(|| format!("failed to write kmerge log file: {}", path.display()))?;
    }
    Ok(())
}

fn build_progress_callback(cb: Option<&Arc<Py<PyAny>>>, stage: usize) -> Option<ProgressFn> {
    let cb = Arc::clone(cb?);
    Some(Arc::new(move |done: u64, total: u64| -> Result<()> {
        emit_progress_callback(Some(cb.as_ref()), stage, done as usize, total as usize)
    }))
}

fn emit_progress_callback(
    cb: Option<&Py<PyAny>>,
    stage: usize,
    done: usize,
    total: usize,
) -> Result<()> {
    let total_use = total.max(1);
    let done_use = done.min(total_use);
    if let Some(cb) = cb {
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            cb.call1(py, (stage, done_use, total_use))?;
            Ok(())
        })
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    }
    Ok(())
}

fn prepare_output_space(
    out_dir: &Path,
    tmp_dir: &Path,
    args: &KmergeArgs,
    _samples: &[SampleEntry],
) -> Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create output dir: {}", out_dir.display()))?;

    let outputs = [
        out_dir.join(format!("{}.idv", args.prefix)),
        out_dir.join(format!("{}.bkmer", args.prefix)),
        out_dir.join(format!("{}.bsite", args.prefix)),
        out_dir.join(format!("{}.meta.json", args.prefix)),
    ];

    if !args.resume {
        for path in outputs.iter() {
            if path.exists() && !args.force {
                bail!(
                    "output file already exists: {} (use --force to overwrite)",
                    path.display()
                );
            }
        }
        if tmp_dir.exists() {
            if args.force {
                fs::remove_dir_all(tmp_dir).with_context(|| {
                    format!(
                        "failed to clear tmp dir before rerun: {}",
                        tmp_dir.display()
                    )
                })?;
            } else if tmp_dir.read_dir()?.next().is_some() {
                bail!(
                    "tmp dir already exists and is not empty: {} (use --resume or --force)",
                    tmp_dir.display()
                );
            }
        }
    }
    fs::create_dir_all(tmp_dir)
        .with_context(|| format!("failed to create tmp dir: {}", tmp_dir.display()))?;
    Ok(())
}

fn load_existing_summary(out_dir: &Path, prefix: &str) -> Result<Stage3Summary> {
    let meta_path = out_dir.join(format!("{}.meta.json", prefix));
    let file = File::open(&meta_path)
        .with_context(|| format!("failed to open meta file: {}", meta_path.display()))?;
    let meta: KmergeMeta = serde_json::from_reader(file)
        .with_context(|| format!("failed to parse meta file: {}", meta_path.display()))?;
    Ok(Stage3Summary {
        idv_path: out_dir.join(meta.idv_file),
        bkmer_path: out_dir.join(meta.bkmer_file),
        bsite_path: out_dir.join(meta.bsite_file),
        meta_path,
        n_samples: meta.n_samples,
        n_kmers: meta.n_kmers,
        bytes_per_col: meta.bytes_per_col,
        k: meta.k,
    })
}

fn max_records_per_flush(
    memory_mb: usize,
    threads: usize,
    max_run_size_mb: usize,
) -> Result<usize> {
    let memory_bytes = memory_mb
        .checked_mul(1024)
        .and_then(|x| x.checked_mul(1024))
        .ok_or_else(|| anyhow::anyhow!("memory size overflow"))?;
    let max_run_bytes = max_run_size_mb
        .checked_mul(1024)
        .and_then(|x| x.checked_mul(1024))
        .ok_or_else(|| anyhow::anyhow!("max-run-size overflow"))?;
    let per_worker = (memory_bytes / threads.max(1)).max(16 * 1024 * 1024);
    let usable_by_memory = (per_worker / 2).max(16 * 1024);
    let usable_bytes = usable_by_memory.min(max_run_bytes.max(16 * 1024));
    Ok((usable_bytes / 16).max(1))
}

fn is_stage_done(path: &Path) -> bool {
    path.is_file()
}

trait ExpandUser {
    fn expanduser(&self) -> Result<PathBuf>;
}

impl ExpandUser for PathBuf {
    fn expanduser(&self) -> Result<PathBuf> {
        let text = self.to_string_lossy();
        if text == "~" || text.starts_with("~/") {
            let home = std::env::var("HOME").context("HOME is not set")?;
            let suffix = text.strip_prefix('~').unwrap_or_default();
            return Ok(PathBuf::from(home).join(suffix.trim_start_matches('/')));
        }
        Ok(self.clone())
    }
}
