use crate::kmer::ffi::kmc_reader::KmcReader;
use crate::kmer::format::{KmergeMeta, SampleEntry};
use crate::kmer::stage1_bucket::{run_stage1, Stage1Config};
use crate::kmer::stage2_merge::{load_stage2_manifest, run_stage2, Stage2Config};
use crate::kmer::stage3_concat::{run_stage3, Stage3Config, Stage3Summary};
use anyhow::{bail, Context, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct KmergeArgs {
    pub db_inputs: Vec<String>,
    pub sample_ids: Option<Vec<String>>,
    pub out_dir: PathBuf,
    pub prefix: String,
    pub threads: usize,
    pub memory_mb: usize,
    pub min_presence: u32,
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
        min_presence = 5,
        freq = 0.02,
        tmp_dir = None,
        max_run_size = 2048,
        bucket_bits = 10,
        batch_size = 1_048_576,
        min_count = 1,
        resume = false,
        keep_tmp = false,
        force = false,
        log_path = None
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
    min_presence: u32,
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
) -> PyResult<Py<PyDict>> {
    let args = KmergeArgs {
        db_inputs,
        sample_ids,
        out_dir: PathBuf::from(out),
        prefix: prefix.to_string(),
        threads: thread.max(1),
        memory_mb: memory.max(1),
        min_presence,
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
    let k = inspect_common_k(&samples)?;
    let log_path = args.log_path.as_deref();

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
        run_stage1(&Stage1Config {
            tmp_dir: &tmp_dir,
            samples: &samples,
            k,
            min_count: args.min_count,
            batch_size: args.batch_size,
            bucket_bits: args.bucket_bits,
            max_records_per_flush,
            threads: args.threads,
        })?;
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
        run_stage2(&Stage2Config {
            tmp_dir: &tmp_dir,
            n_samples: samples.len(),
            min_presence: args.min_presence,
            freq: args.freq,
            bucket_bits: args.bucket_bits,
            threads: args.threads,
        })?
    };

    if parts.iter().map(|part| part.n_kmers).sum::<u64>() == 0 {
        bail!("no k-mers survived filters across all buckets");
    }

    emit_run_line(log_path, "Stage 3/3: concat parts -> final matrix")?;
    let summary = run_stage3(&Stage3Config {
        out_dir: &out_dir,
        prefix: &args.prefix,
        tmp_dir: &tmp_dir,
        samples: &samples,
        parts: &parts,
        k,
        min_count: args.min_count,
        min_presence: args.min_presence,
        freq: args.freq,
        bucket_bits: args.bucket_bits,
    })?;

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

fn resolve_samples(
    db_inputs: &[String],
    sample_ids_override: Option<&Vec<String>>,
) -> Result<Vec<SampleEntry>> {
    let mut raw_pairs = if db_inputs.len() == 1 && looks_like_db_list_file(&db_inputs[0]) {
        parse_db_list_file(&db_inputs[0])?
    } else {
        db_inputs
            .iter()
            .map(|raw| {
                let prefix = normalize_prefix(raw)?;
                Ok((default_sample_id(&prefix), prefix))
            })
            .collect::<Result<Vec<_>>>()?
    };

    if let Some(overrides) = sample_ids_override {
        if overrides.len() != raw_pairs.len() {
            bail!(
                "sample-id count mismatch: got {}, expected {}",
                overrides.len(),
                raw_pairs.len()
            );
        }
        for (idx, sample_id) in overrides.iter().enumerate() {
            let sid = sample_id.trim();
            if sid.is_empty() {
                bail!("sample-id cannot contain empty values");
            }
            raw_pairs[idx].0 = sid.to_string();
        }
    }

    let mut samples = Vec::with_capacity(raw_pairs.len());
    for (idx, (sample_id, prefix)) in raw_pairs.into_iter().enumerate() {
        validate_prefix(&prefix)?;
        samples.push(SampleEntry {
            index: idx as u32,
            sample_id,
            kmc_prefix: prefix,
        });
    }
    Ok(samples)
}

fn parse_db_list_file(path: &str) -> Result<Vec<(String, String)>> {
    let file =
        File::open(path).with_context(|| format!("failed to open KMC db list file: {path}"))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let text = line.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let fields = text.split_whitespace().collect::<Vec<_>>();
        let (sample_id, prefix_raw) = if fields.len() == 1 {
            let prefix = normalize_prefix(fields[0])?;
            (default_sample_id(&prefix), prefix)
        } else {
            (fields[0].to_string(), normalize_prefix(fields[1])?)
        };
        out.push((sample_id, prefix_raw));
    }
    if out.is_empty() {
        bail!("KMC db list file is empty: {path}");
    }
    Ok(out)
}

fn inspect_common_k(samples: &[SampleEntry]) -> Result<u32> {
    let mut common_k = None::<u32>;
    for sample in samples {
        let reader = KmcReader::open(&sample.kmc_prefix, 0)
            .with_context(|| format!("failed to inspect KMC db: {}", sample.kmc_prefix))?;
        let k = reader.info().kmer_length;
        if k == 0 {
            bail!("invalid KMC k-mer length for {}", sample.kmc_prefix);
        }
        if k > 31 {
            bail!("current Rust kmerge only supports k <= 31, got {k}");
        }
        if let Some(prev) = common_k {
            if prev != k {
                bail!(
                    "all KMC databases must share one k-mer length, got {} and {}",
                    prev,
                    k
                );
            }
        } else {
            common_k = Some(k);
        }
    }
    common_k.ok_or_else(|| anyhow::anyhow!("no samples available after resolving KMC inputs"))
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

fn looks_like_db_list_file(raw: &str) -> bool {
    let raw = raw.trim();
    if raw.is_empty() {
        return false;
    }
    let lower = raw.to_ascii_lowercase();
    if lower.ends_with(".kmc_pre") || lower.ends_with(".kmc_suf") {
        return false;
    }
    Path::new(raw).is_file()
}

fn normalize_prefix(path_or_prefix: &str) -> Result<String> {
    let value = path_or_prefix.trim();
    if value.is_empty() {
        bail!("KMC prefix cannot be empty");
    }
    let lower = value.to_ascii_lowercase();
    if lower.ends_with(".kmc_pre") {
        return Ok(value[..value.len() - ".kmc_pre".len()].to_string());
    }
    if lower.ends_with(".kmc_suf") {
        return Ok(value[..value.len() - ".kmc_suf".len()].to_string());
    }
    Ok(value.to_string())
}

fn validate_prefix(prefix: &str) -> Result<()> {
    let pre = PathBuf::from(format!("{prefix}.kmc_pre"));
    let suf = PathBuf::from(format!("{prefix}.kmc_suf"));
    if !pre.is_file() || !suf.is_file() {
        bail!(
            "KMC database not complete for prefix `{prefix}`. Expected {} and {}",
            pre.display(),
            suf.display()
        );
    }
    Ok(())
}

fn default_sample_id(prefix: &str) -> String {
    let name = Path::new(prefix)
        .file_name()
        .and_then(|x| x.to_str())
        .unwrap_or("sample");
    let mapped = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    let trimmed = mapped.trim_matches(|ch| matches!(ch, '.' | '_' | '-'));
    if trimmed.is_empty() {
        "sample".to_string()
    } else {
        trimmed.to_string()
    }
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
