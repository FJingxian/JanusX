use crate::kmer::ffi::kmc_count::{run_kmc_count, KmcCountStatsResult};
use anyhow::{bail, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Clone, Debug)]
struct KmcCountArgs {
    input_files: Vec<String>,
    output_prefix: String,
    tmp_dir: String,
    kmer_len: u32,
    threads: usize,
    max_ram_gb: u32,
    cutoff_min: u64,
    cutoff_max: u64,
    counter_max: u64,
    canonical: bool,
    input_type: String,
}

#[pyfunction(
    name = "kmer_count_run",
    signature = (
        input_files,
        output_prefix,
        tmp_dir = ".",
        kmer_len = 31,
        threads = 0,
        max_ram_gb = 12,
        cutoff_min = 2,
        cutoff_max = 1_000_000_000,
        counter_max = 255,
        canonical = true,
        input_type = "fastq"
    )
)]
pub fn kmer_count_run_py(
    py: Python<'_>,
    input_files: Vec<String>,
    output_prefix: &str,
    tmp_dir: &str,
    kmer_len: u32,
    threads: usize,
    max_ram_gb: u32,
    cutoff_min: u64,
    cutoff_max: u64,
    counter_max: u64,
    canonical: bool,
    input_type: &str,
) -> PyResult<Py<PyDict>> {
    let args = KmcCountArgs {
        input_files,
        output_prefix: output_prefix.to_string(),
        tmp_dir: tmp_dir.to_string(),
        kmer_len,
        threads,
        max_ram_gb,
        cutoff_min,
        cutoff_max,
        counter_max,
        canonical,
        input_type: input_type.to_string(),
    };

    let stats = py
        .detach(|| run_kmer_count_checked(args))
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    let out = PyDict::new(py);
    out.set_item("stage1_time_s", stats.stage1_time_s)?;
    out.set_item("stage2_time_s", stats.stage2_time_s)?;
    out.set_item("n_sequences", stats.n_sequences)?;
    out.set_item("tmp_size_stage1", stats.tmp_size_stage1)?;
    out.set_item("n_total_kmers", stats.n_total_kmers)?;
    out.set_item("n_unique_kmers", stats.n_unique_kmers)?;
    out.set_item("n_below_cutoff_min", stats.n_below_cutoff_min)?;
    out.set_item("n_above_cutoff_max", stats.n_above_cutoff_max)?;
    out.set_item("max_disk_usage", stats.max_disk_usage)?;
    Ok(out.unbind())
}

fn run_kmer_count_checked(args: KmcCountArgs) -> Result<KmcCountStatsResult> {
    if args.input_files.is_empty() {
        bail!("input_files cannot be empty");
    }
    if args.output_prefix.trim().is_empty() {
        bail!("output_prefix cannot be empty");
    }
    if args.tmp_dir.trim().is_empty() {
        bail!("tmp_dir cannot be empty");
    }
    if args.kmer_len == 0 {
        bail!("kmer_len must be > 0");
    }
    if args.max_ram_gb == 0 {
        bail!("max_ram_gb must be > 0");
    }

    run_kmc_count(
        &args.input_files,
        &args.output_prefix,
        &args.tmp_dir,
        args.kmer_len,
        u32::try_from(args.threads).unwrap_or(u32::MAX),
        args.max_ram_gb,
        args.cutoff_min,
        args.cutoff_max,
        args.counter_max,
        args.canonical,
        &args.input_type,
    )
}
