use crate::kmer::ffi::kmc_raw::{jx_kmc_count_run, jx_kmc_last_error, JxKmcCountStats};
use anyhow::{bail, Result};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};

#[derive(Clone, Debug)]
pub struct KmcCountStatsResult {
    pub stage1_time_s: f64,
    pub stage2_time_s: f64,
    pub n_sequences: u64,
    pub tmp_size_stage1: u64,
    pub n_total_kmers: u64,
    pub n_unique_kmers: u64,
    pub n_below_cutoff_min: u64,
    pub n_above_cutoff_max: u64,
    pub max_disk_usage: u64,
}

#[allow(clippy::too_many_arguments)]
pub fn run_kmc_count(
    input_files: &[String],
    output_prefix: &str,
    tmp_dir: &str,
    kmer_len: u32,
    threads: u32,
    max_ram_gb: u32,
    cutoff_min: u64,
    cutoff_max: u64,
    counter_max: u64,
    canonical: bool,
    input_type: &str,
) -> Result<KmcCountStatsResult> {
    if input_files.is_empty() {
        bail!("input_files cannot be empty");
    }

    let c_input_files = input_files
        .iter()
        .map(|path| CString::new(path.as_str()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let input_ptrs = c_input_files
        .iter()
        .map(|path| path.as_ptr())
        .collect::<Vec<*const c_char>>();
    let c_output_prefix = CString::new(output_prefix)?;
    let c_tmp_dir = CString::new(tmp_dir)?;
    let c_input_type = CString::new(input_type)?;

    let mut raw = JxKmcCountStats::default();
    let ok = unsafe {
        jx_kmc_count_run(
            input_ptrs.as_ptr(),
            input_ptrs.len(),
            c_output_prefix.as_ptr(),
            c_tmp_dir.as_ptr(),
            kmer_len,
            threads,
            max_ram_gb,
            cutoff_min,
            cutoff_max,
            counter_max,
            if canonical { 1 as c_int } else { 0 as c_int },
            c_input_type.as_ptr(),
            &mut raw as *mut JxKmcCountStats,
        )
    };
    if ok == 0 {
        bail!("KMC count failed: {}", last_error());
    }

    Ok(KmcCountStatsResult {
        stage1_time_s: raw.stage1_time_s,
        stage2_time_s: raw.stage2_time_s,
        n_sequences: raw.n_sequences,
        tmp_size_stage1: raw.tmp_size_stage1,
        n_total_kmers: raw.n_total_kmers,
        n_unique_kmers: raw.n_unique_kmers,
        n_below_cutoff_min: raw.n_below_cutoff_min,
        n_above_cutoff_max: raw.n_above_cutoff_max,
        max_disk_usage: raw.max_disk_usage,
    })
}

fn last_error() -> String {
    unsafe {
        let ptr = jx_kmc_last_error();
        if ptr.is_null() {
            return "unknown error".to_string();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}
