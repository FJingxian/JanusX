use std::os::raw::{c_char, c_int};

#[repr(C)]
pub struct JxKmcOpaque {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct JxKmcDbInfo {
    pub kmer_length: u32,
    pub min_count: u32,
    pub max_count: u64,
    pub total_kmers: u64,
    pub counter_size: u32,
    pub both_strands: u8,
    pub reserved: [u8; 7],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct JxKmcCountStats {
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

unsafe extern "C" {
    pub fn jx_kmc_open(db_prefix: *const c_char, k_hint: u32) -> *mut JxKmcOpaque;
    pub fn jx_kmc_info(handle: *mut JxKmcOpaque, out_info: *mut JxKmcDbInfo) -> c_int;
    pub fn jx_kmc_count_run(
        input_files: *const *const c_char,
        input_files_len: usize,
        output_prefix: *const c_char,
        tmp_dir: *const c_char,
        kmer_len: u32,
        threads: u32,
        max_ram_gb: u32,
        cutoff_min: u64,
        cutoff_max: u64,
        counter_max: u64,
        canonical: c_int,
        input_type: *const c_char,
        out_stats: *mut JxKmcCountStats,
    ) -> c_int;
    pub fn jx_kmc_read_batch_u64(
        handle: *mut JxKmcOpaque,
        kmer_buf: *mut u64,
        count_buf: *mut u32,
        max_records: usize,
    ) -> usize;
    pub fn jx_kmc_is_eof(handle: *mut JxKmcOpaque) -> c_int;
    pub fn jx_kmc_close(handle: *mut JxKmcOpaque);
    pub fn jx_kmc_last_error() -> *const c_char;
}
