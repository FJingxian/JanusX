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

unsafe extern "C" {
    pub fn jx_kmc_open(db_prefix: *const c_char, k_hint: u32) -> *mut JxKmcOpaque;
    pub fn jx_kmc_info(handle: *mut JxKmcOpaque, out_info: *mut JxKmcDbInfo) -> c_int;
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
