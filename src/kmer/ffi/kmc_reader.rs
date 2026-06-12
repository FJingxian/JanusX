use crate::kmer::ffi::kmc_raw::{
    jx_kmc_close, jx_kmc_info, jx_kmc_is_eof, jx_kmc_last_error, jx_kmc_open,
    jx_kmc_read_batch_u64, JxKmcDbInfo, JxKmcOpaque,
};
use anyhow::{bail, Result};
use std::ffi::{CStr, CString};

#[derive(Clone, Debug)]
pub struct KmcDbInfo {
    pub kmer_length: u32,
    pub min_count: u32,
    pub max_count: u64,
    pub total_kmers: u64,
    pub counter_size: u32,
    pub both_strands: bool,
}

pub struct KmcReader {
    handle: *mut JxKmcOpaque,
    info: KmcDbInfo,
}

unsafe impl Send for KmcReader {}

impl KmcReader {
    pub fn open(prefix: &str, k_hint: u32) -> Result<Self> {
        let c_prefix = CString::new(prefix)?;
        let handle = unsafe { jx_kmc_open(c_prefix.as_ptr(), k_hint) };
        if handle.is_null() {
            bail!("failed to open KMC database `{prefix}`: {}", last_error());
        }

        let mut raw = JxKmcDbInfo::default();
        let ok = unsafe { jx_kmc_info(handle, &mut raw as *mut JxKmcDbInfo) };
        if ok == 0 {
            unsafe { jx_kmc_close(handle) };
            bail!(
                "failed to inspect KMC database `{prefix}`: {}",
                last_error()
            );
        }

        Ok(Self {
            handle,
            info: KmcDbInfo {
                kmer_length: raw.kmer_length,
                min_count: raw.min_count,
                max_count: raw.max_count,
                total_kmers: raw.total_kmers,
                counter_size: raw.counter_size,
                both_strands: raw.both_strands != 0,
            },
        })
    }

    pub fn info(&self) -> &KmcDbInfo {
        &self.info
    }

    pub fn kmers_are_canonical(&self) -> bool {
        self.info.both_strands
    }

    pub fn read_batch(&mut self, kmers: &mut [u64], counts: &mut [u32]) -> Result<usize> {
        assert_eq!(kmers.len(), counts.len());
        let n = unsafe {
            jx_kmc_read_batch_u64(
                self.handle,
                kmers.as_mut_ptr(),
                counts.as_mut_ptr(),
                kmers.len(),
            )
        };
        if n == 0 {
            let eof = unsafe { jx_kmc_is_eof(self.handle) };
            if eof == 0 {
                bail!("KMC batch read failed: {}", last_error());
            }
        }
        Ok(n)
    }
}

impl Drop for KmcReader {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { jx_kmc_close(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
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
