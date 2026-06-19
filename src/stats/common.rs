use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::Arc;
use std::thread::JoinHandle;

pub(crate) const INTERRUPTED_MSG: &str = "Interrupted by user (Ctrl+C).";

#[inline]
pub(crate) fn check_ctrlc() -> Result<(), String> {
    Python::attach(|py| py.check_signals()).map_err(|_| INTERRUPTED_MSG.to_string())
}

#[inline]
pub(crate) fn map_err_string_to_py(err: String) -> PyErr {
    if err.contains(INTERRUPTED_MSG) {
        PyKeyboardInterrupt::new_err(err)
    } else {
        PyRuntimeError::new_err(err)
    }
}

#[inline]
pub(crate) fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let t = v.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on")
        })
        .unwrap_or(false)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProcessMemoryUsage {
    pub(crate) current_bytes: u64,
    pub(crate) metric: &'static str,
    pub(crate) rss_bytes: Option<u64>,
    pub(crate) footprint_bytes: Option<u64>,
}

#[inline]
pub(crate) fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.1} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

#[inline]
fn env_disabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let t = v.trim().to_ascii_lowercase();
            matches!(t.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

#[inline]
pub(crate) fn admx_madvise_dontneed_bytes(ptr: *const u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    if env_disabled("JANUSX_ADMX_MADVISE_DONTNEED") || env_disabled("JX_ADMX_MADVISE_DONTNEED") {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return;
        }
        let page = page_size as usize;
        let start = ptr as usize;
        let end = start.saturating_add(len);
        if end <= start {
            return;
        }
        let aligned_start = start & !(page - 1);
        let aligned_end = end.saturating_add(page - 1) & !(page - 1);
        let advise_len = aligned_end.saturating_sub(aligned_start);
        if advise_len == 0 {
            return;
        }
        unsafe {
            let _ = libc::madvise(
                aligned_start as *mut libc::c_void,
                advise_len,
                libc::MADV_DONTNEED,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (ptr, len);
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn process_memory_usage() -> Option<ProcessMemoryUsage> {
    let text = std::fs::read_to_string("/proc/self/statm").ok()?;
    let mut fields = text.split_whitespace();
    let _size_pages = fields.next()?;
    let rss_pages: u64 = fields.next()?.parse().ok()?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return None;
    }
    let rss = rss_pages.saturating_mul(page_size as u64);
    Some(ProcessMemoryUsage {
        current_bytes: rss,
        metric: "rss",
        rss_bytes: Some(rss),
        footprint_bytes: None,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn process_memory_usage() -> Option<ProcessMemoryUsage> {
    unsafe extern "C" {
        fn proc_pid_rusage(
            pid: libc::c_int,
            flavor: libc::c_int,
            buffer: *mut libc::c_void,
        ) -> libc::c_int;
    }

    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    let rc = unsafe {
        proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr() as *mut libc::c_void,
        )
    };
    if rc == 0 {
        let info = unsafe { info.assume_init() };
        return Some(ProcessMemoryUsage {
            current_bytes: info.ri_phys_footprint,
            metric: "phys_footprint",
            rss_bytes: Some(info.ri_resident_size),
            footprint_bytes: Some(info.ri_phys_footprint),
        });
    }

    let mut ru = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    let ru = unsafe { ru.assume_init() };
    let rss_peak = ru.ru_maxrss as u64;
    Some(ProcessMemoryUsage {
        current_bytes: rss_peak,
        metric: "rss_peak",
        rss_bytes: Some(rss_peak),
        footprint_bytes: None,
    })
}

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
pub(crate) fn process_memory_usage() -> Option<ProcessMemoryUsage> {
    let mut ru = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    let ru = unsafe { ru.assume_init() };
    let rss_peak = (ru.ru_maxrss as u64).saturating_mul(1024);
    Some(ProcessMemoryUsage {
        current_bytes: rss_peak,
        metric: "rss_peak",
        rss_bytes: Some(rss_peak),
        footprint_bytes: None,
    })
}

#[cfg(not(unix))]
pub(crate) fn process_memory_usage() -> Option<ProcessMemoryUsage> {
    None
}

pub(crate) fn admx_memory_limit_bytes_from_env() -> Option<u64> {
    for name in ["JANUSX_ADMX_MAX_FOOTPRINT_GB", "JANUSX_ADMX_MAX_RSS_GB"] {
        let Ok(raw) = std::env::var(name) else {
            continue;
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = trimmed.parse::<f64>().ok()?;
        if !value.is_finite() || value <= 0.0 {
            return None;
        }
        return Some((value * 1024.0_f64 * 1024.0_f64 * 1024.0_f64) as u64);
    }
    None
}

#[inline]
pub(crate) fn check_admx_memory_limit(stage: &str) -> Result<(), String> {
    let Some(limit) = admx_memory_limit_bytes_from_env() else {
        return Ok(());
    };
    let Some(usage) = process_memory_usage() else {
        return Ok(());
    };
    if usage.current_bytes <= limit {
        return Ok(());
    }
    let rss = usage
        .rss_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "NA".to_string());
    let footprint = usage
        .footprint_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "NA".to_string());
    Err(format!(
        "ADAMIXTURE memory limit exceeded at {stage}: {}={} (rss={}, footprint={}, limit={}).",
        usage.metric,
        format_bytes(usage.current_bytes),
        rss,
        footprint,
        format_bytes(limit),
    ))
}

thread_local! {
    static LOCAL_RAYON_POOLS: RefCell<HashMap<usize, Arc<rayon::ThreadPool>>> =
        RefCell::new(HashMap::new());
}

#[inline]
pub(crate) fn get_cached_pool(threads: usize) -> PyResult<Option<Arc<rayon::ThreadPool>>> {
    if threads == 0 {
        return Ok(None);
    }
    LOCAL_RAYON_POOLS.with(|cell| {
        let mut pools = cell.borrow_mut();
        if let Some(tp) = pools.get(&threads) {
            return Ok(Some(Arc::clone(tp)));
        }
        let tp = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("rayon pool: {e}")))?,
        );
        pools.insert(threads, Arc::clone(&tp));
        Ok(Some(tp))
    })
}

pub(crate) fn parse_index_vec_i64(
    indices: &[i64],
    upper_bound: usize,
    label: &str,
) -> PyResult<Vec<usize>> {
    let mut out = Vec::with_capacity(indices.len());
    for (i, &v) in indices.iter().enumerate() {
        if v < 0 {
            return Err(PyRuntimeError::new_err(format!(
                "{label}[{i}] must be >= 0, got {v}"
            )));
        }
        let u = v as usize;
        if u >= upper_bound {
            return Err(PyRuntimeError::new_err(format!(
                "{label}[{i}] out of range: {u} >= {upper_bound}"
            )));
        }
        out.push(u);
    }
    Ok(out)
}

pub(crate) struct AsyncTsvWriter {
    path: String,
    tx: Option<SyncSender<Vec<u8>>>,
    handle: Option<JoinHandle<Result<(), String>>>,
}

impl AsyncTsvWriter {
    pub(crate) fn with_config(
        path: &str,
        header: &[u8],
        writer_capacity: usize,
        queue_depth: usize,
    ) -> Result<Self, String> {
        const MIN_FLUSH_BYTES: usize = 8 * 1024 * 1024;
        const MAX_FLUSH_BYTES: usize = 64 * 1024 * 1024;
        const MIN_QUEUE_DEPTH: usize = 16;
        const MAX_QUEUE_DEPTH: usize = 64;
        let path_owned = path.to_string();
        let header_owned = header.to_vec();
        let queue_depth_eff = queue_depth
            .max((writer_capacity / (4 * 1024 * 1024)).max(1))
            .clamp(MIN_QUEUE_DEPTH, MAX_QUEUE_DEPTH);
        let (tx, rx) = sync_channel::<Vec<u8>>(queue_depth_eff);
        let path_for_writer = path_owned.clone();
        let flush_threshold = writer_capacity.clamp(MIN_FLUSH_BYTES, MAX_FLUSH_BYTES);
        let handle = std::thread::spawn(move || -> Result<(), String> {
            let out_file = File::create(&path_for_writer)
                .map_err(|e| format!("create {path_for_writer}: {e}"))?;
            let mut writer = BufWriter::with_capacity(writer_capacity.max(1), out_file);
            writer
                .write_all(&header_owned)
                .map_err(|e| format!("write header {path_for_writer}: {e}"))?;
            writer
                .flush()
                .map_err(|e| format!("flush header {path_for_writer}: {e}"))?;
            let mut pending_bytes = 0usize;
            for block in rx {
                if !block.is_empty() {
                    writer
                        .write_all(&block)
                        .map_err(|e| format!("write {path_for_writer}: {e}"))?;
                    pending_bytes = pending_bytes.saturating_add(block.len());
                    if pending_bytes >= flush_threshold {
                        writer
                            .flush()
                            .map_err(|e| format!("flush {path_for_writer}: {e}"))?;
                        pending_bytes = 0usize;
                    }
                }
            }
            writer
                .flush()
                .map_err(|e| format!("flush {path_for_writer}: {e}"))?;
            Ok(())
        });
        Ok(Self {
            path: path_owned,
            tx: Some(tx),
            handle: Some(handle),
        })
    }

    #[inline]
    pub(crate) fn send(&self, block: Vec<u8>) -> Result<(), String> {
        if block.is_empty() {
            return Ok(());
        }
        self.tx
            .as_ref()
            .ok_or_else(|| format!("writer channel already closed for {}", self.path))?
            .send(block)
            .map_err(|e| format!("send {}: {e}", self.path))
    }

    pub(crate) fn finish(mut self) -> Result<(), String> {
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| format!("writer thread panicked for {}", self.path))?
        } else {
            Ok(())
        }
    }
}

impl Drop for AsyncTsvWriter {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
