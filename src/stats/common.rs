use numpy::ndarray::{ArrayView, Dimension};
use numpy::{Element, PyReadonlyArray};
use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
#[cfg(unix)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Copy an ndarray view in logical standard (row-major) order.
///
/// Native kernels index flat buffers as row-major. ndarray iteration follows
/// logical row order regardless of the physical layout, including strided views.
#[inline]
pub(crate) fn copy_array_standard_layout<T: Copy, D: Dimension>(
    view: &ArrayView<'_, T, D>,
) -> Vec<T> {
    view.iter().copied().collect()
}

/// Borrow only standard-layout NumPy buffers, never column-major storage.
/// NumPy's `as_slice()` accepts both C- and Fortran-contiguous arrays.
#[inline]
pub(crate) fn array_standard_slice<'a, T: Element, D: Dimension>(
    array: &'a PyReadonlyArray<'_, T, D>,
) -> Result<&'a [T], numpy::NotContiguousError> {
    if array.as_array().is_standard_layout() {
        array.as_slice()
    } else {
        Err(numpy::NotContiguousError)
    }
}

#[inline]
pub(crate) fn array_standard_cow<'a, T: Element + Copy, D: Dimension>(
    array: &'a PyReadonlyArray<'_, T, D>,
) -> std::borrow::Cow<'a, [T]> {
    match array_standard_slice(array) {
        Ok(slice) => std::borrow::Cow::Borrowed(slice),
        Err(_) => std::borrow::Cow::Owned(copy_array_standard_layout(&array.as_array())),
    }
}

pub(crate) const INTERRUPTED_MSG: &str = "Interrupted by user (Ctrl+C).";
static INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
static PREV_SIGINT_HANDLER: AtomicUsize = AtomicUsize::new(0);
#[cfg(unix)]
static PREV_SIGTERM_HANDLER: AtomicUsize = AtomicUsize::new(0);

#[cfg(unix)]
extern "C" {
    fn signal(signum: libc::c_int, handler: usize) -> usize;
}

#[cfg(unix)]
extern "C" fn janusx_native_interrupt_handler(sig: i32) {
    INTERRUPT_REQUESTED.store(true, Ordering::SeqCst);
    let prev = match sig {
        libc::SIGINT => PREV_SIGINT_HANDLER.load(Ordering::SeqCst),
        libc::SIGTERM => PREV_SIGTERM_HANDLER.load(Ordering::SeqCst),
        _ => 0usize,
    };
    let self_handler = janusx_native_interrupt_handler as *const () as usize;
    if prev <= 1 || prev == self_handler || prev == usize::MAX {
        return;
    }
    let chained: extern "C" fn(i32) = unsafe { std::mem::transmute(prev) };
    chained(sig);
}

#[inline]
pub(crate) fn arm_interrupt_trap() {
    INTERRUPT_REQUESTED.store(false, Ordering::SeqCst);
    #[cfg(unix)]
    unsafe {
        let handler = janusx_native_interrupt_handler as *const () as usize;
        let prev_int = signal(libc::SIGINT, handler);
        if prev_int > 1 && prev_int != handler && prev_int != usize::MAX {
            PREV_SIGINT_HANDLER.store(prev_int, Ordering::SeqCst);
        }
        let prev_term = signal(libc::SIGTERM, handler);
        if prev_term > 1 && prev_term != handler && prev_term != usize::MAX {
            PREV_SIGTERM_HANDLER.store(prev_term, Ordering::SeqCst);
        }
    }
}

#[inline]
pub(crate) fn interrupt_requested() -> bool {
    INTERRUPT_REQUESTED.load(Ordering::SeqCst)
}

#[inline]
pub(crate) fn check_ctrlc() -> Result<(), String> {
    if interrupt_requested() {
        return Err(INTERRUPTED_MSG.to_string());
    }
    Python::attach(|py| py.check_signals()).map_err(|_| INTERRUPTED_MSG.to_string())?;
    if interrupt_requested() {
        Err(INTERRUPTED_MSG.to_string())
    } else {
        Ok(())
    }
}

#[inline]
pub(crate) fn map_err_string_to_py(err: String) -> PyErr {
    if err.contains(INTERRUPTED_MSG) {
        PyKeyboardInterrupt::new_err(err)
    } else {
        PyRuntimeError::new_err(err)
    }
}

/// Call a Python-side progress hook from a detached Rust computation.
///
/// Callers are expected to invoke this only at coarse block boundaries.  The
/// callback is intentionally kept out of the inner genotype loops so a TTY
/// progress bar cannot perturb the hot path.
#[inline]
pub(crate) fn emit_progress_callback(
    callback: Option<&Py<PyAny>>,
    stage: usize,
    done: usize,
    total: usize,
) -> Result<(), String> {
    let total_use = total.max(1);
    let done_use = done.min(total_use);
    if let Some(callback) = callback {
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            callback.call1(py, (stage, done_use, total_use))?;
            Ok(())
        })
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// Choose a bounded callback cadence without requiring callers to scan BIM
/// twice just to determine a display interval.
#[inline]
pub(crate) fn progress_step(total: usize, requested: usize) -> usize {
    if requested > 0 {
        requested
    } else {
        total.div_ceil(200).max(1)
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
        "FastPop memory limit exceeded at {stage}: {}={} (rss={}, footprint={}, limit={}).",
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

fn parse_index_vec_i64_impl(
    indices: &[i64],
    upper_bound: usize,
    label: &str,
) -> Result<Vec<usize>, String> {
    let mut out = Vec::with_capacity(indices.len());
    for (i, &v) in indices.iter().enumerate() {
        if v < 0 {
            return Err(format!("{label}[{i}] must be >= 0, got {v}"));
        }
        let u = v as usize;
        if u >= upper_bound {
            return Err(format!("{label}[{i}] out of range: {u} >= {upper_bound}"));
        }
        out.push(u);
    }
    Ok(out)
}

pub(crate) fn parse_index_vec_i64(
    indices: &[i64],
    upper_bound: usize,
    label: &str,
) -> PyResult<Vec<usize>> {
    parse_index_vec_i64_impl(indices, upper_bound, label).map_err(PyRuntimeError::new_err)
}

pub(crate) fn parse_index_vec_i64_value_error(
    indices: &[i64],
    upper_bound: usize,
    label: &str,
) -> PyResult<Vec<usize>> {
    parse_index_vec_i64_impl(indices, upper_bound, label).map_err(PyValueError::new_err)
}

pub(crate) fn parse_index_vec_i64_result(
    indices: &[i64],
    upper_bound: usize,
    label: &str,
) -> Result<Vec<usize>, String> {
    parse_index_vec_i64_impl(indices, upper_bound, label)
}

#[derive(Default)]
struct WriterBuffers {
    in_flight: usize,
    closed: bool,
    recycled: Option<Vec<u8>>,
}

struct WriterBudget {
    buffers: Mutex<WriterBuffers>,
    ready: Condvar,
    limit: usize,
}

struct WriterBlock {
    bytes: Vec<u8>,
    budget: Arc<WriterBudget>,
}

impl Drop for WriterBlock {
    fn drop(&mut self) {
        let mut state = self
            .budget
            .buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        state.in_flight -= self.bytes.capacity();
        // One reusable allocation, independently bounded to a quarter of the
        // in-flight budget. Large one-shot blocks are freed, not retained.
        if !state.closed && self.bytes.capacity() <= self.budget.limit / 4 {
            self.bytes.clear();
            state.recycled = Some(std::mem::take(&mut self.bytes));
        }
        self.budget.ready.notify_all();
    }
}

struct WriterExit(Arc<WriterBudget>);

impl Drop for WriterExit {
    fn drop(&mut self) {
        let mut state = self.0.buffers.lock().unwrap_or_else(|e| e.into_inner());
        state.closed = true;
        state.recycled = None;
        self.0.ready.notify_all();
    }
}

pub(crate) struct AsyncTsvWriter {
    path: String,
    tx: Option<SyncSender<WriterBlock>>,
    budget: Arc<WriterBudget>,
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
        const MAX_QUEUE_DEPTH: usize = 64;
        let path_owned = path.to_string();
        let header_owned = header.to_vec();
        let queue_depth_eff = queue_depth.clamp(1, MAX_QUEUE_DEPTH);
        let (tx, rx) = sync_channel::<WriterBlock>(queue_depth_eff);
        let path_for_writer = path_owned.clone();
        let flush_threshold = writer_capacity.clamp(MIN_FLUSH_BYTES, MAX_FLUSH_BYTES);
        let budget = Arc::new(WriterBudget {
            buffers: Mutex::new(WriterBuffers::default()),
            ready: Condvar::new(),
            limit: flush_threshold,
        });
        let writer_budget = Arc::clone(&budget);
        let handle = std::thread::spawn(move || -> Result<(), String> {
            // Wake blocked producers on every exit, including I/O error/panic.
            let _exit = WriterExit(writer_budget);
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
                if !block.bytes.is_empty() {
                    writer
                        .write_all(&block.bytes)
                        .map_err(|e| format!("write {path_for_writer}: {e}"))?;
                    pending_bytes = pending_bytes.saturating_add(block.bytes.len());
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
            budget,
            handle: Some(handle),
        })
    }

    #[inline]
    pub(crate) fn send(&self, block: Vec<u8>) -> Result<(), String> {
        if block.is_empty() {
            return Ok(());
        }
        let capacity = block.capacity();
        let mut state = self
            .budget
            .buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // A single oversized caller-owned block is allowed only when the
        // writer has no other in-flight blocks. Never enqueue several of them.
        while !state.closed
            && state.in_flight != 0
            && capacity > self.budget.limit.saturating_sub(state.in_flight)
        {
            state = self
                .budget
                .ready
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
        if state.closed {
            return Err(format!("writer closed for {}", self.path));
        }
        state.in_flight += capacity;
        drop(state);
        let message = WriterBlock {
            bytes: block,
            budget: Arc::clone(&self.budget),
        };
        self.tx
            .as_ref()
            .ok_or_else(|| format!("writer channel already closed for {}", self.path))?
            .send(message)
            .map_err(|e| format!("send {}: {e}", self.path))
    }

    /// Transfer formatted UTF-8 bytes without a copy and reclaim an empty
    /// buffer from the preceding write when available.
    pub(crate) fn send_text(&self, text: &mut String) -> Result<(), String> {
        if text.is_empty() {
            return Ok(());
        }
        self.send(std::mem::take(text).into_bytes())?;
        let recycled = self
            .budget
            .buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .recycled
            .take();
        if let Some(bytes) = recycled {
            *text = String::from_utf8(bytes).expect("recycled writer buffer is empty UTF-8");
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::progress_step;

    #[test]
    fn progress_step_is_bounded_to_about_two_hundred_updates() {
        assert_eq!(progress_step(0, 0), 1);
        assert_eq!(progress_step(1, 0), 1);
        assert_eq!(progress_step(200, 0), 1);
        assert_eq!(progress_step(201, 0), 2);
        assert_eq!(progress_step(1_604_281, 0), 8_022);
        assert_eq!(progress_step(1_604_281, 17), 17);
    }
}
