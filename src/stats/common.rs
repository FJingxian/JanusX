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
