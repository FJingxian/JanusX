use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

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
