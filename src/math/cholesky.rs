use std::collections::HashMap;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::mem::{align_of, size_of};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use faer::dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
use faer::reborrow::*;
use faer::sparse::linalg::amd;
use faer::sparse::linalg::cholesky::{simplicial, supernodal, LltRegularization};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{mat, Conj, Index, Parallelism, Side};
use memmap2::Mmap;

use crate::spgrm::SparseGrmCsc;

const JXGRM_HEADER_BYTES: usize = 16;
const JXGRM_VALUES_ALIGN_BYTES: usize = size_of::<f64>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SparseJxgrmAnalyzeProgressStage {
    OpenFile,
    ValidateCsc,
    DirectSamples,
    SymbolicAnalyze,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct SparseFileCacheKey {
    path: String,
    file_len: u64,
    modified_ns: u128,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct SparseAnalysisCacheKey {
    path: String,
    file_len: u64,
    modified_ns: u128,
    sample_len: usize,
    sample_hash: u64,
}

fn sparse_analysis_cache(
) -> &'static Mutex<HashMap<SparseAnalysisCacheKey, Arc<SparseJxgrmCholeskyAnalysis>>> {
    static CACHE: OnceLock<
        Mutex<HashMap<SparseAnalysisCacheKey, Arc<SparseJxgrmCholeskyAnalysis>>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn sparse_validation_cache() -> &'static Mutex<HashMap<SparseFileCacheKey, ()>> {
    static CACHE: OnceLock<Mutex<HashMap<SparseFileCacheKey, ()>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn sparse_modified_ns(meta: &fs::Metadata) -> u128 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0u128)
}

fn sparse_file_cache_key_from_meta(path: &str, meta: &fs::Metadata) -> SparseFileCacheKey {
    SparseFileCacheKey {
        path: path.to_string(),
        file_len: meta.len(),
        modified_ns: sparse_modified_ns(meta),
    }
}

fn sparse_sample_hash(sample_idx: Option<&[usize]>) -> (usize, u64) {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let len = sample_idx.map(|v| v.len()).unwrap_or(0usize);
    len.hash(&mut hasher);
    if let Some(idx) = sample_idx {
        for &sid in idx {
            sid.hash(&mut hasher);
        }
    }
    (len, hasher.finish())
}

fn sparse_analysis_cache_key(
    path: &str,
    sample_idx: Option<&[usize]>,
) -> Result<SparseAnalysisCacheKey, String> {
    let meta = fs::metadata(path)
        .map_err(|e| format!("read sparse JXGRM metadata for cache key failed: {e}"))?;
    let file_key = sparse_file_cache_key_from_meta(path, &meta);
    let (sample_len, sample_hash) = sparse_sample_hash(sample_idx);
    Ok(SparseAnalysisCacheKey {
        path: file_key.path,
        file_len: file_key.file_len,
        modified_ns: file_key.modified_ns,
        sample_len,
        sample_hash,
    })
}

#[inline]
fn sparse_numeric_parallelism(threads: usize) -> Parallelism {
    if threads <= 1 {
        Parallelism::None
    } else {
        Parallelism::Rayon(threads)
    }
}

fn env_truthy_local(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "no" || s == "off")
        })
        .unwrap_or(false)
}

fn sparse_factor_timing_enabled() -> bool {
    env_truthy_local("JX_SPARSE_FACTOR_TIMING")
        || env_truthy_local("JX_SPARSE_REML_TIMING")
        || env_truthy_local("JX_SPLMM_PREPARE_STAGE_TIMING")
}

#[derive(Clone, Copy, Debug, Default)]
struct SparseFactorizeCoreTiming {
    numeric_secs: f64,
    logdet_secs: f64,
    total_secs: f64,
}

fn emit_sparse_factor_timing(
    stage: &str,
    prep_secs: f64,
    numeric_secs: f64,
    logdet_secs: f64,
    cleanup_secs: f64,
    total_secs: f64,
    n: usize,
    a_nnz: usize,
    factor_nnz: usize,
    diag_n: usize,
    parallelism: &str,
) {
    let other_secs = (total_secs - prep_secs - numeric_secs - logdet_secs - cleanup_secs).max(0.0);
    eprintln!(
        "Sparse factor timing: stage={stage}, prep={prep_secs:.3}s, numeric={numeric_secs:.3}s, logdet={logdet_secs:.3}s, cleanup={cleanup_secs:.3}s, other={other_secs:.3}s, total={total_secs:.3}s, n={n}, a_nnz={a_nnz}, factor_nnz={factor_nnz}, diag_n={diag_n}, parallelism={parallelism}",
    );
}

pub(crate) trait SparseGrmCscView {
    fn n_samples(&self) -> usize;
    fn nnz(&self) -> usize;
    fn col_ptr(&self) -> &[u64];
    fn row_indices(&self) -> &[u32];
    fn values(&self) -> &[f64];
}

impl SparseGrmCscView for SparseGrmCsc {
    #[inline]
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.nnz
    }

    #[inline]
    fn col_ptr(&self) -> &[u64] {
        self.col_ptr.as_slice()
    }

    #[inline]
    fn row_indices(&self) -> &[u32] {
        self.row_indices.as_slice()
    }

    #[inline]
    fn values(&self) -> &[f64] {
        self.values.as_slice()
    }
}

#[derive(Debug)]
pub(crate) struct MmapSparseGrmCsc {
    mmap: Mmap,
    n_samples: usize,
    nnz: usize,
    col_ptr_offset: usize,
    row_indices_offset: usize,
    values_offset: usize,
}

impl SparseGrmCscView for MmapSparseGrmCsc {
    #[inline]
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.nnz
    }

    #[inline]
    fn col_ptr(&self) -> &[u64] {
        let start = self.col_ptr_offset;
        let end = start + (self.n_samples + 1) * size_of::<u64>();
        unsafe { cast_aligned_slice_unchecked::<u64>(&self.mmap[start..end]) }
    }

    #[inline]
    fn row_indices(&self) -> &[u32] {
        let start = self.row_indices_offset;
        let end = start + self.nnz * size_of::<u32>();
        unsafe { cast_aligned_slice_unchecked::<u32>(&self.mmap[start..end]) }
    }

    #[inline]
    fn values(&self) -> &[f64] {
        let start = self.values_offset;
        let end = start + self.nnz * size_of::<f64>();
        unsafe { cast_aligned_slice_unchecked::<f64>(&self.mmap[start..end]) }
    }
}

impl MmapSparseGrmCsc {
    #[inline]
    pub(crate) fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub(crate) fn open(path: &str) -> Result<Self, String> {
        Self::open_with_progress(path, |_done, _total| Ok(()))
    }

    pub(crate) fn open_with_progress<F>(path: &str, mut progress: F) -> Result<Self, String>
    where
        F: FnMut(usize, usize) -> Result<(), String>,
    {
        let file = File::open(path)
            .map_err(|e| format!("failed to open sparse GRM CSC file {path}: {e}"))?;
        let file_meta = file
            .metadata()
            .map_err(|e| format!("failed to read sparse GRM CSC metadata {path}: {e}"))?;
        let validation_key = sparse_file_cache_key_from_meta(path, &file_meta);
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| format!("failed to mmap {path}: {e}"))?;
        if mmap.len() < JXGRM_HEADER_BYTES {
            return Err(format!(
                "Sparse GRM CSC file is too short: got {} bytes, need at least {JXGRM_HEADER_BYTES}",
                mmap.len()
            ));
        }
        let n_samples = usize::try_from(read_le_u64(&mmap, 0, "n_samples")?)
            .map_err(|_| "Sparse GRM header n_samples does not fit into usize".to_string())?;
        let nnz = usize::try_from(read_le_u64(&mmap, 8, "nnz")?)
            .map_err(|_| "Sparse GRM header nnz does not fit into usize".to_string())?;
        let col_ptr_offset = JXGRM_HEADER_BYTES;
        let col_ptr_bytes = (n_samples + 1)
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| "Sparse GRM col_ptr byte size overflow".to_string())?;
        let row_indices_offset = col_ptr_offset
            .checked_add(col_ptr_bytes)
            .ok_or_else(|| "Sparse GRM row_indices offset overflow".to_string())?;
        let row_indices_bytes = nnz
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| "Sparse GRM row_indices byte size overflow".to_string())?;
        let values_offset_unpadded = row_indices_offset
            .checked_add(row_indices_bytes)
            .ok_or_else(|| "Sparse GRM values offset overflow".to_string())?;
        let values_offset_padded = align_up_to(values_offset_unpadded, JXGRM_VALUES_ALIGN_BYTES)
            .ok_or_else(|| "Sparse GRM padded values offset overflow".to_string())?;
        let values_bytes = nnz
            .checked_mul(size_of::<f64>())
            .ok_or_else(|| "Sparse GRM values byte size overflow".to_string())?;
        let expected_len_legacy = values_offset_unpadded
            .checked_add(values_bytes)
            .ok_or_else(|| "Sparse GRM total byte size overflow".to_string())?;
        let expected_len_padded = values_offset_padded
            .checked_add(values_bytes)
            .ok_or_else(|| "Sparse GRM padded total byte size overflow".to_string())?;
        let (values_offset, padded_layout) = if mmap.len() == expected_len_padded {
            (values_offset_padded, true)
        } else if mmap.len() == expected_len_legacy {
            (values_offset_unpadded, false)
        } else {
            return Err(format!(
                "Sparse GRM CSC file size mismatch: got {} bytes, expected {} (legacy) or {} (padded) from header",
                mmap.len(),
                expected_len_legacy,
                expected_len_padded
            ));
        };
        let col_ptr_end = row_indices_offset;
        let row_indices_end = values_offset_unpadded;
        cast_aligned_slice_checked::<u64>(&mmap[col_ptr_offset..col_ptr_end], "col_ptr")?;
        cast_aligned_slice_checked::<u32>(
            &mmap[row_indices_offset..row_indices_end],
            "row_indices",
        )?;
        if padded_layout && values_offset > row_indices_end {
            let pad = &mmap[row_indices_end..values_offset];
            if pad.iter().any(|&b| b != 0) {
                return Err(format!(
                    "Sparse GRM padded layout contains non-zero alignment bytes before values payload in {path}"
                ));
            }
        }
        if let Err(err) = cast_aligned_slice_checked::<f64>(
            &mmap[values_offset..values_offset + values_bytes],
            "values",
        ) {
            if !padded_layout && values_offset != values_offset_padded {
                return Err(format!(
                    "Sparse GRM uses legacy unpadded values layout that cannot be mmap-aligned for zero-copy load: {path}. Rebuild this .spgrm with the current JanusX writer. Detail: {err}"
                ));
            }
            return Err(err);
        }
        let out = Self {
            mmap,
            n_samples,
            nnz,
            col_ptr_offset,
            row_indices_offset,
            values_offset,
        };
        let total = out.n_samples().saturating_add(1).max(1);
        let already_validated = sparse_validation_cache()
            .lock()
            .map_err(|_| "sparse validation cache lock poisoned".to_string())?
            .contains_key(&validation_key);
        if already_validated {
            progress(total, total)?;
        } else {
            progress(1usize.min(total), total)?;
            validate_sparse_grm_csc_view_with_progress(&out, |done, validate_total| {
                let mapped_done = 1usize.saturating_add(done).min(total);
                let mapped_total = 1usize.saturating_add(validate_total).max(total);
                progress(mapped_done, mapped_total)
            })?;
            sparse_validation_cache()
                .lock()
                .map_err(|_| "sparse validation cache lock poisoned".to_string())?
                .insert(validation_key, ());
        }
        Ok(out)
    }

    pub(crate) fn to_owned(&self) -> SparseGrmCsc {
        SparseGrmCsc {
            n_samples: self.n_samples,
            nnz: self.nnz,
            col_ptr: self.col_ptr().to_vec(),
            row_indices: self.row_indices().to_vec(),
            values: self.values().to_vec(),
        }
    }
}

pub(crate) fn sparse_jxgrm_header_n_samples(path: &str) -> Result<usize, String> {
    let mut file =
        File::open(path).map_err(|e| format!("open sparse JXGRM header {path} failed: {e}"))?;
    let mut header = [0u8; JXGRM_HEADER_BYTES];
    file.read_exact(&mut header)
        .map_err(|e| format!("read sparse JXGRM header {path} failed: {e}"))?;
    usize::try_from(read_le_u64(&header, 0, "n_samples")?)
        .map_err(|_| "Sparse JXGRM header n_samples does not fit into usize".to_string())
}

#[inline]
fn read_le_u64(bytes: &[u8], offset: usize, label: &str) -> Result<u64, String> {
    let end = offset
        .checked_add(size_of::<u64>())
        .ok_or_else(|| format!("Sparse GRM {label} offset overflow"))?;
    let src = bytes
        .get(offset..end)
        .ok_or_else(|| format!("Sparse GRM missing {label} bytes"))?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(src);
    Ok(u64::from_le_bytes(buf))
}

#[inline]
fn align_up_to(offset: usize, align: usize) -> Option<usize> {
    if align == 0 || !align.is_power_of_two() {
        return None;
    }
    let mask = align - 1;
    offset.checked_add(mask).map(|x| x & !mask)
}

#[inline]
fn cast_aligned_slice_checked<'a, T>(bytes: &'a [u8], label: &str) -> Result<&'a [T], String> {
    if !cfg!(target_endian = "little") {
        return Err(format!(
            "Sparse GRM mmap fast path requires little-endian target for {label}"
        ));
    }
    if bytes.len() % size_of::<T>() != 0 {
        return Err(format!(
            "Sparse GRM mmap slice {label} has invalid byte length {} for element size {}",
            bytes.len(),
            size_of::<T>()
        ));
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(align_of::<T>()) {
        return Err(format!(
            "Sparse GRM mmap slice {label} is not aligned to {} bytes",
            align_of::<T>()
        ));
    }
    Ok(unsafe { cast_aligned_slice_unchecked::<T>(bytes) })
}

#[inline]
unsafe fn cast_aligned_slice_unchecked<'a, T>(bytes: &'a [u8]) -> &'a [T] {
    std::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / size_of::<T>())
}

fn validate_sparse_grm_csc_view<V: SparseGrmCscView + ?Sized>(csc: &V) -> Result<(), String> {
    validate_sparse_grm_csc_view_with_progress(csc, |_done, _total| Ok(()))
}

fn validate_sparse_grm_csc_view_with_progress<V, F>(csc: &V, mut progress: F) -> Result<(), String>
where
    V: SparseGrmCscView + ?Sized,
    F: FnMut(usize, usize) -> Result<(), String>,
{
    if csc.n_samples() == 0 {
        return Err("Sparse GRM CSC requires n_samples > 0".to_string());
    }
    if csc.col_ptr().len() != csc.n_samples() + 1 {
        return Err(format!(
            "Sparse GRM CSC col_ptr length mismatch: got {}, expected {}",
            csc.col_ptr().len(),
            csc.n_samples() + 1
        ));
    }
    if csc.row_indices().len() != csc.nnz() || csc.values().len() != csc.nnz() {
        return Err(format!(
            "Sparse GRM CSC nnz mismatch: row_indices={}, values={}, header nnz={}",
            csc.row_indices().len(),
            csc.values().len(),
            csc.nnz()
        ));
    }
    if csc.col_ptr()[0] != 0 {
        return Err(format!(
            "Sparse GRM CSC must start with col_ptr[0] = 0, got {}",
            csc.col_ptr()[0]
        ));
    }
    if csc.col_ptr()[csc.n_samples()] != csc.nnz() as u64 {
        return Err(format!(
            "Sparse GRM CSC terminal col_ptr mismatch: got {}, expected {}",
            csc.col_ptr()[csc.n_samples()],
            csc.nnz()
        ));
    }

    let total = csc.n_samples().max(1);
    let notify_step = (total / 256).max(1);
    let mut last_done = 0usize;
    progress(0, total)?;
    for col in 0..csc.n_samples() {
        let start = usize::try_from(csc.col_ptr()[col])
            .map_err(|_| format!("col_ptr[{col}] does not fit into usize"))?;
        let end = usize::try_from(csc.col_ptr()[col + 1])
            .map_err(|_| format!("col_ptr[{}] does not fit into usize", col + 1))?;
        if start > end || end > csc.nnz() {
            return Err(format!(
                "Sparse GRM CSC invalid column span at col {col}: [{start}, {end}) with nnz={}",
                csc.nnz()
            ));
        }
        let mut prev_row = None::<u32>;
        let mut seen_diag = false;
        for idx in start..end {
            let row = csc.row_indices()[idx];
            if row as usize >= csc.n_samples() {
                return Err(format!(
                    "Sparse GRM CSC row index out of bounds at col {col}: row={row}, n={}",
                    csc.n_samples()
                ));
            }
            if (row as usize) < col {
                return Err(format!(
                    "Sparse GRM CSC must store lower triangle, but found row={row} < col={col}"
                ));
            }
            if let Some(prev) = prev_row {
                if row <= prev {
                    return Err(format!(
                        "Sparse GRM CSC row indices must be strictly increasing within each column; col={col}, prev={prev}, row={row}"
                    ));
                }
            }
            if row as usize == col {
                seen_diag = true;
            }
            let val = csc.values()[idx];
            if !val.is_finite() {
                return Err(format!(
                    "Sparse GRM CSC contains non-finite value at col={col}, row={row}"
                ));
            }
            prev_row = Some(row);
        }
        if !seen_diag {
            return Err(format!(
                "Sparse GRM CSC missing diagonal entry at column {col}; .spgrm from spgrm should keep all diagonals"
            ));
        }
        let done = col + 1;
        if done == total || done >= last_done.saturating_add(notify_step) {
            progress(done, total)?;
            last_done = done;
        }
    }
    Ok(())
}

pub fn read_sparse_grm_csc(path: &str) -> Result<SparseGrmCsc, String> {
    Ok(MmapSparseGrmCsc::open(path)?.to_owned())
}

fn sparse_analysis_cached_progress<F>(
    path: &str,
    sample_idx: Option<&[usize]>,
    mut progress: F,
) -> Result<Arc<SparseJxgrmCholeskyAnalysis>, String>
where
    F: FnMut(SparseJxgrmAnalyzeProgressStage, usize, usize) -> Result<(), String>,
{
    let cache_key = sparse_analysis_cache_key(path, sample_idx)?;
    if let Some(cached) = sparse_analysis_cache()
        .lock()
        .map_err(|_| "sparse analysis cache lock poisoned".to_string())?
        .get(&cache_key)
        .cloned()
    {
        progress(SparseJxgrmAnalyzeProgressStage::OpenFile, 1, 1)?;
        progress(SparseJxgrmAnalyzeProgressStage::ValidateCsc, 1, 1)?;
        progress(SparseJxgrmAnalyzeProgressStage::DirectSamples, 1, 1)?;
        progress(SparseJxgrmAnalyzeProgressStage::SymbolicAnalyze, 1, 1)?;
        return Ok(cached);
    }

    progress(SparseJxgrmAnalyzeProgressStage::OpenFile, 0, 1)?;
    let csc = MmapSparseGrmCsc::open_with_progress(path, |done, total| {
        progress(SparseJxgrmAnalyzeProgressStage::ValidateCsc, done, total)
    })?;
    progress(SparseJxgrmAnalyzeProgressStage::OpenFile, 1, 1)?;

    progress(SparseJxgrmAnalyzeProgressStage::DirectSamples, 0, 1)?;
    let analysis = if let Some(idx) = sample_idx {
        if idx.len() == csc.n_samples() && idx.iter().enumerate().all(|(i, &sid)| sid == i) {
            progress(SparseJxgrmAnalyzeProgressStage::DirectSamples, 1, 1)?;
            progress(SparseJxgrmAnalyzeProgressStage::SymbolicAnalyze, 0, 1)?;
            let analysis = sparse_cholesky_analyze_jxgrm_view(&csc)?;
            progress(SparseJxgrmAnalyzeProgressStage::SymbolicAnalyze, 1, 1)?;
            analysis
        } else {
            let subset = subset_sparse_grm_csc(&csc, idx)?;
            progress(SparseJxgrmAnalyzeProgressStage::DirectSamples, 1, 1)?;
            progress(SparseJxgrmAnalyzeProgressStage::SymbolicAnalyze, 0, 1)?;
            let analysis = sparse_cholesky_analyze_jxgrm_csc(&subset)?;
            progress(SparseJxgrmAnalyzeProgressStage::SymbolicAnalyze, 1, 1)?;
            analysis
        }
    } else {
        progress(SparseJxgrmAnalyzeProgressStage::DirectSamples, 1, 1)?;
        progress(SparseJxgrmAnalyzeProgressStage::SymbolicAnalyze, 0, 1)?;
        let analysis = sparse_cholesky_analyze_jxgrm_view(&csc)?;
        progress(SparseJxgrmAnalyzeProgressStage::SymbolicAnalyze, 1, 1)?;
        analysis
    };

    let analysis = Arc::new(analysis);
    let mut cache = sparse_analysis_cache()
        .lock()
        .map_err(|_| "sparse analysis cache lock poisoned".to_string())?;
    cache
        .entry(cache_key)
        .or_insert_with(|| Arc::clone(&analysis));
    Ok(analysis)
}

pub(crate) fn sparse_cholesky_analyze_subset_jxgrm_path_cached(
    path: &str,
    sample_idx: Option<&[usize]>,
) -> Result<Arc<SparseJxgrmCholeskyAnalysis>, String> {
    sparse_analysis_cached_progress(path, sample_idx, |_stage, _done, _total| Ok(()))
}

pub(crate) fn sparse_cholesky_analyze_subset_jxgrm_path_cached_with_progress<F>(
    path: &str,
    sample_idx: Option<&[usize]>,
    progress: F,
) -> Result<Arc<SparseJxgrmCholeskyAnalysis>, String>
where
    F: FnMut(SparseJxgrmAnalyzeProgressStage, usize, usize) -> Result<(), String>,
{
    sparse_analysis_cached_progress(path, sample_idx, progress)
}

pub(crate) fn subset_sparse_grm_csc<V: SparseGrmCscView + ?Sized>(
    csc: &V,
    sample_idx: &[usize],
) -> Result<SparseGrmCsc, String> {
    if sample_idx.is_empty() {
        return Err("Sparse GRM subset requires at least one sample".to_string());
    }
    if sample_idx.iter().any(|&sid| sid >= csc.n_samples()) {
        return Err(format!(
            "Sparse GRM subset index out of range for n_samples={}",
            csc.n_samples()
        ));
    }
    let mut orig_to_new = vec![usize::MAX; csc.n_samples()];
    for (new_idx, &orig_idx) in sample_idx.iter().enumerate() {
        if orig_to_new[orig_idx] != usize::MAX {
            return Err(format!(
                "Sparse GRM subset contains duplicated sample index: {orig_idx}"
            ));
        }
        orig_to_new[orig_idx] = new_idx;
    }

    let n_samples = sample_idx.len();
    let mut per_col = vec![Vec::<(u32, f64)>::new(); n_samples];
    for old_col in 0..csc.n_samples() {
        let new_col = orig_to_new[old_col];
        if new_col == usize::MAX {
            continue;
        }
        let start = csc.col_ptr()[old_col] as usize;
        let end = csc.col_ptr()[old_col + 1] as usize;
        for idx in start..end {
            let old_row = csc.row_indices()[idx] as usize;
            let new_row = orig_to_new[old_row];
            if new_row == usize::MAX {
                continue;
            }
            let (row_out, col_out) = if new_row >= new_col {
                (new_row, new_col)
            } else {
                (new_col, new_row)
            };
            per_col[col_out].push((row_out as u32, csc.values()[idx]));
        }
    }

    let mut col_ptr = vec![0u64; n_samples + 1];
    let mut row_indices = Vec::<u32>::new();
    let mut sub_values = Vec::<f64>::new();
    for col in 0..n_samples {
        let entries = &mut per_col[col];
        entries.sort_unstable_by_key(|&(row, _)| row);
        let mut write = 0usize;
        for read in 0..entries.len() {
            let (row, val) = entries[read];
            if (row as usize) < col {
                return Err(format!(
                    "Sparse GRM subset expects lower-triangle ordering; got row={} < col={col}",
                    row as usize
                ));
            }
            if write > 0 && entries[write - 1].0 == row {
                entries[write - 1].1 += val;
            } else {
                entries[write] = (row, val);
                write += 1;
            }
        }
        entries.truncate(write);
        for &(row, val) in entries.iter() {
            row_indices.push(row);
            sub_values.push(val);
        }
        col_ptr[col + 1] = row_indices.len() as u64;
    }

    let nnz = row_indices.len();
    let subset = SparseGrmCsc {
        n_samples,
        nnz,
        col_ptr,
        row_indices,
        values: sub_values,
    };
    validate_sparse_grm_csc_view(&subset)?;
    Ok(subset)
}

fn find_diag_positions_from_symbolic(
    n: usize,
    col_ptr: &[usize],
    row_indices: &[usize],
) -> Result<Vec<usize>, String> {
    let mut diag_pos = vec![usize::MAX; n];
    for col in 0..n {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        let mut found = false;
        for idx in start..end {
            if row_indices[idx] == col {
                diag_pos[col] = idx;
                found = true;
                break;
            }
        }
        if !found {
            return Err(format!(
                "Sparse Cholesky cannot locate diagonal entry in column {col}"
            ));
        }
    }
    Ok(diag_pos)
}

fn supernodal_llt_logdet(
    symbolic: &supernodal::SymbolicSupernodalCholesky<usize>,
    l_values: &[f64],
) -> Result<f64, String> {
    let mut sum_log_diag = 0.0_f64;
    for s in 0..symbolic.n_supernodes() {
        let start = symbolic.supernode_begin()[s].zx();
        let end = symbolic.supernode_end()[s].zx();
        let ncols = end.saturating_sub(start);
        let row_pat_start = symbolic.col_ptrs_for_row_indices()[s].zx();
        let row_pat_end = symbolic.col_ptrs_for_row_indices()[s + 1].zx();
        let nrows = ncols + row_pat_end.saturating_sub(row_pat_start);
        let val_start = symbolic.col_ptrs_for_values()[s].zx();
        let val_end = symbolic.col_ptrs_for_values()[s + 1].zx();
        let vals = &l_values[val_start..val_end];
        for j in 0..ncols {
            let diag = vals[j * nrows + j];
            if !diag.is_finite() || diag <= 0.0 {
                return Err(format!(
                    "Sparse Cholesky produced invalid diagonal entry at supernode {s}, local col {j}: {diag}"
                ));
            }
            sum_log_diag += diag.ln();
        }
    }
    Ok(2.0 * sum_log_diag)
}

fn build_solve_workspace_req(
    symbolic: &supernodal::SymbolicSupernodalCholesky<usize>,
    n: usize,
    n_rhs: usize,
) -> Result<StackReq, String> {
    StackReq::try_all_of([
        faer::perm::permute_rows_in_place_req::<usize, f64>(n, n_rhs)
            .map_err(|e| format!("build sparse Cholesky permute StackReq failed: {e}"))?,
        symbolic
            .solve_in_place_req::<f64>(n_rhs)
            .map_err(|e| format!("build sparse Cholesky solve StackReq failed: {e}"))?,
    ])
    .map_err(|e| format!("combine sparse Cholesky solve StackReq failed: {e}"))
}

fn build_sparse_cholesky_analysis<V: SparseGrmCscView + ?Sized>(
    csc: &V,
) -> Result<
    (
        usize,
        Vec<usize>,
        Vec<usize>,
        supernodal::SymbolicSupernodalCholesky<usize>,
        Vec<usize>,
        Vec<usize>,
        Vec<f64>,
        Vec<usize>,
    ),
    String,
> {
    validate_sparse_grm_csc_view(csc)?;
    let n = csc.n_samples();
    let col_ptr = csc
        .col_ptr()
        .iter()
        .copied()
        .map(|x| usize::try_from(x).map_err(|_| "col_ptr entry does not fit usize".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let row_indices = csc
        .row_indices()
        .iter()
        .copied()
        .map(|x| usize::try_from(x).map_err(|_| "row index does not fit usize".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let base_lower = SparseColMatRef::<usize, f64>::new(
        unsafe { SymbolicSparseColMatRef::new_unchecked(n, n, &col_ptr, None, &row_indices) },
        csc.values(),
    );
    let a_nnz = row_indices.len();

    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];
    let mut amd_mem = GlobalPodBuffer::try_new(
        amd::order_req::<usize>(n, a_nnz)
            .map_err(|e| format!("build sparse AMD StackReq failed: {e}"))?,
    )
    .map_err(|e| format!("allocate sparse AMD workspace failed: {e}"))?;
    amd::order(
        &mut perm,
        &mut perm_inv,
        base_lower.symbolic(),
        amd::Control::default(),
        PodStack::new(&mut amd_mem),
    )
    .map_err(|e| format!("sparse AMD ordering failed: {e}"))?;
    let perm_ref = unsafe { faer::perm::PermRef::new_unchecked(&perm, &perm_inv) };

    let mut a_perm_col_ptr = vec![0usize; n + 1];
    let mut a_perm_row_indices = vec![0usize; a_nnz];
    let mut a_perm_values = vec![0.0_f64; a_nnz];
    let mut perm_mem = GlobalPodBuffer::try_new(
        faer::sparse::utils::permute_hermitian_req::<usize>(n)
            .map_err(|e| format!("build sparse permutation StackReq failed: {e}"))?,
    )
    .map_err(|e| format!("allocate sparse permutation workspace failed: {e}"))?;
    faer::sparse::utils::permute_hermitian::<usize, f64>(
        &mut a_perm_values,
        &mut a_perm_col_ptr,
        &mut a_perm_row_indices,
        base_lower,
        perm_ref,
        Side::Lower,
        Side::Lower,
        PodStack::new(&mut perm_mem),
    );
    let a_perm_lower = SparseColMatRef::<usize, f64>::new(
        unsafe {
            SymbolicSparseColMatRef::new_unchecked(n, n, &a_perm_col_ptr, None, &a_perm_row_indices)
        },
        a_perm_values.as_slice(),
    );
    let a_perm_upper = a_perm_lower
        .transpose()
        .symbolic()
        .to_col_major()
        .map_err(|e| format!("convert permuted sparse matrix to upper CSC failed: {e}"))?;

    let mut symbolic_mem = GlobalPodBuffer::try_new(
        StackReq::try_any_of([
            simplicial::prefactorize_symbolic_cholesky_req::<usize>(n, a_nnz)
                .map_err(|e| format!("build sparse prefactorize StackReq failed: {e}"))?,
            supernodal::factorize_supernodal_symbolic_cholesky_req::<usize>(n)
                .map_err(|e| format!("build sparse symbolic StackReq failed: {e}"))?,
        ])
        .map_err(|e| format!("combine sparse symbolic StackReq failed: {e}"))?,
    )
    .map_err(|e| format!("allocate sparse symbolic Cholesky workspace failed: {e}"))?;
    let mut stack = PodStack::new(&mut symbolic_mem);
    let mut etree = vec![0isize; n];
    let mut col_counts = vec![0usize; n];
    simplicial::prefactorize_symbolic_cholesky(
        &mut etree,
        &mut col_counts,
        a_perm_upper.as_ref(),
        stack.rb_mut(),
    );
    let symbolic = supernodal::factorize_supernodal_symbolic(
        a_perm_upper.as_ref(),
        unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
        &col_counts,
        stack.rb_mut(),
        faer::sparse::linalg::SymbolicSupernodalParams {
            relax: Some(&[(usize::MAX, 1.0)]),
        },
    )
    .map_err(|e| format!("sparse symbolic Cholesky analysis failed: {e}"))?;

    let diag_positions =
        find_diag_positions_from_symbolic(n, &a_perm_col_ptr, &a_perm_row_indices)?;
    Ok((
        n,
        perm,
        perm_inv,
        symbolic,
        a_perm_col_ptr,
        a_perm_row_indices,
        a_perm_values,
        diag_positions,
    ))
}

#[derive(Debug)]
pub struct SparseJxgrmCholeskyAnalysis {
    n: usize,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
    symbolic: Arc<supernodal::SymbolicSupernodalCholesky<usize>>,
    base_perm_col_ptr: Vec<usize>,
    base_perm_row_indices: Vec<usize>,
    base_perm_values: Vec<f64>,
    diag_positions: Vec<usize>,
}

impl SparseJxgrmCholeskyAnalysis {
    pub fn dim(&self) -> usize {
        self.n
    }

    pub fn factor_nnz_estimate(&self) -> usize {
        self.symbolic.len_values()
    }

    pub fn diag_stats_scaled(&self, scale: f64, diag_add: f64) -> Result<(f64, f64, f64), String> {
        if self.n == 0 {
            return Err("Sparse JXGRM diagonal stats require n_samples > 0".to_string());
        }
        let mut sum_abs = 0.0_f64;
        let mut min_diag = f64::INFINITY;
        let mut max_diag = f64::NEG_INFINITY;
        for &idx in self.diag_positions.iter() {
            let diag = self.base_perm_values.get(idx).copied().ok_or_else(|| {
                format!("Sparse JXGRM diagonal position out of bounds: idx={idx}")
            })? * scale
                + diag_add;
            if !diag.is_finite() {
                return Err(format!(
                    "Sparse JXGRM diagonal contains non-finite value at permuted index {idx}"
                ));
            }
            let abs_diag = diag.abs();
            sum_abs += abs_diag;
            if diag < min_diag {
                min_diag = diag;
            }
            if diag > max_diag {
                max_diag = diag;
            }
        }
        Ok((
            (sum_abs / (self.n as f64)).max(1e-30_f64),
            min_diag,
            max_diag,
        ))
    }

    fn factorize_from_perm_values_with_parallelism_timed(
        &self,
        perm_values: &[f64],
        parallelism: Parallelism,
    ) -> Result<(SparseJxgrmCholesky, SparseFactorizeCoreTiming), String> {
        let total_t0 = Instant::now();
        let a_perm_lower = SparseColMatRef::<usize, f64>::new(
            unsafe {
                SymbolicSparseColMatRef::new_unchecked(
                    self.n,
                    self.n,
                    &self.base_perm_col_ptr,
                    None,
                    &self.base_perm_row_indices,
                )
            },
            perm_values,
        );
        let mut l_values = vec![0.0_f64; self.symbolic.len_values()];
        let numeric_t0 = Instant::now();
        let mut numeric_mem = GlobalPodBuffer::try_new(
            supernodal::factorize_supernodal_numeric_llt_req::<usize, f64>(
                &self.symbolic,
                parallelism,
            )
            .map_err(|e| format!("build sparse numeric LLT StackReq failed: {e}"))?,
        )
        .map_err(|e| format!("allocate sparse numeric Cholesky workspace failed: {e}"))?;
        supernodal::factorize_supernodal_numeric_llt::<usize, f64>(
            &mut l_values,
            a_perm_lower,
            LltRegularization::default(),
            &self.symbolic,
            parallelism,
            PodStack::new(&mut numeric_mem),
        )
        .map_err(|e| format!("sparse numeric LLT factorization failed: {e}"))?;
        let numeric_secs = numeric_t0.elapsed().as_secs_f64();
        let logdet_t0 = Instant::now();
        let logdet = supernodal_llt_logdet(&self.symbolic, &l_values)?;
        let logdet_secs = logdet_t0.elapsed().as_secs_f64();
        Ok((
            SparseJxgrmCholesky {
                n: self.n,
                perm: self.perm.clone(),
                perm_inv: self.perm_inv.clone(),
                symbolic: Arc::clone(&self.symbolic),
                l_values,
                logdet,
            },
            SparseFactorizeCoreTiming {
                numeric_secs,
                logdet_secs,
                total_secs: total_t0.elapsed().as_secs_f64(),
            },
        ))
    }

    pub fn factorize_diag_shifted(&self, diag_shift: f64) -> Result<SparseJxgrmCholesky, String> {
        if !diag_shift.is_finite() || diag_shift < 0.0 {
            return Err(format!(
                "Sparse Cholesky diagonal shift must be finite and >= 0, got {diag_shift}"
            ));
        }
        let prep_t0 = Instant::now();
        let mut perm_values = self.base_perm_values.clone();
        if diag_shift != 0.0 {
            for &idx in self.diag_positions.iter() {
                perm_values[idx] += diag_shift;
            }
        }
        let prep_secs = prep_t0.elapsed().as_secs_f64();
        let parallelism = Parallelism::None;
        let parallelism_desc = match &parallelism {
            Parallelism::None => "none".to_string(),
            Parallelism::Rayon(n_threads) => format!("rayon({n_threads})"),
        };
        let (factor, timing) =
            self.factorize_from_perm_values_with_parallelism_timed(&perm_values, parallelism)?;
        if sparse_factor_timing_enabled() {
            emit_sparse_factor_timing(
                "diag_shifted",
                prep_secs,
                timing.numeric_secs,
                timing.logdet_secs,
                0.0,
                prep_secs + timing.total_secs,
                self.n,
                self.base_perm_values.len(),
                factor.factor_nnz(),
                self.diag_positions.len(),
                &parallelism_desc,
            );
        }
        Ok(factor)
    }

    pub fn base_perm_values(&self) -> &[f64] {
        &self.base_perm_values
    }

    pub fn factorize_k_plus_lambda_i(&self, lambda: f64) -> Result<SparseJxgrmCholesky, String> {
        self.factorize_diag_shifted(lambda)
    }

    pub fn factorize_scaled_plus_diag(
        &self,
        scale: f64,
        diag: &[f64],
    ) -> Result<SparseJxgrmCholesky, String> {
        if !scale.is_finite() || scale < 0.0 {
            return Err(format!(
                "Sparse Cholesky scaled factorization requires finite non-negative scale, got {scale}"
            ));
        }
        if diag.len() != self.n {
            return Err(format!(
                "Sparse Cholesky scaled factorization diagonal length mismatch: got {}, expected {}",
                diag.len(),
                self.n
            ));
        }
        if diag.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(
                "Sparse Cholesky scaled factorization requires finite non-negative diagonal entries"
                    .to_string(),
            );
        }
        let prep_t0 = Instant::now();
        let mut perm_values = self.base_perm_values.clone();
        if scale != 1.0 {
            for value in perm_values.iter_mut() {
                *value *= scale;
            }
        }
        for (perm_col, &idx) in self.diag_positions.iter().enumerate() {
            let orig_col = self.perm.get(perm_col).copied().ok_or_else(|| {
                format!("Sparse Cholesky permutation index out of bounds: {perm_col}")
            })?;
            perm_values[idx] += diag[orig_col];
        }
        let prep_secs = prep_t0.elapsed().as_secs_f64();
        let parallelism = Parallelism::None;
        let parallelism_desc = match &parallelism {
            Parallelism::None => "none".to_string(),
            Parallelism::Rayon(n_threads) => format!("rayon({n_threads})"),
        };
        let (factor, timing) =
            self.factorize_from_perm_values_with_parallelism_timed(&perm_values, parallelism)?;
        if sparse_factor_timing_enabled() {
            emit_sparse_factor_timing(
                "scaled_plus_diag",
                prep_secs,
                timing.numeric_secs,
                timing.logdet_secs,
                0.0,
                prep_secs + timing.total_secs,
                self.n,
                self.base_perm_values.len(),
                factor.factor_nnz(),
                self.diag_positions.len(),
                &parallelism_desc,
            );
        }
        Ok(factor)
    }

    /// Like factorize_k_plus_lambda_i but writes into `buf` instead of cloning
    /// base_perm_values internally.  Caller must ensure `buf.len() ==
    /// self.base_perm_values.len()`.
    pub fn factorize_k_plus_lambda_i_buffered(
        &self,
        lambda: f64,
        buf: &mut [f64],
    ) -> Result<SparseJxgrmCholesky, String> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(format!(
                "Sparse Cholesky buffered factorization requires finite non-negative lambda, got {lambda}"
            ));
        }
        if buf.len() != self.base_perm_values.len() {
            return Err(format!(
                "Sparse Cholesky buffered factorization buffer length mismatch: got {}, expected {}",
                buf.len(),
                self.base_perm_values.len()
            ));
        }
        let prep_t0 = Instant::now();
        buf.copy_from_slice(&self.base_perm_values);
        if lambda != 0.0 {
            for &idx in self.diag_positions.iter() {
                buf[idx] += lambda;
            }
        }
        let prep_secs = prep_t0.elapsed().as_secs_f64();
        let parallelism = Parallelism::None;
        let parallelism_desc = match &parallelism {
            Parallelism::None => "none".to_string(),
            Parallelism::Rayon(n_threads) => format!("rayon({n_threads})"),
        };
        let (result, timing) =
            self.factorize_from_perm_values_with_parallelism_timed(buf, parallelism)?;
        let cleanup_t0 = Instant::now();
        buf.copy_from_slice(&self.base_perm_values);
        let cleanup_secs = cleanup_t0.elapsed().as_secs_f64();
        if sparse_factor_timing_enabled() {
            emit_sparse_factor_timing(
                "buffered_lambda",
                prep_secs,
                timing.numeric_secs,
                timing.logdet_secs,
                cleanup_secs,
                prep_secs + timing.total_secs + cleanup_secs,
                self.n,
                self.base_perm_values.len(),
                result.factor_nnz(),
                self.diag_positions.len(),
                &parallelism_desc,
            );
        }
        Ok(result)
    }

    pub fn factorize_k_plus_lambda_i_buffered_parallel(
        &self,
        lambda: f64,
        buf: &mut [f64],
        threads: usize,
    ) -> Result<SparseJxgrmCholesky, String> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(format!(
                "Sparse Cholesky buffered parallel factorization requires finite non-negative lambda, got {lambda}"
            ));
        }
        if buf.len() != self.base_perm_values.len() {
            return Err(format!(
                "Sparse Cholesky buffered parallel factorization buffer length mismatch: got {}, expected {}",
                buf.len(),
                self.base_perm_values.len()
            ));
        }
        let prep_t0 = Instant::now();
        buf.copy_from_slice(&self.base_perm_values);
        if lambda != 0.0 {
            for &idx in self.diag_positions.iter() {
                buf[idx] += lambda;
            }
        }
        let prep_secs = prep_t0.elapsed().as_secs_f64();
        let parallelism = sparse_numeric_parallelism(threads);
        let parallelism_desc = match &parallelism {
            Parallelism::None => "none".to_string(),
            Parallelism::Rayon(n_threads) => format!("rayon({n_threads})"),
        };
        let (result, timing) =
            self.factorize_from_perm_values_with_parallelism_timed(buf, parallelism)?;
        let cleanup_t0 = Instant::now();
        buf.copy_from_slice(&self.base_perm_values);
        let cleanup_secs = cleanup_t0.elapsed().as_secs_f64();
        if sparse_factor_timing_enabled() {
            emit_sparse_factor_timing(
                "buffered_lambda_parallel",
                prep_secs,
                timing.numeric_secs,
                timing.logdet_secs,
                cleanup_secs,
                prep_secs + timing.total_secs + cleanup_secs,
                self.n,
                self.base_perm_values.len(),
                result.factor_nnz(),
                self.diag_positions.len(),
                &parallelism_desc,
            );
        }
        Ok(result)
    }

    pub fn factorize_sigma_g2_k_plus_sigma_e2_i(
        &self,
        sigma_g2: f64,
        sigma_e2: f64,
    ) -> Result<SparseJxgrmCholesky, String> {
        self.factorize_sigma_g2_k_plus_sigma_e2_i_with_diag_shift(sigma_g2, sigma_e2, 0.0)
    }

    pub fn factorize_sigma_g2_k_plus_sigma_e2_i_with_diag_shift(
        &self,
        sigma_g2: f64,
        sigma_e2: f64,
        diag_shift: f64,
    ) -> Result<SparseJxgrmCholesky, String> {
        if !sigma_g2.is_finite() || sigma_g2 < 0.0 || !sigma_e2.is_finite() || sigma_e2 < 0.0 {
            return Err(format!(
                "Sparse Cholesky requires finite non-negative variance components, got sigma_g2={sigma_g2}, sigma_e2={sigma_e2}"
            ));
        }
        if !diag_shift.is_finite() || diag_shift < 0.0 {
            return Err(format!(
                "Sparse Cholesky diagonal shift must be finite and >= 0, got {diag_shift}"
            ));
        }
        let prep_t0 = Instant::now();
        let mut perm_values = self.base_perm_values.clone();
        for value in perm_values.iter_mut() {
            *value *= sigma_g2;
        }
        if sigma_e2 != 0.0 || diag_shift != 0.0 {
            for &idx in self.diag_positions.iter() {
                perm_values[idx] += sigma_e2 + diag_shift;
            }
        }
        let prep_secs = prep_t0.elapsed().as_secs_f64();
        let parallelism = Parallelism::None;
        let parallelism_desc = match &parallelism {
            Parallelism::None => "none".to_string(),
            Parallelism::Rayon(n_threads) => format!("rayon({n_threads})"),
        };
        let (factor, timing) =
            self.factorize_from_perm_values_with_parallelism_timed(&perm_values, parallelism)?;
        if sparse_factor_timing_enabled() {
            emit_sparse_factor_timing(
                "sigma_shifted",
                prep_secs,
                timing.numeric_secs,
                timing.logdet_secs,
                0.0,
                prep_secs + timing.total_secs,
                self.n,
                self.base_perm_values.len(),
                factor.factor_nnz(),
                self.diag_positions.len(),
                &parallelism_desc,
            );
        }
        Ok(factor)
    }

    pub fn factorize_sigma_g2_k_plus_sigma_e2_i_with_diag_shift_parallel(
        &self,
        sigma_g2: f64,
        sigma_e2: f64,
        diag_shift: f64,
        threads: usize,
    ) -> Result<SparseJxgrmCholesky, String> {
        if !sigma_g2.is_finite() || sigma_g2 < 0.0 || !sigma_e2.is_finite() || sigma_e2 < 0.0 {
            return Err(format!(
                "Sparse Cholesky requires finite non-negative variance components, got sigma_g2={sigma_g2}, sigma_e2={sigma_e2}"
            ));
        }
        if !diag_shift.is_finite() || diag_shift < 0.0 {
            return Err(format!(
                "Sparse Cholesky diagonal shift must be finite and >= 0, got {diag_shift}"
            ));
        }
        let prep_t0 = Instant::now();
        let mut perm_values = self.base_perm_values.clone();
        for value in perm_values.iter_mut() {
            *value *= sigma_g2;
        }
        if sigma_e2 != 0.0 || diag_shift != 0.0 {
            for &idx in self.diag_positions.iter() {
                perm_values[idx] += sigma_e2 + diag_shift;
            }
        }
        let prep_secs = prep_t0.elapsed().as_secs_f64();
        let parallelism = sparse_numeric_parallelism(threads);
        let parallelism_desc = match &parallelism {
            Parallelism::None => "none".to_string(),
            Parallelism::Rayon(n_threads) => format!("rayon({n_threads})"),
        };
        let (factor, timing) =
            self.factorize_from_perm_values_with_parallelism_timed(&perm_values, parallelism)?;
        if sparse_factor_timing_enabled() {
            emit_sparse_factor_timing(
                "sigma_shifted_parallel",
                prep_secs,
                timing.numeric_secs,
                timing.logdet_secs,
                0.0,
                prep_secs + timing.total_secs,
                self.n,
                self.base_perm_values.len(),
                factor.factor_nnz(),
                self.diag_positions.len(),
                &parallelism_desc,
            );
        }
        Ok(factor)
    }
}

pub struct SparseJxgrmSolveWorkspace {
    n: usize,
    n_rhs_capacity: usize,
    mem: GlobalPodBuffer,
}

impl SparseJxgrmSolveWorkspace {
    pub fn n_rhs_capacity(&self) -> usize {
        self.n_rhs_capacity
    }
}

#[derive(Debug)]
pub struct SparseJxgrmCholesky {
    n: usize,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
    symbolic: Arc<supernodal::SymbolicSupernodalCholesky<usize>>,
    l_values: Vec<f64>,
    logdet: f64,
}

impl SparseJxgrmCholesky {
    pub fn dim(&self) -> usize {
        self.n
    }

    pub fn factor_nnz(&self) -> usize {
        self.l_values.len()
    }

    pub fn logdet(&self) -> f64 {
        self.logdet
    }

    pub fn make_solve_workspace(
        &self,
        n_rhs_capacity: usize,
    ) -> Result<SparseJxgrmSolveWorkspace, String> {
        if n_rhs_capacity == 0 {
            return Err("Sparse Cholesky solve workspace requires n_rhs_capacity > 0".to_string());
        }
        let req = build_solve_workspace_req(&self.symbolic, self.n, n_rhs_capacity)?;
        let mem = GlobalPodBuffer::try_new(req)
            .map_err(|e| format!("allocate sparse Cholesky solve workspace failed: {e}"))?;
        Ok(SparseJxgrmSolveWorkspace {
            n: self.n,
            n_rhs_capacity,
            mem,
        })
    }

    pub fn solve_in_place_with_workspace(
        &self,
        rhs_col_major: &mut [f64],
        n_rhs: usize,
        workspace: &mut SparseJxgrmSolveWorkspace,
    ) -> Result<(), String> {
        if workspace.n != self.n {
            return Err(format!(
                "Sparse Cholesky solve workspace dimension mismatch: workspace n={}, factor n={}",
                workspace.n, self.n
            ));
        }
        if n_rhs == 0 || n_rhs > workspace.n_rhs_capacity {
            return Err(format!(
                "Sparse Cholesky solve requested n_rhs={} but workspace capacity is {}",
                n_rhs, workspace.n_rhs_capacity
            ));
        }
        if rhs_col_major.len() != self.n.saturating_mul(n_rhs) {
            return Err(format!(
                "Sparse Cholesky solve RHS length mismatch: got {}, expected {} for n={} and n_rhs={n_rhs}",
                rhs_col_major.len(),
                self.n.saturating_mul(n_rhs),
                self.n
            ));
        }
        let mut rhs = mat::from_column_major_slice_mut::<f64>(rhs_col_major, self.n, n_rhs);
        let perm = unsafe { faer::perm::PermRef::new_unchecked(&self.perm, &self.perm_inv) };
        let mut stack = PodStack::new(&mut workspace.mem);
        let llt =
            supernodal::SupernodalLltRef::<'_, usize, f64>::new(&self.symbolic, &self.l_values);
        faer::perm::permute_rows_in_place(rhs.rb_mut(), perm, stack.rb_mut());
        llt.solve_in_place_with_conj(Conj::No, rhs.rb_mut(), Parallelism::None, stack.rb_mut());
        faer::perm::permute_rows_in_place(rhs.rb_mut(), perm.inverse(), stack.rb_mut());
        Ok(())
    }

    /// Solve V·X = B for n_rhs right-hand sides, tiling columns across `workspaces`
    /// (one per tile).  When `pool` is provided, tiles are dispatched on that
    /// pool (via `pool.install` + `par_iter_mut`); otherwise the global rayon
    /// pool is used.
    ///
    /// The L factor is shared read-only; all tiles traverse the same
    /// elimination tree, so L1/L2/L3 cache is naturally shared.
    ///
    /// Each workspace element must have `n_rhs_capacity >= (cols in that tile)`;
    /// work items are also validated for `workspace.n == self.n`.
    pub fn solve_in_place_tiled(
        &self,
        rhs_col_major: &mut [f64],
        n_rhs: usize,
        workspaces: &mut [SparseJxgrmSolveWorkspace],
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<(), String> {
        const MIN_TILE_COLS: usize = 32;
        let n = self.n;
        if n_rhs == 0 {
            return Err("n_rhs must be > 0".to_string());
        }
        if rhs_col_major.len() != n.saturating_mul(n_rhs) {
            return Err(format!(
                "tiled solve RHS length mismatch: got {}, expected {}",
                rhs_col_major.len(),
                n.saturating_mul(n_rhs)
            ));
        }
        if workspaces.is_empty() || n_rhs < MIN_TILE_COLS {
            // Pathological or tiny RHS – allocate one workspace and solve sequentially.
            let mut ws = self.make_solve_workspace(n_rhs.max(1))?;
            return self.solve_in_place_with_workspace(rhs_col_major, n_rhs, &mut ws);
        }

        let nt = workspaces.len();
        let tile_cols = n_rhs.div_ceil(nt);
        let n_rows = self.n;

        // Partition the RHS slice into non-overlapping per-tile &mut slices,
        // paired with their workspace by index (par_iter_mut provides distinct
        // &mut access to each workspace element).
        let mut run = || -> Result<(), String> {
            use rayon::prelude::*;
            // Pair each workspace with its column-range metadata.
            workspaces
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(t, ws)| {
                    let col_start = t.saturating_mul(tile_cols);
                    let col_end = (col_start.saturating_add(tile_cols)).min(n_rhs);
                    let cols_here = col_end.saturating_sub(col_start);
                    if cols_here == 0 {
                        return Ok(());
                    }
                    // RHS is column-major: tile `t` starts at byte offset
                    // col_start * n_rows * sizeof(f64).
                    let byte_offset = col_start
                        .saturating_mul(n_rows)
                        .saturating_mul(std::mem::size_of::<f64>());
                    let len = cols_here.saturating_mul(n_rows);
                    // SAFETY: Tiles partition `rhs_col_major` without overlap;
                    // each rayon worker gets a unique `t` and therefore a
                    // disjoint portion of the slice.  The raw pointer is
                    // derived from `rhs_col_major` (valid, aligned, non-null)
                    // and the region stays within bounds.
                    let tile_slice = unsafe {
                        let ptr = rhs_col_major.as_ptr().byte_add(byte_offset) as *mut f64;
                        std::slice::from_raw_parts_mut(ptr, len)
                    };
                    self.solve_in_place_with_workspace(tile_slice, cols_here, ws)
                })
        };

        if let Some(tp) = pool {
            tp.install(move || run())
        } else {
            run()
        }
    }

    pub fn solve_in_place(&self, rhs_col_major: &mut [f64], n_rhs: usize) -> Result<(), String> {
        let mut workspace = self.make_solve_workspace(n_rhs.max(1))?;
        self.solve_in_place_with_workspace(rhs_col_major, n_rhs, &mut workspace)
    }

    pub fn solve_vec(&self, rhs: &[f64]) -> Result<Vec<f64>, String> {
        if rhs.len() != self.n {
            return Err(format!(
                "Sparse Cholesky solve_vec expects rhs length {}, got {}",
                self.n,
                rhs.len()
            ));
        }
        let mut out = rhs.to_vec();
        self.solve_in_place(&mut out, 1)?;
        Ok(out)
    }
}

pub(crate) fn sparse_cholesky_analyze_jxgrm_view<V: SparseGrmCscView + ?Sized>(
    csc: &V,
) -> Result<SparseJxgrmCholeskyAnalysis, String> {
    let (
        n,
        perm,
        perm_inv,
        symbolic,
        base_perm_col_ptr,
        base_perm_row_indices,
        base_perm_values,
        diag_positions,
    ) = build_sparse_cholesky_analysis(csc)?;
    Ok(SparseJxgrmCholeskyAnalysis {
        n,
        perm,
        perm_inv,
        symbolic: Arc::new(symbolic),
        base_perm_col_ptr,
        base_perm_row_indices,
        base_perm_values,
        diag_positions,
    })
}

pub fn sparse_cholesky_analyze_jxgrm_csc(
    csc: &SparseGrmCsc,
) -> Result<SparseJxgrmCholeskyAnalysis, String> {
    sparse_cholesky_analyze_jxgrm_view(csc)
}

pub fn sparse_cholesky_from_jxgrm_csc(
    csc: &SparseGrmCsc,
    diag_shift: f64,
) -> Result<SparseJxgrmCholesky, String> {
    sparse_cholesky_analyze_jxgrm_csc(csc)?.factorize_diag_shifted(diag_shift)
}

pub fn sparse_cholesky_from_jxgrm_path(
    path: &str,
    diag_shift: f64,
) -> Result<SparseJxgrmCholesky, String> {
    sparse_cholesky_analyze_jxgrm_path(path)?.factorize_diag_shifted(diag_shift)
}

pub fn sparse_cholesky_analyze_jxgrm_path(
    path: &str,
) -> Result<SparseJxgrmCholeskyAnalysis, String> {
    let csc = MmapSparseGrmCsc::open(path)?;
    sparse_cholesky_analyze_jxgrm_view(&csc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spgrm::write_sparse_grm_csc;

    fn approx_eq_slice(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (i, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (a - b).abs() <= tol,
                "mismatch at {i}: lhs={a}, rhs={b}, tol={tol}"
            );
        }
    }

    #[test]
    fn sparse_jxgrm_cholesky_solves_small_spd_system() {
        let csc = SparseGrmCsc {
            n_samples: 3,
            nnz: 5,
            col_ptr: vec![0, 2, 4, 5],
            row_indices: vec![0, 1, 1, 2, 2],
            values: vec![4.0, 1.0, 3.0, 1.0, 2.0],
        };
        let chol = sparse_cholesky_from_jxgrm_csc(&csc, 0.0).unwrap();
        let rhs = vec![6.0, 10.0, 8.0];
        let sol = chol.solve_vec(&rhs).unwrap();
        approx_eq_slice(&sol, &[1.0, 2.0, 3.0], 1e-10);
    }

    #[test]
    fn sparse_jxgrm_csc_rejects_upper_triangle_entries() {
        let bad = SparseGrmCsc {
            n_samples: 2,
            nnz: 3,
            col_ptr: vec![0, 2, 3],
            row_indices: vec![0, 1, 0],
            values: vec![1.0, 0.1, 1.0],
        };
        let err = sparse_cholesky_from_jxgrm_csc(&bad, 0.0).unwrap_err();
        assert!(err.contains("lower triangle"));
    }

    #[test]
    fn subset_sparse_grm_csc_reorders_columns_into_sorted_lower_csc() {
        let csc = SparseGrmCsc {
            n_samples: 3,
            nnz: 6,
            col_ptr: vec![0, 3, 5, 6],
            row_indices: vec![0, 1, 2, 1, 2, 2],
            values: vec![1.0, 0.2, 0.3, 1.0, 0.4, 1.0],
        };
        let subset = subset_sparse_grm_csc(&csc, &[2, 0, 1]).unwrap();
        assert_eq!(subset.n_samples, 3);
        assert_eq!(subset.col_ptr, vec![0, 3, 5, 6]);
        assert_eq!(subset.row_indices, vec![0, 1, 2, 1, 2, 2]);
        approx_eq_slice(&subset.values, &[1.0, 0.3, 0.4, 1.0, 0.2, 1.0], 1e-12);
    }

    #[test]
    fn sparse_jxgrm_solve_workspace_reuses_buffer_for_multi_rhs() {
        let csc = SparseGrmCsc {
            n_samples: 3,
            nnz: 5,
            col_ptr: vec![0, 2, 4, 5],
            row_indices: vec![0, 1, 1, 2, 2],
            values: vec![4.0, 1.0, 3.0, 1.0, 2.0],
        };
        let analysis = sparse_cholesky_analyze_jxgrm_csc(&csc).unwrap();
        let chol = analysis.factorize_k_plus_lambda_i(0.5).unwrap();
        let mut ws = chol.make_solve_workspace(2).unwrap();
        let mut rhs = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        chol.solve_in_place_with_workspace(&mut rhs, 2, &mut ws)
            .unwrap();
        let mut rhs2 = vec![1.0, 2.0, 3.0];
        chol.solve_in_place_with_workspace(&mut rhs2, 1, &mut ws)
            .unwrap();
        assert!(rhs.iter().all(|v| v.is_finite()));
        assert!(rhs2.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mmap_sparse_jxgrm_uses_borrowed_values_for_padded_odd_nnz_file() {
        let csc = SparseGrmCsc {
            n_samples: 2,
            nnz: 3,
            col_ptr: vec![0, 2, 3],
            row_indices: vec![0, 1, 1],
            values: vec![1.0, 0.25, 0.9],
        };
        let mut out_path = std::env::temp_dir();
        out_path.push(format!(
            "janusx_cholesky_pad_test_{}_{}.spgrm",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let out_str = out_path.to_string_lossy().to_string();
        write_sparse_grm_csc(&out_str, &csc).unwrap();
        let mmap_csc = MmapSparseGrmCsc::open(&out_str).unwrap();
        assert_eq!(mmap_csc.values_offset % JXGRM_VALUES_ALIGN_BYTES, 0);
        approx_eq_slice(mmap_csc.values(), csc.values.as_slice(), 0.0);
        let _ = std::fs::remove_file(out_str);
    }

    #[test]
    fn mmap_sparse_jxgrm_validation_cache_skips_full_revalidate_on_reopen() {
        let csc = SparseGrmCsc {
            n_samples: 4,
            nnz: 7,
            col_ptr: vec![0, 2, 4, 6, 7],
            row_indices: vec![0, 1, 1, 2, 2, 3, 3],
            values: vec![1.0, 0.1, 1.0, 0.2, 1.0, 0.3, 1.0],
        };
        let mut out_path = std::env::temp_dir();
        out_path.push(format!(
            "janusx_cholesky_validate_cache_test_{}_{}.spgrm",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let out_str = out_path.to_string_lossy().to_string();
        write_sparse_grm_csc(&out_str, &csc).unwrap();

        let mut first_progress = Vec::<(usize, usize)>::new();
        let _first = MmapSparseGrmCsc::open_with_progress(&out_str, |done, total| {
            first_progress.push((done, total));
            Ok(())
        })
        .unwrap();
        let mut second_progress = Vec::<(usize, usize)>::new();
        let _second = MmapSparseGrmCsc::open_with_progress(&out_str, |done, total| {
            second_progress.push((done, total));
            Ok(())
        })
        .unwrap();

        assert!(first_progress.len() > 1);
        assert_eq!(second_progress.len(), 1);
        assert_eq!(second_progress[0], (csc.n_samples + 1, csc.n_samples + 1));
        let _ = std::fs::remove_file(out_str);
    }
}
