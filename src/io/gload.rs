//! Unified genotype-loading abstraction.
//!
//! Two back-ends implement [`GenotypeMatrix`]:
//! * [`BedMmapMatrix`] — the BED file is memory-mapped; genotype rows are
//!   zero-copy slices into the mmap, indexed by source SNP position.
//! * [`PackedBedMatrix`] — the filtered BED payload is owned in-memory
//!   (`Arc<[u8]>`), packed contiguously across kept SNPs.
//!
//! [`GlobalStats`] carries per-marker statistics (MAF / miss / flip) computed
//! once at load time.  Together with a [`GenotypeMatrix`] they form a
//! self-contained [`UnifiedInput`] that can be passed to downstream scanners.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use memmap2::{Mmap, MmapOptions};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::bedmath::{decode_mean_imputed_additive_packed_block_rows_f32, packed_byte_lut};
use crate::decode::{decode_centered_block_packed_model_f32, PackedGeneticModel};
use crate::gfcore;
use crate::gstats::{compute_bed_filter_meta, BedFilterMetaMode};

const BED_HEADER_LEN: usize = 3;

#[inline]
fn gload_force_sequential_owned() -> bool {
    match std::env::var("JX_GLOAD_FORCE_SEQUENTIAL") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => false,
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PlinkBedHeaderInfo {
    pub(crate) bytes_per_snp: usize,
    pub(crate) file_size: usize,
    pub(crate) payload_size: usize,
    pub(crate) n_snps_total: usize,
}

#[inline]
fn validate_plink_bed_header(header: [u8; BED_HEADER_LEN]) -> Result<(), String> {
    if header != [0x6C, 0x1B, 0x01] {
        return Err("Only SNP-major BED supported".to_string());
    }
    Ok(())
}

pub(crate) fn read_plink_bed_header_info(
    bed_path: &Path,
    n_samples_full: usize,
) -> Result<PlinkBedHeaderInfo, String> {
    if n_samples_full == 0 {
        return Err("no samples found in BED".to_string());
    }
    let mut file = File::open(bed_path).map_err(|e| format!("open {}: {e}", bed_path.display()))?;
    let mut header = [0u8; BED_HEADER_LEN];
    file.read_exact(&mut header)
        .map_err(|e| format!("read {} header: {e}", bed_path.display()))?;
    validate_plink_bed_header(header)?;
    let bytes_per_snp = n_samples_full.div_ceil(4);
    let file_size = usize::try_from(
        file.metadata()
            .map_err(|e| format!("stat {}: {e}", bed_path.display()))?
            .len(),
    )
    .map_err(|_| format!("BED file size overflow: {}", bed_path.display()))?;
    if file_size < BED_HEADER_LEN {
        return Err(format!("BED file is too small: {}", bed_path.display()));
    }
    let payload_size = file_size - BED_HEADER_LEN;
    if bytes_per_snp == 0 || payload_size % bytes_per_snp != 0 {
        return Err(format!(
            "BED payload size mismatch: payload={payload_size}, bytes_per_snp={bytes_per_snp}"
        ));
    }
    Ok(PlinkBedHeaderInfo {
        bytes_per_snp,
        file_size,
        payload_size,
        n_snps_total: payload_size / bytes_per_snp,
    })
}

pub(crate) fn open_plink_bed_reader_after_header(
    bed_path: &Path,
) -> Result<BufReader<File>, String> {
    let file = File::open(bed_path).map_err(|e| format!("open {}: {e}", bed_path.display()))?;
    let mut reader = BufReader::new(file);
    let mut header = [0u8; BED_HEADER_LEN];
    reader
        .read_exact(&mut header)
        .map_err(|e| format!("read {} header: {e}", bed_path.display()))?;
    validate_plink_bed_header(header)?;
    Ok(reader)
}

pub(crate) fn mmap_readonly_file(path: &Path) -> Result<Mmap, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", path.display()))
}

#[inline]
fn file_size_usize(file: &File, path: &Path) -> Result<usize, String> {
    usize::try_from(
        file.metadata()
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len(),
    )
    .map_err(|_| format!("file size overflow: {}", path.display()))
}

fn load_file_owned_range_from_file(
    file: &File,
    path: &Path,
    offset: usize,
    expected_size: usize,
) -> Result<Vec<u8>, String> {
    let file_size = file_size_usize(file, path)?;
    let end = offset
        .checked_add(expected_size)
        .ok_or_else(|| format!("file range overflow for {}", path.display()))?;
    if end > file_size {
        return Err(format!(
            "file range out of bounds for {}: offset={} size={} file_size={}",
            path.display(),
            offset,
            expected_size,
            file_size
        ));
    }
    if expected_size == 0 {
        return Ok(Vec::new());
    }

    let threads = if gload_force_sequential_owned() {
        1usize
    } else {
        rayon::current_num_threads().max(1)
    };
    let parallel_threshold = 64usize << 20;
    let min_chunk = 8usize << 20;
    let mut buf = vec![0u8; expected_size];
    if threads <= 1 || expected_size < parallel_threshold {
        #[cfg(any(unix, windows))]
        {
            read_file_exact_at(file, &mut buf, offset as u64)?;
            return Ok(buf);
        }
        #[cfg(not(any(unix, windows)))]
        {
            use std::io::{Seek, SeekFrom};
            let mut reader_file = file
                .try_clone()
                .map_err(|e| format!("clone {} for sequential read: {e}", path.display()))?;
            reader_file
                .seek(SeekFrom::Start(offset as u64))
                .map_err(|e| format!("seek {}: {e}", path.display()))?;
            let mut reader = BufReader::new(reader_file);
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("read {}: {e}", path.display()))?;
            return Ok(buf);
        }
    }

    let nominal = expected_size.div_ceil(threads);
    let chunk_size = nominal.max(min_chunk);
    buf.par_chunks_mut(chunk_size)
        .enumerate()
        .try_for_each(|(chunk_idx, chunk)| {
            let chunk_offset = chunk_idx
                .checked_mul(chunk_size)
                .ok_or_else(|| format!("chunk offset overflow: {}", path.display()))?;
            let file_offset = offset
                .checked_add(chunk_offset)
                .ok_or_else(|| format!("file offset overflow: {}", path.display()))?
                as u64;
            read_file_exact_at(file, chunk, file_offset)
        })?;
    Ok(buf)
}

pub(crate) fn load_file_owned(path: &Path) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let file_size = file_size_usize(&file, path)?;
    load_file_owned_range_from_file(&file, path, 0, file_size)
}

pub(crate) fn load_file_owned_exact(path: &Path, expected_size: usize) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let file_size = file_size_usize(&file, path)?;
    if file_size != expected_size {
        return Err(format!(
            "file size mismatch for {}: expected {}, got {}",
            path.display(),
            expected_size,
            file_size
        ));
    }
    load_file_owned_range_from_file(&file, path, 0, expected_size)
}

pub(crate) fn load_file_owned_range_exact(
    path: &Path,
    offset: usize,
    expected_size: usize,
) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    load_file_owned_range_from_file(&file, path, offset, expected_size)
}

#[inline]
fn system_page_size() -> usize {
    #[cfg(unix)]
    {
        let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if ps > 0 {
            ps as usize
        } else {
            4096
        }
    }
    #[cfg(not(unix))]
    {
        4096
    }
}

#[cfg(unix)]
pub(crate) fn read_file_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<(), String> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
        .map_err(|e| format!("BED positioned read: {e}"))
}

#[cfg(windows)]
pub(crate) fn read_file_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<(), String> {
    use std::os::windows::fs::FileExt;

    let mut done = 0usize;
    while done < buf.len() {
        let n = file
            .seek_read(&mut buf[done..], offset + done as u64)
            .map_err(|e| format!("BED positioned read: {e}"))?;
        if n == 0 {
            return Err("BED positioned read: unexpected EOF".to_string());
        }
        done += n;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn read_file_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<(), String> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = file
        .try_clone()
        .map_err(|e| format!("BED file clone for positioned read: {e}"))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("BED seek: {e}"))?;
    file.read_exact(buf).map_err(|e| format!("BED read: {e}"))
}

/// Per-marker statistics computed once at load time.
///
/// All vectors are in **kept-marker order** (i.e. after MAF/missing/het
/// filtering).  `site_keep` is the only field indexed in the **original**
/// SNP order.
#[derive(Clone)]
#[allow(dead_code)]
pub struct GlobalStats {
    pub maf: Vec<f32>,
    pub miss: Vec<f32>,
    pub row_flip: Vec<bool>,
    pub row_source_indices: Vec<usize>,
    pub site_keep: Vec<bool>,
    pub n_samples_full: usize,
    pub n_markers_total: usize,
    pub bytes_per_snp: usize,
}

impl GlobalStats {
    #[inline]
    pub fn n_markers(&self) -> usize {
        if self.maf.is_empty() {
            self.n_markers_total
        } else {
            self.maf.len()
        }
    }

    /// Row-source slice for a block of kept markers, suitable for
    /// `decode_mean_imputed_additive_packed_block_rows_f32`.
    #[inline]
    pub fn row_source_slice(&self, row_start: usize, rows_here: usize) -> &[usize] {
        &self.row_source_indices[row_start..][..rows_here]
    }

    /// Row-flip slice for a block of kept markers.
    #[inline]
    pub fn row_flip_slice(&self, row_start: usize, rows_here: usize) -> &[bool] {
        &self.row_flip[row_start..][..rows_here]
    }

    /// MAF slice for a block of kept markers.
    #[inline]
    pub fn row_maf_slice(&self, row_start: usize, rows_here: usize) -> &[f32] {
        &self.maf[row_start..][..rows_here]
    }
}

// ---------------------------------------------------------------------------
// GenotypeMatrix trait
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub trait GenotypeMatrix: Send + Sync {
    fn n_samples_full(&self) -> usize;
    fn bytes_per_snp(&self) -> usize;

    /// Full packed payload.  For mmap-backed matrices this covers the
    /// **entire** BED (all original SNPs); callers must index through
    /// row_source_indices to reach kept rows.
    fn packed_flat(&self) -> &[u8];

    /// Raw packed bytes at a specific source SNP index (original BIM row).
    /// This is the primitive accessor for mmap matrices.
    fn source_row_bytes(&self, source_idx: usize) -> &[u8];

    /// Ensure the packed source rows for a kept-marker block are available to
    /// `source_row_bytes`. Resident backends need no preparation; streaming
    /// sources may fill a bounded block cache here.
    fn prepare_source_rows(
        &mut self,
        _stats: &GlobalStats,
        _row_start: usize,
        _rows_here: usize,
    ) -> Result<(), String> {
        Ok(())
    }

    /// MAF values produced by a streaming decoder for the most recent block.
    /// Resident BED backends return `None` because their metadata is already
    /// available in `GlobalStats`; k-file adapters use this bounded slice for
    /// direct TSV output without allocating one vector per marker.
    fn block_maf(&self, _rows_here: usize) -> Option<&[f32]> {
        None
    }

    /// Decode a block of kept markers via `decode_mean_imputed_additive_packed_block_rows_f32`.
    /// Default impl uses `packed_flat()` with absolute source offsets.
    /// Streaming backends override to pre-read file windows.
    fn decode_additive_block(
        &mut self,
        stats: &GlobalStats,
        row_start: usize,
        out: &mut [f32],
        sample_idx: &[usize],
        sample_identity: bool,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) -> Result<(), String> {
        let cols = if sample_identity {
            stats.n_samples_full
        } else {
            sample_idx.len()
        };
        let rows_here = out.len().saturating_div(cols.max(1));
        if rows_here == 0 {
            return Ok(());
        }
        let code4_lut = &packed_byte_lut().code4;
        decode_mean_imputed_additive_packed_block_rows_f32(
            self.packed_flat(),
            stats.bytes_per_snp,
            stats.n_samples_full,
            stats.row_flip_slice(row_start, rows_here),
            stats.row_maf_slice(row_start, rows_here),
            sample_idx,
            sample_identity,
            Some(stats.row_source_slice(row_start, rows_here)),
            0,
            out,
            code4_lut,
            pool,
        )
    }
}

// ---------------------------------------------------------------------------
/// Direct source-order BED reader with a caller-owned packed destination.
/// It validates the file header and row shape once, then reads only requested
/// source rows into the supplied buffer.
/// Single process-wide resident 2-bit packed BED buffer populated by `-memload`.
/// Keeping strictly one slot guarantees at most one packed BED copy in RAM.
fn next_resident_generation() -> u64 {
    static NEXT_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

pub(crate) struct ResidentPackedBed {
    pub(crate) key: String,
    pub(crate) generation: u64,
    pub(crate) n_samples_full: usize,
    pub(crate) n_snps: usize,
    pub(crate) bytes_per_snp: usize,
    pub(crate) packed: Vec<u8>,
    pub(crate) meta_bytes: Option<Arc<Vec<u8>>>,
    pub(crate) meta_is_zst: bool,
}

struct ArcBytesCursor {
    data: Arc<Vec<u8>>,
    pos: usize,
}

impl std::io::Read for ArcBytesCursor {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let rem = &self.data[self.pos..];
        let n = rem.len().min(buf.len());
        buf[..n].copy_from_slice(&rem[..n]);
        self.pos += n;
        Ok(n)
    }
}

impl std::io::BufRead for ArcBytesCursor {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        Ok(&self.data[self.pos..])
    }

    fn consume(&mut self, amt: usize) {
        self.pos = self.pos.saturating_add(amt).min(self.data.len());
    }
}

fn read_file_nocache_aligned(path: &Path) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    hint_file_nocache(&file);
    let total_bytes = file
        .metadata()
        .map_err(|e| format!("stat {}: {e}", path.display()))?
        .len() as usize;
    let mut out = vec![0u8; total_bytes];
    const IO_CHUNK: usize = 4 * 1024 * 1024;
    let mut offset = 0usize;
    while offset < total_bytes {
        crate::stats_common::check_ctrlc()?;
        let to_read = (total_bytes - offset).min(IO_CHUNK);
        file.read_exact(&mut out[offset..offset + to_read])
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        hint_file_dontneed(&file, offset as u64, to_read as u64);
        offset += to_read;
    }
    Ok(out)
}

pub(crate) fn hint_file_nocache(file: &File) {
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::io::AsRawFd;
        unsafe {
            let _ = libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
        }
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        unsafe {
            let _ = libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL);
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = file;
    }
}

pub(crate) fn hint_file_dontneed(file: &File, offset: u64, len: u64) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        unsafe {
            let _ = libc::posix_fadvise(
                file.as_raw_fd(),
                offset as libc::off_t,
                len as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, offset, len);
    }
}

struct ActiveResidentSlot {
    entry: Arc<ResidentPackedBed>,
    owners: std::collections::HashSet<u64>,
}

struct ResidentSlotState {
    active: Option<ActiveResidentSlot>,
    last_weak: std::sync::Weak<ResidentPackedBed>,
}

fn resident_packed_bed_slot() -> &'static Mutex<ResidentSlotState> {
    static SLOT: OnceLock<Mutex<ResidentSlotState>> = OnceLock::new();
    SLOT.get_or_init(|| {
        Mutex::new(ResidentSlotState {
            active: None,
            last_weak: std::sync::Weak::new(),
        })
    })
}

fn resident_packed_bed_load_lock() -> &'static Mutex<()> {
    static LOAD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOAD_LOCK.get_or_init(|| Mutex::new(()))
}

fn next_resident_owner_token() -> u64 {
    static NEXT_TOKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT_TOKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn resolve_resident_source(prefix: &str) -> (bool, String, String) {
    let is_pgen = crate::pgen::path_uses_pgen(prefix);
    let kind = if is_pgen { "pgen" } else { "bed" };
    let mut trimmed = prefix.trim().to_string();
    let lower = trimmed.to_ascii_lowercase();
    for suffix in [
        ".pvar.zst",
        ".pgen",
        ".pvar",
        ".psam",
        ".bed",
        ".bim",
        ".fam",
    ] {
        if lower.ends_with(suffix) {
            trimmed.truncate(trimmed.len().saturating_sub(suffix.len()));
            break;
        }
    }
    let path = Path::new(&trimmed);
    let canon_prefix = if let Some(name) = path.file_name() {
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
        let parent_dir = parent.unwrap_or_else(|| Path::new("."));
        if let Ok(canon_parent) = std::fs::canonicalize(parent_dir) {
            canon_parent.join(name).to_string_lossy().to_string()
        } else if let Ok(cwd) = std::env::current_dir() {
            cwd.join(path).to_string_lossy().to_string()
        } else {
            trimmed
        }
    } else if let Ok(canon) = std::fs::canonicalize(path) {
        canon.to_string_lossy().to_string()
    } else {
        trimmed
    };
    let key = format!("{kind}:{canon_prefix}");
    (is_pgen, canon_prefix, key)
}

pub(crate) fn resident_packed_bed_key(prefix: &str) -> String {
    resolve_resident_source(prefix).2
}

pub(crate) fn lookup_resident_packed_bed(prefix: &str) -> Option<Arc<ResidentPackedBed>> {
    let key = resident_packed_bed_key(prefix);
    let guard = resident_packed_bed_slot().lock().ok()?;
    guard
        .active
        .as_ref()
        .filter(|state| state.entry.key == key)
        .map(|state| Arc::clone(&state.entry))
}

pub(crate) fn open_resident_variant_meta_reader(
    prefix: &str,
) -> Result<Option<Box<dyn std::io::BufRead + Send>>, String> {
    let Some(entry) = lookup_resident_packed_bed(prefix) else {
        return Ok(None);
    };
    let Some(bytes) = entry.meta_bytes.as_ref() else {
        return Ok(None);
    };
    let cursor = ArcBytesCursor {
        data: Arc::clone(bytes),
        pos: 0,
    };
    if entry.meta_is_zst {
        let decoder = zstd::stream::read::Decoder::new(cursor)
            .map_err(|e| format!("open resident zstd metadata reader: {e}"))?;
        let reader = std::io::BufReader::with_capacity(1 << 20, decoder);
        Ok(Some(Box::new(reader)))
    } else {
        Ok(Some(Box::new(cursor)))
    }
}

pub(crate) fn clear_resident_packed_bed(token: Option<u64>) {
    if let Ok(mut guard) = resident_packed_bed_slot().lock() {
        match token {
            None => {
                if let Some(prev) = guard.active.take() {
                    guard.last_weak = Arc::downgrade(&prev.entry);
                }
            }
            Some(tok) => {
                let should_clear = if let Some(state) = guard.active.as_mut() {
                    state.owners.remove(&tok);
                    state.owners.is_empty()
                } else {
                    false
                };
                if should_clear {
                    if let Some(prev) = guard.active.take() {
                        guard.last_weak = Arc::downgrade(&prev.entry);
                    }
                }
            }
        }
    }
}

pub(crate) fn load_resident_packed_bed(
    prefix: &str,
    threads: usize,
) -> Result<(Arc<ResidentPackedBed>, u64), String> {
    // Serialize concurrent registrations so two threads never allocate two full
    // packed buffers at the same time.
    let _load_guard = resident_packed_bed_load_lock()
        .lock()
        .map_err(|_| "resident packed BED load lock was poisoned".to_string())?;
    let (pgen_source, canon_prefix, key) = resolve_resident_source(prefix);
    let token = next_resident_owner_token();
    {
        let mut guard = resident_packed_bed_slot()
            .lock()
            .map_err(|_| "resident packed BED cache lock was poisoned".to_string())?;
        if let Some(state) = guard.active.as_mut() {
            if state.entry.key == key {
                state.owners.insert(token);
                return Ok((Arc::clone(&state.entry), token));
            }
            if !state.owners.is_empty() || Arc::strong_count(&state.entry) > 1 {
                return Err(format!(
                    "cannot preload '{}' while resident genotype cache '{}' is still held by {} active owner(s) or reader(s); release it first",
                    key,
                    state.entry.key,
                    state.owners.len()
                ));
            }
            if let Some(prev) = guard.active.take() {
                guard.last_weak = Arc::downgrade(&prev.entry);
            }
        }
        if let Some(live_prev) = guard.last_weak.upgrade() {
            if live_prev.key == key {
                let mut owners = std::collections::HashSet::new();
                owners.insert(token);
                guard.active = Some(ActiveResidentSlot {
                    entry: Arc::clone(&live_prev),
                    owners,
                });
                return Ok((live_prev, token));
            }
            return Err(format!(
                "cannot preload '{}' while a previous resident genotype buffer '{}' is still held by an active reader",
                key, live_prev.key
            ));
        }
    }

    let (n_samples_full, n_snps, bytes_per_snp, packed) = if pgen_source {
        let n_samples_full = crate::pgen::count_psam_samples(&canon_prefix)?;
        let n_snps = crate::pgen::count_pgen_variants(&canon_prefix, n_samples_full)?;
        let bytes_per_snp = n_samples_full.div_ceil(4);
        let total_bytes = n_snps
            .checked_mul(bytes_per_snp)
            .ok_or_else(|| "resident packed PGEN size overflow".to_string())?;
        let pgen_path = Path::new(&format!("{canon_prefix}.pgen")).to_path_buf();
        let mut reader =
            crate::pgen::PgenReader::open_for_preload(&pgen_path, n_samples_full, n_snps)?;
        let mut packed = vec![0u8; total_bytes];
        let pool = if threads > 0 {
            crate::stats_common::get_cached_pool(threads).map_err(|e| e.to_string())?
        } else {
            None
        };
        const BLOCK_ROWS: usize = 1 << 16;
        let mut start = 0usize;
        while start < n_snps {
            crate::stats_common::check_ctrlc()?;
            let rows = (n_snps - start).min(BLOCK_ROWS);
            let byte_start = start.saturating_mul(bytes_per_snp);
            let byte_end = byte_start.saturating_add(rows.saturating_mul(bytes_per_snp));
            let dst = &mut packed[byte_start..byte_end];
            let got = if let Some(tp) = pool.as_ref() {
                tp.install(|| reader.read_source_block(start, rows, dst))?
            } else {
                reader.read_source_block(start, rows, dst)?
            };
            if got != rows {
                return Err(format!(
                    "PGEN resident load returned {got} rows; expected {rows} at {start}"
                ));
            }
            start = start.saturating_add(rows);
        }
        reader.advise_preload_dontneed();
        (n_samples_full, n_snps, bytes_per_snp, packed)
    } else {
        let n_samples_full = gfcore::read_fam(&canon_prefix)?.len();
        let bed_path = format!("{canon_prefix}.bed");
        let info = read_plink_bed_header_info(Path::new(&bed_path), n_samples_full)?;
        if info.n_snps_total == 0 {
            return Err("no SNP sites found in PLINK BED input".to_string());
        }
        let mut file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        // Set F_NOCACHE / sequential hint BEFORE any read, and read from offset 0
        // in page-aligned 16 MiB chunks so macOS F_NOCACHE and Linux DONTNEED
        // apply to every full page instead of leaving unaligned tail pages cached.
        hint_file_nocache(&file);
        let total_file_bytes = BED_HEADER_LEN
            .checked_add(info.payload_size)
            .ok_or_else(|| "resident packed BED size overflow".to_string())?;
        let mut packed = vec![0u8; info.payload_size];
        const IO_CHUNK: usize = 4 * 1024 * 1024;
        let mut aligned_buf = vec![0u8; IO_CHUNK.min(total_file_bytes)];
        let mut file_offset = 0usize;
        let mut dst_offset = 0usize;
        while file_offset < total_file_bytes {
            crate::stats_common::check_ctrlc()?;
            let to_read = (total_file_bytes - file_offset).min(aligned_buf.len());
            file.read_exact(&mut aligned_buf[..to_read])
                .map_err(|e| format!("read BED payload {bed_path}: {e}"))?;
            if file_offset == 0 {
                if to_read < BED_HEADER_LEN {
                    return Err(format!("read BED header {bed_path}: truncated file"));
                }
                validate_plink_bed_header([aligned_buf[0], aligned_buf[1], aligned_buf[2]])?;
                let copy_len = to_read - BED_HEADER_LEN;
                packed[..copy_len].copy_from_slice(&aligned_buf[BED_HEADER_LEN..to_read]);
                dst_offset = copy_len;
            } else {
                let dst = &mut packed[dst_offset..dst_offset + to_read];
                dst.copy_from_slice(&aligned_buf[..to_read]);
                dst_offset += to_read;
            }
            hint_file_dontneed(&file, file_offset as u64, to_read as u64);
            file_offset += to_read;
        }
        (
            n_samples_full,
            info.n_snps_total,
            info.bytes_per_snp,
            packed,
        )
    };

    let (meta_bytes, meta_is_zst) = if pgen_source {
        let pvar_path = crate::pgen::pvar_path_for_prefix(&canon_prefix);
        if pvar_path.is_file() {
            let is_zst = pvar_path
                .to_string_lossy()
                .to_ascii_lowercase()
                .ends_with(".zst");
            let bytes = read_file_nocache_aligned(&pvar_path)?;
            (Some(Arc::new(bytes)), is_zst)
        } else {
            (None, false)
        }
    } else {
        let bim_path = Path::new(&format!("{canon_prefix}.bim")).to_path_buf();
        if bim_path.is_file() {
            let bytes = read_file_nocache_aligned(&bim_path)?;
            (Some(Arc::new(bytes)), false)
        } else {
            (None, false)
        }
    };

    let entry = Arc::new(ResidentPackedBed {
        key,
        generation: next_resident_generation(),
        n_samples_full,
        n_snps,
        bytes_per_snp,
        packed,
        meta_bytes,
        meta_is_zst,
    });
    let mut owners = std::collections::HashSet::new();
    owners.insert(token);
    let mut guard = resident_packed_bed_slot()
        .lock()
        .map_err(|_| "resident packed BED cache lock was poisoned".to_string())?;
    guard.last_weak = Arc::downgrade(&entry);
    guard.active = Some(ActiveResidentSlot {
        entry: Arc::clone(&entry),
        owners,
    });
    Ok((entry, token))
}

#[pyfunction]
#[pyo3(signature = (prefix, threads=0))]
pub fn gwas_memload_register(
    py: Python<'_>,
    prefix: String,
    threads: usize,
) -> PyResult<(usize, usize, usize, u64)> {
    py.detach(move || {
        let (entry, token) =
            load_resident_packed_bed(&prefix, threads).map_err(PyRuntimeError::new_err)?;
        Ok((
            entry.n_samples_full,
            entry.n_snps,
            entry.packed.len(),
            token,
        ))
    })
}

#[pyfunction]
#[pyo3(signature = (token=None))]
pub fn gwas_memload_clear(token: Option<u64>) -> PyResult<()> {
    clear_resident_packed_bed(token);
    Ok(())
}

pub(crate) struct PackedBedSourceReader {
    source: PackedBedSource,
    n_samples_full: usize,
    bytes_per_snp: usize,
    n_snps: usize,
    qc_scratch: Vec<u8>,
}

enum PackedBedSource {
    Bed(File),
    Pgen(crate::pgen::PgenReader),
    Memory(Arc<ResidentPackedBed>),
}

impl PackedBedSourceReader {
    pub(crate) fn open(prefix: &str, n_samples_full: usize) -> Result<Self, String> {
        if let Some(resident) = lookup_resident_packed_bed(prefix) {
            if resident.n_samples_full == n_samples_full {
                let bytes_per_snp = resident.bytes_per_snp;
                let n_snps = resident.n_snps;
                return Ok(Self {
                    source: PackedBedSource::Memory(resident),
                    n_samples_full,
                    bytes_per_snp,
                    n_snps,
                    qc_scratch: Vec::new(),
                });
            }
        }
        let pgen_source = crate::pgen::path_uses_pgen(prefix);
        let bed_prefix = normalize_plink_prefix(prefix);
        if pgen_source {
            let pgen_prefix = crate::pgen::normalize_pgen_prefix(prefix);
            let psam_samples = crate::pgen::count_psam_samples(&pgen_prefix)?;
            if psam_samples != n_samples_full {
                return Err(format!(
                    "PGEN/PSAM sample count mismatch: reader was given {n_samples_full}, PSAM has {psam_samples}"
                ));
            }
            // The PGEN header already contains the authoritative variant
            // count.  Counting PVAR rows here turns opening a large PGEN
            // into a full scan of the companion text file before any
            // progress callback can be emitted.
            let n_snps = crate::pgen::count_pgen_variants(&pgen_prefix, n_samples_full)?;
            let pgen_path = Path::new(&format!("{pgen_prefix}.pgen")).to_path_buf();
            let reader = crate::pgen::PgenReader::open(&pgen_path, n_samples_full, n_snps)?;
            return Ok(Self {
                source: PackedBedSource::Pgen(reader),
                n_samples_full,
                bytes_per_snp: n_samples_full.div_ceil(4),
                n_snps,
                qc_scratch: Vec::new(),
            });
        }
        let bed_path = format!("{bed_prefix}.bed");
        let info = read_plink_bed_header_info(Path::new(&bed_path), n_samples_full)?;
        if info.n_snps_total == 0 {
            return Err("no SNP sites found in PLINK BED input".to_string());
        }
        let file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        Ok(Self {
            source: PackedBedSource::Bed(file),
            n_samples_full,
            bytes_per_snp: info.bytes_per_snp,
            n_snps: info.n_snps_total,
            qc_scratch: Vec::new(),
        })
    }

    #[inline]
    pub(crate) fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }

    #[inline]
    pub(crate) fn bytes_per_snp(&self) -> usize {
        self.bytes_per_snp
    }

    #[inline]
    pub(crate) fn n_snps(&self) -> usize {
        self.n_snps
    }

    pub(crate) fn read_source_block(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [u8],
    ) -> Result<usize, String> {
        if source_start > self.n_snps {
            return Err(format!(
                "BED source row {source_start} exceeds marker count {}",
                self.n_snps
            ));
        }
        let rows = requested_rows.min(self.n_snps - source_start);
        let byte_len = rows.saturating_mul(self.bytes_per_snp);
        if out.len() < byte_len {
            return Err(format!(
                "packed BED destination is too small: got {}, need {byte_len}",
                out.len()
            ));
        }
        if rows > 0 {
            match &mut self.source {
                PackedBedSource::Bed(file) => {
                    let offset = (BED_HEADER_LEN as u64)
                        .saturating_add((source_start.saturating_mul(self.bytes_per_snp)) as u64);
                    read_file_exact_at(file, &mut out[..byte_len], offset)
                        .map_err(|e| format!("read BED source rows [{source_start}..): {e}"))?;
                }
                PackedBedSource::Pgen(reader) => {
                    let rows_read =
                        reader.read_source_block(source_start, rows, &mut out[..byte_len])?;
                    if rows_read != rows {
                        return Err(format!(
                            "PGEN reader returned {rows_read} rows; requested {rows} starting at {source_start}"
                        ));
                    }
                }
                PackedBedSource::Memory(resident) => {
                    let src_start = source_start.saturating_mul(self.bytes_per_snp);
                    let src_end = src_start.saturating_add(byte_len);
                    out[..byte_len].copy_from_slice(&resident.packed[src_start..src_end]);
                }
            }
        }
        Ok(rows)
    }

    /// Read a small, arbitrarily ordered set of source rows. PGEN uses direct
    /// reads from the nearest LD-compression anchor so sampling does not load
    /// a complete 65k-variant block for every candidate.
    pub(crate) fn read_source_rows_sparse(
        &mut self,
        source_rows: &[usize],
        out: &mut [u8],
    ) -> Result<(), String> {
        let expected_len = source_rows
            .len()
            .checked_mul(self.bytes_per_snp)
            .ok_or_else(|| "sparse BED source output size overflow".to_string())?;
        if out.len() < expected_len {
            return Err(format!(
                "sparse BED source destination is too small: got {}, need {expected_len}",
                out.len()
            ));
        }
        if let Some(&source_row) = source_rows.iter().find(|&&row| row >= self.n_snps) {
            return Err(format!(
                "BED source row {source_row} exceeds marker count {}",
                self.n_snps
            ));
        }
        match &mut self.source {
            PackedBedSource::Bed(file) => {
                for (output_row, &source_row) in source_rows.iter().enumerate() {
                    let offset = (BED_HEADER_LEN as u64)
                        .checked_add(
                            (source_row as u64)
                                .checked_mul(self.bytes_per_snp as u64)
                                .ok_or_else(|| "BED sparse row offset overflow".to_string())?,
                        )
                        .ok_or_else(|| "BED sparse row offset overflow".to_string())?;
                    let output_start = output_row * self.bytes_per_snp;
                    read_file_exact_at(
                        file,
                        &mut out[output_start..output_start + self.bytes_per_snp],
                        offset,
                    )
                    .map_err(|e| format!("read BED source row {source_row}: {e}"))?;
                }
            }
            PackedBedSource::Pgen(reader) => {
                reader.read_source_rows_sparse(source_rows, out)?;
            }
            PackedBedSource::Memory(resident) => {
                for (output_row, &source_row) in source_rows.iter().enumerate() {
                    let src_start = source_row.saturating_mul(self.bytes_per_snp);
                    let dst_start = output_row.saturating_mul(self.bytes_per_snp);
                    out[dst_start..dst_start + self.bytes_per_snp].copy_from_slice(
                        &resident.packed[src_start..src_start + self.bytes_per_snp],
                    );
                }
            }
        }
        Ok(())
    }

    /// Reads a row-major f32 hard-call matrix, decoding PGEN category bytes
    /// directly to dosages instead of passing through BED-packed rows.
    #[allow(dead_code)] // Shared Rust streaming API for future genotype consumers.
    pub(crate) fn read_source_block_f32(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [f32],
    ) -> Result<usize, String> {
        self.read_source_block_values_f32(source_start, requested_rows, out)
    }

    /// f64 counterpart for streaming consumers that need double-precision
    /// genotype rows.
    #[allow(dead_code)] // Shared Rust streaming API for future genotype consumers.
    pub(crate) fn read_source_block_f64(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [f64],
    ) -> Result<usize, String> {
        self.read_source_block_values_f64(source_start, requested_rows, out)
    }

    #[allow(dead_code)]
    fn read_source_block_values_f32(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [f32],
    ) -> Result<usize, String> {
        if source_start > self.n_snps {
            return Err(format!(
                "genotype source row {source_start} exceeds marker count {}",
                self.n_snps
            ));
        }
        match &mut self.source {
            PackedBedSource::Pgen(reader) => {
                reader.read_source_block_f32(source_start, requested_rows, out)
            }
            PackedBedSource::Bed(file) => read_bed_source_block_values(
                file,
                self.n_samples_full,
                self.bytes_per_snp,
                self.n_snps,
                &mut self.qc_scratch,
                source_start,
                requested_rows,
                out,
            ),
            PackedBedSource::Memory(resident) => read_packed_slice_block_values(
                &resident.packed,
                self.n_samples_full,
                self.bytes_per_snp,
                self.n_snps,
                source_start,
                requested_rows,
                out,
            ),
        }
    }

    #[allow(dead_code)]
    fn read_source_block_values_f64(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        out: &mut [f64],
    ) -> Result<usize, String> {
        if source_start > self.n_snps {
            return Err(format!(
                "genotype source row {source_start} exceeds marker count {}",
                self.n_snps
            ));
        }
        match &mut self.source {
            PackedBedSource::Pgen(reader) => {
                reader.read_source_block_f64(source_start, requested_rows, out)
            }
            PackedBedSource::Bed(file) => read_bed_source_block_values(
                file,
                self.n_samples_full,
                self.bytes_per_snp,
                self.n_snps,
                &mut self.qc_scratch,
                source_start,
                requested_rows,
                out,
            ),
            PackedBedSource::Memory(resident) => read_packed_slice_block_values(
                &resident.packed,
                self.n_samples_full,
                self.bytes_per_snp,
                self.n_snps,
                source_start,
                requested_rows,
                out,
            ),
        }
    }

    /// Shared QC-aware read contract for BED and PGEN consumers. `stats_out`
    /// and `keep_out` are indexed by source row; rejected packed rows are
    /// zeroed and must be ignored by callers. The predicate is intentionally
    /// caller-defined so GWAS, GS, and GARFIELD can share decode/QC mechanics
    /// while retaining their own filtering rules.
    pub(crate) fn read_source_block_with_qc<F>(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        sample_selection: crate::pgen::GenotypeSampleSelection<'_>,
        out_packed: &mut [u8],
        stats_out: &mut [crate::pgen::GenotypeRowStats],
        keep_out: &mut [bool],
        keep_variant: &F,
    ) -> Result<usize, String>
    where
        F: Fn(usize, crate::pgen::GenotypeRowStats) -> bool + Sync,
    {
        if source_start > self.n_snps {
            return Err(format!(
                "BED source row {source_start} exceeds marker count {}",
                self.n_snps
            ));
        }
        let rows = requested_rows.min(self.n_snps - source_start);
        let byte_len = rows
            .checked_mul(self.bytes_per_snp)
            .ok_or_else(|| "packed genotype QC block size overflow".to_string())?;
        if out_packed.len() < byte_len {
            return Err(format!(
                "packed genotype QC destination is too small: got {}, need {byte_len}",
                out_packed.len()
            ));
        }
        if stats_out.len() < rows || keep_out.len() < rows {
            return Err(format!(
                "genotype QC metadata destination too small: stats={}, keep={}, need {rows}",
                stats_out.len(),
                keep_out.len()
            ));
        }
        if rows == 0 {
            return Ok(0);
        }

        match &mut self.source {
            PackedBedSource::Pgen(reader) => reader.read_source_block_with_qc(
                source_start,
                rows,
                sample_selection,
                &mut out_packed[..byte_len],
                &mut stats_out[..rows],
                &mut keep_out[..rows],
                keep_variant,
            ),
            PackedBedSource::Bed(file) => {
                sample_selection.validate(self.n_samples_full)?;
                let offset = (BED_HEADER_LEN as u64)
                    .saturating_add((source_start.saturating_mul(self.bytes_per_snp)) as u64);
                read_file_exact_at(file, &mut out_packed[..byte_len], offset)
                    .map_err(|e| format!("read BED source rows [{source_start}..): {e}"))?;
                for row_idx in 0..rows {
                    let row_start = row_idx * self.bytes_per_snp;
                    let row_end = row_start + self.bytes_per_snp;
                    let stats = crate::pgen::count_bed_genotype_row(
                        &out_packed[row_start..row_end],
                        self.n_samples_full,
                        sample_selection,
                    );
                    let keep = keep_variant(source_start + row_idx, stats);
                    stats_out[row_idx] = stats;
                    keep_out[row_idx] = keep;
                    if !keep {
                        out_packed[row_start..row_end].fill(0);
                    }
                }
                Ok(rows)
            }
            PackedBedSource::Memory(resident) => {
                sample_selection.validate(self.n_samples_full)?;
                let src_offset = source_start.saturating_mul(self.bytes_per_snp);
                out_packed[..byte_len]
                    .copy_from_slice(&resident.packed[src_offset..src_offset + byte_len]);
                for row_idx in 0..rows {
                    let row_start = row_idx * self.bytes_per_snp;
                    let row_end = row_start + self.bytes_per_snp;
                    let stats = crate::pgen::count_bed_genotype_row(
                        &out_packed[row_start..row_end],
                        self.n_samples_full,
                        sample_selection,
                    );
                    let keep = keep_variant(source_start + row_idx, stats);
                    stats_out[row_idx] = stats;
                    keep_out[row_idx] = keep;
                    if !keep {
                        out_packed[row_start..row_end].fill(0);
                    }
                }
                Ok(rows)
            }
        }
    }

    /// Shared QC-only read contract. PGEN rows are counted while decoding and
    /// are not converted to packed BED; BED input is scanned through a reusable
    /// packed scratch block. Consumers receive per-source-row statistics and a
    /// keep mask without allocating a packed output matrix.
    pub(crate) fn read_source_block_qc_only<F>(
        &mut self,
        source_start: usize,
        requested_rows: usize,
        sample_selection: crate::pgen::GenotypeSampleSelection<'_>,
        stats_out: &mut [crate::pgen::GenotypeRowStats],
        keep_out: &mut [bool],
        keep_variant: &F,
    ) -> Result<usize, String>
    where
        F: Fn(usize, crate::pgen::GenotypeRowStats) -> bool + Sync,
    {
        if source_start > self.n_snps {
            return Err(format!(
                "BED source row {source_start} exceeds marker count {}",
                self.n_snps
            ));
        }
        let rows = requested_rows.min(self.n_snps - source_start);
        if stats_out.len() < rows || keep_out.len() < rows {
            return Err(format!(
                "genotype QC metadata destination too small: stats={}, keep={}, need {rows}",
                stats_out.len(),
                keep_out.len()
            ));
        }
        if rows == 0 {
            return Ok(0);
        }

        match &mut self.source {
            PackedBedSource::Pgen(reader) => reader.read_source_block_qc_only(
                source_start,
                rows,
                sample_selection,
                &mut stats_out[..rows],
                &mut keep_out[..rows],
                keep_variant,
            ),
            PackedBedSource::Bed(file) => {
                sample_selection.validate(self.n_samples_full)?;
                let byte_len = rows
                    .checked_mul(self.bytes_per_snp)
                    .ok_or_else(|| "packed genotype QC scratch size overflow".to_string())?;
                self.qc_scratch.resize(byte_len, 0);
                let offset = (BED_HEADER_LEN as u64)
                    .saturating_add((source_start.saturating_mul(self.bytes_per_snp)) as u64);
                read_file_exact_at(file, &mut self.qc_scratch, offset)
                    .map_err(|e| format!("read BED source rows [{source_start}..): {e}"))?;
                for row_idx in 0..rows {
                    let row_start = row_idx * self.bytes_per_snp;
                    let row_end = row_start + self.bytes_per_snp;
                    let stats = crate::pgen::count_bed_genotype_row(
                        &self.qc_scratch[row_start..row_end],
                        self.n_samples_full,
                        sample_selection,
                    );
                    stats_out[row_idx] = stats;
                    keep_out[row_idx] = keep_variant(source_start + row_idx, stats);
                }
                Ok(rows)
            }
            PackedBedSource::Memory(resident) => {
                sample_selection.validate(self.n_samples_full)?;
                let packed = resident.packed.as_slice();
                let bps = self.bytes_per_snp;
                let n_samples = self.n_samples_full;
                stats_out[..rows]
                    .par_iter_mut()
                    .zip(keep_out[..rows].par_iter_mut())
                    .enumerate()
                    .for_each(|(row_idx, (stats_dst, keep_dst))| {
                        let src_start = (source_start + row_idx) * bps;
                        let src_end = src_start + bps;
                        let stats = crate::pgen::count_bed_genotype_row(
                            &packed[src_start..src_end],
                            n_samples,
                            sample_selection,
                        );
                        *stats_dst = stats;
                        *keep_dst = keep_variant(source_start + row_idx, stats);
                    });
                Ok(rows)
            }
        }
    }
}

#[allow(dead_code)]
fn read_packed_slice_block_values<T: crate::pgen::PgenDosageValue>(
    packed: &[u8],
    n_samples: usize,
    bytes_per_snp: usize,
    n_snps: usize,
    source_start: usize,
    requested_rows: usize,
    out: &mut [T],
) -> Result<usize, String> {
    let rows = requested_rows.min(n_snps - source_start);
    let value_len = rows
        .checked_mul(n_samples)
        .ok_or_else(|| "genotype dosage block size overflow".to_string())?;
    if out.len() < value_len {
        return Err(format!(
            "genotype dosage destination is too small: got {}, need {value_len}",
            out.len()
        ));
    }
    if rows == 0 {
        return Ok(0);
    }
    for row_idx in 0..rows {
        let packed_start = (source_start + row_idx) * bytes_per_snp;
        let packed_end = packed_start + bytes_per_snp;
        let values_start = row_idx * n_samples;
        let values_end = values_start + n_samples;
        crate::pgen::decode_bed_row_to_values(
            &packed[packed_start..packed_end],
            n_samples,
            &mut out[values_start..values_end],
        );
    }
    Ok(rows)
}

#[allow(dead_code)]
fn read_bed_source_block_values<T: crate::pgen::PgenDosageValue>(
    file: &File,
    n_samples: usize,
    bytes_per_snp: usize,
    n_snps: usize,
    scratch: &mut Vec<u8>,
    source_start: usize,
    requested_rows: usize,
    out: &mut [T],
) -> Result<usize, String> {
    let rows = requested_rows.min(n_snps - source_start);
    let value_len = rows
        .checked_mul(n_samples)
        .ok_or_else(|| "genotype dosage block size overflow".to_string())?;
    if out.len() < value_len {
        return Err(format!(
            "genotype dosage destination is too small: got {}, need {value_len}",
            out.len()
        ));
    }
    if rows == 0 {
        return Ok(0);
    }
    let byte_len = rows
        .checked_mul(bytes_per_snp)
        .ok_or_else(|| "packed genotype dosage scratch size overflow".to_string())?;
    scratch.resize(byte_len, 0);
    let offset =
        (BED_HEADER_LEN as u64).saturating_add((source_start.saturating_mul(bytes_per_snp)) as u64);
    read_file_exact_at(file, scratch, offset)
        .map_err(|e| format!("read BED source rows [{source_start}..): {e}"))?;
    for row_idx in 0..rows {
        let packed_start = row_idx * bytes_per_snp;
        let packed_end = packed_start + bytes_per_snp;
        let values_start = row_idx * n_samples;
        let values_end = values_start + n_samples;
        crate::pgen::decode_bed_row_to_values(
            &scratch[packed_start..packed_end],
            n_samples,
            &mut out[values_start..values_end],
        );
    }
    Ok(rows)
}

// StreamingBedMatrix — chunked file I/O, no mmap
// ---------------------------------------------------------------------------

/// Reads BED rows on-demand via `pread` (or `seek`+`read`) instead of
/// mmap-ing the whole file.  Dramatically reduces RSS for large BEDs
/// when only sequential access is needed (e.g. GWAS scan).
#[allow(dead_code)]
pub struct StreamingBedMatrix {
    file: std::fs::File,
    n_samples_full: usize,
    bytes_per_snp: usize,
    /// Reusable read buffer sized for one block's worth of packed rows.
    buf: Vec<u8>,
    /// Byte range currently held in `buf` (start, end) in source-index units.
    buf_range: (usize, usize),
}

#[allow(dead_code)]
impl StreamingBedMatrix {
    /// Open a BED prefix for streaming reads.
    pub fn open(prefix: &str, block_rows: usize) -> Result<Self, String> {
        let bed_prefix = normalize_plink_prefix(prefix);
        let n_samples_full = crate::gfcore::read_fam(&bed_prefix)
            .map_err(|e| e.to_string())?
            .len();
        if n_samples_full == 0 {
            return Err("no samples found in BED".to_string());
        }
        let bytes_per_snp = n_samples_full.div_ceil(4);
        let bed_path = format!("{bed_prefix}.bed");
        let file = std::fs::File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        // 3-byte BED header is skipped via offset in read_rows.
        let buf = vec![0u8; block_rows.saturating_mul(bytes_per_snp).max(1)];
        Ok(Self {
            file,
            n_samples_full,
            bytes_per_snp,
            buf,
            buf_range: (usize::MAX, 0),
        })
    }

    /// Ensure `buf` contains the packed bytes for source rows `[start .. end)`.
    /// Returns a slice covering the requested range.
    pub fn read_source_range(&mut self, start: usize, end: usize) -> Result<&[u8], String> {
        let bps = self.bytes_per_snp;
        let need = end.saturating_sub(start).saturating_mul(bps);
        if need > self.buf.len() {
            self.buf.resize(need, 0);
        }
        if start == self.buf_range.0 && end == self.buf_range.1 {
            return Ok(&self.buf[..need]);
        }
        let offset = 3u64.saturating_add((start.saturating_mul(bps)) as u64);
        read_file_exact_at(&self.file, &mut self.buf[..need], offset)
            .map_err(|e| format!("read BED rows [{start}..{end}): {e}"))?;
        self.buf_range = (start, end);
        Ok(&self.buf[..need])
    }
}

#[allow(dead_code)]
impl StreamingBedMatrix {
    /// Pre-fetch the byte range for a block of kept markers and return a
    /// contiguous packed slice + relative row-source indices ready for
    /// `decode_mean_imputed_additive_packed_block_rows_f32`.
    pub fn prepare_block(
        &mut self,
        stats: &GlobalStats,
        row_start: usize,
        rows_here: usize,
        rel_indices_out: &mut Vec<usize>,
    ) -> Result<&[u8], String> {
        let src_start = stats.row_source_indices[row_start];
        let src_end = stats.row_source_indices[row_start + rows_here - 1] + 1;
        let window = self.read_source_range(src_start, src_end)?;
        rel_indices_out.clear();
        rel_indices_out.extend(
            (row_start..row_start + rows_here)
                .map(|i| stats.row_source_indices[i].saturating_sub(src_start)),
        );
        if rel_indices_out.len() != rows_here {
            return Err("prepare_block: rel_indices mismatch".to_string());
        }
        Ok(window)
    }
}

impl GenotypeMatrix for StreamingBedMatrix {
    fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }
    fn bytes_per_snp(&self) -> usize {
        self.bytes_per_snp
    }
    fn packed_flat(&self) -> &[u8] {
        &self.buf[..(self
            .buf_range
            .1
            .saturating_sub(self.buf_range.0)
            .saturating_mul(self.bytes_per_snp))]
    }
    fn source_row_bytes(&self, _source_idx: usize) -> &[u8] {
        &[]
    }

    fn decode_additive_block(
        &mut self,
        stats: &GlobalStats,
        row_start: usize,
        out: &mut [f32],
        sample_idx: &[usize],
        sample_identity: bool,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) -> Result<(), String> {
        let cols = if sample_identity {
            stats.n_samples_full
        } else {
            sample_idx.len()
        };
        let rows_here = out.len().saturating_div(cols.max(1));
        if rows_here == 0 {
            return Ok(());
        }
        let mut rel_indices = Vec::with_capacity(rows_here);
        let packed_slice = self.prepare_block(stats, row_start, rows_here, &mut rel_indices)?;
        let code4_lut = &packed_byte_lut().code4;
        decode_mean_imputed_additive_packed_block_rows_f32(
            packed_slice,
            stats.bytes_per_snp,
            stats.n_samples_full,
            stats.row_flip_slice(row_start, rows_here),
            stats.row_maf_slice(row_start, rows_here),
            sample_idx,
            sample_identity,
            Some(&rel_indices),
            0,
            out,
            code4_lut,
            pool,
        )
    }
}

#[cfg(test)]
mod packed_source_tests {
    use super::{choose_lm_lmm_bed_pread, PackedBedSourceReader, WindowedBedMatrix};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn windowed_bed_source_block_preserves_raw_bim_order() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_packed_source_{stamp}"));
        fs::create_dir_all(&dir).expect("create temporary BED directory");
        let prefix = dir.join("tiny");
        fs::write(
            prefix.with_extension("bed"),
            [0x6c, 0x1b, 0x01, 0x31, 0x72, 0xB4, 0xA5],
        )
        .expect("write BED fixture");

        let mut reader = PackedBedSourceReader::open(prefix.to_str().unwrap(), 4)
            .expect("open direct packed BED source");
        assert_eq!(reader.n_snps(), 4);
        let mut out = vec![0u8; 3];
        assert_eq!(reader.read_source_block(1, 3, &mut out).unwrap(), 3);
        assert_eq!(out, [0x72, 0xB4, 0xA5]);

        let mut short = vec![0u8; 3];
        assert_eq!(reader.read_source_block(3, 3, &mut short).unwrap(), 1);
        assert_eq!(short[0], 0xA5);
        assert_eq!(reader.read_source_block(4, 1, &mut short).unwrap(), 0);
        assert!(reader.read_source_block(5, 1, &mut short).is_err());

        fs::remove_dir_all(dir).expect("remove temporary BED directory");
    }

    #[test]
    fn shared_qc_reader_applies_subset_stats_and_returns_keep_mask_for_bed() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_qc_source_{stamp}"));
        fs::create_dir_all(&dir).expect("create temporary BED directory");
        let prefix = dir.join("tiny");
        fs::write(
            prefix.with_extension("bed"),
            [0x6c, 0x1b, 0x01, 0x78, 0x02, 0x00, 0x00],
        )
        .expect("write BED fixture");

        let mut reader =
            PackedBedSourceReader::open(prefix.to_str().unwrap(), 5).expect("open packed source");
        let sample_indices = [1usize, 2, 4];
        let mut packed = [0xaau8; 4];
        let mut stats = [crate::pgen::GenotypeRowStats::default(); 2];
        let mut keep = [true; 2];
        assert_eq!(
            reader
                .read_source_block_with_qc(
                    0,
                    2,
                    crate::pgen::GenotypeSampleSelection::include(&sample_indices),
                    &mut packed,
                    &mut stats,
                    &mut keep,
                    &|source_idx, _| source_idx == 1,
                )
                .unwrap(),
            2
        );
        assert_eq!(stats[0].sample_count, 3);
        assert_eq!(stats[0].missing_count, 0);
        assert_eq!(stats[0].heterozygote_count, 2);
        assert_eq!(stats[0].homozygous_alt_count, 1);
        assert_eq!(
            stats[1],
            crate::pgen::GenotypeRowStats {
                sample_count: 3,
                ..crate::pgen::GenotypeRowStats::default()
            }
        );
        assert_eq!(keep, [false, true]);
        assert_eq!(packed, [0, 0, 0, 0]);

        let mut exclude_reader =
            PackedBedSourceReader::open(prefix.to_str().unwrap(), 5).expect("reopen packed source");
        let excluded = [0usize, 3];
        let mut exclude_packed = [0xaau8; 4];
        let mut exclude_stats = [crate::pgen::GenotypeRowStats::default(); 2];
        let mut exclude_keep = [true; 2];
        exclude_reader
            .read_source_block_with_qc(
                0,
                2,
                crate::pgen::GenotypeSampleSelection::Exclude(&excluded),
                &mut exclude_packed,
                &mut exclude_stats,
                &mut exclude_keep,
                &|source_idx, _| source_idx == 1,
            )
            .unwrap();
        assert_eq!(exclude_stats, stats);
        assert_eq!(exclude_keep, keep);
        assert_eq!(exclude_packed, packed);

        let mut stats_reader =
            PackedBedSourceReader::open(prefix.to_str().unwrap(), 5).expect("reopen packed source");
        let mut stats_only = [crate::pgen::GenotypeRowStats::default(); 2];
        let mut stats_only_keep = [false; 2];
        stats_reader
            .read_source_block_qc_only(
                0,
                2,
                crate::pgen::GenotypeSampleSelection::include(&sample_indices),
                &mut stats_only,
                &mut stats_only_keep,
                &|source_idx, _| source_idx == 1,
            )
            .unwrap();
        assert_eq!(stats_only, stats);
        assert_eq!(stats_only_keep, keep);

        let mut float_reader =
            PackedBedSourceReader::open(prefix.to_str().unwrap(), 5).expect("reopen BED source");
        let mut values_f32 = [0.0_f32; 10];
        assert_eq!(
            float_reader
                .read_source_block_f32(0, 2, &mut values_f32)
                .unwrap(),
            2
        );
        assert_eq!(
            values_f32,
            [0.0, 1.0, 2.0, -9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );

        let mut double_reader =
            PackedBedSourceReader::open(prefix.to_str().unwrap(), 5).expect("reopen BED source");
        let mut values_f64 = [0.0_f64; 10];
        assert_eq!(
            double_reader
                .read_source_block_f64(0, 2, &mut values_f64)
                .unwrap(),
            2
        );
        assert_eq!(
            values_f64,
            [0.0, 1.0, 2.0, -9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );

        fs::remove_dir_all(dir).expect("remove temporary BED directory");
    }

    #[test]
    fn lm_lmm_pread_switch_defaults_on_and_accepts_explicit_disable() {
        assert!(choose_lm_lmm_bed_pread(None));
        assert!(choose_lm_lmm_bed_pread(Some("1")));
        assert!(choose_lm_lmm_bed_pread(Some(" true ")));
        assert!(!choose_lm_lmm_bed_pread(Some("0")));
        assert!(!choose_lm_lmm_bed_pread(Some(" off ")));
    }

    #[test]
    fn pread_window_returns_the_same_raw_rows_as_mmap_window() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_window_pread_{stamp}"));
        fs::create_dir_all(&dir).expect("create temporary BED directory");
        let prefix = dir.join("tiny");
        fs::write(
            prefix.with_extension("bed"),
            [0x6c, 0x1b, 0x01, 0x31, 0x72, 0xB4, 0xA5],
        )
        .expect("write BED fixture");
        fs::write(
            prefix.with_extension("fam"),
            "1 1 0 0 1 -9\n2 2 0 0 1 -9\n3 3 0 0 1 -9\n4 4 0 0 1 -9\n",
        )
        .expect("write FAM fixture");

        let mut mmap =
            WindowedBedMatrix::open(prefix.to_str().unwrap(), 1).expect("open mmap BED window");
        let mut pread = WindowedBedMatrix::open_pread(prefix.to_str().unwrap(), 1)
            .expect("open positioned-read BED window");
        assert_eq!(
            mmap.read_source_range(1, 4).unwrap(),
            pread.read_source_range(1, 4).unwrap()
        );
        let mut mmap_rows = Vec::new();
        let mut pread_rows = Vec::new();
        let mmap_block = mmap
            .prepare_source_rows(&[0, 2, 3], &mut mmap_rows)
            .unwrap()
            .to_vec();
        let pread_block = pread
            .prepare_source_rows(&[0, 2, 3], &mut pread_rows)
            .unwrap()
            .to_vec();
        assert_eq!(pread_rows, mmap_rows);
        assert_eq!(pread_block, mmap_block);

        fs::remove_dir_all(dir).expect("remove temporary BED directory");
    }

    #[test]
    fn windowed_bed_matrix_populates_non_empty_platform_fingerprint() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_window_fp_{stamp}"));
        fs::create_dir_all(&dir).expect("create temporary BED directory");
        let prefix = dir.join("tiny");
        fs::write(
            prefix.with_extension("bed"),
            [0x6c, 0x1b, 0x01, 0x31, 0x72, 0xB4, 0xA5],
        )
        .expect("write BED fixture");
        fs::write(
            prefix.with_extension("fam"),
            "1 1 0 0 1 -9\n2 2 0 0 1 -9\n3 3 0 0 1 -9\n4 4 0 0 1 -9\n",
        )
        .expect("write FAM fixture");

        let mut matrix =
            WindowedBedMatrix::open(prefix.to_str().unwrap(), 1).expect("open BED window");
        let fp = matrix
            .source_fingerprint()
            .expect("BED source fingerprint must be populated")
            .to_string();
        assert!(!fp.is_empty());
        assert!(fp.starts_with("bed:"));
        #[cfg(unix)]
        {
            assert!(fp.contains("|dev=") && fp.contains("|ino="));
        }
        #[cfg(windows)]
        {
            assert!(fp.contains("|vol=") && fp.contains("|idx="));
        }
        let dup = matrix.duplicate().expect("duplicate BED window");
        assert_eq!(dup.source_fingerprint(), Some(fp.as_str()));

        matrix.clear_source_fingerprint_for_test();
        assert_eq!(matrix.source_fingerprint(), None);

        fs::remove_dir_all(dir).expect("remove temporary BED directory");
    }

    #[cfg(windows)]
    #[test]
    fn windows_file_identity_returns_non_empty_id_and_rejects_invalid_handle() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_win_id_{stamp}"));
        fs::create_dir_all(&dir).expect("create temporary dir");
        let path = dir.join("sample.bed");
        fs::write(&path, [0x6c, 0x1b, 0x01, 0x31]).expect("write temp file");

        let file = fs::File::open(&path).expect("open temp file");
        let (vol, idx, crt, wrt) =
            super::windows_file_identity(&file).expect("valid Windows file handle identity");
        assert!(vol != 0 || idx != 0);
        assert!(crt != 0 || wrt != 0);

        assert!(super::windows_file_identity_raw(std::ptr::null_mut()).is_err());
        assert!(super::windows_file_identity_raw((-1isize) as *mut std::ffi::c_void).is_err());

        drop(file);
        fs::remove_dir_all(dir).expect("remove temporary dir");
    }
}

// ---------------------------------------------------------------------------
// WindowedBedMatrix — bounded mmap or positioned-read blocks for BED scans
// ---------------------------------------------------------------------------

/// Use positioned reads for sequential LM/LMM BED scans by default.
/// Set `JX_LM_LMM_BED_PREAD=0` to retain the windowed mmap reader.
pub(crate) fn choose_lm_lmm_bed_pread(value: Option<&str>) -> bool {
    !matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("0" | "false" | "no" | "off")
    )
}

#[allow(dead_code)]
#[cfg(windows)]
fn windows_file_identity_raw(
    handle: *mut std::ffi::c_void,
) -> Result<(u32, u64, u64, u64), String> {
    #[repr(C)]
    struct FileTime {
        dw_low_date_time: u32,
        dw_high_date_time: u32,
    }

    #[repr(C)]
    struct ByHandleFileInformation {
        dw_file_attributes: u32,
        ft_creation_time: FileTime,
        ft_last_access_time: FileTime,
        ft_last_write_time: FileTime,
        dw_volume_serial_number: u32,
        n_file_size_high: u32,
        n_file_size_low: u32,
        n_number_of_links: u32,
        n_file_index_high: u32,
        n_file_index_low: u32,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetFileInformationByHandle(
            h_file: *mut std::ffi::c_void,
            lp_file_information: *mut ByHandleFileInformation,
        ) -> i32;
    }

    let mut info = std::mem::MaybeUninit::<ByHandleFileInformation>::zeroed();
    let ok = unsafe { GetFileInformationByHandle(handle, info.as_mut_ptr()) };
    if ok == 0 {
        return Err(format!(
            "GetFileInformationByHandle failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let info = unsafe { info.assume_init() };
    let file_index = ((info.n_file_index_high as u64) << 32) | (info.n_file_index_low as u64);
    if info.dw_volume_serial_number == 0 && file_index == 0 {
        return Err("Windows file handle returned empty volume/file identity".to_string());
    }
    let ctime = ((info.ft_creation_time.dw_high_date_time as u64) << 32)
        | (info.ft_creation_time.dw_low_date_time as u64);
    let wtime = ((info.ft_last_write_time.dw_high_date_time as u64) << 32)
        | (info.ft_last_write_time.dw_low_date_time as u64);
    Ok((info.dw_volume_serial_number, file_index, ctime, wtime))
}

#[allow(dead_code)]
#[cfg(windows)]
fn windows_file_identity(file: &std::fs::File) -> Result<(u32, u64, u64, u64), String> {
    use std::os::windows::io::AsRawHandle;
    windows_file_identity_raw(file.as_raw_handle())
}

fn pgen_file_fingerprint(pgen_path: &Path) -> Option<String> {
    let file = std::fs::File::open(pgen_path).ok()?;
    let meta = file.metadata().ok()?;
    let pgen_len = meta.len();
    let canon = std::fs::canonicalize(pgen_path)
        .ok()?
        .to_string_lossy()
        .into_owned();
    let mtime_ns = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let _ = &file;
        Some(format!(
            "pgen:{canon}|dev={}|ino={}|len={pgen_len}|mod={mtime_ns}|chg={}:{}",
            meta.dev(),
            meta.ino(),
            meta.ctime(),
            meta.ctime_nsec()
        ))
    }
    #[cfg(windows)]
    {
        let (vol, idx, crt, wrt) = windows_file_identity(&file).ok()?;
        if vol == 0 && idx == 0 {
            return None;
        }
        Some(format!(
            "pgen:{canon}|vol={vol}|idx={idx}|len={pgen_len}|mod={mtime_ns}|crt={crt}|wrt={wrt}"
        ))
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, canon, pgen_len, mtime_ns);
        None
    }
}

pub struct WindowedBedMatrix {
    file: Option<std::fs::File>,
    pgen: Option<crate::pgen::PgenReader>,
    pgen_path: Option<std::path::PathBuf>,
    resident: Option<Arc<ResidentPackedBed>>,
    source_fingerprint: Option<String>,
    n_samples_full: usize,
    bytes_per_snp: usize,
    bed_len: usize,
    mmap: Option<Mmap>,
    mmap_offset: usize,
    window_start_snp: usize,
    window_len_snps: usize,
    target_window_snps: usize,
    pread: bool,
    scratch: Vec<u8>,
    pread_buffer: Vec<u8>,
}

#[allow(dead_code)]
impl WindowedBedMatrix {
    pub fn open(prefix: &str, mmap_window_mb: usize) -> Result<Self, String> {
        Self::open_inner(prefix, mmap_window_mb, false)
    }

    pub fn open_pread(prefix: &str, window_mb: usize) -> Result<Self, String> {
        Self::open_inner(prefix, window_mb, true)
    }

    pub fn open_lm_lmm(prefix: &str, window_mb: usize) -> Result<Self, String> {
        if choose_lm_lmm_bed_pread(std::env::var("JX_LM_LMM_BED_PREAD").ok().as_deref()) {
            Self::open_pread(prefix, window_mb)
        } else {
            Self::open(prefix, window_mb)
        }
    }

    fn open_inner(prefix: &str, mmap_window_mb: usize, pread: bool) -> Result<Self, String> {
        if mmap_window_mb == 0 {
            return Err("mmap_window_mb must be > 0".to_string());
        }
        if let Some(resident) = lookup_resident_packed_bed(prefix) {
            let n_samples_full = resident.n_samples_full;
            let bytes_per_snp = resident.bytes_per_snp;
            let n_snps = resident.n_snps;
            let bed_len = BED_HEADER_LEN.saturating_add(resident.packed.len());
            let source_fingerprint = Some(format!(
                "resident:{}|gen={}|len={}",
                resident.key,
                resident.generation,
                resident.packed.len()
            ));
            return Ok(Self {
                file: None,
                pgen: None,
                pgen_path: None,
                resident: Some(resident),
                source_fingerprint,
                n_samples_full,
                bytes_per_snp,
                bed_len,
                mmap: None,
                mmap_offset: 0,
                window_start_snp: 0,
                window_len_snps: n_snps,
                target_window_snps: n_snps.max(1),
                pread: false,
                scratch: Vec::new(),
                pread_buffer: Vec::new(),
            });
        }
        if crate::pgen::path_uses_pgen(prefix) {
            let pgen_prefix = crate::pgen::normalize_pgen_prefix(prefix);
            let n_samples_full = crate::pgen::count_psam_samples(&pgen_prefix)?;
            if n_samples_full == 0 {
                return Err("no samples found in PGEN".to_string());
            }
            let bytes_per_snp = n_samples_full.div_ceil(4);
            let n_snps = crate::pgen::count_pgen_variants(&pgen_prefix, n_samples_full)?;
            let pgen_path = Path::new(&format!("{pgen_prefix}.pgen")).to_path_buf();
            let reader = crate::pgen::PgenReader::open(&pgen_path, n_samples_full, n_snps)?;
            let source_fingerprint = pgen_file_fingerprint(&pgen_path);
            let payload_len = n_snps
                .checked_mul(bytes_per_snp)
                .ok_or_else(|| "PGEN packed payload size overflow".to_string())?;
            let bed_len = BED_HEADER_LEN
                .checked_add(payload_len)
                .ok_or_else(|| "PGEN packed length overflow".to_string())?;
            let window_bytes = mmap_window_mb.saturating_mul(1024 * 1024);
            let target_window_snps = std::cmp::max(1, window_bytes / bytes_per_snp);
            return Ok(Self {
                file: None,
                pgen: Some(reader),
                pgen_path: Some(pgen_path),
                resident: None,
                source_fingerprint,
                n_samples_full,
                bytes_per_snp,
                bed_len,
                mmap: None,
                mmap_offset: 0,
                window_start_snp: 0,
                window_len_snps: 0,
                target_window_snps,
                pread: false,
                scratch: Vec::new(),
                pread_buffer: Vec::new(),
            });
        }
        let bed_prefix = normalize_plink_prefix(prefix);
        let n_samples_full = gfcore::read_fam(&bed_prefix)
            .map_err(|e| e.to_string())?
            .len();
        if n_samples_full == 0 {
            return Err("no samples found in BED".to_string());
        }
        let bytes_per_snp = n_samples_full.div_ceil(4);
        let bed_path = format!("{bed_prefix}.bed");
        let file = std::fs::File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        let mut header = [0u8; BED_HEADER_LEN];
        read_file_exact_at(&file, &mut header, 0)
            .map_err(|e| format!("read BED header {bed_path}: {e}"))?;
        if header != [0x6C, 0x1B, 0x01] {
            return Err("only SNP-major BED supported".to_string());
        }
        let meta = file
            .metadata()
            .map_err(|e| format!("metadata {bed_path}: {e}"))?;
        let bed_len = meta.len() as usize;
        if bed_len < BED_HEADER_LEN {
            return Err("BED too small".to_string());
        }
        let source_fingerprint = {
            let base = std::fs::canonicalize(&bed_path)
                .ok()
                .and_then(|canon_path| {
                    let mtime_ns = meta
                        .modified()
                        .ok()?
                        .duration_since(std::time::UNIX_EPOCH)
                        .ok()?
                        .as_nanos();
                    Some((canon_path.to_string_lossy().into_owned(), mtime_ns))
                });
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                base.map(|(canon, mtime_ns)| {
                    format!(
                        "bed:{canon}|dev={}|ino={}|len={bed_len}|mod={mtime_ns}|chg={}:{}",
                        meta.dev(),
                        meta.ino(),
                        meta.ctime(),
                        meta.ctime_nsec()
                    )
                })
            }
            #[cfg(windows)]
            {
                base.and_then(|(canon, mtime_ns)| {
                    let (vol, idx, crt, wrt) = windows_file_identity(&file).ok()?;
                    Some(format!(
                        "bed:{canon}|vol={vol}|idx={idx}|len={bed_len}|mod={mtime_ns}|crt={crt}|wrt={wrt}"
                    ))
                })
            }
            #[cfg(not(any(unix, windows)))]
            {
                let _ = base;
                None
            }
        };
        let payload_len = bed_len - BED_HEADER_LEN;
        if bytes_per_snp == 0 || payload_len % bytes_per_snp != 0 {
            return Err(format!(
                "invalid BED payload length: data_len={payload_len}, bytes_per_snp={bytes_per_snp}"
            ));
        }
        let window_bytes = mmap_window_mb.saturating_mul(1024 * 1024);
        let target_window_snps = std::cmp::max(1, window_bytes / bytes_per_snp);
        let (mmap, mmap_offset, window_len_snps) = if pread {
            (None, 0, 0)
        } else {
            let (mmap, mmap_offset, window_len_snps) =
                Self::map_window(&file, bed_len, 0, target_window_snps, bytes_per_snp)?;
            (Some(mmap), mmap_offset, window_len_snps)
        };
        Ok(Self {
            file: Some(file),
            pgen: None,
            pgen_path: None,
            resident: None,
            source_fingerprint,
            n_samples_full,
            bytes_per_snp,
            bed_len,
            mmap,
            mmap_offset,
            window_start_snp: 0,
            window_len_snps,
            target_window_snps,
            pread,
            scratch: Vec::new(),
            pread_buffer: Vec::new(),
        })
    }

    #[inline]
    pub(crate) fn source_fingerprint(&self) -> Option<&str> {
        self.source_fingerprint.as_deref()
    }

    #[cfg(test)]
    pub(crate) fn clear_source_fingerprint_for_test(&mut self) {
        self.source_fingerprint = None;
    }

    /// Create an independent mmap/window cursor over the same BED payload.
    ///
    /// The duplicate is used by streamed Bayesian double buffering: one
    /// cursor belongs to the decode worker while the caller consumes the
    /// previously decoded block.  The file and mmap are independent, but the
    /// underlying BED pages remain shared by the operating system.
    pub fn duplicate(&self) -> Result<Self, String> {
        if let Some(resident) = self.resident.as_ref() {
            return Ok(Self {
                file: None,
                pgen: None,
                pgen_path: None,
                resident: Some(Arc::clone(resident)),
                source_fingerprint: self.source_fingerprint.clone(),
                n_samples_full: self.n_samples_full,
                bytes_per_snp: self.bytes_per_snp,
                bed_len: self.bed_len,
                mmap: None,
                mmap_offset: 0,
                window_start_snp: 0,
                window_len_snps: self.window_len_snps,
                target_window_snps: self.target_window_snps,
                pread: false,
                scratch: Vec::new(),
                pread_buffer: Vec::new(),
            });
        }
        if let Some(pgen_path) = self.pgen_path.as_ref() {
            let reader = crate::pgen::PgenReader::open(
                pgen_path,
                self.n_samples_full,
                self.n_source_snps(),
            )?;
            return Ok(Self {
                file: None,
                pgen: Some(reader),
                pgen_path: Some(pgen_path.clone()),
                resident: None,
                source_fingerprint: self.source_fingerprint.clone(),
                n_samples_full: self.n_samples_full,
                bytes_per_snp: self.bytes_per_snp,
                bed_len: self.bed_len,
                mmap: None,
                mmap_offset: 0,
                window_start_snp: 0,
                window_len_snps: 0,
                target_window_snps: self.target_window_snps,
                pread: false,
                scratch: Vec::new(),
                pread_buffer: Vec::new(),
            });
        }
        let file = self
            .file
            .as_ref()
            .ok_or_else(|| "internal error: missing BED file handle".to_string())?
            .try_clone()
            .map_err(|e| e.to_string())?;
        if self.pread {
            return Ok(Self {
                file: Some(file),
                pgen: None,
                pgen_path: None,
                resident: None,
                source_fingerprint: self.source_fingerprint.clone(),
                n_samples_full: self.n_samples_full,
                bytes_per_snp: self.bytes_per_snp,
                bed_len: self.bed_len,
                mmap: None,
                mmap_offset: 0,
                window_start_snp: 0,
                window_len_snps: 0,
                target_window_snps: self.target_window_snps,
                pread: true,
                scratch: Vec::new(),
                pread_buffer: Vec::new(),
            });
        }
        // Every Bayesian scan starts at marker zero.  Do not inherit the
        // caller's last window (which is usually near the end of the BED
        // file), otherwise the worker would immediately remap before its
        // first block.
        let (mmap, mmap_offset, window_len_snps) = Self::map_window(
            &file,
            self.bed_len,
            0,
            self.target_window_snps,
            self.bytes_per_snp,
        )?;
        Ok(Self {
            file: Some(file),
            pgen: None,
            pgen_path: None,
            resident: None,
            source_fingerprint: self.source_fingerprint.clone(),
            n_samples_full: self.n_samples_full,
            bytes_per_snp: self.bytes_per_snp,
            bed_len: self.bed_len,
            mmap: Some(mmap),
            mmap_offset,
            window_start_snp: 0,
            window_len_snps,
            target_window_snps: self.target_window_snps,
            pread: false,
            scratch: Vec::new(),
            pread_buffer: Vec::new(),
        })
    }

    #[inline]
    pub fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }

    #[inline]
    pub fn bytes_per_snp(&self) -> usize {
        self.bytes_per_snp
    }

    #[inline]
    pub fn n_source_snps(&self) -> usize {
        self.bed_len
            .saturating_sub(BED_HEADER_LEN)
            .saturating_div(self.bytes_per_snp.max(1))
    }

    fn map_window(
        file: &std::fs::File,
        bed_len: usize,
        start_snp: usize,
        window_snps: usize,
        bytes_per_snp: usize,
    ) -> Result<(Mmap, usize, usize), String> {
        let data_len = bed_len
            .checked_sub(BED_HEADER_LEN)
            .ok_or_else(|| "BED too small".to_string())?;
        let max_snps = data_len / bytes_per_snp;
        if start_snp >= max_snps {
            return Err("window start SNP out of range".to_string());
        }
        let remaining_snps = max_snps - start_snp;
        let map_snps = remaining_snps.min(window_snps.max(1));
        let desired_offset = BED_HEADER_LEN + start_snp.saturating_mul(bytes_per_snp);
        let page_size = system_page_size();
        let aligned_offset = desired_offset / page_size * page_size;
        let leading = desired_offset - aligned_offset;
        let desired_len = leading + map_snps.saturating_mul(bytes_per_snp);
        let max_len = bed_len - aligned_offset;
        let map_len = desired_len.min(max_len);
        let mmap = unsafe {
            MmapOptions::new()
                .offset(aligned_offset as u64)
                .len(map_len)
                .map(file)
                .map_err(|e| e.to_string())?
        };
        Ok((mmap, aligned_offset, map_snps))
    }

    fn remap_window_for_range(&mut self, start_snp: usize, end_snp: usize) -> Result<(), String> {
        if end_snp <= start_snp {
            return Err("invalid BED window range".to_string());
        }
        let file = self
            .file
            .as_ref()
            .ok_or_else(|| "internal error: missing BED file handle".to_string())?;
        let span_snps = end_snp - start_snp;
        let window_snps = self.target_window_snps.max(span_snps);
        let (mmap, mmap_offset, window_len_snps) = Self::map_window(
            file,
            self.bed_len,
            start_snp,
            window_snps,
            self.bytes_per_snp,
        )?;
        self.mmap = Some(mmap);
        self.mmap_offset = mmap_offset;
        self.window_start_snp = start_snp;
        self.window_len_snps = window_len_snps;
        Ok(())
    }

    pub fn read_source_range(&mut self, start: usize, end: usize) -> Result<&[u8], String> {
        if end <= start {
            return Ok(&[]);
        }
        if self.resident.is_none() && self.pgen.is_none() && !self.pread {
            let window_end = self.window_start_snp + self.window_len_snps;
            if start < self.window_start_snp || end > window_end {
                self.remap_window_for_range(start, end)?;
            }
        }
        if let Some(resident) = self.resident.as_ref() {
            if end > resident.n_snps {
                return Err(format!(
                    "resident BED range out of bounds: start={start}, end={end}, n_snps={}",
                    resident.n_snps
                ));
            }
            let byte_start = start.saturating_mul(self.bytes_per_snp);
            let byte_end = end.saturating_mul(self.bytes_per_snp);
            return Ok(&resident.packed[byte_start..byte_end]);
        }
        if let Some(pgen) = self.pgen.as_mut() {
            let n_snps = self
                .bed_len
                .saturating_sub(BED_HEADER_LEN)
                .saturating_div(self.bytes_per_snp.max(1));
            if end > n_snps {
                return Err(format!(
                    "PGEN range out of bounds: start={start}, end={end}, n_snps={n_snps}"
                ));
            }
            let rows = end - start;
            let len = rows.saturating_mul(self.bytes_per_snp);
            self.pread_buffer.resize(len, 0);
            let got = pgen.read_source_block(start, rows, &mut self.pread_buffer)?;
            if got != rows {
                return Err(format!(
                    "PGEN read returned {got} rows; requested {rows} starting at {start}"
                ));
            }
            return Ok(&self.pread_buffer);
        }
        if self.pread {
            if end > self.n_source_snps() {
                return Err(format!(
                    "BED positioned-read range out of bounds: start={start}, end={end}, n_snps={}",
                    self.n_source_snps()
                ));
            }
            let file = self
                .file
                .as_ref()
                .ok_or_else(|| "internal error: missing BED file handle".to_string())?;
            let len = (end - start).saturating_mul(self.bytes_per_snp);
            self.pread_buffer.resize(len, 0);
            let offset =
                BED_HEADER_LEN.saturating_add(start.saturating_mul(self.bytes_per_snp)) as u64;
            read_file_exact_at(file, &mut self.pread_buffer, offset)?;
            return Ok(&self.pread_buffer);
        }
        let desired_offset = BED_HEADER_LEN + start.saturating_mul(self.bytes_per_snp);
        let local_offset = desired_offset
            .checked_sub(self.mmap_offset)
            .ok_or_else(|| "BED window local offset underflow".to_string())?;
        let len = (end - start).saturating_mul(self.bytes_per_snp);
        let end_offset = local_offset.saturating_add(len);
        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| "internal error: mmap BED window is unavailable".to_string())?;
        if end_offset > mmap.len() {
            return Err(format!(
                "BED window slice out of bounds: local_offset={local_offset}, len={len}, mmap_len={}",
                mmap.len()
            ));
        }
        Ok(&mmap[local_offset..end_offset])
    }

    fn decode_sparse_rows_to_scratch(
        &mut self,
        source_indices: &[usize],
        rel_indices_out: &mut Vec<usize>,
    ) -> Result<&[u8], String> {
        let rows_here = source_indices.len();
        let bps = self.bytes_per_snp;
        let need = rows_here.saturating_mul(self.bytes_per_snp);
        if self.scratch.len() < need {
            self.scratch.resize(need, 0);
        }
        rel_indices_out.clear();
        if let Some(pgen) = self.pgen.as_mut() {
            pgen.read_source_rows_sparse(source_indices, &mut self.scratch[..need])?;
            rel_indices_out.extend(0..rows_here);
            return Ok(&self.scratch[..need]);
        }
        for (local_idx, &src_idx) in source_indices.iter().enumerate() {
            let row_bytes = self.read_source_range(src_idx, src_idx + 1)?.to_vec();
            let dst = &mut self.scratch[local_idx * bps..(local_idx + 1) * bps];
            dst.copy_from_slice(row_bytes.as_slice());
            rel_indices_out.push(local_idx);
        }
        Ok(&self.scratch[..need])
    }

    pub fn prepare_source_rows(
        &mut self,
        source_indices: &[usize],
        rel_indices_out: &mut Vec<usize>,
    ) -> Result<&[u8], String> {
        if source_indices.is_empty() {
            rel_indices_out.clear();
            return Ok(&[]);
        }
        let monotonic = source_indices.windows(2).all(|w| w[0] <= w[1]);
        let src_start = source_indices[0];
        let src_end = source_indices[source_indices.len() - 1].saturating_add(1);
        let source_span = src_end.saturating_sub(src_start);
        let max_span = source_indices.len().saturating_mul(8).max(1);
        let zero_copy_ok = monotonic && src_end > src_start && source_span <= max_span;
        if zero_copy_ok {
            rel_indices_out.clear();
            rel_indices_out.extend(source_indices.iter().map(|&src_idx| src_idx - src_start));
            self.read_source_range(src_start, src_end)
        } else {
            self.decode_sparse_rows_to_scratch(source_indices, rel_indices_out)
        }
    }

    pub fn prepare_block(
        &mut self,
        stats: &GlobalStats,
        row_start: usize,
        rows_here: usize,
        rel_indices_out: &mut Vec<usize>,
    ) -> Result<&[u8], String> {
        let source_indices = stats.row_source_slice(row_start, rows_here);
        self.prepare_source_rows(source_indices, rel_indices_out)
    }
}

impl GenotypeMatrix for WindowedBedMatrix {
    fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }
    fn bytes_per_snp(&self) -> usize {
        self.bytes_per_snp
    }
    fn packed_flat(&self) -> &[u8] {
        self.resident
            .as_deref()
            .map_or(&[], |res| res.packed.as_slice())
    }
    fn source_row_bytes(&self, source_idx: usize) -> &[u8] {
        if let Some(res) = self.resident.as_deref() {
            let start = source_idx.saturating_mul(self.bytes_per_snp);
            let end = start.saturating_add(self.bytes_per_snp);
            res.packed.get(start..end).unwrap_or(&[])
        } else {
            &[]
        }
    }

    fn decode_additive_block(
        &mut self,
        stats: &GlobalStats,
        row_start: usize,
        out: &mut [f32],
        sample_idx: &[usize],
        sample_identity: bool,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) -> Result<(), String> {
        let cols = if sample_identity {
            stats.n_samples_full
        } else {
            sample_idx.len()
        };
        let rows_here = out.len().saturating_div(cols.max(1));
        if rows_here == 0 {
            return Ok(());
        }
        let mut rel_indices = Vec::with_capacity(rows_here);
        let packed_slice = self.prepare_block(stats, row_start, rows_here, &mut rel_indices)?;
        let code4_lut = &packed_byte_lut().code4;
        decode_mean_imputed_additive_packed_block_rows_f32(
            packed_slice,
            stats.bytes_per_snp,
            stats.n_samples_full,
            stats.row_flip_slice(row_start, rows_here),
            stats.row_maf_slice(row_start, rows_here),
            sample_idx,
            sample_identity,
            Some(rel_indices.as_slice()),
            0,
            out,
            code4_lut,
            pool,
        )
    }
}

// ---------------------------------------------------------------------------
// BedMmapMatrix
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct BedMmapMatrix {
    mmap: Arc<Mmap>,
    n_samples_full: usize,
    bytes_per_snp: usize,
    bed_prefix: String,
}

#[allow(dead_code)]
impl BedMmapMatrix {
    pub fn open(prefix: &str) -> Result<Self, String> {
        let bed_prefix = normalize_plink_prefix(prefix);
        let n_samples_full = gfcore::read_fam(&bed_prefix)
            .map_err(|e| e.to_string())?
            .len();
        if n_samples_full == 0 {
            return Err("no samples found in BED".to_string());
        }
        let bytes_per_snp = n_samples_full.div_ceil(4);
        let bed_path = format!("{bed_prefix}.bed");
        let bed_file = File::open(&bed_path).map_err(|e| format!("open {bed_path}: {e}"))?;
        let mmap = unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("mmap {bed_path}: {e}"))?;
        if mmap.len() < 3 || mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
            return Err("invalid or non-SNP-major BED".to_string());
        }
        Ok(Self {
            mmap: Arc::new(mmap),
            n_samples_full,
            bytes_per_snp,
            bed_prefix,
        })
    }

    pub fn bed_prefix(&self) -> &str {
        &self.bed_prefix
    }

    /// Number of total SNP rows in the mmap'd BED.
    pub fn n_snps_total(&self) -> usize {
        let payload = self.mmap.len().saturating_sub(3);
        payload / self.bytes_per_snp
    }
}

impl GenotypeMatrix for BedMmapMatrix {
    fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }
    fn bytes_per_snp(&self) -> usize {
        self.bytes_per_snp
    }

    fn packed_flat(&self) -> &[u8] {
        &self.mmap[3..]
    }

    fn source_row_bytes(&self, source_idx: usize) -> &[u8] {
        let offset = 3usize.saturating_add(source_idx.saturating_mul(self.bytes_per_snp));
        &self.mmap[offset..][..self.bytes_per_snp]
    }
}

// ---------------------------------------------------------------------------
// PackedBedMatrix
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct PackedBedMatrix {
    payload: Arc<[u8]>,
    n_samples_full: usize,
    bytes_per_snp: usize,
}

#[allow(dead_code)]
impl PackedBedMatrix {
    pub fn from_packed(payload: Arc<[u8]>, n_samples_full: usize) -> Result<Self, String> {
        let bytes_per_snp = n_samples_full.div_ceil(4);
        if payload.len() % bytes_per_snp != 0 {
            return Err("packed payload length is not a multiple of bytes_per_snp".to_string());
        }
        Ok(Self {
            payload,
            n_samples_full,
            bytes_per_snp,
        })
    }

    pub fn n_markers_contiguous(&self) -> usize {
        self.payload.len() / self.bytes_per_snp
    }
}

impl GenotypeMatrix for PackedBedMatrix {
    fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }
    fn bytes_per_snp(&self) -> usize {
        self.bytes_per_snp
    }

    fn packed_flat(&self) -> &[u8] {
        &self.payload
    }

    /// For PackedBedMatrix, the payload IS the contiguous kept-marker data,
    /// so local_idx directly indexes into the payload.
    fn source_row_bytes(&self, local_idx: usize) -> &[u8] {
        let start = local_idx.saturating_mul(self.bytes_per_snp);
        &self.payload[start..start.saturating_add(self.bytes_per_snp)]
    }
}

// ---------------------------------------------------------------------------
// UnifiedInput
// ---------------------------------------------------------------------------

pub struct UnifiedInput<G: GenotypeMatrix> {
    pub matrix: G,
    pub stats: GlobalStats,
}

#[allow(dead_code)]
impl<G: GenotypeMatrix> UnifiedInput<G> {
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.stats.n_samples_full
    }

    #[inline]
    pub fn n_markers(&self) -> usize {
        self.stats.n_markers()
    }

    /// Block-decode a range of kept markers (additive, mean-imputed) into
    /// `out` which must be `rows_here × n_samples` f32 contiguous.
    ///
    /// `sample_idx` / `sample_identity` control per-sample byte-index
    /// mapping.  When all kept samples are included in identity order, pass
    /// `(&[], true)`.
    pub fn decode_additive_block(
        &self,
        row_start: usize,
        out: &mut [f32],
        sample_idx: &[usize],
        sample_identity: bool,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) -> Result<(), String> {
        let cols = if sample_identity {
            self.n_samples()
        } else {
            sample_idx.len()
        };
        let rows_here = out.len().saturating_div(cols.max(1));
        if rows_here == 0 {
            return Ok(());
        }
        let code4_lut = &packed_byte_lut().code4;
        let row_source_indices = self.stats.row_source_slice(row_start, rows_here);
        decode_mean_imputed_additive_packed_block_rows_f32(
            self.matrix.packed_flat(),
            self.stats.bytes_per_snp,
            self.stats.n_samples_full,
            self.stats.row_flip_slice(row_start, rows_here),
            self.stats.row_maf_slice(row_start, rows_here),
            sample_idx,
            sample_identity,
            Some(row_source_indices),
            0,
            out,
            code4_lut,
            pool,
        )
    }

    /// Decode from a pre-read packed slice (used by streaming backends).
    /// `packed_slice` covers the source range for this block; `rel_indices`
    /// are source-index offsets relative to the start of `packed_slice`.
    #[inline]
    pub fn decode_additive_block_from_slice(
        &self,
        packed_slice: &[u8],
        rel_indices: &[usize],
        row_start: usize,
        rows_here: usize,
        out: &mut [f32],
        sample_idx: &[usize],
        sample_identity: bool,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) -> Result<(), String> {
        let code4_lut = &packed_byte_lut().code4;
        decode_mean_imputed_additive_packed_block_rows_f32(
            packed_slice,
            self.stats.bytes_per_snp,
            self.stats.n_samples_full,
            self.stats.row_flip_slice(row_start, rows_here),
            self.stats.row_maf_slice(row_start, rows_here),
            sample_idx,
            sample_identity,
            Some(rel_indices),
            0,
            out,
            code4_lut,
            pool,
        )
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub fn compute_global_stats_mmap(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    sample_indices: Option<&[usize]>,
    threads: usize,
) -> Result<GlobalStats, String> {
    let meta = compute_bed_filter_meta(
        prefix,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
        sample_indices,
        BedFilterMetaMode::FullDecodeMeta,
        threads,
    )?;
    Ok(GlobalStats {
        maf: meta.maf,
        miss: meta.miss,
        row_flip: meta.row_flip,
        row_source_indices: meta.row_source_indices,
        site_keep: meta.site_keep,
        n_samples_full: meta.n_samples_full,
        n_markers_total: meta.n_markers_total,
        bytes_per_snp: meta.bytes_per_snp,
    })
}

#[allow(dead_code)]
pub fn open_bed_mmap_unified(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    sample_indices: Option<&[usize]>,
    threads: usize,
) -> Result<UnifiedInput<BedMmapMatrix>, String> {
    let stats = compute_global_stats_mmap(
        prefix,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
        sample_indices,
        threads,
    )?;
    let matrix = BedMmapMatrix::open(prefix)?;
    Ok(UnifiedInput { matrix, stats })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn normalize_plink_prefix(prefix: &str) -> String {
    let mut out = prefix.trim().to_string();
    let lower = out.to_ascii_lowercase();
    for suffix in [
        ".bed",
        ".bim",
        ".fam",
        ".pgen",
        ".pvar.zst",
        ".pvar",
        ".psam",
    ] {
        if lower.ends_with(suffix) {
            out.truncate(out.len().saturating_sub(suffix.len()));
            break;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Centered decode (zero-mean, used by LMM / FastLMM / FvLMM)
// ---------------------------------------------------------------------------

/// Centered block decode for any GenotypeMatrix backend.
/// Uses the shared `crate::decode` packed centered-decode path internally.
#[allow(dead_code)]
pub fn decode_centered_block_unified<G: GenotypeMatrix>(
    matrix: &G,
    stats: &GlobalStats,
    row_start: usize,
    gm: PackedGeneticModel,
    out: &mut [f32],
    sample_idx: &[usize],
    sample_identity: bool,
) -> Result<(), String> {
    let n = if sample_identity {
        stats.n_samples_full
    } else {
        sample_idx.len()
    };
    let rows_here = out.len().saturating_div(n.max(1));
    if rows_here == 0 {
        return Ok(());
    }
    decode_centered_block_packed_model_f32(
        matrix.packed_flat(),
        stats.bytes_per_snp,
        stats.row_flip_slice(row_start, rows_here),
        stats.row_maf_slice(row_start, rows_here),
        Some(stats.row_source_slice(row_start, rows_here)),
        row_start,
        rows_here,
        n,
        gm,
        sample_identity,
        None,
        None,
        out,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    static RESIDENT_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn write_toy_pgen_dataset(prefix: &str, n_samples: usize, n_snps: usize) -> Vec<u8> {
        let mut psam = String::from("#FID\tIID\tSEX\n");
        for i in 0..n_samples {
            psam.push_str(&format!("F{i}\tI{i}\t0\n"));
        }
        std::fs::write(format!("{prefix}.psam"), psam).expect("write psam");

        let mut pvar = String::from("#CHROM\tPOS\tID\tREF\tALT\n");
        for j in 0..n_snps {
            pvar.push_str(&format!("1\t{}\trs{j}\tA\tG\n", (j + 1) * 100));
        }
        std::fs::write(format!("{prefix}.pvar"), pvar).expect("write pvar");

        let bps = n_samples.div_ceil(4);
        let mut pgen = vec![0x6C, 0x1B, 0x02];
        pgen.extend_from_slice(&(n_snps as u32).to_le_bytes());
        pgen.extend_from_slice(&(n_samples as u32).to_le_bytes());
        pgen.push(0);
        let mut expected_bed = vec![0u8; n_snps * bps];
        for r in 0..n_snps {
            let mut cat_row = vec![0u8; bps];
            for s in 0..n_samples {
                let cat = ((r * 3 + s) % 4) as u8;
                cat_row[s / 4] |= cat << ((s % 4) * 2);
            }
            pgen.extend_from_slice(&cat_row);
            for b in 0..bps {
                expected_bed[r * bps + b] = crate::pgen::pgen_category_byte_to_row_byte(cat_row[b]);
            }
        }
        std::fs::write(format!("{prefix}.pgen"), pgen).expect("write pgen");
        expected_bed
    }

    fn write_toy_pgen_var_dataset(prefix: &str, n_samples: usize, n_snps: usize) -> Vec<u8> {
        let mut psam = String::from("#FID\tIID\tSEX\n");
        for i in 0..n_samples {
            psam.push_str(&format!("F{i}\tI{i}\t0\n"));
        }
        std::fs::write(format!("{prefix}.psam"), psam).expect("write psam");

        let mut pvar = String::from("#CHROM\tPOS\tID\tREF\tALT\n");
        for j in 0..n_snps {
            pvar.push_str(&format!("1\t{}\trs{j}\tA\tG\n", (j + 1) * 100));
        }
        std::fs::write(format!("{prefix}.pvar"), pvar).expect("write pvar");

        let bps = n_samples.div_ceil(4);
        assert!(bps <= u8::MAX as usize);
        let mut meta = vec![0u8; n_snps.div_ceil(2)];
        meta.resize(n_snps.div_ceil(2) + n_snps, bps as u8);
        let data_offset = (12 + 8 + meta.len()) as u64;
        let mut pgen = Vec::new();
        pgen.extend_from_slice(&[0x6C, 0x1B, 0x10]);
        pgen.extend_from_slice(&(n_snps as u32).to_le_bytes());
        pgen.extend_from_slice(&(n_samples as u32).to_le_bytes());
        pgen.push(0);
        pgen.extend_from_slice(&data_offset.to_le_bytes());
        pgen.extend_from_slice(&meta);
        let mut expected_bed = vec![0u8; n_snps * bps];
        for r in 0..n_snps {
            let mut cat_row = vec![0u8; bps];
            for s in 0..n_samples {
                let cat = ((r * 5 + s + 1) % 4) as u8;
                cat_row[s / 4] |= cat << ((s % 4) * 2);
            }
            pgen.extend_from_slice(&cat_row);
            for b in 0..bps {
                expected_bed[r * bps + b] = crate::pgen::pgen_category_byte_to_row_byte(cat_row[b]);
            }
        }
        std::fs::write(format!("{prefix}.pgen"), pgen).expect("write pgen");
        expected_bed
    }

    #[test]
    fn windowed_bed_matrix_reads_pgen_dense_sparse_duplicate_and_resident() {
        pyo3::Python::initialize();
        let n_samples = 13usize;
        let n_snps = 40usize;
        let bps = n_samples.div_ceil(4);
        let tag = format!("janusx_gload_pgen_win_{}", std::process::id());
        let dir = std::env::temp_dir().join(tag);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let prefix = dir.join("toy").to_string_lossy().to_string();
        let expected_bed = write_toy_pgen_dataset(&prefix, n_samples, n_snps);

        let mut mat = WindowedBedMatrix::open(&prefix, 1).expect("open PGEN in WindowedBedMatrix");
        assert_eq!(mat.n_samples_full(), n_samples);
        assert_eq!(mat.bytes_per_snp(), bps);
        assert_eq!(mat.n_source_snps(), n_snps);
        assert!(mat.source_fingerprint().is_some());

        let full = mat.read_source_range(0, n_snps).expect("read full PGEN");
        assert_eq!(full, expected_bed.as_slice());

        let mut rel = Vec::new();
        let dense_idx = [2usize, 3, 5, 7];
        let dense_slice = mat
            .prepare_source_rows(&dense_idx, &mut rel)
            .expect("dense PGEN rows");
        assert_eq!(rel, vec![0, 1, 3, 5]);
        for (&src_r, &rel_r) in dense_idx.iter().zip(rel.iter()) {
            assert_eq!(
                &dense_slice[rel_r * bps..(rel_r + 1) * bps],
                &expected_bed[src_r * bps..(src_r + 1) * bps]
            );
        }

        let sparse_idx = [1usize, 15, 38];
        let sparse_slice = mat
            .prepare_source_rows(&sparse_idx, &mut rel)
            .expect("sparse PGEN rows");
        assert_eq!(rel, vec![0, 1, 2]);
        for (&src_r, &rel_r) in sparse_idx.iter().zip(rel.iter()) {
            assert_eq!(
                &sparse_slice[rel_r * bps..(rel_r + 1) * bps],
                &expected_bed[src_r * bps..(src_r + 1) * bps]
            );
        }

        let mut dup = mat.duplicate().expect("duplicate PGEN cursor");
        let dup_slice = dup.read_source_range(4, 12).expect("read dup PGEN");
        assert_eq!(dup_slice, &expected_bed[4 * bps..12 * bps]);
        drop(dup);
        drop(mat);

        let _resident_guard = RESIDENT_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (entry, token) = load_resident_packed_bed(&prefix, 2).expect("preload PGEN resident");
        assert_eq!(entry.packed.as_slice(), expected_bed.as_slice());
        let mut res_mat = WindowedBedMatrix::open(&prefix, 1).expect("open resident PGEN");
        assert_eq!(
            res_mat.read_source_range(0, n_snps).expect("read resident"),
            expected_bed.as_slice()
        );
        let mut reader = PackedBedSourceReader::open(&prefix, n_samples).expect("open reader");
        let mut stats = vec![crate::pgen::GenotypeRowStats::default(); n_snps];
        let mut keep = vec![false; n_snps];
        let got = reader
            .read_source_block_qc_only(
                0,
                n_snps,
                crate::pgen::GenotypeSampleSelection::all(),
                &mut stats,
                &mut keep,
                &|_, _| true,
            )
            .expect("resident parallel qc");
        assert_eq!(got, n_snps);
        assert!(keep.iter().all(|&k| k));
        drop(reader);
        drop(res_mat);
        drop(entry);
        clear_resident_packed_bed(Some(token));

        std::fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }

    #[test]
    fn windowed_bed_matrix_reads_variable_mode_0x10_pgen() {
        pyo3::Python::initialize();
        let n_samples = 13usize;
        let n_snps = 40usize;
        let bps = n_samples.div_ceil(4);
        let tag = format!("janusx_gload_pgen_var10_{}", std::process::id());
        let dir = std::env::temp_dir().join(tag);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let prefix = dir.join("toy_var").to_string_lossy().to_string();
        let expected_bed = write_toy_pgen_var_dataset(&prefix, n_samples, n_snps);

        let mut mat = WindowedBedMatrix::open(&prefix, 1).expect("open 0x10 PGEN");
        assert_eq!(mat.n_samples_full(), n_samples);
        assert_eq!(mat.bytes_per_snp(), bps);
        assert_eq!(mat.n_source_snps(), n_snps);
        assert_eq!(
            mat.read_source_range(0, n_snps).expect("read full 0x10"),
            expected_bed.as_slice()
        );

        let mut rel = Vec::new();
        let sparse_idx = [0usize, 17, 39];
        let sparse_slice = mat
            .prepare_source_rows(&sparse_idx, &mut rel)
            .expect("sparse 0x10 rows");
        assert_eq!(rel, vec![0, 1, 2]);
        for (&src_r, &rel_r) in sparse_idx.iter().zip(rel.iter()) {
            assert_eq!(
                &sparse_slice[rel_r * bps..(rel_r + 1) * bps],
                &expected_bed[src_r * bps..(src_r + 1) * bps]
            );
        }

        let mut dup = mat.duplicate().expect("duplicate 0x10 PGEN");
        assert_eq!(
            dup.read_source_range(5, 15).expect("read dup 0x10"),
            &expected_bed[5 * bps..15 * bps]
        );
        drop(dup);
        drop(mat);

        let _resident_guard = RESIDENT_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (entry, token) = load_resident_packed_bed(&prefix, 2).expect("preload 0x10 resident");
        assert_eq!(entry.packed.as_slice(), expected_bed.as_slice());
        drop(entry);
        clear_resident_packed_bed(Some(token));

        std::fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }
}
