use memmap2::Mmap;
use nalgebra::{DMatrix, SymmetricEigen};
use numpy::ndarray::{Array1, Array2, ArrayView2};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::BoundObject;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::convert::TryFrom;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::mem::align_of;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;
use std::time::{Instant, UNIX_EPOCH};

use crate::bedmath::packed_byte_lut;
use crate::blas::{
    cblas_sgemm_dispatch, BlasThreadGuard, CblasInt, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS,
};
use crate::gfcore::{
    parse_positive_env_usize, process_snp_row, read_fam, BedSnpIter, HmpSnpIter, TxtSnpIter,
    VcfSnpIter,
};
use crate::gload::{
    load_file_owned_exact, mmap_readonly_file, open_plink_bed_reader_after_header,
    read_plink_bed_header_info,
};
use crate::he::{row_major_block_mul_mat_f32, row_major_block_t_mul_mat_accum_f32};
use crate::rsvd::{
    rsvd_block_rows_with_target_mb, rsvd_packed_compute_a_omega,
    rsvd_packed_compute_at_random_omega, rsvd_packed_compute_ata_omega,
    rsvd_packed_compute_gram_aq, rsvd_project_sample_eigvecs, rsvd_right_singular_from_gram,
    PackedRsvdView, RsvdKernelTiming,
};
use crate::stats_common::{
    admx_madvise_dontneed_bytes, check_admx_memory_limit, process_memory_usage,
};

const EPS64: f64 = 1e-5;
const EPS32: f32 = 1e-5;

#[inline]
fn clip64(v: f64) -> f64 {
    v.clamp(EPS64, 1.0 - EPS64)
}

#[inline]
fn clip32(v: f32) -> f32 {
    v.clamp(EPS32, 1.0 - EPS32)
}

struct StreamRsvdConfig {
    genotype_path: String,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
    delimiter: Option<String>,
    mmap_window_mb: usize,
    memory_mb: usize,
}

enum StreamSnpIter {
    Bed(BedSnpIter),
    Vcf(VcfSnpIter),
    Hmp(HmpSnpIter),
    Txt(TxtSnpIter),
}

impl StreamSnpIter {
    fn n_samples(&self) -> usize {
        match self {
            StreamSnpIter::Bed(it) => it.n_samples(),
            StreamSnpIter::Vcf(it) => it.n_samples(),
            StreamSnpIter::Hmp(it) => it.n_samples(),
            StreamSnpIter::Txt(it) => it.n_samples(),
        }
    }

    fn next_row_raw(&mut self) -> Option<(Vec<f32>, String, String)> {
        match self {
            StreamSnpIter::Bed(it) => it
                .next_snp_raw()
                .map(|(row, site)| (row, site.ref_allele, site.alt_allele)),
            StreamSnpIter::Vcf(it) => it
                .next_snp_raw()
                .map(|(row, site)| (row, site.ref_allele, site.alt_allele)),
            StreamSnpIter::Hmp(it) => it
                .next_snp_raw()
                .map(|(row, site)| (row, site.ref_allele, site.alt_allele)),
            StreamSnpIter::Txt(it) => it
                .next_snp()
                .map(|(row, site)| (row, site.ref_allele, site.alt_allele)),
        }
    }
}

fn is_dna_base(c: char) -> bool {
    matches!(c, 'A' | 'C' | 'G' | 'T')
}

fn is_simple_snp_alleles(ref_allele: &str, alt_allele: &str) -> bool {
    let r = ref_allele.trim().to_ascii_uppercase();
    let a = alt_allele.trim().to_ascii_uppercase();
    if r.len() != 1 || a.len() != 1 {
        return false;
    }
    let rc = r.chars().next().unwrap_or('N');
    let ac = a.chars().next().unwrap_or('N');
    is_dna_base(rc) && is_dna_base(ac)
}

fn normalize_numeric_row(row: &mut [f32]) {
    for g in row.iter_mut() {
        if !g.is_finite() || *g < 0.0 {
            *g = -9.0;
        } else if *g >= 3.0 {
            // Keep parity with internal missing encodings (e.g. 3 in u8 pipelines).
            *g = -9.0;
        }
    }
}

fn detect_bed_prefix(path_or_prefix: &str) -> Option<String> {
    let p = Path::new(path_or_prefix);
    let ext = p
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let prefix: PathBuf = if matches!(ext.as_str(), "bed" | "bim" | "fam") {
        p.with_extension("")
    } else {
        p.to_path_buf()
    };
    let prefix_s = prefix.to_string_lossy().to_string();
    let bed = PathBuf::from(format!("{prefix_s}.bed"));
    let bim = PathBuf::from(format!("{prefix_s}.bim"));
    let fam = PathBuf::from(format!("{prefix_s}.fam"));
    if bed.exists() && bim.exists() && fam.exists() {
        Some(prefix_s)
    } else {
        None
    }
}

enum PackedBedStorage {
    Mmap(Mmap),
    Owned(Vec<u8>),
}

impl PackedBedStorage {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            PackedBedStorage::Mmap(mmap) => mmap,
            PackedBedStorage::Owned(buf) => buf,
        }
    }

    #[inline]
    fn is_mmap(&self) -> bool {
        matches!(self, PackedBedStorage::Mmap(_))
    }

    #[inline]
    fn label(&self) -> &'static str {
        match self {
            PackedBedStorage::Mmap(_) => "mmap",
            PackedBedStorage::Owned(_) => "owned",
        }
    }
}

struct PackedBedRsvd {
    storage: PackedBedStorage,
    payload_offset: usize,
    compact_rows: bool,
    bytes_per_snp: usize,
    n_samples: usize,
    n_snps: usize,
    n_total_sites: usize,
    active_row_idx: Vec<usize>,
    row_freq: Vec<f32>,
    row_flip: Vec<bool>,
    total_variance: f64,
    block_target_mb: Option<usize>,
}

const PACKED_BED_CACHE_MAGIC: &[u8; 8] = b"JXRSVDC1";
const PACKED_BED_CACHE_VERSION: u32 = 2;

#[inline]
fn packed_bed_cache_enabled() -> bool {
    match std::env::var("JANUSX_RSVD_BED_CACHE") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => true,
    }
}

fn packed_bed_cache_path(prefix: &str, cfg: &StreamRsvdConfig) -> PathBuf {
    PathBuf::from(format!(
        "{prefix}.rsvd_bedcache.maf{:08x}.miss{:08x}.snps{}.bin",
        cfg.maf.to_bits(),
        cfg.missing_rate.to_bits(),
        if cfg.snps_only { 1 } else { 0 }
    ))
}

fn packed_bed_compact_cache_path(prefix: &str, cfg: &StreamRsvdConfig) -> PathBuf {
    PathBuf::from(format!(
        "{prefix}.rsvd_bedpack.maf{:08x}.miss{:08x}.snps{}.bin",
        cfg.maf.to_bits(),
        cfg.missing_rate.to_bits(),
        if cfg.snps_only { 1 } else { 0 }
    ))
}

#[inline]
fn packed_bed_compact_cache_enabled() -> bool {
    match std::env::var("JANUSX_ADMX_BED_COMPACT_CACHE") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => true,
    }
}

#[inline]
fn packed_bed_owned_storage_enabled() -> bool {
    match std::env::var("JANUSX_ADMX_BED_OWNED_PACKED") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => true,
    }
}

#[inline]
fn file_signature(path: &Path) -> Option<(u64, u64)> {
    let meta = path.metadata().ok()?;
    let size = meta.len();
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos()
        .min(u64::MAX as u128) as u64;
    Some((size, mtime))
}

fn scan_bim_site_count_and_simple_mask(
    prefix: &str,
    snps_only: bool,
) -> Result<(usize, Option<Vec<bool>>), String> {
    let bim_path = format!("{prefix}.bim");
    let file = File::open(&bim_path).map_err(|e| format!("{bim_path}: {e}"))?;
    let reader = BufReader::new(file);
    let mut n_sites = 0usize;
    let mut simple_mask = if snps_only { Some(Vec::new()) } else { None };

    for (line_no, line) in reader.lines().enumerate() {
        let l = line.map_err(|e| format!("{bim_path}:{}: {e}", line_no + 1))?;
        let cols: Vec<&str> = l.split_whitespace().collect();
        if cols.len() < 6 {
            return Err(format!(
                "Malformed BIM line at {bim_path}:{}: {l}",
                line_no + 1
            ));
        }
        if let Some(mask) = simple_mask.as_mut() {
            mask.push(is_simple_snp_alleles(cols[4], cols[5]));
        }
        n_sites += 1;
    }
    Ok((n_sites, simple_mask))
}

#[inline]
fn push_u32_le(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_u64_le(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_f32_le(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn read_fixed_from_bytes<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Option<[u8; N]> {
    let end = cursor.checked_add(N)?;
    let slice = bytes.get(*cursor..end)?;
    let mut out = [0u8; N];
    out.copy_from_slice(slice);
    *cursor = end;
    Some(out)
}

#[inline]
fn read_u32_le_from_bytes(bytes: &[u8], cursor: &mut usize) -> Option<u32> {
    Some(u32::from_le_bytes(read_fixed_from_bytes::<4>(
        bytes, cursor,
    )?))
}

#[inline]
fn read_u64_le_from_bytes(bytes: &[u8], cursor: &mut usize) -> Option<u64> {
    Some(u64::from_le_bytes(read_fixed_from_bytes::<8>(
        bytes, cursor,
    )?))
}

#[inline]
fn read_f64_le_from_bytes(bytes: &[u8], cursor: &mut usize) -> Option<f64> {
    Some(f64::from_le_bytes(read_fixed_from_bytes::<8>(
        bytes, cursor,
    )?))
}

#[inline]
fn copy_le_f32_bytes(bytes: &[u8], out: &mut [f32]) -> bool {
    if bytes.len() != out.len().saturating_mul(std::mem::size_of::<f32>()) {
        return false;
    }
    if cfg!(target_endian = "little") && (bytes.as_ptr() as usize).is_multiple_of(align_of::<f32>())
    {
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
        }
        return true;
    }
    for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    true
}

#[inline]
fn decode_le_u32_indices(bytes: &[u8], n_total_sites: usize) -> Option<Vec<usize>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    if cfg!(target_endian = "little") && (bytes.as_ptr() as usize).is_multiple_of(align_of::<u32>())
    {
        let src = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, n) };
        for &idx_u32 in src {
            let idx = idx_u32 as usize;
            if idx >= n_total_sites {
                return None;
            }
            out.push(idx);
        }
        return Some(out);
    }
    for chunk in bytes.chunks_exact(4) {
        let idx = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize;
        if idx >= n_total_sites {
            return None;
        }
        out.push(idx);
    }
    Some(out)
}

fn try_load_packed_bed_cache(
    cache_path: &Path,
    bed_path: &Path,
    bim_path: &Path,
    fam_path: &Path,
    n_samples: usize,
    n_total_sites: usize,
    bytes_per_snp: usize,
) -> Option<(Vec<usize>, Vec<f32>, Vec<bool>, f64)> {
    if !packed_bed_cache_enabled() || !cache_path.exists() {
        return None;
    }
    let (bed_size, bed_mtime) = file_signature(bed_path)?;
    let (bim_size, bim_mtime) = file_signature(bim_path)?;
    let (fam_size, fam_mtime) = file_signature(fam_path)?;
    let mmap = mmap_readonly_file(cache_path).ok()?;
    let bytes = mmap.as_ref();
    let mut cursor = 0usize;
    let magic = read_fixed_from_bytes::<8>(bytes, &mut cursor)?;
    if &magic != PACKED_BED_CACHE_MAGIC {
        return None;
    }
    if read_u32_le_from_bytes(bytes, &mut cursor)? != PACKED_BED_CACHE_VERSION {
        return None;
    }
    if read_u64_le_from_bytes(bytes, &mut cursor)? != bed_size
        || read_u64_le_from_bytes(bytes, &mut cursor)? != bed_mtime
        || read_u64_le_from_bytes(bytes, &mut cursor)? != bim_size
        || read_u64_le_from_bytes(bytes, &mut cursor)? != bim_mtime
        || read_u64_le_from_bytes(bytes, &mut cursor)? != fam_size
        || read_u64_le_from_bytes(bytes, &mut cursor)? != fam_mtime
        || read_u64_le_from_bytes(bytes, &mut cursor)? != n_samples as u64
        || read_u64_le_from_bytes(bytes, &mut cursor)? != n_total_sites as u64
        || read_u64_le_from_bytes(bytes, &mut cursor)? != bytes_per_snp as u64
    {
        return None;
    }
    let n_active = usize::try_from(read_u64_le_from_bytes(bytes, &mut cursor)?).ok()?;
    if n_active > n_total_sites {
        return None;
    }
    let total_variance = read_f64_le_from_bytes(bytes, &mut cursor)?;
    if !total_variance.is_finite() || total_variance <= 0.0 {
        return None;
    }
    let mut row_freq = vec![0.0_f32; n_active];
    let row_freq_bytes_len = n_active.checked_mul(4)?;
    let row_freq_end = cursor.checked_add(row_freq_bytes_len)?;
    let row_freq_bytes = bytes.get(cursor..row_freq_end)?;
    if !copy_le_f32_bytes(row_freq_bytes, &mut row_freq) {
        return None;
    }
    cursor = row_freq_end;
    let flip_end = cursor.checked_add(n_active)?;
    let flip_buf = bytes.get(cursor..flip_end)?;
    let row_flip: Vec<bool> = flip_buf.iter().map(|&v| v != 0).collect();
    cursor = flip_end;
    let idx_bytes_len = n_active.checked_mul(4)?;
    let idx_end = cursor.checked_add(idx_bytes_len)?;
    let idx_bytes = bytes.get(cursor..idx_end)?;
    let active_row_idx = decode_le_u32_indices(idx_bytes, n_total_sites)?;
    cursor = idx_end;
    if cursor != bytes.len() {
        return None;
    }
    Some((active_row_idx, row_freq, row_flip, total_variance))
}

fn try_store_packed_bed_cache(
    cache_path: &Path,
    bed_path: &Path,
    bim_path: &Path,
    fam_path: &Path,
    bytes_per_snp: usize,
    n_total_sites: usize,
    data: &PackedBedRsvd,
) -> Result<(), String> {
    if !packed_bed_cache_enabled() {
        return Ok(());
    }
    let (bed_size, bed_mtime) =
        file_signature(bed_path).ok_or_else(|| "missing BED signature".to_string())?;
    let (bim_size, bim_mtime) =
        file_signature(bim_path).ok_or_else(|| "missing BIM signature".to_string())?;
    let (fam_size, fam_mtime) =
        file_signature(fam_path).ok_or_else(|| "missing FAM signature".to_string())?;
    if data
        .active_row_idx
        .iter()
        .any(|&idx| idx > u32::MAX as usize)
    {
        return Err("active_row_idx exceeds u32 range for RSVD cache".to_string());
    }
    let n_active = data.active_row_idx.len();
    let mut buf = Vec::with_capacity(8 + 4 + 10 * 8 + 8 + n_active * (4 + 1 + 4));
    buf.extend_from_slice(PACKED_BED_CACHE_MAGIC);
    push_u32_le(&mut buf, PACKED_BED_CACHE_VERSION);
    push_u64_le(&mut buf, bed_size);
    push_u64_le(&mut buf, bed_mtime);
    push_u64_le(&mut buf, bim_size);
    push_u64_le(&mut buf, bim_mtime);
    push_u64_le(&mut buf, fam_size);
    push_u64_le(&mut buf, fam_mtime);
    push_u64_le(&mut buf, data.n_samples as u64);
    push_u64_le(&mut buf, n_total_sites as u64);
    push_u64_le(&mut buf, bytes_per_snp as u64);
    push_u64_le(&mut buf, n_active as u64);
    buf.extend_from_slice(&data.total_variance.to_le_bytes());
    for &freq in data.row_freq.iter() {
        push_f32_le(&mut buf, freq);
    }
    for &flip in data.row_flip.iter() {
        buf.push(if flip { 1 } else { 0 });
    }
    for &idx in data.active_row_idx.iter() {
        push_u32_le(&mut buf, idx as u32);
    }
    let tmp_path = cache_path.with_extension("tmp");
    fs::write(&tmp_path, &buf).map_err(|e| e.to_string())?;
    fs::rename(&tmp_path, cache_path).map_err(|e| e.to_string())?;
    Ok(())
}

fn try_open_packed_bed_compact_cache_storage(
    compact_path: &Path,
    n_active: usize,
    bytes_per_snp: usize,
) -> Option<PackedBedStorage> {
    if !packed_bed_compact_cache_enabled() || !compact_path.exists() {
        return None;
    }
    let expected_size = n_active.checked_mul(bytes_per_snp)?;
    if expected_size == 0 {
        return None;
    }
    let file = File::open(compact_path).ok()?;
    let file_size = usize::try_from(file.metadata().ok()?.len()).ok()?;
    if file_size != expected_size {
        return None;
    }
    if packed_bed_owned_storage_enabled() {
        let buf = load_file_owned_exact(compact_path, expected_size).ok()?;
        return Some(PackedBedStorage::Owned(buf));
    }
    mmap_readonly_file(compact_path)
        .ok()
        .map(PackedBedStorage::Mmap)
}

fn build_compact_cache_from_active_rows(
    compact_path: &Path,
    bed_path: &Path,
    n_total_sites: usize,
    n_samples: usize,
    bytes_per_snp: usize,
    active_row_idx: &[usize],
    target_mb: Option<usize>,
) -> Result<(), String> {
    if !packed_bed_compact_cache_enabled() {
        return Err("compact BED row cache is disabled".to_string());
    }
    let mut bed_reader = BufReader::new(File::open(bed_path).map_err(|e| e.to_string())?);
    let mut header = [0u8; 3];
    bed_reader
        .read_exact(&mut header)
        .map_err(|e| e.to_string())?;
    if header != [0x6C, 0x1B, 0x01] {
        return Err("Only SNP-major BED supported".to_string());
    }

    let tmp_path = compact_path.with_extension("tmp");
    let mut writer = BufWriter::new(File::create(&tmp_path).map_err(|e| e.to_string())?);
    let chunk_rows = packed_backend_stats_chunk_rows(n_total_sites, n_samples, target_mb);
    let mut chunk_buf = Vec::<u8>::new();
    let mut active_ptr = 0usize;
    for chunk_start in (0..n_total_sites).step_by(chunk_rows) {
        check_admx_memory_limit("bed_backend/compact_cache_build")?;
        let chunk_end = (chunk_start + chunk_rows).min(n_total_sites);
        let row_count = chunk_end - chunk_start;
        chunk_buf.resize(row_count * bytes_per_snp, 0);
        bed_reader
            .read_exact(&mut chunk_buf)
            .map_err(|e| e.to_string())?;
        while active_ptr < active_row_idx.len() {
            let snp_idx = active_row_idx[active_ptr];
            if snp_idx < chunk_start {
                return Err("active_row_idx is not sorted".to_string());
            }
            if snp_idx >= chunk_end {
                break;
            }
            let local = snp_idx - chunk_start;
            let row_start = local * bytes_per_snp;
            writer
                .write_all(&chunk_buf[row_start..row_start + bytes_per_snp])
                .map_err(|e| e.to_string())?;
            active_ptr += 1;
        }
        emit_rsvd_rss_debug(
            "packed_backend/compact_cache_build",
            &format!(
                "rows={}/{} active_written={}/{}",
                chunk_end,
                n_total_sites,
                active_ptr,
                active_row_idx.len()
            ),
        );
    }
    if active_ptr != active_row_idx.len() {
        return Err(format!(
            "compact cache build wrote {} active rows, expected {}",
            active_ptr,
            active_row_idx.len()
        ));
    }
    writer.flush().map_err(|e| e.to_string())?;
    drop(writer);
    fs::rename(&tmp_path, compact_path).map_err(|e| e.to_string())?;
    Ok(())
}

#[inline]
fn env_truthy_local(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => false,
    }
}

#[inline]
fn admx_rsvd_kp(k_eff: usize, max_cols: usize) -> usize {
    let oversample =
        parse_positive_env_usize(&["JANUSX_ADMX_RSVD_OVERSAMPLE", "JX_ADMX_RSVD_OVERSAMPLE"])
            .unwrap_or(6);
    let kp_min =
        parse_positive_env_usize(&["JANUSX_ADMX_RSVD_KP_MIN", "JX_ADMX_RSVD_KP_MIN"]).unwrap_or(12);
    let kp_cap = parse_positive_env_usize(&["JANUSX_ADMX_RSVD_KP_MAX", "JX_ADMX_RSVD_KP_MAX"]);

    let limit = max_cols.max(1);
    let mut kp = k_eff.max(1).saturating_add(oversample).max(kp_min);
    if let Some(cap) = kp_cap {
        kp = kp.min(cap.max(k_eff.max(1)));
    }
    kp.min(limit)
}

#[inline]
fn rsvd_rss_debug_enabled() -> bool {
    env_truthy_local("JX_RSVD_RSS_DEBUG") || env_truthy_local("JX_PCA_RSS_DEBUG")
}

#[inline]
fn rsvd_time_debug_enabled() -> bool {
    env_truthy_local("JX_RSVD_TIME_STAGE")
        || env_truthy_local("JX_RSVD_TIME_DEBUG")
        || env_truthy_local("JX_PCA_TIME_STAGE")
}

#[inline]
fn format_debug_bytes_local(bytes: u64) -> String {
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

fn emit_rsvd_rss_debug(stage: &str, detail: &str) {
    if !rsvd_rss_debug_enabled() {
        return;
    }
    match process_memory_usage() {
        Some(usage) => {
            let rss = usage
                .rss_bytes
                .map(format_debug_bytes_local)
                .unwrap_or_else(|| "NA".to_string());
            let footprint = usage
                .footprint_bytes
                .map(format_debug_bytes_local)
                .unwrap_or_else(|| "NA".to_string());
            eprintln!(
                "[RSVD-DEBUG] {stage} mem={} metric={} rss={} footprint={} {detail}",
                format_debug_bytes_local(usage.current_bytes),
                usage.metric,
                rss,
                footprint,
            );
        }
        None => {
            eprintln!(
                "[RSVD-DEBUG] {stage} mem=NA metric=unavailable rss=NA footprint=NA {detail}"
            );
        }
    }
    let _ = std::io::stderr().flush();
}

fn emit_rsvd_time_debug(stage: &str, detail: &str) {
    if !rsvd_time_debug_enabled() {
        return;
    }
    println!("[RSVD-TIME] {stage} {detail}");
    let _ = std::io::stdout().flush();
}

#[inline]
fn fmt_stage_secs(v: f64) -> String {
    format!("{v:.3}s")
}

#[inline]
fn format_f32_vec_bytes(len: usize) -> String {
    format_debug_bytes_local((len.saturating_mul(std::mem::size_of::<f32>())) as u64)
}

#[inline]
fn packed_row_nonmissing_alt_sumsq_full(row: &[u8], n_samples: usize) -> (usize, f64, f64) {
    let byte_lut = packed_byte_lut();
    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;
    let mut non_missing = 0usize;
    let mut alt_sum = 0.0_f64;
    let mut sq_sum = 0.0_f64;
    for &b in row.iter().take(full_bytes) {
        let idx = b as usize;
        non_missing += byte_lut.nonmiss[idx] as usize;
        alt_sum += byte_lut.alt_sum[idx] as f64;
        sq_sum += byte_lut.sq_sum[idx] as f64;
    }
    if rem > 0 {
        let codes = &byte_lut.code4[row[full_bytes] as usize];
        for &code in codes.iter().take(rem) {
            match code {
                0b00 => {
                    non_missing += 1;
                }
                0b10 => {
                    non_missing += 1;
                    alt_sum += 1.0;
                    sq_sum += 1.0;
                }
                0b11 => {
                    non_missing += 1;
                    alt_sum += 2.0;
                    sq_sum += 4.0;
                }
                _ => {}
            }
        }
    }
    (non_missing, alt_sum, sq_sum)
}

#[inline]
fn stream_rsvd_target_mb(cfg: &StreamRsvdConfig) -> Option<usize> {
    if cfg.memory_mb > 0 {
        Some(cfg.memory_mb)
    } else {
        None
    }
}

#[inline]
fn stream_rsvd_block_rows(cfg: &StreamRsvdConfig, n_cols: usize, max_rows: usize) -> usize {
    rsvd_block_rows_with_target_mb(n_cols, max_rows, stream_rsvd_target_mb(cfg))
}

#[inline]
fn packed_backend_block_rows(backend: &PackedBedRsvd, max_rows: usize) -> usize {
    rsvd_block_rows_with_target_mb(backend.n_samples, max_rows, backend.block_target_mb)
}

#[inline]
fn packed_backend_stats_chunk_rows(
    n_rows: usize,
    n_samples: usize,
    target_mb: Option<usize>,
) -> usize {
    let rows = rsvd_block_rows_with_target_mb(n_samples, n_rows, target_mb)
        .saturating_mul(16)
        .clamp(1024, 16384);
    rows.min(n_rows.max(1))
}

#[inline]
fn madvise_mmap_bed_rows(
    bytes: &[u8],
    payload_offset: usize,
    bytes_per_snp: usize,
    first_row: usize,
    rows: usize,
) {
    if rows == 0 || bytes_per_snp == 0 {
        return;
    }
    let Some(start) = first_row
        .checked_mul(bytes_per_snp)
        .and_then(|v| payload_offset.checked_add(v))
    else {
        return;
    };
    let Some(end) = first_row
        .checked_add(rows)
        .and_then(|r| r.checked_mul(bytes_per_snp))
        .and_then(|v| payload_offset.checked_add(v))
    else {
        return;
    };
    if end <= start || end > bytes.len() {
        return;
    }
    let ptr = unsafe { bytes.as_ptr().add(start) };
    admx_madvise_dontneed_bytes(ptr, end - start);
}

#[inline]
fn madvise_backend_active_rows(backend: &PackedBedRsvd, active_start: usize, rows: usize) {
    if rows == 0 || backend.bytes_per_snp == 0 || !backend.storage.is_mmap() {
        return;
    }
    if backend.compact_rows {
        madvise_mmap_bed_rows(
            backend.storage.as_slice(),
            backend.payload_offset,
            backend.bytes_per_snp,
            active_start,
            rows,
        );
        return;
    }
    let Some(active_end) = active_start.checked_add(rows) else {
        return;
    };
    let Some(slice) = backend.active_row_idx.get(active_start..active_end) else {
        return;
    };
    let Some((&first, rest)) = slice.split_first() else {
        return;
    };
    let mut min_row = first;
    let mut max_row = first;
    for &idx in rest {
        min_row = min_row.min(idx);
        max_row = max_row.max(idx);
    }
    madvise_mmap_bed_rows(
        backend.storage.as_slice(),
        backend.payload_offset,
        backend.bytes_per_snp,
        min_row,
        max_row.saturating_sub(min_row).saturating_add(1),
    );
}

#[inline]
fn packed_backend_row_indices(backend: &PackedBedRsvd) -> Option<&[usize]> {
    if backend.compact_rows {
        None
    } else {
        Some(&backend.active_row_idx)
    }
}

fn packed_bed_fast_enabled() -> bool {
    match std::env::var("JANUSX_RSVD_BED_PACKED") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => true,
    }
}

fn try_build_packed_bed_backend(cfg: &StreamRsvdConfig) -> Result<Option<PackedBedRsvd>, String> {
    if !packed_bed_fast_enabled() {
        return Ok(None);
    }
    let Some(prefix) = detect_bed_prefix(&cfg.genotype_path) else {
        return Ok(None);
    };
    let t_backend = Instant::now();
    emit_rsvd_rss_debug("packed_backend/open_start", &format!("prefix={prefix}"));
    emit_rsvd_time_debug("packed_backend/start", &format!("prefix={prefix}"));
    let bed_path = PathBuf::from(format!("{prefix}.bed"));
    let bim_path = PathBuf::from(format!("{prefix}.bim"));
    let fam_path = PathBuf::from(format!("{prefix}.fam"));
    let cache_path = packed_bed_cache_path(&prefix, cfg);
    let compact_path = packed_bed_compact_cache_path(&prefix, cfg);

    let t_stage = Instant::now();
    let samples = read_fam(&prefix)?;
    let fam_s = t_stage.elapsed().as_secs_f64();
    let n_samples = samples.len();
    if n_samples == 0 {
        return Err("no samples found in PLINK input".to_string());
    }
    check_admx_memory_limit("bed_backend/after_fam")?;
    emit_rsvd_rss_debug(
        "packed_backend/fam_ready",
        &format!("prefix={} n_samples={}", prefix, n_samples),
    );
    emit_rsvd_time_debug(
        "packed_backend/fam",
        &format!("elapsed={} n_samples={}", fmt_stage_secs(fam_s), n_samples),
    );

    let t_stage = Instant::now();
    let bed_info = read_plink_bed_header_info(&bed_path, n_samples)?;
    let bed_header_s = t_stage.elapsed().as_secs_f64();
    let bytes_per_snp = bed_info.bytes_per_snp;
    let n_snps_total = bed_info.n_snps_total;
    let expected_payload = bed_info.payload_size;
    check_admx_memory_limit("bed_backend/after_bed_header")?;
    emit_rsvd_rss_debug(
        "packed_backend/bed_header",
        &format!(
            "prefix={} n_samples={} n_total_sites={} bytes_per_snp={} mapped_payload={}",
            prefix,
            n_samples,
            n_snps_total,
            bytes_per_snp,
            format_debug_bytes_local(expected_payload as u64),
        ),
    );
    emit_rsvd_time_debug(
        "packed_backend/bed_header",
        &format!(
            "elapsed={} n_samples={} n_total_sites={} bytes_per_snp={} file_size={} payload={}",
            fmt_stage_secs(bed_header_s),
            n_samples,
            n_snps_total,
            bytes_per_snp,
            format_debug_bytes_local(bed_info.file_size as u64),
            format_debug_bytes_local(expected_payload as u64),
        ),
    );

    let t_stage = Instant::now();
    let cached_meta = try_load_packed_bed_cache(
        &cache_path,
        &bed_path,
        &bim_path,
        &fam_path,
        n_samples,
        n_snps_total,
        bytes_per_snp,
    );
    let meta_cache_s = t_stage.elapsed().as_secs_f64();
    if let Some((active_row_idx, row_freq, row_flip, total_variance)) = cached_meta {
        emit_rsvd_rss_debug(
            "packed_backend/cache_hit",
            &format!(
                "prefix={} cache={} active_rows={}",
                prefix,
                cache_path.display(),
                row_freq.len(),
            ),
        );
        emit_rsvd_time_debug(
            "packed_backend/meta_cache_hit",
            &format!(
                "elapsed={} cache={} active_rows={}",
                fmt_stage_secs(meta_cache_s),
                cache_path.display(),
                row_freq.len(),
            ),
        );
        if row_freq.is_empty() {
            return Err("no SNPs remain after BED filters".to_string());
        }
        let t_stage = Instant::now();
        if let Some(storage) =
            try_open_packed_bed_compact_cache_storage(&compact_path, row_freq.len(), bytes_per_snp)
        {
            let compact_open_s = t_stage.elapsed().as_secs_f64();
            emit_rsvd_rss_debug(
                "packed_backend/compact_cache_hit",
                &format!(
                    "prefix={} cache={} active_rows={} packed_payload={} storage={}",
                    prefix,
                    compact_path.display(),
                    row_freq.len(),
                    format_debug_bytes_local((row_freq.len().saturating_mul(bytes_per_snp)) as u64),
                    storage.label(),
                ),
            );
            emit_rsvd_time_debug(
                "packed_backend/compact_cache_hit",
                &format!(
                    "elapsed={} cache={} active_rows={} storage={}",
                    fmt_stage_secs(compact_open_s),
                    compact_path.display(),
                    row_freq.len(),
                    storage.label(),
                ),
            );
            emit_rsvd_time_debug(
                "packed_backend/summary",
                &format!(
                    "elapsed={} mode=meta_cache+compact_cache active_rows={} total_sites={}",
                    fmt_stage_secs(t_backend.elapsed().as_secs_f64()),
                    row_freq.len(),
                    n_snps_total,
                ),
            );
            return Ok(Some(PackedBedRsvd {
                storage,
                payload_offset: 0,
                compact_rows: true,
                bytes_per_snp,
                n_samples,
                n_snps: row_freq.len(),
                n_total_sites: n_snps_total,
                active_row_idx,
                row_freq,
                row_flip,
                total_variance,
                block_target_mb: stream_rsvd_target_mb(cfg),
            }));
        }
        let compact_open_s = t_stage.elapsed().as_secs_f64();
        emit_rsvd_rss_debug(
            "packed_backend/compact_cache_miss",
            &format!("prefix={} cache={}", prefix, compact_path.display()),
        );
        emit_rsvd_time_debug(
            "packed_backend/compact_cache_miss",
            &format!(
                "elapsed={} cache={} active_rows={}",
                fmt_stage_secs(compact_open_s),
                compact_path.display(),
                row_freq.len(),
            ),
        );
        let t_stage = Instant::now();
        match build_compact_cache_from_active_rows(
            &compact_path,
            &bed_path,
            n_snps_total,
            n_samples,
            bytes_per_snp,
            &active_row_idx,
            stream_rsvd_target_mb(cfg),
        ) {
            Ok(()) => {
                let compact_build_s = t_stage.elapsed().as_secs_f64();
                emit_rsvd_time_debug(
                    "packed_backend/compact_cache_build",
                    &format!(
                        "elapsed={} cache={} active_rows={}",
                        fmt_stage_secs(compact_build_s),
                        compact_path.display(),
                        row_freq.len(),
                    ),
                );
                let t_stage = Instant::now();
                if let Some(storage) = try_open_packed_bed_compact_cache_storage(
                    &compact_path,
                    row_freq.len(),
                    bytes_per_snp,
                ) {
                    let compact_ready_s = t_stage.elapsed().as_secs_f64();
                    emit_rsvd_rss_debug(
                        "packed_backend/compact_cache_built",
                        &format!(
                            "prefix={} cache={} active_rows={} storage={}",
                            prefix,
                            compact_path.display(),
                            row_freq.len(),
                            storage.label(),
                        ),
                    );
                    emit_rsvd_time_debug(
                        "packed_backend/compact_cache_ready",
                        &format!(
                            "elapsed={} cache={} active_rows={} storage={}",
                            fmt_stage_secs(compact_ready_s),
                            compact_path.display(),
                            row_freq.len(),
                            storage.label(),
                        ),
                    );
                    emit_rsvd_time_debug(
                        "packed_backend/summary",
                        &format!(
                            "elapsed={} mode=meta_cache+compact_rebuild active_rows={} total_sites={}",
                            fmt_stage_secs(t_backend.elapsed().as_secs_f64()),
                            row_freq.len(),
                            n_snps_total,
                        ),
                    );
                    return Ok(Some(PackedBedRsvd {
                        storage,
                        payload_offset: 0,
                        compact_rows: true,
                        bytes_per_snp,
                        n_samples,
                        n_snps: row_freq.len(),
                        n_total_sites: n_snps_total,
                        active_row_idx,
                        row_freq,
                        row_flip,
                        total_variance,
                        block_target_mb: stream_rsvd_target_mb(cfg),
                    }));
                }
            }
            Err(e) => {
                let compact_build_s = t_stage.elapsed().as_secs_f64();
                emit_rsvd_rss_debug(
                    "packed_backend/compact_cache_build_failed",
                    &format!("prefix={} error={}", prefix, e),
                );
                emit_rsvd_time_debug(
                    "packed_backend/compact_cache_build_failed",
                    &format!(
                        "elapsed={} cache={} error={}",
                        fmt_stage_secs(compact_build_s),
                        compact_path.display(),
                        e,
                    ),
                );
            }
        }
        let t_stage = Instant::now();
        let mmap = mmap_readonly_file(&bed_path)?;
        let mmap_s = t_stage.elapsed().as_secs_f64();
        emit_rsvd_rss_debug(
            "packed_backend/original_mmap_fallback",
            &format!(
                "prefix={} active_rows={} mapped_payload={}",
                prefix,
                row_freq.len(),
                format_debug_bytes_local(expected_payload as u64),
            ),
        );
        emit_rsvd_time_debug(
            "packed_backend/original_mmap_fallback",
            &format!(
                "elapsed={} active_rows={} mapped_payload={}",
                fmt_stage_secs(mmap_s),
                row_freq.len(),
                format_debug_bytes_local(expected_payload as u64),
            ),
        );
        emit_rsvd_time_debug(
            "packed_backend/summary",
            &format!(
                "elapsed={} mode=meta_cache+original_mmap active_rows={} total_sites={}",
                fmt_stage_secs(t_backend.elapsed().as_secs_f64()),
                row_freq.len(),
                n_snps_total,
            ),
        );
        return Ok(Some(PackedBedRsvd {
            storage: PackedBedStorage::Mmap(mmap),
            payload_offset: 3,
            compact_rows: false,
            bytes_per_snp,
            n_samples,
            n_snps: row_freq.len(),
            n_total_sites: n_snps_total,
            active_row_idx,
            row_freq,
            row_flip,
            total_variance,
            block_target_mb: stream_rsvd_target_mb(cfg),
        }));
    }
    emit_rsvd_time_debug(
        "packed_backend/meta_cache_miss",
        &format!(
            "elapsed={} cache={} total_sites={}",
            fmt_stage_secs(meta_cache_s),
            cache_path.display(),
            n_snps_total,
        ),
    );

    let t_stage = Instant::now();
    let (bim_sites, simple_snp_mask) = scan_bim_site_count_and_simple_mask(&prefix, cfg.snps_only)?;
    let bim_scan_s = t_stage.elapsed().as_secs_f64();
    if bim_sites != n_snps_total {
        return Err(format!(
            "BED/BIM site count mismatch: bed={n_snps_total}, bim={bim_sites}"
        ));
    }
    check_admx_memory_limit("bed_backend/after_bim_scan")?;
    emit_rsvd_rss_debug(
        "packed_backend/cache_miss_bim_ready",
        &format!(
            "prefix={} n_total_sites={} snps_only={}",
            prefix, n_snps_total, cfg.snps_only
        ),
    );
    emit_rsvd_time_debug(
        "packed_backend/bim_scan",
        &format!(
            "elapsed={} n_total_sites={} snps_only={}",
            fmt_stage_secs(bim_scan_s),
            n_snps_total,
            cfg.snps_only,
        ),
    );

    let mut active_row_idx: Vec<usize> = Vec::with_capacity(n_snps_total);
    let mut row_freq: Vec<f32> = Vec::with_capacity(n_snps_total);
    let mut row_flip: Vec<bool> = Vec::with_capacity(n_snps_total);
    let mut total_ss = 0.0_f64;
    let mut total_var = 0.0_f64;
    let n_samples_f64 = n_samples as f64;
    let chunk_rows =
        packed_backend_stats_chunk_rows(n_snps_total, n_samples, stream_rsvd_target_mb(cfg));
    let t_stage = Instant::now();
    let mut bed_reader = open_plink_bed_reader_after_header(&bed_path)?;
    let compact_tmp_path = compact_path.with_extension("tmp");
    let mut compact_writer = if packed_bed_compact_cache_enabled() {
        match File::create(&compact_tmp_path) {
            Ok(file) => Some(BufWriter::new(file)),
            Err(e) => {
                emit_rsvd_rss_debug(
                    "packed_backend/compact_cache_create_failed",
                    &format!(
                        "prefix={} cache={} error={}",
                        prefix,
                        compact_path.display(),
                        e
                    ),
                );
                None
            }
        }
    } else {
        None
    };
    let scan_setup_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_time_debug(
        "packed_backend/scan_setup",
        &format!(
            "elapsed={} chunk_rows={} compact_cache_enabled={}",
            fmt_stage_secs(scan_setup_s),
            chunk_rows,
            packed_bed_compact_cache_enabled(),
        ),
    );
    let mut chunk_buf = Vec::<u8>::new();
    let t_stage = Instant::now();
    for chunk_start in (0..n_snps_total).step_by(chunk_rows) {
        check_admx_memory_limit("bed_backend/filter_scan")?;
        let chunk_end = (chunk_start + chunk_rows).min(n_snps_total);
        let row_count = chunk_end - chunk_start;
        chunk_buf.resize(row_count * bytes_per_snp, 0);
        bed_reader
            .read_exact(&mut chunk_buf)
            .map_err(|e| e.to_string())?;
        let keep_rows: Vec<Option<(usize, usize, f32, bool, f64, f64)>> = (0..row_count)
            .into_par_iter()
            .map(|local_idx| {
                let snp_idx = chunk_start + local_idx;
                if cfg.snps_only
                    && !simple_snp_mask
                        .as_ref()
                        .and_then(|mask| mask.get(snp_idx))
                        .copied()
                        .unwrap_or(false)
                {
                    return None;
                }
                let row_start = local_idx * bytes_per_snp;
                let row = &chunk_buf[row_start..row_start + bytes_per_snp];
                let (non_missing, alt_sum, sq_sum) =
                    packed_row_nonmissing_alt_sumsq_full(row, n_samples);

                let miss_rate = 1.0_f64 - (non_missing as f64 / n_samples_f64);
                if miss_rate > cfg.missing_rate as f64 {
                    return None;
                }
                if non_missing == 0 {
                    if cfg.maf > 0.0 {
                        return None;
                    }
                    return Some((local_idx, snp_idx, 0.0_f32, false, 0.0_f64, 0.0_f64));
                }

                let p = alt_sum / (2.0 * non_missing as f64);
                let flip = p > 0.5;
                let p_minor = if flip { 1.0 - p } else { p };
                if p_minor < cfg.maf as f64 {
                    return None;
                }
                let row_mean = alt_sum / non_missing as f64;
                let row_ss = (sq_sum - alt_sum * row_mean).max(0.0_f64);
                let row_var = (2.0_f64 * p_minor * (1.0_f64 - p_minor)).max(0.0_f64);
                Some((local_idx, snp_idx, p_minor as f32, flip, row_ss, row_var))
            })
            .collect();
        for item in keep_rows.into_iter().flatten() {
            let (local_idx, snp_idx, freq, flip, row_ss, row_var) = item;
            active_row_idx.push(snp_idx);
            row_freq.push(freq);
            row_flip.push(flip);
            total_ss += row_ss;
            total_var += row_var;
            if let Some(writer) = compact_writer.as_mut() {
                let row_start = local_idx * bytes_per_snp;
                writer
                    .write_all(&chunk_buf[row_start..row_start + bytes_per_snp])
                    .map_err(|e| e.to_string())?;
            }
        }
        if rsvd_rss_debug_enabled()
            && (((chunk_start / chunk_rows) % 64 == 0) || chunk_end == n_snps_total)
        {
            emit_rsvd_rss_debug(
                "packed_backend/filter_scan",
                &format!(
                    "rows={}/{} active_rows={}",
                    chunk_end,
                    n_snps_total,
                    row_freq.len()
                ),
            );
        }
    }
    let filter_scan_s = t_stage.elapsed().as_secs_f64();

    let n_snps = row_freq.len();
    if n_snps == 0 {
        return Err("no SNPs remain after BED filters".to_string());
    }
    if !(total_var.is_finite() && total_var > 0.0) {
        return Err("invalid RSVD packed BED total variance denominator".to_string());
    }
    let total_variance = total_ss / total_var;
    if !(total_variance.is_finite() && total_variance > 0.0) {
        return Err("invalid RSVD packed BED trace(K)".to_string());
    }
    emit_rsvd_rss_debug(
        "packed_backend/filter_done",
        &format!(
            "prefix={} active_rows={} dropped={} row_freq={} row_flip={} traceK={:.6e}",
            prefix,
            n_snps,
            n_snps_total.saturating_sub(n_snps),
            format_debug_bytes_local((row_freq.len() * std::mem::size_of::<f32>()) as u64),
            format_debug_bytes_local((row_flip.len() * std::mem::size_of::<bool>()) as u64),
            total_variance,
        ),
    );
    emit_rsvd_time_debug(
        "packed_backend/filter_scan",
        &format!(
            "elapsed={} active_rows={} dropped={} traceK={:.6e} chunk_rows={}",
            fmt_stage_secs(filter_scan_s),
            n_snps,
            n_snps_total.saturating_sub(n_snps),
            total_variance,
            chunk_rows,
        ),
    );
    let compact_ready = if let Some(mut writer) = compact_writer {
        writer.flush().map_err(|e| e.to_string())?;
        drop(writer);
        fs::rename(&compact_tmp_path, &compact_path).map_err(|e| e.to_string())?;
        true
    } else {
        false
    };
    let storage;
    let payload_offset;
    let compact_rows;
    let expected_compact_payload = n_snps
        .checked_mul(bytes_per_snp)
        .ok_or_else(|| "compact BED payload size overflow".to_string())?;
    let t_stage = Instant::now();
    if compact_ready {
        storage = try_open_packed_bed_compact_cache_storage(&compact_path, n_snps, bytes_per_snp)
            .ok_or_else(|| "failed to open compact BED row cache after build".to_string())?;
        payload_offset = 0;
        compact_rows = true;
        let storage_ready_s = t_stage.elapsed().as_secs_f64();
        emit_rsvd_rss_debug(
            "packed_backend/compact_cache_ready",
            &format!(
                "prefix={} cache={} packed_payload={} storage={}",
                prefix,
                compact_path.display(),
                format_debug_bytes_local(expected_compact_payload as u64),
                storage.label(),
            ),
        );
        emit_rsvd_time_debug(
            "packed_backend/compact_cache_ready",
            &format!(
                "elapsed={} cache={} packed_payload={} storage={}",
                fmt_stage_secs(storage_ready_s),
                compact_path.display(),
                format_debug_bytes_local(expected_compact_payload as u64),
                storage.label(),
            ),
        );
    } else {
        let mmap = mmap_readonly_file(&bed_path)?;
        storage = PackedBedStorage::Mmap(mmap);
        payload_offset = 3;
        compact_rows = false;
        let storage_ready_s = t_stage.elapsed().as_secs_f64();
        emit_rsvd_rss_debug(
            "packed_backend/original_mmap_ready",
            &format!(
                "prefix={} mapped_payload={}",
                prefix,
                format_debug_bytes_local(expected_payload as u64),
            ),
        );
        emit_rsvd_time_debug(
            "packed_backend/original_mmap_ready",
            &format!(
                "elapsed={} mapped_payload={}",
                fmt_stage_secs(storage_ready_s),
                format_debug_bytes_local(expected_payload as u64),
            ),
        );
    }
    let backend = PackedBedRsvd {
        storage,
        payload_offset,
        compact_rows,
        bytes_per_snp,
        n_samples,
        n_snps,
        n_total_sites: n_snps_total,
        active_row_idx,
        row_freq,
        row_flip,
        total_variance,
        block_target_mb: stream_rsvd_target_mb(cfg),
    };
    let t_stage = Instant::now();
    let _ = try_store_packed_bed_cache(
        &cache_path,
        &bed_path,
        &bim_path,
        &fam_path,
        bytes_per_snp,
        n_snps_total,
        &backend,
    );
    let cache_store_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_time_debug(
        "packed_backend/cache_store",
        &format!(
            "elapsed={} cache={} active_rows={}",
            fmt_stage_secs(cache_store_s),
            cache_path.display(),
            backend.n_snps,
        ),
    );
    emit_rsvd_time_debug(
        "packed_backend/summary",
        &format!(
            "elapsed={} mode=build active_rows={} total_sites={} compact_rows={}",
            fmt_stage_secs(t_backend.elapsed().as_secs_f64()),
            backend.n_snps,
            backend.n_total_sites,
            backend.compact_rows,
        ),
    );
    Ok(Some(backend))
}

fn open_packed_bed_backend_for_training(
    genotype_path: &str,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
    memory_mb: usize,
) -> Result<(PackedBedRsvd, String), String> {
    if !(0.0..=0.5).contains(&maf) {
        return Err("maf must be within [0, 0.5]".to_string());
    }
    if !(0.0..=1.0).contains(&missing_rate) {
        return Err("missing_rate must be within [0, 1]".to_string());
    }
    let prefix = detect_bed_prefix(genotype_path)
        .ok_or_else(|| format!("BED prefix not found for genotype input: {genotype_path}"))?;
    let cfg = StreamRsvdConfig {
        genotype_path: prefix.clone(),
        snps_only,
        maf,
        missing_rate,
        delimiter: None,
        mmap_window_mb: 0,
        memory_mb,
    };
    let Some(backend) = try_build_packed_bed_backend(&cfg)? else {
        return Err("BED backend is unavailable or disabled for this input".to_string());
    };
    Ok((backend, prefix))
}

#[pyclass]
pub struct AdmxBedBackend {
    backend: Arc<PackedBedRsvd>,
    prefix: String,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
}

#[pyclass]
pub struct AdmxBedFoldBackend {
    backend: Arc<PackedBedRsvd>,
    prefix: String,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
    fold_id: usize,
    folds: usize,
    cv_seed: u64,
    row_freq_train: Vec<f32>,
    train_observed: usize,
    holdout_observed: usize,
}

#[pyclass]
pub struct AdmxBedTrainingSession {
    backend: Arc<PackedBedRsvd>,
    prefix: String,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
}

struct AdmxBedFitResult {
    p: Vec<f32>,
    q: Vec<f32>,
    ll_final: f64,
    adam_iter: usize,
    init_ll: f64,
    als_iter: usize,
}

fn compute_a_omega_packed(
    backend: &PackedBedRsvd,
    omega: &[f32], // (n, kp)
    kp: usize,
) -> Result<Vec<f32>, String> {
    let sample_idx: Vec<usize> = (0..backend.n_samples).collect();
    let packed_view = PackedRsvdView {
        packed_flat: &backend.storage.as_slice()[backend.payload_offset..],
        bytes_per_snp: backend.bytes_per_snp,
        n_samples: backend.n_samples,
        row_freq: &backend.row_freq,
        row_flip: &backend.row_flip,
        sample_idx: &sample_idx,
        packed_row_indices: packed_backend_row_indices(backend),
        block_target_mb: backend.block_target_mb,
    };
    rsvd_packed_compute_a_omega(packed_view, omega, kp)
}

fn compute_at_random_omega_packed(
    backend: &PackedBedRsvd,
    kp: usize,
    seed: u64,
) -> Result<Vec<f32>, String> {
    let sample_idx: Vec<usize> = (0..backend.n_samples).collect();
    let packed_view = PackedRsvdView {
        packed_flat: &backend.storage.as_slice()[backend.payload_offset..],
        bytes_per_snp: backend.bytes_per_snp,
        n_samples: backend.n_samples,
        row_freq: &backend.row_freq,
        row_flip: &backend.row_flip,
        sample_idx: &sample_idx,
        packed_row_indices: packed_backend_row_indices(backend),
        block_target_mb: backend.block_target_mb,
    };
    rsvd_packed_compute_at_random_omega(packed_view, kp, seed)
}

fn compute_ata_omega_packed(
    backend: &PackedBedRsvd,
    omega: &[f32], // (n, kp)
    kp: usize,
    timing: Option<&mut RsvdKernelTiming>,
) -> Result<Vec<f32>, String> {
    let sample_idx: Vec<usize> = (0..backend.n_samples).collect();
    let packed_view = PackedRsvdView {
        packed_flat: &backend.storage.as_slice()[backend.payload_offset..],
        bytes_per_snp: backend.bytes_per_snp,
        n_samples: backend.n_samples,
        row_freq: &backend.row_freq,
        row_flip: &backend.row_flip,
        sample_idx: &sample_idx,
        packed_row_indices: packed_backend_row_indices(backend),
        block_target_mb: backend.block_target_mb,
    };
    rsvd_packed_compute_ata_omega(packed_view, omega, kp, timing)
}

fn compute_gram_aq_packed(
    backend: &PackedBedRsvd,
    q: &[f32],
    kp: usize,
) -> Result<Vec<f64>, String> {
    let sample_idx: Vec<usize> = (0..backend.n_samples).collect();
    let packed_view = PackedRsvdView {
        packed_flat: &backend.storage.as_slice()[backend.payload_offset..],
        bytes_per_snp: backend.bytes_per_snp,
        n_samples: backend.n_samples,
        row_freq: &backend.row_freq,
        row_flip: &backend.row_flip,
        sample_idx: &sample_idx,
        packed_row_indices: packed_backend_row_indices(backend),
        block_target_mb: backend.block_target_mb,
    };
    rsvd_packed_compute_gram_aq(packed_view, q, kp)
}

#[inline]
fn packed_backend_row_slice(backend: &PackedBedRsvd, active_row: usize) -> &[u8] {
    let snp_idx = if backend.compact_rows {
        active_row
    } else {
        backend.active_row_idx[active_row]
    };
    let start = backend.payload_offset + snp_idx * backend.bytes_per_snp;
    &backend.storage.as_slice()[start..start + backend.bytes_per_snp]
}

#[inline]
fn packed_code_minor_allele_g(code: u8, flip: bool) -> Option<f32> {
    match code {
        0b00 => Some(if flip { 2.0_f32 } else { 0.0_f32 }),
        0b10 => Some(1.0_f32),
        0b11 => Some(if flip { 0.0_f32 } else { 2.0_f32 }),
        _ => None,
    }
}

#[inline]
fn cv_fold_hash64(flat_idx: u64, seed: u64) -> u64 {
    let mix = seed ^ 0x9E3779B97F4A7C15_u64;
    let mut x = flat_idx ^ mix;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9_u64);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB_u64);
    x ^ (x >> 31)
}

#[inline]
fn cv_fold_id(flat_idx: u64, folds: usize, seed: u64) -> usize {
    (cv_fold_hash64(flat_idx, seed) % (folds.max(1) as u64)) as usize
}

#[inline]
fn validate_cv_folds(folds: usize) -> Result<(), String> {
    if folds < 2 {
        return Err(format!("CVerror requires folds >= 2, got {folds}."));
    }
    Ok(())
}

fn open_stream_iter(cfg: &StreamRsvdConfig) -> Result<StreamSnpIter, String> {
    let path = cfg.genotype_path.as_str();
    let path_lower = path.to_ascii_lowercase();
    if let Some(prefix) = detect_bed_prefix(path) {
        let it = if cfg.mmap_window_mb > 0 {
            BedSnpIter::new_with_fill_window(
                &prefix,
                0.0,
                1.0,
                false,
                false,
                0.02,
                cfg.mmap_window_mb,
            )?
        } else {
            BedSnpIter::new_with_fill(&prefix, 0.0, 1.0, false, false, 0.02)?
        };
        return Ok(StreamSnpIter::Bed(it));
    }
    if path_lower.ends_with(".vcf") || path_lower.ends_with(".vcf.gz") {
        return Ok(StreamSnpIter::Vcf(VcfSnpIter::new_with_fill(
            path, 0.0, 1.0, false, false, 0.02,
        )?));
    }
    if path_lower.ends_with(".hmp") || path_lower.ends_with(".hmp.gz") {
        return Ok(StreamSnpIter::Hmp(HmpSnpIter::new_with_fill(
            path, 0.0, 1.0, false, false, 0.02,
        )?));
    }
    Ok(StreamSnpIter::Txt(TxtSnpIter::new(
        path,
        cfg.delimiter.as_deref(),
    )?))
}

fn write_p_site_from_backend_impl(
    prefix: &str,
    active_row_idx: &[usize],
    row_flip: &[bool],
    out_path: &str,
) -> Result<(), String> {
    if active_row_idx.len() != row_flip.len() {
        return Err(format!(
            "active_row_idx / row_flip length mismatch: {} vs {}",
            active_row_idx.len(),
            row_flip.len()
        ));
    }
    let bim_path = format!("{prefix}.bim");
    let file = File::open(&bim_path).map_err(|e| format!("{bim_path}: {e}"))?;
    let reader = BufReader::new(file);
    let mut writer = BufWriter::with_capacity(
        8 * 1024 * 1024,
        File::create(out_path).map_err(|e| format!("{out_path}: {e}"))?,
    );
    let mut want_ptr = 0usize;

    for (line_no, line) in reader.lines().enumerate() {
        if want_ptr >= active_row_idx.len() {
            break;
        }
        if line_no != active_row_idx[want_ptr] {
            continue;
        }
        let l = line.map_err(|e| format!("{bim_path}:{}: {e}", line_no + 1))?;
        let cols: Vec<&str> = l.split_whitespace().collect();
        if cols.len() < 6 {
            return Err(format!(
                "Malformed BIM line at {bim_path}:{}: {l}",
                line_no + 1
            ));
        }
        let (ref_allele, alt_allele) = if row_flip[want_ptr] {
            (cols[5], cols[4])
        } else {
            (cols[4], cols[5])
        };
        writeln!(
            writer,
            "{}\t{}\t{}\t{}",
            cols[0], cols[3], ref_allele, alt_allele
        )
        .map_err(|e| format!("{out_path}: {e}"))?;
        want_ptr += 1;
    }

    if want_ptr != active_row_idx.len() {
        return Err(format!(
            "BIM row count mismatch while writing P.site: wrote {}, expected {} from {}",
            want_ptr,
            active_row_idx.len(),
            bim_path
        ));
    }
    writer.flush().map_err(|e| format!("{out_path}: {e}"))?;
    Ok(())
}

fn for_each_processed_row<F>(
    cfg: &StreamRsvdConfig,
    mut on_row: F,
) -> Result<(usize, usize), String>
where
    F: FnMut(usize, &[f32]) -> Result<(), String>,
{
    let mut it = open_stream_iter(cfg)?;
    let n = it.n_samples();
    if n == 0 {
        return Err("no samples found in genotype input".to_string());
    }

    let mut row_idx: usize = 0;
    while let Some((mut row, mut ref_allele, mut alt_allele)) = it.next_row_raw() {
        if row.len() != n {
            return Err(format!(
                "inconsistent sample count while streaming: expected {n}, got {}",
                row.len()
            ));
        }
        normalize_numeric_row(&mut row);

        if cfg.snps_only && !is_simple_snp_alleles(&ref_allele, &alt_allele) {
            continue;
        }

        let keep = process_snp_row(
            &mut row,
            &mut ref_allele,
            &mut alt_allele,
            cfg.maf,
            cfg.missing_rate,
            true,
            false,
            0.02,
        );
        if !keep {
            continue;
        }

        on_row(row_idx, &row)?;
        row_idx += 1;
    }
    Ok((row_idx, n))
}

#[inline]
fn checked_cblas_dim_local(v: usize, label: &str) -> Result<CblasInt, String> {
    if v > CblasInt::MAX as usize {
        return Err(format!("dimension overflow for {label}: {v}"));
    }
    Ok(v as CblasInt)
}

fn for_each_processed_centered_block<F>(
    cfg: &StreamRsvdConfig,
    row_freq: &[f32],
    block_rows: usize,
    timing: Option<&mut RsvdKernelTiming>,
    mut on_block: F,
) -> Result<(usize, usize), String>
where
    F: FnMut(usize, usize, &[f32]) -> Result<(), String>,
{
    let mut it = open_stream_iter(cfg)?;
    let n = it.n_samples();
    if n == 0 {
        return Err("no samples found in genotype input".to_string());
    }
    let block_rows_use = block_rows.max(1);
    let mut block = vec![0.0_f32; block_rows_use * n];
    let mut row_idx: usize = 0;
    let mut filled: usize = 0;
    let measure_timing = timing.is_some();
    let mut decode_s = 0.0_f64;
    let mut gemm_s = 0.0_f64;

    while let Some((mut row, mut ref_allele, mut alt_allele)) = it.next_row_raw() {
        let t_decode = if measure_timing {
            Some(Instant::now())
        } else {
            None
        };
        if row.len() != n {
            return Err(format!(
                "inconsistent sample count while streaming: expected {n}, got {}",
                row.len()
            ));
        }
        normalize_numeric_row(&mut row);

        if cfg.snps_only && !is_simple_snp_alleles(&ref_allele, &alt_allele) {
            continue;
        }

        let keep = process_snp_row(
            &mut row,
            &mut ref_allele,
            &mut alt_allele,
            cfg.maf,
            cfg.missing_rate,
            true,
            false,
            0.02,
        );
        if !keep {
            continue;
        }
        if row_idx >= row_freq.len() {
            return Err("streamed row count exceeded expected row_freq length".to_string());
        }

        let mean_g = 2.0_f32 * row_freq[row_idx];
        let dst = &mut block[filled * n..(filled + 1) * n];
        for col in 0..n {
            dst[col] = row[col] - mean_g;
        }
        if let Some(t0) = t_decode {
            decode_s += t0.elapsed().as_secs_f64();
        }
        filled += 1;
        row_idx += 1;

        if filled == block_rows_use {
            let t_gemm = if measure_timing {
                Some(Instant::now())
            } else {
                None
            };
            on_block(row_idx - filled, filled, &block[..filled * n])?;
            if let Some(t0) = t_gemm {
                gemm_s += t0.elapsed().as_secs_f64();
            }
            filled = 0;
        }
    }

    if filled > 0 {
        let t_gemm = if measure_timing {
            Some(Instant::now())
        } else {
            None
        };
        on_block(row_idx - filled, filled, &block[..filled * n])?;
        if let Some(t0) = t_gemm {
            gemm_s += t0.elapsed().as_secs_f64();
        }
    }
    if let Some(t) = timing {
        t.decode_s += decode_s;
        t.gemm_s += gemm_s;
    }
    Ok((row_idx, n))
}

#[inline]
fn accum_gram_lower_f64_local(
    block: &[f32],
    rows: usize,
    cols: usize,
    gram: &mut [f64],
    gram_block: &mut [f32],
) -> Result<(), String> {
    debug_assert_eq!(block.len(), rows * cols);
    debug_assert_eq!(gram.len(), cols * cols);
    debug_assert_eq!(gram_block.len(), cols * cols);
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    gram_block.fill(0.0);
    let rows_i = checked_cblas_dim_local(rows, "rows")?;
    let cols_i = checked_cblas_dim_local(cols, "cols")?;
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            cols_i,
            cols_i,
            rows_i,
            1.0_f32,
            block.as_ptr(),
            cols_i,
            block.as_ptr(),
            cols_i,
            0.0_f32,
            gram_block.as_mut_ptr(),
            cols_i,
        );
    }
    for i in 0..(cols * cols) {
        gram[i] += gram_block[i] as f64;
    }
    Ok(())
}

#[inline]
fn fill_random_omega_block(rng: &mut StdRng, omega_block: &mut [f32]) {
    for v in omega_block.iter_mut() {
        *v = rng.sample::<f64, _>(StandardNormal) as f32;
    }
}

#[inline]
fn subtract_scaled_inplace_f32(dst: &mut [f32], alpha: f32, rhs: &[f32]) {
    debug_assert_eq!(dst.len(), rhs.len());
    dst.par_iter_mut()
        .zip(rhs.par_iter())
        .for_each(|(yi, qi)| *yi -= alpha * *qi);
}

fn thin_svd_from_tall(
    x: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    if rows == 0 || cols == 0 {
        return Err("invalid matrix shape for thin SVD".to_string());
    }
    if x.len() != rows * cols {
        return Err(format!(
            "matrix buffer size mismatch in thin SVD: len={}, expected={}",
            x.len(),
            rows * cols
        ));
    }

    let rows_i = checked_cblas_dim_local(rows, "rows")?;
    let cols_i = checked_cblas_dim_local(cols, "cols")?;
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));
    let mut gram_f32 = vec![0.0_f32; cols * cols];
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            cols_i,
            cols_i,
            rows_i,
            1.0_f32,
            x.as_ptr(),
            cols_i,
            x.as_ptr(),
            cols_i,
            0.0_f32,
            gram_f32.as_mut_ptr(),
            cols_i,
        );
    }
    let gram: Vec<f64> = gram_f32.into_iter().map(|v| v as f64).collect();

    let eig = SymmetricEigen::new(DMatrix::from_row_slice(cols, cols, &gram));
    let mut order: Vec<usize> = (0..cols).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));

    let mut s = vec![0.0_f32; cols];
    let mut v = vec![0.0_f32; cols * cols]; // row-major
    for (new_col, &src_col) in order.iter().enumerate() {
        let eval = eig.eigenvalues[src_col].max(1e-12);
        s[new_col] = (eval as f32).sqrt();
        for r in 0..cols {
            v[r * cols + new_col] = eig.eigenvectors[(r, src_col)] as f32;
        }
    }

    let mut v_scaled = v.clone();
    for c in 0..cols {
        let inv = if s[c] > 1e-12 {
            1.0_f32 / s[c]
        } else {
            0.0_f32
        };
        for r in 0..cols {
            v_scaled[r * cols + c] *= inv;
        }
    }
    let mut u = vec![0.0_f32; rows * cols]; // row-major
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            rows_i,
            cols_i,
            cols_i,
            1.0_f32,
            x.as_ptr(),
            cols_i,
            v_scaled.as_ptr(),
            cols_i,
            0.0_f32,
            u.as_mut_ptr(),
            cols_i,
        );
    }
    Ok((u, s, v))
}

fn compute_at_random_omega_stream(
    cfg: &StreamRsvdConfig,
    row_freq: &[f32], // (m,)
    m: usize,
    n: usize,
    kp: usize,
    seed: u64,
) -> Result<Vec<f32>, String> {
    if row_freq.len() != m {
        return Err("row_freq length mismatch in streaming A^T*Omega_random".to_string());
    }

    let mut y = vec![0.0_f32; n * kp]; // (n, kp)
    let block_rows = stream_rsvd_block_rows(cfg, n, m);
    let _n_i = checked_cblas_dim_local(n, "n")?;
    let _kp_i = checked_cblas_dim_local(kp, "kp")?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega_block = vec![0.0_f32; block_rows * kp];
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));
    let (seen, seen_n) = for_each_processed_centered_block(
        cfg,
        row_freq,
        block_rows,
        None,
        |_row_start, rows_here, block| {
            let rows_i = checked_cblas_dim_local(rows_here, "rows_here")?;
            let omega_cur = &mut omega_block[..rows_here * kp];
            fill_random_omega_block(&mut rng, omega_cur);
            let _ = rows_i;
            row_major_block_t_mul_mat_accum_f32(block, rows_here, n, omega_cur, kp, &mut y, None);
            Ok(())
        },
    )?;
    if seen_n != n || seen != m {
        return Err(format!(
            "A^T*Omega_random pass row mismatch: seen={seen}, expected={m}, n_seen={seen_n}, n_expected={n}"
        ));
    }
    Ok(y)
}

fn compute_a_omega_stream(
    cfg: &StreamRsvdConfig,
    omega: &[f32],    // (n, kp)
    row_freq: &[f32], // (m,)
    m: usize,
    n: usize,
    kp: usize,
) -> Result<Vec<f32>, String> {
    if omega.len() != n * kp {
        return Err("omega shape mismatch in streaming A*Omega".to_string());
    }
    if row_freq.len() != m {
        return Err("row_freq length mismatch in streaming A*Omega".to_string());
    }

    let mut out = vec![0.0_f32; m * kp]; // (m, kp)
    let block_rows = stream_rsvd_block_rows(cfg, n, m);
    let _n_i = checked_cblas_dim_local(n, "n")?;
    let _kp_i = checked_cblas_dim_local(kp, "kp")?;
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));
    let (seen, seen_n) = for_each_processed_centered_block(
        cfg,
        row_freq,
        block_rows,
        None,
        |row_start, rows_here, block| {
            let rows_i = checked_cblas_dim_local(rows_here, "rows_here")?;
            let out_block = &mut out[row_start * kp..(row_start + rows_here) * kp];
            let _ = rows_i;
            row_major_block_mul_mat_f32(block, rows_here, n, omega, kp, out_block, None);
            Ok(())
        },
    )?;
    if seen_n != n || seen != m {
        return Err(format!(
            "A*Omega pass row mismatch: seen={seen}, expected={m}, n_seen={seen_n}, n_expected={n}"
        ));
    }
    Ok(out)
}

fn compute_ata_omega_stream(
    cfg: &StreamRsvdConfig,
    omega: &[f32],    // (n, kp)
    row_freq: &[f32], // (m,)
    m: usize,
    n: usize,
    kp: usize,
    timing: Option<&mut RsvdKernelTiming>,
) -> Result<Vec<f32>, String> {
    if omega.len() != n * kp {
        return Err("omega shape mismatch in streaming A^T*(A*Omega)".to_string());
    }
    if row_freq.len() != m {
        return Err("row_freq length mismatch in streaming A^T*(A*Omega)".to_string());
    }

    let mut out = vec![0.0_f32; n * kp];
    let block_rows = stream_rsvd_block_rows(cfg, n, m);
    let _n_i = checked_cblas_dim_local(n, "n")?;
    let _kp_i = checked_cblas_dim_local(kp, "kp")?;
    let mut g_block = vec![0.0_f32; block_rows * kp];
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));
    let (seen, seen_n) = for_each_processed_centered_block(
        cfg,
        row_freq,
        block_rows,
        timing,
        |_row_start, rows_here, block| {
            let rows_i = checked_cblas_dim_local(rows_here, "rows_here")?;
            let cur_g = &mut g_block[..rows_here * kp];
            row_major_block_mul_mat_f32(block, rows_here, n, omega, kp, cur_g, None);
            let _ = rows_i;
            row_major_block_t_mul_mat_accum_f32(block, rows_here, n, cur_g, kp, &mut out, None);
            Ok(())
        },
    )?;
    if seen_n != n || seen != m {
        return Err(format!(
            "A^T*(A*Omega) pass row mismatch: seen={seen}, expected={m}, n_seen={seen_n}, n_expected={n}"
        ));
    }
    Ok(out)
}

fn compute_gram_aq_stream(
    cfg: &StreamRsvdConfig,
    q: &[f32],        // (n, kp)
    row_freq: &[f32], // (m,)
    m: usize,
    n: usize,
    kp: usize,
) -> Result<Vec<f64>, String> {
    if q.len() != n * kp {
        return Err("q shape mismatch in streaming gram(AQ)".to_string());
    }
    if row_freq.len() != m {
        return Err("row_freq length mismatch in streaming gram(AQ)".to_string());
    }

    let block_rows = stream_rsvd_block_rows(cfg, n, m);
    let _n_i = checked_cblas_dim_local(n, "n")?;
    let _kp_i = checked_cblas_dim_local(kp, "kp")?;
    let mut gram = vec![0.0_f64; kp * kp];
    let mut gram_block = vec![0.0_f32; kp * kp];
    let mut g_block = vec![0.0_f32; block_rows * kp];
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));
    let (seen, seen_n) = for_each_processed_centered_block(
        cfg,
        row_freq,
        block_rows,
        None,
        |_row_start, rows_here, block| {
            let rows_i = checked_cblas_dim_local(rows_here, "rows_here")?;
            let cur_g = &mut g_block[..rows_here * kp];
            row_major_block_mul_mat_f32(block, rows_here, n, q, kp, cur_g, None);
            let _ = rows_i;
            accum_gram_lower_f64_local(cur_g, rows_here, kp, &mut gram, &mut gram_block)?;
            Ok(())
        },
    )?;
    if seen_n != n || seen != m {
        return Err(format!(
            "gram(AQ) pass row mismatch: seen={seen}, expected={m}, n_seen={seen_n}, n_expected={n}"
        ));
    }
    for i in 0..kp {
        for j in 0..i {
            gram[j * kp + i] = gram[i * kp + j];
        }
    }
    Ok(gram)
}

#[pyfunction]
#[pyo3(signature = (
    genotype_path,
    k,
    seed=42,
    power=5,
    tol=1e-1,
    snps_only=true,
    maf=0.02,
    missing_rate=0.05,
    delimiter=None,
    mmap_window_mb=0,
    memory_mb=0
))]
pub fn admx_rsvd_stream<'py>(
    py: Python<'py>,
    genotype_path: String,
    k: usize,
    seed: u64,
    power: usize,
    tol: f32,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
    delimiter: Option<String>,
    mmap_window_mb: usize,
    memory_mb: usize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>)> {
    if k == 0 {
        return Err(PyRuntimeError::new_err("k must be > 0"));
    }
    if !(0.0..=0.5).contains(&maf) {
        return Err(PyRuntimeError::new_err("maf must be within [0, 0.5]"));
    }
    if !(0.0..=1.0).contains(&missing_rate) {
        return Err(PyRuntimeError::new_err(
            "missing_rate must be within [0, 1]",
        ));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }

    let cfg = StreamRsvdConfig {
        genotype_path,
        snps_only,
        maf,
        missing_rate,
        delimiter,
        mmap_window_mb,
        memory_mb,
    };
    let t_total = Instant::now();
    let mut t_stage = Instant::now();

    let packed_backend = try_build_packed_bed_backend(&cfg).map_err(PyRuntimeError::new_err)?;

    let mut row_freq: Vec<f32> = Vec::new();
    let (m, n) = if let Some(backend) = packed_backend.as_ref() {
        row_freq = backend.row_freq.clone();
        (backend.n_snps, backend.n_samples)
    } else {
        for_each_processed_row(&cfg, |_idx, row| {
            let mut alt_sum = 0.0_f64;
            let mut non_missing = 0usize;
            for &g in row.iter() {
                if g >= 0.0 && g.is_finite() {
                    alt_sum += g as f64;
                    non_missing += 1;
                }
            }
            let freq = if non_missing > 0 {
                (alt_sum / (2.0 * non_missing as f64)) as f32
            } else {
                0.0_f32
            };
            row_freq.push(freq);
            Ok(())
        })
        .map_err(PyRuntimeError::new_err)?
    };
    let backend_s = t_stage.elapsed().as_secs_f64();

    if m == 0 {
        return Err(PyRuntimeError::new_err(
            "no SNPs passed filtering in streaming RSVD",
        ));
    }
    if n == 0 {
        return Err(PyRuntimeError::new_err(
            "no samples available for streaming RSVD",
        ));
    }

    let k_eff = k.min(m);
    let kp = admx_rsvd_kp(k_eff, m.max(1).min(n.max(1)));
    let block_rows = if let Some(backend) = packed_backend.as_ref() {
        packed_backend_block_rows(backend, m)
    } else {
        stream_rsvd_block_rows(&cfg, n, m)
    };
    let backend_extra = if let Some(backend) = packed_backend.as_ref() {
        format!(
            " total_sites={} active_row_idx={}",
            backend.n_total_sites,
            format_debug_bytes_local(
                (backend.active_row_idx.len() * std::mem::size_of::<usize>()) as u64
            ),
        )
    } else {
        String::new()
    };
    emit_rsvd_rss_debug(
        "stream/backend_ready",
        &format!(
            "mode={} n_samples={} n_snps={} kp={} row_freq={}{}",
            if packed_backend.is_some() {
                "packed_mmap"
            } else {
                "stream"
            },
            n,
            m,
            kp,
            format_f32_vec_bytes(row_freq.len()),
            backend_extra,
        ),
    );
    emit_rsvd_time_debug(
        "stream/backend_ready",
        &format!("elapsed={}", fmt_stage_secs(backend_s)),
    );

    emit_rsvd_rss_debug(
        "stream/omega_ready",
        &format!(
            "omega_mode=random_block rows={} kp={} omega_scratch={}",
            block_rows,
            kp,
            format_f32_vec_bytes(block_rows * kp)
        ),
    );
    t_stage = Instant::now();
    let mut y = if let Some(backend) = packed_backend.as_ref() {
        compute_at_random_omega_packed(backend, kp, seed)
    } else {
        compute_at_random_omega_stream(&cfg, &row_freq, m, n, kp, seed)
    }
    .map_err(PyRuntimeError::new_err)?;
    let init_proj_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream/y_ready",
        &format!(
            "y_shape=({}, {}) y={}",
            n,
            kp,
            format_f32_vec_bytes(y.len())
        ),
    );
    emit_rsvd_time_debug(
        "stream/init_proj",
        &format!("elapsed={}", fmt_stage_secs(init_proj_s)),
    );
    t_stage = Instant::now();
    let (mut q, _, _) = thin_svd_from_tall(&y, n, kp).map_err(PyRuntimeError::new_err)?;
    let init_q_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream/q_ready",
        &format!(
            "q_shape=({}, {}) q={}",
            n,
            kp,
            format_f32_vec_bytes(q.len())
        ),
    );
    emit_rsvd_time_debug(
        "stream/init_q",
        &format!("elapsed={}", fmt_stage_secs(init_q_s)),
    );

    let mut sk = vec![0.0_f32; kp];
    let mut alpha = 0.0_f32;
    let mut power_mul_s = 0.0_f64;
    let mut power_mul_decode_s = 0.0_f64;
    let mut power_mul_gemm_s = 0.0_f64;
    let mut power_qr_s = 0.0_f64;
    let mut power_iters_done = 0usize;
    let measure_kernel_timing = rsvd_time_debug_enabled();
    for it in 0..power {
        let t_power_mul = Instant::now();
        let mut kernel_timing = RsvdKernelTiming::default();
        y = if let Some(backend) = packed_backend.as_ref() {
            compute_ata_omega_packed(
                backend,
                &q,
                kp,
                if measure_kernel_timing {
                    Some(&mut kernel_timing)
                } else {
                    None
                },
            )
        } else {
            compute_ata_omega_stream(
                &cfg,
                &q,
                &row_freq,
                m,
                n,
                kp,
                if measure_kernel_timing {
                    Some(&mut kernel_timing)
                } else {
                    None
                },
            )
        }
        .map_err(PyRuntimeError::new_err)?;
        power_mul_s += t_power_mul.elapsed().as_secs_f64();
        power_mul_decode_s += kernel_timing.decode_s;
        power_mul_gemm_s += kernel_timing.gemm_s;
        subtract_scaled_inplace_f32(&mut y, alpha, &q);
        let t_power_qr = Instant::now();
        let (q_new, s_y, _) = thin_svd_from_tall(&y, n, kp).map_err(PyRuntimeError::new_err)?;
        power_qr_s += t_power_qr.elapsed().as_secs_f64();
        q = q_new;
        power_iters_done = it + 1;

        if it > 0 {
            let mut max_rel = 0.0_f32;
            for i in 0..k_eff {
                let sk_now = s_y[i] + alpha;
                let denom = sk_now.max(1e-12);
                let rel = ((sk_now - sk[i]).abs()) / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
                sk[i] = sk_now;
            }
            if max_rel < tol {
                break;
            }
        } else {
            for i in 0..kp {
                sk[i] = s_y[i] + alpha;
            }
        }

        let tail = s_y[kp - 1];
        if alpha < tail {
            alpha = 0.5 * (alpha + tail);
        }
    }

    t_stage = Instant::now();
    let g_small = if let Some(backend) = packed_backend.as_ref() {
        compute_a_omega_packed(backend, &q, kp)
    } else {
        compute_a_omega_stream(&cfg, &q, &row_freq, m, n, kp)
    }
    .map_err(PyRuntimeError::new_err)?;
    let final_proj_s = t_stage.elapsed().as_secs_f64();
    t_stage = Instant::now();
    let (u_small, s_all, _) =
        thin_svd_from_tall(&g_small, m, kp).map_err(PyRuntimeError::new_err)?;
    let final_svd_s = t_stage.elapsed().as_secs_f64();

    let mut eigvals = vec![0.0_f32; k_eff];
    for i in 0..k_eff {
        eigvals[i] = s_all[i] * s_all[i];
    }
    let mut eigvecs = vec![0.0_f32; m * k_eff];
    t_stage = Instant::now();
    for r in 0..m {
        let src = &u_small[r * kp..(r + 1) * kp];
        let dst = &mut eigvecs[r * k_eff..(r + 1) * k_eff];
        dst.copy_from_slice(&src[..k_eff]);
    }
    let final_vec_s = t_stage.elapsed().as_secs_f64();

    let eval_arr = PyArray1::<f32>::zeros(py, [k_eff], false).into_bound();
    let evec_arr = PyArray2::<f32>::zeros(py, [m, k_eff], false).into_bound();
    let eval_slice = unsafe {
        eval_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("eigvals output not contiguous"))?
    };
    let evec_slice = unsafe {
        evec_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("eigvecs output not contiguous"))?
    };
    emit_rsvd_rss_debug(
        "stream/final_ready",
        &format!(
            "eigvals={} eigvecs_shape=({}, {}) eigvecs={}",
            format_f32_vec_bytes(eigvals.len()),
            m,
            k_eff,
            format_f32_vec_bytes(eigvecs.len()),
        ),
    );
    let total_s = t_total.elapsed().as_secs_f64();
    emit_rsvd_time_debug(
        "stream/summary",
        &format!(
            "total={} backend={} init_proj={} init_q={} power_mul={} power_mul_decode={} power_mul_gemm={} power_qr={} final_proj={} final_svd={} final_vec={} power_iters={} block_rows={}",
            fmt_stage_secs(total_s),
            fmt_stage_secs(backend_s),
            fmt_stage_secs(init_proj_s),
            fmt_stage_secs(init_q_s),
            fmt_stage_secs(power_mul_s),
            fmt_stage_secs(power_mul_decode_s),
            fmt_stage_secs(power_mul_gemm_s),
            fmt_stage_secs(power_qr_s),
            fmt_stage_secs(final_proj_s),
            fmt_stage_secs(final_svd_s),
            fmt_stage_secs(final_vec_s),
            power_iters_done,
            block_rows,
        ),
    );
    eval_slice.copy_from_slice(&eigvals);
    evec_slice.copy_from_slice(&eigvecs);
    Ok((eval_arr, evec_arr))
}

#[pyfunction]
#[pyo3(signature = (
    genotype_path,
    k,
    seed=42,
    power=5,
    tol=1e-1,
    snps_only=true,
    maf=0.02,
    missing_rate=0.05,
    delimiter=None,
    mmap_window_mb=0,
    memory_mb=0
))]
pub fn admx_rsvd_stream_sample<'py>(
    py: Python<'py>,
    genotype_path: String,
    k: usize,
    seed: u64,
    power: usize,
    tol: f32,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
    delimiter: Option<String>,
    mmap_window_mb: usize,
    memory_mb: usize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>, f64)> {
    if k == 0 {
        return Err(PyRuntimeError::new_err("k must be > 0"));
    }
    if !(0.0..=0.5).contains(&maf) {
        return Err(PyRuntimeError::new_err("maf must be within [0, 0.5]"));
    }
    if !(0.0..=1.0).contains(&missing_rate) {
        return Err(PyRuntimeError::new_err(
            "missing_rate must be within [0, 1]",
        ));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyRuntimeError::new_err("tol must be positive and finite"));
    }

    let cfg = StreamRsvdConfig {
        genotype_path,
        snps_only,
        maf,
        missing_rate,
        delimiter,
        mmap_window_mb,
        memory_mb,
    };
    let t_total = Instant::now();
    let mut t_stage = Instant::now();

    let packed_backend = try_build_packed_bed_backend(&cfg).map_err(PyRuntimeError::new_err)?;
    let packed_backend_s = t_stage.elapsed().as_secs_f64();
    if let Some(backend) = packed_backend.as_ref() {
        let (eigvals, eigvecs_sample, total_variance) =
            rsvd_stream_sample_packed_impl(backend, k, seed, power, tol, packed_backend_s)
                .map_err(PyRuntimeError::new_err)?;
        let k_eff = eigvals.len();
        let eval_arr = row_freq_to_pyarray(py, &eigvals);
        let evec_arr = vec_to_pyarray2_f32(py, backend.n_samples, k_eff, &eigvecs_sample)?;
        return Ok((eval_arr, evec_arr, total_variance));
    }

    let mut row_freq: Vec<f32> = Vec::new();
    let mut varsum: f64 = 0.0;
    let mut total_ss: f64 = 0.0;
    let (m, n) = for_each_processed_row(&cfg, |_idx, row| {
        let mut alt_sum = 0.0_f64;
        let mut non_missing = 0usize;
        let mut sq_sum = 0.0_f64;
        for &g in row.iter() {
            if g >= 0.0 && g.is_finite() {
                alt_sum += g as f64;
                sq_sum += (g as f64) * (g as f64);
                non_missing += 1;
            }
        }
        let freq = if non_missing > 0 {
            (alt_sum / (2.0 * non_missing as f64)) as f32
        } else {
            0.0_f32
        };
        varsum += 2.0_f64 * (freq as f64) * (1.0_f64 - freq as f64);
        if non_missing > 0 {
            let row_mean = alt_sum / non_missing as f64;
            total_ss += (sq_sum - alt_sum * row_mean).max(0.0_f64);
        }
        row_freq.push(freq);
        Ok(())
    })
    .map_err(PyRuntimeError::new_err)?;
    let backend_s = t_stage.elapsed().as_secs_f64();

    if m == 0 {
        return Err(PyRuntimeError::new_err(
            "no SNPs passed filtering in streaming RSVD",
        ));
    }
    if n == 0 {
        return Err(PyRuntimeError::new_err(
            "no samples available for streaming RSVD",
        ));
    }
    if !(varsum.is_finite() && varsum > 0.0) {
        return Err(PyRuntimeError::new_err(
            "invalid scaling denominator in streaming RSVD (varsum <= 0)",
        ));
    }
    let total_variance = total_ss / varsum;
    if !(total_variance.is_finite() && total_variance > 0.0) {
        return Err(PyRuntimeError::new_err(
            "invalid total variance in streaming RSVD (trace(K) <= 0)",
        ));
    }

    let k_eff = k.min(n);
    let kp = admx_rsvd_kp(k_eff, m.max(1).min(n.max(1)));
    let block_rows = if let Some(backend) = packed_backend.as_ref() {
        packed_backend_block_rows(backend, m)
    } else {
        stream_rsvd_block_rows(&cfg, n, m)
    };
    let backend_extra = if let Some(backend) = packed_backend.as_ref() {
        format!(
            " total_sites={} active_row_idx={}",
            backend.n_total_sites,
            format_debug_bytes_local(
                (backend.active_row_idx.len() * std::mem::size_of::<usize>()) as u64
            ),
        )
    } else {
        String::new()
    };
    emit_rsvd_rss_debug(
        "stream_sample/backend_ready",
        &format!(
            "mode={} n_samples={} n_snps={} kp={} row_freq={} varsum={:.6e} traceK={:.6e}{}",
            if packed_backend.is_some() {
                "packed_mmap"
            } else {
                "stream"
            },
            n,
            m,
            kp,
            format_f32_vec_bytes(row_freq.len()),
            varsum,
            total_variance,
            backend_extra,
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/backend_ready",
        &format!("elapsed={}", fmt_stage_secs(backend_s)),
    );

    emit_rsvd_rss_debug(
        "stream_sample/omega_ready",
        &format!(
            "omega_mode=random_block rows={} kp={} omega_scratch={}",
            block_rows,
            kp,
            format_f32_vec_bytes(block_rows * kp)
        ),
    );
    t_stage = Instant::now();
    let mut y = if let Some(backend) = packed_backend.as_ref() {
        compute_at_random_omega_packed(backend, kp, seed)
    } else {
        compute_at_random_omega_stream(&cfg, &row_freq, m, n, kp, seed)
    }
    .map_err(PyRuntimeError::new_err)?;
    let init_proj_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/y_ready",
        &format!(
            "y_shape=({}, {}) y={}",
            n,
            kp,
            format_f32_vec_bytes(y.len())
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/init_proj",
        &format!("elapsed={}", fmt_stage_secs(init_proj_s)),
    );
    t_stage = Instant::now();
    let (mut q, _, _) = thin_svd_from_tall(&y, n, kp).map_err(PyRuntimeError::new_err)?;
    let init_q_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/q_ready",
        &format!(
            "q_shape=({}, {}) q={}",
            n,
            kp,
            format_f32_vec_bytes(q.len())
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/init_q",
        &format!("elapsed={}", fmt_stage_secs(init_q_s)),
    );

    let mut sk = vec![0.0_f32; kp];
    let mut alpha = 0.0_f32;
    let mut power_mul_s = 0.0_f64;
    let mut power_mul_decode_s = 0.0_f64;
    let mut power_mul_gemm_s = 0.0_f64;
    let mut power_qr_s = 0.0_f64;
    let mut power_iters_done = 0usize;
    let measure_kernel_timing = rsvd_time_debug_enabled();
    for it in 0..power {
        let t_power_mul = Instant::now();
        let mut kernel_timing = RsvdKernelTiming::default();
        y = if let Some(backend) = packed_backend.as_ref() {
            compute_ata_omega_packed(
                backend,
                &q,
                kp,
                if measure_kernel_timing {
                    Some(&mut kernel_timing)
                } else {
                    None
                },
            )
        } else {
            compute_ata_omega_stream(
                &cfg,
                &q,
                &row_freq,
                m,
                n,
                kp,
                if measure_kernel_timing {
                    Some(&mut kernel_timing)
                } else {
                    None
                },
            )
        }
        .map_err(PyRuntimeError::new_err)?;
        power_mul_s += t_power_mul.elapsed().as_secs_f64();
        power_mul_decode_s += kernel_timing.decode_s;
        power_mul_gemm_s += kernel_timing.gemm_s;
        subtract_scaled_inplace_f32(&mut y, alpha, &q);
        let t_power_qr = Instant::now();
        let (q_new, s_y, _) = thin_svd_from_tall(&y, n, kp).map_err(PyRuntimeError::new_err)?;
        power_qr_s += t_power_qr.elapsed().as_secs_f64();
        q = q_new;
        power_iters_done = it + 1;

        if it > 0 {
            let mut max_rel = 0.0_f32;
            for i in 0..k_eff {
                let sk_now = s_y[i] + alpha;
                let denom = sk_now.max(1e-12);
                let rel = ((sk_now - sk[i]).abs()) / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
                sk[i] = sk_now;
            }
            if max_rel < tol {
                break;
            }
        } else {
            for i in 0..kp {
                sk[i] = s_y[i] + alpha;
            }
        }

        let tail = s_y[kp - 1];
        if alpha < tail {
            alpha = 0.5 * (alpha + tail);
        }
    }

    t_stage = Instant::now();
    let gram = if let Some(backend) = packed_backend.as_ref() {
        compute_gram_aq_packed(backend, &q, kp).map_err(PyRuntimeError::new_err)?
    } else {
        compute_gram_aq_stream(&cfg, &q, &row_freq, m, n, kp).map_err(PyRuntimeError::new_err)?
    };
    let gram_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/gram_ready",
        &format!(
            "gram_shape=({}, {}) gram={}",
            kp,
            kp,
            format_debug_bytes_local(
                (gram.len().saturating_mul(std::mem::size_of::<f64>())) as u64
            ),
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/gram",
        &format!("elapsed={}", fmt_stage_secs(gram_s)),
    );
    t_stage = Instant::now();
    let (s_all, v_small) =
        rsvd_right_singular_from_gram(&gram, kp).map_err(PyRuntimeError::new_err)?;
    let eig_s = t_stage.elapsed().as_secs_f64();
    let mut eigvals = vec![0.0_f32; k_eff];
    // Align PCA eigenvalue scaling with the GRM pathway in python/janusx/script/pca.py.
    // For sample-side PCA: lambda = sigma^2 / varsum, where
    // varsum = sum_i 2p_i(1-p_i) over retained variants.
    let scale = varsum as f32;
    for i in 0..k_eff {
        eigvals[i] = (s_all[i] * s_all[i]) / scale;
    }

    // Sample-side eigenvectors/right singular vectors: q @ v_small[:, :k_eff]
    t_stage = Instant::now();
    let eigvecs_sample =
        rsvd_project_sample_eigvecs(&q, n, kp, &v_small, k_eff).map_err(PyRuntimeError::new_err)?;
    let final_vec_s = t_stage.elapsed().as_secs_f64();

    let eval_arr = PyArray1::<f32>::zeros(py, [k_eff], false).into_bound();
    let evec_arr = PyArray2::<f32>::zeros(py, [n, k_eff], false).into_bound();
    let eval_slice = unsafe {
        eval_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("eigvals output not contiguous"))?
    };
    let evec_slice = unsafe {
        evec_arr
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("sample eigvecs output not contiguous"))?
    };
    emit_rsvd_rss_debug(
        "stream_sample/final_ready",
        &format!(
            "eigvals={} eigvecs_shape=({}, {}) eigvecs={}",
            format_f32_vec_bytes(eigvals.len()),
            n,
            k_eff,
            format_f32_vec_bytes(eigvecs_sample.len()),
        ),
    );
    let total_s = t_total.elapsed().as_secs_f64();
    emit_rsvd_time_debug(
        "stream_sample/summary",
        &format!(
            "total={} backend={} init_proj={} init_q={} power_mul={} power_mul_decode={} power_mul_gemm={} power_qr={} gram={} eig={} final_vec={} power_iters={} block_rows={}",
            fmt_stage_secs(total_s),
            fmt_stage_secs(backend_s),
            fmt_stage_secs(init_proj_s),
            fmt_stage_secs(init_q_s),
            fmt_stage_secs(power_mul_s),
            fmt_stage_secs(power_mul_decode_s),
            fmt_stage_secs(power_mul_gemm_s),
            fmt_stage_secs(power_qr_s),
            fmt_stage_secs(gram_s),
            fmt_stage_secs(eig_s),
            fmt_stage_secs(final_vec_s),
            power_iters_done,
            block_rows,
        ),
    );
    eval_slice.copy_from_slice(&eigvals);
    evec_slice.copy_from_slice(&eigvecs_sample);
    Ok((eval_arr, evec_arr, total_variance))
}

fn loglikelihood_f32_impl(g: &ArrayView2<'_, u8>, p: &[f32], q: &[f32], n: usize, k: usize) -> f64 {
    let (m, _) = g.dim();
    (0..m)
        .into_par_iter()
        .map(|i| {
            let p_row = &p[i * k..(i + 1) * k];
            let mut part = 0.0_f64;
            for j in 0..n {
                let gv = g[(i, j)];
                if gv == 3 {
                    continue;
                }
                let q_row = &q[j * k..(j + 1) * k];
                let mut rec = 0.0_f32;
                for kk in 0..k {
                    rec += p_row[kk] * q_row[kk];
                }
                let rec = rec.clamp(1e-6, 1.0 - 1e-6) as f64;
                let g_d = gv as f64;
                part += g_d * rec.ln() + (2.0 - g_d) * (1.0 - rec).ln();
            }
            part
        })
        .sum()
}

fn loglikelihood_packed_f32_impl(backend: &PackedBedRsvd, p: &[f32], q: &[f32], k: usize) -> f64 {
    let n = backend.n_samples;
    let full_bytes = n / 4;
    let rem = n % 4;
    let byte_lut = packed_byte_lut();
    (0..backend.n_snps)
        .into_par_iter()
        .map(|i| {
            let row = packed_backend_row_slice(backend, i);
            let p_row = &p[i * k..(i + 1) * k];
            let flip = backend.row_flip[i];
            let mut part = 0.0_f64;
            let mut col = 0usize;
            for &b in row.iter().take(full_bytes) {
                let codes = &byte_lut.code4[b as usize];
                for &code in codes.iter() {
                    if let Some(gv_f32) = packed_code_minor_allele_g(code, flip) {
                        let q_row = &q[col * k..(col + 1) * k];
                        let mut rec = 0.0_f32;
                        for kk in 0..k {
                            rec += p_row[kk] * q_row[kk];
                        }
                        let rec = rec.clamp(1e-6, 1.0 - 1e-6) as f64;
                        let gv = gv_f32 as f64;
                        part += gv * rec.ln() + (2.0 - gv) * (1.0 - rec).ln();
                    }
                    col += 1;
                }
            }
            if rem > 0 {
                let codes = &byte_lut.code4[row[full_bytes] as usize];
                for &code in codes.iter().take(rem) {
                    if let Some(gv_f32) = packed_code_minor_allele_g(code, flip) {
                        let q_row = &q[col * k..(col + 1) * k];
                        let mut rec = 0.0_f32;
                        for kk in 0..k {
                            rec += p_row[kk] * q_row[kk];
                        }
                        let rec = rec.clamp(1e-6, 1.0 - 1e-6) as f64;
                        let gv = gv_f32 as f64;
                        part += gv * rec.ln() + (2.0 - gv) * (1.0 - rec).ln();
                    }
                    col += 1;
                }
            }
            part
        })
        .sum()
}

fn row_freq_to_pyarray<'py>(py: Python<'py>, row_freq: &[f32]) -> Bound<'py, PyArray1<f32>> {
    #[allow(deprecated)]
    PyArray1::from_owned_array(py, Array1::from_vec(row_freq.to_vec())).into_bound()
}

fn vec_to_pyarray2_f32<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    data: &[f32],
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let out = PyArray2::<f32>::zeros(py, [rows, cols], false).into_bound();
    let out_s = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_s.copy_from_slice(data);
    Ok(out)
}

fn vec_into_pyarray2_f32<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    data: Vec<f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if data.len() != rows.saturating_mul(cols) {
        return Err(PyRuntimeError::new_err(format!(
            "output buffer size mismatch: got {}, expected {}",
            data.len(),
            rows.saturating_mul(cols)
        )));
    }
    let arr = Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to build output array: {e}")))?;
    #[allow(deprecated)]
    Ok(PyArray2::from_owned_array(py, arr).into_bound())
}

fn vec_to_pyarray1_i64<'py>(py: Python<'py>, data: &[i64]) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let out = PyArray1::<i64>::zeros(py, [data.len()], false).into_bound();
    let out_s = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_s.copy_from_slice(data);
    Ok(out)
}

fn cv_observed_fold_counts_packed_impl(
    backend: &PackedBedRsvd,
    folds: usize,
    seed: u64,
) -> Result<(usize, Vec<i64>), String> {
    validate_cv_folds(folds)?;
    let n = backend.n_samples;
    let m = backend.n_snps;
    if n == 0 || m == 0 {
        return Ok((0, vec![0_i64; folds]));
    }
    let full_bytes = n / 4;
    let rem = n % 4;
    let byte_lut = packed_byte_lut();
    let chunk_rows = packed_backend_block_rows(backend, m)
        .saturating_mul(16)
        .clamp(1024, 16384)
        .min(m.max(1));
    let mut observed_total = 0usize;
    let mut fold_counts = vec![0_i64; folds];

    for chunk_start in (0..m).step_by(chunk_rows) {
        let chunk_end = (chunk_start + chunk_rows).min(m);
        let (obs_chunk, fold_chunk) = (chunk_start..chunk_end)
            .into_par_iter()
            .fold(
                || (0usize, vec![0_i64; folds]),
                |(mut obs_local, mut fold_local), i| {
                    let row = packed_backend_row_slice(backend, i);
                    let row_offset = (i as u64) * (n as u64);
                    let mut col = 0usize;
                    for &packed in row.iter().take(full_bytes) {
                        let codes = &byte_lut.code4[packed as usize];
                        for &code in codes.iter() {
                            if code != 0b01 {
                                let fid = cv_fold_id(row_offset + (col as u64), folds, seed);
                                fold_local[fid] += 1;
                                obs_local += 1;
                            }
                            col += 1;
                        }
                    }
                    if rem > 0 {
                        let codes = &byte_lut.code4[row[full_bytes] as usize];
                        for &code in codes.iter().take(rem) {
                            if code != 0b01 {
                                let fid = cv_fold_id(row_offset + (col as u64), folds, seed);
                                fold_local[fid] += 1;
                                obs_local += 1;
                            }
                            col += 1;
                        }
                    }
                    (obs_local, fold_local)
                },
            )
            .reduce(
                || (0usize, vec![0_i64; folds]),
                |(obs_a, mut fold_a), (obs_b, fold_b)| {
                    for f in 0..folds {
                        fold_a[f] += fold_b[f];
                    }
                    (obs_a + obs_b, fold_a)
                },
            );
        observed_total += obs_chunk;
        for f in 0..folds {
            fold_counts[f] += fold_chunk[f];
        }
    }

    Ok((observed_total, fold_counts))
}

fn cv_training_row_freq_packed_impl(
    backend: &PackedBedRsvd,
    fold_id_keepout: usize,
    folds: usize,
    seed: u64,
) -> Result<(Vec<f32>, usize, usize), String> {
    validate_cv_folds(folds)?;
    if fold_id_keepout >= folds {
        return Err(format!(
            "fold_id must be within [0, {}), got {fold_id_keepout}",
            folds
        ));
    }
    let n = backend.n_samples;
    let m = backend.n_snps;
    let full_bytes = n / 4;
    let rem = n % 4;
    let byte_lut = packed_byte_lut();
    let chunk_rows = packed_backend_block_rows(backend, m)
        .saturating_mul(16)
        .clamp(1024, 16384)
        .min(m.max(1));
    let mut row_freq = vec![0.0_f32; m];
    let mut train_observed_total = 0usize;
    let mut holdout_observed_total = 0usize;

    for chunk_start in (0..m).step_by(chunk_rows) {
        let chunk_end = (chunk_start + chunk_rows).min(m);
        let chunk_stats: Vec<(f32, usize, usize)> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|i| {
                let row = packed_backend_row_slice(backend, i);
                let flip = backend.row_flip[i];
                let row_offset = (i as u64) * (n as u64);
                let mut train_non_missing = 0usize;
                let mut holdout_non_missing = 0usize;
                let mut alt_sum = 0.0_f64;
                let mut col = 0usize;
                for &packed in row.iter().take(full_bytes) {
                    let codes = &byte_lut.code4[packed as usize];
                    for &code in codes.iter() {
                        if let Some(gv) = packed_code_minor_allele_g(code, flip) {
                            if cv_fold_id(row_offset + (col as u64), folds, seed) == fold_id_keepout
                            {
                                holdout_non_missing += 1;
                            } else {
                                train_non_missing += 1;
                                alt_sum += gv as f64;
                            }
                        }
                        col += 1;
                    }
                }
                if rem > 0 {
                    let codes = &byte_lut.code4[row[full_bytes] as usize];
                    for &code in codes.iter().take(rem) {
                        if let Some(gv) = packed_code_minor_allele_g(code, flip) {
                            if cv_fold_id(row_offset + (col as u64), folds, seed) == fold_id_keepout
                            {
                                holdout_non_missing += 1;
                            } else {
                                train_non_missing += 1;
                                alt_sum += gv as f64;
                            }
                        }
                        col += 1;
                    }
                }
                let freq = if train_non_missing > 0 {
                    (alt_sum / (2.0_f64 * train_non_missing as f64)) as f32
                } else {
                    0.0_f32
                };
                (freq, train_non_missing, holdout_non_missing)
            })
            .collect();
        for (off, (freq, train_ct, holdout_ct)) in chunk_stats.into_iter().enumerate() {
            row_freq[chunk_start + off] = freq;
            train_observed_total += train_ct;
            holdout_observed_total += holdout_ct;
        }
    }

    Ok((row_freq, train_observed_total, holdout_observed_total))
}

fn decode_train_masked_block_rows_f32(
    backend: &PackedBedRsvd,
    row_freq_train: &[f32],
    row_start: usize,
    rows: usize,
    fold_id_keepout: usize,
    folds: usize,
    cv_seed: u64,
    out: &mut [f32],
) -> Result<(), String> {
    let n = backend.n_samples;
    if row_freq_train.len() != backend.n_snps {
        return Err(format!(
            "row_freq_train length mismatch: got {}, expected {}",
            row_freq_train.len(),
            backend.n_snps
        ));
    }
    if out.len() != rows * n {
        return Err(format!(
            "masked decode output length mismatch: got {}, expected {}",
            out.len(),
            rows * n
        ));
    }
    let full_bytes = n / 4;
    let rem = n % 4;
    let code4 = &packed_byte_lut().code4;
    out.par_chunks_mut(n)
        .enumerate()
        .for_each(|(local_row, out_row)| {
            let i = row_start + local_row;
            let row = packed_backend_row_slice(backend, i);
            let mean_g = 2.0_f32 * row_freq_train[i];
            let value_lut: [f32; 4] = if backend.row_flip[i] {
                [
                    2.0_f32 - mean_g,
                    0.0_f32,
                    1.0_f32 - mean_g,
                    0.0_f32 - mean_g,
                ]
            } else {
                [
                    0.0_f32 - mean_g,
                    0.0_f32,
                    1.0_f32 - mean_g,
                    2.0_f32 - mean_g,
                ]
            };
            let row_offset = (i as u64) * (n as u64);
            let mut col = 0usize;
            for &packed in row.iter().take(full_bytes) {
                let codes = &code4[packed as usize];
                for &code in codes.iter() {
                    let flat_idx = row_offset + (col as u64);
                    out_row[col] = if code == 0b01
                        || cv_fold_id(flat_idx, folds, cv_seed) == fold_id_keepout
                    {
                        0.0_f32
                    } else {
                        value_lut[code as usize]
                    };
                    col += 1;
                }
            }
            if rem > 0 {
                let codes = &code4[row[full_bytes] as usize];
                for &code in codes.iter().take(rem) {
                    let flat_idx = row_offset + (col as u64);
                    out_row[col] = if code == 0b01
                        || cv_fold_id(flat_idx, folds, cv_seed) == fold_id_keepout
                    {
                        0.0_f32
                    } else {
                        value_lut[code as usize]
                    };
                    col += 1;
                }
            }
        });
    Ok(())
}

fn compute_a_omega_packed_fold_impl(
    backend: &PackedBedRsvd,
    row_freq_train: &[f32],
    omega: &[f32],
    kp: usize,
    fold_id_keepout: usize,
    folds: usize,
    cv_seed: u64,
) -> Result<Vec<f32>, String> {
    validate_cv_folds(folds)?;
    let m = backend.n_snps;
    let n = backend.n_samples;
    if row_freq_train.len() != m {
        return Err(format!(
            "row_freq_train length mismatch: got {}, expected {m}",
            row_freq_train.len()
        ));
    }
    if omega.len() != n * kp {
        return Err("omega shape mismatch in masked packed RSVD A*Omega".to_string());
    }
    let block_rows = packed_backend_block_rows(backend, m);
    let mut out = vec![0.0_f32; m * kp];
    let mut block = vec![0.0_f32; block_rows * n];
    let _n_i = checked_cblas_dim_local(n, "n")?;
    let _kp_i = checked_cblas_dim_local(kp, "kp")?;
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));

    for row_start in (0..m).step_by(block_rows) {
        let row_end = (row_start + block_rows).min(m);
        let cur_rows = row_end - row_start;
        let cur_block = &mut block[..cur_rows * n];
        decode_train_masked_block_rows_f32(
            backend,
            row_freq_train,
            row_start,
            cur_rows,
            fold_id_keepout,
            folds,
            cv_seed,
            cur_block,
        )?;
        let out_block = &mut out[row_start * kp..row_end * kp];
        let _ = checked_cblas_dim_local(cur_rows, "cur_rows")?;
        row_major_block_mul_mat_f32(cur_block, cur_rows, n, omega, kp, out_block, None);
    }
    Ok(out)
}

fn compute_at_random_omega_packed_fold_impl(
    backend: &PackedBedRsvd,
    row_freq_train: &[f32],
    kp: usize,
    seed: u64,
    fold_id_keepout: usize,
    folds: usize,
    cv_seed: u64,
) -> Result<Vec<f32>, String> {
    validate_cv_folds(folds)?;
    let m = backend.n_snps;
    let n = backend.n_samples;
    if row_freq_train.len() != m {
        return Err(format!(
            "row_freq_train length mismatch: got {}, expected {m}",
            row_freq_train.len()
        ));
    }
    let block_rows = packed_backend_block_rows(backend, m);
    let mut out = vec![0.0_f32; n * kp];
    let mut block = vec![0.0_f32; block_rows * n];
    let mut omega_block = vec![0.0_f32; block_rows * kp];
    let mut rng = StdRng::seed_from_u64(seed);
    let _n_i = checked_cblas_dim_local(n, "n")?;
    let _kp_i = checked_cblas_dim_local(kp, "kp")?;
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));

    for row_start in (0..m).step_by(block_rows) {
        let row_end = (row_start + block_rows).min(m);
        let cur_rows = row_end - row_start;
        let cur_block = &mut block[..cur_rows * n];
        decode_train_masked_block_rows_f32(
            backend,
            row_freq_train,
            row_start,
            cur_rows,
            fold_id_keepout,
            folds,
            cv_seed,
            cur_block,
        )?;
        let cur_omega = &mut omega_block[..cur_rows * kp];
        fill_random_omega_block(&mut rng, cur_omega);
        let _ = checked_cblas_dim_local(cur_rows, "cur_rows")?;
        row_major_block_t_mul_mat_accum_f32(cur_block, cur_rows, n, cur_omega, kp, &mut out, None);
    }
    Ok(out)
}

fn compute_ata_omega_packed_fold_impl(
    backend: &PackedBedRsvd,
    row_freq_train: &[f32],
    omega: &[f32],
    kp: usize,
    fold_id_keepout: usize,
    folds: usize,
    cv_seed: u64,
    timing: Option<&mut RsvdKernelTiming>,
) -> Result<Vec<f32>, String> {
    validate_cv_folds(folds)?;
    let m = backend.n_snps;
    let n = backend.n_samples;
    if row_freq_train.len() != m {
        return Err(format!(
            "row_freq_train length mismatch: got {}, expected {m}",
            row_freq_train.len()
        ));
    }
    if omega.len() != n * kp {
        return Err("omega shape mismatch in masked packed RSVD A^T*(A*Omega)".to_string());
    }
    let block_rows = packed_backend_block_rows(backend, m);
    let mut out = vec![0.0_f32; n * kp];
    let mut block = vec![0.0_f32; block_rows * n];
    let mut g_block = vec![0.0_f32; block_rows * kp];
    let _n_i = checked_cblas_dim_local(n, "n")?;
    let _kp_i = checked_cblas_dim_local(kp, "kp")?;
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));
    let measure_timing = timing.is_some();
    let mut decode_s = 0.0_f64;
    let mut gemm_s = 0.0_f64;

    for row_start in (0..m).step_by(block_rows) {
        let row_end = (row_start + block_rows).min(m);
        let cur_rows = row_end - row_start;
        let cur_rows_i = checked_cblas_dim_local(cur_rows, "cur_rows")?;
        let cur_block = &mut block[..cur_rows * n];
        let t_decode = if measure_timing {
            Some(Instant::now())
        } else {
            None
        };
        decode_train_masked_block_rows_f32(
            backend,
            row_freq_train,
            row_start,
            cur_rows,
            fold_id_keepout,
            folds,
            cv_seed,
            cur_block,
        )?;
        if let Some(t0) = t_decode {
            decode_s += t0.elapsed().as_secs_f64();
        }
        let cur_g = &mut g_block[..cur_rows * kp];
        let t_gemm = if measure_timing {
            Some(Instant::now())
        } else {
            None
        };
        row_major_block_mul_mat_f32(cur_block, cur_rows, n, omega, kp, cur_g, None);
        let _ = cur_rows_i;
        row_major_block_t_mul_mat_accum_f32(cur_block, cur_rows, n, cur_g, kp, &mut out, None);
        if let Some(t0) = t_gemm {
            gemm_s += t0.elapsed().as_secs_f64();
        }
    }

    if let Some(t) = timing {
        t.decode_s += decode_s;
        t.gemm_s += gemm_s;
    }
    Ok(out)
}

fn compute_gram_aq_packed_fold_impl(
    backend: &PackedBedRsvd,
    row_freq_train: &[f32],
    q: &[f32],
    kp: usize,
    fold_id_keepout: usize,
    folds: usize,
    cv_seed: u64,
) -> Result<Vec<f64>, String> {
    validate_cv_folds(folds)?;
    let m = backend.n_snps;
    let n = backend.n_samples;
    if row_freq_train.len() != m {
        return Err(format!(
            "row_freq_train length mismatch: got {}, expected {m}",
            row_freq_train.len()
        ));
    }
    if q.len() != n * kp {
        return Err("Q shape mismatch in masked packed RSVD gram(AQ)".to_string());
    }
    let block_rows = packed_backend_block_rows(backend, m);
    let mut block = vec![0.0_f32; block_rows * n];
    let mut aq_block = vec![0.0_f32; block_rows * kp];
    let mut gram = vec![0.0_f64; kp * kp];
    let mut gram_block = vec![0.0_f32; kp * kp];
    let _n_i = checked_cblas_dim_local(n, "n")?;
    let _kp_i = checked_cblas_dim_local(kp, "kp")?;
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));

    for row_start in (0..m).step_by(block_rows) {
        let row_end = (row_start + block_rows).min(m);
        let cur_rows = row_end - row_start;
        let cur_rows_i = checked_cblas_dim_local(cur_rows, "cur_rows")?;
        let cur_block = &mut block[..cur_rows * n];
        decode_train_masked_block_rows_f32(
            backend,
            row_freq_train,
            row_start,
            cur_rows,
            fold_id_keepout,
            folds,
            cv_seed,
            cur_block,
        )?;
        let cur_aq = &mut aq_block[..cur_rows * kp];
        row_major_block_mul_mat_f32(cur_block, cur_rows, n, q, kp, cur_aq, None);
        let _ = cur_rows_i;
        accum_gram_lower_f64_local(cur_aq, cur_rows, kp, &mut gram, &mut gram_block)?;
    }
    Ok(gram)
}

fn rsvd_stream_sample_packed_impl(
    backend: &PackedBedRsvd,
    k: usize,
    seed: u64,
    power: usize,
    tol: f32,
    backend_elapsed_s: f64,
) -> Result<(Vec<f32>, Vec<f32>, f64), String> {
    if k == 0 {
        return Err("k must be > 0".to_string());
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err("tol must be positive and finite".to_string());
    }

    let t_total = Instant::now();
    let n = backend.n_samples;
    let m = backend.n_snps;
    let row_freq = &backend.row_freq;
    let total_variance = backend.total_variance;
    let mut varsum: f64 = 0.0;
    for &freq in row_freq.iter() {
        varsum += 2.0_f64 * (freq as f64) * (1.0_f64 - freq as f64);
    }

    if m == 0 {
        return Err("no SNPs passed filtering in streaming RSVD".to_string());
    }
    if n == 0 {
        return Err("no samples available for streaming RSVD".to_string());
    }
    if !(varsum.is_finite() && varsum > 0.0) {
        return Err("invalid scaling denominator in streaming RSVD (varsum <= 0)".to_string());
    }
    check_admx_memory_limit("rsvd_stream_sample/start")?;

    let k_eff = k.min(n);
    let kp = admx_rsvd_kp(k_eff, m.max(1).min(n.max(1)));
    let block_rows = packed_backend_block_rows(backend, m);
    emit_rsvd_rss_debug(
        "stream_sample/backend_ready",
        &format!(
            "mode=packed_session n_samples={} n_snps={} kp={} row_freq={} varsum={:.6e} traceK={:.6e} total_sites={} active_row_idx={}",
            n,
            m,
            kp,
            format_f32_vec_bytes(row_freq.len()),
            varsum,
            total_variance,
            backend.n_total_sites,
            format_debug_bytes_local(
                (backend.active_row_idx.len() * std::mem::size_of::<usize>()) as u64
            ),
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/backend_ready",
        &format!("elapsed={}", fmt_stage_secs(backend_elapsed_s)),
    );
    emit_rsvd_rss_debug(
        "stream_sample/omega_ready",
        &format!(
            "omega_mode=random_block rows={} kp={} omega_scratch={}",
            block_rows,
            kp,
            format_f32_vec_bytes(block_rows * kp)
        ),
    );

    let t_stage = Instant::now();
    let mut y = compute_at_random_omega_packed(backend, kp, seed)?;
    check_admx_memory_limit("rsvd_stream_sample/init_proj")?;
    let init_proj_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/y_ready",
        &format!(
            "y_shape=({}, {}) y={}",
            n,
            kp,
            format_f32_vec_bytes(y.len())
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/init_proj",
        &format!("elapsed={}", fmt_stage_secs(init_proj_s)),
    );

    let t_stage = Instant::now();
    let (mut q, _, _) = thin_svd_from_tall(&y, n, kp)?;
    check_admx_memory_limit("rsvd_stream_sample/init_q")?;
    let init_q_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/q_ready",
        &format!(
            "q_shape=({}, {}) q={}",
            n,
            kp,
            format_f32_vec_bytes(q.len())
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/init_q",
        &format!("elapsed={}", fmt_stage_secs(init_q_s)),
    );

    let mut sk = vec![0.0_f32; kp];
    let mut alpha = 0.0_f32;
    let mut power_mul_s = 0.0_f64;
    let mut power_mul_decode_s = 0.0_f64;
    let mut power_mul_gemm_s = 0.0_f64;
    let mut power_qr_s = 0.0_f64;
    let mut power_iters_done = 0usize;
    let measure_kernel_timing = rsvd_time_debug_enabled();
    for it in 0..power {
        check_admx_memory_limit("rsvd_stream_sample/power_start")?;
        let t_power_mul = Instant::now();
        let mut kernel_timing = RsvdKernelTiming::default();
        y = compute_ata_omega_packed(
            backend,
            &q,
            kp,
            if measure_kernel_timing {
                Some(&mut kernel_timing)
            } else {
                None
            },
        )?;
        power_mul_s += t_power_mul.elapsed().as_secs_f64();
        power_mul_decode_s += kernel_timing.decode_s;
        power_mul_gemm_s += kernel_timing.gemm_s;
        subtract_scaled_inplace_f32(&mut y, alpha, &q);
        let t_power_qr = Instant::now();
        let (q_new, s_y, _) = thin_svd_from_tall(&y, n, kp)?;
        check_admx_memory_limit("rsvd_stream_sample/power_qr")?;
        power_qr_s += t_power_qr.elapsed().as_secs_f64();
        q = q_new;
        power_iters_done = it + 1;

        if it > 0 {
            let mut max_rel = 0.0_f32;
            for i in 0..k_eff {
                let sk_now = s_y[i] + alpha;
                let denom = sk_now.max(1e-12);
                let rel = ((sk_now - sk[i]).abs()) / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
                sk[i] = sk_now;
            }
            if max_rel < tol {
                break;
            }
        } else {
            for i in 0..kp {
                sk[i] = s_y[i] + alpha;
            }
        }

        let tail = s_y[kp - 1];
        if alpha < tail {
            alpha = 0.5 * (alpha + tail);
        }
    }

    let t_stage = Instant::now();
    let gram = compute_gram_aq_packed(backend, &q, kp)?;
    check_admx_memory_limit("rsvd_stream_sample/gram")?;
    let gram_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/gram_ready",
        &format!(
            "gram_shape=({}, {}) gram={}",
            kp,
            kp,
            format_debug_bytes_local(
                (gram.len().saturating_mul(std::mem::size_of::<f64>())) as u64
            ),
        ),
    );
    emit_rsvd_time_debug(
        "stream_sample/gram",
        &format!("elapsed={}", fmt_stage_secs(gram_s)),
    );

    let t_stage = Instant::now();
    let (s_all, v_small) = rsvd_right_singular_from_gram(&gram, kp)?;
    let eig_s = t_stage.elapsed().as_secs_f64();
    let mut eigvals = vec![0.0_f32; k_eff];
    let scale = varsum as f32;
    for i in 0..k_eff {
        eigvals[i] = (s_all[i] * s_all[i]) / scale;
    }

    let t_stage = Instant::now();
    let eigvecs_sample = rsvd_project_sample_eigvecs(&q, n, kp, &v_small, k_eff)?;
    let final_vec_s = t_stage.elapsed().as_secs_f64();
    emit_rsvd_rss_debug(
        "stream_sample/final_ready",
        &format!(
            "eigvals={} eigvecs_shape=({}, {}) eigvecs={}",
            format_f32_vec_bytes(eigvals.len()),
            n,
            k_eff,
            format_f32_vec_bytes(eigvecs_sample.len()),
        ),
    );
    let total_s = backend_elapsed_s + t_total.elapsed().as_secs_f64();
    emit_rsvd_time_debug(
        "stream_sample/summary",
        &format!(
            "total={} backend={} init_proj={} init_q={} power_mul={} power_mul_decode={} power_mul_gemm={} power_qr={} gram={} eig={} final_vec={} power_iters={} block_rows={}",
            fmt_stage_secs(total_s),
            fmt_stage_secs(backend_elapsed_s),
            fmt_stage_secs(init_proj_s),
            fmt_stage_secs(init_q_s),
            fmt_stage_secs(power_mul_s),
            fmt_stage_secs(power_mul_decode_s),
            fmt_stage_secs(power_mul_gemm_s),
            fmt_stage_secs(power_qr_s),
            fmt_stage_secs(gram_s),
            fmt_stage_secs(eig_s),
            fmt_stage_secs(final_vec_s),
            power_iters_done,
            block_rows,
        ),
    );
    Ok((eigvals, eigvecs_sample, total_variance))
}

fn rsvd_stream_sample_packed_fold_impl(
    backend: &PackedBedRsvd,
    row_freq_train: &[f32],
    k: usize,
    seed: u64,
    power: usize,
    tol: f32,
    fold_id_keepout: usize,
    folds: usize,
    cv_seed: u64,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    validate_cv_folds(folds)?;
    if fold_id_keepout >= folds {
        return Err(format!(
            "fold_id must be within [0, {}), got {fold_id_keepout}",
            folds
        ));
    }
    if k == 0 {
        return Err("k must be > 0".to_string());
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err("tol must be positive and finite".to_string());
    }

    let t_total = Instant::now();
    let n = backend.n_samples;
    let m = backend.n_snps;
    if row_freq_train.len() != m {
        return Err(format!(
            "row_freq_train length mismatch: got {}, expected {m}",
            row_freq_train.len()
        ));
    }

    let mut varsum = 0.0_f64;
    for &freq in row_freq_train.iter() {
        varsum += 2.0_f64 * (freq as f64) * (1.0_f64 - freq as f64);
    }
    if m == 0 {
        return Err("no SNPs passed filtering in streaming RSVD".to_string());
    }
    if n == 0 {
        return Err("no samples available for streaming RSVD".to_string());
    }
    if !(varsum.is_finite() && varsum > 0.0) {
        return Err(
            "invalid scaling denominator in masked streaming RSVD (varsum <= 0)".to_string(),
        );
    }
    check_admx_memory_limit("rsvd_stream_sample_fold/start")?;

    let k_eff = k.min(n);
    let kp = admx_rsvd_kp(k_eff, m.max(1).min(n.max(1)));
    let block_rows = packed_backend_block_rows(backend, m);
    let mut y = compute_at_random_omega_packed_fold_impl(
        backend,
        row_freq_train,
        kp,
        seed,
        fold_id_keepout,
        folds,
        cv_seed,
    )?;
    check_admx_memory_limit("rsvd_stream_sample_fold/init_proj")?;
    let init_proj_s = t_total.elapsed().as_secs_f64();
    let (mut q, _, _) = thin_svd_from_tall(&y, n, kp)?;
    check_admx_memory_limit("rsvd_stream_sample_fold/init_q")?;
    let init_q_s = t_total.elapsed().as_secs_f64() - init_proj_s;

    let mut sk = vec![0.0_f32; kp];
    let mut alpha = 0.0_f32;
    let mut power_mul_s = 0.0_f64;
    let mut power_mul_decode_s = 0.0_f64;
    let mut power_mul_gemm_s = 0.0_f64;
    let mut power_qr_s = 0.0_f64;
    let mut power_iters_done = 0usize;
    let measure_kernel_timing = rsvd_time_debug_enabled();
    for it in 0..power {
        check_admx_memory_limit("rsvd_stream_sample_fold/power_start")?;
        let t_power_mul = Instant::now();
        let mut kernel_timing = RsvdKernelTiming::default();
        y = compute_ata_omega_packed_fold_impl(
            backend,
            row_freq_train,
            &q,
            kp,
            fold_id_keepout,
            folds,
            cv_seed,
            if measure_kernel_timing {
                Some(&mut kernel_timing)
            } else {
                None
            },
        )?;
        power_mul_s += t_power_mul.elapsed().as_secs_f64();
        power_mul_decode_s += kernel_timing.decode_s;
        power_mul_gemm_s += kernel_timing.gemm_s;
        subtract_scaled_inplace_f32(&mut y, alpha, &q);
        let t_power_qr = Instant::now();
        let (q_new, s_y, _) = thin_svd_from_tall(&y, n, kp)?;
        check_admx_memory_limit("rsvd_stream_sample_fold/power_qr")?;
        power_qr_s += t_power_qr.elapsed().as_secs_f64();
        q = q_new;
        power_iters_done = it + 1;

        if it > 0 {
            let mut max_rel = 0.0_f32;
            for i in 0..k_eff {
                let sk_now = s_y[i] + alpha;
                let denom = sk_now.max(1e-12);
                let rel = ((sk_now - sk[i]).abs()) / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
                sk[i] = sk_now;
            }
            if max_rel < tol {
                break;
            }
        } else {
            for i in 0..kp {
                sk[i] = s_y[i] + alpha;
            }
        }

        let tail = s_y[kp - 1];
        if alpha < tail {
            alpha = 0.5 * (alpha + tail);
        }
    }

    let gram = compute_gram_aq_packed_fold_impl(
        backend,
        row_freq_train,
        &q,
        kp,
        fold_id_keepout,
        folds,
        cv_seed,
    )?;
    check_admx_memory_limit("rsvd_stream_sample_fold/gram")?;
    let gram_s =
        t_total.elapsed().as_secs_f64() - init_proj_s - init_q_s - power_mul_s - power_qr_s;
    let (s_all, v_small) = rsvd_right_singular_from_gram(&gram, kp)?;
    let mut eigvals = vec![0.0_f32; k_eff];
    let scale = varsum as f32;
    for i in 0..k_eff {
        eigvals[i] = (s_all[i] * s_all[i]) / scale;
    }

    let eigvecs_sample = rsvd_project_sample_eigvecs(&q, n, kp, &v_small, k_eff)?;

    let total_s = t_total.elapsed().as_secs_f64();
    emit_rsvd_time_debug(
        "stream_sample_fold/summary",
        &format!(
            "total={} init_proj={} init_q={} power_mul={} power_mul_decode={} power_mul_gemm={} power_qr={} gram={} power_iters={} block_rows={} fold={}/{}",
            fmt_stage_secs(total_s),
            fmt_stage_secs(init_proj_s),
            fmt_stage_secs(init_q_s),
            fmt_stage_secs(power_mul_s),
            fmt_stage_secs(power_mul_decode_s),
            fmt_stage_secs(power_mul_gemm_s),
            fmt_stage_secs(power_qr_s),
            fmt_stage_secs(gram_s.max(0.0)),
            power_iters_done,
            block_rows,
            fold_id_keepout + 1,
            folds,
        ),
    );
    Ok((eigvals, eigvecs_sample))
}

fn multiply_a_omega_packed_checked_impl(
    backend: &PackedBedRsvd,
    omega: &[f32],
    n: usize,
    kp: usize,
    row_freq_len: usize,
) -> Result<Vec<f32>, String> {
    if n == 0 || kp == 0 {
        return Err("omega must be a non-empty 2D matrix with shape (n_samples, k)".to_string());
    }
    if backend.n_samples != n {
        return Err(format!(
            "omega sample dimension mismatch: got {n}, expected {}",
            backend.n_samples
        ));
    }
    if backend.n_snps != row_freq_len {
        return Err(format!(
            "row_freq length mismatch: got {}, expected {}",
            row_freq_len, backend.n_snps
        ));
    }
    compute_a_omega_packed(backend, omega, kp)
}

fn loglikelihood_packed_checked_impl(
    backend: &PackedBedRsvd,
    p: &ArrayView2<'_, f32>,
    q: &ArrayView2<'_, f32>,
) -> Result<f64, String> {
    let (m, k) = p.dim();
    let (n, k2) = q.dim();
    if k != k2 {
        return Err(format!(
            "P/Q K mismatch: p=({}, {}), q=({}, {})",
            m, k, n, k2
        ));
    }
    if backend.n_snps != m || backend.n_samples != n {
        return Err(format!(
            "shape mismatch: expected P=({}, {}), Q=({}, {}), got P=({}, {}), Q=({}, {})",
            backend.n_snps, k, backend.n_samples, k, m, k, n, k2
        ));
    }
    let p_s = p
        .as_slice()
        .ok_or_else(|| "P is not contiguous".to_string())?;
    let q_s = q
        .as_slice()
        .ok_or_else(|| "Q is not contiguous".to_string())?;
    Ok(loglikelihood_packed_f32_impl(backend, p_s, q_s, k))
}

fn symmetric_pinv_f32(mat: &[f32], k: usize) -> Result<Vec<f32>, String> {
    if k == 0 || mat.len() != k * k {
        return Err(format!(
            "invalid square matrix for pseudo-inverse: len={}, k={k}",
            mat.len()
        ));
    }
    let mat64: Vec<f64> = mat.iter().map(|&v| v as f64).collect();
    let eig = SymmetricEigen::new(DMatrix::from_row_slice(k, k, &mat64));
    let max_eval = eig
        .eigenvalues
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let cutoff = (max_eval * 1e-12_f64).max(1e-12_f64);
    let mut inv = vec![0.0_f32; k * k];
    for ev_idx in 0..k {
        let eval = eig.eigenvalues[ev_idx];
        if !eval.is_finite() || eval.abs() <= cutoff {
            continue;
        }
        let inv_eval = 1.0_f64 / eval;
        for r in 0..k {
            let vr = eig.eigenvectors[(r, ev_idx)];
            for c in 0..k {
                inv[r * k + c] += (vr * eig.eigenvectors[(c, ev_idx)] * inv_eval) as f32;
            }
        }
    }
    Ok(inv)
}

fn gram_xtx_f32(x: &[f32], rows: usize, k: usize) -> Result<Vec<f32>, String> {
    if rows == 0 || k == 0 || x.len() != rows * k {
        return Err(format!(
            "invalid matrix for X^T X: len={}, expected={}",
            x.len(),
            rows.saturating_mul(k)
        ));
    }
    let rows_i = checked_cblas_dim_local(rows, "rows")?;
    let k_i = checked_cblas_dim_local(k, "k")?;
    let mut out = vec![0.0_f32; k * k];
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            k_i,
            k_i,
            rows_i,
            1.0_f32,
            x.as_ptr(),
            k_i,
            x.as_ptr(),
            k_i,
            0.0_f32,
            out.as_mut_ptr(),
            k_i,
        );
    }
    Ok(out)
}

fn matmul_rowmajor_f32(
    a: &[f32],
    rows_a: usize,
    cols_a: usize,
    b: &[f32],
    cols_b: usize,
) -> Result<Vec<f32>, String> {
    if rows_a == 0 || cols_a == 0 || cols_b == 0 {
        return Err("invalid empty matrix for matmul".to_string());
    }
    if a.len() != rows_a * cols_a || b.len() != cols_a * cols_b {
        return Err(format!(
            "matmul shape mismatch: A len={} expected={}, B len={} expected={}",
            a.len(),
            rows_a.saturating_mul(cols_a),
            b.len(),
            cols_a.saturating_mul(cols_b)
        ));
    }
    let rows_i = checked_cblas_dim_local(rows_a, "rows_a")?;
    let cols_a_i = checked_cblas_dim_local(cols_a, "cols_a")?;
    let cols_b_i = checked_cblas_dim_local(cols_b, "cols_b")?;
    let mut out = vec![0.0_f32; rows_a * cols_b];
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            rows_i,
            cols_b_i,
            cols_a_i,
            1.0_f32,
            a.as_ptr(),
            cols_a_i,
            b.as_ptr(),
            cols_b_i,
            0.0_f32,
            out.as_mut_ptr(),
            cols_b_i,
        );
    }
    Ok(out)
}

fn matmul_at_b_f32(a: &[f32], b: &[f32], rows: usize, k: usize) -> Result<Vec<f32>, String> {
    if rows == 0 || k == 0 || a.len() != rows * k || b.len() != rows * k {
        return Err(format!(
            "invalid matrices for A^T B: A len={}, B len={}, expected={}",
            a.len(),
            b.len(),
            rows.saturating_mul(k)
        ));
    }
    let rows_i = checked_cblas_dim_local(rows, "rows")?;
    let k_i = checked_cblas_dim_local(k, "k")?;
    let mut out = vec![0.0_f32; k * k];
    let _blas_guard = BlasThreadGuard::enter(rayon::current_num_threads().max(1));
    unsafe {
        cblas_sgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            k_i,
            k_i,
            rows_i,
            1.0_f32,
            a.as_ptr(),
            k_i,
            b.as_ptr(),
            k_i,
            0.0_f32,
            out.as_mut_ptr(),
            k_i,
        );
    }
    Ok(out)
}

fn right_multiply_pinv_f32(x: &[f32], rows: usize, k: usize, reg: f32) -> Result<Vec<f32>, String> {
    let mut gram = gram_xtx_f32(x, rows, k)?;
    for d in 0..k {
        gram[d * k + d] += reg;
    }
    let pinv = symmetric_pinv_f32(&gram, k)?;
    matmul_rowmajor_f32(x, rows, k, &pinv, k)
}

fn map_q_rows_inplace_f32(q: &mut [f32], n: usize, k: usize) {
    q.par_chunks_mut(k).take(n).for_each(|row| {
        let mut sum = 0.0_f32;
        for v in row.iter_mut() {
            *v = clip32(*v);
            sum += *v;
        }
        if sum <= 0.0_f32 || !sum.is_finite() {
            let v = 1.0_f32 / (k as f32).max(1.0_f32);
            row.fill(v);
        } else {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    });
}

fn map_p_rows_inplace_f32(p: &mut [f32]) {
    p.par_iter_mut().for_each(|v| {
        *v = clip32(*v);
    });
}

fn weighted_colsums_f32(x: &[f32], weights: &[f32], rows: usize, k: usize) -> Vec<f32> {
    (0..rows)
        .into_par_iter()
        .fold(
            || vec![0.0_f32; k],
            |mut acc, r| {
                let w = weights[r];
                let row = &x[r * k..(r + 1) * k];
                for j in 0..k {
                    acc[j] += row[j] * w;
                }
                acc
            },
        )
        .reduce(
            || vec![0.0_f32; k],
            |mut a, b| {
                for j in 0..k {
                    a[j] += b[j];
                }
                a
            },
        )
}

fn colsums_f32(x: &[f32], rows: usize, k: usize) -> Vec<f32> {
    (0..rows)
        .into_par_iter()
        .fold(
            || vec![0.0_f32; k],
            |mut acc, r| {
                let row = &x[r * k..(r + 1) * k];
                for j in 0..k {
                    acc[j] += row[j];
                }
                acc
            },
        )
        .reduce(
            || vec![0.0_f32; k],
            |mut a, b| {
                for j in 0..k {
                    a[j] += b[j];
                }
                a
            },
        )
}

fn rmse_slices_f32(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(format!(
            "RMSE shape mismatch: len(a)={}, len(b)={}",
            a.len(),
            b.len()
        ));
    }
    if a.is_empty() {
        return Ok(0.0_f32);
    }
    let acc: f64 = a
        .par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| {
            let d = (x - y) as f64;
            d * d
        })
        .sum();
    Ok(((acc / a.len() as f64) as f32).sqrt())
}

fn compute_q_from_p_projection_f32(
    z: &[f32],
    v: &[f32],
    row_freq: &[f32],
    i_mat: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    let zt_i = matmul_at_b_f32(z, i_mat, m, k)?;
    let mut q = matmul_rowmajor_f32(v, n, k, &zt_i, k)?;
    let fsum = weighted_colsums_f32(i_mat, row_freq, m, k);
    q.par_chunks_mut(k).for_each(|row| {
        for j in 0..k {
            row[j] = 0.5_f32 * row[j] + fsum[j];
        }
    });
    map_q_rows_inplace_f32(&mut q, n, k);
    Ok(q)
}

fn compute_p_from_q_projection_f32(
    z: &[f32],
    v: &[f32],
    row_freq: &[f32],
    i_mat: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    let vt_i = matmul_at_b_f32(v, i_mat, n, k)?;
    let mut p = matmul_rowmajor_f32(z, m, k, &vt_i, k)?;
    let isum = colsums_f32(i_mat, n, k);
    p.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        let f = row_freq[i];
        for j in 0..k {
            row[j] = clip32(0.5_f32 * row[j] + f * isum[j]);
        }
    });
    Ok(p)
}

fn adam_seed_init_rust(m: usize, n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut p = vec![0.0_f32; m * k];
    for v in p.iter_mut() {
        *v = clip32(rng.random::<f32>());
    }
    let mut q = vec![0.0_f32; n * k];
    for v in q.iter_mut() {
        *v = clip32(rng.random::<f32>());
    }
    map_q_rows_inplace_f32(&mut q, n, k);
    (p, q)
}

fn als_init_packed_session_impl(
    backend: &PackedBedRsvd,
    z: &[f32],
    v: &[f32],
    seed: u64,
    k: usize,
    max_iter: usize,
    tol: f32,
    reg: f32,
) -> Result<(Vec<f32>, Vec<f32>, f64, usize), String> {
    let m = backend.n_snps;
    let n = backend.n_samples;
    if m == 0 || n == 0 || k == 0 {
        return Err("invalid empty matrix for BED training session ALS".to_string());
    }
    if z.len() != m * k || v.len() != n * k || backend.row_freq.len() != m {
        return Err(format!(
            "ALS shape mismatch: z={}, expected {}; v={}, expected {}; row_freq={}, expected {m}",
            z.len(),
            m.saturating_mul(k),
            v.len(),
            n.saturating_mul(k),
            backend.row_freq.len()
        ));
    }
    let (mut p, _) = adam_seed_init_rust(m, n, k, seed);
    let mut i_mat = right_multiply_pinv_f32(&p, m, k, reg)?;
    let mut q = compute_q_from_p_projection_f32(z, v, &backend.row_freq, &i_mat, m, n, k)?;
    let mut q_prev = q.clone();
    let mut rmse_best = f32::INFINITY;
    let mut stall_counter = 0usize;
    let mut high_corr = false;
    let mut p_best: Option<Vec<f32>> = None;
    let mut q_best: Option<Vec<f32>> = None;
    let mut last_iter = 0usize;
    check_admx_memory_limit("als_init/start")?;

    for it in 0..max_iter {
        check_admx_memory_limit("als_init/iter_start")?;
        last_iter = it + 1;
        let i_q = right_multiply_pinv_f32(&q, n, k, reg)?;
        p = compute_p_from_q_projection_f32(z, v, &backend.row_freq, &i_q, m, n, k)?;
        map_p_rows_inplace_f32(&mut p);

        let g_p = gram_xtx_f32(&p, m, k)?;
        i_mat = {
            let mut g_reg = g_p.clone();
            for d in 0..k {
                g_reg[d * k + d] += reg;
            }
            let pinv = symmetric_pinv_f32(&g_reg, k)?;
            matmul_rowmajor_f32(&p, m, k, &pinv, k)?
        };
        q = compute_q_from_p_projection_f32(z, v, &backend.row_freq, &i_mat, m, n, k)?;

        let rmse_err = rmse_slices_f32(&q, &q_prev)?;
        if !high_corr {
            let mut max_corr = 0.0_f32;
            for a in 0..k {
                let va = g_p[a * k + a].max(1e-12_f32).sqrt();
                for b in 0..k {
                    if a == b {
                        continue;
                    }
                    let vb = g_p[b * k + b].max(1e-12_f32).sqrt();
                    let denom = (va * vb).max(1e-10_f32);
                    max_corr = max_corr.max((g_p[a * k + b] / denom).abs());
                }
            }
            if max_corr > 0.95_f32 {
                high_corr = true;
            }
        }

        if high_corr {
            if rmse_err < rmse_best {
                rmse_best = rmse_err;
                p_best = Some(p.clone());
                q_best = Some(q.clone());
                stall_counter = 0;
            } else {
                stall_counter += 1;
            }
            if stall_counter >= 20 {
                if let (Some(pb), Some(qb)) = (p_best.take(), q_best.take()) {
                    p = pb;
                    q = qb;
                }
                break;
            }
        }

        if rmse_err < tol {
            break;
        }
        q_prev.copy_from_slice(&q);
    }

    let init_ll = loglikelihood_packed_f32_impl(backend, &p, &q, k);
    check_admx_memory_limit("als_init/done")?;
    Ok((p, q, init_ll, last_iter))
}

fn fit_admx_bed_training_session_impl(
    backend: &PackedBedRsvd,
    k: usize,
    seed: u64,
    solver: &str,
    power: usize,
    tol: f32,
    max_als: usize,
    reg_als: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    max_iter: usize,
    check_every: usize,
    lr_decay: f32,
    min_lr: f32,
) -> Result<AdmxBedFitResult, String> {
    let m = backend.n_snps;
    let n = backend.n_samples;
    if m == 0 || n == 0 || k == 0 {
        return Err("invalid empty input for BED training session".to_string());
    }
    let solver_norm = solver.trim().to_ascii_lowercase();
    let solver_mode = if solver_norm == "adam" {
        "adam"
    } else {
        "adam-em"
    };
    emit_rsvd_rss_debug(
        "train_session/start",
        &format!(
            "n_snps={} n_samples={} k={} solver={}",
            m, n, k, solver_mode
        ),
    );

    let (mut p, mut q, init_ll, als_iter) = if solver_mode == "adam" {
        let (p0, q0) = adam_seed_init_rust(m, n, k, seed);
        (p0, q0, f64::NAN, 0usize)
    } else {
        let (_eigvals, v, _total_variance) =
            rsvd_stream_sample_packed_impl(backend, k, seed, power, tol, 0.0)?;
        check_admx_memory_limit("train_session/after_rsvd")?;
        let k_eff = if n == 0 { 0 } else { k.min(n) };
        if k_eff != k {
            return Err(format!(
                "K={k} exceeds sample-side RSVD rank {k_eff}; reduce K or use dense fallback"
            ));
        }
        let z = multiply_a_omega_packed_checked_impl(backend, &v, n, k, m)?;
        check_admx_memory_limit("train_session/after_z")?;
        als_init_packed_session_impl(backend, &z, &v, seed, k, max_als, tol, reg_als)?
    };

    emit_rsvd_rss_debug(
        "train_session/adam_start",
        &format!(
            "p={} q={} max_iter={} check_every={}",
            format_f32_vec_bytes(p.len()),
            format_f32_vec_bytes(q.len()),
            max_iter,
            check_every.max(1)
        ),
    );
    let (ll_final, adam_iter) = adam_optimize_packed_inplace_impl(
        backend,
        &mut p,
        &mut q,
        m,
        n,
        k,
        lr,
        beta1,
        beta2,
        epsilon,
        max_iter,
        check_every,
        lr_decay,
        min_lr,
    )?;
    emit_rsvd_rss_debug(
        "train_session/done",
        &format!("ll={:.3} adam_iter={}", ll_final, adam_iter),
    );
    Ok(AdmxBedFitResult {
        p,
        q,
        ll_final,
        adam_iter,
        init_ll,
        als_iter,
    })
}

fn loglikelihood_packed_train_fold_f32_impl(
    backend: &PackedBedRsvd,
    p: &[f32],
    q: &[f32],
    k: usize,
    fold_id_keepout: usize,
    folds: usize,
    cv_seed: u64,
) -> f64 {
    let n = backend.n_samples;
    let full_bytes = n / 4;
    let rem = n % 4;
    let byte_lut = packed_byte_lut();
    (0..backend.n_snps)
        .into_par_iter()
        .map(|i| {
            let row = packed_backend_row_slice(backend, i);
            let p_row = &p[i * k..(i + 1) * k];
            let flip = backend.row_flip[i];
            let row_offset = (i as u64) * (n as u64);
            let mut part = 0.0_f64;
            let mut col = 0usize;
            for &packed in row.iter().take(full_bytes) {
                let codes = &byte_lut.code4[packed as usize];
                for &code in codes.iter() {
                    if let Some(gv_f32) = packed_code_minor_allele_g(code, flip) {
                        if cv_fold_id(row_offset + (col as u64), folds, cv_seed) != fold_id_keepout
                        {
                            let q_row = &q[col * k..(col + 1) * k];
                            let mut rec = 0.0_f32;
                            for kk in 0..k {
                                rec += p_row[kk] * q_row[kk];
                            }
                            let rec = rec.clamp(1e-6, 1.0 - 1e-6) as f64;
                            let gv = gv_f32 as f64;
                            part += gv * rec.ln() + (2.0 - gv) * (1.0 - rec).ln();
                        }
                    }
                    col += 1;
                }
            }
            if rem > 0 {
                let codes = &byte_lut.code4[row[full_bytes] as usize];
                for &code in codes.iter().take(rem) {
                    if let Some(gv_f32) = packed_code_minor_allele_g(code, flip) {
                        if cv_fold_id(row_offset + (col as u64), folds, cv_seed) != fold_id_keepout
                        {
                            let q_row = &q[col * k..(col + 1) * k];
                            let mut rec = 0.0_f32;
                            for kk in 0..k {
                                rec += p_row[kk] * q_row[kk];
                            }
                            let rec = rec.clamp(1e-6, 1.0 - 1e-6) as f64;
                            let gv = gv_f32 as f64;
                            part += gv * rec.ln() + (2.0 - gv) * (1.0 - rec).ln();
                        }
                    }
                    col += 1;
                }
            }
            part
        })
        .sum()
}

fn holdout_nll_packed_fold_f32_impl(
    backend: &PackedBedRsvd,
    p: &[f32],
    q: &[f32],
    k: usize,
    fold_id_keepout: usize,
    folds: usize,
    cv_seed: u64,
) -> (f64, usize) {
    let n = backend.n_samples;
    let full_bytes = n / 4;
    let rem = n % 4;
    let byte_lut = packed_byte_lut();
    let ln2 = 2.0_f64.ln();
    (0..backend.n_snps)
        .into_par_iter()
        .map(|i| {
            let row = packed_backend_row_slice(backend, i);
            let p_row = &p[i * k..(i + 1) * k];
            let flip = backend.row_flip[i];
            let row_offset = (i as u64) * (n as u64);
            let mut ll_sum = 0.0_f64;
            let mut n_sum = 0usize;
            let mut col = 0usize;
            for &packed in row.iter().take(full_bytes) {
                let codes = &byte_lut.code4[packed as usize];
                for &code in codes.iter() {
                    if let Some(gv_f32) = packed_code_minor_allele_g(code, flip) {
                        if cv_fold_id(row_offset + (col as u64), folds, cv_seed) == fold_id_keepout
                        {
                            let q_row = &q[col * k..(col + 1) * k];
                            let mut rec = 0.0_f32;
                            for kk in 0..k {
                                rec += p_row[kk] * q_row[kk];
                            }
                            let rec = rec.clamp(1e-8, 1.0 - 1e-8) as f64;
                            let gv = gv_f32 as f64;
                            let log_comb = if (gv - 1.0_f64).abs() < f64::EPSILON {
                                ln2
                            } else {
                                0.0_f64
                            };
                            ll_sum += log_comb + gv * rec.ln() + (2.0 - gv) * (1.0 - rec).ln();
                            n_sum += 1;
                        }
                    }
                    col += 1;
                }
            }
            if rem > 0 {
                let codes = &byte_lut.code4[row[full_bytes] as usize];
                for &code in codes.iter().take(rem) {
                    if let Some(gv_f32) = packed_code_minor_allele_g(code, flip) {
                        if cv_fold_id(row_offset + (col as u64), folds, cv_seed) == fold_id_keepout
                        {
                            let q_row = &q[col * k..(col + 1) * k];
                            let mut rec = 0.0_f32;
                            for kk in 0..k {
                                rec += p_row[kk] * q_row[kk];
                            }
                            let rec = rec.clamp(1e-8, 1.0 - 1e-8) as f64;
                            let gv = gv_f32 as f64;
                            let log_comb = if (gv - 1.0_f64).abs() < f64::EPSILON {
                                ln2
                            } else {
                                0.0_f64
                            };
                            ll_sum += log_comb + gv * rec.ln() + (2.0 - gv) * (1.0 - rec).ln();
                            n_sum += 1;
                        }
                    }
                    col += 1;
                }
            }
            (ll_sum, n_sum)
        })
        .reduce(
            || (0.0_f64, 0usize),
            |(ll_a, n_a), (ll_b, n_b)| (ll_a + ll_b, n_a + n_b),
        )
}

#[allow(dead_code)]
fn em_step_packed_fold_f32_impl(
    backend: &PackedBedRsvd,
    p_src: &[f32],
    q_src: &[f32],
    p_em_slice: &mut [f32],
    q_em_slice: &mut [f32],
    q_bat: &mut [f32],
    t_acc: &mut [f32],
    fold_id_keepout: usize,
    folds: usize,
    cv_seed: u64,
) -> Result<(), String> {
    let m = backend.n_snps;
    let n = backend.n_samples;
    if m == 0 || n == 0 {
        return Ok(());
    }
    if p_src.len() % m != 0 {
        return Err("invalid P shape in masked packed EM helper".to_string());
    }
    let k = p_src.len() / m;
    if q_src.len() != n * k {
        return Err("invalid Q shape in masked packed EM helper".to_string());
    }
    if p_em_slice.len() != m * k || q_em_slice.len() != n * k {
        return Err("invalid masked packed EM output buffer size".to_string());
    }
    if q_bat.len() != n || t_acc.len() != n * k {
        return Err("invalid masked packed EM scratch buffer size".to_string());
    }

    let full_bytes = n / 4;
    let rem = n % 4;
    let byte_lut = packed_byte_lut();
    let blocks = rayon::current_num_threads().max(1);
    let rows_per_block = (m + blocks - 1) / blocks;
    let chunk_elems = (rows_per_block * k).max(k);
    let (qb_sum, ta_sum, _, _) = p_em_slice
        .par_chunks_mut(chunk_elems)
        .enumerate()
        .fold(
            || {
                (
                    vec![0.0_f32; n],
                    vec![0.0_f32; n * k],
                    vec![0.0_f32; k],
                    vec![0.0_f32; k],
                )
            },
            |(mut qb, mut ta, mut a, mut b), (chunk_idx, em_chunk)| {
                let row_start = chunk_idx * rows_per_block;
                let row_count = em_chunk.len() / k;
                for r in 0..row_count {
                    let i = row_start + r;
                    if i >= m {
                        break;
                    }
                    let row = packed_backend_row_slice(backend, i);
                    let flip = backend.row_flip[i];
                    let row_offset = (i as u64) * (n as u64);
                    let p_row = &p_src[i * k..(i + 1) * k];
                    let out_row = &mut em_chunk[r * k..(r + 1) * k];
                    a.fill(0.0_f32);
                    b.fill(0.0_f32);
                    let mut col = 0usize;
                    for &packed in row.iter().take(full_bytes) {
                        let codes = &byte_lut.code4[packed as usize];
                        for &code in codes.iter() {
                            if let Some(g_f) = packed_code_minor_allele_g(code, flip) {
                                if cv_fold_id(row_offset + (col as u64), folds, cv_seed)
                                    != fold_id_keepout
                                {
                                    qb[col] += 2.0_f32;
                                    let q_row = &q_src[col * k..(col + 1) * k];
                                    let mut rec = 0.0_f32;
                                    for kk in 0..k {
                                        rec += p_row[kk] * q_row[kk];
                                    }
                                    rec = rec.clamp(1e-6, 1.0 - 1e-6);
                                    let aa = g_f / rec;
                                    let bb = (2.0_f32 - g_f) / (1.0_f32 - rec);
                                    let t_row = &mut ta[col * k..(col + 1) * k];
                                    for kk in 0..k {
                                        let qv = q_row[kk];
                                        a[kk] += qv * aa;
                                        b[kk] += qv * bb;
                                        t_row[kk] += p_row[kk] * (aa - bb) + bb;
                                    }
                                }
                            }
                            col += 1;
                        }
                    }
                    if rem > 0 {
                        let codes = &byte_lut.code4[row[full_bytes] as usize];
                        for &code in codes.iter().take(rem) {
                            if let Some(g_f) = packed_code_minor_allele_g(code, flip) {
                                if cv_fold_id(row_offset + (col as u64), folds, cv_seed)
                                    != fold_id_keepout
                                {
                                    qb[col] += 2.0_f32;
                                    let q_row = &q_src[col * k..(col + 1) * k];
                                    let mut rec = 0.0_f32;
                                    for kk in 0..k {
                                        rec += p_row[kk] * q_row[kk];
                                    }
                                    rec = rec.clamp(1e-6, 1.0 - 1e-6);
                                    let aa = g_f / rec;
                                    let bb = (2.0_f32 - g_f) / (1.0_f32 - rec);
                                    let t_row = &mut ta[col * k..(col + 1) * k];
                                    for kk in 0..k {
                                        let qv = q_row[kk];
                                        a[kk] += qv * aa;
                                        b[kk] += qv * bb;
                                        t_row[kk] += p_row[kk] * (aa - bb) + bb;
                                    }
                                }
                            }
                            col += 1;
                        }
                    }
                    for kk in 0..k {
                        let denom = p_row[kk] * (a[kk] - b[kk]) + b[kk];
                        let v = if denom.abs() < 1e-8 {
                            p_row[kk]
                        } else {
                            (a[kk] * p_row[kk]) / denom
                        };
                        out_row[kk] = clip32(v);
                    }
                }
                madvise_backend_active_rows(backend, row_start, row_count);
                (qb, ta, a, b)
            },
        )
        .reduce(
            || {
                (
                    vec![0.0_f32; n],
                    vec![0.0_f32; n * k],
                    vec![0.0_f32; k],
                    vec![0.0_f32; k],
                )
            },
            |(mut qb1, mut ta1, a1, b1), (qb2, ta2, _, _)| {
                for col in 0..n {
                    qb1[col] += qb2[col];
                }
                for idx in 0..(n * k) {
                    ta1[idx] += ta2[idx];
                }
                (qb1, ta1, a1, b1)
            },
        );
    q_bat.copy_from_slice(&qb_sum);
    t_acc.copy_from_slice(&ta_sum);

    q_em_slice
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(col, out_row)| {
            let q_row = &q_src[col * k..(col + 1) * k];
            if q_bat[col] <= 0.0_f32 {
                for kk in 0..k {
                    out_row[kk] = clip32(q_row[kk]);
                }
            } else {
                let inv = 1.0_f32 / q_bat[col];
                let t_row = &t_acc[col * k..(col + 1) * k];
                for kk in 0..k {
                    out_row[kk] = clip32(q_row[kk] * t_row[kk] * inv);
                }
            }
            let sum = out_row.iter().sum::<f32>();
            if sum <= 0.0_f32 || !sum.is_finite() {
                let v = 1.0_f32 / (k as f32).max(1.0_f32);
                out_row.fill(v);
            } else {
                for kk in 0..k {
                    out_row[kk] /= sum;
                }
            }
        });
    Ok(())
}

fn adam_optimize_packed_fold_inplace_impl(
    backend: &PackedBedRsvd,
    p: &mut [f32],
    q: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    max_iter: usize,
    check_every: usize,
    lr_decay: f32,
    min_lr: f32,
    fold_id_keepout: usize,
    folds: usize,
    cv_seed: u64,
) -> Result<(f64, usize), String> {
    validate_cv_folds(folds)?;
    if m != backend.n_snps || n != backend.n_samples {
        return Err(format!(
            "shape mismatch: expected P/Q rows=({}, {}), got ({}, {})",
            backend.n_snps, backend.n_samples, m, n
        ));
    }
    if m == 0 || n == 0 || k == 0 {
        return Err("invalid empty input matrix for fold-masked BED ADAM optimization".to_string());
    }
    if p.len() != m * k || q.len() != n * k {
        return Err(
            "invalid contiguous slice length for fold-masked BED ADAM optimization".to_string(),
        );
    }

    let mut p_best = vec![0.0_f32; m * k];
    let mut q_best = vec![0.0_f32; n * k];
    let mut ll_best = f64::NEG_INFINITY;
    let mut no_improve = 0usize;
    let mut lr_cur = lr;
    let mut last_iter = 0usize;
    let mut beta1_pow = 1.0_f32;
    let mut beta2_pow = 1.0_f32;
    let mut have_best = false;

    let mut m_p = vec![0.0_f32; m * k];
    let mut v_p = vec![0.0_f32; m * k];
    let mut m_q = vec![0.0_f32; n * k];
    let mut v_q = vec![0.0_f32; n * k];
    let mut q_em = vec![0.0_f32; n * k];
    let check_every = check_every.max(1);
    let full_bytes = n / 4;
    let rem = n % 4;
    let byte_lut = packed_byte_lut();
    let blocks = rayon::current_num_threads().max(1);
    let rows_per_block = (m + blocks - 1) / blocks;
    let chunk_elems = (rows_per_block * k).max(k);
    check_admx_memory_limit("adam_fold/scratch_ready")?;

    for it in 0..max_iter {
        check_admx_memory_limit("adam_fold/iter_start")?;
        last_iter = it + 1;
        let one_b1 = 1.0_f32 - beta1;
        let one_b2 = 1.0_f32 - beta2;
        beta1_pow *= beta1;
        beta2_pow *= beta2;
        let m_scale = if (1.0_f32 - beta1_pow).abs() < 1e-8 {
            1.0_f32
        } else {
            1.0_f32 / (1.0_f32 - beta1_pow)
        };
        let v_scale = if (1.0_f32 - beta2_pow).abs() < 1e-8 {
            1.0_f32
        } else {
            1.0_f32 / (1.0_f32 - beta2_pow)
        };

        let (qb_sum, ta_sum, _, _) = p
            .par_chunks_mut(chunk_elems)
            .zip(m_p.par_chunks_mut(chunk_elems))
            .zip(v_p.par_chunks_mut(chunk_elems))
            .enumerate()
            .fold(
                || {
                    (
                        vec![0.0_f32; n],
                        vec![0.0_f32; n * k],
                        vec![0.0_f32; k],
                        vec![0.0_f32; k],
                    )
                },
                |(mut qb, mut ta, mut a, mut b), (chunk_idx, ((p_chunk, m_chunk), v_chunk))| {
                    let row_start = chunk_idx * rows_per_block;
                    let row_count = p_chunk.len() / k;
                    for r in 0..row_count {
                        let i = row_start + r;
                        if i >= m {
                            break;
                        }
                        let row = packed_backend_row_slice(backend, i);
                        let flip = backend.row_flip[i];
                        let row_offset = (i as u64) * (n as u64);
                        let p_row = &mut p_chunk[r * k..(r + 1) * k];
                        let m_row = &mut m_chunk[r * k..(r + 1) * k];
                        let v_row = &mut v_chunk[r * k..(r + 1) * k];
                        a.fill(0.0_f32);
                        b.fill(0.0_f32);
                        let mut col = 0usize;
                        let geno_lut = if flip {
                            [2.0_f32, -1.0_f32, 1.0_f32, 0.0_f32]
                        } else {
                            [0.0_f32, -1.0_f32, 1.0_f32, 2.0_f32]
                        };
                        for &packed in row.iter().take(full_bytes) {
                            let codes = &byte_lut.code4[packed as usize];
                            for &code in codes.iter() {
                                let g_f = geno_lut[code as usize];
                                if g_f >= 0.0_f32 {
                                    if cv_fold_id(row_offset + (col as u64), folds, cv_seed)
                                        != fold_id_keepout
                                    {
                                        qb[col] += 2.0_f32;
                                        let q_row = &q[col * k..(col + 1) * k];
                                        let mut rec = 0.0_f32;
                                        for kk in 0..k {
                                            rec += p_row[kk] * q_row[kk];
                                        }
                                        rec = rec.clamp(1e-6, 1.0 - 1e-6);
                                        let aa = g_f / rec;
                                        let bb = (2.0_f32 - g_f) / (1.0_f32 - rec);
                                        let t_row = &mut ta[col * k..(col + 1) * k];
                                        for kk in 0..k {
                                            let qv = q_row[kk];
                                            a[kk] += qv * aa;
                                            b[kk] += qv * bb;
                                            t_row[kk] += p_row[kk] * (aa - bb) + bb;
                                        }
                                    }
                                }
                                col += 1;
                            }
                        }
                        if rem > 0 {
                            let codes = &byte_lut.code4[row[full_bytes] as usize];
                            for &code in codes.iter().take(rem) {
                                let g_f = geno_lut[code as usize];
                                if g_f >= 0.0_f32 {
                                    if cv_fold_id(row_offset + (col as u64), folds, cv_seed)
                                        != fold_id_keepout
                                    {
                                        qb[col] += 2.0_f32;
                                        let q_row = &q[col * k..(col + 1) * k];
                                        let mut rec = 0.0_f32;
                                        for kk in 0..k {
                                            rec += p_row[kk] * q_row[kk];
                                        }
                                        rec = rec.clamp(1e-6, 1.0 - 1e-6);
                                        let aa = g_f / rec;
                                        let bb = (2.0_f32 - g_f) / (1.0_f32 - rec);
                                        let t_row = &mut ta[col * k..(col + 1) * k];
                                        for kk in 0..k {
                                            let qv = q_row[kk];
                                            a[kk] += qv * aa;
                                            b[kk] += qv * bb;
                                            t_row[kk] += p_row[kk] * (aa - bb) + bb;
                                        }
                                    }
                                }
                                col += 1;
                            }
                        }
                        for kk in 0..k {
                            let denom = p_row[kk] * (a[kk] - b[kk]) + b[kk];
                            let p_em = if denom.abs() < 1e-8_f32 {
                                p_row[kk]
                            } else {
                                (a[kk] * p_row[kk]) / denom
                            };
                            let delta = p_em - p_row[kk];
                            let mcur = beta1 * m_row[kk] + one_b1 * delta;
                            let vcur = beta2 * v_row[kk] + one_b2 * delta * delta;
                            let m_hat = mcur * m_scale;
                            let v_hat = vcur * v_scale;
                            let step = lr_cur * m_hat / (v_hat.sqrt() + epsilon);
                            p_row[kk] = clip32(p_row[kk] + step);
                            m_row[kk] = mcur;
                            v_row[kk] = vcur;
                        }
                    }
                    madvise_backend_active_rows(backend, row_start, row_count);
                    (qb, ta, a, b)
                },
            )
            .reduce(
                || {
                    (
                        vec![0.0_f32; n],
                        vec![0.0_f32; n * k],
                        vec![0.0_f32; k],
                        vec![0.0_f32; k],
                    )
                },
                |(mut qb1, mut ta1, a1, b1), (qb2, ta2, _, _)| {
                    for (dst, src) in qb1.iter_mut().zip(qb2.iter()) {
                        *dst += *src;
                    }
                    for (dst, src) in ta1.iter_mut().zip(ta2.iter()) {
                        *dst += *src;
                    }
                    (qb1, ta1, a1, b1)
                },
            );
        q_em.par_chunks_mut(k)
            .enumerate()
            .for_each(|(col, out_row)| {
                let q_row = &q[col * k..(col + 1) * k];
                if qb_sum[col] <= 0.0_f32 {
                    for kk in 0..k {
                        out_row[kk] = clip32(q_row[kk]);
                    }
                } else {
                    let inv = 1.0_f32 / qb_sum[col];
                    let t_row = &ta_sum[col * k..(col + 1) * k];
                    for kk in 0..k {
                        out_row[kk] = clip32(q_row[kk] * t_row[kk] * inv);
                    }
                }
                let sum = out_row.iter().sum::<f32>();
                if sum <= 0.0_f32 || !sum.is_finite() {
                    let v = 1.0_f32 / (k as f32).max(1.0_f32);
                    out_row.fill(v);
                } else {
                    for kk in 0..k {
                        out_row[kk] /= sum;
                    }
                }
            });

        q.par_chunks_mut(k)
            .zip(m_q.par_chunks_mut(k))
            .zip(v_q.par_chunks_mut(k))
            .enumerate()
            .for_each(|(i, ((q_row, m_row), v_row))| {
                let base = i * k;
                let mut sum = 0.0_f32;
                for j in 0..k {
                    let idx = base + j;
                    let delta = q_em[idx] - q_row[j];
                    let mcur = beta1 * m_row[j] + one_b1 * delta;
                    let vcur = beta2 * v_row[j] + one_b2 * delta * delta;
                    let m_hat = mcur * m_scale;
                    let v_hat = vcur * v_scale;
                    let step = lr_cur * m_hat / (v_hat.sqrt() + epsilon);
                    let qv = clip32(q_row[j] + step);
                    q_row[j] = qv;
                    m_row[j] = mcur;
                    v_row[j] = vcur;
                    sum += qv;
                }
                if sum <= 0.0_f32 || !sum.is_finite() {
                    let v = 1.0_f32 / (k as f32).max(1.0_f32);
                    q_row.fill(v);
                } else {
                    for j in 0..k {
                        q_row[j] /= sum;
                    }
                }
            });

        if last_iter % check_every != 0 {
            continue;
        }

        let ll_cur = loglikelihood_packed_train_fold_f32_impl(
            backend,
            &p,
            &q,
            k,
            fold_id_keepout,
            folds,
            cv_seed,
        );
        check_admx_memory_limit("adam_fold/loglik")?;
        if (ll_cur - ll_best).abs() < 0.1_f64 {
            break;
        }

        if ll_cur > ll_best {
            ll_best = ll_cur;
            p_best.copy_from_slice(&p);
            q_best.copy_from_slice(&q);
            have_best = true;
            no_improve = 0;
        } else {
            no_improve += 1;
            lr_cur = (lr_cur * lr_decay).max(min_lr);
            if no_improve >= 2 {
                break;
            }
        }
    }

    if !ll_best.is_finite() {
        ll_best = loglikelihood_packed_train_fold_f32_impl(
            backend,
            &p,
            &q,
            k,
            fold_id_keepout,
            folds,
            cv_seed,
        );
        p_best.copy_from_slice(&p);
        q_best.copy_from_slice(&q);
        have_best = true;
    }
    if have_best {
        p.copy_from_slice(&p_best);
        q.copy_from_slice(&q_best);
    }
    Ok((ll_best, last_iter))
}

fn adam_optimize_packed_fold_impl(
    backend: &PackedBedRsvd,
    p0: &ArrayView2<'_, f32>,
    q0: &ArrayView2<'_, f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    max_iter: usize,
    check_every: usize,
    lr_decay: f32,
    min_lr: f32,
    fold_id_keepout: usize,
    folds: usize,
    cv_seed: u64,
) -> Result<(Vec<f32>, Vec<f32>, f64, usize), String> {
    let (m, k) = p0.dim();
    let (n, k2) = q0.dim();
    if k != k2 {
        return Err(format!(
            "shape mismatch: expected shared K, got P0=({}, {}), Q0=({}, {})",
            m, k, n, k2
        ));
    }
    let p0s = p0
        .as_slice()
        .ok_or_else(|| "P0 not contiguous".to_string())?;
    let q0s = q0
        .as_slice()
        .ok_or_else(|| "Q0 not contiguous".to_string())?;

    let mut p = p0s.to_vec();
    let mut q = q0s.to_vec();
    let (ll_best, last_iter) = adam_optimize_packed_fold_inplace_impl(
        backend,
        &mut p,
        &mut q,
        m,
        n,
        k,
        lr,
        beta1,
        beta2,
        epsilon,
        max_iter,
        check_every,
        lr_decay,
        min_lr,
        fold_id_keepout,
        folds,
        cv_seed,
    )?;
    Ok((p, q, ll_best, last_iter))
}

fn em_step_inplace_f32_tiled_impl(
    g: &ArrayView2<'_, u8>,
    p_src: &[f32],
    q_src: &[f32],
    p_em_slice: &mut [f32],
    q_em_slice: &mut [f32],
    q_bat: &mut [f32],
    t_acc: &mut [f32],
    tile_cols: usize,
) -> Result<(), String> {
    let (m, n) = g.dim();
    if m == 0 || n == 0 {
        return Ok(());
    }
    if p_src.len() % m != 0 {
        return Err("invalid P shape in EM helper".to_string());
    }
    let k = p_src.len() / m;
    if q_src.len() != n * k {
        return Err("invalid Q shape in EM helper".to_string());
    }
    if p_em_slice.len() != m * k || q_em_slice.len() != n * k {
        return Err("invalid EM output buffer size".to_string());
    }
    if q_bat.len() != n || t_acc.len() != n * k {
        return Err("invalid EM scratch buffer size".to_string());
    }

    // Fast path (tile >= n): fuse P-step and Q accumulators in one pass (same idea as ADAMIXTURE-main),
    // so each reconstruction is computed only once.
    if tile_cols >= n {
        let blocks = rayon::current_num_threads().max(1);
        let rows_per_block = (m + blocks - 1) / blocks;
        let chunk_elems = (rows_per_block * k).max(k);
        let (qb_sum, ta_sum, _, _) = p_em_slice
            .par_chunks_mut(chunk_elems)
            .enumerate()
            .fold(
                || {
                    (
                        vec![0.0_f32; n],
                        vec![0.0_f32; n * k],
                        vec![0.0_f32; k],
                        vec![0.0_f32; k],
                    )
                },
                |(mut qb, mut ta, mut a, mut b), (chunk_idx, em_chunk)| {
                    let row_start = chunk_idx * rows_per_block;
                    let row_count = em_chunk.len() / k;
                    for r in 0..row_count {
                        let i = row_start + r;
                        if i >= m {
                            break;
                        }
                        let p_row = &p_src[i * k..(i + 1) * k];
                        let out_row = &mut em_chunk[r * k..(r + 1) * k];
                        a.fill(0.0);
                        b.fill(0.0);
                        for col in 0..n {
                            let gv = g[(i, col)];
                            if gv == 3 {
                                continue;
                            }
                            qb[col] += 2.0;
                            let q_row = &q_src[col * k..(col + 1) * k];
                            let mut rec = 0.0_f32;
                            for kk in 0..k {
                                rec += p_row[kk] * q_row[kk];
                            }
                            rec = rec.clamp(1e-6, 1.0 - 1e-6);
                            let g_f = gv as f32;
                            let aa = g_f / rec;
                            let bb = (2.0 - g_f) / (1.0 - rec);
                            let t_row = &mut ta[col * k..(col + 1) * k];
                            for kk in 0..k {
                                let qv = q_row[kk];
                                a[kk] += qv * aa;
                                b[kk] += qv * bb;
                                t_row[kk] += p_row[kk] * (aa - bb) + bb;
                            }
                        }
                        for kk in 0..k {
                            let denom = p_row[kk] * (a[kk] - b[kk]) + b[kk];
                            let v = if denom.abs() < 1e-8 {
                                p_row[kk]
                            } else {
                                (a[kk] * p_row[kk]) / denom
                            };
                            out_row[kk] = clip32(v);
                        }
                    }
                    (qb, ta, a, b)
                },
            )
            .reduce(
                || {
                    (
                        vec![0.0_f32; n],
                        vec![0.0_f32; n * k],
                        vec![0.0_f32; k],
                        vec![0.0_f32; k],
                    )
                },
                |(mut qb1, mut ta1, a1, b1), (qb2, ta2, _, _)| {
                    for col in 0..n {
                        qb1[col] += qb2[col];
                    }
                    for idx in 0..(n * k) {
                        ta1[idx] += ta2[idx];
                    }
                    (qb1, ta1, a1, b1)
                },
            );
        q_bat.copy_from_slice(&qb_sum);
        t_acc.copy_from_slice(&ta_sum);
    } else {
        p_em_slice
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(i, out_row)| {
                let p_row = &p_src[i * k..(i + 1) * k];
                let mut a = vec![0.0_f32; k];
                let mut b = vec![0.0_f32; k];
                for col in 0..n {
                    let gv = g[(i, col)];
                    if gv == 3 {
                        continue;
                    }
                    let q_row = &q_src[col * k..(col + 1) * k];
                    let mut rec = 0.0_f32;
                    for kk in 0..k {
                        rec += p_row[kk] * q_row[kk];
                    }
                    rec = rec.clamp(1e-6, 1.0 - 1e-6);
                    let g_f = gv as f32;
                    let aa = g_f / rec;
                    let bb = (2.0 - g_f) / (1.0 - rec);
                    for kk in 0..k {
                        a[kk] += q_row[kk] * aa;
                        b[kk] += q_row[kk] * bb;
                    }
                }
                for kk in 0..k {
                    let denom = p_row[kk] * (a[kk] - b[kk]) + b[kk];
                    let v = if denom.abs() < 1e-8 {
                        p_row[kk]
                    } else {
                        (a[kk] * p_row[kk]) / denom
                    };
                    out_row[kk] = clip32(v);
                }
            });

        q_bat.fill(0.0);
        t_acc.fill(0.0);
        let tile = tile_cols.max(1).min(n);
        for block_start in (0..n).step_by(tile) {
            let block_len = (n - block_start).min(tile);
            let (q_bat_blk, t_blk) = (0..m)
                .into_par_iter()
                .fold(
                    || (vec![0.0_f32; block_len], vec![0.0_f32; block_len * k]),
                    |(mut qb, mut tb), i| {
                        let p_row = &p_src[i * k..(i + 1) * k];
                        for off in 0..block_len {
                            let col = block_start + off;
                            let gv = g[(i, col)];
                            if gv == 3 {
                                continue;
                            }
                            qb[off] += 2.0;
                            let q_row = &q_src[col * k..(col + 1) * k];
                            let mut rec = 0.0_f32;
                            for kk in 0..k {
                                rec += p_row[kk] * q_row[kk];
                            }
                            rec = rec.clamp(1e-6, 1.0 - 1e-6);
                            let g_f = gv as f32;
                            let aa = g_f / rec;
                            let bb = (2.0 - g_f) / (1.0 - rec);
                            let t_row = &mut tb[off * k..(off + 1) * k];
                            for kk in 0..k {
                                t_row[kk] += p_row[kk] * (aa - bb) + bb;
                            }
                        }
                        (qb, tb)
                    },
                )
                .reduce(
                    || (vec![0.0_f32; block_len], vec![0.0_f32; block_len * k]),
                    |(mut qb1, mut tb1), (qb2, tb2)| {
                        for off in 0..block_len {
                            qb1[off] += qb2[off];
                        }
                        for idx in 0..(block_len * k) {
                            tb1[idx] += tb2[idx];
                        }
                        (qb1, tb1)
                    },
                );

            q_bat[block_start..block_start + block_len].copy_from_slice(&q_bat_blk);
            t_acc[block_start * k..(block_start + block_len) * k].copy_from_slice(&t_blk);
        }
    }

    q_em_slice
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(col, out_row)| {
            let q_row = &q_src[col * k..(col + 1) * k];
            if q_bat[col] <= 0.0 {
                for kk in 0..k {
                    out_row[kk] = clip32(q_row[kk]);
                }
            } else {
                let inv = 1.0 / q_bat[col];
                let t_row = &t_acc[col * k..(col + 1) * k];
                for kk in 0..k {
                    out_row[kk] = clip32(q_row[kk] * t_row[kk] * inv);
                }
            }
            let sum = out_row.iter().sum::<f32>();
            if sum <= 0.0 || !sum.is_finite() {
                let v = 1.0 / (k as f32).max(1.0);
                out_row.fill(v);
            } else {
                for kk in 0..k {
                    out_row[kk] /= sum;
                }
            }
        });
    Ok(())
}

#[allow(dead_code)]
fn em_step_packed_f32_impl(
    backend: &PackedBedRsvd,
    p_src: &[f32],
    q_src: &[f32],
    p_em_slice: &mut [f32],
    q_em_slice: &mut [f32],
    q_bat: &mut [f32],
    t_acc: &mut [f32],
) -> Result<(), String> {
    let m = backend.n_snps;
    let n = backend.n_samples;
    if m == 0 || n == 0 {
        return Ok(());
    }
    if p_src.len() % m != 0 {
        return Err("invalid P shape in packed EM helper".to_string());
    }
    let k = p_src.len() / m;
    if q_src.len() != n * k {
        return Err("invalid Q shape in packed EM helper".to_string());
    }
    if p_em_slice.len() != m * k || q_em_slice.len() != n * k {
        return Err("invalid packed EM output buffer size".to_string());
    }
    if q_bat.len() != n || t_acc.len() != n * k {
        return Err("invalid packed EM scratch buffer size".to_string());
    }

    let full_bytes = n / 4;
    let rem = n % 4;
    let byte_lut = packed_byte_lut();
    let blocks = rayon::current_num_threads().max(1);
    let rows_per_block = (m + blocks - 1) / blocks;
    let chunk_elems = (rows_per_block * k).max(k);
    let (qb_sum, ta_sum, _, _) = p_em_slice
        .par_chunks_mut(chunk_elems)
        .enumerate()
        .fold(
            || {
                (
                    vec![0.0_f32; n],
                    vec![0.0_f32; n * k],
                    vec![0.0_f32; k],
                    vec![0.0_f32; k],
                )
            },
            |(mut qb, mut ta, mut a, mut b), (chunk_idx, em_chunk)| {
                let row_start = chunk_idx * rows_per_block;
                let row_count = em_chunk.len() / k;
                for r in 0..row_count {
                    let i = row_start + r;
                    if i >= m {
                        break;
                    }
                    let row = packed_backend_row_slice(backend, i);
                    let flip = backend.row_flip[i];
                    let p_row = &p_src[i * k..(i + 1) * k];
                    let out_row = &mut em_chunk[r * k..(r + 1) * k];
                    a.fill(0.0);
                    b.fill(0.0);
                    let mut col = 0usize;
                    for &packed in row.iter().take(full_bytes) {
                        let codes = &byte_lut.code4[packed as usize];
                        for &code in codes.iter() {
                            if let Some(g_f) = packed_code_minor_allele_g(code, flip) {
                                qb[col] += 2.0;
                                let q_row = &q_src[col * k..(col + 1) * k];
                                let mut rec = 0.0_f32;
                                for kk in 0..k {
                                    rec += p_row[kk] * q_row[kk];
                                }
                                rec = rec.clamp(1e-6, 1.0 - 1e-6);
                                let aa = g_f / rec;
                                let bb = (2.0 - g_f) / (1.0 - rec);
                                let t_row = &mut ta[col * k..(col + 1) * k];
                                for kk in 0..k {
                                    let qv = q_row[kk];
                                    a[kk] += qv * aa;
                                    b[kk] += qv * bb;
                                    t_row[kk] += p_row[kk] * (aa - bb) + bb;
                                }
                            }
                            col += 1;
                        }
                    }
                    if rem > 0 {
                        let codes = &byte_lut.code4[row[full_bytes] as usize];
                        for &code in codes.iter().take(rem) {
                            if let Some(g_f) = packed_code_minor_allele_g(code, flip) {
                                qb[col] += 2.0;
                                let q_row = &q_src[col * k..(col + 1) * k];
                                let mut rec = 0.0_f32;
                                for kk in 0..k {
                                    rec += p_row[kk] * q_row[kk];
                                }
                                rec = rec.clamp(1e-6, 1.0 - 1e-6);
                                let aa = g_f / rec;
                                let bb = (2.0 - g_f) / (1.0 - rec);
                                let t_row = &mut ta[col * k..(col + 1) * k];
                                for kk in 0..k {
                                    let qv = q_row[kk];
                                    a[kk] += qv * aa;
                                    b[kk] += qv * bb;
                                    t_row[kk] += p_row[kk] * (aa - bb) + bb;
                                }
                            }
                            col += 1;
                        }
                    }
                    for kk in 0..k {
                        let denom = p_row[kk] * (a[kk] - b[kk]) + b[kk];
                        let v = if denom.abs() < 1e-8 {
                            p_row[kk]
                        } else {
                            (a[kk] * p_row[kk]) / denom
                        };
                        out_row[kk] = clip32(v);
                    }
                }
                madvise_backend_active_rows(backend, row_start, row_count);
                (qb, ta, a, b)
            },
        )
        .reduce(
            || {
                (
                    vec![0.0_f32; n],
                    vec![0.0_f32; n * k],
                    vec![0.0_f32; k],
                    vec![0.0_f32; k],
                )
            },
            |(mut qb1, mut ta1, a1, b1), (qb2, ta2, _, _)| {
                for col in 0..n {
                    qb1[col] += qb2[col];
                }
                for idx in 0..(n * k) {
                    ta1[idx] += ta2[idx];
                }
                (qb1, ta1, a1, b1)
            },
        );
    q_bat.copy_from_slice(&qb_sum);
    t_acc.copy_from_slice(&ta_sum);

    q_em_slice
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(col, out_row)| {
            let q_row = &q_src[col * k..(col + 1) * k];
            if q_bat[col] <= 0.0 {
                for kk in 0..k {
                    out_row[kk] = clip32(q_row[kk]);
                }
            } else {
                let inv = 1.0 / q_bat[col];
                let t_row = &t_acc[col * k..(col + 1) * k];
                for kk in 0..k {
                    out_row[kk] = clip32(q_row[kk] * t_row[kk] * inv);
                }
            }
            let sum = out_row.iter().sum::<f32>();
            if sum <= 0.0 || !sum.is_finite() {
                let v = 1.0 / (k as f32).max(1.0);
                out_row.fill(v);
            } else {
                for kk in 0..k {
                    out_row[kk] /= sum;
                }
            }
        });
    Ok(())
}

fn adam_optimize_packed_inplace_impl(
    backend: &PackedBedRsvd,
    p: &mut [f32],
    q: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    max_iter: usize,
    check_every: usize,
    lr_decay: f32,
    min_lr: f32,
) -> Result<(f64, usize), String> {
    if m != backend.n_snps || n != backend.n_samples {
        return Err(format!(
            "shape mismatch: expected P/Q rows=({}, {}), got ({}, {})",
            backend.n_snps, backend.n_samples, m, n
        ));
    }
    if m == 0 || n == 0 || k == 0 {
        return Err("invalid empty input matrix for BED-backed ADAM optimization".to_string());
    }
    if p.len() != m * k || q.len() != n * k {
        return Err("invalid contiguous slice length for BED-backed ADAM optimization".to_string());
    }

    let mut p_best = vec![0.0_f32; m * k];
    let mut q_best = vec![0.0_f32; n * k];
    let mut ll_best = f64::NEG_INFINITY;
    let mut no_improve: usize = 0;
    let mut lr_cur = lr;
    let mut last_iter = 0usize;
    let mut beta1_pow = 1.0_f32;
    let mut beta2_pow = 1.0_f32;
    let mut have_best = false;

    let mut m_p = vec![0.0_f32; m * k];
    let mut v_p = vec![0.0_f32; m * k];
    let mut m_q = vec![0.0_f32; n * k];
    let mut v_q = vec![0.0_f32; n * k];
    let mut q_em = vec![0.0_f32; n * k];
    let check_every = check_every.max(1);
    let full_bytes = n / 4;
    let rem = n % 4;
    let byte_lut = packed_byte_lut();
    let blocks = rayon::current_num_threads().max(1);
    let rows_per_block = (m + blocks - 1) / blocks;
    let chunk_elems = (rows_per_block * k).max(k);
    check_admx_memory_limit("adam/scratch_ready")?;

    for it in 0..max_iter {
        check_admx_memory_limit("adam/iter_start")?;
        last_iter = it + 1;
        let one_b1 = 1.0_f32 - beta1;
        let one_b2 = 1.0_f32 - beta2;
        beta1_pow *= beta1;
        beta2_pow *= beta2;
        let m_scale = if (1.0_f32 - beta1_pow).abs() < 1e-8 {
            1.0_f32
        } else {
            1.0_f32 / (1.0_f32 - beta1_pow)
        };
        let v_scale = if (1.0_f32 - beta2_pow).abs() < 1e-8 {
            1.0_f32
        } else {
            1.0_f32 / (1.0_f32 - beta2_pow)
        };

        let (qb_sum, ta_sum, _, _) = p
            .par_chunks_mut(chunk_elems)
            .zip(m_p.par_chunks_mut(chunk_elems))
            .zip(v_p.par_chunks_mut(chunk_elems))
            .enumerate()
            .fold(
                || {
                    (
                        vec![0.0_f32; n],
                        vec![0.0_f32; n * k],
                        vec![0.0_f32; k],
                        vec![0.0_f32; k],
                    )
                },
                |(mut qb, mut ta, mut a, mut b), (chunk_idx, ((p_chunk, m_chunk), v_chunk))| {
                    let row_start = chunk_idx * rows_per_block;
                    let row_count = p_chunk.len() / k;
                    for r in 0..row_count {
                        let i = row_start + r;
                        if i >= m {
                            break;
                        }
                        let row = packed_backend_row_slice(backend, i);
                        let flip = backend.row_flip[i];
                        let p_row = &mut p_chunk[r * k..(r + 1) * k];
                        let m_row = &mut m_chunk[r * k..(r + 1) * k];
                        let v_row = &mut v_chunk[r * k..(r + 1) * k];
                        a.fill(0.0_f32);
                        b.fill(0.0_f32);
                        let mut col = 0usize;
                        let geno_lut = if flip {
                            [2.0_f32, -1.0_f32, 1.0_f32, 0.0_f32]
                        } else {
                            [0.0_f32, -1.0_f32, 1.0_f32, 2.0_f32]
                        };
                        for &packed in row.iter().take(full_bytes) {
                            let codes = &byte_lut.code4[packed as usize];
                            for &code in codes.iter() {
                                let g_f = geno_lut[code as usize];
                                if g_f >= 0.0_f32 {
                                    qb[col] += 2.0_f32;
                                    let q_row = &q[col * k..(col + 1) * k];
                                    let mut rec = 0.0_f32;
                                    for kk in 0..k {
                                        rec += p_row[kk] * q_row[kk];
                                    }
                                    rec = rec.clamp(1e-6, 1.0 - 1e-6);
                                    let aa = g_f / rec;
                                    let bb = (2.0_f32 - g_f) / (1.0_f32 - rec);
                                    let t_row = &mut ta[col * k..(col + 1) * k];
                                    for kk in 0..k {
                                        let qv = q_row[kk];
                                        a[kk] += qv * aa;
                                        b[kk] += qv * bb;
                                        t_row[kk] += p_row[kk] * (aa - bb) + bb;
                                    }
                                }
                                col += 1;
                            }
                        }
                        if rem > 0 {
                            let codes = &byte_lut.code4[row[full_bytes] as usize];
                            for &code in codes.iter().take(rem) {
                                let g_f = geno_lut[code as usize];
                                if g_f >= 0.0_f32 {
                                    qb[col] += 2.0_f32;
                                    let q_row = &q[col * k..(col + 1) * k];
                                    let mut rec = 0.0_f32;
                                    for kk in 0..k {
                                        rec += p_row[kk] * q_row[kk];
                                    }
                                    rec = rec.clamp(1e-6, 1.0 - 1e-6);
                                    let aa = g_f / rec;
                                    let bb = (2.0_f32 - g_f) / (1.0_f32 - rec);
                                    let t_row = &mut ta[col * k..(col + 1) * k];
                                    for kk in 0..k {
                                        let qv = q_row[kk];
                                        a[kk] += qv * aa;
                                        b[kk] += qv * bb;
                                        t_row[kk] += p_row[kk] * (aa - bb) + bb;
                                    }
                                }
                                col += 1;
                            }
                        }
                        for kk in 0..k {
                            let denom = p_row[kk] * (a[kk] - b[kk]) + b[kk];
                            let p_em = if denom.abs() < 1e-8_f32 {
                                p_row[kk]
                            } else {
                                (a[kk] * p_row[kk]) / denom
                            };
                            let delta = p_em - p_row[kk];
                            let mcur = beta1 * m_row[kk] + one_b1 * delta;
                            let vcur = beta2 * v_row[kk] + one_b2 * delta * delta;
                            let m_hat = mcur * m_scale;
                            let v_hat = vcur * v_scale;
                            let step = lr_cur * m_hat / (v_hat.sqrt() + epsilon);
                            p_row[kk] = clip32(p_row[kk] + step);
                            m_row[kk] = mcur;
                            v_row[kk] = vcur;
                        }
                    }
                    madvise_backend_active_rows(backend, row_start, row_count);
                    (qb, ta, a, b)
                },
            )
            .reduce(
                || {
                    (
                        vec![0.0_f32; n],
                        vec![0.0_f32; n * k],
                        vec![0.0_f32; k],
                        vec![0.0_f32; k],
                    )
                },
                |(mut qb1, mut ta1, a1, b1), (qb2, ta2, _, _)| {
                    for (dst, src) in qb1.iter_mut().zip(qb2.iter()) {
                        *dst += *src;
                    }
                    for (dst, src) in ta1.iter_mut().zip(ta2.iter()) {
                        *dst += *src;
                    }
                    (qb1, ta1, a1, b1)
                },
            );
        q_em.par_chunks_mut(k)
            .enumerate()
            .for_each(|(col, out_row)| {
                let q_row = &q[col * k..(col + 1) * k];
                if qb_sum[col] <= 0.0_f32 {
                    for kk in 0..k {
                        out_row[kk] = clip32(q_row[kk]);
                    }
                } else {
                    let inv = 1.0_f32 / qb_sum[col];
                    let t_row = &ta_sum[col * k..(col + 1) * k];
                    for kk in 0..k {
                        out_row[kk] = clip32(q_row[kk] * t_row[kk] * inv);
                    }
                }
                let sum = out_row.iter().sum::<f32>();
                if sum <= 0.0_f32 || !sum.is_finite() {
                    let v = 1.0_f32 / (k as f32).max(1.0_f32);
                    out_row.fill(v);
                } else {
                    for kk in 0..k {
                        out_row[kk] /= sum;
                    }
                }
            });

        q.par_chunks_mut(k)
            .zip(m_q.par_chunks_mut(k))
            .zip(v_q.par_chunks_mut(k))
            .enumerate()
            .for_each(|(i, ((q_row, m_row), v_row))| {
                let base = i * k;
                let mut sum = 0.0_f32;
                for j in 0..k {
                    let idx = base + j;
                    let delta = q_em[idx] - q_row[j];
                    let mcur = beta1 * m_row[j] + one_b1 * delta;
                    let vcur = beta2 * v_row[j] + one_b2 * delta * delta;
                    let m_hat = mcur * m_scale;
                    let v_hat = vcur * v_scale;
                    let step = lr_cur * m_hat / (v_hat.sqrt() + epsilon);
                    let qv = clip32(q_row[j] + step);
                    q_row[j] = qv;
                    m_row[j] = mcur;
                    v_row[j] = vcur;
                    sum += qv;
                }
                if sum <= 0.0 || !sum.is_finite() {
                    let v = 1.0 / (k as f32).max(1.0);
                    q_row.fill(v);
                } else {
                    for j in 0..k {
                        q_row[j] /= sum;
                    }
                }
            });

        if last_iter % check_every != 0 {
            continue;
        }

        let ll_cur = loglikelihood_packed_f32_impl(backend, &p, &q, k);
        check_admx_memory_limit("adam/loglik")?;
        if (ll_cur - ll_best).abs() < 0.1 {
            break;
        }

        if ll_cur > ll_best {
            ll_best = ll_cur;
            p_best.copy_from_slice(&p);
            q_best.copy_from_slice(&q);
            have_best = true;
            no_improve = 0;
        } else {
            no_improve += 1;
            lr_cur = (lr_cur * lr_decay).max(min_lr);
            if no_improve >= 2 {
                break;
            }
        }
    }

    if !ll_best.is_finite() {
        ll_best = loglikelihood_packed_f32_impl(backend, &p, &q, k);
        p_best.copy_from_slice(&p);
        q_best.copy_from_slice(&q);
        have_best = true;
    }
    if have_best {
        p.copy_from_slice(&p_best);
        q.copy_from_slice(&q_best);
    }
    Ok((ll_best, last_iter))
}

fn adam_optimize_packed_impl(
    backend: &PackedBedRsvd,
    p0: &ArrayView2<'_, f32>,
    q0: &ArrayView2<'_, f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    max_iter: usize,
    check_every: usize,
    lr_decay: f32,
    min_lr: f32,
) -> Result<(Vec<f32>, Vec<f32>, f64, usize), String> {
    let (m, k) = p0.dim();
    let (n, k2) = q0.dim();
    if k != k2 {
        return Err(format!(
            "shape mismatch: expected shared K, got P0=({}, {}), Q0=({}, {})",
            m, k, n, k2
        ));
    }
    let p0s = p0
        .as_slice()
        .ok_or_else(|| "P0 not contiguous".to_string())?;
    let q0s = q0
        .as_slice()
        .ok_or_else(|| "Q0 not contiguous".to_string())?;

    let mut p = p0s.to_vec();
    let mut q = q0s.to_vec();
    let (ll_best, last_iter) = adam_optimize_packed_inplace_impl(
        backend,
        &mut p,
        &mut q,
        m,
        n,
        k,
        lr,
        beta1,
        beta2,
        epsilon,
        max_iter,
        check_every,
        lr_decay,
        min_lr,
    )?;
    Ok((p, q, ll_best, last_iter))
}

#[pymethods]
impl AdmxBedTrainingSession {
    #[new]
    #[pyo3(signature = (
        genotype_path,
        snps_only=true,
        maf=0.02,
        missing_rate=0.05,
        memory_mb=0
    ))]
    fn new(
        genotype_path: String,
        snps_only: bool,
        maf: f32,
        missing_rate: f32,
        memory_mb: usize,
    ) -> PyResult<Self> {
        let (backend, prefix) = open_packed_bed_backend_for_training(
            &genotype_path,
            snps_only,
            maf,
            missing_rate,
            memory_mb,
        )
        .map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            backend: Arc::new(backend),
            prefix,
            snps_only,
            maf,
            missing_rate,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (
        genotype_path,
        snps_only=true,
        maf=0.02,
        missing_rate=0.05,
        memory_mb=0
    ))]
    fn open(
        py: Python<'_>,
        genotype_path: String,
        snps_only: bool,
        maf: f32,
        missing_rate: f32,
        memory_mb: usize,
    ) -> PyResult<Self> {
        let (backend, prefix) = py
            .detach(move || {
                open_packed_bed_backend_for_training(
                    &genotype_path,
                    snps_only,
                    maf,
                    missing_rate,
                    memory_mb,
                )
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            backend: Arc::new(backend),
            prefix,
            snps_only,
            maf,
            missing_rate,
        })
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.backend.n_samples
    }

    #[getter]
    fn n_snps(&self) -> usize {
        self.backend.n_snps
    }

    #[getter]
    fn prefix(&self) -> String {
        self.prefix.clone()
    }

    fn row_freq<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        row_freq_to_pyarray(py, &self.backend.row_freq)
    }

    fn write_site_file(&self, out_path: String) -> PyResult<()> {
        write_p_site_from_backend_impl(
            &self.prefix,
            &self.backend.active_row_idx,
            &self.backend.row_flip,
            &out_path,
        )
        .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (
        k,
        seed,
        solver,
        power,
        tol,
        max_als,
        reg_als,
        lr,
        beta1,
        beta2,
        epsilon,
        max_iter,
        check_every,
        lr_decay,
        min_lr
    ))]
    fn fit_k<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        seed: u64,
        solver: String,
        power: usize,
        tol: f32,
        max_als: usize,
        reg_als: f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        max_iter: usize,
        check_every: usize,
        lr_decay: f32,
        min_lr: f32,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        f64,
        usize,
        f64,
        usize,
    )> {
        let backend = Arc::clone(&self.backend);
        let result = py
            .detach(move || {
                fit_admx_bed_training_session_impl(
                    backend.as_ref(),
                    k,
                    seed,
                    &solver,
                    power,
                    tol,
                    max_als,
                    reg_als,
                    lr,
                    beta1,
                    beta2,
                    epsilon,
                    max_iter,
                    check_every,
                    lr_decay,
                    min_lr,
                )
            })
            .map_err(PyRuntimeError::new_err)?;
        let AdmxBedFitResult {
            p,
            q,
            ll_final,
            adam_iter,
            init_ll,
            als_iter,
        } = result;
        let p_out = vec_into_pyarray2_f32(py, self.backend.n_snps, k, p)?;
        let q_out = vec_into_pyarray2_f32(py, self.backend.n_samples, k, q)?;
        Ok((p_out, q_out, ll_final, adam_iter, init_ll, als_iter))
    }

    fn __repr__(&self) -> String {
        format!(
            "AdmxBedTrainingSession(prefix='{}', n_snps={}, n_samples={}, snps_only={}, maf={}, missing_rate={})",
            self.prefix,
            self.backend.n_snps,
            self.backend.n_samples,
            self.snps_only,
            self.maf,
            self.missing_rate,
        )
    }
}

#[pymethods]
impl AdmxBedBackend {
    #[new]
    #[pyo3(signature = (
        genotype_path,
        snps_only=true,
        maf=0.02,
        missing_rate=0.05,
        memory_mb=0
    ))]
    fn new(
        genotype_path: String,
        snps_only: bool,
        maf: f32,
        missing_rate: f32,
        memory_mb: usize,
    ) -> PyResult<Self> {
        let (backend, prefix) = open_packed_bed_backend_for_training(
            &genotype_path,
            snps_only,
            maf,
            missing_rate,
            memory_mb,
        )
        .map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            backend: Arc::new(backend),
            prefix,
            snps_only,
            maf,
            missing_rate,
        })
    }

    fn row_freq<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        row_freq_to_pyarray(py, &self.backend.row_freq)
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.backend.n_samples
    }

    #[getter]
    fn n_snps(&self) -> usize {
        self.backend.n_snps
    }

    #[getter]
    fn prefix(&self) -> String {
        self.prefix.clone()
    }

    fn rsvd_stream_sample<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        seed: u64,
        power: usize,
        tol: f32,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>, f64)> {
        let (eigvals, eigvecs_sample, total_variance) =
            rsvd_stream_sample_packed_impl(self.backend.as_ref(), k, seed, power, tol, 0.0)
                .map_err(PyRuntimeError::new_err)?;
        let k_eff = eigvals.len();
        let eval_arr = row_freq_to_pyarray(py, &eigvals);
        let evec_arr = vec_to_pyarray2_f32(py, self.backend.n_samples, k_eff, &eigvecs_sample)?;
        Ok((eval_arr, evec_arr, total_variance))
    }

    fn multiply_a_omega<'py>(
        &self,
        py: Python<'py>,
        omega: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let omega = omega.as_array();
        let (n, kp) = omega.dim();
        let omega_s = omega
            .as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("omega is not contiguous"))?;
        let out_vec = multiply_a_omega_packed_checked_impl(
            self.backend.as_ref(),
            omega_s,
            n,
            kp,
            self.backend.n_snps,
        )
        .map_err(PyRuntimeError::new_err)?;
        vec_to_pyarray2_f32(py, self.backend.n_snps, kp, &out_vec)
    }

    fn loglikelihood_f32(
        &self,
        p: PyReadonlyArray2<'_, f32>,
        q: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<f64> {
        loglikelihood_packed_checked_impl(self.backend.as_ref(), &p.as_array(), &q.as_array())
            .map_err(PyRuntimeError::new_err)
    }

    fn write_site_file(&self, out_path: String) -> PyResult<()> {
        write_p_site_from_backend_impl(
            &self.prefix,
            &self.backend.active_row_idx,
            &self.backend.row_flip,
            &out_path,
        )
        .map_err(PyRuntimeError::new_err)
    }

    fn cv_observed_fold_counts<'py>(
        &self,
        py: Python<'py>,
        folds: usize,
        seed: u64,
    ) -> PyResult<(usize, Bound<'py, PyArray1<i64>>)> {
        let (observed, fold_counts) =
            cv_observed_fold_counts_packed_impl(self.backend.as_ref(), folds, seed)
                .map_err(PyRuntimeError::new_err)?;
        Ok((observed, vec_to_pyarray1_i64(py, &fold_counts)?))
    }

    fn make_cv_fold_backend(
        &self,
        fold_id: usize,
        folds: usize,
        seed: u64,
    ) -> PyResult<AdmxBedFoldBackend> {
        let (row_freq_train, train_observed, holdout_observed) =
            cv_training_row_freq_packed_impl(self.backend.as_ref(), fold_id, folds, seed)
                .map_err(PyRuntimeError::new_err)?;
        Ok(AdmxBedFoldBackend {
            backend: Arc::clone(&self.backend),
            prefix: self.prefix.clone(),
            snps_only: self.snps_only,
            maf: self.maf,
            missing_rate: self.missing_rate,
            fold_id,
            folds,
            cv_seed: seed,
            row_freq_train,
            train_observed,
            holdout_observed,
        })
    }

    #[pyo3(signature = (
        p0,
        q0,
        lr=0.005,
        beta1=0.80,
        beta2=0.88,
        epsilon=1e-8,
        max_iter=1500,
        check_every=5,
        lr_decay=0.5,
        min_lr=1e-6
    ))]
    fn adam_optimize_f32<'py>(
        &self,
        py: Python<'py>,
        p0: PyReadonlyArray2<'py, f32>,
        q0: PyReadonlyArray2<'py, f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        max_iter: usize,
        check_every: usize,
        lr_decay: f32,
        min_lr: f32,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        f64,
        usize,
    )> {
        let (p_best, q_best, ll_best, last_iter) = adam_optimize_packed_impl(
            self.backend.as_ref(),
            &p0.as_array(),
            &q0.as_array(),
            lr,
            beta1,
            beta2,
            epsilon,
            max_iter,
            check_every,
            lr_decay,
            min_lr,
        )
        .map_err(PyRuntimeError::new_err)?;
        let p_out = vec_to_pyarray2_f32(py, self.backend.n_snps, p0.shape()[1], &p_best)?;
        let q_out = vec_to_pyarray2_f32(py, self.backend.n_samples, q0.shape()[1], &q_best)?;
        Ok((p_out, q_out, ll_best, last_iter))
    }

    #[pyo3(signature = (
        p,
        q,
        lr=0.005,
        beta1=0.80,
        beta2=0.88,
        epsilon=1e-8,
        max_iter=1500,
        check_every=5,
        lr_decay=0.5,
        min_lr=1e-6
    ))]
    fn adam_optimize_inplace_f32<'py>(
        &self,
        mut p: PyReadwriteArray2<'py, f32>,
        mut q: PyReadwriteArray2<'py, f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        max_iter: usize,
        check_every: usize,
        lr_decay: f32,
        min_lr: f32,
    ) -> PyResult<(f64, usize)> {
        let mut p_arr = p.as_array_mut();
        let mut q_arr = q.as_array_mut();
        let (m, k) = p_arr.dim();
        let (n, k2) = q_arr.dim();
        if k != k2 {
            return Err(PyRuntimeError::new_err(format!(
                "shape mismatch: expected shared K, got P=({}, {}), Q=({}, {})",
                m, k, n, k2
            )));
        }
        let p_s = p_arr
            .as_slice_mut()
            .ok_or_else(|| PyRuntimeError::new_err("P is not contiguous"))?;
        let q_s = q_arr
            .as_slice_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Q is not contiguous"))?;
        adam_optimize_packed_inplace_impl(
            self.backend.as_ref(),
            p_s,
            q_s,
            m,
            n,
            k,
            lr,
            beta1,
            beta2,
            epsilon,
            max_iter,
            check_every,
            lr_decay,
            min_lr,
        )
        .map_err(PyRuntimeError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "AdmxBedBackend(prefix='{}', n_snps={}, n_samples={}, snps_only={}, maf={}, missing_rate={})",
            self.prefix,
            self.backend.n_snps,
            self.backend.n_samples,
            self.snps_only,
            self.maf,
            self.missing_rate,
        )
    }
}

#[pymethods]
impl AdmxBedFoldBackend {
    fn row_freq<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        row_freq_to_pyarray(py, &self.row_freq_train)
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.backend.n_samples
    }

    #[getter]
    fn n_snps(&self) -> usize {
        self.backend.n_snps
    }

    #[getter]
    fn prefix(&self) -> String {
        self.prefix.clone()
    }

    #[getter]
    fn train_observed(&self) -> usize {
        self.train_observed
    }

    #[getter]
    fn holdout_observed(&self) -> usize {
        self.holdout_observed
    }

    fn rsvd_stream_sample<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        seed: u64,
        power: usize,
        tol: f32,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>)> {
        let (eigvals, eigvecs_sample) = rsvd_stream_sample_packed_fold_impl(
            self.backend.as_ref(),
            &self.row_freq_train,
            k,
            seed,
            power,
            tol,
            self.fold_id,
            self.folds,
            self.cv_seed,
        )
        .map_err(PyRuntimeError::new_err)?;
        let k_eff = eigvals.len();
        let eval_arr = row_freq_to_pyarray(py, &eigvals);
        let evec_arr = vec_to_pyarray2_f32(py, self.backend.n_samples, k_eff, &eigvecs_sample)?;
        Ok((eval_arr, evec_arr))
    }

    fn multiply_a_omega<'py>(
        &self,
        py: Python<'py>,
        omega: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let omega = omega.as_array();
        let (n, kp) = omega.dim();
        let omega_s = omega
            .as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("omega is not contiguous"))?;
        let out_vec = compute_a_omega_packed_fold_impl(
            self.backend.as_ref(),
            &self.row_freq_train,
            omega_s,
            kp,
            self.fold_id,
            self.folds,
            self.cv_seed,
        )
        .map_err(PyRuntimeError::new_err)?;
        if n != self.backend.n_samples {
            return Err(PyRuntimeError::new_err(format!(
                "omega sample dimension mismatch: got {n}, expected {}",
                self.backend.n_samples
            )));
        }
        vec_to_pyarray2_f32(py, self.backend.n_snps, kp, &out_vec)
    }

    fn loglikelihood_f32(
        &self,
        p: PyReadonlyArray2<'_, f32>,
        q: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<f64> {
        let p = p.as_array();
        let q = q.as_array();
        let (m, k) = p.dim();
        let (n, k2) = q.dim();
        if m != self.backend.n_snps || n != self.backend.n_samples || k != k2 {
            return Err(PyRuntimeError::new_err(format!(
                "shape mismatch: expected P=({}, {}), Q=({}, {}), got P=({}, {}), Q=({}, {})",
                self.backend.n_snps, k, self.backend.n_samples, k, m, k, n, k2
            )));
        }
        let p_s = p
            .as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("P is not contiguous"))?;
        let q_s = q
            .as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("Q is not contiguous"))?;
        Ok(loglikelihood_packed_train_fold_f32_impl(
            self.backend.as_ref(),
            p_s,
            q_s,
            k,
            self.fold_id,
            self.folds,
            self.cv_seed,
        ))
    }

    fn holdout_nll_f32(
        &self,
        p: PyReadonlyArray2<'_, f32>,
        q: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<(f64, f64, usize)> {
        let p = p.as_array();
        let q = q.as_array();
        let (m, k) = p.dim();
        let (n, k2) = q.dim();
        if m != self.backend.n_snps || n != self.backend.n_samples || k != k2 {
            return Err(PyRuntimeError::new_err(format!(
                "shape mismatch: expected P=({}, {}), Q=({}, {}), got P=({}, {}), Q=({}, {})",
                self.backend.n_snps, k, self.backend.n_samples, k, m, k, n, k2
            )));
        }
        let p_s = p
            .as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("P is not contiguous"))?;
        let q_s = q
            .as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("Q is not contiguous"))?;
        let (ll_sum, n_holdout) = holdout_nll_packed_fold_f32_impl(
            self.backend.as_ref(),
            p_s,
            q_s,
            k,
            self.fold_id,
            self.folds,
            self.cv_seed,
        );
        if n_holdout == 0 {
            return Ok((f64::NAN, f64::NAN, 0));
        }
        let mean_nll = -ll_sum / (n_holdout as f64);
        let deviance = -2.0_f64 * ll_sum;
        Ok((mean_nll, deviance, n_holdout))
    }

    fn write_site_file(&self, out_path: String) -> PyResult<()> {
        write_p_site_from_backend_impl(
            &self.prefix,
            &self.backend.active_row_idx,
            &self.backend.row_flip,
            &out_path,
        )
        .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (
        p0,
        q0,
        lr=0.005,
        beta1=0.80,
        beta2=0.88,
        epsilon=1e-8,
        max_iter=1500,
        check_every=5,
        lr_decay=0.5,
        min_lr=1e-6
    ))]
    fn adam_optimize_f32<'py>(
        &self,
        py: Python<'py>,
        p0: PyReadonlyArray2<'py, f32>,
        q0: PyReadonlyArray2<'py, f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        max_iter: usize,
        check_every: usize,
        lr_decay: f32,
        min_lr: f32,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        f64,
        usize,
    )> {
        let (p_best, q_best, ll_best, last_iter) = adam_optimize_packed_fold_impl(
            self.backend.as_ref(),
            &p0.as_array(),
            &q0.as_array(),
            lr,
            beta1,
            beta2,
            epsilon,
            max_iter,
            check_every,
            lr_decay,
            min_lr,
            self.fold_id,
            self.folds,
            self.cv_seed,
        )
        .map_err(PyRuntimeError::new_err)?;
        let p_out = vec_to_pyarray2_f32(py, self.backend.n_snps, p0.shape()[1], &p_best)?;
        let q_out = vec_to_pyarray2_f32(py, self.backend.n_samples, q0.shape()[1], &q_best)?;
        Ok((p_out, q_out, ll_best, last_iter))
    }

    #[pyo3(signature = (
        p,
        q,
        lr=0.005,
        beta1=0.80,
        beta2=0.88,
        epsilon=1e-8,
        max_iter=1500,
        check_every=5,
        lr_decay=0.5,
        min_lr=1e-6
    ))]
    fn adam_optimize_inplace_f32<'py>(
        &self,
        mut p: PyReadwriteArray2<'py, f32>,
        mut q: PyReadwriteArray2<'py, f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        max_iter: usize,
        check_every: usize,
        lr_decay: f32,
        min_lr: f32,
    ) -> PyResult<(f64, usize)> {
        let mut p_arr = p.as_array_mut();
        let mut q_arr = q.as_array_mut();
        let (m, k) = p_arr.dim();
        let (n, k2) = q_arr.dim();
        if k != k2 {
            return Err(PyRuntimeError::new_err(format!(
                "shape mismatch: expected shared K, got P=({}, {}), Q=({}, {})",
                m, k, n, k2
            )));
        }
        let p_s = p_arr
            .as_slice_mut()
            .ok_or_else(|| PyRuntimeError::new_err("P is not contiguous"))?;
        let q_s = q_arr
            .as_slice_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Q is not contiguous"))?;
        adam_optimize_packed_fold_inplace_impl(
            self.backend.as_ref(),
            p_s,
            q_s,
            m,
            n,
            k,
            lr,
            beta1,
            beta2,
            epsilon,
            max_iter,
            check_every,
            lr_decay,
            min_lr,
            self.fold_id,
            self.folds,
            self.cv_seed,
        )
        .map_err(PyRuntimeError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "AdmxBedFoldBackend(prefix='{}', fold={}/{}, n_snps={}, n_samples={}, snps_only={}, maf={}, missing_rate={}, train_observed={}, holdout_observed={})",
            self.prefix,
            self.fold_id + 1,
            self.folds,
            self.backend.n_snps,
            self.backend.n_samples,
            self.snps_only,
            self.maf,
            self.missing_rate,
            self.train_observed,
            self.holdout_observed,
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    genotype_path,
    omega,
    row_freq,
    snps_only=true,
    maf=0.02,
    missing_rate=0.05,
    delimiter=None,
    mmap_window_mb=0,
    memory_mb=0
))]
pub fn admx_multiply_a_omega_bed<'py>(
    py: Python<'py>,
    genotype_path: String,
    omega: PyReadonlyArray2<'py, f32>,
    row_freq: PyReadonlyArray1<'py, f32>,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
    delimiter: Option<String>,
    mmap_window_mb: usize,
    memory_mb: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let omega = omega.as_array();
    let (n, kp) = omega.dim();
    let omega_s = omega
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("omega is not contiguous"))?;
    let row_freq_s = row_freq
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("row_freq is not contiguous"))?;
    let cfg = StreamRsvdConfig {
        genotype_path,
        snps_only,
        maf,
        missing_rate,
        delimiter,
        mmap_window_mb,
        memory_mb,
    };
    let packed_backend = try_build_packed_bed_backend(&cfg).map_err(PyRuntimeError::new_err)?;
    let out_vec = if let Some(backend) = packed_backend.as_ref() {
        multiply_a_omega_packed_checked_impl(backend, omega_s, n, kp, row_freq_s.len())
            .map_err(PyRuntimeError::new_err)?
    } else {
        compute_a_omega_stream(&cfg, omega_s, row_freq_s, row_freq_s.len(), n, kp)
            .map_err(PyRuntimeError::new_err)?
    };
    vec_to_pyarray2_f32(py, row_freq_s.len(), kp, &out_vec)
}

#[pyfunction]
#[pyo3(signature = (
    genotype_path,
    snps_only=true,
    maf=0.02,
    missing_rate=0.05,
    memory_mb=0
))]
pub fn admx_bed_training_meta<'py>(
    py: Python<'py>,
    genotype_path: String,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
    memory_mb: usize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, usize)> {
    if !(0.0..=0.5).contains(&maf) {
        return Err(PyRuntimeError::new_err("maf must be within [0, 0.5]"));
    }
    if !(0.0..=1.0).contains(&missing_rate) {
        return Err(PyRuntimeError::new_err(
            "missing_rate must be within [0, 1]",
        ));
    }
    let (backend, _prefix) = py
        .detach(move || {
            open_packed_bed_backend_for_training(
                &genotype_path,
                snps_only,
                maf,
                missing_rate,
                memory_mb,
            )
        })
        .map_err(PyRuntimeError::new_err)?;
    let row_freq_arr = row_freq_to_pyarray(py, &backend.row_freq);
    let n_samples = backend.n_samples;
    Ok((row_freq_arr, n_samples))
}

#[pyfunction]
#[pyo3(signature = (
    genotype_path,
    p,
    q,
    snps_only=true,
    maf=0.02,
    missing_rate=0.05,
    memory_mb=0
))]
pub fn admx_loglikelihood_bed_f32(
    genotype_path: String,
    p: PyReadonlyArray2<'_, f32>,
    q: PyReadonlyArray2<'_, f32>,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
    memory_mb: usize,
) -> PyResult<f64> {
    let (backend, _prefix) = open_packed_bed_backend_for_training(
        &genotype_path,
        snps_only,
        maf,
        missing_rate,
        memory_mb,
    )
    .map_err(PyRuntimeError::new_err)?;
    loglikelihood_packed_checked_impl(&backend, &p.as_array(), &q.as_array())
        .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
pub fn admx_set_threads(threads: usize) -> PyResult<()> {
    if threads == 0 {
        return Err(PyRuntimeError::new_err(
            "admx_set_threads expects a positive thread count",
        ));
    }
    match ThreadPoolBuilder::new().num_threads(threads).build_global() {
        Ok(()) => Ok(()),
        Err(_) => Ok(()),
    }
}

#[pyfunction]
pub fn admx_multiply_at_omega<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<'py, u8>,
    omega: PyReadonlyArray2<'py, f32>,
    f: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let g = g.as_array();
    let omega = omega.as_array();
    let f = f.as_array();
    let (m, n) = g.dim();
    let (m2, kp) = omega.dim();
    if m != m2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G is ({m},{n}), omega is ({m2},{kp}), expected omega rows == G rows"
        )));
    }
    if f.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: f length={} expected {}",
            f.len(),
            m
        )));
    }

    let mut out_vec = vec![0.0_f32; n * kp];
    out_vec
        .par_chunks_mut(kp)
        .enumerate()
        .for_each(|(l, out_row)| {
            for i in 0..m {
                let gv = g[(i, l)];
                if gv == 3 {
                    continue;
                }
                let centered = gv as f32 - 2.0_f32 * f[i];
                for j in 0..kp {
                    out_row[j] += centered * omega[(i, j)];
                }
            }
        });
    let out = PyArray2::<f32>::zeros(py, [n, kp], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&out_vec);
    Ok(out)
}

fn multiply_a_omega_inplace_impl(
    g: &ArrayView2<'_, u8>,
    omega: &[f32],
    f: &[f32],
    out: &mut [f32],
    kp: usize,
) -> Result<(), String> {
    let (m, n) = g.dim();
    if f.len() != m {
        return Err(format!(
            "shape mismatch: f length={} expected {}",
            f.len(),
            m
        ));
    }
    if omega.len() != n * kp {
        return Err(format!(
            "shape mismatch: omega length={} expected {}",
            omega.len(),
            n * kp
        ));
    }
    if out.len() != m * kp {
        return Err(format!(
            "shape mismatch: out length={} expected {}",
            out.len(),
            m * kp
        ));
    }
    out.par_chunks_mut(kp).enumerate().for_each(|(i, row)| {
        let f2 = 2.0_f32 * f[i];
        for j in 0..kp {
            let mut acc = 0.0_f32;
            for l in 0..n {
                let gv = g[(i, l)];
                if gv == 3 {
                    continue;
                }
                acc += (gv as f32 - f2) * omega[l * kp + j];
            }
            row[j] = acc;
        }
    });
    Ok(())
}

fn multiply_at_omega_inplace_impl(
    g: &ArrayView2<'_, u8>,
    omega: &[f32],
    f: &[f32],
    out: &mut [f32],
    kp: usize,
    tile_cols: usize,
) -> Result<(), String> {
    let (m, n) = g.dim();
    if f.len() != m {
        return Err(format!(
            "shape mismatch: f length={} expected {}",
            f.len(),
            m
        ));
    }
    if omega.len() != m * kp {
        return Err(format!(
            "shape mismatch: omega length={} expected {}",
            omega.len(),
            m * kp
        ));
    }
    if out.len() != n * kp {
        return Err(format!(
            "shape mismatch: out length={} expected {}",
            out.len(),
            n * kp
        ));
    }
    out.fill(0.0);
    let tile = tile_cols.max(1).min(n);
    for block_start in (0..n).step_by(tile) {
        let block_len = (n - block_start).min(tile);
        let blk = (0..m)
            .into_par_iter()
            .fold(
                || vec![0.0_f32; block_len * kp],
                |mut local, i| {
                    let f2 = 2.0_f32 * f[i];
                    let omega_row = &omega[i * kp..(i + 1) * kp];
                    for off in 0..block_len {
                        let col = block_start + off;
                        let gv = g[(i, col)];
                        if gv == 3 {
                            continue;
                        }
                        let centered = gv as f32 - f2;
                        let dst = &mut local[off * kp..(off + 1) * kp];
                        for j in 0..kp {
                            dst[j] += centered * omega_row[j];
                        }
                    }
                    local
                },
            )
            .reduce(
                || vec![0.0_f32; block_len * kp],
                |mut a, b| {
                    for idx in 0..(block_len * kp) {
                        a[idx] += b[idx];
                    }
                    a
                },
            );
        out[block_start * kp..(block_start + block_len) * kp].copy_from_slice(&blk);
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (g, omega, f, out, tile_cols=None))]
pub fn admx_multiply_at_omega_inplace(
    g: PyReadonlyArray2<'_, u8>,
    omega: PyReadonlyArray2<'_, f32>,
    f: PyReadonlyArray1<'_, f32>,
    out: &Bound<'_, PyArray2<f32>>,
    tile_cols: Option<usize>,
) -> PyResult<()> {
    let g = g.as_array();
    let omega = omega.as_array();
    let f = f.as_array();
    let (m, _) = g.dim();
    let (m2, kp) = omega.dim();
    if m != m2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G rows={}, omega rows={}",
            m, m2
        )));
    }
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    multiply_at_omega_inplace_impl(
        &g,
        omega
            .as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("omega not contiguous"))?,
        f.as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("f not contiguous"))?,
        out_slice,
        kp,
        tile_cols.unwrap_or(1024),
    )
    .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
pub fn admx_multiply_a_omega_inplace(
    g: PyReadonlyArray2<'_, u8>,
    omega: PyReadonlyArray2<'_, f32>,
    f: PyReadonlyArray1<'_, f32>,
    out: &Bound<'_, PyArray2<f32>>,
) -> PyResult<()> {
    let g = g.as_array();
    let omega = omega.as_array();
    let f = f.as_array();
    let (_, kp) = omega.dim();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    multiply_a_omega_inplace_impl(
        &g,
        omega
            .as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("omega not contiguous"))?,
        f.as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("f not contiguous"))?,
        out_slice,
        kp,
    )
    .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
#[pyo3(signature = (g, omega, f, y_out, g_small_out, tile_cols=None))]
pub fn admx_rsvd_power_step_inplace(
    g: PyReadonlyArray2<'_, u8>,
    omega: PyReadonlyArray2<'_, f32>,
    f: PyReadonlyArray1<'_, f32>,
    y_out: &Bound<'_, PyArray2<f32>>,
    g_small_out: &Bound<'_, PyArray2<f32>>,
    tile_cols: Option<usize>,
) -> PyResult<()> {
    let g = g.as_array();
    let omega = omega.as_array();
    let f = f.as_array();
    let (_, kp) = omega.dim();

    let g_small_slice = unsafe {
        g_small_out
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("g_small_out not contiguous"))?
    };
    let y_slice = unsafe {
        y_out
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("y_out not contiguous"))?
    };

    let omega_slice = omega
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("omega not contiguous"))?;
    let f_slice = f
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("f not contiguous"))?;

    multiply_a_omega_inplace_impl(&g, omega_slice, f_slice, g_small_slice, kp)
        .map_err(PyRuntimeError::new_err)?;
    multiply_at_omega_inplace_impl(
        &g,
        g_small_slice,
        f_slice,
        y_slice,
        kp,
        tile_cols.unwrap_or(1024),
    )
    .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
pub fn admx_multiply_a_omega<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<'py, u8>,
    omega: PyReadonlyArray2<'py, f32>,
    f: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let g = g.as_array();
    let omega = omega.as_array();
    let f = f.as_array();
    let (m, n) = g.dim();
    let (n2, kp) = omega.dim();
    if n != n2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G is ({m},{n}), omega is ({n2},{kp}), expected omega rows == G cols"
        )));
    }
    if f.len() != m {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: f length={} expected {}",
            f.len(),
            m
        )));
    }

    let mut out_vec = vec![0.0_f32; m * kp];
    out_vec.par_chunks_mut(kp).enumerate().for_each(|(i, row)| {
        let f2 = 2.0_f32 * f[i];
        for j in 0..kp {
            let mut acc = 0.0_f32;
            for l in 0..n {
                let gv = g[(i, l)];
                if gv == 3 {
                    continue;
                }
                let centered = gv as f32 - f2;
                acc += centered * omega[(l, j)];
            }
            row[j] = acc;
        }
    });
    let out = PyArray2::<f32>::zeros(py, [m, kp], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&out_vec);
    Ok(out)
}

#[pyfunction]
pub fn admx_allele_frequency<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let g = g.as_array();
    let (m, n) = g.dim();
    let mut out_vec = vec![0.0_f32; m];
    out_vec.par_iter_mut().enumerate().for_each(|(i, dst)| {
        let mut sum_val = 0.0_f32;
        let mut denom = 0.0_f32;
        for j in 0..n {
            let v = g[(i, j)];
            if v == 3 {
                continue;
            }
            sum_val += v as f32;
            denom += 2.0;
        }
        *dst = if denom > 0.0 { sum_val / denom } else { 0.0 };
    });
    let out = PyArray1::<f32>::zeros(py, [m], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    out_slice.copy_from_slice(&out_vec);
    Ok(out)
}

#[pyfunction]
pub fn admx_loglikelihood(
    g: PyReadonlyArray2<'_, u8>,
    p: PyReadonlyArray2<'_, f64>,
    q: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let g = g.as_array();
    let p = p.as_array();
    let q = q.as_array();
    let (m, n) = g.dim();
    let (m2, k) = p.dim();
    let (n2, k2) = q.dim();
    if m != m2 || n != n2 || k != k2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G=({m},{n}), P=({m2},{k}), Q=({n2},{k2})"
        )));
    }
    let ll: f64 = (0..m)
        .into_par_iter()
        .map(|i| {
            let mut part = 0.0_f64;
            for j in 0..n {
                let gv = g[(i, j)];
                if gv == 3 {
                    continue;
                }
                let mut rec = 0.0_f64;
                for kk in 0..k {
                    rec += p[(i, kk)] * q[(j, kk)];
                }
                rec = rec.clamp(1e-12, 1.0 - 1e-12);
                let g_d = gv as f64;
                part += g_d * rec.ln() + (2.0 - g_d) * (1.0 - rec).ln();
            }
            part
        })
        .sum();
    Ok(ll)
}

#[pyfunction]
pub fn admx_loglikelihood_f32(
    g: PyReadonlyArray2<'_, u8>,
    p: PyReadonlyArray2<'_, f32>,
    q: PyReadonlyArray2<'_, f32>,
) -> PyResult<f64> {
    let g = g.as_array();
    let p = p.as_array();
    let q = q.as_array();
    let (m, n) = g.dim();
    let (m2, k) = p.dim();
    let (n2, k2) = q.dim();
    if m != m2 || n != n2 || k != k2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G=({m},{n}), P=({m2},{k}), Q=({n2},{k2})"
        )));
    }
    let ll: f64 = (0..m)
        .into_par_iter()
        .map(|i| {
            let mut part = 0.0_f64;
            for j in 0..n {
                let gv = g[(i, j)];
                if gv == 3 {
                    continue;
                }
                let mut rec = 0.0_f32;
                for kk in 0..k {
                    rec += p[(i, kk)] * q[(j, kk)];
                }
                let rec = rec.clamp(1e-6, 1.0 - 1e-6) as f64;
                let g_d = gv as f64;
                part += g_d * rec.ln() + (2.0 - g_d) * (1.0 - rec).ln();
            }
            part
        })
        .sum();
    Ok(ll)
}

#[pyfunction]
pub fn admx_rmse_f32(
    q1: PyReadonlyArray2<'_, f32>,
    q2: PyReadonlyArray2<'_, f32>,
) -> PyResult<f32> {
    let q1 = q1.as_array();
    let q2 = q2.as_array();
    if q1.dim() != q2.dim() {
        return Err(PyRuntimeError::new_err("shape mismatch in admx_rmse_f32"));
    }
    let (n, k) = q1.dim();
    let t = (n * k) as f32;
    if t <= 0.0 {
        return Ok(0.0);
    }
    let mut acc = 0.0_f64;
    for i in 0..n {
        for j in 0..k {
            let d = (q1[(i, j)] - q2[(i, j)]) as f64;
            acc += d * d;
        }
    }
    Ok(((acc as f32) / t).sqrt())
}

#[pyfunction]
pub fn admx_rmse_f64(
    q1: PyReadonlyArray2<'_, f64>,
    q2: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let q1 = q1.as_array();
    let q2 = q2.as_array();
    if q1.dim() != q2.dim() {
        return Err(PyRuntimeError::new_err("shape mismatch in admx_rmse_f64"));
    }
    let (n, k) = q1.dim();
    let t = (n * k) as f64;
    if t <= 0.0 {
        return Ok(0.0);
    }
    let mut acc = 0.0_f64;
    for i in 0..n {
        for j in 0..k {
            let d = q1[(i, j)] - q2[(i, j)];
            acc += d * d;
        }
    }
    Ok((acc / t).sqrt())
}

#[pyfunction]
pub fn admx_kl_divergence(
    q1: PyReadonlyArray2<'_, f64>,
    q2: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let q1 = q1.as_array();
    let q2 = q2.as_array();
    if q1.dim() != q2.dim() {
        return Err(PyRuntimeError::new_err(
            "shape mismatch in admx_kl_divergence",
        ));
    }
    let (n, k) = q1.dim();
    if n == 0 {
        return Ok(0.0);
    }
    let eps = 1e-10_f64;
    let mut acc = 0.0_f64;
    for i in 0..n {
        for j in 0..k {
            let ai = q1[(i, j)];
            let bi = q2[(i, j)];
            let mid = 0.5 * (ai + bi);
            acc += ai * ((ai / mid) + eps).ln();
        }
    }
    Ok(acc / n as f64)
}

#[pyfunction]
pub fn admx_map_q_f32<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let q = q.as_array();
    let (n, k) = q.dim();
    let out = PyArray2::<f32>::zeros(py, [n, k], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    for i in 0..n {
        let row = &mut out_slice[i * k..(i + 1) * k];
        let mut sum = 0.0_f32;
        for j in 0..k {
            let v = clip32(q[(i, j)]);
            row[j] = v;
            sum += v;
        }
        if sum <= 0.0 {
            let v = 1.0_f32 / (k as f32).max(1.0);
            row.fill(v);
        } else {
            for j in 0..k {
                row[j] /= sum;
            }
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn admx_map_p_f32<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let p = p.as_array();
    let (m, k) = p.dim();
    let out = PyArray2::<f32>::zeros(py, [m, k], false).into_bound();
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output not contiguous"))?
    };
    for i in 0..m {
        let row = &mut out_slice[i * k..(i + 1) * k];
        for j in 0..k {
            row[j] = clip32(p[(i, j)]);
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn admx_em_step<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<'py, u8>,
    p: PyReadonlyArray2<'py, f64>,
    q: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let g = g.as_array();
    let p = p.as_array();
    let q = q.as_array();
    let (m, n) = g.dim();
    let (m2, k) = p.dim();
    let (n2, k2) = q.dim();
    if m != m2 || n != n2 || k != k2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G=({m},{n}), P=({m2},{k}), Q=({n2},{k2})"
        )));
    }

    let p_src = p
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("P not contiguous"))?;
    let q_src = q
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("Q not contiguous"))?;

    // Pass 1: compute P_EM row-wise (parallel over SNP rows).
    let mut p_em_vec = vec![0.0_f64; m * k];
    p_em_vec
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(i, out_row)| {
            let p_row = &p_src[i * k..(i + 1) * k];
            let mut a = vec![0.0_f64; k];
            let mut b = vec![0.0_f64; k];
            for col in 0..n {
                let gv = g[(i, col)];
                if gv == 3 {
                    continue;
                }
                let q_row = &q_src[col * k..(col + 1) * k];
                let mut rec = 0.0_f64;
                for kk in 0..k {
                    rec += p_row[kk] * q_row[kk];
                }
                rec = rec.clamp(1e-12, 1.0 - 1e-12);
                let g_f = gv as f64;
                let aa = g_f / rec;
                let bb = (2.0 - g_f) / (1.0 - rec);
                for kk in 0..k {
                    a[kk] += q_row[kk] * aa;
                    b[kk] += q_row[kk] * bb;
                }
            }
            for kk in 0..k {
                let denom = p_row[kk] * (a[kk] - b[kk]) + b[kk];
                let v = if denom.abs() < 1e-14 {
                    p_row[kk]
                } else {
                    (a[kk] * p_row[kk]) / denom
                };
                out_row[kk] = clip64(v);
            }
        });

    // Pass 2: compute Q_EM sample-wise (fast path with thread-local accumulators).
    let (q_bat, t_acc) = (0..m)
        .into_par_iter()
        .fold(
            || (vec![0.0_f64; n], vec![0.0_f64; n * k]),
            |(mut qb, mut ta), i| {
                let p_row = &p_src[i * k..(i + 1) * k];
                for col in 0..n {
                    let gv = g[(i, col)];
                    if gv == 3 {
                        continue;
                    }
                    qb[col] += 2.0;
                    let q_row = &q_src[col * k..(col + 1) * k];
                    let mut rec = 0.0_f64;
                    for kk in 0..k {
                        rec += p_row[kk] * q_row[kk];
                    }
                    rec = rec.clamp(1e-12, 1.0 - 1e-12);
                    let g_f = gv as f64;
                    let aa = g_f / rec;
                    let bb = (2.0 - g_f) / (1.0 - rec);
                    let t_row = &mut ta[col * k..(col + 1) * k];
                    for kk in 0..k {
                        t_row[kk] += p_row[kk] * (aa - bb) + bb;
                    }
                }
                (qb, ta)
            },
        )
        .reduce(
            || (vec![0.0_f64; n], vec![0.0_f64; n * k]),
            |(mut qb1, mut ta1), (qb2, ta2)| {
                for j in 0..n {
                    qb1[j] += qb2[j];
                }
                for idx in 0..(n * k) {
                    ta1[idx] += ta2[idx];
                }
                (qb1, ta1)
            },
        );

    let mut q_em_vec = vec![0.0_f64; n * k];
    q_em_vec
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(col, out_row)| {
            let q_row = &q_src[col * k..(col + 1) * k];
            if q_bat[col] <= 0.0 {
                for kk in 0..k {
                    out_row[kk] = clip64(q_row[kk]);
                }
            } else {
                let inv = 1.0 / q_bat[col];
                let t_row = &t_acc[col * k..(col + 1) * k];
                for kk in 0..k {
                    out_row[kk] = clip64(q_row[kk] * t_row[kk] * inv);
                }
            }
            let sum = out_row.iter().sum::<f64>();
            if sum <= 0.0 || !sum.is_finite() {
                let v = 1.0 / (k as f64).max(1.0);
                out_row.fill(v);
            } else {
                for kk in 0..k {
                    out_row[kk] /= sum;
                }
            }
        });

    let p_em = PyArray2::<f64>::zeros(py, [m, k], false).into_bound();
    let p_em_slice = unsafe {
        p_em.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output P_EM not contiguous"))?
    };
    p_em_slice.copy_from_slice(&p_em_vec);
    let q_em = PyArray2::<f64>::zeros(py, [n, k], false).into_bound();
    let q_em_slice = unsafe {
        q_em.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("output Q_EM not contiguous"))?
    };
    q_em_slice.copy_from_slice(&q_em_vec);
    Ok((p_em, q_em))
}

#[pyfunction]
pub fn admx_adam_update_p<'py>(
    py: Python<'py>,
    p0: PyReadonlyArray2<'py, f64>,
    p1: PyReadonlyArray2<'py, f64>,
    m_p: PyReadonlyArray2<'py, f64>,
    v_p: PyReadonlyArray2<'py, f64>,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let p0 = p0.as_array();
    let p1 = p1.as_array();
    let m_p = m_p.as_array();
    let v_p = v_p.as_array();
    if p0.dim() != p1.dim() || p0.dim() != m_p.dim() || p0.dim() != v_p.dim() {
        return Err(PyRuntimeError::new_err(
            "shape mismatch in admx_adam_update_p (p0/p1/m_p/v_p)",
        ));
    }
    let (m, k) = p0.dim();
    let mut p_out = vec![0.0_f64; m * k];
    let mut m_out = vec![0.0_f64; m * k];
    let mut v_out = vec![0.0_f64; m * k];
    let one_b1 = 1.0 - beta1;
    let one_b2 = 1.0 - beta2;
    let beta1_t = beta1.powi(t as i32);
    let beta2_t = beta2.powi(t as i32);
    let m_scale = if (1.0 - beta1_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta1_t)
    };
    let v_scale = if (1.0 - beta2_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta2_t)
    };
    let p0s = p0
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("p0 not contiguous"))?;
    let p1s = p1
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("p1 not contiguous"))?;
    let m0s = m_p
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("m_p not contiguous"))?;
    let v0s = v_p
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("v_p not contiguous"))?;
    p_out
        .par_chunks_mut(k)
        .zip(m_out.par_chunks_mut(k))
        .zip(v_out.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, ((p_row, m_row), v_row))| {
            let base = i * k;
            for j in 0..k {
                let idx = base + j;
                let delta = p1s[idx] - p0s[idx];
                let mcur = beta1 * m0s[idx] + one_b1 * delta;
                let vcur = beta2 * v0s[idx] + one_b2 * delta * delta;
                let m_hat = mcur * m_scale;
                let v_hat = vcur * v_scale;
                let step = alpha * m_hat / (v_hat.sqrt() + epsilon);
                p_row[j] = clip64(p0s[idx] + step);
                m_row[j] = mcur;
                v_row[j] = vcur;
            }
        });
    let p_new = PyArray2::<f64>::zeros(py, [m, k], false).into_bound();
    let m_new = PyArray2::<f64>::zeros(py, [m, k], false).into_bound();
    let v_new = PyArray2::<f64>::zeros(py, [m, k], false).into_bound();
    let p_slice = unsafe {
        p_new
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("p_new not contiguous"))?
    };
    let m_slice = unsafe {
        m_new
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("m_new not contiguous"))?
    };
    let v_slice = unsafe {
        v_new
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("v_new not contiguous"))?
    };
    p_slice.copy_from_slice(&p_out);
    m_slice.copy_from_slice(&m_out);
    v_slice.copy_from_slice(&v_out);
    Ok((p_new, m_new, v_new))
}

#[pyfunction]
pub fn admx_adam_update_q<'py>(
    py: Python<'py>,
    q0: PyReadonlyArray2<'py, f64>,
    q1: PyReadonlyArray2<'py, f64>,
    m_q: PyReadonlyArray2<'py, f64>,
    v_q: PyReadonlyArray2<'py, f64>,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let q0 = q0.as_array();
    let q1 = q1.as_array();
    let m_q = m_q.as_array();
    let v_q = v_q.as_array();
    if q0.dim() != q1.dim() || q0.dim() != m_q.dim() || q0.dim() != v_q.dim() {
        return Err(PyRuntimeError::new_err(
            "shape mismatch in admx_adam_update_q (q0/q1/m_q/v_q)",
        ));
    }
    let (n, k) = q0.dim();
    let mut q_out = vec![0.0_f64; n * k];
    let mut m_out = vec![0.0_f64; n * k];
    let mut v_out = vec![0.0_f64; n * k];
    let one_b1 = 1.0 - beta1;
    let one_b2 = 1.0 - beta2;
    let beta1_t = beta1.powi(t as i32);
    let beta2_t = beta2.powi(t as i32);
    let m_scale = if (1.0 - beta1_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta1_t)
    };
    let v_scale = if (1.0 - beta2_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta2_t)
    };
    let q0s = q0
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("q0 not contiguous"))?;
    let q1s = q1
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("q1 not contiguous"))?;
    let m0s = m_q
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("m_q not contiguous"))?;
    let v0s = v_q
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("v_q not contiguous"))?;
    q_out
        .par_chunks_mut(k)
        .zip(m_out.par_chunks_mut(k))
        .zip(v_out.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, ((q_row, m_row), v_row))| {
            let mut sum = 0.0_f64;
            let base = i * k;
            for j in 0..k {
                let idx = base + j;
                let delta = q1s[idx] - q0s[idx];
                let mcur = beta1 * m0s[idx] + one_b1 * delta;
                let vcur = beta2 * v0s[idx] + one_b2 * delta * delta;
                let m_hat = mcur * m_scale;
                let v_hat = vcur * v_scale;
                let step = alpha * m_hat / (v_hat.sqrt() + epsilon);
                let qv = clip64(q0s[idx] + step);
                q_row[j] = qv;
                m_row[j] = mcur;
                v_row[j] = vcur;
                sum += qv;
            }
            if sum <= 0.0 || !sum.is_finite() {
                let v = 1.0 / (k as f64).max(1.0);
                q_row.fill(v);
            } else {
                for j in 0..k {
                    q_row[j] /= sum;
                }
            }
        });
    let q_new = PyArray2::<f64>::zeros(py, [n, k], false).into_bound();
    let m_new = PyArray2::<f64>::zeros(py, [n, k], false).into_bound();
    let v_new = PyArray2::<f64>::zeros(py, [n, k], false).into_bound();
    let q_slice = unsafe {
        q_new
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("q_new not contiguous"))?
    };
    let m_slice = unsafe {
        m_new
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("m_new not contiguous"))?
    };
    let v_slice = unsafe {
        v_new
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("v_new not contiguous"))?
    };
    q_slice.copy_from_slice(&q_out);
    m_slice.copy_from_slice(&m_out);
    v_slice.copy_from_slice(&v_out);
    Ok((q_new, m_new, v_new))
}

#[pyfunction]
pub fn admx_em_step_inplace<'py>(
    g: PyReadonlyArray2<'py, u8>,
    p: PyReadonlyArray2<'py, f64>,
    q: PyReadonlyArray2<'py, f64>,
    p_em: Bound<'py, PyArray2<f64>>,
    q_em: Bound<'py, PyArray2<f64>>,
) -> PyResult<()> {
    let g = g.as_array();
    let p = p.as_array();
    let q = q.as_array();
    let (m, n) = g.dim();
    let (m2, k) = p.dim();
    let (n2, k2) = q.dim();
    if m != m2 || n != n2 || k != k2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G=({m},{n}), P=({m2},{k}), Q=({n2},{k2})"
        )));
    }
    let p_shape = p_em.shape();
    let q_shape = q_em.shape();
    if p_shape.len() != 2
        || q_shape.len() != 2
        || p_shape[0] != m
        || p_shape[1] != k
        || q_shape[0] != n
        || q_shape[1] != k
    {
        return Err(PyRuntimeError::new_err(format!(
            "output shape mismatch: p_em={:?}, q_em={:?}, expected ({m},{k}) and ({n},{k})",
            p_shape, q_shape
        )));
    }

    let p_src = p
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("P not contiguous"))?;
    let q_src = q
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("Q not contiguous"))?;
    let p_em_slice = unsafe {
        p_em.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("p_em is not contiguous"))?
    };
    let q_em_slice = unsafe {
        q_em.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("q_em is not contiguous"))?
    };

    p_em_slice
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(i, out_row)| {
            let p_row = &p_src[i * k..(i + 1) * k];
            let mut a = vec![0.0_f64; k];
            let mut b = vec![0.0_f64; k];
            for col in 0..n {
                let gv = g[(i, col)];
                if gv == 3 {
                    continue;
                }
                let q_row = &q_src[col * k..(col + 1) * k];
                let mut rec = 0.0_f64;
                for kk in 0..k {
                    rec += p_row[kk] * q_row[kk];
                }
                rec = rec.clamp(1e-12, 1.0 - 1e-12);
                let g_f = gv as f64;
                let aa = g_f / rec;
                let bb = (2.0 - g_f) / (1.0 - rec);
                for kk in 0..k {
                    a[kk] += q_row[kk] * aa;
                    b[kk] += q_row[kk] * bb;
                }
            }
            for kk in 0..k {
                let denom = p_row[kk] * (a[kk] - b[kk]) + b[kk];
                let v = if denom.abs() < 1e-14 {
                    p_row[kk]
                } else {
                    (a[kk] * p_row[kk]) / denom
                };
                out_row[kk] = clip64(v);
            }
        });

    let (q_bat, t_acc) = (0..m)
        .into_par_iter()
        .fold(
            || (vec![0.0_f64; n], vec![0.0_f64; n * k]),
            |(mut qb, mut ta), i| {
                let p_row = &p_src[i * k..(i + 1) * k];
                for col in 0..n {
                    let gv = g[(i, col)];
                    if gv == 3 {
                        continue;
                    }
                    qb[col] += 2.0;
                    let q_row = &q_src[col * k..(col + 1) * k];
                    let mut rec = 0.0_f64;
                    for kk in 0..k {
                        rec += p_row[kk] * q_row[kk];
                    }
                    rec = rec.clamp(1e-12, 1.0 - 1e-12);
                    let g_f = gv as f64;
                    let aa = g_f / rec;
                    let bb = (2.0 - g_f) / (1.0 - rec);
                    let t_row = &mut ta[col * k..(col + 1) * k];
                    for kk in 0..k {
                        t_row[kk] += p_row[kk] * (aa - bb) + bb;
                    }
                }
                (qb, ta)
            },
        )
        .reduce(
            || (vec![0.0_f64; n], vec![0.0_f64; n * k]),
            |(mut qb1, mut ta1), (qb2, ta2)| {
                for j in 0..n {
                    qb1[j] += qb2[j];
                }
                for idx in 0..(n * k) {
                    ta1[idx] += ta2[idx];
                }
                (qb1, ta1)
            },
        );

    q_em_slice
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(col, out_row)| {
            let q_row = &q_src[col * k..(col + 1) * k];
            if q_bat[col] <= 0.0 {
                for kk in 0..k {
                    out_row[kk] = clip64(q_row[kk]);
                }
            } else {
                let inv = 1.0 / q_bat[col];
                let t_row = &t_acc[col * k..(col + 1) * k];
                for kk in 0..k {
                    out_row[kk] = clip64(q_row[kk] * t_row[kk] * inv);
                }
            }
            let sum = out_row.iter().sum::<f64>();
            if sum <= 0.0 || !sum.is_finite() {
                let v = 1.0 / (k as f64).max(1.0);
                out_row.fill(v);
            } else {
                for kk in 0..k {
                    out_row[kk] /= sum;
                }
            }
        });
    Ok(())
}

#[pyfunction]
pub fn admx_adam_update_p_inplace<'py>(
    p0: Bound<'py, PyArray2<f64>>,
    p1: PyReadonlyArray2<'py, f64>,
    m_p: Bound<'py, PyArray2<f64>>,
    v_p: Bound<'py, PyArray2<f64>>,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
) -> PyResult<()> {
    let p1 = p1.as_array();
    let p1s = p1
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("p1 not contiguous"))?;

    let p0_shape = p0.shape();
    let m_shape = m_p.shape();
    let v_shape = v_p.shape();
    if p0_shape.len() != 2 || m_shape != p0_shape || v_shape != p0_shape {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch in admx_adam_update_p_inplace: p0={:?}, m_p={:?}, v_p={:?}",
            p0_shape, m_shape, v_shape
        )));
    }
    let m = p0_shape[0];
    let k = p0_shape[1];
    if p1.shape() != [m, k] {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: p1={:?}, expected [{},{}]",
            p1.shape(),
            m,
            k
        )));
    }

    let p0s = unsafe {
        p0.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("p0 is not contiguous"))?
    };
    let m0s = unsafe {
        m_p.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("m_p is not contiguous"))?
    };
    let v0s = unsafe {
        v_p.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("v_p is not contiguous"))?
    };

    let one_b1 = 1.0 - beta1;
    let one_b2 = 1.0 - beta2;
    let beta1_t = beta1.powi(t as i32);
    let beta2_t = beta2.powi(t as i32);
    let m_scale = if (1.0 - beta1_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta1_t)
    };
    let v_scale = if (1.0 - beta2_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta2_t)
    };

    p0s.par_chunks_mut(k)
        .zip(m0s.par_chunks_mut(k))
        .zip(v0s.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, ((p_row, m_row), v_row))| {
            let base = i * k;
            for j in 0..k {
                let idx = base + j;
                let delta = p1s[idx] - p_row[j];
                let mcur = beta1 * m_row[j] + one_b1 * delta;
                let vcur = beta2 * v_row[j] + one_b2 * delta * delta;
                let m_hat = mcur * m_scale;
                let v_hat = vcur * v_scale;
                let step = alpha * m_hat / (v_hat.sqrt() + epsilon);
                p_row[j] = clip64(p_row[j] + step);
                m_row[j] = mcur;
                v_row[j] = vcur;
            }
        });
    Ok(())
}

#[pyfunction]
pub fn admx_adam_update_q_inplace<'py>(
    q0: Bound<'py, PyArray2<f64>>,
    q1: PyReadonlyArray2<'py, f64>,
    m_q: Bound<'py, PyArray2<f64>>,
    v_q: Bound<'py, PyArray2<f64>>,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
) -> PyResult<()> {
    let q1 = q1.as_array();
    let q1s = q1
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("q1 not contiguous"))?;

    let q0_shape = q0.shape();
    let m_shape = m_q.shape();
    let v_shape = v_q.shape();
    if q0_shape.len() != 2 || m_shape != q0_shape || v_shape != q0_shape {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch in admx_adam_update_q_inplace: q0={:?}, m_q={:?}, v_q={:?}",
            q0_shape, m_shape, v_shape
        )));
    }
    let n = q0_shape[0];
    let k = q0_shape[1];
    if q1.shape() != [n, k] {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: q1={:?}, expected [{},{}]",
            q1.shape(),
            n,
            k
        )));
    }

    let q0s = unsafe {
        q0.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("q0 is not contiguous"))?
    };
    let m0s = unsafe {
        m_q.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("m_q is not contiguous"))?
    };
    let v0s = unsafe {
        v_q.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("v_q is not contiguous"))?
    };

    let one_b1 = 1.0 - beta1;
    let one_b2 = 1.0 - beta2;
    let beta1_t = beta1.powi(t as i32);
    let beta2_t = beta2.powi(t as i32);
    let m_scale = if (1.0 - beta1_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta1_t)
    };
    let v_scale = if (1.0 - beta2_t).abs() < 1e-14 {
        1.0
    } else {
        1.0 / (1.0 - beta2_t)
    };

    q0s.par_chunks_mut(k)
        .zip(m0s.par_chunks_mut(k))
        .zip(v0s.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, ((q_row, m_row), v_row))| {
            let mut sum = 0.0_f64;
            let base = i * k;
            for j in 0..k {
                let idx = base + j;
                let delta = q1s[idx] - q_row[j];
                let mcur = beta1 * m_row[j] + one_b1 * delta;
                let vcur = beta2 * v_row[j] + one_b2 * delta * delta;
                let m_hat = mcur * m_scale;
                let v_hat = vcur * v_scale;
                let step = alpha * m_hat / (v_hat.sqrt() + epsilon);
                let qv = clip64(q_row[j] + step);
                q_row[j] = qv;
                m_row[j] = mcur;
                v_row[j] = vcur;
                sum += qv;
            }
            if sum <= 0.0 || !sum.is_finite() {
                let v = 1.0 / (k as f64).max(1.0);
                q_row.fill(v);
            } else {
                for j in 0..k {
                    q_row[j] /= sum;
                }
            }
        });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (g, p, q, p_em, q_em, tile_cols=None))]
pub fn admx_em_step_inplace_f32<'py>(
    g: PyReadonlyArray2<'py, u8>,
    p: PyReadonlyArray2<'py, f32>,
    q: PyReadonlyArray2<'py, f32>,
    p_em: Bound<'py, PyArray2<f32>>,
    q_em: Bound<'py, PyArray2<f32>>,
    tile_cols: Option<usize>,
) -> PyResult<()> {
    let g = g.as_array();
    let p = p.as_array();
    let q = q.as_array();
    let (m, n) = g.dim();
    let (m2, k) = p.dim();
    let (n2, k2) = q.dim();
    if m != m2 || n != n2 || k != k2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G=({m},{n}), P=({m2},{k}), Q=({n2},{k2})"
        )));
    }
    let p_shape = p_em.shape();
    let q_shape = q_em.shape();
    if p_shape.len() != 2
        || q_shape.len() != 2
        || p_shape[0] != m
        || p_shape[1] != k
        || q_shape[0] != n
        || q_shape[1] != k
    {
        return Err(PyRuntimeError::new_err(format!(
            "output shape mismatch: p_em={:?}, q_em={:?}, expected ({m},{k}) and ({n},{k})",
            p_shape, q_shape
        )));
    }

    let p_src = p
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("P not contiguous"))?;
    let q_src = q
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("Q not contiguous"))?;
    let p_em_slice = unsafe {
        p_em.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("p_em is not contiguous"))?
    };
    let q_em_slice = unsafe {
        q_em.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("q_em is not contiguous"))?
    };
    let mut q_bat = vec![0.0_f32; n];
    let mut t_acc = vec![0.0_f32; n * k];
    let tile = tile_cols.unwrap_or(n).max(1);
    em_step_inplace_f32_tiled_impl(
        &g, p_src, q_src, p_em_slice, q_em_slice, &mut q_bat, &mut t_acc, tile,
    )
    .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
pub fn admx_adam_update_p_inplace_f32<'py>(
    p0: Bound<'py, PyArray2<f32>>,
    p1: PyReadonlyArray2<'py, f32>,
    m_p: Bound<'py, PyArray2<f32>>,
    v_p: Bound<'py, PyArray2<f32>>,
    alpha: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
) -> PyResult<()> {
    let p1 = p1.as_array();
    let p1s = p1
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("p1 not contiguous"))?;

    let p0_shape = p0.shape();
    let m_shape = m_p.shape();
    let v_shape = v_p.shape();
    if p0_shape.len() != 2 || m_shape != p0_shape || v_shape != p0_shape {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch in admx_adam_update_p_inplace_f32: p0={:?}, m_p={:?}, v_p={:?}",
            p0_shape, m_shape, v_shape
        )));
    }
    let m = p0_shape[0];
    let k = p0_shape[1];
    if p1.shape() != [m, k] {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: p1={:?}, expected [{},{}]",
            p1.shape(),
            m,
            k
        )));
    }

    let p0s = unsafe {
        p0.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("p0 is not contiguous"))?
    };
    let m0s = unsafe {
        m_p.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("m_p is not contiguous"))?
    };
    let v0s = unsafe {
        v_p.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("v_p is not contiguous"))?
    };

    let one_b1 = 1.0_f32 - beta1;
    let one_b2 = 1.0_f32 - beta2;
    let beta1_t = beta1.powi(t as i32);
    let beta2_t = beta2.powi(t as i32);
    let m_scale = if (1.0_f32 - beta1_t).abs() < 1e-8 {
        1.0_f32
    } else {
        1.0_f32 / (1.0_f32 - beta1_t)
    };
    let v_scale = if (1.0_f32 - beta2_t).abs() < 1e-8 {
        1.0_f32
    } else {
        1.0_f32 / (1.0_f32 - beta2_t)
    };

    p0s.par_chunks_mut(k)
        .zip(m0s.par_chunks_mut(k))
        .zip(v0s.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, ((p_row, m_row), v_row))| {
            let base = i * k;
            for j in 0..k {
                let idx = base + j;
                let delta = p1s[idx] - p_row[j];
                let mcur = beta1 * m_row[j] + one_b1 * delta;
                let vcur = beta2 * v_row[j] + one_b2 * delta * delta;
                let m_hat = mcur * m_scale;
                let v_hat = vcur * v_scale;
                let step = alpha * m_hat / (v_hat.sqrt() + epsilon);
                p_row[j] = clip32(p_row[j] + step);
                m_row[j] = mcur;
                v_row[j] = vcur;
            }
        });
    Ok(())
}

#[pyfunction]
pub fn admx_adam_update_q_inplace_f32<'py>(
    q0: Bound<'py, PyArray2<f32>>,
    q1: PyReadonlyArray2<'py, f32>,
    m_q: Bound<'py, PyArray2<f32>>,
    v_q: Bound<'py, PyArray2<f32>>,
    alpha: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
) -> PyResult<()> {
    let q1 = q1.as_array();
    let q1s = q1
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("q1 not contiguous"))?;

    let q0_shape = q0.shape();
    let m_shape = m_q.shape();
    let v_shape = v_q.shape();
    if q0_shape.len() != 2 || m_shape != q0_shape || v_shape != q0_shape {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch in admx_adam_update_q_inplace_f32: q0={:?}, m_q={:?}, v_q={:?}",
            q0_shape, m_shape, v_shape
        )));
    }
    let n = q0_shape[0];
    let k = q0_shape[1];
    if q1.shape() != [n, k] {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: q1={:?}, expected [{},{}]",
            q1.shape(),
            n,
            k
        )));
    }

    let q0s = unsafe {
        q0.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("q0 is not contiguous"))?
    };
    let m0s = unsafe {
        m_q.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("m_q is not contiguous"))?
    };
    let v0s = unsafe {
        v_q.as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("v_q is not contiguous"))?
    };

    let one_b1 = 1.0_f32 - beta1;
    let one_b2 = 1.0_f32 - beta2;
    let beta1_t = beta1.powi(t as i32);
    let beta2_t = beta2.powi(t as i32);
    let m_scale = if (1.0_f32 - beta1_t).abs() < 1e-8 {
        1.0_f32
    } else {
        1.0_f32 / (1.0_f32 - beta1_t)
    };
    let v_scale = if (1.0_f32 - beta2_t).abs() < 1e-8 {
        1.0_f32
    } else {
        1.0_f32 / (1.0_f32 - beta2_t)
    };

    q0s.par_chunks_mut(k)
        .zip(m0s.par_chunks_mut(k))
        .zip(v0s.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, ((q_row, m_row), v_row))| {
            let mut sum = 0.0_f32;
            let base = i * k;
            for j in 0..k {
                let idx = base + j;
                let delta = q1s[idx] - q_row[j];
                let mcur = beta1 * m_row[j] + one_b1 * delta;
                let vcur = beta2 * v_row[j] + one_b2 * delta * delta;
                let m_hat = mcur * m_scale;
                let v_hat = vcur * v_scale;
                let step = alpha * m_hat / (v_hat.sqrt() + epsilon);
                let qv = clip32(q_row[j] + step);
                q_row[j] = qv;
                m_row[j] = mcur;
                v_row[j] = vcur;
                sum += qv;
            }
            if sum <= 0.0 || !sum.is_finite() {
                let v = 1.0 / (k as f32).max(1.0);
                q_row.fill(v);
            } else {
                for j in 0..k {
                    q_row[j] /= sum;
                }
            }
        });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    g,
    p0,
    q0,
    lr=0.005,
    beta1=0.80,
    beta2=0.88,
    epsilon=1e-8,
    max_iter=1500,
    check_every=5,
    lr_decay=0.5,
    min_lr=1e-6,
    tile_cols=None
))]
pub fn admx_adam_optimize_f32<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<'py, u8>,
    p0: PyReadonlyArray2<'py, f32>,
    q0: PyReadonlyArray2<'py, f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    max_iter: usize,
    check_every: usize,
    lr_decay: f32,
    min_lr: f32,
    tile_cols: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    f64,
    usize,
)> {
    let g = g.as_array();
    let p0 = p0.as_array();
    let q0 = q0.as_array();
    let (m, n) = g.dim();
    let (m2, k) = p0.dim();
    let (n2, k2) = q0.dim();
    if m != m2 || n != n2 || k != k2 {
        return Err(PyRuntimeError::new_err(format!(
            "shape mismatch: G=({m},{n}), P0=({m2},{k}), Q0=({n2},{k2})"
        )));
    }
    if m == 0 || n == 0 || k == 0 {
        return Err(PyRuntimeError::new_err(
            "invalid empty input matrix for ADAM optimization",
        ));
    }

    let p0s = p0
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("P0 not contiguous"))?;
    let q0s = q0
        .as_slice()
        .ok_or_else(|| PyRuntimeError::new_err("Q0 not contiguous"))?;

    let mut p = p0s.to_vec();
    let mut q = q0s.to_vec();
    let mut p_best = p.clone();
    let mut q_best = q.clone();
    let mut ll_best = f64::NEG_INFINITY;
    let mut no_improve: usize = 0;
    let mut lr_cur = lr;
    let mut last_iter = 0usize;
    let mut beta1_pow = 1.0_f32;
    let mut beta2_pow = 1.0_f32;

    let mut m_p = vec![0.0_f32; m * k];
    let mut v_p = vec![0.0_f32; m * k];
    let mut m_q = vec![0.0_f32; n * k];
    let mut v_q = vec![0.0_f32; n * k];
    let mut p_em = vec![0.0_f32; m * k];
    let mut q_em = vec![0.0_f32; n * k];
    let mut q_bat = vec![0.0_f32; n];
    let mut t_acc = vec![0.0_f32; n * k];
    let tile = tile_cols.unwrap_or(n).max(1);
    let check_every = check_every.max(1);

    for it in 0..max_iter {
        last_iter = it + 1;
        em_step_inplace_f32_tiled_impl(
            &g, &p, &q, &mut p_em, &mut q_em, &mut q_bat, &mut t_acc, tile,
        )
        .map_err(PyRuntimeError::new_err)?;

        let one_b1 = 1.0_f32 - beta1;
        let one_b2 = 1.0_f32 - beta2;
        beta1_pow *= beta1;
        beta2_pow *= beta2;
        let m_scale = if (1.0_f32 - beta1_pow).abs() < 1e-8 {
            1.0_f32
        } else {
            1.0_f32 / (1.0_f32 - beta1_pow)
        };
        let v_scale = if (1.0_f32 - beta2_pow).abs() < 1e-8 {
            1.0_f32
        } else {
            1.0_f32 / (1.0_f32 - beta2_pow)
        };

        p.par_chunks_mut(k)
            .zip(m_p.par_chunks_mut(k))
            .zip(v_p.par_chunks_mut(k))
            .enumerate()
            .for_each(|(i, ((p_row, m_row), v_row))| {
                let base = i * k;
                for j in 0..k {
                    let idx = base + j;
                    let delta = p_em[idx] - p_row[j];
                    let mcur = beta1 * m_row[j] + one_b1 * delta;
                    let vcur = beta2 * v_row[j] + one_b2 * delta * delta;
                    let m_hat = mcur * m_scale;
                    let v_hat = vcur * v_scale;
                    let step = lr_cur * m_hat / (v_hat.sqrt() + epsilon);
                    p_row[j] = clip32(p_row[j] + step);
                    m_row[j] = mcur;
                    v_row[j] = vcur;
                }
            });

        q.par_chunks_mut(k)
            .zip(m_q.par_chunks_mut(k))
            .zip(v_q.par_chunks_mut(k))
            .enumerate()
            .for_each(|(i, ((q_row, m_row), v_row))| {
                let base = i * k;
                let mut sum = 0.0_f32;
                for j in 0..k {
                    let idx = base + j;
                    let delta = q_em[idx] - q_row[j];
                    let mcur = beta1 * m_row[j] + one_b1 * delta;
                    let vcur = beta2 * v_row[j] + one_b2 * delta * delta;
                    let m_hat = mcur * m_scale;
                    let v_hat = vcur * v_scale;
                    let step = lr_cur * m_hat / (v_hat.sqrt() + epsilon);
                    let qv = clip32(q_row[j] + step);
                    q_row[j] = qv;
                    m_row[j] = mcur;
                    v_row[j] = vcur;
                    sum += qv;
                }
                if sum <= 0.0 || !sum.is_finite() {
                    let v = 1.0 / (k as f32).max(1.0);
                    q_row.fill(v);
                } else {
                    for j in 0..k {
                        q_row[j] /= sum;
                    }
                }
            });

        if last_iter % check_every != 0 {
            continue;
        }

        let ll_cur = loglikelihood_f32_impl(&g, &p, &q, n, k);
        if (ll_cur - ll_best).abs() < 0.1 {
            break;
        }

        if ll_cur > ll_best {
            ll_best = ll_cur;
            p_best.copy_from_slice(&p);
            q_best.copy_from_slice(&q);
            no_improve = 0;
        } else {
            no_improve += 1;
            lr_cur = (lr_cur * lr_decay).max(min_lr);
            if no_improve >= 2 {
                break;
            }
        }
    }

    if !ll_best.is_finite() {
        ll_best = loglikelihood_f32_impl(&g, &p, &q, n, k);
        p_best.copy_from_slice(&p);
        q_best.copy_from_slice(&q);
    }

    let p_out = PyArray2::<f32>::zeros(py, [m, k], false).into_bound();
    let q_out = PyArray2::<f32>::zeros(py, [n, k], false).into_bound();
    let p_out_s = unsafe {
        p_out
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("p_out is not contiguous"))?
    };
    let q_out_s = unsafe {
        q_out
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("q_out is not contiguous"))?
    };
    p_out_s.copy_from_slice(&p_best);
    q_out_s.copy_from_slice(&q_best);
    Ok((p_out, q_out, ll_best, last_iter))
}

#[pyfunction]
#[pyo3(signature = (
    genotype_path,
    p0,
    q0,
    lr=0.005,
    beta1=0.80,
    beta2=0.88,
    epsilon=1e-8,
    max_iter=1500,
    check_every=5,
    lr_decay=0.5,
    min_lr=1e-6,
    snps_only=true,
    maf=0.02,
    missing_rate=0.05,
    memory_mb=0
))]
pub fn admx_adam_optimize_bed_f32<'py>(
    py: Python<'py>,
    genotype_path: String,
    p0: PyReadonlyArray2<'py, f32>,
    q0: PyReadonlyArray2<'py, f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    max_iter: usize,
    check_every: usize,
    lr_decay: f32,
    min_lr: f32,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
    memory_mb: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    f64,
    usize,
)> {
    let (backend, _prefix) = open_packed_bed_backend_for_training(
        &genotype_path,
        snps_only,
        maf,
        missing_rate,
        memory_mb,
    )
    .map_err(PyRuntimeError::new_err)?;
    let p0v = p0.as_array();
    let q0v = q0.as_array();
    let (p_best, q_best, ll_best, last_iter) = adam_optimize_packed_impl(
        &backend,
        &p0v,
        &q0v,
        lr,
        beta1,
        beta2,
        epsilon,
        max_iter,
        check_every,
        lr_decay,
        min_lr,
    )
    .map_err(PyRuntimeError::new_err)?;
    let p_out = vec_to_pyarray2_f32(py, backend.n_snps, p0v.shape()[1], &p_best)?;
    let q_out = vec_to_pyarray2_f32(py, backend.n_samples, q0v.shape()[1], &q_best)?;
    Ok((p_out, q_out, ll_best, last_iter))
}
