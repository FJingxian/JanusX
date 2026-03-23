use nalgebra::{DMatrix, SymmetricEigen};
use numpy::ndarray::ArrayView2;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
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
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::gfcore::{process_snp_row, read_bim, read_fam, BedSnpIter, HmpSnpIter, TxtSnpIter, VcfSnpIter};

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

#[derive(Clone)]
struct StreamRsvdConfig {
    genotype_path: String,
    snps_only: bool,
    maf: f32,
    missing_rate: f32,
    delimiter: Option<String>,
    mmap_window_mb: usize,
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

struct PackedBedRsvd {
    packed: Vec<u8>,
    bytes_per_snp: usize,
    n_samples: usize,
    n_snps: usize,
    row_freq: Vec<f32>,
    row_flip: Vec<bool>,
}

impl PackedBedRsvd {
    #[inline]
    fn row_bytes(&self, snp_idx: usize) -> &[u8] {
        let start = snp_idx * self.bytes_per_snp;
        &self.packed[start..start + self.bytes_per_snp]
    }
}

#[inline]
fn decode_plink_2bit(row: &[u8], sample_idx: usize) -> u8 {
    let b = row[sample_idx / 4];
    (b >> ((sample_idx % 4) * 2)) & 0b11
}

#[inline]
fn centered_from_plink_code(code: u8, f2: f32, flip: bool) -> Option<f32> {
    let mut g = match code {
        0b00 => 0.0_f32,
        0b10 => 1.0_f32,
        0b11 => 2.0_f32,
        _ => return None, // 0b01 missing
    };
    if flip {
        g = 2.0_f32 - g;
    }
    Some(g - f2)
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

fn rsvd_tile_cols_env() -> usize {
    match std::env::var("JANUSX_ADMX_RSVD_TILE") {
        Ok(v) => v.trim().parse::<usize>().ok().unwrap_or(1024).max(1),
        Err(_) => 1024,
    }
}

fn try_build_packed_bed_backend(cfg: &StreamRsvdConfig) -> Result<Option<PackedBedRsvd>, String> {
    if !packed_bed_fast_enabled() {
        return Ok(None);
    }
    let Some(prefix) = detect_bed_prefix(&cfg.genotype_path) else {
        return Ok(None);
    };

    let samples = read_fam(&prefix)?;
    let n_samples = samples.len();
    if n_samples == 0 {
        return Err("no samples found in PLINK input".to_string());
    }
    let sites = read_bim(&prefix)?;
    let n_snps_total = sites.len();

    let bed_path = format!("{prefix}.bed");
    let mut file = File::open(&bed_path).map_err(|e| e.to_string())?;
    let mut header = [0u8; 3];
    file.read_exact(&mut header).map_err(|e| e.to_string())?;
    if header != [0x6C, 0x1B, 0x01] {
        return Err("Only SNP-major BED supported".to_string());
    }
    let bytes_per_snp = (n_samples + 3) / 4;
    let expected_payload = n_snps_total
        .checked_mul(bytes_per_snp)
        .ok_or_else(|| "BED payload size overflow".to_string())?;
    let mut payload = vec![0u8; expected_payload];
    file.read_exact(&mut payload).map_err(|e| e.to_string())?;

    let mut packed_kept: Vec<u8> = Vec::with_capacity(expected_payload);
    let mut row_freq: Vec<f32> = Vec::with_capacity(n_snps_total);
    let mut row_flip: Vec<bool> = Vec::with_capacity(n_snps_total);
    let n_samples_f64 = n_samples as f64;

    for snp_idx in 0..n_snps_total {
        let site = &sites[snp_idx];
        if cfg.snps_only && !is_simple_snp_alleles(&site.ref_allele, &site.alt_allele) {
            continue;
        }
        let row = &payload[snp_idx * bytes_per_snp..(snp_idx + 1) * bytes_per_snp];

        let mut non_missing: usize = 0;
        let mut alt_sum: f64 = 0.0;
        for l in 0..n_samples {
            let code = decode_plink_2bit(row, l);
            match code {
                0b00 => {
                    non_missing += 1;
                }
                0b10 => {
                    non_missing += 1;
                    alt_sum += 1.0;
                }
                0b11 => {
                    non_missing += 1;
                    alt_sum += 2.0;
                }
                _ => {}
            }
        }

        let miss_rate = 1.0_f64 - (non_missing as f64 / n_samples_f64);
        if miss_rate > cfg.missing_rate as f64 {
            continue;
        }
        if non_missing == 0 {
            if cfg.maf > 0.0 {
                continue;
            }
            packed_kept.extend_from_slice(row);
            row_freq.push(0.0);
            row_flip.push(false);
            continue;
        }

        let p = alt_sum / (2.0 * non_missing as f64);
        let flip = p > 0.5;
        let p_minor = if flip { 1.0 - p } else { p };
        if p_minor < cfg.maf as f64 {
            continue;
        }
        packed_kept.extend_from_slice(row);
        row_freq.push(p_minor as f32);
        row_flip.push(flip);
    }

    let n_snps = row_freq.len();
    Ok(Some(PackedBedRsvd {
        packed: packed_kept,
        bytes_per_snp,
        n_samples,
        n_snps,
        row_freq,
        row_flip,
    }))
}

fn compute_a_omega_packed(
    backend: &PackedBedRsvd,
    omega: &[f32], // (n, kp)
    kp: usize,
) -> Result<Vec<f32>, String> {
    let m = backend.n_snps;
    let n = backend.n_samples;
    if omega.len() != n * kp {
        return Err("omega shape mismatch in packed A*Omega".to_string());
    }
    let mut out = vec![0.0_f32; m * kp];
    out.par_chunks_mut(kp).enumerate().for_each(|(i, out_row)| {
        let row = backend.row_bytes(i);
        let f2 = 2.0_f32 * backend.row_freq[i];
        let flip = backend.row_flip[i];
        for l in 0..n {
            let code = decode_plink_2bit(row, l);
            if let Some(centered) = centered_from_plink_code(code, f2, flip) {
                let omega_row = &omega[l * kp..(l + 1) * kp];
                for j in 0..kp {
                    out_row[j] += centered * omega_row[j];
                }
            }
        }
    });
    Ok(out)
}

fn compute_at_omega_packed(
    backend: &PackedBedRsvd,
    omega: &[f32], // (m, kp)
    kp: usize,
    tile_cols: usize,
) -> Result<Vec<f32>, String> {
    let m = backend.n_snps;
    let n = backend.n_samples;
    if omega.len() != m * kp {
        return Err("omega shape mismatch in packed A^T*Omega".to_string());
    }
    let mut out = vec![0.0_f32; n * kp];
    let tile = tile_cols.max(1).min(n);
    for block_start in (0..n).step_by(tile) {
        let block_len = (n - block_start).min(tile);
        let blk = (0..m)
            .into_par_iter()
            .fold(
                || vec![0.0_f32; block_len * kp],
                |mut local, i| {
                    let row = backend.row_bytes(i);
                    let f2 = 2.0_f32 * backend.row_freq[i];
                    let flip = backend.row_flip[i];
                    let omega_row = &omega[i * kp..(i + 1) * kp];
                    for off in 0..block_len {
                        let l = block_start + off;
                        let code = decode_plink_2bit(row, l);
                        if let Some(centered) = centered_from_plink_code(code, f2, flip) {
                            let dst = &mut local[off * kp..(off + 1) * kp];
                            for j in 0..kp {
                                dst[j] += centered * omega_row[j];
                            }
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
    Ok(out)
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

fn random_omega(m: usize, kp: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega = vec![0.0_f32; m * kp];
    for v in omega.iter_mut() {
        *v = rng.sample::<f64, _>(StandardNormal) as f32;
    }
    omega
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

    let mut gram = vec![0.0_f64; cols * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        for i in 0..cols {
            let xi = row[i] as f64;
            for j in 0..=i {
                gram[i * cols + j] += xi * (row[j] as f64);
            }
        }
    }
    for i in 0..cols {
        for j in 0..i {
            gram[j * cols + i] = gram[i * cols + j];
        }
    }

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

    let mut u = vec![0.0_f32; rows * cols]; // row-major
    for r in 0..rows {
        let x_row = &x[r * cols..(r + 1) * cols];
        let u_row = &mut u[r * cols..(r + 1) * cols];
        for c in 0..cols {
            let mut acc = 0.0_f64;
            for t in 0..cols {
                acc += (x_row[t] as f64) * (v[t * cols + c] as f64);
            }
            let inv = if s[c] > 1e-12 {
                1.0_f64 / (s[c] as f64)
            } else {
                0.0_f64
            };
            u_row[c] = (acc * inv) as f32;
        }
    }
    Ok((u, s, v))
}

fn compute_at_omega_stream(
    cfg: &StreamRsvdConfig,
    omega: &[f32],    // (m, kp)
    row_freq: &[f32], // (m,)
    m: usize,
    n: usize,
    kp: usize,
) -> Result<Vec<f32>, String> {
    if omega.len() != m * kp {
        return Err("omega shape mismatch in streaming A^T*Omega".to_string());
    }
    if row_freq.len() != m {
        return Err("row_freq length mismatch in streaming A^T*Omega".to_string());
    }

    let mut y = vec![0.0_f32; n * kp]; // (n, kp)
    let (seen, seen_n) = for_each_processed_row(cfg, |idx, row| {
        if idx >= m {
            return Err("streamed row count exceeded expected m in A^T*Omega".to_string());
        }
        let omega_row = &omega[idx * kp..(idx + 1) * kp];
        let f2 = 2.0_f32 * row_freq[idx];
        for l in 0..n {
            let gv = row[l];
            if gv < 0.0 || !gv.is_finite() {
                continue;
            }
            let centered = gv - f2;
            let y_row = &mut y[l * kp..(l + 1) * kp];
            for j in 0..kp {
                y_row[j] += centered * omega_row[j];
            }
        }
        Ok(())
    })?;
    if seen_n != n || seen != m {
        return Err(format!(
            "A^T*Omega pass row mismatch: seen={seen}, expected={m}, n_seen={seen_n}, n_expected={n}"
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
    let (seen, seen_n) = for_each_processed_row(cfg, |idx, row| {
        if idx >= m {
            return Err("streamed row count exceeded expected m in A*Omega".to_string());
        }
        let f2 = 2.0_f32 * row_freq[idx];
        let out_row = &mut out[idx * kp..(idx + 1) * kp];
        for j in 0..kp {
            let mut acc = 0.0_f32;
            for l in 0..n {
                let gv = row[l];
                if gv < 0.0 || !gv.is_finite() {
                    continue;
                }
                acc += (gv - f2) * omega[l * kp + j];
            }
            out_row[j] = acc;
        }
        Ok(())
    })?;
    if seen_n != n || seen != m {
        return Err(format!(
            "A*Omega pass row mismatch: seen={seen}, expected={m}, n_seen={seen_n}, n_expected={n}"
        ));
    }
    Ok(out)
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
    mmap_window_mb=0
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
    };

    let packed_backend = try_build_packed_bed_backend(&cfg).map_err(PyRuntimeError::new_err)?;
    let tile_cols = rsvd_tile_cols_env();

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
    let kp = (k_eff + 10).max(20).min(m.max(1));

    let omega = random_omega(m, kp, seed);
    let mut y = if let Some(backend) = packed_backend.as_ref() {
        compute_at_omega_packed(backend, &omega, kp, tile_cols)
    } else {
        compute_at_omega_stream(&cfg, &omega, &row_freq, m, n, kp)
    }
    .map_err(PyRuntimeError::new_err)?;
    let (mut q, _, _) = thin_svd_from_tall(&y, n, kp).map_err(PyRuntimeError::new_err)?;

    let mut sk = vec![0.0_f32; kp];
    let mut alpha = 0.0_f32;
    for it in 0..power {
        let g_small = if let Some(backend) = packed_backend.as_ref() {
            compute_a_omega_packed(backend, &q, kp)
        } else {
            compute_a_omega_stream(&cfg, &q, &row_freq, m, n, kp)
        }
        .map_err(PyRuntimeError::new_err)?;
        y = if let Some(backend) = packed_backend.as_ref() {
            compute_at_omega_packed(backend, &g_small, kp, tile_cols)
        } else {
            compute_at_omega_stream(&cfg, &g_small, &row_freq, m, n, kp)
        }
        .map_err(PyRuntimeError::new_err)?;
        for idx in 0..(n * kp) {
            y[idx] -= alpha * q[idx];
        }
        let (q_new, s_y, _) = thin_svd_from_tall(&y, n, kp).map_err(PyRuntimeError::new_err)?;
        q = q_new;

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

    let g_small = if let Some(backend) = packed_backend.as_ref() {
        compute_a_omega_packed(backend, &q, kp)
    } else {
        compute_a_omega_stream(&cfg, &q, &row_freq, m, n, kp)
    }
    .map_err(PyRuntimeError::new_err)?;
    let (u_small, s_all, _) =
        thin_svd_from_tall(&g_small, m, kp).map_err(PyRuntimeError::new_err)?;

    let mut eigvals = vec![0.0_f32; k_eff];
    for i in 0..k_eff {
        eigvals[i] = s_all[i] * s_all[i];
    }
    let mut eigvecs = vec![0.0_f32; m * k_eff];
    for r in 0..m {
        let src = &u_small[r * kp..(r + 1) * kp];
        let dst = &mut eigvecs[r * k_eff..(r + 1) * k_eff];
        dst.copy_from_slice(&src[..k_eff]);
    }

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
    mmap_window_mb=0
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
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>)> {
    if k == 0 {
        return Err(PyRuntimeError::new_err("k must be > 0"));
    }
    if !(0.0..=0.5).contains(&maf) {
        return Err(PyRuntimeError::new_err("maf must be within [0, 0.5]"));
    }
    if !(0.0..=1.0).contains(&missing_rate) {
        return Err(PyRuntimeError::new_err("missing_rate must be within [0, 1]"));
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
    };

    let packed_backend = try_build_packed_bed_backend(&cfg).map_err(PyRuntimeError::new_err)?;
    let tile_cols = rsvd_tile_cols_env();

    let mut row_freq: Vec<f32> = Vec::new();
    let mut varsum: f64 = 0.0;
    let (m, n) = if let Some(backend) = packed_backend.as_ref() {
        row_freq = backend.row_freq.clone();
        for &freq in row_freq.iter() {
            varsum += 2.0_f64 * (freq as f64) * (1.0_f64 - freq as f64);
        }
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
            varsum += 2.0_f64 * (freq as f64) * (1.0_f64 - freq as f64);
            row_freq.push(freq);
            Ok(())
        })
        .map_err(PyRuntimeError::new_err)?
    };

    if m == 0 {
        return Err(PyRuntimeError::new_err(
            "no SNPs passed filtering in streaming RSVD",
        ));
    }
    if n == 0 {
        return Err(PyRuntimeError::new_err("no samples available for streaming RSVD"));
    }
    if !(varsum.is_finite() && varsum > 0.0) {
        return Err(PyRuntimeError::new_err(
            "invalid scaling denominator in streaming RSVD (varsum <= 0)",
        ));
    }

    let k_eff = k.min(n);
    let kp = (k_eff + 10).max(20).min(m.max(1));

    let omega = random_omega(m, kp, seed);
    let mut y = if let Some(backend) = packed_backend.as_ref() {
        compute_at_omega_packed(backend, &omega, kp, tile_cols)
    } else {
        compute_at_omega_stream(&cfg, &omega, &row_freq, m, n, kp)
    }
    .map_err(PyRuntimeError::new_err)?;
    let (mut q, _, _) = thin_svd_from_tall(&y, n, kp).map_err(PyRuntimeError::new_err)?;

    let mut sk = vec![0.0_f32; kp];
    let mut alpha = 0.0_f32;
    for it in 0..power {
        let g_small = if let Some(backend) = packed_backend.as_ref() {
            compute_a_omega_packed(backend, &q, kp)
        } else {
            compute_a_omega_stream(&cfg, &q, &row_freq, m, n, kp)
        }
        .map_err(PyRuntimeError::new_err)?;
        y = if let Some(backend) = packed_backend.as_ref() {
            compute_at_omega_packed(backend, &g_small, kp, tile_cols)
        } else {
            compute_at_omega_stream(&cfg, &g_small, &row_freq, m, n, kp)
        }
        .map_err(PyRuntimeError::new_err)?;
        for idx in 0..(n * kp) {
            y[idx] -= alpha * q[idx];
        }
        let (q_new, s_y, _) = thin_svd_from_tall(&y, n, kp).map_err(PyRuntimeError::new_err)?;
        q = q_new;

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

    let g_small = if let Some(backend) = packed_backend.as_ref() {
        compute_a_omega_packed(backend, &q, kp)
    } else {
        compute_a_omega_stream(&cfg, &q, &row_freq, m, n, kp)
    }
    .map_err(PyRuntimeError::new_err)?;
    let (_, s_all, v_small) = thin_svd_from_tall(&g_small, m, kp).map_err(PyRuntimeError::new_err)?;

    let mut eigvals = vec![0.0_f32; k_eff];
    // Align PCA eigenvalue scaling with the GRM pathway in python/janusx/script/pca.py.
    // For sample-side PCA: lambda = sigma^2 / varsum, where
    // varsum = sum_i 2p_i(1-p_i) over retained variants.
    let scale = varsum as f32;
    for i in 0..k_eff {
        eigvals[i] = (s_all[i] * s_all[i]) / scale;
    }

    // Sample-side eigenvectors/right singular vectors: q @ v_small[:, :k_eff]
    let mut eigvecs_sample = vec![0.0_f32; n * k_eff];
    for r in 0..n {
        let q_row = &q[r * kp..(r + 1) * kp];
        let out_row = &mut eigvecs_sample[r * k_eff..(r + 1) * k_eff];
        for c in 0..k_eff {
            let mut acc = 0.0_f64;
            for t in 0..kp {
                acc += (q_row[t] as f64) * (v_small[t * kp + c] as f64);
            }
            out_row[c] = acc as f32;
        }
    }

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
    eval_slice.copy_from_slice(&eigvals);
    evec_slice.copy_from_slice(&eigvecs_sample);
    Ok((eval_arr, evec_arr))
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
