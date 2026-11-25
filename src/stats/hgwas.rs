//! Haploid / multi-parent ancestry GWAS (hGWAS) over streaming `.jhibd` archives.
//!
//! At each genomic bin `b`, local founder ancestry dosage `Z_b` (`n x P_b`, sparse)
//! is modeled as a random effect (Wei & Xu 2016 *Genetics* / Xiao et al. 2016 *New Phytologist*):
//!
//! ```text
//! y = X beta + g + Z_b u_b + eps,
//! g ~ N(0, sigma_g^2 K),   u_b ~ N(0, sigma_b^2 I),   eps ~ N(0, sigma_e^2 I)
//! ```
//!
//! Two exact likelihood-ratio test (LRT) modes are provided (no Wald test):
//! - **FVLMM** (`exact_lmm = false`): fixes the background polygenic variance ratio
//!   `lambda_g = sigma_g^2 / sigma_e^2` from the null model (`H_0: sigma_b^2 = 0`),
//!   precomputes the exact original-space projection `M_0 = U P_0 U^T` and `v_0 = U P_0 y_rot`
//!   once, and maintains `(r_b = Z_b^T v_0, A_b = Z_b^T M_0 Z_b, B_b = M_0 Z_b)` via
//!   fused rank-1/rank-2 sweep-line delta updates across adjacent Union-of-Breakpoints bins.
//! - **LMM** (`exact_lmm = true`): re-estimates all three variance components
//!   `(sigma_g^2, sigma_b^2, sigma_e^2)` per bin via profile REML over `(lambda_g, tau)`.
//!   When `G * n^2 * 8 <= 256 MiB`, all `G` precomputed background grid points use the same
//!   O(k * n) original-space incremental update as FVLMM (100% mathematically identical to
//!   recomputing `Z_rot^T W_g Z_rot` per bin), with zero-allocation Brent refinement over
//!   incrementally maintained `Z_rot = U^T Z`. Larger sample sizes fall back automatically
//!   to the low-memory `Z_rot`-only grid path.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::blas::{
    cblas_dgemm_dispatch, BlasThreadGuard, CblasInt, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS,
};
use crate::brent::brent_minimize_with_init;
use crate::eigh::symmetric_eigh_f64_row_major;
use crate::haploid::ancestry::ChromosomeAncestryBinsSparse;
use crate::haploid::sparse_io::{
    inspect_sparse_archive_meta, read_ancestry_segments_archive, AncestrySegmentsArchive,
};
use crate::haploid::types::HaploidIbdMode;
use crate::linalg::{
    chi2_sf_df1, cholesky_gram_inplace, cholesky_logdet, cholesky_solve_into,
    validate_fixed_effect_design,
};

const MIN_DIAG_FLOOR: f64 = 1e-12;
const MIN_RSS_FLOOR: f64 = 1e-30;
const TAU_LOG_LOW: f64 = -18.0;
const TAU_LOG_HIGH: f64 = 18.0;
const DEFAULT_BIN_BATCH_COLS: usize = 512;
/// Number of bins processed per independent incremental chunk (resets floating-point
/// drift periodically while keeping chunk-init overhead < 1% of scan time).
const CHUNK_BINS: usize = 1024;
/// Maximum `p x p` size handled by the zero-allocation inline Jacobi eigensolver.
const INLINE_JACOBI_MAX_P: usize = 64;
/// Memory budget for LMM multi-grid original-space `M_g` matrices (256 MiB).
const LMM_MULTI_GRID_MAX_BYTES: usize = 256 * 1024 * 1024;

/// Output statistics for a single bin.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HgwasBinResult {
    pub(crate) n_parents: usize,
    pub(crate) sigma_g2: f64,
    pub(crate) sigma_b2: f64,
    pub(crate) sigma_e2: f64,
    pub(crate) h2_bin: f64,
    pub(crate) lrt: f64,
    pub(crate) plrt: f64,
}

/// Compact sweep-line representation of one chromosome's Union-of-Breakpoints bins.
///
/// Memory footprint is O(n_segments + n_chunks * n_samples) instead of O(n_bins * n_samples)
/// (~28 MB vs ~7.8 GB on CUBIC 476k bins), and each bin step processes only the O(k)
/// active sample transitions instead of O(n) two-pointer bin diffs.
#[derive(Clone, Debug)]
struct CompactChromEvents {
    chrom_idx: u16,
    bin_starts_bp: Vec<u64>,
    /// Length `n_bins + 1`: delta entries for bin `b` lie in `delta_ptr[b]..delta_ptr[b+1]`.
    /// Always empty for chunk-start bins (`b % CHUNK_BINS == 0`), which initialize from
    /// `chunk_init_*`. Entries within each bin are sorted by `(sample, parent)`.
    delta_ptr: Vec<u64>,
    delta_sample: Vec<u32>,
    delta_parent: Vec<u32>,
    delta_old: Vec<f32>,
    delta_new: Vec<f32>,
    /// Length `n_chunks + 1`: initial active `(sample, parent, dosage)` entries
    /// for chunk `c` (bin `c * CHUNK_BINS`), sorted by `(sample, parent)`.
    chunk_init_ptr: Vec<u64>,
    chunk_init_sample: Vec<u32>,
    chunk_init_parent: Vec<u32>,
    chunk_init_dosage: Vec<f32>,
}

/// Compact in-memory hGWAS archive reusable across traits and models (`-fvlmm`/`-lmm`)
/// without re-reading or re-decompressing `.jhibd` from disk.
#[derive(Clone, Debug)]
pub(crate) struct CompactHgwasArchive {
    pub(crate) version: u32,
    pub(crate) target_sample_ids: Vec<String>,
    pub(crate) reference_sample_ids: Vec<String>,
    pub(crate) chrom_names: Vec<String>,
    chrom_events: Vec<CompactChromEvents>,
    pub(crate) total_bins: usize,
}

/// Null-model REML quantities evaluated at a given `lambda_g = sigma_g^2 / sigma_e^2`.
#[derive(Clone, Debug)]
struct BackgroundState {
    lambda_g: f64,
    /// `w[i] = 1.0 / (lambda_g * s[i] + 1.0)`
    w: Vec<f64>,
    /// `log|D| = sum_i ln(lambda_g * s[i] + 1.0)`
    logdet_d: f64,
    /// `V_x = W X_rot L_x^{-T}` stored in `(q x n)` row-major so `V_x * z_col` is contiguous.
    vx_t: Vec<f64>,
    /// `log|X^T W X|`
    logdet_xwx: f64,
    /// `P_0 y = W y - W X (X^T W X)^{-1} X^T W y` (length `n`).
    p0_y: Vec<f64>,
    /// `RSS_0 = y^T P_0 y`.
    rss0: f64,
    /// Null REML log-likelihood at `tau = 0`.
    reml_ll0: f64,
    /// `sigma_e2_0 = RSS_0 / (n - q)`.
    sigma_e2_0: f64,
}

/// Exact 50:50 mixture p-value for boundary variance-component LRT (`H_0: sigma_b^2 = 0`).
#[inline]
pub(crate) fn variance_component_lrt_pvalue(lrt: f64) -> f64 {
    if !lrt.is_finite() || lrt <= 0.0 {
        return 1.0;
    }
    let p = 0.5 * chi2_sf_df1(lrt);
    if p.is_finite() {
        p.clamp(f64::MIN_POSITIVE, 1.0)
    } else {
        1.0
    }
}

fn build_sample_remap(
    n_archive_samples: usize,
    sample_indices: Option<&[i64]>,
) -> Result<(Vec<i32>, usize), String> {
    if let Some(sel) = sample_indices {
        if sel.is_empty() {
            return Err("sample_indices must not be empty".to_string());
        }
        let mut remap = vec![-1i32; n_archive_samples];
        for (new_idx, &raw_idx) in sel.iter().enumerate() {
            if raw_idx < 0 || (raw_idx as usize) >= n_archive_samples {
                return Err(format!(
                    "sample_indices[{}]={} out of range 0..{}",
                    new_idx, raw_idx, n_archive_samples
                ));
            }
            let slot = &mut remap[raw_idx as usize];
            if *slot >= 0 {
                return Err(format!(
                    "duplicate sample index {} in sample_indices",
                    raw_idx
                ));
            }
            *slot = new_idx as i32;
        }
        Ok((remap, sel.len()))
    } else {
        let remap = (0..n_archive_samples).map(|i| i as i32).collect();
        Ok((remap, n_archive_samples))
    }
}

fn validate_incremental_sparse_bins(
    chrom_bins: &[ChromosomeAncestryBinsSparse],
    n_archive_samples: usize,
    p_total: usize,
) -> Result<(), String> {
    for cb in chrom_bins {
        let n_bins = cb.bin_starts_bp.len();
        if n_bins == 0 {
            continue;
        }
        if cb.bin_ptr.len() != n_bins + 1 {
            return Err(format!(
                "invalid .jhibd bin index: chromosome {} has {} bins but {} bin_ptr entries",
                cb.chrom_idx,
                n_bins,
                cb.bin_ptr.len()
            ));
        }
        if cb.sample_idx.len() != cb.parent_idx.len() || cb.sample_idx.len() != cb.dosage.len() {
            return Err(format!(
                "invalid .jhibd sparse columns on chromosome {}",
                cb.chrom_idx
            ));
        }
        if cb.n_target_samples > n_archive_samples {
            return Err(format!(
                "invalid .jhibd sample count: chromosome {} declares {}, archive has {}",
                cb.chrom_idx, cb.n_target_samples, n_archive_samples
            ));
        }
        if cb.bin_ptr[0] != 0 {
            return Err(format!(
                "invalid .jhibd bin index: chromosome {} does not start at entry 0",
                cb.chrom_idx
            ));
        }

        for b in 0..n_bins {
            let start = usize::try_from(cb.bin_ptr[b])
                .map_err(|_| format!("bin_ptr overflow on chromosome {}", cb.chrom_idx))?;
            let end = usize::try_from(cb.bin_ptr[b + 1])
                .map_err(|_| format!("bin_ptr overflow on chromosome {}", cb.chrom_idx))?;
            if start > end || end > cb.sample_idx.len() {
                return Err(format!(
                    "invalid .jhibd bin range on chromosome {} bin {}",
                    cb.chrom_idx, b
                ));
            }

            let mut previous: Option<(u32, u32)> = None;
            for pos in start..end {
                let sample = cb.sample_idx[pos];
                let parent = cb.parent_idx[pos];
                if sample as usize >= cb.n_target_samples || parent as usize >= p_total {
                    return Err(format!(
                        "out-of-range ancestry entry on chromosome {} bin {}",
                        cb.chrom_idx, b
                    ));
                }
                let dosage = cb.dosage[pos];
                if !dosage.is_finite() || dosage < 0.0 {
                    return Err(format!(
                        "invalid dosage on chromosome {} bin {}",
                        cb.chrom_idx, b
                    ));
                }
                let key = (sample, parent);
                if previous.is_some_and(|prev| key <= prev) {
                    return Err(format!(
                        "ancestry entries must be strictly sorted and unique on chromosome {} bin {}",
                        cb.chrom_idx, b
                    ));
                }
                previous = Some(key);
            }
        }
    }
    Ok(())
}

/// Build `CompactChromEvents` for one chromosome directly from its `AncestrySegment`s
/// via a sweep-line over segment start/end breakpoints, matching
/// `build_union_breakpoint_sparse_matrix` identically (including diploid 0.5/1.0
/// coalescence and skipping all-uncalled empty gaps).
fn build_compact_events_from_segments(
    chrom_idx: u16,
    n_target_samples: usize,
    n_parents: usize,
    target_is_inbred: &[bool],
    chrom_segs: &[&crate::haploid::types::AncestrySegment],
) -> CompactChromEvents {
    let mut out = CompactChromEvents {
        chrom_idx,
        bin_starts_bp: Vec::new(),
        delta_ptr: vec![0u64],
        delta_sample: Vec::new(),
        delta_parent: Vec::new(),
        delta_old: Vec::new(),
        delta_new: Vec::new(),
        chunk_init_ptr: vec![0u64],
        chunk_init_sample: Vec::new(),
        chunk_init_parent: Vec::new(),
        chunk_init_dosage: Vec::new(),
    };
    if chrom_segs.is_empty() || n_target_samples == 0 || n_parents == 0 {
        return out;
    }

    // Raw endpoint events: `(bp, sample, parent, signed_weight)`.
    let mut events: Vec<(u64, u32, u32, f32)> = Vec::with_capacity(chrom_segs.len() * 2);
    for seg in chrom_segs {
        if seg.end_bp < seg.start_bp {
            continue;
        }
        let s = seg.target_sample as usize;
        let p = seg.parent_sample as usize;
        if s >= n_target_samples || p >= n_parents {
            continue;
        }
        let w = if target_is_inbred.get(s).copied().unwrap_or(true) {
            1.0f32
        } else {
            0.5f32
        };
        events.push((seg.start_bp, seg.target_sample, seg.parent_sample, w));
        events.push((
            seg.end_bp.saturating_add(1),
            seg.target_sample,
            seg.parent_sample,
            -w,
        ));
    }
    if events.len() < 2 {
        return out;
    }
    events.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    let mut active_map: HashMap<(u32, u32), f32> = HashMap::with_capacity(n_target_samples * 2);
    // Tracks net transitions across any skipped empty (all-uncalled) gaps so the next
    // emitted bin receives the exact diff relative to the previous emitted bin.
    let mut pending_from_prev_emitted: HashMap<(u32, u32), (f32, f32)> = HashMap::new();
    let mut snap_buf: Vec<(u32, u32, f32)> = Vec::with_capacity(n_target_samples * 2);
    let mut delta_buf: Vec<(u32, u32, f32, f32)> = Vec::with_capacity(64);
    let mut has_emitted_any = false;

    let mut idx = 0usize;
    let total_ev = events.len();
    while idx < total_ev {
        let bp = events[idx].0;
        let mut group_end = idx + 1;
        while group_end < total_ev && events[group_end].0 == bp {
            group_end += 1;
        }

        // Coalesce consecutive events with identical `(sample, parent)` at `bp`.
        delta_buf.clear();
        let mut j = idx;
        while j < group_end {
            let s = events[j].1;
            let p = events[j].2;
            let mut net = events[j].3;
            j += 1;
            while j < group_end && events[j].1 == s && events[j].2 == p {
                net += events[j].3;
                j += 1;
            }
            if net.abs() <= 1e-6 {
                continue;
            }
            let d_old = active_map.get(&(s, p)).copied().unwrap_or(0.0);
            let mut d_new = d_old + net;
            if d_new <= 1e-6 {
                d_new = 0.0;
                active_map.remove(&(s, p));
            } else {
                active_map.insert((s, p), d_new);
            }
            if (d_new - d_old).abs() > 1e-6 {
                delta_buf.push((s, p, d_old, d_new));
            }
        }

        let has_next_bp = group_end < total_ev;
        if has_next_bp && !active_map.is_empty() {
            let b = out.bin_starts_bp.len();
            out.bin_starts_bp.push(bp);
            if b % CHUNK_BINS == 0 {
                snap_buf.clear();
                for (&(s, p), &d) in &active_map {
                    snap_buf.push((s, p, d));
                }
                snap_buf.sort_unstable_by_key(|&(s, p, _)| (s, p));
                for &(s, p, d) in &snap_buf {
                    out.chunk_init_sample.push(s);
                    out.chunk_init_parent.push(p);
                    out.chunk_init_dosage.push(d);
                }
                out.chunk_init_ptr.push(out.chunk_init_sample.len() as u64);
                out.delta_ptr.push(out.delta_sample.len() as u64);
                pending_from_prev_emitted.clear();
            } else if pending_from_prev_emitted.is_empty() {
                for &(s, p, d_old, d_new) in &delta_buf {
                    out.delta_sample.push(s);
                    out.delta_parent.push(p);
                    out.delta_old.push(d_old);
                    out.delta_new.push(d_new);
                }
                out.delta_ptr.push(out.delta_sample.len() as u64);
            } else {
                for &(s, p, d_old, d_new) in &delta_buf {
                    pending_from_prev_emitted
                        .entry((s, p))
                        .and_modify(|entry| entry.1 = d_new)
                        .or_insert((d_old, d_new));
                }
                let mut merged: Vec<(u32, u32, f32, f32)> = pending_from_prev_emitted
                    .drain()
                    .filter_map(|((s, p), (d_old, d_new))| {
                        if (d_new - d_old).abs() > 1e-6 {
                            Some((s, p, d_old, d_new))
                        } else {
                            None
                        }
                    })
                    .collect();
                merged.sort_unstable_by_key(|&(s, p, _, _)| (s, p));
                for (s, p, d_old, d_new) in merged {
                    out.delta_sample.push(s);
                    out.delta_parent.push(p);
                    out.delta_old.push(d_old);
                    out.delta_new.push(d_new);
                }
                out.delta_ptr.push(out.delta_sample.len() as u64);
            }
            has_emitted_any = true;
        } else if has_emitted_any {
            // Skipped empty gap bin: record transitions relative to last emitted bin.
            for &(s, p, d_old, d_new) in &delta_buf {
                pending_from_prev_emitted
                    .entry((s, p))
                    .and_modify(|entry| entry.1 = d_new)
                    .or_insert((d_old, d_new));
            }
        }

        idx = group_end;
    }

    out
}

/// Fallback converter from pre-expanded `ChromosomeAncestryBinsSparse` (only used
/// for synthetic bin-only `JHIBD01` test fixtures where `segments` is empty).
fn build_compact_events_from_sparse_bins(cb: &ChromosomeAncestryBinsSparse) -> CompactChromEvents {
    let n_bins = cb.bin_starts_bp.len();
    let mut out = CompactChromEvents {
        chrom_idx: cb.chrom_idx,
        bin_starts_bp: cb.bin_starts_bp.clone(),
        delta_ptr: Vec::with_capacity(n_bins + 1),
        delta_sample: Vec::new(),
        delta_parent: Vec::new(),
        delta_old: Vec::new(),
        delta_new: Vec::new(),
        chunk_init_ptr: vec![0u64],
        chunk_init_sample: Vec::new(),
        chunk_init_parent: Vec::new(),
        chunk_init_dosage: Vec::new(),
    };
    out.delta_ptr.push(0u64);

    let mut prev_start = 0usize;
    let mut prev_end = 0usize;
    for b in 0..n_bins {
        let cur_start = cb.bin_ptr[b] as usize;
        let cur_end = cb.bin_ptr[b + 1] as usize;
        if b % CHUNK_BINS == 0 {
            for pos in cur_start..cur_end {
                let d = cb.dosage[pos];
                if d != 0.0 && d.is_finite() {
                    out.chunk_init_sample.push(cb.sample_idx[pos]);
                    out.chunk_init_parent.push(cb.parent_idx[pos]);
                    out.chunk_init_dosage.push(d);
                }
            }
            out.chunk_init_ptr.push(out.chunk_init_sample.len() as u64);
            out.delta_ptr.push(out.delta_sample.len() as u64);
        } else {
            let mut i = prev_start;
            let mut j = cur_start;
            while i < prev_end && j < cur_end {
                let ki = (cb.sample_idx[i], cb.parent_idx[i]);
                let kj = (cb.sample_idx[j], cb.parent_idx[j]);
                if ki == kj {
                    let d_old = cb.dosage[i];
                    let d_new = cb.dosage[j];
                    if (d_new - d_old).abs() > 1e-6 {
                        out.delta_sample.push(ki.0);
                        out.delta_parent.push(ki.1);
                        out.delta_old.push(d_old);
                        out.delta_new.push(d_new);
                    }
                    i += 1;
                    j += 1;
                } else if ki < kj {
                    let d_old = cb.dosage[i];
                    if d_old != 0.0 {
                        out.delta_sample.push(ki.0);
                        out.delta_parent.push(ki.1);
                        out.delta_old.push(d_old);
                        out.delta_new.push(0.0);
                    }
                    i += 1;
                } else {
                    let d_new = cb.dosage[j];
                    if d_new != 0.0 {
                        out.delta_sample.push(kj.0);
                        out.delta_parent.push(kj.1);
                        out.delta_old.push(0.0);
                        out.delta_new.push(d_new);
                    }
                    j += 1;
                }
            }
            while i < prev_end {
                let d_old = cb.dosage[i];
                if d_old != 0.0 {
                    out.delta_sample.push(cb.sample_idx[i]);
                    out.delta_parent.push(cb.parent_idx[i]);
                    out.delta_old.push(d_old);
                    out.delta_new.push(0.0);
                }
                i += 1;
            }
            while j < cur_end {
                let d_new = cb.dosage[j];
                if d_new != 0.0 {
                    out.delta_sample.push(cb.sample_idx[j]);
                    out.delta_parent.push(cb.parent_idx[j]);
                    out.delta_old.push(0.0);
                    out.delta_new.push(d_new);
                }
                j += 1;
            }
            out.delta_ptr.push(out.delta_sample.len() as u64);
        }
        prev_start = cur_start;
        prev_end = cur_end;
    }
    out
}

impl CompactHgwasArchive {
    pub(crate) fn from_segments_archive(anc: AncestrySegmentsArchive) -> Result<Self, String> {
        let n_chroms = anc.chrom_names.len();
        let n_samples = anc.target_sample_ids.len();
        let n_parents = anc.reference_sample_ids.len().max(1);

        let chrom_events: Vec<CompactChromEvents> = if !anc.segments.is_empty() {
            let mut segs_by_chrom: Vec<Vec<&crate::haploid::types::AncestrySegment>> =
                vec![Vec::new(); n_chroms];
            for seg in &anc.segments {
                let c = seg.chrom_idx as usize;
                if c < n_chroms {
                    segs_by_chrom[c].push(seg);
                }
            }
            segs_by_chrom
                .into_par_iter()
                .enumerate()
                .map(|(c, chrom_segs)| {
                    build_compact_events_from_segments(
                        c as u16,
                        n_samples,
                        n_parents,
                        &anc.target_is_inbred,
                        &chrom_segs,
                    )
                })
                .collect()
        } else {
            validate_incremental_sparse_bins(&anc.fallback_chrom_bins, n_samples, n_parents)?;
            anc.fallback_chrom_bins
                .par_iter()
                .map(build_compact_events_from_sparse_bins)
                .collect()
        };

        let total_bins: usize = chrom_events.iter().map(|ce| ce.bin_starts_bp.len()).sum();
        Ok(Self {
            version: anc.version,
            target_sample_ids: anc.target_sample_ids,
            reference_sample_ids: anc.reference_sample_ids,
            chrom_names: anc.chrom_names,
            chrom_events,
            total_bins,
        })
    }

    pub(crate) fn open(path: &str) -> Result<Self, String> {
        let anc = read_ancestry_segments_archive(Path::new(path))?;
        Self::from_segments_archive(anc)
    }
}

/// Evaluate background REML state at a fixed `lambda_g = sigma_g^2 / sigma_e^2 >= 0`.
fn evaluate_background_state(
    s: &[f64],
    x_rot: &[f64],
    y_rot: &[f64],
    n: usize,
    q: usize,
    lambda_g: f64,
) -> Option<BackgroundState> {
    let df = (n.checked_sub(q)?) as f64;
    if df <= 0.0 || !lambda_g.is_finite() || lambda_g < 0.0 {
        return None;
    }

    let mut w = vec![0.0_f64; n];
    let mut logdet_d = 0.0_f64;
    for i in 0..n {
        let d = (lambda_g * s[i] + 1.0).max(MIN_DIAG_FLOOR);
        w[i] = 1.0 / d;
        logdet_d += d.ln();
    }

    let mut xwx = vec![0.0_f64; q * q];
    let mut xwy = vec![0.0_f64; q];
    let mut ywy = 0.0_f64;
    for i in 0..n {
        let wi = w[i];
        let yi = y_rot[i];
        ywy += wi * yi * yi;
        let xi = &x_rot[i * q..(i + 1) * q];
        for a in 0..q {
            let wxa = wi * xi[a];
            xwy[a] += wxa * yi;
            let row = a * q;
            for b in 0..=a {
                xwx[row + b] += wxa * xi[b];
            }
        }
    }
    for a in 0..q {
        for b in (a + 1)..q {
            xwx[a * q + b] = xwx[b * q + a];
        }
    }

    cholesky_gram_inplace(&mut xwx, q)?;
    let logdet_xwx = cholesky_logdet(&xwx, q);
    let mut beta = vec![0.0_f64; q];
    cholesky_solve_into(&xwx, q, &xwy, &mut beta);

    let quad_x: f64 = xwy.iter().zip(&beta).map(|(a, b)| a * b).sum();
    let rss0 = (ywy - quad_x).max(MIN_RSS_FLOOR);
    let sigma_e2_0 = (rss0 / df).max(MIN_RSS_FLOOR);

    let mut p0_y = vec![0.0_f64; n];
    let mut vx_t = vec![0.0_f64; q * n];
    let mut tmp = vec![0.0_f64; q];
    for i in 0..n {
        let wi = w[i];
        let xi = &x_rot[i * q..(i + 1) * q];
        let mut xb = 0.0_f64;
        for a in 0..q {
            xb += xi[a] * beta[a];
            let mut sum = wi * xi[a];
            let l_row = a * q;
            for k in 0..a {
                sum -= xwx[l_row + k] * tmp[k];
            }
            let val = sum / xwx[l_row + a];
            tmp[a] = val;
            vx_t[a * n + i] = val;
        }
        p0_y[i] = wi * (y_rot[i] - xb);
    }

    let const_term = df * ((2.0 * std::f64::consts::PI).ln() + 1.0);
    let reml_ll0 = -0.5 * (const_term + df * sigma_e2_0.ln() + logdet_d + logdet_xwx);

    Some(BackgroundState {
        lambda_g,
        w,
        logdet_d,
        vx_t,
        logdet_xwx,
        p0_y,
        rss0,
        reml_ll0,
        sigma_e2_0,
    })
}

/// Zero-allocation in-place background evaluator reused across all Brent steps in LMM.
fn evaluate_background_state_into(
    s: &[f64],
    x_rot: &[f64],
    y_rot: &[f64],
    n: usize,
    q: usize,
    lambda_g: f64,
    out: &mut BackgroundState,
    xwx: &mut [f64],
    xwy: &mut [f64],
    beta: &mut [f64],
    tmp: &mut [f64],
) -> bool {
    let Some(df_u) = n.checked_sub(q) else {
        return false;
    };
    let df = df_u as f64;
    if df <= 0.0 || !lambda_g.is_finite() || lambda_g < 0.0 {
        return false;
    }

    out.lambda_g = lambda_g;
    let mut logdet_d = 0.0_f64;
    for i in 0..n {
        let d = (lambda_g * s[i] + 1.0).max(MIN_DIAG_FLOOR);
        out.w[i] = 1.0 / d;
        logdet_d += d.ln();
    }
    out.logdet_d = logdet_d;

    xwx.fill(0.0);
    xwy.fill(0.0);
    let mut ywy = 0.0_f64;
    for i in 0..n {
        let wi = out.w[i];
        let yi = y_rot[i];
        ywy += wi * yi * yi;
        let xi = &x_rot[i * q..(i + 1) * q];
        for a in 0..q {
            let wxa = wi * xi[a];
            xwy[a] += wxa * yi;
            let row = a * q;
            for b in 0..=a {
                xwx[row + b] += wxa * xi[b];
            }
        }
    }
    for a in 0..q {
        for b in (a + 1)..q {
            xwx[a * q + b] = xwx[b * q + a];
        }
    }

    if cholesky_gram_inplace(xwx, q).is_none() {
        return false;
    }
    let logdet_xwx = cholesky_logdet(xwx, q);
    out.logdet_xwx = logdet_xwx;
    cholesky_solve_into(xwx, q, xwy, beta);

    let quad_x: f64 = xwy.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
    let rss0 = (ywy - quad_x).max(MIN_RSS_FLOOR);
    let sigma_e2_0 = (rss0 / df).max(MIN_RSS_FLOOR);
    out.rss0 = rss0;
    out.sigma_e2_0 = sigma_e2_0;

    for i in 0..n {
        let wi = out.w[i];
        let xi = &x_rot[i * q..(i + 1) * q];
        let mut xb = 0.0_f64;
        for a in 0..q {
            xb += xi[a] * beta[a];
            let mut sum = wi * xi[a];
            let l_row = a * q;
            for k in 0..a {
                sum -= xwx[l_row + k] * tmp[k];
            }
            let val = sum / xwx[l_row + a];
            tmp[a] = val;
            out.vx_t[a * n + i] = val;
        }
        out.p0_y[i] = wi * (y_rot[i] - xb);
    }

    let const_term = df * ((2.0 * std::f64::consts::PI).ln() + 1.0);
    out.reml_ll0 = -0.5 * (const_term + df * sigma_e2_0.ln() + logdet_d + logdet_xwx);
    true
}

/// Optimize the null model (`tau = 0`) over `lambda_g` in `[10^low, 10^high]`
/// (plus boundary `lambda_g = 0`) to obtain the exact REML null log-likelihood.
fn fit_null_reml_background(
    s: &[f64],
    x_rot: &[f64],
    y_rot: &[f64],
    n: usize,
    q: usize,
    log10_low: f64,
    log10_high: f64,
    init_lambda_g: Option<f64>,
) -> Result<BackgroundState, String> {
    let mut best = evaluate_background_state(s, x_rot, y_rot, n, q, 0.0)
        .ok_or_else(|| "failed to evaluate null REML at lambda_g=0".to_string())?;

    let mut best_log10 = log10_low;
    let n_grid = 25usize;
    for g in 0..=n_grid {
        let frac = (g as f64) / (n_grid as f64);
        let lg = log10_low + frac * (log10_high - log10_low);
        let lam = 10.0_f64.powf(lg);
        if let Some(st) = evaluate_background_state(s, x_rot, y_rot, n, q, lam) {
            if st.reml_ll0 > best.reml_ll0 {
                best = st;
                best_log10 = lg;
            }
        }
    }

    if let Some(init_lam) = init_lambda_g {
        if init_lam.is_finite() && init_lam > 0.0 {
            let lg = init_lam.log10().clamp(log10_low, log10_high);
            if let Some(st) = evaluate_background_state(s, x_rot, y_rot, n, q, 10.0_f64.powf(lg)) {
                if st.reml_ll0 > best.reml_ll0 {
                    best = st;
                    best_log10 = lg;
                }
            }
        }
    }

    let step = (log10_high - log10_low) / (n_grid as f64);
    let b_lo = (best_log10 - step).max(log10_low);
    let b_hi = (best_log10 + step).min(log10_high);
    let (opt_lg, _) = brent_minimize_with_init(
        |lg| {
            evaluate_background_state(s, x_rot, y_rot, n, q, 10.0_f64.powf(lg))
                .map(|st| -st.reml_ll0)
                .unwrap_or(f64::INFINITY)
        },
        b_lo,
        b_hi,
        1e-6,
        60,
        Some(best_log10),
    );
    if let Some(st) = evaluate_background_state(s, x_rot, y_rot, n, q, 10.0_f64.powf(opt_lg)) {
        if st.reml_ll0 > best.reml_ll0 {
            best = st;
        }
    }
    Ok(best)
}

/// Fast zero-allocation cyclic Jacobi eigensolver for small symmetric `p x p` matrices
/// (`p <= INLINE_JACOBI_MAX_P`). Overwrites `a` with diagonal eigenvalues and writes
/// eigenvectors as columns in row-major `v` (`v[row * p + col]`).
fn jacobi_sym_eigh_inplace(a: &mut [f64], v: &mut [f64], p: usize) {
    for val in v.iter_mut() {
        *val = 0.0;
    }
    for i in 0..p {
        v[i * p + i] = 1.0;
    }
    if p <= 1 {
        return;
    }

    let mut diag_scale = 0.0_f64;
    for i in 0..p {
        let d = a[i * p + i].abs();
        if d > diag_scale {
            diag_scale = d;
        }
    }
    let eps = 1e-13 * diag_scale.max(1.0);
    let max_sweeps = 24usize;

    for _sweep in 0..max_sweeps {
        let mut max_off = 0.0_f64;
        for row in 0..p {
            let r_off = row * p;
            for col in (row + 1)..p {
                let val = a[r_off + col].abs();
                if val > max_off {
                    max_off = val;
                }
            }
        }
        if max_off <= eps {
            break;
        }
        let thresh = 0.2 * max_off;
        for i in 0..p {
            for j in (i + 1)..p {
                let aij = a[i * p + j];
                if aij.abs() <= thresh && max_off > 10.0 * eps {
                    continue;
                }
                if aij.abs() <= 1e-30 {
                    continue;
                }
                let aii = a[i * p + i];
                let ajj = a[j * p + j];
                let tau = (ajj - aii) / (2.0 * aij);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[i * p + i] = aii - t * aij;
                a[j * p + j] = ajj + t * aij;
                a[i * p + j] = 0.0;
                a[j * p + i] = 0.0;

                for k in 0..p {
                    if k != i && k != j {
                        let aki = a[k * p + i];
                        let akj = a[k * p + j];
                        let new_ki = c * aki - s * akj;
                        let new_kj = s * aki + c * akj;
                        a[k * p + i] = new_ki;
                        a[i * p + k] = new_ki;
                        a[k * p + j] = new_kj;
                        a[j * p + k] = new_kj;
                    }
                }
                for k in 0..p {
                    let vki = v[k * p + i];
                    let vkj = v[k * p + j];
                    v[k * p + i] = c * vki - s * vkj;
                    v[k * p + j] = s * vki + c * vkj;
                }
            }
        }
    }
}

/// Scratch workspace reused across bins inside each Rayon worker thread (zero
/// per-bin heap allocations in both FVLMM and LMM).
struct TauSolverScratch {
    a_work: Vec<f64>,
    v_work: Vec<f64>,
    mu: Vec<f64>,
    u2: Vec<f64>,
    r_sub: Vec<f64>,
    v_qp: Vec<f64>,
    wz_col: Vec<f64>,
}

impl TauSolverScratch {
    fn with_capacity(p_max: usize, n: usize, q: usize) -> Self {
        Self {
            a_work: vec![0.0_f64; p_max * p_max],
            v_work: vec![0.0_f64; p_max * p_max],
            mu: Vec::with_capacity(p_max),
            u2: Vec::with_capacity(p_max),
            r_sub: Vec::with_capacity(p_max),
            v_qp: vec![0.0_f64; q * p_max],
            wz_col: vec![0.0_f64; n],
        }
    }
}

/// Reusable scratch buffers for zero-allocation LMM Brent refinement.
struct LmmBrentScratch {
    bg: BackgroundState,
    xwx: Vec<f64>,
    xwy: Vec<f64>,
    beta: Vec<f64>,
    tmp: Vec<f64>,
    tau: TauSolverScratch,
}

impl LmmBrentScratch {
    fn new(p_max: usize, n: usize, q: usize) -> Self {
        Self {
            bg: BackgroundState {
                lambda_g: 0.0,
                w: vec![0.0_f64; n],
                logdet_d: 0.0,
                vx_t: vec![0.0_f64; q * n],
                logdet_xwx: 0.0,
                p0_y: vec![0.0_f64; n],
                rss0: 0.0,
                reml_ll0: 0.0,
                sigma_e2_0: 0.0,
            },
            xwx: vec![0.0_f64; q * q],
            xwy: vec![0.0_f64; q],
            beta: vec![0.0_f64; q],
            tmp: vec![0.0_f64; q],
            tau: TauSolverScratch::with_capacity(p_max, n, q),
        }
    }
}

/// Exact 1D REML optimization over `tau = sigma_b^2 / sigma_e^2 >= 0` given the
/// `p x p` symmetric PSD matrix `A = Z^T P_0 Z` (stored in `scratch.a_work[..p*p]`)
/// and vector `r = Z^T P_0 y` (stored in `scratch.r_sub[..p]`).
fn solve_tau_from_a_r(
    bg: &BackgroundState,
    df: f64,
    p: usize,
    scratch: &mut TauSolverScratch,
) -> (f64, f64, f64) {
    if p == 0 {
        return (bg.reml_ll0, 0.0, bg.sigma_e2_0);
    }

    let mut trace_a = 0.0_f64;
    let mut r_norm2 = 0.0_f64;
    for c in 0..p {
        trace_a += scratch.a_work[c * p + c];
        r_norm2 += scratch.r_sub[c] * scratch.r_sub[c];
    }
    if !trace_a.is_finite() || trace_a <= 1e-14 || !r_norm2.is_finite() || r_norm2 <= 1e-30 {
        return (bg.reml_ll0, 0.0, bg.sigma_e2_0);
    }

    scratch.mu.clear();
    scratch.u2.clear();
    let mu_tol = 1e-12 * trace_a.max(1.0);
    let mut rss_proj_max = 0.0_f64;

    if p <= INLINE_JACOBI_MAX_P {
        let a_slice = &mut scratch.a_work[..p * p];
        let v_slice = &mut scratch.v_work[..p * p];
        jacobi_sym_eigh_inplace(a_slice, v_slice, p);
        for j in 0..p {
            let mj = a_slice[j * p + j];
            if mj > mu_tol && mj.is_finite() {
                let mut uj = 0.0_f64;
                for c in 0..p {
                    uj += v_slice[c * p + j] * scratch.r_sub[c];
                }
                let uj2 = uj * uj;
                scratch.mu.push(mj);
                scratch.u2.push(uj2);
                rss_proj_max += uj2 / mj;
            }
        }
    } else {
        let (eigvals, eigvecs, _) = match symmetric_eigh_f64_row_major(&scratch.a_work[..p * p], p)
        {
            Ok(res) => res,
            Err(_) => return (bg.reml_ll0, 0.0, bg.sigma_e2_0),
        };
        for j in 0..p {
            let mj = eigvals[j];
            if mj > mu_tol && mj.is_finite() {
                let mut uj = 0.0_f64;
                for c in 0..p {
                    uj += eigvecs[c * p + j] * scratch.r_sub[c];
                }
                let uj2 = uj * uj;
                scratch.mu.push(mj);
                scratch.u2.push(uj2);
                rss_proj_max += uj2 / mj;
            }
        }
    }

    if scratch.mu.is_empty() {
        return (bg.reml_ll0, 0.0, bg.sigma_e2_0);
    }
    let rss_perp = (bg.rss0 - rss_proj_max).max(0.0);

    // Exact mathematical certificate for boundary optimum tau = 0:
    if rss_perp > MIN_RSS_FLOOR {
        let mut max_ratio = 0.0_f64;
        for (&mj, &uj2) in scratch.mu.iter().zip(&scratch.u2) {
            let ratio = uj2 / mj;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
        if (df / rss_perp) * max_ratio <= 1.0 {
            return (bg.reml_ll0, 0.0, bg.sigma_e2_0);
        }
    }

    let const_term = df * ((2.0 * std::f64::consts::PI).ln() + 1.0);
    let base_logdet = bg.logdet_d + bg.logdet_xwx;
    let mu_sl = &scratch.mu;
    let u2_sl = &scratch.u2;

    let eval_reml_derivs = |tau: f64| -> (f64, f64, f64, f64) {
        let mut logdet_m = 0.0_f64;
        let mut rss_var = 0.0_f64;
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        let mut r1 = 0.0_f64;
        let mut r2 = 0.0_f64;
        for (&mj, &uj2) in mu_sl.iter().zip(u2_sl.iter()) {
            let tm = tau * mj;
            let inv_d = 1.0 / (1.0 + tm);
            logdet_m += (1.0 + tm).ln();
            let qj = uj2 * inv_d;
            rss_var += qj / mj;
            let tj = tm * inv_d;
            s1 += tj;
            s2 += tj * inv_d;
            let r1j = tau * uj2 * inv_d * inv_d;
            r1 += r1j;
            r2 += r1j * (1.0 - 2.0 * tm * inv_d);
        }
        let rss = (rss_perp + rss_var).max(MIN_RSS_FLOOR);
        let sigma_e2 = (rss / df).max(MIN_RSS_FLOOR);
        let ll = -0.5 * (const_term + df * sigma_e2.ln() + base_logdet + logdet_m);
        let inv_rss = 1.0 / rss;
        let g = 0.5 * (df * r1 * inv_rss - s1);
        let h = 0.5 * (df * (r2 * inv_rss + (r1 * inv_rss) * (r1 * inv_rss)) - s2);
        (ll, sigma_e2, g, h)
    };

    const PROBES: [f64; 9] = [-14.0, -10.0, -6.0, -3.0, -1.0, 1.0, 3.0, 7.0, 13.0];
    let mut best_ll = bg.reml_ll0;
    let mut best_tau = 0.0_f64;
    let mut best_se2 = bg.sigma_e2_0;
    let mut best_idx: Option<usize> = None;
    let mut probe_g = [0.0_f64; 9];

    for (k, &lt) in PROBES.iter().enumerate() {
        let tau = lt.exp();
        let (ll, se2, g, _) = eval_reml_derivs(tau);
        probe_g[k] = g;
        if ll > best_ll {
            best_ll = ll;
            best_tau = tau;
            best_se2 = se2;
            best_idx = Some(k);
        }
    }

    let score_slope = (df / bg.rss0) * u2_sl.iter().sum::<f64>() - mu_sl.iter().sum::<f64>();
    if best_idx.is_none() && score_slope <= 0.0 {
        let mut any_pos = false;
        for (k, &g) in probe_g.iter().enumerate() {
            if g > 0.0 {
                best_idx = Some(k);
                any_pos = true;
                break;
            }
        }
        if !any_pos {
            return (bg.reml_ll0, 0.0, bg.sigma_e2_0);
        }
    }

    let (mut lo, mut hi, mut x) = if let Some(k) = best_idx {
        let lo_k = if k == 0 { TAU_LOG_LOW } else { PROBES[k - 1] };
        let hi_k = if k + 1 >= PROBES.len() {
            TAU_LOG_HIGH
        } else {
            PROBES[k + 1]
        };
        (lo_k, hi_k, PROBES[k])
    } else {
        (TAU_LOG_LOW, PROBES[0], 0.5 * (TAU_LOG_LOW + PROBES[0]))
    };

    for _iter in 0..16 {
        let tau = x.exp();
        let (ll, se2, g, h) = eval_reml_derivs(tau);
        if ll > best_ll {
            best_ll = ll;
            best_tau = tau;
            best_se2 = se2;
        }
        if g.abs() < 1e-9 || (hi - lo) < 1e-8 {
            break;
        }
        if g > 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let next_x = if h < -1e-12 {
            let cand = x - g / h;
            if cand > lo && cand < hi {
                cand
            } else {
                0.5 * (lo + hi)
            }
        } else {
            0.5 * (lo + hi)
        };
        if (next_x - x).abs() < 1e-8 {
            let t_fin = next_x.exp();
            let (ll_f, se2_f, _, _) = eval_reml_derivs(t_fin);
            if ll_f > best_ll {
                best_ll = ll_f;
                best_tau = t_fin;
                best_se2 = se2_f;
            }
            break;
        }
        x = next_x;
    }

    (best_ll, best_tau, best_se2)
}

/// Zero-allocation version of `optimize_tau_for_background_cols` operating on
/// active parent indices into contiguous `(p_total x n)` `z_rot`.
fn optimize_tau_for_background_zrot(
    bg: &BackgroundState,
    z_rot: &[f64],
    active_parents: &[usize],
    n: usize,
    q: usize,
    scratch: &mut TauSolverScratch,
) -> (f64, f64, f64) {
    let p = active_parents.len();
    if p == 0 {
        return (bg.reml_ll0, 0.0, bg.sigma_e2_0);
    }
    let df = (n - q) as f64;
    let w = &bg.w;
    let p0_y = &bg.p0_y;

    scratch.r_sub.clear();
    scratch.r_sub.resize(p, 0.0);

    for (c, &pc) in active_parents.iter().enumerate() {
        let zc = &z_rot[pc * n..(pc + 1) * n];
        let mut rc = 0.0_f64;
        for i in 0..n {
            let zci = zc[i];
            rc += zci * p0_y[i];
            scratch.wz_col[i] = w[i] * zci;
        }
        scratch.r_sub[c] = rc;

        for a in 0..q {
            let vxa = &bg.vx_t[a * n..(a + 1) * n];
            let mut dot = 0.0_f64;
            for i in 0..n {
                dot += vxa[i] * zc[i];
            }
            scratch.v_qp[a * p + c] = dot;
        }

        let row = c * p;
        for (d, &pd) in active_parents.iter().enumerate().take(c + 1) {
            let zd = &z_rot[pd * n..(pd + 1) * n];
            let mut zwz_cd = 0.0_f64;
            for i in 0..n {
                zwz_cd += scratch.wz_col[i] * zd[i];
            }
            let mut sub = 0.0_f64;
            for a in 0..q {
                sub += scratch.v_qp[a * p + c] * scratch.v_qp[a * p + d];
            }
            let val = zwz_cd - sub;
            scratch.a_work[row + d] = val;
            scratch.a_work[d * p + c] = val;
        }
    }

    solve_tau_from_a_r(bg, df, p, scratch)
}

/// Build the exact original-space projection matrix `M_0 = U P_0 U^T` (`n x n`, row-major)
/// and vector `v_0 = U P_0 y_rot` (`n`) for a fixed `BackgroundState`.
fn build_original_space_projection(
    u_mat: &[f64],
    bg: &BackgroundState,
    n: usize,
    q: usize,
    threads: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut uw = vec![0.0_f64; n * n];
    for i in 0..n {
        let u_row = &u_mat[i * n..(i + 1) * n];
        let uw_row = &mut uw[i * n..(i + 1) * n];
        for j in 0..n {
            uw_row[j] = u_row[j] * bg.w[j];
        }
    }
    let mut m0 = vec![0.0_f64; n * n];
    {
        let _blas_guard = BlasThreadGuard::enter(threads.max(1));
        unsafe {
            cblas_dgemm_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                n as CblasInt,
                n as CblasInt,
                n as CblasInt,
                1.0,
                uw.as_ptr(),
                n as CblasInt,
                u_mat.as_ptr(),
                n as CblasInt,
                0.0,
                m0.as_mut_ptr(),
                n as CblasInt,
            );
        }
    }

    if q > 0 {
        let mut u_vx = vec![0.0_f64; n * q];
        {
            let _blas_guard = BlasThreadGuard::enter(threads.max(1));
            unsafe {
                cblas_dgemm_dispatch(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_TRANS,
                    n as CblasInt,
                    q as CblasInt,
                    n as CblasInt,
                    1.0,
                    u_mat.as_ptr(),
                    n as CblasInt,
                    bg.vx_t.as_ptr(),
                    n as CblasInt,
                    0.0,
                    u_vx.as_mut_ptr(),
                    q as CblasInt,
                );
                cblas_dgemm_dispatch(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_TRANS,
                    n as CblasInt,
                    n as CblasInt,
                    q as CblasInt,
                    -1.0,
                    u_vx.as_ptr(),
                    q as CblasInt,
                    u_vx.as_ptr(),
                    q as CblasInt,
                    1.0,
                    m0.as_mut_ptr(),
                    n as CblasInt,
                );
            }
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (m0[i * n + j] + m0[j * n + i]);
            m0[i * n + j] = avg;
            m0[j * n + i] = avg;
        }
    }

    let mut v0 = vec![0.0_f64; n];
    for i in 0..n {
        let u_row = &u_mat[i * n..(i + 1) * n];
        let mut sum = 0.0_f64;
        for j in 0..n {
            sum += u_row[j] * bg.p0_y[j];
        }
        v0[i] = sum;
    }

    (m0, v0)
}

#[inline]
fn build_result_from_opt(
    p_active: usize,
    best_ll: f64,
    best_tau: f64,
    best_se2: f64,
    best_lg: f64,
    null_reml_ll0: f64,
    c_k: f64,
    c_b: f64,
) -> HgwasBinResult {
    let lrt = (2.0 * (best_ll - null_reml_ll0)).max(0.0);
    let plrt = variance_component_lrt_pvalue(lrt);
    let sigma_e2 = best_se2;
    let sigma_g2 = best_lg * sigma_e2;
    let sigma_b2 = best_tau * sigma_e2;
    let denom = sigma_g2 * c_k + sigma_b2 * c_b + sigma_e2;
    let h2_bin = if denom > 0.0 && denom.is_finite() {
        ((sigma_b2 * c_b) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    HgwasBinResult {
        n_parents: p_active,
        sigma_g2,
        sigma_b2,
        sigma_e2,
        h2_bin,
        lrt,
        plrt,
    }
}

#[inline]
fn format_bin_result_row(buf: &mut String, chrom_name: &str, start_bp: u64, res: &HgwasBinResult) {
    let _ = writeln!(
        buf,
        "{}\t{}\t{}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6}\t{:.6}\t{:.6e}",
        chrom_name,
        start_bp,
        res.n_parents,
        res.sigma_g2,
        res.sigma_b2,
        res.sigma_e2,
        res.h2_bin,
        res.lrt,
        res.plrt
    );
}

/// Mutable incremental projection state `(r = Z^T v, A = Z^T M Z, B = M Z)` for
/// one fixed background `M = U P U^T`, `v = U P y_rot`.
struct IncrementalProjState {
    r_full: Vec<f64>,
    a_full: Vec<f64>,
    b_full: Vec<f64>,
}

impl IncrementalProjState {
    fn new(p_total: usize, n: usize) -> Self {
        Self {
            r_full: vec![0.0_f64; p_total],
            a_full: vec![0.0_f64; p_total * p_total],
            b_full: vec![0.0_f64; p_total * n],
        }
    }

    fn init_from_entries(
        &mut self,
        entries: &[(usize, usize, f64)],
        p_nnz: &[u32],
        p_total: usize,
        n: usize,
        m0: &[f64],
        v0: &[f64],
    ) {
        self.r_full.fill(0.0);
        self.a_full.fill(0.0);
        self.b_full.fill(0.0);
        for &(si, p_raw, d) in entries {
            self.r_full[p_raw] += d * v0[si];
            let m0_row = &m0[si * n..(si + 1) * n];
            let bp = &mut self.b_full[p_raw * n..(p_raw + 1) * n];
            for j in 0..n {
                bp[j] += d * m0_row[j];
            }
        }
        for &(si, p_raw, d) in entries {
            let row = p_raw * p_total;
            for p2 in 0..p_total {
                if p_nnz[p2] > 0 {
                    self.a_full[row + p2] += d * self.b_full[p2 * n + si];
                }
            }
        }
        for p1 in 0..p_total {
            for p2 in (p1 + 1)..p_total {
                let avg = 0.5 * (self.a_full[p1 * p_total + p2] + self.a_full[p2 * p_total + p1]);
                self.a_full[p1 * p_total + p2] = avg;
                self.a_full[p2 * p_total + p1] = avg;
            }
        }
    }

    #[inline]
    fn apply_single_delta(
        &mut self,
        s_new: usize,
        p_raw: usize,
        delta: f64,
        p_total: usize,
        n: usize,
        m0: &[f64],
        v0: &[f64],
    ) {
        self.r_full[p_raw] += delta * v0[s_new];
        let m0_row = &m0[s_new * n..(s_new + 1) * n];
        for p2 in 0..p_total {
            let d_bq = delta * self.b_full[p2 * n + s_new];
            self.a_full[p_raw * p_total + p2] += d_bq;
            self.a_full[p2 * p_total + p_raw] += d_bq;
        }
        self.a_full[p_raw * p_total + p_raw] += delta * delta * m0_row[s_new];
        let bp = &mut self.b_full[p_raw * n..(p_raw + 1) * n];
        for j in 0..n {
            bp[j] += delta * m0_row[j];
        }
    }

    /// Fused rank-2 update for a paired parent switch on the same sample:
    /// `p_dec` changes by `-delta` and `p_inc` changes by `+delta`.
    /// Reads `m0_row` once instead of twice, halving memory bandwidth.
    #[inline]
    fn apply_paired_switch(
        &mut self,
        s_new: usize,
        p_dec: usize,
        p_inc: usize,
        delta: f64,
        p_total: usize,
        n: usize,
        m0: &[f64],
        v0: &[f64],
    ) {
        let dv = delta * v0[s_new];
        self.r_full[p_dec] -= dv;
        self.r_full[p_inc] += dv;
        let m0_row = &m0[s_new * n..(s_new + 1) * n];
        for q in 0..p_total {
            let d_bq = delta * self.b_full[q * n + s_new];
            self.a_full[p_dec * p_total + q] -= d_bq;
            self.a_full[q * p_total + p_dec] -= d_bq;
            self.a_full[p_inc * p_total + q] += d_bq;
            self.a_full[q * p_total + p_inc] += d_bq;
        }
        let dm_diag = delta * delta * m0_row[s_new];
        self.a_full[p_dec * p_total + p_dec] += dm_diag;
        self.a_full[p_inc * p_total + p_inc] += dm_diag;
        self.a_full[p_dec * p_total + p_inc] -= dm_diag;
        self.a_full[p_inc * p_total + p_dec] -= dm_diag;

        if p_dec < p_inc {
            let (left, right) = self.b_full.split_at_mut(p_inc * n);
            let b_dec = &mut left[p_dec * n..(p_dec + 1) * n];
            let b_inc = &mut right[..n];
            for j in 0..n {
                let dm = delta * m0_row[j];
                b_dec[j] -= dm;
                b_inc[j] += dm;
            }
        } else {
            let (left, right) = self.b_full.split_at_mut(p_dec * n);
            let b_inc = &mut left[p_inc * n..(p_inc + 1) * n];
            let b_dec = &mut right[..n];
            for j in 0..n {
                let dm = delta * m0_row[j];
                b_dec[j] -= dm;
                b_inc[j] += dm;
            }
        }
    }

    #[inline]
    fn extract_active_into(
        &self,
        active_parents: &[usize],
        p_total: usize,
        scratch: &mut TauSolverScratch,
    ) {
        let p_active = active_parents.len();
        scratch.r_sub.clear();
        for (i_loc, &pi) in active_parents.iter().enumerate() {
            scratch.r_sub.push(self.r_full[pi]);
            let dst_row = i_loc * p_active;
            let src_row = pi * p_total;
            for (j_loc, &pj) in active_parents.iter().enumerate() {
                scratch.a_work[dst_row + j_loc] = self.a_full[src_row + pj];
            }
        }
    }
}

/// Apply remapped delta list to parent summary stats `(p_sum, p_sq_sum, p_nnz)`.
#[inline]
fn update_parent_stats(
    p_raw: usize,
    d_old: f64,
    d_new: f64,
    p_sum: &mut [f64],
    p_sq_sum: &mut [f64],
    p_nnz: &mut [u32],
) {
    let delta = d_new - d_old;
    p_sum[p_raw] += delta;
    p_sq_sum[p_raw] += d_new * d_new - d_old * d_old;
    if d_old == 0.0 && d_new != 0.0 {
        p_nnz[p_raw] = p_nnz[p_raw].saturating_add(1);
    } else if d_old != 0.0 && d_new == 0.0 {
        p_nnz[p_raw] = p_nnz[p_raw].saturating_sub(1);
    }
}

/// Apply a sorted slice of remapped deltas `(s_new, p_raw, d_old, d_new)` to `z_rot`
/// (`p_total x n`), fusing paired opposite switches on the same sample.
#[inline]
fn apply_deltas_to_zrot(
    deltas: &[(usize, usize, f64, f64)],
    z_rot: &mut [f64],
    u_mat: &[f64],
    n: usize,
) {
    let mut k = 0usize;
    while k < deltas.len() {
        let (s0, p0, o0, n0) = deltas[k];
        let d0 = n0 - o0;
        if k + 1 < deltas.len() {
            let (s1, p1, o1, n1) = deltas[k + 1];
            let d1 = n1 - o1;
            if s0 == s1 && p0 != p1 && (d0 + d1).abs() <= 1e-12 {
                let (p_dec, p_inc, delta) = if d0 < 0.0 { (p0, p1, d1) } else { (p1, p0, d0) };
                let u_row = &u_mat[s0 * n..(s0 + 1) * n];
                if p_dec < p_inc {
                    let (left, right) = z_rot.split_at_mut(p_inc * n);
                    let z_dec = &mut left[p_dec * n..(p_dec + 1) * n];
                    let z_inc = &mut right[..n];
                    for j in 0..n {
                        let du = delta * u_row[j];
                        z_dec[j] -= du;
                        z_inc[j] += du;
                    }
                } else {
                    let (left, right) = z_rot.split_at_mut(p_dec * n);
                    let z_inc = &mut left[p_inc * n..(p_inc + 1) * n];
                    let z_dec = &mut right[..n];
                    for j in 0..n {
                        let du = delta * u_row[j];
                        z_dec[j] -= du;
                        z_inc[j] += du;
                    }
                }
                k += 2;
                continue;
            }
        }
        let u_row = &u_mat[s0 * n..(s0 + 1) * n];
        let zp = &mut z_rot[p0 * n..(p0 + 1) * n];
        for j in 0..n {
            zp[j] += d0 * u_row[j];
        }
        k += 1;
    }
}

/// Apply a sorted slice of remapped deltas `(s_new, p_raw, d_old, d_new)` to one
/// `IncrementalProjState`, fusing paired opposite switches on the same sample.
#[inline]
fn apply_deltas_to_proj_state(
    deltas: &[(usize, usize, f64, f64)],
    state: &mut IncrementalProjState,
    p_total: usize,
    n: usize,
    m0: &[f64],
    v0: &[f64],
) {
    let mut k = 0usize;
    while k < deltas.len() {
        let (s0, p0, o0, n0) = deltas[k];
        let d0 = n0 - o0;
        if k + 1 < deltas.len() {
            let (s1, p1, o1, n1) = deltas[k + 1];
            let d1 = n1 - o1;
            if s0 == s1 && p0 != p1 && (d0 + d1).abs() <= 1e-12 {
                let (p_dec, p_inc, delta) = if d0 < 0.0 { (p0, p1, d1) } else { (p1, p0, d0) };
                state.apply_paired_switch(s0, p_dec, p_inc, delta, p_total, n, m0, v0);
                k += 2;
                continue;
            }
        }
        state.apply_single_delta(s0, p0, d0, p_total, n, m0, v0);
        k += 1;
    }
}

/// Process a chunk of bins `[b_start, b_end)` in **FVLMM** mode using sweep-line
/// checkpoints + fused paired-switch rank-1/rank-2 updates, returning formatted TSV.
#[allow(clippy::too_many_arguments)]
fn scan_chunk_fvlmm_events(
    ce: &CompactChromEvents,
    chrom_name: &str,
    chunk_idx: usize,
    b_start: usize,
    b_end: usize,
    sample_remap: &[i32],
    n: usize,
    q: usize,
    p_total: usize,
    c_k: f64,
    null_bg: &BackgroundState,
    m0: &[f64],
    v0: &[f64],
) -> String {
    let df = (n - q) as f64;
    let inv_n = 1.0 / (n as f64);
    let mut tsv_buf = String::with_capacity((b_end - b_start) * 96);

    let mut state = IncrementalProjState::new(p_total, n);
    let mut p_sum = vec![0.0_f64; p_total];
    let mut p_sq_sum = vec![0.0_f64; p_total];
    let mut p_nnz = vec![0u32; p_total];
    let mut active_parents = Vec::with_capacity(p_total);
    let mut init_entries: Vec<(usize, usize, f64)> = Vec::with_capacity(n);
    let mut deltas: Vec<(usize, usize, f64, f64)> = Vec::with_capacity(64);
    let mut scratch = TauSolverScratch::with_capacity(p_total, n, q);

    let mut last_res = HgwasBinResult {
        n_parents: 0,
        sigma_g2: null_bg.lambda_g * null_bg.sigma_e2_0,
        sigma_b2: 0.0,
        sigma_e2: null_bg.sigma_e2_0,
        h2_bin: 0.0,
        lrt: 0.0,
        plrt: 1.0,
    };

    for b in b_start..b_end {
        if b == b_start {
            init_entries.clear();
            let c0 = ce.chunk_init_ptr[chunk_idx] as usize;
            let c1 = ce.chunk_init_ptr[chunk_idx + 1] as usize;
            for pos in c0..c1 {
                let s_raw = ce.chunk_init_sample[pos] as usize;
                let p_raw = ce.chunk_init_parent[pos] as usize;
                if s_raw >= sample_remap.len() || p_raw >= p_total {
                    continue;
                }
                let s_new = sample_remap[s_raw];
                if s_new < 0 {
                    continue;
                }
                let d = ce.chunk_init_dosage[pos] as f64;
                if d == 0.0 || !d.is_finite() {
                    continue;
                }
                init_entries.push((s_new as usize, p_raw, d));
                p_sum[p_raw] += d;
                p_sq_sum[p_raw] += d * d;
                p_nnz[p_raw] = p_nnz[p_raw].saturating_add(1);
            }
            state.init_from_entries(&init_entries, &p_nnz, p_total, n, m0, v0);
        } else {
            deltas.clear();
            let d0 = ce.delta_ptr[b] as usize;
            let d1 = ce.delta_ptr[b + 1] as usize;
            for pos in d0..d1 {
                let s_raw = ce.delta_sample[pos] as usize;
                let p_raw = ce.delta_parent[pos] as usize;
                if s_raw >= sample_remap.len() || p_raw >= p_total {
                    continue;
                }
                let s_new = sample_remap[s_raw];
                if s_new < 0 {
                    continue;
                }
                let old_v = ce.delta_old[pos] as f64;
                let new_v = ce.delta_new[pos] as f64;
                if (new_v - old_v).abs() <= 1e-12 {
                    continue;
                }
                deltas.push((s_new as usize, p_raw, old_v, new_v));
                update_parent_stats(p_raw, old_v, new_v, &mut p_sum, &mut p_sq_sum, &mut p_nnz);
            }
            if deltas.is_empty() {
                format_bin_result_row(&mut tsv_buf, chrom_name, ce.bin_starts_bp[b], &last_res);
                continue;
            }
            apply_deltas_to_proj_state(&deltas, &mut state, p_total, n, m0, v0);
        }

        active_parents.clear();
        let mut centered_trace_sum = 0.0_f64;
        for p in 0..p_total {
            if p_nnz[p] > 0 {
                active_parents.push(p);
                let s1 = p_sum[p];
                let mean = s1 * inv_n;
                centered_trace_sum += (p_sq_sum[p] - s1 * mean).max(0.0);
            }
        }
        let p_active = active_parents.len();
        let c_b = centered_trace_sum * inv_n;

        if p_active == 0 || c_b <= 1e-14 {
            last_res = HgwasBinResult {
                n_parents: p_active,
                sigma_g2: null_bg.lambda_g * null_bg.sigma_e2_0,
                sigma_b2: 0.0,
                sigma_e2: null_bg.sigma_e2_0,
                h2_bin: 0.0,
                lrt: 0.0,
                plrt: 1.0,
            };
            format_bin_result_row(&mut tsv_buf, chrom_name, ce.bin_starts_bp[b], &last_res);
            continue;
        }

        state.extract_active_into(&active_parents, p_total, &mut scratch);
        let (best_ll, best_tau, best_se2) = solve_tau_from_a_r(null_bg, df, p_active, &mut scratch);
        last_res = build_result_from_opt(
            p_active,
            best_ll,
            best_tau,
            best_se2,
            null_bg.lambda_g,
            null_bg.reml_ll0,
            c_k,
            c_b,
        );
        format_bin_result_row(&mut tsv_buf, chrom_name, ce.bin_starts_bp[b], &last_res);
    }

    tsv_buf
}

/// Process a chunk of bins `[b_start, b_end)` in exact **LMM** mode.
///
/// When `grid_projs` is `Some` (fits in the 256 MiB budget), every precomputed grid
/// point (`null_bg` + `cached_grid`) uses O(k * n) original-space incremental updates
/// (identical to FVLMM), and `z_rot` is maintained incrementally solely for zero-allocation
/// Brent refinement when `best_tau > 0`. When `grid_projs` is `None`, falls back cleanly
/// to the low-memory `z_rot`-only grid evaluation.
#[allow(clippy::too_many_arguments)]
fn scan_chunk_lmm_events(
    ce: &CompactChromEvents,
    chrom_name: &str,
    chunk_idx: usize,
    b_start: usize,
    b_end: usize,
    sample_remap: &[i32],
    s: &[f64],
    x_rot: &[f64],
    y_rot: &[f64],
    u_mat: &[f64],
    n: usize,
    q: usize,
    p_total: usize,
    c_k: f64,
    null_bg: &BackgroundState,
    cached_grid: &[(f64, BackgroundState)],
    grid_projs: Option<&[(Vec<f64>, Vec<f64>)]>,
    grid_step: f64,
    log10_low: f64,
    log10_high: f64,
    max_iter: usize,
    tol: f64,
) -> String {
    let df = (n - q) as f64;
    let inv_n = 1.0 / (n as f64);
    let mut tsv_buf = String::with_capacity((b_end - b_start) * 96);

    let mut z_rot = vec![0.0_f64; p_total * n];
    let mut grid_states: Vec<IncrementalProjState> = match grid_projs {
        Some(projs) => (0..projs.len())
            .map(|_| IncrementalProjState::new(p_total, n))
            .collect(),
        None => Vec::new(),
    };
    let mut p_sum = vec![0.0_f64; p_total];
    let mut p_sq_sum = vec![0.0_f64; p_total];
    let mut p_nnz = vec![0u32; p_total];
    let mut active_parents = Vec::with_capacity(p_total);
    let mut init_entries: Vec<(usize, usize, f64)> = Vec::with_capacity(n);
    let mut deltas: Vec<(usize, usize, f64, f64)> = Vec::with_capacity(64);
    let mut scratch = TauSolverScratch::with_capacity(p_total, n, q);
    let mut brent_scratch = LmmBrentScratch::new(p_total, n, q);

    let mut last_res = HgwasBinResult {
        n_parents: 0,
        sigma_g2: null_bg.lambda_g * null_bg.sigma_e2_0,
        sigma_b2: 0.0,
        sigma_e2: null_bg.sigma_e2_0,
        h2_bin: 0.0,
        lrt: 0.0,
        plrt: 1.0,
    };

    for b in b_start..b_end {
        if b == b_start {
            init_entries.clear();
            let c0 = ce.chunk_init_ptr[chunk_idx] as usize;
            let c1 = ce.chunk_init_ptr[chunk_idx + 1] as usize;
            for pos in c0..c1 {
                let s_raw = ce.chunk_init_sample[pos] as usize;
                let p_raw = ce.chunk_init_parent[pos] as usize;
                if s_raw >= sample_remap.len() || p_raw >= p_total {
                    continue;
                }
                let s_new = sample_remap[s_raw];
                if s_new < 0 {
                    continue;
                }
                let d = ce.chunk_init_dosage[pos] as f64;
                if d == 0.0 || !d.is_finite() {
                    continue;
                }
                let si = s_new as usize;
                init_entries.push((si, p_raw, d));
                p_sum[p_raw] += d;
                p_sq_sum[p_raw] += d * d;
                p_nnz[p_raw] = p_nnz[p_raw].saturating_add(1);
                let u_row = &u_mat[si * n..(si + 1) * n];
                let zp = &mut z_rot[p_raw * n..(p_raw + 1) * n];
                for j in 0..n {
                    zp[j] += d * u_row[j];
                }
            }
            if let Some(projs) = grid_projs {
                for (st, (m_g, v_g)) in grid_states.iter_mut().zip(projs.iter()) {
                    st.init_from_entries(&init_entries, &p_nnz, p_total, n, m_g, v_g);
                }
            }
        } else {
            deltas.clear();
            let d0 = ce.delta_ptr[b] as usize;
            let d1 = ce.delta_ptr[b + 1] as usize;
            for pos in d0..d1 {
                let s_raw = ce.delta_sample[pos] as usize;
                let p_raw = ce.delta_parent[pos] as usize;
                if s_raw >= sample_remap.len() || p_raw >= p_total {
                    continue;
                }
                let s_new = sample_remap[s_raw];
                if s_new < 0 {
                    continue;
                }
                let old_v = ce.delta_old[pos] as f64;
                let new_v = ce.delta_new[pos] as f64;
                if (new_v - old_v).abs() <= 1e-12 {
                    continue;
                }
                deltas.push((s_new as usize, p_raw, old_v, new_v));
                update_parent_stats(p_raw, old_v, new_v, &mut p_sum, &mut p_sq_sum, &mut p_nnz);
            }
            if deltas.is_empty() {
                format_bin_result_row(&mut tsv_buf, chrom_name, ce.bin_starts_bp[b], &last_res);
                continue;
            }
            apply_deltas_to_zrot(&deltas, &mut z_rot, u_mat, n);
            if let Some(projs) = grid_projs {
                for (st, (m_g, v_g)) in grid_states.iter_mut().zip(projs.iter()) {
                    apply_deltas_to_proj_state(&deltas, st, p_total, n, m_g, v_g);
                }
            }
        }

        active_parents.clear();
        let mut centered_trace_sum = 0.0_f64;
        for p in 0..p_total {
            if p_nnz[p] > 0 {
                active_parents.push(p);
                let s1 = p_sum[p];
                let mean = s1 * inv_n;
                centered_trace_sum += (p_sq_sum[p] - s1 * mean).max(0.0);
            }
        }
        let p_active = active_parents.len();
        let c_b = centered_trace_sum * inv_n;

        if p_active == 0 || c_b <= 1e-14 {
            last_res = HgwasBinResult {
                n_parents: p_active,
                sigma_g2: null_bg.lambda_g * null_bg.sigma_e2_0,
                sigma_b2: 0.0,
                sigma_e2: null_bg.sigma_e2_0,
                h2_bin: 0.0,
                lrt: 0.0,
                plrt: 1.0,
            };
            format_bin_result_row(&mut tsv_buf, chrom_name, ce.bin_starts_bp[b], &last_res);
            continue;
        }

        // Step 1 + Step 2: Evaluate null_bg + cached_grid either via O(1) incremental
        // `(r_g, A_g)` extraction or via `z_rot` fallback.
        let (mut best_ll, mut best_tau, mut best_se2) = if grid_projs.is_some() {
            grid_states[0].extract_active_into(&active_parents, p_total, &mut scratch);
            solve_tau_from_a_r(null_bg, df, p_active, &mut scratch)
        } else {
            optimize_tau_for_background_zrot(null_bg, &z_rot, &active_parents, n, q, &mut scratch)
        };
        let mut best_lg = null_bg.lambda_g;
        let mut best_log10 = if null_bg.lambda_g > 0.0 {
            null_bg.lambda_g.log10().clamp(log10_low, log10_high)
        } else {
            log10_low
        };

        for (gi, (lg_log10, bg_cand)) in cached_grid.iter().enumerate() {
            let (ll, tau, se2) = if grid_projs.is_some() {
                grid_states[gi + 1].extract_active_into(&active_parents, p_total, &mut scratch);
                solve_tau_from_a_r(bg_cand, df, p_active, &mut scratch)
            } else {
                optimize_tau_for_background_zrot(
                    bg_cand,
                    &z_rot,
                    &active_parents,
                    n,
                    q,
                    &mut scratch,
                )
            };
            if ll > best_ll {
                best_ll = ll;
                best_tau = tau;
                best_se2 = se2;
                best_lg = bg_cand.lambda_g;
                if lg_log10.is_finite() {
                    best_log10 = *lg_log10;
                }
            }
        }

        // Step 3: Zero-allocation tight-bracket Brent refinement ONLY when a positive tau was found.
        if best_tau > 0.0 && best_ll > null_bg.reml_ll0 {
            let b_lo = (best_log10 - grid_step).max(log10_low);
            let b_hi = (best_log10 + grid_step).min(log10_high);
            let (opt_log10, _) = brent_minimize_with_init(
                |lg_log10| {
                    let lam = 10.0_f64.powf(lg_log10);
                    if evaluate_background_state_into(
                        s,
                        x_rot,
                        y_rot,
                        n,
                        q,
                        lam,
                        &mut brent_scratch.bg,
                        &mut brent_scratch.xwx,
                        &mut brent_scratch.xwy,
                        &mut brent_scratch.beta,
                        &mut brent_scratch.tmp,
                    ) {
                        -optimize_tau_for_background_zrot(
                            &brent_scratch.bg,
                            &z_rot,
                            &active_parents,
                            n,
                            q,
                            &mut brent_scratch.tau,
                        )
                        .0
                    } else {
                        f64::INFINITY
                    }
                },
                b_lo,
                b_hi,
                tol.max(1e-3),
                max_iter.clamp(8, 20),
                Some(best_log10),
            );
            let opt_lam = 10.0_f64.powf(opt_log10);
            if evaluate_background_state_into(
                s,
                x_rot,
                y_rot,
                n,
                q,
                opt_lam,
                &mut brent_scratch.bg,
                &mut brent_scratch.xwx,
                &mut brent_scratch.xwy,
                &mut brent_scratch.beta,
                &mut brent_scratch.tmp,
            ) {
                let (ll, tau, se2) = optimize_tau_for_background_zrot(
                    &brent_scratch.bg,
                    &z_rot,
                    &active_parents,
                    n,
                    q,
                    &mut brent_scratch.tau,
                );
                if ll > best_ll {
                    best_ll = ll;
                    best_tau = tau;
                    best_se2 = se2;
                    best_lg = opt_lam;
                }
            }
        }

        last_res = build_result_from_opt(
            p_active,
            best_ll,
            best_tau,
            best_se2,
            best_lg,
            null_bg.reml_ll0,
            c_k,
            c_b,
        );
        format_bin_result_row(&mut tsv_buf, chrom_name, ce.bin_starts_bp[b], &last_res);
    }

    tsv_buf
}

/// Core streaming ancestry GRM builder from `CompactHgwasArchive`.
///
/// `method == 1`: centered ancestry GRM `K = Z_c Z_c^T / (tr(Z_c Z_c^T) / n)`.
/// `method == 2`: standardized ancestry GRM `K = Z_s Z_s^T / m_eff`.
pub(crate) fn build_ancestry_grm_from_compact(
    archive: &CompactHgwasArchive,
    sample_indices: Option<&[i64]>,
    method: i32,
    threads: usize,
) -> Result<(Vec<f64>, usize, usize), String> {
    if method != 1 && method != 2 {
        return Err(format!("GRM method must be 1 or 2, got {}", method));
    }
    let (sample_remap, n) = build_sample_remap(archive.target_sample_ids.len(), sample_indices)?;
    if n == 0 {
        return Err("empty sample selection for ancestry GRM".to_string());
    }

    let p_total = archive.reference_sample_ids.len().max(1);
    let mut grm = vec![0.0_f64; n * n];
    let mut z_batch: Vec<f64> = Vec::with_capacity(DEFAULT_BIN_BATCH_COLS * n);
    let mut batch_cols = 0usize;
    let mut eff_bins = 0usize;
    let mut eff_cols = 0usize;
    let inv_n = 1.0 / (n as f64);

    let flush_batch = |z_colmajor: &mut Vec<f64>, cols: &mut usize, grm_acc: &mut [f64]| {
        if *cols == 0 {
            return;
        }
        let c = *cols;
        let _blas_guard = BlasThreadGuard::enter(threads.max(1));
        unsafe {
            cblas_dgemm_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                CBLAS_NO_TRANS,
                n as CblasInt,
                n as CblasInt,
                c as CblasInt,
                1.0,
                z_colmajor.as_ptr(),
                n as CblasInt,
                z_colmajor.as_ptr(),
                n as CblasInt,
                1.0,
                grm_acc.as_mut_ptr(),
                n as CblasInt,
            );
        }
        z_colmajor.clear();
        *cols = 0;
    };

    // Maintain current bin's dense `(p_total x n)` dosage matrix via O(k) deltas.
    let mut cur_z = vec![0.0_f64; p_total * n];
    let mut p_sum = vec![0.0_f64; p_total];
    let mut p_sq_sum = vec![0.0_f64; p_total];
    let mut p_nnz = vec![0u32; p_total];

    for ce in &archive.chrom_events {
        let n_bins = ce.bin_starts_bp.len();
        for b in 0..n_bins {
            if b % CHUNK_BINS == 0 {
                cur_z.fill(0.0);
                p_sum.fill(0.0);
                p_sq_sum.fill(0.0);
                p_nnz.fill(0);
                let chunk_idx = b / CHUNK_BINS;
                let c0 = ce.chunk_init_ptr[chunk_idx] as usize;
                let c1 = ce.chunk_init_ptr[chunk_idx + 1] as usize;
                for pos in c0..c1 {
                    let s_raw = ce.chunk_init_sample[pos] as usize;
                    let p_raw = ce.chunk_init_parent[pos] as usize;
                    if s_raw >= sample_remap.len() || p_raw >= p_total {
                        continue;
                    }
                    let s_new = sample_remap[s_raw];
                    if s_new < 0 {
                        continue;
                    }
                    let d = ce.chunk_init_dosage[pos] as f64;
                    if d == 0.0 || !d.is_finite() {
                        continue;
                    }
                    cur_z[p_raw * n + s_new as usize] = d;
                    p_sum[p_raw] += d;
                    p_sq_sum[p_raw] += d * d;
                    p_nnz[p_raw] = p_nnz[p_raw].saturating_add(1);
                }
            } else {
                let d0 = ce.delta_ptr[b] as usize;
                let d1 = ce.delta_ptr[b + 1] as usize;
                for pos in d0..d1 {
                    let s_raw = ce.delta_sample[pos] as usize;
                    let p_raw = ce.delta_parent[pos] as usize;
                    if s_raw >= sample_remap.len() || p_raw >= p_total {
                        continue;
                    }
                    let s_new = sample_remap[s_raw];
                    if s_new < 0 {
                        continue;
                    }
                    let old_v = ce.delta_old[pos] as f64;
                    let new_v = ce.delta_new[pos] as f64;
                    cur_z[p_raw * n + s_new as usize] = new_v;
                    update_parent_stats(p_raw, old_v, new_v, &mut p_sum, &mut p_sq_sum, &mut p_nnz);
                }
            }

            let mut kept_in_bin = 0usize;
            for p in 0..p_total {
                if p_nnz[p] == 0 {
                    continue;
                }
                let mean = p_sum[p] * inv_n;
                let var = (p_sq_sum[p] - p_sum[p] * mean).max(0.0) * inv_n;
                if var <= 1e-12 {
                    continue;
                }
                let col_off = z_batch.len();
                z_batch.resize(col_off + n, 0.0);
                let src = &cur_z[p * n..(p + 1) * n];
                let dst = &mut z_batch[col_off..col_off + n];
                if method == 1 {
                    for i in 0..n {
                        dst[i] = src[i] - mean;
                    }
                } else {
                    let inv_sd = 1.0 / var.sqrt();
                    for i in 0..n {
                        dst[i] = (src[i] - mean) * inv_sd;
                    }
                }
                kept_in_bin += 1;
                batch_cols += 1;
            }
            if kept_in_bin > 0 {
                eff_bins += 1;
                eff_cols += kept_in_bin;
            }
            if batch_cols >= DEFAULT_BIN_BATCH_COLS {
                flush_batch(&mut z_batch, &mut batch_cols, &mut grm);
            }
        }
    }
    flush_batch(&mut z_batch, &mut batch_cols, &mut grm);

    if eff_cols == 0 {
        return Err("no polymorphic ancestry bins found in .jhibd archive".to_string());
    }

    let denom = if method == 1 {
        let tr: f64 = (0..n).map(|i| grm[i * n + i]).sum();
        if !tr.is_finite() || tr <= 0.0 {
            return Err("ancestry GRM has non-positive trace".to_string());
        }
        tr * inv_n
    } else {
        eff_cols as f64
    };
    let scale = 1.0 / denom;
    for v in &mut grm {
        *v *= scale;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (grm[i * n + j] + grm[j * n + i]);
            grm[i * n + j] = avg;
            grm[j * n + i] = avg;
        }
    }

    Ok((grm, n, eff_bins))
}

/// Core streaming hGWAS scan over a `CompactHgwasArchive`, writing results to `out_tsv`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_hgwas_compact_to_tsv(
    archive: &CompactHgwasArchive,
    out_tsv: &str,
    s: &[f64],
    x_rot: &[f64],
    y_rot: &[f64],
    u_t: &[f64],
    n: usize,
    q: usize,
    sample_indices: Option<&[i64]>,
    exact_lmm: bool,
    lambda_g_null: Option<f64>,
    log10_low: f64,
    log10_high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    mut on_progress: impl FnMut(usize, usize),
) -> Result<(usize, f64), String> {
    if s.len() != n || y_rot.len() != n || x_rot.len() != n * q || u_t.len() != n * n {
        return Err("dimension mismatch in rotated inputs for hGWAS".to_string());
    }
    validate_fixed_effect_design(x_rot, n, q).map_err(|e| e.to_string())?;
    let (sample_remap, n_sel) =
        build_sample_remap(archive.target_sample_ids.len(), sample_indices)?;
    if n_sel != n {
        return Err(format!(
            "selected sample count {} does not match phenotype length {}",
            n_sel, n
        ));
    }

    let c_k = (s.iter().sum::<f64>() / (n as f64)).max(1e-12);
    let null_bg = if let Some(lg_fixed) = lambda_g_null {
        if !exact_lmm && lg_fixed.is_finite() && lg_fixed >= 0.0 {
            evaluate_background_state(s, x_rot, y_rot, n, q, lg_fixed)
                .ok_or_else(|| "failed to evaluate fixed null background state".to_string())?
        } else {
            fit_null_reml_background(s, x_rot, y_rot, n, q, log10_low, log10_high, Some(lg_fixed))?
        }
    } else {
        fit_null_reml_background(s, x_rot, y_rot, n, q, log10_low, log10_high, None)?
    };

    // Transpose U^T -> U (n x n, row-major) once so U[i, :] is contiguous.
    let mut u_mat = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            u_mat[i * n + j] = u_t[j * n + i];
        }
    }

    let fvlmm_proj = if !exact_lmm {
        let proj = build_original_space_projection(&u_mat, &null_bg, n, q, threads);
        // Free `u_mat` immediately in FVLMM mode (not needed during scan).
        u_mat.clear();
        u_mat.shrink_to_fit();
        Some(proj)
    } else {
        None
    };

    let n_grid_lmm = 8usize;
    let grid_step = (log10_high - log10_low) / (n_grid_lmm as f64);
    let cached_grid: Vec<(f64, BackgroundState)> = if exact_lmm {
        let mut grid = Vec::with_capacity(n_grid_lmm + 2);
        if let Some(bg0) = evaluate_background_state(s, x_rot, y_rot, n, q, 0.0) {
            grid.push((f64::NEG_INFINITY, bg0));
        }
        for g in 0..=n_grid_lmm {
            let lg = log10_low + (g as f64) * grid_step;
            let lam = 10.0_f64.powf(lg);
            if let Some(bg) = evaluate_background_state(s, x_rot, y_rot, n, q, lam) {
                grid.push((lg, bg));
            }
        }
        grid
    } else {
        Vec::new()
    };

    let pool = if threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| format!("failed to create rayon thread pool: {}", e))?,
        )
    } else {
        None
    };

    // Precompute original-space `(M_g, v_g)` for `null_bg` + `cached_grid` when they
    // fit comfortably inside the 256 MiB memory budget (gives 5-10x exact LMM speedup).
    let lmm_grid_projs: Option<Vec<(Vec<f64>, Vec<f64>)>> = if exact_lmm {
        let n_states = 1 + cached_grid.len();
        let est_bytes = n_states
            .saturating_mul(n)
            .saturating_mul(n)
            .saturating_mul(std::mem::size_of::<f64>());
        if est_bytes <= LMM_MULTI_GRID_MAX_BYTES {
            let build_all = || -> Vec<(Vec<f64>, Vec<f64>)> {
                let mut refs: Vec<&BackgroundState> = Vec::with_capacity(n_states);
                refs.push(&null_bg);
                for (_, bg) in &cached_grid {
                    refs.push(bg);
                }
                refs.par_iter()
                    .map(|bg| build_original_space_projection(&u_mat, bg, n, q, 1))
                    .collect()
            };
            let projs = if let Some(tp) = pool.as_ref() {
                tp.install(build_all)
            } else {
                build_all()
            };
            Some(projs)
        } else {
            None
        }
    } else {
        None
    };

    let total_bins = archive.total_bins;
    let file = File::create(out_tsv)
        .map_err(|e| format!("failed to create hGWAS TSV '{}': {}", out_tsv, e))?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);
    writeln!(
        writer,
        "chrom\tpos\tn_parents\tsigma_g2\tsigma_b2\tsigma_e2\th2_bin\tlrt\tplrt"
    )
    .map_err(|e| format!("failed to write hGWAS TSV header: {}", e))?;

    let p_total = archive.reference_sample_ids.len().max(1);

    // Build flat chunk descriptors `(chrom_vec_idx, chunk_idx, b_start, b_end)`.
    let mut chunks: Vec<(usize, usize, usize, usize)> = Vec::new();
    for (ci, ce) in archive.chrom_events.iter().enumerate() {
        let n_bins = ce.bin_starts_bp.len();
        let mut b = 0usize;
        let mut chunk_idx = 0usize;
        while b < n_bins {
            let b_end = (b + CHUNK_BINS).min(n_bins);
            chunks.push((ci, chunk_idx, b, b_end));
            b = b_end;
            chunk_idx += 1;
        }
    }

    let wave_size = (threads.max(1) * 4).clamp(4, 64);
    let mut bins_done = 0usize;

    for wave in chunks.chunks(wave_size) {
        let run_wave = || -> Vec<String> {
            wave.par_iter()
                .map(|&(ci, chunk_idx, b_start, b_end)| {
                    let ce = &archive.chrom_events[ci];
                    let chrom_name = archive
                        .chrom_names
                        .get(ce.chrom_idx as usize)
                        .map(|s| s.as_str())
                        .unwrap_or("chr?");
                    if let Some((ref m0, ref v0)) = fvlmm_proj {
                        scan_chunk_fvlmm_events(
                            ce,
                            chrom_name,
                            chunk_idx,
                            b_start,
                            b_end,
                            &sample_remap,
                            n,
                            q,
                            p_total,
                            c_k,
                            &null_bg,
                            m0,
                            v0,
                        )
                    } else {
                        scan_chunk_lmm_events(
                            ce,
                            chrom_name,
                            chunk_idx,
                            b_start,
                            b_end,
                            &sample_remap,
                            s,
                            x_rot,
                            y_rot,
                            &u_mat,
                            n,
                            q,
                            p_total,
                            c_k,
                            &null_bg,
                            &cached_grid,
                            lmm_grid_projs.as_deref(),
                            grid_step,
                            log10_low,
                            log10_high,
                            max_iter,
                            tol,
                        )
                    }
                })
                .collect()
        };

        let wave_texts = if let Some(tp) = pool.as_ref() {
            tp.install(run_wave)
        } else {
            run_wave()
        };

        for (&(_, _, b_start, b_end), chunk_text) in wave.iter().zip(&wave_texts) {
            writer
                .write_all(chunk_text.as_bytes())
                .map_err(|e| format!("failed to write hGWAS TSV chunk: {}", e))?;
            bins_done += b_end - b_start;
        }
        on_progress(bins_done, total_bins);
    }

    writer
        .flush()
        .map_err(|e| format!("failed to flush hGWAS TSV: {}", e))?;

    let null_pve = {
        let sg2 = null_bg.lambda_g * null_bg.sigma_e2_0;
        let denom = sg2 * c_k + null_bg.sigma_e2_0;
        if denom > 0.0 {
            (sg2 * c_k / denom).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    Ok((bins_done, null_pve))
}

// ============================================================
// PyO3 bindings (reusable handle + backward-compatible functions)
// ============================================================

/// Reusable compact hGWAS archive handle (`#[pyclass]`) that loads `.jhibd`
/// (`JHIBD01` or `JHIBD02`) once into a compact sweep-line event index and can
/// be reused across ancestry GRM construction, multiple traits, and both
/// `-fvlmm` and `-lmm` without re-decompressing from disk.
#[pyclass]
pub struct HgwasArchive {
    inner: Arc<CompactHgwasArchive>,
}

#[pymethods]
impl HgwasArchive {
    #[staticmethod]
    pub fn open(py: Python<'_>, jhibd_path: String) -> PyResult<Self> {
        let compact = py
            .detach(move || CompactHgwasArchive::open(&jhibd_path))
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Arc::new(compact),
        })
    }

    pub fn inspect(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("sample_ids", &self.inner.target_sample_ids)?;
        dict.set_item("parent_ids", &self.inner.reference_sample_ids)?;
        dict.set_item("chrom_names", &self.inner.chrom_names)?;
        dict.set_item("n_bins", self.inner.total_bins)?;
        dict.set_item("version", self.inner.version)?;
        Ok(dict.into_any().unbind())
    }

    #[pyo3(signature = (sample_indices=None, method=1, threads=0))]
    pub fn build_grm<'py>(
        &self,
        py: Python<'py>,
        sample_indices: Option<PyReadonlyArray1<'py, i64>>,
        method: i32,
        threads: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, usize)> {
        let sel_vec: Option<Vec<i64>> = sample_indices
            .map(|arr| {
                arr.as_slice()
                    .map(|sl| sl.to_vec())
                    .map_err(|_| PyValueError::new_err("sample_indices must be contiguous i64"))
            })
            .transpose()?;
        let arc = Arc::clone(&self.inner);
        let (grm, n, eff_bins) = py
            .detach(move || {
                build_ancestry_grm_from_compact(&arc, sel_vec.as_deref(), method, threads)
            })
            .map_err(PyRuntimeError::new_err)?;
        let arr = PyArray2::<f64>::from_vec2(
            py,
            &(0..n)
                .map(|i| grm[i * n..(i + 1) * n].to_vec())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create GRM array: {}", e)))?;
        Ok((arr, eff_bins))
    }

    #[pyo3(signature = (
        out_tsv,
        s,
        x_rot,
        y_rot,
        u_t,
        sample_indices=None,
        exact_lmm=false,
        lambda_g_null=None,
        low=-5.0,
        high=5.0,
        max_iter=30,
        tol=1e-2,
        threads=0,
        progress_callback=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn scan_to_tsv(
        &self,
        py: Python<'_>,
        out_tsv: String,
        s: PyReadonlyArray1<'_, f64>,
        x_rot: PyReadonlyArray2<'_, f64>,
        y_rot: PyReadonlyArray1<'_, f64>,
        u_t: PyReadonlyArray2<'_, f64>,
        sample_indices: Option<PyReadonlyArray1<'_, i64>>,
        exact_lmm: bool,
        lambda_g_null: Option<f64>,
        low: f64,
        high: f64,
        max_iter: usize,
        tol: f64,
        threads: usize,
        progress_callback: Option<Py<PyAny>>,
    ) -> PyResult<(usize, f64)> {
        let s_vec = s
            .as_slice()
            .map_err(|_| PyValueError::new_err("s must be contiguous f64"))?
            .to_vec();
        let y_vec = y_rot
            .as_slice()
            .map_err(|_| PyValueError::new_err("y_rot must be contiguous f64"))?
            .to_vec();
        let x_shape = x_rot.shape();
        let n = s_vec.len();
        if y_vec.len() != n || x_shape[0] != n || x_shape[1] == 0 {
            return Err(PyValueError::new_err(
                "dimension mismatch between s, y_rot, and x_rot",
            ));
        }
        let q = x_shape[1];
        let x_vec = x_rot
            .as_slice()
            .map_err(|_| PyValueError::new_err("x_rot must be contiguous f64"))?
            .to_vec();
        let u_shape = u_t.shape();
        if u_shape[0] != n || u_shape[1] != n {
            return Err(PyValueError::new_err("u_t must have shape (n, n)"));
        }
        let u_vec = u_t
            .as_slice()
            .map_err(|_| PyValueError::new_err("u_t must be contiguous f64"))?
            .to_vec();
        let sel_vec: Option<Vec<i64>> = sample_indices
            .map(|arr| {
                arr.as_slice()
                    .map(|sl| sl.to_vec())
                    .map_err(|_| PyValueError::new_err("sample_indices must be contiguous i64"))
            })
            .transpose()?;

        let arc = Arc::clone(&self.inner);
        let cb_ref = progress_callback.map(|cb| cb.clone_ref(py));
        py.detach(move || {
            scan_hgwas_compact_to_tsv(
                &arc,
                &out_tsv,
                &s_vec,
                &x_vec,
                &y_vec,
                &u_vec,
                n,
                q,
                sel_vec.as_deref(),
                exact_lmm,
                lambda_g_null,
                low,
                high,
                max_iter,
                tol,
                threads,
                |done, total| {
                    if let Some(cb) = cb_ref.as_ref() {
                        let _ = Python::attach(|py_inner| cb.call1(py_inner, (done, total)));
                    }
                },
            )
        })
        .map_err(PyRuntimeError::new_err)
    }
}

/// Inspect metadata of a `.jhibd` archive for hGWAS CLI setup.
#[pyfunction]
pub fn hgwas_inspect_jhibd(py: Python<'_>, jhibd_path: String) -> PyResult<Py<PyAny>> {
    let meta = py
        .detach(move || {
            let meta = inspect_sparse_archive_meta(Path::new(&jhibd_path))?;
            if meta.mode != HaploidIbdMode::MultiParentAncestry {
                return Err(format!(
                    "hGWAS requires a MultiParentAncestry .jhibd archive, got {:?}",
                    meta.mode
                ));
            }
            Ok(meta)
        })
        .map_err(PyValueError::new_err)?;
    let dict = PyDict::new(py);
    dict.set_item("sample_ids", meta.target_sample_ids)?;
    dict.set_item("parent_ids", meta.reference_sample_ids)?;
    dict.set_item("chrom_names", meta.chrom_names)?;
    dict.set_item("n_bins", meta.n_bins)?;
    dict.set_item("version", meta.version)?;
    Ok(dict.into_any().unbind())
}

/// Build an ancestry GRM directly from `.jhibd` founder dosage bins.
#[pyfunction]
#[pyo3(signature = (jhibd_path, sample_indices=None, method=1, threads=0))]
pub fn hgwas_grm_from_jhibd_f64<'py>(
    py: Python<'py>,
    jhibd_path: String,
    sample_indices: Option<PyReadonlyArray1<'py, i64>>,
    method: i32,
    threads: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, usize)> {
    let sel_vec: Option<Vec<i64>> = sample_indices
        .map(|arr| {
            arr.as_slice()
                .map(|sl| sl.to_vec())
                .map_err(|_| PyValueError::new_err("sample_indices must be contiguous i64"))
        })
        .transpose()?;
    let (grm, n, eff_bins) = py
        .detach(move || {
            let compact = CompactHgwasArchive::open(&jhibd_path)?;
            build_ancestry_grm_from_compact(&compact, sel_vec.as_deref(), method, threads)
        })
        .map_err(PyRuntimeError::new_err)?;
    let arr = PyArray2::<f64>::from_vec2(
        py,
        &(0..n)
            .map(|i| grm[i * n..(i + 1) * n].to_vec())
            .collect::<Vec<_>>(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("failed to create GRM array: {}", e)))?;
    Ok((arr, eff_bins))
}

/// Streaming hGWAS scan over `.jhibd` into TSV using either FVLMM (`exact_lmm=False`)
/// or exact per-bin LMM (`exact_lmm=True`).
#[pyfunction]
#[pyo3(signature = (
    jhibd_path,
    out_tsv,
    s,
    x_rot,
    y_rot,
    u_t,
    sample_indices=None,
    exact_lmm=false,
    lambda_g_null=None,
    low=-5.0,
    high=5.0,
    max_iter=30,
    tol=1e-2,
    threads=0,
    progress_callback=None
))]
#[allow(clippy::too_many_arguments)]
pub fn hgwas_scan_jhibd_to_tsv_f64(
    py: Python<'_>,
    jhibd_path: String,
    out_tsv: String,
    s: PyReadonlyArray1<'_, f64>,
    x_rot: PyReadonlyArray2<'_, f64>,
    y_rot: PyReadonlyArray1<'_, f64>,
    u_t: PyReadonlyArray2<'_, f64>,
    sample_indices: Option<PyReadonlyArray1<'_, i64>>,
    exact_lmm: bool,
    lambda_g_null: Option<f64>,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<(usize, f64)> {
    let handle = HgwasArchive::open(py, jhibd_path)?;
    handle.scan_to_tsv(
        py,
        out_tsv,
        s,
        x_rot,
        y_rot,
        u_t,
        sample_indices,
        exact_lmm,
        lambda_g_null,
        low,
        high,
        max_iter,
        tol,
        threads,
        progress_callback,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::haploid::ancestry::build_union_breakpoint_sparse_matrix;
    use crate::haploid::phasing::SampleHaplotypeLayout;
    use crate::haploid::sparse_io::{
        write_sparse_archive, write_sparse_archive_v1_with_threads,
        write_sparse_archive_v2_with_threads, HaploidSparseArchive, HaploidSparsePayload,
    };
    use crate::haploid::types::AncestrySegment;
    use crate::linalg::{cholesky_inplace, cholesky_logdet, cholesky_solve_into};

    /// Brute-force dense n x n REML log-likelihood for `lambda_g * K + tau * Z Z^T + I`.
    fn dense_reml_ll(
        k: &[f64],
        z: &[f64],
        x: &[f64],
        y: &[f64],
        n: usize,
        q: usize,
        p: usize,
        lambda_g: f64,
        tau: f64,
    ) -> f64 {
        let mut h = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut zzt = 0.0_f64;
                for c in 0..p {
                    zzt += z[i * p + c] * z[j * p + c];
                }
                h[i * n + j] = lambda_g * k[i * n + j] + tau * zzt + if i == j { 1.0 } else { 0.0 };
            }
        }
        let mut l = h.clone();
        cholesky_inplace(&mut l, n).expect("dense H cholesky");
        let logdet_h = cholesky_logdet(&l, n);

        let mut hinv_y = vec![0.0_f64; n];
        cholesky_solve_into(&l, n, y, &mut hinv_y);

        let mut hinv_x = vec![0.0_f64; n * q];
        let mut col_in = vec![0.0_f64; n];
        let mut col_out = vec![0.0_f64; n];
        for a in 0..q {
            for i in 0..n {
                col_in[i] = x[i * q + a];
            }
            cholesky_solve_into(&l, n, &col_in, &mut col_out);
            for i in 0..n {
                hinv_x[i * q + a] = col_out[i];
            }
        }

        let mut xhx = vec![0.0_f64; q * q];
        let mut xhy = vec![0.0_f64; q];
        for i in 0..n {
            for a in 0..q {
                xhy[a] += x[i * q + a] * hinv_y[i];
                for b in 0..q {
                    xhx[a * q + b] += x[i * q + a] * hinv_x[i * q + b];
                }
            }
        }
        let mut lx = xhx.clone();
        cholesky_gram_inplace(&mut lx, q).expect("dense X^T H^{-1} X cholesky");
        let logdet_xhx = cholesky_logdet(&lx, q);

        let mut beta = vec![0.0_f64; q];
        cholesky_solve_into(&lx, q, &xhy, &mut beta);
        let yhy: f64 = y.iter().zip(&hinv_y).map(|(a, b)| a * b).sum();
        let quad: f64 = xhy.iter().zip(&beta).map(|(a, b)| a * b).sum();
        let rss = (yhy - quad).max(MIN_RSS_FLOOR);
        let df = (n - q) as f64;
        let se2 = rss / df;
        let const_term = df * ((2.0 * std::f64::consts::PI).ln() + 1.0);
        -0.5 * (const_term + df * se2.ln() + logdet_h + logdet_xhx)
    }

    fn parse_tsv_rows(text: &str) -> Vec<(u64, usize, f64, f64, f64, f64, f64)> {
        let mut lines = text.lines();
        let header = lines.next().expect("header");
        assert_eq!(
            header,
            "chrom\tpos\tn_parents\tsigma_g2\tsigma_b2\tsigma_e2\th2_bin\tlrt\tplrt"
        );
        lines
            .map(|ln| {
                let c: Vec<&str> = ln.split('\t').collect();
                assert_eq!(c.len(), 9);
                (
                    c[1].parse::<u64>().unwrap(),
                    c[2].parse::<usize>().unwrap(),
                    c[3].parse::<f64>().unwrap(),
                    c[4].parse::<f64>().unwrap(),
                    c[5].parse::<f64>().unwrap(),
                    c[7].parse::<f64>().unwrap(),
                    c[8].parse::<f64>().unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn jacobi_small_eigh_matches_lapack() {
        let p = 64usize;
        let mut state: u64 = 0xfeed_face_cafe_0001;
        let mut next_f64 = || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut m = vec![0.0_f64; p * p];
        for v in &mut m {
            *v = next_f64();
        }
        let mut a = vec![0.0_f64; p * p];
        for i in 0..p {
            for j in 0..p {
                let mut s = 0.0;
                for k in 0..p {
                    s += m[i * p + k] * m[j * p + k];
                }
                a[i * p + j] = s;
            }
        }
        let a_orig = a.clone();
        let (ref_evals, _, _) = symmetric_eigh_f64_row_major(&a, p).unwrap();
        let mut a_j = a.clone();
        let mut v_j = vec![0.0_f64; p * p];
        jacobi_sym_eigh_inplace(&mut a_j, &mut v_j, p);
        let mut j_evals: Vec<f64> = (0..p).map(|i| a_j[i * p + i]).collect();
        j_evals.sort_by(|x, y| x.total_cmp(y));
        for (e1, e2) in ref_evals.iter().zip(&j_evals) {
            assert!((e1 - e2).abs() < 1e-10, "eigenvalue mismatch: {e1} vs {e2}");
        }
        for j in 0..p {
            let eval = a_j[j * p + j];
            let mut residual = 0.0_f64;
            for i in 0..p {
                let mut av = 0.0_f64;
                for k in 0..p {
                    av += a_orig[i * p + k] * v_j[k * p + j];
                }
                residual += (av - eval * v_j[i * p + j]).powi(2);
            }
            assert!(
                residual < 1e-10,
                "eigenvector residual too large for column {j}: {residual}"
            );
        }
    }

    #[test]
    fn fvlmm_woodbury_reml_matches_dense_cholesky() {
        let n = 40usize;
        let q = 2usize;
        let p = 5usize;

        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next_f64 = || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };

        let mut g_raw = vec![0.0_f64; n * 15];
        for v in &mut g_raw {
            *v = next_f64();
        }
        let mut k = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for m in 0..15 {
                    sum += g_raw[i * 15 + m] * g_raw[j * 15 + m];
                }
                k[i * n + j] = sum / 15.0;
            }
        }

        let mut x = vec![0.0_f64; n * q];
        for i in 0..n {
            x[i * q] = 1.0;
            x[i * q + 1] = next_f64();
        }

        let mut z = vec![0.0_f64; n * p];
        let parent_effects = [1.5, -1.0, 0.4, -0.8, 0.9];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let par = i % p;
            z[i * p + par] = 1.0;
            y[i] = 2.0 + 0.5 * x[i * q + 1] + parent_effects[par] + 0.3 * next_f64();
        }

        let (s, u, _) = symmetric_eigh_f64_row_major(&k, n).expect("eigh K");
        let mut u_t = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                u_t[i * n + j] = u[j * n + i];
            }
        }
        let mut x_rot = vec![0.0_f64; n * q];
        let mut y_rot = vec![0.0_f64; n];
        let mut z_rot = vec![0.0_f64; p * n];
        for i in 0..n {
            for j in 0..n {
                let uij = u_t[i * n + j];
                y_rot[i] += uij * y[j];
                for a in 0..q {
                    x_rot[i * q + a] += uij * x[j * q + a];
                }
                for c in 0..p {
                    z_rot[c * n + i] += uij * z[j * p + c];
                }
            }
        }

        let lambda_g = 1.25_f64;
        let bg = evaluate_background_state(&s, &x_rot, &y_rot, n, q, lambda_g).unwrap();
        let dense_ll0 = dense_reml_ll(&k, &z, &x, &y, n, q, p, lambda_g, 0.0);
        assert!(
            (bg.reml_ll0 - dense_ll0).abs() < 1e-9,
            "null REML mismatch: {} vs {}",
            bg.reml_ll0,
            dense_ll0
        );

        let active: Vec<usize> = (0..p).collect();
        let mut scratch = TauSolverScratch::with_capacity(p, n, q);
        let (woodbury_ll1, opt_tau, _) =
            optimize_tau_for_background_zrot(&bg, &z_rot, &active, n, q, &mut scratch);
        let dense_ll1 = dense_reml_ll(&k, &z, &x, &y, n, q, p, lambda_g, opt_tau);
        assert!(opt_tau > 0.1, "expected positive tau for causal bin");
        assert!(
            (woodbury_ll1 - dense_ll1).abs() < 1e-8,
            "alt REML mismatch: {} vs {}",
            woodbury_ll1,
            dense_ll1
        );
    }

    #[test]
    fn sweep_line_fvlmm_and_multigrid_lmm_match_zrot_and_dense_reml() {
        let n = 48usize;
        let q = 2usize;
        let p_total = 6usize;

        let mut state: u64 = 0x9876_5432_10fe_dcba;
        fn lcg_u64(state: &mut u64) -> u64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *state
        }
        fn lcg_f64(state: &mut u64) -> f64 {
            ((lcg_u64(state) >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        }

        let mut g_raw = vec![0.0_f64; n * 20];
        for v in &mut g_raw {
            *v = lcg_f64(&mut state);
        }
        let mut k = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for m in 0..20 {
                    sum += g_raw[i * 20 + m] * g_raw[j * 20 + m];
                }
                k[i * n + j] = sum / 20.0;
            }
        }
        let mut x = vec![0.0_f64; n * q];
        for i in 0..n {
            x[i * q] = 1.0;
            x[i * q + 1] = lcg_f64(&mut state);
        }

        // Mix of inbred (weight 1.0) and diploid (weight 0.5/hap) samples with
        // recombination switches AND an all-uncalled empty gap in the middle.
        let is_inbred: Vec<bool> = (0..n).map(|i| i % 4 != 0).collect();
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(is_inbred.clone());
        let mut segments: Vec<AncestrySegment> = Vec::new();
        for s_idx in 0..n {
            let n_haps = if is_inbred[s_idx] { 1u8 } else { 2u8 };
            for h in 0..n_haps {
                let bp1 = 50_000 + ((s_idx as u64 * 13 + h as u64 * 7) % 5) * 10_000;
                let p0 = ((s_idx + h as usize) % p_total) as u32;
                let p1 = ((s_idx + h as usize + 2) % p_total) as u32;
                let p2 = ((s_idx + h as usize + 4) % p_total) as u32;
                segments.push(AncestrySegment {
                    target_sample: s_idx as u32,
                    hap: h,
                    chrom_idx: 0,
                    start_bp: 1,
                    end_bp: bp1,
                    start_cm: 0.0,
                    end_cm: 0.5,
                    parent_sample: p0,
                    parent_hap: 0,
                    confidence: 0.99,
                    ambiguous_parents: 1,
                    n_markers: 20,
                });
                segments.push(AncestrySegment {
                    target_sample: s_idx as u32,
                    hap: h,
                    chrom_idx: 0,
                    start_bp: bp1 + 1,
                    end_bp: 120_000,
                    start_cm: 0.5,
                    end_cm: 1.2,
                    parent_sample: p1,
                    parent_hap: 0,
                    confidence: 0.99,
                    ambiguous_parents: 1,
                    n_markers: 20,
                });
                // All-uncalled gap [120_001..150_000], then resume [150_001..250_000].
                segments.push(AncestrySegment {
                    target_sample: s_idx as u32,
                    hap: h,
                    chrom_idx: 0,
                    start_bp: 150_001,
                    end_bp: 250_000,
                    start_cm: 1.5,
                    end_cm: 2.5,
                    parent_sample: p2,
                    parent_hap: 0,
                    confidence: 0.99,
                    ambiguous_parents: 1,
                    n_markers: 20,
                });
            }
        }

        let sparse_bins = build_union_breakpoint_sparse_matrix(0, &tgt_layout, p_total, &segments);
        let seg_refs: Vec<&AncestrySegment> = segments.iter().collect();
        let compact_from_segs =
            build_compact_events_from_segments(0, n, p_total, &is_inbred, &seg_refs);
        let compact_from_bins = build_compact_events_from_sparse_bins(&sparse_bins);
        assert_eq!(
            compact_from_segs.bin_starts_bp, compact_from_bins.bin_starts_bp,
            "sweep-line bins must match build_union_breakpoint_sparse_matrix exactly"
        );

        let n_bins = sparse_bins.bin_starts_bp.len();
        let mut z_per_bin: Vec<Vec<f64>> = vec![vec![0.0_f64; n * p_total]; n_bins];
        for b in 0..n_bins {
            let s0 = sparse_bins.bin_ptr[b] as usize;
            let s1 = sparse_bins.bin_ptr[b + 1] as usize;
            for pos in s0..s1 {
                let si = sparse_bins.sample_idx[pos] as usize;
                let pi = sparse_bins.parent_idx[pos] as usize;
                z_per_bin[b][si * p_total + pi] = sparse_bins.dosage[pos] as f64;
            }
        }

        let causal_bin = n_bins / 2;
        let effects = [2.2, -1.8, 1.0, -1.2, 0.7, -0.9];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let mut g_eff = 0.0;
            for p in 0..p_total {
                g_eff += z_per_bin[causal_bin][i * p_total + p] * effects[p];
            }
            y[i] = 3.0 + 0.4 * x[i * q + 1] + g_eff + 0.35 * lcg_f64(&mut state);
        }

        let (s, u, _) = symmetric_eigh_f64_row_major(&k, n).unwrap();
        let mut u_t = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                u_t[i * n + j] = u[j * n + i];
            }
        }
        let mut x_rot = vec![0.0_f64; n * q];
        let mut y_rot = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                let uij = u_t[i * n + j];
                y_rot[i] += uij * y[j];
                for a in 0..q {
                    x_rot[i * q + a] += uij * x[j * q + a];
                }
            }
        }

        let remap: Vec<i32> = (0..n).map(|i| i as i32).collect();
        let null_bg = fit_null_reml_background(&s, &x_rot, &y_rot, n, q, -5.0, 5.0, None).unwrap();
        let c_k = (s.iter().sum::<f64>() / (n as f64)).max(1e-12);
        let (m0, v0) = build_original_space_projection(&u, &null_bg, n, q, 1);

        let fv_text_segs = scan_chunk_fvlmm_events(
            &compact_from_segs,
            "chr1",
            0,
            0,
            n_bins,
            &remap,
            n,
            q,
            p_total,
            c_k,
            &null_bg,
            &m0,
            &v0,
        );
        let fv_text_bins = scan_chunk_fvlmm_events(
            &compact_from_bins,
            "chr1",
            0,
            0,
            n_bins,
            &remap,
            n,
            q,
            p_total,
            c_k,
            &null_bg,
            &m0,
            &v0,
        );
        assert_eq!(fv_text_segs, fv_text_bins);

        let header = "chrom\tpos\tn_parents\tsigma_g2\tsigma_b2\tsigma_e2\th2_bin\tlrt\tplrt\n";
        let fv_rows = parse_tsv_rows(&format!("{header}{fv_text_segs}"));
        for (b, row) in fv_rows.iter().enumerate() {
            let tau_hat = if row.4 > 0.0 { row.3 / row.4 } else { 0.0 };
            let dense_ll0 = dense_reml_ll(
                &k,
                &z_per_bin[b],
                &x,
                &y,
                n,
                q,
                p_total,
                null_bg.lambda_g,
                0.0,
            );
            let dense_ll1 = dense_reml_ll(
                &k,
                &z_per_bin[b],
                &x,
                &y,
                n,
                q,
                p_total,
                null_bg.lambda_g,
                tau_hat,
            );
            let dense_lrt = (2.0 * (dense_ll1 - dense_ll0)).max(0.0);
            assert!(
                (row.5 - dense_lrt).abs() < 1e-5,
                "bin {b} FVLMM LRT mismatch: fast={} vs dense={}",
                row.5,
                dense_lrt
            );
        }

        // Compare multi-grid incremental LMM vs z_rot-only LMM vs dense Cholesky REML.
        let n_grid_lmm = 8usize;
        let grid_step = 10.0_f64 / (n_grid_lmm as f64);
        let mut cached_grid = Vec::with_capacity(n_grid_lmm + 2);
        if let Some(bg0) = evaluate_background_state(&s, &x_rot, &y_rot, n, q, 0.0) {
            cached_grid.push((f64::NEG_INFINITY, bg0));
        }
        for g in 0..=n_grid_lmm {
            let lg = -5.0 + (g as f64) * grid_step;
            if let Some(bg) = evaluate_background_state(&s, &x_rot, &y_rot, n, q, 10.0_f64.powf(lg))
            {
                cached_grid.push((lg, bg));
            }
        }
        let mut grid_projs = Vec::with_capacity(1 + cached_grid.len());
        grid_projs.push((m0, v0));
        for (_, bg) in &cached_grid {
            grid_projs.push(build_original_space_projection(&u, bg, n, q, 1));
        }

        let lmm_text_mg = scan_chunk_lmm_events(
            &compact_from_segs,
            "chr1",
            0,
            0,
            n_bins,
            &remap,
            &s,
            &x_rot,
            &y_rot,
            &u,
            n,
            q,
            p_total,
            c_k,
            &null_bg,
            &cached_grid,
            Some(&grid_projs),
            grid_step,
            -5.0,
            5.0,
            30,
            1e-2,
        );
        let lmm_text_zr = scan_chunk_lmm_events(
            &compact_from_segs,
            "chr1",
            0,
            0,
            n_bins,
            &remap,
            &s,
            &x_rot,
            &y_rot,
            &u,
            n,
            q,
            p_total,
            c_k,
            &null_bg,
            &cached_grid,
            None,
            grid_step,
            -5.0,
            5.0,
            30,
            1e-2,
        );
        let mg_rows = parse_tsv_rows(&format!("{header}{lmm_text_mg}"));
        let zr_rows = parse_tsv_rows(&format!("{header}{lmm_text_zr}"));
        for b in 0..n_bins {
            assert!(
                (mg_rows[b].5 - zr_rows[b].5).abs() < 1e-5,
                "bin {b} multi-grid LMM ({}) != z_rot LMM ({})",
                mg_rows[b].5,
                zr_rows[b].5
            );
            let lam_hat = if mg_rows[b].4 > 0.0 {
                mg_rows[b].2 / mg_rows[b].4
            } else {
                0.0
            };
            let tau_hat = if mg_rows[b].4 > 0.0 {
                mg_rows[b].3 / mg_rows[b].4
            } else {
                0.0
            };
            let dense_ll0 = dense_reml_ll(
                &k,
                &z_per_bin[b],
                &x,
                &y,
                n,
                q,
                p_total,
                null_bg.lambda_g,
                0.0,
            );
            let dense_ll1 =
                dense_reml_ll(&k, &z_per_bin[b], &x, &y, n, q, p_total, lam_hat, tau_hat);
            let dense_lrt = (2.0 * (dense_ll1 - dense_ll0)).max(0.0);
            assert!(
                (mg_rows[b].5 - dense_lrt).abs() < 1e-5,
                "bin {b} multi-grid LMM LRT mismatch: fast={} vs dense={}",
                mg_rows[b].5,
                dense_lrt
            );
        }
    }

    #[test]
    fn end_to_end_jhibd_v1_and_v2_hgwas_match_and_detect_causal_bin() {
        let n = 60usize;
        let n_parents = 6usize;
        let n_bins = 20usize;
        let causal_bin = 7usize;

        let mut state: u64 = 0xdead_beef_cafe_babe;
        let mut next_u64 = || -> u64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state
        };

        let mut segments: Vec<AncestrySegment> = Vec::with_capacity(n * n_bins);
        let mut causal_parent = vec![0usize; n];
        for b in 0..n_bins {
            let s_bp = (b as u64) * 100_000 + 1;
            let e_bp = ((b + 1) as u64) * 100_000;
            for i in 0..n {
                let p = (next_u64() as usize) % n_parents;
                if b == causal_bin {
                    causal_parent[i] = p;
                }
                segments.push(AncestrySegment {
                    target_sample: i as u32,
                    hap: 0,
                    chrom_idx: 0,
                    start_bp: s_bp,
                    end_bp: e_bp,
                    start_cm: b as f32,
                    end_cm: (b + 1) as f32,
                    parent_sample: p as u32,
                    parent_hap: 0,
                    confidence: 0.99,
                    ambiguous_parents: 1,
                    n_markers: 20,
                });
            }
        }
        let tgt_layout = SampleHaplotypeLayout::from_inbred_flags(vec![true; n]);
        let bins = build_union_breakpoint_sparse_matrix(0, &tgt_layout, n_parents, &segments);

        let archive = HaploidSparseArchive {
            version: 1,
            mode: HaploidIbdMode::MultiParentAncestry,
            target_sample_ids: (0..n).map(|i| format!("S{:03}", i)).collect(),
            reference_sample_ids: (0..n_parents).map(|p| format!("P{}", p)).collect(),
            chrom_names: vec!["chr1".to_string()],
            payload: HaploidSparsePayload::MultiParentAncestry {
                segments,
                chrom_bins: vec![bins],
            },
        };

        let tmp_dir = std::env::temp_dir();
        let v1_path = tmp_dir.join("test_hgwas_v1_roundtrip.jhibd");
        let v2_path = tmp_dir.join("test_hgwas_v2_roundtrip.jhibd");
        let fvlmm_v1 = tmp_dir.join("test_hgwas_fvlmm_v1.tsv");
        let fvlmm_v2 = tmp_dir.join("test_hgwas_fvlmm_v2.tsv");
        let lmm_v2 = tmp_dir.join("test_hgwas_lmm_v2.tsv");

        write_sparse_archive_v1_with_threads(&v1_path, &archive, 1).expect("write v1");
        write_sparse_archive_v2_with_threads(&v2_path, &archive, &vec![true; n], n_bins as u64, 1)
            .expect("write v2");

        let meta_v1 = inspect_sparse_archive_meta(&v1_path).expect("meta v1");
        let meta_v2 = inspect_sparse_archive_meta(&v2_path).expect("meta v2");
        assert_eq!(meta_v1.n_bins, n_bins);
        assert_eq!(meta_v2.n_bins, n_bins);

        let compact_v1 = CompactHgwasArchive::open(v1_path.to_str().unwrap()).expect("open v1");
        let compact_v2 = CompactHgwasArchive::open(v2_path.to_str().unwrap()).expect("open v2");

        let (grm, n_grm, eff_bins) =
            build_ancestry_grm_from_compact(&compact_v2, None, 1, 1).expect("build GRM");
        assert_eq!(n_grm, n);
        assert_eq!(eff_bins, n_bins);

        let (s, u, _) = symmetric_eigh_f64_row_major(&grm, n).expect("eigh");
        let mut u_t = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                u_t[i * n + j] = u[j * n + i];
            }
        }
        let effects = [2.5, -2.0, 1.5, -1.5, 0.5, -1.0];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let eps = ((next_u64() >> 11) as f64) / ((1u64 << 53) as f64) * 0.4 - 0.2;
            y[i] = 10.0 + effects[causal_parent[i]] + eps;
        }
        let mut x_rot = vec![0.0_f64; n];
        let mut y_rot = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                let uij = u_t[i * n + j];
                x_rot[i] += uij;
                y_rot[i] += uij * y[j];
            }
        }

        let (n1, _) = scan_hgwas_compact_to_tsv(
            &compact_v1,
            fvlmm_v1.to_str().unwrap(),
            &s,
            &x_rot,
            &y_rot,
            &u_t,
            n,
            1,
            None,
            false,
            None,
            -5.0,
            5.0,
            30,
            1e-2,
            1,
            |_, _| {},
        )
        .expect("fvlmm v1");
        let (n2, _) = scan_hgwas_compact_to_tsv(
            &compact_v2,
            fvlmm_v2.to_str().unwrap(),
            &s,
            &x_rot,
            &y_rot,
            &u_t,
            n,
            1,
            None,
            false,
            None,
            -5.0,
            5.0,
            30,
            1e-2,
            1,
            |_, _| {},
        )
        .expect("fvlmm v2");
        let (n3, _) = scan_hgwas_compact_to_tsv(
            &compact_v2,
            lmm_v2.to_str().unwrap(),
            &s,
            &x_rot,
            &y_rot,
            &u_t,
            n,
            1,
            None,
            true,
            None,
            -5.0,
            5.0,
            30,
            1e-2,
            1,
            |_, _| {},
        )
        .expect("lmm v2");
        assert_eq!(n1, n_bins);
        assert_eq!(n2, n_bins);
        assert_eq!(n3, n_bins);

        let t1 = std::fs::read_to_string(&fvlmm_v1).unwrap();
        let t2 = std::fs::read_to_string(&fvlmm_v2).unwrap();
        let t3 = std::fs::read_to_string(&lmm_v2).unwrap();
        assert_eq!(
            t1, t2,
            "JHIBD01 and JHIBD02 must produce identical hGWAS TSV"
        );

        for text in [&t2, &t3] {
            let rows = parse_tsv_rows(text);
            assert_eq!(rows.len(), n_bins);
            assert!(rows[causal_bin].6 < 1e-6);
            let min_null = rows
                .iter()
                .enumerate()
                .filter(|&(idx, _)| idx != causal_bin)
                .map(|(_, r)| r.6)
                .fold(1.0_f64, f64::min);
            assert!(rows[causal_bin].6 < min_null * 1e-4);
        }

        // Also verify default `write_sparse_archive` emits JHIBD02 and loads cleanly.
        let v_def = tmp_dir.join("test_hgwas_default_roundtrip.jhibd");
        write_sparse_archive(&v_def, &archive).expect("write default");
        let loaded_def = CompactHgwasArchive::open(v_def.to_str().unwrap()).expect("open default");
        assert_eq!(loaded_def.total_bins, n_bins);

        let _ = std::fs::remove_file(v1_path);
        let _ = std::fs::remove_file(v2_path);
        let _ = std::fs::remove_file(v_def);
        let _ = std::fs::remove_file(fvlmm_v1);
        let _ = std::fs::remove_file(fvlmm_v2);
        let _ = std::fs::remove_file(lmm_v2);
    }
}
