use crate::bitwise::{and_popcount, bitand_assign, popcount};
use crate::gfcore::TxtSnpIter;
use crate::score::score_cont_corr_packed;
use memmap2::Mmap;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const BIN01_MAGIC: &[u8; 8] = b"JXBIN001";
const BIN01_HEADER_LEN: usize = 32;
const BEAM_PAR_MIN_TOTAL_CANDS: usize = 100_000;
const BEAM_SIMD_MIN_WORDS: usize = 16;
const BEAM_BNB_MIN_ROWS: usize = 128;
const BEAM_BNB_DENSITY_MAX: f64 = 0.08;

#[derive(Clone, Debug)]
pub struct BeamAndResult {
    pub selected_indices: Vec<usize>,
    pub score: f64,
    pub combined_bits: Vec<u64>,
}

#[derive(Clone, Debug)]
struct BeamNode {
    selected: Vec<usize>,
    combined: Vec<u64>,
    score: f64,
    last_index: usize,
}

#[derive(Clone, Debug)]
struct BinaryBeamNode {
    parent_slot: Option<usize>,
    last_index: usize,
    depth: usize,
    combined: Vec<u64>,
    tp: u64,
    score: f64,
}

#[derive(Clone, Copy, Debug)]
struct BinaryBeamRuntimeOptions {
    use_simd_fast_path: bool,
    use_upper_bound_prune: bool,
}

#[derive(Clone, Debug)]
struct BeamWindow {
    chrom: String,
    bp_start: i32,
    bp_end: i32,
    indices: Vec<usize>,
}

#[inline]
fn parse_env_bool(name: &str) -> bool {
    std::env::var(name).map_or(false, |v| {
        let t = v.trim().to_ascii_lowercase();
        matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on")
    })
}

#[inline]
fn beam_force_scalar_runtime() -> bool {
    static FORCE_SCALAR: OnceLock<bool> = OnceLock::new();
    *FORCE_SCALAR.get_or_init(|| parse_env_bool("JANUSX_BEAM_FORCE_SCALAR"))
}

#[inline]
fn beam_disable_bnb_runtime() -> bool {
    static DISABLE_BNB: OnceLock<bool> = OnceLock::new();
    *DISABLE_BNB.get_or_init(|| parse_env_bool("JANUSX_BEAM_DISABLE_BNB"))
}

#[inline]
fn beam_force_bnb_runtime() -> bool {
    static FORCE_BNB: OnceLock<bool> = OnceLock::new();
    *FORCE_BNB.get_or_init(|| parse_env_bool("JANUSX_BEAM_FORCE_BNB"))
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn beam_avx2_runtime_available() -> bool {
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn beam_neon_runtime_available() -> bool {
    static NEON: OnceLock<bool> = OnceLock::new();
    *NEON.get_or_init(|| std::arch::is_aarch64_feature_detected!("neon"))
}

#[inline]
fn binary_beam_runtime_options() -> BinaryBeamRuntimeOptions {
    BinaryBeamRuntimeOptions {
        use_simd_fast_path: !beam_force_scalar_runtime(),
        use_upper_bound_prune: !beam_disable_bnb_runtime(),
    }
}

#[inline]
fn words_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(64).max(1)
}

fn parse_bin01_header(bytes: &[u8], ctx: &str) -> Result<(usize, usize, usize, usize), String> {
    if bytes.len() < BIN01_HEADER_LEN {
        return Err(format!("{ctx}: BIN file too small"));
    }
    if &bytes[0..8] != BIN01_MAGIC {
        return Err(format!("{ctx}: invalid BIN magic (expected JXBIN001)"));
    }

    let n_rows_u64 = u64::from_le_bytes(
        bytes[8..16]
            .try_into()
            .map_err(|_| format!("{ctx}: malformed BIN header n_rows"))?,
    );
    let n_samples_u64 = u64::from_le_bytes(
        bytes[16..24]
            .try_into()
            .map_err(|_| format!("{ctx}: malformed BIN header n_samples"))?,
    );

    let n_rows = usize::try_from(n_rows_u64)
        .map_err(|_| format!("{ctx}: n_rows too large for this platform"))?;
    let n_samples = usize::try_from(n_samples_u64)
        .map_err(|_| format!("{ctx}: n_samples too large for this platform"))?;
    if n_samples == 0 {
        return Err(format!("{ctx}: n_samples is zero"));
    }

    let row_bytes = n_samples.div_ceil(8);
    let data_bytes = n_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| format!("{ctx}: BIN payload size overflow"))?;
    let expected = BIN01_HEADER_LEN
        .checked_add(data_bytes)
        .ok_or_else(|| format!("{ctx}: BIN file size overflow"))?;
    if bytes.len() < expected {
        return Err(format!(
            "{ctx}: BIN payload truncated (have {}, need at least {})",
            bytes.len(),
            expected
        ));
    }

    Ok((n_rows, n_samples, row_bytes, BIN01_HEADER_LEN))
}

fn load_bin01_as_u64_words(path: &str) -> Result<(Vec<u64>, usize, usize, usize), String> {
    let ctx = "load_bin01_as_u64_words";
    let file = File::open(path).map_err(|e| format!("{ctx}: failed to open {path}: {e}"))?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| format!("{ctx}: failed to mmap {}: {}", path, e))?;
    let (n_rows, n_samples, row_bytes, data_offset) = parse_bin01_header(&mmap[..], ctx)?;

    let row_words = words_for_samples(n_samples);
    let total_words = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: n_rows * row_words overflow"))?;
    let mut bits_flat = vec![0u64; total_words];

    let mask = tail_mask(n_samples);
    for r in 0..n_rows {
        let src_start = data_offset
            .checked_add(
                r.checked_mul(row_bytes)
                    .ok_or_else(|| format!("{ctx}: row byte offset overflow"))?,
            )
            .ok_or_else(|| format!("{ctx}: row byte offset overflow"))?;
        let src_end = src_start
            .checked_add(row_bytes)
            .ok_or_else(|| format!("{ctx}: row byte end overflow"))?;
        let src = &mmap[src_start..src_end];

        let dst_start = r
            .checked_mul(row_words)
            .ok_or_else(|| format!("{ctx}: row word offset overflow"))?;
        let dst = &mut bits_flat[dst_start..dst_start + row_words];

        for w in 0..row_words {
            let b0 = w * 8;
            if b0 >= row_bytes {
                break;
            }
            let b1 = (b0 + 8).min(row_bytes);
            let mut buf = [0u8; 8];
            let n = b1 - b0;
            buf[..n].copy_from_slice(&src[b0..b1]);
            dst[w] = u64::from_le_bytes(buf);
        }
        apply_tail_mask(dst, mask);
    }

    Ok((bits_flat, row_words, n_rows, n_samples))
}

fn load_bin01_selected_rows_as_u64_words(
    path: &str,
    row_indices: &[usize],
) -> Result<(Vec<u64>, usize, usize, usize), String> {
    let ctx = "load_bin01_selected_rows_as_u64_words";
    if row_indices.is_empty() {
        return Err(format!("{ctx}: row_indices is empty"));
    }
    let file = File::open(path).map_err(|e| format!("{ctx}: failed to open {path}: {e}"))?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| format!("{ctx}: failed to mmap {}: {}", path, e))?;
    let (n_rows_all, n_samples, row_bytes, data_offset) = parse_bin01_header(&mmap[..], ctx)?;

    let row_words = words_for_samples(n_samples);
    let n_rows = row_indices.len();
    let total_words = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: n_rows * row_words overflow"))?;
    let mut bits_flat = vec![0u64; total_words];

    let mask = tail_mask(n_samples);
    for (ri, &src_row_idx) in row_indices.iter().enumerate() {
        if src_row_idx >= n_rows_all {
            return Err(format!(
                "{ctx}: row index out of range: {} (n_rows={})",
                src_row_idx, n_rows_all
            ));
        }
        let src_start = data_offset
            .checked_add(
                src_row_idx
                    .checked_mul(row_bytes)
                    .ok_or_else(|| format!("{ctx}: row byte offset overflow"))?,
            )
            .ok_or_else(|| format!("{ctx}: row byte offset overflow"))?;
        let src_end = src_start
            .checked_add(row_bytes)
            .ok_or_else(|| format!("{ctx}: row byte end overflow"))?;
        let src = &mmap[src_start..src_end];

        let dst_start = ri
            .checked_mul(row_words)
            .ok_or_else(|| format!("{ctx}: row word offset overflow"))?;
        let dst = &mut bits_flat[dst_start..dst_start + row_words];

        for w in 0..row_words {
            let b0 = w * 8;
            if b0 >= row_bytes {
                break;
            }
            let b1 = (b0 + 8).min(row_bytes);
            let mut buf = [0u8; 8];
            let n = b1 - b0;
            buf[..n].copy_from_slice(&src[b0..b1]);
            dst[w] = u64::from_le_bytes(buf);
        }
        apply_tail_mask(dst, mask);
    }

    Ok((bits_flat, row_words, n_rows, n_samples))
}

#[inline]
fn append_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut os: OsString = prefix.as_os_str().to_os_string();
    os.push(suffix);
    PathBuf::from(os)
}

fn discover_site_sidecar(bin_path: &str) -> Option<String> {
    let p = Path::new(bin_path);
    let mut prefix = p.to_path_buf();
    let is_bin = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("bin"))
        .unwrap_or(false);
    if is_bin {
        prefix.set_extension("");
    }
    let candidates = [
        append_suffix(&prefix, ".bsite"),
        append_suffix(&prefix, ".site"),
        append_suffix(&prefix, ".site.tsv"),
        append_suffix(&prefix, ".site.txt"),
        append_suffix(&prefix, ".site.csv"),
        append_suffix(&prefix, ".sites.tsv"),
        append_suffix(&prefix, ".sites.txt"),
        append_suffix(&prefix, ".sites.csv"),
        append_suffix(&prefix, ".bin.site"),
    ];
    for cand in candidates {
        if cand.exists() {
            return Some(cand.to_string_lossy().to_string());
        }
    }
    None
}

#[inline]
fn split_chrom_base(chrom: &str) -> String {
    let s = chrom.trim();
    if s.len() > 2 && (s.ends_with("_1") || s.ends_with("_2")) {
        return s[..s.len() - 2].to_string();
    }
    if s.len() > 1 && (s.ends_with('-') || s.ends_with('+')) {
        return s[..s.len() - 1].to_string();
    }
    s.to_string()
}

fn global_window(n_rows: usize) -> Vec<BeamWindow> {
    vec![BeamWindow {
        chrom: "ALL".to_string(),
        bp_start: 0,
        bp_end: 0,
        indices: (0..n_rows).collect(),
    }]
}

fn build_windows_from_sites(
    sites: &[crate::gfcore::SiteInfo],
    n_rows: usize,
    extension: usize,
    step: usize,
) -> Vec<BeamWindow> {
    if n_rows == 0 {
        return Vec::new();
    }
    if extension == 0 || step == 0 {
        return global_window(n_rows);
    }
    if sites.is_empty() {
        return global_window(n_rows);
    }

    let mut groups: HashMap<String, Vec<(i32, usize)>> = HashMap::new();
    let mut chrom_order: Vec<String> = Vec::new();
    for (idx, site) in sites.iter().enumerate().take(n_rows) {
        let chrom = split_chrom_base(&site.chrom);
        if !groups.contains_key(&chrom) {
            groups.insert(chrom.clone(), Vec::new());
            chrom_order.push(chrom.clone());
        }
        if let Some(v) = groups.get_mut(&chrom) {
            v.push((site.pos, idx));
        }
    }
    if groups.is_empty() {
        return global_window(n_rows);
    }

    let mut windows: Vec<BeamWindow> = Vec::new();
    for chrom in chrom_order {
        let Some(mut pairs) = groups.remove(&chrom) else {
            continue;
        };
        if pairs.is_empty() {
            continue;
        }
        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let n = pairs.len();
        let bps: Vec<i32> = pairs.iter().map(|(bp, _)| *bp).collect();
        let idxs: Vec<usize> = pairs.iter().map(|(_, i)| *i).collect();
        let min_bp = bps[0];
        let max_bp = bps[n - 1];
        let mut l = 0usize;
        let mut r = 0usize;
        let mut center = min_bp;
        let mut prev_sig: Option<(usize, usize, usize)> = None;

        loop {
            let left_i64 = (center as i64)
                .saturating_sub(extension as i64)
                .max(min_bp as i64);
            let right_i64 = (center as i64)
                .saturating_add(extension as i64)
                .min(max_bp as i64);

            while l < n && (bps[l] as i64) < left_i64 {
                l += 1;
            }
            if r < l {
                r = l;
            }
            while r < n && (bps[r] as i64) <= right_i64 {
                r += 1;
            }

            if r > l {
                let chunk = idxs[l..r].to_vec();
                let sig = (chunk[0], chunk[chunk.len() - 1], chunk.len());
                if prev_sig.map_or(true, |s| s != sig) {
                    let bp_start = if left_i64 > i32::MAX as i64 {
                        i32::MAX
                    } else if left_i64 < i32::MIN as i64 {
                        i32::MIN
                    } else {
                        left_i64 as i32
                    };
                    let bp_end = if right_i64 > i32::MAX as i64 {
                        i32::MAX
                    } else if right_i64 < i32::MIN as i64 {
                        i32::MIN
                    } else {
                        right_i64 as i32
                    };
                    windows.push(BeamWindow {
                        chrom: chrom.clone(),
                        bp_start,
                        bp_end,
                        indices: chunk,
                    });
                    prev_sig = Some(sig);
                }
            }

            if center >= max_bp {
                break;
            }
            let next = (center as i64).saturating_add(step as i64);
            if next > i32::MAX as i64 {
                break;
            }
            center = next as i32;
        }

        let has_chrom_window = windows.last().map(|w| w.chrom == chrom).unwrap_or(false);
        if !has_chrom_window {
            windows.push(BeamWindow {
                chrom: chrom.clone(),
                bp_start: min_bp,
                bp_end: max_bp.saturating_add(1),
                indices: idxs,
            });
        }
    }

    if windows.is_empty() {
        return global_window(n_rows);
    }
    windows
}

fn load_windows_for_bin(
    bin_path: &str,
    n_rows: usize,
    extension: usize,
    step: usize,
) -> Result<Vec<BeamWindow>, String> {
    let sidecar = discover_site_sidecar(bin_path);
    let Some(sidecar_path) = sidecar else {
        return Ok(global_window(n_rows));
    };
    if sidecar_path.to_ascii_lowercase().ends_with(".bin.site") {
        // Legacy k-mer sidecar has no genomic CHR/BP metadata.
        return Ok(global_window(n_rows));
    }

    let it = TxtSnpIter::new(bin_path, None)?;
    Ok(build_windows_from_sites(&it.sites, n_rows, extension, step))
}

fn gather_rows_by_indices(
    bits_flat: &[u64],
    row_words: usize,
    row_indices: &[usize],
) -> Result<Vec<u64>, String> {
    let ctx = "gather_rows_by_indices";
    if row_words == 0 {
        return Err(format!("{ctx}: row_words must be > 0"));
    }
    if row_indices.is_empty() {
        return Ok(Vec::new());
    }
    let n_rows_all = bits_flat.len() / row_words;
    let total_words = row_indices
        .len()
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: output size overflow"))?;
    let mut out = vec![0u64; total_words];
    for (dst_r, &src_r) in row_indices.iter().enumerate() {
        if src_r >= n_rows_all {
            return Err(format!(
                "{ctx}: row index out of range: {} (n_rows={})",
                src_r, n_rows_all
            ));
        }
        let src_start = src_r
            .checked_mul(row_words)
            .ok_or_else(|| format!("{ctx}: source offset overflow"))?;
        let dst_start = dst_r
            .checked_mul(row_words)
            .ok_or_else(|| format!("{ctx}: target offset overflow"))?;
        out[dst_start..dst_start + row_words]
            .copy_from_slice(&bits_flat[src_start..src_start + row_words]);
    }
    Ok(out)
}

#[inline]
fn score_key(s: f64) -> f64 {
    if s.is_nan() {
        f64::NEG_INFINITY
    } else {
        s
    }
}

#[inline]
fn cmp_nodes(a: &BeamNode, b: &BeamNode) -> Ordering {
    // Better node comes first (Ordering::Less)
    let sa = score_key(a.score);
    let sb = score_key(b.score);
    match sb.partial_cmp(&sa).unwrap_or(Ordering::Equal) {
        Ordering::Equal => match a.selected.len().cmp(&b.selected.len()) {
            Ordering::Equal => a.selected.cmp(&b.selected),
            other => other,
        },
        other => other,
    }
}

#[inline]
fn push_top_k_streaming(nodes: &mut Vec<BeamNode>, cand: BeamNode, k: usize) {
    if k == 0 {
        return;
    }
    if nodes.len() < k {
        nodes.push(cand);
        return;
    }

    // Current top-k is full: find worst node in-place (k is typically small).
    let mut worst_idx = 0usize;
    for i in 1..nodes.len() {
        if cmp_nodes(&nodes[i], &nodes[worst_idx]) == Ordering::Greater {
            worst_idx = i;
        }
    }

    // Replace the worst only if incoming candidate is better.
    if cmp_nodes(&cand, &nodes[worst_idx]) == Ordering::Less {
        nodes[worst_idx] = cand;
    }
}

#[inline]
fn cmp_binary_nodes(a: &BinaryBeamNode, b: &BinaryBeamNode) -> Ordering {
    let sa = score_key(a.score);
    let sb = score_key(b.score);
    match sb.partial_cmp(&sa).unwrap_or(Ordering::Equal) {
        Ordering::Equal => match a.depth.cmp(&b.depth) {
            Ordering::Equal => match a.parent_slot.cmp(&b.parent_slot) {
                Ordering::Equal => a.last_index.cmp(&b.last_index),
                other => other,
            },
            other => other,
        },
        other => other,
    }
}

#[inline]
fn push_top_k_streaming_binary(
    nodes: &mut Vec<BinaryBeamNode>,
    cand: BinaryBeamNode,
    k: usize,
) -> bool {
    if k == 0 {
        return false;
    }
    if nodes.len() < k {
        nodes.push(cand);
        return true;
    }

    let mut worst_idx = 0usize;
    for i in 1..nodes.len() {
        if cmp_binary_nodes(&nodes[i], &nodes[worst_idx]) == Ordering::Greater {
            worst_idx = i;
        }
    }

    if cmp_binary_nodes(&cand, &nodes[worst_idx]) == Ordering::Less {
        nodes[worst_idx] = cand;
        return true;
    }
    false
}

#[inline]
fn worst_score_key_in_binary_topk(nodes: &[BinaryBeamNode], k: usize) -> Option<f64> {
    if nodes.len() < k || nodes.is_empty() {
        return None;
    }
    let mut worst = f64::INFINITY;
    for n in nodes {
        let s = score_key(n.score);
        if s < worst {
            worst = s;
        }
    }
    Some(worst)
}

#[inline]
fn mcc_upper_bound_from_tp(tp: u64, y_pos: u64, n_samples: usize) -> f64 {
    let fnv = y_pos.saturating_sub(tp);
    let tn = (n_samples as u64).saturating_sub(y_pos);
    mcc_from_confusion(tp, tn, 0, fnv)
}

#[inline]
fn should_parallel_expand(beam_len: usize, n_rows: usize) -> bool {
    if rayon::current_num_threads() <= 1 || beam_len <= 1 {
        return false;
    }
    beam_len.saturating_mul(n_rows) >= BEAM_PAR_MIN_TOTAL_CANDS
}

#[inline]
fn validate_bit_matrix(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    ctx: &str,
) -> Result<usize, String> {
    if n_rows == 0 {
        return Err(format!("{ctx}: n_rows must be > 0"));
    }
    if row_words == 0 {
        return Err(format!("{ctx}: row_words must be > 0"));
    }
    if n_samples == 0 {
        return Err(format!("{ctx}: n_samples must be > 0"));
    }

    let needed_words = words_for_samples(n_samples);
    if row_words < needed_words {
        return Err(format!(
            "{ctx}: row_words={} is smaller than required {} for n_samples={}",
            row_words, needed_words, n_samples
        ));
    }

    let total_words = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: n_rows * row_words overflow"))?;
    if bits_flat.len() < total_words {
        return Err(format!(
            "{ctx}: bits length={} smaller than n_rows*row_words={}",
            bits_flat.len(),
            total_words
        ));
    }

    Ok(needed_words)
}

#[inline]
fn tail_mask(n_samples: usize) -> Option<u64> {
    let rem = n_samples & 63;
    if rem == 0 {
        None
    } else {
        Some((1u64 << rem) - 1u64)
    }
}

#[inline]
fn apply_tail_mask(bits: &mut [u64], mask: Option<u64>) {
    if let Some(m) = mask {
        if let Some(last) = bits.last_mut() {
            *last &= m;
        }
    }
}

#[inline]
fn row_prefix<'a>(
    bits_flat: &'a [u64],
    row_words: usize,
    row_idx: usize,
    needed_words: usize,
) -> &'a [u64] {
    let start = row_idx * row_words;
    &bits_flat[start..start + needed_words]
}

#[inline]
fn estimate_density_for_bnb(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    needed_words: usize,
    n_samples: usize,
) -> f64 {
    let sample_rows = n_rows.min(128);
    if sample_rows == 0 {
        return 1.0;
    }
    let step = (n_rows / sample_rows).max(1);
    let mut ones = 0u64;
    let mut seen = 0usize;
    let mut r = 0usize;
    while r < n_rows && seen < sample_rows {
        let row = row_prefix(bits_flat, row_words, r, needed_words);
        for &w in row {
            ones += w.count_ones() as u64;
        }
        seen += 1;
        r = r.saturating_add(step);
    }
    if seen == 0 {
        return 1.0;
    }
    let denom = (seen as f64) * (n_samples as f64);
    if denom > 0.0 {
        (ones as f64) / denom
    } else {
        1.0
    }
}

#[inline]
fn should_enable_upper_bound_prune(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    needed_words: usize,
    n_samples: usize,
    max_depth: usize,
) -> bool {
    if beam_force_bnb_runtime() {
        return true;
    }
    if max_depth <= 1 || n_rows < BEAM_BNB_MIN_ROWS {
        return false;
    }
    estimate_density_for_bnb(bits_flat, row_words, n_rows, needed_words, n_samples)
        <= BEAM_BNB_DENSITY_MAX
}

/// Beam search over AND-combinations of bit-vectors.
///
/// - `bits_flat`: row-major packed bit-matrix, shape `(n_rows, row_words)`.
/// - `max_pick`: up to this many vectors can be selected.
/// - `beam_width`: keep top-K states at each depth.
/// - `score_fn`: larger score is better.
///
/// Returns the best state found across depth 1..=max_pick.
pub fn beam_search_and_with_score<F>(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
    score_fn: F,
) -> Result<BeamAndResult, String>
where
    F: Fn(&[u64]) -> f64 + Sync,
{
    let ctx = "beam_search_and_with_score";
    let needed_words = validate_bit_matrix(bits_flat, row_words, n_rows, n_samples, ctx)?;

    if max_pick == 0 {
        return Err(format!("{ctx}: max_pick must be > 0"));
    }
    if beam_width == 0 {
        return Err(format!("{ctx}: beam_width must be > 0"));
    }

    let max_depth = max_pick.min(n_rows);
    let mask = tail_mask(n_samples);

    let layer_cap = beam_width.min(n_rows);
    let mut beam: Vec<BeamNode> = Vec::with_capacity(layer_cap);
    for i in 0..n_rows {
        let mut combined = row_prefix(bits_flat, row_words, i, needed_words).to_vec();
        apply_tail_mask(&mut combined, mask);
        let score = score_fn(&combined);
        push_top_k_streaming(
            &mut beam,
            BeamNode {
                selected: vec![i],
                combined,
                score,
                last_index: i,
            },
            layer_cap,
        );
    }

    if beam.is_empty() {
        return Err(format!("{ctx}: no candidates"));
    }
    beam.sort_by(cmp_nodes);

    let mut best = beam[0].clone();

    for _depth in 2..=max_depth {
        let next_cap = beam_width.min(n_rows);
        let mut next: Vec<BeamNode> = if should_parallel_expand(beam.len(), n_rows) {
            let local_tops: Vec<Vec<BeamNode>> = (0..beam.len())
                .into_par_iter()
                .map(|bi| {
                    let node = &beam[bi];
                    let mut local: Vec<BeamNode> = Vec::with_capacity(next_cap);
                    // Canonicalize combinations by requiring strictly increasing indices.
                    // This avoids duplicate permutations for commutative AND.
                    for cand in (node.last_index + 1)..n_rows {
                        let row = row_prefix(bits_flat, row_words, cand, needed_words);
                        let mut combined = node.combined.clone();
                        bitand_assign(&mut combined, row);

                        let score = score_fn(&combined);
                        let mut selected = node.selected.clone();
                        selected.push(cand);

                        push_top_k_streaming(
                            &mut local,
                            BeamNode {
                                selected,
                                combined,
                                score,
                                last_index: cand,
                            },
                            beam_width,
                        );
                    }
                    local
                })
                .collect();

            let mut merged: Vec<BeamNode> = Vec::with_capacity(next_cap);
            for local in local_tops {
                for cand in local {
                    push_top_k_streaming(&mut merged, cand, beam_width);
                }
            }
            merged
        } else {
            let mut seq: Vec<BeamNode> = Vec::with_capacity(next_cap);
            for node in &beam {
                // Canonicalize combinations by requiring strictly increasing indices.
                // This avoids duplicate permutations for commutative AND.
                for cand in (node.last_index + 1)..n_rows {
                    let row = row_prefix(bits_flat, row_words, cand, needed_words);
                    let mut combined = node.combined.clone();
                    bitand_assign(&mut combined, row);

                    let score = score_fn(&combined);
                    let mut selected = node.selected.clone();
                    selected.push(cand);

                    push_top_k_streaming(
                        &mut seq,
                        BeamNode {
                            selected,
                            combined,
                            score,
                            last_index: cand,
                        },
                        beam_width,
                    );
                }
            }
            seq
        };

        if next.is_empty() {
            break;
        }

        next.sort_by(cmp_nodes);
        beam = next;
        if cmp_nodes(&beam[0], &best) == Ordering::Less {
            best = beam[0].clone();
        }
    }

    Ok(BeamAndResult {
        selected_indices: best.selected,
        score: best.score,
        combined_bits: best.combined,
    })
}

#[inline]
fn validate_binary_y(y: &[u8], n_samples: usize, ctx: &str) -> Result<(), String> {
    if y.len() < n_samples {
        return Err(format!(
            "{ctx}: y length={} smaller than n_samples={}",
            y.len(),
            n_samples
        ));
    }
    if let Some((idx, bad)) = y.iter().take(n_samples).enumerate().find(|(_, v)| **v > 1) {
        return Err(format!("{ctx}: y must be binary 0/1; found y[{idx}]={bad}"));
    }
    Ok(())
}

#[inline]
fn validate_continuous_y(y: &[f64], n_samples: usize, ctx: &str) -> Result<(), String> {
    if y.len() < n_samples {
        return Err(format!(
            "{ctx}: y length={} smaller than n_samples={}",
            y.len(),
            n_samples
        ));
    }
    if let Some((idx, bad)) = y
        .iter()
        .take(n_samples)
        .enumerate()
        .find(|(_, v)| !v.is_finite())
    {
        return Err(format!(
            "{ctx}: y must be finite for continuous mode; found y[{idx}]={bad}"
        ));
    }
    Ok(())
}

#[inline]
fn pack_binary_y_to_bits(y: &[u8], n_samples: usize) -> (Vec<u64>, u64) {
    let words = words_for_samples(n_samples);
    let mut out = vec![0u64; words];
    let mut y_pos = 0u64;

    for (i, &yv) in y.iter().take(n_samples).enumerate() {
        if yv != 0 {
            out[i >> 6] |= 1u64 << (i & 63);
            y_pos += 1;
        }
    }

    (out, y_pos)
}

#[inline]
fn masked_xpos_tp(bits: &[u64], y_bits: &[u64], n_samples: usize) -> (u64, u64) {
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;

    let mut x_pos = 0u64;
    let mut tp = 0u64;

    if full_words > 0 {
        x_pos += popcount(&bits[..full_words]);
        tp += and_popcount(&bits[..full_words], &y_bits[..full_words]);
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        let xb = bits[full_words] & mask;
        let yb = y_bits[full_words] & mask;
        x_pos += xb.count_ones() as u64;
        tp += (xb & yb).count_ones() as u64;
    }

    (x_pos, tp)
}

#[inline]
fn mcc_from_confusion(tp: u64, tn: u64, fp: u64, fnv: u64) -> f64 {
    let num = (tp as f64) * (tn as f64) - (fp as f64) * (fnv as f64);
    let a = (tp + fp) as f64;
    let b = (tp + fnv) as f64;
    let c = (tn + fp) as f64;
    let d = (tn + fnv) as f64;
    let den = (a * b * c * d).sqrt();
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

#[inline]
fn mcc_from_xpos_tp(x_pos: u64, tp: u64, y_pos: u64, n_samples: usize) -> f64 {
    let fp = x_pos.saturating_sub(tp);
    let fnv = y_pos.saturating_sub(tp);
    let tn = (n_samples as u64).saturating_sub(tp + fp + fnv);
    mcc_from_confusion(tp, tn, fp, fnv)
}

#[inline]
fn abs_corr_from_x_bits(y: &[f64], bits: &[u64], n_samples: usize) -> f64 {
    score_cont_corr_packed(y, bits, n_samples).abs()
}

#[inline]
fn and_assign_xpos_tp_full_words_scalar(
    dst: &mut [u64],
    rhs: &[u64],
    y_bits: &[u64],
) -> (u64, u64) {
    debug_assert_eq!(dst.len(), rhs.len());
    debug_assert_eq!(dst.len(), y_bits.len());
    let mut x_pos = 0u64;
    let mut tp = 0u64;
    let mut i = 0usize;
    let n = dst.len();

    while i + 4 <= n {
        let v0 = dst[i] & rhs[i];
        let v1 = dst[i + 1] & rhs[i + 1];
        let v2 = dst[i + 2] & rhs[i + 2];
        let v3 = dst[i + 3] & rhs[i + 3];

        dst[i] = v0;
        dst[i + 1] = v1;
        dst[i + 2] = v2;
        dst[i + 3] = v3;

        x_pos += v0.count_ones() as u64
            + v1.count_ones() as u64
            + v2.count_ones() as u64
            + v3.count_ones() as u64;
        tp += (v0 & y_bits[i]).count_ones() as u64
            + (v1 & y_bits[i + 1]).count_ones() as u64
            + (v2 & y_bits[i + 2]).count_ones() as u64
            + (v3 & y_bits[i + 3]).count_ones() as u64;
        i += 4;
    }

    while i < n {
        let v = dst[i] & rhs[i];
        dst[i] = v;
        x_pos += v.count_ones() as u64;
        tp += (v & y_bits[i]).count_ones() as u64;
        i += 1;
    }

    (x_pos, tp)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn popcount_u8x32_avx2(
    v: core::arch::x86_64::__m256i,
    lut4: core::arch::x86_64::__m256i,
    low_mask: core::arch::x86_64::__m256i,
    zero: core::arch::x86_64::__m256i,
) -> u64 {
    use core::arch::x86_64::*;
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    let cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut4, lo), _mm256_shuffle_epi8(lut4, hi));
    let sum64 = _mm256_sad_epu8(cnt, zero);
    (_mm256_extract_epi64(sum64, 0) as u64)
        + (_mm256_extract_epi64(sum64, 1) as u64)
        + (_mm256_extract_epi64(sum64, 2) as u64)
        + (_mm256_extract_epi64(sum64, 3) as u64)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn and_assign_xpos_tp_full_words_avx2(
    dst: &mut [u64],
    rhs: &[u64],
    y_bits: &[u64],
) -> (u64, u64) {
    use core::arch::x86_64::*;
    debug_assert_eq!(dst.len(), rhs.len());
    debug_assert_eq!(dst.len(), y_bits.len());

    let lut4 = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f_i8);
    let zero = _mm256_setzero_si256();

    let mut x_pos = 0u64;
    let mut tp = 0u64;
    let mut i = 0usize;
    let n = dst.len();

    while i + 4 <= n {
        let d = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
        let r = _mm256_loadu_si256(rhs.as_ptr().add(i) as *const __m256i);
        let v = _mm256_and_si256(d, r);
        _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, v);
        x_pos += popcount_u8x32_avx2(v, lut4, low_mask, zero);

        let yv = _mm256_loadu_si256(y_bits.as_ptr().add(i) as *const __m256i);
        tp += popcount_u8x32_avx2(_mm256_and_si256(v, yv), lut4, low_mask, zero);
        i += 4;
    }

    let (tx, ttp) = and_assign_xpos_tp_full_words_scalar(&mut dst[i..], &rhs[i..], &y_bits[i..]);
    (x_pos + tx, tp + ttp)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn popcount_u64x2_neon(v: core::arch::aarch64::uint64x2_t) -> u64 {
    use core::arch::aarch64::*;
    let cnt8 = vcntq_u8(vreinterpretq_u8_u64(v));
    let sum16 = vpaddlq_u8(cnt8);
    let sum32 = vpaddlq_u16(sum16);
    let sum64 = vpaddlq_u32(sum32);
    vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1)
}

#[cfg(target_arch = "aarch64")]
unsafe fn and_assign_xpos_tp_full_words_neon(
    dst: &mut [u64],
    rhs: &[u64],
    y_bits: &[u64],
) -> (u64, u64) {
    use core::arch::aarch64::*;
    debug_assert_eq!(dst.len(), rhs.len());
    debug_assert_eq!(dst.len(), y_bits.len());

    let mut x_pos = 0u64;
    let mut tp = 0u64;
    let mut i = 0usize;
    let n = dst.len();

    while i + 2 <= n {
        let d = vld1q_u64(dst.as_ptr().add(i));
        let r = vld1q_u64(rhs.as_ptr().add(i));
        let v = vandq_u64(d, r);
        vst1q_u64(dst.as_mut_ptr().add(i), v);
        x_pos += popcount_u64x2_neon(v);

        let yv = vld1q_u64(y_bits.as_ptr().add(i));
        tp += popcount_u64x2_neon(vandq_u64(v, yv));
        i += 2;
    }

    let (tx, ttp) = and_assign_xpos_tp_full_words_scalar(&mut dst[i..], &rhs[i..], &y_bits[i..]);
    (x_pos + tx, tp + ttp)
}

#[inline]
fn and_assign_xpos_tp_full_words_dispatch(
    dst: &mut [u64],
    rhs: &[u64],
    y_bits: &[u64],
    use_simd_fast_path: bool,
) -> (u64, u64) {
    if use_simd_fast_path && dst.len() >= BEAM_SIMD_MIN_WORDS {
        #[cfg(target_arch = "x86_64")]
        {
            if beam_avx2_runtime_available() {
                // SAFETY: gated by runtime AVX2 detection.
                return unsafe { and_assign_xpos_tp_full_words_avx2(dst, rhs, y_bits) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if beam_neon_runtime_available() {
                // SAFETY: gated by runtime NEON detection.
                return unsafe { and_assign_xpos_tp_full_words_neon(dst, rhs, y_bits) };
            }
        }
    }
    and_assign_xpos_tp_full_words_scalar(dst, rhs, y_bits)
}

#[inline]
fn and_assign_xpos_tp_inplace(
    combined: &mut [u64],
    rhs: &[u64],
    y_bits: &[u64],
    n_samples: usize,
    use_simd_fast_path: bool,
) -> (u64, u64) {
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;
    let mut x_pos = 0u64;
    let mut tp = 0u64;

    if full_words > 0 {
        let (xx, tt) = and_assign_xpos_tp_full_words_dispatch(
            &mut combined[..full_words],
            &rhs[..full_words],
            &y_bits[..full_words],
            use_simd_fast_path,
        );
        x_pos += xx;
        tp += tt;
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        let v = (combined[full_words] & rhs[full_words]) & mask;
        combined[full_words] = v;
        x_pos += v.count_ones() as u64;
        tp += (v & y_bits[full_words]).count_ones() as u64;
    }

    (x_pos, tp)
}

fn reconstruct_selected_from_layers(
    layers: &[Vec<(Option<usize>, usize)>],
    best_depth: usize,
    best_slot: usize,
) -> Vec<usize> {
    let mut out_rev: Vec<usize> = Vec::with_capacity(best_depth);
    let mut slot = best_slot;
    for d in (1..=best_depth).rev() {
        let (parent, last) = layers[d - 1][slot];
        out_rev.push(last);
        slot = parent.unwrap_or(0);
    }
    out_rev.reverse();
    out_rev
}

/// Beam search for binary-y objective using MCC score.
///
/// The score of each AND-combined bit-vector `X` is `MCC(y, X)`.
fn beam_search_and_binary_mcc_with_options(
    y: &[u8],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
    options: BinaryBeamRuntimeOptions,
) -> Result<BeamAndResult, String> {
    let ctx = "beam_search_and_binary_mcc";
    validate_binary_y(y, n_samples, ctx)?;
    let needed_words = validate_bit_matrix(bits_flat, row_words, n_rows, n_samples, ctx)?;
    if max_pick == 0 {
        return Err(format!("{ctx}: max_pick must be > 0"));
    }
    if beam_width == 0 {
        return Err(format!("{ctx}: beam_width must be > 0"));
    }
    let max_depth = max_pick.min(n_rows);
    let use_upper_bound_prune = options.use_upper_bound_prune
        && should_enable_upper_bound_prune(
            bits_flat,
            row_words,
            n_rows,
            needed_words,
            n_samples,
            max_depth,
        );

    let (y_bits, y_pos) = pack_binary_y_to_bits(y, n_samples);
    let mask = tail_mask(n_samples);

    let layer_cap = beam_width.min(n_rows);
    let mut beam: Vec<BinaryBeamNode> = Vec::with_capacity(layer_cap);
    for i in 0..n_rows {
        let mut combined = row_prefix(bits_flat, row_words, i, needed_words).to_vec();
        apply_tail_mask(&mut combined, mask);
        let (x_pos, tp) = masked_xpos_tp(&combined, &y_bits, n_samples);
        let score = mcc_from_xpos_tp(x_pos, tp, y_pos, n_samples);
        push_top_k_streaming_binary(
            &mut beam,
            BinaryBeamNode {
                parent_slot: None,
                last_index: i,
                depth: 1,
                combined,
                tp,
                score,
            },
            layer_cap,
        );
    }
    if beam.is_empty() {
        return Err(format!("{ctx}: no candidates"));
    }
    beam.sort_by(cmp_binary_nodes);

    let mut layers: Vec<Vec<(Option<usize>, usize)>> = Vec::with_capacity(max_depth);
    layers.push(
        beam.iter()
            .map(|n| (n.parent_slot, n.last_index))
            .collect::<Vec<_>>(),
    );

    let mut best_depth = 1usize;
    let mut best_slot = 0usize;
    let mut best_score = beam[0].score;
    let mut best_combined = beam[0].combined.clone();

    for depth in 2..=max_depth {
        let next_cap = beam_width.min(n_rows);
        let can_descend_more = depth < max_depth;
        let best_score_cut = score_key(best_score);
        let mut next: Vec<BinaryBeamNode> = if should_parallel_expand(beam.len(), n_rows) {
            let local_tops: Vec<Vec<BinaryBeamNode>> = (0..beam.len())
                .into_par_iter()
                .map(|bi| {
                    let node = &beam[bi];
                    if use_upper_bound_prune {
                        let parent_ub =
                            score_key(mcc_upper_bound_from_tp(node.tp, y_pos, n_samples));
                        if parent_ub <= best_score_cut {
                            return Vec::new();
                        }
                    }

                    let mut local: Vec<BinaryBeamNode> = Vec::with_capacity(next_cap);
                    for cand in (node.last_index + 1)..n_rows {
                        let row = row_prefix(bits_flat, row_words, cand, needed_words);
                        let mut combined = node.combined.clone();
                        let (x_pos, tp) = and_assign_xpos_tp_inplace(
                            &mut combined,
                            row,
                            &y_bits,
                            n_samples,
                            options.use_simd_fast_path,
                        );
                        if use_upper_bound_prune && can_descend_more {
                            let child_ub = score_key(mcc_upper_bound_from_tp(tp, y_pos, n_samples));
                            if child_ub <= best_score_cut {
                                continue;
                            }
                        }
                        let score = mcc_from_xpos_tp(x_pos, tp, y_pos, n_samples);
                        push_top_k_streaming_binary(
                            &mut local,
                            BinaryBeamNode {
                                parent_slot: Some(bi),
                                last_index: cand,
                                depth,
                                combined,
                                tp,
                                score,
                            },
                            beam_width,
                        );
                    }
                    local
                })
                .collect();
            let mut merged: Vec<BinaryBeamNode> = Vec::with_capacity(next_cap);
            for local in local_tops {
                for cand in local {
                    push_top_k_streaming_binary(&mut merged, cand, beam_width);
                }
            }
            merged
        } else {
            let mut seq: Vec<BinaryBeamNode> = Vec::with_capacity(next_cap);
            let mut layer_score_cut = worst_score_key_in_binary_topk(&seq, beam_width);
            for (bi, node) in beam.iter().enumerate() {
                if use_upper_bound_prune {
                    let parent_ub = score_key(mcc_upper_bound_from_tp(node.tp, y_pos, n_samples));
                    if parent_ub <= best_score_cut {
                        continue;
                    }
                    if let Some(cut) = layer_score_cut {
                        if parent_ub < cut {
                            continue;
                        }
                    }
                }

                for cand in (node.last_index + 1)..n_rows {
                    let row = row_prefix(bits_flat, row_words, cand, needed_words);
                    let mut combined = node.combined.clone();
                    let (x_pos, tp) = and_assign_xpos_tp_inplace(
                        &mut combined,
                        row,
                        &y_bits,
                        n_samples,
                        options.use_simd_fast_path,
                    );
                    if use_upper_bound_prune && can_descend_more {
                        let child_ub = score_key(mcc_upper_bound_from_tp(tp, y_pos, n_samples));
                        if child_ub <= best_score_cut {
                            continue;
                        }
                        if let Some(cut) = layer_score_cut {
                            if child_ub < cut {
                                continue;
                            }
                        }
                    }
                    let score = mcc_from_xpos_tp(x_pos, tp, y_pos, n_samples);
                    let inserted = push_top_k_streaming_binary(
                        &mut seq,
                        BinaryBeamNode {
                            parent_slot: Some(bi),
                            last_index: cand,
                            depth,
                            combined,
                            tp,
                            score,
                        },
                        beam_width,
                    );
                    if use_upper_bound_prune && inserted {
                        layer_score_cut = worst_score_key_in_binary_topk(&seq, beam_width);
                    }
                }
            }
            seq
        };

        if next.is_empty() {
            break;
        }
        next.sort_by(cmp_binary_nodes);

        layers.push(
            next.iter()
                .map(|n| (n.parent_slot, n.last_index))
                .collect::<Vec<_>>(),
        );

        let top = &next[0];
        let top_score = score_key(top.score);
        let best_score_key = score_key(best_score);
        if top_score > best_score_key || (top_score == best_score_key && depth < best_depth) {
            best_depth = depth;
            best_slot = 0;
            best_score = top.score;
            best_combined = top.combined.clone();
        }
        beam = next;
    }

    let selected = reconstruct_selected_from_layers(&layers, best_depth, best_slot);
    Ok(BeamAndResult {
        selected_indices: selected,
        score: best_score,
        combined_bits: best_combined,
    })
}

pub fn beam_search_and_binary_mcc(
    y: &[u8],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
) -> Result<BeamAndResult, String> {
    beam_search_and_binary_mcc_with_options(
        y,
        bits_flat,
        row_words,
        n_rows,
        n_samples,
        max_pick,
        beam_width,
        binary_beam_runtime_options(),
    )
}

pub fn beam_search_and_continuous_abs_corr(
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
) -> Result<BeamAndResult, String> {
    let ctx = "beam_search_and_continuous_abs_corr";
    validate_continuous_y(y, n_samples, ctx)?;
    beam_search_and_with_score(
        bits_flat,
        row_words,
        n_rows,
        n_samples,
        max_pick,
        beam_width,
        |combined| abs_corr_from_x_bits(y, combined, n_samples),
    )
}

#[inline]
fn readonly_u8_to_cow<'a>(arr: &'a PyReadonlyArray1<'a, u8>) -> std::borrow::Cow<'a, [u8]> {
    match arr.as_slice() {
        Ok(s) => std::borrow::Cow::Borrowed(s),
        Err(_) => std::borrow::Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

#[inline]
fn readonly_f64_to_cow<'a>(arr: &'a PyReadonlyArray1<'a, f64>) -> std::borrow::Cow<'a, [f64]> {
    match arr.as_slice() {
        Ok(s) => std::borrow::Cow::Borrowed(s),
        Err(_) => std::borrow::Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

type BeamWindowScanOutput = (
    Vec<usize>,
    f64,
    String,
    i32,
    i32,
    usize,
    usize,
    Vec<(usize, String, i32, i32, usize, f64, Vec<usize>)>,
);

fn scan_windows_with_objective<F>(
    bin_path: &str,
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    extension: usize,
    step_v: usize,
    return_window_results: bool,
    search_fn: F,
) -> Result<BeamWindowScanOutput, String>
where
    F: Fn(&[u64], usize) -> Result<BeamAndResult, String>,
{
    let windows = load_windows_for_bin(bin_path, n_rows, extension, step_v)?;
    if windows.is_empty() {
        return Err("no windows available".to_string());
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut best_selected: Vec<usize> = Vec::new();
    let mut best_chrom = "ALL".to_string();
    let mut best_bp_start = 0i32;
    let mut best_bp_end = 0i32;
    let mut best_n_candidates = 0usize;
    let mut evaluated = 0usize;

    let mut trace: Vec<(usize, String, i32, i32, usize, f64, Vec<usize>)> = Vec::new();
    for (wi0, win) in windows.iter().enumerate() {
        if win.indices.is_empty() {
            continue;
        }
        let wi = wi0 + 1;

        let local_out = if win.indices.len() == n_rows {
            search_fn(bits_flat, n_rows)?
        } else {
            let sub = gather_rows_by_indices(bits_flat, row_words, &win.indices)?;
            search_fn(&sub, win.indices.len())?
        };

        let selected_global = local_out
            .selected_indices
            .iter()
            .map(|&i| win.indices[i])
            .collect::<Vec<_>>();
        let score = local_out.score;
        evaluated += 1;

        let better = score_key(score) > score_key(best_score);
        let tie_better = (score_key(score) - score_key(best_score)).abs() <= 1e-12
            && selected_global.len() < best_selected.len();
        if best_selected.is_empty() || better || tie_better {
            best_score = score;
            best_selected = selected_global.clone();
            best_chrom = win.chrom.clone();
            best_bp_start = win.bp_start;
            best_bp_end = win.bp_end;
            best_n_candidates = win.indices.len();
        }

        if return_window_results {
            trace.push((
                wi,
                win.chrom.clone(),
                win.bp_start,
                win.bp_end,
                win.indices.len(),
                score,
                selected_global,
            ));
        }
    }

    if best_selected.is_empty() {
        return Err("no valid window/candidate found".to_string());
    }

    Ok((
        best_selected,
        best_score,
        best_chrom,
        best_bp_start,
        best_bp_end,
        best_n_candidates,
        evaluated,
        trace,
    ))
}

/// Run MCC-based AND beam search directly on JXBIN001 (`jx gformat` .bin) input.
///
/// Returns `(selected_indices, best_score)`.
#[pyfunction(name = "beam_search_and_binary_mcc_bin")]
#[pyo3(signature = (bin_path, y, max_pick=3, beam_width=128))]
pub fn beam_search_and_binary_mcc_bin_py(
    bin_path: String,
    y: PyReadonlyArray1<'_, u8>,
    max_pick: usize,
    beam_width: usize,
) -> PyResult<(Vec<usize>, f64)> {
    let yv = readonly_u8_to_cow(&y);
    let (bits_flat, row_words, n_rows, n_samples) =
        load_bin01_as_u64_words(&bin_path).map_err(PyRuntimeError::new_err)?;
    if yv.len() < n_samples {
        return Err(PyRuntimeError::new_err(format!(
            "beam_search_and_binary_mcc_bin: y length={} is smaller than bin n_samples={}",
            yv.len(),
            n_samples
        )));
    }
    let out = beam_search_and_binary_mcc(
        yv.as_ref(),
        &bits_flat,
        row_words,
        n_rows,
        n_samples,
        max_pick,
        beam_width,
    )
    .map_err(PyRuntimeError::new_err)?;
    Ok((out.selected_indices, out.score))
}

/// Run MCC-based AND beam search on selected row indices from JXBIN001 input.
///
/// Returns `(selected_indices_global, best_score)`.
#[pyfunction(name = "beam_search_and_binary_mcc_bin_indices")]
#[pyo3(signature = (bin_path, y, snp_indices, max_pick=3, beam_width=128))]
pub fn beam_search_and_binary_mcc_bin_indices_py(
    bin_path: String,
    y: PyReadonlyArray1<'_, u8>,
    snp_indices: Vec<usize>,
    max_pick: usize,
    beam_width: usize,
) -> PyResult<(Vec<usize>, f64)> {
    let yv = readonly_u8_to_cow(&y);
    let (bits_flat, row_words, n_rows, n_samples) =
        load_bin01_selected_rows_as_u64_words(&bin_path, &snp_indices)
            .map_err(PyRuntimeError::new_err)?;
    if yv.len() < n_samples {
        return Err(PyRuntimeError::new_err(format!(
            "beam_search_and_binary_mcc_bin_indices: y length={} is smaller than bin n_samples={}",
            yv.len(),
            n_samples
        )));
    }
    let out = beam_search_and_binary_mcc(
        yv.as_ref(),
        &bits_flat,
        row_words,
        n_rows,
        n_samples,
        max_pick,
        beam_width,
    )
    .map_err(PyRuntimeError::new_err)?;
    let selected_global = out
        .selected_indices
        .iter()
        .map(|&i| snp_indices[i])
        .collect::<Vec<_>>();
    Ok((selected_global, out.score))
}

/// Run abs-correlation-based AND beam search directly on JXBIN001 (`jx gformat` .bin) input.
///
/// Returns `(selected_indices, best_score)`, where `best_score = max(abs(corr(y, xcombine)))`.
#[pyfunction(name = "beam_search_and_continuous_corr_bin")]
#[pyo3(signature = (bin_path, y, max_pick=3, beam_width=128))]
pub fn beam_search_and_continuous_corr_bin_py(
    bin_path: String,
    y: PyReadonlyArray1<'_, f64>,
    max_pick: usize,
    beam_width: usize,
) -> PyResult<(Vec<usize>, f64)> {
    let yv = readonly_f64_to_cow(&y);
    let (bits_flat, row_words, n_rows, n_samples) =
        load_bin01_as_u64_words(&bin_path).map_err(PyRuntimeError::new_err)?;
    let out = beam_search_and_continuous_abs_corr(
        yv.as_ref(),
        &bits_flat,
        row_words,
        n_rows,
        n_samples,
        max_pick,
        beam_width,
    )
    .map_err(PyRuntimeError::new_err)?;
    Ok((out.selected_indices, out.score))
}

/// Run abs-correlation-based AND beam search on selected row indices from JXBIN001 input.
///
/// Returns `(selected_indices_global, best_score)`.
#[pyfunction(name = "beam_search_and_continuous_corr_bin_indices")]
#[pyo3(signature = (bin_path, y, snp_indices, max_pick=3, beam_width=128))]
pub fn beam_search_and_continuous_corr_bin_indices_py(
    bin_path: String,
    y: PyReadonlyArray1<'_, f64>,
    snp_indices: Vec<usize>,
    max_pick: usize,
    beam_width: usize,
) -> PyResult<(Vec<usize>, f64)> {
    let yv = readonly_f64_to_cow(&y);
    let (bits_flat, row_words, n_rows, n_samples) =
        load_bin01_selected_rows_as_u64_words(&bin_path, &snp_indices)
            .map_err(PyRuntimeError::new_err)?;
    let out = beam_search_and_continuous_abs_corr(
        yv.as_ref(),
        &bits_flat,
        row_words,
        n_rows,
        n_samples,
        max_pick,
        beam_width,
    )
    .map_err(PyRuntimeError::new_err)?;
    let selected_global = out
        .selected_indices
        .iter()
        .map(|&i| snp_indices[i])
        .collect::<Vec<_>>();
    Ok((selected_global, out.score))
}

/// Run MCC-based AND beam search over chromosome windows derived from site metadata.
///
/// This keeps window construction and scanning in Rust to avoid Python-side
/// per-site object conversion and per-window dispatch overhead.
///
/// Returns:
/// - selected indices on global BIN row space
/// - best score
/// - best window chrom/start/end and candidate count
/// - number of non-empty windows evaluated
/// - optional per-window summaries when `return_window_results=true`
#[pyfunction(name = "beam_scan_windows_binary_mcc_bin")]
#[pyo3(signature = (
    bin_path,
    y,
    max_pick=3,
    beam_width=5,
    extension=50000,
    step=None,
    return_window_results=false
))]
pub fn beam_scan_windows_binary_mcc_bin_py(
    bin_path: String,
    y: PyReadonlyArray1<'_, u8>,
    max_pick: usize,
    beam_width: usize,
    extension: usize,
    step: Option<usize>,
    return_window_results: bool,
) -> PyResult<(
    Vec<usize>,
    f64,
    String,
    i32,
    i32,
    usize,
    usize,
    Vec<(usize, String, i32, i32, usize, f64, Vec<usize>)>,
)> {
    if extension == 0 {
        return Err(PyRuntimeError::new_err(
            "beam_scan_windows_binary_mcc_bin: extension must be > 0",
        ));
    }
    let step_v = step.unwrap_or((extension / 2).max(1));
    if step_v == 0 {
        return Err(PyRuntimeError::new_err(
            "beam_scan_windows_binary_mcc_bin: step must be > 0",
        ));
    }

    let yv = readonly_u8_to_cow(&y);
    let (bits_flat, row_words, n_rows, n_samples) =
        load_bin01_as_u64_words(&bin_path).map_err(PyRuntimeError::new_err)?;
    if yv.len() < n_samples {
        return Err(PyRuntimeError::new_err(format!(
            "beam_scan_windows_binary_mcc_bin: y length={} is smaller than bin n_samples={}",
            yv.len(),
            n_samples
        )));
    }

    scan_windows_with_objective(
        &bin_path,
        &bits_flat,
        row_words,
        n_rows,
        extension,
        step_v,
        return_window_results,
        |bits, rows| {
            beam_search_and_binary_mcc(
                yv.as_ref(),
                bits,
                row_words,
                rows,
                n_samples,
                max_pick,
                beam_width,
            )
        },
    )
    .map_err(|e| PyRuntimeError::new_err(format!("beam_scan_windows_binary_mcc_bin: {e}")))
}

/// Run abs-correlation-based AND beam search over chromosome windows.
///
/// Returns:
/// - selected indices on global BIN row space
/// - best score (abs corr)
/// - best window chrom/start/end and candidate count
/// - number of non-empty windows evaluated
/// - optional per-window summaries when `return_window_results=true`
#[pyfunction(name = "beam_scan_windows_continuous_corr_bin")]
#[pyo3(signature = (
    bin_path,
    y,
    max_pick=3,
    beam_width=5,
    extension=50000,
    step=None,
    return_window_results=false
))]
pub fn beam_scan_windows_continuous_corr_bin_py(
    bin_path: String,
    y: PyReadonlyArray1<'_, f64>,
    max_pick: usize,
    beam_width: usize,
    extension: usize,
    step: Option<usize>,
    return_window_results: bool,
) -> PyResult<BeamWindowScanOutput> {
    if extension == 0 {
        return Err(PyRuntimeError::new_err(
            "beam_scan_windows_continuous_corr_bin: extension must be > 0",
        ));
    }
    let step_v = step.unwrap_or((extension / 2).max(1));
    if step_v == 0 {
        return Err(PyRuntimeError::new_err(
            "beam_scan_windows_continuous_corr_bin: step must be > 0",
        ));
    }

    let yv = readonly_f64_to_cow(&y);
    let (bits_flat, row_words, n_rows, n_samples) =
        load_bin01_as_u64_words(&bin_path).map_err(PyRuntimeError::new_err)?;
    validate_continuous_y(
        yv.as_ref(),
        n_samples,
        "beam_scan_windows_continuous_corr_bin",
    )
    .map_err(PyRuntimeError::new_err)?;

    scan_windows_with_objective(
        &bin_path,
        &bits_flat,
        row_words,
        n_rows,
        extension,
        step_v,
        return_window_results,
        |bits, rows| {
            beam_search_and_continuous_abs_corr(
                yv.as_ref(),
                bits,
                row_words,
                rows,
                n_samples,
                max_pick,
                beam_width,
            )
        },
    )
    .map_err(|e| PyRuntimeError::new_err(format!("beam_scan_windows_continuous_corr_bin: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn pack_rows_01(rows: &[Vec<u8>]) -> (Vec<u64>, usize, usize, usize) {
        let n_rows = rows.len();
        assert!(n_rows > 0);
        let n_samples = rows[0].len();
        let row_words = n_samples.div_ceil(64).max(1);

        let mut flat = vec![0u64; n_rows * row_words];
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), n_samples);
            for (i, &b) in row.iter().enumerate() {
                if b != 0 {
                    flat[r * row_words + (i >> 6)] |= 1u64 << (i & 63);
                }
            }
        }

        (flat, row_words, n_rows, n_samples)
    }

    fn brute_force_best_mcc(
        bits_flat: &[u64],
        row_words: usize,
        n_rows: usize,
        n_samples: usize,
        y_bits: &[u64],
        y_pos: u64,
        max_pick: usize,
    ) -> (f64, Vec<usize>) {
        fn rec(
            bits_flat: &[u64],
            row_words: usize,
            n_rows: usize,
            n_samples: usize,
            y_bits: &[u64],
            y_pos: u64,
            max_pick: usize,
            start: usize,
            current: Option<Vec<u64>>,
            selected: &mut Vec<usize>,
            best: &mut (f64, Vec<usize>),
        ) {
            for idx in start..n_rows {
                let row = row_prefix(bits_flat, row_words, idx, row_words);
                let mut combined = match current.as_ref() {
                    Some(c) => {
                        let mut out = c.clone();
                        bitand_assign(&mut out, row);
                        out
                    }
                    None => row.to_vec(),
                };
                apply_tail_mask(&mut combined, tail_mask(n_samples));
                selected.push(idx);

                let (x_pos, tp) = masked_xpos_tp(&combined, y_bits, n_samples);
                let score = mcc_from_xpos_tp(x_pos, tp, y_pos, n_samples);
                let best_score_key = score_key(best.0);
                let score_k = score_key(score);
                if score_k > best_score_key
                    || (score_k == best_score_key && selected.len() < best.1.len())
                {
                    best.0 = score;
                    best.1 = selected.clone();
                }

                if selected.len() < max_pick {
                    rec(
                        bits_flat,
                        row_words,
                        n_rows,
                        n_samples,
                        y_bits,
                        y_pos,
                        max_pick,
                        idx + 1,
                        Some(combined),
                        selected,
                        best,
                    );
                }

                selected.pop();
            }
        }

        let mut best = (f64::NEG_INFINITY, Vec::<usize>::new());
        let mut selected = Vec::<usize>::new();
        rec(
            bits_flat,
            row_words,
            n_rows,
            n_samples,
            y_bits,
            y_pos,
            max_pick,
            0,
            None,
            &mut selected,
            &mut best,
        );
        best
    }

    #[test]
    fn test_beam_binary_mcc_finds_known_best_pair() {
        // Best pair is row0 & row1 == y
        let y = vec![1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let rows = vec![
            vec![1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            vec![1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ];
        let (bits_flat, row_words, n_rows, n_samples) = pack_rows_01(&rows);

        let res = beam_search_and_binary_mcc(&y, &bits_flat, row_words, n_rows, n_samples, 2, 8)
            .expect("beam search should succeed");

        assert!((res.score - 1.0).abs() < 1e-12);
        assert_eq!(res.selected_indices, vec![0, 1]);

        let uniq: HashSet<usize> = res.selected_indices.iter().copied().collect();
        assert_eq!(uniq.len(), res.selected_indices.len());
    }

    #[test]
    fn test_beam_respects_max_pick() {
        // Only triple intersection reaches y exactly.
        let y = vec![1, 1, 0, 0, 0, 0, 0, 0];
        let rows = vec![
            vec![1, 1, 1, 1, 0, 0, 0, 0],
            vec![1, 1, 0, 1, 1, 0, 0, 0],
            vec![1, 1, 1, 0, 1, 0, 0, 0],
        ];
        let (bits_flat, row_words, n_rows, n_samples) = pack_rows_01(&rows);

        let res2 = beam_search_and_binary_mcc(&y, &bits_flat, row_words, n_rows, n_samples, 2, 8)
            .expect("beam search m=2 should succeed");
        assert!(res2.score < 1.0 - 1e-12);
        assert!(res2.selected_indices.len() <= 2);

        let res3 = beam_search_and_binary_mcc(&y, &bits_flat, row_words, n_rows, n_samples, 3, 8)
            .expect("beam search m=3 should succeed");
        assert!((res3.score - 1.0).abs() < 1e-12);
        assert_eq!(res3.selected_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_beam_with_custom_score_prefers_single_best_row() {
        let rows = vec![
            vec![1, 1, 1, 0, 0, 0, 0, 0], // popcount 3
            vec![1, 1, 1, 1, 1, 0, 0, 0], // popcount 5 (best single)
            vec![1, 1, 0, 0, 0, 0, 0, 0], // popcount 2
        ];
        let (bits_flat, row_words, n_rows, n_samples) = pack_rows_01(&rows);

        let res = beam_search_and_with_score(&bits_flat, row_words, n_rows, n_samples, 3, 8, |x| {
            popcount(x) as f64
        })
        .expect("beam search should succeed");

        assert_eq!(res.selected_indices, vec![1]);
        assert_eq!(res.score, 5.0);
    }

    #[test]
    fn test_beam_continuous_abs_corr_prefers_signal_row() {
        let y = vec![2.0_f64, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let rows = vec![
            vec![1, 1, 1, 0, 0, 0, 0, 0], // corr ~= 1.0
            vec![1, 0, 1, 0, 1, 0, 1, 0], // weaker
            vec![0, 1, 0, 1, 0, 1, 0, 1], // inverse weaker
        ];
        let (bits_flat, row_words, n_rows, n_samples) = pack_rows_01(&rows);

        let out =
            beam_search_and_continuous_abs_corr(&y, &bits_flat, row_words, n_rows, n_samples, 2, 8)
                .expect("continuous beam search should succeed");

        assert_eq!(out.selected_indices, vec![0]);
        assert!((out.score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_invalid_config_errors() {
        let y = vec![0u8, 1, 0, 1];
        let rows = vec![vec![0, 1, 0, 1]];
        let (bits_flat, row_words, n_rows, n_samples) = pack_rows_01(&rows);

        let e = beam_search_and_binary_mcc(&y, &bits_flat, row_words, n_rows, n_samples, 0, 4)
            .expect_err("max_pick=0 should error");
        assert!(e.contains("max_pick"));
    }

    #[test]
    fn test_load_bin01_and_search_flow() {
        let rows = vec![
            vec![1u8, 1, 1, 1, 1, 1, 1, 1],
            vec![1u8, 1, 1, 1, 0, 0, 0, 0],
            vec![1u8, 1, 0, 0, 0, 0, 0, 0],
        ];
        let y = vec![1u8, 1, 1, 1, 0, 0, 0, 0];

        let (flat, _row_words, n_rows, n_samples) = pack_rows_01(&rows);
        let row_bytes = n_samples.div_ceil(8);
        assert_eq!(row_bytes, 1);

        let mut payload = Vec::<u8>::with_capacity(n_rows * row_bytes);
        for r in 0..n_rows {
            let b = (flat[r] & 0xFF) as u8;
            payload.push(b);
        }

        let mut bytes = Vec::<u8>::new();
        bytes.extend_from_slice(BIN01_MAGIC);
        bytes.extend_from_slice(&(n_rows as u64).to_le_bytes());
        bytes.extend_from_slice(&(n_samples as u64).to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&payload);

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("janusx_beam_{stamp}.bin"));
        fs::write(&path, bytes).expect("write temp bin");

        let (bits_flat, row_words, nr, ns) =
            load_bin01_as_u64_words(path.to_str().expect("utf8 path")).expect("load bin");
        assert_eq!(nr, n_rows);
        assert_eq!(ns, n_samples);
        assert_eq!(row_words, 1);

        let out = beam_search_and_binary_mcc(&y, &bits_flat, row_words, nr, ns, 2, 8)
            .expect("search should succeed");
        assert!((out.score - 1.0).abs() < 1e-12);
        assert_eq!(out.selected_indices, vec![1]);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_bnb_and_simd_paths_match_scalar_reference() {
        let y = vec![1u8, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0];
        let rows = vec![
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            vec![1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            vec![1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
            vec![0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
            vec![1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        ];
        let (bits_flat, row_words, n_rows, n_samples) = pack_rows_01(&rows);
        let (y_bits, y_pos) = pack_binary_y_to_bits(&y, n_samples);
        let (brute_score, _brute_sel) =
            brute_force_best_mcc(&bits_flat, row_words, n_rows, n_samples, &y_bits, y_pos, 3);

        let scalar_no_bnb = beam_search_and_binary_mcc_with_options(
            &y,
            &bits_flat,
            row_words,
            n_rows,
            n_samples,
            3,
            64,
            BinaryBeamRuntimeOptions {
                use_simd_fast_path: false,
                use_upper_bound_prune: false,
            },
        )
        .expect("scalar no-bnb should succeed");

        let simd_bnb = beam_search_and_binary_mcc_with_options(
            &y,
            &bits_flat,
            row_words,
            n_rows,
            n_samples,
            3,
            64,
            BinaryBeamRuntimeOptions {
                use_simd_fast_path: true,
                use_upper_bound_prune: true,
            },
        )
        .expect("simd+bnb should succeed");

        assert!((scalar_no_bnb.score - brute_score).abs() < 1e-12);
        assert!((simd_bnb.score - brute_score).abs() < 1e-12);
    }
}
