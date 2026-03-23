use crate::gfreader::SiteInfo;
use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Clone)]
enum TreeNode {
    Leaf(String),
    Internal {
        left: usize,
        right: usize,
        left_len: f64,
        right_len: f64,
    },
}

#[derive(Copy, Clone)]
struct RapidRowEntry {
    lb: f64,
    row: usize,
    epoch: usize,
    token: u64,
}

impl PartialEq for RapidRowEntry {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row
            && self.epoch == other.epoch
            && self.token == other.token
            && self.lb.to_bits() == other.lb.to_bits()
    }
}

impl Eq for RapidRowEntry {}

impl PartialOrd for RapidRowEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RapidRowEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // reverse order to make BinaryHeap work as a min-heap by lb.
        other
            .lb
            .total_cmp(&self.lb)
            .then_with(|| self.row.cmp(&other.row))
            .then_with(|| self.epoch.cmp(&other.epoch))
            .then_with(|| self.token.cmp(&other.token))
    }
}

#[derive(Copy, Clone)]
struct RowCandEntry {
    d: f64,
    j: usize,
}

impl PartialEq for RowCandEntry {
    fn eq(&self, other: &Self) -> bool {
        self.j == other.j && self.d.to_bits() == other.d.to_bits()
    }
}

impl Eq for RowCandEntry {}

impl PartialOrd for RowCandEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RowCandEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // reverse order to make BinaryHeap work as a min-heap by distance.
        other
            .d
            .total_cmp(&self.d)
            .then_with(|| self.j.cmp(&other.j))
    }
}

#[derive(Default)]
struct RapidCoreStats {
    row_builds: u64,
    row_seed_items: u64,
    refresh_calls: u64,
    eval_calls: u64,
    stale_promotions: u64,
    strict_rechecks: u64,
    strict_updates: u64,
    strict_batch_rows: u64,
    chain_steps: u64,
    chain_mutual: u64,
    hybrid_replays: u64,
    hybrid_replay_rows: u64,
    mode_fullscan: u64,
    heap_pops: u64,
    inactive_skips: u64,
    token_skips: u64,
    lb_breaks: u64,
    popcap_breaks: u64,
    seen_skips: u64,
    fullscan_fallback: u64,
    fullscan_verify: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum BionjVarMode {
    Binom,
    Dist,
    Jc,
    Auto,
}

impl BionjVarMode {
    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            Self::Binom => "binom",
            Self::Dist => "dist",
            Self::Jc => "jc",
            Self::Auto => "auto",
        }
    }
}

#[derive(Clone)]
struct BaseBitplanes {
    a: Vec<u64>,
    c: Vec<u64>,
    g: Vec<u64>,
    t: Vec<u64>,
}

#[inline]
fn ascii_upper(b: u8) -> u8 {
    if b.is_ascii_lowercase() {
        b.to_ascii_uppercase()
    } else {
        b
    }
}

#[inline]
fn is_acgt(b: u8) -> bool {
    matches!(b, b'A' | b'C' | b'G' | b'T')
}

fn quote_newick_name(name: &str) -> String {
    let safe = name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'));
    if safe {
        return name.to_string();
    }
    let escaped = name.replace('\'', "_");
    format!("'{}'", escaped)
}

fn add_dist_node(dist: &mut Vec<Vec<f64>>) -> usize {
    let next = dist.len();
    for row in dist.iter_mut() {
        row.push(0.0);
    }
    dist.push(vec![0.0; next + 1]);
    next
}

fn render_newick(node_id: usize, nodes: &[TreeNode]) -> String {
    match &nodes[node_id] {
        TreeNode::Leaf(name) => quote_newick_name(name),
        TreeNode::Internal {
            left,
            right,
            left_len,
            right_len,
        } => {
            let ltxt = render_newick(*left, nodes);
            let rtxt = render_newick(*right, nodes);
            format!("({}:{:.10},{}:{:.10})", ltxt, left_len, rtxt, right_len)
        }
    }
}

fn resolve_threads(requested: usize) -> usize {
    if requested > 0 {
        return requested;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[inline]
fn env_usize_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

#[inline]
fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
}

fn env_bool_or(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "y" | "on")
        }
        Err(_) => default,
    }
}

#[inline]
fn env_f64_or(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(default)
}

#[inline]
fn env_u64_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn resolve_bionj_var_mode() -> Result<BionjVarMode, String> {
    let raw = std::env::var("JANUSX_BIONJ_VAR_MODE")
        .ok()
        .unwrap_or_else(|| "dist".to_string());
    let m = raw.trim().to_ascii_lowercase();
    let out = match m.as_str() {
        "" | "jc" | "jc69" => BionjVarMode::Jc,
        "dist" | "d" | "var-d" | "vard" => BionjVarMode::Dist,
        "binom" | "p" | "mismatch" => BionjVarMode::Binom,
        "auto" => BionjVarMode::Auto,
        _ => {
            return Err(format!(
                "invalid JANUSX_BIONJ_VAR_MODE='{}' (expected one of: jc, dist, binom, auto)",
                raw
            ))
        }
    };
    Ok(out)
}

#[inline]
fn sanitize_base_pair(mut ref_base: u8, mut alt_base: u8) -> (u8, u8) {
    ref_base = ascii_upper(ref_base);
    alt_base = ascii_upper(alt_base);
    if !is_acgt(ref_base) {
        ref_base = b'N';
    }
    if !is_acgt(alt_base) {
        alt_base = b'N';
    }
    if ref_base == b'N' && alt_base == b'N' {
        (b'A', b'G')
    } else if ref_base == b'N' {
        if alt_base != b'A' {
            (b'A', alt_base)
        } else {
            (b'C', alt_base)
        }
    } else if alt_base == b'N' {
        if ref_base != b'G' {
            (ref_base, b'G')
        } else {
            (ref_base, b'T')
        }
    } else {
        (ref_base, alt_base)
    }
}

#[inline]
fn first_base_code(s: &str) -> u8 {
    for b in s.as_bytes() {
        if b.is_ascii_whitespace() {
            continue;
        }
        let u = ascii_upper(*b);
        if is_acgt(u) {
            return u;
        }
        return b'N';
    }
    b'N'
}

#[inline]
fn het_iupac(ref_base: u8, alt_base: u8) -> u8 {
    if ref_base == alt_base {
        return ref_base;
    }
    match (ref_base, alt_base) {
        (b'A', b'C') | (b'C', b'A') => b'M',
        (b'A', b'G') | (b'G', b'A') => b'R',
        (b'A', b'T') | (b'T', b'A') => b'W',
        (b'C', b'G') | (b'G', b'C') => b'S',
        (b'C', b'T') | (b'T', b'C') => b'Y',
        (b'G', b'T') | (b'T', b'G') => b'K',
        _ => b'N',
    }
}

fn build_base_bitplanes(
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    n_taxa: usize,
    n_sites: usize,
    pool: Option<&ThreadPool>,
) -> Vec<BaseBitplanes> {
    let n_words = (n_sites + 63) / 64;
    let build_one = |i: usize| -> BaseBitplanes {
        let mut va = vec![0u64; n_words];
        let mut vc = vec![0u64; n_words];
        let mut vg = vec![0u64; n_words];
        let mut vt = vec![0u64; n_words];
        for k in 0..n_sites {
            let w = k >> 6;
            let bit = 1u64 << (k & 63);
            match ascii_upper(a[[i, k]]) {
                b'A' => va[w] |= bit,
                b'C' => vc[w] |= bit,
                b'G' => vg[w] |= bit,
                b'T' => vt[w] |= bit,
                // Ambiguous bases / gaps / missing are all treated as unknown here.
                _ => {}
            }
        }
        BaseBitplanes {
            a: va,
            c: vc,
            g: vg,
            t: vt,
        }
    };

    if let Some(p) = pool {
        p.install(|| (0..n_taxa).into_par_iter().map(build_one).collect())
    } else {
        (0..n_taxa).map(build_one).collect()
    }
}

#[inline]
fn mismatch_to_jc69_distance(p_mismatch: f64) -> f64 {
    let p = p_mismatch.clamp(0.0, 0.75 - 1e-12);
    let denom = (1.0 - (4.0 * p / 3.0)).max(1e-12);
    let d = -0.75 * denom.ln();
    if d.is_finite() { d.max(0.0) } else { 1.0 }
}

#[inline]
fn pair_mismatch_stats_bitset(
    bits: &[BaseBitplanes],
    i: usize,
    j: usize,
    min_ov: u64,
) -> Option<(f64, u64)> {
    let bi = &bits[i];
    let bj = &bits[j];
    let n_words = bi.a.len();
    let mut valid: u64 = 0;
    let mut same: u64 = 0;
    for w in 0..n_words {
        let vi = bi.a[w] | bi.c[w] | bi.g[w] | bi.t[w];
        let vj = bj.a[w] | bj.c[w] | bj.g[w] | bj.t[w];
        valid += ((vi & vj).count_ones()) as u64;
        same += ((bi.a[w] & bj.a[w]).count_ones()) as u64;
        same += ((bi.c[w] & bj.c[w]).count_ones()) as u64;
        same += ((bi.g[w] & bj.g[w]).count_ones()) as u64;
        same += ((bi.t[w] & bj.t[w]).count_ones()) as u64;
    }
    if valid >= min_ov && valid > 0 {
        let p = ((valid - same) as f64) / (valid as f64);
        Some((p.clamp(0.0, 1.0), valid))
    } else {
        None
    }
}

#[inline]
fn bionj_variance_from_stats(
    p_mismatch: f64,
    valid_sites: u64,
    dist_jc: f64,
    var_mode: BionjVarMode,
) -> f64 {
    if valid_sites == 0 {
        return 1.0;
    }
    let p = p_mismatch.clamp(0.0, 1.0);
    let l = valid_sites as f64;
    let var_p = (p * (1.0 - p) / l).max(1e-12);
    match var_mode {
        BionjVarMode::Binom => var_p,
        BionjVarMode::Dist => dist_jc.max(1e-12),
        BionjVarMode::Jc | BionjVarMode::Auto => {
            let p_clip = p.min(0.75 - 1e-12);
            let denom = (1.0 - (4.0 * p_clip / 3.0)).max(1e-12);
            (var_p / (denom * denom)).max(1e-12)
        }
    }
}

#[inline]
fn pair_distance_and_variance_bitset(
    bits: &[BaseBitplanes],
    i: usize,
    j: usize,
    min_ov: u64,
    var_mode: BionjVarMode,
) -> (f64, f64) {
    if let Some((p_mismatch, valid_sites)) = pair_mismatch_stats_bitset(bits, i, j, min_ov) {
        let d_jc = mismatch_to_jc69_distance(p_mismatch);
        let v = bionj_variance_from_stats(p_mismatch, valid_sites, d_jc, var_mode);
        (d_jc, v)
    } else {
        (1.0, 1.0)
    }
}

#[inline]
fn pair_distance_bitset(bits: &[BaseBitplanes], i: usize, j: usize, min_ov: u64) -> f64 {
    if let Some((p_mismatch, _valid_sites)) = pair_mismatch_stats_bitset(bits, i, j, min_ov) {
        mismatch_to_jc69_distance(p_mismatch)
    } else {
        1.0
    }
}

#[inline]
fn pair_mismatch_rate_bitset(bits: &[BaseBitplanes], i: usize, j: usize, min_ov: u64) -> f64 {
    if let Some((p_mismatch, _valid_sites)) = pair_mismatch_stats_bitset(bits, i, j, min_ov) {
        p_mismatch
    } else {
        1.0
    }
}

fn build_distance_matrix(
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    n_taxa: usize,
    min_ov: u64,
    pool: Option<&ThreadPool>,
) -> Vec<Vec<f64>> {
    let n_sites = a.shape()[1];
    let bits = build_base_bitplanes(a, n_taxa, n_sites, pool);
    let mut dist: Vec<Vec<f64>> = if let Some(p) = pool {
        p.install(|| {
            (0..n_taxa)
                .into_par_iter()
                .map(|i| {
                    let mut row = vec![0.0; n_taxa];
                    for j in (i + 1)..n_taxa {
                        row[j] = pair_distance_bitset(&bits, i, j, min_ov);
                    }
                    row
                })
                .collect()
        })
    } else {
        let mut rows = vec![vec![0.0; n_taxa]; n_taxa];
        for i in 0..n_taxa {
            for j in (i + 1)..n_taxa {
                rows[i][j] = pair_distance_bitset(&bits, i, j, min_ov);
            }
        }
        rows
    };

    for i in 0..n_taxa {
        for j in (i + 1)..n_taxa {
            let v = dist[i][j];
            dist[j][i] = v;
        }
    }
    dist
}

fn build_distance_matrix_mismatch(
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    n_taxa: usize,
    min_ov: u64,
    pool: Option<&ThreadPool>,
) -> Vec<Vec<f64>> {
    let n_sites = a.shape()[1];
    let bits = build_base_bitplanes(a, n_taxa, n_sites, pool);
    let mut dist: Vec<Vec<f64>> = if let Some(p) = pool {
        p.install(|| {
            (0..n_taxa)
                .into_par_iter()
                .map(|i| {
                    let mut row = vec![0.0; n_taxa];
                    for j in (i + 1)..n_taxa {
                        row[j] = pair_mismatch_rate_bitset(&bits, i, j, min_ov);
                    }
                    row
                })
                .collect()
        })
    } else {
        let mut rows = vec![vec![0.0; n_taxa]; n_taxa];
        for i in 0..n_taxa {
            for j in (i + 1)..n_taxa {
                rows[i][j] = pair_mismatch_rate_bitset(&bits, i, j, min_ov);
            }
        }
        rows
    };

    for i in 0..n_taxa {
        for j in (i + 1)..n_taxa {
            let v = dist[i][j];
            dist[j][i] = v;
        }
    }
    dist
}

fn build_distance_and_variance_matrix(
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    n_taxa: usize,
    min_ov: u64,
    pool: Option<&ThreadPool>,
    var_mode: BionjVarMode,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n_sites = a.shape()[1];
    let bits = build_base_bitplanes(a, n_taxa, n_sites, pool);
    let mut dist = vec![vec![0.0; n_taxa]; n_taxa];
    let mut var = vec![vec![0.0; n_taxa]; n_taxa];

    if let Some(p) = pool {
        let rows: Vec<(usize, Vec<f64>, Vec<f64>)> = p.install(|| {
            (0..n_taxa)
                .into_par_iter()
                .map(|i| {
                    let mut drow = vec![0.0; n_taxa];
                    let mut vrow = vec![0.0; n_taxa];
                    for j in (i + 1)..n_taxa {
                        let (d, v) =
                            pair_distance_and_variance_bitset(&bits, i, j, min_ov, var_mode);
                        drow[j] = d;
                        vrow[j] = v;
                    }
                    (i, drow, vrow)
                })
                .collect()
        });
        for (i, drow, vrow) in rows {
            for j in (i + 1)..n_taxa {
                dist[i][j] = drow[j];
                var[i][j] = vrow[j];
            }
        }
    } else {
        for i in 0..n_taxa {
            for j in (i + 1)..n_taxa {
                let (d, v) = pair_distance_and_variance_bitset(&bits, i, j, min_ov, var_mode);
                dist[i][j] = d;
                var[i][j] = v;
            }
        }
    }

    for i in 0..n_taxa {
        for j in (i + 1)..n_taxa {
            let d = dist[i][j];
            let v = var[i][j];
            dist[j][i] = d;
            var[j][i] = v;
        }
    }
    (dist, var)
}

#[derive(Clone)]
struct LowerTriDist {
    cap: usize,
    data: Vec<f64>,
}

impl LowerTriDist {
    fn with_capacity(cap: usize) -> Result<Self, String> {
        let nn = cap
            .checked_mul(cap.saturating_sub(1))
            .ok_or_else(|| "distance matrix size overflow".to_string())?;
        let len = nn / 2;
        Ok(Self {
            cap,
            data: vec![0.0; len],
        })
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.cap
    }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.cap && j < self.cap && i != j);
        let (a, b) = if i > j { (i, j) } else { (j, i) };
        a * (a - 1) / 2 + b
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        if i == j {
            0.0
        } else {
            self.data[self.idx(i, j)]
        }
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, v: f64) {
        if i == j {
            return;
        }
        let idx = self.idx(i, j);
        self.data[idx] = v;
    }
}

fn build_distance_lowertri(
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    n_taxa: usize,
    min_ov: u64,
    pool: Option<&ThreadPool>,
) -> Result<LowerTriDist, String> {
    build_distance_lowertri_with_capacity(a, n_taxa, min_ov, n_taxa, pool)
}

fn build_distance_lowertri_with_capacity(
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    n_taxa: usize,
    min_ov: u64,
    cap_nodes: usize,
    pool: Option<&ThreadPool>,
) -> Result<LowerTriDist, String> {
    if cap_nodes < n_taxa {
        return Err(format!(
            "invalid lower-tri capacity: cap_nodes={} < n_taxa={}",
            cap_nodes, n_taxa
        ));
    }
    let n_sites = a.shape()[1];
    let bits = build_base_bitplanes(a, n_taxa, n_sites, pool);
    let mut dist = LowerTriDist::with_capacity(cap_nodes)?;
    if let Some(p) = pool {
        let rows: Vec<(usize, Vec<f64>)> = p.install(|| {
            (1..n_taxa)
                .into_par_iter()
                .map(|i| {
                    let mut row = vec![0.0; i];
                    for j in 0..i {
                        row[j] = pair_distance_bitset(&bits, i, j, min_ov);
                    }
                    (i, row)
                })
                .collect()
        });
        for (i, row) in rows {
            let start = i * (i - 1) / 2;
            dist.data[start..(start + i)].copy_from_slice(&row);
        }
    } else {
        for i in 1..n_taxa {
            for j in 0..i {
                dist.set(i, j, pair_distance_bitset(&bits, i, j, min_ov));
            }
        }
    }
    Ok(dist)
}

#[inline]
fn lt_q_criterion(m: f64, d: f64, r_i: f64, r_j: f64) -> f64 {
    (m - 2.0) * d - r_i - r_j
}

fn compute_r_sums_lt(active: &[usize], dist: &LowerTriDist) -> Vec<f64> {
    let mut r = vec![0.0; dist.capacity()];
    for ai in 0..active.len() {
        let i = active[ai];
        for &j in active.iter().skip(ai + 1) {
            let d = dist.get(i, j);
            r[i] += d;
            r[j] += d;
        }
    }
    r
}

fn lt_set_best_hit(
    i: usize,
    active: &[usize],
    dist: &LowerTriDist,
    r: &[f64],
    m: f64,
) -> (usize, f64) {
    let mut best_j = usize::MAX;
    let mut best_q = f64::INFINITY;
    for &j in active {
        if j == i {
            continue;
        }
        let q = lt_q_criterion(m, dist.get(i, j), r[i], r[j]);
        if q < best_q || (q == best_q && j < best_j) {
            best_q = q;
            best_j = j;
        }
    }
    (best_j, best_q)
}

fn lt_strict_recheck_pair(
    active: &[usize],
    dist: &LowerTriDist,
    r: &[f64],
    m: f64,
    best_i: usize,
    best_j: usize,
    best_q: f64,
    max_steps: usize,
) -> (usize, usize, f64) {
    if max_steps == 0 || !best_q.is_finite() {
        return (best_i, best_j, best_q);
    }

    let (mut out_i, mut out_j) = if best_i <= best_j {
        (best_i, best_j)
    } else {
        (best_j, best_i)
    };
    let mut out_q = best_q;

    let mut frontier: Vec<usize> = Vec::with_capacity(max_steps.saturating_add(2));
    frontier.push(best_i);
    if best_j != best_i {
        frontier.push(best_j);
    }
    let mut idx = 0usize;
    let mut steps = 0usize;
    while idx < frontier.len() && steps < max_steps {
        let row = frontier[idx];
        idx += 1;
        steps += 1;
        let (j, q) = lt_set_best_hit(row, active, dist, r, m);
        if j == usize::MAX || !q.is_finite() {
            continue;
        }
        let (a, b) = if row <= j { (row, j) } else { (j, row) };
        if q < out_q || (q == out_q && (a < out_i || (a == out_i && b < out_j))) {
            out_i = a;
            out_j = b;
            out_q = q;
        }
        if !frontier.contains(&j) {
            frontier.push(j);
        }
    }
    (out_i, out_j, out_q)
}

fn lt_chain_recheck_pair(
    active: &[usize],
    active_mask: &[bool],
    dist: &LowerTriDist,
    r: &[f64],
    m: f64,
    start: usize,
    max_steps: usize,
) -> (usize, usize, f64, usize, bool) {
    if max_steps == 0 || start >= active_mask.len() || !active_mask[start] || active.len() < 2 {
        return (start, start, f64::INFINITY, 0, false);
    }

    let mut prev = usize::MAX;
    let mut cur = start;
    let mut best = (start, start, f64::INFINITY);
    let mut steps = 0usize;
    let mut mutual = false;

    for _ in 0..max_steps {
        let (nxt, q) = lt_set_best_hit(cur, active, dist, r, m);
        steps += 1;
        if nxt == usize::MAX || nxt == cur || nxt >= active_mask.len() || !active_mask[nxt] {
            break;
        }
        let (a, b) = if cur <= nxt { (cur, nxt) } else { (nxt, cur) };
        if q < best.2 || (q == best.2 && (a < best.0 || (a == best.0 && b < best.1))) {
            best = (a, b, q);
        }
        if nxt == prev {
            mutual = true;
            break;
        }
        prev = cur;
        cur = nxt;
    }
    (best.0, best.1, best.2, steps, mutual)
}

fn lt_best_pair_from_seed_rows(
    seed_rows: &[usize],
    active: &[usize],
    active_mask: &[bool],
    dist: &LowerTriDist,
    r: &[f64],
    m: f64,
    pool: Option<&ThreadPool>,
) -> (usize, usize, f64) {
    if seed_rows.is_empty() {
        return (active[0], active[0], f64::INFINITY);
    }

    if let Some(p) = pool {
        p.install(|| {
            seed_rows
                .par_iter()
                .map(|&i| {
                    if i >= active_mask.len() || !active_mask[i] {
                        return (i, i, f64::INFINITY);
                    }
                    let (j, q) = lt_set_best_hit(i, active, dist, r, m);
                    if j == usize::MAX || !q.is_finite() {
                        return (i, i, f64::INFINITY);
                    }
                    let (a, b) = if i <= j { (i, j) } else { (j, i) };
                    (a, b, q)
                })
                .reduce(
                    || (active[0], active[0], f64::INFINITY),
                    |x, y| {
                        if y.2 < x.2 || (y.2 == x.2 && (y.0 < x.0 || (y.0 == x.0 && y.1 < x.1))) {
                            y
                        } else {
                            x
                        }
                    },
                )
        })
    } else {
        let mut best = (active[0], active[0], f64::INFINITY);
        for &i in seed_rows {
            if i >= active_mask.len() || !active_mask[i] {
                continue;
            }
            let (j, q) = lt_set_best_hit(i, active, dist, r, m);
            if j == usize::MAX || !q.is_finite() {
                continue;
            }
            let (a, b) = if i <= j { (i, j) } else { (j, i) };
            if q < best.2 || (q == best.2 && (a < best.0 || (a == best.0 && b < best.1))) {
                best = (a, b, q);
            }
        }
        best
    }
}

fn row_top_k_from_active_lt(
    dist: &LowerTriDist,
    i: usize,
    active: &[usize],
    k: usize,
) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }
    let mut pairs: Vec<(f64, usize)> =
        Vec::with_capacity(active.len().min(k.saturating_mul(2) + 8));
    for &j in active {
        if j == i {
            continue;
        }
        let d = dist.get(i, j);
        if d.is_finite() {
            pairs.push((d, j));
        }
    }
    if pairs.len() > k {
        let kth = k - 1;
        pairs.select_nth_unstable_by(kth, |a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        pairs.truncate(k);
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    pairs.into_iter().map(|(_, j)| j).collect()
}

fn lt_best_pair_full_scan(
    active: &[usize],
    dist: &LowerTriDist,
    r: &[f64],
    m: f64,
) -> (usize, usize, f64) {
    let mut best = (active[0], active[1], f64::INFINITY);
    for ai in 0..(active.len() - 1) {
        let i = active[ai];
        for &j in active.iter().skip(ai + 1) {
            let q = lt_q_criterion(m, dist.get(i, j), r[i], r[j]);
            if q < best.2 || (q == best.2 && (i < best.0 || (i == best.0 && j < best.1))) {
                best = (i, j, q);
            }
        }
    }
    best
}

fn nj_newick_lowertri_lazyq(
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    sample_ids: Vec<String>,
    min_ov: u64,
    pool: Option<&ThreadPool>,
) -> Result<String, String> {
    let n_taxa = sample_ids.len();
    let mut dist = build_distance_lowertri(a, n_taxa, min_ov, pool)?;
    let mut nodes: Vec<TreeNode> = sample_ids.into_iter().map(TreeNode::Leaf).collect();

    // Fixed slots [0..n_taxa) are reused after joins, so distance storage stays O(N^2).
    let mut slot_node: Vec<usize> = (0..n_taxa).collect();
    let mut active: Vec<usize> = (0..n_taxa).collect();
    let mut active_pos: Vec<usize> = (0..n_taxa).collect();
    let mut active_mask: Vec<bool> = vec![true; n_taxa];

    let mut r = compute_r_sums_lt(&active, &dist);
    let mut visible_j: Vec<usize> = vec![usize::MAX; n_taxa];
    let mut visible_q: Vec<f64> = vec![f64::INFINITY; n_taxa];
    let refine_every = env_usize_or("JANUSX_LT_LAZYQ_REFINE_EVERY", 0);
    let hill_steps = env_usize_or("JANUSX_LT_LAZYQ_HILL_STEPS", 64).max(1);
    let mut iter_idx = 0usize;

    {
        let m0 = n_taxa as f64;
        if let Some(p) = pool {
            let rows: Vec<(usize, usize, f64)> = p.install(|| {
                active
                    .par_iter()
                    .map(|&i| {
                        let (j, q) = lt_set_best_hit(i, &active, &dist, &r, m0);
                        (i, j, q)
                    })
                    .collect()
            });
            for (i, j, q) in rows {
                visible_j[i] = j;
                visible_q[i] = q;
            }
        } else {
            for &i in &active {
                let (j, q) = lt_set_best_hit(i, &active, &dist, &r, m0);
                visible_j[i] = j;
                visible_q[i] = q;
            }
        }
    }

    while active.len() > 2 {
        let m = active.len() as f64;

        let mut best = (active[0], active[1], f64::INFINITY);
        for &i in &active {
            let j = visible_j[i];
            if j >= n_taxa || !active_mask[j] || j == i {
                let (bj, bq) = lt_set_best_hit(i, &active, &dist, &r, m);
                visible_j[i] = bj;
                visible_q[i] = bq;
            } else {
                visible_q[i] = lt_q_criterion(m, dist.get(i, j), r[i], r[j]);
            }
            let q = visible_q[i];
            let j2 = visible_j[i];
            if j2 < n_taxa && active_mask[j2] && j2 != i {
                let (p0, p1) = if i < j2 { (i, j2) } else { (j2, i) };
                if q < best.2 || (q == best.2 && (p0 < best.0 || (p0 == best.0 && p1 < best.1))) {
                    best = (p0, p1, q);
                }
            }
        }

        // Relaxed-NJ hill-climb on visible candidates (FastTree-style).
        let mut i = best.0;
        let mut j = best.1;
        let mut climbed = false;
        for _ in 0..hill_steps {
            let (bi, bqi) = lt_set_best_hit(i, &active, &dist, &r, m);
            visible_j[i] = bi;
            visible_q[i] = bqi;
            if bi != j {
                j = bi;
                climbed = true;
                continue;
            }

            let (bj, bqj) = lt_set_best_hit(j, &active, &dist, &r, m);
            visible_j[j] = bj;
            visible_q[j] = bqj;
            if bj != i {
                i = bj;
                climbed = true;
                continue;
            }
            break;
        }
        if i == usize::MAX || j == usize::MAX || i == j || !active_mask[i] || !active_mask[j] {
            let ex = lt_best_pair_full_scan(&active, &dist, &r, m);
            i = ex.0;
            j = ex.1;
        } else if i > j {
            std::mem::swap(&mut i, &mut j);
        }
        if refine_every > 0 && (iter_idx % refine_every == 0) {
            let ex = lt_best_pair_full_scan(&active, &dist, &r, m);
            i = ex.0;
            j = ex.1;
        }

        let dij = dist.get(i, j);
        let denom = m - 2.0;
        if denom <= 0.0 {
            return Err("failed to build NJ tree: invalid denominator".to_string());
        }
        let delta = (r[i] - r[j]) / denom;
        let mut li = 0.5 * (dij + delta);
        let mut lj = dij - li;
        if !li.is_finite() {
            li = 0.0;
        }
        if !lj.is_finite() {
            lj = 0.0;
        }
        li = li.max(0.0);
        lj = lj.max(0.0);

        let left = slot_node[i];
        let right = slot_node[j];
        nodes.push(TreeNode::Internal {
            left,
            right,
            left_len: li,
            right_len: lj,
        });
        let new_node = nodes.len() - 1;
        slot_node[i] = new_node;
        slot_node[j] = usize::MAX;

        let mut r_i_new = 0.0f64;
        for &k in &active {
            if k == i || k == j {
                continue;
            }
            let dik = dist.get(k, i);
            let djk = dist.get(k, j);
            let duk = 0.5 * (dik + djk - dij);
            let v = if duk.is_finite() { duk.max(0.0) } else { 0.0 };
            dist.set(k, i, v);
            r[k] = r[k] - dik - djk + v;
            r_i_new += v;
        }
        r[i] = r_i_new;
        r[j] = 0.0;

        active_mask[j] = false;
        let pos_j = active_pos[j];
        let last = *active.last().unwrap();
        active.swap_remove(pos_j);
        if pos_j < active.len() {
            active_pos[last] = pos_j;
        }
        active_pos[j] = usize::MAX;

        if active.len() >= 2 {
            let m_new = active.len() as f64;
            let (bj, bq) = lt_set_best_hit(i, &active, &dist, &r, m_new);
            visible_j[i] = bj;
            visible_q[i] = bq;
            for &k in &active {
                if k == i {
                    continue;
                }
                let old = visible_j[k];
                if old >= n_taxa || !active_mask[old] || old == k {
                    let (bk, bkq) = lt_set_best_hit(k, &active, &dist, &r, m_new);
                    visible_j[k] = bk;
                    visible_q[k] = bkq;
                    continue;
                }
                let q_old = lt_q_criterion(m_new, dist.get(k, old), r[k], r[old]);
                let q_new = lt_q_criterion(m_new, dist.get(k, i), r[k], r[i]);
                if q_new < q_old || (q_new == q_old && i < old) {
                    visible_j[k] = i;
                    visible_q[k] = q_new;
                } else {
                    visible_q[k] = q_old;
                }
            }
        }
        iter_idx += 1;

        if climbed && active.len() <= 4 {
            // no-op marker to keep compiler from optimizing away climbed in release profiles
            std::hint::black_box(climbed);
        }
    }

    let root_id = if active.len() == 2 {
        let a0 = active[0];
        let a1 = active[1];
        let d = if dist.get(a0, a1).is_finite() {
            dist.get(a0, a1).max(0.0)
        } else {
            0.0
        };
        let l = 0.5 * d;
        nodes.push(TreeNode::Internal {
            left: slot_node[a0],
            right: slot_node[a1],
            left_len: l,
            right_len: l,
        });
        nodes.len() - 1
    } else {
        slot_node[active[0]]
    };
    let mut out = render_newick(root_id, &nodes);
    out.push(';');
    Ok(out)
}

#[derive(Clone)]
struct MlNode {
    parent: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
    blen_to_parent: f64,
    name: Option<String>,
    leaf_ix: Option<usize>,
}

#[derive(Clone)]
struct MlTree {
    nodes: Vec<MlNode>,
    root: usize,
}

impl MlTree {
    #[inline]
    fn is_internal(&self, u: usize) -> bool {
        self.nodes[u].left.is_some() && self.nodes[u].right.is_some()
    }

    #[inline]
    fn children(&self, u: usize) -> Option<(usize, usize)> {
        match (self.nodes[u].left, self.nodes[u].right) {
            (Some(l), Some(r)) => Some((l, r)),
            _ => None,
        }
    }
}

#[derive(Copy, Clone)]
struct NniEdge {
    p: usize,
    c: usize,
    s: usize,
    c1: usize,
    c2: usize,
}

#[derive(Copy, Clone)]
struct NniMove {
    p: usize,
    c: usize,
    s: usize,
    swap_child: usize,
    quick_score: f64,
}

#[inline]
fn clamp_blen(x: f64) -> f64 {
    if x.is_finite() {
        x.max(1e-8)
    } else {
        1e-8
    }
}

fn build_nj_tree_exact(
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    sample_ids: &[String],
    min_ov: u64,
    use_jc_dist: bool,
    pool: Option<&ThreadPool>,
) -> Result<(Vec<TreeNode>, usize), String> {
    let n_taxa = sample_ids.len();
    if n_taxa < 2 {
        return Err("need at least 2 samples for NJ initializer".to_string());
    }
    let mut dist = if use_jc_dist {
        build_distance_matrix(a, n_taxa, min_ov, pool)
    } else {
        // Legacy initializer: mismatch-rate p-distance.
        build_distance_matrix_mismatch(a, n_taxa, min_ov, pool)
    };
    let mut nodes: Vec<TreeNode> = sample_ids.iter().cloned().map(TreeNode::Leaf).collect();
    let mut active: Vec<usize> = (0..n_taxa).collect();

    while active.len() > 2 {
        let m = active.len() as f64;
        let mut r = vec![0.0f64; dist.len()];
        for ai in 0..active.len() {
            let i = active[ai];
            for &j in active.iter().skip(ai + 1) {
                let d = dist[i][j];
                r[i] += d;
                r[j] += d;
            }
        }

        let mut best_q = f64::INFINITY;
        let mut best_i = active[0];
        let mut best_j = active[1];
        for ai in 0..active.len() {
            let i = active[ai];
            for &j in active.iter().skip(ai + 1) {
                let q = (m - 2.0) * dist[i][j] - r[i] - r[j];
                if q < best_q || (q == best_q && (i < best_i || (i == best_i && j < best_j))) {
                    best_q = q;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        let dij = dist[best_i][best_j];
        let denom = (m - 2.0).max(1.0);
        let mut li = 0.5 * dij + (r[best_i] - r[best_j]) / (2.0 * denom);
        let mut lj = dij - li;
        if !li.is_finite() || li < 0.0 {
            li = 0.0;
        }
        if !lj.is_finite() || lj < 0.0 {
            lj = 0.0;
        }

        let u = nodes.len();
        nodes.push(TreeNode::Internal {
            left: best_i,
            right: best_j,
            left_len: li,
            right_len: lj,
        });
        let ud = add_dist_node(&mut dist);
        if ud != u {
            return Err("NJ initializer internal index mismatch".to_string());
        }

        for &k in &active {
            if k == best_i || k == best_j {
                continue;
            }
            let dik = dist[best_i][k];
            let djk = dist[best_j][k];
            let mut duk = 0.5 * (dik + djk - dij);
            if !duk.is_finite() || duk < 0.0 {
                duk = 0.0;
            }
            dist[u][k] = duk;
            dist[k][u] = duk;
        }

        active.retain(|&x| x != best_i && x != best_j);
        active.push(u);
    }

    if active.len() != 2 {
        return Err("NJ initializer failed to end with exactly 2 active nodes".to_string());
    }
    let a0 = active[0];
    let a1 = active[1];
    let mut d = dist[a0][a1];
    if !d.is_finite() || d < 0.0 {
        d = 0.0;
    }
    let l = 0.5 * d;
    let root = nodes.len();
    nodes.push(TreeNode::Internal {
        left: a0,
        right: a1,
        left_len: l,
        right_len: l,
    });
    Ok((nodes, root))
}

fn convert_to_ml_tree(nodes: &[TreeNode], root: usize, n_leaves: usize) -> Result<MlTree, String> {
    if root >= nodes.len() {
        return Err("invalid root index for ML tree conversion".to_string());
    }
    let mut out = vec![
        MlNode {
            parent: None,
            left: None,
            right: None,
            blen_to_parent: 0.0,
            name: None,
            leaf_ix: None,
        };
        nodes.len()
    ];
    let mut seen = vec![false; nodes.len()];
    let mut stack: Vec<(usize, Option<usize>, f64)> = vec![(root, None, 0.0)];
    while let Some((u, parent, blen)) = stack.pop() {
        if u >= nodes.len() {
            return Err("tree conversion encountered invalid node id".to_string());
        }
        if seen[u] {
            return Err("tree conversion detected repeated node visit".to_string());
        }
        seen[u] = true;
        out[u].parent = parent;
        out[u].blen_to_parent = if parent.is_some() {
            clamp_blen(blen)
        } else {
            0.0
        };
        match &nodes[u] {
            TreeNode::Leaf(name) => {
                out[u].name = Some(name.clone());
                out[u].leaf_ix = if u < n_leaves { Some(u) } else { None };
            }
            TreeNode::Internal {
                left,
                right,
                left_len,
                right_len,
            } => {
                out[u].left = Some(*left);
                out[u].right = Some(*right);
                stack.push((*left, Some(u), *left_len));
                stack.push((*right, Some(u), *right_len));
            }
        }
    }
    Ok(MlTree { nodes: out, root })
}

#[inline]
fn jc_leaf_partials(b: u8) -> [f64; 4] {
    match ascii_upper(b) {
        b'A' => [1.0, 0.0, 0.0, 0.0],
        b'C' => [0.0, 1.0, 0.0, 0.0],
        b'G' => [0.0, 0.0, 1.0, 0.0],
        b'T' | b'U' => [0.0, 0.0, 0.0, 1.0],
        b'R' => [1.0, 0.0, 1.0, 0.0],
        b'Y' => [0.0, 1.0, 0.0, 1.0],
        b'S' => [0.0, 1.0, 1.0, 0.0],
        b'W' => [1.0, 0.0, 0.0, 1.0],
        b'K' => [0.0, 0.0, 1.0, 1.0],
        b'M' => [1.0, 1.0, 0.0, 0.0],
        b'B' => [0.0, 1.0, 1.0, 1.0],
        b'D' => [1.0, 0.0, 1.0, 1.0],
        b'H' => [1.0, 1.0, 0.0, 1.0],
        b'V' => [1.0, 1.0, 1.0, 0.0],
        _ => [1.0, 1.0, 1.0, 1.0],
    }
}

#[inline]
fn is_unambiguous_dna_base_u8(b: u8) -> bool {
    matches!(ascii_upper(b), b'A' | b'C' | b'G' | b'T' | b'U')
}

fn sanitize_alignment_compat_strict(aln: &numpy::ndarray::ArrayView2<'_, u8>) -> Array2<u8> {
    let mut out = aln.to_owned();
    for v in out.iter_mut() {
        if !is_unambiguous_dna_base_u8(*v) {
            *v = b'N';
        }
    }
    out
}

#[inline]
fn dna_state4_index(b: u8) -> Option<usize> {
    match ascii_upper(b) {
        b'A' => Some(0),
        b'C' => Some(1),
        b'G' => Some(2),
        b'T' | b'U' => Some(3),
        _ => None,
    }
}

fn build_catlite_site_rates(a: &numpy::ndarray::ArrayView2<'_, u8>, n_cat: usize) -> Vec<f64> {
    let n_taxa = a.shape()[0];
    let n_sites = a.shape()[1];
    if n_sites == 0 || n_cat <= 1 {
        return vec![1.0; n_sites];
    }
    let n_cat_eff = n_cat.min(n_sites).max(1);
    let min_rate = env_f64_or("JANUSX_ML_COMPAT_CAT_MIN_RATE", 0.05)
        .max(1e-6)
        .min(10.0);
    let max_rate = env_f64_or("JANUSX_ML_COMPAT_CAT_MAX_RATE", 20.0)
        .max(min_rate + 1e-6)
        .min(1e6);
    let mut raw = vec![1.0f64; n_sites];
    for k in 0..n_sites {
        let mut cnt = [0usize; 4];
        let mut n_obs = 0usize;
        for i in 0..n_taxa {
            if let Some(s) = dna_state4_index(a[[i, k]]) {
                cnt[s] += 1;
                n_obs += 1;
            }
        }
        if n_obs <= 1 {
            raw[k] = min_rate;
            continue;
        }
        let major = *cnt.iter().max().unwrap_or(&0usize);
        let p = (1.0 - (major as f64) / (n_obs as f64)).clamp(0.0, 0.749_999_999_999);
        let r = if p <= 0.0 {
            min_rate
        } else {
            let inner = 1.0 - (4.0 / 3.0) * p;
            if inner <= 1e-15 {
                max_rate
            } else {
                (-0.75 * inner.ln()).clamp(min_rate, max_rate)
            }
        };
        raw[k] = r;
    }

    let mut order: Vec<usize> = (0..n_sites).collect();
    order.sort_by(|&i, &j| raw[i].total_cmp(&raw[j]));
    let mut out = vec![1.0f64; n_sites];
    for c in 0..n_cat_eff {
        let start = c * n_sites / n_cat_eff;
        let mut end = (c + 1) * n_sites / n_cat_eff;
        if end <= start {
            end = (start + 1).min(n_sites);
        }
        let mut sum = 0.0f64;
        for pos in start..end {
            sum += raw[order[pos]];
        }
        let mean = (sum / ((end - start) as f64)).max(min_rate);
        for pos in start..end {
            out[order[pos]] = mean;
        }
    }

    let mean_rate = out.iter().sum::<f64>() / (n_sites as f64);
    if mean_rate.is_finite() && mean_rate > 0.0 {
        let inv = 1.0 / mean_rate;
        for r in out.iter_mut() {
            *r *= inv;
        }
    }
    out
}

fn build_cat_rate_grid(n_cat: usize) -> Vec<f64> {
    let n = n_cat.max(1);
    if n == 1 {
        return vec![1.0];
    }
    let min_rate = env_f64_or("JANUSX_ML_COMPAT_CAT_MIN_RATE", 0.05)
        .max(1e-6)
        .min(10.0);
    let max_rate = env_f64_or("JANUSX_ML_COMPAT_CAT_MAX_RATE", 20.0)
        .max(min_rate + 1e-6)
        .min(1e6);
    let ln_min = min_rate.ln();
    let ln_max = max_rate.ln();
    let mut out = vec![1.0f64; n];
    for i in 0..n {
        let t = if n > 1 {
            (i as f64) / ((n - 1) as f64)
        } else {
            0.0
        };
        out[i] = (ln_min + (ln_max - ln_min) * t).exp();
    }
    let mean = out.iter().sum::<f64>() / (n as f64);
    if mean.is_finite() && mean > 0.0 {
        let inv = 1.0 / mean;
        for v in out.iter_mut() {
            *v *= inv;
        }
    }
    out
}

fn assign_sites_to_cat_grid(initial_rates: &[f64], cat_rates: &[f64]) -> (Vec<u8>, Vec<f64>) {
    let n_sites = initial_rates.len();
    let n_cat = cat_rates.len().max(1);
    if n_cat == 1 {
        return (vec![0u8; n_sites], vec![1.0f64; n_sites]);
    }
    let mut site_cat = vec![0u8; n_sites];
    let mut site_rates = vec![1.0f64; n_sites];
    let mut cat_log = vec![0.0f64; n_cat];
    for i in 0..n_cat {
        cat_log[i] = cat_rates[i].max(1e-8).ln();
    }
    for k in 0..n_sites {
        let r0 = initial_rates
            .get(k)
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0);
        let lr = r0.ln();
        let mut best = 0usize;
        let mut best_d = (lr - cat_log[0]).abs();
        for c in 1..n_cat {
            let d = (lr - cat_log[c]).abs();
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        site_cat[k] = best as u8;
        site_rates[k] = cat_rates[best];
    }
    (site_cat, site_rates)
}

fn compat_cat_prior_from_assign(site_cat: &[u8], n_cat: usize, alpha: f64) -> Vec<f64> {
    let n = n_cat.max(1);
    let a = alpha.max(0.0);
    let mut prior = vec![a; n];
    for &c in site_cat {
        let idx = (c as usize).min(n - 1);
        prior[idx] += 1.0;
    }
    let sum = prior.iter().sum::<f64>();
    if sum.is_finite() && sum > 0.0 {
        let inv = 1.0 / sum;
        for p in prior.iter_mut() {
            *p *= inv;
        }
    } else {
        let u = 1.0 / (n as f64);
        for p in prior.iter_mut() {
            *p = u;
        }
    }
    prior
}

fn compat_cat_update_site_subset(n_sites: usize, budget: usize, seed: u64) -> Vec<usize> {
    if budget >= n_sites {
        ml_build_site_indices(n_sites, 0)
    } else {
        sample_site_indices_stratified(n_sites, budget.max(1), seed)
    }
}

fn compat_cat_reassign_subset(
    tree: &MlTree,
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    cat_rates: &[f64],
    site_cat: &mut [u8],
    site_rates: &mut [f64],
    update_sites: &[usize],
    cat_prior: Option<&[f64]>,
    posterior_tau: f64,
    keep_threshold: f64,
    soft_rate: bool,
    prior_weight: f64,
    pool: Option<&ThreadPool>,
) -> Result<(usize, f64), String> {
    let n_sites = a.shape()[1];
    if update_sites.is_empty() || n_sites == 0 {
        return Ok((0, 0.0));
    }
    if site_cat.len() != n_sites || site_rates.len() != n_sites {
        return Err("CAT reassignment received mismatched site buffers".to_string());
    }
    let n_cat = cat_rates.len();
    if n_cat == 0 {
        return Ok((0, 0.0));
    }
    let post = ml_postorder(tree)?;
    let n_samples = a.shape()[0];
    let n_cols = a.shape()[1];
    let n_nodes = tree.nodes.len();
    let tau = posterior_tau.max(1e-6);
    let keep_p = keep_threshold.clamp(0.0, 1.0);
    let p_w = prior_weight.clamp(0.0, 8.0);
    let mut log_prior = vec![0.0f64; n_cat];
    if let Some(pr) = cat_prior {
        if pr.len() == n_cat {
            for i in 0..n_cat {
                log_prior[i] = pr[i].max(1e-12).ln();
            }
        } else {
            let u = (1.0 / (n_cat as f64)).ln();
            for i in 0..n_cat {
                log_prior[i] = u;
            }
        }
    } else {
        let u = (1.0 / (n_cat as f64)).ln();
        for i in 0..n_cat {
            log_prior[i] = u;
        }
    }
    let prev_cat = site_cat.to_vec();
    let par_min_sites = env_usize_or("JANUSX_ML_COMPAT_CAT_PAR_MIN_SITES", 256);

    let eval_site = |k: usize,
                     clv_buf: &mut [[f64; 4]],
                     score_buf: &mut [f64],
                     post_buf: &mut [f64]|
     -> Result<(usize, u8, f64, f64, bool), String> {
        if k >= n_cols {
            return Err("CAT reassignment site index out of range".to_string());
        }
        let prev = (prev_cat[k] as usize).min(n_cat - 1);
        let mut best_cat = prev;
        let mut best_score = f64::NEG_INFINITY;
        for c in 0..n_cat {
            let r = cat_rates[c].max(1e-8);
            let ll = jc69_loglik_one_site(tree, a, &post, n_samples, n_cols, k, r, clv_buf)?;
            let s = if ll.is_finite() {
                ll + p_w * log_prior[c]
            } else {
                f64::NEG_INFINITY
            };
            score_buf[c] = s;
            if s > best_score {
                best_score = s;
                best_cat = c;
            }
        }
        if !best_score.is_finite() {
            return Ok((k, prev as u8, cat_rates[prev], 0.0, false));
        }
        let mut zsum = 0.0f64;
        for c in 0..n_cat {
            let s = score_buf[c];
            let z = if s.is_finite() {
                ((s - best_score) / tau).clamp(-60.0, 0.0).exp()
            } else {
                0.0
            };
            post_buf[c] = z;
            zsum += z;
        }
        if !zsum.is_finite() || zsum <= 0.0 {
            let changed = best_cat != prev;
            return Ok((k, best_cat as u8, cat_rates[best_cat], 1.0, changed));
        }
        let best_post = post_buf[best_cat] / zsum;
        let mut new_cat = best_cat;
        if best_post < keep_p {
            new_cat = prev;
        }
        let new_rate = if soft_rate {
            let mut r = 0.0f64;
            for c in 0..n_cat {
                r += (post_buf[c] / zsum) * cat_rates[c];
            }
            if r.is_finite() && r > 0.0 {
                r
            } else {
                cat_rates[new_cat]
            }
        } else {
            cat_rates[new_cat]
        };
        Ok((k, new_cat as u8, new_rate, best_post, new_cat != prev))
    };

    let out: Vec<(usize, u8, f64, f64, bool)> = if update_sites.len() >= par_min_sites {
        let run_par = || -> Result<Vec<(usize, u8, f64, f64, bool)>, String> {
            update_sites
                .par_iter()
                .map_init(
                    || {
                        (
                            vec![[0.0f64; 4]; n_nodes],
                            vec![0.0f64; n_cat],
                            vec![0.0f64; n_cat],
                        )
                    },
                    |(clv_buf, score_buf, post_buf), &k| {
                        eval_site(
                            k,
                            clv_buf.as_mut_slice(),
                            score_buf.as_mut_slice(),
                            post_buf.as_mut_slice(),
                        )
                    },
                )
                .collect()
        };
        if let Some(tp) = pool {
            tp.install(run_par)?
        } else {
            run_par()?
        }
    } else {
        let mut out_local = Vec::with_capacity(update_sites.len());
        let mut clv_buf = vec![[0.0f64; 4]; n_nodes];
        let mut score_buf = vec![0.0f64; n_cat];
        let mut post_buf = vec![0.0f64; n_cat];
        for &k in update_sites {
            out_local.push(eval_site(
                k,
                clv_buf.as_mut_slice(),
                score_buf.as_mut_slice(),
                post_buf.as_mut_slice(),
            )?);
        }
        out_local
    };

    let mut changed = 0usize;
    let mut conf_sum = 0.0f64;
    let mut conf_n = 0usize;
    for (k, c, r, conf, moved) in out {
        if k >= n_sites {
            continue;
        }
        site_cat[k] = c;
        site_rates[k] = if r.is_finite() && r > 0.0 { r } else { 1.0 };
        if moved {
            changed += 1;
        }
        if conf.is_finite() {
            conf_sum += conf;
            conf_n += 1;
        }
    }
    let avg_conf = if conf_n > 0 {
        conf_sum / (conf_n as f64)
    } else {
        0.0
    };
    Ok((changed, avg_conf))
}

#[inline]
fn site_rate_global(site_rates: Option<&[f64]>, site_k: usize) -> f64 {
    if let Some(sr) = site_rates {
        if let Some(&r) = sr.get(site_k) {
            if r.is_finite() && r > 0.0 {
                return r;
            }
        }
    }
    1.0
}

#[inline]
fn jc_child_message(child: &[f64; 4], t: f64) -> [f64; 4] {
    let tt = if t.is_finite() {
        t.max(1e-8).min(10.0)
    } else {
        1e-8
    };
    let e = (-4.0 * tt / 3.0).exp();
    let same = 0.25 + 0.75 * e;
    let diff = 0.25 - 0.25 * e;
    let k = same - diff;
    let sum = child[0] + child[1] + child[2] + child[3];
    [
        diff * sum + k * child[0],
        diff * sum + k * child[1],
        diff * sum + k * child[2],
        diff * sum + k * child[3],
    ]
}

fn ml_postorder(tree: &MlTree) -> Result<Vec<usize>, String> {
    if tree.root >= tree.nodes.len() {
        return Err("ML tree root out of range".to_string());
    }
    let mut out = Vec::with_capacity(tree.nodes.len());
    let mut stack: Vec<(usize, bool)> = vec![(tree.root, false)];
    while let Some((u, visited)) = stack.pop() {
        if !visited {
            stack.push((u, true));
            if let Some((l, r)) = tree.children(u) {
                stack.push((l, false));
                stack.push((r, false));
            }
        } else {
            out.push(u);
        }
    }
    Ok(out)
}

fn ml_preorder(tree: &MlTree) -> Result<Vec<usize>, String> {
    if tree.root >= tree.nodes.len() {
        return Err("ML tree root out of range".to_string());
    }
    let mut out = Vec::with_capacity(tree.nodes.len());
    let mut stack: Vec<usize> = vec![tree.root];
    while let Some(u) = stack.pop() {
        out.push(u);
        if let Some((l, r)) = tree.children(u) {
            // push right first so left is visited earlier.
            stack.push(r);
            stack.push(l);
        }
    }
    Ok(out)
}

#[inline]
fn normalize_vec4_inplace(v: &mut [f64; 4]) {
    let mx = v[0].max(v[1]).max(v[2]).max(v[3]);
    if !mx.is_finite() || mx <= 0.0 {
        *v = [1.0, 1.0, 1.0, 1.0];
        return;
    }
    let inv = 1.0 / mx;
    for s in 0..4 {
        v[s] *= inv;
    }
}

fn jc69_loglik_one_site(
    tree: &MlTree,
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    post: &[usize],
    n_samples: usize,
    n_cols: usize,
    k: usize,
    site_rate: f64,
    clv: &mut [[f64; 4]],
) -> Result<f64, String> {
    if k >= n_cols {
        return Err("site index out of range in ML likelihood".to_string());
    }
    let mut log_scale = 0.0f64;
    for &u in post {
        if let Some((l, r)) = tree.children(u) {
            let ml = jc_child_message(&clv[l], tree.nodes[l].blen_to_parent * site_rate);
            let mr = jc_child_message(&clv[r], tree.nodes[r].blen_to_parent * site_rate);
            let mut v = [0.0f64; 4];
            for s in 0..4 {
                v[s] = ml[s] * mr[s];
            }
            let max_v = v[0].max(v[1]).max(v[2]).max(v[3]);
            if !max_v.is_finite() || max_v <= 0.0 {
                return Ok(f64::NEG_INFINITY);
            }
            let inv = 1.0 / max_v;
            for s in 0..4 {
                clv[u][s] = v[s] * inv;
            }
            log_scale += max_v.ln();
        } else {
            let leaf_ix = tree.nodes[u]
                .leaf_ix
                .ok_or_else(|| "ML leaf node missing sample index".to_string())?;
            if leaf_ix >= n_samples {
                return Err("ML leaf sample index out of range".to_string());
            }
            clv[u] = jc_leaf_partials(a[[leaf_ix, k]]);
        }
    }
    let root = tree.root;
    let sum = clv[root][0] + clv[root][1] + clv[root][2] + clv[root][3];
    if !sum.is_finite() || sum <= 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(log_scale + (0.25 * sum).ln())
}

fn jc69_loglik_sites_with_pool(
    tree: &MlTree,
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    sites: &[usize],
    pool: Option<&ThreadPool>,
) -> Result<f64, String> {
    jc69_loglik_sites_with_pool_rates(tree, a, sites, pool, None)
}

fn jc69_loglik_sites_with_pool_rates(
    tree: &MlTree,
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    sites: &[usize],
    pool: Option<&ThreadPool>,
    site_rates: Option<&[f64]>,
) -> Result<f64, String> {
    let n_samples = a.shape()[0];
    let n_cols = a.shape()[1];
    let post = ml_postorder(tree)?;
    let n_nodes = tree.nodes.len();
    let par_min_sites = env_usize_or("JANUSX_ML_LOGLIK_PAR_MIN_SITES", 16384);

    if sites.len() >= par_min_sites {
        let run_par = || -> Result<Vec<f64>, String> {
            sites
                .par_iter()
                .map_init(
                    || vec![[0.0f64; 4]; n_nodes],
                    |clv_buf, &k| {
                        let r = site_rate_global(site_rates, k);
                        jc69_loglik_one_site(
                            tree,
                            a,
                            &post,
                            n_samples,
                            n_cols,
                            k,
                            r,
                            clv_buf.as_mut_slice(),
                        )
                    },
                )
                .collect()
        };
        let site_lls = if let Some(tp) = pool {
            tp.install(run_par)?
        } else {
            run_par()?
        };
        if site_lls.iter().all(|v| v.is_finite()) {
            Ok(site_lls.iter().sum())
        } else {
            Ok(f64::NEG_INFINITY)
        }
    } else {
        let mut clv = vec![[0.0f64; 4]; n_nodes];
        let mut ll = 0.0f64;
        for &k in sites {
            let r = site_rate_global(site_rates, k);
            let sll = jc69_loglik_one_site(tree, a, &post, n_samples, n_cols, k, r, &mut clv)?;
            if !sll.is_finite() {
                return Ok(f64::NEG_INFINITY);
            }
            ll += sll;
        }
        Ok(ll)
    }
}

fn ml_build_site_indices(n_sites: usize, budget: usize) -> Vec<usize> {
    if budget == 0 || budget >= n_sites {
        return (0..n_sites).collect();
    }
    let mut out = Vec::with_capacity(budget + 1);
    for i in 0..budget {
        let idx = i.saturating_mul(n_sites) / budget;
        if out.last().copied() != Some(idx) {
            out.push(idx);
        }
    }
    if out.last().copied() != Some(n_sites - 1) {
        out.push(n_sites - 1);
    }
    out
}

#[derive(Clone)]
struct JcSiteCache {
    sites: Vec<usize>,
    site_rates: Vec<f64>,
    clv: Vec<f32>,      // layout: [site][node][state]
    logscale: Vec<f32>, // layout: [site][node]
    n_nodes: usize,
    total_ll: f64,
}

#[inline]
fn cache_clv_off(cache: &JcSiteCache, site_ix: usize, node: usize, state: usize) -> usize {
    ((site_ix * cache.n_nodes + node) * 4) + state
}

#[inline]
fn cache_scale_off(cache: &JcSiteCache, site_ix: usize, node: usize) -> usize {
    site_ix * cache.n_nodes + node
}

#[inline]
fn cache_get_clv4(cache: &JcSiteCache, site_ix: usize, node: usize) -> [f64; 4] {
    let off = cache_clv_off(cache, site_ix, node, 0);
    [
        f64::from(cache.clv[off]),
        f64::from(cache.clv[off + 1]),
        f64::from(cache.clv[off + 2]),
        f64::from(cache.clv[off + 3]),
    ]
}

#[inline]
fn cache_get_scale(cache: &JcSiteCache, site_ix: usize, node: usize) -> f64 {
    f64::from(cache.logscale[cache_scale_off(cache, site_ix, node)])
}

#[inline]
fn cache_site_rate(cache: &JcSiteCache, site_ix: usize) -> f64 {
    cache
        .site_rates
        .get(site_ix)
        .copied()
        .filter(|r| r.is_finite() && *r > 0.0)
        .unwrap_or(1.0)
}

#[inline]
fn up_off(n_nodes: usize, site_ix: usize, node: usize, state: usize) -> usize {
    ((site_ix * n_nodes + node) * 4) + state
}

#[inline]
fn up_get4(up: &[f32], n_nodes: usize, site_ix: usize, node: usize) -> [f64; 4] {
    let off = up_off(n_nodes, site_ix, node, 0);
    [
        f64::from(up[off]),
        f64::from(up[off + 1]),
        f64::from(up[off + 2]),
        f64::from(up[off + 3]),
    ]
}

fn build_jc_site_cache(
    tree: &MlTree,
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    sites: Vec<usize>,
    pool: Option<&ThreadPool>,
) -> Result<JcSiteCache, String> {
    build_jc_site_cache_with_rates(tree, a, sites, pool, None)
}

fn build_jc_site_cache_with_rates(
    tree: &MlTree,
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    sites: Vec<usize>,
    pool: Option<&ThreadPool>,
    site_rates: Option<&[f64]>,
) -> Result<JcSiteCache, String> {
    let n_nodes = tree.nodes.len();
    if n_nodes == 0 {
        return Err("empty tree in JC site cache".to_string());
    }
    let post = ml_postorder(tree)?;
    let n_samples = a.shape()[0];
    let n_cols = a.shape()[1];
    let n_sites = sites.len();
    let mut clv = vec![0.0f32; n_sites * n_nodes * 4];
    let mut logscale = vec![0.0f32; n_sites * n_nodes];
    for &k in &sites {
        if k >= n_cols {
            return Err("site index out of range when building JC cache".to_string());
        }
        if let Some(sr) = site_rates {
            if k >= sr.len() {
                return Err("site index out of range for CAT-lite site rates".to_string());
            }
        }
    }
    let cache_site_rates: Vec<f64> = sites
        .iter()
        .map(|&k| site_rate_global(site_rates, k))
        .collect();
    let site_stride = n_nodes * 4;
    let par_min_sites = env_usize_or("JANUSX_ML_CACHE_PAR_MIN_SITES", 128);

    let eval_site =
        |sidx: usize, clv_site: &mut [f32], log_site: &mut [f32]| -> Result<f64, String> {
            let k = sites[sidx];
            let rate = cache_site_rates[sidx];
            for &u in &post {
                if let Some((l, r)) = tree.children(u) {
                    let l_off = l * 4;
                    let r_off = r * 4;
                    let l_vec = [
                        f64::from(clv_site[l_off]),
                        f64::from(clv_site[l_off + 1]),
                        f64::from(clv_site[l_off + 2]),
                        f64::from(clv_site[l_off + 3]),
                    ];
                    let r_vec = [
                        f64::from(clv_site[r_off]),
                        f64::from(clv_site[r_off + 1]),
                        f64::from(clv_site[r_off + 2]),
                        f64::from(clv_site[r_off + 3]),
                    ];
                    let ml = jc_child_message(&l_vec, tree.nodes[l].blen_to_parent * rate);
                    let mr = jc_child_message(&r_vec, tree.nodes[r].blen_to_parent * rate);
                    let mut v = [0.0f64; 4];
                    for st in 0..4 {
                        v[st] = ml[st] * mr[st];
                    }
                    let max_v = v[0].max(v[1]).max(v[2]).max(v[3]);
                    if !max_v.is_finite() || max_v <= 0.0 {
                        return Ok(f64::NEG_INFINITY);
                    }
                    let inv = 1.0 / max_v;
                    let u_off = u * 4;
                    clv_site[u_off] = (v[0] * inv) as f32;
                    clv_site[u_off + 1] = (v[1] * inv) as f32;
                    clv_site[u_off + 2] = (v[2] * inv) as f32;
                    clv_site[u_off + 3] = (v[3] * inv) as f32;
                    log_site[u] =
                        (f64::from(log_site[l]) + f64::from(log_site[r]) + max_v.ln()) as f32;
                } else {
                    let leaf_ix = tree.nodes[u].leaf_ix.ok_or_else(|| {
                        "ML leaf node missing sample index in JC cache".to_string()
                    })?;
                    if leaf_ix >= n_samples {
                        return Err("ML leaf sample index out of range in JC cache".to_string());
                    }
                    let lp = jc_leaf_partials(a[[leaf_ix, k]]);
                    let u_off = u * 4;
                    clv_site[u_off] = lp[0] as f32;
                    clv_site[u_off + 1] = lp[1] as f32;
                    clv_site[u_off + 2] = lp[2] as f32;
                    clv_site[u_off + 3] = lp[3] as f32;
                    log_site[u] = 0.0f32;
                }
            }
            let r_off = tree.root * 4;
            let sum = f64::from(clv_site[r_off])
                + f64::from(clv_site[r_off + 1])
                + f64::from(clv_site[r_off + 2])
                + f64::from(clv_site[r_off + 3]);
            if !sum.is_finite() || sum <= 0.0 {
                return Ok(f64::NEG_INFINITY);
            }
            Ok(f64::from(log_site[tree.root]) + (0.25 * sum).ln())
        };

    let site_lls: Vec<f64> = if n_sites >= par_min_sites {
        let mut run_par = || -> Result<Vec<f64>, String> {
            clv.par_chunks_mut(site_stride)
                .zip(logscale.par_chunks_mut(n_nodes))
                .enumerate()
                .map(|(sidx, (clv_site, log_site))| eval_site(sidx, clv_site, log_site))
                .collect()
        };
        if let Some(tp) = pool {
            tp.install(run_par)?
        } else {
            run_par()?
        }
    } else {
        let mut out = Vec::with_capacity(n_sites);
        for sidx in 0..n_sites {
            let c0 = sidx * site_stride;
            let c1 = c0 + site_stride;
            let l0 = sidx * n_nodes;
            let l1 = l0 + n_nodes;
            out.push(eval_site(sidx, &mut clv[c0..c1], &mut logscale[l0..l1])?);
        }
        out
    };
    let total_ll = if site_lls.iter().all(|v| v.is_finite()) {
        site_lls.iter().sum()
    } else {
        f64::NEG_INFINITY
    };

    Ok(JcSiteCache {
        sites,
        site_rates: cache_site_rates,
        clv,
        logscale,
        n_nodes,
        total_ll,
    })
}

fn build_jc_up_messages(
    tree: &MlTree,
    cache: &JcSiteCache,
    pool: Option<&ThreadPool>,
) -> Result<Vec<f32>, String> {
    let n_nodes = tree.nodes.len();
    let n_sites = cache.sites.len();
    let mut up = vec![0.0f32; n_sites * n_nodes * 4];
    let preorder = ml_preorder(tree)?;
    let root = tree.root;
    let site_stride = n_nodes * 4;
    let par_min_sites = env_usize_or("JANUSX_ML_UP_PAR_MIN_SITES", 128);

    let eval_site = |sidx: usize, up_site: &mut [f32]| -> Result<(), String> {
        let rate = cache_site_rate(cache, sidx);
        let root_off = root * 4;
        up_site[root_off] = 0.25f32;
        up_site[root_off + 1] = 0.25f32;
        up_site[root_off + 2] = 0.25f32;
        up_site[root_off + 3] = 0.25f32;

        for &p in &preorder {
            if let Some((l, r)) = tree.children(p) {
                let t_l = tree.nodes[l].blen_to_parent * rate;
                let t_r = tree.nodes[r].blen_to_parent * rate;
                let cl = cache_get_clv4(cache, sidx, l);
                let cr = cache_get_clv4(cache, sidx, r);
                let ml = jc_child_message(&cl, t_l);
                let mr = jc_child_message(&cr, t_r);

                let p_off = p * 4;
                let up_p = [
                    f64::from(up_site[p_off]),
                    f64::from(up_site[p_off + 1]),
                    f64::from(up_site[p_off + 2]),
                    f64::from(up_site[p_off + 3]),
                ];

                let mut ctx_l = [0.0f64; 4];
                let mut ctx_r = [0.0f64; 4];
                for st in 0..4 {
                    ctx_l[st] = up_p[st] * mr[st];
                    ctx_r[st] = up_p[st] * ml[st];
                }
                let mut up_l = jc_child_message(&ctx_l, t_l);
                let mut up_r = jc_child_message(&ctx_r, t_r);
                normalize_vec4_inplace(&mut up_l);
                normalize_vec4_inplace(&mut up_r);

                let l_off = l * 4;
                let r_off = r * 4;
                up_site[l_off] = up_l[0] as f32;
                up_site[l_off + 1] = up_l[1] as f32;
                up_site[l_off + 2] = up_l[2] as f32;
                up_site[l_off + 3] = up_l[3] as f32;
                up_site[r_off] = up_r[0] as f32;
                up_site[r_off + 1] = up_r[1] as f32;
                up_site[r_off + 2] = up_r[2] as f32;
                up_site[r_off + 3] = up_r[3] as f32;
            }
        }
        Ok(())
    };

    if n_sites >= par_min_sites {
        let mut run_par = || -> Result<Vec<()>, String> {
            up.par_chunks_mut(site_stride)
                .enumerate()
                .map(|(sidx, up_site)| eval_site(sidx, up_site))
                .collect()
        };
        if let Some(tp) = pool {
            let _ = tp.install(run_par)?;
        } else {
            let _ = run_par()?;
        }
    } else {
        for sidx in 0..n_sites {
            let u0 = sidx * site_stride;
            let u1 = u0 + site_stride;
            eval_site(sidx, &mut up[u0..u1])?;
        }
    }
    Ok(up)
}

fn collect_affected_nodes_nni(tree: &MlTree, p: usize, c: usize) -> Vec<usize> {
    let mut out = Vec::new();
    if tree.is_internal(c) {
        out.push(c);
    }
    let mut cur = Some(p);
    while let Some(u) = cur {
        if tree.is_internal(u) {
            out.push(u);
        }
        cur = tree.nodes[u].parent;
    }
    out
}

fn collect_affected_nodes_branch(tree: &MlTree, child: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut cur = tree.nodes[child].parent;
    while let Some(u) = cur {
        if tree.is_internal(u) {
            out.push(u);
        }
        cur = tree.nodes[u].parent;
    }
    out
}

fn jc69_loglik_from_cache_with_affected(
    tree: &MlTree,
    cache: &JcSiteCache,
    affected: &[usize],
    affected_pos: &mut [usize],
    touched: &mut Vec<usize>,
    tmp_clv: &mut Vec<[f64; 4]>,
    tmp_scale: &mut Vec<f64>,
) -> Result<f64, String> {
    if affected.is_empty() {
        return Ok(cache.total_ll);
    }
    touched.clear();
    for (pos, &u) in affected.iter().enumerate() {
        affected_pos[u] = pos;
        touched.push(u);
    }
    tmp_clv.resize(affected.len(), [0.0; 4]);
    tmp_scale.resize(affected.len(), 0.0);
    let root = tree.root;
    let mut ll = 0.0f64;
    for sidx in 0..cache.sites.len() {
        let rate = cache_site_rate(cache, sidx);
        for (pos, &u) in affected.iter().enumerate() {
            let (l, r) = tree
                .children(u)
                .ok_or_else(|| "affected node is not internal in cache eval".to_string())?;
            let lp = affected_pos[l];
            let rp = affected_pos[r];
            let l_vec = if lp != usize::MAX {
                tmp_clv[lp]
            } else {
                cache_get_clv4(cache, sidx, l)
            };
            let r_vec = if rp != usize::MAX {
                tmp_clv[rp]
            } else {
                cache_get_clv4(cache, sidx, r)
            };
            let l_scale = if lp != usize::MAX {
                tmp_scale[lp]
            } else {
                cache_get_scale(cache, sidx, l)
            };
            let r_scale = if rp != usize::MAX {
                tmp_scale[rp]
            } else {
                cache_get_scale(cache, sidx, r)
            };
            let ml = jc_child_message(&l_vec, tree.nodes[l].blen_to_parent * rate);
            let mr = jc_child_message(&r_vec, tree.nodes[r].blen_to_parent * rate);
            let mut v = [0.0f64; 4];
            for st in 0..4 {
                v[st] = ml[st] * mr[st];
            }
            let max_v = v[0].max(v[1]).max(v[2]).max(v[3]);
            if !max_v.is_finite() || max_v <= 0.0 {
                for &u2 in touched.iter() {
                    affected_pos[u2] = usize::MAX;
                }
                return Ok(f64::NEG_INFINITY);
            }
            let inv = 1.0 / max_v;
            tmp_clv[pos] = [v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv];
            tmp_scale[pos] = l_scale + r_scale + max_v.ln();
        }
        let rp = affected_pos[root];
        let (root_vec, root_scale) = if rp != usize::MAX {
            (tmp_clv[rp], tmp_scale[rp])
        } else {
            (
                cache_get_clv4(cache, sidx, root),
                cache_get_scale(cache, sidx, root),
            )
        };
        let sum = root_vec[0] + root_vec[1] + root_vec[2] + root_vec[3];
        if !sum.is_finite() || sum <= 0.0 {
            for &u2 in touched.iter() {
                affected_pos[u2] = usize::MAX;
            }
            return Ok(f64::NEG_INFINITY);
        }
        ll += root_scale + (0.25 * sum).ln();
    }
    for &u2 in touched.iter() {
        affected_pos[u2] = usize::MAX;
    }
    Ok(ll)
}

fn collect_nni_edges(tree: &MlTree) -> Vec<NniEdge> {
    let mut out = Vec::new();
    for c in 0..tree.nodes.len() {
        let Some(p) = tree.nodes[c].parent else {
            continue;
        };
        if !tree.is_internal(c) || !tree.is_internal(p) {
            continue;
        }
        let Some((pl, pr)) = tree.children(p) else {
            continue;
        };
        let s = if pl == c {
            pr
        } else if pr == c {
            pl
        } else {
            continue;
        };
        let Some((c1, c2)) = tree.children(c) else {
            continue;
        };
        out.push(NniEdge { p, c, s, c1, c2 });
    }
    out
}

fn nni_edge_for_pc(tree: &MlTree, p: usize, c: usize) -> Option<NniEdge> {
    if !tree.is_internal(p) || !tree.is_internal(c) {
        return None;
    }
    let (pl, pr) = tree.children(p)?;
    let s = if pl == c {
        pr
    } else if pr == c {
        pl
    } else {
        return None;
    };
    let (c1, c2) = tree.children(c)?;
    Some(NniEdge { p, c, s, c1, c2 })
}

fn select_nni_edges(tree: &MlTree, edges: &[NniEdge], budget: usize) -> Vec<NniEdge> {
    if budget == 0 || edges.len() <= budget {
        return edges.to_vec();
    }
    let mut idx: Vec<usize> = (0..edges.len()).collect();
    idx.sort_by(|&ia, &ib| {
        let la = tree.nodes[edges[ia].c].blen_to_parent;
        let lb = tree.nodes[edges[ib].c].blen_to_parent;
        la.total_cmp(&lb)
    });
    idx.into_iter().take(budget).map(|i| edges[i]).collect()
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum CompatEdgeSelectMode {
    Short,
    Window,
    Mixed,
}

fn resolve_compat_edge_select_mode() -> CompatEdgeSelectMode {
    let raw = std::env::var("JANUSX_ML_COMPAT_EDGE_SELECT")
        .ok()
        .unwrap_or_else(|| "short".to_string());
    match raw.trim().to_ascii_lowercase().as_str() {
        "short" | "shortest" | "blen" => CompatEdgeSelectMode::Short,
        "window" | "sweep" | "roundrobin" | "rr" => CompatEdgeSelectMode::Window,
        _ => CompatEdgeSelectMode::Mixed,
    }
}

fn select_nni_edges_window(
    edges: &[NniEdge],
    budget: usize,
    cursor: &mut usize,
) -> Vec<NniEdge> {
    if budget == 0 || edges.len() <= budget {
        return edges.to_vec();
    }
    let n = edges.len();
    let start = *cursor % n;
    let mut out: Vec<NniEdge> = Vec::with_capacity(budget);
    let mut used = vec![false; n];
    for k in 0..budget {
        let mut idx = (start + (k * n) / budget) % n;
        if used[idx] {
            let mut probe = 0usize;
            while probe < n && used[idx] {
                idx = (idx + 1) % n;
                probe += 1;
            }
            if probe >= n {
                break;
            }
        }
        used[idx] = true;
        out.push(edges[idx]);
    }
    let step = (n / budget).max(1);
    *cursor = (start + step) % n;
    out
}

fn select_nni_edges_compat(
    tree: &MlTree,
    edges: &[NniEdge],
    budget: usize,
    cursor: &mut usize,
    mode: CompatEdgeSelectMode,
) -> Vec<NniEdge> {
    if budget == 0 || edges.len() <= budget {
        return edges.to_vec();
    }
    match mode {
        CompatEdgeSelectMode::Short => select_nni_edges(tree, edges, budget),
        CompatEdgeSelectMode::Window => select_nni_edges_window(edges, budget, cursor),
        CompatEdgeSelectMode::Mixed => {
            let mut out: Vec<NniEdge> = Vec::with_capacity(budget);
            let window_keep = ((budget * 3) / 4).max(1).min(budget);
            let mut seen: HashSet<(usize, usize, usize)> =
                HashSet::with_capacity(budget.saturating_mul(2));

            for e in select_nni_edges_window(edges, window_keep, cursor) {
                let key = (e.p, e.c, e.s);
                if seen.insert(key) {
                    out.push(e);
                }
            }

            if out.len() < budget {
                let mut idx: Vec<usize> = (0..edges.len()).collect();
                idx.sort_by(|&ia, &ib| {
                    let la = tree.nodes[edges[ia].c].blen_to_parent;
                    let lb = tree.nodes[edges[ib].c].blen_to_parent;
                    la.total_cmp(&lb)
                });
                for i in idx {
                    let e = edges[i];
                    let key = (e.p, e.c, e.s);
                    if seen.insert(key) {
                        out.push(e);
                        if out.len() >= budget {
                            break;
                        }
                    }
                }
            }

            if out.len() < budget {
                for e in select_nni_edges_window(edges, budget, cursor) {
                    let key = (e.p, e.c, e.s);
                    if seen.insert(key) {
                        out.push(e);
                        if out.len() >= budget {
                            break;
                        }
                    }
                }
            }
            out
        }
    }
}

fn replace_child(parent: &mut MlNode, old_child: usize, new_child: usize) -> Result<(), String> {
    match (parent.left, parent.right) {
        (Some(l), Some(r)) => {
            if l == old_child {
                parent.left = Some(new_child);
                Ok(())
            } else if r == old_child {
                parent.right = Some(new_child);
                Ok(())
            } else {
                Err("failed NNI swap: old child not found".to_string())
            }
        }
        _ => Err("failed NNI swap: parent is not binary internal node".to_string()),
    }
}

fn apply_nni_swap(tree: &mut MlTree, p: usize, c: usize, take_from_c: usize) -> Result<(), String> {
    if !tree.is_internal(p) || !tree.is_internal(c) {
        return Err("failed NNI swap: p/c must both be internal".to_string());
    }
    let (pl, pr) = tree
        .children(p)
        .ok_or_else(|| "failed NNI swap: invalid parent children".to_string())?;
    let s = if pl == c {
        pr
    } else if pr == c {
        pl
    } else {
        return Err("failed NNI swap: c is not child of p".to_string());
    };
    let (c1, c2) = tree
        .children(c)
        .ok_or_else(|| "failed NNI swap: invalid child children".to_string())?;
    if take_from_c != c1 && take_from_c != c2 {
        return Err("failed NNI swap: chosen subtree is not child of c".to_string());
    }

    {
        let parent = &mut tree.nodes[p];
        replace_child(parent, s, take_from_c)?;
    }
    {
        let child = &mut tree.nodes[c];
        replace_child(child, take_from_c, s)?;
    }
    tree.nodes[take_from_c].parent = Some(p);
    tree.nodes[s].parent = Some(c);
    Ok(())
}

fn nni_local_focus_nodes(tree: &MlTree, m: &NniMove) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    let mut push_unique = |u: usize| {
        if u < tree.nodes.len() && tree.nodes[u].parent.is_some() && !out.contains(&u) {
            out.push(u);
        }
    };
    push_unique(m.c);
    push_unique(m.s);
    push_unique(m.swap_child);
    if let Some((x, y)) = tree.children(m.c) {
        push_unique(x);
        push_unique(y);
    }
    if tree.nodes[m.p].parent.is_some() {
        push_unique(m.p);
    }
    out
}

fn optimize_local_branch_lengths_jc69(
    tree: &mut MlTree,
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    quick_sites: &[usize],
    focus_nodes: &[usize],
    passes: usize,
    pool: Option<&ThreadPool>,
) -> Result<(), String> {
    if passes == 0 || focus_nodes.is_empty() {
        return Ok(());
    }
    let n_nodes = tree.nodes.len();
    let mut affected_pos = vec![usize::MAX; n_nodes];
    let mut touched = Vec::<usize>::new();
    let mut tmp_clv = Vec::<[f64; 4]>::new();
    let mut tmp_scale = Vec::<f64>::new();
    let coarse = [0.55f64, 0.72, 0.90, 1.00, 1.12, 1.30, 1.55];

    for _ in 0..passes {
        let mut pass_improved = false;
        for &child in focus_nodes {
            if child >= tree.nodes.len() || tree.nodes[child].parent.is_none() {
                continue;
            }
            let affected = collect_affected_nodes_branch(tree, child);
            if affected.is_empty() {
                continue;
            }
            let cache = build_jc_site_cache(tree, a, quick_sites.to_vec(), pool)?;
            let cur_ll = cache.total_ll;
            if !cur_ll.is_finite() {
                continue;
            }

            let base = tree.nodes[child].blen_to_parent;
            let mut best_len = base;
            let mut best_ll = cur_ll;

            for &f in &coarse {
                let cand = clamp_blen(base * f);
                tree.nodes[child].blen_to_parent = cand;
                let q = jc69_loglik_from_cache_with_affected(
                    tree,
                    &cache,
                    &affected,
                    &mut affected_pos,
                    &mut touched,
                    &mut tmp_clv,
                    &mut tmp_scale,
                )?;
                if q.is_finite() && q > best_ll {
                    best_ll = q;
                    best_len = cand;
                }
            }

            let step = (best_len * 0.18).max(1e-4);
            for cand in [best_len - step, best_len + step] {
                let cand = clamp_blen(cand);
                tree.nodes[child].blen_to_parent = cand;
                let q = jc69_loglik_from_cache_with_affected(
                    tree,
                    &cache,
                    &affected,
                    &mut affected_pos,
                    &mut touched,
                    &mut tmp_clv,
                    &mut tmp_scale,
                )?;
                if q.is_finite() && q > best_ll {
                    best_ll = q;
                    best_len = cand;
                }
            }

            tree.nodes[child].blen_to_parent = best_len;
            if best_ll > cur_ll + 1e-10 {
                pass_improved = true;
            }
        }
        if !pass_improved {
            break;
        }
    }
    Ok(())
}

fn optimize_local_branch_lengths_jc69_reuse(
    tree: &mut MlTree,
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    quick_sites: &[usize],
    focus_nodes: &[usize],
    passes: usize,
    pool: Option<&ThreadPool>,
) -> Result<(), String> {
    if passes == 0 || focus_nodes.is_empty() {
        return Ok(());
    }
    let n_nodes = tree.nodes.len();
    let mut affected_pos = vec![usize::MAX; n_nodes];
    let mut touched = Vec::<usize>::new();
    let mut tmp_clv = Vec::<[f64; 4]>::new();
    let mut tmp_scale = Vec::<f64>::new();
    let coarse = [0.55f64, 0.72, 0.90, 1.00, 1.12, 1.30, 1.55];

    for _ in 0..passes {
        let cache = build_jc_site_cache(tree, a, quick_sites.to_vec(), pool)?;
        let cur_ll = cache.total_ll;
        if !cur_ll.is_finite() {
            break;
        }

        let mut pass_improved = false;
        let mut updates: Vec<(usize, f64)> = Vec::with_capacity(focus_nodes.len());

        for &child in focus_nodes {
            if child >= tree.nodes.len() || tree.nodes[child].parent.is_none() {
                continue;
            }
            let affected = collect_affected_nodes_branch(tree, child);
            if affected.is_empty() {
                continue;
            }

            let base = tree.nodes[child].blen_to_parent;
            let mut best_len = base;
            let mut best_ll = cur_ll;

            for &f in &coarse {
                let cand = clamp_blen(base * f);
                tree.nodes[child].blen_to_parent = cand;
                let q = jc69_loglik_from_cache_with_affected(
                    tree,
                    &cache,
                    &affected,
                    &mut affected_pos,
                    &mut touched,
                    &mut tmp_clv,
                    &mut tmp_scale,
                )?;
                if q.is_finite() && q > best_ll {
                    best_ll = q;
                    best_len = cand;
                }
            }

            let step = (best_len * 0.18).max(1e-4);
            for cand in [best_len - step, best_len + step] {
                let cand = clamp_blen(cand);
                tree.nodes[child].blen_to_parent = cand;
                let q = jc69_loglik_from_cache_with_affected(
                    tree,
                    &cache,
                    &affected,
                    &mut affected_pos,
                    &mut touched,
                    &mut tmp_clv,
                    &mut tmp_scale,
                )?;
                if q.is_finite() && q > best_ll {
                    best_ll = q;
                    best_len = cand;
                }
            }

            tree.nodes[child].blen_to_parent = base;
            updates.push((child, best_len));
            if best_ll > cur_ll + 1e-10 {
                pass_improved = true;
            }
        }

        for (child, best_len) in updates {
            tree.nodes[child].blen_to_parent = best_len;
        }
        if !pass_improved {
            break;
        }
    }
    Ok(())
}

fn optimize_nni_ml_jc69(
    tree: &mut MlTree,
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    mode: &str,
    pool: Option<&ThreadPool>,
) -> Result<(usize, f64), String> {
    let n_sites = a.shape()[1];
    if n_sites == 0 {
        return Ok((0, f64::NEG_INFINITY));
    }
    let is_exact = mode == "exact";
    let quick_site_budget = if is_exact {
        env_usize_or("JANUSX_ML_QUICK_SITES_EXACT", 4096)
    } else {
        env_usize_or("JANUSX_ML_QUICK_SITES_APPROX", 512)
    };
    let max_iter = if is_exact {
        env_usize_or("JANUSX_ML_EXACT_MAX_ITER", 12).max(1)
    } else {
        env_usize_or("JANUSX_ML_APPROX_MAX_ITER", 4).max(1)
    };
    let edge_budget = if is_exact {
        env_usize_or("JANUSX_ML_EXACT_EDGE_BUDGET", 384).max(1)
    } else {
        env_usize_or("JANUSX_ML_APPROX_EDGE_BUDGET", 96).max(1)
    };
    let verify_topk = if is_exact {
        env_usize_or("JANUSX_ML_EXACT_VERIFY_TOPK", 5).max(1)
    } else {
        env_usize_or("JANUSX_ML_APPROX_VERIFY_TOPK", 1).max(1)
    };
    let base_tol = if is_exact {
        env_f64_or("JANUSX_ML_EXACT_MIN_IMPROVE", 1e-6).max(0.0)
    } else {
        env_f64_or("JANUSX_ML_APPROX_MIN_IMPROVE", 1e-5).max(0.0)
    };
    let stall_limit = if is_exact {
        env_usize_or("JANUSX_ML_EXACT_STALL_LIMIT", 2).max(1)
    } else {
        1
    };
    let exact_full_recheck = if is_exact {
        env_bool_or("JANUSX_ML_EXACT_FULL_RECHECK", true)
    } else {
        false
    };
    let exact_full_recheck_every = if is_exact {
        env_usize_or("JANUSX_ML_EXACT_FULL_RECHECK_EVERY", 1).max(1)
    } else {
        1
    };
    let local_blen_passes = if is_exact {
        env_usize_or("JANUSX_ML_LOCAL_BLEN_PASSES_EXACT", 2)
    } else {
        env_usize_or("JANUSX_ML_LOCAL_BLEN_PASSES_APPROX", 1)
    };
    let local_blen_reuse = env_bool_or("JANUSX_ML_LOCAL_BLEN_REUSE", true);

    let full_sites = ml_build_site_indices(n_sites, 0);
    let quick_sites = ml_build_site_indices(n_sites, quick_site_budget);
    let mut cur_full = jc69_loglik_sites_with_pool(tree, a, &full_sites, pool)?;
    if !cur_full.is_finite() {
        return Ok((0, cur_full));
    }

    let mut accepted = 0usize;
    let mut stall_rounds = 0usize;
    let mut affected_pos = vec![usize::MAX; tree.nodes.len()];
    let mut touched = Vec::<usize>::new();
    let mut tmp_clv = Vec::<[f64; 4]>::new();
    let mut tmp_scale = Vec::<f64>::new();

    for iter_idx in 0..max_iter {
        let quick_cache = build_jc_site_cache(tree, a, quick_sites.clone(), pool)?;
        let cur_quick = quick_cache.total_ll;
        if !cur_quick.is_finite() {
            break;
        }

        let edges_all = collect_nni_edges(tree);
        if edges_all.is_empty() {
            break;
        }
        let edges = select_nni_edges(tree, &edges_all, edge_budget);
        let mut cand = Vec::<NniMove>::new();

        for e in edges {
            for swap_child in [e.c1, e.c2] {
                apply_nni_swap(tree, e.p, e.c, swap_child)?;
                let affected = collect_affected_nodes_nni(tree, e.p, e.c);
                let q = jc69_loglik_from_cache_with_affected(
                    tree,
                    &quick_cache,
                    &affected,
                    &mut affected_pos,
                    &mut touched,
                    &mut tmp_clv,
                    &mut tmp_scale,
                )?;
                apply_nni_swap(tree, e.p, e.c, e.s)?;
                if q.is_finite() && q > cur_quick + 1e-12 {
                    cand.push(NniMove {
                        p: e.p,
                        c: e.c,
                        s: e.s,
                        swap_child,
                        quick_score: q,
                    });
                }
            }
        }

        if cand.is_empty() {
            stall_rounds += 1;
            if !is_exact || stall_rounds >= stall_limit {
                break;
            }
            continue;
        }

        cand.sort_by(|x, y| y.quick_score.total_cmp(&x.quick_score));
        let verify_n = verify_topk.min(cand.len());
        let mut verify_list: Vec<NniMove> = Vec::new();
        if is_exact && exact_full_recheck && (iter_idx % exact_full_recheck_every == 0) {
            verify_list.extend(cand.iter().copied().take(verify_n));
        } else {
            verify_list.push(cand[0]);
        }

        let mut best_move: Option<NniMove> = None;
        let mut best_full = cur_full;
        for m in verify_list {
            apply_nni_swap(tree, m.p, m.c, m.swap_child)?;
            let f = jc69_loglik_sites_with_pool(tree, a, &full_sites, pool)?;
            apply_nni_swap(tree, m.p, m.c, m.s)?;
            if f.is_finite() && f > best_full + base_tol {
                best_full = f;
                best_move = Some(m);
            }
        }

        let Some(best) = best_move else {
            stall_rounds += 1;
            if !is_exact || stall_rounds >= stall_limit {
                break;
            }
            continue;
        };

        let backup_tree = tree.clone();
        apply_nni_swap(tree, best.p, best.c, best.swap_child)?;
        let focus = nni_local_focus_nodes(tree, &best);
        if local_blen_reuse {
            optimize_local_branch_lengths_jc69_reuse(
                tree,
                a,
                &quick_sites,
                &focus,
                local_blen_passes,
                pool,
            )?;
        } else {
            optimize_local_branch_lengths_jc69(
                tree,
                a,
                &quick_sites,
                &focus,
                local_blen_passes,
                pool,
            )?;
        }
        let new_full = jc69_loglik_sites_with_pool(tree, a, &full_sites, pool)?;
        if !new_full.is_finite() || new_full <= cur_full + base_tol {
            *tree = backup_tree;
            stall_rounds += 1;
            if !is_exact || stall_rounds >= stall_limit {
                break;
            }
            continue;
        }

        cur_full = new_full;
        accepted += 1;
        stall_rounds = 0;
    }

    Ok((accepted, cur_full))
}

fn compat_pick_quartet_swap(
    tree: &MlTree,
    cache: &JcSiteCache,
    up: &[f32],
    e: &NniEdge,
    eps: f64,
    optimize_t: bool,
) -> Result<Option<(usize, f64)>, String> {
    let n_cache_sites = cache.sites.len();
    if n_cache_sites == 0 {
        return Ok(None);
    }
    let n_nodes = tree.nodes.len();
    let p = e.p;
    let c = e.c;
    let a_node = e.c1;
    let b_node = e.c2;
    let c_node = e.s;

    let t_pc = tree.nodes[c].blen_to_parent;
    let t_a = tree.nodes[a_node].blen_to_parent;
    let t_b = tree.nodes[b_node].blen_to_parent;
    let t_c = tree.nodes[c_node].blen_to_parent;

    let mut msg_a_v = vec![[0.0f64; 4]; n_cache_sites];
    let mut msg_b_v = vec![[0.0f64; 4]; n_cache_sites];
    let mut msg_c_v = vec![[0.0f64; 4]; n_cache_sites];
    let mut msg_d_v = vec![[0.0f64; 4]; n_cache_sites];
    for sidx in 0..n_cache_sites {
        let rate = cache_site_rate(cache, sidx);
        let clv_a = cache_get_clv4(cache, sidx, a_node);
        let clv_b = cache_get_clv4(cache, sidx, b_node);
        let clv_c = cache_get_clv4(cache, sidx, c_node);
        let mut msg_a = jc_child_message(&clv_a, t_a * rate);
        let mut msg_b = jc_child_message(&clv_b, t_b * rate);
        let mut msg_c = jc_child_message(&clv_c, t_c * rate);
        let mut msg_d = up_get4(up, n_nodes, sidx, p);
        normalize_vec4_inplace(&mut msg_a);
        normalize_vec4_inplace(&mut msg_b);
        normalize_vec4_inplace(&mut msg_c);
        normalize_vec4_inplace(&mut msg_d);
        msg_a_v[sidx] = msg_a;
        msg_b_v[sidx] = msg_b;
        msg_c_v[sidx] = msg_c;
        msg_d_v[sidx] = msg_d;
    }

    let (t0, t1, t2) = if optimize_t && n_cache_sites >= 16 {
        (
            optimize_quartet_middle_t(
                &msg_a_v,
                &msg_b_v,
                &msg_c_v,
                &msg_d_v,
                0,
                t_pc,
                Some(cache.site_rates.as_slice()),
                eps,
            ),
            optimize_quartet_middle_t(
                &msg_a_v,
                &msg_b_v,
                &msg_c_v,
                &msg_d_v,
                1,
                t_pc,
                Some(cache.site_rates.as_slice()),
                eps,
            ),
            optimize_quartet_middle_t(
                &msg_a_v,
                &msg_b_v,
                &msg_c_v,
                &msg_d_v,
                2,
                t_pc,
                Some(cache.site_rates.as_slice()),
                eps,
            ),
        )
    } else {
        (t_pc, t_pc, t_pc)
    };

    let ll0 = quartet_total_loglik(
        &msg_a_v,
        &msg_b_v,
        &msg_c_v,
        &msg_d_v,
        0,
        t0,
        Some(cache.site_rates.as_slice()),
        eps,
    );
    let ll1 = quartet_total_loglik(
        &msg_a_v,
        &msg_b_v,
        &msg_c_v,
        &msg_d_v,
        1,
        t1,
        Some(cache.site_rates.as_slice()),
        eps,
    );
    let ll2 = quartet_total_loglik(
        &msg_a_v,
        &msg_b_v,
        &msg_c_v,
        &msg_d_v,
        2,
        t2,
        Some(cache.site_rates.as_slice()),
        eps,
    );
    if !ll0.is_finite() || !ll1.is_finite() || !ll2.is_finite() {
        return Ok(None);
    }

    let mut best_pair = 0usize;
    let mut best_ll = ll0;
    if ll1 > best_ll {
        best_ll = ll1;
        best_pair = 1;
    }
    if ll2 > best_ll {
        best_ll = ll2;
        best_pair = 2;
    }
    if best_pair == 0 {
        return Ok(None);
    }
    // Mapping to apply_nni_swap:
    // pairing-1 (AC|BD) => swap child b (c2) with sibling s
    // pairing-2 (AD|BC) => swap child a (c1) with sibling s
    let swap_child = if best_pair == 1 { e.c2 } else { e.c1 };
    Ok(Some((swap_child, best_ll - ll0)))
}

#[inline]
fn compat_jc_dist_from_miss_sum(miss_sum: f64, n_sites: usize) -> f64 {
    if n_sites == 0 {
        return 3.0;
    }
    let p = (miss_sum / (n_sites as f64)).clamp(0.0, 0.749_999_999_999);
    if p <= 0.0 {
        return 0.0;
    }
    let inner = 1.0 - (4.0 / 3.0) * p;
    if inner <= 1e-15 {
        return 3.0;
    }
    let d = -0.75 * inner.ln();
    if d.is_finite() {
        d
    } else {
        3.0
    }
}

fn compat_pick_quartet_swap_me(
    tree: &MlTree,
    cache: &JcSiteCache,
    up: &[f32],
    e: &NniEdge,
) -> Result<Option<(usize, f64)>, String> {
    let picked = compat_pick_quartet_swap_me_with_gap(tree, cache, up, e)?;
    Ok(picked.map(|(swap_child, delta, _gap)| (swap_child, delta)))
}

fn compat_pick_quartet_swap_me_with_gap(
    tree: &MlTree,
    cache: &JcSiteCache,
    up: &[f32],
    e: &NniEdge,
) -> Result<Option<(usize, f64, f64)>, String> {
    let n_cache_sites = cache.sites.len();
    if n_cache_sites == 0 {
        return Ok(None);
    }
    let n_nodes = tree.nodes.len();
    let p = e.p;
    let a_node = e.c1;
    let b_node = e.c2;
    let c_node = e.s;

    let t_a = tree.nodes[a_node].blen_to_parent;
    let t_b = tree.nodes[b_node].blen_to_parent;
    let t_c = tree.nodes[c_node].blen_to_parent;

    let mut miss_ab = 0.0f64;
    let mut miss_ac = 0.0f64;
    let mut miss_ad = 0.0f64;
    let mut miss_bc = 0.0f64;
    let mut miss_bd = 0.0f64;
    let mut miss_cd = 0.0f64;

    for sidx in 0..n_cache_sites {
        let rate = cache_site_rate(cache, sidx);
        let clv_a = cache_get_clv4(cache, sidx, a_node);
        let clv_b = cache_get_clv4(cache, sidx, b_node);
        let clv_c = cache_get_clv4(cache, sidx, c_node);
        let mut msg_a = jc_child_message(&clv_a, t_a * rate);
        let mut msg_b = jc_child_message(&clv_b, t_b * rate);
        let mut msg_c = jc_child_message(&clv_c, t_c * rate);
        let mut msg_d = up_get4(up, n_nodes, sidx, p);
        normalize_vec4_inplace(&mut msg_a);
        normalize_vec4_inplace(&mut msg_b);
        normalize_vec4_inplace(&mut msg_c);
        normalize_vec4_inplace(&mut msg_d);

        let mut m_ab = 0.0f64;
        let mut m_ac = 0.0f64;
        let mut m_ad = 0.0f64;
        let mut m_bc = 0.0f64;
        let mut m_bd = 0.0f64;
        let mut m_cd = 0.0f64;
        for k in 0..4 {
            m_ab += msg_a[k] * msg_b[k];
            m_ac += msg_a[k] * msg_c[k];
            m_ad += msg_a[k] * msg_d[k];
            m_bc += msg_b[k] * msg_c[k];
            m_bd += msg_b[k] * msg_d[k];
            m_cd += msg_c[k] * msg_d[k];
        }
        miss_ab += (1.0 - m_ab).clamp(0.0, 0.749_999_999_999);
        miss_ac += (1.0 - m_ac).clamp(0.0, 0.749_999_999_999);
        miss_ad += (1.0 - m_ad).clamp(0.0, 0.749_999_999_999);
        miss_bc += (1.0 - m_bc).clamp(0.0, 0.749_999_999_999);
        miss_bd += (1.0 - m_bd).clamp(0.0, 0.749_999_999_999);
        miss_cd += (1.0 - m_cd).clamp(0.0, 0.749_999_999_999);
    }

    let d_ab = compat_jc_dist_from_miss_sum(miss_ab, n_cache_sites);
    let d_ac = compat_jc_dist_from_miss_sum(miss_ac, n_cache_sites);
    let d_ad = compat_jc_dist_from_miss_sum(miss_ad, n_cache_sites);
    let d_bc = compat_jc_dist_from_miss_sum(miss_bc, n_cache_sites);
    let d_bd = compat_jc_dist_from_miss_sum(miss_bd, n_cache_sites);
    let d_cd = compat_jc_dist_from_miss_sum(miss_cd, n_cache_sites);

    let s0 = d_ab + d_cd;
    let s1 = d_ac + d_bd;
    let s2 = d_ad + d_bc;
    let mut best_pair = 0usize;
    let mut best_s = s0;
    if s1 < best_s {
        best_s = s1;
        best_pair = 1;
    }
    if s2 < best_s {
        best_s = s2;
        best_pair = 2;
    }
    let mut trio = [s0, s1, s2];
    trio.sort_by(|x, y| x.total_cmp(y));
    let gap = (trio[1] - trio[0]).max(0.0);
    if best_pair == 0 {
        return Ok(None);
    }
    let delta = s0 - best_s;
    if !delta.is_finite() || delta <= 0.0 {
        return Ok(None);
    }
    // Same mapping as ML quartet selection:
    // pairing-1 (AC|BD) => swap child b (c2) with sibling s
    // pairing-2 (AD|BC) => swap child a (c1) with sibling s
    let swap_child = if best_pair == 1 { e.c2 } else { e.c1 };
    Ok(Some((swap_child, delta, gap)))
}

fn compat_apply_spr_chain_up_me(
    tree: &mut MlTree,
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    sites: &[usize],
    site_rates: Option<&[f64]>,
    seed_edge: NniEdge,
    seed_swap_child: usize,
    max_steps: usize,
    delta_min: f64,
    pool: Option<&ThreadPool>,
) -> Result<usize, String> {
    if max_steps == 0 {
        return Ok(0);
    }
    if apply_nni_swap(tree, seed_edge.p, seed_edge.c, seed_swap_child).is_err() {
        return Ok(0);
    }
    let mut applied = 1usize;
    let moving = seed_swap_child;

    for _ in 1..max_steps {
        let Some(c) = tree.nodes[moving].parent else {
            break;
        };
        let Some(p) = tree.nodes[c].parent else {
            break;
        };
        let Some(edge) = nni_edge_for_pc(tree, p, c) else {
            break;
        };
        if edge.c1 != moving && edge.c2 != moving {
            break;
        }
        let cache = build_jc_site_cache_with_rates(tree, a, sites.to_vec(), pool, site_rates)?;
        if !cache.total_ll.is_finite() {
            break;
        }
        let up = build_jc_up_messages(tree, &cache, pool)?;
        let Some((swap_child, delta)) = compat_pick_quartet_swap_me(tree, &cache, &up, &edge)?
        else {
            break;
        };
        if swap_child != moving || delta < delta_min {
            break;
        }
        if apply_nni_swap(tree, p, c, moving).is_err() {
            break;
        }
        applied += 1;
    }
    Ok(applied)
}

fn optimize_nni_ml_compat(
    tree: &mut MlTree,
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    pool: Option<&ThreadPool>,
) -> Result<(usize, f64), String> {
    let n_taxa = a.shape()[0];
    let n_sites = a.shape()[1];
    if n_sites == 0 {
        return Ok((0, f64::NEG_INFINITY));
    }
    let cat_enabled = env_bool_or("JANUSX_ML_COMPAT_CAT", true);
    let cat_ncat = env_usize_or("JANUSX_ML_COMPAT_CAT_NCAT", 20)
        .max(2)
        .min(255);
    let cat_update_every_raw = env_usize_or("JANUSX_ML_COMPAT_CAT_UPDATE_EVERY", 1);
    let cat_dynamic_enabled = cat_enabled && cat_update_every_raw > 0;
    let cat_update_every = cat_update_every_raw.max(1);
    let cat_update_sites_default = if n_sites <= 4096 {
        n_sites
    } else {
        n_sites.min(2048)
    };
    let cat_update_sites = env_usize_or(
        "JANUSX_ML_COMPAT_CAT_UPDATE_SITES",
        cat_update_sites_default,
    )
    .max(32)
    .min(n_sites);
    let cat_post_tau = env_f64_or("JANUSX_ML_COMPAT_CAT_POST_TAU", 1.0).clamp(0.1, 10.0);
    let cat_keep_pmin = env_f64_or("JANUSX_ML_COMPAT_CAT_KEEP_PMIN", 0.55).clamp(0.0, 1.0);
    let cat_soft_rate = env_bool_or("JANUSX_ML_COMPAT_CAT_SOFT_RATE", false);
    let cat_prior_weight = env_f64_or("JANUSX_ML_COMPAT_CAT_PRIOR_WEIGHT", 0.5).clamp(0.0, 8.0);
    let cat_prior_alpha = env_f64_or("JANUSX_ML_COMPAT_CAT_PRIOR_ALPHA", 0.5).max(0.0);
    let mut cat_rates: Vec<f64> = Vec::new();
    let mut cat_prior: Vec<f64> = Vec::new();
    let mut site_cat_owned: Option<Vec<u8>> = None;
    let mut site_rates_owned: Option<Vec<f64>> = None;
    if cat_enabled {
        cat_rates = build_cat_rate_grid(cat_ncat);
        let init_rates = build_catlite_site_rates(a, cat_ncat);
        let (site_cat, site_rates) = assign_sites_to_cat_grid(&init_rates, &cat_rates);
        cat_prior = compat_cat_prior_from_assign(&site_cat, cat_rates.len(), cat_prior_alpha);
        site_cat_owned = Some(site_cat);
        site_rates_owned = Some(site_rates);
    }
    let mut site_rates = site_rates_owned.as_deref();

    let log2_n = if n_taxa > 1 {
        (n_taxa as f64).log2()
    } else {
        1.0
    };
    let default_rounds = (2.0 * log2_n.ceil()).max(2.0) as usize;
    let nni_rounds = env_usize_or("JANUSX_ML_COMPAT_NNI_ROUNDS", default_rounds).max(1);
    let me_rounds = env_usize_or("JANUSX_ML_COMPAT_ME_ROUNDS", 0);
    let min_improve = env_f64_or("JANUSX_ML_COMPAT_MIN_IMPROVE", 0.1).max(0.0);
    let local_blen_passes = env_usize_or("JANUSX_ML_COMPAT_LOCAL_BLEN_PASSES", 1);
    let local_blen_reuse = env_bool_or("JANUSX_ML_COMPAT_LOCAL_BLEN_REUSE", true);
    let quartet_opt_t = env_bool_or("JANUSX_ML_COMPAT_OPT_T", true);
    let quartet_eps = env_f64_or("JANUSX_ML_COMPAT_EPS", 1e-300).max(1e-300);
    let quartet_delta_min = env_f64_or("JANUSX_ML_COMPAT_QUARTET_DELTA_MIN", 0.01).max(0.0);
    let default_quartet_sites = n_sites.min(2048).max(256);
    let quartet_sites = env_usize_or("JANUSX_ML_COMPAT_QUARTET_SITES", default_quartet_sites)
        .max(32)
        .min(n_sites);
    let compat_seed = env_u64_or("JANUSX_ML_BOOTSTRAP_SEED", 20260321);
    // FastTree-like behavior: accept local-improving quartet swaps and avoid
    // per-swap global likelihood recomputation. Keep optional safety audits.
    let strict_recheck = env_bool_or("JANUSX_ML_COMPAT_STRICT_RECHECK", false);
    let audit_every = env_usize_or("JANUSX_ML_COMPAT_AUDIT_EVERY", 0);
    let audit_tol = env_f64_or("JANUSX_ML_COMPAT_AUDIT_TOL", 0.05).max(0.0);
    let round_drop_tol = env_f64_or("JANUSX_ML_COMPAT_ROUND_DROP_TOL", 0.0).max(0.0);
    let compat_batch = env_usize_or("JANUSX_ML_COMPAT_BATCH", 8).max(1);
    let round_passes = env_usize_or("JANUSX_ML_COMPAT_PASSES_PER_ROUND", 1).max(1);
    let default_edge_budget = if n_taxa > 512 { 512 } else { 0 };
    let compat_edge_budget = env_usize_or("JANUSX_ML_COMPAT_EDGE_BUDGET", default_edge_budget);
    let compat_fullscan_rounds = env_usize_or("JANUSX_ML_COMPAT_FULLSCAN_ROUNDS", 0);
    let me_delta_min = env_f64_or("JANUSX_ML_COMPAT_ME_DELTA_MIN", 1e-4).max(0.0);
    let me_batch = env_usize_or("JANUSX_ML_COMPAT_ME_BATCH", 4).max(1);
    let me_passes = env_usize_or("JANUSX_ML_COMPAT_ME_PASSES_PER_ROUND", 1).max(1);
    let me_local_blen_passes = env_usize_or("JANUSX_ML_COMPAT_ME_LOCAL_BLEN_PASSES", 0);
    let me_edge_budget = env_usize_or("JANUSX_ML_COMPAT_ME_EDGE_BUDGET", compat_edge_budget);
    let me_fullscan_rounds = env_usize_or("JANUSX_ML_COMPAT_ME_FULLSCAN_ROUNDS", 0);
    let default_me_sites = n_sites.min(768).max(128);
    let me_sites = env_usize_or("JANUSX_ML_COMPAT_ME_SITES", default_me_sites)
        .max(32)
        .min(n_sites);
    let spr_rounds = env_usize_or("JANUSX_ML_COMPAT_SPR_ROUNDS", 0);
    let spr_len = env_usize_or("JANUSX_ML_COMPAT_SPR_LEN", 2).max(1);
    let spr_chains_per_round = env_usize_or("JANUSX_ML_COMPAT_SPR_CHAINS_PER_ROUND", 2).max(1);
    let spr_delta_min = env_f64_or("JANUSX_ML_COMPAT_SPR_DELTA_MIN", 1e-4).max(0.0);
    let spr_edge_budget = env_usize_or("JANUSX_ML_COMPAT_SPR_EDGE_BUDGET", compat_edge_budget);
    let spr_fullscan_rounds = env_usize_or("JANUSX_ML_COMPAT_SPR_FULLSCAN_ROUNDS", 0);
    let default_spr_sites = n_sites.min(512).max(128);
    let spr_sites = env_usize_or("JANUSX_ML_COMPAT_SPR_SITES", default_spr_sites)
        .max(32)
        .min(n_sites);
    let spr_round_drop_tol = env_f64_or("JANUSX_ML_COMPAT_SPR_ROUND_DROP_TOL", 0.0).max(0.0);
    let spr_lowconf_only = env_bool_or("JANUSX_ML_COMPAT_SPR_LOWCONF_ONLY", true);
    let spr_lowconf_gap_q = env_f64_or("JANUSX_ML_COMPAT_SPR_LOWCONF_GAP_Q", 0.35).clamp(0.0, 1.0);
    let spr_lowconf_gap_cap = env_f64_or("JANUSX_ML_COMPAT_SPR_LOWCONF_GAP_CAP", 0.015).max(0.0);
    let spr_lowconf_delta_min = env_f64_or(
        "JANUSX_ML_COMPAT_SPR_LOWCONF_DELTA_MIN",
        (spr_delta_min * 4.0).max(5e-4),
    )
    .max(0.0);
    let spr_lowconf_fallback = env_bool_or("JANUSX_ML_COMPAT_SPR_LOWCONF_FALLBACK", false);
    let spr_lowconf_edge_frac =
        env_f64_or("JANUSX_ML_COMPAT_SPR_LOWCONF_EDGE_FRAC", 0.35).clamp(0.01, 1.0);
    let spr_lowconf_edge_min = env_usize_or("JANUSX_ML_COMPAT_SPR_LOWCONF_EDGE_MIN", 48).max(1);
    let edge_select_mode = resolve_compat_edge_select_mode();
    let mut me_edge_cursor = 0usize;
    let mut nni_edge_cursor = 0usize;
    let mut spr_edge_cursor = 0usize;

    if cat_dynamic_enabled {
        let init_sites = compat_cat_update_site_subset(
            n_sites,
            cat_update_sites,
            compat_seed ^ 0xA24BAED4963EE407u64,
        );
        if let (Some(site_cat), Some(site_rates_mut)) =
            (site_cat_owned.as_mut(), site_rates_owned.as_mut())
        {
            let prior_opt = if cat_prior.is_empty() {
                None
            } else {
                Some(cat_prior.as_slice())
            };
            let _ = compat_cat_reassign_subset(
                tree,
                a,
                &cat_rates,
                site_cat.as_mut_slice(),
                site_rates_mut.as_mut_slice(),
                &init_sites,
                prior_opt,
                cat_post_tau,
                cat_keep_pmin,
                cat_soft_rate,
                cat_prior_weight,
                pool,
            )?;
            cat_prior =
                compat_cat_prior_from_assign(site_cat.as_slice(), cat_rates.len(), cat_prior_alpha);
        }
        site_rates = site_rates_owned.as_deref();
    }

    let mut accepted = 0usize;
    if me_rounds > 0 {
        for me_round_idx in 0..me_rounds {
            if cat_dynamic_enabled && (me_round_idx % cat_update_every == 0) {
                let cat_sites = compat_cat_update_site_subset(
                    n_sites,
                    cat_update_sites,
                    compat_seed
                        ^ 0x9E3779B97F4A7C15u64
                        ^ ((me_round_idx as u64) << 31)
                        ^ (accepted as u64),
                );
                if let (Some(site_cat), Some(site_rates_mut)) =
                    (site_cat_owned.as_mut(), site_rates_owned.as_mut())
                {
                    let prior_opt = if cat_prior.is_empty() {
                        None
                    } else {
                        Some(cat_prior.as_slice())
                    };
                    let _ = compat_cat_reassign_subset(
                        tree,
                        a,
                        &cat_rates,
                        site_cat.as_mut_slice(),
                        site_rates_mut.as_mut_slice(),
                        &cat_sites,
                        prior_opt,
                        cat_post_tau,
                        cat_keep_pmin,
                        cat_soft_rate,
                        cat_prior_weight,
                        pool,
                    )?;
                    cat_prior = compat_cat_prior_from_assign(
                        site_cat.as_slice(),
                        cat_rates.len(),
                        cat_prior_alpha,
                    );
                }
                site_rates = site_rates_owned.as_deref();
            }
            let mut me_changes = 0usize;
            for me_pass_idx in 0..me_passes {
                let sampled_sites = if me_sites >= n_sites {
                    ml_build_site_indices(n_sites, 0)
                } else {
                    sample_site_indices_stratified(
                        n_sites,
                        me_sites,
                        compat_seed
                            ^ (0x9E3779B97F4A7C15u64)
                            ^ ((me_round_idx as u64) << 40)
                            ^ ((me_pass_idx as u64) << 24)
                            ^ (me_changes as u64),
                    )
                };
                let cache =
                    build_jc_site_cache_with_rates(tree, a, sampled_sites, pool, site_rates)?;
                if !cache.total_ll.is_finite() {
                    break;
                }
                let up = build_jc_up_messages(tree, &cache, pool)?;
                let edges_all = collect_nni_edges(tree);
                if edges_all.is_empty() {
                    break;
                }
                let me_budget_this_round = if me_round_idx < me_fullscan_rounds {
                    0
                } else {
                    me_edge_budget
                };
                let edges = select_nni_edges_compat(
                    tree,
                    &edges_all,
                    me_budget_this_round,
                    &mut me_edge_cursor,
                    edge_select_mode,
                );

                #[derive(Copy, Clone)]
                struct MeCand {
                    e: NniEdge,
                    swap_child: usize,
                    score: f64,
                }
                let cand_res: Vec<Result<Option<MeCand>, String>> = if edges.len() >= 64 {
                    if let Some(tp) = pool {
                        tp.install(|| {
                            edges
                                .par_iter()
                                .map(|&e| {
                                    let picked =
                                        compat_pick_quartet_swap_me(tree, &cache, &up, &e)?;
                                    Ok(match picked {
                                        Some((swap_child, delta)) if delta >= me_delta_min => {
                                            Some(MeCand {
                                                e,
                                                swap_child,
                                                score: delta,
                                            })
                                        }
                                        _ => None,
                                    })
                                })
                                .collect()
                        })
                    } else {
                        edges
                            .par_iter()
                            .map(|&e| {
                                let picked = compat_pick_quartet_swap_me(tree, &cache, &up, &e)?;
                                Ok(match picked {
                                    Some((swap_child, delta)) if delta >= me_delta_min => {
                                        Some(MeCand {
                                            e,
                                            swap_child,
                                            score: delta,
                                        })
                                    }
                                    _ => None,
                                })
                            })
                            .collect()
                    }
                } else {
                    edges
                        .iter()
                        .map(|&e| {
                            let picked = compat_pick_quartet_swap_me(tree, &cache, &up, &e)?;
                            Ok(match picked {
                                Some((swap_child, delta)) if delta >= me_delta_min => {
                                    Some(MeCand {
                                        e,
                                        swap_child,
                                        score: delta,
                                    })
                                }
                                _ => None,
                            })
                        })
                        .collect()
                };
                let mut cand: Vec<MeCand> = Vec::new();
                for r in cand_res {
                    if let Some(c) = r? {
                        cand.push(c);
                    }
                }
                if cand.is_empty() {
                    break;
                }
                cand.sort_by(|x, y| y.score.total_cmp(&x.score));
                let mut selected: Vec<MeCand> = Vec::with_capacity(me_batch);
                let mut used_nodes: HashSet<usize> = HashSet::with_capacity(me_batch * 6);
                for c in cand {
                    let touched = [c.e.p, c.e.c, c.e.s, c.e.c1, c.e.c2];
                    if touched.iter().any(|u| used_nodes.contains(u)) {
                        continue;
                    }
                    for u in touched {
                        used_nodes.insert(u);
                    }
                    selected.push(c);
                    if selected.len() >= me_batch {
                        break;
                    }
                }

                let mut changed_once = false;
                for c in selected {
                    let e = c.e;
                    if apply_nni_swap(tree, e.p, e.c, c.swap_child).is_err() {
                        continue;
                    }
                    if me_local_blen_passes > 0 {
                        let focus = nni_local_focus_nodes(
                            tree,
                            &NniMove {
                                p: e.p,
                                c: e.c,
                                s: e.s,
                                swap_child: c.swap_child,
                                quick_score: c.score,
                            },
                        );
                        if local_blen_reuse {
                            optimize_local_branch_lengths_jc69_reuse(
                                tree,
                                a,
                                &cache.sites,
                                &focus,
                                me_local_blen_passes,
                                pool,
                            )?;
                        } else {
                            optimize_local_branch_lengths_jc69(
                                tree,
                                a,
                                &cache.sites,
                                &focus,
                                me_local_blen_passes,
                                pool,
                            )?;
                        }
                    }
                    me_changes += 1;
                    changed_once = true;
                }
                if !changed_once {
                    break;
                }
            }
            if me_changes == 0 {
                break;
            }
        }
    }

    let full_sites = ml_build_site_indices(n_sites, 0);
    let mut cur_full = jc69_loglik_sites_with_pool_rates(tree, a, &full_sites, pool, site_rates)?;
    if !cur_full.is_finite() {
        return Ok((0, cur_full));
    }

    for round_idx in 0..nni_rounds {
        if cat_dynamic_enabled && (round_idx % cat_update_every == 0) {
            let cat_sites = compat_cat_update_site_subset(
                n_sites,
                cat_update_sites,
                compat_seed
                    ^ 0xD6E8FEB86659FD93u64
                    ^ ((round_idx as u64) << 33)
                    ^ (accepted as u64),
            );
            if let (Some(site_cat), Some(site_rates_mut)) =
                (site_cat_owned.as_mut(), site_rates_owned.as_mut())
            {
                let prior_opt = if cat_prior.is_empty() {
                    None
                } else {
                    Some(cat_prior.as_slice())
                };
                let _ = compat_cat_reassign_subset(
                    tree,
                    a,
                    &cat_rates,
                    site_cat.as_mut_slice(),
                    site_rates_mut.as_mut_slice(),
                    &cat_sites,
                    prior_opt,
                    cat_post_tau,
                    cat_keep_pmin,
                    cat_soft_rate,
                    cat_prior_weight,
                    pool,
                )?;
                cat_prior = compat_cat_prior_from_assign(
                    site_cat.as_slice(),
                    cat_rates.len(),
                    cat_prior_alpha,
                );
            }
            site_rates = site_rates_owned.as_deref();
        }
        let round_start_full = cur_full;
        let round_backup = tree.clone();
        let mut round_changes = 0usize;
        for pass_idx in 0..round_passes {
            let sampled_sites = if quartet_sites >= n_sites {
                ml_build_site_indices(n_sites, 0)
            } else {
                sample_site_indices_stratified(
                    n_sites,
                    quartet_sites,
                    compat_seed
                        ^ ((round_idx as u64) << 48)
                        ^ ((pass_idx as u64) << 32)
                        ^ (accepted as u64)
                        ^ (round_changes as u64),
                )
            };
            let cache = build_jc_site_cache_with_rates(tree, a, sampled_sites, pool, site_rates)?;
            if !cache.total_ll.is_finite() {
                break;
            }
            let up = build_jc_up_messages(tree, &cache, pool)?;
            let edges_all = collect_nni_edges(tree);
            if edges_all.is_empty() {
                break;
            }
            let compat_budget_this_round = if round_idx < compat_fullscan_rounds {
                0
            } else {
                compat_edge_budget
            };
            let edges = select_nni_edges_compat(
                tree,
                &edges_all,
                compat_budget_this_round,
                &mut nni_edge_cursor,
                edge_select_mode,
            );
            #[derive(Copy, Clone)]
            struct CompatCand {
                e: NniEdge,
                swap_child: usize,
                score: f64,
            }
            let cand_res: Vec<Result<Option<CompatCand>, String>> = if edges.len() >= 64 {
                if let Some(tp) = pool {
                    tp.install(|| {
                        edges
                            .par_iter()
                            .map(|&e| {
                                let picked = compat_pick_quartet_swap(
                                    tree,
                                    &cache,
                                    &up,
                                    &e,
                                    quartet_eps,
                                    quartet_opt_t,
                                )?;
                                Ok(match picked {
                                    Some((swap_child, delta)) if delta >= quartet_delta_min => {
                                        Some(CompatCand {
                                            e,
                                            swap_child,
                                            score: delta,
                                        })
                                    }
                                    _ => None,
                                })
                            })
                            .collect()
                    })
                } else {
                    edges
                        .par_iter()
                        .map(|&e| {
                            let picked = compat_pick_quartet_swap(
                                tree,
                                &cache,
                                &up,
                                &e,
                                quartet_eps,
                                quartet_opt_t,
                            )?;
                            Ok(match picked {
                                Some((swap_child, delta)) if delta >= quartet_delta_min => {
                                    Some(CompatCand {
                                        e,
                                        swap_child,
                                        score: delta,
                                    })
                                }
                                _ => None,
                            })
                        })
                        .collect()
                }
            } else {
                edges
                    .iter()
                    .map(|&e| {
                        let picked = compat_pick_quartet_swap(
                            tree,
                            &cache,
                            &up,
                            &e,
                            quartet_eps,
                            quartet_opt_t,
                        )?;
                        Ok(match picked {
                            Some((swap_child, delta)) if delta >= quartet_delta_min => {
                                Some(CompatCand {
                                    e,
                                    swap_child,
                                    score: delta,
                                })
                            }
                            _ => None,
                        })
                    })
                    .collect()
            };
            let mut cand: Vec<CompatCand> = Vec::new();
            for r in cand_res {
                if let Some(c) = r? {
                    cand.push(c);
                }
            }
            if cand.is_empty() {
                break;
            }
            cand.sort_by(|x, y| y.score.total_cmp(&x.score));
            let batch_limit = if strict_recheck { 1 } else { compat_batch };
            let mut selected: Vec<CompatCand> = Vec::with_capacity(batch_limit);
            let mut used_nodes: HashSet<usize> = HashSet::with_capacity(batch_limit * 6);
            for c in cand {
                let touched = [c.e.p, c.e.c, c.e.s, c.e.c1, c.e.c2];
                if touched.iter().any(|u| used_nodes.contains(u)) {
                    continue;
                }
                for u in touched {
                    used_nodes.insert(u);
                }
                selected.push(c);
                if selected.len() >= batch_limit {
                    break;
                }
            }

            let mut changed_once = false;
            for c in selected {
                let e = c.e;
                let backup_tree = tree.clone();
                if apply_nni_swap(tree, e.p, e.c, c.swap_child).is_err() {
                    *tree = backup_tree;
                    continue;
                }
                let focus = nni_local_focus_nodes(
                    tree,
                    &NniMove {
                        p: e.p,
                        c: e.c,
                        s: e.s,
                        swap_child: c.swap_child,
                        quick_score: c.score,
                    },
                );
                if local_blen_passes > 0 {
                    if local_blen_reuse {
                        optimize_local_branch_lengths_jc69_reuse(
                            tree,
                            a,
                            &cache.sites,
                            &focus,
                            local_blen_passes,
                            pool,
                        )?;
                    } else {
                        optimize_local_branch_lengths_jc69(
                            tree,
                            a,
                            &cache.sites,
                            &focus,
                            local_blen_passes,
                            pool,
                        )?;
                    }
                }
                if strict_recheck {
                    let new_full =
                        jc69_loglik_sites_with_pool_rates(tree, a, &full_sites, pool, site_rates)?;
                    if !new_full.is_finite() || new_full <= cur_full + min_improve {
                        *tree = backup_tree;
                        continue;
                    }
                    cur_full = new_full;
                } else if audit_every > 0 && ((accepted + 1) % audit_every == 0) {
                    let audit_full =
                        jc69_loglik_sites_with_pool_rates(tree, a, &full_sites, pool, site_rates)?;
                    if !audit_full.is_finite() || audit_full + audit_tol < cur_full {
                        *tree = backup_tree;
                        continue;
                    }
                    cur_full = audit_full;
                }
                accepted += 1;
                round_changes += 1;
                changed_once = true;
            }
            if !changed_once {
                break;
            }
        }
        if round_changes == 0 {
            break;
        }
        // One global refresh per round (instead of per accepted move) keeps
        // convergence checks stable while preserving FastTree-like speed.
        let round_full = jc69_loglik_sites_with_pool_rates(tree, a, &full_sites, pool, site_rates)?;
        if !round_full.is_finite() {
            *tree = round_backup;
            cur_full = round_start_full;
            break;
        }
        if round_full + round_drop_tol < round_start_full {
            *tree = round_backup;
            cur_full = round_start_full;
            break;
        }
        cur_full = round_full;
        if cur_full <= round_start_full + min_improve {
            break;
        }
    }

    if spr_rounds > 0 {
        let mut spr_full = cur_full;
        if !spr_full.is_finite() {
            return Ok((accepted, spr_full));
        }
        for spr_round_idx in 0..spr_rounds {
            if cat_dynamic_enabled && (spr_round_idx % cat_update_every == 0) {
                let cat_sites = compat_cat_update_site_subset(
                    n_sites,
                    cat_update_sites,
                    compat_seed
                        ^ 0x94D049BB133111EBu64
                        ^ ((spr_round_idx as u64) << 29)
                        ^ (accepted as u64),
                );
                if let (Some(site_cat), Some(site_rates_mut)) =
                    (site_cat_owned.as_mut(), site_rates_owned.as_mut())
                {
                    let prior_opt = if cat_prior.is_empty() {
                        None
                    } else {
                        Some(cat_prior.as_slice())
                    };
                    let _ = compat_cat_reassign_subset(
                        tree,
                        a,
                        &cat_rates,
                        site_cat.as_mut_slice(),
                        site_rates_mut.as_mut_slice(),
                        &cat_sites,
                        prior_opt,
                        cat_post_tau,
                        cat_keep_pmin,
                        cat_soft_rate,
                        cat_prior_weight,
                        pool,
                    )?;
                    cat_prior = compat_cat_prior_from_assign(
                        site_cat.as_slice(),
                        cat_rates.len(),
                        cat_prior_alpha,
                    );
                }
                site_rates = site_rates_owned.as_deref();
            }
            let round_backup = tree.clone();
            let round_start = spr_full;
            let mut round_moves = 0usize;
            for chain_idx in 0..spr_chains_per_round {
                let sampled_sites = if spr_sites >= n_sites {
                    ml_build_site_indices(n_sites, 0)
                } else {
                    sample_site_indices_stratified(
                        n_sites,
                        spr_sites,
                        compat_seed
                            ^ (0xC2B2AE3D27D4EB4Fu64)
                            ^ ((spr_round_idx as u64) << 44)
                            ^ ((chain_idx as u64) << 28)
                            ^ (round_moves as u64),
                    )
                };
                let cache = build_jc_site_cache_with_rates(
                    tree,
                    a,
                    sampled_sites.clone(),
                    pool,
                    site_rates,
                )?;
                if !cache.total_ll.is_finite() {
                    break;
                }
                let up = build_jc_up_messages(tree, &cache, pool)?;
                let edges_all = collect_nni_edges(tree);
                if edges_all.is_empty() {
                    break;
                }
                let spr_budget_this_round = if spr_round_idx < spr_fullscan_rounds {
                    0
                } else {
                    spr_edge_budget
                };
                let edges = select_nni_edges_compat(
                    tree,
                    &edges_all,
                    spr_budget_this_round,
                    &mut spr_edge_cursor,
                    edge_select_mode,
                );
                let spr_edges: &[NniEdge] = if spr_lowconf_only && !edges.is_empty() {
                    let keep_n = (((edges.len() as f64) * spr_lowconf_edge_frac).ceil() as usize)
                        .max(spr_lowconf_edge_min)
                        .min(edges.len());
                    &edges[..keep_n]
                } else {
                    &edges
                };

                #[derive(Copy, Clone)]
                struct SprSeed {
                    e: NniEdge,
                    swap_child: usize,
                    score: f64,
                    gap: f64,
                }
                let seeds_res: Vec<Result<Option<SprSeed>, String>> = if edges.len() >= 64 {
                    if let Some(tp) = pool {
                        tp.install(|| {
                            spr_edges
                                .par_iter()
                                .map(|&e| {
                                    let picked = compat_pick_quartet_swap_me_with_gap(
                                        tree, &cache, &up, &e,
                                    )?;
                                    Ok(match picked {
                                        Some((swap_child, delta, gap))
                                            if delta >= spr_delta_min =>
                                        {
                                            Some(SprSeed {
                                                e,
                                                swap_child,
                                                score: delta,
                                                gap,
                                            })
                                        }
                                        _ => None,
                                    })
                                })
                                .collect()
                        })
                    } else {
                        spr_edges
                            .par_iter()
                            .map(|&e| {
                                let picked =
                                    compat_pick_quartet_swap_me_with_gap(tree, &cache, &up, &e)?;
                                Ok(match picked {
                                    Some((swap_child, delta, gap)) if delta >= spr_delta_min => {
                                        Some(SprSeed {
                                            e,
                                            swap_child,
                                            score: delta,
                                            gap,
                                        })
                                    }
                                    _ => None,
                                })
                            })
                            .collect()
                    }
                } else {
                    spr_edges
                        .iter()
                        .map(|&e| {
                            let picked =
                                compat_pick_quartet_swap_me_with_gap(tree, &cache, &up, &e)?;
                            Ok(match picked {
                                Some((swap_child, delta, gap)) if delta >= spr_delta_min => {
                                    Some(SprSeed {
                                        e,
                                        swap_child,
                                        score: delta,
                                        gap,
                                    })
                                }
                                _ => None,
                            })
                        })
                        .collect()
                };
                let mut seeds: Vec<SprSeed> = Vec::new();
                for r in seeds_res {
                    if let Some(s) = r? {
                        seeds.push(s);
                    }
                }
                if seeds.is_empty() {
                    break;
                }
                seeds.sort_by(|x, y| y.score.total_cmp(&x.score));
                let best = if spr_lowconf_only {
                    let mut gaps: Vec<f64> = seeds
                        .iter()
                        .map(|s| s.gap)
                        .filter(|g| g.is_finite() && *g >= 0.0)
                        .collect();
                    gaps.sort_by(|x, y| x.total_cmp(y));
                    let gap_q = if gaps.is_empty() {
                        0.0
                    } else {
                        let idx = ((gaps.len().saturating_sub(1) as f64) * spr_lowconf_gap_q)
                            .round() as usize;
                        gaps[idx.min(gaps.len() - 1)]
                    };
                    let gap_cut = if spr_lowconf_gap_cap > 0.0 {
                        gap_q.min(spr_lowconf_gap_cap)
                    } else {
                        gap_q
                    };
                    let mut lowconf = Vec::<SprSeed>::new();
                    for s in &seeds {
                        if s.gap <= gap_cut + 1e-12 && s.score >= spr_lowconf_delta_min {
                            lowconf.push(*s);
                        }
                    }
                    if lowconf.is_empty() {
                        if spr_lowconf_fallback {
                            seeds[0]
                        } else {
                            continue;
                        }
                    } else {
                        lowconf.sort_by(|x, y| y.score.total_cmp(&x.score));
                        lowconf[0]
                    }
                } else {
                    seeds[0]
                };
                let backup_chain = tree.clone();
                let applied = compat_apply_spr_chain_up_me(
                    tree,
                    a,
                    &sampled_sites,
                    site_rates,
                    best.e,
                    best.swap_child,
                    spr_len,
                    spr_delta_min,
                    pool,
                )?;
                if applied == 0 {
                    *tree = backup_chain;
                    continue;
                }
                round_moves += applied;
            }
            if round_moves == 0 {
                break;
            }
            let round_full =
                jc69_loglik_sites_with_pool_rates(tree, a, &full_sites, pool, site_rates)?;
            if !round_full.is_finite() || round_full + spr_round_drop_tol < round_start {
                *tree = round_backup;
                break;
            }
            spr_full = round_full;
        }
        cur_full = spr_full;
    }
    Ok((accepted, cur_full))
}

fn infer_ml_tree_jc69(
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    sample_ids: &[String],
    min_ov: u64,
    pool: Option<&ThreadPool>,
    mode: &str,
) -> Result<MlTree, String> {
    let n_taxa = sample_ids.len();
    // Empirical compat trade-off:
    // - small n: JC-initialized NJ often tracks FastTree better
    // - large n: mismatch-initialized NJ tends to be more stable for our current
    //   downstream compat optimizer.
    let compat_jc_init_small_taxa = env_usize_or("JANUSX_ML_COMPAT_JC_INIT_SMALL_TAXA", 256).max(2);
    let use_jc_nj_init_default = n_taxa <= compat_jc_init_small_taxa;
    let use_jc_nj_init = mode == "compat"
        && env_bool_or("JANUSX_ML_COMPAT_JC_INIT", use_jc_nj_init_default);
    let (nodes, root) = build_nj_tree_exact(a, sample_ids, min_ov, use_jc_nj_init, pool)?;
    let mut tree = convert_to_ml_tree(&nodes, root, n_taxa)?;
    if n_taxa >= 4 {
        match mode {
            "compat" => {
                let _ = optimize_nni_ml_compat(&mut tree, a, pool)?;
            }
            _ => {
                let _ = optimize_nni_ml_jc69(&mut tree, a, mode, pool)?;
            }
        }
    }
    Ok(tree)
}

fn ml_leafsets(tree: &MlTree, n_leaves: usize) -> Result<Vec<Vec<u64>>, String> {
    let n_words = (n_leaves + 63) / 64;
    let mut sets = vec![vec![0u64; n_words]; tree.nodes.len()];
    let post = ml_postorder(tree)?;
    for &u in &post {
        if let Some((l, r)) = tree.children(u) {
            for w in 0..n_words {
                sets[u][w] = sets[l][w] | sets[r][w];
            }
        } else {
            let leaf_ix = tree.nodes[u]
                .leaf_ix
                .ok_or_else(|| "leaf node missing index in split extraction".to_string())?;
            if leaf_ix >= n_leaves {
                return Err("leaf index out of range in split extraction".to_string());
            }
            let w = leaf_ix >> 6;
            let b = leaf_ix & 63;
            sets[u][w] |= 1u64 << b;
        }
    }
    Ok(sets)
}

fn canonical_split_key(bits: &[u64], n_leaves: usize) -> Option<Vec<u64>> {
    let n_words = bits.len();
    let mut cnt = 0usize;
    for &w in bits {
        cnt += w.count_ones() as usize;
    }
    if cnt == 0 || cnt == n_leaves {
        return None;
    }
    let comp_cnt = n_leaves - cnt;
    let mut comp = vec![0u64; n_words];
    for i in 0..n_words {
        comp[i] = !bits[i];
    }
    let rem = n_leaves & 63;
    if rem != 0 {
        let mask = (1u64 << rem) - 1;
        comp[n_words - 1] &= mask;
    }
    let choose_comp = if comp_cnt < cnt {
        true
    } else if comp_cnt > cnt {
        false
    } else {
        let mut less = false;
        for i in 0..n_words {
            if comp[i] < bits[i] {
                less = true;
                break;
            }
            if comp[i] > bits[i] {
                break;
            }
        }
        less
    };
    if choose_comp {
        Some(comp)
    } else {
        Some(bits.to_vec())
    }
}

fn ml_node_split_keys(tree: &MlTree, n_leaves: usize) -> Result<Vec<Option<Vec<u64>>>, String> {
    let sets = ml_leafsets(tree, n_leaves)?;
    let mut out = vec![None; tree.nodes.len()];
    for u in 0..tree.nodes.len() {
        if tree.nodes[u].parent.is_some() && tree.is_internal(u) {
            out[u] = canonical_split_key(&sets[u], n_leaves);
        }
    }
    Ok(out)
}

#[inline]
fn jc_quartet_topology_like(
    xa: &[f64; 4],
    xb: &[f64; 4],
    ya: &[f64; 4],
    yb: &[f64; 4],
    t_xy: f64,
) -> f64 {
    let tt = if t_xy.is_finite() {
        t_xy.max(1e-8).min(10.0)
    } else {
        1e-8
    };
    let e = (-4.0 * tt / 3.0).exp();
    let same = 0.25 + 0.75 * e;
    let diff = 0.25 - 0.25 * e;
    let k = same - diff;

    let mut px = [0.0f64; 4];
    let mut py = [0.0f64; 4];
    for st in 0..4 {
        px[st] = xa[st] * xb[st];
        py[st] = ya[st] * yb[st];
    }
    let sum_py = py[0] + py[1] + py[2] + py[3];

    let mut like = 0.0f64;
    for sx in 0..4 {
        // Sum_y P(sx -> sy) * py[sy]
        let to_y = diff * sum_py + k * py[sx];
        like += 0.25 * px[sx] * to_y;
    }
    like
}

fn build_resample_index_matrix(
    n_sites: usize,
    niter: usize,
    draws: usize,
    seed: u64,
) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::<usize>::with_capacity(niter.saturating_mul(draws));
    for _ in 0..niter.saturating_mul(draws) {
        out.push(rng.random_range(0..n_sites));
    }
    out
}

fn build_resample_index_matrix_weighted(
    n_sites: usize,
    niter: usize,
    draws: usize,
    seed: u64,
    weights: Option<&[f64]>,
) -> Vec<usize> {
    if weights.is_none() {
        return build_resample_index_matrix(n_sites, niter, draws, seed);
    }
    let w = weights.unwrap();
    if w.len() != n_sites {
        return build_resample_index_matrix(n_sites, niter, draws, seed);
    }

    let mut cdf = Vec::<f64>::with_capacity(n_sites);
    let mut acc = 0.0f64;
    for &x in w {
        let v = if x.is_finite() && x > 0.0 { x } else { 0.0 };
        acc += v;
        cdf.push(acc);
    }
    if !acc.is_finite() || acc <= 0.0 {
        return build_resample_index_matrix(n_sites, niter, draws, seed);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::<usize>::with_capacity(niter.saturating_mul(draws));
    for _ in 0..niter.saturating_mul(draws) {
        let x = rng.random_range(0.0..acc);
        let mut idx = cdf.partition_point(|&v| v <= x);
        if idx >= n_sites {
            idx = n_sites - 1;
        }
        out.push(idx);
    }
    out
}

fn sample_site_indices_stratified(n_sites: usize, budget: usize, seed: u64) -> Vec<usize> {
    if budget >= n_sites {
        return (0..n_sites).collect();
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::<usize>::with_capacity(budget);
    for i in 0..budget {
        let lo = i * n_sites / budget;
        let hi_excl = ((i + 1) * n_sites / budget).max(lo + 1);
        let idx = if hi_excl > lo + 1 {
            rng.random_range(lo..hi_excl)
        } else {
            lo
        };
        out.push(idx.min(n_sites - 1));
    }
    out.sort_unstable();
    out.dedup();
    if out.len() < budget {
        let extras = ml_build_site_indices(n_sites, budget);
        for idx in extras {
            if out.binary_search(&idx).is_err() {
                out.push(idx);
                if out.len() >= budget {
                    break;
                }
            }
        }
        out.sort_unstable();
    }
    out
}

fn site_informativeness_weights(
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    sites: &[usize],
) -> Vec<f64> {
    let n_taxa = a.shape()[0];
    let mut out = Vec::<f64>::with_capacity(sites.len());
    for &k in sites {
        let mut c = [0usize; 4];
        let mut eff = 0usize;
        for i in 0..n_taxa {
            match ascii_upper(a[[i, k]]) {
                b'A' => {
                    c[0] += 1;
                    eff += 1;
                }
                b'C' => {
                    c[1] += 1;
                    eff += 1;
                }
                b'G' => {
                    c[2] += 1;
                    eff += 1;
                }
                b'T' | b'U' => {
                    c[3] += 1;
                    eff += 1;
                }
                _ => {}
            }
        }
        if eff < 2 {
            out.push(1e-8);
            continue;
        }
        let eff_f = eff as f64;
        let mut h = 1.0f64;
        let mut ge2 = 0usize;
        for &v in &c {
            if v > 0 {
                let p = (v as f64) / eff_f;
                h -= p * p;
            }
            if v >= 2 {
                ge2 += 1;
            }
        }
        let coverage = eff_f / (n_taxa as f64).max(1.0);
        let informative_boost = if ge2 >= 2 { 1.25 } else { 1.0 };
        let score = (h.max(0.0) * coverage.max(0.05) * informative_boost).max(1e-8);
        out.push(score);
    }
    out
}

fn temper_site_weights(mut w: Vec<f64>, power: f64, uniform_mix: f64) -> Vec<f64> {
    if w.is_empty() {
        return w;
    }
    let p = if power.is_finite() {
        power.clamp(0.1, 2.0)
    } else {
        1.0
    };
    for x in w.iter_mut() {
        *x = x.max(1e-12).powf(p);
    }
    let mix = if uniform_mix.is_finite() {
        uniform_mix.clamp(0.0, 1.0)
    } else {
        0.0
    };
    if mix > 0.0 {
        let mean = (w.iter().sum::<f64>() / (w.len() as f64)).max(1e-12);
        for x in w.iter_mut() {
            *x = (1.0 - mix) * *x + mix * mean;
        }
    }
    w
}

fn winsorize_in_place(values: &mut [f64], low_q: f64, high_q: f64) {
    let n = values.len();
    if n < 8 {
        return;
    }
    let lq = if low_q.is_finite() {
        low_q.clamp(0.0, 0.49)
    } else {
        0.0
    };
    let hq = if high_q.is_finite() {
        high_q.clamp(0.51, 1.0)
    } else {
        1.0
    };
    if lq <= 0.0 && hq >= 1.0 {
        return;
    }

    let mut buf = values.to_vec();
    let max_idx = n - 1;
    let mut lo_idx = ((lq * (max_idx as f64)).floor() as usize).min(max_idx);
    let mut hi_idx = ((hq * (max_idx as f64)).ceil() as usize).min(max_idx);
    if lo_idx > hi_idx {
        std::mem::swap(&mut lo_idx, &mut hi_idx);
    }
    let (_, lo_ref, _) = buf.select_nth_unstable_by(lo_idx, |a, b| a.total_cmp(b));
    let lo = *lo_ref;
    let (_, hi_ref, _) = buf.select_nth_unstable_by(hi_idx, |a, b| a.total_cmp(b));
    let hi = *hi_ref;
    for x in values.iter_mut() {
        if *x < lo {
            *x = lo;
        } else if *x > hi {
            *x = hi;
        }
    }
}

#[inline]
fn quartet_like_with_pairing(
    ma: &[f64; 4],
    mb: &[f64; 4],
    mc: &[f64; 4],
    md: &[f64; 4],
    t_mid: f64,
    pairing: usize,
    eps: f64,
) -> f64 {
    let v = match pairing {
        0 => jc_quartet_topology_like(ma, mb, mc, md, t_mid),
        1 => jc_quartet_topology_like(ma, mc, mb, md, t_mid),
        _ => jc_quartet_topology_like(ma, md, mb, mc, t_mid),
    };
    v.max(eps)
}

fn quartet_total_loglik(
    msg_a: &[[f64; 4]],
    msg_b: &[[f64; 4]],
    msg_c: &[[f64; 4]],
    msg_d: &[[f64; 4]],
    pairing: usize,
    t_mid: f64,
    site_rates: Option<&[f64]>,
    eps: f64,
) -> f64 {
    let mut ll = 0.0f64;
    for sidx in 0..msg_a.len() {
        let rate = site_rates
            .and_then(|v| v.get(sidx))
            .copied()
            .filter(|r| r.is_finite() && *r > 0.0)
            .unwrap_or(1.0);
        ll += quartet_like_with_pairing(
            &msg_a[sidx],
            &msg_b[sidx],
            &msg_c[sidx],
            &msg_d[sidx],
            t_mid * rate,
            pairing,
            eps,
        )
        .ln();
    }
    ll
}

fn optimize_quartet_middle_t(
    msg_a: &[[f64; 4]],
    msg_b: &[[f64; 4]],
    msg_c: &[[f64; 4]],
    msg_d: &[[f64; 4]],
    pairing: usize,
    t_init: f64,
    site_rates: Option<&[f64]>,
    eps: f64,
) -> f64 {
    let mut best_t = if t_init.is_finite() {
        t_init.max(1e-8).min(10.0)
    } else {
        1e-3
    };
    let mut best_ll = f64::NEG_INFINITY;

    let mut seeds = vec![
        best_t * 0.25,
        best_t * 0.5,
        best_t * 0.75,
        best_t,
        best_t * 1.25,
        best_t * 1.5,
        best_t * 2.0,
        best_t + 0.01,
        best_t + 0.05,
        best_t - 0.01,
        best_t - 0.05,
    ];
    seeds.push(1e-6);
    seeds.push(1e-4);
    seeds.push(1e-2);
    for &t in &seeds {
        let tc = t.max(1e-8).min(10.0);
        let ll = quartet_total_loglik(msg_a, msg_b, msg_c, msg_d, pairing, tc, site_rates, eps);
        if ll > best_ll {
            best_ll = ll;
            best_t = tc;
        }
    }

    let mut step = (best_t * 0.35).max(0.01);
    for _ in 0..2 {
        for d in [-1.0f64, -0.5, 0.0, 0.5, 1.0] {
            let t = (best_t + d * step).max(1e-8).min(10.0);
            let ll = quartet_total_loglik(msg_a, msg_b, msg_c, msg_d, pairing, t, site_rates, eps);
            if ll > best_ll {
                best_ll = ll;
                best_t = t;
            }
        }
        step *= 0.5;
    }
    best_t
}

fn shlike_support_on_cache(
    tree: &MlTree,
    cache: &JcSiteCache,
    up: &[f32],
    targets: &[usize],
    niter: usize,
    draws: usize,
    seed: u64,
    eps: f64,
    site_weights: Option<&[f64]>,
    optimize_t: bool,
    pool: Option<&ThreadPool>,
) -> Result<Vec<(usize, f64)>, String> {
    let n_nodes = tree.nodes.len();
    let n_cache_sites = cache.sites.len();
    let adaptive_enabled = env_bool_or("JANUSX_ML_SHLIKE_ADAPTIVE", true) && niter >= 32;
    let adaptive_min_mult = env_f64_or("JANUSX_ML_SHLIKE_ADAPTIVE_MIN_MULT", 0.5).clamp(0.05, 1.0);
    let adaptive_max_mult = env_f64_or("JANUSX_ML_SHLIKE_ADAPTIVE_MAX_MULT", 1.35).clamp(1.0, 4.0);
    let pilot_default = (niter / 6).max(12);
    let pilot_reps = if adaptive_enabled {
        env_usize_or("JANUSX_ML_SHLIKE_ADAPTIVE_PILOT", pilot_default)
            .max(8)
            .min(niter)
    } else {
        niter
    };
    let adaptive_min_reps = env_usize_or("JANUSX_ML_SHLIKE_ADAPTIVE_MIN_REPS", (niter / 4).max(8));
    let adaptive_margin_k = env_f64_or("JANUSX_ML_SHLIKE_ADAPTIVE_MARGIN_K", 8.0).max(0.0);
    let adaptive_blend = env_f64_or("JANUSX_ML_SHLIKE_ADAPTIVE_BLEND", 0.65).clamp(0.0, 1.0);
    let winsor_enabled = env_bool_or("JANUSX_ML_SHLIKE_WINSOR", true);
    let winsor_continuous = env_bool_or("JANUSX_ML_SHLIKE_WINSOR_CONTINUOUS", false);
    let winsor_lowconf_only = env_bool_or("JANUSX_ML_SHLIKE_WINSOR_LOWCONF_ONLY", false);
    let winsor_low = env_f64_or("JANUSX_ML_SHLIKE_WINSOR_LOW", 0.01);
    let winsor_high = env_f64_or("JANUSX_ML_SHLIKE_WINSOR_HIGH", 0.99);
    let winsor_margin_lo = env_f64_or("JANUSX_ML_SHLIKE_WINSOR_MARGIN_LO", 0.005).max(0.0);
    let winsor_margin_hi =
        env_f64_or("JANUSX_ML_SHLIKE_WINSOR_MARGIN_HI", 0.05).max(winsor_margin_lo + 1e-12);
    let winsor_gamma = env_f64_or("JANUSX_ML_SHLIKE_WINSOR_GAMMA", 1.0).clamp(0.25, 4.0);
    let winsor_margin_max = env_f64_or("JANUSX_ML_SHLIKE_WINSOR_MARGIN_MAX", 0.005).max(0.0);
    let shlike_fasttree_formula = env_bool_or("JANUSX_ML_SHLIKE_FASTTREE_FORMULA", false);
    let max_local_reps = if adaptive_enabled {
        (((niter as f64) * adaptive_max_mult).round() as usize)
            .max(pilot_reps)
            .max(1)
    } else {
        niter.max(1)
    };
    let sample_ix = build_resample_index_matrix_weighted(
        n_cache_sites,
        max_local_reps,
        draws,
        seed,
        site_weights,
    );

    let eval_one = |&c: &usize| -> Result<(usize, f64), String> {
        let p = tree.nodes[c]
            .parent
            .ok_or_else(|| "SH-like support target missing parent".to_string())?;
        let (a_node, b_node) = tree
            .children(c)
            .ok_or_else(|| "SH-like support target must be internal".to_string())?;
        let (pl, pr) = tree
            .children(p)
            .ok_or_else(|| "SH-like support parent must be internal".to_string())?;
        let c_node = if pl == c {
            pr
        } else if pr == c {
            pl
        } else {
            return Err("SH-like support found inconsistent parent-child linkage".to_string());
        };

        let t_pc = tree.nodes[c].blen_to_parent;
        let t_a = tree.nodes[a_node].blen_to_parent;
        let t_b = tree.nodes[b_node].blen_to_parent;
        let t_c = tree.nodes[c_node].blen_to_parent;

        let mut msg_a_v = vec![[0.0f64; 4]; n_cache_sites];
        let mut msg_b_v = vec![[0.0f64; 4]; n_cache_sites];
        let mut msg_c_v = vec![[0.0f64; 4]; n_cache_sites];
        let mut msg_d_v = vec![[0.0f64; 4]; n_cache_sites];
        let mut ll0 = vec![0.0f64; n_cache_sites];
        let mut d1 = vec![0.0f64; n_cache_sites];
        let mut d2 = vec![0.0f64; n_cache_sites];
        for sidx in 0..n_cache_sites {
            let rate = cache_site_rate(cache, sidx);
            let clv_a = cache_get_clv4(cache, sidx, a_node);
            let clv_b = cache_get_clv4(cache, sidx, b_node);
            let clv_c = cache_get_clv4(cache, sidx, c_node);
            let mut msg_a = jc_child_message(&clv_a, t_a * rate);
            let mut msg_b = jc_child_message(&clv_b, t_b * rate);
            let mut msg_c = jc_child_message(&clv_c, t_c * rate);
            let mut msg_d = up_get4(up, n_nodes, sidx, p);
            normalize_vec4_inplace(&mut msg_a);
            normalize_vec4_inplace(&mut msg_b);
            normalize_vec4_inplace(&mut msg_c);
            normalize_vec4_inplace(&mut msg_d);
            msg_a_v[sidx] = msg_a;
            msg_b_v[sidx] = msg_b;
            msg_c_v[sidx] = msg_c;
            msg_d_v[sidx] = msg_d;
        }

        let (t0, t1, t2) = if optimize_t && n_cache_sites >= 16 {
            (
                optimize_quartet_middle_t(
                    &msg_a_v,
                    &msg_b_v,
                    &msg_c_v,
                    &msg_d_v,
                    0,
                    t_pc,
                    Some(cache.site_rates.as_slice()),
                    eps,
                ),
                optimize_quartet_middle_t(
                    &msg_a_v,
                    &msg_b_v,
                    &msg_c_v,
                    &msg_d_v,
                    1,
                    t_pc,
                    Some(cache.site_rates.as_slice()),
                    eps,
                ),
                optimize_quartet_middle_t(
                    &msg_a_v,
                    &msg_b_v,
                    &msg_c_v,
                    &msg_d_v,
                    2,
                    t_pc,
                    Some(cache.site_rates.as_slice()),
                    eps,
                ),
            )
        } else {
            (t_pc, t_pc, t_pc)
        };

        for sidx in 0..n_cache_sites {
            let rate = cache_site_rate(cache, sidx);
            let l0 = quartet_like_with_pairing(
                &msg_a_v[sidx],
                &msg_b_v[sidx],
                &msg_c_v[sidx],
                &msg_d_v[sidx],
                t0 * rate,
                0,
                eps,
            );
            let l1 = quartet_like_with_pairing(
                &msg_a_v[sidx],
                &msg_b_v[sidx],
                &msg_c_v[sidx],
                &msg_d_v[sidx],
                t1 * rate,
                1,
                eps,
            );
            let l2 = quartet_like_with_pairing(
                &msg_a_v[sidx],
                &msg_b_v[sidx],
                &msg_c_v[sidx],
                &msg_d_v[sidx],
                t2 * rate,
                2,
                eps,
            );
            let ll0_s = l0.ln();
            ll0[sidx] = ll0_s;
            d1[sidx] = ll0_s - l1.ln();
            d2[sidx] = ll0_s - l2.ln();
        }
        let run_reps = |ll0v: &[f64],
                        d1v: &[f64],
                        d2v: &[f64],
                        rep_from: usize,
                        rep_to: usize,
                        wins_ref: &mut usize| {
            let (loglk0, loglk1, loglk2, delta) = if shlike_fasttree_formula {
                let lk0 = ll0v.iter().sum::<f64>();
                let lk1 = ll0v.iter().zip(d1v.iter()).map(|(a, d)| a - d).sum::<f64>();
                let lk2 = ll0v.iter().zip(d2v.iter()).map(|(a, d)| a - d).sum::<f64>();
                let dlt = (lk0 - lk1).min(lk0 - lk2);
                (lk0, lk1, lk2, dlt)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };
            for rep in rep_from..rep_to {
                let base = rep * draws;
                if shlike_fasttree_formula {
                    // FastTree-like SH support decision based on resampled
                    // 3-topology log-likelihood differences.
                    let mut r0 = -loglk0;
                    let mut r1 = -loglk1;
                    let mut r2 = -loglk2;
                    for &k in &sample_ix[base..base + draws] {
                        let l0 = ll0v[k];
                        r0 += l0;
                        r1 += l0 - d1v[k];
                        r2 += l0 - d2v[k];
                    }
                    let (best, o1, o2) = if r0 >= r1 && r0 >= r2 {
                        (r0, r1, r2)
                    } else if r1 >= r0 && r1 >= r2 {
                        (r1, r0, r2)
                    } else {
                        (r2, r0, r1)
                    };
                    let res_delta = (best - o1).min(best - o2);
                    if res_delta < delta {
                        *wins_ref += 1;
                    }
                } else {
                    // Legacy local-bootstrap style: count when topology-0
                    // remains better than both alternatives in the replicate.
                    let mut s1 = 0.0f64;
                    let mut s2 = 0.0f64;
                    for &k in &sample_ix[base..base + draws] {
                        s1 += d1v[k];
                        s2 += d2v[k];
                    }
                    if s1 >= 0.0 && s2 >= 0.0 {
                        *wins_ref += 1;
                    }
                }
            }
        };

        // Optional robustification: only winsorize uncertain branches.
        // We use a deterministic uncertainty proxy to avoid introducing
        // extra stochastic jitter between different niter settings.
        let d1_mu_raw = d1.iter().sum::<f64>() / (n_cache_sites as f64).max(1.0);
        let d2_mu_raw = d2.iter().sum::<f64>() / (n_cache_sites as f64).max(1.0);
        let mean_margin_raw = 0.5 * (d1_mu_raw.abs() + d2_mu_raw.abs());
        let mut apply_winsor = false;
        let mut low_q_eff = 0.0f64;
        let mut high_q_eff = 1.0f64;
        if winsor_enabled {
            if winsor_continuous {
                // Continuous-strength winsorization:
                // high-confidence branch -> strength~0 (almost no clipping),
                // low-confidence branch -> strength~1 (full clipping).
                let u_margin = ((winsor_margin_hi - mean_margin_raw)
                    / (winsor_margin_hi - winsor_margin_lo))
                    .clamp(0.0, 1.0);
                let strength = u_margin.powf(winsor_gamma);
                low_q_eff = (winsor_low * strength).clamp(0.0, 0.49);
                high_q_eff = (1.0 - (1.0 - winsor_high) * strength).clamp(0.51, 1.0);
                apply_winsor = low_q_eff > 1e-12 || high_q_eff < (1.0 - 1e-12);
            } else {
                // Legacy hard-gate behavior.
                apply_winsor = !winsor_lowconf_only || mean_margin_raw <= winsor_margin_max;
                low_q_eff = winsor_low;
                high_q_eff = winsor_high;
            }
        }
        if apply_winsor {
            winsorize_in_place(&mut d1, low_q_eff, high_q_eff);
            winsorize_in_place(&mut d2, low_q_eff, high_q_eff);
        }

        let mut wins = 0usize;

        let local_reps = if adaptive_enabled {
            run_reps(&ll0, &d1, &d2, 0, pilot_reps, &mut wins);
            let q = (wins as f64) / (pilot_reps as f64);
            let u_binom = (4.0 * q * (1.0 - q)).clamp(0.0, 1.0);
            let d1_mu = d1.iter().sum::<f64>() / (n_cache_sites as f64).max(1.0);
            let d2_mu = d2.iter().sum::<f64>() / (n_cache_sites as f64).max(1.0);
            let mean_margin = 0.5 * (d1_mu.abs() + d2_mu.abs());
            let u_margin = (1.0 / (1.0 + adaptive_margin_k * mean_margin)).clamp(0.0, 1.0);
            let u = (adaptive_blend * u_binom + (1.0 - adaptive_blend) * u_margin).clamp(0.0, 1.0);
            let reps_f =
                (niter as f64) * (adaptive_min_mult + (adaptive_max_mult - adaptive_min_mult) * u);
            let mut reps = reps_f.round() as usize;
            let min_floor = adaptive_min_reps.min(max_local_reps);
            reps = reps.max(min_floor).max(pilot_reps).min(max_local_reps);
            if reps > pilot_reps {
                run_reps(&ll0, &d1, &d2, pilot_reps, reps, &mut wins);
            }
            reps
        } else {
            run_reps(&ll0, &d1, &d2, 0, niter, &mut wins);
            niter
        };

        if local_reps == 0 {
            return Ok((c, 0.0));
        }
        Ok((c, (wins as f64) / (local_reps as f64)))
    };

    if let Some(tp) = pool {
        let rs: Vec<Result<(usize, f64), String>> =
            tp.install(|| targets.par_iter().map(eval_one).collect());
        let mut out = Vec::with_capacity(rs.len());
        for r in rs {
            out.push(r?);
        }
        Ok(out)
    } else {
        let mut out = Vec::with_capacity(targets.len());
        for &u in targets {
            out.push(eval_one(&u)?);
        }
        Ok(out)
    }
}

fn shlike_support_map(
    tree: &MlTree,
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    pool: Option<&ThreadPool>,
    niter: usize,
) -> Result<Vec<Option<f64>>, String> {
    if niter == 0 {
        return Ok(vec![None; tree.nodes.len()]);
    }
    let n_sites = a.shape()[1];
    if n_sites == 0 {
        return Err("cannot compute SH-like support on empty alignment".to_string());
    }
    let n_nodes = tree.nodes.len();
    if n_nodes == 0 {
        return Ok(Vec::new());
    }

    // Cache memory per site: CLV + UP + logscale ~= n_nodes * (32 + 32 + 8) bytes.
    let bytes_per_site = (n_nodes as u128).saturating_mul(72);
    let cache_mb = env_usize_or("JANUSX_ML_SHLIKE_CACHE_MB", 112).max(16);
    let cache_budget_bytes = (cache_mb as u128) * 1024u128 * 1024u128;
    let budget_sites = (cache_budget_bytes / bytes_per_site).max(64) as usize;
    let default_cache_sites = n_sites.min(2048).max(256);
    let requested_cache_sites = env_usize_or("JANUSX_ML_SHLIKE_CACHE_SITES", default_cache_sites);
    let cache_sites = n_sites
        .min(requested_cache_sites.max(64))
        .min(budget_sites.max(1));

    let default_draws = cache_sites;
    let draws = env_usize_or("JANUSX_ML_SHLIKE_DRAW_SITES", default_draws)
        .max(1)
        .min(cache_sites);
    let eps = env_f64_or("JANUSX_ML_SHLIKE_EPS", 1e-300).max(1e-300);
    let seed = env_u64_or("JANUSX_ML_BOOTSTRAP_SEED", 20260321);
    let use_weighted = env_bool_or("JANUSX_ML_SHLIKE_WEIGHTED", true);
    let optimize_t = env_bool_or("JANUSX_ML_SHLIKE_OPT_T", false);
    let weight_power = env_f64_or("JANUSX_ML_SHLIKE_WEIGHT_POWER", 0.80);
    let weight_uniform_mix = env_f64_or("JANUSX_ML_SHLIKE_WEIGHT_UNIFORM_MIX", 0.15);
    let cache_sites_1 =
        sample_site_indices_stratified(n_sites, cache_sites, seed ^ 0x9E37_79B9_7F4A_7C15);
    let cache = build_jc_site_cache(tree, a, cache_sites_1, pool)?;
    if !cache.total_ll.is_finite() {
        return Ok(vec![None; tree.nodes.len()]);
    }
    let up = build_jc_up_messages(tree, &cache, pool)?;
    let site_w1 = if use_weighted {
        Some(temper_site_weights(
            site_informativeness_weights(a, &cache.sites),
            weight_power,
            weight_uniform_mix,
        ))
    } else {
        None
    };
    let targets: Vec<usize> = (0..n_nodes)
        .filter(|&u| tree.nodes[u].parent.is_some() && tree.is_internal(u))
        .collect();

    let mut support = vec![None; tree.nodes.len()];
    let pairs_1 = shlike_support_on_cache(
        tree,
        &cache,
        &up,
        &targets,
        niter,
        draws,
        seed,
        eps,
        site_w1.as_deref(),
        optimize_t,
        pool,
    )?;
    for (u, s) in pairs_1 {
        support[u] = Some(s);
    }

    // Optional second pass on uncertain branches to stabilize support values.
    let do_recheck = env_bool_or("JANUSX_ML_SHLIKE_RECHECK", true) && niter >= 50;
    if do_recheck {
        let low = env_f64_or("JANUSX_ML_SHLIKE_RECHECK_LOW", 0.20).clamp(0.0, 1.0);
        let high = env_f64_or("JANUSX_ML_SHLIKE_RECHECK_HIGH", 0.95).clamp(low, 1.0);
        let mut uncertain = Vec::<usize>::new();
        for &u in &targets {
            if let Some(s) = support[u] {
                if s >= low && s <= high {
                    uncertain.push(u);
                }
            }
        }
        if !uncertain.is_empty() {
            // Release first-pass caches before allocating second-pass cache.
            drop(site_w1);
            drop(up);
            drop(cache);
            let niter2 = env_usize_or("JANUSX_ML_SHLIKE_RECHECK_NITER", (niter / 2).max(1));
            if niter2 > 0 {
                let seed2 = seed.wrapping_add(0xD1B5_4A32_D192_ED03);
                let cache_sites_2 = sample_site_indices_stratified(
                    n_sites,
                    cache_sites,
                    seed2 ^ 0xA5A5_A5A5_A5A5_A5A5,
                );
                let cache2 = build_jc_site_cache(tree, a, cache_sites_2, pool)?;
                if cache2.total_ll.is_finite() {
                    let up2 = build_jc_up_messages(tree, &cache2, pool)?;
                    let site_w2 = if use_weighted {
                        Some(temper_site_weights(
                            site_informativeness_weights(a, &cache2.sites),
                            weight_power,
                            weight_uniform_mix,
                        ))
                    } else {
                        None
                    };
                    let pairs_2 = shlike_support_on_cache(
                        tree,
                        &cache2,
                        &up2,
                        &uncertain,
                        niter2,
                        draws.min(cache2.sites.len()),
                        seed2,
                        eps,
                        site_w2.as_deref(),
                        optimize_t,
                        pool,
                    )?;
                    let w1 = niter as f64;
                    let w2 = niter2 as f64;
                    let denom = w1 + w2;
                    for (u, s2) in pairs_2 {
                        if let Some(s1) = support[u] {
                            support[u] = Some(((s1 * w1) + (s2 * w2)) / denom);
                        } else {
                            support[u] = Some(s2);
                        }
                    }
                }
            }
        }
    }
    Ok(support)
}

fn bootstrap_resample_alignment(
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    rng: &mut StdRng,
) -> Array2<u8> {
    let n_taxa = a.shape()[0];
    let n_sites = a.shape()[1];
    let mut out = Array2::<u8>::zeros((n_taxa, n_sites));
    for k in 0..n_sites {
        let src_k = rng.random_range(0..n_sites);
        for i in 0..n_taxa {
            out[[i, k]] = a[[i, src_k]];
        }
    }
    out
}

fn bootstrap_support_map(
    base_tree: &MlTree,
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    sample_ids: &[String],
    min_ov: u64,
    pool: Option<&ThreadPool>,
    mode: &str,
    niter: usize,
) -> Result<Vec<Option<f64>>, String> {
    if niter == 0 {
        return Ok(vec![None; base_tree.nodes.len()]);
    }
    let n_leaves = sample_ids.len();
    let base_keys = ml_node_split_keys(base_tree, n_leaves)?;
    let mut target_keys = HashSet::<Vec<u64>>::new();
    for key in base_keys.iter().flatten() {
        target_keys.insert(key.clone());
    }
    let mut counts: HashMap<Vec<u64>, usize> = HashMap::new();
    for k in target_keys.iter() {
        counts.insert(k.clone(), 0);
    }

    let mut rng = StdRng::seed_from_u64(env_u64_or("JANUSX_ML_BOOTSTRAP_SEED", 20260321));
    for _ in 0..niter {
        let boot = bootstrap_resample_alignment(a, &mut rng);
        let boot_view = boot.view();
        let rep_tree = infer_ml_tree_jc69(&boot_view, sample_ids, min_ov, pool, mode)?;
        let rep_keys = ml_node_split_keys(&rep_tree, n_leaves)?;
        let mut seen = HashSet::<Vec<u64>>::new();
        for key in rep_keys.into_iter().flatten() {
            if target_keys.contains(&key) {
                seen.insert(key);
            }
        }
        for key in seen {
            if let Some(v) = counts.get_mut(&key) {
                *v += 1;
            }
        }
    }

    let mut support = vec![None; base_tree.nodes.len()];
    let denom = niter as f64;
    for (u, key_opt) in base_keys.into_iter().enumerate() {
        if let Some(key) = key_opt {
            let c = counts.get(&key).copied().unwrap_or(0);
            support[u] = Some((c as f64) / denom);
        }
    }
    Ok(support)
}

fn render_ml_newick(
    tree: &MlTree,
    u: usize,
    support: Option<&[Option<f64>]>,
) -> Result<String, String> {
    if u >= tree.nodes.len() {
        return Err("ML render encountered invalid node id".to_string());
    }
    if let Some((l, r)) = tree.children(u) {
        let ltxt = render_ml_newick(tree, l, support)?;
        let rtxt = render_ml_newick(tree, r, support)?;
        let ll = tree.nodes[l].blen_to_parent;
        let rl = tree.nodes[r].blen_to_parent;
        let label = if tree.nodes[u].parent.is_some() {
            if let Some(sup) = support.and_then(|v| v.get(u)).and_then(|x| *x) {
                format!("{:.3}", sup)
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        Ok(format!(
            "({}:{:.10},{}:{:.10}){}",
            ltxt, ll, rtxt, rl, label
        ))
    } else {
        let nm = tree.nodes[u]
            .name
            .as_deref()
            .ok_or_else(|| "ML render leaf missing name".to_string())?;
        Ok(quote_newick_name(nm))
    }
}

#[pyfunction]
#[pyo3(signature = (
    aln,
    sample_ids,
    min_overlap=1usize,
    max_taxa=2000usize,
    threads=0usize,
    ml_mode="exact",
    bootstrap_niter=0usize,
    support_mode="bootstrap"
))]
pub fn ml_newick_from_alignment_u8(
    aln: PyReadonlyArray2<u8>,
    sample_ids: Vec<String>,
    min_overlap: usize,
    max_taxa: usize,
    threads: usize,
    ml_mode: &str,
    bootstrap_niter: usize,
    support_mode: &str,
) -> PyResult<String> {
    let a_in = aln.as_array();
    let shape = a_in.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(
            "alignment must be a 2D uint8 matrix: (n_samples, n_sites)",
        ));
    }
    let n_taxa = shape[0];
    let n_sites = shape[1];
    if n_taxa < 2 {
        return Err(PyValueError::new_err(
            "need at least 2 samples to build a tree",
        ));
    }
    if n_sites == 0 {
        return Err(PyValueError::new_err("alignment has zero sites"));
    }
    if sample_ids.len() != n_taxa {
        return Err(PyValueError::new_err(format!(
            "sample_ids length mismatch: got {}, expected {}",
            sample_ids.len(),
            n_taxa
        )));
    }
    if max_taxa == 0 {
        return Err(PyValueError::new_err("max_taxa must be > 0"));
    }
    if n_taxa > max_taxa {
        return Err(PyValueError::new_err(format!(
            "n_samples={} exceeds max_taxa={}; ML v1 uses NJ init (O(N^3))",
            n_taxa, max_taxa
        )));
    }

    let mode = ml_mode.to_ascii_lowercase();
    if mode != "exact" && mode != "approx" && mode != "compat" {
        return Err(PyValueError::new_err(format!(
            "invalid ml_mode='{}' (expected 'exact', 'approx', or 'compat')",
            ml_mode
        )));
    }
    let use_compat_strict_ambig =
        mode == "compat" && env_bool_or("JANUSX_ML_COMPAT_STRICT_AMBIG", true);
    let a_compat_owned = if use_compat_strict_ambig {
        Some(sanitize_alignment_compat_strict(&a_in))
    } else {
        None
    };
    let a = if let Some(ref owned) = a_compat_owned {
        owned.view()
    } else {
        a_in
    };
    let support_mode_l = support_mode.to_ascii_lowercase();
    if support_mode_l != "bootstrap" && support_mode_l != "shlike" && support_mode_l != "local" {
        return Err(PyValueError::new_err(format!(
            "invalid support_mode='{}' (expected 'bootstrap' or 'shlike')",
            support_mode
        )));
    }

    let min_ov = min_overlap.max(1) as u64;
    let chosen_threads = resolve_threads(threads);
    let use_parallel = chosen_threads > 1;
    let pool = if use_parallel {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(chosen_threads)
                .build()
                .map_err(|e| PyValueError::new_err(format!("failed to create thread pool: {e}")))?,
        )
    } else {
        None
    };

    let tree = infer_ml_tree_jc69(&a, &sample_ids, min_ov, pool.as_ref(), mode.as_str())
        .map_err(PyValueError::new_err)?;

    let support = if bootstrap_niter > 0 {
        if support_mode_l == "bootstrap" {
            Some(
                bootstrap_support_map(
                    &tree,
                    &a,
                    &sample_ids,
                    min_ov,
                    pool.as_ref(),
                    mode.as_str(),
                    bootstrap_niter,
                )
                .map_err(PyValueError::new_err)?,
            )
        } else {
            Some(
                shlike_support_map(&tree, &a, pool.as_ref(), bootstrap_niter)
                    .map_err(PyValueError::new_err)?,
            )
        }
    } else {
        None
    };

    let mut out =
        render_ml_newick(&tree, tree.root, support.as_deref()).map_err(PyValueError::new_err)?;
    out.push(';');
    Ok(out)
}

fn ensure_row_state_lt(
    i: usize,
    active: &[usize],
    row_built: &mut [bool],
    row_order: &mut [Vec<usize>],
    row_ptr: &mut [usize],
    row_new_scan: &mut [usize],
    row_new_heap: &mut [BinaryHeap<RowCandEntry>],
    created_nodes: &[usize],
    row_init_k: usize,
    row_new_window: usize,
    row_extra_cap: usize,
    stats: &mut RapidCoreStats,
    dist: &LowerTriDist,
) {
    if !row_built[i] {
        let k = row_init_k.min(active.len().saturating_sub(1)).max(1);
        let ord = row_top_k_from_active_lt(dist, i, active, k);
        row_order[i] = ord;
        row_built[i] = true;
        row_ptr[i] = 0;
        row_new_scan[i] = if row_new_window == 0 {
            created_nodes.len()
        } else {
            created_nodes.len().saturating_sub(row_new_window)
        };
        row_new_heap[i].clear();
        stats.row_builds += 1;
        stats.row_seed_items += row_order[i].len() as u64;
        return;
    }

    let mut scan_from = row_new_scan[i].min(created_nodes.len());
    if row_new_window > 0 {
        let recent_from = created_nodes.len().saturating_sub(row_new_window);
        if scan_from < recent_from {
            scan_from = recent_from;
        }
    }
    if scan_from >= created_nodes.len() {
        return;
    }
    let hp = &mut row_new_heap[i];
    for &j in created_nodes.iter().skip(scan_from) {
        if j == i {
            continue;
        }
        let d = dist.get(i, j);
        if d.is_finite() {
            hp.push(RowCandEntry { d, j });
        }
    }
    if row_extra_cap > 0 && hp.len() > row_extra_cap {
        let keep_n = row_extra_cap;
        let mut keep: Vec<RowCandEntry> = Vec::with_capacity(keep_n);
        for _ in 0..keep_n {
            if let Some(ent) = hp.pop() {
                keep.push(ent);
            } else {
                break;
            }
        }
        hp.clear();
        for ent in keep {
            hp.push(ent);
        }
    }
    row_new_scan[i] = created_nodes.len();
}

fn top_extra_active(
    i: usize,
    active_mask: &[bool],
    row_new_heap: &mut [BinaryHeap<RowCandEntry>],
) -> Option<RowCandEntry> {
    let hp = &mut row_new_heap[i];
    loop {
        let top = hp.peek().copied();
        match top {
            Some(ent) if ent.j != i && ent.j < active_mask.len() && active_mask[ent.j] => {
                return Some(ent);
            }
            Some(_) => {
                hp.pop();
            }
            None => return None,
        }
    }
}

fn row_lb_from_state_lt(
    i: usize,
    m: f64,
    r_i: f64,
    r_max: f64,
    active_mask: &[bool],
    row_order: &[Vec<usize>],
    row_ptr: &mut [usize],
    row_new_heap: &mut [BinaryHeap<RowCandEntry>],
    dist: &LowerTriDist,
) -> f64 {
    let ord = &row_order[i];
    let mut p = row_ptr[i].min(ord.len());
    while p < ord.len() {
        let j = ord[p];
        if j != i && j < active_mask.len() && active_mask[j] {
            break;
        }
        p += 1;
    }
    row_ptr[i] = p;

    let mut best_d = f64::INFINITY;
    if p < ord.len() {
        let j = ord[p];
        let d = dist.get(i, j);
        if d.is_finite() {
            best_d = best_d.min(d);
        }
    }
    if let Some(ent) = top_extra_active(i, active_mask, row_new_heap) {
        best_d = best_d.min(ent.d);
    }
    if !best_d.is_finite() {
        return f64::INFINITY;
    }
    (m - 2.0) * best_d - r_i - r_max
}

fn refresh_row_lb_lt(
    i: usize,
    m: f64,
    r_i: f64,
    r_max: f64,
    active: &[usize],
    active_mask: &[bool],
    row_built: &mut [bool],
    row_order: &mut [Vec<usize>],
    row_ptr: &mut [usize],
    row_new_scan: &mut [usize],
    row_new_heap: &mut [BinaryHeap<RowCandEntry>],
    created_nodes: &[usize],
    row_init_k: usize,
    row_new_window: usize,
    row_extra_cap: usize,
    stats: &mut RapidCoreStats,
    dist: &LowerTriDist,
) -> f64 {
    stats.refresh_calls += 1;
    ensure_row_state_lt(
        i,
        active,
        row_built,
        row_order,
        row_ptr,
        row_new_scan,
        row_new_heap,
        created_nodes,
        row_init_k,
        row_new_window,
        row_extra_cap,
        stats,
        dist,
    );
    row_lb_from_state_lt(
        i,
        m,
        r_i,
        r_max,
        active_mask,
        row_order,
        row_ptr,
        row_new_heap,
        dist,
    )
}

fn eval_row_best_q_with_pruning_lt(
    i: usize,
    m: f64,
    r: &[f64],
    r_max: f64,
    active: &[usize],
    active_mask: &[bool],
    row_built: &mut [bool],
    row_order: &mut [Vec<usize>],
    row_ptr: &mut [usize],
    row_new_scan: &mut [usize],
    row_new_heap: &mut [BinaryHeap<RowCandEntry>],
    created_nodes: &[usize],
    row_init_k: usize,
    row_new_window: usize,
    row_extra_cap: usize,
    stats: &mut RapidCoreStats,
    dist: &LowerTriDist,
) -> (f64, usize, f64) {
    stats.eval_calls += 1;
    ensure_row_state_lt(
        i,
        active,
        row_built,
        row_order,
        row_ptr,
        row_new_scan,
        row_new_heap,
        created_nodes,
        row_init_k,
        row_new_window,
        row_extra_cap,
        stats,
        dist,
    );

    let lb_now = row_lb_from_state_lt(
        i,
        m,
        r[i],
        r_max,
        active_mask,
        row_order,
        row_ptr,
        row_new_heap,
        dist,
    );
    if !lb_now.is_finite() {
        return (f64::INFINITY, i, f64::INFINITY);
    }

    let ord = &row_order[i];
    let p = row_ptr[i].min(ord.len());
    let mut best_q = f64::INFINITY;
    let mut best_j = i;

    for &j in ord.iter().skip(p) {
        if j == i || j >= active_mask.len() || !active_mask[j] {
            continue;
        }
        let d = dist.get(i, j);
        if !d.is_finite() {
            continue;
        }
        let rem_lb = (m - 2.0) * d - r[i] - r_max;
        if rem_lb >= best_q {
            break;
        }
        let q = (m - 2.0) * d - r[i] - r[j];
        if q < best_q || (q == best_q && j < best_j) {
            best_q = q;
            best_j = j;
        }
    }

    let mut touched: Vec<RowCandEntry> = Vec::new();
    loop {
        let top = top_extra_active(i, active_mask, row_new_heap);
        let Some(ent) = top else {
            break;
        };
        let rem_lb = (m - 2.0) * ent.d - r[i] - r_max;
        if rem_lb >= best_q {
            break;
        }
        let popped = row_new_heap[i].pop();
        if let Some(cand) = popped {
            let j = cand.j;
            let d = dist.get(i, j);
            if d.is_finite() {
                let q = (m - 2.0) * d - r[i] - r[j];
                if q < best_q || (q == best_q && j < best_j) {
                    best_q = q;
                    best_j = j;
                }
            }
            touched.push(cand);
        } else {
            break;
        }
    }
    for cand in touched {
        row_new_heap[i].push(cand);
    }

    (best_q, best_j, lb_now)
}

fn nj_newick_lowertri_rapid_core(
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    sample_ids: Vec<String>,
    min_ov: u64,
    pool: Option<&ThreadPool>,
) -> Result<String, String> {
    let n_taxa = sample_ids.len();
    let max_nodes = n_taxa
        .checked_mul(2)
        .and_then(|x| x.checked_sub(1))
        .ok_or_else(|| "node count overflow in rapid lower-tri NJ".to_string())?;
    let mut dist = build_distance_lowertri_with_capacity(a, n_taxa, min_ov, max_nodes, pool)?;
    let mut nodes: Vec<TreeNode> = sample_ids.into_iter().map(TreeNode::Leaf).collect();
    let mut active: Vec<usize> = (0..n_taxa).collect();
    let mut r = compute_r_sums_lt(&active, &dist);

    let mut row_order = vec![Vec::<usize>::new(); max_nodes];
    let mut row_ptr = vec![0usize; max_nodes];
    let mut row_built = vec![false; max_nodes];
    let mut row_new_scan = vec![0usize; max_nodes];
    let mut row_new_heap = vec![BinaryHeap::<RowCandEntry>::new(); max_nodes];
    let mut row_epoch = vec![0usize; max_nodes];
    let mut row_seen = vec![0usize; max_nodes];
    let mut row_lb_cache = vec![f64::INFINITY; max_nodes];
    let mut row_token = vec![0u64; max_nodes];
    let mut rapid_heap = BinaryHeap::<RapidRowEntry>::new();
    let mut active_mask = vec![false; max_nodes];
    let mut created_nodes: Vec<usize> = Vec::with_capacity(max_nodes.saturating_sub(n_taxa));

    let rapid_mode_raw = std::env::var("JANUSX_RAPID_CORE_MODE")
        .unwrap_or_else(|_| "fast".to_string())
        .trim()
        .to_ascii_lowercase();
    let rapid_mode = match rapid_mode_raw.as_str() {
        "accurate" | "lowmem" | "hybrid" | "fast" => rapid_mode_raw.as_str(),
        _ => "fast",
    };
    let mode_accurate = rapid_mode == "accurate";
    let mode_lowmem = rapid_mode == "lowmem";
    let mode_hybrid = rapid_mode == "hybrid";

    let rapid_pop_mult = env_usize("JANUSX_RAPID_POP_MULT")
        .unwrap_or(if mode_accurate { 8 } else { 4 })
        .max(1);
    let rapid_unrefreshed_allow = env_usize("JANUSX_RAPID_UNREFRESHED");
    let rapid_min_refresh = env_usize("JANUSX_RAPID_MIN_REFRESH")
        .unwrap_or(if mode_accurate { 32 } else { 16 })
        .max(1);
    let rapid_batch_rows = env_usize("JANUSX_RAPID_BATCH_ROWS").unwrap_or(if mode_accurate {
        24
    } else if mode_hybrid {
        16
    } else if mode_lowmem {
        12
    } else {
        8
    });
    let rapid_new_window = env_usize("JANUSX_RAPID_NEW_WINDOW")
        .unwrap_or(if mode_lowmem || mode_hybrid { 256 } else { 0 });
    let rapid_extra_cap = env_usize("JANUSX_RAPID_EXTRA_CAP")
        .unwrap_or(if mode_lowmem || mode_hybrid { 64 } else { 0 });
    let rapid_strict_steps = env_usize("JANUSX_RAPID_STRICT_STEPS").unwrap_or(if mode_accurate {
        8
    } else if mode_hybrid {
        8
    } else if mode_lowmem {
        6
    } else {
        2
    });
    let rapid_chain_steps = env_usize("JANUSX_RAPID_CHAIN_STEPS").unwrap_or(if mode_accurate {
        16
    } else if mode_hybrid {
        16
    } else if mode_lowmem {
        12
    } else {
        8
    });
    let rapid_verify_every_env = env_usize("JANUSX_RAPID_VERIFY_EVERY");
    let rapid_verify_hi = env_usize("JANUSX_RAPID_VERIFY_HI").unwrap_or(if mode_accurate {
        16
    } else if mode_hybrid {
        32
    } else {
        0
    });
    let rapid_verify_mid = env_usize("JANUSX_RAPID_VERIFY_MID").unwrap_or(if mode_accurate {
        64
    } else if mode_hybrid {
        64
    } else {
        0
    });
    let rapid_verify_low = env_usize("JANUSX_RAPID_VERIFY_LOW").unwrap_or(0);
    let rapid_hi_active =
        env_usize("JANUSX_RAPID_HI_ACTIVE").unwrap_or((n_taxa.saturating_mul(3) / 4).max(16));
    let rapid_mid_active = env_usize("JANUSX_RAPID_MID_ACTIVE").unwrap_or((n_taxa / 3).max(8));
    let rapid_hybrid_replay_rows =
        env_usize("JANUSX_RAPID_HYBRID_REPLAY_ROWS").unwrap_or(if mode_hybrid {
            64
        } else if mode_lowmem {
            48
        } else {
            0
        });
    let rapid_hybrid_replay_every =
        env_usize("JANUSX_RAPID_HYBRID_REPLAY_EVERY").unwrap_or(if mode_hybrid {
            8
        } else if mode_lowmem {
            8
        } else {
            0
        });
    let rapid_dynamic_recheck =
        env_bool_or("JANUSX_RAPID_DYNAMIC_RECHECK", mode_lowmem || mode_hybrid);
    let rapid_dynamic_max_boost = env_usize("JANUSX_RAPID_DYNAMIC_MAX_BOOST")
        .unwrap_or(if mode_hybrid {
            4
        } else if mode_lowmem {
            3
        } else {
            1
        })
        .max(1);
    let rapid_lowmem_global_rows =
        env_usize("JANUSX_RAPID_LOWMEM_GLOBAL_ROWS").unwrap_or(if mode_lowmem {
            // lowmem default tuned to keep RF drift under control
            // while preserving >2x speedup on the mouse_hs1940 benchmark.
            4
        } else if mode_hybrid {
            4
        } else {
            0
        });
    let rapid_lowmem_local_every =
        env_usize("JANUSX_RAPID_LOWMEM_LOCAL_EVERY").unwrap_or(if mode_lowmem {
            2
        } else if mode_hybrid {
            4
        } else {
            0
        });
    let rapid_accurate_bootstrap_rounds = env_usize("JANUSX_RAPID_ACCURATE_BOOTSTRAP_ROUNDS")
        .unwrap_or(if mode_accurate { 96 } else { 0 });
    let rapid_accurate_bootstrap_every = env_usize("JANUSX_RAPID_ACCURATE_BOOTSTRAP_EVERY")
        .unwrap_or(if mode_accurate { 8 } else { 0 });
    let rapid_accurate_recheck_fullscan =
        env_bool_or("JANUSX_RAPID_ACCURATE_RECHECK_FULLSCAN", mode_accurate);
    let rapid_accurate_recheck_active =
        env_usize("JANUSX_RAPID_ACCURATE_RECHECK_ACTIVE").unwrap_or((n_taxa / 2).max(16));
    let rapid_row_k = env_usize("JANUSX_RAPID_CORE_ROW_K")
        .unwrap_or(if mode_accurate {
            128
        } else if mode_lowmem {
            48
        } else {
            64
        })
        .max(8);
    let rapid_profile = env_bool_or("JANUSX_RAPID_CORE_PROFILE", false);
    let mut stats = RapidCoreStats::default();

    active_mask.fill(false);
    for &x in &active {
        active_mask[x] = true;
    }
    let mut rapid_epoch: usize = 1;
    for &i in &active {
        row_order[i].clear();
        row_ptr[i] = 0;
        row_built[i] = false;
        row_new_scan[i] = 0;
        row_new_heap[i].clear();
        row_epoch[i] = rapid_epoch;
        row_lb_cache[i] = f64::NEG_INFINITY;
        row_token[i] = row_token[i].wrapping_add(1);
        rapid_heap.push(RapidRowEntry {
            lb: f64::NEG_INFINITY,
            row: i,
            epoch: 0,
            token: row_token[i],
        });
    }

    let mut iter_idx: usize = 0;
    let mut instability_level: usize = 0;
    let mut accurate_force_fullscan_next = false;
    while active.len() > 2 {
        let m = active.len() as f64;
        active_mask.fill(false);
        for &x in &active {
            active_mask[x] = true;
        }
        let max_instability = rapid_dynamic_max_boost.saturating_sub(1);
        let dynamic_boost = if rapid_dynamic_recheck {
            1usize.saturating_add(instability_level.min(max_instability))
        } else {
            1
        };
        let iter_new_window = if rapid_new_window == 0 {
            0
        } else {
            rapid_new_window
                .saturating_mul(dynamic_boost)
                .min(active.len().saturating_mul(2).max(rapid_new_window))
        };
        let iter_extra_cap = if rapid_extra_cap == 0 {
            0
        } else {
            rapid_extra_cap
                .saturating_mul(dynamic_boost)
                .min(rapid_row_k.saturating_mul(4).max(rapid_extra_cap))
        };

        rapid_epoch = rapid_epoch.wrapping_add(1);
        if rapid_epoch == 0 {
            rapid_epoch = 1;
            row_seen.fill(0);
            row_epoch.fill(0);
        }
        let mut r_max = f64::NEG_INFINITY;
        for &x in &active {
            if r[x] > r_max {
                r_max = r[x];
            }
        }

        let mut best = (f64::INFINITY, active[0], active[0]);
        let mut rapid_carry: Vec<RapidRowEntry> = Vec::new();
        let mut pending_refresh = active.len();
        let mut pop_count: usize = 0;
        let pop_cap = active.len().min(rapid_pop_mult.saturating_mul(16).max(16));
        let mut eval_rows: Vec<usize> = Vec::with_capacity(pop_cap.saturating_add(8));
        let pending_allow = match rapid_unrefreshed_allow {
            Some(v) => v.min(active.len()),
            None => active
                .len()
                .saturating_sub(rapid_min_refresh.min(active.len())),
        };
        loop {
            let Some(ent) = rapid_heap.pop() else {
                break;
            };
            stats.heap_pops += 1;
            let i = ent.row;
            if i >= active_mask.len() || !active_mask[i] {
                stats.inactive_skips += 1;
                continue;
            }
            if ent.token != row_token[i] {
                stats.token_skips += 1;
                continue;
            }

            if ent.epoch != rapid_epoch {
                let force_refresh = pending_refresh > pending_allow;
                let lb = if force_refresh {
                    let v = refresh_row_lb_lt(
                        i,
                        m,
                        r[i],
                        r_max,
                        &active,
                        &active_mask,
                        &mut row_built,
                        &mut row_order,
                        &mut row_ptr,
                        &mut row_new_scan,
                        &mut row_new_heap,
                        &created_nodes,
                        rapid_row_k,
                        iter_new_window,
                        iter_extra_cap,
                        &mut stats,
                        &dist,
                    );
                    if pending_refresh > 0 {
                        pending_refresh -= 1;
                    }
                    v
                } else {
                    // Once enough rows are refreshed this round, keep remaining rows lazy.
                    stats.stale_promotions += 1;
                    ent.lb
                };
                row_epoch[i] = rapid_epoch;
                row_lb_cache[i] = lb;
                row_token[i] = row_token[i].wrapping_add(1);
                rapid_heap.push(RapidRowEntry {
                    lb,
                    row: i,
                    epoch: rapid_epoch,
                    token: row_token[i],
                });
                continue;
            }

            if pending_refresh <= pending_allow && ent.lb >= best.0 {
                stats.lb_breaks += 1;
                rapid_carry.push(ent);
                break;
            }

            if row_seen[i] == rapid_epoch {
                stats.seen_skips += 1;
                rapid_carry.push(ent);
                continue;
            }

            let (rq, rj, lb_new) = eval_row_best_q_with_pruning_lt(
                i,
                m,
                &r,
                r_max,
                &active,
                &active_mask,
                &mut row_built,
                &mut row_order,
                &mut row_ptr,
                &mut row_new_scan,
                &mut row_new_heap,
                &created_nodes,
                rapid_row_k,
                iter_new_window,
                iter_extra_cap,
                &mut stats,
                &dist,
            );
            row_seen[i] = rapid_epoch;
            row_epoch[i] = rapid_epoch;
            row_lb_cache[i] = lb_new;
            row_token[i] = row_token[i].wrapping_add(1);
            rapid_heap.push(RapidRowEntry {
                lb: lb_new,
                row: i,
                epoch: rapid_epoch,
                token: row_token[i],
            });
            eval_rows.push(i);

            if rq < best.0 || (rq == best.0 && (i < best.1 || (i == best.1 && rj < best.2))) {
                best = (rq, i, rj);
            }
            pop_count += 1;
            if pending_refresh <= pending_allow && pop_count >= pop_cap && best.0.is_finite() {
                stats.popcap_breaks += 1;
                break;
            }
        }
        for ent in rapid_carry {
            rapid_heap.push(ent);
        }

        let rapid_verify_every = if let Some(v) = rapid_verify_every_env {
            v
        } else if active.len() >= rapid_hi_active {
            rapid_verify_hi
        } else if active.len() >= rapid_mid_active {
            rapid_verify_mid
        } else {
            rapid_verify_low
        };
        let periodic_fullscan = rapid_verify_every > 0 && (iter_idx % rapid_verify_every == 0);
        let accurate_bootstrap_fullscan = mode_accurate
            && rapid_accurate_bootstrap_every > 0
            && iter_idx < rapid_accurate_bootstrap_rounds
            && (iter_idx % rapid_accurate_bootstrap_every == 0);
        let mode_fullscan_trigger =
            periodic_fullscan || accurate_bootstrap_fullscan || accurate_force_fullscan_next;
        if mode_fullscan_trigger {
            stats.mode_fullscan += 1;
        }
        accurate_force_fullscan_next = false;
        let use_fullscan = !best.0.is_finite() || mode_fullscan_trigger;
        let (mut best_i, mut best_j, best_q, unstable_round) = if use_fullscan {
            if !best.0.is_finite() {
                stats.fullscan_fallback += 1;
            } else {
                stats.fullscan_verify += 1;
            }
            let (i, j, q) = lt_best_pair_full_scan(&active, &dist, &r, m);
            (i, j, q, false)
        } else {
            let mut cand_i = best.1;
            let mut cand_j = best.2;
            let mut cand_q = best.0;
            let mut round_changed = false;
            if rapid_strict_steps > 0 && cand_q.is_finite() {
                stats.strict_rechecks += 1;
                let (si, sj, sq) = lt_strict_recheck_pair(
                    &active,
                    &dist,
                    &r,
                    m,
                    cand_i,
                    cand_j,
                    cand_q,
                    rapid_strict_steps,
                );
                if sq < cand_q || (sq == cand_q && (si < cand_i || (si == cand_i && sj < cand_j))) {
                    stats.strict_updates += 1;
                    cand_i = si;
                    cand_j = sj;
                    cand_q = sq;
                    round_changed = true;
                }
            }
            if rapid_batch_rows > 0 && cand_q.is_finite() && !eval_rows.is_empty() {
                let mut rows = eval_rows.clone();
                if !rows.contains(&cand_i) {
                    rows.push(cand_i);
                }
                if !rows.contains(&cand_j) {
                    rows.push(cand_j);
                }
                rows.sort_by(|a, b| {
                    row_lb_cache[*a]
                        .total_cmp(&row_lb_cache[*b])
                        .then_with(|| a.cmp(b))
                });
                rows.dedup();
                let keep = rapid_batch_rows.max(2).min(rows.len());
                rows.truncate(keep);
                stats.strict_batch_rows += rows.len() as u64;

                let strict_pairs: Vec<(usize, usize, f64)> = if rows.len() > 1 {
                    if let Some(p) = pool {
                        p.install(|| {
                            rows.par_iter()
                                .map(|&ri| {
                                    let (rj, rq) = lt_set_best_hit(ri, &active, &dist, &r, m);
                                    if rj == usize::MAX {
                                        (ri, ri, f64::INFINITY)
                                    } else {
                                        let (a, b) = if ri <= rj { (ri, rj) } else { (rj, ri) };
                                        (a, b, rq)
                                    }
                                })
                                .collect()
                        })
                    } else {
                        rows.iter()
                            .map(|&ri| {
                                let (rj, rq) = lt_set_best_hit(ri, &active, &dist, &r, m);
                                if rj == usize::MAX {
                                    (ri, ri, f64::INFINITY)
                                } else {
                                    let (a, b) = if ri <= rj { (ri, rj) } else { (rj, ri) };
                                    (a, b, rq)
                                }
                            })
                            .collect()
                    }
                } else {
                    Vec::new()
                };
                for (a, b, q) in strict_pairs {
                    if !q.is_finite() {
                        continue;
                    }
                    if q < cand_q || (q == cand_q && (a < cand_i || (a == cand_i && b < cand_j))) {
                        stats.strict_updates += 1;
                        cand_i = a;
                        cand_j = b;
                        cand_q = q;
                        round_changed = true;
                    }
                }
            }
            if rapid_chain_steps > 0 && cand_q.is_finite() {
                let starts = [cand_i, cand_j];
                for &start in &starts {
                    stats.strict_rechecks += 1;
                    let (ci, cj, cq, steps, mutual) = lt_chain_recheck_pair(
                        &active,
                        &active_mask,
                        &dist,
                        &r,
                        m,
                        start,
                        rapid_chain_steps,
                    );
                    stats.chain_steps += steps as u64;
                    if mutual {
                        stats.chain_mutual += 1;
                    }
                    if cq.is_finite()
                        && (cq < cand_q
                            || (cq == cand_q && (ci < cand_i || (ci == cand_i && cj < cand_j))))
                    {
                        stats.strict_updates += 1;
                        cand_i = ci;
                        cand_j = cj;
                        cand_q = cq;
                        round_changed = true;
                    }
                }
            }
            if (mode_hybrid || mode_lowmem) && rapid_hybrid_replay_rows > 0 && cand_q.is_finite() {
                let periodic_replay =
                    rapid_hybrid_replay_every > 0 && (iter_idx % rapid_hybrid_replay_every == 0);
                if round_changed || periodic_replay {
                    let mut seeds = eval_rows.clone();
                    if !seeds.contains(&cand_i) {
                        seeds.push(cand_i);
                    }
                    if !seeds.contains(&cand_j) {
                        seeds.push(cand_j);
                    }
                    let local_probe = rapid_lowmem_global_rows > 0
                        && round_changed
                        && (rapid_lowmem_local_every == 0
                            || (iter_idx % rapid_lowmem_local_every == 0));
                    if local_probe {
                        let mut finite_rows: Vec<usize> = active
                            .iter()
                            .copied()
                            .filter(|&ri| ri < row_lb_cache.len() && row_lb_cache[ri].is_finite())
                            .collect();
                        finite_rows.sort_by(|a, b| {
                            row_lb_cache[*a]
                                .total_cmp(&row_lb_cache[*b])
                                .then_with(|| a.cmp(b))
                        });
                        let mut added = 0usize;
                        for ri in finite_rows.into_iter().take(rapid_lowmem_global_rows) {
                            seeds.push(ri);
                            added += 1;
                        }
                        // If row lb cache still has sparse finite coverage, probe a few stale rows too.
                        if added < rapid_lowmem_global_rows {
                            for &ri in &active {
                                if ri >= row_lb_cache.len() || row_lb_cache[ri].is_finite() {
                                    continue;
                                }
                                seeds.push(ri);
                                added += 1;
                                if added >= rapid_lowmem_global_rows {
                                    break;
                                }
                            }
                        }
                    }
                    seeds.sort_by(|a, b| {
                        row_lb_cache[*a]
                            .total_cmp(&row_lb_cache[*b])
                            .then_with(|| a.cmp(b))
                    });
                    seeds.dedup();
                    let replay_boost = if rapid_dynamic_recheck {
                        1usize.saturating_add(instability_level.min(max_instability))
                    } else {
                        1
                    };
                    let target_rows = rapid_hybrid_replay_rows.saturating_mul(replay_boost);
                    let keep = target_rows.max(2).min(seeds.len());
                    seeds.truncate(keep);
                    if !seeds.is_empty() {
                        stats.hybrid_replays += 1;
                        stats.hybrid_replay_rows += seeds.len() as u64;
                        let (ri, rj, rq) = lt_best_pair_from_seed_rows(
                            &seeds,
                            &active,
                            &active_mask,
                            &dist,
                            &r,
                            m,
                            pool,
                        );
                        if rq.is_finite()
                            && (rq < cand_q
                                || (rq == cand_q && (ri < cand_i || (ri == cand_i && rj < cand_j))))
                        {
                            stats.strict_updates += 1;
                            cand_i = ri;
                            cand_j = rj;
                            cand_q = rq;
                        }
                    }
                }
            }
            if rapid_accurate_recheck_fullscan
                && mode_accurate
                && round_changed
                && cand_q.is_finite()
                && active.len() >= rapid_accurate_recheck_active
            {
                stats.fullscan_verify += 1;
                let (fi, fj, fq) = lt_best_pair_full_scan(&active, &dist, &r, m);
                cand_i = fi;
                cand_j = fj;
                cand_q = fq;
                round_changed = false;
            }
            (cand_i, cand_j, cand_q, round_changed)
        };
        if rapid_dynamic_recheck {
            if unstable_round {
                instability_level = instability_level.saturating_add(1).min(max_instability);
            } else {
                instability_level = instability_level.saturating_sub(1);
            }
        }
        if mode_accurate && unstable_round && active.len() >= rapid_accurate_recheck_active {
            accurate_force_fullscan_next = true;
        }
        if !best_q.is_finite() {
            return Err("failed to build rapid lower-tri NJ tree: no valid pair found".to_string());
        }
        if best_i > best_j {
            std::mem::swap(&mut best_i, &mut best_j);
        }

        let dij = dist.get(best_i, best_j);
        let denom = m - 2.0;
        if denom <= 0.0 {
            return Err("failed to build rapid lower-tri NJ tree: invalid denominator".to_string());
        }
        let delta = (r[best_i] - r[best_j]) / denom;
        let mut li = 0.5 * (dij + delta);
        let mut lj = dij - li;
        if !li.is_finite() {
            li = 0.0;
        }
        if !lj.is_finite() {
            lj = 0.0;
        }
        li = li.max(0.0);
        lj = lj.max(0.0);

        let u = nodes.len();
        if u >= max_nodes {
            return Err("node index overflow in rapid lower-tri NJ".to_string());
        }
        for &k in &active {
            if k == best_i || k == best_j {
                continue;
            }
            let duk = 0.5 * (dist.get(best_i, k) + dist.get(best_j, k) - dij);
            let v = if duk.is_finite() { duk.max(0.0) } else { 0.0 };
            dist.set(u, k, v);
        }
        nodes.push(TreeNode::Internal {
            left: best_i,
            right: best_j,
            left_len: li,
            right_len: lj,
        });

        let mut next_active: Vec<usize> = Vec::with_capacity(active.len() - 1);
        for &x in &active {
            if x != best_i && x != best_j {
                next_active.push(x);
            }
        }
        next_active.push(u);

        let mut r_u = 0.0f64;
        for &k in &active {
            if k == best_i || k == best_j {
                continue;
            }
            let dik = dist.get(k, best_i);
            let djk = dist.get(k, best_j);
            let duk = dist.get(u, k);
            r[k] = r[k] - dik - djk + duk;
            r_u += duk;
        }
        r[u] = r_u;
        r[best_i] = 0.0;
        r[best_j] = 0.0;

        created_nodes.push(u);
        row_order[u].clear();
        row_ptr[u] = 0;
        row_built[u] = false;
        row_new_scan[u] = created_nodes.len();
        row_new_heap[u].clear();
        row_epoch[u] = 0;
        row_seen[u] = 0;
        row_lb_cache[best_i] = f64::INFINITY;
        row_lb_cache[best_j] = f64::INFINITY;
        row_lb_cache[u] = f64::NEG_INFINITY;

        row_token[best_i] = row_token[best_i].wrapping_add(1);
        row_token[best_j] = row_token[best_j].wrapping_add(1);
        row_token[u] = row_token[u].wrapping_add(1);
        rapid_heap.push(RapidRowEntry {
            lb: f64::NEG_INFINITY,
            row: u,
            epoch: 0,
            token: row_token[u],
        });

        active = next_active;
        iter_idx += 1;
    }

    let root_id = if active.len() == 2 {
        let a0 = active[0];
        let a1 = active[1];
        let d = if dist.get(a0, a1).is_finite() {
            dist.get(a0, a1).max(0.0)
        } else {
            0.0
        };
        let l = 0.5 * d;
        nodes.push(TreeNode::Internal {
            left: a0,
            right: a1,
            left_len: l,
            right_len: l,
        });
        nodes.len() - 1
    } else {
        active[0]
    };
    let mut out = render_newick(root_id, &nodes);
    out.push(';');
    if rapid_profile {
        eprintln!(
            "rapid-core profile: mode={}, row_k={}, pop_mult={}, min_refresh={}, batch_rows={}, new_window={}, extra_cap={}, strict_steps={}, chain_steps={}, row_builds={}, row_seed_items={}, refresh_calls={}, eval_calls={}, stale_promotions={}, strict_rechecks={}, strict_updates={}, strict_batch_rows={}, chain_eval_steps={}, chain_mutual={}, hybrid_replays={}, hybrid_replay_rows={}, mode_fullscan={}, heap_pops={}, inactive_skips={}, token_skips={}, seen_skips={}, lb_breaks={}, popcap_breaks={}, fullscan_fallback={}, fullscan_verify={}",
            rapid_mode,
            rapid_row_k,
            rapid_pop_mult,
            rapid_min_refresh,
            rapid_batch_rows,
            rapid_new_window,
            rapid_extra_cap,
            rapid_strict_steps,
            rapid_chain_steps,
            stats.row_builds,
            stats.row_seed_items,
            stats.refresh_calls,
            stats.eval_calls,
            stats.stale_promotions,
            stats.strict_rechecks,
            stats.strict_updates,
            stats.strict_batch_rows,
            stats.chain_steps,
            stats.chain_mutual,
            stats.hybrid_replays,
            stats.hybrid_replay_rows,
            stats.mode_fullscan,
            stats.heap_pops,
            stats.inactive_skips,
            stats.token_skips,
            stats.seen_skips,
            stats.lb_breaks,
            stats.popcap_breaks,
            stats.fullscan_fallback,
            stats.fullscan_verify,
        );
    }
    Ok(out)
}

fn compute_r_sums(active: &[usize], dist: &[Vec<f64>], pool: Option<&ThreadPool>) -> Vec<f64> {
    let mut r = vec![0.0f64; dist.len()];
    if let Some(p) = pool {
        let r_pairs: Vec<(usize, f64)> = p.install(|| {
            active
                .par_iter()
                .map(|&i| {
                    let mut s = 0.0f64;
                    for &k in active {
                        if i != k {
                            s += dist[i][k];
                        }
                    }
                    (i, s)
                })
                .collect()
        });
        for (i, s) in r_pairs {
            r[i] = s;
        }
    } else {
        for &i in active {
            let mut s = 0.0f64;
            for &k in active {
                if i != k {
                    s += dist[i][k];
                }
            }
            r[i] = s;
        }
    }
    r
}

fn best_pair_full_scan(
    active: &[usize],
    dist: &[Vec<f64>],
    r: &[f64],
    m: f64,
    pool: Option<&ThreadPool>,
) -> (f64, usize, usize) {
    if let Some(p) = pool {
        p.install(|| {
            active
                .par_iter()
                .enumerate()
                .map(|(ai, &i)| {
                    let mut local_q = f64::INFINITY;
                    let mut local_j = i;
                    for &j in active.iter().skip(ai + 1) {
                        let q = (m - 2.0) * dist[i][j] - r[i] - r[j];
                        if q < local_q {
                            local_q = q;
                            local_j = j;
                        }
                    }
                    (local_q, i, local_j)
                })
                .reduce(
                    || (f64::INFINITY, active[0], active[0]),
                    |a, b| if a.0 <= b.0 { a } else { b },
                )
        })
    } else {
        let mut best_q = f64::INFINITY;
        let mut best_i = active[0];
        let mut best_j = active[1];
        for ai in 0..(active.len() - 1) {
            let i = active[ai];
            for &j in active.iter().skip(ai + 1) {
                let q = (m - 2.0) * dist[i][j] - r[i] - r[j];
                if q < best_q {
                    best_q = q;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        (best_q, best_i, best_j)
    }
}

fn bionj_newick_from_matrices(
    mut dist: Vec<Vec<f64>>,
    mut var: Vec<Vec<f64>>,
    sample_ids: &[String],
    pool: Option<&ThreadPool>,
) -> Result<(String, f64), String> {
    let n_taxa = sample_ids.len();
    let mut nodes: Vec<TreeNode> = sample_ids.iter().cloned().map(TreeNode::Leaf).collect();
    let mut active: Vec<usize> = (0..n_taxa).collect();

    while active.len() > 2 {
        let m = active.len() as f64;
        let r = compute_r_sums(&active, &dist, pool);
        let (best_q, best_i, best_j) = best_pair_full_scan(&active, &dist, &r, m, pool);
        if !best_q.is_finite() {
            return Err("failed to build BIONJ tree: no valid pair found".to_string());
        }
        let dij = dist[best_i][best_j];
        let denom = m - 2.0;
        if denom <= 0.0 {
            return Err("failed to build BIONJ tree: invalid denominator".to_string());
        }
        let delta = (r[best_i] - r[best_j]) / denom;
        let mut li = 0.5 * (dij + delta);
        let mut lj = dij - li;
        if !li.is_finite() {
            li = 0.0;
        }
        if !lj.is_finite() {
            lj = 0.0;
        }
        li = li.max(0.0);
        lj = lj.max(0.0);

        let lambda = if active.len() <= 3 {
            0.5
        } else {
            let vij = var[best_i][best_j].max(1e-12);
            let mut numer = 0.0;
            for &k in &active {
                if k == best_i || k == best_j {
                    continue;
                }
                numer += var[best_j][k] - var[best_i][k];
            }
            let dlam = 2.0 * (m - 2.0) * vij;
            let mut lam = if dlam.abs() > 1e-18 {
                0.5 + numer / dlam
            } else {
                0.5
            };
            if !lam.is_finite() {
                lam = 0.5;
            }
            lam.clamp(0.0, 1.0)
        };
        let one_minus_lambda = 1.0 - lambda;

        let u = add_dist_node(&mut dist);
        let uv = add_dist_node(&mut var);
        if uv != u {
            return Err("failed to build BIONJ tree: variance/distance index mismatch".to_string());
        }
        for &k in &active {
            if k == best_i || k == best_j {
                continue;
            }
            let dik = dist[best_i][k];
            let djk = dist[best_j][k];
            let mut duk = lambda * (dik - li) + one_minus_lambda * (djk - lj);
            if !duk.is_finite() || duk < 0.0 {
                duk = 0.5 * (dik + djk - dij);
            }
            if !duk.is_finite() {
                duk = 0.0;
            }
            duk = duk.max(0.0);
            dist[u][k] = duk;
            dist[k][u] = duk;

            let vik = var[best_i][k];
            let vjk = var[best_j][k];
            let vij = var[best_i][best_j];
            let mut vuk = lambda * vik + one_minus_lambda * vjk - lambda * one_minus_lambda * vij;
            if !vuk.is_finite() || vuk < 0.0 {
                vuk = 0.5 * (vik + vjk);
            }
            if !vuk.is_finite() {
                vuk = 1e-12;
            }
            var[u][k] = vuk.max(1e-12);
            var[k][u] = var[u][k];
        }
        nodes.push(TreeNode::Internal {
            left: best_i,
            right: best_j,
            left_len: li,
            right_len: lj,
        });
        let mut next_active: Vec<usize> = Vec::with_capacity(active.len() - 1);
        for &x in &active {
            if x != best_i && x != best_j {
                next_active.push(x);
            }
        }
        next_active.push(u);
        active = next_active;
    }

    let root_id = if active.len() == 2 {
        let a0 = active[0];
        let a1 = active[1];
        let mut d = dist[a0][a1];
        if !d.is_finite() || d < 0.0 {
            d = 0.0;
        }
        let l = 0.5 * d;
        nodes.push(TreeNode::Internal {
            left: a0,
            right: a1,
            left_len: l,
            right_len: l,
        });
        nodes.len() - 1
    } else {
        active[0]
    };
    let mut out = render_newick(root_id, &nodes);
    out.push(';');
    let mut tree_len = 0.0f64;
    for node in &nodes {
        if let TreeNode::Internal {
            left_len,
            right_len,
            ..
        } = node
        {
            tree_len += left_len.max(0.0) + right_len.max(0.0);
        }
    }
    Ok((out, tree_len))
}

fn nj_newick_bionj_from_alignment(
    a: &numpy::ndarray::ArrayView2<'_, u8>,
    sample_ids: Vec<String>,
    min_ov: u64,
    pool: Option<&ThreadPool>,
    var_mode_override: Option<BionjVarMode>,
) -> Result<String, String> {
    let n_taxa = sample_ids.len();
    if n_taxa < 2 {
        return Err("need at least 2 samples to build a tree".to_string());
    }
    let var_mode = match var_mode_override {
        Some(m) => m,
        None => resolve_bionj_var_mode()?,
    };
    let bionj_profile = env_bool_or("JANUSX_BIONJ_PROFILE", false);

    if var_mode == BionjVarMode::Auto {
        let mut best_tree: Option<(String, f64, BionjVarMode)> = None;
        for mode in [BionjVarMode::Dist, BionjVarMode::Jc] {
            let (dist, var) = build_distance_and_variance_matrix(a, n_taxa, min_ov, pool, mode);
            let (nwk, tree_len) = bionj_newick_from_matrices(dist, var, &sample_ids, pool)?;
            if bionj_profile {
                eprintln!(
                    "bionj auto candidate: var_mode={}, tree_len={:.6}",
                    mode.as_str(),
                    tree_len
                );
            }
            match &best_tree {
                Some((_, best_len, _)) if tree_len >= *best_len => {}
                _ => {
                    best_tree = Some((nwk, tree_len, mode));
                }
            }
        }
        if let Some((nwk, best_len, best_mode)) = best_tree {
            if bionj_profile {
                eprintln!(
                    "bionj auto selected: var_mode={}, tree_len={:.6}",
                    best_mode.as_str(),
                    best_len
                );
            }
            return Ok(nwk);
        }
        return Err("failed to build BIONJ tree in auto mode".to_string());
    }

    let (dist, var) = build_distance_and_variance_matrix(a, n_taxa, min_ov, pool, var_mode);
    let (nwk, tree_len) = bionj_newick_from_matrices(dist, var, &sample_ids, pool)?;
    if bionj_profile {
        eprintln!(
            "bionj run: var_mode={}, tree_len={:.6}",
            var_mode.as_str(),
            tree_len
        );
    }
    Ok(nwk)
}

fn row_top_k_from_active(dist: &[Vec<f64>], i: usize, active: &[usize], k: usize) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }
    let mut pairs: Vec<(f64, usize)> =
        Vec::with_capacity(active.len().min(k.saturating_mul(2) + 8));
    for &j in active {
        if j == i {
            continue;
        }
        let d = dist[i][j];
        if d.is_finite() {
            pairs.push((d, j));
        }
    }
    if pairs.len() > k {
        let kth = k - 1;
        pairs.select_nth_unstable_by(kth, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(Ordering::Greater)
        });
        pairs.truncate(k);
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    pairs.into_iter().map(|(_, j)| j).collect()
}

fn init_top_hits(
    dist: &[Vec<f64>],
    active: &[usize],
    k: usize,
    pool: Option<&ThreadPool>,
) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::<usize>::new(); dist.len()];
    if k == 0 || active.len() < 2 {
        return out;
    }
    if let Some(p) = pool {
        let rows: Vec<(usize, Vec<usize>)> = p.install(|| {
            active
                .par_iter()
                .map(|&i| (i, row_top_k_from_active(dist, i, active, k)))
                .collect()
        });
        for (i, row) in rows {
            out[i] = row;
        }
    } else {
        for &i in active {
            out[i] = row_top_k_from_active(dist, i, active, k);
        }
    }
    out
}

fn best_pair_top_hits(
    active: &[usize],
    active_mask: &[bool],
    top_hits: &[Vec<usize>],
    dist: &[Vec<f64>],
    r: &[f64],
    m: f64,
    pool: Option<&ThreadPool>,
) -> (f64, usize, usize) {
    if let Some(p) = pool {
        p.install(|| {
            active
                .par_iter()
                .map(|&i| {
                    let mut local = (f64::INFINITY, i, i);
                    for &j in &top_hits[i] {
                        if j == i || j >= active_mask.len() || !active_mask[j] {
                            continue;
                        }
                        let q = (m - 2.0) * dist[i][j] - r[i] - r[j];
                        if q < local.0 {
                            local = (q, i, j);
                        }
                    }
                    local
                })
                .reduce(
                    || (f64::INFINITY, active[0], active[0]),
                    |a, b| if a.0 <= b.0 { a } else { b },
                )
        })
    } else {
        let mut best = (f64::INFINITY, active[0], active[0]);
        for &i in active {
            for &j in &top_hits[i] {
                if j == i || j >= active_mask.len() || !active_mask[j] {
                    continue;
                }
                let q = (m - 2.0) * dist[i][j] - r[i] - r[j];
                if q < best.0 {
                    best = (q, i, j);
                }
            }
        }
        best
    }
}

fn ensure_row_sorted_with_extras(
    i: usize,
    active: &[usize],
    row_built: &mut [bool],
    row_order: &mut [Vec<usize>],
    row_extra: &mut [Vec<usize>],
    row_ptr: &mut [usize],
    dist: &[Vec<f64>],
) {
    if !row_built[i] {
        let mut ord: Vec<usize> = active.iter().copied().filter(|&j| j != i).collect();
        ord.sort_by(|a, b| dist[i][*a].total_cmp(&dist[i][*b]).then_with(|| a.cmp(b)));
        row_order[i] = ord;
        row_built[i] = true;
        row_ptr[i] = 0;
    }

    if row_extra[i].is_empty() {
        return;
    }
    let extras = std::mem::take(&mut row_extra[i]);
    let ord = &mut row_order[i];
    for j in extras {
        if j == i {
            continue;
        }
        let dj = dist[i][j];
        let pos =
            ord.binary_search_by(|cand| dist[i][*cand].total_cmp(&dj).then_with(|| cand.cmp(&j)));
        if let Err(pos) = pos {
            ord.insert(pos, j);
            if pos <= row_ptr[i] {
                row_ptr[i] = pos;
            }
        }
    }
}

fn refresh_row_lb(
    i: usize,
    m: f64,
    r_i: f64,
    r_max: f64,
    active: &[usize],
    active_mask: &[bool],
    row_built: &mut [bool],
    row_order: &mut [Vec<usize>],
    row_extra: &mut [Vec<usize>],
    row_ptr: &mut [usize],
    dist: &[Vec<f64>],
) -> f64 {
    ensure_row_sorted_with_extras(i, active, row_built, row_order, row_extra, row_ptr, dist);
    let ord = &row_order[i];
    let mut p = row_ptr[i].min(ord.len());
    while p < ord.len() {
        let j = ord[p];
        if j != i && j < active_mask.len() && active_mask[j] {
            break;
        }
        p += 1;
    }
    row_ptr[i] = p;
    if p >= ord.len() {
        return f64::INFINITY;
    }
    let j = ord[p];
    let d = dist[i][j];
    if !d.is_finite() {
        return f64::INFINITY;
    }
    (m - 2.0) * d - r_i - r_max
}

fn eval_row_best_q_with_pruning(
    i: usize,
    m: f64,
    r: &[f64],
    r_max: f64,
    active: &[usize],
    active_mask: &[bool],
    row_built: &mut [bool],
    row_order: &mut [Vec<usize>],
    row_extra: &mut [Vec<usize>],
    row_ptr: &mut [usize],
    dist: &[Vec<f64>],
) -> (f64, usize) {
    ensure_row_sorted_with_extras(i, active, row_built, row_order, row_extra, row_ptr, dist);
    let ord = &row_order[i];
    let mut p = row_ptr[i].min(ord.len());
    while p < ord.len() {
        let j = ord[p];
        if j != i && j < active_mask.len() && active_mask[j] {
            break;
        }
        p += 1;
    }
    row_ptr[i] = p;
    if p >= ord.len() {
        return (f64::INFINITY, i);
    }

    let mut best_q = f64::INFINITY;
    let mut best_j = i;
    for &j in ord.iter().skip(p) {
        if j == i || j >= active_mask.len() || !active_mask[j] {
            continue;
        }
        let d = dist[i][j];
        if !d.is_finite() {
            continue;
        }
        let rem_lb = (m - 2.0) * d - r[i] - r_max;
        if rem_lb >= best_q {
            break;
        }
        let q = (m - 2.0) * d - r[i] - r[j];
        if q < best_q || (q == best_q && j < best_j) {
            best_q = q;
            best_j = j;
        }
    }
    (best_q, best_j)
}

fn convert_geno_to_alignment_u8(
    g: &numpy::ndarray::ArrayView2<'_, f32>,
    rc: &[u8],
    ac: &[u8],
) -> Result<Array2<u8>, String> {
    let shape = g.shape();
    if shape.len() != 2 {
        return Err("geno chunk must be 2D float32 array: (n_sites, n_samples)".to_string());
    }
    let m = shape[0];
    let n = shape[1];
    if rc.len() != m || ac.len() != m {
        return Err(format!(
            "ref/alt length mismatch: expected {}, got {}/{}",
            m,
            rc.len(),
            ac.len()
        ));
    }
    if m == 0 || n == 0 {
        return Ok(Array2::<u8>::zeros((n, m)));
    }

    let total = m
        .checked_mul(n)
        .ok_or_else(|| "matrix size overflow in conversion".to_string())?;

    let mut refs = vec![b'N'; m];
    let mut alts = vec![b'N'; m];
    let mut hets = vec![b'N'; m];
    for i in 0..m {
        let (r, a) = sanitize_base_pair(rc[i], ac[i]);
        refs[i] = r;
        alts[i] = a;
        hets[i] = het_iupac(r, a);
    }

    let mut row_major = vec![b'N'; total];
    row_major
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, out_row)| {
            let r = refs[i];
            let a = alts[i];
            let h = hets[i];
            for j in 0..n {
                let x = g[[i, j]];
                let b = if !x.is_finite() || x < 0.0 {
                    b'N'
                } else if x <= 0.5 {
                    r
                } else if x >= 1.5 {
                    a
                } else {
                    h
                };
                out_row[j] = b;
            }
        });

    let mut transposed = vec![b'N'; total];
    transposed
        .par_chunks_mut(m)
        .enumerate()
        .for_each(|(j, out_row)| {
            for i in 0..m {
                out_row[i] = row_major[i * n + j];
            }
        });
    Array2::from_shape_vec((n, m), transposed)
        .map_err(|e| format!("failed to build alignment chunk: {e}"))
}

#[pyfunction]
pub fn geno_chunk_to_alignment_u8<'py>(
    py: Python<'py>,
    geno: PyReadonlyArray2<'py, f32>,
    ref_codes: PyReadonlyArray1<'py, u8>,
    alt_codes: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let g = geno.as_array();
    let rc = ref_codes.as_array();
    let ac = alt_codes.as_array();
    let rc_vec: Vec<u8> = rc.iter().copied().collect();
    let ac_vec: Vec<u8> = ac.iter().copied().collect();
    let arr = convert_geno_to_alignment_u8(&g, &rc_vec, &ac_vec).map_err(PyValueError::new_err)?;
    Ok(PyArray2::from_owned_array(py, arr))
}

#[pyfunction]
pub fn geno_chunk_to_alignment_u8_sites<'py>(
    py: Python<'py>,
    geno: PyReadonlyArray2<'py, f32>,
    ref_alleles: Vec<String>,
    alt_alleles: Vec<String>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    if ref_alleles.len() != alt_alleles.len() {
        return Err(PyValueError::new_err(format!(
            "ref_alleles/alt_alleles length mismatch: {}/{}",
            ref_alleles.len(),
            alt_alleles.len()
        )));
    }
    let rc: Vec<u8> = ref_alleles.iter().map(|s| first_base_code(s)).collect();
    let ac: Vec<u8> = alt_alleles.iter().map(|s| first_base_code(s)).collect();
    let g = geno.as_array();
    let arr = convert_geno_to_alignment_u8(&g, &rc, &ac).map_err(PyValueError::new_err)?;
    Ok(PyArray2::from_owned_array(py, arr))
}

#[pyfunction]
pub fn geno_chunk_to_alignment_u8_siteinfo<'py>(
    py: Python<'py>,
    geno: PyReadonlyArray2<'py, f32>,
    sites: Vec<SiteInfo>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let g = geno.as_array();
    let m = g.shape()[0];
    if sites.len() != m {
        return Err(PyValueError::new_err(format!(
            "sites length mismatch: expected {}, got {}",
            m,
            sites.len()
        )));
    }
    let rc: Vec<u8> = sites
        .iter()
        .map(|s| first_base_code(s.ref_allele.as_str()))
        .collect();
    let ac: Vec<u8> = sites
        .iter()
        .map(|s| first_base_code(s.alt_allele.as_str()))
        .collect();
    let arr = convert_geno_to_alignment_u8(&g, &rc, &ac).map_err(PyValueError::new_err)?;
    Ok(PyArray2::from_owned_array(py, arr))
}

#[pyfunction]
#[pyo3(signature = (
    aln,
    sample_ids,
    min_overlap=1usize,
    max_taxa=2000usize,
    threads=0usize,
    nj_approx="exact",
    top_hits_k=64usize
))]
pub fn nj_newick_from_alignment_u8(
    aln: PyReadonlyArray2<u8>,
    sample_ids: Vec<String>,
    min_overlap: usize,
    max_taxa: usize,
    threads: usize,
    nj_approx: &str,
    top_hits_k: usize,
) -> PyResult<String> {
    let a = aln.as_array();
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(
            "alignment must be a 2D uint8 matrix: (n_samples, n_sites)",
        ));
    }
    let n_taxa = shape[0];
    let n_sites = shape[1];

    if n_taxa < 2 {
        return Err(PyValueError::new_err(
            "need at least 2 samples to build a tree",
        ));
    }
    if n_sites == 0 {
        return Err(PyValueError::new_err("alignment has zero sites"));
    }
    if sample_ids.len() != n_taxa {
        return Err(PyValueError::new_err(format!(
            "sample_ids length mismatch: got {}, expected {}",
            sample_ids.len(),
            n_taxa
        )));
    }
    if max_taxa == 0 {
        return Err(PyValueError::new_err("max_taxa must be > 0"));
    }
    if n_taxa > max_taxa {
        return Err(PyValueError::new_err(format!(
            "n_samples={} exceeds max_taxa={}; NJ in phase-1 is O(N^3)",
            n_taxa, max_taxa
        )));
    }
    let approx_mode = nj_approx.to_ascii_lowercase();
    let (use_top_hits, use_rapid_nj, use_lt_lazyq, use_rapid_core, use_bionj, bionj_var_override) =
        match approx_mode.as_str() {
            "exact" => (false, false, false, false, false, None),
            "bionj" | "bio-nj" | "bio_nj" => (false, false, false, false, true, None),
            "bionj-dist" | "bionj_dist" | "bionjdist" => {
                (false, false, false, false, true, Some(BionjVarMode::Dist))
            }
            "bionj-jc" | "bionj_jc" | "bionjjc" | "bionj-jc69" | "bionj_jc69" => {
                (false, false, false, false, true, Some(BionjVarMode::Jc))
            }
            "bionj-binom" | "bionj_binom" | "bionjbinom" => {
                (false, false, false, false, true, Some(BionjVarMode::Binom))
            }
            "bionj-auto" | "bionj_auto" | "bionjauto" => {
                (false, false, false, false, true, Some(BionjVarMode::Auto))
            }
            "top-hits" | "tophits" | "top_hits" => (true, false, false, false, false, None),
            "rapidnj" | "rapid-nj" | "rapid_nj" => (false, true, false, false, false, None),
        "lt-lazyq" | "lazyq-lt" | "lazyq" | "rapid-lt" | "rapidlt" | "nnc-lazyq" => {
            (false, false, true, false, false, None)
        }
        "rapid-core" | "rapid_core" | "rapidcore" | "lt-rapid-core" | "rapid-lt-core" => {
            (false, false, false, true, false, None)
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "invalid nj_approx='{}' (expected 'exact', 'bionj', 'bionj-dist', 'bionj-jc', 'bionj-binom', 'bionj-auto', 'top-hits', 'rapidnj', 'lt-lazyq', or 'rapid-core')",
                nj_approx
            )))
        }
    };
    if use_top_hits && top_hits_k == 0 {
        return Err(PyValueError::new_err(
            "top_hits_k must be > 0 when nj_approx='top-hits'",
        ));
    }

    let min_ov = min_overlap.max(1) as u64;
    let chosen_threads = resolve_threads(threads);
    let use_parallel = chosen_threads > 1;
    let pool = if use_parallel {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(chosen_threads)
                .build()
                .map_err(|e| PyValueError::new_err(format!("failed to create thread pool: {e}")))?,
        )
    } else {
        None
    };

    if use_bionj {
        let nwk = nj_newick_bionj_from_alignment(
            &a,
            sample_ids,
            min_ov,
            pool.as_ref(),
            bionj_var_override,
        )
            .map_err(PyValueError::new_err)?;
        return Ok(nwk);
    }

    if use_rapid_core {
        let nwk = nj_newick_lowertri_rapid_core(&a, sample_ids, min_ov, pool.as_ref())
            .map_err(PyValueError::new_err)?;
        return Ok(nwk);
    }

    if use_lt_lazyq {
        let nwk = nj_newick_lowertri_lazyq(&a, sample_ids, min_ov, pool.as_ref())
            .map_err(PyValueError::new_err)?;
        return Ok(nwk);
    }

    let mut dist = build_distance_matrix(&a, n_taxa, min_ov, pool.as_ref());

    let mut nodes: Vec<TreeNode> = sample_ids.into_iter().map(TreeNode::Leaf).collect();
    let mut active: Vec<usize> = (0..n_taxa).collect();
    let mut top_hits = if use_top_hits {
        init_top_hits(&dist, &active, top_hits_k, pool.as_ref())
    } else {
        Vec::new()
    };
    let mut r = if use_rapid_nj {
        compute_r_sums(&active, &dist, pool.as_ref())
    } else {
        Vec::new()
    };
    let mut row_order = if use_rapid_nj {
        vec![Vec::<usize>::new(); dist.len()]
    } else {
        Vec::new()
    };
    let mut row_extra = if use_rapid_nj {
        vec![Vec::<usize>::new(); dist.len()]
    } else {
        Vec::new()
    };
    let mut row_ptr = if use_rapid_nj {
        vec![0usize; dist.len()]
    } else {
        Vec::new()
    };
    let mut row_built = if use_rapid_nj {
        vec![false; dist.len()]
    } else {
        Vec::new()
    };
    let mut row_epoch = if use_rapid_nj {
        vec![0usize; dist.len()]
    } else {
        Vec::new()
    };
    let mut row_seen = if use_rapid_nj {
        vec![0usize; dist.len()]
    } else {
        Vec::new()
    };
    let mut row_token = if use_rapid_nj {
        vec![0u64; dist.len()]
    } else {
        Vec::new()
    };
    let mut rapid_heap = if use_rapid_nj {
        Some(BinaryHeap::<RapidRowEntry>::new())
    } else {
        None
    };
    let mut active_mask = vec![false; dist.len()];
    let refill_threshold = if top_hits_k < 4 { 1 } else { top_hits_k / 4 };
    let mut iter_idx: usize = 0;
    let refresh_every: usize = 64;
    let mut rapid_epoch: usize = 0;
    let rapid_pop_mult = env_usize_or("JANUSX_RAPID_POP_MULT", 8).max(1);
    let rapid_unrefreshed_allow = env_usize_or("JANUSX_RAPID_UNREFRESHED", 0);
    let rapid_verify_every = env_usize_or("JANUSX_RAPID_VERIFY_EVERY", 0);

    if use_rapid_nj {
        active_mask.fill(false);
        for &x in &active {
            active_mask[x] = true;
        }
        rapid_epoch = 1;
        for &i in &active {
            row_order[i].clear();
            row_extra[i].clear();
            row_ptr[i] = 0;
            row_built[i] = false;
            row_epoch[i] = rapid_epoch;
            row_token[i] = row_token[i].wrapping_add(1);
            if let Some(h) = rapid_heap.as_mut() {
                h.push(RapidRowEntry {
                    // Lazily build row-sorted index at first revalidation.
                    lb: f64::NEG_INFINITY,
                    row: i,
                    epoch: 0,
                    token: row_token[i],
                });
            }
        }
    }

    while active.len() > 2 {
        let m = active.len() as f64;
        if !use_rapid_nj {
            r = compute_r_sums(&active, &dist, pool.as_ref());
        }
        if active_mask.len() < dist.len() {
            active_mask.resize(dist.len(), false);
        }
        active_mask.fill(false);
        for &x in &active {
            active_mask[x] = true;
        }
        if use_top_hits {
            if iter_idx > 0 && iter_idx % refresh_every == 0 {
                top_hits = init_top_hits(&dist, &active, top_hits_k, pool.as_ref());
            }
        }

        let (best_q, best_i, best_j) = if use_rapid_nj {
            rapid_epoch = rapid_epoch.wrapping_add(1);
            if rapid_epoch == 0 {
                rapid_epoch = 1;
                row_seen.fill(0);
                row_epoch.fill(0);
            }
            let mut r_max = f64::NEG_INFINITY;
            for &x in &active {
                if r[x] > r_max {
                    r_max = r[x];
                }
            }

            let mut best = (f64::INFINITY, active[0], active[0]);
            let mut rapid_carry: Vec<RapidRowEntry> = Vec::new();
            let mut pending_refresh = active.len();
            let mut pop_count: usize = 0;
            let pop_cap = active.len().saturating_mul(rapid_pop_mult).max(64);
            let pending_allow = rapid_unrefreshed_allow.min(active.len());
            loop {
                let ent = if let Some(h) = rapid_heap.as_mut() {
                    h.pop()
                } else {
                    None
                };
                let Some(ent) = ent else {
                    break;
                };

                let i = ent.row;
                if i >= active_mask.len() || !active_mask[i] {
                    continue;
                }
                if ent.token != row_token[i] {
                    continue;
                }

                if ent.epoch != rapid_epoch {
                    let lb = refresh_row_lb(
                        i,
                        m,
                        r[i],
                        r_max,
                        &active,
                        &active_mask,
                        &mut row_built,
                        &mut row_order,
                        &mut row_extra,
                        &mut row_ptr,
                        &dist,
                    );
                    row_epoch[i] = rapid_epoch;
                    if pending_refresh > 0 {
                        pending_refresh -= 1;
                    }
                    row_token[i] = row_token[i].wrapping_add(1);
                    if let Some(h) = rapid_heap.as_mut() {
                        h.push(RapidRowEntry {
                            lb,
                            row: i,
                            epoch: rapid_epoch,
                            token: row_token[i],
                        });
                    }
                    continue;
                }

                // Rapid stop when enough rows are refreshed and lower-bound cannot beat best.
                if pending_refresh <= pending_allow && ent.lb >= best.0 {
                    rapid_carry.push(ent);
                    break;
                }

                if row_seen[i] == rapid_epoch {
                    rapid_carry.push(ent);
                    continue;
                }

                let (rq, rj) = eval_row_best_q_with_pruning(
                    i,
                    m,
                    &r,
                    r_max,
                    &active,
                    &active_mask,
                    &mut row_built,
                    &mut row_order,
                    &mut row_extra,
                    &mut row_ptr,
                    &dist,
                );
                row_seen[i] = rapid_epoch;
                let lb_new = refresh_row_lb(
                    i,
                    m,
                    r[i],
                    r_max,
                    &active,
                    &active_mask,
                    &mut row_built,
                    &mut row_order,
                    &mut row_extra,
                    &mut row_ptr,
                    &dist,
                );
                row_epoch[i] = rapid_epoch;
                row_token[i] = row_token[i].wrapping_add(1);
                if let Some(h) = rapid_heap.as_mut() {
                    h.push(RapidRowEntry {
                        lb: lb_new,
                        row: i,
                        epoch: rapid_epoch,
                        token: row_token[i],
                    });
                }

                if rq < best.0 || (rq == best.0 && (i < best.1 || (i == best.1 && rj < best.2))) {
                    best = (rq, i, rj);
                }
                pop_count += 1;
                if pending_refresh <= pending_allow && pop_count >= pop_cap && best.0.is_finite() {
                    break;
                }
            }
            if let Some(h) = rapid_heap.as_mut() {
                for ent in rapid_carry {
                    h.push(ent);
                }
            }

            if !best.0.is_finite() {
                best_pair_full_scan(&active, &dist, &r, m, pool.as_ref())
            } else {
                if rapid_verify_every > 0 && (iter_idx % rapid_verify_every == 0) {
                    best_pair_full_scan(&active, &dist, &r, m, pool.as_ref())
                } else {
                    best
                }
            }
        } else if use_top_hits {
            let mut best = best_pair_top_hits(
                &active,
                &active_mask,
                &top_hits,
                &dist,
                &r,
                m,
                pool.as_ref(),
            );
            if !best.0.is_finite() {
                // Sparse candidate lists can temporarily miss valid pairs; fallback keeps robustness.
                best = best_pair_full_scan(&active, &dist, &r, m, pool.as_ref());
            }
            best
        } else {
            best_pair_full_scan(&active, &dist, &r, m, pool.as_ref())
        };
        if !best_q.is_finite() {
            return Err(PyValueError::new_err(
                "failed to build NJ tree: no valid pair found",
            ));
        }

        let dij = dist[best_i][best_j];
        let denom = m - 2.0;
        if denom <= 0.0 {
            return Err(PyValueError::new_err(
                "failed to build NJ tree: invalid denominator",
            ));
        }
        let delta = (r[best_i] - r[best_j]) / denom;
        let mut li = 0.5 * (dij + delta);
        let mut lj = dij - li;
        if !li.is_finite() {
            li = 0.0;
        }
        if !lj.is_finite() {
            lj = 0.0;
        }
        li = li.max(0.0);
        lj = lj.max(0.0);

        let u = add_dist_node(&mut dist);
        debug_assert_eq!(u, nodes.len());
        for &k in &active {
            if k == best_i || k == best_j {
                continue;
            }
            let duk = 0.5 * (dist[best_i][k] + dist[best_j][k] - dij);
            let v = if duk.is_finite() { duk.max(0.0) } else { 0.0 };
            dist[u][k] = v;
            dist[k][u] = v;
        }
        nodes.push(TreeNode::Internal {
            left: best_i,
            right: best_j,
            left_len: li,
            right_len: lj,
        });
        let mut next_active: Vec<usize> = Vec::with_capacity(active.len() - 1);
        for &x in &active {
            if x != best_i && x != best_j {
                next_active.push(x);
            }
        }
        next_active.push(u);

        if use_rapid_nj {
            let mut r_u = 0.0f64;
            for &k in &active {
                if k == best_i || k == best_j {
                    continue;
                }
                let duk = dist[u][k];
                r[k] = r[k] - dist[k][best_i] - dist[k][best_j] + duk;
                r_u += duk;
            }
            if r.len() < dist.len() {
                r.resize(dist.len(), 0.0);
            }
            r[u] = r_u;
            r[best_i] = 0.0;
            r[best_j] = 0.0;
        }

        if use_top_hits {
            if top_hits.len() < dist.len() {
                top_hits.resize_with(dist.len(), Vec::new);
            }
            if active_mask.len() < dist.len() {
                active_mask.resize(dist.len(), false);
            }
            active_mask.fill(false);
            for &x in &next_active {
                active_mask[x] = true;
            }

            for &k in &next_active {
                if k == u {
                    continue;
                }
                let list = &mut top_hits[k];
                list.retain(|&x| x != k && x < active_mask.len() && active_mask[x]);
                if !list.contains(&u) {
                    if list.len() < top_hits_k {
                        list.push(u);
                    } else if top_hits_k > 0 {
                        let mut worst_idx = 0usize;
                        let mut worst_d = f64::NEG_INFINITY;
                        for (idx, &cand) in list.iter().enumerate() {
                            let d = dist[k][cand];
                            if d > worst_d {
                                worst_d = d;
                                worst_idx = idx;
                            }
                        }
                        let du = dist[k][u];
                        if du.is_finite() && du < worst_d {
                            list[worst_idx] = u;
                        }
                    }
                }
                list.sort_by(|&a, &b| {
                    dist[k][a]
                        .partial_cmp(&dist[k][b])
                        .unwrap_or(Ordering::Equal)
                });
                list.dedup();
                if list.len() > top_hits_k {
                    list.truncate(top_hits_k);
                }
                if list.len() < refill_threshold {
                    *list = row_top_k_from_active(&dist, k, &next_active, top_hits_k);
                }
            }
            top_hits[u] = row_top_k_from_active(&dist, u, &next_active, top_hits_k);
        }

        if use_rapid_nj {
            if row_order.len() < dist.len() {
                row_order.resize_with(dist.len(), Vec::new);
            }
            if row_extra.len() < dist.len() {
                row_extra.resize_with(dist.len(), Vec::new);
            }
            if row_ptr.len() < dist.len() {
                row_ptr.resize(dist.len(), 0);
            }
            if row_built.len() < dist.len() {
                row_built.resize(dist.len(), false);
            }
            if row_epoch.len() < dist.len() {
                row_epoch.resize(dist.len(), 0);
            }
            if row_seen.len() < dist.len() {
                row_seen.resize(dist.len(), 0);
            }
            if row_token.len() < dist.len() {
                row_token.resize(dist.len(), 0);
            }

            for &k in &next_active {
                if k != u && row_built[k] {
                    row_extra[k].push(u);
                }
            }
            row_order[u].clear();
            row_extra[u].clear();
            row_ptr[u] = 0;
            row_built[u] = false;
            row_epoch[u] = 0;
            row_seen[u] = 0;

            row_token[best_i] = row_token[best_i].wrapping_add(1);
            row_token[best_j] = row_token[best_j].wrapping_add(1);
            row_token[u] = row_token[u].wrapping_add(1);
            if let Some(h) = rapid_heap.as_mut() {
                h.push(RapidRowEntry {
                    lb: f64::NEG_INFINITY,
                    row: u,
                    epoch: 0,
                    token: row_token[u],
                });
            }
        }

        active = next_active;
        iter_idx += 1;
    }

    let root_id = if active.len() == 2 {
        let a0 = active[0];
        let a1 = active[1];
        let d = if dist[a0][a1].is_finite() {
            dist[a0][a1].max(0.0)
        } else {
            0.0
        };
        let l = 0.5 * d;
        nodes.push(TreeNode::Internal {
            left: a0,
            right: a1,
            left_len: l,
            right_len: l,
        });
        nodes.len() - 1
    } else {
        active[0]
    };

    let mut out = render_newick(root_id, &nodes);
    out.push(';');
    Ok(out)
}
