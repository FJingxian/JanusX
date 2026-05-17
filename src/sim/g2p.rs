use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use numpy::ndarray::Array1;
use numpy::PyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp, Gamma, StandardNormal};

use crate::gfcore as core;
use crate::gfcore::{BedSnpIter, HmpSnpIter, TxtSnpIter, VcfSnpIter};

#[derive(Clone, Debug)]
struct SimSiteRecord {
    chrom: String,
    chrom_norm: String,
    pos: i32,
    ref_allele: String,
    alt_allele: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackgroundDist {
    Normal,
    Gamma,
    Laplace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LogicOp {
    And,
    Or,
}

#[derive(Clone, Debug)]
struct LogicPoolSpec {
    pool_indices: Vec<usize>,
    op: LogicOp,
}

#[derive(Clone, Debug)]
struct CausalTerm {
    members: Vec<usize>,
    op: Option<LogicOp>,
    values: Vec<f64>,
    effect: f64,
    label: String,
}

const SAMPLE_PAR_THRESHOLD: usize = 10_000;
const SAMPLE_PAR_CHUNK: usize = 4096;

struct G2pSimConfig {
    path_or_prefix: String,
    delimiter: Option<String>,
    maf_threshold: f32,
    max_missing_rate: f32,
    seed: u64,
    residual_var: f64,
    bg_pve: f64,
    background_dist: BackgroundDist,
    gamma_shape: f64,
    gamma_scale: f64,
    laplace_scale: f64,
    causal_count: usize,
    causal_pve: Option<f64>,
    bim_ranges: Vec<(String, i32, i32)>,
    logic_mode: Option<String>,
    logic_gate_count: Option<usize>,
    logic_k_min: usize,
    logic_k_max: usize,
    logic_ld_max: f64,
    logic_het_max: f64,
    logic_af_min: f64,
    logic_af_max: f64,
    logic_max_iter: usize,
    logic_window_bp: Option<i32>,
    snps_only: bool,
    pheno_prefix: Option<String>,
    fixed_effects_path: Option<String>,
    random_effects_path: Option<String>,
    causal_sites_path: Option<String>,
    trait_name: Option<String>,
    na_rate: f64,
    progress_callback: Option<Py<PyAny>>,
    progress_total_hint: Option<usize>,
    progress_every: usize,
}

struct G2pSimResult {
    sample_ids: Vec<String>,
    phenotype: Vec<f64>,
    trait_name: String,
    causal_sites: Vec<(String, i32, i32)>,
    fixed_rows: Vec<(usize, String, String, String, String, f64)>,
    n_background_sites: usize,
    n_causal_terms: usize,
    bg_pve: f64,
    causal_pve: f64,
    residual_var: f64,
}

enum SourceReader {
    Bed(BedSnpIter),
    Vcf(VcfSnpIter),
    Hmp(HmpSnpIter),
    Txt(TxtSnpIter),
}

impl SourceReader {
    fn sample_ids(&self) -> &[String] {
        match self {
            Self::Bed(it) => &it.samples,
            Self::Vcf(it) => &it.samples,
            Self::Hmp(it) => &it.samples,
            Self::Txt(it) => &it.samples,
        }
    }

    fn next_row_raw(&mut self) -> Option<(Vec<f32>, core::SiteInfo)> {
        match self {
            Self::Bed(it) => it.next_snp_raw(),
            Self::Vcf(it) => it.next_snp_raw(),
            Self::Hmp(it) => it.next_snp_raw(),
            Self::Txt(it) => it.next_snp(),
        }
    }
}

#[inline]
fn normalize_chrom(chrom: &str) -> String {
    let s = chrom.trim();
    if s.len() >= 3 && s[..3].eq_ignore_ascii_case("chr") {
        s[3..].trim().to_ascii_uppercase()
    } else {
        s.to_ascii_uppercase()
    }
}

#[inline]
fn is_simple_snp_allele(a: &str) -> bool {
    matches!(
        a.trim().to_ascii_uppercase().as_str(),
        "A" | "C" | "G" | "T"
    )
}

#[inline]
fn mean_f64(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

#[inline]
fn variance_f64(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mu = mean_f64(x);
    let mut acc = 0.0_f64;
    for &v in x.iter() {
        let d = v - mu;
        acc += d * d;
    }
    acc / x.len() as f64
}

#[inline]
fn centered_row_to_owned_f64(row: &[f32]) -> Vec<f64> {
    if row.is_empty() {
        return Vec::new();
    }
    let mu = row.iter().map(|&v| v as f64).sum::<f64>() / row.len() as f64;
    let mut out = Vec::with_capacity(row.len());
    for &v in row.iter() {
        out.push(v as f64 - mu);
    }
    out
}

#[inline]
fn axpy_inplace(dst: &mut [f64], src: &[f64], alpha: f64) {
    debug_assert_eq!(dst.len(), src.len());
    if alpha == 0.0 || dst.is_empty() {
        return;
    }
    if dst.len() >= SAMPLE_PAR_THRESHOLD {
        dst.par_chunks_mut(SAMPLE_PAR_CHUNK)
            .zip(src.par_chunks(SAMPLE_PAR_CHUNK))
            .for_each(|(dst_chunk, src_chunk)| {
                for (d, &s) in dst_chunk.iter_mut().zip(src_chunk.iter()) {
                    *d += alpha * s;
                }
            });
    } else {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d += alpha * s;
        }
    }
}

#[inline]
fn add_scaled_centered_row(dst: &mut [f64], row: &[f32], alpha: f64) {
    debug_assert_eq!(dst.len(), row.len());
    if alpha == 0.0 || row.is_empty() {
        return;
    }
    let mu = row.iter().map(|&v| v as f64).sum::<f64>() / row.len() as f64;
    if dst.len() >= SAMPLE_PAR_THRESHOLD {
        dst.par_chunks_mut(SAMPLE_PAR_CHUNK)
            .zip(row.par_chunks(SAMPLE_PAR_CHUNK))
            .for_each(|(dst_chunk, row_chunk)| {
                for (d, &v) in dst_chunk.iter_mut().zip(row_chunk.iter()) {
                    *d += (v as f64 - mu) * alpha;
                }
            });
    } else {
        for (d, &v) in dst.iter_mut().zip(row.iter()) {
            *d += (v as f64 - mu) * alpha;
        }
    }
}

#[inline]
fn background_dist_name(dist: BackgroundDist) -> &'static str {
    match dist {
        BackgroundDist::Normal => "normal",
        BackgroundDist::Gamma => "gamma",
        BackgroundDist::Laplace => "laplace",
    }
}

fn collapse_to_logic_bin01(row: &[f32], het_max: f64) -> Option<Vec<u8>> {
    if row.is_empty() {
        return None;
    }
    let mut valid: Vec<u8> = Vec::with_capacity(row.len());
    let mut valid_idx: Vec<usize> = Vec::with_capacity(row.len());
    for (i, &v) in row.iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            continue;
        }
        let r = v.round();
        let g = if r <= 0.0 {
            0u8
        } else if r >= 2.0 {
            2u8
        } else {
            1u8
        };
        valid.push(g);
        valid_idx.push(i);
    }
    if valid.is_empty() {
        return None;
    }
    let het = valid.iter().filter(|&&g| g == 1).count() as f64 / valid.len() as f64;
    if het > het_max {
        return None;
    }
    let c0 = valid.iter().filter(|&&g| g == 0).count();
    let c2 = valid.iter().filter(|&&g| g == 2).count();
    let mode02 = if c2 > c0 { 2u8 } else { 0u8 };
    let mut out = vec![mode02; row.len()];
    for (&idx, &g) in valid_idx.iter().zip(valid.iter()) {
        out[idx] = if g == 1 { mode02 } else { g };
    }
    Some(
        out.into_iter()
            .map(|g| if g > 0 { 1u8 } else { 0u8 })
            .collect(),
    )
}

#[inline]
fn binary_r2(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i] as f64;
        sb += b[i] as f64;
    }
    let ma = sa / n as f64;
    let mb = sb / n as f64;
    let mut cov = 0.0_f64;
    let mut va = 0.0_f64;
    let mut vb = 0.0_f64;
    for i in 0..n {
        let da = a[i] as f64 - ma;
        let db = b[i] as f64 - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 1e-12 || vb <= 1e-12 {
        return 1.0;
    }
    let r = cov / (va.sqrt() * vb.sqrt());
    let r2 = r * r;
    if r2.is_finite() {
        r2.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn logic_gate_values(rows: &[Vec<u8>], op: LogicOp) -> Vec<f64> {
    if rows.is_empty() {
        return Vec::new();
    }
    let n = rows[0].len();
    let mut raw = vec![0.0_f64; n];
    match op {
        LogicOp::And => {
            for i in 0..n {
                let mut ok = true;
                for row in rows.iter() {
                    if row[i] == 0 {
                        ok = false;
                        break;
                    }
                }
                raw[i] = if ok { 1.0 } else { 0.0 };
            }
        }
        LogicOp::Or => {
            for i in 0..n {
                let mut ok = false;
                for row in rows.iter() {
                    if row[i] != 0 {
                        ok = true;
                        break;
                    }
                }
                raw[i] = if ok { 1.0 } else { 0.0 };
            }
        }
    }
    let mu = mean_f64(&raw);
    raw.into_iter().map(|v| v - mu).collect()
}

fn term_label(sites: &[SimSiteRecord], members: &[usize], op: Option<LogicOp>) -> String {
    let joiner = match op {
        Some(LogicOp::And) => "&",
        Some(LogicOp::Or) => "|",
        None => "",
    };
    let mut parts: Vec<String> = Vec::with_capacity(members.len());
    for &idx in members.iter() {
        let s = &sites[idx];
        parts.push(format!(
            "{}_{}[{}>{}]",
            s.chrom, s.pos, s.ref_allele, s.alt_allele
        ));
    }
    if joiner.is_empty() {
        parts.join("")
    } else {
        parts.join(joiner)
    }
}

fn normalize_plink_prefix_local(p: &str) -> String {
    let low = p.to_ascii_lowercase();
    for ext in [".bed", ".bim", ".fam"] {
        if low.ends_with(ext) {
            let keep = p.len().saturating_sub(ext.len());
            return p[..keep].to_string();
        }
    }
    p.to_string()
}

fn open_source_reader(path_or_prefix: &str, delimiter: Option<&str>) -> Result<SourceReader, String> {
    let p = path_or_prefix.trim();
    if p.is_empty() {
        return Err("path_or_prefix must not be empty".to_string());
    }
    let low = p.to_ascii_lowercase();
    if low.ends_with(".vcf") || low.ends_with(".vcf.gz") {
        return Ok(SourceReader::Vcf(VcfSnpIter::new_with_fill(
            p, 0.0, 1.0, false, false, 0.02,
        )?));
    }
    if low.ends_with(".hmp") || low.ends_with(".hmp.gz") {
        return Ok(SourceReader::Hmp(HmpSnpIter::new_with_fill(
            p, 0.0, 1.0, false, false, 0.02,
        )?));
    }
    if low.ends_with(".txt")
        || low.ends_with(".tsv")
        || low.ends_with(".csv")
        || low.ends_with(".npy")
        || low.ends_with(".bin")
    {
        return Ok(SourceReader::Txt(TxtSnpIter::new(p, delimiter)?));
    }
    let prefix = normalize_plink_prefix_local(p);
    if Path::new(&(prefix.clone() + ".bed")).exists()
        && Path::new(&(prefix.clone() + ".bim")).exists()
        && Path::new(&(prefix.clone() + ".fam")).exists()
    {
        return Ok(SourceReader::Bed(BedSnpIter::new_with_fill(
            &prefix, 0.0, 1.0, false, false, 0.02,
        )?));
    }
    if Path::new(&(p.to_string() + ".npy")).exists()
        || Path::new(&(p.to_string() + ".txt")).exists()
        || Path::new(&(p.to_string() + ".tsv")).exists()
        || Path::new(&(p.to_string() + ".csv")).exists()
        || Path::new(&(p.to_string() + ".bin")).exists()
    {
        return Ok(SourceReader::Txt(TxtSnpIter::new(p, delimiter)?));
    }
    Err(
        "Unable to infer genotype input type. Provide a VCF/HMP path, a PLINK prefix, or a FILE matrix path/prefix."
            .to_string(),
    )
}

fn iterate_filtered_rows<F>(
    path_or_prefix: &str,
    delimiter: Option<&str>,
    maf_threshold: f32,
    max_missing_rate: f32,
    snps_only: bool,
    mut callback: F,
) -> Result<Vec<String>, String>
where
    F: FnMut(usize, &[f32], &core::SiteInfo) -> Result<(), String>,
{
    let mut reader = open_source_reader(path_or_prefix, delimiter)?;
    let sample_ids = reader.sample_ids().to_vec();
    let mut kept = 0usize;
    while let Some((mut row, mut site)) = reader.next_row_raw() {
        let keep = core::process_snp_row(
            &mut row,
            &mut site.ref_allele,
            &mut site.alt_allele,
            maf_threshold,
            max_missing_rate,
            true,
            false,
            0.02,
        );
        if !keep {
            continue;
        }
        if snps_only
            && (!is_simple_snp_allele(&site.ref_allele) || !is_simple_snp_allele(&site.alt_allele))
        {
            continue;
        }
        callback(kept, &row, &site)?;
        kept = kept.saturating_add(1);
    }
    Ok(sample_ids)
}

fn site_in_range(site: &SimSiteRecord, range: &(String, i32, i32)) -> bool {
    site.chrom_norm == normalize_chrom(&range.0) && site.pos >= range.1 && site.pos <= range.2
}

fn build_range_pools(
    sites: &[SimSiteRecord],
    ranges: &[(String, i32, i32)],
) -> Result<Vec<Vec<usize>>, String> {
    let mut out = Vec::with_capacity(ranges.len());
    for (ri, rg) in ranges.iter().enumerate() {
        let mut idx = Vec::new();
        for (i, site) in sites.iter().enumerate() {
            if site_in_range(site, rg) {
                idx.push(i);
            }
        }
        if idx.is_empty() {
            return Err(format!(
                "bimrange[{ri}] has no eligible sites after QC: {}:{}-{}",
                rg.0, rg.1, rg.2
            ));
        }
        out.push(idx);
    }
    Ok(out)
}

fn sample_without_replacement(
    pool: &[usize],
    k: usize,
    rng: &mut StdRng,
) -> Result<Vec<usize>, String> {
    if k > pool.len() {
        return Err(format!(
            "cannot sample {k} unique sites from pool size {}",
            pool.len()
        ));
    }
    let mut tmp = pool.to_vec();
    tmp.shuffle(rng);
    tmp.truncate(k);
    Ok(tmp)
}

fn select_additive_indices(
    sites: &[SimSiteRecord],
    causal_count: usize,
    ranges: &[(String, i32, i32)],
    rng: &mut StdRng,
) -> Result<Vec<usize>, String> {
    if sites.is_empty() {
        return Err("no eligible sites remain after QC".to_string());
    }
    if causal_count == 0 && ranges.is_empty() {
        return Ok(Vec::new());
    }
    let mut selected: Vec<usize> = Vec::new();
    let mut used: HashSet<usize> = HashSet::new();
    let pools = build_range_pools(sites, ranges)?;
    for pool in pools.iter() {
        let avail: Vec<usize> = pool
            .iter()
            .copied()
            .filter(|idx| !used.contains(idx))
            .collect();
        if avail.is_empty() {
            return Err("explicit bimrange pools overlap completely; no unique causal site can be drawn".to_string());
        }
        let pick = avail[rng.random_range(0..avail.len())];
        used.insert(pick);
        selected.push(pick);
    }
    let target = causal_count.max(selected.len());
    if target > sites.len() {
        return Err(format!(
            "requested causal sites exceed eligible site count: target={target}, eligible={}",
            sites.len()
        ));
    }
    if selected.len() < target {
        let mut rest: Vec<usize> = (0..sites.len()).filter(|idx| !used.contains(idx)).collect();
        rest.shuffle(rng);
        for idx in rest.into_iter().take(target - selected.len()) {
            used.insert(idx);
            selected.push(idx);
        }
    }
    if selected.len() != target {
        return Err(format!(
            "unable to draw enough unique causal sites: target={target}, got={}",
            selected.len()
        ));
    }
    selected.sort_unstable();
    Ok(selected)
}

fn reservoir_sample(mut pool: Vec<usize>, cap: usize, rng: &mut StdRng) -> Vec<usize> {
    if pool.len() <= cap {
        return pool;
    }
    pool.shuffle(rng);
    pool.truncate(cap);
    pool
}

fn logic_op_from_mode(mode: &str, rng: &mut StdRng) -> Result<LogicOp, String> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "and" => Ok(LogicOp::And),
        "or" => Ok(LogicOp::Or),
        "and_or" => {
            if rng.random::<bool>() {
                Ok(LogicOp::And)
            } else {
                Ok(LogicOp::Or)
            }
        }
        other => Err(format!("unsupported logic mode: {other}")),
    }
}

fn build_logic_pool_specs(
    sites: &[SimSiteRecord],
    ranges: &[(String, i32, i32)],
    causal_count: usize,
    logic_gate_count: Option<usize>,
    logic_mode: &str,
    logic_k_min: usize,
    logic_window_bp: Option<i32>,
    rng: &mut StdRng,
) -> Result<Vec<LogicPoolSpec>, String> {
    if sites.is_empty() {
        return Err("no eligible sites remain after QC".to_string());
    }
    if logic_k_min == 0 {
        return Err("logic_k_min must be > 0".to_string());
    }
    let mut out: Vec<LogicPoolSpec> = Vec::new();
    let range_pools = build_range_pools(sites, ranges)?;
    for pool in range_pools.into_iter() {
        if pool.len() < logic_k_min {
            return Err(format!(
                "an explicit bimrange does not contain enough sites for logic gate size: required >= {logic_k_min}, got {}",
                pool.len()
            ));
        }
        out.push(LogicPoolSpec {
            pool_indices: reservoir_sample(pool, 1024, rng),
            op: logic_op_from_mode(logic_mode, rng)?,
        });
    }
    let target = logic_gate_count.unwrap_or(causal_count.max(1)).max(out.len());
    while out.len() < target {
        let pool = if let Some(window_bp) = logic_window_bp {
            let mut tries = 0usize;
            let mut found: Option<Vec<usize>> = None;
            while tries < 128 {
                tries += 1;
                let anchor_idx = rng.random_range(0..sites.len());
                let anchor = &sites[anchor_idx];
                let lo = anchor.pos.saturating_sub(window_bp);
                let hi = anchor.pos.saturating_add(window_bp);
                let mut cand = Vec::new();
                for (i, site) in sites.iter().enumerate() {
                    if site.chrom_norm == anchor.chrom_norm && site.pos >= lo && site.pos <= hi {
                        cand.push(i);
                    }
                }
                if cand.len() >= logic_k_min {
                    found = Some(reservoir_sample(cand, 1024, rng));
                    break;
                }
            }
            found.unwrap_or_else(|| reservoir_sample((0..sites.len()).collect(), 1024, rng))
        } else {
            reservoir_sample((0..sites.len()).collect(), 1024, rng)
        };
        if pool.len() < logic_k_min {
            return Err(format!(
                "unable to build a logic-gate candidate pool with at least {logic_k_min} sites"
            ));
        }
        out.push(LogicPoolSpec {
            pool_indices: pool,
            op: logic_op_from_mode(logic_mode, rng)?,
        });
    }
    Ok(out)
}

fn draw_background_effect(
    dist: BackgroundDist,
    gamma: Option<&Gamma<f64>>,
    laplace: Option<&Exp<f64>>,
    rng: &mut StdRng,
) -> f64 {
    match dist {
        BackgroundDist::Normal => StandardNormal.sample(rng),
        BackgroundDist::Gamma => {
            let mag = gamma
                .expect("gamma distribution must be initialized")
                .sample(rng);
            if rng.random::<bool>() {
                mag
            } else {
                -mag
            }
        }
        BackgroundDist::Laplace => {
            let mag = laplace
                .expect("laplace exponential distribution must be initialized")
                .sample(rng);
            if rng.random::<bool>() {
                mag
            } else {
                -mag
            }
        }
    }
}

fn select_logic_terms(
    pool_specs: &[LogicPoolSpec],
    row_map: &HashMap<usize, Vec<f32>>,
    sites: &[SimSiteRecord],
    logic_k_min: usize,
    logic_k_max: usize,
    logic_ld_max: f64,
    logic_het_max: f64,
    logic_af_min: f64,
    logic_af_max: f64,
    logic_max_iter: usize,
    rng: &mut StdRng,
) -> Result<Vec<CausalTerm>, String> {
    if logic_k_min == 0 {
        return Err("logic_k_min must be > 0".to_string());
    }
    if logic_k_max < logic_k_min {
        return Err("logic_k_max must be >= logic_k_min".to_string());
    }
    let af_center = 0.5 * (logic_af_min + logic_af_max);
    let mut used_global: HashSet<usize> = HashSet::new();
    let mut out = Vec::with_capacity(pool_specs.len());

    for (ti, spec) in pool_specs.iter().enumerate() {
        let mut bin_map: HashMap<usize, Vec<u8>> = HashMap::new();
        for &idx in spec.pool_indices.iter() {
            if let Some(row) = row_map.get(&idx) {
                if let Some(bin) = collapse_to_logic_bin01(row, logic_het_max) {
                    bin_map.insert(idx, bin);
                }
            }
        }
        if bin_map.len() < logic_k_min {
            return Err(format!(
                "logic-gate candidate pool {ti} has too few usable sites after heterozygosity filtering: need >= {logic_k_min}, got {}",
                bin_map.len()
            ));
        }

        let mut best: Option<(Vec<usize>, Vec<f64>, f64)> = None;
        for _ in 0..logic_max_iter.max(1) {
            let prefer_unused: Vec<usize> = spec
                .pool_indices
                .iter()
                .copied()
                .filter(|idx| bin_map.contains_key(idx) && !used_global.contains(idx))
                .collect();
            let all_avail: Vec<usize> = spec
                .pool_indices
                .iter()
                .copied()
                .filter(|idx| bin_map.contains_key(idx))
                .collect();
            let pool = if prefer_unused.len() >= logic_k_min {
                prefer_unused
            } else {
                all_avail
            };
            if pool.len() < logic_k_min {
                break;
            }
            let k_hi = logic_k_max.min(pool.len());
            let k = if k_hi == logic_k_min {
                logic_k_min
            } else {
                rng.random_range(logic_k_min..=k_hi)
            };
            let members = sample_without_replacement(&pool, k, rng)?;
            let mut ld_ok = true;
            if logic_ld_max < 0.999_999 {
                for a in 0..members.len() {
                    for b in (a + 1)..members.len() {
                        let r2 = binary_r2(
                            bin_map
                                .get(&members[a])
                                .ok_or_else(|| "missing logic row".to_string())?,
                            bin_map
                                .get(&members[b])
                                .ok_or_else(|| "missing logic row".to_string())?,
                        );
                        if r2 > logic_ld_max + 1e-12 {
                            ld_ok = false;
                            break;
                        }
                    }
                    if !ld_ok {
                        break;
                    }
                }
            }
            if !ld_ok {
                continue;
            }
            let gate_rows: Vec<Vec<u8>> = members
                .iter()
                .map(|idx| {
                    bin_map
                        .get(idx)
                        .cloned()
                        .ok_or_else(|| "missing logic row".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let gate = logic_gate_values(&gate_rows, spec.op);
            let raw_af = gate.iter().filter(|&&v| v > 0.0).count() as f64 / gate.len() as f64;
            let var = variance_f64(&gate);
            if var <= 1e-12 {
                continue;
            }
            if best
                .as_ref()
                .map(|(_, _, af)| (raw_af - af_center).abs() < (*af - af_center).abs())
                .unwrap_or(true)
            {
                best = Some((members.clone(), gate.clone(), raw_af));
            }
            if raw_af >= logic_af_min && raw_af <= logic_af_max {
                let label = term_label(sites, &members, Some(spec.op));
                for idx in members.iter() {
                    used_global.insert(*idx);
                }
                out.push(CausalTerm {
                    members,
                    op: Some(spec.op),
                    values: gate,
                    effect: 0.0,
                    label,
                });
                break;
            }
        }

        if out.len() != ti + 1 {
            if let Some((members, gate, _af)) = best {
                let label = term_label(sites, &members, Some(spec.op));
                for idx in members.iter() {
                    used_global.insert(*idx);
                }
                out.push(CausalTerm {
                    members,
                    op: Some(spec.op),
                    values: gate,
                    effect: 0.0,
                    label,
                });
            } else {
                return Err(format!(
                    "unable to build a valid logic gate for pool {ti}; try relaxing gate size / LD / AF / heterozygosity constraints"
                ));
            }
        }
    }
    Ok(out)
}

fn build_additive_terms(
    selected_indices: &[usize],
    row_map: &HashMap<usize, Vec<f32>>,
    sites: &[SimSiteRecord],
) -> Result<Vec<CausalTerm>, String> {
    let mut out = Vec::with_capacity(selected_indices.len());
    for &idx in selected_indices.iter() {
        let row = row_map
            .get(&idx)
            .ok_or_else(|| format!("selected causal site row missing for index {idx}"))?;
        let values = centered_row_to_owned_f64(row);
        if variance_f64(&values) <= 1e-12 {
            return Err(format!(
                "selected causal site has zero variance after centering: {}:{}",
                sites[idx].chrom, sites[idx].pos
            ));
        }
        out.push(CausalTerm {
            members: vec![idx],
            op: None,
            values,
            effect: 0.0,
            label: term_label(sites, &[idx], None),
        });
    }
    Ok(out)
}

fn write_pheno_files(
    prefix: &str,
    sample_ids: &[String],
    y: &[f64],
    trait_name: &str,
    na_rate: f64,
    seed: u64,
) -> Result<(), String> {
    if sample_ids.len() != y.len() {
        return Err(format!(
            "sample/phenotype length mismatch: ids={}, y={}",
            sample_ids.len(),
            y.len()
        ));
    }
    let pheno_path = format!("{prefix}.pheno");
    let pheno_txt_path = format!("{prefix}.pheno.txt");
    let pheno_na_path = format!("{prefix}.pheno.NA.txt");

    let mut w3 = BufWriter::new(File::create(&pheno_path).map_err(|e| e.to_string())?);
    for (sid, &yv) in sample_ids.iter().zip(y.iter()) {
        writeln!(w3, "{sid}\t{sid}\t{yv:.6}").map_err(|e| e.to_string())?;
    }
    w3.flush().map_err(|e| e.to_string())?;

    let mut w2 = BufWriter::new(File::create(&pheno_txt_path).map_err(|e| e.to_string())?);
    writeln!(w2, "IID\t{trait_name}").map_err(|e| e.to_string())?;
    for (sid, &yv) in sample_ids.iter().zip(y.iter()) {
        writeln!(w2, "{sid}\t{yv:.6}").map_err(|e| e.to_string())?;
    }
    w2.flush().map_err(|e| e.to_string())?;

    let mut na_idx: Vec<usize> = (0..sample_ids.len()).collect();
    let mut rng = StdRng::seed_from_u64(seed ^ 0xD9A3_5C71_4B1E_208Du64);
    na_idx.shuffle(&mut rng);
    let k = ((sample_ids.len() as f64) * na_rate.clamp(0.0, 1.0)).round() as usize;
    let na_set: HashSet<usize> = na_idx.into_iter().take(k).collect();
    let mut wna = BufWriter::new(File::create(&pheno_na_path).map_err(|e| e.to_string())?);
    writeln!(wna, "IID\t{trait_name}").map_err(|e| e.to_string())?;
    for (i, sid) in sample_ids.iter().enumerate() {
        if na_set.contains(&i) {
            writeln!(wna, "{sid}\tNA").map_err(|e| e.to_string())?;
        } else {
            writeln!(wna, "{sid}\t{:.6}", y[i]).map_err(|e| e.to_string())?;
        }
    }
    wna.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn write_random_effects(
    path: &str,
    sites: &[SimSiteRecord],
    effects: &[f64],
) -> Result<(), String> {
    if sites.len() != effects.len() {
        return Err(format!(
            "random effect metadata length mismatch: sites={}, effects={}",
            sites.len(),
            effects.len()
        ));
    }
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    writeln!(w, "chrom\tpos\tref\talt\teffect").map_err(|e| e.to_string())?;
    for (site, &eff) in sites.iter().zip(effects.iter()) {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{eff:.10}",
            site.chrom, site.pos, site.ref_allele, site.alt_allele
        )
        .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn write_fixed_effects(path: &str, terms: &[CausalTerm], sites: &[SimSiteRecord]) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    writeln!(w, "term_id\tkind\tlogic\tsites\tlabel\teffect").map_err(|e| e.to_string())?;
    for (i, term) in terms.iter().enumerate() {
        let kind = if term.op.is_some() { "logic_gate" } else { "additive" };
        let logic = match term.op {
            Some(LogicOp::And) => "and",
            Some(LogicOp::Or) => "or",
            None => "single",
        };
        let site_text = term
            .members
            .iter()
            .map(|&idx| format!("{}:{}", sites[idx].chrom, sites[idx].pos))
            .collect::<Vec<String>>()
            .join(";");
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{:.10}",
            i + 1,
            kind,
            logic,
            site_text,
            term.label,
            term.effect
        )
        .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn write_causal_sites(path: &str, terms: &[CausalTerm], sites: &[SimSiteRecord]) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    let mut seen: HashSet<(String, i32)> = HashSet::new();
    for term in terms.iter() {
        for &idx in term.members.iter() {
            let site = &sites[idx];
            let key = (site.chrom.clone(), site.pos);
            if !seen.insert(key) {
                continue;
            }
            writeln!(w, "{}\t{}\t{}", site.chrom, site.pos, site.pos).map_err(|e| e.to_string())?;
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn parse_background_dist(name: &str) -> Result<BackgroundDist, String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "normal" | "gaussian" => Ok(BackgroundDist::Normal),
        "gamma" => Ok(BackgroundDist::Gamma),
        "laplace" => Ok(BackgroundDist::Laplace),
        other => Err(format!("unsupported background distribution: {other}")),
    }
}

fn g2p_progress_notify(
    progress_callback: Option<&Py<PyAny>>,
    stage: &str,
    done: usize,
    total: usize,
    notify_step: usize,
    last_notified: &mut usize,
    force: bool,
) -> Result<(), String> {
    let done_clamped = if total > 0 { done.min(total) } else { done };
    if !force && done_clamped < last_notified.saturating_add(notify_step.max(1)) {
        return Ok(());
    }
    *last_notified = done_clamped;
    Python::attach(|py2| -> PyResult<()> {
        py2.check_signals()?;
        if let Some(cb) = progress_callback {
            cb.call1(py2, (stage, done_clamped, total))?;
        }
        Ok(())
    })
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn g2p_simulate_core(config: G2pSimConfig) -> Result<G2pSimResult, String> {
    if !(0.0..=1.0).contains(&config.bg_pve) {
        return Err("bg_pve must be within [0, 1].".to_string());
    }
    if !(0.0..=1.0).contains(&config.na_rate) {
        return Err("na_rate must be within [0, 1].".to_string());
    }
    if config.residual_var < 0.0 || !config.residual_var.is_finite() {
        return Err("residual_var must be finite and >= 0.".to_string());
    }
    if config.logic_k_min == 0 {
        return Err("logic_k_min must be > 0.".to_string());
    }
    if config.logic_k_max < config.logic_k_min {
        return Err("logic_k_max must be >= logic_k_min.".to_string());
    }
    if !(0.0..=1.0).contains(&config.logic_ld_max) {
        return Err("logic_ld_max must be within [0, 1].".to_string());
    }
    if !(0.0..=1.0).contains(&config.logic_het_max) {
        return Err("logic_het_max must be within [0, 1].".to_string());
    }
    if !(0.0..=1.0).contains(&config.logic_af_min)
        || !(0.0..=1.0).contains(&config.logic_af_max)
    {
        return Err("logic_af_min/logic_af_max must be within [0, 1].".to_string());
    }
    if config.logic_af_min > config.logic_af_max {
        return Err("logic_af_min must be <= logic_af_max.".to_string());
    }

    let gamma_dist = if matches!(config.background_dist, BackgroundDist::Gamma) {
        Some(Gamma::new(config.gamma_shape, config.gamma_scale).map_err(|e| {
            format!("invalid gamma params: {e}")
        })?)
    } else {
        None
    };
    let laplace_exp = if matches!(config.background_dist, BackgroundDist::Laplace) {
        if !(config.laplace_scale > 0.0) || !config.laplace_scale.is_finite() {
            return Err("laplace_scale must be finite and > 0.".to_string());
        }
        Some(Exp::new(1.0 / config.laplace_scale).map_err(|e| {
            format!("invalid laplace scale: {e}")
        })?)
    } else {
        None
    };

    let logic_requested = config
        .logic_mode
        .as_ref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    let base_term_count = if logic_requested {
        config.logic_gate_count.unwrap_or(config.causal_count.max(1))
    } else {
        config.causal_count
    };
    let effective_term_count = base_term_count.max(config.bim_ranges.len());
    let causal_pve_target = config.causal_pve.unwrap_or_else(|| {
        if effective_term_count == 0 {
            0.0
        } else {
            (0.05_f64 * effective_term_count as f64).min(0.95)
        }
    });
    if !(0.0..=1.0).contains(&causal_pve_target) {
        return Err("causal_pve must be within [0, 1].".to_string());
    }

    let residual_var_eff = 1.0 - config.bg_pve - causal_pve_target;
    if residual_var_eff < -1e-12 {
        return Err(
            "bg_pve + causal_pve must be <= 1.0 under the final-variance PVE definition."
                .to_string(),
        );
    }
    let residual_var_eff = residual_var_eff.max(0.0);
    let needs_causal_scan = effective_term_count > 0 && causal_pve_target > 0.0;
    let scan_passes = 1usize + if needs_causal_scan { 1 } else { 0 };
    let progress_site_total = config.progress_total_hint.unwrap_or(0);
    let progress_overall_total = progress_site_total.saturating_mul(scan_passes);
    let progress_every = config.progress_every.max(1);

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut sites: Vec<SimSiteRecord> = Vec::new();
    let mut bg_raw_effects: Vec<f64> = Vec::new();
    let mut bg_score: Vec<f64> = Vec::new();
    let mut bg_progress_seen = 0usize;
    let mut bg_last_notified = 0usize;

    let sample_ids = iterate_filtered_rows(
        &config.path_or_prefix,
        config.delimiter.as_deref(),
        config.maf_threshold,
        config.max_missing_rate,
        config.snps_only,
        |_, row, site| {
            bg_progress_seen = bg_progress_seen.saturating_add(1);
            if progress_overall_total > 0 {
                g2p_progress_notify(
                    config.progress_callback.as_ref(),
                    "background",
                    bg_progress_seen,
                    progress_overall_total,
                    progress_every,
                    &mut bg_last_notified,
                    false,
                )?;
            }
            if bg_score.is_empty() {
                bg_score.resize(row.len(), 0.0);
            }
            let draw = if config.bg_pve > 0.0 {
                draw_background_effect(
                    config.background_dist,
                    gamma_dist.as_ref(),
                    laplace_exp.as_ref(),
                    &mut rng,
                )
            } else {
                0.0
            };
            add_scaled_centered_row(&mut bg_score, row, draw);
            bg_raw_effects.push(draw);
            sites.push(SimSiteRecord {
                chrom: site.chrom.clone(),
                chrom_norm: normalize_chrom(&site.chrom),
                pos: site.pos,
                ref_allele: site.ref_allele.clone(),
                alt_allele: site.alt_allele.clone(),
            });
            Ok(())
        },
    )?;
    if progress_overall_total > 0 {
        g2p_progress_notify(
            config.progress_callback.as_ref(),
            "background",
            progress_site_total,
            progress_overall_total,
            progress_every,
            &mut bg_last_notified,
            true,
        )?;
    }

    if sample_ids.is_empty() {
        return Err("no samples found in genotype input after inspection".to_string());
    }
    if sites.is_empty() {
        return Err("no eligible variants remain after QC filtering".to_string());
    }

    let n = sample_ids.len();
    let mut y = vec![0.0_f64; n];
    if residual_var_eff > 0.0 {
        let sd = residual_var_eff.sqrt();
        for yi in y.iter_mut() {
            let z: f64 = StandardNormal.sample(&mut rng);
            *yi = z * sd;
        }
    }
    let bg_var_target = config.bg_pve;
    let bg_score_var = variance_f64(&bg_score);
    let bg_scale = if bg_var_target > 0.0 && bg_score_var > 1e-12 {
        (bg_var_target / bg_score_var).sqrt()
    } else {
        0.0
    };
    axpy_inplace(&mut y, &bg_score, bg_scale);
    let bg_effects_scaled: Vec<f64> = bg_raw_effects.iter().map(|&v| v * bg_scale).collect();

    let mut causal_terms: Vec<CausalTerm> = Vec::new();
    if needs_causal_scan {
        let mut causal_progress_seen = 0usize;
        let mut causal_last_notified = 0usize;
        let causal_offset = progress_site_total;
        if logic_requested {
            let logic_mode_str = config.logic_mode.as_deref().unwrap_or("and");
            let mut pool_rng = StdRng::seed_from_u64(config.seed ^ 0xB28F_6A91_C547_31D1u64);
            let pool_specs = build_logic_pool_specs(
                &sites,
                &config.bim_ranges,
                config.causal_count,
                config.logic_gate_count,
                logic_mode_str,
                config.logic_k_min,
                config.logic_window_bp,
                &mut pool_rng,
            )?;
            let mut need_rows: HashSet<usize> = HashSet::with_capacity(1024);
            for spec in pool_specs.iter() {
                for &idx in spec.pool_indices.iter() {
                    need_rows.insert(idx);
                }
            }
            let mut row_map: HashMap<usize, Vec<f32>> = HashMap::with_capacity(need_rows.len());
            iterate_filtered_rows(
                &config.path_or_prefix,
                config.delimiter.as_deref(),
                config.maf_threshold,
                config.max_missing_rate,
                config.snps_only,
                |kept_idx, row, _site| {
                    causal_progress_seen = causal_progress_seen.saturating_add(1);
                    if progress_overall_total > 0 {
                        g2p_progress_notify(
                            config.progress_callback.as_ref(),
                            "causal_logic",
                            causal_offset.saturating_add(causal_progress_seen),
                            progress_overall_total,
                            progress_every,
                            &mut causal_last_notified,
                            false,
                        )?;
                    }
                    if need_rows.contains(&kept_idx) {
                        row_map.insert(kept_idx, row.to_vec());
                    }
                    Ok(())
                },
            )?;
            if progress_overall_total > 0 {
                g2p_progress_notify(
                    config.progress_callback.as_ref(),
                    "causal_logic",
                    causal_offset.saturating_add(progress_site_total),
                    progress_overall_total,
                    progress_every,
                    &mut causal_last_notified,
                    true,
                )?;
            }
            let mut logic_rng = StdRng::seed_from_u64(config.seed ^ 0x9F4B_72E0_1A33_4F0Cu64);
            causal_terms = select_logic_terms(
                &pool_specs,
                &row_map,
                &sites,
                config.logic_k_min,
                config.logic_k_max,
                config.logic_ld_max,
                config.logic_het_max,
                config.logic_af_min,
                config.logic_af_max,
                config.logic_max_iter,
                &mut logic_rng,
            )?;
        } else {
            let mut sel_rng = StdRng::seed_from_u64(config.seed ^ 0xA54D_3F9E_6721_8CB7u64);
            let selected = select_additive_indices(
                &sites,
                config.causal_count,
                &config.bim_ranges,
                &mut sel_rng,
            )?;
            let selected_set: HashSet<usize> = selected.iter().copied().collect();
            let mut row_map: HashMap<usize, Vec<f32>> = HashMap::with_capacity(selected.len());
            iterate_filtered_rows(
                &config.path_or_prefix,
                config.delimiter.as_deref(),
                config.maf_threshold,
                config.max_missing_rate,
                config.snps_only,
                |kept_idx, row, _site| {
                    causal_progress_seen = causal_progress_seen.saturating_add(1);
                    if progress_overall_total > 0 {
                        g2p_progress_notify(
                            config.progress_callback.as_ref(),
                            "causal_additive",
                            causal_offset.saturating_add(causal_progress_seen),
                            progress_overall_total,
                            progress_every,
                            &mut causal_last_notified,
                            false,
                        )?;
                    }
                    if selected_set.contains(&kept_idx) {
                        row_map.insert(kept_idx, row.to_vec());
                    }
                    Ok(())
                },
            )?;
            if progress_overall_total > 0 {
                g2p_progress_notify(
                    config.progress_callback.as_ref(),
                    "causal_additive",
                    causal_offset.saturating_add(progress_site_total),
                    progress_overall_total,
                    progress_every,
                    &mut causal_last_notified,
                    true,
                )?;
            }
            causal_terms = build_additive_terms(&selected, &row_map, &sites)?;
        }
    }

    if !causal_terms.is_empty() && causal_pve_target > 0.0 {
        let mut gamma0: Vec<f64> = Vec::with_capacity(causal_terms.len());
        for _ in 0..causal_terms.len() {
            gamma0.push(rng.random_range(-1.0_f64..1.0_f64));
        }
        let mut causal_score = vec![0.0_f64; n];
        for (coef, term) in gamma0.iter().zip(causal_terms.iter()) {
            axpy_inplace(&mut causal_score, &term.values, *coef);
        }
        let var_causal_raw = variance_f64(&causal_score);
        if causal_pve_target > 0.0 && var_causal_raw > 1e-12 {
            let scale = (causal_pve_target / var_causal_raw).sqrt();
            for (coef, term) in gamma0.iter().zip(causal_terms.iter_mut()) {
                term.effect = scale * *coef;
            }
            for term in causal_terms.iter() {
                axpy_inplace(&mut y, &term.values, term.effect);
            }
        }
    }

    let default_trait = format!(
        "sim_bg{:.3}_cs{:.3}_{}",
        config.bg_pve,
        causal_pve_target,
        background_dist_name(config.background_dist)
    );
    let trait_name = config
        .trait_name
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or(default_trait);

    if let Some(prefix) = config.pheno_prefix.as_ref() {
        write_pheno_files(prefix, &sample_ids, &y, &trait_name, config.na_rate, config.seed)?;
    }
    if let Some(path) = config.random_effects_path.as_ref() {
        write_random_effects(path, &sites, &bg_effects_scaled)?;
    }
    if let Some(path) = config.fixed_effects_path.as_ref() {
        write_fixed_effects(path, &causal_terms, &sites)?;
    }
    if let Some(path) = config.causal_sites_path.as_ref() {
        write_causal_sites(path, &causal_terms, &sites)?;
    }
    if progress_overall_total > 0 {
        let mut final_last_notified = progress_overall_total;
        g2p_progress_notify(
            config.progress_callback.as_ref(),
            "finalize",
            progress_overall_total,
            progress_overall_total,
            progress_every,
            &mut final_last_notified,
            true,
        )?;
    }

    let causal_sites: Vec<(String, i32, i32)> = {
        let mut seen: HashSet<(String, i32)> = HashSet::new();
        let mut out: Vec<(String, i32, i32)> = Vec::new();
        for term in causal_terms.iter() {
            for &idx in term.members.iter() {
                let site = &sites[idx];
                let key = (site.chrom.clone(), site.pos);
                if seen.insert(key.clone()) {
                    out.push((key.0, key.1, key.1));
                }
            }
        }
        out
    };

    let fixed_rows: Vec<(usize, String, String, String, String, f64)> = causal_terms
        .iter()
        .enumerate()
        .map(|(i, term)| {
            let kind = if term.op.is_some() {
                "logic_gate".to_string()
            } else {
                "additive".to_string()
            };
            let logic = match term.op {
                Some(LogicOp::And) => "and".to_string(),
                Some(LogicOp::Or) => "or".to_string(),
                None => "single".to_string(),
            };
            let site_text = term
                .members
                .iter()
                .map(|&idx| format!("{}:{}", sites[idx].chrom, sites[idx].pos))
                .collect::<Vec<String>>()
                .join(";");
            (i + 1, kind, logic, site_text, term.label.clone(), term.effect)
        })
        .collect();

    Ok(G2pSimResult {
        sample_ids,
        phenotype: y,
        trait_name,
        causal_sites,
        fixed_rows,
        n_background_sites: sites.len(),
        n_causal_terms: causal_terms.len(),
        bg_pve: config.bg_pve,
        causal_pve: causal_pve_target,
        residual_var: residual_var_eff,
    })
}

#[pyfunction(name = "g2p_simulate")]
#[pyo3(signature = (
    path_or_prefix,
    chunk_size=100_000,
    maf_threshold=0.02_f32,
    max_missing_rate=0.05_f32,
    seed=1_u64,
    residual_var=1.0_f64,
    bg_pve=0.5_f64,
    background_dist="normal",
    gamma_shape=1.0_f64,
    gamma_scale=1.0_f64,
    laplace_scale=1.0_f64,
    causal_count=1_usize,
    causal_pve=None,
    bim_ranges=None,
    logic_mode=None,
    logic_gate_count=None,
    logic_k_min=2_usize,
    logic_k_max=2_usize,
    logic_ld_max=1.0_f64,
    logic_het_max=1.0_f64,
    logic_af_min=0.0_f64,
    logic_af_max=1.0_f64,
    logic_max_iter=256_usize,
    logic_window_bp=None,
    delimiter=None,
    snps_only=true,
    pheno_prefix=None,
    fixed_effects_path=None,
    random_effects_path=None,
    causal_sites_path=None,
    trait_name=None,
    na_rate=0.1_f64,
    progress_callback=None,
    progress_total_hint=None,
    progress_every=10_000_usize,
))]
pub fn g2p_simulate_py<'py>(
    py: Python<'py>,
    path_or_prefix: String,
    chunk_size: usize,
    maf_threshold: f32,
    max_missing_rate: f32,
    seed: u64,
    residual_var: f64,
    bg_pve: f64,
    background_dist: &str,
    gamma_shape: f64,
    gamma_scale: f64,
    laplace_scale: f64,
    causal_count: usize,
    causal_pve: Option<f64>,
    bim_ranges: Option<Vec<(String, i32, i32)>>,
    logic_mode: Option<String>,
    logic_gate_count: Option<usize>,
    logic_k_min: usize,
    logic_k_max: usize,
    logic_ld_max: f64,
    logic_het_max: f64,
    logic_af_min: f64,
    logic_af_max: f64,
    logic_max_iter: usize,
    logic_window_bp: Option<i32>,
    delimiter: Option<String>,
    snps_only: bool,
    pheno_prefix: Option<String>,
    fixed_effects_path: Option<String>,
    random_effects_path: Option<String>,
    causal_sites_path: Option<String>,
    trait_name: Option<String>,
    na_rate: f64,
    progress_callback: Option<Py<PyAny>>,
    progress_total_hint: Option<usize>,
    progress_every: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if !(0.0..=1.0).contains(&bg_pve) {
        return Err(PyValueError::new_err("bg_pve must be within [0, 1]."));
    }
    if !(0.0..=1.0).contains(&na_rate) {
        return Err(PyValueError::new_err("na_rate must be within [0, 1]."));
    }
    if residual_var < 0.0 || !residual_var.is_finite() {
        return Err(PyValueError::new_err(
            "residual_var must be finite and >= 0.",
        ));
    }
    if logic_k_min == 0 {
        return Err(PyValueError::new_err("logic_k_min must be > 0."));
    }
    if logic_k_max < logic_k_min {
        return Err(PyValueError::new_err(
            "logic_k_max must be >= logic_k_min.",
        ));
    }
    if !(0.0..=1.0).contains(&logic_ld_max) {
        return Err(PyValueError::new_err("logic_ld_max must be within [0, 1]."));
    }
    if !(0.0..=1.0).contains(&logic_het_max) {
        return Err(PyValueError::new_err(
            "logic_het_max must be within [0, 1].",
        ));
    }
    if !(0.0..=1.0).contains(&logic_af_min) || !(0.0..=1.0).contains(&logic_af_max) {
        return Err(PyValueError::new_err(
            "logic_af_min/logic_af_max must be within [0, 1].",
        ));
    }
    if logic_af_min > logic_af_max {
        return Err(PyValueError::new_err(
            "logic_af_min must be <= logic_af_max.",
        ));
    }

    let _ = chunk_size;
    let bg_dist =
        parse_background_dist(background_dist).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let config = G2pSimConfig {
        path_or_prefix,
        delimiter,
        maf_threshold,
        max_missing_rate,
        seed,
        residual_var,
        bg_pve,
        background_dist: bg_dist,
        gamma_shape,
        gamma_scale,
        laplace_scale,
        causal_count,
        causal_pve,
        bim_ranges: bim_ranges.unwrap_or_default(),
        logic_mode,
        logic_gate_count,
        logic_k_min,
        logic_k_max,
        logic_ld_max,
        logic_het_max,
        logic_af_min,
        logic_af_max,
        logic_max_iter,
        logic_window_bp,
        snps_only,
        pheno_prefix,
        fixed_effects_path,
        random_effects_path,
        causal_sites_path,
        trait_name,
        na_rate,
        progress_callback,
        progress_total_hint,
        progress_every,
    };
    let sim = py
        .detach(move || g2p_simulate_core(config))
        .map_err(PyRuntimeError::new_err)?;

    let G2pSimResult {
        sample_ids,
        phenotype,
        trait_name,
        causal_sites,
        fixed_rows,
        n_background_sites,
        n_causal_terms,
        bg_pve,
        causal_pve,
        residual_var,
    } = sim;

    #[allow(deprecated)]
    let y_arr = PyArray1::from_owned_array(py, Array1::from_vec(phenotype));
    let out = PyDict::new(py);
    out.set_item("sample_ids", sample_ids)?;
    out.set_item("phenotype", y_arr)?;
    out.set_item("trait_name", trait_name)?;
    out.set_item("causal_sites", causal_sites)?;
    out.set_item("fixed_rows", fixed_rows)?;
    out.set_item("n_background_sites", n_background_sites)?;
    out.set_item("n_causal_terms", n_causal_terms)?;
    out.set_item("bg_pve", bg_pve)?;
    out.set_item("causal_pve", causal_pve)?;
    out.set_item("ve", residual_var)?;
    out.set_item("residual_var", residual_var)?;
    Ok(out)
}
