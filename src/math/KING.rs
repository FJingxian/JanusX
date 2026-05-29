// KING (Kinship-based INference for GWAS) implementation in Rust.
// Ani Manichaikul, Josyf C. Mychaleckyj, Stephen S. Rich, Kathy Daly, Michèle Sale, Wei-Min Chen,
// Robust relationship inference in genome-wide association studies,
// Bioinformatics, Volume 26, Issue 22, November 2010, Pages 2867–2873, https://doi.org/10.1093/bioinformatics/btq559
use rayon::prelude::*;
use std::collections::BinaryHeap;

use crate::bedmath::packed_byte_lut;
use crate::bitwise::{king_pair_counts, KingBitCounts};
use crate::gfreader::prepare_bed_2bit_packed_owned;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

const KING_DEFAULT_KINSHIP_THRESHOLD: f64 = 0.05_f64;
const KING_DEFAULT_SAMPLE_BLOCK_SIZE: usize = 256;
const KING_DEFAULT_WORD_CHUNK_WORDS: usize = 64;
const KING_DEFAULT_MAX_EXACT_PAIRS: u64 = 8_000_000_000;

#[derive(Clone, Debug)]
pub struct KingBitplanes {
    hom_ref: Vec<u64>,
    het: Vec<u64>,
    hom_alt: Vec<u64>,
    n_samples: usize,
    n_sites: usize,
    words_per_sample: usize,
}

impl KingBitplanes {
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[inline]
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    #[inline]
    pub fn words_per_sample(&self) -> usize {
        self.words_per_sample
    }

    #[inline]
    fn sample_word_offset(&self, sample_idx: usize) -> usize {
        sample_idx * self.words_per_sample
    }

    #[inline]
    pub fn storage_bytes(&self) -> usize {
        (self.hom_ref.len() + self.het.len() + self.hom_alt.len()) * std::mem::size_of::<u64>()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KingPairStats {
    pub shared_nonmissing: usize,
    pub ibs0: usize,
    pub ibs1: usize,
    pub ibs2: usize,
    pub het_i_obs: usize,
    pub het_j_obs: usize,
    pub both_het: usize,
    pub kinship: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KingRelatedPairRow {
    pub sample_i: u32,
    pub sample_j: u32,
    pub ibs0: u32,
    pub kinship: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KingRelatedGraph {
    pub neighbors: Vec<Vec<u32>>,
    pub degrees: Vec<i32>,
    pub edge_count: usize,
    pub n_sites: usize,
    pub kinship_threshold: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KingPruneResult {
    pub neighbors: Vec<Vec<u32>>,
    pub degrees: Vec<i32>,
    pub edge_count: usize,
    pub n_sites: usize,
    pub kinship_threshold: f64,
    pub kept: Vec<u32>,
    pub removed: Vec<u32>,
}

#[inline]
fn validate_king_inputs(packed_flat: &[u8], n_samples: usize) -> Result<(usize, usize), String> {
    if n_samples == 0 {
        return Err("KING requires n_samples > 0".to_string());
    }
    if n_samples > (u32::MAX as usize) {
        return Err(format!(
            "KING adjacency uses u32 sample ids; got n_samples={n_samples} > {}",
            u32::MAX
        ));
    }
    let bytes_per_snp = n_samples.div_ceil(4);
    if bytes_per_snp == 0 {
        return Err("invalid KING bytes_per_snp=0".to_string());
    }
    if packed_flat.is_empty() {
        return Err("KING requires non-empty packed genotype data".to_string());
    }
    if packed_flat.len() % bytes_per_snp != 0 {
        return Err(format!(
            "KING packed payload length mismatch: packed_bytes={} not divisible by bytes_per_snp={bytes_per_snp}",
            packed_flat.len()
        ));
    }
    let n_sites = packed_flat.len() / bytes_per_snp;
    if n_sites == 0 {
        return Err("KING requires at least one variant site".to_string());
    }
    Ok((bytes_per_snp, n_sites))
}

#[inline]
fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

#[inline]
fn parse_env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

#[inline]
fn parse_env_bool_default(name: &str, default: bool) -> bool {
    std::env::var(name).map_or(default, |v| {
        let raw = v.trim().to_ascii_lowercase();
        if matches!(raw.as_str(), "1" | "true" | "yes" | "y" | "on") {
            true
        } else if matches!(raw.as_str(), "0" | "false" | "no" | "n" | "off") {
            false
        } else {
            default
        }
    })
}

#[inline]
fn king_build_pool(threads: usize) -> Result<Option<rayon::ThreadPool>, String> {
    if threads <= 1 {
        return Ok(None);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map(Some)
        .map_err(|e| format!("rayon pool: {e}"))
}

#[inline]
fn king_sample_block_size(n_samples: usize) -> usize {
    parse_env_usize(
        "JANUSX_KING_SAMPLE_BLOCK_SIZE",
        KING_DEFAULT_SAMPLE_BLOCK_SIZE,
    )
    .min(n_samples.max(1))
    .max(1)
}

#[inline]
fn king_word_chunk_words(words_per_sample: usize) -> usize {
    parse_env_usize(
        "JANUSX_KING_WORD_CHUNK_WORDS",
        KING_DEFAULT_WORD_CHUNK_WORDS,
    )
    .min(words_per_sample.max(1))
    .max(1)
}

#[inline]
fn king_exact_max_pairs() -> u64 {
    parse_env_u64("JANUSX_KING_MAX_EXACT_PAIRS", KING_DEFAULT_MAX_EXACT_PAIRS)
}

#[inline]
fn king_early_stop_enabled() -> bool {
    parse_env_bool_default("JANUSX_KING_EARLY_STOP", true)
}

#[inline]
fn king_enforce_exact_budget(n_samples: usize) -> Result<(), String> {
    let max_pairs = king_exact_max_pairs();
    if max_pairs == 0 {
        return Ok(());
    }
    let exact_pairs =
        (n_samples as u128).saturating_mul(n_samples.saturating_sub(1) as u128) / 2u128;
    if exact_pairs > (max_pairs as u128) {
        return Err(format!(
            "KING exact all-pairs budget exceeded: pairs={} > limit={max_pairs}. Set JANUSX_KING_MAX_EXACT_PAIRS=0 to force, or raise the limit explicitly.",
            exact_pairs
        ));
    }
    Ok(())
}

#[inline]
fn king_stats_from_counts(counts: KingBitCounts) -> KingPairStats {
    let shared_nonmissing = counts.shared_nonmissing as usize;
    let ibs0 = counts.ibs0 as usize;
    let both_het = counts.both_het as usize;
    let same_hom = counts.same_hom as usize;
    let ibs2 = both_het + same_hom;
    let het_i_obs = counts.het_i_obs as usize;
    let het_j_obs = counts.het_j_obs as usize;
    let ibs1 = shared_nonmissing.saturating_sub(ibs0 + ibs2);
    let denom = het_i_obs + het_j_obs;
    let kinship = if denom > 0 {
        (both_het as f64 - 2.0_f64 * ibs0 as f64) / (denom as f64)
    } else {
        f64::NAN
    };
    KingPairStats {
        shared_nonmissing,
        ibs0,
        ibs1,
        ibs2,
        het_i_obs,
        het_j_obs,
        both_het,
        kinship,
    }
}

#[inline]
fn king_kinship_upper_bound(counts: KingBitCounts, remaining_sites: usize) -> f64 {
    let a = counts.both_het as f64 - 2.0_f64 * counts.ibs0 as f64;
    let b = (counts.het_i_obs + counts.het_j_obs) as f64;
    let k = remaining_sites as f64;
    let future_all_both_het = if remaining_sites > 0 {
        (a + k) / (b + 2.0_f64 * k)
    } else {
        f64::NEG_INFINITY
    };
    if b > 0.0_f64 {
        (a / b).max(future_all_both_het)
    } else if remaining_sites > 0 {
        future_all_both_het
    } else {
        f64::NAN
    }
}

pub fn king_bitplanes_from_packed(
    packed_flat: &[u8],
    n_samples: usize,
) -> Result<KingBitplanes, String> {
    let (bytes_per_snp, n_sites) = validate_king_inputs(packed_flat, n_samples)?;
    let words_per_sample = n_sites.div_ceil(64);
    let total_words = n_samples * words_per_sample;
    let code4_lut = &packed_byte_lut().code4;

    let mut hom_ref = vec![0u64; total_words];
    let mut het = vec![0u64; total_words];
    let mut hom_alt = vec![0u64; total_words];

    let full_bytes = n_samples / 4;
    let rem = n_samples % 4;

    for site_idx in 0..n_sites {
        let word_idx = site_idx / 64;
        let bit = 1u64 << (site_idx & 63);
        let row_off = site_idx * bytes_per_snp;
        let row = &packed_flat[row_off..row_off + bytes_per_snp];

        let mut dst = word_idx;
        for &byte in row.iter().take(full_bytes) {
            let codes = &code4_lut[byte as usize];
            match codes[0] {
                0b00 => hom_ref[dst] |= bit,
                0b10 => het[dst] |= bit,
                0b11 => hom_alt[dst] |= bit,
                _ => {}
            }
            dst += words_per_sample;
            match codes[1] {
                0b00 => hom_ref[dst] |= bit,
                0b10 => het[dst] |= bit,
                0b11 => hom_alt[dst] |= bit,
                _ => {}
            }
            dst += words_per_sample;
            match codes[2] {
                0b00 => hom_ref[dst] |= bit,
                0b10 => het[dst] |= bit,
                0b11 => hom_alt[dst] |= bit,
                _ => {}
            }
            dst += words_per_sample;
            match codes[3] {
                0b00 => hom_ref[dst] |= bit,
                0b10 => het[dst] |= bit,
                0b11 => hom_alt[dst] |= bit,
                _ => {}
            }
            dst += words_per_sample;
        }
        if rem > 0 {
            let codes = &code4_lut[row[full_bytes] as usize];
            for &code in codes.iter().take(rem) {
                match code {
                    0b00 => hom_ref[dst] |= bit,
                    0b10 => het[dst] |= bit,
                    0b11 => hom_alt[dst] |= bit,
                    _ => {}
                }
                dst += words_per_sample;
            }
        }
    }

    Ok(KingBitplanes {
        hom_ref,
        het,
        hom_alt,
        n_samples,
        n_sites,
        words_per_sample,
    })
}

fn king_pair_stats_unchecked(
    bitplanes: &KingBitplanes,
    sample_i: usize,
    sample_j: usize,
) -> KingPairStats {
    let off_i = bitplanes.sample_word_offset(sample_i);
    let off_j = bitplanes.sample_word_offset(sample_j);
    let counts = king_pair_counts(
        &bitplanes.hom_ref[off_i..off_i + bitplanes.words_per_sample],
        &bitplanes.het[off_i..off_i + bitplanes.words_per_sample],
        &bitplanes.hom_alt[off_i..off_i + bitplanes.words_per_sample],
        &bitplanes.hom_ref[off_j..off_j + bitplanes.words_per_sample],
        &bitplanes.het[off_j..off_j + bitplanes.words_per_sample],
        &bitplanes.hom_alt[off_j..off_j + bitplanes.words_per_sample],
    );
    king_stats_from_counts(counts)
}

fn king_pair_stats_meets_threshold_unchecked(
    bitplanes: &KingBitplanes,
    sample_i: usize,
    sample_j: usize,
    kinship_threshold: f64,
    word_chunk_words: usize,
    enable_early_stop: bool,
) -> Option<KingPairStats> {
    let off_i = bitplanes.sample_word_offset(sample_i);
    let off_j = bitplanes.sample_word_offset(sample_j);
    let zi = &bitplanes.hom_ref[off_i..off_i + bitplanes.words_per_sample];
    let hi = &bitplanes.het[off_i..off_i + bitplanes.words_per_sample];
    let ai = &bitplanes.hom_alt[off_i..off_i + bitplanes.words_per_sample];
    let zj = &bitplanes.hom_ref[off_j..off_j + bitplanes.words_per_sample];
    let hj = &bitplanes.het[off_j..off_j + bitplanes.words_per_sample];
    let aj = &bitplanes.hom_alt[off_j..off_j + bitplanes.words_per_sample];

    let mut counts = KingBitCounts::default();
    let chunk_words = word_chunk_words
        .min(bitplanes.words_per_sample.max(1))
        .max(1);
    let early_stop = enable_early_stop && kinship_threshold.is_finite();
    for word_start in (0..bitplanes.words_per_sample).step_by(chunk_words) {
        let word_end = (word_start + chunk_words).min(bitplanes.words_per_sample);
        counts.add_assign(king_pair_counts(
            &zi[word_start..word_end],
            &hi[word_start..word_end],
            &ai[word_start..word_end],
            &zj[word_start..word_end],
            &hj[word_start..word_end],
            &aj[word_start..word_end],
        ));
        if early_stop {
            let processed_sites = (word_end * 64).min(bitplanes.n_sites);
            let remaining_sites = bitplanes.n_sites.saturating_sub(processed_sites);
            let upper_bound = king_kinship_upper_bound(counts, remaining_sites);
            if upper_bound.is_finite() && upper_bound < kinship_threshold {
                return None;
            }
        }
    }

    let stats = king_stats_from_counts(counts);
    if stats.kinship.is_finite() && stats.kinship >= kinship_threshold {
        Some(stats)
    } else {
        None
    }
}

pub fn king_pair_stats(
    bitplanes: &KingBitplanes,
    sample_i: usize,
    sample_j: usize,
) -> Result<KingPairStats, String> {
    if sample_i >= bitplanes.n_samples {
        return Err(format!(
            "KING sample_i out of range: {sample_i} >= {}",
            bitplanes.n_samples
        ));
    }
    if sample_j >= bitplanes.n_samples {
        return Err(format!(
            "KING sample_j out of range: {sample_j} >= {}",
            bitplanes.n_samples
        ));
    }
    Ok(king_pair_stats_unchecked(bitplanes, sample_i, sample_j))
}

pub fn king_related_pairs_from_bitplanes(
    bitplanes: &KingBitplanes,
    kinship_threshold: f64,
    threads: usize,
) -> Result<Vec<KingRelatedPairRow>, String> {
    if !kinship_threshold.is_finite() {
        return Err("KING kinship_threshold must be finite".to_string());
    }
    if bitplanes.n_samples > (u32::MAX as usize) {
        return Err(format!(
            "KING adjacency uses u32 sample ids; got n_samples={} > {}",
            bitplanes.n_samples,
            u32::MAX
        ));
    }

    king_enforce_exact_budget(bitplanes.n_samples)?;
    let n = bitplanes.n_samples;
    let sample_block = king_sample_block_size(n);
    let word_chunk = king_word_chunk_words(bitplanes.words_per_sample);
    let enable_early_stop = king_early_stop_enabled();
    let use_parallel = n >= sample_block && threads != 1;
    let pool = king_build_pool(threads)?;
    let block_starts: Vec<usize> = (0..n).step_by(sample_block).collect();

    let build_serial = || -> Vec<KingRelatedPairRow> {
        let mut rows = Vec::<KingRelatedPairRow>::new();
        for &block_start in block_starts.iter() {
            let block_end = (block_start + sample_block).min(n);
            for i in block_start..block_end {
                for j in (i + 1)..n {
                    if let Some(stats) = king_pair_stats_meets_threshold_unchecked(
                        bitplanes,
                        i,
                        j,
                        kinship_threshold,
                        word_chunk,
                        enable_early_stop,
                    ) {
                        rows.push(KingRelatedPairRow {
                            sample_i: i as u32,
                            sample_j: j as u32,
                            ibs0: stats.ibs0 as u32,
                            kinship: stats.kinship,
                        });
                    }
                }
            }
        }
        rows
    };

    let build_parallel = || -> Vec<KingRelatedPairRow> {
        let chunks: Vec<Vec<KingRelatedPairRow>> = block_starts
            .par_iter()
            .copied()
            .map(|block_start| {
                let block_end = (block_start + sample_block).min(n);
                let mut rows = Vec::<KingRelatedPairRow>::new();
                for i in block_start..block_end {
                    for j in (i + 1)..n {
                        if let Some(stats) = king_pair_stats_meets_threshold_unchecked(
                            bitplanes,
                            i,
                            j,
                            kinship_threshold,
                            word_chunk,
                            enable_early_stop,
                        ) {
                            rows.push(KingRelatedPairRow {
                                sample_i: i as u32,
                                sample_j: j as u32,
                                ibs0: stats.ibs0 as u32,
                                kinship: stats.kinship,
                            });
                        }
                    }
                }
                rows
            })
            .collect();
        let mut rows = Vec::<KingRelatedPairRow>::new();
        for chunk in chunks {
            rows.extend(chunk);
        }
        rows
    };

    let rows = if use_parallel {
        if let Some(tp) = pool.as_ref() {
            tp.install(build_parallel)
        } else {
            build_parallel()
        }
    } else {
        build_serial()
    };
    Ok(rows)
}

pub fn king_related_pairs_from_packed(
    packed_flat: &[u8],
    n_samples: usize,
    kinship_threshold: f64,
    threads: usize,
) -> Result<Vec<KingRelatedPairRow>, String> {
    let bitplanes = king_bitplanes_from_packed(packed_flat, n_samples)?;
    king_related_pairs_from_bitplanes(&bitplanes, kinship_threshold, threads)
}

pub fn king_related_graph_from_bitplanes(
    bitplanes: &KingBitplanes,
    kinship_threshold: f64,
    threads: usize,
) -> Result<KingRelatedGraph, String> {
    if !kinship_threshold.is_finite() {
        return Err("KING kinship_threshold must be finite".to_string());
    }
    if bitplanes.n_samples > (u32::MAX as usize) {
        return Err(format!(
            "KING adjacency uses u32 sample ids; got n_samples={} > {}",
            bitplanes.n_samples,
            u32::MAX
        ));
    }

    king_enforce_exact_budget(bitplanes.n_samples)?;
    let n = bitplanes.n_samples;
    let sample_block = king_sample_block_size(n);
    let word_chunk = king_word_chunk_words(bitplanes.words_per_sample);
    let enable_early_stop = king_early_stop_enabled();
    let use_parallel = n >= sample_block && threads != 1;
    let pool = king_build_pool(threads)?;
    let block_starts: Vec<usize> = (0..n).step_by(sample_block).collect();

    let build_serial = || -> Vec<(usize, Vec<Vec<u32>>)> {
        let mut blocks = Vec::<(usize, Vec<Vec<u32>>)>::new();
        for &block_start in block_starts.iter() {
            let block_end = (block_start + sample_block).min(n);
            let mut upper = vec![Vec::<u32>::new(); block_end - block_start];
            for (offset, rel) in upper.iter_mut().enumerate() {
                let i = block_start + offset;
                for j in (i + 1)..n {
                    if king_pair_stats_meets_threshold_unchecked(
                        bitplanes,
                        i,
                        j,
                        kinship_threshold,
                        word_chunk,
                        enable_early_stop,
                    )
                    .is_some()
                    {
                        rel.push(j as u32);
                    }
                }
            }
            blocks.push((block_start, upper));
        }
        blocks
    };

    let build_parallel = || -> Vec<(usize, Vec<Vec<u32>>)> {
        block_starts
            .par_iter()
            .copied()
            .map(|block_start| {
                let block_end = (block_start + sample_block).min(n);
                let mut upper = vec![Vec::<u32>::new(); block_end - block_start];
                for (offset, rel) in upper.iter_mut().enumerate() {
                    let i = block_start + offset;
                    for j in (i + 1)..n {
                        if king_pair_stats_meets_threshold_unchecked(
                            bitplanes,
                            i,
                            j,
                            kinship_threshold,
                            word_chunk,
                            enable_early_stop,
                        )
                        .is_some()
                        {
                            rel.push(j as u32);
                        }
                    }
                }
                (block_start, upper)
            })
            .collect()
    };

    let upper_blocks = if use_parallel {
        if let Some(tp) = pool.as_ref() {
            tp.install(build_parallel)
        } else {
            build_parallel()
        }
    } else {
        build_serial()
    };

    let mut neighbors = vec![Vec::<u32>::new(); n];
    for (block_start, upper) in upper_blocks.iter() {
        for (offset, rel) in upper.iter().enumerate() {
            neighbors[block_start + offset].reserve(rel.len());
        }
    }
    for (block_start, upper) in upper_blocks.into_iter() {
        for (offset, rel) in upper.into_iter().enumerate() {
            let i = block_start + offset;
            neighbors[i].extend(rel.iter().copied());
            for j in rel {
                neighbors[j as usize].push(i as u32);
            }
        }
    }

    let degrees: Vec<i32> = neighbors.iter().map(|rel| rel.len() as i32).collect();
    let edge_count = degrees.iter().map(|&deg| deg as usize).sum::<usize>() / 2;

    Ok(KingRelatedGraph {
        neighbors,
        degrees,
        edge_count,
        n_sites: bitplanes.n_sites,
        kinship_threshold,
    })
}

pub fn king_related_graph_from_packed(
    packed_flat: &[u8],
    n_samples: usize,
    kinship_threshold: f64,
    threads: usize,
) -> Result<KingRelatedGraph, String> {
    let bitplanes = king_bitplanes_from_packed(packed_flat, n_samples)?;
    king_related_graph_from_bitplanes(&bitplanes, kinship_threshold, threads)
}

pub fn king_prune_related_graph(graph: KingRelatedGraph) -> Result<KingPruneResult, String> {
    let KingRelatedGraph {
        neighbors,
        degrees,
        edge_count,
        n_sites,
        kinship_threshold,
    } = graph;
    let n = neighbors.len();
    if degrees.len() != n {
        return Err(format!(
            "KING degree vector length mismatch: degrees={} neighbors={n}",
            degrees.len()
        ));
    }
    for (i, rel) in neighbors.iter().enumerate() {
        for &nbr in rel.iter() {
            if (nbr as usize) >= n {
                return Err(format!(
                    "KING neighbor index out of range: neighbors[{i}] contains {nbr} >= {n}"
                ));
            }
        }
    }

    let mut live_degrees = degrees.clone();
    let mut is_active = vec![true; n];
    let mut heap = BinaryHeap::<(i32, u32)>::with_capacity(n);
    for (i, &deg) in live_degrees.iter().enumerate() {
        heap.push((deg, i as u32));
    }

    let mut removed = Vec::<u32>::new();
    while let Some((heap_deg, sample_id)) = heap.pop() {
        let idx = sample_id as usize;
        if !is_active[idx] {
            continue;
        }
        let real_deg = live_degrees[idx];
        if heap_deg != real_deg {
            continue;
        }
        if real_deg <= 0 {
            break;
        }

        is_active[idx] = false;
        live_degrees[idx] = 0;
        removed.push(sample_id);

        for &nbr in neighbors[idx].iter() {
            let nbr_idx = nbr as usize;
            if is_active[nbr_idx] {
                live_degrees[nbr_idx] -= 1;
                heap.push((live_degrees[nbr_idx], nbr));
            }
        }
    }

    let kept = is_active
        .iter()
        .enumerate()
        .filter_map(|(idx, &active)| active.then_some(idx as u32))
        .collect();

    Ok(KingPruneResult {
        neighbors,
        degrees,
        edge_count,
        n_sites,
        kinship_threshold,
        kept,
        removed,
    })
}

pub fn king_unrelated_set_from_packed(
    packed_flat: &[u8],
    n_samples: usize,
    kinship_threshold: f64,
    threads: usize,
) -> Result<KingPruneResult, String> {
    let graph = king_related_graph_from_packed(packed_flat, n_samples, kinship_threshold, threads)?;
    king_prune_related_graph(graph)
}

pub fn king_unrelated_set_from_bed(
    prefix: &str,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    kinship_threshold: f64,
    threads: usize,
) -> Result<KingPruneResult, String> {
    let prepared = prepare_bed_2bit_packed_owned(
        prefix,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
    )?;
    king_unrelated_set_from_packed(
        prepared.packed.as_slice(),
        prepared.n_samples,
        kinship_threshold,
        threads,
    )
}

pub fn king_unrelated_set_from_bed_default(
    prefix: &str,
    threads: usize,
) -> Result<KingPruneResult, String> {
    king_unrelated_set_from_bed(
        prefix,
        0.01_f32,
        0.1_f32,
        0.0_f32,
        false,
        KING_DEFAULT_KINSHIP_THRESHOLD,
        threads,
    )
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    maf_threshold=0.01_f32,
    max_missing_rate=0.1_f32,
    het_threshold=0.0_f32,
    snps_only=false,
    kinship_threshold=KING_DEFAULT_KINSHIP_THRESHOLD,
    threads=0
))]
pub fn king_unrelated_set_from_bed_py(
    prefix: String,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    kinship_threshold: f64,
    threads: usize,
) -> PyResult<(Vec<u32>, Vec<u32>, usize, usize)> {
    let out = king_unrelated_set_from_bed(
        &prefix,
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
        kinship_threshold,
        threads,
    )
    .map_err(PyRuntimeError::new_err)?;
    Ok((out.kept, out.removed, out.edge_count, out.n_sites))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_site_major_dosages(sample_major: &[Vec<Option<u8>>]) -> Vec<u8> {
        let n_samples = sample_major.len();
        let n_sites = sample_major.first().map(|v| v.len()).unwrap_or(0);
        let bytes_per_snp = n_samples.div_ceil(4);
        let mut packed = vec![0u8; n_sites * bytes_per_snp];
        for site_idx in 0..n_sites {
            for (sample_idx, sample) in sample_major.iter().enumerate() {
                let code = match sample[site_idx] {
                    Some(0) => 0b00u8,
                    Some(1) => 0b10u8,
                    Some(2) => 0b11u8,
                    None => 0b01u8,
                    Some(v) => panic!("invalid dosage {v}"),
                };
                let byte_off = site_idx * bytes_per_snp + (sample_idx >> 2);
                let lane_shift = (sample_idx & 3) * 2;
                packed[byte_off] |= code << lane_shift;
            }
        }
        packed
    }

    #[test]
    fn king_pair_stats_match_expected_counts() {
        let geno = vec![
            vec![Some(0), Some(1), Some(2), Some(0)],
            vec![Some(2), Some(1), Some(0), Some(1)],
        ];
        let packed = pack_site_major_dosages(&geno);
        let bitplanes = king_bitplanes_from_packed(&packed, geno.len()).unwrap();
        let stats = king_pair_stats(&bitplanes, 0, 1).unwrap();
        assert_eq!(stats.shared_nonmissing, 4);
        assert_eq!(stats.ibs0, 2);
        assert_eq!(stats.ibs1, 1);
        assert_eq!(stats.ibs2, 1);
        assert_eq!(stats.het_i_obs, 1);
        assert_eq!(stats.het_j_obs, 2);
        assert_eq!(stats.both_het, 1);
        assert!((stats.kinship + 1.0_f64).abs() < 1e-12_f64);
    }

    #[test]
    fn king_prune_graph_lazy_deletion_keeps_independent_set() {
        let graph = KingRelatedGraph {
            neighbors: vec![vec![1, 2], vec![0], vec![0, 3], vec![2]],
            degrees: vec![2, 1, 2, 1],
            edge_count: 3,
            n_sites: 10,
            kinship_threshold: 0.05_f64,
        };
        let pruned = king_prune_related_graph(graph).unwrap();
        assert_eq!(pruned.kept, vec![0, 3]);
        assert_eq!(pruned.removed, vec![2, 1]);
        assert_eq!(pruned.degrees, vec![2, 1, 2, 1]);
    }

    #[test]
    fn king_related_graph_finds_related_pair() {
        let geno = vec![
            vec![Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)],
            vec![Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)],
            vec![Some(0), Some(2), Some(0), Some(2), Some(0), Some(2)],
        ];
        let packed = pack_site_major_dosages(&geno);
        let graph = king_related_graph_from_packed(&packed, geno.len(), 0.05_f64, 1).unwrap();
        assert_eq!(graph.neighbors[0], vec![1]);
        assert_eq!(graph.neighbors[1], vec![0]);
        assert!(graph.neighbors[2].is_empty());
        assert_eq!(graph.edge_count, 1);
    }

    #[test]
    fn king_related_pairs_emit_sparse_four_column_rows() {
        let geno = vec![
            vec![Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)],
            vec![Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)],
            vec![Some(0), Some(2), Some(0), Some(2), Some(0), Some(2)],
        ];
        let packed = pack_site_major_dosages(&geno);
        let rows = king_related_pairs_from_packed(&packed, geno.len(), 0.05_f64, 1).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].sample_i, 0);
        assert_eq!(rows[0].sample_j, 1);
        assert_eq!(rows[0].ibs0, 0);
        assert!(rows[0].kinship >= 0.05_f64);
    }
}
