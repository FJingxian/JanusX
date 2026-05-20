mod bs;
mod residual;
mod sampling;
mod score;

use crate::beam::{beam_search_and_binary_mcc, beam_search_and_continuous_abs_corr, BeamAndResult};
use crate::bitwise::bitand_assign;
use crate::gfcore::{process_snp_row, BedSnpIter, HmpSnpIter, SiteInfo, TxtSnpIter, VcfSnpIter};
use crate::gfreader::{
    build_sample_selection, prepare_bed_2bit_packed_owned, prepare_bed_logic_meta_owned,
};
use crate::grm::grm_packed_f64_from_stats_rust;
use crate::linalg::{format_chisq_value, student_t_p_two_sided};
use crate::ml::common::{
    parse_importance, parse_permutation_scoring, topk_indices, ImportanceKind, PermutationConfig,
    ResponseKind,
};
use crate::ml::engine::{compute_feature_scores, parse_ml_engine, MlEngine};
use crate::ml::extra_trees::ExtraTreesConfig;
use crate::score::{score_binary_mcc_packed, score_cont_corr_packed};
use memmap2::Mmap;
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::BoundObject;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

#[allow(unused_imports)]
pub use bs::{
    beam_search_train_test_continuous, evaluate_rule_continuous, materialize_rule_bits,
    rank_rule_score_components, BeamBinaryOp, BeamLiteral, BeamLogicGateMode, BeamRankMode,
    BeamRule, BeamRuleCandidate, BeamSearchParams,
};
pub use residual::{garfield_residualize_bed_py, garfield_residualize_grm_py};
use residual::{garfield_residualize_exact_from_grm_rust, GarfieldResidualResult};
#[allow(unused_imports)]
pub use sampling::stratified_test_mask;
#[allow(unused_imports)]
pub use score::{score_cont_weighted_mean_diff_packed, ContinuousRuleScore};

const BIN01_MAGIC: &[u8; 8] = b"JXBIN001";
const BIN01_HEADER_LEN: usize = 32;
const GARFIELD_CONSTRAINED_BEAM_PAR_MIN_TOTAL_CANDS: usize = 1_024;
const GARFIELD_CONSTRAINED_BEAM_PAR_CHUNK_CANDS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldInputKind {
    Auto,
    Bfile,
    Vcf,
    Hmp,
    File,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldBinMode {
    Bin,
    Mbin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarfieldResponse {
    Binary,
    Continuous,
}

#[derive(Clone)]
struct EncodedRow {
    site: SiteInfo,
    bits: Vec<u8>,
}

enum GarfieldInputReader {
    Bed(BedSnpIter),
    Vcf(VcfSnpIter),
    Hmp(HmpSnpIter),
    Txt(TxtSnpIter),
}

impl GarfieldInputReader {
    fn sample_ids(&self) -> &[String] {
        match self {
            GarfieldInputReader::Bed(it) => &it.samples,
            GarfieldInputReader::Vcf(it) => &it.samples,
            GarfieldInputReader::Hmp(it) => &it.samples,
            GarfieldInputReader::Txt(it) => &it.samples,
        }
    }
}

#[derive(Clone, Debug)]
struct GarfieldWindow {
    chrom: String,
    bp_start: i32,
    bp_end: i32,
    indices: Vec<usize>,
}

#[derive(Clone, Debug)]
struct ConstrainedBeamNode {
    selected: Vec<usize>,
    combined: Vec<u64>,
    score: f64,
    last_index: usize,
}

#[inline]
fn parse_response(response: &str) -> Result<GarfieldResponse, String> {
    let t = response.trim().to_ascii_lowercase();
    match t.as_str() {
        "binary" | "bin" | "b" => Ok(GarfieldResponse::Binary),
        "continuous" | "cont" | "c" => Ok(GarfieldResponse::Continuous),
        _ => Err("response must be 'binary' or 'continuous'".to_string()),
    }
}

#[inline]
fn parse_input_kind(input_kind: &str) -> Result<GarfieldInputKind, String> {
    let t = input_kind.trim().to_ascii_lowercase();
    match t.as_str() {
        "" | "auto" => Ok(GarfieldInputKind::Auto),
        "bfile" | "plink" | "bed" => Ok(GarfieldInputKind::Bfile),
        "vcf" => Ok(GarfieldInputKind::Vcf),
        "hmp" | "hapmap" => Ok(GarfieldInputKind::Hmp),
        "file" | "txt" | "npy" | "bin" => Ok(GarfieldInputKind::File),
        _ => Err("input_kind must be one of: auto, bfile, vcf, hmp, file".to_string()),
    }
}

#[inline]
fn parse_bin_mode(mode: &str) -> Result<GarfieldBinMode, String> {
    let t = mode.trim().to_ascii_lowercase();
    match t.as_str() {
        "bin" | "bin02" => Ok(GarfieldBinMode::Bin),
        "mbin" => Ok(GarfieldBinMode::Mbin),
        _ => Err("mode must be one of: bin, mbin".to_string()),
    }
}

#[inline]
fn parse_beam_logic_gate_mode(mode: &str) -> Result<BeamLogicGateMode, String> {
    let t = mode.trim().to_ascii_lowercase();
    match t.as_str() {
        "" | "ao" | "and_or" | "andor" | "and-or" => Ok(BeamLogicGateMode::AndOr),
        "a" | "and" => Ok(BeamLogicGateMode::AndOnly),
        "o" | "or" => Ok(BeamLogicGateMode::OrOnly),
        _ => Err("logic_gate must be one of: A, O, AO".to_string()),
    }
}

#[inline]
fn parse_beam_rank_mode(mode: &str) -> Result<BeamRankMode, String> {
    let t = mode.trim().to_ascii_lowercase();
    if let Some(rest) = t.strip_prefix("gain_from_layer:") {
        let start = rest
            .parse::<usize>()
            .map_err(|_| "rank_score gain_from_layer:<N> requires integer N >= 1".to_string())?;
        if start < 1 {
            return Err("rank_score gain_from_layer:<N> requires integer N >= 1".to_string());
        }
        return Ok(BeamRankMode::GainFromLayer(start));
    }
    match t.as_str() {
        "" | "interaction_gain" | "gain" | "interaction-gain" | "interactiongain" => {
            Ok(BeamRankMode::InteractionGain)
        }
        "exhaustive_then_gain"
        | "exhaustive-then-gain"
        | "exhaustivethengain"
        | "staged_gain"
        | "staged-gain"
        | "stagedgain"
        | "beam_gain"
        | "beam-gain"
        | "beamgain" => Ok(BeamRankMode::ExhaustiveThenGain),
        "raw" | "score" => Ok(BeamRankMode::Raw),
        _ => Err(
            "rank_score must be one of: raw, interaction_gain, exhaustive_then_gain, gain_from_layer:<N>"
                .to_string(),
        ),
    }
}

#[inline]
fn garfield_progress_notify(
    progress_callback: Option<&Py<PyAny>>,
    done: usize,
    total: usize,
) -> PyResult<()> {
    if let Some(cb) = progress_callback {
        Python::attach(|py2| -> PyResult<()> {
            py2.check_signals()?;
            cb.call1(py2, (done, total))?;
            Ok(())
        })?;
    }
    Ok(())
}

#[inline]
fn words_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(64).max(1)
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
    let st = row_idx * row_words;
    &bits_flat[st..st + needed_words]
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
fn cmp_constrained_nodes(a: &ConstrainedBeamNode, b: &ConstrainedBeamNode) -> std::cmp::Ordering {
    let sa = score_key(a.score);
    let sb = score_key(b.score);
    match sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal) {
        std::cmp::Ordering::Equal => match a.selected.len().cmp(&b.selected.len()) {
            std::cmp::Ordering::Equal => a.selected.cmp(&b.selected),
            other => other,
        },
        other => other,
    }
}

#[inline]
fn constrained_node_better(a: &ConstrainedBeamNode, b: &ConstrainedBeamNode) -> bool {
    cmp_constrained_nodes(a, b) == std::cmp::Ordering::Less
}

#[inline]
fn push_top_k_constrained_streaming(
    nodes: &mut Vec<ConstrainedBeamNode>,
    cand: ConstrainedBeamNode,
    k: usize,
) {
    if k == 0 {
        return;
    }
    if nodes.len() < k {
        nodes.push(cand);
        return;
    }

    let mut worst_idx = 0usize;
    for i in 1..nodes.len() {
        if cmp_constrained_nodes(&nodes[i], &nodes[worst_idx]) == std::cmp::Ordering::Greater {
            worst_idx = i;
        }
    }

    if cmp_constrained_nodes(&cand, &nodes[worst_idx]) == std::cmp::Ordering::Less {
        nodes[worst_idx] = cand;
    }
}

#[inline]
fn garfield_should_parallel_constrained(total_cands: usize) -> bool {
    rayon::current_num_threads() > 1 && total_cands >= GARFIELD_CONSTRAINED_BEAM_PAR_MIN_TOTAL_CANDS
}

#[inline]
fn sort_truncate_nodes(mut nodes: Vec<ConstrainedBeamNode>, k: usize) -> Vec<ConstrainedBeamNode> {
    nodes.sort_by(cmp_constrained_nodes);
    if nodes.len() > k {
        nodes.truncate(k);
    }
    nodes
}

#[inline]
fn validate_constrained_inputs(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
    group_ids: &[usize],
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
    if max_pick == 0 {
        return Err(format!("{ctx}: max_pick must be > 0"));
    }
    if beam_width == 0 {
        return Err(format!("{ctx}: beam_width must be > 0"));
    }
    if group_ids.len() != n_rows {
        return Err(format!(
            "{ctx}: group_ids length mismatch: {} vs n_rows={}",
            group_ids.len(),
            n_rows
        ));
    }
    let needed_words = words_for_samples(n_samples);
    if row_words < needed_words {
        return Err(format!(
            "{ctx}: row_words={} smaller than required {}",
            row_words, needed_words
        ));
    }
    let total_words = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| format!("{ctx}: n_rows*row_words overflow"))?;
    if bits_flat.len() < total_words {
        return Err(format!(
            "{ctx}: bits length={} smaller than required {}",
            bits_flat.len(),
            total_words
        ));
    }
    Ok(needed_words)
}

fn beam_search_and_with_group_exclusion<F>(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    max_pick: usize,
    beam_width: usize,
    group_ids: &[usize],
    score_fn: F,
) -> Result<BeamAndResult, String>
where
    F: Fn(&[u64]) -> f64 + Sync,
{
    let ctx = "garfield::beam_search_and_with_group_exclusion";
    let needed_words = validate_constrained_inputs(
        bits_flat, row_words, n_rows, n_samples, max_pick, beam_width, group_ids, ctx,
    )?;

    let mask = tail_mask(n_samples);
    let max_depth = max_pick.min(n_rows);
    let layer_cap = beam_width.min(n_rows);

    let mut beam: Vec<ConstrainedBeamNode> = if garfield_should_parallel_constrained(n_rows) {
        let mut work = Vec::<(usize, usize)>::new();
        let chunk = GARFIELD_CONSTRAINED_BEAM_PAR_CHUNK_CANDS.max(1);
        let mut start = 0usize;
        while start < n_rows {
            let end = (start + chunk).min(n_rows);
            work.push((start, end));
            start = end;
        }
        let local_tops: Vec<Vec<ConstrainedBeamNode>> = work
            .into_par_iter()
            .map(|(start, end)| {
                let mut local: Vec<ConstrainedBeamNode> = Vec::with_capacity(layer_cap);
                for i in start..end {
                    let mut combined = row_prefix(bits_flat, row_words, i, needed_words).to_vec();
                    apply_tail_mask(&mut combined, mask);
                    let score = score_fn(&combined);
                    push_top_k_constrained_streaming(
                        &mut local,
                        ConstrainedBeamNode {
                            selected: vec![i],
                            combined,
                            score,
                            last_index: i,
                        },
                        layer_cap,
                    );
                }
                local
            })
            .collect();
        let mut merged: Vec<ConstrainedBeamNode> = Vec::with_capacity(layer_cap);
        for local in local_tops {
            for cand in local {
                push_top_k_constrained_streaming(&mut merged, cand, layer_cap);
            }
        }
        merged
    } else {
        let mut seq: Vec<ConstrainedBeamNode> = Vec::with_capacity(layer_cap);
        for i in 0..n_rows {
            let mut combined = row_prefix(bits_flat, row_words, i, needed_words).to_vec();
            apply_tail_mask(&mut combined, mask);
            let score = score_fn(&combined);
            push_top_k_constrained_streaming(
                &mut seq,
                ConstrainedBeamNode {
                    selected: vec![i],
                    combined,
                    score,
                    last_index: i,
                },
                layer_cap,
            );
        }
        seq
    };
    beam = sort_truncate_nodes(beam, layer_cap);
    if beam.is_empty() {
        return Err(format!("{ctx}: no candidates"));
    }

    let mut best = beam[0].clone();
    for _depth in 2..=max_depth {
        let total_expand = beam
            .iter()
            .map(|node| n_rows.saturating_sub(node.last_index.saturating_add(1)))
            .sum::<usize>();
        let next: Vec<ConstrainedBeamNode> = if garfield_should_parallel_constrained(total_expand) {
            let mut work = Vec::<(usize, usize, usize)>::new();
            let chunk = GARFIELD_CONSTRAINED_BEAM_PAR_CHUNK_CANDS.max(1);
            for (bi, node) in beam.iter().enumerate() {
                let mut start = node.last_index + 1;
                while start < n_rows {
                    let end = (start + chunk).min(n_rows);
                    work.push((bi, start, end));
                    start = end;
                }
            }
            let local_tops: Vec<Vec<ConstrainedBeamNode>> = work
                .into_par_iter()
                .map(|(bi, start, end)| {
                    let node = &beam[bi];
                    let mut local: Vec<ConstrainedBeamNode> = Vec::with_capacity(layer_cap);
                    for cand in start..end {
                        let cg = group_ids[cand];
                        if node.selected.iter().any(|&s| group_ids[s] == cg) {
                            continue;
                        }
                        let row = row_prefix(bits_flat, row_words, cand, needed_words);
                        let mut combined = node.combined.clone();
                        bitand_assign(&mut combined, row);
                        apply_tail_mask(&mut combined, mask);
                        let score = score_fn(&combined);
                        let mut selected = node.selected.clone();
                        selected.push(cand);
                        push_top_k_constrained_streaming(
                            &mut local,
                            ConstrainedBeamNode {
                                selected,
                                combined,
                                score,
                                last_index: cand,
                            },
                            layer_cap,
                        );
                    }
                    local
                })
                .collect();
            let mut merged: Vec<ConstrainedBeamNode> = Vec::with_capacity(layer_cap);
            for local in local_tops {
                for cand in local {
                    push_top_k_constrained_streaming(&mut merged, cand, layer_cap);
                }
            }
            merged
        } else {
            let mut seq: Vec<ConstrainedBeamNode> = Vec::with_capacity(layer_cap);
            for node in beam.iter() {
                for cand in (node.last_index + 1)..n_rows {
                    let cg = group_ids[cand];
                    if node.selected.iter().any(|&s| group_ids[s] == cg) {
                        continue;
                    }
                    let row = row_prefix(bits_flat, row_words, cand, needed_words);
                    let mut combined = node.combined.clone();
                    bitand_assign(&mut combined, row);
                    apply_tail_mask(&mut combined, mask);
                    let score = score_fn(&combined);
                    let mut selected = node.selected.clone();
                    selected.push(cand);
                    push_top_k_constrained_streaming(
                        &mut seq,
                        ConstrainedBeamNode {
                            selected,
                            combined,
                            score,
                            last_index: cand,
                        },
                        layer_cap,
                    );
                }
            }
            seq
        };
        if next.is_empty() {
            break;
        }
        beam = sort_truncate_nodes(next, layer_cap);
        if constrained_node_better(&beam[0], &best) {
            best = beam[0].clone();
        }
    }

    Ok(BeamAndResult {
        selected_indices: best.selected,
        score: best.score,
        combined_bits: best.combined,
    })
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
    let ctx = "garfield::load_bin01_as_u64_words";
    let bytes = fs::read(path).map_err(|e| format!("{ctx}: failed to read {path}: {e}"))?;
    let (n_rows, n_samples, row_bytes, data_offset) = parse_bin01_header(&bytes, ctx)?;

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
        let src = &bytes[src_start..src_end];

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
    let ctx = "garfield::load_bin01_selected_rows_as_u64_words";
    if row_indices.is_empty() {
        return Err(format!("{ctx}: row_indices is empty"));
    }
    let bytes = fs::read(path).map_err(|e| format!("{ctx}: failed to read {path}: {e}"))?;
    let (n_rows_all, n_samples, row_bytes, data_offset) = parse_bin01_header(&bytes, ctx)?;

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
        let src = &bytes[src_start..src_end];

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

fn resolve_bin01_path(path: &str) -> Result<PathBuf, String> {
    let raw = path.trim();
    if raw.is_empty() {
        return Err("BIN path must not be empty".to_string());
    }
    let p = Path::new(raw);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    let mut os = p.as_os_str().to_os_string();
    os.push(".bin");
    let with_bin = PathBuf::from(os);
    if with_bin.exists() {
        return Ok(with_bin);
    }
    Err(format!("BIN file not found: {raw}"))
}

#[inline]
fn infer_site_sidecar_delimiter(line: &str) -> Option<char> {
    if line.contains('\t') {
        Some('\t')
    } else if line.contains(',') {
        Some(',')
    } else {
        None
    }
}

#[inline]
fn split_site_sidecar_tokens<'a>(line: &'a str, delimiter: Option<char>) -> Vec<&'a str> {
    match delimiter {
        Some(delim) => line
            .split(delim)
            .map(|x| x.trim())
            .filter(|x| !x.is_empty())
            .collect(),
        None => line.split_whitespace().collect(),
    }
}

fn read_site_sidecar_text(path: &Path) -> Result<Vec<SiteInfo>, String> {
    let fr = BufReader::new(File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?);
    let is_bim = path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| x.eq_ignore_ascii_case("bim"))
        .unwrap_or(false);
    let mut sites: Vec<SiteInfo> = Vec::new();
    let mut delimiter: Option<char> = None;
    let mut header_ready = false;
    let mut idx_chr = 0usize;
    let mut idx_pos = 1usize;
    let mut idx_ref = 2usize;
    let mut idx_alt = 3usize;

    for (ln, line) in fr.lines().enumerate() {
        let raw = line.map_err(|e| format!("read {}:{}: {e}", path.display(), ln + 1))?;
        let s = raw.trim();
        if s.is_empty() {
            continue;
        }
        if delimiter.is_none() {
            delimiter = infer_site_sidecar_delimiter(s);
        }
        let toks = split_site_sidecar_tokens(s, delimiter);
        if toks.is_empty() {
            continue;
        }
        if !header_ready {
            header_ready = true;
            let low: Vec<String> = toks.iter().map(|x| x.trim().to_ascii_lowercase()).collect();
            let pick = |cands: &[&str]| -> Option<usize> {
                low.iter().position(|v| cands.iter().any(|cand| v == cand))
            };
            if let (Some(chr_i), Some(pos_i), Some(ref_i), Some(alt_i)) = (
                pick(&["#chrom", "chrom", "chr", "chromosome"]),
                pick(&["pos", "bp", "position", "ps"]),
                pick(&["ref", "a0", "allele0", "allele_0", "ref_allele"]),
                pick(&["alt", "a1", "allele1", "allele_1", "alt_allele"]),
            ) {
                idx_chr = chr_i;
                idx_pos = pos_i;
                idx_ref = ref_i;
                idx_alt = alt_i;
                continue;
            }
        }
        if (is_bim || (idx_pos == 1 && idx_ref == 2 && idx_alt == 3)) && toks.len() >= 6 {
            idx_chr = 0;
            idx_pos = 3;
            idx_ref = 4;
            idx_alt = 5;
        }
        if toks.len() <= idx_alt {
            return Err(format!(
                "Malformed site sidecar line at {}:{}: {s}",
                path.display(),
                ln + 1
            ));
        }
        let pos = toks[idx_pos].parse::<i32>().map_err(|_| {
            format!(
                "invalid position at {}:{} -> {}",
                path.display(),
                ln + 1,
                toks[idx_pos]
            )
        })?;
        sites.push(SiteInfo {
            chrom: toks[idx_chr].to_string(),
            pos,
            ref_allele: toks[idx_ref].to_string(),
            alt_allele: toks[idx_alt].to_string(),
        });
    }
    if sites.is_empty() {
        return Err(format!("no site rows found in {}", path.display()));
    }
    Ok(sites)
}

fn load_bin01_sites(path: &str, n_rows: usize) -> Result<Option<Vec<SiteInfo>>, String> {
    if let Ok(it) = TxtSnpIter::new(path, None) {
        if it.sites.len() < n_rows {
            return Err(format!(
                "BIN sidecar site count mismatch: {} < n_rows={}",
                it.sites.len(),
                n_rows
            ));
        }
        return Ok(Some(it.sites));
    }
    let Some(site_path) = discover_site_sidecar(path) else {
        return Ok(None);
    };
    let sites = read_site_sidecar_text(&site_path)?;
    if sites.len() < n_rows {
        return Err(format!(
            "BIN sidecar site count mismatch: {} < n_rows={}",
            sites.len(),
            n_rows
        ));
    }
    Ok(Some(sites))
}

fn compute_bin01_group_ids(path: &str, n_rows: usize) -> Result<Vec<u64>, String> {
    let Some(sites) = load_bin01_sites(path, n_rows)? else {
        return Ok((0..n_rows).map(|i| i as u64).collect());
    };
    Ok(build_feature_group_ids(&sites, n_rows)
        .into_iter()
        .map(|g| g as u64)
        .collect())
}

fn load_bin01_packed_owned(
    path: &str,
    require_grouped_rows: bool,
) -> Result<(Vec<u8>, Vec<u64>, usize, usize, usize), String> {
    let ctx = if require_grouped_rows {
        "garfield::load_mbin_packed"
    } else {
        "garfield::load_bin01_packed"
    };
    let resolved = resolve_bin01_path(path)?;
    let resolved_str = resolved.to_string_lossy().to_string();
    let bytes = fs::read(&resolved)
        .map_err(|e| format!("{ctx}: failed to read {}: {e}", resolved.display()))?;
    let (n_rows, n_samples, row_bytes, data_offset) = parse_bin01_header(&bytes, ctx)?;
    let payload_len = n_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| format!("{ctx}: packed payload size overflow"))?;
    let payload_end = data_offset
        .checked_add(payload_len)
        .ok_or_else(|| format!("{ctx}: packed payload end overflow"))?;
    let packed = bytes[data_offset..payload_end].to_vec();
    let group_ids = compute_bin01_group_ids(&resolved_str, n_rows)?;
    if require_grouped_rows {
        let mut seen: HashSet<u64> = HashSet::with_capacity(group_ids.len());
        let mut has_duplicate_group = false;
        for &gid in group_ids.iter() {
            if !seen.insert(gid) {
                has_duplicate_group = true;
                break;
            }
        }
        if !has_duplicate_group {
            return Err(format!(
                "{ctx}: no repeated feature groups detected; this looks like a BIN cache, not MBIN"
            ));
        }
    }
    Ok((packed, group_ids, n_samples, n_rows, row_bytes))
}

#[inline]
fn append_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut os: OsString = prefix.as_os_str().to_os_string();
    os.push(suffix);
    PathBuf::from(os)
}

fn bin_prefix(path: &str) -> PathBuf {
    let p = Path::new(path);
    let mut out = p.to_path_buf();
    let is_bin = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("bin"))
        .unwrap_or(false);
    if is_bin {
        out.set_extension("");
    }
    out
}

fn discover_site_sidecar(path: &str) -> Option<PathBuf> {
    let prefix = bin_prefix(path);
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
        append_suffix(&prefix, ".bim"),
    ];
    for cand in candidates {
        if cand.exists() {
            return Some(cand);
        }
    }
    None
}

fn discover_id_sidecar(path: &str) -> Option<PathBuf> {
    let prefix = bin_prefix(path);
    let candidates = [
        append_suffix(&prefix, ".bin.id"),
        append_suffix(&prefix, ".id"),
        append_suffix(&prefix, ".fam"),
    ];
    for cand in candidates {
        if cand.exists() {
            return Some(cand);
        }
    }
    None
}

fn copy_site_sidecar(src_bin: &str, dst_bin: &str) -> Result<(), String> {
    let Some(src) = discover_site_sidecar(src_bin) else {
        return Ok(());
    };
    let src_name = src
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| "invalid sidecar filename".to_string())?;
    let src_prefix = bin_prefix(src_bin);
    let dst_prefix = bin_prefix(dst_bin);
    let src_prefix_name = src_prefix
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| "invalid src prefix".to_string())?;

    let suffix = src_name
        .strip_prefix(src_prefix_name)
        .unwrap_or("")
        .to_string();
    if suffix.is_empty() {
        return Ok(());
    }
    let dst = append_suffix(&dst_prefix, &suffix);
    fs::copy(&src, &dst)
        .map_err(|e| format!("copy {} -> {}: {e}", src.display(), dst.display()))?;
    Ok(())
}

fn copy_id_sidecar(src_bin: &str, dst_bin: &str) -> Result<(), String> {
    let Some(src) = discover_id_sidecar(src_bin) else {
        return Ok(());
    };
    let src_name = src
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| "invalid id-sidecar filename".to_string())?;
    let src_prefix = bin_prefix(src_bin);
    let dst_prefix = bin_prefix(dst_bin);
    let src_prefix_name = src_prefix
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| "invalid src prefix".to_string())?;

    let suffix = src_name
        .strip_prefix(src_prefix_name)
        .unwrap_or("")
        .to_string();
    if suffix.is_empty() {
        return Ok(());
    }
    let dst = append_suffix(&dst_prefix, &suffix);
    fs::copy(&src, &dst)
        .map_err(|e| format!("copy {} -> {}: {e}", src.display(), dst.display()))?;
    Ok(())
}

fn read_sample_ids_from_sidecar(path: &Path) -> Result<Vec<String>, String> {
    let fr = BufReader::new(File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?);
    let is_fam = path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| x.eq_ignore_ascii_case("fam"))
        .unwrap_or(false);
    let mut out: Vec<String> = Vec::new();
    for (ln, line) in fr.lines().enumerate() {
        let raw = line.map_err(|e| format!("read {}:{}: {e}", path.display(), ln + 1))?;
        let s = raw.trim();
        if s.is_empty() {
            continue;
        }
        if is_fam {
            let toks: Vec<&str> = s.split_whitespace().collect();
            if toks.len() < 2 {
                return Err(format!(
                    "{}:{} malformed FAM row (need >=2 columns)",
                    path.display(),
                    ln + 1
                ));
            }
            out.push(toks[1].to_string());
        } else {
            let toks: Vec<&str> = s.split_whitespace().collect();
            if toks.is_empty() {
                continue;
            }
            out.push(toks[0].to_string());
        }
    }
    Ok(out)
}

fn write_subset_id_sidecar(
    src_bin: &str,
    dst_bin: &str,
    sample_indices: &[usize],
) -> Result<bool, String> {
    let Some(src_id_path) = discover_id_sidecar(src_bin) else {
        return Ok(false);
    };
    let src_ids = read_sample_ids_from_sidecar(&src_id_path)?;
    let dst_id_path = out_bin_id_path(dst_bin);
    if let Some(parent) = dst_id_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let mut fw = BufWriter::new(
        File::create(&dst_id_path).map_err(|e| format!("create {}: {e}", dst_id_path.display()))?,
    );
    for &si in sample_indices.iter() {
        if si >= src_ids.len() {
            return Err(format!(
                "sample index out of range while subsetting IDs: {} >= {}",
                si,
                src_ids.len()
            ));
        }
        fw.write_all(src_ids[si].as_bytes())
            .map_err(|e| format!("write {}: {e}", dst_id_path.display()))?;
        fw.write_all(b"\n")
            .map_err(|e| format!("write {}: {e}", dst_id_path.display()))?;
    }
    fw.flush()
        .map_err(|e| format!("flush {}: {e}", dst_id_path.display()))?;
    Ok(true)
}

#[inline]
fn normalize_plink_prefix(path_or_prefix: &str) -> String {
    let s = path_or_prefix.trim();
    let low = s.to_ascii_lowercase();
    if low.ends_with(".bed") || low.ends_with(".bim") || low.ends_with(".fam") {
        s[..s.len() - 4].to_string()
    } else {
        s.to_string()
    }
}

fn make_input_reader(
    input_path: &str,
    input_kind: GarfieldInputKind,
) -> Result<GarfieldInputReader, String> {
    let p = input_path.trim();
    if p.is_empty() {
        return Err("input_path must not be empty".to_string());
    }
    let low = p.to_ascii_lowercase();
    match input_kind {
        GarfieldInputKind::Bfile => {
            let prefix = normalize_plink_prefix(p);
            let it = BedSnpIter::new_with_fill(&prefix, 0.0, 1.0, false, false, 0.02)?;
            Ok(GarfieldInputReader::Bed(it))
        }
        GarfieldInputKind::Vcf => {
            let it = VcfSnpIter::new_with_fill(p, 0.0, 1.0, false, false, 0.02)?;
            Ok(GarfieldInputReader::Vcf(it))
        }
        GarfieldInputKind::Hmp => {
            let it = HmpSnpIter::new_with_fill(p, 0.0, 1.0, false, false, 0.02)?;
            Ok(GarfieldInputReader::Hmp(it))
        }
        GarfieldInputKind::File => {
            let it = TxtSnpIter::new(p, None)?;
            Ok(GarfieldInputReader::Txt(it))
        }
        GarfieldInputKind::Auto => {
            if low.ends_with(".vcf") || low.ends_with(".vcf.gz") {
                let it = VcfSnpIter::new_with_fill(p, 0.0, 1.0, false, false, 0.02)?;
                return Ok(GarfieldInputReader::Vcf(it));
            }
            if low.ends_with(".hmp") || low.ends_with(".hmp.gz") {
                let it = HmpSnpIter::new_with_fill(p, 0.0, 1.0, false, false, 0.02)?;
                return Ok(GarfieldInputReader::Hmp(it));
            }
            let prefix = normalize_plink_prefix(p);
            let bed = format!("{prefix}.bed");
            let bim = format!("{prefix}.bim");
            let fam = format!("{prefix}.fam");
            if Path::new(&bed).exists() && Path::new(&bim).exists() && Path::new(&fam).exists() {
                let it = BedSnpIter::new_with_fill(&prefix, 0.0, 1.0, false, false, 0.02)?;
                return Ok(GarfieldInputReader::Bed(it));
            }
            let it = TxtSnpIter::new(p, None)?;
            Ok(GarfieldInputReader::Txt(it))
        }
    }
}

#[inline]
fn row_bytes_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(8).max(1)
}

#[inline]
fn normalize_genotype3(v: f32) -> Option<u8> {
    if !v.is_finite() || v < 0.0 {
        return None;
    }
    let r = v.round();
    if r <= 0.0 {
        Some(0)
    } else if r >= 2.0 {
        Some(2)
    } else {
        Some(1)
    }
}

fn write_bin01_header(file: &mut File, n_rows: u64, n_samples: usize) -> Result<(), String> {
    file.seek(SeekFrom::Start(0))
        .map_err(|e| format!("seek BIN header: {e}"))?;
    file.write_all(BIN01_MAGIC)
        .map_err(|e| format!("write BIN magic: {e}"))?;
    file.write_all(&n_rows.to_le_bytes())
        .map_err(|e| format!("write BIN n_rows: {e}"))?;
    file.write_all(&(n_samples as u64).to_le_bytes())
        .map_err(|e| format!("write BIN n_samples: {e}"))?;
    file.write_all(&(0u64).to_le_bytes())
        .map_err(|e| format!("write BIN reserved: {e}"))?;
    Ok(())
}

#[inline]
fn bin_output_prefix(out_bin_path: &str) -> PathBuf {
    let p = Path::new(out_bin_path);
    let mut out = p.to_path_buf();
    let is_bin = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("bin"))
        .unwrap_or(false);
    if is_bin {
        out.set_extension("");
    }
    out
}

#[inline]
fn out_bin_id_path(out_bin_path: &str) -> PathBuf {
    append_suffix(&bin_output_prefix(out_bin_path), ".bin.id")
}

#[inline]
fn out_bin_site_path(out_bin_path: &str) -> PathBuf {
    append_suffix(&bin_output_prefix(out_bin_path), ".bin.site")
}

fn write_sample_id_sidecar(out_bin_path: &str, sample_ids: &[String]) -> Result<(), String> {
    let id_path = out_bin_id_path(out_bin_path);
    if let Some(parent) = id_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create parent dir {}: {e}", parent.display()))?;
    }
    let mut fw = BufWriter::new(
        File::create(&id_path).map_err(|e| format!("create {}: {e}", id_path.display()))?,
    );
    for sid in sample_ids.iter() {
        fw.write_all(sid.as_bytes())
            .map_err(|e| format!("write {}: {e}", id_path.display()))?;
        fw.write_all(b"\n")
            .map_err(|e| format!("write {}: {e}", id_path.display()))?;
    }
    fw.flush()
        .map_err(|e| format!("flush {}: {e}", id_path.display()))?;
    Ok(())
}

fn row_select_by_indices(row: Vec<f32>, sample_indices: &[usize]) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(sample_indices.len());
    for &idx in sample_indices.iter() {
        if idx >= row.len() {
            return Err(format!(
                "sample index out of range while slicing row: {} >= {}",
                idx,
                row.len()
            ));
        }
        out.push(row[idx]);
    }
    Ok(out)
}

fn mbin_site_variants(site: &SiteInfo) -> [SiteInfo; 3] {
    let dom = SiteInfo {
        chrom: site.chrom.clone(),
        pos: site.pos,
        ref_allele: site.ref_allele.clone(),
        alt_allele: format!("{}|DOM", site.alt_allele),
    };
    let rec = SiteInfo {
        chrom: site.chrom.clone(),
        pos: site.pos,
        ref_allele: site.ref_allele.clone(),
        alt_allele: format!("{}|REC", site.alt_allele),
    };
    let het = SiteInfo {
        chrom: site.chrom.clone(),
        pos: site.pos,
        ref_allele: site.ref_allele.clone(),
        alt_allele: format!("{}|HET", site.alt_allele),
    };
    [dom, rec, het]
}

fn encode_row_to_bits(
    mut row: Vec<f32>,
    mut site: SiteInfo,
    mode: GarfieldBinMode,
    row_bytes: usize,
    maf: f32,
    geno: f32,
    impute: bool,
    het: f32,
) -> Option<Vec<EncodedRow>> {
    let keep = process_snp_row(
        &mut row,
        &mut site.ref_allele,
        &mut site.alt_allele,
        maf,
        geno,
        impute,
        false,
        het,
    );
    if !keep {
        return None;
    }
    if matches!(mode, GarfieldBinMode::Bin) {
        let mut non_missing = 0usize;
        let mut het_count = 0usize;
        for &v in row.iter() {
            if let Some(g) = normalize_genotype3(v) {
                non_missing += 1;
                if g == 1 {
                    het_count += 1;
                }
            }
        }
        if non_missing == 0 {
            return None;
        }
        let het_rate = het_count as f32 / non_missing as f32;
        if het_rate > het {
            return None;
        }
    }

    match mode {
        GarfieldBinMode::Bin => {
            let mut c0 = 0usize;
            let mut c2 = 0usize;
            for &v in row.iter() {
                match normalize_genotype3(v) {
                    Some(0) => c0 += 1,
                    Some(2) => c2 += 1,
                    _ => {}
                }
            }
            let mode02 = if c2 > c0 { 2u8 } else { 0u8 };
            let mut bits = vec![0u8; row_bytes];
            for (i, &v) in row.iter().enumerate() {
                let g = normalize_genotype3(v).unwrap_or(mode02);
                let is_one = if g == 0 {
                    false
                } else if g == 2 {
                    true
                } else {
                    mode02 == 2
                };
                if is_one {
                    bits[i >> 3] |= 1u8 << (i & 7);
                }
            }
            site.alt_allele = format!("{}|BIN", site.alt_allele);
            Some(vec![EncodedRow { site, bits }])
        }
        GarfieldBinMode::Mbin => {
            let mut dom = vec![0u8; row_bytes];
            let mut rec = vec![0u8; row_bytes];
            let mut het_bits = vec![0u8; row_bytes];
            for (i, &v) in row.iter().enumerate() {
                let Some(g) = normalize_genotype3(v) else {
                    continue;
                };
                let mask = 1u8 << (i & 7);
                if g > 0 {
                    dom[i >> 3] |= mask;
                }
                if g == 2 {
                    rec[i >> 3] |= mask;
                }
                if g == 1 {
                    het_bits[i >> 3] |= mask;
                }
            }
            let [s_dom, s_rec, s_het] = mbin_site_variants(&site);
            Some(vec![
                EncodedRow {
                    site: s_dom,
                    bits: dom,
                },
                EncodedRow {
                    site: s_rec,
                    bits: rec,
                },
                EncodedRow {
                    site: s_het,
                    bits: het_bits,
                },
            ])
        }
    }
}

fn encode_batch_rows(
    batch: Vec<(Vec<f32>, SiteInfo)>,
    mode: GarfieldBinMode,
    row_bytes: usize,
    maf: f32,
    geno: f32,
    impute: bool,
    het: f32,
    pool: Option<&rayon::ThreadPool>,
) -> Vec<Option<Vec<EncodedRow>>> {
    if let Some(p) = pool {
        p.install(|| {
            batch
                .into_par_iter()
                .map(|(row, site)| {
                    encode_row_to_bits(row, site, mode, row_bytes, maf, geno, impute, het)
                })
                .collect::<Vec<_>>()
        })
    } else {
        batch
            .into_iter()
            .map(|(row, site)| {
                encode_row_to_bits(row, site, mode, row_bytes, maf, geno, impute, het)
            })
            .collect::<Vec<_>>()
    }
}

struct GarfieldBinWriter {
    file: File,
    site_fw: BufWriter<File>,
    row_bytes: usize,
    n_rows_written: usize,
}

impl GarfieldBinWriter {
    fn new(out_bin_path: &str, n_samples: usize) -> Result<Self, String> {
        let out_path = Path::new(out_bin_path);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create parent dir {}: {e}", parent.display()))?;
        }
        let mut file =
            File::create(out_path).map_err(|e| format!("create {}: {e}", out_path.display()))?;
        write_bin01_header(&mut file, 0, n_samples)?;

        let site_path = out_bin_site_path(out_bin_path);
        if let Some(parent) = site_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create parent dir {}: {e}", parent.display()))?;
        }
        let site_fw = BufWriter::new(
            File::create(&site_path).map_err(|e| format!("create {}: {e}", site_path.display()))?,
        );
        Ok(Self {
            file,
            site_fw,
            row_bytes: row_bytes_for_samples(n_samples),
            n_rows_written: 0,
        })
    }

    fn write_rows(&mut self, rows: &[EncodedRow]) -> Result<(), String> {
        for row in rows.iter() {
            if row.bits.len() != self.row_bytes {
                return Err(format!(
                    "encoded row byte mismatch: got {}, expected {}",
                    row.bits.len(),
                    self.row_bytes
                ));
            }
            self.file
                .write_all(&row.bits)
                .map_err(|e| format!("write BIN row: {e}"))?;
            writeln!(
                self.site_fw,
                "{}\t{}\t{}\t{}",
                row.site.chrom, row.site.pos, row.site.ref_allele, row.site.alt_allele
            )
            .map_err(|e| format!("write BIN site row: {e}"))?;
            self.n_rows_written = self.n_rows_written.saturating_add(1);
        }
        Ok(())
    }

    fn finish(mut self, n_samples: usize) -> Result<usize, String> {
        self.site_fw
            .flush()
            .map_err(|e| format!("flush BIN site sidecar: {e}"))?;
        self.file
            .flush()
            .map_err(|e| format!("flush BIN file: {e}"))?;
        let n_rows_u64 =
            u64::try_from(self.n_rows_written).map_err(|_| "n_rows overflows u64".to_string())?;
        write_bin01_header(&mut self.file, n_rows_u64, n_samples)?;
        self.file
            .flush()
            .map_err(|e| format!("flush BIN header rewrite: {e}"))?;
        Ok(self.n_rows_written)
    }
}

#[inline]
fn normalize_chrom(chrom: &str) -> String {
    let s = chrom.trim();
    if s.len() > 2 && (s.ends_with("_1") || s.ends_with("_2")) {
        return s[..s.len() - 2].to_string();
    }
    if s.len() > 1 && (s.ends_with('-') || s.ends_with('+')) {
        return s[..s.len() - 1].to_string();
    }
    s.to_string()
}

fn build_chrom_index(sites: &[SiteInfo], n_rows: usize) -> HashMap<String, Vec<(i32, usize)>> {
    let mut idx: HashMap<String, Vec<(i32, usize)>> = HashMap::new();
    for (i, s) in sites.iter().take(n_rows).enumerate() {
        idx.entry(normalize_chrom(&s.chrom))
            .or_default()
            .push((s.pos, i));
    }
    for v in idx.values_mut() {
        v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    }
    idx
}

fn interval_indices(
    chrom_idx: &HashMap<String, Vec<(i32, usize)>>,
    chrom: &str,
    start: i32,
    end: i32,
) -> Vec<usize> {
    let key = normalize_chrom(chrom);
    let Some(v) = chrom_idx.get(&key) else {
        return Vec::new();
    };
    if v.is_empty() {
        return Vec::new();
    }
    let (lo, hi) = if start <= end {
        (start, end)
    } else {
        (end, start)
    };
    let mut out: Vec<usize> = Vec::new();
    for (pos, idx) in v.iter() {
        if *pos < lo {
            continue;
        }
        if *pos > hi {
            break;
        }
        out.push(*idx);
    }
    out
}

fn gather_rows_by_indices(
    bits_flat: &[u64],
    row_words: usize,
    row_indices: &[usize],
) -> Result<Vec<u64>, String> {
    let ctx = "garfield::gather_rows_by_indices";
    if row_indices.is_empty() {
        return Ok(Vec::new());
    }
    let n_rows_all = bits_flat.len() / row_words;
    let mut out = vec![0u64; row_indices.len() * row_words];
    for (dst_r, &src_r) in row_indices.iter().enumerate() {
        if src_r >= n_rows_all {
            return Err(format!(
                "{ctx}: row index out of range: {} >= {}",
                src_r, n_rows_all
            ));
        }
        let src_st = src_r * row_words;
        let dst_st = dst_r * row_words;
        out[dst_st..dst_st + row_words].copy_from_slice(&bits_flat[src_st..src_st + row_words]);
    }
    Ok(out)
}

fn build_feature_group_ids(sites: &[SiteInfo], n_rows: usize) -> Vec<usize> {
    let mut map: HashMap<(String, i32), usize> = HashMap::new();
    let mut out = vec![0usize; n_rows];
    for (i, site) in sites.iter().take(n_rows).enumerate() {
        let key = (normalize_chrom(&site.chrom), site.pos);
        let gid = if let Some(g) = map.get(&key) {
            *g
        } else {
            let g = map.len();
            map.insert(key, g);
            g
        };
        out[i] = gid;
    }
    out
}

fn run_beam_with_feature_exclusion(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    response: GarfieldResponse,
    y_train_f64: &[f64],
    y_train_bin: Option<&[u8]>,
    max_pick: usize,
    beam_width: usize,
    local_group_ids: Option<&[usize]>,
) -> Result<BeamAndResult, String> {
    if let Some(groups) = local_group_ids {
        return match response {
            GarfieldResponse::Binary => {
                let yb = y_train_bin.ok_or_else(|| "binary y is not prepared".to_string())?;
                beam_search_and_with_group_exclusion(
                    bits_flat,
                    row_words,
                    n_rows,
                    n_samples,
                    max_pick,
                    beam_width,
                    groups,
                    |combined| score_binary_mcc_packed(yb, combined, n_samples),
                )
            }
            GarfieldResponse::Continuous => beam_search_and_with_group_exclusion(
                bits_flat,
                row_words,
                n_rows,
                n_samples,
                max_pick,
                beam_width,
                groups,
                |combined| score_cont_corr_packed(y_train_f64, combined, n_samples).abs(),
            ),
        };
    }

    match response {
        GarfieldResponse::Binary => {
            let yb = y_train_bin.ok_or_else(|| "binary y is not prepared".to_string())?;
            beam_search_and_binary_mcc(
                yb, bits_flat, row_words, n_rows, n_samples, max_pick, beam_width,
            )
        }
        GarfieldResponse::Continuous => beam_search_and_continuous_abs_corr(
            y_train_f64,
            bits_flat,
            row_words,
            n_rows,
            n_samples,
            max_pick,
            beam_width,
        ),
    }
}

fn build_windows_from_sites(
    sites: &[SiteInfo],
    n_rows: usize,
    extension: usize,
    step: usize,
) -> Vec<GarfieldWindow> {
    if n_rows == 0 || sites.is_empty() || extension == 0 || step == 0 {
        return vec![GarfieldWindow {
            chrom: "ALL".to_string(),
            bp_start: 0,
            bp_end: 0,
            indices: (0..n_rows).collect(),
        }];
    }

    let mut groups: HashMap<String, Vec<(i32, usize)>> = HashMap::new();
    let mut chrom_order: Vec<String> = Vec::new();
    for (idx, site) in sites.iter().enumerate().take(n_rows) {
        let chrom = normalize_chrom(&site.chrom);
        if !groups.contains_key(&chrom) {
            groups.insert(chrom.clone(), Vec::new());
            chrom_order.push(chrom.clone());
        }
        if let Some(v) = groups.get_mut(&chrom) {
            v.push((site.pos, idx));
        }
    }
    if groups.is_empty() {
        return vec![GarfieldWindow {
            chrom: "ALL".to_string(),
            bp_start: 0,
            bp_end: 0,
            indices: (0..n_rows).collect(),
        }];
    }

    let mut windows: Vec<GarfieldWindow> = Vec::new();
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
                    windows.push(GarfieldWindow {
                        chrom: chrom.clone(),
                        bp_start: left_i64 as i32,
                        bp_end: right_i64 as i32,
                        indices: chunk,
                    });
                    prev_sig = Some(sig);
                }
            }

            if center >= max_bp {
                break;
            }
            center = (center as i64).saturating_add(step as i64) as i32;
            if center <= min_bp && step > 0 {
                break;
            }
        }

        let has_chrom_window = windows.last().map(|w| w.chrom == chrom).unwrap_or(false);
        if !has_chrom_window {
            windows.push(GarfieldWindow {
                chrom: chrom.clone(),
                bp_start: min_bp,
                bp_end: max_bp.saturating_add(1),
                indices: idxs,
            });
        }
    }

    if windows.is_empty() {
        vec![GarfieldWindow {
            chrom: "ALL".to_string(),
            bp_start: 0,
            bp_end: 0,
            indices: (0..n_rows).collect(),
        }]
    } else {
        windows
    }
}

fn read_y_f64(arr: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    match arr.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => arr.as_array().iter().copied().collect(),
    }
}

fn validate_binary_y(y: &[f64]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(y.len());
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("binary y contains non-finite value at index {}", i));
        }
        if v == 0.0 {
            out.push(0u8);
        } else if v == 1.0 {
            out.push(1u8);
        } else {
            return Err(format!(
                "binary response requires y in {{0,1}}; got y[{}]={}",
                i, v
            ));
        }
    }
    Ok(out)
}

fn validate_continuous_y(y: &[f64]) -> Result<(), String> {
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!(
                "continuous y contains non-finite value at index {}",
                i
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct GarfieldLogicUnit {
    label: String,
    indices: Vec<usize>,
}

#[derive(Clone, Debug)]
struct GarfieldLogicBits {
    bits_flat: Vec<u64>,
    row_words: usize,
    sample_ids: Vec<String>,
    sites: Vec<SiteInfo>,
    group_ids: Vec<usize>,
    n_samples: usize,
}

#[derive(Clone, Debug)]
struct GarfieldLogicRuleRecord {
    unit_name: String,
    unit_kind: String,
    unit_index: usize,
    region_size: usize,
    ml_feature_count: usize,
    selected_row_indices: Vec<usize>,
    snp_name: String,
    expr: String,
    chrom_field: String,
    pos: i32,
    beta: f64,
    se: f64,
    chisq: f64,
    pwald: f64,
    train_score: f64,
    test_score: f64,
    full_bits: Vec<u64>,
}

#[derive(Clone, Debug)]
struct GarfieldLogicPipelineResult {
    pseudo_prefix: Option<String>,
    rules_tsv: Option<String>,
    records: Vec<GarfieldLogicRuleRecord>,
    simbench_count: usize,
    split_applied: bool,
    n_train: usize,
    n_test: usize,
    n_samples: usize,
    units_total: usize,
    units_scanned: usize,
    train_fit: GarfieldResidualResult,
    test_fit: GarfieldResidualResult,
}

#[derive(Clone, Copy, Debug, Default)]
struct GarfieldRuleWald {
    beta: f64,
    se: f64,
    chisq: f64,
    pwald: f64,
}

#[inline]
fn nan_garfield_rule_wald() -> GarfieldRuleWald {
    GarfieldRuleWald {
        beta: f64::NAN,
        se: f64::NAN,
        chisq: f64::NAN,
        pwald: f64::NAN,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimBenchLogic {
    Single,
    And,
    Or,
}

#[derive(Clone, Debug)]
struct SimBenchTerm {
    term_id: usize,
    kind: String,
    logic: SimBenchLogic,
    logic_text: String,
    sites_text: String,
    sites: Vec<(String, i32)>,
    label: String,
}

fn parse_simbench_logic(raw: &str) -> Result<SimBenchLogic, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "single" => Ok(SimBenchLogic::Single),
        "and" => Ok(SimBenchLogic::And),
        "or" => Ok(SimBenchLogic::Or),
        other => Err(format!(
            "simbench logic must be one of: single, and, or; got {other}"
        )),
    }
}

fn parse_simbench_sites(raw: &str) -> Result<Vec<(String, i32)>, String> {
    let mut out = Vec::<(String, i32)>::new();
    for token in raw.split(';') {
        let site_txt = token.trim();
        if site_txt.is_empty() {
            continue;
        }
        let Some((chrom, pos_txt)) = site_txt.split_once(':') else {
            return Err(format!(
                "invalid simbench site token '{site_txt}'; expected chrom:pos"
            ));
        };
        let pos = pos_txt.trim().parse::<i32>().map_err(|e| {
            format!("invalid simbench site position '{pos_txt}' in '{site_txt}': {e}")
        })?;
        out.push((chrom.trim().to_string(), pos));
    }
    if out.is_empty() {
        return Err("simbench sites column is empty".to_string());
    }
    Ok(out)
}

fn parse_simbench_terms(path: &str) -> Result<Vec<SimBenchTerm>, String> {
    let file = File::open(path).map_err(|e| format!("open simbench file {path}: {e}"))?;
    let mut rdr = BufReader::new(file);
    let mut header = String::new();
    let n_read = rdr
        .read_line(&mut header)
        .map_err(|e| format!("read simbench header {path}: {e}"))?;
    if n_read == 0 {
        return Err(format!("simbench file is empty: {path}"));
    }
    let cols = header
        .trim_end_matches(&['\r', '\n'][..])
        .split('\t')
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    let mut col_idx = HashMap::<String, usize>::new();
    for (i, col) in cols.iter().enumerate() {
        col_idx.insert(col.to_ascii_lowercase(), i);
    }
    let term_idx = col_idx
        .get("term_id")
        .copied()
        .ok_or_else(|| format!("simbench file {path} is missing required column: term_id"))?;
    let kind_idx = col_idx
        .get("kind")
        .copied()
        .ok_or_else(|| format!("simbench file {path} is missing required column: kind"))?;
    let logic_idx = col_idx
        .get("logic")
        .copied()
        .ok_or_else(|| format!("simbench file {path} is missing required column: logic"))?;
    let sites_idx = col_idx
        .get("sites")
        .copied()
        .ok_or_else(|| format!("simbench file {path} is missing required column: sites"))?;
    let label_idx = col_idx
        .get("label")
        .copied()
        .ok_or_else(|| format!("simbench file {path} is missing required column: label"))?;

    let mut out = Vec::<SimBenchTerm>::new();
    for (line_no0, line_res) in rdr.lines().enumerate() {
        let line_no = line_no0 + 2;
        let line = line_res.map_err(|e| format!("read {path}:{line_no}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').map(|s| s.trim()).collect::<Vec<_>>();
        let get_field = |idx: usize| -> &str {
            if idx < fields.len() {
                fields[idx]
            } else {
                ""
            }
        };
        let term_id = get_field(term_idx)
            .parse::<usize>()
            .unwrap_or(out.len() + 1);
        let kind = get_field(kind_idx).to_string();
        let logic_text = get_field(logic_idx).to_string();
        let logic =
            parse_simbench_logic(&logic_text).map_err(|e| format!("{path}:{line_no}: {e}"))?;
        let sites_text = get_field(sites_idx).to_string();
        let sites =
            parse_simbench_sites(&sites_text).map_err(|e| format!("{path}:{line_no}: {e}"))?;
        let label = get_field(label_idx).to_string();
        out.push(SimBenchTerm {
            term_id,
            kind,
            logic,
            logic_text,
            sites_text,
            sites,
            label,
        });
    }
    Ok(out)
}

fn collapse_logic_bin01_from_raw_row(row: &[f32]) -> Result<Vec<u8>, String> {
    if row.is_empty() {
        return Err("empty genotype row".to_string());
    }
    let mut valid = Vec::<u8>::with_capacity(row.len());
    let mut valid_idx = Vec::<usize>::with_capacity(row.len());
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
        return Err("all genotype values are missing".to_string());
    }
    let c0 = valid.iter().filter(|&&g| g == 0).count();
    let c2 = valid.iter().filter(|&&g| g == 2).count();
    let mode02 = if c2 > c0 { 2u8 } else { 0u8 };
    let mut out = vec![mode02; row.len()];
    for (&idx, &g) in valid_idx.iter().zip(valid.iter()) {
        out[idx] = if g == 1 { mode02 } else { g };
    }
    Ok(out
        .into_iter()
        .map(|g| if g > 0 { 1u8 } else { 0u8 })
        .collect())
}

fn combine_logic_bin01_rows(rows: &[Vec<u8>], logic: SimBenchLogic) -> Result<Vec<u8>, String> {
    if rows.is_empty() {
        return Err("simbench term has no genotype rows".to_string());
    }
    let n = rows[0].len();
    if rows.iter().any(|r| r.len() != n) {
        return Err("simbench genotype rows have inconsistent sample lengths".to_string());
    }
    let mut out = rows[0].clone();
    match logic {
        SimBenchLogic::Single => {
            if rows.len() != 1 {
                return Err(format!(
                    "simbench logic 'single' expects exactly 1 site, got {}",
                    rows.len()
                ));
            }
        }
        SimBenchLogic::And => {
            for row in rows.iter().skip(1) {
                for (dst, &v) in out.iter_mut().zip(row.iter()) {
                    *dst &= v;
                }
            }
        }
        SimBenchLogic::Or => {
            for row in rows.iter().skip(1) {
                for (dst, &v) in out.iter_mut().zip(row.iter()) {
                    *dst |= v;
                }
            }
        }
    }
    Ok(out)
}

fn pack_bin01_to_words(bits: &[u8]) -> Vec<u64> {
    let row_words = words_for_samples(bits.len());
    let mut out = vec![0u64; row_words];
    for (i, &v) in bits.iter().enumerate() {
        if v != 0 {
            out[i >> 6] |= 1u64 << (i & 63);
        }
    }
    apply_tail_mask(&mut out, tail_mask(bits.len()));
    out
}

fn simbench_logic_symbol(logic: SimBenchLogic) -> &'static str {
    match logic {
        SimBenchLogic::Single => "",
        SimBenchLogic::And => " & ",
        SimBenchLogic::Or => " | ",
    }
}

fn simbench_rule_name(logic: SimBenchLogic, sites: &[SiteInfo], label: &str) -> String {
    let trimmed = label.trim();
    if !trimmed.is_empty() {
        trimmed.to_string()
    } else {
        sites
            .iter()
            .map(site_base_label)
            .collect::<Vec<_>>()
            .join(simbench_logic_symbol(logic))
    }
}

fn simbench_rule_expr(logic: SimBenchLogic, sites: &[SiteInfo]) -> Result<String, String> {
    let mut it = sites.iter();
    let Some(first) = it.next() else {
        return Err("simbench term has no sites".to_string());
    };
    let mut out = literal_expr(first, false);
    let op_txt = match logic {
        SimBenchLogic::Single => "",
        SimBenchLogic::And => "AND",
        SimBenchLogic::Or => "OR",
    };
    if logic != SimBenchLogic::Single {
        for site in it {
            out.push(' ');
            out.push_str(op_txt);
            out.push(' ');
            out.push_str(&literal_expr(site, false));
        }
    }
    Ok(out)
}

fn evaluate_simbench_terms(
    prefix: &str,
    simbench_path: &str,
    selected_sample_indices: &[usize],
    train_idx_local: &[usize],
    assoc_sample_indices: &[usize],
    y_train: &[f64],
    y_assoc: &[f64],
    beam_params: BeamSearchParams,
) -> Result<Vec<GarfieldLogicRuleRecord>, String> {
    let terms = parse_simbench_terms(simbench_path)?;
    if terms.is_empty() {
        return Ok(Vec::new());
    }
    let bed = BedSnpIter::new_with_fill(prefix, 0.0, 1.0, false, false, 1.0)?;
    let mut site_lookup = HashMap::<(String, i32), usize>::with_capacity(bed.sites.len());
    for (idx, site) in bed.sites.iter().enumerate() {
        site_lookup
            .entry((normalize_chrom(&site.chrom), site.pos))
            .or_insert(idx);
    }

    let n_selected = selected_sample_indices.len();
    let row_words_full = words_for_samples(n_selected);
    let mut out = Vec::<GarfieldLogicRuleRecord>::with_capacity(terms.len());
    for term in terms.into_iter() {
        let mut gate_rows = Vec::<Vec<u8>>::with_capacity(term.sites.len());
        let mut member_bits_full = Vec::<Vec<u64>>::with_capacity(term.sites.len());
        let mut bench_sites = Vec::<SiteInfo>::with_capacity(term.sites.len());
        let mut selected_row_indices = Vec::<usize>::with_capacity(term.sites.len());
        for (chrom, pos) in term.sites.iter() {
            let key = (normalize_chrom(chrom), *pos);
            let src_idx = *site_lookup.get(&key).ok_or_else(|| {
                format!(
                    "simbench term {} site not found in BED/BIM: {}:{}",
                    term.term_id, chrom, pos
                )
            })?;
            let (row, site) = bed
                .get_snp_row_selected_raw(src_idx, selected_sample_indices)
                .ok_or_else(|| {
                    format!(
                        "simbench term {} failed to decode site {}:{} from BED",
                        term.term_id, chrom, pos
                    )
                })?;
            gate_rows.push(collapse_logic_bin01_from_raw_row(&row).map_err(|e| {
                format!(
                    "simbench term {} failed to collapse site {}:{} to BIN01: {e}",
                    term.term_id, chrom, pos
                )
            })?);
            member_bits_full.push(pack_bin01_to_words(
                gate_rows
                    .last()
                    .ok_or_else(|| "simbench collapsed row missing".to_string())?
                    .as_slice(),
            ));
            bench_sites.push(site);
            selected_row_indices.push(src_idx);
        }
        let gate = combine_logic_bin01_rows(gate_rows.as_slice(), term.logic).map_err(|e| {
            format!(
                "simbench term {} ('{}' {} {}) failed: {e}",
                term.term_id, term.kind, term.logic_text, term.sites_text
            )
        })?;
        let full_bits = pack_bin01_to_words(gate.as_slice());
        let (train_bits, _) = packed_rows_subset_from_full_bits(
            full_bits.as_slice(),
            row_words_full,
            &[0usize],
            train_idx_local,
            1,
            n_selected,
        )?;
        let (assoc_bits, _) = packed_rows_subset_from_full_bits(
            full_bits.as_slice(),
            row_words_full,
            &[0usize],
            assoc_sample_indices,
            1,
            n_selected,
        )?;
        let train_sc = score_cont_weighted_mean_diff_packed(
            y_train,
            train_bits.as_slice(),
            train_idx_local.len(),
        );
        let assoc_sc = score_cont_weighted_mean_diff_packed(
            y_assoc,
            assoc_bits.as_slice(),
            assoc_sample_indices.len(),
        );
        let mut max_singleton_train_raw = f64::NEG_INFINITY;
        let mut max_singleton_assoc_raw = f64::NEG_INFINITY;
        for member_bits in member_bits_full.iter() {
            let (member_train_bits, _) = packed_rows_subset_from_full_bits(
                member_bits.as_slice(),
                row_words_full,
                &[0usize],
                train_idx_local,
                1,
                n_selected,
            )?;
            let (member_assoc_bits, _) = packed_rows_subset_from_full_bits(
                member_bits.as_slice(),
                row_words_full,
                &[0usize],
                assoc_sample_indices,
                1,
                n_selected,
            )?;
            let sc_train = score_cont_weighted_mean_diff_packed(
                y_train,
                member_train_bits.as_slice(),
                train_idx_local.len(),
            );
            let sc_assoc = score_cont_weighted_mean_diff_packed(
                y_assoc,
                member_assoc_bits.as_slice(),
                assoc_sample_indices.len(),
            );
            max_singleton_train_raw = max_singleton_train_raw.max(sc_train.raw_score);
            max_singleton_assoc_raw = max_singleton_assoc_raw.max(sc_assoc.raw_score);
        }
        if !max_singleton_train_raw.is_finite() {
            max_singleton_train_raw = train_sc.raw_score;
        }
        if !max_singleton_assoc_raw.is_finite() {
            max_singleton_assoc_raw = assoc_sc.raw_score;
        }
        let train_score = rank_rule_score_components(
            term.sites.len(),
            0,
            train_sc.raw_score,
            max_singleton_train_raw,
            &beam_params,
        );
        let test_score = rank_rule_score_components(
            term.sites.len(),
            0,
            assoc_sc.raw_score,
            max_singleton_assoc_raw,
            &beam_params,
        );
        let assoc =
            fit_binary_rule_wald_from_bits(y_assoc, full_bits.as_slice(), assoc_sample_indices);
        let first_site = bench_sites
            .first()
            .ok_or_else(|| format!("simbench term {} has no sites", term.term_id))?;
        out.push(GarfieldLogicRuleRecord {
            unit_name: simbench_rule_name(term.logic, bench_sites.as_slice(), &term.label),
            unit_kind: "simbench".to_string(),
            unit_index: term.term_id,
            region_size: bench_sites.len(),
            ml_feature_count: 0,
            selected_row_indices,
            snp_name: simbench_rule_name(term.logic, bench_sites.as_slice(), &term.label),
            expr: simbench_rule_expr(term.logic, bench_sites.as_slice())?,
            chrom_field: first_site.chrom.clone(),
            pos: first_site.pos,
            beta: assoc.beta,
            se: assoc.se,
            chisq: assoc.chisq,
            pwald: assoc.pwald,
            train_score,
            test_score,
            full_bits,
        });
    }
    Ok(out)
}

fn fit_binary_rule_wald_from_bits(
    y: &[f64],
    full_bits: &[u64],
    sample_indices: &[usize],
) -> GarfieldRuleWald {
    if y.len() != sample_indices.len() || y.len() < 3 {
        return nan_garfield_rule_wald();
    }
    let mut n_hit = 0usize;
    let mut n_miss = 0usize;
    let mut sum_hit = 0.0_f64;
    let mut sum_miss = 0.0_f64;
    let mut ss_hit = 0.0_f64;
    let mut ss_miss = 0.0_f64;

    for (&yv, &idx) in y.iter().zip(sample_indices.iter()) {
        let hit = ((full_bits[idx >> 6] >> (idx & 63)) & 1u64) != 0;
        if hit {
            n_hit += 1;
            sum_hit += yv;
            ss_hit += yv * yv;
        } else {
            n_miss += 1;
            sum_miss += yv;
            ss_miss += yv * yv;
        }
    }
    if n_hit == 0 || n_miss == 0 {
        return nan_garfield_rule_wald();
    }

    let mean_hit = sum_hit / n_hit as f64;
    let mean_miss = sum_miss / n_miss as f64;
    let beta = mean_hit - mean_miss;
    let df = (y.len() as i32) - 2;
    if df <= 0 {
        return nan_garfield_rule_wald();
    }

    let sse_hit = (ss_hit - (sum_hit * sum_hit) / n_hit as f64).max(0.0);
    let sse_miss = (ss_miss - (sum_miss * sum_miss) / n_miss as f64).max(0.0);
    let sigma2 = (sse_hit + sse_miss) / (df as f64);
    let se2 = sigma2 * ((1.0 / n_hit as f64) + (1.0 / n_miss as f64));
    if !se2.is_finite() || se2 < 0.0 {
        return nan_garfield_rule_wald();
    }
    if se2 == 0.0 {
        if beta.abs() <= 1e-15 {
            return GarfieldRuleWald {
                beta,
                se: 0.0,
                chisq: 0.0,
                pwald: 1.0,
            };
        }
        return GarfieldRuleWald {
            beta,
            se: 0.0,
            chisq: f64::INFINITY,
            pwald: f64::MIN_POSITIVE,
        };
    }
    let se = se2.sqrt();
    let t = beta / se;
    GarfieldRuleWald {
        beta,
        se,
        chisq: t * t,
        pwald: student_t_p_two_sided(t, df),
    }
}

#[inline]
fn cmp_logic_rule_records(
    a: &GarfieldLogicRuleRecord,
    b: &GarfieldLogicRuleRecord,
) -> std::cmp::Ordering {
    b.test_score
        .partial_cmp(&a.test_score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| {
            b.train_score
                .partial_cmp(&a.train_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            a.selected_row_indices
                .len()
                .cmp(&b.selected_row_indices.len())
        })
        .then_with(|| a.snp_name.cmp(&b.snp_name))
        .then_with(|| a.unit_name.cmp(&b.unit_name))
}

fn dedup_logic_rule_records(records: Vec<GarfieldLogicRuleRecord>) -> Vec<GarfieldLogicRuleRecord> {
    let mut best_by_bits = HashMap::<Vec<u64>, GarfieldLogicRuleRecord>::new();
    for rec in records.into_iter() {
        match best_by_bits.entry(rec.full_bits.clone()) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(rec);
            }
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                if cmp_logic_rule_records(&rec, slot.get()) == std::cmp::Ordering::Less {
                    slot.insert(rec);
                }
            }
        }
    }
    let mut out = best_by_bits.into_values().collect::<Vec<_>>();
    out.sort_by(cmp_logic_rule_records);
    out
}

#[inline]
fn chrom_sort_key(chrom: &str) -> (u8, i32, String) {
    let norm = normalize_chrom(chrom);
    let bare = norm
        .strip_prefix("chr")
        .or_else(|| norm.strip_prefix("CHR"))
        .unwrap_or(norm.as_str());
    if let Ok(v) = bare.parse::<i32>() {
        return (0u8, v, norm);
    }
    let upper = bare.to_ascii_uppercase();
    match upper.as_str() {
        "X" => (1u8, 23, norm),
        "Y" => (1u8, 24, norm),
        "M" | "MT" => (1u8, 25, norm),
        _ => (2u8, i32::MAX, norm),
    }
}

#[inline]
fn parse_logic_record_primary_site(rec: &GarfieldLogicRuleRecord) -> Option<(String, i32)> {
    let first = rec.snp_name.split('&').next()?.trim();
    let (chrom, pos_txt) = first.rsplit_once('_')?;
    let pos = pos_txt.parse::<i32>().ok()?;
    Some((normalize_chrom(chrom), pos))
}

#[inline]
fn cmp_logic_rule_records_plink_order(
    a: &GarfieldLogicRuleRecord,
    b: &GarfieldLogicRuleRecord,
) -> std::cmp::Ordering {
    let (a_chrom, a_pos) = parse_logic_record_primary_site(a)
        .unwrap_or_else(|| (normalize_chrom(&a.chrom_field), a.pos));
    let (b_chrom, b_pos) = parse_logic_record_primary_site(b)
        .unwrap_or_else(|| (normalize_chrom(&b.chrom_field), b.pos));
    chrom_sort_key(&a_chrom)
        .cmp(&chrom_sort_key(&b_chrom))
        .then_with(|| a_pos.cmp(&b_pos))
        .then_with(|| a.snp_name.cmp(&b.snp_name))
        .then_with(|| a.unit_name.cmp(&b.unit_name))
}

#[inline]
fn effective_threads_local(threads: usize) -> usize {
    if threads > 0 {
        threads.max(1)
    } else {
        std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
            .max(1)
    }
}

#[inline]
fn mask_to_indices(mask: &[bool], choose_value: bool) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(i, &v)| if v == choose_value { Some(i) } else { None })
        .collect()
}

fn subset_vec_f64(values: &[f64], indices: &[usize]) -> Result<Vec<f64>, String> {
    let mut out = Vec::<f64>::with_capacity(indices.len());
    for &idx in indices.iter() {
        let v = *values
            .get(idx)
            .ok_or_else(|| format!("sample index out of range while subsetting y: {idx}"))?;
        out.push(v);
    }
    Ok(out)
}

fn subset_cov_f64(
    x_cov: Option<&[f64]>,
    n_samples: usize,
    q_cov: usize,
    indices: &[usize],
) -> Result<Option<Vec<f64>>, String> {
    let Some(cov) = x_cov else {
        return Ok(None);
    };
    if cov.len() != n_samples.saturating_mul(q_cov) {
        return Err(format!(
            "x_cov payload length mismatch: got {}, expected {}",
            cov.len(),
            n_samples.saturating_mul(q_cov)
        ));
    }
    let mut out = vec![0.0_f64; indices.len().saturating_mul(q_cov)];
    for (dst_r, &src_r) in indices.iter().enumerate() {
        if src_r >= n_samples {
            return Err(format!(
                "sample index out of range while subsetting x_cov: {src_r}"
            ));
        }
        let src_st = src_r * q_cov;
        let dst_st = dst_r * q_cov;
        out[dst_st..dst_st + q_cov].copy_from_slice(&cov[src_st..src_st + q_cov]);
    }
    Ok(Some(out))
}

#[inline]
fn site_base_label(site: &SiteInfo) -> String {
    format!("{}_{}", site.chrom, site.pos)
}

#[inline]
fn site_model_label(site: &SiteInfo) -> &'static str {
    let alt = site.alt_allele.to_ascii_uppercase();
    if alt.contains("|DOM") {
        "DOM"
    } else if alt.contains("|REC") {
        "REC"
    } else if alt.contains("|HET") {
        "HET"
    } else {
        "BIN"
    }
}

#[inline]
fn literal_expr(site: &SiteInfo, negated: bool) -> String {
    if negated {
        format!("NOT {}({})", site_model_label(site), site_base_label(site))
    } else {
        format!("{}({})", site_model_label(site), site_base_label(site))
    }
}

fn literal_name(site: &SiteInfo, negated: bool) -> String {
    let base = site_base_label(site);
    if negated {
        format!("!{base}")
    } else {
        base
    }
}

fn rule_expr(rule: &BeamRule, local_sites: &[SiteInfo]) -> Result<String, String> {
    let first = local_sites
        .get(rule.first.row_index)
        .ok_or_else(|| format!("rule row index out of range: {}", rule.first.row_index))?;
    let mut out = literal_expr(first, rule.first.negated);
    for (op, lit) in rule.rest.iter() {
        let site = local_sites
            .get(lit.row_index)
            .ok_or_else(|| format!("rule row index out of range: {}", lit.row_index))?;
        let op_txt = match op {
            BeamBinaryOp::And => "AND",
            BeamBinaryOp::Or => "OR",
        };
        out.push(' ');
        out.push_str(op_txt);
        out.push(' ');
        out.push_str(&literal_expr(site, lit.negated));
    }
    Ok(out)
}

fn rule_snp_name(rule: &BeamRule, local_sites: &[SiteInfo]) -> Result<String, String> {
    let first = local_sites
        .get(rule.first.row_index)
        .ok_or_else(|| format!("rule row index out of range: {}", rule.first.row_index))?;
    let mut out = literal_name(first, rule.first.negated);
    for (op, lit) in rule.rest.iter() {
        let site = local_sites
            .get(lit.row_index)
            .ok_or_else(|| format!("rule row index out of range: {}", lit.row_index))?;
        let op_txt = match op {
            BeamBinaryOp::And => " & ",
            BeamBinaryOp::Or => " | ",
        };
        out.push_str(op_txt);
        out.push_str(&literal_name(site, lit.negated));
    }
    Ok(out)
}

fn rule_selected_global_rows(
    rule: &BeamRule,
    selected_global_rows: &[usize],
) -> Result<Vec<usize>, String> {
    let mut out = Vec::<usize>::with_capacity(rule.len());
    let first = *selected_global_rows
        .get(rule.first.row_index)
        .ok_or_else(|| format!("rule row index out of range: {}", rule.first.row_index))?;
    out.push(first);
    for (_, lit) in rule.rest.iter() {
        let idx = *selected_global_rows
            .get(lit.row_index)
            .ok_or_else(|| format!("rule row index out of range: {}", lit.row_index))?;
        out.push(idx);
    }
    Ok(out)
}

fn build_logic_units_from_groups(
    sites: &[SiteInfo],
    groups: &[Vec<(String, i32, i32)>],
    group_names: Option<&[String]>,
) -> Vec<GarfieldLogicUnit> {
    let chrom_idx = build_chrom_index(sites, sites.len());
    let mut out = Vec::<GarfieldLogicUnit>::new();
    for (gi, group) in groups.iter().enumerate() {
        let mut idx_all: Vec<usize> = Vec::new();
        for (chrom, start, end) in group.iter() {
            let mut iv = interval_indices(&chrom_idx, chrom, *start, *end);
            idx_all.append(&mut iv);
        }
        if idx_all.is_empty() {
            continue;
        }
        idx_all.sort_unstable();
        idx_all.dedup();
        let label = group_names
            .and_then(|v| v.get(gi).cloned())
            .unwrap_or_else(|| format!("group_{}", gi + 1));
        out.push(GarfieldLogicUnit {
            label,
            indices: idx_all,
        });
    }
    out
}

fn build_logic_units(
    sites: &[SiteInfo],
    unit_kind: &str,
    groups: Option<&[Vec<(String, i32, i32)>]>,
    group_names: Option<&[String]>,
    extension: usize,
    step: Option<usize>,
) -> Result<Vec<GarfieldLogicUnit>, String> {
    let kind = unit_kind.trim().to_ascii_lowercase();
    match kind.as_str() {
        "" | "window" => {
            let ext = extension.max(1);
            let step_eff = step.unwrap_or((ext / 2).max(1)).max(1);
            let windows = build_windows_from_sites(sites, sites.len(), ext, step_eff);
            Ok(windows
                .into_iter()
                .map(|w| GarfieldLogicUnit {
                    label: format!("{}:{}-{}", w.chrom, w.bp_start, w.bp_end),
                    indices: w.indices,
                })
                .collect())
        }
        "gene" | "geneset" | "group" => {
            let g = groups.ok_or_else(|| {
                "groups must be provided for unit_kind in {gene, geneset, group}".to_string()
            })?;
            Ok(build_logic_units_from_groups(sites, g, group_names))
        }
        other => Err(format!(
            "unit_kind must be one of: window, gene, geneset, group; got {other}"
        )),
    }
}

fn packed_row_genotype(row: &[u8], sample_idx: usize) -> Option<u8> {
    let byte = row[sample_idx >> 2];
    let code = (byte >> ((sample_idx & 3) * 2)) & 0b11;
    match code {
        0b00 => Some(0u8),
        0b10 => Some(1u8),
        0b11 => Some(2u8),
        _ => None,
    }
}

#[inline]
fn fill_bin_logic_row_bits(row: &[u8], sample_indices: &[usize], dst_words: &mut [u64]) {
    let mut c0 = 0usize;
    let mut c2 = 0usize;
    for &src_s in sample_indices.iter() {
        match packed_row_genotype(row, src_s) {
            Some(0) => c0 += 1,
            Some(2) => c2 += 1,
            _ => {}
        }
    }
    let mode02 = if c2 > c0 { 2u8 } else { 0u8 };
    for (dst_s, &src_s) in sample_indices.iter().enumerate() {
        let g = packed_row_genotype(row, src_s).unwrap_or(mode02);
        let is_one = if g == 0 {
            false
        } else if g == 2 {
            true
        } else {
            mode02 == 2
        };
        if is_one {
            dst_words[dst_s >> 6] |= 1u64 << (dst_s & 63);
        }
    }
}

#[inline]
fn fill_mbin_logic_row_bits(
    row: &[u8],
    sample_indices: &[usize],
    row_words: usize,
    dst_words: &mut [u64],
) {
    let mut c0 = 0usize;
    let mut c1 = 0usize;
    let mut c2 = 0usize;
    for &src_s in sample_indices.iter() {
        match packed_row_genotype(row, src_s) {
            Some(0) => c0 += 1,
            Some(1) => c1 += 1,
            Some(2) => c2 += 1,
            _ => {}
        }
    }
    let mode3 = if c2 >= c1 && c2 >= c0 {
        2u8
    } else if c1 >= c0 {
        1u8
    } else {
        0u8
    };
    let (dom_bits, rem) = dst_words.split_at_mut(row_words);
    let (rec_bits, het_bits) = rem.split_at_mut(row_words);
    for (dst_s, &src_s) in sample_indices.iter().enumerate() {
        let g = packed_row_genotype(row, src_s).unwrap_or(mode3);
        let word_idx = dst_s >> 6;
        let bit = 1u64 << (dst_s & 63);
        if g > 0 {
            dom_bits[word_idx] |= bit;
        }
        if g == 2 {
            rec_bits[word_idx] |= bit;
        }
        if g == 1 {
            het_bits[word_idx] |= bit;
        }
    }
}

#[allow(dead_code)]
fn convert_prepared_bed_to_logic_bits(
    packed: &[u8],
    bytes_per_snp: usize,
    n_samples_total: usize,
    sites: &[SiteInfo],
    sample_indices: &[usize],
    sample_ids: &[String],
    mode: GarfieldBinMode,
) -> Result<GarfieldLogicBits, String> {
    if bytes_per_snp == 0 {
        return Err("bytes_per_snp must be > 0".to_string());
    }
    if packed.len() != sites.len().saturating_mul(bytes_per_snp) {
        return Err(format!(
            "packed/site length mismatch: packed={}, sites={}, bytes_per_snp={bytes_per_snp}",
            packed.len(),
            sites.len()
        ));
    }
    if sample_ids.len() != sample_indices.len() {
        return Err(format!(
            "sample id count mismatch: got {}, expected {}",
            sample_ids.len(),
            sample_indices.len()
        ));
    }
    for &idx in sample_indices.iter() {
        if idx >= n_samples_total {
            return Err(format!(
                "sample index out of range while converting packed BED: {idx} >= {n_samples_total}"
            ));
        }
    }

    let n_samples = sample_indices.len();
    let row_words = words_for_samples(n_samples);
    let row_mul = match mode {
        GarfieldBinMode::Bin => 1usize,
        GarfieldBinMode::Mbin => 3usize,
    };
    let n_rows = sites.len().saturating_mul(row_mul);
    let mut bits_flat = vec![0u64; n_rows.saturating_mul(row_words)];
    let mut out_sites = Vec::<SiteInfo>::with_capacity(n_rows);

    for (src_row, site) in sites.iter().enumerate() {
        let row = &packed[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
        match mode {
            GarfieldBinMode::Bin => {
                let mut c0 = 0usize;
                let mut c2 = 0usize;
                for &src_s in sample_indices.iter() {
                    match packed_row_genotype(row, src_s) {
                        Some(0) => c0 += 1,
                        Some(2) => c2 += 1,
                        _ => {}
                    }
                }
                let mode02 = if c2 > c0 { 2u8 } else { 0u8 };
                let dst_row = src_row;
                for (dst_s, &src_s) in sample_indices.iter().enumerate() {
                    let g = packed_row_genotype(row, src_s).unwrap_or(mode02);
                    let is_one = if g == 0 {
                        false
                    } else if g == 2 {
                        true
                    } else {
                        mode02 == 2
                    };
                    if is_one {
                        bits_flat[dst_row * row_words + (dst_s >> 6)] |= 1u64 << (dst_s & 63);
                    }
                }
                let mut site_bin = site.clone();
                site_bin.alt_allele = format!("{}|BIN", site_bin.alt_allele);
                out_sites.push(site_bin);
            }
            GarfieldBinMode::Mbin => {
                let mut c0 = 0usize;
                let mut c1 = 0usize;
                let mut c2 = 0usize;
                for &src_s in sample_indices.iter() {
                    match packed_row_genotype(row, src_s) {
                        Some(0) => c0 += 1,
                        Some(1) => c1 += 1,
                        Some(2) => c2 += 1,
                        _ => {}
                    }
                }
                let mode3 = if c2 >= c1 && c2 >= c0 {
                    2u8
                } else if c1 >= c0 {
                    1u8
                } else {
                    0u8
                };
                let base = src_row * 3;
                let [dom_site, rec_site, het_site] = mbin_site_variants(site);
                for (dst_s, &src_s) in sample_indices.iter().enumerate() {
                    let g = packed_row_genotype(row, src_s).unwrap_or(mode3);
                    let word_idx = dst_s >> 6;
                    let bit = 1u64 << (dst_s & 63);
                    if g > 0 {
                        bits_flat[base * row_words + word_idx] |= bit;
                    }
                    if g == 2 {
                        bits_flat[(base + 1) * row_words + word_idx] |= bit;
                    }
                    if g == 1 {
                        bits_flat[(base + 2) * row_words + word_idx] |= bit;
                    }
                }
                out_sites.push(dom_site);
                out_sites.push(rec_site);
                out_sites.push(het_site);
            }
        }
    }

    let group_ids = build_feature_group_ids(&out_sites, out_sites.len());
    Ok(GarfieldLogicBits {
        bits_flat,
        row_words,
        sample_ids: sample_ids.to_vec(),
        sites: out_sites,
        group_ids,
        n_samples,
    })
}

fn convert_bed_prefix_to_logic_bits(
    prefix: &str,
    bytes_per_snp: usize,
    n_samples_total: usize,
    site_keep: &[bool],
    sites: &[SiteInfo],
    sample_indices: &[usize],
    sample_ids: &[String],
    mode: GarfieldBinMode,
) -> Result<GarfieldLogicBits, String> {
    if bytes_per_snp == 0 {
        return Err("bytes_per_snp must be > 0".to_string());
    }
    if sample_ids.len() != sample_indices.len() {
        return Err(format!(
            "sample id count mismatch: got {}, expected {}",
            sample_ids.len(),
            sample_indices.len()
        ));
    }
    for &idx in sample_indices.iter() {
        if idx >= n_samples_total {
            return Err(format!(
                "sample index out of range while converting BED to logic bits: {idx} >= {n_samples_total}"
            ));
        }
    }

    let kept_n = site_keep.iter().filter(|&&keep| keep).count();
    if kept_n != sites.len() {
        return Err(format!(
            "site_keep/site metadata mismatch: keep count={}, filtered sites={}",
            kept_n,
            sites.len()
        ));
    }

    let bed_prefix = normalize_plink_prefix(prefix);
    let bed_path = format!("{bed_prefix}.bed");
    let bed_file = File::open(&bed_path).map_err(|e| format!("failed to open {bed_path}: {e}"))?;
    let mmap =
        unsafe { Mmap::map(&bed_file) }.map_err(|e| format!("failed to mmap {bed_path}: {e}"))?;
    if mmap.len() < 3 {
        return Err("BED too small".to_string());
    }
    if mmap[0] != 0x6C || mmap[1] != 0x1B || mmap[2] != 0x01 {
        return Err("Only SNP-major BED supported".to_string());
    }
    let data_len = mmap.len() - 3;
    if data_len % bytes_per_snp != 0 {
        return Err(format!(
            "invalid BED payload length: data_len={data_len}, bytes_per_snp={bytes_per_snp}"
        ));
    }
    let n_snps_bed = data_len / bytes_per_snp;
    if n_snps_bed != site_keep.len() {
        return Err(format!(
            "BED/site_keep SNP count mismatch: bed={n_snps_bed}, site_keep={}",
            site_keep.len()
        ));
    }

    let packed_full = &mmap[3..];
    let n_samples = sample_indices.len();
    let row_words = words_for_samples(n_samples);
    let row_mul = match mode {
        GarfieldBinMode::Bin => 1usize,
        GarfieldBinMode::Mbin => 3usize,
    };
    let n_rows = sites.len().saturating_mul(row_mul);
    let mut bits_flat = vec![0u64; n_rows.saturating_mul(row_words)];
    let keep_src_rows = site_keep
        .iter()
        .enumerate()
        .filter_map(|(src_row, &keep)| if keep { Some(src_row) } else { None })
        .collect::<Vec<_>>();
    if keep_src_rows.len() != sites.len() {
        return Err(format!(
            "decoded kept site count mismatch: decoded={}, expected={}",
            keep_src_rows.len(),
            sites.len()
        ));
    }

    // Each kept site owns a disjoint row (or MBIN triplet), so row-wise conversion can
    // safely fan out across Rayon workers without synchronization.
    bits_flat
        .par_chunks_mut(row_mul * row_words)
        .enumerate()
        .for_each(|(kept_idx, dst_rows)| {
            let src_row = keep_src_rows[kept_idx];
            let row = &packed_full[src_row * bytes_per_snp..(src_row + 1) * bytes_per_snp];
            match mode {
                GarfieldBinMode::Bin => fill_bin_logic_row_bits(row, sample_indices, dst_rows),
                GarfieldBinMode::Mbin => {
                    fill_mbin_logic_row_bits(row, sample_indices, row_words, dst_rows)
                }
            }
        });

    let mut out_sites = Vec::<SiteInfo>::with_capacity(n_rows);
    match mode {
        GarfieldBinMode::Bin => {
            for site in sites.iter() {
                let mut site_bin = site.clone();
                site_bin.alt_allele = format!("{}|BIN", site_bin.alt_allele);
                out_sites.push(site_bin);
            }
        }
        GarfieldBinMode::Mbin => {
            for site in sites.iter() {
                let [dom_site, rec_site, het_site] = mbin_site_variants(site);
                out_sites.push(dom_site);
                out_sites.push(rec_site);
                out_sites.push(het_site);
            }
        }
    }

    let group_ids = build_feature_group_ids(&out_sites, out_sites.len());
    Ok(GarfieldLogicBits {
        bits_flat,
        row_words,
        sample_ids: sample_ids.to_vec(),
        sites: out_sites,
        group_ids,
        n_samples,
    })
}

fn dense_binary_rows_from_full_bits(
    bits_flat: &[u64],
    row_words: usize,
    row_indices: &[usize],
    sample_indices: &[usize],
    n_rows_all: usize,
    n_samples_all: usize,
) -> Result<Vec<Vec<u8>>, String> {
    if row_words != words_for_samples(n_samples_all) {
        return Err(format!(
            "row_words mismatch for full bit matrix: got {row_words}, expected {}",
            words_for_samples(n_samples_all)
        ));
    }
    if bits_flat.len() != n_rows_all.saturating_mul(row_words) {
        return Err("full bit matrix length mismatch".to_string());
    }
    let mut out = Vec::<Vec<u8>>::with_capacity(row_indices.len());
    for &row_idx in row_indices.iter() {
        if row_idx >= n_rows_all {
            return Err(format!(
                "row index out of range while densifying: {row_idx}"
            ));
        }
        let row = &bits_flat[row_idx * row_words..(row_idx + 1) * row_words];
        let mut dense = vec![0u8; sample_indices.len()];
        for (j, &sid) in sample_indices.iter().enumerate() {
            if sid >= n_samples_all {
                return Err(format!("sample index out of range while densifying: {sid}"));
            }
            dense[j] = ((row[sid >> 6] >> (sid & 63)) & 1u64) as u8;
        }
        out.push(dense);
    }
    Ok(out)
}

fn packed_rows_subset_from_full_bits(
    bits_flat: &[u64],
    row_words_full: usize,
    row_indices: &[usize],
    sample_indices: &[usize],
    n_rows_all: usize,
    n_samples_all: usize,
) -> Result<(Vec<u64>, usize), String> {
    if row_words_full != words_for_samples(n_samples_all) {
        return Err(format!(
            "row_words mismatch for full bit matrix: got {row_words_full}, expected {}",
            words_for_samples(n_samples_all)
        ));
    }
    if bits_flat.len() != n_rows_all.saturating_mul(row_words_full) {
        return Err("full bit matrix length mismatch".to_string());
    }
    let row_words_sub = words_for_samples(sample_indices.len());
    let mut out = vec![0u64; row_indices.len().saturating_mul(row_words_sub)];
    for (dst_r, &src_r) in row_indices.iter().enumerate() {
        if src_r >= n_rows_all {
            return Err(format!("row index out of range while subsetting: {src_r}"));
        }
        let src = &bits_flat[src_r * row_words_full..(src_r + 1) * row_words_full];
        for (dst_s, &src_s) in sample_indices.iter().enumerate() {
            if src_s >= n_samples_all {
                return Err(format!(
                    "sample index out of range while subsetting: {src_s}"
                ));
            }
            let bit = (src[src_s >> 6] >> (src_s & 63)) & 1u64;
            if bit != 0 {
                out[dst_r * row_words_sub + (dst_s >> 6)] |= 1u64 << (dst_s & 63);
            }
        }
    }
    Ok((out, row_words_sub))
}

#[inline]
fn resolve_ml_keep_k(n_region: usize, ml_top_k: usize, ml_top_frac: f64) -> usize {
    if n_region == 0 {
        return 0;
    }
    let keep_k = if ml_top_k > 0 {
        ml_top_k.min(n_region)
    } else {
        ((ml_top_frac.clamp(0.0, 1.0) * (n_region as f64)).ceil() as usize)
            .max(1)
            .min(n_region)
    };
    keep_k.max(1).min(n_region)
}

fn select_ml_top_local_indices(
    dense_train: &[Vec<u8>],
    y_train: &[f64],
    response: ResponseKind,
    engine: MlEngine,
    tree_cfg: ExtraTreesConfig,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    keep_k: usize,
) -> Result<Vec<usize>, String> {
    if keep_k == 0 || dense_train.is_empty() {
        return Ok(Vec::new());
    }
    let n_region = dense_train.len();
    let keep_k = keep_k.min(n_region).max(1);
    match engine {
        MlEngine::Gbdt2 => {
            let coarse_k = n_region.min(
                keep_k
                    .saturating_mul(4)
                    .max(keep_k.saturating_add(8))
                    .max(32),
            );
            let coarse_scores = compute_feature_scores(
                dense_train,
                y_train,
                response,
                MlEngine::Gbdt,
                tree_cfg,
                ImportanceKind::Imp,
                perm_cfg,
            )?;
            let coarse_local = topk_indices(coarse_scores.as_slice(), coarse_k);
            if coarse_local.is_empty() {
                return Ok(Vec::new());
            }
            if coarse_local.len() <= keep_k {
                return Ok(coarse_local);
            }
            let refined_rows = coarse_local
                .iter()
                .map(|&idx| dense_train[idx].clone())
                .collect::<Vec<_>>();
            let refined_scores = compute_feature_scores(
                refined_rows.as_slice(),
                y_train,
                response,
                MlEngine::Gbdt,
                tree_cfg,
                importance,
                perm_cfg,
            )?;
            let refined_local = topk_indices(refined_scores.as_slice(), keep_k);
            Ok(refined_local
                .into_iter()
                .map(|idx| coarse_local[idx])
                .collect())
        }
        _ => {
            let scores = compute_feature_scores(
                dense_train,
                y_train,
                response,
                engine,
                tree_cfg,
                importance,
                perm_cfg,
            )?;
            Ok(topk_indices(scores.as_slice(), keep_k))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_logic_unit_continuous(
    ui: usize,
    unit: &GarfieldLogicUnit,
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    y_train: &[f64],
    y_test: &[f64],
    assoc_y: &[f64],
    assoc_sample_indices: &[usize],
    beam_params: BeamSearchParams,
    top_rules_per_unit: usize,
    unit_kind_lc: &str,
) -> Result<Vec<GarfieldLogicRuleRecord>, String> {
    if unit.indices.is_empty() {
        return Ok(Vec::new());
    }

    let selected_global_rows = if let Some(engine_one) = engine {
        let dense_train = dense_binary_rows_from_full_bits(
            logic_bits.bits_flat.as_slice(),
            logic_bits.row_words,
            unit.indices.as_slice(),
            train_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        if dense_train.is_empty() {
            return Ok(Vec::new());
        }
        let n_region = dense_train.len();
        let keep_k = resolve_ml_keep_k(n_region, ml_top_k, ml_top_frac);
        let top_local = select_ml_top_local_indices(
            dense_train.as_slice(),
            y_train,
            response,
            engine_one,
            ExtraTreesConfig {
                allow_parallel: beam_params.allow_parallel,
                ..tree_cfg
            },
            importance,
            perm_cfg,
            keep_k,
        )?;
        if top_local.is_empty() {
            return Ok(Vec::new());
        }
        top_local
            .iter()
            .map(|&idx| unit.indices[idx])
            .collect::<Vec<_>>()
    } else {
        unit.indices.clone()
    };
    if selected_global_rows.is_empty() {
        return Ok(Vec::new());
    }

    let local_sites = selected_global_rows
        .iter()
        .map(|&idx| logic_bits.sites[idx].clone())
        .collect::<Vec<_>>();
    let local_groups = selected_global_rows
        .iter()
        .map(|&idx| logic_bits.group_ids[idx])
        .collect::<Vec<_>>();
    let (train_bits, row_words_train) = packed_rows_subset_from_full_bits(
        logic_bits.bits_flat.as_slice(),
        logic_bits.row_words,
        selected_global_rows.as_slice(),
        train_idx_local,
        logic_bits.sites.len(),
        logic_bits.n_samples,
    )?;
    let (test_bits, row_words_test) = packed_rows_subset_from_full_bits(
        logic_bits.bits_flat.as_slice(),
        logic_bits.row_words,
        selected_global_rows.as_slice(),
        test_idx_local,
        logic_bits.sites.len(),
        logic_bits.n_samples,
    )?;
    let beam_hits = beam_search_train_test_continuous(
        y_train,
        train_bits.as_slice(),
        row_words_train,
        selected_global_rows.len(),
        train_idx_local.len(),
        y_test,
        test_bits.as_slice(),
        row_words_test,
        test_idx_local.len(),
        local_groups.as_slice(),
        beam_params,
    )?;
    if beam_hits.is_empty() {
        return Ok(Vec::new());
    }

    let selected_bits_full = gather_rows_by_indices(
        logic_bits.bits_flat.as_slice(),
        logic_bits.row_words,
        selected_global_rows.as_slice(),
    )?;
    let keep_rules = if top_rules_per_unit == 0 {
        beam_hits.len()
    } else {
        extend_keep_with_score_ties(
            beam_hits.as_slice(),
            top_rules_per_unit.min(beam_hits.len()),
            |cand| cand.test_score,
        )
    };
    let mut out = Vec::<GarfieldLogicRuleRecord>::with_capacity(keep_rules.max(1));
    for (cand_idx, cand) in beam_hits.iter().enumerate() {
        let keep = if top_rules_per_unit == 0 {
            true
        } else {
            cand_idx < keep_rules
        };
        if !keep {
            continue;
        }
        let expr = rule_expr(&cand.rule, local_sites.as_slice())?;
        let snp_name = rule_snp_name(&cand.rule, local_sites.as_slice())?;
        let first_site = local_sites
            .get(cand.rule.first.row_index)
            .ok_or_else(|| "beam result first row index out of range".to_string())?;
        let full_bits = materialize_rule_bits(
            &cand.rule,
            selected_bits_full.as_slice(),
            logic_bits.row_words,
            selected_global_rows.len(),
            logic_bits.n_samples,
        )?;
        let assoc =
            fit_binary_rule_wald_from_bits(assoc_y, full_bits.as_slice(), assoc_sample_indices);
        let selected_row_indices =
            rule_selected_global_rows(&cand.rule, selected_global_rows.as_slice())?;
        out.push(GarfieldLogicRuleRecord {
            unit_name: unit.label.clone(),
            unit_kind: unit_kind_lc.to_string(),
            unit_index: ui + 1,
            region_size: unit.indices.len(),
            ml_feature_count: selected_global_rows.len(),
            selected_row_indices,
            snp_name,
            expr,
            chrom_field: unit_kind_lc.to_string(),
            pos: first_site.pos,
            beta: assoc.beta,
            se: assoc.se,
            chisq: assoc.chisq,
            pwald: assoc.pwald,
            train_score: cand.train_score,
            test_score: cand.test_score,
            full_bits,
        });
    }
    Ok(out)
}

#[inline]
fn parse_optional_ml_engine(method: &str) -> Result<Option<MlEngine>, String> {
    let norm = method.trim().to_ascii_lowercase();
    match norm.as_str() {
        "" | "none" | "skip" | "direct" => Ok(None),
        _ => parse_ml_engine(norm.as_str()).map(Some),
    }
}

#[inline]
fn normalized_rank_score(score: f64) -> f64 {
    if score.is_nan() {
        f64::NEG_INFINITY
    } else {
        score
    }
}

#[inline]
fn scores_tied_for_keep(a: f64, b: f64) -> bool {
    let aa = normalized_rank_score(a);
    let bb = normalized_rank_score(b);
    if aa == bb {
        return true;
    }
    if !aa.is_finite() || !bb.is_finite() {
        return false;
    }
    let scale = aa.abs().max(bb.abs()).max(1.0);
    (aa - bb).abs() <= 1e-12 * scale
}

fn extend_keep_with_score_ties<T, F>(items: &[T], keep_n: usize, score_of: F) -> usize
where
    F: Fn(&T) -> f64,
{
    if keep_n == 0 || items.is_empty() {
        return 0;
    }
    let mut keep = keep_n.min(items.len());
    if keep >= items.len() {
        return items.len();
    }
    let cutoff = score_of(&items[keep - 1]);
    while keep < items.len() && scores_tied_for_keep(score_of(&items[keep]), cutoff) {
        keep += 1;
    }
    keep
}

fn apply_logic_rule_output_limit(
    records: &mut Vec<GarfieldLogicRuleRecord>,
    max_output_rules: usize,
    max_output_ratio: f64,
) -> Result<(), String> {
    if max_output_rules > 0 {
        if records.len() > max_output_rules {
            let keep_n = extend_keep_with_score_ties(records.as_slice(), max_output_rules, |rec| {
                rec.test_score
            });
            records.truncate(keep_n);
        }
        return Ok(());
    }
    if max_output_ratio == 0.0 {
        return Ok(());
    }
    if !max_output_ratio.is_finite() || max_output_ratio <= 0.0 || max_output_ratio > 1.0 {
        return Err(format!(
            "max_output_ratio must be within (0, 1], got {}",
            max_output_ratio
        ));
    }
    let keep_n = ((records.len() as f64) * max_output_ratio).ceil().max(1.0) as usize;
    if records.len() > keep_n {
        let keep_n = extend_keep_with_score_ties(records.as_slice(), keep_n, |rec| rec.test_score);
        records.truncate(keep_n);
    }
    Ok(())
}

#[inline]
fn plink2bits_from_logic_g(g: i8) -> u8 {
    match g {
        0 => 0b00,
        2 => 0b11,
        _ => 0b01,
    }
}

fn write_logic_pseudo_plink(
    prefix: &str,
    sample_ids: &[String],
    records: &[GarfieldLogicRuleRecord],
) -> Result<(), String> {
    let bed_path = format!("{prefix}.bed");
    let bim_path = format!("{prefix}.bim");
    let fam_path = format!("{prefix}.fam");
    let mut fam = BufWriter::new(File::create(&fam_path).map_err(|e| e.to_string())?);
    for sid in sample_ids.iter() {
        writeln!(fam, "{0}\t{0}\t0\t0\t1\t-9", sid).map_err(|e| e.to_string())?;
    }
    fam.flush().map_err(|e| e.to_string())?;

    let mut bed = BufWriter::with_capacity(
        8 * 1024 * 1024,
        File::create(&bed_path).map_err(|e| e.to_string())?,
    );
    bed.write_all(&[0x6C, 0x1B, 0x01])
        .map_err(|e| e.to_string())?;
    let mut bim = BufWriter::with_capacity(
        4 * 1024 * 1024,
        File::create(&bim_path).map_err(|e| e.to_string())?,
    );

    let bytes_per_snp = sample_ids.len().div_ceil(4);
    let mut row_buf = vec![0u8; bytes_per_snp];
    let mut ordered = records.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| cmp_logic_rule_records_plink_order(a, b));
    for rec in ordered.into_iter() {
        writeln!(
            bim,
            "{}\t{}\t0\t{}\tA\tT",
            rec.chrom_field, rec.snp_name, rec.pos
        )
        .map_err(|e| e.to_string())?;
        row_buf.fill(0u8);
        for s in 0..sample_ids.len() {
            let hit = ((rec.full_bits[s >> 6] >> (s & 63)) & 1u64) != 0;
            let g = if hit { 2i8 } else { 0i8 };
            row_buf[s >> 2] |= plink2bits_from_logic_g(g) << ((s & 3) * 2);
        }
        bed.write_all(&row_buf).map_err(|e| e.to_string())?;
    }
    bim.flush().map_err(|e| e.to_string())?;
    bed.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn write_logic_rules_tsv(path: &str, records: &[GarfieldLogicRuleRecord]) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    writeln!(
        w,
        "unit_kind\tunit_index\tunit_name\tregion_size\tml_feature_count\tsnp_name\texpr\tbeta\tse\tchisq\tpwald\ttrain_score"
    )
    .map_err(|e| e.to_string())?;
    for rec in records.iter() {
        let chisq_txt = format_chisq_value(rec.chisq);
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.10}\t{:.10}\t{}\t{:.4e}\t{:.6}",
            rec.unit_kind,
            rec.unit_index,
            rec.unit_name,
            rec.region_size,
            rec.ml_feature_count,
            rec.snp_name,
            rec.expr,
            rec.beta,
            rec.se,
            chisq_txt,
            rec.pwald,
            rec.train_score,
        )
        .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}

fn slice_square_matrix_by_indices(
    matrix: &[f64],
    n: usize,
    indices: &[usize],
    ctx: &str,
) -> Result<Vec<f64>, String> {
    if matrix.len() != n.saturating_mul(n) {
        return Err(format!(
            "{ctx}: square-matrix payload length mismatch: got {}, expected {}",
            matrix.len(),
            n.saturating_mul(n)
        ));
    }
    let k = indices.len();
    let mut out = vec![0.0_f64; k.saturating_mul(k)];
    for (i_out, &i_src) in indices.iter().enumerate() {
        if i_src >= n {
            return Err(format!(
                "{ctx}: row index {} out of range for n={}",
                i_src, n
            ));
        }
        for (j_out, &j_src) in indices.iter().enumerate() {
            if j_src >= n {
                return Err(format!(
                    "{ctx}: col index {} out of range for n={}",
                    j_src, n
                ));
            }
            out[i_out * k + j_out] = matrix[i_src * n + j_src];
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn garfield_logic_search_bed_owned(
    prefix: String,
    y: Vec<f64>,
    grm: Option<Vec<f64>>,
    x_cov: Option<Vec<f64>>,
    q_cov: usize,
    sample_ids: Option<Vec<String>>,
    sample_indices: Option<Vec<usize>>,
    unit_kind: String,
    groups: Option<Vec<Vec<(String, i32, i32)>>>,
    group_names: Option<Vec<String>>,
    extension: usize,
    step: Option<usize>,
    bin_mode: String,
    ml_method: String,
    ml_importance: String,
    ml_top_k: usize,
    ml_top_frac: f64,
    permutation_repeats: usize,
    permutation_scoring: String,
    tree_cfg: ExtraTreesConfig,
    fold: usize,
    seed: u64,
    max_pick: usize,
    exhaustive_depth: usize,
    beam_width: usize,
    logic_gate: String,
    rank_score: String,
    candidate_keep_ratio: f64,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    block_cols: usize,
    threads: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
    out_prefix: Option<String>,
    simbench_path: Option<String>,
    top_rules_per_unit: usize,
    max_output_rules: usize,
    max_output_ratio: f64,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> Result<GarfieldLogicPipelineResult, String> {
    validate_continuous_y(&y)?;
    let using_external_grm = grm.is_some();
    let mut packed_for_grm: Option<Vec<u8>> = None;
    let mut maf_for_grm: Option<Vec<f32>> = None;
    let mut row_flip_for_grm: Option<Vec<bool>> = None;
    let (
        selected_sample_indices,
        selected_sample_ids,
        site_keep,
        filtered_sites,
        n_samples_total,
        bytes_per_snp,
    ) = if using_external_grm {
        let meta = prepare_bed_logic_meta_owned(
            &prefix,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
        )?;
        let (selected_sample_indices, selected_sample_ids) =
            build_sample_selection(meta.sample_ids.as_slice(), sample_ids, sample_indices)?;
        (
            selected_sample_indices,
            selected_sample_ids,
            meta.site_keep,
            meta.sites,
            meta.n_samples,
            meta.bytes_per_snp,
        )
    } else {
        let prepared = prepare_bed_2bit_packed_owned(
            &prefix,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
        )?;
        let (selected_sample_indices, selected_sample_ids) =
            build_sample_selection(prepared.sample_ids.as_slice(), sample_ids, sample_indices)?;
        packed_for_grm = Some(prepared.packed);
        maf_for_grm = Some(prepared.maf);
        row_flip_for_grm = Some(prepared.row_flip);
        (
            selected_sample_indices,
            selected_sample_ids,
            prepared.site_keep,
            prepared.sites,
            prepared.n_samples,
            prepared.bytes_per_snp,
        )
    };
    let n_selected = selected_sample_indices.len();
    if y.len() != n_selected {
        return Err(format!(
            "y length mismatch: got {}, expected {} selected samples from BED input",
            y.len(),
            n_selected
        ));
    }
    if q_cov > 0 {
        let cov = x_cov
            .as_ref()
            .ok_or_else(|| "q_cov > 0 but x_cov is missing".to_string())?;
        if cov.len() != n_selected.saturating_mul(q_cov) {
            return Err(format!(
                "x_cov payload length mismatch: got {}, expected {}",
                cov.len(),
                n_selected.saturating_mul(q_cov)
            ));
        }
    }

    let split_applied = fold >= 2;
    let (train_idx_local, test_idx_local) = if split_applied {
        let test_mask = stratified_test_mask(&y, fold, seed)?;
        let train_idx_local = mask_to_indices(&test_mask, false);
        let test_idx_local = mask_to_indices(&test_mask, true);
        if train_idx_local.is_empty() || test_idx_local.is_empty() {
            return Err("train/test split produced an empty partition".to_string());
        }
        (train_idx_local, test_idx_local)
    } else {
        let full = (0..n_selected).collect::<Vec<_>>();
        (full.clone(), full)
    };
    let train_idx = train_idx_local
        .iter()
        .map(|&i| selected_sample_indices[i])
        .collect::<Vec<_>>();
    let test_idx = test_idx_local
        .iter()
        .map(|&i| selected_sample_indices[i])
        .collect::<Vec<_>>();

    let y_train = subset_vec_f64(&y, &train_idx_local)?;
    let y_test = subset_vec_f64(&y, &test_idx_local)?;
    let x_train = subset_cov_f64(x_cov.as_deref(), n_selected, q_cov, &train_idx_local)?;
    let x_test = subset_cov_f64(x_cov.as_deref(), n_selected, q_cov, &test_idx_local)?;

    let threads_eff = effective_threads_local(threads);
    let (train_fit, test_fit) = if split_applied {
        let (grm_train, eff_m_train) = if let Some(full_grm) = grm.as_ref() {
            (
                slice_square_matrix_by_indices(
                    full_grm.as_slice(),
                    n_selected,
                    train_idx_local.as_slice(),
                    "garfield_logic_search_bed: train GRM slice",
                )?,
                None,
            )
        } else {
            let packed = packed_for_grm
                .as_ref()
                .ok_or_else(|| "internal error: packed BED missing for train GRM".to_string())?;
            let row_flip = row_flip_for_grm
                .as_ref()
                .ok_or_else(|| "internal error: row_flip missing for train GRM".to_string())?;
            let maf = maf_for_grm
                .as_ref()
                .ok_or_else(|| "internal error: maf missing for train GRM".to_string())?;
            let (grm_train, eff_m_train) = grm_packed_f64_from_stats_rust(
                packed.as_slice(),
                bytes_per_snp,
                n_samples_total,
                row_flip.as_slice(),
                maf.as_slice(),
                train_idx.as_slice(),
                1,
                block_cols,
                threads_eff,
            )?;
            (grm_train, Some(eff_m_train.round().max(0.0) as usize))
        };
        let train_fit = garfield_residualize_exact_from_grm_rust(
            grm_train,
            train_idx.len(),
            y_train,
            x_train,
            q_cov,
            threads_eff,
            low,
            high,
            max_iter,
            tol,
            add_intercept,
            exact_n_max,
            require_lapack,
            eff_m_train,
        )?;

        let (grm_test, eff_m_test) = if let Some(full_grm) = grm.as_ref() {
            (
                slice_square_matrix_by_indices(
                    full_grm.as_slice(),
                    n_selected,
                    test_idx_local.as_slice(),
                    "garfield_logic_search_bed: test GRM slice",
                )?,
                None,
            )
        } else {
            let packed = packed_for_grm
                .as_ref()
                .ok_or_else(|| "internal error: packed BED missing for test GRM".to_string())?;
            let row_flip = row_flip_for_grm
                .as_ref()
                .ok_or_else(|| "internal error: row_flip missing for test GRM".to_string())?;
            let maf = maf_for_grm
                .as_ref()
                .ok_or_else(|| "internal error: maf missing for test GRM".to_string())?;
            let (grm_test, eff_m_test) = grm_packed_f64_from_stats_rust(
                packed.as_slice(),
                bytes_per_snp,
                n_samples_total,
                row_flip.as_slice(),
                maf.as_slice(),
                test_idx.as_slice(),
                1,
                block_cols,
                threads_eff,
            )?;
            (grm_test, Some(eff_m_test.round().max(0.0) as usize))
        };
        let test_fit = garfield_residualize_exact_from_grm_rust(
            grm_test,
            test_idx.len(),
            y_test,
            x_test,
            q_cov,
            threads_eff,
            low,
            high,
            max_iter,
            tol,
            add_intercept,
            exact_n_max,
            require_lapack,
            eff_m_test,
        )?;
        (train_fit, test_fit)
    } else {
        let (grm_full, eff_m_full) = if let Some(full_grm) = grm.as_ref() {
            if full_grm.len() != n_selected.saturating_mul(n_selected) {
                return Err(format!(
                    "garfield_logic_search_bed: provided GRM length mismatch: got {}, expected {}",
                    full_grm.len(),
                    n_selected.saturating_mul(n_selected)
                ));
            }
            (full_grm.clone(), None)
        } else {
            let packed = packed_for_grm
                .as_ref()
                .ok_or_else(|| "internal error: packed BED missing for full GRM".to_string())?;
            let row_flip = row_flip_for_grm
                .as_ref()
                .ok_or_else(|| "internal error: row_flip missing for full GRM".to_string())?;
            let maf = maf_for_grm
                .as_ref()
                .ok_or_else(|| "internal error: maf missing for full GRM".to_string())?;
            let (grm_full, eff_m_full) = grm_packed_f64_from_stats_rust(
                packed.as_slice(),
                bytes_per_snp,
                n_samples_total,
                row_flip.as_slice(),
                maf.as_slice(),
                train_idx.as_slice(),
                1,
                block_cols,
                threads_eff,
            )?;
            (grm_full, Some(eff_m_full.round().max(0.0) as usize))
        };
        let fit = garfield_residualize_exact_from_grm_rust(
            grm_full,
            train_idx.len(),
            y_train,
            x_train,
            q_cov,
            threads_eff,
            low,
            high,
            max_iter,
            tol,
            add_intercept,
            exact_n_max,
            require_lapack,
            eff_m_full,
        )?;
        (fit.clone(), fit)
    };

    let mode = parse_bin_mode(&bin_mode)?;
    drop(packed_for_grm.take());
    drop(maf_for_grm.take());
    drop(row_flip_for_grm.take());
    let logic_bits = convert_bed_prefix_to_logic_bits(
        &prefix,
        bytes_per_snp,
        n_samples_total,
        site_keep.as_slice(),
        filtered_sites.as_slice(),
        selected_sample_indices.as_slice(),
        selected_sample_ids.as_slice(),
        mode,
    )?;
    let units = build_logic_units(
        logic_bits.sites.as_slice(),
        &unit_kind,
        groups.as_deref(),
        group_names.as_deref(),
        extension,
        step,
    )?;
    if units.is_empty() {
        return Err("no scan units were built from the provided input".to_string());
    }

    let response = ResponseKind::Continuous;
    let engine = parse_optional_ml_engine(&ml_method)?;
    let importance = parse_importance(&ml_importance)?;
    let logic_gate_mode = parse_beam_logic_gate_mode(&logic_gate)?;
    let rank_mode = parse_beam_rank_mode(&rank_score)?;
    let perm_cfg = PermutationConfig {
        n_repeats: permutation_repeats.max(1),
        scoring: parse_permutation_scoring(&permutation_scoring)?,
        seed,
    };
    let beam_params = BeamSearchParams {
        max_pick: max_pick.max(1),
        beam_width: beam_width.max(1),
        candidate_keep_ratio: candidate_keep_ratio.clamp(1e-6, 1.0),
        lambda_len: 0.0,
        lambda_not: 0.0,
        exhaustive_depth: exhaustive_depth.max(1),
        logic_gate_mode,
        rank_mode,
        allow_parallel: true,
    };
    let unit_kind_lc = unit_kind.trim().to_ascii_lowercase();

    let total_units = units.len();
    let notify_step = if progress_every == 0 {
        (total_units / 200).max(1)
    } else {
        progress_every.max(1)
    };
    garfield_progress_notify(progress_callback.as_ref(), 0, total_units)
        .map_err(|e| e.to_string())?;
    let (assoc_y, assoc_sample_indices): (&[f64], &[usize]) = if split_applied {
        (
            test_fit.residualized_y.as_slice(),
            test_idx_local.as_slice(),
        )
    } else {
        (
            train_fit.residualized_y.as_slice(),
            train_idx_local.as_slice(),
        )
    };
    let unit_parallel_threads = threads_eff.min(total_units.max(1));
    let scan_beam_params = BeamSearchParams {
        allow_parallel: unit_parallel_threads <= 1,
        ..beam_params
    };
    let progress_done = AtomicUsize::new(0);
    let progress_callback_parallel = progress_callback.as_ref();
    let unit_results = if unit_parallel_threads > 1 {
        let pool = ThreadPoolBuilder::new()
            .num_threads(unit_parallel_threads)
            .build()
            .map_err(|e| format!("build GARFIELD unit thread pool: {e}"))?;
        pool.install(|| {
            units
                .par_iter()
                .enumerate()
                .map(|(ui, unit)| {
                    let out = evaluate_logic_unit_continuous(
                        ui,
                        unit,
                        response,
                        engine,
                        importance,
                        perm_cfg,
                        ml_top_k,
                        ml_top_frac,
                        tree_cfg,
                        &logic_bits,
                        train_idx_local.as_slice(),
                        test_idx_local.as_slice(),
                        train_fit.residualized_y.as_slice(),
                        test_fit.residualized_y.as_slice(),
                        assoc_y,
                        assoc_sample_indices,
                        scan_beam_params,
                        top_rules_per_unit,
                        &unit_kind_lc,
                    )?;
                    let done = progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % notify_step == 0 || done == total_units {
                        garfield_progress_notify(progress_callback_parallel, done, total_units)
                            .map_err(|e| e.to_string())?;
                    }
                    Ok(out)
                })
                .collect::<Vec<Result<Vec<GarfieldLogicRuleRecord>, String>>>()
        })
    } else {
        let mut out =
            Vec::<Result<Vec<GarfieldLogicRuleRecord>, String>>::with_capacity(total_units);
        for (ui, unit) in units.iter().enumerate() {
            let unit_out = evaluate_logic_unit_continuous(
                ui,
                unit,
                response,
                engine,
                importance,
                perm_cfg,
                ml_top_k,
                ml_top_frac,
                tree_cfg,
                &logic_bits,
                train_idx_local.as_slice(),
                test_idx_local.as_slice(),
                train_fit.residualized_y.as_slice(),
                test_fit.residualized_y.as_slice(),
                assoc_y,
                assoc_sample_indices,
                scan_beam_params,
                top_rules_per_unit,
                &unit_kind_lc,
            );
            let done = progress_done.fetch_add(1, Ordering::Relaxed) + 1;
            if done % notify_step == 0 || done == total_units {
                garfield_progress_notify(progress_callback.as_ref(), done, total_units)
                    .map_err(|e| e.to_string())?;
            }
            out.push(unit_out);
        }
        out
    };
    let mut records = Vec::<GarfieldLogicRuleRecord>::new();
    for unit_out in unit_results.into_iter() {
        records.extend(unit_out?);
    }

    records = dedup_logic_rule_records(records);
    apply_logic_rule_output_limit(&mut records, max_output_rules, max_output_ratio)?;
    let simbench_count = if let Some(path) = simbench_path.as_ref() {
        let simbench_records = evaluate_simbench_terms(
            &prefix,
            path,
            selected_sample_indices.as_slice(),
            train_idx_local.as_slice(),
            assoc_sample_indices,
            train_fit.residualized_y.as_slice(),
            assoc_y,
            beam_params,
        )?;
        let n = simbench_records.len();
        records.extend(simbench_records);
        n
    } else {
        0usize
    };

    let mut pseudo_prefix_out = None;
    let mut rules_tsv_out = None;
    if let Some(prefix_out) = out_prefix.as_ref() {
        write_logic_pseudo_plink(
            prefix_out,
            logic_bits.sample_ids.as_slice(),
            records.as_slice(),
        )?;
        let rules_tsv = format!("{prefix_out}.rules.tsv");
        write_logic_rules_tsv(&rules_tsv, records.as_slice())?;
        pseudo_prefix_out = Some(prefix_out.clone());
        rules_tsv_out = Some(rules_tsv);
    }

    Ok(GarfieldLogicPipelineResult {
        pseudo_prefix: pseudo_prefix_out,
        rules_tsv: rules_tsv_out,
        records,
        simbench_count,
        split_applied,
        n_train: train_idx_local.len(),
        n_test: test_idx_local.len(),
        n_samples: logic_bits.n_samples,
        units_total: total_units,
        units_scanned: total_units,
        train_fit,
        test_fit,
    })
}

#[pyfunction(name = "garfield_logic_search_bed")]
#[pyo3(signature = (
    prefix,
    y,
    grm=None,
    x_cov=None,
    sample_ids=None,
    sample_indices=None,
    unit_kind="window",
    groups=None,
    group_names=None,
    extension=100000,
    step=None,
    bin_mode="bin",
    ml_method="rf",
    ml_importance="imp",
    ml_top_k=64,
    ml_top_frac=0.0,
    permutation_repeats=5,
    permutation_scoring="auto",
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=1,
    min_samples_split=2,
    bootstrap=true,
    feature_subsample=0.0,
    fold=0,
    seed=42,
    max_pick=3,
    exhaustive_depth=1,
    beam_width=5,
    logic_gate="ao",
    rank_score="interaction_gain",
    candidate_keep_ratio=0.10,
    maf_threshold=0.02,
    max_missing_rate=0.05,
    het_threshold=0.0,
    snps_only=false,
    block_cols=65536,
    threads=0,
    low=-5.0,
    high=5.0,
    max_iter=50,
    tol=1e-3,
    add_intercept=true,
    exact_n_max=15000,
    require_lapack=false,
    out_prefix=None,
    simbench_path=None,
    top_rules_per_unit=1,
    max_output_rules=0,
    max_output_ratio=0.0,
    progress_callback=None,
    progress_every=0
))]
pub fn garfield_logic_search_bed_py<'py>(
    py: Python<'py>,
    prefix: String,
    y: PyReadonlyArray1<'py, f64>,
    grm: Option<PyReadonlyArray2<'py, f64>>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    sample_ids: Option<Vec<String>>,
    sample_indices: Option<Vec<usize>>,
    unit_kind: &str,
    groups: Option<Vec<Vec<(String, i32, i32)>>>,
    group_names: Option<Vec<String>>,
    extension: usize,
    step: Option<usize>,
    bin_mode: &str,
    ml_method: &str,
    ml_importance: &str,
    ml_top_k: usize,
    ml_top_frac: f64,
    permutation_repeats: usize,
    permutation_scoring: &str,
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    bootstrap: bool,
    feature_subsample: f64,
    fold: usize,
    seed: u64,
    max_pick: usize,
    exhaustive_depth: usize,
    beam_width: usize,
    logic_gate: &str,
    rank_score: &str,
    candidate_keep_ratio: f64,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    block_cols: usize,
    threads: usize,
    low: f64,
    high: f64,
    max_iter: usize,
    tol: f64,
    add_intercept: bool,
    exact_n_max: usize,
    require_lapack: bool,
    out_prefix: Option<String>,
    simbench_path: Option<String>,
    top_rules_per_unit: usize,
    max_output_rules: usize,
    max_output_ratio: f64,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let y_vec = read_y_f64(&y);
    let grm_vec = if let Some(arr) = grm {
        let mat = arr.as_array();
        if mat.ndim() != 2 || mat.shape()[0] != mat.shape()[1] {
            return Err(PyValueError::new_err(
                "grm must be a square float64 matrix.",
            ));
        }
        Some(match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => mat.iter().copied().collect(),
        })
    } else {
        None
    };
    let (x_cov_vec, q_cov) = if let Some(arr) = x_cov {
        let mat = arr.as_array();
        if mat.ndim() != 2 {
            return Err(PyValueError::new_err(
                "x_cov must be 2D (n_samples, q_cov).",
            ));
        }
        let q_cov = mat.shape()[1];
        let cov_vec = match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => mat.iter().copied().collect(),
        };
        (Some(cov_vec), q_cov)
    } else {
        (None, 0usize)
    };
    let tree_cfg = ExtraTreesConfig {
        n_estimators: n_estimators.max(1),
        max_depth: max_depth.max(1),
        min_samples_leaf: min_samples_leaf.max(1),
        min_samples_split: min_samples_split.max(2),
        bootstrap,
        feature_subsample,
        seed,
        allow_parallel: true,
    };
    let result = py
        .detach(move || {
            garfield_logic_search_bed_owned(
                prefix,
                y_vec,
                grm_vec,
                x_cov_vec,
                q_cov,
                sample_ids,
                sample_indices,
                unit_kind.to_string(),
                groups,
                group_names,
                extension,
                step,
                bin_mode.to_string(),
                ml_method.to_string(),
                ml_importance.to_string(),
                ml_top_k,
                ml_top_frac,
                permutation_repeats,
                permutation_scoring.to_string(),
                tree_cfg,
                fold,
                seed,
                max_pick,
                exhaustive_depth,
                beam_width,
                logic_gate.to_string(),
                rank_score.to_string(),
                candidate_keep_ratio,
                maf_threshold,
                max_missing_rate,
                het_threshold,
                snps_only,
                block_cols,
                threads,
                low,
                high,
                max_iter,
                tol,
                add_intercept,
                exact_n_max,
                require_lapack,
                out_prefix,
                simbench_path,
                top_rules_per_unit,
                max_output_rules,
                max_output_ratio,
                progress_callback,
                progress_every,
            )
        })
        .map_err(PyRuntimeError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("pseudo_prefix", result.pseudo_prefix)?;
    out.set_item("rules_tsv", result.rules_tsv)?;
    out.set_item("n_rules", result.records.len())?;
    out.set_item("n_simbench", result.simbench_count)?;
    out.set_item("split_applied", result.split_applied)?;
    out.set_item("n_train", result.n_train)?;
    out.set_item("n_test", result.n_test)?;
    out.set_item("n_samples", result.n_samples)?;
    out.set_item("units_total", result.units_total)?;
    out.set_item("units_scanned", result.units_scanned)?;
    out.set_item("train_pve", result.train_fit.pve)?;
    out.set_item("test_pve", result.test_fit.pve)?;
    out.set_item("train_sigma_g2", result.train_fit.sigma_g2)?;
    out.set_item("train_sigma_e2", result.train_fit.sigma_e2)?;
    out.set_item("test_sigma_g2", result.test_fit.sigma_g2)?;
    out.set_item("test_sigma_e2", result.test_fit.sigma_e2)?;
    out.set_item("train_eigh_backend", result.train_fit.eigh_backend)?;
    out.set_item("test_eigh_backend", result.test_fit.eigh_backend)?;
    out.set_item(
        "snp_names",
        result
            .records
            .iter()
            .map(|r| r.snp_name.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "expressions",
        result
            .records
            .iter()
            .map(|r| r.expr.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "unit_names",
        result
            .records
            .iter()
            .map(|r| r.unit_name.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "unit_kinds",
        result
            .records
            .iter()
            .map(|r| r.unit_kind.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "train_scores",
        result
            .records
            .iter()
            .map(|r| r.train_score)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "test_scores",
        result
            .records
            .iter()
            .map(|r| r.test_score)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "positions",
        result.records.iter().map(|r| r.pos).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "selected_row_indices",
        result
            .records
            .iter()
            .map(|r| r.selected_row_indices.clone())
            .collect::<Vec<_>>(),
    )?;
    Ok(out)
}

#[pyfunction(name = "load_bin01_packed")]
pub fn load_bin01_packed_py<'py>(
    py: Python<'py>,
    path: String,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<u64>>, usize)> {
    let (packed, group_ids, n_samples, n_rows, row_bytes) = py
        .detach(move || load_bin01_packed_owned(&path, false))
        .map_err(PyRuntimeError::new_err)?;
    let packed_mat = Array2::from_shape_vec((n_rows, row_bytes), packed)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let gid_arr = Array1::from_vec(group_ids);
    #[allow(deprecated)]
    let packed_arr = PyArray2::from_owned_array(py, packed_mat).into_bound();
    #[allow(deprecated)]
    let gid_arr = PyArray1::from_owned_array(py, gid_arr).into_bound();
    Ok((packed_arr, gid_arr, n_samples))
}

#[pyfunction(name = "load_mbin_packed")]
pub fn load_mbin_packed_py<'py>(
    py: Python<'py>,
    path: String,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<u64>>, usize)> {
    let (packed, group_ids, n_samples, n_rows, row_bytes) = py
        .detach(move || load_bin01_packed_owned(&path, true))
        .map_err(PyRuntimeError::new_err)?;
    let packed_mat = Array2::from_shape_vec((n_rows, row_bytes), packed)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let gid_arr = Array1::from_vec(group_ids);
    #[allow(deprecated)]
    let packed_arr = PyArray2::from_owned_array(py, packed_mat).into_bound();
    #[allow(deprecated)]
    let gid_arr = PyArray1::from_owned_array(py, gid_arr).into_bound();
    Ok((packed_arr, gid_arr, n_samples))
}

#[pyfunction(name = "garfield_prepare_input_bin")]
#[pyo3(signature = (
    input_path,
    out_bin_path,
    input_kind="auto",
    mode="bin",
    threads=0,
    maf=0.0,
    geno=1.0,
    impute=false,
    het=0.05,
    sample_ids=None,
    sample_indices=None
))]
pub fn garfield_prepare_input_bin_py(
    input_path: String,
    out_bin_path: String,
    input_kind: &str,
    mode: &str,
    threads: usize,
    maf: f64,
    geno: f64,
    impute: bool,
    het: f64,
    sample_ids: Option<Vec<String>>,
    sample_indices: Option<Vec<usize>>,
) -> PyResult<(usize, usize, usize, usize)> {
    if !(0.0..=0.5).contains(&maf) {
        return Err(PyValueError::new_err("maf must be within [0, 0.5]"));
    }
    if !(0.0..=1.0).contains(&geno) {
        return Err(PyValueError::new_err("geno must be within [0, 1]"));
    }
    if !(0.0..=1.0).contains(&het) {
        return Err(PyValueError::new_err("het must be within [0, 1]"));
    }

    let input_kind = parse_input_kind(input_kind).map_err(PyValueError::new_err)?;
    let mode = parse_bin_mode(mode).map_err(PyValueError::new_err)?;
    let mut reader = make_input_reader(&input_path, input_kind).map_err(PyRuntimeError::new_err)?;

    let (selected_indices, selected_ids) =
        build_sample_selection(reader.sample_ids(), sample_ids, sample_indices)
            .map_err(PyRuntimeError::new_err)?;
    if selected_indices.is_empty() {
        return Err(PyValueError::new_err("selected sample set is empty"));
    }
    write_sample_id_sidecar(&out_bin_path, &selected_ids).map_err(PyRuntimeError::new_err)?;

    let n_samples = selected_indices.len();
    let identity_selection = selected_indices
        .iter()
        .enumerate()
        .all(|(i, &idx)| i == idx);
    let row_bytes = row_bytes_for_samples(n_samples);
    let mut writer =
        GarfieldBinWriter::new(&out_bin_path, n_samples).map_err(PyRuntimeError::new_err)?;

    let available_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1usize);
    let n_threads = if threads == 0 {
        available_threads
    } else {
        threads.min(available_threads).max(1)
    };
    let pool = if n_threads > 1 {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("build thread pool: {e}")))?,
        )
    } else {
        None
    };
    let pool_ref = pool.as_ref();

    let maf_f32 = maf as f32;
    let geno_f32 = geno as f32;
    let het_f32 = het as f32;

    let mut n_sites_seen = 0usize;
    let mut n_sites_written = 0usize;
    let mut n_rows_written = 0usize;

    match &mut reader {
        GarfieldInputReader::Bed(bed) => {
            if n_threads > 1 {
                let n_sites = bed.sites.len();
                let target_bytes = 16usize * 1024 * 1024;
                let mut chunk_size = target_bytes / n_samples.max(1);
                if chunk_size == 0 {
                    chunk_size = 1;
                }
                if chunk_size > 4096 {
                    chunk_size = 4096;
                }
                for start in (0..n_sites).step_by(chunk_size) {
                    let end = (start + chunk_size).min(n_sites);
                    let rows_res: Vec<Result<Option<Vec<EncodedRow>>, String>> = pool_ref
                        .expect("thread pool exists when n_threads>1")
                        .install(|| {
                            (start..end)
                                .into_par_iter()
                                .map(|idx| {
                                    let (row, site) = if identity_selection {
                                        bed.get_snp_row_raw(idx).ok_or_else(|| {
                                            format!("BED decode failed at row {}", idx)
                                        })?
                                    } else {
                                        bed.get_snp_row_selected_raw(idx, &selected_indices)
                                            .ok_or_else(|| {
                                                format!("BED decode with sample selection failed at row {}", idx)
                                            })?
                                    };
                                    Ok(encode_row_to_bits(
                                        row,
                                        site,
                                        mode,
                                        row_bytes,
                                        maf_f32,
                                        geno_f32,
                                        impute,
                                        het_f32,
                                    ))
                                })
                                .collect()
                        });

                    for item in rows_res.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        let maybe_rows = item.map_err(PyRuntimeError::new_err)?;
                        if let Some(rows) = maybe_rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows.len());
                            writer.write_rows(&rows).map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            } else {
                while let Some((row, site)) = if identity_selection {
                    bed.next_snp_raw()
                } else {
                    bed.next_snp_selected_raw(&selected_indices)
                } {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows) = encode_row_to_bits(
                        row, site, mode, row_bytes, maf_f32, geno_f32, impute, het_f32,
                    ) {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows.len());
                        writer.write_rows(&rows).map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
        GarfieldInputReader::Vcf(vcf) => {
            let mut batch: Vec<(Vec<f32>, SiteInfo)> = Vec::with_capacity(1024);
            loop {
                let Some((row_raw, site)) = vcf.next_snp_raw() else {
                    break;
                };
                let row = if identity_selection {
                    row_raw
                } else {
                    row_select_by_indices(row_raw, &selected_indices)
                        .map_err(PyRuntimeError::new_err)?
                };
                batch.push((row, site));
                if batch.len() >= 1024 {
                    let results = encode_batch_rows(
                        std::mem::take(&mut batch),
                        mode,
                        row_bytes,
                        maf_f32,
                        geno_f32,
                        impute,
                        het_f32,
                        pool_ref,
                    );
                    for rows in results.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        if let Some(rows_kept) = rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                            writer
                                .write_rows(&rows_kept)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            }
            if !batch.is_empty() {
                let results = encode_batch_rows(
                    std::mem::take(&mut batch),
                    mode,
                    row_bytes,
                    maf_f32,
                    geno_f32,
                    impute,
                    het_f32,
                    pool_ref,
                );
                for rows in results.into_iter() {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows_kept) = rows {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                        writer
                            .write_rows(&rows_kept)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
        GarfieldInputReader::Hmp(hmp) => {
            let mut batch: Vec<(Vec<f32>, SiteInfo)> = Vec::with_capacity(1024);
            loop {
                let Some((row_raw, site)) = hmp.next_snp_raw() else {
                    break;
                };
                let row = if identity_selection {
                    row_raw
                } else {
                    row_select_by_indices(row_raw, &selected_indices)
                        .map_err(PyRuntimeError::new_err)?
                };
                batch.push((row, site));
                if batch.len() >= 1024 {
                    let results = encode_batch_rows(
                        std::mem::take(&mut batch),
                        mode,
                        row_bytes,
                        maf_f32,
                        geno_f32,
                        impute,
                        het_f32,
                        pool_ref,
                    );
                    for rows in results.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        if let Some(rows_kept) = rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                            writer
                                .write_rows(&rows_kept)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            }
            if !batch.is_empty() {
                let results = encode_batch_rows(
                    std::mem::take(&mut batch),
                    mode,
                    row_bytes,
                    maf_f32,
                    geno_f32,
                    impute,
                    het_f32,
                    pool_ref,
                );
                for rows in results.into_iter() {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows_kept) = rows {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                        writer
                            .write_rows(&rows_kept)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
        GarfieldInputReader::Txt(txt) => {
            let mut batch: Vec<(Vec<f32>, SiteInfo)> = Vec::with_capacity(1024);
            while let Some((row_raw, site)) = txt.next_snp() {
                let row = if identity_selection {
                    row_raw
                } else {
                    row_select_by_indices(row_raw, &selected_indices)
                        .map_err(PyRuntimeError::new_err)?
                };
                batch.push((row, site));
                if batch.len() >= 1024 {
                    let results = encode_batch_rows(
                        std::mem::take(&mut batch),
                        mode,
                        row_bytes,
                        maf_f32,
                        geno_f32,
                        impute,
                        het_f32,
                        pool_ref,
                    );
                    for rows in results.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        if let Some(rows_kept) = rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                            writer
                                .write_rows(&rows_kept)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            }
            if !batch.is_empty() {
                let results = encode_batch_rows(
                    std::mem::take(&mut batch),
                    mode,
                    row_bytes,
                    maf_f32,
                    geno_f32,
                    impute,
                    het_f32,
                    pool_ref,
                );
                for rows in results.into_iter() {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows_kept) = rows {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                        writer
                            .write_rows(&rows_kept)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
    }

    let header_rows = writer.finish(n_samples).map_err(PyRuntimeError::new_err)?;
    if header_rows != n_rows_written {
        return Err(PyRuntimeError::new_err(format!(
            "internal row count mismatch: header_rows={}, tracked_rows={}",
            header_rows, n_rows_written
        )));
    }

    Ok((n_sites_seen, n_sites_written, n_rows_written, n_samples))
}

#[pyfunction(name = "garfield_subset_bin_samples")]
pub fn garfield_subset_bin_samples_py(
    bin_path: String,
    out_bin_path: String,
    sample_indices: Vec<usize>,
) -> PyResult<()> {
    let ctx = "garfield_subset_bin_samples";
    if sample_indices.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{ctx}: sample_indices is empty"
        )));
    }
    let bytes = fs::read(&bin_path).map_err(|e| PyRuntimeError::new_err(format!("{ctx}: {e}")))?;
    let (n_rows, n_samples, row_bytes, data_offset) =
        parse_bin01_header(&bytes, ctx).map_err(PyRuntimeError::new_err)?;
    for &si in sample_indices.iter() {
        if si >= n_samples {
            return Err(PyValueError::new_err(format!(
                "{ctx}: sample index out of range {} >= {}",
                si, n_samples
            )));
        }
    }

    let n_out = sample_indices.len();
    let row_bytes_out = n_out.div_ceil(8);
    let payload_len = n_rows
        .checked_mul(row_bytes_out)
        .ok_or_else(|| PyRuntimeError::new_err(format!("{ctx}: payload overflow")))?;
    let mut out = Vec::<u8>::with_capacity(BIN01_HEADER_LEN + payload_len);
    out.extend_from_slice(BIN01_MAGIC);
    out.extend_from_slice(&(n_rows as u64).to_le_bytes());
    out.extend_from_slice(&(n_out as u64).to_le_bytes());
    out.extend_from_slice(&(0u64).to_le_bytes());

    for r in 0..n_rows {
        let src_start =
            data_offset
                .checked_add(r.checked_mul(row_bytes).ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{ctx}: row offset overflow"))
                })?)
                .ok_or_else(|| PyRuntimeError::new_err(format!("{ctx}: row offset overflow")))?;
        let src_end = src_start
            .checked_add(row_bytes)
            .ok_or_else(|| PyRuntimeError::new_err(format!("{ctx}: row end overflow")))?;
        let src = &bytes[src_start..src_end];

        let mut row = vec![0u8; row_bytes_out];
        for (dst_i, &src_i) in sample_indices.iter().enumerate() {
            let src_b = src[src_i >> 3];
            let bit = (src_b >> (src_i & 7)) & 1u8;
            if bit != 0 {
                row[dst_i >> 3] |= 1u8 << (dst_i & 7);
            }
        }
        out.extend_from_slice(&row);
    }

    let mut fw = File::create(&out_bin_path)
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: create {out_bin_path}: {e}")))?;
    fw.write_all(&out)
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: write {out_bin_path}: {e}")))?;
    fw.flush()
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: flush {out_bin_path}: {e}")))?;

    copy_site_sidecar(&bin_path, &out_bin_path).map_err(PyRuntimeError::new_err)?;
    let wrote_subset_ids = write_subset_id_sidecar(&bin_path, &out_bin_path, &sample_indices)
        .map_err(PyRuntimeError::new_err)?;
    if !wrote_subset_ids {
        copy_id_sidecar(&bin_path, &out_bin_path).map_err(PyRuntimeError::new_err)?;
    }
    Ok(())
}

#[pyfunction(name = "garfield_scan_groups_bin")]
#[pyo3(signature = (
    bin_path,
    y_train,
    groups,
    response="continuous",
    max_pick=3,
    beam_width=5,
    enforce_feature_exclusion=true,
    progress_callback=None,
    progress_every=0
))]
pub fn garfield_scan_groups_bin_py(
    bin_path: String,
    y_train: PyReadonlyArray1<'_, f64>,
    groups: Vec<Vec<(String, i32, i32)>>,
    response: &str,
    max_pick: usize,
    beam_width: usize,
    enforce_feature_exclusion: bool,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<Vec<(usize, usize, f64, Vec<usize>)>> {
    if groups.is_empty() {
        return Ok(Vec::new());
    }
    let resp = parse_response(response).map_err(PyValueError::new_err)?;
    let y_vec = read_y_f64(&y_train);
    let y_bin_owned = if resp == GarfieldResponse::Binary {
        Some(validate_binary_y(&y_vec).map_err(PyValueError::new_err)?)
    } else {
        validate_continuous_y(&y_vec).map_err(PyValueError::new_err)?;
        None
    };
    let y_bin = y_bin_owned.as_deref();

    let (bits_flat_all, row_words, n_rows_all, n_samples) =
        load_bin01_as_u64_words(&bin_path).map_err(PyRuntimeError::new_err)?;
    if y_vec.len() < n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_scan_groups_bin: y length={} smaller than n_samples={}",
            y_vec.len(),
            n_samples
        )));
    }

    let sites = TxtSnpIter::new(&bin_path, None)
        .map_err(PyRuntimeError::new_err)?
        .sites;
    let chrom_idx = build_chrom_index(&sites, n_rows_all);
    let feature_group_ids_all = if enforce_feature_exclusion {
        Some(build_feature_group_ids(&sites, n_rows_all))
    } else {
        None
    };
    let total_groups = groups.len();
    let notify_step = if progress_every == 0 {
        (total_groups / 200).max(1)
    } else {
        progress_every.max(1)
    };
    let mut last_notified = 0usize;
    garfield_progress_notify(progress_callback.as_ref(), 0, total_groups)?;

    let mut out: Vec<(usize, usize, f64, Vec<usize>)> = Vec::with_capacity(groups.len());
    for (gi, group) in groups.iter().enumerate() {
        let mut idx_all: Vec<usize> = Vec::new();
        for (chrom, start, end) in group.iter() {
            let mut iv = interval_indices(&chrom_idx, chrom, *start, *end);
            idx_all.append(&mut iv);
        }
        if idx_all.is_empty() {
            out.push((gi, 0usize, f64::NEG_INFINITY, Vec::new()));
            let done = gi + 1;
            if done - last_notified >= notify_step || done == total_groups {
                garfield_progress_notify(progress_callback.as_ref(), done, total_groups)?;
                last_notified = done;
            }
            continue;
        }
        idx_all.sort_unstable();
        idx_all.dedup();

        let n_rows = idx_all.len();
        let mut bits_flat = vec![0u64; n_rows * row_words];
        for (ri, &src_idx) in idx_all.iter().enumerate() {
            let src_start = src_idx * row_words;
            let dst_start = ri * row_words;
            bits_flat[dst_start..dst_start + row_words]
                .copy_from_slice(&bits_flat_all[src_start..src_start + row_words]);
        }

        let local_group_ids = if let Some(global_ids) = feature_group_ids_all.as_ref() {
            Some(
                idx_all
                    .iter()
                    .map(|&gi| global_ids[gi])
                    .collect::<Vec<usize>>(),
            )
        } else {
            None
        };
        let res = run_beam_with_feature_exclusion(
            &bits_flat,
            row_words,
            n_rows,
            n_samples,
            resp,
            y_vec.as_slice(),
            y_bin,
            max_pick,
            beam_width,
            local_group_ids.as_deref(),
        )
        .map_err(PyRuntimeError::new_err)?;
        let selected_global = res
            .selected_indices
            .iter()
            .map(|&local_idx| idx_all[local_idx])
            .collect::<Vec<_>>();
        out.push((gi, idx_all.len(), res.score, selected_global));
        let done = gi + 1;
        if done - last_notified >= notify_step || done == total_groups {
            garfield_progress_notify(progress_callback.as_ref(), done, total_groups)?;
            last_notified = done;
        }
    }
    Ok(out)
}

#[pyfunction(name = "garfield_scan_windows_bin")]
#[pyo3(signature = (
    bin_path,
    y_train,
    response="continuous",
    max_pick=3,
    beam_width=5,
    extension=50000,
    step=None,
    enforce_feature_exclusion=true,
    progress_callback=None,
    progress_every=0
))]
pub fn garfield_scan_windows_bin_py(
    bin_path: String,
    y_train: PyReadonlyArray1<'_, f64>,
    response: &str,
    max_pick: usize,
    beam_width: usize,
    extension: usize,
    step: Option<usize>,
    enforce_feature_exclusion: bool,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<Vec<(usize, String, i32, i32, usize, f64, Vec<usize>)>> {
    let resp = parse_response(response).map_err(PyValueError::new_err)?;
    if extension == 0 {
        return Err(PyValueError::new_err(
            "garfield_scan_windows_bin: extension must be > 0",
        ));
    }
    let step_v = step.unwrap_or((extension / 2).max(1));
    if step_v == 0 {
        return Err(PyValueError::new_err(
            "garfield_scan_windows_bin: step must be > 0",
        ));
    }

    let y_vec = read_y_f64(&y_train);
    let y_bin_owned = if resp == GarfieldResponse::Binary {
        Some(validate_binary_y(&y_vec).map_err(PyValueError::new_err)?)
    } else {
        validate_continuous_y(&y_vec).map_err(PyValueError::new_err)?;
        None
    };
    let y_bin = y_bin_owned.as_deref();

    let (bits_flat_all, row_words, n_rows_all, n_samples) =
        load_bin01_as_u64_words(&bin_path).map_err(PyRuntimeError::new_err)?;
    if y_vec.len() < n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_scan_windows_bin: y length={} smaller than n_samples={}",
            y_vec.len(),
            n_samples
        )));
    }

    let sites = TxtSnpIter::new(&bin_path, None)
        .map_err(PyRuntimeError::new_err)?
        .sites;
    let windows = build_windows_from_sites(&sites, n_rows_all, extension, step_v);
    if windows.is_empty() {
        return Ok(Vec::new());
    }
    let total_windows = windows.len();
    let notify_step = if progress_every == 0 {
        (total_windows / 200).max(1)
    } else {
        progress_every.max(1)
    };
    let mut last_notified = 0usize;
    garfield_progress_notify(progress_callback.as_ref(), 0, total_windows)?;
    let feature_group_ids_all = if enforce_feature_exclusion {
        Some(build_feature_group_ids(&sites, n_rows_all))
    } else {
        None
    };

    let mut out: Vec<(usize, String, i32, i32, usize, f64, Vec<usize>)> =
        Vec::with_capacity(windows.len());
    for (wi0, win) in windows.iter().enumerate() {
        if win.indices.is_empty() {
            let done = wi0 + 1;
            if done - last_notified >= notify_step || done == total_windows {
                garfield_progress_notify(progress_callback.as_ref(), done, total_windows)?;
                last_notified = done;
            }
            continue;
        }
        let n_rows = win.indices.len();
        let sub = if n_rows == n_rows_all {
            bits_flat_all.clone()
        } else {
            gather_rows_by_indices(&bits_flat_all, row_words, &win.indices)
                .map_err(PyRuntimeError::new_err)?
        };

        let local_group_ids = if let Some(global_ids) = feature_group_ids_all.as_ref() {
            Some(
                win.indices
                    .iter()
                    .map(|&gi| global_ids[gi])
                    .collect::<Vec<usize>>(),
            )
        } else {
            None
        };

        let res = run_beam_with_feature_exclusion(
            &sub,
            row_words,
            n_rows,
            n_samples,
            resp,
            y_vec.as_slice(),
            y_bin,
            max_pick,
            beam_width,
            local_group_ids.as_deref(),
        )
        .map_err(PyRuntimeError::new_err)?;

        let selected_global = res
            .selected_indices
            .iter()
            .map(|&local_idx| win.indices[local_idx])
            .collect::<Vec<_>>();
        out.push((
            wi0 + 1,
            win.chrom.clone(),
            win.bp_start,
            win.bp_end,
            n_rows,
            res.score,
            selected_global,
        ));
        let done = wi0 + 1;
        if done - last_notified >= notify_step || done == total_windows {
            garfield_progress_notify(progress_callback.as_ref(), done, total_windows)?;
            last_notified = done;
        }
    }
    Ok(out)
}

#[pyfunction(name = "garfield_eval_rule_bin")]
#[pyo3(signature = (bin_path, y, snp_indices, response="continuous"))]
pub fn garfield_eval_rule_bin_py(
    bin_path: String,
    y: PyReadonlyArray1<'_, f64>,
    snp_indices: Vec<usize>,
    response: &str,
) -> PyResult<(f64, usize, Vec<u8>)> {
    let resp = parse_response(response).map_err(PyValueError::new_err)?;
    if snp_indices.is_empty() {
        return Ok((f64::NEG_INFINITY, 0usize, Vec::new()));
    }
    let y_vec = read_y_f64(&y);
    let y_bin = if resp == GarfieldResponse::Binary {
        Some(validate_binary_y(&y_vec).map_err(PyValueError::new_err)?)
    } else {
        validate_continuous_y(&y_vec).map_err(PyValueError::new_err)?;
        None
    };

    let (bits_flat, row_words, n_rows, n_samples) =
        load_bin01_selected_rows_as_u64_words(&bin_path, &snp_indices)
            .map_err(PyRuntimeError::new_err)?;
    if n_rows == 0 {
        return Ok((f64::NEG_INFINITY, 0usize, Vec::new()));
    }
    if y_vec.len() < n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_eval_rule_bin: y length={} smaller than n_samples={}",
            y_vec.len(),
            n_samples
        )));
    }

    let needed_words = words_for_samples(n_samples);
    let mut combined = bits_flat[0..needed_words].to_vec();
    for r in 1..n_rows {
        let st = r * row_words;
        let row = &bits_flat[st..st + needed_words];
        for (a, &b) in combined.iter_mut().zip(row.iter()) {
            *a &= b;
        }
    }
    apply_tail_mask(&mut combined, tail_mask(n_samples));

    let score = match resp {
        GarfieldResponse::Binary => {
            let yb = y_bin.as_ref().expect("binary y prepared");
            score_binary_mcc_packed(yb.as_slice(), &combined, n_samples)
        }
        GarfieldResponse::Continuous => {
            score_cont_corr_packed(y_vec.as_slice(), &combined, n_samples).abs()
        }
    };
    let support = combined
        .iter()
        .map(|w| w.count_ones() as usize)
        .sum::<usize>();
    let mut xcombine = vec![0u8; n_samples];
    for i in 0..n_samples {
        let bit = (combined[i >> 6] >> (i & 63)) & 1u64;
        xcombine[i] = if bit != 0 { 1u8 } else { 0u8 };
    }
    Ok((score, support, xcombine))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_dir(prefix: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("janusx_garfield_{prefix}_{stamp}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_row_bits(n_samples: usize, ones: &[usize]) -> Vec<u8> {
        let mut bits = vec![0u8; row_bytes_for_samples(n_samples)];
        for &idx in ones.iter() {
            bits[idx >> 3] |= 1u8 << (idx & 7);
        }
        bits
    }

    fn build_test_bits(n_rows: usize, n_samples: usize) -> (Vec<u64>, usize, Vec<usize>) {
        let row_words = words_for_samples(n_samples);
        let mut bits_flat = vec![0u64; n_rows * row_words];
        for row in 0..n_rows {
            for word in 0..row_words {
                let a = (row as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let b = (word as u64 + 3).wrapping_mul(0xD1B5_4A32_D192_ED03);
                let rot = ((row * 11 + word * 7) % 63 + 1) as u32;
                let dense = (!0u64).rotate_right(((row * 5 + word * 13) % 64) as u32);
                bits_flat[row * row_words + word] = (a ^ b).rotate_left(rot) | dense;
            }
        }
        let group_ids = (0..n_rows).map(|i| i % 96).collect::<Vec<_>>();
        (bits_flat, row_words, group_ids)
    }

    #[test]
    fn test_group_exclusion_parallel_matches_serial() {
        let n_rows = 1024usize;
        let n_samples = 256usize;
        let max_pick = 3usize;
        let beam_width = 8usize;
        let (bits_flat, row_words, group_ids) = build_test_bits(n_rows, n_samples);
        let score_fn =
            |combined: &[u64]| combined.iter().map(|w| w.count_ones() as f64).sum::<f64>();

        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("build serial pool")
            .install(|| {
                beam_search_and_with_group_exclusion(
                    &bits_flat, row_words, n_rows, n_samples, max_pick, beam_width, &group_ids,
                    score_fn,
                )
                .expect("serial constrained beam")
            });

        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("build parallel pool")
            .install(|| {
                beam_search_and_with_group_exclusion(
                    &bits_flat, row_words, n_rows, n_samples, max_pick, beam_width, &group_ids,
                    score_fn,
                )
                .expect("parallel constrained beam")
            });

        assert_eq!(serial.selected_indices, parallel.selected_indices);
        assert_eq!(serial.combined_bits, parallel.combined_bits);
        assert_eq!(serial.score, parallel.score);
    }

    #[test]
    fn test_load_bin01_packed_roundtrip() {
        let dir = make_temp_dir("bin01_load");
        let bin_path = dir.join("toy.bin");
        let bin_str = bin_path.to_str().unwrap();
        let n_samples = 10usize;
        let sample_ids = (0..n_samples).map(|i| format!("s{i}")).collect::<Vec<_>>();
        write_sample_id_sidecar(bin_str, &sample_ids).unwrap();
        let mut writer = GarfieldBinWriter::new(bin_str, n_samples).unwrap();
        writer
            .write_rows(&[
                EncodedRow {
                    site: SiteInfo {
                        chrom: "1".to_string(),
                        pos: 10,
                        ref_allele: "A".to_string(),
                        alt_allele: "T|BIN".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[1, 3, 8]),
                },
                EncodedRow {
                    site: SiteInfo {
                        chrom: "1".to_string(),
                        pos: 20,
                        ref_allele: "A".to_string(),
                        alt_allele: "C|BIN".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[0, 4, 9]),
                },
            ])
            .unwrap();
        writer.finish(n_samples).unwrap();

        let (packed, group_ids, got_n_samples, n_rows, row_bytes) =
            load_bin01_packed_owned(bin_str, false).unwrap();
        assert_eq!(got_n_samples, n_samples);
        assert_eq!(n_rows, 2);
        assert_eq!(row_bytes, row_bytes_for_samples(n_samples));
        assert_eq!(group_ids, vec![0u64, 1u64]);
        assert_eq!(
            packed,
            [
                make_row_bits(n_samples, &[1, 3, 8]),
                make_row_bits(n_samples, &[0, 4, 9]),
            ]
            .concat()
        );
    }

    #[test]
    fn test_load_mbin_packed_groups_triplets() {
        let dir = make_temp_dir("mbin_load");
        let bin_path = dir.join("toy.mbin.bin");
        let bin_str = bin_path.to_str().unwrap();
        let n_samples = 9usize;
        let sample_ids = (0..n_samples).map(|i| format!("s{i}")).collect::<Vec<_>>();
        write_sample_id_sidecar(bin_str, &sample_ids).unwrap();
        let mut writer = GarfieldBinWriter::new(bin_str, n_samples).unwrap();
        writer
            .write_rows(&[
                EncodedRow {
                    site: SiteInfo {
                        chrom: "2".to_string(),
                        pos: 100,
                        ref_allele: "A".to_string(),
                        alt_allele: "G|DOM".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[0, 1, 4, 8]),
                },
                EncodedRow {
                    site: SiteInfo {
                        chrom: "2".to_string(),
                        pos: 100,
                        ref_allele: "A".to_string(),
                        alt_allele: "G|REC".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[1, 8]),
                },
                EncodedRow {
                    site: SiteInfo {
                        chrom: "2".to_string(),
                        pos: 100,
                        ref_allele: "A".to_string(),
                        alt_allele: "G|HET".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[0, 4]),
                },
                EncodedRow {
                    site: SiteInfo {
                        chrom: "2".to_string(),
                        pos: 150,
                        ref_allele: "C".to_string(),
                        alt_allele: "T|DOM".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[2, 3, 5]),
                },
                EncodedRow {
                    site: SiteInfo {
                        chrom: "2".to_string(),
                        pos: 150,
                        ref_allele: "C".to_string(),
                        alt_allele: "T|REC".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[5]),
                },
                EncodedRow {
                    site: SiteInfo {
                        chrom: "2".to_string(),
                        pos: 150,
                        ref_allele: "C".to_string(),
                        alt_allele: "T|HET".to_string(),
                    },
                    bits: make_row_bits(n_samples, &[2, 3]),
                },
            ])
            .unwrap();
        writer.finish(n_samples).unwrap();

        let (packed, group_ids, got_n_samples, n_rows, row_bytes) =
            load_bin01_packed_owned(bin_str, true).unwrap();
        assert_eq!(got_n_samples, n_samples);
        assert_eq!(n_rows, 6);
        assert_eq!(row_bytes, row_bytes_for_samples(n_samples));
        assert_eq!(group_ids, vec![0u64, 0, 0, 1, 1, 1]);
        assert_eq!(packed.len(), n_rows * row_bytes);
    }

    #[test]
    fn test_parse_beam_logic_gate_mode_aliases() {
        assert_eq!(
            parse_beam_logic_gate_mode("A").unwrap(),
            BeamLogicGateMode::AndOnly
        );
        assert_eq!(
            parse_beam_logic_gate_mode("or").unwrap(),
            BeamLogicGateMode::OrOnly
        );
        assert_eq!(
            parse_beam_logic_gate_mode("and_or").unwrap(),
            BeamLogicGateMode::AndOr
        );
    }

    #[test]
    fn test_apply_logic_rule_output_limit_ratio() {
        let mut records = (0..10usize)
            .map(|i| GarfieldLogicRuleRecord {
                unit_name: format!("u{i}"),
                unit_kind: "window".to_string(),
                unit_index: i + 1,
                region_size: 8,
                ml_feature_count: 4,
                selected_row_indices: vec![i],
                snp_name: format!("snp{i}"),
                expr: format!("expr{i}"),
                chrom_field: "window".to_string(),
                pos: i as i32,
                beta: i as f64,
                se: 0.1 + i as f64 * 0.01,
                chisq: 5.0 + i as f64,
                pwald: 1e-4 * (i as f64 + 1.0),
                train_score: 10.0 - i as f64,
                test_score: 20.0 - i as f64,
                full_bits: vec![i as u64 + 1],
            })
            .collect::<Vec<_>>();
        apply_logic_rule_output_limit(&mut records, 0, 0.3).unwrap();
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn test_extend_keep_with_score_ties_keeps_full_tie_block() {
        let scores = vec![10.0_f64, 9.0, 9.0, 8.0];
        let keep = extend_keep_with_score_ties(scores.as_slice(), 2, |v| *v);
        assert_eq!(keep, 3);
    }

    #[test]
    fn test_apply_logic_rule_output_limit_count_keeps_ties() {
        let mut records = vec![
            GarfieldLogicRuleRecord {
                unit_name: "u1".to_string(),
                unit_kind: "window".to_string(),
                unit_index: 1,
                region_size: 8,
                ml_feature_count: 4,
                selected_row_indices: vec![1],
                snp_name: "snp1".to_string(),
                expr: "expr1".to_string(),
                chrom_field: "window".to_string(),
                pos: 1,
                beta: 1.0,
                se: 0.1,
                chisq: 10.0,
                pwald: 1e-4,
                train_score: 8.0,
                test_score: 10.0,
                full_bits: vec![1],
            },
            GarfieldLogicRuleRecord {
                unit_name: "u2".to_string(),
                unit_kind: "window".to_string(),
                unit_index: 2,
                region_size: 8,
                ml_feature_count: 4,
                selected_row_indices: vec![2],
                snp_name: "snp2".to_string(),
                expr: "expr2".to_string(),
                chrom_field: "window".to_string(),
                pos: 2,
                beta: 2.0,
                se: 0.2,
                chisq: 11.0,
                pwald: 2e-4,
                train_score: 7.0,
                test_score: 9.0,
                full_bits: vec![2],
            },
            GarfieldLogicRuleRecord {
                unit_name: "u3".to_string(),
                unit_kind: "window".to_string(),
                unit_index: 3,
                region_size: 8,
                ml_feature_count: 4,
                selected_row_indices: vec![3],
                snp_name: "snp3".to_string(),
                expr: "expr3".to_string(),
                chrom_field: "window".to_string(),
                pos: 3,
                beta: 3.0,
                se: 0.3,
                chisq: 12.0,
                pwald: 3e-4,
                train_score: 6.0,
                test_score: 9.0,
                full_bits: vec![3],
            },
            GarfieldLogicRuleRecord {
                unit_name: "u4".to_string(),
                unit_kind: "window".to_string(),
                unit_index: 4,
                region_size: 8,
                ml_feature_count: 4,
                selected_row_indices: vec![4],
                snp_name: "snp4".to_string(),
                expr: "expr4".to_string(),
                chrom_field: "window".to_string(),
                pos: 4,
                beta: 4.0,
                se: 0.4,
                chisq: 13.0,
                pwald: 4e-4,
                train_score: 5.0,
                test_score: 8.0,
                full_bits: vec![4],
            },
        ];
        apply_logic_rule_output_limit(&mut records, 2, 0.0).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[1].test_score, records[2].test_score);
    }
}
