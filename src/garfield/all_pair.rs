//! Independent exhaustive order-2 scanner for packed GARFIELD BIN rows.
//!
//! This module is deliberately separate from the Beam implementation.  An
//! all-pair scan must evaluate every `i < j` pair and all four signed AND
//! variants without using parent-gain, min-gain, singleton-prefix, or Beam
//! retention rules.  An optional fifth, canonical XOR mask can be enabled for
//! order-2 exploratory searches.  It is a proposal/raw-design primitive;
//! formal inference remains owned by the existing GARFIELD pipeline.

use crate::breader::load_bin01_as_u64_words;
use crate::bstats::{tail_mask, words_for_samples};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::logic_classifier::{
    array2_to_vec, bool_array2_to_vec, summaries_to_py_dict, summarize_binary_pairs_marker_major,
    BinaryEncoding,
};
use super::score::{
    score_cont_centered_gain_from_sum_and_n_hit, validate_continuous_y, PackedYSumLookup,
};

/// The gate used for an all-pair candidate.
///
/// `Xor` is represented canonically without a polarity bit: XNOR is the
/// complement of XOR and has the same centered-Gain score, so evaluating both
/// would only duplicate work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum AllPairGate {
    And,
    Xor,
}

/// The best signed interpretation for one unordered marker pair.
///
/// For `And`, `polarity` is a two-bit mask: bit 0 negates `first`, bit 1
/// negates `second`.  For `Xor`, polarity is always zero because XOR/XNOR are
/// centered-Gain complements.  The scanner keeps one record per *site pair*,
/// not one record per gate variant, so pair Top-K remains bounded.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AllPairCandidate {
    pub(crate) first: usize,
    pub(crate) second: usize,
    pub(crate) gate: AllPairGate,
    pub(crate) polarity: u8,
    pub(crate) raw_score: f64,
    pub(crate) support: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AllPairScanResult {
    pub(crate) candidates: Vec<AllPairCandidate>,
    pub(crate) pairs_evaluated: usize,
    pub(crate) polarity_evaluated: usize,
    pub(crate) n_rows: usize,
    pub(crate) n_samples: usize,
}

#[inline]
fn candidate_cmp(a: &AllPairCandidate, b: &AllPairCandidate) -> Ordering {
    let gate_rank = |gate: AllPairGate| match gate {
        // Preserve the historical AND winner for exact ties.  XOR remains
        // deterministic, but does not replace an equally scoring AND rule.
        AllPairGate::Xor => 0u8,
        AllPairGate::And => 1u8,
    };
    a.raw_score
        .total_cmp(&b.raw_score)
        // Lower pair indices are the deterministic winner for exact ties.
        .then_with(|| b.first.cmp(&a.first))
        .then_with(|| b.second.cmp(&a.second))
        .then_with(|| gate_rank(a.gate).cmp(&gate_rank(b.gate)))
        .then_with(|| b.polarity.cmp(&a.polarity))
}

#[derive(Clone, Debug)]
struct HeapEntry(AllPairCandidate);

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.first == other.0.first
            && self.0.second == other.0.second
            && self.0.gate == other.0.gate
            && self.0.polarity == other.0.polarity
            && self.0.support == other.0.support
            && self.0.raw_score.to_bits() == other.0.raw_score.to_bits()
    }
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        candidate_cmp(&self.0, &other.0)
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn push_top_k(
    heap: &mut BinaryHeap<Reverse<HeapEntry>>,
    candidate: AllPairCandidate,
    top_k: usize,
) {
    if top_k == 0 {
        return;
    }
    let item = Reverse(HeapEntry(candidate));
    if heap.len() < top_k {
        heap.push(item);
        return;
    }
    let replace = heap
        .peek()
        .map(|worst| item.0.cmp(&worst.0) == Ordering::Greater)
        .unwrap_or(true);
    if replace {
        let _ = heap.pop();
        heap.push(item);
    }
}

#[inline]
fn row_slice<'a>(bits_flat: &'a [u64], row_words: usize, row: usize, words: usize) -> &'a [u64] {
    let start = row * row_words;
    &bits_flat[start..start + words]
}

#[inline]
fn best_pair_candidate_fused(
    first: usize,
    second: usize,
    bits_flat: &[u64],
    negated: &[u64],
    row_words: usize,
    needed_words: usize,
    tail: Option<u64>,
    n_samples: usize,
    lookup: &PackedYSumLookup,
    total_sum: f64,
    xor_search: bool,
) -> AllPairCandidate {
    let first_pos = row_slice(bits_flat, row_words, first, needed_words);
    let second_pos = row_slice(bits_flat, row_words, second, needed_words);
    let first_neg = row_slice(negated, needed_words, first, needed_words);
    let second_neg = row_slice(negated, needed_words, second, needed_words);
    let n_variants = if xor_search { 5 } else { 4 };
    let mut counts = [0usize; 5];
    let mut sums = [0.0f64; 5];
    for word_idx in 0..needed_words {
        let mut lhs = first_pos[word_idx];
        let mut rhs = second_pos[word_idx];
        if word_idx + 1 == needed_words {
            if let Some(mask) = tail {
                lhs &= mask;
                rhs &= mask;
            }
        }
        let and_words = [
            lhs & rhs,
            first_neg[word_idx] & rhs,
            lhs & second_neg[word_idx],
            first_neg[word_idx] & second_neg[word_idx],
        ];
        for (variant, word) in and_words.into_iter().enumerate() {
            counts[variant] = counts[variant].saturating_add(word.count_ones() as usize);
            sums[variant] += lookup.sum_word(word_idx, word);
        }
        if xor_search {
            let xor_word = lhs ^ rhs;
            counts[4] = counts[4].saturating_add(xor_word.count_ones() as usize);
            sums[4] += lookup.sum_word(word_idx, xor_word);
        }
    }
    let mut best: Option<AllPairCandidate> = None;
    for variant in 0..n_variants {
        let is_xor = variant == 4;
        let polarity = if is_xor { 0 } else { variant as u8 };
        let score = score_cont_centered_gain_from_sum_and_n_hit(
            total_sum,
            sums[variant],
            n_samples,
            counts[variant],
        );
        let candidate = AllPairCandidate {
            first,
            second,
            gate: if is_xor {
                AllPairGate::Xor
            } else {
                AllPairGate::And
            },
            polarity,
            raw_score: score.raw_score,
            support: counts[variant],
        };
        if best
            .as_ref()
            .map(|current| candidate_cmp(&candidate, current) == Ordering::Greater)
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }
    best.expect("pair has at least four polarity variants")
}

fn validate_scan_inputs(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    y: &[f64],
    n_samples: usize,
) -> Result<usize, String> {
    if n_rows == 0 {
        return Err("garfield_all_pair_scan: n_rows must be > 0".to_string());
    }
    if n_samples == 0 {
        return Err("garfield_all_pair_scan: n_samples must be > 0".to_string());
    }
    if y.len() < n_samples {
        return Err(format!(
            "garfield_all_pair_scan: y length={} smaller than n_samples={n_samples}",
            y.len()
        ));
    }
    validate_continuous_y(y, n_samples, "garfield_all_pair_scan")?;
    let words = words_for_samples(n_samples).max(1);
    if row_words < words {
        return Err(format!(
            "garfield_all_pair_scan: row_words={row_words} smaller than required {words}"
        ));
    }
    let required = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| "garfield_all_pair_scan: packed matrix size overflow".to_string())?;
    if bits_flat.len() < required {
        return Err(format!(
            "garfield_all_pair_scan: bits length={} smaller than n_rows*row_words={required}",
            bits_flat.len()
        ));
    }
    Ok(words)
}

/// Enumerate every unordered pair and its four signed AND variants.
///
/// This function intentionally has no search pruning.  It only keeps the
/// requested number of best *site pairs* in a bounded heap; all four polarity
/// variants are evaluated before selecting the best variant for that pair.
/// The historical API keeps XOR disabled; callers that need the optional
/// canonical XOR gate should use
/// [`scan_all_pairs_continuous_packed_with_parallel_and_xor`].
#[cfg(test)]
pub(crate) fn scan_all_pairs_continuous_packed(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    y: &[f64],
    n_samples: usize,
    top_k: usize,
) -> Result<AllPairScanResult, String> {
    scan_all_pairs_continuous_packed_with_parallel(
        bits_flat, row_words, n_rows, y, n_samples, top_k, true,
    )
}

/// Internal variant used by the window scanner to avoid nested Rayon pools.
/// When many windows already run in parallel, callers pass `false` and let
/// the outer scheduler provide the parallelism; standalone callers retain
/// the historical parallel default through [`scan_all_pairs_continuous_packed`].
#[cfg(test)]
pub(crate) fn scan_all_pairs_continuous_packed_with_parallel(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    y: &[f64],
    n_samples: usize,
    top_k: usize,
    allow_parallel: bool,
) -> Result<AllPairScanResult, String> {
    scan_all_pairs_continuous_packed_with_parallel_and_xor(
        bits_flat,
        row_words,
        n_rows,
        y,
        n_samples,
        top_k,
        allow_parallel,
        false,
    )
}

/// Internal all-pair scanner with an optional canonical XOR gate.
///
/// `xor_search=false` is the historical four-AND path and keeps its candidate
/// ordering and counters unchanged.  When enabled, each pair evaluates one
/// additional `lhs ^ rhs` mask; its complement (XNOR) is intentionally omitted
/// because centered Gain is invariant to group complementation.
pub(crate) fn scan_all_pairs_continuous_packed_with_parallel_and_xor(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    y: &[f64],
    n_samples: usize,
    top_k: usize,
    allow_parallel: bool,
    xor_search: bool,
) -> Result<AllPairScanResult, String> {
    let needed_words = validate_scan_inputs(bits_flat, row_words, n_rows, y, n_samples)?;
    let tail = tail_mask(n_samples);

    // Materialize complements once.  This is tiny compared with the genotype
    // matrix and avoids rebuilding `!row` four times for every pair.
    let total_words = n_rows
        .checked_mul(needed_words)
        .ok_or_else(|| "garfield_all_pair_scan: complement size overflow".to_string())?;
    let mut negated = vec![0u64; total_words];
    for r in 0..n_rows {
        let src = row_slice(bits_flat, row_words, r, needed_words);
        let dst_start = r * needed_words;
        let dst = &mut negated[dst_start..dst_start + needed_words];
        for (out, &word) in dst.iter_mut().zip(src.iter()) {
            *out = !word;
        }
        if let Some(mask) = tail {
            if let Some(last) = dst.last_mut() {
                *last &= mask;
            }
        }
    }

    let lookup = PackedYSumLookup::build(y, n_samples)?;
    let total_sum = y.iter().take(n_samples).copied().sum::<f64>();
    let first_rows = 0..n_rows.saturating_sub(1);
    let heap_capacity = top_k.min(1024);
    let heap = if allow_parallel && rayon::current_num_threads() > 1 && n_rows >= 32 {
        // Fold into one bounded heap per Rayon worker rather than one heap per
        // first-row index.  The latter scales as O(n_rows * top_k) in peak
        // intermediate storage and can dominate the actual pair scan for
        // large windows.
        first_rows
            .into_par_iter()
            .fold(
                || BinaryHeap::<Reverse<HeapEntry>>::with_capacity(heap_capacity),
                |mut local_heap, first| {
                    for second in (first + 1)..n_rows {
                        let candidate = best_pair_candidate_fused(
                            first,
                            second,
                            bits_flat,
                            &negated,
                            row_words,
                            needed_words,
                            tail,
                            n_samples,
                            &lookup,
                            total_sum,
                            xor_search,
                        );
                        push_top_k(&mut local_heap, candidate, top_k);
                    }
                    local_heap
                },
            )
            .reduce(
                || BinaryHeap::<Reverse<HeapEntry>>::with_capacity(heap_capacity),
                |mut left, right| {
                    for Reverse(entry) in right {
                        push_top_k(&mut left, entry.0, top_k);
                    }
                    left
                },
            )
    } else {
        let mut serial_heap = BinaryHeap::<Reverse<HeapEntry>>::with_capacity(heap_capacity);
        for first in first_rows {
            for second in (first + 1)..n_rows {
                let candidate = best_pair_candidate_fused(
                    first,
                    second,
                    bits_flat,
                    &negated,
                    row_words,
                    needed_words,
                    tail,
                    n_samples,
                    &lookup,
                    total_sum,
                    xor_search,
                );
                push_top_k(&mut serial_heap, candidate, top_k);
            }
        }
        serial_heap
    };
    let pairs_evaluated = n_rows
        .checked_mul(n_rows.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| "garfield_all_pair_scan: pair counter overflow".to_string())?;
    let polarity_evaluated = pairs_evaluated
        .checked_mul(if xor_search { 5 } else { 4 })
        .ok_or_else(|| "garfield_all_pair_scan: polarity counter overflow".to_string())?;

    let mut candidates = heap
        .into_iter()
        .map(|Reverse(entry)| entry.0)
        .collect::<Vec<_>>();
    candidates.sort_unstable_by(|a, b| candidate_cmp(b, a));
    Ok(AllPairScanResult {
        candidates,
        pairs_evaluated,
        polarity_evaluated,
        n_rows,
        n_samples,
    })
}

/// Python diagnostic wrapper for the independent order-2 scanner.
///
/// Returns `(candidates, pairs_evaluated, polarity_evaluated, n_rows,
/// n_samples)`, with candidates represented as `(first, second, polarity,
/// raw_score, support)` tuples.  The API is intentionally standalone so the
/// production Beam/search path remains unchanged while all-pair benchmarks
/// establish the raw-score oracle ceiling.
fn scan_all_pair_bin_py_impl<'py>(
    bin_path: &str,
    y: PyReadonlyArray1<'py, f64>,
    top_k: usize,
    xor_search: bool,
) -> PyResult<AllPairScanResult> {
    let (bits, row_words, n_rows, n_samples) =
        load_bin01_as_u64_words(bin_path, "garfield_all_pair_scan_bin")
            .map_err(PyRuntimeError::new_err)?;
    let y_vec = y
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("garfield_all_pair_scan_bin: {e}")))?;
    scan_all_pairs_continuous_packed_with_parallel_and_xor(
        &bits, row_words, n_rows, y_vec, n_samples, top_k, true, xor_search,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction(name = "garfield_all_pair_scan_bin")]
#[pyo3(signature = (bin_path, y, top_k=100))]
pub fn garfield_all_pair_scan_bin_py(
    bin_path: String,
    y: PyReadonlyArray1<'_, f64>,
    top_k: usize,
) -> PyResult<(
    Vec<(usize, usize, u8, f64, usize)>,
    usize,
    usize,
    usize,
    usize,
)> {
    let result = scan_all_pair_bin_py_impl(&bin_path, y, top_k, false)?;
    let candidates = result
        .candidates
        .into_iter()
        .map(|candidate| {
            (
                candidate.first,
                candidate.second,
                candidate.polarity,
                candidate.raw_score,
                candidate.support,
            )
        })
        .collect();
    Ok((
        candidates,
        result.pairs_evaluated,
        result.polarity_evaluated,
        result.n_rows,
        result.n_samples,
    ))
}

/// Python benchmark wrapper for the five-gate order-2 scanner.
///
/// This preserves the historical `garfield_all_pair_scan_bin` tuple and adds
/// an explicit gate label for the optional canonical XOR candidate.  The
/// wrapper is intentionally diagnostic/benchmark-facing; the normal GARFIELD
/// CLI enables the same gate through `--xor-search`.
#[pyfunction(name = "garfield_all_pair_scan_bin_xor")]
#[pyo3(signature = (bin_path, y, top_k=100))]
pub fn garfield_all_pair_scan_bin_xor_py(
    bin_path: String,
    y: PyReadonlyArray1<'_, f64>,
    top_k: usize,
) -> PyResult<(
    Vec<(usize, usize, String, u8, f64, usize)>,
    usize,
    usize,
    usize,
    usize,
)> {
    let result = scan_all_pair_bin_py_impl(&bin_path, y, top_k, true)?;
    let candidates = result
        .candidates
        .into_iter()
        .map(|candidate| {
            (
                candidate.first,
                candidate.second,
                match candidate.gate {
                    AllPairGate::And => "AND".to_string(),
                    AllPairGate::Xor => "XOR".to_string(),
                },
                candidate.polarity,
                candidate.raw_score,
                candidate.support,
            )
        })
        .collect();
    Ok((
        candidates,
        result.pairs_evaluated,
        result.polarity_evaluated,
        result.n_rows,
        result.n_samples,
    ))
}

/// Run the unchanged Logic5 all-pair discovery and post-process only its
/// bounded Top-K pairs into four-cell summaries/classifications.
///
/// The returned dictionary keeps the discovery candidates and counters while
/// adding the columns produced by `garfield_logic_pair_cell_summary`.  The
/// existing `garfield_all_pair_scan_bin` and `_xor` tuple APIs are intentionally
/// untouched for regression compatibility.
#[pyfunction(name = "garfield_all_pair_scan_bin_with_cells")]
#[pyo3(signature = (bin_path, y, genotypes, top_k=100, valid_mask=None, encoding="auto", min_cell_count=1, xor_search=true))]
pub fn garfield_all_pair_scan_bin_with_cells_py<'py>(
    py: Python<'py>,
    bin_path: String,
    y: PyReadonlyArray1<'py, f64>,
    genotypes: PyReadonlyArray2<'py, f64>,
    top_k: usize,
    valid_mask: Option<PyReadonlyArray2<'py, bool>>,
    encoding: &str,
    min_cell_count: usize,
    xor_search: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let result = scan_all_pair_bin_py_impl(&bin_path, y.clone(), top_k, xor_search)?;
    let genotype_shape = genotypes.shape();
    if genotype_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "genotypes must be a 2D marker-major array",
        ));
    }
    let n_markers = genotype_shape[0];
    let n_samples = genotype_shape[1];
    if n_markers != result.n_rows || n_samples != result.n_samples {
        return Err(PyValueError::new_err(format!(
            "genotypes shape=({n_markers}, {n_samples}) does not match BIN shape=({}, {})",
            result.n_rows, result.n_samples
        )));
    }
    let genotype_vec = array2_to_vec(&genotypes);
    let mask_vec = valid_mask.as_ref().map(bool_array2_to_vec);
    if let Some(mask) = mask_vec.as_ref() {
        if valid_mask.as_ref().map(|array| array.shape()) != Some(&genotype_shape[..]) {
            return Err(PyValueError::new_err(
                "valid_mask must have the same shape as genotypes",
            ));
        }
        if mask.len() != genotype_vec.len() {
            return Err(PyValueError::new_err(
                "valid_mask size does not match genotypes",
            ));
        }
    }
    let encoding = BinaryEncoding::parse(encoding).map_err(PyValueError::new_err)?;
    let pairs = result
        .candidates
        .iter()
        .map(|candidate| (candidate.first, candidate.second))
        .collect::<Vec<_>>();
    let y_vec = y
        .as_array()
        .iter()
        .take(n_samples)
        .copied()
        .collect::<Vec<_>>();
    let summaries = summarize_binary_pairs_marker_major(
        genotype_vec.as_slice(),
        n_markers,
        n_samples,
        y_vec.as_slice(),
        pairs.as_slice(),
        mask_vec.as_deref(),
        encoding,
        min_cell_count,
    )
    .map_err(PyValueError::new_err)?;
    let out = summaries_to_py_dict(py, pairs.as_slice(), summaries.as_slice())?;
    let candidates = result
        .candidates
        .iter()
        .map(|candidate| {
            (
                candidate.first,
                candidate.second,
                match candidate.gate {
                    AllPairGate::And => "AND",
                    AllPairGate::Xor => "XOR",
                },
                candidate.polarity,
                candidate.raw_score,
                candidate.support,
            )
        })
        .collect::<Vec<_>>();
    out.set_item("candidates", candidates)?;
    out.set_item("pairs_evaluated", result.pairs_evaluated)?;
    out.set_item("polarity_evaluated", result.polarity_evaluated)?;
    out.set_item("n_rows", result.n_rows)?;
    out.set_item("n_samples", result.n_samples)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(n_samples: usize, ones: &[usize]) -> Vec<u64> {
        let mut out = vec![0u64; n_samples.div_ceil(64).max(1)];
        for &idx in ones {
            out[idx >> 6] |= 1u64 << (idx & 63);
        }
        if let Some(rem) = n_samples.checked_rem(64) {
            if rem != 0 {
                let last = out.len() - 1;
                out[last] &= (1u64 << rem) - 1;
            }
        }
        out
    }

    #[test]
    fn all_pair_scan_visits_every_pair_and_polarity() {
        let n_samples = 7;
        let bits = [
            row(n_samples, &[0, 1, 4]),
            row(n_samples, &[1, 2, 5]),
            row(n_samples, &[0, 2, 3, 6]),
            row(n_samples, &[3, 4, 6]),
        ]
        .concat();
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let got = scan_all_pairs_continuous_packed(&bits, 1, 4, &y, n_samples, 3).unwrap();
        assert_eq!(got.pairs_evaluated, 6);
        assert_eq!(got.polarity_evaluated, 24);
        assert_eq!(got.n_rows, 4);
        assert_eq!(got.n_samples, n_samples);
    }

    #[test]
    fn zero_top_k_still_scans_all_pairs() {
        let n_samples = 5;
        let bits = [
            row(n_samples, &[0, 1]),
            row(n_samples, &[1, 2]),
            row(n_samples, &[2, 3]),
        ]
        .concat();
        let y = [0.0, 1.0, 2.0, 3.0, 4.0];
        let got = scan_all_pairs_continuous_packed(&bits, 1, 3, &y, n_samples, 0).unwrap();
        assert!(got.candidates.is_empty());
        assert_eq!(got.pairs_evaluated, 3);
        assert_eq!(got.polarity_evaluated, 12);
    }

    #[test]
    fn streaming_top_k_matches_reference_sort() {
        let n_samples = 9;
        let bits = [
            row(n_samples, &[0, 1, 4, 8]),
            row(n_samples, &[1, 2, 5, 8]),
            row(n_samples, &[0, 2, 3, 6]),
            row(n_samples, &[3, 4, 6, 7]),
        ]
        .concat();
        let y = [0.0, 1.5, 2.0, 3.5, 4.0, 5.5, 6.0, 7.5, 8.0];
        let got = scan_all_pairs_continuous_packed(&bits, 1, 4, &y, n_samples, 3).unwrap();
        let reference = reference_all_pairs(&bits, 1, 4, &y);
        assert_eq!(got.candidates.len(), 3);
        for (got, expected) in got.candidates.iter().zip(reference.iter().take(3)) {
            assert_eq!(
                (got.first, got.second, got.polarity, got.support),
                (
                    expected.first,
                    expected.second,
                    expected.polarity,
                    expected.support
                )
            );
            assert!((got.raw_score - expected.raw_score).abs() < 1e-9);
        }
    }

    #[test]
    fn polarity_complement_masks_tail_and_matches_reference() {
        let n_samples = 65;
        let bits = [row(n_samples, &[0, 1, 63, 64]), row(n_samples, &[1, 2, 63])].concat();
        let y = (0..n_samples).map(|v| v as f64).collect::<Vec<_>>();
        let got = scan_all_pairs_continuous_packed(&bits, 2, 2, &y, n_samples, 4).unwrap();
        let reference = reference_all_pairs(&bits, 2, 2, &y);
        assert_eq!(got.candidates.len(), reference.len());
        for (got, expected) in got.candidates.iter().zip(reference.iter()) {
            assert_eq!(
                (got.first, got.second, got.polarity, got.support),
                (
                    expected.first,
                    expected.second,
                    expected.polarity,
                    expected.support
                )
            );
            assert!((got.raw_score - expected.raw_score).abs() < 1e-8);
        }
        assert!(got
            .candidates
            .iter()
            .all(|candidate| candidate.support <= n_samples));
    }

    #[test]
    fn parallel_all_pair_matches_serial_results() {
        let n_samples = 137;
        let bits = [
            row(n_samples, &[0, 1, 4, 65, 100]),
            row(n_samples, &[1, 2, 5, 70, 100]),
            row(n_samples, &[0, 2, 3, 66, 101]),
            row(n_samples, &[3, 4, 6, 67, 102]),
            row(n_samples, &[10, 20, 30, 90, 120]),
            row(n_samples, &[11, 21, 31, 91, 121]),
        ]
        .concat();
        let y = (0..n_samples)
            .map(|idx| (idx as f64) * 0.17 - 4.0 + ((idx % 11) as f64) * 0.03)
            .collect::<Vec<_>>();
        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let serial = serial_pool
            .install(|| scan_all_pairs_continuous_packed(&bits, 3, 6, &y, n_samples, 8).unwrap());
        let parallel = parallel_pool
            .install(|| scan_all_pairs_continuous_packed(&bits, 3, 6, &y, n_samples, 8).unwrap());
        assert_eq!(serial, parallel);
    }

    #[test]
    fn xor_search_adds_one_canonical_gate_to_the_four_and_masks() {
        // A xor B is the high-residual group; its complement (XNOR) has the
        // same centered-Gain score and therefore does not need a second scan.
        let n_samples = 8;
        let bits = [row(n_samples, &[0, 1, 2, 3]), row(n_samples, &[0, 1, 4, 5])].concat();
        let y = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];

        let without_xor = scan_all_pairs_continuous_packed_with_parallel_and_xor(
            &bits, 1, 2, &y, n_samples, 2, false, false,
        )
        .unwrap();
        assert_eq!(without_xor.polarity_evaluated, 4);
        assert_eq!(without_xor.candidates[0].gate, AllPairGate::And);

        let with_xor = scan_all_pairs_continuous_packed_with_parallel_and_xor(
            &bits, 1, 2, &y, n_samples, 2, false, true,
        )
        .unwrap();
        assert_eq!(with_xor.polarity_evaluated, 5);
        assert_eq!(with_xor.candidates[0].gate, AllPairGate::Xor);
        assert_eq!(with_xor.candidates[0].polarity, 0);
        assert_eq!(with_xor.candidates[0].support, 4);
    }

    #[test]
    fn xor_and_its_complement_have_identical_centered_gain() {
        let n_samples = 9;
        let lhs = row(n_samples, &[0, 1, 4, 7]);
        let rhs = row(n_samples, &[1, 2, 4, 8]);
        let xor = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&a, &b)| a ^ b)
            .collect::<Vec<_>>();
        let xnor = xor.iter().map(|word| !word).collect::<Vec<_>>();
        let mut xnor = xnor;
        if let Some(mask) = tail_mask(n_samples) {
            *xnor.last_mut().expect("at least one packed word") &= mask;
        }
        let y = [0.2, -0.4, 1.1, 0.7, -0.2, 0.9, -0.8, 0.3, 1.4];
        let lookup = PackedYSumLookup::build(&y, n_samples).unwrap();
        let total_sum = y.iter().sum::<f64>();
        let (xor_n, xor_sum) =
            xor.iter()
                .enumerate()
                .fold((0usize, 0.0), |(count, sum), (word_idx, &word)| {
                    (
                        count + word.count_ones() as usize,
                        sum + lookup.sum_word(word_idx, word),
                    )
                });
        let (xnor_n, xnor_sum) =
            xnor.iter()
                .enumerate()
                .fold((0usize, 0.0), |(count, sum), (word_idx, &word)| {
                    (
                        count + word.count_ones() as usize,
                        sum + lookup.sum_word(word_idx, word),
                    )
                });
        let xor_score =
            score_cont_centered_gain_from_sum_and_n_hit(total_sum, xor_sum, n_samples, xor_n);
        let xnor_score =
            score_cont_centered_gain_from_sum_and_n_hit(total_sum, xnor_sum, n_samples, xnor_n);
        assert_eq!(xor_n + xnor_n, n_samples);
        assert!((xor_score.raw_score - xnor_score.raw_score).abs() < 1e-12);
    }

    fn reference_all_pairs(
        bits_flat: &[u64],
        row_words: usize,
        n_rows: usize,
        y: &[f64],
    ) -> Vec<AllPairCandidate> {
        let n_samples = y.len();
        let words = words_for_samples(n_samples).max(1);
        let tail = tail_mask(n_samples);
        let total_sum = y.iter().copied().sum::<f64>();
        let mut out = Vec::new();
        for first in 0..n_rows.saturating_sub(1) {
            for second in (first + 1)..n_rows {
                let first_row = row_slice(bits_flat, row_words, first, words);
                let second_row = row_slice(bits_flat, row_words, second, words);
                let mut best = None;
                for polarity in 0u8..4 {
                    let mut combined = vec![0u64; words];
                    for w in 0..words {
                        let lhs = if polarity & 1 != 0 {
                            !first_row[w]
                        } else {
                            first_row[w]
                        };
                        let rhs = if polarity & 2 != 0 {
                            !second_row[w]
                        } else {
                            second_row[w]
                        };
                        combined[w] = lhs & rhs;
                    }
                    if let Some(mask) = tail {
                        combined[words - 1] &= mask;
                    }
                    let support = combined.iter().map(|word| word.count_ones() as usize).sum();
                    let score = super::super::score::score_cont_centered_gain_packed_with_sum(
                        y, &combined, n_samples, total_sum,
                    );
                    let candidate = AllPairCandidate {
                        first,
                        second,
                        gate: AllPairGate::And,
                        polarity,
                        raw_score: score.raw_score,
                        support,
                    };
                    if best
                        .as_ref()
                        .map(|current| candidate_cmp(&candidate, current) == Ordering::Greater)
                        .unwrap_or(true)
                    {
                        best = Some(candidate);
                    }
                }
                out.push(best.expect("four pair polarities"));
            }
        }
        out.sort_unstable_by(|a, b| candidate_cmp(b, a));
        out
    }
}
