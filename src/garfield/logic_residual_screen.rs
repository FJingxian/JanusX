//! Packed four-cell sufficient statistics for residual screening.
//!
//! This module is intentionally independent from the Logic5 discovery path.
//! It will provide a GRAMMAR-like screen from an already residualized
//! phenotype, while exact GARFIELD/GLS inference remains a separate step.

use super::score::{validate_continuous_y, PackedYSumLookup};
use crate::blas::ensure_garfield_blas_backend;
use crate::breader::{
    bin01_shape, gather_rows_by_indices, load_bin01_as_u64_words,
    load_bin01_selected_rows_as_u64_words,
};
use crate::bstats::{tail_mask, words_for_samples};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;

// Keep the historical names private to this module while making the
// four-cell representation and all derived scores come from one shared
// primitive.
pub(crate) use super::four_cell::{
    derive_centered_product_screen_score, derive_gate5_gate6d_scores,
    derive_logic5_scores as derive_residual_gate_scores, FourCellGate,
    FourCellGateScore as ResidualGateScore, FourCellSufficientStats, Gate5Gate6dScores,
};

/// Proposal policy used by the residual screen. This is a proposal-layer
/// choice; four-cell summaries and downstream exact inference are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidualProposalMode {
    /// Historical best-of-four-ANDs plus optional XOR behavior.
    Max5,
    /// Max5 plus the standardized one-df four-cell dosage interaction gate.
    Max6,
    /// EXPERIMENTAL ONLY: Max5 plus the centered endpoint-product gate.
    ///
    /// This is the proposal-stage half of the experimental
    /// `nMax6 -> GRM-nMax6` variant. It is not the production Max6 screen and
    /// must not be selected implicitly by benchmark runners or mixed into the
    /// formal Max6 results. Keep this mode opt-in and report it separately.
    NMax6,
}

impl ResidualProposalMode {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "max5" | "max-5" | "baseline" => Ok(Self::Max5),
            "max6" | "max-6" | "logic6" | "standardized-contrast" => Ok(Self::Max6),
            "nmax6" | "nmax-6" | "unified6" | "centered-product" | "product6" => Ok(Self::NMax6),
            other => Err(format!(
                "proposal_mode must be one of max5, max6, or nmax6; got {other}"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Max5 => "max5",
            Self::Max6 => "max6",
            Self::NMax6 => "nmax6",
        }
    }
}

/// Inner sufficient-statistics kernel used by the residual screen.
///
/// `FourCell` is the historical reference implementation. `OneCell` is a
/// missing-free margin-reconstruction kernel: it computes only the 11 cell
/// per marker pair and recovers the other three cells from marker margins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidualScreenKernel {
    FourCell,
    OneCell,
}

impl ResidualScreenKernel {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "four_cell" | "four-cell" | "fourcell" | "legacy" | "reference" => Ok(Self::FourCell),
            "one_cell" | "one-cell" | "onecell" | "margin" => Ok(Self::OneCell),
            other => Err(format!(
                "screen_kernel must be one of four_cell or one_cell; got {other}"
            )),
        }
    }

    /// The margin kernel is valid only for a complete (missing-free) sample
    /// set. A supplied valid mask changes the four-cell universe per pair, so
    /// safely use the historical four-cell implementation in that case.
    fn effective_for_valid_mask(self, valid_mask_supplied: bool) -> Self {
        if self == Self::OneCell && valid_mask_supplied {
            Self::FourCell
        } else {
            self
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::FourCell => "four_cell",
            Self::OneCell => "one_cell",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct OneCellMargins {
    counts: Vec<usize>,
    sums: Vec<f64>,
    n_samples: usize,
    total_y: f64,
}

/// Marker-level one-cell margins that can be shared by several scan windows.
///
/// The table is deliberately indexed by an explicit sorted marker list rather
/// than by the whole genotype matrix.  This keeps the optimization cheap for
/// interval-restricted scans while still allowing a whole-genome caller to
/// provide one table for all active markers.
#[derive(Clone, Debug)]
pub(crate) struct OneCellMarginTable {
    marker_indices: Vec<usize>,
    counts: Vec<usize>,
    sums: Vec<f64>,
    n_samples: usize,
    total_y: f64,
}

impl OneCellMarginTable {
    pub(crate) fn build_for_rows(
        bits_flat: &[u64],
        row_words: usize,
        marker_indices: &[usize],
        residual: &[f64],
        n_samples: usize,
        lookup: &PackedYSumLookup,
    ) -> Result<Self, String> {
        Self::build_for_rows_with_pool(
            bits_flat,
            row_words,
            marker_indices,
            residual,
            n_samples,
            lookup,
            None,
        )
    }

    pub(crate) fn build_for_rows_with_pool(
        bits_flat: &[u64],
        row_words: usize,
        marker_indices: &[usize],
        residual: &[f64],
        n_samples: usize,
        lookup: &PackedYSumLookup,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<Self, String> {
        Self::build_for_rows_impl(
            bits_flat,
            row_words,
            marker_indices,
            residual,
            n_samples,
            lookup,
            pool,
            None,
        )
    }

    pub(crate) fn build_for_rows_with_pool_and_progress(
        bits_flat: &[u64],
        row_words: usize,
        marker_indices: &[usize],
        residual: &[f64],
        n_samples: usize,
        lookup: &PackedYSumLookup,
        pool: Option<&rayon::ThreadPool>,
        progress: &mut dyn FnMut(usize, usize) -> Result<(), String>,
    ) -> Result<Self, String> {
        Self::build_for_rows_impl(
            bits_flat,
            row_words,
            marker_indices,
            residual,
            n_samples,
            lookup,
            pool,
            Some(progress),
        )
    }

    fn build_for_rows_impl(
        bits_flat: &[u64],
        row_words: usize,
        marker_indices: &[usize],
        residual: &[f64],
        n_samples: usize,
        lookup: &PackedYSumLookup,
        pool: Option<&rayon::ThreadPool>,
        mut progress: Option<&mut dyn FnMut(usize, usize) -> Result<(), String>>,
    ) -> Result<Self, String> {
        if marker_indices.is_empty() {
            return Err("one-cell margin table requires at least one marker".to_string());
        }
        if marker_indices.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(
                "one-cell margin table marker indices must be sorted and unique".to_string(),
            );
        }
        if row_words == 0 {
            return Err("one-cell margin table requires row_words > 0".to_string());
        }
        let needed_words = words_for_samples(n_samples);
        if row_words < needed_words {
            return Err(format!(
                "one-cell margin table row_words={row_words} smaller than required {needed_words}"
            ));
        }
        validate_continuous_y(residual, n_samples, "one-cell margin table")?;
        let last_marker = *marker_indices.last().unwrap();
        let required = last_marker
            .checked_add(1)
            .and_then(|rows| rows.checked_mul(row_words))
            .ok_or_else(|| "one-cell margin table packed size overflow".to_string())?;
        if bits_flat.len() < required {
            return Err(format!(
                "one-cell margin table bits length={} smaller than required {required}",
                bits_flat.len()
            ));
        }
        if lookup.n_words() != needed_words {
            return Err(format!(
                "one-cell margin table lookup words={} does not match required {needed_words}",
                lookup.n_words()
            ));
        }

        let sample_tail = tail_mask(n_samples);
        let margin_for_marker = |&marker: &usize| {
            let row_start = marker * row_words;
            let mut count = 0usize;
            let mut sum = 0.0f64;
            for word_idx in 0..needed_words {
                let mut word = bits_flat[row_start + word_idx];
                if word_idx + 1 == needed_words {
                    if let Some(mask) = sample_tail {
                        word &= mask;
                    }
                }
                count = count.saturating_add(word.count_ones() as usize);
                sum += lookup.sum_word(word_idx, word);
            }
            (count, sum)
        };
        let margins = if let Some(progress) = progress.as_deref_mut() {
            let total = marker_indices.len();
            let chunk_size = (total / 200).max(1);
            let mut margins = Vec::with_capacity(total);
            progress(0, total)?;
            for chunk in marker_indices.chunks(chunk_size) {
                let chunk_margins = if let Some(pool) = pool {
                    pool.install(|| {
                        chunk
                            .par_iter()
                            .map(|marker| margin_for_marker(marker))
                            .collect::<Vec<_>>()
                    })
                } else {
                    chunk
                        .iter()
                        .map(|marker| margin_for_marker(marker))
                        .collect()
                };
                margins.extend(chunk_margins);
                progress(margins.len(), total)?;
            }
            margins
        } else if let Some(pool) = pool {
            pool.install(|| {
                marker_indices
                    .par_iter()
                    .map(margin_for_marker)
                    .collect::<Vec<_>>()
            })
        } else {
            marker_indices.iter().map(margin_for_marker).collect()
        };
        let (counts, sums): (Vec<_>, Vec<_>) = margins.into_iter().unzip();

        Ok(Self {
            marker_indices: marker_indices.to_vec(),
            counts,
            sums,
            n_samples,
            total_y: residual.iter().copied().sum::<f64>(),
        })
    }

    pub(crate) fn select(&self, marker_indices: &[usize]) -> Result<OneCellMargins, String> {
        let mut counts = Vec::with_capacity(marker_indices.len());
        let mut sums = Vec::with_capacity(marker_indices.len());
        for &marker in marker_indices {
            let local = self
                .marker_indices
                .binary_search(&marker)
                .map_err(|_| format!("one-cell margin table is missing marker {marker}"))?;
            counts.push(self.counts[local]);
            sums.push(self.sums[local]);
        }
        Ok(OneCellMargins {
            counts,
            sums,
            n_samples: self.n_samples,
            total_y: self.total_y,
        })
    }
}

/// Inputs for one draw in the exact Max6 null-batch scanner. The genotype
/// pair is traversed once for the whole batch; each draw retains its own
/// phenotype lookup and marker margins.
pub(crate) struct Max6ScreenNullDraw<'a> {
    pub(crate) residual: &'a [f64],
    pub(crate) lookup: &'a PackedYSumLookup,
    pub(crate) margins: &'a OneCellMarginTable,
}

/// Add one canonical pair's exact Max6 score to each draw's running maximum.
/// The accumulation order within each draw is intentionally the same as the
/// scalar one-cell screen: words are visited in sample order and each word
/// uses the existing adaptive weighted-sum kernel. `first_margin_slot` and
/// `second_margin_slot` index the shared, sorted active-marker table.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_max6_one_cell_pair_batch(
    first: &[u64],
    second: &[u64],
    n_samples: usize,
    first_margin_slot: usize,
    second_margin_slot: usize,
    pair_global_rows: Option<(usize, usize)>,
    draws: &[Max6ScreenNullDraw<'_>],
    sum11: &mut [f64],
    maxima: &mut [f64],
    mut maximizing_pairs: Option<&mut [Option<(usize, usize)>]>,
    mut pair_scores: Option<&mut [Option<ResidualGateScore>]>,
) -> Result<(), String> {
    if draws.is_empty()
        || draws.len() != sum11.len()
        || draws.len() != maxima.len()
        || maximizing_pairs
            .as_ref()
            .is_some_and(|pairs| pairs.len() != draws.len())
        || pair_scores
            .as_ref()
            .is_some_and(|scores| scores.len() != draws.len())
    {
        return Err(format!(
            "Max6 null batch shape mismatch: draws={}, sum11={}, maxima={}, argmax_pairs={}",
            draws.len(),
            sum11.len(),
            maxima.len(),
            maximizing_pairs.as_ref().map_or(0, |pairs| pairs.len()),
        ));
    }
    if maximizing_pairs.is_some() && pair_global_rows.is_none() {
        return Err("Max6 argmax tracking requires the current global pair identity".to_string());
    }
    if pair_global_rows.is_some_and(|(first, second)| first >= second) {
        return Err("Max6 argmax pair identity must be canonical (i < j)".to_string());
    }
    let needed_words = words_for_samples(n_samples);
    if first.len() < needed_words || second.len() < needed_words {
        return Err(format!(
            "Max6 null pair row shorter than required {needed_words} words"
        ));
    }
    let first_draw = &draws[0];
    if first_margin_slot >= first_draw.margins.counts.len()
        || second_margin_slot >= first_draw.margins.counts.len()
    {
        return Err("Max6 null margin slot is out of range".to_string());
    }
    if first_draw.margins.n_samples != n_samples {
        return Err(format!(
            "Max6 null margin sample count {} != {n_samples}",
            first_draw.margins.n_samples
        ));
    }
    for (draw_idx, draw) in draws.iter().enumerate() {
        if draw.residual.len() < n_samples
            || draw.lookup.n_words() != needed_words
            || draw.margins.n_samples != n_samples
            || draw.margins.counts.len() != first_draw.margins.counts.len()
            || first_margin_slot >= draw.margins.counts.len()
            || second_margin_slot >= draw.margins.counts.len()
        {
            return Err(format!(
                "Max6 null draw {draw_idx} is incompatible with the shared pair/margin layout"
            ));
        }
    }

    sum11.fill(0.0);
    let tail = tail_mask(n_samples);
    let mut n11 = 0usize;
    for word_idx in 0..needed_words {
        let mut intersection = first[word_idx] & second[word_idx];
        if word_idx + 1 == needed_words {
            if let Some(mask) = tail {
                intersection &= mask;
            }
        }
        n11 = n11.saturating_add(intersection.count_ones() as usize);
        for (draw_idx, draw) in draws.iter().enumerate() {
            sum11[draw_idx] += draw
                .lookup
                .sum_word_adaptive(draw.residual, word_idx, intersection);
        }
    }

    let n_first = first_draw.margins.counts[first_margin_slot];
    let n_second = first_draw.margins.counts[second_margin_slot];
    if n11 > n_first || n11 > n_second {
        return Err(format!(
            "Max6 null pair intersection count {n11} exceeds marker margins {n_first}/{n_second}"
        ));
    }
    let n10 = n_first - n11;
    let n01 = n_second - n11;
    let n_union = n_first
        .checked_add(n_second)
        .and_then(|value| value.checked_sub(n11))
        .ok_or_else(|| "Max6 null cell-count overflow".to_string())?;
    let n00 = n_samples
        .checked_sub(n_union)
        .ok_or_else(|| "Max6 null cell-count underflow".to_string())?;

    for (draw_idx, draw) in draws.iter().enumerate() {
        let s_first = draw.margins.sums[first_margin_slot];
        let s_second = draw.margins.sums[second_margin_slot];
        let s11 = sum11[draw_idx];
        let stats = FourCellSufficientStats {
            counts: [n00, n01, n10, n11],
            sums: [
                draw.margins.total_y - s_first - s_second + s11,
                s_second - s11,
                s_first - s11,
                s11,
            ],
            n_valid: n_samples,
        };
        if let Some(scores) = pair_scores.as_deref_mut() {
            scores[draw_idx] = None;
        }
        if let Some(best) = best_proposal_gate_fused(stats, ResidualProposalMode::Max6, true) {
            if let Some(scores) = pair_scores.as_deref_mut() {
                scores[draw_idx] = Some(best);
            }
            let order = best.score.total_cmp(&maxima[draw_idx]);
            let should_replace = if order.is_gt() {
                true
            } else if order.is_eq() {
                match (pair_global_rows, maximizing_pairs.as_deref()) {
                    (Some(pair), Some(pairs)) => {
                        pairs[draw_idx].is_none_or(|previous| pair < previous)
                    }
                    _ => false,
                }
            } else {
                false
            };
            if should_replace {
                maxima[draw_idx] = best.score;
                if let (Some(pair), Some(pairs)) =
                    (pair_global_rows, maximizing_pairs.as_deref_mut())
                {
                    pairs[draw_idx] = Some(pair);
                }
            }
        }
    }
    Ok(())
}

impl OneCellMargins {
    fn build(
        bits_flat: &[u64],
        row_words: usize,
        n_rows: usize,
        residual: &[f64],
        n_samples: usize,
        lookup: &PackedYSumLookup,
    ) -> Result<Self, String> {
        if n_rows == 0 {
            return Err("one-cell residual screen requires n_rows > 0".to_string());
        }
        let marker_indices = (0..n_rows).collect::<Vec<_>>();
        OneCellMarginTable::build_for_rows(
            bits_flat,
            row_words,
            marker_indices.as_slice(),
            residual,
            n_samples,
            lookup,
        )?
        .select(marker_indices.as_slice())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidualProposalSource {
    Max5,
    Max6,
    NMax6,
}

impl ResidualProposalSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Max5 => "max5",
            Self::Max6 => "max6",
            Self::NMax6 => "nmax6",
        }
    }

    fn rank(self) -> u8 {
        match self {
            Self::Max5 | Self::Max6 | Self::NMax6 => 0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn four_cell_sufficient_stats_packed(
    first: &[u64],
    second: &[u64],
    first_valid: Option<&[u64]>,
    second_valid: Option<&[u64]>,
    residual: &[f64],
    n_samples: usize,
    lookup: &PackedYSumLookup,
) -> Result<FourCellSufficientStats, String> {
    if n_samples == 0 {
        return Err("four-cell residual screen requires n_samples > 0".to_string());
    }
    validate_continuous_y(residual, n_samples, "four-cell residual screen")?;
    validate_packed_cell_rows(first, second, first_valid, second_valid, n_samples)?;
    Ok(four_cell_sufficient_stats_packed_unchecked(
        first,
        second,
        first_valid,
        second_valid,
        n_samples,
        lookup,
    ))
}

#[inline]
#[cfg(test)]
fn validate_packed_cell_rows(
    first: &[u64],
    second: &[u64],
    first_valid: Option<&[u64]>,
    second_valid: Option<&[u64]>,
    n_samples: usize,
) -> Result<(), String> {
    let words = words_for_samples(n_samples);
    if first.len() < words || second.len() < words {
        return Err(format!(
            "four-cell residual screen packed row is shorter than {words} words"
        ));
    }
    for (name, row) in [("first_valid", first_valid), ("second_valid", second_valid)] {
        if let Some(row) = row {
            if row.len() < words {
                return Err(format!(
                    "four-cell residual screen {name} row is shorter than {words} words"
                ));
            }
        }
    }
    Ok(())
}

/// Fast inner-loop form of [`four_cell_sufficient_stats_packed`].  The caller
/// must have validated the packed rows, sample count, residual, and lookup.
#[inline]
fn four_cell_sufficient_stats_packed_unchecked(
    first: &[u64],
    second: &[u64],
    first_valid: Option<&[u64]>,
    second_valid: Option<&[u64]>,
    n_samples: usize,
    lookup: &PackedYSumLookup,
) -> FourCellSufficientStats {
    debug_assert!(first.len() >= words_for_samples(n_samples));
    debug_assert!(second.len() >= words_for_samples(n_samples));
    debug_assert!(first_valid
        .map(|row| row.len() >= words_for_samples(n_samples))
        .unwrap_or(true));
    debug_assert!(second_valid
        .map(|row| row.len() >= words_for_samples(n_samples))
        .unwrap_or(true));
    let words = words_for_samples(n_samples);
    let tail = tail_mask(n_samples);
    let mut counts = [0usize; 4];
    let mut sums = [0.0f64; 4];
    for word_idx in 0..words {
        let x = first[word_idx];
        let z = second[word_idx];
        let mut valid = match (first_valid, second_valid) {
            (Some(lhs), Some(rhs)) => lhs[word_idx] & rhs[word_idx],
            (Some(lhs), None) => lhs[word_idx],
            (None, Some(rhs)) => rhs[word_idx],
            (None, None) => u64::MAX,
        };
        if word_idx + 1 == words {
            if let Some(mask) = tail {
                valid &= mask;
            }
        }
        let cells = [
            (!x & !z) & valid,
            (!x & z) & valid,
            (x & !z) & valid,
            (x & z) & valid,
        ];
        for (cell, &mask) in cells.iter().enumerate() {
            counts[cell] = counts[cell].saturating_add(mask.count_ones() as usize);
            sums[cell] += lookup.sum_word(word_idx, mask);
        }
    }
    FourCellSufficientStats {
        counts,
        sums,
        n_valid: counts.iter().sum(),
    }
}

/// Missing-free four-cell summaries with the optional validity branch removed.
///
/// This preserves the lookup and accumulation order of the general primitive
/// while avoiding `Option` matching and the temporary four-cell mask array in
/// the hottest pair/word loop.
#[inline]
fn four_cell_sufficient_stats_packed_no_valid(
    first: &[u64],
    second: &[u64],
    needed_words: usize,
    tail: Option<u64>,
    lookup: &PackedYSumLookup,
) -> FourCellSufficientStats {
    let mut counts = [0usize; 4];
    let mut sums = [0.0f64; 4];
    for word_idx in 0..needed_words {
        let x = first[word_idx];
        let z = second[word_idx];
        let mut valid = u64::MAX;
        if word_idx + 1 == needed_words {
            if let Some(mask) = tail {
                valid &= mask;
            }
        }
        let cell00 = (!x & !z) & valid;
        let cell01 = (!x & z) & valid;
        let cell10 = (x & !z) & valid;
        let cell11 = (x & z) & valid;
        counts[0] = counts[0].saturating_add(cell00.count_ones() as usize);
        counts[1] = counts[1].saturating_add(cell01.count_ones() as usize);
        counts[2] = counts[2].saturating_add(cell10.count_ones() as usize);
        counts[3] = counts[3].saturating_add(cell11.count_ones() as usize);
        sums[0] += lookup.sum_word(word_idx, cell00);
        sums[1] += lookup.sum_word(word_idx, cell01);
        sums[2] += lookup.sum_word(word_idx, cell10);
        sums[3] += lookup.sum_word(word_idx, cell11);
    }
    FourCellSufficientStats {
        counts,
        sums,
        n_valid: counts.iter().sum(),
    }
}

/// Missing-free one-cell/margin sufficient statistics.
///
/// Only the packed 11 intersection is scanned for the pair. The remaining
/// cell counts and weighted sums are reconstructed from the per-marker
/// margins in `margins`. The arithmetic is statistically exact, while its
/// floating-point accumulation order intentionally differs from the
/// four-cell reference path.
#[inline]
fn one_cell_sufficient_stats_packed_no_valid(
    first: &[u64],
    second: &[u64],
    needed_words: usize,
    tail: Option<u64>,
    y: &[f64],
    first_marker: usize,
    second_marker: usize,
    margins: &OneCellMargins,
    lookup: &PackedYSumLookup,
) -> FourCellSufficientStats {
    debug_assert!(first_marker < margins.counts.len());
    debug_assert!(second_marker < margins.counts.len());
    debug_assert!(first.len() >= needed_words);
    debug_assert!(second.len() >= needed_words);

    let mut n11 = 0usize;
    let mut s11 = 0.0f64;
    for word_idx in 0..needed_words {
        let mut intersection = first[word_idx] & second[word_idx];
        if word_idx + 1 == needed_words {
            if let Some(mask) = tail {
                intersection &= mask;
            }
        }
        n11 = n11.saturating_add(intersection.count_ones() as usize);
        s11 += lookup.sum_word_adaptive(y, word_idx, intersection);
    }

    let n_first = margins.counts[first_marker];
    let n_second = margins.counts[second_marker];
    debug_assert!(n11 <= n_first);
    debug_assert!(n11 <= n_second);
    let n10 = n_first - n11;
    let n01 = n_second - n11;
    let union = n_first
        .checked_add(n_second)
        .and_then(|value| value.checked_sub(n11))
        .expect("one-cell residual screen cell-count overflow");
    let n00 = margins
        .n_samples
        .checked_sub(union)
        .expect("one-cell residual screen cell-count underflow");
    let s_first = margins.sums[first_marker];
    let s_second = margins.sums[second_marker];
    let s10 = s_first - s11;
    let s01 = s_second - s11;
    let s00 = margins.total_y - s_first - s_second + s11;

    FourCellSufficientStats {
        counts: [n00, n01, n10, n11],
        sums: [s00, s01, s10, s11],
        n_valid: margins.n_samples,
    }
}

/// Experimental one-cell variant: skip empty intersections and reuse the
/// popcount already needed to recover n11 for the adaptive weighted lookup.
#[inline]
fn one_cell_sufficient_stats_packed_no_valid_skip_zero(
    first: &[u64],
    second: &[u64],
    needed_words: usize,
    tail: Option<u64>,
    y: &[f64],
    first_marker: usize,
    second_marker: usize,
    margins: &OneCellMargins,
    lookup: &PackedYSumLookup,
) -> FourCellSufficientStats {
    debug_assert!(first_marker < margins.counts.len());
    debug_assert!(second_marker < margins.counts.len());
    debug_assert!(first.len() >= needed_words);
    debug_assert!(second.len() >= needed_words);

    let mut n11 = 0usize;
    let mut s11 = 0.0f64;
    for word_idx in 0..needed_words {
        let mut intersection = first[word_idx] & second[word_idx];
        if word_idx + 1 == needed_words {
            if let Some(mask) = tail {
                intersection &= mask;
            }
        }
        let popcount = intersection.count_ones();
        n11 = n11.saturating_add(popcount as usize);
        if popcount != 0 {
            s11 += lookup.sum_word_adaptive_with_popcount(y, word_idx, intersection, popcount);
        }
    }

    let n_first = margins.counts[first_marker];
    let n_second = margins.counts[second_marker];
    debug_assert!(n11 <= n_first);
    debug_assert!(n11 <= n_second);
    let n10 = n_first - n11;
    let n01 = n_second - n11;
    let union = n_first
        .checked_add(n_second)
        .and_then(|value| value.checked_sub(n11))
        .expect("one-cell residual screen cell-count overflow");
    let n00 = margins
        .n_samples
        .checked_sub(union)
        .expect("one-cell residual screen cell-count underflow");
    let s_first = margins.sums[first_marker];
    let s_second = margins.sums[second_marker];
    let s10 = s_first - s11;
    let s01 = s_second - s11;
    let s00 = margins.total_y - s_first - s_second + s11;

    FourCellSufficientStats {
        counts: [n00, n01, n10, n11],
        sums: [s00, s01, s10, s11],
        n_valid: margins.n_samples,
    }
}

/// Return the exact feasible interval for the 11-cell count when all four
/// complete-case cells must contain at least `min_cell` samples.
#[inline]
fn one_cell_min_cell_n11_bounds(
    n_samples: usize,
    first_count: usize,
    second_count: usize,
    min_cell: usize,
) -> Option<(usize, usize)> {
    if first_count > n_samples || second_count > n_samples {
        return None;
    }
    if min_cell
        .checked_mul(4)
        .is_none_or(|minimum| n_samples < minimum)
    {
        return None;
    }
    let n = n_samples as i128;
    let a = first_count as i128;
    let b = second_count as i128;
    let m = min_cell as i128;
    let lower = m.max(a + b + m - n);
    let upper = (a - m).min(b - m);
    if lower > upper || lower < 0 || upper < 0 {
        return None;
    }
    Some((usize::try_from(lower).ok()?, usize::try_from(upper).ok()?))
}

/// Count only the 11 intersection and retain each word's popcount for the
/// accepted pair's weighted-sum pass. The scratch belongs to one scan worker.
#[inline]
fn packed_n11_min_cell_passes(
    first: &[u64],
    second: &[u64],
    n_samples: usize,
    lower: usize,
    upper: usize,
    popcounts: &mut [u8],
) -> bool {
    let needed_words = words_for_samples(n_samples);
    debug_assert!(first.len() >= needed_words);
    debug_assert!(second.len() >= needed_words);
    debug_assert!(popcounts.len() >= needed_words);
    let tail = tail_mask(n_samples);
    let mut n11 = 0usize;
    let mut processed = 0usize;
    for word_idx in 0..needed_words {
        let mut intersection = first[word_idx] & second[word_idx];
        let valid_bits = if word_idx + 1 == needed_words {
            if let Some(mask) = tail {
                intersection &= mask;
                mask.count_ones() as usize
            } else {
                64
            }
        } else {
            64
        };
        let popcount = intersection.count_ones();
        popcounts[word_idx] = popcount as u8;
        n11 = n11.saturating_add(popcount as usize);
        processed = processed.saturating_add(valid_bits);
        if n11 > upper || n11.saturating_add(n_samples.saturating_sub(processed)) < lower {
            return false;
        }
    }
    (lower..=upper).contains(&n11)
}

/// Four-word-unrolled MinCell gate. Threshold checks happen once per four
/// complete words; an optional partial tail is masked and checked afterward.
/// For short vectors, keep the scalar early-exit path.
#[inline]
fn packed_n11_min_cell_passes_unrolled4(
    first: &[u64],
    second: &[u64],
    n_samples: usize,
    lower: usize,
    upper: usize,
    popcounts: &mut [u8],
) -> bool {
    let needed_words = words_for_samples(n_samples);
    debug_assert!(first.len() >= needed_words);
    debug_assert!(second.len() >= needed_words);
    debug_assert!(popcounts.len() >= needed_words);
    let full_words = n_samples / 64;
    if full_words < 4 {
        return packed_n11_min_cell_passes(first, second, n_samples, lower, upper, popcounts);
    }

    let mut n11 = 0usize;
    let mut word_idx = 0usize;
    while word_idx + 4 <= full_words {
        let count0 = (first[word_idx] & second[word_idx]).count_ones() as u8;
        let count1 = (first[word_idx + 1] & second[word_idx + 1]).count_ones() as u8;
        let count2 = (first[word_idx + 2] & second[word_idx + 2]).count_ones() as u8;
        let count3 = (first[word_idx + 3] & second[word_idx + 3]).count_ones() as u8;
        popcounts[word_idx] = count0;
        popcounts[word_idx + 1] = count1;
        popcounts[word_idx + 2] = count2;
        popcounts[word_idx + 3] = count3;
        n11 = n11
            .saturating_add(count0 as usize + count1 as usize + count2 as usize + count3 as usize);
        word_idx += 4;
        let processed = word_idx * 64;
        if n11 > upper || n11.saturating_add(n_samples.saturating_sub(processed)) < lower {
            return false;
        }
    }

    while word_idx < full_words {
        let count = (first[word_idx] & second[word_idx]).count_ones();
        popcounts[word_idx] = count as u8;
        n11 = n11.saturating_add(count as usize);
        word_idx += 1;
    }
    if let Some(mask) = tail_mask(n_samples) {
        let last_word = full_words;
        let count = (first[last_word] & second[last_word] & mask).count_ones();
        popcounts[last_word] = count as u8;
        n11 = n11.saturating_add(count as usize);
    }
    (lower..=upper).contains(&n11)
}

#[inline]
fn one_cell_sufficient_stats_packed_no_valid_from_popcounts(
    first: &[u64],
    second: &[u64],
    needed_words: usize,
    tail: Option<u64>,
    y: &[f64],
    first_marker: usize,
    second_marker: usize,
    margins: &OneCellMargins,
    lookup: &PackedYSumLookup,
    popcounts: &[u8],
    skip_zero: bool,
) -> FourCellSufficientStats {
    debug_assert!(first_marker < margins.counts.len());
    debug_assert!(second_marker < margins.counts.len());
    debug_assert!(popcounts.len() >= needed_words);
    let mut n11 = 0usize;
    let mut s11 = 0.0f64;
    for word_idx in 0..needed_words {
        let mut intersection = first[word_idx] & second[word_idx];
        if word_idx + 1 == needed_words {
            if let Some(mask) = tail {
                intersection &= mask;
            }
        }
        let popcount = u32::from(popcounts[word_idx]);
        n11 = n11.saturating_add(popcount as usize);
        if !skip_zero || popcount != 0 {
            s11 += lookup.sum_word_adaptive_with_popcount(y, word_idx, intersection, popcount);
        }
    }

    let n_first = margins.counts[first_marker];
    let n_second = margins.counts[second_marker];
    debug_assert!(n11 <= n_first);
    debug_assert!(n11 <= n_second);
    let n10 = n_first - n11;
    let n01 = n_second - n11;
    let union = n_first
        .checked_add(n_second)
        .and_then(|value| value.checked_sub(n11))
        .expect("one-cell residual screen cell-count overflow");
    let n00 = margins
        .n_samples
        .checked_sub(union)
        .expect("one-cell residual screen cell-count underflow");
    let s_first = margins.sums[first_marker];
    let s_second = margins.sums[second_marker];
    FourCellSufficientStats {
        counts: [n00, n01, n10, n11],
        sums: [
            margins.total_y - s_first - s_second + s11,
            s_second - s11,
            s_first - s11,
            s11,
        ],
        n_valid: margins.n_samples,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResidualScreenCandidate {
    pub(crate) first: usize,
    pub(crate) second: usize,
    pub(crate) best_gate: ResidualGateScore,
    pub(crate) proposal_score: f64,
    pub(crate) proposal_source: ResidualProposalSource,
    pub(crate) stats: FourCellSufficientStats,
    pub(crate) derived: Gate5Gate6dScores,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResidualScreenResult {
    pub(crate) candidates: Vec<ResidualScreenCandidate>,
    pub(crate) pairs_evaluated: usize,
    pub(crate) pairs_scored: usize,
    pub(crate) n_rows: usize,
    pub(crate) n_samples: usize,
}

pub(crate) struct ResidualScreenBatchUnit<'a> {
    pub(crate) selected_global_rows: &'a [usize],
    pub(crate) pair_groups: Option<&'a [usize]>,
}

pub(crate) struct ResidualScreenBatchResult {
    pub(crate) units: Vec<ResidualScreenResult>,
    pub(crate) pairs_scored_unique: usize,
    pub(crate) score_offers: usize,
    pub(crate) used_direct_fallback: bool,
    pub(crate) timing: ResidualScreenBatchTiming,
}

/// Wall spans are disjoint. Heap reduction task time overlaps the scan span
/// and must not be subtracted from it or added to the wall-time total.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ResidualScreenBatchTiming {
    pub(crate) prepare_wall_seconds: f64,
    pub(crate) scan_merge_wall_seconds: f64,
    pub(crate) merge_task_seconds: f64,
    pub(crate) fold_task_count: usize,
    pub(crate) fold_task_wall_sum_seconds: f64,
    pub(crate) fold_task_wall_max_seconds: f64,
    pub(crate) fold_anchor_count_min: usize,
    pub(crate) fold_anchor_count_max: usize,
    pub(crate) fold_active_workers: usize,
    pub(crate) worker_task_wall_min_seconds: f64,
    pub(crate) worker_task_wall_median_seconds: f64,
    pub(crate) worker_task_wall_max_seconds: f64,
    pub(crate) worker_finish_skew_seconds: f64,
    pub(crate) post_last_fold_tail_seconds: f64,
    pub(crate) topk_merge_task_seconds: f64,
    pub(crate) topk_merge_max_task_seconds: f64,
    pub(crate) topk_merge_task_count: usize,
    pub(crate) topk_merge_critical_path_seconds: f64,
    pub(crate) scan_reduce_critical_path_seconds: f64,
    pub(crate) finalize_wall_seconds: f64,
    pub(crate) source_rows: usize,
    pub(crate) memberships: usize,
    pub(crate) parallel: bool,
}

#[inline]
fn reduce_critical_path_seconds(left: f64, right: f64, merge: f64) -> f64 {
    left.max(right) + merge
}

#[inline]
fn best_gate_score(
    scores: &[ResidualGateScore; 5],
    include_xor: bool,
) -> Option<ResidualGateScore> {
    // Logic5 historically evaluates positive-first AND11, then AND01,
    // AND10, AND00.  Retain that priority for exact ties even though the
    // sufficient-statistics arrays are stored in natural cell order
    // [00, 01, 10, 11].
    let order = if include_xor {
        [3usize, 1, 2, 0, 4]
    } else {
        [3usize, 1, 2, 0, usize::MAX]
    };
    // Keep the first gate on exact ties.  This makes the derived gate label
    // deterministic and preserves the historical Logic5 ordering (AND gates
    // before XOR) without changing the score used for ranking.
    let mut best = None;
    for &gate_idx in order.iter().take(if include_xor { 5 } else { 4 }) {
        let score = scores[gate_idx];
        if !score.score.is_finite() {
            continue;
        }
        if best
            .map(|current: ResidualGateScore| score.score > current.score)
            .unwrap_or(true)
        {
            best = Some(score);
        }
    }
    best
}

#[inline]
fn best_gate6_score(
    scores: &[ResidualGateScore; 5],
    dosage_score: Option<f64>,
    contrast: Option<f64>,
    include_xor: bool,
) -> Option<ResidualGateScore> {
    let dosage_signed_score = dosage_score
        .filter(|score| score.is_finite())
        .and_then(|score| {
            contrast
                .filter(|value| value.is_finite())
                .map(|value| value.signum() * score.sqrt())
        });
    best_gate6_score_with_signed(scores, dosage_score, dosage_signed_score, include_xor)
}

#[inline]
fn best_gate6_score_with_signed(
    scores: &[ResidualGateScore; 5],
    dosage_score: Option<f64>,
    dosage_signed_score: Option<f64>,
    include_xor: bool,
) -> Option<ResidualGateScore> {
    let mut best = best_gate_score(scores, include_xor)?;
    let dosage_score = dosage_score.filter(|score| score.is_finite());
    if let Some(score) = dosage_score {
        if score > best.score {
            best = ResidualGateScore {
                gate: super::four_cell::FourCellGate::DosageInteraction,
                score,
                signed_score: dosage_signed_score.unwrap_or(f64::NAN),
                n_hit: 0,
            };
        }
    }
    Some(best)
}

#[inline]
fn derive_proposal_scores(
    stats: FourCellSufficientStats,
    proposal_mode: ResidualProposalMode,
) -> Gate5Gate6dScores {
    match proposal_mode {
        ResidualProposalMode::Max5 => Gate5Gate6dScores {
            logic5: derive_residual_gate_scores(stats),
            dosage_interaction_contrast: None,
            dosage_interaction_standardized_score: None,
            dosage_product_score: None,
            dosage_product_signed_score: None,
        },
        ResidualProposalMode::Max6 => derive_gate5_gate6d_scores(stats),
        ResidualProposalMode::NMax6 => {
            let (dosage_product_score, dosage_product_signed_score) =
                derive_centered_product_screen_score(stats)
                    .map(|(score, signed)| (Some(score), Some(signed)))
                    .unwrap_or((None, None));
            Gate5Gate6dScores {
                logic5: derive_residual_gate_scores(stats),
                dosage_interaction_contrast: None,
                dosage_interaction_standardized_score: None,
                dosage_product_score,
                dosage_product_signed_score,
            }
        }
    }
}

/// Fused proposal-stage score selection for the hot all-pair loop.
///
/// The materialized `Gate5Gate6dScores` form is still used when a finalist is
/// rebuilt, because the result needs all channel details.  During the full
/// scan we only need the winning proposal gate, so compute the same formulas
/// directly and avoid constructing the intermediate five-element score
/// array/struct on every pair.
#[inline(always)]
fn fused_logic_gate_score(
    gate: FourCellGate,
    n_hit: usize,
    sum_hit: f64,
    total: f64,
    n_total: f64,
    n_valid: usize,
) -> Option<ResidualGateScore> {
    if n_valid == 0 || n_hit == 0 || n_hit >= n_valid {
        return None;
    }
    let n_miss = n_valid - n_hit;
    let hit_f = n_hit as f64;
    let miss_f = n_miss as f64;
    let denominator = hit_f * miss_f / n_total;
    if !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let centered_sum = sum_hit - (n_hit as f64 / n_total) * total;
    let signed_score = centered_sum / denominator.sqrt();
    let score = n_total * centered_sum * centered_sum / (hit_f * miss_f);
    Some(ResidualGateScore {
        gate,
        score,
        signed_score,
        n_hit,
    })
}

#[inline(always)]
fn fused_standardized_dosage_score(stats: FourCellSufficientStats) -> Option<(f64, f64)> {
    if stats.n_valid == 0 || stats.counts.iter().any(|&count| count == 0) {
        return None;
    }
    let means = [
        stats.sums[0] / stats.counts[0] as f64,
        stats.sums[1] / stats.counts[1] as f64,
        stats.sums[2] / stats.counts[2] as f64,
        stats.sums[3] / stats.counts[3] as f64,
    ];
    let contrast = means[3] - means[2] - means[1] + means[0];
    let inverse_count_sum = 1.0 / stats.counts[0] as f64
        + 1.0 / stats.counts[1] as f64
        + 1.0 / stats.counts[2] as f64
        + 1.0 / stats.counts[3] as f64;
    if !contrast.is_finite() || !inverse_count_sum.is_finite() || inverse_count_sum <= 0.0 {
        return None;
    }
    let score = contrast * contrast / inverse_count_sum;
    score
        .is_finite()
        .then_some((score, contrast.signum() * score.sqrt()))
}

#[inline(always)]
fn best_max6_gate_fused(
    stats: FourCellSufficientStats,
    include_xor: bool,
) -> Option<ResidualGateScore> {
    let total = stats.sums.iter().sum::<f64>();
    let n_total = stats.n_valid as f64;
    let mut best = None;

    macro_rules! consider_gate {
        ($gate:expr, $n_hit:expr, $sum_hit:expr) => {
            if let Some(candidate) =
                fused_logic_gate_score($gate, $n_hit, $sum_hit, total, n_total, stats.n_valid)
            {
                if best
                    .map(|current: ResidualGateScore| candidate.score > current.score)
                    .unwrap_or(true)
                {
                    best = Some(candidate);
                }
            }
        };
    }

    // Preserve the historical tie priority: AND11, AND01, AND10, AND00,
    // then XOR.  The macro expansion keeps the hot Max6 path branch-light
    // while retaining the exact score arithmetic of the reference loop.
    consider_gate!(FourCellGate::And11, stats.counts[3], stats.sums[3]);
    consider_gate!(FourCellGate::And01, stats.counts[1], stats.sums[1]);
    consider_gate!(FourCellGate::And10, stats.counts[2], stats.sums[2]);
    consider_gate!(FourCellGate::And00, stats.counts[0], stats.sums[0]);
    if include_xor {
        consider_gate!(
            FourCellGate::Xor,
            stats.counts[1].saturating_add(stats.counts[2]),
            stats.sums[1] + stats.sums[2]
        );
    }

    let mut best = best?;
    if let Some((dosage_score, dosage_signed_score)) = fused_standardized_dosage_score(stats) {
        if dosage_score > best.score {
            best = ResidualGateScore {
                gate: FourCellGate::DosageInteraction,
                score: dosage_score,
                signed_score: dosage_signed_score,
                n_hit: 0,
            };
        }
    }
    Some(best)
}

#[inline]
fn best_proposal_gate_fused(
    stats: FourCellSufficientStats,
    proposal_mode: ResidualProposalMode,
    include_xor: bool,
) -> Option<ResidualGateScore> {
    if proposal_mode == ResidualProposalMode::Max6 {
        return best_max6_gate_fused(stats, include_xor);
    }
    let total = stats.sums.iter().sum::<f64>();
    let n_total = stats.n_valid as f64;
    let gate_order = if include_xor {
        [3usize, 1, 2, 0, 4]
    } else {
        [3usize, 1, 2, 0, usize::MAX]
    };
    let gate_count = if include_xor { 5 } else { 4 };
    let mut best = None;
    for &gate_index in gate_order.iter().take(gate_count) {
        let (gate, n_hit, sum_hit) = match gate_index {
            0 => (FourCellGate::And00, stats.counts[0], stats.sums[0]),
            1 => (FourCellGate::And01, stats.counts[1], stats.sums[1]),
            2 => (FourCellGate::And10, stats.counts[2], stats.sums[2]),
            3 => (FourCellGate::And11, stats.counts[3], stats.sums[3]),
            4 => (
                FourCellGate::Xor,
                stats.counts[1].saturating_add(stats.counts[2]),
                stats.sums[1] + stats.sums[2],
            ),
            _ => continue,
        };
        if stats.n_valid == 0 || n_hit == 0 || n_hit >= stats.n_valid {
            continue;
        }
        let n_miss = stats.n_valid - n_hit;
        let hit_f = n_hit as f64;
        let miss_f = n_miss as f64;
        let denominator = hit_f * miss_f / n_total;
        if !denominator.is_finite() || denominator <= 0.0 {
            continue;
        }
        let centered_sum = sum_hit - (n_hit as f64 / n_total) * total;
        let signed_score = centered_sum / denominator.sqrt();
        let score = n_total * centered_sum * centered_sum / (hit_f * miss_f);
        let candidate = ResidualGateScore {
            gate,
            score,
            signed_score,
            n_hit,
        };
        if best
            .map(|current: ResidualGateScore| candidate.score > current.score)
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }
    let mut best = best?;
    match proposal_mode {
        ResidualProposalMode::Max5 => Some(best),
        ResidualProposalMode::Max6 => unreachable!("Max6 uses the specialized fused path"),
        ResidualProposalMode::NMax6 => {
            if let Some((dosage_score, dosage_signed_score)) =
                derive_centered_product_screen_score(stats)
            {
                if dosage_score > best.score {
                    best = ResidualGateScore {
                        gate: FourCellGate::DosageInteraction,
                        score: dosage_score,
                        signed_score: dosage_signed_score,
                        n_hit: 0,
                    };
                }
            }
            Some(best)
        }
    }
}

#[inline]
fn best_gate_for_proposal(
    derived: &Gate5Gate6dScores,
    proposal_mode: ResidualProposalMode,
    include_xor: bool,
) -> Option<ResidualGateScore> {
    match proposal_mode {
        ResidualProposalMode::Max5 => best_gate_score(&derived.logic5, include_xor),
        ResidualProposalMode::Max6 => best_gate6_score(
            &derived.logic5,
            derived.dosage_interaction_standardized_score,
            derived.dosage_interaction_contrast,
            include_xor,
        ),
        ResidualProposalMode::NMax6 => best_gate6_score_with_signed(
            &derived.logic5,
            derived.dosage_product_score,
            derived.dosage_product_signed_score,
            include_xor,
        ),
    }
}

#[inline]
fn proposal_cmp(
    lhs_score: f64,
    lhs_first: usize,
    lhs_second: usize,
    lhs_source: ResidualProposalSource,
    rhs_score: f64,
    rhs_first: usize,
    rhs_second: usize,
    rhs_source: ResidualProposalSource,
) -> Ordering {
    lhs_score
        .total_cmp(&rhs_score)
        .then_with(|| rhs_first.cmp(&lhs_first))
        .then_with(|| rhs_second.cmp(&lhs_second))
        .then_with(|| rhs_source.rank().cmp(&lhs_source.rank()))
}

#[inline]
fn candidate_cmp(lhs: &ResidualScreenCandidate, rhs: &ResidualScreenCandidate) -> Ordering {
    proposal_cmp(
        lhs.proposal_score,
        lhs.first,
        lhs.second,
        lhs.proposal_source,
        rhs.proposal_score,
        rhs.first,
        rhs.second,
        rhs.proposal_source,
    )
}

#[derive(Clone, Copy, Debug)]
struct CandidateHeapEntry {
    first: usize,
    second: usize,
    best_gate: ResidualGateScore,
    proposal_score: f64,
    proposal_source: ResidualProposalSource,
}

impl PartialEq for CandidateHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.first == other.first
            && self.second == other.second
            && self.best_gate.gate == other.best_gate.gate
            && self.best_gate.score.to_bits() == other.best_gate.score.to_bits()
            && self.proposal_score.to_bits() == other.proposal_score.to_bits()
            && self.proposal_source == other.proposal_source
    }
}

impl Eq for CandidateHeapEntry {}

impl Ord for CandidateHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        proposal_cmp(
            self.proposal_score,
            self.first,
            self.second,
            self.proposal_source,
            other.proposal_score,
            other.first,
            other.second,
            other.proposal_source,
        )
    }
}

impl PartialOrd for CandidateHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn push_top_k(
    heap: &mut BinaryHeap<Reverse<CandidateHeapEntry>>,
    candidate: CandidateHeapEntry,
    top_k: usize,
) {
    if top_k == 0 {
        return;
    }
    let item = Reverse(candidate);
    if heap.len() < top_k {
        heap.push(item);
    } else if heap
        .peek()
        .map(|worst| item.0.cmp(&worst.0) == Ordering::Greater)
        .unwrap_or(true)
    {
        let _ = heap.pop();
        heap.push(item);
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ResidualScreenProfile {
    pub(crate) sample_pairs: usize,
    pub(crate) total_pairs: usize,
    pub(crate) sample_words: usize,
    pub(crate) nonzero_intersections: usize,
    pub(crate) intersection_popcount_sum: usize,
    pub(crate) lookup_calls: usize,
    pub(crate) lut8_word_calls: usize,
    pub(crate) setbit_calls: usize,
    pub(crate) bitset_seconds: f64,
    pub(crate) weighted_y_lookup_seconds: f64,
    pub(crate) margin_recovery_seconds: f64,
    pub(crate) score_seconds: f64,
    pub(crate) heap_seconds: f64,
    pub(crate) final_sort_seconds: f64,
    pub(crate) heap_pushes: usize,
    pub(crate) heap_replacements: usize,
    pub(crate) sample_pairs_scored: usize,
    pub(crate) score_checksum: f64,
}

/// Profile the current one-cell implementation on a bounded sample of pairs.
///
/// This is intentionally a diagnostic-only second pass. It separates the
/// packed intersection, continuous phenotype lookup, margin reconstruction,
/// score, and Top-K phases without putting an `Instant::now()` call inside the
/// production pair loop. The normal scanner and its statistical definition are
/// untouched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn profile_one_cell_sample(
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
    profile_pairs: usize,
    residual: &[f64],
    margins: &OneCellMargins,
    lookup: &PackedYSumLookup,
) -> Result<ResidualScreenProfile, String> {
    if n_rows < 2 || profile_pairs == 0 {
        return Ok(ResidualScreenProfile {
            total_pairs: n_rows.saturating_mul(n_rows.saturating_sub(1)) / 2,
            ..ResidualScreenProfile::default()
        });
    }

    let needed_words = words_for_samples(n_samples);
    let sample_tail = tail_mask(n_samples);
    let total_pairs = n_rows
        .checked_mul(n_rows.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| "residual screen profile pair counter overflow".to_string())?;
    let sample_pairs = profile_pairs.min(total_pairs);
    let mut pairs = Vec::with_capacity(sample_pairs);
    'outer: for first in 0..n_rows.saturating_sub(1) {
        for second in (first + 1)..n_rows {
            pairs.push((first, second));
            if pairs.len() == sample_pairs {
                break 'outer;
            }
        }
    }

    let mut intersections = vec![0u64; sample_pairs.saturating_mul(needed_words)];
    let mut n11_values = vec![0usize; sample_pairs];
    let mut s11_values = vec![0.0f64; sample_pairs];
    let mut stats = vec![
        FourCellSufficientStats {
            counts: [0; 4],
            sums: [0.0; 4],
            n_valid: n_samples,
        };
        sample_pairs
    ];
    let mut profile = ResidualScreenProfile {
        sample_pairs,
        total_pairs,
        sample_words: sample_pairs.saturating_mul(needed_words),
        ..ResidualScreenProfile::default()
    };

    let bitset_started = Instant::now();
    for (pair_index, &(first, second)) in pairs.iter().enumerate() {
        let first_row = &bits_flat[first * row_words..first * row_words + needed_words];
        let second_row = &bits_flat[second * row_words..second * row_words + needed_words];
        let mut n11 = 0usize;
        for word_idx in 0..needed_words {
            let mut intersection = first_row[word_idx] & second_row[word_idx];
            if word_idx + 1 == needed_words {
                if let Some(mask) = sample_tail {
                    intersection &= mask;
                }
            }
            intersections[pair_index * needed_words + word_idx] = intersection;
            let popcount = intersection.count_ones() as usize;
            n11 = n11.saturating_add(popcount);
            profile.intersection_popcount_sum =
                profile.intersection_popcount_sum.saturating_add(popcount);
            if intersection != 0 {
                profile.nonzero_intersections = profile.nonzero_intersections.saturating_add(1);
            }
        }
        n11_values[pair_index] = n11;
    }
    profile.bitset_seconds = bitset_started.elapsed().as_secs_f64();

    let lookup_started = Instant::now();
    for pair_index in 0..sample_pairs {
        let mut s11 = 0.0f64;
        for word_idx in 0..needed_words {
            let word = intersections[pair_index * needed_words + word_idx];
            if PackedYSumLookup::adaptive_uses_setbit(word) {
                profile.setbit_calls = profile.setbit_calls.saturating_add(1);
            } else {
                profile.lut8_word_calls = profile.lut8_word_calls.saturating_add(1);
            }
            s11 += lookup.sum_word_adaptive(residual, word_idx, word);
            profile.lookup_calls = profile.lookup_calls.saturating_add(1);
        }
        s11_values[pair_index] = s11;
    }
    profile.weighted_y_lookup_seconds = lookup_started.elapsed().as_secs_f64();

    let margin_started = Instant::now();
    for (pair_index, &(first, second)) in pairs.iter().enumerate() {
        let n11 = n11_values[pair_index];
        let n_first = margins.counts[first];
        let n_second = margins.counts[second];
        let n10 = n_first - n11;
        let n01 = n_second - n11;
        let n00 = margins.n_samples - n_first - n_second + n11;
        let s11 = s11_values[pair_index];
        let s_first = margins.sums[first];
        let s_second = margins.sums[second];
        stats[pair_index] = FourCellSufficientStats {
            counts: [n00, n01, n10, n11],
            sums: [
                margins.total_y - s_first - s_second + s11,
                s_second - s11,
                s_first - s11,
                s11,
            ],
            n_valid: margins.n_samples,
        };
    }
    profile.margin_recovery_seconds = margin_started.elapsed().as_secs_f64();

    let score_started = Instant::now();
    let mut entries = Vec::with_capacity(sample_pairs);
    for (pair_index, &(first, second)) in pairs.iter().enumerate() {
        let derived = derive_proposal_scores(stats[pair_index], ResidualProposalMode::Max6);
        let best_gate = best_gate_for_proposal(&derived, ResidualProposalMode::Max6, include_xor);
        if let Some(best_gate) = best_gate {
            profile.sample_pairs_scored = profile.sample_pairs_scored.saturating_add(1);
            profile.score_checksum += best_gate.score;
            entries.push(CandidateHeapEntry {
                first,
                second,
                best_gate,
                proposal_score: best_gate.score,
                proposal_source: ResidualProposalSource::Max6,
            });
        }
    }
    profile.score_seconds = score_started.elapsed().as_secs_f64();

    let heap_started = Instant::now();
    let mut heap = BinaryHeap::<Reverse<CandidateHeapEntry>>::with_capacity(top_k.min(1024));
    for entry in entries {
        if top_k == 0 {
            continue;
        }
        let item = Reverse(entry);
        if heap.len() < top_k {
            heap.push(item);
            profile.heap_pushes = profile.heap_pushes.saturating_add(1);
        } else if heap
            .peek()
            .map(|worst| item.0.cmp(&worst.0) == Ordering::Greater)
            .unwrap_or(true)
        {
            let _ = heap.pop();
            heap.push(item);
            profile.heap_pushes = profile.heap_pushes.saturating_add(1);
            profile.heap_replacements = profile.heap_replacements.saturating_add(1);
        }
    }
    profile.heap_seconds = heap_started.elapsed().as_secs_f64();

    let mut top_entries = heap
        .into_iter()
        .map(|Reverse(entry)| entry)
        .collect::<Vec<_>>();
    let sort_started = Instant::now();
    top_entries.sort_unstable_by(|lhs, rhs| {
        proposal_cmp(
            rhs.proposal_score,
            rhs.first,
            rhs.second,
            rhs.proposal_source,
            lhs.proposal_score,
            lhs.first,
            lhs.second,
            lhs.proposal_source,
        )
    });
    profile.final_sort_seconds = sort_started.elapsed().as_secs_f64();
    Ok(profile)
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn scan_residual_pairs_packed(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
) -> Result<ResidualScreenResult, String> {
    scan_residual_pairs_packed_with_parallel(
        bits_flat,
        valid_flat,
        row_words,
        n_rows,
        residual,
        n_samples,
        top_k,
        include_xor,
        true,
    )
}

/// Internal variant used by callers that already parallelize over windows.
/// Standalone Python calls use Rayon; nested callers can pass `false` to keep
/// one level of thread-local Top-K heaps.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn scan_residual_pairs_packed_with_parallel(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
    allow_parallel: bool,
) -> Result<ResidualScreenResult, String> {
    scan_residual_pairs_packed_with_mode(
        bits_flat,
        valid_flat,
        row_words,
        n_rows,
        residual,
        n_samples,
        top_k,
        include_xor,
        ResidualProposalMode::Max5,
        allow_parallel,
    )
}

/// Internal scanner with an explicit Gate5/Gate6d proposal policy.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn scan_residual_pairs_packed_with_mode(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
    proposal_mode: ResidualProposalMode,
    allow_parallel: bool,
) -> Result<ResidualScreenResult, String> {
    scan_residual_pairs_packed_with_mode_and_kernel(
        bits_flat,
        valid_flat,
        row_words,
        n_rows,
        residual,
        n_samples,
        top_k,
        include_xor,
        proposal_mode,
        ResidualScreenKernel::FourCell,
        allow_parallel,
    )
}

/// Internal scanner with explicit proposal and sufficient-statistics kernels.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_residual_pairs_packed_with_mode_and_kernel(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
    proposal_mode: ResidualProposalMode,
    screen_kernel: ResidualScreenKernel,
    allow_parallel: bool,
) -> Result<ResidualScreenResult, String> {
    let lookup = PackedYSumLookup::build(residual, n_samples)?;
    let one_cell_margins = if screen_kernel == ResidualScreenKernel::OneCell {
        Some(OneCellMargins::build(
            bits_flat, row_words, n_rows, residual, n_samples, &lookup,
        )?)
    } else {
        None
    };
    scan_residual_pairs_packed_with_mode_and_kernel_with_context(
        bits_flat,
        valid_flat,
        row_words,
        n_rows,
        residual,
        n_samples,
        top_k,
        include_xor,
        proposal_mode,
        screen_kernel,
        &lookup,
        one_cell_margins,
        allow_parallel,
    )
}

/// Scanner entry point with reusable phenotype lookup/margin context.
///
/// The caller owns the lifetime of `lookup` and may reuse it across windows
/// for the same residualized phenotype.  For the one-cell kernel,
/// `one_cell_margins` must be in the same row order as `bits_flat`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_residual_pairs_packed_with_mode_and_kernel_with_context(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
    proposal_mode: ResidualProposalMode,
    screen_kernel: ResidualScreenKernel,
    lookup: &PackedYSumLookup,
    one_cell_margins: Option<OneCellMargins>,
    allow_parallel: bool,
) -> Result<ResidualScreenResult, String> {
    scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups(
        bits_flat,
        valid_flat,
        row_words,
        n_rows,
        residual,
        n_samples,
        top_k,
        include_xor,
        proposal_mode,
        screen_kernel,
        lookup,
        one_cell_margins,
        None,
        allow_parallel,
    )
}

/// Scanner entry point with an optional pair-group filter.
///
/// When `pair_groups` is supplied, pairs whose two rows have the same group
/// id are skipped before any cell statistics are computed. This is used for
/// multi-window genesets; permutation pools deliberately do not use it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
    proposal_mode: ResidualProposalMode,
    screen_kernel: ResidualScreenKernel,
    lookup: &PackedYSumLookup,
    one_cell_margins: Option<OneCellMargins>,
    pair_groups: Option<&[usize]>,
    allow_parallel: bool,
) -> Result<ResidualScreenResult, String> {
    if proposal_mode == ResidualProposalMode::Max6 {
        ensure_garfield_blas_backend()?;
    }
    scan_residual_pairs_packed_with_mode_options(
        bits_flat,
        valid_flat,
        row_words,
        n_rows,
        residual,
        n_samples,
        top_k,
        include_xor,
        proposal_mode,
        screen_kernel,
        lookup,
        one_cell_margins,
        pair_groups,
        0,
        PairTraversalMode::CrossGroupOnly,
        None,
        allow_parallel,
        None,
        None,
    )
}

/// Thresholded screen entry point used by the Max6 operational null gate.
///
/// When `score_threshold` is supplied, `top_k` is only retained as a legacy
/// API placeholder: every finite pair whose proposal score is strictly
/// greater than the threshold is returned.  This is deliberately different
/// from a Top-K proposal, because a fixed Top-K cap could discard a pair that
/// is significant under the independently calibrated screen threshold.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
    proposal_mode: ResidualProposalMode,
    screen_kernel: ResidualScreenKernel,
    lookup: &PackedYSumLookup,
    one_cell_margins: Option<OneCellMargins>,
    pair_groups: Option<&[usize]>,
    min_cell: usize,
    score_threshold: Option<f64>,
    allow_parallel: bool,
) -> Result<ResidualScreenResult, String> {
    scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold_impl(
        bits_flat,
        valid_flat,
        row_words,
        n_rows,
        residual,
        n_samples,
        top_k,
        include_xor,
        proposal_mode,
        screen_kernel,
        lookup,
        one_cell_margins,
        pair_groups,
        min_cell,
        score_threshold,
        allow_parallel,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold_with_progress(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
    proposal_mode: ResidualProposalMode,
    screen_kernel: ResidualScreenKernel,
    lookup: &PackedYSumLookup,
    one_cell_margins: Option<OneCellMargins>,
    pair_groups: Option<&[usize]>,
    min_cell: usize,
    score_threshold: Option<f64>,
    allow_parallel: bool,
    progress: &mut dyn FnMut(usize, usize) -> Result<(), String>,
) -> Result<ResidualScreenResult, String> {
    scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold_impl(
        bits_flat,
        valid_flat,
        row_words,
        n_rows,
        residual,
        n_samples,
        top_k,
        include_xor,
        proposal_mode,
        screen_kernel,
        lookup,
        one_cell_margins,
        pair_groups,
        min_cell,
        score_threshold,
        allow_parallel,
        Some(progress),
    )
}

#[allow(clippy::too_many_arguments)]
fn scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold_impl(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
    proposal_mode: ResidualProposalMode,
    screen_kernel: ResidualScreenKernel,
    lookup: &PackedYSumLookup,
    one_cell_margins: Option<OneCellMargins>,
    pair_groups: Option<&[usize]>,
    min_cell: usize,
    score_threshold: Option<f64>,
    allow_parallel: bool,
    progress: Option<&mut dyn FnMut(usize, usize) -> Result<(), String>>,
) -> Result<ResidualScreenResult, String> {
    scan_residual_pairs_packed_with_mode_options(
        bits_flat,
        valid_flat,
        row_words,
        n_rows,
        residual,
        n_samples,
        top_k,
        include_xor,
        proposal_mode,
        screen_kernel,
        lookup,
        one_cell_margins,
        pair_groups,
        min_cell,
        PairTraversalMode::CrossGroupOnly,
        score_threshold,
        allow_parallel,
        progress,
        None,
    )
}

/// Build sorted local-row membership lists for a multi-window geneset.
///
/// The scanner uses these lists to enumerate only cross-group pairs.  Each
/// membership list is sorted because rows are appended in local-row order;
/// that lets the caller skip all members whose index is not greater than the
/// current anchor without materializing a second pair list.
fn build_sorted_group_members(groups: &[usize]) -> Vec<(usize, Vec<usize>)> {
    let mut by_group = HashMap::<usize, Vec<usize>>::new();
    for (row, &group) in groups.iter().enumerate() {
        by_group.entry(group).or_default().push(row);
    }
    let mut members = by_group.into_iter().collect::<Vec<_>>();
    members.sort_unstable_by_key(|(group, _)| *group);
    members
}

/// Visit each cross-group row whose local index is greater than `first`.
///
/// This is deliberately callback-based: the scanner avoids allocating an
/// `O(number_of_cross_pairs)` index vector while still skipping all within-
/// group pairs before entering the packed sufficient-statistics kernel.
fn for_each_cross_group_second(
    group_members: &[(usize, Vec<usize>)],
    groups: &[usize],
    first: usize,
    mut visit: impl FnMut(usize),
) {
    let first_group = groups[first];
    for &(group, ref members) in group_members.iter() {
        if group == first_group {
            continue;
        }
        let start = members.partition_point(|&second| second <= first);
        for &second in members[start..].iter() {
            visit(second);
        }
    }
}

fn packed_pair_cells_meet_min(
    first: &[u64],
    second: &[u64],
    first_valid: Option<&[u64]>,
    second_valid: Option<&[u64]>,
    n_samples: usize,
    min_cell: usize,
) -> bool {
    let mut counts = [0usize; 4];
    let needed_words = words_for_samples(n_samples);
    let mut processed = 0usize;
    for word in 0..needed_words {
        let tail = if word + 1 == needed_words {
            tail_mask(n_samples).unwrap_or(u64::MAX)
        } else {
            u64::MAX
        };
        let valid = first_valid.map(|bits| bits[word]).unwrap_or(u64::MAX)
            & second_valid.map(|bits| bits[word]).unwrap_or(u64::MAX)
            & tail;
        let first_bits = first[word] & valid;
        let second_bits = second[word] & valid;
        counts[0] =
            counts[0].saturating_add((!first_bits & !second_bits & valid).count_ones() as usize);
        counts[1] =
            counts[1].saturating_add((!first_bits & second_bits & valid).count_ones() as usize);
        counts[2] =
            counts[2].saturating_add((first_bits & !second_bits & valid).count_ones() as usize);
        counts[3] =
            counts[3].saturating_add((first_bits & second_bits & valid).count_ones() as usize);
        processed = processed.saturating_add(valid.count_ones() as usize);
        let remaining = n_samples.saturating_sub(processed);
        if counts
            .iter()
            .any(|&count| count.saturating_add(remaining) < min_cell)
        {
            return false;
        }
    }
    counts.iter().all(|&count| count >= min_cell)
}

#[cfg(test)]
fn cross_group_second_indices(
    group_members: &[(usize, Vec<usize>)],
    groups: &[usize],
    first: usize,
) -> Vec<usize> {
    let mut seconds = Vec::new();
    for_each_cross_group_second(group_members, groups, first, |second| seconds.push(second));
    seconds.sort_unstable();
    seconds
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PairTraversalMode {
    LegacyGroupFilter,
    CrossGroupOnly,
}

/// Common Gate5/Gate6d scanner implementation.
#[allow(clippy::too_many_arguments)]
fn scan_residual_pairs_packed_with_mode_options(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
    proposal_mode: ResidualProposalMode,
    screen_kernel: ResidualScreenKernel,
    lookup: &PackedYSumLookup,
    one_cell_margins: Option<OneCellMargins>,
    pair_groups: Option<&[usize]>,
    min_cell: usize,
    pair_traversal: PairTraversalMode,
    score_threshold: Option<f64>,
    allow_parallel: bool,
    mut progress: Option<&mut dyn FnMut(usize, usize) -> Result<(), String>>,
    pair_scan_seconds: Option<&mut f64>,
) -> Result<ResidualScreenResult, String> {
    if n_rows < 2 {
        return Err("four-cell residual screen requires at least two markers".to_string());
    }
    let needed_words = words_for_samples(n_samples);
    let sample_tail = tail_mask(n_samples);
    if row_words < needed_words {
        return Err(format!(
            "four-cell residual screen row_words={row_words} smaller than required {needed_words}"
        ));
    }
    let required = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| "four-cell residual screen packed size overflow".to_string())?;
    if bits_flat.len() < required {
        return Err(format!(
            "four-cell residual screen bits length={} smaller than required {required}",
            bits_flat.len()
        ));
    }
    if let Some(valid) = valid_flat {
        if valid.len() < required {
            return Err(format!(
                "four-cell residual screen valid bits length={} smaller than required {required}",
                valid.len()
            ));
        }
    }
    validate_continuous_y(residual, n_samples, "four-cell residual screen")?;
    if lookup.n_words() != needed_words {
        return Err(format!(
            "four-cell residual screen lookup words={} does not match required {needed_words}",
            lookup.n_words()
        ));
    }
    if score_threshold.is_some_and(|threshold| !threshold.is_finite()) {
        return Err("four-cell residual screen threshold must be finite".to_string());
    }
    if screen_kernel == ResidualScreenKernel::OneCell && valid_flat.is_some() {
        return Err(
            "one-cell residual screen requires valid_bits=None; use four_cell for masked data"
                .to_string(),
        );
    }
    if screen_kernel == ResidualScreenKernel::OneCell
        && one_cell_margins
            .as_ref()
            .is_none_or(|margins| margins.counts.len() != n_rows)
    {
        return Err(
            "one-cell residual screen requires margins for every packed marker row".to_string(),
        );
    }
    if let Some(groups) = pair_groups {
        if groups.len() != n_rows {
            return Err(format!(
                "four-cell residual screen pair-group length={} does not match n_rows={n_rows}",
                groups.len()
            ));
        }
    }
    let heap_capacity = top_k.min(1024);
    let first_rows_end = n_rows.saturating_sub(1);
    let proposal_source = match proposal_mode {
        ResidualProposalMode::Max5 => ResidualProposalSource::Max5,
        ResidualProposalMode::Max6 => ResidualProposalSource::Max6,
        ResidualProposalMode::NMax6 => ResidualProposalSource::NMax6,
    };
    let one_cell_skip_zero = std::env::var("JX_GARFIELD_EXPERIMENTAL_SCREEN_ZERO_SKIP")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        })
        .unwrap_or(false);
    let group_members = pair_groups.map(build_sorted_group_members);
    let use_cross_group_only = pair_traversal == PairTraversalMode::CrossGroupOnly;
    let use_margin_min_cell =
        screen_kernel == ResidualScreenKernel::OneCell && valid_flat.is_none() && min_cell > 0;
    let score_pair = |first: usize,
                      first_row: &[u64],
                      first_valid: Option<&[u64]>,
                      second: usize,
                      popcount_scratch: &mut [u8]|
     -> Option<CandidateHeapEntry> {
        let second_start = second * row_words;
        let second_row = &bits_flat[second_start..second_start + needed_words];
        let second_valid =
            valid_flat.map(|valid| &valid[second_start..second_start + needed_words]);
        let one_cell_stats = if use_margin_min_cell {
            let margins = one_cell_margins
                .as_ref()
                .expect("one-cell margins must be initialized");
            let (lower, upper) = one_cell_min_cell_n11_bounds(
                n_samples,
                margins.counts[first],
                margins.counts[second],
                min_cell,
            )?;
            if !packed_n11_min_cell_passes_unrolled4(
                first_row,
                second_row,
                n_samples,
                lower,
                upper,
                popcount_scratch,
            ) {
                return None;
            }
            Some(one_cell_sufficient_stats_packed_no_valid_from_popcounts(
                first_row,
                second_row,
                needed_words,
                sample_tail,
                residual,
                first,
                second,
                margins,
                lookup,
                popcount_scratch,
                one_cell_skip_zero,
            ))
        } else {
            None
        };
        if !use_margin_min_cell
            && min_cell > 0
            && !packed_pair_cells_meet_min(
                first_row,
                second_row,
                first_valid,
                second_valid,
                n_samples,
                min_cell,
            )
        {
            return None;
        }
        let stats = if let Some(stats) = one_cell_stats {
            stats
        } else {
            match screen_kernel {
                ResidualScreenKernel::OneCell => {
                    let margins = one_cell_margins
                        .as_ref()
                        .expect("one-cell margins must be initialized");
                    if one_cell_skip_zero {
                        one_cell_sufficient_stats_packed_no_valid_skip_zero(
                            first_row,
                            second_row,
                            needed_words,
                            sample_tail,
                            residual,
                            first,
                            second,
                            margins,
                            lookup,
                        )
                    } else {
                        one_cell_sufficient_stats_packed_no_valid(
                            first_row,
                            second_row,
                            needed_words,
                            sample_tail,
                            residual,
                            first,
                            second,
                            margins,
                            lookup,
                        )
                    }
                }
                ResidualScreenKernel::FourCell => {
                    if valid_flat.is_none() {
                        four_cell_sufficient_stats_packed_no_valid(
                            first_row,
                            second_row,
                            needed_words,
                            sample_tail,
                            lookup,
                        )
                    } else {
                        four_cell_sufficient_stats_packed_unchecked(
                            first_row,
                            second_row,
                            first_valid,
                            second_valid,
                            n_samples,
                            lookup,
                        )
                    }
                }
            }
        };
        let best_gate = best_proposal_gate_fused(stats, proposal_mode, include_xor)?;
        Some(CandidateHeapEntry {
            first,
            second,
            best_gate,
            proposal_score: best_gate.score,
            proposal_source,
        })
    };
    let scan_first = |first: usize,
                      mut heap: BinaryHeap<Reverse<CandidateHeapEntry>>,
                      mut pairs_scored: usize,
                      popcount_scratch: &mut [u8]|
     -> (BinaryHeap<Reverse<CandidateHeapEntry>>, usize) {
        let first_start = first * row_words;
        let first_row = &bits_flat[first_start..first_start + needed_words];
        let first_valid = valid_flat.map(|valid| &valid[first_start..first_start + needed_words]);
        let mut score_second = |second| {
            if let Some(entry) = score_pair(first, first_row, first_valid, second, popcount_scratch)
            {
                pairs_scored = pairs_scored.saturating_add(1);
                push_top_k(&mut heap, entry, top_k);
            }
        };
        if use_cross_group_only {
            if let (Some(groups), Some(group_members)) = (pair_groups, group_members.as_ref()) {
                for_each_cross_group_second(group_members, groups, first, &mut score_second);
            } else {
                for second in (first + 1)..n_rows {
                    score_second(second);
                }
            }
        } else {
            for second in (first + 1)..n_rows {
                if let Some(groups) = pair_groups {
                    if groups[first] == groups[second] {
                        continue;
                    }
                }
                score_second(second);
            }
        }
        (heap, pairs_scored)
    };
    let empty_heap = || BinaryHeap::<Reverse<CandidateHeapEntry>>::with_capacity(heap_capacity);
    let all_pairs = n_rows
        .checked_mul(n_rows.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| "four-cell residual screen pair counter overflow".to_string())?;
    let pairs_evaluated = if let Some(groups) = pair_groups {
        let group_members = group_members
            .as_ref()
            .expect("pair-group membership must be built when groups are supplied");
        let within_group = group_members
            .iter()
            .map(|(_, members)| {
                members
                    .len()
                    .saturating_mul(members.len().saturating_sub(1))
                    / 2
            })
            .sum::<usize>();
        debug_assert_eq!(groups.len(), n_rows);
        all_pairs.saturating_sub(within_group)
    } else {
        all_pairs
    };
    let pair_scan_started = pair_scan_seconds.as_ref().map(|_| Instant::now());
    let (entries, pairs_scored) =
        if score_threshold.is_some() || (top_k > 0 && top_k >= pairs_evaluated) {
            // When K covers the entire window, a heap only inserts every pair
            // twice (once into a local heap and once into the global heap). Keep a
            // flat exact entry vector and sort finalists once below.
            let mut entries = Vec::with_capacity(pairs_evaluated);
            let mut pairs_scored = 0usize;
            let progress_total = pairs_evaluated.max(1);
            let progress_chunk = (pairs_evaluated / 200).max(1);
            let mut next_progress = progress_chunk;
            let mut progress_pairs = 0usize;
            let mut popcount_scratch = vec![0u8; needed_words];
            if let Some(callback) = progress.as_deref_mut() {
                callback(0, progress_total)?;
            }
            for first in 0..first_rows_end {
                let first_start = first * row_words;
                let first_row = &bits_flat[first_start..first_start + needed_words];
                let first_valid =
                    valid_flat.map(|valid| &valid[first_start..first_start + needed_words]);
                let row_pairs = {
                    let mut row_pairs = 0usize;
                    let mut score_second = |second| {
                        row_pairs = row_pairs.saturating_add(1);
                        if let Some(entry) = score_pair(
                            first,
                            first_row,
                            first_valid,
                            second,
                            popcount_scratch.as_mut_slice(),
                        ) {
                            pairs_scored = pairs_scored.saturating_add(1);
                            if score_threshold
                                .map(|threshold| entry.proposal_score > threshold)
                                .unwrap_or(true)
                            {
                                entries.push(entry);
                            }
                        }
                    };
                    if use_cross_group_only {
                        if let (Some(groups), Some(group_members)) =
                            (pair_groups, group_members.as_ref())
                        {
                            for_each_cross_group_second(
                                group_members,
                                groups,
                                first,
                                &mut score_second,
                            );
                        } else {
                            for second in (first + 1)..n_rows {
                                score_second(second);
                            }
                        }
                    } else {
                        for second in (first + 1)..n_rows {
                            if let Some(groups) = pair_groups {
                                if groups[first] == groups[second] {
                                    continue;
                                }
                            }
                            score_second(second);
                        }
                    }
                    row_pairs
                };
                progress_pairs = progress_pairs.saturating_add(row_pairs);
                if let Some(callback) = progress.as_deref_mut() {
                    if progress_pairs >= next_progress || first + 1 == first_rows_end {
                        callback(progress_pairs, progress_total)?;
                        while next_progress <= progress_pairs {
                            next_progress = next_progress.saturating_add(progress_chunk);
                        }
                    }
                }
            }
            (entries, pairs_scored)
        } else {
            let first_rows = 0..first_rows_end;
            let (heap, pairs_scored) =
                if allow_parallel && rayon::current_num_threads() > 1 && n_rows >= 32 {
                    let (heap, pairs_scored, _) = first_rows
                        .into_par_iter()
                        .fold(
                            || (empty_heap(), 0usize, vec![0u8; needed_words]),
                            |(heap, pairs_scored, mut scratch), first| {
                                let (heap, pairs_scored) =
                                    scan_first(first, heap, pairs_scored, scratch.as_mut_slice());
                                (heap, pairs_scored, scratch)
                            },
                        )
                        .reduce(
                            || (empty_heap(), 0usize, Vec::new()),
                            |(mut left_heap, left_scored, _), (right_heap, right_scored, _)| {
                                for Reverse(entry) in right_heap {
                                    push_top_k(&mut left_heap, entry, top_k);
                                }
                                (
                                    left_heap,
                                    left_scored.saturating_add(right_scored),
                                    Vec::new(),
                                )
                            },
                        );
                    (heap, pairs_scored)
                } else {
                    let mut serial_heap = empty_heap();
                    let mut serial_scored = 0usize;
                    let mut scratch = vec![0u8; needed_words];
                    for first in first_rows {
                        let (row_heap, row_scored) =
                            scan_first(first, empty_heap(), 0, scratch.as_mut_slice());
                        for Reverse(entry) in row_heap {
                            push_top_k(&mut serial_heap, entry, top_k);
                        }
                        serial_scored = serial_scored.saturating_add(row_scored);
                    }
                    (serial_heap, serial_scored)
                };
            (
                heap.into_iter()
                    .map(|Reverse(entry)| entry)
                    .collect::<Vec<_>>(),
                pairs_scored,
            )
        };
    if let (Some(elapsed_out), Some(started)) = (pair_scan_seconds, pair_scan_started) {
        *elapsed_out = started.elapsed().as_secs_f64();
    }
    let rebuild_candidate = |entry: CandidateHeapEntry| {
        let first_start = entry.first * row_words;
        let second_start = entry.second * row_words;
        let first_row = &bits_flat[first_start..first_start + needed_words];
        let second_row = &bits_flat[second_start..second_start + needed_words];
        let stats = match screen_kernel {
            ResidualScreenKernel::OneCell => one_cell_sufficient_stats_packed_no_valid(
                first_row,
                second_row,
                needed_words,
                sample_tail,
                residual,
                entry.first,
                entry.second,
                one_cell_margins
                    .as_ref()
                    .expect("one-cell margins must be initialized"),
                &lookup,
            ),
            ResidualScreenKernel::FourCell => {
                if valid_flat.is_none() {
                    four_cell_sufficient_stats_packed_no_valid(
                        first_row,
                        second_row,
                        needed_words,
                        sample_tail,
                        &lookup,
                    )
                } else {
                    let first_valid =
                        valid_flat.map(|valid| &valid[first_start..first_start + needed_words]);
                    let second_valid =
                        valid_flat.map(|valid| &valid[second_start..second_start + needed_words]);
                    four_cell_sufficient_stats_packed_unchecked(
                        first_row,
                        second_row,
                        first_valid,
                        second_valid,
                        n_samples,
                        &lookup,
                    )
                }
            }
        };
        let derived = derive_proposal_scores(stats, proposal_mode);
        ResidualScreenCandidate {
            first: entry.first,
            second: entry.second,
            best_gate: entry.best_gate,
            proposal_score: entry.proposal_score,
            proposal_source: entry.proposal_source,
            stats,
            derived,
        }
    };
    let mut candidates = entries
        .into_iter()
        .map(rebuild_candidate)
        .collect::<Vec<_>>();
    candidates.sort_unstable_by(|lhs, rhs| candidate_cmp(rhs, lhs));
    Ok(ResidualScreenResult {
        candidates,
        pairs_evaluated,
        pairs_scored,
        n_rows,
        n_samples,
    })
}

#[derive(Clone, Copy)]
struct ResidualBatchMembershipRecord {
    global_row: usize,
    margin_slot: usize,
    unit_index: usize,
    local_index: usize,
    group: Option<usize>,
}

#[derive(Clone, Copy)]
struct ResidualBatchMembership {
    unit_index: usize,
    local_index: usize,
    group: Option<usize>,
}

struct ResidualBatchSourceRow {
    global_row: usize,
    margin_slot: usize,
    membership_start: usize,
    membership_end: usize,
}

struct ResidualBatchPartnerCursor {
    unit_index: usize,
    anchor_group: Option<usize>,
    next_local_index: usize,
    current_local_index: usize,
}

#[derive(Clone, Copy)]
struct ResidualBatchRecipient {
    unit_index: usize,
    second_local_index: usize,
}

#[derive(Clone, Copy, Debug)]
struct ResidualBatchFoldTiming {
    worker_index: usize,
    anchors: usize,
    wall_seconds: f64,
    finish_offset_seconds: f64,
}

struct ResidualBatchWorkerState {
    merge_task_seconds: f64,
    fold_started_at: Option<Instant>,
    fold_worker_index: Option<usize>,
    fold_anchor_count: usize,
    fold_timings: Vec<ResidualBatchFoldTiming>,
    topk_merge_task_seconds: f64,
    topk_merge_max_task_seconds: f64,
    topk_merge_task_count: usize,
    topk_merge_critical_path_seconds: f64,
    scan_reduce_critical_path_seconds: f64,
    heaps: Vec<BinaryHeap<Reverse<CandidateHeapEntry>>>,
    pairs_evaluated: Vec<usize>,
    pairs_scored: Vec<usize>,
    pairs_scored_unique: usize,
    score_offers: usize,
    popcounts: Vec<u8>,
    cursors: Vec<ResidualBatchPartnerCursor>,
    cursor_heap: BinaryHeap<Reverse<(usize, usize)>>,
    recipients: Vec<ResidualBatchRecipient>,
}

impl ResidualBatchWorkerState {
    fn new(n_units: usize, top_k: usize, needed_words: usize) -> Self {
        let heap_capacity = top_k.min(1024);
        Self {
            merge_task_seconds: 0.0,
            fold_started_at: None,
            fold_worker_index: None,
            fold_anchor_count: 0,
            fold_timings: Vec::new(),
            topk_merge_task_seconds: 0.0,
            topk_merge_max_task_seconds: 0.0,
            topk_merge_task_count: 0,
            topk_merge_critical_path_seconds: 0.0,
            scan_reduce_critical_path_seconds: 0.0,
            heaps: (0..n_units)
                .map(|_| BinaryHeap::with_capacity(heap_capacity))
                .collect(),
            pairs_evaluated: vec![0; n_units],
            pairs_scored: vec![0; n_units],
            pairs_scored_unique: 0,
            score_offers: 0,
            popcounts: vec![0; needed_words],
            cursors: Vec::with_capacity(n_units),
            cursor_heap: BinaryHeap::with_capacity(n_units),
            recipients: Vec::with_capacity(n_units),
        }
    }
}

fn residual_batch_advance_partner(
    units: &[ResidualScreenBatchUnit<'_>],
    anchor_global_row: usize,
    cursor: &mut ResidualBatchPartnerCursor,
) -> Option<usize> {
    let rows = units[cursor.unit_index].selected_global_rows;
    while cursor.next_local_index < rows.len() {
        let local_index = cursor.next_local_index;
        cursor.next_local_index += 1;
        let global_row = rows[local_index];
        if global_row <= anchor_global_row {
            continue;
        }
        if let Some(groups) = units[cursor.unit_index].pair_groups {
            if Some(groups[local_index]) == cursor.anchor_group {
                continue;
            }
        }
        cursor.current_local_index = local_index;
        return Some(global_row);
    }
    None
}

/// Whether two units share at least one pair that is eligible in both units.
/// Common rows are visited once; group counts avoid a quadratic pair walk.
fn residual_batch_units_share_eligible_pair(
    left: &ResidualScreenBatchUnit<'_>,
    right: &ResidualScreenBatchUnit<'_>,
) -> bool {
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    let mut common_seen = 0usize;
    let mut left_group_counts = HashMap::<usize, usize>::new();
    let mut right_group_counts = HashMap::<usize, usize>::new();
    let mut group_pair_counts = HashMap::<(usize, usize), usize>::new();

    while left_index < left.selected_global_rows.len()
        && right_index < right.selected_global_rows.len()
    {
        let left_row = left.selected_global_rows[left_index];
        let right_row = right.selected_global_rows[right_index];
        if left_row < right_row {
            left_index += 1;
            continue;
        }
        if right_row < left_row {
            right_index += 1;
            continue;
        }

        let left_group = left.pair_groups.map(|groups| groups[left_index]);
        let right_group = right.pair_groups.map(|groups| groups[right_index]);
        let same_left_count = left_group
            .and_then(|group| left_group_counts.get(&group).copied())
            .unwrap_or(0);
        let same_right_count = right_group
            .and_then(|group| right_group_counts.get(&group).copied())
            .unwrap_or(0);
        let same_both_count = match (left_group, right_group) {
            (Some(left_group), Some(right_group)) => group_pair_counts
                .get(&(left_group, right_group))
                .copied()
                .unwrap_or(0),
            _ => 0,
        };
        let excluded_previous = same_left_count
            .saturating_add(same_right_count)
            .saturating_sub(same_both_count);
        if common_seen > excluded_previous {
            return true;
        }

        common_seen += 1;
        if let Some(group) = left_group {
            *left_group_counts.entry(group).or_default() += 1;
        }
        if let Some(group) = right_group {
            *right_group_counts.entry(group).or_default() += 1;
        }
        if let (Some(left_group), Some(right_group)) = (left_group, right_group) {
            *group_pair_counts
                .entry((left_group, right_group))
                .or_default() += 1;
        }
        left_index += 1;
        right_index += 1;
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn residual_batch_scan_anchor(
    anchor_index: usize,
    bits_flat: &[u64],
    row_words: usize,
    units: &[ResidualScreenBatchUnit<'_>],
    margin_slots_by_unit: &[Vec<usize>],
    source_rows: &[ResidualBatchSourceRow],
    memberships: &[ResidualBatchMembership],
    margins: &OneCellMargins,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    min_cell: usize,
    include_xor: bool,
    lookup: &PackedYSumLookup,
    state: &mut ResidualBatchWorkerState,
) {
    let anchor = &source_rows[anchor_index];
    let anchor_row_start = anchor.global_row * row_words;
    let anchor_row = &bits_flat[anchor_row_start..anchor_row_start + words_for_samples(n_samples)];
    state.cursors.clear();
    state.cursor_heap.clear();
    state.recipients.clear();

    for member in &memberships[anchor.membership_start..anchor.membership_end] {
        let mut cursor = ResidualBatchPartnerCursor {
            unit_index: member.unit_index,
            anchor_group: member.group,
            next_local_index: member.local_index + 1,
            current_local_index: member.local_index,
        };
        if let Some(partner_global_row) =
            residual_batch_advance_partner(units, anchor.global_row, &mut cursor)
        {
            let cursor_index = state.cursors.len();
            state.cursors.push(cursor);
            state
                .cursor_heap
                .push(Reverse((partner_global_row, cursor_index)));
        }
    }

    while let Some(Reverse((partner_global_row, _))) = state.cursor_heap.peek().copied() {
        state.recipients.clear();
        while let Some(Reverse((current_partner, cursor_index))) = state.cursor_heap.peek().copied()
        {
            if current_partner != partner_global_row {
                break;
            }
            state.cursor_heap.pop();
            let cursor = &mut state.cursors[cursor_index];
            state.recipients.push(ResidualBatchRecipient {
                unit_index: cursor.unit_index,
                second_local_index: cursor.current_local_index,
            });
            if let Some(next_global_row) =
                residual_batch_advance_partner(units, anchor.global_row, cursor)
            {
                state
                    .cursor_heap
                    .push(Reverse((next_global_row, cursor_index)));
            }
        }

        for recipient in state.recipients.iter().copied() {
            state.pairs_evaluated[recipient.unit_index] =
                state.pairs_evaluated[recipient.unit_index].saturating_add(1);
        }
        let first_margin_slot = anchor.margin_slot;
        let first_count = margins.counts[first_margin_slot];
        let first_local = state.recipients[0].second_local_index;
        let first_unit = state.recipients[0].unit_index;
        let second_margin_slot = margin_slots_by_unit[first_unit][first_local];
        let second_count = margins.counts[second_margin_slot];
        let partner_row_start = partner_global_row * row_words;
        let partner_row =
            &bits_flat[partner_row_start..partner_row_start + words_for_samples(n_samples)];

        if min_cell > 0 {
            let min_cell_passes =
                one_cell_min_cell_n11_bounds(n_samples, first_count, second_count, min_cell)
                    .is_some_and(|(lower, upper)| {
                        packed_n11_min_cell_passes_unrolled4(
                            anchor_row,
                            partner_row,
                            n_samples,
                            lower,
                            upper,
                            state.popcounts.as_mut_slice(),
                        )
                    });
            if !min_cell_passes {
                continue;
            }
        }
        let stats = if min_cell > 0 {
            one_cell_sufficient_stats_packed_no_valid_from_popcounts(
                anchor_row,
                partner_row,
                words_for_samples(n_samples),
                tail_mask(n_samples),
                residual,
                first_margin_slot,
                second_margin_slot,
                margins,
                lookup,
                state.popcounts.as_slice(),
                false,
            )
        } else {
            one_cell_sufficient_stats_packed_no_valid(
                anchor_row,
                partner_row,
                words_for_samples(n_samples),
                tail_mask(n_samples),
                residual,
                first_margin_slot,
                second_margin_slot,
                margins,
                lookup,
            )
        };
        let Some(best_gate) =
            best_proposal_gate_fused(stats, ResidualProposalMode::Max6, include_xor)
        else {
            continue;
        };
        let entry = CandidateHeapEntry {
            first: anchor.global_row,
            second: partner_global_row,
            best_gate,
            proposal_score: best_gate.score,
            proposal_source: ResidualProposalSource::Max6,
        };
        state.pairs_scored_unique = state.pairs_scored_unique.saturating_add(1);
        state.score_offers = state.score_offers.saturating_add(state.recipients.len());
        for recipient in state.recipients.iter().copied() {
            state.pairs_scored[recipient.unit_index] =
                state.pairs_scored[recipient.unit_index].saturating_add(1);
            push_top_k(&mut state.heaps[recipient.unit_index], entry, top_k);
        }
    }
}

/// Screen a bounded batch of units by merging their actual sorted LD-tag
/// memberships. Shared eligible pairs are scored once and offered to every
/// unit-local heap; batches without a shared eligible pair use the direct
/// per-unit scanner.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_residual_pairs_packed_max6_batch(
    bits_flat: &[u64],
    row_words: usize,
    units: &[ResidualScreenBatchUnit<'_>],
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    min_cell: usize,
    lookup: &PackedYSumLookup,
    margin_table: &OneCellMarginTable,
    allow_parallel: bool,
) -> Result<ResidualScreenBatchResult, String> {
    let min_anchor_len = std::env::var("JX_GARFIELD_MAX6_FOLD_MIN_ANCHORS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1);
    scan_residual_pairs_packed_max6_batch_with_min_anchor_len(
        bits_flat,
        row_words,
        units,
        residual,
        n_samples,
        top_k,
        min_cell,
        lookup,
        margin_table,
        allow_parallel,
        min_anchor_len,
    )
}

#[allow(clippy::too_many_arguments)]
fn scan_residual_pairs_packed_max6_batch_with_min_anchor_len(
    bits_flat: &[u64],
    row_words: usize,
    units: &[ResidualScreenBatchUnit<'_>],
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    min_cell: usize,
    lookup: &PackedYSumLookup,
    margin_table: &OneCellMarginTable,
    allow_parallel: bool,
    min_anchor_len: usize,
) -> Result<ResidualScreenBatchResult, String> {
    let prepare_started = Instant::now();
    if row_words == 0 {
        return Err("Max6 batch screen row_words must be greater than zero".to_string());
    }
    let needed_words = words_for_samples(n_samples);
    if row_words < needed_words {
        return Err(format!(
            "Max6 batch screen row_words={row_words} smaller than required {needed_words}"
        ));
    }
    validate_continuous_y(residual, n_samples, "Max6 batch screen")?;
    if lookup.n_words() != needed_words {
        return Err(format!(
            "Max6 batch screen lookup words={} does not match required {needed_words}",
            lookup.n_words()
        ));
    }
    if margin_table.n_samples != n_samples {
        return Err(format!(
            "Max6 batch screen margin samples={} does not match n_samples={n_samples}",
            margin_table.n_samples
        ));
    }
    let packed_rows = bits_flat.len() / row_words;
    let membership_count = units
        .iter()
        .try_fold(0usize, |count, unit| {
            count.checked_add(unit.selected_global_rows.len())
        })
        .ok_or_else(|| "Max6 batch screen membership count overflow".to_string())?;
    if units.iter().any(|unit| {
        unit.selected_global_rows
            .windows(2)
            .any(|rows| rows[0] >= rows[1])
            || unit
                .selected_global_rows
                .last()
                .is_some_and(|&row| row >= packed_rows)
            || unit
                .pair_groups
                .is_some_and(|groups| groups.len() != unit.selected_global_rows.len())
    }) {
        return Err("Max6 batch screen unit rows/groups are invalid or unsorted".to_string());
    }
    if units.is_empty() {
        return Ok(ResidualScreenBatchResult {
            units: Vec::new(),
            pairs_scored_unique: 0,
            score_offers: 0,
            used_direct_fallback: true,
            timing: ResidualScreenBatchTiming::default(),
        });
    }
    ensure_garfield_blas_backend()?;

    let has_shared_eligible_pair = (0..units.len()).any(|left| {
        ((left + 1)..units.len())
            .any(|right| residual_batch_units_share_eligible_pair(&units[left], &units[right]))
    });
    if !has_shared_eligible_pair {
        let prepare_wall_seconds = prepare_started.elapsed().as_secs_f64();
        let scan_started = Instant::now();
        let scan_one =
            |unit: &ResidualScreenBatchUnit<'_>| -> Result<ResidualScreenResult, String> {
                let n_rows = unit.selected_global_rows.len();
                if n_rows < 2 {
                    return Ok(ResidualScreenResult {
                        candidates: Vec::new(),
                        pairs_evaluated: 0,
                        pairs_scored: 0,
                        n_rows,
                        n_samples,
                    });
                }
                let packed = gather_rows_by_indices(
                    bits_flat,
                    row_words,
                    unit.selected_global_rows,
                    "Max6 direct batch fallback rows",
                )?;
                let margins = margin_table.select(unit.selected_global_rows)?;
                scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold(
                    packed.as_slice(),
                    None,
                    row_words,
                    n_rows,
                    residual,
                    n_samples,
                    top_k,
                    true,
                    ResidualProposalMode::Max6,
                    ResidualScreenKernel::OneCell,
                    lookup,
                    Some(margins),
                    unit.pair_groups,
                    min_cell,
                    None,
                    false,
                )
            };
        let results = if allow_parallel && rayon::current_num_threads() > 1 && units.len() > 1 {
            units
                .par_iter()
                .map(scan_one)
                .collect::<Result<Vec<_>, String>>()?
        } else {
            units
                .iter()
                .map(scan_one)
                .collect::<Result<Vec<_>, String>>()?
        };
        let pairs_scored = results
            .iter()
            .map(|result| result.pairs_scored)
            .fold(0usize, usize::saturating_add);
        return Ok(ResidualScreenBatchResult {
            units: results,
            pairs_scored_unique: pairs_scored,
            score_offers: pairs_scored,
            used_direct_fallback: true,
            timing: ResidualScreenBatchTiming {
                prepare_wall_seconds,
                scan_merge_wall_seconds: scan_started.elapsed().as_secs_f64(),
                memberships: membership_count,
                parallel: allow_parallel && rayon::current_num_threads() > 1 && units.len() > 1,
                ..Default::default()
            },
        });
    }

    let mut all_rows = Vec::with_capacity(membership_count);
    for unit in units {
        all_rows.extend_from_slice(unit.selected_global_rows);
    }
    all_rows.sort_unstable();
    all_rows.dedup();
    let margins = margin_table.select(all_rows.as_slice())?;
    let mut margin_slots_by_unit = Vec::with_capacity(units.len());
    let mut records = Vec::with_capacity(membership_count);
    for (unit_index, unit) in units.iter().enumerate() {
        let mut margin_slots = Vec::with_capacity(unit.selected_global_rows.len());
        for (local_index, &global_row) in unit.selected_global_rows.iter().enumerate() {
            let margin_slot = all_rows
                .binary_search(&global_row)
                .expect("batch margin rows contain every unit membership");
            margin_slots.push(margin_slot);
            records.push(ResidualBatchMembershipRecord {
                global_row,
                margin_slot,
                unit_index,
                local_index,
                group: unit.pair_groups.map(|groups| groups[local_index]),
            });
        }
        margin_slots_by_unit.push(margin_slots);
    }
    records.sort_unstable_by_key(|record| (record.global_row, record.unit_index));
    let mut source_rows = Vec::<ResidualBatchSourceRow>::new();
    let mut memberships = Vec::<ResidualBatchMembership>::with_capacity(records.len());
    let mut record_start = 0usize;
    while record_start < records.len() {
        let global_row = records[record_start].global_row;
        let margin_slot = records[record_start].margin_slot;
        let membership_start = memberships.len();
        let mut record_end = record_start + 1;
        memberships.push(ResidualBatchMembership {
            unit_index: records[record_start].unit_index,
            local_index: records[record_start].local_index,
            group: records[record_start].group,
        });
        while record_end < records.len() && records[record_end].global_row == global_row {
            let record = records[record_end];
            memberships.push(ResidualBatchMembership {
                unit_index: record.unit_index,
                local_index: record.local_index,
                group: record.group,
            });
            record_end += 1;
        }
        source_rows.push(ResidualBatchSourceRow {
            global_row,
            margin_slot,
            membership_start,
            membership_end: memberships.len(),
        });
        record_start = record_end;
    }

    let prepare_wall_seconds = prepare_started.elapsed().as_secs_f64();
    let parallel = allow_parallel && rayon::current_num_threads() > 1 && source_rows.len() >= 32;
    let scan_started = Instant::now();
    let worker_state = || ResidualBatchWorkerState::new(units.len(), top_k, needed_words);
    let mut state = if parallel {
        (0..source_rows.len())
            .into_par_iter()
            .with_min_len(min_anchor_len.max(1))
            .fold(worker_state, |mut state, anchor_index| {
                if state.fold_started_at.is_none() {
                    state.fold_started_at = Some(Instant::now());
                    state.fold_worker_index = rayon::current_thread_index();
                }
                state.fold_anchor_count += 1;
                residual_batch_scan_anchor(
                    anchor_index,
                    bits_flat,
                    row_words,
                    units,
                    margin_slots_by_unit.as_slice(),
                    source_rows.as_slice(),
                    memberships.as_slice(),
                    &margins,
                    residual,
                    n_samples,
                    top_k,
                    min_cell,
                    true,
                    lookup,
                    &mut state,
                );
                state
            })
            .map(|mut state| {
                if let Some(started_at) = state.fold_started_at.take() {
                    let wall_seconds = started_at.elapsed().as_secs_f64();
                    let worker_index = state.fold_worker_index.unwrap_or(0);
                    state.scan_reduce_critical_path_seconds = wall_seconds;
                    state.fold_timings.push(ResidualBatchFoldTiming {
                        worker_index,
                        anchors: state.fold_anchor_count,
                        wall_seconds,
                        finish_offset_seconds: scan_started.elapsed().as_secs_f64(),
                    });
                }
                state
            })
            .reduce(worker_state, |mut left, mut right| {
                let merge_started = Instant::now();
                let topk_merge_started = Instant::now();
                for unit_index in 0..units.len() {
                    for Reverse(entry) in right.heaps[unit_index].drain() {
                        push_top_k(&mut left.heaps[unit_index], entry, top_k);
                    }
                }
                let topk_merge_seconds = topk_merge_started.elapsed().as_secs_f64();
                let left_merge_critical_path = left.topk_merge_critical_path_seconds;
                let right_merge_critical_path = right.topk_merge_critical_path_seconds;
                left.topk_merge_critical_path_seconds = reduce_critical_path_seconds(
                    left_merge_critical_path,
                    right_merge_critical_path,
                    topk_merge_seconds,
                );
                let left_scan_critical_path = left.scan_reduce_critical_path_seconds;
                let right_scan_critical_path = right.scan_reduce_critical_path_seconds;
                left.scan_reduce_critical_path_seconds = reduce_critical_path_seconds(
                    left_scan_critical_path,
                    right_scan_critical_path,
                    topk_merge_seconds,
                );
                left.topk_merge_task_seconds += right.topk_merge_task_seconds + topk_merge_seconds;
                left.topk_merge_max_task_seconds = left
                    .topk_merge_max_task_seconds
                    .max(right.topk_merge_max_task_seconds)
                    .max(topk_merge_seconds);
                left.topk_merge_task_count = left
                    .topk_merge_task_count
                    .saturating_add(right.topk_merge_task_count)
                    .saturating_add(1);
                left.fold_timings.append(&mut right.fold_timings);
                for unit_index in 0..units.len() {
                    left.pairs_evaluated[unit_index] = left.pairs_evaluated[unit_index]
                        .saturating_add(right.pairs_evaluated[unit_index]);
                    left.pairs_scored[unit_index] = left.pairs_scored[unit_index]
                        .saturating_add(right.pairs_scored[unit_index]);
                }
                left.pairs_scored_unique = left
                    .pairs_scored_unique
                    .saturating_add(right.pairs_scored_unique);
                left.score_offers = left.score_offers.saturating_add(right.score_offers);
                left.merge_task_seconds +=
                    right.merge_task_seconds + merge_started.elapsed().as_secs_f64();
                left
            })
    } else {
        let mut state = worker_state();
        for anchor_index in 0..source_rows.len() {
            residual_batch_scan_anchor(
                anchor_index,
                bits_flat,
                row_words,
                units,
                margin_slots_by_unit.as_slice(),
                source_rows.as_slice(),
                memberships.as_slice(),
                &margins,
                residual,
                n_samples,
                top_k,
                min_cell,
                true,
                lookup,
                &mut state,
            );
        }
        state
    };

    let scan_merge_wall_seconds = scan_started.elapsed().as_secs_f64();
    let fold_task_count = state.fold_timings.len();
    let fold_task_wall_sum_seconds = state
        .fold_timings
        .iter()
        .map(|timing| timing.wall_seconds)
        .sum::<f64>();
    let fold_task_wall_max_seconds = state
        .fold_timings
        .iter()
        .map(|timing| timing.wall_seconds)
        .fold(0.0, f64::max);
    let fold_anchor_count_min = state
        .fold_timings
        .iter()
        .map(|timing| timing.anchors)
        .min()
        .unwrap_or(0);
    let fold_anchor_count_max = state
        .fold_timings
        .iter()
        .map(|timing| timing.anchors)
        .max()
        .unwrap_or(0);
    let last_fold_finish_seconds = state
        .fold_timings
        .iter()
        .map(|timing| timing.finish_offset_seconds)
        .fold(0.0, f64::max);
    let mut worker_task_wall_seconds = Vec::<f64>::new();
    let mut worker_last_finish_seconds = Vec::<Option<f64>>::new();
    for timing in &state.fold_timings {
        if worker_task_wall_seconds.len() <= timing.worker_index {
            worker_task_wall_seconds.resize(timing.worker_index + 1, 0.0);
            worker_last_finish_seconds.resize(timing.worker_index + 1, None);
        }
        worker_task_wall_seconds[timing.worker_index] += timing.wall_seconds;
        worker_last_finish_seconds[timing.worker_index] = Some(
            worker_last_finish_seconds[timing.worker_index]
                .unwrap_or(0.0_f64)
                .max(timing.finish_offset_seconds),
        );
    }
    let mut active_worker_task_wall_seconds = worker_task_wall_seconds
        .iter()
        .copied()
        .filter(|seconds| *seconds > 0.0)
        .collect::<Vec<_>>();
    active_worker_task_wall_seconds.sort_unstable_by(f64::total_cmp);
    let fold_active_workers = active_worker_task_wall_seconds.len();
    let worker_task_wall_min_seconds = active_worker_task_wall_seconds
        .first()
        .copied()
        .unwrap_or(0.0);
    let worker_task_wall_median_seconds = active_worker_task_wall_seconds
        .get(fold_active_workers / 2)
        .copied()
        .unwrap_or(0.0);
    let worker_task_wall_max_seconds = active_worker_task_wall_seconds
        .last()
        .copied()
        .unwrap_or(0.0);
    let active_worker_finish_times = worker_last_finish_seconds
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let worker_finish_skew_seconds = active_worker_finish_times
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - active_worker_finish_times
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
    let worker_finish_skew_seconds = if worker_finish_skew_seconds.is_finite() {
        worker_finish_skew_seconds
    } else {
        0.0
    };
    let post_last_fold_tail_seconds = if last_fold_finish_seconds.is_finite() {
        (scan_merge_wall_seconds - last_fold_finish_seconds).max(0.0)
    } else {
        0.0
    };
    let finalize_started = Instant::now();
    let mut results = Vec::with_capacity(units.len());
    for (unit_index, unit) in units.iter().enumerate() {
        let mut candidates = Vec::with_capacity(state.heaps[unit_index].len());
        for Reverse(entry) in state.heaps[unit_index].drain() {
            let first_local = unit
                .selected_global_rows
                .binary_search(&entry.first)
                .expect("batch finalist first endpoint belongs to unit");
            let second_local = unit
                .selected_global_rows
                .binary_search(&entry.second)
                .expect("batch finalist second endpoint belongs to unit");
            let first_start = entry.first * row_words;
            let second_start = entry.second * row_words;
            let first_row = &bits_flat[first_start..first_start + needed_words];
            let second_row = &bits_flat[second_start..second_start + needed_words];
            let stats = one_cell_sufficient_stats_packed_no_valid(
                first_row,
                second_row,
                needed_words,
                tail_mask(n_samples),
                residual,
                margin_slots_by_unit[unit_index][first_local],
                margin_slots_by_unit[unit_index][second_local],
                &margins,
                lookup,
            );
            candidates.push(ResidualScreenCandidate {
                first: first_local,
                second: second_local,
                best_gate: entry.best_gate,
                proposal_score: entry.proposal_score,
                proposal_source: entry.proposal_source,
                stats,
                derived: derive_proposal_scores(stats, ResidualProposalMode::Max6),
            });
        }
        candidates.sort_unstable_by(|left, right| candidate_cmp(right, left));
        results.push(ResidualScreenResult {
            candidates,
            pairs_evaluated: state.pairs_evaluated[unit_index],
            pairs_scored: state.pairs_scored[unit_index],
            n_rows: unit.selected_global_rows.len(),
            n_samples,
        });
    }
    Ok(ResidualScreenBatchResult {
        units: results,
        pairs_scored_unique: state.pairs_scored_unique,
        score_offers: state.score_offers,
        used_direct_fallback: false,
        timing: ResidualScreenBatchTiming {
            prepare_wall_seconds,
            scan_merge_wall_seconds,
            merge_task_seconds: state.merge_task_seconds,
            fold_task_count,
            fold_task_wall_sum_seconds,
            fold_task_wall_max_seconds,
            fold_anchor_count_min,
            fold_anchor_count_max,
            fold_active_workers,
            worker_task_wall_min_seconds,
            worker_task_wall_median_seconds,
            worker_task_wall_max_seconds,
            worker_finish_skew_seconds,
            post_last_fold_tail_seconds,
            topk_merge_task_seconds: state.topk_merge_task_seconds,
            topk_merge_max_task_seconds: state.topk_merge_max_task_seconds,
            topk_merge_task_count: state.topk_merge_task_count,
            topk_merge_critical_path_seconds: state.topk_merge_critical_path_seconds,
            scan_reduce_critical_path_seconds: state.scan_reduce_critical_path_seconds,
            finalize_wall_seconds: finalize_started.elapsed().as_secs_f64(),
            source_rows: source_rows.len(),
            memberships: membership_count,
            parallel,
        },
    })
}

/// Recompute the frozen Max6 score and four-cell summaries for a supplied
/// candidate list.
///
/// The proposal stage may use a dense weighted-Gram kernel and a different
/// floating-point accumulation order.  This function deliberately uses the
/// same packed four-cell primitive as the exhaustive scanner, so the retained
/// candidates have the exact Max6 score, gate label, tie-breaking fields, and
/// cell sums used by the reference implementation.  `row_ids` maps the
/// compact packed rows in `bits_flat` back to their marker indices in the full
/// window.
#[allow(clippy::too_many_arguments)]
fn rerank_residual_pairs_packed(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    row_ids: &[usize],
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    candidate_pairs: &[(usize, usize)],
    top_k: usize,
    include_xor: bool,
) -> Result<ResidualScreenResult, String> {
    if n_rows < 2 {
        return Err("four-cell residual rerank requires at least two markers".to_string());
    }
    let needed_words = words_for_samples(n_samples);
    let sample_tail = tail_mask(n_samples);
    if row_words < needed_words {
        return Err(format!(
            "four-cell residual rerank row_words={row_words} smaller than required {needed_words}"
        ));
    }
    let required = row_ids
        .len()
        .checked_mul(row_words)
        .ok_or_else(|| "four-cell residual rerank packed size overflow".to_string())?;
    if bits_flat.len() < required {
        return Err(format!(
            "four-cell residual rerank bits length={} smaller than required {required}",
            bits_flat.len()
        ));
    }
    if let Some(valid) = valid_flat {
        if valid.len() < required {
            return Err(format!(
                "four-cell residual rerank valid bits length={} smaller than required {required}",
                valid.len()
            ));
        }
    }
    validate_continuous_y(residual, n_samples, "four-cell residual rerank")?;

    let mut compact_rows = HashMap::with_capacity(row_ids.len());
    for (compact_idx, &row_id) in row_ids.iter().enumerate() {
        if row_id >= n_rows {
            return Err(format!(
                "four-cell residual rerank row id out of range: {row_id} (n_rows={n_rows})"
            ));
        }
        if compact_rows.insert(row_id, compact_idx).is_some() {
            return Err(format!(
                "four-cell residual rerank row id appears more than once: {row_id}"
            ));
        }
    }

    let lookup = PackedYSumLookup::build(residual, n_samples)?;
    let mut seen = HashSet::with_capacity(candidate_pairs.len());
    let mut candidates = Vec::with_capacity(candidate_pairs.len().min(top_k));
    let mut pairs_scored = 0usize;

    for &(raw_first, raw_second) in candidate_pairs {
        if raw_first == raw_second {
            return Err(format!(
                "four-cell residual rerank candidate is not a pair: ({raw_first}, {raw_second})"
            ));
        }
        let (first, second) = if raw_first < raw_second {
            (raw_first, raw_second)
        } else {
            (raw_second, raw_first)
        };
        if first >= n_rows || second >= n_rows {
            return Err(format!(
                "four-cell residual rerank candidate out of range: ({first}, {second}) (n_rows={n_rows})"
            ));
        }
        if !seen.insert((first, second)) {
            continue;
        }
        let first_compact = *compact_rows.get(&first).ok_or_else(|| {
            format!("four-cell residual rerank candidate first marker {first} was not loaded")
        })?;
        let second_compact = *compact_rows.get(&second).ok_or_else(|| {
            format!("four-cell residual rerank candidate second marker {second} was not loaded")
        })?;
        let first_start = first_compact * row_words;
        let second_start = second_compact * row_words;
        let first_row = &bits_flat[first_start..first_start + needed_words];
        let second_row = &bits_flat[second_start..second_start + needed_words];
        let stats = if valid_flat.is_none() {
            four_cell_sufficient_stats_packed_no_valid(
                first_row,
                second_row,
                needed_words,
                sample_tail,
                &lookup,
            )
        } else {
            let first_valid =
                valid_flat.map(|valid| &valid[first_start..first_start + needed_words]);
            let second_valid =
                valid_flat.map(|valid| &valid[second_start..second_start + needed_words]);
            four_cell_sufficient_stats_packed_unchecked(
                first_row,
                second_row,
                first_valid,
                second_valid,
                n_samples,
                &lookup,
            )
        };
        let derived = derive_gate5_gate6d_scores(stats);
        let Some(best_gate) = best_gate6_score(
            &derived.logic5,
            derived.dosage_interaction_standardized_score,
            derived.dosage_interaction_contrast,
            include_xor,
        ) else {
            continue;
        };
        pairs_scored = pairs_scored.saturating_add(1);
        candidates.push(ResidualScreenCandidate {
            first,
            second,
            best_gate,
            proposal_score: best_gate.score,
            proposal_source: ResidualProposalSource::Max6,
            stats,
            derived,
        });
    }

    candidates.sort_unstable_by(|lhs, rhs| candidate_cmp(rhs, lhs));
    candidates.truncate(top_k);
    Ok(ResidualScreenResult {
        candidates,
        pairs_evaluated: seen.len(),
        pairs_scored,
        n_rows,
        n_samples,
    })
}

fn u64_array2_to_vec(array: &PyReadonlyArray2<'_, u64>) -> Vec<u64> {
    array.as_array().iter().copied().collect()
}

fn set_screen_columns<'py>(
    out: &Bound<'py, PyDict>,
    candidates: &[ResidualScreenCandidate],
) -> PyResult<()> {
    let cell_names = ["00", "01", "10", "11"];
    for (cell, name) in cell_names.iter().enumerate() {
        out.set_item(
            format!("n{name}"),
            candidates
                .iter()
                .map(|candidate| candidate.stats.counts[cell])
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            format!("sum{name}"),
            candidates
                .iter()
                .map(|candidate| candidate.stats.sums[cell])
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            format!("residual_mean{name}"),
            candidates
                .iter()
                .map(|candidate| {
                    let count = candidate.stats.counts[cell];
                    if count == 0 {
                        f64::NAN
                    } else {
                        candidate.stats.sums[cell] / count as f64
                    }
                })
                .collect::<Vec<_>>(),
        )?;
    }
    out.set_item(
        "n_valid",
        candidates
            .iter()
            .map(|candidate| candidate.stats.n_valid)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "best_gate",
        candidates
            .iter()
            .map(|candidate| candidate.best_gate.gate.as_str())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "screen_score",
        candidates
            .iter()
            .map(|candidate| candidate.proposal_score)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "proposal_score",
        candidates
            .iter()
            .map(|candidate| candidate.proposal_score)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "proposal_source",
        candidates
            .iter()
            .map(|candidate| candidate.proposal_source.as_str())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "signed_score",
        candidates
            .iter()
            .map(|candidate| candidate.best_gate.signed_score)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "n_hit",
        candidates
            .iter()
            .map(|candidate| candidate.best_gate.n_hit)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "dosage_interaction_contrast",
        candidates
            .iter()
            .map(|candidate| {
                candidate
                    .derived
                    .dosage_interaction_contrast
                    .unwrap_or(f64::NAN)
            })
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "dosage_interaction_standardized_score",
        candidates
            .iter()
            .map(|candidate| {
                candidate
                    .derived
                    .dosage_interaction_standardized_score
                    .unwrap_or(f64::NAN)
            })
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "dosage_product_score",
        candidates
            .iter()
            .map(|candidate| candidate.derived.dosage_product_score.unwrap_or(f64::NAN))
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "dosage_product_signed_score",
        candidates
            .iter()
            .map(|candidate| {
                candidate
                    .derived
                    .dosage_product_signed_score
                    .unwrap_or(f64::NAN)
            })
            .collect::<Vec<_>>(),
    )?;
    let gate_names = ["and00", "and01", "and10", "and11", "xor"];
    let all_scores = candidates
        .iter()
        .map(|candidate| candidate.derived.logic5)
        .collect::<Vec<_>>();
    for (gate_idx, name) in gate_names.iter().enumerate() {
        out.set_item(
            format!("{name}_score"),
            all_scores
                .iter()
                .map(|scores| scores[gate_idx].score)
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            format!("{name}_signed_score"),
            all_scores
                .iter()
                .map(|scores| scores[gate_idx].signed_score)
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            format!("{name}_n_hit"),
            all_scores
                .iter()
                .map(|scores| scores[gate_idx].n_hit)
                .collect::<Vec<_>>(),
        )?;
    }
    Ok(())
}

/// Scan every packed marker pair with a residual-only Max5/Max6 primitive.
///
/// `proposal_mode="nmax6"` is an EXPERIMENTAL proposal variant only. The
/// production/default benchmark path remains `proposal_mode="max6"`; nMax6
/// results must be kept separate from Max6 results even when the same
/// candidate reranker is used.
///
/// `bin_path` stores marker-major 0/1 bits.  `valid_bits`, when provided, is
/// a marker-major `(n_rows, row_words)` uint64 matrix; pair validity is the
/// intersection of the two marker rows. The default missing-free path is the
/// exact one-cell/margin kernel. If `valid_bits` is supplied, a one-cell
/// request safely falls back to the exact four-cell reference kernel because
/// reusable global margins no longer describe each pair's valid sample set.
/// The returned screen scores are for ranking only; exact GRM-aware inference
/// must use the existing exact backend on retained candidates.
#[pyfunction(name = "garfield_logic_residual_screen_bin")]
#[pyo3(signature = (bin_path, residual, top_k=100, valid_bits=None, xor_search=true, proposal_mode="max5", screen_kernel="one_cell"))]
pub fn garfield_logic_residual_screen_bin_py<'py>(
    py: Python<'py>,
    bin_path: String,
    residual: PyReadonlyArray1<'py, f64>,
    top_k: usize,
    valid_bits: Option<PyReadonlyArray2<'py, u64>>,
    xor_search: bool,
    proposal_mode: &str,
    screen_kernel: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let (bits, row_words, n_rows, n_samples) =
        load_bin01_as_u64_words(&bin_path, "garfield_logic_residual_screen_bin")
            .map_err(PyRuntimeError::new_err)?;
    let residual_slice = residual
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("garfield_logic_residual_screen_bin: {e}")))?;
    let valid_owned = if let Some(valid) = valid_bits.as_ref() {
        let shape = valid.shape();
        if shape != [n_rows, row_words] {
            return Err(PyValueError::new_err(format!(
                "valid_bits shape=({},{}) expected ({n_rows},{row_words})",
                shape[0], shape[1]
            )));
        }
        Some(u64_array2_to_vec(valid))
    } else {
        None
    };
    let proposal_mode =
        ResidualProposalMode::parse(proposal_mode).map_err(PyValueError::new_err)?;
    let requested_screen_kernel =
        ResidualScreenKernel::parse(screen_kernel).map_err(PyValueError::new_err)?;
    let screen_kernel = requested_screen_kernel.effective_for_valid_mask(valid_owned.is_some());
    let result = scan_residual_pairs_packed_with_mode_and_kernel(
        &bits,
        valid_owned.as_deref(),
        row_words,
        n_rows,
        residual_slice,
        n_samples,
        top_k,
        xor_search,
        proposal_mode,
        screen_kernel,
        true,
    )
    .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item(
        "first",
        result
            .candidates
            .iter()
            .map(|candidate| candidate.first)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "second",
        result
            .candidates
            .iter()
            .map(|candidate| candidate.second)
            .collect::<Vec<_>>(),
    )?;
    set_screen_columns(&out, result.candidates.as_slice())?;
    out.set_item("pairs_evaluated", result.pairs_evaluated)?;
    out.set_item("pairs_scored", result.pairs_scored)?;
    out.set_item("n_rows", result.n_rows)?;
    out.set_item("n_samples", result.n_samples)?;
    out.set_item("valid_mask_supplied", valid_bits.is_some())?;
    out.set_item("xor_search", xor_search)?;
    out.set_item("proposal_mode", proposal_mode.as_str())?;
    out.set_item("screen_kernel", screen_kernel.as_str())?;
    out.set_item("requested_screen_kernel", requested_screen_kernel.as_str())?;
    Ok(out)
}

#[pyfunction(name = "garfield_logic_residual_rerank_candidates_bin")]
#[pyo3(signature = (bin_path, residual, first, second, top_k=500, valid_bits=None, xor_search=true))]
pub fn garfield_logic_residual_rerank_candidates_bin_py<'py>(
    py: Python<'py>,
    bin_path: String,
    residual: PyReadonlyArray1<'py, f64>,
    first: PyReadonlyArray1<'py, i64>,
    second: PyReadonlyArray1<'py, i64>,
    top_k: usize,
    valid_bits: Option<PyReadonlyArray2<'py, u64>>,
    xor_search: bool,
) -> PyResult<Bound<'py, PyDict>> {
    ensure_garfield_blas_backend().map_err(PyRuntimeError::new_err)?;
    let (n_rows, row_words, n_samples) =
        bin01_shape(&bin_path, "garfield_logic_residual_rerank_candidates_bin")
            .map_err(PyRuntimeError::new_err)?;
    let residual_slice = residual.as_slice().map_err(|e| {
        PyValueError::new_err(format!(
            "garfield_logic_residual_rerank_candidates_bin: {e}"
        ))
    })?;
    let first_values = first.as_array();
    let second_values = second.as_array();
    if first_values.len() != second_values.len() {
        return Err(PyValueError::new_err(format!(
            "garfield_logic_residual_rerank_candidates_bin: first and second lengths differ ({} vs {})",
            first_values.len(),
            second_values.len()
        )));
    }
    let mut candidate_pairs = Vec::with_capacity(first_values.len());
    let mut row_indices = Vec::with_capacity(first_values.len().saturating_mul(2));
    let mut seen_pairs = HashSet::with_capacity(first_values.len());
    let mut seen_rows = HashSet::with_capacity(first_values.len().saturating_mul(2));
    for (&raw_first, &raw_second) in first_values.iter().zip(second_values.iter()) {
        let first = usize::try_from(raw_first).map_err(|_| {
            PyValueError::new_err(format!(
                "garfield_logic_residual_rerank_candidates_bin: marker index must be non-negative, got {raw_first}"
            ))
        })?;
        let second = usize::try_from(raw_second).map_err(|_| {
            PyValueError::new_err(format!(
                "garfield_logic_residual_rerank_candidates_bin: marker index must be non-negative, got {raw_second}"
            ))
        })?;
        let pair = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        if pair.0 == pair.1 {
            return Err(PyValueError::new_err(format!(
                "garfield_logic_residual_rerank_candidates_bin: candidate is not a pair: ({first}, {second})"
            )));
        }
        if pair.0 >= n_rows || pair.1 >= n_rows {
            return Err(PyValueError::new_err(format!(
                "garfield_logic_residual_rerank_candidates_bin: candidate out of range: ({}, {}) (n_rows={n_rows})",
                pair.0, pair.1
            )));
        }
        if seen_pairs.insert(pair) {
            candidate_pairs.push(pair);
            if seen_rows.insert(pair.0) {
                row_indices.push(pair.0);
            }
            if seen_rows.insert(pair.1) {
                row_indices.push(pair.1);
            }
        }
    }

    let valid_owned = if let Some(valid) = valid_bits.as_ref() {
        let shape = valid.shape();
        if shape != [n_rows, row_words] {
            return Err(PyValueError::new_err(format!(
                "valid_bits shape=({},{}) expected ({n_rows},{row_words})",
                shape[0], shape[1]
            )));
        }
        Some(u64_array2_to_vec(valid))
    } else {
        None
    };

    let result = if candidate_pairs.is_empty() {
        validate_continuous_y(
            residual_slice,
            n_samples,
            "garfield_logic_residual_rerank_candidates_bin",
        )
        .map_err(PyValueError::new_err)?;
        ResidualScreenResult {
            candidates: Vec::new(),
            pairs_evaluated: 0,
            pairs_scored: 0,
            n_rows,
            n_samples,
        }
    } else {
        let (bits, selected_row_words, selected_n_rows, selected_n_samples) =
            load_bin01_selected_rows_as_u64_words(
                &bin_path,
                &row_indices,
                "garfield_logic_residual_rerank_candidates_bin",
            )
            .map_err(PyRuntimeError::new_err)?;
        debug_assert_eq!(selected_row_words, row_words);
        debug_assert_eq!(selected_n_rows, row_indices.len());
        debug_assert_eq!(selected_n_samples, n_samples);
        let valid_selected = valid_owned
            .as_deref()
            .map(|valid| {
                gather_rows_by_indices(
                    valid,
                    row_words,
                    &row_indices,
                    "garfield_logic_residual_rerank_candidates_bin",
                )
            })
            .transpose()
            .map_err(PyValueError::new_err)?;
        rerank_residual_pairs_packed(
            &bits,
            valid_selected.as_deref(),
            row_words,
            &row_indices,
            n_rows,
            residual_slice,
            n_samples,
            &candidate_pairs,
            top_k,
            xor_search,
        )
        .map_err(PyValueError::new_err)?
    };

    let out = PyDict::new(py);
    out.set_item(
        "first",
        result
            .candidates
            .iter()
            .map(|candidate| candidate.first)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "second",
        result
            .candidates
            .iter()
            .map(|candidate| candidate.second)
            .collect::<Vec<_>>(),
    )?;
    set_screen_columns(&out, result.candidates.as_slice())?;
    out.set_item("pairs_evaluated", result.pairs_evaluated)?;
    out.set_item("pairs_scored", result.pairs_scored)?;
    out.set_item("n_rows", result.n_rows)?;
    out.set_item("n_samples", result.n_samples)?;
    out.set_item("valid_mask_supplied", valid_bits.is_some())?;
    out.set_item("xor_search", xor_search)?;
    out.set_item("proposal_mode", "max6")?;
    out.set_item("candidate_only", true)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garfield::four_cell::{
        derive_dosage_interaction_contrast, derive_logic5_scores as derive_residual_gate_scores,
        derive_standardized_dosage_interaction_score, FourCellGate as ResidualGate,
    };
    use crate::garfield::score::score_cont_centered_gain_from_sum_and_n_hit;

    fn pack(values: &[bool]) -> Vec<u64> {
        let mut out = vec![0u64; values.len().div_ceil(64).max(1)];
        for (idx, &value) in values.iter().enumerate() {
            if value {
                out[idx >> 6] |= 1u64 << (idx & 63);
            }
        }
        out
    }

    #[test]
    fn n11_min_cell_interval_matches_explicit_four_cell_counts() {
        for n_samples in 0usize..=24 {
            for first_count in 0usize..=n_samples {
                for second_count in 0usize..=n_samples {
                    for min_cell in 0usize..=7 {
                        let bounds = one_cell_min_cell_n11_bounds(
                            n_samples,
                            first_count,
                            second_count,
                            min_cell,
                        );
                        for n11 in 0usize..=n_samples {
                            let explicit_cells = if n11 <= first_count && n11 <= second_count {
                                let zero_count =
                                    n_samples as i128 - first_count as i128 - second_count as i128
                                        + n11 as i128;
                                (zero_count >= 0).then_some([
                                    zero_count as usize,
                                    second_count - n11,
                                    first_count - n11,
                                    n11,
                                ])
                            } else {
                                None
                            };
                            let expected = explicit_cells
                                .is_some_and(|cells| cells.iter().all(|&count| count >= min_cell));
                            let observed =
                                bounds.is_some_and(|(lower, upper)| lower <= n11 && n11 <= upper);
                            assert_eq!(observed, expected, "n={n_samples}, a={first_count}, b={second_count}, m={min_cell}, t={n11}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn four_cell_stats_match_dense_reference_and_xor_derivation() {
        let first = pack(&[false, false, true, true, false, true, true, false, true]);
        let second = pack(&[false, true, false, true, false, true, false, true, true]);
        let residual = [-4.0, 1.0, 2.0, 3.0, -2.0, 0.0, 4.0, -1.0, -3.0];
        let lookup = PackedYSumLookup::build(&residual, residual.len()).unwrap();
        let stats = four_cell_sufficient_stats_packed(
            &first,
            &second,
            None,
            None,
            &residual,
            residual.len(),
            &lookup,
        )
        .unwrap();
        assert_eq!(stats.counts, [2, 2, 2, 3]);
        assert_eq!(stats.sums, [-6.0, 0.0, 6.0, 0.0]);
        assert_eq!(stats.n_valid, 9);
        assert_eq!(stats.sums[1] + stats.sums[2], 6.0);
        assert_eq!(stats.counts[1] + stats.counts[2], 4);
    }

    #[test]
    fn missing_free_stats_fast_path_matches_reference_path() {
        let first = pack(&[
            false, true, false, true, true, false, true, false, true, false, true, true,
        ]);
        let second = pack(&[
            true, true, false, false, true, true, false, true, false, false, true, false,
        ]);
        let residual = [
            -3.5, 1.25, 0.0, 4.0, -2.0, 3.0, 1.5, -1.0, 2.25, -4.0, 0.75, 5.0,
        ];
        let lookup = PackedYSumLookup::build(&residual, residual.len()).unwrap();
        let reference = four_cell_sufficient_stats_packed_unchecked(
            &first,
            &second,
            None,
            None,
            residual.len(),
            &lookup,
        );
        let fast = four_cell_sufficient_stats_packed_no_valid(
            &first,
            &second,
            words_for_samples(residual.len()),
            tail_mask(residual.len()),
            &lookup,
        );
        assert_eq!(fast, reference);
    }

    #[test]
    fn one_cell_margin_scan_matches_reference_four_cell_statistics() {
        let rows = [
            pack(&[
                false, true, false, true, true, false, true, false, true, false, true, true,
            ]),
            pack(&[
                true, true, false, false, true, true, false, true, false, false, true, false,
            ]),
            pack(&[
                false, false, true, true, false, true, true, true, false, true, false, true,
            ]),
        ];
        let bits_flat = rows.concat();
        let residual = [
            -3.5, 1.25, 0.0, 4.0, -2.0, 3.0, 1.5, -1.0, 2.25, -4.0, 0.75, 5.0,
        ];
        let lookup = PackedYSumLookup::build(&residual, residual.len()).unwrap();
        let margins = OneCellMargins::build(
            &bits_flat,
            1,
            rows.len(),
            &residual,
            residual.len(),
            &lookup,
        )
        .unwrap();
        let one_cell = scan_residual_pairs_packed_with_mode_and_kernel(
            &bits_flat,
            None,
            1,
            rows.len(),
            &residual,
            residual.len(),
            3,
            true,
            ResidualProposalMode::Max6,
            ResidualScreenKernel::OneCell,
            false,
        )
        .unwrap();
        let reference = scan_residual_pairs_packed_with_mode(
            &bits_flat,
            None,
            1,
            rows.len(),
            &residual,
            residual.len(),
            3,
            true,
            ResidualProposalMode::Max6,
            false,
        )
        .unwrap();

        assert_eq!(margins.counts, vec![7, 6, 7]);
        assert_eq!(margins.total_y, residual.iter().sum::<f64>());
        assert_eq!(one_cell.pairs_evaluated, reference.pairs_evaluated);
        assert_eq!(one_cell.pairs_scored, reference.pairs_scored);
        assert_eq!(one_cell.candidates.len(), reference.candidates.len());
        for (got, want) in one_cell.candidates.iter().zip(reference.candidates.iter()) {
            assert_eq!((got.first, got.second), (want.first, want.second));
            assert_eq!(got.stats.counts, want.stats.counts);
            for (got_sum, want_sum) in got.stats.sums.iter().zip(want.stats.sums.iter()) {
                assert!((got_sum - want_sum).abs() < 1.0e-12);
            }
            assert_eq!(got.derived.logic5.len(), want.derived.logic5.len());
            for (got_gate, want_gate) in got.derived.logic5.iter().zip(want.derived.logic5.iter()) {
                assert_eq!(got_gate.gate, want_gate.gate);
                assert_eq!(got_gate.n_hit, want_gate.n_hit);
                assert!((got_gate.score - want_gate.score).abs() < 1.0e-10);
            }
            assert_eq!(got.best_gate.gate, want.best_gate.gate);
            assert!((got.proposal_score - want.proposal_score).abs() < 1.0e-10);
        }
    }

    #[test]
    fn one_cell_zero_skip_and_reused_popcount_match_reference_stats_bitwise() {
        let n_samples = 130usize;
        let row_words = words_for_samples(n_samples);
        let first = [0b0101_0101u64, 0, 0b10];
        let second = [0b1100_1100u64, 1, 0b10];
        let bits = [first.as_slice(), second.as_slice()].concat();
        let y = (0..n_samples)
            .map(|index| (index as f64 - 65.0) * 0.03125)
            .collect::<Vec<_>>();
        let lookup = PackedYSumLookup::build(&y, n_samples).unwrap();
        let margins = OneCellMargins::build(&bits, row_words, 2, &y, n_samples, &lookup).unwrap();
        let expected = one_cell_sufficient_stats_packed_no_valid(
            &first,
            &second,
            row_words,
            tail_mask(n_samples),
            &y,
            0,
            1,
            &margins,
            &lookup,
        );
        let got = one_cell_sufficient_stats_packed_no_valid_skip_zero(
            &first,
            &second,
            row_words,
            tail_mask(n_samples),
            &y,
            0,
            1,
            &margins,
            &lookup,
        );

        assert_eq!(got.counts, expected.counts);
        assert_eq!(got.n_valid, expected.n_valid);
        for (got_sum, expected_sum) in got.sums.iter().zip(expected.sums.iter()) {
            assert_eq!(got_sum.to_bits(), expected_sum.to_bits());
        }
    }

    #[test]
    fn min_cell_popcount_scratch_preserves_one_cell_stats_bitwise() {
        let n_samples = 130usize;
        let row_words = words_for_samples(n_samples);
        let first_values = (0..n_samples)
            .map(|index| index % 3 == 0 || index % 7 == 1)
            .collect::<Vec<_>>();
        let second_values = (0..n_samples)
            .map(|index| index % 4 <= 1 || index % 11 == 0)
            .collect::<Vec<_>>();
        let first = pack(first_values.as_slice());
        let second = pack(second_values.as_slice());
        let bits = [first.as_slice(), second.as_slice()].concat();
        let y = (0..n_samples)
            .map(|index| (index as f64 - 65.0) * 0.03125)
            .collect::<Vec<_>>();
        let lookup = PackedYSumLookup::build(&y, n_samples).unwrap();
        let margins = OneCellMargins::build(&bits, row_words, 2, &y, n_samples, &lookup).unwrap();
        let bounds =
            one_cell_min_cell_n11_bounds(n_samples, margins.counts[0], margins.counts[1], 2)
                .unwrap();
        let mut popcounts = vec![0u8; row_words];
        assert!(packed_n11_min_cell_passes(
            first.as_slice(),
            second.as_slice(),
            n_samples,
            bounds.0,
            bounds.1,
            popcounts.as_mut_slice(),
        ));
        let expected = one_cell_sufficient_stats_packed_no_valid(
            first.as_slice(),
            second.as_slice(),
            row_words,
            tail_mask(n_samples),
            &y,
            0,
            1,
            &margins,
            &lookup,
        );
        let observed = one_cell_sufficient_stats_packed_no_valid_from_popcounts(
            first.as_slice(),
            second.as_slice(),
            row_words,
            tail_mask(n_samples),
            &y,
            0,
            1,
            &margins,
            &lookup,
            popcounts.as_slice(),
            false,
        );
        assert_eq!(observed.counts, expected.counts);
        assert_eq!(observed.n_valid, expected.n_valid);
        for (observed_sum, expected_sum) in observed.sums.iter().zip(expected.sums.iter()) {
            assert_eq!(observed_sum.to_bits(), expected_sum.to_bits());
        }
    }

    #[test]
    fn geneset_cross_window_scan_excludes_within_group_pairs() {
        let rows = [
            pack(&[false, true, false, true, true, false, true, false]),
            pack(&[true, true, false, false, true, true, false, true]),
            pack(&[false, false, true, true, false, true, true, true]),
            pack(&[true, false, true, false, true, false, false, true]),
        ];
        let bits_flat = rows.concat();
        let residual = [-3.5, 1.25, 0.0, 4.0, -2.0, 3.0, 1.5, -1.0];
        let lookup = PackedYSumLookup::build(&residual, residual.len()).unwrap();
        let margins = OneCellMargins::build(
            &bits_flat,
            1,
            rows.len(),
            &residual,
            residual.len(),
            &lookup,
        )
        .unwrap();
        let groups = [0usize, 0, 1, 1];
        let result = scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups(
            &bits_flat,
            None,
            1,
            rows.len(),
            &residual,
            residual.len(),
            6,
            true,
            ResidualProposalMode::Max6,
            ResidualScreenKernel::OneCell,
            &lookup,
            Some(margins),
            Some(&groups),
            false,
        )
        .unwrap();

        assert_eq!(result.pairs_evaluated, 4);
        assert!(result
            .candidates
            .iter()
            .all(|candidate| groups[candidate.first] != groups[candidate.second]));
    }

    #[test]
    fn cross_group_second_indices_are_exact_and_ordered() {
        let groups = [0usize, 0, 1, 2, 1, 2];
        let group_members = build_sorted_group_members(&groups);

        assert_eq!(
            cross_group_second_indices(&group_members, &groups, 0),
            vec![2, 3, 4, 5]
        );
        assert_eq!(
            cross_group_second_indices(&group_members, &groups, 2),
            vec![3, 5]
        );
        assert_eq!(
            cross_group_second_indices(&group_members, &groups, 4),
            vec![5]
        );
        assert_eq!(
            cross_group_second_indices(&group_members, &groups, 5),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn min_cell_filters_pairs_before_max6_scoring_and_topk() {
        let n_samples = 64usize;
        let first = (0..n_samples)
            .map(|sample| sample >= 32)
            .collect::<Vec<_>>();
        let second = (0..n_samples)
            .map(|sample| sample % 32 >= 16)
            .collect::<Vec<_>>();
        let third = (0..n_samples)
            .map(|sample| {
                if sample < 10 {
                    true
                } else if (32..42).contains(&sample) {
                    false
                } else {
                    first[sample]
                }
            })
            .collect::<Vec<_>>();
        let fourth = first.clone();
        let fifth = (0..n_samples)
            .map(|sample| {
                if sample < 5 {
                    true
                } else if (32..37).contains(&sample) {
                    false
                } else {
                    first[sample]
                }
            })
            .collect::<Vec<_>>();
        let rows = [
            pack(&first),
            pack(&second),
            pack(&third),
            pack(&fourth),
            pack(&fifth),
        ];
        let bits_flat = rows.concat();
        let residual = (0..n_samples)
            .map(|sample| ((sample * 17 % 29) as f64 - 14.0) / 7.0)
            .collect::<Vec<_>>();
        let lookup = PackedYSumLookup::build(&residual, n_samples).unwrap();
        let margins = OneCellMargins::build(
            &bits_flat,
            rows[0].len(),
            rows.len(),
            &residual,
            n_samples,
            &lookup,
        )
        .unwrap();

        let filtered =
            scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold(
                &bits_flat,
                None,
                rows[0].len(),
                rows.len(),
                &residual,
                n_samples,
                10,
                true,
                ResidualProposalMode::Max6,
                ResidualScreenKernel::OneCell,
                &lookup,
                Some(margins.clone()),
                None,
                10,
                None,
                false,
            )
            .unwrap();

        assert_eq!(filtered.pairs_evaluated, 10);
        let filtered_pairs = filtered
            .candidates
            .iter()
            .map(|candidate| (candidate.first, candidate.second))
            .collect::<Vec<_>>();
        assert!(filtered_pairs.contains(&(0, 2))); // two cells are exactly at min_cell=10
        assert!(!filtered_pairs.contains(&(0, 3))); // two cells are absent
        assert!(!filtered_pairs.contains(&(0, 4))); // the smaller cells contain only five samples

        let unfiltered =
            scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold(
                &bits_flat,
                None,
                rows[0].len(),
                rows.len(),
                &residual,
                n_samples,
                10,
                true,
                ResidualProposalMode::Max6,
                ResidualScreenKernel::OneCell,
                &lookup,
                Some(margins),
                None,
                0,
                None,
                false,
            )
            .unwrap();
        assert_eq!(unfiltered.pairs_evaluated, 10);
        assert!(unfiltered.pairs_scored > filtered.pairs_scored);
        assert!(unfiltered
            .candidates
            .iter()
            .any(|candidate| (candidate.first, candidate.second) == (0, 4)));
    }

    #[test]
    fn max6_batch_scan_matches_direct_unit_scans_with_local_groups() {
        let n_samples = 130usize;
        let row_words = words_for_samples(n_samples);
        let mut rows = Vec::new();
        for row in 0..45usize {
            let pattern = if row < 40 { row % 17 } else { row };
            let values = (0..n_samples)
                .map(|sample| (sample * 31 + pattern * 17 + sample / 7) % 101 < 48)
                .collect::<Vec<_>>();
            rows.push(pack(values.as_slice()));
        }
        let bits_flat = rows.concat();
        let residual = (0..n_samples)
            .map(|sample| ((sample * 37 % 103) as f64 - 51.0) / 13.0)
            .collect::<Vec<_>>();
        let lookup = PackedYSumLookup::build(&residual, n_samples).unwrap();
        let margin_rows = (0..rows.len()).collect::<Vec<_>>();
        let margin_table = OneCellMarginTable::build_for_rows(
            &bits_flat,
            row_words,
            margin_rows.as_slice(),
            &residual,
            n_samples,
            &lookup,
        )
        .unwrap();
        let unit_rows = [
            (0..40usize).collect::<Vec<_>>(),
            (0..40usize).collect::<Vec<_>>(),
            (40..45usize).collect::<Vec<_>>(),
        ];
        let groups_a = (0..40usize).map(|row| row % 2).collect::<Vec<_>>();
        let groups_b = (0..40usize).map(|row| row % 3).collect::<Vec<_>>();
        let units = [
            ResidualScreenBatchUnit {
                selected_global_rows: &unit_rows[0],
                pair_groups: Some(groups_a.as_slice()),
            },
            ResidualScreenBatchUnit {
                selected_global_rows: &unit_rows[1],
                pair_groups: Some(groups_b.as_slice()),
            },
            ResidualScreenBatchUnit {
                selected_global_rows: &unit_rows[2],
                pair_groups: None,
            },
        ];
        let scan_direct_unit = |unit: &ResidualScreenBatchUnit<'_>| {
            let packed = gather_rows_by_indices(
                bits_flat.as_slice(),
                row_words,
                unit.selected_global_rows,
                "batch parity test rows",
            )
            .unwrap();
            let margins = margin_table.select(unit.selected_global_rows).unwrap();
            scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold(
                packed.as_slice(),
                None,
                row_words,
                unit.selected_global_rows.len(),
                residual.as_slice(),
                n_samples,
                12,
                true,
                ResidualProposalMode::Max6,
                ResidualScreenKernel::OneCell,
                &lookup,
                Some(margins),
                unit.pair_groups,
                3,
                None,
                false,
            )
            .unwrap()
        };
        let direct = units.iter().map(scan_direct_unit).collect::<Vec<_>>();

        let serial = scan_residual_pairs_packed_max6_batch(
            bits_flat.as_slice(),
            row_words,
            &units,
            residual.as_slice(),
            n_samples,
            12,
            3,
            &lookup,
            &margin_table,
            false,
        )
        .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let parallel = pool
            .install(|| {
                scan_residual_pairs_packed_max6_batch(
                    bits_flat.as_slice(),
                    row_words,
                    &units,
                    residual.as_slice(),
                    n_samples,
                    12,
                    3,
                    &lookup,
                    &margin_table,
                    true,
                )
            })
            .unwrap();

        assert!(serial.pairs_scored_unique < serial.score_offers);
        assert_eq!(serial.units, direct);
        assert_eq!(parallel.units, direct);
        assert_eq!(serial.units, parallel.units);

        let same_group = vec![0usize; 40];
        let no_shared_pair_units = [
            ResidualScreenBatchUnit {
                selected_global_rows: &unit_rows[0],
                pair_groups: Some(same_group.as_slice()),
            },
            ResidualScreenBatchUnit {
                selected_global_rows: &unit_rows[0],
                pair_groups: None,
            },
        ];
        let no_shared_pair_batch = scan_residual_pairs_packed_max6_batch(
            bits_flat.as_slice(),
            row_words,
            &no_shared_pair_units,
            residual.as_slice(),
            n_samples,
            12,
            3,
            &lookup,
            &margin_table,
            false,
        )
        .unwrap();
        let no_shared_pair_direct = no_shared_pair_units
            .iter()
            .map(scan_direct_unit)
            .collect::<Vec<_>>();
        assert!(no_shared_pair_batch.used_direct_fallback);
        assert_eq!(no_shared_pair_batch.units, no_shared_pair_direct);
    }

    #[test]
    #[ignore = "diagnostic benchmark; run explicitly"]
    fn geneset_cross_group_fastpath_benchmark() {
        let n_samples = 1493usize;
        let n_rows = 800usize;
        let row_words = words_for_samples(n_samples);
        let mut bits_flat = vec![0u64; n_rows * row_words];
        for row in 0..n_rows {
            for word in 0..row_words {
                let mut value = (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (word as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                value ^= value.rotate_left((row % 63 + 1) as u32);
                bits_flat[row * row_words + word] = value;
            }
        }
        let groups = (0..n_rows).map(|row| row % 2).collect::<Vec<_>>();
        let residual = (0..n_samples)
            .map(|sample| ((sample * 37 % 101) as f64 - 50.0) / 10.0)
            .collect::<Vec<_>>();
        let lookup = PackedYSumLookup::build(&residual, n_samples).unwrap();
        let margins =
            OneCellMargins::build(&bits_flat, row_words, n_rows, &residual, n_samples, &lookup)
                .unwrap();

        let run = |mode| {
            let started = Instant::now();
            let result = scan_residual_pairs_packed_with_mode_options(
                &bits_flat,
                None,
                row_words,
                n_rows,
                &residual,
                n_samples,
                5_000,
                true,
                ResidualProposalMode::Max6,
                ResidualScreenKernel::OneCell,
                &lookup,
                Some(margins.clone()),
                Some(&groups),
                0,
                mode,
                None,
                false,
                None,
                None,
            )
            .unwrap();
            assert_eq!(result.pairs_evaluated, 160_000);
            std::hint::black_box((result.pairs_scored, result.candidates.len()));
            (started.elapsed().as_secs_f64(), result)
        };

        let (_, legacy_warmup) = run(PairTraversalMode::LegacyGroupFilter);
        let (_, optimized_warmup) = run(PairTraversalMode::CrossGroupOnly);
        assert_eq!(
            legacy_warmup
                .candidates
                .iter()
                .map(|candidate| (candidate.first, candidate.second))
                .collect::<Vec<_>>(),
            optimized_warmup
                .candidates
                .iter()
                .map(|candidate| (candidate.first, candidate.second))
                .collect::<Vec<_>>()
        );

        let mut legacy_seconds = Vec::new();
        let mut optimized_seconds = Vec::new();
        for _ in 0..5 {
            legacy_seconds.push(run(PairTraversalMode::LegacyGroupFilter).0);
            optimized_seconds.push(run(PairTraversalMode::CrossGroupOnly).0);
        }
        legacy_seconds.sort_by(f64::total_cmp);
        optimized_seconds.sort_by(f64::total_cmp);
        let legacy_median = legacy_seconds[legacy_seconds.len() / 2];
        let optimized_median = optimized_seconds[optimized_seconds.len() / 2];
        println!(
            "geneset benchmark: legacy={legacy_median:.6}s optimized={optimized_median:.6}s speedup={:.3}x",
            legacy_median / optimized_median
        );
    }

    #[test]
    fn shared_one_cell_context_matches_window_local_context() {
        let rows = [
            pack(&[
                false, true, false, true, true, false, true, false, true, false, true, true,
            ]),
            pack(&[
                true, true, false, false, true, true, false, true, false, false, true, false,
            ]),
            pack(&[
                false, false, true, true, false, true, true, true, false, true, false, true,
            ]),
        ];
        let bits_flat = rows.concat();
        let residual = [
            -3.5, 1.25, 0.0, 4.0, -2.0, 3.0, 1.5, -1.0, 2.25, -4.0, 0.75, 5.0,
        ];
        let lookup = PackedYSumLookup::build(&residual, residual.len()).unwrap();
        let table = OneCellMarginTable::build_for_rows(
            &bits_flat,
            1,
            &[0, 1, 2],
            &residual,
            residual.len(),
            &lookup,
        )
        .unwrap();
        let selected = table.select(&[2, 0, 1]).unwrap();
        let direct = OneCellMargins::build(
            &rows.concat(),
            1,
            rows.len(),
            &residual,
            residual.len(),
            &lookup,
        )
        .unwrap();
        assert_eq!(
            selected.counts,
            vec![direct.counts[2], direct.counts[0], direct.counts[1]]
        );
        assert_eq!(
            selected.sums,
            vec![direct.sums[2], direct.sums[0], direct.sums[1]]
        );

        let local_context = scan_residual_pairs_packed_with_mode_and_kernel(
            &rows.concat(),
            None,
            1,
            rows.len(),
            &residual,
            residual.len(),
            3,
            true,
            ResidualProposalMode::Max6,
            ResidualScreenKernel::OneCell,
            false,
        )
        .unwrap();
        let reordered_bits = [rows[2].clone(), rows[0].clone(), rows[1].clone()].concat();
        let shared_context = scan_residual_pairs_packed_with_mode_and_kernel_with_context(
            &reordered_bits,
            None,
            1,
            rows.len(),
            &residual,
            residual.len(),
            3,
            true,
            ResidualProposalMode::Max6,
            ResidualScreenKernel::OneCell,
            &lookup,
            Some(selected),
            false,
        )
        .unwrap();
        assert_eq!(
            shared_context.pairs_evaluated,
            local_context.pairs_evaluated
        );
        assert_eq!(shared_context.pairs_scored, local_context.pairs_scored);
        assert_eq!(
            shared_context.candidates.len(),
            local_context.candidates.len()
        );
    }

    #[test]
    fn one_cell_margin_progress_reports_chunks_without_changing_results() {
        let rows = [
            pack(&[false, true, false, true, true, false, true, false]),
            pack(&[true, true, false, false, true, true, false, true]),
            pack(&[false, false, true, true, false, true, true, true]),
        ];
        let bits_flat = rows.concat();
        let residual = [-3.5, 1.25, 0.0, 4.0, -2.0, 3.0, 1.5, -1.0];
        let lookup = PackedYSumLookup::build(&residual, residual.len()).unwrap();
        let baseline = OneCellMarginTable::build_for_rows(
            &bits_flat,
            1,
            &[0, 1, 2],
            &residual,
            residual.len(),
            &lookup,
        )
        .unwrap();
        let mut progress = Vec::new();
        let observed = OneCellMarginTable::build_for_rows_with_pool_and_progress(
            &bits_flat,
            1,
            &[0, 1, 2],
            &residual,
            residual.len(),
            &lookup,
            None,
            &mut |done, total| {
                progress.push((done, total));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(observed.counts, baseline.counts);
        assert_eq!(observed.sums, baseline.sums);
        assert_eq!(progress, vec![(0, 3), (1, 3), (2, 3), (3, 3)]);
    }

    #[test]
    fn threshold_pair_progress_preserves_screen_output() {
        let rows = [
            pack(&[false, true, false, true, true, false, true, false]),
            pack(&[true, true, false, false, true, true, false, true]),
            pack(&[false, false, true, true, false, true, true, true]),
        ];
        let bits_flat = rows.concat();
        let residual = [-3.5, 1.25, 0.0, 4.0, -2.0, 3.0, 1.5, -1.0];
        let lookup = PackedYSumLookup::build(&residual, residual.len()).unwrap();
        let margins = OneCellMarginTable::build_for_rows(
            &bits_flat,
            1,
            &[0, 1, 2],
            &residual,
            residual.len(),
            &lookup,
        )
        .unwrap();
        let baseline =
            scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold(
                &bits_flat,
                None,
                1,
                rows.len(),
                &residual,
                residual.len(),
                0,
                true,
                ResidualProposalMode::Max6,
                ResidualScreenKernel::OneCell,
                &lookup,
                Some(margins.select(&[0, 1, 2]).unwrap()),
                None,
                0,
                Some(1.0e12),
                false,
            )
            .unwrap();
        let mut progress = Vec::new();
        let observed = scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold_with_progress(
            &bits_flat,
            None,
            1,
            rows.len(),
            &residual,
            residual.len(),
            0,
            true,
            ResidualProposalMode::Max6,
            ResidualScreenKernel::OneCell,
            &lookup,
            Some(margins.select(&[0, 1, 2]).unwrap()),
            None,
            0,
            Some(1.0e12),
            false,
            &mut |done, total| {
                progress.push((done, total));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(observed, baseline);
        assert_eq!(progress, vec![(0, 3), (2, 3), (3, 3)]);
    }

    #[test]
    fn fused_proposal_gate_matches_materialized_scores() {
        let stats = FourCellSufficientStats {
            counts: [5, 7, 6, 8],
            sums: [-1.5, 2.25, -3.0, 4.75],
            n_valid: 26,
        };
        for mode in [
            ResidualProposalMode::Max5,
            ResidualProposalMode::Max6,
            ResidualProposalMode::NMax6,
        ] {
            for include_xor in [false, true] {
                let derived = derive_proposal_scores(stats, mode);
                let expected = best_gate_for_proposal(&derived, mode, include_xor);
                let actual = best_proposal_gate_fused(stats, mode, include_xor);
                assert_eq!(actual, expected, "mode={mode:?}, include_xor={include_xor}");
            }
        }
    }

    #[test]
    fn one_cell_margin_scan_rejects_valid_masked_rows() {
        let rows = [
            pack(&[false, true, false, true, true, false, true, false]),
            pack(&[true, true, false, false, true, true, false, true]),
        ];
        let valid = [
            pack(&[true, true, true, true, true, true, true, true]),
            pack(&[true, true, true, true, false, true, true, true]),
        ];
        let residual = [-3.5, 1.25, 0.0, 4.0, -2.0, 3.0, 1.5, -1.0];
        let error = scan_residual_pairs_packed_with_mode_and_kernel(
            &rows.concat(),
            Some(&valid.concat()),
            1,
            rows.len(),
            &residual,
            residual.len(),
            1,
            true,
            ResidualProposalMode::Max6,
            ResidualScreenKernel::OneCell,
            false,
        )
        .unwrap_err();
        assert!(error.contains("requires valid_bits=None"));
    }

    #[test]
    fn one_cell_public_default_falls_back_for_masked_rows() {
        assert_eq!(
            ResidualScreenKernel::OneCell.effective_for_valid_mask(false),
            ResidualScreenKernel::OneCell
        );
        assert_eq!(
            ResidualScreenKernel::OneCell.effective_for_valid_mask(true),
            ResidualScreenKernel::FourCell
        );
        assert_eq!(
            ResidualScreenKernel::FourCell.effective_for_valid_mask(true),
            ResidualScreenKernel::FourCell
        );
    }

    #[test]
    fn strict_pair_valid_mask_excludes_missing_or_heterozygous_samples() {
        let first = pack(&[false, false, true, true, false, true, true, false]);
        let second = pack(&[false, true, false, true, false, true, false, true]);
        let first_valid = pack(&[true, true, false, true, true, true, true, true]);
        let second_valid = pack(&[true, true, true, true, false, true, true, true]);
        let residual = [-3.0, 1.0, 8.0, 2.0, 9.0, -1.0, 4.0, -2.0];
        let lookup = PackedYSumLookup::build(&residual, residual.len()).unwrap();
        let stats = four_cell_sufficient_stats_packed(
            &first,
            &second,
            Some(&first_valid),
            Some(&second_valid),
            &residual,
            residual.len(),
            &lookup,
        )
        .unwrap();
        assert_eq!(stats.counts, [1, 2, 1, 2]);
        assert_eq!(stats.sums, [-3.0, -1.0, 4.0, 1.0]);
        assert_eq!(stats.n_valid, 6);
    }

    #[test]
    fn residual_gate_scores_match_centered_gain_reference() {
        let stats = FourCellSufficientStats {
            counts: [2, 2, 2, 3],
            sums: [-6.0, 0.0, 6.0, 0.0],
            n_valid: 9,
        };
        let gates = derive_residual_gate_scores(stats);
        let total = stats.sums.iter().sum::<f64>();
        let expected = [
            (2usize, -6.0),
            (2usize, 0.0),
            (2usize, 6.0),
            (3usize, 0.0),
            (4usize, 6.0),
        ];
        for (gate, (n_hit, sum_hit)) in gates.iter().zip(expected) {
            let reference =
                score_cont_centered_gain_from_sum_and_n_hit(total, sum_hit, stats.n_valid, n_hit);
            assert!(
                (gate.score - reference.raw_score).abs() < 1e-12,
                "gate={:?} n_hit={} signed={} got={} expected={}",
                gate.gate,
                n_hit,
                gate.signed_score,
                gate.score,
                reference.raw_score
            );
            assert_eq!(gate.n_hit, n_hit);
        }
    }

    #[test]
    fn zero_variance_gate_is_unidentifiable() {
        let stats = FourCellSufficientStats {
            counts: [4, 0, 0, 0],
            sums: [0.0, 0.0, 0.0, 0.0],
            n_valid: 4,
        };
        let gates = derive_residual_gate_scores(stats);
        assert!(gates[0].score.is_nan());
        assert!(gates[0].signed_score.is_nan());
    }

    #[test]
    fn pair_scan_returns_all_pairs_in_deterministic_score_order() {
        let rows = [
            pack(&[false, false, true, true, false]),
            pack(&[false, true, false, true, false]),
            pack(&[true, true, true, false, false]),
        ];
        let bits_flat = rows.concat();
        let residual = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = scan_residual_pairs_packed(
            &bits_flat,
            None,
            1,
            rows.len(),
            &residual,
            residual.len(),
            3,
            true,
        )
        .unwrap();
        assert_eq!(result.pairs_evaluated, 3);
        assert_eq!(result.pairs_scored, 3);
        assert_eq!(result.candidates.len(), 3);

        // The scanner and the public primitive must agree on every returned
        // candidate, including the one-pass four-cell statistics.
        let lookup = PackedYSumLookup::build(&residual, residual.len()).unwrap();
        for candidate in &result.candidates {
            let first = &rows[candidate.first];
            let second = &rows[candidate.second];
            let expected = four_cell_sufficient_stats_packed(
                first,
                second,
                None,
                None,
                &residual,
                residual.len(),
                &lookup,
            )
            .unwrap();
            assert_eq!(candidate.stats, expected);
            assert_eq!(
                candidate.derived,
                Gate5Gate6dScores {
                    logic5: derive_residual_gate_scores(expected),
                    dosage_interaction_contrast: None,
                    dosage_interaction_standardized_score: None,
                    dosage_product_score: None,
                    dosage_product_signed_score: None,
                },
                "candidate branches must be derived from the same four-cell table"
            );
            let expected_gate = best_gate_score(&derive_residual_gate_scores(expected), true)
                .expect("at least one gate should have variance");
            assert_eq!(candidate.best_gate, expected_gate);
        }
        for pair in result.candidates.windows(2) {
            assert!(candidate_cmp(&pair[0], &pair[1]) != Ordering::Less);
        }
    }

    #[test]
    fn pair_scan_intersects_strict_valid_masks() {
        let rows = [
            pack(&[false, false, true, true, false]),
            pack(&[false, true, false, true, false]),
        ];
        let valid_rows = [
            pack(&[true, true, false, true, true]),
            pack(&[true, true, true, true, false]),
        ];
        let bits_flat = rows.concat();
        let valid_flat = valid_rows.concat();
        let residual = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = scan_residual_pairs_packed(
            &bits_flat,
            Some(&valid_flat),
            1,
            rows.len(),
            &residual,
            residual.len(),
            10,
            true,
        )
        .unwrap();
        assert_eq!(result.pairs_evaluated, 1);
        assert_eq!(result.pairs_scored, 1);
        assert_eq!(result.candidates[0].stats.n_valid, 3);
        assert_eq!(result.candidates[0].stats.counts.iter().sum::<usize>(), 3);
    }

    #[test]
    fn missing_free_screen_matches_legacy_logic5_pair_scores() {
        let rows = [
            pack(&[false, false, true, true, false]),
            pack(&[false, true, false, true, false]),
            pack(&[true, true, true, false, false]),
        ];
        let bits_flat = rows.concat();
        let residual = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let legacy =
            crate::garfield::all_pair::scan_all_pairs_continuous_packed_with_parallel_and_xor(
                &bits_flat,
                1,
                rows.len(),
                &residual,
                residual.len(),
                3,
                false,
                true,
            )
            .unwrap();
        let screen = scan_residual_pairs_packed(
            &bits_flat,
            None,
            1,
            rows.len(),
            &residual,
            residual.len(),
            3,
            true,
        )
        .unwrap();
        assert_eq!(legacy.candidates.len(), screen.candidates.len());
        for (old, current) in legacy.candidates.iter().zip(screen.candidates.iter()) {
            assert_eq!((old.first, old.second), (current.first, current.second));
            let expected_gate = match (old.gate, old.polarity) {
                (crate::garfield::all_pair::AllPairGate::And, 0) => ResidualGate::And11,
                (crate::garfield::all_pair::AllPairGate::And, 1) => ResidualGate::And01,
                (crate::garfield::all_pair::AllPairGate::And, 2) => ResidualGate::And10,
                (crate::garfield::all_pair::AllPairGate::And, 3) => ResidualGate::And00,
                (crate::garfield::all_pair::AllPairGate::Xor, 0) => ResidualGate::Xor,
                _ => panic!("unexpected legacy gate/polarity"),
            };
            assert_eq!(current.best_gate.gate, expected_gate);
            // AND masks use exactly the same scalar score expression.  XOR
            // is derived from the two cell sums, so allow only the rounding
            // from that final two-term addition.
            if matches!(expected_gate, ResidualGate::Xor) {
                assert!((old.raw_score - current.best_gate.score).abs() < 1e-12);
            } else {
                assert_eq!(old.raw_score.to_bits(), current.best_gate.score.to_bits());
            }
        }
    }

    #[test]
    fn missing_free_fast_scan_preserves_reference_ordering() {
        let n_samples = 130usize;
        let rows = (0..8usize)
            .map(|row| {
                let values = (0..n_samples)
                    .map(|sample| (sample * 17 + row * 13 + (sample >> 2)) % 11 < 5)
                    .collect::<Vec<_>>();
                pack(&values)
            })
            .collect::<Vec<_>>();
        let bits_flat = rows.concat();
        let residual = (0..n_samples)
            .map(|sample| {
                let base = ((sample * 37) % 101) as f64 - 50.0;
                base * 0.013 + (sample % 3) as f64 * 0.007
            })
            .collect::<Vec<_>>();

        let fast = scan_residual_pairs_packed_with_mode(
            &bits_flat,
            None,
            rows[0].len(),
            rows.len(),
            &residual,
            n_samples,
            20,
            true,
            ResidualProposalMode::Max6,
            false,
        )
        .unwrap();

        let lookup = PackedYSumLookup::build(&residual, n_samples).unwrap();
        let mut reference = Vec::new();
        for first in 0..rows.len() - 1 {
            for second in first + 1..rows.len() {
                let stats = four_cell_sufficient_stats_packed_unchecked(
                    &rows[first],
                    &rows[second],
                    None,
                    None,
                    n_samples,
                    &lookup,
                );
                let derived = derive_gate5_gate6d_scores(stats);
                let Some(best_gate) = best_gate6_score(
                    &derived.logic5,
                    derived.dosage_interaction_standardized_score,
                    derived.dosage_interaction_contrast,
                    true,
                ) else {
                    continue;
                };
                reference.push(ResidualScreenCandidate {
                    first,
                    second,
                    best_gate,
                    proposal_score: best_gate.score,
                    proposal_source: ResidualProposalSource::Max6,
                    stats,
                    derived,
                });
            }
        }
        reference.sort_unstable_by(|lhs, rhs| candidate_cmp(rhs, lhs));
        reference.truncate(20);

        assert_eq!(fast.pairs_evaluated, 28);
        assert_eq!(fast.pairs_scored, 28);
        assert_eq!(fast.candidates.len(), reference.len());
        for (got, want) in fast.candidates.iter().zip(reference.iter()) {
            assert_eq!((got.first, got.second), (want.first, want.second));
            assert_eq!(got.stats.counts, want.stats.counts);
            for (got_sum, want_sum) in got.stats.sums.iter().zip(want.stats.sums.iter()) {
                assert!((got_sum - want_sum).abs() < 1.0e-10);
            }
            assert!((got.proposal_score - want.proposal_score).abs() < 1.0e-9);
            assert_eq!(got.best_gate.gate, want.best_gate.gate);
        }
    }

    #[test]
    fn thresholded_screen_returns_all_and_only_pairs_above_threshold() {
        let n_samples = 32usize;
        let rows = (0..6usize)
            .map(|row| {
                let values = (0..n_samples)
                    .map(|sample| (sample * 7 + row * 11 + (sample >> 1)) % 13 < 6)
                    .collect::<Vec<_>>();
                pack(&values)
            })
            .collect::<Vec<_>>();
        let bits_flat = rows.concat();
        let residual = (0..n_samples)
            .map(|sample| ((sample * 19 % 23) as f64 - 11.0) * 0.07)
            .collect::<Vec<_>>();
        let all = scan_residual_pairs_packed_with_mode_and_kernel(
            &bits_flat,
            None,
            rows[0].len(),
            rows.len(),
            &residual,
            n_samples,
            100,
            true,
            ResidualProposalMode::Max6,
            ResidualScreenKernel::OneCell,
            false,
        )
        .unwrap();
        let threshold = all.candidates[2].proposal_score;
        let thresholded =
            scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold(
                &bits_flat,
                None,
                rows[0].len(),
                rows.len(),
                &residual,
                n_samples,
                1,
                true,
                ResidualProposalMode::Max6,
                ResidualScreenKernel::OneCell,
                &PackedYSumLookup::build(&residual, n_samples).unwrap(),
                Some(
                    OneCellMargins::build(
                        &bits_flat,
                        rows[0].len(),
                        rows.len(),
                        &residual,
                        n_samples,
                        &PackedYSumLookup::build(&residual, n_samples).unwrap(),
                    )
                    .unwrap(),
                ),
                None,
                0,
                Some(threshold),
                false,
            )
            .unwrap();

        let expected = all
            .candidates
            .iter()
            .filter(|candidate| candidate.proposal_score > threshold)
            .map(|candidate| (candidate.first, candidate.second))
            .collect::<Vec<_>>();
        let observed = thresholded
            .candidates
            .iter()
            .map(|candidate| (candidate.first, candidate.second))
            .collect::<Vec<_>>();
        assert_eq!(thresholded.pairs_evaluated, all.pairs_evaluated);
        assert_eq!(thresholded.pairs_scored, all.pairs_scored);
        assert_eq!(observed, expected);
        assert!(thresholded
            .candidates
            .iter()
            .all(|candidate| candidate.proposal_score > threshold));
    }

    #[test]
    fn candidate_only_rerank_matches_full_scan_when_all_pairs_are_candidates() {
        let n_samples = 130usize;
        let rows = (0..8usize)
            .map(|row| {
                let values = (0..n_samples)
                    .map(|sample| (sample * 17 + row * 13 + (sample >> 2)) % 11 < 5)
                    .collect::<Vec<_>>();
                pack(&values)
            })
            .collect::<Vec<_>>();
        let bits_flat = rows.concat();
        let residual = (0..n_samples)
            .map(|sample| {
                let base = ((sample * 37) % 101) as f64 - 50.0;
                base * 0.013 + (sample % 3) as f64 * 0.007
            })
            .collect::<Vec<_>>();
        let all_pairs = (0..rows.len() - 1)
            .flat_map(|first| ((first + 1)..rows.len()).map(move |second| (first, second)))
            .collect::<Vec<_>>();
        let row_ids = (0..rows.len()).collect::<Vec<_>>();

        let full = scan_residual_pairs_packed_with_mode(
            &bits_flat,
            None,
            rows[0].len(),
            rows.len(),
            &residual,
            n_samples,
            20,
            true,
            ResidualProposalMode::Max6,
            false,
        )
        .unwrap();
        let reranked = rerank_residual_pairs_packed(
            &bits_flat,
            None,
            rows[0].len(),
            &row_ids,
            rows.len(),
            &residual,
            n_samples,
            &all_pairs,
            20,
            true,
        )
        .unwrap();

        assert_eq!(reranked.pairs_evaluated, all_pairs.len());
        assert_eq!(reranked.pairs_scored, full.pairs_scored);
        assert_eq!(reranked.candidates.len(), full.candidates.len());
        for (got, want) in reranked.candidates.iter().zip(full.candidates.iter()) {
            assert_eq!((got.first, got.second), (want.first, want.second));
            assert_eq!(got.best_gate, want.best_gate);
            assert_eq!(got.stats, want.stats);
            for (got_gate, want_gate) in got.derived.logic5.iter().zip(want.derived.logic5.iter()) {
                assert_eq!(got_gate.gate, want_gate.gate);
                assert_eq!(got_gate.n_hit, want_gate.n_hit);
                if got_gate.score.is_nan() || want_gate.score.is_nan() {
                    assert!(got_gate.score.is_nan() && want_gate.score.is_nan());
                    assert!(got_gate.signed_score.is_nan() && want_gate.signed_score.is_nan());
                } else {
                    assert_eq!(got_gate.score.to_bits(), want_gate.score.to_bits());
                    assert_eq!(
                        got_gate.signed_score.to_bits(),
                        want_gate.signed_score.to_bits()
                    );
                }
            }
            assert_eq!(
                got.derived.dosage_interaction_contrast,
                want.derived.dosage_interaction_contrast
            );
            assert_eq!(
                got.derived.dosage_interaction_standardized_score,
                want.derived.dosage_interaction_standardized_score
            );
            assert_eq!(got.proposal_score.to_bits(), want.proposal_score.to_bits());
        }
    }

    #[test]
    fn four_cell_dosage_contrast_is_the_saturated_interaction_contrast() {
        let stats = FourCellSufficientStats {
            counts: [2, 3, 4, 5],
            sums: [2.0, 6.0, 12.0, 35.0],
            n_valid: 14,
        };
        // Cell means are [1, 2, 3, 7], hence beta_12 in the saturated
        // 0/1-by-0/1 model is 7 - 3 - 2 + 1 = 3.
        let contrast = derive_dosage_interaction_contrast(stats).unwrap();
        assert!((contrast - 3.0).abs() < 1.0e-12);
        let score = derive_standardized_dosage_interaction_score(stats).unwrap();
        let expected = 9.0 / (0.5 + 1.0 / 3.0 + 0.25 + 0.2);
        assert!((score - expected).abs() < 1.0e-12);
    }

    #[test]
    fn four_cell_dosage_contrast_requires_all_cells() {
        let stats = FourCellSufficientStats {
            counts: [2, 0, 4, 5],
            sums: [2.0, 0.0, 12.0, 35.0],
            n_valid: 11,
        };
        assert!(derive_dosage_interaction_contrast(stats).is_none());
    }

    #[test]
    fn max6_proposal_mode_parses_and_exposes_standardized_dosage_gate() {
        assert_eq!(
            ResidualProposalMode::parse("max6").unwrap(),
            ResidualProposalMode::Max6
        );
        assert_eq!(ResidualProposalMode::Max6.as_str(), "max6");

        let stats = FourCellSufficientStats {
            counts: [1, 1, 1, 1],
            sums: [1.0, -1.0, -1.0, 1.0],
            n_valid: 4,
        };
        let logic = derive_residual_gate_scores(stats);
        let dosage = derive_standardized_dosage_interaction_score(stats).expect("complete cells");
        // Use a deliberately larger calibrated dosage score here: the helper
        // must select the sixth gate whenever its proposal score wins, even
        // when a Logic5 gate ties or exceeds the raw contrast on this table.
        let calibrated_dosage = dosage + 10.0;
        let best = best_gate6_score(&logic, Some(calibrated_dosage), Some(4.0), true)
            .expect("finite gate");
        assert_eq!(best.gate, ResidualGate::DosageInteraction);
        assert_eq!(best.score, calibrated_dosage);
    }

    #[test]
    fn nmax6_proposal_mode_uses_centered_product_score() {
        assert_eq!(
            ResidualProposalMode::parse("nmax6").unwrap(),
            ResidualProposalMode::NMax6
        );
        assert_eq!(ResidualProposalMode::NMax6.as_str(), "nmax6");

        let stats = FourCellSufficientStats {
            counts: [2, 3, 4, 5],
            sums: [2.0, 6.0, 12.0, 35.0],
            n_valid: 14,
        };
        let derived = derive_proposal_scores(stats, ResidualProposalMode::NMax6);
        let (expected_score, expected_signed) =
            derive_centered_product_screen_score(stats).unwrap();
        assert_eq!(derived.dosage_product_score, Some(expected_score));
        assert_eq!(derived.dosage_product_signed_score, Some(expected_signed));
        assert_eq!(derived.dosage_interaction_standardized_score, None);

        let best = best_gate_for_proposal(&derived, ResidualProposalMode::NMax6, true)
            .expect("one of the six proposal channels must be finite");
        assert!(best.score >= expected_score || best.gate != ResidualGate::DosageInteraction);

        // Force the sixth channel to win to verify its gate construction and
        // signed-score propagation independently of this particular table's
        // Logic5 winner.
        let mut forced = derived;
        forced.dosage_product_score = Some(best.score + 1.0);
        forced.dosage_product_signed_score = Some(expected_signed.abs() + 1.0);
        let forced_best = best_gate_for_proposal(&forced, ResidualProposalMode::NMax6, true)
            .expect("forced sixth channel must be finite");
        assert_eq!(forced_best.gate, ResidualGate::DosageInteraction);
        assert_eq!(forced_best.score, best.score + 1.0);
        assert_eq!(forced_best.signed_score, expected_signed.abs() + 1.0);
    }

    #[test]
    fn max6_scan_selects_gate6d_when_the_contrast_wins() {
        let rows = [
            pack(&[
                false, false, false, false, false, true, true, true, true, true, true, true, true,
                true,
            ]),
            pack(&[
                false, false, true, true, true, false, false, false, false, true, true, true, true,
                true,
            ]),
        ];
        let residual = [
            -2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        ];
        let result = scan_residual_pairs_packed_with_mode(
            &rows.concat(),
            None,
            1,
            rows.len(),
            &residual,
            residual.len(),
            1,
            true,
            ResidualProposalMode::Max6,
            false,
        )
        .unwrap();
        let candidate = &result.candidates[0];
        assert_eq!(candidate.best_gate.gate, ResidualGate::DosageInteraction);
        assert_eq!(candidate.derived.dosage_interaction_contrast, Some(-3.0));
        let expected = 9.0 / (0.5 + 1.0 / 3.0 + 0.25 + 0.2);
        assert_eq!(
            candidate.derived.dosage_interaction_standardized_score,
            Some(expected)
        );
        assert_eq!(candidate.proposal_score, expected);
    }
}
