use super::*;

#[cfg(test)]
pub(super) fn scan_max6_screen_null_replicate_maximum(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    null_y: &[f64],
    replicate: usize,
    pool: Option<&rayon::ThreadPool>,
) -> Result<f64, String> {
    if null_y.len() != logic_bits.n_samples {
        return Err(format!(
            "Max6 null phenotype {replicate} length {} != n_samples {}",
            null_y.len(),
            logic_bits.n_samples
        ));
    }
    let lookup = PackedYSumLookup::build(null_y, logic_bits.n_samples)?;
    let margins = OneCellMarginTable::build_for_rows_with_pool(
        logic_bits.bits(),
        logic_bits.row_words,
        universe.active_rows.as_slice(),
        null_y,
        logic_bits.n_samples,
        &lookup,
        pool,
    )?;
    let scan_unit = |unit: &Max6GrmMax6LdPreparedUnit| -> Result<f64, String> {
        let packed = gather_rows_by_indices(
            logic_bits.bits(),
            logic_bits.row_words,
            unit.selected_global_rows.as_slice(),
            "max6 parametric null selected rows",
        )?;
        let selected_margins = margins.select(unit.selected_global_rows.as_slice())?;
        let screen = logic_residual_screen::scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold(
            packed.as_slice(),
            None,
            logic_bits.row_words,
            unit.selected_global_rows.len(),
            null_y,
            logic_bits.n_samples,
            1,
            true,
            ResidualProposalMode::Max6,
            ResidualScreenKernel::OneCell,
            &lookup,
            Some(selected_margins),
            unit.pair_groups.as_deref(),
            0,
            None,
            pool.is_some(),
        )?;
        Ok(screen
            .candidates
            .first()
            .map(|candidate| candidate.proposal_score)
            .unwrap_or(f64::NEG_INFINITY))
    };
    let unit_maxima = if let Some(pool) = pool {
        pool.install(|| {
            universe
                .units
                .par_iter()
                .map(scan_unit)
                .collect::<Result<Vec<_>, _>>()
        })?
    } else {
        universe
            .units
            .iter()
            .map(scan_unit)
            .collect::<Result<Vec<_>, _>>()?
    };
    let maximum = unit_maxima.into_iter().fold(f64::NEG_INFINITY, f64::max);
    // If every pair is unidentifiable under a draw, the screen has no
    // rejection opportunity. Max6 scores are nonnegative, so zero is a
    // conservative finite representation of that empty maximum.
    Ok(if maximum.is_finite() { maximum } else { 0.0 })
}

pub(super) const MAX6_SCREEN_NULL_BATCH_SIZE: usize = 8;

pub(super) fn scan_max6_screen_null_batch(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    null_batch: &[Vec<f64>],
    pool: Option<&rayon::ThreadPool>,
    track_maximizing_pairs: bool,
) -> Result<(Vec<f64>, Option<Vec<Option<(usize, usize)>>>, usize), String> {
    crate::blas::ensure_garfield_blas_backend()?;
    if null_batch.is_empty() {
        return Err("Max6 null batch must contain at least one replicate".to_string());
    }
    for (offset, null_y) in null_batch.iter().enumerate() {
        if null_y.len() != logic_bits.n_samples {
            return Err(format!(
                "Max6 null phenotype in batch at offset {offset} has length {}, expected {}",
                null_y.len(),
                logic_bits.n_samples
            ));
        }
    }

    // Lookup and margins are phenotype-specific, but the selected genotype
    // rows and pair universe are shared across this batch.
    let lookups = null_batch
        .iter()
        .map(|null_y| PackedYSumLookup::build(null_y, logic_bits.n_samples))
        .collect::<Result<Vec<_>, _>>()?;
    let margins = null_batch
        .iter()
        .zip(lookups.iter())
        .map(|(null_y, lookup)| {
            OneCellMarginTable::build_for_rows_with_pool(
                logic_bits.bits(),
                logic_bits.row_words,
                universe.active_rows.as_slice(),
                null_y,
                logic_bits.n_samples,
                lookup,
                pool,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let draws = null_batch
        .iter()
        .zip(lookups.iter())
        .zip(margins.iter())
        .map(
            |((residual, lookup), margins)| logic_residual_screen::Max6ScreenNullDraw {
                residual: residual.as_slice(),
                lookup,
                margins,
            },
        )
        .collect::<Vec<_>>();

    let needed_words = logic_bits.n_samples.div_ceil(64);
    let bits = logic_bits.bits();
    let scan_unit = |unit_index: usize,
                     unit: &Max6GrmMax6LdPreparedUnit,
                     maxima: &mut [f64],
                     mut maximizing_pairs: Option<&mut [Option<(usize, usize)>]>,
                     sum11: &mut [f64]|
     -> Result<usize, String> {
        let rows = unit.selected_global_rows.as_slice();
        let mut owned_pairs = 0usize;
        for first_local in 0..rows.len().saturating_sub(1) {
            let first_global = rows[first_local];
            let first_start = first_global * logic_bits.row_words;
            let first_row = &bits[first_start..first_start + needed_words];
            for second_local in (first_local + 1)..rows.len() {
                let Some(pair) =
                    max6_screen_owned_pair(universe, unit_index, unit, first_local, second_local)?
                else {
                    continue;
                };
                let second_global = pair.second_global_row;
                let second_start = second_global * logic_bits.row_words;
                let second_row = &bits[second_start..second_start + needed_words];
                let pair_global_rows = (
                    pair.first_global_row.min(pair.second_global_row),
                    pair.first_global_row.max(pair.second_global_row),
                );
                logic_residual_screen::update_max6_one_cell_pair_batch(
                    first_row,
                    second_row,
                    logic_bits.n_samples,
                    pair.first_active_slot,
                    pair.second_active_slot,
                    track_maximizing_pairs.then_some(pair_global_rows),
                    draws.as_slice(),
                    sum11,
                    maxima,
                    maximizing_pairs.as_deref_mut(),
                    None,
                )?;
                owned_pairs = owned_pairs.saturating_add(1);
            }
        }
        Ok(owned_pairs)
    };

    let initial = || {
        (
            vec![f64::NEG_INFINITY; null_batch.len()],
            track_maximizing_pairs.then(|| vec![None; null_batch.len()]),
            0usize,
        )
    };
    let (mut maxima, maximizing_pairs, unique_pair_count) = if let Some(pool) = pool {
        pool.install(|| {
            universe
                .units
                .par_iter()
                .enumerate()
                .fold(
                    || {
                        let (maxima, pairs, count) = initial();
                        (Ok((maxima, pairs, count)), vec![0.0; null_batch.len()])
                    },
                    |(state, mut sum11), (unit_index, unit)| {
                        let (mut maxima, mut maximizing_pairs, mut count) = match state {
                            Ok(state) => state,
                            Err(error) => return (Err(error), sum11),
                        };
                        match scan_unit(
                            unit_index,
                            unit,
                            maxima.as_mut_slice(),
                            maximizing_pairs.as_deref_mut(),
                            sum11.as_mut_slice(),
                        ) {
                            Ok(unit_count) => {
                                count = count.saturating_add(unit_count);
                                (Ok((maxima, maximizing_pairs, count)), sum11)
                            }
                            Err(error) => (Err(error), sum11),
                        }
                    },
                )
                .map(|(state, _)| state)
                .reduce(
                    || Ok(initial()),
                    |left, right| match (left, right) {
                        (
                            Ok((mut left_maxima, mut left_pairs, left_count)),
                            Ok((right_maxima, right_pairs, right_count)),
                        ) => {
                            for index in 0..left_maxima.len() {
                                match right_maxima[index].total_cmp(&left_maxima[index]) {
                                    std::cmp::Ordering::Greater => {
                                        left_maxima[index] = right_maxima[index];
                                        if let (Some(left), Some(right)) =
                                            (left_pairs.as_mut(), right_pairs.as_ref())
                                        {
                                            left[index] = right[index];
                                        }
                                    }
                                    std::cmp::Ordering::Equal => {
                                        if let (Some(left), Some(right)) =
                                            (left_pairs.as_mut(), right_pairs.as_ref())
                                        {
                                            left[index] = match (left[index], right[index]) {
                                                (Some(a), Some(b)) => Some(a.min(b)),
                                                (Some(a), None) => Some(a),
                                                (None, Some(b)) => Some(b),
                                                (None, None) => None,
                                            };
                                        }
                                    }
                                    std::cmp::Ordering::Less => {}
                                }
                            }
                            Ok((
                                left_maxima,
                                left_pairs,
                                left_count.saturating_add(right_count),
                            ))
                        }
                        (Err(error), _) | (_, Err(error)) => Err(error),
                    },
                )
        })?
    } else {
        let mut maxima = vec![f64::NEG_INFINITY; null_batch.len()];
        let mut maximizing_pairs = track_maximizing_pairs.then(|| vec![None; null_batch.len()]);
        let mut sum11 = vec![0.0; null_batch.len()];
        let mut unique_pair_count = 0usize;
        for (unit_index, unit) in universe.units.iter().enumerate() {
            unique_pair_count = unique_pair_count.saturating_add(scan_unit(
                unit_index,
                unit,
                maxima.as_mut_slice(),
                maximizing_pairs.as_deref_mut(),
                sum11.as_mut_slice(),
            )?);
        }
        (maxima, maximizing_pairs, unique_pair_count)
    };

    for maximum in maxima.iter_mut() {
        if !maximum.is_finite() {
            // Preserve the scalar scanner's finite zero for an empty or
            // wholly unidentifiable universe.
            *maximum = 0.0;
        }
    }
    Ok((maxima, maximizing_pairs, unique_pair_count))
}

pub(super) struct Max6ScreenTailAccumulator {
    pub(super) top_pairs: Vec<sampled_screen::TailTopK>,
    pub(super) sampled_ahead_counts: Vec<Vec<usize>>,
    pub(super) pair_count: usize,
}

impl Max6ScreenTailAccumulator {
    pub(super) fn new(draw_count: usize, sampled_target_count: usize) -> Self {
        Self {
            top_pairs: (0..draw_count)
                .map(|_| sampled_screen::TailTopK::new(sampled_screen::TAIL_AUDIT_TOP_K))
                .collect(),
            sampled_ahead_counts: vec![vec![0; sampled_target_count]; draw_count],
            pair_count: 0,
        }
    }

    pub(super) fn merge(&mut self, other: Self) {
        for (target, source) in self.top_pairs.iter_mut().zip(other.top_pairs) {
            target.merge(source);
        }
        for (target, source) in self
            .sampled_ahead_counts
            .iter_mut()
            .zip(other.sampled_ahead_counts)
        {
            for (target_count, source_count) in target.iter_mut().zip(source) {
                *target_count = target_count.saturating_add(source_count);
            }
        }
        self.pair_count = self.pair_count.saturating_add(other.pair_count);
    }
}

pub(super) struct Max6ScreenExceedanceAccumulator {
    pub(super) top_pairs: Vec<sampled_screen::TailTopK>,
    pub(super) full_differences: Vec<i64>,
    pub(super) sampled_differences: Vec<i64>,
    pub(super) draw_count: usize,
    pub(super) threshold_count: usize,
    pub(super) sample_slot_count: usize,
    pub(super) pair_count: usize,
}

impl Max6ScreenExceedanceAccumulator {
    pub(super) fn new(
        draw_count: usize,
        threshold_count: usize,
        sample_slot_count: usize,
    ) -> Result<Self, String> {
        let threshold_stride = threshold_count
            .checked_add(1)
            .ok_or_else(|| "exceedance threshold stride overflow".to_string())?;
        let full_len = draw_count
            .checked_mul(5)
            .and_then(|value| value.checked_mul(threshold_stride))
            .ok_or_else(|| "full exceedance counter size overflow".to_string())?;
        let sampled_len = draw_count
            .checked_mul(sample_slot_count)
            .and_then(|value| value.checked_mul(5))
            .and_then(|value| value.checked_mul(threshold_stride))
            .ok_or_else(|| "sampled exceedance counter size overflow".to_string())?;
        let mut full_differences = Vec::new();
        full_differences
            .try_reserve_exact(full_len)
            .map_err(|error| format!("unable to reserve full exceedance counters: {error}"))?;
        full_differences.resize(full_len, 0i64);
        let mut sampled_differences = Vec::new();
        sampled_differences
            .try_reserve_exact(sampled_len)
            .map_err(|error| format!("unable to reserve sampled exceedance counters: {error}"))?;
        sampled_differences.resize(sampled_len, 0i64);
        let mut top_pairs = Vec::new();
        top_pairs
            .try_reserve_exact(draw_count)
            .map_err(|error| format!("unable to reserve per-null Top-K metadata: {error}"))?;
        for _ in 0..draw_count {
            top_pairs.push(sampled_screen::TailTopK::try_new(
                sampled_screen::TAIL_AUDIT_TOP_K,
            )?);
        }
        Ok(Self {
            top_pairs,
            full_differences,
            sampled_differences,
            draw_count,
            threshold_count,
            sample_slot_count,
            pair_count: 0,
        })
    }

    pub(super) fn add_range(
        differences: &mut [i64],
        start: usize,
        prefix_len: usize,
        threshold_count: usize,
    ) -> Result<(), String> {
        if prefix_len == 0 {
            return Ok(());
        }
        if prefix_len > threshold_count {
            return Err(format!(
                "strict exceedance prefix {prefix_len} exceeds threshold count {threshold_count}"
            ));
        }
        let end = start
            .checked_add(prefix_len)
            .ok_or_else(|| "exceedance difference index overflow".to_string())?;
        let first = differences
            .get_mut(start)
            .ok_or_else(|| "exceedance difference start is out of range".to_string())?;
        *first = first
            .checked_add(1)
            .ok_or_else(|| "exceedance difference counter overflow".to_string())?;
        let last = differences
            .get_mut(end)
            .ok_or_else(|| "exceedance difference end is out of range".to_string())?;
        *last = last
            .checked_sub(1)
            .ok_or_else(|| "exceedance difference counter underflow".to_string())?;
        Ok(())
    }

    pub(super) fn add_full(
        &mut self,
        draw: usize,
        stratum: usize,
        prefix_len: usize,
    ) -> Result<(), String> {
        let stride = self.threshold_count + 1;
        let start = (draw * 5 + stratum) * stride;
        Self::add_range(
            self.full_differences.as_mut_slice(),
            start,
            prefix_len,
            self.threshold_count,
        )
    }

    pub(super) fn add_sampled(
        &mut self,
        draw: usize,
        slot: usize,
        stratum: usize,
        prefix_len: usize,
    ) -> Result<(), String> {
        if slot >= self.sample_slot_count {
            return Err(format!("sample slot {slot} is out of range"));
        }
        let stride = self.threshold_count + 1;
        let start = ((draw * self.sample_slot_count + slot) * 5 + stratum) * stride;
        Self::add_range(
            self.sampled_differences.as_mut_slice(),
            start,
            prefix_len,
            self.threshold_count,
        )
    }

    pub(super) fn merge(&mut self, other: Self) -> Result<(), String> {
        if self.draw_count != other.draw_count
            || self.threshold_count != other.threshold_count
            || self.sample_slot_count != other.sample_slot_count
        {
            return Err("Max6 exceedance worker accumulator shape mismatch".to_string());
        }
        for (target, source) in self.full_differences.iter_mut().zip(other.full_differences) {
            *target = target
                .checked_add(source)
                .ok_or_else(|| "full exceedance difference merge overflow".to_string())?;
        }
        for (target, source) in self
            .sampled_differences
            .iter_mut()
            .zip(other.sampled_differences)
        {
            *target = target
                .checked_add(source)
                .ok_or_else(|| "sampled exceedance difference merge overflow".to_string())?;
        }
        for (target, source) in self.top_pairs.iter_mut().zip(other.top_pairs) {
            target.merge(source);
        }
        self.pair_count = self
            .pair_count
            .checked_add(other.pair_count)
            .ok_or_else(|| "Max6 exceedance pair count overflow".to_string())?;
        Ok(())
    }

    pub(super) fn reserved_bytes(&self) -> usize {
        self.full_differences
            .capacity()
            .saturating_mul(std::mem::size_of::<i64>())
            .saturating_add(
                self.sampled_differences
                    .capacity()
                    .saturating_mul(std::mem::size_of::<i64>()),
            )
            .saturating_add(
                self.top_pairs
                    .iter()
                    .map(|top| top.storage_bytes())
                    .sum::<usize>(),
            )
    }

    pub(super) fn into_counts(
        self,
    ) -> Result<
        (
            Vec<u32>,
            Vec<u32>,
            Vec<Vec<sampled_screen::TailPairScore>>,
            usize,
        ),
        String,
    > {
        let full_len = self.draw_count * 5 * self.threshold_count;
        let sampled_len = self.draw_count * self.sample_slot_count * 5 * self.threshold_count;
        let mut full_counts = Vec::new();
        full_counts.try_reserve_exact(full_len).map_err(|error| {
            format!("unable to reserve finalized full exceedance counts: {error}")
        })?;
        let mut sampled_counts = Vec::new();
        sampled_counts
            .try_reserve_exact(sampled_len)
            .map_err(|error| {
                format!("unable to reserve finalized sampled exceedance counts: {error}")
            })?;
        let stride = self.threshold_count + 1;
        for draw in 0..self.draw_count {
            for stratum in 0..5 {
                let start = (draw * 5 + stratum) * stride;
                let mut running = 0i64;
                for index in 0..self.threshold_count {
                    running = running
                        .checked_add(self.full_differences[start + index])
                        .ok_or_else(|| "full exceedance prefix sum overflow".to_string())?;
                    if !(0..=u32::MAX as i64).contains(&running) {
                        return Err(format!(
                            "full exceedance count {running} is outside u32 range"
                        ));
                    }
                    full_counts.push(running as u32);
                }
                if running + self.full_differences[start + self.threshold_count] != 0 {
                    return Err(
                        "full exceedance difference array did not close to zero".to_string()
                    );
                }
            }
            for slot in 0..self.sample_slot_count {
                for stratum in 0..5 {
                    let start = ((draw * self.sample_slot_count + slot) * 5 + stratum) * stride;
                    let mut running = 0i64;
                    for index in 0..self.threshold_count {
                        running = running
                            .checked_add(self.sampled_differences[start + index])
                            .ok_or_else(|| "sampled exceedance prefix sum overflow".to_string())?;
                        if !(0..=u32::MAX as i64).contains(&running) {
                            return Err(format!(
                                "sampled exceedance count {running} is outside u32 range"
                            ));
                        }
                        sampled_counts.push(running as u32);
                    }
                    if running + self.sampled_differences[start + self.threshold_count] != 0 {
                        return Err(
                            "sampled exceedance difference array did not close to zero".to_string()
                        );
                    }
                }
            }
        }
        Ok((
            full_counts,
            sampled_counts,
            self.top_pairs
                .into_iter()
                .map(sampled_screen::TailTopK::into_ranked)
                .collect(),
            self.pair_count,
        ))
    }
}

pub(super) fn scan_max6_screen_tail_batch(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    null_batch: &[Vec<f64>],
    sampled_winners: &[Vec<Option<sampled_screen::TailPairScore>>],
    pool: Option<&rayon::ThreadPool>,
) -> Result<
    (
        Vec<Vec<sampled_screen::TailPairScore>>,
        Vec<Vec<Option<usize>>>,
        usize,
    ),
    String,
> {
    crate::blas::ensure_garfield_blas_backend()?;
    if null_batch.is_empty() || null_batch.len() > MAX6_SCREEN_NULL_BATCH_SIZE {
        return Err(format!(
            "Max6 tail scan batch size must be in 1..={MAX6_SCREEN_NULL_BATCH_SIZE}, got {}",
            null_batch.len()
        ));
    }
    if sampled_winners.is_empty()
        || sampled_winners
            .iter()
            .any(|target| target.len() != null_batch.len())
    {
        return Err("Max6 tail sampled-winner batch shape mismatch".to_string());
    }
    for (offset, null_y) in null_batch.iter().enumerate() {
        if null_y.len() != logic_bits.n_samples {
            return Err(format!(
                "Max6 tail null phenotype at batch offset {offset} has length {}, expected {}",
                null_y.len(),
                logic_bits.n_samples
            ));
        }
    }

    let lookups = null_batch
        .iter()
        .map(|null_y| PackedYSumLookup::build(null_y, logic_bits.n_samples))
        .collect::<Result<Vec<_>, _>>()?;
    let margins = null_batch
        .iter()
        .zip(lookups.iter())
        .map(|(null_y, lookup)| {
            OneCellMarginTable::build_for_rows_with_pool(
                logic_bits.bits(),
                logic_bits.row_words,
                universe.active_rows.as_slice(),
                null_y,
                logic_bits.n_samples,
                lookup,
                pool,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let draws = null_batch
        .iter()
        .zip(lookups.iter())
        .zip(margins.iter())
        .map(
            |((residual, lookup), margins)| logic_residual_screen::Max6ScreenNullDraw {
                residual: residual.as_slice(),
                lookup,
                margins,
            },
        )
        .collect::<Vec<_>>();
    let words = logic_bits.n_samples.div_ceil(64);
    let bits = logic_bits.bits();

    let scan_units = |unit_offset: usize,
                      units: &[Max6GrmMax6LdPreparedUnit]|
     -> Result<Max6ScreenTailAccumulator, String> {
        let mut acc = Max6ScreenTailAccumulator::new(null_batch.len(), sampled_winners.len());
        let mut maxima = vec![f64::NEG_INFINITY; null_batch.len()];
        let mut sum11 = vec![0.0; null_batch.len()];
        let mut pair_scores = vec![None; null_batch.len()];
        for (local_unit_index, unit) in units.iter().enumerate() {
            let unit_index = unit_offset + local_unit_index;
            let rows = unit.selected_global_rows.as_slice();
            for first_local in 0..rows.len().saturating_sub(1) {
                for second_local in (first_local + 1)..rows.len() {
                    let Some(pair) = max6_screen_owned_pair(
                        universe,
                        unit_index,
                        unit,
                        first_local,
                        second_local,
                    )?
                    else {
                        continue;
                    };
                    let first = pair.first_global_row.min(pair.second_global_row);
                    let second = pair.first_global_row.max(pair.second_global_row);
                    let first_start = first * logic_bits.row_words;
                    let second_start = second * logic_bits.row_words;
                    logic_residual_screen::update_max6_one_cell_pair_batch(
                        &bits[first_start..first_start + words],
                        &bits[second_start..second_start + words],
                        logic_bits.n_samples,
                        universe.pair_ownership.active_slot(first),
                        universe.pair_ownership.active_slot(second),
                        None,
                        draws.as_slice(),
                        sum11.as_mut_slice(),
                        maxima.as_mut_slice(),
                        None,
                        Some(pair_scores.as_mut_slice()),
                    )?;
                    for draw_index in 0..null_batch.len() {
                        let Some(gate_score) = pair_scores[draw_index] else {
                            continue;
                        };
                        let candidate = sampled_screen::TailPairScore {
                            pair: (first, second),
                            score: gate_score.score,
                            best_channel: gate_score.gate,
                        };
                        acc.top_pairs[draw_index].consider(
                            candidate.pair,
                            candidate.score,
                            candidate.best_channel,
                        )?;
                        for target_index in 0..sampled_winners.len() {
                            if sampled_winners[target_index][draw_index].is_some_and(|sampled| {
                                sampled_screen::full_rank_cmp(&candidate, &sampled).is_lt()
                            }) {
                                acc.sampled_ahead_counts[draw_index][target_index] = acc
                                    .sampled_ahead_counts[draw_index][target_index]
                                    .saturating_add(1);
                            }
                        }
                    }
                    acc.pair_count = acc.pair_count.saturating_add(1);
                }
            }
        }
        Ok(acc)
    };

    let mut partials = if let Some(pool) = pool {
        pool.install(|| {
            universe
                .units
                .par_chunks(32)
                .enumerate()
                .map(|(chunk_index, units)| scan_units(chunk_index * 32, units))
                .collect::<Result<Vec<_>, _>>()
        })?
    } else {
        vec![scan_units(0, universe.units.as_slice())?]
    };
    let mut total = Max6ScreenTailAccumulator::new(null_batch.len(), sampled_winners.len());
    for partial in partials.drain(..) {
        total.merge(partial);
    }
    let top_pairs = total
        .top_pairs
        .into_iter()
        .map(sampled_screen::TailTopK::into_ranked)
        .collect::<Vec<_>>();
    let ranks = total
        .sampled_ahead_counts
        .into_iter()
        .map(|ahead| {
            ahead
                .into_iter()
                .map(|count| Some(count.saturating_add(1)))
                .collect()
        })
        .collect::<Vec<_>>();
    Ok((top_pairs, ranks, total.pair_count))
}

pub(super) fn max6_screen_exceedance_accumulator_preflight_bytes(
    draw_count: usize,
    threshold_count: usize,
    sample_slot_count: usize,
    worker_count: usize,
) -> Result<usize, String> {
    let threshold_stride = threshold_count
        .checked_add(1)
        .ok_or_else(|| "exceedance threshold stride overflow".to_string())?;
    let count_cells_per_worker = draw_count
        .checked_mul(
            5usize
                .checked_add(
                    sample_slot_count
                        .checked_mul(5)
                        .ok_or_else(|| "sampled exceedance stratum count overflow".to_string())?,
                )
                .ok_or_else(|| "sampled exceedance stratum count overflow".to_string())?,
        )
        .and_then(|value| value.checked_mul(threshold_stride))
        .ok_or_else(|| "exceedance accumulator cell count overflow".to_string())?;
    let diff_bytes_per_worker = count_cells_per_worker
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| "exceedance accumulator byte count overflow".to_string())?;
    let top_bytes_per_worker = draw_count
        .checked_mul(sampled_screen::TAIL_AUDIT_TOP_K)
        .and_then(|value| value.checked_mul(std::mem::size_of::<sampled_screen::TailPairScore>()))
        .ok_or_else(|| "Top-K accumulator byte count overflow".to_string())?;
    let scratch_bytes_per_worker = draw_count
        .checked_mul(
            std::mem::size_of::<f64>() * 2
                + std::mem::size_of::<Option<logic_residual_screen::ResidualGateScore>>(),
        )
        .ok_or_else(|| "exceedance scorer scratch byte count overflow".to_string())?;
    diff_bytes_per_worker
        .checked_add(top_bytes_per_worker)
        .and_then(|value| value.checked_add(scratch_bytes_per_worker))
        .and_then(|value| value.checked_mul(worker_count))
        .ok_or_else(|| "total exceedance worker memory estimate overflow".to_string())
}

pub(super) fn max6_screen_exceedance_anchor_task_count(
    universe: &Max6ScreenPreparedUniverse,
) -> Result<usize, String> {
    const ANCHORS_PER_EXCEEDANCE_TASK: usize = 16;
    universe.units.iter().try_fold(0usize, |total, unit| {
        total
            .checked_add(
                unit.selected_global_rows
                    .len()
                    .saturating_sub(1)
                    .div_ceil(ANCHORS_PER_EXCEEDANCE_TASK),
            )
            .ok_or_else(|| "exceedance anchor task count overflow".to_string())
    })
}

pub(super) fn scan_max6_screen_exceedance_batch(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    null_batch: &[Vec<f64>],
    thresholds: &[sampled_screen::ExceedanceAuditThreshold],
    sampling_plan: &sampled_screen::WeightedExceedanceSamplingPlan,
    pool: Option<&rayon::ThreadPool>,
    mut notify_unit_progress: impl FnMut(usize, usize) -> Result<(), String>,
) -> Result<sampled_screen::WeightedExceedanceFullScan, String> {
    crate::blas::ensure_garfield_blas_backend()?;
    if null_batch.is_empty() || null_batch.len() > MAX6_SCREEN_NULL_BATCH_SIZE {
        return Err(format!(
            "Max6 exceedance scan batch size must be in 1..={MAX6_SCREEN_NULL_BATCH_SIZE}, got {}",
            null_batch.len()
        ));
    }
    if thresholds.is_empty()
        || thresholds
            .windows(2)
            .any(|pair| pair[0].score >= pair[1].score)
    {
        return Err(
            "Max6 exceedance thresholds must be nonempty and strictly increasing".to_string(),
        );
    }
    if sampling_plan.seeds.len() != 5 {
        return Err(format!(
            "weighted Max6 exceedance scan requires five pair-sample seeds, got {}",
            sampling_plan.seeds.len()
        ));
    }
    for (offset, null_y) in null_batch.iter().enumerate() {
        if null_y.len() != logic_bits.n_samples {
            return Err(format!(
                "Max6 exceedance null at batch offset {offset} has length {}, expected {}",
                null_y.len(),
                logic_bits.n_samples
            ));
        }
    }

    let lookups = null_batch
        .iter()
        .map(|null_y| PackedYSumLookup::build(null_y, logic_bits.n_samples))
        .collect::<Result<Vec<_>, _>>()?;
    let margins = null_batch
        .iter()
        .zip(lookups.iter())
        .map(|(null_y, lookup)| {
            OneCellMarginTable::build_for_rows_with_pool(
                logic_bits.bits(),
                logic_bits.row_words,
                universe.active_rows.as_slice(),
                null_y,
                logic_bits.n_samples,
                lookup,
                pool,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let draws = null_batch
        .iter()
        .zip(lookups.iter())
        .zip(margins.iter())
        .map(
            |((residual, lookup), margins)| logic_residual_screen::Max6ScreenNullDraw {
                residual: residual.as_slice(),
                lookup,
                margins,
            },
        )
        .collect::<Vec<_>>();

    const ANCHORS_PER_EXCEEDANCE_TASK: usize = 16;
    let task_count = max6_screen_exceedance_anchor_task_count(universe)?;
    let worker_count = pool
        .map(rayon::ThreadPool::current_num_threads)
        .unwrap_or(1)
        .min(task_count.max(1));
    let preflight_bytes = max6_screen_exceedance_accumulator_preflight_bytes(
        null_batch.len(),
        thresholds.len(),
        sampling_plan.sample_slot_count(),
        worker_count,
    )?;
    let bits = logic_bits.bits();
    let words = logic_bits.n_samples.div_ceil(64);
    let supports = sampled_screen::support_counts_by_active_slot(logic_bits, universe);
    let mut tasks = Vec::<(usize, usize, usize)>::new();
    for (unit_index, unit) in universe.units.iter().enumerate() {
        let anchor_end = unit.selected_global_rows.len().saturating_sub(1);
        for first_start in (0..anchor_end).step_by(ANCHORS_PER_EXCEEDANCE_TASK) {
            tasks.push((
                unit_index,
                first_start,
                (first_start + ANCHORS_PER_EXCEEDANCE_TASK).min(anchor_end),
            ));
        }
    }
    if tasks.is_empty() {
        return Err("Max6 exceedance scan has no eligible anchor tasks".to_string());
    }
    if tasks.len() != max6_screen_exceedance_anchor_task_count(universe)? {
        return Err("Max6 exceedance anchor task count changed during preflight".to_string());
    }
    let work_metadata_bytes = tasks
        .capacity()
        .checked_mul(std::mem::size_of::<(usize, usize, usize)>())
        .ok_or_else(|| "exceedance task metadata byte count overflow".to_string())?;
    let mut worker_accumulators = (0..worker_count)
        .map(|_| {
            Max6ScreenExceedanceAccumulator::new(
                null_batch.len(),
                thresholds.len(),
                sampling_plan.sample_slot_count(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let scan_anchor_task = |acc: &mut Max6ScreenExceedanceAccumulator,
                            (unit_index, first_start, first_end): (usize, usize, usize)|
     -> Result<(), String> {
        let unit = &universe.units[unit_index];
        let mut maxima = vec![f64::NEG_INFINITY; null_batch.len()];
        let mut sum11 = vec![0.0; null_batch.len()];
        let mut pair_scores = vec![None; null_batch.len()];
        let rows = unit.selected_global_rows.as_slice();
        for first_local in first_start..first_end {
            for second_local in (first_local + 1)..rows.len() {
                let Some(pair) =
                    max6_screen_owned_pair(universe, unit_index, unit, first_local, second_local)?
                else {
                    continue;
                };
                let first = pair.first_global_row.min(pair.second_global_row);
                let second = pair.first_global_row.max(pair.second_global_row);
                let stratum = sampled_screen::pair_min_cell_stratum(
                    logic_bits,
                    universe,
                    supports.as_slice(),
                    first,
                    second,
                )?;
                let mut memberships = [[false; 6]; 5];
                for seed_index in 0..sampling_plan.seeds.len() {
                    memberships[seed_index] =
                        sampling_plan.memberships_for_pair(seed_index, (first, second), stratum)?;
                }
                let first_start = first * logic_bits.row_words;
                let second_start = second * logic_bits.row_words;
                logic_residual_screen::update_max6_one_cell_pair_batch(
                    &bits[first_start..first_start + words],
                    &bits[second_start..second_start + words],
                    logic_bits.n_samples,
                    universe.pair_ownership.active_slot(first),
                    universe.pair_ownership.active_slot(second),
                    None,
                    draws.as_slice(),
                    sum11.as_mut_slice(),
                    maxima.as_mut_slice(),
                    None,
                    Some(pair_scores.as_mut_slice()),
                )?;
                for draw_index in 0..null_batch.len() {
                    let Some(gate_score) = pair_scores[draw_index] else {
                        continue;
                    };
                    let score = gate_score.score;
                    let candidate = sampled_screen::TailPairScore {
                        pair: (first, second),
                        score,
                        best_channel: gate_score.gate,
                    };
                    acc.top_pairs[draw_index].consider(
                        candidate.pair,
                        candidate.score,
                        candidate.best_channel,
                    )?;
                    let strict_prefix =
                        thresholds.partition_point(|threshold| threshold.score < score);
                    acc.add_full(draw_index, stratum, strict_prefix)?;
                    for seed_index in 0..memberships.len() {
                        for sample_kind in 0..6 {
                            if memberships[seed_index][sample_kind] {
                                let slot = seed_index * 6 + sample_kind;
                                acc.add_sampled(draw_index, slot, stratum, strict_prefix)?;
                            }
                        }
                    }
                }
                acc.pair_count = acc
                    .pair_count
                    .checked_add(1)
                    .ok_or_else(|| "Max6 exceedance scanned pair count overflow".to_string())?;
            }
        }
        Ok(())
    };

    notify_unit_progress(0, tasks.len())?;
    for task_start in (0..tasks.len()).step_by(worker_count) {
        let task_end = (task_start + worker_count).min(tasks.len());
        if let Some(pool) = pool {
            pool.install(|| {
                worker_accumulators.par_iter_mut().enumerate().try_for_each(
                    |(worker_index, accumulator)| {
                        let task_index = task_start + worker_index;
                        if task_index < task_end {
                            scan_anchor_task(accumulator, tasks[task_index])?;
                        }
                        Ok::<(), String>(())
                    },
                )
            })?;
        } else {
            scan_anchor_task(&mut worker_accumulators[0], tasks[task_start])?;
        }
        notify_unit_progress(task_end, tasks.len())?;
    }

    let reserved_bytes = worker_accumulators
        .iter()
        .map(Max6ScreenExceedanceAccumulator::reserved_bytes)
        .sum::<usize>();
    let mut accumulator_iter = worker_accumulators.into_iter();
    let mut total = accumulator_iter
        .next()
        .ok_or_else(|| "Max6 exceedance scan produced no worker accumulator".to_string())?;
    for worker in accumulator_iter {
        total.merge(worker)?;
    }
    let (full_counts, sampled_counts, top_pairs, unique_pair_count) = total.into_counts()?;
    Ok(sampled_screen::WeightedExceedanceFullScan {
        full_counts,
        sampled_counts,
        top_pairs,
        unique_pair_count,
        threshold_count: thresholds.len(),
        sample_slot_count: sampling_plan.sample_slot_count(),
        preflight_accumulator_bytes: preflight_bytes,
        reserved_accumulator_bytes: reserved_bytes,
        preflight_output_bytes: 0,
        reserved_output_bytes: 0,
        work_metadata_bytes,
        preflight_total_bytes: preflight_bytes.saturating_add(work_metadata_bytes),
        elapsed_seconds: 0.0,
    })
}

pub(super) fn scan_max6_screen_exceedance_with_progress(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    calibration_nulls: &[Vec<f64>],
    calibration_maxima: &[f64],
    thresholds: &[sampled_screen::ExceedanceAuditThreshold],
    sampling_plan: &sampled_screen::WeightedExceedanceSamplingPlan,
    pool: Option<&rayon::ThreadPool>,
    mut notify_progress: impl FnMut(usize, usize, usize, usize) -> Result<(), String>,
) -> Result<sampled_screen::WeightedExceedanceFullScan, String> {
    let started = Instant::now();
    if calibration_nulls.is_empty() || calibration_nulls.len() != calibration_maxima.len() {
        return Err(format!(
            "weighted exceedance audit requires the paired full-MaxT calibration nulls/maxima (nulls={}, maxima={})",
            calibration_nulls.len(),
            calibration_maxima.len()
        ));
    }
    let full_cells = calibration_nulls
        .len()
        .checked_mul(5)
        .and_then(|value| value.checked_mul(thresholds.len()))
        .ok_or_else(|| "retained full exceedance count size overflow".to_string())?;
    let sampled_cells = calibration_nulls
        .len()
        .checked_mul(sampling_plan.sample_slot_count())
        .and_then(|value| value.checked_mul(5))
        .and_then(|value| value.checked_mul(thresholds.len()))
        .ok_or_else(|| "retained sampled exceedance count size overflow".to_string())?;
    let tasks_per_scan = max6_screen_exceedance_anchor_task_count(universe)?.max(1);
    let progress_total = calibration_nulls
        .len()
        .checked_mul(tasks_per_scan)
        .ok_or_else(|| "exceedance progress work size overflow".to_string())?;
    let max_batch_draw_count = calibration_nulls.len().min(MAX6_SCREEN_NULL_BATCH_SIZE);
    let worker_count = pool
        .map(rayon::ThreadPool::current_num_threads)
        .unwrap_or(1)
        .min(tasks_per_scan.max(1));
    let preflight_accumulator_bytes = max6_screen_exceedance_accumulator_preflight_bytes(
        max_batch_draw_count,
        thresholds.len(),
        sampling_plan.sample_slot_count(),
        worker_count,
    )?;
    let preflight_output_bytes = full_cells
        .checked_add(sampled_cells)
        .and_then(|cells| cells.checked_mul(std::mem::size_of::<u32>()))
        .and_then(|counts| {
            calibration_nulls
                .len()
                .checked_mul(sampled_screen::TAIL_AUDIT_TOP_K)
                .and_then(|entries| {
                    entries.checked_mul(std::mem::size_of::<sampled_screen::TailPairScore>())
                })
                .and_then(|top| counts.checked_add(top))
        })
        .ok_or_else(|| "retained exceedance output byte preflight overflow".to_string())?;
    let work_metadata_bytes = tasks_per_scan
        .checked_mul(std::mem::size_of::<(usize, usize, usize)>())
        .ok_or_else(|| "exceedance task metadata byte preflight overflow".to_string())?;
    let preflight_total_bytes = preflight_output_bytes
        .checked_add(preflight_accumulator_bytes)
        .and_then(|bytes| bytes.checked_add(work_metadata_bytes))
        .ok_or_else(|| "total exceedance audit memory preflight overflow".to_string())?;
    notify_progress(
        0,
        progress_total,
        sampling_plan.unique_universe_count,
        preflight_total_bytes,
    )?;
    let mut full_counts = Vec::new();
    full_counts
        .try_reserve_exact(full_cells)
        .map_err(|error| format!("unable to reserve retained full exceedance counts: {error}"))?;
    let mut sampled_counts = Vec::new();
    sampled_counts
        .try_reserve_exact(sampled_cells)
        .map_err(|error| {
            format!("unable to reserve retained sampled exceedance counts: {error}")
        })?;
    let mut top_pairs = Vec::new();
    top_pairs
        .try_reserve_exact(calibration_nulls.len())
        .map_err(|error| format!("unable to reserve Top-K null results: {error}"))?;
    let mut unique_pair_count = None;
    let mut preflight_accumulator_bytes = 0usize;
    let mut reserved_accumulator_bytes = 0usize;
    for (batch_index, null_batch) in calibration_nulls
        .chunks(MAX6_SCREEN_NULL_BATCH_SIZE)
        .enumerate()
    {
        let scan = scan_max6_screen_exceedance_batch(
            logic_bits,
            universe,
            null_batch,
            thresholds,
            sampling_plan,
            pool,
            |done_units, total_units| {
                if total_units != tasks_per_scan {
                    return Err(format!(
                        "exceedance progress anchor task count {total_units} != {tasks_per_scan}"
                    ));
                }
                let completed_nulls = batch_index * MAX6_SCREEN_NULL_BATCH_SIZE;
                let done = completed_nulls
                    .checked_mul(tasks_per_scan)
                    .and_then(|value| {
                        done_units
                            .checked_mul(null_batch.len())
                            .and_then(|batch_work| value.checked_add(batch_work))
                    })
                    .ok_or_else(|| "exceedance progress count overflow".to_string())?;
                notify_progress(
                    done,
                    progress_total,
                    sampling_plan.unique_universe_count,
                    preflight_total_bytes,
                )
            },
        )?;
        if unique_pair_count.is_some_and(|previous| previous != scan.unique_pair_count) {
            return Err("Max6 exceedance pair count changed between null batches".to_string());
        }
        unique_pair_count = Some(scan.unique_pair_count);
        preflight_accumulator_bytes =
            preflight_accumulator_bytes.max(scan.preflight_accumulator_bytes);
        reserved_accumulator_bytes =
            reserved_accumulator_bytes.max(scan.reserved_accumulator_bytes);
        full_counts.extend(scan.full_counts);
        sampled_counts.extend(scan.sampled_counts);
        top_pairs.extend(scan.top_pairs);
        let completed_nulls =
            ((batch_index + 1) * MAX6_SCREEN_NULL_BATCH_SIZE).min(calibration_nulls.len());
        notify_progress(
            completed_nulls * tasks_per_scan,
            progress_total,
            scan.unique_pair_count,
            preflight_total_bytes,
        )?;
    }
    if unique_pair_count != Some(sampling_plan.unique_universe_count) {
        return Err(format!(
            "weighted exceedance scan observed {} pairs, sample plan universe has {}",
            unique_pair_count.unwrap_or(0),
            sampling_plan.unique_universe_count
        ));
    }
    let reserved_output_bytes = full_counts
        .capacity()
        .saturating_mul(std::mem::size_of::<u32>())
        .saturating_add(
            sampled_counts
                .capacity()
                .saturating_mul(std::mem::size_of::<u32>()),
        )
        .saturating_add(
            top_pairs
                .iter()
                .map(|pairs| {
                    pairs
                        .capacity()
                        .saturating_mul(std::mem::size_of::<sampled_screen::TailPairScore>())
                })
                .sum::<usize>(),
        );
    Ok(sampled_screen::WeightedExceedanceFullScan {
        full_counts,
        sampled_counts,
        top_pairs,
        unique_pair_count: unique_pair_count.unwrap_or(0),
        threshold_count: thresholds.len(),
        sample_slot_count: sampling_plan.sample_slot_count(),
        preflight_accumulator_bytes,
        reserved_accumulator_bytes,
        preflight_output_bytes,
        reserved_output_bytes,
        work_metadata_bytes,
        preflight_total_bytes,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

pub(super) fn scan_max6_screen_tail_with_progress(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    null_draws: &[Vec<f64>],
    sampled_winners: &[Vec<Option<sampled_screen::TailPairScore>>],
    pool: Option<&rayon::ThreadPool>,
    mut notify_progress: impl FnMut(usize, usize, usize) -> Result<(), String>,
) -> Result<sampled_screen::TailAuditFullScan, String> {
    let started = Instant::now();
    if null_draws.is_empty()
        || sampled_winners
            .iter()
            .any(|target| target.len() != null_draws.len())
    {
        return Err(
            "Max6 full tail audit requires paired sampled maxima for every null draw".to_string(),
        );
    }
    let mut all_top_pairs = Vec::with_capacity(null_draws.len());
    let mut all_sampled_full_ranks = Vec::with_capacity(null_draws.len());
    let mut pair_count = None;
    for (batch_index, null_batch) in null_draws.chunks(MAX6_SCREEN_NULL_BATCH_SIZE).enumerate() {
        let start = batch_index * MAX6_SCREEN_NULL_BATCH_SIZE;
        let end = start + null_batch.len();
        let batch_sampled_winners = sampled_winners
            .iter()
            .map(|target| target[start..end].to_vec())
            .collect::<Vec<_>>();
        let (batch_top, batch_ranks, batch_count) = scan_max6_screen_tail_batch(
            logic_bits,
            universe,
            null_batch,
            batch_sampled_winners.as_slice(),
            pool,
        )?;
        if pair_count.is_some_and(|previous| previous != batch_count) {
            return Err("Max6 tail scanner pair count changed between null batches".to_string());
        }
        pair_count = Some(batch_count);
        all_top_pairs.extend(batch_top);
        all_sampled_full_ranks.extend(batch_ranks);
        notify_progress(all_top_pairs.len(), null_draws.len(), batch_count)?;
    }
    Ok(sampled_screen::TailAuditFullScan {
        top_pairs: all_top_pairs,
        sampled_full_ranks: all_sampled_full_ranks,
        unique_pair_count: pair_count.unwrap_or(0),
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

pub(super) fn scan_max6_screen_null_maxima_with_progress(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    null_residualized_y: &[Vec<f64>],
    pool: Option<&rayon::ThreadPool>,
    track_maximizing_pairs: bool,
    mut notify_progress: impl FnMut(usize, usize, usize) -> Result<(), String>,
) -> Result<(Vec<f64>, Option<Vec<Option<(usize, usize)>>>, usize), String> {
    if null_residualized_y.is_empty() {
        return Err("Max6 screen null scan requires at least one null replicate".to_string());
    }
    let total = null_residualized_y.len();
    notify_progress(0, total, 0)?;
    // Report progress at a useful cadence without imposing a callback on each
    // draw. Replicates remain in seed order; each batch shares pair traversal,
    // while Rayon distributes disjoint prepared units.
    let progress_batch = MAX6_SCREEN_NULL_BATCH_SIZE;
    let mut maxima = Vec::with_capacity(total);
    let mut maximizing_pairs = track_maximizing_pairs.then(|| Vec::with_capacity(total));
    let mut unique_pair_count = None::<usize>;
    for batch in null_residualized_y.chunks(progress_batch) {
        let (batch_maxima, batch_pairs, batch_unique_pairs) =
            scan_max6_screen_null_batch(logic_bits, universe, batch, pool, track_maximizing_pairs)?;
        if let Some(previous) = unique_pair_count {
            if previous != batch_unique_pairs {
                return Err(format!(
                    "Max6 unique-pair count changed between null batches: previous={previous}, current={batch_unique_pairs}"
                ));
            }
        } else {
            unique_pair_count = Some(batch_unique_pairs);
        }
        maxima.extend(batch_maxima);
        if let (Some(all_pairs), Some(batch_pairs)) = (maximizing_pairs.as_mut(), batch_pairs) {
            all_pairs.extend(batch_pairs);
        }
        notify_progress(maxima.len(), total, batch_unique_pairs)?;
    }
    Ok((maxima, maximizing_pairs, unique_pair_count.unwrap_or(0)))
}

/// Calibrate the production Max6 screen on fitted conditional parametric-null
/// residual phenotypes. This deliberately calibrates only the screen score:
/// the downstream GRM-Max6 exact reranker is a subset-ranking stage and does
/// not receive a second FWER interpretation. Overlapping windows are
/// canonicalized by deterministic earliest-eligible-unit ownership, so the
/// max is computed over the union of unique global marker pairs, exactly
/// equivalent to rescoring duplicates because Max6 pair scores are window
/// invariant. Null draws are processed in batches to share each pair traversal.
pub(super) fn calibrate_max6_screen_threshold(
    logic_bits: &GarfieldLogicBits,
    universe: &Max6ScreenPreparedUniverse,
    null_residualized_y: &[Vec<f64>],
    quantile: f64,
    pool: Option<&rayon::ThreadPool>,
    track_maximizing_pairs: bool,
    notify_progress: impl FnMut(usize, usize, usize) -> Result<(), String>,
) -> Result<Max6ScreenCalibration, String> {
    if null_residualized_y.is_empty() {
        return Err(
            "Max6 parametric screen calibration requires at least one fitted null replicate"
                .to_string(),
        );
    }
    if !quantile.is_finite() || !(0.0..=1.0).contains(&quantile) {
        return Err(format!(
            "Max6 parametric screen calibration quantile must be in [0, 1], got {quantile}"
        ));
    }

    let (maxima, maximizing_pairs, pair_universe_size) =
        scan_max6_screen_null_maxima_with_progress(
            logic_bits,
            universe,
            null_residualized_y,
            pool,
            track_maximizing_pairs,
            notify_progress,
        )?;
    let empirical_threshold =
        empirical_score_quantile(maxima.as_slice(), quantile).ok_or_else(|| {
            "Max6 parametric screen calibration produced no finite maxima".to_string()
        })?;
    let thresholds = max6_permutation_threshold_candidates(maxima.as_slice(), quantile);
    Ok(Max6ScreenCalibration {
        maxima,
        maximizing_pairs,
        empirical_threshold,
        gumbel_threshold: thresholds.gumbel,
        pair_universe_size,
    })
}

pub(super) fn max6_grmmax6_permutation_max_score(
    job: &Max6GrmMax6PermutationJob,
    logic_bits: &GarfieldLogicBits,
    p_c: Option<&[f64]>,
    no_grm_projector: Option<&NoGrmCovariateProjector>,
    score_scale: f64,
) -> Result<Max6PermutationJobResult, String> {
    let mut timing = Max6PermutationJobTiming::default();
    let projected_y_started = Instant::now();
    let projected_y = if let Some(p_c) = p_c {
        project_phenotype_for_candidate_rerank(p_c, logic_bits.n_samples, job.permuted_y.as_slice())
    } else {
        no_grm_projector
            .ok_or_else(|| "Max6 permutation has no projection representation".to_string())?
            .project(job.permuted_y.as_slice())?
    };
    timing.projected_y_seconds = projected_y_started.elapsed().as_secs_f64();
    let mut max_score = f64::NEG_INFINITY;
    for (first, second) in job.grouped.values() {
        let endpoint_prepare_started = Instant::now();
        let candidate_pairs = first
            .iter()
            .copied()
            .zip(second.iter().copied())
            .collect::<Vec<_>>();
        let (x, pair_i, pair_j, endpoint_rows) = dense_candidate_endpoint_matrix_global(
            logic_bits.bits(),
            logic_bits.row_words,
            logic_bits.n_samples,
            candidate_pairs.as_slice(),
        )?;
        let candidate_first = pair_i.iter().map(|&index| index as i64).collect::<Vec<_>>();
        let candidate_second = pair_j.iter().map(|&index| index as i64).collect::<Vec<_>>();
        timing.endpoint_prepare_seconds += endpoint_prepare_started.elapsed().as_secs_f64();
        timing.pair_scores = timing.pair_scores.saturating_add(candidate_first.len());
        let result = if let Some(p_c) = p_c {
            candidate_rerank_exact_with_py_trusted_scaled(
                x.as_slice(),
                logic_bits.n_samples,
                endpoint_rows.len(),
                p_c,
                job.permuted_y.as_slice(),
                projected_y.as_slice(),
                candidate_first.as_slice(),
                candidate_second.as_slice(),
                first.len(),
                DosageChannel::FwlProduct,
                score_scale,
            )?
        } else {
            let projector = no_grm_projector.ok_or_else(|| {
                "Max6 permutation has no-GRM projection representation".to_string()
            })?;
            let packed = gather_rows_by_indices(
                logic_bits.bits(),
                logic_bits.row_words,
                endpoint_rows.as_slice(),
                "max6 no-GRM permutation endpoint rows",
            )?;
            candidate_rerank_exact_packed_no_grm_covariate_with_py_trusted_scaled(
                packed.as_slice(),
                logic_bits.row_words,
                logic_bits.n_samples,
                endpoint_rows.len(),
                projector,
                job.permuted_y.as_slice(),
                projected_y.as_slice(),
                candidate_first.as_slice(),
                candidate_second.as_slice(),
                first.len(),
                DosageChannel::FwlProduct,
                score_scale,
            )?
        };
        timing.rerank.pcx_u_seconds += result.timing.pcx_u_seconds;
        timing.rerank.py_seconds += result.timing.py_seconds;
        timing.rerank.candidate_geometry_seconds += result.timing.candidate_geometry_seconds;
        timing.rerank.q_seconds += result.timing.q_seconds;
        timing.rerank.score_seconds += result.timing.score_seconds;
        timing.rerank.topk_seconds += result.timing.topk_seconds;
        timing.rerank.kernel_seconds += result.timing.kernel_seconds;
        if let Some(value) = result
            .max6
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .max_by(f64::total_cmp)
        {
            max_score = max_score.max(value);
        }
    }
    Ok(Max6PermutationJobResult {
        max_score: max_score.is_finite().then_some(max_score),
        timing,
    })
}

pub(super) fn run_max6_permutation_jobs_batched<T, R, F, P>(
    jobs: &[T],
    pool: Option<&rayon::ThreadPool>,
    score_job: F,
    mut notify_progress: P,
) -> Result<Vec<R>, String>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> Result<R, String> + Send + Sync,
    P: FnMut(usize, usize) -> Result<(), String>,
{
    let total = jobs.len();
    notify_progress(0, total)?;
    if total == 0 {
        return Ok(Vec::new());
    }

    // Keep each batch large enough to use the pool, but report at most about
    // 100 progress updates for large permutation runs. The callback executes
    // after each parallel batch on the caller thread, never in a Rayon worker.
    let parallelism = pool.map_or(1, rayon::ThreadPool::current_num_threads);
    let batch_size = parallelism.max(total.div_ceil(100)).min(total);
    let mut results = Vec::with_capacity(total);
    for batch in jobs.chunks(batch_size) {
        let batch_results = if let Some(pool) = pool {
            pool.install(|| {
                batch
                    .par_iter()
                    .map(&score_job)
                    .collect::<Result<Vec<_>, String>>()
            })?
        } else {
            batch
                .iter()
                .map(&score_job)
                .collect::<Result<Vec<_>, String>>()?
        };
        results.extend(batch_results);
        notify_progress(results.len(), total)?;
    }
    Ok(results)
}

#[inline]
pub(super) fn unrank_upper_pair(mut ordinal: usize, n_markers: usize) -> (usize, usize) {
    for first in 0..n_markers.saturating_sub(1) {
        let width = n_markers - first - 1;
        if ordinal < width {
            return (first, first + ordinal + 1);
        }
        ordinal -= width;
    }
    (n_markers.saturating_sub(2), n_markers.saturating_sub(1))
}

#[inline]
pub(super) fn empirical_score_quantile(scores: &[f64], quantile: f64) -> Option<f64> {
    let mut finite = scores
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(f64::total_cmp);
    let q = if quantile.is_finite() {
        quantile.clamp(0.0, 1.0)
    } else {
        0.99
    };
    let index = ((q * finite.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(finite.len() - 1);
    finite.get(index).copied()
}

pub(super) fn max6_permutation_threshold_candidates(
    scores: &[f64],
    quantile: f64,
) -> Max6PermutationThresholdCandidates {
    let quantile = if quantile.is_finite() {
        quantile.clamp(0.0, 1.0)
    } else {
        0.99
    };
    Max6PermutationThresholdCandidates {
        empirical: empirical_score_quantile(scores, quantile),
        gumbel: gumbel_penalty_from_maxima(scores, 1.0 - quantile),
    }
}
