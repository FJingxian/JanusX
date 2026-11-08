use super::*;

#[derive(Clone, Debug)]
pub(super) struct Max6ScreenPreparedUniverse {
    pub(super) units: Vec<Max6GrmMax6LdPreparedUnit>,
    pub(super) original_unit_indices: Vec<usize>,
    pub(super) active_rows: Vec<usize>,
    pub(super) pair_ownership: Max6ScreenPairOwnership,
    pub(super) window_pair_count: usize,
}

/// LD-selected geometry before the optional ownership index is built.
/// Genotype-global tag preparation only needs these fields; ownership is
/// built once later, after selected rows have been remapped to source IDs.
pub(super) struct Max6ScreenSelectedUnits {
    pub(super) units: Vec<Max6GrmMax6LdPreparedUnit>,
    pub(super) original_unit_indices: Vec<usize>,
    pub(super) active_rows: Vec<usize>,
    pub(super) window_pair_count: usize,
    pub(super) support_cache_build_seconds: f64,
    pub(super) support_cache_rows: usize,
    pub(super) support_cache_payload_bytes: usize,
}

pub(super) fn max6_ld_selection_fingerprint(
    units: &[Max6GrmMax6LdPreparedUnit],
    original_unit_indices: &[usize],
) -> u64 {
    pub(super) fn feed(hash: &mut u64, value: u64) {
        for byte in value.to_le_bytes() {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }

    debug_assert_eq!(units.len(), original_unit_indices.len());
    let mut hash = 0xcbf2_9ce4_8422_2325;
    feed(&mut hash, units.len() as u64);
    for (unit, &original_unit_index) in units.iter().zip(original_unit_indices) {
        feed(&mut hash, original_unit_index as u64);
        feed(&mut hash, unit.selected_global_rows.len() as u64);
        for &row in &unit.selected_global_rows {
            feed(&mut hash, row as u64);
        }
        match unit.pair_groups.as_deref() {
            Some(groups) => {
                feed(&mut hash, 1);
                feed(&mut hash, groups.len() as u64);
                for &group in groups {
                    feed(&mut hash, group as u64);
                }
            }
            None => feed(&mut hash, 0),
        }
    }
    hash
}

/// Compact CSR index used to assign each globally canonical pair to the
/// earliest prepared unit in which it is eligible. Pair scores are invariant
/// to window membership, so this represents the union of eligible pairs
/// without materializing a potentially multi-billion-element pair vector.
#[derive(Clone, Debug)]
pub(super) struct Max6ScreenPairOwnership {
    pub(super) global_row_start: usize,
    pub(super) global_to_active: Vec<u32>,
    pub(super) single_membership_units: Vec<u32>,
    pub(super) offsets: Vec<u32>,
    pub(super) membership_units: Vec<u32>,
    pub(super) membership_groups: Option<Vec<u32>>,
}

impl Max6ScreenPairOwnership {
    pub(super) const NO_GROUP: u32 = u32::MAX;

    pub(super) fn build(
        units: &[Max6GrmMax6LdPreparedUnit],
        active_rows: &[usize],
        n_global_rows: usize,
    ) -> Result<Self, String> {
        if active_rows.len() > u32::MAX as usize {
            return Err("Max6 unique-pair ownership exceeds u32 active-row capacity".to_string());
        }
        if units.len() > u32::MAX as usize {
            return Err("Max6 unique-pair ownership exceeds u32 unit capacity".to_string());
        }
        if active_rows
            .iter()
            .any(|&global_row| global_row >= n_global_rows)
        {
            return Err("Max6 active marker row is out of range".to_string());
        }
        let global_row_start = active_rows.first().copied().unwrap_or(0);
        let global_row_end = active_rows.last().copied().unwrap_or(global_row_start);
        let global_row_span = if active_rows.is_empty() {
            0
        } else {
            global_row_end
                .checked_sub(global_row_start)
                .and_then(|span| span.checked_add(1))
                .ok_or_else(|| "Max6 active marker row span overflow".to_string())?
        };
        let mut global_to_active = vec![u32::MAX; global_row_span];
        for (slot, &global_row) in active_rows.iter().enumerate() {
            let local_row = global_row.saturating_sub(global_row_start);
            let target = global_to_active
                .get_mut(local_row)
                .ok_or_else(|| format!("Max6 active marker row {global_row} is out of range"))?;
            *target = slot as u32;
        }

        let mut counts = vec![0u32; active_rows.len()];
        let mut single_membership_units = vec![Self::NO_GROUP; active_rows.len()];
        let has_groups = units.iter().any(|unit| unit.pair_groups.is_some());
        for (unit_index, unit) in units.iter().enumerate() {
            if unit
                .pair_groups
                .as_ref()
                .is_some_and(|groups| groups.len() != unit.selected_global_rows.len())
            {
                return Err(
                    "Max6 pair-group length does not match selected marker rows".to_string()
                );
            }
            for &global_row in unit.selected_global_rows.iter() {
                let local_row = global_row.saturating_sub(global_row_start);
                let slot = *global_to_active.get(local_row).ok_or_else(|| {
                    format!("Max6 selected marker row {global_row} is out of range")
                })?;
                if slot == u32::MAX {
                    return Err(format!(
                        "Max6 selected marker row {global_row} is missing from active rows"
                    ));
                }
                let slot = slot as usize;
                counts[slot] = counts[slot]
                    .checked_add(1)
                    .ok_or_else(|| "Max6 marker membership count overflow".to_string())?;
                if counts[slot] == 1 {
                    single_membership_units[slot] = unit_index as u32;
                } else {
                    single_membership_units[slot] = Self::NO_GROUP;
                }
            }
        }
        if counts.iter().all(|&count| count <= 1) {
            return Ok(Self {
                global_row_start,
                global_to_active,
                single_membership_units,
                offsets: vec![0u32; 1],
                membership_units: Vec::new(),
                membership_groups: None,
            });
        }

        let mut offsets = vec![0u32; active_rows.len() + 1];
        let mut total = 0u64;
        for (slot, &count) in counts.iter().enumerate() {
            total = total.saturating_add(u64::from(count));
            if total > u64::from(u32::MAX) {
                return Err(
                    "Max6 unique-pair ownership exceeds u32 membership capacity".to_string()
                );
            }
            offsets[slot + 1] = total as u32;
        }
        let mut membership_units = vec![0u32; total as usize];
        let mut membership_groups = has_groups.then(|| vec![Self::NO_GROUP; total as usize]);
        let mut cursor = offsets[..active_rows.len()].to_vec();
        for (unit_index, unit) in units.iter().enumerate() {
            for (local_index, &global_row) in unit.selected_global_rows.iter().enumerate() {
                let local_row = global_row.saturating_sub(global_row_start);
                let slot = global_to_active[local_row] as usize;
                let output_index = cursor[slot] as usize;
                membership_units[output_index] = unit_index as u32;
                if let (Some(dst), Some(groups)) =
                    (membership_groups.as_mut(), unit.pair_groups.as_ref())
                {
                    let group = groups[local_index];
                    if group >= u32::MAX as usize {
                        return Err("Max6 pair-group id exceeds u32 capacity".to_string());
                    }
                    dst[output_index] = group as u32;
                }
                cursor[slot] += 1;
            }
        }
        Ok(Self {
            global_row_start,
            global_to_active,
            single_membership_units,
            offsets,
            membership_units,
            membership_groups,
        })
    }

    #[inline]
    pub(super) fn active_slot(&self, global_row: usize) -> usize {
        self.global_to_active[global_row.saturating_sub(self.global_row_start)] as usize
    }

    #[inline]
    pub(super) fn pair_owner(&self, first_slot: usize, second_slot: usize) -> Option<usize> {
        let mut first_idx = self.offsets[first_slot] as usize;
        let first_end = self.offsets[first_slot + 1] as usize;
        let mut second_idx = self.offsets[second_slot] as usize;
        let second_end = self.offsets[second_slot + 1] as usize;
        while first_idx < first_end && second_idx < second_end {
            let first_unit = self.membership_units[first_idx];
            let second_unit = self.membership_units[second_idx];
            if first_unit < second_unit {
                first_idx += 1;
            } else if second_unit < first_unit {
                second_idx += 1;
            } else {
                let eligible = self.membership_groups.as_ref().is_none_or(|groups| {
                    let first_group = groups[first_idx];
                    let second_group = groups[second_idx];
                    first_group == Self::NO_GROUP
                        || second_group == Self::NO_GROUP
                        || first_group != second_group
                });
                if eligible {
                    return Some(first_unit as usize);
                }
                first_idx += 1;
                second_idx += 1;
            }
        }
        None
    }
}

/// Build output-ordered, memory-bounded Max6 screen batches from the actual
/// prepared tag membership. The batch scorer itself decides whether any
/// pair is eligible for sharing; rows are never inferred from coordinates.
pub(super) const MAX6_MEMBERSHIP_BATCH_MAX_UNITS: usize = 32;
pub(super) const MAX6_MEMBERSHIP_BATCH_MAX_MEMBERSHIPS: usize = 131_072;
pub(super) const MAX6_MEMBERSHIP_BATCH_MAX_UNIQUE_ROWS: usize = 65_536;
pub(super) const MAX6_MEMBERSHIP_BATCH_HEAP_ENTRIES_PER_WORKER: usize = 32_768;

#[inline]
pub(super) fn max6_membership_scan_batch_unit_cap(top_k: usize) -> usize {
    (MAX6_MEMBERSHIP_BATCH_HEAP_ENTRIES_PER_WORKER / top_k.max(1))
        .max(1)
        .min(MAX6_MEMBERSHIP_BATCH_MAX_UNITS)
}

pub(super) fn max6_membership_scan_task_chunks(
    scan_unit_indices: &[usize],
    units: &[GarfieldLogicUnit],
    prepared: &Max6ScreenPreparedUniverse,
    threads: usize,
    top_k: usize,
) -> Option<Vec<Vec<usize>>> {
    if scan_unit_indices.len() < 2
        || prepared.units.len() != prepared.original_unit_indices.len()
        || prepared
            .active_rows
            .windows(2)
            .any(|rows| rows[0] >= rows[1])
    {
        return None;
    }
    let prepared_slot_by_original = prepared
        .original_unit_indices
        .iter()
        .enumerate()
        .map(|(slot, &original)| (original, slot))
        .collect::<HashMap<_, _>>();
    let prepared_units = scan_unit_indices
        .iter()
        .map(|unit_index| {
            units.get(*unit_index)?;
            let prepared_slot = *prepared_slot_by_original.get(unit_index)?;
            let prepared_unit = prepared.units.get(prepared_slot)?;
            Some((*unit_index, prepared_unit.selected_global_rows.as_slice()))
        })
        .collect::<Option<Vec<_>>>()?;

    // Score reuse happens within a screen batch. Group enough windows to
    // expose parallel unique-pair jobs while the row and membership budgets
    // continue to bound memory.
    let max_batch_units = max6_membership_scan_batch_unit_cap(top_k);
    let target_tasks = threads
        .max(1)
        .saturating_mul(2)
        .min(scan_unit_indices.len());
    let total_work = prepared_units.iter().fold(0usize, |sum, (_, rows)| {
        sum.saturating_add(rows.len().max(1))
    });
    let target_work = total_work.div_ceil(target_tasks.max(1)).max(1);

    let union_rows = |left: &[usize], right: &[usize]| -> Option<Vec<usize>> {
        let mut merged = Vec::with_capacity(
            (left.len() + right.len()).min(MAX6_MEMBERSHIP_BATCH_MAX_UNIQUE_ROWS),
        );
        let (mut i, mut j) = (0usize, 0usize);
        while i < left.len() || j < right.len() {
            let value = if j == right.len() || (i < left.len() && left[i] < right[j]) {
                let value = left[i];
                i += 1;
                value
            } else if i == left.len() || right[j] < left[i] {
                let value = right[j];
                j += 1;
                value
            } else {
                let value = left[i];
                i += 1;
                j += 1;
                value
            };
            if merged.last().copied() != Some(value) {
                if merged.len() == MAX6_MEMBERSHIP_BATCH_MAX_UNIQUE_ROWS {
                    return None;
                }
                merged.push(value);
            }
        }
        Some(merged)
    };

    let mut chunks = Vec::<Vec<usize>>::new();
    let mut current_units = Vec::<usize>::new();
    let mut current_union = Vec::<usize>::new();
    let mut current_memberships = 0usize;
    let mut current_work = 0usize;
    for &(unit_index, selected_rows) in &prepared_units {
        let too_large_alone = selected_rows.len() > MAX6_MEMBERSHIP_BATCH_MAX_MEMBERSHIPS
            || selected_rows.len() > MAX6_MEMBERSHIP_BATCH_MAX_UNIQUE_ROWS;
        let work_target_reached = current_units.len() >= 2
            && current_work >= target_work
            && chunks.len() + 1 < target_tasks;
        let proposed_union = if too_large_alone {
            None
        } else if current_units.is_empty() {
            Some(selected_rows.to_vec())
        } else {
            union_rows(current_union.as_slice(), selected_rows)
        };
        let proposed_memberships = current_memberships.saturating_add(selected_rows.len());
        let over_budget = proposed_union.is_none()
            || proposed_memberships > MAX6_MEMBERSHIP_BATCH_MAX_MEMBERSHIPS
            || current_units.len() >= max_batch_units
            || work_target_reached;
        if !current_units.is_empty() && over_budget {
            chunks.push(std::mem::take(&mut current_units));
            current_union.clear();
            current_memberships = 0;
            current_work = 0;
        }

        if too_large_alone {
            chunks.push(vec![unit_index]);
            continue;
        }
        let merged = if current_units.is_empty() {
            Some(selected_rows.to_vec())
        } else {
            union_rows(current_union.as_slice(), selected_rows)
        }?;
        current_units.push(unit_index);
        current_union = merged;
        current_memberships = current_memberships.saturating_add(selected_rows.len());
        current_work = current_work.saturating_add(selected_rows.len().max(1));
    }
    if !current_units.is_empty() {
        chunks.push(current_units);
    }
    (chunks.iter().map(Vec::len).sum::<usize>() == scan_unit_indices.len()).then_some(chunks)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Max6ScreenOwnedPair {
    pub(super) first_global_row: usize,
    pub(super) second_global_row: usize,
    pub(super) first_active_slot: usize,
    pub(super) second_active_slot: usize,
}

/// Shared eligibility/canonical-owner predicate for both the exhaustive
/// full-maxT scanner and the diagnostic-only sampled-pair selector.
#[inline]
pub(super) fn max6_screen_owned_pair(
    universe: &Max6ScreenPreparedUniverse,
    unit_index: usize,
    unit: &Max6GrmMax6LdPreparedUnit,
    first_local: usize,
    second_local: usize,
) -> Result<Option<Max6ScreenOwnedPair>, String> {
    let rows = unit.selected_global_rows.as_slice();
    let groups = unit.pair_groups.as_deref();
    if groups.is_some_and(|groups| groups[first_local] == groups[second_local]) {
        return Ok(None);
    }
    let first_global_row = rows[first_local];
    let second_global_row = rows[second_local];
    let first_active_slot = universe.pair_ownership.active_slot(first_global_row);
    let second_active_slot = universe.pair_ownership.active_slot(second_global_row);
    if !universe.pair_ownership.membership_units.is_empty() {
        let owner = universe
            .pair_ownership
            .pair_owner(first_active_slot, second_active_slot)
            .ok_or_else(|| {
                format!(
                    "eligible Max6 pair ({first_global_row}, {second_global_row}) has no canonical owner"
                )
            })?;
        if owner != unit_index {
            return Ok(None);
        }
    }
    Ok(Some(Max6ScreenOwnedPair {
        first_global_row,
        second_global_row,
        first_active_slot,
        second_active_slot,
    }))
}

pub(super) fn count_max6_screen_pairs(
    universe: &Max6ScreenPreparedUniverse,
) -> Result<usize, String> {
    count_max6_screen_pairs_by_signature(universe)
}

pub(super) fn count_max6_screen_pairs_with_pool(
    universe: &Max6ScreenPreparedUniverse,
    pool: Option<&rayon::ThreadPool>,
) -> Result<usize, String> {
    let _ = pool;
    count_max6_screen_pairs_by_signature(universe)
}

pub(super) fn count_max6_screen_pairs_with_progress(
    universe: &Max6ScreenPreparedUniverse,
    pool: Option<&rayon::ThreadPool>,
    progress: &mut dyn FnMut(usize, usize, usize) -> Result<(), String>,
) -> Result<usize, String> {
    let _ = pool;
    progress(0, 1, 0)?;
    let count = count_max6_screen_pairs_by_signature(universe)?;
    progress(1, 1, count)?;
    Ok(count)
}

pub(super) const MAX6_PAIR_SIGNATURE_SUBSET_WORK: usize = 50_000_000;
pub(super) const MAX6_PAIR_SIGNATURE_CLASS_PAIR_WORK: usize = 50_000_000;

pub(super) fn count_choose_two_u128(n: u128) -> Result<u128, String> {
    n.checked_mul(n.saturating_sub(1))
        .map(|product| product / 2)
        .ok_or_else(|| "Max6 unique-pair count overflow in choose-two".to_string())
}

pub(super) fn checked_i128_from_u128(value: u128, context: &str) -> Result<i128, String> {
    i128::try_from(value).map_err(|_| {
        format!("Max6 unique-pair count exceeds signed accumulation range ({context})")
    })
}

pub(super) fn max6_signature_pair_eligible(left: &[(u32, u32)], right: &[(u32, u32)]) -> bool {
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.len() && j < right.len() {
        match left[i].0.cmp(&right[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                let left_group = left[i].1;
                let right_group = right[j].1;
                if left_group == Max6ScreenPairOwnership::NO_GROUP
                    || right_group == Max6ScreenPairOwnership::NO_GROUP
                    || left_group != right_group
                {
                    return true;
                }
                i += 1;
                j += 1;
            }
        }
    }
    false
}

pub(super) fn count_grouped_max6_signature_pairs(
    signatures: &HashMap<Vec<(u32, u32)>, u128>,
) -> Result<(u128, usize), String> {
    let class_count = signatures.len();
    let pair_work = class_count
        .checked_mul(class_count.saturating_add(1))
        .map(|value| value / 2)
        .ok_or_else(|| "Max6 signature-class pair work overflow".to_string())?;
    let mut ordered = signatures.iter().collect::<Vec<_>>();
    ordered.sort_by(|(left, _), (right, _)| left.cmp(right));
    if pair_work > MAX6_PAIR_SIGNATURE_CLASS_PAIR_WORK {
        return count_grouped_max6_signature_pairs_bitset(&ordered);
    }
    let mut total = 0u128;
    let mut work = 0usize;
    for left_idx in 0..ordered.len() {
        let (left_signature, left_count) = ordered[left_idx];
        for (right_idx, (right_signature, right_count)) in ordered.iter().enumerate().skip(left_idx)
        {
            work += 1;
            if !max6_signature_pair_eligible(left_signature, right_signature) {
                continue;
            }
            let contribution = if left_idx == right_idx {
                count_choose_two_u128(*left_count)?
            } else {
                (*left_count).checked_mul(**right_count).ok_or_else(|| {
                    "Max6 unique-pair count overflow across signatures".to_string()
                })?
            };
            total = total
                .checked_add(contribution)
                .ok_or_else(|| "Max6 unique-pair count overflow across signatures".to_string())?;
        }
    }
    Ok((total, work))
}

/// Count compatible signature-class pairs without enumerating class pairs.
/// A marker has at most one group label per unit in the ownership CSR. For a
/// left class labeled `g` in unit `u`, compatible right classes are all classes
/// in `u` except those with the same `g`; an ungrouped membership is compatible
/// with every class in that unit. The per-class bitset union implements the
/// existing any-compatible-shared-unit predicate exactly.
pub(super) fn count_grouped_max6_signature_pairs_bitset(
    ordered: &[(&Vec<(u32, u32)>, &u128)],
) -> Result<(u128, usize), String> {
    let class_count = ordered.len();
    if class_count == 0 {
        return Ok((0, 0));
    }
    let word_count = class_count.div_ceil(64);
    if class_count > u32::MAX as usize {
        return Err("Max6 signature-class bitset exceeds u32 class-index capacity".to_string());
    }

    let mut class_weights = Vec::with_capacity(class_count);
    let mut unit_classes = HashMap::<u32, Vec<u64>>::new();
    let mut same_group_classes = HashMap::<(u32, u32), Vec<u32>>::new();
    for (class_index, entry) in ordered.iter().enumerate() {
        let (signature, marker_count) = *entry;
        class_weights.push(u64::try_from(*marker_count).map_err(|_| {
            "Max6 signature-class marker count exceeds u64 bitset-weight capacity".to_string()
        })?);
        if signature
            .windows(2)
            .any(|membership| membership[0].0 >= membership[1].0)
        {
            return Err(
                "Max6 bitset counting requires one strictly ordered group label per unit"
                    .to_string(),
            );
        }
        let class_bit = 1u64 << (class_index % 64);
        let class_slot = u32::try_from(class_index)
            .map_err(|_| "Max6 signature-class index exceeds u32 capacity".to_string())?;
        for &(unit, group) in signature.iter() {
            unit_classes
                .entry(unit)
                .or_insert_with(|| vec![0u64; word_count])[class_index / 64] |= class_bit;
            if group != Max6ScreenPairOwnership::NO_GROUP {
                same_group_classes
                    .entry((unit, group))
                    .or_default()
                    .push(class_slot);
            }
        }
    }

    // A byte lookup table sums class multiplicities for an arbitrary bit mask.
    // It avoids walking every set class bit for every source class when the
    // compatible-neighbor bitsets are dense.
    let byte_blocks = class_count.div_ceil(8);
    let lookup_len = byte_blocks
        .checked_mul(256)
        .ok_or_else(|| "Max6 bitset weight lookup size overflow".to_string())?;
    let mut weighted_byte_sums = vec![0u64; lookup_len];
    for block in 0..byte_blocks {
        let block_offset = block * 8;
        for mask in 1usize..256 {
            let bit = mask.trailing_zeros() as usize;
            let previous_mask = mask & (mask - 1);
            let weight = class_weights.get(block_offset + bit).copied().unwrap_or(0);
            let previous = weighted_byte_sums[block * 256 + previous_mask];
            weighted_byte_sums[block * 256 + mask] = previous
                .checked_add(weight)
                .ok_or_else(|| "Max6 bitset byte-weight sum overflow".to_string())?;
        }
    }

    let mut total = 0u128;
    let mut work = 0usize;
    let mut compatible_classes = vec![0u64; word_count];
    let mut unit_compatible_classes = vec![0u64; word_count];
    for (class_index, entry) in ordered.iter().enumerate() {
        let (signature, marker_count) = *entry;
        compatible_classes.fill(0);
        for &(unit, group) in signature.iter() {
            let unit_bits = unit_classes
                .get(&unit)
                .ok_or_else(|| format!("Max6 bitset index is missing ownership unit {unit}"))?;
            if group == Max6ScreenPairOwnership::NO_GROUP {
                for (neighbors, &members) in compatible_classes.iter_mut().zip(unit_bits.iter()) {
                    *neighbors |= members;
                }
                work = work.saturating_add(word_count);
            } else {
                unit_compatible_classes.copy_from_slice(unit_bits.as_slice());
                if let Some(same_group) = same_group_classes.get(&(unit, group)) {
                    for &same_class in same_group.iter() {
                        let same_class = same_class as usize;
                        unit_compatible_classes[same_class / 64] &= !(1u64 << (same_class % 64));
                    }
                    work = work.saturating_add(same_group.len());
                }
                for (neighbors, &members) in compatible_classes
                    .iter_mut()
                    .zip(unit_compatible_classes.iter())
                {
                    *neighbors |= members;
                }
                work = work.saturating_add(word_count.saturating_mul(2));
            }
        }

        let class_bit = 1u64 << (class_index % 64);
        if compatible_classes[class_index / 64] & class_bit != 0 {
            total = total
                .checked_add(count_choose_two_u128(*marker_count)?)
                .ok_or_else(|| "Max6 unique-pair count overflow within signature".to_string())?;
        }

        let start_class = class_index + 1;
        let first_word = start_class / 64;
        let mut right_weight_sum = 0u64;
        for (word_index, &neighbors) in compatible_classes.iter().enumerate().skip(first_word) {
            let mut byte_mask = neighbors;
            if word_index == first_word {
                let first_bit = start_class % 64;
                if first_bit != 0 {
                    byte_mask &= u64::MAX << first_bit;
                }
            }
            for byte_index in 0..8 {
                let block = word_index * 8 + byte_index;
                if block >= byte_blocks {
                    break;
                }
                let mask = (byte_mask & 0xff) as usize;
                let lookup_index = block * 256 + mask;
                right_weight_sum = right_weight_sum
                    .checked_add(weighted_byte_sums[lookup_index])
                    .ok_or_else(|| "Max6 compatible marker-weight sum overflow".to_string())?;
                byte_mask >>= 8;
            }
        }
        work = work.saturating_add(byte_blocks);
        let cross_class_pairs = marker_count
            .checked_mul(u128::from(right_weight_sum))
            .ok_or_else(|| "Max6 unique-pair count overflow across signatures".to_string())?;
        total = total
            .checked_add(cross_class_pairs)
            .ok_or_else(|| "Max6 unique-pair count overflow across signatures".to_string())?;
    }

    Ok((total, work))
}

/// Count the union of ungrouped signature-class pairs through sparse unit
/// postings. Generation marks make a class pair contribute only once even
/// when its two markers share multiple scan units.
pub(super) fn count_ungrouped_max6_signature_pairs_sparse(
    signatures: &HashMap<Vec<u32>, u128>,
) -> Result<(u128, usize), String> {
    let mut ordered = signatures.iter().collect::<Vec<_>>();
    ordered.sort_by(|(left, _), (right, _)| left.cmp(right));

    let mut unit_classes = HashMap::<u32, Vec<usize>>::new();
    for (class_index, entry) in ordered.iter().enumerate() {
        let (signature, _) = *entry;
        if signature
            .windows(2)
            .any(|membership| membership[0] >= membership[1])
        {
            return Err(
                "Max6 sparse ungrouped counting requires strictly ordered unit signatures"
                    .to_string(),
            );
        }
        for &unit in signature.iter() {
            unit_classes.entry(unit).or_default().push(class_index);
        }
    }

    let mut seen_generation = vec![usize::MAX; ordered.len()];
    let mut total = 0u128;
    let mut work = 0usize;
    for left_index in 0..ordered.len() {
        let (left_signature, left_count) = ordered[left_index];
        for &unit in left_signature.iter() {
            let classes = unit_classes
                .get(&unit)
                .ok_or_else(|| format!("Max6 sparse index is missing ownership unit {unit}"))?;
            let first_right = classes.partition_point(|&right_index| right_index < left_index);
            for &right_index in classes[first_right..].iter() {
                work = work.saturating_add(1);
                if seen_generation[right_index] == left_index {
                    continue;
                }
                seen_generation[right_index] = left_index;

                let contribution = if left_index == right_index {
                    count_choose_two_u128(*left_count)?
                } else {
                    (*left_count)
                        .checked_mul(*ordered[right_index].1)
                        .ok_or_else(|| {
                            "Max6 unique-pair count overflow across sparse signatures".to_string()
                        })?
                };
                total = total.checked_add(contribution).ok_or_else(|| {
                    "Max6 unique-pair count overflow in sparse signature union".to_string()
                })?;
            }
        }
    }

    Ok((total, work))
}

/// Count the union of pair families without walking pair occurrences. For
/// ungrouped scan units, inclusion-exclusion over the unit-membership
/// signatures is exact and scales with distinct signatures/subsets. Grouped
/// units use exact signature-class compatibility because same-group pairs
/// are excluded within a unit.
pub(super) fn count_max6_screen_pairs_by_signature(
    universe: &Max6ScreenPreparedUniverse,
) -> Result<usize, String> {
    count_max6_pairs_by_membership(
        universe.active_rows.as_slice(),
        universe.window_pair_count,
        &universe.pair_ownership,
    )
}

pub(super) fn count_max6_pairs_by_membership(
    active_rows: &[usize],
    window_pair_count: usize,
    ownership: &Max6ScreenPairOwnership,
) -> Result<usize, String> {
    if active_rows.is_empty() {
        return Err("Max6 pair universe has no active tag rows".to_string());
    }
    if ownership.membership_units.is_empty() {
        return Ok(window_pair_count);
    }
    let has_group_labels = ownership.membership_groups.as_ref().is_some_and(|groups| {
        groups
            .iter()
            .any(|&group| group != Max6ScreenPairOwnership::NO_GROUP)
    });

    let count = if has_group_labels {
        let mut signatures = HashMap::<Vec<(u32, u32)>, u128>::new();
        for slot in 0..active_rows.len() {
            let start = ownership.offsets[slot] as usize;
            let end = ownership.offsets[slot + 1] as usize;
            let signature = (start..end)
                .map(|entry| {
                    let group = ownership
                        .membership_groups
                        .as_ref()
                        .map(|groups| groups[entry])
                        .unwrap_or(Max6ScreenPairOwnership::NO_GROUP);
                    (ownership.membership_units[entry], group)
                })
                .collect::<Vec<_>>();
            let count = signatures.entry(signature).or_default();
            *count = count
                .checked_add(1)
                .ok_or_else(|| "Max6 marker-signature count overflow".to_string())?;
        }
        count_grouped_max6_signature_pairs(&signatures)?.0
    } else {
        let mut signatures = HashMap::<Vec<u32>, u128>::new();
        let mut max_degree = 0usize;
        for slot in 0..active_rows.len() {
            let start = ownership.offsets[slot] as usize;
            let end = ownership.offsets[slot + 1] as usize;
            let signature = ownership.membership_units[start..end].to_vec();
            max_degree = max_degree.max(signature.len());
            let count = signatures.entry(signature).or_default();
            *count = count
                .checked_add(1)
                .ok_or_else(|| "Max6 marker-signature count overflow".to_string())?;
        }

        let expansion_work = signatures.iter().try_fold(0usize, |total, (signature, _)| {
            let degree = signature.len();
            let subsets = 1usize.checked_shl(degree as u32)?.saturating_sub(1);
            total.checked_add(subsets)
        });
        let bounded_expansion =
            expansion_work.is_some_and(|work| work <= MAX6_PAIR_SIGNATURE_SUBSET_WORK);
        if !bounded_expansion {
            // In unusually high-overlap layouts, the class-pair algorithm can
            // still be inexpensive when the number of distinct signatures is
            // small. Otherwise fail explicitly instead of falling back to the
            // multi-billion occurrence walk.
            let class_count = signatures.len();
            let pair_work = class_count
                .checked_mul(class_count.saturating_add(1))
                .map(|value| value / 2)
                .ok_or_else(|| "Max6 signature-class pair work overflow".to_string())?;
            if pair_work > MAX6_PAIR_SIGNATURE_CLASS_PAIR_WORK {
                count_ungrouped_max6_signature_pairs_sparse(&signatures)?.0
            } else {
                let grouped = signatures
                    .into_iter()
                    .map(|(signature, count)| {
                        (
                            signature
                                .into_iter()
                                .map(|unit| (unit, Max6ScreenPairOwnership::NO_GROUP))
                                .collect::<Vec<_>>(),
                            count,
                        )
                    })
                    .collect::<HashMap<_, _>>();
                count_grouped_max6_signature_pairs(&grouped)?.0
            }
        } else {
            let mut subset_counts = HashMap::<Vec<u32>, u128>::new();
            for (signature, &marker_count) in signatures.iter() {
                let subset_count = 1usize.checked_shl(signature.len() as u32).ok_or_else(|| {
                    format!(
                        "Max6 membership signature degree {} exceeds exact subset-count capacity",
                        signature.len()
                    )
                })?;
                for subset_mask in 1..subset_count {
                    let subset = signature
                        .iter()
                        .enumerate()
                        .filter_map(|(bit, &unit)| ((subset_mask >> bit) & 1 == 1).then_some(unit))
                        .collect::<Vec<_>>();
                    let count = subset_counts.entry(subset).or_default();
                    *count = count
                        .checked_add(marker_count)
                        .ok_or_else(|| "Max6 signature-subset marker count overflow".to_string())?;
                }
            }
            let mut total = 0i128;
            for (subset, &marker_count) in subset_counts.iter() {
                let pairs = checked_i128_from_u128(
                    count_choose_two_u128(marker_count)?,
                    "signature inclusion-exclusion",
                )?;
                let signed = if subset.len() % 2 == 1 { pairs } else { -pairs };
                total = total
                    .checked_add(signed)
                    .ok_or_else(|| "Max6 signature inclusion-exclusion overflow".to_string())?;
            }
            u128::try_from(total)
                .map_err(|_| "Max6 signature inclusion-exclusion became negative".to_string())?
        }
    };
    usize::try_from(count)
        .map_err(|_| format!("Max6 unique-pair count {count} exceeds usize capacity"))
}

#[cfg(test)]
pub(super) fn count_max6_screen_pairs_owner_walk(
    universe: &Max6ScreenPreparedUniverse,
) -> Result<usize, String> {
    let mut count = 0usize;
    for (unit_index, unit) in universe.units.iter().enumerate() {
        for first_local in 0..unit.selected_global_rows.len().saturating_sub(1) {
            for second_local in first_local + 1..unit.selected_global_rows.len() {
                if max6_screen_owned_pair(universe, unit_index, unit, first_local, second_local)?
                    .is_some()
                {
                    count = count
                        .checked_add(1)
                        .ok_or_else(|| "Max6 test owner-walk count overflow".to_string())?;
                }
            }
        }
    }
    Ok(count)
}

#[derive(Clone, Debug)]
pub(super) struct Max6ScreenValidation {
    pub(super) maxima: Vec<f64>,
    pub(super) rejections: usize,
    pub(super) empirical_fwer: f64,
    pub(super) ci_lower: f64,
    pub(super) ci_upper: f64,
}

#[inline]
pub(super) fn legacy_max6_permutation_active(
    ld_tag_exhaustive: bool,
    rule_permutation: bool,
    permutation_repeats: usize,
) -> bool {
    !ld_tag_exhaustive && rule_permutation && permutation_repeats > 0
}

#[inline]
pub(super) fn max6_null_calibration_enabled(
    ld_tag_exhaustive: bool,
    rule_permutation: bool,
    explicit_audit: bool,
) -> bool {
    rule_permutation && (!ld_tag_exhaustive || explicit_audit)
}

pub(super) fn audit_max6_final_pairs_are_screen_subset(
    screen_pairs: &[(usize, usize)],
    final_pairs: &[(usize, usize)],
) -> Result<usize, String> {
    let mut screen_set = HashSet::with_capacity(screen_pairs.len());
    for &(first, second) in screen_pairs {
        if first >= second {
            return Err(format!(
                "Max6 screen pair is not canonical (first < second): ({first}, {second})"
            ));
        }
        if !screen_set.insert((first, second)) {
            return Err(format!(
                "Max6 screen emitted duplicate pair ({first}, {second})"
            ));
        }
    }
    let mut final_set = HashSet::with_capacity(final_pairs.len());
    for &(first, second) in final_pairs {
        if first >= second {
            return Err(format!(
                "Max6 final pair is not canonical (first < second): ({first}, {second})"
            ));
        }
        if !final_set.insert((first, second)) {
            return Err(format!(
                "Max6 exact rerank emitted duplicate final pair ({first}, {second})"
            ));
        }
        if !screen_set.contains(&(first, second)) {
            return Err(format!(
                "D_final is not a subset of D_screen: final pair ({first}, {second}) was absent from the screen candidate set"
            ));
        }
    }
    Ok(final_set.len())
}

pub(super) fn clopper_pearson_interval(
    rejections: usize,
    repeats: usize,
) -> Result<(f64, f64), String> {
    if repeats == 0 || rejections > repeats {
        return Err(format!(
            "invalid binomial counts for FWER interval: rejections={rejections}, repeats={repeats}"
        ));
    }
    let alpha = 0.05;
    let lower = if rejections == 0 {
        0.0
    } else {
        Beta::new(rejections as f64, (repeats - rejections + 1) as f64)
            .map_err(|error| format!("failed to build lower FWER interval beta: {error}"))?
            .inverse_cdf(alpha / 2.0)
    };
    let upper = if rejections == repeats {
        1.0
    } else {
        Beta::new((rejections + 1) as f64, (repeats - rejections) as f64)
            .map_err(|error| format!("failed to build upper FWER interval beta: {error}"))?
            .inverse_cdf(1.0 - alpha / 2.0)
    };
    Ok((lower, upper))
}

pub(super) fn summarize_max6_screen_validation(
    maxima: &[f64],
    threshold: f64,
) -> Result<Max6ScreenValidation, String> {
    if !threshold.is_finite() {
        return Err(format!(
            "Max6 screen validation threshold must be finite, got {threshold}"
        ));
    }
    if maxima.is_empty() {
        return Err("Max6 screen validation requires at least one null replicate".to_string());
    }
    if let Some((index, score)) = maxima
        .iter()
        .enumerate()
        .find(|(_, score)| !score.is_finite())
    {
        return Err(format!(
            "Max6 screen validation maximum at replicate {index} is non-finite: {score}"
        ));
    }
    let rejections = maxima.iter().filter(|&&score| score > threshold).count();
    let repeats = maxima.len();
    let (ci_lower, ci_upper) = clopper_pearson_interval(rejections, repeats)?;
    Ok(Max6ScreenValidation {
        maxima: maxima.to_vec(),
        rejections,
        empirical_fwer: rejections as f64 / repeats as f64,
        ci_lower,
        ci_upper,
    })
}

#[inline]
pub(super) fn max6_ld_support_cache_required(ld_tag_r2: f64, full_identity_samples: bool) -> bool {
    full_identity_samples && ld_tag_r2 < 1.0
}

pub(super) fn prepare_max6_screen_selection(
    logic_bits: &GarfieldLogicBits,
    units: &[GarfieldLogicUnit],
    scan_unit_indices: &[usize],
    unit_kind_lc: &str,
    train_idx_local: &[usize],
    pool: Option<&rayon::ThreadPool>,
    mut progress: Option<&mut dyn FnMut(usize, usize, usize) -> Result<(), String>>,
    ld_tag_r2: f64,
) -> Result<Max6ScreenSelectedUnits, String> {
    let mut prepared = Vec::<Max6GrmMax6LdPreparedUnit>::new();
    let mut original_unit_indices = Vec::<usize>::new();
    let progress_total = scan_unit_indices.len().max(1);
    let mut selected_marker_total = 0usize;
    let mut any_eligible_unit = false;
    if let Some(callback) = progress.as_deref_mut() {
        callback(0, progress_total, selected_marker_total)?;
    }
    let worker_count = pool
        .map_or(1, rayon::ThreadPool::current_num_threads)
        .max(1);
    let task_batch_size = worker_count.saturating_mul(2).max(1);
    let full_identity_samples =
        sample_indices_are_full_identity(train_idx_local, logic_bits.n_samples);
    let support_cache_required = max6_ld_support_cache_required(ld_tag_r2, full_identity_samples);
    let support_cache_build_started = Instant::now();
    let support_cache = if support_cache_required {
        let support_cache_rows = collect_scan_ld_cache_rows(
            units,
            scan_unit_indices,
            logic_bits.sites.len(),
            unit_kind_lc,
        );
        let build_cache =
            || build_garfield_ld_support_cache(logic_bits, &support_cache_rows, train_idx_local);
        Some(if let Some(pool) = pool {
            pool.install(build_cache)?
        } else {
            build_cache()?
        })
    } else {
        None
    };
    let support_cache_build_seconds = support_cache_build_started.elapsed().as_secs_f64();
    let support_cache_rows = support_cache
        .as_ref()
        .map_or(0, |cache| cache.support.len());
    let support_cache_payload_bytes = support_cache
        .as_ref()
        .map_or(0, GarfieldLdSupportCache::payload_bytes);
    for (batch_index, unit_batch) in scan_unit_indices.chunks(task_batch_size).enumerate() {
        let batch_results = if let Some(pool) = pool {
            pool.install(|| {
                unit_batch
                    .par_iter()
                    .map(|&unit_index| {
                        let unit = units.get(unit_index).ok_or_else(|| {
                            format!("Max6 scan unit index out of range: {unit_index}")
                        })?;
                        prepare_max6_grmmax6_ld_unit_with_support_cache(
                            unit,
                            unit_kind_lc,
                            logic_bits,
                            train_idx_local,
                            support_cache.as_ref(),
                            ld_tag_r2,
                        )
                    })
                    .collect::<Result<Vec<_>, String>>()
            })?
        } else {
            unit_batch
                .iter()
                .map(|&unit_index| {
                    let unit = units.get(unit_index).ok_or_else(|| {
                        format!("Max6 scan unit index out of range: {unit_index}")
                    })?;
                    prepare_max6_grmmax6_ld_unit_with_support_cache(
                        unit,
                        unit_kind_lc,
                        logic_bits,
                        train_idx_local,
                        support_cache.as_ref(),
                        ld_tag_r2,
                    )
                })
                .collect::<Result<Vec<_>, String>>()?
        };
        for (&unit_index, prepared_unit) in unit_batch.iter().zip(batch_results.into_iter()) {
            any_eligible_unit |= prepared_unit.selected_global_rows.len() >= 2;
            selected_marker_total =
                selected_marker_total.saturating_add(prepared_unit.selected_global_rows.len());
            prepared.push(prepared_unit);
            original_unit_indices.push(unit_index);
        }
        if let Some(callback) = progress.as_deref_mut() {
            let done = ((batch_index + 1) * task_batch_size).min(scan_unit_indices.len());
            callback(done, progress_total, selected_marker_total)?;
        }
    }
    if prepared.is_empty() || !any_eligible_unit {
        return Err(
            "Max6 scan found no eligible LD-tag unit with at least two markers".to_string(),
        );
    }

    let mut active_rows = prepared
        .iter()
        .flat_map(|unit| unit.selected_global_rows.iter().copied())
        .collect::<Vec<_>>();
    active_rows.sort_unstable();
    active_rows.dedup();

    // Count eligible (unit, pair) occurrences for diagnostics. The exact maxT
    // calculation below uses the canonical union, not this occurrence count.
    let mut window_pair_count = 0usize;
    for unit in prepared.iter() {
        let n = unit.selected_global_rows.len();
        let all_pairs = n.saturating_mul(n.saturating_sub(1)) / 2;
        let excluded = unit
            .pair_groups
            .as_ref()
            .map(|groups| {
                let mut counts = HashMap::<usize, usize>::new();
                for &group in groups.iter() {
                    *counts.entry(group).or_default() += 1;
                }
                counts
                    .values()
                    .map(|&count| count.saturating_mul(count.saturating_sub(1)) / 2)
                    .sum::<usize>()
            })
            .unwrap_or(0);
        window_pair_count = window_pair_count.saturating_add(all_pairs.saturating_sub(excluded));
    }
    Ok(Max6ScreenSelectedUnits {
        units: prepared,
        original_unit_indices,
        active_rows,
        window_pair_count,
        support_cache_build_seconds,
        support_cache_rows,
        support_cache_payload_bytes,
    })
}

pub(super) fn prepare_max6_screen_universe(
    logic_bits: &GarfieldLogicBits,
    units: &[GarfieldLogicUnit],
    scan_unit_indices: &[usize],
    unit_kind_lc: &str,
    train_idx_local: &[usize],
    pool: Option<&rayon::ThreadPool>,
    progress: Option<&mut dyn FnMut(usize, usize, usize) -> Result<(), String>>,
    ld_tag_r2: f64,
) -> Result<Max6ScreenPreparedUniverse, String> {
    let selected = prepare_max6_screen_selection(
        logic_bits,
        units,
        scan_unit_indices,
        unit_kind_lc,
        train_idx_local,
        pool,
        progress,
        ld_tag_r2,
    )?;
    let pair_ownership = Max6ScreenPairOwnership::build(
        selected.units.as_slice(),
        selected.active_rows.as_slice(),
        logic_bits.sites.len(),
    )?;
    Ok(Max6ScreenPreparedUniverse {
        units: selected.units,
        original_unit_indices: selected.original_unit_indices,
        active_rows: selected.active_rows,
        pair_ownership,
        window_pair_count: selected.window_pair_count,
    })
}

pub(super) fn remap_max6_prepared_universe_for_trait(
    tags: &Max6PreparedGenotypeTags,
    trait_source_rows: &[usize],
    n_logic_rows: usize,
) -> Result<(Vec<GarfieldLogicUnit>, Max6ScreenPreparedUniverse), String> {
    if tags.units.len() != tags.screen_units.len()
        || tags.units.len() != tags.original_unit_indices.len()
    {
        return Err("prepared Max6 source-unit metadata lengths disagree".to_string());
    }
    if trait_source_rows.windows(2).any(|rows| rows[0] >= rows[1]) {
        return Err("trait-valid Max6 source rows must be strictly increasing".to_string());
    }
    if trait_source_rows.len() != n_logic_rows || n_logic_rows >= u32::MAX as usize {
        return Err("trait-valid Max6 row count is invalid".to_string());
    }
    let mut source_to_local = vec![u32::MAX; tags.source_marker_count];
    for (local_row, &source_row) in trait_source_rows.iter().enumerate() {
        let slot = source_to_local.get_mut(source_row).ok_or_else(|| {
            format!("trait-valid Max6 source row {source_row} exceeds source marker count")
        })?;
        *slot = local_row as u32;
    }

    let mut units = Vec::with_capacity(tags.units.len());
    let mut prepared_units = Vec::with_capacity(tags.screen_units.len());
    let mut active_flags = vec![false; n_logic_rows];
    let mut any_eligible_unit = false;
    for ((metadata, global_unit), &original_unit_index) in tags
        .units
        .iter()
        .zip(tags.screen_units.iter())
        .zip(tags.original_unit_indices.iter())
    {
        if metadata.original_unit_index != original_unit_index {
            return Err("prepared Max6 original unit IDs are inconsistent".to_string());
        }
        if global_unit
            .pair_groups
            .as_ref()
            .is_some_and(|groups| groups.len() != global_unit.selected_global_rows.len())
        {
            return Err("prepared Max6 pair-group metadata length mismatch".to_string());
        }
        let mut local_rows = Vec::with_capacity(global_unit.selected_global_rows.len());
        let mut local_groups = global_unit
            .pair_groups
            .as_ref()
            .map(|_| Vec::with_capacity(global_unit.selected_global_rows.len()));
        for (position, &source_row) in global_unit.selected_global_rows.iter().enumerate() {
            let local_row = source_to_local.get(source_row).copied().unwrap_or(u32::MAX);
            if local_row != u32::MAX {
                let local_row = local_row as usize;
                local_rows.push(local_row);
                active_flags[local_row] = true;
                if let (Some(groups), Some(local_groups)) =
                    (global_unit.pair_groups.as_ref(), local_groups.as_mut())
                {
                    local_groups.push(groups[position]);
                }
            }
        }
        any_eligible_unit |= local_rows.len() >= 2;
        units.push(GarfieldLogicUnit {
            label: metadata.label.clone(),
            indices: Vec::new(),
            spans: metadata.spans.clone(),
        });
        prepared_units.push(Max6GrmMax6LdPreparedUnit {
            selected_global_rows: local_rows,
            pair_groups: local_groups,
            // Genotype-global selection no longer records LD compression;
            // the phenotype-specific scorer does not consume it either.
            ld_compression: Vec::new(),
        });
    }
    if !any_eligible_unit {
        return Err("trait QC left no prepared Max6 unit with at least two LD tags".to_string());
    }
    drop(source_to_local);
    let active_rows = active_flags
        .iter()
        .enumerate()
        .filter_map(|(row, &active)| active.then_some(row))
        .collect::<Vec<_>>();
    let pair_ownership = Max6ScreenPairOwnership::build(
        prepared_units.as_slice(),
        active_rows.as_slice(),
        n_logic_rows,
    )?;
    let window_pair_count = prepared_units.iter().fold(0usize, |total, unit| {
        let n = unit.selected_global_rows.len();
        let all_pairs = n.saturating_mul(n.saturating_sub(1)) / 2;
        let excluded = unit
            .pair_groups
            .as_ref()
            .map(|groups| {
                let mut counts = HashMap::<usize, usize>::new();
                for &group in groups.iter() {
                    *counts.entry(group).or_default() += 1;
                }
                counts
                    .values()
                    .map(|&count| count.saturating_mul(count.saturating_sub(1)) / 2)
                    .sum::<usize>()
            })
            .unwrap_or(0);
        total.saturating_add(all_pairs.saturating_sub(excluded))
    });
    Ok((
        units,
        Max6ScreenPreparedUniverse {
            units: prepared_units,
            original_unit_indices: tags.original_unit_indices.clone(),
            active_rows,
            pair_ownership,
            window_pair_count,
        },
    ))
}
