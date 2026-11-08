use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Max6CrossWindowBatchStats {
    pub(super) windows: usize,
    pub(super) direct_windows: usize,
    pub(super) workspace_fallback_windows: usize,
    pub(super) candidate_occurrences: usize,
    pub(super) unique_candidate_pairs: usize,
    pub(super) unique_endpoints: usize,
    pub(super) estimated_workspace_bytes: usize,
    pub(super) rerank_seconds: f64,
    pub(super) batch_wall_seconds: f64,
    pub(super) parity_audit: Max6CrossWindowParityAuditStats,
}

impl Max6CrossWindowBatchStats {
    pub(super) fn accumulate(&mut self, other: Self) {
        self.windows = self.windows.saturating_add(other.windows);
        self.direct_windows = self.direct_windows.saturating_add(other.direct_windows);
        self.workspace_fallback_windows = self
            .workspace_fallback_windows
            .saturating_add(other.workspace_fallback_windows);
        self.candidate_occurrences = self
            .candidate_occurrences
            .saturating_add(other.candidate_occurrences);
        self.unique_candidate_pairs = self
            .unique_candidate_pairs
            .saturating_add(other.unique_candidate_pairs);
        self.unique_endpoints = self.unique_endpoints.saturating_add(other.unique_endpoints);
        self.estimated_workspace_bytes = self
            .estimated_workspace_bytes
            .max(other.estimated_workspace_bytes);
        self.rerank_seconds += other.rerank_seconds;
        self.batch_wall_seconds += other.batch_wall_seconds;
        self.parity_audit.accumulate(other.parity_audit);
    }
}

#[derive(Clone, Copy)]
pub(super) struct Max6CrossWindowExactScore {
    pub(super) logic: [f64; 5],
    pub(super) dosage: f64,
    pub(super) max6: f64,
    pub(super) best_channel: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Max6CrossWindowPairScoreDifference {
    pub(super) max_abs_by_channel: [f64; 7],
    pub(super) max_rel_by_channel: [f64; 7],
    pub(super) finite_mask_mismatch: bool,
    pub(super) best_channel_mismatch: bool,
}

pub(super) fn max6_cross_window_pair_score_difference(
    local_orientation: Max6CrossWindowExactScore,
    canonical_orientation: Max6CrossWindowExactScore,
    local_is_canonical: bool,
) -> Max6CrossWindowPairScoreDifference {
    let mut local_logic = local_orientation.logic;
    if !local_is_canonical {
        local_logic.swap(1, 2);
    }
    let local_best_channel = max6_cross_window_channel_for_orientation(
        local_orientation.best_channel,
        local_is_canonical,
    );
    let local_values = [
        local_logic[0],
        local_logic[1],
        local_logic[2],
        local_logic[3],
        local_logic[4],
        local_orientation.dosage,
        local_orientation.max6,
    ];
    let canonical_values = [
        canonical_orientation.logic[0],
        canonical_orientation.logic[1],
        canonical_orientation.logic[2],
        canonical_orientation.logic[3],
        canonical_orientation.logic[4],
        canonical_orientation.dosage,
        canonical_orientation.max6,
    ];
    let mut difference = Max6CrossWindowPairScoreDifference {
        best_channel_mismatch: local_best_channel != canonical_orientation.best_channel,
        ..Default::default()
    };
    for channel in 0..local_values.len() {
        let local = local_values[channel];
        let canonical = canonical_values[channel];
        if local.is_finite() != canonical.is_finite() {
            difference.finite_mask_mismatch = true;
            difference.max_abs_by_channel[channel] = f64::INFINITY;
            difference.max_rel_by_channel[channel] = f64::INFINITY;
        } else if local.is_finite() {
            let absolute = (local - canonical).abs();
            let relative = absolute / local.abs().max(canonical.abs()).max(1.0e-300);
            difference.max_abs_by_channel[channel] = absolute;
            difference.max_rel_by_channel[channel] = relative;
        }
    }
    difference
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Max6CrossWindowParityAuditStats {
    pub(super) windows_checked: usize,
    pub(super) pairs_checked: usize,
    pub(super) max_abs_by_channel: [f64; 7],
    pub(super) max_rel_by_channel: [f64; 7],
    pub(super) abs_gt_1e_12_by_channel: [usize; 7],
    pub(super) finite_mask_mismatch_pairs: usize,
    pub(super) best_channel_mismatch_pairs: usize,
    pub(super) rank_changed_pairs: usize,
    pub(super) max_rank_displacement: usize,
    pub(super) topk_set_symmetric_difference: [usize; 4],
    pub(super) topk_order_changed_windows: [usize; 4],
}

impl Max6CrossWindowParityAuditStats {
    fn observe_pair(&mut self, difference: Max6CrossWindowPairScoreDifference) {
        self.pairs_checked = self.pairs_checked.saturating_add(1);
        self.finite_mask_mismatch_pairs += usize::from(difference.finite_mask_mismatch);
        self.best_channel_mismatch_pairs += usize::from(difference.best_channel_mismatch);
        for channel in 0..self.max_abs_by_channel.len() {
            self.max_abs_by_channel[channel] =
                self.max_abs_by_channel[channel].max(difference.max_abs_by_channel[channel]);
            self.max_rel_by_channel[channel] =
                self.max_rel_by_channel[channel].max(difference.max_rel_by_channel[channel]);
            self.abs_gt_1e_12_by_channel[channel] +=
                usize::from(difference.max_abs_by_channel[channel] > 1.0e-12);
        }
    }

    fn accumulate(&mut self, other: Self) {
        self.windows_checked += other.windows_checked;
        self.pairs_checked += other.pairs_checked;
        for channel in 0..self.max_abs_by_channel.len() {
            self.max_abs_by_channel[channel] =
                self.max_abs_by_channel[channel].max(other.max_abs_by_channel[channel]);
            self.max_rel_by_channel[channel] =
                self.max_rel_by_channel[channel].max(other.max_rel_by_channel[channel]);
            self.abs_gt_1e_12_by_channel[channel] += other.abs_gt_1e_12_by_channel[channel];
        }
        self.finite_mask_mismatch_pairs += other.finite_mask_mismatch_pairs;
        self.best_channel_mismatch_pairs += other.best_channel_mismatch_pairs;
        self.rank_changed_pairs += other.rank_changed_pairs;
        self.max_rank_displacement = self.max_rank_displacement.max(other.max_rank_displacement);
        for top_k in 0..self.topk_set_symmetric_difference.len() {
            self.topk_set_symmetric_difference[top_k] += other.topk_set_symmetric_difference[top_k];
            self.topk_order_changed_windows[top_k] += other.topk_order_changed_windows[top_k];
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Max6CrossWindowRankAudit {
    pub(super) rank_changed_pairs: usize,
    pub(super) max_rank_displacement: usize,
    pub(super) topk_set_symmetric_difference: [usize; 4],
    pub(super) topk_order_changed_windows: [usize; 4],
}

pub(super) fn max6_cross_window_rank_audit(
    reference_scores: &[f64],
    union_scores: &[f64],
    tie_keys: &[usize],
) -> Result<Max6CrossWindowRankAudit, String> {
    if reference_scores.len() != union_scores.len() || reference_scores.len() != tie_keys.len() {
        return Err("cross-window parity rank inputs have inconsistent lengths".to_string());
    }
    let reference_order = max6_order_by_score_and_tie(reference_scores, tie_keys)?;
    let union_order = max6_order_by_score_and_tie(union_scores, tie_keys)?;
    let mut reference_rank = vec![0usize; reference_scores.len()];
    let mut union_rank = vec![0usize; union_scores.len()];
    for (rank, &candidate) in reference_order.iter().enumerate() {
        reference_rank[candidate] = rank + 1;
    }
    for (rank, &candidate) in union_order.iter().enumerate() {
        union_rank[candidate] = rank + 1;
    }
    let mut audit = Max6CrossWindowRankAudit::default();
    for candidate in 0..reference_scores.len() {
        let displacement = reference_rank[candidate].abs_diff(union_rank[candidate]);
        audit.rank_changed_pairs += usize::from(displacement > 0);
        audit.max_rank_displacement = audit.max_rank_displacement.max(displacement);
    }
    for (slot, top_k) in [1usize, 10, 100, 500].into_iter().enumerate() {
        let count = top_k.min(reference_order.len());
        let mut in_reference = vec![false; reference_scores.len()];
        let mut in_union = vec![false; union_scores.len()];
        for &candidate in reference_order.iter().take(count) {
            in_reference[candidate] = true;
        }
        for &candidate in union_order.iter().take(count) {
            in_union[candidate] = true;
        }
        audit.topk_set_symmetric_difference[slot] = in_reference
            .iter()
            .zip(in_union.iter())
            .filter(|(left, right)| left != right)
            .count();
        audit.topk_order_changed_windows[slot] = usize::from(
            reference_order
                .iter()
                .take(count)
                .ne(union_order.iter().take(count)),
        );
    }
    Ok(audit)
}

pub(super) fn max6_cross_window_score_local_reference(
    screen: &Max6DeferredWindowScreen,
    logic_bits: &GarfieldLogicBits,
    p_c: Option<&[f64]>,
    no_grm_projector: Option<&NoGrmCovariateProjector>,
    projected_y: &[f64],
    y: &[f64],
    score_scale: f64,
) -> Result<CandidateRerankResult, String> {
    let packed_window = gather_rows_by_indices(
        logic_bits.bits(),
        logic_bits.row_words,
        screen.selected_global_rows.as_slice(),
        "cross-window parity reference window",
    )?;
    let candidate_first = screen
        .candidates
        .iter()
        .map(|candidate| candidate.first as i64)
        .collect::<Vec<_>>();
    let candidate_second = screen
        .candidates
        .iter()
        .map(|candidate| candidate.second as i64)
        .collect::<Vec<_>>();
    let candidate_count = screen.candidates.len();
    if let Some(p_c) = p_c {
        candidate_rerank_exact_packed_with_py_trusted_scaled(
            packed_window.as_slice(),
            logic_bits.row_words,
            logic_bits.n_samples,
            screen.selected_global_rows.len(),
            p_c,
            y,
            projected_y,
            candidate_first.as_slice(),
            candidate_second.as_slice(),
            candidate_count,
            DosageChannel::FwlProduct,
            score_scale,
        )
    } else {
        let projector = no_grm_projector.ok_or_else(|| {
            "cross-window parity reference has neither P_C nor no-GRM projector".to_string()
        })?;
        candidate_rerank_exact_packed_no_grm_covariate_with_py_trusted_scaled(
            packed_window.as_slice(),
            logic_bits.row_words,
            logic_bits.n_samples,
            screen.selected_global_rows.len(),
            projector,
            y,
            projected_y,
            candidate_first.as_slice(),
            candidate_second.as_slice(),
            candidate_count,
            DosageChannel::FwlProduct,
            score_scale,
        )
    }
}

pub(super) fn max6_cross_window_audit_window_parity(
    unit_index: usize,
    screen: &Max6DeferredWindowScreen,
    union_score_by_pair: &HashMap<u64, Max6CrossWindowExactScore>,
    union_scores_by_candidate: &[f64],
    tie_keys: &[usize],
    logic_bits: &GarfieldLogicBits,
    p_c: Option<&[f64]>,
    no_grm_projector: Option<&NoGrmCovariateProjector>,
    projected_y: &[f64],
    y: &[f64],
    score_scale: f64,
) -> Result<Max6CrossWindowParityAuditStats, String> {
    let reference = max6_cross_window_score_local_reference(
        screen,
        logic_bits,
        p_c,
        no_grm_projector,
        projected_y,
        y,
        score_scale,
    )?;
    let mut candidate_index_by_pair = HashMap::with_capacity(screen.candidates.len());
    for (candidate_index, candidate) in screen.candidates.iter().enumerate() {
        let left = screen.selected_global_rows[candidate.first];
        let right = screen.selected_global_rows[candidate.second];
        let key = max6_diagnostic_pair_key(left, right)?;
        if candidate_index_by_pair
            .insert(key, candidate_index)
            .is_some()
        {
            return Err(format!(
                "cross-window parity reference received duplicate screen pair key {key}"
            ));
        }
    }
    if reference.pair_i.len() != screen.candidates.len()
        || reference.pair_j.len() != screen.candidates.len()
        || reference.logic.len() != screen.candidates.len()
        || reference.dosage.len() != screen.candidates.len()
        || reference.max6.len() != screen.candidates.len()
        || reference.best_max6_channel.len() != screen.candidates.len()
    {
        return Err(format!(
            "cross-window parity reference returned {} scores for {} candidates",
            reference.max6.len(),
            screen.candidates.len()
        ));
    }
    if union_scores_by_candidate.len() != screen.candidates.len()
        || tie_keys.len() != screen.candidates.len()
    {
        return Err("cross-window parity union score dimensions are inconsistent".to_string());
    }

    let mut reference_scores_by_candidate = vec![f64::NAN; screen.candidates.len()];
    let mut seen_candidates = vec![false; screen.candidates.len()];
    let mut audit = Max6CrossWindowParityAuditStats {
        windows_checked: 1,
        ..Default::default()
    };
    for reference_index in 0..reference.max6.len() {
        let local_left = reference.pair_i[reference_index];
        let local_right = reference.pair_j[reference_index];
        let global_left = *screen
            .selected_global_rows
            .get(local_left)
            .ok_or_else(|| "parity reference left endpoint is out of range".to_string())?;
        let global_right = *screen
            .selected_global_rows
            .get(local_right)
            .ok_or_else(|| "parity reference right endpoint is out of range".to_string())?;
        let pair_key = max6_diagnostic_pair_key(global_left, global_right)?;
        let candidate_index = *candidate_index_by_pair.get(&pair_key).ok_or_else(|| {
            format!("cross-window parity reference emitted unexpected pair key {pair_key}")
        })?;
        if std::mem::replace(&mut seen_candidates[candidate_index], true) {
            return Err(format!(
                "cross-window parity reference emitted duplicate pair key {pair_key}"
            ));
        }
        let union_score = *union_score_by_pair
            .get(&pair_key)
            .ok_or_else(|| format!("cross-window parity union is missing pair key {pair_key}"))?;
        let local_score = Max6CrossWindowExactScore {
            logic: reference.logic[reference_index],
            dosage: reference.dosage[reference_index],
            max6: reference.max6[reference_index],
            best_channel: reference.best_max6_channel[reference_index],
        };
        audit.observe_pair(max6_cross_window_pair_score_difference(
            local_score,
            union_score,
            global_left < global_right,
        ));
        reference_scores_by_candidate[candidate_index] = reference.max6[reference_index];
    }
    if seen_candidates.iter().any(|seen| !seen) {
        return Err(
            "cross-window parity reference omitted one or more screen candidates".to_string(),
        );
    }
    let rank_audit = max6_cross_window_rank_audit(
        reference_scores_by_candidate.as_slice(),
        union_scores_by_candidate,
        tie_keys,
    )?;
    if rank_audit.topk_order_changed_windows[1] > 0 {
        let reference_order =
            max6_order_by_score_and_tie(reference_scores_by_candidate.as_slice(), tie_keys)?;
        let union_order = max6_order_by_score_and_tie(union_scores_by_candidate, tie_keys)?;
        for (rank, (&reference_candidate, &union_candidate)) in reference_order
            .iter()
            .zip(union_order.iter())
            .take(10)
            .enumerate()
        {
            if reference_candidate == union_candidate {
                continue;
            }
            let reference_pair = &screen.candidates[reference_candidate];
            let union_pair = &screen.candidates[union_candidate];
            let reference_key = max6_diagnostic_pair_key(
                screen.selected_global_rows[reference_pair.first],
                screen.selected_global_rows[reference_pair.second],
            )?;
            let union_key = max6_diagnostic_pair_key(
                screen.selected_global_rows[union_pair.first],
                screen.selected_global_rows[union_pair.second],
            )?;
            let reference_rows = (
                screen.selected_global_rows[reference_pair.first],
                screen.selected_global_rows[reference_pair.second],
            );
            let union_rows = (
                screen.selected_global_rows[union_pair.first],
                screen.selected_global_rows[union_pair.second],
            );
            eprintln!(
                "[MAX6-CROSS-WINDOW-PARITY-TOP10] unit={} rank={} reference_pair={}-{} reference_key={} reference_score={:.17e} reference_union_score={:.17e} union_pair={}-{} union_key={} union_score={:.17e} union_reference_score={:.17e} reference_pair_gap={:.17e} union_pair_gap={:.17e}",
                unit_index,
                rank + 1,
                reference_rows.0,
                reference_rows.1,
                reference_key,
                reference_scores_by_candidate[reference_candidate],
                union_scores_by_candidate[reference_candidate],
                union_rows.0,
                union_rows.1,
                union_key,
                union_scores_by_candidate[union_candidate],
                reference_scores_by_candidate[union_candidate],
                reference_scores_by_candidate[reference_candidate]
                    - reference_scores_by_candidate[union_candidate],
                union_scores_by_candidate[reference_candidate]
                    - union_scores_by_candidate[union_candidate],
            );
        }
    }
    audit.rank_changed_pairs = rank_audit.rank_changed_pairs;
    audit.max_rank_displacement = rank_audit.max_rank_displacement;
    audit.topk_set_symmetric_difference = rank_audit.topk_set_symmetric_difference;
    audit.topk_order_changed_windows = rank_audit.topk_order_changed_windows;
    Ok(audit)
}

pub(super) fn max6_cross_window_channel_for_orientation(
    channel: &'static str,
    same_orientation: bool,
) -> &'static str {
    if same_orientation {
        channel
    } else {
        match channel {
            "AND01" => "AND10",
            "AND10" => "AND01",
            other => other,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) const MAX6_CROSS_WINDOW_WORKSPACE_PREFLIGHT_PREFIX: &str =
    "cross-window exact batch preflight estimates ";

pub(super) fn max6_cross_window_union_enabled(
    unit_kind: &str,
    unit_count: usize,
    override_value: Option<&str>,
) -> bool {
    // Window overlap has been checked for score/output parity. Grouped modes
    // retain their per-unit ownership and rerank path until separately audited.
    // Explicit =0 is the reference-path escape hatch for window scans.
    unit_kind == "window"
        && unit_count > 1
        && override_value
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "y" | "on"
                )
            })
            .unwrap_or(true)
}

pub(super) fn rerank_max6_grmmax6_screen_batch_with_audit(
    scans: &mut [(usize, Max6GrmMax6UnitScan)],
    logic_bits: &GarfieldLogicBits,
    p_c: Option<&[f64]>,
    no_grm_projector: Option<&NoGrmCovariateProjector>,
    projected_y: &[f64],
    y: &[f64],
    score_scale: f64,
    scaled_total_quadratic: f64,
    final_top_k: usize,
    rule_permutation: bool,
    collect_candidate_pair_keys: bool,
    collect_candidate_scores: bool,
    audit_reference_window_scores: bool,
    max_workspace_bytes: usize,
) -> Result<Max6CrossWindowBatchStats, String> {
    let batch_started = Instant::now();
    let mut pair_keys = Vec::<u64>::new();
    let mut candidate_occurrences = 0usize;
    for (_, scan) in scans.iter() {
        if let Some(screen) = scan.deferred_screen.as_ref() {
            candidate_occurrences = candidate_occurrences
                .checked_add(screen.candidates.len())
                .ok_or_else(|| "cross-window candidate count overflow".to_string())?;
            pair_keys
                .try_reserve(screen.candidates.len())
                .map_err(|error| format!("cannot reserve cross-window pair keys: {error}"))?;
            for candidate in &screen.candidates {
                let left = *screen
                    .selected_global_rows
                    .get(candidate.first)
                    .ok_or_else(|| "cross-window left endpoint is out of range".to_string())?;
                let right = *screen
                    .selected_global_rows
                    .get(candidate.second)
                    .ok_or_else(|| "cross-window right endpoint is out of range".to_string())?;
                pair_keys.push(max6_diagnostic_pair_key(left, right)?);
            }
        }
    }
    if candidate_occurrences == 0 {
        return Ok(Max6CrossWindowBatchStats {
            windows: scans.len(),
            batch_wall_seconds: batch_started.elapsed().as_secs_f64(),
            ..Default::default()
        });
    }
    pair_keys.sort_unstable();
    pair_keys.dedup();
    let unique_candidate_pairs = pair_keys.len();
    let mut endpoints = Vec::<usize>::with_capacity(unique_candidate_pairs.saturating_mul(2));
    for &key in &pair_keys {
        endpoints.push((key >> 32) as u32 as usize);
        endpoints.push(key as u32 as usize);
    }
    endpoints.sort_unstable();
    endpoints.dedup();
    let unique_endpoints = endpoints.len();
    if unique_endpoints < 2 {
        return Err(format!(
            "cross-window batch has {unique_candidate_pairs} pairs but only {unique_endpoints} unique endpoints"
        ));
    }

    // Conservative preflight for packed rows, decoded X/Z workspaces, pair
    // geometry, score channels, and output/index arrays. A batch that exceeds
    // this cap fails before allocating the dense exact-rerank workspace.
    let n = logic_bits.n_samples;
    let dense_endpoint_bytes = n
        .checked_mul(unique_endpoints)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| "cross-window dense endpoint workspace overflow".to_string())?;
    let packed_endpoint_bytes = unique_endpoints
        .checked_mul(logic_bits.row_words)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u64>()))
        .ok_or_else(|| "cross-window packed endpoint workspace overflow".to_string())?;
    let pair_workspace_bytes = unique_candidate_pairs
        .checked_mul(256)
        .ok_or_else(|| "cross-window pair workspace overflow".to_string())?;
    let parity_reference_bytes = if audit_reference_window_scores {
        let max_window_markers = scans
            .iter()
            .filter_map(|(_, scan)| scan.deferred_screen.as_ref())
            .map(|screen| screen.selected_global_rows.len())
            .max()
            .unwrap_or(0);
        let max_window_candidates = scans
            .iter()
            .filter_map(|(_, scan)| scan.deferred_screen.as_ref())
            .map(|screen| screen.candidates.len())
            .max()
            .unwrap_or(0);
        n.checked_mul(max_window_markers)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f64>()))
            .and_then(|values| values.checked_mul(4))
            .and_then(|values| {
                max_window_candidates
                    .checked_mul(128)
                    .and_then(|scores| values.checked_add(scores))
            })
            .ok_or_else(|| "cross-window parity reference workspace overflow".to_string())?
    } else {
        0
    };
    let estimated_workspace_bytes = dense_endpoint_bytes
        .checked_mul(4)
        .and_then(|values| values.checked_add(packed_endpoint_bytes))
        .and_then(|values| values.checked_add(pair_workspace_bytes))
        .and_then(|values| values.checked_add(parity_reference_bytes))
        .ok_or_else(|| "cross-window workspace estimate overflow".to_string())?;
    if estimated_workspace_bytes > max_workspace_bytes {
        return Err(format!(
            "{MAX6_CROSS_WINDOW_WORKSPACE_PREFLIGHT_PREFIX}{estimated_workspace_bytes} bytes for {} pairs / {} endpoints; per-worker limit is {}",
            unique_candidate_pairs, unique_endpoints, max_workspace_bytes
        ));
    }

    let packed_endpoints = gather_rows_by_indices(
        logic_bits.bits(),
        logic_bits.row_words,
        endpoints.as_slice(),
        "cross-window exact candidate endpoints",
    )?;
    let mut candidate_first = Vec::<i64>::with_capacity(unique_candidate_pairs);
    let mut candidate_second = Vec::<i64>::with_capacity(unique_candidate_pairs);
    for &key in &pair_keys {
        let left = ((key >> 32) as u32) as usize;
        let right = (key as u32) as usize;
        let local_left = endpoints
            .binary_search(&left)
            .map_err(|_| "cross-window left endpoint mapping is incomplete".to_string())?;
        let local_right = endpoints
            .binary_search(&right)
            .map_err(|_| "cross-window right endpoint mapping is incomplete".to_string())?;
        candidate_first.push(local_left as i64);
        candidate_second.push(local_right as i64);
    }
    let rerank_started = Instant::now();
    let exact = if let Some(p_c) = p_c {
        candidate_rerank_exact_packed_with_py_trusted_scaled(
            packed_endpoints.as_slice(),
            logic_bits.row_words,
            logic_bits.n_samples,
            unique_endpoints,
            p_c,
            y,
            projected_y,
            candidate_first.as_slice(),
            candidate_second.as_slice(),
            unique_candidate_pairs,
            DosageChannel::FwlProduct,
            score_scale,
        )?
    } else {
        let projector = no_grm_projector.ok_or_else(|| {
            "cross-window Max6 rerank has neither P_C nor no-GRM projector".to_string()
        })?;
        candidate_rerank_exact_packed_no_grm_covariate_with_py_trusted_scaled(
            packed_endpoints.as_slice(),
            logic_bits.row_words,
            logic_bits.n_samples,
            unique_endpoints,
            projector,
            y,
            projected_y,
            candidate_first.as_slice(),
            candidate_second.as_slice(),
            unique_candidate_pairs,
            DosageChannel::FwlProduct,
            score_scale,
        )?
    };
    let rerank_seconds = rerank_started.elapsed().as_secs_f64();
    if exact.pair_i.len() != unique_candidate_pairs
        || exact.pair_j.len() != unique_candidate_pairs
        || exact.max6.len() != unique_candidate_pairs
        || exact.logic.len() != unique_candidate_pairs
        || exact.dosage.len() != unique_candidate_pairs
        || exact.best_max6_channel.len() != unique_candidate_pairs
    {
        return Err(format!(
            "cross-window exact scorer returned {} pairs for a {unique_candidate_pairs}-pair union",
            exact.max6.len()
        ));
    }
    let mut score_by_pair =
        HashMap::<u64, Max6CrossWindowExactScore>::with_capacity(unique_candidate_pairs);
    for candidate_index in 0..unique_candidate_pairs {
        let local_left = exact.pair_i[candidate_index];
        let local_right = exact.pair_j[candidate_index];
        let global_left = *endpoints
            .get(local_left)
            .ok_or_else(|| "cross-window scored left endpoint is out of range".to_string())?;
        let global_right = *endpoints
            .get(local_right)
            .ok_or_else(|| "cross-window scored right endpoint is out of range".to_string())?;
        let key = max6_diagnostic_pair_key(global_left, global_right)?;
        score_by_pair.insert(
            key,
            Max6CrossWindowExactScore {
                logic: exact.logic[candidate_index],
                dosage: exact.dosage[candidate_index],
                max6: exact.max6[candidate_index],
                best_channel: exact.best_max6_channel[candidate_index],
            },
        );
    }
    if score_by_pair.len() != unique_candidate_pairs {
        return Err(format!(
            "cross-window score map contains {} keys for {unique_candidate_pairs} unique pairs",
            score_by_pair.len()
        ));
    }

    let total_candidate_count = candidate_occurrences.max(1) as f64;
    let mut parity_audit = Max6CrossWindowParityAuditStats::default();
    for (unit_index, scan) in scans.iter_mut() {
        let Some(screen) = scan.deferred_screen.take() else {
            continue;
        };
        let n_markers = screen.selected_global_rows.len();
        let candidate_count = screen.candidates.len();
        let mut candidate_scores = Vec::with_capacity(candidate_count);
        let mut tie_keys = Vec::with_capacity(candidate_count);
        let mut pair_key_by_candidate = Vec::with_capacity(candidate_count);
        let mut output_orientation = Vec::with_capacity(candidate_count);
        let mut window_pair_seen = HashSet::with_capacity(candidate_count);
        for candidate in &screen.candidates {
            let local_first = candidate.first;
            let local_second = candidate.second;
            if local_first >= n_markers || local_second >= n_markers || local_first == local_second
            {
                return Err("cross-window screen candidate indices are invalid".to_string());
            }
            let global_first = screen.selected_global_rows[local_first];
            let global_second = screen.selected_global_rows[local_second];
            let pair_key = max6_diagnostic_pair_key(global_first, global_second)?;
            if !window_pair_seen.insert(pair_key) {
                return Err(format!(
                    "cross-window input screen emitted duplicate pair {pair_key} within unit {unit_index}"
                ));
            }
            let score = score_by_pair.get(&pair_key).ok_or_else(|| {
                format!("cross-window union is missing screen pair key {pair_key}")
            })?;
            candidate_scores.push(score.max6);
            tie_keys.push(
                local_first * n_markers - local_first * (local_first + 1) / 2
                    + (local_second - local_first - 1),
            );
            pair_key_by_candidate.push(pair_key);
            output_orientation.push(global_first < global_second);
        }
        if audit_reference_window_scores {
            let window_audit = max6_cross_window_audit_window_parity(
                *unit_index,
                &screen,
                &score_by_pair,
                candidate_scores.as_slice(),
                tie_keys.as_slice(),
                logic_bits,
                p_c,
                no_grm_projector,
                projected_y,
                y,
                score_scale,
            )?;
            parity_audit.accumulate(window_audit);
        }
        let output_finalize_started = Instant::now();
        let local_rank_started = Instant::now();
        let order = max6_order_by_score_and_tie(&candidate_scores, &tie_keys)?;
        let local_rank_seconds = local_rank_started.elapsed().as_secs_f64();
        let top_count = final_top_k.max(1).min(order.len());
        let mut outputs = Vec::with_capacity(top_count);
        let mut final_local_pairs = Vec::with_capacity(top_count);
        for (rank, &candidate_index) in order.iter().take(top_count).enumerate() {
            let candidate = &screen.candidates[candidate_index];
            let global_first = screen.selected_global_rows[candidate.first];
            let global_second = screen.selected_global_rows[candidate.second];
            let score = score_by_pair[&pair_key_by_candidate[candidate_index]];
            let same_orientation = output_orientation[candidate_index];
            let best_channel =
                max6_cross_window_channel_for_orientation(score.best_channel, same_orientation);
            final_local_pairs.push((candidate.first, candidate.second));
            outputs.push(Max6GrmMax6PairOutput {
                unit_index: *unit_index,
                candidate_rank: rank + 1,
                first_global_row: global_first,
                second_global_row: global_second,
                four_cell_counts: None,
                dosage: score.dosage,
                max6: score.max6,
                best_channel,
                pair_pve: pair_pve_from_scaled_score(score.max6, scaled_total_quadratic),
            });
        }
        let screen_local_pairs = screen
            .candidates
            .iter()
            .map(|candidate| (candidate.first, candidate.second))
            .collect::<Vec<_>>();
        scan.final_screen_subset_audit_pairs_checked =
            audit_max6_final_pairs_are_screen_subset(&screen_local_pairs, &final_local_pairs)?;
        scan.outputs = outputs;
        scan.screen_pairs_passed = candidate_count;
        scan.timing.exact_candidates = candidate_count;
        scan.timing.grm_int_candidates = 0;
        scan.timing.exact_input_duplicates = 0;
        scan.timing.exact_input_self_pairs = 0;
        scan.timing.output_finalize_seconds += output_finalize_started.elapsed().as_secs_f64();
        scan.permutation_pool = rule_permutation.then(|| Max6GrmMax6PermutationPool {
            unit_index: *unit_index,
            selected_global_rows: screen.selected_global_rows.clone(),
        });
        let weight = candidate_count as f64 / total_candidate_count;
        let mut shared_timing = exact.timing;
        shared_timing.endpoint_prepare_seconds *= weight;
        shared_timing.pcx_u_seconds *= weight;
        shared_timing.py_seconds *= weight;
        shared_timing.candidate_geometry_seconds *= weight;
        shared_timing.geometry_marker_moments_seconds *= weight;
        shared_timing.geometry_pairwise_seconds *= weight;
        shared_timing.geometry_au_validation_seconds *= weight;
        shared_timing.geometry_au_allocation_seconds *= weight;
        shared_timing.geometry_au_materialization_seconds *= weight;
        shared_timing.geometry_au_gemm_seconds *= weight;
        shared_timing.geometry_au_candidate_lookup_seconds *= weight;
        shared_timing.geometry_au_fused_py_seconds *= weight;
        shared_timing.geometry_au_copy_seconds *= weight;
        shared_timing.geometry_lookup_seconds *= weight;
        shared_timing.geometry_interaction_py_seconds *= weight;
        shared_timing.q_seconds *= weight;
        shared_timing.q_validation_seconds *= weight;
        shared_timing.q_bitset_seconds *= weight;
        shared_timing.q_cell_seconds *= weight;
        shared_timing.q_enumerate_seconds *= weight;
        shared_timing.q_quadratic_seconds *= weight;
        shared_timing.q_reconstruct_seconds *= weight;
        shared_timing.score_seconds *= weight;
        shared_timing.topk_seconds = (shared_timing.topk_seconds + local_rank_seconds) * weight;
        shared_timing.kernel_seconds *= weight;
        scan.timing.rerank = shared_timing;

        if collect_candidate_pair_keys || collect_candidate_scores {
            let audit_window_index = u32::try_from(*unit_index).map_err(|_| {
                format!("Max6 cross-window audit index {unit_index} exceeds u32 storage")
            })?;
            let mut endpoint_seen = vec![false; n_markers];
            if collect_candidate_pair_keys {
                scan.candidate_pair_keys.reserve(candidate_count);
            }
            if collect_candidate_scores {
                scan.candidate_score_observations.reserve(candidate_count);
            }
            for (candidate_index, candidate) in screen.candidates.iter().enumerate() {
                endpoint_seen[candidate.first] = true;
                endpoint_seen[candidate.second] = true;
                let pair_key = pair_key_by_candidate[candidate_index];
                if collect_candidate_pair_keys {
                    scan.candidate_pair_keys.push(pair_key);
                }
                if collect_candidate_scores {
                    let tie_key = u32::try_from(tie_keys[candidate_index]).map_err(|_| {
                        format!(
                            "Max6 cross-window tie key {} exceeds u32",
                            tie_keys[candidate_index]
                        )
                    })?;
                    let score = score_by_pair[&pair_key];
                    let same_orientation = output_orientation[candidate_index];
                    let channel = max6_cross_window_channel_for_orientation(
                        score.best_channel,
                        same_orientation,
                    );
                    scan.candidate_score_observations
                        .push(Max6CandidateScoreObservation {
                            pair_key,
                            window_index: audit_window_index,
                            tie_key,
                            score: score.max6,
                            cached_score: score.max6,
                            channel_id: max6_channel_audit_id(channel)?,
                        });
                }
            }
            scan.window_unique_candidate_endpoints =
                endpoint_seen.into_iter().filter(|&seen| seen).count();
        }
    }

    Ok(Max6CrossWindowBatchStats {
        windows: scans.len(),
        direct_windows: 0,
        workspace_fallback_windows: 0,
        candidate_occurrences,
        unique_candidate_pairs,
        unique_endpoints,
        estimated_workspace_bytes,
        rerank_seconds,
        batch_wall_seconds: batch_started.elapsed().as_secs_f64(),
        parity_audit,
    })
}
