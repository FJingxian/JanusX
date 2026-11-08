use super::*;

pub(super) fn resolve_ml_keep_k(n_region: usize, ml_top_k: usize, ml_top_frac: f64) -> usize {
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

#[inline]
pub(super) fn resolve_stage1_candidate_pool_k(
    unit_kind_lc: &str,
    unit: &GarfieldLogicUnit,
    n_region: usize,
    ml_top_k: usize,
    ml_top_frac: f64,
) -> usize {
    let beam_k = resolve_ml_keep_k(n_region, ml_top_k, ml_top_frac)
        .max(geneset_min_keep_k(unit_kind_lc, unit))
        .min(n_region);
    if beam_k == 0 {
        return 0;
    }
    if unit_kind_lc == "geneset" && unit.spans.len() > 1 {
        return beam_k;
    }
    if let Some((mul, add)) = garfield_single_window_candidate_override() {
        return beam_k
            .saturating_mul(mul)
            .max(beam_k.saturating_add(add))
            .min(n_region)
            .max(beam_k);
    }
    resolve_dynamic_single_window_candidate_pool_k(beam_k, n_region)
}

#[inline]
pub(super) fn corr_candidate_pool_prescreen_k(n_region: usize, keep_k: usize) -> usize {
    if n_region == 0 {
        return 0;
    }
    let keep_k = keep_k.max(1).min(n_region);
    let slack = (keep_k / 4).clamp(
        GARFIELD_GENESET_CORR_PRESCREEN_SLACK_MIN,
        GARFIELD_GENESET_CORR_PRESCREEN_SLACK_MAX,
    );
    n_region.min(keep_k.saturating_add(slack)).max(keep_k)
}

pub(super) fn select_single_window_candidate_pool_rows_with_cache_trace(
    candidate_global_rows: &[usize],
    scores: &[f64],
    target_k: usize,
    logic_bits: &GarfieldLogicBits,
    sample_indices: &[usize],
    support_cache: Option<&GarfieldLdSupportCache>,
) -> Result<GarfieldLdSelection, String> {
    if candidate_global_rows.is_empty() || target_k == 0 {
        return Ok(GarfieldLdSelection::default());
    }
    let target_k = target_k.max(1).min(candidate_global_rows.len());
    if candidate_global_rows.len() <= target_k {
        let rank_t0 = garfield_stage_profile_start();
        let priority = geneset_priority_order_from_scores(scores);
        garfield_stage_profile_end(rank_t0, &GARFIELD_CORR_STAGE1_POOL_PRIORITY_NS);
        return prune_candidate_rows_by_ld_priority_with_cache_limit_trace(
            candidate_global_rows,
            priority.as_slice(),
            logic_bits,
            sample_indices,
            support_cache,
            Some(target_k),
        )
        .map(|mut picked| {
            if picked.selected_global_rows.len() > target_k {
                picked.selected_global_rows.truncate(target_k);
            }
            picked
        });
    }
    let mut prescreen_k = corr_candidate_pool_prescreen_k(candidate_global_rows.len(), target_k);
    loop {
        let rank_t0 = garfield_stage_profile_start();
        let priority_local = topk_indices(scores, prescreen_k);
        garfield_stage_profile_end(rank_t0, &GARFIELD_CORR_STAGE1_POOL_PRIORITY_NS);
        // Only the score-prioritized prescreen rows can be retained by this
        // pass. Packing the rest of the window would add LD work without
        // changing the greedy result.
        let prescreen_global_rows = priority_local
            .iter()
            .filter_map(|&local_idx| candidate_global_rows.get(local_idx).copied())
            .collect::<Vec<_>>();
        let prescreen_priority_local = (0..prescreen_global_rows.len()).collect::<Vec<_>>();
        let mut pruned = prune_candidate_rows_by_ld_priority_with_cache_limit_trace(
            prescreen_global_rows.as_slice(),
            prescreen_priority_local.as_slice(),
            logic_bits,
            sample_indices,
            support_cache,
            Some(target_k),
        )?;
        if pruned.selected_global_rows.len() > target_k {
            pruned.selected_global_rows.truncate(target_k);
        }
        if pruned.selected_global_rows.len() >= target_k
            || prescreen_k >= candidate_global_rows.len()
        {
            return Ok(pruned);
        }
        let next_k = corr_candidate_pool_prescreen_k(
            candidate_global_rows.len(),
            prescreen_k.saturating_mul(2),
        );
        if next_k <= prescreen_k {
            return Ok(pruned);
        }
        prescreen_k = next_k;
    }
}

#[inline]
pub(super) fn geneset_corr_stage1_allow_parallel(
    unit_kind_lc: &str,
    n_region: usize,
    allow_parallel: bool,
) -> bool {
    allow_parallel
        && !(unit_kind_lc == "geneset" && n_region <= GARFIELD_GENESET_CORR_SERIAL_MAX_ROWS)
}

#[inline]
pub(super) fn resolve_ml_top_random_counts(keep_k: usize, top_frac: f64) -> (usize, usize) {
    if keep_k == 0 {
        return (0, 0);
    }
    if keep_k == 1 {
        return (1, 0);
    }
    let frac = top_frac.clamp(0.0, 1.0);
    if frac >= 1.0 {
        return (keep_k, 0);
    }
    if frac <= 0.0 {
        return (1, keep_k - 1);
    }
    let top_keep = ((keep_k as f64) * frac).floor() as usize;
    let top_keep = top_keep.clamp(1, keep_k - 1);
    (top_keep, keep_k - top_keep)
}

pub(super) fn select_ml_top_only_local_indices(
    dense_train: &[Vec<u8>],
    y_train: &[f64],
    response: ResponseKind,
    engine: MlEngine,
    tree_cfg: ExtraTreesConfig,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    keep_k: usize,
    feature_group_ids: Option<&[usize]>,
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
            let coarse_scores = compute_feature_scores_grouped(
                dense_train,
                y_train,
                response,
                MlEngine::Gbdt,
                tree_cfg,
                ImportanceKind::Imp,
                perm_cfg,
                feature_group_ids,
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
            let refined_group_ids = feature_group_ids.map(|groups| {
                coarse_local
                    .iter()
                    .map(|&idx| groups[idx])
                    .collect::<Vec<_>>()
            });
            let refined_scores = compute_feature_scores_grouped(
                refined_rows.as_slice(),
                y_train,
                response,
                MlEngine::Gbdt,
                tree_cfg,
                importance,
                perm_cfg,
                refined_group_ids.as_deref(),
            )?;
            let refined_local = topk_indices(refined_scores.as_slice(), keep_k);
            Ok(refined_local
                .into_iter()
                .map(|idx| coarse_local[idx])
                .collect())
        }
        _ => {
            let scores = compute_feature_scores_grouped(
                dense_train,
                y_train,
                response,
                engine,
                tree_cfg,
                importance,
                perm_cfg,
                feature_group_ids,
            )?;
            Ok(topk_indices(scores.as_slice(), keep_k))
        }
    }
}

pub(super) fn select_ml_top_local_indices_with_scores(
    dense_train: &[Vec<u8>],
    y_train: &[f64],
    response: ResponseKind,
    engine: MlEngine,
    tree_cfg: ExtraTreesConfig,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    keep_k: usize,
    keep_policy: GarfieldMlKeepPolicy,
    selection_seed: u64,
    feature_group_ids: Option<&[usize]>,
) -> Result<(Vec<usize>, Vec<f64>), String> {
    if keep_k == 0 || dense_train.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let n_region = dense_train.len();
    let keep_k = keep_k.min(n_region).max(1);
    let scores = match engine {
        MlEngine::Gbdt2 => compute_feature_scores_grouped(
            dense_train,
            y_train,
            response,
            MlEngine::Gbdt,
            tree_cfg,
            ImportanceKind::Imp,
            perm_cfg,
            feature_group_ids,
        )?,
        _ => compute_feature_scores_grouped(
            dense_train,
            y_train,
            response,
            engine,
            tree_cfg,
            importance,
            perm_cfg,
            feature_group_ids,
        )?,
    };
    let selected = match keep_policy {
        GarfieldMlKeepPolicy::TopK => topk_indices(scores.as_slice(), keep_k),
        GarfieldMlKeepPolicy::TopKPlusRandom { top_frac } => {
            let (top_keep, rand_keep) = resolve_ml_top_random_counts(keep_k, top_frac);
            let mut out = topk_indices(scores.as_slice(), top_keep);
            if rand_keep > 0 && out.len() < n_region {
                let mut picked = vec![false; n_region];
                for &idx in out.iter() {
                    picked[idx] = true;
                }
                let remaining = (0..n_region)
                    .filter(|&idx| !picked[idx])
                    .collect::<Vec<_>>();
                if !remaining.is_empty() {
                    let sample_n = rand_keep.min(remaining.len());
                    let mut rng = StdRng::seed_from_u64(selection_seed);
                    let sampled =
                        sample_indices_without_replacement(&mut rng, remaining.len(), sample_n);
                    let mut sampled_indices = sampled
                        .into_vec()
                        .into_iter()
                        .map(|idx| remaining[idx])
                        .collect::<Vec<_>>();
                    sampled_indices.sort_unstable();
                    out.extend(sampled_indices);
                }
            }
            out
        }
    };
    Ok((selected, scores))
}

#[inline]
pub(super) fn insert_topk_density_hit(
    bucket: &mut Vec<(usize, f64)>,
    n_not: usize,
    score: f64,
    keep_k: usize,
) {
    bucket.push((n_not, score));
    bucket.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    if bucket.len() > keep_k {
        bucket.truncate(keep_k);
    }
}

pub(super) fn select_ml_top_local_indices(
    dense_train: &[Vec<u8>],
    y_train: &[f64],
    response: ResponseKind,
    engine: MlEngine,
    tree_cfg: ExtraTreesConfig,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    keep_k: usize,
    keep_policy: GarfieldMlKeepPolicy,
    selection_seed: u64,
    feature_group_ids: Option<&[usize]>,
) -> Result<Vec<usize>, String> {
    let (selected, _) = select_ml_top_local_indices_with_scores(
        dense_train,
        y_train,
        response,
        engine,
        tree_cfg,
        importance,
        perm_cfg,
        keep_k,
        keep_policy,
        selection_seed,
        feature_group_ids,
    )?;
    Ok(selected)
}

pub(super) fn proposal_pool_for_rows(
    proposal_config: ProposalConfig,
    window_id: &str,
    marker_ids: &[usize],
    genotype_rows: &[Vec<u8>],
    residual: &[f64],
    replicate_seed: u64,
) -> Result<proposal::MergedSitePool, String> {
    let maf = genotype_rows
        .iter()
        .map(|row| {
            if row.is_empty() {
                0.0
            } else {
                let allele_frequency = row.iter().map(|&value| f64::from(value)).sum::<f64>()
                    / (2.0 * row.len() as f64);
                allele_frequency.min(1.0 - allele_frequency).clamp(0.0, 0.5)
            }
        })
        .collect::<Vec<_>>();
    build_proposal_site_pool(
        proposal_config,
        window_id,
        marker_ids,
        genotype_rows,
        residual,
        maf.as_slice(),
        GARFIELD_LD_CLUMP_R2_DEFAULT,
        replicate_seed,
    )
}

/// Build a Corr-only proposal pool from genotype bitplanes.  Corr's dense
/// implementation binarizes every non-zero dosage, so the GE1 stage-1
/// summaries are sufficient for the statistic.  Only the ranked proposals
/// (at most `k_site`) are densified afterwards for LD credit-set clustering;
/// the full window never materializes as `Vec<Vec<u8>>`.
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_corr_proposal_pool(
    proposal_config: ProposalConfig,
    candidate_global_rows: &[usize],
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    y_train: &[f64],
    stage1_y_lookup: Option<&PackedYSumLookup>,
    allow_parallel: bool,
) -> Result<proposal::MergedSitePool, String> {
    if !matches!(proposal_config.mode, ProposalMode::Corr) {
        return Err("packed Corr proposal pool requires Corr mode".to_string());
    }
    if candidate_global_rows.is_empty() {
        return Ok(proposal::MergedSitePool {
            representatives: Vec::new(),
            clusters: Vec::new(),
            proxy_to_representative: BTreeMap::new(),
            k_pre: 0,
            k_site: 0,
        });
    }
    let scores = packed_corr_scores_from_full_bits(
        candidate_global_rows,
        logic_bits,
        train_idx_local,
        y_train,
        stage1_y_lookup,
        allow_parallel,
    )?;
    let proposals = CorrChannel::ranked_proposals(
        candidate_global_rows,
        scores.as_slice(),
        proposal_config.k_site,
    );
    if proposals.is_empty() {
        return Ok(proposal::MergedSitePool {
            representatives: Vec::new(),
            clusters: Vec::new(),
            proxy_to_representative: BTreeMap::new(),
            k_pre: 0,
            k_site: 0,
        });
    }
    let proposal_rows = proposals
        .iter()
        .map(|proposal| proposal.marker_id)
        .collect::<Vec<_>>();
    let genotype_rows = dense_dosage_rows_from_full_bits_cached(
        logic_bits.bits(),
        logic_bits.bits_hi_flat.as_deref(),
        logic_bits.row_words,
        proposal_rows.as_slice(),
        train_idx_local,
        logic_bits.sites.len(),
        logic_bits.n_samples,
    )?;
    let marker_rows = proposal_rows
        .iter()
        .copied()
        .zip(genotype_rows)
        .collect::<BTreeMap<_, _>>();
    merge_and_select_site_pool(
        proposals,
        &marker_rows,
        GARFIELD_LD_CLUMP_R2_DEFAULT,
        ChannelQuota::new([(ProposalChannelId::Corr, proposal_config.k_site)]),
        proposal_config.k_site,
    )
}

pub(super) fn packed_corr_scores_from_full_bits(
    candidate_global_rows: &[usize],
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    y_train: &[f64],
    stage1_y_lookup: Option<&PackedYSumLookup>,
    allow_parallel: bool,
) -> Result<Vec<f64>, String> {
    let summary_t0 = garfield_stage_profile_start();
    let summaries = dosage_stage1_dual_summaries_from_full_bits(
        logic_bits.bits(),
        logic_bits.bits_hi_flat.as_deref(),
        logic_bits.row_words,
        candidate_global_rows,
        train_idx_local,
        y_train,
        logic_bits.sites.len(),
        logic_bits.n_samples,
        stage1_y_lookup,
        allow_parallel,
    )?;
    garfield_stage_profile_end(summary_t0, &GARFIELD_CORR_STAGE1_SUMMARY_NS);
    let total_sum_y = y_train.iter().copied().sum::<f64>();
    let total_sum_y2 = y_train.iter().map(|value| value * value).sum::<f64>();
    let score_t0 = garfield_stage_profile_start();
    let scores = corr_scores_from_dual_summaries(
        summaries.as_slice(),
        total_sum_y,
        total_sum_y2,
        y_train.len(),
        allow_parallel,
    );
    garfield_stage_profile_end(score_t0, &GARFIELD_CORR_STAGE1_SCORE_NS);
    Ok(scores)
}

/// Packed Corr+RF proposal path: calculate Corr over the full packed window,
/// decode only a Corr-ranked shortlist, then fit RF on that shortlist.
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_corr_rf_proposal_pool(
    proposal_config: ProposalConfig,
    candidate_global_rows: &[usize],
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    y_train: &[f64],
    stage1_y_lookup: Option<&PackedYSumLookup>,
    allow_parallel: bool,
    replicate_seed: u64,
    window_id: &str,
) -> Result<proposal::MergedSitePool, String> {
    if !matches!(proposal_config.mode, ProposalMode::CorrRf) {
        return Err("packed Corr+RF proposal pool requires CorrRf mode".to_string());
    }
    let scores = packed_corr_scores_from_full_bits(
        candidate_global_rows,
        logic_bits,
        train_idx_local,
        y_train,
        stage1_y_lookup,
        allow_parallel,
    )?;
    let shortlist_k = proposal_config.k_site.min(candidate_global_rows.len());
    let shortlist_local = topk_indices(scores.as_slice(), shortlist_k);
    let rf_marker_ids = shortlist_local
        .iter()
        .map(|&local_idx| candidate_global_rows[local_idx])
        .collect::<Vec<_>>();
    let rf_genotype_rows = dense_dosage_rows_from_full_bits_cached(
        logic_bits.bits(),
        logic_bits.bits_hi_flat.as_deref(),
        logic_bits.row_words,
        rf_marker_ids.as_slice(),
        train_idx_local,
        logic_bits.sites.len(),
        logic_bits.n_samples,
    )?;
    self::proposal::build_corr_rf_site_pool_from_ranked_corr(
        proposal_config,
        window_id,
        candidate_global_rows,
        scores.as_slice(),
        rf_marker_ids.as_slice(),
        rf_genotype_rows.as_slice(),
        y_train,
        GARFIELD_LD_CLUMP_R2_DEFAULT,
        replicate_seed,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn select_logic_unit_global_rows(
    unit: &GarfieldLogicUnit,
    unit_kind_lc: &str,
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    y_train: &[f64],
    stage1_y_lookup: Option<&PackedYSumLookup>,
    allow_parallel: bool,
) -> Result<Option<GarfieldSelectedRows>, String> {
    select_logic_unit_global_rows_with_ld_cache(
        unit,
        unit_kind_lc,
        response,
        engine,
        importance,
        perm_cfg,
        ml_top_k,
        ml_top_frac,
        tree_cfg,
        logic_bits,
        train_idx_local,
        y_train,
        stage1_y_lookup,
        allow_parallel,
        None,
        false,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select_logic_unit_global_rows_with_ld_cache(
    unit: &GarfieldLogicUnit,
    unit_kind_lc: &str,
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    y_train: &[f64],
    stage1_y_lookup: Option<&PackedYSumLookup>,
    allow_parallel: bool,
    support_cache: Option<&GarfieldLdSupportCache>,
    diagnostics: bool,
    proposal_config_override: Option<ProposalConfig>,
) -> Result<Option<GarfieldSelectedRows>, String> {
    check_ctrlc()?;
    if unit.indices.is_empty() {
        return Ok(None);
    }
    // Keep the full unit candidate set until the engine-specific single-site
    // score is available.  LD clumping is applied once, after this stage, for
    // every unit kind and engine rather than only for multi-span genesets.
    let candidate_global_rows = unit.indices.clone();
    if candidate_global_rows.is_empty() {
        return Ok(None);
    }
    let stage1_candidate_global_rows = diagnostics.then(|| candidate_global_rows.clone());
    let proposal_config = proposal_config_override.unwrap_or(resolve_proposal_config()?);
    if !matches!(proposal_config.mode, ProposalMode::Off) {
        if matches!(proposal_config.mode, ProposalMode::Corr) {
            let pool_t0 = garfield_stage_profile_start();
            let pool = packed_corr_proposal_pool(
                proposal_config,
                candidate_global_rows.as_slice(),
                logic_bits,
                train_idx_local,
                y_train,
                stage1_y_lookup,
                allow_parallel,
            )?;
            garfield_stage_profile_end(pool_t0, &GARFIELD_CORR_STAGE1_POOL_NS);
            if pool.representatives.is_empty() {
                return Ok(None);
            }
            return Ok(Some(GarfieldSelectedRows {
                selected_global_rows: pool.representatives,
                stage1_candidate_global_rows,
                // Proposal clusters retain a complete proxy map inside the
                // proposal contract.  The legacy compression record is left
                // empty here until diagnostics gain a source-aware schema.
                ld_compression: Vec::new(),
                train_literal_scores: None,
            }));
        }
        if matches!(proposal_config.mode, ProposalMode::CorrRf) {
            let pool_t0 = garfield_stage_profile_start();
            let pool = packed_corr_rf_proposal_pool(
                proposal_config,
                candidate_global_rows.as_slice(),
                logic_bits,
                train_idx_local,
                y_train,
                stage1_y_lookup,
                allow_parallel,
                proposal_config.k_site as u64 ^ tree_cfg.seed,
                unit.label.as_str(),
            )?;
            garfield_stage_profile_end(pool_t0, &GARFIELD_CORR_STAGE1_POOL_NS);
            if pool.representatives.is_empty() {
                return Ok(None);
            }
            return Ok(Some(GarfieldSelectedRows {
                selected_global_rows: pool.representatives,
                stage1_candidate_global_rows,
                ld_compression: Vec::new(),
                train_literal_scores: None,
            }));
        }
        // Proposal mode intentionally starts from the complete unit.  It
        // creates its own Corr/RF union and LD credit-set pool, so the legacy
        // engine-specific stage-1 ranking is not silently mixed into the
        // development comparison.
        let genotype_rows = dense_dosage_rows_from_full_bits(
            logic_bits.bits(),
            logic_bits.bits_hi_flat.as_deref(),
            logic_bits.row_words,
            candidate_global_rows.as_slice(),
            train_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        let pool = proposal_pool_for_rows(
            proposal_config,
            unit.label.as_str(),
            candidate_global_rows.as_slice(),
            genotype_rows.as_slice(),
            y_train,
            proposal_config.k_site as u64 ^ tree_cfg.seed,
        )?;
        if pool.representatives.is_empty() {
            return Ok(None);
        }
        return Ok(Some(GarfieldSelectedRows {
            selected_global_rows: pool.representatives,
            stage1_candidate_global_rows,
            // Proposal clusters retain a complete proxy map inside the new
            // proposal contract.  The legacy compression record is left
            // empty here until diagnostics gain a source-aware schema.
            ld_compression: Vec::new(),
            train_literal_scores: None,
        }));
    }
    let selected_global_rows = if let Some(engine_one) = engine {
        let n_region = candidate_global_rows.len();
        let beam_keep_k = resolve_ml_keep_k(n_region, ml_top_k, ml_top_frac)
            .max(geneset_min_keep_k(unit_kind_lc, unit))
            .min(n_region);
        let candidate_pool_k =
            resolve_stage1_candidate_pool_k(unit_kind_lc, unit, n_region, ml_top_k, ml_top_frac);

        // ---- PairwiseAnd fast path: packed subset + cached stage-1 stats ----
        if engine_one == MlEngine::PairwiseAnd {
            let t0 = Instant::now();
            let (
                packed_bits_ge1,
                packed_bits_ge2,
                row_words_sub,
                feat_sum_x,
                feat_sum_x2,
                feat_sum_xy,
            ) = packed_dosage_rows_subset_from_full_bits_with_stage1(
                logic_bits.bits(),
                logic_bits.bits_hi_flat.as_deref(),
                logic_bits.row_words,
                candidate_global_rows.as_slice(),
                train_idx_local,
                y_train,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            let scores = feature_scores_pairwise_and_packed_dual_with_stage1(
                packed_bits_ge1.as_slice(),
                packed_bits_ge2.as_slice(),
                row_words_sub,
                n_region,
                y_train,
                train_idx_local.len(),
                feat_sum_x.as_slice(),
                feat_sum_x2.as_slice(),
                feat_sum_xy.as_slice(),
            );
            let selected = if unit_kind_lc == "geneset" && unit.spans.len() > 1 {
                select_geneset_window_candidate_pool_rows_with_cache_trace(
                    unit,
                    candidate_global_rows.as_slice(),
                    scores.as_slice(),
                    beam_keep_k,
                    logic_bits,
                    train_idx_local,
                    true,
                    support_cache,
                )?
            } else {
                select_single_window_candidate_pool_rows_with_cache_trace(
                    candidate_global_rows.as_slice(),
                    scores.as_slice(),
                    candidate_pool_k,
                    logic_bits,
                    train_idx_local,
                    support_cache,
                )?
            };
            GARFIELD_ML_SELECT_NS.fetch_add(
                t0.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            if selected.selected_global_rows.is_empty() {
                return Ok(None);
            }
            return Ok(Some(GarfieldSelectedRows {
                selected_global_rows: selected.selected_global_rows,
                stage1_candidate_global_rows,
                ld_compression: selected.compressed,
                train_literal_scores: None,
            }));
        }

        if engine_one == MlEngine::Corr {
            let t0 = Instant::now();
            let corr_allow_parallel =
                geneset_corr_stage1_allow_parallel(unit_kind_lc, n_region, allow_parallel);
            let summary_t0 = garfield_stage_profile_start();
            let summaries = dosage_stage1_dual_summaries_from_full_bits(
                logic_bits.bits(),
                logic_bits.bits_hi_flat.as_deref(),
                logic_bits.row_words,
                candidate_global_rows.as_slice(),
                train_idx_local,
                y_train,
                logic_bits.sites.len(),
                logic_bits.n_samples,
                stage1_y_lookup,
                corr_allow_parallel,
            )?;
            garfield_stage_profile_end(summary_t0, &GARFIELD_CORR_STAGE1_SUMMARY_NS);
            let total_sum_y = y_train.iter().copied().sum::<f64>();
            let score_t0 = garfield_stage_profile_start();
            let scores = dosage_stage1_raw_scores_from_dual_summaries(
                summaries.as_slice(),
                total_sum_y,
                y_train.len(),
                corr_allow_parallel,
            );
            garfield_stage_profile_end(score_t0, &GARFIELD_CORR_STAGE1_SCORE_NS);
            let pool_t0 = garfield_stage_profile_start();
            let selected = if unit_kind_lc == "geneset" && unit.spans.len() > 1 {
                select_geneset_window_candidate_pool_rows_with_cache_trace(
                    unit,
                    candidate_global_rows.as_slice(),
                    scores.as_slice(),
                    beam_keep_k,
                    logic_bits,
                    train_idx_local,
                    true,
                    support_cache,
                )?
            } else {
                select_single_window_candidate_pool_rows_with_cache_trace(
                    candidate_global_rows.as_slice(),
                    scores.as_slice(),
                    candidate_pool_k,
                    logic_bits,
                    train_idx_local,
                    support_cache,
                )?
            };
            garfield_stage_profile_end(pool_t0, &GARFIELD_CORR_STAGE1_POOL_NS);
            GARFIELD_ML_SELECT_NS.fetch_add(
                t0.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            if selected.selected_global_rows.is_empty() {
                return Ok(None);
            }
            return Ok(Some(GarfieldSelectedRows {
                train_literal_scores: Some(
                    build_cached_literal_scores_from_selected_dual_summaries(
                        candidate_global_rows.as_slice(),
                        selected.selected_global_rows.as_slice(),
                        summaries.as_slice(),
                        total_sum_y,
                        y_train.len(),
                        logic_bits.bits_hi_flat.is_none(),
                    )?,
                ),
                selected_global_rows: selected.selected_global_rows,
                stage1_candidate_global_rows,
                ld_compression: selected.compressed,
            }));
        }

        // ---- General ML path (Vec<Vec<u8>>) ----
        let dense_train = dense_dosage_rows_from_full_bits(
            logic_bits.bits(),
            logic_bits.bits_hi_flat.as_deref(),
            logic_bits.row_words,
            candidate_global_rows.as_slice(),
            train_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        if dense_train.is_empty() {
            return Ok(None);
        }
        let ml_group_ids = build_unit_window_group_ids(
            unit,
            candidate_global_rows.as_slice(),
            logic_bits.sites.as_slice(),
            unit_kind_lc,
        );
        let (_top_local_base, scores) = select_ml_top_local_indices_with_scores(
            dense_train.as_slice(),
            y_train,
            response,
            engine_one,
            ExtraTreesConfig {
                allow_parallel,
                ..tree_cfg
            },
            importance,
            perm_cfg,
            if unit_kind_lc == "geneset" && unit.spans.len() > 1 {
                beam_keep_k.saturating_mul(unit.spans.len()).min(n_region)
            } else {
                candidate_pool_k
            },
            GarfieldMlKeepPolicy::TopK,
            tree_cfg.seed ^ 0xB6D5_0C11_8E91_3F27,
            ml_group_ids.as_deref(),
        )?;
        let selected = if unit_kind_lc == "geneset" && unit.spans.len() > 1 {
            select_geneset_window_candidate_pool_rows_with_cache_trace(
                unit,
                candidate_global_rows.as_slice(),
                scores.as_slice(),
                beam_keep_k,
                logic_bits,
                train_idx_local,
                true,
                support_cache,
            )?
        } else {
            select_single_window_candidate_pool_rows_with_cache_trace(
                candidate_global_rows.as_slice(),
                scores.as_slice(),
                candidate_pool_k,
                logic_bits,
                train_idx_local,
                support_cache,
            )?
        };
        if selected.selected_global_rows.is_empty() {
            return Ok(None);
        }
        GarfieldSelectedRows {
            selected_global_rows: selected.selected_global_rows,
            stage1_candidate_global_rows,
            ld_compression: selected.compressed,
            train_literal_scores: None,
        }
    } else {
        let n_region = candidate_global_rows.len();
        let summaries = dosage_stage1_dual_summaries_from_full_bits(
            logic_bits.bits(),
            logic_bits.bits_hi_flat.as_deref(),
            logic_bits.row_words,
            candidate_global_rows.as_slice(),
            train_idx_local,
            y_train,
            logic_bits.sites.len(),
            logic_bits.n_samples,
            stage1_y_lookup,
            allow_parallel,
        )?;
        let scores = dosage_stage1_raw_scores_from_dual_summaries(
            summaries.as_slice(),
            y_train.iter().copied().sum::<f64>(),
            y_train.len(),
            allow_parallel,
        );
        let selected = if unit_kind_lc == "geneset" && unit.spans.len() > 1 {
            select_geneset_window_candidate_pool_rows_with_cache_trace(
                unit,
                candidate_global_rows.as_slice(),
                scores.as_slice(),
                n_region,
                logic_bits,
                train_idx_local,
                true,
                support_cache,
            )?
        } else {
            select_single_window_candidate_pool_rows_with_cache_trace(
                candidate_global_rows.as_slice(),
                scores.as_slice(),
                n_region,
                logic_bits,
                train_idx_local,
                support_cache,
            )?
        };
        GarfieldSelectedRows {
            selected_global_rows: selected.selected_global_rows,
            stage1_candidate_global_rows,
            ld_compression: selected.compressed,
            train_literal_scores: None,
        }
    };
    if selected_global_rows.selected_global_rows.is_empty() {
        return Ok(None);
    }
    Ok(Some(selected_global_rows))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_logic_unit_continuous(
    unit: &GarfieldLogicUnit,
    unit_kind_lc: &str,
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    _test_idx_local: &[usize],
    y_train: &[f64],
    stage1_y_lookup: Option<&PackedYSumLookup>,
    beam_params: BeamSearchParams,
) -> Result<Option<GarfieldUnitPrepared>, String> {
    prepare_logic_unit_continuous_with_ld_cache(
        unit,
        unit_kind_lc,
        response,
        engine,
        importance,
        perm_cfg,
        ml_top_k,
        ml_top_frac,
        tree_cfg,
        logic_bits,
        train_idx_local,
        _test_idx_local,
        y_train,
        stage1_y_lookup,
        beam_params.clone(),
        None,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_logic_unit_continuous_with_ld_cache(
    unit: &GarfieldLogicUnit,
    unit_kind_lc: &str,
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    _test_idx_local: &[usize],
    y_train: &[f64],
    stage1_y_lookup: Option<&PackedYSumLookup>,
    beam_params: BeamSearchParams,
    support_cache: Option<&GarfieldLdSupportCache>,
    diagnostics: bool,
) -> Result<Option<GarfieldUnitPrepared>, String> {
    check_ctrlc()?;
    let Some(selected_rows) = select_logic_unit_global_rows_with_ld_cache(
        unit,
        unit_kind_lc,
        response,
        engine,
        importance,
        perm_cfg,
        ml_top_k,
        ml_top_frac,
        tree_cfg,
        logic_bits,
        train_idx_local,
        y_train,
        stage1_y_lookup,
        beam_params.allow_parallel,
        support_cache,
        diagnostics,
        Some(effective_proposal_config(
            beam_params.search_backend,
            beam_params.max_pick,
        )?),
    )?
    else {
        return Ok(None);
    };
    let selected_global_rows = selected_rows.selected_global_rows;
    if !selected_rows.ld_compression.is_empty() {
        GARFIELD_LD_CLUMP_ROWS_COMPRESSED.fetch_add(
            u64::try_from(selected_rows.ld_compression.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        GARFIELD_LD_CLUMP_UNITS.fetch_add(1, Ordering::Relaxed);
    }

    let local_groups = build_unit_window_group_ids(
        unit,
        selected_global_rows.as_slice(),
        logic_bits.sites.as_slice(),
        unit_kind_lc,
    )
    .unwrap_or_else(|| {
        selected_global_rows
            .iter()
            .map(|&idx| logic_bits.group_ids[idx])
            .collect::<Vec<_>>()
    });

    let geneset_stage_group_target =
        geneset_stage_group_target(unit_kind_lc, unit, local_groups.as_slice());
    let null_unit_group_count = null_unit_group_count_for_penalty(unit_kind_lc, unit);
    Ok(Some(GarfieldUnitPrepared {
        selected_global_rows,
        stage1_candidate_global_rows: selected_rows.stage1_candidate_global_rows,
        ld_compression: selected_rows.ld_compression,
        local_groups,
        geneset_stage_group_target,
        null_unit_group_bin: null_unit_group_bin_for_group_count(
            unit_kind_lc,
            null_unit_group_count,
        ),
        train_literal_scores: selected_rows.train_literal_scores,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_logic_chunk_continuous(
    chunk: GarfieldNullChunk,
    row_mul: usize,
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    _test_idx_local: &[usize],
    y_train: &[f64],
    beam_params: BeamSearchParams,
) -> Result<Option<GarfieldUnitPrepared>, String> {
    check_ctrlc()?;
    if chunk.end_index <= chunk.start_index || row_mul == 0 {
        return Ok(None);
    }
    let row_start = chunk.start_index.saturating_mul(row_mul);
    let row_end = chunk.end_index.saturating_mul(row_mul);
    if row_end > logic_bits.sites.len() {
        return Ok(None);
    }

    let n_region = row_end.saturating_sub(row_start);
    let keep_k = resolve_ml_keep_k(n_region, ml_top_k, ml_top_frac);
    let keep_policy = GarfieldMlKeepPolicy::TopKPlusRandom {
        top_frac: GARFIELD_NULL_ML_TOP_FRAC,
    };
    let selection_seed = perm_cfg.seed ^ chunk.window_id ^ 0x94D0_49BB_1331_11EB;

    let proposal_config =
        effective_proposal_config(beam_params.search_backend, beam_params.max_pick)?;
    if !matches!(proposal_config.mode, ProposalMode::Off) {
        let candidate_global_rows = (row_start..row_end).collect::<Vec<_>>();
        if matches!(proposal_config.mode, ProposalMode::Corr) {
            let pool = packed_corr_proposal_pool(
                proposal_config,
                candidate_global_rows.as_slice(),
                logic_bits,
                train_idx_local,
                y_train,
                None,
                beam_params.allow_parallel,
            )?;
            if pool.representatives.is_empty() {
                return Ok(None);
            }
            let local_groups = pool
                .representatives
                .iter()
                .map(|&idx| logic_bits.group_ids[idx])
                .collect::<Vec<_>>();
            return Ok(Some(GarfieldUnitPrepared {
                selected_global_rows: pool.representatives,
                stage1_candidate_global_rows: None,
                ld_compression: Vec::new(),
                local_groups,
                geneset_stage_group_target: None,
                null_unit_group_bin: 0,
                train_literal_scores: None,
            }));
        }
        if matches!(proposal_config.mode, ProposalMode::CorrRf) {
            let window_id = format!("null-chunk-{}", chunk.window_id);
            let pool = packed_corr_rf_proposal_pool(
                proposal_config,
                candidate_global_rows.as_slice(),
                logic_bits,
                train_idx_local,
                y_train,
                None,
                beam_params.allow_parallel,
                selection_seed,
                window_id.as_str(),
            )?;
            if pool.representatives.is_empty() {
                return Ok(None);
            }
            let local_groups = pool
                .representatives
                .iter()
                .map(|&idx| logic_bits.group_ids[idx])
                .collect::<Vec<_>>();
            return Ok(Some(GarfieldUnitPrepared {
                selected_global_rows: pool.representatives,
                stage1_candidate_global_rows: None,
                ld_compression: Vec::new(),
                local_groups,
                geneset_stage_group_target: None,
                null_unit_group_bin: 0,
                train_literal_scores: None,
            }));
        }
        let genotype_rows = dense_dosage_rows_from_full_bits_range(
            logic_bits.bits(),
            logic_bits.bits_hi_flat.as_deref(),
            logic_bits.row_words,
            row_start,
            row_end,
            train_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        let window_id = format!("null-chunk-{}", chunk.window_id);
        let pool = proposal_pool_for_rows(
            proposal_config,
            window_id.as_str(),
            candidate_global_rows.as_slice(),
            genotype_rows.as_slice(),
            y_train,
            selection_seed,
        )?;
        if pool.representatives.is_empty() {
            return Ok(None);
        }
        let local_groups = pool
            .representatives
            .iter()
            .map(|&idx| logic_bits.group_ids[idx])
            .collect::<Vec<_>>();
        return Ok(Some(GarfieldUnitPrepared {
            selected_global_rows: pool.representatives,
            stage1_candidate_global_rows: None,
            ld_compression: Vec::new(),
            local_groups,
            geneset_stage_group_target: None,
            null_unit_group_bin: 0,
            train_literal_scores: None,
        }));
    }

    let selected_global_rows = if let Some(engine_one) = engine {
        // ---- PairwiseAnd fast path: packed subset + cached stage-1 stats ----
        if engine_one == MlEngine::PairwiseAnd {
            let (
                packed_bits_ge1,
                packed_bits_ge2,
                row_words_sub,
                feat_sum_x,
                feat_sum_x2,
                feat_sum_xy,
            ) = packed_dosage_rows_subset_from_full_bits_range_with_stage1(
                logic_bits.bits(),
                logic_bits.bits_hi_flat.as_deref(),
                logic_bits.row_words,
                row_start,
                row_end,
                train_idx_local,
                y_train,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            let scores = feature_scores_pairwise_and_packed_dual_with_stage1(
                packed_bits_ge1.as_slice(),
                packed_bits_ge2.as_slice(),
                row_words_sub,
                n_region,
                y_train,
                train_idx_local.len(),
                feat_sum_x.as_slice(),
                feat_sum_x2.as_slice(),
                feat_sum_xy.as_slice(),
            );
            let (top_keep, rand_keep) =
                resolve_ml_top_random_counts(keep_k, GARFIELD_NULL_ML_TOP_FRAC);
            let mut top_local = topk_indices(&scores, top_keep.max(1));
            if rand_keep > 0 && top_local.len() < n_region {
                let mut picked = vec![false; n_region];
                for &idx in top_local.iter() {
                    if idx < n_region {
                        picked[idx] = true;
                    }
                }
                let remaining: Vec<usize> = (0..n_region).filter(|&i| !picked[i]).collect();
                if !remaining.is_empty() {
                    let sample_n = rand_keep.min(remaining.len());
                    let mut rng = StdRng::seed_from_u64(selection_seed);
                    let sampled =
                        sample_indices_without_replacement(&mut rng, remaining.len(), sample_n);
                    let mut extra: Vec<usize> = sampled
                        .into_vec()
                        .into_iter()
                        .map(|i| remaining[i])
                        .collect();
                    extra.sort_unstable();
                    top_local.extend(extra);
                }
            }
            if top_local.is_empty() {
                return Ok(None);
            }
            // Fall through to common path: convert local → global indices
            top_local
                .into_iter()
                .map(|idx| row_start + idx)
                .collect::<Vec<_>>()
        } else if engine_one == MlEngine::Corr {
            let t0 = Instant::now();
            let summaries = dosage_stage1_dual_summaries_from_full_bits_range(
                logic_bits.bits(),
                logic_bits.bits_hi_flat.as_deref(),
                logic_bits.row_words,
                row_start,
                row_end,
                train_idx_local,
                y_train,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            let scores = dosage_stage1_raw_scores_from_dual_summaries(
                summaries.as_slice(),
                y_train.iter().copied().sum::<f64>(),
                y_train.len(),
                true,
            );
            let (top_keep, rand_keep) =
                resolve_ml_top_random_counts(keep_k, GARFIELD_NULL_ML_TOP_FRAC);
            let mut top_local = topk_indices(&scores, top_keep.max(1));
            if rand_keep > 0 && top_local.len() < n_region {
                let mut picked = vec![false; n_region];
                for &idx in top_local.iter() {
                    if idx < n_region {
                        picked[idx] = true;
                    }
                }
                let remaining: Vec<usize> = (0..n_region).filter(|&i| !picked[i]).collect();
                if !remaining.is_empty() {
                    let sample_n = rand_keep.min(remaining.len());
                    let mut rng = StdRng::seed_from_u64(selection_seed);
                    let sampled =
                        sample_indices_without_replacement(&mut rng, remaining.len(), sample_n);
                    let mut extra: Vec<usize> = sampled
                        .into_vec()
                        .into_iter()
                        .map(|i| remaining[i])
                        .collect();
                    extra.sort_unstable();
                    top_local.extend(extra);
                }
            }
            GARFIELD_ML_SELECT_NS.fetch_add(
                t0.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            if top_local.is_empty() {
                return Ok(None);
            }
            top_local
                .into_iter()
                .map(|idx| row_start + idx)
                .collect::<Vec<_>>()
        } else {
            // ---- General ML path ----
            let dense_train = dense_dosage_rows_from_full_bits_range(
                logic_bits.bits(),
                logic_bits.bits_hi_flat.as_deref(),
                logic_bits.row_words,
                row_start,
                row_end,
                train_idx_local,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            if dense_train.is_empty() {
                return Ok(None);
            }
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
                keep_policy,
                selection_seed,
                None,
            )?;
            if top_local.is_empty() {
                return Ok(None);
            }
            top_local
                .into_iter()
                .map(|idx| row_start + idx)
                .collect::<Vec<_>>()
        }
    } else {
        // Window null calibration must retain a bounded, genotype-only pool.
        // Taking all rows here would make the null stage scale with the full
        // genome (the physical null chunks can contain tens of thousands of
        // markers), while using the observed ML ranking would leak phenotype
        // selection into the null.  A deterministic genotype-row sample keeps
        // the candidate budget comparable to the observed scan without using
        // y_train.
        let sample_n = keep_k.min(n_region).max(1);
        if sample_n >= n_region {
            (row_start..row_end).collect::<Vec<_>>()
        } else {
            let mut rng = StdRng::seed_from_u64(selection_seed ^ 0xD1B5_4A32_D192_ED03);
            let mut selected = sample_indices_without_replacement(&mut rng, n_region, sample_n)
                .into_vec()
                .into_iter()
                .map(|idx| row_start + idx)
                .collect::<Vec<_>>();
            selected.sort_unstable();
            selected
        }
    };
    if selected_global_rows.is_empty() {
        return Ok(None);
    }

    // Null chunks use the same final LD clump as the observed scan.  The
    // selected order already places ML/Corr top hits before random rescue
    // rows, so it is a deterministic priority order for this stage.
    let null_priority = (0..selected_global_rows.len()).collect::<Vec<_>>();
    let selected = prune_candidate_rows_by_ld_priority_with_cache_limit_trace(
        selected_global_rows.as_slice(),
        null_priority.as_slice(),
        logic_bits,
        train_idx_local,
        None,
        None,
    )?;
    let selected_global_rows = selected.selected_global_rows;
    if selected_global_rows.is_empty() {
        return Ok(None);
    }
    if !selected.compressed.is_empty() {
        GARFIELD_LD_CLUMP_ROWS_COMPRESSED.fetch_add(
            u64::try_from(selected.compressed.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        GARFIELD_LD_CLUMP_UNITS.fetch_add(1, Ordering::Relaxed);
    }

    let local_groups = selected_global_rows
        .iter()
        .map(|&idx| logic_bits.group_ids[idx])
        .collect::<Vec<_>>();

    Ok(Some(GarfieldUnitPrepared {
        selected_global_rows,
        stage1_candidate_global_rows: None,
        ld_compression: selected.compressed,
        local_groups,
        geneset_stage_group_target: None,
        null_unit_group_bin: 0,
        train_literal_scores: None,
    }))
}

pub(super) fn collect_rule_permutation_nulls_for_repeat(
    prepared: &GarfieldUnitPrepared,
    prepared_bits: &GarfieldUnitBitMatrices<'_>,
    y_train: &[f64],
    y_test: &[f64],
    split_applied: bool,
    beam_params: BeamSearchParams,
    keep_topk: usize,
    rep_seed: u64,
    rep_index: usize,
    null_y_replicates: Option<&[Vec<f64>]>,
) -> Result<Vec<GarfieldPermutationNullScores>, String> {
    let (perm_train, perm_test) = permutation_y_for_repeat(
        y_train,
        y_test,
        split_applied,
        rep_seed,
        rep_index,
        null_y_replicates,
    )?;
    collect_rule_permutation_nulls_for_repeat_with_y(
        prepared,
        prepared_bits,
        perm_train.as_slice(),
        perm_test.as_slice(),
        split_applied,
        beam_params,
        keep_topk,
    )
}

pub(super) fn permutation_y_for_repeat(
    y_train: &[f64],
    y_test: &[f64],
    split_applied: bool,
    rep_seed: u64,
    rep_index: usize,
    null_y_replicates: Option<&[Vec<f64>]>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let perm_train = if let Some(nulls) = null_y_replicates.filter(|nulls| !nulls.is_empty()) {
        let null_y = nulls.get(rep_index % nulls.len()).ok_or_else(|| {
            format!(
                "GARFIELD conditional null replicate {} is unavailable (n={})",
                rep_index,
                nulls.len()
            )
        })?;
        if null_y.len() != y_train.len() {
            return Err(format!(
                "GARFIELD conditional null length mismatch: got {}, expected {}",
                null_y.len(),
                y_train.len()
            ));
        }
        null_y.to_vec()
    } else {
        shuffled_copy_f64(y_train, rep_seed)
    };
    let perm_test = if split_applied {
        shuffled_copy_f64(y_test, rep_seed ^ 0xD1B5_4A32_D192_ED03)
    } else {
        perm_train.clone()
    };
    Ok((perm_train, perm_test))
}

pub(super) fn collect_rule_permutation_nulls_for_repeat_with_y(
    prepared: &GarfieldUnitPrepared,
    prepared_bits: &GarfieldUnitBitMatrices<'_>,
    perm_train: &[f64],
    perm_test: &[f64],
    _split_applied: bool,
    beam_params: BeamSearchParams,
    keep_topk: usize,
) -> Result<Vec<GarfieldPermutationNullScores>, String> {
    if prepared.selected_global_rows.is_empty() {
        return Ok(Vec::new());
    }
    let beam_params = beam_params_for_prepared(prepared, beam_params);
    let null_complexity_bin = beam_params.null_complexity_bin;
    let null_max_rule_len = beam_params.max_pick.max(1);
    let perm_hits = beam_search_train_test_continuous_dispatch(
        perm_train,
        prepared_bits,
        prepared.selected_global_rows.len(),
        perm_test,
        prepared.local_groups.as_slice(),
        beam_params.clone(),
        None,
    )?;
    let mut out = Vec::<GarfieldPermutationNullScores>::new();

    for (bucket, score) in collect_topk_rule_null_metric_by_group_len_bucket(
        perm_hits
            .iter()
            .filter_map(|cand| {
                cand.train_score.is_finite().then_some((
                    bucket_from_rule_with_complexity(
                        &cand.rule,
                        cand.train.dosage_maf,
                        null_complexity_bin,
                    ),
                    cand.train_score,
                ))
            })
            .collect::<Vec<_>>()
            .as_slice(),
        null_max_rule_len,
        keep_topk,
        true,
    ) {
        out.push(GarfieldPermutationNullScores {
            bucket,
            search_train_score: score,
            search_test_score: f64::NAN,
            output_train_raw_score: f64::NAN,
            output_test_raw_score: f64::NAN,
            delta_train_score: f64::NAN,
            delta_test_score: f64::NAN,
        });
    }

    for (bucket, score) in collect_topk_rule_null_metric_by_group_len_bucket(
        perm_hits
            .iter()
            .filter_map(|cand| {
                cand.test_score.is_finite().then_some((
                    bucket_from_rule_with_complexity(
                        &cand.rule,
                        cand.test.dosage_maf,
                        null_complexity_bin,
                    ),
                    cand.test_score,
                ))
            })
            .collect::<Vec<_>>()
            .as_slice(),
        null_max_rule_len,
        keep_topk,
        true,
    ) {
        out.push(GarfieldPermutationNullScores {
            bucket,
            search_train_score: f64::NAN,
            search_test_score: score,
            output_train_raw_score: f64::NAN,
            output_test_raw_score: f64::NAN,
            delta_train_score: f64::NAN,
            delta_test_score: f64::NAN,
        });
    }

    for (bucket, score) in collect_topk_rule_null_metric_by_group_len_bucket(
        perm_hits
            .iter()
            .filter_map(|cand| {
                cand.train.raw_score.is_finite().then_some((
                    bucket_from_rule_with_complexity(
                        &cand.rule,
                        cand.train.dosage_maf,
                        null_complexity_bin,
                    ),
                    cand.train.raw_score,
                ))
            })
            .collect::<Vec<_>>()
            .as_slice(),
        null_max_rule_len,
        keep_topk,
        false,
    ) {
        out.push(GarfieldPermutationNullScores {
            bucket,
            search_train_score: f64::NAN,
            search_test_score: f64::NAN,
            output_train_raw_score: score,
            output_test_raw_score: f64::NAN,
            delta_train_score: f64::NAN,
            delta_test_score: f64::NAN,
        });
    }

    for (bucket, score) in collect_topk_rule_null_metric_by_group_len_bucket(
        perm_hits
            .iter()
            .filter_map(|cand| {
                cand.test.raw_score.is_finite().then_some((
                    bucket_from_rule_with_complexity(
                        &cand.rule,
                        cand.test.dosage_maf,
                        null_complexity_bin,
                    ),
                    cand.test.raw_score,
                ))
            })
            .collect::<Vec<_>>()
            .as_slice(),
        null_max_rule_len,
        keep_topk,
        false,
    ) {
        out.push(GarfieldPermutationNullScores {
            bucket,
            search_train_score: f64::NAN,
            search_test_score: f64::NAN,
            output_train_raw_score: f64::NAN,
            output_test_raw_score: score,
            delta_train_score: f64::NAN,
            delta_test_score: f64::NAN,
        });
    }

    // Calibrate the best-parent incremental statistic separately from the
    // absolute raw score.  The parent is the best immediate subrule among
    // the complete beam result, rather than merely the path parent used by
    // the search expansion.
    let mut delta_train_scored = Vec::<(RuleNullBucket, f64)>::new();
    let mut delta_test_scored = Vec::<(RuleNullBucket, f64)>::new();
    for cand in perm_hits.iter().filter(|cand| cand.rule.len() > 1) {
        if cand.train.raw_score.is_finite() {
            let (_, parent) = best_parent_raw_for_candidate(
                cand,
                perm_hits.as_slice(),
                perm_train,
                prepared_bits.train_bits(),
                prepared_bits.train_bits_hi(),
                prepared_bits.row_words_train,
                prepared.selected_global_rows.len(),
                perm_train.len(),
                &beam_params,
            )?;
            if let Some(parent) = parent {
                delta_train_scored.push((
                    bucket_from_rule_with_complexity(
                        &cand.rule,
                        cand.train.dosage_maf,
                        null_complexity_bin,
                    ),
                    cand.train.raw_score - parent,
                ));
            }
        }
        if cand.test.raw_score.is_finite() {
            let (_, parent) = best_parent_raw_for_candidate(
                cand,
                perm_hits.as_slice(),
                perm_test,
                prepared_bits.test_bits(),
                prepared_bits.test_bits_hi(),
                prepared_bits.row_words_test,
                prepared.selected_global_rows.len(),
                perm_test.len(),
                &beam_params,
            )?;
            if let Some(parent) = parent {
                delta_test_scored.push((
                    bucket_from_rule_with_complexity(
                        &cand.rule,
                        cand.test.dosage_maf,
                        null_complexity_bin,
                    ),
                    cand.test.raw_score - parent,
                ));
            }
        }
    }
    for (bucket, score) in collect_topk_rule_null_metric_by_group_len_bucket(
        delta_train_scored.as_slice(),
        null_max_rule_len,
        keep_topk,
        false,
    ) {
        out.push(GarfieldPermutationNullScores {
            bucket,
            search_train_score: f64::NAN,
            search_test_score: f64::NAN,
            output_train_raw_score: f64::NAN,
            output_test_raw_score: f64::NAN,
            delta_train_score: score,
            delta_test_score: f64::NAN,
        });
    }
    for (bucket, score) in collect_topk_rule_null_metric_by_group_len_bucket(
        delta_test_scored.as_slice(),
        null_max_rule_len,
        keep_topk,
        false,
    ) {
        out.push(GarfieldPermutationNullScores {
            bucket,
            search_train_score: f64::NAN,
            search_test_score: f64::NAN,
            output_train_raw_score: f64::NAN,
            output_test_raw_score: f64::NAN,
            delta_train_score: f64::NAN,
            delta_test_score: score,
        });
    }
    Ok(out)
}

#[derive(Clone, Debug, Default)]
pub(super) struct AdaptiveStructurePriorUnitOutcome {
    pub(super) calibrator: RuleStructurePriorCalibrator,
    pub(super) repeats_used: usize,
}

#[inline]
pub(super) fn logsumexp_scaled_scores(scores: &[f64], temperature: f64) -> Option<f64> {
    if scores.is_empty() {
        return None;
    }
    let inv_temp = 1.0 / temperature.max(1e-12);
    let max_scaled = scores
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .map(|score| score * inv_temp)
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_scaled.is_finite() {
        return None;
    }
    let tail = scores
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .map(|score| ((score * inv_temp) - max_scaled).exp())
        .sum::<f64>()
        .max(1e-12);
    Some(max_scaled + tail.ln())
}

#[inline]
pub(super) fn len_prob_rank_signature(probs: &[f64; 6]) -> [usize; 5] {
    let mut order = [1usize, 2usize, 3usize, 4usize, 5usize];
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    order
}

#[inline]
pub(super) fn kl_divergence_len_probs(p: &[f64; 6], q: &[f64; 6]) -> f64 {
    let mut out = 0.0_f64;
    for len in 1..=5usize {
        let pp = p[len].max(1e-12);
        let qq = q[len].max(1e-12);
        out += pp * (pp / qq).ln();
    }
    out
}

#[inline]
pub(super) fn symmetric_kl_len_probs(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    0.5 * (kl_divergence_len_probs(a, b) + kl_divergence_len_probs(b, a))
}

pub(super) fn collect_rule_structure_posterior_for_repeat(
    prepared: &GarfieldUnitPrepared,
    prepared_bits: &GarfieldUnitBitMatrices<'_>,
    y_train: &[f64],
    y_test: &[f64],
    split_applied: bool,
    beam_params: BeamSearchParams,
    structure_prior_cfg: &RuleStructurePriorConfig,
    rep_seed: u64,
) -> Result<Vec<(usize, usize, f64, f64)>, String> {
    if prepared.selected_global_rows.is_empty() || y_train.is_empty() {
        return Ok(Vec::new());
    }
    let beam_params = beam_params_for_prepared(prepared, beam_params);

    let boot_train_idx = bootstrap_sample_indices(y_train.len(), rep_seed);
    let boot_y_train = subset_vec_f64(y_train, boot_train_idx.as_slice())?;
    let boot_train_bits = {
        let row_indices = (0..prepared.selected_global_rows.len()).collect::<Vec<_>>();
        let (bits, _) = packed_rows_subset_from_full_bits(
            prepared_bits.train_bits(),
            prepared_bits.row_words_train,
            row_indices.as_slice(),
            boot_train_idx.as_slice(),
            prepared.selected_global_rows.len(),
            y_train.len(),
        )?;
        bits
    };
    let boot_train_bits_hi = if let Some(train_bits_hi) = prepared_bits.train_bits_hi() {
        let row_indices = (0..prepared.selected_global_rows.len()).collect::<Vec<_>>();
        let (bits, _) = packed_rows_subset_from_full_bits(
            train_bits_hi,
            prepared_bits.row_words_train,
            row_indices.as_slice(),
            boot_train_idx.as_slice(),
            prepared.selected_global_rows.len(),
            y_train.len(),
        )?;
        Some(bits)
    } else {
        None
    };
    let boot_row_words_train = words_for_samples(boot_y_train.len());

    let (boot_y_test, boot_test_bits, boot_test_bits_hi, boot_row_words_test) = if split_applied {
        if y_test.is_empty() {
            return Ok(Vec::new());
        }
        let boot_test_idx =
            bootstrap_sample_indices(y_test.len(), rep_seed ^ 0xD1B5_4A32_D192_ED03);
        let boot_y_test = subset_vec_f64(y_test, boot_test_idx.as_slice())?;
        let row_indices = (0..prepared.selected_global_rows.len()).collect::<Vec<_>>();
        let (bits, _) = packed_rows_subset_from_full_bits(
            prepared_bits.test_bits(),
            prepared_bits.row_words_test,
            row_indices.as_slice(),
            boot_test_idx.as_slice(),
            prepared.selected_global_rows.len(),
            y_test.len(),
        )?;
        let bits_hi = if let Some(test_bits_hi) = prepared_bits.test_bits_hi() {
            let (bits_hi, _) = packed_rows_subset_from_full_bits(
                test_bits_hi,
                prepared_bits.row_words_test,
                row_indices.as_slice(),
                boot_test_idx.as_slice(),
                prepared.selected_global_rows.len(),
                y_test.len(),
            )?;
            Some(bits_hi)
        } else {
            None
        };
        let boot_row_words_test = words_for_samples(boot_y_test.len());
        (boot_y_test, bits, bits_hi, boot_row_words_test)
    } else {
        (
            boot_y_train.clone(),
            boot_train_bits.clone(),
            boot_train_bits_hi.clone(),
            boot_row_words_train,
        )
    };

    // Keep structure-prior bootstrap searches on the same backend as the
    // observed/null scan.  The old code called the historical Beam directly
    // here, which meant `--search-backend pair-triple` was only partially
    // applied and mixed null/observed search spaces.  Build a short-lived bit
    // matrix around the bootstrap buffers so the common dispatch can select
    // PairTriple (or retain the legacy fuzzy path) without copying them.
    let perm_hits = if matches!(
        beam_params.search_backend,
        GarfieldSearchBackend::PairTriple
    ) {
        let boot_matrices = GarfieldUnitBitMatrices {
            train_bits: Cow::Borrowed(boot_train_bits.as_slice()),
            train_bits_hi: boot_train_bits_hi
                .as_ref()
                .map(|bits| Cow::Borrowed(bits.as_slice())),
            row_words_train: boot_row_words_train,
            test_bits: Some(Cow::Borrowed(boot_test_bits.as_slice())),
            test_bits_hi: boot_test_bits_hi
                .as_ref()
                .map(|bits| Cow::Borrowed(bits.as_slice())),
            row_words_test: boot_row_words_test,
            selected_bits_full: None,
            selected_bits_full_hi: None,
            selected_bits_full_alias_train: false,
            selected_bits_full_hi_alias_train: false,
        };
        beam_search_train_test_continuous_dispatch(
            boot_y_train.as_slice(),
            &boot_matrices,
            prepared.selected_global_rows.len(),
            boot_y_test.as_slice(),
            prepared.local_groups.as_slice(),
            beam_params,
            None,
        )?
    } else if let Some(boot_train_hi) = boot_train_bits_hi.as_ref() {
        let boot_test_hi = boot_test_bits_hi
            .as_ref()
            .ok_or_else(|| "internal error: fuzzy bootstrap test bitplane missing".to_string())?;
        beam_search_train_test_continuous_fuzzy(
            boot_y_train.as_slice(),
            boot_train_bits.as_slice(),
            boot_train_hi.as_slice(),
            boot_row_words_train,
            prepared.selected_global_rows.len(),
            boot_y_train.len(),
            boot_y_test.as_slice(),
            boot_test_bits.as_slice(),
            boot_test_hi.as_slice(),
            boot_row_words_test,
            boot_y_test.len(),
            prepared.local_groups.as_slice(),
            beam_params,
        )?
    } else {
        beam_search_train_test_continuous(
            boot_y_train.as_slice(),
            boot_train_bits.as_slice(),
            boot_row_words_train,
            prepared.selected_global_rows.len(),
            boot_y_train.len(),
            boot_y_test.as_slice(),
            boot_test_bits.as_slice(),
            boot_row_words_test,
            boot_y_test.len(),
            prepared.local_groups.as_slice(),
            beam_params,
        )?
    };

    if perm_hits.is_empty() {
        return Ok(Vec::new());
    }

    let mut topk_by_len = vec![Vec::<(usize, f64)>::new(); 6];
    for cand in perm_hits.iter() {
        let rule_len = cand.rule.len().clamp(1, 5);
        let score = if split_applied {
            cand.test_score
        } else {
            cand.train_score
        };
        if !score.is_finite() || score <= 0.0 {
            continue;
        }
        let bucket = &mut topk_by_len[rule_len];
        insert_topk_density_hit(
            bucket,
            cand.rule.not_count(),
            score,
            DEFAULT_RULE_STRUCTURE_DENSITY_TOPK,
        );
    }
    let pooled_scores = topk_by_len
        .iter()
        .skip(1)
        .flat_map(|hits| hits.iter().map(|(_, score)| *score))
        .collect::<Vec<_>>();
    if pooled_scores.is_empty() {
        return Ok(Vec::new());
    }

    let score_scale = pooled_scores.iter().sum::<f64>() / (pooled_scores.len() as f64);
    let temperature = (score_scale * 1.1).clamp(0.02, 0.08);
    let observed = (1..=5usize)
        .filter_map(|rule_len| {
            let hits = &topk_by_len[rule_len];
            if hits.is_empty() {
                return None;
            }
            let best_n_not = hits[0].0;
            let best_score = hits[0].1;
            let len_scores = hits.iter().map(|(_, score)| *score).collect::<Vec<_>>();
            let log_density = logsumexp_scaled_scores(len_scores.as_slice(), temperature)?;
            Some((rule_len, best_n_not, best_score, log_density))
        })
        .collect::<Vec<_>>();
    if observed.is_empty() {
        return Ok(Vec::new());
    }
    let max_logw = observed
        .iter()
        .map(|(rule_len, _, _, log_density)| {
            structure_prior_cfg.len_alpha(*rule_len).ln() + *log_density
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let mut weights = Vec::<f64>::with_capacity(observed.len());
    for (rule_len, _, _, log_density) in observed.iter() {
        let logw = structure_prior_cfg.len_alpha(*rule_len).ln() + *log_density;
        weights.push((logw - max_logw).exp());
    }
    let total = weights.iter().sum::<f64>().max(1e-12);
    let mut out = Vec::<(usize, usize, f64, f64)>::with_capacity(observed.len());
    for ((rule_len, n_not, score, _log_density), weight_raw) in
        observed.into_iter().zip(weights.into_iter())
    {
        out.push((rule_len, n_not, score, weight_raw / total));
    }
    Ok(out)
}

pub(super) fn collect_rule_structure_posterior_for_unit_adaptive(
    prepared: &GarfieldUnitPrepared,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    y_train: &[f64],
    y_test: &[f64],
    split_applied: bool,
    beam_params: BeamSearchParams,
    structure_prior_cfg: &RuleStructurePriorConfig,
    unit_seed: u64,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
) -> Result<AdaptiveStructurePriorUnitOutcome, String> {
    if prepared.selected_global_rows.is_empty() || y_train.is_empty() {
        return Ok(AdaptiveStructurePriorUnitOutcome::default());
    }
    let prepared_bits = materialize_prepared_bit_matrices(
        prepared,
        logic_bits,
        train_idx_local,
        test_idx_local,
        false,
    )?;
    if let Some(tracker) = mem_tracker {
        tracker.sample_now();
    }
    let mut local = RuleStructurePriorCalibrator::default();
    let mut informative_repeats = 0usize;
    let mut prev_probs = None::<[f64; 6]>;
    let mut stable_streak = 0usize;
    let mut repeats_used = 0usize;

    for rep in 0..DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MAX_REPEATS {
        if (rep & 7) == 0 {
            check_ctrlc()?;
        }
        let rep_seed =
            unit_seed ^ ((rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) ^ 0xA24B_AED4_963E_E407;
        let repeat_out = collect_rule_structure_posterior_for_repeat(
            prepared,
            &prepared_bits,
            y_train,
            y_test,
            split_applied,
            beam_params.clone(),
            structure_prior_cfg,
            rep_seed,
        )?;
        if let Some(tracker) = mem_tracker {
            tracker.sample_now();
        }
        if repeat_out.is_empty() {
            repeats_used += 1;
            continue;
        }
        for (rule_len, n_not, score, weight) in repeat_out.into_iter() {
            local.insert(rule_len, n_not, score, weight);
        }
        repeats_used += 1;
        informative_repeats += 1;
        if informative_repeats < DEFAULT_RULE_STRUCTURE_BOOTSTRAP_MIN_REPEATS {
            continue;
        }
        let Some(cur_probs) = local.observed_len_probs_preview() else {
            continue;
        };
        if let Some(prev) = prev_probs {
            let same_rank = len_prob_rank_signature(&prev) == len_prob_rank_signature(&cur_probs);
            let kl = symmetric_kl_len_probs(&prev, &cur_probs);
            if same_rank && kl <= DEFAULT_RULE_STRUCTURE_BOOTSTRAP_KL_THRESHOLD {
                stable_streak += 1;
            } else {
                stable_streak = 0;
            }
            if stable_streak >= DEFAULT_RULE_STRUCTURE_BOOTSTRAP_STABLE_REPEATS {
                break;
            }
        }
        prev_probs = Some(cur_probs);
    }

    Ok(AdaptiveStructurePriorUnitOutcome {
        calibrator: local,
        repeats_used,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_logic_unit_prepared_continuous(
    ui: usize,
    unit: &GarfieldLogicUnit,
    prepared: &GarfieldUnitPrepared,
    prepared_bits: &GarfieldUnitBitMatrices<'_>,
    logic_bits: &GarfieldLogicBits,
    test_idx_local: &[usize],
    y_train: &[f64],
    y_test: &[f64],
    beam_params: BeamSearchParams,
    output_null_penalties: Option<Arc<RuleNullPenaltyLookup>>,
    top_rules_per_unit: usize,
    raw_design: bool,
    unit_kind_lc: &str,
    debug_probe: Option<&GarfieldBeamDebugProbe>,
    diagnostics: bool,
) -> Result<GarfieldUnitEvaluationOutput, String> {
    let beam_params_search = beam_params_for_prepared(prepared, beam_params.clone());
    if env_truthy("JX_GARFIELD_LAYER_DEBUG") {
        eprintln!(
            "[GARFIELD-LAYER-DEBUG] unit={} stage=beam_start selected_rows={} beam_width={} max_pick={}",
            unit.label,
            prepared.selected_global_rows.len(),
            beam_params_search.beam_width,
            beam_params_search.max_pick,
        );
    }
    let literal_scores = if prepared_bits.test_bits.is_none()
        && prepared_bits.test_bits_hi.is_none()
        && y_train == y_test
    {
        prepared.train_literal_scores.as_deref()
    } else {
        None
    };
    let beam_hits = if prepared_bits.has_fuzzy_bin() {
        beam_search_train_test_continuous_dispatch(
            y_train,
            prepared_bits,
            prepared.selected_global_rows.len(),
            y_test,
            prepared.local_groups.as_slice(),
            beam_params_search.clone(),
            literal_scores,
        )?
    } else {
        beam_search_train_test_continuous_dispatch(
            y_train,
            prepared_bits,
            prepared.selected_global_rows.len(),
            y_test,
            prepared.local_groups.as_slice(),
            beam_params_search.clone(),
            literal_scores,
        )?
    };
    if env_truthy("JX_GARFIELD_LAYER_DEBUG") {
        eprintln!(
            "[GARFIELD-LAYER-DEBUG] unit={} stage=beam_end hits={}",
            unit.label,
            beam_hits.len(),
        );
    }
    let local_sites =
        local_sites_from_selected_rows(prepared.selected_global_rows.as_slice(), logic_bits)?;
    let frontier_trace = take_garfield_frontier_trace();
    let frontier_diagnostics = frontier_trace
        .into_iter()
        .map(|trace: BeamFrontierTraceRecord| {
            let child_rule = rule_expr_with_polarity(
                &trace.child,
                local_sites.as_slice(),
                GarfieldRuleDisplayPolarity::Original,
            )?;
            let parent_rule = if let Some(parent) = trace.parent.as_ref() {
                rule_expr_with_polarity(
                    parent,
                    local_sites.as_slice(),
                    GarfieldRuleDisplayPolarity::Original,
                )?
            } else {
                String::new()
            };
            let best_feasible_parent = if let Some(parent) = trace.best_feasible_parent.as_ref() {
                rule_expr_with_polarity(
                    parent,
                    local_sites.as_slice(),
                    GarfieldRuleDisplayPolarity::Original,
                )?
            } else {
                String::new()
            };
            let global_rows =
                rule_selected_global_rows(&trace.child, prepared.selected_global_rows.as_slice())?;
            Ok::<GarfieldFrontierDiagnosticRecord, String>(GarfieldFrontierDiagnosticRecord {
                unit_name: unit.label.clone(),
                unit_index: ui + 1,
                region_size: unit.indices.len(),
                depth: trace.layer,
                frontier_rank: trace.frontier_rank,
                parent_rule,
                child_rule,
                raw_score: trace.raw_score,
                search_score: trace.search_score,
                beam_cutoff_score: trace.beam_cutoff_score,
                frontier_rank_before: trace.frontier_rank_before,
                frontier_rank_after: trace.frontier_rank_after,
                retained: trace.retained,
                support: trace.support,
                support_current_parent: trace.support_current_parent,
                support_any_parent: trace.support_any_parent,
                n_feasible_parents: trace.n_feasible_parents,
                rescued_by_alt_parent: trace.rescued_by_alt_parent,
                best_feasible_parent,
                best_parent_delta: trace.best_parent_delta,
                global_rows,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    if beam_hits.is_empty() {
        if !diagnostics {
            return Ok(GarfieldUnitEvaluationOutput::default());
        }
        let mut diagnostics_out = Vec::<GarfieldRecallDiagnosticRecord>::new();
        append_ld_compression_diagnostics(
            &mut diagnostics_out,
            unit,
            ui + 1,
            logic_bits,
            prepared.ld_compression.as_slice(),
        );
        if let Some(stage1_rows) = prepared.stage1_candidate_global_rows.as_ref() {
            diagnostics_out.push(GarfieldRecallDiagnosticRecord {
                unit_name: unit.label.clone(),
                unit_index: ui + 1,
                region_size: unit.indices.len(),
                stage: "stage1_candidate",
                rank: 0,
                n_rows: stage1_rows.len(),
                global_rows: stage1_rows.clone(),
                snp_name: diagnostic_site_names(logic_bits, stage1_rows.as_slice()),
                expr: String::new(),
                raw_score: None,
                output_score: None,
                best_parent_raw: None,
                best_parent_delta: None,
                formal_eligible: false,
                reported: false,
            });
        }
        diagnostics_out.push(GarfieldRecallDiagnosticRecord {
            unit_name: unit.label.clone(),
            unit_index: ui + 1,
            region_size: unit.indices.len(),
            stage: "selected",
            rank: 0,
            n_rows: prepared.selected_global_rows.len(),
            global_rows: prepared.selected_global_rows.clone(),
            snp_name: diagnostic_site_names(logic_bits, prepared.selected_global_rows.as_slice()),
            expr: String::new(),
            raw_score: None,
            output_score: None,
            best_parent_raw: None,
            best_parent_delta: None,
            formal_eligible: false,
            reported: false,
        });
        return Ok(GarfieldUnitEvaluationOutput {
            records: Vec::new(),
            diagnostics: diagnostics_out,
            frontier_diagnostics,
        });
    }
    let selected_bits_full = prepared_bits.selected_bits_full().ok_or_else(|| {
        "internal error: selected_bits_full missing for GARFIELD evaluation".to_string()
    })?;
    let selected_bits_full_hi = prepared_bits.selected_bits_full_hi();
    let null_complexity_bin = beam_params_search.null_complexity_bin;

    let mut ranked_hits = beam_hits
        .iter()
        .enumerate()
        .map(|(idx, cand)| {
            (
                idx,
                final_output_score_for_candidate(
                    cand,
                    output_null_penalties.as_deref(),
                    false,
                    null_complexity_bin,
                ),
            )
        })
        .collect::<Vec<_>>();
    ranked_hits.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| cmp_candidate(&beam_hits[a.0], &beam_hits[b.0]))
    });
    let gated_ranked_hits = if raw_design {
        ranked_hits.clone()
    } else {
        gated_ranked_hits_with_collapse(
            beam_hits.as_slice(),
            ranked_hits.as_slice(),
            output_null_penalties.as_deref(),
            null_complexity_bin,
            y_test,
            prepared_bits.test_bits(),
            prepared_bits.test_bits_hi(),
            prepared_bits.row_words_test,
            prepared.selected_global_rows.len(),
            y_test.len(),
            &beam_params_search,
        )?
    };
    maybe_write_beam_debug_probe_tsv(
        debug_probe,
        unit,
        prepared,
        beam_hits.as_slice(),
        local_sites.as_slice(),
        selected_bits_full,
        selected_bits_full_hi,
        logic_bits,
        output_null_penalties.as_deref(),
        null_complexity_bin,
    )?;
    let report_hits = select_reportable_ranked_hits(
        beam_hits.as_slice(),
        gated_ranked_hits.as_slice(),
        top_rules_per_unit,
        raw_design,
    );
    let reported_indices = report_hits
        .iter()
        .map(|(idx, _)| *idx)
        .collect::<std::collections::HashSet<_>>();
    let reported_rule_keys = report_hits
        .iter()
        .filter_map(|(idx, _)| beam_hits.get(*idx))
        .map(|cand| cand.rule.lexical_key())
        .collect::<HashSet<_>>();
    let mut diagnostics_out = Vec::<GarfieldRecallDiagnosticRecord>::new();
    if diagnostics {
        append_ld_compression_diagnostics(
            &mut diagnostics_out,
            unit,
            ui + 1,
            logic_bits,
            prepared.ld_compression.as_slice(),
        );
        if let Some(stage1_rows) = prepared.stage1_candidate_global_rows.as_ref() {
            diagnostics_out.push(GarfieldRecallDiagnosticRecord {
                unit_name: unit.label.clone(),
                unit_index: ui + 1,
                region_size: unit.indices.len(),
                stage: "stage1_candidate",
                rank: 0,
                n_rows: stage1_rows.len(),
                global_rows: stage1_rows.clone(),
                snp_name: diagnostic_site_names(logic_bits, stage1_rows.as_slice()),
                expr: String::new(),
                raw_score: None,
                output_score: None,
                best_parent_raw: None,
                best_parent_delta: None,
                formal_eligible: false,
                reported: false,
            });
        }
        diagnostics_out.push(GarfieldRecallDiagnosticRecord {
            unit_name: unit.label.clone(),
            unit_index: ui + 1,
            region_size: unit.indices.len(),
            stage: "selected",
            rank: 0,
            n_rows: prepared.selected_global_rows.len(),
            global_rows: prepared.selected_global_rows.clone(),
            snp_name: diagnostic_site_names(logic_bits, prepared.selected_global_rows.as_slice()),
            expr: String::new(),
            raw_score: None,
            output_score: None,
            best_parent_raw: None,
            best_parent_delta: None,
            formal_eligible: false,
            reported: false,
        });
        for (beam_rank, (cand_idx, output_score)) in ranked_hits.iter().enumerate() {
            let cand = &beam_hits[*cand_idx];
            let rendered = render_candidate_rule_for_output(
                cand,
                local_sites.as_slice(),
                selected_bits_full,
                selected_bits_full_hi,
                logic_bits,
                prepared.selected_global_rows.len(),
            )?;
            let global_rows =
                rule_selected_global_rows(&cand.rule, prepared.selected_global_rows.as_slice())?;
            let (best_parent_raw, best_parent_delta) = if cand.rule.len() > 1 {
                let (_, parent_raw) = best_parent_raw_for_candidate(
                    cand,
                    beam_hits.as_slice(),
                    y_test,
                    prepared_bits.test_bits(),
                    prepared_bits.test_bits_hi(),
                    prepared_bits.row_words_test,
                    prepared.selected_global_rows.len(),
                    y_test.len(),
                    &beam_params_search,
                )?;
                (
                    parent_raw,
                    parent_raw.map(|parent| cand.test.raw_score - parent),
                )
            } else {
                (None, None)
            };
            diagnostics_out.push(GarfieldRecallDiagnosticRecord {
                unit_name: unit.label.clone(),
                unit_index: ui + 1,
                region_size: unit.indices.len(),
                stage: "beam",
                rank: beam_rank + 1,
                n_rows: global_rows.len(),
                global_rows,
                snp_name: rendered.snp_name,
                expr: rendered.expr,
                raw_score: Some(cand.test.raw_score),
                output_score: Some(*output_score),
                best_parent_raw,
                best_parent_delta,
                formal_eligible: !raw_design
                    && gated_ranked_hits
                        .iter()
                        .any(|(eligible_idx, _)| *eligible_idx == *cand_idx),
                reported: reported_indices.contains(cand_idx)
                    || reported_rule_keys.contains(&cand.rule.lexical_key()),
            });
        }
    }
    let mut out = Vec::<GarfieldLogicRuleRecord>::with_capacity(report_hits.len().max(1));
    for (cand_idx, output_score) in report_hits.iter() {
        let cand = &beam_hits[*cand_idx];
        let rendered = render_candidate_rule_for_output(
            cand,
            local_sites.as_slice(),
            selected_bits_full,
            selected_bits_full_hi,
            logic_bits,
            prepared.selected_global_rows.len(),
        )?;
        let delta_score = build_rule_delta_score_annotation(
            &cand.rule,
            y_test,
            prepared_bits.test_bits(),
            prepared_bits.test_bits_hi(),
            prepared_bits.row_words_test,
            prepared.selected_global_rows.len(),
            test_idx_local.len(),
            &beam_params_search,
            rendered.polarity,
        )?;
        let selected_row_indices =
            rule_selected_global_rows(&cand.rule, prepared.selected_global_rows.as_slice())?;
        out.push(GarfieldLogicRuleRecord {
            unit_name: unit.label.clone(),
            unit_kind: unit_kind_lc.to_string(),
            unit_index: ui + 1,
            region_size: unit.indices.len(),
            ml_feature_count: prepared.selected_global_rows.len(),
            ml_rank: rule_ml_rank_name_with_polarity(&cand.rule, rendered.polarity),
            selected_row_indices,
            display_ops: rendered.display_ops,
            display_negated: rendered.display_negated,
            snp_name: rendered.snp_name,
            expr: rendered.expr,
            chrom_field: rendered.chrom_field,
            bim_snp_name: rendered.bim_snp_name,
            bim_allele0: rendered.bim_allele0,
            bim_allele1: rendered.bim_allele1,
            allele1_maf: cand.test.dosage_maf,
            pos: rendered.pos,
            score: *output_score,
            delta_score,
            perm_pvalue: None,
            perm_fdr: None,
            support_bits: None,
        });
    }
    Ok(GarfieldUnitEvaluationOutput {
        records: out,
        diagnostics: diagnostics_out,
        frontier_diagnostics,
    })
}

#[allow(dead_code)]
pub(super) fn precompute_literal_singleton_scores_for_unit(
    prepared: &GarfieldUnitPrepared,
    prepared_bits: &GarfieldUnitBitMatrices<'_>,
    y_train: &[f64],
    y_test: &[f64],
) -> Result<Vec<LiteralSingletonScore>, String> {
    // Legacy packed-0/1 singleton precompute helper retained only so the
    // disabled non-fuzzy route can be resurrected or inspected if needed.
    let requests = [LiteralScoreBatchRequest {
        bits_train: prepared_bits.train_bits(),
        row_words_train: prepared_bits.row_words_train,
        bits_test: prepared_bits.test_bits(),
        row_words_test: prepared_bits.row_words_test,
        n_rows: prepared.selected_global_rows.len(),
    }];
    let mut out = precompute_literal_singleton_scores_batched(
        y_train,
        y_train.len(),
        y_test,
        y_test.len(),
        requests.as_slice(),
    )?;
    if out.len() != 1 {
        return Err(format!(
            "GARFIELD unit literal score batch size mismatch: got {}, expected 1",
            out.len()
        ));
    }
    Ok(out.swap_remove(0))
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(super) fn evaluate_logic_unit_continuous(
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
    beam_params: BeamSearchParams,
    output_null_penalties: Option<Arc<RuleNullPenaltyLookup>>,
    top_rules_per_unit: usize,
    raw_design: bool,
    unit_kind_lc: &str,
) -> Result<Vec<GarfieldLogicRuleRecord>, String> {
    let Some(prepared) = prepare_logic_unit_continuous(
        unit,
        unit_kind_lc,
        response,
        engine,
        importance,
        perm_cfg,
        ml_top_k,
        ml_top_frac,
        tree_cfg,
        logic_bits,
        train_idx_local,
        test_idx_local,
        y_train,
        None,
        beam_params.clone(),
    )?
    else {
        return Ok(Vec::new());
    };
    let prepared_bits = materialize_prepared_bit_matrices(
        &prepared,
        logic_bits,
        train_idx_local,
        test_idx_local,
        true,
    )?;
    Ok(evaluate_logic_unit_prepared_continuous(
        ui,
        unit,
        &prepared,
        &prepared_bits,
        logic_bits,
        test_idx_local,
        y_train,
        y_test,
        beam_params,
        output_null_penalties,
        top_rules_per_unit,
        raw_design,
        unit_kind_lc,
        None,
        false,
    )?
    .records)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_scan_unit_continuous(
    ui: usize,
    units: &[GarfieldLogicUnit],
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
    stage1_y_lookup: Option<&PackedYSumLookup>,
    y_test: &[f64],
    beam_params: BeamSearchParams,
    output_null_penalties: Option<Arc<RuleNullPenaltyLookup>>,
    top_rules_per_unit: usize,
    raw_design: bool,
    unit_kind_lc: &str,
    progress_callback: Option<&Py<PyAny>>,
    scan_progress_done: &AtomicUsize,
    scan_notify_step: usize,
    scanned_units: usize,
    skipped_units: &Arc<Mutex<Vec<GarfieldSkippedUnitInfo>>>,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
    debug_probe: Option<&GarfieldBeamDebugProbe>,
    support_cache: Option<&GarfieldLdSupportCache>,
    diagnostics: bool,
) -> Result<GarfieldUnitEvaluationOutput, String> {
    check_ctrlc()?;
    let unit = &units[ui];
    let mut out = GarfieldUnitEvaluationOutput::default();
    let wg_debug = beam_params.whole_genome_dev_mode;
    let unit_t0 = wg_debug.then(Instant::now);
    if let Some(t0) = unit_t0 {
        garfield_whole_genome_unit_breakpoint(
            unit.label.as_str(),
            "prepare:start",
            unit.indices.len(),
            t0.elapsed().as_secs_f64(),
        )?;
    }
    if let Some(prepared) = prepare_logic_unit_continuous_with_ld_cache(
        unit,
        unit_kind_lc,
        response,
        engine,
        importance,
        perm_cfg,
        ml_top_k,
        ml_top_frac,
        tree_cfg,
        logic_bits,
        train_idx_local,
        test_idx_local,
        y_train,
        stage1_y_lookup,
        beam_params.clone(),
        support_cache,
        diagnostics,
    )? {
        if let Some(t0) = unit_t0 {
            garfield_whole_genome_unit_breakpoint(
                unit.label.as_str(),
                "prepare:end",
                prepared.selected_global_rows.len(),
                t0.elapsed().as_secs_f64(),
            )?;
        }
        let prepared_bits = materialize_prepared_bit_matrices(
            &prepared,
            logic_bits,
            train_idx_local,
            test_idx_local,
            true,
        )?;
        if let Some(t0) = unit_t0 {
            garfield_whole_genome_unit_breakpoint(
                unit.label.as_str(),
                "materialize:end",
                prepared.selected_global_rows.len(),
                t0.elapsed().as_secs_f64(),
            )?;
        }
        if let Some(tracker) = mem_tracker {
            tracker.sample_now();
        }
        if let Some(t0) = unit_t0 {
            garfield_whole_genome_unit_breakpoint(
                unit.label.as_str(),
                "literal_scores:start",
                prepared.selected_global_rows.len(),
                t0.elapsed().as_secs_f64(),
            )?;
        }
        if let Some(t0) = unit_t0 {
            garfield_whole_genome_unit_breakpoint(
                unit.label.as_str(),
                "literal_scores:end",
                prepared.selected_global_rows.len(),
                t0.elapsed().as_secs_f64(),
            )?;
            garfield_whole_genome_unit_breakpoint(
                unit.label.as_str(),
                "beam:dispatch",
                prepared.selected_global_rows.len(),
                t0.elapsed().as_secs_f64(),
            )?;
        }
        out = match evaluate_logic_unit_prepared_continuous(
            ui,
            unit,
            &prepared,
            &prepared_bits,
            logic_bits,
            test_idx_local,
            y_train,
            y_test,
            beam_params.clone(),
            output_null_penalties.clone(),
            top_rules_per_unit,
            raw_design,
            unit_kind_lc,
            debug_probe,
            diagnostics,
        ) {
            Ok(v) => v,
            Err(err) if is_no_valid_initial_literals_error(&err) => {
                skipped_units
                    .lock()
                    .map_err(|_| "GARFIELD skipped-units mutex poisoned".to_string())?
                    .push(make_skipped_unit_info("scan", &unit.label, &err));
                GarfieldUnitEvaluationOutput::default()
            }
            Err(err) => return Err(err),
        };
        if let Some(t0) = unit_t0 {
            garfield_whole_genome_unit_breakpoint(
                unit.label.as_str(),
                "beam:return",
                prepared.selected_global_rows.len(),
                t0.elapsed().as_secs_f64(),
            )?;
        }
        if let Some(tracker) = mem_tracker {
            tracker.sample_now();
        }
    }
    let done = scan_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
    if done % scan_notify_step == 0 || done == scanned_units {
        garfield_stage_progress_notify(progress_callback, "scan", done, scanned_units.max(1), None)
            .map_err(|e| e.to_string())?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_rule_permutation_task_chunk<'bits>(
    slot_indices: &[usize],
    rep_start: usize,
    rep_end: usize,
    null_prepared: &[(GarfieldPermutationNullSource, GarfieldUnitPrepared)],
    prepared_bits_cache: Option<&[GarfieldUnitBitMatrices<'bits>]>,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    y_train: &[f64],
    y_test: &[f64],
    null_y_replicates: Option<&[Vec<f64>]>,
    split_applied: bool,
    perm_beam_params: BeamSearchParams,
    keep_topk: usize,
    seed: u64,
    progress_callback: Option<&Py<PyAny>>,
    null_progress_done: &AtomicUsize,
    null_notify_step: usize,
    permutation_task_total: usize,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
    null_reselection: Option<GarfieldNullReselectionConfig>,
) -> Result<Vec<(usize, Vec<GarfieldPermutationNullScores>)>, String> {
    let mut out = Vec::<(usize, Vec<GarfieldPermutationNullScores>)>::with_capacity(
        slot_indices.len() * (rep_end - rep_start),
    );
    for &slot in slot_indices.iter() {
        check_ctrlc()?;
        let (source, prepared) = &null_prepared[slot];
        let prepared_bits_owned = if prepared_bits_cache.is_none() && null_reselection.is_none() {
            Some(materialize_prepared_bit_matrices(
                prepared,
                logic_bits,
                train_idx_local,
                test_idx_local,
                false,
            )?)
        } else {
            None
        };
        if prepared_bits_cache.is_none() && null_reselection.is_none() {
            if let Some(tracker) = mem_tracker {
                tracker.sample_now();
            }
        }
        let prepared_bits = if let Some(cache) = prepared_bits_cache {
            Some(cache.get(slot).ok_or_else(|| {
                format!(
                    "GARFIELD permutation bit cache missing slot {} (cache size={})",
                    slot,
                    cache.len()
                )
            })?)
        } else {
            prepared_bits_owned.as_ref()
        };
        for rep in rep_start..rep_end {
            if ((rep - rep_start) & 7) == 0 {
                check_ctrlc()?;
            }
            let rep_seed = seed
                ^ (source.seed_id().wrapping_mul(0x94D0_49BB_1331_11EB))
                ^ ((rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                ^ ((slot as u64).wrapping_mul(0xA24B_AED4_963E_E407));
            let dynamic_requested = null_reselection.is_some()
                && matches!(source, GarfieldPermutationNullSource::Chunk(_));
            let dynamic_prepared = if let (Some(cfg), GarfieldPermutationNullSource::Chunk(chunk)) =
                (null_reselection, *source)
            {
                let (perm_train, _) = permutation_y_for_repeat(
                    y_train,
                    y_test,
                    split_applied,
                    rep_seed,
                    rep,
                    null_y_replicates,
                )?;
                prepare_logic_chunk_continuous(
                    chunk,
                    cfg.row_mul,
                    cfg.response,
                    cfg.engine,
                    cfg.importance,
                    PermutationConfig {
                        seed: rep_seed,
                        ..cfg.perm_cfg
                    },
                    cfg.ml_top_k,
                    cfg.ml_top_frac,
                    cfg.tree_cfg,
                    logic_bits,
                    train_idx_local,
                    test_idx_local,
                    perm_train.as_slice(),
                    perm_beam_params.clone(),
                )?
            } else {
                None
            };
            let dynamic_bits = if let Some(dynamic_prepared) = dynamic_prepared.as_ref() {
                Some(materialize_prepared_bit_matrices(
                    dynamic_prepared,
                    logic_bits,
                    train_idx_local,
                    test_idx_local,
                    false,
                )?)
            } else {
                None
            };
            let vals_result = if dynamic_requested {
                if let Some(dynamic_prepared) = dynamic_prepared.as_ref() {
                    let prepared_bits_use = dynamic_bits
                        .as_ref()
                        .expect("dynamic null preparation must materialize bit matrices");
                    let (perm_train, perm_test) = permutation_y_for_repeat(
                        y_train,
                        y_test,
                        split_applied,
                        rep_seed,
                        rep,
                        null_y_replicates,
                    )?;
                    collect_rule_permutation_nulls_for_repeat_with_y(
                        dynamic_prepared,
                        prepared_bits_use,
                        perm_train.as_slice(),
                        perm_test.as_slice(),
                        split_applied,
                        perm_beam_params.clone(),
                        keep_topk,
                    )
                } else {
                    // A permutation can legitimately have no valid ML rows
                    // after MAF/support filtering; contribute an empty null
                    // candidate set rather than phenotype-leaky fixed rows.
                    Ok(Vec::new())
                }
            } else {
                let prepared_bits_use = prepared_bits.ok_or_else(|| {
                    "GARFIELD permutation prepared bits missing for fixed null task".to_string()
                })?;
                collect_rule_permutation_nulls_for_repeat(
                    prepared,
                    prepared_bits_use,
                    y_train,
                    y_test,
                    split_applied,
                    perm_beam_params.clone(),
                    keep_topk,
                    rep_seed,
                    rep,
                    null_y_replicates,
                )
            };
            let vals = match vals_result {
                Ok(v) => v,
                Err(err) if is_no_valid_initial_literals_error(&err) => Vec::new(),
                Err(err) => return Err(err),
            };
            if let Some(tracker) = mem_tracker {
                tracker.sample_now();
            }
            let done = null_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
            if done % null_notify_step == 0 || done == permutation_task_total {
                garfield_stage_progress_notify(
                    progress_callback,
                    "null_penalty",
                    done,
                    permutation_task_total.max(1),
                    None,
                )
                .map_err(|e| e.to_string())?;
            }
            out.push((rep, vals));
        }
    }
    Ok(out)
}

/// Like `process_rule_permutation_task_chunk` but takes a flat `&[(slot, rep)]`
/// list so callers can balance work across workers regardless of the
/// slot/repeat ratio.
#[allow(dead_code)]
pub(super) fn process_rule_permutation_task_chunk_flat<'bits>(
    task_chunk: &[(usize, usize)],
    null_prepared: &[(GarfieldPermutationNullSource, GarfieldUnitPrepared)],
    prepared_bits_cache: Option<&[GarfieldUnitBitMatrices<'bits>]>,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    y_train: &[f64],
    y_test: &[f64],
    null_y_replicates: Option<&[Vec<f64>]>,
    split_applied: bool,
    perm_beam_params: BeamSearchParams,
    keep_topk: usize,
    seed: u64,
    progress_callback: Option<&Py<PyAny>>,
    null_progress_done: &AtomicUsize,
    null_notify_step: usize,
    permutation_task_total: usize,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
    null_reselection: Option<GarfieldNullReselectionConfig>,
) -> Result<Vec<(usize, Vec<GarfieldPermutationNullScores>)>, String> {
    let mut out =
        Vec::<(usize, Vec<GarfieldPermutationNullScores>)>::with_capacity(task_chunk.len());
    for &(slot, rep) in task_chunk.iter() {
        check_ctrlc()?;
        let (source, prepared) = &null_prepared[slot];
        let prepared_bits_owned = if prepared_bits_cache.is_none() && null_reselection.is_none() {
            Some(materialize_prepared_bit_matrices(
                prepared,
                logic_bits,
                train_idx_local,
                test_idx_local,
                false,
            )?)
        } else {
            None
        };
        if prepared_bits_cache.is_none() && null_reselection.is_none() {
            if let Some(tracker) = mem_tracker {
                tracker.sample_now();
            }
        }
        let prepared_bits = if let Some(cache) = prepared_bits_cache {
            Some(cache.get(slot).ok_or_else(|| {
                format!(
                    "GARFIELD permutation bit cache missing slot {} (cache size={})",
                    slot,
                    cache.len()
                )
            })?)
        } else {
            prepared_bits_owned.as_ref()
        };
        let rep_seed = seed
            ^ (source.seed_id().wrapping_mul(0x94D0_49BB_1331_11EB))
            ^ ((rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ ((slot as u64).wrapping_mul(0xA24B_AED4_963E_E407));
        let dynamic_requested =
            null_reselection.is_some() && matches!(source, GarfieldPermutationNullSource::Chunk(_));
        let dynamic_prepared = if let (Some(cfg), GarfieldPermutationNullSource::Chunk(chunk)) =
            (null_reselection, *source)
        {
            let (perm_train, _) = permutation_y_for_repeat(
                y_train,
                y_test,
                split_applied,
                rep_seed,
                rep,
                null_y_replicates,
            )?;
            prepare_logic_chunk_continuous(
                chunk,
                cfg.row_mul,
                cfg.response,
                cfg.engine,
                cfg.importance,
                PermutationConfig {
                    seed: rep_seed,
                    ..cfg.perm_cfg
                },
                cfg.ml_top_k,
                cfg.ml_top_frac,
                cfg.tree_cfg,
                logic_bits,
                train_idx_local,
                test_idx_local,
                perm_train.as_slice(),
                perm_beam_params.clone(),
            )?
        } else {
            None
        };
        let dynamic_bits = if let Some(dynamic_prepared) = dynamic_prepared.as_ref() {
            Some(materialize_prepared_bit_matrices(
                dynamic_prepared,
                logic_bits,
                train_idx_local,
                test_idx_local,
                false,
            )?)
        } else {
            None
        };
        let vals_result = if dynamic_requested {
            if let Some(dynamic_prepared) = dynamic_prepared.as_ref() {
                let prepared_bits_use = dynamic_bits
                    .as_ref()
                    .expect("dynamic null preparation must materialize bit matrices");
                let (perm_train, perm_test) = permutation_y_for_repeat(
                    y_train,
                    y_test,
                    split_applied,
                    rep_seed,
                    rep,
                    null_y_replicates,
                )?;
                collect_rule_permutation_nulls_for_repeat_with_y(
                    dynamic_prepared,
                    prepared_bits_use,
                    perm_train.as_slice(),
                    perm_test.as_slice(),
                    split_applied,
                    perm_beam_params.clone(),
                    keep_topk,
                )
            } else {
                Ok(Vec::new())
            }
        } else {
            let prepared_bits_use = prepared_bits.ok_or_else(|| {
                "GARFIELD permutation prepared bits missing for fixed null task".to_string()
            })?;
            collect_rule_permutation_nulls_for_repeat(
                prepared,
                prepared_bits_use,
                y_train,
                y_test,
                split_applied,
                perm_beam_params.clone(),
                keep_topk,
                rep_seed,
                rep,
                null_y_replicates,
            )
        };
        let vals = match vals_result {
            Ok(v) => v,
            Err(err) if is_no_valid_initial_literals_error(&err) => Vec::new(),
            Err(err) => return Err(err),
        };
        if let Some(tracker) = mem_tracker {
            tracker.sample_now();
        }
        let done = null_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
        if done % null_notify_step == 0 || done == permutation_task_total {
            garfield_stage_progress_notify(
                progress_callback,
                "null_penalty",
                done,
                permutation_task_total.max(1),
                None,
            )
            .map_err(|e| e.to_string())?;
        }
        out.push((rep, vals));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_structure_prior_unit_chunk(
    slot_start: usize,
    chunk: &[(usize, GarfieldUnitPrepared)],
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    y_train: &[f64],
    y_test: &[f64],
    split_applied: bool,
    posterior_beam_params: BeamSearchParams,
    structure_prior_cfg: &RuleStructurePriorConfig,
    seed: u64,
    structure_scores: &Arc<Mutex<RuleStructurePriorCalibrator>>,
    structure_repeats_used: &AtomicUsize,
    structure_progress_done: &AtomicUsize,
    structure_notify_step: usize,
    structure_task_total: usize,
    structure_prior_display_len: usize,
    progress_callback: Option<&Py<PyAny>>,
    mem_tracker: Option<&GarfieldStageMemoryTracker>,
) -> Result<(), String> {
    for (local_idx, (ui, prepared)) in chunk.iter().enumerate() {
        check_ctrlc()?;
        let slot = slot_start + local_idx;
        let unit_seed = seed
            ^ ((*ui as u64).wrapping_mul(0xD134_2543_DE82_EF95))
            ^ ((slot as u64).wrapping_mul(0xA24B_AED4_963E_E407));
        let out = collect_rule_structure_posterior_for_unit_adaptive(
            prepared,
            logic_bits,
            train_idx_local,
            test_idx_local,
            y_train,
            y_test,
            split_applied,
            posterior_beam_params.clone(),
            structure_prior_cfg,
            unit_seed,
            mem_tracker,
        )?;
        structure_repeats_used.fetch_add(out.repeats_used, Ordering::Relaxed);
        let alpha_meta = {
            let mut guard = structure_scores
                .lock()
                .map_err(|_| "GARFIELD structure prior mutex poisoned".to_string())?;
            guard.merge_from(&out.calibrator);
            format_structure_alpha_meta(&guard, structure_prior_cfg, structure_prior_display_len)
        };
        let done = structure_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
        if done % structure_notify_step == 0 || done == structure_task_total {
            garfield_stage_progress_notify(
                progress_callback,
                "structure_prior",
                done,
                structure_task_total.max(1),
                Some(alpha_meta.as_str()),
            )
            .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
