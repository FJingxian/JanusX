use super::*;

pub(super) fn max6_fused_grm_int_scan_supported(
    unit_kind: &str,
    groups: Option<&[Vec<(String, i32, i32)>]>,
) -> bool {
    match unit_kind {
        "window" => true,
        "gene" => groups.is_some_and(|groups| groups.iter().all(|group| group.len() == 1)),
        "geneset" => groups.is_some(),
        _ => false,
    }
}

pub(super) fn garfield_logic_search_bed_owned(
    prefix: String,
    y: Vec<f64>,
    grm: Option<Vec<f64>>,
    x_cov: Option<Vec<f64>>,
    q_cov: usize,
    sample_ids: Option<Vec<String>>,
    sample_indices: Option<Vec<usize>>,
    site_keep_precomputed: Option<Vec<bool>>,
    unit_kind: String,
    groups: Option<Vec<Vec<(String, i32, i32)>>>,
    null_groups: Option<Vec<Vec<(String, i32, i32)>>>,
    group_names: Option<Vec<String>>,
    extension: usize,
    step: Option<usize>,
    scan_bimranges: Option<Vec<String>>,
    bin_mode: String,
    ml_method: String,
    ml_importance: String,
    ml_top_k: usize,
    ml_top_frac: f64,
    permutation_repeats: usize,
    permutation_scoring: String,
    rule_null_penalty_method: String,
    rule_null_quantile: f64,
    rule_null_report_pvalue: bool,
    tree_cfg: ExtraTreesConfig,
    seed: u64,
    max_pick: usize,
    exhaustive_depth: usize,
    beam_width: usize,
    rank_score: String,
    maf_threshold: f32,
    logic_maf_threshold: f32,
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
    residual_method: String,
    pcg_max_iter: usize,
    pcg_tol: f64,
    pcg_small_n_max: usize,
    out_prefix: Option<String>,
    simbench_path: Option<String>,
    top_rules_per_unit: usize,
    max_output_rules: usize,
    max_output_ratio: f64,
    search_backend: String,
    rule_permutation: bool,
    prior_len: Option<Vec<f64>>,
    no_clean: bool,
    raw_design: bool,
    filter_xor_substates: bool,
    xor_search_enabled: bool,
    whole_genome_dev_mode: bool,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    debug_probe: Option<GarfieldBeamDebugProbe>,
    logic_mem_budget_bytes: u64,
    diagnostics: bool,
    sampled_screen_null_diagnostic: bool,
    max6_screen_exceedance_audit: bool,
    max6_screen_pot_audit: bool,
    sampled_screen_pair_seed: u64,
    sampled_screen_pair_seeds_raw: String,
    sampled_screen_maf_cutpoints: String,
    sampled_screen_r2_cutpoints: String,
    max6_screen_tail_audit: bool,
    max6_screen_tail_audit_repeats: usize,
    max6_screen_tail_pair_seed: u64,
    max6_screen_tail_support_aware: bool,
    max6_screen_tail_support_shares: Vec<usize>,
    prepared_max6_universe: Option<Arc<Max6PreparedGenotypeUniverse>>,
    prepared_sample_ids: Option<Vec<String>>,
    grm_int_only: bool,
    min_cell: usize,
    min_cell_fraction: f64,
    ld_tag_r2: f64,
    summary_callback: Option<Py<PyAny>>,
    write_provenance: bool,
) -> Result<GarfieldLogicPipelineResult, String> {
    let total_wall_t0 = Instant::now();
    let search_backend = GarfieldSearchBackend::parse(search_backend.as_str())?;
    let max6_grm_backend = matches!(
        search_backend,
        GarfieldSearchBackend::Max6GrmMax6 | GarfieldSearchBackend::Max6GrmMax6Legacy
    );
    let max6_ld_tag_backend = matches!(search_backend, GarfieldSearchBackend::Max6GrmMax6);
    validate_max6_ld_tag_r2(ld_tag_r2)?;
    if ld_tag_r2 != GARFIELD_MAX6_LD_TAG_R2_DEFAULT && !max6_ld_tag_backend {
        return Err("ld_tag_r2 requires the max6-grmmax6 backend".to_string());
    }
    min_cell::resolve_min_cell(min_cell, min_cell_fraction, 0)?;
    if min_cell_fraction > 0.0 && !max6_grm_backend {
        return Err("min_cell_fraction requires a Max6 GARFIELD search backend".to_string());
    }
    if grm_int_only && !max6_ld_tag_backend {
        return Err("GRM-INT reranking requires --search-backend max6-grmmax6".to_string());
    }
    let max6_explicit_audit = sampled_screen_null_diagnostic
        || max6_screen_exceedance_audit
        || max6_screen_pot_audit
        || max6_screen_tail_audit;
    let sampled_screen_pair_seeds = if sampled_screen_null_diagnostic {
        let raw = if sampled_screen_pair_seeds_raw.trim().is_empty() {
            sampled_screen_pair_seed.to_string()
        } else {
            sampled_screen_pair_seeds_raw
        };
        sampled_screen::parse_pair_sample_seeds(raw.as_str(), seed)?
    } else {
        Vec::new()
    };
    if sampled_screen_null_diagnostic {
        if !max6_ld_tag_backend || !rule_permutation {
            return Err(
                "sampled screen null diagnostic requires the max6-grmmax6 backend and fitted parametric null draws".to_string(),
            );
        }
        if permutation_repeats != 100 {
            return Err(format!(
                "sampled screen first-pilot diagnostic requires exactly 100 null draws; got {permutation_repeats}"
            ));
        }
        if out_prefix.is_none() {
            return Err(
                "sampled screen diagnostic requires out_prefix for its audit files".to_string(),
            );
        }
        sampled_screen::parse_cutpoints(&sampled_screen_maf_cutpoints, "min-MAF", 0.5)?;
        sampled_screen::parse_cutpoints(&sampled_screen_r2_cutpoints, "pairwise-r2", 1.0)?;
    }
    if max6_screen_tail_audit {
        // Tail audits may target a restricted, disjoint BIM region; requiring
        // whole-genome scan mode here would prevent the intended regional
        // replication audits. Python still requires the explicit -dev opt-in.
        if !max6_ld_tag_backend || !rule_permutation {
            return Err("Max6 full-tail audit requires -dev, --search-backend max6-grmmax6, and the parametric null pipeline".to_string());
        }
        if residual_method != "exact" {
            return Err("Max6 full-tail audit requires exact LMM residualization".to_string());
        }
        if !matches!(max6_screen_tail_audit_repeats, 100 | 250 | 500) {
            return Err(format!(
                "Max6 full-tail audit repeats must be 100, 250, or 500; got {max6_screen_tail_audit_repeats}"
            ));
        }
        if max6_screen_tail_pair_seed == seed {
            return Err("tail pair-sample seed must be separate from the null seed".to_string());
        }
        if out_prefix.is_none() {
            return Err("Max6 full-tail audit requires out_prefix".to_string());
        }
    }
    if max6_screen_tail_support_aware && !max6_screen_tail_audit {
        return Err(
            "support-aware Max6 tail sampling requires --max6-screen-tail-audit".to_string(),
        );
    }
    if max6_screen_exceedance_audit {
        if !max6_ld_tag_backend || !rule_permutation {
            return Err(
                "weighted Max6 exceedance audit requires -dev, max6-grmmax6, and full-MaxT calibration"
                    .to_string(),
            );
        }
        if residual_method.trim().to_ascii_lowercase() != "exact" {
            return Err(
                "weighted Max6 exceedance audit requires exact LMM residualization".to_string(),
            );
        }
        if !matches!(permutation_repeats, 100 | 500) {
            return Err(format!(
                "weighted Max6 exceedance audit requires 100 or 500 calibration nulls; got {permutation_repeats}"
            ));
        }
        if out_prefix.is_none() {
            return Err(
                "weighted Max6 exceedance audit requires an external output prefix".to_string(),
            );
        }
        validate_max6_audit_output_prefix_is_external(
            out_prefix.as_deref().expect("checked above"),
        )?;
        if sampled_screen_null_diagnostic || max6_screen_tail_audit {
            return Err(
                "weighted Max6 exceedance audit cannot be combined with the sampled-max or Top-4096 tail-audit modes"
                    .to_string(),
            );
        }
        if sampled_screen::WEIGHTED_EXCEEDANCE_PAIR_SEEDS.contains(&seed) {
            return Err(
                "weighted exceedance pair-sample seeds must be separate from the null seed"
                    .to_string(),
            );
        }
    }
    if max6_screen_pot_audit {
        if !max6_ld_tag_backend || !rule_permutation {
            return Err(
                "Max6 POT audit requires -dev, max6-grmmax6, and fitted parametric null draws"
                    .to_string(),
            );
        }
        if residual_method.trim().to_ascii_lowercase() != "exact" {
            return Err("Max6 POT audit requires exact LMM residualization".to_string());
        }
        if !matches!(permutation_repeats, 100 | 500) {
            return Err(format!(
                "Max6 POT audit requires 100 or 500 calibration null draws; got {permutation_repeats}"
            ));
        }
        if out_prefix.is_none() {
            return Err("Max6 POT audit requires an external output prefix".to_string());
        }
        validate_max6_audit_output_prefix_is_external(
            out_prefix.as_deref().expect("checked above"),
        )?;
        if sampled_screen_null_diagnostic || max6_screen_tail_audit {
            return Err(
                "Max6 POT audit cannot be combined with sampled-max or Top-4096 tail-audit modes"
                    .to_string(),
            );
        }
        if sampled_screen::POT_PAIR_SAMPLE_SEEDS.contains(&seed) {
            return Err("POT pair-sample seeds must be separate from the null seed".to_string());
        }
    }
    // Resolve hidden proposal settings before any BED/GRM work.  The default
    // is Off, which leaves the historical scan path untouched; malformed
    // development settings fail early instead of partially scanning data.
    let proposal_config = effective_proposal_config(search_backend, max_pick)?;
    if !matches!(proposal_config.mode, ProposalMode::Off) {
        garfield_prepare_breakpoint(
            "proposal_mode_ready",
            Some(&format!(
                "mode={:?} k_site={} k2={} max_order={}",
                proposal_config.mode,
                proposal_config.k_site,
                proposal_config.k2,
                proposal_config.max_order
            )),
        )?;
    }
    let rss_debug_enabled = garfield_rss_debug_enabled();
    let mut memory_debug = if rss_debug_enabled {
        Some(GarfieldMemoryDebugSummary::default())
    } else {
        None
    };
    validate_continuous_y(&y)?;
    let using_external_grm = grm.is_some();
    let residual_method_lc = residual_method.trim().to_ascii_lowercase();
    let use_lm_residualization = residual_method_lc == "lm";
    let use_pcg_residualization = matches!(
        residual_method_lc.as_str(),
        "rrblup_pcg" | "rrblup-pcg" | "pcg"
    );
    if !use_lm_residualization && !use_pcg_residualization && residual_method_lc != "exact" {
        return Err(format!(
            "unsupported residual_method '{}'; expected 'lm', 'exact', or 'rrblup_pcg'",
            residual_method
        ));
    }
    if use_lm_residualization && using_external_grm {
        return Err("LM residualization cannot be combined with a provided GRM".to_string());
    }
    if max6_grm_backend && use_pcg_residualization {
        return Err(
            "max6-grmmax6 requires exact LMM residualization because the exact candidate reranker needs P_C; rrblup_pcg is not supported on this backend"
                .to_string(),
        );
    }
    let rule_permutation =
        max6_null_calibration_enabled(max6_ld_tag_backend, rule_permutation, max6_explicit_audit);
    if use_pcg_residualization {
        if pcg_max_iter == 0 {
            return Err("pcg_max_iter must be > 0 for rrblup_pcg residualization".to_string());
        }
        if !pcg_tol.is_finite() || pcg_tol <= 0.0 {
            return Err(format!(
                "pcg_tol must be finite and > 0 for rrblup_pcg residualization, got {pcg_tol}"
            ));
        }
        if pcg_small_n_max == 0 {
            return Err("pcg_small_n_max must be > 0 for rrblup_pcg residualization".to_string());
        }
    }
    let fam_sample_ids = read_fam(normalize_plink_prefix(&prefix).as_str())?;
    let (selected_sample_indices, selected_sample_ids) =
        build_sample_selection(fam_sample_ids.as_slice(), sample_ids, sample_indices)?;
    match (
        prepared_max6_universe.as_ref(),
        prepared_sample_ids.as_ref(),
    ) {
        (Some(prepared), Some(global_sample_ids)) => {
            if !max6_ld_tag_backend {
                return Err(
                    "a prepared Max6 universe is supported only by max6-grmmax6".to_string()
                );
            }
            if using_external_grm && site_keep_precomputed.is_some() {
                return Err(
                    "external-GRM Max6 tag preparation cannot be combined with site_keep"
                        .to_string(),
                );
            }
            if max6_explicit_audit {
                return Err(
                    "explicit Max6 audit modes must use the unprepared diagnostic path".to_string(),
                );
            }
            let (global_indices, resolved_global_ids) = build_sample_selection(
                fam_sample_ids.as_slice(),
                Some(global_sample_ids.clone()),
                None,
            )?;
            if resolved_global_ids != *global_sample_ids {
                return Err(
                    "prepared Max6 sample-pool order does not match current FAM identities"
                        .to_string(),
                );
            }
            let config = Max6PreparationConfigFingerprint::new(
                unit_kind.trim().to_ascii_lowercase().as_str(),
                groups.as_deref(),
                group_names.as_deref(),
                extension,
                step,
                scan_bimranges.as_deref(),
                global_indices.as_slice(),
                maf_threshold,
                max_missing_rate,
                het_threshold,
                snps_only,
                ld_tag_r2,
            );
            let current = build_max6_prepared_input_fingerprint(
                prefix.as_str(),
                resolved_global_ids,
                &config,
            )?;
            prepared.ensure_compatible_input(&current)?;
            if prepared.tags.unit_kind != unit_kind.trim().to_ascii_lowercase() {
                return Err(format!(
                    "prepared Max6 unit kind '{}' does not match current '{}'",
                    prepared.tags.unit_kind,
                    unit_kind.trim().to_ascii_lowercase()
                ));
            }
        }
        (Some(_), None) => {
            return Err("prepared Max6 universe requires its ordered sample IDs".to_string());
        }
        (None, Some(_)) => {
            return Err("prepared sample IDs were supplied without a Max6 universe".to_string());
        }
        (None, None) => {}
    }
    let threads_eff = effective_threads_local(threads);
    let scheduler_scan_tasks = new_garfield_scheduler_task_collector();
    let scheduler_permutation_tasks = new_garfield_scheduler_task_collector();
    let unit_kind_lc = unit_kind.trim().to_ascii_lowercase();
    let fused_max6_grm_int_scan = max6_ld_tag_backend
        && grm_int_only
        && max6_fused_grm_int_scan_supported(unit_kind_lc.as_str(), groups.as_deref())
        && !max6_explicit_audit
        && prepared_max6_universe.is_none()
        && site_keep_precomputed.is_none();
    garfield_stage_progress_notify(
        progress_callback.as_ref(),
        "garfield_marker_qc",
        0,
        1,
        Some(
            if fused_max6_grm_int_scan && (using_external_grm || use_lm_residualization) {
                "deferring marker QC to the fused scan"
            } else {
                "reading PLINK metadata and applying marker QC"
            },
        ),
    )
    .map_err(|error| error.to_string())?;
    let grouped_active_mode = matches!(unit_kind_lc.as_str(), "gene" | "geneset" | "group");
    let grouped_null_mode = matches!(unit_kind_lc.as_str(), "gene" | "geneset");
    let scan_bimranges = scan_bimranges
        .as_deref()
        .map(parse_scan_bimranges)
        .transpose()?
        .unwrap_or_default();
    if unit_kind_lc == "wholegenome" && !scan_bimranges.is_empty() {
        return Err("--bimrange is not supported under unit_kind 'wholegenome'".to_string());
    }
    let mut grouped_scan_groups_selected: Option<Vec<Vec<(String, i32, i32)>>> = None;
    let mut grouped_scan_group_names_selected: Option<Vec<String>> = None;
    let mut grouped_null_group_defs = Vec::<Vec<(String, i32, i32)>>::new();
    let mut grouped_null_pool_total = 0usize;
    if grouped_active_mode {
        let groups_ref = groups.as_deref().ok_or_else(|| {
            format!(
                "groups must be provided for unit_kind in {{gene, geneset, group}}; got '{}'",
                unit_kind_lc
            )
        })?;
        let (scan_groups_selected, scan_group_names_selected, scan_unit_defs) =
            select_logic_group_interval_defs_for_scan(
                groups_ref,
                group_names.as_deref(),
                scan_bimranges.as_slice(),
                unit_kind_lc.as_str(),
            )?;
        garfield_prepare_breakpoint(
            "grouped_units_full_ready",
            Some(&format!("units={}", scan_unit_defs.len())),
        )?;
        if rule_permutation && grouped_null_mode {
            let (null_group_defs, null_pool_total) = build_synthetic_geneset_null_group_defs(
                scan_unit_defs.as_slice(),
                null_groups.as_deref(),
                groups.as_deref(),
                DEFAULT_RULE_NULL_PHYSICAL_CHUNKS,
                extension.max(1),
                seed,
            );
            grouped_null_group_defs = null_group_defs;
            grouped_null_pool_total = null_pool_total;
            garfield_prepare_breakpoint(
                "grouped_null_units_full_ready",
                Some(&format!(
                    "null_units={} null_pool_total={}",
                    grouped_null_group_defs.len(),
                    grouped_null_pool_total
                )),
            )?;
        }
        grouped_scan_groups_selected = Some(scan_groups_selected);
        grouped_scan_group_names_selected = Some(scan_group_names_selected);
    }
    let (
        grm_row_source_indices,
        mut logic_row_flip_external,
        filtered_sites,
        n_samples_total,
        bytes_per_snp,
    ) = if fused_max6_grm_int_scan && (using_external_grm || use_lm_residualization) {
        let header = BedSnpIter::new_for_grm_window_header_count(
            normalize_plink_prefix(&prefix).as_str(),
            garfield_logic_mmap_window_mb(),
        )?;
        (
            Vec::new(),
            Some(Vec::new()),
            Vec::new(),
            header.n_samples(),
            header.bytes_per_snp(),
        )
    } else if let Some(prepared) = prepared_max6_universe
        .as_ref()
        .filter(|_| using_external_grm)
    {
        let meta = prepare_bed_logic_meta_owned_for_cached_compact_source_rows_pure_line(
            &prefix,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            Some(selected_sample_indices.as_slice()),
            prepared.tags.active_source_rows.as_slice(),
            prepared.tags.active_source_sites.as_ref(),
        )?;
        (
            meta.row_source_indices,
            Some(meta.row_flip),
            meta.sites,
            meta.n_samples,
            meta.bytes_per_snp,
        )
    } else if grouped_active_mode && using_external_grm {
        let mut active_interval_groups = grouped_scan_groups_selected
            .as_ref()
            .cloned()
            .ok_or_else(|| "internal error: grouped scan intervals missing".to_string())?;
        active_interval_groups.extend(grouped_null_group_defs.iter().cloned());
        let meta = prepare_bed_logic_meta_owned_for_stats_samples_pure_line_intervals(
            &prefix,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            active_interval_groups.as_slice(),
            Some(selected_sample_indices.as_slice()),
        )?;
        garfield_prepare_breakpoint(
            "active_rows_ready",
            Some(&format!("rows={}", meta.row_source_indices.len())),
        )?;
        garfield_prepare_breakpoint(
            "subset_logic_metadata_ready",
            Some(&format!("rows={}", meta.row_source_indices.len())),
        )?;
        (
            meta.row_source_indices,
            Some(meta.row_flip),
            meta.sites,
            meta.n_samples,
            meta.bytes_per_snp,
        )
    } else if grouped_active_mode {
        let meta = prepare_bed_logic_meta_owned_for_stats_samples_pure_line_with_mmap_window(
            &prefix,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            Some(selected_sample_indices.as_slice()),
            true,
            None,
            threads_eff,
        )?;
        (
            meta.row_source_indices,
            Some(meta.row_flip),
            Vec::new(),
            meta.n_samples,
            meta.bytes_per_snp,
        )
    } else {
        let meta = if let Some(site_keep_precomputed) = site_keep_precomputed {
            prepare_bed_logic_meta_owned_for_precomputed_site_keep(&prefix, site_keep_precomputed)?
        } else {
            prepare_bed_logic_meta_owned_for_stats_samples_pure_line_with_mmap_window(
                &prefix,
                maf_threshold,
                max_missing_rate,
                het_threshold,
                snps_only,
                Some(selected_sample_indices.as_slice()),
                false,
                Some(garfield_logic_mmap_window_mb()),
                threads_eff,
            )?
        };
        let logic_row_flip = if using_external_grm || use_lm_residualization {
            Some(meta.row_flip)
        } else {
            None
        };
        (
            meta.row_source_indices,
            logic_row_flip,
            meta.sites,
            meta.n_samples,
            meta.bytes_per_snp,
        )
    };
    garfield_prepare_breakpoint(
        "logic_meta_ready",
        Some(&format!("rows={}", grm_row_source_indices.len())),
    )?;
    let marker_qc_detail =
        if fused_max6_grm_int_scan && (using_external_grm || use_lm_residualization) {
            format!("marker QC will run in the fused {unit_kind_lc} scan")
        } else {
            format!("{} retained marker rows", grm_row_source_indices.len())
        };
    garfield_stage_progress_notify(
        progress_callback.as_ref(),
        "garfield_marker_qc",
        1,
        1,
        Some(marker_qc_detail.as_str()),
    )
    .map_err(|error| error.to_string())?;
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

    let split_applied = false;
    let full = (0..n_selected).collect::<Vec<_>>();
    let (train_idx_local, test_idx_local) = (full.clone(), full);
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
    let ml_min_samples_leaf =
        strict_maf_support_count(train_idx_local.len(), logic_maf_threshold).max(1);
    let tree_cfg = ExtraTreesConfig {
        min_samples_leaf: tree_cfg.min_samples_leaf.max(ml_min_samples_leaf),
        min_samples_split: tree_cfg
            .min_samples_split
            .max(ml_min_samples_leaf.saturating_mul(2))
            .max(2),
        ..tree_cfg
    };

    let rule_null_penalty_method =
        parse_rule_null_penalty_method(rule_null_penalty_method.as_str(), rule_null_quantile)?;
    let null_keep_topk = if rule_null_penalty_method.uses_gev() {
        1usize
    } else {
        null_topk_per_repeat_for_bucket(RuleNullBucket {
            rule_len: 1,
            complexity_bin: 0,
        })
        .max(1)
    };
    let mut auto_grm_full: Option<Vec<f64>> = None;
    let mut auto_eff_m_full: Option<usize> = None;
    let mut logic_row_flip_auto: Option<Vec<bool>> = None;
    if !using_external_grm && !use_pcg_residualization && !use_lm_residualization {
        garfield_stage_progress_notify(
            progress_callback.as_ref(),
            "garfield_auto_grm",
            0,
            1,
            Some("constructing additive GRM from QC markers"),
        )
        .map_err(|error| error.to_string())?;
        let row_meta = compute_bed_row_meta_owned_for_source_rows(
            &prefix,
            grm_row_source_indices.as_slice(),
            Some(selected_sample_indices.as_slice()),
        )?;
        if row_meta.n_samples != n_samples_total {
            return Err(format!(
                "internal error: additive row-meta sample count {} != expected {}",
                row_meta.n_samples, n_samples_total
            ));
        }
        if row_meta.bytes_per_snp != bytes_per_snp {
            return Err(format!(
                "internal error: additive row-meta bytes_per_snp {} != expected {}",
                row_meta.bytes_per_snp, bytes_per_snp
            ));
        }
        let (grm_full_auto, _row_sum, varsum_full) = build_grm_from_meta_stream(
            &prefix,
            grm_row_source_indices.as_slice(),
            row_meta.row_flip.as_slice(),
            row_meta.maf.as_slice(),
            selected_sample_indices.as_slice(),
            StreamKernelMode::Additive,
            block_cols,
            threads_eff,
            None,
            0usize,
            None::<fn(usize, usize) -> Result<(), String>>,
        )?;
        auto_eff_m_full = Some(varsum_full.round().max(0.0) as usize);
        if !grouped_active_mode {
            logic_row_flip_auto = Some(row_meta.row_flip);
        }
        auto_grm_full = Some(grm_full_auto);
        garfield_stage_progress_notify(
            progress_callback.as_ref(),
            "garfield_auto_grm",
            1,
            1,
            Some("additive GRM ready"),
        )
        .map_err(|error| error.to_string())?;
    }
    let pcg_marker_meta: Option<(Vec<usize>, Vec<bool>, Vec<f32>)> = if use_pcg_residualization {
        // The precomputed mask above belongs to the pure-line logic scan. It
        // must not be reused for additive residualization: PCG uses the full
        // standard-QC marker set, including heterozygous sites.
        let full_qc = prepare_bed_logic_meta_owned_for_stats_samples_with_mmap_window(
            &prefix,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            Some(selected_sample_indices.as_slice()),
            true,
            None,
            threads_eff,
        )?;
        let pcg_rows = full_qc.row_source_indices;
        let meta = compute_bed_row_meta_owned_for_source_rows(
            prefix.as_str(),
            pcg_rows.as_slice(),
            Some(selected_sample_indices.as_slice()),
        )?;
        let (pcg_row_flip, pcg_maf) = (meta.row_flip, meta.maf);
        if pcg_rows.is_empty() {
            return Err("PCG residualization retained zero full-genome QC markers".to_string());
        }
        Some((pcg_rows, pcg_row_flip, pcg_maf))
    } else {
        None
    };
    if use_pcg_residualization && !grouped_active_mode {
        let logic_meta = compute_bed_row_meta_owned_for_source_rows(
            prefix.as_str(),
            grm_row_source_indices.as_slice(),
            Some(selected_sample_indices.as_slice()),
        )?;
        if using_external_grm {
            logic_row_flip_external = Some(logic_meta.row_flip);
        } else {
            logic_row_flip_auto = Some(logic_meta.row_flip);
        }
    }
    let pcg_config = || PcgResidualConfig {
        block_rows: block_cols.max(1),
        threads: threads_eff,
        small_n_max: pcg_small_n_max,
        reml_low: low,
        reml_high: high,
        reml_max_iter: max_iter,
        reml_tol: tol,
        pcg_max_iter,
        pcg_tol,
        seed,
        ..PcgResidualConfig::default()
    };
    let pcg_null_repeats = if rule_permutation {
        permutation_repeats
            .max(DEFAULT_RULE_NULL_ADAPTIVE_MIN_REPEATS)
            .min(DEFAULT_RULE_NULL_MAX_REPEATS)
    } else {
        0
    };
    garfield_stage_progress_notify(
        progress_callback.as_ref(),
        "garfield_residual_fit",
        0,
        1,
        Some("fitting observed-y null and generating calibration/validation draws"),
    )
    .map_err(|error| error.to_string())?;
    // The exact Max6 reranker needs the reusable covariance-metric projection
    // in addition to the fitted residual phenotype.
    // The ordinary exact fit hands its eigenvectors directly to P_C assembly;
    // only paths without that retained eigensystem need a separate GRM copy.
    let mut projection_eigenvectors = Vec::new();
    let projection_grm_for_max6 = if max6_grm_backend && (split_applied || use_pcg_residualization)
    {
        if let Some(external) = grm.as_ref() {
            Some(external.clone())
        } else {
            auto_grm_full.clone()
        }
    } else {
        None
    };
    let (train_fit, test_fit) = if split_applied {
        let train_fit = if use_pcg_residualization {
            let (row_source_indices, row_flip, row_maf) = pcg_marker_meta
                .as_ref()
                .ok_or_else(|| "internal error: PCG marker metadata is missing".to_string())?;
            garfield_residualize_pcg_from_bed(
                prefix.as_str(),
                row_source_indices.clone(),
                row_flip.clone(),
                row_maf.clone(),
                train_idx.clone(),
                y_train,
                x_train,
                q_cov,
                threads_eff,
                add_intercept,
                pcg_config(),
                0,
                seed,
            )?
        } else if use_lm_residualization {
            garfield_residualize_lm_rust(y_train, x_train, q_cov, add_intercept, 0, seed, 0)?
        } else {
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
                (
                    slice_square_matrix_by_indices(
                        auto_grm_full
                            .as_ref()
                            .ok_or_else(|| {
                                "internal error: auto GRM missing for train slice".to_string()
                            })?
                            .as_slice(),
                        n_selected,
                        train_idx_local.as_slice(),
                        "garfield_logic_search_bed: auto train GRM slice",
                    )?,
                    auto_eff_m_full,
                )
            };
            garfield_residualize_exact_from_grm_rust(
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
                0,
                0,
                0,
            )?
        };

        let test_fit = if use_pcg_residualization {
            let (row_source_indices, row_flip, row_maf) = pcg_marker_meta
                .as_ref()
                .ok_or_else(|| "internal error: PCG marker metadata is missing".to_string())?;
            garfield_residualize_pcg_from_bed(
                prefix.as_str(),
                row_source_indices.clone(),
                row_flip.clone(),
                row_maf.clone(),
                test_idx.clone(),
                y_test,
                x_test,
                q_cov,
                threads_eff,
                add_intercept,
                pcg_config(),
                0,
                seed.wrapping_add(1),
            )?
        } else if use_lm_residualization {
            garfield_residualize_lm_rust(
                y_test,
                x_test,
                q_cov,
                add_intercept,
                0,
                seed.wrapping_add(1),
                0,
            )?
        } else {
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
                (
                    slice_square_matrix_by_indices(
                        auto_grm_full
                            .as_ref()
                            .ok_or_else(|| {
                                "internal error: auto GRM missing for test slice".to_string()
                            })?
                            .as_slice(),
                        n_selected,
                        test_idx_local.as_slice(),
                        "garfield_logic_search_bed: auto test GRM slice",
                    )?,
                    auto_eff_m_full,
                )
            };
            garfield_residualize_exact_from_grm_rust(
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
                0,
                0,
                0,
            )?
        };
        (Arc::new(train_fit), Arc::new(test_fit))
    } else if use_pcg_residualization {
        let (row_source_indices, row_flip, row_maf) = pcg_marker_meta
            .as_ref()
            .ok_or_else(|| "internal error: PCG marker metadata is missing".to_string())?;
        let fit = garfield_residualize_pcg_from_bed(
            prefix.as_str(),
            row_source_indices.clone(),
            row_flip.clone(),
            row_maf.clone(),
            train_idx,
            y_train,
            x_train,
            q_cov,
            threads_eff,
            add_intercept,
            pcg_config(),
            pcg_null_repeats,
            seed,
        )?;
        let fit = Arc::new(fit);
        (fit.clone(), fit)
    } else if use_lm_residualization {
        let fit = garfield_residualize_lm_rust(
            y_train,
            x_train,
            q_cov,
            add_intercept,
            pcg_null_repeats,
            seed,
            if max6_screen_tail_audit {
                max6_screen_tail_audit_repeats
            } else {
                0
            },
        )?;
        let fit = Arc::new(fit);
        (fit.clone(), fit)
    } else {
        let (grm_full, eff_m_full) = if let Some(full_grm) = grm {
            if full_grm.len() != n_selected.saturating_mul(n_selected) {
                return Err(format!(
                    "garfield_logic_search_bed: provided GRM length mismatch: got {}, expected {}",
                    full_grm.len(),
                    n_selected.saturating_mul(n_selected)
                ));
            }
            (full_grm, None)
        } else {
            (
                auto_grm_full
                    .take()
                    .ok_or_else(|| "internal error: auto GRM missing for full fit".to_string())?,
                auto_eff_m_full,
            )
        };
        let fit = garfield_residualize_exact_with_eigenvectors_rust(
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
            pcg_null_repeats,
            seed,
            if max6_screen_tail_audit {
                max6_screen_tail_audit_repeats
            } else {
                0
            },
            if max6_grm_backend {
                Some(&mut projection_eigenvectors)
            } else {
                None
            },
        )?;
        let fit = Arc::new(fit);
        (fit.clone(), fit)
    };
    garfield_stage_progress_notify(
        progress_callback.as_ref(),
        "garfield_residual_fit",
        1,
        1,
        Some(&format!("n={n_selected}; null draws fitted")),
    )
    .map_err(|error| error.to_string())?;
    garfield_prepare_breakpoint(
        "residual_fit_ready",
        Some(&format!("n_selected={n_selected}")),
    )?;
    garfield_stage_progress_notify(
        summary_callback.as_ref().or(progress_callback.as_ref()),
        "garfield_null_summary",
        1,
        1,
        Some(&serde_json::json!({ "full_pve": train_fit.pve }).to_string()),
    )
    .map_err(|error| error.to_string())?;
    garfield_stage_progress_notify(
        progress_callback.as_ref(),
        "garfield_projection",
        0,
        1,
        Some("building exact covariance-metric projection"),
    )
    .map_err(|error| error.to_string())?;
    let (max6_projection, no_grm_projector) = if max6_grm_backend {
        if train_fit.no_grm_boundary {
            let (x_full, p) =
                build_design_matrix(x_cov.as_deref(), train_fit.n_samples, q_cov, add_intercept)?;
            (
                None,
                Some(build_no_grm_covariate_projector(
                    x_full.as_slice(),
                    train_fit.n_samples,
                    p,
                )?),
            )
        } else {
            let projection = if !projection_eigenvectors.is_empty() {
                build_exact_projection_from_eigensystem_rust(
                    &train_fit.eigenvalues,
                    &projection_eigenvectors,
                    train_fit.n_samples,
                    x_cov.as_deref(),
                    q_cov,
                    threads_eff,
                    train_fit.lbd,
                    add_intercept,
                )?
            } else {
                let grm_for_projection = projection_grm_for_max6.as_deref().ok_or_else(|| {
                    "internal error: GRM/eigensystem missing for max6-grmmax6 P_C construction"
                        .to_string()
                })?;
                build_exact_projection_from_grm_rust(
                    grm_for_projection,
                    train_fit.n_samples,
                    x_cov.as_deref(),
                    q_cov,
                    threads_eff,
                    train_fit.lbd,
                    add_intercept,
                    require_lapack,
                )?
            };
            (Some(projection), None)
        }
    } else {
        (None, None)
    };
    drop(projection_eigenvectors);
    drop(projection_grm_for_max6);
    garfield_stage_progress_notify(
        progress_callback.as_ref(),
        "garfield_projection",
        1,
        1,
        Some("projection ready"),
    )
    .map_err(|error| error.to_string())?;

    if fused_max6_grm_int_scan {
        let response = ResponseKind::Continuous;
        let engine = parse_optional_ml_engine(&ml_method)?;
        let importance = parse_importance(&ml_importance)?;
        let rank_mode = parse_beam_rank_mode(&rank_score)?;
        let perm_cfg = PermutationConfig {
            n_repeats: permutation_repeats
                .max(DEFAULT_RULE_NULL_ADAPTIVE_MIN_REPEATS)
                .min(DEFAULT_RULE_NULL_MAX_REPEATS),
            scoring: parse_permutation_scoring(&permutation_scoring)?,
            seed,
        };
        let beam_params = apply_proposal_search_config(
            BeamSearchParams {
                search_backend,
                max_pick: max_pick.max(1),
                beam_width: beam_width.max(1),
                min_gain: 0.0,
                min_parent_abs_gain: 0.0,
                surrogate_test_gain_max: 0.0,
                surrogate_hamming_frac_max: 0.0,
                maf_threshold: logic_maf_threshold.clamp(0.0, 0.5) as f64,
                lambda_len: 0.0,
                lambda_not: 0.0,
                exhaustive_depth: if whole_genome_dev_mode {
                    1
                } else {
                    exhaustive_depth.max(1)
                },
                rank_mode,
                null_penalties: None,
                structure_prior: None,
                y_sum_lookup: None,
                disable_parent_delta: false,
                null_complexity_bin: 0,
                group_constraint: BeamGroupConstraintMode::AlwaysExclude,
                allow_parallel: true,
                whole_genome_dev_mode,
                filter_xor_substates,
                xor_search_enabled,
            },
            proposal_config,
        );
        return run_max6_fused_grm_int_scan(
            unit_kind_lc.as_str(),
            if matches!(unit_kind_lc.as_str(), "gene" | "geneset") {
                grouped_scan_groups_selected.as_deref()
            } else {
                None
            },
            if matches!(unit_kind_lc.as_str(), "gene" | "geneset") {
                grouped_scan_group_names_selected.as_deref()
            } else {
                None
            },
            prefix.as_str(),
            selected_sample_ids.as_slice(),
            selected_sample_indices.as_slice(),
            extension,
            step,
            scan_bimranges.as_slice(),
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            threads_eff,
            logic_mem_budget_bytes,
            ld_tag_r2,
            y.as_slice(),
            train_fit,
            max6_projection.as_deref(),
            no_grm_projector.as_ref(),
            train_idx_local.as_slice(),
            test_idx_local.as_slice(),
            response,
            engine,
            importance,
            perm_cfg,
            ml_top_k,
            ml_top_frac,
            tree_cfg,
            &beam_params,
            top_rules_per_unit,
            min_cell,
            min_cell_fraction,
            seed,
            out_prefix.as_deref(),
            progress_callback.as_ref(),
            write_provenance,
        );
    }

    let mode = parse_bin_mode(&bin_mode)?;
    let logic_row_mul = 1usize;
    let mut logic_row_flip = if let Some(prepared_row_flip) = logic_row_flip_external.as_ref() {
        prepared_row_flip.clone()
    } else if grouped_active_mode {
        Vec::new()
    } else {
        logic_row_flip_auto
            .as_ref()
            .ok_or_else(|| {
                "internal error: auto logic row_flip missing for BED conversion".to_string()
            })?
            .clone()
    };
    let null_chunk_bp = extension.max(1).saturating_mul(2);
    let null_chunk_target = if rule_permutation {
        DEFAULT_RULE_NULL_PHYSICAL_CHUNKS
    } else {
        0
    };
    let mut logic_row_source_indices = grm_row_source_indices;
    let mut logic_sites = filtered_sites;
    if let Some(prepared) = prepared_max6_universe
        .as_ref()
        .filter(|_| !using_external_grm)
    {
        let trait_meta = prepare_bed_logic_meta_owned_for_cached_compact_source_rows_pure_line(
            &prefix,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            Some(selected_sample_indices.as_slice()),
            prepared.tags.active_source_rows.as_slice(),
            prepared.tags.active_source_sites.as_ref(),
        )?;
        logic_row_source_indices = trait_meta.row_source_indices;
        logic_row_flip = trait_meta.row_flip;
        logic_sites = trait_meta.sites;
    }
    if grouped_active_mode && !using_external_grm && prepared_max6_universe.is_none() {
        let mut active_interval_groups = grouped_scan_groups_selected
            .as_ref()
            .cloned()
            .ok_or_else(|| "internal error: grouped scan intervals missing".to_string())?;
        active_interval_groups.extend(grouped_null_group_defs.iter().cloned());
        let active_meta = prepare_bed_logic_meta_owned_for_stats_samples_pure_line_intervals(
            &prefix,
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            active_interval_groups.as_slice(),
            Some(selected_sample_indices.as_slice()),
        )?;
        if active_meta.n_samples != n_samples_total {
            return Err(format!(
                "internal error: active interval sample count {} != expected {}",
                active_meta.n_samples, n_samples_total
            ));
        }
        if active_meta.bytes_per_snp != bytes_per_snp {
            return Err(format!(
                "internal error: active interval bytes_per_snp {} != expected {}",
                active_meta.bytes_per_snp, bytes_per_snp
            ));
        }
        garfield_prepare_breakpoint(
            "active_rows_ready",
            Some(&format!("rows={}", active_meta.row_source_indices.len())),
        )?;
        garfield_prepare_breakpoint(
            "subset_logic_metadata_ready",
            Some(&format!("rows={}", active_meta.row_source_indices.len())),
        )?;
        logic_row_source_indices = active_meta.row_source_indices;
        logic_sites = active_meta.sites;
        logic_row_flip = active_meta.row_flip;
    }
    let prepared_trait_screen = if let Some(prepared) = prepared_max6_universe.as_ref() {
        Some(remap_max6_prepared_universe_for_trait(
            &prepared.tags,
            logic_row_source_indices.as_slice(),
            logic_row_source_indices.len(),
        )?)
    } else {
        None
    };
    let cached_max6_logic_sites = if let Some(prepared) = prepared_max6_universe.as_ref() {
        Some(build_max6_logic_sites_from_shared_metadata(
            prepared.tags.active_source_rows.as_slice(),
            prepared.tags.active_source_sites.clone(),
            logic_row_source_indices.as_slice(),
            logic_row_flip.as_slice(),
        )?)
    } else {
        None
    };
    let logic_bits_progress_detail = if prepared_max6_universe.is_some() {
        "packing LD-selected tag genotypes for screen/rerank"
    } else {
        "packing QC genotypes for screen/rerank"
    };
    garfield_stage_progress_notify(
        progress_callback.as_ref(),
        "garfield_logic_bits",
        0,
        1,
        Some(logic_bits_progress_detail),
    )
    .map_err(|error| error.to_string())?;
    let global_bits_mem_tracker = GarfieldStageMemoryTracker::new(rss_debug_enabled);
    let global_bits_mem_start = global_bits_mem_tracker.start_stage();
    let logic_bits = convert_bed_prefix_to_logic_bits(
        &prefix,
        bytes_per_snp,
        n_samples_total,
        logic_row_source_indices.as_slice(),
        logic_row_flip.as_slice(),
        logic_sites,
        cached_max6_logic_sites,
        selected_sample_indices.as_slice(),
        selected_sample_ids,
        mode,
        logic_mem_budget_bytes,
        Some(&global_bits_mem_tracker),
    )?;
    let logic_bits_backend = if logic_bits.bits_mmap.is_some() {
        "mmap"
    } else {
        "resident"
    }
    .to_string();
    let logic_bits_bytes = logic_bits
        .bits()
        .len()
        .saturating_mul(std::mem::size_of::<u64>());
    drop(logic_row_source_indices);
    drop(logic_row_flip);
    garfield_prepare_breakpoint(
        "logic_bits_ready",
        Some(&format!(
            "rows={} samples={}",
            logic_bits.sites.len(),
            logic_bits.n_samples
        )),
    )?;
    garfield_stage_progress_notify(
        progress_callback.as_ref(),
        "garfield_logic_bits",
        1,
        1,
        Some(&format!(
            "{} marker rows × {} samples",
            logic_bits.sites.len(),
            logic_bits.n_samples
        )),
    )
    .map_err(|error| error.to_string())?;
    if let Some(debug) = memory_debug.as_mut() {
        debug.global_bits_loaded = global_bits_mem_tracker.finish_stage(global_bits_mem_start);
    }
    let (null_chunks, mut null_chunk_valid_total) = if rule_permutation && !grouped_null_mode {
        sample_null_chunks_stratified(
            logic_bits.sites.as_slice(),
            extension.max(1),
            DEFAULT_RULE_NULL_PHYSICAL_CHUNKS,
            DEFAULT_RULE_NULL_MIN_SNPS_PER_CHUNK,
            seed,
        )?
    } else {
        (Vec::new(), grouped_null_pool_total)
    };
    let (units, prepared_screen_universe) =
        if let Some((prepared_units, prepared_screen)) = prepared_trait_screen {
            (prepared_units, Some(prepared_screen))
        } else if grouped_active_mode {
            build_logic_units(
                logic_bits.sites.as_slice(),
                &unit_kind,
                grouped_scan_groups_selected.as_deref(),
                grouped_scan_group_names_selected.as_deref(),
                extension,
                step,
            )
            .map(|units| (units, None))?
        } else {
            build_logic_units(
                logic_bits.sites.as_slice(),
                &unit_kind,
                groups.as_deref(),
                group_names.as_deref(),
                extension,
                step,
            )
            .map(|units| (units, None))?
        };
    if units.is_empty() {
        return Err("no scan units were built from the provided input".to_string());
    }

    let response = ResponseKind::Continuous;
    let engine = parse_optional_ml_engine(&ml_method)?;
    let importance = parse_importance(&ml_importance)?;
    let rank_mode = parse_beam_rank_mode(&rank_score)?;
    let perm_cfg = PermutationConfig {
        n_repeats: permutation_repeats
            .max(DEFAULT_RULE_NULL_ADAPTIVE_MIN_REPEATS)
            .min(DEFAULT_RULE_NULL_MAX_REPEATS),
        scoring: parse_permutation_scoring(&permutation_scoring)?,
        seed,
    };
    let structure_prior_cfg = RuleStructurePriorConfig::from_len_alpha_values(prior_len.as_deref());
    let structure_prior_display_len = max_pick.clamp(1, 5);
    let beam_params = apply_proposal_search_config(
        BeamSearchParams {
            search_backend,
            max_pick: max_pick.max(1),
            beam_width: beam_width.max(1),
            min_gain: 0.0,
            min_parent_abs_gain: 0.0,
            surrogate_test_gain_max: 0.0,
            surrogate_hamming_frac_max: 0.0,
            maf_threshold: logic_maf_threshold.clamp(0.0, 0.5) as f64,
            lambda_len: 0.0,
            lambda_not: 0.0,
            exhaustive_depth: if whole_genome_dev_mode {
                1
            } else {
                exhaustive_depth.max(1)
            },
            rank_mode,
            null_penalties: None,
            structure_prior: None,
            y_sum_lookup: None,
            disable_parent_delta: false,
            null_complexity_bin: 0,
            group_constraint: BeamGroupConstraintMode::AlwaysExclude,
            allow_parallel: true,
            whole_genome_dev_mode,
            filter_xor_substates,
            xor_search_enabled,
        },
        proposal_config,
    );
    let total_units = units.len();
    let scan_unit_indices = if grouped_active_mode {
        (0..total_units).collect::<Vec<_>>()
    } else if scan_bimranges.is_empty() {
        (0..total_units).collect::<Vec<_>>()
    } else {
        units
            .iter()
            .enumerate()
            .filter_map(|(ui, unit)| {
                if unit_overlaps_scan_bimranges(unit, scan_bimranges.as_slice()) {
                    Some(ui)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    };
    if scan_unit_indices.is_empty() {
        let joined = scan_bimranges
            .iter()
            .map(|r| format!("{}:{}-{}", r.chrom, r.bp_start, r.bp_end))
            .collect::<Vec<_>>()
            .join(",");
        return Err(format!(
            "--bimrange matched no scan units under unit_kind '{}': {}",
            unit_kind_lc, joined
        ));
    }
    let scanned_units = scan_unit_indices.len();
    if max6_grm_backend {
        return run_max6_grmmax6_pipeline(
            total_wall_t0,
            y.as_slice(),
            max6_projection.as_deref(),
            no_grm_projector.as_ref(),
            train_fit,
            &logic_bits,
            units.as_slice(),
            scan_unit_indices.as_slice(),
            unit_kind_lc.as_str(),
            response,
            engine,
            importance,
            perm_cfg,
            ml_top_k,
            ml_top_frac,
            tree_cfg,
            train_idx_local.as_slice(),
            test_idx_local.as_slice(),
            beam_params,
            threads_eff,
            top_rules_per_unit,
            min_cell,
            min_cell_fraction,
            grm_int_only,
            permutation_repeats,
            rule_permutation,
            rule_null_quantile,
            diagnostics,
            sampled_screen_null_diagnostic,
            max6_screen_exceedance_audit,
            max6_screen_pot_audit,
            sampled_screen_pair_seeds.as_slice(),
            sampled_screen_maf_cutpoints.as_str(),
            sampled_screen_r2_cutpoints.as_str(),
            max6_screen_tail_audit,
            max6_screen_tail_audit_repeats,
            max6_screen_tail_pair_seed,
            max6_screen_tail_support_aware,
            max6_screen_tail_support_shares,
            max6_ld_tag_backend,
            prepared_max6_universe.as_deref(),
            prepared_screen_universe,
            seed,
            out_prefix.as_deref(),
            progress_callback.as_ref(),
            progress_every,
            ld_tag_r2,
            write_provenance,
        );
    }
    let ld_support_cache_t0 = Instant::now();
    let ld_support_cache_rows = if garfield_ld_support_cache_enabled()
        && sample_indices_are_full_identity(train_idx_local.as_slice(), logic_bits.n_samples)
    {
        collect_scan_ld_cache_rows(
            units.as_slice(),
            scan_unit_indices.as_slice(),
            logic_bits.sites.len(),
            unit_kind_lc.as_str(),
        )
    } else {
        Vec::new()
    };
    let ld_support_cache = if ld_support_cache_rows.is_empty() {
        None
    } else {
        Some(Arc::new(build_garfield_ld_support_cache(
            &logic_bits,
            ld_support_cache_rows.as_slice(),
            train_idx_local.as_slice(),
        )?))
    };
    let timing_ld_support_cache_s = ld_support_cache_t0.elapsed().as_secs_f64();
    let ld_support_cache_rows_count = ld_support_cache
        .as_ref()
        .map(|cache| {
            if cache.sparse_global_to_local.is_some() {
                cache.support.len()
            } else {
                cache.n_rows
            }
        })
        .unwrap_or(0);
    let ld_support_cache_payload_bytes = ld_support_cache
        .as_ref()
        .map(|cache| cache.payload_bytes())
        .unwrap_or(0);
    let max_null_unit_group_count = if grouped_null_mode {
        scan_unit_indices
            .iter()
            .filter_map(|&ui| units.get(ui))
            .map(|unit| unit.spans.len().max(1))
            .max()
            .unwrap_or(1)
            .min(usize::from(u8::MAX).saturating_add(1))
    } else {
        1usize
    };
    let progress_callback_parallel = progress_callback.as_ref();
    let (null_geneset_units, null_geneset_pool_total) = if rule_permutation && grouped_null_mode {
        (
            build_logic_units(
                logic_bits.sites.as_slice(),
                &unit_kind,
                Some(grouped_null_group_defs.as_slice()),
                None,
                extension,
                step,
            )?,
            grouped_null_pool_total,
        )
    } else {
        (Vec::new(), 0usize)
    };
    if grouped_null_mode {
        null_chunk_valid_total = null_geneset_pool_total;
    }
    let null_prep_total = if grouped_null_mode {
        null_geneset_units.len()
    } else {
        null_chunks.len()
    };
    let mut null_prepared = Vec::<(GarfieldPermutationNullSource, GarfieldUnitPrepared)>::new();
    if null_prep_total > 0 {
        let prep_beam_params = BeamSearchParams {
            allow_parallel: false,
            ..beam_params.clone()
        };
        let prep_notify_step = if progress_every == 0 {
            (null_prep_total.max(1) / 200).max(1)
        } else {
            progress_every.max(1)
        };
        let prep_progress_done = AtomicUsize::new(0);
        garfield_stage_progress_notify(
            progress_callback.as_ref(),
            "null_prep",
            0,
            null_prep_total.max(1),
            None,
        )
        .map_err(|e| e.to_string())?;
        let prep_threads = threads_eff.min(null_prep_total.max(1));
        let null_prep_engine = null_prepare_engine_for_scan(unit_kind_lc.as_str(), engine);
        let prep_results = if grouped_null_mode {
            if prep_threads > 1 {
                let pool = ThreadPoolBuilder::new()
                    .num_threads(prep_threads)
                    .build()
                    .map_err(|e| format!("build GARFIELD null-prep thread pool: {e}"))?;
                pool.install(|| {
                    null_geneset_units
                        .par_iter()
                        .enumerate()
                        .map(|(null_idx, unit)| {
                            let prepared = prepare_logic_unit_continuous_with_ld_cache(
                                unit,
                                &unit_kind_lc,
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
                                None,
                                prep_beam_params.clone(),
                                ld_support_cache.as_deref(),
                                false,
                            );
                            let done = prep_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                            if done % prep_notify_step == 0 || done == null_prep_total {
                                garfield_stage_progress_notify(
                                    progress_callback_parallel,
                                    "null_prep",
                                    done,
                                    null_prep_total.max(1),
                                    None,
                                )
                                .map_err(|e| e.to_string())?;
                                garfield_prepare_breakpoint(
                                    "null_prep_progress",
                                    Some(&format!("done={done}/{}", null_prep_total.max(1))),
                                )?;
                            }
                            prepared.map(|opt| {
                                opt.map(|p| {
                                    (GarfieldPermutationNullSource::SyntheticGeneset(null_idx), p)
                                })
                            })
                        })
                        .collect::<Vec<
                            Result<
                                Option<(GarfieldPermutationNullSource, GarfieldUnitPrepared)>,
                                String,
                            >,
                        >>()
                })
            } else {
                let mut out = Vec::<
                    Result<Option<(GarfieldPermutationNullSource, GarfieldUnitPrepared)>, String>,
                >::with_capacity(null_geneset_units.len());
                for (null_idx, unit) in null_geneset_units.iter().enumerate() {
                    let prepared = prepare_logic_unit_continuous_with_ld_cache(
                        unit,
                        &unit_kind_lc,
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
                        None,
                        prep_beam_params.clone(),
                        ld_support_cache.as_deref(),
                        false,
                    );
                    let done = prep_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % prep_notify_step == 0 || done == null_prep_total {
                        garfield_stage_progress_notify(
                            progress_callback.as_ref(),
                            "null_prep",
                            done,
                            null_prep_total.max(1),
                            None,
                        )
                        .map_err(|e| e.to_string())?;
                        garfield_prepare_breakpoint(
                            "null_prep_progress",
                            Some(&format!("done={done}/{}", null_prep_total.max(1))),
                        )?;
                    }
                    out.push(prepared.map(|opt| {
                        opt.map(|p| (GarfieldPermutationNullSource::SyntheticGeneset(null_idx), p))
                    }));
                }
                out
            }
        } else if prep_threads > 1 {
            let pool = ThreadPoolBuilder::new()
                .num_threads(prep_threads)
                .build()
                .map_err(|e| format!("build GARFIELD null-prep thread pool: {e}"))?;
            pool.install(|| {
                null_chunks
                    .par_iter()
                    .copied()
                    .map(|chunk| {
                        let prepared = prepare_logic_chunk_continuous(
                            chunk,
                            logic_row_mul,
                            response,
                            null_prep_engine,
                            importance,
                            perm_cfg,
                            ml_top_k,
                            ml_top_frac,
                            tree_cfg,
                            &logic_bits,
                            train_idx_local.as_slice(),
                            test_idx_local.as_slice(),
                            train_fit.residualized_y.as_slice(),
                            prep_beam_params.clone(),
                        );
                        let done = prep_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % prep_notify_step == 0 || done == null_prep_total {
                            garfield_stage_progress_notify(
                                progress_callback_parallel,
                                "null_prep",
                                done,
                                null_prep_total.max(1),
                                None,
                            )
                            .map_err(|e| e.to_string())?;
                            garfield_prepare_breakpoint(
                                "null_prep_progress",
                                Some(&format!("done={done}/{}", null_prep_total.max(1))),
                            )?;
                        }
                        prepared.map(|opt| {
                            opt.map(|p| (GarfieldPermutationNullSource::Chunk(chunk), p))
                        })
                    })
                    .collect::<Vec<
                        Result<
                            Option<(GarfieldPermutationNullSource, GarfieldUnitPrepared)>,
                            String,
                        >,
                    >>()
            })
        } else {
            let mut out = Vec::<
                Result<Option<(GarfieldPermutationNullSource, GarfieldUnitPrepared)>, String>,
            >::with_capacity(null_chunks.len());
            for chunk in null_chunks.iter().copied() {
                let prepared = prepare_logic_chunk_continuous(
                    chunk,
                    logic_row_mul,
                    response,
                    null_prep_engine,
                    importance,
                    perm_cfg,
                    ml_top_k,
                    ml_top_frac,
                    tree_cfg,
                    &logic_bits,
                    train_idx_local.as_slice(),
                    test_idx_local.as_slice(),
                    train_fit.residualized_y.as_slice(),
                    prep_beam_params.clone(),
                );
                let done = prep_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                if done % prep_notify_step == 0 || done == null_prep_total {
                    garfield_stage_progress_notify(
                        progress_callback.as_ref(),
                        "null_prep",
                        done,
                        null_prep_total.max(1),
                        None,
                    )
                    .map_err(|e| e.to_string())?;
                    garfield_prepare_breakpoint(
                        "null_prep_progress",
                        Some(&format!("done={done}/{}", null_prep_total.max(1))),
                    )?;
                }
                out.push(
                    prepared
                        .map(|opt| opt.map(|p| (GarfieldPermutationNullSource::Chunk(chunk), p))),
                );
            }
            out
        };
        for prep_out in prep_results.into_iter() {
            if let Some((source, prepared)) = prep_out? {
                null_prepared.push((source, prepared));
            }
        }
        garfield_prepare_breakpoint(
            "null_prep_ready",
            Some(&format!("prepared={}", null_prepared.len())),
        )?;
    }
    let null_chunk_selected = null_prepared.len();
    let null_permutation_active = rule_permutation && !null_prepared.is_empty();
    let representative_units_target = if rule_permutation && !GARFIELD_DISABLE_STRUCTURE_PRIOR {
        DEFAULT_RULE_PERMUTATION_REPRESENTATIVE_UNITS.min(units.len())
    } else {
        0
    };

    let representative_units = if rule_permutation && !GARFIELD_DISABLE_STRUCTURE_PRIOR {
        choose_representative_indices(
            units
                .iter()
                .map(|unit| unit.indices.len())
                .collect::<Vec<_>>()
                .as_slice(),
            representative_units_target,
        )
    } else {
        Vec::new()
    };
    let mut representative_prepared = Vec::<(usize, GarfieldUnitPrepared)>::new();
    if !representative_units.is_empty() {
        let prep_beam_params = BeamSearchParams {
            allow_parallel: false,
            ..beam_params.clone()
        };
        let prep_notify_step = if progress_every == 0 {
            (representative_units.len().max(1) / 200).max(1)
        } else {
            progress_every.max(1)
        };
        let prep_progress_done = AtomicUsize::new(0);
        garfield_stage_progress_notify(
            progress_callback.as_ref(),
            "structure_prep",
            0,
            representative_units.len().max(1),
            None,
        )
        .map_err(|e| e.to_string())?;
        let prep_threads = threads_eff.min(representative_units.len().max(1));
        let prep_results = if prep_threads > 1 {
            let pool = ThreadPoolBuilder::new()
                .num_threads(prep_threads)
                .build()
                .map_err(|e| format!("build GARFIELD prior-prep thread pool: {e}"))?;
            pool.install(|| {
                representative_units
                    .par_iter()
                    .copied()
                    .map(|ui| {
                        let prepared = prepare_logic_unit_continuous_with_ld_cache(
                            &units[ui],
                            &unit_kind_lc,
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
                            None,
                            prep_beam_params.clone(),
                            ld_support_cache.as_deref(),
                            false,
                        );
                        let done = prep_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % prep_notify_step == 0 || done == representative_units.len() {
                            garfield_stage_progress_notify(
                                progress_callback_parallel,
                                "structure_prep",
                                done,
                                representative_units.len().max(1),
                                None,
                            )
                            .map_err(|e| e.to_string())?;
                            garfield_prepare_breakpoint(
                                "structure_prep_progress",
                                Some(&format!(
                                    "done={done}/{}",
                                    representative_units.len().max(1)
                                )),
                            )?;
                        }
                        prepared.map(|opt| opt.map(|p| (ui, p)))
                    })
                    .collect::<Vec<Result<Option<(usize, GarfieldUnitPrepared)>, String>>>()
            })
        } else {
            let mut out =
                Vec::<Result<Option<(usize, GarfieldUnitPrepared)>, String>>::with_capacity(
                    representative_units.len(),
                );
            for &ui in representative_units.iter() {
                let prepared = prepare_logic_unit_continuous_with_ld_cache(
                    &units[ui],
                    &unit_kind_lc,
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
                    None,
                    prep_beam_params.clone(),
                    ld_support_cache.as_deref(),
                    false,
                );
                let done = prep_progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                if done % prep_notify_step == 0 || done == representative_units.len() {
                    garfield_stage_progress_notify(
                        progress_callback.as_ref(),
                        "structure_prep",
                        done,
                        representative_units.len().max(1),
                        None,
                    )
                    .map_err(|e| e.to_string())?;
                    garfield_prepare_breakpoint(
                        "structure_prep_progress",
                        Some(&format!(
                            "done={done}/{}",
                            representative_units.len().max(1)
                        )),
                    )?;
                }
                out.push(prepared.map(|opt| opt.map(|p| (ui, p))));
            }
            out
        };
        for prep_out in prep_results.into_iter() {
            if let Some((ui, prepared)) = prep_out? {
                representative_prepared.push((ui, prepared));
            }
        }
        garfield_prepare_breakpoint(
            "structure_prep_ready",
            Some(&format!("prepared={}", representative_prepared.len())),
        )?;
    }
    let representative_units_used = representative_prepared.len();
    let soft_structure_mode = rule_permutation
        && !GARFIELD_DISABLE_STRUCTURE_PRIOR
        && !representative_prepared.is_empty();
    let effective_null_max_rule_len = beam_params
        .max_pick
        .max(
            scan_unit_indices
                .iter()
                .filter_map(|&ui| units.get(ui))
                .map(|unit| {
                    if unit_kind_lc == "geneset" {
                        unit.spans.len().max(1)
                    } else {
                        1usize
                    }
                })
                .max()
                .unwrap_or(1),
        )
        .max(
            null_geneset_units
                .iter()
                .map(|unit| {
                    if unit_kind_lc == "geneset" {
                        unit.spans.len().max(1)
                    } else {
                        1usize
                    }
                })
                .max()
                .unwrap_or(1),
        )
        .max(1);
    let permutation_task_total = if null_permutation_active {
        null_prepared
            .len()
            .saturating_mul(perm_cfg.n_repeats.max(1))
    } else {
        0
    };
    let structure_task_total = if soft_structure_mode {
        representative_prepared.len()
    } else {
        0
    };

    let mut permutation_null_repeats_used = 0usize;
    let (rule_search_null_lookup, rule_output_null_lookup): (
        Option<Arc<RuleNullPenaltyLookup>>,
        Option<Arc<RuleNullPenaltyLookup>>,
    ) = if null_permutation_active {
        let null_mem_tracker = GarfieldStageMemoryTracker::new(rss_debug_enabled);
        let null_mem_start = null_mem_tracker.start_stage();
        let null_notify_step = if progress_every == 0 {
            (permutation_task_total.max(1) / 200).max(1)
        } else {
            progress_every.max(1)
        };
        let null_progress_done = AtomicUsize::new(0);
        garfield_stage_progress_notify(
            progress_callback.as_ref(),
            "null_penalty",
            0,
            permutation_task_total.max(1),
            None,
        )
        .map_err(|e| e.to_string())?;
        let perm_beam_params = BeamSearchParams {
            allow_parallel: false,
            ..beam_params.clone()
        };
        let mut search_bucket_scores =
            RuleNullCalibrator::with_layout(effective_null_max_rule_len, max_null_unit_group_count);
        let mut output_bucket_scores =
            RuleNullCalibrator::with_layout(effective_null_max_rule_len, max_null_unit_group_count);
        let mut output_delta_bucket_scores =
            RuleNullCalibrator::with_layout(effective_null_max_rule_len, max_null_unit_group_count);
        // Development-only reducer for a shared-inner maxT diagnostic.  The
        // normal calibrators above remain unchanged; when enabled, one
        // maximum is recorded per family for each inner replicate after all
        // prepared windows have been evaluated.  This keeps the null draw
        // shared across windows without retaining candidate-level rows.
        let shared_max_t_enabled = !matches!(proposal_config.mode, ProposalMode::Off)
            && env_truthy("JX_GARFIELD_PROPOSAL_SHARED_MAXT");
        let mut proposal_shared_max_t = shared_max_t_enabled.then(SharedMaxTAccumulator::default);
        let min_perm_repeats =
            DEFAULT_RULE_NULL_ADAPTIVE_MIN_REPEATS.min(perm_cfg.n_repeats.max(1));
        let mut stable_rounds = 0usize;
        let mut prev_search_lookup: Option<RuleNullPenaltyLookup> = None;
        let mut prev_output_lookup: Option<RuleNullPenaltyLookup> = None;
        let perm_threads = threads_eff.min(
            null_prepared
                .len()
                .saturating_mul(perm_cfg.n_repeats.max(1))
                .max(1),
        );
        let repeats_per_batch = if null_prepared.is_empty() {
            1usize
        } else {
            threads_eff.div_ceil(null_prepared.len().max(1)).clamp(1, 4)
        };
        // Window nulls must repeat the phenotype-driven ML selection for each
        // permutation.  Reuse the fixed prepared pool only for grouped
        // synthetic units, whose layout is part of the grouping null model.
        let null_reselection = if !grouped_null_mode
            && (engine.is_some() || !matches!(proposal_config.mode, ProposalMode::Off))
        {
            Some(GarfieldNullReselectionConfig {
                row_mul: logic_row_mul,
                response,
                engine,
                importance,
                perm_cfg,
                ml_top_k,
                ml_top_frac,
                tree_cfg,
            })
        } else {
            None
        };
        let perm_pool = if perm_threads > 1 {
            Some(
                ThreadPoolBuilder::new()
                    .num_threads(perm_threads)
                    .build()
                    .map_err(|e| format!("build GARFIELD permutation thread pool: {e}"))?,
            )
        } else {
            None
        };
        // A null slot is evaluated for several permutation repeats. Keep its
        // prepared bit matrix alive across repeats instead of rebuilding the
        // same row gathers and allocations for every slot x repeat job. Fall
        // back to the historical per-slot materialization when the estimated
        // cache would exceed the bounded memory budget.
        let permutation_bits_cache = if estimate_permutation_bits_cache_bytes(
            null_prepared.as_slice(),
            &logic_bits,
            train_idx_local.as_slice(),
            test_idx_local.as_slice(),
        ) <= garfield_perm_bits_cache_max_bytes()
            && null_reselection.is_none()
        {
            let cache = if let Some(pool) = perm_pool.as_ref() {
                pool.install(|| {
                    null_prepared
                        .par_iter()
                        .map(|(_, prepared)| {
                            materialize_prepared_bit_matrices(
                                prepared,
                                &logic_bits,
                                train_idx_local.as_slice(),
                                test_idx_local.as_slice(),
                                false,
                            )
                        })
                        .collect::<Result<Vec<_>, String>>()
                })?
            } else {
                null_prepared
                    .iter()
                    .map(|(_, prepared)| {
                        materialize_prepared_bit_matrices(
                            prepared,
                            &logic_bits,
                            train_idx_local.as_slice(),
                            test_idx_local.as_slice(),
                            false,
                        )
                    })
                    .collect::<Result<Vec<_>, String>>()?
            };
            null_mem_tracker.sample_now();
            Some(cache)
        } else {
            None
        };
        let max_perm_repeats = perm_cfg.n_repeats.max(1);
        let mut rep_start = 0usize;
        'perm_batches: while rep_start < max_perm_repeats {
            let rep_end = (rep_start + repeats_per_batch).min(max_perm_repeats);
            let n_slots = null_prepared.len();
            let slot_indices = (0..n_slots).collect::<Vec<_>>();
            let slot_grain = if perm_threads > 1 {
                let default_grain = n_slots.div_ceil(perm_threads.saturating_mul(4)).max(1);
                match garfield_perm_slot_task_max_slots() {
                    0 => default_grain,
                    max_slots => default_grain.min(max_slots.max(1)),
                }
            } else {
                n_slots.max(1)
            };
            let use_flat_perm_tasks = garfield_scheduler_optimization_enabled()
                && permutation_bits_cache.is_some()
                && perm_threads > 1
                && (n_slots < perm_threads.saturating_mul(2) || repeats_per_batch > 1);
            let flat_perm_tasks = if use_flat_perm_tasks {
                let mut tasks = Vec::<(usize, usize)>::with_capacity(
                    n_slots.saturating_mul(rep_end.saturating_sub(rep_start)),
                );
                for slot in 0..n_slots {
                    for rep in rep_start..rep_end {
                        tasks.push((slot, rep));
                    }
                }
                Some(tasks)
            } else {
                None
            };
            let flat_perm_grain = flat_perm_tasks.as_ref().map(|tasks| {
                garfield_task_chunk_units(
                    tasks.len(),
                    perm_threads,
                    garfield_perm_task_coalesce_max_tasks(),
                )
            });
            let batch_results: Vec<
                Result<Vec<(usize, Vec<GarfieldPermutationNullScores>)>, String>,
            > = if let (Some(pool), Some(tasks), Some(grain)) = (
                perm_pool.as_ref(),
                flat_perm_tasks.as_ref(),
                flat_perm_grain,
            ) {
                pool.install(|| {
                    tasks
                        .par_chunks(grain)
                        .map(|task_chunk| {
                            let task_t0 =
                                scheduler_permutation_tasks.as_ref().map(|_| Instant::now());
                            let task_work_units = if task_t0.is_some() {
                                task_chunk
                                    .iter()
                                    .map(|&(slot, _)| {
                                        null_prepared[slot].1.selected_global_rows.len()
                                    })
                                    .sum()
                            } else {
                                0
                            };
                            let result = process_rule_permutation_task_chunk_flat(
                                task_chunk,
                                null_prepared.as_slice(),
                                permutation_bits_cache.as_deref(),
                                &logic_bits,
                                train_idx_local.as_slice(),
                                test_idx_local.as_slice(),
                                train_fit.residualized_y.as_slice(),
                                test_fit.residualized_y.as_slice(),
                                Some(train_fit.null_residualized_y.as_slice()),
                                split_applied,
                                perm_beam_params.clone(),
                                null_keep_topk,
                                seed,
                                progress_callback_parallel,
                                &null_progress_done,
                                null_notify_step,
                                permutation_task_total,
                                Some(&null_mem_tracker),
                                null_reselection,
                            );
                            if let Some(task_t0) = task_t0 {
                                record_garfield_scheduler_task(
                                    scheduler_permutation_tasks.as_ref(),
                                    task_t0.elapsed().as_secs_f64(),
                                    task_work_units,
                                );
                            }
                            result
                        })
                        .collect()
                })
            } else if let Some(pool) = perm_pool.as_ref() {
                pool.install(|| {
                    slot_indices
                        .par_chunks(slot_grain)
                        .map(|slot_chunk| {
                            let task_t0 =
                                scheduler_permutation_tasks.as_ref().map(|_| Instant::now());
                            let task_work_units = if task_t0.is_some() {
                                slot_chunk
                                    .iter()
                                    .map(|&slot| {
                                        null_prepared[slot]
                                            .1
                                            .selected_global_rows
                                            .len()
                                            .saturating_mul(rep_end.saturating_sub(rep_start))
                                    })
                                    .sum()
                            } else {
                                0
                            };
                            let result = process_rule_permutation_task_chunk(
                                slot_chunk,
                                rep_start,
                                rep_end,
                                null_prepared.as_slice(),
                                permutation_bits_cache.as_deref(),
                                &logic_bits,
                                train_idx_local.as_slice(),
                                test_idx_local.as_slice(),
                                train_fit.residualized_y.as_slice(),
                                test_fit.residualized_y.as_slice(),
                                Some(train_fit.null_residualized_y.as_slice()),
                                split_applied,
                                perm_beam_params.clone(),
                                null_keep_topk,
                                seed,
                                progress_callback_parallel,
                                &null_progress_done,
                                null_notify_step,
                                permutation_task_total,
                                Some(&null_mem_tracker),
                                null_reselection,
                            );
                            if let Some(task_t0) = task_t0 {
                                record_garfield_scheduler_task(
                                    scheduler_permutation_tasks.as_ref(),
                                    task_t0.elapsed().as_secs_f64(),
                                    task_work_units,
                                );
                            }
                            result
                        })
                        .collect()
                })
            } else {
                slot_indices
                    .chunks(slot_grain)
                    .map(|slot_chunk| {
                        let task_t0 = scheduler_permutation_tasks.as_ref().map(|_| Instant::now());
                        let task_work_units = if task_t0.is_some() {
                            slot_chunk
                                .iter()
                                .map(|&slot| {
                                    null_prepared[slot]
                                        .1
                                        .selected_global_rows
                                        .len()
                                        .saturating_mul(rep_end.saturating_sub(rep_start))
                                })
                                .sum()
                        } else {
                            0
                        };
                        let result = process_rule_permutation_task_chunk(
                            slot_chunk,
                            rep_start,
                            rep_end,
                            null_prepared.as_slice(),
                            permutation_bits_cache.as_deref(),
                            &logic_bits,
                            train_idx_local.as_slice(),
                            test_idx_local.as_slice(),
                            train_fit.residualized_y.as_slice(),
                            test_fit.residualized_y.as_slice(),
                            Some(train_fit.null_residualized_y.as_slice()),
                            split_applied,
                            perm_beam_params.clone(),
                            null_keep_topk,
                            seed,
                            progress_callback.as_ref(),
                            &null_progress_done,
                            null_notify_step,
                            permutation_task_total,
                            Some(&null_mem_tracker),
                            null_reselection,
                        );
                        if let Some(task_t0) = task_t0 {
                            record_garfield_scheduler_task(
                                scheduler_permutation_tasks.as_ref(),
                                task_t0.elapsed().as_secs_f64(),
                                task_work_units,
                            );
                        }
                        result
                    })
                    .collect()
            };
            let mut by_rep = vec![Vec::<GarfieldPermutationNullScores>::new(); rep_end - rep_start];
            for chunk_out in batch_results.into_iter() {
                for (rep, vals) in chunk_out? {
                    by_rep[rep - rep_start].extend(vals);
                }
            }
            for (rep_offset, rep_vals) in by_rep.into_iter().enumerate() {
                if let Some(accumulator) = proposal_shared_max_t.as_mut() {
                    accumulator.begin_replicate();
                }
                for vals in rep_vals.iter() {
                    if let Some(accumulator) = proposal_shared_max_t.as_mut() {
                        // Full-sample GARFIELD uses the train statistic as its
                        // primary output.  Keep this reducer max-only and
                        // finite-value aware; test statistics are calibrated
                        // by the existing split-aware lookup when applicable.
                        accumulator.observe_rule(
                            vals.bucket.rule_len,
                            vals.output_train_raw_score,
                            vals.delta_train_score,
                        );
                    }
                    search_bucket_scores.insert(
                        vals.bucket,
                        vals.search_train_score,
                        vals.search_test_score,
                    );
                }
                // Search calibration remains window-local because it controls
                // beam expansion inside each unit.  Final output calibration,
                // however, is for the complete scan: retain one global max per
                // length/group bucket for this repeat so the penalty targets
                // scan-wide null extremes rather than a per-window q99.
                for (bucket, score) in collect_global_max_rule_null_metric_by_group_len_bucket(
                    rep_vals.as_slice(),
                    true,
                ) {
                    output_bucket_scores.insert(bucket, score, f64::NAN);
                }
                for (bucket, score) in collect_global_max_rule_null_metric_by_group_len_bucket(
                    rep_vals.as_slice(),
                    false,
                ) {
                    output_bucket_scores.insert(bucket, f64::NAN, score);
                }
                for (bucket, score) in collect_global_max_rule_null_delta_by_group_len_bucket(
                    rep_vals.as_slice(),
                    true,
                ) {
                    output_delta_bucket_scores.insert(bucket, score, f64::NAN);
                }
                for (bucket, score) in collect_global_max_rule_null_delta_by_group_len_bucket(
                    rep_vals.as_slice(),
                    false,
                ) {
                    output_delta_bucket_scores.insert(bucket, f64::NAN, score);
                }
                if let Some(score) =
                    collect_global_max_rule_null_family_4plus(rep_vals.as_slice(), true)
                {
                    output_bucket_scores.insert_family_4plus(score, f64::NAN);
                }
                if let Some(score) =
                    collect_global_max_rule_null_family_4plus(rep_vals.as_slice(), false)
                {
                    output_bucket_scores.insert_family_4plus(f64::NAN, score);
                }
                if let Some(score) =
                    collect_global_max_rule_null_delta_family_4plus(rep_vals.as_slice(), true)
                {
                    output_delta_bucket_scores.insert_family_4plus(score, f64::NAN);
                }
                if let Some(score) =
                    collect_global_max_rule_null_delta_family_4plus(rep_vals.as_slice(), false)
                {
                    output_delta_bucket_scores.insert_family_4plus(f64::NAN, score);
                }
                if let Some(accumulator) = proposal_shared_max_t.as_mut() {
                    accumulator.finish_replicate();
                }
                permutation_null_repeats_used = rep_start + rep_offset + 1;
                if permutation_null_repeats_used < min_perm_repeats {
                    continue;
                }
                let current_search_lookup =
                    search_bucket_scores.finalize_with_method(rule_null_penalty_method);
                let mut current_output_lookup =
                    output_bucket_scores.finalize_with_method(rule_null_penalty_method);
                let output_delta_lookup =
                    output_delta_bucket_scores.finalize_with_method(rule_null_penalty_method);
                current_output_lookup.attach_delta_penalties_from(&output_delta_lookup);
                if let (Some(prev_search), Some(prev_output)) =
                    (prev_search_lookup.as_ref(), prev_output_lookup.as_ref())
                {
                    let search_stable = current_search_lookup
                        .penalty_converged_against(prev_search)
                        || (!current_search_lookup.has_signal() && !prev_search.has_signal());
                    let output_stable = current_output_lookup
                        .penalty_converged_against(prev_output)
                        || (!current_output_lookup.has_signal() && !prev_output.has_signal());
                    if search_stable && output_stable {
                        stable_rounds += 1;
                    } else {
                        stable_rounds = 0;
                    }
                }
                prev_search_lookup = Some(current_search_lookup);
                prev_output_lookup = Some(current_output_lookup);
                if rule_null_penalty_allows_adaptive_early_stop(rule_null_penalty_method)
                    && stable_rounds >= DEFAULT_RULE_NULL_ADAPTIVE_STABLE_REPEATS
                {
                    break 'perm_batches;
                }
            }
            rep_start = rep_end;
        }
        if null_progress_done.load(Ordering::Relaxed) < permutation_task_total {
            garfield_stage_progress_notify(
                progress_callback.as_ref(),
                "null_penalty",
                permutation_task_total.max(1),
                permutation_task_total.max(1),
                None,
            )
            .map_err(|e| e.to_string())?;
        }
        if let Some(debug) = memory_debug.as_mut() {
            debug.null_penalty = null_mem_tracker.finish_stage(null_mem_start);
        }
        if let Some(accumulator) = proposal_shared_max_t.as_ref() {
            let singleton_n = accumulator.values(ProposalFamily::Singleton, false).len();
            let two_plus_n = accumulator.values(ProposalFamily::TwoPlus, false).len();
            let three_plus_n = accumulator.values(ProposalFamily::ThreePlus, false).len();
            eprintln!(
                "[GARFIELD-PROPOSAL] shared_inner_maxT ready: n={{1:{singleton_n},2+:{two_plus_n},3+:{three_plus_n}}} q99={{1:{},2+:{},3+:{}}} delta_q99={{1:{},2+:{},3+:{}}}",
                format_optional_score(accumulator.q99(ProposalFamily::Singleton, false)),
                format_optional_score(accumulator.q99(ProposalFamily::TwoPlus, false)),
                format_optional_score(accumulator.q99(ProposalFamily::ThreePlus, false)),
                format_optional_score(accumulator.q99(ProposalFamily::Singleton, true)),
                format_optional_score(accumulator.q99(ProposalFamily::TwoPlus, true)),
                format_optional_score(accumulator.q99(ProposalFamily::ThreePlus, true)),
            );
        }
        let search_lookup = prev_search_lookup
            .unwrap_or_else(|| search_bucket_scores.finalize_with_method(rule_null_penalty_method));
        let mut output_lookup = prev_output_lookup
            .unwrap_or_else(|| output_bucket_scores.finalize_with_method(rule_null_penalty_method));
        let output_delta_lookup =
            output_delta_bucket_scores.finalize_with_method(rule_null_penalty_method);
        output_lookup.attach_delta_penalties_from(&output_delta_lookup);
        (Some(Arc::new(search_lookup)), Some(Arc::new(output_lookup)))
    } else {
        (None, None)
    };
    let bg_noise_summary = match (
        rule_search_null_lookup.as_deref(),
        rule_output_null_lookup.as_deref(),
    ) {
        (Some(search_lookup), Some(output_lookup)) => {
            let active_is_train = !split_applied;
            let len_bucket_count = search_lookup
                .len_bucket_count()
                .max(output_lookup.len_bucket_count());
            let mut buckets = Vec::<GarfieldBgNoiseBucketSummary>::new();
            if grouped_null_mode {
                for unit_group_bin in 0..search_lookup
                    .unit_group_bin_count()
                    .max(output_lookup.unit_group_bin_count())
                {
                    for idx in 0..len_bucket_count {
                        let search = search_lookup.group_len_bucket_summary_by_index(
                            unit_group_bin as u8,
                            idx,
                            active_is_train,
                        );
                        let output = output_lookup.group_len_bucket_summary_by_index(
                            unit_group_bin as u8,
                            idx,
                            active_is_train,
                        );
                        if search.is_none() && output.is_none() {
                            continue;
                        }
                        buckets.push(GarfieldBgNoiseBucketSummary {
                            label: format!(
                                "{} {}",
                                rule_null_unit_group_label(unit_group_bin),
                                rule_null_layer_label(idx)
                            ),
                            search,
                            output,
                        });
                    }
                }
            } else {
                buckets.reserve(len_bucket_count);
                for idx in 0..len_bucket_count {
                    buckets.push(GarfieldBgNoiseBucketSummary {
                        label: rule_null_layer_label(idx).to_string(),
                        search: search_lookup.len_bucket_summary_by_index(idx, active_is_train),
                        output: output_lookup.len_bucket_summary_by_index(idx, active_is_train),
                    });
                }
            }
            Some(GarfieldBgNoiseSummary {
                dataset: if split_applied {
                    "test".to_string()
                } else {
                    "full".to_string()
                },
                buckets,
                output_delta_penalties_train: (1..=5)
                    .map(|rule_len| output_lookup.delta_penalty(rule_len, true))
                    .collect(),
                output_delta_penalties_test: (1..=5)
                    .map(|rule_len| output_lookup.delta_penalty(rule_len, false))
                    .collect(),
                output_family_4plus_train: output_lookup.four_plus_family_penalty(true),
                output_family_4plus_test: output_lookup.four_plus_family_penalty(false),
                output_delta_family_4plus_train: output_lookup.delta_four_plus_family_penalty(true),
                output_delta_family_4plus_test: output_lookup.delta_four_plus_family_penalty(false),
            })
        }
        _ => None,
    };
    let mut structure_bootstrap_repeats_used_total = 0usize;
    let rule_structure_prior: Option<Arc<RuleStructurePrior>> = if soft_structure_mode {
        let structure_mem_tracker = GarfieldStageMemoryTracker::new(rss_debug_enabled);
        let structure_mem_start = structure_mem_tracker.start_stage();
        let structure_notify_step = if progress_every == 0 {
            (structure_task_total.max(1) / 200).max(1)
        } else {
            progress_every.max(1)
        };
        let structure_progress_done = AtomicUsize::new(0);
        let structure_scores = Arc::new(Mutex::new(RuleStructurePriorCalibrator::default()));
        let structure_repeats_used = AtomicUsize::new(0);
        garfield_stage_progress_notify(
            progress_callback.as_ref(),
            "structure_prior",
            0,
            structure_task_total.max(1),
            Some(format_structure_alpha_placeholder(structure_prior_display_len).as_str()),
        )
        .map_err(|e| e.to_string())?;
        let posterior_beam_params = BeamSearchParams {
            allow_parallel: false,
            null_penalties: rule_search_null_lookup.clone(),
            structure_prior: None,
            ..beam_params.clone()
        };
        let bootstrap_threads = threads_eff.min(representative_prepared.len().max(1));
        let structure_chunk_units = garfield_task_chunk_units(
            representative_prepared.len(),
            bootstrap_threads,
            garfield_structure_task_coalesce_max_units(),
        );
        if bootstrap_threads > 1 {
            let pool = ThreadPoolBuilder::new()
                .num_threads(bootstrap_threads)
                .build()
                .map_err(|e| format!("build GARFIELD bootstrap thread pool: {e}"))?;
            pool.install(|| {
                representative_prepared
                    .par_chunks(structure_chunk_units)
                    .enumerate()
                    .map(|(chunk_idx, chunk)| {
                        process_structure_prior_unit_chunk(
                            chunk_idx.saturating_mul(structure_chunk_units),
                            chunk,
                            &logic_bits,
                            train_idx_local.as_slice(),
                            test_idx_local.as_slice(),
                            train_fit.residualized_y.as_slice(),
                            test_fit.residualized_y.as_slice(),
                            split_applied,
                            posterior_beam_params.clone(),
                            &structure_prior_cfg,
                            seed,
                            &structure_scores,
                            &structure_repeats_used,
                            &structure_progress_done,
                            structure_notify_step,
                            structure_task_total,
                            structure_prior_display_len,
                            progress_callback_parallel,
                            Some(&structure_mem_tracker),
                        )
                    })
                    .collect::<Vec<Result<(), String>>>()
            })
        } else {
            for (chunk_idx, chunk) in representative_prepared
                .chunks(structure_chunk_units)
                .enumerate()
            {
                process_structure_prior_unit_chunk(
                    chunk_idx.saturating_mul(structure_chunk_units),
                    chunk,
                    &logic_bits,
                    train_idx_local.as_slice(),
                    test_idx_local.as_slice(),
                    train_fit.residualized_y.as_slice(),
                    test_fit.residualized_y.as_slice(),
                    split_applied,
                    posterior_beam_params.clone(),
                    &structure_prior_cfg,
                    seed,
                    &structure_scores,
                    &structure_repeats_used,
                    &structure_progress_done,
                    structure_notify_step,
                    structure_task_total,
                    structure_prior_display_len,
                    progress_callback.as_ref(),
                    Some(&structure_mem_tracker),
                )?;
            }
            Vec::<Result<(), String>>::new()
        }
        .into_iter()
        .try_for_each(|res| res)?;
        if let Some(debug) = memory_debug.as_mut() {
            debug.structure_prior = structure_mem_tracker.finish_stage(structure_mem_start);
        }
        structure_bootstrap_repeats_used_total = structure_repeats_used.load(Ordering::Relaxed);
        let final_structure_scores = structure_scores
            .lock()
            .map_err(|_| "GARFIELD structure prior mutex poisoned".to_string())?
            .clone();
        Some(Arc::new(
            final_structure_scores.finalize(&structure_prior_cfg),
        ))
    } else {
        None
    };
    let beam_params = BeamSearchParams {
        null_penalties: rule_search_null_lookup.clone(),
        structure_prior: rule_structure_prior.clone(),
        disable_parent_delta: soft_structure_mode,
        min_gain: if no_clean {
            0.0
        } else {
            GARFIELD_SCAN_MIN_GAIN_EPS
        },
        min_parent_abs_gain: 0.0,
        surrogate_test_gain_max: if no_clean {
            0.0
        } else {
            GARFIELD_SCAN_SURROGATE_TEST_GAIN_MAX
        },
        surrogate_hamming_frac_max: if no_clean {
            0.0
        } else {
            GARFIELD_SCAN_SURROGATE_HAMMING_FRAC_MAX
        },
        ..beam_params
    };

    let scan_notify_step = if progress_every == 0 {
        (scanned_units.max(1) / 200).max(1)
    } else {
        progress_every.max(1)
    };
    reset_garfield_beam_profile();
    reset_pairwise_profile();
    PACKED_EXTRACT_FLAT_NS.store(0, Ordering::Relaxed);
    GARFIELD_DENSE_DOSAGE_DECODE_NS.store(0, Ordering::Relaxed);
    GARFIELD_LD_CLUMP_NS.store(0, Ordering::Relaxed);
    GARFIELD_LD_CLUMP_EXACT_PAIRS.store(0, Ordering::Relaxed);
    GARFIELD_LD_CLUMP_ROWS_TOTAL.store(0, Ordering::Relaxed);
    GARFIELD_LD_CLUMP_ROWS_KEPT.store(0, Ordering::Relaxed);
    GARFIELD_LD_CLUMP_ROWS_COMPRESSED.store(0, Ordering::Relaxed);
    GARFIELD_LD_CLUMP_UNITS.store(0, Ordering::Relaxed);
    garfield_ld_unit_stats_reset();
    GARFIELD_MATERIALIZE_BITS_NS.store(0, Ordering::Relaxed);
    GARFIELD_ML_SELECT_NS.store(0, Ordering::Relaxed);
    GARFIELD_CORR_STAGE1_SUMMARY_NS.store(0, Ordering::Relaxed);
    GARFIELD_CORR_STAGE1_SCORE_NS.store(0, Ordering::Relaxed);
    GARFIELD_CORR_STAGE1_POOL_NS.store(0, Ordering::Relaxed);
    GARFIELD_CORR_STAGE1_POOL_PRIORITY_NS.store(0, Ordering::Relaxed);
    GARFIELD_CORR_STAGE1_POOL_LD_NS.store(0, Ordering::Relaxed);
    let scan_stage_t0 = Instant::now();
    let scan_progress_done = AtomicUsize::new(0);
    garfield_stage_progress_notify(
        progress_callback.as_ref(),
        "scan",
        0,
        scanned_units.max(1),
        None,
    )
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
    let unit_parallel_threads = threads_eff.max(1);
    let scan_mem_tracker = GarfieldStageMemoryTracker::new(rss_debug_enabled);
    let scan_mem_start = scan_mem_tracker.start_stage();
    let scan_stage1_y_lookup = Arc::new(PackedYSumLookup::build(
        train_fit.residualized_y.as_slice(),
        train_idx_local.len(),
    )?);
    let scan_unit_chunks = garfield_scan_task_chunks(
        scan_unit_indices.as_slice(),
        units.as_slice(),
        None,
        unit_parallel_threads,
        garfield_scan_task_coalesce_max_units(),
    );
    // A window scan already has many independent units in the outer pool.
    // Nested row/beam Rayon loops make each unit repeatedly split tiny jobs
    // and compete with sibling units, which lowers sustained CPU utilization.
    // Base the decision on coalesced task count rather than raw unit count:
    // small scans may be one task even when they contain dozens of windows.
    let allow_nested_scan_parallel =
        unit_parallel_threads <= 1 || scan_unit_chunks.len() < unit_parallel_threads;
    let scan_beam_params = BeamSearchParams {
        allow_parallel: allow_nested_scan_parallel,
        y_sum_lookup: Some(scan_stage1_y_lookup.clone()),
        ..beam_params.clone()
    };
    let skipped_units = Arc::new(Mutex::new(Vec::<GarfieldSkippedUnitInfo>::new()));
    let unit_results = if unit_parallel_threads > 1 {
        let pool = ThreadPoolBuilder::new()
            .num_threads(unit_parallel_threads)
            .build()
            .map_err(|e| format!("build GARFIELD unit thread pool: {e}"))?;
        let skipped_units_parallel = skipped_units.clone();
        pool.install(|| {
            scan_unit_chunks
                .par_iter()
                .map(|chunk| {
                    let task_t0 = scheduler_scan_tasks.as_ref().map(|_| Instant::now());
                    let task_work_units = if task_t0.is_some() {
                        chunk.iter().map(|&ui| units[ui].indices.len()).sum()
                    } else {
                        0
                    };
                    let mut chunk_out =
                        Vec::<Result<GarfieldUnitEvaluationOutput, String>>::with_capacity(
                            chunk.len(),
                        );
                    for &ui in chunk.iter() {
                        chunk_out.push(process_scan_unit_continuous(
                            ui,
                            units.as_slice(),
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
                            Some(scan_stage1_y_lookup.as_ref()),
                            test_fit.residualized_y.as_slice(),
                            scan_beam_params.clone(),
                            rule_output_null_lookup.clone(),
                            top_rules_per_unit,
                            raw_design,
                            &unit_kind_lc,
                            progress_callback_parallel,
                            &scan_progress_done,
                            scan_notify_step,
                            scanned_units,
                            &skipped_units_parallel,
                            Some(&scan_mem_tracker),
                            debug_probe.as_ref(),
                            ld_support_cache.as_deref(),
                            diagnostics,
                        ));
                    }
                    if let Some(task_t0) = task_t0 {
                        record_garfield_scheduler_task(
                            scheduler_scan_tasks.as_ref(),
                            task_t0.elapsed().as_secs_f64(),
                            task_work_units,
                        );
                    }
                    chunk_out
                })
                .collect::<Vec<Vec<Result<GarfieldUnitEvaluationOutput, String>>>>()
                .into_iter()
                .flatten()
                .collect::<Vec<Result<GarfieldUnitEvaluationOutput, String>>>()
        })
    } else {
        let mut out = Vec::<Result<GarfieldUnitEvaluationOutput, String>>::with_capacity(
            scan_unit_indices.len(),
        );
        let task_t0 = scheduler_scan_tasks.as_ref().map(|_| Instant::now());
        let task_work_units = if task_t0.is_some() {
            scan_unit_indices
                .iter()
                .map(|&ui| units[ui].indices.len())
                .sum()
        } else {
            0
        };
        for &ui in scan_unit_indices.iter() {
            let chunk_out = process_scan_unit_continuous(
                ui,
                units.as_slice(),
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
                Some(scan_stage1_y_lookup.as_ref()),
                test_fit.residualized_y.as_slice(),
                scan_beam_params.clone(),
                rule_output_null_lookup.clone(),
                top_rules_per_unit,
                raw_design,
                &unit_kind_lc,
                progress_callback.as_ref(),
                &scan_progress_done,
                scan_notify_step,
                scanned_units,
                &skipped_units,
                Some(&scan_mem_tracker),
                debug_probe.as_ref(),
                ld_support_cache.as_deref(),
                diagnostics,
            );
            out.push(chunk_out);
        }
        if let Some(task_t0) = task_t0 {
            record_garfield_scheduler_task(
                scheduler_scan_tasks.as_ref(),
                task_t0.elapsed().as_secs_f64(),
                task_work_units,
            );
        }
        out
    };
    let mut records = Vec::<GarfieldLogicRuleRecord>::new();
    let mut diagnostics_records = Vec::<GarfieldRecallDiagnosticRecord>::new();
    let mut frontier_diagnostics_records = Vec::<GarfieldFrontierDiagnosticRecord>::new();
    for unit_out in unit_results.into_iter() {
        let unit_out = unit_out?;
        records.extend(unit_out.records);
        diagnostics_records.extend(unit_out.diagnostics);
        frontier_diagnostics_records.extend(unit_out.frontier_diagnostics);
    }
    let skipped_units = Arc::try_unwrap(skipped_units)
        .map_err(|_| "GARFIELD skipped-units still shared".to_string())?
        .into_inner()
        .map_err(|_| "GARFIELD skipped-units mutex poisoned".to_string())?;
    let skipped_messages = skipped_units
        .iter()
        .map(format_skipped_unit_message)
        .collect::<Vec<_>>();
    if let Some(debug) = memory_debug.as_mut() {
        debug.scan = scan_mem_tracker.finish_stage(scan_mem_start);
    }
    let scan_stage_wall_s = scan_stage_t0.elapsed().as_secs_f64();
    let ld_unit_stats = garfield_ld_unit_stats_snapshot();
    let timing_scan_ml_select_wall_s =
        (GARFIELD_ML_SELECT_NS.load(Ordering::Relaxed) as f64) * 1e-9;
    let timing_corr_stage1_summary_s =
        (GARFIELD_CORR_STAGE1_SUMMARY_NS.load(Ordering::Relaxed) as f64) * 1e-9;
    let timing_corr_stage1_score_s =
        (GARFIELD_CORR_STAGE1_SCORE_NS.load(Ordering::Relaxed) as f64) * 1e-9;
    let timing_corr_stage1_pool_s =
        (GARFIELD_CORR_STAGE1_POOL_NS.load(Ordering::Relaxed) as f64) * 1e-9;
    let timing_corr_stage1_pool_priority_s =
        (GARFIELD_CORR_STAGE1_POOL_PRIORITY_NS.load(Ordering::Relaxed) as f64) * 1e-9;
    let timing_corr_stage1_pool_ld_s =
        (GARFIELD_CORR_STAGE1_POOL_LD_NS.load(Ordering::Relaxed) as f64) * 1e-9;
    let scan_beam_profile = snapshot_garfield_beam_profile();
    let (pw_marg, pw_pack, pw_kern, pw_comb) = snapshot_pairwise_profile();

    records = dedup_logic_rule_records(records, &logic_bits)?;
    apply_logic_rule_output_limit(&mut records, max_output_rules, max_output_ratio)?;
    let simbench_count = if let Some(path) = simbench_path.as_ref() {
        let simbench_terms = parse_simbench_terms(path)?;
        let mut logic_site_lookup =
            HashMap::<(String, i32), Vec<usize>>::with_capacity(logic_bits.sites.len());
        for (logic_idx, site) in logic_bits.sites.iter().enumerate() {
            logic_site_lookup
                .entry((normalize_chrom(site.garfield_chrom()), site.garfield_pos()))
                .or_default()
                .push(logic_idx);
        }
        let simbench_ml_contexts = build_simbench_ml_contexts(
            simbench_terms.as_slice(),
            units.as_slice(),
            scan_unit_indices.as_slice(),
            &unit_kind_lc,
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
            scan_beam_params.clone(),
            unit_parallel_threads <= 1,
        )?;
        let simbench_records = evaluate_simbench_terms(
            simbench_terms.as_slice(),
            &logic_bits,
            &logic_site_lookup,
            train_idx_local.as_slice(),
            assoc_sample_indices,
            train_fit.residualized_y.as_slice(),
            assoc_y,
            beam_params,
            rule_output_null_lookup.clone(),
            simbench_ml_contexts.as_slice(),
        )?;
        let n = simbench_records.len();
        records.extend(simbench_records);
        n
    } else {
        0usize
    };
    if rule_null_report_pvalue {
        maybe_attach_logic_rule_permutation_significance(
            &mut records,
            rule_output_null_lookup.as_deref(),
        );
    }
    let total_wall_s = total_wall_t0.elapsed().as_secs_f64();
    let timing_scan_beam_wall_s = scan_beam_profile.total_s;
    let timing_scan_literal_score_wall_s = scan_beam_profile.literal_precompute_s;
    let timing_literal_score_share_of_total_pct =
        timing_share_pct(timing_scan_literal_score_wall_s, total_wall_s);
    let timing_literal_score_share_of_scan_pct =
        timing_share_pct(timing_scan_literal_score_wall_s, scan_stage_wall_s);
    let timing_literal_score_share_of_beam_pct =
        timing_share_pct(timing_scan_literal_score_wall_s, timing_scan_beam_wall_s);
    let timing_beam_share_of_total_pct = timing_share_pct(timing_scan_beam_wall_s, total_wall_s);
    let timing_beam_share_of_scan_pct =
        timing_share_pct(timing_scan_beam_wall_s, scan_stage_wall_s);
    let scheduler_scan = summarize_garfield_scheduler_task_collector(scheduler_scan_tasks.as_ref());
    let scheduler_permutation =
        summarize_garfield_scheduler_task_collector(scheduler_permutation_tasks.as_ref());

    let structure_prior_for_output = rule_structure_prior.clone();
    let mut pseudo_prefix_out = None;
    let mut rules_tsv_out = None;
    let mut diagnostics_tsv_out = None;
    let mut frontier_diagnostics_tsv_out = None;
    let rules_compare_tsv_out = None;
    let mut posterior_json_out = None;
    if let Some(prefix_out) = out_prefix.as_ref() {
        write_logic_pseudo_plink(
            prefix_out,
            logic_bits.sample_ids.as_slice(),
            records.as_slice(),
            &logic_bits,
        )?;
        let rules_tsv = format!("{prefix_out}.rules.tsv");
        write_logic_rules_tsv(&rules_tsv, records.as_slice())?;
        if diagnostics {
            let diagnostics_tsv = format!("{prefix_out}.diagnostics.tsv");
            write_garfield_recall_diagnostics(&diagnostics_tsv, diagnostics_records.as_slice())?;
            diagnostics_tsv_out = Some(diagnostics_tsv);
            if !frontier_diagnostics_records.is_empty() {
                let frontier_tsv = format!("{prefix_out}.frontier.tsv");
                write_garfield_frontier_diagnostics(
                    &frontier_tsv,
                    frontier_diagnostics_records.as_slice(),
                )?;
                frontier_diagnostics_tsv_out = Some(frontier_tsv);
            }
        }
        if let Some(prior) = structure_prior_for_output.as_deref() {
            let posterior_json = format!("{prefix_out}.posterior.json");
            write_rule_structure_prior_json(&posterior_json, prior)?;
            posterior_json_out = Some(posterior_json);
        }
        pseudo_prefix_out = Some(prefix_out.clone());
        rules_tsv_out = Some(rules_tsv);
    }

    Ok(GarfieldLogicPipelineResult {
        pseudo_prefix: pseudo_prefix_out,
        rules_tsv: rules_tsv_out,
        diagnostics_tsv: diagnostics_tsv_out,
        frontier_diagnostics_tsv: frontier_diagnostics_tsv_out,
        rules_compare_tsv: rules_compare_tsv_out,
        posterior_json: posterior_json_out,
        memory_debug,
        bg_noise_summary,
        scheduler_scan,
        scheduler_permutation,
        rule_permutation_active: null_permutation_active || soft_structure_mode,
        null_chunk_bp,
        null_chunk_min_snps: DEFAULT_RULE_NULL_MIN_SNPS_PER_CHUNK,
        null_chunk_target,
        null_chunk_valid_total,
        null_chunk_selected,
        representative_units_target,
        representative_units_used,
        permutation_null_repeats: if null_permutation_active {
            permutation_null_repeats_used
        } else {
            0
        },
        permutation_bootstrap_repeats: if soft_structure_mode {
            structure_bootstrap_repeats_used_total
        } else {
            0
        },
        records,
        simbench_count,
        n_samples: logic_bits.n_samples,
        logic_bits_backend,
        logic_bits_bytes,
        units_total: total_units,
        units_scanned: scanned_units,
        timing_total_wall_s: total_wall_s,
        timing_scan_wall_s: scan_stage_wall_s,
        max6_observed_scan_timing: Max6ObservedScanTiming::default(),
        timing_max6_screen_calibration_wall_s: 0.0,
        timing_max6_screen_validation_wall_s: 0.0,
        timing_max6_screen_sampled_diagnostic_wall_s: 0.0,
        timing_max6_screen_tail_audit_wall_s: 0.0,
        timing_max6_screen_exceedance_audit_wall_s: 0.0,
        timing_max6_screen_pot_audit_wall_s: 0.0,
        timing_scan_ml_select_wall_s,
        timing_scan_beam_wall_s,
        timing_scan_literal_score_wall_s,
        timing_corr_stage1_summary_s,
        timing_corr_stage1_score_s,
        timing_corr_stage1_pool_s,
        timing_corr_stage1_pool_priority_s,
        timing_corr_stage1_pool_ld_s,
        timing_ld_support_cache_s,
        ld_support_cache_rows: ld_support_cache_rows_count,
        ld_support_cache_payload_bytes,
        timing_scan_beam_calls: scan_beam_profile.calls,
        timing_clone_bits_s: scan_beam_profile.clone_bits_s,
        timing_sum_y_both1_s: scan_beam_profile.sum_y_both1_s,
        timing_parent_baseline_s: scan_beam_profile.parent_baseline_s,
        timing_pw_marginal_s: pw_marg,
        timing_pw_pack_s: pw_pack,
        timing_pw_kernel_s: pw_kern,
        timing_pw_combine_s: pw_comb,
        timing_dense_extract_s: (PACKED_EXTRACT_FLAT_NS.load(Ordering::Relaxed) as f64) * 1e-9,
        timing_dense_decode_s: (GARFIELD_DENSE_DOSAGE_DECODE_NS.load(Ordering::Relaxed) as f64)
            * 1e-9,
        timing_geneset_ld_prune_s: (GARFIELD_LD_CLUMP_NS.load(Ordering::Relaxed) as f64) * 1e-9,
        ld_exact_pairs: GARFIELD_LD_CLUMP_EXACT_PAIRS.load(Ordering::Relaxed),
        ld_rows_total: GARFIELD_LD_CLUMP_ROWS_TOTAL.load(Ordering::Relaxed),
        ld_rows_kept: GARFIELD_LD_CLUMP_ROWS_KEPT.load(Ordering::Relaxed),
        ld_clump_rows_compressed: GARFIELD_LD_CLUMP_ROWS_COMPRESSED.load(Ordering::Relaxed),
        ld_clump_units: GARFIELD_LD_CLUMP_UNITS.load(Ordering::Relaxed),
        ld_units_eligible: ld_unit_stats.ld_units_eligible,
        ld_units_pruned: ld_unit_stats.ld_units_pruned,
        ld_rows_pruned: ld_unit_stats.ld_rows_pruned,
        ld_unit_rows_total_median: ld_unit_stats.ld_unit_rows_total_median,
        ld_unit_rows_total_p95: ld_unit_stats.ld_unit_rows_total_p95,
        ld_unit_rows_total_max: ld_unit_stats.ld_unit_rows_total_max,
        ld_unit_rows_kept_median: ld_unit_stats.ld_unit_rows_kept_median,
        ld_unit_rows_kept_p95: ld_unit_stats.ld_unit_rows_kept_p95,
        ld_unit_rows_kept_max: ld_unit_stats.ld_unit_rows_kept_max,
        ld_unit_rows_pruned_median: ld_unit_stats.ld_unit_rows_pruned_median,
        ld_unit_rows_pruned_p95: ld_unit_stats.ld_unit_rows_pruned_p95,
        ld_unit_rows_pruned_max: ld_unit_stats.ld_unit_rows_pruned_max,
        ld_unit_exact_pairs_median: ld_unit_stats.ld_unit_exact_pairs_median,
        ld_unit_exact_pairs_p95: ld_unit_stats.ld_unit_exact_pairs_p95,
        ld_unit_exact_pairs_max: ld_unit_stats.ld_unit_exact_pairs_max,
        timing_materialize_bits_s: (GARFIELD_MATERIALIZE_BITS_NS.load(Ordering::Relaxed) as f64)
            * 1e-9,
        timing_literal_score_share_of_total_pct,
        timing_literal_score_share_of_scan_pct,
        timing_literal_score_share_of_beam_pct,
        timing_beam_share_of_total_pct,
        timing_beam_share_of_scan_pct,
        skipped_units,
        skipped_messages,
        train_fit,
        max6_pairs_tsv: None,
        max6_pairs_provenance_tsv: None,
        max6_ld_tags_tsv: None,
        max6_threshold_tsv: None,
        max6_screen_maxima_tsv: None,
        max6_screen_validation_maxima_tsv: None,
        max6_screen_sampled_maxima_tsv: None,
        max6_screen_sample_manifest_json: None,
        max6_screen_sample_strata_tsv: None,
        max6_screen_full_maximizer_geometry_tsv: None,
        max6_screen_tail_top_pairs_tsv: None,
        max6_screen_tail_sampled_ranks_tsv: None,
        max6_screen_tail_summary_json: None,
        max6_screen_tail_support_aware_maxima_tsv: None,
        max6_screen_tail_support_aware_strata_tsv: None,
        max6_screen_tail_support_aware_summary_json: None,
        max6_screen_exceedance_audit_per_null_tsv: None,
        max6_screen_exceedance_audit_summary_tsv: None,
        max6_screen_exceedance_audit_top_pairs_tsv: None,
        max6_screen_exceedance_audit_manifest_json: None,
        max6_screen_pot_audit_manifest_json: None,
        max6_screen_pot_sample_scores_manifest_json: None,
        max6_permutation_maxima_tsv: None,
        max6_pair_pve_total_quadratic: None,
        max6_pair_universe_size: 0,
        max6_bonferroni_q_cut: None,
        max6_bonferroni_p_cut: None,
        max6_score_threshold: None,
        max6_gumbel_score_threshold: None,
        max6_screen_threshold: None,
        max6_screen_threshold_method: None,
        max6_screen_null_repeats: 0,
        max6_screen_validation_repeats: 0,
        max6_screen_pair_universe_size: 0,
        max6_screen_window_pair_occurrences: 0,
        max6_screen_candidate_count: 0,
        max6_screen_validation_rejections: 0,
        max6_screen_validation_fwer: None,
        max6_screen_validation_ci_lower: None,
        max6_screen_validation_ci_upper: None,
        max6_final_screen_subset_audit_passed: None,
        max6_final_screen_subset_audit_pairs_checked: 0,
        max6_final_screen_subset_audit_violations: 0,
        max6_score_scale: None,
        max6_permutation_score_count: 0,
        max6_permutation_pair_pool_size: 0,
        max6_permutation_pairs_per_repeat: 0,
        max6_permutation_pair_draw_count: 0,
        max6_permutation_timing: Max6PermutationTiming::default(),
        max6_pair_count: 0,
    })
}
