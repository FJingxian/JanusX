use super::*;

pub(super) fn run_max6_grmmax6_pipeline(
    total_wall_t0: Instant,
    y: &[f64],
    p_c: Option<&[f64]>,
    no_grm_projector: Option<&NoGrmCovariateProjector>,
    train_fit: Arc<GarfieldResidualResult>,
    logic_bits: &GarfieldLogicBits,
    units: &[GarfieldLogicUnit],
    scan_unit_indices: &[usize],
    unit_kind_lc: &str,
    response: ResponseKind,
    engine: Option<MlEngine>,
    importance: ImportanceKind,
    perm_cfg: PermutationConfig,
    ml_top_k: usize,
    ml_top_frac: f64,
    tree_cfg: ExtraTreesConfig,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    beam_params: BeamSearchParams,
    threads: usize,
    final_top_k: usize,
    min_cell: usize,
    min_cell_fraction: f64,
    grm_int_only: bool,
    permutation_repeats: usize,
    rule_permutation: bool,
    rule_null_quantile: f64,
    diagnostics_enabled: bool,
    sampled_screen_null_diagnostic: bool,
    max6_screen_exceedance_audit: bool,
    max6_screen_pot_audit: bool,
    sampled_screen_pair_seeds: &[u64],
    sampled_screen_maf_cutpoints: &str,
    sampled_screen_r2_cutpoints: &str,
    max6_screen_tail_audit: bool,
    max6_screen_tail_audit_repeats: usize,
    max6_screen_tail_pair_seed: u64,
    max6_screen_tail_support_aware: bool,
    max6_screen_tail_support_shares: Vec<usize>,
    ld_tag_exhaustive: bool,
    prepared_max6_universe: Option<&Max6PreparedGenotypeUniverse>,
    prepared_screen_universe: Option<Max6ScreenPreparedUniverse>,
    seed: u64,
    out_prefix: Option<&str>,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
    ld_tag_r2: f64,
    write_provenance: bool,
) -> Result<GarfieldLogicPipelineResult, String> {
    let min_cell = min_cell::resolve_min_cell(min_cell, min_cell_fraction, logic_bits.n_samples)?;
    let final_top_k = final_top_k.max(1);
    let screen_top_k = max6_grmmax6_screen_top_k(grm_int_only, final_top_k);
    let threads = threads.max(1);
    let screen_pool = if threads > 1 {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("janusx-max6-null-{index}"))
                .build()
                .map_err(|error| format!("failed to build Max6 screen pool: {error}"))?,
        )
    } else {
        None
    };
    match (p_c, no_grm_projector) {
        (Some(p_c), None) => validate_grm_int_dense_projection(logic_bits.n_samples, p_c)?,
        (None, Some(projector)) if projector.n_samples() == logic_bits.n_samples => {}
        (None, Some(projector)) => {
            return Err(format!(
                "Max6 -> GRM-Max6 no-GRM projector sample count {} != n_samples {}",
                projector.n_samples(),
                logic_bits.n_samples
            ));
        }
        (Some(_), Some(_)) => {
            return Err(
                "Max6 -> GRM-Max6 received both dense P_C and no-GRM projector".to_string(),
            );
        }
        (None, None) => {
            return Err(
                "Max6 -> GRM-Max6 requires either dense P_C or no-GRM projector".to_string(),
            );
        }
    }
    if y.len() != logic_bits.n_samples {
        return Err(format!(
            "Max6 -> GRM-Max6 phenotype length {} != n_samples {}",
            y.len(),
            logic_bits.n_samples
        ));
    }
    if max6_screen_tail_audit {
        if !ld_tag_exhaustive || !rule_permutation {
            return Err(
                "Max6 tail audit requires the max6-grmmax6 backend and fitted parametric null model"
                    .to_string(),
            );
        }
        if train_fit.tail_audit_residualized_y.len() != max6_screen_tail_audit_repeats {
            return Err(format!(
                "Max6 tail audit requested {} null draws but residual fit supplied {}",
                max6_screen_tail_audit_repeats,
                train_fit.tail_audit_residualized_y.len()
            ));
        }
        if out_prefix.is_none() {
            return Err("Max6 tail audit requires an output prefix".to_string());
        }
    }
    let score_scale = if train_fit.no_grm_boundary {
        if !(train_fit.sigma_e2.is_finite() && train_fit.sigma_e2 > 0.0) {
            return Err(format!(
                "Max6 -> GRM-Max6 no-GRM boundary requires positive finite sigma_e2, got {}",
                train_fit.sigma_e2
            ));
        }
        1.0 / train_fit.sigma_e2
    } else {
        if !(train_fit.sigma_g2.is_finite() && train_fit.sigma_g2 > 0.0) {
            return Err(format!(
                "Max6 -> GRM-Max6 requires positive finite sigma_g2, got {}",
                train_fit.sigma_g2
            ));
        }
        1.0 / train_fit.sigma_g2
    };
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err(format!(
            "Max6 -> GRM-Max6 score scale is invalid: sigma_g2={}, sigma_e2={}, scale={score_scale}",
            train_fit.sigma_g2,
            train_fit.sigma_e2
        ));
    }
    let projected_y = if let Some(p_c) = p_c {
        project_phenotype_for_candidate_rerank(p_c, logic_bits.n_samples, y)
    } else {
        no_grm_projector
            .expect("validated no-GRM projector")
            .project(y)?
    };
    let scaled_total_quadratic = score_scale
        * y.iter()
            .zip(projected_y.iter())
            .map(|(&value, &projected)| value * projected)
            .sum::<f64>();
    let max6_pair_pve_total_quadratic = scaled_total_quadratic
        .is_finite()
        .then_some(scaled_total_quadratic)
        .filter(|&value| value > 0.0);
    let scan_t0 = Instant::now();
    // Max6 screen context is phenotype-specific, not window-specific. Build
    // it once for the active marker rows and pass cheap selected-row views to
    // each window instead of rebuilding lookup tables and marker margins in
    // every scan unit.
    let shared_y_lookup =
        PackedYSumLookup::build(train_fit.residualized_y.as_slice(), logic_bits.n_samples)?;
    let one_cell_margin_table = {
        let active_margin_rows = build_max6_active_margin_rows(
            prepared_screen_universe
                .as_ref()
                .map(|universe| universe.active_rows.as_slice()),
            scan_unit_indices,
            units,
        );
        if let Some(callback) = progress_callback {
            let mut notify = |done: usize, total: usize| {
                let detail = format!(
                    "{done}/{total} active marker rows; n={} samples",
                    logic_bits.n_samples
                );
                garfield_stage_progress_notify(
                    Some(callback),
                    "max6_setup_margins",
                    done,
                    total,
                    Some(detail.as_str()),
                )
                .map_err(|error| error.to_string())
            };
            OneCellMarginTable::build_for_rows_with_pool_and_progress(
                logic_bits.bits(),
                logic_bits.row_words,
                active_margin_rows.as_ref(),
                train_fit.residualized_y.as_slice(),
                logic_bits.n_samples,
                &shared_y_lookup,
                screen_pool.as_ref(),
                &mut notify,
            )?
        } else {
            OneCellMarginTable::build_for_rows_with_pool(
                logic_bits.bits(),
                logic_bits.row_words,
                active_margin_rows.as_ref(),
                train_fit.residualized_y.as_slice(),
                logic_bits.n_samples,
                &shared_y_lookup,
                screen_pool.as_ref(),
            )?
        }
    };
    let max6_screen_universe = if ld_tag_exhaustive {
        if let Some(prepared_screen) = prepared_screen_universe {
            Some(prepared_screen)
        } else if prepared_max6_universe.is_some() {
            return Err(
                "prepared Max6 handle is missing its trait-mapped unit universe".to_string(),
            );
        } else {
            if let Some(callback) = progress_callback {
                let mut notify = |done: usize, total: usize, selected_markers: usize| {
                    let detail =
                        format!("{done}/{total} scan units; {selected_markers} LD tags selected");
                    garfield_stage_progress_notify(
                        Some(callback),
                        "max6_setup_ld_tags",
                        done,
                        total,
                        Some(detail.as_str()),
                    )
                    .map_err(|error| error.to_string())
                };
                Some(prepare_max6_screen_universe(
                    logic_bits,
                    units,
                    scan_unit_indices,
                    unit_kind_lc,
                    train_idx_local,
                    screen_pool.as_ref(),
                    Some(&mut notify),
                    ld_tag_r2,
                )?)
            } else {
                Some(prepare_max6_screen_universe(
                    logic_bits,
                    units,
                    scan_unit_indices,
                    unit_kind_lc,
                    train_idx_local,
                    screen_pool.as_ref(),
                    None,
                    ld_tag_r2,
                )?)
            }
        }
    } else {
        None
    };
    let max6_pair_universe_size = if let Some(prepared) = prepared_max6_universe {
        prepared.pair_universe_size
    } else if let Some(universe) = max6_screen_universe.as_ref() {
        if universe.pair_ownership.membership_units.is_empty() {
            count_max6_screen_pairs(universe)?
        } else if let Some(callback) = progress_callback {
            let mut notify = |done: usize, total: usize, unique_pairs: usize| {
                let detail =
                    format!("{done}/{total} candidate pairs checked; {unique_pairs} unique pairs");
                garfield_stage_progress_notify(
                    Some(callback),
                    "max6_pair_count",
                    done,
                    total,
                    Some(detail.as_str()),
                )
                .map_err(|error| error.to_string())
            };
            count_max6_screen_pairs_with_progress(universe, screen_pool.as_ref(), &mut notify)?
        } else {
            count_max6_screen_pairs_with_pool(universe, screen_pool.as_ref())?
        }
    } else {
        0
    };
    let (max6_bonferroni_q_cut, max6_bonferroni_p_cut) = if ld_tag_exhaustive {
        let (q_cut, p_cut) = if let Some(prepared) = prepared_max6_universe {
            (prepared.bonferroni_q_cut, prepared.bonferroni_p_cut)
        } else {
            max6_bonferroni_cutoff(max6_pair_universe_size)?
        };
        (Some(q_cut), Some(p_cut))
    } else {
        (None, None)
    };
    let max6_screen_calibration_t0 = Instant::now();
    let max6_screen_calibration_needed = !grm_int_only
        || sampled_screen_null_diagnostic
        || max6_screen_exceedance_audit
        || max6_screen_pot_audit
        || max6_screen_tail_audit;
    let max6_screen_calibration =
        if ld_tag_exhaustive && rule_permutation && max6_screen_calibration_needed {
            let universe = max6_screen_universe.as_ref().ok_or_else(|| {
                "Max6 screen calibration requires the LD-tag pair universe".to_string()
            })?;
            Some(calibrate_max6_screen_threshold(
                logic_bits,
                universe,
                train_fit.null_residualized_y.as_slice(),
                rule_null_quantile,
                screen_pool.as_ref(),
                sampled_screen_null_diagnostic,
                |done, total, unique_pairs| {
                    let meta = if done == 0 {
                        format!(
                        "{} window-pair occurrences; canonical unique count pending; {} thread(s)",
                        universe.window_pair_count, threads
                    )
                    } else {
                        format!(
                            "{} window-pair occurrences → {} canonical unique pairs; {} thread(s)",
                            universe.window_pair_count, unique_pairs, threads
                        )
                    };
                    garfield_stage_progress_notify(
                        progress_callback,
                        "max6_screen_calibration",
                        done,
                        total,
                        Some(meta.as_str()),
                    )
                    .map_err(|error| error.to_string())
                },
            )?)
        } else {
            None
        };
    if let Some(calibration) = max6_screen_calibration.as_ref() {
        if calibration.pair_universe_size != max6_pair_universe_size {
            return Err(format!(
                "Max6 pair count {} differs from calibration count {}",
                max6_pair_universe_size, calibration.pair_universe_size
            ));
        }
    }
    let max6_screen_pot_audit_t0 = Instant::now();
    let max6_screen_pot_audit_result = if max6_screen_pot_audit {
        let universe = max6_screen_universe.as_ref().ok_or_else(|| {
            "Max6 POT audit requires the prepared canonical unique-pair universe".to_string()
        })?;
        let calibration = max6_screen_calibration.as_ref().ok_or_else(|| {
            "Max6 POT audit requires the full-MaxT calibration null stream".to_string()
        })?;
        let prefix = out_prefix
            .ok_or_else(|| "Max6 POT audit requires an external output prefix".to_string())?;
        Some(sampled_screen::run_sampled_screen_pot_audit(
            logic_bits,
            universe,
            train_fit.null_residualized_y.as_slice(),
            calibration,
            seed,
            prefix,
            screen_pool.as_ref(),
            |stage, done, total, estimate| {
                let meta = format!("{estimate} bytes preflight/observed; {threads} thread(s)");
                garfield_stage_progress_notify(
                    progress_callback,
                    stage,
                    done,
                    total,
                    Some(meta.as_str()),
                )
                .map_err(|error| error.to_string())
            },
        )?)
    } else {
        None
    };
    let timing_max6_screen_pot_audit_wall_s = if max6_screen_pot_audit_result.is_some() {
        max6_screen_pot_audit_t0.elapsed().as_secs_f64()
    } else {
        0.0
    };
    let max6_screen_sampled_diagnostic_t0 = Instant::now();
    let max6_screen_sampled_diagnostic = if sampled_screen_null_diagnostic {
        let universe = max6_screen_universe.as_ref().ok_or_else(|| {
            "sampled Max6 diagnostic requires the prepared canonical pair universe".to_string()
        })?;
        let calibration = max6_screen_calibration.as_ref().ok_or_else(|| {
            "sampled Max6 diagnostic requires the full-maxT reference calibration".to_string()
        })?;
        let prefix = out_prefix.ok_or_else(|| {
            "sampled Max6 diagnostic requires out_prefix for its audit files".to_string()
        })?;
        let maf_cutpoints =
            sampled_screen::parse_cutpoints(sampled_screen_maf_cutpoints, "min-MAF", 0.5)?;
        let r2_cutpoints =
            sampled_screen::parse_cutpoints(sampled_screen_r2_cutpoints, "pairwise-r2", 1.0)?;
        Some(sampled_screen::run_sampled_screen_null_diagnostics(
            logic_bits,
            universe,
            train_fit.null_residualized_y.as_slice(),
            seed,
            sampled_screen_pair_seeds,
            calibration,
            maf_cutpoints.as_slice(),
            r2_cutpoints.as_slice(),
            prefix,
            screen_pool.as_ref(),
            |stage, done, total, count| {
                let meta = format!("{} eligible pairs; {} thread(s)", count, threads);
                garfield_stage_progress_notify(
                    progress_callback,
                    stage,
                    done,
                    total,
                    Some(meta.as_str()),
                )
                .map_err(|error| error.to_string())
            },
        )?)
    } else {
        None
    };
    let timing_max6_screen_sampled_diagnostic_wall_s = if max6_screen_sampled_diagnostic.is_some() {
        max6_screen_sampled_diagnostic_t0.elapsed().as_secs_f64()
    } else {
        0.0
    };
    let timing_max6_screen_calibration_wall_s = if max6_screen_calibration.is_some() {
        max6_screen_calibration_t0.elapsed().as_secs_f64()
    } else {
        0.0
    };
    let max6_screen_exceedance_audit_t0 = Instant::now();
    let max6_screen_exceedance_audit_result = if max6_screen_exceedance_audit {
        let universe = max6_screen_universe.as_ref().ok_or_else(|| {
            "weighted exceedance audit requires the prepared canonical unique-pair universe"
                .to_string()
        })?;
        let calibration = max6_screen_calibration.as_ref().ok_or_else(|| {
            "weighted exceedance audit requires the existing full-MaxT calibration".to_string()
        })?;
        if train_fit.null_residualized_y.len() != calibration.maxima.len() {
            return Err(format!(
                "weighted exceedance audit must reuse paired full-MaxT calibration nulls; nulls={} maxima={}",
                train_fit.null_residualized_y.len(),
                calibration.maxima.len()
            ));
        }
        if calibration.maxima.len() != permutation_repeats {
            return Err(format!(
                "weighted exceedance audit calibration draw count {} != requested null repeats {}",
                calibration.maxima.len(),
                permutation_repeats
            ));
        }
        let prefix = out_prefix.ok_or_else(|| {
            "weighted exceedance audit requires an external output prefix".to_string()
        })?;
        if sampled_screen::WEIGHTED_EXCEEDANCE_PAIR_SEEDS.contains(&seed) {
            return Err(
                "weighted exceedance pair-sample seeds must be separate from the null seed"
                    .to_string(),
            );
        }
        let grid = sampled_screen::build_exceedance_audit_grid(calibration.maxima.as_slice())?;
        let sampling_plan = sampled_screen::prepare_weighted_exceedance_sampling_plan(
            logic_bits,
            universe,
            sampled_screen::WEIGHTED_EXCEEDANCE_PAIR_SEEDS,
            |stage, done, total, count| {
                let meta = if stage == "max6_exceedance_pair_preflight" {
                    format!(
                        "sample heaps preflight {:.2} GiB; {} thread(s)",
                        count as f64 / (1024.0 * 1024.0 * 1024.0),
                        threads
                    )
                } else {
                    format!(
                        "{} canonical pairs; deterministic sample preparation; {} thread(s)",
                        count, threads
                    )
                };
                garfield_stage_progress_notify(
                    progress_callback,
                    stage,
                    done,
                    total,
                    Some(meta.as_str()),
                )
                .map_err(|error| error.to_string())
            },
        )?;
        if sampling_plan.unique_universe_count != calibration.pair_universe_size {
            return Err(format!(
                "weighted exceedance sample universe {} != full-MaxT universe {}",
                sampling_plan.unique_universe_count, calibration.pair_universe_size
            ));
        }
        let scan = scan_max6_screen_exceedance_with_progress(
            logic_bits,
            universe,
            train_fit.null_residualized_y.as_slice(),
            calibration.maxima.as_slice(),
            grid.thresholds.as_slice(),
            &sampling_plan,
            screen_pool.as_ref(),
            |done, total, count, preflight_bytes| {
                let meta = format!(
                    "{count} unique pairs; accumulator/output preflight {:.2} GiB; {} thread(s)",
                    preflight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    threads
                );
                garfield_stage_progress_notify(
                    progress_callback,
                    "max6_screen_exceedance_full_scan",
                    done,
                    total,
                    Some(meta.as_str()),
                )
                .map_err(|error| error.to_string())
            },
        )?;
        if scan.unique_pair_count != calibration.pair_universe_size {
            return Err(format!(
                "weighted exceedance scan universe {} != full-MaxT universe {}",
                scan.unique_pair_count, calibration.pair_universe_size
            ));
        }
        for (null_index, (&expected_maximum, top_pairs)) in calibration
            .maxima
            .iter()
            .zip(scan.top_pairs.iter())
            .enumerate()
        {
            let observed_maximum = top_pairs.first().map(|pair| pair.score).ok_or_else(|| {
                format!(
                    "weighted exceedance Top-4096 is empty for null {}",
                    null_index + 1
                )
            })?;
            if observed_maximum.to_bits() != expected_maximum.to_bits() {
                return Err(format!(
                    "weighted exceedance exact Top1 differs from the paired full-MaxT maximum at null {}: {:?} vs {:?}",
                    null_index + 1,
                    observed_maximum,
                    expected_maximum
                ));
            }
        }
        Some(sampled_screen::write_weighted_exceedance_audit(
            logic_bits,
            universe,
            train_fit.null_residualized_y.as_slice(),
            calibration.maxima.as_slice(),
            &grid,
            &sampling_plan,
            &scan,
            seed,
            seed ^ 0xB471_8E2D_69C5_A30F,
            prefix,
        )?)
    } else {
        None
    };
    let timing_max6_screen_exceedance_audit_wall_s =
        if max6_screen_exceedance_audit_result.is_some() {
            max6_screen_exceedance_audit_t0.elapsed().as_secs_f64()
        } else {
            0.0
        };
    let max6_screen_validation_t0 = Instant::now();
    let max6_screen_validation = if let (Some(universe), Some(calibration)) = (
        max6_screen_universe.as_ref(),
        max6_screen_calibration.as_ref(),
    ) {
        if train_fit.null_validation_residualized_y.len() != calibration.maxima.len() {
            return Err(format!(
                "independent Max6 null validation repeats {} != calibration repeats {}",
                train_fit.null_validation_residualized_y.len(),
                calibration.maxima.len()
            ));
        }
        let (validation_maxima, _, validation_pair_count) =
            scan_max6_screen_null_maxima_with_progress(
                logic_bits,
                universe,
                train_fit.null_validation_residualized_y.as_slice(),
                screen_pool.as_ref(),
                false,
                |done, total, unique_pairs| {
                    let meta = format!(
                        "{} window-pair occurrences; {} canonical unique pairs; {} thread(s)",
                        universe.window_pair_count, unique_pairs, threads
                    );
                    garfield_stage_progress_notify(
                        progress_callback,
                        "max6_screen_validation",
                        done,
                        total,
                        Some(meta.as_str()),
                    )
                    .map_err(|error| error.to_string())
                },
            )?;
        if validation_pair_count != calibration.pair_universe_size {
            return Err(format!(
                "Max6 calibration/validation canonical pair universe mismatch: {} != {}",
                calibration.pair_universe_size, validation_pair_count
            ));
        }
        Some(summarize_max6_screen_validation(
            validation_maxima.as_slice(),
            calibration.empirical_threshold,
        )?)
    } else {
        None
    };
    let timing_max6_screen_validation_wall_s = if max6_screen_validation.is_some() {
        max6_screen_validation_t0.elapsed().as_secs_f64()
    } else {
        0.0
    };
    let max6_screen_tail_audit_t0 = Instant::now();
    let max6_screen_tail_audit = if max6_screen_tail_audit {
        let universe = max6_screen_universe.as_ref().ok_or_else(|| {
            "Max6 tail audit requires the prepared canonical unique-pair universe".to_string()
        })?;
        let out_prefix = out_prefix.ok_or_else(|| {
            "Max6 tail audit requires an output prefix for audit files".to_string()
        })?;
        let tail_null_seed = residual::null_tail_audit_seed(seed);
        let sample = sampled_screen::prepare_tail_audit_sample(
            universe,
            max6_screen_tail_pair_seed,
            |done, total, count| {
                let meta = format!("{count} canonical pairs observed; seed-priority bottom-k");
                garfield_stage_progress_notify(
                    progress_callback,
                    "max6_screen_tail_sample_selection",
                    done,
                    total,
                    Some(meta.as_str()),
                )
                .map_err(|error| error.to_string())
            },
        )?;
        let sampled = sampled_screen::score_tail_audit_sampled_maxima(
            logic_bits,
            universe,
            sample,
            train_fit.tail_audit_residualized_y.as_slice(),
            screen_pool.as_ref(),
            |done, total, count| {
                let meta =
                    format!("{count} unique pairs; nested sampled maxima; {threads} thread(s)");
                garfield_stage_progress_notify(
                    progress_callback,
                    "max6_screen_tail_sample_scoring",
                    done,
                    total,
                    Some(meta.as_str()),
                )
                .map_err(|error| error.to_string())
            },
        )?;
        let mut sampled_winners_for_full_scan = sampled.winners.clone();
        let support_aware = if max6_screen_tail_support_aware {
            Some(sampled_screen::run_support_aware_tail_audit_sampling(
                logic_bits,
                universe,
                max6_screen_tail_pair_seed,
                train_fit.tail_audit_residualized_y.as_slice(),
                screen_pool.as_ref(),
                &max6_screen_tail_support_shares,
                |stage, done, total, count| {
                    let meta =
                        format!("{count} unique pairs; support-aware quotas; {threads} thread(s)");
                    garfield_stage_progress_notify(
                        progress_callback,
                        stage,
                        done,
                        total,
                        Some(meta.as_str()),
                    )
                    .map_err(|error| error.to_string())
                },
            )?)
        } else {
            None
        };
        if let Some(support_aware) = support_aware.as_ref() {
            for design in support_aware {
                sampled_winners_for_full_scan.extend(design.winners.clone());
            }
        }
        let full = scan_max6_screen_tail_with_progress(
            logic_bits,
            universe,
            train_fit.tail_audit_residualized_y.as_slice(),
            sampled_winners_for_full_scan.as_slice(),
            screen_pool.as_ref(),
            |done, total, count| {
                let meta = format!(
                    "{count} canonical pairs; bounded Top-{}; {threads} thread(s)",
                    sampled_screen::TAIL_AUDIT_TOP_K
                );
                garfield_stage_progress_notify(
                    progress_callback,
                    "max6_screen_tail_full_scan",
                    done,
                    total,
                    Some(meta.as_str()),
                )
                .map_err(|error| error.to_string())
            },
        )?;
        let expected_pair_count = max6_screen_calibration
            .as_ref()
            .map(|calibration| calibration.pair_universe_size)
            .ok_or_else(|| {
                "Max6 tail audit requires the standard full-maxT reference".to_string()
            })?;
        if full.unique_pair_count != expected_pair_count {
            return Err(format!(
                "Max6 tail scanner unique-pair count {} differs from standard full-maxT count {}",
                full.unique_pair_count, expected_pair_count
            ));
        }
        let mut paths = sampled_screen::ranked_geometry_summary(
            logic_bits,
            universe,
            &sampled,
            &full,
            tail_null_seed,
            out_prefix,
        )?;
        if let Some(support_aware) = support_aware.as_ref() {
            let support_paths = sampled_screen::support_aware_geometry_summary(
                logic_bits,
                universe,
                &sampled,
                support_aware,
                &full,
                tail_null_seed,
                out_prefix,
            )?;
            paths.support_aware_maxima_tsv = Some(support_paths.maxima_tsv);
            paths.support_aware_strata_tsv = Some(support_paths.strata_tsv);
            paths.support_aware_summary_json = Some(support_paths.summary_json);
        }
        Some(paths)
    } else {
        None
    };
    let timing_max6_screen_tail_audit_wall_s = if max6_screen_tail_audit.is_some() {
        max6_screen_tail_audit_t0.elapsed().as_secs_f64()
    } else {
        0.0
    };
    let screen_score_threshold = (!grm_int_only)
        .then(|| {
            max6_screen_calibration
                .as_ref()
                .map(|calibration| calibration.empirical_threshold)
        })
        .flatten();
    // The POT audit needs these exact calibration and independent validation
    // maxima for the post-freeze analysis phases even when general diagnostics
    // are disabled. Emitting them is opt-in with the POT audit only.
    let emit_max6_screen_maxima = diagnostics_enabled || max6_screen_pot_audit;
    let max6_screen_maxima_tsv = if emit_max6_screen_maxima {
        if let (Some(prefix), Some(calibration)) = (out_prefix, max6_screen_calibration.as_ref()) {
            let path = format!("{prefix}.max6_grmmax6.screen_null_maxima.tsv");
            write_max6_screen_null_maxima(path.as_str(), calibration.maxima.as_slice())?;
            Some(path)
        } else {
            None
        }
    } else {
        None
    };
    let max6_screen_validation_maxima_tsv = if emit_max6_screen_maxima {
        if let (Some(prefix), Some(validation), Some(calibration)) = (
            out_prefix,
            max6_screen_validation.as_ref(),
            max6_screen_calibration.as_ref(),
        ) {
            let path = format!("{prefix}.max6_grmmax6.screen_validation_maxima.tsv");
            write_max6_screen_validation_maxima(
                path.as_str(),
                validation,
                calibration.empirical_threshold,
            )?;
            Some(path)
        } else {
            None
        }
    } else {
        None
    };
    // The calibration/validation pool is no longer needed below. Drop it
    // before constructing the observed-scan pool so `-t` does not leave two
    // independent Rayon pools resident at once.
    drop(screen_pool);
    let pair_spool_path = out_prefix
        .map(|prefix| format!("{prefix}.max6_grmmax6.pairs.tsv.tmp.{}", std::process::id()));
    let mut pair_spool = if let Some(path) = pair_spool_path.as_ref() {
        Some(BufWriter::with_capacity(
            MAX6_GRMMAX6_IO_BUFFER_BYTES,
            File::create(path)
                .map_err(|error| format!("failed to create Max6 pair spool '{}': {error}", path))?,
        ))
    } else {
        None
    };
    let pair_provenance_spool_path = if grm_int_only && write_provenance {
        out_prefix.map(|prefix| {
            format!(
                "{prefix}.max6_grmmax6.provenance.tsv.tmp.{}",
                std::process::id()
            )
        })
    } else {
        None
    };
    let mut pair_provenance_spool = if let Some(path) = pair_provenance_spool_path.as_ref() {
        Some(BufWriter::with_capacity(
            MAX6_GRMMAX6_IO_BUFFER_BYTES,
            File::create(path).map_err(|error| {
                format!("failed to create Max6 provenance spool '{path}': {error}")
            })?,
        ))
    } else {
        None
    };
    let mut provenance_output_row_index = 0usize;
    let max6_permutation_active = !grm_int_only
        && legacy_max6_permutation_active(ld_tag_exhaustive, rule_permutation, permutation_repeats);
    let permutation_spool_path = if max6_permutation_active {
        Some(out_prefix.map_or_else(
            || {
                env::temp_dir()
                    .join(format!(
                        "janusx-max6-grmmax6-permutation.{}.bin.tmp",
                        std::process::id()
                    ))
                    .to_string_lossy()
                    .into_owned()
            },
            |prefix| {
                format!(
                    "{prefix}.max6_grmmax6.permutation.bin.tmp.{}",
                    std::process::id()
                )
            },
        ))
    } else {
        None
    };
    let mut permutation_spool = if let Some(path) = permutation_spool_path.as_ref() {
        Some(BufWriter::with_capacity(
            MAX6_GRMMAX6_IO_BUFFER_BYTES,
            File::create(path).map_err(|error| {
                format!(
                    "failed to create Max6 permutation spool '{}': {error}",
                    path
                )
            })?,
        ))
    } else {
        None
    };
    let mut permutation_spool_offset = 0u64;
    let mut output_count = 0usize;
    let mut max6_screen_candidate_count = 0usize;
    let mut max6_final_screen_subset_audit_pairs_checked = 0usize;
    let mut max6_observed_scan_timing = Max6ObservedScanTiming::default();
    let cross_window_union_override = env::var("JX_GARFIELD_MAX6_CROSS_WINDOW_UNION_RERANK").ok();
    let cross_window_union_rerank = !grm_int_only
        && max6_cross_window_union_enabled(
            unit_kind_lc,
            scan_unit_indices.len(),
            cross_window_union_override.as_deref(),
        );
    let cross_window_parity_audit = env_truthy("JX_GARFIELD_MAX6_CROSS_WINDOW_PARITY_AUDIT");
    if cross_window_parity_audit && !cross_window_union_rerank {
        return Err(
            "JX_GARFIELD_MAX6_CROSS_WINDOW_PARITY_AUDIT requires cross-window union reranking"
                .to_string(),
        );
    }
    let cross_window_batch_size =
        parse_env_usize("JX_GARFIELD_MAX6_UNION_BATCH_WINDOWS").unwrap_or(4);
    if cross_window_union_rerank && !(2..=16).contains(&cross_window_batch_size) {
        return Err(format!(
            "JX_GARFIELD_MAX6_UNION_BATCH_WINDOWS must be in 2..=16, got {cross_window_batch_size}"
        ));
    }
    const MAX_CROSS_WINDOW_CONCURRENT_WORKSPACE_BYTES: usize = 2_147_483_648;
    let cross_window_worker_workspace_limit =
        MAX_CROSS_WINDOW_CONCURRENT_WORKSPACE_BYTES / threads.max(1);
    let mut cross_window_batch_summary = Max6CrossWindowBatchStats::default();
    let candidate_score_reuse_audit = env::var("JX_GARFIELD_MAX6_SCORE_REUSE_AUDIT")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        })
        .unwrap_or(false);
    let candidate_reuse_diagnostic = env::var("JX_GARFIELD_MAX6_REUSE_DIAGNOSTIC")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        })
        .unwrap_or(false)
        || candidate_score_reuse_audit;
    max6_observed_scan_timing.candidate_score_reuse_audit_enabled = candidate_score_reuse_audit;
    const REUSE_SCREEN_PROFILE_PAIRS: usize = 100_000;
    const MAX_REUSE_DIAGNOSTIC_BUFFER_BYTES: usize = 1_073_741_824;
    let mut candidate_pair_keys = Vec::<u64>::new();
    let mut candidate_score_observations = Vec::<Max6CandidateScoreObservation>::new();
    let mut candidate_window_pair_counts = Vec::<usize>::new();
    let mut candidate_window_screen_pairs_evaluated = Vec::<usize>::new();
    let mut candidate_window_endpoint_counts = Vec::<usize>::new();
    if candidate_reuse_diagnostic {
        max6_observed_scan_timing.candidate_set_hash_by_window_order = MAX6_DIAGNOSTIC_FNV_OFFSET;
        max6_observed_scan_timing.exact_topk_set_hash_by_window_order = MAX6_DIAGNOSTIC_FNV_OFFSET;
        max6_observed_scan_timing.exact_topk_order_hash_by_window_order =
            MAX6_DIAGNOSTIC_FNV_OFFSET;
        let estimated_pair_capacity = scan_unit_indices
            .len()
            .checked_mul(screen_top_k)
            .ok_or_else(|| "Max6 candidate diagnostic capacity overflow".to_string())?;
        let estimated_key_bytes = estimated_pair_capacity
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| "Max6 candidate diagnostic byte estimate overflow".to_string())?;
        let estimated_score_bytes = if candidate_score_reuse_audit {
            estimated_pair_capacity
                .checked_mul(std::mem::size_of::<Max6CandidateScoreObservation>())
                .ok_or_else(|| "Max6 score reuse audit byte estimate overflow".to_string())?
        } else {
            0
        };
        let estimated_total_bytes = estimated_key_bytes
            .checked_add(estimated_score_bytes)
            .ok_or_else(|| "Max6 reuse diagnostic total byte estimate overflow".to_string())?;
        if estimated_total_bytes > MAX_REUSE_DIAGNOSTIC_BUFFER_BYTES {
            return Err(format!(
                "Max6 reuse diagnostic estimates {estimated_total_bytes} bytes of pair/score buffers, above the 1 GiB safety cap"
            ));
        }
        candidate_pair_keys
            .try_reserve_exact(estimated_pair_capacity)
            .map_err(|error| {
                format!(
                    "unable to reserve {estimated_key_bytes} bytes for Max6 candidate reuse diagnostic: {error}"
                )
            })?;
        if candidate_score_reuse_audit {
            candidate_score_observations
                .try_reserve_exact(estimated_pair_capacity)
                .map_err(|error| {
                    format!(
                        "unable to reserve {estimated_score_bytes} bytes for Max6 score reuse audit: {error}"
                    )
                })?;
        }
        candidate_window_pair_counts.reserve(scan_unit_indices.len());
        candidate_window_screen_pairs_evaluated.reserve(scan_unit_indices.len());
        candidate_window_endpoint_counts.reserve(scan_unit_indices.len());
    }
    let screen_profile_unit_indices = if candidate_reuse_diagnostic && !scan_unit_indices.is_empty()
    {
        let positions = [0, scan_unit_indices.len() / 2, scan_unit_indices.len() - 1];
        let mut selected = Vec::with_capacity(positions.len());
        for position in positions {
            let unit_index = scan_unit_indices[position];
            if !selected.contains(&unit_index) {
                selected.push(unit_index);
            }
        }
        selected
    } else {
        Vec::new()
    };
    let mut permutation_pool_meta = Vec::<Max6GrmMax6PermutationPoolMeta>::new();
    let mut skipped_messages = Vec::<String>::new();
    let outer_parallel = threads > 1 && scan_unit_indices.len() > 1;
    let mut unit_beam_params = beam_params;
    // A worker owns one scan window. Disable nested Rayon work inside that
    // window when the outer pool is active; otherwise -t would be multiplied
    // by the inner pair/scoring pools.
    unit_beam_params.allow_parallel = unit_beam_params.allow_parallel && !outer_parallel;
    let outer_pool = if outer_parallel {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("janusx-garfield-window-{index}"))
                .build()
                .map_err(|error| format!("failed to build GARFIELD window pool: {error}"))?,
        )
    } else {
        None
    };
    let scan_deferred_unit = |unit_index: usize| -> Result<Max6GrmMax6UnitScan, String> {
        let unit = units
            .get(unit_index)
            .ok_or_else(|| format!("Max6 backend unit index out of range: {unit_index}"))?;
        scan_max6_grmmax6_unit(
            unit,
            unit_index,
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
            train_fit.residualized_y.as_slice(),
            &shared_y_lookup,
            &one_cell_margin_table,
            p_c,
            no_grm_projector,
            projected_y.as_slice(),
            y,
            score_scale,
            scaled_total_quadratic,
            &unit_beam_params,
            final_top_k,
            min_cell,
            grm_int_only,
            max6_permutation_active,
            ld_tag_exhaustive,
            max6_screen_universe
                .as_ref()
                .and_then(|universe| universe.units.get(unit_index)),
            screen_score_threshold,
            candidate_reuse_diagnostic,
            candidate_score_reuse_audit,
            screen_profile_unit_indices
                .contains(&unit_index)
                .then_some(REUSE_SCREEN_PROFILE_PAIRS),
            true,
            None,
            None,
        )
    };
    let scan_direct_unit = |unit_index: usize| -> Result<Max6GrmMax6UnitScan, String> {
        let unit = units
            .get(unit_index)
            .ok_or_else(|| format!("Max6 backend unit index out of range: {unit_index}"))?;
        scan_max6_grmmax6_unit(
            unit,
            unit_index,
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
            train_fit.residualized_y.as_slice(),
            &shared_y_lookup,
            &one_cell_margin_table,
            p_c,
            no_grm_projector,
            projected_y.as_slice(),
            y,
            score_scale,
            scaled_total_quadratic,
            &unit_beam_params,
            final_top_k,
            min_cell,
            grm_int_only,
            max6_permutation_active,
            ld_tag_exhaustive,
            max6_screen_universe
                .as_ref()
                .and_then(|universe| universe.units.get(unit_index)),
            screen_score_threshold,
            candidate_reuse_diagnostic,
            candidate_score_reuse_audit,
            screen_profile_unit_indices
                .contains(&unit_index)
                .then_some(REUSE_SCREEN_PROFILE_PAIRS),
            false,
            None,
            if scan_unit_indices.len() == 1 {
                progress_callback
            } else {
                None
            },
        )
    };
    // candidate_rerank_exact contains BLAS calls. Keep them single-threaded
    // while windows are parallel so -t is a process-level cap.
    let _blas_guard = outer_parallel.then(|| BlasThreadGuard::enter(1));
    // Reuse the same cost-aware/coalesced scheduler as the regular window
    // path.  A fixed `2 * threads` chunk count leaves the next wave waiting
    // for the slowest window in the previous wave and was especially visible
    // when candidate rerank cost varied with the number of selected SNPs.
    let prepared_slot_by_original = max6_screen_universe.as_ref().map(|universe| {
        universe
            .original_unit_indices
            .iter()
            .enumerate()
            .map(|(slot, &original)| (original, slot))
            .collect::<HashMap<_, _>>()
    });
    let max6_batch_chunks = if grm_int_only
        && !candidate_reuse_diagnostic
        && !candidate_score_reuse_audit
        && !cross_window_union_rerank
    {
        max6_screen_universe.as_ref().and_then(|universe| {
            max6_membership_scan_task_chunks(
                scan_unit_indices,
                units,
                universe,
                threads,
                max6_grmmax6_screen_top_k(true, final_top_k),
            )
        })
    } else {
        None
    };
    let use_max6_batch_screen = max6_batch_chunks.is_some();
    let scan_unit_chunks = if let Some(chunks) = max6_batch_chunks {
        chunks
    } else if outer_parallel {
        garfield_scan_task_chunks(
            scan_unit_indices,
            units,
            max6_screen_universe
                .as_ref()
                .map(|universe| universe.units.as_slice()),
            threads,
            garfield_scan_task_coalesce_max_units(),
        )
    } else {
        vec![scan_unit_indices.to_vec()]
    };
    let scan_progress_detail = if grm_int_only {
        "Max6 Top-K screen, conditional interaction scoring, and final subset audit"
    } else {
        "screen candidates, exact Max6 rerank, and final subset audit"
    };
    garfield_stage_progress_notify(
        progress_callback,
        "max6_grmmax6_scan",
        0,
        scan_unit_indices.len().max(1),
        Some(scan_progress_detail),
    )
    .map_err(|error| error.to_string())?;
    let mut done = 0usize;
    let scan_precomputed_unit = |unit_index: usize,
                                 precomputed_screen: Max6PrecomputedScreen|
     -> Result<Max6GrmMax6UnitScan, String> {
        let unit = units
            .get(unit_index)
            .ok_or_else(|| format!("Max6 backend unit index out of range: {unit_index}"))?;
        let prepared_unit = max6_screen_universe.as_ref().and_then(|universe| {
            prepared_slot_by_original
                .as_ref()
                .and_then(|slots| slots.get(&unit_index))
                .and_then(|&slot| universe.units.get(slot))
        });
        scan_max6_grmmax6_unit(
            unit,
            unit_index,
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
            train_fit.residualized_y.as_slice(),
            &shared_y_lookup,
            &one_cell_margin_table,
            p_c,
            no_grm_projector,
            projected_y.as_slice(),
            y,
            score_scale,
            scaled_total_quadratic,
            &unit_beam_params,
            final_top_k,
            min_cell,
            grm_int_only,
            max6_permutation_active,
            ld_tag_exhaustive,
            prepared_unit,
            screen_score_threshold,
            candidate_reuse_diagnostic,
            candidate_score_reuse_audit,
            None,
            false,
            Some(precomputed_screen),
            None,
        )
    };
    let shared_grm_int_score_pairs = parse_env_usize("JX_GARFIELD_GRM_INT_SCORE_PAIRS")
        .unwrap_or(if cfg!(windows) { 32 } else { 64 });
    for unit_chunk in scan_unit_chunks.iter() {
        let chunk_results = if cross_window_union_rerank {
            let batches = unit_chunk
                .chunks(cross_window_batch_size)
                .map(|batch| batch.to_vec())
                .collect::<Vec<_>>();
            let run_batch = |batch: &[usize]| -> Result<_, String> {
                if batch.len() == 1 {
                    return Ok((
                        vec![(batch[0], scan_direct_unit(batch[0])?)],
                        Max6CrossWindowBatchStats {
                            direct_windows: 1,
                            ..Default::default()
                        },
                    ));
                }
                let screened = if let Some(pool) = outer_pool.as_ref() {
                    pool.install(|| {
                        batch
                            .par_iter()
                            .map(|&unit_index| scan_deferred_unit(unit_index))
                            .collect::<Result<Vec<_>, String>>()
                    })?
                } else {
                    batch
                        .iter()
                        .map(|&unit_index| scan_deferred_unit(unit_index))
                        .collect::<Result<Vec<_>, String>>()?
                };
                let mut indexed = batch.iter().copied().zip(screened).collect::<Vec<_>>();
                let stats = rerank_max6_grmmax6_screen_batch_with_audit(
                    indexed.as_mut_slice(),
                    logic_bits,
                    p_c,
                    no_grm_projector,
                    projected_y.as_slice(),
                    y,
                    score_scale,
                    scaled_total_quadratic,
                    final_top_k,
                    max6_permutation_active,
                    candidate_reuse_diagnostic,
                    candidate_score_reuse_audit,
                    cross_window_parity_audit,
                    cross_window_worker_workspace_limit,
                );
                let stats = match stats {
                    Ok(stats) => stats,
                    Err(error)
                        if !cross_window_parity_audit
                            && error.starts_with(MAX6_CROSS_WINDOW_WORKSPACE_PREFLIGHT_PREFIX) =>
                    {
                        let direct = batch
                            .iter()
                            .map(|&unit_index| Ok((unit_index, scan_direct_unit(unit_index)?)))
                            .collect::<Result<Vec<_>, String>>()?;
                        return Ok((
                            direct,
                            Max6CrossWindowBatchStats {
                                workspace_fallback_windows: batch.len(),
                                ..Default::default()
                            },
                        ));
                    }
                    Err(error) => return Err(error),
                };
                Ok((indexed, stats))
            };
            let batch_results = if let Some(pool) = outer_pool.as_ref() {
                pool.install(|| {
                    batches
                        .par_iter()
                        .map(|batch| run_batch(batch.as_slice()))
                        .collect::<Result<Vec<_>, String>>()
                })?
            } else {
                batches
                    .iter()
                    .map(|batch| run_batch(batch.as_slice()))
                    .collect::<Result<Vec<_>, String>>()?
            };
            let mut indexed_results = Vec::with_capacity(unit_chunk.len());
            for (mut batch_scans, batch_stats) in batch_results {
                cross_window_batch_summary.accumulate(batch_stats);
                indexed_results.append(&mut batch_scans);
            }
            indexed_results
                .into_iter()
                .map(|(_, scan)| scan)
                .collect::<Vec<_>>()
        } else if use_max6_batch_screen {
            let batch_prepare_started = Instant::now();
            let universe = max6_screen_universe
                .as_ref()
                .ok_or_else(|| "Max6 batch scan requires a prepared screen universe".to_string())?;
            let slot_by_original = prepared_slot_by_original
                .as_ref()
                .ok_or_else(|| "Max6 batch scan is missing prepared unit mapping".to_string())?;
            let batch_units = unit_chunk
                .iter()
                .map(|&unit_index| {
                    let slot = *slot_by_original.get(&unit_index).ok_or_else(|| {
                        format!("Max6 prepared scan unit mapping is missing unit {unit_index}")
                    })?;
                    let prepared = universe.units.get(slot).ok_or_else(|| {
                        format!("Max6 prepared scan unit slot {slot} is out of range")
                    })?;
                    Ok(logic_residual_screen::ResidualScreenBatchUnit {
                        selected_global_rows: prepared.selected_global_rows.as_slice(),
                        pair_groups: prepared.pair_groups.as_deref(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            max6_observed_scan_timing.screen_batch_prepare_wall_seconds +=
                batch_prepare_started.elapsed().as_secs_f64();
            let batch_screen_started = Instant::now();
            let run_batch_screen = || {
                logic_residual_screen::scan_residual_pairs_packed_max6_batch(
                    logic_bits.bits(),
                    logic_bits.row_words,
                    batch_units.as_slice(),
                    train_fit.residualized_y.as_slice(),
                    logic_bits.n_samples,
                    max6_grmmax6_screen_top_k(true, final_top_k),
                    min_cell,
                    &shared_y_lookup,
                    &one_cell_margin_table,
                    outer_pool.is_some() || unit_beam_params.allow_parallel,
                )
            };
            let batch_screen = if let Some(pool) = outer_pool.as_ref() {
                pool.install(run_batch_screen)?
            } else {
                run_batch_screen()?
            };
            let batch_kernel_seconds = batch_screen_started.elapsed().as_secs_f64();
            let timing = batch_screen.timing;
            max6_observed_scan_timing.screen_membership_prepare_wall_seconds +=
                timing.prepare_wall_seconds;
            max6_observed_scan_timing.screen_pair_scan_merge_wall_seconds +=
                timing.scan_merge_wall_seconds;
            max6_observed_scan_timing.screen_heap_merge_task_seconds += timing.merge_task_seconds;
            max6_observed_scan_timing.screen_finalist_wall_seconds += timing.finalize_wall_seconds;
            max6_observed_scan_timing.screen_parallel_batches += usize::from(timing.parallel);
            max6_observed_scan_timing.screen_serial_batches += usize::from(!timing.parallel);
            max6_observed_scan_timing.screen_source_rows_sum += timing.source_rows;
            max6_observed_scan_timing.screen_source_rows_max = max6_observed_scan_timing
                .screen_source_rows_max
                .max(timing.source_rows);
            max6_observed_scan_timing.screen_memberships_sum += timing.memberships;
            max6_observed_scan_timing
                .screen_batch_profiles
                .push(Max6BatchScanProfile {
                    scan_merge_wall_seconds: timing.scan_merge_wall_seconds,
                    fold_task_count: timing.fold_task_count,
                    fold_task_wall_sum_seconds: timing.fold_task_wall_sum_seconds,
                    fold_task_wall_max_seconds: timing.fold_task_wall_max_seconds,
                    fold_anchor_count_min: timing.fold_anchor_count_min,
                    fold_anchor_count_max: timing.fold_anchor_count_max,
                    fold_active_workers: timing.fold_active_workers,
                    worker_task_wall_min_seconds: timing.worker_task_wall_min_seconds,
                    worker_task_wall_median_seconds: timing.worker_task_wall_median_seconds,
                    worker_task_wall_max_seconds: timing.worker_task_wall_max_seconds,
                    worker_finish_skew_seconds: timing.worker_finish_skew_seconds,
                    post_last_fold_tail_seconds: timing.post_last_fold_tail_seconds,
                    topk_merge_task_seconds: timing.topk_merge_task_seconds,
                    topk_merge_max_task_seconds: timing.topk_merge_max_task_seconds,
                    topk_merge_task_count: timing.topk_merge_task_count,
                    topk_merge_critical_path_seconds: timing.topk_merge_critical_path_seconds,
                    scan_reduce_critical_path_seconds: timing.scan_reduce_critical_path_seconds,
                    pair_profile_sampled_anchors: timing.pair_scan_profile.sampled_anchors,
                    pair_profile_sampled_pairs: timing.pair_scan_profile.sampled_pairs,
                    pair_profile_sampled_scored_pairs: timing
                        .pair_scan_profile
                        .sampled_scored_pairs,
                    pair_profile_sampled_score_offers: timing
                        .pair_scan_profile
                        .sampled_score_offers,
                    pair_profile_cursor_merge_seconds: timing
                        .pair_scan_profile
                        .cursor_merge_seconds,
                    pair_profile_min_cell_gate_seconds: timing
                        .pair_scan_profile
                        .min_cell_gate_seconds,
                    pair_profile_sufficient_stats_seconds: timing
                        .pair_scan_profile
                        .sufficient_stats_seconds,
                    pair_profile_topk_offer_seconds: timing.pair_scan_profile.topk_offer_seconds,
                });
            max6_observed_scan_timing.screen_batch_count = max6_observed_scan_timing
                .screen_batch_count
                .saturating_add(1);
            if batch_screen.used_direct_fallback {
                max6_observed_scan_timing.screen_batch_direct_fallbacks = max6_observed_scan_timing
                    .screen_batch_direct_fallbacks
                    .saturating_add(1);
            } else {
                max6_observed_scan_timing.screen_batch_pairs_scored_unique =
                    max6_observed_scan_timing
                        .screen_batch_pairs_scored_unique
                        .saturating_add(batch_screen.pairs_scored_unique);
                max6_observed_scan_timing.screen_batch_score_offers = max6_observed_scan_timing
                    .screen_batch_score_offers
                    .saturating_add(batch_screen.score_offers);
            }
            if batch_screen.units.len() != unit_chunk.len() {
                return Err(format!(
                    "Max6 batch scanner returned {} unit screens for {} requested units",
                    batch_screen.units.len(),
                    unit_chunk.len()
                ));
            }
            let per_unit_kernel_seconds =
                batch_kernel_seconds / batch_screen.units.len().max(1) as f64;
            let grm_int_batch_started = Instant::now();
            let shared_grm_int_by_unit = if grm_int_only && unit_kind_lc == "window" {
                // A GRM-INT score is a property of the global marker pair and
                // trait, so overlapping windows can reuse its exact value.
                let selected_rows_by_unit = batch_units
                    .iter()
                    .map(|unit| unit.selected_global_rows)
                    .collect::<Vec<_>>();
                let local_pairs_by_unit = batch_screen
                    .units
                    .iter()
                    .map(|screen| {
                        screen
                            .candidates
                            .iter()
                            .map(|candidate| (candidate.first, candidate.second))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let pair_plan = max6_shared_candidate_pair_plan(
                    selected_rows_by_unit.as_slice(),
                    local_pairs_by_unit.as_slice(),
                )?;
                let bits = logic_bits.bits();
                let marker_count = bits.len() / logic_bits.row_words;
                let score_pair_chunk = |candidate_pairs: &[(usize, usize)]| {
                    grm_max6::grm_int_interaction_scores_packed(
                        bits,
                        logic_bits.row_words,
                        logic_bits.n_samples,
                        marker_count,
                        candidate_pairs,
                        p_c,
                        no_grm_projector,
                        projected_y.as_slice(),
                        score_scale,
                    )
                };
                let score_chunks = if let Some(pool) = outer_pool.as_ref() {
                    pool.install(|| {
                        pair_plan
                            .global_pairs
                            .par_chunks(shared_grm_int_score_pairs)
                            .map(score_pair_chunk)
                            .collect::<Result<Vec<_>, String>>()
                    })?
                } else {
                    pair_plan
                        .global_pairs
                        .chunks(shared_grm_int_score_pairs)
                        .map(score_pair_chunk)
                        .collect::<Result<Vec<_>, String>>()?
                };

                let mut unique_chisq = Vec::with_capacity(pair_plan.global_pairs.len());
                let mut unique_invalid = Vec::with_capacity(pair_plan.global_pairs.len());
                let mut shared_diagnostics = grm_max6::GrmIntScoreDiagnostics::default();
                for batch in score_chunks {
                    if batch.chisq.len() != batch.invalid.len() {
                        return Err(
                            "shared GRM-INT scores/statuses have different lengths".to_string()
                        );
                    }
                    unique_chisq.extend(batch.chisq);
                    unique_invalid.extend(batch.invalid);
                    shared_diagnostics.batches = shared_diagnostics
                        .batches
                        .saturating_add(batch.diagnostics.batches);
                    shared_diagnostics.max_workspace_bytes = shared_diagnostics
                        .max_workspace_bytes
                        .max(batch.diagnostics.max_workspace_bytes);
                    shared_diagnostics.elapsed_seconds += batch.diagnostics.elapsed_seconds;
                }
                if unique_chisq.len() != pair_plan.global_pairs.len()
                    || unique_invalid.len() != pair_plan.global_pairs.len()
                {
                    return Err(format!(
                        "shared GRM-INT scorer returned {} scores / {} statuses for {} unique candidates",
                        unique_chisq.len(),
                        unique_invalid.len(),
                        pair_plan.global_pairs.len()
                    ));
                }

                let mut scores_by_unit = Vec::with_capacity(pair_plan.unit_pair_indices.len());
                let diagnostic_recipient =
                    max6_shared_grm_int_diagnostic_recipient(&pair_plan.unit_pair_indices);
                for (unit_index, pair_indices) in pair_plan.unit_pair_indices.iter().enumerate() {
                    let mut chisq = Vec::with_capacity(pair_indices.len());
                    let mut invalid = Vec::with_capacity(pair_indices.len());
                    let mut diagnostics = grm_max6::GrmIntScoreDiagnostics::default();
                    for &pair_index in pair_indices {
                        let Some(&score) = unique_chisq.get(pair_index) else {
                            return Err(format!(
                                "shared GRM-INT pair index {pair_index} is out of range"
                            ));
                        };
                        let Some(&invalid_status) = unique_invalid.get(pair_index) else {
                            return Err(format!(
                                "shared GRM-INT invalid-status index {pair_index} is out of range"
                            ));
                        };
                        chisq.push(score);
                        invalid.push(invalid_status);
                        match invalid_status {
                            Some(grm_max6::GrmIntInvalidScore::SingularEndpointDesign) => {
                                diagnostics.singular_endpoint_design =
                                    diagnostics.singular_endpoint_design.saturating_add(1);
                            }
                            Some(
                                grm_max6::GrmIntInvalidScore::NegligibleInteractionInformation,
                            ) => {
                                diagnostics.negligible_residual_interaction_information =
                                    diagnostics
                                        .negligible_residual_interaction_information
                                        .saturating_add(1);
                            }
                            Some(grm_max6::GrmIntInvalidScore::Nonfinite) => {
                                diagnostics.nonfinite = diagnostics.nonfinite.saturating_add(1);
                            }
                            None => {}
                        }
                    }
                    if diagnostic_recipient == Some(unit_index) {
                        diagnostics.batches = shared_diagnostics.batches;
                        diagnostics.max_workspace_bytes = shared_diagnostics.max_workspace_bytes;
                        diagnostics.elapsed_seconds = shared_diagnostics.elapsed_seconds;
                    }
                    scores_by_unit.push(grm_max6::GrmIntScoreBatch {
                        chisq,
                        invalid,
                        diagnostics,
                    });
                }
                Some(scores_by_unit)
            } else {
                None
            };
            let shared_grm_int_by_unit = shared_grm_int_by_unit
                .map(|scores| scores.into_iter().map(Some).collect::<Vec<_>>())
                .unwrap_or_else(|| vec![None; batch_screen.units.len()]);
            let precomputed_screens = batch_screen
                .units
                .into_iter()
                .zip(shared_grm_int_by_unit)
                .map(|(screen, shared_grm_int)| Max6PrecomputedScreen {
                    screen,
                    kernel_seconds: per_unit_kernel_seconds,
                    shared_grm_int,
                })
                .collect::<Vec<_>>();
            let scored = if let Some(pool) = outer_pool.as_ref() {
                pool.install(|| {
                    unit_chunk
                        .par_iter()
                        .copied()
                        .zip(precomputed_screens.into_par_iter())
                        .map(|(unit_index, precomputed)| {
                            scan_precomputed_unit(unit_index, precomputed)
                        })
                        .collect::<Result<Vec<_>, String>>()
                })?
            } else {
                unit_chunk
                    .iter()
                    .copied()
                    .zip(precomputed_screens)
                    .map(|(unit_index, precomputed)| scan_precomputed_unit(unit_index, precomputed))
                    .collect::<Result<Vec<_>, String>>()?
            };
            max6_observed_scan_timing.grm_int_batch_wall_seconds +=
                grm_int_batch_started.elapsed().as_secs_f64();
            scored
        } else if let Some(pool) = outer_pool.as_ref() {
            pool.install(|| {
                unit_chunk
                    .par_iter()
                    .map(|&unit_index| {
                        let unit = units.get(unit_index).ok_or_else(|| {
                            format!("Max6 backend unit index out of range: {unit_index}")
                        })?;
                        scan_max6_grmmax6_unit(
                            unit,
                            unit_index,
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
                            train_fit.residualized_y.as_slice(),
                            &shared_y_lookup,
                            &one_cell_margin_table,
                            p_c,
                            no_grm_projector,
                            projected_y.as_slice(),
                            y,
                            score_scale,
                            scaled_total_quadratic,
                            &unit_beam_params,
                            final_top_k,
                            min_cell,
                            grm_int_only,
                            max6_permutation_active,
                            ld_tag_exhaustive,
                            max6_screen_universe
                                .as_ref()
                                .and_then(|universe| universe.units.get(unit_index)),
                            screen_score_threshold,
                            candidate_reuse_diagnostic,
                            candidate_score_reuse_audit,
                            screen_profile_unit_indices
                                .contains(&unit_index)
                                .then_some(REUSE_SCREEN_PROFILE_PAIRS),
                            false,
                            None,
                            None,
                        )
                    })
                    .collect::<Result<Vec<_>, String>>()
            })?
        } else {
            unit_chunk
                .iter()
                .map(|&unit_index| {
                    let unit = units.get(unit_index).ok_or_else(|| {
                        format!("Max6 backend unit index out of range: {unit_index}")
                    })?;
                    scan_max6_grmmax6_unit(
                        unit,
                        unit_index,
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
                        train_fit.residualized_y.as_slice(),
                        &shared_y_lookup,
                        &one_cell_margin_table,
                        p_c,
                        no_grm_projector,
                        projected_y.as_slice(),
                        y,
                        score_scale,
                        scaled_total_quadratic,
                        &unit_beam_params,
                        final_top_k,
                        min_cell,
                        grm_int_only,
                        max6_permutation_active,
                        ld_tag_exhaustive,
                        max6_screen_universe
                            .as_ref()
                            .and_then(|universe| universe.units.get(unit_index)),
                        screen_score_threshold,
                        candidate_reuse_diagnostic,
                        candidate_score_reuse_audit,
                        screen_profile_unit_indices
                            .contains(&unit_index)
                            .then_some(REUSE_SCREEN_PROFILE_PAIRS),
                        false,
                        None,
                        if scan_unit_indices.len() == 1 {
                            progress_callback
                        } else {
                            None
                        },
                    )
                })
                .collect::<Result<Vec<_>, String>>()?
        };
        let result_drain_started = Instant::now();
        for (unit_index, unit_result) in unit_chunk.iter().zip(chunk_results.into_iter()) {
            max6_observed_scan_timing.accumulate_unit(unit_result.timing);
            if candidate_reuse_diagnostic {
                let mut window_pair_keys = unit_result.candidate_pair_keys.clone();
                window_pair_keys.sort_unstable();
                let mut set_hash =
                    max6_diagnostic_hash_u64(MAX6_DIAGNOSTIC_FNV_OFFSET, *unit_index as u64);
                set_hash = max6_diagnostic_hash_u64(set_hash, window_pair_keys.len() as u64);
                for &pair_key in &window_pair_keys {
                    set_hash = max6_diagnostic_hash_u64(set_hash, pair_key);
                }
                max6_observed_scan_timing.candidate_set_hash_by_window_order =
                    max6_diagnostic_hash_u64(
                        max6_observed_scan_timing.candidate_set_hash_by_window_order,
                        set_hash,
                    );

                let mut top_hash =
                    max6_diagnostic_hash_u64(MAX6_DIAGNOSTIC_FNV_OFFSET, *unit_index as u64);
                top_hash = max6_diagnostic_hash_u64(top_hash, unit_result.outputs.len() as u64);
                let mut top_set_pair_keys = Vec::with_capacity(unit_result.outputs.len());
                for record in &unit_result.outputs {
                    let pair_key = max6_diagnostic_pair_key(
                        record.first_global_row,
                        record.second_global_row,
                    )?;
                    top_hash = max6_diagnostic_hash_u64(top_hash, pair_key);
                    top_set_pair_keys.push(pair_key);
                }
                top_set_pair_keys.sort_unstable();
                let mut top_set_hash =
                    max6_diagnostic_hash_u64(MAX6_DIAGNOSTIC_FNV_OFFSET, *unit_index as u64);
                top_set_hash =
                    max6_diagnostic_hash_u64(top_set_hash, top_set_pair_keys.len() as u64);
                for pair_key in top_set_pair_keys {
                    top_set_hash = max6_diagnostic_hash_u64(top_set_hash, pair_key);
                }
                max6_observed_scan_timing.exact_topk_set_hash_by_window_order =
                    max6_diagnostic_hash_u64(
                        max6_observed_scan_timing.exact_topk_set_hash_by_window_order,
                        top_set_hash,
                    );
                max6_observed_scan_timing.exact_topk_order_hash_by_window_order =
                    max6_diagnostic_hash_u64(
                        max6_observed_scan_timing.exact_topk_order_hash_by_window_order,
                        top_hash,
                    );
                max6_observed_scan_timing.candidate_hash_window_count = max6_observed_scan_timing
                    .candidate_hash_window_count
                    .saturating_add(1);

                candidate_window_pair_counts.push(
                    unit_result
                        .timing
                        .exact_candidates
                        .saturating_add(unit_result.timing.grm_int_candidates),
                );
                candidate_window_screen_pairs_evaluated
                    .push(unit_result.timing.screen_pairs_evaluated);
                candidate_window_endpoint_counts
                    .push(unit_result.window_unique_candidate_endpoints);
                let next_key_bytes = candidate_pair_keys
                    .len()
                    .saturating_add(unit_result.candidate_pair_keys.len())
                    .saturating_mul(std::mem::size_of::<u64>());
                let next_score_bytes = candidate_score_observations
                    .len()
                    .saturating_add(unit_result.candidate_score_observations.len())
                    .saturating_mul(std::mem::size_of::<Max6CandidateScoreObservation>());
                if next_key_bytes.saturating_add(next_score_bytes)
                    > MAX_REUSE_DIAGNOSTIC_BUFFER_BYTES
                {
                    return Err(
                        "Max6 reuse diagnostic exceeded its 1 GiB pair/score buffer safety cap"
                            .to_string(),
                    );
                }
                candidate_pair_keys.extend(unit_result.candidate_pair_keys);
                candidate_score_observations.extend(unit_result.candidate_score_observations);
            }
            output_count = output_count.saturating_add(unit_result.outputs.len());
            max6_screen_candidate_count =
                max6_screen_candidate_count.saturating_add(unit_result.screen_pairs_passed);
            max6_final_screen_subset_audit_pairs_checked =
                max6_final_screen_subset_audit_pairs_checked
                    .saturating_add(unit_result.final_screen_subset_audit_pairs_checked);
            if let Some(writer) = pair_spool.as_mut() {
                for record in unit_result.outputs.iter() {
                    let four_cell_counts = match record.four_cell_counts {
                        Some(counts) => counts,
                        None => max6_pair_four_cell_counts(
                            logic_bits,
                            record.first_global_row,
                            record.second_global_row,
                            train_idx_local,
                        )?,
                    };
                    write_max6_grmmax6_pair_spool_record(
                        writer,
                        record,
                        logic_bits.sites.as_slice(),
                        four_cell_counts,
                        grm_int_only,
                    )?;
                    if let Some(provenance_writer) = pair_provenance_spool.as_mut() {
                        let unit = units.get(record.unit_index).ok_or_else(|| {
                            format!(
                                "Max6 GRM-INT output references missing unit index {}",
                                record.unit_index
                            )
                        })?;
                        provenance_output_row_index = provenance_output_row_index.saturating_add(1);
                        write_max6_grm_int_provenance_spool_record(
                            provenance_writer,
                            provenance_output_row_index,
                            record.unit_index,
                            unit.label.as_str(),
                        )?;
                    }
                }
            }
            if let Some(pool) = unit_result.permutation_pool {
                if let Some(writer) = permutation_spool.as_mut() {
                    permutation_pool_meta.push(write_max6_grmmax6_permutation_rows(
                        writer,
                        &mut permutation_spool_offset,
                        pool.unit_index,
                        pool.selected_global_rows.as_slice(),
                    )?);
                } else {
                    return Err(
                        "Max6 permutation pool exists but permutation spool is unavailable"
                            .to_string(),
                    );
                }
            }
            if let Some(message) = unit_result.skipped_message {
                skipped_messages.push(message);
            }
            done += 1;
            if let Some(callback) = progress_callback {
                if progress_every == 0
                    || done % progress_every.max(1) == 0
                    || done == scan_unit_indices.len()
                {
                    let unit_label = units.get(*unit_index).map(|unit| unit.label.as_str());
                    garfield_stage_progress_notify(
                        Some(callback),
                        "max6_grmmax6_scan",
                        done,
                        scan_unit_indices.len().max(1),
                        unit_label,
                    )
                    .map_err(|error| error.to_string())?;
                }
            }
        }
        max6_observed_scan_timing.scan_result_drain_wall_seconds +=
            result_drain_started.elapsed().as_secs_f64();
    }

    if cross_window_union_rerank {
        if cross_window_parity_audit {
            let parity = cross_window_batch_summary.parity_audit;
            eprintln!(
                "[MAX6-CROSS-WINDOW-PARITY] channels=[AND00,AND01,AND10,AND11,XOR,DOSAGE,MAX6] windows={} pairs={} max_abs={:?} max_rel={:?} pairs_abs_gt_1e-12={:?} finite_mask_mismatch_pairs={} best_channel_mismatch_pairs={} rank_changed_pairs={} max_rank_displacement={} topk_set_symmetric_difference=[1:{},10:{},100:{},500:{}] topk_order_changed_windows=[1:{},10:{},100:{},500:{}]",
                parity.windows_checked,
                parity.pairs_checked,
                parity.max_abs_by_channel,
                parity.max_rel_by_channel,
                parity.abs_gt_1e_12_by_channel,
                parity.finite_mask_mismatch_pairs,
                parity.best_channel_mismatch_pairs,
                parity.rank_changed_pairs,
                parity.max_rank_displacement,
                parity.topk_set_symmetric_difference[0],
                parity.topk_set_symmetric_difference[1],
                parity.topk_set_symmetric_difference[2],
                parity.topk_set_symmetric_difference[3],
                parity.topk_order_changed_windows[0],
                parity.topk_order_changed_windows[1],
                parity.topk_order_changed_windows[2],
                parity.topk_order_changed_windows[3],
            );
        }
    }

    if candidate_reuse_diagnostic {
        finalize_max6_candidate_reuse_diagnostic(
            &mut max6_observed_scan_timing,
            &mut candidate_pair_keys,
            candidate_window_pair_counts.as_mut_slice(),
            candidate_window_screen_pairs_evaluated.as_mut_slice(),
            candidate_window_endpoint_counts.as_mut_slice(),
            logic_bits.sites.len(),
        )?;
        // The summary is complete; release the temporary ~120 MB pair-key
        // buffer before permutation/output finalization continues.
        drop(candidate_pair_keys);
    }
    if candidate_score_reuse_audit {
        if candidate_score_observations.len() != max6_observed_scan_timing.scored_candidate_pairs {
            return Err(format!(
                "Max6 score reuse audit collected {} scores but Max6/GRM-INT scoring processed {} pairs",
                candidate_score_observations.len(),
                max6_observed_scan_timing.scored_candidate_pairs
            ));
        }
        let score_audit_started = Instant::now();
        max6_observed_scan_timing.candidate_score_reuse_summary =
            Some(summarize_max6_candidate_score_reuse_observations(
                candidate_score_observations.as_mut_slice(),
            ));
        max6_observed_scan_timing.candidate_score_audit_buffer_bytes = candidate_score_observations
            .capacity()
            * std::mem::size_of::<Max6CandidateScoreObservation>();
        max6_observed_scan_timing.candidate_score_audit_finalize_wall_seconds =
            score_audit_started.elapsed().as_secs_f64();
        drop(candidate_score_observations);
    }

    let mut max6_permutation_timing = Max6PermutationTiming::default();
    let mut max6_permutation_pair_pool_size = 0usize;
    let mut max6_permutation_pairs_per_repeat = 0usize;
    let mut max6_permutation_pair_draw_count = 0usize;
    let mut max6_gumbel_score_threshold = (!grm_int_only)
        .then(|| {
            max6_screen_calibration
                .as_ref()
                .and_then(|calibration| calibration.gumbel_threshold)
        })
        .flatten();
    let mut max6_permutation_maxima_tsv = None;
    let (score_threshold, permutation_score_maxima) = if max6_permutation_active
        && !permutation_pool_meta.is_empty()
    {
        let permutation_setup_started = Instant::now();
        if let Some(mut writer) = permutation_spool.take() {
            writer.flush().map_err(|error| error.to_string())?;
        }
        let mut total_pairs = 0usize;
        let mut offsets = Vec::<usize>::with_capacity(permutation_pool_meta.len());
        for pool in permutation_pool_meta.iter() {
            offsets.push(total_pairs);
            let n = pool.n_rows;
            total_pairs = total_pairs.saturating_add(n.saturating_mul(n.saturating_sub(1)) / 2);
        }
        max6_permutation_pair_pool_size = total_pairs;
        let sample_count = MAX6_GRMMAX6_PERMUTATION_PAIRS.min(total_pairs);
        max6_permutation_pairs_per_repeat = sample_count;
        max6_permutation_pair_draw_count = sample_count.saturating_mul(permutation_repeats);
        let mut jobs = Vec::<Max6GrmMax6PermutationJob>::with_capacity(permutation_repeats);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xC7A4_6D2F_91B8_5301);
        let mut requests = HashMap::<usize, Vec<(usize, usize, usize)>>::new();
        for _repeat in 0..permutation_repeats {
            let mut permuted_y = y.to_vec();
            for index in (1..permuted_y.len()).rev() {
                let swap_index = rng.random_range(0..=index);
                permuted_y.swap(index, swap_index);
            }
            let repeat = jobs.len();
            for _ in 0..sample_count {
                let ordinal = rng.random_range(0..total_pairs);
                let pool_index = offsets
                    .partition_point(|&offset| offset <= ordinal)
                    .saturating_sub(1);
                let pool = &permutation_pool_meta[pool_index];
                let n_pool = pool.n_rows;
                let n_pairs = n_pool.saturating_mul(n_pool.saturating_sub(1)) / 2;
                let (left, right) =
                    unrank_upper_pair((ordinal - offsets[pool_index]) % n_pairs, n_pool);
                requests
                    .entry(pool_index)
                    .or_default()
                    .push((repeat, left, right));
            }
            jobs.push(Max6GrmMax6PermutationJob {
                permuted_y,
                grouped: HashMap::new(),
            });
        }
        let mut requested_pool_indices = requests.keys().copied().collect::<Vec<_>>();
        requested_pool_indices.sort_unstable();
        if let Some(spool_path) = permutation_spool_path.as_ref() {
            let mut reader = File::open(spool_path).map_err(|error| error.to_string())?;
            for pool_index in requested_pool_indices {
                let pool = &permutation_pool_meta[pool_index];
                let selected_global_rows = read_max6_grmmax6_permutation_rows(&mut reader, pool)?;
                let Some(pool_requests) = requests.get(&pool_index) else {
                    continue;
                };
                for &(repeat, left, right) in pool_requests {
                    let left_global = *selected_global_rows.get(left).ok_or_else(|| {
                        format!(
                            "Max6 permutation left index out of range: pool={}, left={}, n_rows={}",
                            pool_index,
                            left,
                            selected_global_rows.len()
                        )
                    })?;
                    let right_global = *selected_global_rows.get(right).ok_or_else(|| {
                            format!(
                                "Max6 permutation right index out of range: pool={}, right={}, n_rows={}",
                                pool_index,
                                right,
                                selected_global_rows.len()
                            )
                        })?;
                    let entry = jobs
                        .get_mut(repeat)
                        .ok_or_else(|| format!("Max6 permutation repeat out of range: {repeat}"))?
                        .grouped
                        .entry(pool.unit_index)
                        .or_default();
                    entry.0.push(left_global);
                    entry.1.push(right_global);
                }
            }
        }
        max6_permutation_timing.setup_wall_seconds =
            permutation_setup_started.elapsed().as_secs_f64();
        let permutation_score_started = Instant::now();
        let job_results = run_max6_permutation_jobs_batched(
            jobs.as_slice(),
            outer_pool.as_ref(),
            |job| {
                max6_grmmax6_permutation_max_score(
                    job,
                    logic_bits,
                    p_c,
                    no_grm_projector,
                    score_scale,
                )
            },
            |done, total| {
                let detail = format!("{done}/{total} permutations");
                garfield_stage_progress_notify(
                    progress_callback,
                    "max6_grmmax6_permutation",
                    done,
                    total,
                    Some(detail.as_str()),
                )
                .map_err(|error| error.to_string())
            },
        )?;
        max6_permutation_timing.score_wall_seconds =
            permutation_score_started.elapsed().as_secs_f64();
        for result in job_results.iter() {
            max6_permutation_timing.accumulate_job(result.timing);
        }
        let raw_maxima = job_results
            .iter()
            .map(|result| result.max_score)
            .collect::<Vec<_>>();
        let maxima = raw_maxima.iter().flatten().copied().collect::<Vec<_>>();
        let maxima_write_started = Instant::now();
        if diagnostics_enabled {
            if let Some(prefix) = out_prefix {
                let path = format!("{prefix}.max6_grmmax6.permutation_maxima.tsv");
                let mut writer =
                    BufWriter::new(File::create(path.as_str()).map_err(|error| error.to_string())?);
                writeln!(writer, "repeat\tmax_score\tfinite").map_err(|error| error.to_string())?;
                for (repeat, score) in raw_maxima.iter().enumerate() {
                    match score {
                        Some(value) if value.is_finite() => {
                            writeln!(writer, "{repeat}\t{value:.17}\ttrue")
                                .map_err(|error| error.to_string())?;
                        }
                        _ => {
                            writeln!(writer, "{repeat}\t\tfalse")
                                .map_err(|error| error.to_string())?;
                        }
                    }
                }
                writer.flush().map_err(|error| error.to_string())?;
                max6_permutation_maxima_tsv = Some(path);
            }
        }
        max6_permutation_timing.maxima_write_wall_seconds =
            maxima_write_started.elapsed().as_secs_f64();
        let threshold_started = Instant::now();
        let thresholds =
            max6_permutation_threshold_candidates(maxima.as_slice(), rule_null_quantile);
        max6_gumbel_score_threshold = thresholds.gumbel;
        max6_permutation_timing.threshold_wall_seconds = threshold_started.elapsed().as_secs_f64();
        (thresholds.empirical, maxima)
    } else {
        (
            if grm_int_only {
                None
            } else if ld_tag_exhaustive && max6_screen_calibration.is_none() {
                max6_bonferroni_q_cut
            } else {
                screen_score_threshold
            },
            Vec::new(),
        )
    };
    if let Some(mut writer) = permutation_spool.take() {
        writer.flush().map_err(|error| error.to_string())?;
    }
    if let Some(path) = permutation_spool_path.as_ref() {
        fs::remove_file(path).map_err(|error| {
            format!(
                "failed to remove Max6 permutation spool '{}': {error}",
                path
            )
        })?;
    }
    let mut max6_pairs_tsv = None;
    let mut max6_pairs_provenance_tsv = None;
    let max6_ld_tags_tsv = None;
    let max6_threshold_tsv = None;
    if let Some(prefix) = out_prefix {
        let pair_path = max6_grmmax6_pairs_output_path(prefix);
        let spool_path = pair_spool_path.as_ref().ok_or_else(|| {
            "internal error: Max6 pair spool path is missing despite output prefix".to_string()
        })?;
        if let Some(mut writer) = pair_spool.take() {
            writer.flush().map_err(|error| error.to_string())?;
        }
        write_max6_grmmax6_pairs_tsv(pair_path.as_str(), spool_path, grm_int_only)?;
        fs::remove_file(spool_path).map_err(|error| {
            format!("failed to remove Max6 pair spool '{}': {error}", spool_path)
        })?;
        if grm_int_only && write_provenance {
            let provenance_spool_path = pair_provenance_spool_path.as_ref().ok_or_else(|| {
                "internal error: Max6 GRM-INT provenance spool path is missing".to_string()
            })?;
            if let Some(mut writer) = pair_provenance_spool.take() {
                writer.flush().map_err(|error| error.to_string())?;
            }
            let provenance_path = max6_grm_int_provenance_output_path(prefix);
            write_max6_grm_int_provenance_tsv(provenance_path.as_str(), provenance_spool_path)?;
            fs::remove_file(provenance_spool_path).map_err(|error| {
                format!(
                    "failed to remove Max6 provenance spool '{}': {error}",
                    provenance_spool_path
                )
            })?;
            max6_pairs_provenance_tsv = Some(provenance_path);
        } else {
            remove_stale_max6_grm_int_provenance(prefix, false)?;
        }
        let threshold_path = format!("{prefix}.max6_grmmax6.threshold.tsv");
        if Path::new(threshold_path.as_str()).exists() {
            fs::remove_file(threshold_path.as_str()).map_err(|error| {
                format!(
                    "failed to remove legacy Max6 threshold file '{}': {error}",
                    threshold_path
                )
            })?;
        }
        max6_pairs_tsv = Some(pair_path);
    }

    let timing_scan_wall_s = scan_t0.elapsed().as_secs_f64();
    let timing_total_wall_s = total_wall_t0.elapsed().as_secs_f64();
    Ok(GarfieldLogicPipelineResult {
        pseudo_prefix: None,
        rules_tsv: None,
        diagnostics_tsv: None,
        frontier_diagnostics_tsv: None,
        rules_compare_tsv: None,
        posterior_json: None,
        memory_debug: None,
        bg_noise_summary: None,
        scheduler_scan: None,
        scheduler_permutation: None,
        rule_permutation_active: max6_permutation_active,
        null_chunk_bp: 0,
        null_chunk_min_snps: 0,
        null_chunk_target: 0,
        null_chunk_valid_total: 0,
        null_chunk_selected: 0,
        representative_units_target: 0,
        representative_units_used: 0,
        permutation_null_repeats: if max6_permutation_active {
            permutation_repeats
        } else {
            0
        },
        permutation_bootstrap_repeats: 0,
        records: Vec::new(),
        simbench_count: 0,
        n_samples: logic_bits.n_samples,
        logic_bits_backend: if logic_bits.bits_mmap.is_some() {
            "mmap".to_string()
        } else {
            "resident".to_string()
        },
        logic_bits_bytes: logic_bits.bits().len() * std::mem::size_of::<u64>(),
        units_total: units.len(),
        units_scanned: scan_unit_indices.len(),
        timing_total_wall_s,
        timing_scan_wall_s,
        max6_observed_scan_timing,
        timing_max6_screen_calibration_wall_s,
        timing_max6_screen_validation_wall_s,
        timing_max6_screen_sampled_diagnostic_wall_s,
        timing_max6_screen_tail_audit_wall_s,
        timing_max6_screen_exceedance_audit_wall_s,
        timing_max6_screen_pot_audit_wall_s,
        timing_scan_beam_wall_s: 0.0,
        timing_scan_literal_score_wall_s: 0.0,
        timing_scan_ml_select_wall_s: 0.0,
        timing_corr_stage1_summary_s: 0.0,
        timing_corr_stage1_score_s: 0.0,
        timing_corr_stage1_pool_s: 0.0,
        timing_corr_stage1_pool_priority_s: 0.0,
        timing_corr_stage1_pool_ld_s: 0.0,
        timing_ld_support_cache_s: 0.0,
        ld_support_cache_rows: 0,
        ld_support_cache_payload_bytes: 0,
        timing_scan_beam_calls: 0,
        timing_clone_bits_s: 0.0,
        timing_sum_y_both1_s: 0.0,
        timing_parent_baseline_s: 0.0,
        timing_pw_marginal_s: 0.0,
        timing_pw_pack_s: 0.0,
        timing_pw_kernel_s: 0.0,
        timing_pw_combine_s: 0.0,
        timing_dense_extract_s: 0.0,
        timing_dense_decode_s: 0.0,
        timing_geneset_ld_prune_s: 0.0,
        ld_exact_pairs: 0,
        ld_rows_total: 0,
        ld_rows_kept: 0,
        ld_clump_rows_compressed: 0,
        ld_clump_units: 0,
        ld_units_eligible: 0,
        ld_units_pruned: 0,
        ld_rows_pruned: 0,
        ld_unit_rows_total_median: 0,
        ld_unit_rows_total_p95: 0,
        ld_unit_rows_total_max: 0,
        ld_unit_rows_kept_median: 0,
        ld_unit_rows_kept_p95: 0,
        ld_unit_rows_kept_max: 0,
        ld_unit_rows_pruned_median: 0,
        ld_unit_rows_pruned_p95: 0,
        ld_unit_rows_pruned_max: 0,
        ld_unit_exact_pairs_median: 0,
        ld_unit_exact_pairs_p95: 0,
        ld_unit_exact_pairs_max: 0,
        timing_materialize_bits_s: 0.0,
        timing_literal_score_share_of_total_pct: 0.0,
        timing_literal_score_share_of_scan_pct: 0.0,
        timing_literal_score_share_of_beam_pct: 0.0,
        timing_beam_share_of_total_pct: 0.0,
        timing_beam_share_of_scan_pct: 0.0,
        skipped_units: Vec::new(),
        skipped_messages,
        train_fit,
        max6_pairs_tsv,
        max6_pairs_provenance_tsv,
        max6_ld_tags_tsv,
        max6_threshold_tsv,
        max6_screen_maxima_tsv,
        max6_screen_validation_maxima_tsv,
        max6_screen_sampled_maxima_tsv: max6_screen_sampled_diagnostic
            .as_ref()
            .map(|diagnostic| diagnostic.maxima_tsv.clone()),
        max6_screen_sample_manifest_json: max6_screen_sampled_diagnostic
            .as_ref()
            .map(|diagnostic| diagnostic.manifest_json.clone()),
        max6_screen_sample_strata_tsv: max6_screen_sampled_diagnostic
            .as_ref()
            .map(|diagnostic| diagnostic.strata_tsv.clone()),
        max6_screen_full_maximizer_geometry_tsv: max6_screen_sampled_diagnostic
            .as_ref()
            .map(|diagnostic| diagnostic.full_maximizer_geometry_tsv.clone()),
        max6_screen_tail_top_pairs_tsv: max6_screen_tail_audit
            .as_ref()
            .map(|paths| paths.top_pairs_tsv.clone()),
        max6_screen_tail_sampled_ranks_tsv: max6_screen_tail_audit
            .as_ref()
            .map(|paths| paths.sampled_ranks_tsv.clone()),
        max6_screen_tail_summary_json: max6_screen_tail_audit
            .as_ref()
            .map(|paths| paths.summary_json.clone()),
        max6_screen_tail_support_aware_maxima_tsv: max6_screen_tail_audit
            .as_ref()
            .and_then(|paths| paths.support_aware_maxima_tsv.clone()),
        max6_screen_tail_support_aware_strata_tsv: max6_screen_tail_audit
            .as_ref()
            .and_then(|paths| paths.support_aware_strata_tsv.clone()),
        max6_screen_tail_support_aware_summary_json: max6_screen_tail_audit
            .as_ref()
            .and_then(|paths| paths.support_aware_summary_json.clone()),
        max6_screen_exceedance_audit_per_null_tsv: max6_screen_exceedance_audit_result
            .as_ref()
            .map(|paths| paths.per_null_tsv.clone()),
        max6_screen_exceedance_audit_summary_tsv: max6_screen_exceedance_audit_result
            .as_ref()
            .map(|paths| paths.summary_tsv.clone()),
        max6_screen_exceedance_audit_top_pairs_tsv: max6_screen_exceedance_audit_result
            .as_ref()
            .map(|paths| paths.top_pairs_tsv.clone()),
        max6_screen_exceedance_audit_manifest_json: max6_screen_exceedance_audit_result
            .as_ref()
            .map(|paths| paths.manifest_json.clone()),
        max6_screen_pot_audit_manifest_json: max6_screen_pot_audit_result
            .as_ref()
            .map(|paths| paths.manifest_json.clone()),
        max6_screen_pot_sample_scores_manifest_json: max6_screen_pot_audit_result
            .as_ref()
            .map(|paths| paths.sample_scores_manifest_json.clone()),
        max6_permutation_maxima_tsv,
        max6_pair_pve_total_quadratic,
        max6_pair_universe_size,
        max6_bonferroni_q_cut,
        max6_bonferroni_p_cut,
        max6_score_threshold: score_threshold,
        max6_gumbel_score_threshold,
        max6_screen_threshold: screen_score_threshold,
        max6_screen_threshold_method: max6_screen_calibration.as_ref().map(|_| {
            "conditional_parametric_null_empirical_maxT_order_statistic_weak_global_null"
                .to_string()
        }),
        max6_screen_null_repeats: max6_screen_calibration
            .as_ref()
            .map(|calibration| calibration.maxima.len())
            .unwrap_or(0),
        max6_screen_validation_repeats: max6_screen_validation
            .as_ref()
            .map(|validation| validation.maxima.len())
            .unwrap_or(0),
        max6_screen_pair_universe_size: max6_pair_universe_size,
        max6_screen_window_pair_occurrences: max6_screen_universe
            .as_ref()
            .map(|universe| universe.window_pair_count)
            .unwrap_or(0),
        max6_screen_candidate_count,
        max6_screen_validation_rejections: max6_screen_validation
            .as_ref()
            .map(|validation| validation.rejections)
            .unwrap_or(0),
        max6_screen_validation_fwer: max6_screen_validation
            .as_ref()
            .map(|validation| validation.empirical_fwer),
        max6_screen_validation_ci_lower: max6_screen_validation
            .as_ref()
            .map(|validation| validation.ci_lower),
        max6_screen_validation_ci_upper: max6_screen_validation
            .as_ref()
            .map(|validation| validation.ci_upper),
        max6_final_screen_subset_audit_passed: Some(true),
        max6_final_screen_subset_audit_pairs_checked,
        max6_final_screen_subset_audit_violations: 0,
        max6_score_scale: Some(score_scale),
        max6_permutation_score_count: permutation_score_maxima.len(),
        max6_permutation_pair_pool_size,
        max6_permutation_pairs_per_repeat,
        max6_permutation_pair_draw_count,
        max6_permutation_timing,
        max6_pair_count: output_count,
    })
}
