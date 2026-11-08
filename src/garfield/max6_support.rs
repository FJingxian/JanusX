use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Max6GrmMax6LdPreparedUnit {
    pub(super) selected_global_rows: Vec<usize>,
    pub(super) pair_groups: Option<Vec<usize>>,
    pub(super) ld_compression: Vec<GarfieldLdCompressionRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Max6PreparedFileFingerprint {
    pub(super) path: String,
    pub(super) length: u64,
    pub(super) modified_unix_nanos: u128,
}

impl Max6PreparedFileFingerprint {
    pub(super) fn from_path(path: &Path) -> Result<Self, String> {
        let canonical_path = fs::canonicalize(path).map_err(|error| {
            format!(
                "unable to resolve prepared Max6 source '{}': {error}",
                path.display()
            )
        })?;
        let metadata = fs::metadata(canonical_path.as_path()).map_err(|error| {
            format!(
                "unable to inspect prepared Max6 source '{}': {error}",
                canonical_path.display()
            )
        })?;
        let modified_unix_nanos = metadata
            .modified()
            .map_err(|error| {
                format!(
                    "unable to read modification time for prepared Max6 source '{}': {error}",
                    canonical_path.display()
                )
            })?
            .duration_since(UNIX_EPOCH)
            .map_err(|error| {
                format!(
                    "prepared Max6 source '{}' has a modification time before UNIX epoch: {error}",
                    canonical_path.display()
                )
            })?
            .as_nanos();
        Ok(Self {
            path: canonical_path.to_string_lossy().into_owned(),
            length: metadata.len(),
            modified_unix_nanos,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Max6PreparedInputFingerprint {
    pub(super) files: Vec<Max6PreparedFileFingerprint>,
    pub(super) ordered_sample_ids: Vec<String>,
    pub(super) config_fingerprint: u64,
}

impl Max6PreparedInputFingerprint {
    pub(super) fn ensure_compatible(&self, current: &Self) -> Result<(), String> {
        if self.files != current.files {
            return Err("prepared Max6 universe source files have changed".to_string());
        }
        if self.ordered_sample_ids != current.ordered_sample_ids {
            return Err("prepared Max6 universe sample IDs or order have changed".to_string());
        }
        if self.config_fingerprint != current.config_fingerprint {
            return Err("prepared Max6 universe scan/QC parameters have changed".to_string());
        }
        Ok(())
    }
}

pub(super) fn max6_preparation_config_fingerprint<T: Hash>(config: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    config.hash(&mut hasher);
    hasher.finish()
}

#[derive(Clone, Debug, Hash)]
pub(super) struct Max6PreparationConfigFingerprint {
    pub(super) unit_kind: String,
    pub(super) groups: Option<Vec<Vec<(String, i32, i32)>>>,
    pub(super) group_names: Option<Vec<String>>,
    pub(super) extension: usize,
    pub(super) step: Option<usize>,
    pub(super) scan_bimranges: Option<Vec<String>>,
    pub(super) sample_indices: Vec<usize>,
    pub(super) maf_threshold_bits: u32,
    pub(super) max_missing_rate_bits: u32,
    pub(super) het_threshold_bits: u32,
    pub(super) snps_only: bool,
    pub(super) ld_r2_threshold_bits: u64,
}

impl Max6PreparationConfigFingerprint {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        unit_kind: &str,
        groups: Option<&[Vec<(String, i32, i32)>]>,
        group_names: Option<&[String]>,
        extension: usize,
        step: Option<usize>,
        scan_bimranges: Option<&[String]>,
        sample_indices: &[usize],
        maf_threshold: f32,
        max_missing_rate: f32,
        het_threshold: f32,
        snps_only: bool,
        ld_tag_r2: f64,
    ) -> Self {
        Self {
            unit_kind: unit_kind.trim().to_ascii_lowercase(),
            groups: groups.map(<[_]>::to_vec),
            group_names: group_names.map(<[_]>::to_vec),
            extension,
            step,
            scan_bimranges: scan_bimranges.map(<[_]>::to_vec),
            sample_indices: sample_indices.to_vec(),
            maf_threshold_bits: maf_threshold.to_bits(),
            max_missing_rate_bits: max_missing_rate.to_bits(),
            het_threshold_bits: het_threshold.to_bits(),
            snps_only,
            ld_r2_threshold_bits: ld_tag_r2.to_bits(),
        }
    }
}

pub(super) fn build_max6_prepared_input_fingerprint(
    prefix: &str,
    ordered_sample_ids: Vec<String>,
    config: &Max6PreparationConfigFingerprint,
) -> Result<Max6PreparedInputFingerprint, String> {
    let base = normalize_plink_prefix(prefix);
    let input_extensions = if crate::pgen::path_uses_pgen(base.as_str()) {
        ["pgen", "pvar", "psam"]
    } else {
        ["bed", "bim", "fam"]
    };
    let files = input_extensions
        .into_iter()
        .map(|extension| {
            Max6PreparedFileFingerprint::from_path(Path::new(
                format!("{base}.{extension}").as_str(),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Max6PreparedInputFingerprint {
        files,
        ordered_sample_ids,
        config_fingerprint: max6_preparation_config_fingerprint(config),
    })
}

#[derive(Clone, Debug)]
pub(super) struct Max6PreparedUnitMetadata {
    pub(super) original_unit_index: usize,
    pub(super) label: String,
    pub(super) spans: Vec<GarfieldUnitSpan>,
}

#[derive(Debug)]
pub(super) struct Max6PreparedGenotypeTags {
    pub(super) input_fingerprint: Max6PreparedInputFingerprint,
    pub(super) unit_kind: String,
    pub(super) units: Vec<Max6PreparedUnitMetadata>,
    pub(super) screen_units: Vec<Max6GrmMax6LdPreparedUnit>,
    pub(super) original_unit_indices: Vec<usize>,
    pub(super) active_source_rows: Vec<usize>,
    pub(super) active_source_sites: Arc<CompactMax6SiteMetadata>,
    pub(super) source_marker_count: usize,
    pub(super) window_pair_count: usize,
    pub(super) global_site_keep: Vec<bool>,
}

#[derive(Clone, Debug)]
pub(super) struct Max6PreparedGenotypeUniverse {
    pub(super) tags: Arc<Max6PreparedGenotypeTags>,
    pub(super) pair_universe_size: usize,
    pub(super) bonferroni_q_cut: f64,
    pub(super) bonferroni_p_cut: f64,
}

impl Max6PreparedGenotypeUniverse {
    pub(super) fn new(
        tags: Arc<Max6PreparedGenotypeTags>,
        pair_universe_size: usize,
    ) -> Result<Self, String> {
        if tags.active_source_rows.len() < 2 || pair_universe_size == 0 {
            return Err(
                "Max6 prepared universe requires a nonempty unique-pair universe".to_string(),
            );
        }
        if tags.units.len() != tags.screen_units.len()
            || tags.original_unit_indices.len() != tags.screen_units.len()
            || tags.active_source_sites.len() != tags.active_source_rows.len()
            || tags
                .active_source_rows
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err("prepared Max6 unit/tag metadata is internally inconsistent".to_string());
        }
        let (bonferroni_q_cut, bonferroni_p_cut) = max6_bonferroni_cutoff(pair_universe_size)?;
        Ok(Self {
            tags,
            pair_universe_size,
            bonferroni_q_cut,
            bonferroni_p_cut,
        })
    }

    pub(super) fn ensure_compatible_input(
        &self,
        current: &Max6PreparedInputFingerprint,
    ) -> Result<(), String> {
        self.tags.input_fingerprint.ensure_compatible(current)
    }
}

/// Prepare the deterministic LD-tag pair universe used by the production
/// Max6 screen.  The same helper is used by null calibration and observed
/// scanning so the calibrated threshold and the final discovery set refer to
/// exactly the same marker identities and cross-window exclusions.
pub(super) fn prepare_max6_grmmax6_ld_unit(
    unit: &GarfieldLogicUnit,
    unit_kind_lc: &str,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
) -> Result<Max6GrmMax6LdPreparedUnit, String> {
    prepare_max6_grmmax6_ld_unit_with_support_cache(
        unit,
        unit_kind_lc,
        logic_bits,
        train_idx_local,
        None,
        GARFIELD_MAX6_LD_TAG_R2_DEFAULT,
    )
}

pub(super) fn prepare_max6_grmmax6_ld_unit_with_support_cache(
    unit: &GarfieldLogicUnit,
    unit_kind_lc: &str,
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    support_cache: Option<&GarfieldLdSupportCache>,
    ld_tag_r2: f64,
) -> Result<Max6GrmMax6LdPreparedUnit, String> {
    let mut candidate_global_rows = unit.indices.clone();
    candidate_global_rows.sort_unstable();
    candidate_global_rows.dedup();
    let group_ids = if unit_kind_lc == "geneset" && unit.spans.len() > 1 {
        build_unit_window_group_ids(
            unit,
            candidate_global_rows.as_slice(),
            logic_bits.sites.as_slice(),
            unit_kind_lc,
        )
    } else {
        None
    };
    let selection = select_max6_ld_tag_rows_impl(
        candidate_global_rows.as_slice(),
        logic_bits,
        train_idx_local,
        group_ids.as_deref(),
        ld_tag_r2,
        false,
        support_cache,
    )?;
    let GarfieldLdSelection {
        selected_global_rows,
        compressed: ld_compression,
    } = selection;
    let selected_group_ids = group_ids.map(|group_ids| {
        let mut selected_group_ids = Vec::with_capacity(selected_global_rows.len());
        for &global_row in selected_global_rows.iter() {
            let local_idx = candidate_global_rows
                .binary_search(&global_row)
                .expect("selected Max6 LD tag row must be in candidate rows");
            selected_group_ids.push(group_ids[local_idx]);
        }
        selected_group_ids
    });
    Ok(Max6GrmMax6LdPreparedUnit {
        selected_global_rows,
        pair_groups: selected_group_ids,
        ld_compression,
    })
}

pub(super) struct Max6GrmMax6PermutationJob {
    pub(super) permuted_y: Vec<f64>,
    pub(super) grouped: HashMap<usize, (Vec<usize>, Vec<usize>)>,
}

pub(super) struct Max6PermutationJobResult {
    pub(super) max_score: Option<f64>,
    pub(super) timing: Max6PermutationJobTiming,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Max6PermutationJobTiming {
    pub(super) projected_y_seconds: f64,
    pub(super) endpoint_prepare_seconds: f64,
    pub(super) rerank: CandidateRerankTiming,
    pub(super) pair_scores: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Max6PermutationTiming {
    pub(super) setup_wall_seconds: f64,
    pub(super) score_wall_seconds: f64,
    pub(super) threshold_wall_seconds: f64,
    pub(super) maxima_write_wall_seconds: f64,
    pub(super) projected_y_cpu_seconds: f64,
    pub(super) endpoint_prepare_cpu_seconds: f64,
    pub(super) rerank_kernel_cpu_seconds: f64,
    pub(super) rerank_pcx_cpu_seconds: f64,
    pub(super) rerank_geometry_cpu_seconds: f64,
    pub(super) rerank_q_cpu_seconds: f64,
    pub(super) rerank_score_cpu_seconds: f64,
    pub(super) rerank_topk_cpu_seconds: f64,
    pub(super) rerank_other_cpu_seconds: f64,
    pub(super) pair_score_count: usize,
}

impl Max6PermutationTiming {
    pub(super) fn accumulate_job(&mut self, job: Max6PermutationJobTiming) {
        self.projected_y_cpu_seconds += job.projected_y_seconds;
        self.endpoint_prepare_cpu_seconds += job.endpoint_prepare_seconds;
        self.rerank_kernel_cpu_seconds += job.rerank.kernel_seconds;
        self.rerank_pcx_cpu_seconds += job.rerank.pcx_u_seconds;
        self.rerank_geometry_cpu_seconds += job.rerank.candidate_geometry_seconds;
        self.rerank_q_cpu_seconds += job.rerank.q_seconds;
        self.rerank_score_cpu_seconds += job.rerank.score_seconds;
        self.rerank_topk_cpu_seconds += job.rerank.topk_seconds;
        let measured_components = job.rerank.endpoint_prepare_seconds
            + job.rerank.pcx_u_seconds
            + job.rerank.py_seconds
            + job.rerank.candidate_geometry_seconds
            + job.rerank.q_seconds
            + job.rerank.score_seconds
            + job.rerank.topk_seconds;
        self.rerank_other_cpu_seconds += (job.rerank.kernel_seconds - measured_components).max(0.0);
        self.pair_score_count = self.pair_score_count.saturating_add(job.pair_scores);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Max6PermutationThresholdCandidates {
    pub(super) empirical: Option<f64>,
    pub(super) gumbel: Option<f64>,
}

pub(super) fn write_max6_grmmax6_permutation_rows(
    writer: &mut BufWriter<File>,
    byte_offset: &mut u64,
    unit_index: usize,
    selected_global_rows: &[usize],
) -> Result<Max6GrmMax6PermutationPoolMeta, String> {
    let start = *byte_offset;
    for &row in selected_global_rows {
        let row = u32::try_from(row).map_err(|_| {
            format!(
                "Max6 permutation marker index exceeds u32 storage: unit={unit_index}, row={row}"
            )
        })?;
        writer
            .write_all(&row.to_le_bytes())
            .map_err(|error| error.to_string())?;
        *byte_offset = (*byte_offset).saturating_add(4);
    }
    Ok(Max6GrmMax6PermutationPoolMeta {
        unit_index,
        byte_offset: start,
        n_rows: selected_global_rows.len(),
    })
}

pub(super) fn read_max6_grmmax6_permutation_rows(
    reader: &mut File,
    meta: &Max6GrmMax6PermutationPoolMeta,
) -> Result<Vec<usize>, String> {
    reader
        .seek(SeekFrom::Start(meta.byte_offset))
        .map_err(|error| error.to_string())?;
    let byte_len = meta
        .n_rows
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "Max6 permutation row spool length overflow".to_string())?;
    let mut bytes = vec![0u8; byte_len];
    reader
        .read_exact(bytes.as_mut_slice())
        .map_err(|error| error.to_string())?;
    Ok(bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| {
            usize::try_from(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .unwrap_or(usize::MAX)
        })
        .collect())
}

pub(super) fn grm_int_chisq_order(chisq: &[f64], max6_selected: &[usize]) -> Vec<usize> {
    let mut order = max6_selected.to_vec();
    order.sort_by(
        |&left, &right| match (chisq[left].is_finite(), chisq[right].is_finite()) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => chisq[right]
                .total_cmp(&chisq[left])
                .then_with(|| left.cmp(&right)),
        },
    );
    order
}

#[inline]
pub(super) fn max6_grmmax6_screen_top_k(grm_int_only: bool, final_top_k: usize) -> usize {
    if grm_int_only {
        final_top_k.max(1)
    } else {
        MAX6_GRMMAX6_SCREEN_TOPK
    }
}

#[inline]
pub(super) fn max6_grmmax6_screen_threshold(
    grm_int_only: bool,
    screen_score_threshold: Option<f64>,
) -> Option<f64> {
    if grm_int_only {
        None
    } else {
        screen_score_threshold
    }
}

#[inline]
pub(super) fn max6_screen_gate_output_id(
    gate: logic_residual_screen::FourCellGate,
) -> &'static str {
    use logic_residual_screen::FourCellGate;
    match gate {
        FourCellGate::And00 => "AND00",
        FourCellGate::And01 => "AND01",
        FourCellGate::And10 => "AND10",
        FourCellGate::And11 => "AND11",
        FourCellGate::Xor => "XOR",
        FourCellGate::DosageInteraction => "DOSAGE_PRODUCT",
    }
}

pub(super) fn max6_grm_int_outputs_from_screen(
    unit_index: usize,
    selected_global_rows: &[usize],
    sites: &[GarfieldLogicSite],
    candidates: &[logic_residual_screen::ResidualScreenCandidate],
    chisq: &[f64],
) -> Result<Vec<Max6GrmMax6PairOutput>, String> {
    if candidates.len() != chisq.len() {
        return Err(format!(
            "GRM-INT screen candidate/score lengths differ: {} vs {}",
            candidates.len(),
            chisq.len()
        ));
    }
    let order = grm_int_chisq_order(chisq, &(0..candidates.len()).collect::<Vec<_>>());
    let mut outputs = Vec::with_capacity(order.len());
    for candidate_index in order {
        let candidate = &candidates[candidate_index];
        if candidate.first >= selected_global_rows.len()
            || candidate.second >= selected_global_rows.len()
            || candidate.first == candidate.second
        {
            return Err(format!(
                "GRM-INT screen candidate {candidate_index} has invalid local endpoints ({}, {}) for {} selected markers",
                candidate.first,
                candidate.second,
                selected_global_rows.len()
            ));
        }
        let screen_rank = candidate_index + 1;
        let first_global_row = selected_global_rows[candidate.first];
        let second_global_row = selected_global_rows[candidate.second];
        let first_site = sites
            .get(first_global_row)
            .ok_or_else(|| format!("GRM-INT first site index out of range: {first_global_row}"))?;
        let second_site = sites.get(second_global_row).ok_or_else(|| {
            format!("GRM-INT second site index out of range: {second_global_row}")
        })?;
        outputs.push(Max6GrmMax6PairOutput {
            unit_index,
            // The output's max6_rank column keeps the original Max6 screen
            // rank. Rows themselves are ordered by conditional GRM-INT chi-square.
            candidate_rank: screen_rank,
            first_global_row,
            second_global_row,
            four_cell_counts: Some(max6_four_cell_counts_in_physical_order(
                candidate.stats.counts,
                first_site,
                second_site,
            )),
            dosage: chisq[candidate_index],
            max6: candidate.proposal_score,
            best_channel: max6_screen_gate_output_id(candidate.best_gate.gate),
            pair_pve: f64::NAN,
        });
    }
    Ok(outputs)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scan_max6_grmmax6_unit(
    unit: &GarfieldLogicUnit,
    unit_index: usize,
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
    test_idx_local: &[usize],
    residualized_y: &[f64],
    stage1_y_lookup: &PackedYSumLookup,
    one_cell_margin_table: &OneCellMarginTable,
    p_c: Option<&[f64]>,
    no_grm_projector: Option<&NoGrmCovariateProjector>,
    projected_y: &[f64],
    y: &[f64],
    score_scale: f64,
    scaled_total_quadratic: f64,
    beam_params: &BeamSearchParams,
    final_top_k: usize,
    min_cell: usize,
    grm_int_only: bool,
    rule_permutation: bool,
    ld_tag_exhaustive: bool,
    prepared_unit: Option<&Max6GrmMax6LdPreparedUnit>,
    screen_score_threshold: Option<f64>,
    collect_candidate_pair_keys: bool,
    collect_candidate_scores: bool,
    screen_profile_pairs: Option<usize>,
    defer_exact_rerank: bool,
    precomputed_screen: Option<Max6PrecomputedScreen>,
    progress_callback: Option<&Py<PyAny>>,
) -> Result<Max6GrmMax6UnitScan, String> {
    let screen_input_prepare_started = Instant::now();
    let (selected, pair_groups) = if let Some(prepared) = prepared_unit {
        (
            prepared.selected_global_rows.clone(),
            prepared.pair_groups.clone(),
        )
    } else if ld_tag_exhaustive {
        let prepared =
            prepare_max6_grmmax6_ld_unit(unit, unit_kind_lc, logic_bits, train_idx_local)?;
        (prepared.selected_global_rows, prepared.pair_groups)
    } else {
        let Some(prepared) = prepare_logic_unit_continuous_with_ld_cache(
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
            residualized_y,
            Some(stage1_y_lookup),
            beam_params.clone(),
            None,
            false,
        )?
        else {
            return Ok(Max6GrmMax6UnitScan {
                outputs: Vec::new(),
                permutation_pool: None,
                deferred_screen: None,
                screen_pairs_passed: 0,
                final_screen_subset_audit_pairs_checked: 0,
                candidate_pair_keys: Vec::new(),
                candidate_score_observations: Vec::new(),
                window_unique_candidate_endpoints: 0,
                timing: Max6ObservedUnitTiming::default(),
                skipped_message: Some(format!(
                    "unit '{}' had no valid ML/CORR-prefiltered SNPs",
                    unit.label
                )),
            });
        };
        let pair_groups = if unit_kind_lc == "geneset" && unit.spans.len() > 1 {
            Some(prepared.local_groups)
        } else {
            None
        };
        (prepared.selected_global_rows, pair_groups)
    };
    if selected.len() < 2 {
        return Ok(Max6GrmMax6UnitScan {
            outputs: Vec::new(),
            permutation_pool: None,
            deferred_screen: None,
            screen_pairs_passed: 0,
            final_screen_subset_audit_pairs_checked: 0,
            candidate_pair_keys: Vec::new(),
            candidate_score_observations: Vec::new(),
            window_unique_candidate_endpoints: 0,
            timing: Max6ObservedUnitTiming::default(),
            skipped_message: Some(format!(
                "unit '{}' has fewer than two SNPs after Max6 marker selection",
                unit.label
            )),
        });
    }
    let screen_top_k = max6_grmmax6_screen_top_k(grm_int_only, final_top_k);
    let effective_screen_score_threshold =
        max6_grmmax6_screen_threshold(grm_int_only, screen_score_threshold);
    let packed = gather_rows_by_indices(
        logic_bits.bits(),
        logic_bits.row_words,
        selected.as_slice(),
        "max6_grmmax6 selected rows",
    )?;
    let selected_margins = one_cell_margin_table.select(selected.as_slice())?;
    let selected_margins_for_profile = screen_profile_pairs.map(|_| selected_margins.clone());
    let screen_input_prepare_seconds = screen_input_prepare_started.elapsed().as_secs_f64();
    let (screen, screen_kernel_seconds, precomputed_grm_int) = if let Some(precomputed) =
        precomputed_screen
    {
        (
            precomputed.screen,
            precomputed.kernel_seconds,
            precomputed.shared_grm_int,
        )
    } else {
        let screen_kernel_started = Instant::now();
        let screen = if let Some(callback) = progress_callback {
            let unit_label = unit.label.as_str();
            let mut notify = |done: usize, total: usize| {
                let detail = format!("{done}/{total} candidate pair checks; unit={unit_label}");
                garfield_stage_progress_notify(
                    Some(callback),
                    "max6_grmmax6_scan_pairs",
                    done,
                    total,
                    Some(detail.as_str()),
                )
                .map_err(|error| error.to_string())
            };
            logic_residual_screen::scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold_with_progress(
            packed.as_slice(),
            None,
            logic_bits.row_words,
            selected.len(),
            residualized_y,
            logic_bits.n_samples,
            screen_top_k,
            true,
            ResidualProposalMode::Max6,
            ResidualScreenKernel::OneCell,
            stage1_y_lookup,
            Some(selected_margins),
            pair_groups.as_deref(),
            min_cell,
            effective_screen_score_threshold,
            beam_params.allow_parallel,
            &mut notify,
        )?
        } else {
            logic_residual_screen::scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups_threshold(
            packed.as_slice(),
            None,
            logic_bits.row_words,
            selected.len(),
            residualized_y,
            logic_bits.n_samples,
            screen_top_k,
            true,
            ResidualProposalMode::Max6,
            ResidualScreenKernel::OneCell,
            stage1_y_lookup,
            Some(selected_margins),
            pair_groups.as_deref(),
            min_cell,
            effective_screen_score_threshold,
            beam_params.allow_parallel,
        )?
        };
        (screen, screen_kernel_started.elapsed().as_secs_f64(), None)
    };
    let screen_profile = if let (Some(profile_pairs), Some(profile_margins)) =
        (screen_profile_pairs, selected_margins_for_profile.as_ref())
    {
        let profile_started = Instant::now();
        let profile = logic_residual_screen::profile_one_cell_sample(
            packed.as_slice(),
            logic_bits.row_words,
            selected.len(),
            logic_bits.n_samples,
            screen_top_k,
            true,
            profile_pairs,
            residualized_y,
            profile_margins,
            stage1_y_lookup,
        )?;
        Max6ObservedScreenProfile {
            units: 1,
            sample_pairs: profile.sample_pairs,
            total_pairs: profile.total_pairs,
            sample_words: profile.sample_words,
            nonzero_intersections: profile.nonzero_intersections,
            intersection_popcount_sum: profile.intersection_popcount_sum,
            lookup_calls: profile.lookup_calls,
            lut8_word_calls: profile.lut8_word_calls,
            setbit_calls: profile.setbit_calls,
            bitset_seconds: profile.bitset_seconds,
            weighted_y_lookup_seconds: profile.weighted_y_lookup_seconds,
            margin_recovery_seconds: profile.margin_recovery_seconds,
            score_seconds: profile.score_seconds,
            heap_seconds: profile.heap_seconds,
            final_sort_seconds: profile.final_sort_seconds,
            replay_seconds: profile_started.elapsed().as_secs_f64(),
            heap_pushes: profile.heap_pushes,
            heap_replacements: profile.heap_replacements,
        }
    } else {
        Max6ObservedScreenProfile::default()
    };
    if screen.candidates.is_empty() {
        return Ok(Max6GrmMax6UnitScan {
            outputs: Vec::new(),
            permutation_pool: None,
            deferred_screen: None,
            screen_pairs_passed: 0,
            final_screen_subset_audit_pairs_checked: 0,
            candidate_pair_keys: Vec::new(),
            candidate_score_observations: Vec::new(),
            window_unique_candidate_endpoints: 0,
            timing: Max6ObservedUnitTiming {
                screen_input_prepare_seconds,
                screen_kernel_seconds,
                screen_pairs_evaluated: screen.pairs_evaluated,
                screen_profile,
                ..Default::default()
            },
            skipped_message: Some(format!(
                "unit '{}' produced no finite Max6 screen candidates",
                unit.label
            )),
        });
    }
    let screen_candidate_count = screen.candidates.len();
    if grm_int_only {
        let candidate_prepare_started = Instant::now();
        let candidate_pairs = screen
            .candidates
            .iter()
            .map(|candidate| (candidate.first, candidate.second))
            .collect::<Vec<_>>();
        let candidate_prepare_seconds = candidate_prepare_started.elapsed().as_secs_f64();
        let grm_int = if let Some(precomputed) = precomputed_grm_int {
            if precomputed.chisq.len() != screen_candidate_count
                || precomputed.invalid.len() != screen_candidate_count
            {
                return Err(format!(
                    "shared GRM-INT scorer returned {} scores / {} statuses for {screen_candidate_count} screen candidates",
                    precomputed.chisq.len(),
                    precomputed.invalid.len()
                ));
            }
            precomputed
        } else {
            grm_max6::grm_int_interaction_scores_packed(
                packed.as_slice(),
                logic_bits.row_words,
                logic_bits.n_samples,
                selected.len(),
                candidate_pairs.as_slice(),
                p_c,
                no_grm_projector,
                projected_y,
                score_scale,
            )?
        };

        let mut candidate_pair_keys = Vec::new();
        let mut candidate_score_observations = Vec::new();
        let mut window_unique_candidate_endpoints = 0usize;
        let candidate_key_collection_started = collect_candidate_pair_keys.then(Instant::now);
        if collect_candidate_pair_keys {
            let mut endpoint_seen = vec![false; selected.len()];
            candidate_pair_keys
                .try_reserve_exact(screen_candidate_count)
                .map_err(|error| {
                    format!(
                        "unable to reserve {} Max6 screen candidate diagnostic keys: {error}",
                        screen_candidate_count
                    )
                })?;
            for candidate in &screen.candidates {
                if candidate.first >= selected.len() || candidate.second >= selected.len() {
                    return Err(
                        "Max6 screen candidate diagnostic endpoint is out of range".to_string()
                    );
                }
                endpoint_seen[candidate.first] = true;
                endpoint_seen[candidate.second] = true;
                candidate_pair_keys.push(max6_diagnostic_pair_key(
                    selected[candidate.first],
                    selected[candidate.second],
                )?);
            }
            window_unique_candidate_endpoints =
                endpoint_seen.into_iter().filter(|&seen| seen).count();
        }
        if collect_candidate_scores {
            let audit_window_index = u32::try_from(unit_index).map_err(|_| {
                format!("Max6 score reuse audit window index {unit_index} exceeds u32 range")
            })?;
            candidate_score_observations
                .try_reserve_exact(screen_candidate_count)
                .map_err(|error| {
                    format!(
                        "unable to reserve {screen_candidate_count} Max6 score reuse observations: {error}"
                    )
                })?;
            for (candidate_index, candidate) in screen.candidates.iter().enumerate() {
                let tie_key = u32::try_from(candidate_index).map_err(|_| {
                    format!(
                        "Max6 screen score reuse audit tie key {candidate_index} exceeds u32 range"
                    )
                })?;
                candidate_score_observations.push(Max6CandidateScoreObservation {
                    pair_key: max6_diagnostic_pair_key(
                        selected[candidate.first],
                        selected[candidate.second],
                    )?,
                    window_index: audit_window_index,
                    tie_key,
                    score: candidate.proposal_score,
                    cached_score: candidate.proposal_score,
                    channel_id: max6_channel_audit_id(max6_screen_gate_output_id(
                        candidate.best_gate.gate,
                    ))?,
                });
            }
        }
        let candidate_key_collection_seconds = if collect_candidate_pair_keys {
            candidate_key_collection_started
                .expect("candidate-key timer exists when collection is enabled")
                .elapsed()
                .as_secs_f64()
        } else {
            0.0
        };
        let output_finalize_started = Instant::now();
        let outputs = max6_grm_int_outputs_from_screen(
            unit_index,
            selected.as_slice(),
            logic_bits.sites.as_slice(),
            screen.candidates.as_slice(),
            grm_int.chisq.as_slice(),
        )?;
        let screen_local_pairs = candidate_pairs;
        let output_order = grm_int_chisq_order(
            grm_int.chisq.as_slice(),
            &(0..screen_candidate_count).collect::<Vec<_>>(),
        );
        let final_local_pairs = output_order
            .iter()
            .map(|&candidate_index| {
                let candidate = &screen.candidates[candidate_index];
                (candidate.first, candidate.second)
            })
            .collect::<Vec<_>>();
        let final_screen_subset_audit_pairs_checked =
            audit_max6_final_pairs_are_screen_subset(&screen_local_pairs, &final_local_pairs)?;
        let output_finalize_seconds = output_finalize_started.elapsed().as_secs_f64();
        return Ok(Max6GrmMax6UnitScan {
            outputs,
            // GRM-INT is a nominal one-df statistic and has no Max6
            // permutation pool; the legacy exact permutation route is bypassed.
            permutation_pool: None,
            deferred_screen: None,
            screen_pairs_passed: screen_candidate_count,
            final_screen_subset_audit_pairs_checked,
            candidate_pair_keys,
            candidate_score_observations,
            window_unique_candidate_endpoints,
            timing: Max6ObservedUnitTiming {
                screen_input_prepare_seconds,
                screen_kernel_seconds,
                candidate_prepare_seconds,
                candidate_key_collection_seconds,
                grm_int: grm_int.diagnostics,
                output_finalize_seconds,
                screen_pairs_evaluated: screen.pairs_evaluated,
                screen_candidates: screen_candidate_count,
                grm_int_candidates: grm_int.chisq.len(),
                screen_profile,
                ..Default::default()
            },
            skipped_message: None,
        });
    }
    if defer_exact_rerank {
        return Ok(Max6GrmMax6UnitScan {
            outputs: Vec::new(),
            permutation_pool: None,
            deferred_screen: Some(Max6DeferredWindowScreen {
                selected_global_rows: selected,
                candidates: screen.candidates,
            }),
            screen_pairs_passed: screen_candidate_count,
            final_screen_subset_audit_pairs_checked: 0,
            candidate_pair_keys: Vec::new(),
            candidate_score_observations: Vec::new(),
            window_unique_candidate_endpoints: 0,
            timing: Max6ObservedUnitTiming {
                screen_input_prepare_seconds,
                screen_kernel_seconds,
                screen_pairs_evaluated: screen.pairs_evaluated,
                screen_candidates: screen_candidate_count,
                screen_profile,
                ..Default::default()
            },
            skipped_message: None,
        });
    }
    let candidate_prepare_started = Instant::now();
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
    let candidate_prepare_seconds = candidate_prepare_started.elapsed().as_secs_f64();
    let rerank_call_started = Instant::now();
    let exact = if let Some(p_c) = p_c {
        candidate_rerank_exact_packed_with_py_trusted_scaled(
            packed.as_slice(),
            logic_bits.row_words,
            logic_bits.n_samples,
            selected.len(),
            p_c,
            y,
            projected_y,
            candidate_first.as_slice(),
            candidate_second.as_slice(),
            final_top_k,
            DosageChannel::FwlProduct,
            score_scale,
        )?
    } else {
        let projector = no_grm_projector.ok_or_else(|| {
            "Max6 -> GRM-Max6 exact rerank has neither dense P_C nor no-GRM projector".to_string()
        })?;
        candidate_rerank_exact_packed_no_grm_covariate_with_py_trusted_scaled(
            packed.as_slice(),
            logic_bits.row_words,
            logic_bits.n_samples,
            selected.len(),
            projector,
            y,
            projected_y,
            candidate_first.as_slice(),
            candidate_second.as_slice(),
            final_top_k,
            DosageChannel::FwlProduct,
            score_scale,
        )?
    };
    let exact_candidate_count = exact.pair_i.len();
    let mut candidate_pair_keys = Vec::new();
    let mut candidate_score_observations = Vec::new();
    let mut window_unique_candidate_endpoints = 0usize;
    let candidate_key_collection_started = collect_candidate_pair_keys.then(Instant::now);
    if collect_candidate_pair_keys {
        let mut endpoint_seen = vec![false; selected.len()];
        candidate_pair_keys
            .try_reserve_exact(exact_candidate_count)
            .map_err(|error| {
                format!(
                    "unable to reserve {} Max6 candidate diagnostic keys: {error}",
                    exact_candidate_count
                )
            })?;
        for (&local_left, &local_right) in exact.pair_i.iter().zip(exact.pair_j.iter()) {
            if local_left >= selected.len() || local_right >= selected.len() {
                return Err("Max6 candidate diagnostic endpoint is out of range".to_string());
            }
            endpoint_seen[local_left] = true;
            endpoint_seen[local_right] = true;
            let global_left = selected[local_left];
            let global_right = selected[local_right];
            if global_left == global_right {
                return Err("Max6 candidate diagnostic encountered a self pair".to_string());
            }
            let (global_left, global_right) = if global_left < global_right {
                (global_left, global_right)
            } else {
                (global_right, global_left)
            };
            let global_left = u32::try_from(global_left).map_err(|_| {
                format!("Max6 diagnostic marker index {global_left} exceeds u32 range")
            })?;
            let global_right = u32::try_from(global_right).map_err(|_| {
                format!("Max6 diagnostic marker index {global_right} exceeds u32 range")
            })?;
            candidate_pair_keys.push(((global_left as u64) << 32) | global_right as u64);
        }
        window_unique_candidate_endpoints = endpoint_seen.into_iter().filter(|&seen| seen).count();
    }
    if collect_candidate_scores {
        if exact.max6.len() != exact_candidate_count
            || exact.pair_j.len() != exact_candidate_count
            || exact.global_pair_id.len() != exact_candidate_count
        {
            return Err(
                "Max6 score reuse audit candidate arrays have inconsistent lengths".to_string(),
            );
        }
        let audit_window_index = u32::try_from(unit_index).map_err(|_| {
            format!("Max6 score reuse audit window index {unit_index} exceeds u32 range")
        })?;
        candidate_score_observations
            .try_reserve_exact(exact_candidate_count)
            .map_err(|error| {
                format!(
                    "unable to reserve {exact_candidate_count} Max6 score reuse observations: {error}"
                )
            })?;
        for candidate in 0..exact_candidate_count {
            let local_left = exact.pair_i[candidate];
            let local_right = exact.pair_j[candidate];
            if local_left >= selected.len() || local_right >= selected.len() {
                return Err("Max6 score reuse audit endpoint is out of range".to_string());
            }
            let tie_key = u32::try_from(exact.global_pair_id[candidate]).map_err(|_| {
                format!(
                    "Max6 score reuse audit tie key {} exceeds u32 range",
                    exact.global_pair_id[candidate]
                )
            })?;
            let pair_key = max6_diagnostic_pair_key(selected[local_left], selected[local_right])?;
            let channel_id = max6_channel_audit_id(exact.best_max6_channel[candidate])?;
            candidate_score_observations.push(Max6CandidateScoreObservation {
                pair_key,
                window_index: audit_window_index,
                tie_key,
                score: exact.max6[candidate],
                cached_score: exact.max6[candidate],
                channel_id,
            });
        }
    }
    let candidate_key_collection_seconds = if collect_candidate_pair_keys {
        candidate_key_collection_started
            .expect("candidate-key timer exists when collection is enabled")
            .elapsed()
            .as_secs_f64()
    } else {
        0.0
    };
    let mut rerank_timing = exact.timing;
    rerank_timing.kernel_seconds = rerank_call_started.elapsed().as_secs_f64();
    let output_finalize_started = Instant::now();
    let output_order = exact.top_order.clone();
    let final_local_pairs = output_order
        .iter()
        .map(|&candidate_index| (exact.pair_i[candidate_index], exact.pair_j[candidate_index]))
        .collect::<Vec<_>>();
    let screen_local_pairs = screen
        .candidates
        .iter()
        .map(|candidate| (candidate.first, candidate.second))
        .collect::<Vec<_>>();
    let final_screen_subset_audit_pairs_checked =
        audit_max6_final_pairs_are_screen_subset(&screen_local_pairs, &final_local_pairs)?;
    let mut outputs = Vec::with_capacity(output_order.len());
    for &candidate_index in output_order.iter() {
        let local_first = exact.pair_i[candidate_index];
        let local_second = exact.pair_j[candidate_index];
        outputs.push(Max6GrmMax6PairOutput {
            unit_index,
            candidate_rank: exact.candidate_rank[candidate_index],
            first_global_row: selected[local_first],
            second_global_row: selected[local_second],
            four_cell_counts: None,
            dosage: exact.dosage[candidate_index],
            max6: exact.max6[candidate_index],
            best_channel: exact.best_max6_channel[candidate_index],
            pair_pve: pair_pve_from_scaled_score(
                exact.max6[candidate_index],
                scaled_total_quadratic,
            ),
        });
    }
    let permutation_pool = if rule_permutation {
        Some(Max6GrmMax6PermutationPool {
            unit_index,
            selected_global_rows: selected.clone(),
        })
    } else {
        None
    };
    let output_finalize_seconds = output_finalize_started.elapsed().as_secs_f64();
    Ok(Max6GrmMax6UnitScan {
        outputs,
        permutation_pool,
        deferred_screen: None,
        screen_pairs_passed: screen.candidates.len(),
        final_screen_subset_audit_pairs_checked,
        candidate_pair_keys,
        candidate_score_observations,
        window_unique_candidate_endpoints,
        timing: Max6ObservedUnitTiming {
            screen_input_prepare_seconds,
            screen_kernel_seconds,
            candidate_prepare_seconds,
            candidate_key_collection_seconds,
            rerank: rerank_timing,
            grm_int: GrmIntScoreDiagnostics::default(),
            output_finalize_seconds,
            screen_pairs_evaluated: screen.pairs_evaluated,
            screen_candidates: screen.candidates.len(),
            exact_candidates: exact_candidate_count,
            grm_int_candidates: 0,
            exact_input_duplicates: exact.input_duplicate_count,
            exact_input_self_pairs: exact.input_self_count,
            screen_profile,
        },
        skipped_message: None,
    })
}
