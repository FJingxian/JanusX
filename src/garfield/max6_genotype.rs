use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_max6_genotype_tags_owned(
    prefix: String,
    sample_ids: Vec<String>,
    sample_indices: Vec<usize>,
    unit_kind: String,
    groups: Option<Vec<Vec<(String, i32, i32)>>>,
    group_names: Option<Vec<String>>,
    extension: usize,
    step: Option<usize>,
    scan_bimranges: Option<Vec<String>>,
    maf_threshold: f32,
    max_missing_rate: f32,
    het_threshold: f32,
    snps_only: bool,
    threads: usize,
    logic_mem_budget_bytes: u64,
    ld_tag_r2: f64,
) -> Result<Max6PreparedGenotypeTags, String> {
    validate_max6_ld_tag_r2(ld_tag_r2)?;
    let prep_profiler = Max6GenotypePrepProfiler::from_env();
    if sample_ids.is_empty() || sample_ids.len() != sample_indices.len() {
        return Err(format!(
            "Max6 genotype preparation requires matching nonempty sample IDs/indices; got {}/{}",
            sample_ids.len(),
            sample_indices.len()
        ));
    }
    let prefix = normalize_plink_prefix(&prefix);
    let fam_sample_ids = read_fam(prefix.as_str())?;
    let (resolved_sample_indices, resolved_sample_ids) =
        build_sample_selection(fam_sample_ids.as_slice(), Some(sample_ids.clone()), None)?;
    if resolved_sample_indices != sample_indices || resolved_sample_ids != sample_ids {
        return Err("Max6 global sample selection did not preserve requested order".to_string());
    }

    let unit_kind_lc = unit_kind.trim().to_ascii_lowercase();
    let grouped_active_mode = matches!(unit_kind_lc.as_str(), "gene" | "geneset" | "group");
    let parsed_scan_bimranges = scan_bimranges
        .as_deref()
        .map(parse_scan_bimranges)
        .transpose()?
        .unwrap_or_default();
    if unit_kind_lc == "wholegenome" && !parsed_scan_bimranges.is_empty() {
        return Err("--bimrange is not supported under unit_kind 'wholegenome'".to_string());
    }
    let (selected_groups, selected_group_names) = if grouped_active_mode {
        let groups_ref = groups.as_deref().ok_or_else(|| {
            format!(
                "groups must be provided for unit_kind in {{gene, geneset, group}}; got '{}'",
                unit_kind_lc
            )
        })?;
        let (selected_groups, selected_names, _) = select_logic_group_interval_defs_for_scan(
            groups_ref,
            group_names.as_deref(),
            parsed_scan_bimranges.as_slice(),
            unit_kind_lc.as_str(),
        )?;
        (Some(selected_groups), Some(selected_names))
    } else {
        (groups.clone(), group_names.clone())
    };

    let config = Max6PreparationConfigFingerprint::new(
        unit_kind_lc.as_str(),
        groups.as_deref(),
        group_names.as_deref(),
        extension,
        step,
        scan_bimranges.as_deref(),
        sample_indices.as_slice(),
        maf_threshold,
        max_missing_rate,
        het_threshold,
        snps_only,
        ld_tag_r2,
    );
    let input_fingerprint =
        build_max6_prepared_input_fingerprint(prefix.as_str(), sample_ids.clone(), &config)?;

    let stream_window = matches!(unit_kind_lc.as_str(), "" | "window");
    if stream_window {
        let fused_phase = prep_profiler.start_phase();
        let stream_result = prepare_max6_window_streamed_ld(
            prefix.as_str(),
            sample_ids.as_slice(),
            sample_indices.as_slice(),
            extension,
            step,
            parsed_scan_bimranges.as_slice(),
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            effective_threads_local(threads),
            ld_tag_r2,
            if prep_profiler.enabled() {
                Some(fused_phase.tracker())
            } else {
                None
            },
        );
        let detail = match stream_result.as_ref() {
            Ok(prepared) => format!(
                "source_markers={} qc_rows={} windows={} selected_windows={} active_tags={} pair_occurrences={} max_active_rows={} max_batch_bytes={} threads={} storage=bounded_window",
                prepared.source_marker_count,
                prepared.qc_row_count,
                prepared.distinct_window_count,
                prepared.selected_window_count,
                prepared.active_source_rows.len(),
                prepared.window_pair_count,
                prepared.max_active_rows,
                prepared.max_task_batch_bytes,
                effective_threads_local(threads)
            ),
            Err(error) if error == MAX6_WINDOW_STREAM_UNSUPPORTED_ORDER => {
                "input_order=unsupported; retrying compatibility stream".to_string()
            }
            Err(_) => "status=error".to_string(),
        };
        prep_profiler.finish_phase(fused_phase, "streamed_window_qc_ld", detail.as_str());
        match stream_result {
            Ok(prepared) => {
                let metadata_phase = prep_profiler.start_phase();
                let compact_tag_sites_result = read_compact_max6_bim_rows(
                    prefix.as_str(),
                    prepared.source_marker_count,
                    prepared.active_source_rows.as_slice(),
                );
                let metadata_detail = match compact_tag_sites_result.as_ref() {
                    Ok(sites) => format!(
                        "requested_rows={} loaded_rows={}",
                        prepared.active_source_rows.len(),
                        sites.len()
                    ),
                    Err(_) => "status=error".to_string(),
                };
                prep_profiler.finish_phase(
                    metadata_phase,
                    "tag_metadata_bim_read",
                    metadata_detail.as_str(),
                );
                let active_source_sites = Arc::new(compact_tag_sites_result?);
                return Ok(Max6PreparedGenotypeTags {
                    input_fingerprint,
                    unit_kind: unit_kind_lc,
                    units: prepared.units,
                    screen_units: prepared.screen_units,
                    original_unit_indices: prepared.original_unit_indices,
                    active_source_rows: prepared.active_source_rows,
                    active_source_sites,
                    source_marker_count: prepared.source_marker_count,
                    window_pair_count: prepared.window_pair_count,
                    global_site_keep: prepared.global_site_keep,
                });
            }
            Err(error) if error == MAX6_WINDOW_STREAM_UNSUPPORTED_ORDER => {}
            Err(error) => return Err(error),
        }
    }
    let fused_phase = prep_profiler.start_phase();
    let (global_site_keep, global_qc_source_rows, logic_bits, window_layout) = if stream_window {
        let stream_result = prepare_max6_window_stream(
            prefix.as_str(),
            sample_ids.as_slice(),
            sample_indices.as_slice(),
            extension,
            step,
            parsed_scan_bimranges.as_slice(),
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            effective_threads_local(threads),
            if prep_profiler.enabled() {
                Some(fused_phase.tracker())
            } else {
                None
            },
        );
        let detail = match stream_result.as_ref() {
            Ok(prepared) => format!(
                "source_markers={} qc_rows={} packed_rows={} units={} samples={} row_words={} packed_bytes={} threads={} storage=async_mmap",
                prepared.global_site_keep.len(),
                prepared.global_site_keep.iter().filter(|&&keep| keep).count(),
                prepared.packed_source_rows.len(),
                prepared.units.len(),
                prepared.logic_bits.n_samples,
                prepared.logic_bits.row_words,
                prepared
                    .packed_source_rows
                    .len()
                    .saturating_mul(prepared.logic_bits.row_words)
                    .saturating_mul(std::mem::size_of::<u64>()),
                effective_threads_local(threads)
            ),
            Err(_) => "status=error".to_string(),
        };
        prep_profiler.finish_phase(fused_phase, "streamed_window_qc_pack", detail.as_str());
        let prepared = stream_result?;
        (
            prepared.global_site_keep,
            prepared.packed_source_rows,
            prepared.logic_bits,
            Some((prepared.units, prepared.source_unit_indices)),
        )
    } else {
        let fused_result = prepare_max6_global_qc_and_logic_bits(
            prefix.as_str(),
            sample_ids.as_slice(),
            sample_indices.as_slice(),
            maf_threshold,
            max_missing_rate,
            het_threshold,
            snps_only,
            effective_threads_local(threads),
            logic_mem_budget_bytes,
            if prep_profiler.enabled() {
                Some(fused_phase.tracker())
            } else {
                None
            },
        );
        let detail = match fused_result.as_ref() {
            Ok((site_keep, source_rows, bits)) => format!(
                "source_markers={} qc_rows={} samples={} row_words={} packed_bytes={} threads={} storage={}",
                site_keep.len(),
                source_rows.len(),
                bits.n_samples,
                bits.row_words,
                source_rows
                    .len()
                    .saturating_mul(bits.row_words)
                    .saturating_mul(std::mem::size_of::<u64>()),
                effective_threads_local(threads),
                if bits.bits_mmap.is_some() { "mmap" } else { "heap" }
            ),
            Err(_) => "status=error".to_string(),
        };
        prep_profiler.finish_phase(
            fused_phase,
            "fused_genotype_qc_and_logic_pack",
            detail.as_str(),
        );
        let (site_keep, source_rows, bits) = fused_result?;
        (site_keep, source_rows, bits, None)
    };
    let source_marker_count = global_site_keep.len();

    let units_phase = prep_profiler.start_phase();
    let units_result = (|| {
        if let Some((units, source_unit_indices)) = window_layout {
            if units.is_empty() {
                return Err("no scan units were built from QC-passing window markers".to_string());
            }
            if units.len() != source_unit_indices.len() {
                return Err("streamed window unit index map length mismatch".to_string());
            }
            let scan_unit_indices = (0..units.len()).collect::<Vec<_>>();
            Ok((units, scan_unit_indices, Some(source_unit_indices)))
        } else {
            let units = build_logic_units(
                logic_bits.sites.as_slice(),
                unit_kind_lc.as_str(),
                selected_groups.as_deref(),
                selected_group_names.as_deref(),
                extension,
                step,
            )?;
            if units.is_empty() {
                return Err(
                    "no scan units were built from the globally QC-passing markers".to_string(),
                );
            }
            let scan_unit_indices = if grouped_active_mode || parsed_scan_bimranges.is_empty() {
                (0..units.len()).collect::<Vec<_>>()
            } else {
                units
                    .iter()
                    .enumerate()
                    .filter_map(|(index, unit)| {
                        unit_overlaps_scan_bimranges(unit, parsed_scan_bimranges.as_slice())
                            .then_some(index)
                    })
                    .collect::<Vec<_>>()
            };
            if scan_unit_indices.is_empty() {
                return Err("--bimrange matched no Max6 scan units".to_string());
            }
            Ok((units, scan_unit_indices, None))
        }
    })();
    let units_detail = match units_result.as_ref() {
        Ok((units, scan_indices, _)) => {
            format!(
                "units={} selected_units={}",
                units.len(),
                scan_indices.len()
            )
        }
        Err(_) => "status=error".to_string(),
    };
    prep_profiler.finish_phase(units_phase, "build_scan_units", units_detail.as_str());
    let (units, scan_unit_indices, source_unit_indices) = units_result?;

    let ld_phase = prep_profiler.start_phase();
    let pool = if threads > 1 {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(effective_threads_local(threads))
                .thread_name(|index| format!("janusx-max6-ld-{index}"))
                .build()
                .map_err(|error| format!("failed to build Max6 LD-tag pool: {error}"))?,
        )
    } else {
        None
    };
    let train_indices = (0..sample_ids.len()).collect::<Vec<_>>();
    let local_universe_result = if prep_profiler.enabled() {
        let mut progress = |_done: usize, _total: usize, _selected: usize| {
            ld_phase.sample_now();
            Ok(())
        };
        prepare_max6_screen_selection(
            &logic_bits,
            units.as_slice(),
            scan_unit_indices.as_slice(),
            unit_kind_lc.as_str(),
            train_indices.as_slice(),
            pool.as_ref(),
            Some(&mut progress),
            ld_tag_r2,
        )
    } else {
        prepare_max6_screen_selection(
            &logic_bits,
            units.as_slice(),
            scan_unit_indices.as_slice(),
            unit_kind_lc.as_str(),
            train_indices.as_slice(),
            pool.as_ref(),
            None,
            ld_tag_r2,
        )
    };
    let ld_detail = match local_universe_result.as_ref() {
        Ok(universe) => {
            let common = format!(
                "units={} active_rows={} pair_occurrences={} workers={}",
                universe.units.len(),
                universe.active_rows.len(),
                universe.window_pair_count,
                pool.as_ref()
                    .map_or(1, rayon::ThreadPool::current_num_threads)
            );
            if prep_profiler.enabled() {
                format!(
                    "{common} support_cache_build={:.3}s support_cache_rows={} support_cache_bytes={} tag_fingerprint={:016x}",
                    universe.support_cache_build_seconds,
                    universe.support_cache_rows,
                    universe.support_cache_payload_bytes,
                    max6_ld_selection_fingerprint(
                        universe.units.as_slice(),
                        universe.original_unit_indices.as_slice()
                    )
                )
            } else {
                common
            }
        }
        Err(_) => "status=error".to_string(),
    };
    prep_profiler.finish_phase(ld_phase, "ld_selection", ld_detail.as_str());
    let local_universe = local_universe_result?;
    // LD selection has consumed the full QC bitset. Release its packed rows
    // and site metadata before remapping tags and rereading their BIM fields.
    drop(logic_bits);

    let remap_phase = prep_profiler.start_phase();
    let source_row_for_local = |local_row: usize| {
        global_qc_source_rows
            .get(local_row)
            .copied()
            .ok_or_else(|| {
                format!("Max6 selected logic row {local_row} is outside global-QC source-row map")
            })
    };
    let mut screen_units = Vec::with_capacity(local_universe.units.len());
    for local_unit in local_universe.units.iter() {
        let selected_global_rows = local_unit
            .selected_global_rows
            .iter()
            .map(|&row| source_row_for_local(row))
            .collect::<Result<Vec<_>, _>>()?;
        let ld_compression = local_unit
            .ld_compression
            .iter()
            .map(|record| {
                Ok(GarfieldLdCompressionRecord {
                    kept_global_row: source_row_for_local(record.kept_global_row)?,
                    dropped_global_row: source_row_for_local(record.dropped_global_row)?,
                    r2: record.r2,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        screen_units.push(Max6GrmMax6LdPreparedUnit {
            selected_global_rows,
            pair_groups: local_unit.pair_groups.clone(),
            ld_compression,
        });
    }
    let mut active_source_rows = local_universe
        .active_rows
        .iter()
        .map(|&row| source_row_for_local(row))
        .collect::<Result<Vec<_>, _>>()?;
    active_source_rows.sort_unstable();
    active_source_rows.dedup();
    let remap_detail = format!(
        "units={} active_rows={} selected_rows={} compression_records={}",
        screen_units.len(),
        active_source_rows.len(),
        screen_units
            .iter()
            .map(|unit| unit.selected_global_rows.len())
            .sum::<usize>(),
        screen_units
            .iter()
            .map(|unit| unit.ld_compression.len())
            .sum::<usize>()
    );
    prep_profiler.finish_phase(remap_phase, "selected_row_remap", remap_detail.as_str());
    // All selected rows map back to the global-QC-passing source rows above.
    // Only their original BIM identity is still needed here; genotype QC was
    // already done before LD selection and trait-specific QC runs per scan.
    let metadata_phase = prep_profiler.start_phase();
    let compact_tag_sites_result = read_compact_max6_bim_rows(
        prefix.as_str(),
        source_marker_count,
        active_source_rows.as_slice(),
    );
    let metadata_detail = match compact_tag_sites_result.as_ref() {
        Ok(sites) => format!(
            "requested_rows={} loaded_rows={}",
            active_source_rows.len(),
            sites.len()
        ),
        Err(_) => "status=error".to_string(),
    };
    prep_profiler.finish_phase(
        metadata_phase,
        "tag_metadata_bim_read",
        metadata_detail.as_str(),
    );
    let compact_tag_sites = compact_tag_sites_result?;
    let active_source_sites = Arc::new(compact_tag_sites);
    let mut units_meta = Vec::with_capacity(screen_units.len());
    let mut original_unit_indices = Vec::with_capacity(screen_units.len());
    for &selected_unit_index in local_universe.original_unit_indices.iter() {
        let original_unit_index = source_unit_indices
            .as_ref()
            .and_then(|indices| indices.get(selected_unit_index))
            .copied()
            .unwrap_or(selected_unit_index);
        let unit_index = if source_unit_indices.is_some() {
            selected_unit_index
        } else {
            original_unit_index
        };
        let unit = units
            .get(unit_index)
            .ok_or_else(|| format!("prepared Max6 source unit {unit_index} is out of range"))?;
        units_meta.push(Max6PreparedUnitMetadata {
            original_unit_index,
            label: unit.label.clone(),
            spans: unit.spans.clone(),
        });
        original_unit_indices.push(original_unit_index);
    }
    if units_meta.len() != screen_units.len() {
        return Err("prepared Max6 unit metadata length mismatch".to_string());
    }

    Ok(Max6PreparedGenotypeTags {
        input_fingerprint,
        unit_kind: unit_kind_lc,
        units: units_meta,
        screen_units,
        original_unit_indices,
        active_source_rows,
        active_source_sites,
        source_marker_count,
        window_pair_count: local_universe.window_pair_count,
        global_site_keep,
    })
}

pub(super) fn finalize_max6_genotype_universe(
    tags: Arc<Max6PreparedGenotypeTags>,
) -> Result<Max6PreparedGenotypeUniverse, String> {
    let ownership = Max6ScreenPairOwnership::build(
        tags.screen_units.as_slice(),
        tags.active_source_rows.as_slice(),
        tags.source_marker_count,
    )?;
    let pair_universe_size = count_max6_pairs_by_membership(
        tags.active_source_rows.as_slice(),
        tags.window_pair_count,
        &ownership,
    )?;
    Max6PreparedGenotypeUniverse::new(tags, pair_universe_size)
}
