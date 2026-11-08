use super::bs::conditional_split_support_passes;
use super::*;

#[test]
fn max6_pgen_row_path_matches_equivalent_bed_input() {
    let pgen_dir = make_temp_dir("max6_pgen_row_path");
    let bed_dir = make_temp_dir("max6_bed_row_path");
    let pgen_prefix = pgen_dir.join("toy");
    let bed_prefix = bed_dir.join("toy");

    let mut pgen = vec![0x6c, 0x1b, 0x10];
    pgen.extend_from_slice(&2u32.to_le_bytes());
    pgen.extend_from_slice(&5u32.to_le_bytes());
    pgen.push(0); // 4-bit record types and 1-byte record lengths.
    pgen.extend_from_slice(&23u64.to_le_bytes()); // first record block.
    pgen.extend_from_slice(&[0x00, 2, 2]); // two type-0 records, each two bytes.
    pgen.extend_from_slice(&[0xc4, 0x01, 0xaa, 0x02]);
    fs::write(pgen_prefix.with_extension("pgen"), pgen).unwrap();
    fs::write(
        pgen_prefix.with_extension("psam"),
        "#FID IID\nf s1\nf s2\nf s3\nf s4\nf s5\n",
    )
    .unwrap();
    fs::write(
        pgen_prefix.with_extension("pvar"),
        "#CHROM POS ID REF ALT\n1 10 rs1 A G\n1 20 rs2 C T\n",
    )
    .unwrap();

    let mut bed = vec![0x6c, 0x1b, 0x01];
    bed.extend_from_slice(&[0x48, 0x02, 0xff, 0x03]);
    fs::write(bed_prefix.with_extension("bed"), bed).unwrap();
    fs::write(
        bed_prefix.with_extension("fam"),
        "f s1 0 0 0 -9\nf s2 0 0 0 -9\nf s3 0 0 0 -9\nf s4 0 0 0 -9\nf s5 0 0 0 -9\n",
    )
    .unwrap();
    fs::write(
        bed_prefix.with_extension("bim"),
        "1 rs1 0 10 A G\n1 rs2 0 20 C T\n",
    )
    .unwrap();

    let sample_ids = (1..=5).map(|idx| format!("s{idx}")).collect::<Vec<_>>();
    let sample_indices = (0..5).collect::<Vec<_>>();
    let run = |prefix: &Path| {
        prepare_max6_global_qc_and_logic_bits(
            prefix.to_str().unwrap(),
            &sample_ids,
            &sample_indices,
            0.0,
            1.0,
            1.0,
            true,
            2,
            1_000_000,
            None,
        )
        .unwrap()
    };

    let (pgen_keep, pgen_rows, pgen_logic) = run(&pgen_prefix);
    let (bed_keep, bed_rows, bed_logic) = run(&bed_prefix);
    assert_eq!(pgen_keep, bed_keep);
    assert_eq!(pgen_rows, bed_rows);
    assert_eq!(pgen_logic.bits(), bed_logic.bits());
    let site_metadata = |logic: &GarfieldLogicBits| {
        logic
            .sites
            .iter()
            .map(|site| {
                (
                    site.chrom().to_string(),
                    site.snp().to_string(),
                    site.pos,
                    site.ref_allele().to_string(),
                    site.alt_allele().to_string(),
                    site.mode,
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(site_metadata(&pgen_logic), site_metadata(&bed_logic));

    let _ = fs::remove_dir_all(pgen_dir);
    let _ = fs::remove_dir_all(bed_dir);
}

#[test]
fn max6_cross_window_order_uses_score_then_window_pair_tie_key() {
    let scores = [4.0, 9.0, 9.0, 4.0];
    let tie_keys = [7, 12, 3, 2];
    assert_eq!(
        max6_order_by_score_and_tie(&scores, &tie_keys).unwrap(),
        vec![2, 1, 3, 0]
    );
    assert!(max6_order_by_score_and_tie(&scores, &tie_keys[..3]).is_err());
    assert_eq!(
        max6_cross_window_channel_for_orientation("AND01", false),
        "AND10"
    );
    assert_eq!(
        max6_cross_window_channel_for_orientation("AND10", false),
        "AND01"
    );
    assert_eq!(
        max6_cross_window_channel_for_orientation("DOSAGE_PRODUCT", false),
        "DOSAGE_PRODUCT"
    );
}

#[test]
fn max6_ordered_subset_lut_packing_matches_scalar_genotypes_across_word_boundary() {
    let n_source_samples = 83usize;
    let sample_indices = (0..n_source_samples)
        .filter(|index| index % 7 != 3)
        .collect::<Vec<_>>();
    let mut row = vec![0u8; n_source_samples.div_ceil(4)];
    let source_codes = (0..n_source_samples)
        .map(|index| ((index * 3 + index / 5) % 4) as u8)
        .collect::<Vec<_>>();
    for (sample, &code) in source_codes.iter().enumerate() {
        row[sample >> 2] |= code << ((sample & 3) * 2);
    }

    let plan = build_logic_sample_plan(sample_indices.as_slice());
    let row_words = words_for_samples(sample_indices.len());
    let mut expected_logic_missing = 0usize;
    let mut expected_hom_alt = 0usize;
    let mut expected_hom_ref = 0usize;
    let mut expected_has_raw_missing = false;
    let mut expected_has_het = false;
    for &src_sample in &sample_indices {
        match source_codes[src_sample] {
            0b00 => expected_hom_ref += 1,
            0b01 => {
                expected_logic_missing += 1;
                expected_has_raw_missing = true;
            }
            0b10 => {
                expected_logic_missing += 1;
                expected_has_het = true;
            }
            0b11 => expected_hom_alt += 1,
            _ => unreachable!(),
        }
    }

    let mut actual_alt = vec![0u64; row_words];
    let mut actual_missing = vec![0u64; row_words];
    let (
        observed_logic_missing,
        observed_hom_alt,
        observed_hom_ref,
        observed_has_het,
        observed_has_raw_missing,
    ) = scan_max6_subset_row_and_pack(
        row.as_slice(),
        plan.as_slice(),
        &mut actual_alt,
        &mut actual_missing,
    );
    assert_eq!(observed_logic_missing, expected_logic_missing);
    assert_eq!(observed_hom_alt, expected_hom_alt);
    assert_eq!(observed_hom_ref, expected_hom_ref);
    assert_eq!(observed_has_raw_missing, expected_has_raw_missing);
    assert_eq!(observed_has_het, expected_has_het);

    let mut expected_alt = vec![0u64; row_words];
    let mut expected_missing = vec![0u64; row_words];
    for (dst_sample, &src_sample) in sample_indices.iter().enumerate() {
        let code = source_codes[src_sample];
        if code == 0b11 {
            expected_alt[dst_sample >> 6] |= 1u64 << (dst_sample & 63);
        } else if code == 0b01 || code == 0b10 {
            expected_missing[dst_sample >> 6] |= 1u64 << (dst_sample & 63);
        }
    }

    assert_eq!(actual_alt, expected_alt);
    assert_eq!(actual_missing, expected_missing);

    let mut subset_scan_alt = vec![0u64; row_words];
    let mut subset_scan_missing = vec![0u64; row_words];
    let qc = scan_max6_row_qc_and_pack_logic(
        row.as_slice(),
        sample_indices.len(),
        false,
        plan.as_slice(),
        0.0,
        1.0,
        true,
        &mut subset_scan_alt,
        &mut subset_scan_missing,
    );
    let mut expected_usable_homo = 0usize;
    let mut expected_hom_ref = 0usize;
    let mut expected_hom_alt = 0usize;
    for &src_sample in &sample_indices {
        match source_codes[src_sample] {
            0b00 => {
                expected_hom_ref += 1;
                expected_usable_homo += 1;
            }
            0b11 => {
                expected_hom_alt += 1;
                expected_usable_homo += 1;
            }
            _ => {}
        }
    }
    let expected_flip =
        expected_usable_homo > 0 && (expected_hom_alt as f32 / expected_usable_homo as f32) > 0.5;
    let expected_majority_is_one = if expected_flip {
        expected_hom_ref > expected_hom_alt
    } else {
        expected_hom_alt > expected_hom_ref
    };
    let mut expected_final_words = vec![0u64; row_words];
    for (dst_sample, &src_sample) in sample_indices.iter().enumerate() {
        let code = source_codes[src_sample];
        let value = match code {
            0b00 => expected_flip,
            0b11 => !expected_flip,
            _ => expected_majority_is_one,
        };
        if value {
            expected_final_words[dst_sample >> 6] |= 1u64 << (dst_sample & 63);
        }
    }
    assert!(qc.keep);
    assert_eq!(qc.row_flip, expected_flip);
    assert_eq!(subset_scan_alt, expected_final_words);
    assert_eq!(subset_scan_missing, expected_missing);

    let reordered_samples = sample_indices.iter().rev().copied().collect::<Vec<_>>();
    let reordered_plan = build_logic_sample_plan(reordered_samples.as_slice());
    assert!(reordered_plan.iter().any(|entry| !entry.dst_contiguous));
    let mut reordered_alt = vec![0u64; row_words];
    let mut reordered_missing = vec![0u64; row_words];
    let (reordered_logic_missing, reordered_hom_alt, reordered_hom_ref, _, _) =
        scan_max6_subset_row_and_pack(
            row.as_slice(),
            reordered_plan.as_slice(),
            &mut reordered_alt,
            &mut reordered_missing,
        );
    let mut expected_reordered_alt = vec![0u64; row_words];
    let mut expected_reordered_missing = vec![0u64; row_words];
    let mut expected_reordered_logic_missing = 0usize;
    let mut expected_reordered_hom_alt = 0usize;
    let mut expected_reordered_hom_ref = 0usize;
    for (dst_sample, &src_sample) in reordered_samples.iter().enumerate() {
        match source_codes[src_sample] {
            0b00 => expected_reordered_hom_ref += 1,
            0b01 | 0b10 => {
                expected_reordered_logic_missing += 1;
                expected_reordered_missing[dst_sample >> 6] |= 1u64 << (dst_sample & 63);
            }
            0b11 => {
                expected_reordered_hom_alt += 1;
                expected_reordered_alt[dst_sample >> 6] |= 1u64 << (dst_sample & 63);
            }
            _ => unreachable!(),
        }
    }
    assert_eq!(reordered_alt, expected_reordered_alt);
    assert_eq!(reordered_missing, expected_reordered_missing);
    assert_eq!(reordered_logic_missing, expected_reordered_logic_missing);
    assert_eq!(reordered_hom_alt, expected_reordered_hom_alt);
    assert_eq!(reordered_hom_ref, expected_reordered_hom_ref);
}

#[test]
fn max6_cross_window_parity_compares_unrounded_scores_in_canonical_orientation() {
    let local_orientation = Max6CrossWindowExactScore {
        logic: [1.0, 2.0, 3.0, 4.0, 5.0],
        dosage: 6.0,
        max6: 6.0,
        best_channel: "AND01",
    };
    let canonical_orientation = Max6CrossWindowExactScore {
        logic: [1.0, 3.0, 2.0, 4.0, 5.0],
        dosage: 6.0,
        max6: 6.0,
        best_channel: "AND10",
    };

    let equal =
        max6_cross_window_pair_score_difference(local_orientation, canonical_orientation, false);
    assert_eq!(equal.max_abs_by_channel, [0.0; 7]);
    assert_eq!(equal.max_rel_by_channel, [0.0; 7]);
    assert!(!equal.finite_mask_mismatch);
    assert!(!equal.best_channel_mismatch);

    let mut perturbed = canonical_orientation;
    perturbed.logic[0] += 2.5e-13;
    let changed = max6_cross_window_pair_score_difference(local_orientation, perturbed, false);
    assert!((changed.max_abs_by_channel[0] - 2.5e-13).abs() < 1.0e-16);
    assert!(changed.max_abs_by_channel[0] > 0.0);

    perturbed.dosage = f64::NAN;
    let non_finite = max6_cross_window_pair_score_difference(local_orientation, perturbed, false);
    assert!(non_finite.finite_mask_mismatch);
}

#[test]
fn cross_window_union_defaults_only_for_multi_window_scans() {
    assert!(max6_cross_window_union_enabled("window", 2, None));
    assert!(!max6_cross_window_union_enabled("window", 1, None));
    assert!(!max6_cross_window_union_enabled("gene", 2, None));
    assert!(!max6_cross_window_union_enabled("geneset", 2, Some("1")));
    assert!(!max6_cross_window_union_enabled("group", 2, None));
    assert!(!max6_cross_window_union_enabled("window", 2, Some("0")));
    assert!(max6_cross_window_union_enabled("window", 2, Some("1")));
}

#[test]
fn max6_cross_window_union_rerank_matches_independent_window_exact_orders() {
    let n_samples = 96usize;
    let mut logic_bits = build_test_logic_bits(5, n_samples);
    logic_bits.bits_flat.fill(0);
    for marker in 0..5 {
        for sample in 0..n_samples {
            if (sample * 13 + marker * 7 + sample * marker * 3) % 19 < 7 {
                logic_bits.bits_flat[marker * logic_bits.row_words + sample / 64] |=
                    1u64 << (sample % 64);
            }
        }
    }
    let y = (0..n_samples)
        .map(|sample| ((sample * 17 % 31) as f64 - 15.0) * 0.07)
        .collect::<Vec<_>>();
    let projected_y = y.clone();
    let mut p_c = vec![0.0; n_samples * n_samples];
    for sample in 0..n_samples {
        p_c[sample * n_samples + sample] = 1.0;
    }
    let make_candidate = |first, second, proposal_score| {
        let stats = crate::garfield::four_cell::FourCellSufficientStats {
            counts: [20, 20, 20, 36],
            sums: [0.1, 0.2, 0.3, 0.4],
            n_valid: n_samples,
        };
        logic_residual_screen::ResidualScreenCandidate {
            first,
            second,
            best_gate: crate::garfield::four_cell::FourCellGateScore {
                gate: crate::garfield::four_cell::FourCellGate::And00,
                score: proposal_score,
                signed_score: proposal_score,
                n_hit: 20,
            },
            proposal_score,
            proposal_source: logic_residual_screen::ResidualProposalSource::Max6,
            stats,
            derived: crate::garfield::four_cell::derive_gate5_gate6d_scores(stats),
        }
    };
    let windows = [vec![0usize, 1, 2, 3], vec![1usize, 2, 3, 4]];
    let window_candidates = [
        vec![
            make_candidate(0, 1, 5.0),
            make_candidate(1, 2, 4.0),
            make_candidate(2, 3, 3.0),
        ],
        vec![
            make_candidate(0, 1, 5.0),
            make_candidate(1, 2, 4.0),
            make_candidate(2, 3, 3.0),
        ],
    ];
    let mut expected = Vec::new();
    for (selected, candidates) in windows.iter().zip(window_candidates.iter()) {
        let packed = gather_rows_by_indices(
            logic_bits.bits(),
            logic_bits.row_words,
            selected,
            "cross-window parity fixture",
        )
        .unwrap();
        let candidate_first = candidates
            .iter()
            .map(|candidate| candidate.first as i64)
            .collect::<Vec<_>>();
        let candidate_second = candidates
            .iter()
            .map(|candidate| candidate.second as i64)
            .collect::<Vec<_>>();
        let exact = candidate_rerank_exact_packed_with_py_trusted_scaled(
            packed.as_slice(),
            logic_bits.row_words,
            n_samples,
            selected.len(),
            p_c.as_slice(),
            y.as_slice(),
            projected_y.as_slice(),
            candidate_first.as_slice(),
            candidate_second.as_slice(),
            candidates.len(),
            DosageChannel::FwlProduct,
            1.0,
        )
        .unwrap();
        expected.push(
            exact
                .top_order
                .iter()
                .map(|&index| {
                    (
                        max6_diagnostic_pair_key(
                            selected[exact.pair_i[index]],
                            selected[exact.pair_j[index]],
                        )
                        .unwrap(),
                        exact.max6[index],
                        exact.best_max6_channel[index],
                    )
                })
                .collect::<Vec<_>>(),
        );
    }
    let mut scans = windows
        .iter()
        .zip(window_candidates)
        .enumerate()
        .map(|(unit_index, (selected, candidates))| {
            (
                unit_index,
                Max6GrmMax6UnitScan {
                    outputs: Vec::new(),
                    permutation_pool: None,
                    deferred_screen: Some(Max6DeferredWindowScreen {
                        selected_global_rows: selected.clone(),
                        candidates,
                    }),
                    screen_pairs_passed: 3,
                    final_screen_subset_audit_pairs_checked: 0,
                    candidate_pair_keys: Vec::new(),
                    candidate_score_observations: Vec::new(),
                    window_unique_candidate_endpoints: 0,
                    timing: Max6ObservedUnitTiming::default(),
                    skipped_message: None,
                },
            )
        })
        .collect::<Vec<_>>();
    let stats = rerank_max6_grmmax6_screen_batch_with_audit(
        scans.as_mut_slice(),
        &logic_bits,
        Some(p_c.as_slice()),
        None,
        projected_y.as_slice(),
        y.as_slice(),
        1.0,
        100.0,
        3,
        false,
        false,
        false,
        true,
        1_073_741_824,
    )
    .unwrap();

    assert_eq!(stats.candidate_occurrences, 6);
    assert_eq!(stats.unique_candidate_pairs, 4);
    assert_eq!(stats.parity_audit.windows_checked, 2);
    assert_eq!(stats.parity_audit.pairs_checked, 6);
    assert_eq!(stats.parity_audit.finite_mask_mismatch_pairs, 0);
    assert_eq!(stats.parity_audit.best_channel_mismatch_pairs, 0);
    assert!(stats
        .parity_audit
        .max_abs_by_channel
        .iter()
        .all(|error| *error <= 1.0e-12));
    assert_eq!(stats.parity_audit.rank_changed_pairs, 0);
    assert_eq!(stats.parity_audit.max_rank_displacement, 0);
    assert_eq!(stats.parity_audit.topk_set_symmetric_difference, [0; 4]);
    for (window_index, (_, scan)) in scans.iter().enumerate() {
        let observed = scan
            .outputs
            .iter()
            .map(|output| {
                (
                    max6_diagnostic_pair_key(output.first_global_row, output.second_global_row)
                        .unwrap(),
                    output.max6,
                    output.best_channel,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            observed.iter().map(|item| item.0).collect::<Vec<_>>(),
            expected[window_index]
                .iter()
                .map(|item| item.0)
                .collect::<Vec<_>>()
        );
        for (got, want) in observed.iter().zip(expected[window_index].iter()) {
            assert!((got.1 - want.1).abs() <= 1.0e-12);
            assert_eq!(got.2, want.2);
        }
    }
}

#[test]
fn max6_candidate_score_reuse_summary_detects_repeated_score_differences() {
    let mut observations = vec![
        Max6CandidateScoreObservation {
            pair_key: 2,
            window_index: 0,
            tie_key: 2,
            score: 3.5,
            cached_score: 3.5,
            channel_id: 2,
        },
        Max6CandidateScoreObservation {
            pair_key: 1,
            window_index: 0,
            tie_key: 1,
            score: 1.25,
            cached_score: 1.25,
            channel_id: 0,
        },
        Max6CandidateScoreObservation {
            pair_key: 2,
            window_index: 1,
            tie_key: 2,
            score: 3.0,
            cached_score: 3.0,
            channel_id: 3,
        },
        Max6CandidateScoreObservation {
            pair_key: 3,
            window_index: 2,
            tie_key: 3,
            score: 7.0,
            cached_score: 7.0,
            channel_id: 4,
        },
        Max6CandidateScoreObservation {
            pair_key: 1,
            window_index: 1,
            tie_key: 1,
            score: 1.25,
            cached_score: 1.25,
            channel_id: 0,
        },
    ];

    let summary = summarize_max6_candidate_score_reuse_observations(&mut observations);

    assert_eq!(summary.observation_count, 5);
    assert_eq!(summary.unique_pair_count, 3);
    assert_eq!(summary.repeated_pair_count, 2);
    assert_eq!(summary.duplicate_occurrence_count, 2);
    assert_eq!(summary.bitwise_equal_repeated_pair_count, 1);
    assert_eq!(summary.score_mismatch_pair_count, 1);
    assert_eq!(summary.best_channel_mismatch_pair_count, 1);
    assert_eq!(summary.max_abs_score_difference, 0.5);
    assert_eq!(summary.topk_stability[0].windows_compared, 3);
    assert_eq!(summary.topk_stability[0].windows_with_membership_change, 0);
}

#[test]
fn max6_candidate_score_reuse_topk_audit_detects_first_seen_score_membership_change() {
    let mut observations = Vec::new();
    for pair_key in 1..=11u64 {
        let first_score = if pair_key == 1 { 10.0 } else { 0.0 };
        let later_score = if pair_key == 1 {
            0.0
        } else {
            (pair_key - 1) as f64
        };
        observations.push(Max6CandidateScoreObservation {
            pair_key,
            window_index: 0,
            tie_key: pair_key as u32,
            score: first_score,
            cached_score: first_score,
            channel_id: 0,
        });
        observations.push(Max6CandidateScoreObservation {
            pair_key,
            window_index: 1,
            tie_key: pair_key as u32,
            score: later_score,
            cached_score: later_score,
            channel_id: 0,
        });
    }

    let summary = summarize_max6_candidate_score_reuse_observations(&mut observations);

    assert_eq!(summary.topk_stability[0].windows_compared, 2);
    assert_eq!(summary.topk_stability[0].windows_with_membership_change, 1);
    assert_eq!(
        summary.topk_stability[0].total_membership_symmetric_difference,
        2
    );
    assert_eq!(
        summary.topk_stability[0].max_membership_symmetric_difference,
        2
    );
}

#[test]
fn max6_prep_profile_line_reports_wall_time_and_memory_metrics() {
    assert_eq!(
            format_max6_genotype_prep_profile_line(
                "ld_selection",
                1.25,
                2.5,
                Some(1_073_741_824),
                Some(1_610_612_736),
                Some(1_610_612_736),
                Some(2_147_483_648),
                Some(2_684_354_560),
                Some(2_684_354_560),
                "units=41206",
            ),
            "[MAX6-PREP] phase=ld_selection wall=1.250s total=2.500s rss=1.00->1.50GiB (sampled_peak=1.50GiB) footprint=2.00->2.50GiB (sampled_peak=2.50GiB) units=41206"
        );
}
use crate::ml::univariate::feature_scores_abs_corr_dosage_x;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn test_max6_prepared_fingerprint(
    sample_ids: &[&str],
    config_fingerprint: u64,
    file_generation: u128,
) -> Max6PreparedInputFingerprint {
    Max6PreparedInputFingerprint {
        files: vec![Max6PreparedFileFingerprint {
            path: "/tmp/genotype/test.bed".to_string(),
            length: 96,
            modified_unix_nanos: file_generation,
        }],
        ordered_sample_ids: sample_ids
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        config_fingerprint,
    }
}

#[test]
fn max6_prepared_fingerprint_tracks_pgen_file_trio() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "janusx-max6-pgen-fingerprint-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    let prefix = dir.join("tiny");
    for extension in ["pgen", "pvar", "psam"] {
        fs::write(prefix.with_extension(extension), b"test").unwrap();
    }
    let config = Max6PreparationConfigFingerprint::new(
        "window",
        None,
        None,
        5000,
        Some(10000),
        None,
        &[0, 1],
        0.02,
        0.05,
        1.0,
        false,
        GARFIELD_MAX6_LD_TAG_R2_DEFAULT,
    );

    let fingerprint = build_max6_prepared_input_fingerprint(
        prefix.to_str().unwrap(),
        vec!["s1".to_string(), "s2".to_string()],
        &config,
    )
    .unwrap();
    let fingerprint_with_suffix = build_max6_prepared_input_fingerprint(
        format!("{}.pgen", prefix.display()).as_str(),
        vec!["s1".to_string(), "s2".to_string()],
        &config,
    )
    .unwrap();

    assert_eq!(fingerprint.files.len(), 3);
    assert_eq!(fingerprint.files, fingerprint_with_suffix.files);
    assert!(fingerprint.files[0].path.ends_with("tiny.pgen"));
    assert!(fingerprint.files[1].path.ends_with("tiny.pvar"));
    assert!(fingerprint.files[2].path.ends_with("tiny.psam"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn max6_genotype_preparation_does_not_write_ld_tag_spool() {
    Python::initialize();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let fixture_dir = std::env::temp_dir().join(format!(
        "janusx-max6-no-ld-spool-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&fixture_dir).unwrap();
    let prefix = fixture_dir.join("tiny");
    let sample_ids = (0..32usize)
        .map(|index| format!("sample{index}"))
        .collect::<Vec<_>>();

    let mut fam = BufWriter::new(File::create(format!("{}.fam", prefix.display())).unwrap());
    for sample_id in &sample_ids {
        writeln!(fam, "{sample_id}\t{sample_id}\t0\t0\t1\t-9").unwrap();
    }
    fam.flush().unwrap();

    let genotype_rows = (0..3usize)
        .map(|row| {
            (0..sample_ids.len())
                .map(|sample| u8::from((sample >> row) & 1 == 0) * 2)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut bed = BufWriter::new(File::create(format!("{}.bed", prefix.display())).unwrap());
    bed.write_all(&[0x6c, 0x1b, 0x01]).unwrap();
    let bytes_per_snp = sample_ids.len().div_ceil(4);
    for row in &genotype_rows {
        let mut packed = vec![0u8; bytes_per_snp];
        for (sample, &genotype) in row.iter().enumerate() {
            packed[sample >> 2] |= plink2bits_from_logic_g(genotype as i8) << ((sample & 3) * 2);
        }
        bed.write_all(&packed).unwrap();
    }
    bed.flush().unwrap();

    let mut bim = BufWriter::new(File::create(format!("{}.bim", prefix.display())).unwrap());
    for row in 0..genotype_rows.len() {
        writeln!(bim, "1\trs{row}\t0\t{}\tA\tG", (row + 1) * 100).unwrap();
    }
    bim.flush().unwrap();

    let temp_dir = std::env::temp_dir();
    let spool_prefix = format!("janusx-max6-prepared-ld-tags.{}.", std::process::id());
    let spool_names = || {
        fs::read_dir(&temp_dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                name.starts_with(&spool_prefix).then_some(name)
            })
            .collect::<std::collections::BTreeSet<_>>()
    };
    let spool_names_before = spool_names();

    let tags = prepare_max6_genotype_tags_owned(
        prefix.to_string_lossy().into_owned(),
        sample_ids,
        (0..32usize).collect(),
        "wholegenome".to_string(),
        None,
        None,
        0,
        None,
        None,
        0.01,
        0.0,
        1.0,
        false,
        1,
        0,
        GARFIELD_MAX6_LD_TAG_R2_DEFAULT,
    )
    .unwrap();

    assert_eq!(tags.active_source_rows.len(), 3);
    assert_eq!(spool_names(), spool_names_before);

    let universe = finalize_max6_genotype_universe(Arc::new(tags)).unwrap();
    assert_eq!(universe.pair_universe_size, 3);

    drop(universe);
    fs::remove_dir_all(fixture_dir).unwrap();
}

#[test]
fn max6_prepared_input_rejects_sample_order_resource_and_parameter_changes() {
    let original = test_max6_prepared_fingerprint(&["s1", "s2", "s3"], 41, 100);

    assert!(original.ensure_compatible(&original).is_ok());
    assert!(original
        .ensure_compatible(&test_max6_prepared_fingerprint(
            &["s2", "s1", "s3"],
            41,
            100,
        ))
        .is_err());
    assert!(original
        .ensure_compatible(&test_max6_prepared_fingerprint(
            &["s1", "s2", "s3"],
            41,
            101
        ))
        .is_err());
    assert!(original
        .ensure_compatible(&test_max6_prepared_fingerprint(
            &["s1", "s2", "s3"],
            42,
            100
        ))
        .is_err());
}

#[test]
fn max6_prepared_universe_rejects_empty_pair_family() {
    let fingerprint = test_max6_prepared_fingerprint(&["s1", "s2"], 7, 100);
    let tags = Arc::new(Max6PreparedGenotypeTags {
        input_fingerprint: fingerprint,
        unit_kind: "window".to_string(),
        units: Vec::new(),
        screen_units: Vec::new(),
        original_unit_indices: Vec::new(),
        active_source_rows: Vec::new(),
        active_source_sites: Arc::new(CompactMax6SiteMetadata::default()),
        source_marker_count: 0,
        window_pair_count: 0,
        global_site_keep: Vec::new(),
    });
    let result = Max6PreparedGenotypeUniverse::new(tags, 0);

    assert!(result
        .unwrap_err()
        .contains("nonempty unique-pair universe"));
}

#[test]
fn max6_prepared_trait_qc_drops_tags_without_reselection_and_keeps_global_family() {
    let metadata = Max6PreparedUnitMetadata {
        original_unit_index: 8,
        label: "window-8".to_string(),
        spans: vec![GarfieldUnitSpan {
            chrom: "1".to_string(),
            bp_start: 10,
            bp_end: 30,
        }],
    };
    let tags = Arc::new(Max6PreparedGenotypeTags {
        input_fingerprint: test_max6_prepared_fingerprint(&["s1", "s2", "s3"], 7, 100),
        unit_kind: "window".to_string(),
        units: vec![metadata],
        screen_units: vec![Max6GrmMax6LdPreparedUnit {
            selected_global_rows: vec![10, 20, 30],
            pair_groups: Some(vec![0, 1, 2]),
            ld_compression: Vec::new(),
        }],
        original_unit_indices: vec![8],
        active_source_rows: vec![10, 20, 30],
        active_source_sites: Arc::new(
            CompactMax6SiteMetadata::from_sites(
                (0..3)
                    .map(|index| SiteInfo {
                        chrom: "1".to_string(),
                        pos: index,
                        snp: format!("rs{index}"),
                        ref_allele: "A".to_string(),
                        alt_allele: "G".to_string(),
                    })
                    .collect(),
            )
            .unwrap(),
        ),
        source_marker_count: 40,
        window_pair_count: 3,
        global_site_keep: vec![true; 40],
    });
    let universe = Max6PreparedGenotypeUniverse::new(tags.clone(), 3).unwrap();

    let (units, trait_universe) =
        remap_max6_prepared_universe_for_trait(&universe.tags, &[10, 30], 2).unwrap();
    let trait_pair_count = count_max6_pairs_by_membership(
        trait_universe.active_rows.as_slice(),
        trait_universe.window_pair_count,
        &trait_universe.pair_ownership,
    )
    .unwrap();

    assert_eq!(universe.pair_universe_size, 3);
    assert_eq!(units.len(), 1);
    // The prepared unit owns these rows; display/scheduling metadata must
    // not retain a second copy for every overlapping scan window.
    assert!(units[0].indices.is_empty());
    assert_eq!(trait_universe.original_unit_indices, vec![8]);
    assert_eq!(trait_universe.units[0].selected_global_rows, vec![0, 1]);
    assert_eq!(trait_universe.units[0].pair_groups, Some(vec![0, 2]));
    assert_eq!(trait_pair_count, 1);
}

#[test]
fn max6_prepared_screen_universe_preserves_source_unit_ids_when_singletons_are_skipped() {
    let (bits_flat, row_words) = pack_test_binary_rows(&[
        vec![1u8, 1, 0, 0, 0, 1],
        vec![1u8, 1, 0, 0, 1, 0],
        vec![1u8, 0, 1, 0, 0, 1],
    ]);
    let logic_bits = test_logic_bits_with_rows(bits_flat, row_words, 3, 6);
    let units = vec![
        GarfieldLogicUnit {
            label: "singleton".to_string(),
            indices: vec![0],
            spans: vec![GarfieldUnitSpan {
                chrom: "1".to_string(),
                bp_start: 10,
                bp_end: 10,
            }],
        },
        GarfieldLogicUnit {
            label: "pair".to_string(),
            indices: vec![1, 2],
            spans: vec![GarfieldUnitSpan {
                chrom: "1".to_string(),
                bp_start: 20,
                bp_end: 30,
            }],
        },
    ];

    let universe = prepare_max6_screen_universe(
        &logic_bits,
        &units,
        &[0, 1],
        "window",
        &[0, 1, 2, 3, 4, 5],
        None,
        None,
        GARFIELD_MAX6_LD_TAG_R2_DEFAULT,
    )
    .unwrap();

    assert_eq!(universe.original_unit_indices, vec![0, 1]);
    assert_eq!(universe.units.len(), 2);
    assert_eq!(universe.units[0].selected_global_rows.len(), 1);
    assert_eq!(universe.units[1].selected_global_rows.len(), 2);
}

#[test]
fn max6_membership_scan_planner_keeps_overlapping_units_in_one_batch() {
    let prepared_units = vec![
        Max6GrmMax6LdPreparedUnit {
            selected_global_rows: (0..8).collect(),
            pair_groups: None,
            ld_compression: Vec::new(),
        },
        Max6GrmMax6LdPreparedUnit {
            selected_global_rows: (4..12).collect(),
            pair_groups: None,
            ld_compression: Vec::new(),
        },
    ];
    let active_rows = (0..12).collect::<Vec<_>>();
    let pair_ownership = Max6ScreenPairOwnership::build(&prepared_units, &active_rows, 12).unwrap();
    let universe = Max6ScreenPreparedUniverse {
        units: prepared_units,
        original_unit_indices: vec![0, 1],
        active_rows,
        pair_ownership,
        window_pair_count: 56,
    };
    let units = vec![
        GarfieldLogicUnit {
            label: "left".to_string(),
            indices: Vec::new(),
            spans: Vec::new(),
        },
        GarfieldLogicUnit {
            label: "right".to_string(),
            indices: Vec::new(),
            spans: Vec::new(),
        },
    ];

    let chunks = max6_membership_scan_task_chunks(&[0, 1], &units, &universe, 4, 10).unwrap();

    assert_eq!(chunks, vec![vec![0, 1]]);
}

#[test]
fn max6_ld_tag_preparation_is_deterministic_across_thread_counts() {
    let n_samples = 24usize;
    let rows = (0..12usize)
        .map(|row| {
            (0..n_samples)
                .map(|sample| u8::from((sample * 13 + row * 7 + sample * row) % 31 < 11))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (bits_flat, row_words) = pack_test_binary_rows(&rows);
    let logic_bits = test_logic_bits_with_rows(bits_flat, row_words, rows.len(), n_samples);
    let units = vec![
        GarfieldLogicUnit {
            label: "w0".to_string(),
            indices: (0..8).collect(),
            spans: vec![GarfieldUnitSpan {
                chrom: "1".to_string(),
                bp_start: 1,
                bp_end: 8,
            }],
        },
        GarfieldLogicUnit {
            label: "w1".to_string(),
            indices: (4..12).collect(),
            spans: vec![GarfieldUnitSpan {
                chrom: "1".to_string(),
                bp_start: 5,
                bp_end: 12,
            }],
        },
    ];
    let scan_indices = [0usize, 1];
    let train_indices = (0..n_samples).collect::<Vec<_>>();
    let serial = prepare_max6_screen_universe(
        &logic_bits,
        &units,
        &scan_indices,
        "window",
        &train_indices,
        None,
        None,
        GARFIELD_MAX6_LD_TAG_R2_DEFAULT,
    )
    .unwrap();
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let parallel = prepare_max6_screen_universe(
        &logic_bits,
        &units,
        &scan_indices,
        "window",
        &train_indices,
        Some(&pool),
        None,
        GARFIELD_MAX6_LD_TAG_R2_DEFAULT,
    )
    .unwrap();

    assert_eq!(parallel.units, serial.units);
    assert_eq!(parallel.original_unit_indices, serial.original_unit_indices);
    assert_eq!(parallel.active_rows, serial.active_rows);
    assert_eq!(parallel.window_pair_count, serial.window_pair_count);
    assert_eq!(
        count_max6_screen_pairs_by_signature(&parallel).unwrap(),
        count_max6_screen_pairs_by_signature(&serial).unwrap()
    );

    let genotype_global = prepare_max6_screen_selection(
        &logic_bits,
        &units,
        &scan_indices,
        "window",
        &train_indices,
        Some(&pool),
        None,
        GARFIELD_MAX6_LD_TAG_R2_DEFAULT,
    )
    .unwrap();
    assert_eq!(genotype_global.units, parallel.units);
    assert_eq!(genotype_global.active_rows, parallel.active_rows);
    assert_eq!(
        genotype_global.original_unit_indices,
        parallel.original_unit_indices
    );
    assert_eq!(
        genotype_global.window_pair_count,
        parallel.window_pair_count
    );
}

#[test]
fn max6_permutation_batches_report_monotone_progress_on_caller_thread() {
    let jobs = (0..23usize).collect::<Vec<_>>();
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let caller_thread = std::thread::current().id();
    let mut progress = Vec::new();

    let scores = run_max6_permutation_jobs_batched(
        jobs.as_slice(),
        Some(&pool),
        |job| Ok(job * 2),
        |done, total| {
            assert_eq!(std::thread::current().id(), caller_thread);
            progress.push((done, total));
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(scores, (0..23usize).map(|job| job * 2).collect::<Vec<_>>());
    assert_eq!(
        progress,
        vec![
            (0, 23),
            (4, 23),
            (8, 23),
            (12, 23),
            (16, 23),
            (20, 23),
            (23, 23),
        ]
    );
}

#[test]
fn max6_permutation_batches_report_serial_completion() {
    let jobs = [3usize, 5, 8];
    let mut progress = Vec::new();
    let scores = run_max6_permutation_jobs_batched(
        &jobs,
        None,
        |job| Ok(job * job),
        |done, total| {
            progress.push((done, total));
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(scores, vec![9, 25, 64]);
    assert_eq!(progress, vec![(0, 3), (1, 3), (2, 3), (3, 3)]);
}

#[test]
fn max6_grmmax6_output_schema_and_score_format_are_stable() {
    assert_eq!(
            MAX6_GRMMAX6_PAIRS_HEADER,
            "snp1\tsnp2\tchrom1\tpos1\tchrom2\tpos2\tGRM_MAX6\tbest_channel\t4cell\tmincell\tpair_PVE\tpwald"
        );
    assert_eq!(format_max6_score_value(0.123456), "0.1235");
    assert_eq!(format_max6_score_value(0.00009999), "9.9990e-5");
    assert_eq!(format_max6_score_value(0.0), "0.0000e0");
    assert_eq!(
        max6_grmmax6_pairs_output_path("test.20260724/cubic2.PH.garfield"),
        "test.20260724/cubic2.PH.garfield.tsv"
    );
}

#[test]
fn max6_analytic_pair_row_writes_pvalue_from_unrounded_exact_score() {
    let dir = make_temp_dir("max6_pvalue_row");
    let path = dir.join("pair.tsv");
    let mut writer = BufWriter::new(File::create(&path).unwrap());
    let row = Max6GrmMax6PairOutput {
        unit_index: 0,
        candidate_rank: 1,
        first_global_row: 0,
        second_global_row: 1,
        four_cell_counts: None,
        dosage: f64::NAN,
        max6: 3.841458820694124,
        best_channel: "AND11",
        pair_pve: 0.125,
    };
    let sites = vec![
        GarfieldLogicSite::owned(
            "Chr01",
            100,
            "Chr01:100",
            "A",
            "G",
            GarfieldLogicSiteMode::Bin,
        ),
        GarfieldLogicSite::owned(
            "Chr01",
            200,
            "Chr01:200",
            "C",
            "T",
            GarfieldLogicSiteMode::Bin,
        ),
    ];
    let four_cell_counts = [10, 11, 200, 300];
    write_max6_grmmax6_pair_spool_record(&mut writer, &row, &sites, four_cell_counts, false)
        .unwrap();
    let mut small_p_row = row.clone();
    small_p_row.max6 = 25.0;
    small_p_row.best_channel = "DOSAGE_PRODUCT";
    write_max6_grmmax6_pair_spool_record(
        &mut writer,
        &small_p_row,
        &sites,
        four_cell_counts,
        false,
    )
    .unwrap();
    let mut near_cut_row = row.clone();
    near_cut_row.max6 = 15.0;
    write_max6_grmmax6_pair_spool_record(
        &mut writer,
        &near_cut_row,
        &sites,
        four_cell_counts,
        false,
    )
    .unwrap();
    writer.flush().unwrap();
    let line = fs::read_to_string(&path).unwrap();
    let rows = line
        .lines()
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_eq!(rows[0].len(), 12);
    assert_eq!(rows[0][7], "AND(GT)");
    assert_eq!(rows[0][8], "AC:10;AT:11;GC:200;GT:300");
    assert_eq!(rows[0][9], "10");
    assert_eq!(rows[0][10], "0.1250");
    assert_eq!(rows[0][11], "5.0000e-2");
    assert_eq!(rows[1][7], "Dosage");
    assert_eq!(rows[1][11], "5.7330e-7");
    assert_eq!(rows[2][11], "1.0751e-4");
    assert_eq!(format_max6_pvalue(Some(1.0)), "1.0000e0");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn max6_output_channel_labels_follow_physical_allele_order() {
    let later_site = GarfieldLogicSite::owned(
        "Chr02",
        200,
        "Chr02:200",
        "C",
        "T",
        GarfieldLogicSiteMode::Bin,
    );
    let earlier_site = GarfieldLogicSite::owned(
        "Chr01",
        100,
        "Chr01:100",
        "A",
        "G",
        GarfieldLogicSiteMode::Bin,
    );

    assert_eq!(
        max6_output_channel_label("AND00", &later_site, &earlier_site),
        "AND(AC)"
    );
    assert_eq!(
        max6_output_channel_label("AND01", &later_site, &earlier_site),
        "AND(GC)"
    );
    assert_eq!(
        max6_output_channel_label("AND10", &later_site, &earlier_site),
        "AND(AT)"
    );
    assert_eq!(
        max6_output_channel_label("AND11", &later_site, &earlier_site),
        "AND(GT)"
    );
    assert_eq!(
        max6_output_channel_label("XOR", &later_site, &earlier_site),
        "XOR(AT,GC)"
    );
    assert_eq!(
        max6_output_channel_label("DOSAGE_PRODUCT", &later_site, &earlier_site),
        "Dosage"
    );
}

#[test]
fn max6_grm_int_pair_table_reports_exact_max6_channel_and_interaction_chisq() {
    let dir = make_temp_dir("max6_grm_int_row");
    let spool_path = dir.join("spool.tsv");
    let output_path = dir.join("pairs.tsv");
    let mut spool = BufWriter::new(File::create(&spool_path).unwrap());
    let row = Max6GrmMax6PairOutput {
        unit_index: 0,
        candidate_rank: 6,
        first_global_row: 0,
        second_global_row: 1,
        four_cell_counts: None,
        dosage: 3.841458820694124,
        max6: 101.0,
        best_channel: "AND01",
        pair_pve: f64::NAN,
    };
    let sites = vec![
        GarfieldLogicSite::owned(
            "Chr01",
            100,
            "Chr01:100",
            "A",
            "G",
            GarfieldLogicSiteMode::Bin,
        ),
        GarfieldLogicSite::owned(
            "Chr01",
            200,
            "Chr01:200",
            "C",
            "T",
            GarfieldLogicSiteMode::Bin,
        ),
    ];
    let four_cell_counts = [10, 11, 200, 300];
    write_max6_grmmax6_pair_spool_record(&mut spool, &row, &sites, four_cell_counts, true).unwrap();
    let mut invalid_row = row.clone();
    invalid_row.dosage = f64::NEG_INFINITY;
    write_max6_grmmax6_pair_spool_record(&mut spool, &invalid_row, &sites, four_cell_counts, true)
        .unwrap();
    spool.flush().unwrap();
    write_max6_grmmax6_pairs_tsv(
        output_path.to_str().unwrap(),
        spool_path.to_str().unwrap(),
        true,
    )
    .unwrap();

    let contents = fs::read_to_string(&output_path).unwrap();
    let rows = contents
        .lines()
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_eq!(
            rows[0].join("\t"),
            "snp1\tsnp2\tchrom1\tpos1\tchrom2\tpos2\tmax6_rank\tmax6_score\tbest_channel\t4cell\tmincell\tgrm_int_chisq\tpwald"
        );
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[1].len(), 13);
    assert_eq!(rows[1][6], "6");
    assert_eq!(rows[1][7], "101.0000");
    assert_eq!(rows[1][8], "AND(AT)");
    assert_eq!(rows[1][9], "AC:10;AT:11;GC:200;GT:300");
    assert_eq!(rows[1][10], "10");
    assert!((rows[1][11].parse::<f64>().unwrap() - 3.841458820694124).abs() < 1e-4);
    assert_eq!(rows[1][12], "5.0000e-2");
    assert_eq!(rows[2][11], "NA");
    assert_eq!(rows[2][12], "NA");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn max6_analytic_mode_disables_implicit_null_draws() {
    assert!(!max6_null_calibration_enabled(true, true, false));
    assert!(max6_null_calibration_enabled(true, true, true));
    assert!(max6_null_calibration_enabled(false, true, false));
    assert!(!max6_null_calibration_enabled(false, false, true));
}

#[test]
fn max6_analytic_reference_pvalue_is_raw_chi_square_tail() {
    assert_eq!(max6_chi2_reference_pvalue(0.0), Some(1.0));
    assert!((max6_chi2_reference_pvalue(3.841458820694124).unwrap() - 0.05).abs() < 1e-14);
    assert_eq!(max6_chi2_reference_pvalue(10_000.0), Some(0.0));
    assert_eq!(max6_chi2_reference_pvalue(-1.0), None);
    assert_eq!(max6_chi2_reference_pvalue(f64::NAN), None);
}

#[test]
fn grm_int_keeps_max6_topk_members_and_ranks_only_that_subset_by_chisq() {
    let max6_selected = [1, 0];
    let interaction_chisq = [2.0, 8.0, 80.0, f64::NAN];
    let order = grm_int_chisq_order(&interaction_chisq, &max6_selected);
    assert_eq!(order, vec![1, 0]);

    let max6_selected_with_nonfinite = [1, 0, 3];
    let order = grm_int_chisq_order(&interaction_chisq, &max6_selected_with_nonfinite);
    assert_eq!(order, vec![1, 0, 3]);
}

#[test]
fn grm_int_projection_validation_rejects_nonfinite_values() {
    assert!(validate_grm_int_dense_projection(2, &[1.0, 0.0, 0.0, 1.0]).is_ok());
    let error = validate_grm_int_dense_projection(2, &[1.0, f64::NAN, 0.0, 1.0]).unwrap_err();
    assert!(error.contains("finite"));
}

#[test]
fn grm_int_provenance_sidecar_maps_output_rows_to_source_units() {
    let dir = make_temp_dir("max6_grm_int_provenance");
    let spool_path = dir.join("provenance.spool.tsv");
    let output_path = dir.join("provenance.tsv");
    let mut spool = BufWriter::new(File::create(&spool_path).unwrap());
    write_max6_grm_int_provenance_spool_record(&mut spool, 1, 7, "Chr01:100-200\twindow").unwrap();
    spool.flush().unwrap();

    write_max6_grm_int_provenance_tsv(output_path.to_str().unwrap(), spool_path.to_str().unwrap())
        .unwrap();

    let contents = fs::read_to_string(&output_path).unwrap();
    assert_eq!(
        contents,
        "output_row_index\tunit_index_zero_based\tunit_label\n1\t7\tChr01:100-200\\twindow\n"
    );
    assert_eq!(
        max6_grm_int_provenance_output_path("rice.garfield"),
        "rice.garfield.max6_grmmax6.provenance.tsv"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn exact_max6_output_removes_stale_grm_int_provenance_sidecar() {
    let dir = make_temp_dir("max6_stale_grm_int_provenance");
    let prefix = dir.join("run.garfield").to_string_lossy().into_owned();
    let provenance_path = max6_grm_int_provenance_output_path(prefix.as_str());
    fs::write(&provenance_path, "stale").unwrap();

    remove_stale_max6_grm_int_provenance(prefix.as_str(), false).unwrap();

    assert!(!Path::new(provenance_path.as_str()).exists());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn grm_int_threshold_metadata_does_not_reuse_exact_max6_cutoff() {
    assert_eq!(
        max6_threshold_method_for_output(true, Some(12.0), None, true),
        "grm_int_nominal_chi2_1_after_selection"
    );
    assert_eq!(
        max6_threshold_method_for_output(false, Some(12.0), None, true),
        "chi2_1_isf_alpha_over_6N_unique_pairs"
    );
    assert_eq!(
        max6_threshold_method_for_output(false, None, Some("screen_maxT"), true),
        "screen_maxT"
    );
    assert_eq!(
        rule_null_generation_for_output(true, false, true, true, false),
        None
    );
    assert_eq!(
        rule_null_generation_for_output(false, false, true, true, false),
        Some("global_pair_score_permutation_fixed_pc")
    );
    assert_eq!(
        rule_null_generation_for_output(false, false, false, false, true),
        Some("lm_parametric_bootstrap")
    );
}

#[test]
fn grm_int_screen_uses_requested_topk_and_ignores_screen_threshold() {
    assert_eq!(max6_grmmax6_screen_top_k(true, 1), 1);
    assert_eq!(max6_grmmax6_screen_top_k(true, 37), 37);
    assert_eq!(max6_grmmax6_screen_top_k(true, 0), 1);
    assert_eq!(
        max6_grmmax6_screen_top_k(false, 37),
        MAX6_GRMMAX6_SCREEN_TOPK
    );

    assert_eq!(max6_grmmax6_screen_threshold(true, Some(9.0)), None);
    assert_eq!(max6_grmmax6_screen_threshold(false, Some(9.0)), Some(9.0));
    assert_eq!(max6_grmmax6_screen_threshold(false, None), None);
}

fn grm_int_test_screen_candidate(
    first: usize,
    second: usize,
    gate: logic_residual_screen::FourCellGate,
    proposal_score: f64,
) -> logic_residual_screen::ResidualScreenCandidate {
    let stats = logic_residual_screen::FourCellSufficientStats {
        counts: [3, 2, 2, 3],
        sums: [0.3, -0.2, 0.1, 0.7],
        n_valid: 10,
    };
    let derived = super::four_cell::derive_gate5_gate6d_scores(stats);
    let best_gate = super::four_cell::FourCellGateScore {
        gate,
        score: proposal_score,
        signed_score: proposal_score,
        n_hit: 3,
    };
    logic_residual_screen::ResidualScreenCandidate {
        first,
        second,
        best_gate,
        proposal_score,
        proposal_source: logic_residual_screen::ResidualProposalSource::Max6,
        stats,
        derived,
    }
}

#[test]
fn grm_int_outputs_sort_only_selected_pairs_and_keep_screen_metadata() {
    let selected_global_rows = [0, 1, 2, 3];
    let sites = vec![
        GarfieldLogicSite::owned(
            "Chr01",
            100,
            "Chr01:100",
            "A",
            "G",
            GarfieldLogicSiteMode::Bin,
        ),
        GarfieldLogicSite::owned(
            "Chr01",
            200,
            "Chr01:200",
            "C",
            "T",
            GarfieldLogicSiteMode::Bin,
        ),
        GarfieldLogicSite::owned(
            "Chr01",
            300,
            "Chr01:300",
            "A",
            "C",
            GarfieldLogicSiteMode::Bin,
        ),
        GarfieldLogicSite::owned(
            "Chr01",
            400,
            "Chr01:400",
            "G",
            "T",
            GarfieldLogicSiteMode::Bin,
        ),
    ];
    let candidates = [
        grm_int_test_screen_candidate(0, 1, logic_residual_screen::FourCellGate::And10, 9.0),
        grm_int_test_screen_candidate(1, 2, logic_residual_screen::FourCellGate::Xor, 8.0),
        grm_int_test_screen_candidate(
            2,
            3,
            logic_residual_screen::FourCellGate::DosageInteraction,
            7.0,
        ),
    ];
    let chisq = [2.0, 11.0, 50.0];
    let outputs =
        max6_grm_int_outputs_from_screen(4, &selected_global_rows, &sites, &candidates, &chisq)
            .unwrap();

    assert_eq!(outputs.len(), 3);
    assert_eq!(
        outputs
            .iter()
            .map(|output| (output.first_global_row, output.second_global_row))
            .collect::<Vec<_>>(),
        vec![(2, 3), (1, 2), (0, 1)]
    );
    assert_eq!(outputs[0].unit_index, 4);
    assert_eq!(outputs[0].candidate_rank, 3);
    assert_eq!(outputs[0].max6, 7.0);
    assert_eq!(outputs[0].best_channel, "DOSAGE_PRODUCT");
    assert_eq!(outputs[0].dosage, 50.0);
    assert!(outputs[0].pair_pve.is_nan());
    assert!(outputs[0].four_cell_counts.is_some());
    assert_eq!(outputs[1].best_channel, "XOR");
    assert_eq!(outputs[1].max6, 8.0);
    assert_eq!(outputs[1].dosage, 11.0);
    assert_eq!(outputs[2].best_channel, "AND10");
    assert_eq!(outputs[2].max6, 9.0);
    assert_eq!(outputs[2].dosage, 2.0);
}

#[test]
fn grm_int_keeps_same_pair_occurrence_for_each_overlapping_unit() {
    let selected_global_rows = [0, 1];
    let sites = vec![
        GarfieldLogicSite::owned(
            "Chr01",
            100,
            "Chr01:100",
            "A",
            "G",
            GarfieldLogicSiteMode::Bin,
        ),
        GarfieldLogicSite::owned(
            "Chr01",
            200,
            "Chr01:200",
            "C",
            "T",
            GarfieldLogicSiteMode::Bin,
        ),
    ];
    let candidates = [grm_int_test_screen_candidate(
        0,
        1,
        logic_residual_screen::FourCellGate::And11,
        12.0,
    )];
    let first_unit =
        max6_grm_int_outputs_from_screen(2, &selected_global_rows, &sites, &candidates, &[4.0])
            .unwrap();
    let second_unit =
        max6_grm_int_outputs_from_screen(9, &selected_global_rows, &sites, &candidates, &[4.0])
            .unwrap();

    assert_eq!(first_unit.len(), 1);
    assert_eq!(second_unit.len(), 1);
    assert_eq!(first_unit[0].unit_index, 2);
    assert_eq!(second_unit[0].unit_index, 9);
    assert_eq!(
        (
            first_unit[0].first_global_row,
            first_unit[0].second_global_row
        ),
        (
            second_unit[0].first_global_row,
            second_unit[0].second_global_row
        )
    );
}

#[test]
fn grm_int_diagnostics_count_selected_occurrences_without_exact_scores() {
    let mut timing = Max6ObservedScanTiming::default();
    timing.accumulate_unit(Max6ObservedUnitTiming {
        screen_candidates: 3,
        grm_int_candidates: 3,
        grm_int: GrmIntScoreDiagnostics {
            singular_endpoint_design: 1,
            negligible_residual_interaction_information: 1,
            nonfinite: 0,
            batches: 1,
            max_workspace_bytes: 4096,
            elapsed_seconds: 0.25,
        },
        ..Default::default()
    });

    assert_eq!(timing.screen_candidate_pairs, 3);
    assert_eq!(timing.exact_candidate_pairs, 0);
    assert_eq!(timing.scored_candidate_pairs, 3);
    assert_eq!(timing.grm_int_pair_occurrences, 3);
    assert_eq!(timing.grm_int_singular_endpoint_design_pairs, 1);
    assert_eq!(timing.grm_int_negligible_interaction_information_pairs, 1);
    assert_eq!(timing.grm_int_batches, 1);
    assert_eq!(timing.grm_int_max_workspace_bytes, 4096);
    assert_eq!(timing.grm_int_task_seconds, 0.25);
}

#[test]
fn max6_analytic_bonferroni_cutoff_uses_six_times_full_family() {
    let (q_cut, p_cut) = max6_bonferroni_cutoff(20_000).unwrap();
    assert!((p_cut - 0.05 / 120_000.0).abs() < 1e-20);
    assert!((max6_chi2_reference_pvalue(q_cut).unwrap() / p_cut - 1.0).abs() < 1e-12);
    assert!(max6_bonferroni_cutoff(0).is_err());
}

#[test]
fn max6_analytic_universe_count_is_not_bounded_by_screen_topk() {
    let rows = (0..101).collect::<Vec<_>>();
    let units = vec![Max6GrmMax6LdPreparedUnit {
        selected_global_rows: rows.clone(),
        pair_groups: None,
        ld_compression: Vec::new(),
    }];
    let ownership = Max6ScreenPairOwnership::build(&units, &rows, rows.len()).unwrap();
    let universe = Max6ScreenPreparedUniverse {
        original_unit_indices: vec![0],
        units,
        active_rows: rows,
        pair_ownership: ownership,
        window_pair_count: 5_050,
    };
    assert_eq!(count_max6_screen_pairs(&universe).unwrap(), 5_050);
}

#[test]
fn max6_signature_pair_count_matches_owner_walk_for_overlapping_units() {
    let units = vec![
        Max6GrmMax6LdPreparedUnit {
            selected_global_rows: vec![0, 1, 2, 3],
            pair_groups: None,
            ld_compression: Vec::new(),
        },
        Max6GrmMax6LdPreparedUnit {
            selected_global_rows: vec![1, 2, 3, 4],
            pair_groups: None,
            ld_compression: Vec::new(),
        },
        Max6GrmMax6LdPreparedUnit {
            selected_global_rows: vec![2, 3, 4, 5],
            pair_groups: None,
            ld_compression: Vec::new(),
        },
    ];
    let active_rows = (0..6).collect::<Vec<_>>();
    let ownership =
        Max6ScreenPairOwnership::build(&units, &active_rows, active_rows.len()).unwrap();
    let universe = Max6ScreenPreparedUniverse {
        original_unit_indices: vec![0, 1, 2],
        units,
        active_rows,
        pair_ownership: ownership,
        window_pair_count: 18,
    };

    assert_eq!(count_max6_screen_pairs_by_signature(&universe).unwrap(), 12);
    assert_eq!(
        count_max6_screen_pairs_by_signature(&universe).unwrap(),
        count_max6_screen_pairs_owner_walk(&universe).unwrap()
    );
}

#[test]
fn max6_signature_pair_count_matches_owner_walk_with_group_exclusions() {
    let units = vec![
        Max6GrmMax6LdPreparedUnit {
            selected_global_rows: vec![0, 1, 2, 3],
            pair_groups: Some(vec![0, 0, 1, 1]),
            ld_compression: Vec::new(),
        },
        Max6GrmMax6LdPreparedUnit {
            selected_global_rows: vec![0, 1, 2, 3, 4],
            pair_groups: Some(vec![0, 1, 1, 0, 2]),
            ld_compression: Vec::new(),
        },
    ];
    let active_rows = (0..5).collect::<Vec<_>>();
    let ownership =
        Max6ScreenPairOwnership::build(&units, &active_rows, active_rows.len()).unwrap();
    let universe = Max6ScreenPreparedUniverse {
        original_unit_indices: vec![0, 1],
        units,
        active_rows,
        pair_ownership: ownership,
        window_pair_count: 10,
    };

    assert_eq!(count_max6_screen_pairs_by_signature(&universe).unwrap(), 10);
    assert_eq!(
        count_max6_screen_pairs_by_signature(&universe).unwrap(),
        count_max6_screen_pairs_owner_walk(&universe).unwrap()
    );
}

#[test]
fn max6_grouped_signature_count_handles_large_exact_class_set() {
    let class_count = 10_000u32;
    let signatures = (0..class_count)
        .map(|group| (vec![(0, group)], 1u128))
        .collect::<HashMap<_, _>>();
    let expected = u128::from(class_count) * u128::from(class_count - 1) / 2;

    let (count, _) = count_grouped_max6_signature_pairs(&signatures)
        .expect("large exact signature-class count should complete");

    assert_eq!(count, expected);
}

#[test]
fn max6_grouped_signature_bitset_matches_pairwise_weighted_reference() {
    let signatures = HashMap::from([
        (vec![(0, 0), (1, 0)], 2u128),
        (vec![(0, 0), (1, 1)], 3u128),
        (vec![(0, 1)], 4u128),
        (vec![(0, Max6ScreenPairOwnership::NO_GROUP), (1, 0)], 5u128),
        (vec![(3, 2)], 6u128),
        (Vec::new(), 7u128),
    ]);
    let mut ordered = signatures.iter().collect::<Vec<_>>();
    ordered.sort_by(|(left, _), (right, _)| left.cmp(right));
    let mut expected = 0u128;
    for left_index in 0..ordered.len() {
        let (left_signature, left_count) = ordered[left_index];
        for (right_index, (right_signature, right_count)) in
            ordered.iter().enumerate().skip(left_index)
        {
            if !max6_signature_pair_eligible(left_signature, right_signature) {
                continue;
            }
            let contribution = if left_index == right_index {
                count_choose_two_u128(*left_count).unwrap()
            } else {
                *left_count * *right_count
            };
            expected += contribution;
        }
    }

    let (observed, _) = count_grouped_max6_signature_pairs_bitset(&ordered).unwrap();

    assert_eq!(observed, expected);
}

#[test]
fn max6_grouped_signature_count_keeps_multigroup_pairwise_semantics() {
    let signatures = HashMap::from([
        (vec![(0, 0), (0, 1)], 2u128),
        (vec![(0, 0)], 3u128),
        (vec![(0, 0), (0, 2)], 4u128),
        (vec![(1, Max6ScreenPairOwnership::NO_GROUP)], 5u128),
    ]);
    let mut expected = 0u128;
    let ordered = signatures.iter().collect::<Vec<_>>();
    for left_index in 0..ordered.len() {
        let (left_signature, left_count) = ordered[left_index];
        for (right_index, (right_signature, right_count)) in
            ordered.iter().enumerate().skip(left_index)
        {
            if !max6_signature_pair_eligible(left_signature, right_signature) {
                continue;
            }
            expected += if left_index == right_index {
                count_choose_two_u128(*left_count).unwrap()
            } else {
                *left_count * *right_count
            };
        }
    }

    assert_eq!(
        count_grouped_max6_signature_pairs(&signatures).unwrap().0,
        expected
    );
}

#[test]
fn max6_ungrouped_signature_count_falls_back_to_sparse_exact_union() {
    let class_count = 10_000usize;
    let degree = 13usize;
    let mut offsets = Vec::with_capacity(class_count + 1);
    let mut signatures = Vec::with_capacity(class_count + 1);
    for class in 0..class_count {
        let pair = (class / 2) as u32;
        let private_start = class_count as u32 + (class * (degree - 2)) as u32;
        let mut signature = vec![pair, (class_count / 2) as u32 + pair];
        signature.extend(private_start..private_start + (degree - 2) as u32);
        signatures.push(signature);
    }
    signatures.push(signatures[0].clone());

    offsets.push(0u32);
    let mut membership_units = Vec::with_capacity(signatures.len() * degree);
    for signature in signatures.iter() {
        membership_units.extend_from_slice(signature.as_slice());
        offsets.push(membership_units.len() as u32);
    }
    let active_rows = (0..signatures.len()).collect::<Vec<_>>();
    let ownership = Max6ScreenPairOwnership {
        global_row_start: 0,
        global_to_active: (0..signatures.len() as u32).collect(),
        single_membership_units: vec![Max6ScreenPairOwnership::NO_GROUP; signatures.len()],
        offsets,
        membership_units,
        membership_groups: None,
    };

    let observed = count_max6_pairs_by_membership(&active_rows, 0, &ownership)
        .expect("large ungrouped signature sets should use the exact sparse fallback");

    assert_eq!(observed, class_count / 2 + 2);
}

#[test]
fn max6_signature_pair_count_reports_exact_completion_without_occurrence_walk() {
    let rows = (0..1_600).collect::<Vec<_>>();
    let units = vec![
        Max6GrmMax6LdPreparedUnit {
            selected_global_rows: (0..1_400).collect(),
            pair_groups: None,
            ld_compression: Vec::new(),
        },
        Max6GrmMax6LdPreparedUnit {
            selected_global_rows: (200..1_600).collect(),
            pair_groups: None,
            ld_compression: Vec::new(),
        },
    ];
    let ownership = Max6ScreenPairOwnership::build(&units, &rows, rows.len()).unwrap();
    let universe = Max6ScreenPreparedUniverse {
        original_unit_indices: vec![0, 1],
        units,
        active_rows: rows,
        pair_ownership: ownership,
        window_pair_count: 2 * 1_400 * 1_399 / 2,
    };
    let mut progress = Vec::new();
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    let count = count_max6_screen_pairs_with_progress(
        &universe,
        Some(&pool),
        &mut |done, total, unique| {
            progress.push((done, total, unique));
            Ok(())
        },
    )
    .unwrap();

    let total_work = 2 * 1_400 * 1_399 / 2;
    let unique_pairs = total_work - 1_200 * 1_199 / 2;
    assert_eq!(count, unique_pairs);
    assert_eq!(progress, vec![(0, 1, 0), (1, 1, unique_pairs)]);
}

#[test]
fn max6_disjoint_pair_count_uses_prepared_geometry() {
    let rows = (0..700).collect::<Vec<_>>();
    let units = vec![Max6GrmMax6LdPreparedUnit {
        selected_global_rows: rows.clone(),
        pair_groups: Some((0..700).map(|row| row % 2).collect()),
        ld_compression: Vec::new(),
    }];
    let ownership = Max6ScreenPairOwnership::build(&units, &rows, rows.len()).unwrap();
    let expected = 350 * 350;
    let universe = Max6ScreenPreparedUniverse {
        original_unit_indices: vec![0],
        units,
        active_rows: rows,
        pair_ownership: ownership,
        window_pair_count: expected,
    };
    let mut progress = Vec::new();
    let count =
        count_max6_screen_pairs_with_progress(&universe, None, &mut |done, total, unique| {
            progress.push((done, total, unique));
            Ok(())
        })
        .unwrap();
    assert_eq!(count, expected);
    assert_eq!(progress, vec![(0, 1, 0), (1, 1, expected)]);
}

#[test]
fn pair_pve_uses_the_scaled_total_quadratic_and_rejects_invalid_denominators() {
    assert!((pair_pve_from_scaled_score(25.0, 100.0) - 0.25).abs() < 1.0e-15);
    assert!(pair_pve_from_scaled_score(f64::NAN, 100.0).is_nan());
    assert!(pair_pve_from_scaled_score(25.0, 0.0).is_nan());
    assert!(pair_pve_from_scaled_score(25.0, f64::INFINITY).is_nan());
}

#[test]
fn empirical_score_quantile_uses_explicit_order_statistic() {
    let scores = [1.0, f64::NAN, 3.0, 2.0, f64::INFINITY];
    assert_eq!(empirical_score_quantile(&scores, 0.5), Some(2.0));
    assert_eq!(empirical_score_quantile(&scores, 0.99), Some(3.0));
    assert_eq!(empirical_score_quantile(&[], 0.99), None);
}

#[test]
fn max6_permutation_threshold_candidates_report_empirical_and_gumbel() {
    let scores = (1..=1_000).map(f64::from).collect::<Vec<_>>();
    let candidates = max6_permutation_threshold_candidates(scores.as_slice(), 0.99);
    assert_eq!(candidates.empirical, Some(990.0));
    assert!(candidates.gumbel.is_some_and(|value| value > 990.0));
}

#[test]
fn max6_validation_fwer_uses_strict_exceedance_and_binomial_interval() {
    let summary = summarize_max6_screen_validation(&[1.0, 2.0, 2.0, 3.0], 2.0).unwrap();
    assert_eq!(summary.maxima.len(), 4);
    assert_eq!(summary.rejections, 1);
    assert!((summary.empirical_fwer - 0.25).abs() < 1.0e-15);
    assert!(summary.ci_lower <= summary.empirical_fwer);
    assert!(summary.ci_upper >= summary.empirical_fwer);

    let no_rejections = summarize_max6_screen_validation(&[1.0, 2.0], 2.0).unwrap();
    assert_eq!(no_rejections.rejections, 0);
    assert_eq!(no_rejections.empirical_fwer, 0.0);
    assert!(no_rejections.ci_upper > 0.0);
    assert!(summarize_max6_screen_validation(&[1.0, f64::NAN], 2.0).is_err());
}

#[test]
fn max6_null_maxima_parallel_scan_matches_serial_and_reports_progress() {
    let n_rows = 12usize;
    let n_samples = 96usize;
    let mut logic_bits = build_test_logic_bits(n_rows, n_samples);
    logic_bits.bits_flat.fill(0);
    for row in 0..n_rows {
        for sample in 0..n_samples {
            if (sample * 17 + row * 11 + sample * row * 3) % 29 < 9 {
                logic_bits.bits_flat[row * logic_bits.row_words + (sample >> 6)] |=
                    1u64 << (sample & 63);
            }
        }
    }
    let units = vec![Max6GrmMax6LdPreparedUnit {
        selected_global_rows: (0..n_rows).collect(),
        pair_groups: None,
        ld_compression: Vec::new(),
    }];
    let active_rows = (0..n_rows).collect::<Vec<_>>();
    let universe = Max6ScreenPreparedUniverse {
        pair_ownership: Max6ScreenPairOwnership::build(
            units.as_slice(),
            active_rows.as_slice(),
            logic_bits.sites.len(),
        )
        .unwrap(),
        original_unit_indices: vec![0],
        units,
        active_rows,
        window_pair_count: n_rows * (n_rows - 1) / 2,
    };
    let null_y = (0..7usize)
        .map(|replicate| {
            (0..n_samples)
                .map(|sample| ((sample * 17 + replicate * 11) % 31) as f64 - 15.0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let (serial, _, serial_unique_pairs) = scan_max6_screen_null_maxima_with_progress(
        &logic_bits,
        &universe,
        null_y.as_slice(),
        None,
        false,
        |_, _, _| Ok(()),
    )
    .unwrap();
    let pool = ThreadPoolBuilder::new().num_threads(3).build().unwrap();
    let mut progress = Vec::new();
    let (parallel, _, parallel_unique_pairs) = scan_max6_screen_null_maxima_with_progress(
        &logic_bits,
        &universe,
        null_y.as_slice(),
        Some(&pool),
        false,
        |done, total, _| {
            progress.push((done, total));
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(serial, parallel);
    assert_eq!(serial_unique_pairs, n_rows * (n_rows - 1) / 2);
    assert_eq!(parallel_unique_pairs, serial_unique_pairs);
    for (replicate, null_y) in null_y.iter().enumerate() {
        let reference = scan_max6_screen_null_replicate_maximum(
            &logic_bits,
            &universe,
            null_y.as_slice(),
            replicate,
            None,
        )
        .unwrap();
        assert_eq!(serial[replicate].to_bits(), reference.to_bits());
    }
    assert_eq!(progress.first(), Some(&(0, null_y.len())));
    assert_eq!(progress.last(), Some(&(null_y.len(), null_y.len())));
    assert!(progress.windows(2).all(|window| window[0].0 <= window[1].0));
}

#[test]
fn max6_full_null_scanner_reports_deterministic_argmax_pair_on_ties() {
    let n_rows = 3usize;
    let n_samples = 96usize;
    let mut logic_bits = build_test_logic_bits(n_rows, n_samples);
    logic_bits.bits_flat.fill(0);
    for row in 0..n_rows {
        for sample in (0..n_samples).step_by(3) {
            logic_bits.bits_flat[row * logic_bits.row_words + (sample >> 6)] |=
                1u64 << (sample & 63);
        }
    }
    let units = vec![Max6GrmMax6LdPreparedUnit {
        selected_global_rows: (0..n_rows).collect(),
        pair_groups: None,
        ld_compression: Vec::new(),
    }];
    let active_rows = (0..n_rows).collect::<Vec<_>>();
    let universe = Max6ScreenPreparedUniverse {
        pair_ownership: Max6ScreenPairOwnership::build(
            units.as_slice(),
            active_rows.as_slice(),
            logic_bits.sites.len(),
        )
        .unwrap(),
        original_unit_indices: vec![0],
        units,
        active_rows,
        window_pair_count: 3,
    };
    let null_y = (0..n_samples)
        .map(|sample| ((sample * 17) % 31) as f64 - 15.0)
        .collect::<Vec<_>>();

    let (scores, argmax_pairs, unique_pairs) =
        scan_max6_screen_null_batch(&logic_bits, &universe, &[null_y], None, true).unwrap();

    assert_eq!(unique_pairs, 3);
    assert_eq!(argmax_pairs.unwrap(), vec![Some((0, 1))]);
    assert!(scores[0].is_finite());
}

#[test]
fn max6_tail_scan_matches_exact_pair_order_and_sampled_full_ranks() {
    let n_rows = 4usize;
    let n_samples = 96usize;
    let mut logic_bits = build_test_logic_bits(n_rows, n_samples);
    logic_bits.bits_flat.fill(0);
    for row in 0..n_rows {
        for sample in 0..n_samples {
            if (sample * 17 + row * 11 + sample * row * 3) % 29 < 9 + row {
                logic_bits.bits_flat[row * logic_bits.row_words + (sample >> 6)] |=
                    1u64 << (sample & 63);
            }
        }
    }
    let units = vec![Max6GrmMax6LdPreparedUnit {
        selected_global_rows: (0..n_rows).collect(),
        pair_groups: None,
        ld_compression: Vec::new(),
    }];
    let active_rows = (0..n_rows).collect::<Vec<_>>();
    let universe = Max6ScreenPreparedUniverse {
        pair_ownership: Max6ScreenPairOwnership::build(
            units.as_slice(),
            active_rows.as_slice(),
            logic_bits.sites.len(),
        )
        .unwrap(),
        original_unit_indices: vec![0],
        units,
        active_rows,
        window_pair_count: n_rows * (n_rows - 1) / 2,
    };
    let null_y = (0..n_samples)
        .map(|sample| ((sample * 13 + 7) % 37) as f64 - 18.0)
        .collect::<Vec<_>>();
    let lookup = PackedYSumLookup::build(&null_y, n_samples).unwrap();
    let margins = OneCellMarginTable::build_for_rows(
        logic_bits.bits(),
        logic_bits.row_words,
        universe.active_rows.as_slice(),
        &null_y,
        n_samples,
        &lookup,
    )
    .unwrap();
    let draw = logic_residual_screen::Max6ScreenNullDraw {
        residual: &null_y,
        lookup: &lookup,
        margins: &margins,
    };
    let mut reference = Vec::new();
    let mut sum11 = vec![0.0];
    let mut pair_scores = vec![None];
    for first in 0..n_rows - 1 {
        for second in first + 1..n_rows {
            let mut score_maxima = vec![f64::NEG_INFINITY];
            logic_residual_screen::update_max6_one_cell_pair_batch(
                &logic_bits.bits()[first * logic_bits.row_words
                    ..first * logic_bits.row_words + logic_bits.row_words],
                &logic_bits.bits()[second * logic_bits.row_words
                    ..second * logic_bits.row_words + logic_bits.row_words],
                n_samples,
                universe.pair_ownership.active_slot(first),
                universe.pair_ownership.active_slot(second),
                None,
                std::slice::from_ref(&draw),
                sum11.as_mut_slice(),
                score_maxima.as_mut_slice(),
                None,
                Some(pair_scores.as_mut_slice()),
            )
            .unwrap();
            let score = pair_scores[0].unwrap();
            reference.push(sampled_screen::TailPairScore {
                pair: (first, second),
                score: score.score,
                best_channel: score.gate,
            });
        }
    }
    reference.sort_by(sampled_screen::full_rank_cmp);
    let sample_indices = [0usize, 2, 4, 5, 1, 2, 3, 0, 4, 5, 1, 3, 2];
    let sampled_winners = sample_indices
        .map(|index| vec![Some(reference[index])])
        .to_vec();
    let (serial_top, serial_ranks, count) = scan_max6_screen_tail_batch(
        &logic_bits,
        &universe,
        std::slice::from_ref(&null_y),
        sampled_winners.as_slice(),
        None,
    )
    .unwrap();
    assert_eq!(count, reference.len());
    assert_eq!(serial_top[0].len(), reference.len());
    for (observed, expected) in serial_top[0].iter().zip(reference.iter()) {
        assert_eq!(observed.pair, expected.pair);
        assert_eq!(observed.score.to_bits(), expected.score.to_bits());
        assert_eq!(observed.best_channel, expected.best_channel);
    }
    assert_eq!(
        serial_ranks[0],
        sample_indices
            .iter()
            .map(|rank_index| Some(rank_index + 1))
            .collect::<Vec<_>>()
    );

    let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    let (parallel_top, parallel_ranks, parallel_count) = scan_max6_screen_tail_batch(
        &logic_bits,
        &universe,
        std::slice::from_ref(&null_y),
        sampled_winners.as_slice(),
        Some(&pool),
    )
    .unwrap();
    assert_eq!(parallel_count, count);
    assert_eq!(parallel_ranks, serial_ranks);
    assert_eq!(parallel_top[0].len(), serial_top[0].len());
    for (observed, expected) in parallel_top[0].iter().zip(serial_top[0].iter()) {
        assert_eq!(observed.pair, expected.pair);
        assert_eq!(observed.score.to_bits(), expected.score.to_bits());
        assert_eq!(observed.best_channel, expected.best_channel);
    }
}

#[test]
fn max6_overlapping_units_use_unique_pairs_and_duplicate_scores_match_exactly() {
    let n_rows = 5usize;
    let n_samples = 137usize;
    let mut logic_bits = build_test_logic_bits(n_rows, n_samples);
    logic_bits.bits_flat.fill(0);
    for row in 0..n_rows {
        for sample in 0..n_samples {
            if (sample * 17 + row * 11 + sample * row * 3) % 29 < 9 {
                logic_bits.bits_flat[row * logic_bits.row_words + (sample >> 6)] |=
                    1u64 << (sample & 63);
            }
        }
    }
    let units = vec![
        Max6GrmMax6LdPreparedUnit {
            selected_global_rows: vec![0, 1, 2, 3],
            pair_groups: None,
            ld_compression: Vec::new(),
        },
        Max6GrmMax6LdPreparedUnit {
            selected_global_rows: vec![1, 2, 3, 4],
            pair_groups: None,
            ld_compression: Vec::new(),
        },
    ];
    let active_rows = (0..n_rows).collect::<Vec<_>>();
    let ownership = Max6ScreenPairOwnership::build(
        units.as_slice(),
        active_rows.as_slice(),
        logic_bits.sites.len(),
    )
    .unwrap();
    let slots = |global| ownership.active_slot(global);
    assert_eq!(ownership.pair_owner(slots(1), slots(2)), Some(0));
    assert_eq!(ownership.pair_owner(slots(1), slots(3)), Some(0));
    assert_eq!(ownership.pair_owner(slots(2), slots(3)), Some(0));

    let universe = Max6ScreenPreparedUniverse {
        original_unit_indices: vec![0, 1],
        units,
        active_rows,
        pair_ownership: ownership,
        window_pair_count: 12,
    };
    assert_eq!(count_max6_screen_pairs(&universe).unwrap(), 9);
    let null_y = (0..3usize)
        .map(|replicate| {
            (0..n_samples)
                .map(|sample| ((sample * 19 + replicate * 7) % 43) as f64 - 21.0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (batch_maxima, _, unique_pair_count) =
        scan_max6_screen_null_batch(&logic_bits, &universe, null_y.as_slice(), None, false)
            .unwrap();
    assert_eq!(unique_pair_count, 9);
    let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    let (parallel_maxima, _, parallel_unique_count) = scan_max6_screen_null_batch(
        &logic_bits,
        &universe,
        null_y.as_slice(),
        Some(&pool),
        false,
    )
    .unwrap();
    assert_eq!(parallel_unique_count, unique_pair_count);
    assert_eq!(parallel_maxima, batch_maxima);

    let mut reference_scores = HashMap::<(usize, usize), u64>::new();
    let first_y = null_y[0].as_slice();
    let lookup = PackedYSumLookup::build(first_y, n_samples).unwrap();
    let margins = OneCellMarginTable::build_for_rows(
        logic_bits.bits(),
        logic_bits.row_words,
        universe.active_rows.as_slice(),
        first_y,
        n_samples,
        &lookup,
    )
    .unwrap();
    for unit in universe.units.iter() {
        let packed = gather_rows_by_indices(
            logic_bits.bits(),
            logic_bits.row_words,
            unit.selected_global_rows.as_slice(),
            "max6 duplicate score audit",
        )
        .unwrap();
        let local_margins = margins
            .select(unit.selected_global_rows.as_slice())
            .unwrap();
        let local = logic_residual_screen::scan_residual_pairs_packed_with_mode_and_kernel_with_context_and_groups(
                packed.as_slice(),
                None,
                logic_bits.row_words,
                unit.selected_global_rows.len(),
                first_y,
                n_samples,
                16,
                true,
                ResidualProposalMode::Max6,
                ResidualScreenKernel::OneCell,
                &lookup,
                Some(local_margins),
                unit.pair_groups.as_deref(),
                false,
            )
            .unwrap();
        for candidate in local.candidates {
            let pair = (
                unit.selected_global_rows[candidate.first],
                unit.selected_global_rows[candidate.second],
            );
            if let Some(previous) =
                reference_scores.insert(pair, candidate.proposal_score.to_bits())
            {
                assert_eq!(
                    previous,
                    candidate.proposal_score.to_bits(),
                    "duplicate pair {pair:?}"
                );
            }
        }
    }
    assert_eq!(reference_scores.len(), unique_pair_count);
    assert_eq!(reference_scores.len() - 3, 6);

    for (replicate, y) in null_y.iter().enumerate() {
        let scalar_maximum = scan_max6_screen_null_replicate_maximum(
            &logic_bits,
            &universe,
            y.as_slice(),
            replicate,
            None,
        )
        .unwrap();
        assert_eq!(batch_maxima[replicate].to_bits(), scalar_maximum.to_bits());
    }
}

#[test]
fn max6_unique_pair_owner_skips_ineligible_earlier_group_membership() {
    let units = vec![
        Max6GrmMax6LdPreparedUnit {
            selected_global_rows: vec![0, 1, 2],
            pair_groups: Some(vec![0, 0, 1]),
            ld_compression: Vec::new(),
        },
        Max6GrmMax6LdPreparedUnit {
            selected_global_rows: vec![0, 1, 2, 3],
            pair_groups: Some(vec![0, 1, 1, 0]),
            ld_compression: Vec::new(),
        },
    ];
    let ownership = Max6ScreenPairOwnership::build(units.as_slice(), &[0, 1, 2, 3], 4).unwrap();
    assert_eq!(
        ownership.pair_owner(ownership.active_slot(0), ownership.active_slot(1)),
        Some(1)
    );
    assert_eq!(
        ownership.pair_owner(ownership.active_slot(0), ownership.active_slot(2)),
        Some(0)
    );
    assert_eq!(
        ownership.pair_owner(ownership.active_slot(1), ownership.active_slot(2)),
        Some(0)
    );
    let universe = Max6ScreenPreparedUniverse {
        original_unit_indices: vec![0, 1],
        units,
        active_rows: vec![0, 1, 2, 3],
        pair_ownership: ownership,
        window_pair_count: 6,
    };
    assert_eq!(count_max6_screen_pairs(&universe).unwrap(), 5);
}

#[test]
fn max6_final_output_subset_audit_rejects_pairs_absent_from_screen() {
    assert_eq!(
        audit_max6_final_pairs_are_screen_subset(&[(0, 1), (0, 2), (1, 2)], &[(0, 2)]),
        Ok(1)
    );
    assert!(audit_max6_final_pairs_are_screen_subset(&[(0, 1)], &[(0, 2)]).is_err());
    assert!(audit_max6_final_pairs_are_screen_subset(&[(1, 0)], &[(1, 0)]).is_err());
    assert!(audit_max6_final_pairs_are_screen_subset(&[(0, 1), (0, 1)], &[]).is_err());
    assert!(audit_max6_final_pairs_are_screen_subset(&[(0, 1)], &[(0, 1), (0, 1)]).is_err());
}

#[test]
fn max6_screen_maxt_calibration_is_not_reported_as_permutation() {
    assert!(!legacy_max6_permutation_active(true, true, 100));
    assert!(legacy_max6_permutation_active(false, true, 100));
    assert!(!legacy_max6_permutation_active(false, true, 0));
    assert!(!legacy_max6_permutation_active(false, false, 100));
}

#[test]
fn max6_permutation_timing_accumulates_parallel_job_components() {
    let mut total = Max6PermutationTiming::default();
    total.accumulate_job(Max6PermutationJobTiming {
        projected_y_seconds: 0.5,
        endpoint_prepare_seconds: 1.0,
        rerank: super::grm_max6::CandidateRerankTiming {
            pcx_u_seconds: 2.0,
            py_seconds: 0.0,
            candidate_geometry_seconds: 3.0,
            q_seconds: 4.0,
            score_seconds: 5.0,
            topk_seconds: 6.0,
            kernel_seconds: 24.0,
            ..Default::default()
        },
        pair_scores: 7,
    });
    total.accumulate_job(Max6PermutationJobTiming {
        projected_y_seconds: 1.5,
        endpoint_prepare_seconds: 2.0,
        rerank: super::grm_max6::CandidateRerankTiming {
            pcx_u_seconds: 3.0,
            py_seconds: 0.0,
            candidate_geometry_seconds: 4.0,
            q_seconds: 5.0,
            score_seconds: 6.0,
            topk_seconds: 7.0,
            kernel_seconds: 30.0,
            ..Default::default()
        },
        pair_scores: 11,
    });
    assert_eq!(total.projected_y_cpu_seconds, 2.0);
    assert_eq!(total.endpoint_prepare_cpu_seconds, 3.0);
    assert_eq!(total.rerank_pcx_cpu_seconds, 5.0);
    assert_eq!(total.rerank_geometry_cpu_seconds, 7.0);
    assert_eq!(total.rerank_q_cpu_seconds, 9.0);
    assert_eq!(total.rerank_score_cpu_seconds, 11.0);
    assert_eq!(total.rerank_topk_cpu_seconds, 13.0);
    assert_eq!(total.rerank_other_cpu_seconds, 9.0);
    assert_eq!(total.pair_score_count, 18);
}

#[test]
fn max6_observed_timing_accumulates_window_and_rerank_stages() {
    let mut total = Max6ObservedScanTiming::default();
    total.accumulate_unit(Max6ObservedUnitTiming {
        screen_input_prepare_seconds: 0.25,
        screen_kernel_seconds: 1.0,
        candidate_prepare_seconds: 0.15,
        rerank: super::grm_max6::CandidateRerankTiming {
            endpoint_prepare_seconds: 0.5,
            pcx_u_seconds: 2.0,
            candidate_geometry_seconds: 3.0,
            geometry_marker_moments_seconds: 0.7,
            geometry_pairwise_seconds: 1.1,
            geometry_lookup_seconds: 0.5,
            geometry_interaction_py_seconds: 0.7,
            q_seconds: 4.0,
            q_validation_seconds: 0.4,
            q_bitset_seconds: 0.1,
            q_cell_seconds: 0.2,
            q_enumerate_seconds: 0.3,
            q_quadratic_seconds: 2.6,
            q_reconstruct_seconds: 0.4,
            score_seconds: 5.0,
            topk_seconds: 6.0,
            kernel_seconds: 27.0,
            ..Default::default()
        },
        output_finalize_seconds: 0.35,
        screen_candidates: 10_000,
        exact_candidates: 9_998,
        ..Default::default()
    });
    total.accumulate_unit(Max6ObservedUnitTiming {
        screen_input_prepare_seconds: 0.75,
        screen_kernel_seconds: 2.0,
        candidate_prepare_seconds: 0.25,
        rerank: super::grm_max6::CandidateRerankTiming {
            endpoint_prepare_seconds: 1.0,
            pcx_u_seconds: 3.0,
            candidate_geometry_seconds: 4.0,
            geometry_marker_moments_seconds: 1.0,
            geometry_pairwise_seconds: 1.4,
            geometry_lookup_seconds: 0.6,
            geometry_interaction_py_seconds: 0.8,
            q_seconds: 5.0,
            q_validation_seconds: 0.5,
            q_bitset_seconds: 0.2,
            q_cell_seconds: 0.3,
            q_enumerate_seconds: 0.4,
            q_quadratic_seconds: 3.1,
            q_reconstruct_seconds: 0.5,
            score_seconds: 6.0,
            topk_seconds: 7.0,
            kernel_seconds: 32.0,
            ..Default::default()
        },
        output_finalize_seconds: 0.65,
        screen_candidates: 20_000,
        exact_candidates: 19_997,
        ..Default::default()
    });

    assert_eq!(total.units_timed, 2);
    assert_eq!(total.screen_input_prepare_task_seconds, 1.0);
    assert_eq!(total.screen_kernel_task_seconds, 3.0);
    assert_eq!(total.candidate_prepare_task_seconds, 0.4);
    assert_eq!(total.rerank_endpoint_prepare_task_seconds, 1.5);
    assert_eq!(total.rerank_pcx_task_seconds, 5.0);
    assert_eq!(total.rerank_geometry_task_seconds, 7.0);
    assert_eq!(total.rerank_geometry_marker_moments_task_seconds, 1.7);
    assert_eq!(total.rerank_geometry_pairwise_task_seconds, 2.5);
    assert_eq!(total.rerank_geometry_lookup_task_seconds, 1.1);
    assert_eq!(total.rerank_geometry_interaction_py_task_seconds, 1.5);
    assert_eq!(total.rerank_q_task_seconds, 9.0);
    assert_eq!(total.rerank_q_validation_task_seconds, 0.9);
    assert_eq!(total.rerank_q_quadratic_task_seconds, 5.7);
    assert_eq!(total.rerank_score_task_seconds, 11.0);
    assert_eq!(total.rerank_topk_task_seconds, 13.0);
    assert_eq!(total.rerank_other_task_seconds, 12.5);
    assert_eq!(total.output_finalize_task_seconds, 1.0);
    assert_eq!(total.rerank_kernel_task_seconds, 59.0);
    assert_eq!(total.screen_candidate_pairs, 30_000);
    assert_eq!(total.exact_candidate_pairs, 29_995);
}

#[test]
fn max6_permutation_endpoint_matrix_preserves_pair_encoding() {
    let n_samples = 8usize;
    let row_words = words_for_samples(n_samples);
    let bits = vec![
        0b0000_1111u64,
        0b0011_0011u64,
        0b0101_0101u64,
        0b1111_0000u64,
    ];
    let selected_global_rows = vec![0usize, 1, 2, 3];
    let candidate_pairs = vec![(0usize, 2usize), (1usize, 3usize)];

    let (x, pair_i, pair_j, endpoint_rows) = dense_candidate_endpoint_matrix(
        &bits,
        row_words,
        n_samples,
        &selected_global_rows,
        &candidate_pairs,
    )
    .expect("endpoint matrix should be constructible");

    assert_eq!(endpoint_rows, vec![0usize, 1, 2, 3]);
    assert_eq!(pair_i, vec![0usize, 1]);
    assert_eq!(pair_j, vec![2usize, 3]);
    assert_eq!(x.len(), n_samples * endpoint_rows.len());
    for sample in 0..n_samples {
        for (endpoint, &global_row) in endpoint_rows.iter().enumerate() {
            let expected = ((bits[global_row] >> sample) & 1) as f64;
            assert_eq!(x[sample * endpoint_rows.len() + endpoint], expected);
        }
    }
}

#[test]
fn max6_permutation_endpoint_rerank_matches_full_window_rerank() {
    let n_samples = 8usize;
    let n_markers = 4usize;
    let row_words = words_for_samples(n_samples);
    let bits = vec![
        0b0000_1111u64,
        0b0011_0011u64,
        0b0101_0101u64,
        0b1111_0000u64,
    ];
    let selected_global_rows = vec![0usize, 1, 2, 3];
    let candidate_pairs = vec![(0usize, 2usize), (1usize, 3usize)];
    let mut full_x = vec![0.0; n_samples * n_markers];
    for sample in 0..n_samples {
        for marker in 0..n_markers {
            full_x[sample * n_markers + marker] = ((bits[marker] >> sample) & 1) as f64;
        }
    }
    let p_c = (0..n_samples * n_samples)
        .map(|index| {
            if index / n_samples == index % n_samples {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let y = (0..n_samples)
        .map(|index| (index as f64 - 3.5) / 2.0)
        .collect::<Vec<_>>();
    let full_first = candidate_pairs
        .iter()
        .map(|&(left, _)| left as i64)
        .collect::<Vec<_>>();
    let full_second = candidate_pairs
        .iter()
        .map(|&(_, right)| right as i64)
        .collect::<Vec<_>>();
    let full = candidate_rerank_exact(
        &full_x,
        n_samples,
        n_markers,
        &p_c,
        &y,
        &full_first,
        &full_second,
        candidate_pairs.len(),
        DosageChannel::FwlProduct,
    )
    .expect("full-window rerank should succeed");

    let (endpoint_x, pair_i, pair_j, endpoint_rows) = dense_candidate_endpoint_matrix(
        &bits,
        row_words,
        n_samples,
        &selected_global_rows,
        &candidate_pairs,
    )
    .expect("endpoint matrix should be constructible");
    let endpoint_first = pair_i.iter().map(|&index| index as i64).collect::<Vec<_>>();
    let endpoint_second = pair_j.iter().map(|&index| index as i64).collect::<Vec<_>>();
    let endpoint = candidate_rerank_exact(
        &endpoint_x,
        n_samples,
        endpoint_rows.len(),
        &p_c,
        &y,
        &endpoint_first,
        &endpoint_second,
        candidate_pairs.len(),
        DosageChannel::FwlProduct,
    )
    .expect("endpoint rerank should succeed");

    assert_eq!(full.pair_i, endpoint.pair_i);
    assert_eq!(full.pair_j, endpoint.pair_j);
    assert_eq!(full.top_order, endpoint.top_order);
    assert_eq!(full.candidate_rank, endpoint.candidate_rank);
    assert_eq!(full.best_max6_channel, endpoint.best_max6_channel);
    let assert_same_score = |left: f64, right: f64| {
        if left.is_finite() && right.is_finite() {
            assert!((left - right).abs() < 1e-12, "{left} != {right}");
        } else {
            assert_eq!(left, right);
        }
    };
    for (left, right) in full
        .max6
        .iter()
        .zip(endpoint.max6.iter())
        .chain(full.dosage.iter().zip(endpoint.dosage.iter()))
    {
        assert_same_score(*left, *right);
    }
    for (left, right) in full.logic.iter().zip(endpoint.logic.iter()) {
        for (left, right) in left.iter().zip(right.iter()) {
            assert_same_score(*left, *right);
        }
    }
}

#[test]
fn conditional_split_support_requires_both_parent_children() {
    // Parent has 100 observations; a proposed literal must split it into
    // two sufficiently populated conditional branches.  A child with 2
    // observations is an extreme-subset artefact and must be rejected.
    assert!(conditional_split_support_passes(100, 40, 100, 20));
    assert!(!conditional_split_support_passes(100, 2, 100, 20));
    assert!(!conditional_split_support_passes(100, 98, 100, 20));
}

#[test]
fn pair_triple_maps_canonical_xor_candidate_to_xor_rule() {
    let candidate = AllPairCandidate {
        first: 2,
        second: 5,
        gate: AllPairGate::Xor,
        polarity: 0,
        raw_score: 1.0,
        support: 10,
    };
    let rule = pair_triple_rule_from_candidate(&candidate, &[0, 1, 2, 3, 4, 5]).unwrap();
    assert_eq!(rule.len(), 2);
    assert_eq!(rule.first.row_index, 2);
    assert!(!rule.first.negated);
    assert_eq!(rule.rest[0].0, BeamBinaryOp::Xor);
    assert_eq!(rule.rest[0].1.row_index, 5);
    assert!(!rule.rest[0].1.negated);
}

#[test]
fn pair_triple_rejects_noncanonical_xor_polarity() {
    let candidate = AllPairCandidate {
        first: 0,
        second: 1,
        gate: AllPairGate::Xor,
        polarity: 1,
        raw_score: 1.0,
        support: 10,
    };
    let error = pair_triple_rule_from_candidate(&candidate, &[0, 1]).unwrap_err();
    assert!(error.contains("non-canonical XOR polarity"));
}

#[test]
fn pair_triple_search_emits_xor_pair_when_enabled() {
    let n_samples = 8usize;
    let row_words = words_for_samples(n_samples);
    let mut bits = vec![0u64; 2 * row_words];
    bits[0] = 0b0000_1111;
    bits[row_words] = 0b0011_0011;
    let y = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let prepared = GarfieldUnitBitMatrices {
        train_bits: Cow::Owned(bits),
        train_bits_hi: None,
        row_words_train: row_words,
        test_bits: None,
        test_bits_hi: None,
        row_words_test: row_words,
        selected_bits_full: None,
        selected_bits_full_hi: None,
        selected_bits_full_alias_train: false,
        selected_bits_full_hi_alias_train: false,
    };
    let mut params = BeamSearchParams::default();
    params.search_backend = GarfieldSearchBackend::PairTriple;
    params.max_pick = 2;
    params.beam_width = 8;
    params.allow_parallel = false;
    params.maf_threshold = 0.0;
    params.xor_search_enabled = true;
    let candidates =
        pair_triple_search_train_test_continuous(&y, &prepared, 2, &y, &[0, 1], params, None)
            .unwrap();
    assert!(candidates.iter().any(|candidate| {
        candidate
            .rule
            .rest
            .first()
            .map(|(op, _)| *op == BeamBinaryOp::Xor)
            .unwrap_or(false)
    }));
}

#[test]
fn pair_triple_rejects_xor_order_three_until_gate_is_supported() {
    let prepared = GarfieldUnitBitMatrices {
        train_bits: Cow::Owned(vec![0b0000_1111, 0b0011_0011, 0b0101_0101]),
        train_bits_hi: None,
        row_words_train: 1,
        test_bits: None,
        test_bits_hi: None,
        row_words_test: 1,
        selected_bits_full: None,
        selected_bits_full_hi: None,
        selected_bits_full_alias_train: false,
        selected_bits_full_hi_alias_train: false,
    };
    let mut params = BeamSearchParams::default();
    params.search_backend = GarfieldSearchBackend::PairTriple;
    params.max_pick = 3;
    params.xor_search_enabled = true;
    let error = pair_triple_search_train_test_continuous(
        &[0.0; 8],
        &prepared,
        3,
        &[0.0; 8],
        &[0, 1, 2],
        params,
        None,
    )
    .unwrap_err();
    assert!(error.contains("only for order-2 AllPair"));
}

#[test]
fn packed_corr_scores_match_dense_binary_corr() {
    let n_samples = 137usize;
    let marker_rows = vec![
        (0..n_samples)
            .map(|i| u8::from(i % 3 == 0 || i % 11 == 1))
            .collect::<Vec<_>>(),
        (0..n_samples)
            .map(|i| u8::from(i % 5 < 2))
            .collect::<Vec<_>>(),
        (0..n_samples)
            .map(|i| u8::from(i % 7 == 3 || i >= 96))
            .collect::<Vec<_>>(),
    ];
    let y = (0..n_samples)
        .map(|i| ((i as f64) * 0.03125) - 2.0 + ((i % 9) as f64) * 0.17)
        .collect::<Vec<_>>();
    let row_words = words_for_samples(n_samples);
    let mut bits = vec![0u64; marker_rows.len() * row_words];
    for (row_idx, row) in marker_rows.iter().enumerate() {
        for (sample_idx, &value) in row.iter().enumerate() {
            if value != 0 {
                bits[row_idx * row_words + (sample_idx >> 6)] |= 1u64 << (sample_idx & 63);
            }
        }
    }
    let row_indices = vec![0usize, 1, 2];
    let sample_indices = (0..n_samples).collect::<Vec<_>>();
    let summaries = dosage_stage1_dual_summaries_from_full_bits(
        &bits,
        None,
        row_words,
        &row_indices,
        &sample_indices,
        &y,
        marker_rows.len(),
        n_samples,
        None,
        false,
    )
    .unwrap();
    let packed = corr_scores_from_dual_summaries(
        &summaries,
        y.iter().copied().sum::<f64>(),
        y.iter().map(|value| value * value).sum::<f64>(),
        n_samples,
        false,
    );
    let dense = feature_scores_abs_corr_dosage_x(&marker_rows, &y);
    for (packed_score, dense_score) in packed.iter().zip(dense.iter()) {
        assert!((packed_score - dense_score).abs() < 1e-12);
    }
}

#[test]
fn cached_proposal_decode_matches_uncached_rows() {
    let n_samples = 73usize;
    let row_words = words_for_samples(n_samples);
    let mut bits = vec![0u64; 3 * row_words];
    for row in 0..3usize {
        for sample in 0..n_samples {
            if (sample + row * 3) % (row + 3) == 0 {
                bits[row * row_words + (sample >> 6)] |= 1u64 << (sample & 63);
            }
        }
    }
    let rows = vec![2usize, 0, 2, 1];
    let sample_indices = (0..n_samples).collect::<Vec<_>>();
    let uncached = dense_dosage_rows_from_full_bits(
        &bits,
        None,
        row_words,
        &rows,
        &sample_indices,
        3,
        n_samples,
    )
    .unwrap();
    let cached_first = dense_dosage_rows_from_full_bits_cached(
        &bits,
        None,
        row_words,
        &rows,
        &sample_indices,
        3,
        n_samples,
    )
    .unwrap();
    let cached_second = dense_dosage_rows_from_full_bits_cached(
        &bits,
        None,
        row_words,
        &rows,
        &sample_indices,
        3,
        n_samples,
    )
    .unwrap();
    assert_eq!(cached_first, uncached);
    assert_eq!(cached_second, uncached);
}

#[test]
fn proposal_search_config_off_preserves_legacy_parameters() {
    let params = BeamSearchParams::default();
    let configured = apply_proposal_search_config(params.clone(), ProposalConfig::default());
    assert_eq!(configured, params);
}

#[test]
fn proposal_search_config_enables_bounded_pair_refinement() {
    let params = BeamSearchParams::default();
    let configured = apply_proposal_search_config(
        params,
        ProposalConfig {
            mode: ProposalMode::CorrRf,
            k_site: 64,
            k2: 17,
            max_order: 3,
        },
    );
    assert_eq!(configured.beam_width, 17);
    assert_eq!(configured.exhaustive_depth, 3);
    assert_eq!(configured.rank_mode, BeamRankMode::ExhaustiveThenGain);
    assert_eq!(configured.max_pick, 3);
}

#[test]
fn best_parent_delta_gate_accepts_only_delta_above_null_max_t() {
    assert!(best_parent_delta_passes(12.0, 7.0, 4.0, 3));
    assert!(!best_parent_delta_passes(10.0, 7.0, 4.0, 3));
    assert!(best_parent_delta_passes(10.0, 7.0, 2.5, 2));
}

#[test]
fn four_plus_family_gate_rejects_unvalidated_high_order_rules() {
    assert!(four_plus_family_max_t_passes(4, 12.0, 10.0));
    assert!(four_plus_family_max_t_passes(3, -100.0, 10.0));
    assert!(!four_plus_family_max_t_passes(4, 9.99, 10.0));
    assert!(!four_plus_family_max_t_passes(5, f64::NAN, 10.0));
}

#[test]
fn logic_bit_storage_switches_to_mmap_only_over_budget() {
    assert!(!should_use_mapped_logic_bits(1024, 0));
    assert!(!should_use_mapped_logic_bits(1024, 1024));
    assert!(should_use_mapped_logic_bits(1025, 1024));
}

#[test]
fn active_logic_mode_rejects_legacy_mbin_alias() {
    assert!(parse_bin_mode("mbin").is_err());
    assert!(parse_bin_mode("bin").is_ok());
}

#[test]
fn test_scheduler_profile_summary_reports_percentiles_and_work() {
    let samples = vec![
        GarfieldSchedulerTaskSample {
            elapsed_s: 1.0,
            work_units: 10,
        },
        GarfieldSchedulerTaskSample {
            elapsed_s: 2.0,
            work_units: 20,
        },
        GarfieldSchedulerTaskSample {
            elapsed_s: 3.0,
            work_units: 30,
        },
        GarfieldSchedulerTaskSample {
            elapsed_s: 4.0,
            work_units: 40,
        },
    ];
    let summary = summarize_scheduler_task_samples(&samples);
    assert_eq!(summary.task_count, 4);
    assert_eq!(summary.total_work_units, 100);
    assert_eq!(summary.min_elapsed_s, 1.0);
    assert_eq!(summary.max_elapsed_s, 4.0);
    assert_eq!(summary.p50_elapsed_s, 2.0);
    assert_eq!(summary.p95_elapsed_s, 4.0);
}

#[test]
fn test_scheduler_profile_summary_handles_empty_samples() {
    let summary = summarize_scheduler_task_samples(&[]);
    assert_eq!(summary.task_count, 0);
    assert_eq!(summary.total_work_units, 0);
    assert_eq!(summary.min_elapsed_s, 0.0);
    assert_eq!(summary.p50_elapsed_s, 0.0);
    assert_eq!(summary.p95_elapsed_s, 0.0);
    assert_eq!(summary.max_elapsed_s, 0.0);
}

#[test]
fn test_scan_scheduler_preserves_order_and_balances_work() {
    let units = (1usize..=8)
        .map(|n| GarfieldLogicUnit {
            label: format!("u{n}"),
            indices: vec![0usize; n],
            spans: Vec::new(),
        })
        .collect::<Vec<_>>();
    let indices = (0usize..units.len()).collect::<Vec<_>>();
    let chunks = garfield_cost_balanced_unit_chunks(&indices, &units, None, 2, 8);
    let flattened = chunks.iter().flatten().copied().collect::<Vec<_>>();
    assert_eq!(flattened, indices);
    assert!(chunks.len() >= 2);
    let work = chunks
        .iter()
        .map(|chunk| {
            chunk
                .iter()
                .map(|&ui| units[ui].indices.len())
                .sum::<usize>()
        })
        .collect::<Vec<_>>();
    let max_work = *work.iter().max().unwrap();
    let min_work = *work.iter().min().unwrap();
    assert!(max_work.saturating_sub(min_work) <= 8);
}

fn make_temp_dir(prefix: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("janusx_garfield_{prefix}_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn make_row_bits(n_samples: usize, ones: &[usize]) -> Vec<u8> {
    let mut bits = vec![0u8; row_bytes_for_samples(n_samples)];
    for &idx in ones.iter() {
        bits[idx >> 3] |= 1u8 << (idx & 7);
    }
    bits
}

fn build_test_bits(n_rows: usize, n_samples: usize) -> (Vec<u64>, usize, Vec<usize>) {
    let row_words = words_for_samples(n_samples);
    let mut bits_flat = vec![0u64; n_rows * row_words];
    for row in 0..n_rows {
        for word in 0..row_words {
            let a = (row as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let b = (word as u64 + 3).wrapping_mul(0xD1B5_4A32_D192_ED03);
            let rot = ((row * 11 + word * 7) % 63 + 1) as u32;
            let dense = (!0u64).rotate_right(((row * 5 + word * 13) % 64) as u32);
            bits_flat[row * row_words + word] = (a ^ b).rotate_left(rot) | dense;
        }
    }
    let group_ids = (0..n_rows).map(|i| i % 96).collect::<Vec<_>>();
    (bits_flat, row_words, group_ids)
}

fn build_test_logic_bits(n_rows: usize, n_samples: usize) -> GarfieldLogicBits {
    let (bits_flat, row_words, group_ids) = build_test_bits(n_rows, n_samples);
    GarfieldLogicBits {
        bits_flat,
        bits_mmap: None,
        bits_mmap_words: 0,
        bits_mmap_owner: None,
        bits_hi_flat: None,
        row_words,
        sample_ids: (0..n_samples).map(|i| format!("s{i}")).collect(),
        sites: (0..n_rows)
            .map(|i| {
                GarfieldLogicSite::owned(
                    "1",
                    i as i32 + 1,
                    format!("snp{i}"),
                    "A",
                    "T",
                    GarfieldLogicSiteMode::Bin,
                )
            })
            .collect(),
        group_ids,
        n_samples,
    }
}

#[test]
fn test_ld_support_cache_matches_uncached_priority_clumping() {
    Python::initialize();
    let n_rows = 96usize;
    let n_samples = 130usize;
    let row_words = words_for_samples(n_samples);
    let mut bits_flat = vec![0u64; n_rows * row_words];
    for row in 0..n_rows {
        for sample in 0..n_samples {
            let hit = ((sample * 17 + row * 11 + (row / 7) * 3) % 29) < 9;
            if hit {
                bits_flat[row * row_words + (sample >> 6)] |= 1u64 << (sample & 63);
            }
        }
    }
    let group_ids = (0..n_rows).collect::<Vec<_>>();
    let logic_bits = GarfieldLogicBits {
        bits_flat,
        bits_mmap: None,
        bits_mmap_words: 0,
        bits_mmap_owner: None,
        bits_hi_flat: None,
        row_words,
        sample_ids: (0..n_samples).map(|i| format!("s{i}")).collect(),
        sites: (0..n_rows)
            .map(|i| {
                GarfieldLogicSite::owned(
                    "1",
                    i as i32 + 1,
                    format!("snp{i}"),
                    "A",
                    "T",
                    GarfieldLogicSiteMode::Bin,
                )
            })
            .collect(),
        group_ids,
        n_samples,
    };
    let sample_indices = (0..n_samples).collect::<Vec<_>>();
    let candidate_global_rows = (0..n_rows).collect::<Vec<_>>();
    let priority_local_rows = (0..n_rows).rev().collect::<Vec<_>>();
    let cache = build_garfield_ld_support_cache(
        &logic_bits,
        candidate_global_rows.as_slice(),
        sample_indices.as_slice(),
    )
    .unwrap();

    let uncached = prune_candidate_rows_by_ld_priority(
        candidate_global_rows.as_slice(),
        priority_local_rows.as_slice(),
        &logic_bits,
        sample_indices.as_slice(),
    )
    .unwrap();
    let cached = prune_candidate_rows_by_ld_priority_with_cache(
        candidate_global_rows.as_slice(),
        priority_local_rows.as_slice(),
        &logic_bits,
        sample_indices.as_slice(),
        Some(&cache),
    )
    .unwrap();
    assert_eq!(uncached, cached);
    assert!(!cached.is_empty());

    let limited = prune_candidate_rows_by_ld_priority_with_cache_limit_trace(
        candidate_global_rows.as_slice(),
        priority_local_rows.as_slice(),
        &logic_bits,
        sample_indices.as_slice(),
        Some(&cache),
        Some(3),
    )
    .unwrap();
    assert_eq!(
        limited.selected_global_rows,
        cached.iter().take(3).copied().collect::<Vec<_>>()
    );

    let sparse_rows = (3..n_rows).step_by(3).collect::<Vec<_>>();
    let sparse_priority = sparse_rows.iter().copied().rev().collect::<Vec<_>>();
    let sparse_cache = build_garfield_ld_support_cache(
        &logic_bits,
        sparse_rows.as_slice(),
        sample_indices.as_slice(),
    )
    .unwrap();
    let sparse_uncached = prune_candidate_rows_by_ld_priority(
        sparse_rows.as_slice(),
        sparse_priority.as_slice(),
        &logic_bits,
        sample_indices.as_slice(),
    )
    .unwrap();
    let sparse_cached = prune_candidate_rows_by_ld_priority_with_cache(
        sparse_rows.as_slice(),
        sparse_priority.as_slice(),
        &logic_bits,
        sample_indices.as_slice(),
        Some(&sparse_cache),
    )
    .unwrap();
    assert_eq!(sparse_uncached, sparse_cached);
}

#[test]
fn test_ld_support_cache_same_key_is_built_once_under_concurrency() {
    let registry = Arc::new(Mutex::new(HashMap::new()));
    let build_count = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(std::sync::Barrier::new(8));
    let key = (18_731usize, 0.8f64.to_bits());
    let workers = (0..8)
        .map(|_| {
            let registry = Arc::clone(&registry);
            let build_count = Arc::clone(&build_count);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                cached_ld_support_entry(registry.as_ref(), key, || {
                    build_count.fetch_add(1, Ordering::Relaxed);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    GarfieldLdSupportCacheEntry {
                        conflict_supports: vec![vec![1, 2]],
                        exact_bounds: vec![vec![GarfieldLdExactBounds::default()]],
                    }
                })
            })
        })
        .collect::<Vec<_>>();
    let entries = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect::<Vec<_>>();

    assert_eq!(build_count.load(Ordering::Relaxed), 1);
    assert!(entries
        .windows(2)
        .all(|pair| Arc::ptr_eq(&pair[0], &pair[1])));
}

#[test]
fn max6_ld_selection_fingerprint_tracks_unit_rows_and_groups() {
    let mut selected = Max6GrmMax6LdPreparedUnit {
        selected_global_rows: vec![2, 5, 9],
        pair_groups: Some(vec![0, 1, 1]),
        ld_compression: Vec::new(),
    };
    let unit_ids = vec![17];
    let baseline = max6_ld_selection_fingerprint(std::slice::from_ref(&selected), &unit_ids);
    assert_eq!(
        baseline,
        max6_ld_selection_fingerprint(std::slice::from_ref(&selected), &unit_ids)
    );

    selected.selected_global_rows[1] = 6;
    assert_ne!(
        baseline,
        max6_ld_selection_fingerprint(std::slice::from_ref(&selected), &unit_ids)
    );
    selected.selected_global_rows[1] = 5;
    selected.pair_groups = Some(vec![0, 1, 2]);
    assert_ne!(
        baseline,
        max6_ld_selection_fingerprint(std::slice::from_ref(&selected), &unit_ids)
    );
}

#[test]
fn max6_ld_tag_selection_matches_with_precomputed_row_support_cache() {
    let logic_bits = build_test_logic_bits(8, 130);
    let rows = (0..logic_bits.sites.len()).collect::<Vec<_>>();
    let samples = (0..logic_bits.n_samples).collect::<Vec<_>>();
    let cache = build_garfield_ld_support_cache(&logic_bits, &rows, &samples).unwrap();
    let uncached =
        select_max6_ld_tag_rows_impl(&rows, &logic_bits, &samples, None, 0.95, false, None)
            .unwrap();
    let cached = select_max6_ld_tag_rows_impl(
        &rows,
        &logic_bits,
        &samples,
        None,
        0.95,
        false,
        Some(&cache),
    )
    .unwrap();

    assert_eq!(uncached.selected_global_rows, cached.selected_global_rows);
}

#[test]
fn test_materialize_prepared_bit_matrices_borrows_train_bits_for_full_identity_range() {
    let logic_bits = build_test_logic_bits(6, 130);
    let prepared = GarfieldUnitPrepared {
        selected_global_rows: vec![1, 2, 3],
        stage1_candidate_global_rows: None,
        ld_compression: Vec::new(),
        local_groups: vec![0, 1, 2],
        geneset_stage_group_target: None,
        null_unit_group_bin: 0,
        train_literal_scores: None,
    };
    let train_idx_local = (0..logic_bits.n_samples).collect::<Vec<_>>();
    let prepared_bits = materialize_prepared_bit_matrices(
        &prepared,
        &logic_bits,
        train_idx_local.as_slice(),
        train_idx_local.as_slice(),
        true,
    )
    .unwrap();
    assert!(matches!(prepared_bits.train_bits, Cow::Borrowed(_)));
    assert!(prepared_bits.test_bits.is_none());
    assert!(prepared_bits.selected_bits_full_alias_train);
    let expected = &logic_bits.bits()[logic_bits.row_words..logic_bits.row_words.saturating_mul(4)];
    assert_eq!(prepared_bits.train_bits(), expected);
    assert_eq!(prepared_bits.selected_bits_full().unwrap(), expected);
}

#[test]
fn test_materialize_prepared_bit_matrices_keeps_full_bits_borrowed_for_subset_samples() {
    let logic_bits = build_test_logic_bits(6, 130);
    let prepared = GarfieldUnitPrepared {
        selected_global_rows: vec![2, 3, 4],
        stage1_candidate_global_rows: None,
        ld_compression: Vec::new(),
        local_groups: vec![0, 1, 2],
        geneset_stage_group_target: None,
        null_unit_group_bin: 0,
        train_literal_scores: None,
    };
    let train_idx_local = (0..96).collect::<Vec<_>>();
    let prepared_bits = materialize_prepared_bit_matrices(
        &prepared,
        &logic_bits,
        train_idx_local.as_slice(),
        train_idx_local.as_slice(),
        true,
    )
    .unwrap();
    assert!(matches!(prepared_bits.train_bits, Cow::Owned(_)));
    assert!(matches!(
        prepared_bits.selected_bits_full,
        Some(Cow::Borrowed(_))
    ));
    let expected = &logic_bits.bits()
        [logic_bits.row_words.saturating_mul(2)..logic_bits.row_words.saturating_mul(5)];
    assert_eq!(prepared_bits.selected_bits_full().unwrap(), expected);
}

fn pack_test_binary_rows(rows: &[Vec<u8>]) -> (Vec<u64>, usize) {
    let n_samples = rows.first().map(|row| row.len()).unwrap_or(0);
    let row_words = words_for_samples(n_samples);
    let mut bits_flat = vec![0u64; rows.len().saturating_mul(row_words)];
    for (row_idx, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), n_samples);
        for (sample_idx, &v) in row.iter().enumerate() {
            if v != 0 {
                bits_flat[row_idx * row_words + (sample_idx >> 6)] |= 1u64 << (sample_idx & 63);
            }
        }
    }
    (bits_flat, row_words)
}

fn test_logic_bits_with_rows(
    bits_flat: Vec<u64>,
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
) -> GarfieldLogicBits {
    GarfieldLogicBits {
        bits_flat,
        bits_mmap: None,
        bits_mmap_words: 0,
        bits_mmap_owner: None,
        bits_hi_flat: None,
        row_words,
        sample_ids: (0..n_samples).map(|i| format!("s{i}")).collect(),
        sites: (0..n_rows)
            .map(|i| {
                GarfieldLogicSite::owned(
                    "1",
                    i32::try_from(i + 1).unwrap(),
                    format!("snp{i}"),
                    "A",
                    "G",
                    GarfieldLogicSiteMode::Bin,
                )
            })
            .collect(),
        group_ids: (0..n_rows).collect(),
        n_samples,
    }
}

#[test]
fn test_group_exclusion_parallel_matches_serial() {
    let n_rows = 1024usize;
    let n_samples = 256usize;
    let max_pick = 3usize;
    let beam_width = 8usize;
    let (bits_flat, row_words, group_ids) = build_test_bits(n_rows, n_samples);
    let score_fn = |combined: &[u64]| combined.iter().map(|w| w.count_ones() as f64).sum::<f64>();

    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("build serial pool")
        .install(|| {
            beam_search_and_with_group_exclusion(
                &bits_flat, row_words, n_rows, n_samples, max_pick, beam_width, &group_ids,
                score_fn,
            )
            .expect("serial constrained beam")
        });

    let parallel = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build parallel pool")
        .install(|| {
            beam_search_and_with_group_exclusion(
                &bits_flat, row_words, n_rows, n_samples, max_pick, beam_width, &group_ids,
                score_fn,
            )
            .expect("parallel constrained beam")
        });

    assert_eq!(serial.selected_indices, parallel.selected_indices);
    assert_eq!(serial.combined_bits, parallel.combined_bits);
    assert_eq!(serial.score, parallel.score);
}

#[test]
fn test_load_bin01_packed_roundtrip() {
    let dir = make_temp_dir("bin01_load");
    let bin_path = dir.join("toy.bin");
    let bin_str = bin_path.to_str().unwrap();
    let n_samples = 10usize;
    let sample_ids = (0..n_samples).map(|i| format!("s{i}")).collect::<Vec<_>>();
    write_sample_id_sidecar(bin_str, &sample_ids).unwrap();
    let mut writer = Bin01Writer::new(bin_str, n_samples, Bin01SiteMode::TextTsv).unwrap();
    write_encoded_rows_bin(
        &mut writer,
        &[
            EncodedRow {
                site: SiteInfo {
                    chrom: "1".to_string(),
                    pos: 10,
                    snp: "bin_row_10".to_string(),
                    ref_allele: "A".to_string(),
                    alt_allele: "T|BIN".to_string(),
                },
                bits: make_row_bits(n_samples, &[1, 3, 8]),
            },
            EncodedRow {
                site: SiteInfo {
                    chrom: "1".to_string(),
                    pos: 20,
                    snp: "bin_row_20".to_string(),
                    ref_allele: "A".to_string(),
                    alt_allele: "C|BIN".to_string(),
                },
                bits: make_row_bits(n_samples, &[0, 4, 9]),
            },
        ],
    )
    .unwrap();
    writer.finish().unwrap();

    let (packed, group_ids, got_n_samples, n_rows, row_bytes) =
        load_bin01_packed_owned(bin_str).unwrap();
    assert_eq!(got_n_samples, n_samples);
    assert_eq!(n_rows, 2);
    assert_eq!(row_bytes, row_bytes_for_samples(n_samples));
    assert_eq!(group_ids, vec![0u64, 1u64]);
    assert_eq!(
        packed,
        [
            make_row_bits(n_samples, &[1, 3, 8]),
            make_row_bits(n_samples, &[0, 4, 9]),
        ]
        .concat()
    );
}

#[test]
fn test_load_bin01_rejects_retired_mbin_feature_rows() {
    let dir = make_temp_dir("mbin_load");
    let bin_path = dir.join("toy.mbin.bin");
    let bin_str = bin_path.to_str().unwrap();
    let n_samples = 9usize;
    let sample_ids = (0..n_samples).map(|i| format!("s{i}")).collect::<Vec<_>>();
    write_sample_id_sidecar(bin_str, &sample_ids).unwrap();
    let mut writer = Bin01Writer::new(bin_str, n_samples, Bin01SiteMode::TextTsv).unwrap();
    write_encoded_rows_bin(
        &mut writer,
        &[
            EncodedRow {
                site: SiteInfo {
                    chrom: "2".to_string(),
                    pos: 100,
                    snp: "mbin_100_dom".to_string(),
                    ref_allele: "A".to_string(),
                    alt_allele: "G|DOM".to_string(),
                },
                bits: make_row_bits(n_samples, &[0, 1, 4, 8]),
            },
            EncodedRow {
                site: SiteInfo {
                    chrom: "2".to_string(),
                    pos: 100,
                    snp: "mbin_100_rec".to_string(),
                    ref_allele: "A".to_string(),
                    alt_allele: "G|REC".to_string(),
                },
                bits: make_row_bits(n_samples, &[1, 8]),
            },
            EncodedRow {
                site: SiteInfo {
                    chrom: "2".to_string(),
                    pos: 100,
                    snp: "mbin_100_het".to_string(),
                    ref_allele: "A".to_string(),
                    alt_allele: "G|HET".to_string(),
                },
                bits: make_row_bits(n_samples, &[0, 4]),
            },
            EncodedRow {
                site: SiteInfo {
                    chrom: "2".to_string(),
                    pos: 150,
                    snp: "mbin_150_dom".to_string(),
                    ref_allele: "C".to_string(),
                    alt_allele: "T|DOM".to_string(),
                },
                bits: make_row_bits(n_samples, &[2, 3, 5]),
            },
            EncodedRow {
                site: SiteInfo {
                    chrom: "2".to_string(),
                    pos: 150,
                    snp: "mbin_150_rec".to_string(),
                    ref_allele: "C".to_string(),
                    alt_allele: "T|REC".to_string(),
                },
                bits: make_row_bits(n_samples, &[5]),
            },
            EncodedRow {
                site: SiteInfo {
                    chrom: "2".to_string(),
                    pos: 150,
                    snp: "mbin_150_het".to_string(),
                    ref_allele: "C".to_string(),
                    alt_allele: "T|HET".to_string(),
                },
                bits: make_row_bits(n_samples, &[2, 3]),
            },
        ],
    )
    .unwrap();
    writer.finish().unwrap();

    let err = load_bin01_packed_owned(bin_str).unwrap_err();
    assert!(err.contains("DOM/REC/HET feature rows are retired"));
}

#[test]
fn test_apply_logic_rule_output_limit_ratio() {
    let mut records = (0..10usize)
        .map(|i| GarfieldLogicRuleRecord {
            unit_name: format!("u{i}"),
            unit_kind: "window".to_string(),
            unit_index: i + 1,
            region_size: 8,
            ml_feature_count: 4,
            ml_rank: ".".to_string(),
            selected_row_indices: vec![i],
            display_ops: Vec::new(),
            display_negated: vec![false],
            snp_name: format!("snp{i}"),
            expr: format!("expr{i}"),
            chrom_field: "1".to_string(),
            bim_snp_name: format!("snp{i}"),
            bim_allele0: "A".to_string(),
            bim_allele1: "T".to_string(),
            allele1_maf: 0.0,
            pos: i as i32,
            score: 20.0 - i as f64,
            delta_score: format!("{}->{}", 20.0 - i as f64, 20.0 - i as f64),
            perm_pvalue: None,
            perm_fdr: None,
            support_bits: None,
        })
        .collect::<Vec<_>>();
    apply_logic_rule_output_limit(&mut records, 0, 0.3).unwrap();
    assert_eq!(records.len(), 3);
}

#[test]
fn test_write_logic_rules_tsv_omits_permutation_columns_by_default() {
    let dir = make_temp_dir("rules_tsv_no_perm");
    let path = dir.join("rules.tsv");
    let path_str = path.to_string_lossy().to_string();
    let records = vec![GarfieldLogicRuleRecord {
        unit_name: "u1".to_string(),
        unit_kind: "window".to_string(),
        unit_index: 1,
        region_size: 8,
        ml_feature_count: 4,
        ml_rank: ".".to_string(),
        selected_row_indices: vec![0],
        display_ops: Vec::new(),
        display_negated: vec![false],
        snp_name: "snp1".to_string(),
        expr: "expr1".to_string(),
        chrom_field: "1".to_string(),
        bim_snp_name: "snp1".to_string(),
        bim_allele0: "A".to_string(),
        bim_allele1: "T".to_string(),
        allele1_maf: 0.0,
        pos: 1,
        score: 10.0,
        delta_score: "10->10".to_string(),
        perm_pvalue: None,
        perm_fdr: None,
        support_bits: None,
    }];
    write_logic_rules_tsv(path_str.as_str(), records.as_slice()).unwrap();
    let header = fs::read_to_string(path).unwrap();
    let first_line = header.lines().next().unwrap();
    assert_eq!(
            first_line,
            "unit_name\tregion_size\tml_feature_count\tMLrank\tsnp_name\tallele1_maf\tscore\tdelta_score"
        );
}

#[test]
fn test_write_logic_rules_tsv_appends_permutation_columns_when_present() {
    let dir = make_temp_dir("rules_tsv_with_perm");
    let path = dir.join("rules.tsv");
    let path_str = path.to_string_lossy().to_string();
    let records = vec![GarfieldLogicRuleRecord {
        unit_name: "u1".to_string(),
        unit_kind: "window".to_string(),
        unit_index: 1,
        region_size: 8,
        ml_feature_count: 4,
        ml_rank: ".".to_string(),
        selected_row_indices: vec![0],
        display_ops: Vec::new(),
        display_negated: vec![false],
        snp_name: "snp1".to_string(),
        expr: "expr1".to_string(),
        chrom_field: "1".to_string(),
        bim_snp_name: "snp1".to_string(),
        bim_allele0: "A".to_string(),
        bim_allele1: "T".to_string(),
        allele1_maf: 0.0,
        pos: 1,
        score: 10.0,
        delta_score: "10->10".to_string(),
        perm_pvalue: Some(1.2e-4),
        perm_fdr: Some(2.5e-4),
        support_bits: None,
    }];
    write_logic_rules_tsv(path_str.as_str(), records.as_slice()).unwrap();
    let content = fs::read_to_string(path).unwrap();
    let mut lines = content.lines();
    assert_eq!(
            lines.next().unwrap(),
            "unit_name\tregion_size\tml_feature_count\tMLrank\tsnp_name\tallele1_maf\tscore\tdelta_score\tperm_pvalue\tperm_fdr"
        );
    assert_eq!(
        lines.next().unwrap(),
        "u1\t8\t4\t.\tsnp1\t0.0000\t10.0000\t10->10\t1.2e-4\t2.5e-4"
    );
}

#[test]
fn test_extend_keep_with_score_ties_keeps_full_tie_block() {
    let scores = vec![10.0_f64, 9.0, 9.0, 8.0];
    let keep = extend_keep_with_score_ties(scores.as_slice(), 2, |v| *v);
    assert_eq!(keep, 3);
}

#[test]
fn test_collect_topk_rule_null_metric_by_group_len_bucket_keeps_each_layer() {
    let max_rule_len = 5usize;
    let keep_topk = 2usize;
    let selected = collect_topk_rule_null_metric_by_group_len_bucket(
        &[
            (
                RuleNullBucket {
                    rule_len: 1,
                    complexity_bin: rule_null_context_bin(0, 0),
                },
                1.0,
            ),
            (
                RuleNullBucket {
                    rule_len: 1,
                    complexity_bin: rule_null_context_bin(0, 1),
                },
                0.8,
            ),
            (
                RuleNullBucket {
                    rule_len: 2,
                    complexity_bin: rule_null_context_bin(0, 0),
                },
                10.0,
            ),
            (
                RuleNullBucket {
                    rule_len: 2,
                    complexity_bin: rule_null_context_bin(0, 1),
                },
                9.0,
            ),
            (
                RuleNullBucket {
                    rule_len: 3,
                    complexity_bin: rule_null_context_bin(0, 0),
                },
                100.0,
            ),
            (
                RuleNullBucket {
                    rule_len: 4,
                    complexity_bin: rule_null_context_bin(0, 1),
                },
                99.0,
            ),
            (
                RuleNullBucket {
                    rule_len: 5,
                    complexity_bin: rule_null_context_bin(0, 2),
                },
                98.0,
            ),
        ],
        max_rule_len,
        keep_topk,
        false,
    );
    assert_eq!(
        selected
            .iter()
            .filter(|(bucket, _)| {
                rule_null_group_len_bucket_index(
                    bucket.unit_group_bin(),
                    bucket.rule_len,
                    max_rule_len,
                    3,
                ) == rule_null_group_len_bucket_index(0, 1, max_rule_len, 3)
            })
            .count(),
        1
    );
    assert_ne!(
        rule_null_group_len_bucket_index(0, 3, max_rule_len, 3),
        rule_null_group_len_bucket_index(0, 4, max_rule_len, 3)
    );
    assert_ne!(
        rule_null_group_len_bucket_index(0, 4, max_rule_len, 3),
        rule_null_group_len_bucket_index(0, 5, max_rule_len, 3)
    );
    assert_eq!(
        selected
            .iter()
            .filter(|(bucket, _)| {
                rule_null_group_len_bucket_index(
                    bucket.unit_group_bin(),
                    bucket.rule_len,
                    max_rule_len,
                    3,
                ) == rule_null_group_len_bucket_index(1, 1, max_rule_len, 3)
            })
            .count(),
        1
    );
    assert_eq!(
        selected
            .iter()
            .filter(|(bucket, _)| {
                rule_null_group_len_bucket_index(
                    bucket.unit_group_bin(),
                    bucket.rule_len,
                    max_rule_len,
                    3,
                ) == rule_null_group_len_bucket_index(0, 2, max_rule_len, 3)
            })
            .count(),
        1
    );
    assert!(selected
        .iter()
        .any(|(bucket, score)| bucket.rule_len == 1 && (*score - 1.0).abs() < 1e-12));
    assert!(selected
        .iter()
        .any(|(bucket, score)| bucket.rule_len == 2 && (*score - 10.0).abs() < 1e-12));
    assert!(selected
        .iter()
        .any(|(bucket, score)| bucket.rule_len == 3 && (*score - 100.0).abs() < 1e-12));
    assert!(selected
        .iter()
        .any(|(bucket, score)| bucket.rule_len == 4 && (*score - 99.0).abs() < 1e-12));
}

#[test]
fn test_collect_global_max_rule_null_metric_by_group_len_bucket_collapses_windows() {
    let vals = [
        GarfieldPermutationNullScores {
            bucket: RuleNullBucket {
                rule_len: 4,
                complexity_bin: 0,
            },
            search_train_score: f64::NAN,
            search_test_score: f64::NAN,
            output_train_raw_score: 12.0,
            output_test_raw_score: f64::NAN,
            delta_train_score: f64::NAN,
            delta_test_score: f64::NAN,
        },
        GarfieldPermutationNullScores {
            bucket: RuleNullBucket {
                rule_len: 4,
                complexity_bin: 0,
            },
            search_train_score: f64::NAN,
            search_test_score: f64::NAN,
            output_train_raw_score: 18.0,
            output_test_raw_score: f64::NAN,
            delta_train_score: f64::NAN,
            delta_test_score: f64::NAN,
        },
        GarfieldPermutationNullScores {
            bucket: RuleNullBucket {
                rule_len: 5,
                complexity_bin: 0,
            },
            search_train_score: f64::NAN,
            search_test_score: f64::NAN,
            output_train_raw_score: 21.0,
            output_test_raw_score: f64::NAN,
            delta_train_score: f64::NAN,
            delta_test_score: f64::NAN,
        },
        GarfieldPermutationNullScores {
            bucket: RuleNullBucket {
                rule_len: 4,
                complexity_bin: 1,
            },
            search_train_score: f64::NAN,
            search_test_score: f64::NAN,
            output_train_raw_score: 30.0,
            output_test_raw_score: f64::NAN,
            delta_train_score: f64::NAN,
            delta_test_score: f64::NAN,
        },
    ];
    let selected = collect_global_max_rule_null_metric_by_group_len_bucket(vals.as_slice(), true);
    assert_eq!(selected.len(), 3);
    assert!(selected.iter().any(|(bucket, score)| {
        bucket.rule_len == 4 && bucket.unit_group_bin() == 0 && (*score - 18.0).abs() < 1e-12
    }));
    assert!(selected.iter().any(|(bucket, score)| {
        bucket.rule_len == 5 && bucket.unit_group_bin() == 0 && (*score - 21.0).abs() < 1e-12
    }));
    assert!(selected.iter().any(|(bucket, score)| {
        bucket.rule_len == 4 && bucket.unit_group_bin() == 1 && (*score - 30.0).abs() < 1e-12
    }));
}

#[test]
fn test_rule_null_gev_uses_full_repeat_budget_for_tail_calibration() {
    assert!(!rule_null_penalty_allows_adaptive_early_stop(
        RuleNullPenaltyMethod::GevGumbel { fwer_alpha: 0.01 }
    ));
    assert!(!rule_null_penalty_allows_adaptive_early_stop(
        RuleNullPenaltyMethod::BayesHierarchical { fwer_alpha: 0.01 }
    ));
    assert!(rule_null_penalty_allows_adaptive_early_stop(
        RuleNullPenaltyMethod::quantile(0.99)
    ));
}

#[test]
fn test_null_window_preparation_does_not_reuse_phenotype_ml_selection() {
    assert_eq!(
        null_prepare_engine_for_scan("window", Some(MlEngine::Corr)),
        None
    );
    assert_eq!(
        null_prepare_engine_for_scan("geneset", Some(MlEngine::Corr)),
        Some(MlEngine::Corr)
    );
}

#[test]
fn test_final_output_score_for_candidate_uses_raw_score_not_gain_score() {
    let cand = BeamRuleCandidate {
        rule: BeamRule {
            first: BeamLiteral {
                row_index: 0,
                group_id: 0,
                negated: false,
            },
            rest: vec![(
                BeamBinaryOp::And,
                BeamLiteral {
                    row_index: 1,
                    group_id: 1,
                    negated: false,
                },
            )],
        },
        train_score: 1.5,
        test_score: 2.0,
        train: ContinuousRuleScore {
            score: 0.0,
            raw_score: 8.0,
            mean_hit: 0.0,
            mean_miss: 0.0,
            support_frac: 0.5,
            dosage_maf: 0.25,
            n_hit: 4,
            n_ge2: 0,
            n_miss: 4,
        },
        test: ContinuousRuleScore {
            score: 0.0,
            raw_score: 10.0,
            mean_hit: 0.0,
            mean_miss: 0.0,
            support_frac: 0.5,
            dosage_maf: 0.25,
            n_hit: 4,
            n_ge2: 0,
            n_miss: 4,
        },
    };
    let bucket = crate::garfield::permutation::bucket_from_rule(&cand.rule, cand.test.dosage_maf);
    let mut cal = RuleNullCalibrator::new();
    for _ in 0..10 {
        cal.insert(bucket, f64::NAN, 3.0);
    }
    let lookup = cal.finalize();
    let score = final_output_score_for_candidate(&cand, Some(&lookup), false, 0);
    assert!((score - 7.0).abs() < 1e-12);
}

fn make_test_rule_candidate(
    row_indices: &[usize],
    ops: &[BeamBinaryOp],
    test_score: f64,
) -> BeamRuleCandidate {
    let first = BeamLiteral {
        row_index: row_indices[0],
        group_id: row_indices[0],
        negated: false,
    };
    let rest = row_indices
        .iter()
        .copied()
        .skip(1)
        .enumerate()
        .map(|(i, row_index)| {
            (
                ops.get(i).copied().unwrap_or(BeamBinaryOp::And),
                BeamLiteral {
                    row_index,
                    group_id: row_index,
                    negated: false,
                },
            )
        })
        .collect::<Vec<_>>();
    BeamRuleCandidate {
        rule: BeamRule { first, rest },
        train_score: test_score,
        test_score,
        train: ContinuousRuleScore {
            score: test_score,
            raw_score: test_score,
            mean_hit: 0.0,
            mean_miss: 0.0,
            support_frac: 0.5,
            dosage_maf: 0.25,
            n_hit: 4,
            n_ge2: 0,
            n_miss: 4,
        },
        test: ContinuousRuleScore {
            score: test_score,
            raw_score: test_score,
            mean_hit: 0.0,
            mean_miss: 0.0,
            support_frac: 0.5,
            dosage_maf: 0.25,
            n_hit: 4,
            n_ge2: 0,
            n_miss: 4,
        },
    }
}

#[test]
fn test_select_reportable_ranked_hits_falls_back_to_best_singleton_when_all_nonpositive() {
    let beam_hits = vec![
        make_test_rule_candidate(&[0, 1], &[BeamBinaryOp::Xor], -0.25),
        make_test_rule_candidate(&[2], &[], -1.0),
    ];
    let ranked_hits = vec![(0usize, -0.25_f64), (1usize, -1.0_f64)];
    let selected =
        select_reportable_ranked_hits(beam_hits.as_slice(), ranked_hits.as_slice(), 1, false);
    assert_eq!(selected, vec![(1usize, -1.0_f64)]);
}

#[test]
fn test_select_reportable_ranked_hits_uses_output_score_after_penalty() {
    let mut combo = make_test_rule_candidate(&[0, 1], &[BeamBinaryOp::Xor], 12.0);
    combo.test.raw_score = 5.0;
    let mut singleton = make_test_rule_candidate(&[2], &[], 10.0);
    singleton.test.raw_score = 20.0;
    let beam_hits = vec![combo, singleton];
    let ranked_hits = vec![(0usize, 12.0_f64), (1usize, 10.0_f64)];
    let selected =
        select_reportable_ranked_hits(beam_hits.as_slice(), ranked_hits.as_slice(), 1, false);
    assert_eq!(selected, vec![(0usize, 12.0_f64)]);
}

#[test]
fn test_select_reportable_ranked_hits_drops_nonpositive_combo_when_singleton_is_positive() {
    let beam_hits = vec![
        make_test_rule_candidate(&[0, 1], &[BeamBinaryOp::And], -0.10),
        make_test_rule_candidate(&[2], &[], 0.25),
    ];
    let ranked_hits = vec![(1usize, 0.25_f64), (0usize, -0.10_f64)];
    let selected =
        select_reportable_ranked_hits(beam_hits.as_slice(), ranked_hits.as_slice(), 1, false);
    assert_eq!(selected, vec![(1usize, 0.25_f64)]);
}

#[test]
fn test_select_reportable_ranked_hits_applies_topk_to_singleton_only_unit() {
    let mut beam_hits = vec![
        make_test_rule_candidate(&[0], &[], -0.10),
        make_test_rule_candidate(&[1], &[], -0.20),
        make_test_rule_candidate(&[2], &[], -0.30),
        make_test_rule_candidate(&[3, 4], &[BeamBinaryOp::And], -0.40),
    ];
    beam_hits[0].test.raw_score = -0.40;
    beam_hits[1].test.raw_score = 0.40;
    let ranked_hits = vec![
        (0usize, -0.10_f64),
        (1usize, -0.20_f64),
        (2usize, -0.30_f64),
        (3usize, -0.40_f64),
    ];
    let selected =
        select_reportable_ranked_hits(beam_hits.as_slice(), ranked_hits.as_slice(), 2, false);
    assert_eq!(selected, vec![(0usize, -0.10_f64)]);
}

#[test]
fn test_select_reportable_ranked_hits_keeps_raw_design_ordering() {
    let beam_hits = vec![
        make_test_rule_candidate(&[0, 1], &[BeamBinaryOp::Xor], -0.25),
        make_test_rule_candidate(&[2], &[], -1.0),
    ];
    let ranked_hits = vec![(0usize, -0.25_f64), (1usize, -1.0_f64)];
    let selected =
        select_reportable_ranked_hits(beam_hits.as_slice(), ranked_hits.as_slice(), 1, true);
    assert_eq!(selected, vec![(0usize, -0.25_f64)]);
}

#[test]
fn test_apply_logic_rule_output_limit_count_keeps_ties() {
    let mut records = vec![
        GarfieldLogicRuleRecord {
            unit_name: "u1".to_string(),
            unit_kind: "window".to_string(),
            unit_index: 1,
            region_size: 8,
            ml_feature_count: 4,
            ml_rank: ".".to_string(),
            selected_row_indices: vec![1],
            display_ops: Vec::new(),
            display_negated: vec![false],
            snp_name: "snp1".to_string(),
            expr: "expr1".to_string(),
            chrom_field: "1".to_string(),
            bim_snp_name: "snp1".to_string(),
            bim_allele0: "A".to_string(),
            bim_allele1: "T".to_string(),
            allele1_maf: 0.0,
            pos: 1,
            score: 10.0,
            delta_score: "10->10".to_string(),
            perm_pvalue: None,
            perm_fdr: None,
            support_bits: None,
        },
        GarfieldLogicRuleRecord {
            unit_name: "u2".to_string(),
            unit_kind: "window".to_string(),
            unit_index: 2,
            region_size: 8,
            ml_feature_count: 4,
            ml_rank: ".".to_string(),
            selected_row_indices: vec![2],
            display_ops: Vec::new(),
            display_negated: vec![false],
            snp_name: "snp2".to_string(),
            expr: "expr2".to_string(),
            chrom_field: "1".to_string(),
            bim_snp_name: "snp2".to_string(),
            bim_allele0: "A".to_string(),
            bim_allele1: "T".to_string(),
            allele1_maf: 0.0,
            pos: 2,
            score: 9.0,
            delta_score: "9->9".to_string(),
            perm_pvalue: None,
            perm_fdr: None,
            support_bits: None,
        },
        GarfieldLogicRuleRecord {
            unit_name: "u3".to_string(),
            unit_kind: "window".to_string(),
            unit_index: 3,
            region_size: 8,
            ml_feature_count: 4,
            ml_rank: ".".to_string(),
            selected_row_indices: vec![3],
            display_ops: Vec::new(),
            display_negated: vec![false],
            snp_name: "snp3".to_string(),
            expr: "expr3".to_string(),
            chrom_field: "1".to_string(),
            bim_snp_name: "snp3".to_string(),
            bim_allele0: "A".to_string(),
            bim_allele1: "T".to_string(),
            allele1_maf: 0.0,
            pos: 3,
            score: 9.0,
            delta_score: "9->9".to_string(),
            perm_pvalue: None,
            perm_fdr: None,
            support_bits: None,
        },
        GarfieldLogicRuleRecord {
            unit_name: "u4".to_string(),
            unit_kind: "window".to_string(),
            unit_index: 4,
            region_size: 8,
            ml_feature_count: 4,
            ml_rank: ".".to_string(),
            selected_row_indices: vec![4],
            display_ops: Vec::new(),
            display_negated: vec![false],
            snp_name: "snp4".to_string(),
            expr: "expr4".to_string(),
            chrom_field: "1".to_string(),
            bim_snp_name: "snp4".to_string(),
            bim_allele0: "A".to_string(),
            bim_allele1: "T".to_string(),
            allele1_maf: 0.0,
            pos: 4,
            score: 8.0,
            delta_score: "8->8".to_string(),
            perm_pvalue: None,
            perm_fdr: None,
            support_bits: None,
        },
    ];
    apply_logic_rule_output_limit(&mut records, 2, 0.0).unwrap();
    assert_eq!(records.len(), 3);
    assert_eq!(records[1].score, records[2].score);
}

#[test]
fn test_preferred_display_polarity_keeps_original_for_all_negated_rule() {
    let rule = BeamRule {
        first: BeamLiteral {
            row_index: 0,
            group_id: 0,
            negated: true,
        },
        rest: vec![(
            BeamBinaryOp::And,
            BeamLiteral {
                row_index: 1,
                group_id: 1,
                negated: true,
            },
        )],
    };
    let full_bits = vec![0b0011u64];
    let polarity = preferred_display_polarity(&rule, full_bits.as_slice(), 4);
    assert_eq!(polarity, GarfieldRuleDisplayPolarity::Original);
}

#[test]
fn test_rule_display_polarity_keeps_negated_and_form() {
    let sites = vec![test_site("1", 100), test_site("1", 200)];
    let rule = BeamRule {
        first: BeamLiteral {
            row_index: 0,
            group_id: 0,
            negated: true,
        },
        rest: vec![(
            BeamBinaryOp::And,
            BeamLiteral {
                row_index: 1,
                group_id: 1,
                negated: true,
            },
        )],
    };
    let polarity = GarfieldRuleDisplayPolarity::Original;
    let expr = rule_expr_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
    let snp_name = rule_snp_name_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
    let bim_name = rule_bim_name_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
    let ml_rank = rule_ml_rank_name_with_polarity(&rule, polarity);
    let (a0, a1) = rule_bim_alleles_with_polarity(&rule, sites.as_slice(), polarity).unwrap();

    assert_eq!(expr, "NOT BIN(1_100) AND NOT BIN(1_200)");
    assert_eq!(snp_name, "1_100[A]&1_200[A]");
    assert_eq!(bim_name, "1_100[A]&1_200[A]");
    assert_eq!(ml_rank, "!1&!2");
    assert_eq!(a0, "G,G");
    assert_eq!(a1, "A,A");
}

#[test]
fn test_rule_bim_alleles_use_component_delimiter_for_composite_rules() {
    let sites = vec![test_site("1", 100), test_site("1", 200)];
    let rule = BeamRule {
        first: BeamLiteral {
            row_index: 0,
            group_id: 0,
            negated: false,
        },
        rest: vec![(
            BeamBinaryOp::And,
            BeamLiteral {
                row_index: 1,
                group_id: 1,
                negated: false,
            },
        )],
    };
    let (allele0, allele1) = rule_bim_alleles_with_polarity(
        &rule,
        sites.as_slice(),
        GarfieldRuleDisplayPolarity::Original,
    )
    .unwrap();
    assert_eq!(allele0, "A,A");
    assert_eq!(allele1, "G,G");
}

#[test]
fn test_rule_display_polarity_preserves_original_or_rule() {
    let sites = vec![test_site("1", 100), test_site("1", 200)];
    let rule = BeamRule {
        first: BeamLiteral {
            row_index: 0,
            group_id: 0,
            negated: false,
        },
        rest: vec![(
            BeamBinaryOp::Or,
            BeamLiteral {
                row_index: 1,
                group_id: 1,
                negated: false,
            },
        )],
    };
    let polarity = GarfieldRuleDisplayPolarity::Original;
    let expr = rule_expr_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
    let snp_name = rule_snp_name_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
    let bim_name = rule_bim_name_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
    let ml_rank = rule_ml_rank_name_with_polarity(&rule, polarity);

    assert_eq!(expr, "BIN(1_100) OR BIN(1_200)");
    assert_eq!(snp_name, "1_100[G]|1_200[G]");
    assert_eq!(bim_name, "1_100[G]|1_200[G]");
    assert_eq!(ml_rank, "1|2");
}

#[test]
fn test_rule_display_polarity_preserves_original_xor_rule() {
    let sites = vec![test_site("1", 100), test_site("1", 200)];
    let rule = BeamRule {
        first: BeamLiteral {
            row_index: 0,
            group_id: 0,
            negated: false,
        },
        rest: vec![(
            BeamBinaryOp::Xor,
            BeamLiteral {
                row_index: 1,
                group_id: 1,
                negated: false,
            },
        )],
    };
    let polarity = GarfieldRuleDisplayPolarity::Original;
    let expr = rule_expr_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
    let snp_name = rule_snp_name_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
    let bim_name = rule_bim_name_with_polarity(&rule, sites.as_slice(), polarity).unwrap();
    let ml_rank = rule_ml_rank_name_with_polarity(&rule, polarity);

    assert_eq!(expr, "BIN(1_100) XOR BIN(1_200)");
    assert_eq!(snp_name, "1_100[G]^1_200[G]");
    assert_eq!(bim_name, "1_100[G]^1_200[G]");
    assert_eq!(ml_rank, "1^2");
}

#[test]
fn test_logic_sites_prefer_inherited_unique_snp_names() {
    let logic_sites = build_logic_sites_from_metadata(
        &vec![
            test_named_site("1", 100, "rsA"),
            test_named_site("1", 200, "rsB"),
        ],
        &[false, false],
        GarfieldBinMode::Bin,
    );
    let rule = BeamRule {
        first: BeamLiteral {
            row_index: 0,
            group_id: 0,
            negated: false,
        },
        rest: vec![(
            BeamBinaryOp::And,
            BeamLiteral {
                row_index: 1,
                group_id: 1,
                negated: false,
            },
        )],
    };
    let polarity = GarfieldRuleDisplayPolarity::Original;
    let expr = rule_expr_with_polarity(&rule, logic_sites.as_slice(), polarity).unwrap();
    let snp_name = rule_snp_name_with_polarity(&rule, logic_sites.as_slice(), polarity).unwrap();
    let bim_name = rule_bim_name_with_polarity(&rule, logic_sites.as_slice(), polarity).unwrap();

    assert_eq!(expr, "BIN(rsA) AND BIN(rsB)");
    assert_eq!(snp_name, "1_100[G]&1_200[G]");
    assert_eq!(bim_name, "1_100[G]&1_200[G]");
}

#[test]
fn test_logic_sites_fall_back_to_coordinates_when_snp_names_duplicate() {
    let logic_sites = build_logic_sites_from_metadata(
        &vec![
            test_named_site("1", 100, "dup"),
            test_named_site("1", 200, "dup"),
        ],
        &[false, false],
        GarfieldBinMode::Bin,
    );
    let rule = BeamRule {
        first: BeamLiteral {
            row_index: 0,
            group_id: 0,
            negated: false,
        },
        rest: vec![(
            BeamBinaryOp::And,
            BeamLiteral {
                row_index: 1,
                group_id: 1,
                negated: false,
            },
        )],
    };
    let polarity = GarfieldRuleDisplayPolarity::Original;
    let expr = rule_expr_with_polarity(&rule, logic_sites.as_slice(), polarity).unwrap();
    let snp_name = rule_snp_name_with_polarity(&rule, logic_sites.as_slice(), polarity).unwrap();

    assert_eq!(expr, "BIN(1_100) AND BIN(1_200)");
    assert_eq!(snp_name, "1_100[G]&1_200[G]");
}

#[test]
fn test_owned_logic_sites_direct_builder_preserves_unique_and_duplicate_names() {
    let unique = build_logic_sites_from_metadata_owned_direct(
        vec![
            test_named_site("1", 100, "rsA"),
            test_named_site("1", 200, "rsB"),
        ],
        &[false, false],
        GarfieldBinMode::Bin,
    );
    assert_eq!(unique.len(), 2);
    assert_eq!(unique[0].snp(), "rsA");
    assert_eq!(unique[1].snp(), "rsB");

    let duplicate = build_logic_sites_from_metadata_owned_direct(
        vec![
            test_named_site("1", 100, "dup"),
            test_named_site("1", 200, "dup"),
        ],
        &[false, false],
        GarfieldBinMode::Bin,
    );
    assert_eq!(duplicate.len(), 2);
    assert_eq!(duplicate[0].snp(), "1_100");
    assert_eq!(duplicate[1].snp(), "1_200");
}

#[test]
fn cached_max6_site_builder_shares_source_metadata_and_trait_orientation() {
    let source_sites = Arc::new(
        CompactMax6SiteMetadata::from_sites(vec![
            test_named_site("1", 100, "dup"),
            test_named_site("1", 200, "dup"),
            test_named_site("1", 300, "rsC"),
        ])
        .unwrap(),
    );
    let unique = build_max6_logic_sites_from_shared_metadata(
        &[10, 20, 30],
        source_sites.clone(),
        &[10, 30],
        &[false, true],
    )
    .unwrap();
    assert_eq!(unique[0].garfield_snp_name(), "dup");
    assert_eq!(unique[1].garfield_snp_name(), "rsC");
    assert_eq!(unique[1].garfield_ref_allele(), "G");
    assert_eq!(unique[1].garfield_alt_allele(), "A");
    assert!(matches!(
        &unique[1].backing,
        GarfieldLogicSiteBacking::SharedMax6 { source_sites: shared, .. }
            if Arc::ptr_eq(shared, &source_sites)
    ));

    let duplicate = build_max6_logic_sites_from_shared_metadata(
        &[10, 20, 30],
        source_sites.clone(),
        &[10, 20],
        &[false, false],
    )
    .unwrap();
    assert_eq!(duplicate[0].garfield_snp_name(), "1_100");
    assert_eq!(duplicate[1].garfield_snp_name(), "1_200");
    assert!(build_max6_logic_sites_from_shared_metadata(
        &[10, 20, 30],
        source_sites,
        &[10, 40],
        &[false, false],
    )
    .is_err());
}

#[test]
fn max6_active_margin_rows_borrows_prepared_rows_and_owns_fallback() {
    let prepared_rows = vec![2usize, 7, 11];
    let borrowed = build_max6_active_margin_rows(Some(prepared_rows.as_slice()), &[], &[]);
    assert!(matches!(borrowed, std::borrow::Cow::Borrowed(_)));
    assert!(std::ptr::eq(borrowed.as_ptr(), prepared_rows.as_ptr()));
    assert_eq!(borrowed.as_ref(), &[2, 7, 11]);

    let units = vec![
        GarfieldLogicUnit {
            label: "a".to_string(),
            indices: vec![8, 3, 8],
            spans: Vec::new(),
        },
        GarfieldLogicUnit {
            label: "b".to_string(),
            indices: vec![5, 3],
            spans: Vec::new(),
        },
    ];
    let owned = build_max6_active_margin_rows(None, &[1, 0], &units);
    assert!(matches!(owned, std::borrow::Cow::Owned(_)));
    assert_eq!(owned.as_ref(), &[3, 5, 8]);
}

#[test]
fn test_build_logic_sites_from_metadata_applies_row_flip_to_alleles() {
    let logic_sites = build_logic_sites_from_metadata(
        &vec![
            test_named_site("1", 100, "rsA"),
            test_named_site("1", 200, "rsB"),
        ],
        &[true, false],
        GarfieldBinMode::Bin,
    );
    assert_eq!(logic_sites.len(), 2);
    assert_eq!(logic_sites[0].garfield_ref_allele(), "G");
    assert_eq!(logic_sites[0].garfield_alt_allele(), "A");
    assert_eq!(logic_sites[1].garfield_ref_allele(), "A");
    assert_eq!(logic_sites[1].garfield_alt_allele(), "G");
}

#[test]
fn test_dedup_logic_rule_records_prefers_and_form_same_support() {
    let full_bits = vec![0b1010u64];
    let records = vec![
        GarfieldLogicRuleRecord {
            unit_name: "u_not".to_string(),
            unit_kind: "window".to_string(),
            unit_index: 1,
            region_size: 2,
            ml_feature_count: 2,
            ml_rank: "!1&!2".to_string(),
            selected_row_indices: vec![0, 1],
            display_ops: vec![BeamBinaryOp::And],
            display_negated: vec![true, true],
            snp_name: "1_100(A)&1_200(A)".to_string(),
            expr: "NOT BIN(1_100) AND NOT BIN(1_200)".to_string(),
            chrom_field: "1".to_string(),
            bim_snp_name: "1_100(A)&1_200(A)".to_string(),
            bim_allele0: "G,G".to_string(),
            bim_allele1: "A,A".to_string(),
            allele1_maf: 0.0,
            pos: 100,
            score: 5.0,
            delta_score: "0.1&0.2->5.0".to_string(),
            perm_pvalue: None,
            perm_fdr: None,
            support_bits: Some(GarfieldRuleSupportBits::Binary(
                full_bits.clone().into_boxed_slice(),
            )),
        },
        GarfieldLogicRuleRecord {
            unit_name: "u_pos".to_string(),
            unit_kind: "window".to_string(),
            unit_index: 1,
            region_size: 2,
            ml_feature_count: 2,
            ml_rank: "1|2".to_string(),
            selected_row_indices: vec![0, 1],
            display_ops: vec![BeamBinaryOp::Or],
            display_negated: vec![false, false],
            snp_name: "1_100(G)|1_200(G)".to_string(),
            expr: "BIN(1_100) OR BIN(1_200)".to_string(),
            chrom_field: "1".to_string(),
            bim_snp_name: "1_100(G)|1_200(G)".to_string(),
            bim_allele0: "A,A".to_string(),
            bim_allele1: "G,G".to_string(),
            allele1_maf: 0.0,
            pos: 100,
            score: 5.0,
            delta_score: "0.1|0.2->5.0".to_string(),
            perm_pvalue: None,
            perm_fdr: None,
            support_bits: Some(GarfieldRuleSupportBits::Binary(
                full_bits.into_boxed_slice(),
            )),
        },
    ];
    let empty_logic_bits = GarfieldLogicBits {
        bits_flat: Vec::new(),
        bits_mmap: None,
        bits_mmap_words: 0,
        bits_mmap_owner: None,
        bits_hi_flat: None,
        row_words: 0,
        sample_ids: Vec::new(),
        sites: Vec::new(),
        group_ids: Vec::new(),
        n_samples: 0,
    };
    let deduped = dedup_logic_rule_records(records, &empty_logic_bits).unwrap();
    assert_eq!(deduped.len(), 1);
    assert_eq!(deduped[0].snp_name, "1_100(A)&1_200(A)");
    assert_eq!(deduped[0].expr, "NOT BIN(1_100) AND NOT BIN(1_200)");
}

#[test]
fn test_dedup_logic_rule_records_keeps_same_support_across_gene_units() {
    let full_bits = vec![0b1010u64];
    let records = vec![
        GarfieldLogicRuleRecord {
            unit_name: "gene_a".to_string(),
            unit_kind: "gene".to_string(),
            unit_index: 11,
            region_size: 2,
            ml_feature_count: 2,
            ml_rank: "1".to_string(),
            selected_row_indices: vec![0],
            display_ops: vec![],
            display_negated: vec![false],
            snp_name: "1_100(G)".to_string(),
            expr: "BIN(1_100)".to_string(),
            chrom_field: "1".to_string(),
            bim_snp_name: "1_100(G)".to_string(),
            bim_allele0: "A".to_string(),
            bim_allele1: "G".to_string(),
            allele1_maf: 0.0,
            pos: 100,
            score: 8.0,
            delta_score: "8.0->8.0".to_string(),
            perm_pvalue: None,
            perm_fdr: None,
            support_bits: Some(GarfieldRuleSupportBits::Binary(
                full_bits.clone().into_boxed_slice(),
            )),
        },
        GarfieldLogicRuleRecord {
            unit_name: "gene_b".to_string(),
            unit_kind: "gene".to_string(),
            unit_index: 12,
            region_size: 2,
            ml_feature_count: 2,
            ml_rank: "1".to_string(),
            selected_row_indices: vec![0],
            display_ops: vec![],
            display_negated: vec![false],
            snp_name: "1_100(G)".to_string(),
            expr: "BIN(1_100)".to_string(),
            chrom_field: "1".to_string(),
            bim_snp_name: "1_100(G)".to_string(),
            bim_allele0: "A".to_string(),
            bim_allele1: "G".to_string(),
            allele1_maf: 0.0,
            pos: 100,
            score: 7.0,
            delta_score: "7.0->7.0".to_string(),
            perm_pvalue: None,
            perm_fdr: None,
            support_bits: Some(GarfieldRuleSupportBits::Binary(
                full_bits.into_boxed_slice(),
            )),
        },
    ];
    let empty_logic_bits = GarfieldLogicBits {
        bits_flat: Vec::new(),
        bits_mmap: None,
        bits_mmap_words: 0,
        bits_mmap_owner: None,
        bits_hi_flat: None,
        row_words: 0,
        sample_ids: Vec::new(),
        sites: Vec::new(),
        group_ids: Vec::new(),
        n_samples: 0,
    };
    let deduped = dedup_logic_rule_records(records, &empty_logic_bits).unwrap();
    assert_eq!(deduped.len(), 2);
    assert_eq!(deduped[0].unit_name, "gene_a");
    assert_eq!(deduped[1].unit_name, "gene_b");
}

#[test]
fn test_write_logic_pseudo_plink_expands_singletons_and_multi_pos_children() {
    let dir = make_temp_dir("pseudo_export_expand");
    let prefix = dir.join("pseudo");
    let prefix_str = prefix.to_string_lossy().to_string();
    let sample_ids = vec![
        "S1".to_string(),
        "S2".to_string(),
        "S3".to_string(),
        "S4".to_string(),
    ];
    let sites = build_logic_sites_from_metadata(
        &vec![test_site("1", 100), test_site("1", 200)],
        &[false, false],
        GarfieldBinMode::Bin,
    );
    let logic_bits = GarfieldLogicBits {
        bits_flat: vec![0b0101u64, 0b0110u64],
        bits_mmap: None,
        bits_mmap_words: 0,
        bits_mmap_owner: None,
        bits_hi_flat: None,
        row_words: 1,
        sample_ids: sample_ids.clone(),
        sites,
        group_ids: vec![0, 1],
        n_samples: 4,
    };
    let records = vec![GarfieldLogicRuleRecord {
        unit_name: "u1".to_string(),
        unit_kind: "window".to_string(),
        unit_index: 1,
        region_size: 2,
        ml_feature_count: 2,
        ml_rank: "1&!2".to_string(),
        selected_row_indices: vec![0, 1],
        display_ops: vec![BeamBinaryOp::And],
        display_negated: vec![false, true],
        snp_name: "1_100(G)&1_200(A)".to_string(),
        expr: "BIN(1_100) AND NOT BIN(1_200)".to_string(),
        chrom_field: "1".to_string(),
        bim_snp_name: "1_100(G)&1_200(A)".to_string(),
        bim_allele0: "A,C".to_string(),
        bim_allele1: "G,T".to_string(),
        allele1_maf: 0.0,
        pos: 100,
        score: 10.0,
        delta_score: "0.1&0.2->10".to_string(),
        perm_pvalue: None,
        perm_fdr: None,
        support_bits: None,
    }];

    write_logic_pseudo_plink(
        prefix_str.as_str(),
        sample_ids.as_slice(),
        records.as_slice(),
        &logic_bits,
    )
    .unwrap();

    let bim_txt = fs::read_to_string(format!("{prefix_str}.bim")).unwrap();
    assert_eq!(
        bim_txt.lines().collect::<Vec<_>>(),
        vec![
            "1\t1_100[G]\t0\t100\tA\tG",
            "1\t1_200[A]\t0\t200\tG\tA",
            "1\t1_100(G)&1_200(A)\t0\t100\tA,C\tG,T",
            "1\t1_100(G)&1_200(A)\t0\t200\tA,C\tG,T",
        ]
    );

    let bed_bytes = fs::read(format!("{prefix_str}.bed")).unwrap();
    assert_eq!(bed_bytes, vec![0x6C, 0x1B, 0x01, 0x33, 0xC3, 0x03, 0x03]);
}

#[test]
fn test_build_rule_delta_score_annotation_reports_raw_prefix_chain() {
    let rows = vec![vec![1u8, 1, 1, 0], vec![1u8, 1, 0, 1], vec![1u8, 0, 1, 1]];
    let (bits_flat, row_words) = pack_test_binary_rows(&rows);
    let y = vec![2.0, 1.0, -1.0, -2.0];
    let params = BeamSearchParams::default();
    let lit0 = BeamLiteral {
        row_index: 0,
        group_id: 0,
        negated: false,
    };
    let lit1 = BeamLiteral {
        row_index: 1,
        group_id: 1,
        negated: false,
    };
    let lit2 = BeamLiteral {
        row_index: 2,
        group_id: 2,
        negated: false,
    };
    let rule = BeamRule {
        first: lit0,
        rest: vec![(BeamBinaryOp::And, lit1), (BeamBinaryOp::And, lit2)],
    };

    let actual = build_rule_delta_score_annotation(
        &rule,
        y.as_slice(),
        bits_flat.as_slice(),
        None,
        row_words,
        rows.len(),
        y.len(),
        &params,
        GarfieldRuleDisplayPolarity::Original,
    )
    .unwrap();
    let single0 = evaluate_rule_test_raw_score(
        &singleton_rule_from_literal(lit0),
        y.as_slice(),
        bits_flat.as_slice(),
        None,
        row_words,
        rows.len(),
        y.len(),
        &params,
    )
    .unwrap();
    let single1 = evaluate_rule_test_raw_score(
        &singleton_rule_from_literal(lit1),
        y.as_slice(),
        bits_flat.as_slice(),
        None,
        row_words,
        rows.len(),
        y.len(),
        &params,
    )
    .unwrap();
    let single2 = evaluate_rule_test_raw_score(
        &singleton_rule_from_literal(lit2),
        y.as_slice(),
        bits_flat.as_slice(),
        None,
        row_words,
        rows.len(),
        y.len(),
        &params,
    )
    .unwrap();
    let pair = evaluate_rule_test_raw_score(
        &BeamRule {
            first: lit0,
            rest: vec![(BeamBinaryOp::And, lit1)],
        },
        y.as_slice(),
        bits_flat.as_slice(),
        None,
        row_words,
        rows.len(),
        y.len(),
        &params,
    )
    .unwrap();
    let triple = evaluate_rule_test_raw_score(
        &rule,
        y.as_slice(),
        bits_flat.as_slice(),
        None,
        row_words,
        rows.len(),
        y.len(),
        &params,
    )
    .unwrap();
    let expected = format!(
        "{}&{}->{}&{}->{}",
        format_delta_metric_value(single0),
        format_delta_metric_value(single1),
        format_delta_metric_value(pair),
        format_delta_metric_value(single2),
        format_delta_metric_value(triple),
    );
    assert_eq!(actual, expected);
}

#[test]
fn test_build_simbench_delta_score_annotation_reports_raw_prefix_chain() {
    let rows = vec![vec![1u8, 1, 1, 0], vec![1u8, 1, 0, 1], vec![1u8, 0, 1, 1]];
    let (bits_flat, row_words) = pack_test_binary_rows(&rows);
    let y = vec![2.0, 1.0, -1.0, -2.0];
    let params = BeamSearchParams::default();
    let sites = vec![
        test_site("1", 100),
        test_site("1", 200),
        test_site("1", 300),
    ];
    let negated = vec![false, false, false];

    let actual = build_simbench_delta_score_annotation(
        SimBenchLogic::And,
        negated.as_slice(),
        sites.as_slice(),
        y.as_slice(),
        bits_flat.as_slice(),
        None,
        row_words,
        rows.len(),
        y.len(),
        &params,
    )
    .unwrap();
    let lit0 = BeamLiteral {
        row_index: 0,
        group_id: 0,
        negated: false,
    };
    let lit1 = BeamLiteral {
        row_index: 1,
        group_id: 1,
        negated: false,
    };
    let lit2 = BeamLiteral {
        row_index: 2,
        group_id: 2,
        negated: false,
    };
    let single0 = evaluate_rule_test_raw_score(
        &singleton_rule_from_literal(lit0),
        y.as_slice(),
        bits_flat.as_slice(),
        None,
        row_words,
        rows.len(),
        y.len(),
        &params,
    )
    .unwrap();
    let single1 = evaluate_rule_test_raw_score(
        &singleton_rule_from_literal(lit1),
        y.as_slice(),
        bits_flat.as_slice(),
        None,
        row_words,
        rows.len(),
        y.len(),
        &params,
    )
    .unwrap();
    let single2 = evaluate_rule_test_raw_score(
        &singleton_rule_from_literal(lit2),
        y.as_slice(),
        bits_flat.as_slice(),
        None,
        row_words,
        rows.len(),
        y.len(),
        &params,
    )
    .unwrap();
    let pair = evaluate_rule_test_raw_score(
        &BeamRule {
            first: lit0,
            rest: vec![(BeamBinaryOp::And, lit1)],
        },
        y.as_slice(),
        bits_flat.as_slice(),
        None,
        row_words,
        rows.len(),
        y.len(),
        &params,
    )
    .unwrap();
    let triple = evaluate_rule_test_raw_score(
        &BeamRule {
            first: lit0,
            rest: vec![(BeamBinaryOp::And, lit1), (BeamBinaryOp::And, lit2)],
        },
        y.as_slice(),
        bits_flat.as_slice(),
        None,
        row_words,
        rows.len(),
        y.len(),
        &params,
    )
    .unwrap();
    let expected = format!(
        "{}&{}->{}&{}->{}",
        format_delta_metric_value(single0),
        format_delta_metric_value(single1),
        format_delta_metric_value(pair),
        format_delta_metric_value(single2),
        format_delta_metric_value(triple),
    );
    assert_eq!(actual, expected);
}

#[test]
fn test_write_logic_dual_bits_as_plink_row_preserves_heterozygotes() {
    let mut row_buf = vec![0u8; 1];
    let mut out = Vec::<u8>::new();
    write_logic_dual_bits_as_plink_row(
        &mut out,
        row_buf.as_mut_slice(),
        &[0b1110u64],
        &[0b0100u64],
        4,
        false,
    )
    .unwrap();
    assert_eq!(out, vec![0xB8u8]);

    out.clear();
    write_logic_dual_bits_as_plink_row(
        &mut out,
        row_buf.as_mut_slice(),
        &[0b1110u64],
        &[0b0100u64],
        4,
        true,
    )
    .unwrap();
    assert_eq!(out, vec![0x8Bu8]);
}

fn normalize_test_simbench_raw_row(row: &mut [f32], site: &mut SiteInfo) -> Result<(), String> {
    let ref_upper = site.ref_allele.trim().to_ascii_uppercase();
    let alt_upper = site.alt_allele.trim().to_ascii_uppercase();
    if ref_upper > alt_upper {
        for v in row.iter_mut() {
            if let Some(g) = normalize_genotype3(*v) {
                *v = f32::from(2u8.saturating_sub(g));
            }
        }
        site.ref_allele = alt_upper;
        site.alt_allele = ref_upper;
    } else {
        site.ref_allele = ref_upper;
        site.alt_allele = alt_upper;
    }
    Ok(())
}

fn pack_test_simbench_raw_row_dual_words(row: &[f32]) -> Result<(Vec<u64>, Vec<u64>), String> {
    let row_words = words_for_samples(row.len());
    let mut ge1 = vec![0u64; row_words];
    let mut ge2 = vec![0u64; row_words];
    for (i, &v) in row.iter().enumerate() {
        let Some(g) = normalize_genotype3(v) else {
            continue;
        };
        if g >= 1 {
            ge1[i >> 6] |= 1u64 << (i & 63);
        }
        if g >= 2 {
            ge2[i >> 6] |= 1u64 << (i & 63);
        }
    }
    Ok((ge1, ge2))
}

fn test_site(chrom: &str, pos: i32) -> SiteInfo {
    SiteInfo {
        chrom: chrom.to_string(),
        pos,
        snp: format!("{chrom}_{pos}"),
        ref_allele: "A".to_string(),
        alt_allele: "G".to_string(),
    }
}

fn test_named_site(chrom: &str, pos: i32, snp: &str) -> SiteInfo {
    SiteInfo {
        chrom: chrom.to_string(),
        pos,
        snp: snp.to_string(),
        ref_allele: "A".to_string(),
        alt_allele: "G".to_string(),
    }
}

#[test]
fn test_build_valid_null_chunks_uses_fixed_non_overlapping_windows() {
    let mut sites = Vec::<SiteInfo>::new();
    for pos in 0..55 {
        sites.push(test_site("1", pos * 1_000));
    }
    for pos in 200..260 {
        sites.push(test_site("1", pos * 1_000));
    }
    for pos in 500..520 {
        sites.push(test_site("1", pos * 1_000));
    }
    let (chunks, spans) = build_valid_null_chunks(&sites, 200_000, 50).unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].start_index, 0);
    assert_eq!(chunks[0].end_index, 55);
    assert_eq!(chunks[0].window_index, 0);
    assert_eq!(chunks[1].start_index, 55);
    assert_eq!(chunks[1].end_index, 115);
    assert_eq!(chunks[1].window_index, 1);
}

#[test]
fn test_sample_null_chunks_stratified_preserves_chromosome_proportions() {
    let mut sites = Vec::<SiteInfo>::new();
    for win in 0..8 {
        let start = win * 200_000;
        for offset in 0..60 {
            sites.push(test_site("1", start + offset));
        }
    }
    for win in 0..2 {
        let start = win * 200_000;
        for offset in 0..60 {
            sites.push(test_site("2", start + offset));
        }
    }
    let (picked, total_valid) = sample_null_chunks_stratified(&sites, 100_000, 5, 50, 7).unwrap();
    assert_eq!(total_valid, 10);
    assert_eq!(picked.len(), 5);
    let chr1 = picked.iter().filter(|c| c.chrom_code == 1).count();
    let chr2 = picked.iter().filter(|c| c.chrom_code == 2).count();
    assert_eq!(chr1, 4);
    assert_eq!(chr2, 1);
}

#[test]
fn test_resolve_ml_top_random_counts_keeps_random_slice_for_small_keep_k() {
    assert_eq!(resolve_ml_top_random_counts(2, 0.80), (1, 1));
    assert_eq!(resolve_ml_top_random_counts(3, 0.80), (2, 1));
    assert_eq!(resolve_ml_top_random_counts(10, 0.80), (8, 2));
}

#[test]
fn test_parse_scan_bimrange_item_accepts_bp_interval() {
    let parsed = parse_scan_bimrange_item("chr10:111000000-111200000").unwrap();
    assert_eq!(parsed.chrom, "chr10");
    assert_eq!(parsed.bp_start, 111_000_000);
    assert_eq!(parsed.bp_end, 111_200_000);
}

#[test]
fn selected_window_builder_matches_full_builder_for_bimrange() {
    Python::initialize();
    let sites = (0..200)
        .map(|index| SiteInfo {
            chrom: "1".to_string(),
            pos: 10 + index * 10,
            snp: format!("rs{index}"),
            ref_allele: "A".to_string(),
            alt_allele: "G".to_string(),
        })
        .collect::<Vec<_>>();
    let ranges = vec![GarfieldScanBimRange {
        chrom: "1".to_string(),
        bp_start: 500,
        bp_end: 900,
    }];
    let full = build_logic_units(&sites, "window", None, None, 100, Some(50)).unwrap();
    let expected = full
        .iter()
        .enumerate()
        .filter_map(|(index, unit)| {
            unit_overlaps_scan_bimranges(unit, ranges.as_slice()).then_some(index)
        })
        .collect::<Vec<_>>();
    let (selected, original_indices) =
        build_window_units_overlapping_bimranges(&sites, 100, 50, ranges.as_slice()).unwrap();
    assert_eq!(original_indices, expected);
    assert_eq!(
        selected
            .iter()
            .map(|unit| unit.label.as_str())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|&index| full[index].label.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn streaming_logic_writer_preserves_rows_and_removes_temp_file() {
    let expected = vec![0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 7, 9, 11, 13];
    let mut writer = Max6StreamingLogicBitsAccumulator::new(3, 2).unwrap();
    writer.append_rows(&expected[..3]).unwrap();
    writer.append_rows(&expected[3..]).unwrap();
    let (flat, mmap, words, owner) = writer.finish().unwrap();
    assert!(flat.is_empty());
    let mmap = mmap.expect("streaming logic writer should use mapped storage");
    let observed = unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const u64, words) };
    assert_eq!(observed, expected.as_slice());
    let path = owner.path.clone();
    drop(mmap);
    drop(owner);
    assert!(!path.exists());
}

#[test]
fn streamed_window_preparation_matches_full_qc_and_scopes_bimrange() {
    Python::initialize();
    let fixture_dir = make_temp_dir("max6-window-stream");
    let prefix = fixture_dir.join("toy");
    let sample_ids = (0..64usize)
        .map(|index| format!("sample{index}"))
        .collect::<Vec<_>>();
    let mut fam = BufWriter::new(File::create(format!("{}.fam", prefix.display())).unwrap());
    for sample_id in &sample_ids {
        writeln!(fam, "{sample_id}\t{sample_id}\t0\t0\t1\t-9").unwrap();
    }
    fam.flush().unwrap();

    let mut bed = BufWriter::new(File::create(format!("{}.bed", prefix.display())).unwrap());
    bed.write_all(&[0x6c, 0x1b, 0x01]).unwrap();
    for row_index in 0..100usize {
        let mut packed = vec![0u8; sample_ids.len().div_ceil(4)];
        for sample in 0..sample_ids.len() {
            let genotype = if (sample + row_index * 7) % 13 < 6 {
                2i8
            } else {
                0i8
            };
            packed[sample >> 2] |= plink2bits_from_logic_g(genotype) << ((sample & 3) * 2);
        }
        bed.write_all(&packed).unwrap();
    }
    bed.flush().unwrap();

    let mut bim = BufWriter::new(File::create(format!("{}.bim", prefix.display())).unwrap());
    for row_index in 0..100usize {
        let (chrom, local_index) = if row_index < 80 {
            ("1", row_index)
        } else {
            ("2", row_index - 80)
        };
        writeln!(
            bim,
            "{chrom}\trs{row_index}\t0\t{}\tA\tG",
            (local_index + 1) * 100
        )
        .unwrap();
    }
    bim.flush().unwrap();

    let sample_indices = (0..sample_ids.len()).collect::<Vec<_>>();
    let (full_keep, full_source_rows, full_logic) = prepare_max6_global_qc_and_logic_bits(
        prefix.to_str().unwrap(),
        sample_ids.as_slice(),
        sample_indices.as_slice(),
        0.01,
        0.0,
        1.0,
        false,
        2,
        0,
        None,
    )
    .unwrap();
    let full_units = build_logic_units(
        full_logic.sites.as_slice(),
        "window",
        None,
        None,
        500,
        Some(250),
    )
    .unwrap();

    let full_stream = prepare_max6_window_stream(
        prefix.to_str().unwrap(),
        sample_ids.as_slice(),
        sample_indices.as_slice(),
        500,
        Some(250),
        &[],
        0.01,
        0.0,
        1.0,
        false,
        2,
        None,
    )
    .unwrap();
    let mut progress_updates = Vec::<(usize, usize)>::new();
    let mut record_progress = |done: usize, total: usize| {
        progress_updates.push((done, total));
        Ok::<(), String>(())
    };
    let progress_stream = prepare_max6_window_streamed_ld_with_callback(
        prefix.to_str().unwrap(),
        sample_ids.as_slice(),
        sample_indices.as_slice(),
        500,
        Some(250),
        &[],
        0.01,
        0.0,
        1.0,
        false,
        2,
        0,
        1.0,
        None,
        false,
        None,
        Some(&mut record_progress),
    )
    .unwrap();
    drop(progress_stream);
    assert!(progress_updates.len() >= 2);
    assert_eq!(progress_updates[0], (0, 100));
    assert_eq!(progress_updates.last().unwrap().1, 100);
    assert!(progress_updates.last().unwrap().0 > progress_updates[0].0);
    assert!(progress_updates
        .windows(2)
        .all(|pair| pair[0].0 <= pair[1].0));
    assert_eq!(full_stream.global_site_keep, full_keep);
    assert_eq!(full_stream.packed_source_rows, full_source_rows);
    assert_eq!(full_stream.logic_bits.bits(), full_logic.bits());
    assert_eq!(full_stream.units.len(), full_units.len());
    assert!(full_stream
        .units
        .iter()
        .zip(full_units.iter())
        .all(|(streamed, full)| streamed.label == full.label && streamed.indices == full.indices));

    let scan_ranges = vec![GarfieldScanBimRange {
        chrom: "1".to_string(),
        bp_start: 3000,
        bp_end: 5000,
    }];
    let selected_full = full_units
        .iter()
        .enumerate()
        .filter_map(|(index, unit)| {
            unit_overlaps_scan_bimranges(unit, scan_ranges.as_slice()).then_some(index)
        })
        .collect::<Vec<_>>();
    let range_stream = prepare_max6_window_stream(
        prefix.to_str().unwrap(),
        sample_ids.as_slice(),
        sample_indices.as_slice(),
        500,
        Some(250),
        scan_ranges.as_slice(),
        0.01,
        0.0,
        1.0,
        false,
        2,
        None,
    )
    .unwrap();

    let train_indices = (0..sample_ids.len()).collect::<Vec<_>>();
    let remap_units = |units: &[Max6GrmMax6LdPreparedUnit], source_rows: &[usize]| {
        units
            .iter()
            .map(|unit| Max6GrmMax6LdPreparedUnit {
                selected_global_rows: unit
                    .selected_global_rows
                    .iter()
                    .map(|&row| source_rows[row])
                    .collect(),
                pair_groups: unit.pair_groups.clone(),
                ld_compression: unit
                    .ld_compression
                    .iter()
                    .map(|record| GarfieldLdCompressionRecord {
                        kept_global_row: source_rows[record.kept_global_row],
                        dropped_global_row: source_rows[record.dropped_global_row],
                        r2: record.r2,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>()
    };
    let remap_active_rows = |rows: &[usize], source_rows: &[usize]| {
        let mut mapped = rows.iter().map(|&row| source_rows[row]).collect::<Vec<_>>();
        mapped.sort_unstable();
        mapped.dedup();
        mapped
    };
    for ld_tag_r2 in [1.0, 0.80] {
        let streamed_full = prepare_max6_window_streamed_ld(
            prefix.to_str().unwrap(),
            sample_ids.as_slice(),
            sample_indices.as_slice(),
            500,
            Some(250),
            &[],
            0.01,
            0.0,
            1.0,
            false,
            2,
            ld_tag_r2,
            None,
        )
        .unwrap();
        let full_reference = prepare_max6_screen_selection(
            &full_stream.logic_bits,
            full_stream.units.as_slice(),
            &(0..full_stream.units.len()).collect::<Vec<_>>(),
            "window",
            train_indices.as_slice(),
            None,
            None,
            ld_tag_r2,
        )
        .unwrap();
        assert_eq!(
            streamed_full
                .units
                .iter()
                .map(|unit| unit.original_unit_index)
                .collect::<Vec<_>>(),
            full_reference.original_unit_indices
        );
        assert_eq!(
            streamed_full.screen_units,
            remap_units(
                &full_reference.units,
                full_stream.packed_source_rows.as_slice()
            )
        );
        assert_eq!(
            streamed_full.active_source_rows,
            remap_active_rows(
                full_reference.active_rows.as_slice(),
                full_stream.packed_source_rows.as_slice()
            )
        );
        assert_eq!(
            streamed_full.window_pair_count,
            full_reference.window_pair_count
        );
        assert_eq!(streamed_full.global_site_keep, full_stream.global_site_keep);
        assert_eq!(
            streamed_full
                .units
                .iter()
                .map(|unit| unit.label.as_str())
                .collect::<Vec<_>>(),
            full_stream
                .units
                .iter()
                .map(|unit| unit.label.as_str())
                .collect::<Vec<_>>()
        );

        let streamed_range = prepare_max6_window_streamed_ld(
            prefix.to_str().unwrap(),
            sample_ids.as_slice(),
            sample_indices.as_slice(),
            500,
            Some(250),
            scan_ranges.as_slice(),
            0.01,
            0.0,
            1.0,
            false,
            2,
            ld_tag_r2,
            None,
        )
        .unwrap();
        let range_reference = prepare_max6_screen_selection(
            &range_stream.logic_bits,
            range_stream.units.as_slice(),
            &(0..range_stream.units.len()).collect::<Vec<_>>(),
            "window",
            train_indices.as_slice(),
            None,
            None,
            ld_tag_r2,
        )
        .unwrap();
        let expected_range_indices = range_reference
            .original_unit_indices
            .iter()
            .map(|&index| range_stream.source_unit_indices[index])
            .collect::<Vec<_>>();
        assert_eq!(
            streamed_range
                .units
                .iter()
                .map(|unit| unit.original_unit_index)
                .collect::<Vec<_>>(),
            expected_range_indices
        );
        assert_eq!(
            streamed_range.screen_units,
            remap_units(
                &range_reference.units,
                range_stream.packed_source_rows.as_slice()
            )
        );
        assert_eq!(
            streamed_range.active_source_rows,
            remap_active_rows(
                range_reference.active_rows.as_slice(),
                range_stream.packed_source_rows.as_slice()
            )
        );
        assert_eq!(
            streamed_range.window_pair_count,
            range_reference.window_pair_count
        );
        assert!(streamed_range
            .global_site_keep
            .iter()
            .enumerate()
            .all(|(row, &keep)| keep == (row < 80 && full_keep[row])));
        assert_eq!(streamed_range.selected_window_count, selected_full.len());
    }

    let streamed_default = prepare_max6_window_streamed_ld(
        prefix.to_str().unwrap(),
        sample_ids.as_slice(),
        sample_indices.as_slice(),
        500,
        Some(250),
        &[],
        0.01,
        0.0,
        1.0,
        false,
        2,
        GARFIELD_MAX6_LD_TAG_R2_DEFAULT,
        None,
    )
    .unwrap();
    let production_tags = prepare_max6_genotype_tags_owned(
        prefix.to_string_lossy().into_owned(),
        sample_ids.clone(),
        sample_indices.clone(),
        String::new(),
        None,
        None,
        500,
        Some(250),
        None,
        0.01,
        0.0,
        1.0,
        false,
        2,
        0,
        GARFIELD_MAX6_LD_TAG_R2_DEFAULT,
    )
    .unwrap();
    assert_eq!(production_tags.global_site_keep, full_keep);
    assert_eq!(
        production_tags
            .units
            .iter()
            .map(|unit| unit.original_unit_index)
            .collect::<Vec<_>>(),
        streamed_default
            .units
            .iter()
            .map(|unit| unit.original_unit_index)
            .collect::<Vec<_>>()
    );
    assert_eq!(production_tags.screen_units, streamed_default.screen_units);
    assert_eq!(
        production_tags.active_source_rows,
        streamed_default.active_source_rows
    );
    assert_eq!(
        production_tags.window_pair_count,
        streamed_default.window_pair_count
    );

    assert_eq!(range_stream.source_unit_indices, selected_full);
    assert_eq!(range_stream.units.len(), selected_full.len());
    assert!(range_stream
        .global_site_keep
        .iter()
        .enumerate()
        .all(|(row, &keep)| keep == (row < 80 && full_keep[row])));
    for (streamed, &full_index) in range_stream.units.iter().zip(selected_full.iter()) {
        let streamed_source_rows = streamed
            .indices
            .iter()
            .map(|&row| range_stream.packed_source_rows[row])
            .collect::<Vec<_>>();
        let expected_source_rows = full_units[full_index]
            .indices
            .iter()
            .map(|&row| full_source_rows[row])
            .collect::<Vec<_>>();
        assert_eq!(streamed.label, full_units[full_index].label);
        assert_eq!(streamed_source_rows, expected_source_rows);
    }

    drop(full_stream);
    drop(range_stream);
    drop(full_logic);
    let _ = fs::remove_dir_all(fixture_dir);
}

#[test]
fn test_unit_overlaps_scan_bimranges_uses_any_span_overlap() {
    let unit = GarfieldLogicUnit {
        label: "u1".to_string(),
        indices: vec![1, 2, 3],
        spans: vec![
            GarfieldUnitSpan {
                chrom: "10".to_string(),
                bp_start: 111_000_000,
                bp_end: 111_100_000,
            },
            GarfieldUnitSpan {
                chrom: "11".to_string(),
                bp_start: 222_000_000,
                bp_end: 222_100_000,
            },
        ],
    };
    assert!(unit_overlaps_scan_bimranges(
        &unit,
        &[GarfieldScanBimRange {
            chrom: "chr10".to_string(),
            bp_start: 111_050_000,
            bp_end: 111_250_000,
        }]
    ));
    assert!(!unit_overlaps_scan_bimranges(
        &unit,
        &[GarfieldScanBimRange {
            chrom: "12".to_string(),
            bp_start: 111_050_000,
            bp_end: 111_250_000,
        }]
    ));
}

#[test]
fn test_ld_clump_drops_cross_chrom_high_ld_rows() {
    let (bits_flat, row_words) =
        pack_test_binary_rows(&[vec![1u8, 1, 0, 0], vec![1u8, 1, 0, 0], vec![1u8, 0, 1, 0]]);
    let logic_bits = GarfieldLogicBits {
        bits_flat,
        bits_mmap: None,
        bits_mmap_words: 0,
        bits_mmap_owner: None,
        bits_hi_flat: None,
        row_words,
        sample_ids: vec![
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
        ],
        sites: vec![
            GarfieldLogicSite::owned("1", 100, "chr1.s_100", "A", "G", GarfieldLogicSiteMode::Bin),
            GarfieldLogicSite::owned("2", 200, "chr2.s_200", "A", "G", GarfieldLogicSiteMode::Bin),
            GarfieldLogicSite::owned("3", 300, "chr3.s_300", "A", "G", GarfieldLogicSiteMode::Bin),
        ],
        group_ids: vec![0, 1, 2],
        n_samples: 4,
    };
    let kept = prune_candidate_rows_by_ld_priority_with_cache_limit_trace(
        &[0usize, 1, 2],
        &[0usize, 1, 2],
        &logic_bits,
        &[0, 1, 2, 3],
        None,
        None,
    )
    .unwrap()
    .selected_global_rows;
    assert_eq!(kept, vec![0usize, 2usize]);
}

#[test]
fn test_max6_ld_tag_selection_keeps_one_tag_per_ld_redundancy_set() {
    let (bits_flat, row_words) = pack_test_binary_rows(&[
        vec![1u8, 1, 0, 0, 1, 0],
        vec![1u8, 1, 0, 0, 1, 0],
        vec![1u8, 0, 1, 0, 0, 1],
    ]);
    let logic_bits = test_logic_bits_with_rows(bits_flat, row_words, 3, 6);

    let selected = select_max6_ld_tag_rows(
        &[0usize, 1, 2],
        &logic_bits,
        &[0, 1, 2, 3, 4, 5],
        None,
        0.95,
    )
    .unwrap();

    assert_eq!(selected.selected_global_rows, vec![0usize, 2usize]);
    assert_eq!(selected.compressed.len(), 1);
    assert_eq!(selected.compressed[0].kept_global_row, 0);
    assert_eq!(selected.compressed[0].dropped_global_row, 1);
    assert!((selected.compressed[0].r2 - 1.0).abs() < 1e-12);

    let without_trace = select_max6_ld_tag_rows_impl(
        &[0usize, 1, 2],
        &logic_bits,
        &[0, 1, 2, 3, 4, 5],
        None,
        0.95,
        false,
        None,
    )
    .unwrap();
    assert_eq!(
        without_trace.selected_global_rows,
        selected.selected_global_rows
    );
    assert!(without_trace.compressed.is_empty());
}

#[test]
fn max6_exact_tags_respect_sample_subset_groups_and_tail_padding() {
    let (mut bits_flat, row_words) = pack_test_binary_rows(&[
        vec![1u8, 1, 0, 0, 1, 0],
        vec![1u8, 1, 0, 0, 0, 0],
        vec![0u8, 0, 1, 1, 0, 1],
    ]);
    bits_flat[0] |= 1u64 << 63;
    let logic_bits = test_logic_bits_with_rows(bits_flat, row_words, 3, 6);
    let selected =
        select_max6_ld_tag_rows(&[2, 1, 0], &logic_bits, &[0, 1, 2, 3], None, 1.0).unwrap();
    assert_eq!(selected.selected_global_rows, vec![0, 2]);
    assert_eq!(selected.compressed[0].dropped_global_row, 1);
    let grouped = select_max6_ld_tag_rows(
        &[0, 1, 2],
        &logic_bits,
        &[0, 1, 2, 3],
        Some(&[0, 1, 0]),
        1.0,
    )
    .unwrap();
    assert_eq!(grouped.selected_global_rows, vec![0, 1, 2]);
    let full =
        select_max6_ld_tag_rows(&[0, 1, 2], &logic_bits, &[0, 1, 2, 3, 4, 5], None, 1.0).unwrap();
    assert_eq!(full.selected_global_rows, vec![0, 1, 2]);
}

#[test]
fn max6_default_tags_preserve_distinct_high_ld_and_complement_rows() {
    let first = (0..452).map(|i| u8::from(i < 145)).collect::<Vec<_>>();
    let proxy = (0..452)
        .map(|i| u8::from((1..149).contains(&i)))
        .collect::<Vec<_>>();
    let complement = first.iter().map(|&v| 1 - v).collect::<Vec<_>>();
    let (bits_flat, row_words) = pack_test_binary_rows(&[first.clone(), proxy, first, complement]);
    let logic_bits = test_logic_bits_with_rows(bits_flat, row_words, 4, 452);
    let unit = GarfieldLogicUnit {
        label: "recall_regression".to_string(),
        indices: vec![0, 1, 2, 3],
        spans: vec![],
    };
    let samples = (0..452).collect::<Vec<_>>();
    let prepared = prepare_max6_grmmax6_ld_unit(&unit, "window", &logic_bits, &samples).unwrap();
    assert_eq!(prepared.selected_global_rows, vec![0, 1, 3]);
}

#[test]
fn max6_prepared_unit_does_not_retain_unused_ld_trace() {
    let (bits_flat, row_words) = pack_test_binary_rows(&[
        vec![1u8, 1, 0, 0, 1, 0],
        vec![1u8, 1, 0, 0, 1, 0],
        vec![1u8, 0, 1, 0, 0, 1],
    ]);
    let logic_bits = test_logic_bits_with_rows(bits_flat, row_words, 3, 6);
    let unit = GarfieldLogicUnit {
        label: "window".to_string(),
        indices: vec![0, 1, 2],
        spans: vec![GarfieldUnitSpan {
            chrom: "1".to_string(),
            bp_start: 1,
            bp_end: 3,
        }],
    };
    let prepared =
        prepare_max6_grmmax6_ld_unit(&unit, "window", &logic_bits, &[0, 1, 2, 3, 4, 5]).unwrap();
    assert_eq!(prepared.selected_global_rows, vec![0, 2]);
    assert!(prepared.ld_compression.is_empty());
}

#[test]
fn test_max6_ld_tag_selection_never_merges_across_scan_groups() {
    let (bits_flat, row_words) = pack_test_binary_rows(&[
        vec![1u8, 1, 0, 0, 1, 0],
        vec![1u8, 1, 0, 0, 1, 0],
        vec![1u8, 1, 0, 0, 1, 0],
    ]);
    let logic_bits = test_logic_bits_with_rows(bits_flat, row_words, 3, 6);

    let selected = select_max6_ld_tag_rows(
        &[0usize, 1, 2],
        &logic_bits,
        &[0, 1, 2, 3, 4, 5],
        Some(&[0usize, 1, 1usize]),
        0.95,
    )
    .unwrap();

    assert_eq!(selected.selected_global_rows, vec![0usize, 1usize]);
    assert_eq!(selected.compressed.len(), 1);
    assert_eq!(selected.compressed[0].kept_global_row, 1);
    assert_eq!(selected.compressed[0].dropped_global_row, 2);

    let without_trace = select_max6_ld_tag_rows_impl(
        &[0usize, 1, 2],
        &logic_bits,
        &[0, 1, 2, 3, 4, 5],
        Some(&[0usize, 1, 1usize]),
        0.95,
        false,
        None,
    )
    .unwrap();
    assert_eq!(
        without_trace.selected_global_rows,
        selected.selected_global_rows
    );
    assert!(without_trace.compressed.is_empty());
}

#[test]
fn test_ld_exact_bounds_match_direct_r2_grid() {
    for n_samples in 4usize..=16usize {
        for &r2_threshold in &[0.2_f64, 0.5_f64, 0.8_f64] {
            for support_i in 1usize..n_samples {
                for support_j in 1usize..n_samples {
                    if binary_pair_abs_r2_upper_bound_from_support(support_i, support_j, n_samples)
                        < r2_threshold
                    {
                        continue;
                    }
                    let bounds =
                        ld_exact_bounds_from_support(support_i, support_j, n_samples, r2_threshold);
                    let both_min = support_i
                        .saturating_add(support_j)
                        .saturating_sub(n_samples);
                    let both_max = support_i.min(support_j);
                    for both_one in both_min..=both_max {
                        let direct = binary_pair_high_ld_from_support_and_both_one(
                            support_i,
                            support_j,
                            both_one,
                            n_samples,
                            r2_threshold,
                        );
                        let fast = bounds.matches(both_one);
                        assert_eq!(
                                fast, direct,
                                "n_samples={n_samples} r2={r2_threshold} support_i={support_i} support_j={support_j} both_one={both_one} bounds={bounds:?}"
                            );
                    }
                }
            }
        }
    }
}

#[test]
fn test_select_logic_unit_global_rows_applies_ld_prune_to_plain_unit_without_ml() {
    Python::initialize();
    let (bits_flat, row_words) =
        pack_test_binary_rows(&[vec![1u8, 1, 0, 0], vec![1u8, 1, 0, 0], vec![1u8, 0, 1, 0]]);
    let logic_bits = GarfieldLogicBits {
        bits_flat,
        bits_mmap: None,
        bits_mmap_words: 0,
        bits_mmap_owner: None,
        bits_hi_flat: None,
        row_words,
        sample_ids: vec![
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
        ],
        sites: vec![
            GarfieldLogicSite::owned("1", 100, "chr1.s_100", "A", "G", GarfieldLogicSiteMode::Bin),
            GarfieldLogicSite::owned("1", 200, "chr2.s_200", "A", "G", GarfieldLogicSiteMode::Bin),
            GarfieldLogicSite::owned("1", 300, "chr3.s_300", "A", "G", GarfieldLogicSiteMode::Bin),
        ],
        group_ids: vec![0, 1, 2],
        n_samples: 4,
    };
    let unit = GarfieldLogicUnit {
        label: "triad".to_string(),
        indices: vec![0, 1, 2],
        spans: vec![GarfieldUnitSpan {
            chrom: "1".to_string(),
            bp_start: 90,
            bp_end: 310,
        }],
    };
    let selected = select_logic_unit_global_rows(
        &unit,
        "region",
        ResponseKind::Continuous,
        None,
        ImportanceKind::Imp,
        PermutationConfig {
            n_repeats: 1,
            scoring: crate::ml::common::PermutationScoring::Auto,
            seed: 1,
        },
        64,
        0.0,
        ExtraTreesConfig {
            n_estimators: 1,
            max_depth: 1,
            min_samples_leaf: 1,
            min_samples_split: 2,
            bootstrap: false,
            feature_subsample: 0.0,
            seed: 1,
            allow_parallel: false,
        },
        &logic_bits,
        &[0, 1, 2, 3],
        &[0.1, 0.2, 0.3, 0.4],
        None,
        false,
    )
    .unwrap()
    .unwrap();
    assert_eq!(selected.selected_global_rows, vec![0usize, 2usize]);
    assert_eq!(selected.ld_compression.len(), 1);
    assert_eq!(selected.ld_compression[0].kept_global_row, 0);
    assert_eq!(selected.ld_compression[0].dropped_global_row, 1);
    assert!((selected.ld_compression[0].r2 - 1.0).abs() < 1e-12);
    assert!(selected.train_literal_scores.is_none());
}

#[test]
fn test_resolve_simbench_ml_rank_preserves_site_order_and_missing_marks() {
    let contexts = vec![GarfieldUnitMlContext {
        unit_index: 7,
        unit_name: "u7".to_string(),
        selected_global_rows: vec![100, 300, 200],
        ranked_global_rows: vec![100, 300, 200],
        null_unit_group_bin: 0,
    }];
    let site_row_candidates = vec![vec![300, 301], vec![999], vec![200, 201]];
    let (rank_txt, ml_feature_count) = resolve_simbench_ml_rank(
        &[false, false, false],
        site_row_candidates.as_slice(),
        contexts.as_slice(),
    );
    assert_eq!(rank_txt, "2&.&3");
    assert_eq!(ml_feature_count, 3);
}

#[test]
fn test_resolve_simbench_ml_rank_falls_back_to_full_ranking_when_not_selected() {
    let contexts = vec![GarfieldUnitMlContext {
        unit_index: 9,
        unit_name: "u9".to_string(),
        selected_global_rows: vec![100, 200],
        ranked_global_rows: vec![400, 300, 200, 100],
        null_unit_group_bin: 0,
    }];
    let site_row_candidates = vec![vec![300], vec![400]];
    let (rank_txt, ml_feature_count) = resolve_simbench_ml_rank(
        &[false, false],
        site_row_candidates.as_slice(),
        contexts.as_slice(),
    );
    assert_eq!(rank_txt, "2&1");
    assert_eq!(ml_feature_count, 4);
}

#[test]
fn test_parse_simbench_logic_accepts_new_shortcodes() {
    assert_eq!(parse_simbench_logic("a").unwrap(), SimBenchLogic::And);
    assert_eq!(parse_simbench_logic("na").unwrap(), SimBenchLogic::Na);
    assert_eq!(parse_simbench_logic("an").unwrap(), SimBenchLogic::An);
    assert_eq!(parse_simbench_logic("nan").unwrap(), SimBenchLogic::Nan);
    assert_eq!(parse_simbench_logic("x").unwrap(), SimBenchLogic::Xor);
    assert_eq!(parse_simbench_logic("xor").unwrap(), SimBenchLogic::Xor);
}

#[test]
fn test_parse_simbench_terms_accepts_new_logic_shortcodes() {
    let dir = make_temp_dir("simbench_logic_shortcodes");
    let path = dir.join("simbench.tsv");
    fs::write(
        &path,
        concat!(
            "unit_kind\tunit_name\tkind\tsites\teffect\n",
            "gene\tgeneA\ta\t1:10;1:20\t0.1\n",
            "gene\tgeneB\tna\t2:30;2:40\t0.2\n",
            "gene\tgeneC\tan\t3:50;3:60\t0.3\n",
            "gene\tgeneD\tnan\t4:70;4:80\t0.4\n",
            "gene\tgeneE\tx\t5:90;5:100\t0.5\n",
        ),
    )
    .unwrap();
    let terms = parse_simbench_terms(path.to_str().unwrap()).unwrap();
    assert_eq!(terms.len(), 5);
    assert_eq!(terms[0].term_id, 1);
    assert_eq!(terms[0].unit_name, "geneA");
    assert_eq!(terms[0].label, "");
    assert_eq!(terms[0].logic, SimBenchLogic::And);
    assert_eq!(terms[1].logic, SimBenchLogic::Na);
    assert_eq!(terms[2].logic, SimBenchLogic::An);
    assert_eq!(terms[3].logic, SimBenchLogic::Nan);
    assert_eq!(terms[4].logic, SimBenchLogic::Xor);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn test_simbench_rule_from_negations_preserves_negation_semantics() {
    let na_rule = simbench_rule_from_negations(SimBenchLogic::Na, &[true, false]).unwrap();
    assert!(na_rule.first.negated);
    assert_eq!(na_rule.rest.len(), 1);
    assert_eq!(na_rule.rest[0].0, BeamBinaryOp::And);
    assert!(!na_rule.rest[0].1.negated);

    let an_rule = simbench_rule_from_negations(SimBenchLogic::An, &[false, true]).unwrap();
    assert!(!an_rule.first.negated);
    assert_eq!(an_rule.rest.len(), 1);
    assert_eq!(an_rule.rest[0].0, BeamBinaryOp::And);
    assert!(an_rule.rest[0].1.negated);

    let xor_rule =
        simbench_rule_from_negations(SimBenchLogic::Xor, &[false, false, false]).unwrap();
    assert_eq!(xor_rule.rest.len(), 2);
    assert_eq!(xor_rule.rest[0].0, BeamBinaryOp::Xor);
    assert_eq!(xor_rule.rest[1].0, BeamBinaryOp::And);
}

#[test]
fn test_parse_simbench_label_alleles_accepts_target_only_format() {
    let parsed = parse_simbench_label_alleles("8_1[G]&8_2[T]");
    assert_eq!(
        parsed,
        vec![
            SimBenchLabelAlleleSpec::Target("G".to_string()),
            SimBenchLabelAlleleSpec::Target("T".to_string()),
        ]
    );
}

#[test]
fn test_simbench_effective_negations_refalt_label_does_not_flip_minor_coded_rows() {
    let term = SimBenchTerm {
        term_id: 1,
        kind: "an".to_string(),
        logic: SimBenchLogic::An,
        logic_text: "an".to_string(),
        sites_text: "2:4932241;2:4977372".to_string(),
        sites: vec![("2".to_string(), 4_932_241), ("2".to_string(), 4_977_372)],
        unit_name: "Zm00001d002053".to_string(),
        label: "2_4932241[G>T]&!2_4977372[G>A]".to_string(),
    };
    let sites = vec![
        GarfieldLogicSite::owned(
            "2",
            4_932_241,
            "2_4932241",
            "T",
            "G",
            GarfieldLogicSiteMode::Bin,
        ),
        GarfieldLogicSite::owned(
            "2",
            4_977_372,
            "2_4977372",
            "A",
            "G",
            GarfieldLogicSiteMode::Bin,
        ),
    ];
    assert_eq!(
        simbench_effective_negations(&term, sites.as_slice()),
        vec![false, true]
    );
}

#[test]
fn test_simbench_effective_negations_target_label_uses_absolute_target_allele() {
    let term = SimBenchTerm {
        term_id: 1,
        kind: "a".to_string(),
        logic: SimBenchLogic::And,
        logic_text: "a".to_string(),
        sites_text: "2:4932241;2:4977372".to_string(),
        sites: vec![("2".to_string(), 4_932_241), ("2".to_string(), 4_977_372)],
        unit_name: "Zm00001d002053".to_string(),
        label: "2_4932241[G]&2_4977372[A]".to_string(),
    };
    let sites = vec![
        GarfieldLogicSite::owned(
            "2",
            4_932_241,
            "2_4932241",
            "T",
            "G",
            GarfieldLogicSiteMode::Bin,
        ),
        GarfieldLogicSite::owned(
            "2",
            4_977_372,
            "2_4977372",
            "A",
            "G",
            GarfieldLogicSiteMode::Bin,
        ),
    ];
    assert_eq!(
        simbench_effective_negations(&term, sites.as_slice()),
        vec![false, true]
    );
}

#[test]
fn test_format_simbench_ml_rank_marks_negated_members() {
    let ranks = vec![Some(2usize), Some(5usize), None];
    assert_eq!(
        format_simbench_ml_rank(&[true, true, false], &ranks),
        "!2&!5&."
    );
    assert_eq!(
        format_simbench_ml_rank(&[false, true, false], &ranks),
        "2&!5&."
    );
}

#[test]
fn test_cached_bin_singletons_keep_binary_maf_semantics() {
    let candidate_global_rows = vec![11usize];
    let selected_global_rows = vec![11usize];
    let summaries = vec![DosageStage1DualSummary {
        n_ge1: 8usize,
        n_ge2: 0usize,
        sum_ge1: 4.0_f64,
        sum_ge2: 0.0_f64,
    }];

    let binary = build_cached_literal_scores_from_selected_dual_summaries(
        candidate_global_rows.as_slice(),
        selected_global_rows.as_slice(),
        summaries.as_slice(),
        10.0_f64,
        10usize,
        true,
    )
    .unwrap();
    let dual = build_cached_literal_scores_from_selected_dual_summaries(
        candidate_global_rows.as_slice(),
        selected_global_rows.as_slice(),
        summaries.as_slice(),
        10.0_f64,
        10usize,
        false,
    )
    .unwrap();

    assert_eq!(binary.len(), 2);
    assert_eq!(dual.len(), 2);
    assert!((binary[0].train.raw_score - dual[0].train.raw_score).abs() < 1e-12);
    assert!((binary[1].train.raw_score - dual[1].train.raw_score).abs() < 1e-12);
    assert!((binary[0].train.dosage_maf - 0.2_f64).abs() < 1e-12);
    assert!((binary[1].train.dosage_maf - 0.2_f64).abs() < 1e-12);
    assert!((dual[0].train.dosage_maf - 0.4_f64).abs() < 1e-12);
    assert!((dual[1].train.dosage_maf - 0.4_f64).abs() < 1e-12);
}

#[test]
fn test_simbench_raw_normalization_is_allele_orientation_invariant() {
    let mut row_a = vec![0.0_f32, 0.0, 0.0, 1.0, 2.0];
    let mut site_a = SiteInfo {
        chrom: "10".to_string(),
        pos: 111_023_852,
        snp: "site_a".to_string(),
        ref_allele: "C".to_string(),
        alt_allele: "T".to_string(),
    };
    let mut row_b = vec![2.0_f32, 2.0, 2.0, 1.0, 0.0];
    let mut site_b = SiteInfo {
        chrom: "10".to_string(),
        pos: 111_023_852,
        snp: "site_b".to_string(),
        ref_allele: "T".to_string(),
        alt_allele: "C".to_string(),
    };

    normalize_test_simbench_raw_row(&mut row_a, &mut site_a).unwrap();
    normalize_test_simbench_raw_row(&mut row_b, &mut site_b).unwrap();

    let (ge1_a, ge2_a) = pack_test_simbench_raw_row_dual_words(&row_a).unwrap();
    let (ge1_b, ge2_b) = pack_test_simbench_raw_row_dual_words(&row_b).unwrap();
    assert_eq!(site_a.ref_allele, site_b.ref_allele);
    assert_eq!(site_a.alt_allele, site_b.alt_allele);
    assert_eq!(ge1_a, ge1_b);
    assert_eq!(ge2_a, ge2_b);
}

#[test]
fn test_fill_bin_logic_row_bits_fuzzy_respects_row_flip() {
    fn pack_raw_hardcalls(geno: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; geno.len().div_ceil(4)];
        for (i, &g) in geno.iter().enumerate() {
            let code = match g {
                0 => 0b00u8,
                1 => 0b10u8,
                2 => 0b11u8,
                _ => 0b01u8,
            };
            out[i >> 2] |= code << ((i & 3) * 2);
        }
        out
    }

    fn unpack_dual_bits(ge1: &[u64], ge2: &[u64], n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| {
                (((ge1[i >> 6] >> (i & 63)) & 1u64) + ((ge2[i >> 6] >> (i & 63)) & 1u64)) as u8
            })
            .collect()
    }

    let raw_row = vec![2u8, 2, 2, 1, 0];
    let packed_row = pack_raw_hardcalls(raw_row.as_slice());
    let sample_indices = (0usize..raw_row.len()).collect::<Vec<_>>();
    let sample_plan = build_logic_sample_plan(sample_indices.as_slice());
    let mut dst_ge1 = vec![0u64; words_for_samples(raw_row.len())];
    let mut dst_ge2 = vec![0u64; words_for_samples(raw_row.len())];
    fill_bin_logic_row_bits_fuzzy(
        packed_row.as_slice(),
        true,
        sample_plan.as_slice(),
        dst_ge1.as_mut_slice(),
        dst_ge2.as_mut_slice(),
    );

    let mut row = raw_row.iter().map(|&g| g as f32).collect::<Vec<_>>();
    let mut site = SiteInfo {
        chrom: "10".to_string(),
        pos: 111_023_852,
        snp: "fill_flip_site".to_string(),
        ref_allele: "T".to_string(),
        alt_allele: "C".to_string(),
    };
    normalize_test_simbench_raw_row(&mut row, &mut site).unwrap();
    let (exp_ge1, exp_ge2) = pack_test_simbench_raw_row_dual_words(&row).unwrap();

    assert_eq!(
        unpack_dual_bits(dst_ge1.as_slice(), dst_ge2.as_slice(), raw_row.len()),
        unpack_dual_bits(exp_ge1.as_slice(), exp_ge2.as_slice(), raw_row.len())
    );
}

#[test]
fn test_fill_bin_logic_row_bits_one_pass_matches_reference() {
    fn pack_raw_hardcalls(geno: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; geno.len().div_ceil(4)];
        for (i, &g) in geno.iter().enumerate() {
            let code = match g {
                0 => 0b00u8,
                1 => 0b10u8,
                2 => 0b11u8,
                _ => 0b01u8,
            };
            out[i >> 2] |= code << ((i & 3) * 2);
        }
        out
    }

    let raw_row = vec![0u8, 2, 3, 1, 2, 0, 1, 3, 2, 0, 1];
    let packed_row = pack_raw_hardcalls(raw_row.as_slice());
    let sample_indices = vec![10usize, 0, 7, 2, 8, 3, 5, 1, 9, 4, 6];
    let sample_plan = build_logic_sample_plan(sample_indices.as_slice());
    for flip in [false, true] {
        let mut expected = vec![0u64; words_for_samples(sample_indices.len())];
        fill_bin_logic_row_bits(
            packed_row.as_slice(),
            flip,
            sample_plan.as_slice(),
            expected.as_mut_slice(),
        );

        let mut actual = vec![0u64; expected.len()];
        let mut missing = vec![0u64; expected.len()];
        fill_bin_logic_row_bits_one_pass(
            packed_row.as_slice(),
            flip,
            sample_plan.as_slice(),
            actual.as_mut_slice(),
            missing.as_mut_slice(),
        );
        assert_eq!(actual, expected, "flip={flip}");
    }
}

#[test]
fn test_fill_bin_logic_row_bits_identity_matches_sample_plan() {
    fn pack_raw_hardcalls(geno: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; geno.len().div_ceil(4)];
        for (i, &g) in geno.iter().enumerate() {
            let code = match g {
                0 => 0b00u8,
                1 => 0b10u8,
                2 => 0b11u8,
                _ => 0b01u8,
            };
            out[i >> 2] |= code << ((i & 3) * 2);
        }
        out
    }

    let geno = (0usize..77)
        .map(|i| match i % 4 {
            0 => 0u8,
            1 => 1u8,
            2 => 2u8,
            _ => 3u8,
        })
        .collect::<Vec<_>>();
    let packed = pack_raw_hardcalls(geno.as_slice());
    let sample_indices = (0..geno.len()).collect::<Vec<_>>();
    let sample_plan = build_logic_sample_plan(sample_indices.as_slice());
    for flip in [false, true] {
        let mut expected = vec![0u64; words_for_samples(geno.len())];
        fill_bin_logic_row_bits(
            packed.as_slice(),
            flip,
            sample_plan.as_slice(),
            expected.as_mut_slice(),
        );

        let mut actual = vec![0u64; expected.len()];
        fill_bin_logic_row_bits_identity(
            packed.as_slice(),
            geno.len(),
            flip,
            actual.as_mut_slice(),
        );
        assert_eq!(actual, expected, "flip={flip}");

        let mut one_pass = vec![0u64; expected.len()];
        let mut one_pass_missing = vec![0u64; expected.len()];
        fill_bin_logic_row_bits_one_pass(
            packed.as_slice(),
            flip,
            sample_plan.as_slice(),
            one_pass.as_mut_slice(),
            one_pass_missing.as_mut_slice(),
        );
        assert_eq!(one_pass, expected, "one-pass flip={flip}");
    }
}

#[test]
fn test_materialize_rule_support_bits_uses_selected_rows_not_full_prefix() {
    fn pack_dual_flat(rows: &[Vec<f32>]) -> (Vec<u64>, Vec<u64>, usize) {
        let row_words = words_for_samples(rows[0].len());
        let mut ge1_flat = Vec::<u64>::with_capacity(rows.len() * row_words);
        let mut ge2_flat = Vec::<u64>::with_capacity(rows.len() * row_words);
        for row in rows.iter() {
            let (ge1, ge2) = pack_test_simbench_raw_row_dual_words(row.as_slice()).unwrap();
            ge1_flat.extend_from_slice(ge1.as_slice());
            ge2_flat.extend_from_slice(ge2.as_slice());
        }
        (ge1_flat, ge2_flat, row_words)
    }

    fn unpack_dual_bits(ge1: &[u64], ge2: &[u64], n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| {
                (((ge1[i >> 6] >> (i & 63)) & 1u64) + ((ge2[i >> 6] >> (i & 63)) & 1u64)) as u8
            })
            .collect()
    }

    let rows = vec![
        vec![2.0f32, 2.0, 0.0, 0.0],
        vec![0.0f32, 2.0, 0.0, 2.0],
        vec![2.0f32, 2.0, 2.0, 0.0],
    ];
    let (ge1_flat, ge2_flat, row_words) = pack_dual_flat(rows.as_slice());
    let logic_bits = GarfieldLogicBits {
        bits_flat: ge1_flat,
        bits_mmap: None,
        bits_mmap_words: 0,
        bits_mmap_owner: None,
        bits_hi_flat: Some(ge2_flat),
        row_words,
        sample_ids: vec!["s1".into(), "s2".into(), "s3".into(), "s4".into()],
        sites: vec![
            test_site("1", 100),
            test_site("1", 200),
            test_site("1", 300),
        ]
        .into_iter()
        .map(|site| {
            GarfieldLogicSite::owned(
                site.chrom,
                site.pos,
                site.snp,
                site.ref_allele,
                site.alt_allele,
                GarfieldLogicSiteMode::Bin,
            )
        })
        .collect(),
        group_ids: vec![0, 1, 2],
        n_samples: 4,
    };
    let rule = BeamRule {
        first: BeamLiteral {
            row_index: 0,
            group_id: 0,
            negated: false,
        },
        rest: vec![(
            BeamBinaryOp::And,
            BeamLiteral {
                row_index: 1,
                group_id: 1,
                negated: false,
            },
        )],
    };
    let support = materialize_rule_support_bits_from_selected_rows_full(
        &rule,
        &[0usize, 2usize],
        &logic_bits,
        "selected_rows_rule",
    )
    .unwrap();
    let GarfieldRuleSupportBits::Dual { ge1, ge2 } = support else {
        panic!("expected dual support bits");
    };
    assert_eq!(
        unpack_dual_bits(ge1.as_ref(), ge2.as_ref(), 4),
        vec![2u8, 2, 0, 0]
    );
}

#[test]
fn test_insert_topk_density_hit_keeps_highest_scores() {
    let mut bucket = Vec::<(usize, f64)>::new();
    insert_topk_density_hit(&mut bucket, 2, 5.0, 3);
    insert_topk_density_hit(&mut bucket, 1, 9.0, 3);
    insert_topk_density_hit(&mut bucket, 4, 7.0, 3);
    insert_topk_density_hit(&mut bucket, 0, 8.0, 3);
    assert_eq!(bucket.len(), 3);
    assert_eq!(bucket[0], (1, 9.0));
    assert_eq!(bucket[1], (0, 8.0));
    assert_eq!(bucket[2], (4, 7.0));
}

#[test]
fn test_build_unit_window_group_ids_assigns_geneset_spans() {
    let unit = GarfieldLogicUnit {
        label: "triad".to_string(),
        indices: vec![0, 1, 2],
        spans: vec![
            GarfieldUnitSpan {
                chrom: "1".to_string(),
                bp_start: 100,
                bp_end: 200,
            },
            GarfieldUnitSpan {
                chrom: "1".to_string(),
                bp_start: 300,
                bp_end: 400,
            },
            GarfieldUnitSpan {
                chrom: "2".to_string(),
                bp_start: 500,
                bp_end: 600,
            },
        ],
    };
    let sites = vec![
        SiteInfo {
            chrom: "1".to_string(),
            pos: 150,
            snp: "triad_1".to_string(),
            ref_allele: "A".to_string(),
            alt_allele: "G".to_string(),
        },
        SiteInfo {
            chrom: "1".to_string(),
            pos: 350,
            snp: "triad_2".to_string(),
            ref_allele: "A".to_string(),
            alt_allele: "G".to_string(),
        },
        SiteInfo {
            chrom: "2".to_string(),
            pos: 580,
            snp: "triad_3".to_string(),
            ref_allele: "A".to_string(),
            alt_allele: "G".to_string(),
        },
    ];
    let gids = build_unit_window_group_ids(&unit, &[0, 1, 2], sites.as_slice(), "geneset").unwrap();
    assert_eq!(gids, vec![0usize, 1usize, 2usize]);
}

#[test]
fn test_build_unit_window_group_ids_prefers_nearest_overlap_span() {
    let unit = GarfieldLogicUnit {
        label: "overlap".to_string(),
        indices: vec![0],
        spans: vec![
            GarfieldUnitSpan {
                chrom: "1".to_string(),
                bp_start: 100,
                bp_end: 300,
            },
            GarfieldUnitSpan {
                chrom: "1".to_string(),
                bp_start: 220,
                bp_end: 420,
            },
        ],
    };
    let sites = vec![SiteInfo {
        chrom: "1".to_string(),
        pos: 260,
        snp: "overlap_1".to_string(),
        ref_allele: "A".to_string(),
        alt_allele: "G".to_string(),
    }];
    let gids = build_unit_window_group_ids(&unit, &[0], sites.as_slice(), "geneset").unwrap();
    assert_eq!(gids, vec![1usize]);
}

#[test]
fn test_geneset_stage_group_target_uses_surviving_spans() {
    let unit = GarfieldLogicUnit {
        label: "pair".to_string(),
        indices: vec![0, 1],
        spans: vec![
            GarfieldUnitSpan {
                chrom: "2".to_string(),
                bp_start: 100,
                bp_end: 200,
            },
            GarfieldUnitSpan {
                chrom: "8".to_string(),
                bp_start: 300,
                bp_end: 400,
            },
        ],
    };
    assert_eq!(
        geneset_stage_group_target("geneset", &unit, &[0usize, 1usize]),
        Some(2usize)
    );
    assert_eq!(
        geneset_stage_group_target("geneset", &unit, &[1usize, 1usize]),
        Some(1usize)
    );
}

#[test]
fn test_geneset_stage_group_target_ignores_non_span_fallback_groups() {
    let unit = GarfieldLogicUnit {
        label: "pair".to_string(),
        indices: vec![0, 1],
        spans: vec![
            GarfieldUnitSpan {
                chrom: "2".to_string(),
                bp_start: 100,
                bp_end: 200,
            },
            GarfieldUnitSpan {
                chrom: "8".to_string(),
                bp_start: 300,
                bp_end: 400,
            },
        ],
    };
    assert_eq!(
        geneset_stage_group_target("geneset", &unit, &[2usize, 3usize]),
        Some(1usize)
    );
}

#[test]
fn test_beam_params_for_prepared_promotes_multigroup_geneset_depth() {
    let prepared = GarfieldUnitPrepared {
        selected_global_rows: vec![0, 1, 2],
        stage1_candidate_global_rows: None,
        ld_compression: Vec::new(),
        local_groups: vec![0, 1, 2],
        geneset_stage_group_target: Some(3),
        null_unit_group_bin: 0,
        train_literal_scores: None,
    };
    let out = beam_params_for_prepared(
        &prepared,
        BeamSearchParams {
            max_pick: 1,
            ..BeamSearchParams::default()
        },
    );
    assert_eq!(out.max_pick, 3);
    assert_eq!(
        out.group_constraint,
        BeamGroupConstraintMode::ExcludeUntilDistinctGroups(3)
    );
}

#[test]
fn test_beam_params_for_prepared_clamps_multigroup_geneset_depth_to_group_count() {
    let prepared = GarfieldUnitPrepared {
        selected_global_rows: vec![0, 1],
        stage1_candidate_global_rows: None,
        ld_compression: Vec::new(),
        local_groups: vec![0, 1],
        geneset_stage_group_target: Some(2),
        null_unit_group_bin: 0,
        train_literal_scores: None,
    };
    let out = beam_params_for_prepared(
        &prepared,
        BeamSearchParams {
            max_pick: 5,
            ..BeamSearchParams::default()
        },
    );
    assert_eq!(out.max_pick, 2);
    assert_eq!(
        out.group_constraint,
        BeamGroupConstraintMode::ExcludeUntilDistinctGroups(2)
    );
}

#[test]
fn test_take_prefix_per_group_rows_caps_each_group() {
    let picked = take_prefix_per_group_rows(
        &[
            vec![10usize, 11usize, 12usize],
            vec![20usize],
            vec![30usize, 31usize],
        ],
        2,
    );
    assert_eq!(picked, vec![10usize, 11usize, 20usize, 30usize, 31usize]);
}

#[test]
fn test_build_synthetic_geneset_null_units_matches_real_window_buckets() {
    let scan_units = vec![
        GarfieldLogicUnit {
            label: "u1".to_string(),
            indices: vec![0, 1],
            spans: vec![GarfieldUnitSpan {
                chrom: "1".to_string(),
                bp_start: 100,
                bp_end: 200,
            }],
        },
        GarfieldLogicUnit {
            label: "u2".to_string(),
            indices: vec![0, 2, 3],
            spans: vec![
                GarfieldUnitSpan {
                    chrom: "1".to_string(),
                    bp_start: 100,
                    bp_end: 200,
                },
                GarfieldUnitSpan {
                    chrom: "2".to_string(),
                    bp_start: 500,
                    bp_end: 600,
                },
            ],
        },
        GarfieldLogicUnit {
            label: "u3".to_string(),
            indices: vec![1, 3, 4, 5],
            spans: vec![
                GarfieldUnitSpan {
                    chrom: "1".to_string(),
                    bp_start: 100,
                    bp_end: 200,
                },
                GarfieldUnitSpan {
                    chrom: "2".to_string(),
                    bp_start: 500,
                    bp_end: 600,
                },
                GarfieldUnitSpan {
                    chrom: "3".to_string(),
                    bp_start: 880,
                    bp_end: 940,
                },
            ],
        },
    ];
    let null_groups = vec![
        vec![("1".to_string(), 100, 200)],
        vec![("1".to_string(), 280, 340)],
        vec![("2".to_string(), 500, 600)],
        vec![("2".to_string(), 680, 740)],
        vec![("3".to_string(), 880, 940)],
    ];
    let (null_groups, atom_pool_size) = build_synthetic_geneset_null_group_defs(
        scan_units.as_slice(),
        Some(null_groups.as_slice()),
        None,
        9,
        50,
        7,
    );
    assert_eq!(atom_pool_size, 5);
    assert_eq!(null_groups.len(), 9);
    let mut hist = BTreeMap::<usize, usize>::new();
    for group in null_groups.iter() {
        assert!(!group.is_empty());
        for (_, start, end) in group.iter() {
            assert_eq!(i64::from(*end) - i64::from(*start), 100);
        }
        *hist.entry(group.len()).or_insert(0) += 1;
    }
    assert_eq!(hist.get(&1), Some(&3usize));
    assert_eq!(hist.get(&2), Some(&3usize));
    assert_eq!(hist.get(&3), Some(&3usize));
}
