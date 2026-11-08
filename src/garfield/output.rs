use super::*;

#[inline]
pub(super) fn plink2bits_from_logic_g(g: i8) -> u8 {
    match g {
        0 => 0b00,
        1 => 0b10,
        2 => 0b11,
        _ => 0b01,
    }
}

#[derive(Clone, Debug)]
pub(super) struct GarfieldPseudoLiteralExport {
    chrom: String,
    pos: i32,
    snp_name: String,
    allele0: String,
    allele1: String,
    selected_row_index: usize,
    display_negated: bool,
}

pub(super) fn collect_logic_pseudo_literal_exports(
    rec: &GarfieldLogicRuleRecord,
    logic_bits: &GarfieldLogicBits,
) -> Result<Vec<GarfieldPseudoLiteralExport>, String> {
    let expected = rec.selected_row_indices.len();
    if rec.display_negated.len() != expected {
        return Err(format!(
            "GARFIELD pseudo export negation mismatch for {}: rows={}, negated={}",
            rec.snp_name,
            expected,
            rec.display_negated.len()
        ));
    }
    let mut out = Vec::<GarfieldPseudoLiteralExport>::with_capacity(expected);
    for idx in 0..expected {
        let row_idx = rec.selected_row_indices[idx];
        let site = logic_bits
            .sites
            .get(row_idx)
            .ok_or_else(|| format!("GARFIELD pseudo export row index out of range: {row_idx}"))?;
        let negated = rec.display_negated[idx];
        let snp_name = literal_target_name(site, negated);
        let (allele0, allele1) = literal_bim_alleles(site, negated);
        out.push(GarfieldPseudoLiteralExport {
            chrom: site.chrom().to_string(),
            pos: site.pos,
            snp_name,
            allele0,
            allele1,
            selected_row_index: row_idx,
            display_negated: negated,
        });
    }
    Ok(out)
}

#[inline]
pub(super) fn write_logic_bits_as_plink_row<W: Write>(
    w: &mut W,
    row_buf: &mut [u8],
    bits: &[u64],
    n_samples: usize,
    complement: bool,
) -> Result<(), String> {
    row_buf.fill(0u8);
    for s in 0..n_samples {
        let mut hit = ((bits[s >> 6] >> (s & 63)) & 1u64) != 0;
        if complement {
            hit = !hit;
        }
        let g = if hit { 2i8 } else { 0i8 };
        row_buf[s >> 2] |= plink2bits_from_logic_g(g) << ((s & 3) * 2);
    }
    w.write_all(row_buf).map_err(|e| e.to_string())
}

#[inline]
pub(super) fn write_logic_dual_bits_as_plink_row<W: Write>(
    w: &mut W,
    row_buf: &mut [u8],
    ge1_bits: &[u64],
    ge2_bits: &[u64],
    n_samples: usize,
    complement: bool,
) -> Result<(), String> {
    row_buf.fill(0u8);
    for s in 0..n_samples {
        let ge1 = ((ge1_bits[s >> 6] >> (s & 63)) & 1u64) as i8;
        let ge2 = ((ge2_bits[s >> 6] >> (s & 63)) & 1u64) as i8;
        let mut g = ge1 + ge2;
        if complement {
            g = 2 - g;
        }
        row_buf[s >> 2] |= plink2bits_from_logic_g(g) << ((s & 3) * 2);
    }
    w.write_all(row_buf).map_err(|e| e.to_string())
}

pub(super) fn write_logic_pseudo_plink(
    prefix: &str,
    sample_ids: &[String],
    records: &[GarfieldLogicRuleRecord],
    logic_bits: &GarfieldLogicBits,
) -> Result<(), String> {
    let bed_path = format!("{prefix}.bed");
    let bim_path = format!("{prefix}.bim");
    let fam_path = format!("{prefix}.fam");
    let mut fam = BufWriter::new(File::create(&fam_path).map_err(|e| e.to_string())?);
    for sid in sample_ids.iter() {
        writeln!(fam, "{0}\t{0}\t0\t0\t1\t-9", sid).map_err(|e| e.to_string())?;
    }
    fam.flush().map_err(|e| e.to_string())?;

    let mut bed = BufWriter::with_capacity(
        8 * 1024 * 1024,
        File::create(&bed_path).map_err(|e| e.to_string())?,
    );
    bed.write_all(&[0x6C, 0x1B, 0x01])
        .map_err(|e| e.to_string())?;
    let mut bim = BufWriter::with_capacity(
        4 * 1024 * 1024,
        File::create(&bim_path).map_err(|e| e.to_string())?,
    );

    let bytes_per_snp = sample_ids.len().div_ceil(4);
    let mut row_buf = vec![0u8; bytes_per_snp];
    let mut ordered = records.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| cmp_logic_rule_records_plink_order(a, b));
    let mut emitted_singletons = HashSet::<(String, i32, String)>::new();
    for rec in ordered.into_iter() {
        let support = materialize_logic_rule_record_support_bits(rec, logic_bits)?;
        let literals = collect_logic_pseudo_literal_exports(rec, logic_bits)?;
        if literals.len() <= 1 {
            let singleton_key = (rec.chrom_field.clone(), rec.pos, rec.bim_snp_name.clone());
            if emitted_singletons.insert(singleton_key) {
                writeln!(
                    bim,
                    "{}\t{}\t0\t{}\t{}\t{}",
                    rec.chrom_field, rec.bim_snp_name, rec.pos, rec.bim_allele0, rec.bim_allele1
                )
                .map_err(|e| e.to_string())?;
                match &support {
                    GarfieldRuleSupportBits::Binary(bits) => {
                        write_logic_bits_as_plink_row(
                            &mut bed,
                            row_buf.as_mut_slice(),
                            bits.as_ref(),
                            sample_ids.len(),
                            false,
                        )?;
                    }
                    GarfieldRuleSupportBits::Dual { ge1, ge2 } => {
                        write_logic_dual_bits_as_plink_row(
                            &mut bed,
                            row_buf.as_mut_slice(),
                            ge1.as_ref(),
                            ge2.as_ref(),
                            sample_ids.len(),
                            false,
                        )?;
                    }
                }
            }
            continue;
        }

        for lit in literals.iter() {
            let singleton_key = (lit.chrom.clone(), lit.pos, lit.snp_name.clone());
            if emitted_singletons.insert(singleton_key) {
                writeln!(
                    bim,
                    "{}\t{}\t0\t{}\t{}\t{}",
                    lit.chrom, lit.snp_name, lit.pos, lit.allele0, lit.allele1
                )
                .map_err(|e| e.to_string())?;
                let row_start = lit.selected_row_index.saturating_mul(logic_bits.row_words);
                let row_end = row_start.saturating_add(logic_bits.row_words);
                let bits = logic_bits.bits().get(row_start..row_end).ok_or_else(|| {
                    format!(
                        "GARFIELD pseudo export literal row slice out of range: row={} row_words={}",
                        lit.selected_row_index, logic_bits.row_words
                    )
                })?;
                if let Some(bits_hi_flat) = logic_bits.bits_hi_flat.as_ref() {
                    let bits_hi = bits_hi_flat.get(row_start..row_end).ok_or_else(|| {
                        format!(
                            "GARFIELD pseudo export literal high-bit row slice out of range: row={} row_words={}",
                            lit.selected_row_index, logic_bits.row_words
                        )
                    })?;
                    write_logic_dual_bits_as_plink_row(
                        &mut bed,
                        row_buf.as_mut_slice(),
                        bits,
                        bits_hi,
                        sample_ids.len(),
                        lit.display_negated,
                    )?;
                } else {
                    write_logic_bits_as_plink_row(
                        &mut bed,
                        row_buf.as_mut_slice(),
                        bits,
                        sample_ids.len(),
                        lit.display_negated,
                    )?;
                }
            }
        }

        for lit in literals.iter() {
            writeln!(
                bim,
                "{}\t{}\t0\t{}\t{}\t{}",
                lit.chrom, rec.bim_snp_name, lit.pos, rec.bim_allele0, rec.bim_allele1
            )
            .map_err(|e| e.to_string())?;
            match &support {
                GarfieldRuleSupportBits::Binary(bits) => {
                    write_logic_bits_as_plink_row(
                        &mut bed,
                        row_buf.as_mut_slice(),
                        bits.as_ref(),
                        sample_ids.len(),
                        false,
                    )?;
                }
                GarfieldRuleSupportBits::Dual { ge1, ge2 } => {
                    write_logic_dual_bits_as_plink_row(
                        &mut bed,
                        row_buf.as_mut_slice(),
                        ge1.as_ref(),
                        ge2.as_ref(),
                        sample_ids.len(),
                        false,
                    )?;
                }
            }
        }
    }
    bim.flush().map_err(|e| e.to_string())?;
    bed.flush().map_err(|e| e.to_string())?;
    Ok(())
}

#[inline]
pub(super) fn normalize_sci_text(text: &str) -> String {
    let Some((mantissa_raw, exp_raw)) = text.split_once('e') else {
        return text.to_string();
    };
    let mantissa = mantissa_raw.trim_end_matches('0').trim_end_matches('.');
    let mantissa = if mantissa.is_empty() || mantissa == "-0" || mantissa == "+0" {
        "0"
    } else {
        mantissa
    };
    let exp = exp_raw.parse::<i32>().unwrap_or(0);
    format!("{mantissa}e{exp}")
}

#[inline]
pub(super) fn format_optional_sci4(value: Option<f64>) -> String {
    match value {
        Some(v) if v.is_finite() => normalize_sci_text(format!("{v:.4e}").as_str()),
        _ => String::new(),
    }
}

#[inline]
pub(super) fn format_optional_score(value: Option<f64>) -> String {
    value
        .filter(|score| score.is_finite())
        .map(|score| format!("{score:.6}"))
        .unwrap_or_else(|| "NA".to_string())
}

#[inline]
pub(super) fn format_max6_score_value(value: f64) -> String {
    if !value.is_finite() {
        return "NA".to_string();
    }
    if value.abs() < 0.0001 {
        format!("{value:.4e}")
    } else {
        format!("{value:.4}")
    }
}

pub(super) fn attach_logic_rule_permutation_fdr(records: &mut [GarfieldLogicRuleRecord]) {
    let mut ranked = records
        .iter()
        .enumerate()
        .filter_map(|(idx, rec)| {
            rec.perm_pvalue
                .filter(|p| p.is_finite())
                .map(|p| (idx, p.clamp(f64::MIN_POSITIVE, 1.0)))
        })
        .collect::<Vec<_>>();
    if ranked.is_empty() {
        return;
    }
    ranked.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let m = ranked.len();
    let mut adjusted = vec![1.0_f64; m];
    let mut running = 1.0_f64;
    for rev_idx in (0..m).rev() {
        let rank = rev_idx + 1;
        let candidate = (ranked[rev_idx].1 * (m as f64) / (rank as f64)).clamp(0.0, 1.0);
        running = running.min(candidate);
        adjusted[rev_idx] = running;
    }
    for ((rec_idx, _), fdr) in ranked.into_iter().zip(adjusted.into_iter()) {
        records[rec_idx].perm_fdr = Some(fdr);
    }
}

pub(super) fn maybe_attach_logic_rule_permutation_significance(
    records: &mut [GarfieldLogicRuleRecord],
    output_null_penalties: Option<&RuleNullPenaltyLookup>,
) {
    let Some(lookup) = output_null_penalties else {
        return;
    };
    for rec in records.iter_mut() {
        rec.perm_pvalue = lookup.test_score_pvalue_greater(rec.score);
        rec.perm_fdr = None;
    }
    attach_logic_rule_permutation_fdr(records);
}

pub(super) fn write_logic_rules_tsv(
    path: &str,
    records: &[GarfieldLogicRuleRecord],
) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    let mut ordered = records.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| cmp_logic_rule_records_output(a, b));
    let has_perm_sig = ordered
        .iter()
        .any(|rec| rec.perm_pvalue.is_some() || rec.perm_fdr.is_some());
    if has_perm_sig {
        writeln!(
            w,
            "unit_name\tregion_size\tml_feature_count\tMLrank\tsnp_name\tallele1_maf\tscore\tdelta_score\tperm_pvalue\tperm_fdr"
        )
        .map_err(|e| e.to_string())?;
    } else {
        writeln!(
            w,
            "unit_name\tregion_size\tml_feature_count\tMLrank\tsnp_name\tallele1_maf\tscore\tdelta_score"
        )
        .map_err(|e| e.to_string())?;
    }
    for rec in ordered.into_iter() {
        if has_perm_sig {
            writeln!(
                w,
                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{}",
                rec.unit_name,
                rec.region_size,
                rec.ml_feature_count,
                rec.ml_rank,
                rec.snp_name,
                rec.allele1_maf,
                rec.score,
                rec.delta_score,
                format_optional_sci4(rec.perm_pvalue),
                format_optional_sci4(rec.perm_fdr),
            )
            .map_err(|e| e.to_string())?;
        } else {
            writeln!(
                w,
                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}",
                rec.unit_name,
                rec.region_size,
                rec.ml_feature_count,
                rec.ml_rank,
                rec.snp_name,
                rec.allele1_maf,
                rec.score,
                rec.delta_score,
            )
            .map_err(|e| e.to_string())?;
        }
    }
    w.flush().map_err(|e| e.to_string())
}

pub(super) const MAX6_GRMMAX6_SCREEN_TOPK: usize = 5_000;
pub(super) const MAX6_GRMMAX6_PERMUTATION_PAIRS: usize = 1_024;
pub(super) const MAX6_GRMMAX6_IO_BUFFER_BYTES: usize = 8 * 1024 * 1024;
pub(super) const MAX6_GRMMAX6_BONFERRONI_ALPHA: f64 = 0.05;
pub(super) const MAX6_GRMMAX6_BONFERRONI_CHANNELS: usize = 6;

#[inline]

pub(super) fn write_max6_screen_null_maxima(path: &str, maxima: &[f64]) -> Result<(), String> {
    let mut writer = BufWriter::new(File::create(path).map_err(|error| error.to_string())?);
    writeln!(writer, "repeat\tmax_screen_score\tfinite").map_err(|e| e.to_string())?;
    for (repeat, &score) in maxima.iter().enumerate() {
        if score.is_finite() {
            writeln!(writer, "{repeat}\t{score:.17}\ttrue").map_err(|e| e.to_string())?;
        } else {
            writeln!(writer, "{repeat}\t\tfalse").map_err(|e| e.to_string())?;
        }
    }
    writer.flush().map_err(|error| error.to_string())
}

pub(super) fn write_max6_screen_validation_maxima(
    path: &str,
    validation: &Max6ScreenValidation,
    threshold: f64,
) -> Result<(), String> {
    let mut writer = BufWriter::new(File::create(path).map_err(|error| error.to_string())?);
    writeln!(writer, "repeat\tmax_screen_score\tthreshold\trejected")
        .map_err(|error| error.to_string())?;
    for (repeat, &score) in validation.maxima.iter().enumerate() {
        writeln!(
            writer,
            "{repeat}\t{score:.17}\t{threshold:.17}\t{}",
            score > threshold
        )
        .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

pub(super) const MAX6_GRMMAX6_PAIRS_HEADER: &str =
    "snp1\tsnp2\tchrom1\tpos1\tchrom2\tpos2\tGRM_MAX6\tbest_channel\t4cell\tmincell\tpair_PVE\tpwald";
pub(super) const MAX6_GRM_INT_PAIRS_HEADER: &str =
    "snp1\tsnp2\tchrom1\tpos1\tchrom2\tpos2\tmax6_rank\tmax6_score\tbest_channel\t4cell\tmincell\tgrm_int_chisq\tpwald";
pub(super) const MAX6_GRM_INT_PROVENANCE_HEADER: &str =
    "output_row_index\tunit_index_zero_based\tunit_label";

#[inline]
pub(super) fn max6_grmmax6_pairs_output_path(prefix: &str) -> String {
    format!("{prefix}.tsv")
}

#[inline]
pub(super) fn max6_grm_int_provenance_output_path(prefix: &str) -> String {
    format!("{prefix}.max6_grmmax6.provenance.tsv")
}

pub(super) fn remove_stale_max6_grm_int_provenance(
    prefix: &str,
    grm_int_only: bool,
) -> Result<(), String> {
    if grm_int_only {
        return Ok(());
    }
    let path = max6_grm_int_provenance_output_path(prefix);
    if Path::new(path.as_str()).exists() {
        fs::remove_file(path.as_str()).map_err(|error| {
            format!("failed to remove stale Max6 GRM-INT provenance file '{path}': {error}")
        })?;
    }
    Ok(())
}

pub(super) fn validate_grm_int_dense_projection(
    n_samples: usize,
    p_c: &[f64],
) -> Result<(), String> {
    let expected = n_samples
        .checked_mul(n_samples)
        .ok_or_else(|| "GRM-INT P_C shape overflow".to_string())?;
    if p_c.len() != expected {
        return Err(format!(
            "GRM-INT P_C shape mismatch: got {} values, expected {expected}",
            p_c.len()
        ));
    }
    if p_c.iter().any(|value| !value.is_finite()) {
        return Err("GRM-INT P_C values must be finite".to_string());
    }
    Ok(())
}

pub(super) fn max6_threshold_method_for_output(
    grm_int_only: bool,
    bonferroni_q_cut: Option<f64>,
    screen_threshold_method: Option<&str>,
    has_pair_output: bool,
) -> String {
    if grm_int_only {
        return "grm_int_nominal_chi2_1_after_selection".to_string();
    }
    if bonferroni_q_cut.is_some() && screen_threshold_method.is_none() {
        return "chi2_1_isf_alpha_over_6N_unique_pairs".to_string();
    }
    screen_threshold_method
        .map(str::to_string)
        .unwrap_or_else(|| {
            if has_pair_output {
                "empirical_order_statistic".to_string()
            } else {
                "none".to_string()
            }
        })
}

pub(super) fn rule_null_generation_for_output(
    grm_int_only: bool,
    has_screen_calibration: bool,
    has_max6_pair_output: bool,
    exact_lmm: bool,
    lm: bool,
) -> Option<&'static str> {
    if grm_int_only {
        None
    } else if has_screen_calibration {
        Some("lmm_parametric_bootstrap_max6_screen")
    } else if has_max6_pair_output {
        Some("global_pair_score_permutation_fixed_pc")
    } else if exact_lmm {
        Some("lmm_parametric_bootstrap")
    } else if lm {
        Some("lm_parametric_bootstrap")
    } else {
        Some("pcg_residual_permutation")
    }
}

pub(super) fn escape_tsv_field(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

pub(super) fn write_max6_grm_int_provenance_spool_record(
    writer: &mut BufWriter<File>,
    output_row_index: usize,
    unit_index: usize,
    unit_label: &str,
) -> Result<(), String> {
    writeln!(
        writer,
        "{output_row_index}\t{unit_index}\t{}",
        escape_tsv_field(unit_label)
    )
    .map_err(|error| error.to_string())
}

pub(super) fn write_max6_grm_int_provenance_tsv(
    path: &str,
    spool_path: &str,
) -> Result<(), String> {
    let mut writer = BufWriter::with_capacity(
        MAX6_GRMMAX6_IO_BUFFER_BYTES,
        File::create(path).map_err(|error| error.to_string())?,
    );
    writeln!(writer, "{MAX6_GRM_INT_PROVENANCE_HEADER}").map_err(|error| error.to_string())?;
    let reader = BufReader::with_capacity(
        MAX6_GRMMAX6_IO_BUFFER_BYTES,
        File::open(spool_path).map_err(|error| error.to_string())?,
    );
    for line in reader.lines() {
        writeln!(writer, "{}", line.map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

#[inline]
pub(super) fn max6_chi2_reference_pvalue(score: f64) -> Option<f64> {
    (score.is_finite() && score >= 0.0).then(|| libm::erfc((0.5 * score).sqrt()))
}

#[inline]
pub(super) fn format_max6_pvalue(pvalue: Option<f64>) -> String {
    match pvalue.filter(|p| p.is_finite() && (0.0..=1.0).contains(p)) {
        Some(p) => format!("{p:.4e}"),
        None => "NA".to_string(),
    }
}

pub(super) fn max6_output_channel_label(
    channel: &str,
    first_site: &GarfieldLogicSite,
    second_site: &GarfieldLogicSite,
) -> String {
    let first_key = (
        chrom_sort_key(first_site.chrom()),
        first_site.pos,
        first_site.snp(),
    );
    let second_key = (
        chrom_sort_key(second_site.chrom()),
        second_site.pos,
        second_site.snp(),
    );
    let (left_site, right_site, first_is_left) = if first_key <= second_key {
        (first_site, second_site, true)
    } else {
        (second_site, first_site, false)
    };
    let haplotype = |first: usize, second: usize| {
        let (left, right) = if first_is_left {
            (first, second)
        } else {
            (second, first)
        };
        let left_allele = if left == 0 {
            left_site.ref_allele()
        } else {
            left_site.alt_allele()
        };
        let right_allele = if right == 0 {
            right_site.ref_allele()
        } else {
            right_site.alt_allele()
        };
        format!("{left_allele}{right_allele}")
    };
    match channel {
        "AND00" => format!("AND({})", haplotype(0, 0)),
        "AND01" => format!("AND({})", haplotype(0, 1)),
        "AND10" => format!("AND({})", haplotype(1, 0)),
        "AND11" => format!("AND({})", haplotype(1, 1)),
        "XOR" => {
            let mut haplotypes = [haplotype(0, 1), haplotype(1, 0)];
            haplotypes.sort();
            format!("XOR({},{})", haplotypes[0], haplotypes[1])
        }
        "DOSAGE_PRODUCT" => "Dosage".to_string(),
        _ => channel.to_string(),
    }
}

#[inline]
pub(super) fn max6_four_cell_counts_in_physical_order(
    counts: [usize; 4],
    first_site: &GarfieldLogicSite,
    second_site: &GarfieldLogicSite,
) -> [usize; 4] {
    let first_key = (
        chrom_sort_key(first_site.chrom()),
        first_site.pos,
        first_site.snp(),
    );
    let second_key = (
        chrom_sort_key(second_site.chrom()),
        second_site.pos,
        second_site.snp(),
    );
    if first_key <= second_key {
        counts
    } else {
        [counts[0], counts[2], counts[1], counts[3]]
    }
}

/// Count the four genotype cells in physical SNP order, using the same
/// allele-oriented and imputed binary rows used by the Max6 search.
/// Cell order is [00, 01, 10, 11], where 0/1 are the rows' displayed
/// ref/alt states after row flipping.
pub(super) fn max6_pair_four_cell_counts(
    logic_bits: &GarfieldLogicBits,
    first_global_row: usize,
    second_global_row: usize,
    sample_indices: &[usize],
) -> Result<[usize; 4], String> {
    let first_site = logic_bits
        .sites
        .get(first_global_row)
        .ok_or_else(|| format!("Max6 4cell first site index out of range: {first_global_row}"))?;
    let second_site = logic_bits
        .sites
        .get(second_global_row)
        .ok_or_else(|| format!("Max6 4cell second site index out of range: {second_global_row}"))?;
    if sample_indices.is_empty() {
        return Err("Max6 4cell requires at least one analysis sample".to_string());
    }
    if sample_indices
        .iter()
        .any(|&sample| sample >= logic_bits.n_samples)
    {
        return Err("Max6 4cell sample index is out of range".to_string());
    }

    let first_key = (
        chrom_sort_key(first_site.chrom()),
        first_site.pos,
        first_site.snp(),
    );
    let second_key = (
        chrom_sort_key(second_site.chrom()),
        second_site.pos,
        second_site.snp(),
    );
    let (left_row, right_row) = if first_key <= second_key {
        (first_global_row, second_global_row)
    } else {
        (second_global_row, first_global_row)
    };

    let needed_words = words_for_samples(logic_bits.n_samples);
    if logic_bits.row_words < needed_words {
        return Err(format!(
            "Max6 4cell row_words={} is smaller than required {needed_words}",
            logic_bits.row_words
        ));
    }
    let row_range = |row: usize| -> Result<std::ops::Range<usize>, String> {
        let start = row
            .checked_mul(logic_bits.row_words)
            .ok_or_else(|| format!("Max6 4cell row offset overflow for row {row}"))?;
        let end = start
            .checked_add(needed_words)
            .ok_or_else(|| format!("Max6 4cell row end overflow for row {row}"))?;
        if end > logic_bits.bits().len() {
            return Err(format!(
                "Max6 4cell row {row} requires words [{start}, {end}), but only {} packed words are available",
                logic_bits.bits().len()
            ));
        }
        Ok(start..end)
    };
    let left_range = row_range(left_row)?;
    let right_range = row_range(right_row)?;
    let bits = logic_bits.bits();
    let mut counts = [0usize; 4];
    if sample_indices_are_full_identity(sample_indices, logic_bits.n_samples) {
        let tail = tail_mask(logic_bits.n_samples).unwrap_or(u64::MAX);
        for word_index in 0..needed_words {
            let valid = if word_index + 1 == needed_words {
                tail
            } else {
                u64::MAX
            };
            let left = bits[left_range.start + word_index] & valid;
            let right = bits[right_range.start + word_index] & valid;
            counts[0] += ((!left & !right) & valid).count_ones() as usize;
            counts[1] += ((!left & right) & valid).count_ones() as usize;
            counts[2] += ((left & !right) & valid).count_ones() as usize;
            counts[3] += ((left & right) & valid).count_ones() as usize;
        }
    } else {
        for &sample in sample_indices {
            let word_index = sample / 64;
            let bit = 1u64 << (sample % 64);
            let left = bits[left_range.start + word_index] & bit != 0;
            let right = bits[right_range.start + word_index] & bit != 0;
            counts[(left as usize) * 2 + right as usize] += 1;
        }
    }
    if counts.iter().sum::<usize>() != sample_indices.len() {
        return Err(format!(
            "Max6 4cell count total {} does not match analysis sample count {}",
            counts.iter().sum::<usize>(),
            sample_indices.len()
        ));
    }
    Ok(counts)
}

pub(super) fn format_max6_four_cell_counts(
    counts: [usize; 4],
    first_site: &GarfieldLogicSite,
    second_site: &GarfieldLogicSite,
) -> String {
    let first_key = (
        chrom_sort_key(first_site.chrom()),
        first_site.pos,
        first_site.snp(),
    );
    let second_key = (
        chrom_sort_key(second_site.chrom()),
        second_site.pos,
        second_site.snp(),
    );
    let (left_site, right_site) = if first_key <= second_key {
        (first_site, second_site)
    } else {
        (second_site, first_site)
    };
    let left0 = left_site.ref_allele();
    let left1 = left_site.alt_allele();
    let right0 = right_site.ref_allele();
    let right1 = right_site.alt_allele();
    format!(
        "{left0}{right0}:{};{left0}{right1}:{};{left1}{right0}:{};{left1}{right1}:{}",
        counts[0], counts[1], counts[2], counts[3]
    )
}

pub(super) fn max6_bonferroni_cutoff(pair_count: usize) -> Result<(f64, f64), String> {
    if pair_count == 0 {
        return Err("Max6 Bonferroni cutoff requires a nonempty unique-pair universe".to_string());
    }
    let p_cut = MAX6_GRMMAX6_BONFERRONI_ALPHA
        / (MAX6_GRMMAX6_BONFERRONI_CHANNELS as f64 * pair_count as f64);
    // Refine the shared inverse-normal approximation against the same exact
    // chi-square(1) survival function used for the output p-value column.
    let initial = crate::linalg::chi2_stat_df1_from_sf(p_cut);
    let mut lo = 0.0;
    let mut hi = initial.max(1.0);
    while max6_chi2_reference_pvalue(hi).is_some_and(|p| p > p_cut) {
        hi *= 2.0;
    }
    for _ in 0..80 {
        let mid = lo + 0.5 * (hi - lo);
        if max6_chi2_reference_pvalue(mid).is_some_and(|p| p > p_cut) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let q_cut = hi;
    if !(q_cut.is_finite() && q_cut >= 0.0) {
        return Err(format!(
            "Max6 Bonferroni cutoff is invalid for N={pair_count}"
        ));
    }
    Ok((q_cut, p_cut))
}

#[inline]
pub(super) fn pair_pve_from_scaled_score(scaled_score: f64, scaled_total_quadratic: f64) -> f64 {
    // The numerator is the winning-channel score after the same 1/sigma_g^2
    // scale used for GRM_MAX6.  Dividing by the correspondingly scaled
    // phenotype quadratic form gives the channel's GRM-metric partial-R2.
    if scaled_score.is_finite()
        && scaled_score >= 0.0
        && scaled_total_quadratic.is_finite()
        && scaled_total_quadratic > 0.0
    {
        scaled_score / scaled_total_quadratic
    } else {
        f64::NAN
    }
}

pub(super) fn write_max6_grmmax6_pair_spool_record(
    writer: &mut BufWriter<File>,
    record: &Max6GrmMax6PairOutput,
    sites: &[GarfieldLogicSite],
    four_cell_counts: [usize; 4],
    grm_int_only: bool,
) -> Result<(), String> {
    let first_site = sites.get(record.first_global_row).ok_or_else(|| {
        format!(
            "Max6 output first site index out of range: {}",
            record.first_global_row
        )
    })?;
    let second_site = sites.get(record.second_global_row).ok_or_else(|| {
        format!(
            "Max6 output second site index out of range: {}",
            record.second_global_row
        )
    })?;
    let best_channel = max6_output_channel_label(record.best_channel, first_site, second_site);
    let four_cell = format_max6_four_cell_counts(four_cell_counts, first_site, second_site);
    let min_cell = four_cell_counts.iter().min().copied().unwrap_or(0);
    if grm_int_only {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            first_site.snp(),
            second_site.snp(),
            first_site.chrom(),
            first_site.pos,
            second_site.chrom(),
            second_site.pos,
            record.candidate_rank,
            format_max6_score_value(record.max6),
            best_channel,
            four_cell,
            min_cell,
            format_max6_score_value(record.dosage),
            format_max6_pvalue(max6_chi2_reference_pvalue(record.dosage)),
        )
        .map_err(|e| e.to_string())
    } else {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            first_site.snp(),
            second_site.snp(),
            first_site.chrom(),
            first_site.pos,
            second_site.chrom(),
            second_site.pos,
            format_max6_score_value(record.max6),
            best_channel,
            four_cell,
            min_cell,
            format_max6_score_value(record.pair_pve),
            format_max6_pvalue(max6_chi2_reference_pvalue(record.max6)),
        )
        .map_err(|e| e.to_string())
    }
}

pub(super) fn write_max6_grmmax6_pairs_tsv(
    path: &str,
    spool_path: &str,
    grm_int_only: bool,
) -> Result<(), String> {
    let mut writer = BufWriter::with_capacity(
        MAX6_GRMMAX6_IO_BUFFER_BYTES,
        File::create(path).map_err(|e| e.to_string())?,
    );
    let header = if grm_int_only {
        MAX6_GRM_INT_PAIRS_HEADER
    } else {
        MAX6_GRMMAX6_PAIRS_HEADER
    };
    writeln!(writer, "{header}").map_err(|e| e.to_string())?;
    let reader = BufReader::with_capacity(
        MAX6_GRMMAX6_IO_BUFFER_BYTES,
        File::open(spool_path).map_err(|e| e.to_string())?,
    );
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        writeln!(writer, "{line}").map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|e| e.to_string())
}

pub(super) fn write_garfield_recall_diagnostics(
    path: &str,
    records: &[GarfieldRecallDiagnosticRecord],
) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    writeln!(
        w,
        "unit_name\tunit_index\tregion_size\tstage\trank\tn_rows\tglobal_rows\tsnp_name\texpr\traw_score\toutput_score\tbest_parent_raw\tbest_parent_delta\tformal_eligible\treported"
    )
    .map_err(|e| e.to_string())?;
    let mut ordered = records.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        a.unit_index
            .cmp(&b.unit_index)
            .then_with(|| a.stage.cmp(b.stage))
            .then_with(|| a.rank.cmp(&b.rank))
    });
    for rec in ordered {
        let raw = rec
            .raw_score
            .map(|v| format!("{v:.10}"))
            .unwrap_or_default();
        let output = rec
            .output_score
            .map(|v| format!("{v:.10}"))
            .unwrap_or_default();
        let best_parent_raw = rec
            .best_parent_raw
            .map(|v| format!("{v:.10}"))
            .unwrap_or_default();
        let best_parent_delta = rec
            .best_parent_delta
            .map(|v| format!("{v:.10}"))
            .unwrap_or_default();
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            rec.unit_name,
            rec.unit_index,
            rec.region_size,
            rec.stage,
            rec.rank,
            rec.n_rows,
            diagnostic_global_row_ids(rec.global_rows.as_slice()),
            rec.snp_name,
            rec.expr,
            raw,
            output,
            best_parent_raw,
            best_parent_delta,
            rec.formal_eligible,
            rec.reported,
        )
        .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}

pub(super) fn write_garfield_frontier_diagnostics(
    path: &str,
    records: &[GarfieldFrontierDiagnosticRecord],
) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    writeln!(
        w,
        "unit_name\tunit_index\tregion_size\tdepth\tfrontier_rank\tfrontier_rank_before\tfrontier_rank_after\tparent_rule\tchild_rule\traw_score\tsearch_score\tbeam_cutoff_score\tretained\tsupport\tsupport_current_parent\tsupport_any_parent\tn_feasible_parents\trescued_by_alt_parent\tbest_feasible_parent\tbest_parent_delta\tglobal_rows"
    )
    .map_err(|e| e.to_string())?;
    let mut ordered = records.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        a.unit_index
            .cmp(&b.unit_index)
            .then_with(|| a.depth.cmp(&b.depth))
            .then_with(|| a.frontier_rank.cmp(&b.frontier_rank))
    });
    for rec in ordered {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.10}\t{:.10}\t{:.10}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.10}\t{}",
            rec.unit_name,
            rec.unit_index,
            rec.region_size,
            rec.depth,
            rec.frontier_rank,
            rec.frontier_rank_before,
            rec.frontier_rank_after,
            rec.parent_rule,
            rec.child_rule,
            rec.raw_score,
            rec.search_score,
            rec.beam_cutoff_score,
            rec.retained,
            rec.support,
            rec.support_current_parent,
            rec.support_any_parent,
            rec.n_feasible_parents,
            rec.rescued_by_alt_parent,
            rec.best_feasible_parent,
            rec.best_parent_delta,
            diagnostic_global_row_ids(rec.global_rows.as_slice()),
        )
        .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}

pub(super) fn write_rule_structure_prior_json(
    path: &str,
    prior: &RuleStructurePrior,
) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    let len_probs = prior.len_probs();
    let prior_cfg = prior.config();
    let prior_len_alpha = prior_cfg.len_alpha_array();
    let expected_len = (1..=5usize)
        .map(|rule_len| (rule_len as f64) * len_probs[rule_len])
        .sum::<f64>();
    writeln!(w, "{{").map_err(|e| e.to_string())?;
    writeln!(w, "  \"not_control\": \"null_penalty_only\",").map_err(|e| e.to_string())?;
    writeln!(w, "  \"prior\": {{").map_err(|e| e.to_string())?;
    writeln!(w, "    \"target_ess\": {:.10},", prior_cfg.target_ess())
        .map_err(|e| e.to_string())?;
    writeln!(w, "    \"len_temper\": {:.10},", prior_cfg.len_temper())
        .map_err(|e| e.to_string())?;
    writeln!(w, "    \"len_alpha\": {{").map_err(|e| e.to_string())?;
    for rule_len in 1..=5usize {
        let suffix = if rule_len < 5 { "," } else { "" };
        writeln!(
            w,
            "      \"{}\": {:.10}{}",
            rule_len, prior_len_alpha[rule_len], suffix
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(w, "    }}").map_err(|e| e.to_string())?;
    writeln!(w, "  }},").map_err(|e| e.to_string())?;
    writeln!(w, "  \"strength\": {:.10},", prior.strength()).map_err(|e| e.to_string())?;
    writeln!(w, "  \"best_log_mass\": {:.10},", prior.best_log_mass())
        .map_err(|e| e.to_string())?;
    writeln!(w, "  \"expected_rule_len\": {:.10},", expected_len).map_err(|e| e.to_string())?;
    writeln!(w, "  \"len_probs\": {{").map_err(|e| e.to_string())?;
    for rule_len in 1..=5usize {
        let suffix = if rule_len < 5 { "," } else { "" };
        writeln!(
            w,
            "    \"{}\": {:.10}{}",
            rule_len,
            prior.len_prob(rule_len),
            suffix
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(w, "  }},").map_err(|e| e.to_string())?;
    writeln!(w, "  \"rows\": [").map_err(|e| e.to_string())?;
    let total_rows = 5usize;
    let mut row_idx = 0usize;
    for rule_len in 1..=5usize {
        row_idx += 1;
        let suffix = if row_idx < total_rows { "," } else { "" };
        writeln!(
            w,
            "    {{\"rule_len\": {}, \"prior_len_alpha\": {:.10}, \"len_prob\": {:.10}, \"prior_mass\": {:.10}, \"prior_log_mass\": {:.10}, \"penalty\": {:.10}, \"expected_rule_len\": {:.10}, \"strength\": {:.10}, \"best_log_mass\": {:.10}}}{}",
            rule_len,
            prior_len_alpha[rule_len],
            prior.len_prob(rule_len),
            prior.mass(rule_len),
            prior.log_mass(rule_len),
            prior.penalty(rule_len, 0),
            expected_len,
            prior.strength(),
            prior.best_log_mass(),
            suffix,
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(w, "  ]").map_err(|e| e.to_string())?;
    writeln!(w, "}}").map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())
}
