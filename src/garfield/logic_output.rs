use super::*;

pub(super) fn singleton_rule_from_literal(lit: BeamLiteral) -> BeamRule {
    BeamRule {
        first: lit,
        rest: Vec::new(),
    }
}

#[inline]
pub(super) fn format_delta_metric_value(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    format!("{value:.4}")
}

#[inline]
pub(super) fn literal_metric_token(value: f64) -> String {
    format_delta_metric_value(value)
}

#[inline]
pub(super) fn logic_symbol_for_display(
    op: BeamBinaryOp,
    polarity: GarfieldRuleDisplayPolarity,
) -> &'static str {
    match (op, polarity) {
        (BeamBinaryOp::Xor, _) => "^",
        (BeamBinaryOp::And, GarfieldRuleDisplayPolarity::Original)
        | (BeamBinaryOp::Or, GarfieldRuleDisplayPolarity::Complement) => "&",
        (BeamBinaryOp::Or, GarfieldRuleDisplayPolarity::Original)
        | (BeamBinaryOp::And, GarfieldRuleDisplayPolarity::Complement) => "|",
    }
}

#[inline]
pub(super) fn expr_symbol_for_display(
    op: BeamBinaryOp,
    polarity: GarfieldRuleDisplayPolarity,
) -> &'static str {
    match (op, polarity) {
        (BeamBinaryOp::Xor, _) => "XOR",
        (BeamBinaryOp::And, GarfieldRuleDisplayPolarity::Original)
        | (BeamBinaryOp::Or, GarfieldRuleDisplayPolarity::Complement) => "AND",
        (BeamBinaryOp::Or, GarfieldRuleDisplayPolarity::Original)
        | (BeamBinaryOp::And, GarfieldRuleDisplayPolarity::Complement) => "OR",
    }
}

#[inline]
pub(super) fn display_literal_negated(
    literal_negated: bool,
    polarity: GarfieldRuleDisplayPolarity,
) -> bool {
    match polarity {
        GarfieldRuleDisplayPolarity::Original => literal_negated,
        GarfieldRuleDisplayPolarity::Complement => !literal_negated,
    }
}

#[inline]
pub(super) fn display_binary_op(
    op: BeamBinaryOp,
    polarity: GarfieldRuleDisplayPolarity,
) -> BeamBinaryOp {
    match polarity {
        GarfieldRuleDisplayPolarity::Original => op,
        GarfieldRuleDisplayPolarity::Complement => match op {
            BeamBinaryOp::And => BeamBinaryOp::Or,
            BeamBinaryOp::Or => BeamBinaryOp::And,
            BeamBinaryOp::Xor => BeamBinaryOp::Xor,
        },
    }
}

pub(super) fn rule_display_negated_with_polarity(
    rule: &BeamRule,
    polarity: GarfieldRuleDisplayPolarity,
) -> Vec<bool> {
    let mut out = Vec::<bool>::with_capacity(rule.len());
    out.push(display_literal_negated(rule.first.negated, polarity));
    out.extend(
        rule.rest
            .iter()
            .map(|(_, lit)| display_literal_negated(lit.negated, polarity)),
    );
    out
}

pub(super) fn rule_display_ops_with_polarity(
    rule: &BeamRule,
    polarity: GarfieldRuleDisplayPolarity,
) -> Vec<BeamBinaryOp> {
    rule.rest
        .iter()
        .map(|(op, _)| display_binary_op(*op, polarity))
        .collect()
}

#[inline]
pub(super) fn complement_bits_in_place(bits: &mut [u64], n_samples: usize) {
    bitnot_masked(bits, n_samples);
}

#[inline]
pub(super) fn complement_dual_bits_in_place(
    ge1_bits: &mut [u64],
    ge2_bits: &mut [u64],
    n_samples: usize,
) {
    let orig_ge1 = ge1_bits.to_vec();
    let orig_ge2 = ge2_bits.to_vec();
    for (dst, &src) in ge1_bits.iter_mut().zip(orig_ge2.iter()) {
        *dst = src;
    }
    for (dst, &src) in ge2_bits.iter_mut().zip(orig_ge1.iter()) {
        *dst = src;
    }
    bitnot_masked(ge1_bits, n_samples);
    bitnot_masked(ge2_bits, n_samples);
}

#[inline]
pub(super) fn preferred_display_polarity(
    rule: &BeamRule,
    full_bits: &[u64],
    n_samples: usize,
) -> GarfieldRuleDisplayPolarity {
    let _ = (rule, full_bits, n_samples);
    // Search is AND/XOR-only and output should preserve the discovered rule
    // family directly instead of rewriting complements into OR-form displays.
    GarfieldRuleDisplayPolarity::Original
}

pub(super) fn rule_expr_with_polarity<S: GarfieldDisplaySite>(
    rule: &BeamRule,
    local_sites: &[S],
    polarity: GarfieldRuleDisplayPolarity,
) -> Result<String, String> {
    let first = local_sites
        .get(rule.first.row_index)
        .ok_or_else(|| format!("rule row index out of range: {}", rule.first.row_index))?;
    let mut out = literal_expr(first, display_literal_negated(rule.first.negated, polarity));
    for (op, lit) in rule.rest.iter() {
        let site = local_sites
            .get(lit.row_index)
            .ok_or_else(|| format!("rule row index out of range: {}", lit.row_index))?;
        out.push(' ');
        out.push_str(expr_symbol_for_display(*op, polarity));
        out.push(' ');
        out.push_str(&literal_expr(
            site,
            display_literal_negated(lit.negated, polarity),
        ));
    }
    Ok(out)
}

pub(super) fn rule_snp_name_with_polarity<S: GarfieldChromPosSite + GarfieldDisplaySite>(
    rule: &BeamRule,
    local_sites: &[S],
    polarity: GarfieldRuleDisplayPolarity,
) -> Result<String, String> {
    let first = local_sites
        .get(rule.first.row_index)
        .ok_or_else(|| format!("rule row index out of range: {}", rule.first.row_index))?;
    let mut out = literal_target_name(first, display_literal_negated(rule.first.negated, polarity));
    for (op, lit) in rule.rest.iter() {
        let site = local_sites
            .get(lit.row_index)
            .ok_or_else(|| format!("rule row index out of range: {}", lit.row_index))?;
        out.push_str(logic_symbol_for_display(*op, polarity));
        out.push_str(&literal_target_name(
            site,
            display_literal_negated(lit.negated, polarity),
        ));
    }
    Ok(out)
}

pub(super) fn rule_bim_name_with_polarity<S: GarfieldChromPosSite + GarfieldDisplaySite>(
    rule: &BeamRule,
    local_sites: &[S],
    polarity: GarfieldRuleDisplayPolarity,
) -> Result<String, String> {
    rule_snp_name_with_polarity(rule, local_sites, polarity)
}

pub(super) fn rule_bim_alleles_with_polarity<S: GarfieldDisplaySite>(
    rule: &BeamRule,
    local_sites: &[S],
    polarity: GarfieldRuleDisplayPolarity,
) -> Result<(String, String), String> {
    let mut allele0 = Vec::<String>::with_capacity(rule.len());
    let mut allele1 = Vec::<String>::with_capacity(rule.len());
    let first = local_sites
        .get(rule.first.row_index)
        .ok_or_else(|| format!("rule row index out of range: {}", rule.first.row_index))?;
    let (a0, a1) =
        literal_bim_alleles(first, display_literal_negated(rule.first.negated, polarity));
    allele0.push(a0);
    allele1.push(a1);
    for (_, lit) in rule.rest.iter() {
        let site = local_sites
            .get(lit.row_index)
            .ok_or_else(|| format!("rule row index out of range: {}", lit.row_index))?;
        let (a0, a1) = literal_bim_alleles(site, display_literal_negated(lit.negated, polarity));
        allele0.push(a0);
        allele1.push(a1);
    }
    Ok((allele0.join(","), allele1.join(",")))
}

pub(super) fn rule_ml_rank_name_with_polarity(
    rule: &BeamRule,
    polarity: GarfieldRuleDisplayPolarity,
) -> String {
    let mut out = literal_ml_rank_name(
        rule.first.row_index,
        display_literal_negated(rule.first.negated, polarity),
    );
    for (op, lit) in rule.rest.iter() {
        out.push_str(logic_symbol_for_display(*op, polarity));
        out.push_str(&literal_ml_rank_name(
            lit.row_index,
            display_literal_negated(lit.negated, polarity),
        ));
    }
    out
}

pub(super) fn build_rule_delta_score_annotation(
    rule: &BeamRule,
    y_test: &[f64],
    bits_test: &[u64],
    bits_test_hi: Option<&[u64]>,
    row_words_test: usize,
    n_rows: usize,
    n_test: usize,
    params: &BeamSearchParams,
    polarity: GarfieldRuleDisplayPolarity,
) -> Result<String, String> {
    let mut score_txt = String::new();
    let mut prefix_rule = singleton_rule_from_literal(BeamLiteral {
        negated: display_literal_negated(rule.first.negated, polarity),
        ..rule.first
    });
    let first_raw = evaluate_rule_test_raw_score(
        &prefix_rule,
        y_test,
        bits_test,
        bits_test_hi,
        row_words_test,
        n_rows,
        n_test,
        params,
    )?;
    score_txt.push_str(&literal_metric_token(first_raw));

    if rule.rest.is_empty() {
        score_txt.push_str("->");
        score_txt.push_str(&format_delta_metric_value(first_raw));
        return Ok(score_txt);
    }

    for (op, lit) in rule.rest.iter() {
        let display_lit = BeamLiteral {
            negated: display_literal_negated(lit.negated, polarity),
            ..*lit
        };
        let literal_raw = evaluate_rule_test_raw_score(
            &singleton_rule_from_literal(display_lit),
            y_test,
            bits_test,
            bits_test_hi,
            row_words_test,
            n_rows,
            n_test,
            params,
        )?;
        score_txt.push_str(logic_symbol_for_display(*op, polarity));
        score_txt.push_str(&literal_metric_token(literal_raw));
        prefix_rule
            .rest
            .push((display_binary_op(*op, polarity), display_lit));
        let prefix_raw = evaluate_rule_test_raw_score(
            &prefix_rule,
            y_test,
            bits_test,
            bits_test_hi,
            row_words_test,
            n_rows,
            n_test,
            params,
        )?;
        score_txt.push_str("->");
        score_txt.push_str(&format_delta_metric_value(prefix_raw));
    }
    Ok(score_txt)
}

#[inline]
pub(super) fn final_output_null_penalty_for_bucket(
    bucket: RuleNullBucket,
    output_null_penalties: Option<&RuleNullPenaltyLookup>,
    is_train: bool,
) -> f64 {
    let Some(lookup) = output_null_penalties else {
        return 0.0;
    };
    if is_train {
        lookup.train_penalty(bucket).unwrap_or(0.0)
    } else {
        lookup.test_penalty(bucket).unwrap_or(0.0)
    }
}

#[inline]
pub(super) fn final_output_score_with_bucket(
    bucket: RuleNullBucket,
    raw_score: f64,
    output_null_penalties: Option<&RuleNullPenaltyLookup>,
    is_train: bool,
) -> f64 {
    // Final reported score uses only raw score minus the bucket null penalty.
    raw_score - final_output_null_penalty_for_bucket(bucket, output_null_penalties, is_train)
}

#[inline]
pub(super) fn final_output_score_for_candidate(
    cand: &BeamRuleCandidate,
    output_null_penalties: Option<&RuleNullPenaltyLookup>,
    is_train: bool,
    null_complexity_bin: u8,
) -> f64 {
    let (rule_score, bucket) = if is_train {
        (
            cand.train.raw_score,
            bucket_from_rule_with_complexity(
                &cand.rule,
                cand.train.dosage_maf,
                null_complexity_bin,
            ),
        )
    } else {
        (
            cand.test.raw_score,
            bucket_from_rule_with_complexity(&cand.rule, cand.test.dosage_maf, null_complexity_bin),
        )
    };
    final_output_score_with_bucket(bucket, rule_score, output_null_penalties, is_train)
}

#[inline]
pub(super) fn best_parent_delta_passes(
    child_raw_score: f64,
    best_parent_raw_score: f64,
    null_max_t: f64,
    rule_len: usize,
) -> bool {
    if rule_len <= 1 {
        return true;
    }
    child_raw_score.is_finite()
        && best_parent_raw_score.is_finite()
        && null_max_t.is_finite()
        && (child_raw_score - best_parent_raw_score) > null_max_t
}

#[inline]
pub(super) fn four_plus_family_max_t_passes(
    rule_len: usize,
    raw_score: f64,
    family_max_t: f64,
) -> bool {
    rule_len < 4 || (raw_score.is_finite() && family_max_t.is_finite() && raw_score > family_max_t)
}

#[inline]
pub(super) fn rule_without_literal_for_output(
    rule: &BeamRule,
    remove_idx: usize,
) -> Option<BeamRule> {
    if rule.len() <= 1 || remove_idx >= rule.len() {
        return None;
    }
    if remove_idx == 0 {
        let (_, first) = *rule.rest.first()?;
        return Some(BeamRule {
            first,
            rest: rule.rest.iter().skip(1).copied().collect(),
        });
    }
    let mut out = BeamRule {
        first: rule.first,
        rest: Vec::with_capacity(rule.rest.len().saturating_sub(1)),
    };
    for (rest_idx, &(op, lit)) in rule.rest.iter().enumerate() {
        if rest_idx + 1 != remove_idx {
            out.rest.push((op, lit));
        }
    }
    Some(out)
}

/// Return the best immediate parent represented in the current beam and its
/// raw test score.  A parent which was not retained by the beam is evaluated
/// directly so the diagnostic/gate statistic does not depend on the search
/// path taken to reach the child.
#[allow(clippy::too_many_arguments)]
pub(super) fn best_parent_raw_for_candidate(
    child: &BeamRuleCandidate,
    beam_hits: &[BeamRuleCandidate],
    y_test: &[f64],
    bits_test: &[u64],
    bits_test_hi: Option<&[u64]>,
    row_words_test: usize,
    n_rows: usize,
    n_test: usize,
    params: &BeamSearchParams,
) -> Result<(Option<usize>, Option<f64>), String> {
    if child.rule.len() <= 1 {
        return Ok((None, None));
    }
    let mut best_idx = None;
    let mut best_raw = f64::NEG_INFINITY;
    for remove_idx in 0..child.rule.len() {
        let Some(parent_rule) = rule_without_literal_for_output(&child.rule, remove_idx) else {
            continue;
        };
        let parent_key = parent_rule.lexical_key();
        if let Some((idx, parent)) = beam_hits
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.rule.lexical_key() == parent_key)
            .max_by(|(_, a), (_, b)| {
                a.test
                    .raw_score
                    .partial_cmp(&b.test.raw_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            if parent.test.raw_score.is_finite() {
                if parent.test.raw_score > best_raw {
                    best_idx = Some(idx);
                    best_raw = parent.test.raw_score;
                }
                continue;
            }
            // A retained parent with a non-finite test score is not a valid
            // baseline.  Re-evaluate the immediate parent directly so the
            // diagnostic and delta gate do not silently lose this statistic.
        }
        let raw = evaluate_rule_test_raw_score(
            &parent_rule,
            y_test,
            bits_test,
            bits_test_hi,
            row_words_test,
            n_rows,
            n_test,
            params,
        )?;
        if raw.is_finite() && raw > best_raw {
            best_idx = None;
            best_raw = raw;
        }
    }
    if best_raw.is_finite() {
        Ok((best_idx, Some(best_raw)))
    } else {
        Ok((None, None))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gated_ranked_hits_with_collapse(
    beam_hits: &[BeamRuleCandidate],
    ranked_hits: &[(usize, f64)],
    output_null_penalties: Option<&RuleNullPenaltyLookup>,
    null_complexity_bin: u8,
    y_test: &[f64],
    bits_test: &[u64],
    bits_test_hi: Option<&[u64]>,
    row_words_test: usize,
    n_rows: usize,
    n_test: usize,
    params: &BeamSearchParams,
) -> Result<Vec<(usize, f64)>, String> {
    let mut out = Vec::with_capacity(ranked_hits.len());
    let mut seen = HashSet::<Vec<(usize, bool, u8)>>::new();
    for &(initial_idx, _) in ranked_hits.iter() {
        let mut current_idx = initial_idx;
        loop {
            let Some(child) = beam_hits.get(current_idx) else {
                break;
            };
            let rule_len = child.rule.len();
            if rule_len <= 1 || output_null_penalties.is_none() {
                let score = final_output_score_for_candidate(
                    child,
                    output_null_penalties,
                    false,
                    null_complexity_bin,
                );
                if seen.insert(child.rule.lexical_key()) {
                    out.push((current_idx, score));
                }
                break;
            }
            let (parent_idx, parent_raw) = best_parent_raw_for_candidate(
                child,
                beam_hits,
                y_test,
                bits_test,
                bits_test_hi,
                row_words_test,
                n_rows,
                n_test,
                params,
            )?;
            let Some(parent_raw) = parent_raw else {
                break;
            };
            let delta_threshold = if rule_len >= 4 {
                output_null_penalties
                    .and_then(|lookup| lookup.delta_four_plus_family_penalty(false))
            } else {
                output_null_penalties.and_then(|lookup| lookup.delta_penalty(rule_len, false))
            };
            // Once an output null lookup exists, a missing delta threshold is
            // not evidence that the increment is safe.  It means that the
            // corresponding null family had no usable calibration samples;
            // keep the candidate in diagnostics but conservatively collapse
            // it for the formal rules table.
            let delta_pass = if output_null_penalties.is_some() {
                delta_threshold
                    .map(|threshold| {
                        best_parent_delta_passes(
                            child.test.raw_score,
                            parent_raw,
                            threshold,
                            rule_len,
                        )
                    })
                    .unwrap_or(false)
            } else {
                true
            };
            let family_threshold = if rule_len >= 4 {
                output_null_penalties.and_then(|lookup| lookup.four_plus_family_penalty(false))
            } else {
                None
            };
            let family_pass = if rule_len >= 4 && output_null_penalties.is_some() {
                family_threshold
                    .map(|threshold| {
                        four_plus_family_max_t_passes(rule_len, child.test.raw_score, threshold)
                    })
                    .unwrap_or(false)
            } else {
                true
            };
            if delta_pass && family_pass {
                let score = final_output_score_for_candidate(
                    child,
                    output_null_penalties,
                    false,
                    null_complexity_bin,
                );
                if seen.insert(child.rule.lexical_key()) {
                    out.push((current_idx, score));
                }
                break;
            }
            let Some(parent_idx) = parent_idx else {
                // The parent was not retained by the beam.  Diagnostics keep
                // the original child, while the formal rules table drops it
                // rather than inventing an unscored replacement.
                break;
            };
            if parent_idx == current_idx {
                break;
            }
            current_idx = parent_idx;
        }
    }
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| cmp_candidate(&beam_hits[a.0], &beam_hits[b.0]))
    });
    Ok(out)
}

#[inline]
pub(super) fn evaluate_rule_test_raw_score(
    rule: &BeamRule,
    y_test: &[f64],
    bits_test: &[u64],
    bits_test_hi: Option<&[u64]>,
    row_words_test: usize,
    n_rows: usize,
    n_test: usize,
    params: &BeamSearchParams,
) -> Result<f64, String> {
    if let Some(bits_test_hi_use) = bits_test_hi {
        Ok(evaluate_rule_continuous_dual(
            rule,
            y_test,
            bits_test,
            bits_test_hi_use,
            row_words_test,
            n_rows,
            n_test,
            params.lambda_len,
            params.lambda_not,
        )?
        .raw_score)
    } else {
        Ok(evaluate_rule_continuous(
            rule,
            y_test,
            bits_test,
            row_words_test,
            n_rows,
            n_test,
            params.lambda_len,
            params.lambda_not,
        )?
        .raw_score)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn render_candidate_rule_for_output(
    cand: &BeamRuleCandidate,
    local_sites: &[GarfieldLogicSite],
    selected_bits_full: &[u64],
    selected_bits_full_hi: Option<&[u64]>,
    logic_bits: &GarfieldLogicBits,
    selected_row_count: usize,
) -> Result<GarfieldRenderedRule, String> {
    let first_site = local_sites
        .get(cand.rule.first.row_index)
        .ok_or_else(|| "beam result first row index out of range".to_string())?;
    let (mut full_bits, mut full_ge2_bits) = if let Some(bits_hi) = selected_bits_full_hi {
        let (ge1, ge2) = materialize_rule_bits_dual(
            &cand.rule,
            selected_bits_full,
            bits_hi,
            logic_bits.row_words,
            selected_row_count,
            logic_bits.n_samples,
        )?;
        (ge1, Some(ge2))
    } else {
        (
            materialize_rule_bits(
                &cand.rule,
                selected_bits_full,
                logic_bits.row_words,
                selected_row_count,
                logic_bits.n_samples,
            )?,
            None,
        )
    };
    let polarity =
        preferred_display_polarity(&cand.rule, full_bits.as_slice(), logic_bits.n_samples);
    if matches!(polarity, GarfieldRuleDisplayPolarity::Complement) {
        if let Some(ge2_bits) = full_ge2_bits.as_mut() {
            complement_dual_bits_in_place(
                full_bits.as_mut_slice(),
                ge2_bits.as_mut_slice(),
                logic_bits.n_samples,
            );
        } else {
            complement_bits_in_place(full_bits.as_mut_slice(), logic_bits.n_samples);
        }
    }
    Ok(GarfieldRenderedRule {
        polarity,
        display_ops: rule_display_ops_with_polarity(&cand.rule, polarity),
        display_negated: rule_display_negated_with_polarity(&cand.rule, polarity),
        snp_name: rule_snp_name_with_polarity(&cand.rule, local_sites, polarity)?,
        expr: rule_expr_with_polarity(&cand.rule, local_sites, polarity)?,
        chrom_field: first_site.chrom().to_string(),
        bim_snp_name: rule_bim_name_with_polarity(&cand.rule, local_sites, polarity)?,
        bim_allele0: rule_bim_alleles_with_polarity(&cand.rule, local_sites, polarity)?.0,
        bim_allele1: rule_bim_alleles_with_polarity(&cand.rule, local_sites, polarity)?.1,
        pos: first_site.pos,
    })
}

pub(super) fn select_reportable_ranked_hits(
    beam_hits: &[BeamRuleCandidate],
    ranked_hits: &[(usize, f64)],
    top_rules_per_unit: usize,
    raw_design: bool,
) -> Vec<(usize, f64)> {
    if ranked_hits.is_empty() {
        return Vec::new();
    }
    // `ranked_hits` is already sorted by the final output score, i.e.
    // raw_score - output_penalty.  Search-time gain and search penalties are
    // applied before this function and must not be substituted here.
    // Raw-design/simulation output intentionally preserves the complete
    // ranking, including non-positive scores.  The production rules table,
    // however, must not promote a non-positive interaction as a discovered
    // rule.  Keep a singleton sentinel for windows where no positive rule
    // survives so the interval remains represented without reporting a
    // spurious higher-order hit.
    if raw_design {
        let base = ranked_hits.to_vec();
        if top_rules_per_unit == 0 || base.len() <= 1 {
            return base;
        }
        let keep = extend_keep_with_score_ties(
            base.as_slice(),
            top_rules_per_unit.min(base.len()),
            |(_, score)| *score,
        );
        return base.into_iter().take(keep).collect::<Vec<_>>();
    }

    let positive = ranked_hits
        .iter()
        .copied()
        .filter(|(_, score)| score.is_finite() && *score > 0.0)
        .collect::<Vec<_>>();
    if positive.is_empty() {
        return ranked_hits
            .iter()
            .copied()
            .find(|(idx, score)| {
                score.is_finite()
                    && beam_hits
                        .get(*idx)
                        .map(|cand| cand.rule.len() == 1)
                        .unwrap_or(false)
            })
            .into_iter()
            .collect();
    }

    let base = positive;
    if top_rules_per_unit == 0 || base.len() <= 1 {
        return base;
    }
    let keep = extend_keep_with_score_ties(
        base.as_slice(),
        top_rules_per_unit.min(base.len()),
        |(_, score)| *score,
    );
    base.into_iter().take(keep).collect::<Vec<_>>()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn maybe_write_beam_debug_probe_tsv(
    probe: Option<&GarfieldBeamDebugProbe>,
    unit: &GarfieldLogicUnit,
    prepared: &GarfieldUnitPrepared,
    beam_hits: &[BeamRuleCandidate],
    local_sites: &[GarfieldLogicSite],
    selected_bits_full: &[u64],
    selected_bits_full_hi: Option<&[u64]>,
    logic_bits: &GarfieldLogicBits,
    output_null_penalties: Option<&RuleNullPenaltyLookup>,
    null_complexity_bin: u8,
) -> Result<(), String> {
    let Some(probe) = probe else {
        return Ok(());
    };
    if probe.unit_name != unit.label {
        return Ok(());
    }
    let mut w = BufWriter::new(File::create(&probe.out_tsv).map_err(|e| e.to_string())?);
    writeln!(
        w,
        "gain_rank\trule_len\tml_rank\tsnp_name\tgain_score\toutput_score\traw_score\tallele1_maf\texpr"
    )
    .map_err(|e| e.to_string())?;
    for (rank_idx, cand) in beam_hits.iter().enumerate() {
        let rendered = render_candidate_rule_for_output(
            cand,
            local_sites,
            selected_bits_full,
            selected_bits_full_hi,
            logic_bits,
            prepared.selected_global_rows.len(),
        )?;
        let output_score = final_output_score_for_candidate(
            cand,
            output_null_penalties,
            false,
            null_complexity_bin,
        );
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{}",
            rank_idx + 1,
            cand.rule.len(),
            rule_ml_rank_name_with_polarity(&cand.rule, rendered.polarity),
            rendered.snp_name,
            cand.test_score,
            output_score,
            cand.test.raw_score,
            cand.test.dosage_maf,
            rendered.expr
        )
        .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}
