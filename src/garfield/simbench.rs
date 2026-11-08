use super::*;

#[derive(Clone, Debug)]
pub(super) struct GarfieldUnitMlContext {
    pub(super) unit_index: usize,
    pub(super) unit_name: String,
    pub(super) selected_global_rows: Vec<usize>,
    pub(super) ranked_global_rows: Vec<usize>,
    pub(super) null_unit_group_bin: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SimBenchLogic {
    Single,
    And,
    Or,
    Na,
    An,
    Nan,
    Xor,
}

#[derive(Clone, Debug)]
pub(super) struct SimBenchTerm {
    pub(super) term_id: usize,
    pub(super) kind: String,
    pub(super) logic: SimBenchLogic,
    pub(super) logic_text: String,
    pub(super) sites_text: String,
    pub(super) sites: Vec<(String, i32)>,
    pub(super) unit_name: String,
    pub(super) label: String,
}

pub(super) fn parse_simbench_logic(raw: &str) -> Result<SimBenchLogic, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "single" | "s" | "additive" => Ok(SimBenchLogic::Single),
        "and" | "a" => Ok(SimBenchLogic::And),
        "or" => Ok(SimBenchLogic::Or),
        "na" => Ok(SimBenchLogic::Na),
        "an" => Ok(SimBenchLogic::An),
        "nan" => Ok(SimBenchLogic::Nan),
        "x" | "xor" => Ok(SimBenchLogic::Xor),
        other => Err(format!(
            "simbench logic must be one of: single, s, additive, and, or, xor, a, na, an, nan, x; got {other}"
        )),
    }
}

#[inline]
pub(super) fn simbench_logic_member_negated(logic: SimBenchLogic, idx: usize) -> bool {
    match logic {
        SimBenchLogic::Single | SimBenchLogic::And | SimBenchLogic::Or | SimBenchLogic::Xor => {
            false
        }
        SimBenchLogic::Na => idx == 0,
        SimBenchLogic::Nan => true,
        SimBenchLogic::An => idx > 0,
    }
}

#[inline]
pub(super) fn simbench_logic_binary_op(
    logic: SimBenchLogic,
    rest_idx: usize,
) -> Option<BeamBinaryOp> {
    match logic {
        SimBenchLogic::Single => None,
        SimBenchLogic::And | SimBenchLogic::Na | SimBenchLogic::An | SimBenchLogic::Nan => {
            Some(BeamBinaryOp::And)
        }
        SimBenchLogic::Or => Some(BeamBinaryOp::Or),
        SimBenchLogic::Xor if rest_idx == 0 => Some(BeamBinaryOp::Xor),
        SimBenchLogic::Xor => Some(BeamBinaryOp::And),
    }
}

pub(super) fn simbench_logic_display_ops(
    logic: SimBenchLogic,
    n_sites: usize,
) -> Vec<BeamBinaryOp> {
    (0..n_sites.saturating_sub(1))
        .filter_map(|rest_idx| simbench_logic_binary_op(logic, rest_idx))
        .collect()
}

pub(super) fn parse_simbench_sites(raw: &str) -> Result<Vec<(String, i32)>, String> {
    let mut out = Vec::<(String, i32)>::new();
    for token in raw.split(';') {
        let site_txt = token.trim();
        if site_txt.is_empty() {
            continue;
        }
        let Some((chrom, pos_txt)) = site_txt.split_once(':') else {
            return Err(format!(
                "invalid simbench site token '{site_txt}'; expected chrom:pos"
            ));
        };
        let pos = pos_txt.trim().parse::<i32>().map_err(|e| {
            format!("invalid simbench site position '{pos_txt}' in '{site_txt}': {e}")
        })?;
        out.push((chrom.trim().to_string(), pos));
    }
    if out.is_empty() {
        return Err("simbench sites column is empty".to_string());
    }
    Ok(out)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum SimBenchLabelAlleleSpec {
    RefAlt(String, String),
    Target(String),
}

pub(super) fn parse_simbench_label_alleles(label: &str) -> Vec<SimBenchLabelAlleleSpec> {
    let mut out = Vec::<SimBenchLabelAlleleSpec>::new();
    let mut rest = label;
    while let Some(lb) = rest.find('[') {
        let after_lb = &rest[lb + 1..];
        let Some(rb_rel) = after_lb.find(']') else {
            break;
        };
        let inside = &after_lb[..rb_rel];
        if let Some((a0, a1)) = inside.split_once('>') {
            out.push(SimBenchLabelAlleleSpec::RefAlt(
                a0.trim().to_ascii_uppercase(),
                a1.trim().to_ascii_uppercase(),
            ));
        } else {
            let allele = inside.trim().to_ascii_uppercase();
            if !allele.is_empty() {
                out.push(SimBenchLabelAlleleSpec::Target(allele));
            }
        }
        rest = &after_lb[rb_rel + 1..];
    }
    out
}

pub(super) fn parse_simbench_terms(path: &str) -> Result<Vec<SimBenchTerm>, String> {
    let file = File::open(path).map_err(|e| format!("open simbench file {path}: {e}"))?;
    let mut rdr = BufReader::new(file);
    let mut header = String::new();
    let n_read = rdr
        .read_line(&mut header)
        .map_err(|e| format!("read simbench header {path}: {e}"))?;
    if n_read == 0 {
        return Err(format!("simbench file is empty: {path}"));
    }
    let cols = header
        .trim_end_matches(&['\r', '\n'][..])
        .split('\t')
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    let mut col_idx = HashMap::<String, usize>::new();
    for (i, col) in cols.iter().enumerate() {
        col_idx.insert(col.to_ascii_lowercase(), i);
    }
    let term_idx = col_idx.get("term_id").copied();
    let kind_idx = col_idx
        .get("kind")
        .or_else(|| col_idx.get("term_kind"))
        .copied()
        .ok_or_else(|| format!("simbench file {path} is missing required column: kind"))?;
    let logic_idx = col_idx.get("logic").copied();
    let sites_idx = col_idx
        .get("sites")
        .copied()
        .ok_or_else(|| format!("simbench file {path} is missing required column: sites"))?;
    let label_idx = col_idx.get("label").copied();
    let unit_name_idx = col_idx.get("unit_name").copied();
    if label_idx.is_none() && unit_name_idx.is_none() {
        return Err(format!(
            "simbench file {path} is missing required column: label (or unit_name)"
        ));
    }

    let mut out = Vec::<SimBenchTerm>::new();
    for (line_no0, line_res) in rdr.lines().enumerate() {
        let line_no = line_no0 + 2;
        let line = line_res.map_err(|e| format!("read {path}:{line_no}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').map(|s| s.trim()).collect::<Vec<_>>();
        let get_field = |idx: usize| -> &str {
            if idx < fields.len() {
                fields[idx]
            } else {
                ""
            }
        };
        let term_id = term_idx
            .map(get_field)
            .and_then(|txt| txt.parse::<usize>().ok())
            .unwrap_or(out.len() + 1);
        let kind = get_field(kind_idx).to_string();
        let logic_text = logic_idx.map(get_field).unwrap_or("").to_string();
        let logic_source = if logic_text.trim().is_empty() {
            kind.as_str()
        } else {
            logic_text.as_str()
        };
        let logic =
            parse_simbench_logic(logic_source).map_err(|e| format!("{path}:{line_no}: {e}"))?;
        let sites_text = get_field(sites_idx).to_string();
        let sites =
            parse_simbench_sites(&sites_text).map_err(|e| format!("{path}:{line_no}: {e}"))?;
        let unit_name = unit_name_idx.map(get_field).unwrap_or("").to_string();
        let label = if let Some(idx) = label_idx {
            get_field(idx).to_string()
        } else {
            let unit_trimmed = unit_name.trim();
            if unit_trimmed.contains('[') && unit_trimmed.contains(']') {
                unit_trimmed.to_string()
            } else {
                String::new()
            }
        };
        out.push(SimBenchTerm {
            term_id,
            kind,
            logic,
            logic_text,
            sites_text,
            sites,
            unit_name,
            label,
        });
    }
    Ok(out)
}

pub(super) fn simbench_rule_from_negations(
    logic: SimBenchLogic,
    negated: &[bool],
) -> Result<BeamRule, String> {
    let n_sites = negated.len();
    if n_sites == 0 {
        return Err("simbench term has no sites".to_string());
    }
    if matches!(logic, SimBenchLogic::Single) && n_sites != 1 {
        return Err(format!(
            "simbench logic 'single' expects exactly 1 site, got {n_sites}"
        ));
    }
    let first = BeamLiteral {
        row_index: 0,
        group_id: 0,
        negated: negated[0],
    };
    let mut rest = Vec::<(BeamBinaryOp, BeamLiteral)>::with_capacity(n_sites.saturating_sub(1));
    for row_index in 1..n_sites {
        if let Some(op) = simbench_logic_binary_op(logic, row_index - 1) {
            rest.push((
                op,
                BeamLiteral {
                    row_index,
                    group_id: row_index,
                    negated: negated[row_index],
                },
            ));
        }
    }
    Ok(BeamRule { first, rest })
}

pub(super) fn simbench_effective_negations<S: GarfieldDisplaySite>(
    term: &SimBenchTerm,
    sites: &[S],
) -> Vec<bool> {
    let label_alleles = parse_simbench_label_alleles(&term.label);
    (0..sites.len())
        .map(|idx| {
            let base_negated = simbench_logic_member_negated(term.logic, idx);
            let Some(label_spec) = label_alleles.get(idx) else {
                return base_negated;
            };
            let site_ref =
                clean_logic_allele_label(sites[idx].garfield_ref_allele()).to_ascii_uppercase();
            let site_alt =
                clean_logic_allele_label(sites[idx].garfield_alt_allele()).to_ascii_uppercase();
            match label_spec {
                // `[ref>alt]` is descriptive metadata from simulation output, not an
                // instruction to flip the logical literal when GARFIELD has already
                // re-oriented the row to its internal allele1 coding.
                SimBenchLabelAlleleSpec::RefAlt(_, _) => base_negated,
                SimBenchLabelAlleleSpec::Target(target) => {
                    if *target == site_alt {
                        false
                    } else if *target == site_ref {
                        true
                    } else {
                        base_negated
                    }
                }
            }
        })
        .collect()
}

pub(super) fn simbench_rule_name<S: GarfieldChromPosSite + GarfieldDisplaySite>(
    logic: SimBenchLogic,
    negated: &[bool],
    sites: &[S],
) -> String {
    let mut it = sites.iter().enumerate();
    let Some((first_idx, first)) = it.next() else {
        return String::new();
    };
    let mut out = literal_target_name(first, negated.get(first_idx).copied().unwrap_or(false));
    for (idx, site) in it {
        if let Some(op) = simbench_logic_binary_op(logic, idx - 1) {
            out.push_str(logic_symbol_for_display(
                op,
                GarfieldRuleDisplayPolarity::Original,
            ));
        }
        out.push_str(&literal_target_name(
            site,
            negated.get(idx).copied().unwrap_or(false),
        ));
    }
    out
}

pub(super) fn simbench_rule_bim_name<S: GarfieldChromPosSite + GarfieldDisplaySite>(
    logic: SimBenchLogic,
    negated: &[bool],
    sites: &[S],
) -> String {
    simbench_rule_name(logic, negated, sites)
}

pub(super) fn simbench_rule_bim_alleles<S: GarfieldDisplaySite>(
    negated: &[bool],
    sites: &[S],
) -> (String, String) {
    let mut allele0 = Vec::<String>::with_capacity(sites.len());
    let mut allele1 = Vec::<String>::with_capacity(sites.len());
    for (idx, site) in sites.iter().enumerate() {
        let (a0, a1) = literal_bim_alleles(site, negated.get(idx).copied().unwrap_or(false));
        allele0.push(a0);
        allele1.push(a1);
    }
    (allele0.join(","), allele1.join(","))
}

pub(super) fn simbench_rule_expr<S: GarfieldDisplaySite>(
    logic: SimBenchLogic,
    negated: &[bool],
    sites: &[S],
) -> Result<String, String> {
    let mut it = sites.iter();
    let Some(first) = it.next() else {
        return Err("simbench term has no sites".to_string());
    };
    let mut out = literal_expr(first, negated.first().copied().unwrap_or(false));
    if sites.len() > 1 {
        for (idx, site) in it.enumerate() {
            out.push(' ');
            if let Some(op) = simbench_logic_binary_op(logic, idx) {
                out.push_str(expr_symbol_for_display(
                    op,
                    GarfieldRuleDisplayPolarity::Original,
                ));
            }
            out.push(' ');
            out.push_str(&literal_expr(
                site,
                negated.get(idx + 1).copied().unwrap_or(false),
            ));
        }
    }
    Ok(out)
}

#[inline]
pub(super) fn simbench_term_unit_name<S: GarfieldChromPosSite>(
    term: &SimBenchTerm,
    sites: &[S],
) -> String {
    let trimmed = term.unit_name.trim();
    if !trimmed.is_empty() {
        trimmed.to_string()
    } else {
        let trimmed_label = term.label.trim();
        if !trimmed_label.is_empty() {
            trimmed_label.to_string()
        } else {
            sites
                .iter()
                .map(site_base_label)
                .collect::<Vec<_>>()
                .join("&")
        }
    }
}

#[inline]
pub(super) fn simbench_term_rule_name<S: GarfieldChromPosSite + GarfieldDisplaySite>(
    term: &SimBenchTerm,
    negated: &[bool],
    sites: &[S],
) -> String {
    simbench_rule_name(term.logic, negated, sites)
}

#[inline]
pub(super) fn simbench_term_bim_name<S: GarfieldChromPosSite + GarfieldDisplaySite>(
    term: &SimBenchTerm,
    negated: &[bool],
    sites: &[S],
) -> String {
    simbench_rule_bim_name(term.logic, negated, sites)
}

#[inline]

pub(super) fn beam_params_for_simbench_context(
    selected_row_count: usize,
    null_unit_group_bin: u8,
    beam_params: BeamSearchParams,
) -> BeamSearchParams {
    BeamSearchParams {
        null_complexity_bin: rule_null_context_bin(
            rule_null_complexity_bin(selected_row_count),
            null_unit_group_bin,
        ),
        ..beam_params
    }
}

pub(super) fn unit_contains_any_simbench_site(
    unit: &GarfieldLogicUnit,
    terms: &[SimBenchTerm],
) -> bool {
    unit.spans.iter().any(|span| {
        terms.iter().any(|term| {
            term.sites
                .iter()
                .any(|(chrom, pos)| unit_span_contains_site(span, chrom, *pos))
        })
    })
}

pub(super) fn build_simbench_ml_contexts(
    terms: &[SimBenchTerm],
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
    logic_bits: &GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    y_train: &[f64],
    beam_params: BeamSearchParams,
    allow_parallel: bool,
) -> Result<Vec<GarfieldUnitMlContext>, String> {
    if terms.is_empty() || scan_unit_indices.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::<GarfieldUnitMlContext>::new();
    for &ui in scan_unit_indices.iter() {
        let unit = &units[ui];
        if !unit_contains_any_simbench_site(unit, terms) {
            continue;
        }
        if unit.indices.is_empty() {
            continue;
        }
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
            continue;
        };
        let dense_train = dense_dosage_rows_from_full_bits(
            logic_bits.bits(),
            logic_bits.bits_hi_flat.as_deref(),
            logic_bits.row_words,
            unit.indices.as_slice(),
            train_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        if dense_train.is_empty() {
            continue;
        }
        let n_region = dense_train.len();
        let ml_group_ids = build_unit_window_group_ids(
            unit,
            unit.indices.as_slice(),
            logic_bits.sites.as_slice(),
            unit_kind_lc,
        );
        let (_selected_global_rows, ranked_global_rows) = if let Some(engine_one) = engine {
            let keep_k = resolve_ml_keep_k(n_region, ml_top_k, ml_top_frac);
            let top_local = select_ml_top_local_indices(
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
                keep_k,
                GarfieldMlKeepPolicy::TopK,
                tree_cfg.seed ^ 0xB6D5_0C11_8E91_3F27,
                ml_group_ids.as_deref(),
            )?;
            if top_local.is_empty() {
                continue;
            }
            let ranked_local = select_ml_top_only_local_indices(
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
                n_region,
                ml_group_ids.as_deref(),
            )?;
            (
                top_local
                    .iter()
                    .map(|&idx| unit.indices[idx])
                    .collect::<Vec<_>>(),
                ranked_local
                    .iter()
                    .map(|&idx| unit.indices[idx])
                    .collect::<Vec<_>>(),
            )
        } else {
            (unit.indices.clone(), unit.indices.clone())
        };
        out.push(GarfieldUnitMlContext {
            unit_index: ui,
            unit_name: unit.label.clone(),
            selected_global_rows: prepared.selected_global_rows,
            ranked_global_rows,
            null_unit_group_bin: prepared.null_unit_group_bin,
        });
    }
    Ok(out)
}

pub(super) fn format_simbench_ml_rank(negated: &[bool], ranks: &[Option<usize>]) -> String {
    if ranks.is_empty() {
        return ".".to_string();
    }
    let joiner = "&";
    ranks
        .iter()
        .enumerate()
        .map(|(idx, rank)| match rank {
            Some(v) if negated.get(idx).copied().unwrap_or(false) => format!("!{v}"),
            Some(v) => v.to_string(),
            None => ".".to_string(),
        })
        .collect::<Vec<_>>()
        .join(joiner)
}

pub(super) fn match_simbench_ml_context<'a>(
    term: &SimBenchTerm,
    site_row_candidates: &[Vec<usize>],
    contexts: &'a [GarfieldUnitMlContext],
) -> Option<&'a GarfieldUnitMlContext> {
    let unit_name = term.unit_name.trim();
    if !unit_name.is_empty() {
        if let Some(ctx) = contexts.iter().find(|ctx| ctx.unit_name == unit_name) {
            return Some(ctx);
        }
    }
    let mut best = None::<(usize, usize, usize, &'a GarfieldUnitMlContext)>;
    for ctx in contexts.iter() {
        let selected_hits = site_row_candidates
            .iter()
            .filter(|cands| {
                cands
                    .iter()
                    .any(|idx| ctx.selected_global_rows.contains(idx))
            })
            .count();
        let ranked_hits = site_row_candidates
            .iter()
            .filter(|cands| cands.iter().any(|idx| ctx.ranked_global_rows.contains(idx)))
            .count();
        if selected_hits == 0 && ranked_hits == 0 {
            continue;
        }
        let candidate = (
            selected_hits,
            ranked_hits,
            usize::MAX - ctx.selected_global_rows.len(),
            ctx,
        );
        if best
            .as_ref()
            .map(|prev| {
                candidate.0 > prev.0
                    || (candidate.0 == prev.0 && (candidate.1, candidate.2) > (prev.1, prev.2))
            })
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }
    best.map(|(_, _, _, ctx)| ctx)
}

pub(super) fn build_simbench_delta_score_annotation<S: GarfieldDisplaySite>(
    logic: SimBenchLogic,
    negated: &[bool],
    sites: &[S],
    y_test: &[f64],
    bits_test: &[u64],
    bits_test_hi: Option<&[u64]>,
    row_words_test: usize,
    n_rows: usize,
    n_test: usize,
    params: &BeamSearchParams,
) -> Result<String, String> {
    if sites.is_empty() {
        return Ok(String::new());
    }
    let mut out = String::new();
    let mut prefix_rule = singleton_rule_from_literal(BeamLiteral {
        row_index: 0,
        group_id: 0,
        negated: negated.get(0).copied().unwrap_or(false),
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
    out.push_str(&literal_metric_token(first_raw));
    if sites.len() == 1 {
        out.push_str("->");
        out.push_str(&format_delta_metric_value(first_raw));
        return Ok(out);
    }
    let display_ops = simbench_logic_display_ops(logic, sites.len());
    for idx in 1..sites.len() {
        let lit = BeamLiteral {
            row_index: idx,
            group_id: idx,
            negated: negated.get(idx).copied().unwrap_or(false),
        };
        let single_raw = evaluate_rule_test_raw_score(
            &singleton_rule_from_literal(lit),
            y_test,
            bits_test,
            bits_test_hi,
            row_words_test,
            n_rows,
            n_test,
            params,
        )?;
        let op = display_ops
            .get(idx - 1)
            .copied()
            .unwrap_or(BeamBinaryOp::And);
        out.push_str(logic_symbol_for_display(
            op,
            GarfieldRuleDisplayPolarity::Original,
        ));
        out.push_str(&literal_metric_token(single_raw));
        prefix_rule.rest.push((op, lit));
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
        out.push_str("->");
        out.push_str(&format_delta_metric_value(prefix_raw));
    }
    Ok(out)
}

pub(super) fn resolve_simbench_ml_rank(
    negated: &[bool],
    site_row_candidates: &[Vec<usize>],
    contexts: &[GarfieldUnitMlContext],
) -> (String, usize) {
    if site_row_candidates.is_empty() {
        return (".".to_string(), 0);
    }
    let mut best_selected = None::<(usize, usize, usize, usize, String, usize)>;
    let mut best_ranked = None::<(usize, usize, usize, usize, String, usize)>;
    for ctx in contexts.iter() {
        let selected_rank_map = ctx
            .selected_global_rows
            .iter()
            .enumerate()
            .map(|(i, &row)| (row, i + 1))
            .collect::<HashMap<usize, usize>>();
        let selected_ranks = site_row_candidates
            .iter()
            .map(|cands| {
                cands
                    .iter()
                    .filter_map(|idx| selected_rank_map.get(idx).copied())
                    .min()
            })
            .collect::<Vec<_>>();
        let selected_matched = selected_ranks.iter().filter(|v| v.is_some()).count();
        if selected_matched > 0 {
            let sum_rank = selected_ranks.iter().flatten().copied().sum::<usize>();
            let max_rank = selected_ranks
                .iter()
                .flatten()
                .copied()
                .max()
                .unwrap_or(usize::MAX);
            let ml_rank = format_simbench_ml_rank(negated, selected_ranks.as_slice());
            let candidate = (
                usize::MAX - selected_matched,
                max_rank,
                sum_rank,
                ctx.unit_index,
                ml_rank,
                ctx.selected_global_rows.len(),
            );
            if best_selected
                .as_ref()
                .map(|best| candidate < *best)
                .unwrap_or(true)
            {
                best_selected = Some(candidate);
            }
        }

        let ranked_rank_map = ctx
            .ranked_global_rows
            .iter()
            .enumerate()
            .map(|(i, &row)| (row, i + 1))
            .collect::<HashMap<usize, usize>>();
        let ranked_ranks = site_row_candidates
            .iter()
            .map(|cands| {
                cands
                    .iter()
                    .filter_map(|idx| ranked_rank_map.get(idx).copied())
                    .min()
            })
            .collect::<Vec<_>>();
        let ranked_matched = ranked_ranks.iter().filter(|v| v.is_some()).count();
        if ranked_matched == 0 {
            continue;
        }
        let sum_rank = ranked_ranks.iter().flatten().copied().sum::<usize>();
        let max_rank = ranked_ranks
            .iter()
            .flatten()
            .copied()
            .max()
            .unwrap_or(usize::MAX);
        let ml_rank = format_simbench_ml_rank(negated, ranked_ranks.as_slice());
        let candidate = (
            usize::MAX - ranked_matched,
            max_rank,
            sum_rank,
            ctx.unit_index,
            ml_rank,
            ctx.ranked_global_rows.len(),
        );
        if best_ranked
            .as_ref()
            .map(|best| candidate < *best)
            .unwrap_or(true)
        {
            best_ranked = Some(candidate);
        }
    }
    if let Some((_, _, _, _, ml_rank, ml_feature_count)) = best_selected {
        (ml_rank, ml_feature_count)
    } else if let Some((_, _, _, _, ml_rank, ml_feature_count)) = best_ranked {
        (ml_rank, ml_feature_count)
    } else {
        (
            format_simbench_ml_rank(negated, &vec![None; site_row_candidates.len()]),
            0,
        )
    }
}

pub(super) fn evaluate_simbench_terms(
    terms: &[SimBenchTerm],
    logic_bits: &GarfieldLogicBits,
    logic_site_lookup: &HashMap<(String, i32), Vec<usize>>,
    train_idx_local: &[usize],
    assoc_sample_indices: &[usize],
    y_train: &[f64],
    y_assoc: &[f64],
    beam_params: BeamSearchParams,
    output_null_penalties: Option<Arc<RuleNullPenaltyLookup>>,
    ml_contexts: &[GarfieldUnitMlContext],
) -> Result<Vec<GarfieldLogicRuleRecord>, String> {
    if terms.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::<GarfieldLogicRuleRecord>::with_capacity(terms.len());
    for term in terms.iter() {
        let mut logic_row_candidates = Vec::<Vec<usize>>::with_capacity(term.sites.len());
        for (chrom, pos) in term.sites.iter() {
            let key = (normalize_chrom(chrom), *pos);
            let row_candidates = logic_site_lookup.get(&key).cloned().ok_or_else(|| {
                format!(
                    "simbench term {} site not found in logic feature rows: {}:{}",
                    term.term_id, chrom, pos
                )
            })?;
            logic_row_candidates.push(row_candidates);
        }
        let matched_ctx =
            match_simbench_ml_context(term, logic_row_candidates.as_slice(), ml_contexts);
        let selected_row_indices = logic_row_candidates
            .iter()
            .map(|cands| {
                if let Some(ctx) = matched_ctx {
                    if let Some(&idx) = cands
                        .iter()
                        .find(|idx| ctx.selected_global_rows.contains(idx))
                    {
                        return Ok(idx);
                    }
                    if let Some(&idx) = cands
                        .iter()
                        .find(|idx| ctx.ranked_global_rows.contains(idx))
                    {
                        return Ok(idx);
                    }
                }
                cands.first().copied().ok_or_else(|| {
                    format!(
                        "simbench term {} has an empty candidate row set",
                        term.term_id
                    )
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let bench_sites = selected_row_indices
            .iter()
            .map(|&idx| {
                logic_bits.sites.get(idx).cloned().ok_or_else(|| {
                    format!(
                        "simbench term {} logic row out of range: {}",
                        term.term_id, idx
                    )
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let effective_negated = simbench_effective_negations(term, bench_sites.as_slice());
        let rule = simbench_rule_from_negations(term.logic, effective_negated.as_slice()).map_err(
            |e| {
                format!(
                    "simbench term {} ('{}' {} {}) failed to build rule: {e}",
                    term.term_id, term.kind, term.logic_text, term.sites_text
                )
            },
        )?;
        let n_rule_rows = bench_sites.len();
        let (train_ge1, row_words_train) = packed_rows_subset_from_full_bits(
            logic_bits.bits(),
            logic_bits.row_words,
            selected_row_indices.as_slice(),
            train_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        let train_ge2 = if let Some(bits_hi_flat) = logic_bits.bits_hi_flat.as_ref() {
            let (bits_hi, row_words_train_hi) = packed_rows_subset_from_full_bits(
                bits_hi_flat.as_slice(),
                logic_bits.row_words,
                selected_row_indices.as_slice(),
                train_idx_local,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            if row_words_train != row_words_train_hi {
                return Err("simbench train dual row-word mismatch".to_string());
            }
            Some(bits_hi)
        } else {
            None
        };
        let (assoc_ge1, row_words_assoc) = packed_rows_subset_from_full_bits(
            logic_bits.bits(),
            logic_bits.row_words,
            selected_row_indices.as_slice(),
            assoc_sample_indices,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        let assoc_ge2 = if let Some(bits_hi_flat) = logic_bits.bits_hi_flat.as_ref() {
            let (bits_hi, row_words_assoc_hi) = packed_rows_subset_from_full_bits(
                bits_hi_flat.as_slice(),
                logic_bits.row_words,
                selected_row_indices.as_slice(),
                assoc_sample_indices,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            if row_words_assoc != row_words_assoc_hi {
                return Err("simbench assoc dual row-word mismatch".to_string());
            }
            Some(bits_hi)
        } else {
            None
        };
        let context_beam_params = matched_ctx
            .map(|ctx| {
                beam_params_for_simbench_context(
                    ctx.selected_global_rows.len(),
                    ctx.null_unit_group_bin,
                    beam_params.clone(),
                )
            })
            .unwrap_or_else(|| beam_params.clone());
        let _train_sc = if let Some(train_ge2_use) = train_ge2.as_ref() {
            evaluate_rule_continuous_dual(
                &rule,
                y_train,
                train_ge1.as_slice(),
                train_ge2_use.as_slice(),
                row_words_train,
                n_rule_rows,
                train_idx_local.len(),
                context_beam_params.lambda_len,
                context_beam_params.lambda_not,
            )
        } else {
            evaluate_rule_continuous(
                &rule,
                y_train,
                train_ge1.as_slice(),
                row_words_train,
                n_rule_rows,
                train_idx_local.len(),
                context_beam_params.lambda_len,
                context_beam_params.lambda_not,
            )
        }
        .map_err(|e| {
            format!(
                "simbench term {} failed to score train rule: {e}",
                term.term_id
            )
        })?;
        let assoc_sc = if let Some(assoc_ge2_use) = assoc_ge2.as_ref() {
            evaluate_rule_continuous_dual(
                &rule,
                y_assoc,
                assoc_ge1.as_slice(),
                assoc_ge2_use.as_slice(),
                row_words_assoc,
                n_rule_rows,
                assoc_sample_indices.len(),
                context_beam_params.lambda_len,
                context_beam_params.lambda_not,
            )
        } else {
            evaluate_rule_continuous(
                &rule,
                y_assoc,
                assoc_ge1.as_slice(),
                row_words_assoc,
                n_rule_rows,
                assoc_sample_indices.len(),
                context_beam_params.lambda_len,
                context_beam_params.lambda_not,
            )
        }
        .map_err(|e| {
            format!(
                "simbench term {} failed to score assoc rule: {e}",
                term.term_id
            )
        })?;
        let sim_expr_txt = simbench_rule_expr(
            term.logic,
            effective_negated.as_slice(),
            bench_sites.as_slice(),
        )?;
        let test_bucket = bucket_from_rule_with_complexity(
            &rule,
            assoc_sc.dosage_maf,
            context_beam_params.null_complexity_bin,
        );
        let test_score = final_output_score_with_bucket(
            test_bucket,
            assoc_sc.raw_score,
            output_null_penalties.as_deref(),
            false,
        );
        let support_bits = materialize_rule_support_bits_from_selected_rows_full(
            &rule,
            selected_row_indices.as_slice(),
            logic_bits,
            term.label.as_str(),
        )
        .map_err(|e| {
            format!(
                "simbench term {} failed to materialize full support bits: {e}",
                term.term_id
            )
        })?;
        let first_site = bench_sites
            .first()
            .ok_or_else(|| format!("simbench term {} has no sites", term.term_id))?;
        let (ml_rank, ml_feature_count) = resolve_simbench_ml_rank(
            effective_negated.as_slice(),
            logic_row_candidates.as_slice(),
            ml_contexts,
        );
        let (bim_allele0, bim_allele1) =
            simbench_rule_bim_alleles(effective_negated.as_slice(), bench_sites.as_slice());
        let unit_name = simbench_term_unit_name(term, bench_sites.as_slice());
        let snp_name =
            simbench_term_rule_name(term, effective_negated.as_slice(), bench_sites.as_slice());
        let bim_snp_name =
            simbench_term_bim_name(term, effective_negated.as_slice(), bench_sites.as_slice());
        let delta_score = build_simbench_delta_score_annotation(
            term.logic,
            effective_negated.as_slice(),
            bench_sites.as_slice(),
            y_assoc,
            assoc_ge1.as_slice(),
            assoc_ge2.as_deref(),
            row_words_assoc,
            n_rule_rows,
            assoc_sample_indices.len(),
            &context_beam_params,
        )
        .map_err(|e| {
            format!(
                "simbench term {} failed to build delta-score annotation: {e}",
                term.term_id
            )
        })?;
        out.push(GarfieldLogicRuleRecord {
            unit_name,
            unit_kind: "simbench".to_string(),
            unit_index: term.term_id,
            region_size: bench_sites.len(),
            ml_feature_count,
            ml_rank,
            selected_row_indices,
            display_ops: simbench_logic_display_ops(term.logic, bench_sites.len()),
            display_negated: effective_negated,
            snp_name,
            expr: sim_expr_txt,
            chrom_field: first_site.chrom().to_string(),
            bim_snp_name,
            bim_allele0,
            bim_allele1,
            allele1_maf: assoc_sc.dosage_maf,
            pos: first_site.pos,
            score: test_score,
            delta_score,
            perm_pvalue: None,
            perm_fdr: None,
            support_bits: Some(support_bits),
        });
    }
    Ok(out)
}
