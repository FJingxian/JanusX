use super::*;

#[derive(Clone, Debug)]
pub(super) struct GarfieldLogicBits {
    pub(super) bits_flat: Vec<u64>,
    pub(super) bits_mmap: Option<Arc<memmap2::Mmap>>,
    pub(super) bits_mmap_words: usize,
    pub(super) bits_mmap_owner: Option<Arc<Max6BitsTempFileOwner>>,
    pub(super) bits_hi_flat: Option<Vec<u64>>,
    pub(super) row_words: usize,
    pub(super) sample_ids: Vec<String>,
    pub(super) sites: Vec<GarfieldLogicSite>,
    pub(super) group_ids: Vec<usize>,
    pub(super) n_samples: usize,
}

impl GarfieldLogicBits {
    #[inline]
    pub(super) fn bits(&self) -> &[u64] {
        if let Some(mmap) = self.bits_mmap.as_ref() {
            // The backing file is created with a u64-aligned offset and a
            // length that is a multiple of eight bytes.
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const u64, self.bits_mmap_words) }
        } else {
            self.bits_flat.as_slice()
        }
    }
}

impl Drop for GarfieldLogicBits {
    fn drop(&mut self) {
        // Close this handle to the mapping before the final owner removes its
        // backing file. This ordering is required on Windows.
        self.bits_mmap.take();
        self.bits_mmap_owner.take();
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct GarfieldLdCompressionRecord {
    pub(super) kept_global_row: usize,
    pub(super) dropped_global_row: usize,
    pub(super) r2: f64,
}

#[derive(Clone, Debug, Default)]
pub(super) struct GarfieldLdSelection {
    pub(super) selected_global_rows: Vec<usize>,
    pub(super) compressed: Vec<GarfieldLdCompressionRecord>,
}

#[derive(Clone, Debug)]
pub(super) struct GarfieldUnitPrepared {
    pub(super) selected_global_rows: Vec<usize>,
    pub(super) stage1_candidate_global_rows: Option<Vec<usize>>,
    pub(super) ld_compression: Vec<GarfieldLdCompressionRecord>,
    pub(super) local_groups: Vec<usize>,
    pub(super) geneset_stage_group_target: Option<usize>,
    pub(super) null_unit_group_bin: u8,
    pub(super) train_literal_scores: Option<Vec<LiteralSingletonScore>>,
}

#[derive(Clone, Debug)]
pub(super) struct GarfieldSelectedRows {
    pub(super) selected_global_rows: Vec<usize>,
    pub(super) stage1_candidate_global_rows: Option<Vec<usize>>,
    pub(super) ld_compression: Vec<GarfieldLdCompressionRecord>,
    pub(super) train_literal_scores: Option<Vec<LiteralSingletonScore>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DosageStage1DualSummary {
    pub(super) n_ge1: usize,
    pub(super) n_ge2: usize,
    pub(super) sum_ge1: f64,
    pub(super) sum_ge2: f64,
}

#[derive(Clone, Debug)]
pub(super) struct GarfieldUnitBitMatrices<'a> {
    pub(super) train_bits: Cow<'a, [u64]>,
    pub(super) train_bits_hi: Option<Cow<'a, [u64]>>,
    pub(super) row_words_train: usize,
    pub(super) test_bits: Option<Cow<'a, [u64]>>,
    pub(super) test_bits_hi: Option<Cow<'a, [u64]>>,
    pub(super) row_words_test: usize,
    pub(super) selected_bits_full: Option<Cow<'a, [u64]>>,
    pub(super) selected_bits_full_hi: Option<Cow<'a, [u64]>>,
    pub(super) selected_bits_full_alias_train: bool,
    pub(super) selected_bits_full_hi_alias_train: bool,
}

impl GarfieldUnitBitMatrices<'_> {
    #[inline]
    pub(super) fn train_bits(&self) -> &[u64] {
        self.train_bits.as_ref()
    }

    #[inline]
    pub(super) fn train_bits_hi(&self) -> Option<&[u64]> {
        self.train_bits_hi.as_deref()
    }

    #[inline]
    pub(super) fn test_bits(&self) -> &[u64] {
        self.test_bits.as_deref().unwrap_or(self.train_bits())
    }

    #[inline]
    pub(super) fn test_bits_hi(&self) -> Option<&[u64]> {
        self.test_bits_hi.as_deref().or(self.train_bits_hi())
    }

    #[inline]
    pub(super) fn has_fuzzy_bin(&self) -> bool {
        self.train_bits_hi.is_some()
    }

    #[inline]
    pub(super) fn selected_bits_full(&self) -> Option<&[u64]> {
        self.selected_bits_full.as_deref().or_else(|| {
            if self.selected_bits_full_alias_train {
                Some(self.train_bits())
            } else {
                None
            }
        })
    }

    #[inline]
    pub(super) fn selected_bits_full_hi(&self) -> Option<&[u64]> {
        self.selected_bits_full_hi.as_deref().or_else(|| {
            if self.selected_bits_full_hi_alias_train {
                self.train_bits_hi()
            } else {
                None
            }
        })
    }
}

#[inline]
pub(super) fn pair_triple_rule_from_indices(
    indices: &[usize],
    polarity: u8,
    group_ids: &[usize],
) -> Result<BeamRule, String> {
    pair_triple_rule_from_indices_with_op(indices, polarity, group_ids, BeamBinaryOp::And)
}

#[inline]
pub(super) fn pair_triple_rule_from_indices_with_op(
    indices: &[usize],
    polarity: u8,
    group_ids: &[usize],
    op: BeamBinaryOp,
) -> Result<BeamRule, String> {
    if indices.is_empty() {
        return Err("pair-triple backend cannot build an empty rule".to_string());
    }
    let first_idx = *indices
        .first()
        .ok_or_else(|| "pair-triple backend missing first rule index".to_string())?;
    let first_group = *group_ids.get(first_idx).ok_or_else(|| {
        format!(
            "pair-triple backend first row {} is outside group_ids={}",
            first_idx,
            group_ids.len()
        )
    })?;
    let first = BeamLiteral {
        row_index: first_idx,
        group_id: first_group,
        negated: polarity & 1 != 0,
    };
    let mut rest = Vec::with_capacity(indices.len().saturating_sub(1));
    for (offset, &row_index) in indices.iter().enumerate().skip(1) {
        let group_id = *group_ids.get(row_index).ok_or_else(|| {
            format!(
                "pair-triple backend row {} is outside group_ids={}",
                row_index,
                group_ids.len()
            )
        })?;
        rest.push((
            op,
            BeamLiteral {
                row_index,
                group_id,
                negated: polarity & (1u8 << offset) != 0,
            },
        ));
    }
    Ok(BeamRule { first, rest })
}

#[inline]
pub(super) fn pair_triple_rule_from_candidate(
    candidate: &AllPairCandidate,
    group_ids: &[usize],
) -> Result<BeamRule, String> {
    let op = match candidate.gate {
        AllPairGate::And => BeamBinaryOp::And,
        AllPairGate::Xor => {
            if candidate.polarity != 0 {
                return Err("pair-triple backend received a non-canonical XOR polarity".to_string());
            }
            BeamBinaryOp::Xor
        }
    };
    pair_triple_rule_from_indices_with_op(
        &[candidate.first, candidate.second],
        candidate.polarity,
        group_ids,
        op,
    )
}

#[inline]
pub(super) fn pair_triple_candidate_from_rule(
    rule: BeamRule,
    y_train: &[f64],
    bits_train: &[u64],
    row_words_train: usize,
    y_test: &[f64],
    bits_test: &[u64],
    row_words_test: usize,
    n_rows: usize,
    params: &BeamSearchParams,
) -> Result<Option<BeamRuleCandidate>, String> {
    let train = evaluate_rule_continuous(
        &rule,
        y_train,
        bits_train,
        row_words_train,
        n_rows,
        y_train.len(),
        params.lambda_len,
        params.lambda_not,
    )?;
    if !train.raw_score.is_finite()
        || (params.maf_threshold > 0.0 && train.dosage_maf < params.maf_threshold)
    {
        return Ok(None);
    }
    let test = evaluate_rule_continuous(
        &rule,
        y_test,
        bits_test,
        row_words_test,
        n_rows,
        y_test.len(),
        params.lambda_len,
        params.lambda_not,
    )?;
    if !test.raw_score.is_finite() {
        return Ok(None);
    }
    // The scanners rank by exact raw centered gain.  Keep that raw score in
    // the BeamRuleCandidate fields so the existing output/gate layer can
    // apply null penalties and best-parent deltas without changing its
    // semantics.
    Ok(Some(BeamRuleCandidate {
        rule,
        train_score: train.raw_score,
        test_score: test.raw_score,
        train,
        test,
    }))
}

pub(super) fn pair_triple_search_train_test_continuous(
    y_train: &[f64],
    prepared_bits: &GarfieldUnitBitMatrices<'_>,
    n_rows: usize,
    y_test: &[f64],
    group_ids: &[usize],
    params: BeamSearchParams,
    literal_scores: Option<&[LiteralSingletonScore]>,
) -> Result<Vec<BeamRuleCandidate>, String> {
    if prepared_bits.has_fuzzy_bin() {
        return Err(
            "GARFIELD pair-triple backend currently supports only binary 0/2 BIN rows; "
                .to_string()
                + "use the legacy backend for fuzzy/dosage rows",
        );
    }
    if n_rows == 0 || group_ids.len() != n_rows {
        return Err(format!(
            "GARFIELD pair-triple backend row/group mismatch: n_rows={}, group_ids={}",
            n_rows,
            group_ids.len()
        ));
    }
    let max_order = params.max_pick.min(3);
    if params.xor_search_enabled && max_order >= 3 {
        return Err(
            "GARFIELD pair-triple backend supports --xor-search only for order-2 AllPair; "
                .to_string()
                + "disable --xor-search or use the legacy backend for order-3 search",
        );
    }
    let mut candidates = Vec::<BeamRuleCandidate>::new();
    let singleton_negations = [false, true];
    for row_index in 0..n_rows {
        for &negated in singleton_negations.iter() {
            let rule = pair_triple_rule_from_indices(&[row_index], u8::from(negated), group_ids)?;
            let candidate = if let Some(scores) = literal_scores
                .and_then(|scores| scores.get(row_index.saturating_mul(2) + usize::from(negated)))
            {
                if !scores.train.raw_score.is_finite()
                    || !scores.test.raw_score.is_finite()
                    || scores.train.n_hit == 0
                {
                    None
                } else {
                    Some(BeamRuleCandidate {
                        rule,
                        train_score: scores.train.raw_score,
                        test_score: scores.test.raw_score,
                        train: scores.train,
                        test: scores.test,
                    })
                }
            } else {
                pair_triple_candidate_from_rule(
                    rule,
                    y_train,
                    prepared_bits.train_bits(),
                    prepared_bits.row_words_train,
                    y_test,
                    prepared_bits.test_bits(),
                    prepared_bits.row_words_test,
                    n_rows,
                    &params,
                )?
            };
            if let Some(candidate) = candidate {
                candidates.push(candidate);
            }
        }
    }
    if max_order < 2 || n_rows < 2 {
        candidates.sort_by(cmp_candidate);
        candidates.truncate(params.beam_width.max(1));
        return Ok(candidates);
    }

    let pair_scan = scan_all_pairs_continuous_packed_with_parallel_and_xor(
        prepared_bits.train_bits(),
        prepared_bits.row_words_train,
        n_rows,
        y_train,
        y_train.len(),
        GARFIELD_PAIR_TRIPLE_K2,
        params.allow_parallel,
        params.xor_search_enabled,
    )?;
    let mut pair_seeds = Vec::<PairTripleSeed>::with_capacity(pair_scan.candidates.len());
    for (rank, pair) in pair_scan.candidates.iter().enumerate() {
        let rule = pair_triple_rule_from_candidate(pair, group_ids)?;
        if let Some(candidate) = pair_triple_candidate_from_rule(
            rule,
            y_train,
            prepared_bits.train_bits(),
            prepared_bits.row_words_train,
            y_test,
            prepared_bits.test_bits(),
            prepared_bits.row_words_test,
            n_rows,
            &params,
        )? {
            candidates.push(candidate);
            pair_seeds.push(PairTripleSeed {
                first: pair.first,
                second: pair.second,
                pair_rank: rank + 1,
            });
        }
    }
    if max_order >= 3 && !pair_seeds.is_empty() && n_rows >= 3 {
        let triple_scan = scan_pair_seeded_triples_continuous_packed(
            prepared_bits.train_bits(),
            prepared_bits.row_words_train,
            n_rows,
            y_train,
            y_train.len(),
            pair_seeds.as_slice(),
            &[],
            params.beam_width.max(1),
        )?;
        for triple in triple_scan.candidates {
            let rule = pair_triple_rule_from_indices(
                &[triple.first, triple.second, triple.third],
                triple.polarity,
                group_ids,
            )?;
            if let Some(candidate) = pair_triple_candidate_from_rule(
                rule,
                y_train,
                prepared_bits.train_bits(),
                prepared_bits.row_words_train,
                y_test,
                prepared_bits.test_bits(),
                prepared_bits.row_words_test,
                n_rows,
                &params,
            )? {
                candidates.push(candidate);
            }
        }
    }

    // The scanners can return a pair/triple that is also present through a
    // different seed.  Keep one canonical signed rule before handing the
    // candidates to the existing gate/output code.
    let mut dedup = HashMap::<Vec<(usize, bool, u8)>, BeamRuleCandidate>::new();
    for candidate in candidates {
        let key = candidate.rule.lexical_key();
        match dedup.get(&key) {
            Some(current) if cmp_candidate(&candidate, current) != std::cmp::Ordering::Less => {}
            _ => {
                dedup.insert(key, candidate);
            }
        }
    }
    let mut candidates = dedup.into_values().collect::<Vec<_>>();
    candidates.sort_by(cmp_candidate);
    candidates.truncate(params.beam_width.max(1).saturating_mul(2));
    Ok(candidates)
}

pub(super) fn beam_search_train_test_continuous_dispatch(
    y_train: &[f64],
    prepared_bits: &GarfieldUnitBitMatrices<'_>,
    n_rows: usize,
    y_test: &[f64],
    group_ids: &[usize],
    params: BeamSearchParams,
    literal_scores: Option<&[LiteralSingletonScore]>,
) -> Result<Vec<BeamRuleCandidate>, String> {
    if matches!(params.search_backend, GarfieldSearchBackend::PairTriple) {
        return pair_triple_search_train_test_continuous(
            y_train,
            prepared_bits,
            n_rows,
            y_test,
            group_ids,
            params,
            literal_scores,
        );
    }
    if let Some(train_hi) = prepared_bits.train_bits_hi() {
        let test_hi = prepared_bits.test_bits_hi().ok_or_else(|| {
            "internal error: fuzzy GARFIELD prepared bits are missing test high bitplane"
                .to_string()
        })?;
        if let Some(scores) = literal_scores {
            beam_search_train_test_continuous_fuzzy_with_literal_scores(
                y_train,
                prepared_bits.train_bits(),
                train_hi,
                prepared_bits.row_words_train,
                n_rows,
                y_train.len(),
                y_test,
                prepared_bits.test_bits(),
                test_hi,
                prepared_bits.row_words_test,
                y_test.len(),
                group_ids,
                params,
                scores,
            )
        } else {
            beam_search_train_test_continuous_fuzzy(
                y_train,
                prepared_bits.train_bits(),
                train_hi,
                prepared_bits.row_words_train,
                n_rows,
                y_train.len(),
                y_test,
                prepared_bits.test_bits(),
                test_hi,
                prepared_bits.row_words_test,
                y_test.len(),
                group_ids,
                params,
            )
        }
    } else {
        if let Some(scores) = literal_scores {
            beam_search_train_test_continuous_with_literal_scores(
                y_train,
                prepared_bits.train_bits(),
                prepared_bits.row_words_train,
                n_rows,
                y_train.len(),
                y_test,
                prepared_bits.test_bits(),
                prepared_bits.row_words_test,
                y_test.len(),
                group_ids,
                params,
                scores,
            )
        } else {
            beam_search_train_test_continuous(
                y_train,
                prepared_bits.train_bits(),
                prepared_bits.row_words_train,
                n_rows,
                y_train.len(),
                y_test,
                prepared_bits.test_bits(),
                prepared_bits.row_words_test,
                y_test.len(),
                group_ids,
                params,
            )
        }
    }
}

#[inline]
pub(super) fn selected_rows_contiguous_range(
    selected_global_rows: &[usize],
) -> Option<(usize, usize)> {
    let start = *selected_global_rows.first()?;
    for (offset, &row_idx) in selected_global_rows.iter().enumerate() {
        if row_idx != start.saturating_add(offset) {
            return None;
        }
    }
    Some((start, start.saturating_add(selected_global_rows.len())))
}

pub(super) fn local_sites_from_selected_rows(
    selected_global_rows: &[usize],
    logic_bits: &GarfieldLogicBits,
) -> Result<Vec<GarfieldLogicSite>, String> {
    let mut out = Vec::<GarfieldLogicSite>::with_capacity(selected_global_rows.len());
    for &idx in selected_global_rows.iter() {
        let site = logic_bits.sites.get(idx).ok_or_else(|| {
            format!("logic site index out of range while building local sites: {idx}")
        })?;
        out.push(site.clone());
    }
    Ok(out)
}

#[inline]
pub(super) fn intern_logic_site_text(pool: &mut HashMap<String, Arc<str>>, text: &str) -> Arc<str> {
    if let Some(existing) = pool.get(text) {
        return existing.clone();
    }
    let shared: Arc<str> = Arc::from(text);
    pool.insert(text.to_string(), shared.clone());
    shared
}

pub(super) fn build_logic_sites_from_metadata(
    sites: &[SiteInfo],
    row_flip: &[bool],
    mode: GarfieldBinMode,
) -> Vec<GarfieldLogicSite> {
    let _ = mode;
    let row_mul = 1usize;
    if row_flip.len() != sites.len() {
        panic!(
            "build_logic_sites_from_metadata row_flip/site mismatch: row_flip={}, sites={}",
            row_flip.len(),
            sites.len()
        );
    }
    let mut out = Vec::<GarfieldLogicSite>::with_capacity(sites.len().saturating_mul(row_mul));
    let mut chrom_pool = HashMap::<String, Arc<str>>::new();
    let mut snp_pool = HashMap::<String, Arc<str>>::new();
    let mut allele_pool = HashMap::<String, Arc<str>>::new();
    let use_inherited_snp_names = garfield_sites_have_distinct_snp_names(sites);
    for (site_idx, site) in sites.iter().enumerate() {
        let chrom = intern_logic_site_text(&mut chrom_pool, site.chrom.as_str());
        let snp_name = if use_inherited_snp_names {
            intern_logic_site_text(&mut snp_pool, site.snp.trim())
        } else {
            intern_logic_site_text(
                &mut snp_pool,
                format!("{}_{}", site.chrom, site.pos).as_str(),
            )
        };
        let flip = row_flip[site_idx];
        let (ref_text, alt_text) = if flip {
            (site.alt_allele.as_str(), site.ref_allele.as_str())
        } else {
            (site.ref_allele.as_str(), site.alt_allele.as_str())
        };
        let ref_allele = intern_logic_site_text(&mut allele_pool, ref_text);
        let alt_allele = intern_logic_site_text(&mut allele_pool, alt_text);
        let push_mode = |out: &mut Vec<GarfieldLogicSite>, mode_one: GarfieldLogicSiteMode| {
            out.push(GarfieldLogicSite::owned(
                chrom.clone(),
                site.pos,
                snp_name.clone(),
                ref_allele.clone(),
                alt_allele.clone(),
                mode_one,
            ));
        };
        push_mode(&mut out, GarfieldLogicSiteMode::Bin);
    }
    out
}

#[inline]
pub(super) fn intern_logic_site_text_owned(pool: &mut HashSet<Arc<str>>, text: &str) -> Arc<str> {
    if let Some(existing) = pool.get(text) {
        return existing.clone();
    }
    let shared: Arc<str> = Arc::from(text);
    pool.insert(shared.clone());
    shared
}

pub(super) fn build_logic_sites_from_metadata_owned_direct(
    sites: Vec<SiteInfo>,
    row_flip: &[bool],
    mode: GarfieldBinMode,
) -> Vec<GarfieldLogicSite> {
    let _ = mode;
    let row_mul = 1usize;
    if row_flip.len() != sites.len() {
        panic!(
            "build_logic_sites_from_metadata_owned row_flip/site mismatch: row_flip={}, sites={}",
            row_flip.len(),
            sites.len()
        );
    }
    let use_inherited_snp_names = garfield_sites_have_distinct_snp_names(sites.as_slice());
    let mut out = Vec::<GarfieldLogicSite>::with_capacity(sites.len().saturating_mul(row_mul));
    let mut chrom_pool = HashSet::<Arc<str>>::new();
    let mut snp_pool = HashSet::<Arc<str>>::new();
    let mut allele_pool = HashSet::<Arc<str>>::new();
    for (site_idx, site) in sites.into_iter().enumerate() {
        let SiteInfo {
            chrom: chrom_text,
            pos,
            snp: snp_text,
            ref_allele: ref_text_owned,
            alt_allele: alt_text_owned,
        } = site;
        let chrom = intern_logic_site_text_owned(&mut chrom_pool, chrom_text.as_str());
        let snp_name = if use_inherited_snp_names {
            if snp_text.trim().len() == snp_text.len() {
                Arc::<str>::from(snp_text)
            } else {
                Arc::<str>::from(snp_text.trim())
            }
        } else {
            let fallback = format!("{}_{}", chrom_text, pos);
            intern_logic_site_text_owned(&mut snp_pool, fallback.as_str())
        };
        let flip = row_flip[site_idx];
        let (ref_text, alt_text) = if flip {
            (alt_text_owned.as_str(), ref_text_owned.as_str())
        } else {
            (ref_text_owned.as_str(), alt_text_owned.as_str())
        };
        let ref_allele = intern_logic_site_text_owned(&mut allele_pool, ref_text);
        let alt_allele = intern_logic_site_text_owned(&mut allele_pool, alt_text);
        let push_mode = |out: &mut Vec<GarfieldLogicSite>, mode_one: GarfieldLogicSiteMode| {
            out.push(GarfieldLogicSite::owned(
                chrom.clone(),
                pos,
                snp_name.clone(),
                ref_allele.clone(),
                alt_allele.clone(),
                mode_one,
            ));
        };
        push_mode(&mut out, GarfieldLogicSiteMode::Bin);
    }
    out
}

/// Build trait-oriented Max6 views over the genotype-global BIM cache.
/// Each view retains only a shared source handle, row index, orientation bit,
/// and (only when needed) a trait-subset fallback SNP label.
pub(super) fn build_max6_logic_sites_from_shared_metadata(
    source_rows: &[usize],
    source_sites: Arc<CompactMax6SiteMetadata>,
    trait_rows: &[usize],
    row_flip: &[bool],
) -> Result<Vec<GarfieldLogicSite>, String> {
    if source_rows.len() != source_sites.len() || trait_rows.len() != row_flip.len() {
        return Err("cached Max6 site metadata length mismatch".to_string());
    }
    if trait_rows.windows(2).any(|rows| rows[0] >= rows[1]) {
        return Err("trait-valid Max6 source rows must be strictly increasing".to_string());
    }
    let mut selected_indices = Vec::<usize>::with_capacity(trait_rows.len());
    let mut source_cursor = 0usize;
    for &row in trait_rows {
        while source_cursor < source_rows.len() && source_rows[source_cursor] < row {
            source_cursor += 1;
        }
        if source_rows.get(source_cursor) != Some(&row) {
            return Err(format!(
                "trait-valid Max6 source row {row} is absent from cached BIM metadata"
            ));
        }
        selected_indices.push(source_cursor);
    }

    // Match the existing trait-subset naming rule: one missing/duplicate BIM
    // ID changes all displayed marker names to chromosome_position.
    let mut seen = HashSet::<&str>::with_capacity(selected_indices.len());
    let inherited_names = selected_indices.iter().all(|&index| {
        let Some(site) = source_sites.fields(index) else {
            return false;
        };
        let name = site.snp.trim();
        !garfield_snp_name_missing(name) && seen.insert(name)
    });
    let mut fallback_pool = HashSet::<Arc<str>>::new();
    let mut out = Vec::with_capacity(selected_indices.len());
    for (source_index, &flipped) in selected_indices.into_iter().zip(row_flip.iter()) {
        let site = source_sites
            .fields(source_index)
            .ok_or_else(|| format!("Max6 compact site index {source_index} is out of range"))?;
        let snp_override = (!inherited_names).then(|| {
            intern_logic_site_text_owned(
                &mut fallback_pool,
                format!("{}_{}", site.chrom, site.pos).as_str(),
            )
        });
        out.push(GarfieldLogicSite {
            backing: GarfieldLogicSiteBacking::SharedMax6 {
                source_sites: source_sites.clone(),
                source_index,
                flipped,
                snp_override,
            },
            pos: site.pos,
            mode: GarfieldLogicSiteMode::Bin,
        });
    }
    Ok(out)
}

/// Build final logic-site metadata directly from the retained BIM rows.
///
/// The normal window path can retain millions of SNPs. Materializing a
/// `Vec<SiteInfo>` first duplicates the metadata allocation while it is
/// immediately converted into `GarfieldLogicSite`. Stream selected BIM rows
/// instead, keeping only the final representation. Duplicate or missing BIM
/// IDs retain the historical chrom_pos fallback.
pub(super) fn build_logic_sites_from_bim_owned_direct(
    prefix: &str,
    n_snps_total: usize,
    row_source_indices: &[usize],
    row_flip: &[bool],
    mode: GarfieldBinMode,
) -> Result<Vec<GarfieldLogicSite>, String> {
    if row_source_indices.len() != row_flip.len() {
        return Err(format!(
            "BIM logic metadata row mismatch: source_rows={}, row_flip={}",
            row_source_indices.len(),
            row_flip.len()
        ));
    }
    if row_source_indices.is_empty() {
        return Err("BIM logic metadata has no retained rows".to_string());
    }
    if row_source_indices.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("BIM logic metadata source rows must be strictly increasing".to_string());
    }
    if row_source_indices.last().copied().unwrap_or(0) >= n_snps_total {
        return Err(format!(
            "BIM logic metadata source row out of range: max={} n_snps={}",
            row_source_indices.last().copied().unwrap_or(0),
            n_snps_total
        ));
    }

    let _ = mode;
    let row_mul = 1usize;
    let mut out =
        Vec::<GarfieldLogicSite>::with_capacity(row_source_indices.len().saturating_mul(row_mul));
    let mut chrom_pool = HashSet::<Arc<str>>::new();
    let mut snp_pool = HashSet::<Arc<str>>::new();
    let mut allele_pool = HashSet::<Arc<str>>::new();
    let mut inherited_names_valid = true;
    let mut source_cursor = 0usize;
    let mut site_idx = 0usize;
    let chunk_rows = 131_072usize;
    let mut bim = BimChunkReader::open(&normalize_plink_prefix(prefix))?;

    let mut append_site = |site: SiteInfo| {
        let SiteInfo {
            chrom: chrom_text,
            pos,
            snp: snp_text,
            ref_allele: ref_text_owned,
            alt_allele: alt_text_owned,
        } = site;
        let chrom = intern_logic_site_text_owned(&mut chrom_pool, chrom_text.as_str());
        let snp_text = snp_text.trim();
        let snp_name = if garfield_snp_name_missing(snp_text) {
            inherited_names_valid = false;
            Arc::<str>::from("")
        } else {
            let candidate = Arc::<str>::from(snp_text);
            if !snp_pool.insert(candidate.clone()) {
                inherited_names_valid = false;
            }
            candidate
        };
        let flip = row_flip[site_idx];
        let (ref_text, alt_text) = if flip {
            (alt_text_owned.as_str(), ref_text_owned.as_str())
        } else {
            (ref_text_owned.as_str(), alt_text_owned.as_str())
        };
        let ref_allele = intern_logic_site_text_owned(&mut allele_pool, ref_text);
        let alt_allele = intern_logic_site_text_owned(&mut allele_pool, alt_text);
        let push_mode = |out: &mut Vec<GarfieldLogicSite>, mode_one: GarfieldLogicSiteMode| {
            out.push(GarfieldLogicSite::owned(
                chrom.clone(),
                pos,
                snp_name.clone(),
                ref_allele.clone(),
                alt_allele.clone(),
                mode_one,
            ));
        };
        push_mode(&mut out, GarfieldLogicSiteMode::Bin);
        site_idx += 1;
    };

    let mut base = 0usize;
    while base < n_snps_total {
        let end = (base + chunk_rows).min(n_snps_total);
        let mut keep = vec![false; end - base];
        while source_cursor < row_source_indices.len() && row_source_indices[source_cursor] < end {
            let source_row = row_source_indices[source_cursor];
            if source_row < base {
                return Err(format!(
                    "BIM logic metadata source row moved backwards: row={} base={}",
                    source_row, base
                ));
            }
            keep[source_row - base] = true;
            source_cursor += 1;
        }
        for site in bim.read_range_masked(base, end, keep.as_slice())? {
            append_site(site);
        }
        base = end;
    }
    bim.ensure_exhausted(n_snps_total)?;
    if source_cursor != row_source_indices.len() || site_idx != row_source_indices.len() {
        return Err(format!(
            "BIM logic metadata selection mismatch: selected_rows={} source_rows={} sites={}",
            source_cursor,
            row_source_indices.len(),
            site_idx
        ));
    }

    if !inherited_names_valid {
        let mut fallback_pool = HashSet::<Arc<str>>::new();
        for site in out.iter_mut() {
            let GarfieldLogicSiteBacking::Owned { chrom, snp, .. } = &mut site.backing else {
                unreachable!("BIM builder only creates owned logic-site backing")
            };
            let fallback = format!("{}_{}", chrom, site.pos);
            *snp = intern_logic_site_text_owned(&mut fallback_pool, fallback.as_str());
        }
    }
    Ok(out)
}

#[inline]
pub(super) fn garfield_snp_name_missing(raw_name: &str) -> bool {
    let text = raw_name.trim();
    text.is_empty() || text == "." || text.eq_ignore_ascii_case("nan")
}

pub(super) fn garfield_sites_have_distinct_snp_names(sites: &[SiteInfo]) -> bool {
    if sites.is_empty() {
        return false;
    }
    // Borrow BIM strings instead of allocating a second String for every
    // marker while checking uniqueness.
    let mut seen = HashSet::<&str>::with_capacity(sites.len());
    for site in sites.iter() {
        let snp_name = site.snp.trim();
        if garfield_snp_name_missing(snp_name) {
            return false;
        }
        if !seen.insert(snp_name) {
            return false;
        }
    }
    true
}

pub(super) fn materialize_prepared_bit_matrices<'a>(
    prepared: &GarfieldUnitPrepared,
    logic_bits: &'a GarfieldLogicBits,
    train_idx_local: &[usize],
    test_idx_local: &[usize],
    include_full_bits: bool,
) -> Result<GarfieldUnitBitMatrices<'a>, String> {
    let t0 = Instant::now();
    let contiguous = selected_rows_contiguous_range(prepared.selected_global_rows.as_slice());
    let train_is_full_sample =
        sample_indices_are_full_identity(train_idx_local, logic_bits.n_samples);
    let (train_bits, row_words_train) = if let Some((row_start, row_end)) = contiguous {
        if train_is_full_sample {
            (
                Cow::Borrowed(borrow_rows_from_full_bits_range(
                    logic_bits.bits(),
                    logic_bits.row_words,
                    row_start,
                    row_end,
                    logic_bits.sites.len(),
                    logic_bits.n_samples,
                    "garfield::train_bits_borrowed_range",
                )?),
                logic_bits.row_words,
            )
        } else {
            let (bits, row_words_train) = packed_rows_subset_from_full_bits_range(
                logic_bits.bits(),
                logic_bits.row_words,
                row_start,
                row_end,
                train_idx_local,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            (Cow::Owned(bits), row_words_train)
        }
    } else {
        let (bits, row_words_train) = packed_rows_subset_from_full_bits(
            logic_bits.bits(),
            logic_bits.row_words,
            prepared.selected_global_rows.as_slice(),
            train_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        (Cow::Owned(bits), row_words_train)
    };
    let train_bits_hi = if let Some(bits_hi_flat) = logic_bits.bits_hi_flat.as_ref() {
        Some(if let Some((row_start, row_end)) = contiguous {
            if train_is_full_sample {
                Cow::Borrowed(borrow_rows_from_full_bits_range(
                    bits_hi_flat.as_slice(),
                    logic_bits.row_words,
                    row_start,
                    row_end,
                    logic_bits.sites.len(),
                    logic_bits.n_samples,
                    "garfield::train_bits_hi_borrowed_range",
                )?)
            } else {
                Cow::Owned(
                    packed_rows_subset_from_full_bits_range(
                        bits_hi_flat.as_slice(),
                        logic_bits.row_words,
                        row_start,
                        row_end,
                        train_idx_local,
                        logic_bits.sites.len(),
                        logic_bits.n_samples,
                    )?
                    .0,
                )
            }
        } else {
            Cow::Owned(
                packed_rows_subset_from_full_bits(
                    bits_hi_flat.as_slice(),
                    logic_bits.row_words,
                    prepared.selected_global_rows.as_slice(),
                    train_idx_local,
                    logic_bits.sites.len(),
                    logic_bits.n_samples,
                )?
                .0,
            )
        })
    } else {
        None
    };
    let (test_bits, row_words_test) = if train_idx_local == test_idx_local {
        (None, row_words_train)
    } else if let Some((row_start, row_end)) = contiguous {
        if sample_indices_are_full_identity(test_idx_local, logic_bits.n_samples) {
            (
                Some(Cow::Borrowed(borrow_rows_from_full_bits_range(
                    logic_bits.bits(),
                    logic_bits.row_words,
                    row_start,
                    row_end,
                    logic_bits.sites.len(),
                    logic_bits.n_samples,
                    "garfield::test_bits_borrowed_range",
                )?)),
                logic_bits.row_words,
            )
        } else {
            let (bits, row_words_test) = packed_rows_subset_from_full_bits_range(
                logic_bits.bits(),
                logic_bits.row_words,
                row_start,
                row_end,
                test_idx_local,
                logic_bits.sites.len(),
                logic_bits.n_samples,
            )?;
            (Some(Cow::Owned(bits)), row_words_test)
        }
    } else {
        let (bits, row_words_test) = packed_rows_subset_from_full_bits(
            logic_bits.bits(),
            logic_bits.row_words,
            prepared.selected_global_rows.as_slice(),
            test_idx_local,
            logic_bits.sites.len(),
            logic_bits.n_samples,
        )?;
        (Some(Cow::Owned(bits)), row_words_test)
    };
    let test_bits_hi = if let Some(bits_hi_flat) = logic_bits.bits_hi_flat.as_ref() {
        if train_idx_local == test_idx_local {
            None
        } else if let Some((row_start, row_end)) = contiguous {
            if sample_indices_are_full_identity(test_idx_local, logic_bits.n_samples) {
                Some(Cow::Borrowed(borrow_rows_from_full_bits_range(
                    bits_hi_flat.as_slice(),
                    logic_bits.row_words,
                    row_start,
                    row_end,
                    logic_bits.sites.len(),
                    logic_bits.n_samples,
                    "garfield::test_bits_hi_borrowed_range",
                )?))
            } else {
                Some(Cow::Owned(
                    packed_rows_subset_from_full_bits_range(
                        bits_hi_flat.as_slice(),
                        logic_bits.row_words,
                        row_start,
                        row_end,
                        test_idx_local,
                        logic_bits.sites.len(),
                        logic_bits.n_samples,
                    )?
                    .0,
                ))
            }
        } else {
            Some(Cow::Owned(
                packed_rows_subset_from_full_bits(
                    bits_hi_flat.as_slice(),
                    logic_bits.row_words,
                    prepared.selected_global_rows.as_slice(),
                    test_idx_local,
                    logic_bits.sites.len(),
                    logic_bits.n_samples,
                )?
                .0,
            ))
        }
    } else {
        None
    };
    let alias_full_to_train =
        include_full_bits && train_is_full_sample && row_words_train == logic_bits.row_words;
    let selected_bits_full = if include_full_bits && !alias_full_to_train {
        Some(if let Some((row_start, row_end)) = contiguous {
            Cow::Borrowed(borrow_rows_from_full_bits_range(
                logic_bits.bits(),
                logic_bits.row_words,
                row_start,
                row_end,
                logic_bits.sites.len(),
                logic_bits.n_samples,
                "garfield::unit_bits_borrowed_range",
            )?)
        } else {
            Cow::Owned(gather_rows_by_indices(
                logic_bits.bits(),
                logic_bits.row_words,
                prepared.selected_global_rows.as_slice(),
                "garfield::unit_bits_indices",
            )?)
        })
    } else {
        None
    };
    let selected_bits_full_hi = if include_full_bits && !alias_full_to_train {
        if let Some(bits_hi_flat) = logic_bits.bits_hi_flat.as_ref() {
            Some(if let Some((row_start, row_end)) = contiguous {
                Cow::Borrowed(borrow_rows_from_full_bits_range(
                    bits_hi_flat.as_slice(),
                    logic_bits.row_words,
                    row_start,
                    row_end,
                    logic_bits.sites.len(),
                    logic_bits.n_samples,
                    "garfield::unit_bits_hi_borrowed_range",
                )?)
            } else {
                Cow::Owned(gather_rows_by_indices(
                    bits_hi_flat.as_slice(),
                    logic_bits.row_words,
                    prepared.selected_global_rows.as_slice(),
                    "garfield::unit_bits_hi_indices",
                )?)
            })
        } else {
            None
        }
    } else {
        None
    };
    let out = GarfieldUnitBitMatrices {
        train_bits,
        train_bits_hi,
        row_words_train,
        test_bits,
        test_bits_hi,
        row_words_test,
        selected_bits_full,
        selected_bits_full_hi,
        selected_bits_full_alias_train: alias_full_to_train,
        selected_bits_full_hi_alias_train: alias_full_to_train && logic_bits.bits_hi_flat.is_some(),
    };
    GARFIELD_MATERIALIZE_BITS_NS.fetch_add(elapsed_ns_saturating(t0), Ordering::Relaxed);
    Ok(out)
}
