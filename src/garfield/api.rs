use super::*;

#[pyclass(name = "Max6PreparedGenotypeTags")]
pub struct PyMax6PreparedGenotypeTags {
    inner: Arc<Max6PreparedGenotypeTags>,
}

#[pymethods]
impl PyMax6PreparedGenotypeTags {
    #[getter]
    fn unit_count(&self) -> usize {
        self.inner.screen_units.len()
    }

    #[getter]
    fn tag_count(&self) -> usize {
        self.inner.active_source_rows.len()
    }
}

#[pyclass(name = "Max6PreparedGenotypeUniverse")]
pub struct PyMax6PreparedGenotypeUniverse {
    inner: Arc<Max6PreparedGenotypeUniverse>,
}

#[pymethods]
impl PyMax6PreparedGenotypeUniverse {
    #[getter]
    fn unique_pair_count(&self) -> usize {
        self.inner.pair_universe_size
    }

    #[getter]
    fn tag_count(&self) -> usize {
        self.inner.tags.active_source_rows.len()
    }

    #[getter]
    fn unit_count(&self) -> usize {
        self.inner.tags.screen_units.len()
    }

    #[getter]
    fn bonferroni_q_cut(&self) -> f64 {
        self.inner.bonferroni_q_cut
    }

    #[getter]
    fn bonferroni_p_cut(&self) -> f64 {
        self.inner.bonferroni_p_cut
    }

    #[getter]
    fn unit_kind(&self) -> String {
        self.inner.tags.unit_kind.clone()
    }

    #[getter]
    fn global_site_keep<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.inner.tags.global_site_keep.clone()).into_bound()
    }
}

#[pyfunction(name = "prepare_max6_genotype_tags")]
#[pyo3(signature = (
    prefix,
    sample_ids,
    sample_indices,
    unit_kind="window",
    groups=None,
    group_names=None,
    extension=50000,
    step=None,
    scan_bimranges=None,
    maf_threshold=0.02,
    max_missing_rate=0.05,
    het_threshold=1.0,
    snps_only=false,
    threads=0,
    logic_mem_budget_bytes=0,
    ld_tag_r2=1.0
))]
pub fn prepare_max6_genotype_tags_py(
    py: Python<'_>,
    prefix: String,
    sample_ids: Vec<String>,
    sample_indices: Vec<usize>,
    unit_kind: &str,
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
) -> PyResult<PyMax6PreparedGenotypeTags> {
    arm_interrupt_trap();
    let input_kind_override_prefix = prefix.clone();
    let tags = py
        .detach(move || {
            let _input_kind_override =
                crate::pgen::scoped_input_kind_override(&input_kind_override_prefix)?;
            prepare_max6_genotype_tags_owned(
                prefix,
                sample_ids,
                sample_indices,
                unit_kind.to_string(),
                groups,
                group_names,
                extension,
                step,
                scan_bimranges,
                maf_threshold,
                max_missing_rate,
                het_threshold,
                snps_only,
                threads,
                logic_mem_budget_bytes,
                ld_tag_r2,
            )
        })
        .map_err(map_err_string_to_py)?;
    Ok(PyMax6PreparedGenotypeTags {
        inner: Arc::new(tags),
    })
}

#[pyfunction(name = "finalize_max6_genotype_universe")]
pub fn finalize_max6_genotype_universe_py(
    py: Python<'_>,
    tags: Py<PyMax6PreparedGenotypeTags>,
) -> PyResult<PyMax6PreparedGenotypeUniverse> {
    let tags = tags.borrow(py).inner.clone();
    let universe = py
        .detach(move || finalize_max6_genotype_universe(tags))
        .map_err(map_err_string_to_py)?;
    Ok(PyMax6PreparedGenotypeUniverse {
        inner: Arc::new(universe),
    })
}

pub fn garfield_debug_probe_single_group_from_files(
    prefix: String,
    pheno_path: String,
    grm_path: String,
    unit_name: String,
    chrom: String,
    bp_start: i32,
    bp_end: i32,
    out_tsv: String,
    logic_maf_threshold: f32,
    threads: usize,
) -> Result<String, String> {
    let prefix_norm = normalize_plink_prefix(&prefix);
    let fam_iids = read_fam(prefix_norm.as_str())?;
    let y = read_single_trait_pheno_aligned(&pheno_path, fam_iids.as_slice())?;
    let (grm_vec, grm_n) = load_square_matrix_f64_from_file(&grm_path, 0.0)?;
    if grm_n != fam_iids.len() {
        return Err(format!(
            "GRM dimension mismatch: grm_n={}, fam_n={}",
            grm_n,
            fam_iids.len()
        ));
    }
    let probe = GarfieldBeamDebugProbe {
        unit_name: unit_name.clone(),
        out_tsv: out_tsv.clone(),
    };
    let _ = garfield_logic_search_bed_owned(
        prefix_norm,
        y,
        Some(grm_vec),
        None,
        0usize,
        None,
        None,
        None,
        "geneset".to_string(),
        Some(vec![vec![(chrom, bp_start, bp_end)]]),
        None,
        Some(vec![unit_name]),
        0usize,
        None,
        None,
        "bin".to_string(),
        "corr".to_string(),
        "imp".to_string(),
        100usize,
        0.0,
        20usize,
        "auto".to_string(),
        "gev".to_string(),
        0.99,
        false,
        ExtraTreesConfig {
            n_estimators: 100,
            max_depth: 5,
            min_samples_leaf: 1,
            min_samples_split: 2,
            bootstrap: true,
            feature_subsample: 0.0,
            seed: 42,
            allow_parallel: true,
        },
        42u64,
        4usize,
        2usize,
        100usize,
        "gain_from_layer:1".to_string(),
        logic_maf_threshold,
        logic_maf_threshold,
        0.05,
        1.0,
        false,
        65_536usize,
        threads,
        -5.0,
        5.0,
        50usize,
        1e-3,
        true,
        15_000usize,
        false,
        "exact".to_string(),
        100usize,
        1e-4,
        15_000usize,
        None,
        None,
        100usize,
        0usize,
        0.0,
        "legacy".to_string(),
        true,
        None,
        false,
        false,
        true,
        false,
        false,
        None,
        0usize,
        Some(probe),
        0,
        false,
        false,
        false,
        false,
        5357123296692292677u64,
        String::new(),
        "0,0.05,0.10,0.20,0.30,0.50".to_string(),
        "0,0.2,0.4,0.6,0.8,1.0".to_string(),
        false,
        500,
        sampled_screen::DEFAULT_TAIL_PAIR_SAMPLE_SEED,
        false,
        Vec::new(),
        None,
        None,
        false,
        0usize,
        0.0,
        GARFIELD_MAX6_LD_TAG_R2_DEFAULT,
        None,
        false,
    )?;
    Ok(out_tsv)
}

#[pyfunction(name = "garfield_logic_search_bed")]
#[pyo3(signature = (
    prefix,
    y,
    grm=None,
    x_cov=None,
    sample_ids=None,
    sample_indices=None,
    site_keep=None,
    unit_kind="window",
    groups=None,
    null_groups=None,
    group_names=None,
    extension=50000,
    step=None,
    scan_bimranges=None,
    bin_mode="bin",
    ml_method="corr",
    ml_importance="imp",
    ml_top_k=64,
    ml_top_frac=0.0,
    permutation_repeats=300,
    permutation_scoring="auto",
    rule_null_penalty_method="gev",
    rule_null_quantile=0.99,
    rule_null_report_pvalue=false,
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=1,
    min_samples_split=2,
    bootstrap=true,
    feature_subsample=0.0,
    seed=42,
    max_pick=4,
    exhaustive_depth=1,
    beam_width=100,
    rank_score="gain_from_layer:1",
    maf_threshold=0.02,
    logic_maf_threshold=0.02,
    max_missing_rate=0.05,
    het_threshold=1.0,
    snps_only=false,
    block_cols=65536,
    threads=0,
    low=-5.0,
    high=5.0,
    max_iter=50,
    tol=1e-3,
    add_intercept=true,
    exact_n_max=15000,
    require_lapack=false,
    residual_method="exact",
    pcg_max_iter=100,
    pcg_tol=1e-4,
    pcg_small_n_max=15000,
    out_prefix=None,
    simbench_path=None,
    top_rules_per_unit=1,
    max_output_rules=0,
    max_output_ratio=0.0,
    search_backend="legacy",
    rule_permutation=true,
    prior_len=None,
    no_clean=false,
    raw_design=false,
    filter_xor_substates=true,
    xor_search=false,
    whole_genome_dev_mode=false,
    progress_callback=None,
    progress_every=0,
    logic_mem_budget_bytes=0,
    diagnostics=false,
    sampled_screen_null_diagnostic=false,
    max6_screen_exceedance_audit=false,
    max6_screen_pot_audit=false,
    sampled_screen_pair_seed=5357123296692292677,
    sampled_screen_pair_seeds="",
    sampled_screen_maf_cutpoints="0,0.05,0.10,0.20,0.30,0.50",
    sampled_screen_r2_cutpoints="0,0.2,0.4,0.6,0.8,1.0",
    max6_screen_tail_audit=false,
    max6_screen_tail_audit_repeats=500,
    max6_screen_tail_pair_seed=5357123296692292677,
    max6_screen_tail_support_aware=false,
    max6_screen_tail_support_shares=vec![40,50,60],
    prepared_max6_universe=None,
    prepared_sample_ids=None,
    grm_int_only=None,
    min_cell=10,
    min_cell_fraction=0.0,
    ld_tag_r2=1.0,
    summary_callback=None,
    write_provenance=false
))]
pub fn garfield_logic_search_bed_py<'py>(
    py: Python<'py>,
    prefix: String,
    y: PyReadonlyArray1<'py, f64>,
    grm: Option<PyReadonlyArray2<'py, f64>>,
    x_cov: Option<PyReadonlyArray2<'py, f64>>,
    sample_ids: Option<Vec<String>>,
    sample_indices: Option<Vec<usize>>,
    site_keep: Option<PyReadonlyArray1<'py, bool>>,
    unit_kind: &str,
    groups: Option<Vec<Vec<(String, i32, i32)>>>,
    null_groups: Option<Vec<Vec<(String, i32, i32)>>>,
    group_names: Option<Vec<String>>,
    extension: usize,
    step: Option<usize>,
    scan_bimranges: Option<Vec<String>>,
    bin_mode: &str,
    ml_method: &str,
    ml_importance: &str,
    ml_top_k: usize,
    ml_top_frac: f64,
    permutation_repeats: usize,
    permutation_scoring: &str,
    rule_null_penalty_method: &str,
    rule_null_quantile: f64,
    rule_null_report_pvalue: bool,
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    bootstrap: bool,
    feature_subsample: f64,
    seed: u64,
    max_pick: usize,
    exhaustive_depth: usize,
    beam_width: usize,
    rank_score: &str,
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
    residual_method: &str,
    pcg_max_iter: usize,
    pcg_tol: f64,
    pcg_small_n_max: usize,
    out_prefix: Option<String>,
    simbench_path: Option<String>,
    top_rules_per_unit: usize,
    max_output_rules: usize,
    max_output_ratio: f64,
    search_backend: &str,
    rule_permutation: bool,
    prior_len: Option<Vec<f64>>,
    no_clean: bool,
    raw_design: bool,
    filter_xor_substates: bool,
    xor_search: bool,
    whole_genome_dev_mode: bool,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
    logic_mem_budget_bytes: u64,
    diagnostics: bool,
    sampled_screen_null_diagnostic: bool,
    max6_screen_exceedance_audit: bool,
    max6_screen_pot_audit: bool,
    sampled_screen_pair_seed: u64,
    sampled_screen_pair_seeds: &str,
    sampled_screen_maf_cutpoints: &str,
    sampled_screen_r2_cutpoints: &str,
    max6_screen_tail_audit: bool,
    max6_screen_tail_audit_repeats: usize,
    max6_screen_tail_pair_seed: u64,
    max6_screen_tail_support_aware: bool,
    max6_screen_tail_support_shares: Vec<usize>,
    prepared_max6_universe: Option<Py<PyMax6PreparedGenotypeUniverse>>,
    prepared_sample_ids: Option<Vec<String>>,
    grm_int_only: Option<bool>,
    min_cell: usize,
    min_cell_fraction: f64,
    ld_tag_r2: f64,
    summary_callback: Option<Py<PyAny>>,
    write_provenance: bool,
) -> PyResult<Bound<'py, PyDict>> {
    arm_interrupt_trap();
    let input_kind_override_prefix = prefix.clone();
    let grm_int_only =
        grm_int_only.unwrap_or_else(|| search_backend.trim().eq_ignore_ascii_case("max6-grmmax6"));
    let y_vec = read_y_f64(&y);
    let grm_vec = if let Some(arr) = grm {
        let mat = arr.as_array();
        if mat.ndim() != 2 || mat.shape()[0] != mat.shape()[1] {
            return Err(PyValueError::new_err(
                "grm must be a square float64 matrix.",
            ));
        }
        Some(match crate::stats_common::array_standard_slice(&arr) {
            Ok(s) => s.to_vec(),
            Err(_) => crate::stats_common::copy_array_standard_layout(&mat),
        })
    } else {
        None
    };
    let (x_cov_vec, q_cov) = if let Some(arr) = x_cov {
        let mat = arr.as_array();
        if mat.ndim() != 2 {
            return Err(PyValueError::new_err(
                "x_cov must be 2D (n_samples, q_cov).",
            ));
        }
        let q_cov = mat.shape()[1];
        let cov_vec = match crate::stats_common::array_standard_slice(&arr) {
            Ok(s) => s.to_vec(),
            Err(_) => crate::stats_common::copy_array_standard_layout(&mat),
        };
        (Some(cov_vec), q_cov)
    } else {
        (None, 0usize)
    };
    let site_keep_vec = if let Some(arr) = site_keep {
        Some(match crate::stats_common::array_standard_slice(&arr) {
            Ok(s) => s.to_vec(),
            Err(_) => crate::stats_common::copy_array_standard_layout(&arr.as_array()),
        })
    } else {
        None
    };
    let prepared_max6_universe =
        prepared_max6_universe.map(|prepared| prepared.borrow(py).inner.clone());
    let tree_cfg = ExtraTreesConfig {
        n_estimators: n_estimators.max(1),
        max_depth: max_depth.max(1),
        min_samples_leaf: min_samples_leaf.max(1),
        min_samples_split: min_samples_split.max(2),
        bootstrap,
        feature_subsample,
        seed,
        allow_parallel: true,
    };
    let result = py
        .detach(move || {
            let _input_kind_override =
                crate::pgen::scoped_input_kind_override(&input_kind_override_prefix)?;
            garfield_logic_search_bed_owned(
                prefix,
                y_vec,
                grm_vec,
                x_cov_vec,
                q_cov,
                sample_ids,
                sample_indices,
                site_keep_vec,
                unit_kind.to_string(),
                groups,
                null_groups,
                group_names,
                extension,
                step,
                scan_bimranges,
                bin_mode.to_string(),
                ml_method.to_string(),
                ml_importance.to_string(),
                ml_top_k,
                ml_top_frac,
                permutation_repeats,
                permutation_scoring.to_string(),
                rule_null_penalty_method.to_string(),
                rule_null_quantile,
                rule_null_report_pvalue,
                tree_cfg,
                seed,
                max_pick,
                exhaustive_depth,
                beam_width,
                rank_score.to_string(),
                maf_threshold,
                logic_maf_threshold,
                max_missing_rate,
                het_threshold,
                snps_only,
                block_cols,
                threads,
                low,
                high,
                max_iter,
                tol,
                add_intercept,
                exact_n_max,
                require_lapack,
                residual_method.to_string(),
                pcg_max_iter,
                pcg_tol,
                pcg_small_n_max,
                out_prefix,
                simbench_path,
                top_rules_per_unit,
                max_output_rules,
                max_output_ratio,
                search_backend.to_string(),
                rule_permutation,
                prior_len,
                no_clean,
                raw_design,
                filter_xor_substates,
                xor_search,
                whole_genome_dev_mode,
                progress_callback,
                progress_every,
                None,
                logic_mem_budget_bytes,
                diagnostics,
                sampled_screen_null_diagnostic,
                max6_screen_exceedance_audit,
                max6_screen_pot_audit,
                sampled_screen_pair_seed,
                sampled_screen_pair_seeds.to_string(),
                sampled_screen_maf_cutpoints.to_string(),
                sampled_screen_r2_cutpoints.to_string(),
                max6_screen_tail_audit,
                max6_screen_tail_audit_repeats,
                max6_screen_tail_pair_seed,
                max6_screen_tail_support_aware,
                max6_screen_tail_support_shares,
                prepared_max6_universe,
                prepared_sample_ids,
                grm_int_only,
                min_cell,
                min_cell_fraction,
                ld_tag_r2,
                summary_callback,
                write_provenance,
            )
        })
        .map_err(map_err_string_to_py)?;

    let out = PyDict::new(py);
    out.set_item("pseudo_prefix", result.pseudo_prefix)?;
    out.set_item("rules_tsv", result.rules_tsv)?;
    out.set_item("diagnostics_tsv", result.diagnostics_tsv)?;
    out.set_item("frontier_diagnostics_tsv", result.frontier_diagnostics_tsv)?;
    out.set_item("rules_compare_tsv", result.rules_compare_tsv)?;
    if let Some(path) = result.max6_pairs_tsv.as_ref() {
        out.set_item("max6_pairs_tsv", path)?;
    } else {
        out.set_item("max6_pairs_tsv", py.None())?;
    }
    if let Some(path) = result.max6_pairs_provenance_tsv.as_ref() {
        out.set_item("max6_pairs_provenance_tsv", path)?;
    } else {
        out.set_item("max6_pairs_provenance_tsv", py.None())?;
    }
    if let Some(path) = result.max6_ld_tags_tsv.as_ref() {
        out.set_item("max6_ld_tags_tsv", path)?;
    } else {
        out.set_item("max6_ld_tags_tsv", py.None())?;
    }
    if result.max6_bonferroni_q_cut.is_some() {
        out.set_item("max6_ld_tag_r2", ld_tag_r2)?;
        out.set_item(
            "max6_ld_tag_mode",
            if ld_tag_r2 == 1.0 {
                "exact_genotype_identity"
            } else {
                "approximate_r2"
            },
        )?;
    } else {
        out.set_item("max6_ld_tag_r2", py.None())?;
    }
    if let Some(path) = result.max6_threshold_tsv.as_ref() {
        out.set_item("max6_threshold_tsv", path)?;
    } else {
        out.set_item("max6_threshold_tsv", py.None())?;
    }
    if let Some(path) = result.max6_permutation_maxima_tsv.as_ref() {
        out.set_item("max6_permutation_maxima_tsv", path)?;
    } else {
        out.set_item("max6_permutation_maxima_tsv", py.None())?;
    }
    if let Some(path) = result.max6_screen_maxima_tsv.as_ref() {
        out.set_item("max6_screen_maxima_tsv", path)?;
    } else {
        out.set_item("max6_screen_maxima_tsv", py.None())?;
    }
    if let Some(path) = result.max6_screen_validation_maxima_tsv.as_ref() {
        out.set_item("max6_screen_validation_maxima_tsv", path)?;
    } else {
        out.set_item("max6_screen_validation_maxima_tsv", py.None())?;
    }
    out.set_item(
        "max6_screen_sampled_maxima_tsv",
        result.max6_screen_sampled_maxima_tsv.as_deref(),
    )?;
    out.set_item(
        "max6_screen_sample_manifest_json",
        result.max6_screen_sample_manifest_json.as_deref(),
    )?;
    out.set_item(
        "max6_screen_sample_strata_tsv",
        result.max6_screen_sample_strata_tsv.as_deref(),
    )?;
    out.set_item(
        "max6_screen_full_maximizer_geometry_tsv",
        result.max6_screen_full_maximizer_geometry_tsv.as_deref(),
    )?;
    out.set_item(
        "max6_screen_tail_top_pairs_tsv",
        result.max6_screen_tail_top_pairs_tsv.as_deref(),
    )?;
    out.set_item(
        "max6_screen_tail_sampled_ranks_tsv",
        result.max6_screen_tail_sampled_ranks_tsv.as_deref(),
    )?;
    out.set_item(
        "max6_screen_tail_summary_json",
        result.max6_screen_tail_summary_json.as_deref(),
    )?;
    out.set_item(
        "max6_screen_tail_support_aware_maxima_tsv",
        result.max6_screen_tail_support_aware_maxima_tsv.as_deref(),
    )?;
    out.set_item(
        "max6_screen_tail_support_aware_strata_tsv",
        result.max6_screen_tail_support_aware_strata_tsv.as_deref(),
    )?;
    out.set_item(
        "max6_screen_tail_support_aware_summary_json",
        result
            .max6_screen_tail_support_aware_summary_json
            .as_deref(),
    )?;
    out.set_item(
        "max6_screen_exceedance_audit_per_null_tsv",
        result.max6_screen_exceedance_audit_per_null_tsv.as_deref(),
    )?;
    out.set_item(
        "max6_screen_exceedance_audit_summary_tsv",
        result.max6_screen_exceedance_audit_summary_tsv.as_deref(),
    )?;
    out.set_item(
        "max6_screen_exceedance_audit_top_pairs_tsv",
        result.max6_screen_exceedance_audit_top_pairs_tsv.as_deref(),
    )?;
    out.set_item(
        "max6_screen_exceedance_audit_manifest_json",
        result.max6_screen_exceedance_audit_manifest_json.as_deref(),
    )?;
    out.set_item(
        "max6_screen_pot_audit_manifest_json",
        result.max6_screen_pot_audit_manifest_json.as_deref(),
    )?;
    out.set_item(
        "max6_screen_pot_sample_scores_manifest_json",
        result
            .max6_screen_pot_sample_scores_manifest_json
            .as_deref(),
    )?;
    out.set_item(
        "timing_max6_screen_tail_audit_wall_s",
        result.timing_max6_screen_tail_audit_wall_s,
    )?;
    out.set_item(
        "timing_max6_screen_exceedance_audit_wall_s",
        result.timing_max6_screen_exceedance_audit_wall_s,
    )?;
    out.set_item(
        "timing_max6_screen_pot_audit_wall_s",
        result.timing_max6_screen_pot_audit_wall_s,
    )?;
    out.set_item(
        "timing_max6_screen_sampled_diagnostic_wall_s",
        result.timing_max6_screen_sampled_diagnostic_wall_s,
    )?;
    out.set_item(
        "max6_pair_pve_total_quadratic",
        result.max6_pair_pve_total_quadratic,
    )?;
    if result.max6_pair_pve_total_quadratic.is_some() {
        out.set_item(
            "max6_pair_pve_definition",
            "scaled_winner_score / scaled_y_quadratic; GRM-metric partial-R2",
        )?;
    } else {
        out.set_item("max6_pair_pve_definition", py.None())?;
    }
    out.set_item("max6_pair_universe_size", result.max6_pair_universe_size)?;
    out.set_item("grm_int_only", grm_int_only)?;
    out.set_item(
        "max6_screen_threshold_applied",
        !grm_int_only && result.max6_screen_threshold.is_some(),
    )?;
    out.set_item(
        "max6_pair_score_source",
        if grm_int_only {
            "max6_screen_proposal_score"
        } else {
            "exact_grm_max6_score"
        },
    )?;
    out.set_item(
        "max6_pair_channel_source",
        if grm_int_only {
            "max6_screen_winning_gate"
        } else {
            "exact_grm_max6_winning_channel"
        },
    )?;
    out.set_item(
        "max6_bonferroni_alpha",
        (!grm_int_only)
            .then_some(result.max6_bonferroni_q_cut)
            .flatten()
            .map(|_| MAX6_GRMMAX6_BONFERRONI_ALPHA),
    )?;
    out.set_item(
        "max6_bonferroni_channel_count",
        (!grm_int_only)
            .then_some(result.max6_bonferroni_q_cut)
            .flatten()
            .map(|_| MAX6_GRMMAX6_BONFERRONI_CHANNELS),
    )?;
    out.set_item(
        "max6_bonferroni_q_cut",
        if grm_int_only {
            None
        } else {
            result.max6_bonferroni_q_cut
        },
    )?;
    out.set_item(
        "max6_bonferroni_p_cut",
        if grm_int_only {
            None
        } else {
            result.max6_bonferroni_p_cut
        },
    )?;
    out.set_item(
        "max6_pvalue_method",
        if grm_int_only {
            Some("chi2_1_reference_pvalue; GRM_FWL_product_interaction_adjusted_for_main_effects")
        } else {
            result
                .max6_bonferroni_q_cut
                .map(|_| "chi2_1_reference_pvalue; fixed_m6_pair_cutoff; unique_pair_universe")
        },
    )?;
    out.set_item("max6_score_threshold", result.max6_score_threshold)?;
    out.set_item(
        "max6_gumbel_score_threshold",
        result.max6_gumbel_score_threshold,
    )?;
    out.set_item("max6_screen_threshold", result.max6_screen_threshold)?;
    out.set_item(
        "max6_screen_threshold_method",
        result.max6_screen_threshold_method.clone(),
    )?;
    out.set_item("max6_screen_null_repeats", result.max6_screen_null_repeats)?;
    out.set_item(
        "max6_screen_validation_repeats",
        result.max6_screen_validation_repeats,
    )?;
    out.set_item(
        "max6_screen_validation_rejections",
        result.max6_screen_validation_rejections,
    )?;
    out.set_item(
        "max6_screen_validation_fwer",
        result.max6_screen_validation_fwer,
    )?;
    out.set_item(
        "max6_screen_validation_ci_lower",
        result.max6_screen_validation_ci_lower,
    )?;
    out.set_item(
        "max6_screen_validation_ci_upper",
        result.max6_screen_validation_ci_upper,
    )?;
    out.set_item(
        "max6_screen_validation_method",
        if result.max6_screen_validation_repeats > 0 {
            "independent_parametric_null_draws_conditioned_on_observed_y_fit"
        } else {
            "not_run"
        },
    )?;
    out.set_item(
        "max6_screen_validation_seed_policy",
        "splitmix64_domain_separated_from_calibration_seed",
    )?;
    out.set_item(
        "max6_screen_pair_universe_size",
        result.max6_screen_pair_universe_size,
    )?;
    out.set_item(
        "max6_screen_window_pair_occurrences",
        result.max6_screen_window_pair_occurrences,
    )?;
    out.set_item(
        "max6_screen_candidate_count",
        result.max6_screen_candidate_count,
    )?;
    out.set_item(
        "max6_final_screen_subset_audit_passed",
        result.max6_final_screen_subset_audit_passed,
    )?;
    out.set_item(
        "max6_final_screen_subset_audit_pairs_checked",
        result.max6_final_screen_subset_audit_pairs_checked,
    )?;
    out.set_item(
        "max6_final_screen_subset_audit_violations",
        result.max6_final_screen_subset_audit_violations,
    )?;
    if let Some(score_scale) = result.max6_score_scale {
        out.set_item("max6_score_scale", score_scale)?;
        out.set_item(
            "max6_score_scale_source",
            if result.train_fit.no_grm_boundary {
                "inverse_sigma_e2_no_grm_boundary"
            } else {
                "inverse_sigma_g2"
            },
        )?;
    } else {
        out.set_item("max6_score_scale", py.None())?;
        out.set_item("max6_score_scale_source", py.None())?;
    }
    out.set_item(
        "max6_permutation_score_count",
        result.max6_permutation_score_count,
    )?;
    out.set_item(
        "max6_permutation_pair_pool_size",
        result.max6_permutation_pair_pool_size,
    )?;
    out.set_item(
        "max6_permutation_pairs_per_repeat",
        result.max6_permutation_pairs_per_repeat,
    )?;
    out.set_item(
        "max6_permutation_pair_draw_count",
        result.max6_permutation_pair_draw_count,
    )?;
    out.set_item(
        "timing_max6_permutation_setup_wall_s",
        result.max6_permutation_timing.setup_wall_seconds,
    )?;
    out.set_item(
        "timing_max6_permutation_score_wall_s",
        result.max6_permutation_timing.score_wall_seconds,
    )?;
    out.set_item(
        "timing_max6_permutation_threshold_wall_s",
        result.max6_permutation_timing.threshold_wall_seconds,
    )?;
    out.set_item(
        "timing_max6_permutation_maxima_write_wall_s",
        result.max6_permutation_timing.maxima_write_wall_seconds,
    )?;
    out.set_item(
        "timing_max6_permutation_projected_y_task_sum_s",
        result.max6_permutation_timing.projected_y_cpu_seconds,
    )?;
    out.set_item(
        "timing_max6_permutation_endpoint_prepare_task_sum_s",
        result.max6_permutation_timing.endpoint_prepare_cpu_seconds,
    )?;
    out.set_item(
        "timing_max6_permutation_rerank_kernel_task_sum_s",
        result.max6_permutation_timing.rerank_kernel_cpu_seconds,
    )?;
    out.set_item(
        "timing_max6_permutation_rerank_pcx_task_sum_s",
        result.max6_permutation_timing.rerank_pcx_cpu_seconds,
    )?;
    out.set_item(
        "timing_max6_permutation_rerank_geometry_task_sum_s",
        result.max6_permutation_timing.rerank_geometry_cpu_seconds,
    )?;
    out.set_item(
        "timing_max6_permutation_rerank_q_task_sum_s",
        result.max6_permutation_timing.rerank_q_cpu_seconds,
    )?;
    out.set_item(
        "timing_max6_permutation_rerank_score_task_sum_s",
        result.max6_permutation_timing.rerank_score_cpu_seconds,
    )?;
    out.set_item(
        "timing_max6_permutation_rerank_topk_task_sum_s",
        result.max6_permutation_timing.rerank_topk_cpu_seconds,
    )?;
    out.set_item(
        "timing_max6_permutation_rerank_other_task_sum_s",
        result.max6_permutation_timing.rerank_other_cpu_seconds,
    )?;
    out.set_item(
        "timing_max6_permutation_pair_score_count",
        result.max6_permutation_timing.pair_score_count,
    )?;
    out.set_item("max6_pair_count", result.max6_pair_count)?;
    out.set_item(
        "max6_threshold_method",
        max6_threshold_method_for_output(
            grm_int_only,
            result.max6_bonferroni_q_cut,
            result.max6_screen_threshold_method.as_deref(),
            result.max6_pairs_tsv.is_some(),
        ),
    )?;
    out.set_item("posterior_tsv", py.None())?;
    out.set_item("posterior_json", result.posterior_json)?;
    if let Some(debug) = result.memory_debug.as_ref() {
        out.set_item("memory_debug", garfield_memory_debug_to_pydict(py, debug)?)?;
    } else {
        out.set_item("memory_debug", py.None())?;
    }
    if let Some(summary) = result.bg_noise_summary.as_ref() {
        out.set_item(
            "bg_noise_summary",
            garfield_bg_noise_summary_to_pydict(py, summary)?,
        )?;
    } else {
        out.set_item("bg_noise_summary", py.None())?;
    }
    if let Some(summary) = result.scheduler_scan.as_ref() {
        out.set_item(
            "scheduler_scan",
            garfield_scheduler_task_summary_to_pydict(py, summary)?,
        )?;
    } else {
        out.set_item("scheduler_scan", py.None())?;
    }
    if let Some(summary) = result.scheduler_permutation.as_ref() {
        out.set_item(
            "scheduler_permutation",
            garfield_scheduler_task_summary_to_pydict(py, summary)?,
        )?;
    } else {
        out.set_item("scheduler_permutation", py.None())?;
    }
    out.set_item("rule_permutation_active", result.rule_permutation_active)?;
    let residualization_method = result.train_fit.residualization_method.clone();
    let rule_null_generation = rule_null_generation_for_output(
        grm_int_only,
        result.max6_screen_threshold_method.is_some(),
        result.max6_pairs_tsv.is_some(),
        residualization_method == "exact_lmm",
        residualization_method == "lm",
    );
    out.set_item("residualization_method", residualization_method)?;
    out.set_item("lambda_source", result.train_fit.lambda_source.clone())?;
    out.set_item("no_grm_boundary", result.train_fit.no_grm_boundary)?;
    out.set_item(
        "lambda_boundary_policy",
        if result.train_fit.no_grm_boundary {
            "explicit_no_grm_ols_projection"
        } else {
            "finite_reml_lambda"
        },
    )?;
    out.set_item("lambda_vc", result.train_fit.lbd)?;
    out.set_item("lambda_marker", result.train_fit.lambda_marker)?;
    out.set_item("residual_marker_count", result.train_fit.eff_m.unwrap_or(0))?;
    out.set_item("pcg_converged", result.train_fit.pcg_converged)?;
    out.set_item("pcg_iters", result.train_fit.pcg_iters)?;
    out.set_item("pcg_rel_res", result.train_fit.pcg_rel_res)?;
    out.set_item("lambda_converged", result.train_fit.lambda_converged)?;
    out.set_item("lambda_iters", result.train_fit.lambda_iters)?;
    out.set_item("lambda_rel_res", result.train_fit.lambda_rel_res)?;
    out.set_item("rule_null_generation", rule_null_generation)?;
    out.set_item("null_chunk_bp", result.null_chunk_bp)?;
    out.set_item("null_chunk_min_snps", result.null_chunk_min_snps)?;
    out.set_item("null_chunk_target", result.null_chunk_target)?;
    out.set_item("null_chunk_valid_total", result.null_chunk_valid_total)?;
    out.set_item("null_chunk_selected", result.null_chunk_selected)?;
    out.set_item(
        "representative_units_target",
        result.representative_units_target,
    )?;
    out.set_item(
        "representative_units_used",
        result.representative_units_used,
    )?;
    out.set_item("permutation_null_repeats", result.permutation_null_repeats)?;
    out.set_item(
        "permutation_bootstrap_repeats",
        result.permutation_bootstrap_repeats,
    )?;
    out.set_item("n_rules", result.records.len())?;
    out.set_item("n_simbench", result.simbench_count)?;
    out.set_item("evaluation_mode", "full_exploratory")?;
    out.set_item("n_samples", result.n_samples)?;
    out.set_item("min_cell", min_cell)?;
    out.set_item("min_cell_fraction", min_cell_fraction)?;
    out.set_item("min_cell_neff", result.n_samples)?;
    out.set_item(
        "min_cell_effective",
        min_cell::resolve_min_cell(min_cell, min_cell_fraction, result.n_samples)
            .map_err(PyValueError::new_err)?,
    )?;
    out.set_item("units_total", result.units_total)?;
    out.set_item("units_scanned", result.units_scanned)?;
    out.set_item("timing_total_wall_s", result.timing_total_wall_s)?;
    out.set_item("timing_scan_wall_s", result.timing_scan_wall_s)?;
    let batch_timing = &result.max6_observed_scan_timing;
    for (name, seconds) in [
        (
            "batch_prepare",
            batch_timing.screen_batch_prepare_wall_seconds,
        ),
        (
            "membership_prepare",
            batch_timing.screen_membership_prepare_wall_seconds,
        ),
        (
            "pair_scan_merge",
            batch_timing.screen_pair_scan_merge_wall_seconds,
        ),
        ("finalists", batch_timing.screen_finalist_wall_seconds),
        ("grm_int", batch_timing.grm_int_batch_wall_seconds),
        ("result_drain", batch_timing.scan_result_drain_wall_seconds),
    ] {
        out.set_item(format!("timing_max6_batch_{name}_wall_s"), seconds)?;
    }
    out.set_item(
        "timing_max6_batch_heap_merge_task_sum_s",
        batch_timing.screen_heap_merge_task_seconds,
    )?;
    out.set_item(
        "max6_batch_parallel_count",
        batch_timing.screen_parallel_batches,
    )?;
    out.set_item(
        "max6_batch_serial_count",
        batch_timing.screen_serial_batches,
    )?;
    out.set_item(
        "max6_batch_source_rows_sum",
        batch_timing.screen_source_rows_sum,
    )?;
    out.set_item(
        "max6_batch_source_rows_max",
        batch_timing.screen_source_rows_max,
    )?;
    out.set_item(
        "max6_batch_memberships_sum",
        batch_timing.screen_memberships_sum,
    )?;
    let batch_scan_profiles = batch_timing
        .screen_batch_profiles
        .iter()
        .map(|profile| {
            vec![
                profile.scan_merge_wall_seconds,
                profile.fold_task_count as f64,
                profile.fold_task_wall_sum_seconds,
                profile.fold_task_wall_max_seconds,
                profile.fold_anchor_count_min as f64,
                profile.fold_anchor_count_max as f64,
                profile.fold_active_workers as f64,
                profile.worker_task_wall_min_seconds,
                profile.worker_task_wall_median_seconds,
                profile.worker_task_wall_max_seconds,
                profile.worker_finish_skew_seconds,
                profile.post_last_fold_tail_seconds,
                profile.topk_merge_task_seconds,
                profile.topk_merge_max_task_seconds,
                profile.topk_merge_task_count as f64,
                profile.topk_merge_critical_path_seconds,
                profile.scan_reduce_critical_path_seconds,
            ]
        })
        .collect::<Vec<_>>();
    out.set_item("max6_batch_scan_profiles", batch_scan_profiles)?;
    if garfield_pair_scan_profile_enabled() {
        let pair_scan_subphase_profiles = batch_timing
            .screen_batch_profiles
            .iter()
            .map(|profile| {
                vec![
                    profile.pair_profile_sampled_anchors as f64,
                    profile.pair_profile_sampled_pairs as f64,
                    profile.pair_profile_sampled_scored_pairs as f64,
                    profile.pair_profile_sampled_score_offers as f64,
                    profile.pair_profile_cursor_merge_seconds,
                    profile.pair_profile_min_cell_gate_seconds,
                    profile.pair_profile_sufficient_stats_seconds,
                    profile.pair_profile_topk_offer_seconds,
                ]
            })
            .collect::<Vec<_>>();
        out.set_item(
            "max6_pair_scan_subphase_profiles",
            pair_scan_subphase_profiles,
        )?;
    }
    out.set_item(
        "max6_observed_units_timed",
        result.max6_observed_scan_timing.units_timed,
    )?;
    out.set_item(
        "max6_observed_screen_candidate_pairs",
        result.max6_observed_scan_timing.screen_candidate_pairs,
    )?;
    out.set_item(
        "max6_observed_screen_pairs_evaluated",
        result.max6_observed_scan_timing.screen_pairs_evaluated,
    )?;
    out.set_item(
        "max6_observed_screen_batch_count",
        result.max6_observed_scan_timing.screen_batch_count,
    )?;
    out.set_item(
        "max6_observed_screen_batch_unique_scored_pairs",
        result
            .max6_observed_scan_timing
            .screen_batch_pairs_scored_unique,
    )?;
    out.set_item(
        "max6_observed_screen_batch_score_offers",
        result.max6_observed_scan_timing.screen_batch_score_offers,
    )?;
    out.set_item(
        "max6_observed_screen_batch_direct_fallbacks",
        result
            .max6_observed_scan_timing
            .screen_batch_direct_fallbacks,
    )?;
    out.set_item(
        "max6_observed_exact_candidate_pairs",
        result.max6_observed_scan_timing.exact_candidate_pairs,
    )?;
    out.set_item(
        "max6_observed_grm_int_pair_occurrences",
        result.max6_observed_scan_timing.grm_int_pair_occurrences,
    )?;
    out.set_item(
        "timing_max6_observed_grm_int_task_sum_s",
        result.max6_observed_scan_timing.grm_int_task_seconds,
    )?;
    out.set_item(
        "max6_observed_grm_int_batches",
        result.max6_observed_scan_timing.grm_int_batches,
    )?;
    out.set_item(
        "max6_observed_grm_int_max_workspace_bytes",
        result.max6_observed_scan_timing.grm_int_max_workspace_bytes,
    )?;
    out.set_item(
        "max6_observed_grm_int_singular_endpoint_design_pairs",
        result
            .max6_observed_scan_timing
            .grm_int_singular_endpoint_design_pairs,
    )?;
    out.set_item(
        "max6_observed_grm_int_negligible_interaction_information_pairs",
        result
            .max6_observed_scan_timing
            .grm_int_negligible_interaction_information_pairs,
    )?;
    out.set_item(
        "max6_observed_grm_int_nonfinite_pairs",
        result.max6_observed_scan_timing.grm_int_nonfinite_pairs,
    )?;
    out.set_item(
        "timing_max6_observed_screen_input_prepare_task_sum_s",
        result
            .max6_observed_scan_timing
            .screen_input_prepare_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_screen_kernel_task_sum_s",
        result.max6_observed_scan_timing.screen_kernel_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_candidate_prepare_task_sum_s",
        result
            .max6_observed_scan_timing
            .candidate_prepare_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_endpoint_prepare_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_endpoint_prepare_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_kernel_task_sum_s",
        result.max6_observed_scan_timing.rerank_kernel_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_pcx_task_sum_s",
        result.max6_observed_scan_timing.rerank_pcx_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_pcx_sparse_support_units",
        result
            .max6_observed_scan_timing
            .rerank_pcx_sparse_support_units,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_geometry_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_geometry_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_geometry_marker_moments_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_geometry_marker_moments_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_geometry_pairwise_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_geometry_pairwise_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_au_validation_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_geometry_au_validation_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_au_allocation_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_geometry_au_allocation_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_au_materialization_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_geometry_au_materialization_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_au_gemm_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_geometry_au_gemm_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_au_candidate_lookup_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_geometry_au_candidate_lookup_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_au_fused_py_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_geometry_au_fused_py_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_au_copy_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_geometry_au_copy_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_au_max_workspace_bytes",
        result
            .max6_observed_scan_timing
            .rerank_geometry_au_max_workspace_bytes,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_geometry_lookup_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_geometry_lookup_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_geometry_interaction_py_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_geometry_interaction_py_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_geometry_unclassified_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_geometry_unclassified_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_q_task_sum_s",
        result.max6_observed_scan_timing.rerank_q_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_q_validation_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_q_validation_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_q_bitset_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_q_bitset_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_q_cell_task_sum_s",
        result.max6_observed_scan_timing.rerank_q_cell_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_q_enumerate_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_q_enumerate_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_q_quadratic_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_q_quadratic_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_q_reconstruct_task_sum_s",
        result
            .max6_observed_scan_timing
            .rerank_q_reconstruct_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_score_task_sum_s",
        result.max6_observed_scan_timing.rerank_score_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_topk_task_sum_s",
        result.max6_observed_scan_timing.rerank_topk_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_rerank_other_task_sum_s",
        result.max6_observed_scan_timing.rerank_other_task_seconds,
    )?;
    out.set_item(
        "timing_max6_observed_output_finalize_task_sum_s",
        result
            .max6_observed_scan_timing
            .output_finalize_task_seconds,
    )?;
    if result
        .max6_observed_scan_timing
        .candidate_reuse_diagnostic_enabled
    {
        let timing = &result.max6_observed_scan_timing;
        let diagnostic = PyDict::new(py);
        diagnostic.set_item("enabled", true)?;
        diagnostic.set_item(
            "screen_candidate_hash_window_count",
            timing.candidate_hash_window_count,
        )?;
        diagnostic.set_item(
            "screen_candidate_sets_hash_by_window_order_fnv1a64",
            format!("{:016x}", timing.candidate_set_hash_by_window_order),
        )?;
        diagnostic.set_item(
            "exact_topk_pair_sets_hash_by_window_order_fnv1a64",
            format!("{:016x}", timing.exact_topk_set_hash_by_window_order),
        )?;
        diagnostic.set_item(
            "exact_topk_order_hash_by_window_order_fnv1a64",
            format!("{:016x}", timing.exact_topk_order_hash_by_window_order),
        )?;
        diagnostic.set_item("window_calls", timing.units_timed)?;
        diagnostic.set_item(
            "screen_candidate_pair_occurrences",
            timing.screen_candidate_pairs,
        )?;
        diagnostic.set_item("screen_pairs_evaluated", timing.screen_pairs_evaluated)?;
        diagnostic.set_item("screen_batch_count", timing.screen_batch_count)?;
        diagnostic.set_item(
            "screen_batch_unique_scored_pairs",
            timing.screen_batch_pairs_scored_unique,
        )?;
        diagnostic.set_item(
            "screen_batch_score_offers",
            timing.screen_batch_score_offers,
        )?;
        diagnostic.set_item(
            "screen_batch_direct_fallbacks",
            timing.screen_batch_direct_fallbacks,
        )?;
        diagnostic.set_item(
            "screen_kernel_task_seconds",
            timing.screen_kernel_task_seconds,
        )?;
        diagnostic.set_item(
            "screen_kernel_nanoseconds_per_evaluated_pair",
            if timing.screen_pairs_evaluated == 0 {
                0.0
            } else {
                timing.screen_kernel_task_seconds * 1e9 / timing.screen_pairs_evaluated as f64
            },
        )?;
        diagnostic.set_item(
            "exact_scored_candidate_pair_occurrences",
            timing.exact_candidate_pairs,
        )?;
        diagnostic.set_item(
            "scored_candidate_pair_occurrences",
            timing.scored_candidate_pairs,
        )?;
        diagnostic.set_item("grm_int_pair_occurrences", timing.grm_int_pair_occurrences)?;
        diagnostic.set_item(
            "grm_int_singular_endpoint_design_pairs",
            timing.grm_int_singular_endpoint_design_pairs,
        )?;
        diagnostic.set_item(
            "grm_int_negligible_interaction_information_pairs",
            timing.grm_int_negligible_interaction_information_pairs,
        )?;
        diagnostic.set_item("grm_int_nonfinite_pairs", timing.grm_int_nonfinite_pairs)?;
        diagnostic.set_item("grm_int_batches", timing.grm_int_batches)?;
        diagnostic.set_item(
            "grm_int_max_workspace_bytes",
            timing.grm_int_max_workspace_bytes,
        )?;
        diagnostic.set_item("grm_int_task_seconds", timing.grm_int_task_seconds)?;
        diagnostic.set_item(
            "global_unique_scored_pairs",
            timing.candidate_unique_pairs_global,
        )?;
        diagnostic.set_item(
            "global_unique_exact_scored_pairs",
            if timing.grm_int_pair_occurrences == 0 {
                timing.candidate_unique_pairs_global
            } else {
                0
            },
        )?;
        diagnostic.set_item(
            "duplicate_pair_occurrences_across_windows",
            timing.candidate_duplicate_occurrences,
        )?;
        diagnostic.set_item(
            "duplicate_fraction_of_scored_occurrences",
            timing.candidate_duplicate_fraction,
        )?;
        diagnostic.set_item(
            "duplicate_fraction_of_exact_scored_occurrences",
            if timing.grm_int_pair_occurrences == 0 {
                timing.candidate_duplicate_fraction
            } else {
                0.0
            },
        )?;
        diagnostic.set_item("pair_multiplicity_p50", timing.candidate_multiplicity_p50)?;
        diagnostic.set_item("pair_multiplicity_p90", timing.candidate_multiplicity_p90)?;
        diagnostic.set_item("pair_multiplicity_p99", timing.candidate_multiplicity_p99)?;
        diagnostic.set_item("pair_multiplicity_max", timing.candidate_multiplicity_max)?;
        diagnostic.set_item(
            "exact_input_duplicate_pairs_within_windows",
            timing.exact_input_duplicate_pairs,
        )?;
        diagnostic.set_item(
            "exact_input_self_pairs_within_windows",
            timing.exact_input_self_pairs,
        )?;
        diagnostic.set_item(
            "screen_to_scored_pairs_removed_within_windows",
            timing
                .screen_candidate_pairs
                .saturating_sub(timing.scored_candidate_pairs),
        )?;
        diagnostic.set_item(
            "screen_to_exact_pairs_removed_within_windows",
            timing
                .screen_candidate_pairs
                .saturating_sub(timing.exact_candidate_pairs),
        )?;
        diagnostic.set_item(
            "unique_endpoint_occurrences_summed_over_windows",
            timing.candidate_unique_endpoints_window_sum,
        )?;
        diagnostic.set_item(
            "globally_unique_candidate_endpoints",
            timing.candidate_unique_endpoints_global,
        )?;
        diagnostic.set_item(
            "endpoint_reuse_factor",
            timing.candidate_endpoint_reuse_factor,
        )?;
        diagnostic.set_item(
            "candidate_key_collection_task_seconds",
            timing.candidate_key_collection_task_seconds,
        )?;
        diagnostic.set_item(
            "candidate_key_sort_and_summary_wall_seconds",
            timing.candidate_key_finalize_wall_seconds,
        )?;
        diagnostic.set_item(
            "candidate_pair_key_buffer_bytes",
            timing.candidate_key_buffer_bytes,
        )?;
        diagnostic.set_item(
            "raw_score_audit_enabled",
            timing.candidate_score_reuse_audit_enabled,
        )?;
        if let Some(summary) = timing.candidate_score_reuse_summary {
            let score_audit = PyDict::new(py);
            score_audit.set_item("comparison", "f64 bitwise equality")?;
            score_audit.set_item(
                "cached_score_policy",
                "reuse score from lowest scan-unit index for the same canonical pair",
            )?;
            score_audit.set_item("score_occurrences", summary.observation_count)?;
            score_audit.set_item("unique_pairs", summary.unique_pair_count)?;
            score_audit.set_item("repeated_pairs", summary.repeated_pair_count)?;
            score_audit.set_item("duplicate_occurrences", summary.duplicate_occurrence_count)?;
            score_audit.set_item(
                "bitwise_equal_repeated_pairs",
                summary.bitwise_equal_repeated_pair_count,
            )?;
            score_audit.set_item("score_mismatch_pairs", summary.score_mismatch_pair_count)?;
            score_audit.set_item(
                "best_channel_mismatch_pairs",
                summary.best_channel_mismatch_pair_count,
            )?;
            score_audit.set_item(
                "nonfinite_score_occurrences",
                summary.nonfinite_score_occurrence_count,
            )?;
            score_audit.set_item("max_abs_score_difference", summary.max_abs_score_difference)?;
            score_audit.set_item(
                "summary_wall_seconds",
                timing.candidate_score_audit_finalize_wall_seconds,
            )?;
            score_audit.set_item(
                "observation_buffer_bytes",
                timing.candidate_score_audit_buffer_bytes,
            )?;
            let topk_stability = PyList::empty(py);
            for stability in summary.topk_stability.iter() {
                let row = PyDict::new(py);
                row.set_item("top_k", stability.top_k)?;
                row.set_item("windows_compared", stability.windows_compared)?;
                row.set_item(
                    "windows_with_membership_change",
                    stability.windows_with_membership_change,
                )?;
                row.set_item(
                    "total_membership_symmetric_difference",
                    stability.total_membership_symmetric_difference,
                )?;
                row.set_item(
                    "max_membership_symmetric_difference",
                    stability.max_membership_symmetric_difference,
                )?;
                row.set_item(
                    "windows_with_order_change",
                    stability.windows_with_order_change,
                )?;
                topk_stability.append(row)?;
            }
            score_audit.set_item("first_seen_score_topk_stability", topk_stability)?;
            diagnostic.set_item("raw_score_audit", score_audit)?;
        } else {
            diagnostic.set_item("raw_score_audit", py.None())?;
        }
        let window_counts = PyDict::new(py);
        window_counts.set_item("scored_pairs_min", timing.candidate_window_pairs_min)?;
        window_counts.set_item("scored_pairs_p50", timing.candidate_window_pairs_p50)?;
        window_counts.set_item("scored_pairs_p90", timing.candidate_window_pairs_p90)?;
        window_counts.set_item("scored_pairs_p99", timing.candidate_window_pairs_p99)?;
        window_counts.set_item("scored_pairs_max", timing.candidate_window_pairs_max)?;
        let exact_window_counts = if timing.grm_int_pair_occurrences == 0 {
            [
                timing.candidate_window_pairs_min,
                timing.candidate_window_pairs_p50,
                timing.candidate_window_pairs_p90,
                timing.candidate_window_pairs_p99,
                timing.candidate_window_pairs_max,
            ]
        } else {
            [0; 5]
        };
        window_counts.set_item("exact_scored_pairs_min", exact_window_counts[0])?;
        window_counts.set_item("exact_scored_pairs_p50", exact_window_counts[1])?;
        window_counts.set_item("exact_scored_pairs_p90", exact_window_counts[2])?;
        window_counts.set_item("exact_scored_pairs_p99", exact_window_counts[3])?;
        window_counts.set_item("exact_scored_pairs_max", exact_window_counts[4])?;
        window_counts.set_item(
            "screen_pairs_evaluated_min",
            timing.candidate_window_screen_pairs_evaluated_min,
        )?;
        window_counts.set_item(
            "screen_pairs_evaluated_p50",
            timing.candidate_window_screen_pairs_evaluated_p50,
        )?;
        window_counts.set_item(
            "screen_pairs_evaluated_p90",
            timing.candidate_window_screen_pairs_evaluated_p90,
        )?;
        window_counts.set_item(
            "screen_pairs_evaluated_p99",
            timing.candidate_window_screen_pairs_evaluated_p99,
        )?;
        window_counts.set_item(
            "screen_pairs_evaluated_max",
            timing.candidate_window_screen_pairs_evaluated_max,
        )?;
        window_counts.set_item(
            "unique_endpoints_min",
            timing.candidate_window_endpoints_min,
        )?;
        window_counts.set_item(
            "unique_endpoints_p50",
            timing.candidate_window_endpoints_p50,
        )?;
        window_counts.set_item(
            "unique_endpoints_p90",
            timing.candidate_window_endpoints_p90,
        )?;
        window_counts.set_item(
            "unique_endpoints_p99",
            timing.candidate_window_endpoints_p99,
        )?;
        window_counts.set_item(
            "unique_endpoints_max",
            timing.candidate_window_endpoints_max,
        )?;
        diagnostic.set_item("per_window", window_counts)?;
        let profile = &timing.screen_profile;
        let profile_dict = PyDict::new(py);
        profile_dict.set_item("profiled_windows", profile.units)?;
        profile_dict.set_item("sampled_pairs", profile.sample_pairs)?;
        profile_dict.set_item("sampled_pair_universe", profile.total_pairs)?;
        profile_dict.set_item("sampled_u64_words", profile.sample_words)?;
        profile_dict.set_item("nonzero_intersection_words", profile.nonzero_intersections)?;
        profile_dict.set_item(
            "average_intersection_popcount_per_word",
            if profile.sample_words == 0 {
                0.0
            } else {
                profile.intersection_popcount_sum as f64 / profile.sample_words as f64
            },
        )?;
        profile_dict.set_item("weighted_y_lookup_calls", profile.lookup_calls)?;
        profile_dict.set_item("lut8_word_calls", profile.lut8_word_calls)?;
        profile_dict.set_item("setbit_calls", profile.setbit_calls)?;
        profile_dict.set_item("heap_pushes", profile.heap_pushes)?;
        profile_dict.set_item("heap_replacements", profile.heap_replacements)?;
        profile_dict.set_item("profile_replay_seconds", profile.replay_seconds)?;
        let profile_stages = PyDict::new(py);
        profile_stages.set_item(
            "bitset_intersection_popcount_seconds",
            profile.bitset_seconds,
        )?;
        profile_stages.set_item(
            "weighted_y_lookup_seconds",
            profile.weighted_y_lookup_seconds,
        )?;
        profile_stages.set_item("margin_recovery_seconds", profile.margin_recovery_seconds)?;
        profile_stages.set_item("score_assembly_seconds", profile.score_seconds)?;
        profile_stages.set_item("heap_topk_seconds", profile.heap_seconds)?;
        profile_stages.set_item("final_sort_seconds", profile.final_sort_seconds)?;
        profile_dict.set_item("stage_seconds", profile_stages)?;
        let profile_ns_per_pair = PyDict::new(py);
        let pairs = profile.sample_pairs as f64;
        profile_ns_per_pair.set_item(
            "bitset_intersection_popcount",
            if pairs == 0.0 {
                0.0
            } else {
                profile.bitset_seconds * 1e9 / pairs
            },
        )?;
        profile_ns_per_pair.set_item(
            "weighted_y_lookup",
            if pairs == 0.0 {
                0.0
            } else {
                profile.weighted_y_lookup_seconds * 1e9 / pairs
            },
        )?;
        profile_ns_per_pair.set_item(
            "margin_recovery",
            if pairs == 0.0 {
                0.0
            } else {
                profile.margin_recovery_seconds * 1e9 / pairs
            },
        )?;
        profile_ns_per_pair.set_item(
            "score_assembly",
            if pairs == 0.0 {
                0.0
            } else {
                profile.score_seconds * 1e9 / pairs
            },
        )?;
        profile_ns_per_pair.set_item(
            "heap_topk",
            if pairs == 0.0 {
                0.0
            } else {
                profile.heap_seconds * 1e9 / pairs
            },
        )?;
        profile_ns_per_pair.set_item(
            "final_sort",
            if pairs == 0.0 {
                0.0
            } else {
                profile.final_sort_seconds * 1e9 / pairs
            },
        )?;
        profile_dict.set_item("stage_ns_per_sampled_pair", profile_ns_per_pair)?;
        diagnostic.set_item("screen_microprofile", profile_dict)?;
        out.set_item("max6_candidate_reuse_diagnostic", diagnostic)?;
    } else {
        out.set_item("max6_candidate_reuse_diagnostic", py.None())?;
    }
    out.set_item(
        "timing_max6_screen_calibration_wall_s",
        result.timing_max6_screen_calibration_wall_s,
    )?;
    out.set_item(
        "timing_max6_screen_validation_wall_s",
        result.timing_max6_screen_validation_wall_s,
    )?;
    out.set_item(
        "timing_scan_ml_select_wall_s",
        result.timing_scan_ml_select_wall_s,
    )?;
    out.set_item(
        "timing_corr_stage1_summary_s",
        result.timing_corr_stage1_summary_s,
    )?;
    out.set_item(
        "timing_corr_stage1_score_s",
        result.timing_corr_stage1_score_s,
    )?;
    out.set_item(
        "timing_corr_stage1_pool_s",
        result.timing_corr_stage1_pool_s,
    )?;
    out.set_item(
        "timing_corr_stage1_pool_priority_s",
        result.timing_corr_stage1_pool_priority_s,
    )?;
    out.set_item(
        "timing_corr_stage1_pool_ld_s",
        result.timing_corr_stage1_pool_ld_s,
    )?;
    out.set_item(
        "timing_ld_support_cache_s",
        result.timing_ld_support_cache_s,
    )?;
    out.set_item("ld_support_cache_rows", result.ld_support_cache_rows)?;
    out.set_item(
        "ld_support_cache_payload_bytes",
        result.ld_support_cache_payload_bytes,
    )?;
    out.set_item("timing_scan_beam_wall_s", result.timing_scan_beam_wall_s)?;
    out.set_item(
        "timing_scan_literal_score_wall_s",
        result.timing_scan_literal_score_wall_s,
    )?;
    out.set_item("timing_scan_beam_calls", result.timing_scan_beam_calls)?;
    out.set_item("timing_clone_bits_s", result.timing_clone_bits_s)?;
    out.set_item("timing_sum_y_both1_s", result.timing_sum_y_both1_s)?;
    out.set_item("timing_parent_baseline_s", result.timing_parent_baseline_s)?;
    out.set_item("timing_pw_marginal_s", result.timing_pw_marginal_s)?;
    out.set_item("timing_pw_pack_s", result.timing_pw_pack_s)?;
    out.set_item("timing_pw_kernel_s", result.timing_pw_kernel_s)?;
    out.set_item("timing_pw_combine_s", result.timing_pw_combine_s)?;
    out.set_item("timing_dense_extract_s", result.timing_dense_extract_s)?;
    out.set_item("timing_dense_decode_s", result.timing_dense_decode_s)?;
    out.set_item(
        "timing_geneset_ld_prune_s",
        result.timing_geneset_ld_prune_s,
    )?;
    out.set_item("ld_exact_pairs", result.ld_exact_pairs)?;
    out.set_item("ld_clump_r2", garfield_ld_clump_r2())?;
    out.set_item("ld_rows_total", result.ld_rows_total)?;
    out.set_item("ld_rows_kept", result.ld_rows_kept)?;
    out.set_item("ld_clump_rows_compressed", result.ld_clump_rows_compressed)?;
    out.set_item("ld_clump_units", result.ld_clump_units)?;
    out.set_item("ld_units_eligible", result.ld_units_eligible)?;
    out.set_item("ld_units_pruned", result.ld_units_pruned)?;
    out.set_item("ld_rows_pruned", result.ld_rows_pruned)?;
    out.set_item(
        "ld_unit_rows_total_median",
        result.ld_unit_rows_total_median,
    )?;
    out.set_item("ld_unit_rows_total_p95", result.ld_unit_rows_total_p95)?;
    out.set_item("ld_unit_rows_total_max", result.ld_unit_rows_total_max)?;
    out.set_item("ld_unit_rows_kept_median", result.ld_unit_rows_kept_median)?;
    out.set_item("ld_unit_rows_kept_p95", result.ld_unit_rows_kept_p95)?;
    out.set_item("ld_unit_rows_kept_max", result.ld_unit_rows_kept_max)?;
    out.set_item(
        "ld_unit_rows_pruned_median",
        result.ld_unit_rows_pruned_median,
    )?;
    out.set_item("ld_unit_rows_pruned_p95", result.ld_unit_rows_pruned_p95)?;
    out.set_item("ld_unit_rows_pruned_max", result.ld_unit_rows_pruned_max)?;
    out.set_item(
        "ld_unit_exact_pairs_median",
        result.ld_unit_exact_pairs_median,
    )?;
    out.set_item("ld_unit_exact_pairs_p95", result.ld_unit_exact_pairs_p95)?;
    out.set_item("ld_unit_exact_pairs_max", result.ld_unit_exact_pairs_max)?;
    out.set_item(
        "timing_materialize_bits_s",
        result.timing_materialize_bits_s,
    )?;
    out.set_item(
        "timing_literal_score_share_of_total_pct",
        result.timing_literal_score_share_of_total_pct,
    )?;
    out.set_item(
        "timing_literal_score_share_of_scan_pct",
        result.timing_literal_score_share_of_scan_pct,
    )?;
    out.set_item(
        "timing_literal_score_share_of_beam_pct",
        result.timing_literal_score_share_of_beam_pct,
    )?;
    out.set_item(
        "timing_beam_share_of_total_pct",
        result.timing_beam_share_of_total_pct,
    )?;
    out.set_item(
        "timing_beam_share_of_scan_pct",
        result.timing_beam_share_of_scan_pct,
    )?;
    let skipped_units_py = PyList::empty(py);
    for skipped in result.skipped_units.iter() {
        skipped_units_py.append(garfield_skipped_unit_to_pydict(py, skipped)?)?;
    }
    out.set_item("n_skipped_units", result.skipped_units.len())?;
    out.set_item("skipped_units", skipped_units_py)?;
    out.set_item("skipped_messages", result.skipped_messages.clone())?;
    out.set_item("full_n", result.n_samples)?;
    out.set_item("logic_bits_backend", result.logic_bits_backend.clone())?;
    out.set_item("logic_bits_bytes", result.logic_bits_bytes)?;
    out.set_item("full_pve", result.train_fit.pve)?;
    out.set_item("full_sigma_g2", result.train_fit.sigma_g2)?;
    out.set_item("full_sigma_e2", result.train_fit.sigma_e2)?;
    out.set_item("full_eigh_backend", result.train_fit.eigh_backend.clone())?;
    out.set_item(
        "snp_names",
        result
            .records
            .iter()
            .map(|r| r.snp_name.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "expressions",
        result
            .records
            .iter()
            .map(|r| r.expr.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "unit_names",
        result
            .records
            .iter()
            .map(|r| r.unit_name.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "unit_kinds",
        result
            .records
            .iter()
            .map(|r| r.unit_kind.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "scores",
        result.records.iter().map(|r| r.score).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "train_scores",
        result.records.iter().map(|r| r.score).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "test_scores",
        result.records.iter().map(|r| r.score).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "positions",
        result.records.iter().map(|r| r.pos).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "selected_row_indices",
        result
            .records
            .iter()
            .map(|r| r.selected_row_indices.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "ml_ranks",
        result
            .records
            .iter()
            .map(|r| r.ml_rank.clone())
            .collect::<Vec<_>>(),
    )?;
    Ok(out)
}

#[pyfunction(name = "load_bin01_packed")]
pub fn load_bin01_packed_py<'py>(
    py: Python<'py>,
    path: String,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<u64>>, usize)> {
    let (packed, group_ids, n_samples, n_rows, row_bytes) = py
        .detach(move || load_bin01_packed_owned(&path))
        .map_err(PyRuntimeError::new_err)?;
    let packed_mat = Array2::from_shape_vec((n_rows, row_bytes), packed)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let gid_arr = Array1::from_vec(group_ids);
    #[allow(deprecated)]
    let packed_arr = PyArray2::from_owned_array(py, packed_mat).into_bound();
    #[allow(deprecated)]
    let gid_arr = PyArray1::from_owned_array(py, gid_arr).into_bound();
    Ok((packed_arr, gid_arr, n_samples))
}

#[pyfunction(name = "garfield_prepare_input_bin")]
#[pyo3(signature = (
    input_path,
    out_bin_path,
    input_kind="auto",
    mode="bin",
    threads=0,
    maf=0.0,
    geno=1.0,
    impute=false,
    het=1.0,
    sample_ids=None,
    sample_indices=None
))]
pub fn garfield_prepare_input_bin_py(
    input_path: String,
    out_bin_path: String,
    input_kind: &str,
    mode: &str,
    threads: usize,
    maf: f64,
    geno: f64,
    impute: bool,
    het: f64,
    sample_ids: Option<Vec<String>>,
    sample_indices: Option<Vec<usize>>,
) -> PyResult<(usize, usize, usize, usize)> {
    if !(0.0..=0.5).contains(&maf) {
        return Err(PyValueError::new_err("maf must be within [0, 0.5]"));
    }
    if !(0.0..=1.0).contains(&geno) {
        return Err(PyValueError::new_err("geno must be within [0, 1]"));
    }
    if !(0.0..=1.0).contains(&het) {
        return Err(PyValueError::new_err("het must be within [0, 1]"));
    }

    let input_kind = parse_input_kind(input_kind).map_err(PyValueError::new_err)?;
    let mode = parse_bin_mode(mode).map_err(PyValueError::new_err)?;
    let mut reader = make_input_reader(&input_path, input_kind).map_err(PyRuntimeError::new_err)?;

    let (selected_indices, selected_ids) =
        build_sample_selection(reader.sample_ids(), sample_ids, sample_indices)
            .map_err(PyRuntimeError::new_err)?;
    if selected_indices.is_empty() {
        return Err(PyValueError::new_err("selected sample set is empty"));
    }
    write_sample_id_sidecar(&out_bin_path, &selected_ids).map_err(PyRuntimeError::new_err)?;

    let n_samples = selected_indices.len();
    let identity_selection = selected_indices
        .iter()
        .enumerate()
        .all(|(i, &idx)| i == idx);
    let row_bytes = row_bytes_for_samples(n_samples);
    let mut writer = Bin01Writer::new(&out_bin_path, n_samples, Bin01SiteMode::TextTsv)
        .map_err(PyRuntimeError::new_err)?;

    let available_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1usize);
    let n_threads = if threads == 0 {
        available_threads
    } else {
        threads.min(available_threads).max(1)
    };
    let pool = if n_threads > 1 {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("build thread pool: {e}")))?,
        )
    } else {
        None
    };
    let pool_ref = pool.as_ref();

    let maf_f32 = maf as f32;
    let geno_f32 = geno as f32;
    let het_f32 = het as f32;

    let mut n_sites_seen = 0usize;
    let mut n_sites_written = 0usize;
    let mut n_rows_written = 0usize;

    match &mut reader {
        GarfieldInputReader::Bed(bed) => {
            if n_threads > 1 {
                let n_sites = bed.sites.len();
                let target_bytes = 16usize * 1024 * 1024;
                let mut chunk_size = target_bytes / n_samples.max(1);
                if chunk_size == 0 {
                    chunk_size = 1;
                }
                if chunk_size > 4096 {
                    chunk_size = 4096;
                }
                for start in (0..n_sites).step_by(chunk_size) {
                    let end = (start + chunk_size).min(n_sites);
                    let rows_res: Vec<Result<Option<Vec<EncodedRow>>, String>> = pool_ref
                        .expect("thread pool exists when n_threads>1")
                        .install(|| {
                            (start..end)
                                .into_par_iter()
                                .map(|idx| {
                                    let (row, site) = if identity_selection {
                                        bed.get_snp_row_raw(idx).ok_or_else(|| {
                                            format!("BED decode failed at row {}", idx)
                                        })?
                                    } else {
                                        bed.get_snp_row_selected_raw(idx, &selected_indices)
                                            .ok_or_else(|| {
                                                format!("BED decode with sample selection failed at row {}", idx)
                                            })?
                                    };
                                    Ok(encode_row_to_bits(
                                        row,
                                        site,
                                        mode,
                                        row_bytes,
                                        maf_f32,
                                        geno_f32,
                                        impute,
                                        het_f32,
                                    ))
                                })
                                .collect()
                        });

                    for item in rows_res.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        let maybe_rows = item.map_err(PyRuntimeError::new_err)?;
                        if let Some(rows) = maybe_rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows.len());
                            write_encoded_rows_bin(&mut writer, &rows)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            } else {
                while let Some((row, site)) = if identity_selection {
                    bed.next_snp_raw()
                } else {
                    bed.next_snp_selected_raw(&selected_indices)
                } {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows) = encode_row_to_bits(
                        row, site, mode, row_bytes, maf_f32, geno_f32, impute, het_f32,
                    ) {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows.len());
                        write_encoded_rows_bin(&mut writer, &rows)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
        GarfieldInputReader::Vcf(vcf) => {
            let mut batch: Vec<(Vec<f32>, SiteInfo)> = Vec::with_capacity(1024);
            loop {
                let Some((row_raw, site)) = vcf.next_snp_raw() else {
                    break;
                };
                let row = if identity_selection {
                    row_raw
                } else {
                    row_select_by_indices(row_raw, &selected_indices)
                        .map_err(PyRuntimeError::new_err)?
                };
                batch.push((row, site));
                if batch.len() >= 1024 {
                    let results = encode_batch_rows(
                        std::mem::take(&mut batch),
                        mode,
                        row_bytes,
                        maf_f32,
                        geno_f32,
                        impute,
                        het_f32,
                        pool_ref,
                    );
                    for rows in results.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        if let Some(rows_kept) = rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                            write_encoded_rows_bin(&mut writer, &rows_kept)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            }
            if !batch.is_empty() {
                let results = encode_batch_rows(
                    std::mem::take(&mut batch),
                    mode,
                    row_bytes,
                    maf_f32,
                    geno_f32,
                    impute,
                    het_f32,
                    pool_ref,
                );
                for rows in results.into_iter() {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows_kept) = rows {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                        write_encoded_rows_bin(&mut writer, &rows_kept)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
        GarfieldInputReader::Hmp(hmp) => {
            let mut batch: Vec<(Vec<f32>, SiteInfo)> = Vec::with_capacity(1024);
            loop {
                let Some((row_raw, site)) = hmp.next_snp_raw() else {
                    break;
                };
                let row = if identity_selection {
                    row_raw
                } else {
                    row_select_by_indices(row_raw, &selected_indices)
                        .map_err(PyRuntimeError::new_err)?
                };
                batch.push((row, site));
                if batch.len() >= 1024 {
                    let results = encode_batch_rows(
                        std::mem::take(&mut batch),
                        mode,
                        row_bytes,
                        maf_f32,
                        geno_f32,
                        impute,
                        het_f32,
                        pool_ref,
                    );
                    for rows in results.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        if let Some(rows_kept) = rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                            write_encoded_rows_bin(&mut writer, &rows_kept)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            }
            if !batch.is_empty() {
                let results = encode_batch_rows(
                    std::mem::take(&mut batch),
                    mode,
                    row_bytes,
                    maf_f32,
                    geno_f32,
                    impute,
                    het_f32,
                    pool_ref,
                );
                for rows in results.into_iter() {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows_kept) = rows {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                        write_encoded_rows_bin(&mut writer, &rows_kept)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
        GarfieldInputReader::Txt(txt) => {
            let mut batch: Vec<(Vec<f32>, SiteInfo)> = Vec::with_capacity(1024);
            while let Some((row_raw, site)) = txt.next_snp() {
                let row = if identity_selection {
                    row_raw
                } else {
                    row_select_by_indices(row_raw, &selected_indices)
                        .map_err(PyRuntimeError::new_err)?
                };
                batch.push((row, site));
                if batch.len() >= 1024 {
                    let results = encode_batch_rows(
                        std::mem::take(&mut batch),
                        mode,
                        row_bytes,
                        maf_f32,
                        geno_f32,
                        impute,
                        het_f32,
                        pool_ref,
                    );
                    for rows in results.into_iter() {
                        n_sites_seen = n_sites_seen.saturating_add(1);
                        if let Some(rows_kept) = rows {
                            n_sites_written = n_sites_written.saturating_add(1);
                            n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                            write_encoded_rows_bin(&mut writer, &rows_kept)
                                .map_err(PyRuntimeError::new_err)?;
                        }
                    }
                }
            }
            if !batch.is_empty() {
                let results = encode_batch_rows(
                    std::mem::take(&mut batch),
                    mode,
                    row_bytes,
                    maf_f32,
                    geno_f32,
                    impute,
                    het_f32,
                    pool_ref,
                );
                for rows in results.into_iter() {
                    n_sites_seen = n_sites_seen.saturating_add(1);
                    if let Some(rows_kept) = rows {
                        n_sites_written = n_sites_written.saturating_add(1);
                        n_rows_written = n_rows_written.saturating_add(rows_kept.len());
                        write_encoded_rows_bin(&mut writer, &rows_kept)
                            .map_err(PyRuntimeError::new_err)?;
                    }
                }
            }
        }
    }

    let header_rows = writer.finish().map_err(PyRuntimeError::new_err)?;
    if header_rows != n_rows_written {
        return Err(PyRuntimeError::new_err(format!(
            "internal row count mismatch: header_rows={}, tracked_rows={}",
            header_rows, n_rows_written
        )));
    }

    Ok((n_sites_seen, n_sites_written, n_rows_written, n_samples))
}

#[pyfunction(name = "garfield_subset_bin_samples")]
pub fn garfield_subset_bin_samples_py(
    bin_path: String,
    out_bin_path: String,
    sample_indices: Vec<usize>,
) -> PyResult<()> {
    let ctx = "garfield_subset_bin_samples";
    if sample_indices.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{ctx}: sample_indices is empty"
        )));
    }
    let bytes = load_file_owned(Path::new(&bin_path))
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: {e}")))?;
    let (n_rows, n_samples, row_bytes, data_offset) =
        parse_bin01_header(&bytes, ctx).map_err(PyRuntimeError::new_err)?;
    for &si in sample_indices.iter() {
        if si >= n_samples {
            return Err(PyValueError::new_err(format!(
                "{ctx}: sample index out of range {} >= {}",
                si, n_samples
            )));
        }
    }

    let n_out = sample_indices.len();
    let row_bytes_out = n_out.div_ceil(8);
    let payload_len = n_rows
        .checked_mul(row_bytes_out)
        .ok_or_else(|| PyRuntimeError::new_err(format!("{ctx}: payload overflow")))?;
    let header = bin01_header_bytes(n_rows as u64, n_out);
    let mut out = Vec::<u8>::with_capacity(header.len() + payload_len);
    out.extend_from_slice(&header);

    for r in 0..n_rows {
        let src_start =
            data_offset
                .checked_add(r.checked_mul(row_bytes).ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{ctx}: row offset overflow"))
                })?)
                .ok_or_else(|| PyRuntimeError::new_err(format!("{ctx}: row offset overflow")))?;
        let src_end = src_start
            .checked_add(row_bytes)
            .ok_or_else(|| PyRuntimeError::new_err(format!("{ctx}: row end overflow")))?;
        let src = &bytes[src_start..src_end];

        let mut row = vec![0u8; row_bytes_out];
        for (dst_i, &src_i) in sample_indices.iter().enumerate() {
            let src_b = src[src_i >> 3];
            let bit = (src_b >> (src_i & 7)) & 1u8;
            if bit != 0 {
                row[dst_i >> 3] |= 1u8 << (dst_i & 7);
            }
        }
        out.extend_from_slice(&row);
    }

    let mut fw = File::create(&out_bin_path)
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: create {out_bin_path}: {e}")))?;
    fw.write_all(&out)
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: write {out_bin_path}: {e}")))?;
    fw.flush()
        .map_err(|e| PyRuntimeError::new_err(format!("{ctx}: flush {out_bin_path}: {e}")))?;

    copy_site_sidecar(&bin_path, &out_bin_path).map_err(PyRuntimeError::new_err)?;
    let wrote_subset_ids = write_subset_id_sidecar(&bin_path, &out_bin_path, &sample_indices)
        .map_err(PyRuntimeError::new_err)?;
    if !wrote_subset_ids {
        copy_id_sidecar(&bin_path, &out_bin_path).map_err(PyRuntimeError::new_err)?;
    }
    Ok(())
}

#[pyfunction(name = "garfield_scan_groups_bin")]
#[pyo3(signature = (
    bin_path,
    y_train,
    groups,
    response="continuous",
    max_pick=4,
    beam_width=100,
    enforce_feature_exclusion=true,
    progress_callback=None,
    progress_every=0
))]
pub fn garfield_scan_groups_bin_py(
    bin_path: String,
    y_train: PyReadonlyArray1<'_, f64>,
    groups: Vec<Vec<(String, i32, i32)>>,
    response: &str,
    max_pick: usize,
    beam_width: usize,
    enforce_feature_exclusion: bool,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<Vec<(usize, usize, f64, Vec<usize>)>> {
    if groups.is_empty() {
        return Ok(Vec::new());
    }
    let resp = parse_response(response).map_err(PyValueError::new_err)?;
    let y_vec = read_y_f64(&y_train);
    let y_bin_owned = if resp == GarfieldResponse::Binary {
        Some(validate_binary_y(&y_vec).map_err(PyValueError::new_err)?)
    } else {
        validate_continuous_y(&y_vec).map_err(PyValueError::new_err)?;
        None
    };
    let y_bin = y_bin_owned.as_deref();

    let (bits_flat_all, row_words, n_rows_all, n_samples) =
        load_bin01_as_u64_words(&bin_path, "garfield_scan_groups_bin")
            .map_err(PyRuntimeError::new_err)?;
    if y_vec.len() < n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_scan_groups_bin: y length={} smaller than n_samples={}",
            y_vec.len(),
            n_samples
        )));
    }

    let sites = TxtSnpIter::new(&bin_path, None)
        .map_err(PyRuntimeError::new_err)?
        .sites;
    let chrom_idx = build_chrom_index(&sites, n_rows_all);
    let feature_group_ids_all = if enforce_feature_exclusion {
        Some(build_feature_group_ids(&sites, n_rows_all))
    } else {
        None
    };
    let total_groups = groups.len();
    let notify_step = if progress_every == 0 {
        (total_groups / 200).max(1)
    } else {
        progress_every.max(1)
    };
    let mut last_notified = 0usize;
    garfield_progress_notify(progress_callback.as_ref(), 0, total_groups)?;

    let mut out: Vec<(usize, usize, f64, Vec<usize>)> = Vec::with_capacity(groups.len());
    for (gi, group) in groups.iter().enumerate() {
        let mut idx_all: Vec<usize> = Vec::new();
        for (chrom, start, end) in group.iter() {
            let mut iv = interval_indices(&chrom_idx, chrom, *start, *end);
            idx_all.append(&mut iv);
        }
        if idx_all.is_empty() {
            out.push((gi, 0usize, f64::NEG_INFINITY, Vec::new()));
            let done = gi + 1;
            if done - last_notified >= notify_step || done == total_groups {
                garfield_progress_notify(progress_callback.as_ref(), done, total_groups)?;
                last_notified = done;
            }
            continue;
        }
        idx_all.sort_unstable();
        idx_all.dedup();

        let n_rows = idx_all.len();
        let mut bits_flat = vec![0u64; n_rows * row_words];
        for (ri, &src_idx) in idx_all.iter().enumerate() {
            let src_start = src_idx * row_words;
            let dst_start = ri * row_words;
            bits_flat[dst_start..dst_start + row_words]
                .copy_from_slice(&bits_flat_all[src_start..src_start + row_words]);
        }

        let local_group_ids = if let Some(global_ids) = feature_group_ids_all.as_ref() {
            Some(
                idx_all
                    .iter()
                    .map(|&gi| global_ids[gi])
                    .collect::<Vec<usize>>(),
            )
        } else {
            None
        };
        let res = run_beam_with_feature_exclusion(
            &bits_flat,
            row_words,
            n_rows,
            n_samples,
            resp,
            y_vec.as_slice(),
            y_bin,
            max_pick,
            beam_width,
            local_group_ids.as_deref(),
        )
        .map_err(PyRuntimeError::new_err)?;
        let selected_global = res
            .selected_indices
            .iter()
            .map(|&local_idx| idx_all[local_idx])
            .collect::<Vec<_>>();
        out.push((gi, idx_all.len(), res.score, selected_global));
        let done = gi + 1;
        if done - last_notified >= notify_step || done == total_groups {
            garfield_progress_notify(progress_callback.as_ref(), done, total_groups)?;
            last_notified = done;
        }
    }
    Ok(out)
}

#[pyfunction(name = "garfield_scan_windows_bin")]
#[pyo3(signature = (
    bin_path,
    y_train,
    response="continuous",
    max_pick=4,
    beam_width=100,
    extension=50000,
    step=None,
    enforce_feature_exclusion=true,
    progress_callback=None,
    progress_every=0
))]
pub fn garfield_scan_windows_bin_py(
    bin_path: String,
    y_train: PyReadonlyArray1<'_, f64>,
    response: &str,
    max_pick: usize,
    beam_width: usize,
    extension: usize,
    step: Option<usize>,
    enforce_feature_exclusion: bool,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<Vec<(usize, String, i32, i32, usize, f64, Vec<usize>)>> {
    let resp = parse_response(response).map_err(PyValueError::new_err)?;
    if extension == 0 {
        return Err(PyValueError::new_err(
            "garfield_scan_windows_bin: extension must be > 0",
        ));
    }
    let step_v = step.unwrap_or((extension / 2).max(1));
    if step_v == 0 {
        return Err(PyValueError::new_err(
            "garfield_scan_windows_bin: step must be > 0",
        ));
    }

    let y_vec = read_y_f64(&y_train);
    let y_bin_owned = if resp == GarfieldResponse::Binary {
        Some(validate_binary_y(&y_vec).map_err(PyValueError::new_err)?)
    } else {
        validate_continuous_y(&y_vec).map_err(PyValueError::new_err)?;
        None
    };
    let y_bin = y_bin_owned.as_deref();

    let (bits_flat_all, row_words, n_rows_all, n_samples) =
        load_bin01_as_u64_words(&bin_path, "garfield_scan_windows_bin")
            .map_err(PyRuntimeError::new_err)?;
    if y_vec.len() < n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_scan_windows_bin: y length={} smaller than n_samples={}",
            y_vec.len(),
            n_samples
        )));
    }

    let sites = TxtSnpIter::new(&bin_path, None)
        .map_err(PyRuntimeError::new_err)?
        .sites;
    let windows = build_windows_from_sites(&sites, n_rows_all, extension, step_v, normalize_chrom);
    if windows.is_empty() {
        return Ok(Vec::new());
    }
    let total_windows = windows.len();
    let notify_step = if progress_every == 0 {
        (total_windows / 200).max(1)
    } else {
        progress_every.max(1)
    };
    let mut last_notified = 0usize;
    garfield_progress_notify(progress_callback.as_ref(), 0, total_windows)?;
    let feature_group_ids_all = if enforce_feature_exclusion {
        Some(build_feature_group_ids(&sites, n_rows_all))
    } else {
        None
    };

    let mut out: Vec<(usize, String, i32, i32, usize, f64, Vec<usize>)> =
        Vec::with_capacity(windows.len());
    for (wi0, win) in windows.iter().enumerate() {
        if win.indices.is_empty() {
            let done = wi0 + 1;
            if done - last_notified >= notify_step || done == total_windows {
                garfield_progress_notify(progress_callback.as_ref(), done, total_windows)?;
                last_notified = done;
            }
            continue;
        }
        let n_rows = win.indices.len();
        let sub = if n_rows == n_rows_all {
            bits_flat_all.clone()
        } else {
            gather_rows_by_indices(
                &bits_flat_all,
                row_words,
                &win.indices,
                "garfield_scan_windows_bin::window_bits",
            )
            .map_err(PyRuntimeError::new_err)?
        };

        let local_group_ids = if let Some(global_ids) = feature_group_ids_all.as_ref() {
            Some(
                win.indices
                    .iter()
                    .map(|&gi| global_ids[gi])
                    .collect::<Vec<usize>>(),
            )
        } else {
            None
        };

        let res = run_beam_with_feature_exclusion(
            &sub,
            row_words,
            n_rows,
            n_samples,
            resp,
            y_vec.as_slice(),
            y_bin,
            max_pick,
            beam_width,
            local_group_ids.as_deref(),
        )
        .map_err(PyRuntimeError::new_err)?;

        let selected_global = res
            .selected_indices
            .iter()
            .map(|&local_idx| win.indices[local_idx])
            .collect::<Vec<_>>();
        out.push((
            wi0 + 1,
            win.chrom.clone(),
            win.bp_start,
            win.bp_end,
            n_rows,
            res.score,
            selected_global,
        ));
        let done = wi0 + 1;
        if done - last_notified >= notify_step || done == total_windows {
            garfield_progress_notify(progress_callback.as_ref(), done, total_windows)?;
            last_notified = done;
        }
    }
    Ok(out)
}

#[pyfunction(name = "garfield_eval_rule_bin")]
#[pyo3(signature = (bin_path, y, snp_indices, response="continuous"))]
pub fn garfield_eval_rule_bin_py(
    bin_path: String,
    y: PyReadonlyArray1<'_, f64>,
    snp_indices: Vec<usize>,
    response: &str,
) -> PyResult<(f64, usize, Vec<u8>)> {
    let resp = parse_response(response).map_err(PyValueError::new_err)?;
    if snp_indices.is_empty() {
        return Ok((f64::NEG_INFINITY, 0usize, Vec::new()));
    }
    let y_vec = read_y_f64(&y);
    let y_bin = if resp == GarfieldResponse::Binary {
        Some(validate_binary_y(&y_vec).map_err(PyValueError::new_err)?)
    } else {
        validate_continuous_y(&y_vec).map_err(PyValueError::new_err)?;
        None
    };

    let (bits_flat, row_words, n_rows, n_samples) =
        load_bin01_selected_rows_as_u64_words(&bin_path, &snp_indices, "garfield_eval_rule_bin")
            .map_err(PyRuntimeError::new_err)?;
    if n_rows == 0 {
        return Ok((f64::NEG_INFINITY, 0usize, Vec::new()));
    }
    if y_vec.len() < n_samples {
        return Err(PyValueError::new_err(format!(
            "garfield_eval_rule_bin: y length={} smaller than n_samples={}",
            y_vec.len(),
            n_samples
        )));
    }

    let needed_words = words_for_samples(n_samples);
    let mut combined = bits_flat[0..needed_words].to_vec();
    for r in 1..n_rows {
        let st = r * row_words;
        let row = &bits_flat[st..st + needed_words];
        for (a, &b) in combined.iter_mut().zip(row.iter()) {
            *a &= b;
        }
    }
    apply_tail_mask(&mut combined, tail_mask(n_samples));

    let score = match resp {
        GarfieldResponse::Binary => {
            let yb = y_bin.as_ref().expect("binary y prepared");
            score_binary_mcc_packed(yb.as_slice(), &combined, n_samples)
        }
        GarfieldResponse::Continuous => {
            score_cont_corr_packed(y_vec.as_slice(), &combined, n_samples).abs()
        }
    };
    let support = combined
        .iter()
        .map(|w| w.count_ones() as usize)
        .sum::<usize>();
    let mut xcombine = vec![0u8; n_samples];
    for i in 0..n_samples {
        let bit = (combined[i >> 6] >> (i & 63)) & 1u64;
        xcombine[i] = if bit != 0 { 1u8 } else { 0u8 };
    }
    Ok((score, support, xcombine))
}
