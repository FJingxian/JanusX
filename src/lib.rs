use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;
use std::path::PathBuf;

// stats
#[path = "stats/algwas.rs"]
mod algwas;
#[path = "stats/bayes.rs"]
mod bayes;
#[path = "stats/bayesr.rs"]
mod bayesr;
#[path = "stats/bsa.rs"]
mod bsa;
#[path = "stats/bstats.rs"]
mod bstats;
#[path = "ct/mod.rs"]
mod ct;
#[path = "decode/decode.rs"]
mod decode;
#[path = "stats/farmcpu.rs"]
mod farmcpu;
#[path = "stats/fst.rs"]
mod fst;
#[path = "stats/fvlmm.rs"]
mod fvlmm;

#[path = "garfield/mod.rs"]
pub mod garfield;
#[path = "stats/gblup.rs"]
pub(crate) mod gblup;
#[path = "stats/glm.rs"]
mod glm;
#[path = "stats/grm.rs"]
mod grm;
#[path = "stats/gstats.rs"]
mod gstats;
#[path = "stats/gwas_unified.rs"]
mod gwas_unified;
#[path = "stats/he.rs"]
mod he;
#[path = "stats/heritability.rs"]
mod heritability;
#[path = "stats/ld.rs"]
mod ld;
#[path = "stats/lm_trait.rs"]
mod lm_trait;
#[path = "stats/lmm.rs"]
mod lmm;
#[path = "stats/lmm_common.rs"]
mod lmm_common;
#[path = "stats/mixed_ld.rs"]
mod mixed_ld;
#[path = "stats/packed.rs"]
mod packed;
#[path = "stats/reml.rs"]
mod reml;
#[path = "stats/rrblup.rs"]
mod rrblup;
#[path = "stats/rsvd.rs"]
mod rsvd;
#[path = "stats/spgrm.rs"]
mod spgrm;
#[path = "stats/splmm.rs"]
mod splmm;
#[allow(dead_code)]
#[path = "stats/splmm_approx.rs"]
mod splmm_approx;
#[path = "stats/spreml.rs"]
mod spreml;
#[path = "stats/common.rs"]
mod stats_common;
#[path = "stats/susie.rs"]
mod susie;
#[allow(dead_code)]
#[path = "stats/top.rs"]
mod top;

#[path = "stats/xpclr.rs"]
mod xpclr;

// io
#[path = "io/assoc2tsv.rs"]
mod assoc2tsv;
#[path = "io/bincore.rs"]
mod bincore;
#[path = "io/binsidecar.rs"]
mod binsidecar;
#[path = "io/binwriter.rs"]
mod binwriter;
#[path = "io/breader.rs"]
mod breader;
#[path = "io/gfcore.rs"]
mod gfcore;
#[path = "io/gffanno.rs"]
mod gffanno;
#[path = "io/gfreader.rs"]
mod gfreader;
#[path = "io/gload.rs"]
mod gload;
use gload::{gwas_memload_clear, gwas_memload_register};
#[path = "io/gmerge.rs"]
mod gmerge;

#[path = "io/gwriter.rs"]
mod gwriter;
#[path = "imp/mod.rs"]
mod imp;
#[path = "io/imp_reader.rs"]
mod imp_reader;
#[path = "io/imp_vcf_writer.rs"]
mod imp_vcf_writer;
#[path = "kmer/mod.rs"]
pub mod kmer;
#[path = "io/pgen.rs"]
mod pgen;
#[path = "io/pgen_writer.rs"]
mod pgen_writer;
#[path = "io/pipeline.rs"]
mod pipeline;
#[path = "io/sim.rs"]
mod sim;
#[path = "io/vcfout.rs"]
mod vcfout;



// math
#[path = "math/active_path.rs"]
mod active_path;
#[path = "math/aireml.rs"]
mod aireml;
#[path = "math/bedmath.rs"]
mod bedmath;
#[path = "math/bitwise.rs"]
mod bitwise;
#[path = "math/blas.rs"]
mod blas;
#[path = "math/brent.rs"]
mod brent;
#[path = "math/cholesky.rs"]
mod cholesky;
#[path = "math/eigh.rs"]
mod eigh;
#[path = "math/FaST.rs"]
mod fast_math;
#[path = "math/lasso.rs"]
mod lasso;
#[path = "math/linalg.rs"]
mod linalg;
#[path = "math/ld.rs"]
mod math_ld;
#[path = "ml/mod.rs"]
mod ml;
#[path = "math/pcg.rs"]
mod pcg;

#[path = "sim/g2p.rs"]
mod sim_g2p;

use algwas::algwas_packed_to_tsv;
use assoc2tsv::GwasAssocTsvWriter;
use bayes::{
    bayesa, bayesa_packed, bayesa_stream_bed, bayesb, bayesb_packed, bayesb_stream_bed, bayesc,
    bayesc_packed, bayesc_stream_bed,
};
use bayesr::{bayesr as bayesr_fn, bayesr_packed, bayesr_stream_bed};
use binwriter::Bin01StreamWriter;
use blas::{
    rust_blas_get_num_threads, rust_blas_set_num_threads, rust_eigh_lapack_backend,
    rust_sgemm_backend,
};
pub use blas::{
    rust_blas_get_num_threads as janusx_rust_blas_get_num_threads,
    rust_blas_set_num_threads as janusx_rust_blas_set_num_threads,
    rust_eigh_lapack_backend as janusx_rust_eigh_lapack_backend,
    rust_sgemm_backend as janusx_rust_sgemm_backend,
};
use bsa::preprocess_bsa;
use eigh::{
    rust_eigh_from_array_f64, rust_eigh_from_array_f64_inplace, rust_eigh_from_matrix_file_f64,
    rust_eigh_from_matrix_file_subset_f64,
};
use farmcpu::farmcpu_packed_to_tsv;
use fst::fst_bed_window_to_tsv;
use fvlmm::{
    fvlmm_assoc_bed_to_tsv_f32, fvlmm_assoc_chunk_f32, fvlmm_assoc_chunk_from_snp_f32,
    fvlmm_assoc_chunk_from_snp_to_tsv_f32, fvlmm_assoc_chunk_from_snp_with_cache_f32,
    fvlmm_assoc_chunk_with_cache_f32, fvlmm_assoc_kfile_to_tsv_f32,
    fvlmm_assoc_packed_f32_to_tsv, fvlmm_assoc_prepare_cache_f32, FvLmmAssocCache,
};

use garfield::{
    finalize_max6_genotype_universe_py, garfield_grm_max6_rerank_candidates_py,
    garfield_logic_residual_screen_bin_py, garfield_logic_search_bed_py,
    garfield_residualize_grm_py, load_bin01_packed_py, prepare_max6_genotype_tags_py,
    PyMax6PreparedGenotypeTags, PyMax6PreparedGenotypeUniverse,
};
use gblup::{
    farmcpu_q_packed_grm_pca_f32, gblup_effect_from_meta_stream, gblup_grm_from_meta_to_npy,
    gblup_reml_npy_grm, gblup_reml_packed_bed, packed_mtm_f64,
};
use gffanno::GffAnnotationIndex;
use gfreader::{
    bed_filter_to_plink_rust, bed_mmap_filter_to_plink_rust, count_hmp_snps, count_vcf_snps,
    load_bed_2bit_packed, load_bim_columns, prepare_bed_2bit_packed, prepare_bed_fused_qc_packed,
    prepare_bed_logic_keep_mask, prepare_bed_logic_keep_mask_pure_line,
    prepare_bed_logic_meta_selected, scan_bed_2bit_packed_stats, BedChunkReader,
    BedChunkReaderFromMeta, HmpChunkReader, PgenChunkReader, SiteInfo, TxtChunkReader,
    VcfChunkReader,
};
use glm::{
    lm_assoc_kfile_to_tsv_f32, lm_block_assoc_f32, lm_block_assoc_packed,
    lm_block_assoc_packed_to_tsv, lm_pgen_to_tsv_f32, lm_stream_bed_segments_compact_to_tsv,
    lm_stream_bed_to_tsv,
};
use gmerge::{convert_genotypes, merge_genotypes, PyConvertStats, PyMergeStats};
use grm::{
    grm_bed_f64_from_meta, grm_kfile_f32, grm_kfile_f32_to_npy, grm_packed_bed_f32, grm_packed_f32,
    grm_packed_f32_with_stats, grm_packed_f64_with_stats, grm_stream_bed_f32,
    grm_stream_bed_f32_to_npy,
};
use gstats::{
    gstats_bed_individual_stats, gstats_bed_joint_stats, gstats_bed_ldscore, gstats_bed_site_stats,
};
use gwas_unified::{
    gwas_packed_unified_to_tsv, gwas_trait_model_dispatch_v2, gwas_trait_model_schedule,
};

use gwriter::{HmpStreamWriter, PgenStreamWriter, PlinkStreamWriter, VcfStreamWriter};
use he::he_pcg_bed;
use heritability::{prepare_sparse_onehot_blup_cache, SparseOneHotBlupCache};
use kmer::{
    kfile_inspect_py, kfile_read_rows_f32_py, kformat_run_py, kmer_count_run_py,
    kmer_resolve_inputs_py, kmerge_run_py, kstats_run_py,
};
use ld::{
    bed_fam_founder_indices, bed_ldblock_r2_rust, bed_packed_ld_prune_maf_priority,
    bed_prune_mask_to_plink_rust, bed_prune_selected_to_plink_rust, bed_prune_to_plink_rust,
    packed_prune_kernel_stats, scan_plink_selected_snp_indices_from_bim_rust,
};
use lm_trait::{lm_trait_assoc_bed_matrix_to_tsv, lm_trait_assoc_bed_to_tsv};
use lmm::{
    lmm_assoc_chunk_f32, lmm_assoc_chunk_from_snp_f32, lmm_reml_assoc_bed_to_tsv_f32,
    lmm_reml_assoc_kfile_to_tsv_f32, lmm_reml_assoc_packed_f32, lmm_reml_assoc_packed_f32_to_tsv,
    lmm_reml_chunk_f32, lmm_reml_chunk_from_snp_f32, lmm_reml_lmm2_assoc_bed_to_tsv_f32,
    lmm_reml_lmm2_chunk_from_snp_f32,
};
use mixed_ld::fvlmm_effective_ld_spectral_f64;
use packed::{
    bed_decode_rows_f32_from_meta, bed_packed_decode_rows_f32, bed_packed_decode_stats_f64,
    bed_packed_empirical_stats_f64, bed_packed_empirical_stats_subset_f64,
    bed_packed_row_flip_mask, bed_packed_signed_hash_f32, bed_packed_signed_hash_kernels_f64,
    bed_packed_signed_hash_ztz_stats_f64, bed_stream_empirical_stats_f64,
    bed_stream_empirical_stats_subset_f64, cross_grm_times_alpha_packed_f64, packed_malpha_f64,
};
use reml::{
    ai_reml_multi_f64, ai_reml_null_f64, lmm_null_fit_f32, lmm_null_pve_se_f64, lmm_reml_null_f32,
    lmm_rotate_x_y_with_ut_f64, lmm_rotate_y_with_ut_f64, ml_loglike_null_f32,
};
use rrblup::{
    rrblup_exact_snp_fit_prepared_bed, rrblup_exact_snp_prepare_bed_from_meta,
    rrblup_exact_snp_prepare_bed_from_raw_cache, rrblup_exact_snp_prepare_raw_bed_from_meta,
    rrblup_pcg_bed, rrblup_standardized_grm_from_source, rrblupc_exact_snp_fit_prepared_bed,
    rrblupc_exact_snp_prepare_bed_from_meta, rrblupc_exact_snp_prepare_bed_from_raw_cache,
    rrblupc_exact_snp_prepare_raw_bed_from_meta, rrblupc_pcg_bed, RrblupCenteredExactSnpCache,
    RrblupExactSnpCache, RrblupExactSnpRawCache,
};
use rsvd::{admx_rsvd_stream_sample, admx_set_threads, py_rsvd_packed_subset};
use sim::{SimChunkGenerator, SimEngine, SimTraitAccumulator};
use sim_g2p::g2p_simulate_py;
use spgrm::{
    grm_bed_f32_row_band_from_meta, grm_bed_f32_row_band_from_meta_to_npy,
    grm_bed_f32_tiled_from_meta_to_npy, spgrm_bed_to_jxgrm, spgrm_bed_to_jxgrm_from_meta,
    spgrm_dense_f32_to_jxgrm, spgrm_dense_npy_to_jxgrm, spgrm_kfile_to_jxgrm,
};
use splmm::{
    splmm_assoc_pcg_bed, splmm_assoc_pcg_bed_to_tsv, splmm_assoc_pcg_bed_to_tsv_fused_approx,
    splmm_assoc_pcg_bed_to_tsv_fused_exact, splmm_assoc_pcg_dense_f32,
    splmm_assoc_pcg_kfile_to_tsv, splmm_load_sparse_grm_subset_dense,
    splmm_residualized_approx_null_fit_from_jxgrm, splmm_sparse_grm_diag_stats,
};
use spreml::{
    spreml_sparse_fastgwa_fixed_vp_brent_from_jxgrm, spreml_sparse_reml_brent_from_jxgrm,
};
use susie::susie_rss_f64;
use top::{top_fit_model_py, top_rank_to_target_values_py};

use xpclr::xpclr_bed_to_tsv;

pub use aireml::{ai_reml_null_from_spectral, AiRemlNullResult};
pub use algwas::{AlgwasConfig, AlgwasStage1PathPoint, AlgwasStage1Result};
pub use cholesky::{
    read_sparse_grm_csc, sparse_cholesky_analyze_jxgrm_csc, sparse_cholesky_analyze_jxgrm_path,
    sparse_cholesky_from_jxgrm_csc, sparse_cholesky_from_jxgrm_path, SparseJxgrmCholesky,
    SparseJxgrmCholeskyAnalysis, SparseJxgrmSolveWorkspace,
};
pub use eigh::symmetric_eigh_f64_row_major;
pub use grm::grm_packed_f64_from_stats_rust;
pub use he::{
    he_variance_components_packed, he_variance_components_packed_with_covariates, HePcgResult,
    RowStdStats, HE_BOUNDARY_INTERIOR, HE_BOUNDARY_ORIGIN, HE_BOUNDARY_SIGMA_E_ZERO,
    HE_BOUNDARY_SIGMA_G_ZERO,
};
pub use lasso::{
    compare_lasso_results, fit_lasso_f32, fit_lasso_f32_reference, fit_lasso_packed_active_f32,
    fit_lasso_packed_active_path_f32, fit_lasso_path_f32, DenseLassoDesign, LassoCompareStats,
    LassoConfig, LassoDesignMatrix, LassoPathPoint, LassoResult, LassoSolverKind,
    PackedBedLassoConfig, PackedBedLassoDesign,
};

// ============================================================
// PyO3 module exports
// ============================================================

#[pyfunction]
#[pyo3(signature = (options_json, progress_callback=None))]
fn imp_run(
    py: Python<'_>,
    options_json: String,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<String> {
    let config = serde_json::from_str::<imp::ImpConfig>(&options_json)
        .map_err(|error| PyRuntimeError::new_err(format!("invalid imp options: {error}")))?;
    let callback = progress_callback.map(|callback| callback.clone_ref(py));
    let summary = py
        .detach(move || {
            imp::run_imp_with_progress(config, |event| {
                emit_imp_progress_callback(callback.as_ref(), event)
            })
        })
        .map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&summary)
        .map_err(|error| PyRuntimeError::new_err(format!("serialize imp summary: {error}")))
}

#[pyfunction]
fn imp_estimate_window_bytes(
    options_json: String,
    target_samples: usize,
    reference_samples: usize,
    max_target_window_markers: usize,
    max_reference_window_markers: usize,
    max_union_window_markers: usize,
    max_initial_pbwt_chunk_markers: usize,
    max_window_cm_span: f64,
) -> PyResult<u64> {
    let config = serde_json::from_str::<imp::ImpConfig>(&options_json)
        .map_err(|error| PyRuntimeError::new_err(format!("invalid imp options: {error}")))?;
    imp::estimate_imp_window_bytes(
        &config,
        target_samples,
        reference_samples,
        max_target_window_markers,
        max_reference_window_markers,
        max_union_window_markers,
        max_initial_pbwt_chunk_markers,
        max_window_cm_span,
    )
    .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
#[pyo3(signature = (input, output_prefix, progress_callback=None))]
fn imp_convert_to_pgen(
    py: Python<'_>,
    input: String,
    output_prefix: String,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<String> {
    let input = PathBuf::from(input);
    let output_prefix = PathBuf::from(output_prefix);
    let callback = progress_callback.map(|callback| callback.clone_ref(py));
    let summary = py
        .detach(move || {
            imp::convert_to_pgen_with_progress(&input, &output_prefix, |event| {
                emit_imp_progress_callback(callback.as_ref(), event)
            })
        })
        .map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&summary).map_err(|error| {
        PyRuntimeError::new_err(format!("serialize imp conversion summary: {error}"))
    })
}

fn emit_imp_progress_callback(
    callback: Option<&Py<PyAny>>,
    event: &imp::ImpProgress,
) -> Result<(), String> {
    if let Some(callback) = callback {
        Python::attach(|py| -> PyResult<()> {
            py.check_signals()?;
            callback.call1(
                py,
                (
                    event.stage.as_str(),
                    event.chrom.as_str(),
                    event.window_index,
                    event.markers as u64,
                    event.variants_written,
                    event.samples as u64,
                ),
            )?;
            Ok(())
        })
        .map_err(|error| error.to_string())?;
    } else {
        Python::attach(|py| py.check_signals()).map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[pymodule]
fn janusx(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<BedChunkReader>()?;
    m.add_class::<BedChunkReaderFromMeta>()?;
    m.add_class::<PgenChunkReader>()?;
    m.add_class::<VcfChunkReader>()?;
    m.add_class::<HmpChunkReader>()?;
    m.add_class::<TxtChunkReader>()?;
    m.add_class::<PlinkStreamWriter>()?;
    m.add_class::<PgenStreamWriter>()?;
    m.add_class::<VcfStreamWriter>()?;
    m.add_class::<HmpStreamWriter>()?;
    m.add_class::<GwasAssocTsvWriter>()?;
    m.add_class::<SimChunkGenerator>()?;
    m.add_class::<SimEngine>()?;
    m.add_class::<SimTraitAccumulator>()?;
    m.add_class::<SiteInfo>()?;
    m.add_class::<GffAnnotationIndex>()?;
    m.add_class::<FvLmmAssocCache>()?;
    m.add_class::<RrblupExactSnpCache>()?;
    m.add_class::<RrblupCenteredExactSnpCache>()?;
    m.add_class::<RrblupExactSnpRawCache>()?;
    m.add_class::<SparseOneHotBlupCache>()?;
    m.add_class::<PyMergeStats>()?;
    m.add_class::<PyConvertStats>()?;
    m.add_class::<PyMax6PreparedGenotypeTags>()?;
    m.add_class::<PyMax6PreparedGenotypeUniverse>()?;
    m.add_class::<Bin01StreamWriter>()?;
    m.add_function(wrap_pyfunction!(imp_run, m)?)?;
    m.add_function(wrap_pyfunction!(imp_estimate_window_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(imp_convert_to_pgen, m)?)?;
    m.add_function(wrap_pyfunction!(spgrm_bed_to_jxgrm, m)?)?;
    m.add_function(wrap_pyfunction!(spgrm_bed_to_jxgrm_from_meta, m)?)?;
    m.add_function(wrap_pyfunction!(spgrm_kfile_to_jxgrm, m)?)?;
    m.add_function(wrap_pyfunction!(grm_bed_f32_row_band_from_meta, m)?)?;
    m.add_function(wrap_pyfunction!(grm_bed_f32_row_band_from_meta_to_npy, m)?)?;
    m.add_function(wrap_pyfunction!(grm_bed_f32_tiled_from_meta_to_npy, m)?)?;
    m.add_function(wrap_pyfunction!(spgrm_dense_f32_to_jxgrm, m)?)?;
    m.add_function(wrap_pyfunction!(spgrm_dense_npy_to_jxgrm, m)?)?;
    m.add_function(wrap_pyfunction!(grm_kfile_f32, m)?)?;
    m.add_function(wrap_pyfunction!(grm_kfile_f32_to_npy, m)?)?;
    m.add_function(wrap_pyfunction!(kmer_count_run_py, m)?)?;
    m.add_function(wrap_pyfunction!(kmerge_run_py, m)?)?;
    m.add_function(wrap_pyfunction!(kmer_resolve_inputs_py, m)?)?;
    m.add_function(wrap_pyfunction!(kstats_run_py, m)?)?;
    m.add_function(wrap_pyfunction!(kfile_inspect_py, m)?)?;
    m.add_function(wrap_pyfunction!(kfile_read_rows_f32_py, m)?)?;
    m.add_function(wrap_pyfunction!(kformat_run_py, m)?)?;

    m.add_function(wrap_pyfunction!(merge_genotypes, m)?)?;
    m.add_function(wrap_pyfunction!(convert_genotypes, m)?)?;
    m.add_function(wrap_pyfunction!(count_vcf_snps, m)?)?;
    m.add_function(wrap_pyfunction!(count_hmp_snps, m)?)?;
    m.add_function(wrap_pyfunction!(load_bed_2bit_packed, m)?)?;
    m.add_function(wrap_pyfunction!(load_bin01_packed_py, m)?)?;
    m.add_function(wrap_pyfunction!(scan_bed_2bit_packed_stats, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_bed_2bit_packed, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_bed_logic_keep_mask, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_bed_logic_keep_mask_pure_line, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_bed_logic_meta_selected, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_bed_fused_qc_packed, m)?)?;
    m.add_function(wrap_pyfunction!(load_bim_columns, m)?)?;
    m.add_function(wrap_pyfunction!(g2p_simulate_py, m)?)?;

    m.add_function(wrap_pyfunction!(lm_block_assoc_f32, m)?)?;
    m.add_function(wrap_pyfunction!(algwas_packed_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(lm_stream_bed_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(lm_pgen_to_tsv_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lm_assoc_kfile_to_tsv_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lm_trait_assoc_bed_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(lm_trait_assoc_bed_matrix_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(lm_stream_bed_segments_compact_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(lm_block_assoc_packed, m)?)?;
    m.add_function(wrap_pyfunction!(lm_block_assoc_packed_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_row_flip_mask, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_decode_rows_f32, m)?)?;
    m.add_function(wrap_pyfunction!(bed_decode_rows_f32_from_meta, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_decode_stats_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_empirical_stats_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_empirical_stats_subset_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bed_stream_empirical_stats_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bed_stream_empirical_stats_subset_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bed_fam_founder_indices, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_ld_prune_maf_priority, m)?)?;
    m.add_function(wrap_pyfunction!(bed_ldblock_r2_rust, m)?)?;
    m.add_function(wrap_pyfunction!(
        scan_plink_selected_snp_indices_from_bim_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(packed_prune_kernel_stats, m)?)?;
    m.add_function(wrap_pyfunction!(bed_prune_mask_to_plink_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bed_prune_selected_to_plink_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bed_prune_to_plink_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_signed_hash_f32, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_signed_hash_ztz_stats_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_signed_hash_kernels_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bed_filter_to_plink_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bed_mmap_filter_to_plink_rust, m)?)?;
    m.add_function(wrap_pyfunction!(cross_grm_times_alpha_packed_f64, m)?)?;
    m.add_function(wrap_pyfunction!(packed_malpha_f64, m)?)?;
    m.add_function(wrap_pyfunction!(he_pcg_bed, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_sparse_onehot_blup_cache, m)?)?;
    m.add_function(wrap_pyfunction!(splmm_assoc_pcg_bed, m)?)?;
    m.add_function(wrap_pyfunction!(splmm_assoc_pcg_bed_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(
        splmm_assoc_pcg_bed_to_tsv_fused_approx,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(splmm_assoc_pcg_bed_to_tsv_fused_exact, m)?)?;
    m.add_function(wrap_pyfunction!(splmm_assoc_pcg_dense_f32, m)?)?;
    m.add_function(wrap_pyfunction!(splmm_assoc_pcg_kfile_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(splmm_load_sparse_grm_subset_dense, m)?)?;
    m.add_function(wrap_pyfunction!(splmm_sparse_grm_diag_stats, m)?)?;
    m.add_function(wrap_pyfunction!(
        splmm_residualized_approx_null_fit_from_jxgrm,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spreml_sparse_fastgwa_fixed_vp_brent_from_jxgrm,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(spreml_sparse_reml_brent_from_jxgrm, m)?)?;
    m.add_function(wrap_pyfunction!(susie_rss_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rrblup_pcg_bed, m)?)?;
    m.add_function(wrap_pyfunction!(rrblup_standardized_grm_from_source, m)?)?;
    m.add_function(wrap_pyfunction!(rrblup_exact_snp_prepare_bed_from_meta, m)?)?;
    m.add_function(wrap_pyfunction!(
        rrblup_exact_snp_prepare_raw_bed_from_meta,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        rrblup_exact_snp_prepare_bed_from_raw_cache,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(rrblup_exact_snp_fit_prepared_bed, m)?)?;
    m.add_function(wrap_pyfunction!(rrblupc_pcg_bed, m)?)?;
    m.add_function(wrap_pyfunction!(
        rrblupc_exact_snp_prepare_bed_from_meta,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        rrblupc_exact_snp_prepare_raw_bed_from_meta,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        rrblupc_exact_snp_prepare_bed_from_raw_cache,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(rrblupc_exact_snp_fit_prepared_bed, m)?)?;
    m.add_function(wrap_pyfunction!(gblup_reml_packed_bed, m)?)?;
    m.add_function(wrap_pyfunction!(gblup_reml_npy_grm, m)?)?;
    m.add_function(wrap_pyfunction!(gblup_effect_from_meta_stream, m)?)?;
    m.add_function(wrap_pyfunction!(gblup_grm_from_meta_to_npy, m)?)?;
    m.add_function(wrap_pyfunction!(rust_sgemm_backend, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_lapack_backend, m)?)?;
    m.add_function(wrap_pyfunction!(rust_blas_set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(rust_blas_get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_from_array_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_from_matrix_file_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_from_matrix_file_subset_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_from_array_f64_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(top_fit_model_py, m)?)?;
    m.add_function(wrap_pyfunction!(top_rank_to_target_values_py, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_bed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(grm_bed_f64_from_meta, m)?)?;
    m.add_function(wrap_pyfunction!(grm_stream_bed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(grm_stream_bed_f32_to_npy, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_f32_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_f64_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(gstats_bed_site_stats, m)?)?;
    m.add_function(wrap_pyfunction!(gstats_bed_joint_stats, m)?)?;
    m.add_function(wrap_pyfunction!(gstats_bed_individual_stats, m)?)?;
    m.add_function(wrap_pyfunction!(gstats_bed_ldscore, m)?)?;
    m.add_function(wrap_pyfunction!(fst_bed_window_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(xpclr_bed_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(packed_mtm_f64, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_q_packed_grm_pca_f32, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_packed_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(gwas_memload_register, m)?)?;
    m.add_function(wrap_pyfunction!(gwas_memload_clear, m)?)?;
    m.add_function(wrap_pyfunction!(gwas_packed_unified_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(gwas_trait_model_dispatch_v2, m)?)?;
    m.add_function(wrap_pyfunction!(gwas_trait_model_schedule, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_null_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_null_fit_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_null_pve_se_f64, m)?)?;
    m.add_function(wrap_pyfunction!(ml_loglike_null_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ai_reml_null_f64, m)?)?;
    m.add_function(wrap_pyfunction!(ai_reml_multi_f64, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_chunk_from_snp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_lmm2_chunk_from_snp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_assoc_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_assoc_chunk_from_snp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_effective_ld_spectral_f64, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_chunk_with_cache_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_chunk_from_snp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_chunk_from_snp_to_tsv_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_bed_to_tsv_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_kfile_to_tsv_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_prepare_cache_f32, m)?)?;
    m.add_function(wrap_pyfunction!(
        fvlmm_assoc_chunk_from_snp_with_cache_f32,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(lmm_rotate_x_y_with_ut_f64, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_rotate_y_with_ut_f64, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_assoc_bed_to_tsv_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_assoc_kfile_to_tsv_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_lmm2_assoc_bed_to_tsv_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_assoc_packed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_assoc_packed_f32_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_packed_f32_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(bayesa, m)?)?;
    m.add_function(wrap_pyfunction!(bayesb, m)?)?;
    m.add_function(wrap_pyfunction!(bayesc, m)?)?;
    m.add_function(wrap_pyfunction!(bayesa_packed, m)?)?;
    m.add_function(wrap_pyfunction!(bayesb_packed, m)?)?;
    m.add_function(wrap_pyfunction!(bayesc_packed, m)?)?;
    m.add_function(wrap_pyfunction!(bayesa_stream_bed, m)?)?;
    m.add_function(wrap_pyfunction!(bayesb_stream_bed, m)?)?;
    m.add_function(wrap_pyfunction!(bayesc_stream_bed, m)?)?;
    m.add_function(wrap_pyfunction!(bayesr_fn, m)?)?;
    m.add_function(wrap_pyfunction!(bayesr_packed, m)?)?;
    m.add_function(wrap_pyfunction!(bayesr_stream_bed, m)?)?;
    m.add_function(wrap_pyfunction!(admx_rsvd_stream_sample, m)?)?;
    m.add_function(wrap_pyfunction!(py_rsvd_packed_subset, m)?)?;
    m.add_function(wrap_pyfunction!(admx_set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_grm_max6_rerank_candidates_py, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_logic_residual_screen_bin_py, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_logic_search_bed_py, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_max6_genotype_tags_py, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_max6_genotype_universe_py, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_residualize_grm_py, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_bsa, m)?)?;
    ct::output::register_py(m)?;
    Ok(())
}
