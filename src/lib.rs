use pyo3::prelude::*;
use pyo3::Bound;

// stats
#[path = "stats/adamixture.rs"]
mod admixture;
#[path = "stats/algwas.rs"]
mod algwas;
#[path = "stats/assoc.rs"]
mod assoc;
#[path = "stats/bayes.rs"]
mod bayes;
#[path = "stats/beam.rs"]
pub mod beam;
#[path = "stats/bsa.rs"]
mod bsa;
#[path = "garfield/mod.rs"]
pub(crate) mod garfield;
#[path = "stats/glm.rs"]
mod glm;
#[path = "stats/grm.rs"]
mod grm;
#[doc(hidden)]
#[path = "grm_bench_support.rs"]
pub mod grm_bench_support;
#[path = "stats/gs_native.rs"]
pub(crate) mod gs_native;
#[path = "stats/gstats.rs"]
mod gstats;
#[path = "stats/gwas_unified.rs"]
mod gwas_unified;
#[path = "stats/he.rs"]
mod he;
#[path = "stats/ld.rs"]
mod ld;
#[path = "stats/lmm.rs"]
mod lmm;
#[path = "stats/lmm_scan.rs"]
mod lmm_scan;
#[path = "stats/logreg.rs"]
mod logreg;
#[path = "stats/packed.rs"]
mod packed;
#[path = "stats/rsvd.rs"]
mod rsvd;
#[path = "stats/spgrm.rs"]
mod spgrm;
#[path = "stats/splmm.rs"]
mod splmm;
#[path = "stats/spreml.rs"]
mod spreml;
#[path = "stats/common.rs"]
mod stats_common;
#[allow(dead_code)]
#[path = "stats/top.rs"]
mod top;
#[path = "stats/tree.rs"]
mod tree;

// io
#[path = "io/gfcore.rs"]
mod gfcore;
#[path = "io/gfreader.rs"]
mod gfreader;
#[path = "io/gload.rs"]
mod gload;
#[path = "io/gmerge.rs"]
mod gmerge;
#[path = "io/gwasio.rs"]
mod gwasio;
#[path = "io/sim.rs"]
mod sim;
#[path = "io/vcfout.rs"]
mod vcfout;
#[path = "io/pipeline.rs"]
mod pipeline;

// workflow (structural layer; currently type-only skeleton)
#[path = "workflow/mod.rs"]
mod workflow;

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
#[path = "math/KING.rs"]
mod king;
#[path = "math/lasso.rs"]
mod lasso;
#[path = "math/linalg.rs"]
mod linalg;
#[path = "math/farmcpu.rs"]
mod math_farmcpu;
#[path = "math/ld.rs"]
mod math_ld;
#[path = "ml/mod.rs"]
mod ml;
#[path = "math/pcg.rs"]
mod pcg;
#[path = "phylo/mod.rs"]
mod phylo;
#[path = "sim/g2p.rs"]
mod sim_g2p;

use admixture::{
    admx_adam_optimize_f32, admx_adam_update_p, admx_adam_update_p_inplace,
    admx_adam_update_p_inplace_f32, admx_adam_update_q, admx_adam_update_q_inplace,
    admx_adam_update_q_inplace_f32, admx_allele_frequency, admx_em_step, admx_em_step_inplace,
    admx_em_step_inplace_f32, admx_kl_divergence, admx_loglikelihood, admx_loglikelihood_f32,
    admx_map_p_f32, admx_map_q_f32, admx_multiply_a_omega, admx_multiply_a_omega_inplace,
    admx_multiply_at_omega, admx_multiply_at_omega_inplace, admx_rmse_f32, admx_rmse_f64,
    admx_rsvd_power_step_inplace, admx_rsvd_stream, admx_rsvd_stream_sample, admx_set_threads,
};
use algwas::algwas_packed_to_tsv;
use assoc::{
    farmcpu_dense, farmcpu_packed_to_tsv, farmcpu_rem_dense, farmcpu_rem_packed,
    farmcpu_super_dense, farmcpu_super_packed, farmcpu_write_assoc_tsv,
};
use bayes::{
    bayesa, bayesa_packed, bayesa_packed_trace, bayesb, bayesb_packed, bayesb_packed_trace,
    bayescpi, bayescpi_packed, bayescpi_packed_trace,
};
use beam::{
    beam_scan_windows_binary_mcc_bin_py, beam_scan_windows_continuous_corr_bin_py,
    beam_search_and_binary_mcc_bin_indices_py, beam_search_and_binary_mcc_bin_py,
    beam_search_and_continuous_corr_bin_indices_py, beam_search_and_continuous_corr_bin_py,
};
use bitwise::{and_popcount_py, bitand_assign_py, bitnot_masked_py, bitor_into_py, popcount_py};
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
    rust_eigh_debug_f64, rust_eigh_from_array_f64, rust_eigh_from_array_f64_inplace,
    rust_eigh_from_matrix_file_f64, rust_eigh_from_matrix_file_subset_f64,
};
use fast_math::fastlmm_prepare_lowrank_f64;
use garfield::{
    garfield_compare_score_cont_centered_gain_batch_metal_vs_cpu_py,
    garfield_compare_score_cont_centered_gain_singleton_backends_py, garfield_eval_rule_bin_py,
    garfield_logic_search_bed_py, garfield_metal_runtime_status_py, garfield_prepare_input_bin_py,
    garfield_residualize_bed_py, garfield_residualize_grm_py, garfield_scan_groups_bin_py,
    garfield_scan_windows_bin_py, garfield_score_cont_centered_gain_batch_packed_cpu_py,
    garfield_score_cont_centered_gain_batch_packed_metal_py, garfield_subset_bin_samples_py,
    load_bin01_packed_py, load_mbin_packed_py, score_binary_ba_mcc_batch_py, score_binary_ba_py,
    score_binary_mcc_py, score_cont_corr_py, score_cont_mean_diff_corr_batch_py,
    score_cont_mean_diff_py,
};
use gfreader::{
    bed_filter_stream_to_plink_rust, bed_filter_to_plink_rust, bed_mmap_filter_to_plink_rust,
    count_hmp_snps, count_vcf_snps, gfd_packbits_from_dosage_block, load_bed_2bit_packed,
    load_bed_u8_matrix, load_site_info, prepare_bed_2bit_packed, prepare_bed_logic_meta_selected,
    scan_bed_2bit_packed_stats, BedChunkReader, BedMmapReader, GwasAssocTsvWriter, HmpChunkReader,
    HmpStreamWriter, NpyMmapReader, PlinkStreamWriter, SiteInfo, TxtChunkReader, VcfChunkReader,
    VcfStreamWriter,
};
use glm::{
    glm_ixx_from_x_qr, glmf32, glmf32_full, glmf32_packed, glmf32_packed_assoc,
    glmf32_packed_assoc_to_tsv, lm_block_assoc_packed, lm_block_assoc_packed_to_tsv,
    lm_stream_bed_to_tsv,
};
use gmerge::{convert_genotypes, merge_genotypes, PyConvertStats, PyMergeStats};
use grm::{
    grm_bed_f64_from_meta, grm_packed_bed_f32, grm_packed_bed_f64, grm_packed_f32,
    grm_packed_f32_with_stats, grm_packed_f64, grm_packed_f64_with_stats, grm_sim_bench_f32,
    grm_stream_bed_f32, grm_stream_bed_f32_to_npy, grm_stream_bed_f64, grm_stream_bed_f64_to_npy,
};
use gs_native::{
    farmcpu_q_packed_grm_pca_f32, gblup_reml_packed_bed, packed_mtm_f64, rrblup_pcg_bed,
};
use gstats::{
    gstats_bed_individual_stats, gstats_bed_joint_stats, gstats_bed_ldscore, gstats_bed_site_stats,
};
use gwas_unified::{
    gwas_lmm_lm_null_lrt_decision, gwas_packed_unified_to_tsv, gwas_trait_model_dispatch_v2,
    gwas_trait_model_schedule,
};
use gwasio::load_gwas_triplet_fast;
use he::he_pcg_bed;
use ld::{
    bed_ldblock_r2_rust, bed_packed_ld_prune_maf_priority, bed_prune_to_plink_rust,
    packed_prune_kernel_stats,
};
use lmm::{fastlmm_assoc_from_snp_f32, fastlmm_reml_chunk_f32, fastlmm_reml_null_f32};
use lmm_scan::{
    ai_reml_multi_f64, ai_reml_null_f64, fastlmm_assoc_chunk_f32, fastlmm_assoc_packed_f32,
    fastlmm_assoc_packed_f32_to_tsv, fvlmm_assoc_bed_to_tsv_f32, fvlmm_assoc_chunk_f32,
    fvlmm_assoc_chunk_from_snp_f32, fvlmm_assoc_chunk_from_snp_to_tsv_f32,
    fvlmm_assoc_chunk_from_snp_with_cache_f32, fvlmm_assoc_chunk_with_cache_f32,
    fvlmm_assoc_packed_f32_to_tsv, fvlmm_assoc_prepare_cache_f32, lmm_assoc_chunk_f32,
    lmm_assoc_chunk_from_snp_f32, lmm_reml_assoc_packed_f32, lmm_reml_assoc_packed_f32_to_tsv,
    lmm_reml_chunk_f32, lmm_reml_chunk_from_snp_f32, lmm_reml_null_f32, lmm_rotate_x_y_with_ut_f64,
    ml_loglike_null_f32, FvLmmAssocCache,
};
use logreg::fit_best_and_not_py;
use ml::{garfield_ml_feature_scores_py, garfield_ml_select_topk_py};
use packed::{
    bed_packed_decode_rows_f32, bed_packed_decode_stats_f64, bed_packed_row_flip_mask,
    bed_packed_signed_hash_f32, bed_packed_signed_hash_kernels_f64,
    bed_packed_signed_hash_ztz_stats_f64, cross_grm_times_alpha_packed_f64, packed_malpha_f64,
    packed_malpha_mode_f64,
};
use rsvd::py_rsvd_packed_subset;
use sim::{sim_trait_accumulate_i8_f32, SimChunkGenerator, SimEngine, SimTraitAccumulator};
use sim_g2p::g2p_simulate_py;
use spgrm::{
    spgrm_bed_to_jxgrm, spgrm_dense_f32_to_jxgrm, spgrm_dense_npy_to_jxgrm, spgrm_packed_to_jxgrm,
};
use splmm::{
    splmm_assoc_pcg_bed, splmm_assoc_pcg_bed_to_tsv, splmm_scan_grammar_packed,
    splmm_scan_exact_packed,
    splmm_sparse_grm_diag_stats,
};
use spreml::{spreml_sparse_reml_brent_from_jxgrm, spreml_sparse_reml_grid_from_jxgrm};
use top::{top_fit_model_py, top_rank_to_target_sample_py, top_rank_to_target_values_py};
use tree::{
    geno_chunk_to_alignment_u8, geno_chunk_to_alignment_u8_siteinfo,
    geno_chunk_to_alignment_u8_sites, ml_newick_from_alignment_u8, nj_newick_from_alignment_u8,
    nj_newick_from_distance_matrix,
};

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
pub use king::{
    king_bitplanes_from_packed, king_pair_stats, king_prune_related_graph,
    king_related_graph_from_bitplanes, king_related_graph_from_packed,
    king_related_pairs_from_bitplanes, king_related_pairs_from_packed, king_unrelated_set_from_bed,
    king_unrelated_set_from_bed_default, king_unrelated_set_from_packed, KingBitplanes,
    KingPairStats, KingPruneResult, KingRelatedGraph, KingRelatedPairRow,
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

#[pymodule]
fn janusx(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<BedChunkReader>()?;
    m.add_class::<BedMmapReader>()?;
    m.add_class::<NpyMmapReader>()?;
    m.add_class::<VcfChunkReader>()?;
    m.add_class::<HmpChunkReader>()?;
    m.add_class::<TxtChunkReader>()?;
    m.add_class::<PlinkStreamWriter>()?;
    m.add_class::<VcfStreamWriter>()?;
    m.add_class::<HmpStreamWriter>()?;
    m.add_class::<GwasAssocTsvWriter>()?;
    m.add_class::<SimChunkGenerator>()?;
    m.add_class::<SimEngine>()?;
    m.add_class::<SimTraitAccumulator>()?;
    m.add_class::<SiteInfo>()?;
    m.add_class::<FvLmmAssocCache>()?;
    m.add_class::<PyMergeStats>()?;
    m.add_class::<PyConvertStats>()?;
    m.add_function(wrap_pyfunction!(popcount_py, m)?)?;
    m.add_function(wrap_pyfunction!(and_popcount_py, m)?)?;
    m.add_function(wrap_pyfunction!(bitand_assign_py, m)?)?;
    m.add_function(wrap_pyfunction!(bitor_into_py, m)?)?;
    m.add_function(wrap_pyfunction!(bitnot_masked_py, m)?)?;
    m.add_function(wrap_pyfunction!(score_binary_ba_py, m)?)?;
    m.add_function(wrap_pyfunction!(score_binary_mcc_py, m)?)?;
    m.add_function(wrap_pyfunction!(score_binary_ba_mcc_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(spgrm_packed_to_jxgrm, m)?)?;
    m.add_function(wrap_pyfunction!(spgrm_bed_to_jxgrm, m)?)?;
    m.add_function(wrap_pyfunction!(spgrm_dense_f32_to_jxgrm, m)?)?;
    m.add_function(wrap_pyfunction!(spgrm_dense_npy_to_jxgrm, m)?)?;
    m.add_function(wrap_pyfunction!(score_cont_mean_diff_py, m)?)?;
    m.add_function(wrap_pyfunction!(score_cont_corr_py, m)?)?;
    m.add_function(wrap_pyfunction!(score_cont_mean_diff_corr_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(beam_search_and_binary_mcc_bin_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        beam_search_and_binary_mcc_bin_indices_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(beam_scan_windows_binary_mcc_bin_py, m)?)?;
    m.add_function(wrap_pyfunction!(beam_search_and_continuous_corr_bin_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        beam_search_and_continuous_corr_bin_indices_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        beam_scan_windows_continuous_corr_bin_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(merge_genotypes, m)?)?;
    m.add_function(wrap_pyfunction!(convert_genotypes, m)?)?;
    m.add_function(wrap_pyfunction!(count_vcf_snps, m)?)?;
    m.add_function(wrap_pyfunction!(count_hmp_snps, m)?)?;
    m.add_function(wrap_pyfunction!(load_bed_2bit_packed, m)?)?;
    m.add_function(wrap_pyfunction!(load_bin01_packed_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_mbin_packed_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        garfield_score_cont_centered_gain_batch_packed_cpu_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        garfield_score_cont_centered_gain_batch_packed_metal_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        garfield_compare_score_cont_centered_gain_batch_metal_vs_cpu_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        garfield_compare_score_cont_centered_gain_singleton_backends_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(garfield_metal_runtime_status_py, m)?)?;
    m.add_function(wrap_pyfunction!(scan_bed_2bit_packed_stats, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_bed_2bit_packed, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_bed_logic_meta_selected, m)?)?;
    m.add_function(wrap_pyfunction!(load_bed_u8_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(load_site_info, m)?)?;
    m.add_function(wrap_pyfunction!(gfd_packbits_from_dosage_block, m)?)?;
    m.add_function(wrap_pyfunction!(sim_trait_accumulate_i8_f32, m)?)?;
    m.add_function(wrap_pyfunction!(g2p_simulate_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_gwas_triplet_fast, m)?)?;
    m.add_function(wrap_pyfunction!(glmf32, m)?)?;
    m.add_function(wrap_pyfunction!(glmf32_full, m)?)?;
    m.add_function(wrap_pyfunction!(glm_ixx_from_x_qr, m)?)?;
    m.add_function(wrap_pyfunction!(glmf32_packed, m)?)?;
    m.add_function(wrap_pyfunction!(glmf32_packed_assoc, m)?)?;
    m.add_function(wrap_pyfunction!(glmf32_packed_assoc_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(algwas_packed_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(lm_stream_bed_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(lm_block_assoc_packed, m)?)?;
    m.add_function(wrap_pyfunction!(lm_block_assoc_packed_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(grm_sim_bench_f32, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_row_flip_mask, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_decode_rows_f32, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_decode_stats_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_ld_prune_maf_priority, m)?)?;
    m.add_function(wrap_pyfunction!(bed_ldblock_r2_rust, m)?)?;
    m.add_function(wrap_pyfunction!(packed_prune_kernel_stats, m)?)?;
    m.add_function(wrap_pyfunction!(bed_prune_to_plink_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_signed_hash_f32, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_signed_hash_ztz_stats_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_signed_hash_kernels_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bed_filter_stream_to_plink_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bed_filter_to_plink_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bed_mmap_filter_to_plink_rust, m)?)?;
    m.add_function(wrap_pyfunction!(cross_grm_times_alpha_packed_f64, m)?)?;
    m.add_function(wrap_pyfunction!(packed_malpha_f64, m)?)?;
    m.add_function(wrap_pyfunction!(packed_malpha_mode_f64, m)?)?;
    m.add_function(wrap_pyfunction!(he_pcg_bed, m)?)?;
    m.add_function(wrap_pyfunction!(splmm_assoc_pcg_bed, m)?)?;
    m.add_function(wrap_pyfunction!(splmm_assoc_pcg_bed_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(splmm_scan_grammar_packed, m)?)?;
    m.add_function(wrap_pyfunction!(splmm_scan_exact_packed, m)?)?;
    m.add_function(wrap_pyfunction!(splmm_sparse_grm_diag_stats, m)?)?;
    m.add_function(wrap_pyfunction!(spreml_sparse_reml_grid_from_jxgrm, m)?)?;
    m.add_function(wrap_pyfunction!(spreml_sparse_reml_brent_from_jxgrm, m)?)?;
    m.add_function(wrap_pyfunction!(king::king_unrelated_set_from_bed_py, m)?)?;
    m.add_function(wrap_pyfunction!(rrblup_pcg_bed, m)?)?;
    m.add_function(wrap_pyfunction!(gblup_reml_packed_bed, m)?)?;
    m.add_function(wrap_pyfunction!(rust_sgemm_backend, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_lapack_backend, m)?)?;
    m.add_function(wrap_pyfunction!(rust_blas_set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(rust_blas_get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_debug_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_from_array_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_from_matrix_file_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_from_matrix_file_subset_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_from_array_f64_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_prepare_lowrank_f64, m)?)?;
    m.add_function(wrap_pyfunction!(top_fit_model_py, m)?)?;
    m.add_function(wrap_pyfunction!(top_rank_to_target_sample_py, m)?)?;
    m.add_function(wrap_pyfunction!(top_rank_to_target_values_py, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_bed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_bed_f64, m)?)?;
    m.add_function(wrap_pyfunction!(grm_bed_f64_from_meta, m)?)?;
    m.add_function(wrap_pyfunction!(grm_stream_bed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(grm_stream_bed_f64, m)?)?;
    m.add_function(wrap_pyfunction!(grm_stream_bed_f32_to_npy, m)?)?;
    m.add_function(wrap_pyfunction!(grm_stream_bed_f64_to_npy, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_f64, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_f32_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_f64_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(gstats_bed_site_stats, m)?)?;
    m.add_function(wrap_pyfunction!(gstats_bed_joint_stats, m)?)?;
    m.add_function(wrap_pyfunction!(gstats_bed_individual_stats, m)?)?;
    m.add_function(wrap_pyfunction!(gstats_bed_ldscore, m)?)?;
    m.add_function(wrap_pyfunction!(packed_mtm_f64, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_q_packed_grm_pca_f32, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_rem_dense, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_rem_packed, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_super_dense, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_super_packed, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_packed_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_dense, m)?)?;
    m.add_function(wrap_pyfunction!(gwas_packed_unified_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(gwas_trait_model_dispatch_v2, m)?)?;
    m.add_function(wrap_pyfunction!(gwas_trait_model_schedule, m)?)?;
    m.add_function(wrap_pyfunction!(gwas_lmm_lm_null_lrt_decision, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_write_assoc_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_null_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ml_loglike_null_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ai_reml_null_f64, m)?)?;
    m.add_function(wrap_pyfunction!(ai_reml_multi_f64, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_chunk_from_snp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_assoc_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_assoc_chunk_from_snp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_chunk_with_cache_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_chunk_from_snp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_chunk_from_snp_to_tsv_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_bed_to_tsv_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_prepare_cache_f32, m)?)?;
    m.add_function(wrap_pyfunction!(
        fvlmm_assoc_chunk_from_snp_with_cache_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(lmm_rotate_x_y_with_ut_f64, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_assoc_packed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_assoc_packed_f32_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_assoc_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_assoc_from_snp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_assoc_packed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_assoc_packed_f32_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(fvlmm_assoc_packed_f32_to_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_reml_null_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_reml_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(bayesa, m)?)?;
    m.add_function(wrap_pyfunction!(bayesb, m)?)?;
    m.add_function(wrap_pyfunction!(bayescpi, m)?)?;
    m.add_function(wrap_pyfunction!(bayesa_packed, m)?)?;
    m.add_function(wrap_pyfunction!(bayesb_packed, m)?)?;
    m.add_function(wrap_pyfunction!(bayescpi_packed, m)?)?;
    m.add_function(wrap_pyfunction!(bayesa_packed_trace, m)?)?;
    m.add_function(wrap_pyfunction!(bayesb_packed_trace, m)?)?;
    m.add_function(wrap_pyfunction!(bayescpi_packed_trace, m)?)?;
    m.add_function(wrap_pyfunction!(admx_multiply_at_omega, m)?)?;
    m.add_function(wrap_pyfunction!(admx_multiply_a_omega, m)?)?;
    m.add_function(wrap_pyfunction!(admx_multiply_at_omega_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(admx_multiply_a_omega_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(admx_rsvd_power_step_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(admx_rsvd_stream, m)?)?;
    m.add_function(wrap_pyfunction!(admx_rsvd_stream_sample, m)?)?;
    m.add_function(wrap_pyfunction!(py_rsvd_packed_subset, m)?)?;
    m.add_function(wrap_pyfunction!(admx_allele_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(admx_loglikelihood, m)?)?;
    m.add_function(wrap_pyfunction!(admx_loglikelihood_f32, m)?)?;
    m.add_function(wrap_pyfunction!(admx_rmse_f32, m)?)?;
    m.add_function(wrap_pyfunction!(admx_rmse_f64, m)?)?;
    m.add_function(wrap_pyfunction!(admx_kl_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(admx_map_q_f32, m)?)?;
    m.add_function(wrap_pyfunction!(admx_map_p_f32, m)?)?;
    m.add_function(wrap_pyfunction!(admx_em_step, m)?)?;
    m.add_function(wrap_pyfunction!(admx_em_step_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(admx_em_step_inplace_f32, m)?)?;
    m.add_function(wrap_pyfunction!(admx_adam_update_p, m)?)?;
    m.add_function(wrap_pyfunction!(admx_adam_update_q, m)?)?;
    m.add_function(wrap_pyfunction!(admx_adam_update_p_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(admx_adam_update_q_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(admx_adam_update_p_inplace_f32, m)?)?;
    m.add_function(wrap_pyfunction!(admx_adam_update_q_inplace_f32, m)?)?;
    m.add_function(wrap_pyfunction!(admx_adam_optimize_f32, m)?)?;
    m.add_function(wrap_pyfunction!(admx_set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(fit_best_and_not_py, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_ml_feature_scores_py, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_ml_select_topk_py, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_subset_bin_samples_py, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_scan_groups_bin_py, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_scan_windows_bin_py, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_eval_rule_bin_py, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_logic_search_bed_py, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_prepare_input_bin_py, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_residualize_grm_py, m)?)?;
    m.add_function(wrap_pyfunction!(garfield_residualize_bed_py, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_bsa, m)?)?;
    m.add_function(wrap_pyfunction!(geno_chunk_to_alignment_u8, m)?)?;
    m.add_function(wrap_pyfunction!(geno_chunk_to_alignment_u8_siteinfo, m)?)?;
    m.add_function(wrap_pyfunction!(geno_chunk_to_alignment_u8_sites, m)?)?;
    m.add_function(wrap_pyfunction!(nj_newick_from_alignment_u8, m)?)?;
    m.add_function(wrap_pyfunction!(nj_newick_from_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(ml_newick_from_alignment_u8, m)?)?;
    phylo::register_py(m)?;
    Ok(())
}
