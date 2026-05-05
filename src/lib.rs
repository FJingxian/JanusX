use pyo3::prelude::*;
use pyo3::Bound;

// stats
#[path = "stats/adamixture.rs"]
mod admixture;
#[path = "stats/assoc.rs"]
mod assoc;
#[path = "stats/bayes.rs"]
mod bayes;
#[path = "stats/beam.rs"]
pub mod beam;
#[path = "stats/bsa.rs"]
mod bsa;
#[path = "stats/glm.rs"]
mod glm;
#[path = "stats/gs_native.rs"]
pub(crate) mod gs_native;
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
#[path = "stats/score.rs"]
pub mod score;
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
#[path = "io/gmerge.rs"]
mod gmerge;
#[path = "io/gwasio.rs"]
mod gwasio;
#[path = "io/sim.rs"]
mod sim;
#[path = "io/vcfout.rs"]
mod vcfout;

// workflow (structural layer; currently type-only skeleton)
#[path = "workflow/mod.rs"]
mod workflow;

// math
#[path = "math/bedmath.rs"]
mod bedmath;
#[path = "math/bitwise.rs"]
mod bitwise;
#[path = "math/blas.rs"]
mod blas;
#[path = "math/brent.rs"]
mod brent;
#[path = "math/eigh.rs"]
mod eigh;
#[path = "math/linalg.rs"]
mod linalg;
#[path = "math/farmcpu.rs"]
mod math_farmcpu;
#[path = "math/ld.rs"]
mod math_ld;
#[path = "ml/mod.rs"]
mod ml;

use admixture::{
    admx_adam_optimize_f32, admx_adam_update_p, admx_adam_update_p_inplace,
    admx_adam_update_p_inplace_f32, admx_adam_update_q, admx_adam_update_q_inplace,
    admx_adam_update_q_inplace_f32, admx_allele_frequency, admx_em_step, admx_em_step_inplace,
    admx_em_step_inplace_f32, admx_kl_divergence, admx_loglikelihood, admx_loglikelihood_f32,
    admx_map_p_f32, admx_map_q_f32, admx_multiply_a_omega, admx_multiply_a_omega_inplace,
    admx_multiply_at_omega, admx_multiply_at_omega_inplace, admx_rmse_f32, admx_rmse_f64,
    admx_rsvd_power_step_inplace, admx_rsvd_stream, admx_rsvd_stream_sample, admx_set_threads,
};
use assoc::{farmcpu_rem_dense, farmcpu_rem_packed, farmcpu_super_dense, farmcpu_super_packed};
use bayes::{bayesa, bayesa_packed, bayesb, bayesb_packed, bayescpi, bayescpi_packed};
use beam::{
    beam_scan_windows_binary_mcc_bin_py, beam_scan_windows_continuous_corr_bin_py,
    beam_search_and_binary_mcc_bin_indices_py, beam_search_and_binary_mcc_bin_py,
    beam_search_and_continuous_corr_bin_indices_py, beam_search_and_continuous_corr_bin_py,
};
use bitwise::{and_popcount_py, bitand_assign_py, bitnot_masked_py, bitor_into_py, popcount_py};
use blas::{rust_blas_get_num_threads, rust_blas_set_num_threads, rust_sgemm_backend};
use bsa::preprocess_bsa;
use eigh::{rust_eigh_debug_f64, rust_eigh_from_array_f64, rust_eigh_from_array_f64_inplace};
use gfreader::{
    bed_filter_to_plink_rust, count_hmp_snps, count_vcf_snps, gfd_packbits_from_dosage_block,
    load_bed_2bit_packed, load_bed_u8_matrix, load_site_info, prepare_bed_2bit_packed,
    BedChunkReader, HmpChunkReader, HmpStreamWriter, PlinkStreamWriter, SiteInfo, TxtChunkReader,
    VcfChunkReader, VcfStreamWriter,
};
use glm::{glmf32, glmf32_full, glmf32_packed};
use gmerge::{convert_genotypes, merge_genotypes, PyConvertStats, PyMergeStats};
use gs_native::{
    gblup_reml_packed_bed, grm_packed_bed_f32, grm_packed_f32, grm_packed_f32_with_stats,
    grm_packed_f64_with_stats, rrblup_pcg_bed,
};
use gwasio::load_gwas_triplet_fast;
use ld::{bed_packed_ld_prune_maf_priority, bed_prune_to_plink_rust, packed_prune_kernel_stats};
use lmm::{fastlmm_reml_chunk_f32, fastlmm_reml_null_f32};
use lmm_scan::{
    ai_reml_multi_f64, ai_reml_null_f64, fastlmm_assoc_chunk_f32, fastlmm_assoc_packed_f32,
    lmm_assoc_chunk_f32, lmm_assoc_chunk_from_snp_f32, lmm_reml_assoc_packed_f32,
    lmm_reml_chunk_f32, lmm_reml_chunk_from_snp_f32, lmm_reml_null_f32, ml_loglike_null_f32,
};
use logreg::fit_best_and_not_py;
use ml::{garfield_ml_feature_scores_py, garfield_ml_select_topk_py};
use packed::{
    bed_packed_decode_rows_f32, bed_packed_decode_stats_f64, bed_packed_row_flip_mask,
    bed_packed_signed_hash_f32, bed_packed_signed_hash_kernels_f64,
    bed_packed_signed_hash_ztz_stats_f64, cross_grm_times_alpha_packed_f64, packed_malpha_f64,
};
use rsvd::py_rsvd_packed_subset;
use score::{
    score_binary_ba_mcc_batch_py, score_binary_ba_py, score_binary_mcc_py, score_cont_corr_py,
    score_cont_mean_diff_corr_batch_py, score_cont_mean_diff_py,
};
use sim::{sim_trait_accumulate_i8_f32, SimChunkGenerator, SimEngine, SimTraitAccumulator};
use top::{top_fit_model_py, top_rank_to_target_sample_py, top_rank_to_target_values_py};
use tree::{
    geno_chunk_to_alignment_u8, geno_chunk_to_alignment_u8_siteinfo,
    geno_chunk_to_alignment_u8_sites, ml_newick_from_alignment_u8, nj_newick_from_alignment_u8,
    nj_newick_from_distance_matrix,
};
// ============================================================
// PyO3 module exports
// ============================================================

#[pymodule]
fn janusx(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<BedChunkReader>()?;
    m.add_class::<VcfChunkReader>()?;
    m.add_class::<HmpChunkReader>()?;
    m.add_class::<TxtChunkReader>()?;
    m.add_class::<PlinkStreamWriter>()?;
    m.add_class::<VcfStreamWriter>()?;
    m.add_class::<HmpStreamWriter>()?;
    m.add_class::<SimChunkGenerator>()?;
    m.add_class::<SimEngine>()?;
    m.add_class::<SimTraitAccumulator>()?;
    m.add_class::<SiteInfo>()?;
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
    m.add_function(wrap_pyfunction!(prepare_bed_2bit_packed, m)?)?;
    m.add_function(wrap_pyfunction!(load_bed_u8_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(load_site_info, m)?)?;
    m.add_function(wrap_pyfunction!(gfd_packbits_from_dosage_block, m)?)?;
    m.add_function(wrap_pyfunction!(sim_trait_accumulate_i8_f32, m)?)?;
    m.add_function(wrap_pyfunction!(load_gwas_triplet_fast, m)?)?;
    m.add_function(wrap_pyfunction!(glmf32, m)?)?;
    m.add_function(wrap_pyfunction!(glmf32_full, m)?)?;
    m.add_function(wrap_pyfunction!(glmf32_packed, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_row_flip_mask, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_decode_rows_f32, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_decode_stats_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_ld_prune_maf_priority, m)?)?;
    m.add_function(wrap_pyfunction!(packed_prune_kernel_stats, m)?)?;
    m.add_function(wrap_pyfunction!(bed_prune_to_plink_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_signed_hash_f32, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_signed_hash_ztz_stats_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_signed_hash_kernels_f64, m)?)?;
    m.add_function(wrap_pyfunction!(bed_filter_to_plink_rust, m)?)?;
    m.add_function(wrap_pyfunction!(cross_grm_times_alpha_packed_f64, m)?)?;
    m.add_function(wrap_pyfunction!(packed_malpha_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rrblup_pcg_bed, m)?)?;
    m.add_function(wrap_pyfunction!(gblup_reml_packed_bed, m)?)?;
    m.add_function(wrap_pyfunction!(rust_sgemm_backend, m)?)?;
    m.add_function(wrap_pyfunction!(rust_blas_set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(rust_blas_get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_debug_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_from_array_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eigh_from_array_f64_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(top_fit_model_py, m)?)?;
    m.add_function(wrap_pyfunction!(top_rank_to_target_sample_py, m)?)?;
    m.add_function(wrap_pyfunction!(top_rank_to_target_values_py, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_bed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_f32_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(grm_packed_f64_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_rem_dense, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_rem_packed, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_super_dense, m)?)?;
    m.add_function(wrap_pyfunction!(farmcpu_super_packed, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_null_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ml_loglike_null_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ai_reml_null_f64, m)?)?;
    m.add_function(wrap_pyfunction!(ai_reml_multi_f64, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_chunk_from_snp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_assoc_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_assoc_chunk_from_snp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_assoc_packed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_assoc_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_assoc_packed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_reml_null_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_reml_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(bayesa, m)?)?;
    m.add_function(wrap_pyfunction!(bayesb, m)?)?;
    m.add_function(wrap_pyfunction!(bayescpi, m)?)?;
    m.add_function(wrap_pyfunction!(bayesa_packed, m)?)?;
    m.add_function(wrap_pyfunction!(bayesb_packed, m)?)?;
    m.add_function(wrap_pyfunction!(bayescpi_packed, m)?)?;
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
    m.add_function(wrap_pyfunction!(preprocess_bsa, m)?)?;
    m.add_function(wrap_pyfunction!(geno_chunk_to_alignment_u8, m)?)?;
    m.add_function(wrap_pyfunction!(geno_chunk_to_alignment_u8_siteinfo, m)?)?;
    m.add_function(wrap_pyfunction!(geno_chunk_to_alignment_u8_sites, m)?)?;
    m.add_function(wrap_pyfunction!(nj_newick_from_alignment_u8, m)?)?;
    m.add_function(wrap_pyfunction!(nj_newick_from_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(ml_newick_from_alignment_u8, m)?)?;
    Ok(())
}
