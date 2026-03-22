use pyo3::prelude::*;
use pyo3::Bound;

// stats
#[path = "stats/adamixture.rs"]
mod admixture;
#[path = "stats/rsvd.rs"]
mod rsvd;
#[path = "stats/assoc.rs"]
mod assoc;
#[path = "stats/bayes.rs"]
mod bayes;
#[path = "stats/bsa.rs"]
mod bsa;
#[path = "stats/lmm.rs"]
mod lmm;
#[path = "stats/logreg.rs"]
mod logreg;
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
#[path = "io/vcfout.rs"]
mod vcfout;

// math
#[path = "math/brent.rs"]
mod brent;
#[path = "math/linalg.rs"]
mod linalg;

use admixture::{
    admx_adam_optimize_f32, admx_adam_update_p, admx_adam_update_p_inplace,
    admx_adam_update_p_inplace_f32, admx_adam_update_q, admx_adam_update_q_inplace,
    admx_adam_update_q_inplace_f32, admx_allele_frequency, admx_em_step, admx_em_step_inplace,
    admx_em_step_inplace_f32, admx_kl_divergence, admx_loglikelihood, admx_loglikelihood_f32,
    admx_map_p_f32, admx_map_q_f32, admx_multiply_a_omega, admx_multiply_a_omega_inplace,
    admx_multiply_at_omega, admx_multiply_at_omega_inplace, admx_rmse_f32, admx_rmse_f64,
    admx_rsvd_power_step_inplace, admx_rsvd_stream, admx_rsvd_stream_sample, admx_set_threads,
};
use rsvd::py_rsvd_packed_subset;
use assoc::{
    ai_reml_multi_f64, ai_reml_null_f64, bed_packed_decode_rows_f32, bed_packed_row_flip_mask,
    farmcpu_rem_dense, farmcpu_rem_packed, farmcpu_super_dense, farmcpu_super_packed,
    fastlmm_assoc_chunk_f32, fastlmm_assoc_packed_f32, glmf32, glmf32_full, glmf32_packed, lmm_assoc_chunk_f32,
    lmm_assoc_chunk_from_snp_f32, lmm_reml_chunk_f32, lmm_reml_chunk_from_snp_f32,
    lmm_reml_null_f32, ml_loglike_null_f32,
};
use bayes::{bayesa, bayesb, bayescpi};
use bsa::preprocess_bsa;
use gfreader::{
    count_hmp_snps, count_vcf_snps, load_bed_2bit_packed, load_bed_u8_matrix, BedChunkReader,
    HmpChunkReader, HmpStreamWriter, PlinkStreamWriter, SiteInfo, TxtChunkReader, VcfChunkReader,
    VcfStreamWriter,
};
use gmerge::{convert_genotypes, merge_genotypes, PyConvertStats, PyMergeStats};
use gwasio::load_gwas_triplet_fast;
use lmm::{fastlmm_reml_chunk_f32, fastlmm_reml_null_f32};
use logreg::fit_best_and_not_py;
use tree::{
    geno_chunk_to_alignment_u8,
    geno_chunk_to_alignment_u8_siteinfo,
    geno_chunk_to_alignment_u8_sites,
    ml_newick_from_alignment_u8,
    nj_newick_from_alignment_u8,
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
    m.add_class::<SiteInfo>()?;
    m.add_class::<PyMergeStats>()?;
    m.add_class::<PyConvertStats>()?;
    m.add_function(wrap_pyfunction!(merge_genotypes, m)?)?;
    m.add_function(wrap_pyfunction!(convert_genotypes, m)?)?;
    m.add_function(wrap_pyfunction!(count_vcf_snps, m)?)?;
    m.add_function(wrap_pyfunction!(count_hmp_snps, m)?)?;
    m.add_function(wrap_pyfunction!(load_bed_2bit_packed, m)?)?;
    m.add_function(wrap_pyfunction!(load_bed_u8_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(load_gwas_triplet_fast, m)?)?;
    m.add_function(wrap_pyfunction!(glmf32, m)?)?;
    m.add_function(wrap_pyfunction!(glmf32_full, m)?)?;
    m.add_function(wrap_pyfunction!(glmf32_packed, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_row_flip_mask, m)?)?;
    m.add_function(wrap_pyfunction!(bed_packed_decode_rows_f32, m)?)?;
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
    m.add_function(wrap_pyfunction!(fastlmm_assoc_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_assoc_packed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_reml_null_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fastlmm_reml_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(bayesa, m)?)?;
    m.add_function(wrap_pyfunction!(bayesb, m)?)?;
    m.add_function(wrap_pyfunction!(bayescpi, m)?)?;
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
    m.add_function(wrap_pyfunction!(preprocess_bsa, m)?)?;
    m.add_function(wrap_pyfunction!(geno_chunk_to_alignment_u8, m)?)?;
    m.add_function(wrap_pyfunction!(geno_chunk_to_alignment_u8_siteinfo, m)?)?;
    m.add_function(wrap_pyfunction!(geno_chunk_to_alignment_u8_sites, m)?)?;
    m.add_function(wrap_pyfunction!(nj_newick_from_alignment_u8, m)?)?;
    m.add_function(wrap_pyfunction!(ml_newick_from_alignment_u8, m)?)?;
    Ok(())
}
