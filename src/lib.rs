use pyo3::prelude::*;
use pyo3::Bound;

mod gfcore;
mod gfreader;
mod gmerge;
mod assoc;
mod bayes;

use assoc::{glmf32, lmm_reml_chunk_f32, lmm_assoc_chunk_f32};
use gfreader::{SiteInfo, BedChunkReader, VcfChunkReader, PlinkStreamWriter, VcfStreamWriter,count_vcf_snps};
use gmerge::{merge_genotypes, PyMergeStats, convert_genotypes, PyConvertStats};
use bayes::{bayesa, bayesb};
// ============================================================
// PyO3 module exports
// ============================================================

#[pymodule]
fn janusx(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<BedChunkReader>()?;
    m.add_class::<VcfChunkReader>()?;
    m.add_class::<PlinkStreamWriter>()?;
    m.add_class::<VcfStreamWriter>()?;
    m.add_class::<SiteInfo>()?;
    m.add_class::<PyMergeStats>()?;
    m.add_class::<PyConvertStats>()?;
    m.add_function(wrap_pyfunction!(merge_genotypes, m)?)?;
    m.add_function(wrap_pyfunction!(convert_genotypes, m)?)?;
    m.add_function(wrap_pyfunction!(count_vcf_snps, m)?)?;
    m.add_function(wrap_pyfunction!(glmf32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_assoc_chunk_f32, m)?)?;
    m.add_function(wrap_pyfunction!(bayesa, m)?)?;
    m.add_function(wrap_pyfunction!(bayesb, m)?)?;
    Ok(())
}
