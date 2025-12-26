use pyo3::prelude::*;
use pyo3::Bound;

mod gfcore;
mod gfreader;
mod assoc;

use assoc::{glmf32, lmm_reml_chunk_f32};
use gfreader::{SiteInfo, BedChunkReader, VcfChunkReader, PlinkStreamWriter, VcfStreamWriter,count_vcf_snps};
// ============================================================
// PyO3 module exports
// ============================================================

#[pymodule]
fn JanusX_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<BedChunkReader>()?;
    m.add_class::<VcfChunkReader>()?;
    m.add_class::<PlinkStreamWriter>()?;
    m.add_class::<VcfStreamWriter>()?;
    m.add_class::<SiteInfo>()?;
    m.add_function(wrap_pyfunction!(count_vcf_snps, m)?)?;
    m.add_function(wrap_pyfunction!(glmf32, m)?)?;
    m.add_function(wrap_pyfunction!(lmm_reml_chunk_f32, m)?)?;
    Ok(())
}