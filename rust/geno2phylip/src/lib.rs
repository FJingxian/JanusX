use pyo3::prelude::*;
use std::path::PathBuf;

mod core;
mod writers;
mod readers;

#[pyclass]
#[derive(Clone)]
pub struct ConvertOptions {
    #[pyo3(get, set)]
    pub out_prefix: String,
    #[pyo3(get, set)]
    pub out_dir: String,

    #[pyo3(get, set)]
    pub min_samples_locus: usize,
    #[pyo3(get, set)]
    pub outgroup: Option<String>,

    #[pyo3(get, set)]
    pub write_phylip: bool,
    #[pyo3(get, set)]
    pub write_fasta: bool,
    #[pyo3(get, set)]
    pub write_nexus: bool,
    #[pyo3(get, set)]
    pub write_nexus_binary: bool,

    #[pyo3(get, set)]
    pub resolve_het: bool,          // random resolve het to ref/alt in DNA
    #[pyo3(get, set)]
    pub used_sites: bool,           // write used sites tsv

    #[pyo3(get, set)]
    pub skip_non_biallelic: bool,   // VCF: skip sites not biallelic SNP
}

#[pymethods]
impl ConvertOptions {
    #[new]
    fn new(out_prefix: String) -> Self {
        Self {
            out_prefix,
            out_dir: "./".to_string(),
            min_samples_locus: 4,
            outgroup: None,
            write_phylip: true,
            write_fasta: false,
            write_nexus: false,
            write_nexus_binary: false,
            resolve_het: false,
            used_sites: false,
            skip_non_biallelic: true,
        }
    }
}

#[pyfunction]
fn convert_from_npz(npz: &str, snp: &str, idv: &str, opts: ConvertOptions) -> PyResult<()> {
    let npz = PathBuf::from(npz);
    let snp = PathBuf::from(snp);
    let idv = PathBuf::from(idv);
    core::convert_npz(&npz, &snp, &idv, opts).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn convert_from_vcf(vcf: &str, opts: ConvertOptions) -> PyResult<()> {
    let vcf = PathBuf::from(vcf);
    core::convert_vcf(&vcf, opts).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn convert_from_plink(bed: &str, bim: &str, fam: &str, opts: ConvertOptions) -> PyResult<()> {
    let bed = PathBuf::from(bed);
    let bim = PathBuf::from(bim);
    let fam = PathBuf::from(fam);
    core::convert_plink(&bed, &bim, &fam, opts).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn geno2phy(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ConvertOptions>()?;
    m.add_function(wrap_pyfunction!(convert_from_npz, m)?)?;
    m.add_function(wrap_pyfunction!(convert_from_vcf, m)?)?;
    m.add_function(wrap_pyfunction!(convert_from_plink, m)?)?;
    Ok(())
}