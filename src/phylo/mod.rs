use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Bound;

mod convert;
mod writers;

#[pyclass]
#[derive(Clone)]
pub struct FastTreePrepOptions {
    #[pyo3(get, set)]
    pub out_prefix: String,
    #[pyo3(get, set)]
    pub outgroup: Option<String>,
    #[pyo3(get, set)]
    pub write_phylip: bool,
    #[pyo3(get, set)]
    pub resolve_het: bool,
    #[pyo3(get, set)]
    pub maf: f64,
    #[pyo3(get, set)]
    pub missing_rate: f64,
    #[pyo3(get, set)]
    pub het: f64,
    #[pyo3(get, set)]
    pub threads: usize,
}

#[pymethods]
impl FastTreePrepOptions {
    #[new]
    fn new(out_prefix: String) -> Self {
        Self {
            out_prefix,
            outgroup: None,
            write_phylip: false,
            resolve_het: false,
            maf: 0.02,
            missing_rate: 0.05,
            het: 0.02,
            threads: 0,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct FastTreePrepResult {
    #[pyo3(get)]
    pub fasta_path: String,
    #[pyo3(get)]
    pub phylip_path: Option<String>,
    #[pyo3(get)]
    pub n_samples: usize,
    #[pyo3(get)]
    pub n_sites: usize,
    #[pyo3(get)]
    pub skipped_sites: usize,
}

#[pyfunction]
fn fasttree_prepare_alignment(
    input_kind: &str,
    input_path: &str,
    opts: FastTreePrepOptions,
) -> PyResult<FastTreePrepResult> {
    if !(0.0..=0.5).contains(&opts.maf) {
        return Err(PyValueError::new_err(
            "FastTree prep `maf` must be in [0, 0.5].",
        ));
    }
    if !(0.0..=1.0).contains(&opts.missing_rate) {
        return Err(PyValueError::new_err(
            "FastTree prep `missing_rate` must be in [0, 1].",
        ));
    }
    if !(0.0..=1.0).contains(&opts.het) {
        return Err(PyValueError::new_err(
            "FastTree prep `het` must be in [0, 1].",
        ));
    }
    convert::prepare_alignment(input_kind, input_path, &opts).map_err(PyValueError::new_err)
}

pub fn register_py(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<FastTreePrepOptions>()?;
    m.add_class::<FastTreePrepResult>()?;
    m.add_function(wrap_pyfunction!(fasttree_prepare_alignment, m)?)?;
    Ok(())
}
