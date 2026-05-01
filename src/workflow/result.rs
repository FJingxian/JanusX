#![allow(dead_code)]

#[derive(Debug, Clone)]
pub struct ModelResult {
    pub model: String,
    pub trait_name: String,
    pub result_file: String,
    pub summary_file: Option<String>,
    pub lambda_gc: Option<f64>,
    pub pve: Option<f64>,
    pub n_sites_tested: usize,
    pub elapsed_sec: f64,
}

#[derive(Debug, Clone)]
pub struct GwasResult {
    pub output_dir: String,
    pub prefix: String,
    pub n_samples: usize,
    pub n_sites_total: usize,
    pub n_sites_used: usize,
    pub n_traits: usize,
    pub elapsed_sec: f64,
    pub models: Vec<ModelResult>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GsResult {
    pub output_dir: String,
    pub prefix: String,
    pub n_samples: usize,
    pub n_sites: usize,
    pub elapsed_sec: f64,
    pub warnings: Vec<String>,
}
