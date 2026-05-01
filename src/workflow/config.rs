#![allow(dead_code)]

#[derive(Debug, Clone)]
pub enum GenotypeInput {
    Vcf(String),
    Hmp(String),
    File(String),
    Bfile(String),
    Ndarray,
    BedPacked,
}

#[derive(Debug, Clone)]
pub enum GwasModel {
    Lm,
    Lmm,
    FastLmm,
    FarmCpu,
    LowRankLmm,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub threads: usize,
    pub chunk_size: usize,
    pub mmap_limit: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            threads: 1,
            chunk_size: 10_000,
            mmap_limit: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GwasConfig {
    pub genotype: GenotypeInput,
    pub phenotype_path: String,
    pub models: Vec<GwasModel>,
    pub traits: Option<Vec<usize>>,
    pub maf: f64,
    pub geno: f64,
    pub het: f64,
    pub snps_only: bool,
    pub model: String,
    pub grm: String,
    pub qcov: String,
    pub out_dir: String,
    pub out_prefix: Option<String>,
    pub runtime: RuntimeConfig,
}

#[derive(Debug, Clone)]
pub struct GsConfig {
    pub genotype: GenotypeInput,
    pub phenotype_path: String,
    pub out_dir: String,
    pub out_prefix: Option<String>,
    pub runtime: RuntimeConfig,
}
