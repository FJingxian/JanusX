use crate::blas::OpenBlasThreadGuard;
use crate::grm::{
    grm_packed_f32_core_no_progress, grm_packed_f64_core_no_progress, grm_rankk_update_raw,
    grm_rankk_update_raw_f64, GrmAccumMode,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const GGVAL_DEFAULT_N_INDIVIDUALS: usize = 5_000;
pub const GGVAL_DEFAULT_NSNP_K: usize = 250;
pub const GGVAL_DEFAULT_SEED: u64 = 20260520_u64;
pub const DEFAULT_BLOCK_COLS: usize = 65_536;
pub const DEFAULT_ACCUM_ROWS: usize = 4_096;
pub const DEFAULT_MISSING_RATE: f32 = 0.01_f32;

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchAccumKernel {
    Syrk,
    Gemm,
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct GrmBenchCase {
    pub n_samples: usize,
    pub n_snps: usize,
    pub bytes_per_snp: usize,
    pub accum_rows: usize,
    pub method: usize,
    pub block_cols: usize,
    pub seed: u64,
    pub missing_rate: f32,
    pub packed: Vec<u8>,
    pub row_flip: Vec<bool>,
    pub row_maf: Vec<f32>,
    pub sample_idx: Vec<usize>,
    pub accum_block_f32: Vec<f32>,
    pub accum_block_f64: Vec<f64>,
}

#[doc(hidden)]
pub fn ggval_default_case(seed: u64) -> GrmBenchCase {
    generate_grm_bench_case(
        GGVAL_DEFAULT_N_INDIVIDUALS,
        GGVAL_DEFAULT_NSNP_K * 1_000,
        DEFAULT_ACCUM_ROWS,
        seed,
        DEFAULT_MISSING_RATE,
        1,
        DEFAULT_BLOCK_COLS,
    )
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn generate_grm_bench_case(
    n_samples: usize,
    n_snps: usize,
    accum_rows: usize,
    seed: u64,
    missing_rate: f32,
    method: usize,
    block_cols: usize,
) -> GrmBenchCase {
    let n = n_samples.max(1);
    let m = n_snps.max(1);
    let accum_ct = accum_rows.max(1).min(m);
    let bytes_per_snp = (n + 3) / 4;
    let mut packed = vec![0_u8; m * bytes_per_snp];
    let mut row_flip = vec![false; m];
    let mut row_maf = vec![0.0_f32; m];
    let mut accum_block_f32 = vec![0.0_f32; accum_ct * n];
    let mut accum_block_f64 = vec![0.0_f64; accum_ct * n];
    let sample_idx = (0..n).collect::<Vec<usize>>();
    let mut rng = StdRng::seed_from_u64(seed);
    let miss = missing_rate.clamp(0.0_f32, 0.25_f32);

    for row_idx in 0..m {
        let p_minor = (0.02_f32 + rng.random::<f32>() * 0.48_f32).clamp(0.02_f32, 0.5_f32);
        let flip = rng.random_bool(0.5);
        let p_raw = if flip {
            (1.0_f32 - p_minor).clamp(0.5_f32, 0.98_f32)
        } else {
            p_minor
        };
        row_flip[row_idx] = flip;
        row_maf[row_idx] = p_minor;
        let mean_g = 2.0_f32 * p_minor;
        let row_off = row_idx * bytes_per_snp;

        let g0 = (1.0_f32 - p_raw) * (1.0_f32 - p_raw);
        let g1 = 2.0_f32 * p_raw * (1.0_f32 - p_raw);

        for sample_idx0 in 0..n {
            let u = rng.random::<f32>();
            let (code, centered) = if u < miss {
                (0b01_u8, 0.0_f32)
            } else {
                let v = (u - miss) / (1.0_f32 - miss).max(1e-12_f32);
                let raw_count = if v < g0 {
                    0_u8
                } else if v < (g0 + g1) {
                    1_u8
                } else {
                    2_u8
                };
                let minor_count = if flip {
                    2.0_f32 - raw_count as f32
                } else {
                    raw_count as f32
                };
                let code = match raw_count {
                    0 => 0b00_u8,
                    1 => 0b10_u8,
                    _ => 0b11_u8,
                };
                (code, minor_count - mean_g)
            };

            let byte_idx = row_off + (sample_idx0 >> 2);
            let shift = ((sample_idx0 & 3) * 2) as u8;
            packed[byte_idx] |= code << shift;

            if row_idx < accum_ct {
                let pos = row_idx * n + sample_idx0;
                accum_block_f32[pos] = centered;
                accum_block_f64[pos] = centered as f64;
            }
        }
    }

    GrmBenchCase {
        n_samples: n,
        n_snps: m,
        bytes_per_snp,
        accum_rows: accum_ct,
        method,
        block_cols: block_cols.max(1),
        seed,
        missing_rate: miss,
        packed,
        row_flip,
        row_maf,
        sample_idx,
        accum_block_f32,
        accum_block_f64,
    }
}

#[doc(hidden)]
pub fn run_packed_grm_f32_digest(case: &GrmBenchCase, threads: usize) -> Result<[f64; 4], String> {
    let (grm, row_sum, varsum) = grm_packed_f32_core_no_progress(
        &case.packed,
        case.n_samples,
        &case.row_flip,
        &case.row_maf,
        &case.sample_idx,
        case.method,
        case.block_cols,
        threads,
    )?;
    Ok(full_digest_f32(&grm, case.n_samples, &row_sum, varsum))
}

#[doc(hidden)]
pub fn run_packed_grm_f64_digest(case: &GrmBenchCase, threads: usize) -> Result<[f64; 4], String> {
    let (grm, row_sum, varsum) = grm_packed_f64_core_no_progress(
        &case.packed,
        case.n_samples,
        &case.row_flip,
        &case.row_maf,
        &case.sample_idx,
        case.method,
        case.block_cols,
        threads,
    )?;
    Ok(full_digest_f64(&grm, case.n_samples, &row_sum, varsum))
}

#[doc(hidden)]
pub fn run_accum_only_f32_digest(
    case: &GrmBenchCase,
    threads: usize,
    kernel: BenchAccumKernel,
) -> Result<[f64; 4], String> {
    let n = case.n_samples;
    let mut grm = vec![0.0_f32; n * n];
    let _blas_guard = OpenBlasThreadGuard::enter(threads.max(1));
    grm_rankk_update_raw(
        grm.as_mut_ptr(),
        &case.accum_block_f32,
        case.accum_rows,
        n,
        accum_mode(kernel),
        false,
        true,
        false,
    )?;
    Ok(accum_digest_f32(&grm, n))
}

#[doc(hidden)]
pub fn run_accum_only_f64_digest(
    case: &GrmBenchCase,
    threads: usize,
    kernel: BenchAccumKernel,
) -> Result<[f64; 4], String> {
    let n = case.n_samples;
    let mut grm = vec![0.0_f64; n * n];
    let _blas_guard = OpenBlasThreadGuard::enter(threads.max(1));
    grm_rankk_update_raw_f64(
        grm.as_mut_ptr(),
        &case.accum_block_f64,
        case.accum_rows,
        n,
        accum_mode(kernel),
        false,
        true,
        false,
    )?;
    Ok(accum_digest_f64(&grm, n))
}

fn accum_mode(kernel: BenchAccumKernel) -> GrmAccumMode {
    match kernel {
        BenchAccumKernel::Syrk => GrmAccumMode::Syrk,
        BenchAccumKernel::Gemm => GrmAccumMode::Gemm,
    }
}

fn full_digest_f32(grm: &[f32], n: usize, row_sum: &[f64], varsum: f64) -> [f64; 4] {
    let mid = n / 2;
    [
        grm[0] as f64,
        grm[mid * n + mid] as f64,
        grm[n * n - 1] as f64,
        varsum + row_sum.first().copied().unwrap_or(0.0),
    ]
}

fn full_digest_f64(grm: &[f64], n: usize, row_sum: &[f64], varsum: f64) -> [f64; 4] {
    let mid = n / 2;
    [
        grm[0],
        grm[mid * n + mid],
        grm[n * n - 1],
        varsum + row_sum.first().copied().unwrap_or(0.0),
    ]
}

fn accum_digest_f32(grm: &[f32], n: usize) -> [f64; 4] {
    let mid = n / 2;
    [
        grm[0] as f64,
        grm[mid * n + mid] as f64,
        grm[n * n - 1] as f64,
        grm.iter().take(64).map(|&v| v as f64).sum::<f64>(),
    ]
}

fn accum_digest_f64(grm: &[f64], n: usize) -> [f64; 4] {
    let mid = n / 2;
    [
        grm[0],
        grm[mid * n + mid],
        grm[n * n - 1],
        grm.iter().take(64).copied().sum::<f64>(),
    ]
}
