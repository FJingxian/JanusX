use criterion::{black_box, criterion_group, criterion_main, Criterion};
use janusx::grm_bench_support::{
    generate_grm_bench_case, ggval_default_case, run_accum_only_f32_digest,
    run_accum_only_f64_digest, run_packed_grm_f32_digest, run_packed_grm_f64_digest,
    BenchAccumKernel, GrmBenchCase, DEFAULT_ACCUM_ROWS, DEFAULT_BLOCK_COLS, DEFAULT_MISSING_RATE,
    GGVAL_DEFAULT_NSNP_K, GGVAL_DEFAULT_N_INDIVIDUALS, GGVAL_DEFAULT_SEED,
};
use std::sync::OnceLock;
use std::time::Duration;

fn env_usize(name: &str, default: usize, min_value: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|v| v.max(min_value))
        .unwrap_or(default.max(min_value))
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32, min_value: f32, max_value: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .map(|v| v.clamp(min_value, max_value))
        .unwrap_or(default.clamp(min_value, max_value))
}

fn bench_threads() -> Vec<usize> {
    let raw = std::env::var("JX_GRM_DTYPE_BENCH_THREADS").unwrap_or_else(|_| "1,8".to_string());
    let mut out = raw
        .split(',')
        .filter_map(|tok| tok.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .collect::<Vec<usize>>();
    if out.is_empty() {
        out.extend_from_slice(&[1, 8]);
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn bench_case() -> &'static GrmBenchCase {
    static CASE: OnceLock<GrmBenchCase> = OnceLock::new();
    CASE.get_or_init(|| {
        let n_samples = env_usize("JX_GRM_DTYPE_BENCH_NIND", GGVAL_DEFAULT_N_INDIVIDUALS, 2);
        let nsnp_k = env_usize("JX_GRM_DTYPE_BENCH_NSNP_K", GGVAL_DEFAULT_NSNP_K, 1);
        let n_snps = nsnp_k.saturating_mul(1_000).max(1);
        let accum_rows =
            env_usize("JX_GRM_DTYPE_BENCH_ACCUM_ROWS", DEFAULT_ACCUM_ROWS, 1).min(n_snps);
        let seed = env_u64("JX_GRM_DTYPE_BENCH_SEED", GGVAL_DEFAULT_SEED);
        let missing_rate = env_f32(
            "JX_GRM_DTYPE_BENCH_MISSING_RATE",
            DEFAULT_MISSING_RATE,
            0.0,
            0.25,
        );
        let block_cols = env_usize("JX_GRM_DTYPE_BENCH_BLOCK_COLS", DEFAULT_BLOCK_COLS, 1);

        if n_samples == GGVAL_DEFAULT_N_INDIVIDUALS
            && nsnp_k == GGVAL_DEFAULT_NSNP_K
            && accum_rows == DEFAULT_ACCUM_ROWS
            && (missing_rate - DEFAULT_MISSING_RATE).abs() < f32::EPSILON
            && block_cols == DEFAULT_BLOCK_COLS
            && seed == GGVAL_DEFAULT_SEED
        {
            ggval_default_case(seed)
        } else {
            generate_grm_bench_case(
                n_samples,
                n_snps,
                accum_rows,
                seed,
                missing_rate,
                1,
                block_cols,
            )
        }
    })
}

fn bench_full_packed(c: &mut Criterion) {
    let case = bench_case();
    for threads in bench_threads() {
        let id_f32 = format!("full_packed_f32_t{threads}");
        c.bench_function(&id_f32, |b| {
            b.iter(|| {
                black_box(run_packed_grm_f32_digest(case, threads).expect("full packed f32 failed"))
            })
        });

        let id_f64 = format!("full_packed_f64_t{threads}");
        c.bench_function(&id_f64, |b| {
            b.iter(|| {
                black_box(run_packed_grm_f64_digest(case, threads).expect("full packed f64 failed"))
            })
        });
    }
}

fn bench_accum(c: &mut Criterion) {
    let case = bench_case();
    for threads in bench_threads() {
        for (label, kernel) in [
            ("accum_syrk", BenchAccumKernel::Syrk),
            ("accum_gemm", BenchAccumKernel::Gemm),
        ] {
            let id_f32 = format!("{label}_f32_t{threads}");
            c.bench_function(&id_f32, |b| {
                b.iter(|| {
                    black_box(
                        run_accum_only_f32_digest(case, threads, kernel)
                            .expect("accum-only f32 failed"),
                    )
                })
            });

            let id_f64 = format!("{label}_f64_t{threads}");
            c.bench_function(&id_f64, |b| {
                b.iter(|| {
                    black_box(
                        run_accum_only_f64_digest(case, threads, kernel)
                            .expect("accum-only f64 failed"),
                    )
                })
            });
        }
    }
}

fn criterion_config() -> Criterion {
    let sample_size = env_usize("JX_GRM_DTYPE_BENCH_SAMPLE_SIZE", 10, 10);
    let warmup_sec = env_usize("JX_GRM_DTYPE_BENCH_WARMUP_SEC", 3, 1) as u64;
    let measure_sec = env_usize("JX_GRM_DTYPE_BENCH_MEASURE_SEC", 12, 1) as u64;
    Criterion::default()
        .sample_size(sample_size)
        .warm_up_time(Duration::from_secs(warmup_sec))
        .measurement_time(Duration::from_secs(measure_sec))
        .configure_from_args()
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_full_packed, bench_accum
}
criterion_main!(benches);
