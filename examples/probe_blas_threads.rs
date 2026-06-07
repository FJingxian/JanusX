use std::env;

use janusx::{
    janusx_rust_blas_get_num_threads, janusx_rust_blas_set_num_threads,
    janusx_rust_eigh_lapack_backend, janusx_rust_sgemm_backend,
};

const BLAS_ENV_KEYS: [&str; 8] = [
    "OMP_NUM_THREADS",
    "OMP_MAX_THREADS",
    "MKL_NUM_THREADS",
    "MKL_MAX_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
];

const SCHED_ENV_KEYS: [&str; 5] = [
    "SLURM_CPUS_PER_TASK",
    "PBS_NP",
    "LSB_DJOB_NUMPROC",
    "NSLOTS",
    "NCPUS",
];

fn parse_requests() -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for arg in env::args().skip(1) {
        for raw in arg.split(',') {
            let text = raw.trim();
            if text.is_empty() {
                continue;
            }
            if let Ok(v) = text.parse::<usize>() {
                if v > 0 {
                    out.push(v);
                }
            }
        }
    }
    if out.is_empty() {
        return vec![1, 2, 4, 8, 12, 24, 48];
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn main() {
    let requests = parse_requests();
    let host = env::var("HOSTNAME")
        .or_else(|_| env::var("HOST"))
        .unwrap_or_else(|_| "unknown".to_string());
    let avail = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1usize);

    println!("host={host}");
    println!("available_parallelism={avail}");
    println!("rust_sgemm_backend={}", janusx_rust_sgemm_backend());
    println!(
        "rust_eigh_lapack_backend={}",
        janusx_rust_eigh_lapack_backend()
    );
    println!("scheduler_env:");
    for key in SCHED_ENV_KEYS {
        if let Ok(val) = env::var(key) {
            println!("  {key}={val}");
        }
    }
    println!("initial_env:");
    for key in BLAS_ENV_KEYS {
        if let Ok(val) = env::var(key) {
            println!("  {key}={val}");
        }
    }
    println!(
        "initial_rust_blas_threads={}",
        janusx_rust_blas_get_num_threads()
    );
    println!();

    for req in requests {
        let ok = janusx_rust_blas_set_num_threads(req);
        let now = janusx_rust_blas_get_num_threads();
        println!("request={req} setter_ok={ok} rust_blas_threads={now}");
        for key in BLAS_ENV_KEYS {
            if let Ok(val) = env::var(key) {
                println!("  {key}={val}");
            }
        }
        println!();
    }
}
