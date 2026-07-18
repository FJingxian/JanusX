use std::hint::black_box;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
const SUM_Y_MASKLOAD_MIN_POPCNT: u32 = 10;

#[cfg(target_arch = "x86_64")]
const SUM_Y_MASK4_LUT: [[i64; 4]; 16] = [
    [0, 0, 0, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [-1, -1, 0, 0],
    [0, 0, -1, 0],
    [-1, 0, -1, 0],
    [0, -1, -1, 0],
    [-1, -1, -1, 0],
    [0, 0, 0, -1],
    [-1, 0, 0, -1],
    [0, -1, 0, -1],
    [-1, -1, 0, -1],
    [0, 0, -1, -1],
    [-1, 0, -1, -1],
    [0, -1, -1, -1],
    [-1, -1, -1, -1],
];

fn words_for_samples(n: usize) -> usize {
    n.div_ceil(64)
}

fn median_ns_per_call(samples_ns: &mut [f64]) -> f64 {
    samples_ns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples_ns[samples_ns.len() / 2]
}

fn make_sum_y_bench_inputs<F>(n_samples: usize, pred: F) -> (Vec<u64>, Vec<u64>, Vec<f64>)
where
    F: Fn(usize) -> (bool, bool),
{
    let mut lhs = vec![0u64; words_for_samples(n_samples).max(1)];
    let mut rhs = vec![0u64; words_for_samples(n_samples).max(1)];
    let mut y = Vec::<f64>::with_capacity(n_samples);
    for i in 0..n_samples {
        let (a, b) = pred(i);
        if a {
            lhs[i >> 6] |= 1u64 << (i & 63);
        }
        if b {
            rhs[i >> 6] |= 1u64 << (i & 63);
        }
        y.push(((i as f64) * 0.03125) - 7.0 + ((i % 13) as f64) * 0.0075);
    }
    (lhs, rhs, y)
}

fn sum_y_from_word_scalar(y: &[f64], base: usize, mut word: u64) -> f64 {
    let mut s = 0.0_f64;
    while word != 0 {
        let tz = word.trailing_zeros() as usize;
        s += y[base + tz];
        word &= word - 1;
    }
    s
}

fn sum_y_where_both1_scalar(lhs: &[u64], rhs: &[u64], y: &[f64], n_samples: usize) -> f64 {
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;

    let mut s1 = 0.0_f64;

    for w_idx in 0..full_words {
        let base = w_idx << 6;
        s1 += sum_y_from_word_scalar(y, base, lhs[w_idx] & rhs[w_idx]);
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        let base = full_words << 6;
        s1 += sum_y_from_word_scalar(y, base, (lhs[full_words] & rhs[full_words]) & mask);
    }

    s1
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_y_where_both1_avx2(lhs: &[u64], rhs: &[u64], y: &[f64], n_samples: usize) -> f64 {
    use core::arch::x86_64::{
        __m256d, __m256i, _mm256_add_pd, _mm256_loadu_pd, _mm256_loadu_si256, _mm256_maskload_pd,
        _mm256_setzero_pd, _mm256_storeu_pd,
    };

    let full_words = n_samples >> 6;
    let rem = n_samples & 63;
    let y_ptr = y.as_ptr();
    let mut scalar_tail = 0.0_f64;
    let mut acc0: __m256d = _mm256_setzero_pd();
    let mut acc1: __m256d = _mm256_setzero_pd();
    let mut acc2: __m256d = _mm256_setzero_pd();
    let mut acc3: __m256d = _mm256_setzero_pd();

    let mut accumulate_word = |base: usize, word: u64, valid_bits: usize| {
        if word == 0 {
            return;
        }
        if word.count_ones() <= SUM_Y_MASKLOAD_MIN_POPCNT {
            scalar_tail += sum_y_from_word_scalar(y, base, word);
            return;
        }
        let blocks = valid_bits.div_ceil(4);
        for block_idx in 0..blocks {
            let nib = ((word >> (block_idx << 2)) & 0xFu64) as usize;
            if nib == 0 {
                continue;
            }
            let ptr = unsafe { y_ptr.add(base + (block_idx << 2)) };
            let vals = if nib == 0xF {
                unsafe { _mm256_loadu_pd(ptr) }
            } else {
                let mask =
                    unsafe { _mm256_loadu_si256(SUM_Y_MASK4_LUT[nib].as_ptr().cast::<__m256i>()) };
                unsafe { _mm256_maskload_pd(ptr, mask) }
            };
            match block_idx & 3 {
                0 => acc0 = _mm256_add_pd(acc0, vals),
                1 => acc1 = _mm256_add_pd(acc1, vals),
                2 => acc2 = _mm256_add_pd(acc2, vals),
                _ => acc3 = _mm256_add_pd(acc3, vals),
            }
        }
    };

    for w_idx in 0..full_words {
        accumulate_word(w_idx << 6, lhs[w_idx] & rhs[w_idx], 64);
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        accumulate_word(
            full_words << 6,
            (lhs[full_words] & rhs[full_words]) & mask,
            rem,
        );
    }

    let acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
    let mut lanes = [0.0_f64; 4];
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
    scalar_tail + lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

fn bench_sum_y_backend<F>(
    lhs: &[u64],
    rhs: &[u64],
    y: &[f64],
    n_samples: usize,
    iters: usize,
    mut f: F,
) -> (f64, f64)
where
    F: FnMut(&[u64], &[u64], &[f64], usize) -> f64,
{
    black_box(f(lhs, rhs, y, n_samples));
    let mut samples = [0.0_f64; 5];
    let mut sink = 0.0_f64;
    for slot in samples.iter_mut() {
        let t0 = Instant::now();
        for _ in 0..iters {
            sink += black_box(f(lhs, rhs, y, n_samples));
        }
        let elapsed = t0.elapsed().as_secs_f64();
        *slot = elapsed * 1e9 / (iters as f64);
    }
    black_box(sink);
    let median = median_ns_per_call(samples.as_mut_slice());
    let checksum = f(lhs, rhs, y, n_samples);
    (median, checksum)
}

fn parse_usize_flag(args: &[String], flag: &str, default: usize) -> Result<usize, String> {
    let mut idx = 0usize;
    while idx < args.len() {
        if args[idx] == flag {
            let Some(value) = args.get(idx + 1) else {
                return Err(format!("{flag} requires an integer value"));
            };
            return value
                .parse::<usize>()
                .map_err(|_| format!("{flag} expects a positive integer, got {value:?}"));
        }
        idx += 1;
    }
    Ok(default)
}

fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let n_samples = parse_usize_flag(args.as_slice(), "--samples", 65_536usize)?;
    let iters = parse_usize_flag(args.as_slice(), "--iters", 512usize)?;
    if n_samples == 0 {
        return Err("--samples must be > 0".to_string());
    }
    if iters == 0 {
        return Err("--iters must be > 0".to_string());
    }

    let sparse = make_sum_y_bench_inputs(n_samples, |i| {
        let lhs = (i % 29) == 0 || (i % 31) == 1 || (i % 37) == 3;
        let rhs = (i % 41) == 0 || (i % 43) == 2 || (i % 47) == 4;
        (lhs, rhs)
    });
    let dense = make_sum_y_bench_inputs(n_samples, |i| {
        let lhs = ((i & 1) == 0) || ((i % 7) <= 2) || ((i % 19) <= 6);
        let rhs = ((i % 3) != 1) || ((i % 11) <= 4) || ((i % 23) <= 7);
        (lhs, rhs)
    });

    let run_case = |label: &str, lhs: &[u64], rhs: &[u64], y: &[f64]| -> Result<(), String> {
        let (scalar_ns, scalar_sum) =
            bench_sum_y_backend(lhs, rhs, y, n_samples, iters, sum_y_where_both1_scalar);
        #[cfg(not(target_arch = "x86_64"))]
        let _ = scalar_sum;
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                let (avx2_ns, avx2_sum) =
                    bench_sum_y_backend(lhs, rhs, y, n_samples, iters, |a, b, c, n| unsafe {
                        sum_y_where_both1_avx2(a, b, c, n)
                    });
                let diff = (scalar_sum - avx2_sum).abs();
                let tol = scalar_sum.abs().max(1.0) * 1e-10;
                if diff > tol {
                    return Err(format!(
                        "checksum drift too large for {label}: diff={diff:.3e} tol={tol:.3e}"
                    ));
                }
                println!(
                    "[sum_y bench][{label}] scalar={scalar_ns:.1} ns/call avx2={avx2_ns:.1} ns/call speedup={:.2}x diff={diff:.3e}",
                    scalar_ns / avx2_ns,
                );
                return Ok(());
            }
        }
        println!("[sum_y bench][{label}] simd backend unavailable; scalar={scalar_ns:.1} ns/call");
        Ok::<(), String>(())
    };

    run_case(
        "sparse",
        sparse.0.as_slice(),
        sparse.1.as_slice(),
        sparse.2.as_slice(),
    )?;
    run_case(
        "dense",
        dense.0.as_slice(),
        dense.1.as_slice(),
        dense.2.as_slice(),
    )?;
    Ok(())
}
