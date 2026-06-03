"""Benchmark packed vs memmap GRM backends on cubic dataset."""

import time
from janusx import janusx as jxrs

PREFIX = "/Users/jingxianfu/cubic/cubic_All.maf0.02"
MAF = 0.001
MISS = 0.1
THREADS = 8


def bench(name: str, fn, warmup: bool = False):
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    tag = " [warmup]" if warmup else ""
    if isinstance(result, tuple):
        grm, eff_m, n = result
        print(
            f"  {name}: {elapsed:.1f}s, GRM shape={grm.shape}, "
            f"eff_m={eff_m}, n={n}{tag}"
        )
    else:
        print(f"  {name}: {elapsed:.1f}s{tag}")
    return elapsed


def main():
    print(f"Dataset: {PREFIX}")
    print(f"MAF={MAF}, MISS={MISS}, THREADS={THREADS}")
    print()

    print("=== f64 GRM benchmarks ===")
    print()

    # Warmup: run packed first to populate FS cache
    print("Warmup (packed, populates FS cache)...")
    bench(
        "packed_bed_f64(warmup)",
        lambda: jxrs.grm_packed_bed_f64(PREFIX, 1, MAF, MISS, 65536, THREADS, None, 0),
        warmup=True,
    )
    print()

    # Benchmark: packed x3
    print("Benchmark packed_bed_f64 x3:")
    for i in range(3):
        bench(
            f"packed_bed_f64 run {i + 1}",
            lambda: jxrs.grm_packed_bed_f64(PREFIX, 1, MAF, MISS, 65536, THREADS, None, 0),
        )
    print()

    # Benchmark: stream x3
    print("Benchmark stream_bed_f64 x3:")
    # First stream run may be IO-bound (cold mmap)
    for i in range(3):
        bench(
            f"stream_bed_f64 run {i + 1}",
            lambda: jxrs.grm_stream_bed_f64(PREFIX, 1, MAF, MISS, 65536, THREADS, None, 0),
        )


if __name__ == "__main__":
    main()
