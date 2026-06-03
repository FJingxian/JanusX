#!/usr/bin/env python3
"""Compare old glmf32_packed vs new lm_block_assoc_packed: speed, memory, consistency."""

import argparse
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from janusx.janusx import (
    glmf32_packed,
    lm_block_assoc_packed,
    glm_ixx_from_x_qr,
    bed_packed_row_flip_mask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rss_memory_mb() -> float:
    """Resident set size in MB via /proc or psutil."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return float("nan")


def mem_peak_from_tracemalloc(trace) -> float:
    """Peak tracked Python memory from tracemalloc trace (MiB)."""
    return trace[1] / (1024 * 1024) if trace else float("nan")


MARKER_ROWS = 2000  # default SNP count for consistency checks


def _mean_impute_g(g_raw: np.ndarray, maf: np.ndarray) -> np.ndarray:
    """Mean-impute hardcall genotypes (0/1/2/NaN) -> (n x m) float32."""
    g = g_raw.astype(np.float32)
    for j in range(g.shape[1]):
        col = g[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            col[mask] = 2.0 * maf[j]
        g[:, j] = col
    return g


def simulate_data(
    n: int, m: int, q0: int, seed: int = 42
) -> dict:
    """
    Generate synthetic data:
      - n samples, m SNPs, q0=1 (intercept) or q0=k covariates
      - returns packed BED-like context, y, X, ixx, maf, flip, sample_indices
    """
    rng = np.random.default_rng(seed)

    # --- genotype ---
    maf = rng.uniform(0.05, 0.5, size=m).astype(np.float32)
    # generate dosages from binomial(2, maf) → hardcalls
    g_hard = np.empty((n, m), dtype=np.float32)
    for j in range(m):
        g_hard[:, j] = rng.binomial(2, maf[j], size=n).astype(np.float32)
    hc_flat = np.empty(n * m, dtype=np.uint8)
    # pack into BED format (0→0b00, 1→0b10, 2→0b11, missing→0b01)
    n_samples = n
    bytes_per_snp = (n_samples + 3) // 4
    packed_flat = np.zeros(m * bytes_per_snp, dtype=np.uint8)
    PLINK_DOSAGE_TO_CODE = np.array([0b00, 0b10, 0b11], dtype=np.uint8)
    for j in range(m):
        for i in range(n):
            code = int(PLINK_DOSAGE_TO_CODE[int(g_hard[i, j])])
            byte_idx = j * bytes_per_snp + (i >> 2)
            shift = (i & 3) * 2
            packed_flat[byte_idx] |= code << shift
    packed = packed_flat.reshape(m, bytes_per_snp)

    # --- flip mask ---
    flip_mask = np.zeros(m, dtype=bool)
    flip = rng.random(m) < 0.2  # flip ~20%
    flip_mask[flip] = True

    # --- covariates ---
    X = np.ones((n, 1 + q0), dtype=np.float64)
    if q0 > 0:
        X[:, 1:] = rng.normal(0, 1, (n, q0))

    # --- phenotype (computed before ixx since ixx_from_x_qr needs y) ---
    g_causal = np.zeros((n, 2), dtype=np.float64)
    for j in range(2):
        col = g_hard[:, j].copy().astype(np.float64)
        col[np.isnan(col)] = 2.0 * maf[j]
        g_causal[:, j] = col
    beta_causal = np.array([0.3, -0.2], dtype=np.float64)
    y = X @ np.ones(1 + q0) * 0.5 + g_causal @ beta_causal + rng.normal(0, 1.0, n)

    ixx = np.asarray(glm_ixx_from_x_qr(y, X))
    assert ixx.shape == (1 + q0, 1 + q0), f"ixx shape mismatch: {ixx.shape}"

    sample_indices = np.arange(n, dtype=np.int64)

    row_maf = maf.copy()
    # mean-centered maf for flip
    row_flip_bool = flip_mask.copy()

    return {
        "y": y,
        "X": X,
        "ixx": ixx,
        "packed": packed,
        "n_samples": n_samples,
        "row_flip": row_flip_bool,
        "row_maf": row_maf,
        "sample_indices": sample_indices,
        "g_hard": g_hard,
        "maf": maf,
    }


def run_old(data: dict, step: int = 10000, threads: int = 1) -> dict:
    """Run glmf32_packed (per-SNP dense path)."""
    t0 = time.perf_counter()
    res = np.asarray(
        glmf32_packed(
            data["y"],
            data["X"],
            data["ixx"],
            data["packed"],
            data["n_samples"],
            data["row_flip"],
            data["row_maf"],
            data["sample_indices"],
            step,
            threads,
        )
    )
    elapsed = time.perf_counter() - t0
    q0 = data["X"].shape[1]  # including intercept
    # cols: beta(0), se(1), p[0..q0-1](2..q0+1), p_snp(q0+2)
    beta = res[:, 0]
    se = res[:, 1]
    p_snp = res[:, -1]  # last column = SNP p-value
    return {"beta": beta, "se": se, "p": p_snp, "elapsed_s": elapsed, "full": res}


def run_new(data: dict, chunk_size: int = 10000, threads: int = 1) -> dict:
    """Run lm_block_assoc_packed (block BLAS)."""
    t0 = time.perf_counter()
    res = np.asarray(
        lm_block_assoc_packed(
            data["y"],
            data["X"],
            data["ixx"],
            data["packed"],
            data["n_samples"],
            data["row_flip"],
            data["row_maf"],
            data["sample_indices"],
            chunk_size,
            threads,
        )
    )
    elapsed = time.perf_counter() - t0
    # cols: beta(0), se(1), pwald(2), plrt(3)
    return {
        "beta": res[:, 0],
        "se": res[:, 1],
        "p": res[:, 2],
        "plrt": res[:, 3],
        "elapsed_s": elapsed,
        "full": res,
    }


def consistency_metrics(old: dict, new: dict) -> dict:
    """Compare beta / se / p between old and new."""
    beta_corr = np.corrcoef(old["beta"], new["beta"])[0, 1]
    beta_max_abs_diff = np.max(np.abs(old["beta"] - new["beta"]))
    se_corr = np.corrcoef(old["se"], new["se"])[0, 1]
    se_max_abs_diff = np.max(np.abs(old["se"] - new["se"]))
    # Compare old p_snp vs new pwald
    p_corr = np.corrcoef(old["p"], new["p"])[0, 1]
    p_max_abs_diff = np.max(np.abs(old["p"] - new["p"]))
    nans_old = np.sum(~np.isfinite(old["p"]))
    nans_new = np.sum(~np.isfinite(new["p"]))
    return {
        "beta_corr": beta_corr,
        "beta_max_abs_diff": beta_max_abs_diff,
        "se_corr": se_corr,
        "se_max_abs_diff": se_max_abs_diff,
        "p_corr": p_corr,
        "p_max_abs_diff": p_max_abs_diff,
        "p_nans_old": int(nans_old),
        "p_nans_new": int(nans_new),
    }


def scipy_ref(data: dict) -> dict:
    """Reference: scipy OLS per-SNP for the first 50 markers."""
    X = data["X"]
    y = data["y"]
    g = data["g_hard"]
    maf = data["maf"]
    n_markers = min(50, g.shape[1])
    beta_ref = np.zeros(n_markers)
    se_ref = np.zeros(n_markers)
    p_ref = np.zeros(n_markers)
    n = len(y)
    for j in range(n_markers):
        gj = g[:, j].astype(np.float64)
        mask = ~np.isfinite(gj)
        if mask.any():
            gj = gj.copy()
            gj[mask] = 2.0 * maf[j]
        Xj = np.column_stack([X, gj])
        try:
            coef, residuals, rank, svals = np.linalg.lstsq(Xj, y, rcond=None)
            beta_hat = coef[-1]
            rss = float(np.sum((y - Xj @ coef) ** 2))
            df = n - Xj.shape[1]
            ve = rss / df
            # se = sqrt(diag(XtX_inv) * ve), we need the last diagonal element
            XtX = Xj.T @ Xj
            XtX_inv = np.linalg.inv(XtX)
            se = np.sqrt(XtX_inv[-1, -1] * ve)
            t = beta_hat / se
            p = 2 * scipy_stats.t.sf(np.abs(t), df)
            beta_ref[j] = beta_hat
            se_ref[j] = se
            p_ref[j] = p
        except Exception:
            beta_ref[j] = np.nan
            se_ref[j] = np.nan
            p_ref[j] = np.nan
    return {"beta": beta_ref, "se": se_ref, "p": p_ref}


def benchmark_speed(data: dict, n_repeats: int = 3, threads: int = 1) -> dict:
    """Measure wall clock times for both backends with multiple repeats."""
    results = {"old": [], "new": []}
    # warmup
    _ = run_old(data, threads=threads)
    _ = run_new(data, threads=threads)

    for _ in range(n_repeats):
        old = run_old(data, threads=threads)
        new = run_new(data, threads=threads)
        results["old"].append(old)
        results["new"].append(new)

    old_times = [r["elapsed_s"] for r in results["old"]]
    new_times = [r["elapsed_s"] for r in results["new"]]
    return {
        "old_mean_s": np.mean(old_times),
        "old_min_s": np.min(old_times),
        "old_max_s": np.max(old_times),
        "new_mean_s": np.mean(new_times),
        "new_min_s": np.min(new_times),
        "new_max_s": np.max(new_times),
        "speedup": np.mean(old_times) / np.mean(new_times) if np.mean(new_times) > 0 else float("inf"),
        "n_repeats": n_repeats,
    }


def benchmark_memory(data: dict, threads: int = 1) -> dict:
    """Measure peak memory via tracemalloc."""
    # warmup
    _ = run_old(data, threads=threads)
    _ = run_new(data, threads=threads)

    tracemalloc.start()
    t0_mem = tracemalloc.take_snapshot()
    _ = run_old(data, threads=threads)
    old_trace = tracemalloc.get_traced_memory()
    old_snap = tracemalloc.take_snapshot()
    _ = run_new(data, threads=threads)
    new_trace = tracemalloc.get_traced_memory()
    new_snap = tracemalloc.take_snapshot()
    tracemalloc.stop()

    old_peak = mem_peak_from_tracemalloc(old_trace)
    new_peak = mem_peak_from_tracemalloc(new_trace)

    # also measure RSS
    rss_before = rss_memory_mb()
    run_old(data, threads=threads)
    rss_old = rss_memory_mb()
    run_new(data, threads=threads)
    rss_new = rss_memory_mb()

    return {
        "old_peak_tracemalloc_mb": old_peak,
        "new_peak_tracemalloc_mb": new_peak,
        "rss_before_mb": rss_before,
        "rss_old_mb": rss_old,
        "rss_new_mb": rss_new,
    }


def benchmark_chunk_sizes(data: dict, chunk_sizes: list, n_repeats: int = 3) -> dict:
    """Test how chunk_size affects new backend performance."""
    results = {}
    for cs in chunk_sizes:
        times = []
        # warmup
        _ = run_new(data, chunk_size=cs)
        for _ in range(n_repeats):
            r = run_new(data, chunk_size=cs)
            times.append(r["elapsed_s"])
        results[str(cs)] = {
            "mean_s": float(np.mean(times)),
            "min_s": float(np.min(times)),
            "max_s": float(np.max(times)),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LM block formula comparison benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--out", type=str, default=None, help="JSON summary output path")
    parser.add_argument("--quick", action="store_true", help="Smaller data for fast smoke test")
    args = parser.parse_args()

    configs = []
    if args.quick:
        configs = [
            {"n": 500, "m": 500, "q0": 1, "label": "n500_m500_q1"},
        ]
    else:
        configs = [
            {"n": 500, "m": 2000, "q0": 1, "label": "n500_m2000_q1"},
            {"n": 2000, "m": 2000, "q0": 1, "label": "n2k_m2k_q1"},
            {"n": 2000, "m": 2000, "q0": 3, "label": "n2k_m2k_q3"},
            {"n": 10000, "m": 5000, "q0": 1, "label": "n10k_m5k_q1"},
        ]

    all_results = {}

    for cfg in configs:
        label = cfg["label"]
        print(f"\n{'='*60}")
        print(f"  {label} (n={cfg['n']}, m={cfg['m']}, q0={cfg['q0']})")
        print(f"{'='*60}")

        data = simulate_data(cfg["n"], cfg["m"], cfg["q0"], seed=args.seed)

        # 1. Consistency check
        old = run_old(data, threads=args.threads)
        new = run_new(data, threads=args.threads)
        cons = consistency_metrics(old, new)
        print(f"  Consistency:")
        print(f"    beta  corr={cons['beta_corr']:.10f}  max_abs_diff={cons['beta_max_abs_diff']:.2e}")
        print(f"    se    corr={cons['se_corr']:.10f}    max_abs_diff={cons['se_max_abs_diff']:.2e}")
        print(f"    pval  corr={cons['p_corr']:.10f}  max_abs_diff={cons['p_max_abs_diff']:.2e}")

        # 2. Scipy reference for first 50 markers
        ref = scipy_ref(data)
        # Compare old vs scipy
        old_beta_ref = old["beta"][:50]
        old_p_ref = old["p"][:50]
        beta_ref_corr_old = np.corrcoef(old_beta_ref, ref["beta"])[0, 1]
        p_ref_corr_old = np.corrcoef(old_p_ref, ref["p"])[0, 1]
        new_beta_ref = new["beta"][:50]
        new_p_ref = new["p"][:50]
        beta_ref_corr_new = np.corrcoef(new_beta_ref, ref["beta"])[0, 1]
        p_ref_corr_new = np.corrcoef(new_p_ref, ref["p"])[0, 1]
        print(f"    vs-scipy (first 50 markers):")
        print(f"      old: beta_corr={beta_ref_corr_old:.10f}  p_corr={p_ref_corr_old:.10f}")
        print(f"      new: beta_corr={beta_ref_corr_new:.10f}  p_corr={p_ref_corr_new:.10f}")

        # 3. Speed benchmark
        speed = benchmark_speed(data, n_repeats=args.n_repeats, threads=args.threads)
        print(f"  Speed ({args.n_repeats} repeats):")
        print(f"    old:  mean={speed['old_mean_s']:.4f}s  min={speed['old_min_s']:.4f}s")
        print(f"    new:  mean={speed['new_mean_s']:.4f}s  min={speed['new_min_s']:.4f}s")
        print(f"    speedup: {speed['speedup']:.2f}x")

        # 4. Memory benchmark
        mem = benchmark_memory(data, threads=args.threads)
        print(f"  Memory:")
        print(f"    old peak (tracemalloc): {mem['old_peak_tracemalloc_mb']:.2f} MB")
        print(f"    new peak (tracemalloc): {mem['new_peak_tracemalloc_mb']:.2f} MB")
        print(f"    RSS: before={mem['rss_before_mb']:.1f}  old={mem['rss_old_mb']:.1f}  new={mem['rss_new_mb']:.1f} MB")

        # 5. Chunk size sweep
        if cfg["m"] >= 2000:
            chunk_sizes = [500, 2000, 5000, 10000, 20000]
        else:
            chunk_sizes = [100, 500, 1000]
        chunk_results = benchmark_chunk_sizes(data, chunk_sizes, n_repeats=max(2, args.n_repeats - 1))
        print(f"  Chunk size sweep:")
        best_cs = min(chunk_results.items(), key=lambda kv: kv[1]["mean_s"])
        for cs, r in sorted(chunk_results.items(), key=lambda kv: int(kv[0])):
            marker = " <-- best" if cs == best_cs[0] else ""
            print(f"    cs={cs:>6s}: mean={r['mean_s']:.4f}s{marker}")

        all_results[label] = {
            "config": cfg,
            "consistency": cons,
            "scipy_ref": {
                "old_beta_corr": float(beta_ref_corr_old),
                "old_p_corr": float(p_ref_corr_old),
                "new_beta_corr": float(beta_ref_corr_new),
                "new_p_corr": float(p_ref_corr_new),
            },
            "speed": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in speed.items()},
            "memory": {k: float(v) if isinstance(v, (int, float)) else v for k, v in mem.items()},
            "chunk_sizes": chunk_results,
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for label, res in all_results.items():
        s = res["speed"]
        c = res["consistency"]
        print(f"  {label}: speedup={s['speedup']:.2f}x  beta_diff={c['beta_max_abs_diff']:.2e}  p_corr={c['p_corr']:.6f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
