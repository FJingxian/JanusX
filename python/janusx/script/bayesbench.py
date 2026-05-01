#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightweight packed-Bayes kernel benchmark.

Usage example:
  python -m janusx.script.bayesbench --n-samples 1200 --n-snps 10000 --repeat 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _bootstrap_repo_python_path() -> None:
    if __package__:
        return
    here = Path(__file__).resolve()
    py_root = here.parents[2]
    if str(py_root) not in sys.path:
        sys.path.insert(0, str(py_root))


_bootstrap_repo_python_path()

from janusx import janusx as _jxrs  # noqa: E402
from janusx.script._common.helptext import (  # noqa: E402
    CliArgumentParser,
    cli_help_formatter,
    minimal_help_epilog,
)


_METHOD_ALIASES = {
    "bayesa": "BayesA",
    "bayesb": "BayesB",
    "bayescpi": "BayesCpi",
}


def _parse_methods(raw: str) -> list[str]:
    vals: list[str] = []
    seen: set[str] = set()
    for part in str(raw).split(","):
        key = part.strip().lower()
        if not key:
            continue
        m = _METHOD_ALIASES.get(key, None)
        if m is None:
            raise ValueError(f"Unsupported method token: {part!r}. Use BayesA,BayesB,BayesCpi.")
        if m in seen:
            continue
        seen.add(m)
        vals.append(m)
    if len(vals) == 0:
        raise ValueError("No valid methods parsed from --methods.")
    return vals


def _parse_row_blocks(raw: str) -> list[int | None]:
    vals: list[int | None] = []
    seen: set[int | None] = set()
    for part in str(raw).split(","):
        tok = part.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except Exception as exc:
            raise ValueError(f"Invalid --row-block value: {part!r}") from exc
        key: int | None = (v if v > 0 else None)
        if key in seen:
            continue
        seen.add(key)
        vals.append(key)
    if len(vals) == 0:
        vals = [None]
    return vals


def _pack_bed_from_dosage(geno: np.ndarray) -> np.ndarray:
    m, n = int(geno.shape[0]), int(geno.shape[1])
    bps = (n + 3) // 4
    out = np.zeros((m, bps), dtype=np.uint8)
    code_map = np.array([0b00, 0b10, 0b11], dtype=np.uint8)
    for r in range(m):
        row = np.asarray(geno[r], dtype=np.int8)
        for i in range(n):
            g = int(row[i])
            if g < 0 or g > 2:
                code = 0b01
            else:
                code = int(code_map[g])
            out[r, i >> 2] |= np.uint8(code << ((i & 3) * 2))
    return out


def _run_one_kernel(
    *,
    method: str,
    y: np.ndarray,
    packed: np.ndarray,
    n_samples: int,
    row_flip: np.ndarray,
    row_maf: np.ndarray,
    row_mean: np.ndarray,
    row_inv_sd: np.ndarray,
    sample_indices: np.ndarray,
    n_iter: int,
    burnin: int,
    thin: int,
    r2: float,
    counts: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    common: dict[str, Any] = {
        "y": y,
        "packed": packed,
        "n_samples": int(n_samples),
        "row_flip": row_flip,
        "row_maf": row_maf,
        "row_mean": row_mean,
        "row_inv_sd": row_inv_sd,
        "sample_indices": sample_indices,
        "n_iter": int(n_iter),
        "burnin": int(burnin),
        "thin": int(thin),
        "r2": float(r2),
        "seed": int(seed),
    }
    if method == "BayesA":
        ret = _jxrs.bayesa_packed(
            **common,
            min_abs_beta=1e-9,
        )
    elif method == "BayesB":
        ret = _jxrs.bayesb_packed(
            **common,
            counts=float(counts),
            prob_in=0.5,
        )
    elif method == "BayesCpi":
        ret = _jxrs.bayescpi_packed(
            **common,
            counts=float(counts),
            prob_in=0.5,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    beta = np.ascontiguousarray(np.asarray(ret[0], dtype=np.float64).reshape(-1), dtype=np.float64)
    alpha = np.ascontiguousarray(np.asarray(ret[1], dtype=np.float64).reshape(-1), dtype=np.float64)
    pve = float(ret[4])
    return beta, alpha, pve


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx bayesbench",
        description="Lightweight benchmark for JanusX packed-native Bayes kernels.",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx bayesbench --n-samples 1200 --n-snps 10000 --repeat 3",
                "jx bayesbench --row-block 0,64,256 --methods BayesA,BayesB",
            ]
        ),
    )
    parser.add_argument(
        "--methods",
        default="BayesA,BayesB,BayesCpi",
        help="Comma list from BayesA,BayesB,BayesCpi.",
    )
    parser.add_argument("--n-samples", type=int, default=1200, help="Total sample size.")
    parser.add_argument("--n-snps", type=int, default=10000, help="Total SNP rows.")
    parser.add_argument("--train-size", type=int, default=1000, help="Training sample size.")
    parser.add_argument("--n-iter", type=int, default=120, help="MCMC iterations.")
    parser.add_argument("--burnin", type=int, default=40, help="Burn-in iterations.")
    parser.add_argument("--thin", type=int, default=2, help="Thinning step.")
    parser.add_argument("--repeat", type=int, default=3, help="Repeat count per setting.")
    parser.add_argument("--seed", type=int, default=20260429, help="Random seed for simulation and kernel.")
    parser.add_argument("--r2", type=float, default=0.6, help="Fixed R2 for kernel benchmark.")
    parser.add_argument("--counts", type=float, default=5.0, help="counts hyperparameter for BayesB/BayesCpi.")
    parser.add_argument(
        "--row-block",
        default="0,256",
        help="Comma list for JX_BAYES_PACKED_ROW_BLOCK; <=0 means auto.",
    )
    parser.add_argument("--out", default=None, help="Optional output TSV path for benchmark summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    methods = _parse_methods(str(args.methods))
    row_blocks = _parse_row_blocks(str(args.row_block))

    n_samples = int(args.n_samples)
    n_snps = int(args.n_snps)
    train_size = int(args.train_size)
    n_iter = int(args.n_iter)
    burnin = int(args.burnin)
    thin = int(args.thin)
    repeat = int(args.repeat)
    seed = int(args.seed)
    r2 = float(args.r2)
    counts = float(args.counts)

    if n_samples <= 8:
        raise ValueError("--n-samples must be > 8.")
    if n_snps <= 16:
        raise ValueError("--n-snps must be > 16.")
    if train_size <= 4 or train_size > n_samples:
        raise ValueError("--train-size must be in (4, n-samples].")
    if n_iter <= burnin:
        raise ValueError("--n-iter must be > --burnin.")
    if thin <= 0:
        raise ValueError("--thin must be >= 1.")
    if repeat <= 0:
        raise ValueError("--repeat must be >= 1.")
    if not (0.0 < r2 < 1.0):
        raise ValueError("--r2 must be in (0, 1).")

    print(
        f"[bayesbench] generating synthetic data: n_samples={n_samples}, n_snps={n_snps}, "
        f"train_size={train_size}, seed={seed}"
    )
    rng = np.random.default_rng(seed)
    geno = rng.integers(0, 3, size=(n_snps, n_samples), dtype=np.int8)
    packed = _pack_bed_from_dosage(geno)
    row_flip = np.zeros((n_snps,), dtype=np.bool_)
    row_maf = np.ascontiguousarray((geno.mean(axis=1) / 2.0).astype(np.float32), dtype=np.float32)
    row_mean = np.ascontiguousarray(geno.mean(axis=1).astype(np.float32), dtype=np.float32)
    row_std = np.asarray(geno.std(axis=1), dtype=np.float64)
    row_inv_sd = np.ascontiguousarray((1.0 / (row_std + 1e-6)).astype(np.float32), dtype=np.float32)
    sample_idx = np.ascontiguousarray(np.arange(train_size, dtype=np.int64), dtype=np.int64)
    y = np.ascontiguousarray(rng.normal(size=(train_size,)).astype(np.float64), dtype=np.float64)

    results: list[dict[str, Any]] = []
    baseline: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    old_row_block = os.environ.get("JX_BAYES_PACKED_ROW_BLOCK", None)

    try:
        for rb in row_blocks:
            if rb is None:
                os.environ.pop("JX_BAYES_PACKED_ROW_BLOCK", None)
                rb_label = "auto"
            else:
                os.environ["JX_BAYES_PACKED_ROW_BLOCK"] = str(int(rb))
                rb_label = str(int(rb))
            print(f"[bayesbench] row_block={rb_label}")

            for method in methods:
                times: list[float] = []
                rep_beta = None
                rep_alpha = None
                rep_pve = None
                for rep in range(repeat):
                    t0 = time.perf_counter()
                    beta, alpha, pve = _run_one_kernel(
                        method=method,
                        y=y,
                        packed=packed,
                        n_samples=n_samples,
                        row_flip=row_flip,
                        row_maf=row_maf,
                        row_mean=row_mean,
                        row_inv_sd=row_inv_sd,
                        sample_indices=sample_idx,
                        n_iter=n_iter,
                        burnin=burnin,
                        thin=thin,
                        r2=r2,
                        counts=counts,
                        seed=seed,
                    )
                    times.append(float(time.perf_counter() - t0))
                    if rep == 0:
                        rep_beta, rep_alpha, rep_pve = beta, alpha, pve
                assert rep_beta is not None and rep_alpha is not None and rep_pve is not None

                if method not in baseline:
                    baseline[method] = (rep_beta, rep_alpha, float(rep_pve))
                    diff_beta = 0.0
                    diff_alpha = 0.0
                    diff_pve = 0.0
                else:
                    b0, a0, p0 = baseline[method]
                    diff_beta = float(np.max(np.abs(rep_beta - b0)))
                    diff_alpha = float(np.max(np.abs(rep_alpha - a0)))
                    diff_pve = float(abs(float(rep_pve) - float(p0)))

                arr = np.asarray(times, dtype=np.float64)
                results.append(
                    {
                        "method": method,
                        "row_block": rb_label,
                        "repeat": int(repeat),
                        "time_mean_sec": float(np.mean(arr)),
                        "time_std_sec": float(np.std(arr, ddof=0)),
                        "time_min_sec": float(np.min(arr)),
                        "beta_max_abs_vs_baseline": float(diff_beta),
                        "alpha_max_abs_vs_baseline": float(diff_alpha),
                        "pve_abs_vs_baseline": float(diff_pve),
                    }
                )
                print(
                    f"[bayesbench] {method:8s} row_block={rb_label:>4s} "
                    f"mean={np.mean(arr):.4f}s min={np.min(arr):.4f}s "
                    f"std={np.std(arr):.4f}s"
                )
    finally:
        if old_row_block is None:
            os.environ.pop("JX_BAYES_PACKED_ROW_BLOCK", None)
        else:
            os.environ["JX_BAYES_PACKED_ROW_BLOCK"] = old_row_block

    df = pd.DataFrame(results)
    print("\n[bayesbench] summary")
    print(df.to_string(index=False))

    out = args.out
    if out is not None and str(out).strip() != "":
        out_path = Path(str(out)).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep="\t", index=False)
        print(f"[bayesbench] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
