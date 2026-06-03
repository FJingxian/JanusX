#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def build_packed_bits(
    n_rows: int,
    n_samples: int,
    density: float,
    seed: int,
    chunk_rows: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    row_words = (n_samples + 63) // 64
    bits = np.empty((n_rows, row_words), dtype=np.uint64)
    full_words = n_samples // 64
    rem = n_samples % 64
    tail_mask = (
        np.uint64((1 << rem) - 1)
        if rem
        else np.uint64(np.iinfo(np.uint64).max)
    )

    if abs(float(density) - 0.5) < 1e-12:
        bits[:] = rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=(n_rows, row_words),
            dtype=np.uint64,
        )
    else:
        chunk_rows = max(1, int(chunk_rows))
        for start in range(0, n_rows, chunk_rows):
            end = min(n_rows, start + chunk_rows)
            chunk = (
                rng.random((end - start, row_words, 64), dtype=np.float32)
                < float(density)
            )
            packed = np.packbits(chunk, axis=2, bitorder="little").reshape(
                end - start, row_words, 8
            )
            bits[start:end] = packed.view(np.uint64).reshape(end - start, row_words)

    if rem:
        bits[:, full_words] &= tail_mask
    return np.ascontiguousarray(bits)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark Garfield singleton centered-gain backends: legacy vs CPU vs Metal GPU."
    )
    ap.add_argument("--rows", type=int, default=65536)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--chunk-rows", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260603)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    import janusx.janusx as jx

    t0 = time.perf_counter()
    y = np.random.default_rng(args.seed + 1).standard_normal(args.samples).astype(np.float64)
    bits = build_packed_bits(
        args.rows,
        args.samples,
        args.density,
        args.seed,
        args.chunk_rows,
    )
    build_s = time.perf_counter() - t0

    report = jx.garfield_compare_score_cont_centered_gain_singleton_backends(
        y,
        bits,
        int(args.samples),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
    )
    report = dict(report)
    report["rows"] = int(args.rows)
    report["samples"] = int(args.samples)
    report["density"] = float(args.density)
    report["chunk_rows"] = int(args.chunk_rows)
    report["build_input_s"] = float(build_s)
    report["input_bits_bytes"] = int(bits.nbytes)
    report["input_y_bytes"] = int(y.nbytes)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if str(args.out).strip():
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
