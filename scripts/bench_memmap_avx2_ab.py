#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B benchmark for JanusX BED memmap filter+write path on x86 AVX2.

This script compares:
  - JANUSX_BED_AVX2_MODE=baseline
  - JANUSX_BED_AVX2_MODE=aggressive

Each run executes janusx.janusx.bed_mmap_filter_to_plink_rust in a fresh
subprocess, so AVX2 mode is reliably applied per run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


def _normalize_plink_prefix(path_or_prefix: str) -> str:
    s = str(path_or_prefix).strip()
    low = s.lower()
    if low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam"):
        return s[:-4]
    return s


def _extract_last_json_line(text: str) -> dict[str, Any]:
    lines = str(text).splitlines()
    for line in reversed(lines):
        s = line.strip()
        if not s:
            continue
        if not (s.startswith("{") and s.endswith("}")):
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _fmt_secs(x: Optional[float]) -> str:
    if x is None or (not math.isfinite(float(x))):
        return "nan"
    return f"{float(x):.3f}"


def _fmt_gb(x: Optional[float]) -> str:
    if x is None or (not math.isfinite(float(x))):
        return "nan"
    return f"{float(x):.3f}"


def _memmap_runner_code() -> str:
    return r"""
import json
import os
import threading
import time
import psutil
import janusx.janusx as jxrs

cfg = json.loads(os.environ["JX_BENCH_CFG"])
if not hasattr(jxrs, "bed_mmap_filter_to_plink_rust"):
    raise RuntimeError("Rust symbol bed_mmap_filter_to_plink_rust is unavailable")

_self_proc = psutil.Process(os.getpid())
_mem = {"rss": 0, "uss": None}
_stop = {"v": False}

def _sample_mem(sample_uss=False):
    try:
        rss_now = int(_self_proc.memory_info().rss)
        if rss_now > int(_mem["rss"]):
            _mem["rss"] = rss_now
    except Exception:
        pass
    if sample_uss:
        try:
            fi = _self_proc.memory_full_info()
            uv = getattr(fi, "uss", None)
            if uv is not None:
                uv_i = int(uv)
                cur = _mem["uss"]
                if cur is None or uv_i > int(cur):
                    _mem["uss"] = uv_i
        except Exception:
            pass

def _mem_worker():
    t_next_uss = 0.0
    while not bool(_stop["v"]):
        now = time.perf_counter()
        do_uss = now >= t_next_uss
        if do_uss:
            t_next_uss = now + 0.5
        _sample_mem(do_uss)
        time.sleep(0.05)

_th = threading.Thread(target=_mem_worker, daemon=True)
_th.start()

t0 = time.perf_counter()
kept = 0
rows_total = 0
n_samples = 0
blocks = 0
try:
    kept, rows_total, n_samples, blocks = jxrs.bed_mmap_filter_to_plink_rust(
        str(cfg["prefix"]),
        str(cfg["out_prefix"]),
        maf_threshold=float(cfg["maf"]),
        max_missing_rate=float(cfg["geno"]),
        het_threshold=0.0,
        block_rows=int(cfg["block_rows"]),
        parallel=bool(cfg["parallel"]),
    )
finally:
    _stop["v"] = True
    _th.join(timeout=0.2)
    _sample_mem(True)

elapsed = max(0.0, time.perf_counter() - t0)
print(json.dumps({
    "elapsed_sec": float(elapsed),
    "n_snps": int(kept),
    "n_snps_scanned": int(rows_total),
    "n_samples": int(n_samples),
    "n_blocks": int(blocks),
    "out_prefix": str(cfg["out_prefix"]),
    "avx2_mode": str(os.environ.get("JANUSX_BED_AVX2_MODE", "")),
    "peak_rss_self_bytes": int(_mem["rss"]),
    "peak_uss_self_bytes": (None if _mem["uss"] is None else int(_mem["uss"])),
}))
"""


@dataclass
class RunRow:
    mode: str
    run_index: int
    is_warmup: bool
    status: str
    returncode: int
    wall_sec: float
    peak_rss_gb: Optional[float]
    peak_uss_gb: Optional[float]
    n_snps_kept: Optional[int]
    n_snps_scanned: Optional[int]
    n_samples: Optional[int]
    n_blocks: Optional[int]
    out_prefix: str
    stderr_tail: str


def _cleanup_plink_prefix(prefix: str) -> None:
    for ext in (".bed", ".bim", ".fam"):
        p = Path(f"{prefix}{ext}")
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as fw:
        w = csv.DictWriter(fw, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _run_one(
    *,
    mode: str,
    cfg: dict[str, Any],
    cwd: str,
) -> tuple[RunRow, dict[str, Any]]:
    env = dict(os.environ)
    env["JX_BENCH_CFG"] = json.dumps(cfg, ensure_ascii=False)
    env["JANUSX_BED_AVX2_MODE"] = str(mode)
    cmd = [sys.executable, "-c", _memmap_runner_code()]
    t0 = time.perf_counter()
    cp = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed = max(0.0, time.perf_counter() - t0)
    obj = _extract_last_json_line(cp.stdout)

    wall = _safe_float(obj.get("elapsed_sec")) if obj else None
    if wall is None:
        wall = float(elapsed)

    rss_b = _safe_int(obj.get("peak_rss_self_bytes")) if obj else None
    uss_b = _safe_int(obj.get("peak_uss_self_bytes")) if obj else None
    peak_rss_gb = (float(rss_b) / (1024.0**3)) if (rss_b is not None and rss_b > 0) else None
    peak_uss_gb = (float(uss_b) / (1024.0**3)) if (uss_b is not None and uss_b > 0) else None

    row = RunRow(
        mode=str(mode),
        run_index=-1,
        is_warmup=False,
        status=("ok" if cp.returncode == 0 else "failed"),
        returncode=int(cp.returncode),
        wall_sec=float(wall),
        peak_rss_gb=peak_rss_gb,
        peak_uss_gb=peak_uss_gb,
        n_snps_kept=_safe_int(obj.get("n_snps")) if obj else None,
        n_snps_scanned=_safe_int(obj.get("n_snps_scanned")) if obj else None,
        n_samples=_safe_int(obj.get("n_samples")) if obj else None,
        n_blocks=_safe_int(obj.get("n_blocks")) if obj else None,
        out_prefix=str(obj.get("out_prefix", cfg.get("out_prefix", ""))) if obj else str(cfg.get("out_prefix", "")),
        stderr_tail=str(cp.stderr)[-5000:],
    )
    return row, obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_memmap_avx2_ab.py",
        description=(
            "A/B compare janusx.janusx.bed_mmap_filter_to_plink_rust under "
            "JANUSX_BED_AVX2_MODE=baseline/aggressive."
        ),
    )
    p.add_argument("--prefix", required=True, help="PLINK prefix (or .bed/.bim/.fam path).")
    p.add_argument("--label", default=None, help="Dataset label in output files.")
    p.add_argument("--outdir", default="bench_avx2_ab_results", help="Output directory.")
    p.add_argument("--repeat", type=int, default=5, help="Measured repeats per mode.")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per mode.")
    p.add_argument("--maf", type=float, default=0.02, help="MAF threshold.")
    p.add_argument("--geno", type=float, default=0.10, help="Max missing-rate threshold.")
    p.add_argument("--block-rows", type=int, default=50_000, help="Block rows for mmap scan.")
    p.add_argument("--parallel", dest="parallel", action="store_true", default=True, help="Enable parallel scan.")
    p.add_argument("--no-parallel", dest="parallel", action="store_false", help="Disable parallel scan.")
    p.add_argument(
        "--modes",
        default="baseline,aggressive",
        help="Comma-separated AVX2 modes (default: baseline,aggressive).",
    )
    p.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep generated PLINK outputs for every run.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    prefix = _normalize_plink_prefix(args.prefix)
    label = str(args.label).strip() if args.label else Path(prefix).name
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for ext in (".bed", ".bim", ".fam"):
        p = Path(f"{prefix}{ext}")
        if not p.exists():
            raise FileNotFoundError(f"missing input file: {p}")

    modes = [x.strip() for x in str(args.modes).split(",") if x.strip()]
    if len(modes) == 0:
        raise ValueError("No AVX2 modes configured. Use --modes baseline,aggressive.")

    print(f"[info] dataset: {label}")
    print(f"[info] prefix : {prefix}")
    print(f"[info] outdir : {outdir}")
    print(f"[info] modes  : {', '.join(modes)}")
    print(
        "[info] params : "
        f"maf={float(args.maf)} geno={float(args.geno)} block_rows={int(max(1, args.block_rows))} "
        f"parallel={bool(args.parallel)} repeat={int(max(1, args.repeat))} warmup={int(max(0, args.warmup))}"
    )

    rows: list[RunRow] = []
    work_root = outdir / "runs"
    total_runs = int(max(0, args.warmup) + max(1, args.repeat))

    for mode in modes:
        print(f"[mode] {mode}")
        for run_idx in range(total_runs):
            is_warmup = run_idx < int(max(0, args.warmup))
            run_dir = work_root / mode / f"run_{run_idx:02d}"
            out_prefix = run_dir / "out" / f"{label}_{mode}_{run_idx:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cfg = {
                "prefix": str(prefix),
                "out_prefix": str(out_prefix),
                "maf": float(args.maf),
                "geno": float(args.geno),
                "block_rows": int(max(1, args.block_rows)),
                "parallel": bool(args.parallel),
            }
            row, obj = _run_one(mode=mode, cfg=cfg, cwd=str(Path.cwd()))
            row.run_index = int(run_idx)
            row.is_warmup = bool(is_warmup)
            rows.append(row)

            tag = "warmup" if is_warmup else f"rep={run_idx - int(max(0, args.warmup)):02d}"
            print(
                f"  {tag} rc={row.returncode} wall={_fmt_secs(row.wall_sec)}s "
                f"rss={_fmt_gb(row.peak_rss_gb)}G uss={_fmt_gb(row.peak_uss_gb)}G "
                f"kept={row.n_snps_kept} scanned={row.n_snps_scanned} blocks={row.n_blocks}"
            )
            if row.returncode != 0 and row.stderr_tail:
                print(f"    stderr_tail: {row.stderr_tail[-300:].replace(chr(10), ' | ')}")

            if (not args.keep_output) and row.out_prefix:
                _cleanup_plink_prefix(row.out_prefix)

            raw_path = run_dir / "result.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_path.open("w", encoding="utf-8") as fw:
                json.dump(obj, fw, ensure_ascii=False, indent=2)

    result_rows = [asdict(r) for r in rows]
    json_path = outdir / "avx2_ab_results.json"
    csv_path = outdir / "avx2_ab_results.csv"
    with json_path.open("w", encoding="utf-8") as fw:
        json.dump(result_rows, fw, ensure_ascii=False, indent=2)
    _write_csv(csv_path, result_rows)

    measured = [
        r
        for r in rows
        if (not r.is_warmup) and r.status == "ok" and math.isfinite(float(r.wall_sec))
    ]
    by_mode: dict[str, list[RunRow]] = {}
    for r in measured:
        by_mode.setdefault(r.mode, []).append(r)

    summary: dict[str, Any] = {
        "dataset": label,
        "prefix": prefix,
        "repeat": int(max(1, args.repeat)),
        "warmup": int(max(0, args.warmup)),
        "maf": float(args.maf),
        "geno": float(args.geno),
        "block_rows": int(max(1, args.block_rows)),
        "parallel": bool(args.parallel),
        "modes": modes,
        "modes_stats": {},
        "speedup_baseline_over_aggressive": math.nan,
    }

    for mode in modes:
        items = by_mode.get(mode, [])
        walls = [float(x.wall_sec) for x in items]
        rss = [float(x.peak_rss_gb) for x in items if x.peak_rss_gb is not None]
        uss = [float(x.peak_uss_gb) for x in items if x.peak_uss_gb is not None]
        summary["modes_stats"][mode] = {
            "n_ok_runs": len(items),
            "wall_sec_mean": (float(sum(walls) / len(walls)) if len(walls) > 0 else math.nan),
            "wall_sec_min": (float(min(walls)) if len(walls) > 0 else math.nan),
            "wall_sec_max": (float(max(walls)) if len(walls) > 0 else math.nan),
            "peak_rss_gb_mean": (float(sum(rss) / len(rss)) if len(rss) > 0 else math.nan),
            "peak_uss_gb_mean": (float(sum(uss) / len(uss)) if len(uss) > 0 else math.nan),
        }

    base_mean = _safe_float(summary["modes_stats"].get("baseline", {}).get("wall_sec_mean"))
    aggr_mean = _safe_float(summary["modes_stats"].get("aggressive", {}).get("wall_sec_mean"))
    if (
        base_mean is not None
        and aggr_mean is not None
        and math.isfinite(base_mean)
        and math.isfinite(aggr_mean)
        and aggr_mean > 0.0
    ):
        summary["speedup_baseline_over_aggressive"] = float(base_mean / aggr_mean)

    summary_path = outdir / "avx2_ab_summary.json"
    with summary_path.open("w", encoding="utf-8") as fw:
        json.dump(summary, fw, ensure_ascii=False, indent=2)

    print(f"[done] results: {json_path}")
    print(f"[done] csv    : {csv_path}")
    print(f"[done] summary: {summary_path}")
    sp = _safe_float(summary.get("speedup_baseline_over_aggressive"))
    if sp is not None and math.isfinite(sp):
        print(
            "[summary] speedup (baseline/aggressive) = "
            f"{sp:.4f}x ({'aggressive faster' if sp > 1.0 else 'aggressive slower'})"
        )


if __name__ == "__main__":
    main()
