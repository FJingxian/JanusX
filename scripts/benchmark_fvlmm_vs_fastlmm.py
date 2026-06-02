#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _basename_prefix(path_text: str) -> str:
    return Path(path_text).name


def _peak_rss_tree_bytes(proc: psutil.Process) -> int:
    total = 0
    try:
        total += int(proc.memory_info().rss)
    except Exception:
        return 0
    try:
        for child in proc.children(recursive=True):
            try:
                total += int(child.memory_info().rss)
            except Exception:
                continue
    except Exception:
        pass
    return total


def _run_model(
    *,
    model_key: str,
    python_exe: str,
    bfile: str,
    pheno: str,
    out_dir: Path,
    trait: str,
    threads: int,
    maf: float,
    geno: float,
    het: float,
    qcov: str,
    grm: str | None,
    use_fast: bool,
    genetic_model: str,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{model_key}.stdout.log"
    bfile_abs = str(Path(bfile).resolve())
    pheno_abs = str(Path(pheno).resolve())
    grm_abs = str(Path(grm).resolve()) if grm is not None and str(grm).strip() != "" else None
    cmd = [
        python_exe,
        "-m",
        "janusx.script.JanusX",
        "gwas",
        "-bfile",
        bfile_abs,
        "-p",
        pheno_abs,
        f"-{model_key}",
        "-n",
        str(trait),
        "-o",
        str(out_dir),
        "-t",
        str(int(threads)),
        "-maf",
        str(float(maf)),
        "-geno",
        str(float(geno)),
        "-het",
        str(float(het)),
        "-q",
        str(qcov),
        "-model",
        str(genetic_model),
    ]
    if grm_abs is not None:
        cmd.extend(["-k", grm_abs])
    if use_fast:
        cmd.append("-fast")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_root() / "python") + os.pathsep + env.get("PYTHONPATH", "")
    cache_root = out_dir / ".runtime-cache"
    mpl_cache = cache_root / "mpl"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_cache)
    env["XDG_CACHE_HOME"] = str(xdg_cache)

    t0 = time.perf_counter()
    peak_rss = 0
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_repo_root()),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        ps_proc = psutil.Process(proc.pid)
        while True:
            rc = proc.poll()
            peak_rss = max(peak_rss, _peak_rss_tree_bytes(ps_proc))
            if rc is not None:
                break
            time.sleep(0.1)
        wall_s = time.perf_counter() - t0

    if proc.returncode != 0:
        raise RuntimeError(
            f"{model_key} benchmark run failed with exit={proc.returncode}. See {log_path}"
        )

    matches = sorted(out_dir.glob(f"*.{model_key}.tsv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {model_key} TSV in {out_dir}, found {len(matches)}."
        )
    tsv_path = matches[0]
    return {
        "model": model_key,
        "cmd": cmd,
        "wall_s": float(wall_s),
        "peak_rss_gb": float(peak_rss / (1024 ** 3)),
        "stdout_log": str(log_path),
        "result_tsv": str(tsv_path),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _load_result(path_text: str) -> pd.DataFrame:
    df = pd.read_csv(path_text, sep="\t")
    if "snp" not in df.columns:
        df["snp"] = df["chrom"].astype(str) + "_" + df["pos"].astype(str)
    return df


def _compare_outputs(fastlmm_tsv: str, fvlmm_tsv: str) -> dict[str, object]:
    df_fast = _load_result(fastlmm_tsv)
    df_fv = _load_result(fvlmm_tsv)
    merged = df_fast.merge(
        df_fv,
        on=["chrom", "pos", "snp", "allele0", "allele1"],
        suffixes=("_fastlmm", "_fvlmm"),
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("No overlapping SNP rows between fastlmm and fvlmm outputs.")

    beta_fast = np.asarray(merged["beta_fastlmm"], dtype=np.float64)
    beta_fv = np.asarray(merged["beta_fvlmm"], dtype=np.float64)
    se_fast = np.asarray(merged["se_fastlmm"], dtype=np.float64)
    se_fv = np.asarray(merged["se_fvlmm"], dtype=np.float64)
    p_fast = np.asarray(merged["pwald_fastlmm"], dtype=np.float64)
    p_fv = np.asarray(merged["pwald_fvlmm"], dtype=np.float64)
    logp_fast = -np.log10(np.clip(p_fast, np.finfo(np.float64).tiny, 1.0))
    logp_fv = -np.log10(np.clip(p_fv, np.finfo(np.float64).tiny, 1.0))

    return {
        "n_overlap": int(len(merged)),
        "beta_corr": _safe_corr(beta_fast, beta_fv),
        "se_corr": _safe_corr(se_fast, se_fv),
        "logp_corr": _safe_corr(logp_fast, logp_fv),
        "max_abs_beta_diff": float(np.max(np.abs(beta_fast - beta_fv))),
        "median_abs_beta_diff": float(np.median(np.abs(beta_fast - beta_fv))),
        "max_abs_logp_diff": float(np.max(np.abs(logp_fast - logp_fv))),
        "median_abs_logp_diff": float(np.median(np.abs(logp_fast - logp_fv))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark GWAS fastlmm vs fvlmm end-to-end speed, memory, and result consistency."
    )
    ap.add_argument("-bfile", "--bfile", required=True, type=str)
    ap.add_argument("-p", "--pheno", required=True, type=str)
    ap.add_argument("-n", "--trait", required=True, type=str, help="Trait column index or name passed to jx gwas -n.")
    ap.add_argument("-o", "--out", required=True, type=str)
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("-k", "--grm", type=str, default="")
    ap.add_argument("-q", "--qcov", type=str, default="0")
    ap.add_argument("-maf", "--maf", type=float, default=0.02)
    ap.add_argument("-geno", "--geno", type=float, default=0.05)
    ap.add_argument("-het", "--het", type=float, default=0.0)
    ap.add_argument("--model", type=str, default="add", choices=["add", "dom", "rec", "het"])
    ap.add_argument("--fast", action="store_true", default=False, help="Also pass -fast to jx gwas.")
    ap.add_argument("--python", type=str, default=sys.executable)
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict[str, object]] = {}
    for model_key in ("fastlmm", "fvlmm"):
        runs[model_key] = _run_model(
            model_key=model_key,
            python_exe=str(args.python),
            bfile=str(args.bfile),
            pheno=str(args.pheno),
            out_dir=out_root / model_key,
            trait=str(args.trait),
            threads=int(args.threads),
            maf=float(args.maf),
            geno=float(args.geno),
            het=float(args.het),
            qcov=str(args.qcov),
            grm=(str(args.grm) if str(args.grm).strip() != "" else None),
            use_fast=bool(args.fast),
            genetic_model=str(args.model),
        )

    comparison = _compare_outputs(
        str(runs["fastlmm"]["result_tsv"]),
        str(runs["fvlmm"]["result_tsv"]),
    )
    speedup = float(runs["fastlmm"]["wall_s"]) / max(float(runs["fvlmm"]["wall_s"]), 1e-12)
    mem_ratio = float(runs["fastlmm"]["peak_rss_gb"]) / max(float(runs["fvlmm"]["peak_rss_gb"]), 1e-12)

    payload = {
        "config": {
            "bfile": str(args.bfile),
            "pheno": str(args.pheno),
            "trait": str(args.trait),
            "threads": int(args.threads),
            "grm": (str(args.grm) if str(args.grm).strip() != "" else None),
            "qcov": str(args.qcov),
            "maf": float(args.maf),
            "geno": float(args.geno),
            "het": float(args.het),
            "genetic_model": str(args.model),
            "fast_mode": bool(args.fast),
            "prefix": _basename_prefix(str(args.bfile)),
        },
        "fastlmm": runs["fastlmm"],
        "fvlmm": runs["fvlmm"],
        "comparison": comparison,
        "relative": {
            "fvlmm_speedup_vs_fastlmm": speedup,
            "fvlmm_peak_rss_ratio_vs_fastlmm": 1.0 / max(mem_ratio, 1e-12),
        },
    }

    summary_json = out_root / "summary.json"
    summary_tsv = out_root / "summary.tsv"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_df = pd.DataFrame(
        [
            {
                "model": "fastlmm",
                "wall_s": float(runs["fastlmm"]["wall_s"]),
                "peak_rss_gb": float(runs["fastlmm"]["peak_rss_gb"]),
                "result_tsv": str(runs["fastlmm"]["result_tsv"]),
            },
            {
                "model": "fvlmm",
                "wall_s": float(runs["fvlmm"]["wall_s"]),
                "peak_rss_gb": float(runs["fvlmm"]["peak_rss_gb"]),
                "result_tsv": str(runs["fvlmm"]["result_tsv"]),
            },
        ]
    )
    summary_df.to_csv(summary_tsv, sep="\t", index=False)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
