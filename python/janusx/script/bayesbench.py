#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bayesian GS benchmark / convergence / reference-comparison workflow.

Modes
-----
- kernel:
    Lightweight synthetic packed-kernel benchmark.
- convergence:
    Multi-chain JanusX BayesA/B/Cpi summary stability experiment.
- compare:
    JanusX vs BGLR long-chain posterior / prediction concordance experiment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
from janusx.gs.workflow import (  # noqa: E402
    _decode_packed_subset_to_dense_standardized,
    _ensure_packed_standard_stats_cached,
    _predict_bayes_packed_from_effects,
)
from janusx.script._common.genoio import determine_genotype_source  # noqa: E402
from janusx.script._common.helptext import (  # noqa: E402
    CliArgumentParser,
    cli_help_formatter,
    minimal_help_epilog,
)
from janusx.script._common.packedctx import prepare_packed_ctx_from_plink  # noqa: E402


_METHOD_ALIASES = {
    "bayesa": "BayesA",
    "bayesb": "BayesB",
    "bayescpi": "BayesCpi",
}

_MODE_TOKENS = {"kernel", "convergence", "compare"}
_REF_ENGINES = {"all", "bglr", "hibayes"}


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


def _parse_reference_engines(raw: str) -> list[str]:
    vals: list[str] = []
    seen: set[str] = set()
    for part in str(raw).split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key == "all":
            for item in ("bglr", "hibayes"):
                if item not in seen:
                    seen.add(item)
                    vals.append(item)
            continue
        if key not in _REF_ENGINES:
            raise ValueError(
                f"Unsupported reference token: {part!r}. Use bglr, hibayes, or all."
            )
        if key in seen:
            continue
        seen.add(key)
        vals.append(key)
    if len(vals) == 0:
        raise ValueError("No valid engines parsed from --reference.")
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


def _parse_single_trait_arg(raw: str | None) -> int | None:
    if raw is None:
        return None
    tok = str(raw).strip()
    if tok == "":
        return None
    if "," in tok or ":" in tok:
        raise ValueError("`-n/--n` for bayesbench currently accepts one zero-based trait index.")
    return int(tok)


def _parse_seed_list(raw: str | None, chains: int, base_seed: int) -> list[int]:
    if raw is None or str(raw).strip() == "":
        return [int(base_seed) + i for i in range(int(chains))]
    vals: list[int] = []
    for part in str(raw).split(","):
        tok = part.strip()
        if tok == "":
            continue
        vals.append(int(tok))
    if len(vals) == 0:
        raise ValueError("`--chain-seeds` is empty after parsing.")
    if len(vals) != int(chains):
        raise ValueError(
            f"`--chain-seeds` count mismatch: got {len(vals)}, expected chains={int(chains)}."
        )
    return vals


def _safe_float(v: Any) -> float:
    try:
        out = float(v)
    except Exception:
        return float("nan")
    return out


def _supports(cmd: str, flag: str) -> bool:
    try:
        proc = subprocess.run(
            [cmd, flag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return False
    if int(proc.returncode) not in (0, 1):
        return False
    txt = f"{proc.stdout}\n{proc.stderr}".lower()
    if "illegal option" in txt or "unknown option" in txt or "invalid option" in txt:
        return False
    return True


def _detect_time_tool() -> list[str]:
    sys_time = Path("/usr/bin/time")
    if sys_time.exists():
        if _supports(str(sys_time), "-v"):
            return [str(sys_time), "-v"]
        if _supports(str(sys_time), "-l"):
            return [str(sys_time), "-l"]
    return []


def _parse_elapsed_text(text: str) -> float:
    s = str(text).strip()
    if s == "":
        return math.nan
    parts = s.split(":")
    try:
        if len(parts) == 3:
            hh = float(parts[0])
            mm = float(parts[1])
            ss = float(parts[2])
            return float(hh * 3600.0 + mm * 60.0 + ss)
        if len(parts) == 2:
            mm = float(parts[0])
            ss = float(parts[1])
            return float(mm * 60.0 + ss)
        return float(s)
    except Exception:
        return math.nan


def _parse_time_file(path: Path) -> tuple[float, float]:
    if not path.exists():
        return math.nan, math.nan
    elapsed = math.nan
    rss = math.nan
    er = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(.+)")
    rr = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
    bsd_elapsed = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+real\b")
    bsd_rss = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$")
    for line in path.read_text(errors="ignore").splitlines():
        m = er.search(line)
        if m:
            elapsed = _parse_elapsed_text(m.group(1))
            continue
        m = rr.search(line)
        if m:
            rss = _safe_float(m.group(1))
            continue
        m = bsd_elapsed.search(line)
        if m:
            elapsed = _safe_float(m.group(1))
            continue
        m = bsd_rss.search(line)
        if m:
            rss = _safe_float(m.group(1)) / 1024.0
    return elapsed, rss


def _run_timed(
    cmd: list[str],
    *,
    log_file: Path,
    time_file: Path,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> tuple[int, float, float]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    time_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    peak_rss_kb = float("nan")

    def _pid_rss_kb(pid: int) -> float:
        try:
            proc_rss = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(int(pid))],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except Exception:
            return float("nan")
        if proc_rss.returncode != 0:
            return float("nan")
        try:
            return float(str(proc_rss.stdout).strip().splitlines()[-1].strip())
        except Exception:
            return float("nan")

    with log_file.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            list(cmd),
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=(str(cwd) if cwd is not None else None),
            env=env,
            text=True,
        )
        while True:
            rss_now = _pid_rss_kb(int(proc.pid))
            if np.isfinite(rss_now):
                peak_rss_kb = (
                    float(rss_now)
                    if not np.isfinite(peak_rss_kb)
                    else max(float(peak_rss_kb), float(rss_now))
                )
            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(0.2)
    wall = max(time.time() - t0, 0.0)
    time_file.write_text(
        (
            f"Elapsed (wall clock) time (h:mm:ss or m:ss): {wall:.6f}\n"
            + (
                "Maximum resident set size (kbytes): NA\n"
                if not np.isfinite(peak_rss_kb)
                else f"Maximum resident set size (kbytes): {float(peak_rss_kb):.0f}\n"
            )
        ),
        encoding="utf-8",
    )
    return int(proc.returncode), float(wall), float(peak_rss_kb)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if int(np.sum(mask)) < 3:
        return float("nan")
    xv = xv[mask]
    yv = yv[mask]
    xs = xv - float(np.mean(xv))
    ys = yv - float(np.mean(yv))
    den = float(np.sqrt(np.sum(xs * xs) * np.sum(ys * ys)))
    if not np.isfinite(den) or den <= 0.0:
        return float("nan")
    return float(np.sum(xs * ys) / den)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xv = pd.Series(np.asarray(x, dtype=np.float64).reshape(-1))
    yv = pd.Series(np.asarray(y, dtype=np.float64).reshape(-1))
    mask = np.isfinite(xv.to_numpy(dtype=float)) & np.isfinite(yv.to_numpy(dtype=float))
    if int(np.sum(mask)) < 3:
        return float("nan")
    xr = xv[mask].rank(method="average").to_numpy(dtype=np.float64)
    yr = yv[mask].rank(method="average").to_numpy(dtype=np.float64)
    return _pearson(xr, yr)


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if int(np.sum(mask)) == 0:
        return float("nan")
    diff = xv[mask] - yv[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def _mae(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if int(np.sum(mask)) == 0:
        return float("nan")
    diff = np.abs(xv[mask] - yv[mask])
    return float(np.mean(diff))


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if int(np.sum(mask)) < 3:
        return float("nan")
    xv = xv[mask]
    yv = yv[mask]
    xs = xv - float(np.mean(xv))
    den = float(np.sum(xs * xs))
    if not np.isfinite(den) or den <= 0.0:
        return float("nan")
    return float(np.sum(xs * (yv - float(np.mean(yv)))) / den)


def _score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "pearson": _pearson(y_true, y_pred),
        "spearman": _spearman(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
    }


def _read_table_guess(path: Path) -> pd.DataFrame:
    tries = [
        dict(sep="\t", engine="python"),
        dict(sep=",", engine="python"),
        dict(sep=r"\s+", engine="python"),
        dict(sep=None, engine="python"),
    ]
    last_err: Exception | None = None
    for kw in tries:
        try:
            df = pd.read_csv(path, **kw)
            if df.shape[1] >= 1:
                return df
        except Exception as ex:
            last_err = ex
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Unable to parse table: {path}")


def _read_pheno(path: Path, trait_idx: int | None, trait_name: str | None) -> tuple[pd.DataFrame, str]:
    df = _read_table_guess(path)
    if df.shape[1] < 2:
        raise ValueError("Phenotype file must contain sample ID and at least one trait column.")
    if trait_idx is not None and trait_name is not None:
        raise ValueError("Use either `-n/--n` or `--trait`, not both.")
    id_col = df.columns[0]
    if trait_name is not None:
        if trait_name not in df.columns:
            raise ValueError(f"Trait {trait_name!r} was not found in phenotype columns.")
        target_col = trait_name
    elif trait_idx is None:
        target_col = df.columns[1]
    else:
        idx = int(trait_idx) + 1
        if idx < 1 or idx >= int(df.shape[1]):
            raise ValueError(
                f"`-n/--n` out of range: {trait_idx}. Valid range is 0..{int(df.shape[1]) - 2}."
            )
        target_col = df.columns[idx]
    out = pd.DataFrame(
        {
            "Taxa": df[id_col].astype(str).str.strip(),
            "PHENO": pd.to_numeric(df[target_col], errors="coerce"),
        }
    )
    out = out[(out["Taxa"] != "") & out["PHENO"].notna()].drop_duplicates("Taxa", keep="first")
    if out.empty:
        raise ValueError("No usable phenotype rows remain after filtering.")
    return out, str(target_col)


def _resolve_plink_prefix_from_args(args: argparse.Namespace) -> str:
    def _has_plink_prefix(prefix: str) -> bool:
        pfx = str(prefix).strip()
        if pfx == "":
            return False
        return all(Path(f"{pfx}.{ext}").is_file() for ext in ("bed", "bim", "fam"))

    gfile, _ = determine_genotype_source(
        vcf=getattr(args, "vcf", None),
        hmp=getattr(args, "hmp", None),
        file=getattr(args, "file", None),
        bfile=getattr(args, "bfile", None),
        prefix=None,
        snps_only=bool(getattr(args, "snps_only", False)),
        delimiter=None,
        apply_cache=bool(getattr(args, "cache_input", True)),
    )
    cand = str(gfile).strip()
    if cand == "":
        raise ValueError("Resolved genotype input is empty.")
    if _has_plink_prefix(cand):
        return cand
    base = cand
    for ext in (".bed", ".bim", ".fam", ".vcf", ".vcf.gz", ".hmp", ".hmp.gz", ".txt", ".tsv", ".csv"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    if _has_plink_prefix(base):
        return base
    raise ValueError(
        f"BayesBench requires a PLINK prefix after cache/normalization, but did not find "
        f"`{cand}.bed/.bim/.fam`."
    )


def _resolve_rscript_bin(args: argparse.Namespace) -> str:
    rscript_bin = str(getattr(args, "rscript", None) or (shutil.which("Rscript") or "Rscript")).strip()
    if rscript_bin == "":
        raise ValueError("--rscript is empty.")
    if shutil.which(rscript_bin) is None and (not Path(rscript_bin).expanduser().exists()):
        raise ValueError(f"Rscript executable was not found: {rscript_bin}")
    return rscript_bin


def _load_builtin_wheat_dataset(args: argparse.Namespace) -> dict[str, Any]:
    trait_idx = _parse_single_trait_arg(getattr(args, "ncol", None))
    trait_name = None if getattr(args, "trait", None) in {None, ""} else str(args.trait).strip()
    rscript_bin = _resolve_rscript_bin(args)
    with tempfile.TemporaryDirectory(prefix="jx_bayesbench_wheat_") as td:
        tmpdir = Path(td)
        x_tsv = tmpdir / "wheat.X.tsv"
        y_tsv = tmpdir / "wheat.Y.tsv"
        r_code = (
            "library(BGLR);"
            "data('wheat', package='BGLR');"
            f"write.table(wheat.X, file='{x_tsv.as_posix()}', sep='\\t', quote=FALSE, row.names=FALSE, col.names=FALSE);"
            f"write.table(wheat.Y, file='{y_tsv.as_posix()}', sep='\\t', quote=FALSE, row.names=FALSE, col.names=FALSE);"
        )
        proc = subprocess.run(
            [str(rscript_bin), "-e", r_code],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to export BGLR built-in wheat dataset.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        x = pd.read_csv(x_tsv, sep="\t", header=None).to_numpy(dtype=np.float64)
        y_mat = pd.read_csv(y_tsv, sep="\t", header=None).to_numpy(dtype=np.float64)

    if x.ndim != 2 or y_mat.ndim != 2:
        raise ValueError("BGLR wheat export produced invalid matrix dimensions.")
    if x.shape[0] != y_mat.shape[0]:
        raise ValueError(
            f"BGLR wheat sample mismatch: X rows={int(x.shape[0])}, Y rows={int(y_mat.shape[0])}."
        )
    trait_labels = [f"trait{i}" for i in range(int(y_mat.shape[1]))]
    if trait_name is not None:
        if trait_name not in trait_labels:
            raise ValueError(
                f"`--trait {trait_name}` is invalid for builtin wheat. "
                f"Valid names are: {', '.join(trait_labels)}."
            )
        target_idx = trait_labels.index(trait_name)
    elif trait_idx is None:
        target_idx = 0
    else:
        target_idx = int(trait_idx)
        if target_idx < 0 or target_idx >= int(y_mat.shape[1]):
            raise ValueError(
                f"`-n/--n` out of range for builtin wheat: {target_idx}. "
                f"Valid range is 0..{int(y_mat.shape[1]) - 1}."
            )
    y = np.asarray(y_mat[:, target_idx], dtype=np.float64).reshape(-1)
    keep_samples = np.isfinite(y)
    if int(np.sum(keep_samples)) < 8:
        raise ValueError(
            f"Builtin wheat trait has too few usable samples after filtering: {int(np.sum(keep_samples))}."
        )
    x_keep = np.ascontiguousarray(x[keep_samples, :], dtype=np.float64)
    y_keep = np.ascontiguousarray(y[keep_samples], dtype=np.float64)
    sample_ids_keep = np.asarray(
        [f"wheat_{i + 1:04d}" for i in np.flatnonzero(keep_samples)],
        dtype=str,
    )
    marker = np.ascontiguousarray(np.rint(x_keep.T).astype(np.int8, copy=False), dtype=np.int8)
    packed = _pack_bed_from_dosage(marker)
    mean = np.ascontiguousarray(marker.mean(axis=1).astype(np.float32, copy=False), dtype=np.float32)
    std = np.ascontiguousarray(marker.std(axis=1).astype(np.float32, copy=False), dtype=np.float32)
    inv = np.zeros_like(std, dtype=np.float32)
    good = std > 0.0
    inv[good] = np.asarray(1.0 / (std[good] + 1e-6), dtype=np.float32)
    maf_like = np.ascontiguousarray(np.minimum(mean, 1.0 - mean).astype(np.float32, copy=False), dtype=np.float32)
    packed_ctx: dict[str, Any] = {
        "packed": packed,
        "missing_rate": np.zeros((int(marker.shape[0]),), dtype=np.float32),
        "maf": maf_like,
        "std_denom": std,
        "row_flip": np.zeros((int(marker.shape[0]),), dtype=np.bool_),
        "site_keep": None,
        "n_samples": int(marker.shape[1]),
        "n_total_sites": int(marker.shape[0]),
        "n_active_sites": int(marker.shape[0]),
        "active_row_idx": np.ascontiguousarray(np.arange(int(marker.shape[0]), dtype=np.int64), dtype=np.int64),
        "packed_filter_mode": "compact",
        "packed_storage": "owned",
        "source_prefix": "BGLR:wheat",
    }
    n_active_before_cap = int(marker.shape[0])
    packed_ctx, row_subset = _limit_active_snps(
        packed_ctx=packed_ctx,
        max_snps=(None if getattr(args, "max_snps", None) is None else int(args.max_snps)),
        seed=int(getattr(args, "seed", 20260429)),
    )
    packed_ctx["__std_row_mean__"] = np.ascontiguousarray(mean[row_subset], dtype=np.float32)
    packed_ctx["__std_row_inv_sd__"] = np.ascontiguousarray(inv[row_subset], dtype=np.float32)
    return {
        "plink_prefix": "BGLR:wheat",
        "sample_ids": sample_ids_keep,
        "packed_ctx": packed_ctx,
        "trait_name": str(trait_labels[target_idx]),
        "overlap_abs_idx": np.ascontiguousarray(np.arange(int(sample_ids_keep.shape[0]), dtype=np.int64), dtype=np.int64),
        "overlap_ids": sample_ids_keep,
        "y_all": y_keep,
        "row_mean": np.ascontiguousarray(np.asarray(mean[row_subset], dtype=np.float32).reshape(-1), dtype=np.float32),
        "row_inv_sd": np.ascontiguousarray(np.asarray(inv[row_subset], dtype=np.float32).reshape(-1), dtype=np.float32),
        "row_subset": np.ascontiguousarray(row_subset, dtype=np.int64),
        "n_active_before_cap": int(n_active_before_cap),
        "n_active_after_cap": int(np.asarray(packed_ctx["packed"]).shape[0]),
        "builtin": "wheat",
    }


def _subset_packed_ctx_rows(
    packed_ctx: dict[str, Any],
    row_idx: np.ndarray,
) -> dict[str, Any]:
    ridx = np.ascontiguousarray(np.asarray(row_idx, dtype=np.int64).reshape(-1), dtype=np.int64)
    out = dict(packed_ctx)
    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"])[ridx, :], dtype=np.uint8)
    out["packed"] = packed
    for key, dtype in (
        ("maf", np.float32),
        ("missing_rate", np.float32),
        ("std_denom", np.float32),
        ("row_flip", np.bool_),
    ):
        raw = packed_ctx.get(key, None)
        if raw is None:
            continue
        out[key] = np.ascontiguousarray(np.asarray(raw, dtype=dtype).reshape(-1)[ridx], dtype=dtype)
    out["n_active_sites"] = int(ridx.shape[0])
    out["active_row_idx"] = np.ascontiguousarray(np.arange(int(ridx.shape[0]), dtype=np.int64), dtype=np.int64)
    out["site_keep"] = None
    out.pop("__std_row_mean__", None)
    out.pop("__std_row_inv_sd__", None)
    return out


def _limit_active_snps(
    packed_ctx: dict[str, Any],
    max_snps: int | None,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    packed = np.asarray(packed_ctx["packed"], dtype=np.uint8)
    m = int(packed.shape[0])
    row_idx = np.ascontiguousarray(np.arange(m, dtype=np.int64), dtype=np.int64)
    if max_snps is None or int(max_snps) <= 0 or m <= int(max_snps):
        return packed_ctx, row_idx
    rng = np.random.default_rng(int(seed))
    keep = np.sort(rng.choice(m, size=int(max_snps), replace=False).astype(np.int64, copy=False))
    return _subset_packed_ctx_rows(packed_ctx, keep), keep


def _load_real_dataset(args: argparse.Namespace) -> dict[str, Any]:
    builtin = str(getattr(args, "builtin", "") or "").strip().lower()
    if builtin == "wheat":
        return _load_builtin_wheat_dataset(args)
    plink_prefix = _resolve_plink_prefix_from_args(args)
    sample_ids, packed_ctx = prepare_packed_ctx_from_plink(
        plink_prefix,
        maf=float(args.maf),
        missing_rate=float(args.geno),
        het_threshold=float(0.0),
        snps_only=bool(args.snps_only),
        filter_mode="compact",
    )
    sample_ids = np.asarray(sample_ids, dtype=str)
    n_active_before_cap = int(np.asarray(packed_ctx["packed"]).shape[0])
    packed_ctx, row_subset = _limit_active_snps(
        packed_ctx=packed_ctx,
        max_snps=(None if getattr(args, "max_snps", None) is None else int(args.max_snps)),
        seed=int(getattr(args, "seed", 20260429)),
    )
    trait_idx = _parse_single_trait_arg(getattr(args, "ncol", None))
    pheno_df, trait_name = _read_pheno(
        Path(str(args.pheno)).expanduser(),
        trait_idx=trait_idx,
        trait_name=(None if getattr(args, "trait", None) in {None, ""} else str(args.trait)),
    )
    pheno_map = {
        str(k): float(v)
        for k, v in zip(pheno_df["Taxa"].astype(str), pheno_df["PHENO"].astype(float))
    }
    overlap_mask = np.fromiter((sid in pheno_map for sid in sample_ids), dtype=bool, count=sample_ids.shape[0])
    if int(np.sum(overlap_mask)) < 8:
        raise ValueError(
            f"Not enough overlapped samples between genotype and phenotype: {int(np.sum(overlap_mask))}."
        )
    overlap_abs_idx = np.flatnonzero(overlap_mask).astype(np.int64, copy=False)
    overlap_ids = np.asarray(sample_ids[overlap_mask], dtype=str)
    y_all = np.asarray([pheno_map[sid] for sid in overlap_ids], dtype=np.float64)
    row_mean, row_inv_sd = _ensure_packed_standard_stats_cached(packed_ctx)
    return {
        "plink_prefix": str(plink_prefix),
        "sample_ids": sample_ids,
        "packed_ctx": packed_ctx,
        "trait_name": str(trait_name),
        "overlap_abs_idx": np.ascontiguousarray(overlap_abs_idx, dtype=np.int64),
        "overlap_ids": overlap_ids,
        "y_all": np.ascontiguousarray(y_all, dtype=np.float64),
        "row_mean": np.ascontiguousarray(np.asarray(row_mean, dtype=np.float32).reshape(-1), dtype=np.float32),
        "row_inv_sd": np.ascontiguousarray(np.asarray(row_inv_sd, dtype=np.float32).reshape(-1), dtype=np.float32),
        "row_subset": np.ascontiguousarray(row_subset, dtype=np.int64),
        "n_active_before_cap": int(n_active_before_cap),
        "n_active_after_cap": int(np.asarray(packed_ctx["packed"]).shape[0]),
    }


def _prefix_rhat_from_matrix(mat: np.ndarray) -> np.ndarray:
    x = np.asarray(mat, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return np.full((int(x.shape[1]) if x.ndim == 2 else 0,), np.nan, dtype=np.float64)
    n_chains, n_keep = x.shape
    idx = np.arange(1, n_keep + 1, dtype=np.float64)
    csum = np.cumsum(x, axis=1, dtype=np.float64)
    csum2 = np.cumsum(x * x, axis=1, dtype=np.float64)
    means = csum / idx.reshape(1, -1)
    vars_ = np.full_like(means, np.nan, dtype=np.float64)
    valid = idx > 1.0
    vars_[:, valid] = (
        csum2[:, valid] - idx[valid].reshape(1, -1) * means[:, valid] * means[:, valid]
    ) / (idx[valid].reshape(1, -1) - 1.0)
    rhat = np.full((n_keep,), np.nan, dtype=np.float64)
    if np.any(valid):
        w = np.nanmean(vars_[:, valid], axis=0)
        b = idx[valid] * np.nanvar(means[:, valid], axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            varhat = ((idx[valid] - 1.0) / idx[valid]) * w + (b / idx[valid])
            rhat_valid = np.sqrt(varhat / w)
        rhat_valid[~np.isfinite(rhat_valid)] = np.nan
        rhat_valid = np.maximum(rhat_valid, 1.0)
        rhat[valid] = rhat_valid
    return np.asarray(rhat, dtype=np.float64)


def _analyze_convergence_outputs(
    *,
    trace_df: pd.DataFrame,
    global_trace_df: pd.DataFrame,
    beta_trace_df: pd.DataFrame,
    out_dir: Path,
    burnin: int,
    rhat_threshold: float,
    stable_min_kept: int,
    plot_top_k_beta: int,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    metrics = ["h2", "var_e", "prob_in", "n_active"]
    prefix_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    plot_files: dict[str, dict[str, str]] = {}
    for method, method_df in trace_df.groupby("method", sort=False):
        method_dir = out_dir / str(method)
        method_dir.mkdir(parents=True, exist_ok=True)
        plot_files[str(method)] = {}
        metric_rhat_kept: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        metric_rhat_plot: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        active_metrics: list[str] = []
        seeds: list[int] = []
        global_method_df = (
            global_trace_df[global_trace_df["method"] == method].copy()
            if not global_trace_df.empty
            else pd.DataFrame()
        )
        for metric in metrics:
            sub = method_df[["chain_seed", "iter", metric]].dropna()
            if sub.empty:
                continue
            active_metrics.append(metric)
            pivot = sub.pivot(index="chain_seed", columns="iter", values=metric).sort_index(axis=0).sort_index(axis=1)
            vals = pivot.to_numpy(dtype=np.float64)
            iters = np.asarray(pivot.columns.to_numpy(dtype=np.int64), dtype=np.int64)
            seeds = [int(x) for x in pivot.index.to_list()]
            rhat_kept = _prefix_rhat_from_matrix(vals)
            metric_rhat_kept[metric] = (iters, rhat_kept)

            plot_sub = global_method_df[["chain_seed", "iter", metric]].dropna() if not global_method_df.empty else pd.DataFrame()
            if not plot_sub.empty:
                plot_pivot = (
                    plot_sub.pivot(index="chain_seed", columns="iter", values=metric)
                    .sort_index(axis=0)
                    .sort_index(axis=1)
                )
                plot_vals = plot_pivot.to_numpy(dtype=np.float64)
                plot_iters = np.asarray(plot_pivot.columns.to_numpy(dtype=np.int64), dtype=np.int64)
                plot_rhat = _prefix_rhat_from_matrix(plot_vals)
            else:
                plot_iters = iters
                plot_rhat = rhat_kept
            metric_rhat_plot[metric] = (plot_iters, plot_rhat)

            for idx, (iter_no, rhat_val) in enumerate(zip(plot_iters, plot_rhat)):
                kept_post = int(np.searchsorted(iters, int(iter_no), side="right"))
                prefix_rows.append(
                    {
                        "method": str(method),
                        "metric": str(metric),
                        "iter": int(iter_no),
                        "phase": (
                            "burnin"
                            if int(iter_no) <= int(burnin)
                            else "post_burnin"
                        ),
                        "prefix_kind": (
                            "full_iter"
                            if not plot_sub.empty
                            else "kept_only"
                        ),
                        "prefix_per_chain": int(idx + 1),
                        "kept_per_chain": int(kept_post),
                        "rhat": float(rhat_val),
                    }
                )
        if not active_metrics:
            continue

        common_iters = metric_rhat_kept[active_metrics[0]][0]
        overall_ok = np.ones((int(common_iters.shape[0]),), dtype=bool)
        for metric in active_metrics:
            _, rhat = metric_rhat_kept[metric]
            ok = np.isfinite(rhat) & (rhat <= float(rhat_threshold))
            if stable_min_kept > 1:
                ok[: int(stable_min_kept) - 1] = False
            overall_ok &= ok
        suffix_ok = np.flip(np.cumprod(np.flip(overall_ok.astype(np.int8)))).astype(bool)
        stable_idx = int(np.flatnonzero(suffix_ok)[0]) if np.any(suffix_ok) else -1
        stable_iter = int(common_iters[stable_idx]) if stable_idx >= 0 else -1
        stable_kept = int(stable_idx + 1) if stable_idx >= 0 else -1
        stability_row: dict[str, Any] = {
            "method": str(method),
            "metrics_used": ",".join(active_metrics),
            "rhat_threshold": float(rhat_threshold),
            "stable_min_kept": int(stable_min_kept),
            "stable_iter": int(stable_iter),
            "stable_kept_per_chain": int(stable_kept),
        }
        for metric in active_metrics:
            _, rhat = metric_rhat_kept[metric]
            stability_row[f"final_rhat_{metric}"] = float(rhat[-1]) if rhat.size > 0 else float("nan")
        stability_rows.append(stability_row)

        n_panels = len(active_metrics)
        plot_df = global_method_df.copy()
        if plot_df.empty:
            plot_df = method_df.copy()
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, max(3.0, 2.6 * n_panels)), sharex=True)
        if n_panels == 1:
            axes = [axes]
        for ax, metric in zip(axes, active_metrics):
            for seed, chain_df in plot_df[["chain_seed", "iter", metric]].dropna().groupby("chain_seed", sort=True):
                chain_df = chain_df.sort_values("iter")
                pre = chain_df[chain_df["iter"].astype(np.int64) <= int(burnin)]
                post = chain_df[chain_df["iter"].astype(np.int64) > int(burnin)]
                if not pre.empty:
                    ax.plot(
                        pre["iter"],
                        pre[metric],
                        color="#9aa0a6",
                        linewidth=0.85,
                        alpha=0.8,
                        linestyle="--",
                    )
                if not post.empty:
                    ax.plot(post["iter"], post[metric], linewidth=0.95, alpha=0.95, label=f"chain {int(seed)}")
            ax.axvline(float(int(burnin)), color="#4b5563", linestyle=":", linewidth=1.0, alpha=0.9)
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.25, linewidth=0.5)
        axes[0].set_title(f"{method} global parameter traces")
        axes[-1].set_xlabel("iteration")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, ncol=min(4, len(handles)), fontsize=8, frameon=False)
        fig.tight_layout()
        trace_plot = method_dir / "global_trace.png"
        trace_plot_pdf = method_dir / "global_trace.pdf"
        fig.savefig(trace_plot, dpi=220, bbox_inches="tight")
        fig.savefig(trace_plot_pdf, bbox_inches="tight")
        plt.close(fig)
        plot_files[str(method)]["global_trace"] = str(trace_plot)
        plot_files[str(method)]["global_trace_pdf"] = str(trace_plot_pdf)

        fig, axes = plt.subplots(n_panels, 1, figsize=(10, max(3.0, 2.6 * n_panels)), sharex=True)
        if n_panels == 1:
            axes = [axes]
        for ax, metric in zip(axes, active_metrics):
            iters, rhat = metric_rhat_plot[metric]
            pre_mask = np.asarray(iters <= int(burnin), dtype=bool)
            post_mask = np.asarray(iters > int(burnin), dtype=bool)
            if np.any(pre_mask):
                ax.plot(
                    iters[pre_mask],
                    rhat[pre_mask],
                    color="#9aa0a6",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.9,
                )
            if np.any(post_mask):
                ax.plot(
                    iters[post_mask],
                    rhat[post_mask],
                    color="#1f77b4",
                    linewidth=1.1,
                    alpha=0.95,
                )
            ax.axhline(float(rhat_threshold), color="#d62728", linestyle="--", linewidth=1.0)
            ax.axvspan(1.0, float(int(burnin)), color="#9aa0a6", alpha=0.12, linewidth=0.0)
            if stable_iter > 0:
                ax.axvline(float(stable_iter), color="#2ca02c", linestyle=":", linewidth=1.0)
            ax.set_ylabel(f"R-hat {metric}")
            ax.grid(True, alpha=0.25, linewidth=0.5)
        axes[0].set_title(f"{method} prefix R-hat")
        axes[-1].set_xlabel("iteration")
        fig.tight_layout()
        rhat_plot = method_dir / "prefix_rhat.png"
        rhat_plot_pdf = method_dir / "prefix_rhat.pdf"
        fig.savefig(rhat_plot, dpi=220, bbox_inches="tight")
        fig.savefig(rhat_plot_pdf, bbox_inches="tight")
        plt.close(fig)
        plot_files[str(method)]["prefix_rhat"] = str(rhat_plot)
        plot_files[str(method)]["prefix_rhat_pdf"] = str(rhat_plot_pdf)

        beta_method = pd.DataFrame()
        if (not beta_trace_df.empty) and ("method" in beta_trace_df.columns):
            beta_method = beta_trace_df[beta_trace_df["method"] == method].copy()
        if not beta_method.empty:
            ranks = sorted(beta_method["rank"].dropna().astype(int).unique().tolist())
            ranks = ranks[: max(1, int(plot_top_k_beta))]
            n_plot = len(ranks)
            n_cols = 2
            n_rows = int(math.ceil(n_plot / n_cols))
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(12, max(3.5, 2.6 * n_rows)),
                sharex=True,
            )
            axes_arr = np.atleast_1d(axes).reshape(-1)
            for ax, rank in zip(axes_arr, ranks):
                sub = beta_method[beta_method["rank"] == int(rank)]
                snp_idx = int(sub["snp_index"].iloc[0])
                for seed, chain_df in sub.groupby("chain_seed", sort=True):
                    chain_df = chain_df.sort_values("iter")
                    ax.plot(chain_df["iter"], chain_df["beta"], linewidth=0.9, alpha=0.9, label=f"chain {int(seed)}")
                ax.set_title(f"rank {rank} / SNP {snp_idx}", fontsize=9)
                ax.grid(True, alpha=0.25, linewidth=0.5)
            for ax in axes_arr[n_plot:]:
                ax.axis("off")
            axes_arr[0].legend(frameon=False, fontsize=8, ncol=min(4, len(seeds) if seeds else 4))
            fig.suptitle(f"{method} top SNP beta traces", y=0.995)
            fig.tight_layout()
            beta_plot = method_dir / "topbeta_trace.png"
            beta_plot_pdf = method_dir / "topbeta_trace.pdf"
            fig.savefig(beta_plot, dpi=220, bbox_inches="tight")
            fig.savefig(beta_plot_pdf, bbox_inches="tight")
            plt.close(fig)
            plot_files[str(method)]["topbeta_trace"] = str(beta_plot)
            plot_files[str(method)]["topbeta_trace_pdf"] = str(beta_plot_pdf)

    prefix_df = pd.DataFrame(prefix_rows)
    stability_df = pd.DataFrame(stability_rows)
    prefix_path = out_dir / "convergence.prefix_rhat.tsv"
    stability_path = out_dir / "convergence.stability.tsv"
    prefix_df.to_csv(prefix_path, sep="\t", index=False)
    stability_df.to_csv(stability_path, sep="\t", index=False)
    md_lines = [
        "# JanusX Bayes Convergence Stability",
        "",
        "| method | stable_iter | kept_per_chain | threshold | metrics |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in stability_df.iterrows():
        md_lines.append(
            f"| {row['method']} | {int(row['stable_iter']) if int(row['stable_iter']) >= 0 else 'NA'} "
            f"| {int(row['stable_kept_per_chain']) if int(row['stable_kept_per_chain']) >= 0 else 'NA'} "
            f"| {float(row['rhat_threshold']):.3f} | {row['metrics_used']} |"
        )
    stability_md_path = out_dir / "convergence.stability.md"
    stability_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {
        "prefix_rhat_path": str(prefix_path),
        "stability_path": str(stability_path),
        "stability_md_path": str(stability_md_path),
        "plot_files": plot_files,
    }


def _make_train_test_split(n: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 8:
        raise ValueError("Need at least 8 samples for compare split.")
    frac = float(test_frac)
    if not (0.0 < frac < 1.0):
        raise ValueError("--test-frac must be in (0, 1).")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n).astype(np.int64, copy=False)
    n_test = int(round(frac * n))
    n_test = max(1, min(n - 1, n_test))
    test_rel = np.sort(perm[:n_test])
    train_rel = np.sort(perm[n_test:])
    return train_rel, test_rel


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


def _fit_janusx_packed(
    *,
    method: str,
    packed_ctx: dict[str, Any],
    sample_indices: np.ndarray,
    y: np.ndarray,
    row_mean: np.ndarray,
    row_inv_sd: np.ndarray,
    n_iter: int,
    burnin: int,
    thin: int,
    r2: float,
    prob_in: float,
    counts: float,
    df0_b: float,
    shape0: float,
    rate0: float | None,
    s0_b: float | None,
    df0_e: float,
    s0_e: float | None,
    threads: int,
    seed: int,
) -> dict[str, Any]:
    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8), dtype=np.uint8)
    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1), dtype=np.float32)
    row_flip = np.ascontiguousarray(np.asarray(packed_ctx["row_flip"], dtype=np.bool_).reshape(-1), dtype=np.bool_)
    common: dict[str, Any] = {
        "y": np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1), dtype=np.float64),
        "packed": packed,
        "n_samples": int(packed_ctx["n_samples"]),
        "row_flip": row_flip,
        "row_maf": maf,
        "row_mean": np.ascontiguousarray(np.asarray(row_mean, dtype=np.float32).reshape(-1), dtype=np.float32),
        "row_inv_sd": np.ascontiguousarray(np.asarray(row_inv_sd, dtype=np.float32).reshape(-1), dtype=np.float32),
        "sample_indices": np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1), dtype=np.int64),
        "x": None,
        "n_iter": int(n_iter),
        "burnin": int(burnin),
        "thin": int(thin),
        "r2": float(r2),
        "df0_b": float(df0_b),
        "df0_e": float(df0_e),
        "threads": int(max(0, int(threads))),
        "seed": int(seed),
    }
    if s0_e is not None:
        common["s0_e"] = float(s0_e)
    if method == "BayesA":
        if rate0 is not None:
            common["rate0"] = float(rate0)
        if s0_b is not None:
            common["s0_b"] = float(s0_b)
        common["shape0"] = float(shape0)
        common["min_abs_beta"] = 1e-9
        ret = _jxrs.bayesa_packed(**common)
        beta_raw, alpha_raw, varb_raw, vare, h2_mean, var_h2 = ret
        prob_in_mean = float("nan")
        n_active_mean = float("nan")
    elif method == "BayesB":
        if rate0 is not None:
            common["rate0"] = float(rate0)
        if s0_b is not None:
            common["s0_b"] = float(s0_b)
        common["shape0"] = float(shape0)
        common["prob_in"] = float(prob_in)
        common["counts"] = float(counts)
        ret = _jxrs.bayesb_packed(**common)
        beta_raw, alpha_raw, varb_raw, vare, h2_mean, var_h2, prob_in_mean, n_active_mean = ret
    elif method == "BayesCpi":
        if s0_b is not None:
            common["s0_b"] = float(s0_b)
        common["prob_in"] = float(prob_in)
        common["counts"] = float(counts)
        ret = _jxrs.bayescpi_packed(**common)
        beta_raw, alpha_raw, varb_raw, vare, h2_mean, var_h2, prob_in_mean, n_active_mean = ret
    else:
        raise ValueError(f"Unsupported method: {method}")
    beta = np.ascontiguousarray(np.asarray(beta_raw, dtype=np.float64).reshape(-1), dtype=np.float64)
    alpha = np.ascontiguousarray(np.asarray(alpha_raw, dtype=np.float64).reshape(-1), dtype=np.float64)
    return {
        "method": str(method),
        "beta": beta,
        "alpha": alpha,
        "alpha0": float(alpha[0]) if int(alpha.shape[0]) > 0 else 0.0,
        "vare": float(vare),
        "h2_mean": float(h2_mean),
        "var_h2": float(var_h2),
        "prob_in_mean": _safe_float(prob_in_mean),
        "n_active_mean": _safe_float(n_active_mean),
        "seed": int(seed),
    }


def _fit_janusx_packed_trace(
    *,
    method: str,
    packed_ctx: dict[str, Any],
    sample_indices: np.ndarray,
    y: np.ndarray,
    row_mean: np.ndarray,
    row_inv_sd: np.ndarray,
    trace_snp_indices: np.ndarray,
    n_iter: int,
    burnin: int,
    thin: int,
    r2: float,
    prob_in: float,
    counts: float,
    df0_b: float,
    shape0: float,
    rate0: float | None,
    s0_b: float | None,
    df0_e: float,
    s0_e: float | None,
    threads: int,
    seed: int,
) -> dict[str, Any]:
    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8), dtype=np.uint8)
    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1), dtype=np.float32)
    row_flip = np.ascontiguousarray(np.asarray(packed_ctx["row_flip"], dtype=np.bool_).reshape(-1), dtype=np.bool_)
    trace_idx = np.ascontiguousarray(np.asarray(trace_snp_indices, dtype=np.int64).reshape(-1), dtype=np.int64)
    common: dict[str, Any] = {
        "y": np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1), dtype=np.float64),
        "packed": packed,
        "n_samples": int(packed_ctx["n_samples"]),
        "row_flip": row_flip,
        "row_maf": maf,
        "row_mean": np.ascontiguousarray(np.asarray(row_mean, dtype=np.float32).reshape(-1), dtype=np.float32),
        "row_inv_sd": np.ascontiguousarray(np.asarray(row_inv_sd, dtype=np.float32).reshape(-1), dtype=np.float32),
        "sample_indices": np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1), dtype=np.int64),
        "x": None,
        "n_iter": int(n_iter),
        "burnin": int(burnin),
        "thin": int(thin),
        "r2": float(r2),
        "df0_b": float(df0_b),
        "df0_e": float(df0_e),
        "trace_snp_indices": trace_idx,
        "threads": int(max(0, int(threads))),
        "seed": int(seed),
    }
    if s0_e is not None:
        common["s0_e"] = float(s0_e)
    if method == "BayesA":
        if rate0 is not None:
            common["rate0"] = float(rate0)
        if s0_b is not None:
            common["s0_b"] = float(s0_b)
        common["shape0"] = float(shape0)
        common["min_abs_beta"] = 1e-9
        ret = _jxrs.bayesa_packed_trace(**common)
    elif method == "BayesB":
        if rate0 is not None:
            common["rate0"] = float(rate0)
        if s0_b is not None:
            common["s0_b"] = float(s0_b)
        common["shape0"] = float(shape0)
        common["prob_in"] = float(prob_in)
        common["counts"] = float(counts)
        ret = _jxrs.bayesb_packed_trace(**common)
    elif method == "BayesCpi":
        if s0_b is not None:
            common["s0_b"] = float(s0_b)
        common["prob_in"] = float(prob_in)
        common["counts"] = float(counts)
        ret = _jxrs.bayescpi_packed_trace(**common)
    else:
        raise ValueError(f"Unsupported method: {method}")
    beta = np.ascontiguousarray(np.asarray(ret["beta"], dtype=np.float64).reshape(-1), dtype=np.float64)
    alpha = np.ascontiguousarray(np.asarray(ret["alpha"], dtype=np.float64).reshape(-1), dtype=np.float64)
    iter_trace = np.ascontiguousarray(np.asarray(ret["iter_trace"], dtype=np.int64).reshape(-1), dtype=np.int64)
    beta_trace_indices = np.ascontiguousarray(
        np.asarray(ret["beta_trace_indices"], dtype=np.int64).reshape(-1),
        dtype=np.int64,
    )
    beta_trace = np.ascontiguousarray(np.asarray(ret["beta_trace"], dtype=np.float64), dtype=np.float64)
    beta_trace = beta_trace.reshape((int(iter_trace.shape[0]), int(beta_trace_indices.shape[0])))
    return {
        "method": str(method),
        "beta": beta,
        "alpha": alpha,
        "alpha0": float(alpha[0]) if int(alpha.shape[0]) > 0 else 0.0,
        "vare": float(ret["vare"]),
        "h2_mean": float(ret["h2_mean"]),
        "var_h2": float(ret["var_h2"]),
        "prob_in_mean": _safe_float(ret["prob_in_mean"]),
        "n_active_mean": _safe_float(ret["n_active_mean"]),
        "iter_trace": iter_trace,
        "h2_trace": np.ascontiguousarray(np.asarray(ret["h2_trace"], dtype=np.float64).reshape(-1), dtype=np.float64),
        "var_e_trace": np.ascontiguousarray(
            np.asarray(ret["var_e_trace"], dtype=np.float64).reshape(-1),
            dtype=np.float64,
        ),
        "prob_in_trace": np.ascontiguousarray(
            np.asarray(ret["prob_in_trace"], dtype=np.float64).reshape(-1),
            dtype=np.float64,
        ),
        "n_active_trace": np.ascontiguousarray(
            np.asarray(ret["n_active_trace"], dtype=np.float64).reshape(-1),
            dtype=np.float64,
        ),
        "full_iter_trace": np.ascontiguousarray(
            np.asarray(ret["full_iter_trace"], dtype=np.int64).reshape(-1),
            dtype=np.int64,
        ),
        "full_h2_trace": np.ascontiguousarray(
            np.asarray(ret["full_h2_trace"], dtype=np.float64).reshape(-1),
            dtype=np.float64,
        ),
        "full_var_e_trace": np.ascontiguousarray(
            np.asarray(ret["full_var_e_trace"], dtype=np.float64).reshape(-1),
            dtype=np.float64,
        ),
        "full_prob_in_trace": np.ascontiguousarray(
            np.asarray(ret["full_prob_in_trace"], dtype=np.float64).reshape(-1),
            dtype=np.float64,
        ),
        "full_n_active_trace": np.ascontiguousarray(
            np.asarray(ret["full_n_active_trace"], dtype=np.float64).reshape(-1),
            dtype=np.float64,
        ),
        "beta_trace_indices": beta_trace_indices,
        "beta_trace": beta_trace,
        "seed": int(seed),
    }


def _predict_janusx(
    *,
    fit: dict[str, Any],
    packed_ctx: dict[str, Any],
    sample_indices: np.ndarray,
    row_mean: np.ndarray,
    row_inv_sd: np.ndarray,
    snp_block_size: int,
    sample_chunk_size: int,
) -> np.ndarray:
    return np.asarray(
        _predict_bayes_packed_from_effects(
            packed_ctx=packed_ctx,
            sample_indices=np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1), dtype=np.int64),
            alpha0=float(fit["alpha0"]),
            beta=np.ascontiguousarray(np.asarray(fit["beta"], dtype=np.float64).reshape(-1), dtype=np.float64),
            row_mean=np.ascontiguousarray(np.asarray(row_mean, dtype=np.float32).reshape(-1), dtype=np.float32),
            row_inv_sd=np.ascontiguousarray(np.asarray(row_inv_sd, dtype=np.float32).reshape(-1), dtype=np.float32),
            snp_block_size=int(max(1, int(snp_block_size))),
            sample_chunk_size=int(max(1, int(sample_chunk_size))),
        ),
        dtype=np.float64,
    ).reshape(-1)


def _write_float32_matrix(path: Path, arr: np.ndarray) -> None:
    x = np.ascontiguousarray(np.asarray(arr, dtype=np.float32), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    x.tofile(str(path))


def _compare_subproc_env(thread_count: int | None = None) -> dict[str, str]:
    env = dict(os.environ)
    py_root = str(Path(__file__).resolve().parents[2])
    old = str(env.get("PYTHONPATH", "")).strip()
    env["PYTHONPATH"] = py_root if old == "" else f"{py_root}{os.pathsep}{old}"
    pin_threads = 1 if thread_count is None or int(thread_count) <= 0 else int(thread_count)
    pin_val = str(pin_threads)
    # Compare mode uses one explicit thread knob (`-t/--thread`) for JanusX,
    # HIBayes and external BLAS/OpenMP backends. Default remains single-threaded
    # for fair cross-software benchmarking, but callers can open the multicore
    # backdoor with `-t N`.
    env.setdefault("OMP_NUM_THREADS", pin_val)
    env.setdefault("OPENBLAS_NUM_THREADS", pin_val)
    env.setdefault("MKL_NUM_THREADS", pin_val)
    env.setdefault("BLIS_NUM_THREADS", pin_val)
    env.setdefault("NUMEXPR_NUM_THREADS", pin_val)
    env.setdefault("VECLIB_MAXIMUM_THREADS", pin_val)
    env.setdefault("ACCELERATE_NTHREADS", pin_val)
    return env


def _write_janusx_compare_script(path: Path) -> None:
    code = r"""#!/usr/bin/env python
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from janusx.script.bayesbench import _fit_janusx_packed, _predict_janusx


def arg_val(name: str, default: str = "") -> str:
    args = list(sys.argv[1:])
    try:
        idx = args.index(name)
    except ValueError:
        return default
    j = idx + 1
    if j >= len(args):
        return default
    return args[j]


meta_path = Path(arg_val("--meta", "")).expanduser().resolve()
out_path = Path(arg_val("--out", "")).expanduser().resolve()
beta_out = Path(arg_val("--beta-out", "")).expanduser().resolve()
pred_out = Path(arg_val("--pred-out", "")).expanduser().resolve()
if str(meta_path) == "" or str(out_path) == "" or str(beta_out) == "" or str(pred_out) == "":
    raise SystemExit("missing --meta/--out/--beta-out/--pred-out")

meta = json.loads(meta_path.read_text(encoding="utf-8"))
packed_ctx = {
    "packed": np.load(meta["packed_npy"], allow_pickle=False),
    "maf": np.load(meta["maf_npy"], allow_pickle=False),
    "row_flip": np.load(meta["row_flip_npy"], allow_pickle=False),
    "n_samples": int(meta["n_samples"]),
}
row_mean = np.load(meta["row_mean_npy"], allow_pickle=False)
row_inv_sd = np.load(meta["row_inv_sd_npy"], allow_pickle=False)
train_abs = np.load(meta["train_abs_npy"], allow_pickle=False)
test_abs = np.load(meta["test_abs_npy"], allow_pickle=False)
train_ids = np.load(meta["train_ids_npy"], allow_pickle=False)
test_ids = np.load(meta["test_ids_npy"], allow_pickle=False)
y_train = np.load(meta["y_train_npy"], allow_pickle=False)
y_test = np.load(meta["y_test_npy"], allow_pickle=False)

fit = _fit_janusx_packed(
    method=str(meta["method"]),
    packed_ctx=packed_ctx,
    sample_indices=train_abs,
    y=y_train,
    row_mean=row_mean,
    row_inv_sd=row_inv_sd,
    n_iter=int(meta["n_iter"]),
    burnin=int(meta["burnin"]),
    thin=int(meta["thin"]),
    r2=float(meta["r2"]),
    prob_in=float(meta["prob_in"]),
    counts=float(meta["counts"]),
    df0_b=float(meta["df0_b"]),
    shape0=float(meta["shape0"]),
    rate0=(None if meta.get("rate0") is None else float(meta["rate0"])),
    s0_b=(None if meta.get("s0_b") is None else float(meta["s0_b"])),
    df0_e=float(meta["df0_e"]),
    s0_e=(None if meta.get("s0_e") is None else float(meta["s0_e"])),
    threads=int(meta["threads"]),
    seed=int(meta["seed"]),
)
pred_train = _predict_janusx(
    fit=fit,
    packed_ctx=packed_ctx,
    sample_indices=train_abs,
    row_mean=row_mean,
    row_inv_sd=row_inv_sd,
    snp_block_size=int(meta["snp_block_size"]),
    sample_chunk_size=int(meta["sample_chunk_size"]),
)
pred_test = _predict_janusx(
    fit=fit,
    packed_ctx=packed_ctx,
    sample_indices=test_abs,
    row_mean=row_mean,
    row_inv_sd=row_inv_sd,
    snp_block_size=int(meta["snp_block_size"]),
    sample_chunk_size=int(meta["sample_chunk_size"]),
)
alpha0 = float(fit["alpha0"])
gebv_train = np.asarray(pred_train - alpha0, dtype=np.float64)
gebv_test = np.asarray(pred_test - alpha0, dtype=np.float64)
var_g_train = float(np.var(gebv_train, ddof=1)) if int(gebv_train.shape[0]) > 1 else float("nan")
h2_compare = (
    float(var_g_train / (var_g_train + float(fit["vare"])))
    if np.isfinite(var_g_train) and np.isfinite(float(fit["vare"])) and (var_g_train + float(fit["vare"])) > 0.0
    else float("nan")
)

beta_df = pd.DataFrame(
    {
        "snp_index": np.arange(int(np.asarray(fit["beta"]).shape[0]), dtype=np.int64),
        "beta": np.asarray(fit["beta"], dtype=np.float64),
        "pip": np.full((int(np.asarray(fit["beta"]).shape[0]),), np.nan, dtype=np.float64),
    }
)
pred_df = pd.DataFrame(
    {
        "split": ["train"] * int(train_ids.shape[0]) + ["test"] * int(test_ids.shape[0]),
        "sample_id": np.concatenate([np.asarray(train_ids, dtype=str), np.asarray(test_ids, dtype=str)]),
        "y_true": np.concatenate([np.asarray(y_train, dtype=np.float64), np.asarray(y_test, dtype=np.float64)]),
        "y_pred": np.concatenate([np.asarray(pred_train, dtype=np.float64), np.asarray(pred_test, dtype=np.float64)]),
        "gebv": np.concatenate([gebv_train, gebv_test]),
        "intercept": alpha0,
    }
)

beta_out.parent.mkdir(parents=True, exist_ok=True)
pred_out.parent.mkdir(parents=True, exist_ok=True)
beta_df.to_csv(beta_out, sep="\t", index=False)
pred_df.to_csv(pred_out, sep="\t", index=False)
result = {
    "status": "ok",
    "engine": "JanusX",
    "method_input": str(meta["method"]),
    "model_reference": str(meta["method"]),
    "vare": float(fit["vare"]),
    "h2": h2_compare,
    "h2_native": float(fit["h2_mean"]),
    "h2_semantics": "train_var_posterior_mean_gebv_over_varg_plus_vare",
    "prob_in_mean": float(fit["prob_in_mean"]),
    "n_active_mean": float(fit["n_active_mean"]),
    "intercept": alpha0,
    "beta_out": str(beta_out),
    "pred_out": str(pred_out),
    "peak_rss_kb": (
        float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        if sys.platform == "darwin"
        else float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    ),
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
"""
    path.write_text(code, encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass


def _write_bglr_compare_script(path: Path) -> None:
    code = r"""#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
arg_val <- function(name, default = "") {
  idx <- which(args == name)
  if (length(idx) == 0) return(default)
  j <- idx[1] + 1
  if (j > length(args)) return(default)
  args[j]
}

meta_path <- arg_val("--meta", "")
out_path <- arg_val("--out", "")
beta_out <- arg_val("--beta-out", "")
pred_out <- arg_val("--pred-out", "")

if (!nzchar(meta_path)) stop("--meta is required")
if (!nzchar(out_path)) stop("--out is required")
if (!nzchar(beta_out)) stop("--beta-out is required")
if (!nzchar(pred_out)) stop("--pred-out is required")
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Missing R package: jsonlite")
if (!requireNamespace("BGLR", quietly = TRUE)) stop("Missing R package: BGLR")

meta <- jsonlite::fromJSON(meta_path)

approx_peak_kb <- function() {
  g <- gc()
  ncells <- as.numeric(g[1, "max used"])
  vcells <- as.numeric(g[2, "max used"])
  ((ncells * 56.0) + (vcells * 8.0)) / 1024.0
}

meta_num <- function(name) {
  x <- meta[[name]]
  if (is.null(x) || length(x) == 0) return(NA_real_)
  as.numeric(x)[1]
}

read_matrix_f32 <- function(path, nrow, ncol) {
  con <- file(path, "rb")
  on.exit(try(close(con), silent = TRUE), add = TRUE)
  vec <- readBin(con, what = "numeric", n = as.integer(nrow) * as.integer(ncol), size = 4, endian = "little")
  if (length(vec) != (as.integer(nrow) * as.integer(ncol))) {
    stop(sprintf("Matrix size mismatch for %s: got %d expected %d", path, length(vec), as.integer(nrow) * as.integer(ncol)))
  }
  matrix(vec, nrow = as.integer(nrow), ncol = as.integer(ncol), byrow = TRUE)
}

method_in <- as.character(meta$method)
model_ref <- if (tolower(method_in) == "bayescpi") "BayesC" else method_in
Xtr <- read_matrix_f32(meta$x_train_bin, meta$n_train, meta$p)
Xte <- read_matrix_f32(meta$x_test_bin, meta$n_test, meta$p)
ytr_df <- read.table(meta$y_train_tsv, header = TRUE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
yte_df <- read.table(meta$y_test_tsv, header = TRUE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
ytr <- as.numeric(ytr_df$y)
yte <- as.numeric(yte_df$y)

eta <- list(X = Xtr, model = model_ref, R2 = meta_num("r2"))
if (is.finite(meta_num("prob_in"))) eta$probIn <- meta_num("prob_in")
if (is.finite(meta_num("counts"))) eta$counts <- meta_num("counts")
if (is.finite(meta_num("df0_b"))) eta$df0 <- meta_num("df0_b")
if (is.finite(meta_num("s0_b"))) eta$S0 <- meta_num("s0_b")
if (model_ref %in% c("BayesA", "BayesB")) {
  if (is.finite(meta_num("shape0"))) eta$shape0 <- meta_num("shape0")
  if (is.finite(meta_num("rate0"))) eta$rate0 <- meta_num("rate0")
}

fit_args <- list(
  y = ytr,
  ETA = list(eta),
  nIter = as.integer(meta$n_iter),
  burnIn = as.integer(meta$burnin),
  thin = as.integer(meta$thin),
  R2 = meta_num("r2"),
  verbose = FALSE,
  rmExistingFiles = TRUE
)
if (is.finite(meta_num("df0_e"))) fit_args$df0 <- meta_num("df0_e")
if (is.finite(meta_num("s0_e"))) fit_args$S0 <- meta_num("s0_e")

t0 <- proc.time()[["elapsed"]]
fit <- do.call(BGLR::BGLR, fit_args)
elapsed_sec <- proc.time()[["elapsed"]] - t0

beta <- as.numeric(fit$ETA[[1]]$b)
mu <- as.numeric(fit$mu)[1]
vare <- as.numeric(fit$varE)[1]
pip <- fit$ETA[[1]]$d
if (is.null(pip)) {
  pip <- rep(NA_real_, length(beta))
}
pip <- as.numeric(pip)
prob_in_mean <- fit$ETA[[1]]$probIn
if (is.null(prob_in_mean) || length(prob_in_mean) == 0) prob_in_mean <- NA_real_
prob_in_mean <- as.numeric(prob_in_mean)[1]
n_active_mean <- if (all(is.na(pip))) sum(abs(beta) > 1e-12) else sum(pip, na.rm = TRUE)
if (model_ref == "BayesA") {
  pip[] <- NA_real_
  prob_in_mean <- NA_real_
  n_active_mean <- NA_real_
}

pred_tr <- as.numeric(mu + Xtr %*% beta)
pred_te <- as.numeric(mu + Xte %*% beta)
gebv_tr <- as.numeric(Xtr %*% beta)
gebv_te <- as.numeric(Xte %*% beta)
var_g <- stats::var(gebv_tr)
h2 <- if (is.finite(var_g) && is.finite(vare) && (var_g + vare) > 0) var_g / (var_g + vare) else NA_real_

beta_df <- data.frame(
  snp_index = seq_along(beta) - 1L,
  beta = beta,
  pip = pip,
  stringsAsFactors = FALSE
)

pred_df <- rbind(
  data.frame(
    split = "train",
    sample_id = as.character(ytr_df$sample_id),
    y_true = ytr,
    y_pred = pred_tr,
    gebv = gebv_tr,
    intercept = rep(mu, length(pred_tr)),
    stringsAsFactors = FALSE
  ),
  data.frame(
    split = "test",
    sample_id = as.character(yte_df$sample_id),
    y_true = yte,
    y_pred = pred_te,
    gebv = gebv_te,
    intercept = rep(mu, length(pred_te)),
    stringsAsFactors = FALSE
  )
)

dir.create(dirname(beta_out), recursive = TRUE, showWarnings = FALSE)
write.table(beta_df, file = beta_out, sep = "\t", quote = FALSE, row.names = FALSE)
dir.create(dirname(pred_out), recursive = TRUE, showWarnings = FALSE)
write.table(pred_df, file = pred_out, sep = "\t", quote = FALSE, row.names = FALSE)

result <- list(
  status = "ok",
  engine = "BGLR",
  method_input = method_in,
  model_reference = model_ref,
  vare = vare,
  h2 = h2,
  h2_native = NA_real_,
  h2_semantics = "train_var_posterior_mean_gebv_over_varg_plus_vare",
  prob_in_mean = prob_in_mean,
  n_active_mean = n_active_mean,
  intercept = mu,
  beta_out = beta_out,
  pred_out = pred_out,
  elapsed_sec = elapsed_sec,
  peak_rss_kb = approx_peak_kb()
)
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
writeLines(jsonlite::toJSON(result, auto_unbox = TRUE, pretty = TRUE), con = out_path)
"""
    path.write_text(code, encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass


def _run_bglr_reference(
    *,
    work_dir: Path,
    method: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    n_iter: int,
    burnin: int,
    thin: int,
    r2: float,
    prob_in: float,
    counts: float,
    df0_b: float,
    shape0: float,
    rate0: float | None,
    s0_b: float | None,
    df0_e: float,
    s0_e: float | None,
    threads: int,
    rscript_bin: str,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    x_train_bin = work_dir / "x_train.bin"
    x_test_bin = work_dir / "x_test.bin"
    y_train_tsv = work_dir / "y_train.tsv"
    y_test_tsv = work_dir / "y_test.tsv"
    meta_json = work_dir / "meta.json"
    out_json = work_dir / "result.json"
    beta_tsv = work_dir / "beta.tsv"
    pred_tsv = work_dir / "pred.tsv"
    script_path = work_dir / "run_bglr_compare.R"
    log_file = work_dir / "run.log"
    time_file = work_dir / "run.time"

    _write_float32_matrix(x_train_bin, x_train)
    _write_float32_matrix(x_test_bin, x_test)
    pd.DataFrame({"sample_id": np.asarray(train_ids, dtype=str), "y": np.asarray(y_train, dtype=np.float64)}).to_csv(
        y_train_tsv, sep="\t", index=False
    )
    pd.DataFrame({"sample_id": np.asarray(test_ids, dtype=str), "y": np.asarray(y_test, dtype=np.float64)}).to_csv(
        y_test_tsv, sep="\t", index=False
    )
    meta = {
        "method": str(method),
        "x_train_bin": str(x_train_bin),
        "x_test_bin": str(x_test_bin),
        "y_train_tsv": str(y_train_tsv),
        "y_test_tsv": str(y_test_tsv),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "p": int(x_train.shape[1]),
        "n_iter": int(n_iter),
        "burnin": int(burnin),
        "thin": int(thin),
        "r2": float(r2),
        "prob_in": float(prob_in),
        "counts": float(counts),
        "df0_b": float(df0_b),
        "shape0": float(shape0),
        "rate0": (None if rate0 is None else float(rate0)),
        "s0_b": (None if s0_b is None else float(s0_b)),
        "df0_e": float(df0_e),
        "s0_e": (None if s0_e is None else float(s0_e)),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_bglr_compare_script(script_path)
    rc, elapsed, rss = _run_timed(
        [
            str(rscript_bin),
            str(script_path),
            "--meta",
            str(meta_json),
            "--out",
            str(out_json),
            "--beta-out",
            str(beta_tsv),
            "--pred-out",
            str(pred_tsv),
        ],
        log_file=log_file,
        time_file=time_file,
        env=_compare_subproc_env(threads),
    )
    if rc != 0:
        raise RuntimeError(
            "BGLR compare run failed.\n"
            f"log:\n{log_file.read_text(encoding='utf-8', errors='ignore')}"
        )
    result = json.loads(out_json.read_text(encoding="utf-8"))
    beta_df = pd.read_csv(beta_tsv, sep="\t")
    pred_df = pd.read_csv(pred_tsv, sep="\t")
    result["beta_df"] = beta_df
    result["pred_df"] = pred_df
    result["elapsed_sec"] = (
        float(elapsed)
        if np.isfinite(elapsed)
        else _safe_float(result.get("elapsed_sec", np.nan))
    )
    result["peak_rss_kb"] = (
        float(rss)
        if np.isfinite(rss)
        else _safe_float(result.get("peak_rss_kb", np.nan))
    )
    result["log_file"] = str(log_file)
    result["time_file"] = str(time_file)
    return result


def _write_hibayes_compare_script(path: Path) -> None:
    code = r"""#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
arg_val <- function(name, default = "") {
  idx <- which(args == name)
  if (length(idx) == 0) return(default)
  j <- idx[1] + 1
  if (j > length(args)) return(default)
  args[j]
}

meta_path <- arg_val("--meta", "")
out_path <- arg_val("--out", "")
beta_out <- arg_val("--beta-out", "")
pred_out <- arg_val("--pred-out", "")
if (meta_path == "" || out_path == "" || beta_out == "" || pred_out == "") {
  stop("missing --meta/--out/--beta-out/--pred-out")
}

suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(hibayes))

meta <- fromJSON(meta_path)
approx_peak_kb <- function() {
  g <- gc()
  ncells <- as.numeric(g[1, "max used"])
  vcells <- as.numeric(g[2, "max used"])
  ((ncells * 56.0) + (vcells * 8.0)) / 1024.0
}
meta_num <- function(name) {
  x <- meta[[name]]
  if (is.null(x) || length(x) == 0) return(NA_real_)
  as.numeric(x)[1]
}

read_matrix_f32 <- function(path, nrow, ncol) {
  con <- file(path, "rb")
  on.exit(try(close(con), silent = TRUE), add = TRUE)
  vec <- readBin(con, what = "numeric", n = as.integer(nrow) * as.integer(ncol), size = 4, endian = "little")
  if (length(vec) != (as.integer(nrow) * as.integer(ncol))) {
    stop(sprintf("Matrix size mismatch for %s: got %d expected %d", path, length(vec), as.integer(nrow) * as.integer(ncol)))
  }
  matrix(vec, nrow = as.integer(nrow), ncol = as.integer(ncol), byrow = TRUE)
}

method_in <- as.character(meta$method)
model_ref <- if (tolower(method_in) == "bayesb") "BayesBpi" else method_in
Xtr <- read_matrix_f32(meta$x_train_bin, meta$n_train, meta$p)
Xte <- read_matrix_f32(meta$x_test_bin, meta$n_test, meta$p)
ytr_df <- read.table(meta$y_train_tsv, header = TRUE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
yte_df <- read.table(meta$y_test_tsv, header = TRUE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
ytr <- as.numeric(ytr_df$y)
yte <- as.numeric(yte_df$y)
dat <- data.frame(sample_id = as.character(ytr_df$sample_id), y = ytr, stringsAsFactors = FALSE)

fit_args <- list(
  formula = y ~ 1,
  data = dat,
  M = Xtr,
  M.id = as.character(dat$sample_id),
  method = model_ref,
  niter = as.integer(meta$n_iter),
  nburn = as.integer(meta$burnin),
  thin = as.integer(meta$thin),
  threads = as.integer(meta$threads),
  verbose = FALSE,
  seed = as.integer(meta$seed)
)

if (model_ref %in% c("BayesB", "BayesC", "BayesCpi", "BayesBpi")) {
  p_in <- meta_num("prob_in")
  if (is.finite(p_in) && p_in > 0 && p_in < 1) {
    fit_args$Pi <- c(1.0 - p_in, p_in)
  }
}

t0 <- proc.time()[["elapsed"]]
fit <- do.call(hibayes::ibrm, fit_args)
elapsed_sec <- proc.time()[["elapsed"]] - t0
alpha <- as.numeric(fit$alpha)
mu <- as.numeric(fit$mu)[1]
vare <- as.numeric(fit$Ve)[1]
h2_native <- as.numeric(fit$h2)[1]
pip <- fit$pip
if (is.null(pip)) pip <- rep(NA_real_, length(alpha))
pip <- as.numeric(pip)
pi_vec <- fit$pi
if (is.null(pi_vec)) pi_vec <- rep(NA_real_, 0)
pi_vec <- as.numeric(pi_vec)
prob_in_mean <- if (length(pi_vec) >= 2) sum(pi_vec[-1], na.rm = TRUE) else NA_real_
n_active_mean <- if (all(is.na(pip))) {
  if (model_ref == "BayesA") NA_real_ else sum(abs(alpha) > 1e-12)
} else {
  sum(pip, na.rm = TRUE)
}
if (model_ref == "BayesA") {
  pip[] <- NA_real_
  prob_in_mean <- NA_real_
  n_active_mean <- NA_real_
}

pred_tr <- as.numeric(mu + Xtr %*% alpha)
pred_te <- as.numeric(mu + Xte %*% alpha)
gebv_tr <- as.numeric(Xtr %*% alpha)
gebv_te <- as.numeric(Xte %*% alpha)
var_g <- stats::var(gebv_tr)
h2 <- if (is.finite(var_g) && is.finite(vare) && (var_g + vare) > 0) var_g / (var_g + vare) else NA_real_

beta_df <- data.frame(
  snp_index = seq_along(alpha) - 1L,
  beta = alpha,
  pip = pip,
  stringsAsFactors = FALSE
)
pred_df <- rbind(
  data.frame(
    split = "train",
    sample_id = as.character(ytr_df$sample_id),
    y_true = ytr,
    y_pred = pred_tr,
    gebv = gebv_tr,
    intercept = rep(mu, length(pred_tr)),
    stringsAsFactors = FALSE
  ),
  data.frame(
    split = "test",
    sample_id = as.character(yte_df$sample_id),
    y_true = yte,
    y_pred = pred_te,
    gebv = gebv_te,
    intercept = rep(mu, length(pred_te)),
    stringsAsFactors = FALSE
  )
)

dir.create(dirname(beta_out), recursive = TRUE, showWarnings = FALSE)
write.table(beta_df, file = beta_out, sep = "\t", quote = FALSE, row.names = FALSE)
dir.create(dirname(pred_out), recursive = TRUE, showWarnings = FALSE)
write.table(pred_df, file = pred_out, sep = "\t", quote = FALSE, row.names = FALSE)

result <- list(
  status = "ok",
  engine = "HIBayes",
  method_input = method_in,
  model_reference = model_ref,
  vare = vare,
  h2 = h2,
  h2_native = h2_native,
  h2_semantics = "train_var_posterior_mean_gebv_over_varg_plus_vare",
  prob_in_mean = prob_in_mean,
  n_active_mean = n_active_mean,
  intercept = mu,
  beta_out = beta_out,
  pred_out = pred_out,
  elapsed_sec = elapsed_sec,
  peak_rss_kb = approx_peak_kb()
)
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
writeLines(jsonlite::toJSON(result, auto_unbox = TRUE, pretty = TRUE), con = out_path)
"""
    path.write_text(code, encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass


def _run_janusx_reference(
    *,
    work_dir: Path,
    method: str,
    packed_ctx: dict[str, Any],
    row_mean: np.ndarray,
    row_inv_sd: np.ndarray,
    train_abs: np.ndarray,
    test_abs: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_iter: int,
    burnin: int,
    thin: int,
    r2: float,
    prob_in: float,
    counts: float,
    df0_b: float,
    shape0: float,
    rate0: float | None,
    s0_b: float | None,
    df0_e: float,
    s0_e: float | None,
    threads: int,
    seed: int,
    snp_block_size: int,
    sample_chunk_size: int,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    packed_npy = work_dir / "packed.npy"
    maf_npy = work_dir / "maf.npy"
    row_flip_npy = work_dir / "row_flip.npy"
    row_mean_npy = work_dir / "row_mean.npy"
    row_inv_sd_npy = work_dir / "row_inv_sd.npy"
    train_abs_npy = work_dir / "train_abs.npy"
    test_abs_npy = work_dir / "test_abs.npy"
    train_ids_npy = work_dir / "train_ids.npy"
    test_ids_npy = work_dir / "test_ids.npy"
    y_train_npy = work_dir / "y_train.npy"
    y_test_npy = work_dir / "y_test.npy"
    meta_json = work_dir / "meta.json"
    out_json = work_dir / "result.json"
    beta_tsv = work_dir / "beta.tsv"
    pred_tsv = work_dir / "pred.tsv"
    log_file = work_dir / "run.log"
    time_file = work_dir / "run.time"
    script_path = work_dir / "run_janusx_compare.py"

    np.save(packed_npy, np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8), dtype=np.uint8), allow_pickle=False)
    np.save(maf_npy, np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1), dtype=np.float32), allow_pickle=False)
    np.save(row_flip_npy, np.ascontiguousarray(np.asarray(packed_ctx["row_flip"], dtype=np.bool_).reshape(-1), dtype=np.bool_), allow_pickle=False)
    np.save(row_mean_npy, np.ascontiguousarray(np.asarray(row_mean, dtype=np.float32).reshape(-1), dtype=np.float32), allow_pickle=False)
    np.save(row_inv_sd_npy, np.ascontiguousarray(np.asarray(row_inv_sd, dtype=np.float32).reshape(-1), dtype=np.float32), allow_pickle=False)
    np.save(train_abs_npy, np.ascontiguousarray(np.asarray(train_abs, dtype=np.int64).reshape(-1), dtype=np.int64), allow_pickle=False)
    np.save(test_abs_npy, np.ascontiguousarray(np.asarray(test_abs, dtype=np.int64).reshape(-1), dtype=np.int64), allow_pickle=False)
    np.save(train_ids_npy, np.asarray(train_ids, dtype=str), allow_pickle=False)
    np.save(test_ids_npy, np.asarray(test_ids, dtype=str), allow_pickle=False)
    np.save(y_train_npy, np.ascontiguousarray(np.asarray(y_train, dtype=np.float64).reshape(-1), dtype=np.float64), allow_pickle=False)
    np.save(y_test_npy, np.ascontiguousarray(np.asarray(y_test, dtype=np.float64).reshape(-1), dtype=np.float64), allow_pickle=False)

    meta = {
        "method": str(method),
        "packed_npy": str(packed_npy),
        "maf_npy": str(maf_npy),
        "row_flip_npy": str(row_flip_npy),
        "row_mean_npy": str(row_mean_npy),
        "row_inv_sd_npy": str(row_inv_sd_npy),
        "train_abs_npy": str(train_abs_npy),
        "test_abs_npy": str(test_abs_npy),
        "train_ids_npy": str(train_ids_npy),
        "test_ids_npy": str(test_ids_npy),
        "y_train_npy": str(y_train_npy),
        "y_test_npy": str(y_test_npy),
        "n_samples": int(packed_ctx["n_samples"]),
        "n_iter": int(n_iter),
        "burnin": int(burnin),
        "thin": int(thin),
        "r2": float(r2),
        "prob_in": float(prob_in),
        "counts": float(counts),
        "df0_b": float(df0_b),
        "shape0": float(shape0),
        "rate0": (None if rate0 is None else float(rate0)),
        "s0_b": (None if s0_b is None else float(s0_b)),
        "df0_e": float(df0_e),
        "s0_e": (None if s0_e is None else float(s0_e)),
        "threads": int(max(0, int(threads))),
        "seed": int(seed),
        "snp_block_size": int(max(1, int(snp_block_size))),
        "sample_chunk_size": int(max(1, int(sample_chunk_size))),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_janusx_compare_script(script_path)
    rc, elapsed, rss = _run_timed(
        [str(sys.executable), str(script_path), "--meta", str(meta_json), "--out", str(out_json), "--beta-out", str(beta_tsv), "--pred-out", str(pred_tsv)],
        log_file=log_file,
        time_file=time_file,
        env=_compare_subproc_env(threads),
    )
    if rc != 0:
        raise RuntimeError(
            "JanusX compare run failed.\n"
            f"log:\n{log_file.read_text(encoding='utf-8', errors='ignore')}"
        )
    result = json.loads(out_json.read_text(encoding="utf-8"))
    result["beta_df"] = pd.read_csv(beta_tsv, sep="\t")
    result["pred_df"] = pd.read_csv(pred_tsv, sep="\t")
    result["elapsed_sec"] = (
        float(elapsed)
        if np.isfinite(elapsed)
        else _safe_float(result.get("elapsed_sec", np.nan))
    )
    result["peak_rss_kb"] = (
        float(rss)
        if np.isfinite(rss)
        else _safe_float(result.get("peak_rss_kb", np.nan))
    )
    result["log_file"] = str(log_file)
    result["time_file"] = str(time_file)
    return result


def _run_hibayes_reference(
    *,
    work_dir: Path,
    method: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    n_iter: int,
    burnin: int,
    thin: int,
    prob_in: float,
    threads: int,
    seed: int,
    rscript_bin: str,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    x_train_bin = work_dir / "x_train.bin"
    x_test_bin = work_dir / "x_test.bin"
    y_train_tsv = work_dir / "y_train.tsv"
    y_test_tsv = work_dir / "y_test.tsv"
    meta_json = work_dir / "meta.json"
    out_json = work_dir / "result.json"
    beta_tsv = work_dir / "beta.tsv"
    pred_tsv = work_dir / "pred.tsv"
    log_file = work_dir / "run.log"
    time_file = work_dir / "run.time"
    script_path = work_dir / "run_hibayes_compare.R"

    _write_float32_matrix(x_train_bin, x_train)
    _write_float32_matrix(x_test_bin, x_test)
    pd.DataFrame({"sample_id": np.asarray(train_ids, dtype=str), "y": np.asarray(y_train, dtype=np.float64)}).to_csv(
        y_train_tsv, sep="\t", index=False
    )
    pd.DataFrame({"sample_id": np.asarray(test_ids, dtype=str), "y": np.asarray(y_test, dtype=np.float64)}).to_csv(
        y_test_tsv, sep="\t", index=False
    )
    meta = {
        "method": str(method),
        "x_train_bin": str(x_train_bin),
        "x_test_bin": str(x_test_bin),
        "y_train_tsv": str(y_train_tsv),
        "y_test_tsv": str(y_test_tsv),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "p": int(x_train.shape[1]),
        "n_iter": int(n_iter),
        "burnin": int(burnin),
        "thin": int(thin),
        "prob_in": float(prob_in),
        "threads": int(max(1, int(threads))),
        "seed": int(seed),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_hibayes_compare_script(script_path)
    rc, elapsed, rss = _run_timed(
        [str(rscript_bin), str(script_path), "--meta", str(meta_json), "--out", str(out_json), "--beta-out", str(beta_tsv), "--pred-out", str(pred_tsv)],
        log_file=log_file,
        time_file=time_file,
        env=_compare_subproc_env(threads),
    )
    if rc != 0:
        raise RuntimeError(
            "HIBayes compare run failed.\n"
            f"log:\n{log_file.read_text(encoding='utf-8', errors='ignore')}"
        )
    result = json.loads(out_json.read_text(encoding="utf-8"))
    result["beta_df"] = pd.read_csv(beta_tsv, sep="\t")
    result["pred_df"] = pd.read_csv(pred_tsv, sep="\t")
    result["elapsed_sec"] = (
        float(elapsed)
        if np.isfinite(elapsed)
        else _safe_float(result.get("elapsed_sec", np.nan))
    )
    result["peak_rss_kb"] = (
        float(rss)
        if np.isfinite(rss)
        else _safe_float(result.get("peak_rss_kb", np.nan))
    )
    result["log_file"] = str(log_file)
    result["time_file"] = str(time_file)
    return result


def _engine_summary_row(
    *,
    method: str,
    trait: str,
    software: str,
    result: dict[str, Any],
    n_train: int,
    n_test: int,
    m: int,
) -> dict[str, Any]:
    return {
        "method": str(method),
        "trait": str(trait),
        "software": str(software),
        "model_reference": str(result.get("model_reference", result.get("method_input", method))),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "m": int(m),
        "h2_compare": _safe_float(result.get("h2", np.nan)),
        "h2_native": _safe_float(result.get("h2_native", np.nan)),
        "h2_semantics": str(result.get("h2_semantics", "")),
        "vare": _safe_float(result.get("vare", np.nan)),
        "prob_in_mean": _safe_float(result.get("prob_in_mean", np.nan)),
        "n_active_mean": _safe_float(result.get("n_active_mean", np.nan)),
        "elapsed_sec": _safe_float(result.get("elapsed_sec", np.nan)),
        "peak_rss_kb": _safe_float(result.get("peak_rss_kb", np.nan)),
        "log_file": str(result.get("log_file", "")),
        "time_file": str(result.get("time_file", "")),
    }


def _pairwise_compare_row(
    *,
    method: str,
    trait: str,
    software_a: str,
    software_b: str,
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    beta_df_a: pd.DataFrame,
    beta_df_b: pd.DataFrame,
    pred_df_a: pd.DataFrame,
    pred_df_b: pd.DataFrame,
    top_cutoffs: list[int],
) -> dict[str, Any]:
    beta_a = pd.DataFrame(beta_df_a)[["snp_index", "beta"]].copy()
    beta_b = pd.DataFrame(beta_df_b)[["snp_index", "beta"]].copy()
    beta_a["snp_index"] = pd.to_numeric(beta_a["snp_index"], errors="coerce").astype("Int64")
    beta_b["snp_index"] = pd.to_numeric(beta_b["snp_index"], errors="coerce").astype("Int64")
    beta_a["beta"] = pd.to_numeric(beta_a["beta"], errors="coerce")
    beta_b["beta"] = pd.to_numeric(beta_b["beta"], errors="coerce")
    beta_join = beta_a.merge(beta_b, on="snp_index", how="inner", suffixes=("_a", "_b")).dropna()
    beta_vec_a = np.asarray(beta_join["beta_a"], dtype=np.float64)
    beta_vec_b = np.asarray(beta_join["beta_b"], dtype=np.float64)

    pred_a = pd.DataFrame(pred_df_a)[["split", "sample_id", "y_true", "y_pred", "gebv"]].copy()
    pred_b = pd.DataFrame(pred_df_b)[["split", "sample_id", "y_pred", "gebv"]].copy()
    pred_join = pred_a.merge(pred_b, on=["split", "sample_id"], how="inner", suffixes=("_a", "_b")).dropna()

    def _pred_slice(split: str | None) -> pd.DataFrame:
        if split is None:
            return pred_join
        return pred_join.loc[pred_join["split"].astype(str) == str(split)].copy()

    def _gebv_stats(df: pd.DataFrame) -> tuple[float, float, float, float, float]:
        ga = np.asarray(df["gebv_a"], dtype=np.float64)
        gb = np.asarray(df["gebv_b"], dtype=np.float64)
        delta = ga - gb
        q95 = float(np.nanquantile(np.abs(delta), 0.95)) if delta.size > 0 else float("nan")
        return (
            _pearson(ga, gb),
            _mae(ga, gb),
            _rmse(ga, gb),
            q95,
            float(np.nanmax(np.abs(delta))) if delta.size > 0 else float("nan"),
        )

    row: dict[str, Any] = {
        "method": str(method),
        "trait": str(trait),
        "software_a": str(software_a),
        "software_b": str(software_b),
        "h2_compare_a": _safe_float(result_a.get("h2", np.nan)),
        "h2_compare_b": _safe_float(result_b.get("h2", np.nan)),
        "h2_native_a": _safe_float(result_a.get("h2_native", np.nan)),
        "h2_native_b": _safe_float(result_b.get("h2_native", np.nan)),
        "h2_semantics_a": str(result_a.get("h2_semantics", "")),
        "h2_semantics_b": str(result_b.get("h2_semantics", "")),
        "m_overlap": int(beta_join.shape[0]),
        "n_pred_overlap": int(pred_join.shape[0]),
        "beta_corr_all": _pearson(beta_vec_a, beta_vec_b),
        "beta_spearman_all": _spearman(beta_vec_a, beta_vec_b),
        "beta_mae_all": _mae(beta_vec_a, beta_vec_b),
        "beta_rmse_all": _rmse(beta_vec_a, beta_vec_b),
        "beta_slope_ab": _slope(beta_vec_a, beta_vec_b),
    }
    rank_idx = np.argsort(-np.abs(beta_vec_a))
    for cutoff in top_cutoffs:
        k = int(min(max(1, int(cutoff)), int(beta_vec_a.shape[0]) if beta_vec_a.ndim == 1 else 0))
        if k <= 0:
            row[f"beta_corr_top{int(cutoff)}"] = float("nan")
            row[f"beta_mae_top{int(cutoff)}"] = float("nan")
            row[f"beta_rmse_top{int(cutoff)}"] = float("nan")
            continue
        idx = rank_idx[:k]
        row[f"beta_corr_top{k}"] = _pearson(beta_vec_a[idx], beta_vec_b[idx])
        row[f"beta_mae_top{k}"] = _mae(beta_vec_a[idx], beta_vec_b[idx])
        row[f"beta_rmse_top{k}"] = _rmse(beta_vec_a[idx], beta_vec_b[idx])

    for split_name, split_key in (("train", "train"), ("test", "test"), ("all", None)):
        sub = _pred_slice(split_key)
        corr, mae, rmse, q95, maxabs = _gebv_stats(sub)
        row[f"gebv_corr_{split_name}"] = corr
        row[f"delta_gebv_mae_{split_name}"] = mae
        row[f"delta_gebv_rmse_{split_name}"] = rmse
        row[f"delta_gebv_abs_q95_{split_name}"] = q95
        row[f"delta_gebv_abs_max_{split_name}"] = maxabs
        row[f"pred_corr_{split_name}_a"] = _pearson(
            np.asarray(sub["y_true"], dtype=np.float64),
            np.asarray(sub["y_pred_a"], dtype=np.float64),
        )
        row[f"pred_corr_{split_name}_b"] = _pearson(
            np.asarray(sub["y_true"], dtype=np.float64),
            np.asarray(sub["y_pred_b"], dtype=np.float64),
        )
        row[f"pred_rmse_{split_name}_a"] = _rmse(
            np.asarray(sub["y_true"], dtype=np.float64),
            np.asarray(sub["y_pred_a"], dtype=np.float64),
        )
        row[f"pred_rmse_{split_name}_b"] = _rmse(
            np.asarray(sub["y_true"], dtype=np.float64),
            np.asarray(sub["y_pred_b"], dtype=np.float64),
        )
    return row


def _build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx bayesbench",
        description="Bayesian GS benchmark: kernel microbench, JanusX convergence, and JanusX vs BGLR/HIBayes comparison.",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx bayesbench --n-samples 1200 --n-snps 10000 --repeat 3",
                "jx bayesbench convergence -bfile example/~mouse_hs1940.snp0 -p example/mouse_hs1940.pheno -n 0 -o outdir",
                "jx bayesbench compare -bfile example/~mouse_hs1940.snp0 -p example/mouse_hs1940.pheno -n 0 --methods BayesA -o outdir",
            ]
        ),
    )
    sub = parser.add_subparsers(dest="mode")

    p_kernel = sub.add_parser(
        "kernel",
        help="Synthetic packed-Bayes kernel microbenchmark.",
        formatter_class=cli_help_formatter(),
    )
    p_kernel.add_argument("--methods", default="BayesA,BayesB,BayesCpi", help="Comma list from BayesA,BayesB,BayesCpi.")
    p_kernel.add_argument("--n-samples", type=int, default=1200, help="Total sample size.")
    p_kernel.add_argument("--n-snps", type=int, default=10000, help="Total SNP rows.")
    p_kernel.add_argument("--train-size", type=int, default=1000, help="Training sample size.")
    p_kernel.add_argument("--n-iter", type=int, default=120, help="MCMC iterations.")
    p_kernel.add_argument("--burnin", type=int, default=40, help="Burn-in iterations.")
    p_kernel.add_argument("--thin", type=int, default=2, help="Thinning step.")
    p_kernel.add_argument("--repeat", type=int, default=3, help="Repeat count per setting.")
    p_kernel.add_argument("--seed", type=int, default=20260429, help="Random seed for simulation and kernel.")
    p_kernel.add_argument("--r2", type=float, default=0.6, help="Fixed R2 for kernel benchmark.")
    p_kernel.add_argument("--counts", type=float, default=5.0, help="counts hyperparameter for BayesB/BayesCpi.")
    p_kernel.add_argument(
        "--row-block",
        default="0,256",
        help="Comma list for JX_BAYES_PACKED_ROW_BLOCK; <=0 means auto.",
    )
    p_kernel.add_argument("--out", default=None, help="Optional output TSV path for benchmark summary.")

    for name, help_text in (
        ("convergence", "Run multi-chain JanusX Bayes convergence / stability summary."),
        ("compare", "Run JanusX vs BGLR/HIBayes long-chain posterior / prediction comparison."),
    ):
        sp = sub.add_parser(name, help=help_text, formatter_class=cli_help_formatter())
        geno = sp.add_argument_group("Genotype Input")
        geno.add_argument("-bfile", "--bfile", type=str, help="PLINK prefix (.bed/.bim/.fam).")
        geno.add_argument("-vcf", "--vcf", type=str, help="VCF/VCF.GZ genotype input.")
        geno.add_argument("-hmp", "--hmp", type=str, help="Hapmap genotype input.")
        geno.add_argument("-file", "--file", type=str, help="Text genotype input.")
        sp.add_argument("--builtin", choices=["wheat"], default=None, help="Use built-in BGLR benchmark dataset instead of genotype/pheno files.")
        sp.add_argument("-p", "--pheno", required=False, type=str, help="Phenotype table. Not required for `--builtin wheat`.")
        sp.add_argument("-n", "--ncol", type=str, default=None, help="One zero-based phenotype column index.")
        sp.add_argument("--trait", type=str, default=None, help="Trait column name in phenotype file.")
        sp.add_argument("--methods", default="BayesA,BayesB,BayesCpi", help="Comma list from BayesA,BayesB,BayesCpi.")
        sp.add_argument("--n-iter", type=int, default=50000, help="Total MCMC iterations.")
        sp.add_argument("--burnin", type=int, default=10000, help="Burn-in iterations.")
        sp.add_argument("--thin", type=int, default=10, help="Keep every thin-th sample.")
        sp.add_argument("--seed", type=int, default=20260429, help="Base random seed.")
        sp.add_argument("--r2", type=float, default=0.5, help="Fixed R2 prior.")
        sp.add_argument("--prob-in", type=float, default=0.5, help="Initial / prior inclusion probability.")
        sp.add_argument("--counts", type=float, default=5.0, help="Prior counts for inclusion probability.")
        sp.add_argument("--df0-b", type=float, default=5.0, help="Marker-effect prior df0.")
        sp.add_argument("--shape0", type=float, default=1.1, help="BayesA/B shape0.")
        sp.add_argument("--rate0", type=float, default=None, help="Optional BayesA/B rate0.")
        sp.add_argument("--s0-b", type=float, default=None, help="Optional marker prior scale S0_b.")
        sp.add_argument("--df0-e", type=float, default=5.0, help="Residual prior df0.")
        sp.add_argument("--s0-e", type=float, default=None, help="Optional residual prior scale S0_e.")
        sp.add_argument("-maf", "--maf", type=float, default=0.02, help="MAF threshold for packed filtering.")
        sp.add_argument("-geno", "--geno", type=float, default=0.05, help="Missing-rate threshold for packed filtering.")
        sp.add_argument("--snps-only", action="store_true", default=False, help="Restrict to SNP sites when caching text/VCF input.")
        sp.add_argument("--cache-input", action="store_true", default=True, help=argparse.SUPPRESS)
        sp.add_argument("--max-snps", type=int, default=None, help="Optional random cap on active SNPs after QC (0 disables).")
        sp.add_argument("--rscript", type=str, default=(shutil.which("Rscript") or "Rscript"), help="Rscript executable. Required for `--builtin wheat` and BGLR compare.")
        sp.add_argument(
            "-t",
            "--thread",
            type=int,
            default=0,
            help=(
                "Thread count backdoor. JanusX uses it for packed kernels; compare-mode also pins "
                "HIBayes/BGLR BLAS/OpenMP threads to the same value. Use -t 1 for fair single-core "
                "cross-software benchmarks; use -t N for multicore experiments."
            ),
        )
        sp.add_argument("--snp-block-size", type=int, default=2048, help="Prediction SNP block size for JanusX packed effects.")
        sp.add_argument("--sample-chunk-size", type=int, default=4096, help="Prediction sample chunk size for JanusX packed effects.")
        sp.add_argument("-o", "--out", required=True, type=str, help="Output directory.")

    p_conv = sub.choices["convergence"]
    p_conv.set_defaults(max_snps=0)
    p_conv.add_argument("--chains", type=int, default=4, help="Number of independent JanusX chains.")
    p_conv.add_argument("--chain-seeds", type=str, default=None, help="Comma-separated explicit chain seeds.")
    p_conv.add_argument("--top-k-beta", type=int, default=20, help="Consensus top-k posterior mean beta rows to report.")
    p_conv.add_argument(
        "--parallel-chains",
        type=int,
        default=0,
        help="How many JanusX chains to run concurrently (0=auto).",
    )
    p_conv.add_argument("--rhat-threshold", type=float, default=1.05, help="R-hat threshold used to declare global-parameter stability.")
    p_conv.add_argument("--stable-min-kept", type=int, default=100, help="Minimum kept posterior samples per chain before stability can be declared.")
    p_conv.add_argument("--plot-top-k-beta", type=int, default=12, help="How many top SNP beta traces to include in the trace figure.")
    p_conv.add_argument("--global-only", action="store_true", default=False, help="Run a single trace pass for global parameters only; skip the second beta-trace rerun.")

    p_cmp = sub.choices["compare"]
    p_cmp.set_defaults(max_snps=5000)
    p_cmp.add_argument("--reference", type=str, default="all", choices=sorted(_REF_ENGINES), help="Reference implementation(s): bglr, hibayes, or all.")
    p_cmp.add_argument("--test-frac", type=float, default=0.2, help="Held-out test fraction for prediction concordance.")
    p_cmp.add_argument("--split-seed", type=int, default=20260429, help="Train/test split seed.")
    p_cmp.add_argument("--top-beta-cutoffs", type=str, default="100,1000", help="Comma list of top-|beta| cutoffs for concordance summaries.")
    return parser


def _validate_real_args(args: argparse.Namespace) -> None:
    builtin = str(getattr(args, "builtin", None) or "").strip().lower()
    n_sources = sum(bool(getattr(args, key, None)) for key in ("bfile", "vcf", "hmp", "file"))
    if builtin:
        if builtin != "wheat":
            raise ValueError(f"Unsupported builtin dataset: {builtin!r}")
        _resolve_rscript_bin(args)
    else:
        if n_sources != 1:
            raise ValueError("Provide exactly one genotype input: -bfile / -vcf / -hmp / -file.")
        if getattr(args, "pheno", None) in {None, ""}:
            raise ValueError("`-p/--pheno` is required unless `--builtin wheat` is used.")
    if int(args.n_iter) <= int(args.burnin):
        raise ValueError("--n-iter must be > --burnin.")
    if int(args.thin) <= 0:
        raise ValueError("--thin must be >= 1.")
    if not (0.0 < float(args.r2) < 1.0):
        raise ValueError("--r2 must be in (0, 1).")
    if not (0.0 < float(args.prob_in) < 1.0):
        raise ValueError("--prob-in must be in (0, 1).")
    if float(args.counts) < 0.0:
        raise ValueError("--counts must be >= 0.")


def run_kernel(args: argparse.Namespace) -> int:
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
                    f"mean={np.mean(arr):.4f}s min={np.min(arr):.4f}s std={np.std(arr):.4f}s"
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


def run_convergence(args: argparse.Namespace) -> int:
    _validate_real_args(args)
    out_dir = Path(str(args.out)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_real_dataset(args)
    methods = _parse_methods(str(args.methods))
    chain_seeds = _parse_seed_list(
        raw=getattr(args, "chain_seeds", None),
        chains=int(args.chains),
        base_seed=int(args.seed),
    )
    rows: list[dict[str, Any]] = []
    beta_by_method: dict[str, list[np.ndarray]] = {}
    beta_seed_by_method: dict[str, list[int]] = {}
    y_all = np.asarray(data["y_all"], dtype=np.float64)
    sample_idx = np.asarray(data["overlap_abs_idx"], dtype=np.int64)
    packed_ctx = data["packed_ctx"]
    row_mean = data["row_mean"]
    row_inv_sd = data["row_inv_sd"]
    n_active_before_cap = int(data.get("n_active_before_cap", np.asarray(packed_ctx["packed"]).shape[0]))
    n_active_after_cap = int(data.get("n_active_after_cap", np.asarray(packed_ctx["packed"]).shape[0]))

    hw_threads = max(1, int(os.cpu_count() or 1))
    requested_parallel = int(max(0, int(getattr(args, "parallel_chains", 0))))
    requested_chain_threads = int(getattr(args, "thread", 0))
    if requested_parallel > 0:
        parallel_chains = int(min(len(chain_seeds), requested_parallel))
    else:
        if requested_chain_threads > 0:
            parallel_chains = max(1, min(len(chain_seeds), hw_threads // max(1, requested_chain_threads)))
        else:
            parallel_chains = max(1, min(len(chain_seeds), hw_threads))
    effective_chain_threads = int(requested_chain_threads)
    if effective_chain_threads <= 0 and parallel_chains > 1:
        # Multiple independent chains generally scale better than letting each
        # chain auto-grab the whole machine.
        effective_chain_threads = 1

    total_fits = int(len(methods) * len(chain_seeds))
    print(
        f"[bayesbench] convergence trait={data['trait_name']} n={int(y_all.shape[0])} "
        f"m={int(np.asarray(packed_ctx['packed']).shape[0])} chains={int(args.chains)} "
        f"methods={','.join(methods)} total_fits={total_fits}"
    )
    if n_active_before_cap != n_active_after_cap:
        print(
            f"[bayesbench] active SNPs after QC={n_active_before_cap}, "
            f"capped to max_snps={n_active_after_cap}"
        )
    else:
        print(f"[bayesbench] active SNPs after QC={n_active_after_cap} (no cap)")
    print(
        f"[bayesbench] chain execution parallel_chains={parallel_chains} "
        f"threads_per_chain={effective_chain_threads if effective_chain_threads > 0 else 'auto'}"
    )
    top_k = int(max(0, int(args.top_k_beta)))
    if bool(getattr(args, "global_only", False)):
        print("[bayesbench] convergence mode=global-only (single trace pass)")
        rows = []
        pair_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        top_rows: list[dict[str, Any]] = []
        trace_rows: list[dict[str, Any]] = []
        global_trace_rows: list[dict[str, Any]] = []
        beta_trace_rows: list[dict[str, Any]] = []
        for method in methods:
            method_trace_rows: list[dict[str, Any]] = []
            empty_trace_idx = np.empty((0,), dtype=np.int64)

            def _run_one_trace_chain(seed: int) -> dict[str, Any]:
                t0 = time.perf_counter()
                fit = _fit_janusx_packed_trace(
                    method=method,
                    packed_ctx=packed_ctx,
                    sample_indices=sample_idx,
                    y=y_all,
                    row_mean=row_mean,
                    row_inv_sd=row_inv_sd,
                    trace_snp_indices=empty_trace_idx,
                    n_iter=int(args.n_iter),
                    burnin=int(args.burnin),
                    thin=int(args.thin),
                    r2=float(args.r2),
                    prob_in=float(args.prob_in),
                    counts=float(args.counts),
                    df0_b=float(args.df0_b),
                    shape0=float(args.shape0),
                    rate0=(None if args.rate0 is None else float(args.rate0)),
                    s0_b=(None if args.s0_b is None else float(args.s0_b)),
                    df0_e=float(args.df0_e),
                    s0_e=(None if args.s0_e is None else float(args.s0_e)),
                    threads=int(effective_chain_threads),
                    seed=int(seed),
                )
                elapsed = float(time.perf_counter() - t0)
                return {
                    "method": method,
                    "trait": str(data["trait_name"]),
                    "chain_seed": int(seed),
                    "n": int(y_all.shape[0]),
                    "m": int(np.asarray(fit["beta"], dtype=np.float64).shape[0]),
                    "h2_mean": float(fit["h2_mean"]),
                    "vare": float(fit["vare"]),
                    "prob_in_mean": _safe_float(fit["prob_in_mean"]),
                    "n_active_mean": _safe_float(fit["n_active_mean"]),
                    "elapsed_sec": elapsed,
                    "beta": np.asarray(fit["beta"], dtype=np.float64),
                    "iter_trace": np.asarray(fit["iter_trace"], dtype=np.int64),
                    "h2_trace": np.asarray(fit["h2_trace"], dtype=np.float64),
                    "var_e_trace": np.asarray(fit["var_e_trace"], dtype=np.float64),
                    "prob_in_trace": np.asarray(fit["prob_in_trace"], dtype=np.float64),
                    "n_active_trace": np.asarray(fit["n_active_trace"], dtype=np.float64),
                    "full_iter_trace": np.asarray(fit["full_iter_trace"], dtype=np.int64),
                    "full_h2_trace": np.asarray(fit["full_h2_trace"], dtype=np.float64),
                    "full_var_e_trace": np.asarray(fit["full_var_e_trace"], dtype=np.float64),
                    "full_prob_in_trace": np.asarray(fit["full_prob_in_trace"], dtype=np.float64),
                    "full_n_active_trace": np.asarray(fit["full_n_active_trace"], dtype=np.float64),
                }

            if parallel_chains <= 1:
                for seed in chain_seeds:
                    one = _run_one_trace_chain(int(seed))
                    method_trace_rows.append(one)
                    print(
                        f"[bayesbench] global-only {method:8s} seed={int(seed)} "
                        f"kept={int(one['iter_trace'].shape[0])} h2={float(one['h2_mean']):.4f} "
                        f"vare={float(one['vare']):.4f} time={float(one['elapsed_sec']):.2f}s"
                    )
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=int(parallel_chains)) as ex:
                    future_map = {
                        ex.submit(_run_one_trace_chain, int(seed)): int(seed)
                        for seed in chain_seeds
                    }
                    for fut in concurrent.futures.as_completed(future_map):
                        one = fut.result()
                        method_trace_rows.append(one)
                        print(
                            f"[bayesbench] global-only {method:8s} seed={int(one['chain_seed'])} "
                            f"kept={int(one['iter_trace'].shape[0])} h2={float(one['h2_mean']):.4f} "
                            f"vare={float(one['vare']):.4f} time={float(one['elapsed_sec']):.2f}s"
                        )

            method_trace_rows.sort(key=lambda x: int(x["chain_seed"]))
            beta_list: list[np.ndarray] = []
            seeds: list[int] = []
            for one in method_trace_rows:
                beta = np.asarray(one["beta"], dtype=np.float64).reshape(-1)
                beta_list.append(beta)
                seeds.append(int(one["chain_seed"]))
                rows.append(
                    {
                        "method": method,
                        "trait": str(data["trait_name"]),
                        "chain_seed": int(one["chain_seed"]),
                        "n": int(one["n"]),
                        "m": int(beta.shape[0]),
                        "h2_mean": float(one["h2_mean"]),
                        "vare": float(one["vare"]),
                        "prob_in_mean": _safe_float(one["prob_in_mean"]),
                        "n_active_mean": _safe_float(one["n_active_mean"]),
                        "elapsed_sec": float(one["elapsed_sec"]),
                    }
                )
                iter_trace = np.asarray(one["iter_trace"], dtype=np.int64).reshape(-1)
                h2_trace = np.asarray(one["h2_trace"], dtype=np.float64).reshape(-1)
                var_e_trace = np.asarray(one["var_e_trace"], dtype=np.float64).reshape(-1)
                prob_in_trace = np.asarray(one["prob_in_trace"], dtype=np.float64).reshape(-1)
                n_active_trace = np.asarray(one["n_active_trace"], dtype=np.float64).reshape(-1)
                full_iter_trace = np.asarray(one["full_iter_trace"], dtype=np.int64).reshape(-1)
                full_h2_trace = np.asarray(one["full_h2_trace"], dtype=np.float64).reshape(-1)
                full_var_e_trace = np.asarray(one["full_var_e_trace"], dtype=np.float64).reshape(-1)
                full_prob_in_trace = np.asarray(one["full_prob_in_trace"], dtype=np.float64).reshape(-1)
                full_n_active_trace = np.asarray(one["full_n_active_trace"], dtype=np.float64).reshape(-1)
                for pos, iter_no in enumerate(iter_trace):
                    trace_rows.append(
                        {
                            "method": method,
                            "trait": str(data["trait_name"]),
                            "chain_seed": int(one["chain_seed"]),
                            "iter": int(iter_no),
                            "h2": float(h2_trace[pos]),
                            "var_e": float(var_e_trace[pos]),
                            "prob_in": _safe_float(prob_in_trace[pos]),
                            "n_active": _safe_float(n_active_trace[pos]),
                        }
                    )
                for pos, iter_no in enumerate(full_iter_trace):
                    global_trace_rows.append(
                        {
                            "method": method,
                            "trait": str(data["trait_name"]),
                            "chain_seed": int(one["chain_seed"]),
                            "iter": int(iter_no),
                            "h2": float(full_h2_trace[pos]),
                            "var_e": float(full_var_e_trace[pos]),
                            "prob_in": _safe_float(full_prob_in_trace[pos]),
                            "n_active": _safe_float(full_n_active_trace[pos]),
                        }
                    )
            for i in range(len(beta_list)):
                for j in range(i + 1, len(beta_list)):
                    pair_rows.append(
                        {
                            "method": method,
                            "seed_i": int(seeds[i]),
                            "seed_j": int(seeds[j]),
                            "beta_corr_all": _pearson(beta_list[i], beta_list[j]),
                            "beta_spearman_all": _spearman(beta_list[i], beta_list[j]),
                        }
                    )
            if beta_list:
                beta_stack = np.vstack(beta_list)
                consensus = np.mean(np.abs(beta_stack), axis=0)
                top_idx = np.argsort(-consensus)[:top_k] if top_k > 0 else np.empty((0,), dtype=np.int64)
                for rank, snp_idx in enumerate(top_idx, start=1):
                    row = {
                        "method": method,
                        "rank": int(rank),
                        "snp_index": int(snp_idx),
                        "mean_abs_beta": float(consensus[snp_idx]),
                    }
                    for seed_i, beta in zip(seeds, beta_list):
                        row[f"beta_seed_{int(seed_i)}"] = float(beta[snp_idx])
                    top_rows.append(row)
            sub = pd.DataFrame([r for r in rows if r["method"] == method])
            pair_sub = pd.DataFrame([r for r in pair_rows if r["method"] == method])
            summary_rows.append(
                {
                    "method": method,
                    "trait": str(data["trait_name"]),
                    "chains": int(sub.shape[0]),
                    "h2_mean_across_chains": float(sub["h2_mean"].mean()),
                    "h2_sd_across_chains": float(sub["h2_mean"].std(ddof=0)),
                    "vare_mean_across_chains": float(sub["vare"].mean()),
                    "vare_sd_across_chains": float(sub["vare"].std(ddof=0)),
                    "prob_in_mean_across_chains": float(sub["prob_in_mean"].mean(skipna=True)),
                    "prob_in_sd_across_chains": float(sub["prob_in_mean"].std(ddof=0, skipna=True)),
                    "n_active_mean_across_chains": float(sub["n_active_mean"].mean(skipna=True)),
                    "n_active_sd_across_chains": float(sub["n_active_mean"].std(ddof=0, skipna=True)),
                    "pair_beta_corr_mean": float(pair_sub["beta_corr_all"].mean()) if not pair_sub.empty else float("nan"),
                    "pair_beta_corr_min": float(pair_sub["beta_corr_all"].min()) if not pair_sub.empty else float("nan"),
                    "trace_status": "single_pass_global_only",
                }
            )

        chains_df = pd.DataFrame(rows)
        summary_df = pd.DataFrame(summary_rows)
        pair_df = pd.DataFrame(pair_rows)
        top_df = pd.DataFrame(top_rows)
        trace_df = pd.DataFrame(trace_rows)
        global_trace_df = pd.DataFrame(global_trace_rows)
        beta_trace_df = pd.DataFrame(beta_trace_rows)
        chains_path = out_dir / "convergence.chains.tsv"
        summary_path = out_dir / "convergence.summary.tsv"
        pair_path = out_dir / "convergence.pairwise.tsv"
        top_path = out_dir / "convergence.topbeta.tsv"
        trace_path = out_dir / "convergence.trace.tsv"
        global_trace_path = out_dir / "convergence.global_trace.tsv"
        beta_trace_path = out_dir / "convergence.beta_trace.tsv"
        meta_path = out_dir / "convergence.meta.json"
        chains_df.to_csv(chains_path, sep="\t", index=False)
        summary_df.to_csv(summary_path, sep="\t", index=False)
        pair_df.to_csv(pair_path, sep="\t", index=False)
        top_df.to_csv(top_path, sep="\t", index=False)
        trace_df.to_csv(trace_path, sep="\t", index=False)
        global_trace_df.to_csv(global_trace_path, sep="\t", index=False)
        beta_trace_df.to_csv(beta_trace_path, sep="\t", index=False)
        analysis_info = _analyze_convergence_outputs(
            trace_df=trace_df,
            global_trace_df=global_trace_df,
            beta_trace_df=beta_trace_df,
            out_dir=out_dir,
            burnin=int(args.burnin),
            rhat_threshold=float(args.rhat_threshold),
            stable_min_kept=int(args.stable_min_kept),
            plot_top_k_beta=int(args.plot_top_k_beta),
        )
        meta = {
            "status": "ok",
            "mode": "convergence",
            "trait": str(data["trait_name"]),
            "methods": methods,
            "chains": int(args.chains),
            "chain_seeds": chain_seeds,
            "parallel_chains": int(parallel_chains),
            "threads_per_chain": int(effective_chain_threads),
            "n": int(y_all.shape[0]),
            "m": int(np.asarray(packed_ctx["packed"]).shape[0]),
            "trace_status": "single_pass_global_only",
            "top_k_beta_trace": 0,
            "rhat_threshold": float(args.rhat_threshold),
            "stable_min_kept": int(args.stable_min_kept),
            "files": {
                "chains": str(chains_path),
                "summary": str(summary_path),
                "pairwise": str(pair_path),
                "topbeta": str(top_path),
                "trace": str(trace_path),
                "global_trace": str(global_trace_path),
                "beta_trace": str(beta_trace_path),
                "prefix_rhat": str(analysis_info["prefix_rhat_path"]),
                "stability": str(analysis_info["stability_path"]),
                "stability_md": str(analysis_info["stability_md_path"]),
            },
            "plots": analysis_info["plot_files"],
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\n[bayesbench] convergence summary")
        print(summary_df.to_string(index=False))
        stability_df = pd.read_csv(Path(str(analysis_info["stability_path"])), sep="\t")
        if not stability_df.empty:
            print("\n[bayesbench] stability summary")
            print(stability_df.to_string(index=False))
        print(f"[bayesbench] wrote: {summary_path}")
        return 0
    for method in methods:
        beta_by_method[method] = []
        beta_seed_by_method[method] = []
        method_rows: list[dict[str, Any]] = []

        def _run_one_chain(seed: int) -> dict[str, Any]:
            t0 = time.perf_counter()
            fit = _fit_janusx_packed(
                method=method,
                packed_ctx=packed_ctx,
                sample_indices=sample_idx,
                y=y_all,
                row_mean=row_mean,
                row_inv_sd=row_inv_sd,
                n_iter=int(args.n_iter),
                burnin=int(args.burnin),
                thin=int(args.thin),
                r2=float(args.r2),
                prob_in=float(args.prob_in),
                counts=float(args.counts),
                df0_b=float(args.df0_b),
                shape0=float(args.shape0),
                rate0=(None if args.rate0 is None else float(args.rate0)),
                s0_b=(None if args.s0_b is None else float(args.s0_b)),
                df0_e=float(args.df0_e),
                s0_e=(None if args.s0_e is None else float(args.s0_e)),
                threads=int(effective_chain_threads),
                seed=int(seed),
            )
            elapsed = float(time.perf_counter() - t0)
            beta = np.asarray(fit["beta"], dtype=np.float64)
            return {
                "method": method,
                "trait": str(data["trait_name"]),
                "chain_seed": int(seed),
                "n": int(y_all.shape[0]),
                "m": int(beta.shape[0]),
                "h2_mean": float(fit["h2_mean"]),
                "vare": float(fit["vare"]),
                "prob_in_mean": _safe_float(fit["prob_in_mean"]),
                "n_active_mean": _safe_float(fit["n_active_mean"]),
                "elapsed_sec": elapsed,
                "beta": beta,
            }

        if parallel_chains <= 1:
            for seed in chain_seeds:
                one = _run_one_chain(int(seed))
                method_rows.append(one)
                print(
                    f"[bayesbench] convergence {method:8s} seed={int(seed)} "
                    f"h2={float(one['h2_mean']):.4f} vare={float(one['vare']):.4f} "
                    f"time={float(one['elapsed_sec']):.2f}s"
                )
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(parallel_chains)) as ex:
                future_map = {
                    ex.submit(_run_one_chain, int(seed)): int(seed)
                    for seed in chain_seeds
                }
                for fut in concurrent.futures.as_completed(future_map):
                    one = fut.result()
                    method_rows.append(one)
                    print(
                        f"[bayesbench] convergence {method:8s} seed={int(one['chain_seed'])} "
                        f"h2={float(one['h2_mean']):.4f} vare={float(one['vare']):.4f} "
                        f"time={float(one['elapsed_sec']):.2f}s"
                    )

        method_rows.sort(key=lambda x: int(x["chain_seed"]))
        for one in method_rows:
            beta = np.asarray(one.pop("beta"), dtype=np.float64)
            beta_by_method[method].append(beta)
            beta_seed_by_method[method].append(int(one["chain_seed"]))
            rows.append(one)
    chains_df = pd.DataFrame(rows)
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    global_trace_rows: list[dict[str, Any]] = []
    beta_trace_rows: list[dict[str, Any]] = []
    trace_top_idx_by_method: dict[str, np.ndarray] = {}
    top_k = int(max(1, int(args.top_k_beta)))
    for method in methods:
        sub = chains_df[chains_df["method"] == method].reset_index(drop=True)
        if sub.empty:
            continue
        beta_list = beta_by_method[method]
        seeds = beta_seed_by_method[method]
        for i in range(len(beta_list)):
            for j in range(i + 1, len(beta_list)):
                pair_rows.append(
                    {
                        "method": method,
                        "seed_i": int(seeds[i]),
                        "seed_j": int(seeds[j]),
                        "beta_corr_all": _pearson(beta_list[i], beta_list[j]),
                        "beta_spearman_all": _spearman(beta_list[i], beta_list[j]),
                    }
                )
        beta_stack = np.vstack(beta_list)
        consensus = np.mean(np.abs(beta_stack), axis=0)
        top_idx = np.argsort(-consensus)[:top_k]
        trace_top_idx_by_method[method] = np.ascontiguousarray(top_idx.astype(np.int64, copy=False), dtype=np.int64)
        for rank, snp_idx in enumerate(top_idx, start=1):
            row = {
                "method": method,
                "rank": int(rank),
                "snp_index": int(snp_idx),
                "mean_abs_beta": float(consensus[snp_idx]),
            }
            for seed_i, beta in zip(seeds, beta_list):
                row[f"beta_seed_{int(seed_i)}"] = float(beta[snp_idx])
            top_rows.append(row)
        pair_sub = pd.DataFrame([r for r in pair_rows if r["method"] == method])
        summary_rows.append(
            {
                "method": method,
                "trait": str(data["trait_name"]),
                "chains": int(sub.shape[0]),
                "h2_mean_across_chains": float(sub["h2_mean"].mean()),
                "h2_sd_across_chains": float(sub["h2_mean"].std(ddof=0)),
                "vare_mean_across_chains": float(sub["vare"].mean()),
                "vare_sd_across_chains": float(sub["vare"].std(ddof=0)),
                "prob_in_mean_across_chains": float(sub["prob_in_mean"].mean(skipna=True)),
                "prob_in_sd_across_chains": float(sub["prob_in_mean"].std(ddof=0, skipna=True)),
                "n_active_mean_across_chains": float(sub["n_active_mean"].mean(skipna=True)),
                "n_active_sd_across_chains": float(sub["n_active_mean"].std(ddof=0, skipna=True)),
                "pair_beta_corr_mean": float(pair_sub["beta_corr_all"].mean()) if not pair_sub.empty else float("nan"),
                "pair_beta_corr_min": float(pair_sub["beta_corr_all"].min()) if not pair_sub.empty else float("nan"),
                "trace_status": "per_iteration_backend",
            }
        )

    print("[bayesbench] rerunning chains for per-iteration traces")
    for method in methods:
        trace_idx = np.ascontiguousarray(trace_top_idx_by_method.get(method, np.empty((0,), dtype=np.int64)), dtype=np.int64)
        method_trace_rows: list[dict[str, Any]] = []

        def _run_one_trace_chain(seed: int) -> dict[str, Any]:
            t0 = time.perf_counter()
            fit = _fit_janusx_packed_trace(
                method=method,
                packed_ctx=packed_ctx,
                sample_indices=sample_idx,
                y=y_all,
                row_mean=row_mean,
                row_inv_sd=row_inv_sd,
                trace_snp_indices=trace_idx,
                n_iter=int(args.n_iter),
                burnin=int(args.burnin),
                thin=int(args.thin),
                r2=float(args.r2),
                prob_in=float(args.prob_in),
                counts=float(args.counts),
                df0_b=float(args.df0_b),
                shape0=float(args.shape0),
                rate0=(None if args.rate0 is None else float(args.rate0)),
                s0_b=(None if args.s0_b is None else float(args.s0_b)),
                df0_e=float(args.df0_e),
                s0_e=(None if args.s0_e is None else float(args.s0_e)),
                threads=int(effective_chain_threads),
                seed=int(seed),
            )
            elapsed = float(time.perf_counter() - t0)
            return {
                "method": method,
                "trait": str(data["trait_name"]),
                "chain_seed": int(seed),
                "elapsed_sec": elapsed,
                "iter_trace": np.asarray(fit["iter_trace"], dtype=np.int64),
                "h2_trace": np.asarray(fit["h2_trace"], dtype=np.float64),
                "var_e_trace": np.asarray(fit["var_e_trace"], dtype=np.float64),
                "prob_in_trace": np.asarray(fit["prob_in_trace"], dtype=np.float64),
                "n_active_trace": np.asarray(fit["n_active_trace"], dtype=np.float64),
                "full_iter_trace": np.asarray(fit["full_iter_trace"], dtype=np.int64),
                "full_h2_trace": np.asarray(fit["full_h2_trace"], dtype=np.float64),
                "full_var_e_trace": np.asarray(fit["full_var_e_trace"], dtype=np.float64),
                "full_prob_in_trace": np.asarray(fit["full_prob_in_trace"], dtype=np.float64),
                "full_n_active_trace": np.asarray(fit["full_n_active_trace"], dtype=np.float64),
                "beta_trace_indices": np.asarray(fit["beta_trace_indices"], dtype=np.int64),
                "beta_trace": np.asarray(fit["beta_trace"], dtype=np.float64),
            }

        if parallel_chains <= 1:
            for seed in chain_seeds:
                one = _run_one_trace_chain(int(seed))
                method_trace_rows.append(one)
                print(
                    f"[bayesbench] trace {method:8s} seed={int(seed)} "
                    f"kept={int(one['iter_trace'].shape[0])} time={float(one['elapsed_sec']):.2f}s"
                )
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(parallel_chains)) as ex:
                future_map = {
                    ex.submit(_run_one_trace_chain, int(seed)): int(seed)
                    for seed in chain_seeds
                }
                for fut in concurrent.futures.as_completed(future_map):
                    one = fut.result()
                    method_trace_rows.append(one)
                    print(
                        f"[bayesbench] trace {method:8s} seed={int(one['chain_seed'])} "
                        f"kept={int(one['iter_trace'].shape[0])} time={float(one['elapsed_sec']):.2f}s"
                    )

        method_trace_rows.sort(key=lambda x: int(x["chain_seed"]))
        for one in method_trace_rows:
            iter_trace = np.asarray(one["iter_trace"], dtype=np.int64).reshape(-1)
            h2_trace = np.asarray(one["h2_trace"], dtype=np.float64).reshape(-1)
            var_e_trace = np.asarray(one["var_e_trace"], dtype=np.float64).reshape(-1)
            prob_in_trace = np.asarray(one["prob_in_trace"], dtype=np.float64).reshape(-1)
            n_active_trace = np.asarray(one["n_active_trace"], dtype=np.float64).reshape(-1)
            full_iter_trace = np.asarray(one["full_iter_trace"], dtype=np.int64).reshape(-1)
            full_h2_trace = np.asarray(one["full_h2_trace"], dtype=np.float64).reshape(-1)
            full_var_e_trace = np.asarray(one["full_var_e_trace"], dtype=np.float64).reshape(-1)
            full_prob_in_trace = np.asarray(one["full_prob_in_trace"], dtype=np.float64).reshape(-1)
            full_n_active_trace = np.asarray(one["full_n_active_trace"], dtype=np.float64).reshape(-1)
            beta_trace_indices = np.asarray(one["beta_trace_indices"], dtype=np.int64).reshape(-1)
            beta_trace = np.asarray(one["beta_trace"], dtype=np.float64)
            if beta_trace.ndim == 1:
                beta_trace = beta_trace.reshape((int(iter_trace.shape[0]), int(beta_trace_indices.shape[0])))
            for pos, iter_no in enumerate(iter_trace):
                trace_rows.append(
                    {
                        "method": method,
                        "trait": str(data["trait_name"]),
                        "chain_seed": int(one["chain_seed"]),
                        "iter": int(iter_no),
                        "h2": float(h2_trace[pos]),
                        "var_e": float(var_e_trace[pos]),
                        "prob_in": _safe_float(prob_in_trace[pos]),
                        "n_active": _safe_float(n_active_trace[pos]),
                        }
                    )
            for pos, iter_no in enumerate(full_iter_trace):
                global_trace_rows.append(
                    {
                        "method": method,
                        "trait": str(data["trait_name"]),
                        "chain_seed": int(one["chain_seed"]),
                        "iter": int(iter_no),
                        "h2": float(full_h2_trace[pos]),
                        "var_e": float(full_var_e_trace[pos]),
                        "prob_in": _safe_float(full_prob_in_trace[pos]),
                        "n_active": _safe_float(full_n_active_trace[pos]),
                    }
                )
            for rank, snp_idx in enumerate(beta_trace_indices, start=1):
                for pos, iter_no in enumerate(iter_trace):
                    beta_trace_rows.append(
                        {
                            "method": method,
                            "trait": str(data["trait_name"]),
                            "chain_seed": int(one["chain_seed"]),
                            "iter": int(iter_no),
                            "rank": int(rank),
                            "snp_index": int(snp_idx),
                            "beta": float(beta_trace[pos, rank - 1]),
                        }
                    )
    summary_df = pd.DataFrame(summary_rows)
    pair_df = pd.DataFrame(pair_rows)
    top_df = pd.DataFrame(top_rows)
    trace_df = pd.DataFrame(trace_rows)
    global_trace_df = pd.DataFrame(global_trace_rows)
    beta_trace_df = pd.DataFrame(beta_trace_rows)
    chains_path = out_dir / "convergence.chains.tsv"
    summary_path = out_dir / "convergence.summary.tsv"
    pair_path = out_dir / "convergence.pairwise.tsv"
    top_path = out_dir / "convergence.topbeta.tsv"
    trace_path = out_dir / "convergence.trace.tsv"
    global_trace_path = out_dir / "convergence.global_trace.tsv"
    beta_trace_path = out_dir / "convergence.beta_trace.tsv"
    meta_path = out_dir / "convergence.meta.json"
    chains_df.to_csv(chains_path, sep="\t", index=False)
    summary_df.to_csv(summary_path, sep="\t", index=False)
    pair_df.to_csv(pair_path, sep="\t", index=False)
    top_df.to_csv(top_path, sep="\t", index=False)
    trace_df.to_csv(trace_path, sep="\t", index=False)
    global_trace_df.to_csv(global_trace_path, sep="\t", index=False)
    beta_trace_df.to_csv(beta_trace_path, sep="\t", index=False)
    analysis_info = _analyze_convergence_outputs(
        trace_df=trace_df,
        global_trace_df=global_trace_df,
        beta_trace_df=beta_trace_df,
        out_dir=out_dir,
        burnin=int(args.burnin),
        rhat_threshold=float(args.rhat_threshold),
        stable_min_kept=int(args.stable_min_kept),
        plot_top_k_beta=int(args.plot_top_k_beta),
    )
    meta = {
        "status": "ok",
        "mode": "convergence",
        "trait": str(data["trait_name"]),
        "methods": methods,
        "chains": int(args.chains),
        "chain_seeds": chain_seeds,
        "parallel_chains": int(parallel_chains),
        "threads_per_chain": int(effective_chain_threads),
        "n": int(y_all.shape[0]),
        "m": int(np.asarray(packed_ctx["packed"]).shape[0]),
        "trace_status": "per_iteration_backend",
        "top_k_beta_trace": int(top_k),
        "rhat_threshold": float(args.rhat_threshold),
        "stable_min_kept": int(args.stable_min_kept),
        "files": {
            "chains": str(chains_path),
            "summary": str(summary_path),
            "pairwise": str(pair_path),
            "topbeta": str(top_path),
            "trace": str(trace_path),
            "global_trace": str(global_trace_path),
            "beta_trace": str(beta_trace_path),
            "prefix_rhat": str(analysis_info["prefix_rhat_path"]),
            "stability": str(analysis_info["stability_path"]),
            "stability_md": str(analysis_info["stability_md_path"]),
        },
        "plots": analysis_info["plot_files"],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n[bayesbench] convergence summary")
    print(summary_df.to_string(index=False))
    stability_df = pd.read_csv(Path(str(analysis_info["stability_path"])), sep="\t")
    if not stability_df.empty:
        print("\n[bayesbench] stability summary")
        print(stability_df.to_string(index=False))
    print(f"[bayesbench] wrote: {summary_path}")
    return 0


def run_compare(args: argparse.Namespace) -> int:
    _validate_real_args(args)
    ref_engines = _parse_reference_engines(str(args.reference))
    rscript_bin = str(args.rscript).strip()
    if rscript_bin == "":
        raise ValueError("--rscript is empty.")
    if shutil.which(rscript_bin) is None and (not Path(rscript_bin).expanduser().exists()):
        raise ValueError(f"Rscript executable was not found: {rscript_bin}")
    out_dir = Path(str(args.out)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_real_dataset(args)
    methods = _parse_methods(str(args.methods))
    y_all = np.asarray(data["y_all"], dtype=np.float64)
    overlap_abs_idx = np.asarray(data["overlap_abs_idx"], dtype=np.int64)
    overlap_ids = np.asarray(data["overlap_ids"], dtype=str)
    train_rel, test_rel = _make_train_test_split(
        n=int(y_all.shape[0]),
        test_frac=float(args.test_frac),
        seed=int(args.split_seed),
    )
    train_abs = np.ascontiguousarray(overlap_abs_idx[train_rel], dtype=np.int64)
    test_abs = np.ascontiguousarray(overlap_abs_idx[test_rel], dtype=np.int64)
    train_ids = np.asarray(overlap_ids[train_rel], dtype=str)
    test_ids = np.asarray(overlap_ids[test_rel], dtype=str)
    y_train = np.ascontiguousarray(y_all[train_rel], dtype=np.float64)
    y_test = np.ascontiguousarray(y_all[test_rel], dtype=np.float64)
    packed_ctx = data["packed_ctx"]
    row_mean = data["row_mean"]
    row_inv_sd = data["row_inv_sd"]
    print(
        f"[bayesbench] compare trait={data['trait_name']} n_train={int(y_train.shape[0])} "
        f"n_test={int(y_test.shape[0])} m={int(np.asarray(packed_ctx['packed']).shape[0])} "
        f"refs={','.join(ref_engines)}"
    )

    x_train_std = _decode_packed_subset_to_dense_standardized(
        packed_ctx=packed_ctx,
        sample_indices=train_abs,
        row_mean=row_mean,
        row_inv_sd=row_inv_sd,
        row_block_size=int(max(1, int(args.snp_block_size))),
        out_dtype=np.float32,
    ).T
    x_test_std = _decode_packed_subset_to_dense_standardized(
        packed_ctx=packed_ctx,
        sample_indices=test_abs,
        row_mean=row_mean,
        row_inv_sd=row_inv_sd,
        row_block_size=int(max(1, int(args.snp_block_size))),
        out_dtype=np.float32,
    ).T
    top_cutoffs: list[int] = []
    for part in str(args.top_beta_cutoffs).split(","):
        tok = part.strip()
        if tok == "":
            continue
        top_cutoffs.append(max(1, int(tok)))
    if len(top_cutoffs) == 0:
        top_cutoffs = [100, 1000]

    engine_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    beta_long_rows: list[pd.DataFrame] = []
    pred_long_rows: list[pd.DataFrame] = []
    work_root = out_dir / "compare_work"
    work_root.mkdir(parents=True, exist_ok=True)
    for method in methods:
        software_results: dict[str, dict[str, Any]] = {}

        jx_res = _run_janusx_reference(
            method=method,
            work_dir=work_root / method.lower() / "janusx",
            packed_ctx=packed_ctx,
            row_mean=row_mean,
            row_inv_sd=row_inv_sd,
            train_abs=train_abs,
            test_abs=test_abs,
            train_ids=train_ids,
            test_ids=test_ids,
            y_train=y_train,
            y_test=y_test,
            n_iter=int(args.n_iter),
            burnin=int(args.burnin),
            thin=int(args.thin),
            r2=float(args.r2),
            prob_in=float(args.prob_in),
            counts=float(args.counts),
            df0_b=float(args.df0_b),
            shape0=float(args.shape0),
            rate0=(None if args.rate0 is None else float(args.rate0)),
            s0_b=(None if args.s0_b is None else float(args.s0_b)),
            df0_e=float(args.df0_e),
            s0_e=(None if args.s0_e is None else float(args.s0_e)),
            threads=int(args.thread),
            seed=int(args.seed),
            snp_block_size=int(args.snp_block_size),
            sample_chunk_size=int(args.sample_chunk_size),
        )
        software_results["JanusX"] = jx_res
        jx_beta_df = pd.DataFrame(jx_res["beta_df"]).copy()
        jx_beta_df.insert(0, "software", "JanusX")
        jx_beta_df.insert(0, "method", method)
        beta_long_rows.append(jx_beta_df)
        jx_pred_df = pd.DataFrame(jx_res["pred_df"]).copy()
        jx_pred_df.insert(0, "software", "JanusX")
        jx_pred_df.insert(0, "method", method)
        pred_long_rows.append(jx_pred_df)
        engine_rows.append(
            _engine_summary_row(
                method=method,
                trait=str(data["trait_name"]),
                software="JanusX",
                result=jx_res,
                n_train=int(train_ids.shape[0]),
                n_test=int(test_ids.shape[0]),
                m=int(np.asarray(jx_beta_df["beta"]).shape[0]),
            )
        )

        for ref_engine in ref_engines:
            if ref_engine == "bglr":
                ref_res = _run_bglr_reference(
                    work_dir=work_root / method.lower() / "bglr",
                    method=method,
                    x_train=x_train_std,
                    x_test=x_test_std,
                    y_train=y_train,
                    y_test=y_test,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    n_iter=int(args.n_iter),
                    burnin=int(args.burnin),
                    thin=int(args.thin),
                    r2=float(args.r2),
                    prob_in=float(args.prob_in),
                    counts=float(args.counts),
                    df0_b=float(args.df0_b),
                    shape0=float(args.shape0),
                    rate0=(None if args.rate0 is None else float(args.rate0)),
                    s0_b=(None if args.s0_b is None else float(args.s0_b)),
                    df0_e=float(args.df0_e),
                    s0_e=(None if args.s0_e is None else float(args.s0_e)),
                    threads=max(1, int(args.thread) if int(args.thread) > 0 else 1),
                    rscript_bin=rscript_bin,
                )
                software_name = "BGLR"
            elif ref_engine == "hibayes":
                ref_res = _run_hibayes_reference(
                    work_dir=work_root / method.lower() / "hibayes",
                    method=method,
                    x_train=x_train_std,
                    x_test=x_test_std,
                    y_train=y_train,
                    y_test=y_test,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    n_iter=int(args.n_iter),
                    burnin=int(args.burnin),
                    thin=int(args.thin),
                    prob_in=float(args.prob_in),
                    threads=max(1, int(args.thread) if int(args.thread) > 0 else 1),
                    seed=int(args.seed),
                    rscript_bin=rscript_bin,
                )
                software_name = "HIBayes"
            else:
                raise ValueError(f"Unsupported reference engine: {ref_engine}")
            software_results[software_name] = ref_res
            ref_beta_df = pd.DataFrame(ref_res["beta_df"]).copy()
            ref_beta_df.insert(0, "software", software_name)
            ref_beta_df.insert(0, "method", method)
            beta_long_rows.append(ref_beta_df)
            ref_pred_df = pd.DataFrame(ref_res["pred_df"]).copy()
            ref_pred_df.insert(0, "software", software_name)
            ref_pred_df.insert(0, "method", method)
            pred_long_rows.append(ref_pred_df)
            engine_rows.append(
                _engine_summary_row(
                    method=method,
                    trait=str(data["trait_name"]),
                    software=software_name,
                    result=ref_res,
                    n_train=int(train_ids.shape[0]),
                    n_test=int(test_ids.shape[0]),
                    m=int(np.asarray(ref_beta_df["beta"]).shape[0]),
                )
            )

        for software_a, software_b in itertools.combinations(software_results.keys(), 2):
            row = _pairwise_compare_row(
                method=method,
                trait=str(data["trait_name"]),
                software_a=str(software_a),
                software_b=str(software_b),
                result_a=software_results[software_a],
                result_b=software_results[software_b],
                beta_df_a=pd.DataFrame(software_results[software_a]["beta_df"]),
                beta_df_b=pd.DataFrame(software_results[software_b]["beta_df"]),
                pred_df_a=pd.DataFrame(software_results[software_a]["pred_df"]),
                pred_df_b=pd.DataFrame(software_results[software_b]["pred_df"]),
                top_cutoffs=top_cutoffs,
            )
            pair_rows.append(row)
            print(
                f"[bayesbench] compare {method:8s} {software_a:7s} vs {software_b:7s} "
                f"beta_corr_all={_safe_float(row.get('beta_corr_all', np.nan)):.4f} "
                f"gebv_corr_test={_safe_float(row.get('gebv_corr_test', np.nan)):.4f}"
            )

    engine_df = pd.DataFrame(engine_rows)
    summary_df = pd.DataFrame(pair_rows)
    beta_df = pd.concat(beta_long_rows, ignore_index=True) if beta_long_rows else pd.DataFrame()
    pred_df = pd.concat(pred_long_rows, ignore_index=True) if pred_long_rows else pd.DataFrame()
    summary_path = out_dir / "compare.summary.tsv"
    engine_path = out_dir / "compare.engine.tsv"
    beta_path = out_dir / "compare.beta.tsv"
    pred_path = out_dir / "compare.pred.tsv"
    meta_path = out_dir / "compare.meta.json"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    engine_df.to_csv(engine_path, sep="\t", index=False)
    beta_df.to_csv(beta_path, sep="\t", index=False)
    pred_df.to_csv(pred_path, sep="\t", index=False)
    meta = {
        "status": "ok",
        "mode": "compare",
        "trait": str(data["trait_name"]),
        "methods": methods,
        "reference": ref_engines,
        "reference_note": (
            "BayesB is mapped to HIBayes BayesBpi because JanusX BayesB samples the inclusion "
            "probability while HIBayes BayesB keeps Pi fixed. "
            "BayesCpi is mapped to BGLR BayesC because BGLR does not expose BayesCpi. "
            "HIBayes uses its native individual-level ibrm interface and native prior parameterization. "
            "Hyperparameters are matched as closely as allowed by the public interfaces."
        ),
        "n_train": int(train_ids.shape[0]),
        "n_test": int(test_ids.shape[0]),
        "m": int(np.asarray(packed_ctx["packed"]).shape[0]),
        "files": {
            "summary": str(summary_path),
            "engine": str(engine_path),
            "beta": str(beta_path),
            "pred": str(pred_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n[bayesbench] engine summary")
    print(engine_df.to_string(index=False))
    print("\n[bayesbench] pairwise summary")
    print(summary_df.to_string(index=False))
    print(f"[bayesbench] wrote: {summary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if len(tokens) == 0:
        tokens = ["kernel"]
    elif tokens[0] not in _MODE_TOKENS and tokens[0] not in {"-h", "--help"}:
        tokens = ["kernel"] + tokens
    parser = _build_parser()
    args = parser.parse_args(tokens)
    mode = args.mode or "kernel"
    if mode == "kernel":
        return run_kernel(args)
    if mode == "convergence":
        return run_convergence(args)
    if mode == "compare":
        return run_compare(args)
    raise ValueError(f"Unsupported mode: {mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
