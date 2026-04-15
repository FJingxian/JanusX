#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GBLUP benchmark launcher for JanusX / sommer / rrBLUP.

Key features:
  - One-command benchmark for three engines:
      * janusx (Python BLUP backend)
      * sommer (R package)
      * rrblup (R package rrBLUP)
  - Unified data filtering / trait selection / CV splits across engines.
  - Per-sample out-of-fold prediction tables for each engine.
  - Automatic pairwise prediction comparison when multiple engines are selected.
  - Runtime + peak RSS profiling via system `time` wrapper.
  - Temporary engine scripts are generated on-the-fly (Python/R).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

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

from janusx.script._common.pathcheck import safe_expanduser  # noqa: E402
from janusx.script._common.threads import detect_effective_threads  # noqa: E402
from janusx.script._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog  # noqa: E402
from janusx.script._common.colspec import parse_zero_based_index_specs  # noqa: E402
from janusx.script.gs import _load_genotype_with_rust_gfreader, build_cv_splits  # noqa: E402


_SUMMARY_TAG = "gblupbench"


@dataclass
class EngineRunResult:
    engine: str
    status: str
    exit_code: int
    elapsed_sec: float
    peak_rss_kb: float
    mean_r2: float
    mean_pearson: float
    mean_spearman: float
    mean_pve: float
    folds: int
    fold_file: str
    prediction_file: str
    result_json: str
    log_file: str
    time_file: str
    note: str


@dataclass
class EnvCheck:
    component: str
    ok: bool
    detail: str


def _strip_default_prefix_suffix(path: str) -> str:
    base = os.path.basename(str(path).rstrip("/\\"))
    low = base.lower()
    if low.endswith(".vcf.gz"):
        return base[: -len(".vcf.gz")]
    if low.endswith(".hmp.gz"):
        return base[: -len(".hmp.gz")]
    for ext in (".vcf", ".hmp", ".txt", ".tsv", ".csv", ".npy"):
        if low.endswith(ext):
            return base[: -len(ext)]
    return base


def _determine_genotype_source_from_args(args: argparse.Namespace) -> tuple[str, str]:
    vcf = getattr(args, "vcf", None)
    hmp = getattr(args, "hmp", None)
    file = getattr(args, "file", None)
    bfile = getattr(args, "bfile", None)
    prefix = getattr(args, "prefix", None)
    if vcf:
        gfile = str(vcf)
        auto_prefix = _strip_default_prefix_suffix(gfile)
    elif hmp:
        gfile = str(hmp)
        auto_prefix = _strip_default_prefix_suffix(gfile)
    elif file:
        gfile = str(file)
        auto_prefix = _strip_default_prefix_suffix(gfile)
    elif bfile:
        gfile = str(bfile)
        auto_prefix = os.path.basename(gfile.rstrip("/\\"))
    else:
        raise ValueError("No genotype input specified. Use -vcf, -hmp, -file or -bfile.")
    resolved_prefix = str(prefix) if prefix is not None else auto_prefix
    return gfile, resolved_prefix


def _safe_float(v: Any, default: float = math.nan) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    return x


def _parse_elapsed_text(text: str) -> float:
    s = str(text).strip()
    m = re.fullmatch(r"(?:(\d+)h\s+)?(\d+)m\s+([0-9]*\.?[0-9]+)s", s)
    if m:
        return float(m.group(1) or 0.0) * 3600.0 + float(m.group(2)) * 60.0 + float(m.group(3))
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 1:
            return float(parts[0])
    except Exception:
        pass
    return math.nan


def _detect_time_tool() -> list[str]:
    gtime = shutil.which("gtime")
    if gtime:
        return [gtime, "-v"]
    if Path("/usr/bin/time").exists():
        rc = subprocess.run(
            ["/usr/bin/time", "-v", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            text=False,
        ).returncode
        if rc == 0:
            return ["/usr/bin/time", "-v"]
        return ["/usr/bin/time", "-l"]
    return []


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
            rss = _safe_float(m.group(1), math.nan)
            continue
        m = bsd_elapsed.search(line)
        if m:
            elapsed = _safe_float(m.group(1), math.nan)
            continue
        m = bsd_rss.search(line)
        if m:
            rss = _safe_float(m.group(1), math.nan) / 1024.0
    return elapsed, rss


def _run_timed(
    cmd: list[str],
    *,
    log_file: Path,
    time_file: Path,
    env: Optional[dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> tuple[int, float, float]:
    """Returns: (exit_code, elapsed_sec, peak_rss_kb)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    time_file.parent.mkdir(parents=True, exist_ok=True)
    time_tool = _detect_time_tool()
    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as lf:
        if time_tool:
            full = [*time_tool, "-o", str(time_file), *cmd]
        else:
            full = list(cmd)
        proc = subprocess.run(
            full,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            check=False,
            text=True,
        )
    wall = max(time.time() - t0, 0.0)
    if not time_tool:
        time_file.write_text(
            "\n".join(
                [
                    f"Elapsed (wall clock) time (h:mm:ss or m:ss): {wall:.6f}",
                    "Maximum resident set size (kbytes): NA",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    elapsed, rss = _parse_time_file(time_file)
    if not math.isfinite(elapsed):
        elapsed = wall
    return int(proc.returncode), float(elapsed), float(rss)


def _read_table_guess(path: Path) -> pd.DataFrame:
    tries = [
        dict(sep="\t", engine="python"),
        dict(sep=",", engine="python"),
        dict(sep=r"\s+", engine="python"),
        dict(sep=None, engine="python"),
    ]
    last_err: Optional[Exception] = None
    for kw in tries:
        try:
            df = pd.read_csv(path, **kw)
            if df.shape[1] >= 1:
                return df
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Unable to parse table: {path}")


def _read_pheno(path: Path, ncol: Optional[list[int]]) -> tuple[pd.DataFrame, str]:
    df = _read_table_guess(path)
    if df.shape[1] < 2:
        raise ValueError("Phenotype file must contain sample ID and at least one trait column.")
    id_col = df.columns[0]
    if ncol is None:
        trait_col = df.columns[1]
    else:
        if len(ncol) == 0:
            raise ValueError("`-n/--n` is empty. Provide one zero-based phenotype column index.")
        if len(ncol) > 1:
            raise ValueError("`jx gblupbench` currently supports one trait at a time; pass a single `-n/--n`.")
        idx = int(ncol[0]) + 1
        if idx < 1 or idx >= int(df.shape[1]):
            raise ValueError(
                f"`-n/--n` out of range: {ncol[0]}. Valid range is 0..{int(df.shape[1]) - 2} "
                "(zero-based phenotype columns, excluding sample ID)."
            )
        trait_col = df.columns[idx]

    out = pd.DataFrame(
        {
            "Taxa": df[id_col].astype(str).str.strip(),
            "PHENO": pd.to_numeric(df[trait_col], errors="coerce"),
        }
    )
    out = out[(out["Taxa"] != "") & out["PHENO"].notna()].drop_duplicates("Taxa", keep="first")
    if out.empty:
        raise ValueError("No usable phenotype rows remain after filtering.")
    return out, str(trait_col)


def _normalize_engine_token(token: str) -> str:
    t = str(token).strip().lower().replace("-", "").replace("_", "")
    if t in {"janusx", "jx", "janus"}:
        return "janusx"
    if t in {"sommer"}:
        return "sommer"
    if t in {"rrblup", "rrb", "rr"}:
        return "rrblup"
    return ""


def _parse_engines(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in str(raw).split(","):
        e = _normalize_engine_token(part)
        if not e or e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def _dev_help_requested(argv: Optional[list[str]] = None) -> bool:
    tokens = list(sys.argv[1:] if argv is None else argv)
    return ("-dev" in tokens) or ("--dev" in tokens)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    show_dev_help = _dev_help_requested(argv)
    tokens = list(sys.argv[1:] if argv is None else argv)
    p = CliArgumentParser(
        prog="jx gblupbench",
        formatter_class=cli_help_formatter(),
        description="Benchmark GBLUP engines: JanusX / sommer / rrBLUP.",
        epilog=minimal_help_epilog(
            [
                "jx gblupbench -bfile example_prefix -p pheno.tsv -n 0",
                "jx gblupbench -vcf example.vcf.gz -p pheno.tsv -n 0 --engines janusx,sommer,rrblup",
                "jx gblupbench -h -dev",
            ]
        ),
    )
    p.add_argument("-dev", "--dev", action="store_true", default=False, help=argparse.SUPPRESS)

    required_group = p.add_argument_group("Required arguments")
    g = required_group.add_mutually_exclusive_group(required=False)
    g.add_argument("-bfile", "--bfile", type=str, help="Input PLINK prefix (.bed/.bim/.fam).")
    g.add_argument("-vcf", "--vcf", type=str, help="Input VCF/VCF.GZ genotype.")
    g.add_argument("-hmp", "--hmp", type=str, help="Input HMP/HMP.GZ genotype.")
    g.add_argument("-file", "--file", type=str, help="Input text/NPY genotype matrix or prefix.")
    required_group.add_argument("-p", "--pheno", required=False, type=str, help="Phenotype file (first col sample ID).")

    optional_group = p.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "-n",
        "--n",
        action="extend",
        nargs="+",
        metavar="COL",
        default=None,
        type=str,
        dest="ncol",
        help=(
            "Phenotype column(s), zero-based index (excluding sample ID), comma list, or numeric range. "
            "This benchmark currently supports one selected trait."
        ),
    )
    optional_group.add_argument(
        "--ncol",
        action="extend",
        nargs="+",
        metavar="COL",
        default=None,
        type=str,
        dest="ncol",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument("-o", "--out", default=".", type=str, help="Output directory.")
    optional_group.add_argument("-prefix", "--prefix", default="gblupbench", type=str, help="Output prefix.")
    optional_group.add_argument("-maf", "--maf", default=0.02, type=float, help="MAF filter (default: %(default)s).")
    optional_group.add_argument("-geno", "--geno", default=0.05, type=float, help="Missing-rate filter (default: %(default)s).")
    optional_group.add_argument("-chunksize", "--chunksize", default=50_000, type=int, help="Genotype chunk size.")
    optional_group.add_argument("-t", "--thread", default=detect_effective_threads(), type=int, help="Threads.")
    optional_group.add_argument("--cv", default=5, type=int, help="CV folds (default: %(default)s).")
    optional_group.add_argument("--seed", default=42, type=int, help="Random seed (default: %(default)s).")
    optional_group.add_argument("--engines", default="janusx,sommer,rrblup", type=str, help="Comma list: janusx,sommer,rrblup.")
    optional_group.add_argument(
        "--check",
        action="store_true",
        default=False,
        help=(
            "Only check environment dependencies for selected engines, then exit. "
            "In this mode, genotype/pheno inputs are optional."
        ),
    )

    advanced_group = p.add_argument_group("Advanced arguments (show with -h -dev)")
    advanced_group.add_argument(
        "--keep-temp",
        action="store_true",
        default=False,
        help=("Keep temporary files." if show_dev_help else argparse.SUPPRESS),
    )

    if any(
        (tk == "-trait") or tk.startswith("-trait=") or (tk == "--trait") or tk.startswith("--trait=")
        for tk in tokens
    ):
        p.error("`-trait/--trait` has been replaced by `-n/--n` (zero-based phenotype column index, excluding sample ID).")

    args, extras = p.parse_known_args(argv)
    if len(extras) > 0:
        p.error("unrecognized arguments: " + " ".join(extras))

    has_genotype = bool(args.bfile or args.vcf or args.hmp or args.file)
    has_pheno = bool(args.pheno)
    if not bool(args.check):
        if (not has_pheno) and (not has_genotype):
            p.error(
                "the following arguments are required: -p/--pheno & "
                "(-vcf VCF | -hmp HMP | -file FILE | -bfile BFILE)"
            )
        if not has_pheno:
            p.error("the following arguments are required: -p/--pheno")
        if not has_genotype:
            p.error(
                "the following arguments are required: "
                "(-vcf VCF | -hmp HMP | -file FILE | -bfile BFILE)"
            )

    try:
        args.ncol = parse_zero_based_index_specs(args.ncol, label="-n/--n")
    except ValueError as e:
        p.error(str(e))
    if args.ncol is not None and len(args.ncol) > 1:
        p.error("`jx gblupbench` currently supports one trait at a time; pass a single `-n/--n`.")

    if int(args.cv) < 2:
        p.error("--cv must be >= 2.")
    if int(args.thread) < 1:
        p.error("--thread must be >= 1.")
    if int(args.chunksize) < 1:
        p.error("--chunksize must be >= 1.")
    return args


def _ensure_rscript() -> Optional[str]:
    r = shutil.which("Rscript")
    return r if r else None


def _check_r_package(rscript: str, pkg: str) -> tuple[bool, str]:
    expr = (
        "ok <- requireNamespace('" + pkg + "', quietly=TRUE);"
        "cat(ifelse(ok, 'OK', 'NO'))"
    )
    proc = subprocess.run(
        [rscript, "-e", expr],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    out = (proc.stdout or "").strip()
    if proc.returncode == 0 and out == "OK":
        return True, f"{pkg} available"
    detail = (proc.stderr or proc.stdout or "").strip()
    if not detail:
        detail = f"{pkg} unavailable"
    return False, detail


def _collect_env_checks(engines: list[str]) -> list[EnvCheck]:
    checks: list[EnvCheck] = []

    if "janusx" in engines:
        try:
            from janusx.pyBLUP.mlm import BLUP as _  # noqa: F401
            checks.append(EnvCheck("janusx.python", True, "OK"))
        except Exception as e:
            checks.append(EnvCheck("janusx.python", False, f"Import failed: {e}"))

    if any(e in {"sommer", "rrblup"} for e in engines):
        rscript = _ensure_rscript()
        if rscript is None:
            checks.append(EnvCheck("Rscript", False, "Rscript not found in PATH"))
        else:
            checks.append(EnvCheck("Rscript", True, f"OK: {rscript}"))
            if "sommer" in engines:
                ok, detail = _check_r_package(rscript, "sommer")
                checks.append(EnvCheck("R:sommer", ok, detail))
            if "rrblup" in engines:
                ok, detail = _check_r_package(rscript, "rrBLUP")
                checks.append(EnvCheck("R:rrBLUP", ok, detail))
    return checks


def _print_env_checks(engines: list[str], checks: list[EnvCheck]) -> bool:
    print(f"[CHECK] engines: {','.join(engines)}")
    ok_all = True
    for c in checks:
        st = "OK" if c.ok else "FAIL"
        print(f"  - {c.component:<12} {st:<4} {c.detail}")
        if not c.ok:
            ok_all = False
    return ok_all


def _build_python_engine_script(path: Path) -> None:
    code = r'''#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from janusx.pyBLUP.mlm import BLUP


def safe_metric(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p)
    if int(np.sum(mask)) < 3:
        return math.nan, math.nan, math.nan
    y = y[mask]
    p = p[mask]
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    ss_res = float(np.sum((y - p) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan
    try:
        pear = float(pearsonr(y, p).statistic)
    except Exception:
        pear = math.nan
    try:
        spear = float(spearmanr(y, p).statistic)
    except Exception:
        spear = math.nan
    return r2, pear, spear


def main() -> None:
    ap = argparse.ArgumentParser(prog='gblupbench_janusx_engine')
    ap.add_argument('--meta', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fold-out', required=True)
    ap.add_argument('--pred-out', required=True)
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text(encoding='utf-8'))
    m = int(meta['m'])
    n = int(meta['n'])
    geno_bin = str(meta['geno_bin'])
    sample_tsv = str(meta['sample_tsv'])

    vec = np.fromfile(geno_bin, dtype=np.float32)
    if vec.size != m * n:
        raise RuntimeError(f'Genotype size mismatch: got {vec.size}, expected {m*n}')
    M = vec.reshape((m, n))

    smp = pd.read_csv(sample_tsv, sep='\t')
    sid = smp['id'].astype(str).to_numpy()
    y = np.asarray(smp['y'].to_numpy(dtype=float), dtype=np.float64)
    fold = np.asarray(smp['fold'].to_numpy(dtype=int), dtype=np.int64)

    rows: list[dict[str, float | int]] = []
    pred_rows: list[pd.DataFrame] = []
    fold_ids = sorted(int(x) for x in np.unique(fold) if int(x) > 0)
    if len(fold_ids) == 0:
        raise RuntimeError('No valid fold IDs in sample table.')

    for fd in fold_ids:
        te = np.where(fold == int(fd))[0]
        tr = np.where(fold != int(fd))[0]
        if te.size == 0 or tr.size == 0:
            continue
        ytr = y[tr].reshape(-1, 1)
        yte = y[te].reshape(-1, 1)
        Mtr = M[:, tr]
        Mte = M[:, te]

        model = BLUP(ytr, Mtr, kinship=1)
        pred = np.asarray(model.predict(Mte), dtype=np.float64).reshape(-1)
        beta0 = float(np.asarray(getattr(model, 'beta', np.array([np.nan])), dtype=np.float64).reshape(-1)[0])
        gebv = pred - beta0 if math.isfinite(beta0) else np.full(pred.shape, np.nan, dtype=np.float64)
        r2, pear, spear = safe_metric(yte, pred)
        rows.append(
            {
                'fold': int(fd),
                'n_train': int(tr.size),
                'n_test': int(te.size),
                'r2': float(r2),
                'pearson': float(pear),
                'spearman': float(spear),
                'pve': float(getattr(model, 'pve', np.nan)),
            }
        )
        pred_rows.append(
            pd.DataFrame(
                {
                    'engine': 'janusx',
                    'sample_id': sid[te],
                    'fold': int(fd),
                    'y_true': y[te],
                    'y_pred': pred,
                    'gebv': gebv,
                    'intercept': beta0,
                    'pve_fold': float(getattr(model, 'pve', np.nan)),
                    'n_train': int(tr.size),
                    'n_test': int(te.size),
                }
            )
        )

    if len(rows) == 0:
        raise RuntimeError('No fold results produced.')

    df = pd.DataFrame(rows)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    fold_out = Path(args.fold_out)
    fold_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fold_out, sep='\t', index=False)
    pred_out = Path(args.pred_out)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_out, sep='\t', index=False)

    out = {
        'status': 'ok',
        'engine': 'janusx',
        'folds': int(df.shape[0]),
        'mean_r2': float(np.nanmean(df['r2'].to_numpy(dtype=float))),
        'mean_pearson': float(np.nanmean(df['pearson'].to_numpy(dtype=float))),
        'mean_spearman': float(np.nanmean(df['spearman'].to_numpy(dtype=float))),
        'mean_pve': float(np.nanmean(df['pve'].to_numpy(dtype=float))),
        'prediction_rows': int(pred_df.shape[0]),
        'prediction_file': str(pred_out),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
'''
    path.write_text(code, encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass


def _build_r_engine_script(path: Path) -> None:
    code = r'''#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
arg_val <- function(name, default = "") {
  idx <- which(args == name)
  if (length(idx) == 0) return(default)
  j <- idx[1] + 1
  if (j > length(args)) return(default)
  args[j]
}

engine <- tolower(trimws(arg_val("--engine", "")))
meta_path <- arg_val("--meta", "")
out_path <- arg_val("--out", "")
fold_out <- arg_val("--fold-out", "")
pred_out <- arg_val("--pred-out", "")

if (!nzchar(engine)) stop("--engine is required")
if (!nzchar(meta_path)) stop("--meta is required")
if (!nzchar(out_path)) stop("--out is required")
if (!nzchar(fold_out)) stop("--fold-out is required")
if (!nzchar(pred_out)) stop("--pred-out is required")

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Missing R package: jsonlite")
}

meta <- jsonlite::fromJSON(meta_path)
m <- as.integer(meta$m)
n <- as.integer(meta$n)

con <- file(meta$geno_bin, "rb")
on.exit(try(close(con), silent = TRUE), add = TRUE)
vec <- readBin(con, what = "numeric", n = m * n, size = 4, endian = "little")
if (length(vec) != (m * n)) {
  stop(sprintf("Genotype size mismatch: got %d, expected %d", length(vec), m * n))
}
M <- matrix(vec, nrow = m, ncol = n, byrow = TRUE)

smp <- read.table(meta$sample_tsv, header = TRUE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
if (!all(c("id", "y", "fold") %in% colnames(smp))) {
  stop("sample_tsv must contain columns: id, y, fold")
}
ids_all <- as.character(smp$id)
y <- as.numeric(smp$y)
fold <- as.integer(smp$fold)

safe_metric <- function(y_true, y_pred) {
  y <- as.numeric(y_true)
  p <- as.numeric(y_pred)
  idx <- which(is.finite(y) & is.finite(p))
  if (length(idx) < 3) {
    return(c(r2 = NA_real_, pearson = NA_real_, spearman = NA_real_))
  }
  y <- y[idx]
  p <- p[idx]
  ss_tot <- sum((y - mean(y))^2)
  ss_res <- sum((y - p)^2)
  r2 <- if (is.finite(ss_tot) && ss_tot > 0) (1.0 - ss_res / ss_tot) else NA_real_
  pear <- suppressWarnings(cor(y, p, method = "pearson"))
  spear <- suppressWarnings(cor(y, p, method = "spearman"))
  c(r2 = as.numeric(r2), pearson = as.numeric(pear), spearman = as.numeric(spear))
}

safe_solve <- function(A, b) {
  A <- as.matrix(A)
  b <- as.numeric(b)
  eps <- 1e-8
  for (k in 0:6) {
    out <- tryCatch(
      solve(A + diag(eps, nrow(A)), b),
      error = function(e) NULL
    )
    if (!is.null(out)) return(as.numeric(out))
    eps <- eps * 10.0
  }
  stop("Matrix solve failed")
}

extract_sommer_lambda <- function(fit, y_tr) {
  beta <- NA_real_
  beta <- tryCatch(as.numeric(fit$Beta$Estimate[1]), error = function(e) NA_real_)
  if (!is.finite(beta)) {
    beta <- tryCatch(as.numeric(fit$Beta[1, 1]), error = function(e) NA_real_)
  }
  if (!is.finite(beta)) beta <- mean(y_tr)

  vc <- tryCatch(as.data.frame(summary(fit)$varcomp), error = function(e) NULL)
  if (is.null(vc)) {
    vc <- tryCatch(as.data.frame(fit$sigma), error = function(e) NULL)
  }
  vu <- NA_real_
  ve <- NA_real_
  if (!is.null(vc) && nrow(vc) >= 1) {
    num_cols <- which(vapply(vc, is.numeric, logical(1)))
    if (length(num_cols) > 0) {
      vals <- as.numeric(vc[[num_cols[1]]])
      rn <- tolower(rownames(vc))
      idx_e <- which(grepl("units|residual|rcov", rn))[1]
      if (is.na(idx_e)) idx_e <- length(vals)
      idx_u <- setdiff(seq_along(vals), idx_e)[1]
      if (is.na(idx_u)) idx_u <- 1
      vu <- vals[idx_u]
      ve <- vals[idx_e]
    }
  }
  if (!is.finite(vu) || vu <= 0 || !is.finite(ve) || ve < 0) {
    lambda <- 1.0
    pve <- NA_real_
  } else {
    lambda <- ve / vu
    pve <- vu / (vu + ve)
  }
  list(beta = as.numeric(beta), lambda = as.numeric(lambda), pve = as.numeric(pve))
}

fit_predict_rrblup <- function(Ktr, Kte, y_tr) {
  if (!requireNamespace("rrBLUP", quietly = TRUE)) {
    stop("Missing R package rrBLUP")
  }
  fit <- rrBLUP::mixed.solve(y = as.numeric(y_tr), K = as.matrix(Ktr), SE = FALSE, return.Hinv = FALSE)
  beta <- as.numeric(fit$beta)[1]
  vu <- as.numeric(fit$Vu)[1]
  ve <- as.numeric(fit$Ve)[1]
  if (!is.finite(vu) || vu <= 0 || !is.finite(ve) || ve < 0) {
    lambda <- 1.0
    pve <- NA_real_
  } else {
    lambda <- ve / vu
    pve <- vu / (vu + ve)
  }
  alpha <- safe_solve(as.matrix(Ktr) + diag(lambda, nrow(Ktr)), as.numeric(y_tr) - beta)
  pred <- as.numeric(beta + as.matrix(Kte) %*% alpha)
  list(pred = pred, pve = as.numeric(pve), beta = as.numeric(beta))
}

fit_predict_sommer <- function(Ktr, Kte, y_tr) {
  if (!requireNamespace("sommer", quietly = TRUE)) {
    stop("Missing R package sommer")
  }
  ids <- as.character(seq_along(y_tr))
  Ktr <- as.matrix(Ktr)
  dimnames(Ktr) <- list(ids, ids)
  dat <- data.frame(id = factor(ids, levels = ids), y = as.numeric(y_tr))
  fit <- tryCatch(
    sommer::mmer(
      fixed = y ~ 1,
      random = ~ sommer::vsr(id, Gu = Ktr),
      rcov = ~ units,
      data = dat,
      verbose = FALSE
    ),
    error = function(e1) {
      sommer::mmer(
        y ~ 1,
        random = ~ sommer::vs(id, Gu = Ktr),
        rcov = ~ units,
        data = dat,
        verbose = FALSE
      )
    }
  )
  ex <- extract_sommer_lambda(fit, y_tr)
  alpha <- safe_solve(as.matrix(Ktr) + diag(ex$lambda, nrow(Ktr)), as.numeric(y_tr) - ex$beta)
  pred <- as.numeric(ex$beta + as.matrix(Kte) %*% alpha)
  list(pred = pred, pve = as.numeric(ex$pve), beta = as.numeric(ex$beta))
}

fold_ids <- sort(unique(fold))
fold_ids <- fold_ids[is.finite(fold_ids) & fold_ids > 0]
if (length(fold_ids) == 0) stop("No valid fold IDs in sample table")

rows <- list()
pred_rows <- list()
for (fd in fold_ids) {
  te <- which(fold == fd)
  tr <- which(fold != fd)
  if (length(te) == 0 || length(tr) == 0) next

  y_tr <- y[tr]
  y_te <- y[te]
  Mtr <- M[, tr, drop = FALSE]
  Mte <- M[, te, drop = FALSE]

  mu <- rowMeans(Mtr)
  Ctr <- sweep(Mtr, 1L, mu, "-")
  Cte <- sweep(Mte, 1L, mu, "-")
  var_sum <- sum(rowMeans(Ctr * Ctr))
  if (!is.finite(var_sum) || var_sum <= 0) var_sum <- 1e-12

  Ktr <- crossprod(Ctr) / var_sum
  Kte <- crossprod(Cte, Ctr) / var_sum

  out <- NULL
  if (engine == "rrblup") {
    out <- fit_predict_rrblup(Ktr, Kte, y_tr)
  } else if (engine == "sommer") {
    out <- fit_predict_sommer(Ktr, Kte, y_tr)
  } else {
    stop(sprintf("Unknown engine: %s", engine))
  }

  met <- safe_metric(y_te, out$pred)
  rows[[length(rows) + 1L]] <- data.frame(
    fold = as.integer(fd),
    n_train = as.integer(length(tr)),
    n_test = as.integer(length(te)),
    r2 = as.numeric(met[["r2"]]),
    pearson = as.numeric(met[["pearson"]]),
    spearman = as.numeric(met[["spearman"]]),
    pve = as.numeric(out$pve)
  )
  pred_rows[[length(pred_rows) + 1L]] <- data.frame(
    engine = engine,
    sample_id = ids_all[te],
    fold = as.integer(fd),
    y_true = as.numeric(y_te),
    y_pred = as.numeric(out$pred),
    gebv = as.numeric(out$pred - out$beta),
    intercept = rep(as.numeric(out$beta), length(te)),
    pve_fold = rep(as.numeric(out$pve), length(te)),
    n_train = rep(as.integer(length(tr)), length(te)),
    n_test = rep(as.integer(length(te)), length(te)),
    stringsAsFactors = FALSE
  )
}

if (length(rows) == 0) stop("No fold results produced")
df <- do.call(rbind, rows)
pred_df <- do.call(rbind, pred_rows)
dir.create(dirname(fold_out), recursive = TRUE, showWarnings = FALSE)
write.table(df, file = fold_out, sep = "\t", quote = FALSE, row.names = FALSE)
dir.create(dirname(pred_out), recursive = TRUE, showWarnings = FALSE)
write.table(pred_df, file = pred_out, sep = "\t", quote = FALSE, row.names = FALSE)

result <- list(
  status = "ok",
  engine = engine,
  folds = as.integer(nrow(df)),
  mean_r2 = as.numeric(mean(df$r2, na.rm = TRUE)),
  mean_pearson = as.numeric(mean(df$pearson, na.rm = TRUE)),
  mean_spearman = as.numeric(mean(df$spearman, na.rm = TRUE)),
  mean_pve = as.numeric(mean(df$pve, na.rm = TRUE)),
  prediction_rows = as.integer(nrow(pred_df)),
  prediction_file = pred_out
)
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
writeLines(jsonlite::toJSON(result, auto_unbox = TRUE, pretty = TRUE), con = out_path)
'''
    path.write_text(code, encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass


def _prepare_dataset(
    args: argparse.Namespace,
    *,
    bench_dir: Path,
) -> dict[str, Any]:
    gfile, _auto_prefix = _determine_genotype_source_from_args(args)
    gfile = str(gfile)

    ph, selected_trait = _read_pheno(safe_expanduser(args.pheno), args.ncol)
    ph_map = {str(k): float(v) for k, v in zip(ph["Taxa"].astype(str), ph["PHENO"].astype(float))}

    sample_ids, geno = _load_genotype_with_rust_gfreader(
        gfile,
        maf=float(args.maf),
        missing_rate=float(args.geno),
        chunk_size=int(args.chunksize),
    )
    sample_ids = np.asarray(sample_ids, dtype=str)

    keep_mask = np.fromiter((sid in ph_map for sid in sample_ids), dtype=bool, count=sample_ids.shape[0])
    if int(np.sum(keep_mask)) < int(args.cv):
        raise ValueError(
            f"Not enough overlapped samples after filtering: {int(np.sum(keep_mask))}. "
            f"Need at least --cv={int(args.cv)} samples."
        )

    ids_used = sample_ids[keep_mask]
    y_used = np.asarray([ph_map[sid] for sid in ids_used], dtype=np.float64)
    M_used = np.asarray(geno[:, keep_mask], dtype=np.float32)

    cv_splits = build_cv_splits(
        n_samples=int(y_used.shape[0]),
        n_splits=int(args.cv),
        seed=int(args.seed),
    )
    fold = np.zeros(y_used.shape[0], dtype=np.int32)
    for i, (test_idx, _train_idx) in enumerate(cv_splits, start=1):
        fold[np.asarray(test_idx, dtype=np.int64)] = int(i)
    if np.any(fold <= 0):
        raise RuntimeError("Failed to assign fold IDs for all samples.")

    input_dir = bench_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    sample_tsv = input_dir / "samples.tsv"
    pd.DataFrame(
        {
            "id": ids_used,
            "y": y_used,
            "fold": fold,
        }
    ).to_csv(sample_tsv, sep="\t", index=False)

    geno_bin = input_dir / "geno.f32.bin"
    np.asarray(M_used, dtype=np.float32).tofile(geno_bin)

    meta = {
        "trait": selected_trait,
        "m": int(M_used.shape[0]),
        "n": int(M_used.shape[1]),
        "cv": int(args.cv),
        "seed": int(args.seed),
        "geno_bin": str(geno_bin),
        "sample_tsv": str(sample_tsv),
    }
    meta_json = input_dir / "dataset.meta.json"
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "gfile": gfile,
        "trait": selected_trait,
        "meta": meta,
        "meta_json": meta_json,
        "sample_tsv": sample_tsv,
        "geno_bin": geno_bin,
    }


def _load_result_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_corr(a: Any, b: Any, *, method: str) -> float:
    sa = pd.Series(np.asarray(a, dtype=float).reshape(-1))
    sb = pd.Series(np.asarray(b, dtype=float).reshape(-1))
    mask = sa.notna() & sb.notna()
    if int(mask.sum()) < 3:
        return math.nan
    try:
        return float(sa[mask].corr(sb[mask], method=method))
    except Exception:
        return math.nan


def _finite_values(v: Any) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _safe_mean(v: Any) -> float:
    arr = _finite_values(v)
    return float(np.mean(arr)) if arr.size > 0 else math.nan


def _safe_std(v: Any) -> float:
    arr = _finite_values(v)
    return float(np.std(arr, ddof=1)) if arr.size >= 2 else math.nan


def _safe_mae(v: Any) -> float:
    arr = _finite_values(v)
    return float(np.mean(np.abs(arr))) if arr.size > 0 else math.nan


def _safe_rmse(v: Any) -> float:
    arr = _finite_values(v)
    return float(np.sqrt(np.mean(arr**2))) if arr.size > 0 else math.nan


def _safe_quantile(v: Any, q: float) -> float:
    arr = _finite_values(v)
    return float(np.quantile(arr, q)) if arr.size > 0 else math.nan


def _load_prediction_table(path: Path, engine: str) -> pd.DataFrame:
    df = _read_table_guess(path)
    if "sample_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "sample_id"})
    if "engine" not in df.columns:
        df["engine"] = str(engine)
    if "gebv" not in df.columns and {"y_pred", "intercept"}.issubset(df.columns):
        df["gebv"] = (
            pd.to_numeric(df["y_pred"], errors="coerce")
            - pd.to_numeric(df["intercept"], errors="coerce")
        )
    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].astype(str).str.strip()
    for col in ("fold", "y_true", "y_pred", "gebv", "intercept", "pve_fold", "n_train", "n_test"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _save_prediction_artifacts(
    out_dir: Path,
    prefix: str,
    rows: list[EngineRunResult],
) -> dict[str, str]:
    out: dict[str, str] = {}
    frames: list[pd.DataFrame] = []
    engine_order: list[str] = []
    seen: set[str] = set()

    for r in rows:
        if r.status != "ok" or not r.prediction_file:
            continue
        pred_path = Path(r.prediction_file)
        if not pred_path.exists():
            continue
        frames.append(_load_prediction_table(pred_path, r.engine))
        if r.engine not in seen:
            seen.add(r.engine)
            engine_order.append(r.engine)

    if len(frames) == 0:
        return out

    out_dir.mkdir(parents=True, exist_ok=True)
    long_df = pd.concat(frames, ignore_index=True, sort=False)

    for col, default in (
        ("engine", ""),
        ("sample_id", ""),
        ("fold", math.nan),
        ("y_true", math.nan),
        ("y_pred", math.nan),
        ("gebv", math.nan),
        ("intercept", math.nan),
        ("pve_fold", math.nan),
        ("n_train", math.nan),
        ("n_test", math.nan),
    ):
        if col not in long_df.columns:
            long_df[col] = default

    long_df = long_df[
        [
            "engine",
            "sample_id",
            "fold",
            "y_true",
            "y_pred",
            "gebv",
            "intercept",
            "pve_fold",
            "n_train",
            "n_test",
        ]
    ].copy()
    long_df = long_df.sort_values(["sample_id", "fold", "engine"], kind="stable").reset_index(drop=True)

    pred_tsv = out_dir / f"{prefix}.{_SUMMARY_TAG}.predictions.tsv"
    long_df.to_csv(pred_tsv, sep="\t", index=False)
    out["predictions_tsv"] = str(pred_tsv)

    y_true_df = (
        long_df.groupby(["sample_id", "fold"], as_index=False)["y_true"]
        .mean()
        .sort_values(["sample_id", "fold"], kind="stable")
    )
    wide_base = long_df[["sample_id", "fold", "engine", "y_pred", "gebv", "intercept"]].copy()
    wide_df = wide_base.pivot_table(
        index=["sample_id", "fold"],
        columns="engine",
        values=["y_pred", "gebv", "intercept"],
        aggfunc="first",
    )
    wide_df.columns = [f"{metric}__{engine}" for metric, engine in wide_df.columns.to_flat_index()]
    wide_df = y_true_df.merge(wide_df.reset_index(), on=["sample_id", "fold"], how="left")
    wide_tsv = out_dir / f"{prefix}.{_SUMMARY_TAG}.predictions.wide.tsv"
    wide_df.to_csv(wide_tsv, sep="\t", index=False)
    out["predictions_wide_tsv"] = str(wide_tsv)

    if len(engine_order) < 2:
        return out

    pair_sample_frames: list[pd.DataFrame] = []
    pair_records: list[dict[str, Any]] = []
    base_cols = ["sample_id", "fold", "y_true", "y_pred", "gebv", "intercept"]

    for i, engine_a in enumerate(engine_order):
        df_a = long_df[long_df["engine"] == engine_a][base_cols].copy()
        df_a = df_a.rename(
            columns={
                "y_true": "y_true_a",
                "y_pred": "y_pred_a",
                "gebv": "gebv_a",
                "intercept": "intercept_a",
            }
        )
        for engine_b in engine_order[i + 1 :]:
            df_b = long_df[long_df["engine"] == engine_b][base_cols].copy()
            df_b = df_b.rename(
                columns={
                    "y_true": "y_true_b",
                    "y_pred": "y_pred_b",
                    "gebv": "gebv_b",
                    "intercept": "intercept_b",
                }
            )
            mm = df_a.merge(df_b, on=["sample_id", "fold"], how="inner")
            if mm.empty:
                continue

            mm["engine_a"] = str(engine_a)
            mm["engine_b"] = str(engine_b)
            mm["y_true"] = pd.to_numeric(mm["y_true_a"], errors="coerce")
            mm["y_true_delta"] = (
                pd.to_numeric(mm["y_true_a"], errors="coerce")
                - pd.to_numeric(mm["y_true_b"], errors="coerce")
            )
            for metric in ("y_pred", "gebv", "intercept"):
                mm[f"diff_{metric}"] = (
                    pd.to_numeric(mm[f"{metric}_a"], errors="coerce")
                    - pd.to_numeric(mm[f"{metric}_b"], errors="coerce")
                )
                mm[f"abs_diff_{metric}"] = mm[f"diff_{metric}"].abs()

            pair_sample_frames.append(
                mm[
                    [
                        "engine_a",
                        "engine_b",
                        "sample_id",
                        "fold",
                        "y_true",
                        "y_pred_a",
                        "y_pred_b",
                        "diff_y_pred",
                        "abs_diff_y_pred",
                        "gebv_a",
                        "gebv_b",
                        "diff_gebv",
                        "abs_diff_gebv",
                        "intercept_a",
                        "intercept_b",
                        "diff_intercept",
                        "abs_diff_intercept",
                        "y_true_delta",
                    ]
                ].copy()
            )

            y_true_delta = _finite_values(mm["y_true_delta"])
            pair_records.append(
                {
                    "engine_a": str(engine_a),
                    "engine_b": str(engine_b),
                    "n_overlap": int(mm.shape[0]),
                    "max_abs_y_true_delta": float(np.max(np.abs(y_true_delta))) if y_true_delta.size > 0 else math.nan,
                    "pearson_y_pred": _safe_corr(mm["y_pred_a"], mm["y_pred_b"], method="pearson"),
                    "spearman_y_pred": _safe_corr(mm["y_pred_a"], mm["y_pred_b"], method="spearman"),
                    "bias_y_pred": _safe_mean(mm["diff_y_pred"]),
                    "sd_diff_y_pred": _safe_std(mm["diff_y_pred"]),
                    "mae_y_pred": _safe_mae(mm["diff_y_pred"]),
                    "rmse_y_pred": _safe_rmse(mm["diff_y_pred"]),
                    "q05_diff_y_pred": _safe_quantile(mm["diff_y_pred"], 0.05),
                    "median_diff_y_pred": _safe_quantile(mm["diff_y_pred"], 0.50),
                    "q95_diff_y_pred": _safe_quantile(mm["diff_y_pred"], 0.95),
                    "pearson_gebv": _safe_corr(mm["gebv_a"], mm["gebv_b"], method="pearson"),
                    "spearman_gebv": _safe_corr(mm["gebv_a"], mm["gebv_b"], method="spearman"),
                    "bias_gebv": _safe_mean(mm["diff_gebv"]),
                    "sd_diff_gebv": _safe_std(mm["diff_gebv"]),
                    "mae_gebv": _safe_mae(mm["diff_gebv"]),
                    "rmse_gebv": _safe_rmse(mm["diff_gebv"]),
                    "q05_diff_gebv": _safe_quantile(mm["diff_gebv"], 0.05),
                    "median_diff_gebv": _safe_quantile(mm["diff_gebv"], 0.50),
                    "q95_diff_gebv": _safe_quantile(mm["diff_gebv"], 0.95),
                }
            )

    if len(pair_sample_frames) == 0:
        return out

    pair_samples_df = pd.concat(pair_sample_frames, ignore_index=True, sort=False)
    pair_samples_tsv = out_dir / f"{prefix}.{_SUMMARY_TAG}.pred.compare.samples.tsv"
    pair_samples_df.to_csv(pair_samples_tsv, sep="\t", index=False)
    out["prediction_compare_samples_tsv"] = str(pair_samples_tsv)

    pair_summary_df = pd.DataFrame(pair_records)
    pair_summary_tsv = out_dir / f"{prefix}.{_SUMMARY_TAG}.pred.compare.tsv"
    pair_summary_md = out_dir / f"{prefix}.{_SUMMARY_TAG}.pred.compare.md"
    pair_summary_df.to_csv(pair_summary_tsv, sep="\t", index=False)
    out["prediction_compare_tsv"] = str(pair_summary_tsv)

    lines: list[str] = []
    lines.append(f"# Prediction Comparison Summary ({prefix})")
    lines.append("")
    lines.append("| engine_a | engine_b | n | pearson(y_pred) | pearson(GEBV) | MAE(y_pred) | MAE(GEBV) | RMSE(GEBV) | q05/q95 diff(GEBV) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for rec in pair_records:
        q05_gebv = _safe_float(rec.get("q05_diff_gebv"), math.nan)
        q95_gebv = _safe_float(rec.get("q95_diff_gebv"), math.nan)
        qlab = "NA" if (not math.isfinite(q05_gebv) or not math.isfinite(q95_gebv)) else f"{q05_gebv:.6f} / {q95_gebv:.6f}"

        def _fmt(x: Any) -> str:
            v = _safe_float(x, math.nan)
            return "NA" if not math.isfinite(v) else f"{v:.6f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    str(rec["engine_a"]),
                    str(rec["engine_b"]),
                    str(int(rec["n_overlap"])),
                    _fmt(rec["pearson_y_pred"]),
                    _fmt(rec["pearson_gebv"]),
                    _fmt(rec["mae_y_pred"]),
                    _fmt(rec["mae_gebv"]),
                    _fmt(rec["rmse_gebv"]),
                    qlab,
                ]
            )
            + " |"
        )
    pair_summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out["prediction_compare_md"] = str(pair_summary_md)

    return out


def _run_engine_janusx(
    *,
    args: argparse.Namespace,
    bench_dir: Path,
    meta_json: Path,
    py_script: Path,
) -> EngineRunResult:
    engine = "janusx"
    run_dir = bench_dir / "runs" / engine
    run_dir.mkdir(parents=True, exist_ok=True)

    result_json = run_dir / f"{args.prefix}.{engine}.result.json"
    fold_file = run_dir / f"{args.prefix}.{engine}.folds.tsv"
    pred_file = run_dir / f"{args.prefix}.{engine}.predictions.tsv"
    log_file = run_dir / f"{engine}.log"
    time_file = run_dir / f"{engine}.time"

    cmd = [
        sys.executable,
        str(py_script),
        "--meta",
        str(meta_json),
        "--out",
        str(result_json),
        "--fold-out",
        str(fold_file),
        "--pred-out",
        str(pred_file),
    ]

    env = os.environ.copy()
    py_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = py_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    rc, elapsed, rss = _run_timed(cmd, log_file=log_file, time_file=time_file, env=env)
    data = _load_result_json(result_json)

    if rc == 0 and str(data.get("status", "")).lower() == "ok":
        return EngineRunResult(
            engine=engine,
            status="ok",
            exit_code=rc,
            elapsed_sec=float(elapsed),
            peak_rss_kb=float(rss),
            mean_r2=_safe_float(data.get("mean_r2"), math.nan),
            mean_pearson=_safe_float(data.get("mean_pearson"), math.nan),
            mean_spearman=_safe_float(data.get("mean_spearman"), math.nan),
            mean_pve=_safe_float(data.get("mean_pve"), math.nan),
            folds=int(_safe_float(data.get("folds"), 0)),
            fold_file=str(fold_file) if fold_file.exists() else "",
            prediction_file=str(pred_file) if pred_file.exists() else "",
            result_json=str(result_json) if result_json.exists() else "",
            log_file=str(log_file),
            time_file=str(time_file),
            note="",
        )

    note = str(data.get("error", "")).strip()
    if not note:
        note = f"engine failed (exit={rc})"
    return EngineRunResult(
        engine=engine,
        status="fail",
        exit_code=rc,
        elapsed_sec=float(elapsed),
        peak_rss_kb=float(rss),
        mean_r2=math.nan,
        mean_pearson=math.nan,
        mean_spearman=math.nan,
        mean_pve=math.nan,
        folds=0,
        fold_file="",
        prediction_file=str(pred_file) if pred_file.exists() else "",
        result_json=str(result_json) if result_json.exists() else "",
        log_file=str(log_file),
        time_file=str(time_file),
        note=note,
    )


def _run_engine_r(
    *,
    args: argparse.Namespace,
    engine: str,
    bench_dir: Path,
    meta_json: Path,
    r_script: Path,
    rscript_bin: str,
) -> EngineRunResult:
    run_dir = bench_dir / "runs" / engine
    run_dir.mkdir(parents=True, exist_ok=True)

    result_json = run_dir / f"{args.prefix}.{engine}.result.json"
    fold_file = run_dir / f"{args.prefix}.{engine}.folds.tsv"
    pred_file = run_dir / f"{args.prefix}.{engine}.predictions.tsv"
    log_file = run_dir / f"{engine}.log"
    time_file = run_dir / f"{engine}.time"

    cmd = [
        rscript_bin,
        str(r_script),
        "--engine",
        str(engine),
        "--meta",
        str(meta_json),
        "--out",
        str(result_json),
        "--fold-out",
        str(fold_file),
        "--pred-out",
        str(pred_file),
    ]

    rc, elapsed, rss = _run_timed(cmd, log_file=log_file, time_file=time_file)
    data = _load_result_json(result_json)

    if rc == 0 and str(data.get("status", "")).lower() == "ok":
        return EngineRunResult(
            engine=engine,
            status="ok",
            exit_code=rc,
            elapsed_sec=float(elapsed),
            peak_rss_kb=float(rss),
            mean_r2=_safe_float(data.get("mean_r2"), math.nan),
            mean_pearson=_safe_float(data.get("mean_pearson"), math.nan),
            mean_spearman=_safe_float(data.get("mean_spearman"), math.nan),
            mean_pve=_safe_float(data.get("mean_pve"), math.nan),
            folds=int(_safe_float(data.get("folds"), 0)),
            fold_file=str(fold_file) if fold_file.exists() else "",
            prediction_file=str(pred_file) if pred_file.exists() else "",
            result_json=str(result_json) if result_json.exists() else "",
            log_file=str(log_file),
            time_file=str(time_file),
            note="",
        )

    note = str(data.get("error", "")).strip()
    if not note:
        note = f"engine failed (exit={rc})"
    return EngineRunResult(
        engine=engine,
        status="fail",
        exit_code=rc,
        elapsed_sec=float(elapsed),
        peak_rss_kb=float(rss),
        mean_r2=math.nan,
        mean_pearson=math.nan,
        mean_spearman=math.nan,
        mean_pve=math.nan,
        folds=0,
        fold_file="",
        prediction_file=str(pred_file) if pred_file.exists() else "",
        result_json=str(result_json) if result_json.exists() else "",
        log_file=str(log_file),
        time_file=str(time_file),
        note=note,
    )


def _save_summary(out_dir: Path, prefix: str, rows: list[EngineRunResult], cfg: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / f"{prefix}.{_SUMMARY_TAG}.tsv"
    md = out_dir / f"{prefix}.{_SUMMARY_TAG}.md"
    cfg_path = out_dir / f"{prefix}.{_SUMMARY_TAG}.config.json"

    records = []
    for r in rows:
        records.append(
            {
                "engine": r.engine,
                "status": r.status,
                "exit_code": int(r.exit_code),
                "elapsed_sec": float(r.elapsed_sec),
                "peak_rss_kb": float(r.peak_rss_kb),
                "mean_r2": float(r.mean_r2),
                "mean_pearson": float(r.mean_pearson),
                "mean_spearman": float(r.mean_spearman),
                "mean_pve": float(r.mean_pve),
                "folds": int(r.folds),
                "fold_file": r.fold_file,
                "prediction_file": r.prediction_file,
                "result_json": r.result_json,
                "log_file": r.log_file,
                "time_file": r.time_file,
                "note": r.note,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(tsv, sep="\t", index=False)

    lines: list[str] = []
    lines.append(f"# GBLUP Benchmark Summary ({prefix})")
    lines.append("")
    lines.append("| engine | status | elapsed(s) | peak_rss(MB) | mean_r2 | mean_pearson | mean_spearman | mean_pve | folds |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in records:
        elapsed = "NA" if not math.isfinite(_safe_float(r["elapsed_sec"])) else f"{float(r['elapsed_sec']):.3f}"
        rss = _safe_float(r["peak_rss_kb"], math.nan)
        rss_mb = "NA" if not math.isfinite(rss) else f"{rss / 1024.0:.1f}"

        def _fmt(x: Any) -> str:
            v = _safe_float(x, math.nan)
            return "NA" if not math.isfinite(v) else f"{v:.6f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["engine"]),
                    str(r["status"]),
                    elapsed,
                    rss_mb,
                    _fmt(r["mean_r2"]),
                    _fmt(r["mean_pearson"]),
                    _fmt(r["mean_spearman"]),
                    _fmt(r["mean_pve"]),
                    str(int(r["folds"])),
                ]
            )
            + " |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    engines = _parse_engines(args.engines)
    if len(engines) == 0:
        raise ValueError("No valid engines selected. Use --engines janusx,sommer,rrblup")

    if args.check:
        checks = _collect_env_checks(engines)
        ok = _print_env_checks(engines, checks)
        raise SystemExit(0 if ok else 1)

    out_dir = safe_expanduser(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_dir = out_dir / f"{args.prefix}.gblup_bench"
    bench_dir.mkdir(parents=True, exist_ok=True)

    data_info = _prepare_dataset(args, bench_dir=bench_dir)

    tmp_dir = bench_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    py_script = tmp_dir / "gblup_janusx_engine.py"
    r_script = tmp_dir / "gblup_r_engine.R"
    _build_python_engine_script(py_script)
    _build_r_engine_script(r_script)

    rscript_bin = _ensure_rscript()

    rows: list[EngineRunResult] = []
    for engine in engines:
        if engine == "janusx":
            rr = _run_engine_janusx(
                args=args,
                bench_dir=bench_dir,
                meta_json=Path(data_info["meta_json"]),
                py_script=py_script,
            )
            rows.append(rr)
            continue

        if engine in {"sommer", "rrblup"}:
            if rscript_bin is None:
                rows.append(
                    EngineRunResult(
                        engine=engine,
                        status="fail",
                        exit_code=127,
                        elapsed_sec=math.nan,
                        peak_rss_kb=math.nan,
                        mean_r2=math.nan,
                        mean_pearson=math.nan,
                        mean_spearman=math.nan,
                        mean_pve=math.nan,
                        folds=0,
                        fold_file="",
                        prediction_file="",
                        result_json="",
                        log_file="",
                        time_file="",
                        note="Rscript not found in PATH",
                    )
                )
                continue
            rr = _run_engine_r(
                args=args,
                engine=engine,
                bench_dir=bench_dir,
                meta_json=Path(data_info["meta_json"]),
                r_script=r_script,
                rscript_bin=rscript_bin,
            )
            rows.append(rr)
            continue

    cfg = {
        "genotype_input": str(data_info["gfile"]),
        "phenotype_input": str(safe_expanduser(args.pheno)),
        "trait": str(data_info["trait"]),
        "engines": engines,
        "maf": float(args.maf),
        "geno": float(args.geno),
        "chunksize": int(args.chunksize),
        "threads": int(args.thread),
        "cv": int(args.cv),
        "seed": int(args.seed),
        "n_samples": int(data_info["meta"]["n"]),
        "n_snps": int(data_info["meta"]["m"]),
        "benchmark_dir": str(bench_dir),
    }
    artifact_paths = _save_prediction_artifacts(bench_dir / "summary", args.prefix, rows)
    if len(artifact_paths) > 0:
        cfg["artifacts"] = artifact_paths
    _save_summary(bench_dir / "summary", args.prefix, rows, cfg)

    if not args.keep_temp:
        for p in [py_script, r_script]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    summary_tsv = bench_dir / "summary" / f"{args.prefix}.{_SUMMARY_TAG}.tsv"
    summary_md = bench_dir / "summary" / f"{args.prefix}.{_SUMMARY_TAG}.md"
    print(f"[DONE] summary table: {summary_tsv}")
    print(f"[DONE] summary markdown: {summary_md}")
    if "predictions_tsv" in artifact_paths:
        print(f"[DONE] sample predictions: {artifact_paths['predictions_tsv']}")
    if "prediction_compare_tsv" in artifact_paths:
        print(f"[DONE] prediction compare: {artifact_paths['prediction_compare_tsv']}")


if __name__ == "__main__":
    main()
