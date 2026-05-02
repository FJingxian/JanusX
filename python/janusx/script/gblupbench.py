#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GBLUP benchmark launcher for JanusX / sommer / rrBLUP / BLUPF90 / BLUPF90-APY / HIBLUP.

Key features:
  - One-command benchmark for multiple engines:
      * janusx (JanusX GBLUP backend)
      * janusxrrblup (JanusX rrBLUP backend)
      * sommer (R package)
      * rrblup (R package rrBLUP)
      * blupf90 / blupf90apy (external BLUPF90 family)
      * hiblup (external HIBLUP)
  - Unified data filtering / trait selection / CV splits across engines.
  - Per-sample out-of-fold prediction tables for each engine.
  - Automatic pairwise prediction comparison when multiple engines are selected.
  - Runtime + peak RSS profiling via system `time` wrapper.
  - Temporary engine scripts are generated on-the-fly (Python/R).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import resource
except Exception:  # pragma: no cover
    resource = None  # type: ignore[assignment]

import numpy as np
import pandas as pd

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:  # pragma: no cover
    _threadpool_limits = None  # type: ignore[assignment]


def _bootstrap_repo_python_path() -> None:
    if __package__:
        return
    here = Path(__file__).resolve()
    py_root = here.parents[2]
    if str(py_root) not in sys.path:
        sys.path.insert(0, str(py_root))


_bootstrap_repo_python_path()

from janusx.script._common.pathcheck import safe_expanduser  # noqa: E402
from janusx.script._common.threads import (  # noqa: E402
    apply_blas_thread_env,
    detect_effective_threads,
    maybe_warn_non_openblas,
    require_openblas_by_default,
)
from janusx.script._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog  # noqa: E402
from janusx.script._common.colspec import parse_zero_based_index_specs  # noqa: E402
from janusx.script._common.status import CliStatus, stdout_is_tty  # noqa: E402
from janusx.script import sim as sim_mod  # noqa: E402
from janusx.gfreader import convert_genotypes, inspect_genotype_file, load_genotype_chunks, save_genotype_streaming  # noqa: E402
from janusx.gs.workflow import build_cv_splits  # noqa: E402


_SUMMARY_TAG = "gblupbench"
_DEFAULT_ENGINES_RUN = "janusx,janusxrrblup,sommer,rrblup"
_DEFAULT_ENGINES_CHECK_ALL = "janusx,janusxrrblup,sommer,rrblup,blupf90,blupf90apy,hiblup"
_ENGINE_LIST_TEXT = "janusx,janusxrrblup,sommer,rrblup,blupf90,blupf90apy,hiblup"
_PREGSF90_APY_MODE_CACHE: dict[str, str] = {}
_BLAS_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


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


@contextlib.contextmanager
def _native_thread_limit_ctx(limit: int):
    lim = max(1, int(limit))
    if _threadpool_limits is None:
        yield
        return
    try:
        with _threadpool_limits(limits=lim, user_api="blas"):
            yield
    except TypeError:
        with _threadpool_limits(limits=lim):
            yield


@contextlib.contextmanager
def _thread_env_stage_ctx(
    *,
    blas_threads: Optional[int],
    rayon_threads: Optional[int],
):
    updates: dict[str, str] = {}
    if blas_threads is not None:
        bt = str(max(1, int(blas_threads)))
        for key in _BLAS_THREAD_ENV_KEYS:
            updates[key] = bt
    if rayon_threads is not None:
        updates["RAYON_NUM_THREADS"] = str(max(1, int(rayon_threads)))

    old_env: dict[str, Optional[str]] = {}
    for key, val in updates.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = str(val)

    try:
        if blas_threads is not None:
            with _native_thread_limit_ctx(int(blas_threads)):
                yield
        else:
            yield
    finally:
        for key, old_val in old_env.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(old_val)


@contextlib.contextmanager
def _gblupbench_prep_stage_ctx(threads: int):
    """
    Preprocessing stage policy:
    - Rust decode/filter path uses Rayon = --thread
    - BLAS/OpenMP is pinned to 1 to avoid oversubscription
    """
    t = max(1, int(threads))
    with _thread_env_stage_ctx(blas_threads=1, rayon_threads=t):
        yield


def _status_enabled() -> bool:
    return bool(stdout_is_tty())


def _fold_status_desc(engine: str, fold_idx: int, fold_total: int, fold_id: int, step: str) -> str:
    return (
        f"Running {str(engine).upper()} fold {int(fold_idx)}/{int(max(1, fold_total))} "
        f"(id={int(fold_id)}) [{str(step)}]..."
    )


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


def _nanmean_or_nan(values: Any) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    mask = np.isfinite(arr)
    if int(np.sum(mask)) == 0:
        return math.nan
    return float(np.mean(arr[mask]))


def _safe_int_token(v: Any) -> Optional[int]:
    s = str(v).strip()
    if s == "":
        return None
    try:
        return int(s)
    except Exception:
        pass
    try:
        x = float(s)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    xr = int(round(x))
    return xr if abs(x - float(xr)) < 1e-8 else None


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
    def _supports(time_bin: str, flag: str) -> bool:
        try:
            rc = subprocess.run(
                [str(time_bin), str(flag), "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                text=False,
            ).returncode
        except Exception:
            return False
        return int(rc) == 0

    gtime = shutil.which("gtime")
    if gtime and _supports(gtime, "-v"):
        return [gtime, "-v"]

    sys_time = Path("/usr/bin/time")
    if sys_time.exists():
        if _supports(str(sys_time), "-v"):
            return [str(sys_time), "-v"]
        if _supports(str(sys_time), "-l"):
            return [str(sys_time), "-l"]
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


def _memory_limit_kb(limit_mem_gb: Optional[float]) -> float:
    if limit_mem_gb is None:
        return math.nan
    try:
        gb = float(limit_mem_gb)
    except Exception:
        return math.nan
    if (not math.isfinite(gb)) or gb <= 0:
        return math.nan
    return float(gb * 1024.0 * 1024.0)


def _proc_tree_rss_kb(root_pid: int) -> float:
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return math.nan
    if proc.returncode != 0:
        return math.nan

    children: dict[int, list[int]] = {}
    rss_kb: dict[int, float] = {}
    for raw in str(proc.stdout or "").splitlines():
        parts = raw.strip().split()
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss = float(parts[2])
        except Exception:
            continue
        rss_kb[pid] = rss
        children.setdefault(ppid, []).append(pid)

    total = 0.0
    stack = [int(root_pid)]
    seen: set[int] = set()
    while len(stack) > 0:
        pid = int(stack.pop())
        if pid in seen:
            continue
        seen.add(pid)
        total += float(rss_kb.get(pid, 0.0))
        stack.extend(children.get(pid, []))
    return float(total)


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(int(proc.pid), signal.SIGKILL)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _build_memory_limit_preexec(limit_kb: float) -> Optional[Any]:
    if resource is None:
        return None
    if not math.isfinite(limit_kb) or limit_kb <= 0:
        return None
    limit_bytes = int(float(limit_kb) * 1024.0)
    if limit_bytes <= 0:
        return None

    def _preexec() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except Exception:
            pass
        if hasattr(resource, "RLIMIT_RSS"):
            try:
                resource.setrlimit(resource.RLIMIT_RSS, (limit_bytes, limit_bytes))
            except Exception:
                pass

    return _preexec


def _run_timed(
    cmd: list[str],
    *,
    log_file: Path,
    time_file: Path,
    env: Optional[dict[str, str]] = None,
    cwd: Optional[Path] = None,
    input_text: Optional[str] = None,
    append_log: bool = False,
    limit_mem_gb: Optional[float] = None,
) -> tuple[int, float, float]:
    """Returns: (exit_code, elapsed_sec, peak_rss_kb)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    time_file.parent.mkdir(parents=True, exist_ok=True)
    time_tool = _detect_time_tool()
    limit_kb = _memory_limit_kb(limit_mem_gb)
    use_mem_limit = math.isfinite(limit_kb) and (float(limit_kb) > 0.0)
    t0 = time.time()
    rc_final = 1
    runtime_peak_rss_kb = math.nan
    with log_file.open(("a" if append_log else "w"), encoding="utf-8") as lf:
        if time_tool:
            full = [*time_tool, "-o", str(time_file.resolve()), *cmd]
        else:
            full = list(cmd)
        if not use_mem_limit:
            proc = subprocess.run(
                full,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                check=False,
                text=True,
                input=input_text,
            )
            rc_final = int(proc.returncode)
        else:
            popen_kw: dict[str, Any] = {
                "stdout": lf,
                "stderr": subprocess.STDOUT,
                "cwd": (str(cwd) if cwd is not None else None),
                "env": env,
                "text": True,
                "start_new_session": True,
            }
            preexec = _build_memory_limit_preexec(limit_kb)
            if preexec is not None:
                popen_kw["preexec_fn"] = preexec
            if input_text is not None:
                popen_kw["stdin"] = subprocess.PIPE
            proc = subprocess.Popen(full, **popen_kw)
            if input_text is not None and proc.stdin is not None:
                try:
                    proc.stdin.write(str(input_text))
                    proc.stdin.flush()
                except Exception:
                    pass
                try:
                    proc.stdin.close()
                except Exception:
                    pass

            mem_killed = False
            while True:
                rss_now = _proc_tree_rss_kb(int(proc.pid))
                if math.isfinite(rss_now):
                    runtime_peak_rss_kb = (
                        float(rss_now)
                        if not math.isfinite(runtime_peak_rss_kb)
                        else max(float(runtime_peak_rss_kb), float(rss_now))
                    )
                rc_poll = proc.poll()
                if rc_poll is not None:
                    rc_final = int(rc_poll)
                    break
                if math.isfinite(rss_now) and float(rss_now) > float(limit_kb):
                    mem_killed = True
                    try:
                        lf.write(
                            (
                                "\n[gblupbench] memory limit exceeded: "
                                f"{float(rss_now) / 1024.0 / 1024.0:.6f} GB > "
                                f"{float(limit_mem_gb):.6f} GB; killing process tree.\n"
                            )
                        )
                        lf.flush()
                    except Exception:
                        pass
                    _kill_process_group(proc)
                    try:
                        rc_final = int(proc.wait(timeout=3))
                    except Exception:
                        rc_final = 137
                    break
                time.sleep(0.2)
            if mem_killed:
                rc_final = 137
    wall = max(time.time() - t0, 0.0)
    if not time_tool:
        rss_txt = "NA"
        if math.isfinite(runtime_peak_rss_kb):
            rss_txt = f"{float(runtime_peak_rss_kb):.0f}"
        time_file.write_text(
            "\n".join(
                [
                    f"Elapsed (wall clock) time (h:mm:ss or m:ss): {wall:.6f}",
                    f"Maximum resident set size (kbytes): {rss_txt}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    elapsed, rss = _parse_time_file(time_file)
    if not math.isfinite(elapsed):
        elapsed = wall
    if math.isfinite(runtime_peak_rss_kb):
        rss = float(runtime_peak_rss_kb) if not math.isfinite(rss) else max(float(rss), float(runtime_peak_rss_kb))
    if use_mem_limit and math.isfinite(rss) and float(rss) > float(limit_kb):
        rc_final = 137
        try:
            with log_file.open("a", encoding="utf-8") as lf:
                lf.write(
                    (
                        "\n[gblupbench] memory limit exceeded (post-check): "
                        f"{float(rss) / 1024.0 / 1024.0:.6f} GB > "
                        f"{float(limit_mem_gb):.6f} GB.\n"
                    )
                )
        except Exception:
            pass
    return int(rc_final), float(elapsed), float(rss)


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
    if t in {"janusxrrblup", "jxrrblup", "janusrrblup", "jxrr", "janusxrr"}:
        return "janusxrrblup"
    if t in {"janusx", "jx", "janus"}:
        return "janusx"
    if t in {"sommer"}:
        return "sommer"
    if t in {"rrblup", "rrb", "rr"}:
        return "rrblup"
    if t in {"blupf90", "blup90", "f90"}:
        return "blupf90"
    if t in {"blupf90apy", "blup90apy", "f90apy", "ssblupapy", "ssgblupapy"}:
        return "blupf90apy"
    if t in {"hiblup", "hib", "hbl"}:
        return "hiblup"
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


def _option_seen(tokens: list[str], names: set[str]) -> bool:
    for tk in tokens:
        head = str(tk).split("=", 1)[0].strip()
        if head in names:
            return True
    return False


def _dev_help_requested(argv: Optional[list[str]] = None) -> bool:
    tokens = list(sys.argv[1:] if argv is None else argv)
    return ("-dev" in tokens) or ("--dev" in tokens)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    show_dev_help = _dev_help_requested(argv)
    tokens = list(sys.argv[1:] if argv is None else argv)
    p = CliArgumentParser(
        prog="jx gblupbench",
        formatter_class=cli_help_formatter(),
        description="Benchmark GBLUP engines: JanusX(GBLUP) / JanusX(rrBLUP) / sommer / rrBLUP / BLUPF90 / BLUPF90-APY / HIBLUP.",
        epilog=minimal_help_epilog(
            [
                "jx gblupbench -bfile example_prefix -p pheno.tsv -n 0",
                "jx gblupbench -vcf example.vcf.gz -p pheno.tsv -n 0 --engines janusx,janusxrrblup,sommer,rrblup",
                "jx gblupbench -bfile example_prefix -p pheno.tsv -n 0 --engines janusx,janusxrrblup,sommer,rrblup,blupf90,blupf90apy,hiblup",
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
    optional_group.add_argument(
        "-limit-mem",
        "--limit-mem",
        default=None,
        type=float,
        help=(
            "Per-command memory limit in GB. "
            "When exceeded, the running benchmark subprocess is killed and marked as failed."
        ),
    )
    optional_group.add_argument("--cv", default=5, type=int, help="CV folds (default: %(default)s).")
    optional_group.add_argument(
        "--run-folds",
        default=1,
        type=int,
        help="How many CV folds to execute (default: %(default)s). Remaining folds are skipped.",
    )
    optional_group.add_argument("--seed", default=42, type=int, help="Random seed (default: %(default)s).")
    optional_group.add_argument(
        "-limit-predtrain",
        "--limit-predtrain",
        "-limit-train",
        "--limit-train",
        default=0,
        type=int,
        dest="limit_predtrain",
        help=(
            "Forward to JanusX gs engine only: cap train-set predictions per CV fold "
            "(0 disables fold train predictions; default: %(default)s)."
        ),
    )
    optional_group.add_argument(
        "--engines",
        default=_DEFAULT_ENGINES_RUN,
        type=str,
        help=f"Comma list: {_ENGINE_LIST_TEXT}.",
    )
    optional_group.add_argument(
        "--check",
        action="store_true",
        default=False,
        help=(
            "Check selected engines and run a small end-to-end smoke test generated by `jx sim`, then exit. "
            "In this mode, genotype/pheno inputs are optional. "
            "If --engines is omitted, all engines are checked."
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
    args._engines_explicit = _option_seen(tokens, {"--engines"})

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
    if int(args.run_folds) < 1:
        p.error("--run-folds must be >= 1.")
    if int(args.run_folds) > int(args.cv):
        p.error("--run-folds must be <= --cv.")
    if int(args.thread) < 1:
        p.error("--thread must be >= 1.")
    if int(args.chunksize) < 1:
        p.error("--chunksize must be >= 1.")
    if args.limit_mem is not None:
        try:
            lm = float(args.limit_mem)
        except Exception:
            p.error("--limit-mem must be a positive number in GB.")
        if (not math.isfinite(lm)) or (lm <= 0.0):
            p.error("--limit-mem must be > 0 (GB).")
    if int(args.limit_predtrain) < 0:
        p.error("--limit-predtrain/--limit-train must be >= 0.")
    return args


def _ensure_rscript() -> Optional[str]:
    r = shutil.which("Rscript")
    return r if r else None


def _is_executable_file(path: Path) -> bool:
    try:
        pp = path.expanduser().resolve()
        return pp.is_file() and os.access(str(pp), os.X_OK)
    except Exception:
        return False


def _iter_probe_names(default_names: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in default_names:
        name = str(raw).strip()
        if name == "":
            continue
        variants = [name]
        if not name.endswith("+"):
            variants.append(f"{name}+")
        if not name.lower().endswith(".exe"):
            variants.append(f"{name}.exe")
        for cand in variants:
            key = cand.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
    return out


def _resolve_bin(default_names: list[str], env_name: str) -> Optional[str]:
    override = str(os.environ.get(env_name, "")).strip()
    if override:
        found = shutil.which(override)
        if found:
            return found
        override_path = Path(override)
        if _is_executable_file(override_path):
            return str(override_path.expanduser().resolve())
        return None

    probe_names = _iter_probe_names(default_names)
    for name in probe_names:
        found = shutil.which(name)
        if found:
            return found

    # Case-insensitive PATH scan fallback for tools shipped with unusual casing.
    path_raw = str(os.environ.get("PATH", "")).strip()
    if path_raw != "":
        wanted = {name.lower() for name in probe_names}
        for folder in path_raw.split(os.pathsep):
            root = Path(folder).expanduser()
            if (not root.exists()) or (not root.is_dir()):
                continue
            try:
                for entry in root.iterdir():
                    if entry.name.lower() not in wanted:
                        continue
                    if _is_executable_file(entry):
                        return str(entry.resolve())
            except Exception:
                continue
    return None


def _ensure_hiblup() -> Optional[str]:
    return _resolve_bin(["hiblup", "HIBLUP", "hiblup-cli"], "JX_GBLUPBENCH_HIBLUP_BIN")


def _ensure_pregsf90() -> Optional[str]:
    return _resolve_bin(["preGSf90", "pregsf90"], "JX_GBLUPBENCH_PREGSF90_BIN")


def _ensure_renumf90() -> Optional[str]:
    return _resolve_bin(["renumf90"], "JX_GBLUPBENCH_RENUMF90_BIN")


def _ensure_blupf90() -> Optional[str]:
    return _resolve_bin(["blupf90", "airemlf90", "remlf90"], "JX_GBLUPBENCH_BLUPF90_BIN")


def _detect_pregsf90_apy_mode(pregsf90_bin: str) -> str:
    override = str(os.environ.get("JX_GBLUPBENCH_APY_MODE", "auto")).strip().lower()
    if override in {"legacy", "modern"}:
        return override

    key = str(Path(str(pregsf90_bin)).expanduser())
    cached = _PREGSF90_APY_MODE_CACHE.get(key)
    if cached in {"legacy", "modern"}:
        return str(cached)

    mode = "modern"
    try:
        proc = subprocess.run(
            [str(pregsf90_bin), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=8,
        )
        txt = str(proc.stdout or "").strip()
        m = re.search(r"(?i)(?:pregsf90|blupf90\+)\s*(?:ver\.?\s*)?([0-9]+(?:\.[0-9]+)?)", txt)
        if m is not None:
            try:
                ver = float(str(m.group(1)))
                if math.isfinite(ver) and ver < 2.0:
                    mode = "legacy"
            except Exception:
                pass
    except Exception:
        pass

    _PREGSF90_APY_MODE_CACHE[key] = mode
    return mode


def _build_apy_option_lines(*, n_levels: int, apy_mode: str, apy_type: int = 2) -> list[str]:
    raw = str(os.environ.get("JX_GBLUPBENCH_APY_OPTION", "")).strip()
    if raw != "":
        return [raw if raw.lower().startswith("option ") else f"OPTION {raw}"]

    if str(apy_mode).strip().lower() == "legacy":
        raw_pos = str(os.environ.get("JX_GBLUPBENCH_APY_POS", "")).strip()
        try:
            pos = max(1, int(raw_pos)) if raw_pos != "" else 1
        except Exception:
            pos = 1

        cond = str(os.environ.get("JX_GBLUPBENCH_APY_CONDITION", "")).strip()
        if cond == "":
            raw_core = str(os.environ.get("JX_GBLUPBENCH_APY_CORE_N", "")).strip()
            if raw_core == "":
                # Heuristic default core size:
                # - for small/medium datasets, sqrt(n) can be too small and harms agreement,
                #   so use ~30% core (at least 100);
                # - for larger datasets, keep sqrt(n)-style growth.
                nlv = max(1, int(n_levels))
                if nlv <= 5000:
                    core_n = int(round(0.30 * float(nlv)))
                    core_n = max(100, core_n)
                else:
                    core_n = int(round(math.sqrt(float(nlv))))
                    core_n = max(200, core_n)
            else:
                try:
                    core_n = int(raw_core)
                except Exception:
                    nlv = max(1, int(n_levels))
                    if nlv <= 5000:
                        core_n = int(round(0.30 * float(nlv)))
                        core_n = max(100, core_n)
                    else:
                        core_n = int(round(math.sqrt(float(nlv))))
                        core_n = max(200, core_n)
            core_n = max(1, min(int(n_levels) - 1, int(core_n)))
            cond = f"core.le.{core_n}"

        apy_type_i = 2
        try:
            apy_type_i = max(1, int(apy_type))
        except Exception:
            apy_type_i = 2
        return [f"OPTION apy {int(apy_type_i)} {int(pos)} {cond}"]

    return ["OPTION apy 2"]


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

    if any(e in {"janusx", "janusxrrblup"} for e in engines):
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

    if "hiblup" in engines:
        hiblup_bin = _ensure_hiblup()
        checks.append(
            EnvCheck(
                "HIBLUP",
                hiblup_bin is not None,
                f"OK: {hiblup_bin}" if hiblup_bin is not None else "hiblup not found in PATH",
            )
        )

    if any(e in {"blupf90", "blupf90apy"} for e in engines):
        pregs_bin = _ensure_pregsf90()
        blup_bin = _ensure_blupf90()
        checks.append(
            EnvCheck(
                "BLUPF90:pregs",
                pregs_bin is not None,
                (
                    f"OK: {pregs_bin}"
                    if pregs_bin is not None
                    else (
                        "preGSf90/pregsf90/preGSf90+ not found in PATH "
                        "(or set JX_GBLUPBENCH_PREGSF90_BIN)"
                    )
                ),
            )
        )
        checks.append(
            EnvCheck(
                "BLUPF90:solver",
                blup_bin is not None,
                (
                    f"OK: {blup_bin}"
                    if blup_bin is not None
                    else "blupf90/airemlf90/remlf90 not found in PATH"
                ),
            )
        )
    return checks


def _run_smoke_checks_with_sim(engines: list[str]) -> list[EnvCheck]:
    checks: list[EnvCheck] = []
    if len(engines) == 0:
        return checks

    py_root = str(Path(__file__).resolve().parents[2])
    env = os.environ.copy()
    env["PYTHONPATH"] = py_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    with tempfile.TemporaryDirectory(prefix="jx_gblupbench_check_") as td:
        td_path = Path(td)
        sim_prefix = td_path / "sim" / "tiny"
        sim_prefix.parent.mkdir(parents=True, exist_ok=True)
        sim_gen_log = td_path / "sim_generation.log"
        try:
            # Tiny yet non-trivial dataset for end-to-end engine smoke checks.
            # nsnp_k=1 -> 1,000 SNPs; n=32; no NA for deterministic CV behavior.
            sim_out = io.StringIO()
            sim_err = io.StringIO()
            with contextlib.redirect_stdout(sim_out), contextlib.redirect_stderr(sim_err):
                sim_mod.main(
                    [
                        "1",
                        "32",
                        str(sim_prefix),
                        "--chunk-size",
                        "256",
                        "--seed",
                        "2026",
                        "--na-rate",
                        "0.0",
                        "--pve",
                        "0.3",
                        "--ve",
                        "1.0",
                        "--trait-name",
                        "test",
                        "--structure",
                        "unrelated",
                    ]
                )
            sim_gen_log.write_text(sim_out.getvalue() + sim_err.getvalue(), encoding="utf-8")
        except Exception as e:
            detail = f"sim dataset generation failed: {e}"
            if sim_gen_log.exists():
                detail = f"{detail} (log: {sim_gen_log})"
            checks.append(EnvCheck("SMOKE:sim", False, detail))
            for engine in engines:
                checks.append(EnvCheck(f"SMOKE:{engine}", False, "skipped (sim failed)"))
            return checks

        pheno = Path(f"{sim_prefix}.pheno.txt")
        if not pheno.exists():
            checks.append(EnvCheck("SMOKE:sim", False, f"sim output missing: {pheno}"))
            for engine in engines:
                checks.append(EnvCheck(f"SMOKE:{engine}", False, "skipped (sim output missing)"))
            return checks
        checks.append(EnvCheck("SMOKE:sim", True, f"dataset ready: {sim_prefix}"))

        out_dir = td_path / "bench"
        prefix = "check_smoke"
        smoke_log = td_path / "smoke.log"
        smoke_time = td_path / "smoke.time"
        cmd = [
            sys.executable,
            "-m",
            "janusx.script.gblupbench",
            "--bfile",
            str(sim_prefix),
            "--pheno",
            str(pheno),
            "--engines",
            ",".join(engines),
            "--out",
            str(out_dir),
            "--prefix",
            prefix,
            "--cv",
            "2",
            "--seed",
            "2026",
            "--thread",
            "1",
            "--chunksize",
            "256",
            "--maf",
            "0.0",
            "--geno",
            "1.0",
            "--keep-temp",
        ]
        rc, _elapsed, _rss = _run_timed(
            cmd,
            log_file=smoke_log,
            time_file=smoke_time,
            env=env,
            cwd=td_path,
        )

        summary_tsv = out_dir / f"{prefix}.gblup_bench" / "summary" / f"{prefix}.{_SUMMARY_TAG}.tsv"
        if not summary_tsv.exists():
            tail = ""
            try:
                lines = smoke_log.read_text(encoding="utf-8", errors="ignore").splitlines()
                if len(lines) > 0:
                    tail = " | ".join(lines[-3:])
            except Exception:
                tail = ""
            detail = f"smoke summary missing (exit={rc})"
            if tail:
                detail = f"{detail}: {tail}"
            for engine in engines:
                checks.append(EnvCheck(f"SMOKE:{engine}", False, detail))
            return checks

        try:
            df = pd.read_csv(summary_tsv, sep="\t")
        except Exception as e:
            for engine in engines:
                checks.append(EnvCheck(f"SMOKE:{engine}", False, f"failed to parse smoke summary: {e}"))
            return checks

        for engine in engines:
            dfe = df[df["engine"].astype(str) == str(engine)]
            if dfe.shape[0] == 0:
                checks.append(EnvCheck(f"SMOKE:{engine}", False, "engine missing in smoke summary"))
                continue
            row = dfe.iloc[0]
            status = str(row.get("status", "")).strip().lower()
            note = str(row.get("note", "")).strip()
            elapsed_sec = _safe_float(row.get("elapsed_sec"), math.nan)
            if status == "ok":
                elapsed_text = "NA" if not math.isfinite(elapsed_sec) else f"{elapsed_sec:.3f}s"
                checks.append(EnvCheck(f"SMOKE:{engine}", True, f"run ok ({elapsed_text})"))
            else:
                detail = note if note else f"status={status or 'unknown'}"
                checks.append(EnvCheck(f"SMOKE:{engine}", False, detail))
    return checks


def _print_env_checks(engines: list[str], checks: list[EnvCheck]) -> bool:
    print(f"[CHECK] engines: {','.join(engines)}")
    ok_all = True
    comp_w = max(12, max((len(str(c.component)) for c in checks), default=12))
    status_w = 4
    term_cols = int(shutil.get_terminal_size((120, 20)).columns)
    for c in checks:
        st = "OK" if c.ok else "FAIL"
        detail = str(c.detail).strip()
        if detail.upper().startswith("OK:"):
            detail = detail.split(":", 1)[1].strip()
        elif c.ok and detail.upper() == "OK":
            detail = "available"
        prefix = f"  - {str(c.component):<{comp_w}} {st:<{status_w}} "
        width = max(16, term_cols - len(prefix))
        wrapped = textwrap.wrap(
            detail if detail != "" else "-",
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        if len(wrapped) == 0:
            wrapped = ["-"]
        print(f"{prefix}{wrapped[0]}")
        cont = " " * len(prefix)
        for line in wrapped[1:]:
            print(f"{cont}{line}")
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
    engines: Optional[list[str]] = None,
) -> dict[str, Any]:
    gfile, _auto_prefix = _determine_genotype_source_from_args(args)
    gfile = str(gfile)

    ph, selected_trait = _read_pheno(safe_expanduser(args.pheno), args.ncol)
    ph_map = {str(k): float(v) for k, v in zip(ph["Taxa"].astype(str), ph["PHENO"].astype(float))}

    sample_ids, _n_sites = inspect_genotype_file(gfile)
    sample_ids = np.asarray(sample_ids, dtype=str)

    keep_mask = np.fromiter((sid in ph_map for sid in sample_ids), dtype=bool, count=sample_ids.shape[0])
    if int(np.sum(keep_mask)) < int(args.cv):
        raise ValueError(
            f"Not enough overlapped samples after filtering: {int(np.sum(keep_mask))}. "
            f"Need at least --cv={int(args.cv)} samples."
        )

    ids_used = sample_ids[keep_mask]
    y_used = np.asarray([ph_map[sid] for sid in ids_used], dtype=np.float64)

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
    run_folds = int(max(1, int(getattr(args, "run_folds", 1))))
    fold_ids_all = sorted(int(x) for x in np.unique(fold) if int(x) > 0)
    if run_folds < len(fold_ids_all):
        keep_folds = set(fold_ids_all[:run_folds])
        fold = np.asarray([int(v) if int(v) in keep_folds else 0 for v in fold], dtype=np.int32)

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

    filtered_plink_prefix = input_dir / "filtered_plink"

    eng_set = {str(e).strip().lower() for e in (engines or []) if str(e).strip() != ""}
    use_streaming_plink_only = bool(len(eng_set) > 0 and eng_set.issubset({"janusx", "janusxrrblup", "hiblup"}))

    if use_streaming_plink_only:
        keep_idx = np.flatnonzero(keep_mask).astype(np.int64, copy=False)
        if keep_idx.size != int(ids_used.shape[0]):
            raise RuntimeError(
                f"Sample index mismatch: keep_idx={int(keep_idx.size)} ids_used={int(ids_used.shape[0])}"
            )
        keep_idx_list = [int(i) for i in keep_idx.tolist()]
        n_snps_written = 0
        prep_mode = "streaming_plink_only"
        prep_note = ""

        # Fast path: Rust-side threaded conversion/filtering/writing.
        # Fallback to Python streaming if current extension build does not
        # expose the extended convert_genotypes signature yet.
        try:
            conv_stats = convert_genotypes(
                str(gfile),
                str(filtered_plink_prefix),
                out_fmt="plink",
                progress_callback=None,
                progress_every=0,
                threads=int(args.thread),
                snps_only=True,
                maf=float(args.maf),
                geno=float(args.geno),
                impute=True,
                model="add",
                het=0.02,
                sample_ids=None,
                sample_indices=keep_idx_list,
            )
            n_snps_written = int(getattr(conv_stats, "n_sites_written", 0))
            prep_mode = "rust_convert_plink"
        except TypeError as ex:
            prep_note = f"rust_convert_fallback_typeerror: {ex}"
        except Exception as ex:
            prep_note = f"rust_convert_fallback_error: {ex}"

        if int(n_snps_written) <= 0:
            stream_chunks = load_genotype_chunks(
                gfile,
                chunk_size=int(args.chunksize),
                maf=float(args.maf),
                missing_rate=float(args.geno),
                impute=True,
                sample_indices=keep_idx_list,
            )
            n_snps_written = 0

            def _iter_plink_chunks_stream() -> Iterable[tuple[np.ndarray, list[Any]]]:
                nonlocal n_snps_written
                for geno_chunk, sites in stream_chunks:
                    arr = np.asarray(geno_chunk, dtype=np.float32)
                    if arr.ndim != 2 or arr.shape[0] == 0:
                        continue
                    if int(arr.shape[1]) != int(ids_used.shape[0]):
                        raise RuntimeError(
                            f"Streamed chunk sample mismatch: got {int(arr.shape[1])}, expected {int(ids_used.shape[0])}"
                        )
                    n_snps_written += int(arr.shape[0])
                    yield arr, list(sites)

            save_genotype_streaming(
                str(filtered_plink_prefix),
                list(map(str, ids_used)),
                _iter_plink_chunks_stream(),
                fmt="plink",
                total_snps=None,
                desc="Writing filtered PLINK benchmark dataset (streaming)",
            )
            prep_mode = "streaming_plink_only"
        if int(n_snps_written) <= 0:
            raise ValueError(
                "No SNPs left after Rust-side filtering. "
                "Please relax --maf/--geno thresholds."
            )

        meta = {
            "trait": selected_trait,
            "m": int(n_snps_written),
            "n": int(ids_used.shape[0]),
            "cv": int(args.cv),
            "run_folds": int(run_folds),
            "seed": int(args.seed),
            "sample_tsv": str(sample_tsv.resolve()),
            "filtered_plink_prefix": str(filtered_plink_prefix.resolve()),
            "prep_mode": str(prep_mode),
        }
        if str(prep_note).strip() != "":
            meta["prep_note"] = str(prep_note)
        meta_json = input_dir / "dataset.meta.json"
        meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "gfile": gfile,
            "trait": selected_trait,
            "meta": meta,
            "meta_json": meta_json,
            "sample_tsv": sample_tsv.resolve(),
            "geno_bin": None,
            "filtered_plink_prefix": filtered_plink_prefix.resolve(),
        }

    chunks = load_genotype_chunks(
        gfile,
        chunk_size=int(args.chunksize),
        maf=float(args.maf),
        missing_rate=float(args.geno),
        impute=True,
    )
    blocks: list[np.ndarray] = []
    sites_all: list[Any] = []
    for geno_chunk, sites in chunks:
        arr = np.asarray(geno_chunk, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        blocks.append(arr)
        sites_all.extend(list(sites))
    if len(blocks) == 0:
        raise ValueError(
            "No SNPs left after Rust-side filtering. "
            "Please relax --maf/--geno thresholds."
        )
    geno = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    M_used = np.asarray(geno[:, keep_mask], dtype=np.float32)
    geno_bin = input_dir / "geno.f32.bin"
    np.asarray(M_used, dtype=np.float32).tofile(geno_bin)

    def _iter_plink_chunks() -> Iterable[tuple[np.ndarray, list[Any]]]:
        step = max(1, int(args.chunksize))
        for st in range(0, int(M_used.shape[0]), step):
            ed = min(st + step, int(M_used.shape[0]))
            yield np.asarray(M_used[st:ed], dtype=np.float32), list(sites_all[st:ed])

    save_genotype_streaming(
        str(filtered_plink_prefix),
        list(map(str, ids_used)),
        _iter_plink_chunks(),
        fmt="plink",
        total_snps=int(M_used.shape[0]),
        desc="Writing filtered PLINK benchmark dataset",
    )

    meta = {
        "trait": selected_trait,
        "m": int(M_used.shape[0]),
        "n": int(M_used.shape[1]),
        "cv": int(args.cv),
        "run_folds": int(run_folds),
        "seed": int(args.seed),
        "geno_bin": str(geno_bin.resolve()),
        "sample_tsv": str(sample_tsv.resolve()),
        "filtered_plink_prefix": str(filtered_plink_prefix.resolve()),
    }
    meta_json = input_dir / "dataset.meta.json"
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "gfile": gfile,
        "trait": selected_trait,
        "meta": meta,
        "meta_json": meta_json,
        "sample_tsv": sample_tsv.resolve(),
        "geno_bin": geno_bin.resolve(),
        "filtered_plink_prefix": filtered_plink_prefix.resolve(),
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


def _safe_prediction_metrics(y_true: Any, y_pred: Any) -> tuple[float, float, float]:
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
    pear = _safe_corr(y, p, method="pearson")
    spear = _safe_corr(y, p, method="spearman")
    return r2, pear, spear


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
    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale artifacts from previous runs under the same prefix so that
    # summary/compare files always reflect the current invocation.
    stale_paths = [
        out_dir / f"{prefix}.{_SUMMARY_TAG}.predictions.tsv",
        out_dir / f"{prefix}.{_SUMMARY_TAG}.predictions.wide.tsv",
        out_dir / f"{prefix}.{_SUMMARY_TAG}.pred.compare.samples.tsv",
        out_dir / f"{prefix}.{_SUMMARY_TAG}.pred.compare.tsv",
        out_dir / f"{prefix}.{_SUMMARY_TAG}.pred.compare.md",
    ]
    for p in stale_paths:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

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


def _write_time_summary(path: Path, elapsed_sec: float, peak_rss_kb: float) -> None:
    path.write_text(
        "\n".join(
            [
                f"Elapsed (wall clock) time (h:mm:ss or m:ss): {elapsed_sec:.6f}",
                "Maximum resident set size (kbytes): "
                + ("NA" if not math.isfinite(peak_rss_kb) else f"{peak_rss_kb:.0f}"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_tool_with_parfile_fallback(
    tool_bin: str,
    parfile: Path,
    *,
    log_file: Path,
    time_file: Path,
    cwd: Path,
    env: Optional[dict[str, str]] = None,
    limit_mem_gb: Optional[float] = None,
) -> tuple[int, float, float]:
    attempts = [
        ([tool_bin, str(parfile.name)], None),
        ([tool_bin], str(parfile.name) + "\n"),
    ]
    total_elapsed = 0.0
    max_rss = math.nan
    last_rc = 1
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception:
            pass
    for idx, (cmd, input_text) in enumerate(attempts, start=1):
        if idx > 1:
            with log_file.open("a", encoding="utf-8") as lf:
                lf.write(f"\n[retry {idx}] {' '.join(cmd)}\n")
        attempt_time = time_file.parent / f"{time_file.stem}.try{idx}{time_file.suffix}"
        rc, elapsed, rss = _run_timed(
            cmd,
            log_file=log_file,
            time_file=attempt_time,
            cwd=cwd,
            env=env,
            input_text=input_text,
            append_log=(idx > 1),
            limit_mem_gb=limit_mem_gb,
        )
        if math.isfinite(elapsed):
            total_elapsed += float(elapsed)
        if math.isfinite(rss):
            max_rss = float(rss) if not math.isfinite(max_rss) else max(float(max_rss), float(rss))
        last_rc = int(rc)
        if rc == 0:
            break
    _write_time_summary(time_file, total_elapsed, max_rss)
    return last_rc, float(total_elapsed), float(max_rss)


def _write_blupf90_snp_file(path: Path, sample_ids: np.ndarray, geno: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for j, sid in enumerate(np.asarray(sample_ids, dtype=str).reshape(-1)):
            row = np.rint(np.asarray(geno[:, j], dtype=np.float64)).astype(np.int16, copy=False)
            row = np.clip(row, 0, 2)
            fh.write(f"{sid} {''.join(str(int(x)) for x in row.tolist())}\n")


def _write_blupf90_fixed_snp_file(path: Path, numeric_ids: np.ndarray, geno: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(numeric_ids, dtype=np.int64).reshape(-1)
    width = max(len(str(int(x))) for x in ids) + 2
    with path.open("w", encoding="utf-8") as fh:
        for j, iid in enumerate(ids):
            row = np.rint(np.asarray(geno[:, j], dtype=np.float64)).astype(np.int16, copy=False)
            row = np.clip(row, 0, 2)
            fh.write(f"{str(int(iid)).ljust(width)}{''.join(str(int(x)) for x in row.tolist())}\n")


def _choose_hiblup_gebv_column(df: pd.DataFrame) -> str:
    preferred = ["GA", "HA", "PA", "GEBV", "EBV", "BV"]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns[1:]:
        low = str(col).strip().lower()
        if low in {"r", "e", "resid", "residual"}:
            continue
        if np.issubdtype(df[col].dtype, np.number):
            return str(col)
    raise ValueError("Unable to detect breeding-value column in HIBLUP .rand output.")


def _parse_hiblup_beta(beta_path: Path, fallback: float) -> float:
    if not beta_path.exists():
        return float(fallback)
    try:
        df = _read_table_guess(beta_path)
    except Exception:
        return float(fallback)
    for _, row in df.iterrows():
        text = " ".join(str(x).strip().lower() for x in row.tolist())
        nums = [float(x) for x in pd.to_numeric(pd.Series(row.tolist()), errors="coerce").dropna().tolist()]
        if ("intercept" in text or " mu" in f" {text}" or text.startswith("mu ")) and len(nums) > 0:
            return float(nums[-1])
    nums_all = pd.to_numeric(df.stack(), errors="coerce").dropna().tolist()
    return float(nums_all[0]) if len(nums_all) > 0 else float(fallback)


def _parse_hiblup_pve(vars_path: Path) -> float:
    if not vars_path.exists():
        return math.nan
    try:
        df = _read_table_guess(vars_path)
    except Exception:
        return math.nan
    lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
    item_col = lower_map.get("item", str(df.columns[0]))
    h2_col = lower_map.get("h2")
    if h2_col is None:
        return math.nan
    for _, row in df.iterrows():
        item = str(row.get(item_col, "")).strip().lower()
        if item.endswith(".ga") or item == "ga" or item.endswith("ga"):
            return _safe_float(row.get(h2_col), math.nan)
    return _safe_float(df[h2_col].iloc[0], math.nan) if int(df.shape[0]) > 0 else math.nan


def _extract_gs_cv_mean_pve(log_text: str, method: str = "GBLUP") -> float:
    """
    Parse JanusX `gs` CLI log and extract mean h2/PVE for a target method from
    the CV table:
      Fold  Method  Pearsonr  Spearmanr  R2  h2/PVE  time(secs)
    """
    lines = str(log_text or "").splitlines()
    target = str(method).strip().lower()
    latest_values: list[float] = []

    for i, line in enumerate(lines):
        low = str(line).strip().lower()
        if ("fold" not in low) or ("method" not in low) or ("h2/pve" not in low):
            continue

        values: list[float] = []
        saw_rows = False
        for raw in lines[i + 1 :]:
            s = str(raw).strip()
            if s == "":
                if saw_rows:
                    break
                continue
            if set(s) <= {"-"}:
                continue
            parts = s.split()
            if len(parts) < 7:
                if saw_rows:
                    break
                continue
            fold_i = _safe_int_token(parts[0])
            if fold_i is None:
                if saw_rows:
                    break
                continue
            saw_rows = True
            method_i = str(parts[1]).strip().lower()
            if method_i != target:
                continue
            pve_i = _safe_float(parts[5], math.nan)
            if math.isfinite(pve_i):
                values.append(float(pve_i))
        if len(values) > 0:
            latest_values = values

    if len(latest_values) == 0:
        return math.nan
    return _nanmean_or_nan(np.asarray(latest_values, dtype=float))


def _extract_gs_cv_rows(log_text: str, method: str = "GBLUP") -> list[dict[str, float | int]]:
    """
    Parse JanusX `gs` CLI log and extract per-fold rows for a target method from
    the CV table:
      Fold  Method  Pearsonr  Spearmanr  R2  h2/PVE  time(secs)
    """
    lines = str(log_text or "").splitlines()
    target = str(method).strip().lower()
    latest_rows: list[dict[str, float | int]] = []

    for i, line in enumerate(lines):
        low = str(line).strip().lower()
        if ("fold" not in low) or ("method" not in low) or ("h2/pve" not in low):
            continue

        rows: list[dict[str, float | int]] = []
        saw_rows = False
        for raw in lines[i + 1 :]:
            s = str(raw).strip()
            if s == "":
                if saw_rows:
                    break
                continue
            if set(s) <= {"-"}:
                continue
            parts = s.split()
            if len(parts) < 7:
                if saw_rows:
                    break
                continue
            fold_i = _safe_int_token(parts[0])
            if fold_i is None:
                if saw_rows:
                    break
                continue
            saw_rows = True
            method_i = str(parts[1]).strip().lower()
            if method_i != target:
                continue
            rows.append(
                {
                    "fold": int(fold_i),
                    "pearson": _safe_float(parts[2], math.nan),
                    "spearman": _safe_float(parts[3], math.nan),
                    "r2": _safe_float(parts[4], math.nan),
                    "pve": _safe_float(parts[5], math.nan),
                    "time_sec": _safe_float(parts[6], math.nan),
                }
            )
        if len(rows) > 0:
            latest_rows = rows

    return latest_rows


def _build_external_engine_result(
    *,
    engine: str,
    args: argparse.Namespace,
    run_dir: Path,
    elapsed: float,
    rss: float,
    fold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    log_file: Path,
    time_file: Path,
    note: str = "",
) -> EngineRunResult:
    result_json = run_dir / f"{args.prefix}.{engine}.result.json"
    fold_file = run_dir / f"{args.prefix}.{engine}.folds.tsv"
    pred_file = run_dir / f"{args.prefix}.{engine}.predictions.tsv"

    if int(fold_df.shape[0]) == 0 or int(pred_df.shape[0]) == 0:
        if note:
            result_json.write_text(json.dumps({"status": "fail", "engine": engine, "error": note}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return EngineRunResult(
            engine=engine,
            status="fail",
            exit_code=1,
            elapsed_sec=float(elapsed),
            peak_rss_kb=float(rss),
            mean_r2=math.nan,
            mean_pearson=math.nan,
            mean_spearman=math.nan,
            mean_pve=math.nan,
            folds=0,
            fold_file="",
            prediction_file="",
            result_json=str(result_json) if result_json.exists() else "",
            log_file=str(log_file),
            time_file=str(time_file),
            note=(note or "No fold results produced."),
        )

    fold_df.to_csv(fold_file, sep="\t", index=False)
    pred_df.to_csv(pred_file, sep="\t", index=False)
    result = {
        "status": "ok",
        "engine": engine,
        "folds": int(fold_df.shape[0]),
        "mean_r2": _nanmean_or_nan(fold_df["r2"].to_numpy(dtype=float)),
        "mean_pearson": _nanmean_or_nan(fold_df["pearson"].to_numpy(dtype=float)),
        "mean_spearman": _nanmean_or_nan(fold_df["spearman"].to_numpy(dtype=float)),
        "mean_pve": _nanmean_or_nan(fold_df["pve"].to_numpy(dtype=float)),
        "prediction_rows": int(pred_df.shape[0]),
        "prediction_file": str(pred_file),
    }
    result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return EngineRunResult(
        engine=engine,
        status="ok",
        exit_code=0,
        elapsed_sec=float(elapsed),
        peak_rss_kb=float(rss),
        mean_r2=_safe_float(result.get("mean_r2"), math.nan),
        mean_pearson=_safe_float(result.get("mean_pearson"), math.nan),
        mean_spearman=_safe_float(result.get("mean_spearman"), math.nan),
        mean_pve=_safe_float(result.get("mean_pve"), math.nan),
        folds=int(_safe_float(result.get("folds"), 0)),
        fold_file=str(fold_file),
        prediction_file=str(pred_file),
        result_json=str(result_json),
        log_file=str(log_file),
        time_file=str(time_file),
        note=str(note or ""),
    )


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_FLOAT_TOKEN_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", str(text))


def _janusx_method_for_engine(engine: str) -> tuple[str, str, str]:
    e = str(engine).strip().lower()
    if e == "janusx":
        return "-GBLUP", "GBLUP", "JANUSX GBLUP"
    if e == "janusxrrblup":
        return "-rrBLUP", "rrBLUP", "JANUSX rrBLUP"
    raise ValueError(f"Unsupported JanusX engine token: {engine}")


def _select_gs_prediction_column(df: pd.DataFrame, method_name: str) -> str:
    cols = [str(c) for c in df.columns]
    if len(cols) < 2:
        raise ValueError("JanusX prediction table must have at least two columns.")
    target = str(method_name).strip().lower()
    for c in cols[1:]:
        if str(c).strip().lower() == target:
            return str(c)
    for c in cols[1:]:
        if str(c).strip().lower() in {"pred", "y_pred", "prediction", "gebv"}:
            return str(c)
    return str(cols[1])


def _extract_gs_method_detail_rows(log_text: str, method_name: str) -> list[tuple[str, str]]:
    lines = [_strip_ansi(x) for x in str(log_text or "").splitlines()]
    target = str(method_name).strip().lower()
    start_idx = -1
    for i, line in enumerate(lines):
        low = str(line).strip().lower()
        if ("finished" in low) and (target in low):
            start_idx = i
    if start_idx < 0:
        return []
    rows: list[tuple[str, str]] = []
    for raw in lines[start_idx + 1 :]:
        s = str(raw).strip()
        if s == "":
            if len(rows) > 0:
                break
            continue
        low = s.lower()
        if ("finished" in low) and ("..." in low):
            break
        if ("fold" in low) and ("method" in low) and ("h2/pve" in low):
            break
        if set(s) <= {"-"}:
            if len(rows) > 0:
                break
            continue
        m = re.match(r"^(.+?)\s{2,}(.+)$", s)
        if m is None:
            if len(rows) > 0:
                break
            continue
        key = str(m.group(1)).strip().rstrip(":")
        val = str(m.group(2)).strip()
        if (key == "") or (val == ""):
            continue
        rows.append((key, val))
    return rows


def _extract_pve_from_detail_rows(rows: list[tuple[str, str]]) -> float:
    for key, val in rows:
        low_key = str(key).strip().lower()
        if ("pve" not in low_key) and ("h2" not in low_key):
            continue
        nums = _FLOAT_TOKEN_RE.findall(str(val))
        if len(nums) == 0:
            continue
        vv = _safe_float(nums[0], math.nan)
        if math.isfinite(vv):
            return float(vv)
    return math.nan


def _run_engine_janusx(
    *,
    args: argparse.Namespace,
    bench_dir: Path,
    meta_json: Path,
    engine: str = "janusx",
) -> EngineRunResult:
    engine = str(engine).strip().lower()
    method_flag, method_name, status_name = _janusx_method_for_engine(engine)
    run_dir = (bench_dir / "runs" / engine).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / f"{engine}.log"
    time_file = run_dir / f"{engine}.time"
    log_file.write_text("", encoding="utf-8")

    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    sample_df = pd.read_csv(str(meta["sample_tsv"]), sep="\t")
    ids = sample_df["id"].astype(str).to_numpy()
    y = sample_df["y"].to_numpy(dtype=float)
    fold = pd.to_numeric(sample_df["fold"], errors="coerce").fillna(0).to_numpy(dtype=int)
    bfile = str(meta["filtered_plink_prefix"])
    trait = "PHENO"
    fold_ids = sorted(int(x) for x in np.unique(fold) if int(x) > 0)
    if len(fold_ids) == 0:
        note = "No positive fold IDs found in sample table."
        _write_time_summary(time_file, 0.0, math.nan)
        return EngineRunResult(
            engine=engine,
            status="fail",
            exit_code=1,
            elapsed_sec=0.0,
            peak_rss_kb=math.nan,
            mean_r2=math.nan,
            mean_pearson=math.nan,
            mean_spearman=math.nan,
            mean_pve=math.nan,
            folds=0,
            fold_file="",
            prediction_file="",
            result_json="",
            log_file=str(log_file),
            time_file=str(time_file),
            note=note,
        )

    env = os.environ.copy()
    py_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = py_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if str(env.get("JX_THREAD_POLICY", "")).strip() == "":
        env["JX_THREAD_POLICY"] = "auto"

    total_elapsed = 0.0
    max_rss = math.nan
    fold_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    param_rows: list[dict[str, Any]] = []

    def _fail(note: str) -> EngineRunResult:
        _write_time_summary(time_file, float(total_elapsed), float(max_rss))
        return EngineRunResult(
            engine=engine,
            status="fail",
            exit_code=1,
            elapsed_sec=float(total_elapsed),
            peak_rss_kb=float(max_rss),
            mean_r2=math.nan,
            mean_pearson=math.nan,
            mean_spearman=math.nan,
            mean_pve=math.nan,
            folds=0,
            fold_file="",
            prediction_file="",
            result_json="",
            log_file=str(log_file),
            time_file=str(time_file),
            note=note,
        )

    with CliStatus(
        f"Running {status_name} holdout (folds={len(fold_ids)}/{int(args.cv)})...",
        enabled=_status_enabled(),
    ) as task:
        for i, fd in enumerate(fold_ids, start=1):
            task.desc = f"Running {status_name} holdout [fold {i}/{len(fold_ids)}]..."
            test_mask = (fold == int(fd))
            n_test = int(np.sum(test_mask))
            n_train = int(np.sum(fold != int(fd)))
            if n_test <= 0 or n_train <= 0:
                continue

            y_fold = np.asarray(y, dtype=float).copy()
            y_fold[test_mask] = np.nan
            pheno_file = (run_dir / f"pheno.fold{fd}.tsv").resolve()
            pd.DataFrame({"id": ids, "PHENO": y_fold}).to_csv(
                pheno_file,
                sep="\t",
                index=False,
                na_rep="NA",
            )

            fold_prefix = f"{engine}.fold{fd}"
            fold_log = run_dir / f"{engine}.fold{fd}.log"
            fold_time = run_dir / f"{engine}.fold{fd}.time"
            cmd = [
                sys.executable,
                "-m",
                "janusx.gs.workflow",
                "-bfile",
                str(bfile),
                "-p",
                str(pheno_file),
                method_flag,
                "-force-fast",
                "-o",
                str(run_dir),
                "-prefix",
                fold_prefix,
                "--limit-predtrain",
                str(int(args.limit_predtrain)),
                "-t",
                str(int(args.thread)),
                "-maf",
                "0.0",
                "-geno",
                "1.0",
            ]
            rc, elapsed_i, rss_i = _run_timed(
                cmd,
                log_file=fold_log,
                time_file=fold_time,
                env=env,
                cwd=run_dir,
                limit_mem_gb=args.limit_mem,
            )
            total_elapsed += max(0.0, _safe_float(elapsed_i, 0.0))
            if math.isfinite(_safe_float(rss_i, math.nan)):
                max_rss = _safe_float(rss_i, math.nan) if not math.isfinite(max_rss) else max(float(max_rss), float(rss_i))

            fold_log_text = fold_log.read_text(encoding="utf-8", errors="ignore") if fold_log.exists() else ""
            with log_file.open("a", encoding="utf-8") as agg:
                agg.write(f"\n===== {engine} fold={fd} =====\n")
                agg.write(fold_log_text)
                if not fold_log_text.endswith("\n"):
                    agg.write("\n")

            if int(rc) != 0:
                note = f"{method_name} fold={fd} failed (exit={int(rc)})."
                if int(rc) == 137 and args.limit_mem is not None:
                    note = f"{method_name} fold={fd} failed: memory limit exceeded ({float(args.limit_mem):g} GB)."
                task.fail(note)
                return _fail(note)

            detail_rows = _extract_gs_method_detail_rows(fold_log_text, method_name)
            for k, v in detail_rows:
                param_rows.append({"fold": int(fd), "param": str(k), "value": str(v)})
            pve_fold = _extract_pve_from_detail_rows(detail_rows)

            pred_path = run_dir / f"{fold_prefix}.{trait}.gs.tsv"
            if not pred_path.exists():
                note = f"Missing prediction file: {pred_path}"
                task.fail(note)
                return _fail(note)
            pred_raw = _read_table_guess(pred_path)
            pred_col = _select_gs_prediction_column(pred_raw, method_name)
            id_col = str(pred_raw.columns[0])
            pred_one = pd.DataFrame(
                {
                    "sample_id": pred_raw[id_col].astype(str).str.strip(),
                    "y_pred": pd.to_numeric(pred_raw[pred_col], errors="coerce"),
                }
            )
            pred_one = pred_one.drop_duplicates("sample_id", keep="first")
            truth_df = pd.DataFrame(
                {
                    "sample_id": ids[test_mask].astype(str),
                    "y_true": y[test_mask],
                }
            )
            merged = truth_df.merge(pred_one, on="sample_id", how="left")
            r2, pear, spear = _safe_prediction_metrics(merged["y_true"], merged["y_pred"])

            fold_rows.append(
                {
                    "fold": int(fd),
                    "n_train": int(n_train),
                    "n_test": int(n_test),
                    "r2": float(r2),
                    "pearson": float(pear),
                    "spearman": float(spear),
                    "pve": float(pve_fold),
                }
            )
            pred_frames.append(
                pd.DataFrame(
                    {
                        "engine": str(engine),
                        "sample_id": merged["sample_id"].astype(str),
                        "fold": int(fd),
                        "y_true": pd.to_numeric(merged["y_true"], errors="coerce"),
                        "y_pred": pd.to_numeric(merged["y_pred"], errors="coerce"),
                        "gebv": math.nan,
                        "intercept": math.nan,
                        "pve_fold": float(pve_fold),
                        "n_train": int(n_train),
                        "n_test": int(n_test),
                    }
                )
            )

        if len(fold_rows) == 0 or len(pred_frames) == 0:
            note = f"{status_name} produced no fold result."
            task.fail(note)
            return _fail(note)
        task.complete(f"Running {status_name} holdout ...Finished")

    _write_time_summary(time_file, float(total_elapsed), float(max_rss))
    note = f"per-fold holdout via janusx.gs.workflow ({method_name}); run_folds={len(fold_ids)}"
    if len(param_rows) > 0:
        params_path = run_dir / f"{args.prefix}.{engine}.fold_params.tsv"
        pd.DataFrame(param_rows).to_csv(params_path, sep="\t", index=False)
        note += f"; fold_params={params_path}"

    fold_df = pd.DataFrame(fold_rows).sort_values("fold", kind="stable").reset_index(drop=True)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    return _build_external_engine_result(
        engine=engine,
        args=args,
        run_dir=run_dir,
        elapsed=float(total_elapsed),
        rss=float(max_rss),
        fold_df=fold_df,
        pred_df=pred_df,
        log_file=log_file,
        time_file=time_file,
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

    rc = 1
    elapsed = math.nan
    rss = math.nan
    run_task = CliStatus(
        f"Running {str(engine).upper()} CV (folds={int(args.cv)})...",
        enabled=_status_enabled(),
    )
    with run_task:
        rc, elapsed, rss = _run_timed(
            cmd,
            log_file=log_file,
            time_file=time_file,
            limit_mem_gb=args.limit_mem,
        )
        if rc != 0:
            run_task.fail(f"{str(engine).upper()} run failed (exit={int(rc)}).")
    data = _load_result_json(result_json)

    if rc == 0 and str(data.get("status", "")).lower() == "ok":
        run_task.complete(f"Running {str(engine).upper()} CV ...Finished")
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
    if int(rc) == 137 and args.limit_mem is not None:
        note = f"engine failed: memory limit exceeded ({float(args.limit_mem):g} GB)"
    run_task.fail(f"{str(engine).upper()} run failed: {note}")
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


def _run_engine_hiblup(
    *,
    args: argparse.Namespace,
    bench_dir: Path,
    meta_json: Path,
    hiblup_bin: str,
) -> EngineRunResult:
    engine = "hiblup"
    run_dir = bench_dir / "runs" / engine
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / f"{engine}.log"
    time_file = run_dir / f"{engine}.time"

    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    sample_df = pd.read_csv(str(meta["sample_tsv"]), sep="\t")
    ids = sample_df["id"].astype(str).to_numpy()
    y = sample_df["y"].to_numpy(dtype=float)
    fold = sample_df["fold"].to_numpy(dtype=int)
    bfile = str(meta["filtered_plink_prefix"])

    total_elapsed = 0.0
    max_rss = math.nan
    fold_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    note = ""

    fold_ids = sorted(int(x) for x in np.unique(fold) if int(x) > 0)
    fold_total = max(1, int(len(fold_ids)))
    for i, fd in enumerate(fold_ids, start=1):
        with CliStatus(
            _fold_status_desc(engine, i, fold_total, fd, "fit"),
            enabled=_status_enabled(),
        ) as fold_task:
            fold_dir = run_dir / f"fold{fd}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            pheno_file = (fold_dir / "pheno.tsv").resolve()
            out_prefix = (fold_dir / "hiblup").resolve()
            beta_file = Path(str(out_prefix) + ".beta")
            vars_file = Path(str(out_prefix) + ".vars")
            rand_file = Path(str(out_prefix) + ".rand")

            y_fold = np.asarray(y, dtype=float).copy()
            y_fold[fold == fd] = np.nan
            pd.DataFrame({"id": ids, "pheno": y_fold}).to_csv(pheno_file, sep="\t", index=False, na_rep="NA")

            attempt_time = run_dir / f"{engine}.fold{fd}.time"
            cmd = [
                str(hiblup_bin),
                "--single-trait",
                "--pheno",
                str(pheno_file),
                "--pheno-pos",
                "2",
                "--bfile",
                str(bfile),
                "--add",
                "--threads",
                str(int(args.thread)),
                "--out",
                str(out_prefix),
            ]
            rc, elapsed, rss = _run_timed(
                cmd,
                log_file=log_file,
                time_file=attempt_time,
                cwd=fold_dir,
                append_log=(i > 1),
                limit_mem_gb=args.limit_mem,
            )
            if math.isfinite(elapsed):
                total_elapsed += float(elapsed)
            if math.isfinite(rss):
                max_rss = float(rss) if not math.isfinite(max_rss) else max(float(max_rss), float(rss))
            if rc != 0:
                note = f"HIBLUP failed on fold {fd} (exit={rc})."
                if int(rc) == 137 and args.limit_mem is not None:
                    note = f"HIBLUP failed on fold {fd}: memory limit exceeded ({float(args.limit_mem):g} GB)."
                fold_task.fail(note)
                break
            if not rand_file.exists():
                note = f"HIBLUP fold {fd} did not produce {rand_file.name}."
                fold_task.fail(note)
                break

            fold_task.desc = _fold_status_desc(engine, i, fold_total, fd, "parse")
            rand_df = _read_table_guess(rand_file)
            id_col = "ID" if "ID" in rand_df.columns else str(rand_df.columns[0])
            gebv_col = _choose_hiblup_gebv_column(rand_df)
            rand_df = rand_df[[id_col, gebv_col]].rename(columns={id_col: "sample_id", gebv_col: "gebv"})
            rand_df["sample_id"] = rand_df["sample_id"].astype(str)
            train_mu = float(np.nanmean(y[fold != fd]))
            train_sd = float(np.nanstd(y[fold != fd]))
            beta0_raw = _parse_hiblup_beta(beta_file, train_mu)
            beta0 = float(beta0_raw)
            # Some HIBLUP versions/output layouts may expose a fixed-effect value that
            # is not the prediction intercept expected here. Guard against obvious offset.
            if math.isfinite(train_sd) and train_sd > 0:
                if abs(float(beta0_raw) - train_mu) > max(5.0 * float(train_sd), 5.0):
                    beta0 = float(train_mu)
            # Prefer fold-local calibration from training data when available:
            # beta = mean(y_train - gebv_train), making y_pred comparable across engines.
            train_df_for_cal = pd.DataFrame({"sample_id": ids[fold != fd], "y_train": y[fold != fd]})
            cal_df = train_df_for_cal.merge(rand_df, on="sample_id", how="left")
            cal_y = pd.to_numeric(cal_df.get("y_train"), errors="coerce")
            cal_g = pd.to_numeric(cal_df.get("gebv"), errors="coerce")
            cal_mask = cal_y.notna() & cal_g.notna()
            if int(cal_mask.sum()) >= 3:
                beta_cal = _nanmean_or_nan((cal_y[cal_mask] - cal_g[cal_mask]).to_numpy(dtype=float))
                if math.isfinite(beta_cal):
                    beta0 = float(beta_cal)
            pve_fold = _parse_hiblup_pve(vars_file)

            test_df = pd.DataFrame({"sample_id": ids[fold == fd], "y_true": y[fold == fd]})
            pred_df = test_df.merge(rand_df, on="sample_id", how="left")
            if pred_df["gebv"].isna().any():
                note = f"HIBLUP fold {fd} missing breeding values for one or more test samples."
                fold_task.fail(note)
                break
            pred_df["engine"] = engine
            pred_df["fold"] = int(fd)
            pred_df["intercept"] = float(beta0)
            pred_df["y_pred"] = float(beta0) + pd.to_numeric(pred_df["gebv"], errors="coerce")
            pred_df["pve_fold"] = float(pve_fold)
            pred_df["n_train"] = int(np.sum(fold != fd))
            pred_df["n_test"] = int(np.sum(fold == fd))

            r2, pear, spear = _safe_prediction_metrics(pred_df["y_true"], pred_df["y_pred"])
            fold_rows.append(
                {
                    "fold": int(fd),
                    "n_train": int(np.sum(fold != fd)),
                    "n_test": int(np.sum(fold == fd)),
                    "r2": float(r2),
                    "pearson": float(pear),
                    "spearman": float(spear),
                    "pve": float(pve_fold),
                }
            )
            pred_frames.append(
                pred_df[
                    ["engine", "sample_id", "fold", "y_true", "y_pred", "gebv", "intercept", "pve_fold", "n_train", "n_test"]
                ].copy()
            )
            fold_task.complete(f"{str(engine).upper()} fold {int(i)}/{int(fold_total)} (id={int(fd)}) ...Finished")

    _write_time_summary(time_file, total_elapsed, max_rss)
    if note:
        return EngineRunResult(
            engine=engine,
            status="fail",
            exit_code=1,
            elapsed_sec=float(total_elapsed),
            peak_rss_kb=float(max_rss),
            mean_r2=math.nan,
            mean_pearson=math.nan,
            mean_spearman=math.nan,
            mean_pve=math.nan,
            folds=0,
            fold_file="",
            prediction_file="",
            result_json="",
            log_file=str(log_file),
            time_file=str(time_file),
            note=note,
        )

    return _build_external_engine_result(
        engine=engine,
        args=args,
        run_dir=run_dir,
        elapsed=total_elapsed,
        rss=max_rss,
        fold_df=pd.DataFrame(fold_rows),
        pred_df=pd.concat(pred_frames, ignore_index=True),
        log_file=log_file,
        time_file=time_file,
    )


def _run_engine_blupf90(
    *,
    args: argparse.Namespace,
    bench_dir: Path,
    meta_json: Path,
    pregsf90_bin: str,
    blupf90_bin: str,
    engine: str = "blupf90",
    use_apy: bool = False,
) -> EngineRunResult:
    engine = str(engine).strip().lower() or "blupf90"
    run_dir = bench_dir / "runs" / engine
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / f"{engine}.log"
    time_file = run_dir / f"{engine}.time"

    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    sample_df = pd.read_csv(str(meta["sample_tsv"]), sep="\t")
    ids = sample_df["id"].astype(str).to_numpy()
    y = sample_df["y"].to_numpy(dtype=float)
    fold = sample_df["fold"].to_numpy(dtype=int)
    m = int(meta["m"])
    n = int(meta["n"])
    geno = np.fromfile(str(meta["geno_bin"]), dtype=np.float32).reshape((m, n))
    numeric_ids = np.arange(1, int(ids.shape[0]) + 1, dtype=np.int64)
    id_map = {str(orig): int(iid) for orig, iid in zip(ids, numeric_ids)}
    n_levels = int(numeric_ids.shape[0])

    snp_file = run_dir / "geno.snp.txt"
    snp_xref = run_dir / "geno.snp_xref.txt"
    ped_file = run_dir / "ped.txt"
    if not snp_file.exists():
        _write_blupf90_fixed_snp_file(snp_file, numeric_ids, geno)
    if not snp_xref.exists():
        with snp_xref.open("w", encoding="utf-8") as fh:
            for iid, sid in zip(numeric_ids, ids):
                fh.write(f"{int(iid)} {sid}\n")
    if not ped_file.exists():
        with ped_file.open("w", encoding="utf-8") as fh:
            for iid in numeric_ids:
                fh.write(f"{int(iid)} 0 0\n")

    total_elapsed = 0.0
    max_rss = math.nan
    fold_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    note = ""

    # Keep thread fairness across engines by default:
    # BLUPF90 follows --thread unless explicitly overridden.
    # (Override via JX_GBLUPBENCH_BLUPF90_THREADS.)
    blup_env = os.environ.copy()
    raw_thr = str(os.environ.get("JX_GBLUPBENCH_BLUPF90_THREADS", "")).strip()
    if raw_thr == "":
        try:
            blup_threads = max(1, int(args.thread))
        except Exception:
            blup_threads = 1
    else:
        try:
            blup_threads = max(1, int(raw_thr))
        except Exception:
            blup_threads = max(1, int(args.thread))
    thr_text = str(int(blup_threads))
    thread_keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    for key in thread_keys:
        blup_env[key] = thr_text
    # Avoid Intel OpenMP warning:
    # "OMP_PROC_BIND ignored because KMP_AFFINITY has been defined"
    blup_env.pop("KMP_AFFINITY", None)
    blup_env.setdefault("OMP_PROC_BIND", "false")

    conda_prefix = str(blup_env.get("CONDA_PREFIX", "")).strip()
    if conda_prefix != "":
        conda_lib = str((Path(conda_prefix) / "lib").resolve())
        old_ld = str(blup_env.get("LD_LIBRARY_PATH", ""))
        blup_env["LD_LIBRARY_PATH"] = (
            f"{conda_lib}:{old_ld}" if old_ld.strip() != "" else conda_lib
        )

    apy_mode = _detect_pregsf90_apy_mode(pregsf90_bin) if use_apy else "modern"
    apy_type = 2
    apy_use_noa22 = str(os.environ.get("JX_GBLUPBENCH_APY_NOA22", "1")).strip().lower() not in {"0", "false", "no"}
    apy_alpha_beta = str(os.environ.get("JX_GBLUPBENCH_APY_ALPHABETA", "0.95 0.05")).strip()
    if apy_alpha_beta == "":
        apy_alpha_beta = "0.95 0.05"
    pregs_env = dict(blup_env)
    pregs_single_thread = False

    fold_ids = sorted(int(x) for x in np.unique(fold) if int(x) > 0)
    fold_total = max(1, int(len(fold_ids)))
    for i, fd in enumerate(fold_ids, start=1):
        fold_dir = run_dir / f"fold{fd}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        data_file = fold_dir / "data.txt"
        pregs_par = fold_dir / "pregs.par"
        blup_par = fold_dir / "blup.par"
        ped_ref = os.path.relpath(str(ped_file), str(fold_dir))
        snp_ref = os.path.relpath(str(snp_file), str(fold_dir))
        xref_ref = os.path.relpath(str(snp_xref), str(fold_dir))

        with data_file.open("w", encoding="utf-8") as fh:
            for sid, yv, fv in zip(ids, y, fold):
                pheno_val = "-999" if int(fv) == int(fd) else f"{float(yv):.12g}"
                fh.write(f"{id_map[str(sid)]} 1 {pheno_val} 1\n")

        def _write_pregs_par(current_apy_mode: str, include_noa22: bool) -> None:
            apy_lines: list[str] = []
            if use_apy:
                apy_lines.extend(
                    _build_apy_option_lines(n_levels=n_levels, apy_mode=current_apy_mode, apy_type=apy_type)
                )
                if include_noa22:
                    apy_lines.append("OPTION noA22directinv")

            pregs_par.write_text(
                "\n".join(
                    [
                        "DATAFILE",
                        data_file.name,
                        "NUMBER_OF_TRAITS",
                        "1",
                        "NUMBER_OF_EFFECTS",
                        "2",
                        "OBSERVATION(S)",
                        "3",
                        "WEIGHT(S)",
                        "4",
                        "EFFECTS:",
                        "2 1 cross",
                        f"1 {n_levels} cross",
                        "RANDOM_RESIDUAL VALUES",
                        "1.0",
                        "RANDOM_GROUP",
                        "2",
                        "RANDOM_TYPE",
                        "add_animal",
                        "FILE",
                        ped_ref,
                        "(CO)VARIANCES",
                        "1.0",
                        f"OPTION SNP_file {snp_ref} {xref_ref}",
                        "OPTION no_quality_control",
                        *apy_lines,
                        f"OPTION AlphaBeta {apy_alpha_beta}",
                        "OPTION tunedG 0",
                        "OPTION saveAscii",
                        "OPTION saveGInverse",
                        "OPTION saveGimA22iRen",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

        _write_pregs_par(apy_mode, apy_use_noa22)

        blup_par.write_text(
            "\n".join(
                [
                    "DATAFILE",
                    data_file.name,
                    "NUMBER_OF_TRAITS",
                    "1",
                    "NUMBER_OF_EFFECTS",
                    "2",
                    "OBSERVATION(S)",
                    "3",
                    "WEIGHT(S)",
                    "4",
                    "EFFECTS:",
                    "2 1 cross",
                    f"1 {n_levels} cross",
                    "RANDOM_RESIDUAL VALUES",
                    "1.0",
                    "RANDOM_GROUP",
                    "2",
                    "RANDOM_TYPE",
                    "user_file",
                    "FILE",
                    "GimA22i_Ren.txt",
                    "(CO)VARIANCES",
                    "1.0",
                    "OPTION missing -999",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        pregs_time = fold_dir / "pregs.time"
        solver_time = fold_dir / "solver.time"
        rc1 = 1
        with CliStatus(
            _fold_status_desc(engine, i, fold_total, fd, "preGSf90"),
            enabled=_status_enabled(),
        ) as pregs_task:
            rc1, elapsed1, rss1 = _run_tool_with_parfile_fallback(
                pregsf90_bin,
                pregs_par,
                log_file=log_file,
                time_file=pregs_time,
                cwd=fold_dir,
                env=pregs_env,
                limit_mem_gb=args.limit_mem,
            )
            if math.isfinite(elapsed1):
                total_elapsed += float(elapsed1)
            if math.isfinite(rss1):
                max_rss = float(rss1) if not math.isfinite(max_rss) else max(float(max_rss), float(rss1))
            if use_apy:
                # Compatibility fallback for older preGSf90 APY syntax.
                # Try switching to legacy APY syntax and/or removing noA22directinv when unsupported.
                for _ in range(5):
                    try:
                        tail = "\n".join(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()[-160:])
                    except Exception:
                        tail = ""

                    retry_needed = False
                    if (apy_mode != "legacy") and ("ERROR: OPTION Apy has 3 arguments" in tail):
                        apy_mode = "legacy"
                        retry_needed = True
                    elif apy_use_noa22 and ('ERROR: The "noA22directinv" option is not supported in this program.' in tail):
                        apy_use_noa22 = False
                        retry_needed = True
                    elif (
                        use_apy
                        and (apy_mode == "legacy")
                        and (int(apy_type) == 2)
                        and ('ERROR: "OPTION apy 2" must be combined with "OPTION noA22directinv".' in tail)
                        and (not apy_use_noa22)
                    ):
                        # Some legacy preGSf90 builds simultaneously reject noA22directinv for APY2,
                        # but require it for APY2. Fall back to legacy APY type 1.
                        apy_type = 1
                        retry_needed = True
                    elif ("SIGSEGV" in tail) or ("segmentation fault" in tail.lower()):
                        # APY2 can be unstable on some preGSf90 builds.
                        # Fall back to a safer APY setup.
                        if int(apy_type) != 1:
                            apy_type = 1
                            retry_needed = True
                        if apy_alpha_beta.strip() != "1.0 0.0":
                            apy_alpha_beta = "1.0 0.0"
                            retry_needed = True
                        if not pregs_single_thread:
                            pregs_single_thread = True
                            for key in thread_keys:
                                pregs_env[key] = "1"
                            retry_needed = True

                    if not retry_needed:
                        break

                    _write_pregs_par(apy_mode, apy_use_noa22)
                    rc1b, elapsed1b, rss1b = _run_tool_with_parfile_fallback(
                        pregsf90_bin,
                        pregs_par,
                        log_file=log_file,
                        time_file=pregs_time,
                        cwd=fold_dir,
                        env=pregs_env,
                        limit_mem_gb=args.limit_mem,
                    )
                    if math.isfinite(elapsed1b):
                        total_elapsed += float(elapsed1b)
                    if math.isfinite(rss1b):
                        max_rss = float(rss1b) if not math.isfinite(max_rss) else max(float(max_rss), float(rss1b))
                    rc1 = int(rc1b)
            if int(rc1) == 0:
                pregs_task.complete(
                    f"{str(engine).upper()} fold {int(i)}/{int(fold_total)} (id={int(fd)}) [preGSf90] ...Finished"
                )

        if rc1 != 0:
            note = f"preGSf90 failed on fold {fd} (exit={rc1})."
            if int(rc1) == 137 and args.limit_mem is not None:
                note = f"preGSf90 failed on fold {fd}: memory limit exceeded ({float(args.limit_mem):g} GB)."
            if use_apy:
                try:
                    tail = "\n".join(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:])
                    if "ERROR: OPTION Apy has 3 arguments" in tail:
                        note = (
                            f"{note} Current preGSf90 requires legacy APY syntax. "
                            "Set JX_GBLUPBENCH_APY_MODE=legacy or upgrade preGSf90."
                        )
                    if 'ERROR: The "noA22directinv" option is not supported in this program.' in tail:
                        note = (
                            f"{note} Current preGSf90 does not support noA22directinv. "
                            "Set JX_GBLUPBENCH_APY_NOA22=0 or upgrade preGSf90."
                        )
                    if 'ERROR: "OPTION apy 2" must be combined with "OPTION noA22directinv".' in tail:
                        note = (
                            f"{note} Current preGSf90 reports incompatible APY2/noA22directinv requirements. "
                            "Try legacy APY type 1 by setting JX_GBLUPBENCH_APY_OPTION='apy 1 1 core.le.100'."
                        )
                    if ("SIGSEGV" in tail) or ("segmentation fault" in tail.lower()):
                        note = (
                            f"{note} preGSf90 APY crashed (SIGSEGV). "
                            "Try JX_GBLUPBENCH_APY_OPTION='apy 1 1 core.le.100' and JX_GBLUPBENCH_BLUPF90_THREADS=1, "
                            "or upgrade preGSf90."
                        )
                    if "No core (or noncore) animals found" in tail:
                        note = (
                            f"{note} APY core/non-core groups were not formed; "
                            "check pedigree/content or add APY-specific preGSf90 options (e.g. snpapy)."
                        )
                except Exception:
                    pass
            pregs_task.fail(note)
            break

        gi_candidates = [
            fold_dir / "Gi",
            fold_dir / "Gi.txt",
            fold_dir / "Gi_ren.txt",
            fold_dir / "Gi_ren",
            fold_dir / "ApyGi",
            fold_dir / "ApyGi_ren.txt",
            fold_dir / "GimA22i_Ren.txt",
            fold_dir / "GimA22i_ren.txt",
        ]
        gi_file = next((p for p in gi_candidates if p.exists()), None)
        if gi_file is None:
            note = (
                f"preGSf90 fold {fd} did not produce expected inverse matrix file "
                "(Gi/ApyGi/GimA22i variants)."
            )
            if use_apy:
                try:
                    tail_lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()[-160:]
                    tail = "\n".join(tail_lines)
                    if "ERROR: OPTION Apy has 3 arguments" in tail:
                        note = (
                            f"{note} Current preGSf90 expects legacy APY syntax with 3 arguments; "
                            "please upgrade preGSf90 or provide APY options compatible with your version."
                        )
                    elif 'ERROR: The "noA22directinv" option is not supported in this program.' in tail:
                        note = (
                            f"{note} Current preGSf90 does not support noA22directinv for APY; "
                            "please upgrade BLUPF90 tools."
                        )
                    elif "ERROR:" in tail:
                        err_line = next((ln.strip() for ln in reversed(tail_lines) if "ERROR:" in ln), "")
                        if err_line:
                            note = f"{note} {err_line}"
                except Exception:
                    pass
            break

        blup_par.write_text(
            "\n".join(
                [
                    "DATAFILE",
                    data_file.name,
                    "NUMBER_OF_TRAITS",
                    "1",
                    "NUMBER_OF_EFFECTS",
                    "2",
                    "OBSERVATION(S)",
                    "3",
                    "WEIGHT(S)",
                    "4",
                    "EFFECTS:",
                    "2 1 cross",
                    f"1 {n_levels} cross",
                    "RANDOM_RESIDUAL VALUES",
                    "1.0",
                    "RANDOM_GROUP",
                    "2",
                    "RANDOM_TYPE",
                    "user_file",
                    "FILE",
                    gi_file.name,
                    "(CO)VARIANCES",
                    "1.0",
                    "OPTION missing -999",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        rc2 = 1
        with CliStatus(
            _fold_status_desc(engine, i, fold_total, fd, Path(blupf90_bin).name),
            enabled=_status_enabled(),
        ) as solver_task:
            rc2, elapsed2, rss2 = _run_tool_with_parfile_fallback(
                blupf90_bin,
                blup_par,
                log_file=log_file,
                time_file=solver_time,
                cwd=fold_dir,
                env=blup_env,
                limit_mem_gb=args.limit_mem,
            )
            if math.isfinite(elapsed2):
                total_elapsed += float(elapsed2)
            if math.isfinite(rss2):
                max_rss = float(rss2) if not math.isfinite(max_rss) else max(float(max_rss), float(rss2))
            if int(rc2) == 0:
                solver_task.complete(
                    f"{str(engine).upper()} fold {int(i)}/{int(fold_total)} "
                    f"(id={int(fd)}) [{Path(blupf90_bin).name}] ...Finished"
                )
        if rc2 != 0:
            note = f"{Path(blupf90_bin).name} failed on fold {fd} (exit={rc2})."
            if int(rc2) == 137 and args.limit_mem is not None:
                note = (
                    f"{Path(blupf90_bin).name} failed on fold {fd}: "
                    f"memory limit exceeded ({float(args.limit_mem):g} GB)."
                )
            try:
                tail = "\n".join(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:])
                if ("SIGSEGV" in tail) or ("segmentation fault" in tail.lower()):
                    note = (
                        f"{note} SIGSEGV detected; retry with "
                        "JX_GBLUPBENCH_BLUPF90_THREADS=1 and ensure CONDA_PREFIX/lib in LD_LIBRARY_PATH."
                    )
            except Exception:
                pass
            solver_task.fail(note)
            break

        sol_file = fold_dir / "solutions"
        if not sol_file.exists():
            note = f"BLUPF90 fold {fd} missing solutions output."
            break

        intercept = float(np.nanmean(y[fold != fd]))
        sol_map: dict[int, float] = {}
        for line in sol_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            value = _safe_float(parts[3], math.nan)
            if not math.isfinite(value):
                continue
            a = _safe_int_token(parts[0])
            b = _safe_int_token(parts[1])
            c = _safe_int_token(parts[2])
            if a is None or b is None or c is None:
                continue
            # BLUPF90 `solutions` column order may vary by build/configuration.
            # Try common layouts:
            #   1) effect, level, trait, value
            #   2) trait, effect, level, value
            #   3) effect, trait, level, value
            candidates = (
                (a, b, c),
                (b, c, a),
                (a, c, b),
            )
            for effect_i, level_i, trait_i in candidates:
                if trait_i != 1:
                    continue
                if effect_i == 1 and level_i == 1:
                    intercept = float(value)
                elif effect_i == 2 and (1 <= int(level_i) <= int(n_levels)):
                    sol_map[int(level_i)] = float(value)
                else:
                    continue
                break

        test_ids = ids[fold == fd]
        gebv = np.asarray([sol_map.get(int(id_map[str(sid)]), math.nan) for sid in test_ids], dtype=float)
        if np.any(~np.isfinite(gebv)):
            note = f"BLUPF90 fold {fd} missing breeding values for one or more test samples."
            break

        pred_df = pd.DataFrame(
            {
                "engine": engine,
                "sample_id": test_ids,
                "fold": int(fd),
                "y_true": y[fold == fd],
                "y_pred": float(intercept) + gebv,
                "gebv": gebv,
                "intercept": float(intercept),
                "pve_fold": math.nan,
                "n_train": int(np.sum(fold != fd)),
                "n_test": int(np.sum(fold == fd)),
            }
        )
        r2, pear, spear = _safe_prediction_metrics(pred_df["y_true"], pred_df["y_pred"])
        fold_rows.append(
            {
                "fold": int(fd),
                "n_train": int(np.sum(fold != fd)),
                "n_test": int(np.sum(fold == fd)),
                "r2": float(r2),
                "pearson": float(pear),
                "spearman": float(spear),
                "pve": math.nan,
            }
        )
        pred_frames.append(pred_df)

    _write_time_summary(time_file, total_elapsed, max_rss)
    if note:
        return EngineRunResult(
            engine=engine,
            status="fail",
            exit_code=1,
            elapsed_sec=float(total_elapsed),
            peak_rss_kb=float(max_rss),
            mean_r2=math.nan,
            mean_pearson=math.nan,
            mean_spearman=math.nan,
            mean_pve=math.nan,
            folds=0,
            fold_file="",
            prediction_file="",
            result_json="",
            log_file=str(log_file),
            time_file=str(time_file),
            note=note,
        )

    return _build_external_engine_result(
        engine=engine,
        args=args,
        run_dir=run_dir,
        elapsed=total_elapsed,
        rss=max_rss,
        fold_df=pd.DataFrame(fold_rows),
        pred_df=pd.concat(pred_frames, ignore_index=True),
        log_file=log_file,
        time_file=time_file,
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

    engines_raw = str(args.engines)
    if bool(args.check) and (not bool(getattr(args, "_engines_explicit", False))):
        engines_raw = _DEFAULT_ENGINES_CHECK_ALL
    engines = _parse_engines(engines_raw)
    if len(engines) == 0:
        raise ValueError(f"No valid engines selected. Use --engines {_ENGINE_LIST_TEXT}")

    # Keep CLI-level thread env aligned with --thread and enforce OpenBLAS policy
    # consistently across JanusX entrypoints.
    apply_blas_thread_env(int(args.thread))
    maybe_warn_non_openblas(strict=require_openblas_by_default())

    if args.check:
        checks = _collect_env_checks(engines)
        checks.extend(_run_smoke_checks_with_sim(engines))
        ok = _print_env_checks(engines, checks)
        raise SystemExit(0 if ok else 1)

    out_dir = safe_expanduser(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_dir = out_dir / f"{args.prefix}.gblup_bench"
    bench_dir.mkdir(parents=True, exist_ok=True)

    with CliStatus("Preparing benchmark dataset...", enabled=_status_enabled()) as prep_task:
        try:
            with _gblupbench_prep_stage_ctx(int(args.thread)):
                data_info = _prepare_dataset(args, bench_dir=bench_dir, engines=engines)
        except Exception:
            prep_task.fail("Preparing benchmark dataset ...Failed")
            raise
        prep_task.complete("Preparing benchmark dataset ...Finished")

    tmp_dir = bench_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    r_script = tmp_dir / "gblup_r_engine.R"
    if any(e in {"sommer", "rrblup"} for e in engines):
        _build_r_engine_script(r_script)

    rscript_bin = _ensure_rscript()
    hiblup_bin = _ensure_hiblup()
    pregsf90_bin = _ensure_pregsf90()
    blupf90_bin = _ensure_blupf90()

    rows: list[EngineRunResult] = []
    for engine in engines:
        if engine in {"janusx", "janusxrrblup"}:
            rr = _run_engine_janusx(
                args=args,
                bench_dir=bench_dir,
                meta_json=Path(data_info["meta_json"]),
                engine=engine,
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

        if engine == "hiblup":
            if hiblup_bin is None:
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
                        note="hiblup not found in PATH",
                    )
                )
                continue
            rr = _run_engine_hiblup(
                args=args,
                bench_dir=bench_dir,
                meta_json=Path(data_info["meta_json"]),
                hiblup_bin=hiblup_bin,
            )
            rows.append(rr)
            continue

        if engine in {"blupf90", "blupf90apy"}:
            if (pregsf90_bin is None) or (blupf90_bin is None):
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
                        note="preGSf90 and/or blupf90/airemlf90/remlf90 not found in PATH",
                    )
                )
                continue
            rr = _run_engine_blupf90(
                args=args,
                bench_dir=bench_dir,
                meta_json=Path(data_info["meta_json"]),
                pregsf90_bin=pregsf90_bin,
                blupf90_bin=blupf90_bin,
                engine=engine,
                use_apy=(engine == "blupf90apy"),
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
        "limit_mem_gb": (None if args.limit_mem is None else float(args.limit_mem)),
        "cv": int(args.cv),
        "run_folds": int(args.run_folds),
        "limit_predtrain": int(args.limit_predtrain),
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
        for p in [r_script]:
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
