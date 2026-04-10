#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FarmCPU benchmark launcher for JanusX / GAPIT / rMVP.

Key features:
  - One-command benchmark for three kernels:
      * janusx (jxpy gwas -farmcpu)
      * gapit  (R, via temporary launcher + run_gwas_r_method.R)
      * rmvp   (R, via temporary launcher + run_gwas_r_method.R)
  - Unified FarmCPU grid policy to align with JanusX FarmCPU defaults:
      * bin.size      default: 5e5, 5e6, 5e7
      * QTNbound      default: int(sqrt(n / log10(n)))
      * bin.selection derived from QTNbound and nbin:
          step = max(1, QTNbound // nbin)
          seq(step, QTNbound, by=step)
  - Outputs:
      * per-kernel GWAS result table
      * per-kernel top-k table
      * benchmark summary table with pseudoQTN/topk/peak RSS/elapsed
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


def _bootstrap_repo_python_path() -> None:
    if __package__:
        return
    here = Path(__file__).resolve()
    # .../python/janusx/script/benchmark.py -> add .../python
    py_root = here.parents[2]
    if str(py_root) not in sys.path:
        sys.path.insert(0, str(py_root))


_bootstrap_repo_python_path()

from janusx.script._common.pathcheck import safe_expanduser  # noqa: E402
from janusx.script._common.threads import detect_effective_threads  # noqa: E402
from janusx.script._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog  # noqa: E402
from janusx.script._common.colspec import parse_zero_based_index_specs  # noqa: E402


_P_COL_KEYS = (
    "p",
    "pvalue",
    "p.value",
    "pval",
    "p_wald",
    "pwald",
    "p_lrt",
    "plrt",
    "p_score",
)
_CHR_COL_KEYS = ("chr", "chrom", "chromosome", "chrm")
_POS_COL_KEYS = ("bp", "pos", "position", "ps", "genpos")
_SNP_COL_KEYS = ("snp", "id", "rs", "rsid", "marker", "markername", "variant", "variantid")
_SUMMARY_TAG = "benchmark"


@dataclass
class RunResult:
    kernel: str
    status: str
    exit_code: int
    elapsed_sec: float
    peak_rss_kb: float
    result_file: str
    topk_file: str
    topk_count: int
    topk_snps: str
    pseudo_qtn: Optional[int]
    log_file: str
    time_file: str
    note: str


@dataclass
class EnvCheck:
    component: str
    ok: bool
    detail: str


@dataclass
class AlignRuntime:
    mode: str
    rmvp_engine: str
    rmvp_vc_method: str
    rmvp_method_bin: str
    gapit_method_bin: str
    farmcpu_threshold_base: float
    farmcpu_threshold_effective: float
    threshold_mode: str
    snps_for_threshold: int


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


def _parse_number_list(raw: str, cast=float) -> list[Any]:
    vals: list[Any] = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        try:
            vals.append(cast(p))
        except Exception:
            continue
    return vals


def _count_bim_snps(bfile_prefix: str) -> int:
    bim_path = Path(f"{bfile_prefix}.bim")
    if not bim_path.exists():
        return 0
    n = 0
    with bim_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return int(n)


def _safe_float(v: Any, default: float = math.nan) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    return x


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
    """
    Returns (elapsed_sec, max_rss_kb).
    """
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


def _run_timed(
    cmd: list[str],
    *,
    log_file: Path,
    time_file: Path,
    env: Optional[dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> tuple[int, float, float]:
    """
    Returns: (exit_code, elapsed_sec, peak_rss_kb)
    """
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
    # 1) tab, 2) csv, 3) whitespace
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
        except Exception as e:  # pragma: no cover - best effort
            last_err = e
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Unable to parse table: {path}")


def _first_col(colmap: dict[str, str], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in colmap:
            return colmap[k]
    return None


def _standardize_assoc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lstrip("#").lower(): c for c in df.columns}
    p_col = _first_col(cols, _P_COL_KEYS)
    if p_col is None:
        raise ValueError("P-value column not found in GWAS result.")

    chr_col = _first_col(cols, _CHR_COL_KEYS)
    pos_col = _first_col(cols, _POS_COL_KEYS)
    snp_col = _first_col(cols, _SNP_COL_KEYS)

    p = pd.to_numeric(df[p_col], errors="coerce")
    if chr_col is not None and pos_col is not None:
        chrom = df[chr_col].astype(str).str.replace(r"^chr", "", regex=True, case=False).str.strip()
        pos = pd.to_numeric(df[pos_col], errors="coerce")
        snp = chrom.astype(str) + "_" + pos.round().astype("Int64").astype(str)
    elif snp_col is not None:
        snp = df[snp_col].astype(str).str.strip()
        chrom = pd.Series(pd.NA, index=df.index, dtype="object")
        pos = pd.Series(pd.NA, index=df.index, dtype="float64")
    else:
        raise ValueError("Neither (chr,pos) nor SNP id columns found.")

    out = pd.DataFrame(
        {
            "SNP": snp,
            "CHR": chrom,
            "POS": pos,
            "P": p,
        }
    )
    out = out.replace([math.inf, -math.inf], math.nan).dropna(subset=["SNP", "P"])
    out = out[(out["P"] > 0) & (out["P"] <= 1)]
    out = out[out["SNP"] != ""]
    out = out.sort_values(["P", "SNP"]).drop_duplicates("SNP", keep="first")
    return out.reset_index(drop=True)


def _extract_topk(
    assoc_file: Path,
    out_topk: Path,
    k: int,
) -> tuple[int, str]:
    df = _read_table_guess(assoc_file)
    std = _standardize_assoc(df)
    top = std.sort_values("P", ascending=True).head(max(1, int(k))).copy()
    out_topk.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(out_topk, sep="\t", index=False)
    top_snps = ";".join(top["SNP"].astype(str).head(10).tolist())
    return int(top.shape[0]), top_snps


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
            raise ValueError("`jx benchmark` currently supports one trait at a time; please pass a single `-n/--n` index.")
        idx = int(ncol[0]) + 1  # phenotype index excludes sample ID column
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


def _read_fam_ids(bfile_prefix: str) -> list[str]:
    fam_path = Path(f"{bfile_prefix}.fam")
    if not fam_path.exists():
        return []
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, engine="python", usecols=[1], dtype=str)
    return fam.iloc[:, 0].astype(str).str.strip().tolist()


def _prepare_gapit_inputs_from_bfile(
    *,
    bfile_prefix: str,
    pheno_for_r: Path,
    out_dir: Path,
    maf: float,
    missing_rate: float,
    snps_only: bool,
    chunk_size: int,
) -> tuple[Path, Path]:
    from janusx.gfreader import load_genotype_chunks

    ph = _read_table_guess(pheno_for_r)
    if ph.shape[1] < 2:
        raise ValueError(f"Invalid phenotype file for GAPIT inputs: {pheno_for_r}")

    taxa = ph.iloc[:, 0].astype(str).str.strip()
    taxa = [x for x in taxa.tolist() if x]
    taxa = list(dict.fromkeys(taxa))
    fam_ids = set(_read_fam_ids(bfile_prefix))
    if len(fam_ids) > 0:
        taxa = [x for x in taxa if x in fam_ids]
    if len(taxa) == 0:
        raise ValueError("No overlapping samples between phenotype and bfile for GAPIT input generation.")

    geno_chunks: list[np.ndarray] = []
    snp_ids: list[str] = []
    chroms: list[str] = []
    positions: list[int] = []
    seen: dict[str, int] = {}

    for geno_chunk, sites in load_genotype_chunks(
        bfile_prefix,
        chunk_size=max(1, int(chunk_size)),
        maf=float(maf),
        missing_rate=float(missing_rate),
        impute=True,
        model="add",
        snps_only=bool(snps_only),
        sample_ids=taxa,
    ):
        arr = np.asarray(geno_chunk, dtype=np.float32)
        if arr.shape[0] == 0:
            continue
        geno_chunks.append(arr)
        for s in sites:
            chrom = str(s.chrom).strip()
            pos = int(s.pos)
            base = f"{chrom}_{pos}"
            n = int(seen.get(base, 0)) + 1
            seen[base] = n
            sid = base if n == 1 else f"{base}_{n}"
            snp_ids.append(sid)
            chroms.append(chrom)
            positions.append(pos)

    if len(geno_chunks) == 0 or len(snp_ids) == 0:
        raise ValueError("No SNPs available after filtering for GAPIT input generation.")

    geno_mn = np.concatenate(geno_chunks, axis=0)
    if geno_mn.shape[0] != len(snp_ids):
        raise RuntimeError("GAPIT input generation mismatch between genotype matrix and site metadata.")
    if geno_mn.shape[1] != len(taxa):
        raise RuntimeError("GAPIT input generation mismatch between genotype matrix and sample IDs.")

    gd = pd.DataFrame(geno_mn.T, columns=snp_ids)
    gd.insert(0, "Taxa", taxa)
    gm = pd.DataFrame({"SNP": snp_ids, "Chromosome": chroms, "Position": positions})

    out_dir.mkdir(parents=True, exist_ok=True)
    gd_path = out_dir / "gapit.GD.tsv"
    gm_path = out_dir / "gapit.GM.tsv"
    gd.to_csv(gd_path, sep="\t", index=False, float_format="%.6g")
    gm.to_csv(gm_path, sep="\t", index=False)
    return gd_path, gm_path


def _compute_farmcpu_grid(
    n_samples: int,
    *,
    nbin: int,
    bound_override: Optional[int],
    bin_size: list[float],
) -> tuple[int, list[int], list[float]]:
    n = max(2, int(n_samples))
    if bound_override is not None and int(bound_override) > 0:
        bound = int(bound_override)
    else:
        den = math.log10(float(n))
        if den <= 0:
            bound = 1
        else:
            bound = max(1, int(math.sqrt(float(n) / den)))
    nbin_den = max(1, int(nbin))
    step = max(1, int(bound // nbin_den))
    bin_selection = list(range(step, bound + 1, step))
    if len(bin_selection) == 0:
        bin_selection = [bound]
    if len(bin_size) == 0:
        bin_size = [5e5, 5e6, 5e7]
    return bound, bin_selection, bin_size


def _resolve_align_runtime(
    args: argparse.Namespace,
    *,
    n_snps_for_threshold: int,
) -> AlignRuntime:
    mode = str(args.align_mode).strip().lower()
    if mode not in {"legacy", "manuscript"}:
        mode = "manuscript"

    # Fixed rMVP defaults for reproducible/high-consistency benchmark behavior.
    rmvp_engine = "mvp"
    rmvp_vc_method = "BRENT"
    rmvp_method_bin = "FaST-LMM"
    gapit_method_bin = "optimum"

    threshold_mode = str(args.farmcpu_threshold_mode).strip().lower()
    if threshold_mode not in {"auto", "fixed", "per_marker"}:
        threshold_mode = "auto"
    if threshold_mode == "auto":
        threshold_mode = "per_marker" if mode == "manuscript" else "fixed"

    base = float(args.farmcpu_threshold)
    if threshold_mode == "per_marker":
        m = max(1, int(n_snps_for_threshold))
        eff = float(base / float(m))
    else:
        eff = float(base)

    return AlignRuntime(
        mode=mode,
        rmvp_engine=rmvp_engine,
        rmvp_vc_method=rmvp_vc_method,
        rmvp_method_bin=rmvp_method_bin,
        gapit_method_bin=gapit_method_bin,
        farmcpu_threshold_base=float(base),
        farmcpu_threshold_effective=float(eff),
        threshold_mode=threshold_mode,
        snps_for_threshold=max(0, int(n_snps_for_threshold)),
    )


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    # Prefer a parent that actually contains the R runner in source tree.
    for base in [here.parent, *here.parents]:
        if (base / "build" / "gwasx86" / "run_gwas_r_method.R").exists():
            return base
    # Fallback: nearest parent with pyproject.toml (source checkout).
    for base in [here.parent, *here.parents]:
        if (base / "pyproject.toml").exists():
            return base
    # Last resort: keep historical behavior.
    return here.parents[3]


def _resolve_runner_script() -> Path:
    override = str(os.environ.get("JANUSX_R_RUNNER", "")).strip()
    if override:
        return Path(safe_expanduser(override))

    here = Path(__file__).resolve()
    env_prefix = Path(sys.prefix).resolve()
    exe_prefix = Path(sys.executable).resolve().parent.parent
    cwd = Path.cwd().resolve()
    candidates = [
        # Source checkout path.
        _find_repo_root() / "build" / "gwasx86" / "run_gwas_r_method.R",
        # Common conda/pip install layouts.
        here.parents[3] / "build" / "gwasx86" / "run_gwas_r_method.R",
        env_prefix / "build" / "gwasx86" / "run_gwas_r_method.R",
        exe_prefix / "build" / "gwasx86" / "run_gwas_r_method.R",
        # Common local clone layouts.
        Path.home() / "github" / "JanusX" / "build" / "gwasx86" / "run_gwas_r_method.R",
        Path.home() / "script" / "JanusX" / "build" / "gwasx86" / "run_gwas_r_method.R",
        Path.home() / "JanusX" / "build" / "gwasx86" / "run_gwas_r_method.R",
        # Legacy/flat fallback near benchmark.py.
        here.parent / "run_gwas_r_method.R",
    ]
    # Try cwd ancestors as lightweight autodiscovery.
    for base in [cwd, *cwd.parents[:4]]:
        candidates.append(base / "build" / "gwasx86" / "run_gwas_r_method.R")

    for path in candidates:
        if path.exists():
            return path

    # Last fallback: download official runner into user cache.
    cache_dir = Path(safe_expanduser("~/.cache/janusx"))
    cache_runner = cache_dir / "run_gwas_r_method.R"
    urls: list[str] = []
    try:
        ver = str(importlib.metadata.version("janusx")).strip()
    except Exception:
        ver = ""
    if ver:
        urls.append(f"https://raw.githubusercontent.com/FJingxian/JanusX/v{ver}/build/gwasx86/run_gwas_r_method.R")
    urls.extend(
        [
            "https://raw.githubusercontent.com/FJingxian/JanusX/main/build/gwasx86/run_gwas_r_method.R",
            "https://raw.githubusercontent.com/FJingxian/JanusX/master/build/gwasx86/run_gwas_r_method.R",
        ]
    )
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "janusx-benchmark/1"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            if "run_gapit" not in content or "run_rmvp" not in content:
                continue
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cache_runner.with_suffix(".tmp")
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(cache_runner)
            return cache_runner
        except Exception:
            continue

    return candidates[0]


def _is_executable_file(path: Path) -> bool:
    return path.exists() and path.is_file() and os.access(str(path), os.X_OK)


def _resolve_jx_subcmd_candidates(subcmd: str) -> list[list[str]]:
    cands: list[list[str]] = []
    env_bin = Path(sys.executable).resolve().parent
    jx = env_bin / "jx"
    jxpy = env_bin / "jxpy"
    if _is_executable_file(jx):
        cands.append([str(jx), subcmd])
    if _is_executable_file(jxpy):
        cands.append([str(jxpy), subcmd])
    return cands


def _pick_working_subcmd_cmd(subcmd: str) -> list[str]:
    cands = _resolve_jx_subcmd_candidates(subcmd)
    if len(cands) > 0:
        return cands[0]
    env_bin = Path(sys.executable).resolve().parent
    raise RuntimeError(f"No jx/jxpy executable found in current env: {env_bin}")


def _resolve_jx_gwas_cmd() -> list[str]:
    return _pick_working_subcmd_cmd("gwas")


def _resolve_gformat_cmd() -> list[str]:
    return _pick_working_subcmd_cmd("gformat")


def _parse_kernels(raw: str) -> list[str]:
    kernels = [k.strip().lower() for k in str(raw).split(",") if k.strip()]
    kernels = [k for k in kernels if k in {"janusx", "gapit", "rmvp"}]
    return kernels


def _probe_cmd(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=max(10, int(timeout)),
        )
        out = proc.stdout if proc.stdout is not None else ""
        return int(proc.returncode), str(out)
    except Exception as e:
        return 127, str(e)


def _collect_env_checks(args: argparse.Namespace, kernels: list[str]) -> list[EnvCheck]:
    items: list[EnvCheck] = []
    need_r = any(k in {"gapit", "rmvp"} for k in kernels)

    # JanusX kernel checks
    if "janusx" in kernels:
        gwas_cands = _resolve_jx_subcmd_candidates("gwas")
        if len(gwas_cands) > 0:
            used_cmd = " ".join(gwas_cands[0])
            items.append(EnvCheck("janusx.gwas_cli", True, f"OK via `{used_cmd}`"))
        else:
            env_bin = Path(sys.executable).resolve().parent
            detail = f"No jx/jxpy executable found in current env: {env_bin}"
            items.append(EnvCheck("janusx.gwas_cli", False, detail))

    r_bin = ""
    r_ok = True
    runner = _resolve_runner_script()

    if need_r:
        r_bin = str(shutil.which("Rscript") or "")

        r_ok = bool(r_bin) and Path(r_bin).exists()
        items.append(
            EnvCheck(
                "rscript",
                r_ok,
                r_bin if r_ok else "Rscript not found in current env PATH",
            )
        )
        items.append(
            EnvCheck(
                "r_runner",
                runner.exists(),
                str(runner) if runner.exists() else f"missing runner: {runner}",
            )
        )
        if r_ok:
            rc, out = _probe_cmd([r_bin, "--version"], timeout=30)
            items.append(
                EnvCheck(
                    "rscript.version",
                    rc == 0,
                    "OK" if rc == 0 else f"failed (exit={rc}): {str(out).splitlines()[:1]}",
                )
            )

    # GAPIT checks
    if "gapit" in kernels:
        if r_ok:
            rc, _out = _probe_cmd(
                [
                    r_bin,
                    "-e",
                    'ok <- requireNamespace("GAPIT", quietly=TRUE) || requireNamespace("GAPIT3", quietly=TRUE); quit(status=if (ok) 0 else 1)',
                ],
                timeout=90,
            )
            pkg_ok = rc == 0
            vendor_path = runner.parent / "vendor" / "gapit_functions.txt"
            vendor_ok = vendor_path.exists()
            if pkg_ok:
                items.append(EnvCheck("gapit.runtime", True, "GAPIT/GAPIT3 package available"))
            elif vendor_ok:
                items.append(
                    EnvCheck(
                        "gapit.runtime",
                        True,
                        f"package missing; using local GAPIT source fallback: {vendor_path}",
                    )
                )
            else:
                items.append(
                    EnvCheck(
                        "gapit.runtime",
                        False,
                        "GAPIT package missing and local vendor fallback not found",
                    )
                )
        else:
            items.append(EnvCheck("gapit.runtime", False, "Rscript unavailable, cannot check GAPIT"))

    # rMVP checks
    if "rmvp" in kernels:
        if r_ok:
            rc, _out = _probe_cmd(
                [
                    r_bin,
                    "-e",
                    'ok <- requireNamespace("rMVP", quietly=TRUE) && requireNamespace("bigmemory", quietly=TRUE); quit(status=if (ok) 0 else 1)',
                ],
                timeout=90,
            )
            items.append(
                EnvCheck(
                    "rmvp.packages",
                    rc == 0,
                    "rMVP+bigmemory available" if rc == 0 else "missing rMVP and/or bigmemory",
                )
            )
        else:
            items.append(EnvCheck("rmvp.packages", False, "Rscript unavailable, cannot check rMVP"))

    return items


def _print_env_checks(kernels: list[str], checks: list[EnvCheck]) -> bool:
    print(f"[CHECK] kernels: {','.join(kernels)}")
    for it in checks:
        tag = "OK" if it.ok else "FAIL"
        print(f"[{tag}] {it.component}: {it.detail}")
    overall = all(it.ok for it in checks) if checks else False
    print(f"[CHECK] overall: {'PASS' if overall else 'FAIL'}")
    return overall


def _as_geno_flag_and_value(args: argparse.Namespace, gfile: str) -> tuple[str, str]:
    if args.bfile:
        return "-bfile", str(gfile)
    if args.vcf:
        return "-vcf", str(gfile)
    if args.hmp:
        return "-hmp", str(gfile)
    return "-file", str(gfile)


def _convert_to_bfile_if_needed(
    args: argparse.Namespace,
    gfile: str,
    out_tmp: Path,
    *,
    log_file: Path,
    time_file: Path,
) -> tuple[str, float, float]:
    if args.bfile:
        return str(gfile), 0.0, math.nan
    out_tmp.mkdir(parents=True, exist_ok=True)
    conv_prefix = out_tmp / "bench_input"
    gfmt = _resolve_gformat_cmd()
    geno_flag, geno_val = _as_geno_flag_and_value(args, gfile)
    cmd = [
        *gfmt,
        geno_flag,
        geno_val,
        "-fmt",
        "plink",
        "-o",
        str(out_tmp),
        "-prefix",
        "bench_input",
        "-maf",
        str(args.maf),
        "-geno",
        str(args.geno),
    ]
    rc, elapsed, rss = _run_timed(cmd, log_file=log_file, time_file=time_file)
    if rc != 0:
        raise RuntimeError(f"gformat conversion to PLINK failed (exit={rc}). See log: {log_file}")
    for ext in ("bed", "bim", "fam"):
        p = Path(f"{conv_prefix}.{ext}")
        if not p.exists():
            raise FileNotFoundError(f"Missing converted PLINK file: {p}")
    return str(conv_prefix), elapsed, rss


def _guess_janusx_result_file(run_out: Path) -> Path:
    cands = sorted(run_out.rglob("*.farmcpu.tsv"))
    if len(cands) == 0:
        raise FileNotFoundError(f"No JanusX FarmCPU result file found under: {run_out}")
    return cands[0]


def _parse_pseudo_from_text(text: str) -> Optional[int]:
    best: Optional[int] = None
    pats = [
        re.compile(r"Found\s+(\d+)\s+QTN", re.IGNORECASE),
        re.compile(r"pseudoQTN[s]?\s*[~:=]\s*(\d+)", re.IGNORECASE),
        re.compile(r"n_pseudo_qtn[^0-9]*(\d+)", re.IGNORECASE),
    ]
    lines = str(text).splitlines()
    for line in lines:
        for pat in pats:
            m = pat.search(line)
            if not m:
                continue
            try:
                v = int(m.group(1))
            except Exception:
                continue
            if best is None or v > best:
                best = v

    # GAPIT/rMVP logs often print pseudo-QTN lists as:
    # "seqQTN" then one or more R-style vector lines (e.g. "[1] 8187 ...").
    # Parse those blocks and count vector entries (excluding the R index token).
    for idx, line in enumerate(lines):
        if not re.search(r"\bseqQTN\b", line, re.IGNORECASE):
            continue
        count = 0
        j = idx + 1
        while j < len(lines):
            ln = lines[j]
            lns = ln.strip()
            if not lns:
                break
            if re.search(
                r"(current loop|optimizing|scanning|number of covariates|farmcpu\.lm|genomic inflation)",
                lns,
                re.IGNORECASE,
            ):
                break
            nums = [int(x) for x in re.findall(r"\b\d+\b", lns)]
            if re.match(r"^\[\d+\]\s*", lns) and nums:
                nums = nums[1:]
            count += len(nums)
            j += 1
        if count > 0 and (best is None or count > best):
            best = count

    # Fallback: some logs report this count explicitly on the next line.
    for idx, line in enumerate(lines):
        if not re.search(r"number of covariates in current loop", line, re.IGNORECASE):
            continue
        if idx + 1 >= len(lines):
            continue
        nums = [int(x) for x in re.findall(r"\b\d+\b", lines[idx + 1])]
        if nums:
            v = max(nums)
            if best is None or v > best:
                best = v
    return best


def _build_r_launcher_script(path: Path) -> None:
    script = r"""#!/usr/bin/env Rscript
env <- Sys.getenv
runner <- env("JX_RUNNER")
method <- env("JX_METHOD")
bfile <- env("JX_BFILE")
pheno <- env("JX_PHENO")
trait <- env("JX_TRAIT")
out_file <- env("JX_OUT_FILE")
workdir <- env("JX_WORKDIR")
threads <- env("JX_THREADS")
gapit_gd <- env("JX_GAPIT_GD")
gapit_gm <- env("JX_GAPIT_GM")
gapit_source <- env("JX_GAPIT_SOURCE")
gapit_source_url <- env("JX_GAPIT_SOURCE_URL")
farmcpu_bin_size <- env("JX_FARMCPU_BIN_SIZE")
farmcpu_bin_selection <- env("JX_FARMCPU_BIN_SELECTION")
farmcpu_max_loop <- env("JX_FARMCPU_MAX_LOOP")
farmcpu_thr <- env("JX_FARMCPU_THRESHOLD")
farmcpu_bound <- env("JX_FARMCPU_BOUND")
farmcpu_maf <- env("JX_FARMCPU_MAF")
rmvp_engine <- tolower(env("JX_RMVP_ENGINE"))
rmvp_vc_method <- toupper(env("JX_RMVP_VC_METHOD"))
rmvp_method_bin <- env("JX_RMVP_METHOD_BIN")
gapit_method_bin <- env("JX_GAPIT_METHOD_BIN")
inner_log <- env("JX_INNER_LOG")
meta_file <- env("JX_META_FILE")
rscript_exec <- env("JX_RSCRIPT_BIN")
if (!nzchar(rscript_exec)) {
  rscript_exec <- file.path(R.home("bin"), "Rscript")
}
if (!nzchar(rmvp_engine)) rmvp_engine <- "mvp"
if (!nzchar(rmvp_vc_method)) rmvp_vc_method <- "BRENT"
if (!nzchar(rmvp_method_bin)) rmvp_method_bin <- "FaST-LMM"
if (!nzchar(gapit_method_bin)) gapit_method_bin <- "optimum"
method_bin <- if (grepl("^rmvp_", method, ignore.case = TRUE)) rmvp_method_bin else gapit_method_bin

if (!file.exists(runner)) {
  stop(sprintf("Missing runner script: %s", runner), call. = FALSE)
}
dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
dir.create(workdir, recursive = TRUE, showWarnings = FALSE)

cmd <- c(
  runner,
  "--method", method,
  "--bfile", bfile,
  "--pheno", pheno,
  "--trait", trait,
  "--out", out_file,
  "--workdir", workdir,
  "--threads", threads,
  "--farmcpu-direct", "true",
  "--rmvp-farmcpu-engine", rmvp_engine,
  "--rmvp-vc-method", rmvp_vc_method,
  "--farmcpu-method-bin", method_bin,
  "--farmcpu-bin-size", farmcpu_bin_size,
  "--farmcpu-bin-selection", farmcpu_bin_selection,
  "--farmcpu-max-loop", farmcpu_max_loop,
  "--farmcpu-threshold-output", farmcpu_thr,
  "--farmcpu-qtn-threshold", farmcpu_thr,
  "--farmcpu-p-threshold", farmcpu_thr,
  "--farmcpu-bound", farmcpu_bound,
  "--farmcpu-maf-threshold", farmcpu_maf
)
if (nzchar(gapit_gd)) {
  cmd <- c(cmd, "--gapit-gd", gapit_gd)
}
if (nzchar(gapit_gm)) {
  cmd <- c(cmd, "--gapit-gm", gapit_gm)
}
if (nzchar(gapit_source)) {
  cmd <- c(cmd, "--gapit-source", gapit_source)
}
if (nzchar(gapit_source_url)) {
  cmd <- c(cmd, "--gapit-source-url", gapit_source_url)
}

out <- system2(rscript_exec, cmd, stdout = TRUE, stderr = TRUE)
status <- attr(out, "status")
if (is.null(status)) status <- 0L
if (is.na(status)) status <- 1L
if (!nzchar(inner_log)) {
  inner_log <- file.path(workdir, paste0(method, ".inner.log"))
}
writeLines(out, con = inner_log)

detect_qtn_from_log <- function(lines) {
  if (length(lines) == 0) return(NA_integer_)
  best <- NA_integer_
  for (ln in lines) {
    if (!grepl("qtn|pseudo", ln, ignore.case = TRUE)) next
    nums <- regmatches(ln, gregexpr("[0-9]+", ln, perl = TRUE))[[1]]
    if (length(nums) == 0) next
    vals <- suppressWarnings(as.integer(nums))
    vals <- vals[is.finite(vals)]
    if (length(vals) == 0) next
    cand <- max(vals)
    if (is.na(best) || cand > best) best <- cand
  }
  best
}

detect_qtn_from_files <- function(root) {
  if (!dir.exists(root)) return(NA_integer_)
  ff <- list.files(root, recursive = TRUE, full.names = TRUE)
  if (length(ff) == 0) return(NA_integer_)
  ff <- ff[grepl("qtn|pseudo", basename(ff), ignore.case = TRUE)]
  if (length(ff) == 0) return(NA_integer_)
  best <- NA_integer_
  for (f in ff) {
    if (dir.exists(f)) next
    n <- NA_integer_
    ext <- tolower(tools::file_ext(f))
    if (ext %in% c("txt", "tsv", "csv", "out", "dat", "pmap")) {
      tab <- try(read.table(f, header = TRUE, sep = "", check.names = FALSE, stringsAsFactors = FALSE), silent = TRUE)
      if (!inherits(tab, "try-error")) {
        n <- as.integer(nrow(tab))
      } else {
        tab2 <- try(read.table(f, header = FALSE, sep = "", check.names = FALSE, stringsAsFactors = FALSE), silent = TRUE)
        if (!inherits(tab2, "try-error")) n <- as.integer(nrow(tab2))
      }
    }
    if (!is.na(n) && n > 0) {
      if (is.na(best) || n > best) best <- n
    }
  }
  best
}

q1 <- detect_qtn_from_log(out)
q2 <- detect_qtn_from_files(workdir)
qtn <- if (is.na(q1)) q2 else if (is.na(q2)) q1 else max(q1, q2)

meta <- data.frame(
  key = c("exit_code", "pseudo_qtn", "inner_log"),
  value = c(as.character(status), ifelse(is.na(qtn), "NA", as.character(qtn)), inner_log),
  stringsAsFactors = FALSE
)
write.table(meta, file = meta_file, sep = "\t", row.names = FALSE, quote = FALSE)
quit(save = "no", status = as.integer(status))
"""
    path.write_text(script, encoding="utf-8")


def _read_meta_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        return {}
    if "key" not in df.columns or "value" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        out[str(r["key"])] = str(r["value"])
    return out


def _run_kernel_janusx(
    args: argparse.Namespace,
    *,
    gfile: str,
    geno_flag: str,
    pheno_for_jx: Path,
    out_dir: Path,
    farm_bound: int,
    farm_bin_size: list[float],
    align_runtime: AlignRuntime,
) -> RunResult:
    kernel = "janusx"
    run_dir = out_dir / kernel
    log_file = run_dir / "janusx.log"
    time_file = run_dir / "janusx.time"
    run_dir.mkdir(parents=True, exist_ok=True)
    gwas_out = run_dir / "results"
    gwas_out.mkdir(parents=True, exist_ok=True)

    gwas_cmd = _resolve_jx_gwas_cmd()
    cmd = [
        *gwas_cmd,
        geno_flag,
        str(gfile),
        "-p",
        str(pheno_for_jx),
        "-farmcpu",
        "-maf",
        str(args.maf),
        "-geno",
        str(args.geno),
        "-chunksize",
        str(args.chunksize),
        "-t",
        str(args.thread),
        "-o",
        str(gwas_out),
        "-prefix",
        args.prefix,
        "--farmcpu-iter",
        str(int(args.farmcpu_iter)),
        "--farmcpu-threshold",
        f"{float(align_runtime.farmcpu_threshold_base):.17g}",
        "--farmcpu-nbin",
        str(int(args.farmcpu_nbin)),
        "--farmcpu-bin-size",
        ",".join(str(int(v)) if float(v).is_integer() else str(v) for v in farm_bin_size),
        "--farmcpu-qtn-bound",
        str(int(farm_bound)),
    ]
    if args.qcov is not None:
        cmd += ["-q", str(args.qcov)]
    for c in args.cov or []:
        cmd += ["-c", str(c)]
    if args.snps_only:
        cmd += ["-snps-only"]
    if args.mmap_limit:
        cmd += ["-mmap-limit"]

    rc, elapsed, rss = _run_timed(cmd, log_file=log_file, time_file=time_file)
    status = "ok" if rc == 0 else "failed"
    result_file = ""
    topk_file = str(run_dir / f"{args.prefix}.janusx.top{args.topk}.tsv")
    topk_count = 0
    topk_snps = ""
    note = ""

    pseudo_qtn = None
    if log_file.exists():
        pseudo_qtn = _parse_pseudo_from_text(log_file.read_text(errors="ignore"))

    if rc == 0:
        try:
            rf = _guess_janusx_result_file(gwas_out)
            result_file = str(rf)
            topk_count, topk_snps = _extract_topk(rf, Path(topk_file), args.topk)
        except Exception as e:
            status = "failed"
            note = f"JanusX result parse failed: {e}"
    else:
        note = "JanusX command failed."

    return RunResult(
        kernel=kernel,
        status=status,
        exit_code=rc,
        elapsed_sec=elapsed,
        peak_rss_kb=rss,
        result_file=result_file,
        topk_file=topk_file if Path(topk_file).exists() else "",
        topk_count=int(topk_count),
        topk_snps=topk_snps,
        pseudo_qtn=pseudo_qtn,
        log_file=str(log_file),
        time_file=str(time_file),
        note=note,
    )


def _run_kernel_r(
    args: argparse.Namespace,
    *,
    kernel: str,
    bfile_prefix: str,
    pheno_for_r: Path,
    out_dir: Path,
    farm_bound: int,
    farm_bin_selection: list[int],
    farm_bin_size: list[float],
    align_runtime: AlignRuntime,
    r_launcher: Path,
) -> RunResult:
    run_dir = out_dir / kernel
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / f"{kernel}.log"
    time_file = run_dir / f"{kernel}.time"
    inner_log = run_dir / f"{kernel}.inner.log"
    meta_file = run_dir / f"{kernel}.meta.tsv"
    result_file = run_dir / f"{args.prefix}.{kernel}.farmcpu.tsv"
    topk_file = run_dir / f"{args.prefix}.{kernel}.top{args.topk}.tsv"
    workdir = run_dir / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    gapit_gd = ""
    gapit_gm = ""
    if kernel == "gapit":
        gd_path, gm_path = _prepare_gapit_inputs_from_bfile(
            bfile_prefix=str(bfile_prefix),
            pheno_for_r=pheno_for_r,
            out_dir=workdir / "gapit_input",
            maf=float(args.maf),
            missing_rate=float(args.geno),
            snps_only=bool(args.snps_only),
            chunk_size=max(1000, int(args.chunksize)),
        )
        gapit_gd = str(gd_path)
        gapit_gm = str(gm_path)

    method = "gapit_farmcpu" if kernel == "gapit" else "rmvp_farmcpu"
    runner = _resolve_runner_script()
    if not runner.exists():
        raise FileNotFoundError(f"R runner not found: {runner}")

    env = os.environ.copy()
    rmvp_force_rebuild = "1" if (kernel == "rmvp" and (not bool(getattr(args, "rmvp_reuse_cache", False)))) else "0"
    env.update(
        {
            "JX_RUNNER": str(runner),
            "JX_METHOD": method,
            "JX_BFILE": str(bfile_prefix),
            "JX_PHENO": str(pheno_for_r),
            "JX_TRAIT": "PHENO",
            "JX_OUT_FILE": str(result_file),
            "JX_WORKDIR": str(workdir),
            "JX_THREADS": str(args.thread),
            "JX_GAPIT_GD": gapit_gd,
            "JX_GAPIT_GM": gapit_gm,
            "JX_GAPIT_SOURCE": "",
            "JX_GAPIT_SOURCE_URL": "",
            "JX_FARMCPU_BIN_SIZE": ",".join(str(int(v)) if float(v).is_integer() else str(v) for v in farm_bin_size),
            "JX_FARMCPU_BIN_SELECTION": ",".join(str(int(v)) for v in farm_bin_selection),
            "JX_FARMCPU_MAX_LOOP": str(args.farmcpu_iter),
            "JX_FARMCPU_THRESHOLD": f"{float(align_runtime.farmcpu_threshold_effective):.17g}",
            "JX_FARMCPU_BOUND": str(int(farm_bound)),
            "JX_FARMCPU_MAF": str(args.maf),
            "JX_RMVP_ENGINE": str(align_runtime.rmvp_engine),
            "JX_RMVP_VC_METHOD": str(align_runtime.rmvp_vc_method),
            "JX_RMVP_METHOD_BIN": str(align_runtime.rmvp_method_bin),
            "JX_GAPIT_METHOD_BIN": str(align_runtime.gapit_method_bin),
            "JX_INNER_LOG": str(inner_log),
            "JX_META_FILE": str(meta_file),
            "JX_RSCRIPT_BIN": "",
            "RMVP_FORCE_REBUILD": rmvp_force_rebuild,
        }
    )

    r_bin = shutil.which("Rscript")
    if not r_bin:
        raise RuntimeError("Rscript not found in current env PATH.")

    cmd = [str(r_bin), str(r_launcher)]
    rc, elapsed, rss = _run_timed(cmd, log_file=log_file, time_file=time_file, env=env)
    status = "ok" if rc == 0 else "failed"
    pseudo_qtn = None
    note = ""

    meta = _read_meta_kv(meta_file)
    if "pseudo_qtn" in meta:
        try:
            if str(meta["pseudo_qtn"]).strip().upper() != "NA":
                pseudo_qtn = int(float(meta["pseudo_qtn"]))
        except Exception:
            pseudo_qtn = None
    if inner_log.exists():
        pseudo_qtn_from_log = _parse_pseudo_from_text(inner_log.read_text(errors="ignore"))
        if pseudo_qtn is None:
            pseudo_qtn = pseudo_qtn_from_log
        elif pseudo_qtn_from_log is not None and pseudo_qtn_from_log > pseudo_qtn:
            pseudo_qtn = pseudo_qtn_from_log

    topk_count = 0
    topk_snps = ""
    if rc == 0:
        if result_file.exists():
            try:
                topk_count, topk_snps = _extract_topk(result_file, topk_file, args.topk)
            except Exception as e:
                status = "failed"
                note = f"{kernel} topk parse failed: {e}"
        else:
            status = "failed"
            note = f"{kernel} result file missing: {result_file}"
    else:
        note = f"{kernel} command failed."

    return RunResult(
        kernel=kernel,
        status=status,
        exit_code=rc,
        elapsed_sec=elapsed,
        peak_rss_kb=rss,
        result_file=str(result_file) if result_file.exists() else "",
        topk_file=str(topk_file) if topk_file.exists() else "",
        topk_count=int(topk_count),
        topk_snps=topk_snps,
        pseudo_qtn=pseudo_qtn,
        log_file=str(log_file),
        time_file=str(time_file),
        note=note,
    )


def _apply_pseudo_qtn_cap(result: RunResult, cap: Optional[int]) -> RunResult:
    if cap is None or int(cap) <= 0:
        return result
    if result.pseudo_qtn is None:
        return result
    if int(result.pseudo_qtn) <= int(cap):
        return result
    result.status = "failed"
    cap_note = (
        f"pseudoQTN cap exceeded: observed={int(result.pseudo_qtn)} > cap={int(cap)}. "
        "Use a larger cap or disable --force-pseudo-qtn-cap."
    )
    if result.note:
        result.note = f"{result.note}; {cap_note}"
    else:
        result.note = cap_note
    return result


def _topk_snp_set_from_summary_row(row: pd.Series) -> set[str]:
    topk_file = str(row.get("topk_file", "") or "").strip()
    if topk_file:
        p = Path(topk_file)
        if p.exists():
            try:
                top_df = _read_table_guess(p)
                if "SNP" in top_df.columns:
                    snps = [str(x).strip() for x in top_df["SNP"].tolist()]
                else:
                    std = _standardize_assoc(top_df)
                    snps = [str(x).strip() for x in std["SNP"].tolist()]
                snps = [x for x in snps if x]
                if snps:
                    return set(snps)
            except Exception:
                pass

    raw = str(row.get("topk_snps", "") or "").strip()
    if not raw:
        return set()
    return {x.strip() for x in raw.split(";") if x.strip()}


def _pseudo_qtn_consistency(a: set[str], b: set[str]) -> Optional[float]:
    if len(a) == 0 or len(b) == 0:
        return None
    denom = min(len(a), len(b))
    if denom <= 0:
        return None
    return float(len(a & b) / float(denom))


def _pseudo_qtn_count_consistency(a: Any, b: Any) -> Optional[float]:
    try:
        av = int(float(a))
        bv = int(float(b))
    except Exception:
        return None
    if av <= 0 or bv <= 0:
        return None
    lo = min(av, bv)
    hi = max(av, bv)
    if hi <= 0:
        return None
    return float(lo / hi)


def _save_summary(out_dir: Path, prefix: str, rows: list[RunResult], extra_cfg: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    recs: list[dict[str, Any]] = []
    for r in rows:
        recs.append(
            {
                "kernel": r.kernel,
                "status": r.status,
                "exit_code": r.exit_code,
                "elapsed_sec": r.elapsed_sec,
                "peak_rss_kb": r.peak_rss_kb,
                "peak_rss_gb": (r.peak_rss_kb / 1024.0 / 1024.0) if math.isfinite(r.peak_rss_kb) else math.nan,
                "pseudo_qtn": r.pseudo_qtn if r.pseudo_qtn is not None else pd.NA,
                "topk_count": r.topk_count,
                "topk_snps": r.topk_snps,
                "result_file": r.result_file,
                "topk_file": r.topk_file,
                "log_file": r.log_file,
                "time_file": r.time_file,
                "note": r.note,
            }
        )
    df = pd.DataFrame(recs)
    tsv = out_dir / f"{prefix}.{_SUMMARY_TAG}.tsv"
    md = out_dir / f"{prefix}.{_SUMMARY_TAG}.md"
    cfg = out_dir / f"{prefix}.{_SUMMARY_TAG}.config.json"
    df.to_csv(tsv, sep="\t", index=False)

    lines = [
        "| Kernel | Status | Time(s) | Peak RSS(GB) | pseudoQTN | topK |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        t = "NA" if pd.isna(r["elapsed_sec"]) else f"{float(r['elapsed_sec']):.2f}"
        m = "NA" if pd.isna(r["peak_rss_gb"]) else f"{float(r['peak_rss_gb']):.3f}"
        q = "NA" if pd.isna(r["pseudo_qtn"]) else str(int(r["pseudo_qtn"]))
        lines.append(f"| {r['kernel']} | {r['status']} | {t} | {m} | {q} | {int(r['topk_count'])} |")

    preferred = ["janusx", "gapit", "rmvp"]
    seen = {str(x) for x in df["kernel"].tolist()}
    kernel_order = [k for k in preferred if k in seen]
    extras = [str(x) for x in df["kernel"].tolist() if str(x) not in preferred]
    kernel_order.extend(extras)

    row_by_kernel: dict[str, pd.Series] = {}
    for _, r in df.iterrows():
        row_by_kernel[str(r["kernel"])] = r

    lines.extend(
        [
            "",
            "**PseudoQTN Count Consistency Matrix (Lower Triangle)**",
            "Formula: `min(qtn_i, qtn_j) / max(qtn_i, qtn_j)`; `NA` means missing/non-positive pseudoQTN.",
        ]
    )
    mat_head_cnt = "| kernel | " + " | ".join(kernel_order) + " |"
    mat_sep_cnt = "|---|" + "|".join(["---:" for _ in kernel_order]) + "|"
    lines.extend([mat_head_cnt, mat_sep_cnt])

    for i, ki in enumerate(kernel_order):
        row_i = row_by_kernel.get(ki)
        qi = row_i["pseudo_qtn"] if row_i is not None else pd.NA
        row_cells = [ki]
        for j, kj in enumerate(kernel_order):
            if j > i:
                row_cells.append("")
                continue
            if i == j:
                val = _pseudo_qtn_count_consistency(qi, qi)
            else:
                row_j = row_by_kernel.get(kj)
                qj = row_j["pseudo_qtn"] if row_j is not None else pd.NA
                val = _pseudo_qtn_count_consistency(qi, qj)
            row_cells.append("NA" if val is None else f"{val:.3f}")
        lines.append("| " + " | ".join(row_cells) + " |")

    topk_sets: dict[str, set[str]] = {}
    for k in kernel_order:
        row = row_by_kernel.get(k)
        if row is None:
            topk_sets[k] = set()
            continue
        topk_sets[k] = _topk_snp_set_from_summary_row(row)

    lines.extend(
        [
            "",
            "**TopK SNP Consistency Matrix (Lower Triangle)**",
            "Formula: `|A∩B| / min(|A|, |B|)`; `NA` means one side has no topK SNPs.",
        ]
    )
    mat_head = "| kernel | " + " | ".join(kernel_order) + " |"
    mat_sep = "|---|" + "|".join(["---:" for _ in kernel_order]) + "|"
    lines.extend([mat_head, mat_sep])

    for i, ki in enumerate(kernel_order):
        row_cells = [ki]
        for j, kj in enumerate(kernel_order):
            if j > i:
                row_cells.append("")
                continue
            if i == j:
                val = 1.0 if len(topk_sets.get(ki, set())) > 0 else None
            else:
                val = _pseudo_qtn_consistency(topk_sets.get(ki, set()), topk_sets.get(kj, set()))
            row_cells.append("NA" if val is None else f"{val:.3f}")
        lines.append("| " + " | ".join(row_cells) + " |")

    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg.write_text(json.dumps(extra_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dev_help_requested(argv: Optional[list[str]] = None) -> bool:
    tokens = list(sys.argv[1:] if argv is None else argv)
    return ("-dev" in tokens) or ("--dev" in tokens)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    show_dev_help = _dev_help_requested(argv)
    tokens = list(sys.argv[1:] if argv is None else argv)
    p = CliArgumentParser(
        prog="jx benchmark",
        formatter_class=cli_help_formatter(),
        description="Benchmark FarmCPU kernels: JanusX / GAPIT / rMVP.",
        epilog=minimal_help_epilog(
            [
                "jx benchmark -bfile example_prefix -p pheno.tsv -n 0",
                "jx benchmark -vcf example.vcf.gz -p pheno.tsv -n 0 --kernels janusx,gapit,rmvp",
                "jx benchmark -h -dev",
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
            "Phenotype column(s), zero-based index (excluding sample ID), "
            "comma list (e.g. 0,2), or numeric range (e.g. 0:2). "
            "Benchmark currently supports one selected trait."
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
    optional_group.add_argument("-prefix", "--prefix", default="benchmark", type=str, help="Output prefix.")
    optional_group.add_argument("-maf", "--maf", default=0.02, type=float, help="MAF filter (default: %(default)s).")
    optional_group.add_argument("-geno", "--geno", default=0.05, type=float, help="Missing-rate filter (default: %(default)s).")
    optional_group.add_argument("-chunksize", "--chunksize", default=10_000, type=int, help="JanusX chunk size.")
    optional_group.add_argument("-t", "--thread", default=detect_effective_threads(), type=int, help="Threads.")
    optional_group.add_argument("-q", "--qcov", default=0, type=int, help="JanusX qcov (PC count).")
    optional_group.add_argument("-c", "--cov", action="append", default=None, help="Additional covariates (JanusX only).")
    optional_group.add_argument("-snps-only", "--snps-only", action="store_true", default=False, help="Use SNPs only.")
    optional_group.add_argument("-mmap-limit", "--mmap-limit", action="store_true", default=False, help="Enable JanusX mmap-limit.")
    optional_group.add_argument("--kernels", default="janusx,gapit,rmvp", type=str, help="Comma list: janusx,gapit,rmvp.")
    optional_group.add_argument("--topk", default=100, type=int, help="Top-k SNP output size (default: %(default)s).")
    optional_group.add_argument(
        "--check",
        action="store_true",
        default=False,
        help=(
            "Only check environment dependencies for selected kernels, then exit. "
            "In this mode, genotype/pheno inputs are optional."
        ),
    )

    advanced_group = p.add_argument_group("Advanced arguments (show with -h -dev)")
    advanced_group.add_argument(
        "--farmcpu-bin-size",
        default="500000,5000000,50000000",
        type=str,
        help=("FarmCPU bin.size CSV." if show_dev_help else argparse.SUPPRESS),
    )
    advanced_group.add_argument(
        "--farmcpu-nbin",
        default=5,
        type=int,
        help=("FarmCPU nbin for derived bin.selection." if show_dev_help else argparse.SUPPRESS),
    )
    advanced_group.add_argument(
        "--farmcpu-bound",
        default=None,
        type=int,
        help=("Override QTNbound." if show_dev_help else argparse.SUPPRESS),
    )
    advanced_group.add_argument(
        "--farmcpu-iter",
        default=20,
        type=int,
        help=("FarmCPU max loop / iteration." if show_dev_help else argparse.SUPPRESS),
    )
    advanced_group.add_argument(
        "--farmcpu-threshold",
        default=0.05,
        type=float,
        help=("FarmCPU threshold (QTN/p threshold/output)." if show_dev_help else argparse.SUPPRESS),
    )
    advanced_group.add_argument(
        "--farmcpu-threshold-mode",
        default="auto",
        choices=["auto", "fixed", "per_marker"],
        type=str,
        help=(
            "Threshold mode for GAPIT/rMVP launcher. "
            "'fixed' uses --farmcpu-threshold; 'per_marker' uses threshold/m; "
            "'auto' = per_marker in manuscript mode, fixed in legacy mode."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    advanced_group.add_argument(
        "--align-mode",
        default="manuscript",
        choices=["manuscript", "legacy"],
        type=str,
        help=(
            "Benchmark alignment preset. manuscript: threshold mode per_marker. legacy: threshold mode fixed."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    advanced_group.add_argument(
        "--force-pseudo-qtn-cap",
        default=None,
        type=int,
        help=(
            "Optional pseudoQTN cap to apply to all kernels (also passed as FarmCPU bound)."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    advanced_group.add_argument(
        "--keep-temp",
        action="store_true",
        default=False,
        help=("Keep temporary files." if show_dev_help else argparse.SUPPRESS),
    )
    advanced_group.add_argument(
        "--rmvp-reuse-cache",
        action="store_true",
        default=False,
        help=(
            "Reuse existing rMVP preprocessing cache (default: rebuild each run for consistency)."
            if show_dev_help
            else argparse.SUPPRESS
        ),
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
        p.error("`jx benchmark` currently supports one trait at a time; please pass a single `-n/--n` index.")

    if int(args.farmcpu_iter) < 1:
        p.error("--farmcpu-iter must be >= 1.")
    if int(args.farmcpu_nbin) < 1:
        p.error("--farmcpu-nbin must be >= 1.")
    if float(args.farmcpu_threshold) <= 0:
        p.error("--farmcpu-threshold must be > 0.")
    if args.force_pseudo_qtn_cap is not None and int(args.force_pseudo_qtn_cap) < 1:
        p.error("--force-pseudo-qtn-cap must be >= 1.")
    return args


def main() -> None:
    args = parse_args()

    kernels = _parse_kernels(args.kernels)
    if len(kernels) == 0:
        raise ValueError("No valid kernels selected. Use --kernels janusx,gapit,rmvp")

    if args.check:
        checks = _collect_env_checks(args, kernels)
        ok = _print_env_checks(kernels, checks)
        raise SystemExit(0 if ok else 1)

    if not any([args.bfile, args.vcf, args.hmp, args.file]):
        raise ValueError("Missing genotype input. Use one of -bfile/-vcf/-hmp/-file.")
    if not args.pheno:
        raise ValueError("Missing phenotype input: -p/--pheno is required.")

    out_dir = safe_expanduser(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_dir = out_dir / f"{args.prefix}.farmcpu_bench"
    bench_dir.mkdir(parents=True, exist_ok=True)

    gfile, _auto_prefix = _determine_genotype_source_from_args(args)
    gfile = str(gfile)
    geno_flag, _ = _as_geno_flag_and_value(args, gfile)

    ph, selected_trait = _read_pheno(safe_expanduser(args.pheno), args.ncol)
    ph_dir = bench_dir / "input"
    ph_dir.mkdir(parents=True, exist_ok=True)
    ph_for_jx = ph_dir / "pheno_for_jx.tsv"
    ph_for_r = ph_dir / "pheno_for_r.tsv"
    # JanusX expects first col sample ID, trait(s) in following columns.
    ph.to_csv(ph_for_jx, sep="\t", index=False)
    ph.to_csv(ph_for_r, sep="\t", index=False)

    # Compute grid from overlap count (prefer bfile sample IDs when available).
    fam_ids: list[str] = []
    bfile_for_r = ""
    conv_elapsed = 0.0
    conv_rss = math.nan

    needs_r = any(k in {"gapit", "rmvp"} for k in kernels)
    if needs_r:
        conv_log = bench_dir / "logs" / "convert_to_bfile.log"
        conv_time = bench_dir / "logs" / "convert_to_bfile.time"
        bfile_for_r, conv_elapsed, conv_rss = _convert_to_bfile_if_needed(
            args,
            gfile,
            bench_dir / "tmp" / "bfile",
            log_file=conv_log,
            time_file=conv_time,
        )
        fam_ids = _read_fam_ids(bfile_for_r)
    elif args.bfile:
        fam_ids = _read_fam_ids(gfile)

    if fam_ids:
        n_overlap = len(set(fam_ids) & set(ph["Taxa"].astype(str).tolist()))
        n_for_grid = max(2, int(n_overlap))
    else:
        n_for_grid = max(2, int(ph.shape[0]))

    n_snps_for_threshold = 0
    if bfile_for_r:
        n_snps_for_threshold = _count_bim_snps(bfile_for_r)
    elif args.bfile:
        n_snps_for_threshold = _count_bim_snps(gfile)

    bin_size = _parse_number_list(args.farmcpu_bin_size, cast=float)
    farm_bound, farm_bin_selection, farm_bin_size = _compute_farmcpu_grid(
        n_for_grid,
        nbin=args.farmcpu_nbin,
        bound_override=args.farmcpu_bound,
        bin_size=bin_size,
    )
    if args.force_pseudo_qtn_cap is not None:
        farm_bound = int(args.force_pseudo_qtn_cap)
        nbin_den = max(1, int(args.farmcpu_nbin))
        step = max(1, int(farm_bound // nbin_den))
        farm_bin_selection = list(range(step, int(farm_bound) + 1, step))
        if len(farm_bin_selection) == 0:
            farm_bin_selection = [int(farm_bound)]

    align_runtime = _resolve_align_runtime(args, n_snps_for_threshold=n_snps_for_threshold)

    r_launcher = bench_dir / "tmp" / "farmcpu_r_launcher.R"
    r_launcher.parent.mkdir(parents=True, exist_ok=True)
    _build_r_launcher_script(r_launcher)

    rows: list[RunResult] = []
    for kernel in kernels:
        if kernel == "janusx":
            rr = _run_kernel_janusx(
                args,
                gfile=gfile,
                geno_flag=geno_flag,
                pheno_for_jx=ph_for_jx,
                out_dir=bench_dir / "runs",
                farm_bound=farm_bound,
                farm_bin_size=farm_bin_size,
                align_runtime=align_runtime,
            )
            rr = _apply_pseudo_qtn_cap(rr, args.force_pseudo_qtn_cap)
            rows.append(rr)
            continue
        if kernel in {"gapit", "rmvp"}:
            rr = _run_kernel_r(
                args,
                kernel=kernel,
                bfile_prefix=bfile_for_r,
                pheno_for_r=ph_for_r,
                out_dir=bench_dir / "runs",
                farm_bound=farm_bound,
                farm_bin_selection=farm_bin_selection,
                farm_bin_size=farm_bin_size,
                align_runtime=align_runtime,
                r_launcher=r_launcher,
            )
            rr = _apply_pseudo_qtn_cap(rr, args.force_pseudo_qtn_cap)
            rows.append(rr)
            continue

    cfg = {
        "genotype_input": gfile,
        "phenotype_input": str(safe_expanduser(args.pheno)),
        "trait": selected_trait,
        "kernels": kernels,
        "maf": float(args.maf),
        "geno": float(args.geno),
        "threads": int(args.thread),
        "chunksize": int(args.chunksize),
        "qcov": int(args.qcov),
        "n_samples_for_grid": int(n_for_grid),
        "farmcpu_bin_size": [float(x) for x in farm_bin_size],
        "farmcpu_nbin": int(args.farmcpu_nbin),
        "farmcpu_bound": int(farm_bound),
        "farmcpu_bin_selection": [int(x) for x in farm_bin_selection],
        "farmcpu_iter": int(args.farmcpu_iter),
        "farmcpu_threshold": float(args.farmcpu_threshold),
        "farmcpu_threshold_mode": str(align_runtime.threshold_mode),
        "farmcpu_threshold_effective_r": float(align_runtime.farmcpu_threshold_effective),
        "snps_for_threshold": int(align_runtime.snps_for_threshold),
        "align_mode": str(align_runtime.mode),
        "rmvp_engine": str(align_runtime.rmvp_engine),
        "rmvp_vc_method": str(align_runtime.rmvp_vc_method),
        "rmvp_reuse_cache": bool(args.rmvp_reuse_cache),
        "rmvp_force_rebuild": (not bool(args.rmvp_reuse_cache)),
        "rmvp_method_bin": str(align_runtime.rmvp_method_bin),
        "gapit_method_bin": str(align_runtime.gapit_method_bin),
        "force_pseudo_qtn_cap": (None if args.force_pseudo_qtn_cap is None else int(args.force_pseudo_qtn_cap)),
        "conversion_elapsed_sec": float(conv_elapsed),
        "conversion_peak_rss_kb": float(conv_rss),
        "benchmark_dir": str(bench_dir),
    }
    _save_summary(bench_dir / "summary", args.prefix, rows, cfg)

    if not args.keep_temp:
        try:
            if r_launcher.exists():
                r_launcher.unlink()
        except Exception:
            pass

    summary_tsv = bench_dir / "summary" / f"{args.prefix}.{_SUMMARY_TAG}.tsv"
    summary_md = bench_dir / "summary" / f"{args.prefix}.{_SUMMARY_TAG}.md"
    print(f"[DONE] summary table: {summary_tsv}")
    print(f"[DONE] summary markdown: {summary_md}")


if __name__ == "__main__":
    main()
