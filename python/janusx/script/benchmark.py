#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FarmCPU benchmark launcher for JanusX / rMVP.

Key features:
  - One-command benchmark for two kernels:
      * janusx (jxpy gwas -farmcpu)
      * rmvp   (R, explicit MVP.FarmCPU via temporary inlined runner script)
  - Unified FarmCPU grid policy to align JanusX and direct rMVP::MVP.FarmCPU inputs:
      * bin.size      default: 5e5, 5e6, 5e7
      * QTNbound      default: int(sqrt(n / log10(n)))
      * bin.selection derived from QTNbound and nbin:
          step = max(1, QTNbound // nbin)
          seq(step, QTNbound, by=step)
  - Raw pseudoQTN de-redundancy is intentionally kept at `|r| >= 0.7`
    because rMVP hardcodes `FarmCPU.Remove(..., threshold = 0.7)`
  - The direct `rMVP::MVP.FarmCPU(...)` entry point in the installed rMVP
    build does not expose `vc.method`; this benchmark therefore aligns
    `method.bin`, grid, and external threshold wiring, while recording
    `vc.method` only as a requested-but-ineffective metadata field.
  - Outputs:
      * per-kernel GWAS result table
      * per-kernel top-k table
      * benchmark summary table with pseudoQTN/topk/peak RSS/elapsed
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import gc
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
from janusx.script._common.genoio import prepare_packed_stats_ctx_from_plink  # noqa: E402
from janusx.script._common.threads import detect_effective_threads  # noqa: E402
from janusx.script._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog  # noqa: E402
from janusx.script._common.cli_args import (  # noqa: E402
    add_common_covariate_file_or_site_arg,
    add_common_genotype_source_args,
    add_common_out_arg,
    add_common_pheno_arg,
    add_common_prefix_arg,
    add_common_thread_arg,
    add_common_trait_selector_args,
    add_common_variant_filter_args,
    parse_trait_selector_specs,
    resolve_trait_selectors,
)
from janusx.gfreader import inspect_genotype_file  # noqa: E402

try:
    from janusx.janusx import bed_ldblock_r2_rust as _bed_ldblock_r2_rust
except Exception:
    _bed_ldblock_r2_rust = None


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
    qtn_file: str
    topk_count: int
    topk_snps: str
    pseudo_qtn: Optional[int]
    log_file: str
    time_file: str
    note: str
    debug_file: str = ""


@dataclass(frozen=True)
class _AssocTopKRecord:
    snp: str
    chrom: Optional[str]
    pos: Optional[int]
    p: float


class _AssocTopKHeapItem:
    __slots__ = ("record",)

    def __init__(self, record: _AssocTopKRecord):
        self.record = record

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _AssocTopKHeapItem):
            return NotImplemented
        return (self.record.p, self.record.snp) > (other.record.p, other.record.snp)


@dataclass
class EnvCheck:
    component: str
    ok: bool
    detail: str


@dataclass
class AlignRuntime:
    rmvp_vc_method: str
    rmvp_method_bin: str
    farmcpu_threshold: float


@dataclass(frozen=True)
class AssocSite:
    snp: str
    chrom: str
    pos: Optional[int]


def _build_benchmark_marker_manifest(
    *,
    bfile_prefix: str,
    maf: float,
    geno: float,
    snps_only: bool,
    expected_n_samples: int | None = None,
) -> tuple[list[int], int]:
    sample_ids, meta_ctx = prepare_packed_stats_ctx_from_plink(
        bfile_prefix,
        maf=float(maf),
        missing_rate=float(geno),
        het_threshold=0.0,
        snps_only=bool(snps_only),
        expected_n_samples=expected_n_samples,
    )
    active_row_idx = meta_ctx.get("active_row_idx", None)
    if active_row_idx is None:
        raise RuntimeError("Packed stats context missing active_row_idx for benchmark marker manifest.")
    active = np.ascontiguousarray(np.asarray(active_row_idx, dtype=np.int64).reshape(-1), dtype=np.int64)
    n_samples = int(len(sample_ids))
    del active_row_idx
    del meta_ctx
    del sample_ids
    gc.collect()
    return active.tolist(), n_samples


def _write_benchmark_marker_manifest(
    *,
    out_path: Path,
    active_rows: list[int],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_idx0": [int(x) for x in active_rows]}).to_csv(
        out_path,
        sep="\t",
        index=False,
    )
    return out_path


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
        rc_v = subprocess.run(
            ["/usr/bin/time", "-v", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            text=False,
        ).returncode
        if rc_v == 0:
            return ["/usr/bin/time", "-v"]
        rc_l = subprocess.run(
            ["/usr/bin/time", "-l", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            text=False,
        ).returncode
        if rc_l == 0:
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


def _merge_env_with_updates(updates: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for key, val in updates.items():
        env[str(key)] = str(val)
    return env


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


def _write_qtn_sidecar_from_assoc_result(
    result_file: Path,
    qtn_file: Path,
    source_rows_1based: list[int],
) -> None:
    if not result_file.exists():
        return
    wanted: list[int] = []
    seen: set[int] = set()
    for idx1 in source_rows_1based:
        try:
            idx0 = int(idx1) - 1
        except Exception:
            continue
        if idx0 < 0 or idx0 in seen:
            continue
        seen.add(idx0)
        wanted.append(idx0)
    qtn_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("r", encoding="utf-8", errors="ignore", newline="") as rf, qtn_file.open(
        "w", encoding="utf-8", newline=""
    ) as wf:
        reader = csv.reader(rf, delimiter="\t")
        writer = csv.writer(wf, delimiter="\t", lineterminator="\n")
        header = next(reader, None)
        if header is None:
            return
        header_low = [str(col).strip().lower() for col in header]
        has_source_row = "source_row" in header_low
        writer.writerow(header if has_source_row else [*header, "source_row"])
        if len(wanted) == 0:
            return
        wanted_set = set(wanted)
        selected_rows: dict[int, list[str]] = {}
        remaining = len(wanted_set)
        for row_idx0, row in enumerate(reader):
            if row_idx0 not in wanted_set:
                continue
            selected_rows[row_idx0] = row
            remaining -= 1
            if remaining <= 0:
                break
        for row_idx0 in wanted:
            row = selected_rows.get(row_idx0)
            if row is None:
                continue
            writer.writerow(row if has_source_row else [*row, str(row_idx0)])


def _collect_assoc_rows_by_index(
    result_file: Path,
    source_rows_0based: list[int],
) -> tuple[list[str], dict[int, list[str]]]:
    wanted: list[int] = []
    seen: set[int] = set()
    for idx0 in source_rows_0based:
        try:
            idx = int(idx0)
        except Exception:
            continue
        if idx < 0 or idx in seen:
            continue
        seen.add(idx)
        wanted.append(idx)
    if not result_file.exists():
        return [], {}
    with result_file.open("r", encoding="utf-8", errors="ignore", newline="") as rf:
        reader = csv.reader(rf, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return [], {}
        if len(wanted) == 0:
            return header, {}
        wanted_set = set(wanted)
        selected_rows: dict[int, list[str]] = {}
        remaining = len(wanted_set)
        for row_idx0, row in enumerate(reader):
            if row_idx0 not in wanted_set:
                continue
            selected_rows[row_idx0] = row
            remaining -= 1
            if remaining <= 0:
                break
    return header, selected_rows


def _write_rmvp_seqqtn_debug_sidecar_from_assoc_result(
    result_file: Path,
    debug_file: Path,
    seq_blocks: list[dict[str, Any]],
) -> None:
    if not result_file.exists():
        return
    source_rows_0based: list[int] = []
    for block in seq_blocks:
        for idx1 in block.get("source_rows_1based", []):
            try:
                idx0 = int(idx1) - 1
            except Exception:
                continue
            if idx0 >= 0:
                source_rows_0based.append(idx0)
    header, selected_rows = _collect_assoc_rows_by_index(result_file, source_rows_0based)
    if not header:
        return
    debug_file.parent.mkdir(parents=True, exist_ok=True)
    debug_header = [
        "block_idx",
        "current_loop",
        "is_final_block",
        "block_qtn_count",
        "qtn_order",
        "source_row1",
        "source_row0",
        *header,
    ]
    final_block_idx = len(seq_blocks)
    with debug_file.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.writer(wf, delimiter="\t", lineterminator="\n")
        writer.writerow(debug_header)
        for block in seq_blocks:
            block_idx = int(block.get("block_idx", 0) or 0)
            current_loop = block.get("current_loop", None)
            current_loop_out = "" if current_loop is None else str(int(current_loop))
            source_rows_1based = [int(x) for x in block.get("source_rows_1based", []) if int(x) > 0]
            block_qtn_count = len(source_rows_1based)
            if block_qtn_count == 0:
                continue
            for qtn_order, idx1 in enumerate(source_rows_1based, start=1):
                idx0 = int(idx1) - 1
                row = selected_rows.get(idx0)
                if row is None:
                    continue
                writer.writerow(
                    [
                        str(block_idx),
                        current_loop_out,
                        "1" if block_idx == final_block_idx else "0",
                        str(block_qtn_count),
                        str(qtn_order),
                        str(idx1),
                        str(idx0),
                        *row,
                    ]
                )


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
    top = _extract_topk_streaming_assoc(assoc_file, max(1, int(k)))
    out_topk.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(out_topk, sep="\t", index=False)
    top_snps = ";".join(top["SNP"].astype(str).head(10).tolist())
    return int(top.shape[0]), top_snps


def _assoc_result_delimiter_from_header(header_line: str) -> Optional[str]:
    if "\t" in header_line:
        return "\t"
    if "," in header_line:
        return ","
    return None


def _assoc_result_header_map(header_fields: list[str]) -> dict[str, int]:
    return {
        str(col).strip().lstrip("#").lower(): idx
        for idx, col in enumerate(header_fields)
    }


def _assoc_record_from_fields(
    fields: list[str],
    *,
    p_idx: int,
    chr_idx: Optional[int],
    pos_idx: Optional[int],
    snp_idx: Optional[int],
) -> Optional[_AssocTopKRecord]:
    try:
        p_raw = float(str(fields[p_idx]).strip())
    except Exception:
        return None
    if (not math.isfinite(p_raw)) or p_raw <= 0.0 or p_raw > 1.0:
        return None

    chrom_out: Optional[str] = None
    pos_out: Optional[int] = None
    snp_out = ""
    if chr_idx is not None and pos_idx is not None:
        try:
            chrom_raw = str(fields[chr_idx]).strip()
            chrom_norm = re.sub(r"^chr", "", chrom_raw, flags=re.IGNORECASE).strip()
            pos_raw = float(str(fields[pos_idx]).strip())
            if chrom_norm == "" or (not math.isfinite(pos_raw)):
                return None
            pos_int = int(round(pos_raw))
        except Exception:
            return None
        chrom_out = chrom_norm
        pos_out = pos_int
        snp_out = f"{chrom_norm}_{pos_int}"
    elif snp_idx is not None:
        snp_out = str(fields[snp_idx]).strip()
        if snp_out == "":
            return None
    else:
        return None

    return _AssocTopKRecord(
        snp=snp_out,
        chrom=chrom_out,
        pos=pos_out,
        p=float(p_raw),
    )


def _prune_topk_heap(
    heap: list[_AssocTopKHeapItem],
    best_by_snp: dict[str, _AssocTopKRecord],
) -> None:
    while heap:
        current = best_by_snp.get(heap[0].record.snp)
        if current is heap[0].record:
            break
        heapq.heappop(heap)


def _extract_topk_streaming_assoc(
    assoc_file: Path,
    k: int,
) -> pd.DataFrame:
    heap: list[_AssocTopKHeapItem] = []
    best_by_snp: dict[str, _AssocTopKRecord] = {}

    with assoc_file.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        header_line = fh.readline()
        if header_line == "":
            raise ValueError(f"Empty association result: {assoc_file}")
        delim = _assoc_result_delimiter_from_header(header_line)
        if delim is None:
            header_fields = header_line.strip().split()

            def row_iter() -> Iterable[list[str]]:
                for line in fh:
                    line = line.strip()
                    if line == "":
                        continue
                    yield line.split()
        else:
            header_fields = next(csv.reader([header_line], delimiter=delim))

            def row_iter() -> Iterable[list[str]]:
                reader = csv.reader(fh, delimiter=delim)
                for row in reader:
                    if len(row) == 0:
                        continue
                    yield row

        cols = _assoc_result_header_map(list(header_fields))
        p_col = _first_col(cols, _P_COL_KEYS)
        if p_col is None:
            raise ValueError("P-value column not found in GWAS result.")
        chr_col = _first_col(cols, _CHR_COL_KEYS)
        pos_col = _first_col(cols, _POS_COL_KEYS)
        snp_col = _first_col(cols, _SNP_COL_KEYS)
        if (chr_col is None or pos_col is None) and snp_col is None:
            raise ValueError("Neither (chr,pos) nor SNP id columns found.")

        p_idx = int(cols[p_col])
        chr_idx = None if chr_col is None else int(cols[chr_col])
        pos_idx = None if pos_col is None else int(cols[pos_col])
        snp_idx = None if snp_col is None else int(cols[snp_col])
        need_cols = [p_idx]
        if chr_idx is not None:
            need_cols.append(chr_idx)
        if pos_idx is not None:
            need_cols.append(pos_idx)
        if snp_idx is not None:
            need_cols.append(snp_idx)
        max_need_idx = max(need_cols)

        for fields in row_iter():
            if len(fields) <= max_need_idx:
                continue
            rec = _assoc_record_from_fields(
                fields,
                p_idx=p_idx,
                chr_idx=chr_idx,
                pos_idx=pos_idx,
                snp_idx=snp_idx,
            )
            if rec is None:
                continue

            current = best_by_snp.get(rec.snp)
            if current is not None:
                if (rec.p, rec.snp) < (current.p, current.snp):
                    best_by_snp[rec.snp] = rec
                    heapq.heappush(heap, _AssocTopKHeapItem(rec))
                continue

            _prune_topk_heap(heap, best_by_snp)
            if len(best_by_snp) < k:
                best_by_snp[rec.snp] = rec
                heapq.heappush(heap, _AssocTopKHeapItem(rec))
                continue

            if not heap:
                best_by_snp[rec.snp] = rec
                heapq.heappush(heap, _AssocTopKHeapItem(rec))
                continue

            worst = heap[0].record
            if (rec.p, rec.snp) < (worst.p, worst.snp):
                del best_by_snp[worst.snp]
                heapq.heapreplace(heap, _AssocTopKHeapItem(rec))
                best_by_snp[rec.snp] = rec

    top_rows = sorted(best_by_snp.values(), key=lambda r: (r.p, r.snp))[:k]
    return pd.DataFrame(
        [
            {
                "SNP": rec.snp,
                "CHR": (rec.chrom if rec.chrom is not None else pd.NA),
                "POS": (rec.pos if rec.pos is not None else pd.NA),
                "P": float(rec.p),
            }
            for rec in top_rows
        ],
        columns=["SNP", "CHR", "POS", "P"],
    )


def _read_pheno(path: Path, ncol: Optional[list[object]]) -> tuple[pd.DataFrame, str]:
    df = _read_table_guess(path)
    if df.shape[1] < 2:
        raise ValueError("Phenotype file must contain sample ID and at least one trait column.")
    id_col = df.columns[0]
    trait_cols = list(df.columns[1:])
    if ncol is None:
        trait_col = trait_cols[0]
    else:
        if len(ncol) == 0:
            raise ValueError("`-n/--n` is empty. Provide one phenotype selector.")
        if len(ncol) > 1:
            raise ValueError("`jx benchmark` currently supports one trait at a time; please pass a single `-n/--n` selector.")
        selected_cols, invalid_specs = resolve_trait_selectors(trait_cols, ncol, label="-n/--n")
        if len(selected_cols) == 0:
            raise ValueError(
                f"`-n/--n` not found: {ncol[0]}. Valid index range is 0..{int(df.shape[1]) - 2} "
                f"or one of columns: {', '.join(str(c) for c in trait_cols[:10])}"
                + (" ..." if len(trait_cols) > 10 else "")
            )
        if len(invalid_specs) > 0:
            raise ValueError(f"`-n/--n` not found: {invalid_specs[0]}")
        trait_col = trait_cols[int(selected_cols[0])]

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


def _resolve_effective_farmcpu_threshold(raw_threshold: Optional[float], n_snps: int) -> float:
    if raw_threshold is not None:
        thr = float(raw_threshold)
        if not math.isfinite(thr) or thr <= 0:
            raise ValueError("--farmcpu-threshold must be > 0.")
        return thr
    return 1.0 / float(max(1, int(n_snps)))


def _resolve_align_runtime(args: argparse.Namespace) -> AlignRuntime:
    # Single aligned benchmark mode:
    # direct rMVP::MVP.FarmCPU(...) with explicit FaST-LMM bin setting.
    # The direct FarmCPU entry point does not expose vc.method in current rMVP.
    return AlignRuntime(
        rmvp_vc_method="BRENT",
        rmvp_method_bin="FaST-LMM",
        farmcpu_threshold=1.0,
    )


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


def _resolve_rscript_path() -> str:
    cands: list[Path] = []
    env_bin = Path(sys.executable).resolve().parent
    cands.append(env_bin / "Rscript")
    for root_var in ("CONDA_PREFIX", "MAMBA_ROOT_PREFIX", "MINICONDA_PREFIX"):
        root = os.environ.get(root_var, "").strip()
        if not root:
            continue
        root_path = Path(root)
        cands.append(root_path / "bin" / "Rscript")
        cands.append(root_path / "lib" / "R" / "bin" / "Rscript")
        envs_dir = root_path / "envs"
        if envs_dir.exists():
            for sub in sorted(envs_dir.iterdir()):
                cands.append(sub / "bin" / "Rscript")
                cands.append(sub / "lib" / "R" / "bin" / "Rscript")
    home = Path.home()
    for base in (
        home / "miniconda3",
        home / "anaconda3",
        home / "mambaforge",
        home / ".conda",
        home / ".docker_mamba",
    ):
        cands.append(base / "bin" / "Rscript")
        cands.append(base / "lib" / "R" / "bin" / "Rscript")
        envs_dir = base / "envs"
        if envs_dir.exists():
            for sub in sorted(envs_dir.iterdir()):
                cands.append(sub / "bin" / "Rscript")
                cands.append(sub / "lib" / "R" / "bin" / "Rscript")
    which = shutil.which("Rscript")
    if which:
        cands.append(Path(which))
    seen: set[str] = set()
    for cand in cands:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists() and cand.is_file() and os.access(str(cand), os.X_OK):
            return str(cand)
    return ""


def _parse_kernels(raw: str) -> list[str]:
    kernels = [k.strip().lower() for k in str(raw).split(",") if k.strip()]
    kernels = [k for k in kernels if k in {"janusx", "rmvp"}]
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
    need_r = "rmvp" in kernels

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
        if str(getattr(args, "pseudo_qtn_match", "exact")).strip().lower() == "ld":
            items.append(
                EnvCheck(
                    "janusx.ld_backend",
                    _bed_ldblock_r2_rust is not None,
                    "bed_ldblock_r2_rust available"
                    if _bed_ldblock_r2_rust is not None
                    else "missing bed_ldblock_r2_rust; LD-match summary would fall back to exact",
                )
            )

    r_bin = ""
    r_ok = True

    if need_r:
        r_bin = _resolve_rscript_path()

        r_ok = bool(r_bin) and Path(r_bin).exists()
        items.append(
            EnvCheck(
                "rscript",
                r_ok,
                r_bin if r_ok else "Rscript not found in current env PATH",
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


def _guess_janusx_qtn_file_from_result_file(result_file: Path) -> Optional[Path]:
    cands: list[Path] = []
    if result_file.suffix.lower() == ".tsv":
        cands.append(result_file.with_suffix(".qtn"))
        cands.append(result_file.with_suffix(".qtn.tsv"))
    cands.extend(sorted(result_file.parent.glob("*.farmcpu.qtn")))
    cands.extend(sorted(result_file.parent.glob("*.farmcpu.qtn.tsv")))
    cands.extend(sorted(result_file.parent.glob("*.qtn")))
    cands.extend(sorted(result_file.parent.glob("*.qtn.tsv")))
    seen: set[str] = set()
    for cand in cands:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand
    return None


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

    # rMVP logs often print pseudo-QTN lists as:
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


def _build_rmvp_runner_script(path: Path) -> None:
    script = r"""#!/usr/bin/env Rscript
parse_args <- function(args) {
  out <- list()
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument: %s", key), call. = FALSE)
    }
    key <- substring(key, 3L)
    if (i == length(args) || startsWith(args[[i + 1L]], "--")) {
      out[[key]] <- "true"
      i <- i + 1L
    } else {
      out[[key]] <- args[[i + 1L]]
      i <- i + 2L
    }
  }
  out
}

require_opt <- function(opts, key) {
  val <- opts[[key]]
  if (is.null(val) || !nzchar(val)) {
    stop(sprintf("Missing required option: --%s", key), call. = FALSE)
  }
  val
}

parse_csv_numeric <- function(text, as_int = FALSE) {
  if (!nzchar(text)) return(numeric())
  parts <- strsplit(text, ",", fixed = TRUE)[[1]]
  parts <- trimws(parts)
  parts <- parts[nzchar(parts)]
  if (!length(parts)) return(numeric())
  vals <- suppressWarnings(as.numeric(parts))
  vals <- vals[is.finite(vals)]
  if (!length(vals)) return(numeric())
  if (as_int) {
    return(as.integer(round(vals)))
  }
  vals
}

detect_qtn_from_lines <- function(lines) {
  if (!length(lines)) return(NA_integer_)
  final_count <- NA_integer_

  for (idx in seq_along(lines)) {
    if (!grepl("number of covariates in current loop", lines[[idx]], ignore.case = TRUE)) next
    if (idx >= length(lines)) next
    nums <- regmatches(lines[[idx + 1L]], gregexpr("\\b[0-9]+\\b", lines[[idx + 1L]], perl = TRUE))[[1]]
    vv <- suppressWarnings(as.integer(nums))
    vv <- vv[is.finite(vv)]
    if (length(vv) > 0L) {
      final_count <- as.integer(vv[[length(vv)]])
    }
  }

  if (is.finite(final_count)) return(final_count)

  for (idx in seq_along(lines)) {
    if (!grepl("\\bseqQTN\\b", lines[[idx]], ignore.case = TRUE)) next
    count <- 0L
    j <- idx + 1L
    while (j <= length(lines)) {
      lns <- trimws(lines[[j]])
      if (!nzchar(lns)) break
      if (grepl("(current loop|optimizing|scanning|number of covariates|farmcpu\\.lm|genomic inflation)", lns, ignore.case = TRUE)) break
      nums <- regmatches(lns, gregexpr("\\b[0-9]+\\b", lns, perl = TRUE))[[1]]
      if (grepl("^\\[[0-9]+\\]\\s*", lns) && length(nums)) {
        nums <- nums[-1]
      }
      count <- count + length(nums)
      j <- j + 1L
    }
    if (count > 0L) {
      final_count <- as.integer(count)
    }
  }

  final_count
}

compute_marker_freq <- function(geno, ind_idx, mrk_idx, threads, max_line, n_markers_total) {
  nsnp_total <- as.integer(n_markers_total)
  if (!is.finite(nsnp_total) || nsnp_total <= 0L) {
    stop("Invalid marker count for rMVP frequency calculation.", call. = FALSE)
  }
  geno_nr <- tryCatch(as.integer(nrow(geno)), error = function(e) NA_integer_)
  geno_nc <- tryCatch(as.integer(ncol(geno)), error = function(e) NA_integer_)
  marker_by_col <- is.finite(geno_nc) && geno_nc == nsnp_total
  marker_by_row <- is.finite(geno_nr) && geno_nr == nsnp_total
  if (!marker_by_col && !marker_by_row) {
    stop(
      sprintf(
        "Cannot infer rMVP genotype orientation: nrow=%s ncol=%s n_markers=%s",
        as.character(geno_nr),
        as.character(geno_nc),
        as.character(nsnp_total)
      ),
      call. = FALSE
    )
  }
  idx <- as.integer(ind_idx)
  idx <- idx[is.finite(idx)]
  if (!length(idx)) {
    idx <- if (marker_by_col) seq_len(geno_nr) else seq_len(geno_nc)
  }
  if (!length(idx)) {
    stop("No valid matched sample indices for marker-frequency calculation.", call. = FALSE)
  }
  mrk <- as.integer(mrk_idx)
  mrk <- mrk[is.finite(mrk)]
  mrk <- unique(mrk)
  if (!length(mrk)) {
    mrk <- NULL
  }
  # rMVP internals vary by version; prefer its native helper when available,
  # otherwise fall back to a direct big.matrix row-mean over the selected samples.
  big_row_mean <- tryCatch(getFromNamespace("BigRowMean", "rMVP"), error = function(e) NULL)
  if (is.function(big_row_mean)) {
    freq_all <- big_row_mean(
      geno@address,
      marker_by_col,
      step = max(1L, as.integer(ifelse(is.finite(max_line) && max_line > 0L, max_line, 10000L))),
      threads = threads,
      geno_ind = idx,
      verbose = FALSE
    ) / 2
    if (!is.null(mrk)) {
      return(freq_all[mrk])
    }
    return(freq_all)
  }
  chunk <- as.integer(max(1L, ifelse(is.finite(max_line) && max_line > 0L, max_line, 10000L)))
  target <- if (is.null(mrk)) seq_len(nsnp_total) else mrk
  out <- numeric(length(target))
  for (start in seq.int(1L, length(target), by = chunk)) {
    end <- min(length(target), start + chunk - 1L)
    sel <- target[start:end]
    if (marker_by_col) {
      block <- geno[idx, sel, drop = FALSE]
      out[start:end] <- colMeans(block, na.rm = TRUE) / 2
    } else {
      block <- geno[sel, idx, drop = FALSE]
      out[start:end] <- rowMeans(block, na.rm = TRUE) / 2
    }
  }
  out
}

function_has_formal <- function(fun, arg) {
  fm <- tryCatch(formals(fun), error = function(e) NULL)
  if (is.null(fm)) return(FALSE)
  fn_names <- names(fm)
  if (is.null(fn_names) || !length(fn_names)) return(FALSE)
  ("..." %in% fn_names) || (arg %in% fn_names)
}

filter_named_args <- function(fun, args, label) {
  fm <- tryCatch(formals(fun), error = function(e) NULL)
  if (is.null(fm)) return(args)
  fn_names <- names(fm)
  if (is.null(fn_names) || !length(fn_names) || ("..." %in% fn_names)) {
    return(args)
  }
  arg_names <- names(args)
  keep <- arg_names %in% fn_names
  dropped <- arg_names[!keep]
  if (length(dropped)) {
    cat("[INFO]", label, "dropping unsupported args:", paste(dropped, collapse = ", "), "\n")
  }
  args[keep]
}

compute_rmvp_kinship <- function(kin_fun, kin_args, threads) {
  fm <- tryCatch(formals(kin_fun), error = function(e) NULL)
  fn_names <- if (is.null(fm)) character() else names(fm)
  supports_priority <- length(fn_names) && ("priority" %in% fn_names || "..." %in% fn_names)
  supports_maxline <- length(fn_names) && ("maxLine" %in% fn_names || "..." %in% fn_names)
  kin_args_use <- kin_args
  if (supports_priority) {
    kin_args_use[["priority"]] <- "memory"
  }
  if (supports_maxline && is.null(kin_args_use[["maxLine"]])) {
    kin_args_use[["maxLine"]] <- 2048L
  }
  mode_label <- if (supports_priority) "priority='memory'" else "chunked maxLine fallback"
  cat("[INFO] Computing rMVP kinship with", mode_label, sprintf("(cpu=%d).\n", threads))
  first_try <- try(do.call(kin_fun, filter_named_args(kin_fun, kin_args_use, "MVP.K.VanRaden")), silent = TRUE)
  if (!inherits(first_try, "try-error")) {
    return(first_try)
  }
  msg <- conditionMessage(attr(first_try, "condition"))
  if (!supports_maxline) {
    stop(first_try)
  }
  cat("[WARN] Initial rMVP kinship attempt failed; retrying with maxLine=512:", msg, "\n")
  kin_args_use[["maxLine"]] <- 512L
  second_try <- try(do.call(kin_fun, filter_named_args(kin_fun, kin_args_use, "MVP.K.VanRaden")), silent = TRUE)
  if (!inherits(second_try, "try-error")) {
    return(second_try)
  }
  stop(second_try)
}

subset_geno_for_individuals <- function(geno, ind_idx, n_markers, cache_prefix, label) {
  idx <- as.integer(ind_idx)
  idx <- idx[is.finite(idx)]
  if (!length(idx)) {
    stop(sprintf("%s: no valid sample indices supplied for genotype subsetting.", label), call. = FALSE)
  }
  nsnp <- as.integer(n_markers)
  if (!is.finite(nsnp) || nsnp <= 0L) {
    stop(sprintf("%s: invalid marker count for genotype subsetting.", label), call. = FALSE)
  }
  geno_nr <- tryCatch(as.integer(nrow(geno)), error = function(e) NA_integer_)
  geno_nc <- tryCatch(as.integer(ncol(geno)), error = function(e) NA_integer_)
  marker_by_col <- is.finite(geno_nc) && geno_nc == nsnp
  marker_by_row <- is.finite(geno_nr) && geno_nr == nsnp
  if (!marker_by_col && !marker_by_row) {
    stop(
      sprintf(
        "%s: cannot infer genotype orientation for compatibility subset (nrow=%s, ncol=%s, n_markers=%s).",
        label,
        as.character(geno_nr),
        as.character(geno_nc),
        as.character(nsnp)
      ),
      call. = FALSE
    )
  }
  list(
    geno = geno,
    ind_idx = idx,
    marker_by_col = marker_by_col
  )
}

subset_geno_for_markers <- function(geno, marker_idx0, n_markers, cache_prefix, label) {
  nsnp <- as.integer(n_markers)
  if (!is.finite(nsnp) || nsnp <= 0L) {
    stop(sprintf("%s: invalid marker count for genotype subsetting.", label), call. = FALSE)
  }
  idx0 <- as.integer(marker_idx0)
  idx0 <- idx0[is.finite(idx0)]
  idx0 <- unique(idx0)
  idx0 <- idx0[idx0 >= 0L & idx0 < nsnp]
  if (!length(idx0)) {
    stop(sprintf("%s: no valid marker indices supplied for genotype subsetting.", label), call. = FALSE)
  }
  idx <- idx0 + 1L
  geno_nr <- tryCatch(as.integer(nrow(geno)), error = function(e) NA_integer_)
  geno_nc <- tryCatch(as.integer(ncol(geno)), error = function(e) NA_integer_)
  marker_by_col <- is.finite(geno_nc) && geno_nc == nsnp
  marker_by_row <- is.finite(geno_nr) && geno_nr == nsnp
  if (!marker_by_col && !marker_by_row) {
    stop(
      sprintf(
        "%s: cannot infer genotype orientation for marker subset (nrow=%s, ncol=%s, n_markers=%s).",
        label,
        as.character(geno_nr),
        as.character(geno_nc),
        as.character(nsnp)
      ),
      call. = FALSE
    )
  }
  cache_dir <- dirname(cache_prefix)
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  backingfile <- paste0(basename(cache_prefix), ".", label, ".geno.bin")
  descriptorfile <- paste0(basename(cache_prefix), ".", label, ".geno.desc")
  bin_path <- file.path(cache_dir, backingfile)
  desc_path <- file.path(cache_dir, descriptorfile)
  if (file.exists(bin_path)) unlink(bin_path, force = TRUE)
  if (file.exists(desc_path)) unlink(desc_path, force = TRUE)
  target_nrow <- if (marker_by_col) geno_nr else length(idx)
  target_ncol <- if (marker_by_col) length(idx) else geno_nc
  bm <- bigmemory::filebacked.big.matrix(
    nrow = target_nrow,
    ncol = target_ncol,
    type = "char",
    backingfile = backingfile,
    backingpath = cache_dir,
    descriptorfile = descriptorfile,
    dimnames = c(NULL, NULL)
  )
  old_typecast_warn <- getOption("bigmemory.typecast.warning")
  options(bigmemory.typecast.warning = FALSE)
  on.exit(options(bigmemory.typecast.warning = old_typecast_warn), add = TRUE)
  chunk <- 1024L
  if (marker_by_col) {
    starts <- seq.int(1L, length(idx), by = chunk)
    for (start in starts) {
      end <- min(length(idx), start + chunk - 1L)
      src <- idx[start:end]
      dst <- start:end
      bm[, dst] <- geno[, src, drop = FALSE]
    }
  } else {
    starts <- seq.int(1L, length(idx), by = chunk)
    for (start in starts) {
      end <- min(length(idx), start + chunk - 1L)
      src <- idx[start:end]
      dst <- start:end
      bm[dst, ] <- geno[src, , drop = FALSE]
    }
  }
  flush(bm)
  list(
    geno = bm,
    marker_by_col = marker_by_col,
    kept_idx0 = idx0
  )
}

patch_rmvp_farmcpu_burger_if_needed <- function() {
  ns <- asNamespace("rMVP")
  if (!exists("FarmCPU.Burger", where = ns, inherits = FALSE)) {
    return(FALSE)
  }
  original <- get("FarmCPU.Burger", envir = ns)
  body_text <- paste(deparse(body(original)), collapse = "\n")
  if (!grepl("m > 2", body_text, fixed = TRUE)) {
    return(FALSE)
  }
  patched <- function(Y = NULL, CV = NULL, GK = NULL, ncpus = 2, method = "FaST-LMM") {
    if (!is.null(CV)) {
      CV = as.matrix(CV)
      theCV = as.matrix(cbind(matrix(1, nrow(CV), 1), CV))
    } else {
      theCV = matrix(1, nrow(Y), 1)
    }
    if (!is.null(GK)) {
      if (is.null(dim(GK))) {
        n = length(GK)
        m = if (n > 0L) 1L else 0L
      } else {
        n = nrow(GK)
        m = ncol(GK)
      }
      if (m >= 2L) {
        theGK = as.matrix(GK)
      } else {
        theGK = if (m == 1L && n > 0L) matrix(GK, nrow = n, ncol = 1L) else NULL
      }
    } else {
      theGK = GK
    }
    if (
      is.null(theGK) ||
      !length(theGK) ||
      is.null(dim(theGK)) ||
      nrow(theGK) < 1L ||
      ncol(theGK) < 1L
    ) {
      return(list(REMLs = Inf, vg = NA_real_, ve = NA_real_, delta = NA_real_))
    }
    if (method == "FaST-LMM") {
      myFaSTREML = FarmCPU.FaSTLMM.LL(pheno = matrix(Y[, -1], nrow(Y), 1), snp.pool = theGK, X0 = theCV, ncpus = ncpus)
      REMLs = -2 * myFaSTREML$LL
      delta = myFaSTREML$delta
      vg = myFaSTREML$vg
      ve = myFaSTREML$ve
    }
    if (method == "EMMA") {
      K <- MVP.K.VanRaden(M = as.big.matrix(theGK), verbose = FALSE)
      myEMMAREML <- MVP.EMMA.Vg.Ve(y = matrix(Y[, -1], nrow(Y), 1), X = theCV, K = K)
      REMLs = -2 * myEMMAREML$REML
      delta = myEMMAREML$delta
      vg = myEMMAREML$vg
      ve = myEMMAREML$ve
    }
    return(list(REMLs = REMLs, vg = vg, ve = ve, delta = delta))
  }
  environment(patched) <- environment(original)
  unlockBinding("FarmCPU.Burger", ns)
  on.exit(lockBinding("FarmCPU.Burger", ns), add = TRUE)
  assign("FarmCPU.Burger", patched, envir = ns)
  TRUE
}

patch_rmvp_fastlmm_ll_if_needed <- function() {
  ns <- asNamespace("rMVP")
  if (!exists("FarmCPU.FaSTLMM.LL", where = ns, inherits = FALSE)) {
    return(FALSE)
  }
  original <- get("FarmCPU.FaSTLMM.LL", envir = ns)
  body_text <- paste(deparse(body(original)), collapse = "\n")
  needle <- "snp.pool = snp.pool[, ]"
  if (!grepl(needle, body_text, fixed = TRUE)) {
    return(FALSE)
  }
  replacement <- paste(
    "if (!is.null(snp.pool)) {",
    "    if (is.null(dim(snp.pool))) {",
    "        snp.pool = matrix(snp.pool, ncol = 1L)",
    "    } else {",
    "        snp.pool = snp.pool[, , drop = FALSE]",
    "    }",
    "}",
    sep = "\n"
  )
  patched_body <- sub(needle, replacement, body_text, fixed = TRUE)
  patched <- original
  body(patched) <- parse(text = patched_body)[[1]]
  environment(patched) <- environment(original)
  unlockBinding("FarmCPU.FaSTLMM.LL", ns)
  on.exit(lockBinding("FarmCPU.FaSTLMM.LL", ns), add = TRUE)
  assign("FarmCPU.FaSTLMM.LL", patched, envir = ns)
  TRUE
}

patch_rmvp_farmcpu_zero_p_guard_if_needed <- function() {
  ns <- asNamespace("rMVP")
  if (!exists("MVP.FarmCPU", where = ns, inherits = FALSE)) {
    return(FALSE)
  }
  original <- get("MVP.FarmCPU", envir = ns)
  body_text <- paste(deparse(body(original)), collapse = "\n")
  needle <- "P[P == 0] <- min(P[P != 0], na.rm = TRUE) * 0.01"
  if (!grepl(needle, body_text, fixed = TRUE)) {
    return(FALSE)
  }
  replacement <- paste(
    "zero_idx <- which(P == 0)",
    "if (length(zero_idx)) {",
    "    p_nonzero <- P[is.finite(P) & P != 0]",
    "    if (length(p_nonzero)) {",
    "        P[zero_idx] <- min(p_nonzero, na.rm = TRUE) * 0.01",
    "    }",
    "}",
    sep = "\n"
  )
  patched_body <- gsub(needle, replacement, body_text, fixed = TRUE)
  patched <- original
  body(patched) <- parse(text = patched_body)[[1]]
  environment(patched) <- environment(original)
  unlockBinding("MVP.FarmCPU", ns)
  on.exit(lockBinding("MVP.FarmCPU", ns), add = TRUE)
  assign("MVP.FarmCPU", patched, envir = ns)
  TRUE
}

write_meta <- function(meta_file, exit_code, pseudo_qtn, inner_log, note, pseudo_qtn_metric) {
  meta <- data.frame(
    key = c("exit_code", "pseudo_qtn", "pseudo_qtn_metric", "inner_log", "note"),
    value = c(
      as.character(as.integer(exit_code)),
      ifelse(is.na(pseudo_qtn), "NA", as.character(as.integer(pseudo_qtn))),
      pseudo_qtn_metric,
      inner_log,
      note
    ),
    stringsAsFactors = FALSE
  )
  utils::write.table(meta, file = meta_file, sep = "\t", row.names = FALSE, quote = FALSE)
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
bfile <- require_opt(opts, "bfile")
pheno <- require_opt(opts, "pheno")
out_file <- require_opt(opts, "out-file")
workdir <- require_opt(opts, "workdir")
inner_log <- require_opt(opts, "inner-log")
meta_file <- require_opt(opts, "meta-file")
cache_prefix <- require_opt(opts, "cache-prefix")
marker_manifest_path <- ifelse(is.null(opts[["marker-manifest"]]), "", trimws(opts[["marker-manifest"]]))

threads <- suppressWarnings(as.integer(opts[["threads"]]))
if (!is.finite(threads) || threads < 1L) threads <- 1L
max_line <- suppressWarnings(as.integer(opts[["max-line"]]))
if (!is.finite(max_line) || max_line < 1L) max_line <- 10000L
max_loop <- suppressWarnings(as.integer(opts[["farmcpu-max-loop"]]))
if (!is.finite(max_loop) || max_loop < 1L) max_loop <- 30L
threshold <- suppressWarnings(as.numeric(opts[["farmcpu-threshold"]]))
threshold_output <- suppressWarnings(as.numeric(opts[["farmcpu-threshold-output"]]))
p_threshold <- suppressWarnings(as.numeric(opts[["farmcpu-p-threshold"]]))
p_threshold_raw <- ifelse(is.null(opts[["farmcpu-p-threshold"]]), "", trimws(opts[["farmcpu-p-threshold"]]))
qtn_threshold <- suppressWarnings(as.numeric(opts[["farmcpu-qtn-threshold"]]))
qtn_threshold_raw <- ifelse(is.null(opts[["farmcpu-qtn-threshold"]]), "", trimws(opts[["farmcpu-qtn-threshold"]]))
farmcpu_bound <- suppressWarnings(as.integer(opts[["farmcpu-bound"]]))
if (length(farmcpu_bound) == 0L || !is.finite(farmcpu_bound) || farmcpu_bound < 1L) farmcpu_bound <- NULL
maf <- suppressWarnings(as.numeric(opts[["farmcpu-maf"]]))
if (!is.finite(maf) || maf <= 0 || maf >= 0.5) maf <- NULL
npc_farmcpu <- suppressWarnings(as.integer(opts[["npc-farmcpu"]]))
if (!is.finite(npc_farmcpu) || npc_farmcpu < 1L) npc_farmcpu <- NULL
force_rebuild <- tolower(ifelse(is.null(opts[["force-rebuild"]]), "0", opts[["force-rebuild"]])) %in% c("1", "true", "yes")
vc_method <- toupper(ifelse(is.null(opts[["rmvp-vc-method"]]), "BRENT", opts[["rmvp-vc-method"]]))
if (!vc_method %in% c("BRENT", "EMMA", "HE")) vc_method <- "BRENT"
method_bin <- ifelse(is.null(opts[["rmvp-method-bin"]]), "FaST-LMM", opts[["rmvp-method-bin"]])
if (!nzchar(method_bin)) method_bin <- "FaST-LMM"
bin_size <- parse_csv_numeric(ifelse(is.null(opts[["farmcpu-bin-size"]]), "", opts[["farmcpu-bin-size"]]), as_int = FALSE)
if (!length(bin_size)) bin_size <- c(5e5, 5e6, 5e7)
bin_selection <- parse_csv_numeric(ifelse(is.null(opts[["farmcpu-bin-selection"]]), "", opts[["farmcpu-bin-selection"]]), as_int = TRUE)
if (!length(bin_selection)) bin_selection <- as.integer(1L)

dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
dir.create(workdir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(inner_log), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(meta_file), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(cache_prefix), recursive = TRUE, showWarnings = FALSE)
if (file.exists(inner_log)) unlink(inner_log, force = TRUE)
file.create(inner_log, showWarnings = FALSE)
sink(inner_log, split = TRUE)

suppressPackageStartupMessages({
  library(rMVP)
  library(bigmemory)
})
if (patch_rmvp_fastlmm_ll_if_needed()) {
  cat("[INFO] Patched rMVP::FarmCPU.FaSTLMM.LL to preserve single-column pseudoQTN matrices.\n")
}
if (patch_rmvp_farmcpu_burger_if_needed()) {
  cat("[INFO] Patched rMVP::FarmCPU.Burger for 1/2-column FaST-LMM pseudoQTN compatibility.\n")
}
if (patch_rmvp_farmcpu_zero_p_guard_if_needed()) {
  cat("[INFO] Patched rMVP::MVP.FarmCPU zero-P fallback to avoid empty min() warnings.\n")
}

status <- 0L
note <- "ok"

tryCatch(
  {
    cache_files <- c(
      paste0(cache_prefix, ".geno.bin"),
      paste0(cache_prefix, ".geno.desc"),
      paste0(cache_prefix, ".geno.map"),
      paste0(cache_prefix, ".geno.ind")
    )
    if (force_rebuild) {
      unlink(paste0(cache_prefix, ".*"), recursive = FALSE, force = TRUE)
    }
    if (force_rebuild || any(!file.exists(cache_files))) {
      cat("[INFO] Building rMVP cache from PLINK prefix:", bfile, "\n")
      rMVP::MVP.Data(
        fileBed = bfile,
        out = cache_prefix,
        maxLine = max_line,
        SNP.impute = "Major",
        verbose = TRUE,
        ncpus = threads
      )
    } else {
      cat("[INFO] Reusing rMVP cache:", cache_prefix, "\n")
    }

    geno <- bigmemory::attach.big.matrix(paste0(cache_prefix, ".geno.desc"))
    map <- utils::read.table(
      paste0(cache_prefix, ".geno.map"),
      header = TRUE,
      sep = "\t",
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    if (ncol(map) < 3L) {
      stop("rMVP map cache must contain at least SNP/CHROM/POS columns.", call. = FALSE)
    }
    map_out <- as.data.frame(map, stringsAsFactors = FALSE)
    marker_manifest_is_identity <- FALSE
    if (nzchar(marker_manifest_path)) {
      if (!file.exists(marker_manifest_path)) {
        stop(sprintf("Marker manifest not found: %s", marker_manifest_path), call. = FALSE)
      }
      marker_manifest <- utils::read.table(
        marker_manifest_path,
        header = TRUE,
        sep = "\t",
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
      if (!("row_idx0" %in% names(marker_manifest))) {
        stop("Marker manifest must contain a `row_idx0` column.", call. = FALSE)
      }
      marker_idx0 <- suppressWarnings(as.integer(marker_manifest[["row_idx0"]]))
      marker_idx0 <- marker_idx0[is.finite(marker_idx0)]
      if (!length(marker_idx0)) {
        stop("Marker manifest is empty after parsing `row_idx0`.", call. = FALSE)
      }
      cat("[INFO] Applying JanusX marker manifest:", marker_manifest_path, "kept=", length(marker_idx0), "\n")
      marker_manifest_is_identity <-
        length(marker_idx0) == nrow(map) &&
        all(marker_idx0 == (seq_len(nrow(map)) - 1L))
      if (marker_manifest_is_identity) {
        cat("[INFO] Marker manifest is identity on the current rMVP cache; using direct FarmCPU cache without marker subsetting.\n")
      } else {
        marker_subset <- subset_geno_for_markers(
          geno = geno,
          marker_idx0 = marker_idx0,
          n_markers = nrow(map_out),
          cache_prefix = cache_prefix,
          label = "farmcpu_markers"
        )
        geno <- marker_subset$geno
        map_out <- map_out[marker_subset$kept_idx0 + 1L, , drop = FALSE]
        rownames(map_out) <- NULL
      }
    }
    # rMVP::MVP.FarmCPU expects only SNP/CHROM/POS here; passing A1/A2 shifts the internal P column.
    map_farmcpu <- map_out[, seq_len(3L), drop = FALSE]
    if (!is.finite(threshold) || threshold <= 0) threshold <- 1 / max(1L, nrow(map_farmcpu))
    if (!is.finite(threshold_output) || threshold_output <= 0) threshold_output <- threshold
    if (!nzchar(p_threshold_raw) || tolower(p_threshold_raw) %in% c("na", "null", "none")) {
      p_threshold <- NA_real_
    } else if (!is.finite(p_threshold) || p_threshold <= 0) {
      p_threshold <- NA_real_
    }
    if (!nzchar(qtn_threshold_raw) || tolower(qtn_threshold_raw) %in% c("na", "null", "none")) {
      qtn_threshold <- NA_real_
    } else if (!is.finite(qtn_threshold) || qtn_threshold <= 0) {
      qtn_threshold <- NA_real_
    }
    ids <- utils::read.table(
      paste0(cache_prefix, ".geno.ind"),
      header = FALSE,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )[[1]]
    ids <- as.character(ids)

    phe_raw <- utils::read.table(
      pheno,
      header = TRUE,
      sep = "\t",
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    if (ncol(phe_raw) < 2L) {
      stop(sprintf("Invalid phenotype file: %s", pheno), call. = FALSE)
    }
    sample_ids <- as.character(phe_raw[[1]])
    trait_vals <- suppressWarnings(as.numeric(phe_raw[[2]]))
    hit <- match(ids, sample_ids)
    matched <- which(!is.na(hit) & is.finite(trait_vals[hit]))
    if (!length(matched)) {
      stop("No overlapping phenotype rows remain after aligning to the PLINK sample order.", call. = FALSE)
    }
    phe <- data.frame(
      Taxa = ids[matched],
      PHENO = trait_vals[hit[matched]],
      stringsAsFactors = FALSE
    )
    ind_idx <- as.integer(matched)
    farmcpu_fun <- rMVP::MVP.FarmCPU
    kin_fun <- rMVP::MVP.K.VanRaden
    farmcpu_supports_ind_idx <- function_has_formal(farmcpu_fun, "ind_idx")
    geno_farmcpu <- geno
    ind_idx_farmcpu <- ind_idx
    marker_by_col_farmcpu <- tryCatch(as.integer(ncol(geno)) == nrow(map_out), error = function(e) TRUE)
    if (!farmcpu_supports_ind_idx) {
      cat("[INFO] MVP.FarmCPU does not expose ind_idx; keeping original big.matrix and carrying explicit matched sample indices.\n")
      subset_res <- subset_geno_for_individuals(
        geno = geno,
        ind_idx = ind_idx,
        n_markers = nrow(map_out),
        cache_prefix = cache_prefix,
        label = "farmcpu"
      )
      geno_farmcpu <- subset_res$geno
      ind_idx_farmcpu <- subset_res$ind_idx
      marker_by_col_farmcpu <- isTRUE(subset_res$marker_by_col)
    }

    CV_farmcpu <- NULL
    marker_freq <- compute_marker_freq(
      geno_farmcpu,
      ind_idx_farmcpu,
      NULL,
      threads,
      max_line,
      nrow(map_out)
    )
    if (!is.null(npc_farmcpu) && npc_farmcpu > 0L) {
      kin_args <- list(
        M = geno_farmcpu,
        maxLine = max_line,
        ind_idx = ind_idx_farmcpu,
        mrk_idx = NULL,
        mrk_freq = marker_freq,
        mrk_bycol = marker_by_col_farmcpu,
        cpu = threads,
        verbose = TRUE,
        checkNA = FALSE
      )
      K <- compute_rmvp_kinship(kin_fun, kin_args, threads)
      eigenK <- eigen(K, symmetric = TRUE)
      npc_keep <- min(nrow(eigenK$vectors), max(3L, npc_farmcpu))
      ipca <- eigenK$vectors[, seq_len(npc_keep), drop = FALSE]
      CV_farmcpu <- ipca[, seq_len(min(ncol(ipca), npc_farmcpu)), drop = FALSE]
    }

    farmcpu_args <- list(
      phe = phe,
      geno = geno_farmcpu,
      map = map_farmcpu,
      CV = CV_farmcpu,
      ind_idx = ind_idx_farmcpu,
      mrk_idx = NULL,
      P = NULL,
      method.sub = "reward",
      method.sub.final = "reward",
      method.bin = method_bin,
      bin.size = bin_size,
      bin.selection = bin_selection,
      memo = "MVP.FarmCPU",
      Prior = NULL,
      ncpus = threads,
      maxLoop = max_loop,
      maxLine = max_line,
      threshold.output = threshold_output,
      converge = 1,
      iteration.output = FALSE,
      bound = farmcpu_bound,
      verbose = TRUE
    )
    if (is.finite(p_threshold) && p_threshold > 0) {
      farmcpu_args[["p.threshold"]] <- p_threshold
    }
    if (is.finite(qtn_threshold) && qtn_threshold > 0) {
      farmcpu_args[["QTN.threshold"]] <- qtn_threshold
    }
    if (nzchar(vc_method)) {
      cat("[INFO] direct MVP.FarmCPU path does not expose vc.method; requested value recorded only:", vc_method, "\n")
    }
    farmcpu_out <- do.call(farmcpu_fun, filter_named_args(farmcpu_fun, farmcpu_args, "MVP.FarmCPU"))
    farmcpu_out <- as.data.frame(farmcpu_out, stringsAsFactors = FALSE)
    if (!nrow(map_out) || !nrow(farmcpu_out)) {
      stop("rMVP returned no FarmCPU result rows.", call. = FALSE)
    }
    if (nrow(map_out) != nrow(farmcpu_out)) {
      stop("rMVP returned mismatched map/result row counts.", call. = FALSE)
    }

    beta <- suppressWarnings(as.numeric(farmcpu_out[[1]]))
    se <- suppressWarnings(as.numeric(farmcpu_out[[2]]))
    pval <- suppressWarnings(as.numeric(farmcpu_out[[ncol(farmcpu_out)]]))
    need_p <- !is.finite(pval) & is.finite(beta) & is.finite(se) & (se > 0)
    if (any(need_p)) {
      pval[need_p] <- stats::pchisq((beta[need_p] / se[need_p])^2, df = 1, lower.tail = FALSE)
    }
    chisq <- rep(NA_real_, length(beta))
    ok <- is.finite(beta) & is.finite(se) & (se > 0)
    chisq[ok] <- (beta[ok] / se[ok])^2
    af <- ifelse(marker_freq > 0.5, 1 - marker_freq, marker_freq)

    out_tab <- data.frame(
      chrom = if ("CHROM" %in% names(map_out)) map_out[["CHROM"]] else NA,
      pos = if ("POS" %in% names(map_out)) suppressWarnings(as.integer(map_out[["POS"]])) else NA_integer_,
      snp = if ("SNP" %in% names(map_out)) as.character(map_out[["SNP"]]) else as.character(seq_len(nrow(map_out))),
      allele0 = if ("A1" %in% names(map_out)) as.character(map_out[["A1"]]) else NA_character_,
      allele1 = if ("A2" %in% names(map_out)) as.character(map_out[["A2"]]) else NA_character_,
      af = af,
      miss = NA_real_,
      beta = beta,
      se = se,
      chisq = chisq,
      pwald = pval,
      stringsAsFactors = FALSE
    )
    utils::write.table(out_tab, file = out_file, sep = "\t", row.names = FALSE, quote = FALSE)
  },
  error = function(e) {
    status <<- 1L
    note <<- conditionMessage(e)
    message(note)
  }
)

while (sink.number() > 0L) sink()

pseudo_qtn <- NA_integer_
if (file.exists(inner_log)) {
  pseudo_qtn <- detect_qtn_from_lines(readLines(inner_log, warn = FALSE))
}
write_meta(meta_file, status, pseudo_qtn, inner_log, note, "rmvp_final_seqQTN_count")
quit(save = "no", status = status)
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
        "-global",
        "-maf",
        str(args.maf),
        "-geno",
        str(args.geno),
        "-t",
        str(args.thread),
        "-o",
        str(gwas_out),
        "-prefix",
        args.prefix,
        "--farmcpu-iter",
        str(int(args.farmcpu_iter)),
        "--farmcpu-threshold",
        f"{float(align_runtime.farmcpu_threshold):.17g}",
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
    if args.mmap_limit:
        cmd += ["-mmap-limit"]

    rc, elapsed, rss = _run_timed(cmd, log_file=log_file, time_file=time_file)
    status = "ok" if rc == 0 else "failed"
    result_file = ""
    topk_file = str(run_dir / f"{args.prefix}.janusx.top{args.topk}.tsv")
    qtn_file = ""
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
            qtn_guess = _guess_janusx_qtn_file_from_result_file(rf)
            if qtn_guess is not None:
                qtn_file = str(qtn_guess)
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
        qtn_file=qtn_file if qtn_file and Path(qtn_file).exists() else "",
        topk_count=int(topk_count),
        topk_snps=topk_snps,
        pseudo_qtn=pseudo_qtn,
        log_file=str(log_file),
        time_file=str(time_file),
        note=note,
        debug_file="",
    )


def _run_kernel_r(
    args: argparse.Namespace,
    *,
    kernel: str,
    bfile_prefix: str,
    marker_manifest: Optional[Path],
    pheno_for_r: Path,
    out_dir: Path,
    farm_bound: int,
    farm_bin_selection: list[int],
    farm_bin_size: list[float],
    align_runtime: AlignRuntime,
    rmvp_runner: Path,
) -> RunResult:
    if kernel != "rmvp":
        raise ValueError(f"Unsupported R kernel: {kernel}")

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
    rmvp_force_rebuild = "1" if (kernel == "rmvp" and (not bool(getattr(args, "rmvp_reuse_cache", False)))) else "0"

    r_bin = _resolve_rscript_path()
    if not r_bin:
        raise RuntimeError("Rscript not found in current env PATH.")

    cmd = [
        str(r_bin),
        str(rmvp_runner),
        "--bfile",
        str(bfile_prefix),
        "--pheno",
        str(pheno_for_r),
        "--out-file",
        str(result_file),
        "--workdir",
        str(workdir),
        "--inner-log",
        str(inner_log),
        "--meta-file",
        str(meta_file),
        "--cache-prefix",
        str(workdir / "rmvp_cache" / "bench_input"),
        "--threads",
        str(args.thread),
        "--max-line",
        str(max(1000, int(args.chunksize))),
        "--farmcpu-bin-size",
        ",".join(str(int(v)) if float(v).is_integer() else str(v) for v in farm_bin_size),
        "--farmcpu-bin-selection",
        ",".join(str(int(v)) for v in farm_bin_selection),
        "--farmcpu-max-loop",
        str(args.farmcpu_iter),
        "--farmcpu-threshold",
        f"{float(align_runtime.farmcpu_threshold):.17g}",
        "--farmcpu-threshold-output",
        f"{float(align_runtime.farmcpu_threshold):.17g}",
        "--farmcpu-bound",
        str(int(farm_bound)),
        "--farmcpu-maf",
        str(args.maf),
        "--rmvp-vc-method",
        str(align_runtime.rmvp_vc_method),
        "--rmvp-method-bin",
        str(align_runtime.rmvp_method_bin),
        "--npc-farmcpu",
        str(max(0, int(args.qcov))),
        "--force-rebuild",
        rmvp_force_rebuild,
    ]
    if marker_manifest is not None:
        cmd += ["--marker-manifest", str(marker_manifest)]
    rc, elapsed, rss = _run_timed(cmd, log_file=log_file, time_file=time_file)
    status = "ok" if rc == 0 else "failed"
    pseudo_qtn = None
    qtn_file = ""
    debug_file = ""
    note = ""
    pseudo_qtn_metric = ""

    meta = _read_meta_kv(meta_file)
    if "pseudo_qtn_metric" in meta:
        pseudo_qtn_metric = str(meta["pseudo_qtn_metric"]).strip()
    if "pseudo_qtn" in meta:
        try:
            if str(meta["pseudo_qtn"]).strip().upper() != "NA":
                pseudo_qtn = int(float(meta["pseudo_qtn"]))
        except Exception:
            pseudo_qtn = None
    if pseudo_qtn is None and inner_log.exists():
        pseudo_qtn_from_log = _parse_pseudo_from_text(inner_log.read_text(errors="ignore"))
        if pseudo_qtn_from_log is not None:
            pseudo_qtn = pseudo_qtn_from_log
            if not pseudo_qtn_metric:
                pseudo_qtn_metric = "log_fallback"

    topk_count = 0
    topk_snps = ""
    if rc == 0:
        if result_file.exists():
            try:
                seq_idx1 = _parse_rmvp_final_seqqtn_indices(inner_log.read_text(errors="ignore")) if inner_log.exists() else []
                qtn_file_path = result_file.with_suffix(".qtn.tsv")
                _write_qtn_sidecar_from_assoc_result(
                    result_file=result_file,
                    qtn_file=qtn_file_path,
                    source_rows_1based=seq_idx1,
                )
                if qtn_file_path.exists():
                    qtn_file = str(qtn_file_path)
                if bool(getattr(args, "rmvp_debug_seqqtn", False)) and inner_log.exists():
                    try:
                        seq_blocks = _parse_rmvp_seqqtn_blocks(inner_log.read_text(errors="ignore"))
                        debug_file_path = result_file.with_suffix(".seqqtn.debug.tsv")
                        _write_rmvp_seqqtn_debug_sidecar_from_assoc_result(
                            result_file=result_file,
                            debug_file=debug_file_path,
                            seq_blocks=seq_blocks,
                        )
                        if debug_file_path.exists():
                            debug_file = str(debug_file_path)
                    except Exception as e:
                        warn = f"rmvp seqQTN debug sidecar failed: {e}"
                        note = f"{note}; {warn}" if note else warn
                topk_count, topk_snps = _extract_topk(result_file, topk_file, args.topk)
            except Exception as e:
                status = "failed"
                note = f"{kernel} topk parse failed: {e}"
        else:
            status = "failed"
            note = f"{kernel} result file missing: {result_file}"
    else:
        note = f"{kernel} command failed."
    if status == "ok" and kernel == "rmvp":
        metric_note = pseudo_qtn_metric or "rmvp_final_seqQTN_count"
        note = f"{note}; pseudo_qtn_metric={metric_note}" if note else f"pseudo_qtn_metric={metric_note}"

    return RunResult(
        kernel=kernel,
        status=status,
        exit_code=rc,
        elapsed_sec=elapsed,
        peak_rss_kb=rss,
        result_file=str(result_file) if result_file.exists() else "",
        topk_file=str(topk_file) if topk_file.exists() else "",
        qtn_file=qtn_file if qtn_file and Path(qtn_file).exists() else "",
        topk_count=int(topk_count),
        topk_snps=topk_snps,
        pseudo_qtn=pseudo_qtn,
        log_file=str(log_file),
        time_file=str(time_file),
        note=note,
        debug_file=debug_file if debug_file and Path(debug_file).exists() else "",
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


def _normalize_chr_token_py(raw: Any) -> str:
    text = str(raw or "").strip()
    if len(text) >= 3 and text[:3].lower() == "chr":
        return text[3:]
    return text


def _ordered_snp_ids_from_assoc_df(df: pd.DataFrame) -> list[str]:
    cols = {str(c).strip().lstrip("#").lower(): c for c in df.columns}
    chr_col = _first_col(cols, _CHR_COL_KEYS)
    pos_col = _first_col(cols, _POS_COL_KEYS)
    snp_col = _first_col(cols, _SNP_COL_KEYS)

    if snp_col is not None:
        snp_series = df[snp_col].astype(str).str.strip()
    elif chr_col is not None and pos_col is not None:
        chrom = df[chr_col].astype(str).str.replace(r"^chr", "", regex=True, case=False).str.strip()
        pos = pd.to_numeric(df[pos_col], errors="coerce")
        snp_series = chrom.astype(str) + "_" + pos.round().astype("Int64").astype(str)
    else:
        raise ValueError("Neither SNP id nor (chr,pos) columns found.")

    out: list[str] = []
    seen: set[str] = set()
    for raw in snp_series.tolist():
        sid = str(raw).strip()
        if (not sid) or (sid.lower() == "nan") or (sid == "<NA>"):
            continue
        if sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def _read_snp_set_from_table(path: Path) -> set[str]:
    df = _read_table_guess(path)
    return set(_ordered_snp_ids_from_assoc_df(df))


def _ordered_assoc_sites_from_assoc_df(df: pd.DataFrame) -> list[AssocSite]:
    std = _standardize_assoc(df)
    out: list[AssocSite] = []
    for _, row in std.iterrows():
        snp = str(row.get("SNP", "") or "").strip()
        if not snp:
            continue
        chrom = _normalize_chr_token_py(row.get("CHR", ""))
        pos_val = pd.to_numeric(pd.Series([row.get("POS", pd.NA)]), errors="coerce").iloc[0]
        pos: Optional[int]
        if pd.isna(pos_val):
            pos = None
        else:
            try:
                pos = int(round(float(pos_val)))
            except Exception:
                pos = None
        out.append(AssocSite(snp=snp, chrom=chrom, pos=pos))
    return out


def _read_assoc_sites_from_table(path: Path) -> list[AssocSite]:
    return _ordered_assoc_sites_from_assoc_df(_read_table_guess(path))


def _selected_assoc_rows_to_df(
    result_file: Path,
    source_rows_0based: list[int],
) -> pd.DataFrame:
    header, selected_rows = _collect_assoc_rows_by_index(result_file, source_rows_0based)
    if not header:
        return pd.DataFrame()
    ordered_rows: list[list[str]] = []
    seen: set[int] = set()
    for idx0 in source_rows_0based:
        try:
            idx = int(idx0)
        except Exception:
            continue
        if idx < 0 or idx in seen:
            continue
        seen.add(idx)
        row = selected_rows.get(idx)
        if row is None:
            continue
        ordered_rows.append(row)
    if len(ordered_rows) == 0:
        return pd.DataFrame(columns=header)
    return pd.DataFrame(ordered_rows, columns=header)


def _guess_janusx_qtn_file_from_summary_row(row: pd.Series) -> Optional[Path]:
    result_file = str(row.get("result_file", "") or "").strip()
    if not result_file:
        return None
    rf = Path(result_file)
    cands: list[Path] = []
    if rf.suffix.lower() == ".tsv":
        cands.append(rf.with_suffix(".qtn"))
    cands.extend(sorted(rf.parent.glob("*.farmcpu.qtn")))
    seen: set[str] = set()
    for cand in cands:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand
    return None


_SEQQTN_BLOCK_STOP_RE = re.compile(
    r"(current loop|optimizing|scanning|number of covariates|farmcpu\.lm|genomic inflation)",
    re.IGNORECASE,
)
_CURRENT_LOOP_RE = re.compile(r"current loop:\s*(\d+)", re.IGNORECASE)


def _unique_positive_ints_preserve_order(values: Iterable[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw in values:
        try:
            val = int(raw)
        except Exception:
            continue
        if val <= 0 or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _parse_rmvp_seqqtn_blocks(text: str) -> list[dict[str, Any]]:
    lines = str(text).splitlines()
    blocks: list[dict[str, Any]] = []
    current_loop: Optional[int] = None
    for idx, line in enumerate(lines):
        m_loop = _CURRENT_LOOP_RE.search(line)
        if m_loop is not None:
            try:
                current_loop = int(m_loop.group(1))
            except Exception:
                current_loop = None
        if not re.search(r"\bseqQTN\b", line, re.IGNORECASE):
            continue
        vals: list[int] = []
        saw_payload = False
        j = idx + 1
        while j < len(lines):
            lns = lines[j].strip()
            if not lns:
                break
            if _SEQQTN_BLOCK_STOP_RE.search(lns):
                break
            saw_payload = True
            if re.fullmatch(r"NULL", lns, re.IGNORECASE):
                vals = []
                break
            nums = [int(x) for x in re.findall(r"\b\d+\b", lns)]
            if re.match(r"^\[\d+\]\s*", lns) and nums:
                nums = nums[1:]
            vals.extend(nums)
            j += 1
        if not saw_payload:
            continue
        blocks.append(
            {
                "block_idx": len(blocks) + 1,
                "current_loop": current_loop,
                "source_rows_1based": _unique_positive_ints_preserve_order(vals),
            }
        )
    return blocks


def _parse_rmvp_final_seqqtn_indices(text: str) -> list[int]:
    blocks = _parse_rmvp_seqqtn_blocks(text)
    if len(blocks) == 0:
        return []
    last = blocks[-1].get("source_rows_1based", [])
    return [int(v) for v in last if int(v) > 0]


def _rmvp_pseudo_qtn_sites_from_summary_row(row: pd.Series) -> list[AssocSite]:
    qtn_file = str(row.get("qtn_file", "") or "").strip()
    if qtn_file:
        qtn_path = Path(qtn_file)
        if qtn_path.exists():
            try:
                return _read_assoc_sites_from_table(qtn_path)
            except Exception:
                pass
    result_file = str(row.get("result_file", "") or "").strip()
    log_file = str(row.get("log_file", "") or "").strip()
    if not result_file or not log_file:
        return []
    rf = Path(result_file)
    lf = Path(log_file)
    if not rf.exists() or not lf.exists():
        return []
    seq_idx1 = _parse_rmvp_final_seqqtn_indices(lf.read_text(errors="ignore"))
    if len(seq_idx1) == 0:
        return []
    df = _selected_assoc_rows_to_df(rf, [int(idx1) - 1 for idx1 in seq_idx1 if int(idx1) > 0])
    if df.empty:
        return []
    return _ordered_assoc_sites_from_assoc_df(df)


def _pseudo_qtn_sites_from_summary_row(row: pd.Series) -> list[AssocSite]:
    kernel = str(row.get("kernel", "") or "").strip().lower()
    if kernel == "rmvp":
        return _rmvp_pseudo_qtn_sites_from_summary_row(row)
    if kernel == "janusx":
        qtn_file = _guess_janusx_qtn_file_from_summary_row(row)
        if qtn_file is None or (not qtn_file.exists()):
            return []
        return _read_assoc_sites_from_table(qtn_file)
    return []


def _site_snp_set(sites: list[AssocSite]) -> set[str]:
    return {site.snp for site in sites if site.snp}


def _site_has_coords(site: AssocSite) -> bool:
    return bool(site.chrom) and (site.pos is not None) and (int(site.pos) >= 0)


def _maximum_bipartite_match_size(adj: list[list[int]], right_n: int) -> int:
    match_r = [-1] * max(0, int(right_n))

    def _dfs(left_idx: int, seen: list[bool]) -> bool:
        for right_idx in adj[left_idx]:
            if right_idx < 0 or right_idx >= right_n or seen[right_idx]:
                continue
            seen[right_idx] = True
            if match_r[right_idx] == -1 or _dfs(match_r[right_idx], seen):
                match_r[right_idx] = left_idx
                return True
        return False

    matched = 0
    for left_idx in range(len(adj)):
        seen = [False] * right_n
        if _dfs(left_idx, seen):
            matched += 1
    return int(matched)


def _ld_match_tp_from_sites(
    pred_sites: list[AssocSite],
    ref_sites: list[AssocSite],
    *,
    bfile_prefix: str,
    r2_threshold: float,
    threads: int,
) -> int:
    if len(pred_sites) == 0 or len(ref_sites) == 0:
        return 0

    adj_sets: list[set[int]] = [set() for _ in pred_sites]
    ref_by_snp: dict[str, list[int]] = {}
    for ref_idx, ref_site in enumerate(ref_sites):
        if ref_site.snp:
            ref_by_snp.setdefault(ref_site.snp, []).append(ref_idx)
    for pred_idx, pred_site in enumerate(pred_sites):
        for ref_idx in ref_by_snp.get(pred_site.snp, []):
            adj_sets[pred_idx].add(ref_idx)

    if _bed_ldblock_r2_rust is None:
        return _maximum_bipartite_match_size([sorted(v) for v in adj_sets], len(ref_sites))

    coord_keys: list[tuple[str, int]] = []
    seen_keys: set[tuple[str, int]] = set()
    for site in [*pred_sites, *ref_sites]:
        if not _site_has_coords(site):
            continue
        key = (str(site.chrom), int(site.pos))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        coord_keys.append(key)

    if len(coord_keys) > 0:
        chrom_ranges = [chrom for chrom, _pos in coord_keys]
        start_bp = [int(pos) for _chrom, pos in coord_keys]
        end_bp = [int(pos) for _chrom, pos in coord_keys]
        selected_chrom = [chrom for chrom, _pos in coord_keys]
        selected_pos = [int(pos) for _chrom, pos in coord_keys]
        ld_raw, chr_raw, pos_raw = _bed_ldblock_r2_rust(
            str(bfile_prefix),
            chrom_ranges,
            start_bp,
            end_bp,
            selected_chrom=selected_chrom,
            selected_pos=selected_pos,
            threads=int(max(0, int(threads))),
        )
        ld_mat = np.ascontiguousarray(np.asarray(ld_raw, dtype=np.float32))
        ld_index: dict[tuple[str, int], int] = {}
        for idx, (chrom, pos) in enumerate(zip(list(chr_raw), list(pos_raw))):
            ld_index[(_normalize_chr_token_py(chrom), int(pos))] = idx

        for pred_idx, pred_site in enumerate(pred_sites):
            if not _site_has_coords(pred_site):
                continue
            pred_key = (str(pred_site.chrom), int(pred_site.pos))
            pred_mat_idx = ld_index.get(pred_key)
            if pred_mat_idx is None:
                continue
            for ref_idx, ref_site in enumerate(ref_sites):
                if not _site_has_coords(ref_site):
                    continue
                if str(pred_site.chrom) != str(ref_site.chrom):
                    continue
                ref_key = (str(ref_site.chrom), int(ref_site.pos))
                ref_mat_idx = ld_index.get(ref_key)
                if ref_mat_idx is None:
                    continue
                if float(ld_mat[pred_mat_idx, ref_mat_idx]) >= float(r2_threshold):
                    adj_sets[pred_idx].add(ref_idx)

    return _maximum_bipartite_match_size([sorted(v) for v in adj_sets], len(ref_sites))


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


def _precision_recall_vs_reference(pred: set[str], ref: set[str]) -> tuple[int, Optional[float], Optional[float]]:
    tp = int(len(pred & ref))
    precision = None if len(pred) == 0 else float(tp / float(len(pred)))
    recall = None if len(ref) == 0 else float(tp / float(len(ref)))
    return tp, precision, recall


def _precision_recall_vs_reference_sites(
    pred_sites: list[AssocSite],
    ref_sites: list[AssocSite],
    *,
    match_mode: str,
    ld_bfile_prefix: str,
    ld_r2_threshold: float,
    threads: int,
) -> tuple[int, Optional[float], Optional[float]]:
    pred_n = int(len(pred_sites))
    ref_n = int(len(ref_sites))
    if str(match_mode).lower() == "ld":
        tp = _ld_match_tp_from_sites(
            pred_sites,
            ref_sites,
            bfile_prefix=ld_bfile_prefix,
            r2_threshold=float(ld_r2_threshold),
            threads=int(max(0, int(threads))),
        )
    else:
        tp = int(len(_site_snp_set(pred_sites) & _site_snp_set(ref_sites)))
    precision = None if pred_n == 0 else float(tp / float(pred_n))
    recall = None if ref_n == 0 else float(tp / float(ref_n))
    return int(tp), precision, recall


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
                "qtn_file": r.qtn_file,
                "log_file": r.log_file,
                "time_file": r.time_file,
                "note": r.note,
                "debug_file": r.debug_file,
            }
        )
    df = pd.DataFrame(recs)
    row_by_kernel: dict[str, pd.Series] = {}
    for _, r in df.iterrows():
        row_by_kernel[str(r["kernel"])] = r

    pseudo_qtn_sites: dict[str, list[AssocSite]] = {}
    for kernel, row in row_by_kernel.items():
        try:
            pseudo_qtn_sites[kernel] = _pseudo_qtn_sites_from_summary_row(row)
        except Exception:
            pseudo_qtn_sites[kernel] = []

    ref_kernel = "rmvp" if "rmvp" in row_by_kernel else ""
    match_mode_requested = str(extra_cfg.get("pseudo_qtn_match_mode", "exact") or "exact").strip().lower()
    if match_mode_requested not in {"exact", "ld"}:
        match_mode_requested = "exact"
    match_mode_effective = match_mode_requested
    match_note = ""
    ld_bfile_prefix = str(extra_cfg.get("pseudo_qtn_ld_bfile", "") or "").strip()
    ld_r2_threshold = _safe_float(extra_cfg.get("pseudo_qtn_ld_r2", 0.7), 0.7)
    ld_threads = int(extra_cfg.get("threads", 0) or 0)
    if ref_kernel and match_mode_effective == "ld":
        if _bed_ldblock_r2_rust is None:
            match_mode_effective = "exact"
            match_note = "requested LD match but `bed_ldblock_r2_rust` is unavailable; fell back to exact SNP matching."
        elif not ld_bfile_prefix or (not Path(f"{ld_bfile_prefix}.bed").exists()):
            match_mode_effective = "exact"
            match_note = "requested LD match but PLINK BED prefix for LD calculation is unavailable; fell back to exact SNP matching."
    extra_cfg["pseudo_qtn_match_mode_requested"] = match_mode_requested
    extra_cfg["pseudo_qtn_match_mode_effective"] = match_mode_effective
    if match_note:
        extra_cfg["pseudo_qtn_match_note"] = match_note

    ref_qtn_sites = pseudo_qtn_sites.get(ref_kernel, []) if ref_kernel else []
    df["pseudo_qtn_set_size"] = [int(len(pseudo_qtn_sites.get(str(k), []))) for k in df["kernel"].tolist()]
    df["pseudo_qtn_match_mode_vs_rmvp"] = (match_mode_effective if ref_kernel else pd.NA)
    df["pseudo_qtn_tp_vs_rmvp"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    df["pseudo_qtn_precision_vs_rmvp"] = math.nan
    df["pseudo_qtn_recall_vs_rmvp"] = math.nan
    if ref_kernel:
        tp_vals: list[Any] = []
        prec_vals: list[float] = []
        rec_vals: list[float] = []
        for kernel in df["kernel"].tolist():
            pred_sites = pseudo_qtn_sites.get(str(kernel), [])
            tp, prec, rec = _precision_recall_vs_reference_sites(
                pred_sites,
                ref_qtn_sites,
                match_mode=match_mode_effective,
                ld_bfile_prefix=ld_bfile_prefix,
                ld_r2_threshold=ld_r2_threshold,
                threads=ld_threads,
            )
            tp_vals.append(int(tp))
            prec_vals.append(math.nan if prec is None else float(prec))
            rec_vals.append(math.nan if rec is None else float(rec))
        df["pseudo_qtn_tp_vs_rmvp"] = pd.Series(tp_vals, dtype="Int64")
        df["pseudo_qtn_precision_vs_rmvp"] = prec_vals
        df["pseudo_qtn_recall_vs_rmvp"] = rec_vals

    tsv = out_dir / f"{prefix}.{_SUMMARY_TAG}.tsv"
    md = out_dir / f"{prefix}.{_SUMMARY_TAG}.md"
    cfg = out_dir / f"{prefix}.{_SUMMARY_TAG}.config.json"
    df.to_csv(tsv, sep="\t", index=False)

    lines = [
        "| Kernel | Status | Time(s) | Peak RSS(GB) | pseudoQTN | QTN Set | Match | Prec(rMVP) | Rec(rMVP) | topK |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        t = "NA" if pd.isna(r["elapsed_sec"]) else f"{float(r['elapsed_sec']):.2f}"
        m = "NA" if pd.isna(r["peak_rss_gb"]) else f"{float(r['peak_rss_gb']):.3f}"
        q = "NA" if pd.isna(r["pseudo_qtn"]) else str(int(r["pseudo_qtn"]))
        qset = "NA" if pd.isna(r["pseudo_qtn_set_size"]) else str(int(r["pseudo_qtn_set_size"]))
        match_txt = "NA" if pd.isna(r["pseudo_qtn_match_mode_vs_rmvp"]) else str(r["pseudo_qtn_match_mode_vs_rmvp"])
        prec = "NA" if pd.isna(r["pseudo_qtn_precision_vs_rmvp"]) else f"{float(r['pseudo_qtn_precision_vs_rmvp']):.3f}"
        rec = "NA" if pd.isna(r["pseudo_qtn_recall_vs_rmvp"]) else f"{float(r['pseudo_qtn_recall_vs_rmvp']):.3f}"
        lines.append(
            f"| {r['kernel']} | {r['status']} | {t} | {m} | {q} | {qset} | {match_txt} | {prec} | {rec} | {int(r['topk_count'])} |"
        )

    preferred = ["janusx", "rmvp"]
    seen = {str(x) for x in df["kernel"].tolist()}
    kernel_order = [k for k in preferred if k in seen]
    extras = [str(x) for x in df["kernel"].tolist() if str(x) not in preferred]
    kernel_order.extend(extras)

    lines.extend(
        [
            "",
            "**PseudoQTN Count Consistency Matrix (Lower Triangle)**",
            "Formula: `min(qtn_i, qtn_j) / max(qtn_i, qtn_j)`; `NA` means missing/non-positive pseudoQTN.",
            "Current direct/aligned benchmark uses explicit `rMVP::MVP.FarmCPU(...)` with `method.bin=FaST-LMM`.",
            "The installed direct `MVP.FarmCPU` entry point does not expose `vc.method`; config therefore records `rmvp_vc_method_requested` only as requested metadata, not as an effective runtime knob.",
            "The embedded rMVP runner trims the FarmCPU `map` input to `SNP/CHROM/POS`; full map columns are kept only for output annotation.",
            "`rmvp` pseudoQTN is the final `seqQTN` / covariate count parsed from the last loop; `janusx` pseudoQTN is the workflow-reported JanusX count.",
            "`qtn_file` is a post-run sidecar and is not included in the timed rMVP execution.",
            "`debug_file` (when enabled via `--rmvp-debug-seqqtn`) is also a post-run sidecar and is not included in the timed rMVP execution.",
        ]
    )
    if match_note:
        lines.append(match_note)
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

    if ref_kernel:
        match_title = "Exact SNP Match"
        match_formula = "Formula: `precision=|A∩R|/|A|`, `recall=|A∩R|/|R|`, where `R` is the final rMVP pseudoQTN SNP set."
        match_detail = "Matching is exact by SNP ID; same-locus but different representative SNPs count as mismatch."
        if match_mode_effective == "ld":
            match_title = f"LD Match (one-to-one, r2 >= {ld_r2_threshold:.3f})"
            match_formula = (
                "Formula: `precision=TP/|A|`, `recall=TP/|R|`, where `TP` is the maximum one-to-one bipartite match count "
                "between predicted and reference pseudoQTN SNPs."
            )
            match_detail = (
                "An edge is created by exact SNP-ID match or same-chrom LD with `r2 >= "
                f"{ld_r2_threshold:.3f}`; same-locus different representative SNPs can therefore match."
            )
        lines.extend(
            [
                "",
                f"**PseudoQTN Precision/Recall vs {ref_kernel} ({match_title})**",
                match_formula,
                match_detail,
                "| kernel | match | pred_qtn | ref_qtn | precision | recall |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        ref_n = int(len(ref_qtn_sites))
        for kernel in kernel_order:
            pred_sites = pseudo_qtn_sites.get(kernel, [])
            tp, prec, rec = _precision_recall_vs_reference_sites(
                pred_sites,
                ref_qtn_sites,
                match_mode=match_mode_effective,
                ld_bfile_prefix=ld_bfile_prefix,
                ld_r2_threshold=ld_r2_threshold,
                threads=ld_threads,
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        kernel,
                        str(int(tp)),
                        str(int(len(pred_sites))),
                        str(ref_n),
                        ("NA" if prec is None else f"{prec:.3f}"),
                        ("NA" if rec is None else f"{rec:.3f}"),
                    ]
                )
                + " |"
            )

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
        description="Benchmark FarmCPU kernels: JanusX -farmcpu -global / explicit rMVP::MVP.FarmCPU.",
        epilog=minimal_help_epilog(
            [
                "jx benchmark -bfile example_prefix -p pheno.tsv -n 0",
                "jx benchmark -vcf example.vcf.gz -p pheno.tsv -n 0 --kernels janusx,rmvp",
                "jx benchmark -h -dev",
            ]
        ),
    )
    p.set_defaults(snps_only=False)
    p.add_argument("-dev", "--dev", action="store_true", default=False, help=argparse.SUPPRESS)

    required_group = p.add_argument_group("Required arguments")
    g = required_group.add_mutually_exclusive_group(required=False)
    add_common_genotype_source_args(g, include_file=True)
    add_common_pheno_arg(required_group, required=False, help_text="Phenotype file (first col sample ID).")

    optional_group = p.add_argument_group("Optional arguments")
    add_common_trait_selector_args(
        optional_group,
        dest="ncol",
        help_text=(
            "Phenotype column selector, accepted as zero-based index (excluding sample ID), "
            "column name, comma list, or numeric range. "
            "Benchmark currently supports one selected trait."
        ),
    )
    add_common_out_arg(optional_group, default=".", help_profile="simple")
    add_common_prefix_arg(optional_group, default="benchmark", help_profile="simple")
    add_common_variant_filter_args(
        optional_group,
        help_profile="short",
        include_maf=True,
        include_geno=True,
        include_het=False,
        maf_default=0.02,
        geno_default=0.05,
    )
    optional_group.add_argument(
        "-chunksize",
        "--chunksize",
        default=10_000,
        type=int,
        help="Auxiliary chunk size for preprocessing / rMVP cache build (default: %(default)s).",
    )
    add_common_thread_arg(optional_group, default_threads=detect_effective_threads(), help_profile="short")
    optional_group.add_argument("-q", "--qcov", default=0, type=int, help="JanusX qcov (PC count).")
    add_common_covariate_file_or_site_arg(
        optional_group,
        dest="cov",
        default=None,
        help_profile="benchmark",
    )
    optional_group.add_argument("-mmap-limit", "--mmap-limit", action="store_true", default=False, help="Enable JanusX mmap-limit.")
    optional_group.add_argument("--kernels", default="janusx,rmvp", type=str, help="Comma list: janusx,rmvp.")
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
        default=30,
        type=int,
        help=("FarmCPU max loop / iteration." if show_dev_help else argparse.SUPPRESS),
    )
    advanced_group.add_argument(
        "--farmcpu-threshold",
        default=None,
        type=float,
        help=("FarmCPU threshold; if unset, defaults to 1 / tested_SNP_count." if show_dev_help else argparse.SUPPRESS),
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
            "Reuse existing rMVP preprocessing cache under the benchmark workdir (default: rebuild each run for consistency)."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    advanced_group.add_argument(
        "--rmvp-debug-seqqtn",
        action="store_true",
        default=False,
        help=(
            "Write a post-run rMVP seqQTN debug sidecar with block/loop/source_row/chr/pos/snp mapping."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    advanced_group.add_argument(
        "--pseudo-qtn-match",
        default="exact",
        choices=["exact", "ld"],
        help=(
            "PseudoQTN precision/recall matching mode: `exact` by SNP ID, or `ld` for looser one-to-one LD matching."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    advanced_group.add_argument(
        "--pseudo-qtn-ld-r2",
        default=0.7,
        type=float,
        help=(
            "When `--pseudo-qtn-match ld` is used, count same-chrom SNP pairs with r2 >= this threshold as match edges."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )

    if any(
        (tk == "-trait") or tk.startswith("-trait=") or (tk == "--trait") or tk.startswith("--trait=")
        for tk in tokens
    ):
        p.error("`-trait/--trait` has been replaced by `-n/--n` (phenotype selector: zero-based index or column name, excluding sample ID).")

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
        args.ncol = parse_trait_selector_specs(args.ncol, label="-n/--n")
    except ValueError as e:
        p.error(str(e))
    if args.ncol is not None and len(args.ncol) > 1:
        p.error("`jx benchmark` currently supports one trait at a time; please pass a single `-n/--n` selector.")

    if int(args.farmcpu_iter) < 1:
        p.error("--farmcpu-iter must be >= 1.")
    if int(args.farmcpu_nbin) < 1:
        p.error("--farmcpu-nbin must be >= 1.")
    if args.farmcpu_threshold is not None and float(args.farmcpu_threshold) <= 0:
        p.error("--farmcpu-threshold must be > 0.")
    if args.force_pseudo_qtn_cap is not None and int(args.force_pseudo_qtn_cap) < 1:
        p.error("--force-pseudo-qtn-cap must be >= 1.")
    if not (0.0 <= float(args.pseudo_qtn_ld_r2) <= 1.0):
        p.error("--pseudo-qtn-ld-r2 must be in [0, 1].")
    return args


def main() -> None:
    args = parse_args()

    kernels = _parse_kernels(args.kernels)
    if len(kernels) == 0:
        raise ValueError("No valid kernels selected. Use --kernels janusx,rmvp")

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
    marker_manifest_path: Optional[Path] = None

    needs_r = "rmvp" in kernels
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
    if bfile_for_r:
        n_snps_for_threshold = max(1, int(_count_bim_snps(bfile_for_r)))
    elif args.bfile:
        n_snps_for_threshold = max(1, int(_count_bim_snps(gfile)))
    else:
        _sample_ids_probe, n_sites_probe = inspect_genotype_file(gfile)
        n_snps_for_threshold = max(1, int(n_sites_probe))
    n_snps_for_threshold_source = f"pre_filter({int(n_snps_for_threshold)})"

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

    if "rmvp" in kernels:
        bfile_for_manifest = bfile_for_r or (str(gfile) if bool(args.bfile) else "")
        if not bfile_for_manifest:
            raise RuntimeError("Benchmark marker manifest requires a PLINK BED prefix for rMVP alignment.")
        active_rows, _manifest_n_samples = _build_benchmark_marker_manifest(
            bfile_prefix=str(bfile_for_manifest),
            maf=float(args.maf),
            geno=float(args.geno),
            snps_only=bool(args.snps_only),
            expected_n_samples=(len(fam_ids) if len(fam_ids) > 0 else None),
        )
        marker_manifest_path = _write_benchmark_marker_manifest(
            out_path=bench_dir / "input" / "janusx_marker_manifest.tsv",
            active_rows=active_rows,
        )
        n_snps_for_threshold = max(1, int(len(active_rows)))
        n_snps_for_threshold_source = f"post_filter_manifest({int(n_snps_for_threshold)})"

    align_runtime = _resolve_align_runtime(args)
    align_runtime.farmcpu_threshold = _resolve_effective_farmcpu_threshold(
        getattr(args, "farmcpu_threshold", None),
        n_snps_for_threshold,
    )

    rmvp_runner = bench_dir / "tmp" / "rmvp_farmcpu_runner.R"
    rmvp_runner.parent.mkdir(parents=True, exist_ok=True)
    _build_rmvp_runner_script(rmvp_runner)

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
        if kernel == "rmvp":
            rr = _run_kernel_r(
                args,
                kernel=kernel,
                bfile_prefix=bfile_for_r,
                marker_manifest=marker_manifest_path,
                pheno_for_r=ph_for_r,
                out_dir=bench_dir / "runs",
                farm_bound=farm_bound,
                farm_bin_selection=farm_bin_selection,
                farm_bin_size=farm_bin_size,
                align_runtime=align_runtime,
                rmvp_runner=rmvp_runner,
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
        "farmcpu_threshold": float(align_runtime.farmcpu_threshold),
        "farmcpu_threshold_source": (
            "cli" if args.farmcpu_threshold is not None else f"auto_1_over_nsnp:{n_snps_for_threshold_source}"
        ),
        "benchmark_mode": "direct_aligned_explicit_mvp_farmcpu",
        "janusx_gwas_row_stat_mode": "global",
        "rmvp_runner_mode": "direct_mvp_farmcpu",
        "rmvp_vc_method_requested": str(align_runtime.rmvp_vc_method),
        "rmvp_vc_method_effective": None,
        "rmvp_vc_method_note": "not_exposed_by_direct_mvp_farmcpu_entrypoint",
        "rmvp_reuse_cache": bool(args.rmvp_reuse_cache),
        "rmvp_force_rebuild": (not bool(args.rmvp_reuse_cache)),
        "rmvp_debug_seqqtn": bool(args.rmvp_debug_seqqtn),
        "rmvp_method_bin": str(align_runtime.rmvp_method_bin),
        "force_pseudo_qtn_cap": (None if args.force_pseudo_qtn_cap is None else int(args.force_pseudo_qtn_cap)),
        "pseudo_qtn_match_mode": str(args.pseudo_qtn_match),
        "pseudo_qtn_ld_r2": float(args.pseudo_qtn_ld_r2),
        "pseudo_qtn_ld_bfile": (str(bfile_for_r) if bfile_for_r else (str(gfile) if bool(args.bfile) else "")),
        "conversion_elapsed_sec": float(conv_elapsed),
        "conversion_peak_rss_kb": float(conv_rss),
        "benchmark_dir": str(bench_dir),
        "rmvp_marker_manifest": (str(marker_manifest_path) if marker_manifest_path is not None else None),
    }
    _save_summary(bench_dir / "summary", args.prefix, rows, cfg)

    if not args.keep_temp:
        try:
            if rmvp_runner.exists():
                rmvp_runner.unlink()
        except Exception:
            pass

    summary_tsv = bench_dir / "summary" / f"{args.prefix}.{_SUMMARY_TAG}.tsv"
    summary_md = bench_dir / "summary" / f"{args.prefix}.{_SUMMARY_TAG}.md"
    print(f"[DONE] summary table: {summary_tsv}")
    print(f"[DONE] summary markdown: {summary_md}")


if __name__ == "__main__":
    main()
