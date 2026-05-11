#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark JanusX full-genotype IO performance on PLINK BED inputs.

Scope
-----
This script compares exactly four methods on full PLINK genotype files:
  - plink      : PLINK full-file filter+write baseline (--make-bed)
  - stream_bed : Rust stream decode + filter + write PLINK
  - packed_bed : Rust packed 2-bit filter + write PLINK
  - memmap_bed : Rust mmap scan + filter + write PLINK

Output
------
- `benchmark_results.json`
- `benchmark_results.csv`
- `benchmark_summary.csv`

Each row contains:
  - wall time
  - peak RSS + peak USS (process tree)
  - effective sample/SNP counts (when available)
  - command and parseable details
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


def _bootstrap_repo_python_path() -> None:
    if __package__:
        return
    here = Path(__file__).resolve()
    repo = here.parents[1]
    py_root = repo / "python"
    if py_root.exists() and str(py_root) not in sys.path:
        sys.path.insert(0, str(py_root))


_bootstrap_repo_python_path()

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


def _safe_float(v: Any, default: float = math.nan) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    return x


def _fmt_secs(x: float) -> str:
    if not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.3f}"


def _fmt_gb(x: Optional[float]) -> str:
    if x is None or (not math.isfinite(float(x))):
        return "nan"
    return f"{float(x):.3f}"


def _normalize_plink_prefix(path_or_prefix: str) -> str:
    s = str(path_or_prefix).strip()
    low = s.lower()
    if low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam"):
        return s[:-4]
    return s


def _count_fam(prefix: str) -> int:
    fam = Path(f"{prefix}.fam")
    n = 0
    with fam.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s and (not s.startswith("#")):
                n += 1
    return int(n)


def _count_bim(prefix: str) -> int:
    bim = Path(f"{prefix}.bim")
    n = 0
    with bim.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s and (not s.startswith("#")):
                n += 1
    return int(n)


def _read_fam_iids(prefix: str) -> list[str]:
    fam = Path(f"{prefix}.fam")
    out: list[str] = []
    with fam.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()
            if len(tok) >= 2:
                iid = str(tok[1]).strip()
                if iid:
                    out.append(iid)
    if len(out) == 0:
        raise ValueError(f"No samples found in {fam}")
    return out


def _read_fam_fid_iid(prefix: str) -> list[tuple[str, str]]:
    fam = Path(f"{prefix}.fam")
    out: list[tuple[str, str]] = []
    with fam.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()
            if len(tok) >= 2:
                out.append((str(tok[0]).strip(), str(tok[1]).strip()))
    if len(out) == 0:
        raise ValueError(f"No samples found in {fam}")
    return out


def _read_bim_chrom_pos(prefix: str) -> list[tuple[str, int]]:
    bim = Path(f"{prefix}.bim")
    out: list[tuple[str, int]] = []
    with bim.open("r", encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()
            if len(tok) < 4:
                raise ValueError(f"Malformed BIM line at {bim}:{ln}")
            chrom = str(tok[0]).strip()
            try:
                pos = int(float(tok[3]))
            except Exception as ex:
                raise ValueError(f"Invalid BIM position at {bim}:{ln}") from ex
            out.append((chrom, pos))
    if len(out) == 0:
        raise ValueError(f"No variants found in {bim}")
    return out


def _select_evenly_spaced_indices(n_total: int, n_keep: int) -> list[int]:
    if n_total <= 0:
        return []
    n_keep = max(1, min(int(n_keep), int(n_total)))
    if n_keep >= n_total:
        return list(range(n_total))
    chosen: list[int] = []
    seen: set[int] = set()
    for i in range(n_keep):
        idx = int(((i + 0.5) * n_total) / n_keep)
        if idx >= n_total:
            idx = n_total - 1
        if idx not in seen:
            seen.add(idx)
            chosen.append(idx)
    if len(chosen) < n_keep:
        for idx in range(n_total):
            if idx not in seen:
                chosen.append(idx)
                seen.add(idx)
                if len(chosen) >= n_keep:
                    break
    chosen.sort()
    return chosen


def _build_keep_ids(all_ids: list[str], keep_fraction: float) -> list[str]:
    frac = float(keep_fraction)
    frac = min(1.0, max(0.001, frac))
    target = max(1, int(round(len(all_ids) * frac)))
    idx = _select_evenly_spaced_indices(len(all_ids), target)
    return [all_ids[i] for i in idx]


def _choose_range(rows: list[tuple[str, int]], snp_fraction: float) -> tuple[str, int, int]:
    by_chr: dict[str, list[int]] = {}
    for chrom, pos in rows:
        by_chr.setdefault(str(chrom), []).append(int(pos))
    if len(by_chr) == 0:
        raise ValueError("No chromosome positions found in BIM")

    # Prefer chromosome with most variants for stable slicing.
    chrom = max(by_chr.keys(), key=lambda c: len(by_chr[c]))
    pos = sorted(by_chr[chrom])
    m = len(pos)
    frac = min(1.0, max(0.001, float(snp_fraction)))
    take = max(1, int(round(m * frac)))
    if take >= m:
        return chrom, int(pos[0]), int(pos[-1])
    start_i = max(0, (m - take) // 2)
    end_i = min(m - 1, start_i + take - 1)
    a = int(pos[start_i])
    b = int(pos[end_i])
    if a > b:
        a, b = b, a
    return chrom, a, b


def _write_keep_file(path: Path, keep_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fw:
        for sid in keep_ids:
            fw.write(f"{sid}\n")


def _write_keep_file_plink(path: Path, fam_pairs: list[tuple[str, str]], keep_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fw:
        for fid, iid in fam_pairs:
            if iid in keep_ids:
                fw.write(f"{fid}\t{iid}\n")


def _write_range_file(path: Path, chrom: str, start: int, end: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fw:
        fw.write(f"{chrom}\t{int(start)}\t{int(end)}\n")


def _read_keep_ids_file(path: Path) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s and (not s.startswith("#")):
                out.append(str(s.split()[0]))
    return out


def _ensure_mmap_slice_npy(
    *,
    prefix: str,
    keep_file: Path,
    range_chr: str,
    range_start: int,
    range_end: int,
    maf: float,
    geno: float,
    chunk_size: int,
    out_npy: Path,
) -> tuple[int, int]:
    import numpy as np
    from janusx.gfreader import load_genotype_chunks

    if out_npy.exists():
        try:
            arr = np.load(str(out_npy), mmap_mode="r")
            if getattr(arr, "ndim", 0) == 2:
                return int(arr.shape[0]), int(arr.shape[1])
        except Exception:
            pass

    keep_ids = _read_keep_ids_file(keep_file)
    if len(keep_ids) == 0:
        raise RuntimeError(f"keep file is empty: {keep_file}")

    ranges = [(str(range_chr), int(range_start), int(range_end))]
    rows = 0
    cols = 0
    for geno_chunk, _ in load_genotype_chunks(
        str(prefix),
        chunk_size=int(max(1, chunk_size)),
        maf=float(maf),
        missing_rate=float(geno),
        impute=False,
        sample_ids=keep_ids,
        ranges=ranges,
    ):
        arr = np.asarray(geno_chunk, dtype=np.float32)
        if arr.ndim != 2 or int(arr.shape[0]) <= 0:
            continue
        rows += int(arr.shape[0])
        cols = int(arr.shape[1])

    if rows <= 0 or cols <= 0:
        raise RuntimeError(
            "Failed to build mmap NPY cache: empty matrix after keep/range/filter."
        )

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    tmp_npy = out_npy.with_suffix(out_npy.suffix + ".tmp")
    if tmp_npy.exists():
        tmp_npy.unlink()

    mm = np.lib.format.open_memmap(
        str(tmp_npy),
        mode="w+",
        dtype=np.float32,
        shape=(int(rows), int(cols)),
    )
    wpos = 0
    try:
        for geno_chunk, _ in load_genotype_chunks(
            str(prefix),
            chunk_size=int(max(1, chunk_size)),
            maf=float(maf),
            missing_rate=float(geno),
            impute=False,
            sample_ids=keep_ids,
            ranges=ranges,
        ):
            arr = np.asarray(geno_chunk, dtype=np.float32)
            if arr.ndim != 2 or int(arr.shape[0]) <= 0:
                continue
            r = int(arr.shape[0])
            mm[wpos : wpos + r, :] = arr
            wpos += r
        if int(wpos) != int(rows):
            raise RuntimeError(f"NPY cache row count mismatch: wrote {wpos}, expected {rows}")
    finally:
        del mm
    tmp_npy.replace(out_npy)
    return int(rows), int(cols)


def _resolve_jx_launcher() -> list[str]:
    raw = str(os.environ.get("JX_BENCH_JX", "")).strip()
    if raw != "":
        return shlex.split(raw)
    jx_bin = shutil.which("jx")
    if jx_bin:
        return [jx_bin]
    return [sys.executable, "-m", "janusx.script.JanusX"]


def _resolve_plink_launcher(raw_cmd: str | None = None) -> Optional[list[str]]:
    raw = str(raw_cmd).strip() if raw_cmd is not None else ""
    if raw != "":
        return shlex.split(raw)
    env_raw = str(os.environ.get("JX_BENCH_PLINK", "")).strip()
    if env_raw != "":
        return shlex.split(env_raw)
    p2 = shutil.which("plink2")
    if p2:
        return [p2]
    p1 = shutil.which("plink")
    if p1:
        return [p1]
    return None


@dataclass
class ProcRun:
    cmd: list[str]
    returncode: int
    wall_sec: float
    peak_rss_bytes: Optional[int]
    peak_uss_bytes: Optional[int]
    stdout: str
    stderr: str


def _proc_tree_rss_bytes(proc: "psutil.Process") -> int:
    rss = 0
    try:
        rss += int(proc.memory_info().rss)
    except Exception:
        return 0
    try:
        for ch in proc.children(recursive=True):
            try:
                rss += int(ch.memory_info().rss)
            except Exception:
                continue
    except Exception:
        pass
    return int(rss)


def _proc_tree_rss_uss_bytes(proc: "psutil.Process") -> tuple[int, Optional[int]]:
    rss = 0
    uss = 0
    uss_seen = False

    def _accumulate_one(p: "psutil.Process") -> None:
        nonlocal rss, uss, uss_seen
        try:
            rss += int(p.memory_info().rss)
        except Exception:
            pass
        try:
            fi = p.memory_full_info()
            uv = getattr(fi, "uss", None)
            if uv is not None:
                uss += int(uv)
                uss_seen = True
        except Exception:
            pass

    _accumulate_one(proc)
    try:
        for ch in proc.children(recursive=True):
            _accumulate_one(ch)
    except Exception:
        pass
    return int(rss), (int(uss) if uss_seen else None)


def _run_monitored(
    cmd: list[str],
    *,
    env: Optional[dict[str, str]] = None,
    cwd: Optional[str] = None,
    poll_sec: float = 0.05,
) -> ProcRun:
    t0 = time.perf_counter()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=cwd,
    )
    peak_rss: Optional[int] = None
    peak_uss: Optional[int] = None
    proc = None
    if psutil is not None:
        try:
            proc = psutil.Process(int(p.pid))
        except Exception:
            proc = None

    while True:
        rc = p.poll()
        if proc is not None:
            rss_now = _proc_tree_rss_bytes(proc)
            if peak_rss is None:
                peak_rss = int(rss_now)
            else:
                peak_rss = max(int(peak_rss), int(rss_now))
        if rc is not None:
            break
        time.sleep(max(0.01, float(poll_sec)))

    stdout, stderr = p.communicate()
    if proc is not None:
        rss_now = _proc_tree_rss_bytes(proc)
        if peak_rss is None:
            peak_rss = int(rss_now)
        else:
            peak_rss = max(int(peak_rss), int(rss_now))
        _rss_end, uss_now = _proc_tree_rss_uss_bytes(proc)
        if uss_now is not None:
            if peak_uss is None:
                peak_uss = int(uss_now)
            else:
                peak_uss = max(int(peak_uss), int(uss_now))

    wall = max(0.0, time.perf_counter() - t0)
    return ProcRun(
        cmd=list(cmd),
        returncode=int(p.returncode),
        wall_sec=float(wall),
        peak_rss_bytes=(None if peak_rss is None else int(peak_rss)),
        peak_uss_bytes=(None if peak_uss is None else int(peak_uss)),
        stdout=str(stdout),
        stderr=str(stderr),
    )


def _extract_last_json_line(text: str) -> dict[str, Any]:
    for line in reversed(str(text).splitlines()):
        s = line.strip()
        if not s:
            continue
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return {}


@dataclass
class BenchRow:
    dataset: str
    prefix: str
    task: str
    repeat_idx: int
    status: str
    returncode: int
    wall_sec: float
    peak_rss_gb: Optional[float]
    peak_uss_gb: Optional[float]
    n_samples: Optional[int]
    n_snps: Optional[int]
    n_chunks: Optional[int]
    cmd: str
    notes: str


def _python_task_runner_code(task_name: str) -> str:
    if task_name == "stream_bed":
        return r"""
import json, os, time
import threading
import psutil
import janusx.janusx as jxrs
cfg = json.loads(os.environ["JX_BENCH_CFG"])
if not hasattr(jxrs, "bed_filter_stream_to_plink_rust"):
    raise RuntimeError("Rust symbol bed_filter_stream_to_plink_rust is unavailable")
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
n_rows = 0
n_scanned = 0
n_samples = 0
try:
    n_rows, n_scanned, n_samples = jxrs.bed_filter_stream_to_plink_rust(
        str(cfg["prefix"]),
        str(cfg["out_prefix"]),
        maf_threshold=float(cfg["maf"]),
        max_missing_rate=float(cfg["geno"]),
        model="add",
        het_threshold=0.0,
    )
finally:
    _stop["v"] = True
    _th.join(timeout=0.2)
    _sample_mem(True)
elapsed = max(0.0, time.perf_counter() - t0)
print(json.dumps({
    "task": "stream_bed",
    "elapsed_sec": float(elapsed),
    "n_samples": int(n_samples),
    "n_snps": int(n_rows),
    "n_snps_scanned": int(n_scanned),
    "n_chunks": None,
    "out_prefix": str(cfg["out_prefix"]),
    "peak_rss_self_bytes": int(_mem["rss"]),
    "peak_uss_self_bytes": (None if _mem["uss"] is None else int(_mem["uss"])),
}))
"""
    if task_name == "packed_bed":
        return r"""
import json, os, time
import threading
import psutil
import janusx.janusx as jxrs
cfg = json.loads(os.environ["JX_BENCH_CFG"])
if not hasattr(jxrs, "bed_filter_to_plink_rust"):
    raise RuntimeError("Rust symbol bed_filter_to_plink_rust is unavailable")
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
try:
    n_kept, n_scanned, n_samples = jxrs.bed_filter_to_plink_rust(
        str(cfg["prefix"]),
        str(cfg["out_prefix"]),
        maf_threshold=float(cfg["maf"]),
        max_missing_rate=float(cfg["geno"]),
        fill_missing=False,
        model="add",
        het_threshold=0.0,
    )
finally:
    _stop["v"] = True
    _th.join(timeout=0.2)
    _sample_mem(True)
elapsed = max(0.0, time.perf_counter() - t0)
print(json.dumps({
    "task": "packed_bed",
    "elapsed_sec": float(elapsed),
    "n_samples": int(n_samples),
    "n_snps": int(n_kept),
    "n_snps_scanned": int(n_scanned),
    "n_chunks": 1,
    "out_prefix": str(cfg["out_prefix"]),
    "peak_rss_self_bytes": int(_mem["rss"]),
    "peak_uss_self_bytes": (None if _mem["uss"] is None else int(_mem["uss"])),
}))
"""
    if task_name == "memmap_bed":
        return r"""
import json, os, time
import threading
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
chunk_size = max(1, int(cfg["chunk_size"]))
maf_thr = float(cfg["maf"])
miss_thr = float(cfg["geno"])

t0 = time.perf_counter()
rows_total = 0
blocks = 0
kept = 0
n_samples = 0
try:
    kept, rows_total, n_samples, blocks = jxrs.bed_mmap_filter_to_plink_rust(
        str(cfg["prefix"]),
        str(cfg["out_prefix"]),
        maf_threshold=float(maf_thr),
        max_missing_rate=float(miss_thr),
        het_threshold=0.0,
        block_rows=int(chunk_size),
        parallel=True,
    )
finally:
    _stop["v"] = True
    _th.join(timeout=0.2)
    _sample_mem(True)

elapsed = max(0.0, time.perf_counter() - t0)
print(json.dumps({
    "task": "memmap_bed",
    "elapsed_sec": float(elapsed),
    "n_samples": int(n_samples),
    "n_snps": int(kept),
    "n_snps_scanned": int(rows_total),
    "n_chunks": int(blocks),
    "out_prefix": str(cfg["out_prefix"]),
    "peak_rss_self_bytes": int(_mem["rss"]),
    "peak_uss_self_bytes": (None if _mem["uss"] is None else int(_mem["uss"])),
}))
"""
    if task_name == "grm_stream":
        return r"""
import json, os, time
import numpy as np
import janusx.janusx as jxrs
cfg = json.loads(os.environ["JX_BENCH_CFG"])
if not hasattr(jxrs, "grm_stream_bed_f32"):
    raise RuntimeError("Rust symbol grm_stream_bed_f32 is unavailable")
t0 = time.perf_counter()
grm, eff_m, n = jxrs.grm_stream_bed_f32(
    str(cfg["prefix"]),
    method=1,
    maf_threshold=float(cfg["maf"]),
    max_missing_rate=float(cfg["geno"]),
    block_cols=max(1, int(cfg["chunk_size"])),
    threads=max(1, int(cfg["threads"])),
    progress_callback=None,
    progress_every=0,
    mmap_window_mb=None,
)
# Ensure materialized and then release to include realistic memory behavior.
_ = np.asarray(grm, dtype=np.float32).shape
del grm
elapsed = max(0.0, time.perf_counter() - t0)
print(json.dumps({
    "task": "grm_stream",
    "elapsed_sec": float(elapsed),
    "n_samples": int(n),
    "n_snps": int(eff_m),
    "n_chunks": None,
}))
"""
    raise ValueError(f"unsupported python task: {task_name}")


def _run_python_task(
    *,
    task: str,
    cfg: dict[str, Any],
    cwd: str,
) -> tuple[ProcRun, dict[str, Any]]:
    code = _python_task_runner_code(task)
    env = dict(os.environ)
    env["JX_BENCH_CFG"] = json.dumps(cfg, ensure_ascii=False)
    cmd = [sys.executable, "-c", code]
    run = _run_monitored(cmd, env=env, cwd=cwd)
    obj = _extract_last_json_line(run.stdout)
    return run, obj


def _run_gformat_cli(
    *,
    jx_launcher: list[str],
    prefix: str,
    out_dir: Path,
    out_prefix_name: str,
    maf: float,
    geno: float,
    keep_file: Path,
    range_file: Path,
    threads: int,
    cwd: str,
) -> tuple[ProcRun, dict[str, Any], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = list(jx_launcher) + [
        "gformat",
        "-bfile",
        str(prefix),
        "-fmt",
        "plink",
        "-o",
        str(out_dir),
        "-prefix",
        str(out_prefix_name),
        "-maf",
        str(float(maf)),
        "-geno",
        str(float(geno)),
        "-keep",
        str(keep_file),
        "-extract",
        "range",
        str(range_file),
        "--thread",
        str(max(1, int(threads))),
    ]
    run = _run_monitored(cmd, cwd=cwd)
    out_prefix = out_dir / out_prefix_name
    info: dict[str, Any] = {}
    if run.returncode == 0:
        fam_n = _count_fam(str(out_prefix))
        bim_n = _count_bim(str(out_prefix))
        info = {
            "n_samples": int(fam_n),
            "n_snps": int(bim_n),
            "n_chunks": None,
        }
    return run, info, str(out_prefix)


def _run_plink_cli(
    *,
    plink_launcher: list[str],
    prefix: str,
    out_dir: Path,
    out_prefix_name: str,
    maf: float,
    geno: float,
    threads: int,
    cwd: str,
) -> tuple[ProcRun, dict[str, Any], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / out_prefix_name
    cmd = list(plink_launcher) + [
        "--bfile",
        str(prefix),
        "--maf",
        str(float(maf)),
        "--geno",
        str(float(geno)),
        "--make-bed",
        "--allow-extra-chr",
        "--threads",
        str(max(1, int(threads))),
        "--out",
        str(out_prefix),
    ]
    run = _run_monitored(cmd, cwd=cwd)
    info: dict[str, Any] = {}
    if run.returncode == 0:
        fam_n = _count_fam(str(out_prefix))
        bim_n = _count_bim(str(out_prefix))
        info = {
            "n_samples": int(fam_n),
            "n_snps": int(bim_n),
            "n_chunks": None,
        }
    return run, info, str(out_prefix)


def _run_plink_source_to_bed(
    *,
    plink_launcher: list[str],
    src_kind: str,
    src_path: str,
    out_dir: Path,
    out_prefix_name: str,
    maf: float,
    geno: float,
    threads: int,
    cwd: str,
) -> tuple[ProcRun, dict[str, Any], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / out_prefix_name
    input_flag = f"--{str(src_kind)}"
    cmd = list(plink_launcher) + [
        input_flag,
        str(src_path),
        "--maf",
        str(float(maf)),
        "--geno",
        str(float(geno)),
        "--make-bed",
        "--allow-extra-chr",
        "--threads",
        str(max(1, int(threads))),
        "--out",
        str(out_prefix),
    ]
    run = _run_monitored(cmd, cwd=cwd)
    info: dict[str, Any] = {}
    if run.returncode == 0:
        fam_n = _count_fam(str(out_prefix))
        bim_n = _count_bim(str(out_prefix))
        info = {
            "n_samples": int(fam_n),
            "n_snps": int(bim_n),
            "n_chunks": None,
        }
    return run, info, str(out_prefix)


@dataclass
class DatasetSpec:
    label: str
    prefix: str


@dataclass
class SourceSpec:
    label: str
    kind: str  # "vcf" | "hmp"
    path: str


def _parse_dataset_specs(values: list[str]) -> list[DatasetSpec]:
    out: list[DatasetSpec] = []
    for raw in values:
        s = str(raw).strip()
        if s == "":
            continue
        if "=" in s:
            label, path = s.split("=", 1)
            label = str(label).strip()
            path = str(path).strip()
            if label == "":
                label = Path(path).name
        else:
            path = s
            label = Path(path).name
        prefix = _normalize_plink_prefix(path)
        out.append(DatasetSpec(label=str(label), prefix=str(prefix)))
    if len(out) == 0:
        raise ValueError("No datasets provided. Use --dataset LABEL=PREFIX at least once.")
    return out


def _parse_source_specs(values: list[str], kind: str) -> list[SourceSpec]:
    out: list[SourceSpec] = []
    for raw in values:
        s = str(raw).strip()
        if s == "":
            continue
        if "=" in s:
            label, path = s.split("=", 1)
            label = str(label).strip()
            path = str(path).strip()
            if label == "":
                label = Path(path).stem
        else:
            path = s
            label = Path(path).stem
        out.append(SourceSpec(label=str(label), kind=str(kind), path=str(path)))
    return out


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


def _summarize(rows: list[BenchRow], baseline_label: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[BenchRow]] = {}
    for r in rows:
        key = (r.dataset, r.task)
        grouped.setdefault(key, []).append(r)

    task_baseline_time: dict[str, float] = {}
    for task in sorted({r.task for r in rows}):
        k = (baseline_label, task)
        vals = [
            x.wall_sec
            for x in grouped.get(k, [])
            if x.status == "ok" and math.isfinite(float(x.wall_sec))
        ]
        if len(vals) > 0:
            task_baseline_time[task] = float(sum(vals) / len(vals))

    task_mean_by_dataset: dict[tuple[str, str], float] = {}
    for (dataset, task), items in grouped.items():
        vals = [
            x.wall_sec
            for x in items
            if x.status == "ok" and math.isfinite(float(x.wall_sec))
        ]
        if len(vals) > 0:
            task_mean_by_dataset[(dataset, task)] = float(sum(vals) / len(vals))

    plink_reference_task_by_task = {
        "plink": "plink",
        "stream_bed": "plink",
        "packed_bed": "plink",
        "memmap_bed": "plink",
    }

    out: list[dict[str, Any]] = []
    for (dataset, task), items in sorted(grouped.items()):
        ok_items = [x for x in items if x.status == "ok"]
        if len(ok_items) == 0:
            out.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "n_runs": len(items),
                    "ok_runs": 0,
                    "wall_sec_mean": math.nan,
                    "wall_sec_min": math.nan,
                    "wall_sec_max": math.nan,
                    "peak_rss_gb_mean": math.nan,
                    "peak_uss_gb_mean": math.nan,
                    "speedup_vs_baseline": math.nan,
                    "speedup_vs_plink_same_dataset": math.nan,
                }
            )
            continue
        walls = [float(x.wall_sec) for x in ok_items]
        peaks = [float(x.peak_rss_gb) for x in ok_items if x.peak_rss_gb is not None]
        peaks_uss = [float(x.peak_uss_gb) for x in ok_items if x.peak_uss_gb is not None]
        mean_wall = float(sum(walls) / len(walls))
        mean_peak = float(sum(peaks) / len(peaks)) if len(peaks) > 0 else math.nan
        mean_peak_uss = float(sum(peaks_uss) / len(peaks_uss)) if len(peaks_uss) > 0 else math.nan
        speedup = math.nan
        base = task_baseline_time.get(task, math.nan)
        if math.isfinite(base) and base > 0.0:
            speedup = float(base / mean_wall)
        speedup_vs_plink = math.nan
        plink_ref_task = plink_reference_task_by_task.get(task, "")
        if plink_ref_task:
            plink_base = task_mean_by_dataset.get((dataset, plink_ref_task), math.nan)
            if math.isfinite(plink_base) and plink_base > 0.0:
                speedup_vs_plink = float(plink_base / mean_wall)
        out.append(
            {
                "dataset": dataset,
                "task": task,
                "n_runs": len(items),
                "ok_runs": len(ok_items),
                "wall_sec_mean": mean_wall,
                "wall_sec_min": min(walls),
                "wall_sec_max": max(walls),
                "peak_rss_gb_mean": mean_peak,
                "peak_uss_gb_mean": mean_peak_uss,
                "speedup_vs_baseline": speedup,
                "speedup_vs_plink_same_dataset": speedup_vs_plink,
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_genotype_io.py",
        description=(
            "Benchmark full-genotype PLINK BED IO methods only: "
            "plink / stream_bed / packed_bed / memmap_bed."
        ),
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset spec: LABEL=PLINK_PREFIX (repeatable). "
            "Example: --dataset HDD=/Volumes/.../cubic_All.maf0.02 "
            "--dataset SSD=/Users/.../cubic_All.maf0.02"
        ),
    )
    p.add_argument("--outdir", default="bench_io_results", help="Output directory.")
    p.add_argument("--repeat", type=int, default=3, help="Measured repeats per task.")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per task.")
    p.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1), help="Thread count.")
    p.add_argument("--chunk-size", type=int, default=50_000, help="Chunk/block size for streaming tasks.")
    p.add_argument("--maf", type=float, default=0.02, help="MAF filter threshold.")
    p.add_argument("--geno", type=float, default=0.10, help="Max missing-rate filter.")
    p.add_argument(
        "--baseline",
        default=None,
        help="Baseline dataset label for speedup (default: first dataset).",
    )
    p.add_argument(
        "--plink-cmd",
        default=None,
        help=(
            "Optional PLINK command override (e.g. 'plink2' or '/path/to/plink'). "
            "Default: JX_BENCH_PLINK env, then auto-detect plink2/plink."
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    specs = _parse_dataset_specs(list(args.dataset)) if len(args.dataset) > 0 else []
    if len(specs) == 0:
        raise ValueError("Provide at least one --dataset LABEL=PLINK_PREFIX.")

    if args.baseline:
        baseline_label = str(args.baseline).strip()
    else:
        baseline_label = specs[0].label
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    work_root = outdir / "runs"
    work_root.mkdir(parents=True, exist_ok=True)

    plink_launcher = _resolve_plink_launcher(args.plink_cmd)
    if plink_launcher is None:
        raise RuntimeError(
            "PLINK launcher not found. Please install plink2/plink or pass --plink-cmd."
        )
    try:
        import janusx.janusx as _jx
        req = [
            "bed_filter_stream_to_plink_rust",
            "bed_filter_to_plink_rust",
            "bed_mmap_filter_to_plink_rust",
        ]
        miss = [k for k in req if not hasattr(_jx, k)]
        if len(miss) > 0:
            raise RuntimeError(f"Rust symbols unavailable: {', '.join(miss)}")
    except Exception as ex:
        raise RuntimeError(
            f"benchmark requires Rust filter+write symbols in janusx.janusx: {ex}"
        ) from ex

    print(f"[info] plink base: {' '.join(shlex.quote(x) for x in plink_launcher)}")
    print(f"[info] output dir : {outdir}")
    print(f"[info] baseline   : {baseline_label}")
    print("[info] tasks      : plink, stream_bed, packed_bed, memmap_bed")

    rows: list[BenchRow] = []

    for ds in specs:
        prefix = _normalize_plink_prefix(ds.prefix)
        if not Path(f"{prefix}.bed").exists():
            raise FileNotFoundError(f"BED not found: {prefix}.bed")
        if not Path(f"{prefix}.bim").exists():
            raise FileNotFoundError(f"BIM not found: {prefix}.bim")
        if not Path(f"{prefix}.fam").exists():
            raise FileNotFoundError(f"FAM not found: {prefix}.fam")

        n_samples_full = _count_fam(prefix)
        n_snps_full = _count_bim(prefix)

        ds_dir = work_root / ds.label
        ds_dir.mkdir(parents=True, exist_ok=True)
        print(f"[dataset] {ds.label}: n={n_samples_full}, m={n_snps_full}")

        task_list = ["plink", "stream_bed", "packed_bed", "memmap_bed"]

        for task in task_list:
            n_total_runs = int(max(0, args.warmup) + max(1, args.repeat))
            print(f"[task] {ds.label} / {task} (warmup={args.warmup}, repeat={args.repeat})")
            for run_idx in range(n_total_runs):
                is_warmup = run_idx < int(max(0, args.warmup))
                rep_idx = run_idx - int(max(0, args.warmup))
                run_dir = ds_dir / task / f"run_{run_idx:02d}"
                run_dir.mkdir(parents=True, exist_ok=True)

                task_cfg = {
                    "prefix": prefix,
                    "maf": float(args.maf),
                    "geno": float(args.geno),
                    "chunk_size": int(max(1, args.chunk_size)),
                    "threads": int(max(1, args.threads)),
                    "out_prefix": str(run_dir / "out" / f"{ds.label}_{task}_{run_idx}"),
                }

                if task == "plink":
                    run, info, out_prefix = _run_plink_cli(
                        plink_launcher=plink_launcher,
                        prefix=prefix,
                        out_dir=run_dir / "out",
                        out_prefix_name=f"{ds.label}_{task}_{run_idx}",
                        maf=float(args.maf),
                        geno=float(args.geno),
                        threads=int(max(1, args.threads)),
                        cwd=str(Path.cwd()),
                    )
                    n_samples = int(info.get("n_samples")) if "n_samples" in info else None
                    n_snps = int(info.get("n_snps")) if "n_snps" in info else None
                    n_chunks = None
                    notes = f"out_prefix={out_prefix}"
                    peak_rss_override_bytes: Optional[int] = None
                    peak_uss_override_bytes: Optional[int] = None
                else:
                    run, obj = _run_python_task(task=task, cfg=task_cfg, cwd=str(Path.cwd()))
                    n_samples = int(obj.get("n_samples")) if "n_samples" in obj else None
                    n_snps = int(obj.get("n_snps")) if "n_snps" in obj else None
                    n_chunks = int(obj.get("n_chunks")) if obj.get("n_chunks") is not None else None
                    notes = json.dumps(obj, ensure_ascii=False) if obj else ""
                    peak_rss_override_bytes = None
                    peak_uss_override_bytes = None
                    if obj:
                        try:
                            v = obj.get("peak_rss_self_bytes", None)
                            if v is not None:
                                peak_rss_override_bytes = int(v)
                        except Exception:
                            peak_rss_override_bytes = None
                        try:
                            v = obj.get("peak_uss_self_bytes", None)
                            if v is not None:
                                peak_uss_override_bytes = int(v)
                        except Exception:
                            peak_uss_override_bytes = None

                peak_rss_bytes = (
                    peak_rss_override_bytes
                    if (peak_rss_override_bytes is not None and peak_rss_override_bytes > 0)
                    else run.peak_rss_bytes
                )
                peak_uss_bytes = (
                    peak_uss_override_bytes
                    if (peak_uss_override_bytes is not None and peak_uss_override_bytes > 0)
                    else run.peak_uss_bytes
                )
                peak_gb = (
                    (float(peak_rss_bytes) / (1024.0**3))
                    if peak_rss_bytes is not None
                    else None
                )
                peak_uss_gb = (
                    (float(peak_uss_bytes) / (1024.0**3))
                    if peak_uss_bytes is not None
                    else None
                )
                status = "ok" if run.returncode == 0 else "failed"
                line = (
                    f"  {'warmup' if is_warmup else f'rep={rep_idx:02d}'} "
                    f"rc={run.returncode} wall={_fmt_secs(run.wall_sec)}s "
                    f"rss={_fmt_gb(peak_gb)}GB uss={_fmt_gb(peak_uss_gb)}GB"
                )
                if n_samples is not None and n_snps is not None:
                    line += f" n={n_samples} m={n_snps}"
                print(line)

                # Save logs for each run.
                (run_dir / "stdout.log").write_text(run.stdout, encoding="utf-8", errors="replace")
                (run_dir / "stderr.log").write_text(run.stderr, encoding="utf-8", errors="replace")
                (run_dir / "cmd.txt").write_text(
                    " ".join(shlex.quote(x) for x in run.cmd),
                    encoding="utf-8",
                )

                if is_warmup:
                    continue

                rows.append(
                    BenchRow(
                        dataset=ds.label,
                        prefix=prefix,
                        task=task,
                        repeat_idx=int(rep_idx),
                        status=status,
                        returncode=int(run.returncode),
                        wall_sec=float(run.wall_sec),
                        peak_rss_gb=peak_gb,
                        peak_uss_gb=peak_uss_gb,
                        n_samples=n_samples,
                        n_snps=n_snps,
                        n_chunks=n_chunks,
                        cmd=" ".join(shlex.quote(x) for x in run.cmd),
                        notes=notes,
                    )
                )

    rows_json = [asdict(r) for r in rows]
    summary = _summarize(rows, baseline_label=baseline_label)

    json_path = outdir / "benchmark_results.json"
    csv_path = outdir / "benchmark_results.csv"
    sum_path = outdir / "benchmark_summary.csv"

    json_path.write_text(
        json.dumps(
            {
                "baseline": baseline_label,
                "generated_at_epoch": time.time(),
                "args": vars(args),
                "results": rows_json,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(csv_path, rows_json)
    _write_csv(sum_path, summary)

    print("\n[done] wrote:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {sum_path}")


if __name__ == "__main__":
    main()
