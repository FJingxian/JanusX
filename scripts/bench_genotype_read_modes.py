#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark stream / packed / memmap genotype read performance on PLINK BED.

Compared modes
--------------
- stream: janusx.gfreader.load_genotype_chunks (Rust stream decoder)
- packed: prepare_bed_2bit_packed + bed_packed_decode_rows_f32
- memmap: BedMmapReader packed-row reads + NumPy-side sample/range/maf/geno filter

Workloads
---------
- sequential: read consecutive query windows
- random: read random query windows

Each query reads multiple chunks in one pass:
  rows_per_query = chunk_size * blocks_per_query
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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


def _normalize_plink_prefix(path_or_prefix: str) -> str:
    s = str(path_or_prefix).strip()
    low = s.lower()
    if low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam"):
        return s[:-4]
    return s


def _normalize_chr_key(chrom: str) -> str:
    s = str(chrom).strip()
    low = s.lower()
    if low.startswith("chr"):
        s = s[3:]
    return s.strip().upper()


def _safe_float(v: Any, default: float = math.nan) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


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


def _fmt_secs(x: float) -> str:
    if not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.3f}"


def _fmt_gb(x: Optional[float]) -> str:
    if x is None or (not math.isfinite(float(x))):
        return "nan"
    return f"{float(x):.3f}"


def _parse_range_token(token: str) -> tuple[str, int, int]:
    s = str(token).strip()
    if ":" not in s:
        raise ValueError(f"Invalid range token: {token}")
    chrom, rest = s.split(":", 1)
    if "-" in rest:
        a, b = rest.split("-", 1)
    elif ":" in rest:
        a, b = rest.split(":", 1)
    else:
        raise ValueError(f"Invalid range token: {token}")
    start = int(a)
    end = int(b)
    if start > end:
        start, end = end, start
    return _normalize_chr_key(chrom), int(start), int(end)


def _read_ranges_file(path: Path) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()
            if len(tok) < 3:
                raise ValueError(f"Malformed ranges file line {ln}: {line.rstrip()}")
            c = _normalize_chr_key(tok[0])
            a = int(tok[1])
            b = int(tok[2])
            if a > b:
                a, b = b, a
            out.append((c, int(a), int(b)))
    if len(out) == 0:
        raise ValueError(f"No ranges found in {path}")
    return out


def _iter_bim(prefix: str):
    bim = Path(f"{prefix}.bim")
    with bim.open("r", encoding="utf-8", errors="replace") as fh:
        for idx, line in enumerate(fh):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()
            if len(tok) < 4:
                raise ValueError(f"Malformed BIM line at {bim}:{idx+1}")
            chrom = _normalize_chr_key(tok[0])
            try:
                pos = int(float(tok[3]))
            except Exception as ex:
                raise ValueError(f"Invalid BIM position at {bim}:{idx+1}") from ex
            yield idx, chrom, pos


def _choose_auto_range(prefix: str, snp_fraction: float) -> tuple[str, int, int]:
    counts: dict[str, int] = {}
    for _, c, _ in _iter_bim(prefix):
        counts[c] = counts.get(c, 0) + 1
    if len(counts) == 0:
        raise ValueError(f"No variants found in {prefix}.bim")
    chrom = max(counts.keys(), key=lambda k: counts[k])

    pos: list[int] = []
    for _, c, p in _iter_bim(prefix):
        if c == chrom:
            pos.append(int(p))
    if len(pos) == 0:
        raise ValueError(f"No variants found for chromosome {chrom}")
    pos.sort()

    frac = min(1.0, max(0.001, float(snp_fraction)))
    m = len(pos)
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


def _collect_indices_for_ranges(prefix: str, ranges: list[tuple[str, int, int]]) -> list[int]:
    rr = [(_normalize_chr_key(c), int(a), int(b)) for c, a, b in ranges]
    out: list[int] = []
    for idx, chrom, pos in _iter_bim(prefix):
        for c, a, b in rr:
            if chrom == c and pos >= a and pos <= b:
                out.append(int(idx))
                break
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


def _read_ids_file(path: Path) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s.split()[0])
    return out


def _write_ids_file(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fw:
        for x in ids:
            fw.write(f"{x}\n")


@dataclass
class ProcRun:
    cmd: list[str]
    returncode: int
    wall_sec: float
    peak_rss_bytes: Optional[int]
    peak_uss_bytes: Optional[int]
    stdout: str
    stderr: str


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
            rss_now, uss_now = _proc_tree_rss_uss_bytes(proc)
            peak_rss = max(int(peak_rss or 0), int(rss_now))
            if uss_now is not None:
                peak_uss = max(int(peak_uss or 0), int(uss_now))
        if rc is not None:
            break
        time.sleep(max(0.01, float(poll_sec)))

    stdout, stderr = p.communicate()
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


def _update_checksum(checksum: int, arr) -> int:
    import numpy as np

    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        return int(checksum)
    s1 = float(np.sum(x, dtype=np.float64))
    s2 = float(np.sum(np.square(x, dtype=np.float64), dtype=np.float64))
    a = int(round(s1 * 1000.0))
    b = int(round(s2 * 1000.0))
    out = (int(checksum) * 0x9E3779B185EBCA87 + (a & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    out = (out ^ (b & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    return int(out)


def _indices_to_runs(idx_arr) -> list[tuple[int, int]]:
    if len(idx_arr) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(idx_arr[0])
    prev = int(idx_arr[0])
    for x in idx_arr[1:]:
        xi = int(x)
        if xi == prev + 1:
            prev = xi
            continue
        runs.append((start, prev - start + 1))
        start = xi
        prev = xi
    runs.append((start, prev - start + 1))
    return runs


def _decode_filter_memmap_block(
    packed_block,
    sample_indices,
    maf_thr: float,
    geno_thr: float,
    impute: bool,
):
    import numpy as np

    pb = np.asarray(packed_block, dtype=np.uint8, order="C")
    if pb.ndim != 2 or pb.shape[0] == 0:
        return np.empty((0, int(len(sample_indices))), dtype=np.float32)

    sidx = np.asarray(sample_indices, dtype=np.int64)
    if sidx.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    byte_idx = (sidx >> 2).astype(np.int64, copy=False)
    shifts = ((sidx & 3) * 2).astype(np.uint8, copy=False)

    codes = (pb[:, byte_idx] >> shifts[None, :]) & 0b11
    lut = np.array([0.0, np.nan, 1.0, 2.0], dtype=np.float32)
    g = lut[codes]

    n_samples = int(g.shape[1])
    if n_samples == 0:
        return np.empty((0, 0), dtype=np.float32)

    valid = np.isfinite(g)
    non_missing = np.sum(valid, axis=1, dtype=np.int64)
    miss_rate = 1.0 - (non_missing.astype(np.float64) / float(n_samples))
    keep = miss_rate <= float(geno_thr)

    if float(maf_thr) > 0.0:
        keep &= non_missing > 0

    alt_sum = np.nansum(g, axis=1, dtype=np.float64)
    p = np.zeros(g.shape[0], dtype=np.float64)
    has_obs = non_missing > 0
    p[has_obs] = alt_sum[has_obs] / (2.0 * non_missing[has_obs].astype(np.float64))

    flip = p > 0.5
    if np.any(flip):
        ridx = np.where(flip)[0]
        sub = g[ridx]
        vsub = valid[ridx]
        sub[vsub] = 2.0 - sub[vsub]
        g[ridx] = sub
        p[flip] = 1.0 - p[flip]

    maf = np.minimum(p, 1.0 - p)
    keep &= maf >= float(maf_thr)

    if impute:
        mean_g = np.zeros(g.shape[0], dtype=np.float32)
        mean_g[has_obs] = (2.0 * p[has_obs]).astype(np.float32)
        miss_r, miss_c = np.where(~np.isfinite(g))
        if miss_r.size > 0:
            g[miss_r, miss_c] = mean_g[miss_r]
    else:
        g[~np.isfinite(g)] = -9.0

    keep_idx = np.where(keep)[0]
    if keep_idx.size == 0:
        return np.empty((0, n_samples), dtype=np.float32)
    return np.ascontiguousarray(g[keep_idx], dtype=np.float32)


def _worker_run(cfg: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from janusx.gfreader import (
        inspect_genotype_file,
        load_genotype_chunks,
        prepare_bed_2bit_packed,
    )
    from janusx.janusx import BedMmapReader, bed_packed_decode_rows_f32

    prefix = _normalize_plink_prefix(str(cfg["prefix"]))
    mode = str(cfg["mode"])
    workload = str(cfg["workload"])
    maf = float(cfg["maf"])
    geno = float(cfg["geno"])
    chunk_size = max(1, int(cfg["chunk_size"]))
    blocks_per_query = max(1, int(cfg["blocks_per_query"]))
    rows_per_query = int(chunk_size * blocks_per_query)
    queries = int(cfg["seq_queries"] if workload == "sequential" else cfg["rand_queries"])
    impute = bool(cfg.get("impute", True))
    seed = int(cfg.get("seed", 20260511))

    sample_ids = _read_ids_file(Path(str(cfg["sample_ids_file"])))
    raw_idx = np.load(str(cfg["raw_indices_file"]), mmap_mode="r")
    if raw_idx.ndim != 1:
        raise RuntimeError("raw_indices_file must store a 1D index array")
    n_candidate = int(raw_idx.shape[0])
    if n_candidate <= 0:
        raise RuntimeError("No SNP index selected by requested ranges")
    if len(sample_ids) == 0:
        raise RuntimeError("No selected samples")

    _, n_total_snps = inspect_genotype_file(prefix)
    if int(n_total_snps) <= 0:
        raise RuntimeError("Invalid SNP count from inspect_genotype_file")

    rng = random.Random(seed)
    starts: list[int] = []
    if queries > 0:
        if workload == "sequential":
            for q in range(queries):
                st = q * rows_per_query
                if st >= n_candidate:
                    break
                starts.append(int(st))
        else:
            max_start = max(0, n_candidate - rows_per_query)
            for _ in range(queries):
                starts.append(int(rng.randint(0, max_start)))

    t0 = time.perf_counter()
    t_init = 0.0
    rows_requested = 0
    rows_kept = 0
    blocks_returned = 0
    checksum = 0
    queries_done = 0
    n_selected_after_filter = None

    if mode == "stream":
        for st in starts:
            ed = min(n_candidate, st + rows_per_query)
            if ed <= st:
                continue
            raw_slice = np.asarray(raw_idx[st:ed], dtype=np.int64)
            rows_requested += int(raw_slice.shape[0])
            it = load_genotype_chunks(
                prefix,
                chunk_size=int(chunk_size),
                maf=float(maf),
                missing_rate=float(geno),
                impute=bool(impute),
                sample_ids=list(sample_ids),
                snp_indices=raw_slice.tolist(),
                mmap_window_mb=None,
            )
            got_any = False
            for geno_chunk, _sites in it:
                arr = np.asarray(geno_chunk, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[0] == 0:
                    continue
                got_any = True
                rows_kept += int(arr.shape[0])
                blocks_returned += 1
                checksum = _update_checksum(checksum, arr)
            if got_any:
                queries_done += 1

    elif mode == "packed":
        ti0 = time.perf_counter()
        (
            packed,
            _miss,
            row_maf,
            _std,
            row_flip,
            site_keep,
            n_all_samples,
            _n_all_snps,
        ) = prepare_bed_2bit_packed(
            prefix,
            maf_threshold=float(maf),
            max_missing_rate=float(geno),
            het_threshold=0.0,
            snps_only=False,
        )
        all_ids, _ = inspect_genotype_file(prefix)
        id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
        sidx = []
        for sid in sample_ids:
            if sid not in id_to_idx:
                raise RuntimeError(f"sample id not found in FAM: {sid}")
            sidx.append(int(id_to_idx[sid]))
        sample_idx_arr = np.asarray(sidx, dtype=np.int64)

        site_keep_np = np.asarray(site_keep, dtype=np.bool_)
        csum = np.cumsum(site_keep_np.astype(np.int64, copy=False), dtype=np.int64)
        raw_full = np.asarray(raw_idx, dtype=np.int64)
        keep_mask = site_keep_np[raw_full]
        packed_idx_for_raw = np.where(keep_mask, csum[raw_full] - 1, -1).astype(np.int64, copy=False)
        n_selected_after_filter = int(np.sum(keep_mask, dtype=np.int64))
        t_init = max(0.0, time.perf_counter() - ti0)

        for st in starts:
            ed = min(n_candidate, st + rows_per_query)
            if ed <= st:
                continue
            raw_slice = np.asarray(raw_idx[st:ed], dtype=np.int64)
            rows_requested += int(raw_slice.shape[0])

            pidx = np.asarray(packed_idx_for_raw[st:ed], dtype=np.int64)
            pidx = pidx[pidx >= 0]
            if pidx.size == 0:
                continue

            got_any = False
            for b0 in range(0, int(pidx.size), int(chunk_size)):
                b1 = min(int(pidx.size), b0 + int(chunk_size))
                chunk_idx = np.ascontiguousarray(pidx[b0:b1], dtype=np.int64)
                mat = bed_packed_decode_rows_f32(
                    packed,
                    int(n_all_samples),
                    chunk_idx,
                    row_flip,
                    row_maf,
                    sample_idx_arr,
                )
                arr = np.asarray(mat, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[0] == 0:
                    continue
                got_any = True
                rows_kept += int(arr.shape[0])
                blocks_returned += 1
                checksum = _update_checksum(checksum, arr)
            if got_any:
                queries_done += 1

    elif mode == "memmap":
        ti0 = time.perf_counter()
        reader = BedMmapReader(prefix)
        all_ids, _ = inspect_genotype_file(prefix)
        id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
        sidx = []
        for sid in sample_ids:
            if sid not in id_to_idx:
                raise RuntimeError(f"sample id not found in FAM: {sid}")
            sidx.append(int(id_to_idx[sid]))
        sample_idx_arr = np.asarray(sidx, dtype=np.int64)
        t_init = max(0.0, time.perf_counter() - ti0)

        for st in starts:
            ed = min(n_candidate, st + rows_per_query)
            if ed <= st:
                continue
            raw_slice = np.asarray(raw_idx[st:ed], dtype=np.int64)
            rows_requested += int(raw_slice.shape[0])
            runs = _indices_to_runs(raw_slice)
            if len(runs) == 0:
                continue

            got_any = False
            for run_start, run_len in runs:
                packed_mat = reader.read_rows_packed(int(run_start), int(run_len))
                arr_packed = np.asarray(packed_mat, dtype=np.uint8)
                if arr_packed.ndim != 2 or arr_packed.shape[0] == 0:
                    continue
                f = _decode_filter_memmap_block(
                    arr_packed,
                    sample_idx_arr,
                    float(maf),
                    float(geno),
                    bool(impute),
                )
                if f.ndim != 2 or f.shape[0] == 0:
                    continue
                got_any = True
                for b0 in range(0, int(f.shape[0]), int(chunk_size)):
                    b1 = min(int(f.shape[0]), b0 + int(chunk_size))
                    blk = f[b0:b1]
                    if blk.shape[0] == 0:
                        continue
                    rows_kept += int(blk.shape[0])
                    blocks_returned += 1
                    checksum = _update_checksum(checksum, blk)
            if got_any:
                queries_done += 1

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    total_sec = max(0.0, time.perf_counter() - t0)
    query_sec = max(0.0, total_sec - t_init)

    out = {
        "mode": mode,
        "workload": workload,
        "elapsed_sec": float(total_sec),
        "init_sec": float(t_init),
        "query_sec": float(query_sec),
        "rows_requested": int(rows_requested),
        "rows_kept": int(rows_kept),
        "blocks_returned": int(blocks_returned),
        "queries_done": int(queries_done),
        "chunk_size": int(chunk_size),
        "blocks_per_query": int(blocks_per_query),
        "rows_per_query": int(rows_per_query),
        "candidate_rows": int(n_candidate),
        "selected_samples": int(len(sample_ids)),
        "checksum_u64": int(checksum),
        "n_selected_after_filter": (
            None if n_selected_after_filter is None else int(n_selected_after_filter)
        ),
    }
    return out


@dataclass
class BenchRow:
    mode: str
    workload: str
    repeat_idx: int
    is_warmup: bool
    status: str
    returncode: int
    wall_sec: float
    elapsed_sec: float
    init_sec: float
    query_sec: float
    peak_rss_gb: Optional[float]
    peak_uss_gb: Optional[float]
    rows_requested: Optional[int]
    rows_kept: Optional[int]
    blocks_returned: Optional[int]
    queries_done: Optional[int]
    checksum_u64: Optional[int]
    cmd: str
    notes: str


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


def _print_summary(rows: list[BenchRow]) -> None:
    ok = [r for r in rows if (r.status == "ok" and (not r.is_warmup))]
    if len(ok) == 0:
        print("No successful measured runs.")
        return
    keys: list[tuple[str, str]] = sorted({(r.mode, r.workload) for r in ok})
    print("")
    print("=== Summary (median over measured repeats) ===")
    print(
        "mode/workload | wall(s) | query(s) | rows/s(query) | peak_rss(GB) | peak_uss(GB) | rows_kept"
    )
    for mode, workload in keys:
        sub = [r for r in ok if r.mode == mode and r.workload == workload]
        if len(sub) == 0:
            continue
        wall = sorted([float(r.wall_sec) for r in sub])[len(sub) // 2]
        qsec = sorted([float(r.query_sec) for r in sub])[len(sub) // 2]
        rss_vals = [r.peak_rss_gb for r in sub if r.peak_rss_gb is not None]
        uss_vals = [r.peak_uss_gb for r in sub if r.peak_uss_gb is not None]
        rows_kept_vals = [int(r.rows_kept or 0) for r in sub]
        rows_med = sorted(rows_kept_vals)[len(rows_kept_vals) // 2]
        if qsec > 0.0:
            thr = rows_med / qsec
        else:
            thr = math.nan
        rss_med = (
            sorted([float(x) for x in rss_vals])[len(rss_vals) // 2]
            if len(rss_vals) > 0
            else None
        )
        uss_med = (
            sorted([float(x) for x in uss_vals])[len(uss_vals) // 2]
            if len(uss_vals) > 0
            else None
        )
        print(
            f"{mode}/{workload} | {_fmt_secs(wall)} | {_fmt_secs(qsec)} | "
            f"{('nan' if not math.isfinite(thr) else f'{thr:,.1f}')} | "
            f"{_fmt_gb(rss_med)} | {_fmt_gb(uss_med)} | {rows_med}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_genotype_read_modes.py",
        description=(
            "Benchmark stream/packed/memmap genotype read speed and memory under "
            "sample subset + genomic range + maf/geno filtering."
        ),
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--bfile", required=False, help="PLINK prefix (or .bed/.bim/.fam path).")
    p.add_argument("--outdir", default="bench_read_modes_results", help="Output directory.")
    p.add_argument("--label", default=None, help="Dataset label for output naming.")

    p.add_argument("--sample-file", default=None, help="Sample ID file (one IID per line).")
    p.add_argument("--sample-frac", type=float, default=0.5, help="Keep fraction when --sample-file is not set.")
    p.add_argument("--sample-count", type=int, default=0, help="Absolute sample count override (>0).")

    p.add_argument("--range", dest="ranges", action="append", default=None, help="Range token: chr:start-end (repeatable).")
    p.add_argument("--ranges-file", default=None, help="Range file: chrom start end per line.")
    p.add_argument("--range-frac", type=float, default=0.2, help="Auto-selected interval fraction on densest chromosome.")

    p.add_argument("--maf", type=float, default=0.02, help="MAF threshold.")
    p.add_argument("--geno", type=float, default=0.10, help="Max missing-rate threshold.")
    p.add_argument("--impute", action="store_true", default=True, help="Enable mean-imputation for missing calls.")
    p.add_argument("--no-impute", dest="impute", action="store_false", help="Disable imputation.")

    p.add_argument("--chunk-size", type=int, default=2048, help="SNP rows per chunk/block.")
    p.add_argument("--blocks-per-query", type=int, default=4, help="Number of chunks read per query.")
    p.add_argument("--seq-queries", type=int, default=4, help="Number of sequential queries.")
    p.add_argument("--rand-queries", type=int, default=4, help="Number of random queries.")
    p.add_argument("--seed", type=int, default=20260511, help="Random seed for random-query sampling.")

    p.add_argument("--repeat", type=int, default=3, help="Measured repeats per mode/workload.")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per mode/workload.")
    p.add_argument("--modes", default="stream,packed,memmap", help="Comma-separated modes.")
    p.add_argument("--workloads", default="sequential,random", help="Comma-separated workloads.")
    p.add_argument("--keep-workdir", action="store_true", help="Keep intermediate plan files.")
    return p


def _worker_entry() -> int:
    raw = os.environ.get("JX_READ_BENCH_CFG", "").strip()
    if raw == "":
        raise RuntimeError("JX_READ_BENCH_CFG is empty")
    cfg = json.loads(raw)
    out = _worker_run(cfg)
    print(json.dumps(out, ensure_ascii=False))
    return 0


def _driver_entry(args: argparse.Namespace) -> int:
    if not args.bfile:
        raise ValueError("--bfile is required in driver mode")

    from janusx.gfreader import inspect_genotype_file
    import numpy as np

    prefix = _normalize_plink_prefix(str(args.bfile))
    label = str(args.label or Path(prefix).name)
    outdir = Path(str(args.outdir)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"{label}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_ids, n_snps = inspect_genotype_file(prefix)
    n_ids = len(all_ids)
    if n_ids <= 0:
        raise RuntimeError("No sample IDs from inspect_genotype_file")

    if args.sample_file:
        req_ids = _read_ids_file(Path(str(args.sample_file)))
        req_set = set(req_ids)
        sample_ids = [sid for sid in all_ids if sid in req_set]
        if len(sample_ids) == 0:
            raise RuntimeError("No overlap between --sample-file and FAM IIDs")
    else:
        if int(args.sample_count) > 0:
            k = int(args.sample_count)
        else:
            frac = min(1.0, max(0.001, float(args.sample_frac)))
            k = max(1, int(round(n_ids * frac)))
        keep_idx = _select_evenly_spaced_indices(n_ids, k)
        sample_ids = [all_ids[i] for i in keep_idx]

    ranges: list[tuple[str, int, int]] = []
    if args.ranges_file:
        ranges.extend(_read_ranges_file(Path(str(args.ranges_file))))
    if args.ranges:
        for tok in args.ranges:
            ranges.append(_parse_range_token(str(tok)))
    if len(ranges) == 0:
        ranges.append(_choose_auto_range(prefix, float(args.range_frac)))

    raw_idx_list = _collect_indices_for_ranges(prefix, ranges)
    if len(raw_idx_list) == 0:
        raise RuntimeError("No SNP index selected from requested ranges")
    raw_idx = np.asarray(raw_idx_list, dtype=np.int64)

    sample_file = run_dir / "selected_samples.txt"
    raw_idx_file = run_dir / "selected_raw_indices.npy"
    _write_ids_file(sample_file, sample_ids)
    np.save(raw_idx_file, raw_idx, allow_pickle=False)

    plan = {
        "prefix": prefix,
        "label": label,
        "n_total_samples": int(n_ids),
        "n_total_snps": int(n_snps),
        "selected_samples": int(len(sample_ids)),
        "selected_ranges": [[c, int(a), int(b)] for c, a, b in ranges],
        "selected_candidate_rows": int(raw_idx.shape[0]),
        "maf": float(args.maf),
        "geno": float(args.geno),
        "impute": bool(args.impute),
        "chunk_size": int(args.chunk_size),
        "blocks_per_query": int(args.blocks_per_query),
        "seq_queries": int(args.seq_queries),
        "rand_queries": int(args.rand_queries),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
    }
    with (run_dir / "plan.json").open("w", encoding="utf-8") as fw:
        json.dump(plan, fw, ensure_ascii=False, indent=2)

    modes = [m.strip().lower() for m in str(args.modes).split(",") if m.strip()]
    workloads = [w.strip().lower() for w in str(args.workloads).split(",") if w.strip()]
    modes = [m for m in modes if m in {"stream", "packed", "memmap"}]
    workloads = [w for w in workloads if w in {"sequential", "random"}]
    if len(modes) == 0:
        raise RuntimeError("No valid mode selected")
    if len(workloads) == 0:
        raise RuntimeError("No valid workload selected")

    rows: list[BenchRow] = []
    exe = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    cwd = str(Path(__file__).resolve().parents[1])

    print(f"Dataset: {label}")
    print(f"  prefix: {prefix}")
    print(f"  samples: {len(sample_ids)}/{n_ids}")
    print(f"  candidate SNP rows in ranges: {int(raw_idx.shape[0])}/{int(n_snps)}")
    print(f"  ranges: {ranges}")
    print(f"  chunk_size={int(args.chunk_size)}, blocks_per_query={int(args.blocks_per_query)}")
    print(f"  seq_queries={int(args.seq_queries)}, rand_queries={int(args.rand_queries)}")
    print(f"  repeat={int(args.repeat)}, warmup={int(args.warmup)}")
    print("")

    run_no = 0
    total_runs = len(modes) * len(workloads) * (int(args.warmup) + int(args.repeat))
    for mode in modes:
        for workload in workloads:
            for phase in ("warmup", "measure"):
                n_loop = int(args.warmup) if phase == "warmup" else int(args.repeat)
                for r in range(n_loop):
                    is_warmup = phase == "warmup"
                    cfg = {
                        "prefix": prefix,
                        "mode": mode,
                        "workload": workload,
                        "maf": float(args.maf),
                        "geno": float(args.geno),
                        "impute": bool(args.impute),
                        "chunk_size": int(args.chunk_size),
                        "blocks_per_query": int(args.blocks_per_query),
                        "seq_queries": int(args.seq_queries),
                        "rand_queries": int(args.rand_queries),
                        "sample_ids_file": str(sample_file),
                        "raw_indices_file": str(raw_idx_file),
                        "seed": int(args.seed) + int(r) + (1000 if workload == "random" else 0),
                    }
                    env = dict(os.environ)
                    env["JX_READ_BENCH_CFG"] = json.dumps(cfg, ensure_ascii=False)
                    run_no += 1
                    tag = f"[{run_no}/{total_runs}] {mode}/{workload} {'warmup' if is_warmup else 'run'}#{r}"
                    print(f"{tag} ...", end="", flush=True)
                    pr = _run_monitored(exe, env=env, cwd=cwd)
                    obj = _extract_last_json_line(pr.stdout)
                    status = "ok" if (pr.returncode == 0 and len(obj) > 0) else "failed"
                    peak_rss_gb = (
                        (float(pr.peak_rss_bytes) / (1024.0**3))
                        if pr.peak_rss_bytes is not None
                        else None
                    )
                    peak_uss_gb = (
                        (float(pr.peak_uss_bytes) / (1024.0**3))
                        if pr.peak_uss_bytes is not None
                        else None
                    )

                    row = BenchRow(
                        mode=mode,
                        workload=workload,
                        repeat_idx=int(r),
                        is_warmup=bool(is_warmup),
                        status=status,
                        returncode=int(pr.returncode),
                        wall_sec=float(pr.wall_sec),
                        elapsed_sec=_safe_float(obj.get("elapsed_sec"), pr.wall_sec),
                        init_sec=_safe_float(obj.get("init_sec"), math.nan),
                        query_sec=_safe_float(obj.get("query_sec"), math.nan),
                        peak_rss_gb=peak_rss_gb,
                        peak_uss_gb=peak_uss_gb,
                        rows_requested=(
                            None if "rows_requested" not in obj else _safe_int(obj["rows_requested"])
                        ),
                        rows_kept=(None if "rows_kept" not in obj else _safe_int(obj["rows_kept"])),
                        blocks_returned=(
                            None if "blocks_returned" not in obj else _safe_int(obj["blocks_returned"])
                        ),
                        queries_done=(None if "queries_done" not in obj else _safe_int(obj["queries_done"])),
                        checksum_u64=(
                            None if "checksum_u64" not in obj else _safe_int(obj["checksum_u64"])
                        ),
                        cmd=" ".join(pr.cmd),
                        notes=(
                            str(pr.stderr).strip()[-8000:]
                            if status != "ok"
                            else str(obj.get("n_selected_after_filter", ""))
                        ),
                    )
                    rows.append(row)
                    if status == "ok":
                        qsec = row.query_sec
                        thr = (
                            (float(row.rows_kept or 0) / qsec)
                            if (math.isfinite(qsec) and qsec > 0.0)
                            else math.nan
                        )
                        ttxt = "nan" if not math.isfinite(thr) else f"{thr:,.1f}"
                        print(
                            f" ok wall={_fmt_secs(row.wall_sec)}s "
                            f"query={_fmt_secs(row.query_sec)}s "
                            f"rows={row.rows_kept} rows/s={ttxt} "
                            f"rss={_fmt_gb(row.peak_rss_gb)}GB"
                        )
                    else:
                        print(f" failed rc={pr.returncode}")

    rows_dict = [asdict(r) for r in rows]
    with (run_dir / "results.json").open("w", encoding="utf-8") as fw:
        json.dump(rows_dict, fw, ensure_ascii=False, indent=2)
    _write_csv(run_dir / "results.csv", rows_dict)
    _print_summary(rows)

    print("")
    print(f"Saved: {run_dir / 'plan.json'}")
    print(f"Saved: {run_dir / 'results.csv'}")
    print(f"Saved: {run_dir / 'results.json'}")

    if not bool(args.keep_workdir):
        # keep only lightweight outputs by default
        try:
            if raw_idx_file.exists():
                raw_idx_file.unlink()
            if sample_file.exists():
                sample_file.unlink()
        except Exception:
            pass

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if bool(args.worker):
        return _worker_entry()
    return _driver_entry(args)


if __name__ == "__main__":
    raise SystemExit(main())

