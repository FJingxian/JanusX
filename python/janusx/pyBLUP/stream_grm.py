from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional, Tuple

import numpy as np
import psutil
import time

try:
    from scipy.linalg.blas import ssyrk as _ssyrk  # type: ignore
    _HAS_SSYRK = True
except Exception:
    _ssyrk = None
    _HAS_SSYRK = False


@dataclass
class StreamingGrmStats:
    eff_m: int
    varsum: float
    used_syrk: bool


def auto_stream_grm_chunk_size(
    n_samples: int,
    requested_chunk_size: int,
    *,
    threads: int = 1,
    prefetch_depth: int = 2,
    min_chunk_size: int = 5_000,
) -> int:
    """
    Auto-tune GRM chunk size to reduce peak memory for large sample sizes.

    The tuned value is always <= requested_chunk_size.
    """
    n = int(max(1, n_samples))
    req = int(max(1, requested_chunk_size))
    t = int(max(1, threads))
    depth = int(max(1, prefetch_depth))
    min_chunk = int(max(1, min_chunk_size))

    bytes_per_row = n * np.dtype(np.float32).itemsize

    try:
        avail = int(psutil.virtual_memory().available)
    except Exception:
        avail = 8 * 1024**3

    target_bytes = int(min(max(avail * 0.20, 256 * 1024**2), 2 * 1024**3))
    in_flight_buffers = depth + 2
    thread_penalty = 1.0 + 0.08 * max(0, t - 1)
    budget_bytes = max(64 * 1024**2, int(target_bytes / (in_flight_buffers * thread_penalty)))

    auto_chunk = int(max(1, budget_bytes // max(1, bytes_per_row)))
    auto_chunk = min(req, auto_chunk)
    if auto_chunk >= 1000:
        auto_chunk = (auto_chunk // 1000) * 1000
    auto_chunk = max(min_chunk, auto_chunk)
    return int(min(req, auto_chunk))


def _syrk_accumulate_upper(grm_upper: np.ndarray, geno: np.ndarray) -> bool:
    if _HAS_SSYRK:
        out = _ssyrk(
            alpha=1.0,
            a=geno,
            beta=1.0,
            c=grm_upper,
            trans=1,   # A^T A where A=(m_chunk, n_samples)
            lower=0,   # update upper triangle only
            overwrite_c=1,
        )
        if out is not grm_upper:
            grm_upper[...] = out
        return True
    grm_upper += geno.T @ geno
    return False


def _gemm_accumulate_upper(grm_upper: np.ndarray, geno: np.ndarray) -> bool:
    grm_upper += geno.T @ geno
    return False


def _select_accumulator(
    probe: np.ndarray,
    *,
    mode: Literal["auto", "syrk", "gemm"],
) -> Callable[[np.ndarray, np.ndarray], bool]:
    if mode == "gemm":
        return _gemm_accumulate_upper
    if mode == "syrk":
        return _syrk_accumulate_upper
    if not _HAS_SSYRK:
        return _gemm_accumulate_upper

    n = int(probe.shape[1])
    c0 = np.zeros((n, n), dtype=np.float32)
    c1 = np.zeros((n, n), dtype=np.float32)

    t0 = time.perf_counter()
    _gemm_accumulate_upper(c0, probe)
    gemm_ticks = time.perf_counter() - t0

    t0 = time.perf_counter()
    _syrk_accumulate_upper(c1, probe)
    syrk_ticks = time.perf_counter() - t0

    if syrk_ticks > 0 and syrk_ticks < gemm_ticks:
        return _syrk_accumulate_upper
    return _gemm_accumulate_upper


def build_streaming_grm_from_chunks(
    chunk_iter: Iterable[Tuple[np.ndarray, object]],
    *,
    n_samples: int,
    method: int = 1,
    eps: float = 1e-8,
    accumulate: Literal["auto", "syrk", "gemm"] = "auto",
    on_chunk: Optional[Callable[[int, int], None]] = None,
) -> tuple[np.ndarray, StreamingGrmStats]:
    """
    Build GRM from SNP-major genotype chunks.

    Parameters
    ----------
    chunk_iter:
        Iterable yielding (genotype_chunk, sites), where genotype_chunk shape is
        (m_chunk, n_samples), dtype float-like.
    n_samples:
        Number of samples.
    method:
        1=centered GRM, 2=standardized GRM.
    eps:
        Numerical floor for standardized method denominator.
    accumulate:
        Kernel backend: "gemm" (A.T@A), "syrk" (BLAS SYRK), or "auto" to
        choose the faster one from a small probe.
    on_chunk:
        Optional callback called as on_chunk(added_snps, total_effective_snps).
    """
    n = int(max(1, n_samples))
    if method not in (1, 2):
        raise ValueError(f"Unsupported GRM method: {method}")

    grm = np.zeros((n, n), dtype=np.float32)
    varsum = 0.0
    eff_m = 0
    used_syrk = False
    eps32 = np.float32(max(float(eps), 1e-12))
    accumulate_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None

    for genosub, _sites in chunk_iter:
        if genosub is None:
            continue
        geno = np.asarray(genosub, dtype=np.float32, order="C")
        m_chunk = int(geno.shape[0])
        if m_chunk <= 0:
            continue

        p = geno.mean(axis=1, dtype=np.float32, keepdims=True) * np.float32(0.5)
        np.subtract(geno, np.float32(2.0) * p, out=geno)

        if method == 1:
            varsum += float(np.sum(np.float32(2.0) * p * (np.float32(1.0) - p), dtype=np.float64))
        else:
            denom = np.float32(2.0) * p * (np.float32(1.0) - p)
            np.maximum(denom, eps32, out=denom)
            np.divide(np.float32(1.0), denom, out=denom)
            np.sqrt(denom, out=denom)
            geno *= denom

        if accumulate_fn is None:
            probe_rows = min(int(geno.shape[0]), 4096)
            probe = np.ascontiguousarray(geno[:probe_rows], dtype=np.float32)
            accumulate_fn = _select_accumulator(probe, mode=accumulate)
        used_syrk = accumulate_fn(grm, geno) or used_syrk
        eff_m += m_chunk
        if on_chunk is not None:
            on_chunk(m_chunk, eff_m)

    if eff_m <= 0:
        raise ValueError("No effective SNPs remained after GRM filtering.")

    if method == 1:
        if varsum <= 0:
            raise ValueError("Invalid GRM denominator (varsum<=0).")
        scale = np.float32(varsum)
    else:
        scale = np.float32(eff_m)

    grm = np.triu(grm)
    grm += np.triu(grm, 1).T
    grm /= scale

    return grm, StreamingGrmStats(
        eff_m=eff_m,
        varsum=varsum,
        used_syrk=used_syrk,
    )
