from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np

from janusx import janusx as jxrs
from janusx.gfreader import load_genotype_chunks


ProgressCallback = Optional[Callable[[str, dict[str, Any]], None]]


def _emit_progress(callback: ProgressCallback, event: str, **payload: Any) -> None:
    if callback is None:
        return
    try:
        callback(str(event), payload)
    except Exception:
        # Progress callback should never break numerical training.
        pass


@dataclass
class ADAMixtureConfig:
    genotype_path: str
    k: int
    outdir: str
    prefix: str
    seed: int = 42
    threads: int = 1
    chunk_size: int = 50000
    snps_only: bool = True
    maf: float = 0.02
    geno: float = 0.05
    solver: str = "adam-em"
    tol: float = 1e-5
    cv: int = 0

    lr: float = 0.005
    beta1: float = 0.80
    beta2: float = 0.88
    reg_adam: float = 1e-8
    lr_decay: float = 0.5
    min_lr: float = 1e-6

    max_iter: int = 1500
    check: int = 5
    max_als: int = 1000
    tole_als: float = 1e-4
    reg_als: float = 1e-5
    power: int = 5
    tole_svd: float = 1e-1


def _eval_holdout_nll(
    g_full: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    holdout_flat_idx: np.ndarray,
    *,
    eval_chunk: int = 250000,
) -> tuple[float, float]:
    """
    Evaluate holdout negative log-likelihood for diploid genotype counts.

    Returns
    -------
    mean_nll : float
        Mean negative log-likelihood per held-out genotype.
    deviance : float
        -2 * log-likelihood over held-out genotypes.
    """
    if holdout_flat_idx.size <= 0:
        return float("nan"), float("nan")

    g = np.asarray(g_full, dtype=np.uint8)
    p_mat = np.asarray(p, dtype=np.float64)
    q_mat = np.asarray(q, dtype=np.float64)
    if g.ndim != 2 or p_mat.ndim != 2 or q_mat.ndim != 2:
        raise ValueError("Invalid matrix rank for CV evaluation.")
    if int(g.shape[0]) != int(p_mat.shape[0]):
        raise ValueError(
            f"P row mismatch in CV evaluation: g={g.shape}, p={p_mat.shape}"
        )
    if int(g.shape[1]) != int(q_mat.shape[0]):
        raise ValueError(
            f"Q row mismatch in CV evaluation: g={g.shape}, q={q_mat.shape}"
        )
    if int(p_mat.shape[1]) != int(q_mat.shape[1]):
        raise ValueError(
            f"P/Q K mismatch in CV evaluation: p={p_mat.shape}, q={q_mat.shape}"
        )

    n_samples = int(g.shape[1])
    g_flat = np.ravel(g)
    ll_sum = 0.0
    n_sum = 0
    step = max(10000, int(eval_chunk))
    ln2 = float(np.log(2.0))

    for start in range(0, int(holdout_flat_idx.size), step):
        idx = np.asarray(holdout_flat_idx[start : start + step], dtype=np.int64)
        rows = np.floor_divide(idx, n_samples)
        cols = np.mod(idx, n_samples)

        g_obs = g_flat[idx].astype(np.float64, copy=False)
        pred_p = np.einsum(
            "ij,ij->i",
            p_mat[rows, :],
            q_mat[cols, :],
            optimize=True,
        )
        pred_p = np.clip(pred_p, 1e-8, 1.0 - 1e-8)

        log_comb = np.where(g_obs == 1.0, ln2, 0.0)
        ll = log_comb + g_obs * np.log(pred_p) + (2.0 - g_obs) * np.log1p(-pred_p)
        ll_sum += float(np.sum(ll, dtype=np.float64))
        n_sum += int(idx.size)

    if n_sum <= 0:
        return float("nan"), float("nan")
    mean_nll = float(-ll_sum / float(n_sum))
    deviance = float(-2.0 * ll_sum)
    return mean_nll, deviance


def _cv_assign_fold_ids(
    flat_idx: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    """Deterministic, low-memory fold assignment from flattened genotype indices."""
    if flat_idx.size <= 0:
        return np.empty((0,), dtype=np.int64)
    x = np.asarray(flat_idx, dtype=np.uint64, order="C")
    mix = np.uint64((int(seed) & 0xFFFFFFFFFFFFFFFF) ^ 0x9E3779B97F4A7C15)
    x = x ^ mix
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return np.asarray(x % np.uint64(max(1, int(folds))), dtype=np.int64)


def _cv_scan_observed_and_fold_counts(
    g_flat: np.ndarray,
    *,
    folds: int,
    seed: int,
    scan_chunk: int,
) -> tuple[int, np.ndarray]:
    """Single pass over genotype vector to collect observed count and per-fold counts."""
    total = int(g_flat.size)
    step = max(10000, int(scan_chunk))
    fold_counts = np.zeros((int(folds),), dtype=np.int64)
    n_obs = 0
    for start in range(0, total, step):
        end = min(total, int(start + step))
        block = g_flat[start:end]
        obs_rel = np.flatnonzero(block <= 2)
        if obs_rel.size <= 0:
            continue
        obs_abs = np.asarray(obs_rel, dtype=np.int64) + np.int64(start)
        fold_ids = _cv_assign_fold_ids(
            obs_abs,
            folds=int(folds),
            seed=int(seed),
        )
        fold_counts += np.bincount(fold_ids, minlength=int(folds)).astype(np.int64, copy=False)
        n_obs += int(obs_abs.size)
    return int(n_obs), np.asarray(fold_counts, dtype=np.int64)


def _cv_mark_holdout_missing(
    g_train_flat: np.ndarray,
    g_src_flat: np.ndarray,
    *,
    fold_id: int,
    folds: int,
    seed: int,
    scan_chunk: int,
) -> int:
    """Mark one fold as missing (3) in-place without materializing global holdout index arrays."""
    total = int(g_src_flat.size)
    step = max(10000, int(scan_chunk))
    holdout_n = 0
    for start in range(0, total, step):
        end = min(total, int(start + step))
        block = g_src_flat[start:end]
        obs_rel = np.flatnonzero(block <= 2)
        if obs_rel.size <= 0:
            continue
        obs_abs = np.asarray(obs_rel, dtype=np.int64) + np.int64(start)
        fold_ids = _cv_assign_fold_ids(
            obs_abs,
            folds=int(folds),
            seed=int(seed),
        )
        sel = np.asarray(fold_ids == int(fold_id), dtype=bool)
        if not np.any(sel):
            continue
        holdout_idx = np.asarray(obs_abs[sel], dtype=np.int64)
        g_train_flat[holdout_idx] = np.uint8(3)
        holdout_n += int(holdout_idx.size)
    return int(holdout_n)


def _eval_holdout_nll_fold_stream(
    g_full: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    *,
    fold_id: int,
    folds: int,
    seed: int,
    scan_chunk: int,
    eval_chunk: int,
) -> tuple[float, float, int]:
    """
    Evaluate one CV fold NLL without storing full holdout index vectors.

    Returns
    -------
    mean_nll : float
        Mean negative log-likelihood per held-out genotype.
    deviance : float
        -2 * log-likelihood over held-out genotypes.
    n_holdout : int
        Number of held-out genotypes in this fold.
    """
    g = np.asarray(g_full, dtype=np.uint8)
    p_mat = np.asarray(p, dtype=np.float64)
    q_mat = np.asarray(q, dtype=np.float64)
    if g.ndim != 2 or p_mat.ndim != 2 or q_mat.ndim != 2:
        raise ValueError("Invalid matrix rank for CV evaluation.")
    if int(g.shape[0]) != int(p_mat.shape[0]):
        raise ValueError(
            f"P row mismatch in CV evaluation: g={g.shape}, p={p_mat.shape}"
        )
    if int(g.shape[1]) != int(q_mat.shape[0]):
        raise ValueError(
            f"Q row mismatch in CV evaluation: g={g.shape}, q={q_mat.shape}"
        )
    if int(p_mat.shape[1]) != int(q_mat.shape[1]):
        raise ValueError(
            f"P/Q K mismatch in CV evaluation: p={p_mat.shape}, q={q_mat.shape}"
        )

    n_samples = int(g.shape[1])
    g_flat = np.ravel(g)
    scan_step = max(10000, int(scan_chunk))
    eval_step = max(10000, int(eval_chunk))
    ln2 = float(np.log(2.0))
    ll_sum = 0.0
    n_sum = 0

    total = int(g_flat.size)
    for start in range(0, total, scan_step):
        end = min(total, int(start + scan_step))
        block = g_flat[start:end]
        obs_rel = np.flatnonzero(block <= 2)
        if obs_rel.size <= 0:
            continue
        obs_abs = np.asarray(obs_rel, dtype=np.int64) + np.int64(start)
        fold_ids = _cv_assign_fold_ids(
            obs_abs,
            folds=int(folds),
            seed=int(seed),
        )
        holdout = np.asarray(obs_abs[fold_ids == int(fold_id)], dtype=np.int64)
        if holdout.size <= 0:
            continue
        for j in range(0, int(holdout.size), eval_step):
            idx = np.asarray(holdout[j : j + eval_step], dtype=np.int64)
            rows = np.floor_divide(idx, n_samples)
            cols = np.mod(idx, n_samples)
            g_obs = g_flat[idx].astype(np.float64, copy=False)
            pred_p = np.einsum(
                "ij,ij->i",
                p_mat[rows, :],
                q_mat[cols, :],
                optimize=True,
            )
            pred_p = np.clip(pred_p, 1e-8, 1.0 - 1e-8)
            log_comb = np.where(g_obs == 1.0, ln2, 0.0)
            ll = log_comb + g_obs * np.log(pred_p) + (2.0 - g_obs) * np.log1p(-pred_p)
            ll_sum += float(np.sum(ll, dtype=np.float64))
            n_sum += int(idx.size)

    if n_sum <= 0:
        return float("nan"), float("nan"), 0
    mean_nll = float(-ll_sum / float(n_sum))
    deviance = float(-2.0 * ll_sum)
    return mean_nll, deviance, int(n_sum)

def _resolve_solver_mode(solver: str) -> str:
    x = str(solver).strip().lower()
    if x in {"auto", ""}:
        return "adam-em"
    if x in {"adam", "adam-em"}:
        return x
    return "adam-em"


def _adam_seed_init(
    m: int,
    n: int,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    p = rng.random(size=(int(m), int(k)), dtype=np.float32)
    p = np.ascontiguousarray(np.clip(p, 1e-5, 1.0 - 1e-5), dtype=np.float32)
    q0 = rng.random(size=(int(n), int(k)), dtype=np.float32)
    q = np.ascontiguousarray(jxrs.admx_map_q_f32(np.ascontiguousarray(q0, dtype=np.float32)))
    return p, q


def _set_thread_env(threads: int) -> None:
    # Default BLAS thread cap follows CLI/runtime threads for consistency.
    # Explicit override remains available via JANUSX_ADMX_BLAS_THREADS.
    default_th = max(1, int(threads))
    try:
        env_th = int(str(os.environ.get("JANUSX_ADMX_BLAS_THREADS", "")).strip())
    except Exception:
        env_th = 0
    th = str(max(1, int(env_th if env_th > 0 else default_th)))
    os.environ["JX_THREADS"] = th
    os.environ["JX_MLM_BLAS_THREADS"] = th
    os.environ["RAYON_NUM_THREADS"] = str(max(1, int(threads)))
    os.environ["MKL_NUM_THREADS"] = th
    os.environ["MKL_MAX_THREADS"] = th
    os.environ["OMP_NUM_THREADS"] = th
    os.environ["OMP_MAX_THREADS"] = th
    os.environ["NUMEXPR_NUM_THREADS"] = th
    os.environ["NUMEXPR_MAX_THREADS"] = th
    os.environ["OPENBLAS_NUM_THREADS"] = th
    os.environ["OPENBLAS_MAX_THREADS"] = th


def _to_u8_missing3(block: np.ndarray) -> np.ndarray:
    x = np.asarray(block, dtype=np.float32)
    miss = ~np.isfinite(x) | (x < 0)
    out = np.rint(np.clip(x, 0.0, 2.0)).astype(np.uint8, copy=False)
    if np.any(miss):
        out = np.array(out, copy=True)
        out[miss] = 3
    return np.ascontiguousarray(out)


def _iter_genotype_chunks(
    genotype_path: str,
    *,
    chunk_size: int,
    snps_only: bool,
    maf: float,
    missing_rate: float,
):
    yield from load_genotype_chunks(
        genotype_path,
        chunk_size=int(chunk_size),
        maf=float(maf),
        missing_rate=float(missing_rate),
        impute=False,
        model="add",
        snps_only=bool(snps_only),
    )


def _adaptive_chunk_size(requested: int, n_samples: int) -> int:
    req = max(1000, int(requested))
    ns = max(1, int(n_samples))
    # raw loader chunk is float32; cap raw chunk buffer around this budget.
    budget_mb = int(os.environ.get("JANUSX_ADMX_LOAD_MB", "32"))
    budget_bytes = max(8, budget_mb) * 1024 * 1024
    rows_by_budget = max(1000, int(budget_bytes // (ns * 4)))
    return int(max(1000, min(req, rows_by_budget)))


def rsvd_streaming(
    genotype_path: str,
    *,
    k: int,
    seed: int = 42,
    power: int = 5,
    tol: float = 1e-1,
    snps_only: bool = True,
    maf: float = 0.02,
    missing_rate: float = 0.05,
    delimiter: Optional[str] = None,
    mmap_window_mb: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Streaming RSVD fully in Rust.

    Returns
    -------
    eigvals : np.ndarray, shape (k_eff,), float32
        Top-k eigenvalues (from singular values^2).

    eigvecs : np.ndarray, shape (m_filtered, k_eff), float32
        Left singular vectors/eigenvectors corresponding to eigvals.
    """
    evals, evecs = jxrs.admx_rsvd_stream(
        str(genotype_path),
        int(k),
        int(seed),
        int(power),
        float(tol),
        bool(snps_only),
        float(maf),
        float(missing_rate),
        (None if delimiter is None else str(delimiter)),
        int(mmap_window_mb),
    )
    return (
        np.ascontiguousarray(np.asarray(evals, dtype=np.float32)),
        np.ascontiguousarray(np.asarray(evecs, dtype=np.float32)),
    )


def _plink_prefix_path(genotype_path: str) -> Optional[Path]:
    p = Path(genotype_path)
    if p.suffix.lower() in {".bed", ".bim", ".fam"}:
        p = p.with_suffix("")
    if p.with_suffix(".bed").exists() and p.with_suffix(".bim").exists():
        return p
    return None


def _count_bim_rows(prefix: Path) -> Optional[int]:
    bim = prefix.with_suffix(".bim")
    if not bim.exists():
        return None
    try:
        with bim.open("rb") as fh:
            return int(sum(1 for _ in fh))
    except Exception:
        return None


def load_genotype_u8_matrix(
    genotype_path: str,
    *,
    chunk_size: int,
    snps_only: bool,
    maf: float,
    missing_rate: float,
) -> np.ndarray:
    prefix = _plink_prefix_path(genotype_path)
    if prefix is not None:
        g = np.ascontiguousarray(
            jxrs.load_bed_u8_matrix(
                str(prefix),
                float(maf),
                float(missing_rate),
                False,
            ),
            dtype=np.uint8,
        )
        if g.ndim != 2 or g.shape[0] <= 0 or g.shape[1] <= 0:
            raise ValueError("No SNPs were loaded from PLINK input.")
        return g

    req_chunk = max(1000, int(chunk_size))
    n_samples: Optional[int] = None

    # Probe first chunk to infer sample size, then adapt chunk size to lower peak RSS.
    probe_iter = _iter_genotype_chunks(
        genotype_path,
        chunk_size=req_chunk,
        snps_only=snps_only,
        maf=maf,
        missing_rate=missing_rate,
    )
    try:
        first_probe, _ = next(probe_iter)
    except StopIteration as exc:
        raise ValueError("No SNPs were loaded from genotype input.") from exc
    if first_probe.ndim != 2:
        raise ValueError(f"Invalid genotype chunk shape: {first_probe.shape}")
    n_samples = int(first_probe.shape[1])
    use_chunk = _adaptive_chunk_size(req_chunk, n_samples)

    # Fast low-memory path for PLINK: use .bim row count as allocation hint (single parse).
    total_rows_hint = _count_bim_rows(prefix) if prefix is not None else None
    if total_rows_hint is not None and total_rows_hint > 0:
        out = np.empty((int(total_rows_hint), int(n_samples)), dtype=np.uint8)
        offset = 0
        for geno, _sites in _iter_genotype_chunks(
            genotype_path,
            chunk_size=use_chunk,
            snps_only=snps_only,
            maf=maf,
            missing_rate=missing_rate,
        ):
            if geno.ndim != 2 or int(geno.shape[1]) != int(n_samples):
                raise ValueError(
                    f"Inconsistent genotype chunk shape: got {geno.shape}, expected (?, {n_samples})"
                )
            g = _to_u8_missing3(geno)
            rows = int(g.shape[0])
            need = offset + rows
            if need > out.shape[0]:
                # If upstream loader yields more rows than bim hint, grow safely.
                grow = max(need, int(out.shape[0] * 1.5))
                buf = np.empty((int(grow), int(n_samples)), dtype=np.uint8)
                if offset > 0:
                    buf[:offset, :] = out[:offset, :]
                out = buf
            out[offset:need, :] = g
            offset = need
        if offset <= 0:
            raise ValueError("No SNPs were loaded from genotype input.")
        return np.ascontiguousarray(out[:offset, :], dtype=np.uint8)

    # Generic fallback: two-pass low-memory allocation.
    total_rows = 0
    if use_chunk == req_chunk:
        total_rows += int(first_probe.shape[0])
        for geno, _sites in probe_iter:
            if geno.ndim != 2:
                raise ValueError(f"Invalid genotype chunk shape: {geno.shape}")
            if int(geno.shape[1]) != int(n_samples):
                raise ValueError(
                    f"Inconsistent sample count across chunks: expected {n_samples}, got {geno.shape[1]}"
                )
            total_rows += int(geno.shape[0])
    else:
        for geno, _sites in _iter_genotype_chunks(
            genotype_path,
            chunk_size=use_chunk,
            snps_only=snps_only,
            maf=maf,
            missing_rate=missing_rate,
        ):
            if geno.ndim != 2:
                raise ValueError(f"Invalid genotype chunk shape: {geno.shape}")
            if int(geno.shape[1]) != int(n_samples):
                raise ValueError(
                    f"Inconsistent sample count across chunks: expected {n_samples}, got {geno.shape[1]}"
                )
            total_rows += int(geno.shape[0])

    if total_rows <= 0:
        raise ValueError("No SNPs were loaded from genotype input.")

    out = np.empty((int(total_rows), int(n_samples)), dtype=np.uint8)
    offset = 0
    for geno, _sites in _iter_genotype_chunks(
        genotype_path,
        chunk_size=use_chunk,
        snps_only=snps_only,
        maf=maf,
        missing_rate=missing_rate,
    ):
        g = _to_u8_missing3(geno)
        rows = int(g.shape[0])
        out[offset : offset + rows, :] = g
        offset += rows
    if offset != total_rows:
        out = out[:offset, :]
    return np.ascontiguousarray(out, dtype=np.uint8)


def _eig_svd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d, v = np.linalg.eigh(x.T @ x)
    d = np.clip(d, 1e-12, None)
    s = np.sqrt(d)
    u = x @ (v * (1.0 / s))
    return (
        np.ascontiguousarray(u[:, ::-1], dtype=np.float32),
        np.ascontiguousarray(s[::-1], dtype=np.float32),
        np.ascontiguousarray(v[:, ::-1], dtype=np.float32),
    )


def _rsvd(
    g: np.ndarray,
    f: np.ndarray,
    *,
    k: int,
    seed: int,
    power: int,
    tol: float,
    logger: logging.Logger,
    callback: ProgressCallback = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, n = g.shape  # SNP x sample
    t0 = time.time()
    rng = np.random.default_rng(int(seed))
    k_prime = max(int(k) + 10, 20)
    alpha = 0.0
    rsvd_tile = max(1, int(os.environ.get("JANUSX_ADMX_RSVD_TILE", "1024")))

    _emit_progress(callback, "rsvd_start", power=int(power))
    logger.info(f"{'RSVD':<22}: start")
    omega = np.ascontiguousarray(
        rng.standard_normal(size=(m, k_prime), dtype=np.float32),
        dtype=np.float32,
    )
    y = np.empty((n, k_prime), dtype=np.float32)
    g_small = np.empty((m, k_prime), dtype=np.float32)
    jxrs.admx_multiply_at_omega_inplace(g, omega, f, y, rsvd_tile)
    q, _, _ = _eig_svd(y)

    sk = np.zeros(k_prime, dtype=np.float32)
    s_idx = 0
    converged_iter: Optional[int] = None
    for i in range(int(power)):
        jxrs.admx_rsvd_power_step_inplace(g, q, f, y, g_small, rsvd_tile)
        y -= alpha * q
        q, s_y, _ = _eig_svd(y)

        if i > 0:
            sk_now = s_y + alpha
            denom = np.maximum(sk_now, 1e-12)
            pve_all = np.abs(sk_now - sk[: len(sk_now)]) / denom
            ei = float(np.max(pve_all[s_idx : int(k) + s_idx])) if len(pve_all) > 0 else 0.0
            if ei < float(tol):
                converged_iter = int(i + 1)
                logger.info(
                    f"{'RSVD converged':<22}: iter {i + 1}/{int(power)}"
                )
                break
            sk[: len(sk_now)] = sk_now
        else:
            sk[: len(s_y)] = s_y + alpha

        if float(alpha) < float(s_y[-1]):
            alpha = float(alpha + s_y[-1]) / 2.0

    jxrs.admx_multiply_a_omega_inplace(g, q, f, g_small)
    u_small, s_all, v_small = _eig_svd(g_small)
    s = np.ascontiguousarray(s_all[: int(k)], dtype=np.float32)
    u = np.ascontiguousarray(u_small[:, : int(k)], dtype=np.float32)
    v = np.ascontiguousarray(q @ v_small[:, : int(k)], dtype=np.float32)
    elapsed = time.time() - t0
    logger.info(f"{'RSVD finished':<22}: {elapsed:.2f}s")
    _emit_progress(
        callback,
        "rsvd_done",
        elapsed=float(elapsed),
        converged_iter=converged_iter,
        max_power=int(power),
    )
    return u, s, v


def _als_init(
    g: np.ndarray,
    u: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
    *,
    seed: int,
    k: int,
    max_iter: int,
    tol: float,
    reg: float,
    logger: logging.Logger,
    callback: ProgressCallback = None,
) -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    m, n = g.shape
    rng = np.random.default_rng(int(seed))
    z = np.ascontiguousarray(u * s, dtype=np.float32)
    reg_eye = np.ascontiguousarray(reg * np.eye(int(k), dtype=np.float32), dtype=np.float32)

    p = rng.random(size=(m, int(k)), dtype=np.float32)
    p = np.ascontiguousarray(np.clip(p, 1e-5, 1.0 - 1e-5), dtype=np.float32)
    i_mat = np.ascontiguousarray(p @ np.linalg.pinv(p.T @ p + reg_eye), dtype=np.float32)

    q = 0.5 * (v @ (z.T @ i_mat)) + (i_mat * f[:, None]).sum(axis=0)
    q = np.ascontiguousarray(jxrs.admx_map_q_f32(np.ascontiguousarray(q, dtype=np.float32)))
    q_prev = np.array(q, copy=True)

    rmse_best = np.inf
    stall_counter = 0
    high_corr = False
    p_best = np.array(p, copy=True)
    q_best = np.array(q, copy=True)
    _emit_progress(callback, "als_start", max_iter=int(max_iter))

    converged_iter: Optional[int] = None
    for it in range(int(max_iter)):
        i_mat = np.ascontiguousarray(q @ np.linalg.pinv(q.T @ q + reg_eye), dtype=np.float32)
        p = 0.5 * (z @ (v.T @ i_mat)) + np.outer(f, i_mat.sum(axis=0))
        p = np.ascontiguousarray(jxrs.admx_map_p_f32(np.ascontiguousarray(p, dtype=np.float32)))

        g_p = np.ascontiguousarray(p.T @ p, dtype=np.float32)
        i_mat = np.ascontiguousarray(p @ np.linalg.pinv(g_p + reg_eye), dtype=np.float32)
        q = 0.5 * (v @ (z.T @ i_mat)) + (i_mat * f[:, None]).sum(axis=0)
        q = np.ascontiguousarray(jxrs.admx_map_q_f32(np.ascontiguousarray(q, dtype=np.float32)))

        rmse_err = float(jxrs.admx_rmse_f32(q, q_prev))
        if not high_corr:
            vv = np.sqrt(np.clip(np.diag(g_p), 1e-12, None))
            denom = np.outer(vv, vv)
            denom[denom == 0] = 1e-10
            corr = g_p / denom
            np.fill_diagonal(corr, 0.0)
            max_corr = float(np.max(np.abs(corr))) if corr.size > 0 else 0.0
            if max_corr > 0.95:
                high_corr = True

        if high_corr:
            if rmse_err < rmse_best:
                rmse_best = rmse_err
                p_best = np.array(p, copy=True)
                q_best = np.array(q, copy=True)
                stall_counter = 0
            else:
                stall_counter += 1
            if stall_counter >= 20:
                converged_iter = int(it + 1)
                logger.info(
                    f"{'ALS stall limit':<22}: iter {it + 1}/{int(max_iter)}, rollback best state"
                )
                p = p_best
                q = q_best
                break

        if rmse_err < float(tol):
            converged_iter = int(it + 1)
            logger.info(
                f"{'ALS converged':<22}: iter {it + 1}/{int(max_iter)}"
            )
            break
        q_prev[...] = q

    elapsed = time.time() - t0
    logger.info(f"{'ALS finished':<22}: {elapsed:.2f}s")
    p = np.ascontiguousarray(p, dtype=np.float32)
    q = np.ascontiguousarray(q, dtype=np.float32)
    ll0 = float(jxrs.admx_loglikelihood_f32(g, p, q))
    logger.info(f"{'Initial log-likelihood':<22}: {ll0:.3f}")
    _emit_progress(
        callback,
        "als_done",
        elapsed=float(elapsed),
        initial_ll=float(ll0),
        converged_iter=converged_iter,
        max_iter=int(max_iter),
    )
    return p, q


def _adam_em_optimize(
    g: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    reg_adam: float,
    max_iter: int,
    check: int,
    lr_decay: float,
    min_lr: float,
    logger: logging.Logger,
    callback: ProgressCallback = None,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.ascontiguousarray(p, dtype=np.float32)
    q = np.ascontiguousarray(q, dtype=np.float32)
    t0 = time.time()
    check_every = max(1, int(check))
    # 0 means auto (full-width fast path); set JANUSX_ADMX_TILE to a positive value
    # to force tiled Q-accumulation when you want lower scratch memory.
    tile_cfg = int(os.environ.get("JANUSX_ADMX_TILE", "0"))
    tile_cols = int(g.shape[1]) if tile_cfg <= 0 else max(1, tile_cfg)
    _emit_progress(
        callback,
        "adam_start",
        max_iter=int(max_iter),
        check_every=int(check_every),
    )
    p_best, q_best, ll_final, n_iter = jxrs.admx_adam_optimize_f32(
        g,
        p,
        q,
        float(lr),
        float(beta1),
        float(beta2),
        float(reg_adam),
        int(max_iter),
        int(check_every),
        float(lr_decay),
        float(min_lr),
        int(tile_cols),
    )
    logger.info(
        f"{'ADAM optimize':<22}: iter={int(n_iter):>4}/{int(max_iter):<4} "
        f"ll={float(ll_final):>14.3f} tile={int(tile_cols)}"
    )
    elapsed = time.time() - t0
    logger.info(
        f"{'Final log-likelihood':<22}: {float(ll_final):.3f} (elapsed {elapsed:.2f}s)"
    )
    _emit_progress(
        callback,
        "adam_done",
        elapsed=float(elapsed),
        final_ll=float(ll_final),
    )
    # Keep external API/output compatibility with previous float64 behavior.
    return (
        np.ascontiguousarray(p_best, dtype=np.float64),
        np.ascontiguousarray(q_best, dtype=np.float64),
    )


def _fit_adamixture_on_u8(
    g: np.ndarray,
    cfg: ADAMixtureConfig,
    logger: logging.Logger,
    callback: ProgressCallback = None,
    *,
    log_data_loaded: bool = True,
    init_p: Optional[np.ndarray] = None,
    init_q: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    m, n = g.shape
    if log_data_loaded:
        logger.info(f"{'Data loaded':<22}: SNPs={m}, samples={n}")
    _emit_progress(callback, "data_loaded", m=int(m), n=int(n))
    solver_mode = _resolve_solver_mode(cfg.solver)
    tol = float(max(1e-12, float(cfg.tol)))
    warm_start_ok = (
        (init_p is not None)
        and (init_q is not None)
    )
    if warm_start_ok:
        p0 = np.ascontiguousarray(np.asarray(init_p, dtype=np.float32))
        q0_raw = np.ascontiguousarray(np.asarray(init_q, dtype=np.float32))
        if p0.shape != (int(m), int(cfg.k)):
            raise ValueError(
                f"Warm-start P shape mismatch: got {p0.shape}, expected {(int(m), int(cfg.k))}."
            )
        if q0_raw.shape != (int(n), int(cfg.k)):
            raise ValueError(
                f"Warm-start Q shape mismatch: got {q0_raw.shape}, expected {(int(n), int(cfg.k))}."
            )
        p0 = np.ascontiguousarray(np.clip(p0, 1e-5, 1.0 - 1e-5), dtype=np.float32)
        q0 = np.ascontiguousarray(
            jxrs.admx_map_q_f32(np.ascontiguousarray(q0_raw, dtype=np.float32)),
            dtype=np.float32,
        )
        logger.info(f"{'Warm-start':<22}: enabled (reuse previous K)")
    else:
        f = np.ascontiguousarray(jxrs.admx_allele_frequency(g), dtype=np.float32)
        u, s, v = _rsvd(
            g,
            f,
            k=int(cfg.k),
            seed=int(cfg.seed),
            power=int(cfg.power),
            tol=tol,
            logger=logger,
            callback=callback,
        )
        if solver_mode == "adam-em":
            p0, q0 = _als_init(
                g,
                u,
                s,
                v,
                f,
                seed=int(cfg.seed),
                k=int(cfg.k),
                max_iter=int(cfg.max_als),
                tol=tol,
                reg=float(cfg.reg_als),
                logger=logger,
                callback=callback,
            )
        else:
            logger.info(f"{'Solver':<22}: ADAM (skip ALS)")
            p0, q0 = _adam_seed_init(m, n, int(cfg.k), int(cfg.seed))
    p, q = _adam_em_optimize(
        g,
        p0,
        q0,
        lr=float(cfg.lr),
        beta1=float(cfg.beta1),
        beta2=float(cfg.beta2),
        reg_adam=float(cfg.reg_adam),
        max_iter=int(cfg.max_iter),
        check=int(cfg.check),
        lr_decay=float(cfg.lr_decay),
        min_lr=float(cfg.min_lr),
        logger=logger,
        callback=callback,
    )
    return p, q, m, n


def evaluate_adamixture_cverror(
    cfg: ADAMixtureConfig,
    logger: logging.Logger,
    progress_callback: ProgressCallback = None,
    g_full: Optional[np.ndarray] = None,
) -> Optional[dict[str, float]]:
    folds = int(max(0, int(cfg.cv)))
    if folds <= 0:
        logger.info(f"{'CVerror':<22}: disabled")
        return None
    if folds < 2:
        raise ValueError(
            f"CVerror requires folds >= 2, got {folds}. "
            "Use -cv N with N>=2."
        )

    _set_thread_env(cfg.threads)
    try:
        jxrs.admx_set_threads(int(max(1, int(cfg.threads))))
    except Exception:
        pass

    if g_full is None:
        g_full = load_genotype_u8_matrix(
            cfg.genotype_path,
            chunk_size=int(cfg.chunk_size),
            snps_only=bool(cfg.snps_only),
            maf=float(cfg.maf),
            missing_rate=float(cfg.geno),
        )
    else:
        g_full = np.ascontiguousarray(np.asarray(g_full, dtype=np.uint8))
    m, n = g_full.shape
    g_flat = np.ravel(g_full)
    scan_chunk = max(10000, int(os.environ.get("JANUSX_ADMX_CV_SCAN_CHUNK", "1000000")))
    n_obs, fold_counts = _cv_scan_observed_and_fold_counts(
        g_flat,
        folds=int(folds),
        seed=int(cfg.seed),
        scan_chunk=int(scan_chunk),
    )
    if n_obs < int(folds):
        raise ValueError(
            f"Not enough observed genotypes for CVerror: observed={n_obs}, folds={folds}."
        )
    if np.any(fold_counts <= 0):
        bad = [str(i + 1) for i in np.flatnonzero(fold_counts <= 0)]
        raise ValueError(
            "Failed to build non-empty CV folds in low-memory assignment; "
            f"empty folds={','.join(bad)}."
        )
    logger.info(
        f"{'CVerror':<22}: estimating from {int(folds)} folds "
        f"(observed={n_obs}, SNPs={m}, samples={n}, scan_chunk={int(scan_chunk)})"
    )
    logger.info(
        f"{'CV fold sizes':<22}: "
        + ", ".join(f"F{i+1}={int(c)}" for i, c in enumerate(fold_counts.tolist()))
    )
    _emit_progress(
        progress_callback,
        "cv_start",
        folds=int(folds),
        mode="k-fold",
    )
    t0 = time.time()
    fold_mean_nll: list[float] = []
    fold_obs_counts: list[int] = []
    eval_chunk = max(10000, int(os.environ.get("JANUSX_ADMX_CV_EVAL_CHUNK", "250000")))

    for i in range(int(folds)):
        fold_id = int(i)
        g_train = np.array(g_full, copy=True, dtype=np.uint8, order="C")
        holdout_n = _cv_mark_holdout_missing(
            np.ravel(g_train),
            g_flat,
            fold_id=int(fold_id),
            folds=int(folds),
            seed=int(cfg.seed),
            scan_chunk=int(scan_chunk),
        )
        if holdout_n <= 0:
            raise ValueError(f"Fold {fold_id + 1} has no held-out genotypes.")

        cfg_fold = replace(cfg, seed=int(cfg.seed) + int(i + 1))
        p_fold, q_fold, _m, _n = _fit_adamixture_on_u8(
            g_train,
            cfg_fold,
            logger,
            callback=None,
            log_data_loaded=False,
        )
        mean_nll, _deviance, n_fold_eval = _eval_holdout_nll_fold_stream(
            g_full,
            p_fold,
            q_fold,
            fold_id=int(fold_id),
            folds=int(folds),
            seed=int(cfg.seed),
            scan_chunk=int(scan_chunk),
            eval_chunk=int(eval_chunk),
        )
        fold_mean_nll.append(float(mean_nll))
        fold_obs_counts.append(int(n_fold_eval))
        _emit_progress(
            progress_callback,
            "cv_fold_done",
            fold=int(i + 1),
            folds=int(folds),
            mean_nll=float(mean_nll),
            holdout_n=int(n_fold_eval),
        )

    mean_arr = np.asarray(fold_mean_nll, dtype=np.float64)
    cverror = float(np.mean(mean_arr))
    cverror_se = float(np.std(mean_arr, ddof=1) / np.sqrt(mean_arr.size)) if mean_arr.size > 1 else 0.0
    elapsed = float(time.time() - t0)
    logger.info(
        f"{'CVerror':<22}: {cverror:.6f} "
        f"(SE={cverror_se:.6f}, folds={int(mean_arr.size)}, elapsed {elapsed:.2f}s)"
    )
    logger.info(
        f"{'CV eval counts':<22}: "
        + ", ".join(f"F{i+1}={int(c)}" for i, c in enumerate(fold_obs_counts))
    )
    _emit_progress(
        progress_callback,
        "cv_done",
        folds=int(mean_arr.size),
        cverror=float(cverror),
        cverror_se=float(cverror_se),
    )
    return {
        "cverror": float(cverror),
        "cverror_se": float(cverror_se),
        "cv_folds": float(mean_arr.size),
        "cv_observed": float(n_obs),
    }


def train_adamixture(
    cfg: ADAMixtureConfig,
    logger: logging.Logger,
    callback: ProgressCallback = None,
    g_matrix: Optional[np.ndarray] = None,
    warm_start: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    _set_thread_env(cfg.threads)
    try:
        jxrs.admx_set_threads(int(max(1, int(cfg.threads))))
    except Exception:
        # Backward-compatible fallback when Rust extension doesn't expose thread setter.
        pass
    np.random.seed(int(cfg.seed))

    if g_matrix is None:
        g = load_genotype_u8_matrix(
            cfg.genotype_path,
            chunk_size=int(cfg.chunk_size),
            snps_only=bool(cfg.snps_only),
            maf=float(cfg.maf),
            missing_rate=float(cfg.geno),
        )
    else:
        g = np.ascontiguousarray(np.asarray(g_matrix, dtype=np.uint8))
    init_p: Optional[np.ndarray] = None
    init_q: Optional[np.ndarray] = None
    if warm_start is not None:
        try:
            init_p, init_q = warm_start
        except Exception as ex:
            raise ValueError("warm_start must be a tuple (P, Q).") from ex
    return _fit_adamixture_on_u8(
        g,
        cfg,
        logger,
        callback=callback,
        log_data_loaded=True,
        init_p=init_p,
        init_q=init_q,
    )
