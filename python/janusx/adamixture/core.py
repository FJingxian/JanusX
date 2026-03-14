from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
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


def _set_thread_env(threads: int) -> None:
    th = str(max(1, int(threads)))
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


def load_genotype_u8_matrix(
    genotype_path: str,
    *,
    chunk_size: int,
    snps_only: bool,
    maf: float,
    missing_rate: float,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n_samples = None
    for geno, _sites in load_genotype_chunks(
        genotype_path,
        chunk_size=int(chunk_size),
        maf=float(maf),
        missing_rate=float(missing_rate),
        impute=False,
        model="add",
        snps_only=bool(snps_only),
    ):
        g = _to_u8_missing3(geno)
        if g.ndim != 2:
            raise ValueError(f"Invalid genotype chunk shape: {g.shape}")
        if n_samples is None:
            n_samples = int(g.shape[1])
        elif int(g.shape[1]) != int(n_samples):
            raise ValueError(
                f"Inconsistent sample count across chunks: expected {n_samples}, got {g.shape[1]}"
            )
        chunks.append(g)
    if len(chunks) == 0:
        raise ValueError("No SNPs were loaded from genotype input.")
    if len(chunks) == 1:
        return np.ascontiguousarray(chunks[0], dtype=np.uint8)
    return np.ascontiguousarray(np.vstack(chunks), dtype=np.uint8)


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

    _emit_progress(callback, "rsvd_start", power=int(power))
    logger.info(f"{'RSVD':<22}: start")
    omega = np.ascontiguousarray(
        rng.standard_normal(size=(m, k_prime), dtype=np.float32),
        dtype=np.float32,
    )
    y = np.ascontiguousarray(jxrs.admx_multiply_at_omega(g, omega, f), dtype=np.float32)
    q, _, _ = _eig_svd(y)

    sk = np.zeros(k_prime, dtype=np.float32)
    s_idx = 0
    converged_iter: Optional[int] = None
    for i in range(int(power)):
        g_small = np.ascontiguousarray(jxrs.admx_multiply_a_omega(g, q, f), dtype=np.float32)
        y = np.ascontiguousarray(jxrs.admx_multiply_at_omega(g, g_small, f), dtype=np.float32)
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

    g_small = np.ascontiguousarray(jxrs.admx_multiply_a_omega(g, q, f), dtype=np.float32)
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
    p64 = np.ascontiguousarray(p, dtype=np.float64)
    q64 = np.ascontiguousarray(q, dtype=np.float64)
    ll0 = float(jxrs.admx_loglikelihood(g, p64, q64))
    logger.info(f"{'Initial log-likelihood':<22}: {ll0:.3f}")
    _emit_progress(
        callback,
        "als_done",
        elapsed=float(elapsed),
        initial_ll=float(ll0),
        converged_iter=converged_iter,
        max_iter=int(max_iter),
    )
    return p64, q64


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
    p = np.ascontiguousarray(p, dtype=np.float64)
    q = np.ascontiguousarray(q, dtype=np.float64)
    t0 = time.time()
    m_p = np.zeros_like(p, dtype=np.float64)
    v_p = np.zeros_like(p, dtype=np.float64)
    m_q = np.zeros_like(q, dtype=np.float64)
    v_q = np.zeros_like(q, dtype=np.float64)
    p_em = np.empty_like(p, dtype=np.float64)
    q_em = np.empty_like(q, dtype=np.float64)

    p_best = np.array(p, copy=True)
    q_best = np.array(q, copy=True)
    ll_best = -np.inf
    no_improve = 0
    lr_cur = float(lr)
    step = 0
    check_every = max(1, int(check))
    _emit_progress(
        callback,
        "adam_start",
        max_iter=int(max_iter),
        check_every=int(check_every),
    )

    for it in range(int(max_iter)):
        step += 1
        jxrs.admx_em_step_inplace(g, p, q, p_em, q_em)
        jxrs.admx_adam_update_p_inplace(
            p,
            p_em,
            m_p,
            v_p,
            float(lr_cur),
            float(beta1),
            float(beta2),
            float(reg_adam),
            int(step),
        )
        jxrs.admx_adam_update_q_inplace(
            q,
            q_em,
            m_q,
            v_q,
            float(lr_cur),
            float(beta1),
            float(beta2),
            float(reg_adam),
            int(step),
        )

        if (it + 1) % check_every != 0:
            continue

        ll_cur = float(jxrs.admx_loglikelihood(g, p, q))
        logger.info(
            f"{'ADAM iteration':<22}: {it + 1:>4}/{int(max_iter):<4} "
            f"ll={ll_cur:>14.3f} lr={lr_cur:>9.3e}"
        )
        _emit_progress(
            callback,
            "adam_iter",
            iteration=int(it + 1),
            max_iter=int(max_iter),
            ll=float(ll_cur),
            lr=float(lr_cur),
        )
        if abs(ll_cur - ll_best) < 0.1:
            _emit_progress(
                callback,
                "adam_stop",
                reason="ll_delta_small",
                iteration=int(it + 1),
            )
            break

        if ll_cur > ll_best:
            ll_best = ll_cur
            p_best = np.array(p, copy=True)
            q_best = np.array(q, copy=True)
            no_improve = 0
        else:
            no_improve += 1
            old_lr = lr_cur
            lr_cur = max(lr_cur * float(lr_decay), float(min_lr))
            logger.info(
                f"{'No improvement':<22}: count={int(no_improve)} "
                f"lr {old_lr:>9.3e} -> {lr_cur:>9.3e}"
            )
            _emit_progress(
                callback,
                "adam_lr_decay",
                iteration=int(it + 1),
                no_improve=int(no_improve),
                old_lr=float(old_lr),
                new_lr=float(lr_cur),
            )
            if no_improve >= 2:
                logger.info(f"{'Early stopping':<22}: triggered")
                _emit_progress(
                    callback,
                    "adam_stop",
                    reason="no_improve",
                    iteration=int(it + 1),
                )
                break

    ll_final = float(jxrs.admx_loglikelihood(g, p_best, q_best))
    elapsed = time.time() - t0
    logger.info(
        f"{'Final log-likelihood':<22}: {ll_final:.3f} (elapsed {elapsed:.2f}s)"
    )
    _emit_progress(
        callback,
        "adam_done",
        elapsed=float(elapsed),
        final_ll=float(ll_final),
    )
    return p_best, q_best


def train_adamixture(
    cfg: ADAMixtureConfig,
    logger: logging.Logger,
    callback: ProgressCallback = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    _set_thread_env(cfg.threads)
    try:
        jxrs.admx_set_threads(int(max(1, int(cfg.threads))))
    except Exception:
        # Backward-compatible fallback when Rust extension doesn't expose thread setter.
        pass
    np.random.seed(int(cfg.seed))

    g = load_genotype_u8_matrix(
        cfg.genotype_path,
        chunk_size=int(cfg.chunk_size),
        snps_only=bool(cfg.snps_only),
        maf=float(cfg.maf),
        missing_rate=float(cfg.geno),
    )
    m, n = g.shape
    logger.info(f"{'Data loaded':<22}: SNPs={m}, samples={n}")
    _emit_progress(callback, "data_loaded", m=int(m), n=int(n))

    f = np.ascontiguousarray(jxrs.admx_allele_frequency(g), dtype=np.float32)
    u, s, v = _rsvd(
        g,
        f,
        k=int(cfg.k),
        seed=int(cfg.seed),
        power=int(cfg.power),
        tol=float(cfg.tole_svd),
        logger=logger,
        callback=callback,
    )
    p0, q0 = _als_init(
        g,
        u,
        s,
        v,
        f,
        seed=int(cfg.seed),
        k=int(cfg.k),
        max_iter=int(cfg.max_als),
        tol=float(cfg.tole_als),
        reg=float(cfg.reg_als),
        logger=logger,
        callback=callback,
    )
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
