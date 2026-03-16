# assoc_api.py
"""
High-level Python APIs wrapping Rust-accelerated association tests.

This module provides:
  - FEM(): fast fixed-effect model (LM / GLM-like) GWAS scan in chunks
  - lmm_reml(): REML-based LMM GWAS scan on rotated genotype chunks
  - fastlmm_*(): FaST-LMM (rank-k + residual) wrappers
  - LMM / LM: convenient OO wrappers for repeated scans
  - FarmCPU utilities (REM / ll / SUPER / farmcpu)

Notes
-----
- Rust backend functions are imported from the local extension module.
- Genotype matrix convention in THIS FILE:
    M (or snp_chunk) is SNP-major: shape = (m_snps, n_samples)
  i.e., rows are SNPs, columns are samples.
- Most computations require contiguous arrays (C-order) and specific dtypes.
  This wrapper enforces them before calling Rust.

Type conventions
----------------
- y: (n,) or (n,1) -> coerced to contiguous float64 1D
- X: (n,p) -> contiguous float64
- M: (m,n) -> contiguous float32 (for Rust f32 kernels)
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in",
)

from joblib import Parallel, delayed, cpu_count
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:
    _threadpool_limits = None

# Rust core kernels (PyO3 extension)
from janusx.janusx import (
    glmf32,
    glmf32_full,
    lmm_reml_chunk_f32,
    lmm_reml_null_f32,
    ml_loglike_null_f32,
    lmm_assoc_chunk_f32,
    RustPCGMatrixFreeState as _RustPCGMatrixFreeState,
    fastlmm_reml_chunk_f32,
    fastlmm_reml_null_f32,
    fastlmm_assoc_chunk_f32,
)

try:
    from janusx.janusx import (
        glmf32_packed as _glmf32_packed,
        bed_packed_row_flip_mask as _bed_packed_row_flip_mask,
        bed_packed_decode_rows_f32 as _bed_packed_decode_rows_f32,
        farmcpu_rem_dense as _farmcpu_rem_dense,
        farmcpu_rem_packed as _farmcpu_rem_packed,
        farmcpu_super_dense as _farmcpu_super_dense,
        farmcpu_super_packed as _farmcpu_super_packed,
    )
except Exception:
    _glmf32_packed = None
    _bed_packed_row_flip_mask = None
    _bed_packed_decode_rows_f32 = None
    _farmcpu_rem_dense = None
    _farmcpu_rem_packed = None
    _farmcpu_super_dense = None
    _farmcpu_super_packed = None


def _coerce_bed_packed_ctx(M: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize BED-packed payload into a dict context.

    Accepted input forms:
    1) dict with keys:
       - packed (uint8, shape=(m, ceil(n/4)))
       - maf (float32/float64, shape=(m,))
       - n_samples (int)
       Optional:
       - missing_rate
       - row_flip (bool, shape=(m,))
    2) tuple/list from `load_bed_2bit_packed(prefix)`:
       (packed, missing_rate, maf, std_denom, n_samples)
    """
    ctx_in: Optional[Dict[str, Any]] = None
    if isinstance(M, dict):
        if "packed" in M and "maf" in M and "n_samples" in M:
            ctx_in = M
    elif isinstance(M, (tuple, list)) and len(M) >= 5:
        packed, miss, maf, _denom, n_samples = M[:5]
        ctx_in = {
            "packed": packed,
            "missing_rate": miss,
            "maf": maf,
            "n_samples": n_samples,
        }
    if ctx_in is None:
        return None

    packed = np.ascontiguousarray(np.asarray(ctx_in["packed"], dtype=np.uint8))
    if packed.ndim != 2:
        raise ValueError("packed must be 2D with shape (m, ceil(n_samples/4)).")

    maf = np.ascontiguousarray(np.asarray(ctx_in["maf"], dtype=np.float32).reshape(-1))
    if maf.shape[0] != packed.shape[0]:
        raise ValueError(
            f"maf length mismatch: got {maf.shape[0]}, expected {packed.shape[0]}"
        )

    n_samples = int(ctx_in["n_samples"])
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0 in packed context.")
    exp_bps = (n_samples + 3) // 4
    if packed.shape[1] != exp_bps:
        raise ValueError(
            f"packed bytes_per_snp mismatch: got {packed.shape[1]}, expected {exp_bps}"
        )

    row_flip = ctx_in.get("row_flip", None)
    if row_flip is None:
        if _bed_packed_row_flip_mask is None:
            raise RuntimeError(
                "Rust extension missing bed_packed_row_flip_mask. Rebuild janusx extension."
            )
        row_flip = np.asarray(
            _bed_packed_row_flip_mask(packed, int(n_samples)),
            dtype=np.bool_,
        )
    row_flip = np.ascontiguousarray(np.asarray(row_flip, dtype=np.bool_).reshape(-1))
    if row_flip.shape[0] != packed.shape[0]:
        raise ValueError(
            f"row_flip length mismatch: got {row_flip.shape[0]}, expected {packed.shape[0]}"
        )

    ctx_out = {
        "packed": packed,
        "maf": maf,
        "n_samples": n_samples,
        "row_flip": row_flip,
    }
    if "missing_rate" in ctx_in:
        ctx_out["missing_rate"] = np.ascontiguousarray(
            np.asarray(ctx_in["missing_rate"], dtype=np.float32).reshape(-1)
        )
    if isinstance(M, dict):
        M.update(ctx_out)
        return M
    return ctx_out


def _normalize_sample_indices(
    sample_indices: Optional[np.ndarray],
    n_samples: int,
    n_target: int,
) -> Optional[np.ndarray]:
    if sample_indices is None:
        return None
    sidx = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    if sidx.size != int(n_target):
        raise ValueError(
            f"sample_indices length mismatch: got {sidx.size}, expected {n_target}"
        )
    if np.any(sidx < 0) or np.any(sidx >= int(n_samples)):
        raise ValueError("sample_indices has out-of-range values.")
    return np.ascontiguousarray(sidx, dtype=np.int64)


def _select_snp_rows(
    M: Any,
    row_idx: np.ndarray,
    *,
    sample_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return selected genotype rows as float32 matrix with shape (k, n_selected).
    """
    ridx = np.asarray(row_idx, dtype=np.int64).reshape(-1)
    packed_ctx = _coerce_bed_packed_ctx(M)
    if packed_ctx is None:
        arr = np.asarray(M, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("M must be 2D (m, n).")
        if np.any(ridx < 0) or np.any(ridx >= arr.shape[0]):
            raise ValueError("row_idx out of range for dense genotype matrix.")
        out = arr[ridx]
        if sample_indices is not None:
            out = out[:, sample_indices]
        return np.ascontiguousarray(out, dtype=np.float32)

    if _bed_packed_decode_rows_f32 is None:
        raise RuntimeError(
            "Rust extension missing bed_packed_decode_rows_f32. Rebuild janusx extension."
        )
    decoded = _bed_packed_decode_rows_f32(
        packed_ctx["packed"],
        int(packed_ctx["n_samples"]),
        np.ascontiguousarray(ridx, dtype=np.int64),
        packed_ctx["row_flip"],
        packed_ctx["maf"],
        sample_indices,
    )
    return np.ascontiguousarray(np.asarray(decoded, dtype=np.float32), dtype=np.float32)


def FEM(
    y: np.ndarray,
    X: np.ndarray,
    M: Any,
    chunksize: int = 50_000,
    threads: int = 1,
    ixx: Optional[np.ndarray] = None,
    sample_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fixed Effects Model (FEM) GWAS scan (fast GLM/LM in Rust, chunked).

    This is a thin wrapper around the Rust function `glmf32`, which evaluates
    SNP-by-SNP association under a fixed-effect linear model.

    Parameters
    ----------
    y : np.ndarray
        Phenotype vector of length n. Accepts shape (n,), (n,1).
        Internally coerced to contiguous float64 1D.

    X : np.ndarray
        Covariate/design matrix of shape (n, p). Must align with y.
        Internally coerced to contiguous float64.

        NOTE: Add an intercept column in your caller if you want one.
        (FarmCPU and LM wrapper already do that.)

    M : np.ndarray or packed tuple/dict
        Either:
        - Dense genotype matrix in SNP-major layout (m, n), or
        - BED-packed payload from `load_bed_2bit_packed(...)`.

    chunksize : int, default=50_000
        Number of SNPs processed per internal chunk in Rust.

        Practical note:
        - Larger chunks reduce overhead (Python <-> Rust calls)
        - Larger chunks increase peak memory traffic / cache pressure

    threads : int, default=1
        Number of Rust worker threads used inside glmf32.
        (This is *not* joblib threads; it's passed to Rust.)

    sample_indices : np.ndarray or None
        Optional sample-index mapping used for packed input. Length must equal
        len(y). Ignored for dense input unless you explicitly provide it.

    Returns
    -------
    result : np.ndarray
        Rust returns an array-like object which is converted to a NumPy array.
        The exact output column layout depends on your Rust implementation.

        In your downstream code you treat it as:
            result[:, [0, 1, -1]]  -> beta, se, p

    Raises
    ------
    ValueError
        If M is not 2D or M.shape[1] != len(y).

    Notes
    -----
    - This function never transposes M. Ensure SNP-major input (m,n).
    - Ensure your Rust `glmf32` expects:
        y: float64[n]
        X: float64[n,p]
        ixx: float64[p,p]  (pinv of X'X)
        M: float32[m,n]
    """
    # ---- Validate / normalize inputs for Rust ----
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    X = np.ascontiguousarray(X, dtype=np.float64)

    # Precompute (X'X)^(-1) once unless caller provides a cached matrix.
    if ixx is None:
        ixx = np.ascontiguousarray(np.linalg.pinv(X.T @ X), dtype=np.float64)
    else:
        ixx = np.ascontiguousarray(ixx, dtype=np.float64)

    packed_ctx = _coerce_bed_packed_ctx(M)
    if packed_ctx is not None:
        if _glmf32_packed is None:
            raise RuntimeError(
                "Rust extension missing glmf32_packed. Rebuild janusx extension."
            )
        sidx = _normalize_sample_indices(
            sample_indices,
            int(packed_ctx["n_samples"]),
            int(y.shape[0]),
        )
        if sidx is None and int(packed_ctx["n_samples"]) != int(y.shape[0]):
            raise ValueError(
                f"Packed n_samples={packed_ctx['n_samples']} does not match len(y)={y.shape[0]}. "
                "Provide sample_indices to align packed genotype samples."
            )
        return np.asarray(
            _glmf32_packed(
                y,
                X,
                ixx,
                packed_ctx["packed"],
                int(packed_ctx["n_samples"]),
                packed_ctx["row_flip"],
                packed_ctx["maf"],
                sidx,
                int(chunksize),
                int(threads),
            )
        )

    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError("M must be a 2D array with shape (m, n) [SNP-major].")
    if sample_indices is not None:
        sidx = _normalize_sample_indices(sample_indices, int(M.shape[1]), int(y.shape[0]))
        M = np.ascontiguousarray(M[:, sidx], dtype=np.float32)
    else:
        if M.shape[1] != y.shape[0]:
            raise ValueError(
                f"M must be shape (m, n). Got M.shape={M.shape}, but n=len(y)={y.shape[0]}"
            )
        M = np.ascontiguousarray(M, dtype=np.float32)

    result = []
    for start in range(0, M.shape[0], chunksize):
        end = min(start + chunksize, M.shape[0])
        result.append(
            glmf32(
                y,
                X,
                ixx,
                M[start:end],
                int(end - start),
                int(threads),
            )
        )
    return np.concatenate(result, axis=0)


def lmm_reml(
    S: np.ndarray,
    utx: np.ndarray,
    uty: np.ndarray,
    utsnp_chunk: np.ndarray,
    bounds: tuple,
    max_iter: int = 30,
    tol: float = 1e-2,
    threads: int = 4,
    nullml: Optional[float] = None,
) -> np.ndarray:
    """
    REML-based LMM scan on a SNP chunk using a Rust kernel (chunked, parallel).

    This wraps the Rust function `lmm_reml_chunk_f32`. The intended workflow is:

    1) You have a kinship K (n x n), compute eigen-decomposition:
           K = U diag(S) U^T
       Store:
           S  : eigenvalues
           Dh : U^T  (often called "transpose of eigenvectors")

    2) Rotate phenotype/covariates once:
           uty  = U^T @ y
           utx  = U^T @ X

    3) For each genotype chunk (m_chunk x n) in SNP-major layout:
           g_rot = snp_chunk @ U
       Then pass g_rot to Rust for SNP-wise REML optimization.

    Parameters
    ----------
    S : np.ndarray, shape (n,)
        Eigenvalues of kinship matrix K. Must be float64 contiguous.

    utx : np.ndarray, shape (n, q)
        Rotated covariates (U^T @ X). Must be float64 contiguous.

    uty : np.ndarray, shape (n,)
        Rotated phenotype (U^T @ y). Must be float64 contiguous.

    g_rot_chunk : np.ndarray, shape (m_chunk, n)
        Rotated SNP-major genotype chunk. dtype can be float32/float64.
        Will be converted to float32 for Rust.

    bounds : tuple (low, high)
        Search bounds in log10(lambda) for Brent/1D optimization.

    max_iter : int
        Maximum iterations in the scalar optimizer (in Rust).

    tol : float
        Convergence tolerance in log10(lambda).

    threads : int
        Number of Rust worker threads.

    nullml : float, optional
        Null-model ML log-likelihood. If provided, plrt is appended.

    Returns
    -------
    beta_se_p : np.ndarray, shape (m_chunk, 3) or (m_chunk, 4)
        Per-SNP results: beta, standard error, pwald.
        If nullml is provided, an extra column plrt is appended.

    Performance notes
    -----------------
    - The bottleneck for large n is often the rotation you do before calling:
          g_rot_chunk = snp_chunk @ U
      This is an (m_chunk x n) by (n x n) multiply, i.e., O(m_chunk * n^2).
      For n=50,000 this is not feasible.

      In practice, for very large n you must avoid explicit n x n rotations.
      Use low-rank / iterative methods, or compute in the eigen space without
      materializing Dh (depends on your model design).
    """
    low, high = bounds

    # ---- Normalize dtypes/contiguity ----
    S = np.ascontiguousarray(S, dtype=np.float64).ravel()
    utx = np.ascontiguousarray(utx, dtype=np.float64)
    uty = np.ascontiguousarray(uty, dtype=np.float64).ravel()
    utsnp_chunk = np.ascontiguousarray(utsnp_chunk, dtype=np.float32)

    # ---- Call Rust kernel ----
    beta_se_p = lmm_reml_chunk_f32(
        S,
        utx,
        uty,
        float(low),
        float(high),
        utsnp_chunk,
        int(max_iter),
        float(tol),
        int(threads),
        None if nullml is None else float(nullml),
    )

    return beta_se_p


def lmm_reml_null(
    S: np.ndarray,
    utx: np.ndarray,
    uty: np.ndarray,
    bounds: tuple,
    max_iter: int = 30,
    tol: float = 1e-2,
) -> Tuple[float, float, float]:
    """
    Null-model REML optimization using the Rust kernel.

    Parameters
    ----------
    S : np.ndarray, shape (n,)
        Eigenvalues of kinship matrix K.

    utx : np.ndarray, shape (n, q)
        Rotated covariates (U^T @ X).

    uty : np.ndarray, shape (n,)
        Rotated phenotype (U^T @ y).

    bounds : tuple (low, high)
        Search bounds in log10(lambda).

    max_iter : int
        Maximum iterations in the scalar optimizer (in Rust).

    tol : float
        Convergence tolerance in log10(lambda).

    Returns
    -------
    lbd : float
        Null-model lambda (ve/vg).

    ml : float
        Null ML log-likelihood (higher is better).

    reml : float
        Null REML log-likelihood (higher is better).
    """
    low, high = bounds
    S = np.ascontiguousarray(S, dtype=np.float64).ravel()
    utx = np.ascontiguousarray(utx, dtype=np.float64)
    uty = np.ascontiguousarray(uty, dtype=np.float64).ravel()

    lbd, ml, reml = lmm_reml_null_f32(
        S,
        utx,
        uty,
        float(low),
        float(high),
        int(max_iter),
        float(tol),
    )
    return lbd, ml, reml


def fastlmm_reml_null(
    S: np.ndarray,
    u1tx: np.ndarray,
    u2tx: np.ndarray,
    u1ty: np.ndarray,
    u2ty: np.ndarray,
    bounds: tuple,
    max_iter: int = 50,
    tol: float = 1e-2,
    model: str = "add",
) -> Tuple[float, float, float]:
    """
    FaST-LMM null-model REML optimization (rank-k + residual space).

    This is a thin wrapper over Rust `fastlmm_reml_null_f32`.

    Parameters
    ----------
    S : np.ndarray, shape (k,)
        Eigenvalues (or s^2) corresponding to the low-rank space U (k components).

    u1tx : np.ndarray, shape (k, p)
        Projected covariates in U space (U^T @ X).

    u2tx : np.ndarray, shape (n, p)
        Residual covariates (X - U @ (U^T @ X)).

    u1ty : np.ndarray, shape (k,)
        Projected phenotype (U^T @ y).

    u2ty : np.ndarray, shape (n,)
        Residual phenotype (y - U @ (U^T @ y)).

    bounds : tuple (low, high)
        Search bounds in log10(lambda).

    max_iter : int
        Max iterations in Brent optimizer.

    tol : float
        Convergence tolerance in log10(lambda).

    model : {"add", "dom", "rec", "het"}
        Genetic effect coding mode. Passed through to Rust kernel.

    Returns
    -------
    lbd : float
        Estimated lambda (ve/vg).

    ml : float
        ML log-likelihood (higher is better).

    reml : float
        REML log-likelihood (higher is better).
    """
    low, high = bounds
    S = np.ascontiguousarray(S, dtype=np.float64).ravel()
    u1tx = np.ascontiguousarray(u1tx, dtype=np.float64)
    u2tx = np.ascontiguousarray(u2tx, dtype=np.float64)
    u1ty = np.ascontiguousarray(u1ty, dtype=np.float64).ravel()
    u2ty = np.ascontiguousarray(u2ty, dtype=np.float64).ravel()

    lbd, ml, reml = fastlmm_reml_null_f32(
        S,
        u1tx,
        u2tx,
        u1ty,
        u2ty,
        float(low),
        float(high),
        int(max_iter),
        float(tol),
        str(model),
    )
    return lbd, ml, reml


def fastlmm_reml(
    S: np.ndarray,
    u1tx: np.ndarray,
    u2tx: np.ndarray,
    u1ty: np.ndarray,
    u2ty: np.ndarray,
    snp_chunk: np.ndarray,
    u1t: np.ndarray,
    bounds: tuple,
    max_iter: int = 50,
    tol: float = 1e-2,
    threads: int = 4,
    nullml: Optional[float] = None,
    model: str = "add",
) -> np.ndarray:
    """
    FaST-LMM REML scan on a SNP chunk (rank-k + residual space).

    This is a thin wrapper over Rust `fastlmm_reml_chunk_f32`.

    Parameters
    ----------
    S : np.ndarray, shape (k,)
        Eigenvalues (or s^2) for the low-rank space.

    u1tx : np.ndarray, shape (k, p)
        Projected covariates in U space (U^T @ X).

    u2tx : np.ndarray, shape (n, p)
        Residual covariates (X - U @ (U^T @ X)).

    u1ty : np.ndarray, shape (k,)
        Projected phenotype (U^T @ y).

    u2ty : np.ndarray, shape (n,)
        Residual phenotype (y - U @ (U^T @ y)).

    snp_chunk : np.ndarray, shape (m, n)
        Raw SNP chunk (SNP-major), before projection.

    u1t : np.ndarray, shape (k, n)
        Projection matrix U^T used by Rust to project SNPs internally.

    bounds : tuple (low, high)
        Search bounds in log10(lambda).

    max_iter : int
        Max iterations in Brent optimizer.

    tol : float
        Convergence tolerance in log10(lambda).

    threads : int
        Rust worker threads.

    nullml : float, optional
        Null-model ML log-likelihood. If provided, plrt is appended.

    model : {"add", "dom", "rec", "het"}
        Genetic effect coding mode applied in Rust before projection.

    Returns
    -------
    beta_se_p : np.ndarray, shape (m, 3) or (m, 4)
        Per-SNP beta, standard error, pwald.
        If nullml is provided, an extra column plrt is appended.
    """
    low, high = bounds
    S = np.ascontiguousarray(S, dtype=np.float64).ravel()
    u1tx = np.ascontiguousarray(u1tx, dtype=np.float64)
    u2tx = np.ascontiguousarray(u2tx, dtype=np.float64)
    u1ty = np.ascontiguousarray(u1ty, dtype=np.float64).ravel()
    u2ty = np.ascontiguousarray(u2ty, dtype=np.float64).ravel()

    snp_chunk = np.ascontiguousarray(snp_chunk, dtype=np.float32)
    u1t = np.ascontiguousarray(u1t, dtype=np.float32)

    beta_se_p = fastlmm_reml_chunk_f32(
        S,
        u1tx,
        u2tx,
        u1ty,
        u2ty,
        snp_chunk,
        u1t,
        float(low),
        float(high),
        int(max_iter),
        float(tol),
        int(threads),
        None if nullml is None else float(nullml),
        str(model),
    )
    return beta_se_p


def fastlmm_assoc_chunk(
    S: np.ndarray,
    u1tx: np.ndarray,
    u2tx: np.ndarray,
    u1ty: np.ndarray,
    u2ty: np.ndarray,
    log10_lbd: float,
    u1tsnp_chunk: np.ndarray,
    u2tsnp_chunk: np.ndarray,
    threads: int = 4,
    nullml: Optional[float] = None,
) -> np.ndarray:
    """
    FaST-LMM fixed-lambda association on a SNP chunk.

    This is a thin wrapper over Rust `fastlmm_assoc_chunk_f32`.

    Parameters
    ----------
    S : np.ndarray, shape (k,)
        Eigenvalues (or s^2) for the low-rank space.

    u1tx : np.ndarray, shape (k, p)
        Projected covariates in U space (U^T @ X).

    u2tx : np.ndarray, shape (n, p)
        Residual covariates (X - U @ (U^T @ X)).

    u1ty : np.ndarray, shape (k,)
        Projected phenotype (U^T @ y).

    u2ty : np.ndarray, shape (n,)
        Residual phenotype (y - U @ (U^T @ y)).

    log10_lbd : float
        Fixed log10(lambda) for all SNPs.

    u1tsnp_chunk : np.ndarray, shape (m, k)
        Projected SNP chunk in U space (SNP-major).

    u2tsnp_chunk : np.ndarray, shape (m, n)
        Residual SNP chunk (SNP-major).

    threads : int
        Rust worker threads.

    nullml : float, optional
        Null-model ML log-likelihood. If provided, plrt is appended.

    Returns
    -------
    beta_se_p : np.ndarray, shape (m, 3) or (m, 4)
        Per-SNP beta, standard error, pwald.
        If nullml is provided, an extra column plrt is appended.
    """
    S = np.ascontiguousarray(S, dtype=np.float64).ravel()
    u1tx = np.ascontiguousarray(u1tx, dtype=np.float64)
    u2tx = np.ascontiguousarray(u2tx, dtype=np.float64)
    u1ty = np.ascontiguousarray(u1ty, dtype=np.float64).ravel()
    u2ty = np.ascontiguousarray(u2ty, dtype=np.float64).ravel()

    u1tsnp_chunk = np.ascontiguousarray(u1tsnp_chunk, dtype=np.float32)
    u2tsnp_chunk = np.ascontiguousarray(u2tsnp_chunk, dtype=np.float32)

    beta_se_p = fastlmm_assoc_chunk_f32(
        S,
        u1tx,
        u2tx,
        u1ty,
        u2ty,
        float(log10_lbd),
        u1tsnp_chunk,
        u2tsnp_chunk,
        int(threads),
        None if nullml is None else float(nullml),
    )
    return beta_se_p


def ml_loglike_null(
    S: np.ndarray,
    utx: np.ndarray,
    uty: np.ndarray,
    log10_lbd: float,
) -> float:
    """
    ML log-likelihood for the null model at a given log10(lambda).
    """
    S = np.ascontiguousarray(S, dtype=np.float64).ravel()
    Xcov = np.ascontiguousarray(utx, dtype=np.float64)
    y_rot = np.ascontiguousarray(uty, dtype=np.float64).ravel()
    return ml_loglike_null_f32(S, Xcov, y_rot, float(log10_lbd))


def lmm_assoc_fixed(
    S: np.ndarray,
    utx: np.ndarray,
    uty: np.ndarray,
    log10_lbd: float,
    utsnp_chunk: np.ndarray,
    threads: int = 4,
    nullml: Optional[float] = None,
) -> np.ndarray:
    """
    Fixed-lambda LMM scan on a SNP chunk using a Rust kernel.

    This wraps `lmm_assoc_chunk_f32` and uses a single fixed log10(lambda)
    for all SNPs in the chunk. If nullml is provided, plrt is appended.

    Parameters
    ----------
    S : np.ndarray, shape (n,)
        Eigenvalues of kinship (same order as rotated data).

    utx : np.ndarray, shape (n, p)
        Rotated covariates (U^T @ X).

    uty : np.ndarray, shape (n,)
        Rotated phenotype (U^T @ y).

    log10_lbd : float
        Fixed log10(lambda) for all SNPs.

    utsnp_chunk : np.ndarray, shape (m, n)
        Rotated SNP chunk (U^T @ G), SNP-major.
    """
    S = np.ascontiguousarray(S, dtype=np.float64).ravel()
    Xcov = np.ascontiguousarray(utx, dtype=np.float64)
    y_rot = np.ascontiguousarray(uty, dtype=np.float64).ravel()

    utsnp_chunk = np.ascontiguousarray(utsnp_chunk, dtype=np.float32)

    beta_se_p = lmm_assoc_chunk_f32(
        S,
        Xcov,
        y_rot,
        float(log10_lbd),
        utsnp_chunk,
        int(threads),
        None if nullml is None else float(nullml),
    )
    return beta_se_p


class _GenotypeKinshipOperator:
    """
    Matrix-free kinship operator from SNP-major genotype G (m, n):
        K = (Gc^T Gc) / m,  Gc = G - row_mean(G)
    """

    def __init__(self, geno_snp_major: np.ndarray):
        g = np.ascontiguousarray(geno_snp_major, dtype=np.float32)
        if g.ndim != 2:
            raise ValueError("geno_snp_major must be 2D (m, n)")
        self.g = g
        self.m = int(g.shape[0])
        self.n = int(g.shape[1])
        if self.m <= 0 or self.n <= 0:
            raise ValueError("geno_snp_major must be non-empty")
        self.inv_m = 1.0 / float(self.m)
        self.row_mean = np.mean(g, axis=1, dtype=np.float64)

        sumsq_col = np.einsum("ij,ij->j", g, g, dtype=np.float64)
        mu_dot_g = self.row_mean @ g  # shape (n,)
        mu2_sum = float(np.dot(self.row_mean, self.row_mean))
        diag = (sumsq_col - 2.0 * mu_dot_g + mu2_sum) * self.inv_m
        self._diag = np.clip(np.asarray(diag, dtype=np.float64), 1e-12, np.inf)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.n:
            raise ValueError("x has incompatible length for genotype operator")
        sx = float(np.sum(x))
        tmp = (self.g @ x) - self.row_mean * sx  # (m,)
        out = self.g.T @ tmp
        corr = float(np.dot(self.row_mean, tmp))
        out = (out - corr) * self.inv_m
        return np.asarray(out, dtype=np.float64)

    def matmat(self, x_block: np.ndarray) -> np.ndarray:
        x_block = np.ascontiguousarray(x_block, dtype=np.float64)
        if x_block.ndim != 2 or x_block.shape[1] != self.n:
            raise ValueError("x_block must be (rhs, n)")
        sx = np.sum(x_block, axis=1, dtype=np.float64)  # (rhs,)
        tmp = self.g @ x_block.T  # (m, rhs)
        tmp -= np.outer(self.row_mean, sx)
        out = self.g.T @ tmp  # (n, rhs)
        corr = self.row_mean @ tmp  # (rhs,)
        out -= corr[np.newaxis, :]
        out *= self.inv_m
        return np.asarray(out.T, dtype=np.float64)

    def diag(self) -> np.ndarray:
        return self._diag


def _normal_sf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * np.vectorize(lambda x: math.erfc(float(x) / math.sqrt(2.0)))(z)


def _pcg_block_solve_operator(
    op: Any,
    lbd: float,
    b_block: np.ndarray,
    m_inv_diag: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Block PCG solve for (K + lambda I) X = B using linear operator op for K.
    Shapes:
      b_block: (rhs, n), returns x_block: (rhs, n), converged: (rhs,)
    """
    b = np.ascontiguousarray(b_block, dtype=np.float64)
    if b.ndim != 2:
        raise ValueError("b_block must be 2D (rhs, n)")
    rhs, n = b.shape
    if rhs == 0:
        return np.zeros((0, n), dtype=np.float64), np.zeros((0,), dtype=bool)
    if m_inv_diag.shape[0] != n:
        raise ValueError("m_inv_diag length mismatch in PCG solve")

    tol = float(tol) if np.isfinite(tol) and tol > 0 else 1e-6
    max_iter = max(1, int(max_iter))
    tiny = 1e-30

    x = np.zeros_like(b, dtype=np.float64)
    r = b.copy()
    z = r * m_inv_diag[np.newaxis, :]
    p = z.copy()

    norm_b = np.linalg.norm(b, axis=1)
    norm_b = np.where(np.isfinite(norm_b) & (norm_b > 0.0), norm_b, 1.0)
    rz_old = np.sum(r * z, axis=1)
    converged = np.zeros(rhs, dtype=bool)

    for _ in range(max_iter):
        ap = op.matmat(p) + lbd * p
        denom = np.sum(p * ap, axis=1)
        alpha = np.zeros(rhs, dtype=np.float64)
        good = np.isfinite(denom) & (np.abs(denom) > tiny)
        alpha[good] = rz_old[good] / denom[good]

        x += alpha[:, np.newaxis] * p
        r -= alpha[:, np.newaxis] * ap

        rel = np.linalg.norm(r, axis=1) / norm_b
        converged |= np.isfinite(rel) & (rel <= tol)
        if bool(np.all(converged)):
            break

        z = r * m_inv_diag[np.newaxis, :]
        rz_new = np.sum(r * z, axis=1)

        beta = np.zeros(rhs, dtype=np.float64)
        good_beta = (
            (~converged)
            & np.isfinite(rz_new)
            & np.isfinite(rz_old)
            & (np.abs(rz_old) > tiny)
        )
        beta[good_beta] = rz_new[good_beta] / rz_old[good_beta]

        p = z + beta[:, np.newaxis] * p
        if bool(np.any(converged)):
            p[converged, :] = 0.0
        rz_old = rz_new

    return x, converged


def _estimate_lbd_he_from_genotype_subsample(
    y: np.ndarray,
    xcov: np.ndarray,
    geno_snp_major: np.ndarray,
    row_mean: np.ndarray,
    *,
    sample_size: int = 256,
    seed: int = 42,
    min_var: float = 1e-8,
) -> float:
    """
    No-EVD lambda estimate from genotype operator using HE moments on a sample
    subset. Avoids constructing full dense K.
    """
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    xcov = np.ascontiguousarray(xcov, dtype=np.float64)
    g = np.ascontiguousarray(geno_snp_major, dtype=np.float32)
    mu = np.ascontiguousarray(row_mean, dtype=np.float64).ravel()

    n = int(y.shape[0])
    m = int(g.shape[0])
    if xcov.shape[0] != n or g.shape[1] != n or mu.shape[0] != m:
        raise ValueError("shape mismatch in lambda estimation from genotype")

    beta_hat, *_ = np.linalg.lstsq(xcov, y, rcond=None)
    r = y - xcov @ beta_hat

    ns = int(max(8, min(n, sample_size)))
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if ns == n else np.sort(rng.choice(n, size=ns, replace=False))
    r_sub = r[idx]

    # Build only a small K_sub from centered genotype columns.
    g_sub = np.asarray(g[:, idx], dtype=np.float64)
    g_sub -= mu[:, np.newaxis]
    k_sub = (g_sub.T @ g_sub) / max(float(m), 1.0)

    diag_k = np.diag(k_sub)
    sum_diag_k = float(np.sum(diag_k))
    sum_diag_k2 = float(np.dot(diag_k, diag_k))
    sum_diag_kr2 = float(np.dot(diag_k, r_sub * r_sub))

    n_pairs = ns * (ns - 1) // 2
    if n_pairs <= 1:
        return 1.0

    sum_k_all = float(np.sum(k_sub))
    sum_k = 0.5 * (sum_k_all - sum_diag_k)

    rr_sum = float(np.sum(r_sub))
    rr2_sum = float(np.dot(r_sub, r_sub))
    sum_z = 0.5 * (rr_sum * rr_sum - rr2_sum)

    sum_kk_all = float(np.sum(k_sub * k_sub))
    sum_kk = 0.5 * (sum_kk_all - sum_diag_k2)

    kr_all = float(r_sub @ (k_sub @ r_sub))
    sum_kz = 0.5 * (kr_all - sum_diag_kr2)

    pairs_f = float(n_pairs)
    denom = sum_kk - (sum_k * sum_k) / pairs_f
    if (not np.isfinite(denom)) or (abs(denom) < 1e-18):
        return 1.0

    sigma_g2 = (sum_kz - (sum_k * sum_z) / pairs_f) / denom
    sigma_g2 = float(max(sigma_g2, min_var))
    mean_r2 = float(rr2_sum / max(ns, 1))
    mean_diag_k = float(sum_diag_k / max(ns, 1))
    sigma_e2 = float(max(mean_r2 - sigma_g2 * mean_diag_k, min_var))

    lbd = sigma_e2 / sigma_g2
    if not np.isfinite(lbd) or lbd <= 0.0:
        return 1.0
    return float(np.clip(lbd, 1e-6, 1e6))


class PCGNullScanState:
    """
    Reusable null/scan state for matrix-free fixed-lambda LMM scan.
    """

    def __init__(
        self,
        op: Any,
        xcov: np.ndarray,
        y: np.ndarray,
        lbd: float,
        *,
        pcg_tol: float = 1e-4,
        pcg_max_iter: int = 200,
        block_size: int = 8,
    ):
        self.op = op
        self.xcov = np.ascontiguousarray(xcov, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64).ravel()
        self.lbd = float(lbd)
        self.pcg_tol = float(pcg_tol)
        self.pcg_max_iter = int(pcg_max_iter)
        self.block_size = int(max(1, block_size))

        n, p_cov = self.xcov.shape
        if self.y.shape[0] != n:
            raise ValueError("xcov/y shape mismatch in PCGNullScanState")
        if n <= p_cov + 1:
            raise ValueError("n must be > p_cov + 1")

        k_diag = np.ascontiguousarray(op.diag(), dtype=np.float64).ravel()
        if k_diag.shape[0] != n:
            raise ValueError("operator diag length mismatch")
        self.m_inv_diag = 1.0 / np.clip(k_diag + self.lbd, 1e-8, np.inf)

        rhs0 = np.vstack([self.xcov.T, self.y[np.newaxis, :]])  # (p+1, n)
        x0, _ = _pcg_block_solve_operator(
            op,
            self.lbd,
            rhs0,
            self.m_inv_diag,
            tol=self.pcg_tol,
            max_iter=self.pcg_max_iter,
        )
        self.vinv_x = np.ascontiguousarray(x0[:p_cov, :].T, dtype=np.float64)  # (n, p)
        self.vinv_y = np.ascontiguousarray(x0[p_cov, :], dtype=np.float64)  # (n,)

        a = self.xcov.T @ self.vinv_x
        b0 = self.xcov.T @ self.vinv_y
        y_vinv_y = float(self.y @ self.vinv_y)
        a = np.asarray(a, dtype=np.float64)
        a.flat[:: p_cov + 1] += 1e-6
        chol = cho_factor(a, overwrite_a=False, check_finite=False)
        a_inv_b = cho_solve(chol, b0, check_finite=False)

        self.a_chol = chol
        self.a_inv_b = np.ascontiguousarray(a_inv_b, dtype=np.float64)
        self.b_aib = float(b0 @ a_inv_b)
        self.y_vinv_y = y_vinv_y
        self.df = int(n - p_cov - 1)
        self.n = int(n)
        self.p_cov = int(p_cov)

    def scan(self, snp_chunk: np.ndarray) -> np.ndarray:
        g = np.ascontiguousarray(snp_chunk, dtype=np.float32)
        if g.ndim != 2 or g.shape[1] != self.n:
            raise ValueError("snp_chunk must be (m, n)")
        m = int(g.shape[0])
        out = np.full((m, 3), np.nan, dtype=np.float64)
        if m == 0:
            return out

        p_cov = self.p_cov
        for start in range(0, m, self.block_size):
            end = min(start + self.block_size, m)
            b = np.asarray(g[start:end], dtype=np.float64)  # (bs, n)
            vinv_g, _ = _pcg_block_solve_operator(
                self.op,
                self.lbd,
                b,
                self.m_inv_diag,
                tol=self.pcg_tol,
                max_iter=self.pcg_max_iter,
            )

            d = np.sum(b * vinv_g, axis=1)
            e = b @ self.vinv_y
            c_mat = self.xcov.T @ vinv_g.T  # (p, bs)

            for j in range(end - start):
                c_vec = c_mat[:, j]
                a_inv_c = cho_solve(self.a_chol, c_vec, check_finite=False)
                ct_aic = float(c_vec @ a_inv_c)
                schur = float(d[j] - ct_aic)
                if (not np.isfinite(schur)) or schur <= 1e-12:
                    continue
                ct_aib = float(c_vec @ self.a_inv_b)
                num = float(e[j] - ct_aib)
                beta = num / schur
                q = self.b_aib + (num * num) / schur
                rwr = max(self.y_vinv_y - q, 0.0)
                sigma2 = rwr / float(self.df)
                se = math.sqrt(max(sigma2 / schur, 0.0))
                if not np.isfinite(beta) or not np.isfinite(se) or se <= 0.0:
                    pval = 1.0
                else:
                    z = abs(beta / se)
                    pval = float(min(max(2.0 * (0.5 * math.erfc(z / math.sqrt(2.0))), np.finfo(np.float64).tiny), 1.0))
                out[start + j, 0] = beta
                out[start + j, 1] = se
                out[start + j, 2] = pval

        return out


class PCGLMMMatrixFree:
    """
    Matrix-free PCG LMM:
      - no dense K construction
      - Rust packed 2-bit genotype operator (hard-call 0/1/2 + missing)
      - optional randomized Nyström preconditioner (`precond_kind="rand_nys"`)
      - `scan_rhs_cap` tunes scan-time RHS tile width (speed/memory trade-off, 0=auto)
      - null-model solve once
      - reusable scan state for repeated chunks
      - `scan_mode="score_vr"` enables score + variance-ratio approximation (faster scan)
      - optional two-stage refine: score_vr full scan + exact rescoring on selected hits
    """

    def __init__(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray],
        kinship_geno: np.ndarray,
        lbd: Optional[float] = None,
        pcg_tol: float = 1e-4,
        pcg_max_iter: int = 200,
        block_size: int = 0,
        scan_rhs_cap: int = 0,
        precond_kind: str = "rand_nys",
        precond_rank: int = 0,
        precond_oversample: int = 2,
        precond_seed: int = 42,
        scan_mode: str = "exact",
        score_vr_calib_snps: int = 128,
        score_vr_refine_topk: int = 0,
        score_vr_refine_p: float = 0.0,
        lbd_sample_size: int = 256,
        lbd_seed: int = 42,
    ):
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        x_arr = (
            np.concatenate([np.ones((y_arr.shape[0], 1)), np.asarray(X, dtype=np.float64)], axis=1)
            if X is not None
            else np.ones((y_arr.shape[0], 1), dtype=np.float64)
        )
        self.y = np.ascontiguousarray(y_arr.ravel(), dtype=np.float64)
        self.X = np.ascontiguousarray(x_arr, dtype=np.float64)
        geno_arr = np.ascontiguousarray(kinship_geno, dtype=np.float32)
        if geno_arr.ndim != 2:
            raise ValueError("kinship_geno must be 2D (m, n)")
        if geno_arr.shape[1] != self.y.shape[0]:
            raise ValueError(
                f"kinship_geno sample size mismatch: geno n={geno_arr.shape[1]}, y n={self.y.shape[0]}"
            )

        self.pcg_tol = float(pcg_tol)
        self.pcg_max_iter = int(pcg_max_iter)
        self.block_size = int(max(0, block_size))
        self.scan_rhs_cap = int(max(0, scan_rhs_cap))
        self.precond_kind = str(precond_kind)
        self.precond_rank = int(max(0, precond_rank))
        self.precond_oversample = int(max(1, precond_oversample))
        self.precond_seed = int(precond_seed)
        self.scan_mode = str(scan_mode)
        self.score_vr_calib_snps = int(max(8, score_vr_calib_snps))
        self.score_vr_refine_topk = int(max(0, score_vr_refine_topk))
        self.score_vr_refine_p = float(max(0.0, score_vr_refine_p))
        self.scan_rhs_cap_resolved = self.scan_rhs_cap
        self.block_size_resolved = self.block_size
        self.score_vr_ratio = np.nan

        if lbd is None:
            valid = np.isfinite(geno_arr) & (geno_arr >= 0.0)
            row_sum = np.sum(np.where(valid, geno_arr, 0.0), axis=1, dtype=np.float64)
            row_cnt = np.sum(valid, axis=1, dtype=np.int64)
            row_mean = np.divide(
                row_sum,
                np.maximum(row_cnt, 1),
                out=np.zeros_like(row_sum, dtype=np.float64),
                where=row_cnt > 0,
            )
            self.lbd_null = _estimate_lbd_he_from_genotype_subsample(
                self.y,
                self.X,
                geno_arr,
                row_mean,
                sample_size=int(lbd_sample_size),
                seed=int(lbd_seed),
            )
        else:
            self.lbd_null = float(lbd)
        self.lbd_null = float(np.clip(self.lbd_null, 1e-6, 1e6))
        self.evd_secs = 0.0
        self.ML0 = np.nan
        self.LL0 = np.nan

        self._rust_state: Optional[Any] = None
        self._state: Optional[PCGNullScanState] = None
        self._fallback_op: Optional[_GenotypeKinshipOperator] = None
        self._backend_error: str = ""

        def _opt_float(v: Any) -> float:
            if v is None:
                return float("nan")
            try:
                return float(v)
            except Exception:
                return float("nan")

        try:
            self._rust_state = _RustPCGMatrixFreeState(
                geno_arr,
                self.X,
                self.y,
                float(np.log10(self.lbd_null)),
                float(self.pcg_tol),
                int(self.pcg_max_iter),
                int(self.block_size),
                str(self.precond_kind),
                int(self.precond_rank),
                int(self.precond_oversample),
                int(self.precond_seed),
                int(self.scan_rhs_cap),
                str(self.scan_mode),
                int(self.score_vr_calib_snps),
            )
            self.pve = float(self._rust_state.pve)
            self.scan_rhs_cap_resolved = int(getattr(self._rust_state, "scan_rhs_cap", self.scan_rhs_cap))
            self.block_size_resolved = int(getattr(self._rust_state, "block_size", self.block_size))
            self.scan_mode = str(getattr(self._rust_state, "scan_mode", self.scan_mode))
            self.score_vr_ratio = _opt_float(getattr(self._rust_state, "score_vr_ratio", np.nan))
        except Exception as e:
            self._backend_error = str(e)
            warnings.warn(
                "RustPCGMatrixFreeState unavailable for current kinship_geno; "
                "fallback to Python operator path. "
                f"reason={self._backend_error}",
                RuntimeWarning,
            )
            self._fallback_op = _GenotypeKinshipOperator(geno_arr)
            mean_diag = float(np.mean(self._fallback_op.diag()))
            self.pve = (
                mean_diag / (mean_diag + self.lbd_null)
                if (mean_diag + self.lbd_null) > 0
                else np.nan
            )
            self.scan_rhs_cap_resolved = self.block_size
            self.block_size_resolved = self.block_size
            if self.scan_mode.lower() != "exact":
                warnings.warn(
                    "scan_mode != 'exact' requires Rust backend; fallback path uses exact mode.",
                    RuntimeWarning,
                )
                self.scan_mode = "exact"
            if self.score_vr_refine_topk > 0 or self.score_vr_refine_p > 0.0:
                warnings.warn(
                    "score_vr refine requires Rust backend; fallback path disables refine.",
                    RuntimeWarning,
                )
                self.score_vr_refine_topk = 0
                self.score_vr_refine_p = 0.0

    def fit_null(self) -> "PCGLMMMatrixFree":
        if self._rust_state is not None:
            self._rust_state.fit_null()
            self.pve = float(self._rust_state.pve)
            v = getattr(self._rust_state, "score_vr_ratio", np.nan)
            self.score_vr_ratio = float("nan") if v is None else float(v)
            return self

        if self._fallback_op is None:
            raise RuntimeError("matrix-free backend not initialized")
        self._state = PCGNullScanState(
            self._fallback_op,
            self.X,
            self.y,
            self.lbd_null,
            pcg_tol=self.pcg_tol,
            pcg_max_iter=self.pcg_max_iter,
            block_size=self.block_size,
        )
        return self

    def gwas(self, snp: np.ndarray, threads: int = 0) -> np.ndarray:
        snp = np.ascontiguousarray(snp, dtype=np.float32)
        if self._rust_state is not None:
            if not bool(self._rust_state.fitted):
                self._rust_state.fit_null()
            use_refine = (
                str(getattr(self._rust_state, "scan_mode", self.scan_mode)).lower() == "score_vr"
                and (self.score_vr_refine_topk > 0 or self.score_vr_refine_p > 0.0)
            )
            out = self._rust_state.scan_chunk(snp, int(threads))

            if use_refine:
                pval = np.asarray(out[:, 2], dtype=np.float64)
                finite = np.isfinite(pval)
                candidates = np.array([], dtype=np.int64)

                if self.score_vr_refine_p > 0.0:
                    candidates = np.flatnonzero(finite & (pval <= self.score_vr_refine_p)).astype(np.int64)

                if self.score_vr_refine_topk > 0 and np.any(finite):
                    finite_idx = np.flatnonzero(finite)
                    k = int(min(self.score_vr_refine_topk, finite_idx.shape[0]))
                    if k > 0:
                        vals = pval[finite_idx]
                        if k < vals.shape[0]:
                            take = np.argpartition(vals, k - 1)[:k]
                        else:
                            take = np.arange(vals.shape[0], dtype=np.int64)
                        top_idx = finite_idx[take].astype(np.int64)
                        candidates = (
                            np.unique(np.concatenate([candidates, top_idx]))
                            if candidates.size > 0
                            else np.unique(top_idx)
                        )

                # Hard-cap exact rescoring volume per chunk to keep runtime stable.
                if self.score_vr_refine_topk > 0 and candidates.size > self.score_vr_refine_topk:
                    vals = pval[candidates]
                    kcap = int(min(self.score_vr_refine_topk, candidates.size))
                    if kcap > 0:
                        if kcap < candidates.size:
                            take = np.argpartition(vals, kcap - 1)[:kcap]
                        else:
                            take = np.arange(candidates.size, dtype=np.int64)
                        candidates = np.asarray(candidates[take], dtype=np.int64)

                if candidates.size > 0:
                    mode_before = str(getattr(self._rust_state, "scan_mode", self.scan_mode))
                    try:
                        self._rust_state.set_scan_mode("exact")
                        snp_sub = np.ascontiguousarray(snp[candidates, :], dtype=np.float32)
                        out_sub = self._rust_state.scan_chunk(snp_sub, int(threads))
                        out[candidates, :] = out_sub
                    finally:
                        try:
                            self._rust_state.set_scan_mode(mode_before)
                        except Exception:
                            pass

            v = getattr(self._rust_state, "score_vr_ratio", np.nan)
            self.score_vr_ratio = float("nan") if v is None else float(v)
            return out

        # Python fallback path
        if self._state is None:
            self.fit_null()
        if self._state is None:
            raise RuntimeError("failed to initialize matrix-free PCG null state")
        n_threads = int(max(1, threads)) if int(threads) > 0 else None
        ctx = (
            _threadpool_limits(limits=n_threads, user_api="blas")
            if (n_threads is not None and _threadpool_limits is not None)
            else nullcontext()
        )
        with ctx:
            return self._state.scan(snp)


class LMM:
    """
    Fast LMM GWAS using eigen-decomposition of kinship + REML per SNP (Rust).

    This class:
      - performs eigen-decomposition of K once (np.linalg.eigh)
      - precomputes rotated phenotype/covariates
      - runs per-chunk REML via Rust kernel

    Parameters
    ----------
    y : np.ndarray
        Phenotype (n,) or (n,1). Internally used as (n,1).

    X : np.ndarray or None
        Covariates (n,p). If provided, an intercept column is added.

    kinship : np.ndarray
        Kinship matrix K of shape (n,n). A small ridge is added:
            K + 1e-6 * I
        to improve numerical stability.

    Attributes
    ----------
    S : np.ndarray, shape (n,)
        Eigenvalues of K (descending).

    Dh : np.ndarray, shape (n,n), dtype=float32
        U^T of eigenvectors (transposed), used to rotate.

    Xcov : np.ndarray, shape (n,q)
        Rotated covariates Dh @ X.

    y : np.ndarray, shape (n,1)
        Rotated phenotype Dh @ y.

    bounds : tuple
        log10(lambda) search bounds, centered around null estimate.
    """

    def __init__(self, y: np.ndarray, X: Optional[np.ndarray], kinship: np.ndarray):
        y = np.asarray(y).reshape(-1, 1)  # ensure (n,1)

        # Add intercept automatically
        X = (
            np.concatenate([np.ones((y.shape[0], 1)), X], axis=1)
            if X is not None
            else np.ones((y.shape[0], 1))
        )

        # Eigen decomposition of kinship (stabilized)
        kinship.flat[::kinship.shape[0]+1] += 1e-6
        t_start = time.time()
        self.S, self.Dh = eigh(kinship, overwrite_a=True, check_finite=False)
        self.evd_secs = float(time.time() - t_start)
        # Drop kinship to save memory
        del kinship
        self.Dh = self.Dh.T.astype('float32')

        # Pre-rotate covariates and phenotype once
        self.Xcov:np.ndarray = self.Dh @ X
        self.y:np.ndarray = self.Dh @ y

        # ---- Estimate null lambda via scalar optimization (Python) ----
        lbd_null, ml0, reml = lmm_reml_null(self.S, self.Xcov, self.y, (-5, 5), max_iter=50, tol=1e-3)
        # print(lbd_null,reml)
        # result = minimize_scalar(
        #     lambda lbd: -self._NULLREML(10 ** (lbd)),
        #     bounds=(-5, 5),
        #     method="bounded",
        #     options={"xatol": 1e-3},
        # )
        # print(10**result.x,-result.fun)
        # lbd_null = 10 ** (result.x)

        vg_null = np.mean(self.S)
        pve = vg_null / (vg_null + lbd_null)

        self.lbd_null = lbd_null
        self.pve = pve
        self.LL0 = reml
        self.ML0 = ml0
        # Adaptive bounds around null (if PVE not degenerate)
        if pve > 0.95 or pve < 0.05:
            self.bounds = (-5, 5)
        else:
            self.bounds = (np.log10(lbd_null) - 2, np.log10(lbd_null) + 2)

    def _NULLREML(self, lbd: float) -> float:
        """
        Restricted Maximum Likelihood (REML) for the null model (no SNP effect).

        Parameters
        ----------
        lbd : float
            Lambda parameter (typically ve/vg).

        Returns
        -------
        ll : float
            Null REML log-likelihood (higher is better).
        """
        try:
            n, p_cov = self.Xcov.shape
            p = p_cov

            V = self.S + lbd
            V_inv = 1.0 / V

            X_cov = self.Xcov

            # Efficiently compute:
            #   X^T V^-1 X   and   X^T V^-1 y
            XTV_invX = (V_inv * X_cov.T) @ X_cov
            XTV_invy = (V_inv * X_cov.T) @ self.y

            beta = np.linalg.solve(XTV_invX, XTV_invy)
            r = self.y - X_cov @ beta

            rTV_invr = (V_inv * r.T @ r)[0, 0]
            log_detV = np.sum(np.log(V))

            sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
            total_log = (n - p) * np.log(rTV_invr) + log_detV + log_detXTV_invX

            # Constant term (matches your original expression)
            c = (n - p) * (np.log(n - p) - 1 - np.log(2 * np.pi)) / 2.0
            return c - total_log / 2.0

        except Exception as e:
            print(f"REML error: {e}, lbd={lbd}")
            return -1e8

    def gwas(self, snp: np.ndarray, threads: int = 1) -> np.ndarray:
        """
        Run LMM GWAS on a SNP-major genotype matrix/chunk.

        Parameters
        ----------
        snp : np.ndarray, shape (m, n)
            SNP-major genotype block. Rows SNPs, columns samples.

        threads : int
            Rust worker threads for per-SNP REML optimization.

        Returns
        -------
        beta_se_p : np.ndarray, shape (m, 4)
            Per-SNP beta, se, pwald, plrt.
        """
        g_rot_chunk = snp @ self.Dh.T
        beta_se_p = lmm_reml(
            self.S,
            self.Xcov,
            self.y,
            g_rot_chunk,
            self.bounds,
            max_iter=30,
            tol=1e-2,
            threads=threads,
            nullml=self.ML0,
        )
        return beta_se_p


class FastLMM(LMM):
    """
    Fast LMM GWAS using a fixed lambda for all SNPs (Rust kernel).
    """

    def gwas(self, snp: np.ndarray, threads: int = 1) -> np.ndarray:
        if self.pve < 0.05 or self.pve > 0.95:
            return super().gwas(snp, threads=threads)

        log10_lbd = float(np.log10(self.lbd_null))
        g_rot_chunk = snp @ self.Dh.T
        beta_se_p = lmm_assoc_fixed(
            self.S,
            self.Xcov,
            self.y,
            log10_lbd,
            g_rot_chunk,
            threads=threads,
            nullml=self.ML0,
        )
        return beta_se_p


class LM:
    """
    Simple linear model GWAS wrapper using the Rust FEM kernel.

    This is the non-kinship (no random effect) version.
    """

    def __init__(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        self.y = np.asarray(y).reshape(-1, 1)
        self.X = (
            np.concatenate([np.ones((self.y.shape[0], 1)), X], axis=1)
            if X is not None
            else np.ones((self.y.shape[0], 1))
        )
        # Cache (X'X)^(-1) for repeated chunk scans on the same trait.
        self.ixx = np.ascontiguousarray(np.linalg.pinv(self.X.T @ self.X), dtype=np.float64)

    def gwas(self, snp: np.ndarray, threads: int = 1) -> np.ndarray:
        """
        Run LM GWAS on SNP-major genotype matrix.

        Parameters
        ----------
        snp : np.ndarray, shape (m, n)
            SNP-major genotype matrix/block.

        threads : int
            Rust worker threads.

        Returns
        -------
        beta_se_p : np.ndarray, shape (m, 3)
            Columns: beta, se, p (as you slice in the original code).
        """
        beta_se_p = FEM(
            self.y,
            self.X,
            snp,
            snp.shape[0],
            threads,
            ixx=self.ixx,
        )[:, [0, 1, -1]]
        return beta_se_p


# ----------------------------
# FarmCPU helper utilities
# ----------------------------

def REM(sz, n, pvalue, pos, M, y, X, sample_indices: Optional[np.ndarray] = None):
    """
    One REM (Random Effect Model) step used by FarmCPU to pick lead SNPs.

    This selects n lead SNPs by:
      1) binning SNPs by pos // sz
      2) within each bin keeping the most significant SNP
      3) selecting top-n leads by p-value
      4) fitting ll() on those SNPs to compute model score

    Returns
    -------
    score : float
        -2 * log-likelihood (smaller is better)

    leadidx : np.ndarray
        Indices of selected lead SNPs
    """
    bin_id = pos // sz
    order = np.lexsort((pvalue, bin_id))  # sort by bin, then pvalue
    lead = order[np.concatenate(([True], bin_id[order][1:] != bin_id[order][:-1]))]
    leadidx = np.sort(lead[np.argsort(pvalue[lead])[:n]])

    lead_rows = _select_snp_rows(M, leadidx, sample_indices=sample_indices)
    results = ll(y, lead_rows.T, X)
    return -2 * results["LL"], leadidx


def _pinv_safe(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Numerically safe pseudo-inverse wrapper."""
    return np.linalg.pinv(A, rcond=rcond)


def _solve_beta_system(
    A: np.ndarray,
    b: np.ndarray,
    *,
    pinv_rcond: float = 1e-12,
) -> np.ndarray:
    """
    Solve A x = b via fast/stable path:
      1) Cholesky (with small ridge retries),
      2) fallback to pseudo-inverse.
    """
    p = int(A.shape[0])
    eye = np.eye(p, dtype=A.dtype)
    for ridge in (0.0, 1e-10, 1e-8, 1e-6):
        try:
            A_use = A if ridge == 0.0 else (A + ridge * eye)
            c, lower = cho_factor(A_use, overwrite_a=False, check_finite=False)
            return cho_solve((c, lower), b, check_finite=False)
        except Exception:
            continue
    return _pinv_safe(A, rcond=pinv_rcond) @ b


def ll(
    pheno: np.ndarray,
    snp_pool: np.ndarray,
    X0: np.ndarray | None = None,
    deltaExpStart: float = -5.0,
    deltaExpEnd: float = 5.0,
    delta_step: float = 0.1,
    svd_eps: float = 1e-8,
    pinv_rcond: float = 1e-12,
):
    """
    Python rewrite of FaST-LMM likelihood under a grid-search over delta.

    Parameters
    ----------
    pheno : np.ndarray, shape (n,) or (n,1)
        Phenotype vector y.

    snp_pool : np.ndarray, shape (n, k)
        Pseudo-QTN matrix (samples x k). No missing, consistent sample order.

    X0 : np.ndarray, shape (n,p), optional
        Covariates. If None, intercept-only.

    deltaExpStart/deltaExpEnd/delta_step : float
        Grid in exp-space: delta = exp(grid).

    svd_eps : float
        Keep singular values > svd_eps.

    pinv_rcond : float
        rcond used in pseudo-inverse for stability.

    Returns
    -------
    dict
        Keys: beta, delta, LL, vg, ve
    """
    # ---- Normalize shapes ----
    y = np.asarray(pheno, dtype=np.float64).reshape(-1, 1)

    snp_pool = np.asarray(snp_pool, dtype=np.float64)
    if snp_pool.ndim == 1:
        snp_pool = snp_pool.reshape(-1, 1)

    n = snp_pool.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"pheno n={y.shape[0]} != snp_pool n={n}")

    # If any SNP has 0 variance, delta search degenerates in original logic
    if snp_pool.size > 0:
        v = np.var(snp_pool, axis=0, ddof=1)
        if np.any(v == 0):
            deltaExpStart = 100.0
            deltaExpEnd = 100.0

    X = np.ones((n, 1), dtype=np.float64) if X0 is None else np.asarray(X0, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != n:
        raise ValueError(f"X0 n={X.shape[0]} != snp_pool n={n}")

    # ---- SVD of snp_pool ----
    U, s, _Vt = np.linalg.svd(snp_pool, full_matrices=False)
    keep = s > svd_eps
    s = s[keep]
    if s.size == 0:
        U1 = np.zeros((n, 0), dtype=np.float64)
        d = np.zeros((0,), dtype=np.float64)
    else:
        d = s**2
        U1 = U[:, keep]

    r = U1.shape[1]

    # Precompute projections
    U1TX = U1.T @ X
    U1TY = U1.T @ y

    yU1TY = y - (U1 @ U1TY)
    XU1TX = X - (U1 @ U1TX)

    IU = -(U1 @ U1.T)
    IU[np.diag_indices(n)] += 1.0

    IUX = IU.T @ X
    IUY = IU.T @ y

    delta_range = np.arange(deltaExpStart, deltaExpEnd + 1e-12, delta_step, dtype=np.float64)

    best_LL = -np.inf
    best_beta = None
    best_delta = None

    p = X.shape[1]

    for expv in delta_range:
        delta = float(np.exp(expv))

        if r > 0:
            w = 1.0 / (d + delta)
            beta1 = (U1TX.T * w) @ U1TX
            beta3 = (U1TX.T * w) @ U1TY
            part12 = float(np.sum(np.log(d + delta)))
        else:
            beta1 = np.zeros((p, p), dtype=np.float64)
            beta3 = np.zeros((p, 1), dtype=np.float64)
            part12 = 0.0

        beta2 = (IUX.T @ IUX) / delta
        beta4 = (IUX.T @ IUY) / delta

        beta = _solve_beta_system(
            beta1 + beta2,
            beta3 + beta4,
            pinv_rcond=pinv_rcond,
        )

        part11 = n * np.log(2.0 * np.pi)
        part13 = (n - r) * np.log(delta)
        part1 = -0.5 * (part11 + part12 + part13)

        if r > 0:
            resid_u = U1TY - (U1TX @ beta)
            part221 = float(np.sum((resid_u[:, 0] ** 2) / (d + delta)))
        else:
            part221 = 0.0

        resid_i = yU1TY - (XU1TX @ beta)
        part222 = float(np.sum(resid_i[:, 0] ** 2) / delta)

        part2 = -0.5 * (n + n * np.log((part221 + part222) / n))
        LL = float(part1 + part2)

        if LL > best_LL:
            best_LL = LL
            best_beta = beta.copy()
            best_delta = delta

    beta = best_beta
    delta = best_delta
    LL = best_LL

    # vg / ve (as in your original logic)
    if r > 0:
        resid_u = U1TY - (U1TX @ beta)
        sigma_a1 = float(np.sum((resid_u[:, 0] ** 2) / (d + delta)))
    else:
        sigma_a1 = 0.0

    resid_i2 = IUY - (IUX @ beta)
    sigma_a2 = float(np.sum(resid_i2[:, 0] ** 2) / delta)

    sigma_a = (sigma_a1 + sigma_a2) / n
    sigma_e = delta * sigma_a

    return {"beta": beta, "delta": delta, "LL": LL, "vg": sigma_a, "ve": sigma_e}


def SUPER(corr: np.ndarray, pval: np.ndarray, thr: float = 0.7) -> np.ndarray:
    """
    LD-based de-redundancy for candidate QTNs (FarmCPU).

    Given a correlation matrix among candidate SNPs, keep only one SNP within
    each highly-correlated group, preferring the one with smaller p-value.

    Parameters
    ----------
    corr : np.ndarray, shape (k, k)
        Correlation matrix among k candidate QTNs.

    pval : array-like, shape (k,)
        P-values corresponding to those candidates.

    thr : float
        Correlation magnitude threshold. If |corr| >= thr, treat as redundant.

    Returns
    -------
    keep : np.ndarray, dtype=bool, shape (k,)
        Boolean mask for which candidates to keep.
    """
    nqtn = corr.shape[0]
    keep = np.ones(nqtn, dtype=np.bool_)

    for i in range(nqtn):
        if keep[i]:
            row = corr[i]
            pi = pval[i]
            for j in range(i + 1, nqtn):
                if keep[j]:
                    cij = row[j]
                    if cij >= thr or cij <= -thr:
                        # keep smaller p-value (more significant)
                        if pi >= pval[j]:
                            keep[i] = False
                        else:
                            keep[j] = False
                            break
    return keep


def farmcpu(
    y: np.ndarray,
    M: Any,
    X: Optional[np.ndarray],
    chrlist: np.ndarray,
    poslist: np.ndarray,
    szbin: list = [5e5, 5e6, 5e7],
    nbin: int = 5,
    QTNbound: Optional[int] = None,
    iter: int = 30,
    threshold: float = 0.05,
    threads: int = 1,
    fem_threads: Optional[int] = None,
    rem_jobs: Optional[int] = None,
    sample_indices: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    return_info: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    FarmCPU GWAS (Fixed and random model Circulating Probability Unification).

    This implementation uses:
      - Rust FEM kernel for fast fixed-effect scanning
      - Packed and dense paths: Rust REM/ll/SUPER helpers when available
      - Python REM/SUPER fallback if Rust helper symbols are unavailable

    Parameters
    ----------
    y : np.ndarray, shape (n,) or (n,1)
        Phenotype.

    M : np.ndarray or packed tuple/dict
        Either:
        - Dense SNP-major genotype matrix with shape (m, n), or
        - BED-packed payload from `janusx.gfreader.load_bed_2bit_packed(...)`
          (tuple or dict with packed/maf/n_samples).

    X : np.ndarray or None, shape (n, p)
        Covariates. Intercept is always added internally.

    chrlist : np.ndarray, shape (m,)
        Chromosome label per SNP.

    poslist : np.ndarray, shape (m,)
        Position per SNP (integer-like).

    szbin : list of float
        Bin sizes for selecting lead SNPs (in bp). Multiple values form a grid.

    nbin : int
        Number of candidate bin counts to try, derived from QTNbound.

    QTNbound : int or None
        Maximum number of QTNs (pseudo-QTNs) allowed. If None, uses:
            int(sqrt(n / log10(n)))

    iter : int
        Maximum FarmCPU iterations.

    threshold : float
        Significance level used to decide if any SNP enters candidate set.
        Uses Bonferroni-like criterion: threshold / m.

    threads : int
        If -1, use all CPU cores. Acts as default thread budget.
    fem_threads : int or None
        Number of threads for Rust FEM kernels. None -> use `threads`.
    rem_jobs : int or None
        Number of joblib workers for REM grid-search. None -> use `threads`
        but capped by current REM task count.
    sample_indices : np.ndarray or None, shape (n_used,)
        Optional genotype sample indices used only for packed input.
        If provided, `y` and `X` must already be aligned to this index order.

    progress_cb : callable or None
        Optional callback receiving `(done_iter, total_iter)` to report
        FarmCPU iteration progress to external UIs (e.g. Rich progress bars).
    return_info : bool, default False
        If True, return `(out, info)` where `info` includes summary metadata
        such as the final pseudo-QTN count.

    Returns
    -------
    out : np.ndarray, shape (m, 3)
        Columns: beta, se, p (final iteration).
        P-values for selected QTNs are replaced by their covariate-min p-values.
    (out, info) : tuple, optional
        Returned when `return_info=True`.
        `info["n_pseudo_qtn"]` is the number of final pseudo-QTNs.
    """
    threads = cpu_count() if threads == -1 else int(threads)
    threads = max(1, threads)
    fem_threads = threads if fem_threads is None else int(fem_threads)
    fem_threads = max(1, fem_threads)
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1))
    packed_ctx = _coerce_bed_packed_ctx(M)
    if packed_ctx is None:
        M_work = np.asarray(M, dtype=np.float32)
        if M_work.ndim != 2:
            raise ValueError("M must be 2D (m, n) for dense FarmCPU input.")
        if sample_indices is not None:
            sidx = _normalize_sample_indices(sample_indices, int(M_work.shape[1]), int(y.shape[0]))
            M_work = np.ascontiguousarray(M_work[:, sidx], dtype=np.float32)
        else:
            if int(M_work.shape[1]) != int(y.shape[0]):
                raise ValueError(
                    f"M sample size mismatch: M.shape[1]={M_work.shape[1]} but len(y)={y.shape[0]}"
                )
            M_work = np.ascontiguousarray(M_work, dtype=np.float32)
        packed_sample_idx = None
        m, n = M_work.shape
    else:
        M_work = packed_ctx
        m = int(M_work["packed"].shape[0])
        packed_sample_idx = _normalize_sample_indices(
            sample_indices,
            int(M_work["n_samples"]),
            int(y.shape[0]),
        )
        if packed_sample_idx is None:
            if int(M_work["n_samples"]) != int(y.shape[0]):
                raise ValueError(
                    f"Packed n_samples={M_work['n_samples']} does not match len(y)={y.shape[0]}. "
                    "Provide sample_indices to align samples."
                )
            n = int(M_work["n_samples"])
        else:
            n = int(packed_sample_idx.shape[0])

    # Map chromosome labels to integer blocks for global ordering
    chrlist = np.asarray(chrlist)
    poslist = np.asarray(poslist, dtype=np.int64)
    _, chr_idx = np.unique(chrlist, return_inverse=True)

    # "global position" = pos + chr_block * 1e12 (avoids chr collisions)
    pos = np.ascontiguousarray(
        poslist + chr_idx.astype(np.int64) * 1_000_000_000_000,
        dtype=np.int64,
    )

    if QTNbound is None:
        QTNbound = int(np.sqrt(n / np.log10(n)))

    szbin = np.array(szbin)
    nbin_den = max(1, int(nbin))
    nbin_step = max(1, int(QTNbound // nbin_den))
    nbin = np.array(range(nbin_step, QTNbound + 1, nbin_step))
    if nbin.size == 0:
        nbin = np.array([int(QTNbound)], dtype=int)

    # Add intercept
    X = np.concatenate([np.ones((y.shape[0], 1)), X], axis=1) if X is not None else np.ones((y.shape[0], 1))

    QTNidx = np.array([], dtype=int)

    for i_iter in range(int(iter)):
        if QTNidx.size > 0:
            qtn_rows = _select_snp_rows(
                M_work,
                QTNidx,
                sample_indices=packed_sample_idx,
            )
            X_QTN = np.concatenate([X, qtn_rows.T], axis=1)
        else:
            X_QTN = X

        FEMresult = FEM(
            y,
            X_QTN,
            M_work,
            threads=fem_threads,
            sample_indices=packed_sample_idx,
        )
        FEMresult[:, 2:] = np.nan_to_num(FEMresult[:, 2:], nan=1)

        # p-values of pseudo-QTNs as covariates
        QTNpval = FEMresult[:, 2 + X.shape[1] : -1].min(axis=0)

        # last column = p for all SNPs
        FEMp = FEMresult[:, -1]
        FEMp[QTNidx] = QTNpval

        # Stop if no SNP passes threshold
        if np.sum(FEMp <= threshold / m) == 0:
            if progress_cb is not None:
                progress_cb(i_iter + 1, int(iter))
            break

        # Build grid tasks for REM
        combine_list = [(sz, n_) for sz in szbin for n_ in nbin]
        rem_jobs_i = threads if rem_jobs is None else int(rem_jobs)
        rem_jobs_i = max(1, min(int(len(combine_list)), int(rem_jobs_i)))
        use_rust_dense_rem = packed_ctx is None and _farmcpu_rem_dense is not None
        use_rust_packed_rem = packed_ctx is not None and _farmcpu_rem_packed is not None
        use_rust_dense_super = packed_ctx is None and _farmcpu_super_dense is not None
        use_rust_packed_super = packed_ctx is not None and _farmcpu_super_packed is not None
        blas_guard = (
            _threadpool_limits(limits=1)
            if _threadpool_limits is not None
            else nullcontext()
        )

        with blas_guard:
            if use_rust_packed_rem:
                femp_contig = np.ascontiguousarray(FEMp, dtype=np.float64)
                xqtn_contig = np.ascontiguousarray(X_QTN, dtype=np.float64)
                REMresult = Parallel(
                    n_jobs=rem_jobs_i,
                    prefer="threads",
                )(
                    delayed(_farmcpu_rem_packed)(
                        int(sz),
                        int(n_),
                        femp_contig,
                        pos,
                        M_work["packed"],
                        int(M_work["n_samples"]),
                        M_work["row_flip"],
                        M_work["maf"],
                        y,
                        xqtn_contig,
                        packed_sample_idx,
                    )
                    for sz, n_ in combine_list
                )
            elif use_rust_dense_rem:
                femp_contig = np.ascontiguousarray(FEMp, dtype=np.float64)
                xqtn_contig = np.ascontiguousarray(X_QTN, dtype=np.float64)
                REMresult = Parallel(
                    n_jobs=rem_jobs_i,
                    prefer="threads",
                )(
                    delayed(_farmcpu_rem_dense)(
                        int(sz),
                        int(n_),
                        femp_contig,
                        pos,
                        M_work,
                        y,
                        xqtn_contig,
                    )
                    for sz, n_ in combine_list
                )
            else:
                REMresult = Parallel(
                    n_jobs=rem_jobs_i,
                    prefer="threads",
                )(
                    delayed(REM)(
                        sz,
                        n_,
                        FEMp,
                        pos,
                        M_work,
                        y,
                        X_QTN,
                        sample_indices=packed_sample_idx,
                    )
                    for sz, n_ in combine_list
                )

        optcombidx = int(np.argmin([l for l, _idx in REMresult]))
        opt_lead = np.asarray(REMresult[optcombidx][1], dtype=int)
        QTNidx_pre = np.unique(np.concatenate([opt_lead, QTNidx]))
        if use_rust_packed_super:
            keep = np.asarray(
                _farmcpu_super_packed(
                    np.ascontiguousarray(QTNidx_pre, dtype=np.int64),
                    np.ascontiguousarray(FEMp[QTNidx_pre], dtype=np.float64),
                    M_work["packed"],
                    int(M_work["n_samples"]),
                    M_work["row_flip"],
                    M_work["maf"],
                    packed_sample_idx,
                ),
                dtype=np.bool_,
            )
        elif use_rust_dense_super:
            keep = np.asarray(
                _farmcpu_super_dense(
                    np.ascontiguousarray(QTNidx_pre, dtype=np.int64),
                    np.ascontiguousarray(FEMp[QTNidx_pre], dtype=np.float64),
                    M_work,
                ),
                dtype=np.bool_,
            )
        else:
            corr_rows = _select_snp_rows(
                M_work,
                QTNidx_pre,
                sample_indices=packed_sample_idx,
            )
            keep = SUPER(np.corrcoef(corr_rows), FEMp[QTNidx_pre])
        QTNidx_pre = QTNidx_pre[keep]

        if np.array_equal(QTNidx_pre, QTNidx):
            if progress_cb is not None:
                progress_cb(i_iter + 1, int(iter))
            break
        QTNidx = QTNidx_pre
        if progress_cb is not None:
            progress_cb(i_iter + 1, int(iter))
    # Final scan with final QTN set
    if QTNidx.size > 0:
        qtn_rows = _select_snp_rows(
            M_work,
            QTNidx,
            sample_indices=packed_sample_idx,
        )
        X_QTN = np.concatenate([X, qtn_rows.T], axis=1)
    else:
        X_QTN = X
    FEMresult = FEM(
        y,
        X_QTN,
        M_work,
        threads=fem_threads,
        sample_indices=packed_sample_idx,
    )
    FEMresult[:, 2:] = np.nan_to_num(FEMresult[:, 2:], nan=1)

    QTNpval = FEMresult[:, 2 + X.shape[1] : -1].min(axis=0)

    beta_se = FEMresult[:, [0, 1]]
    p = FEMresult[:, -1]
    p[QTNidx] = QTNpval

    # Sync beta/se for QTNs using the SNP rows that minimize each QTN p-value.
    if QTNidx.size > 0:
        qtn_cols = slice(2 + X.shape[1], -1)
        min_idx = np.nanargmin(FEMresult[:, qtn_cols], axis=0)
        M_sub = _select_snp_rows(
            M_work,
            min_idx,
            sample_indices=packed_sample_idx,
        )
        ixx = np.ascontiguousarray(np.linalg.pinv(X_QTN.T @ X_QTN), dtype=np.float64)
        full = glmf32_full(
            np.ascontiguousarray(y, dtype=np.float64).ravel(),
            np.ascontiguousarray(X_QTN, dtype=np.float64),
            ixx,
            M_sub,
            int(M_sub.shape[0]),
            int(fem_threads),
        )
        full = np.asarray(full)
        qtn_offset = X.shape[1]
        for j, qidx in enumerate(QTNidx):
            coef_idx = qtn_offset + j
            base = 3 * coef_idx
            beta_se[qidx, 0] = full[j, base]
            beta_se[qidx, 1] = full[j, base + 1]

    out = np.concatenate([beta_se, p.reshape(-1, 1)], axis=1)
    if return_info:
        return out, {"n_pseudo_qtn": int(QTNidx.size)}
    return out
