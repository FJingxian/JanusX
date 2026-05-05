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

import time
import os
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

_WARNED_EIGH_NON_LAPACK = False
_WARNED_EIGH_SCIPY_FALLBACK = False

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
    fastlmm_reml_chunk_f32,
    fastlmm_reml_null_f32,
    fastlmm_assoc_chunk_f32,
)

try:
    from janusx.janusx import (
        lmm_reml_chunk_from_snp_f32 as _lmm_reml_chunk_from_snp_f32,
        lmm_assoc_chunk_from_snp_f32 as _lmm_assoc_chunk_from_snp_f32,
    )
except Exception:
    _lmm_reml_chunk_from_snp_f32 = None
    _lmm_assoc_chunk_from_snp_f32 = None

try:
    from janusx.janusx import rust_eigh_from_array_f64 as _rust_eigh_from_array_f64
except Exception:
    _rust_eigh_from_array_f64 = None

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


def _infer_blas_threads_from_env() -> Optional[int]:
    """
    Best-effort parse of BLAS/OpenMP thread cap from common env vars.
    """
    for key in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        raw = os.environ.get(key, None)
        if raw is None:
            continue
        txt = str(raw).strip()
        if not txt:
            continue
        try:
            val = int(txt)
        except Exception:
            continue
        if val > 0:
            return val
    return None


def _lmm_from_snp_use_py_rotate(
    mode: str,
    m_chunk: int,
    n: int,
    threads: int,
) -> bool:
    """
    Decide whether from-SNP path should rotate in NumPy (BLAS) first.
    """
    m = str(mode).strip().lower()
    if m in {"py", "numpy", "python"}:
        return True
    if m in {"rust", "native"}:
        return False

    # Auto baseline: for large dense chunks, NumPy/BLAS rotation can be faster.
    prefer_py = (int(n) >= 512) and (int(m_chunk) >= 2048)
    if not prefer_py:
        return False

    # If scan stage pins BLAS to 1 thread while Rust uses multiple workers,
    # prefer Rust-side rotation+scan to avoid serial BLAS bottleneck.
    blas_threads = _infer_blas_threads_from_env()
    if (blas_threads is not None) and (blas_threads <= 1) and (int(threads) > 1):
        return False
    return True


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


def lmm_reml_from_snp(
    S: np.ndarray,
    utx: np.ndarray,
    uty: np.ndarray,
    snp_chunk: np.ndarray,
    u_t: np.ndarray,
    bounds: tuple,
    max_iter: int = 30,
    tol: float = 1e-2,
    threads: int = 4,
    nullml: Optional[float] = None,
    rotate_block_rows: int = 256,
) -> np.ndarray:
    """
    REML-based LMM scan on raw SNP chunk with Rust-side rotation + association.
    """
    mode = str(os.environ.get("JX_LMM_FROM_SNP_BACKEND", "auto")).strip().lower()
    m_chunk = int(np.asarray(snp_chunk).shape[0])
    n = int(np.asarray(snp_chunk).shape[1])
    use_py_rotate = _lmm_from_snp_use_py_rotate(
        mode=mode,
        m_chunk=m_chunk,
        n=n,
        threads=threads,
    )

    if (_lmm_reml_chunk_from_snp_f32 is None) or use_py_rotate:
        # Backward compatibility: extension not rebuilt yet.
        snp_chunk = np.ascontiguousarray(snp_chunk, dtype=np.float32)
        u_t = np.ascontiguousarray(u_t, dtype=np.float32)
        g_rot_chunk = snp_chunk @ u_t.T
        return lmm_reml(
            S,
            utx,
            uty,
            g_rot_chunk,
            bounds,
            max_iter=max_iter,
            tol=tol,
            threads=threads,
            nullml=nullml,
        )

    low, high = bounds
    S = np.ascontiguousarray(S, dtype=np.float64).ravel()
    utx = np.ascontiguousarray(utx, dtype=np.float64)
    uty = np.ascontiguousarray(uty, dtype=np.float64).ravel()
    snp_chunk = np.ascontiguousarray(snp_chunk, dtype=np.float32)
    u_t = np.ascontiguousarray(u_t, dtype=np.float32)
    beta_se_p = _lmm_reml_chunk_from_snp_f32(
        S,
        utx,
        uty,
        float(low),
        float(high),
        snp_chunk,
        u_t,
        int(max_iter),
        float(tol),
        int(threads),
        None if nullml is None else float(nullml),
        int(rotate_block_rows),
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


def lmm_assoc_fixed_from_snp(
    S: np.ndarray,
    utx: np.ndarray,
    uty: np.ndarray,
    log10_lbd: float,
    snp_chunk: np.ndarray,
    u_t: np.ndarray,
    threads: int = 4,
    nullml: Optional[float] = None,
    rotate_block_rows: int = 512,
) -> np.ndarray:
    """
    Fixed-lambda LMM scan on raw SNP chunk with Rust-side rotation + association.
    """
    mode = str(os.environ.get("JX_LMM_FROM_SNP_BACKEND", "auto")).strip().lower()
    m_chunk = int(np.asarray(snp_chunk).shape[0])
    n = int(np.asarray(snp_chunk).shape[1])
    use_py_rotate = _lmm_from_snp_use_py_rotate(
        mode=mode,
        m_chunk=m_chunk,
        n=n,
        threads=threads,
    )

    if (_lmm_assoc_chunk_from_snp_f32 is None) or use_py_rotate:
        # Backward compatibility: extension not rebuilt yet.
        snp_chunk = np.ascontiguousarray(snp_chunk, dtype=np.float32)
        u_t = np.ascontiguousarray(u_t, dtype=np.float32)
        g_rot_chunk = snp_chunk @ u_t.T
        return lmm_assoc_fixed(
            S,
            utx,
            uty,
            log10_lbd,
            g_rot_chunk,
            threads=threads,
            nullml=nullml,
        )

    S = np.ascontiguousarray(S, dtype=np.float64).ravel()
    Xcov = np.ascontiguousarray(utx, dtype=np.float64)
    y_rot = np.ascontiguousarray(uty, dtype=np.float64).ravel()
    snp_chunk = np.ascontiguousarray(snp_chunk, dtype=np.float32)
    u_t = np.ascontiguousarray(u_t, dtype=np.float32)
    beta_se_p = _lmm_assoc_chunk_from_snp_f32(
        S,
        Xcov,
        y_rot,
        float(log10_lbd),
        snp_chunk,
        u_t,
        int(threads),
        None if nullml is None else float(nullml),
        int(rotate_block_rows),
    )
    return beta_se_p


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
        global _WARNED_EIGH_NON_LAPACK, _WARNED_EIGH_SCIPY_FALLBACK
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
        eig_done = False
        if _rust_eigh_from_array_f64 is not None:
            try:
                eig_thr = int(_infer_blas_threads_from_env() or 0)
                eval_raw, evec_raw, _blas_backend, _evd_backend, _n, _tb, _ti, _ta, _lapack, _sec = (
                    _rust_eigh_from_array_f64(
                        np.ascontiguousarray(kinship, dtype=np.float64),
                        threads=eig_thr,
                        driver="auto",
                        jobz="V",
                        require_lapack=False,
                    )
                )
                if evec_raw is None:
                    raise RuntimeError("rust_eigh_from_array_f64 returned no eigenvectors")
                self.S = np.asarray(eval_raw, dtype=np.float64)
                self.Dh = np.asarray(evec_raw, dtype=np.float64)
                if not str(_evd_backend).lower().startswith("lapack_"):
                    if not _WARNED_EIGH_NON_LAPACK:
                        warnings.warn(
                            (
                                "Rust eigh did not use LAPACK backend "
                                f"(backend={_evd_backend}); performance may degrade."
                            ),
                            RuntimeWarning,
                        )
                        _WARNED_EIGH_NON_LAPACK = True
                eig_done = True
            except Exception as ex:
                if not _WARNED_EIGH_SCIPY_FALLBACK:
                    warnings.warn(
                        (
                            "Rust eigh failed; fallback to scipy.linalg.eigh "
                            f"(reason: {ex})."
                        ),
                        RuntimeWarning,
                    )
                    _WARNED_EIGH_SCIPY_FALLBACK = True
                eig_done = False
        if not eig_done:
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

    @classmethod
    def from_lmm(cls, other: "LMM") -> "LMM":
        """
        Create a lightweight model clone that reuses precomputed EVD/rotations.
        """
        obj = cls.__new__(cls)
        obj.S = other.S
        obj.Dh = other.Dh
        obj.Xcov = other.Xcov
        obj.y = other.y
        obj.lbd_null = other.lbd_null
        obj.pve = other.pve
        obj.LL0 = other.LL0
        obj.ML0 = other.ML0
        obj.bounds = other.bounds
        obj.evd_secs = float(getattr(other, "evd_secs", 0.0))
        return obj

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
        beta_se_p = lmm_reml_from_snp(
            self.S,
            self.Xcov,
            self.y,
            snp,
            self.Dh,
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
        beta_se_p = lmm_assoc_fixed_from_snp(
            self.S,
            self.Xcov,
            self.y,
            log10_lbd,
            snp,
            self.Dh,
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
    return_info: Union[bool, str] = False,
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
    return_info : bool or str, default False
        Controls optional metadata return:
          - False: return only `out`
          - True: return `(out, info)` with summary metadata
          - "trace": return `(out, info)` and include per-iteration trace under
            `info["trace"]` for debugging/analysis.

    Returns
    -------
    out : np.ndarray, shape (m, 3)
        Columns: beta, se, p (final iteration).
        P-values for selected QTNs are replaced by their covariate-min p-values.
    (out, info) : tuple, optional
        Returned when `return_info` requests metadata (e.g., `True` or `"trace"`).
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

    if isinstance(return_info, str):
        mode = return_info.strip().lower()
        if mode in ("", "0", "false", "off", "no", "none"):
            need_info = False
            collect_trace = False
        elif mode in ("1", "true", "on", "yes", "summary"):
            need_info = True
            collect_trace = False
        elif mode in ("trace", "full", "debug"):
            need_info = True
            collect_trace = True
        else:
            raise ValueError(
                "return_info must be bool or one of: "
                "'summary' / 'trace' / 'full' / 'debug' / "
                "'true' / 'false'"
            )
    else:
        need_info = bool(return_info)
        collect_trace = False

    QTNidx = np.array([], dtype=int)
    qtn_threshold_eff = float(threshold / m)
    trace_records: list[Dict[str, Any]] = []

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
            if collect_trace:
                trace_records.append(
                    {
                        "iter": int(i_iter + 1),
                        "status": "stop_no_signal",
                        "opt_bin_size": None,
                        "opt_n_qtn_grid": None,
                        "n_candidate_leads": int(QTNidx.size),
                        "n_pseudo_qtn": int(QTNidx.size),
                        "qtn_idx": [int(x) for x in np.asarray(QTNidx, dtype=np.int64).tolist()],
                    }
                )
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
        opt_sz, opt_n = combine_list[optcombidx]
        qtnidx_union_pre_super = np.unique(np.concatenate([opt_lead, QTNidx]))
        # Apply p-threshold gate on the unioned pseudo-QTN candidates.
        # From the second iteration onward, retain previously selected
        # pseudo-QTNs even when their current p-values exceed the threshold.
        if qtnidx_union_pre_super.size > 0 and i_iter >= 1:
            qtn_p = np.asarray(FEMp[qtnidx_union_pre_super], dtype=np.float64)
            qmask = np.isfinite(qtn_p) & (qtn_p < qtn_threshold_eff)
            if QTNidx.size > 0:
                qmask = qmask | np.isin(qtnidx_union_pre_super, QTNidx)
            qtnidx_union_pre_super = qtnidx_union_pre_super[qmask]
        if qtnidx_union_pre_super.size == 0:
            QTNidx_pre = np.array([], dtype=int)
            keep = np.array([], dtype=np.bool_)
        elif use_rust_packed_super:
            keep = np.asarray(
                _farmcpu_super_packed(
                    np.ascontiguousarray(qtnidx_union_pre_super, dtype=np.int64),
                    np.ascontiguousarray(FEMp[qtnidx_union_pre_super], dtype=np.float64),
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
                    np.ascontiguousarray(qtnidx_union_pre_super, dtype=np.int64),
                    np.ascontiguousarray(FEMp[qtnidx_union_pre_super], dtype=np.float64),
                    M_work,
                ),
                dtype=np.bool_,
            )
        else:
            corr_rows = _select_snp_rows(
                M_work,
                qtnidx_union_pre_super,
                sample_indices=packed_sample_idx,
            )
            keep = SUPER(np.corrcoef(corr_rows), FEMp[qtnidx_union_pre_super])
        if qtnidx_union_pre_super.size > 0:
            QTNidx_pre = qtnidx_union_pre_super[keep]
        if collect_trace:
            rem_grid = []
            for (grid_sz, grid_n), (grid_score, grid_leadidx) in zip(combine_list, REMresult):
                rem_grid.append(
                    {
                        "bin_size": float(grid_sz),
                        "n_qtn_grid": int(grid_n),
                        "score": float(grid_score),
                        "lead_idx": [int(x) for x in np.asarray(grid_leadidx, dtype=np.int64).tolist()],
                    }
                )
            trace_records.append(
                {
                    "iter": int(i_iter + 1),
                    "status": "converged_no_change" if np.array_equal(QTNidx_pre, QTNidx) else "updated",
                    "opt_bin_size": float(opt_sz),
                    "opt_n_qtn_grid": int(opt_n),
                    "n_candidate_leads": int(opt_lead.size),
                    "n_pseudo_qtn": int(QTNidx_pre.size),
                    "qtn_threshold": float(qtn_threshold_eff),
                    "opt_lead_idx": [int(x) for x in np.asarray(opt_lead, dtype=np.int64).tolist()],
                    "qtn_idx_before_super": [
                        int(x) for x in np.asarray(qtnidx_union_pre_super, dtype=np.int64).tolist()
                    ],
                    "super_keep_mask": [bool(x) for x in np.asarray(keep, dtype=np.bool_).tolist()],
                    "qtn_idx": [int(x) for x in np.asarray(QTNidx_pre, dtype=np.int64).tolist()],
                    "rem_grid": rem_grid,
                }
            )

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
    if need_info:
        info: Dict[str, Any] = {
            "n_pseudo_qtn": int(QTNidx.size),
            "qtn_idx": [int(x) for x in np.asarray(QTNidx, dtype=np.int64).tolist()],
        }
        if collect_trace:
            info["trace"] = trace_records
        return out, info
    return out
