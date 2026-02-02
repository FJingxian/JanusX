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
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import eigh
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in",
)

from joblib import Parallel, delayed, cpu_count
from tqdm import trange

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


def FEM(
    y: np.ndarray,
    X: np.ndarray,
    M: np.ndarray,
    chunksize: int = 50_000,
    threads: int = 1,
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

    M : np.ndarray
        Genotype matrix in SNP-major layout: shape (m, n).
        Rows are SNPs, columns are samples.
        Internally coerced to contiguous float32.

    chunksize : int, default=50_000
        Number of SNPs processed per internal chunk in Rust.

        Practical note:
        - Larger chunks reduce overhead (Python <-> Rust calls)
        - Larger chunks increase peak memory traffic / cache pressure

    threads : int, default=1
        Number of Rust worker threads used inside glmf32.
        (This is *not* joblib threads; it's passed to Rust.)

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

    # Precompute (X'X)^(-1) in Python (matches your Rust signature)
    ixx = np.ascontiguousarray(np.linalg.pinv(X.T @ X), dtype=np.float64)

    if M.ndim != 2:
        raise ValueError("M must be a 2D array with shape (m, n) [SNP-major].")
    if M.shape[1] != y.shape[0]:
        raise ValueError(
            f"M must be shape (m, n). Got M.shape={M.shape}, but n=len(y)={y.shape[0]}"
        )

    result = []
    for start in range(0,M.shape[0],chunksize):
        end = min(start+chunksize,M.shape[0])
        # Genotypes as f32 for Rust SIMD / bandwidth-friendly kernels
        # ---- Call Rust kernel ----
        result.append(glmf32(y, X, ixx, np.ascontiguousarray(M[start:end], dtype=np.float32), int(chunksize), int(threads)))
    return np.concatenate(result,axis=0)


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
    )
    return lbd, ml, reml


def fastlmm_reml(
    S: np.ndarray,
    u1tx: np.ndarray,
    u2tx: np.ndarray,
    u1ty: np.ndarray,
    u2ty: np.ndarray,
    u1tsnp_chunk: np.ndarray,
    u2tsnp_chunk: np.ndarray,
    bounds: tuple,
    max_iter: int = 50,
    tol: float = 1e-2,
    threads: int = 4,
    nullml: Optional[float] = None,
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

    u1tsnp_chunk : np.ndarray, shape (m, k)
        Projected SNP chunk in U space (SNP-major).

    u2tsnp_chunk : np.ndarray, shape (m, n)
        Residual SNP chunk (SNP-major).

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

    u1tsnp_chunk = np.ascontiguousarray(u1tsnp_chunk, dtype=np.float32)
    u2tsnp_chunk = np.ascontiguousarray(u2tsnp_chunk, dtype=np.float32)

    beta_se_p = fastlmm_reml_chunk_f32(
        S,
        u1tx,
        u2tx,
        u1ty,
        u2ty,
        u1tsnp_chunk,
        u2tsnp_chunk,
        float(low),
        float(high),
        int(max_iter),
        float(tol),
        int(threads),
        None if nullml is None else float(nullml),
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
        print('Start Eigen-Decomposition...')
        t_start = time.time()
        self.S, self.Dh = eigh(kinship, overwrite_a=True, check_finite=False)
        print(f'EVD cost {(time.time()-t_start):.2f} secs')
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
        beta_se_p = FEM(self.y, self.X, snp, snp.shape[0], threads)[:, [0, 1, -1]]
        return beta_se_p


# ----------------------------
# FarmCPU helper utilities
# ----------------------------

def REM(sz, n, pvalue, pos, M, y, X):
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

    results = ll(y, M[leadidx].T, X)
    return -2 * results["LL"], leadidx


def _pinv_safe(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Numerically safe pseudo-inverse wrapper."""
    return np.linalg.pinv(A, rcond=rcond)


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

        zw1 = _pinv_safe(beta1 + beta2, rcond=pinv_rcond)
        beta = zw1 @ (beta3 + beta4)

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
    M: np.ndarray,
    X: Optional[np.ndarray],
    chrlist: np.ndarray,
    poslist: np.ndarray,
    szbin: list = [5e5, 5e6, 5e7],
    nbin: int = 5,
    QTNbound: Optional[int] = None,
    iter: int = 30,
    threshold: float = 0.05,
    threads: int = 1,
) -> np.ndarray:
    """
    FarmCPU GWAS (Fixed and random model Circulating Probability Unification).

    This implementation uses:
      - Rust FEM kernel for fast fixed-effect scanning
      - Python logic for binning / lead SNP selection / de-redundancy

    Parameters
    ----------
    y : np.ndarray, shape (n,) or (n,1)
        Phenotype.

    M : np.ndarray, shape (m, n)
        SNP-major genotype matrix (m SNPs, n samples).

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
        If -1, use all CPU cores.
        Used both for:
          - Rust FEM threads
          - joblib Parallel workers (outer loops)

    Returns
    -------
    out : np.ndarray, shape (m, 3)
        Columns: beta, se, p (final iteration).
        P-values for selected QTNs are replaced by their covariate-min p-values.
    """
    threads = cpu_count() if threads == -1 else int(threads)

    m, n = M.shape

    # Map chromosome labels to integer blocks for global ordering
    chrlist = np.asarray(chrlist)
    poslist = np.asarray(poslist, dtype=np.int64)
    _, chr_idx = np.unique(chrlist, return_inverse=True)

    # "global position" = pos + chr_block * 1e12 (avoids chr collisions)
    pos = poslist + chr_idx.astype(np.int64) * 1_000_000_000_000

    if QTNbound is None:
        QTNbound = int(np.sqrt(n / np.log10(n)))

    szbin = np.array(szbin)
    nbin = np.array(range(QTNbound // nbin, QTNbound + 1, QTNbound // nbin))

    # Add intercept
    X = np.concatenate([np.ones((y.shape[0], 1)), X], axis=1) if X is not None else np.ones((y.shape[0], 1))

    QTNidx = np.array([], dtype=int)

    for _ in trange(iter, desc="Process of FarmCPU", ascii=False):
        X_QTN = np.concatenate([X, M[QTNidx].T], axis=1) if QTNidx.size > 0 else X

        FEMresult = FEM(y, X_QTN, M, threads=threads)
        FEMresult[:, 2:] = np.nan_to_num(FEMresult[:, 2:], nan=1)

        # p-values of pseudo-QTNs as covariates
        QTNpval = FEMresult[:, 2 + X.shape[1] : -1].min(axis=0)

        # last column = p for all SNPs
        FEMp = FEMresult[:, -1]
        FEMp[QTNidx] = QTNpval

        # Stop if no SNP passes threshold
        if np.sum(FEMp <= threshold / m) == 0:
            break

        # Build grid tasks for REM
        combine_list = [(sz, n_) for sz in szbin for n_ in nbin]

        REMresult = Parallel(threads,
                            max_nbytes="1M",
                            mmap_mode="r",
                            temp_folder="/tmp/janusx-joblib",)(
            delayed(REM)(sz, n_, FEMp, pos, M, y, X_QTN) for sz, n_ in combine_list
        )

        optcombidx = int(np.argmin([l for l, _idx in REMresult]))
        QTNidx_pre = np.unique(np.concatenate([REMresult[optcombidx][1], QTNidx]))

        keep = SUPER(np.corrcoef(M[QTNidx_pre]), FEMp[QTNidx_pre])
        QTNidx_pre = QTNidx_pre[keep]

        if np.array_equal(QTNidx_pre, QTNidx):
            break
        QTNidx = QTNidx_pre
    print(QTNidx)
    # Final scan with final QTN set
    X_QTN = np.concatenate([X, M[QTNidx].T], axis=1)
    FEMresult = FEM(y, X_QTN, M, threads=threads)
    FEMresult[:, 2:] = np.nan_to_num(FEMresult[:, 2:], nan=1)

    QTNpval = FEMresult[:, 2 + X.shape[1] : -1].min(axis=0)

    beta_se = FEMresult[:, [0, 1]]
    p = FEMresult[:, -1]
    p[QTNidx] = QTNpval

    # Sync beta/se for QTNs using the SNP rows that minimize each QTN p-value.
    if QTNidx.size > 0:
        qtn_cols = slice(2 + X.shape[1], -1)
        min_idx = np.nanargmin(FEMresult[:, qtn_cols], axis=0)
        M_sub = np.ascontiguousarray(M[min_idx], dtype=np.float32)
        ixx = np.ascontiguousarray(np.linalg.pinv(X_QTN.T @ X_QTN), dtype=np.float64)
        full = glmf32_full(
            np.ascontiguousarray(y, dtype=np.float64).ravel(),
            np.ascontiguousarray(X_QTN, dtype=np.float64),
            ixx,
            M_sub,
            int(M_sub.shape[0]),
            int(threads),
        )
        full = np.asarray(full)
        qtn_offset = X.shape[1]
        for j, qidx in enumerate(QTNidx):
            coef_idx = qtn_offset + j
            base = 3 * coef_idx
            beta_se[qidx, 0] = full[j, base]
            beta_se[qidx, 1] = full[j, base + 1]

    return np.concatenate([beta_se, p.reshape(-1, 1)], axis=1)
