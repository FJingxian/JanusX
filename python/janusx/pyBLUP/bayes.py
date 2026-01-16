from __future__ import annotations
from typing import Optional, Tuple
import typing
import numpy as np

from janusx.janusx import bayesa as _bayesa, bayesb as _bayesb, bayescpi as _bayescpi
from janusx.pyBLUP.mlm import BLUP


def _as_1d_f64(arr: np.ndarray, name: str) -> np.ndarray:
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 0:
        raise ValueError(f"{name} must be 1D array-like")
    return np.ascontiguousarray(out.reshape(-1))


def _as_2d_f64(
    arr: np.ndarray,
    name: str,
    n_rows: int,
    *,
    allow_1d: bool = False,
) -> np.ndarray:
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    out = np.asarray(arr, dtype=np.float64)
    if allow_1d and out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if out.shape[0] != n_rows:
        raise ValueError(f"{name} rows must match len(y)")
    return np.ascontiguousarray(out)

def _as_2d_f64_mxn(arr: np.ndarray, name: str, n_cols: int) -> np.ndarray:
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if out.shape[1] != n_cols:
        raise ValueError(f"{name} cols must match len(y)")
    return np.ascontiguousarray(out)


def _call_bayesa(
    y: np.ndarray,
    m: np.ndarray,
    x: Optional[np.ndarray],
    n_iter: int,
    burnin: int,
    thin: int,
    r2: float,
    df0_b: float,
    shape0: float,
    rate0: Optional[float],
    s0_b: Optional[float],
    df0_e: float,
    s0_e: Optional[float],
    min_abs_beta: float,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    n_iter = int(n_iter)
    burnin = int(burnin)
    thin = int(thin)
    if n_iter <= burnin:
        raise ValueError("n_iter must be > burnin")
    if thin < 1:
        raise ValueError("thin must be >= 1")
    if min_abs_beta <= 0.0:
        raise ValueError("min_abs_beta must be > 0")
    if not (0.0 < r2 < 1.0):
        raise ValueError("r2 must be in (0, 1)")
    if df0_b <= 0.0 or df0_e <= 0.0:
        raise ValueError("df0_b and df0_e must be > 0")
    if shape0 <= 0.0:
        raise ValueError("shape0 must be > 0")
    if rate0 is not None and rate0 <= 0.0:
        raise ValueError("rate0 must be > 0")
    if s0_b is not None and s0_b <= 0.0:
        raise ValueError("s0_b must be > 0")
    if s0_e is not None and s0_e <= 0.0:
        raise ValueError("s0_e must be > 0")
    if seed is not None:
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be >= 0")

    return _bayesa(
        y=y,
        m=m,
        x=x,
        n_iter=n_iter,
        burnin=burnin,
        thin=thin,
        r2=float(r2),
        df0_b=float(df0_b),
        shape0=float(shape0),
        rate0=rate0,
        s0_b=s0_b,
        df0_e=float(df0_e),
        s0_e=s0_e,
        min_abs_beta=float(min_abs_beta),
        seed=seed,
    )


def _call_bayesb(
    y: np.ndarray,
    m: np.ndarray,
    x: Optional[np.ndarray],
    n_iter: int,
    burnin: int,
    thin: int,
    r2: float,
    df0_b: float,
    shape0: float,
    rate0: Optional[float],
    s0_b: Optional[float],
    prob_in: float,
    counts: float,
    df0_e: float,
    s0_e: Optional[float],
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    n_iter = int(n_iter)
    burnin = int(burnin)
    thin = int(thin)
    if n_iter <= burnin:
        raise ValueError("n_iter must be > burnin")
    if thin < 1:
        raise ValueError("thin must be >= 1")
    if not (0.0 < r2 < 1.0):
        raise ValueError("r2 must be in (0, 1)")
    if df0_b <= 0.0 or df0_e <= 0.0:
        raise ValueError("df0_b and df0_e must be > 0")
    if shape0 <= 0.0:
        raise ValueError("shape0 must be > 0")
    if rate0 is not None and rate0 <= 0.0:
        raise ValueError("rate0 must be > 0")
    if s0_b is not None and s0_b <= 0.0:
        raise ValueError("s0_b must be > 0")
    if s0_e is not None and s0_e <= 0.0:
        raise ValueError("s0_e must be > 0")
    if not (0.0 < prob_in < 1.0):
        raise ValueError("prob_in must be in (0, 1)")
    if counts < 0.0:
        raise ValueError("counts must be >= 0")
    if seed is not None:
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be >= 0")

    return _bayesb(
        y=y,
        m=m,
        x=x,
        n_iter=n_iter,
        burnin=burnin,
        thin=thin,
        r2=float(r2),
        df0_b=float(df0_b),
        shape0=float(shape0),
        rate0=rate0,
        s0_b=s0_b,
        prob_in=float(prob_in),
        counts=float(counts),
        df0_e=float(df0_e),
        s0_e=s0_e,
        seed=seed,
    )


def _call_bayescpi(
    y: np.ndarray,
    m: np.ndarray,
    x: Optional[np.ndarray],
    n_iter: int,
    burnin: int,
    thin: int,
    r2: float,
    df0_b: float,
    s0_b: Optional[float],
    prob_in: float,
    counts: float,
    df0_e: float,
    s0_e: Optional[float],
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    n_iter = int(n_iter)
    burnin = int(burnin)
    thin = int(thin)
    if n_iter <= burnin:
        raise ValueError("n_iter must be > burnin")
    if thin < 1:
        raise ValueError("thin must be >= 1")
    if not (0.0 < r2 < 1.0):
        raise ValueError("r2 must be in (0, 1)")
    if df0_b <= 0.0 or df0_e <= 0.0:
        raise ValueError("df0_b and df0_e must be > 0")
    if s0_b is not None and s0_b <= 0.0:
        raise ValueError("s0_b must be > 0")
    if s0_e is not None and s0_e <= 0.0:
        raise ValueError("s0_e must be > 0")
    if not (0.0 < prob_in < 1.0):
        raise ValueError("prob_in must be in (0, 1)")
    if counts < 0.0:
        raise ValueError("counts must be >= 0")
    if seed is not None:
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be >= 0")

    return _bayescpi(
        y=y,
        m=m,
        x=x,
        n_iter=n_iter,
        burnin=burnin,
        thin=thin,
        r2=float(r2),
        df0_b=float(df0_b),
        s0_b=s0_b,
        prob_in=float(prob_in),
        counts=float(counts),
        df0_e=float(df0_e),
        s0_e=s0_e,
        seed=seed,
    )


def BayesA(
    y: np.ndarray,
    M: np.ndarray,
    X: Optional[np.ndarray] = None,
    n_iter: int = 400,
    burnin: int = 200,
    thin: int = 1,
    r2: float = 0.5,
    prob_in: float = 0.5,
    counts: float = 5.0,
    df0_b: float = 5.0,
    shape0: float = 1.1,
    rate0: Optional[float] = None,
    s0_b: Optional[float] = None,
    df0_e: float = 5.0,
    s0_e: Optional[float] = None,
    min_abs_beta: float = 1e-9,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Python interface for the Rust BayesA kernel (PyO3).

    This wrapper normalizes inputs to contiguous float64 arrays and passes
    them to the Rust implementation `janusx.janusx.bayesa`.

    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        Phenotype vector. Flattened to 1D float64.
    M : array-like, shape (m, n)
        Marker matrix with markers in rows and samples in columns.
    X : array-like, shape (n, q) or (n,), optional
        Covariate matrix. 1D inputs are treated as a single covariate.
        Include a column of ones here if you want an intercept term. If X is
        None, the Rust backend uses an intercept-only design.
    n_iter : int, default=200
        Total MCMC iterations.
    burnin : int, default=100
        Burn-in iterations. Must be < n_iter.
    thin : int, default=1
        Keep every `thin`-th sample after burn-in.
    r2 : float, default=0.5
        Proportion of variance explained by markers; must be in (0, 1).
    prob_in : float, default=0.5
        Unused for BayesA; kept for API parity.
    counts : float, default=5.0
        Unused for BayesA; kept for API parity.
    df0_b : float, default=5.0
        Prior degrees of freedom for marker effects.
    shape0 : float, default=1.1
        Prior shape parameter for the S update.
    rate0 : float, optional
        Prior rate; if None, computed from data and `shape0`.
    s0_b : float, optional
        Prior scale for marker effects; if None, computed from data.
    df0_e : float, default=5.0
        Prior degrees of freedom for residual variance.
    s0_e : float, optional
        Prior scale for residual variance; if None, derived from data.
    min_abs_beta : float, default=1e-9
        Lower bound on absolute effect size; prevents exact zeros.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    beta : np.ndarray, shape (p,)
        Posterior mean marker effects.
    alpha : np.ndarray, shape (q,)
        Posterior mean covariate effects. If `X` includes an intercept column,
        `alpha[0]` corresponds to the intercept.
    varbeta : np.ndarray, shape (p,)
        Posterior mean marker-specific variances.
    varep : float
        Posterior mean residual variance.
    h2_mean : float
        Posterior mean heritability.
    varh2 : float
        Posterior variance of heritability.

    Raises
    ------
    ValueError
        If shapes are incompatible or hyperparameters are out of range.

    Notes
    -----
    - Inputs are copied to contiguous float64 arrays before calling Rust.
    - M is expected to be (m, n) with n == len(y).
    """
    y_arr = _as_1d_f64(y, "y")
    m_arr = _as_2d_f64_mxn(M, "M", y_arr.shape[0])
    x_arr = None
    if X is not None:
        x_arr = _as_2d_f64(X, "X", y_arr.shape[0], allow_1d=True)

    return _call_bayesa(
        y_arr,
        m_arr,
        x_arr,
        n_iter,
        burnin,
        thin,
        r2,
        df0_b,
        shape0,
        rate0,
        s0_b,
        df0_e,
        s0_e,
        min_abs_beta,
        seed,
    )

def BayesB(
    y: np.ndarray,
    M: np.ndarray,
    X: Optional[np.ndarray] = None,
    n_iter: int = 400,
    burnin: int = 200,
    thin: int = 1,
    r2: float = 0.5,
    prob_in: float = 0.5,
    counts: float = 5.0,
    df0_b: float = 5.0,
    shape0: float = 1.1,
    rate0: Optional[float] = None,
    s0_b: Optional[float] = None,
    df0_e: float = 5.0,
    s0_e: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Python interface for the Rust BayesB kernel (PyO3).

    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        Phenotype vector. Flattened to 1D float64.
    M : array-like, shape (m, n)
        Marker matrix with markers in rows and samples in columns.
    X : array-like, shape (n, q) or (n,), optional
        Covariate matrix. 1D inputs are treated as a single covariate.
        Include a column of ones here if you want an intercept term. If X is
        None, the Rust backend uses an intercept-only design.
    n_iter : int, default=400
        Total MCMC iterations.
    burnin : int, default=200
        Burn-in iterations. Must be < n_iter.
    thin : int, default=1
        Keep every `thin`-th sample after burn-in.
    r2 : float, default=0.5
        Proportion of variance explained by markers; must be in (0, 1).
    prob_in : float, default=0.5
        Prior inclusion probability for markers.
    counts : float, default=5.0
        Prior strength for inclusion probability.
    df0_b : float, default=5.0
        Prior degrees of freedom for marker effects.
    shape0 : float, default=1.1
        Prior shape parameter for the S update.
    rate0 : float, optional
        Prior rate; if None, computed from data and `shape0`.
    s0_b : float, optional
        Prior scale for marker effects; if None, computed from data.
    df0_e : float, default=5.0
        Prior degrees of freedom for residual variance.
    s0_e : float, optional
        Prior scale for residual variance; if None, derived from data.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    beta : np.ndarray, shape (p,)
        Posterior mean marker effects.
    alpha : np.ndarray, shape (q,)
        Posterior mean covariate effects. If `X` includes an intercept column,
        `alpha[0]` corresponds to the intercept.
    varbeta : np.ndarray, shape (p,)
        Posterior mean marker-specific variances.
    varep : float
        Posterior mean residual variance.
    h2_mean : float
        Posterior mean heritability.
    varh2 : float
        Posterior variance of heritability.
    """
    y_arr = _as_1d_f64(y, "y")
    m_arr = _as_2d_f64_mxn(M, "M", y_arr.shape[0])
    x_arr = None
    if X is not None:
        x_arr = _as_2d_f64(X, "X", y_arr.shape[0], allow_1d=True)

    return _call_bayesb(
        y_arr,
        m_arr,
        x_arr,
        n_iter,
        burnin,
        thin,
        r2,
        df0_b,
        shape0,
        rate0,
        s0_b,
        prob_in,
        counts,
        df0_e,
        s0_e,
        seed,
    )


def BayesCpi(
    y: np.ndarray,
    M: np.ndarray,
    X: Optional[np.ndarray] = None,
    n_iter: int = 400,
    burnin: int = 200,
    thin: int = 1,
    r2: float = 0.5,
    prob_in: float = 0.5,
    counts: float = 10.0,
    df0_b: float = 5.0,
    s0_b: Optional[float] = None,
    df0_e: float = 5.0,
    s0_e: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Python interface for the Rust BayesCpi kernel (PyO3).

    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        Phenotype vector. Flattened to 1D float64.
    M : array-like, shape (m, n)
        Marker matrix with markers in rows and samples in columns.
    X : array-like, shape (n, q) or (n,), optional
        Covariate matrix. 1D inputs are treated as a single covariate.
        Include a column of ones here if you want an intercept term. If X is
        None, the Rust backend uses an intercept-only design.
    n_iter : int, default=400
        Total MCMC iterations.
    burnin : int, default=200
        Burn-in iterations. Must be < n_iter.
    thin : int, default=1
        Keep every `thin`-th sample after burn-in.
    r2 : float, default=0.5
        Proportion of variance explained by markers; must be in (0, 1).
    prob_in : float, default=0.5
        Prior inclusion probability for markers.
    counts : float, default=10.0
        Prior strength for inclusion probability.
    df0_b : float, default=5.0
        Prior degrees of freedom for marker effects.
    s0_b : float, optional
        Prior scale for marker effects; if None, computed from data.
    df0_e : float, default=5.0
        Prior degrees of freedom for residual variance.
    s0_e : float, optional
        Prior scale for residual variance; if None, derived from data.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    beta : np.ndarray, shape (p,)
        Posterior mean marker effects.
    alpha : np.ndarray, shape (q,)
        Posterior mean covariate effects. If `X` includes an intercept column,
        `alpha[0]` corresponds to the intercept.
    varbeta : float
        Posterior mean marker variance (shared across markers).
    varep : float
        Posterior mean residual variance.
    h2_mean : float
        Posterior mean heritability.
    varh2 : float
        Posterior variance of heritability.
    """
    y_arr = _as_1d_f64(y, "y")
    m_arr = _as_2d_f64_mxn(M, "M", y_arr.shape[0])
    x_arr = None
    if X is not None:
        x_arr = _as_2d_f64(X, "X", y_arr.shape[0], allow_1d=True)

    return _call_bayescpi(
        y_arr,
        m_arr,
        x_arr,
        n_iter,
        burnin,
        thin,
        r2,
        df0_b,
        s0_b,
        prob_in,
        counts,
        df0_e,
        s0_e,
        seed,
    )


class BAYES:
    def __init__(
        self,
        y: np.ndarray,
        M: np.ndarray,
        cov: np.ndarray | None = None,
        method: typing.Literal["BayesA", "BayesB", "BayesCpi"] = "BayesA",
        n_iter: int = 400,
        burnin: int = 200,
        thin: int = 1,
        r2: Optional[float] = None,
        prob_in: float = 0.5,
        counts: float = 5.0,
        seed: Optional[int] = None,
    ):
        """
        Bayesian genomic prediction using BayesA/B/Cpi with minimal hyperparameters.

        Parameters
        ----------
        y : np.ndarray
            Phenotype vector of shape (n, 1).
        M : np.ndarray
            Marker matrix of shape (m, n) with genotypes coded as 0/1/2.
        cov : np.ndarray, optional
            Fixed-effect design matrix of shape (n, p).
        method : {"BayesA","BayesB","BayesCpi"}
            Bayesian model to fit.
        r2 : float, optional
            Proportion of variance explained by markers. If None, estimated
            via GBLUP (BLUP with kinship=1).
        prob_in : float
            Prior inclusion probability (BayesB/BayesCpi).
        counts : float
            Prior strength for inclusion probability (BayesB/BayesCpi).

        Attributes
        ----------
        beta_hat : np.ndarray
            Posterior mean marker effects.
        alpha_hat : np.ndarray
            Posterior mean covariate effects.
        varbeta_hat : np.ndarray or float
            Posterior mean marker variances.
        varep_hat : float
            Posterior mean residual variance.
        h2_mean : float
            Posterior mean heritability.
        varh2 : float
            Posterior variance of heritability.
        """
        if r2 is None:
            model = BLUP(y, M, cov=cov, kinship=1)
            r2 = model.pve
        r2 = 0.05 if r2 <0.05 else r2; r2 = 0.95 if r2 >0.95 else r2 # optimize r2
        X = (
            np.concatenate([np.ones((M.shape[1], 1)), cov], axis=1)
            if cov is not None
            else np.ones((M.shape[1], 1))
        )
        y = y.reshape(-1, 1)

        self.method = method
        self.beta_hat: np.ndarray
        self.alpha_hat: np.ndarray
        self.varbeta_hat: np.ndarray | float | None = None
        self.varep_hat: float | None = None
        self.pve: float | None = None
        self.varpve: float | None = None

        method_map = {
            "BayesA": BayesA,
            "BayesB": BayesB,
            "BayesCpi": BayesCpi,
        }
        if method not in method_map:
            raise ValueError(f"Unsupported Bayes method: {method}")

        beta, alpha, varbeta, varep, h2_mean, varh2 = method_map[method](
            y,
            M,
            X,
            n_iter=n_iter,
            burnin=burnin,
            thin=thin,
            r2=float(r2),
            prob_in=prob_in,
            counts=counts,
            seed=seed,
        )
        self.beta_hat = beta.reshape(-1, 1);self.varbeta_hat = varbeta
        self.alpha_hat = alpha.reshape(-1, 1)
        self.varep_hat = float(varep)
        self.pve = float(h2_mean);self.varpve = float(varh2)
        
    def predict(self,M:np.ndarray,cov:np.ndarray=None):
        """
        Fast solution of the mixed linear model via Brent's method.

        Parameters
        ----------
        M : np.ndarray
            Marker matrix of shape (m, n) with genotypes coded as 0/1/2.
        cov : np.ndarray, optional
            Fixed-effect design matrix of shape (n, p).
        """
        X = (
            np.concatenate([np.ones((M.shape[1], 1)), cov], axis=1)
            if cov is not None
            else np.ones((M.shape[1], 1))
        )
        return (self.beta_hat.T@M).reshape(-1,1) + X @ self.alpha_hat


bayesA = BayesA
bayesB = BayesB
bayesCpi = BayesCpi

__all__ = ["BayesA", "BayesB", "BayesCpi", "BAYES", "bayesA", "bayesB", "bayesCpi"]
