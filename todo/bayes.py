from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from janusx.janusx import bayesa as _bayesa


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
) -> Tuple[np.ndarray, np.ndarray, float]:
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


def bayesA(
    y: np.ndarray,
    M: np.ndarray,
    X: Optional[np.ndarray] = None,
    n_iter: int = 300,
    burnin: int = 150,
    thin: int = 1,
    r2: float = 0.5,
    df0_b: float = 5.0,
    shape0: float = 1.1,
    rate0: Optional[float] = None,
    s0_b: Optional[float] = None,
    df0_e: float = 5.0,
    s0_e: Optional[float] = None,
    min_abs_beta: float = 1e-9,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Python interface for the Rust BayesA kernel (PyO3).

    This wrapper normalizes inputs to contiguous float64 arrays and passes
    them to the Rust implementation `janusx.janusx.bayesa`.

    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        Phenotype vector. Flattened to 1D float64.
    M : array-like, shape (n, p)
        Marker matrix with samples in rows and markers in columns.
    X : array-like, shape (n, q) or (n,), optional
        Covariate matrix. 1D inputs are treated as a single covariate.
    n_iter : int, default=200
        Total MCMC iterations.
    burnin : int, default=100
        Burn-in iterations. Must be < n_iter.
    thin : int, default=1
        Keep every `thin`-th sample after burn-in.
    r2 : float, default=0.5
        Proportion of variance explained by markers; must be in (0, 1).
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
        Posterior mean covariate effects; empty if `X` is None.
    mu : float
        Posterior mean intercept.

    Raises
    ------
    ValueError
        If shapes are incompatible or hyperparameters are out of range.

    Notes
    -----
    - Inputs are copied to contiguous float64 arrays before calling Rust.
    - M is expected to be (n, p) with n == len(y). If your genotype matrix
      is SNP-major (p, n), transpose before calling.
    """
    y_arr = _as_1d_f64(y, "y")
    m_arr = _as_2d_f64(M, "M", y_arr.shape[0])
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


if __name__ == "__main__":
    from janusx.gfreader.gfreader import load_genotype_chunks,inspect_genotype_file
    from janusx.pyBLUP.mlm import BLUP
    from janusx.pyBLUP.kfold import kfold
    import numpy as np
    import pandas as pd
    # /Users/jingxianfu/script/JanusX/example/mouse_hs1940.vcf.gz
    # /Users/jingxianfu/script/JanusX/example/mouse_hs1940.pheno
    genofile = '/Users/jingxianfu/script/JanusX/example/mouse_hs1940.vcf.gz'
    phenofile = '/Users/jingxianfu/script/JanusX/example/mouse_hs1940.pheno'
    ids, nsnp = inspect_genotype_file(genofile)
    pheno = pd.read_csv(phenofile,sep='\t',index_col=0).iloc[:,0].dropna()
    samplec = np.isin(ids,pheno.index)
    chunks = load_genotype_chunks(genofile,maf=0.01)
    Kiter = kfold(np.sum(samplec),seed=None)
    for chunk,site in chunks:
        phenotype = pheno.loc[np.array(ids)[samplec]].values
        genotype = chunk[:,samplec]
    for testidx,trainidx in Kiter:
        beta_hat, alpha_hat, mu_hat = bayesA(phenotype[trainidx],genotype[:,trainidx].T)
        y_hat = mu_hat + genotype[:,testidx].T @ beta_hat
        print('BayesA:',np.corrcoef(y_hat.ravel(), phenotype[testidx].ravel())[0,1])
        model = BLUP(phenotype[trainidx],genotype[:,trainidx],log=True,kinship=None)
        y_hat = model.predict(genotype[:,testidx])
        print('rrBLUP:',np.corrcoef(y_hat.ravel(),phenotype[testidx])[0,1])
        print('*'*60)
