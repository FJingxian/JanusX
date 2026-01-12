import time
from janusx.gfreader import breader
from janusx.pyBLUP.assoc import lmm_reml
from janusx.bioplotkit.manhanden import GWASPLOT
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from typing import Optional
import numpy as np


class lrLMM:
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

    def __init__(self, y: np.ndarray, X: Optional[np.ndarray], U: np.ndarray, S:np.ndarray):
        y = np.asarray(y).reshape(-1, 1)  # ensure (n,1)

        # Add intercept automatically
        X = (
            np.concatenate([np.ones((y.shape[0], 1)), X], axis=1)
            if X is not None
            else np.ones((y.shape[0], 1))
        )

        # Eigen decomposition of kinship (stabilized)
        self.S = S
        self.UT = U.T

        # Pre-rotate covariates and phenotype once
        self.Xcov = self.UT @ X
        self.y = self.UT @ y

        # ---- Estimate null lambda via scalar optimization (Python) ----
        result = minimize_scalar(
            lambda lbd: -self._NULLREML(10 ** (lbd)),
            bounds=(-5, 5),
            method="bounded",
            options={"xatol": 1e-3},
        )
        lbd_null = 10 ** (result.x)

        # A crude PVE estimate; adjust if your model defines vg differently
        vg_null = np.mean(self.S)
        pve = vg_null / (vg_null + lbd_null)

        self.lbd_null = lbd_null
        self.pve = pve

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
        beta_se_p : np.ndarray, shape (m, 3)
            Per-SNP beta, se, p.
        """
        beta_se_p, lambdas = lmm_reml(
            self.S,
            self.Xcov,
            self.y,
            self.UT,
            snp,
            self.bounds,
            max_iter=30,
            tol=1e-2,
            threads=threads,
        )
        self.lbd = lambdas
        return beta_se_p

if __name__ == '__main__':
    import pandas as pd
    from janusx.janusx import block_randomized_svd_bed
    from janusx.gfreader.gfreader import load_genotype_chunks
    import sys
    geno = breader('/Users/jingxianfu/script/JanusX2/test/mouse_hs1940')
    chrloc = geno.index
    pheno = pd.read_csv('/Users/jingxianfu/script/JanusX2/example/mouse_hs1940.pheno',sep='\t',index_col=0).iloc[:,0].dropna()
    samples = pheno.index.tolist()
    # M = geno.loc[:,pheno.index].values
    y = pheno.values
    
    rank = int(sys.argv[1])
    t = time.time()
    U,S,_ = block_randomized_svd_bed('/Users/jingxianfu/script/JanusX2/test/mouse_hs1940',k=rank,maf=0, miss=0.05,sample_ids=samples,threads=8,return_vt=False,n_iter=0)
    print(time.time()-t,'secs')
    t = time.time()
    U,S,Vh = np.linalg.svd(geno[samples].values.T)
    print(time.time()-t,'secs')
    Ulr = U[:,:rank]
    Slr = S[:rank]
    gwasmodel = lrLMM(y,None,Ulr,Slr*Slr)
    chunks = load_genotype_chunks('/Users/jingxianfu/script/JanusX2/test/mouse_hs1940',maf=0, missing_rate=0.05,sample_ids=samples)
    result = []
    for chunk,site in chunks:
        result.append(gwasmodel.gwas(chunk,threads=4))
    print(np.concatenate(result).shape)
    plotmodel = GWASPLOT(pd.DataFrame(np.concatenate(result),columns=['beta','se','p'],index=chrloc).reset_index())
    plotmodel.manhattan(threshold=4)
    plt.savefig('test.png')