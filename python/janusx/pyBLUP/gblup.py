from typing import Union
import numpy as np
from numpy.typing import NDArray
from janusx.pyBLUP.tdata import _testX,_testY
from janusx.pyBLUP.lmm import LMM


class GBLUP:
    """
    Genomic BLUP predictor with optional FaST-LMM acceleration.

    Parameters
    ----------
    y : np.ndarray, shape (n,) or (n,1)
        Phenotype vector.
    M : np.ndarray, shape (m, n)
        SNP-major genotype matrix (rows SNPs, columns samples).
        This function standardizes SNPs by training mean/std and 1/sqrt(m).
    X : np.ndarray or None, shape (n, p)
        Covariates WITHOUT an intercept. Intercept is added internally.
    FaST : bool
        Force FaST-LMM path. If False, chooses GRM path when n <= m.
    ridge : float
        Stabilizer for SNP standardization.
    """
    def __init__(self,y:NDArray[np.floating],M:NDArray[np.floating],X:Union[NDArray[np.floating],None]=None,FaST:bool=False,ridge=1e-6) -> None:
        y = _testY(y)
        n = y.shape[0]
        if M.shape[1] != n:
            raise ValueError(f"M must be (m, n). Got M.shape={M.shape}, n={n}.")
        X = _testX(X, n, add_intercept=True)
        m = M.shape[0]
        Mmean = M.mean(axis=1,keepdims=True)
        Mstd = M.std(axis=1,keepdims=True)+ridge
        M = (M - Mmean) / Mstd / np.sqrt(m)
        self.M = M
        self.Mmean = Mmean;self.Mstd = Mstd;self.m = m
        self.n_train = n
        self.p = X.shape[1] - 1
        if FaST:
            self.FaST:bool = True
            model = LMM(y, X, M=M.T)
        else:
            if n<=m:
                self.FaST:bool = False
                ktt = M.T@M
                model = LMM(y, X, grm=ktt)
            else:
                self.FaST:bool = True
                model = LMM(y, X, M=M.T)
        self.pve = model.pve
        self.beta = model.beta
        self.result = model
    def predict(self,M:NDArray[np.floating],X:Union[NDArray[np.floating],None]=None,):
        """
        Predict phenotype for new samples.

        Parameters
        ----------
        M : np.ndarray, shape (m, n_new)
            SNP-major genotype matrix for new samples.
        X : np.ndarray or None, shape (n_new, p)
            Covariates WITHOUT intercept. If None, intercept-only is used.
        """
        if M.shape[0] != self.m:
            raise ValueError(f"M must have {self.m} SNP rows. Got M.shape={M.shape}.")
        n = M.shape[1]
        X_design = _testX(X, n, add_intercept=True)
        M = (M - self.Mmean) / self.Mstd / np.sqrt(self.m)
        Knt = M.T@self.M
        if self.FaST:
            Sinv = 1/(self.result.ss+np.power(10,self.result.loglbd))
            S2inv = 1/np.power(10,self.result.loglbd)
            g_hat = Knt@((self.result.u * Sinv)@self.result.r1 + self.result.r2 * S2inv)
        else:
            Sinv = 1/(self.result.ss+np.power(10,self.result.loglbd))
            g_hat = Knt@((self.result.u * Sinv)@self.result.r)
        return g_hat + X_design@self.beta