from typing import Union, Literal
import numpy as np
from scipy.optimize import minimize_scalar
from .QK2 import GRM


class BLUP:
    def __init__(
        self,
        y: np.ndarray,
        M: np.ndarray,
        cov: Union[np.ndarray, None] = None,
        Z: Union[np.ndarray, None] = None,
        kinship: Literal[None, 1] = None,
        log: bool = False,
    ):
        """
        Fast solution of the mixed linear model via Brent's method.

        Parameters
        ----------
        y : np.ndarray
            Phenotype vector of shape (n, 1).
        M : np.ndarray
            Marker matrix of shape (m, n) with genotypes coded as 0/1/2.
        cov : np.ndarray, optional
            Fixed-effect design matrix of shape (n, p).
        Z : np.ndarray, optional
            Random-effect design matrix of shape (n, q).
        kinship : {None, 1}
            Kinship specification; None disables kinship.
        """
        self.log = log
        self.y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        self.M = np.asarray(M, dtype=np.float64)
        if self.M.ndim != 2:
            raise ValueError("M must be a 2D array with shape (m, n).")
        if self.M.shape[1] != self.y.shape[0]:
            raise ValueError(
                f"M must be (m, n) with n=len(y). Got M.shape={self.M.shape}, len(y)={self.y.shape[0]}"
            )
        if cov is not None:
            cov = np.asarray(cov, dtype=np.float64)
            if cov.ndim == 1:
                cov = cov.reshape(-1, 1)
            if cov.shape[0] != self.y.shape[0]:
                raise ValueError(
                    f"cov rows must match len(y). Got cov.shape={cov.shape}, len(y)={self.y.shape[0]}"
                )
        Z = Z if Z is not None else np.eye(self.y.shape[0], dtype=np.float64)  # Design matrix or I matrix
        if self.M.shape[1] != Z.shape[1]:
            raise ValueError(
                f"M.shape[1] must equal Z.shape[1]. Got M.shape={self.M.shape}, Z.shape={Z.shape}"
            )
        self.X = (
            np.concatenate([np.ones((self.y.shape[0], 1)), cov], axis=1)
            if cov is not None
            else np.ones((self.y.shape[0], 1))
        )  # Design matrix of 1st vector
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.kinship = kinship  # control method to calculate kinship matrix
        self._m_mean = None
        self._m_var_sum = None
        usefastlmm = self.M.shape[0] < self.n  # use FaST-LMM if num of snp less than num of samples
        if self.kinship is not None:
            self._m_mean = np.mean(self.M, axis=1, dtype=np.float64, keepdims=True)
            m_var = np.var(self.M, axis=1, dtype=np.float64, keepdims=True)
            self._m_var_sum = float(np.sum(m_var, dtype=np.float64))
            if self._m_var_sum <= 0.0:
                self._m_var_sum = 1e-12
            self.G = GRM(self.M, log=self.log) + 1e-6 * np.eye(self.n)  # Add regular item
            self.Z = Z
        else:
            self.G = np.eye(self.M.shape[0])
            self.Z = Z @ self.M.T
        # Simplify inverse matrix
        if usefastlmm:
            self.Dh, self.S, _ = (
                np.linalg.svd(Z @ self.M.T, full_matrices=False) if Z is not None else np.linalg.svd(self.M.T)
            )
            self.S = self.S * self.S
        else:
            self.S, self.Dh = np.linalg.eigh(self.Z @ self.G @ self.Z.T)
        self.Dh = self.Dh.T
        self.X = self.Dh @ self.X
        self.y = self.Dh @ self.y
        self.Z = self.Dh @ self.Z
        result = minimize_scalar(lambda lbd: -self._REML(10**(lbd)), bounds=(-6, 6), method='bounded')  # minimize REML
        lbd = 10 ** result.x
        self._REML(lbd)
        self.u = self.G @ self.Z.T @ (self.V_inv.ravel() * self.r.T).T
        self.alpha = np.linalg.solve(self.G, self.u) if self.kinship is not None else None
        sigma_g2 = self.rTV_invr / (self.n - self.p)
        sigma_e2 = lbd * sigma_g2
        if self.kinship is None:
            g = self.M.T @ self.u
            var_g = float(np.var(g, ddof=1))
        else:
            mean_s = float(np.mean(self.S))
            var_g = sigma_g2 * mean_s
        self.pve = var_g / (var_g + sigma_e2)
    def _REML(self, lbd: float):
        """Restricted maximum likelihood (REML) for the null model."""
        n, p_cov = self.X.shape
        p = p_cov
        V = self.S + lbd
        V_inv = 1 / V
        X_cov_snp = self.X
        XTV_invX = V_inv * X_cov_snp.T @ X_cov_snp
        XTV_invy = V_inv * X_cov_snp.T @ self.y
        beta = np.linalg.solve(XTV_invX, XTV_invy)
        r = self.y - X_cov_snp @ beta
        rTV_invr = (V_inv * r.T @ r)[0, 0]
        log_detV = np.sum(np.log(V))
        _, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
        total_log = (n - p) * np.log(rTV_invr) + log_detV + log_detXTV_invX  # log items
        c = (n - p) * (np.log(n - p) - 1 - np.log(2 * np.pi)) / 2  # Constant
        self.V_inv = V_inv
        self.rTV_invr = rTV_invr
        self.r = r
        self.beta = beta
        return c - total_log / 2

    def _cross_grm(self, M_new: np.ndarray) -> np.ndarray:
        if self._m_mean is None or self._m_var_sum is None:
            raise RuntimeError("cross GRM requested before kinship statistics are initialized.")
        # Use train-set mean/variance for test-set kernel to avoid leakage and
        # keep predict() consistent with the fitted GRM.
        cross = M_new.T @ self.M
        cross -= M_new.T @ self._m_mean
        cross -= self._m_mean.T @ self.M
        cross += float((self._m_mean.T @ self._m_mean)[0, 0])
        return cross / self._m_var_sum

    def predict(self, M: np.ndarray, cov: Union[np.ndarray, None] = None):
        M = np.asarray(M, dtype=np.float64)
        if M.ndim != 2:
            raise ValueError("M must be a 2D array with shape (m, n_new).")
        if M.shape[0] != self.M.shape[0]:
            raise ValueError(
                f"M must have the same SNP rows as training data. Expected {self.M.shape[0]}, got {M.shape[0]}"
            )
        if cov is not None:
            cov = np.asarray(cov, dtype=np.float64)
            if cov.ndim == 1:
                cov = cov.reshape(-1, 1)
            if cov.shape[0] != M.shape[1]:
                raise ValueError(
                    f"cov rows must match M.shape[1]. Got cov.shape={cov.shape}, M.shape={M.shape}"
                )
        X = (
            np.concatenate([np.ones((M.shape[1], 1)), cov], axis=1)
            if cov is not None
            else np.ones((M.shape[1], 1))
        )
        if self.kinship is not None:
            G_cross = self._cross_grm(M)
            return X @ self.beta + G_cross @ self.alpha
        else:
            return X @ self.beta + M.T @ self.u
