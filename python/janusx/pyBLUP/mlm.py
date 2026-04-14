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
        # Keep marker matrix in float32 by default to avoid a full float64 copy.
        self.M = np.asarray(M)
        if not np.issubdtype(self.M.dtype, np.floating):
            self.M = self.M.astype(np.float32, copy=False)
        elif self.M.dtype != np.float32:
            self.M = self.M.astype(np.float32, copy=False)
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
        # Represent default random-effect design as implicit identity.
        z_is_identity = Z is None
        if not z_is_identity:
            Z = np.asarray(Z, dtype=np.float64)
            if Z.ndim != 2:
                raise ValueError("Z must be a 2D array.")
            if Z.shape[0] != self.y.shape[0] or Z.shape[1] != self.y.shape[0]:
                raise ValueError(
                    f"Z must be square with shape (n, n). Got Z.shape={Z.shape}, n={self.y.shape[0]}"
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
        self._pred_chunk_size = 32_768
        self._alpha_sum = None
        self._m_alpha = None
        self._mean_sq = None
        self._mean_malpha = None
        self._g_eps = 1e-6
        self._implicit_kinship_fast = False
        usefastlmm = self.M.shape[0] < self.n  # use FaST-LMM if num of snp less than num of samples
        if self.kinship is not None:
            self._m_mean = np.mean(self.M, axis=1, dtype=np.float32, keepdims=True)
            m_var = np.var(self.M, axis=1, dtype=np.float32, keepdims=True)
            self._m_var_sum = float(np.sum(m_var, dtype=np.float32))
            if self._m_var_sum <= 0.0:
                self._m_var_sum = 1e-12
            if usefastlmm and z_is_identity:
                # Low-rank kinship path:
                # avoid explicit n x n G and recover u/alpha from SVD spectrum.
                self._implicit_kinship_fast = True
                self.G = None
                self.Z = None
            else:
                self.G = GRM(self.M, log=self.log).astype(np.float32, copy=False)
                self.G += np.float32(self._g_eps) * np.eye(self.n, dtype=np.float32)  # Add regular item
                self.Z = np.eye(self.n, dtype=np.float64) if z_is_identity else Z
        else:
            self.G = np.eye(self.M.shape[0], dtype=np.float32)
            self.Z = self.M.T if z_is_identity else Z @ self.M.T
        # Simplify inverse matrix
        if usefastlmm:
            if self.kinship is None:
                # kinship-free model uses ZM^T as random-effect design.
                fast_input = self.Z
            elif self._implicit_kinship_fast:
                # For default Z=I, avoid materializing an n x n identity matrix.
                fast_input = self.M.T
            else:
                fast_input = self.M.T if z_is_identity else (Z @ self.M.T)
            u, svals, vh = np.linalg.svd(fast_input, full_matrices=False)
            self.Dh = u.T
            self.S = svals.astype(np.float64, copy=False) ** 2
        else:
            if self.kinship is not None and z_is_identity:
                self.S, eigvec = np.linalg.eigh(self.G)
            else:
                self.S, eigvec = np.linalg.eigh(self.Z @ self.G @ self.Z.T)
            self.Dh = eigvec.T
        self.X = self.Dh @ self.X
        self.y = self.Dh @ self.y
        if usefastlmm and self.kinship is None:
            # Dh @ (ZM^T) = diag(s) @ Vh from SVD: avoids one extra large GEMM.
            self.Z = svals[:, None] * vh
        elif self.kinship is not None and z_is_identity:
            self.Z = self.Dh
        else:
            self.Z = self.Dh @ self.Z
        result = minimize_scalar(lambda lbd: -self._REML(10**(lbd)), bounds=(-6, 6), method='bounded')  # minimize REML
        lbd = 10 ** result.x
        self._REML(lbd)
        if self.kinship is not None and self._implicit_kinship_fast:
            # In FaST low-rank mode:
            #   alpha = U * diag(1/(s^2+lambda)) * r_rot = Dh.T @ (V_inv * r)
            #   u     = U * diag(s^2 + eps) * alpha_rot
            vr = (self.V_inv.ravel() * self.r.T).T
            spectral = (self.S + self._g_eps).reshape(-1, 1)
            self.alpha = self.Dh.T @ vr
            self.u = self.Dh.T @ (spectral * vr)
        else:
            self.u = self.G @ self.Z.T @ (self.V_inv.ravel() * self.r.T).T
            self.alpha = np.linalg.solve(self.G, self.u) if self.kinship is not None else None
        if self.kinship is not None and self.alpha is not None:
            # Cache compact terms for prediction:
            # cross(M_new, M_train) @ alpha without materializing cross kernel.
            self._alpha_sum = float(np.sum(self.alpha, dtype=np.float64))
            self._m_alpha = self.M @ self.alpha
            self._mean_sq = float((self._m_mean.T @ self._m_mean)[0, 0]) if self._m_mean is not None else None
            self._mean_malpha = (
                float((self._m_mean.T @ self._m_alpha)[0, 0]) if self._m_mean is not None else None
            )
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

    def _cross_grm_times_alpha(self, M_new: np.ndarray) -> np.ndarray:
        """
        Compute cross_GRM(M_new, M_train) @ alpha in chunks without building
        the full n_new x n_train cross kernel matrix.
        """
        if (
            self._m_mean is None
            or self._m_var_sum is None
            or self._m_alpha is None
            or self._alpha_sum is None
            or self._mean_sq is None
            or self._mean_malpha is None
        ):
            raise RuntimeError("cross GRM compact terms are not initialized.")
        n_new = M_new.shape[1]
        out = np.empty((n_new, 1), dtype=np.float64)
        step = max(1, int(self._pred_chunk_size))
        # Formula:
        # cross @ alpha =
        # [M_new^T(M alpha) - (M_new^T m_mean) * sum(alpha)
        #  - (m_mean^T M alpha) + (m_mean^T m_mean) * sum(alpha)] / var_sum
        for st in range(0, n_new, step):
            ed = min(st + step, n_new)
            blk = M_new[:, st:ed]
            term1 = blk.T @ self._m_alpha
            term2 = blk.T @ self._m_mean
            out[st:ed] = (
                term1
                - term2 * self._alpha_sum
                - self._mean_malpha
                + self._mean_sq * self._alpha_sum
            ) / self._m_var_sum
        return out

    def predict(self, M: np.ndarray, cov: Union[np.ndarray, None] = None):
        M = np.asarray(M, dtype=np.float32)
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
            rand_eff = self._cross_grm_times_alpha(M)
            return X @ self.beta + rand_eff
        else:
            return X @ self.beta + M.T @ self.u
