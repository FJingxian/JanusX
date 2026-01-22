import typing

import numpy as np
import jax.numpy as jnp
from jax import grad
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

def REML(
    theta: typing.Sequence[float],
    y: np.ndarray,
    X: np.ndarray,
    SIGMAlist: typing.Sequence[np.ndarray],
) -> float:
    """
    Restricted Maximum Likelihood (REML) for variance components.

    Parameters
    ----------
    theta : Sequence[float]
        Variance components for each covariance term in SIGMAlist.
    y : np.ndarray
        Response vector with shape (n, 1).
    X : np.ndarray
        Fixed-effect design matrix with shape (n, p).
    SIGMAlist : Sequence[np.ndarray]
        Covariance matrices for random effects plus residual.

    Returns
    -------
    float
        REML objective value (lower is better for minimize).
    """
    theta = jnp.asarray(theta)
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    n, p = X.shape

    V = sum(theta[k] * SIGMAlist[k] for k in range(len(SIGMAlist)))
    L = jnp.linalg.cholesky(V)
    Vinv = jnp.linalg.inv(V)

    XTV_invX = X.T @ Vinv @ X
    XTV_invy = X.T @ Vinv @ y

    beta = jnp.linalg.solve(XTV_invX, XTV_invy)
    r = y - X @ beta

    rTV_invr = (r.T @ Vinv @ r)[0, 0]
    log_detV = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    sign, log_detXTV_invX = jnp.linalg.slogdet(XTV_invX)
    total_log = (n - p) * jnp.log(rTV_invr) + log_detV + log_detXTV_invX

    # Constant term (matches your original expression)
    c = (n - p) * (jnp.log(n - p) - 1 - jnp.log(2 * jnp.pi)) / 2.0
    return total_log / 2.0 - c

def _as_2d(array: np.ndarray, name: str, n_rows: int | None = None) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D array, got shape {arr.shape}")
    if n_rows is not None and arr.shape[0] != n_rows:
        raise ValueError(f"{name} must have {n_rows} rows, got {arr.shape[0]}")
    return arr


def _normalize_Z(
    Z: typing.Sequence[np.ndarray] | np.ndarray | None,
    n_rows: int,
) -> list[np.ndarray]:
    if Z is None:
        return []
    if isinstance(Z, np.ndarray):
        Z_list = [Z]
    else:
        Z_list = list(Z)
    out: list[np.ndarray] = []
    for idx, z in enumerate(Z_list):
        out.append(_as_2d(z, f"Z[{idx}]", n_rows))
    return out


def _split_u(u: np.ndarray, Z_list: list[np.ndarray]) -> list[np.ndarray]:
    u_by_Z: list[np.ndarray] = []
    start = 0
    for z in Z_list:
        end = start + z.shape[1]
        u_by_Z.append(u[start:end])
        start = end
    return u_by_Z


class BLUP:
    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        Z: typing.Sequence[np.ndarray] | np.ndarray | None = None,
        maxiter: int = 100,
        progress: bool = True,
    ):
        """
        Best Linear Unbiased Prediction (BLUP) with multiple random effects.

        Parameters
        ----------
        y : np.ndarray
            Response vector with shape (n, 1) or (n,).
        X : np.ndarray | None
            Covariate matrix of shape (n, p) without an intercept column. An
            intercept is always added internally. If None, intercept only.
        Z : Sequence[np.ndarray] | np.ndarray | None
            One or multiple random-effect design matrices, each with shape (n, q).
            Each Z gets one variance component.
        maxiter : int
            Maximum iterations for REML optimization.
        progress : bool
            Whether to show the REML progress bar.

        Attributes
        ----------
        beta : np.ndarray
            Fixed-effect estimates with shape (p + 1, 1).
        u : np.ndarray
            Random-effect estimates stacked by Z order.
        u_by_Z : list[np.ndarray]
            Random-effect estimates split per Z.
        theta : np.ndarray
            Estimated variance components for each Z plus residual.
        fitted : np.ndarray
            Fitted values for training data.
        residuals : np.ndarray
            Training residuals.
        result : OptimizeResult
            SciPy optimization result from REML.
        """
        y = _as_2d(y, "y")
        if y.shape[1] != 1:
            raise ValueError(f"y must have shape (n, 1), got {y.shape}")
        if X is None:
            X = np.ones((y.shape[0], 1))
        else:
            X = _as_2d(X, "X", y.shape[0])
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        Z_list = _normalize_Z(Z, X.shape[0])
        self.y = y
        self.X = X
        self.Z_list = Z_list
        self.z_cols = [z.shape[1] for z in Z_list]
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.maxiter = maxiter
        self.progress = progress
        self._fit()

    def _fit(self) -> None:
        SIGMAlist = [z @ z.T for z in self.Z_list]
        SIGMAlist.append(np.eye(self.n))
        theta0 = np.ones(len(SIGMAlist))
        gradfn = grad(REML)

        pbar = None
        callback = None
        if self.progress:
            pbar = tqdm(total=self.maxiter, desc="REML", ncols=100)

            def callback(theta):
                pbar.update(1)
                pbar.set_postfix({"theta": np.round(theta, 4)})

        def jac(theta, y, X, SIGMAlist):
            return np.asarray(gradfn(theta, y, X, SIGMAlist), dtype=float)

        result = minimize(
            REML,
            theta0,
            args=(self.y, self.X, SIGMAlist),
            jac=jac,
            method="L-BFGS-B",
            callback=callback,
            options={"maxiter": self.maxiter, "disp": False},
        )
        if pbar is not None:
            pbar.close()
        theta = np.asarray(result.x, dtype=float)

        if theta.shape[0] != len(self.Z_list) + 1:
            raise ValueError(
                f"theta length {theta.shape[0]} does not match Z count {len(self.Z_list)}"
            )

        if self.Z_list:
            V = sum(
                theta[ind] * (z @ z.T) for ind, z in enumerate(self.Z_list)
            ) + theta[-1] * np.eye(self.n)
        else:
            V = theta[-1] * np.eye(self.n)

        VinvX = np.linalg.solve(V, self.X)
        Vinvy = np.linalg.solve(V, self.y)
        beta_hat = np.linalg.solve(self.X.T @ VinvX, self.X.T @ Vinvy)
        r = self.y - self.X @ beta_hat
        Vinvr = np.linalg.solve(V, r)

        if self.Z_list:
            Zall = np.concatenate(self.Z_list, axis=1)
            Gall = np.concatenate(
                [theta[ind] * np.ones(z.shape[1]) for ind, z in enumerate(self.Z_list)]
            )
            u_hat = (Zall.T @ Vinvr) * Gall[:, None]
            fitted = self.X @ beta_hat + Zall @ u_hat
            u_by_Z = _split_u(u_hat, self.Z_list)
        else:
            u_hat = np.zeros((0, 1))
            fitted = self.X @ beta_hat
            u_by_Z = []

        self.theta = theta
        self.beta = beta_hat
        self.u = u_hat
        self.u_by_Z = u_by_Z
        self.fitted = fitted
        self.residuals = self.y - fitted
        self.result = result

    def predict(
        self,
        X: np.ndarray | None = None,
        Z: typing.Sequence[np.ndarray] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict response for new samples using fitted effects.

        Parameters
        ----------
        X : np.ndarray | None
            Covariate matrix of shape (n, p) without an intercept column. An
            intercept is always added internally. If None, intercept only.
        Z : Sequence[np.ndarray] | np.ndarray | None
            Random-effect design matrices for the same effects as training.

        Returns
        -------
        np.ndarray
            Predicted values with shape (n, 1).
        """
        n_rows = None
        if X is not None:
            X = _as_2d(X, "X")
            n_rows = X.shape[0]
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        if Z is not None:
            if isinstance(Z, np.ndarray):
                if n_rows is None:
                    n_rows = Z.shape[0]
                Z_list = _normalize_Z(Z, n_rows)
            else:
                if len(Z) == 0:
                    Z_list = []
                else:
                    if n_rows is None:
                        n_rows = np.asarray(Z[0]).shape[0]
                    Z_list = _normalize_Z(Z, n_rows)
        else:
            Z_list = []
        if n_rows is None:
            n_rows = self.n
        if X is None:
            X = np.ones((n_rows, 1))

        y_hat = X @ self.beta
        if not Z_list:
            return y_hat
        if len(Z_list) != len(self.z_cols):
            raise ValueError(f"Z count mismatch: expected {len(self.z_cols)}, got {len(Z_list)}")
        for idx, (z, expected_cols) in enumerate(zip(Z_list, self.z_cols)):
            if z.shape[1] != expected_cols:
                raise ValueError(
                    f"Z[{idx}] columns mismatch: expected {expected_cols}, got {z.shape[1]}"
                )
        Zall = np.concatenate(Z_list, axis=1)
        if Zall.shape[1] != self.u.shape[0]:
            raise ValueError(
                f"Z columns {Zall.shape[1]} do not match u length {self.u.shape[0]}"
            )
        return y_hat + Zall @ self.u

if __name__ == "__main__":
    from janusx.gfreader import vcfreader
    
    geno = vcfreader('example/mouse_hs1940.vcf.gz',maf=0.02,miss=0.05).iloc[:,2:]
    pheno = pd.read_csv('example/mouse_hs1940.pheno',sep='\t',index_col=0).iloc[:,0].dropna()
    csample = list(set(geno.columns) & set(pheno.index))
    y = pheno.loc[csample].values.reshape(-1,1)
    Z = geno[csample].values.T
    # Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    # Z = [Z]
    # data = pd.read_csv('todo/blup.test2.tsv',sep='\t',).dropna().reset_index().iloc[:,1:]
    # y = data.iloc[:,[1]].values
    # Z = [pd.get_dummies(data.iloc[:,0]).astype(int).values,pd.get_dummies(data.iloc[:,2]).astype(int).values]
    model = BLUP(y, None, Z=Z, maxiter=50, progress=True)
    print(model.beta)
    print(model.u.shape)
    print(model.theta[0]/model.theta.sum())
    print("theta:", model.theta)
    print("nit:", model.result.nit)
    print("status:", model.result.status)
    print("message:", model.result.message)