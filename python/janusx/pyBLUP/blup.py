import typing
import platform
import numpy as np
from types import SimpleNamespace

from scipy.linalg import cho_solve
from scipy.optimize import minimize
from scipy import sparse
from scipy.sparse.linalg import splu
from tqdm import tqdm

try:
    import pypardiso as _pypardiso  # type: ignore[import-not-found]
    _HAS_PYPARDISO = True
except Exception:
    _pypardiso = None  # type: ignore[assignment]
    _HAS_PYPARDISO = False

try:
    from janusx.janusx import ai_reml_multi_f64 as _rust_ai_reml_multi_f64
    from janusx.janusx import ai_reml_null_f64 as _rust_ai_reml_null_f64
    _HAS_RUST_AIREML = True
except Exception:
    _rust_ai_reml_multi_f64 = None  # type: ignore[assignment]
    _rust_ai_reml_null_f64 = None  # type: ignore[assignment]
    _HAS_RUST_AIREML = False

_OBJ_PENALTY = 1e30
_PLAT_MACHINE = platform.machine().lower()
_PREFER_PYPARDISO = bool(
    _HAS_PYPARDISO
    and (_PLAT_MACHINE in {"x86_64", "amd64"})
    and (platform.system() != "Darwin")
)


def _pardiso_spsolve(m: sparse.csc_matrix, b: np.ndarray) -> np.ndarray:
    if _pypardiso is None:
        raise RuntimeError("pypardiso not available")
    b_arr = np.asarray(b, dtype=float)
    if b_arr.ndim == 1:
        x = _pypardiso.spsolve(m, b_arr)
        return np.asarray(x, dtype=float)
    if b_arr.ndim == 2:
        if b_arr.shape[1] == 0:
            return np.zeros((b_arr.shape[0], 0), dtype=float)
        # Keep 2D RHS shape stable (especially n x 1), so downstream matrix math
        # does not silently broadcast to wrong dimensions.
        cols = [
            np.asarray(_pypardiso.spsolve(m, b_arr[:, i]), dtype=float).reshape(-1, 1)
            for i in range(b_arr.shape[1])
        ]
        return np.hstack(cols)
    raise ValueError(f"rhs must be 1D or 2D, got shape={b_arr.shape}")

def REML(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Klist: typing.List[np.ndarray],
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
    Klist : Sequence[np.ndarray]
        Covariance matrices for random effects (each shape: n x n).

    Returns
    -------
    float
        REML objective value (lower is better for minimize).
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    V = theta[-1] * np.eye(n, dtype=float)
    for num, K in enumerate(Klist):
        V += theta[num] * np.asarray(K, dtype=float)
    L = np.linalg.cholesky(V)
    VinvX = cho_solve((L, True), X)
    Vinvy = cho_solve((L, True), y)
    XTV_invX = X.T @ VinvX
    XTV_invy = X.T @ Vinvy

    beta = np.linalg.solve(XTV_invX, XTV_invy)
    r = y - X @ beta
    Vinvr = cho_solve((L, True), r)

    rTV_invr = float((r.T @ Vinvr)[0, 0])
    if not np.isfinite(rTV_invr) or rTV_invr <= 0.0:
        return _OBJ_PENALTY
    log_detV = float(2.0 * np.sum(np.log(np.diag(L))))
    sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
    if sign <= 0 or (not np.isfinite(log_detXTV_invX)):
        return _OBJ_PENALTY
    total_log = (n - p) * np.log(rTV_invr) + log_detV + float(log_detXTV_invX)

    # Constant term (matches your original expression)
    c = (n - p) * (np.log(n - p) - 1 - np.log(2 * np.pi)) / 2.0
    out = float(total_log / 2.0 - c)
    if not np.isfinite(out):
        return _OBJ_PENALTY
    return out

def _testX(X:typing.Union[np.ndarray,None],n:int,) -> np.ndarray:
    if X is None:
        X = np.ones((n, 1))
    else:
        if X.size == n:
            X = X.reshape(-1,1)
            X = np.concatenate([np.ones((n, 1)), X], axis=1)
        elif X.shape[0] == n:
            X = np.concatenate([np.ones((n, 1)), X], axis=1)
        else:
            raise ValueError(f"Row number of X is not equal to {n} in y. (X.shape={X.shape}, n={n})")
    return X

def _to_dense(z: typing.Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
    if sparse.issparse(z):
        return np.asarray(z.toarray(), dtype=float)
    return np.asarray(z, dtype=float)


def _testZ(
    Z: list[typing.Union[np.ndarray, sparse.spmatrix]] | np.ndarray | sparse.spmatrix | None,
    n: int | None = None,
) -> list[typing.Union[np.ndarray, sparse.spmatrix]]:
    Z_list: list[typing.Union[np.ndarray, sparse.spmatrix]]
    if Z is None:
        Z_list = []
    elif isinstance(Z, np.ndarray) or sparse.issparse(Z):
        Z_list = [Z]
    else:
        Z_list = list(Z)
    for idx, z in enumerate(Z_list):
        if (not isinstance(z, np.ndarray)) and (not sparse.issparse(z)):
            raise TypeError(f"Type of Z{idx} is {type(z)}")
        if int(z.shape[0]) == int(n):
            if sparse.issparse(z):
                Z_list[idx] = z.tocsr()
            else:
                Z_list[idx] = np.asarray(z, dtype=float)
        elif (not sparse.issparse(z)) and z.size == n:
            Z_list[idx] = np.asarray(z, dtype=float).reshape(-1, 1)
        else:
            raise ValueError(f"Row number of Z{idx} is not equal to {n} in y. (Z{idx}.shape={z.shape}, n={n})")
    return Z_list

def _testG(G: list[np.ndarray] | np.ndarray | None, n: int | None = None) -> list[np.ndarray]:
    G_list: list[np.ndarray]
    if G is None:
        G_list = []
    elif isinstance(G, np.ndarray):
        G_list = [G]
    else:
        G_list = list(G)
    for idx, g in enumerate(G_list):
        if not isinstance(g, np.ndarray):
            g = np.asarray(g, dtype=float)
            G_list[idx] = g
        else:
            g = np.asarray(g, dtype=float)
            G_list[idx] = g
        if g.ndim != 2:
            raise ValueError(f"G{idx} must be 2D. (G{idx}.shape={g.shape})")
        if n is not None and (g.shape[0] != n or g.shape[1] != n):
            raise ValueError(
                f"G{idx} must be square (n,n). (G{idx}.shape={g.shape}, n={n})"
            )
    return G_list

def _normalize_G(
    G_list: typing.Sequence[np.ndarray],
    ridge: float = 1e-8,
) -> tuple[list[np.ndarray], list[float]]:
    out: list[np.ndarray] = []
    scales: list[float] = []
    for idx, g in enumerate(G_list):
        gs = np.asarray(g, dtype=float)
        gs = (gs + gs.T) / 2.0
        dmean = float(np.mean(np.diag(gs)))
        if (not np.isfinite(dmean)) or (dmean <= 0.0):
            raise ValueError(
                f"G{idx} has invalid mean diagonal ({dmean}); cannot normalize."
            )
        s = dmean + float(ridge)
        out.append(gs / s)
        scales.append(s)
    return out, scales

def _testG_predict(
    G: list[np.ndarray] | np.ndarray | None,
    n_pred: int,
    n_train: int,
) -> list[np.ndarray]:
    G_list: list[np.ndarray]
    if G is None:
        return []
    if isinstance(G, np.ndarray):
        G_list = [G]
    else:
        G_list = list(G)
    checked: list[np.ndarray] = []
    for idx, g in enumerate(G_list):
        g = np.asarray(g, dtype=float)
        if g.ndim != 2:
            raise ValueError(f"G[{idx}] must be 2D. (shape={g.shape})")
        if g.shape == (n_pred, n_train):
            checked.append(g)
        elif g.shape == (n_train, n_train) and n_pred == n_train:
            checked.append(g)
        else:
            raise ValueError(
                f"G[{idx}] shape mismatch: expected ({n_pred},{n_train}) "
                f"or ({n_train},{n_train}) for in-sample, got {g.shape}"
            )
    return checked

def _onehotZ(
    Z_list: typing.Sequence[typing.Union[np.ndarray, sparse.spmatrix]],
    ridge: float = 1e-8,
) -> list[typing.Tuple[float, typing.Union[float, np.ndarray], typing.Union[float, np.ndarray]]]:
    info: list[typing.Tuple[float, typing.Union[float, np.ndarray], typing.Union[float, np.ndarray]]] = []
    for z in Z_list:
        if sparse.issparse(z):
            zc = z.tocsr().astype(float)
            row_nnz = np.diff(zc.indptr)
            row_sum = np.asarray(zc.sum(axis=1)).reshape(-1)
            is_binary = bool(np.all((zc.data >= 0.0) & (zc.data <= 1.0) & np.isin(zc.data, [0.0, 1.0])))
            is_onehot = bool(is_binary and np.all(row_nnz == 1) and np.all(np.abs(row_sum - 1.0) <= 1e-6))
            if is_onehot:
                info.append((1, 0.0, 1.0))
                continue
            q = float(zc.shape[1])
            mean = np.asarray(zc.mean(axis=0)).reshape(-1)
            z2 = zc.copy()
            z2.data = z2.data ** 2
            ex2 = np.asarray(z2.mean(axis=0)).reshape(-1)
            std = np.sqrt(np.clip(ex2 - mean**2, a_min=0.0, a_max=None)) + ridge
            info.append((q, mean, std))
            continue
        zz = np.asarray(z, dtype=float)
        is_onehot = bool((np.isin(zz, [0, 1]).all()) and (zz.sum(axis=1) > 0).all())
        if is_onehot:
            info.append((1, 0.0, 1.0))
        else:
            info.append((float(zz.shape[1]), zz.mean(axis=0), zz.std(axis=0) + ridge))
    return info

def _split_u(
    u: np.ndarray,
    Z_list: list[typing.Union[np.ndarray, sparse.spmatrix]],
) -> list[np.ndarray]:
    u_by_Z: list[np.ndarray] = []
    start = 0
    for z in Z_list:
        end = start + z.shape[1]
        u_by_Z.append(u[start:end])
        start = end
    return u_by_Z


def _is_onehot_design(
    z: typing.Union[np.ndarray, sparse.spmatrix],
    tol: float = 1e-8,
) -> bool:
    if sparse.issparse(z):
        zc = z.tocsr().astype(float)
        if zc.ndim != 2:
            return False
        if np.any(zc.data < -tol) or np.any(zc.data > 1.0 + tol):
            return False
        row_nnz = np.diff(zc.indptr)
        if not np.all(row_nnz == 1):
            return False
        row_sum = np.asarray(zc.sum(axis=1)).reshape(-1)
        return bool(np.all(np.abs(row_sum - 1.0) <= 1e-6))
    zz = np.asarray(z, dtype=float)
    if zz.ndim != 2:
        return False
    if np.any(zz < -tol) or np.any(zz > 1.0 + tol):
        return False
    nnz_row = np.sum(zz > tol, axis=1)
    if not np.all(nnz_row == 1):
        return False
    row_sum = zz.sum(axis=1)
    if not np.all(row_sum > 0.0):
        return False
    # allow tiny fp noise
    return bool(np.all(np.abs(row_sum - 1.0) <= 1e-6))


def _can_use_sparse_z_reml(
    Z_list: list[typing.Union[np.ndarray, sparse.spmatrix]],
    G_list: list[np.ndarray],
    n: int,
) -> bool:
    # Large-sample fast path for one-hot random intercept designs only.
    if len(G_list) > 0 or len(Z_list) == 0:
        return False
    if n < 800:
        return False
    return all(_is_onehot_design(z) for z in Z_list)


def _to_onehot_csr(z: typing.Union[np.ndarray, sparse.spmatrix]) -> sparse.csr_matrix:
    if sparse.issparse(z):
        return z.tocsr().astype(float)
    zz = np.asarray(z, dtype=float)
    if zz.ndim != 2:
        raise ValueError(f"Z must be 2D, got shape={zz.shape}")
    n, q = zz.shape
    rows = np.arange(n, dtype=np.int64)
    cols = np.argmax(zz, axis=1).astype(np.int64, copy=False)
    vals = zz[rows, cols].astype(float, copy=False)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, q), dtype=float)


def _expand_z_theta(theta_z: np.ndarray, z_cols: list[int]) -> np.ndarray:
    return np.concatenate(
        [np.full(int(c), float(theta_z[i]), dtype=float) for i, c in enumerate(z_cols)],
        axis=0,
    )


def _sparse_z_reml_objective(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    z_csc: sparse.csc_matrix,
    ztz_csc: sparse.csc_matrix,
    zty: np.ndarray,
    ztx: np.ndarray,
    z_cols: list[int],
) -> float:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != (len(z_cols) + 1):
        return _OBJ_PENALTY
    if (not np.all(np.isfinite(theta))) or np.any(theta <= 0.0):
        return _OBJ_PENALTY

    n, p = int(X.shape[0]), int(X.shape[1])
    theta_z = theta[:-1]
    lbd = float(theta[-1])
    inv_lbd = 1.0 / lbd
    d = _expand_z_theta(theta_z, z_cols)

    try:
        m = sparse.diags(1.0 / d, format="csc") + (ztz_csc * inv_lbd)
        lu = splu(m)

        tmp_y = lu.solve(np.asarray(zty, dtype=float).reshape(-1)).reshape(-1, 1)
        tmp_x = np.asarray(lu.solve(np.asarray(ztx, dtype=float)), dtype=float)
        if tmp_x.ndim == 1:
            tmp_x = tmp_x.reshape(-1, 1)

        vinvy = y * inv_lbd - (z_csc @ tmp_y) * (inv_lbd * inv_lbd)
        vinvx = X * inv_lbd - (z_csc @ tmp_x) * (inv_lbd * inv_lbd)

        xt_vinv_x = X.T @ vinvx
        xt_vinv_y = X.T @ vinvy
        beta = np.linalg.solve(xt_vinv_x, xt_vinv_y)
        r = y - X @ beta
        vinvr = vinvy - vinvx @ beta
        r_t_vinvr = float((r.T @ vinvr)[0, 0])
        if (not np.isfinite(r_t_vinvr)) or r_t_vinvr <= 0.0:
            return _OBJ_PENALTY

        udiag = np.asarray(lu.U.diagonal(), dtype=float).reshape(-1)
        if np.any(udiag == 0.0):
            return _OBJ_PENALTY
        log_det_m = float(np.sum(np.log(np.abs(udiag))))
        log_det_v = (
            n * np.log(lbd)
            + float(np.sum(np.log(d)))
            + log_det_m
        )
        sign, log_det_xt_vinv_x = np.linalg.slogdet(xt_vinv_x)
        if sign <= 0 or (not np.isfinite(log_det_xt_vinv_x)):
            return _OBJ_PENALTY
        total_log = (n - p) * np.log(r_t_vinvr) + log_det_v + float(log_det_xt_vinv_x)
        c = (n - p) * (np.log(n - p) - 1 - np.log(2 * np.pi)) / 2.0
        out = float(total_log / 2.0 - c)
        if not np.isfinite(out):
            return _OBJ_PENALTY
        return out
    except Exception:
        return _OBJ_PENALTY


def _sparse_z_fit_state(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    z_csc: sparse.csc_matrix,
    ztz_csc: sparse.csc_matrix,
    z_cols: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    theta_z = theta[:-1]
    lbd = float(theta[-1])
    inv_lbd = 1.0 / lbd
    d = _expand_z_theta(theta_z, z_cols)

    m = sparse.diags(1.0 / d, format="csc") + (ztz_csc * inv_lbd)

    zty = np.asarray(z_csc.T @ y, dtype=float).reshape(-1)
    ztx = np.asarray(z_csc.T @ X, dtype=float)
    use_pardiso = _PREFER_PYPARDISO
    if use_pardiso:
        try:
            tmp_y = _pardiso_spsolve(m, zty).reshape(-1, 1)
            tmp_x = _pardiso_spsolve(m, ztx)
        except Exception:
            use_pardiso = False
    if not use_pardiso:
        lu = splu(m)
        tmp_y = lu.solve(zty).reshape(-1, 1)
        tmp_x = lu.solve(ztx)
    tmp_x = np.asarray(tmp_x, dtype=float)
    if tmp_x.ndim == 1:
        tmp_x = tmp_x.reshape(-1, 1)

    vinvy = y * inv_lbd - (z_csc @ tmp_y) * (inv_lbd * inv_lbd)
    vinvx = X * inv_lbd - (z_csc @ tmp_x) * (inv_lbd * inv_lbd)

    xt_vinv_x = X.T @ vinvx
    xt_vinv_y = X.T @ vinvy
    beta = np.linalg.solve(xt_vinv_x, xt_vinv_y)
    r = y - X @ beta
    vinvr = vinvy - vinvx @ beta

    rhs_r = np.asarray(z_csc.T @ r, dtype=float).reshape(-1)
    if use_pardiso:
        u_hat = _pardiso_spsolve(m, rhs_r * inv_lbd).reshape(-1, 1)
    else:
        u_hat = lu.solve(rhs_r * inv_lbd).reshape(-1, 1)
    z_fitted = np.asarray(z_csc @ u_hat, dtype=float)
    fitted = X @ beta + z_fitted
    residuals = y - fitted

    sigma2 = float((r.T @ vinvr)[0, 0]) / max(1, int(X.shape[0]) - int(X.shape[1]))
    cov_beta = np.linalg.pinv(np.asarray(xt_vinv_x, dtype=float)) * sigma2

    u_by_z: list[np.ndarray] = []
    start = 0
    for c in z_cols:
        end = start + int(c)
        u_by_z.append(u_hat[start:end])
        start = end
    return beta, u_hat, z_fitted, u_by_z, fitted, residuals, vinvr, cov_beta


def _fit_multi_kernel_ai_reml_rust(
    y: np.ndarray,
    X: np.ndarray,
    Klist: typing.Sequence[np.ndarray],
    maxiter: int,
) -> tuple[np.ndarray, typing.Any] | None:
    """
    Fast-path: multi-kernel REML solved by Rust AIREML.
    Returns (theta, result_like) or None when unavailable/fallback.
    """
    if (
        (not _HAS_RUST_AIREML)
        or (_rust_ai_reml_multi_f64 is None)
        or (len(Klist) == 0)
    ):
        return None
    try:
        n = int(X.shape[0])
        k_stack = np.stack(
            [((np.asarray(k, dtype=float) + np.asarray(k, dtype=float).T) / 2.0) for k in Klist],
            axis=0,
        )
        if k_stack.ndim != 3 or k_stack.shape[1] != n or k_stack.shape[2] != n:
            return None
        # Use exact trace for mid/small n to stabilize theta;
        # switch to Hutchinson probes for large n.
        if n <= 768:
            trace_probes = 0
        elif n <= 2048:
            trace_probes = 16
        else:
            trace_probes = 12
        theta, ml, reml, nit, converged = _rust_ai_reml_multi_f64(
            np.asarray(k_stack, dtype=np.float64),
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64).reshape(-1),
            max_iter=int(maxiter),
            tol=1e-6,
            min_var=1e-12,
            trace_probes=int(trace_probes),
            trace_seed=42,
        )
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != (len(Klist) + 1):
            return None
        if (not np.all(np.isfinite(theta))) or np.any(theta <= 0.0):
            return None
        result = SimpleNamespace(
            x=theta.copy(),
            success=bool(converged),
            nit=int(nit),
            message="rust_ai_reml",
            fun=float(-reml),
            ml=float(ml),
            reml=float(reml),
            lbd=float(theta[-1] / max(1e-12, theta[0])),
        )
        return theta, result
    except Exception:
        return None

class BLUP:
    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        Z: list[np.ndarray] | np.ndarray | None = None,
        G: list[np.ndarray] | np.ndarray | None = None,
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
        G : Sequence[np.ndarray] | np.ndarray | None
            One or multiple precomputed relationship/kinship covariance matrices,
            each with shape (n, n). Each G gets one variance component.
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
        g : np.ndarray | None
            Sum of covariance-based random effects from G terms, shape (n, 1).
        g_by_G : list[np.ndarray]
            Random effects split per G term.
        theta : np.ndarray
            Estimated variance components for each random term plus residual.
        fitted : np.ndarray
            Fitted values for training data.
        residuals : np.ndarray
            Training residuals.
        result : OptimizeResult
            SciPy optimization result from REML.
        """
        # 确保表型 y 参数 (n,1)
        y = y.reshape(-1,1)
        n = y.shape[0] # Row dimension
        # 检查固定效应 X 参数 (n,p) | None
        X = _testX(X,n)
        p = X.shape[1]
        # 检查随机效应 Z 参数 [(n,q1),(n,q2),...(n,qn)] | (n,q) | None
        self.Z_list = _testZ(Z,n)
        # 检查协方差随机效应 G 参数 [(n,n), ...] | (n,n) | None
        self.G_list = _testG(G,n)
        self.onehot_info = _onehotZ(self.Z_list)
        self.z_cols = [z.shape[1] for z in self.Z_list]
        self.g_count = len(self.G_list)
        self.y = y
        self.X = X
        self.n = n
        self.p = p
        self.maxiter = maxiter
        self.progress = progress
        self._fit()

    def _fit(self) -> None:
        if len(self.Z_list) == 0 and len(self.G_list) == 0: # 无随机协变量 进入一般线性模型
            self.theta = None
            self.V = None
            self.beta = np.linalg.solve(self.X.T@self.X,self.X.T@self.y)
            self.u = None
            self.u_by_Z = None
            self.g = None
            self.g_by_G = []
            self._Vinvr = None
            self._z_theta = None
            self._g_theta = None
            self._g_scales = []
            self._z_fitted = None
            self._z_standardized = False
            self._cov_beta = np.linalg.pinv(self.X.T @ self.X) * (
                float(((self.y - self.X @ self.beta).T @ (self.y - self.X @ self.beta))[0, 0])
                / max(1, self.n - self.p)
            )
            self.fitted = self.X@self.beta
            self.residuals = self.y - self.fitted 
            self.result = None
            pass
        else:
            use_sparse_z = _can_use_sparse_z_reml(self.Z_list, self.G_list, self.n)
            if use_sparse_z:
                z_blocks = [_to_onehot_csr(z) for z in self.Z_list]
                z_csc = sparse.hstack(z_blocks, format="csc")
                ztz_csc = (z_csc.T @ z_csc).tocsc()
                zty = np.asarray(z_csc.T @ self.y, dtype=float)
                ztx = np.asarray(z_csc.T @ self.X, dtype=float)
                z_cols = [int(z.shape[1]) for z in self.Z_list]
                theta0 = np.ones(len(z_cols) + 1, dtype=float)
                bounds = [(1e-12, None)] * theta0.size

                pbar = None
                callback = None
                if self.progress:
                    pbar = tqdm(total=self.maxiter, desc="REML", ncols=100)

                    def callback(theta):
                        pbar.update(1)
                        pbar.set_postfix({"theta": np.round(theta, 4)})

                result = minimize(
                    _sparse_z_reml_objective,
                    theta0,
                    args=(self.y, self.X, z_csc, ztz_csc, zty, ztx, z_cols),
                    method="L-BFGS-B",
                    bounds=bounds,
                    callback=callback,
                    options={"maxiter": self.maxiter},
                )
                if pbar is not None:
                    pbar.close()
                theta = np.asarray(result.x, dtype=float)
                if (not np.all(np.isfinite(theta))) or np.any(theta <= 0.0):
                    raise RuntimeError("Sparse Z REML returned invalid theta.")

                beta_hat, u_hat, z_fitted, u_by_Z, fitted, residuals, Vinvr, cov_beta = _sparse_z_fit_state(
                    theta,
                    self.y,
                    self.X,
                    z_csc,
                    ztz_csc,
                    z_cols,
                )
                z_theta = theta[:-1]
                g_theta = np.asarray([], dtype=float)
                g_scales: list[float] = []
                g_by_G: list[np.ndarray] = []
                g_total = np.zeros((self.n, 1), dtype=float)
                Kdiag_mean = [1.0 for _ in z_cols]
                self._z_standardized = False
                self.theta = theta
                self.var = np.array(
                    [float(theta[i]) * float(Kdiag_mean[i]) for i in range(len(z_cols))]
                    + [float(theta[-1])],
                    dtype=float,
                )
                self.beta = beta_hat
                self.u = u_hat
                self.u_by_Z = u_by_Z
                self.g = g_total
                self.g_by_G = g_by_G
                self._Vinvr = Vinvr
                self._z_theta = z_theta
                self._g_theta = g_theta
                self._g_scales = g_scales
                self._z_fitted = z_fitted
                self._cov_beta = cov_beta
                self.fitted = fitted
                self.residuals = residuals
                self.result = result
            else:
                Zstlist = [
                    np.asarray(
                        (_to_dense(z) - self.onehot_info[num][1])
                        / self.onehot_info[num][2]
                        / np.sqrt(self.onehot_info[num][0])
                    )
                    for num, z in enumerate(self.Z_list)
                ]
                Gnorm_list, g_scales = _normalize_G(self.G_list)
                Klist = [np.asarray(z @ z.T, dtype=float) for z in Zstlist] + Gnorm_list

                result = None
                theta: np.ndarray | None = None
                rust_fit = _fit_multi_kernel_ai_reml_rust(
                    self.y,
                    self.X,
                    Klist,
                    self.maxiter,
                )
                if rust_fit is not None:
                    theta, result = rust_fit
                else:
                    theta0 = np.ones(len(Klist) + 1, dtype=float)

                    pbar = None
                    callback = None
                    if self.progress:
                        pbar = tqdm(total=self.maxiter, desc="REML", ncols=100)

                        def callback(theta):
                            pbar.update(1)
                            pbar.set_postfix({"theta": np.round(theta, 4)})

                    result = minimize(
                        REML,
                        theta0,
                        args=(self.y, self.X, Klist),
                        method="L-BFGS-B",
                        callback=callback,
                        options={"maxiter": self.maxiter},
                    )
                    theta = np.asarray(result.x, dtype=float)
                    if pbar is not None:
                        pbar.close()
                if theta is None:
                    raise RuntimeError("REML optimization failed to produce theta.")

                V = theta[-1]*np.eye(self.n)
                for num, K in enumerate(Klist):
                    V += theta[num] * K
                L = np.linalg.cholesky(V)
                VinvX = cho_solve((L,True), self.X)
                Vinvy = cho_solve((L,True), self.y)
                beta_hat = np.linalg.solve(self.X.T @ VinvX, self.X.T @ Vinvy)
                r = self.y - self.X @ beta_hat
                Vinvr = cho_solve((L,True), r)

                z_term_count = len(Zstlist)
                g_term_count = len(Gnorm_list)
                z_theta = theta[:z_term_count]
                g_theta = theta[z_term_count : z_term_count + g_term_count]

                if z_term_count > 0:
                    Zall = np.concatenate(Zstlist, axis=1)
                    Gall = np.concatenate(
                        [z_theta[ind] * np.ones(z.shape[1]) for ind, z in enumerate(Zstlist)]
                    )
                    u_hat = (Zall.T @ Vinvr) * Gall[:, None]
                    z_fitted = Zall @ u_hat
                    u_by_Z = _split_u(u_hat, Zstlist)
                else:
                    u_hat = None
                    z_fitted = np.zeros((self.n, 1), dtype=float)
                    u_by_Z = []

                g_by_G: list[np.ndarray] = []
                g_total = np.zeros((self.n, 1), dtype=float)
                for idx, gk in enumerate(Gnorm_list):
                    gi = g_theta[idx] * (gk @ Vinvr)
                    g_by_G.append(gi)
                    g_total += gi

                fitted = self.X @ beta_hat + z_fitted + g_total
                sigma2 = float((r.T @ Vinvr)[0, 0]) / max(1, self.n - self.p)
                cov_beta = np.linalg.pinv(self.X.T @ VinvX) * sigma2

                self._z_standardized = True
                self.theta = theta
                self.var = np.array(
                    [theta[ind] * np.diag(Klist[ind]).mean() for ind in range(len(Klist))]
                    + [theta[-1]]
                )
                self.beta = beta_hat
                self.u = u_hat
                self.u_by_Z = u_by_Z
                self.g = g_total
                self.g_by_G = g_by_G
                self._Vinvr = Vinvr
                self._z_theta = z_theta
                self._g_theta = g_theta
                self._g_scales = g_scales
                self._z_fitted = z_fitted
                self._cov_beta = cov_beta
                self.fitted = fitted
                self.residuals = self.y - fitted
                self.result = result

    def predict(
        self,
        X: np.ndarray | None = None,
        Z: list[np.ndarray] | np.ndarray | None = None,
        G: list[np.ndarray] | np.ndarray | None = None,
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
        G : Sequence[np.ndarray] | np.ndarray | None
            Covariance matrices used for prediction.
            - in-sample: (n_train, n_train)
            - out-of-sample: cross-kernel (n_pred, n_train)

        Returns
        -------
        np.ndarray
            Predicted values with shape (n, 1).
        """
        if X is not None:
            n = X.shape[0]
        elif Z is not None:
            if isinstance(Z,np.ndarray):
                n = Z.shape[0]
            elif isinstance(Z,list):
                n = Z[0].shape[0]
            else:
                raise TypeError("Type error of Z matrix")
        elif G is not None:
            if isinstance(G, np.ndarray):
                n = G.shape[0]
            elif isinstance(G, list):
                n = G[0].shape[0]
            else:
                raise TypeError("Type error of G matrix")
        else:
            raise ValueError(f"Input X or Z for prediction") 
        X = _testX(X,n)
        Z_list = _testZ(Z,n) if Z is not None else []
        if len(self.Z_list) == 0 and len(self.G_list) == 0:  # 训练集无随机协变量 进入一般线性模型
            return X @ self.beta
        else:
            y_hat = X @ self.beta

            # Z-term contribution
            if len(self.Z_list) > 0:
                if len(Z_list) > 0:
                    if getattr(self, "_z_standardized", True):
                        Z_list = [
                            (_to_dense(z)-self.onehot_info[num][1])/self.onehot_info[num][2]/np.sqrt(self.onehot_info[num][0])
                            for num,z in enumerate(Z_list)
                        ]
                    else:
                        Z_list = [_to_dense(z) for z in Z_list]
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
                    y_hat = y_hat + Zall @ self.u
                elif n == self.n and self._z_fitted is not None:
                    y_hat = y_hat + self._z_fitted

            # G-term contribution
            if len(self.G_list) > 0:
                if G is None:
                    if n == self.n and self.g is not None:
                        y_hat = y_hat + self.g
                    else:
                        raise ValueError(
                            "G is required for out-of-sample prediction when model has G terms. "
                            "Provide cross-kernel matrix with shape (n_pred, n_train)."
                        )
                else:
                    G_pred_list = _testG_predict(G, n, self.n)
                    if len(G_pred_list) != self.g_count:
                        raise ValueError(
                            f"G count mismatch: expected {self.g_count}, got {len(G_pred_list)}"
                        )
                    if self._Vinvr is None or self._g_theta is None:
                        raise RuntimeError("Model is missing internal G-term state after fitting.")
                    g_eff = np.zeros((n, 1), dtype=float)
                    for idx, gmat in enumerate(G_pred_list):
                        g_std = (np.asarray(gmat, dtype=float) / self._g_scales[idx])
                        g_eff += self._g_theta[idx] * (g_std @ self._Vinvr)
                    y_hat = y_hat + g_eff

            return y_hat
def Gmatrix(mat:np.ndarray,):
    SDmat = mat.std(axis=1)
    mat = mat[SDmat>0]
    SDmat = SDmat[SDmat>0]
    z:np.ndarray = (mat - mat.mean(axis=1,keepdims=True)) / SDmat.reshape(-1,1)
    return z.T@z / mat.shape[0]

if __name__ == "__main__":
    from janusx.gfreader import load_genotype_chunks
    import pandas as pd
    import numpy as np
    from janusx.pyBLUP import kfold
    pheno = pd.read_csv('./example/mouse_hs1940.pheno',sep='\t',index_col=0).iloc[:,0].dropna().to_frame()
    chunks = load_genotype_chunks('./example/mouse_hs1940.vcf.gz',sample_ids=pheno.index.tolist())
    for chunk,site in chunks:
        gadd = chunk
    ghet = (gadd==1).astype(np.float32)
    
    for test,train in kfold(ghet.shape[1],5,seed=42):
        Gadd = Gmatrix(gadd[:,train])
        Ghet = Gmatrix(ghet[:,train])
        model = BLUP(pheno.values[train],G=[Gadd,Ghet],progress=False)
        yhat = model.predict(G=[Gmatrix(gadd)[test, :][:,train],Gmatrix(ghet)[test, :][:,train]])
        print('ADBLUP:',np.corrcoef(pheno.values[test,0],yhat[:,0])[0,1])
        model = BLUP(pheno.values[train],G=[Gadd],progress=False)
        yhat = model.predict(G=[Gmatrix(gadd)[test, :][:,train]])
        print('ABLUP:',np.corrcoef(pheno.values[test,0],yhat[:,0])[0,1])
        print('---')
