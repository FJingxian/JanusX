import typing
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
# from jax.scipy.optimize import minimize
from jax import grad
from scipy.optimize import minimize
from tqdm import tqdm

def REML(
    theta: jnp.ndarray,
    y: jnp.ndarray,
    X: jnp.ndarray,
    Zlist: typing.List[np.ndarray],
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
    V = theta[-1]*jnp.eye(n)
    for num,Z in enumerate(Zlist):
        V += theta[num] * Z@Z.T
    L = jnp.linalg.cholesky(V)
    VinvX = cho_solve((L,True),X)
    Vinvy = cho_solve((L,True),y)
    XTV_invX = X.T @ VinvX
    XTV_invy = X.T @ Vinvy

    beta = jnp.linalg.solve(XTV_invX, XTV_invy)
    r = y - X @ beta
    Vinvr = cho_solve((L,True),r)

    rTV_invr = (r.T @ Vinvr)[0, 0]
    log_detV = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    sign, log_detXTV_invX = jnp.linalg.slogdet(XTV_invX)
    total_log = (n - p) * jnp.log(rTV_invr) + log_detV + log_detXTV_invX

    # Constant term (matches your original expression)
    c = (n - p) * (jnp.log(n - p) - 1 - jnp.log(2 * jnp.pi)) / 2.0
    return total_log / 2.0 - c

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

def _testZ(Z: list[np.ndarray]|np.ndarray|None,n:int|None=None) -> list[np.ndarray]:
    Z_list:list[np.ndarray]
    if Z is None:
        Z_list = []
    elif isinstance(Z, np.ndarray):
        Z_list = [Z]
    else:
        Z_list = list(Z)
    for idx, z in enumerate(Z_list):
        if not isinstance(z,np.ndarray):
            raise TypeError(f"Type of Z{idx} is {type(z)}")
        if z.shape[0] == n:
            pass
        elif z.size == n:
            Z_list[idx] = z.reshape(-1,1)
        else:
            raise ValueError(f"Row number of Z{idx} is not equal to {n} in y. (Z{idx}.shape={z.shape}, n={n})")
    return Z_list

def _onehotZ(Z_list: typing.Sequence[np.ndarray],ridge:float=1e-8) -> list[typing.Tuple[float,float,float]]:
    return list(map(lambda num: (1,0.0,1.0) if (np.isin(Z_list[num], [0, 1]).all()) and (Z_list[num].sum(axis=1) > 0).all() \
        else (Z_list[num].shape[1],Z_list[num].mean(axis=0),Z_list[num].std(axis=0)+ridge),range(len(Z_list))))

def _split_u(u: np.ndarray, Z_list: list[np.ndarray]) -> list[np.ndarray]:
    u_by_Z: list[np.ndarray] = []
    start = 0
    for z in Z_list:
        end = start + z.shape[1]
        u_by_Z.append(u[start:end])
        start = end
    return u_by_Z

def dynamicEigh(Zlist:typing.List[np.ndarray]):
    qdim = [z.shape[1] for z in Zlist]
    idx2eigh = np.argmax(qdim)
    
    return

class BLUP:
    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        Z: list[np.ndarray] | np.ndarray | None = None,
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
        # 确保表型 y 参数 (n,1)
        y = y.reshape(-1,1)
        n = y.shape[0] # Row dimension
        # 检查固定效应 X 参数 (n,p) | None
        X = _testX(X,n)
        p = X.shape[1]
        # 检查随机效应 Z 参数 [(n,q1),(n,q2),...(n,qn)] | (n,q) | None
        self.Z_list = _testZ(Z,n)
        self.onehot_info = _onehotZ(self.Z_list)
        self.z_cols = [z.shape[1] for z in self.Z_list]
        self.y = y
        self.X = X
        self.n = n
        self.p = p
        self.maxiter = maxiter
        self.progress = progress
        self._fit()

    def _fit(self) -> None:
        if len(self.Z_list) == 0: # 无随机协变量 进入一般线性模型
            self.theta = None
            self.V = None
            self.beta = np.linalg.solve(self.X.T@self.X,self.X.T@self.y)
            self.u = None
            self.u_by_Z = None
            self.fitted = self.X@self.beta
            self.residuals = self.y - self.fitted 
            self.result = None
            pass
        else:
            Zstlist = [jnp.array((z-self.onehot_info[num][1])/self.onehot_info[num][2]/np.sqrt(self.onehot_info[num][0])) for num,z in enumerate(self.Z_list)]
            theta0 = jnp.ones(len(Zstlist)+1)
            gradfn = grad(REML)

            pbar = None
            callback = None
            if self.progress:
                pbar = tqdm(total=self.maxiter, desc="REML", ncols=100)

                def callback(theta):
                    pbar.update(1)
                    pbar.set_postfix({"theta": np.round(theta, 4)})

            def jac(theta, y, X, Zstlist):
                return jnp.asarray(gradfn(theta, y, X, Zstlist))

            result = minimize(
                REML,
                theta0,
                args=(self.y, self.X, Zstlist),
                jac=jac,
                method="L-BFGS-B",
                callback=callback,
                options={"maxiter": self.maxiter},
            )
            theta = np.asarray(result.x, dtype=float)
            if pbar is not None:
                pbar.close()
            
            V = theta[-1]*np.eye(self.n)
            for num,Z in enumerate(Zstlist):
                V += theta[num] * Z@Z.T
            L = np.linalg.cholesky(V)
            VinvX = cho_solve((L,True), self.X)
            Vinvy = cho_solve((L,True), self.y)
            beta_hat = np.linalg.solve(self.X.T @ VinvX, self.X.T @ Vinvy)
            r = self.y - self.X @ beta_hat
            Vinvr = cho_solve((L,True), r)

            Zall = np.concatenate(Zstlist, axis=1)
            Gall = np.concatenate(
                [theta[ind] * np.ones(z.shape[1]) for ind, z in enumerate(Zstlist)]
            )
            u_hat = (Zall.T @ Vinvr) * Gall[:, None]
            fitted = self.X @ beta_hat + Zall @ u_hat
            u_by_Z = _split_u(u_hat, Zstlist)

            self.theta = theta
            self.var = np.array([theta[ind]*np.diag(z@z.T).mean() for ind,z in enumerate(Zstlist)]+[theta[-1]])
            self.beta = beta_hat
            self.u = u_hat
            self.u_by_Z = u_by_Z
            self.fitted = fitted
            self.residuals = self.y - fitted
            self.result = result

    def predict(
        self,
        X: np.ndarray | None = None,
        Z: list[np.ndarray] | np.ndarray | None = None,
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
        if X is not None:
            n = X.shape[0]
        elif Z is not None:
            if isinstance(Z,np.ndarray):
                n = Z.shape[0]
            elif isinstance(Z,list):
                n = Z[0].shape[0]
            else:
                raise "Type error of Z matrix"
        else:
            raise ValueError(f"Input X or Z for prediction") 
        X = _testX(X,n)
        Z_list = _testZ(Z,n)
        if len(self.Z_list) == 0:  # 训练集无随机协变量 进入一般线性模型
            return X @ self.beta
        else:
            Z_list = [(z-self.onehot_info[num][1])/self.onehot_info[num][2]/np.sqrt(self.onehot_info[num][0]) for num,z in enumerate(Z_list)]
            y_hat = X @ self.beta
            if len(Z_list) == 0:
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
    import pandas as pd
    geno = vcfreader('example/mouse_hs1940.vcf.gz',maf=0.02,miss=0.05).iloc[:,2:]
    pheno = pd.read_csv('example/mouse_hs1940.pheno',sep='\t',index_col=0).iloc[:,0].dropna()
    csample = list(set(geno.columns) & set(pheno.index))
    y = pheno.loc[csample].values.reshape(-1,1)
    Z = geno[csample].values.T
    data = pd.read_csv('/Users/jingxianfu/Public/SCR.test.tsv',sep='\t',).dropna().reset_index().iloc[:,1:]
    y = data.iloc[:,[1]].values
    Z = [pd.get_dummies(data.iloc[:,0]).astype(int).values,pd.get_dummies(data.iloc[:,2]).astype(int).values]
    model = BLUP(y, None, Z=Z, maxiter=50, progress=True)
    print(model.beta)
    print(model.u.shape)
    print(model.var[0]/model.var.sum())
    print(np.corrcoef(y.ravel(),model.predict(None,Z).ravel()))