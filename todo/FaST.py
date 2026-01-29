from typing import Any, Union
from numpy.typing import NDArray
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar

ArrayorNone = Union[NDArray[np.floating],None]

def FaSTlogREML(
    loglbd: float,
    U1tsnp: ArrayorNone,
    U2tsnp:ArrayorNone,
    U1ty: np.ndarray,
    U2ty: np.ndarray,
    U1tx: np.ndarray,
    U2tx: np.ndarray,
    S: np.ndarray
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
    lbd = np.power(10,loglbd)
    v1 = S+lbd
    v1_inv = 1 / v1
    v2 = lbd
    v2_inv = 1 / lbd
    if (U1tsnp is None) ^ (U2tsnp is None):
        raise ValueError("U1tsnp and U2tsnp must be both provided or both None.")
    if U1tsnp is not None and U2tsnp is not None:
        U1tx = np.column_stack([U1tx, U1tsnp])
        U2tx = np.column_stack([U2tx, U2tsnp])

    p = U1tx.shape[1]
    k = v1.size
    n = U2ty.shape[0]

    XTV_invX = (U1tx.T * v1_inv) @ U1tx + v2_inv * (U2tx.T @ U2tx)
    XTV_invy = (U1tx.T * v1_inv) @ U1ty + v2_inv * (U2tx.T @ U2ty)

    beta = np.linalg.solve(XTV_invX, XTV_invy)
    r1: np.ndarray = U1ty - U1tx @ beta
    r2: np.ndarray = U2ty - U2tx @ beta

    rTV_invr = np.sum(v1_inv * np.ravel(r1) ** 2) + v2_inv * np.sum(np.ravel(r2) ** 2)
    
    log_detV = np.sum(np.log(v1)) + (n - k) * np.log(v2)
    sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
    total_log = (n - p) * np.log(rTV_invr) + log_detV + log_detXTV_invX

    # Constant term (matches your original expression)
    c = (n - p) * (np.log(n - p) - 1 - np.log(2 * np.pi)) / 2.0
    return total_log / 2.0 - c

def logREML(
    loglbd: float,
    Utsnp: ArrayorNone,
    Uty: np.ndarray,
    Utx: np.ndarray,
    S: np.ndarray
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
    lbd = np.power(10,loglbd)
    v = S+lbd
    v_inv = 1 / v
    if Utsnp is not None:
        Utx = np.column_stack([Utx, Utsnp])
    n,p = Utx.shape
    XTV_invX = (Utx.T * v_inv) @ Utx
    XTV_invy = (Utx.T * v_inv) @ Uty

    beta = np.linalg.solve(XTV_invX, XTV_invy)
    r: np.ndarray = Uty - Utx @ beta

    rTV_invr = np.sum(v_inv * np.ravel(r) ** 2)
    log_detV = np.sum(np.log(v))
    sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
    total_log = (n - p) * np.log(rTV_invr) + log_detV + log_detXTV_invX

    # Constant term (matches your original expression)
    c = (n - p) * (np.log(n - p) - 1 - np.log(2 * np.pi)) / 2.0
    return total_log / 2.0 - c


class LMM:
    def __init__(self,y:NDArray[np.float32],
                 X:ArrayorNone=None,
                 grm:ArrayorNone=None,
                 Miter:ArrayorNone=None,
                 M:ArrayorNone=None,
                 u:ArrayorNone=None,s:Any=None,
                 ridge:float=1e-8) -> None:
        """
        Linear Mixed Model (LMM) with spectral / FaST-LMM-style acceleration.

        Parameters
        ----------
        y : NDArray[np.float32]
            Phenotype vector of shape ``(n,)`` or ``(n, 1)``.
        X : NDArray[np.floating] | None, optional
            Fixed-effect covariate matrix of shape ``(n, p)`` **without** an
            intercept column. An intercept is appended internally.
            If ``None``, intercept-only model is used.
        grm : NDArray[np.floating] | None, optional
            Genomic relationship / kinship matrix :math:`K` of shape ``(n, n)``.
            If provided, the model uses eigendecomposition of ``grm``.
            Either ``grm`` or one of ``M`` / (``u``, ``s``) must be provided.
        M : NDArray[np.floating] | None, optional
            Standardized genotype matrix used to represent :math:`K` implicitly.
            Expected shape is ``(n, m)`` (rows are samples, columns are markers).
            Internally, columns are centered and scaled, and the matrix is further
            scaled by ``1/sqrt(m)`` so that ``K ≈ M @ M.T``.
            Only used when ``grm`` is ``None`` and (``u``, ``s``) are not given.
        u : NDArray[np.floating] | None, optional
            Precomputed left singular vectors / eigenvectors in sample space with
            shape ``(n, k)``. This is typically the ``U`` from ``SVD(M)`` or the
            eigenvectors of :math:`K`. Only used when ``grm`` and ``M`` are
            ``None``.
        s : Any, optional
            Singular values corresponding to ``u``. If you pass singular values
            from ``SVD(M)``, then the implied eigenvalues of :math:`K` are
            :math:`S = s^2`. You may pass an array-like of length ``k``.
        ridge : float, optional
            Small stabilizer added to standard deviation during genotype
            standardization to avoid division by zero for near-monomorphic
            markers. Default is ``1e-8``.

        Notes
        -----
        - Exactly one representation of relatedness should be used:
          ``grm`` OR ``M`` OR (``u``, ``s``).
        - In FaST-LMM mode, this implementation follows the spectral-rotation
          idea of:

          Lippert, C., Listgarten, J., Liu, Y., et al. (2011).
          *FaST linear mixed models for genome-wide association studies*.
          Nature Methods, 8, 833-835. doi:10.1038/nmeth.1681

        Attributes (after fitting)
        --------------------------
        beta : NDArray[np.floating]
            Estimated fixed effects (including intercept), shape ``(p+1, 1)``.
        log10_lambda : float
            Estimated :math:`\\log_{10}(\\lambda)`.
        lambda_ : float
            Estimated :math:`\\lambda = \\sigma_e^2/\\sigma_g^2`.
        g : NDArray[np.floating]
            Conditional expectation of the genetic effect,
            :math:`\\hat g = \\mathbb{E}[g\\mid y]`, shape ``(n, 1)``.
        fitted : NDArray[np.floating]
            Fitted values ``X @ beta + g``, shape ``(n, 1)``.
        residuals : NDArray[np.floating]
            Residuals ``y - fitted``, shape ``(n, 1)``.
        result : scipy.optimize.OptimizeResult
            Result object returned by the REML 1D optimizer.

        Raises
        ------
        ValueError
            If none of ``grm``, ``M``, or (``u``, ``s``) is provided, or if input
            shapes are incompatible (e.g., ``X.shape[0] != n``).
        """
        y = y.reshape(-1,1) if y.ndim == 1 else y
        n = y.size
        if X is not None:
            assert X.shape[0] == n, ValueError(f"Shape 0 of X is {X.shape}, and not same with y.")
            X = np.column_stack([X,np.ones(shape=(n,1),dtype='float32')])
        else:
            X = np.ones(shape=(n,1),dtype='float32')
        self.result:Any
        if grm is not None or Miter is not None: # Full matrice for LMM: pre-computed GRM or Miter to build GRM
            self.ss:NDArray;self.u:NDArray
            ss, self.u = la.eigh(grm, overwrite_a=True, check_finite=False)
            self.ss = np.maximum(ss, 0.0)
            self.uty = self.u.T@y
            self.utx = self.u.T@X
            self.result = minimize_scalar(lambda loglbd: logREML(loglbd,
                                                            None,
                                                            self.uty,self.utx,
                                                            self.ss),
                                    bounds=(-6,6),method='bounded') # minimize -logREML
            lbd = np.power(10,self.result.x)
            vinv = 1/(self.ss+lbd)
            XTVinvX = (self.utx.T * vinv) @ self.utx
            XTVinvy = (self.utx.T * vinv) @ self.uty
            self.beta = np.linalg.solve(XTVinvX, XTVinvy)
            self.g = self.ss * vinv * self.u @ (self.uty-self.utx@self.beta)
        else:
            if M is not None: # FaST for LMM
                self.ss:NDArray;self.u:NDArray
                try:
                    self.u, s, _vt = la.svd(M, full_matrices=False, lapack_driver="gesdd")
                except la.LinAlgError:
                    self.u, s, _vt = la.svd(M, full_matrices=False, lapack_driver="gesvd")
                self.ss = s*s
                self.uty = self.u.T@y
                self.utx = self.u.T@X
                self.u2ty = y-self.u@self.uty
                self.u2tx = X-self.u@self.utx
                self.result = minimize_scalar(lambda loglbd: FaSTlogREML(loglbd,
                                                                    None,None,
                                                                    self.uty,self.u2ty,self.utx,self.u2tx,
                                                                    self.ss),
                                        bounds=(-8,8),method='bounded') # minimize -logREML
            elif u is not None and s is not None:
                self.u = u
                self.ss = s*s
                self.uty = self.u.T@y
                self.utx = self.u.T@X
                self.u2ty = y-self.u@self.uty
                self.u2tx = X-self.u@self.utx
                self.result = minimize_scalar(lambda loglbd: FaSTlogREML(loglbd,
                                                                    None,None,
                                                                    self.uty,self.u2ty,self.utx,self.u2tx,
                                                                    self.ss),
                                        bounds=(-6,6),method='bounded') # minimize -logREML
            else:
                raise ValueError("Genotype Relationship Matrix and Genotype Matrix must be provided at least one.")
            lbd = np.power(10,self.result.x)
            v1inv = 1/(self.ss+lbd)
            v2inv = 1/lbd
            XTVinvX = (self.utx.T * v1inv) @ self.utx + v2inv * (self.u2tx.T @ self.u2tx)
            XTVinvy = (self.utx.T * v1inv) @ self.uty + v2inv * (self.u2tx.T @ self.u2ty)
            self.beta = np.linalg.solve(XTVinvX, XTVinvy)
            self.g = self.ss * v1inv * self.u @ (self.uty-self.utx@self.beta)
        varg = self.ss*self.u@self.u/n
        vare = lbd
        self.pve = varg/(varg+vare)
    def gwas(self,SNPiter):
        return

# ------------------------- quick test -------------------------
if __name__ == "__main__":
    from janusx.gfreader import vcfreader
    from janusx.pyBLUP.QK2 import GRM
    import pandas as pd
    geno = vcfreader('example/mouse_hs1940.vcf.gz',maf=0.02,miss=0.05,dtype='float32').iloc[:,2:]
    pheno = pd.read_csv('example/mouse_hs1940.pheno',sep='\t',index_col=0).iloc[:,0].dropna()
    csample = list(set(geno.columns) & set(pheno.index))
    y:np.ndarray = pheno.loc[csample].to_numpy()
    y = y.reshape(-1,1)
    X = None
    M = geno[csample].values
    M = (M-M.mean(axis=1,keepdims=True))/M.std(axis=1,keepdims=True)/np.sqrt(M.shape[0])
    grm = GRM(M)
    u,s,vh = la.svd(M.T,full_matrices=False)
    print(u.shape,s.shape,vh.shape)
    model = LMM(y,X,grm=grm)
    print('* Full')
    print(model.result.fun)
    print(model.result.x)
    print(model.beta)
    print(model.g)
    model = LMM(y,X,M=M.T)
    print('* Full (FaST-LMM)')
    print(model.result.fun)
    print(model.result.x)
    print(model.beta)
    print(model.g)
    # for dim in range(20,1500,100):
    #     print('*',dim)
    #     model = LMM(y,X,u=u[:,:dim],s=s[:dim])
    #     print(model.result.fun)
    #     print(model.result.x)