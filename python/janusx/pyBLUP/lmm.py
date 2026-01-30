from typing import Any, Literal, Union
from joblib import Parallel,delayed
from numpy.typing import NDArray
import numpy as np
import scipy.linalg as la
from scipy.stats import norm
from scipy.optimize import minimize_scalar

from janusx.pyBLUP.assoc import (
    lmm_reml,
    lmm_reml_null,
    fastlmm_reml_null,
    fastlmm_reml,
    fastlmm_assoc_chunk,
    lmm_assoc_fixed
)

ArrayorNone = Union[NDArray[np.floating],None]

def GRM(SNPIter: Any,ridge: float=1e-6, method:Literal[1,2]=1) -> Union[NDArray[np.floating],Any]:
    '''
    GRM calculation for SNPIter generated from gfreader
    
    :param Mchunk: (mchunk,n)
    :type Mchunk: NDArray[np.floating]
    '''
    grm:np.ndarray
    Mchunk:np.ndarray
    z:np.ndarray
    if method == 1:
        sumvar:float = 0
        for niter,(Mchunk,_Site) in enumerate(SNPIter):
            z = (Mchunk-Mchunk.mean(axis=1,keepdims=True))
            sumvar += Mchunk.var(axis=1,keepdims=True).sum()
            if niter == 0:
                grm = z.T@z
            else:
                grm = grm + z.T@z
        return grm/sumvar
    elif method == 2:
        nsnp:int = 0
        for niter,(Mchunk,_Site) in enumerate(SNPIter):
            z = (Mchunk-Mchunk.mean(axis=1,keepdims=True))/(Mchunk.std(axis=1,keepdims=True)+ridge)
            nsnp+= Mchunk.shape[0]
            if niter == 0:
                grm = z.T@z
            else:
                grm = grm + z.T@z
        return grm/nsnp
    else:
        raise ValueError('Wrong in GRM function.')

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
                 Miter:Any=None,
                 M:ArrayorNone=None,
                 u:Any=None,s:Any=None,
                 kmethod:Literal[1,2]=1,
                 ridge:float=1e-8,
                 engine:Literal['rust','python']='rust',
                 bounds:tuple=(-6,6),fixedlbd:bool=False,threads:int=4) -> None:
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
        lambda : float
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
        result:Any
        self.engine = engine
        self.threads = threads
        self.fixedlbd = fixedlbd
        self.ridge = ridge
        self.ml = None
        if grm is not None or Miter is not None: # Full matrice for LMM: pre-computed GRM or Miter to build GRM
            self.FaST:bool = False
            if grm is None:
                if Miter is None:
                    raise ValueError("Either grm or Miter must be provided.")
                grm = GRM(Miter, ridge=self.ridge, method=kmethod)
            self.ss:NDArray;self.u:NDArray
            ss, self.u = la.eigh(grm, overwrite_a=True, check_finite=False)
            self.ss = np.maximum(ss, 0.0)
            self.uty = self.u.T@y
            self.utx = self.u.T@X
            if self.engine == 'python':
                result = minimize_scalar(lambda loglbd: logREML(loglbd,
                                                                None,
                                                                self.uty,self.utx,
                                                                self.ss),
                                        bounds=bounds,method='bounded') # minimize -logREML
                lbd,_reml = np.power(10,result.x),result.fun
            elif self.engine == 'rust':
                lbd, self.ml, _reml = lmm_reml_null(self.ss, self.utx, self.uty, bounds)
            vinv = 1/(self.ss+lbd)
            XTVinvX = (self.utx.T * vinv) @ self.utx
            XTVinvy = (self.utx.T * vinv) @ self.uty
            self.beta = np.linalg.solve(XTVinvX, XTVinvy)
            self.g = self.ss * vinv * self.u @ (self.uty-self.utx@self.beta)
        else:
            self.FaST:bool = True
            self.ss:NDArray;self.u:NDArray
            if u is not None and s is not None:
                pass
            elif M is not None: # FaST for LMM
                try:
                    u, s, _vt = la.svd(M, full_matrices=False, lapack_driver="gesdd")
                except la.LinAlgError:
                    u, s, _vt = la.svd(M, full_matrices=False, lapack_driver="gesvd")
            else:
                raise ValueError("u and s are both None and M is also None.")
            self.u = u
            self.ss = s*s
            self.uty = self.u.T@y
            self.utx = self.u.T@X
            self.u2ty = y-self.u@self.uty
            self.u2tx = X-self.u@self.utx
            if self.engine == 'python':
                result = minimize_scalar(lambda loglbd: FaSTlogREML(loglbd,
                                                                    None,None,
                                                                    self.uty,self.u2ty,self.utx,self.u2tx,
                                                                    self.ss),
                                        bounds=bounds,method='bounded') # minimize -logREML
                lbd,_reml = np.power(10,result.x),result.fun
            elif self.engine == 'rust':
                lbd, self.ml, _reml = fastlmm_reml_null(self.ss, self.utx, self.u2tx, self.uty, self.u2ty, bounds)
            v1inv = 1/(self.ss+lbd)
            v2inv = 1/lbd
            XTVinvX = (self.utx.T * v1inv) @ self.utx + v2inv * (self.u2tx.T @ self.u2tx)
            XTVinvy = (self.utx.T * v1inv) @ self.uty + v2inv * (self.u2tx.T @ self.u2ty)
            self.beta = np.linalg.solve(XTVinvX, XTVinvy)
            self.g = self.ss * v1inv * self.u @ (self.uty-self.utx@self.beta)
        varg = varg = self.ss.sum()/n#(trace(K)/n)
        vare = lbd
        self.loglbd = np.log10(lbd)
        self.pve = varg/(varg+vare)
        self.bounds = (np.log10(lbd)-2,np.log10(lbd)+2) if self.pve >=0.05 and self.pve <= 0.95 else bounds
    def gwas(self,Mchunk:np.ndarray, plrt: bool=False):
        """
        Run GWAS on SNP iterator.

        Notes
        -----
        - Rust backend returns columns: beta, se, pwald (and plrt only if nullml is provided).
        - This wrapper currently does not pass nullml, so results are beta/se/pwald.
        """
        if not plrt:
            self.ml = None
        def _FaSTprocess(u1tsnp,u2tsnp):
            result = minimize_scalar(lambda loglbd: FaSTlogREML(loglbd,
                                                                u1tsnp,u2tsnp,
                                                                self.uty,self.u2ty,self.utx,self.u2tx,
                                                                self.ss),
                                          bounds=self.bounds,method='bounded') # minimize -logREML
            lbd = np.power(10,result.x)
            v1inv = 1/(self.ss+lbd)
            v2inv = 1/lbd
            utx = np.column_stack([self.utx,u1tsnp])
            u2tx = np.column_stack([self.u2tx,u2tsnp])
            XTVinvX = (utx.T * v1inv) @ utx + v2inv * (u2tx.T @ u2tx)
            XTVinvy = (utx.T * v1inv) @ self.uty + v2inv * (u2tx.T @ self.u2ty)
            beta = np.linalg.solve(XTVinvX, XTVinvy)
            # residual in rotated spaces
            r1 = self.uty - utx @ beta;r2 = self.u2ty - u2tx @ beta                   # (n-k,1)

            # r^T V^-1 r
            rtv_invr = float(np.sum(v1inv * (r1.ravel() ** 2)) + v2inv * np.sum(r2.ravel() ** 2))

            # sigma^2 hat
            n = self.u2ty.shape[0]
            p = utx.shape[1]
            sigma2 = rtv_invr / (n - p)

            # var(beta) = sigma2 * inv(XtVinvX)
            XtVinvX_inv = np.linalg.inv(XTVinvX)
            se = float(np.sqrt(sigma2 * XtVinvX_inv[-1, -1]))

            alpha = float(beta[-1, 0])
            z = alpha / se
            pval = 2.0 * norm.sf(abs(z))  # two-sided

            return alpha, se, pval
        def _process(utsnp):
            result = minimize_scalar(lambda loglbd: logREML(loglbd,
                                                                utsnp,
                                                                self.uty,self.utx,
                                                                self.ss),
                                          bounds=self.bounds,method='bounded') # minimize -logREML
            lbd = np.power(10,result.x)
            v1inv = 1/(self.ss+lbd)
            utx = np.column_stack([self.utx,utsnp])
            XTVinvX = (utx.T * v1inv) @ utx
            XTVinvy = (utx.T * v1inv) @ self.uty
            beta = np.linalg.solve(XTVinvX, XTVinvy)
            # residual in rotated spaces
            r1 = self.uty - utx @ beta

            # r^T V^-1 r
            rtv_invr = float(np.sum(v1inv * (r1.ravel() ** 2)))

            # sigma^2 hat
            n,p = utx.shape
            sigma2 = rtv_invr / (n - p)

            # var(beta) = sigma2 * inv(XtVinvX)
            XtVinvX_inv = np.linalg.inv(XTVinvX)
            se = float(np.sqrt(sigma2 * XtVinvX_inv[-1, -1]))

            alpha = float(beta[-1, 0])
            z = alpha / se
            pval = 2.0 * norm.sf(abs(z))  # two-sided

            return alpha, se, pval
        Mchunk = (Mchunk - Mchunk.mean(axis=1,keepdims=True)) / (Mchunk.std(axis=1,keepdims=True)+self.ridge)
        if self.FaST:
            u1Mchunk = Mchunk@self.u
            u2Mchunk = Mchunk-u1Mchunk@self.u.T
            if self.engine == 'python':
                beta_se_p = list(Parallel(n_jobs=self.threads)
                            (delayed(_FaSTprocess)(u1Mchunk[i],u2Mchunk[i]) for i in range(u1Mchunk.shape[0])))
            elif self.engine == 'rust':
                if not self.fixedlbd:
                    beta_se_p = fastlmm_reml(self.ss, self.utx, self.u2tx,
                                                self.uty, self.u2ty,
                                                u1Mchunk, u2Mchunk,
                                                bounds=self.bounds,threads=self.threads,nullml=self.ml)
                else:
                    beta_se_p = fastlmm_assoc_chunk(self.ss,self.utx,self.u2tx,self.uty,self.u2ty,self.loglbd,
                                        u1Mchunk, u2Mchunk,
                                        threads=self.threads,nullml=self.ml)
        else:
            uMchunk = Mchunk @ self.u
            if self.engine == 'python':
                beta_se_p = list(Parallel(n_jobs=self.threads)
                            (delayed(_process)(i) for i in uMchunk))
            elif self.engine == 'rust':
                if not self.fixedlbd:
                    beta_se_p = lmm_reml(self.ss, self.utx, self.uty,
                                        uMchunk,
                                        bounds=self.bounds,threads=self.threads,nullml=self.ml)
                else:
                    beta_se_p = lmm_assoc_fixed(self.ss, self.utx, self.uty,self.loglbd,
                                        uMchunk,threads=self.threads,nullml=self.ml)
        return np.array(beta_se_p)

# ------------------------- quick test -------------------------
if __name__ == "__main__":
    pass
