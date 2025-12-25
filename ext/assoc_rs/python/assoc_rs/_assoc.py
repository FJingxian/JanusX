# from assoc_rs import glmf32,lmm_reml_chunk_f32
import numpy as np
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                        message="invalid value encountered in")
from .assoc_rs import glmf32, lmm_reml_chunk_f32 # rust core


def FEM(y:np.ndarray,X:np.ndarray,M:np.ndarray,chunksize:int=50_000,threads:int=1,):
    '''
    # fastGLM for dtype int8
    
    :param y: trait vector (n,1)
    :type y: np.ndarray
    :param X: indice matrix of fixed effects (n,p)
    :type X: np.ndarray
    :param M: SNP matrix (m,n)
    :type M: np.ndarray
    :param chunksize: chunksize per step
    :type chunksize: int
    :param threads: number of threads
    :type threads: int
    
    :return: beta, se, pvalue
    '''
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    X = np.ascontiguousarray(X, dtype=np.float64)
    ixx = np.ascontiguousarray(np.linalg.pinv(X.T @ X), dtype=np.float64)
    M = np.ascontiguousarray(M, dtype=np.float32)
    if M.ndim != 2:
        raise ValueError("M must be 2D array with shape (m, n)")
    if M.shape[1] != y.shape[0]:
        raise ValueError(f"M must be shape (m, n). Got M.shape={M.shape}, but n=len(y)={y.shape[0]}")
    result:np.ndarray = glmf32(y,X,ixx,M,chunksize,threads)
    return result


def lmm_reml(S:np.ndarray, Xcov:np.ndarray, y_rot:np.ndarray, Dh:np.ndarray, snp_chunk:np.ndarray, bounds:tuple,
                       max_iter=30, tol=1e-2, threads=4):
    """
    Python wrapper for Rust function lmm_reml_chunk_f32.

    This function:
      1. Ensures correct shapes and dtypes for Rust (float64/float32)
      2. Rotates genotype chunk (snp_chunk @ Dh.T)
      3. Performs REML optimization for each SNP in the chunk
      4. Returns beta, se, p, lambda vectors

    Parameters
    ----------
    S : ndarray (n,)
        Eigenvalues of the kinship matrix (float64).
    Xcov : ndarray (n, q)
        Rotated covariates matrix: Dh @ X.
    y_rot : ndarray (n,)
        Rotated phenotype: Dh @ y.
    Dh : ndarray (n, n)
        Eigenvector matrix transpose (U^T).
    snp_chunk : ndarray (m_chunk, n)
        SNP genotype chunk BEFORE rotation.
        dtype can be int8/float32/float64.
    bounds : tuple (low, high)
        log10(lambda) lower and upper bounds.
    max_iter : int
        Max iterations for Brent optimization.
    tol : float
        Convergence tolerance in log10(lambda).
    threads : int
        Number of parallel worker threads.

    Returns
    -------
    beta_se_p : ndarray (m_chunk, 3)
        Columns: beta, se, p.
    lambdas : ndarray (m_chunk,)
        Estimated REML lambda for each SNP.
    """

    low, high = bounds

    # --- Convert all numpy arrays into valid Rust inputs ---
    S = np.ascontiguousarray(S, dtype=np.float64).ravel()
    Xcov = np.ascontiguousarray(Xcov, dtype=np.float64)
    y_rot = np.ascontiguousarray(y_rot, dtype=np.float64).ravel()

    # ----- Rotate genotype chunk: g_rot = snp_chunk @ Dh.T -----
    # snp_chunk: (m, n), Dh.T: (n, n)
    g_rot_chunk = snp_chunk @ Dh.T
    g_rot_chunk = np.ascontiguousarray(g_rot_chunk, dtype=np.float32)

    # ----- Call the Rust core function -----
    beta_se_p, lambdas = lmm_reml_chunk_f32(
        S,
        Xcov,
        y_rot,
        float(low),
        float(high),
        g_rot_chunk,
        max_iter,
        tol,
        threads
    )

    return beta_se_p, lambdas

class LMM:
    def __init__(self,y:np.ndarray=None,X:np.ndarray=None,kinship:np.ndarray=None):
        '''
        Fast Solve of Mixed Linear Model by Brent.
        
        :param y: Phenotype nx1\n
        :param X: Designed matrix for fixed effect nxp\n
        :param kinship: Calculation method of kinship matrix nxn
        '''
        y = y.reshape(-1,1) # ensure the dim of y
        X = np.concatenate([np.ones((y.shape[0],1)),X],axis=1) if X is not None else np.ones((y.shape[0],1))
        # Simplify inverse matrix
        val,vec = np.linalg.eigh(kinship + 1e-6 * np.eye(y.shape[0]))
        idx = np.argsort(val)[::-1]
        val,vec = val[idx],vec[:, idx]
        self.S,self.Dh = val, vec.T.astype(np.float32)
        del kinship
        self.Xcov = self.Dh@X
        self.y = self.Dh@y
        result = minimize_scalar(lambda lbd: -self._NULLREML(10**(lbd)),bounds=(-5,5),method='bounded',options={'xatol': 1e-3},)
        lbd_null = 10**(result.x)
        vg_null = np.mean(self.S)
        pve = vg_null/(vg_null+lbd_null)
        self.lbd_null = lbd_null
        self.pve = pve
        if pve > 0.999 or pve < 0.001:
            self.bounds = (-5,5)
        else:
            self.bounds = (np.log10(lbd_null)-2,np.log10(lbd_null)+2)
    def _NULLREML(self,lbd: float):
        '''Restricted Maximum Likelihood Estimation (REML) of NULL'''
        try:
            n,p_cov = self.Xcov.shape
            p = p_cov
            V = self.S+lbd
            V_inv = 1/V
            X_cov_snp = self.Xcov
            XTV_invX = V_inv*X_cov_snp.T @ X_cov_snp
            XTV_invy = V_inv*X_cov_snp.T @ self.y
            beta = np.linalg.solve(XTV_invX, XTV_invy)
            r = self.y - X_cov_snp@beta
            rTV_invr = (V_inv * r.T@r)[0,0]
            log_detV = np.sum(np.log(V))
            sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
            total_log = (n-p)*np.log(rTV_invr) + log_detV + log_detXTV_invX # log items
            c = (n-p)*(np.log(n-p)-1-np.log(2*np.pi))/2 # Constant
            return c - total_log / 2
        except Exception as e:
            print(f"REML error: {e}, lbd={lbd}")
            return -1e8
    def gwas(self,snp:np.ndarray=None,threads=1):
        '''
        Speed version of mlm
        
        :param snp: Marker matrix, np.ndarray, samples per rows and snp per columns
        :param chunksize: calculation number per times, int
        
        :return: beta coefficients, standard errors and p-values for each SNP, np.ndarray
        '''
        beta_se_p, lambdas = lmm_reml(
            self.S,             # eigenvalues of K
            self.Xcov,          # DH @ X
            self.y,             # DH @ y
            self.Dh,
            snp,      #g rotated: snp_chunk @ DH.T (float32)
            self.bounds,     # log10(lbd lower bound)
            max_iter=30,
            tol=1e-2,
            threads=threads
        )
        self.lbd = lambdas
        return beta_se_p

class LM:
    def __init__(self,y:np.ndarray=None,X:np.ndarray=None):
        self.y = y.reshape(-1,1) # ensure the dim of y
        self.X = np.concatenate([np.ones((y.shape[0],1)),X],axis=1) if X is not None else np.ones((y.shape[0],1))
    def gwas(self,snp:np.ndarray,threads:int=1):
        beta_se_p = FEM(self.y,self.X,snp,snp.shape[0],threads)[:,[0,1,-1]]
        return beta_se_p