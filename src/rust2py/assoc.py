try:
    from jxglm_rs import glmi8
except Exception as e:
    print(f"{e}\nPlease build jxglm_rs for glmrc. Source code is in ext/glm_rs")
try:
    from lmm_rs import lmm_reml_chunk_f32
except Exception as e:
    print(f"{e}\nPlease build lmm_rs for lmm_reml_chunk_f32. Source code is in ext/lmm_rs")
import numpy as np   


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
    M = np.ascontiguousarray(M, dtype=np.int8)
    if M.ndim != 2:
        raise ValueError("M must be 2D array with shape (m, n)")
    if M.shape[1] != y.shape[0]:
        raise ValueError(f"M must be shape (m, n). Got M.shape={M.shape}, but n=len(y)={y.shape[0]}")
    result:np.ndarray = glmi8(y,X,ixx,M,chunksize,threads)
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