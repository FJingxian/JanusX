import numpy as np
try:
    from jxglm_rs import glmi8
except:
    raise "Please build jxglm_rs for glmrc. Source code is in ext/glm_rs"

def fastGLM(y:np.ndarray,X:np.ndarray,M:np.ndarray,chunksize:int=50_000,threads:int=1,):
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
    '''
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    X = np.ascontiguousarray(X, dtype=np.float64)
    ixx = np.ascontiguousarray(np.linalg.pinv(X.T @ X), dtype=np.float64)
    M = np.ascontiguousarray(M, dtype=np.int8)
    if M.ndim != 2:
        raise ValueError("M must be 2D array with shape (m, n)")
    if M.shape[1] != y.shape[0]:
        raise ValueError(f"M must be shape (m, n). Got M.shape={M.shape}, but n=len(y)={y.shape[0]}")
    return glmi8(y,X,ixx,M,chunksize,threads)
