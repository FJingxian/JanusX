import numpy as np
from joblib import Parallel,delayed,cpu_count
from tqdm import trange
try:
    from jxglm_rs import glmi8,mlmi8,mlmpi8
except Exception as e:
    print(f"{e}\nPlease build jxglm_rs for glmrc. Source code is in ext/glm_rs")

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

def REM(sz,n,pvalue,pos,M,y,X):
    bin_id = pos//sz
    order = np.lexsort((pvalue,bin_id),) # sort by bin_id, then sort by pvalue; return sorted idx
    lead = order[np.concatenate(([True], bin_id[order][1:] != bin_id[order][:-1]))]
    leadidx = np.sort(lead[np.argsort(pvalue[lead])[:n]])
    results = ll(y,M[leadidx].T,X)
    return -2*results['LL'],leadidx

def fastMLM(y, X, UT, iUXUX=None, vgs=1.0, G=None, step=10000, threads=0):
    """
    # Per-marker MLM test (G is int8, marker rows).
    
    :param y: (n,) float64
    :param X: (n,q0) float64
    :param UT: (n,n) float64  = U.T
    :param iUXUX: (q0,q0) float64  = pinv((UT@X).T @ (UT@X))
    :param vgs: scalar
    :param G: (m,n) int8
    
    :return: (m,3) float64 -> beta, se, p
    """
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    X = np.ascontiguousarray(X, dtype=np.float64)
    UT = np.ascontiguousarray(UT, dtype=np.float64)
    G = np.ascontiguousarray(G, dtype=np.int8)
    Uy = UT @ y
    UX = UT @ X
    if iUXUX is None:
        iUXUX = np.linalg.pinv(UX.T @ UX)
    iUXUX = np.ascontiguousarray(iUXUX, dtype=np.float64)
    UXUy = UX.T @ Uy
    Uy = np.ascontiguousarray(Uy, dtype=np.float64)
    UX = np.ascontiguousarray(UX, dtype=np.float64)
    UXUy = np.ascontiguousarray(UXUy, dtype=np.float64).ravel()
    return mlmi8(Uy, UX, iUXUX, UXUy, UT, G, float(vgs), step=int(step), threads=int(threads))

def poolMLM(y, X, UT, G_pool, vgs=1.0, ridge=1e-10):
    """
    # Multi-locus test: put all loci in G_pool into fixed effects simultaneously,
    
    :param y: (n,)
    :param X: (n,q0)
    :param UT: (n,n)
    :param G_pool: (k,n) int8 (rows are loci)
    :return: beta/se/p (k,3), for each locus in the pool.
    """
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    X = np.ascontiguousarray(X, dtype=np.float64)
    UT = np.ascontiguousarray(UT, dtype=np.float64)
    G_pool = np.ascontiguousarray(G_pool, dtype=np.int8)
    return mlmpi8(y, X, UT, G_pool, float(vgs), float(ridge))

def _pinv_safe(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    # ginv(beta1+beta2)
    return np.linalg.pinv(A, rcond=rcond)

def ll(pheno: np.ndarray,
        snp_pool: np.ndarray,
        X0: np.ndarray | None = None,
        deltaExpStart: float = -5.0,
        deltaExpEnd: float = 5.0,
        delta_step: float = 0.1,
        svd_eps: float = 1e-8,
        pinv_rcond: float = 1e-12):
    """
    Python rewrite of rMVP::FarmCPU.FaSTLMM.LL (grid-search delta, FaST-LMM style).

    Parameters
    ----------
    pheno : array-like, shape (n,) or (n,1)
        phenotype vector y
    snp_pool : array-like, shape (n, k)
        pseudo QTNs matrix (no missing, same taxa order)
    X0 : array-like, shape (n, p), optional
        covariates; if None -> intercept only
    deltaExpStart/deltaExpEnd/delta_step : float
        scan range in log-scale for delta, delta = exp(grid)
    svd_eps : float
        keep singular values > svd_eps
    pinv_rcond : float
        rcond for pinv used as ginv

    Returns
    -------
    dict with keys: beta, delta, LL, vg, ve
    """
    y = np.asarray(pheno, dtype=np.float64)
    if y.ndim == 2:
        y = y.reshape(-1, 1)
    else:
        y = y.reshape(-1, 1)
    snp_pool = np.asarray(snp_pool, dtype=np.float64)
    if snp_pool.ndim == 1:
        snp_pool = snp_pool.reshape(-1, 1)
    n = snp_pool.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"pheno n={y.shape[0]} != snp_pool n={n}")
    # R: if(!is.null(snp.pool)&&any(apply(snp.pool, 2, var)==0)) deltaExpStart=100; deltaExpEnd=100
    if snp_pool.size > 0:
        v = np.var(snp_pool, axis=0, ddof=1)  # R var uses n-1
        if np.any(v == 0):
            deltaExpStart = 100.0
            deltaExpEnd = 100.0

    if X0 is None:
        X = np.ones((n, 1), dtype=np.float64)
    else:
        X = np.asarray(X0, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != n:
            raise ValueError(f"X0 n={X.shape[0]} != snp_pool n={n}")
    # -------- SVD of snp_pool --------
    # R: K.X.svd <- svd(snp.pool)
    # R: d = d[d>1e-08]; d = d^2; U1 = u[, 1:length(d)]
    U, s, Vt = np.linalg.svd(snp_pool, full_matrices=False)
    keep = s > svd_eps
    s = s[keep]
    if s.size == 0:
        # 退化情况：pool 近似全零或秩=0（此时 U1 为空）
        # 让 U1 为空矩阵，按 R 后续逻辑会主要走 IU 部分
        U1 = np.zeros((n, 0), dtype=np.float64)
        d = np.zeros((0,), dtype=np.float64)
    else:
        d = s**2
        U1 = U[:, keep]  # (n, r)
    # handler of single snp: if(is.null(dim(U1))) U1=matrix(U1,ncol=1)
    # U1 in numpy is 2D
    r = U1.shape[1]  # length(d)
    # R: n=nrow(U1)   (就是个体数)
    # precompute terms
    # U1TX=crossprod(U1,X) = U1^T X   (r,p)
    # U1TY=crossprod(U1,y) = U1^T y   (r,1)
    U1TX = U1.T @ X               # (r,p)
    U1TY = U1.T @ y               # (r,1)
    # yU1TY <- y - U1 %*% U1TY
    # XU1TX <- X - U1 %*% U1TX
    yU1TY = y - (U1 @ U1TY)       # (n,1)
    XU1TX = X - (U1 @ U1TX)       # (n,p)
    # IU = -tcrossprod(U1); diag(IU)=1+diag(IU)
    # tcrossprod(U1) = U1 U1^T
    IU = - (U1 @ U1.T)            # (n,n)
    IU[np.diag_indices(n)] += 1.0
    # IUX=crossprod(IU,X) = IU^T X; IU is symmetric => IU X
    # IUY=crossprod(IU,y) = IU y
    IUX = IU.T @ X
    IUY = IU.T @ y
    # grid
    delta_range = np.arange(deltaExpStart, deltaExpEnd + 1e-12, delta_step, dtype=np.float64)
    best_LL = -np.inf
    best_beta = None
    best_delta = None
    # 为了更快：把循环里会重复用到的量预计算/缓存 shape
    p = X.shape[1]
    # -------- scan delta --------
    for expv in delta_range:
        delta = float(np.exp(expv))
        # ---- beta1: sum_i (u_i^T X)^T (u_i^T X) / (d_i + delta)
        # R里 one=matrix(U1TX[i,], nrow=1); beta=crossprod(one, one/(d[i]+delta))
        if r > 0:
            # U1TX: (r,p)
            # beta1 = Σ (U1TX[i]^T * U1TX[i])/(d[i]+delta)
            w = 1.0 / (d + delta)                         # (r,)
            beta1 = (U1TX.T * w) @ U1TX                   # (p,p)
            # beta3: Σ (U1TX[i]^T * U1TY[i])/(d[i]+delta)
            beta3 = (U1TX.T * w) @ U1TY                   # (p,1)
            # part12: Σ log(d_i + delta)
            part12 = float(np.sum(np.log(d + delta)))
        else:
            beta1 = np.zeros((p, p), dtype=np.float64)
            beta3 = np.zeros((p, 1), dtype=np.float64)
            part12 = 0.0
        # ---- beta2: Σ (IUX[row]^T IUX[row]) / delta = (IUX^T IUX)/delta
        beta2 = (IUX.T @ IUX) / delta                     # (p,p)
        # ---- beta4: Σ (IUX[row]^T IUY[row]) / delta = (IUX^T IUY)/delta
        beta4 = (IUX.T @ IUY) / delta                     # (p,1)
        # ---- final beta
        zw1 = _pinv_safe(beta1 + beta2, rcond=pinv_rcond)  # (p,p)
        zw2 = (beta3 + beta4)                              # (p,1)
        beta = zw1 @ zw2                                   # (p,1)
        # ---- LL part1
        part11 = n * np.log(2.0 * 3.14)
        part13 = (n - r) * np.log(delta)
        part1 = -0.5 * (part11 + part12 + part13)
        # ---- LL part2
        # part221 = Σ (U1TY[i] - U1TX[i]*beta)^2 / (d_i + delta)
        if r > 0:
            resid_u = U1TY - (U1TX @ beta)                # (r,1)
            part221 = float(np.sum((resid_u[:, 0] ** 2) / (d + delta)))
        else:
            part221 = 0.0

        # part222 = Σ (yU1TY[row] - XU1TX[row]*beta)^2 / delta
        resid_i = yU1TY - (XU1TX @ beta)                  # (n,1)
        part222 = float(np.sum(resid_i[:, 0] ** 2) / delta)

        part21 = n
        part22 = n * np.log((part221 + part222) / n)
        part2 = -0.5 * (part21 + part22)

        LL = float(part1 + part2)

        if LL > best_LL:
            best_LL = LL
            best_beta = beta.copy()
            best_delta = delta

    beta = best_beta
    delta = best_delta
    LL = best_LL
    # -------- vg / ve --------
    # sigma_a1 = Σ (U1TY[i] - U1TX[i]*beta)^2/(d_i+delta)
    if r > 0:
        resid_u = U1TY - (U1TX @ beta)
        sigma_a1 = float(np.sum((resid_u[:, 0] ** 2) / (d + delta)))
    else:
        sigma_a1 = 0.0
    # sigma_a2 = Σ (IUY[row] - IUX[row]*beta)^2 / delta
    resid_i2 = IUY - (IUX @ beta)
    sigma_a2 = float(np.sum(resid_i2[:, 0] ** 2) / delta)

    sigma_a = (sigma_a1 + sigma_a2) / n
    sigma_e = delta * sigma_a

    return {
        "beta": beta,      # (p,1)
        "delta": delta,
        "LL": LL,
        "vg": sigma_a,
        "ve": sigma_e,
    }
    
def SUPER(corr:np.ndarray,pval:list,thr:float=0.7):
    nqtn = corr.shape[0]
    keep = np.ones(nqtn, dtype=np.bool_)
    for i in range(nqtn): # 去冗余, 保留最显著QTN
        if keep[i]:
            row = corr[i]
            pi = pval[i]
            for j in range(i + 1, nqtn):
                if keep[j]:
                    cij = row[j]
                    if cij >= thr or cij <= -thr:
                        if pi >= pval[j]:
                            keep[i] = False # 保留pvalue更小的QTN
                        else:
                            keep[j] = False
                            break
    return keep

def farmcpu(y:np.ndarray=None,M:np.ndarray=None,X:np.ndarray=None,chrlist:np.ndarray=None,poslist:np.ndarray=None,
            szbin: list = [5e5,5e6,5e7],nbin: list = range(10,101,10), QTNbound: int = None,
            iteration:int=30,threshold:float=0.05,threads:int=1):
    '''
    Fast Solve of Mixed Linear Model by Brent.
    
    :param y: Phenotype nx1\n
    :param M: SNP matrix mxn\n
    :param X: Designed matrix for fixed effect nxp\n
    :param kinship: Calculation method of kinship matrix nxn
    '''
    threads = cpu_count() if threads == -1 else threads
    m,n = M.shape
    chrunique = np.unique(chrlist);chrdict = dict(zip(chrunique, range(len(chrunique)))) # transform chr to int
    pos = np.array([int(poslist[i]) + chrdict[chrlist[i]]*1e12 for i in range(len(chrlist))])
    QTNbound = np.sqrt(n/np.log10(n)) if QTNbound is None else QTNbound # max number of QTNs
    szbin = np.array(szbin)
    nbin = np.array(nbin)
    nbin[nbin > QTNbound] = QTNbound
    nbin = np.unique(nbin[nbin>0].astype(int))
    X = np.concatenate([np.ones((y.shape[0],1)),X],axis=1) if X is not None else np.ones((y.shape[0],1))
    QTNidx = np.array([],dtype=int) # init for QTNidx
    for _ in trange(iteration,desc=f'Process of FarmCPU',ascii=True):
        X_QTN = np.concatenate([X,M[QTNidx].T],axis=1) if QTNidx.size > 0 else X
        FEMresult = FEM(y,X_QTN,M,threads=threads)
        FEMresult[:,2:] = np.nan_to_num(FEMresult[:,2:],nan=1)
        QTNpval = FEMresult[:,2+X.shape[1]:-1].min(axis=0)
        FEMresult = FEMresult[:,-1]
        FEMresult[QTNidx] = QTNpval
        if np.sum(FEMresult <= threshold/m) == 0:
            break
        else:
            combine_list = []
            for sz in szbin:
                for n in nbin:
                    combine_list.append((sz,n))
            REMresult = Parallel(threads)(delayed(REM)(sz,n,FEMresult,pos,M,y,X_QTN) for sz,n in combine_list)
            optcombidx = np.argmin([l for l,idx in REMresult])
            QTNidx_pre = np.unique(np.concatenate([REMresult[optcombidx][1],QTNidx]))
            keep = SUPER(np.corrcoef(M[QTNidx_pre]),FEMresult[QTNidx_pre])
            QTNidx_pre = QTNidx_pre[keep]
            if np.array_equal(QTNidx_pre, QTNidx):
                break
            else:
                QTNidx = QTNidx_pre
    X_QTN = np.concatenate([X,M[QTNidx].T],axis=1)
    FEMresult = FEM(y,X_QTN,M,threads=threads)
    FEMresult[:,2:] = np.nan_to_num(FEMresult[:,2:],nan=1)
    QTNpval = FEMresult[:,2+X.shape[1]:-1].min(axis=0)
    beta_se = FEMresult[:,[0,1]]
    p = FEMresult[:,-1]
    p[QTNidx] = QTNpval
    return np.concatenate([beta_se,p.reshape(-1,1)],axis=1)