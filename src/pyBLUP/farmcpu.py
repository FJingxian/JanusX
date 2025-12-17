from tqdm import trange
from pyBLUP import GRM
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from joblib import Parallel,delayed

def FEM(snp:np.ndarray,X:np.ndarray,y:np.ndarray):
    '''
    solving beta and its se in multiprocess
    '''
    X = np.column_stack([X, snp])
    XTX = X.T@X
    try:
        XTX_inv = np.linalg.inv(XTX)
    except:
        XTX_inv = np.linalg.inv(XTX+np.eye(XTX.shape[0]))
    XTy = X.T@y
    beta = XTX_inv@XTy
    r = (y-X@beta)
    se = np.sqrt((r.T@r)/(X.shape[0]-XTX.shape[0])*XTX_inv[-1,-1])
    return beta[-1,0],se[0,0]

def REM(lbd:float,y:np.ndarray,Xtrans:np.ndarray,Ktrans:np.ndarray):
    '''
    Restrict Estimate Maximum Likelyhood Function
    '''
    n,p_cov = Xtrans.shape
    p = p_cov
    V = Ktrans+lbd
    V_inv = 1/V
    XTV_invX = V_inv*Xtrans.T @ Xtrans + 1e-8*np.eye(Xtrans.shape[1])
    XTV_invy = V_inv*Xtrans.T @ y
    beta = np.linalg.solve(XTV_invX, XTV_invy)
    r = y - Xtrans@beta
    rTV_invr = (V_inv * r.T@r)[0,0]
    log_detV = np.sum(np.log(V))
    sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
    total_log = (n-p)*np.log(rTV_invr) + log_detV + log_detXTV_invX # log items
    c = (n-p)*(np.log(n-p)-1-np.log(2*np.pi))/2 # Constant
    return c - total_log / 2



def farmcpu(y:np.ndarray=None,M:np.ndarray=None,X:np.ndarray=None,pos:np.ndarray=None,iteration:int=30,threshold:float=0.01,bpbin:list=[],log:bool=True,threads:int=1):
    '''
    Fast Solve of Mixed Linear Model by Brent.
    
    :param y: Phenotype nx1\n
    :param M: SNP matrix mxn\n
    :param X: Designed matrix for fixed effect nxp\n
    :param kinship: Calculation method of kinship matrix nxn
    '''
    print('It is a beta version...')
    m,n = M.shape
    QTNbound = int(np.sqrt(n/np.log10(n))) # max number of QTNs
    szbin = [1e5,5e5,1e6,5e6,1e7,5e7]
    nbin = range(int(QTNbound/5),QTNbound,int(QTNbound/5))
    X = np.concatenate([np.ones((y.shape[0],1)),X],axis=1) if X is not None else np.ones((y.shape[0],1))
    pseudoQTN = np.array([]).reshape(M.shape[1],0)
    QTNidx = np.array([],dtype=int) # Filter QTNidx
    for _ in trange(iteration):
        X_QTN = np.concatenate([X,pseudoQTN],axis=1)
        FEMresult = np.array(Parallel(n_jobs=threads)(delayed(FEM)(i,X_QTN,y) for i in M))
        FEMresult = 2*norm.sf(np.abs(FEMresult[:,0]/(FEMresult[:,1])))
        FEMresult[QTNidx] = np.min(np.nan_to_num(FEMresult,nan=1))
        if np.sum(FEMresult <= threshold) == 0:
            break
        else:
            l = []
            leadidxs = []
            for sz in szbin:
                bin_id = pos//sz
                for n in nbin:
                    order = np.lexsort((FEMresult,bin_id),) # sort by bin_id, then sort by pvalue; return sorted idx
                    lead = order[np.concatenate(([True], bin_id[order][1:] != bin_id[order][:-1]))]
                    leadidx = np.sort(lead[np.argsort(FEMresult[lead])[:n]])
                    leadidxs.append(leadidx)
                    grm = GRM(M[leadidx])
                    S,D = np.linalg.eigh(grm)
                    ytrans = D.T@y
                    xtrans = D.T@X
                    results = minimize_scalar(lambda x: -REM(10**(x),ytrans,xtrans,S),bounds=(-6,6),method='bounded',options={'xatol': 1e-2, 'maxiter': 30},)
                    l.append(results.fun)
            QTNidx_pre = leadidxs[np.argmin(l)]
            pseudoQTN = M[QTNidx_pre].T
            corr = np.corrcoef(pseudoQTN,rowvar=False)
            keep = np.ones(pseudoQTN.shape[1], dtype=bool)
            for i in range(pseudoQTN.shape[1]): # 去冗余, 保留最显著QTN
                if not keep[i]:
                    continue
                for j in range(i + 1, pseudoQTN.shape[1]):
                    if keep[j]:
                        if abs(corr[i, j]) >= 0.8:
                            if FEMresult[QTNidx[i]]>=FEMresult[QTNidx[j]]:
                                keep[j] = False
                            else:
                                keep[i] = False
                                break
            pseudoQTN = pseudoQTN[:,keep]
            QTNidx_pre = QTNidx_pre[keep]
            if np.array_equal(QTNidx_pre, QTNidx):
                break
            else:
                QTNidx = QTNidx_pre
    X_QTN = np.concatenate([X,pseudoQTN],axis=1)
    FEMresult = np.array(Parallel(n_jobs=threads)(delayed(FEM)(i,X_QTN,y) for i in M))
    FEMresult = 2*norm.sf(np.abs(FEMresult[:,0]/(FEMresult[:,1])))
    FEMresult[QTNidx] = 1
    return FEMresult

if __name__ == "__main__":
    pass