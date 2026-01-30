import time
from janusx.pyBLUP.lmm import LMM
from janusx.pyBLUP.kfold import kfold
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

if __name__ == "__main__":
    from janusx.gfreader import vcfreader
    import pandas as pd
    geno = vcfreader('example/mouse_hs1940.vcf.gz',maf=0.02,miss=0.05).iloc[:,2:]
    pheno = pd.read_csv('example/mouse_hs1940.pheno',sep='\t',index_col=0).iloc[:,0].dropna()
    csample = list(set(geno.columns) & set(pheno.index))
    y = pheno.loc[csample].values.reshape(-1,1)
    M:np.ndarray = geno[csample].values
    m,n = M.shape
    for test,train in kfold(len(csample)):
        t = time.time()
        ytrain = y[train]
        Mtrain = M[:,train]
        Mtest = M[:,test]
        Ztrain:np.ndarray = (Mtrain-Mtrain.mean(axis=1,keepdims=True))/Mtrain.std(axis=1,keepdims=True)/np.sqrt(m)
        Ztest:np.ndarray = (Mtest-Mtrain.mean(axis=1,keepdims=True))/Mtrain.std(axis=1,keepdims=True)/np.sqrt(m)
        Ktt = Ztrain.T@Ztrain
        model = LMM(ytrain,grm=Ktt)
        np.fill_diagonal(Ktt,np.diag(Ktt)+np.power(10,model.loglbd))
        Knt = Ztest.T@Ztrain
        r = ytrain - model.beta   # (t,1)
        c, low = la.cho_factor(Ktt, lower=True, check_finite=False)
        alpha = la.cho_solve((c, low), r, check_finite=False)
        g_hat = Knt @ alpha
        y_hat = g_hat + model.beta
        print(np.corrcoef(y_hat.ravel(),y[test].ravel())[0,1])
        print(f'{time.time()-t:.3f}')