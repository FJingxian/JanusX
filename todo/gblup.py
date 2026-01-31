from typing import Union
import time
from janusx.pyBLUP.lmm import LMM
from janusx.pyBLUP.kfold import kfold
import numpy as np
from numpy.typing import NDArray
from janusx.pyBLUP.bayes import BAYES
from janusx.pyBLUP.gblup import GBLUP

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
        model = GBLUP(ytrain,Mtrain,)
        print('Pearsonr:',np.corrcoef(model.predict(Mtest).ravel(),y[test].ravel())[0,1])
        print(f'Time costed: {time.time()-t:.3f}')
        model = BAYES(ytrain,Mtrain)
        print('Pearsonr:',np.corrcoef(model.predict(Mtest).ravel(),y[test].ravel())[0,1])
        print(f'Time costed: {time.time()-t:.3f}')
