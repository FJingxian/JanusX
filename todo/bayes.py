from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from janusx.pyBLUP.bayes import BAYES

if __name__ == "__main__":
    from janusx.gfreader.gfreader import load_genotype_chunks,inspect_genotype_file
    from janusx.pyBLUP.mlm import BLUP
    from janusx.pyBLUP.kfold import kfold
    import numpy as np
    import pandas as pd
    # /Users/jingxianfu/script/JanusX/example/mouse_hs1940.vcf.gz
    # /Users/jingxianfu/script/JanusX/example/mouse_hs1940.pheno
    genofile = './example/mouse_hs1940.vcf.gz'
    phenofile = './example/mouse_hs1940.pheno'
    ids, nsnp = inspect_genotype_file(genofile)
    pheno = pd.read_csv(phenofile,sep='\t',index_col=0).iloc[:,0].dropna()
    samplec = np.isin(ids,pheno.index)
    chunks = load_genotype_chunks(genofile,maf=0.01)
    Kiter = kfold(np.sum(samplec),seed=None)
    for chunk,site in chunks:
        phenotype = pheno.loc[np.array(ids)[samplec]].values
        genotype = chunk[:,samplec]
    for testidx,trainidx in Kiter:
        model = BAYES(phenotype[trainidx],genotype[:,trainidx])
        y_hat = model.predict(genotype[:,testidx])
        print('BayesA:',np.corrcoef(y_hat.ravel(), phenotype[testidx].ravel())[0,1])
        model = BLUP(phenotype[trainidx],genotype[:,trainidx],log=True,kinship=None)
        y_hat = model.predict(genotype[:,testidx])
        print('rrBLUP:',np.corrcoef(y_hat.ravel(),phenotype[testidx])[0,1])
        print('*'*60)
