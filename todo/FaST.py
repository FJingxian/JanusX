from tqdm import tqdm
import numpy as np
from janusx.pyBLUP.lmm import LMM

# ------------------------- quick test -------------------------
if __name__ == "__main__":
    from janusx.gfreader import load_genotype_chunks,inspect_genotype_file
    import pandas as pd
    pheno = pd.read_csv('/Volumes/HP X306W/tmp/SCR.txt',sep='\t',index_col=0).iloc[:,0].dropna()
    csample = pheno.index.unique().tolist()
    y:np.ndarray = pheno.loc[csample].to_numpy()
    y = y.reshape(-1,1)
    X = None
    chunksize = 10000
    genoIter = load_genotype_chunks('/Volumes/HP X306W/tmp/cubic_All.maf0.02',sample_ids=csample,chunk_size=chunksize,maf=0.02)
    model = LMM(y,X,Miter=genoIter,threads=8,fixedlbd=True)
    print('* Full')
    print(model.pve)    
    genoIter = load_genotype_chunks('/Volumes/HP X306W/tmp/cubic_All.maf0.02',sample_ids=csample,chunk_size=chunksize,maf=0.02)
    ids,nsnp = inspect_genotype_file('/Volumes/HP X306W/tmp/cubic_All.maf0.02')
    for mchunk,site in tqdm(genoIter,total=nsnp//chunksize):
        mchunk = mchunk-mchunk.mean(axis=1,keepdims=True)
        result = model.gwas(mchunk,plrt=True)
    print(result)