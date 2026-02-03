import time
from typing import Any
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from janusx.gfreader import load_genotype_chunks
from janusx.pyBLUP.assoc import SUPER
from janusx.garfield.logreg import logreg
from joblib import Parallel,delayed

def timer(func):
    def wrapper(*args,**kwargs):
        t = time.time()
        result = func(*args,**kwargs)
        print(f'Time cost: {time.time()-t:.2f} secs')
        return result
    return wrapper

# @timer
def getLogicgate(y: np.ndarray, M: np.ndarray,
           nsnp: int = 5, n_estimators: int = 200,
           sites:Any=None):
    '''
    chrom
    pos
    ref_allele
    alt_allele
    '''
    if sites is not None:
        sites = np.array([f'{i.chrom}_{i.pos}' for i in sites])
    else:
        sites = np.arange(M.shape[0])
    keep = SUPER(np.corrcoef(M),np.ones(M.shape[0]),thr=0.8)
    M = M[keep]
    sites = sites[keep]
    y = y.ravel()
    Imp = None
    rf = RandomForestRegressor(n_estimators=n_estimators,max_depth=nsnp,bootstrap=True,n_jobs=1)
    rf.fit(M.T,y)
    PI = permutation_importance(rf,M.T,y,scoring='r2',n_jobs=1)
    Imp = PI.importances_mean
    topk = nsnp
    idx = np.argsort(Imp)[::-1][:topk]
    Mchoice = (M[idx]/2).astype(int).T
    return logreg(Mchoice, y, response="continuous",tags=sites[idx])

def process(ChromPos:tuple,genofile,sampleid,y):
    chrom,start,end = ChromPos
    chunks = load_genotype_chunks(genofile,chunk_size=1e6,maf=0.02,missing_rate=0.05,impute=True,bim_range=(str(chrom),start,end),sample_ids=sampleid)
    for chunk,sites in chunks:
        result = getLogicgate(y,chunk,sites=sites)
    return

def main():
    phenofile = "../Garfield/example/test.trait.txt"
    genofile = "../Garfield/example/test.genotype"
    bedfile = "../Garfield/example/test.geneAnno.bed"
    windowsize = 100_000
    threads = 4
    bed = pd.read_csv(bedfile,sep='\t',header=None,index_col=0)
    if bed.size == 0:
        # space split?
        bed = pd.read_csv(bedfile,sep=' ',header=None,index_col=0)
        if bed.size == 0:
            raise ValueError(f"Error in Phenotype file:\n{bed}")
    bed = bed.reset_index()
    bed[1]-=windowsize
    bed[2]+=windowsize
    bedlist = [(bed.loc[i,0],bed.loc[i,1],bed.loc[i,2]) for i in bed.index]
    pheno = pd.read_csv(phenofile,sep='\t',header=None,index_col=0)
    if pheno.size == 0:
        # space split?
        pheno = pd.read_csv(phenofile,sep=' ',header=None,index_col=0)
        if pheno.size == 0:
            raise ValueError(f"Error in Phenotype file:\n{pheno}")
    if all(pheno.index==pheno[1]):
        # Phenotype format of plink
        pheno = pheno.iloc[:,1:]
    else:
        # Phenotype format of normal excel
        pheno.columns = pheno.iloc[0,:]
        pheno = pheno.iloc[1:,:]
    pheno = pheno.iloc[:,0].dropna().to_frame()
    Parallel(n_jobs=threads,backend='loky')(delayed(process)(ChromPos,genofile,pheno.index,pheno.values) for ChromPos in bedlist)

if __name__ == '__main__':
    main()