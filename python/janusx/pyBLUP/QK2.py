import numpy as np
from tqdm import tqdm
import psutil
import typing
process = psutil.Process()
class QK:
    def __init__(self,M:np.ndarray,chunksize:int=100_000,
                 Mcopy:bool=False,maff:float=0.02,missf:float=0.05,
                 impute:typing.Literal['mean','mode']='mode',check:bool=True,
                 log:bool=True):
        """
        Compute Q and K matrices with low memory and high throughput.

        Parameters
        ----------
        M : np.ndarray
            Marker matrix of shape (m, n) with genotypes coded as 0/1/2 (int8).
        chunksize : int
            Chunk size for streaming computation.
        """
        self.log = log
        M = M.copy() if Mcopy else M
        NAmark = M<0 # Mark miss as bool matrix
        M[NAmark] = 0 # Miss -> 0
        miss = np.sum(NAmark,axis=1) # Missing number of idv
        maf:np.ndarray = (np.sum(M,axis=1)+1)/(2*(M.shape[1]-miss)+2) # stat maf
        # Filter
        maftmark = maf>.5
        maf[maftmark] = 1 - maf[maftmark]
        np.subtract(2, M, where=maftmark[:, None], out=M)
        if impute == 'mode':
            M[NAmark] = 0 # Impute using mode
        elif impute == 'mean':
            M = M.astype(np.float32) # Memory will be increased to 4 fold
            M[NAmark] = np.take(2*maf, np.where(NAmark)[0]) # Impute using mean
        del NAmark
        SNPretain = (miss/M.shape[1]<=missf) & (maf>=maff)
        M = M[SNPretain]
        maf = maf[SNPretain]
        maftmark = maftmark[SNPretain]
        maf = maf.astype('float32')
        self.missrate = miss[SNPretain]/M.shape[1]
        self.SNPretain = SNPretain
        self.maftmark = maftmark
        self.maf = maf.reshape(-1,1)
        self.Mmean = 2*self.maf
        self.Mvar = (2*self.maf*(1-self.maf))
        self.M = M
        self.chunksize = chunksize
    def GRM(self,method:typing.Literal[1,2]=1):
        """
        Compute the genomic relationship matrix (GRM).

        Parameters
        ----------
        method : int
            1 for centering, 2 for standardization.

        Returns
        -------
        np.ndarray
            A positive definite or positive semidefinite matrix.
        """
        m,n = self.M.shape
        Mvar_sum = np.sum(self.Mvar)
        Mvar_r = 1/self.Mvar
        grm = np.zeros((n,n),dtype='float32')
        if self.log:
            pbar = tqdm(total=m, desc="Process of GRM",ascii=True)
        for i in range(0,m,self.chunksize):
            i_end = min(i+self.chunksize,m)
            block_i = self.M[i:i_end]-self.Mmean[i:i_end]
            if method == 1:
                grm+=block_i.T@block_i/Mvar_sum
            elif method == 2:
                grm+=Mvar_r[i:i_end].T/m*block_i.T@block_i
            if self.log:
                pbar.update(i_end-i)
                if i % 10 == 0:
                    memory_usage = process.memory_info().rss / 1024**3
                    pbar.set_postfix(memory=f'{memory_usage:.2f} GB')
        return (grm+grm.T)/2
    def PCA(self):
        """
        Perform randomized SVD for PCA.

        Returns
        -------
        tuple
            (eigenvec[:, :dim], eigenval[:dim])
        """
        m,n = self.M.shape
        Mstd = np.sqrt(self.Mvar)
        MTM = np.zeros((n,n),dtype='float32')
        if self.log:
            pbar = tqdm(total=m, desc="Process of PCA",ascii=True)
        for i in range(0,m,self.chunksize):
            i_end = min(i + self.chunksize, m)
            block_i = (self.M[i:i_end]-self.Mmean[i:i_end])/Mstd[i:i_end]
            MTM += block_i.T @ block_i
            if self.log:
                pbar.update(i_end-i)
                if i % 10 == 0:
                    memory_usage = process.memory_info().rss / 1024**3
                    pbar.set_postfix(memory=f'{memory_usage:.2f} GB')
        MTM = (MTM + MTM.T)/2
        eigval,eigvec = np.linalg.eigh(MTM/m)
        idx = np.argsort(eigval)[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        return eigvec,eigval

def GRM(M:np.ndarray,log=False,chunksize=50_000):
    m,n = M.shape
    Mvar = np.var(M,axis=1,dtype='float32').reshape(-1,1)
    Mmean = np.mean(M,axis=1,dtype='float32').reshape(-1,1)
    Mvar_sum = np.sum(Mvar,dtype='float32')
    grm = np.zeros((n,n),dtype='float32')
    if log:
        pbar = tqdm(total=m, desc="Process of GRM",ascii=True)
    for i in range(0,m,chunksize):
        i_end = min(i+chunksize,m)
        block_i = M[i:i_end]-Mmean[i:i_end]
        for j in range(0, n, 200):  # 内部分块
            j_end = min(j + 200, n)
            block_j = block_i[:, j:j_end]
            grm[j:j_end, :] += np.dot(block_j.T, block_i) / Mvar_sum
        # grm+=block_i.T@block_i/Mvar_sum
        if log:
            pbar.update(i_end-i)
            if i % 10 == 0:
                memory_usage = process.memory_info().rss / 1024**3
                pbar.set_postfix(memory=f'{memory_usage:.2f} GB')
    return (grm+grm.T)/2

def Eigendec(grm:np.ndarray):
    eigval,eigvec = np.linalg.eigh(grm)
    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    return eigvec,eigval

if __name__ == '__main__':
    from janusx.gfreader import breader
    geno = breader(r"C:\Users\82788\Desktop\Pyscript\装饰器\geno.45k")
    M = geno.iloc[:,2:].values.T
    qkmodel = QK(M,log=True)
    grm = qkmodel.GRM()
    U,_ = qkmodel.PCA()
    print(grm[:4,:4])
    print(U[:4,:4])
    pass
    
