import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from joblib import Parallel, delayed # for parallel processing
from tqdm import tqdm
import gc # garbage collection
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                        message="invalid value encountered in subtract")
import psutil
process = psutil.Process()

class GWAS:
    def __init__(self,y:np.ndarray=None,X:np.ndarray=None,kinship:np.ndarray=None,log:bool=True):
        '''
        Fast Solve of Mixed Linear Model by Brent.
        
        :param y: Phenotype nx1\n
        :param X: Designed matrix for fixed effect nxp\n
        :param kinship: Calculation method of kinship matrix nxn
        '''
        self.log = log
        y = y.reshape(-1,1) # ensure the dim of y
        X = np.concatenate([np.ones((y.shape[0],1)),X],axis=1) if X is not None else np.ones((y.shape[0],1))
        # Simplify inverse matrix
        val,vec = np.linalg.eigh(kinship + 1e-6 * np.eye(y.shape[0]))
        idx = np.argsort(val)[::-1]
        val,vec = val[idx],vec[:, idx]
        self.S,self.Dh = val, vec.T
        del kinship
        self.Xcov = self.Dh@X
        self.y = self.Dh@y
        result = minimize_scalar(lambda lbd: -self._NULLREML(10**(lbd)),bounds=(-8,8),method='bounded',options={'xatol': 1e-3},)
        lbd_null = 10**(result.x[0,0])
        vg_null = np.mean(self.S)
        pve = vg_null/(vg_null+lbd_null)
        self.lbd_null = lbd_null
        self.pve = pve
        self.bounds = (np.log10(lbd_null)-3,np.log10(lbd_null)+3)
        pass
    def _NULLREML(self,lbd: float):
        '''Restricted Maximum Likelihood Estimation (REML) of NULL'''
        try:
            n,p_cov = self.Xcov.shape
            p = p_cov
            V = self.S+lbd
            V_inv = 1/V
            X_cov_snp = self.Xcov
            XTV_invX = V_inv*X_cov_snp.T @ X_cov_snp
            XTV_invy = V_inv*X_cov_snp.T @ self.y
            beta = np.linalg.solve(XTV_invX, XTV_invy)
            r = self.y - X_cov_snp@beta
            rTV_invr = V_inv * r.T@r
            log_detV = np.sum(np.log(V))
            sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
            total_log = (n-p)*np.log(rTV_invr) + log_detV + log_detXTV_invX # log items
            c = (n-p)*(np.log(n-p)-1-np.log(2*np.pi))/2 # Constant
            return c - total_log / 2
        except Exception as e:
            print(f"REML error: {e}, lbd={lbd}")
            return -1e8
    def _REML(self,lbd: float, snp_vec:np.array):
        '''Restricted Maximum Likelihood Estimation (REML)'''
        try:
            n,p_cov = self.Xcov.shape
            p = p_cov + 1
            V = self.S+lbd
            V_inv = 1/V
            X_cov_snp = np.column_stack([self.Xcov, snp_vec])
            XTV_invX = V_inv*X_cov_snp.T @ X_cov_snp
            XTV_invy = V_inv*X_cov_snp.T @ self.y
            try:
                beta = np.linalg.solve(XTV_invX, XTV_invy)
            except:
                beta = np.linalg.solve(XTV_invX+1e-6*np.eye(XTV_invX.shape[0]), XTV_invy)
            r = self.y - X_cov_snp@beta
            rTV_invr = V_inv * r.T@r
            log_detV = np.sum(np.log(V))
            sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
            total_log = (n-p)*np.log(rTV_invr) + log_detV + log_detXTV_invX # log items
            c = (n-p)*(np.log(n-p)-1-np.log(2*np.pi))/2 # Constant
            return c - total_log / 2
        except:
            return -1e8
    def _fit(self,snp:np.ndarray=None):
        X = np.column_stack([self.Xcov, snp])
        n,p = X.shape
        V_inv = 1/(self.S+self.lbd_null)
        XTV_invX = V_inv*X.T@X + 1e-6*np.eye(X.shape[1])
        XTV_invy = V_inv*X.T@self.y
        beta = np.linalg.solve(XTV_invX,XTV_invy)
        r = self.y - X@beta
        rTV_invr = V_inv * r.T@r
        sigma2 = rTV_invr/(n-p)
        se = np.sqrt(np.linalg.inv(XTV_invX/sigma2)[-1,-1])
        return beta[-1,0],se
    def _HACfit(self,snp:np.ndarray=None):
        result = minimize_scalar(lambda lbd: -self._REML(10**(lbd),snp),bounds=self.bounds,method='bounded',options={'xatol': 1e-2, 'maxiter': 50},) # 寻找lbd 最大化似然函数
        if result.success:
            lbd = self.lbd_null if not result.success else 10**(result.x[0,0])
            X = np.column_stack([self.Xcov, snp])
            n,p = X.shape
            V_inv = 1/(self.S+lbd)
            XTV_invX = V_inv*X.T@X + 1e-6*np.eye(X.shape[1])
            XTV_invy = V_inv*X.T@self.y
            beta = np.linalg.solve(XTV_invX,XTV_invy)
            r = self.y - X@beta
            rTV_invr = V_inv * r.T@r
            sigma2 = rTV_invr/(n-p)
            se = np.sqrt(np.linalg.inv(XTV_invX/sigma2)[-1,-1])
            return beta[-1,0],se,lbd
        else:
            return np.nan,np.nan,np.nan
    def gwas(self,snp:np.ndarray=None,chunksize=100_000,fast:bool=False,threads=1):
        '''
        Speed version of mlm
        
        :param snp: Marker matrix, np.ndarray, samples per rows and snp per columns
        :param chunksize: calculation number per times, int
        
        :return: beta coefficients, standard errors and p-values for each SNP, np.ndarray
        '''
        m,n = snp.shape
        lbds = []
        beta_se_p = []
        pbar = tqdm(total=m, desc="Process of GWAS",ascii=True)
        def ACmode(i):
            '''
            solving beta and its se in multiprocess
            '''
            return self._HACfit(snp_chunk[:, i])
        def FASTmode(i):
            '''
            solving beta and its se in multiprocess
            '''
            return self._fit(snp_chunk[:, i])
        process_col = FASTmode if fast else ACmode
        for i in range(0,m,chunksize):
            i_end = min(i+chunksize,m)
            snp_chunk = self.Dh@snp[i:i_end].T
            if snp_chunk.shape[1]>0:
                results = np.array(Parallel(n_jobs=threads)(delayed(process_col)(i) for i in range(snp_chunk.shape[1])))
                if results.shape[1] == 3:
                    lbds.extend(results[:,-1])
                    results = results[:,:-1]
                beta_se_p.append(np.concatenate([results,2*norm.sf(np.abs(results[:,0]/results[:,1])).reshape(-1,1)],axis=1))
            if self.log:
                pbar.update(i_end-i)
                if i % 10 == 0:
                    memory = psutil.virtual_memory()
                    memory_usage = process.memory_info().rss/1024**3
                    pbar.set_postfix(memory=f'{memory_usage:.2f} GB',total=f'{(memory.used/1024**3):.2f}/{(memory.total/1024**3):.2f} GB')
            gc.collect()
        self.lbd = lbds if not fast else None
        return np.concatenate(beta_se_p)

class LM:
    def __init__(self,y:np.ndarray=None,X:np.ndarray=None,log:bool=True):
        '''
        Fast Solve of Mixed Linear Model by Brent.
        
        :param y: Phenotype nx1\n
        :param X: Designed matrix for fixed effect nxp\n
        :param kinship: Calculation method of kinship matrix nxn
        '''
        self.log = log
        self.y = y.reshape(-1,1) # ensure the dim of y
        self.X = np.concatenate([np.ones((y.shape[0],1)),X],axis=1) if X is not None else np.ones((y.shape[0],1))
        pass
    def gwas(self,snp:np.ndarray=None,chunksize=100_000,threads=1):
        '''
        Speed version of mlm
        
        :param snp: Marker matrix, np.ndarray, SNP per rows and IDV per columns
        :param chunksize: calculation number per times, int
        
        :return: beta coefficients, standard errors and p-values for each SNP, np.ndarray
        '''
        m,n = snp.shape
        beta_se_p = []
        pbar = tqdm(total=m, desc="Process of GWAS",ascii=True)
        def process_col(i):
            '''
            solving beta and its se in multiprocess
            '''
            X = np.column_stack([self.X, snp_chunk[:, i]])
            XTX = X.T@X
            try:
                XTX_inv = np.linalg.inv(XTX)
            except:
                XTX_inv = np.linalg.inv(XTX+np.eye(XTX.shape[0]))
            XTy = X.T@self.y
            beta = XTX_inv@XTy
            r = (self.y-X@beta)
            se = np.sqrt((r.T@r)/(n-XTX.shape[0])*XTX_inv[-1,-1])
            return beta[-1,0],se[0,0]
        for i in range(0,m,chunksize):
            i_end = min(i+chunksize,m)
            snp_chunk = snp[i:i_end].T
            if snp_chunk.shape[1]>0:
                results = np.array(Parallel(n_jobs=threads)(delayed(process_col)(i) for i in range(snp_chunk.shape[1])))
                beta_se_p.append(np.concatenate([results,2*norm.sf(np.abs(results[:,0]/results[:,1])).reshape(-1,1)],axis=1))
            if self.log:
                pbar.update(i_end-i)
                if i % 10 == 0:
                    memory = psutil.virtual_memory()
                    memory_usage = process.memory_info().rss/1024**3
                    pbar.set_postfix(memory=f'{memory_usage:.2f} GB',total=f'{(memory.used/1024**3):.2f}/{(memory.total/1024**3):.2f} GB')
            gc.collect()
        return np.concatenate(beta_se_p)
    
if __name__ == '__main__':
    pass