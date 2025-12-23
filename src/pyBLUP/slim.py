import numpy as np
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                        message="invalid value encountered in")
import psutil
process = psutil.Process()
from rust2py.assoc import lmm_reml,FEM

class LMM:
    def __init__(self,y:np.ndarray=None,X:np.ndarray=None,kinship:np.ndarray=None):
        '''
        Fast Solve of Mixed Linear Model by Brent.
        
        :param y: Phenotype nx1\n
        :param X: Designed matrix for fixed effect nxp\n
        :param kinship: Calculation method of kinship matrix nxn
        '''
        y = y.reshape(-1,1) # ensure the dim of y
        X = np.concatenate([np.ones((y.shape[0],1)),X],axis=1) if X is not None else np.ones((y.shape[0],1))
        # Simplify inverse matrix
        val,vec = np.linalg.eigh(kinship + 1e-6 * np.eye(y.shape[0]))
        idx = np.argsort(val)[::-1]
        val,vec = val[idx],vec[:, idx]
        self.S,self.Dh = val, vec.T.astype(np.float32)
        del kinship
        self.Xcov = self.Dh@X
        self.y = self.Dh@y
        result = minimize_scalar(lambda lbd: -self._NULLREML(10**(lbd)),bounds=(-5,5),method='bounded',options={'xatol': 1e-3},)
        lbd_null = 10**(result.x)
        vg_null = np.mean(self.S)
        pve = vg_null/(vg_null+lbd_null)
        self.lbd_null = lbd_null
        self.pve = pve
        if pve > 0.999 or pve < 0.001:
            self.bounds = (-5,5)
        else:
            self.bounds = (np.log10(lbd_null)-2,np.log10(lbd_null)+2)
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
            rTV_invr = (V_inv * r.T@r)[0,0]
            log_detV = np.sum(np.log(V))
            sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
            total_log = (n-p)*np.log(rTV_invr) + log_detV + log_detXTV_invX # log items
            c = (n-p)*(np.log(n-p)-1-np.log(2*np.pi))/2 # Constant
            return c - total_log / 2
        except Exception as e:
            print(f"REML error: {e}, lbd={lbd}")
            return -1e8
    def gwas(self,snp:np.ndarray=None,threads=1):
        '''
        Speed version of mlm
        
        :param snp: Marker matrix, np.ndarray, samples per rows and snp per columns
        :param chunksize: calculation number per times, int
        
        :return: beta coefficients, standard errors and p-values for each SNP, np.ndarray
        '''
        beta_se_p, lambdas = lmm_reml(
            self.S,             # eigenvalues of K
            self.Xcov,          # DH @ X
            self.y,             # DH @ y
            self.Dh,
            snp,      #g rotated: snp_chunk @ DH.T (float32)
            self.bounds,     # log10(lbd lower bound)
            max_iter=30,
            tol=1e-2,
            threads=threads
        )
        self.lbd = lambdas
        return beta_se_p

class LM:
    def __init__(self,y:np.ndarray=None,X:np.ndarray=None):
        self.y = y.reshape(-1,1) # ensure the dim of y
        self.X = np.concatenate([np.ones((y.shape[0],1)),X],axis=1) if X is not None else np.ones((y.shape[0],1))
    def gwas(self,snp:np.ndarray,threads:int=1):
        beta_se_p = FEM(self.y,self.X,snp,snp.shape[0],threads)[:,[0,1,-1]]
        return beta_se_p

if __name__ == '__main__':
    pass