import typing
import numpy as np
from scipy.optimize import minimize_scalar
from .QK2 import GRM
    
class BLUP:
    def __init__(self,y:np.ndarray,M:np.ndarray,cov:np.ndarray=None,Z:np.ndarray=None, kinship:typing.Literal[None,1]=None,log:bool=False):
        """
        Fast solution of the mixed linear model via Brent's method.

        Parameters
        ----------
        y : np.ndarray
            Phenotype vector of shape (n, 1).
        M : np.ndarray
            Marker matrix of shape (m, n) with genotypes coded as 0/1/2.
        cov : np.ndarray, optional
            Fixed-effect design matrix of shape (n, p).
        Z : np.ndarray, optional
            Random-effect design matrix of shape (n, q).
        kinship : {None, 1}
            Kinship specification; None disables kinship.
        """
        self.log = log
        Z = Z if Z is not None else np.eye(y.shape[0]) # Design matrix or I matrix
        assert M.shape[1] == Z.shape[1] # Test Random factor
        self.X = np.concatenate([np.ones((y.shape[0],1)),cov],axis=1) if cov is not None else np.ones((y.shape[0],1)) # Design matrix of 1st vector
        self.y = y.reshape(-1,1)
        self.M = M
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.kinship = kinship # control method to calculate kinship matrix
        usefastlmm = self.M.shape[0] < self.n # use FaST-LMM if num of snp less than num of samples
        if self.kinship is not None:
            self.G = GRM(M,log=self.log) + 1e-6*np.eye(self.n) # Add regular item
            self.Z = Z
        else:
            self.G = np.eye(M.shape[0])
            self.Z = Z@M.T
        # Simplify inverse matrix
        if usefastlmm:
            self.Dh,self.S,_ = np.linalg.svd(Z@M.T,full_matrices=False) if Z is not None else np.linalg.svd(M.T)
            self.S = self.S*self.S
        else:
            self.S,self.Dh = np.linalg.eigh(self.Z@self.G@self.Z.T)
        self.Dh = self.Dh.T
        self.X = self.Dh@self.X
        self.y = self.Dh@self.y
        self.Z = self.Dh@self.Z
        self.result = minimize_scalar(lambda lbd: -self._REML(10**(lbd)),bounds=(-6,6),method='bounded') # minimize REML
        lbd = 10 ** self.result.x
        self._REML(lbd)
        self.u = self.G@self.Z.T@(self.V_inv.ravel()*self.r.T).T
        sigma_g2 = self.rTV_invr / (self.n - self.p)
        sigma_e2 = lbd * sigma_g2
        if self.kinship is None:
            g = self.M.T @ self.u
            var_g = float(np.var(g, ddof=1))
        else:
            mean_s = float(np.mean(self.S))
            var_g = sigma_g2 * mean_s
        self.pve = var_g / (var_g + sigma_e2)
    def _REML(self,lbd: float):
        """Restricted maximum likelihood (REML) for the null model."""
        n,p_cov = self.X.shape
        p = p_cov
        V = self.S+lbd
        V_inv = 1/V
        X_cov_snp = self.X
        XTV_invX = V_inv*X_cov_snp.T @ X_cov_snp
        XTV_invy = V_inv*X_cov_snp.T @ self.y
        beta = np.linalg.solve(XTV_invX, XTV_invy)
        r = self.y - X_cov_snp@beta
        rTV_invr = (V_inv * r.T@r)[0,0]
        log_detV = np.sum(np.log(V))
        sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
        total_log = (n-p)*np.log(rTV_invr) + log_detV + log_detXTV_invX # log items
        c = (n-p)*(np.log(n-p)-1-np.log(2*np.pi))/2 # Constant
        self.V_inv = V_inv
        self.rTV_invr = rTV_invr
        self.r = r
        self.beta = beta
        return c - total_log / 2
    def predict(self,M:np.ndarray,cov:np.ndarray=None):
        X = np.concatenate([np.ones((M.shape[1],1)),cov],axis=1) if cov is not None else np.ones((M.shape[1],1))
        if self.kinship is not None:
            G = GRM(np.concatenate([self.M, M],axis=1),log=self.log)+1e-6*np.eye(self.M.shape[1]+M.shape[1])# Regular item
            return X@self.beta+G[self.n:, :self.n]@np.linalg.solve(self.G,self.u)
        else:
            return X@self.beta+M.T@self.u
