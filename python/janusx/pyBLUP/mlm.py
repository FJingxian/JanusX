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
        if self.kinship is not None:
            self.G = GRM(M,log=self.log)
            self.G+=1e-6*np.eye(self.G.shape[0]) # Add regular item
            self.Z = Z
        else:
            self.G = np.eye(M.shape[0])
            self.Z = Z@M.T
        # Simplify inverse matrix
        val,vec = np.linalg.eigh(self.Z@self.G@self.Z.T)
        idx = np.argsort(val)[::-1]
        val,vec = val[idx],vec[:, idx]
        self.S,self.Dh = val, vec.T
        # self.D,self.S,self.Dh = np.linalg.svd(self.Z@self.G@self.Z.T)
        self.X = self.Dh@self.X
        self.y = self.Dh@self.y
        self.Z = self.Dh@self.Z
        self.result = minimize_scalar(lambda lbd: -self._REML(10**(lbd)),bounds=(-6,6),method='bounded') # minimize REML
        lbd = 10**(self.result.x[0,0])
        Vg = np.mean(self.S)
        Ve = lbd
        self.pve = Vg/(Vg+Ve)
        self.u = self.G@self.Z.T@(self.V_inv.flatten()*self.r.T).T
    def _REML(self,lbd: float):
        n,p = self.n,self.p
        V = self.S+lbd
        V_inv = 1/V
        XTV_invX = V_inv*self.X.T @ self.X
        XTV_invy = V_inv*self.X.T @ self.y
        self.beta = np.linalg.solve(XTV_invX,XTV_invy)
        r = self.y - self.X@self.beta
        rTV_invr = V_inv * r.T@r
        c = (n-p)*(np.log(n-p)-1-np.log(2*np.pi))/2
        log_detV = np.sum(np.log(V))
        signX, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
        total_log = (n-p)*np.log(rTV_invr) + log_detV + log_detXTV_invX
        self.V_inv,self.r = V_inv,r # Estimate random effect
        return c - total_log / 2
    def predict(self,M:np.ndarray,cov:np.ndarray=None):
        X = np.concatenate([np.ones((M.shape[1],1)),cov],axis=1) if cov is not None else np.ones((M.shape[1],1))
        if self.kinship is not None:
            G = GRM(np.concatenate([self.M, M],axis=1),log=self.log)
            G+=1e-6*np.eye(G.shape[1]) # Regular item
            return X@self.beta+G[self.n:, :self.n]@np.linalg.solve(self.G,self.u)
        else:
            return X@self.beta+M.T@self.u
