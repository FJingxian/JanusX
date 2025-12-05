import typing
import numpy as np
from scipy.optimize import minimize_scalar
from .QK2 import QK
    
class BLUP:
    def __init__(self,y:np.ndarray,M:np.ndarray,cov:np.ndarray=None,Z:np.ndarray=None, kinship:typing.Literal[None,1,2]=None):
        '''Fast Solve of Mixed Linear Model by Brent.
        
        :param y: Phenotype nx1\n
        :param X: Designed matrix for fixed effect nxp\n
        :param Z: Designed matrix for random effect nxq\n
        :param M: Marker matrix mxq(0,1,2 of SNP)\n
        :param kinship: Calculation method of kinship matrix (None,1,2)
        '''
        Z = Z if Z is not None else np.eye(y.shape[0]) # 设计矩阵 或 单位矩阵(一般没有重复则采用单位矩阵)
        assert M.shape[1] == Z.shape[1] # 随机效应和效应值相同
        self.X = np.concatenate([np.ones((y.shape[0],1)),cov],axis=1) if cov is not None else np.ones((y.shape[0],1)) # 设计矩阵 或 n1 向量
        self.y = y.reshape(-1,1)
        self.M = M
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.kinship = kinship # 确保训练和预测的kinship方法一致
        if self.kinship is not None:
            qkmodel = QK(M,low_memory=False)
            self.G = qkmodel.GRM(method=self.kinship)
            self.G+=1e-6*np.eye(self.G.shape[0]) # 添加正则项 确保矩阵正定
            self.Z = Z
        else:
            self.G = np.eye(M.shape[0])
            self.Z = Z@M.T
        # 简化G矩阵求逆
        self.D, self.S, self.Dh = np.linalg.svd(self.Z@self.G@self.Z.T)
        self.X = self.Dh@self.X
        self.y = self.Dh@self.y
        self.Z = self.Dh@self.Z
        self.result = minimize_scalar(lambda lbd: -self._REML(10**(lbd)),bounds=(-6,6),method='bounded') # 寻找lbd 最大化似然函数
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
        self.V_inv,self.r = V_inv,r # 估计随机效应时使用
        return c - total_log / 2
    def predict(self,M:np.ndarray,cov:np.ndarray=None):
        X = np.concatenate([np.ones((M.shape[1],1)),cov],axis=1) if cov is not None else np.ones((M.shape[1],1))
        if self.kinship is not None:
            qkmodel = QK(np.concatenate([self.M, M]))
            G = qkmodel.GRM(method=self.kinship)
            G+=1e-6*np.eye(G.shape[1]) # 添加正则项 确保矩阵正定
            return X@self.beta+G[self.n:, :self.n]@np.linalg.solve(self.G,self.u)
        else:
            return X@self.beta+M.T@self.u