import time
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import xgboost as xgb
import typing
from janusx.gfreader import load_genotype_chunks
from janusx.pyBLUP.mlm import BLUP
from janusx.pyBLUP.assoc import SUPER, FEM
from janusx.garfield.logreg import logregfit
def timer(func):
    def wrapper(*args,**kwargs):
        t = time.time()
        result = func(*args,**kwargs)
        print(f'Time cost: {time.time()-t:.2f} secs')
        return result
    return wrapper

@timer
def getImp(y: np.ndarray, M: np.ndarray,
           nsnp: int = 5, n_estimators: int = 100, threads: int = -1,
           engine: typing.Literal['rf','blup','xgb']='rf'):
    y = y.ravel()
    M = M - M.mean(axis=1,keepdims=True)
    Imp = None
    if engine == 'rf':
        rf = RandomForestRegressor(n_estimators=n_estimators,min_samples_leaf=nsnp,bootstrap=True,n_jobs=threads)
        rf.fit(M.T,y)
        PI = permutation_importance(rf,M.T,y,scoring='r2',n_jobs=threads)
        Imp = PI.importances_mean
    elif engine == 'blup':
        model = BLUP(y,M,kinship=None)
        Imp = np.abs(model.u.ravel())
    elif engine == 'xgb':
        model = xgb.XGBRegressor(n_estimators=n_estimators,max_depth=3,
                                learning_rate=0.05,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                objective="reg:squarederror",
                                n_jobs=threads,
                                tree_method="hist")
        model = model.fit(M.T,y,)
        PI = permutation_importance(
            model, M.T, y,
            scoring='r2',
            n_jobs=threads,
            random_state=222
        )
        Imp = PI.importances_mean
    return Imp


if __name__ == '__main__':
    pheno = pd.read_csv('/Users/jingxianfu/script/JanusX/example/mouse_hs1940.pheno',sep='\t',index_col=0).iloc[:,0].dropna().to_frame()
    chunks = load_genotype_chunks("./test/mouse_hs1940",chunk_size=1e6,maf=0.02,missing_rate=0.05,impute=True,bim_range=('1',0,50_000_000),sample_ids=pheno.index)
    for chunk,sites in chunks:
        keep = SUPER(np.corrcoef(chunk),np.ones(chunk.shape[0]),thr=0.8)
        chunk = chunk[keep]/2 # 去除彼此之间最相关的
        print(chunk.shape)
        Imp = getImp(pheno.values, chunk)
        topk = 5
        idx = np.argsort(Imp)[::-1][:topk]
        top_imps = Imp[idx]
        result = logregfit(chunk[idx].astype(int).T,pheno.values.ravel(),response="continuous",score='mse')
        print(result)