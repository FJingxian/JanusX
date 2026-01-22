# # Function to perform random forest analysis and get feature importance
# random_forest_getImp <- function(data, minnode = 5, seed = 222) {
# 	suppressMessages(library("ranger"))
# 	rf_model <- ranger(factor(data$phe) ~ ., data = data, importance = "permutation", write.forest = TRUE, min.node.size = minnode, seed = seed)
# 	feature_importance <- data.frame(variables = names(importance(rf_model)), IMP = importance(rf_model))
# 	feature_importance <- feature_importance[order(feature_importance$IMP, decreasing = TRUE), ]
# 	positive_imp_features <- feature_importance[which(feature_importance$IMP > 0), ]
# 	return(list(rf_ImpPositive = positive_imp_features, rf_rawImp = feature_importance))
# }

# # Function to select get logic expressions
# logicFS_getImp <- function(data, pheno_ID, B=100, nleaves = 10, ntrees = 1, seed = 222, start = 2, end = -2, iter = 1000){
# 	suppressMessages(library("logicFS"))
# 	binary_snps <- as.matrix(data[, -1])
# 	suppressWarnings(logic_bag <- logic.bagging(binary_snps, factor(data$phe), B = B, nleaves = nleaves, ntrees = ntrees, rand = seed, anneal.control = logreg.anneal.control(start = start, end = end, iter = iter), addMatImp = TRUE))
# 	predicted_genotype <- cbind(data.frame(id = pheno_ID, predict = predict(logic_bag, binary_snps))) #### predicGeno by LogicFS
# 	variable_importance <- data.frame(importance = logic_bag$vim$vim, expression = logic_bag$vim$primes)
# 	variable_importance <- variable_importance[order(-variable_importance$importance), ][variable_importance$importance > 0, ]
# 	# variable_importance <- variable_importance[variable_importance$importance > 0, ]
# 	variable_importance <- variable_importance[!grepl("TEMP_ABCDEFG", variable_importance$expression), ]
# 	return(list(variable_importance = variable_importance, predicted_genotype = predicted_genotype))
# }
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
    pheno = pd.read_csv('./example/mouse_hs1940.pheno',sep='\t',index_col=0).iloc[:,0].dropna().to_frame()
    chunks = load_genotype_chunks("./test/mouse_hs1940",chunk_size = 1e6,maf=0.02,missing_rate=0.05,impute=True,bim_range=('1',0,50_000_000),sample_ids=pheno.index)
    for chunk,sites in chunks:
        keep = SUPER(np.corrcoef(chunk),np.ones(chunk.shape[0]),thr=0.8)
        chunk = chunk[keep] # 去除彼此之间最相关的
        pval = FEM(pheno.values, np.ones(pheno.shape), chunk, threads=0)[:,-1]
        chunk = chunk[pval>=1e-10,]
        print(chunk.shape)
        Imp = getImp(pheno.values, chunk)
        # print(Imp)
        Imp = getImp(pheno.values, chunk, engine='blup')
        # print(Imp)
        Imp = getImp(pheno.values, chunk,engine='xgb')
        # print(Imp)