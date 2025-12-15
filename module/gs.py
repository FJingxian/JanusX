# -*- coding: utf-8 -*-
import logging
import typing
import os
for key in ['MPLBACKEND']:
    if key in os.environ:
        del os.environ[key]
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
from bioplotkit.sci_set import color_set
from bioplotkit import gsplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import argparse
import time
import socket
from _common.log import setup_logging
from gfreader import breader,vcfreader,npyreader
from pyBLUP import QK,BLUP,kfold

def GSapi(Y:np.ndarray,Xtrain:np.ndarray,Xtest:np.ndarray,method:typing.Literal['GBLUP','rrBLUP','RF','SVM'],PCAdec:bool=False):
    if PCAdec:
        Xtt = np.concatenate([Xtrain,Xtest],axis=1)
        Xtt:np.ndarray = (Xtt-np.mean(Xtt,axis=1,keepdims=True))/(np.std(Xtt,axis=1,keepdims=True)+1e-8)
        val,vec = np.linalg.eigh(Xtt.T@Xtt/Xtt.shape[0])
        idx = np.argsort(val)[::-1]
        val,vec = val[idx],vec[:, idx]
        dim = np.sum(np.cumsum(val)/np.sum(val)<=0.9)
        vec = val[:dim]*vec[:,:dim]
        Xtrain,Xtest = vec[:Xtrain.shape[1],:].T,vec[Xtrain.shape[1]:,:].T
    if method == 'GBLUP':
        model = BLUP(Y.reshape(-1,1),Xtrain,kinship=1)
        return model.predict(Xtrain),model.predict(Xtest)
    elif method == 'rrBLUP':
        model = BLUP(Y.reshape(-1,1),Xtrain,kinship=None)
        return model.predict(Xtrain),model.predict(Xtest)
    elif method == 'RF':
        param_grid = {
            'n_estimators': [10,25,50,75],
            'max_depth': [None, 1, 3, 5, 7, 10], # adjust overfit
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8],
        }
        grid = RandomizedSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=-1,n_iter=30)
        grid.fit(Xtrain.T, Y.flatten())
        return grid.predict(Xtrain.T).reshape(-1,1),grid.predict(Xtest.T).reshape(-1,1)
    elif method == 'SVM':
        param_grid = {
            'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
            'gamma': ['scale', 0.01, 0.1]
        }
        grid = GridSearchCV(SVR(), param_grid, cv=5, n_jobs=-1)
        grid.fit(Xtrain.T, Y.flatten())
        return grid.predict(Xtrain.T).reshape(-1,1),grid.predict(Xtest.T).reshape(-1,1)
    elif method == 'AdaBoost':
        param_grid = {
            'n_estimators': range(50,500,50),
            'learning_rate': [0.01, 0.1, 0.5, 1]
        }
        grid = GridSearchCV(AdaBoostRegressor(), param_grid, cv=5, n_jobs=-1)
        grid.fit(Xtrain.T, Y.flatten())
        return grid.predict(Xtrain.T).reshape(-1,1),grid.predict(Xtest.T).reshape(-1,1)

def main(log:bool=True):
    t_start = time.time()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    ## Genotype file
    geno_group.add_argument('-vcf','--vcf', type=str, 
                           help='Input genotype file in VCF format (.vcf or .vcf.gz)')
    geno_group.add_argument('-bfile','--bfile', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .bed, .bim, .fam)')
    geno_group.add_argument('-npy','--npy', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .npz, .snp, .idv)')
    ## Phenotype file
    required_group.add_argument('-p','--pheno', type=str, required=True,
                               help='Phenotype file (tab-delimited with sample IDs in first column)')
    # Model arguments
    model_group = parser.add_argument_group('Model Arguments')
    model_group.add_argument('-GBLUP','--GBLUP', action='store_true', default=False,
                               help='Method of GBLUP to train and predict '
                                   '(default: %(default)s)')
    model_group.add_argument('-rrBLUP','--rrBLUP', action='store_true', default=False,
                               help='Method of rrBLUP to train and predict '
                                   '(default: %(default)s)')
    model_group.add_argument('-SVM','--SVM', action='store_true', default=False,
                               help='Method of support vector machine to train and predict '
                                   '(default: %(default)s)')
    model_group.add_argument('-RF','--RF', action='store_true', default=False,
                               help='Method of random forest to train and predict '
                                   '(default: %(default)s)')
    model_group.add_argument('-ADB','--AdaBoost', action='store_true', default=False,
                               help='Method of AdaBoost to train and predict '
                                   '(default: %(default)s)')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('-pcd','--pcd', action='store_true', default=False,
                               help='Decomposition of data by PCA '
                                   '(default: %(default)s)')
    ## Point out phenotype or snp
    optional_group.add_argument('-n','--ncol', type=int, default=None,
                               help='Only analysis n columns in phenotype ranged from 0-n '
                                   '(default: %(default)s)')
    ## Other optional arguments
    optional_group.add_argument('-plot','--plot', action='store_true', default=False,
                               help='Visualization of 5-fold cross-validation and different model tree '
                                   '(default: %(default)s)')
    optional_group.add_argument('-o', '--out', type=str, default=None,
                               help='Output directory for results'
                                   '(default: %(default)s)')
    optional_group.add_argument('-prefix','--prefix',type=str,default=None,
                               help='prefix of output file'
                                   '(default: %(default)s)')
    args = parser.parse_args()
    # Determine genotype file
    if args.vcf:
        gfile = args.vcf
        args.prefix = os.path.basename(gfile).replace('.gz','').replace('.vcf','') if args.prefix is None else args.prefix
    elif args.bfile:
        gfile = args.bfile
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    elif args.npy:
        gfile = args.npy
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    gfile = gfile.replace('\\','/') # adjust \ in Windows
    args.out = os.path.dirname(gfile) if args.out is None else args.out
    # create log file
    os.makedirs(args.out,0o755,exist_ok=True)
    logger = setup_logging(f'''{args.out}/{args.prefix}.gs.log'''.replace('\\','/').replace('//','/'))
    logger.info('Genomic Selection Module')
    logger.info(f'Host: {socket.gethostname()}\n')
    num = 0
    if log:
        logger.info("*"*60)
        logger.info("GENOMIC SELECTION CONFIGURATION")
        logger.info("*"*60)
        logger.info(f"Genotype file:   {gfile}")
        logger.info(f"Phenotype file:  {args.pheno}")
        logger.info(f"Analysis Pcol:   {args.ncol}") if args.ncol is not None else logger.info(f"Analysis Pcol:   All")
        if args.GBLUP:
            num += 1
            logger.info(f"Used model{num}:     GBLUP")
        if args.rrBLUP:
            num += 1
            logger.info(f"Used model{num}:     rrBLUP")
        if args.SVM:
            num += 1
            logger.info(f"Used model{num}:     Support vecter machine")
        if args.RF:
            num += 1
            logger.info(f"Used model{num}:     Random Forest")
        if args.AdaBoost:
            num += 1
            logger.info(f"Used model{num}:     AdaBoost")
        logger.info(f"Decomposition:   {args.pcd}")
        if args.plot:
            logger.info(f"Plot mode:       {args.plot}")
        logger.info(f"Output prefix:   {args.out}/{args.prefix}")
        logger.info("*"*60 + "\n")
        
    t_loading = time.time()
    logger.info(f'Loading phenotype from {args.pheno}...')
    pheno = pd.read_csv(rf'{args.pheno}',sep='\t') # Col 1 - idv ID; row 1 - pheno tag
    pheno = pheno.groupby(pheno.columns[0]).mean() # Mean of duplicated samples
    pheno.index = pheno.index.astype(str)
    assert pheno.shape[1]>0, f'No phenotype data found, please check the phenotype file format!\n{pheno.head()}'
    if args.ncol is not None: 
        assert args.ncol <= pheno.shape[1], "IndexError: Phenotype column index out of range."
        pheno = pheno.iloc[:,[args.ncol]]
    methods = []
    if args.GBLUP:
        methods.append('GBLUP')
    if args.rrBLUP:
        methods.append('rrBLUP')
    if args.SVM:
        methods.append('SVM')
    if args.RF:
        methods.append('RF')
    if args.AdaBoost:
        methods.append('AdaBoost')
    assert len(methods) > 0, 'No method exists'
    if args.vcf:
        logger.info(f'Loading genotype from {gfile}...')
        geno = vcfreader(rf'{gfile}') # VCF format
    elif args.bfile:
        logger.info(f'Loading genotype from {gfile}.bed...')
        geno = breader(rf'{gfile}') # PLINK format
    elif args.npy:
        logger.info(f'Loading genotype from {gfile}.npz...')
        geno = npyreader(rf'{gfile}') # numpy format
    logger.info(f'Completed, cost: {round(time.time()-t_loading,3)} secs')
    m,n = geno.shape
    n = n - 2
    logger.info(f'Loaded SNP: {m}, individual: {n}')
    samples = geno.columns[2:]
    geno = geno.iloc[:,2:].values
    t_control = time.time()
    logger.info('* Filter SNPs with MAF < 0.01 or missing rate > 0.05; impute with mode.')
    logger.info('Recommended: Use genotype matrix imputed by beagle or impute2 as input')
    qkmodel = QK(geno,maff=0.01)
    geno = qkmodel.M
    logger.info(f'Filter finished, costed {(time.time()-t_control):.2f} secs')
    # Genomic Selection
    for i in pheno.columns:
        t = time.time()
        logger.info('*'*60)
        logger.info(f'* GS process for {i}')
        p = pheno[i]
        namark = p.isna()
        trainmark = np.isin(samples,p.index[~namark])
        testmark = ~trainmark
        # Estimation of train population modeling
        TrainSNP = geno[:,trainmark]
        TrainP = p.loc[samples[trainmark]].values.reshape(-1,1)
        if TrainP.size > 0:
            kfoldset = kfold(TrainSNP.shape[1],k=5,seed=None)
            outpred = []
            for ind,method in enumerate(methods):
                logger.info(f'Method{ind+1}: {method}')
                if method not in ['GBLUP', 'rrBLUP']:
                    print(f'Training the {method} model may take a long time...')
                test4train,train4train = [],[]
                r2_train,r2_test = [],[]
                num = 0
                for test,train in kfoldset:
                    t_fold = time.time()
                    Pred_train,Pred_test = GSapi(TrainP[train],TrainSNP[:,train],TrainSNP[:,test],method=method,PCAdec=args.pcd)
                    ttest = np.concatenate([TrainP[test],Pred_test],axis=1)
                    ttrain = np.concatenate([TrainP[train],Pred_train],axis=1)
                    test4train.append(ttest);train4train.append(ttrain)
                    r2 = 1-np.sum((ttest[:,0]-ttest[:,1])**2)/np.sum((ttest[:,0]-ttest[:,0].mean())**2)
                    r2_test.append(r2)
                    num+=1
                    logger.info(f'Fold{num}: {pearsonr(test4train[num-1][:,0],test4train[num-1][:,1]).statistic:.2f}(pearson), {spearmanr(test4train[num-1][:,0],test4train[num-1][:,1]).statistic:.2f}(spearman), {r2:.2f}(R2). Time costed: {(time.time()-t_fold):.2f} secs')
                showidx = np.argmax(r2_test)
                test4train = test4train[showidx]
                train4train = train4train[showidx]
                if args.plot:
                    fig = plt.figure(figsize=(5,4),dpi=300)
                    gsplot.scatterh(test4train,train4train,color_set=color_set[0],fig=fig)
                    plt.savefig(f'{args.out}/{args.prefix}.{i}.gs.{method}.pdf',transparent=True)
                # Prediction for test population
                TestSNP = geno[:,testmark]
                _TrainP,TestP = GSapi(TrainP,TrainSNP,TestSNP,method=method,PCAdec=args.pcd)
                outpred.append(TestP)
            outpred = pd.DataFrame(np.concatenate(outpred,axis=1),columns=methods,index=samples[testmark])
            outpred.to_csv(f'{args.out}/{args.prefix}.{i}.gs.tsv',sep='\t',float_format='%.4f')
            logger.info(f'Saved in {args.out}/{args.prefix}.{i}.gs.tsv'.replace('//','/'))
    lt = time.localtime()
    endinfo = f'\nFinished, total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
    logger.info(endinfo)

if __name__ == "__main__":
    main()