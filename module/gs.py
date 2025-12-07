# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
for key in ['MPLBACKEND']:
    if key in os.environ:
        del os.environ[key]
import matplotlib as mpl
import logging
mpl.use('Agg')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
from bioplotkit.sci_set import color_set
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import time
import socket
from _common.log import setup_logging
from gfreader import breader,vcfreader,npyreader
from pyBLUP import QK,BLUP,kfold


def main(log:bool=True):
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
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    ## Point out phenotype or snp
    optional_group.add_argument('-n','--ncol', type=int, default=None,
                               help='Only analysis n columns in phenotype ranged from 0-n '
                                   '(default: %(default)s)')
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
    if not os.path.exists(args.out):
        os.mkdir(args.out,0o755)
    logger = setup_logging(f'''{args.out}/{args.prefix}.gs.log'''.replace('\\','/').replace('//','/'))
    logger.info('Genomic Selection Module')
    logger.info(f'Host: {socket.gethostname()}\n')
    # Print configuration summary
    if log:
        logger.info("*"*60)
        logger.info("GENOMIC SELECTION CONFIGURATION")
        logger.info("*"*60)
        logger.info(f"Genotype file:   {gfile}")
        logger.info(f"Phenotype file:  {args.pheno}")
        logger.info(f"Analysis Pcol:   {args.ncol}") if args.ncol is not None else logger.info(f"Analysis Pcol:   All")
        if args.plot:
            logger.info(f"Plot mode:       {args.plot}")
        logger.info(f"Output prefix:   {args.out}/{args.prefix}")
        logger.info("*"*60 + "\n")
    return gfile,args,logger

if __name__ == '__main__':
    t_start = time.time()
    gfile,args,logger = main()
    t_loading = time.time()
    logger.info(f'Loading phenotype from {gfile}...')
    pheno = pd.read_csv(rf'{args.pheno}',sep='\t') # Col 1 - idv ID; row 1 - pheno tag
    pheno = pheno.groupby(pheno.columns[0]).mean() # Mean of duplicated samples
    pheno.index = pheno.index.astype(str)
    assert pheno.shape[1]>0, f'No phenotype data found, please check the phenotype file format!\n{pheno.head()}'
    if args.ncol is not None: 
        assert args.ncol <= pheno.shape[1], "IndexError: Phenotype column index out of range."
        pheno = pheno.iloc[:,[args.ncol]]
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
    logger.info('* Filter SNPs with MAF < 0.01 or missing rate > 0.05; impute with mode...')
    logger.info('Recommended: Use genotype matrix imputed by beagle or impute2 as input')
    qkmodel = QK(geno,maff=0.01)
    logger.info('Completed')
    geno = qkmodel.M
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
            test4train,train4train = [],[]
            mse_train,mse_test = [],[]
            r2_train,r2_test = [],[]
            for test,train in tqdm(kfold(TrainSNP.shape[1],k=5,seed=None),ncols=60):
                model = BLUP(TrainP[train],TrainSNP[:,train],kinship=1)
                ttest = np.concatenate([TrainP[test],model.predict(TrainSNP[:,test])],axis=1)
                ttrain = np.concatenate([TrainP[train],model.predict(TrainSNP[:,train])],axis=1)
                test4train.append(ttest);train4train.append(ttrain)
                mse_train.append(np.sum((ttrain[:,0]-ttrain[:,1])**2)/ttrain.shape[0])
                r2_train.append(1-np.sum((ttrain[:,0]-ttrain[:,1])**2)/np.sum((ttrain[:,0]-ttest[:,0].mean())**2))
                mse_test.append(np.sum((ttest[:,0]-ttest[:,1])**2)/ttest.shape[0])
                r2_test.append(1-np.sum((ttest[:,0]-ttest[:,1])**2)/np.sum((ttest[:,0]-ttest[:,0].mean())**2))
            showidx = np.argmin(np.array(r2_train)-np.array(r2_test))
            test4train = test4train[showidx]
            train4train = train4train[showidx]
            if args.plot:
                fig = plt.figure(figsize=(5,4),dpi=300)
                plt.plot([np.min(test4train),np.max(test4train)],[np.min(test4train),np.max(test4train)],linestyle='--',color=color_set[0][0],alpha=.8,label='y = x (Ideal)')
                plt.scatter(train4train[:,0],train4train[:,1],color=color_set[0][1],alpha=.4,marker='+',label='Train data')
                plt.scatter(test4train[:,0],test4train[:,1],color=color_set[0][0],alpha=.8,marker='*',label='Test data')
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.legend(loc='upper left')
                plt.tight_layout()
                plt.grid(True, alpha=0.3, axis='both')
                plt.gca().text(0.96, 0.04, 
                    f'Train MSE: {np.mean(mse_train):.2f}\nTest MSE: {np.mean(mse_test):.2f}\nTrain R2: {np.mean(r2_train):.2f}\nTest R2: {np.mean(r2_test):.2f}',
                    transform=plt.gca().transAxes,
                    ha='right', va='bottom',
                    color='gray',alpha=.8,
                    multialignment='left')
                plt.savefig(f'{args.out}/{args.prefix}.{i}.gs.cv.pdf',transparent=True)
            # Prediction for test population
            TestSNP = geno[:,testmark]
            model = BLUP(TrainP,TrainSNP,)
            TestP = model.predict(TestSNP)
    lt = time.localtime()
    endinfo = f'\nFinished, total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
    logger.info(endinfo)