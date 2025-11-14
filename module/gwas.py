#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Examples:
  # Basic usage with VCF file
  --vcf genotypes.vcf --pheno phenotypes.txt --out results
  
  # Using PLINK binary files with custom parameters
  --bfile genotypes --pheno phenotypes.txt --out results --grm gemma1 --qcov 5 --thread 8
  
  # Using external kinship matrix and disabling HighAC
  --vcf genotypes.vcf --pheno phenotypes.txt --out results --grm kinship_matrix.txt --qcov 10 --no-AC
  
  # Maximum performance with all threads
  --bfile genotypes --pheno phenotypes.txt --out results --grm VanRanden --qcov 3 --cov covfile.txt --thread -1

File Formats:
  VCF/BFILE:    Standard VCF or PLINK binary format (bed/bim/fam)
  PHENO:        Tab-delimited file with sample IDs in first column and phenotypes in subsequent columns
  GRM File:     Space/tab-delimited kinship matrix file
  QCOV File:    Space/tab-delimited covariate matrix file
  COV File:    Space/tab-delimited covariate matrix file
        
Citation:
  https://github.com/MaizeMan-JxFU/pyBLUP/
'''

from pyBLUP import QK,GWAS
from _gfreader import breader,vcfreader
import pandas as pd
import numpy as np
import argparse
import time
import socket
import logging
import sys
import os

def format_dataframe_for_export(df:pd.DataFrame, scientific_cols=None, float_cols=None):
    """
    Parameters:
    - df: raw DataFrame
    - scientific_cols: 科学计数法列
    - float_cols: 浮点数列
    """
    df_export = df.copy()
    # 科学计数法
    if scientific_cols:
        for col in scientific_cols:
            if col in df_export.columns and df_export[col].dtype in [np.float64, np.int64]:
                df_export[col] = df_export[col].apply(lambda x: f"{x:.4e}")
    # 浮点数
    if float_cols:
        for col in float_cols:
            if col in df_export.columns and df_export[col].dtype in [np.float64, np.int64]:
                df_export[col] = df_export[col].apply(lambda x: f"{x:.4f}")
    return df_export
def setup_logging(log_file_path):
    """set logging"""
    if os.path.exists(log_file_path) and log_file_path[-4:]=='.log':
        os.remove(log_file_path)
    # creart logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # clean exist handler
    logger.handlers.clear()
    # set log format
    formatter = logging.Formatter()
    # file handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    # handler of control panel
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    # add handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
def main(log:bool=True):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument('--vcf', type=str, 
                           help='Input genotype file in VCF format (.vcf or .vcf.gz)')
    geno_group.add_argument('--bfile', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .bed, .bim, .fam)')
    required_group.add_argument('--pheno', type=str, required=True,
                               help='Phenotype file (tab-delimited with sample IDs in first column)')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('--out', type=str, default='test',
                               help='Output directory for results'
                                   '(default: %(default)s)')
    optional_group.add_argument('--grm', type=str,
                               default='VanRanden',
                               help='Kinship matrix calculation method or path to pre-calculated GRM file '
                                   '(default: %(default)s)')
    optional_group.add_argument('--qcov', type=str, default='3',
                               help='Number of principal components for Q matrix or path to covariate matrix file '
                                   '(default: %(default)s)')
    optional_group.add_argument('--cov', type=str, default=None,
                               help='Path to Covariance file '
                                   '(default: %(default)s)')
    optional_group.add_argument('--thread', type=int, default=-1,
                               help='Number of CPU threads to use (-1 for all available cores, default: %(default)s)')
    optional_group.add_argument('--AC', action='store_true', default=True,
                               help='Enable HighAC mode for GWAS (default: %(default)s)')
    optional_group.add_argument('--no-AC', action='store_false', dest='AC',
                               help='Disable HighAC mode')
    args = parser.parse_args()
    # Determine genotype file
    gfile = args.vcf if args.vcf else args.bfile
    # Build argument list for the original script
    sys.argv = [
        sys.argv[0],  # script name
        gfile,
        args.pheno,
        args.out,
        args.grm,
        args.qcov,
        args.cov,
        str(args.AC)
    ]
    # create log file
    if not os.path.exists(args.out):
        os.mkdir(args.out,0o755)
    
    filename = os.path.basename(gfile)
    logger = setup_logging(f'''{args.out}/{filename.replace('.vcf','').replace('.gz','')}.log'''.replace('//','/'))
    logger.info('High Performance Linear Mixed Model Solver for Genome-Wide Association Studies')
    logger.info(f'Host: {socket.gethostname()}\n')
    # Print configuration summary
    if log:
        logger.info("*"*60)
        logger.info("GWAS LMM SOLVER CONFIGURATION")
        logger.info("*"*60)
        logger.info(f"Genotype file:    {gfile}")
        logger.info(f"Phenotype file:   {args.pheno}")
        logger.info(f"Output directory: {args.out}")
        logger.info(f"GRM method:       {args.grm}")
        logger.info(f"Q matrix:         {args.qcov}")
        logger.info(f"Covariant matrix: {args.cov}")
        logger.info(f"Threads:          {args.thread} ({'All cores' if args.thread == -1 else 'User specified'})")
        logger.info(f"HighAC mode:      {args.AC}")
        logger.info("*"*60 + "\n")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.out):
        os.makedirs(args.out, mode=0o755)
        if log:
            print(f"Created output directory: {args.out}")
    return gfile,args,logger

t_start = time.time()
gfile,args,logger = main()
phenofile,outfolder = args.pheno,args.out
kinship_method = args.grm
qdim = args.qcov
cov = args.cov
HighAC = args.AC
threads = args.thread
kcal = True if kinship_method in ['VanRanden', 'gemma1', 'gemma2', 'pearson'] else False
qcal = True if qdim in np.arange(1,20).astype(str) else False

# test exist of all input files
assert os.path.isfile(gfile), f"can not find file: {gfile}"
assert os.path.isfile(phenofile), f"can not find file: {phenofile}"

if not kcal:
    assert os.path.isfile(kinship_method), f"{kinship_method} is not a GRM calculation method of kinship or a file"
if not qcal:
    assert os.path.isfile(qdim), f"{qdim} is not a dimension of q matrix or a file"
if cov is not None:
    assert os.path.isfile(cov), f"{cov} is not a file"

if not os.path.exists(outfolder):
    os.makedirs(outfolder,mode=0o755)
prefix = gfile.replace('.vcf','').replace('.gz','')

if args.vcf:
    logger.info(f'Loading genotype from {gfile}...')
    geno = vcfreader(rf'{gfile}') # VCF format
    ref_alt = geno.iloc[:,:2]
    geno = geno.iloc[:,2:].T 
else:
    logger.info(f'Loading genotype from {gfile}.bed...')
    geno = breader(rf'{gfile}') # PLINK format
    ref_alt = geno.iloc[:,:2]
    geno = geno.iloc[:,2:].T
geno.index = geno.index.astype(str)
famid = geno.index
logger.info(f'Loading phenotype from {phenofile}...')
pheno = pd.read_csv(rf'{phenofile}',sep='\t') # 第一列是样本ID, 第一行是表型名
pheno = pheno.groupby(pheno.columns[0]).mean() # 重复样本表型取均值
pheno.index = pheno.index.astype(str)
logger.info('Geno and Pheno are ready!')

if qcal or kcal:
    if not os.path.exists(f'{prefix}.k.{kinship_method}.txt') or not os.path.exists(f'{prefix}.q.{qdim}.txt') and int(qdim)!=0:
        qkmodel = QK(geno.values,low_memory=True,log=True)
        logger.info(f'Samples and SNP: {geno.shape}')
    if os.path.exists(f'{prefix}.k.{kinship_method}.txt'):
        logger.info(f'* Loading GRM from {prefix}.k.{kinship_method}.txt...')
        kmatrix = pd.read_csv(f'{prefix}.k.{kinship_method}.txt',sep=r'\s+',header=None).values
    else:    
        logger.info(f'* Calculation method of kinship matrix is {kinship_method}')
        kmatrix = qkmodel.kinship(method=kinship_method)
        np.savetxt(f'{prefix}.k.{kinship_method}.txt',kmatrix,fmt='%.6f')

    if os.path.exists(f'{prefix}.q.{qdim}.txt'):
        logger.info(f'* Loading Q matrix from {prefix}.q.{qdim}.txt...')
        qmatrix = pd.read_csv(f'{prefix}.q.{qdim}.txt',sep=r'\s+',header=None).values
    else:
        if int(qdim) > 0:
            logger.info(f'* Dimension of PC for q matrix is {qdim}')
            qmatrix,eigenval = qkmodel.rpca(dim=int(qdim))
            np.savetxt(f'{prefix}.q.{qdim}.txt',qmatrix,fmt='%.6f')
        else:
            qmatrix = np.array([]).reshape(geno.shape[0],0)
else:
    if qdim == '0':
        qmatrix = np.array([]).reshape(geno.shape[0],0)
    elif not qcal and os.path.exists(qdim):
        logger.info(f'* Loading Q matrix from {qdim}...')
        qmatrix = np.genfromtxt(qdim)
    
    if not kcal and os.path.exists(kinship_method):
        logger.info(f'* Loading GRM from {kinship_method}...')
        kmatrix = np.genfromtxt(kinship_method)
        
if cov is not None:
    cov = np.genfromtxt(cov,).reshape(-1,1)
    logger.info(f'Covmatrix {cov.shape}:')
    qmatrix = np.concatenate([qmatrix,cov],axis=1)

logger.info(f'GRM {str(kmatrix.shape)}:')
logger.info(kmatrix[:5,:5])
logger.info(f'Qmatrix {str(qmatrix.shape)}:')
logger.info(qmatrix[:5,:5])

for i in pheno.columns:
    t = time.time()
    logger.info('*'*60)
    p = pheno[i].dropna()
    famid_pheno = [i for i in famid if i in p] # 对齐样本 以geno顺序为准
    famid_geno = [i for i in range(len(famid)) if famid[i] in famid_pheno] # 对齐样本 以geno顺序为准
    p = p.loc[famid_pheno].values.reshape(-1,1)
    if len(p)>0:
        gwasmodel = GWAS(y=p,X=qmatrix[famid_geno,:],kinship=kmatrix[famid_geno,:][:,famid_geno])
        logger.info(f'''Phenotype: {i}, Number of samples: {len(famid_geno)}, Number of SNP: {geno.shape[1]}, pve of null: {round(gwasmodel.pve,3)}, high AC model: {HighAC}''')
        if HighAC:
            results = gwasmodel.gwasHAC(snp=geno.values[famid_geno,:],chunksize=200_000,threads=threads) # gwas running...
            # np.savetxt(f'{outfolder}/{i}.lbd',np.array(gwasmodel.lbd),fmt='%.4f')
        else:
            results = gwasmodel.gwas(snp=geno.values[famid_geno,:],chunksize=200_000,threads=threads) # gwas running...
        logger.info(f'Effective number of SNP: {results.shape[0]}')
        results = pd.DataFrame(results,columns=['beta','se','af','p'],index=ref_alt.index[gwasmodel.snp_retain])
        results = pd.concat([ref_alt.iloc[gwasmodel.snp_retain,:],results],axis=1)
        results = results.reset_index()
        results_save = format_dataframe_for_export(results, scientific_cols=['p'], float_cols=['beta','se','af'])
        results_save.to_csv(f'{outfolder}/{i}.assoc.tsv',sep='\t',index=False)
        logger.info(f'Saved in {outfolder}/{i}.assoc.tsv'.replace('//','/'))
        del results,results_save,gwasmodel,p,famid_pheno,famid_geno
    else:
        logger.info(f'Phenotype {i} has no overlapping samples with genotype, please check sample id. skipped.\n')
    logger.info(f'Time costed: {round(time.time()-t,2)} secs\n')
lt = time.localtime()
endinfo = f'\nFinished, Total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
logger.info(endinfo)