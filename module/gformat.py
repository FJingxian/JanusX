# -*- coding: utf-8 -*-
from gfreader import breader,vcfreader,npyreader,genotype2npy,genotype2vcf
from pyBLUP import QK
import pandas as pd
import numpy as np
import argparse
import time
import socket
from _common.log import setup_logging
import os

def main(log:bool=True):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument('-vcf','--vcf', type=str, 
                           help='Input genotype file in VCF format (.vcf or .vcf.gz)')
    geno_group.add_argument('-ivcf','--ivcf', type=str, 
                           help='Input genotype file in VCF format with int number, such as 0,1,2 (.vcf or .vcf.gz)')
    geno_group.add_argument('-fvcf','--fvcf', type=str, 
                           help='Input genotype file in VCF format with float number, such as 0.1,12.3,2e6 (.vcf or .vcf.gz)')
    geno_group.add_argument('-bfile','--bfile', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .bed, .bim, .fam)')
    geno_group.add_argument('-npy','--npy', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .npz, .snp, .idv)')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('-o', '--out', type=str, default=None,
                               help='Output directory for results'
                                   '(default: %(default)s)')
    optional_group.add_argument('-prefix','--prefix', type=str,
                               default=None,
                               help='prefix of output file'
                                   '(default: %(default)s)')
    optional_group.add_argument('-recode','--recode', type=str,
                               default='npy',
                               help='Supported recode format is vcf or npy'
                                   '(default: %(default)s)')
    optional_group.add_argument('-maf','--maf', type=float,
                               default=0,
                               help='Filter threshold of MAF'
                                   '(default: %(default)s)')
    optional_group.add_argument('-geno','--geno', type=float,
                               default=1,
                               help='Filter threshold of MISS for each SNP'
                                   '(default: %(default)s)')
    args = parser.parse_args()
    # Determine genotype file
    if args.vcf:
        gfile = args.vcf
        args.prefix = os.path.basename(gfile).replace('.gz','').replace('.vcf','') if args.prefix is None else args.prefix
    if args.ivcf:
        gfile = args.ivcf
        args.prefix = os.path.basename(gfile).replace('.gz','').replace('.vcf','') if args.prefix is None else args.prefix
    if args.fvcf:
        gfile = args.fvcf
        args.prefix = os.path.basename(gfile).replace('.gz','').replace('.vcf','') if args.prefix is None else args.prefix
    elif args.bfile:
        gfile = args.bfile
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    elif args.npy:
        gfile = args.npy
        args.prefix = os.path.basename(gfile).replace('.npz','') if args.prefix is None else args.prefix
    gfile = gfile.replace('\\','/')
    args.out = os.path.dirname(gfile) if args.out is None else args.out
    # create log file
    if not os.path.exists(args.out):
        os.mkdir(args.out,0o755)
    logger = setup_logging(f'''{args.out}/{args.prefix}.gformat.log'''.replace('\\','/').replace('//','/'))
    logger.info('Genotype Format Transformer')
    logger.info(f'Host: {socket.gethostname()}\n')
    mafmiss = f'maf{args.maf}.miss{args.geno}'
    # Print configuration summary
    if log:
        logger.info("*"*60)
        logger.info("GFT CONFIGURATION")
        logger.info("*"*60)
        logger.info(f"Genotype file: {gfile}")
        logger.info(f"Filter param : {mafmiss if args.maf>0 or args.geno<1 else 'False'}")
        logger.info(f"Recode format: {args.recode}")
        logger.info(f"Output prefix: {args.out}/{args.prefix}")
        logger.info("*"*60 + "\n")
    return gfile,args,logger


if __name__ == "__main__":
    t_start = time.time()
    gfile,args,logger = main()

    assert args.recode in ['vcf','npy'], f'recode must be vcf or npy'
    t_loading = time.time()
    if args.vcf:
        logger.info(f'Loading genotype from {gfile}...')
        geno = vcfreader(rf'{gfile}') # VCF format
    if args.ivcf:
        logger.info(f'Loading genotype from {gfile}...')
        geno = vcfreader(rf'{gfile}',vcftype='ivcf') # VCF format
    if args.fvcf:
        logger.info(f'Loading genotype from {gfile}...')
        geno = vcfreader(rf'{gfile}',vcftype='fvcf') # VCF format
    elif args.bfile:
        logger.info(f'Loading genotype from {gfile}.bed...')
        geno = breader(rf'{gfile}') # PLINK format
    elif args.npy:
        logger.info(f'Loading genotype from {gfile}.npz...')
        geno = npyreader(rf'{gfile}') # numpy format
    logger.info(f'Completed, cost: {round(time.time()-t_loading,3)} secs')
    # if args.maf>0 or geno<1
    m,n = geno.shape
    n = n - 2
    logger.info(f'Loaded SNP: {m}, individual: {n}')
    if not args.ivcf and not args.fvcf:
        if args.maf>0 or args.geno<1:
            qkmodel = QK(geno.iloc[:,2:].values,maff=args.maf,missf=args.geno)
            samples = geno.columns[2:]
            ref_alt:pd.DataFrame = geno.iloc[:,:2]
            ref_alt = ref_alt.loc[qkmodel.SNPretain]
            ref_alt.iloc[qkmodel.maftmark,[0,1]] = ref_alt.iloc[qkmodel.maftmark,[1,0]]
            geno = pd.concat([ref_alt,pd.DataFrame(qkmodel.M,index=ref_alt.index,columns=samples)],axis=1)
            m,n = geno.shape
            n = n - 2
            logger.info(f'After filtering, SNP: {m}, individual: {n}, mean of maf: {np.mean(qkmodel.maf):.3f}, mean of miss: {np.mean(qkmodel.missrate):.3f}')
            del qkmodel
    logger.info(f'Genotype is transformed to {args.recode} format...')
    if args.recode == 'npy':
        genotype2npy(geno,f'{args.out}/{args.prefix}')
        logger.info(f'Saved in {args.out}/{args.prefix}.npz, {args.out}/{args.prefix}.idv and {args.out}/{args.prefix}.snp')
    if args.recode == 'vcf':
        genotype2vcf(geno,f'{args.out}/{args.prefix}')
        logger.info(f'Saved in {args.out}/{args.prefix}.vcf')

    lt = time.localtime()
    endinfo = f'\nFinished, Total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
    logger.info(endinfo)