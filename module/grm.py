# -*- coding: utf-8 -*-
from gfreader import breader,vcfreader,npyreader
from pyBLUP import QK
import numpy as np
import argparse
import time
import socket
import os
from _common.log import setup_logging

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
    geno_group.add_argument('-bfile','--bfile', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .bed, .bim, .fam)')
    geno_group.add_argument('-npy','--npy', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .npz, .snp, .idv)')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('-o', '--out', type=str, default=None,
                               help='Output directory for results'
                                   '(default: %(default)s)')
    optional_group.add_argument('-prefix','--prefix', type=str,default=None,
                               help='prefix of output file'
                                   '(default: %(default)s)')
    optional_group.add_argument('-m','--method', type=int, default=1,
                               help='GRM calculation method: 1-centralization, 2-standardization'
                                   '(default: %(default)s)')
    optional_group.add_argument('-npz','--npz', action='store_true', default=False,
                               help='Compression'
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
        args.prefix = os.path.basename(gfile).replace('.npz','') if args.prefix is None else args.prefix
    gfile = gfile.replace('\\','/')
    args.out = os.path.dirname(gfile) if args.out is None else args.out
    # create log file
    if not os.path.exists(args.out):
        os.mkdir(args.out,0o755)
    logger = setup_logging(f'''{args.out}/{args.prefix}.pca.log'''.replace('\\','/').replace('//','/'))
    logger.info('Principle Component Analysis')
    logger.info(f'Host: {socket.gethostname()}\n')
    # Print configuration summary
    if log:
        logger.info("*"*60)
        logger.info("GFT CONFIGURATION")
        logger.info("*"*60)
        logger.info(f"Genotype file: {gfile}")
        logger.info(f"Method of GRM: {'Centralization' if args.method==1 else 'Standardization'}")
        if args.npz:
            logger.info(f"Saved as npyz: {args.npz}")
        logger.info(f"Output prefix: {args.out}/{args.prefix}")
        logger.info("*"*60 + "\n")
    return gfile,args,logger

t_start = time.time()
gfile,args,logger = main()

t_loading = time.time()
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
logger.info('* Calculating GRM...')
logger.info(f'Loaded SNP: {m}, individual: {n}')
samples = geno.columns[2:]
geno = geno.iloc[:,2:].values
qkmodel = QK(geno)
logger.info(f'Effective SNP: {qkmodel.M.shape[0]}')
grm = qkmodel.GRM(method=args.method)
del qkmodel
if args.npz:
    np.savetxt(f'{args.out}/{args.prefix}.grm.id',samples.values,fmt='%s');np.savez_compressed(f'{args.out}/{args.prefix}.grm')
    logger.info(f'Saved in {args.out}:\n{args.prefix}.grm.id {args.prefix}.grm.npz')
else:
    np.savetxt(f'{args.out}/{args.prefix}.grm.id',samples.values,fmt='%s');np.savetxt(f'{args.out}/{args.prefix}.grm.txt',grm,fmt='%.6f')
    logger.info(f'Saved in {args.out}:\n{args.prefix}.grm.id {args.prefix}.grm.txt')

lt = time.localtime()
endinfo = f'\nFinished, total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
logger.info(endinfo)