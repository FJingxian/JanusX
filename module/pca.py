# -*- coding: utf-8 -*-
import os
for key in ['MPLBACKEND']:
    if key in os.environ:
        del os.environ[key]
import pandas as pd
import numpy as np
import argparse
import time
import socket
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
from bioplotkit.sci_set import color_set
from bioplotkit import PCSHOW
from gfreader import breader,vcfreader,npyreader
from pyBLUP import QK,Eigendec
from _common.log import setup_logging

def main(log:bool=True):
    t_start = time.time()
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
    geno_group.add_argument('-grm','--grm', type=str, 
                           help='GRM file for PC calculation (prefix for .grm.id and .grm.txt, or .grm.id and .grm.npz)')
    geno_group.add_argument('-pcfile','--pcfile', type=str, 
                           help='PCA result files only for visualization (prefix for .eigenval, .eigenvec, .eigenvec.id)')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('-o', '--out', type=str, default=None,
                               help='Output directory for results'
                                   '(default: %(default)s)')
    optional_group.add_argument('-prefix','--prefix',type=str,default=None,
                               help='prefix of output file'
                                   '(default: %(default)s)')
    optional_group.add_argument('-dim','--dim', type=int, default=3,
                               help='Output dimension of principle component'
                                   '(default: %(default)s)')
    optional_group.add_argument('-plot','--plot', action='store_true', default=False,
                               help='Visulazation of sample in PC1, PC2 and PC3'
                                   '(default: %(default)s)')
    optional_group.add_argument('-plot3D','--plot3D', action='store_true', default=False,
                               help='3D visulazation of sample in PC1, PC2 and PC3'
                                   '(default: %(default)s)')
    optional_group.add_argument('-group','--group', type=str, default=None,
                               help='Group file with 2 columns included samples and groups and without header'
                                   '(default: %(default)s)')
    optional_group.add_argument('-color','--color', type=int, default=1,
                               help='Color style for manhanden and qq figure, 0-6 (default: %(default)s)')
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
    elif args.grm:
        gfile = args.grm
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    elif args.pcfile:
        gfile = args.pcfile
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    gfile = gfile.replace('\\','/')
    args.out = os.path.dirname(gfile) if args.out is None else args.out
    args.color = color_set[args.color]
    # create log file
    if not os.path.exists(args.out):
        os.mkdir(args.out,0o755)
    logger = setup_logging(f'''{args.out}/{args.prefix}.pca.log'''.replace('\\','/').replace('//','/'))
    logger.info('Principle Component Analysis Module')
    logger.info(f'Host: {socket.gethostname()}\n')
    # Print configuration summary
    if log:
        logger.info("*"*60)
        logger.info("PCA CONFIGURATION")
        logger.info("*"*60)
        if args.npy or args.vcf or args.bfile:
            logger.info(f"Genotype file: {gfile}")
            logger.info(f"Out dimension: {args.dim}")
        elif args.grm:
            logger.info(f"grm file prefix: {gfile}")
        elif args.pcfile:
            logger.info(f"PC resultfile: {gfile}")
        if args.plot or args.plot3D:
            logger.info(f"2DVisulaztion: {args.plot}")
            logger.info(f"3DVisulaztion: {args.plot3D}")
        if args.group:
            logger.info(f"Group file: {args.group}")
            logger.info(f"Colors set: {args.color}")
        logger.info(f"Output prefix: {args.out}/{args.prefix}")
        logger.info("*"*60 + "\n")

    t_loading = time.time()
    if args.npy or args.vcf or args.bfile:
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
        logger.info('* Calculating PC...')
        logger.info(f'Loaded SNP: {m}, individual: {n}')
        samples = geno.columns[2:]
        geno = geno.iloc[:,2:].values
        qkmodel = QK(geno)
        logger.info(f'Effective SNP: {qkmodel.M.shape[0]}')
        eigenvec, eigenval = qkmodel.PCA()
        del qkmodel
        np.savetxt(f'{args.out}/{args.prefix}.eigenvec.id',samples.values,fmt='%s')
        np.savetxt(f'{args.out}/{args.prefix}.eigenvec',eigenvec[:,:args.dim],fmt='%.6f')
        np.savetxt(f'{args.out}/{args.prefix}.eigenval',eigenval,fmt='%.2f')
        logger.info(f'Saved in folder "{args.out}" with files named {args.prefix}.eigenvec, {args.prefix}.eigenvec.id and {args.prefix}.eigenval')
    elif args.grm:
        assert os.path.exists(f'{gfile}.grm.id'), f'GRM file is not existed'
        assert os.path.exists(f'{gfile}.grm.txt') or os.path.exists(f'{gfile}.grm.npz'), f'GRM file is not existed'
        if os.path.exists(f'{gfile}.grm.txt'):
            logger.info(f'Loading GRM from {gfile}.grm.txt, {gfile}.grm.id...')
            eigenvec, eigenval = Eigendec(np.genfromtxt(f'{gfile}.grm.txt'))
            samples = np.genfromtxt(f'{gfile}.grm.id',dtype=str)
            np.savetxt(f'{args.out}/{args.prefix}.eigenvec.id',samples,fmt='%s')
            np.savetxt(f'{args.out}/{args.prefix}.eigenvec',eigenvec[:,:args.dim],fmt='%.6f')
            np.savetxt(f'{args.out}/{args.prefix}.eigenval',eigenval,fmt='%.2f')
            logger.info(f'Saved in folder "{args.out}" with files named {args.prefix}.eigenvec, {args.prefix}.eigenvec.id and {args.prefix}.eigenval')
        elif os.path.exists(f'{gfile}.grm.npz'):
            logger.info(f'Loading GRM from {gfile}.grm.npz, {gfile}.grm.id...')
            eigenvec, eigenval = Eigendec(np.load(f'{gfile}.grm.npz')['arr_0'])
            samples = np.genfromtxt(f'{gfile}.grm.id',dtype=str)
            np.savetxt(f'{args.out}/{args.prefix}.eigenvec.id',samples,fmt='%s')
            np.savetxt(f'{args.out}/{args.prefix}.eigenvec',eigenvec[:,:args.dim],fmt='%.6f')
            np.savetxt(f'{args.out}/{args.prefix}.eigenval',eigenval,fmt='%.2f')
            logger.info(f'Saved in folder "{args.out}" with files named {args.prefix}.eigenvec, {args.prefix}.eigenvec.id and {args.prefix}.eigenval')
    elif args.pcfile:
        logger.info(f'Loading PC result from {gfile}.eigenvec, {gfile}.eigenvec.id and {gfile}.eigenval...')
        eigenvec,samples,eigenval = np.genfromtxt(f'{gfile}.eigenvec'),np.genfromtxt(f'{gfile}.eigenvec.id',dtype=str),np.genfromtxt(f'{gfile}.eigenval')
    if args.plot:
        logger.info('* Visualizing...')
        exp = 100*eigenval/np.sum(eigenval)
        df_pc = pd.DataFrame(eigenvec[:,:3],index=samples,columns=[f'''PC{i+1}({round(float(exp[i]),2)}%)''' for i in range(3)])
        if args.group:
            df_pc = pd.concat([df_pc,pd.read_csv(args.group,sep='\t',index_col=0,)],axis=1)
            group = df_pc.columns[3]
            textanno = df_pc.columns[4]
        else:
            group,textanno = None,None
        pcshow = PCSHOW(df_pc)
        fig = plt.figure(figsize=(10,4),dpi=300)
        ax1 = fig.add_subplot(121);ax1.set_xlabel(f'{df_pc.columns[0]}');ax1.set_ylabel(f'{df_pc.columns[1]}')
        ax2 = fig.add_subplot(122);ax2.set_xlabel(f'{df_pc.columns[0]}');ax2.set_ylabel(f'{df_pc.columns[2]}')
        pcshow.pcplot(df_pc.columns[0],df_pc.columns[1],group=group,ax=ax1,color_set=args.color,anno_tag=textanno)
        pcshow.pcplot(df_pc.columns[0],df_pc.columns[2],group=group,ax=ax2,color_set=args.color,anno_tag=textanno)
        plt.tight_layout()
        plt.savefig(f'{args.out}/{args.prefix}.eigenvec.2D.pdf',transparent=True)
        logger.info(f'2D figure was saved in {args.out}/{args.prefix}.eigenvec.2D.pdf')
        plt.close()
    if args.plot3D:
        logger.info('* 3D Visualizing...')
        exp = 100*eigenval/np.sum(eigenval)
        df_pc = pd.DataFrame(eigenvec[:,:3],index=samples,columns=[f'''PC{i+1}({round(float(exp[i]),2)}%)''' for i in range(3)])
        if args.group:
            df_pc = pd.concat([df_pc,pd.read_csv(args.group,sep='\t',index_col=0,)],axis=1)
            group = df_pc.columns[3]
            pcshow = PCSHOW(df_pc)
            fig = pcshow.pcplot3D(df_pc.columns[0],df_pc.columns[1],df_pc.columns[2],group,textanno,color_set[6])
        else:
            pcshow = PCSHOW(df_pc)
            fig = pcshow.pcplot3D(df_pc.columns[0],df_pc.columns[1],df_pc.columns[2])
        fig.write_html(f'{args.out}/{args.prefix}.eigenvec.3D.html')
        logger.info(f'3D figure was saved in {args.out}/{args.prefix}.eigenvec.3D.html')
    lt = time.localtime()
    endinfo = f'\nFinished, total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
    logger.info(endinfo)
    
if __name__ == '__main__':
    main()