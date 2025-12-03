# -*- coding: utf-8 -*-
from gfreader import breader,vcfreader,npyreader
from pyBLUP import QK
import pandas as pd
import numpy as np
import argparse
import time
import socket
import logging
from _common.log import setup_logging
import sys
import os
from bioplotkit.sci_set import color_set

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
    geno_group.add_argument('-pcfile','--pcfile', type=str, 
                           help='PCA result files only for visualization (prefix for .eigenval, .eigenvec, .eigenvec.id)')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('-o', '--out', type=str, default=None,
                               help='Output directory for results'
                                   '(default: %(default)s)')
    optional_group.add_argument('-prefix','--prefix', type=str,default=None,
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
        args.prefix = os.path.basename(gfile).replace('.npz','') if args.prefix is None else args.prefix
    elif args.pcfile:
        gfile = args.pcfile
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    gfile = gfile.replace('\\','/')
    args.out = os.path.dirname(gfile) if args.out is None else args.out
    args.color = color_set[args.color]
    # Build argument list for the original script
    sys.argv = [
        sys.argv[0],  # script name
        gfile,
        args.dim,
        args.plot,
        args.plot3D,
        args.group,
        args.color,
        args.out,
        args.prefix,
    ]
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
        if not args.pcfile:
            logger.info(f"Genotype file: {gfile}")
            logger.info(f"Out dimension: {args.dim}")
        else:
            logger.info(f"PC resultfile: {gfile}")
        if args.plot or args.plot3D:
            logger.info(f"2DVisulaztion: {args.plot}")
            logger.info(f"3DVisulaztion: {args.plot3D}")
        if args.group:
            logger.info(f"Group file: {args.group}")
            logger.info(f"Colors set: {args.color}")
        logger.info(f"Output prefix: {args.out}/{args.prefix}")
        logger.info("*"*60 + "\n")
    return gfile,args,logger


if __name__ == '__main__':
    t_start = time.time()
    gfile,args,logger = main()
    t_loading = time.time()
    if not args.pcfile:
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
    else:
        logger.info(f'Loading genotype from {gfile}.eigenvec, {gfile}.eigenvec.id and {gfile}.eigenval...')
        eigenvec,samples,eigenval = np.genfromtxt(f'{gfile}.eigenvec'),np.genfromtxt(f'{gfile}.eigenvec.id'),np.genfromtxt(f'{gfile}.eigenval')
    if args.plot or args.plot3D:
        from bioplotkit import PCSHOW
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.use('Agg')
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
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
            df_pc = pd.DataFrame(eigenvec[:,:3],index=samples,columns=[f'''PC{i+1}({round(float(exp[i]),2)}%)''' for i in range(3)])
            if args.group:
                df_pc = pd.concat([df_pc,pd.read_csv(args.group,sep='\t',index_col=0,)],axis=1)
                group = df_pc.columns[3]
                pcshow = PCSHOW(df_pc)
                fig = pcshow.pcplot3D(df_pc.columns[0],df_pc.columns[1],df_pc.columns[2],group,textanno,color_set[6])
            else:
                fig = pcshow.pcplot3D(df_pc.columns[0],df_pc.columns[1],df_pc.columns[2])
            fig.write_html(f'{args.out}/{args.prefix}.eigenvec.3D.html')
            logger.info(f'3D figure was saved in {args.out}/{args.prefix}.eigenvec.3D.html')
    lt = time.localtime()
    endinfo = f'\nFinished, total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
    logger.info(endinfo)