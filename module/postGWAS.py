# -*- coding: utf-8 -*-
'''
Examples:
  # Basic usage with gwasplot
  -f result.assoc.txt
  # Usage of different column names with gwasplot
  -f result.assoc.txt -chr "chr" -pos "pos" -pvalue "P_wald"
  # Usage of gwasplot and point out output path
  -f result.assoc.txt -chr "chr" -pos "pos" -pvalue "P_wald" --out test # it will saved in test/result.assoc.qq.pdf and test/result.assoc.manh.pdf
        
Citation:
  https://github.com/MaizeMan-JxFU/JanusX/
'''
import os
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
from bioplotkit import GWASPLOT
from bioplotkit.sci_set import color_set
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import time
import socket
from _common.log import setup_logging
from _common.readanno import readanno
from joblib import Parallel,delayed

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    required_group.add_argument('-f','--file',nargs='+',type=str,required=True,
                               help='File of gwas results')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('-chr', '--chr', type=str, default='#CHROM',
                               help='Column name of chr'
                                   '(default: %(default)s)')
    optional_group.add_argument('-pos','--pos', type=str, default='POS',
                               help='Column name of position'
                                   '(default: %(default)s)')
    optional_group.add_argument('-pvalue','--pvalue', type=str, default='p',
                               help='Column name of pvalue'
                                   '(default: %(default)s)')
    optional_group.add_argument('-threshold','--threshold', type=float, default=None,
                               help='treshold of pvalue'
                                   '(default: %(default)s)')
    optional_group.add_argument('-noplot','--noplot', action='store_false', default=True,
                               help='disabling plot manhanden figure (default: %(default)s)')
    optional_group.add_argument('-color','--color', type=int, default=0,
                               help='Color style for manhanden and qq figure, 0-6 (default: %(default)s)')
    optional_group.add_argument('-hl','--highlight', type=str, default=None,
                               help='Hightlight SNP with gene name in .bed file, eg. 1\t100021\t100021\tgenename\tfunction (default: %(default)s)')
    optional_group.add_argument('-a','--anno', type=str, default=None,
                               help='annotation option, .gff file or .bed file'
                                   '(default: %(default)s)')
    optional_group.add_argument('-ab','--annobroaden', type=float, default=None,
                               help='broaden range of chromosome (Kb)'
                                   '(default: %(default)s)')
    optional_group.add_argument('-descItem','--descItem', type=str, default='description',
                               help='description items in gff file (dev)'
                                   '(default: %(default)s)')
    optional_group.add_argument('-o', '--out', type=str, default=None,
                               help='Output folder path'
                                   '(default: %(default)s)')
    optional_group.add_argument('-prefix','--prefix', type=str,
                               default=None,
                               help='prefix of output file'
                                   '(default: %(default)s)')
    optional_group.add_argument('-t','--thread', type=int, default=-1,
                               help='Number of CPU threads to use (-1 for all available cores, default: %(default)s)')
    args = parser.parse_args()
    assert args.color <= 6, 'colorset error: try 0-6'
    args.color = color_set[args.color]
    args.out = os.path.dirname(args.file[0]) if args.out is None else args.out
    args.prefix = 'JanusX' if args.prefix is None else args.prefix
    # create log file
    # Create output directory if it doesn't exist
    if args.out != '':
        os.makedirs(args.out, mode=0o755,exist_ok=True)
    else:
        args.out = '.'
    logger = setup_logging(f'''{args.out}/{args.prefix}.postGWAS.log'''.replace('//','/'))
    # Print configuration summary
    logger.info('Script of GWAS post analysis')
    logger.info(f'Host: {socket.gethostname()}\n')
    logger.info("*"*60)
    logger.info(f"File:          {args.file}")
    logger.info(f"Chr tag:       {args.chr}")
    logger.info(f"Pos tag:       {args.pos}")
    logger.info(f"Pvalue tag:    {args.pvalue}")
    if args.noplot:
        logger.info("GWAS Visulazation:")
        logger.info(f"Threshold:     {args.threshold if args.threshold else '0.05/nSNP'}")
        logger.info(f"Color:         {args.color}")
        logger.info(f'Highlight bed: {args.highlight}')
    if args.anno:
        logger.info("GWAS Annotation:")
        logger.info(f"Anno file:     {args.anno}")
        logger.info(f"Annobroad(kb): {args.annobroaden}")
    logger.info(f"Output prefix: {args.out}/{args.prefix}")
    logger.info(f"Threads:       {args.thread} ({'All cores' if args.thread == -1 else 'User specified'})")
    logger.info("*"*60 + "\n")
    return args,logger

if __name__ == "__main__":
    t_start = time.time()
    args,logger = main()
    def GWASplot(file):
        args.prefix = os.path.basename(file).replace('.tsv','').replace('.txt','')
        chr_string,pos_string,pvalue_string = args.chr,args.pos,args.pvalue
        df = pd.read_csv(file,sep='\t',usecols=[chr_string,pos_string,pvalue_string])
        threshold = args.threshold if args.threshold is not None else 0.05/df.shape[0]
        if args.noplot:
            t_plot = time.time()
            logger.info('* Visualizing...')
            plotmodel = GWASPLOT(df,chr_string,pos_string,pvalue_string,0.1)
            fig = plt.figure(figsize=(8,4),dpi=300)
            ax =fig.add_subplot(111,)
            if args.highlight:
                df_hl = pd.read_csv(args.highlight,sep='\t',index_col=None,header=None,)
                genenamemask = df_hl[3].isna()
                df_hl.loc[genenamemask,3] = df_hl.loc[genenamemask,0].astype(str)+'_'+df_hl.loc[genenamemask,1].astype(str)
                df_hl = df_hl.set_index([0,1])
                df_hl_idx = df_hl.index[df_hl.index.isin(plotmodel.df.index)]
                assert len(df_hl_idx)>0, 'Nothing to highlight, check bed file.'
                ax.scatter(plotmodel.df.loc[df_hl_idx,'x'],-np.log10(plotmodel.df.loc[df_hl_idx,'y']),marker='D',color='red',zorder=10,s=32,edgecolors='black')
                ax.hlines(y=-np.log10(threshold),xmin=-1e10,xmax=1e10,linestyle='dashed',color='grey')
                for idx in df_hl_idx:
                    text = df_hl.loc[idx,3]
                    ax.text(plotmodel.df.loc[idx,'x'],-np.log10(plotmodel.df.loc[idx,'y']),s=text,ha='center',zorder=11)
                plotmodel.manhattan(None,ax=ax,color_set=args.color,ignore=df_hl_idx)
            else:
                plotmodel.manhattan(-np.log10(threshold),ax=ax,color_set=args.color)
            plt.tight_layout()
            plt.savefig(f'{args.out}/{args.prefix}.manh.pdf',transparent=True)
            plt.close(fig)
            fig = plt.figure(figsize=(5,4),dpi=300)
            ax2 =fig.add_subplot(111,)
            plotmodel.qq(ax=ax2,color_set=args.color)
            plt.tight_layout()
            plt.savefig(f'{args.out}/{args.prefix}.qq.pdf',transparent=True)
            plt.close(fig)
            logger.info(f'Saved in {args.out}/{args.prefix}.manh.pdf and {args.out}/{args.prefix}.qq.pdf')
            logger.info(f'Completed, costed {round(time.time()-t_plot,2)} secs\n')
        if args.anno:
            logger.info('* Annotating...')
            if os.path.exists(args.anno):
                t_anno = time.time()
                df_filter = df.loc[df[pvalue_string]<=threshold,[chr_string,pos_string,pvalue_string]].set_index([chr_string,pos_string])
                anno = readanno(args.anno,args.descItem) # After treating: anno 0-chr,1-start,2-end,3-geneID,4-description1,5-description2
                desc = list(map(lambda x:anno.loc[(anno[0]==x[0])&(anno[1]<=x[1])&(anno[2]>=x[1])], df_filter.index))
                df_filter['desc'] = list(map(lambda x:f'''{x.iloc[0,3]};{x.iloc[0,4]};{x.iloc[0,5]}''' if not x.empty else 'NA;NA;NA', desc))
                if args.annobroaden:
                    desc = list(map(lambda x:anno.loc[(anno[0]==x[0])&(anno[1]<=x[1]+args.annobroaden*1_000)&(anno[2]>=x[1]-args.annobroaden*1_000)], df_filter.index))
                    df_filter['broaden'] = list(map(lambda x:f'''{'|'.join(x.iloc[:,3])};{'|'.join(x.iloc[:,4])};{'|'.join(x.iloc[:,5])}''' if not x.empty else 'NA;NA;NA', desc))
                logger.info(df_filter)
                df_filter.to_csv(f'{args.out}/{args.prefix}.{threshold}.anno.tsv',sep='\t')
                logger.info(f'Saved in {args.out}/{args.prefix}.{threshold}.anno.tsv')
                logger.info(f'Completed, costed {round(time.time()-t_anno,2)} secs\n')
            else:
                logger.info(f'{args.out}/{args.prefix} is an unkwown file\n')
    Parallel(n_jobs=args.thread,backend='loky')(delayed(GWASplot)(i) for i in args.file)
    lt = time.localtime()
    endinfo = f'\nFinished, Total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
    logger.info(endinfo)