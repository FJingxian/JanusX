from _readanno import readanno
from bioplotkit import GWASPLOT
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import time
import socket
import logging
import sys
import os
import matplotlib as mpl
'''
Examples:
  # Basic usage with gwasplot
  --file result.assoc.txt
  # Usage of different column names with gwasplot
  --file result.assoc.txt --chr "chr" --pos "pos" --pvalue "P_wald"
  # Usage of gwasplot and point out output path
  --file result.assoc.txt --chr "chr" --pos "pos" --pvalue "P_wald" --out test # it will saved in test.pdf

File Formats:
  file:    result.assoc.txt (txt with tab as delimiter)
        
Citation:
  https://github.com/MaizeMan-JxFU/gwasplot/
'''
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
        epilog=__doc__,
    )
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    required_group.add_argument('--file', type=str, required=True,
                               help='File of gwas results')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('--chr', type=str, default='#CHROM',
                               help='Column name of chr'
                                   '(default: %(default)s)')
    optional_group.add_argument('--pos', type=str, default='POS',
                               help='Column name of position'
                                   '(default: %(default)s)')
    optional_group.add_argument('--pvalue', type=str, default='p',
                               help='Column name of pvalue'
                                   '(default: %(default)s)')
    optional_group.add_argument('--threshold', type=float, default=None,
                               help='treshold of pvalue'
                                   '(default: %(default)s)')
    optional_group.add_argument('--plot', action='store_true', default=True,
                               help='plot manhanden figure (default: %(default)s)')
    optional_group.add_argument('--noplot', action='store_false', dest='plot',
                               help='do not plot manhanden figure ')
    optional_group.add_argument('--anno', type=str, default=None,
                               help='annotation option, .gff file or .bed file'
                                   '(default: %(default)s)')
    optional_group.add_argument('--annobroaden', type=float, default=None,
                               help='broaden range of chromosome (Kb)'
                                   '(default: %(default)s)')
    optional_group.add_argument('--descItem', type=str, default='description',
                               help='description items in gff file (hidden option)'
                                   '(default: %(default)s)')
    optional_group.add_argument('--out', type=str, default=None,
                               help='Output prefix path'
                                   '(default: %(default)s)')
    args = parser.parse_args()
    # create log file
    args.out = '.'.join(args.file.split('.')[:-1]) if args.out is None else args.out
    folder = os.path.dirname(args.out)
    folder = '.' if folder == '' else folder
    # Create output directory if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder, mode=0o755)
    logger = setup_logging(f'''{args.out}.gwasplot.log'''.replace('//','/'))
    logger.info('Simple script of GWAS visualization')
    logger.info(f'Host: {socket.gethostname()}\n')
    # Build argument list for the original script
    sys.argv = [
        sys.argv[0],  # script name
        args.file,
        args.chr,
        args.pos,
        args.pvalue,
        args.threshold,
        args.out,
        args.anno,
        args.annobroaden,
        args.descItem,
        str(args.plot)
    ]
    # Print configuration summary
    if log:
        logger.info("*"*60)
        logger.info("GWAS plot script")
        logger.info("*"*60)
        logger.info(f"file:          {args.file}")
        logger.info(f"chr:           {args.chr}")
        logger.info(f"pos:           {args.pos}")
        logger.info(f"pvalue:        {args.pvalue}")
        logger.info(f"threshold:     {args.threshold}")
        logger.info(f"plot mode:     {args.plot}")
        logger.info(f"output prefix: {args.out}")
        logger.info(f"annotation:    {args.anno}")
        logger.info(f"annobroad(kb): {args.annobroaden}")
        logger.info("*"*60 + "\n")
    return args,logger

t_start = time.time()
mpl.use('Agg')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
args,logger = main()
file = args.file
chr_string,pos_string,pvalue_string = args.chr,args.pos,args.pvalue
df = pd.read_csv(file,sep='\t',usecols=[chr_string,pos_string,pvalue_string])
threshold = args.threshold if args.threshold is not None else 0.05/df.shape[0]
if args.plot:
    fig = plt.figure(figsize=(10,4),dpi=300)
    ax =fig.add_subplot(121,)
    ax2 =fig.add_subplot(122,)
    logger.info('* Visualizing...')
    plotmodel = GWASPLOT(df,chr_string,pos_string,pvalue_string,0.1)
    plotmodel.manhattan(-np.log10(threshold),ax=ax)
    plotmodel.qq(ax=ax2)
    plt.tight_layout()
    plt.savefig(f'{args.out}.pdf',transparent=True)
    plt.close()
    logger.info(f'Saved in {args.out}.pdf')
if args.anno is not None:
    if os.path.exists(args.anno):
        df_filter = df.loc[df[pvalue_string]<=threshold,[chr_string,pos_string,pvalue_string]].set_index([chr_string,pos_string])
        logger.info('* Annotating...')
        anno = readanno(args.anno,args.descItem) # After treating: anno 0-chr,1-start,2-end,3-geneID,4-description1,5-description2
        desc = list(map(lambda x:anno.loc[(anno[0]==x[0])&(anno[1]<=x[1])&(anno[2]>=x[1])], df_filter.index))
        df_filter['desc'] = list(map(lambda x:f'''{x.iloc[0,3]};{x.iloc[0,4]};{x.iloc[0,5]}''' if not x.empty else 'NA;NA;NA', desc))
        if args.annobroaden is not None:
            desc = list(map(lambda x:anno.loc[(anno[0]==x[0])&(anno[1]<=x[1]+args.annobroaden*1_000)&(anno[2]>=x[1]-args.annobroaden*1_000)], df_filter.index))
            df_filter['broaden'] = list(map(lambda x:f'''{'|'.join(x.iloc[:,3])};{'|'.join(x.iloc[:,4])};{'|'.join(x.iloc[:,5])}''' if not x.empty else 'NA;NA;NA', desc))
        logger.info(df_filter)
        df_filter.to_csv(f'{args.out}.{threshold}.anno.tsv',sep='\t')
        logger.info(f'Saved in {args.out}.{threshold}.anno.tsv')
    else:
        logger.info(f'{args.anno} is an unkwown file')
lt = time.localtime()
endinfo = f'\nFinished, Total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
logger.info(endinfo)