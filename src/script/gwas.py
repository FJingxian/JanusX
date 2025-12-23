# -*- coding: utf-8 -*-
'''
Examples:
  # Basic usage with VCF file
  -vcf genotypes.vcf -p phenotypes.txt -o results
  
  # Using PLINK binary files with custom parameters
  -bfile genotypes -p phenotypes.txt -o results -k 1 -q 3 --thread 8
  
  # Using external kinship matrix and enabling fast mode
  -vcf genotypes.vcf -p phenotypes.txt -o results -k kinship_matrix.txt -qc 10 -fast
  
  # Maximum performance with one thread
  --bfile genotypes --pheno phenotypes.txt --out results --grm 1 --qcov 3 --cov covfile.txt --thread 1

File Formats:
  VCF/BFILE:    Standard VCF or PLINK binary format (bed/bim/fam)
  PHENO:        Tab-delimited file with sample IDs in first column and phenotypes in subsequent columns
  GRM File:     Space/tab-delimited kinship matrix file
  QCOV File:    Space/tab-delimited covariate matrix file
  COV File:    Space/tab-delimited covariate matrix file
        
Citation:
  https://github.com/MaizeMan-JxFU/JanusX/
'''
import os
for key in ['MPLBACKEND']:
    if key in os.environ:
        del os.environ[key]
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from bioplotkit import GWASPLOT
from pyBLUP import GWAS,LM,farmcpu
from pyBLUP import QK
from gfreader import breader,vcfreader,npyreader
from joblib import cpu_count
import pandas as pd
import numpy as np
import argparse
import time
import socket
from ._common.log import setup_logging
from rust2py.gfreader import load_genotype_chunks,inspect_genotype_file
from pyBLUP import slim
import psutil
from tqdm import tqdm
  
def fastplot(gwasresult:pd.DataFrame,phenosub:pd.DataFrame,xlabel:str='',outpdf:str='fastplot.pdf'):
    '''Fast plot for GWASresult data'''
    results = gwasresult.astype({"POS": "int64"})
    fig = plt.figure(figsize=(16,4),dpi=300)
    layout = [['A','B','B','C']]
    axes:dict = fig.subplot_mosaic(mosaic=layout)
    gwasplot = GWASPLOT(results)
    axes['A'].hist(phenosub,bins=15)
    axes['A'].set_xlabel(xlabel)
    axes['A'].set_ylabel('count')
    gwasplot.manhattan(-np.log10(1/results.shape[0]),ax=axes['B'])
    gwasplot.qq(ax=axes['C'])
    plt.tight_layout()
    plt.savefig(f"{outpdf}",transparent=True)

def lowm_GWAS(genofile:os.PathLike, pheno_file:os.PathLike, pheno_col:int,outprefix:str,
              maf_threshold:float=0.01,max_missing_rate:float=0.05, mem:float=None,plot: bool=False,
              mgrm:str='1',pcdim:str='10',model:str='lmm',threads:int=4):
    pheno = pd.read_csv(rf'{pheno_file}',sep='\t') # Col 1 - idv ID; row 1 - pheno tag
    pheno = pheno.groupby(pheno.columns[0]).mean() # Mean of duplicated samples
    pheno.index = pheno.index.astype(str)
    assert pheno.shape[1]>0, f'No phenotype data found, please check the phenotype file format!\n{pheno.head()}'
    if pheno_col is not None: 
        assert np.min(pheno_col) <= pheno.shape[1], "Phenotype column index out of range."
        pheno_col = [i for i in pheno_col if i in range(pheno.shape[1])]
        pheno = pheno.iloc[:,pheno_col]
    modeldict = {'lmm':slim.LMM,'lm':slim.LM}
    gwasmodel = modeldict[model]
    process = psutil.Process()
    method = str(mgrm)  # 1: VanRaden; 2: Yang et al.
    dim = str(pcdim)
    ids, m = inspect_genotype_file(genofile)
    ids = np.array(ids).astype(str)
    n = len(ids)
    mem = psutil.virtual_memory().available if mem is None else mem
    chunk_size = int(mem*0.8/(4*n))
    # Process of Calculating Kinship & PCA
    grm = np.zeros((n,n),dtype='float32')
    if mgrm in ['1','2']:
        method = int(mgrm)
        pbar = tqdm(total=m, desc="grm&pca",ascii=True)
        varsum = 0
        num = 0
        for genosub,sites in load_genotype_chunks(genofile,chunk_size,maf_threshold,max_missing_rate):
            genosub:np.ndarray = genosub
            maf = genosub.mean(axis=1,dtype='float32',keepdims=True)/2
            genosub = genosub - 2*maf
            if method == 1:
                grm += genosub.T @ genosub
                varsum += np.sum(2*maf*(1-maf))
            elif method == 2:
                grm += 1/(2*maf*(1-maf)) * genosub.T @ genosub
            num += genosub.shape[0]
            pbar.update(genosub.shape[0])
            if num % 10*chunk_size == 0:
                memory_usage = process.memory_info().rss / 1024**3
                pbar.set_postfix(memory=f'{memory_usage:.2f} GB')
        if method == 1:
            grm = (grm + grm.T) / varsum / 2
        elif method == 2:
            grm = (grm + grm.T) / num / 2
        m = num # refresh nsnp
        pbar.close()
    elif os.path.isfile(mgrm):
        grm:np.ndarray = np.genfromtxt(mgrm,dtype='float32')
        assert grm.size == n*n, f'GRM file size not match: expected {n*n}, got {grm.size}'
    else:
        raise ValueError(f'Unknown GRM option: {mgrm}')
    if pcdim in np.arange(1,n).astype(str):
        dim = int(pcdim)
        eigval,eigvec = np.linalg.eigh(grm)
        idx = np.argsort(eigval)[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        eigval = eigval[:dim]
        eigvec = eigvec[:, :dim]
    elif pcdim == '0':
        eigvec = np.array([],dtype='float32').reshape(n,0)
    elif os.path.isfile(pcdim):
        eigvec:np.ndarray = np.genfromtxt(pcdim,dtype='float32')
        assert eigvec.shape[0] == n, f'PCA file size not match: expected {n}, got {eigvec.shape[0]}'
    else:
        raise ValueError(f'Unknown PCA option: {pcdim}')
    for p in pheno.columns:
        pheno_sub:pd.DataFrame = pheno[p].dropna()
        sameidx = np.isin(ids,pheno_sub.index)
        pheno_sub = pheno_sub.loc[ids[sameidx]].values
        results = []
        maf = []
        c_p_ref_alt = []
        mod = gwasmodel(y=pheno_sub,X=eigvec[sameidx],kinship=grm[sameidx,:][:,sameidx]) if model == 'lmm' else gwasmodel(y=pheno_sub,X=eigvec[sameidx])
        num = 0
        # Process of GWAS
        print('*'*60)
        if model == 'lmm':
            print(f'''Number of samples: {np.sum(sameidx)}, Number of SNP: {m}, pve of null: {round(mod.pve,3)}''')
        pbar = tqdm(total=m, desc=f"{model}",ascii=True)
        for genosub, sites in load_genotype_chunks(genofile,chunk_size,maf_threshold,max_missing_rate):
            genosub:np.ndarray = genosub[:,sameidx]
            maf.extend(np.mean(genosub,axis=1)/2)
            results.append(mod.gwas(genosub,threads=threads))
            c_p_ref_alt.extend([[i.chrom,i.pos,i.ref_allele,i.alt_allele] for i in sites])
            pbar.update(genosub.shape[0])
            num += genosub.shape[0]
            if num % 10*chunk_size == 0:
                memory_usage = process.memory_info().rss / 1024**3
                pbar.set_postfix(memory=f'{memory_usage:.2f} GB')
        pbar.close()
        results = np.concatenate(results,axis=0)
        c_p_ref_alt = np.array(c_p_ref_alt)
        results = pd.DataFrame(np.concatenate([c_p_ref_alt, results, np.array(maf).reshape(-1,1)],axis=1),columns=['#CHROM','POS','REF','ALT','beta','se','p','maf'])
        results = results[['#CHROM','POS','REF','ALT','maf','beta','se','p']]
        results = results.astype({'POS':int,'maf':float,'beta':float,'se':float,'p':float})
        fastplot(results,pheno_sub,xlabel=p,outpdf=f"{outprefix}.{p}.{model}.pdf") if plot else None
        results = results.astype({"p": "object"})
        results.loc[:,'p'] = results['p'].map(lambda x: f"{x:.4e}")
        results.to_csv(f"{outprefix}.{p}.{model}.tsv",sep="\t",float_format="%.4f",index=None)
        print(f'Saved in {outprefix}.{p}.{model}.tsv'.replace('//','/'))

def main(log:bool=True):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    t_start = time.time()
    # Required arguments
    required_group = parser.add_argument_group('Required arguments')
    ## Genotype file
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument('-vcf','--vcf', type=str, 
                           help='Input genotype file in VCF format (.vcf or .vcf.gz)')
    geno_group.add_argument('-bfile','--bfile', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .bed, .bim, .fam)')
    geno_group.add_argument('-npy','--npy', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .npz, .snp, .idv)')
    ## Phenotype file
    required_group.add_argument('-p','--pheno', type=str, required=True,
                               help='Phenotype file (tab-delimited with sample IDs in first column)')
    ## Model
    models_group = parser.add_argument_group('Model Arguments')
    models_group.add_argument('-lmm','--lmm', action='store_true',default=False,
                               help='Linear mixed model '
                                   '(default: %(default)s)')
    models_group.add_argument('-farmcpu','--farmcpu', action='store_true',default=False,
                               help='FarmCPU model '
                                   '(default: %(default)s)')
    models_group.add_argument('-lm','--lm', action='store_true',default=False,
                               help='General linear model '
                                   '(default: %(default)s)')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    ## Point out phenotype or snp
    optional_group.add_argument('-n','--ncol', action='extend', nargs='*',default=None,type=int,
                               help='Analyed phenotype column, eg. "-n 0 -n 3" is to analyze phenotype 1 and phenetype 4 '
                                   '(default: %(default)s)')
    optional_group.add_argument('-cl','--chrloc', type=str, default=None,
                               help='Only analysis ranged SNP, eg. 1:1000000:3000000 '
                                   '(default: %(default)s)')
    ## More detail arguments
    optional_group.add_argument('-lmem','--lmem', action='store_true', default=False,
                               help='Low memory cost mode '
                                   '(default: %(default)s)')
    optional_group.add_argument('-setmem','--setmem', type=float, default=None,
                               help='Memory limitation for all task '
                                   '(default: %(default)s)')
    optional_group.add_argument('-k','--grm', type=str, default='1',
                               help='Kinship matrix calculation method [1-centralization or 2-standardization] or path to pre-calculated GRM file '
                                   '(default: %(default)s)')
    optional_group.add_argument('-q','--qcov', type=str, default='0',
                               help='Number of principal components for Q matrix or path to covariate matrix file '
                                   '(default: %(default)s)')
    optional_group.add_argument('-c','--cov', type=str, default=None,
                               help='Path to covariance file '
                                   '(default: %(default)s)')
    optional_group.add_argument('-d','--dom', action='store_true', default=False,
                               help='Estimate dominance effects '
                                   '(default: %(default)s)')
    optional_group.add_argument('-csnp','--csnp', type=str, default=None,
                               help='Control snp for conditional GWAS, eg. 1:1200000 '
                                   '(default: %(default)s)')
    optional_group.add_argument('-plot','--plot', action='store_true', default=False,
                               help='Visualization of GWAS result '
                                   '(default: %(default)s)')
    optional_group.add_argument('-t','--thread', type=int, default=-1,
                               help='Number of CPU threads to use (-1 for all available cores, default: %(default)s)')
    optional_group.add_argument('-o', '--out', type=str, default='.',
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
    threads = cpu_count() if args.thread<=0 else args.thread
    mem = 1024**3*args.setmem if args.setmem is not None else psutil.virtual_memory().available
    gfile = gfile.replace('\\','/') # adjust \ in Windows
    # create output folder and log file
    os.makedirs(args.out,0o755,exist_ok=True)
    logger = setup_logging(f'''{args.out}/{args.prefix}.gwas.log'''.replace('\\','/').replace('//','/'))
    logger.info('High Performance Linear Mixed Model Solver for Genome-Wide Association Studies')
    logger.info(f'Host: {socket.gethostname()}\n')
    if log:
        logger.info("*"*60)
        logger.info("GWAS LMM SOLVER CONFIGURATION")
        logger.info("*"*60)
        logger.info(f"Genotype file:    {gfile}")
        logger.info(f"Phenotype file:   {args.pheno}")
        logger.info(f"Analysis nSNP:    {args.chrloc}") if args.chrloc is not None else logger.info(f"Analysis nSNP:    All")
        logger.info(f"Analysis Pcol:    {args.ncol}") if args.ncol is not None else logger.info(f"Analysis Pcol:    All")
        if args.lm:
            logger.info("Estimate Model:   General Linear model")
        if args.lmm:
            logger.info("Estimate Model:   Mixed Linear model")
        if args.farmcpu:
            logger.info("Estimate Model:   FarmCPU")
        logger.info(f"Low memory mode:  {args.ncol}")
        if args.dom: # Dominance model
            logger.info(f"Dominance model:  {args.dom}")
        if args.csnp: # Conditional GWAS
            logger.info(f"Conditional SNP:  {args.csnp}")
        logger.info(f"Estimate of GRM:  {args.grm}")
        if args.qcov != '0':
            logger.info(f"Q matrix:         {args.qcov}")
        if args.cov:
            logger.info(f"Covariant matrix: {args.cov}")
        logger.info(f"Threads:          {threads} ({cpu_count()} available)")
        logger.info(f"Output prefix:    {args.out}/{args.prefix}")
        logger.info("*"*60 + "\n")
    try:
        phenofile,outfolder = args.pheno,args.out
        kinship_method = args.grm
        qdim = args.qcov
        cov = args.cov
        kcal = True if kinship_method in ['1','2'] else False
        qcal = True if qdim in np.arange(0,30).astype(str) else False
        # test exist of all input files
        assert os.path.isfile(phenofile), f"can not find file {phenofile}"
        # test k and q matrix
        assert kcal or os.path.isfile(kinship_method), f'{kinship_method} is not a calculation method or grm file'
        assert qcal or os.path.isfile(qdim), f'{qdim} is not a dimension of PC or PC file'
        assert cov is None or os.path.isfile(cov), f"{cov} is applied, but it is not a file"
        # test exist of calculating model
        assert args.lm or args.lmm or args.farmcpu, 'no model to estimate, try -lm, -farmcpu or -lmm'

        # Loading genotype matrix
        t_loading = time.time()
        if args.lmem:
            if args.farmcpu:
                print('Low mem mode do not support FarmCPU, it will be ignored...')
            if args.lmm:
                results = lowm_GWAS(gfile,phenofile,args.ncol,f'{args.out}/{args.prefix}',mgrm=args.grm,pcdim=args.qcov,model='lmm',threads=threads,mem=mem,plot=args.plot)
            if args.lm:
                results = lowm_GWAS(gfile,phenofile,args.ncol,f'{args.out}/{args.prefix}',mgrm=args.grm,pcdim=args.qcov,model='lm',threads=threads,mem=mem,plot=args.plot)
        else:
            logger.info('* Loading genotype and phenotype')
            if not args.npy:
                logger.info('Recommended: Use numpy format of genotype matrix (just use gformat module to transfer)')
            logger.info(f'** Loading phenotype from {phenofile}...')
            pheno = pd.read_csv(rf'{phenofile}',sep='\t') # Col 1 - idv ID; row 1 - pheno tag
            pheno = pheno.groupby(pheno.columns[0]).mean() # Mean of duplicated samples
            pheno.index = pheno.index.astype(str)
            assert pheno.shape[1]>0, f'No phenotype data found, please check the phenotype file format!\n{pheno.head()}'
            if args.ncol is not None: 
                assert np.min(args.ncol) <= pheno.shape[1], "Phenotype column index out of range."
                args.ncol = [i for i in args.ncol if i in range(pheno.shape[1])]
                logger.info(f'''These phenotype will be analyzed: {'\t'.join(pheno.columns[args.ncol])}''',)
                pheno = pheno.iloc[:,args.ncol]
            if args.vcf:
                logger.info(f'** Loading genotype from {gfile}...')
                geno = vcfreader(rf'{gfile}') # VCF format
            elif args.bfile:
                logger.info(f'Loading genotype from {gfile}.bed...')
                geno = breader(rf'{gfile}') # PLINK format
            elif args.npy:
                logger.info(f'Loading genotype from {gfile}.npz...')
                geno = npyreader(rf'{gfile}') # numpy format
            ref_alt:pd.DataFrame = geno.iloc[:,:2]
            famid = geno.columns[2:].values.astype(str)
            geno = geno.iloc[:,2:].to_numpy(copy=False)
            logger.info(f'Geno and Pheno are ready, costed {(time.time()-t_loading):.2f} secs')

            # GRM & PCA
            t_control = time.time()
            logger.info('* Filter SNPs with MAF < 0.01 or missing rate > 0.05; impute with mode...')
            logger.info('Recommended: Use genotype matrix imputed by beagle or impute2 as input')
            qkmodel = QK(geno,maff=0.01)
            logger.info(f'Filter finished, costed {(time.time()-t_control):.2f} secs')
            geno = qkmodel.M
            if args.dom: # Additive kinship but dominant single SNP
                logger.info('* Transfer additive gmatrix to dominance gmatrix')
                np.subtract(geno,1,out=geno)
                np.absolute(geno, out=geno)
            ref_alt = ref_alt.loc[qkmodel.SNPretain]
            ref_alt.iloc[qkmodel.maftmark,[0,1]] = ref_alt.iloc[qkmodel.maftmark,[1,0]]
            ref_alt['maf'] = qkmodel.maf

            if args.chrloc:
                chr_loc = np.array(args.chrloc.split(':'),dtype=np.int32)
                chr,start,end = chr_loc[0],np.min(chr_loc[1:]),np.max(chr_loc[1:])
                onlySNP = ref_alt.index.to_frame().values
                filt1 = onlySNP[:,0].astype(str)==str(chr)
                filt2 = (onlySNP[filt1,1]<=end) & (onlySNP[filt1,1]>=start)
                if start == 0 and end == 0:
                    geno = geno[filt1]
                    ref_alt = ref_alt.loc[filt1]
                else:
                    geno = geno[filt1][filt2]
                    ref_alt = ref_alt.loc[filt1].loc[filt2]

            assert geno.size>0, 'After filtering, number of SNP is 0'

            prefix = gfile.replace('.vcf','').replace('.gz','')
            if args.lmm:
                logger.info(f'* Preparing GRM and Q matrix for LMM...')
                if kcal:
                    if os.path.exists(f'{prefix}.k.{kinship_method}.txt'):
                        logger.info(f'* Loading GRM from {prefix}.k.{kinship_method}.txt...')
                        kmatrix = np.genfromtxt(f'{prefix}.k.{kinship_method}.txt')
                    else:    
                        logger.info(f'* Calculation method of kinship matrix is {kinship_method}')
                        kmatrix = qkmodel.GRM(method=int(kinship_method))
                        np.savetxt(f'{prefix}.k.{kinship_method}.txt',kmatrix,fmt='%.6f')
                else:
                    logger.info(f'* Loading GRM from {kinship_method}...')
                    kmatrix = np.genfromtxt(kinship_method) if kinship_method[-4:] != '.npz' else np.load(kinship_method,)['arr_0']
            if qcal:
                if os.path.exists(f'{prefix}.q.{qdim}.txt'):
                    logger.info(f'* Loading Q matrix from {prefix}.q.{qdim}.txt...')
                    qmatrix = np.genfromtxt(f'{prefix}.q.{qdim}.txt')
                elif qdim=="0":
                    qmatrix = np.array([]).reshape(geno.shape[1],0)
                else:
                    logger.info(f'* Dimension of PC for q matrix is {qdim}')
                    qmatrix,eigenval = qkmodel.PCA()
                    qmatrix = qmatrix[:,:int(qdim)]
                    np.savetxt(f'{prefix}.q.{qdim}.txt',qmatrix,fmt='%.6f')       
            else:
                logger.info(f'* Loading Q matrix from {qdim}...')
                qmatrix = np.genfromtxt(qdim)
            if cov:
                cov = np.genfromtxt(cov,).reshape(-1,1)
                logger.info(f'Covmatrix {cov.shape}:')
                qmatrix = np.concatenate([qmatrix,cov],axis=1)
            if args.csnp:
                logger.info(f'* Use SNP in {args.csnp} as control for conditional GWAS')
                chr_loc_index = ref_alt.reset_index().iloc[:,:2].astype(str)
                chr_loc_index = pd.Index(chr_loc_index.iloc[:,0]+':'+chr_loc_index.iloc[:,1])
                cov = geno[chr_loc_index.get_loc(args.csnp)].reshape(-1,1)
                logger.info(f'Covmatrix {cov.shape}:')
                qmatrix = np.concatenate([qmatrix,cov],axis=1)
            if args.lmm:
                logger.info(f'GRM {str(kmatrix.shape)}:')
                logger.info(kmatrix[:5,:5])
            logger.info(f'Qmatrix {str(qmatrix.shape)}:')
            logger.info(qmatrix[:5,:5])
            del qkmodel
            # GWAS
            for i in pheno.columns:
                t = time.time()
                logger.info('*'*60)
                logger.info(f'* GWAS process for {i}')
                p = pheno[i].dropna()
                famidretain = np.isin(famid,p.index)
                snp_sub = geno[:,famidretain]
                p_sub = p.loc[famid[famidretain]].values.reshape(-1,1)
                q_sub = qmatrix[famidretain]
                if args.lmm:
                    k_sub = kmatrix[famidretain][:,famidretain]
                if len(p)>0:
                    if args.lmm:
                        logger.info(f'** Mixed Linear Model:')
                        gwasmodel = GWAS(y=p_sub,X=q_sub,kinship=k_sub)
                        logger.info(f'''Number of samples: {np.sum(famidretain)}, Number of SNP: {geno.shape[0]}, pve of null: {round(gwasmodel.pve,3)}''')
                        results = gwasmodel.gwas(snp=snp_sub,chunksize=100_000,threads=threads) # gwas running...
                        results = pd.DataFrame(results,columns=['beta','se','p'],index=ref_alt.index)
                        results = pd.concat([ref_alt,results],axis=1)
                        results = results.reset_index().dropna()
                        logger.info(f'Effective number of SNP: {results.shape[0]}')
                        fastplot(results,p_sub,xlabel=i,outpdf=f"{outfolder}/{args.prefix}.{i}.lmm.pdf") if args.plot else None
                        results = results.astype({"p": "object"})
                        results.loc[:,'p'] = results['p'].map(lambda x: f"{x:.4e}")
                        results.to_csv(f"{outfolder}/{args.prefix}.{i}.lmm.tsv",sep="\t",float_format="%.4f",index=False)
                        logger.info(f'Saved in {outfolder}/{args.prefix}.{i}.lmm.tsv'.replace('//','/'))
                    if args.lm:
                        logger.info(f'** General Linear Model:')
                        gwasmodel = LM(y=p_sub,X=q_sub)
                        results = gwasmodel.gwas(snp=snp_sub,chunksize=100_000,threads=threads) # gwas running...
                        results = pd.DataFrame(results,columns=['beta','se','p'],index=ref_alt.index)
                        results = pd.concat([ref_alt,results],axis=1)
                        results = results.reset_index().dropna()
                        fastplot(results,p_sub,xlabel=i,outpdf=f"{outfolder}/{args.prefix}.{i}.lm.pdf") if args.plot else None
                        results = results.astype({"p": "object"})
                        results.loc[:,'p'] = results['p'].map(lambda x: f"{x:.4e}")
                        results.to_csv(f"{outfolder}/{args.prefix}.{i}.lm.tsv",sep="\t",float_format="%.4f",index=None)
                        logger.info(f'Saved in {outfolder}/{args.prefix}.{i}.lm.tsv'.replace('//','/'))
                    if args.farmcpu:
                        logger.info(f'** FarmCPU Model:')
                        results = farmcpu(y=p_sub,M=snp_sub,X=q_sub,chrlist=ref_alt.reset_index().iloc[:,0].values,poslist=ref_alt.reset_index().iloc[:,1].values,iter=20,threads=threads)
                        results = pd.DataFrame(results,columns=['beta','se','p'],index=ref_alt.index)
                        results = pd.concat([ref_alt,results],axis=1)
                        results = results.reset_index()
                        fastplot(results,p_sub,xlabel=i,outpdf=f"{outfolder}/{args.prefix}.{i}.farmcpu.pdf") if args.plot else None
                        results = results.astype({"p": "object"})
                        results.loc[:,'p'] = results['p'].map(lambda x: f"{x:.4e}")
                        results.to_csv(f"{outfolder}/{args.prefix}.{i}.farmcpu.tsv",sep="\t",float_format="%.4f",index=None)
                        logger.info(f'Saved in {outfolder}/{args.prefix}.{i}.farmcpu.tsv'.replace('//','/'))
                else:
                    logger.info(f'Phenotype {i} has no overlapping samples with genotype, please check sample id. skipped.\n')
                logger.info(f'Time costed: {round(time.time()-t,2)} secs\n')
    except Exception as e:
        logger.exception(f'Error of JanusX: {e}')
    lt = time.localtime()
    endinfo = f'\nFinished, Total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
    logger.info(endinfo)

if __name__ == "__main__":
    main()