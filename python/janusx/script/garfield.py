from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import freeze_support
from tqdm import tqdm
from janusx.garfield.garfield2 import iter_main
from janusx.gfreader import SiteInfo,save_genotype_streaming

def main(bfile,y,sampleid,bimranges,threads=1):
    SNPIter = iter_main(bfile, y, sampleid, bimranges, threads=threads)
    Sites2Genofile = []
    G2Genofile = []
    pbar = tqdm(total=len(bimranges), desc="Processing regions", mininterval=0, miniters=1)
    try:
        for res in SNPIter:
            pbar.update(1)
            pbar.refresh()
            if res is None:
                continue
            g, site = res
            Sites2Genofile.append(SiteInfo(*site))
            G2Genofile.append(g)
            if len(Sites2Genofile) > 100:
                yield np.concatenate(G2Genofile,axis=0),Sites2Genofile
                Sites2Genofile = []
                G2Genofile = []
    finally:
        pbar.close()
        yield np.concatenate(G2Genofile,axis=0),Sites2Genofile

if __name__ == "__main__":
    freeze_support()
    bfile = "./test/mouse_hs1940"
    phenofile = "./test/mouse_hs1940.pheno"
    bedfile = ""
    # 读 phenotype
    pheno = pd.read_csv(phenofile, sep="\t", header=None, index_col=0)
    if pheno.size == 0:
        # 尝试空格分隔
        pheno = pd.read_csv(phenofile, sep=" ", header=None, index_col=0)
        if pheno.size == 0:
            raise ValueError(f"Error in Phenotype file:\n{phenofile}")

    if all(pheno.index == pheno[1]):
        # PLINK 格式: FID IID PHENO...
        pheno = pheno.iloc[:, 1:]
    else:
        # 普通 Excel：第一行是表头
        pheno.columns = pheno.iloc[0, :]
        pheno = pheno.iloc[1:, :]
    pheno = pheno.iloc[:, 0].dropna().to_frame()
    y = pheno.values  # (n,1)
    sampleid = pheno.index.to_list()
    if not Path(bedfile).exists():
        bim = pd.read_csv(f'{bfile}.bim',sep=r'\s+',index_col=None,header=None)
        step = 1_000_000
        window = 5_000_000
        bimranges = []
        for chrom in bim[0].unique():
            bimchr:pd.DataFrame = bim.loc[bim[0]==chrom]
            chrstart,chrend = bimchr[3].min(),bimchr[3].max()
            for loc in range(chrstart,chrend+step,step):
                loc = min(loc,chrend)
                bimranges.append((str(chrom),loc-window,loc+window))
    else:
        bimranges = []
    chunks = main(bfile,y,sampleid,bimranges,1)
    save_genotype_streaming('test/garfield.vcf',sample_ids=sampleid,chunks=chunks,)
