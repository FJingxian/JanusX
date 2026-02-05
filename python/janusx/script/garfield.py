import os
from pathlib import Path
import pandas as pd
from janusx.garfield.garfield2 import main

if __name__ == "__main__":
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
    if not os.path.isfile(bedfile):
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
    results = main(bfile,sampleid,y,bimranges,4,5,50)
    print(results)
