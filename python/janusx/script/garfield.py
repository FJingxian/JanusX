import os
import numpy as np
import pandas as pd
from janusx.garfield.garfield2 import main, window
from janusx.gfreader import SiteInfo
from janusx.gfreader.gfreader import save_genotype_streaming


def write_xcombine_results(
    outprefix: str,
    sample_ids: list[str],
    results: list,
    *,
    chrom: str = "pseudo",
):
    """
    Save xcombine results to PLINK/VCF and write {outprefix}.garfield mapping.
    """
    map_path = f"{outprefix}.pseudo"
    n_samples = len(sample_ids)

    def _iter_chunks():
        pos = 1
        with open(map_path, "w", encoding="utf-8") as f:
            f.write("POS\tEXPRESSION\tSCORE\n")
            for item in results:
                if item is None:
                    continue
                resdict:dict = item
                xcombine = np.asarray(resdict.get("xcombine", []), dtype=np.int8).ravel()
                if xcombine.size != n_samples:
                    raise ValueError("xcombine length does not match sample_ids")
                expr = resdict.get("expression", "")
                score = resdict.get("score", "")
                f.write(f"{pos}\t{expr}\t{score}\n")
                geno = xcombine.reshape(1, -1)
                sites = [SiteInfo(str(chrom), int(pos), "A", "T")]
                yield geno, sites
                pos += 1

    save_genotype_streaming(outprefix, sample_ids, _iter_chunks())


if __name__ == "__main__":
    bfile = "./test/mouse_hs1940"
    phenofile = "./test/mouse_hs1940.pheno"
    bedfile = ""
    famid = pd.read_csv(f'{bfile}.fam',sep=r'\s+',index_col=None,header=None)[0].tolist()
    # 读 phenotype
    pheno = pd.read_csv(phenofile, sep="\t", header=None, index_col=0)
    step = 1_000_000
    windows = 5_000_000
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
    sampleid = pheno.index[np.isin(pheno.index.to_list(),famid)].tolist()
    y = pheno.loc[sampleid].values  # (n,1)
    if not os.path.isfile(bedfile):
        bim = pd.read_csv(f'{bfile}.bim',sep=r'\s+',index_col=None,header=None)
        bimranges = []
        for chrom in bim[0].unique():
            bimchr:pd.DataFrame = bim.loc[bim[0]==chrom]
            chrstart,chrend = bimchr[3].min(),bimchr[3].max()
            for loc in range(chrstart,chrend+step,step):
                loc = min(loc,chrend)
                bimranges.append((str(chrom),loc-windows,loc+windows))
    else:
        bimranges = []
    # results = main(bfile,sampleid,y,bimranges,nsnp=5,n_estimators=50,threads=4)
    results = window(bfile, sampleid, y, step, windows, nsnp=5, n_estimators=50, threads=4)
    outprefix = f"{bfile}.garfield"
    write_xcombine_results(outprefix, sampleid, results)
