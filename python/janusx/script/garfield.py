import os
import numpy as np
import pandas as pd
from janusx.garfield.garfield2 import main, window
from janusx.gfreader import SiteInfo
from janusx.gfreader.gfreader import save_genotype_streaming
from janusx.script._common.readanno import readanno

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
            f.write("chrom\tpos\texpression\tscore\n")
            for item in results:
                if item is None:
                    continue
                resdict:dict = item
                xcombine = np.asarray(resdict.get("xcombine", []), dtype=np.int8).ravel()
                if xcombine.size != n_samples:
                    raise ValueError("xcombine length does not match sample_ids")
                expr = resdict.get("expression", "")
                score = resdict.get("score", "")
                f.write(f"{chrom}\t{pos}\t{expr}\t{score}\n")
                geno = xcombine.reshape(1, -1)
                sites = [SiteInfo(str(chrom), int(pos), "A", "T")]
                yield geno, sites
                pos += 1

    save_genotype_streaming(outprefix, sample_ids, _iter_chunks())


if __name__ == "__main__":
    # bfile = "../Garfield/example/test.mouse1940"
    # phenofile = "../Garfield/example/test.mouse1940.trait.txt"
    # genefile = "../Garfield/example/test.mouse1940.geneset.txt"
    # gff3file = "../Garfield/example/test.mouse1940.gff3"
    bfile = "./test/mouse_hs1940"
    phenofile = "./test/mouse_hs1940.pheno.txt"
    genefile = "../Garfield/example/test.mouse1940.geneset.txtss"
    gff3file = "../Garfield/example/test.mouse1940.gff3sss"
    grmfile = "test/mouse_hs1940.k.1.npy"
    famid = pd.read_csv(f'{bfile}.fam',sep=r'\s+',index_col=None,header=None)[0].tolist()
    # 读 phenotype
    pheno = pd.read_csv(phenofile, sep="\t", header=None, index_col=0)
    step = 1_000_000
    extension = 5_000_000
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
    if os.path.isfile(genefile) and os.path.exists(gff3file): # gene or geneset mode
        dfgene = pd.read_csv(genefile,sep=r'\s+',index_col=None,header=None)
        dfgff3 = readanno(gff3file,'ID').iloc[:,:4].set_index(3)
        dupgenemask = dfgff3.index.duplicated()
        if any(dupgenemask):
            print('Duplicated Genes',np.unique(dfgff3.index[dupgenemask]))
        dfgff3 = dfgff3.loc[~dupgenemask]
        bimranges = []
        for i in dfgene.index:
            geneset = []
            for gene in dfgene.loc[i].values:
                if gene not in dfgff3.index:
                    print(f'Warning: {gene} not exists in gff3')
                    break
                geneset.append((dfgff3.loc[gene,0],dfgff3.loc[gene,1]-extension,dfgff3.loc[gene,2]+extension))
            bimranges.append(geneset)
        results = main(bfile,sampleid,y,bimranges,nsnp=5,n_estimators=50,threads=4)
    else: # window mode
        results = window(bfile, sampleid, y, step, extension, nsnp=5, n_estimators=50, threads=4)
    outprefix = "./test/example.garfield"
    # print(results)
    write_xcombine_results(outprefix, sampleid, results)
