from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import trange
from janusx.gtools.reader import GFFQuery, gffreader

pathlike = Union[str, Path]
class CHRPLOT:
    def __init__(self,gff:Union[pd.DataFrame,pathlike]):
        if isinstance(gff, (str, Path)):
            gff = Path(gff)
            assert gff.exists(),"gff file not exists"
            self.gff = self._gffreader(gff)
        else:
            self.gff = gff
        pass
    def plotchr(self,window=1_000_000,color_dict:dict=None,ax:plt.Axes=None):
        _ = []
        chr_list = self.gff[0].unique()
        for i in chr_list:
            gff_chr = self.gff[(self.gff[0]==i)&(self.gff[2]=='gene')]
            chrend = gff_chr[4].max()
            locr = list(range(0,chrend,window))
            for loc in locr:
                gene_num = gff_chr[(gff_chr[3]>=loc)&(gff_chr[4]<=loc+window)].shape[0]
                _.append([i, loc, gene_num])
        _ = pd.DataFrame(_)
        _[2] = _[2]/_[2].max()/2
        _['l'] = _[0] - _[2]
        _['r'] = _[0] + _[2]
        cd = dict(zip(chr_list,['grey' for _ in chr_list]))
        cd.update(color_dict) if color_dict is not None else None
        for i in chr_list:
            ax.fill_betweenx(_.loc[_[0]==i,1],_.loc[_[0]==i,'l'],_.loc[_[0]==i,'r'],color=cd[i])
        ax.set_xticks(chr_list.astype(int),[f'chr{i}' for  i in chr_list])
        return ax
    def _gffreader(self,gffpath:str):
        gff = gffreader(gffpath)
        chr_num = pd.to_numeric(gff["chrom_norm"], errors="coerce")
        gff = gff[chr_num.notna()].copy()
        gff.loc[:, 0] = chr_num.loc[gff.index].astype(int)
        gff.loc[:, 2] = gff["feature"].astype(str)
        gff.loc[:, 3] = gff["start"].astype(int)
        gff.loc[:, 4] = gff["end"].astype(int)
        gff = gff[gff[0].isin(np.arange(1,100).astype(int))]
        gff = gff.sort_values([0, 3])
        return gff[[0, 2, 3, 4]]


if __name__ == "__main__":
    q = GFFQuery.from_file('/Users/jingxianfu/Public/Triticum_aestivum.IWGSC.62.mapped.gff3.gz')
    print(np.array(q.query_bimrange("1:80-100", features=["gene","CDS"],attr=['ID'])['attribute'].tolist()))
