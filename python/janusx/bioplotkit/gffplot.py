import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class CHRPLOT:
    def __init__(self,gff:pd.DataFrame):
        self.gff = self._gffreader(gff) if os.path.isfile(gff) else gff
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
        cd = dict(zip(chr_list,['grey' for grey in chr_list]))
        cd.update(color_dict) if color_dict is not None else None
        for i in chr_list:
            ax.fill_betweenx(_.loc[_[0]==i,1],_.loc[_[0]==i,'l'],_.loc[_[0]==i,'r'],color=cd[i])
        ax.set_xticks(chr_list.astype(int),[f'chr{i}' for  i in chr_list])
        return ax
    def _gffreader(self,gffpath:str):
        gff = pd.read_csv(gffpath,sep='\t',comment='#',header=None)
        gff[0] = gff[0].str.lower().str.replace('chr0','').str.replace('chr','')
        gff = gff[gff[0].isin(np.arange(1,100).astype(str))]
        gff.loc[:,0] = gff[0].astype(int)
        gff = gff.sort_values([0,3])
        return gff


if __name__ == "__main__":
    print('Debug...\n')
    gffpath = r"D:\genome\ID09.gff3"
    fig = plt.figure(figsize=(6,4),dpi=300)
    ax = fig.add_subplot(111)
    plotmodel = CHRPLOT(gffpath)
    plotmodel.plotchr(color_dict={1:'red',4:'red',7:'red',},ax=ax)
    fig.tight_layout()
    fig.savefig('test.png')
