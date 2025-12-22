import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import beta
import time

def ppoints(n)->np.ndarray:
    '''
    Expected Quantile of Uniform Distribution
    '''
    return np.arange(1,n+1)/(n+1) # (np.arange(1, n+1) - 0.5) / n

class GWASPLOT:
    def __init__(self,df:pd.DataFrame, chr:str='#CHROM', pos:str='POS', pvalue:str='p',interval_rate:int=.1,compression=True):
        '''
        :param df: DataFrame, GWAS result\n
        :param chr: str, colname of CHR\n
        :param pos: str, colname of POS\n
        :param pvalue: str, colname of Pvalue\n
        '''
        if compression: # DataFrame Compression down-sampling
            cnum = df.shape[0] // 100_000 if df.shape[0]>=100_000 else 1 # Final number of SNP to plot is 100_000
            cpvalue = 10_000/df.shape[0] # set compression threshold
            df_cp = df.loc[df.sort_values(pvalue,ascending=False).iloc[:cnum*int(np.sum(df[pvalue]>cpvalue)//cnum),:].index].sort_index() # 过滤大多数阈值线以下的位点
            minloc = np.argmin(df_cp[pvalue].values.reshape(int(df_cp.shape[0]/cnum),cnum),axis=1) # 每cnum个点中选取pvalue最小的点
            self.minidx = df_cp.iloc[[int(i+idx*cnum) for idx,i in enumerate(minloc)]].index.to_list() + df[df[pvalue]<=cpvalue].index.to_list() # 保留位点索引
        else:
            self.minidx = df.index.to_list()
        self.t_start = time.time()
        df = df[[chr, pos, pvalue]].copy()
        self.chruniq = df[chr].unique()
        transdict = dict(zip(self.chruniq,range(1,1+len(self.chruniq))))
        df[chr] = df[chr].map(transdict).astype(int)
        df = df.sort_values(by=[chr,pos])
        self.chrlist = df[chr].unique()
        self.interval = int(interval_rate*df[pos].max())
        df['x'] = df[pos]
        if len(self.chrlist)>1:
            for ii in self.chrlist:
                if ii > 1:
                    df.loc[df[chr]==ii,'x'] = df.loc[df[chr]==ii,'x']+df[df[chr]==ii-1]['x'].max()
        df['x'] = (df[chr]-1)*self.interval+df['x']
        df['y'] = df[pvalue]
        df['z'] = df[chr]
        self.ticks_loc = df.groupby('z')['x'].mean()
        self.df = df.set_index([chr,pos])
        pass
    def manhattan(self, threshold:float=None, color_set:list=[], ax:plt.Axes = None, ignore:list=[],**kwargs):
        df = self.df.iloc[self.minidx,-3:].copy()
        df['y'] = -np.log10(df['y'])
        df = df[df['y']>=0.5]
        if ax == None:
            fig = plt.figure(figsize=[12,6], dpi=300)
            gs = GridSpec(12, 1, figure=fig)
            ax = fig.add_subplot(gs[0:12,0])
        if len(color_set) == 0:
            color_set = ['black','grey']
        colors = dict(zip(self.chrlist,[color_set[i%len(color_set)] for i in range(len(self.chrlist))]))
        ax.scatter(df[~df.index.isin(ignore)]['x'], df[~df.index.isin(ignore)]['y'], s=8, color=df[~df.index.isin(ignore)]['z'].map(colors),**kwargs)
        if threshold != None and max(df['y'])>=threshold:
            df_annote = df[df['y']>=threshold]
            ax.scatter(df_annote[~df_annote.index.isin(ignore)]['x'], df_annote[~df_annote.index.isin(ignore)]['y'], s=16, color='red',**kwargs)
            ax.hlines(y=threshold, xmin=0, xmax=max(df['x']),color='grey', linewidth=1, linestyles='--')
        ax.set_xticks(self.ticks_loc, self.chruniq)
        ax.set_xlim([0-self.interval,max(df['x'])+self.interval])
        ax.set_ylim([0.5,max(df['y'])+0.1*max(df['y'])])
        ax.set_xlabel('Chromosome')
        ax.set_ylabel('-log10(p-value)')
        return ax
    def qq(self, ax:plt.Axes = None, ci:int=95, color_set:list=[]):
        '''
        可选: bootstrap 抽样次数, ci置信区间(分位数)
        '''
        if len(color_set) == 0:
            color_set = ['black','grey']
        df = self.df.copy()
        if ax == None:
            fig = plt.figure(figsize=[12,6], dpi=600)
            gs = GridSpec(12, 1, figure=fig)
            ax = fig.add_subplot(gs[0:12,0])
        p = df['y'].dropna()
        n = len(p)
        # 计算分位数
        o_e = p.sort_values().to_frame()
        o_e['e'] = ppoints(n) # 生成理论分位数
        o_e.columns = ['o','e']
        o_e:pd.DataFrame = -np.log10(o_e.sort_index())
        e_theoretical = o_e['e'].sort_values().values # 对于和性状不关联的位点，其pvalue相当于对均匀分布的随机抽样
        
        xi = np.ceil(10**(-e_theoretical)*n)
        lower = -np.log10(beta.ppf(1 - ci/100,xi,n-xi+1))
        upper = -np.log10(beta.ppf(ci/100,xi,n-xi+1))
        # 绘制置信区间
        ax.fill_between(e_theoretical, lower, upper, color=color_set[0], alpha=0.3,rasterized=True)
        
        # 绘制理论线（y=x）和观测点
        ax.plot([0, np.min(o_e.max(axis=0))], [0, np.min(o_e.max(axis=0))], lw=1,color='grey')
        ax.scatter(o_e.iloc[self.minidx,1], o_e.iloc[self.minidx,0], s=16, alpha=0.8,rasterized=True,color=color_set[0])
        ax.set_xlabel('Expected -log10(p-value)')
        ax.set_ylabel('Observed -log10(p-value)')
        return ax
    

def cal_PVE(df:pd.DataFrame, N:int, beta:str='beta', se_beta:str='se', maf:str='af', n_miss:str='n_miss'):
    '''
    基于gemma或其他程序获得的GWAS结果计算每个位点的pve值\n
    df: gwas结果矩阵, header 需包含 beta(效应值), se of beta(效应值标准误), MAF(ALT基因频率), n_miss(分析每个位点对应的缺失个体数)\n
    N: 被分析的个体总数\n
    Equation: pve = 2*(beta**2)*MAF(1-MAF)/(2*(beta**2)*MAF(1-MAF)+(se**2)*2*(N-n_miss)*MAF(1-MAF))
    '''
    df['pve'] = 2*np.power(df[beta],2)*df[maf]*(1-df[maf])/(2*np.power(df[beta],2)*df[maf]*(1-df[maf])+np.power(df[se_beta],2)*2*(N-df[n_miss])*((1-df[maf])))
    return df