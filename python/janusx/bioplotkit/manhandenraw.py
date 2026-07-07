import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_PVALUE_EPS = float(np.nextafter(0.0, 1.0))
_QQ_SAMPLE_MAX_POINTS = 50_000


def ppoints(n)->np.ndarray:
    '''
    生成理论均匀分布分位数
    '''
    return np.arange(1,n+1)/(n+1)#(np.arange(1, n+1) - 0.5) / n


def _qq_confidence_band_logp(
    n_total: int,
    *,
    ci: float,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(max(1, int(n_total)))
    limit = _QQ_SAMPLE_MAX_POINTS if max_points is None else int(max(1, int(max_points)))
    from janusx import janusx as _jxrs

    exact = getattr(_jxrs, "qq_band_beta_logp_exact", None)
    if not callable(exact):
        raise RuntimeError(
            "janusx.janusx.qq_band_beta_logp_exact is unavailable. "
            "Rebuild/reinstall the JanusX Rust extension."
        )
    x_band, lower, upper = exact(n, ci=float(ci), max_points=limit)
    return (
        np.asarray(x_band, dtype=np.float64).reshape(-1),
        np.asarray(lower, dtype=np.float64).reshape(-1),
        np.asarray(upper, dtype=np.float64).reshape(-1),
    )


def _qq_sample_draw_indices(
    n_total: int,
    *,
    max_points: int | None = None,
) -> np.ndarray:
    n = int(max(1, int(n_total)))
    limit = _QQ_SAMPLE_MAX_POINTS if max_points is None else int(max(1, int(max_points)))
    from janusx import janusx as _jxrs

    sampler = getattr(_jxrs, "qq_rank_sample_zero_based", None)
    if not callable(sampler):
        raise RuntimeError(
            "janusx.janusx.qq_rank_sample_zero_based is unavailable. "
            "Rebuild/reinstall the JanusX Rust extension."
        )
    draw_idx = sampler(n, max_points=limit)
    arr = np.asarray(draw_idx, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise RuntimeError("Rust QQ rank sampler returned no indices.")
    return arr

class GWASPLOT:
    def __init__(self,df:pd.DataFrame, chr:str, pos:str, pvalue:str,interval_rate:int=.2):
        '''
        必须输入的项目df: 矩阵, chr: 染色体, pos: SNP位点, pvalue: 显著性
        '''
        df = df[[chr, pos, pvalue]].copy()
        df[chr] = df[chr].astype(int)
        df[pos] = df[pos].astype(int)
        df = df.sort_values(by=[chr,pos])
        chrlist = df[chr].unique()
        self.interval = int(interval_rate*df[pos].max())
        df['x'] = df[pos]
        if len(chrlist)>1:
            for ii in chrlist:
                if ii > 1:
                    df.loc[df[chr]==ii,'x'] = df.loc[df[chr]==ii,'x']+df[df[chr]==ii-1]['x'].max()
        df['x'] = (df[chr]-1)*self.interval+df['x']
        df['y'] = df[pvalue]
        df['z'] = df[chr]
        self.chrlist = chrlist
        self.ticks_loc = df.groupby('z')['x'].median()
        self.df = df.set_index([chr,pos])
        pass
    def manhattan(self, threshold:float=None, color_set:list=[], ax:plt.Axes = None, ignore:list=[]):
        df = self.df.copy()
        df['y'] = -np.log10(df['y'])
        if ax == None:
            fig = plt.figure(figsize=[12,6], dpi=600)
            gs = GridSpec(12, 1, figure=fig)
            ax = fig.add_subplot(gs[0:12,0])
        if len(color_set) == 0:
            color_set = ['black','grey']
        colors = dict(zip(self.chrlist,[color_set[i%len(color_set)] for i in range(len(self.chrlist))]))
        ax.scatter(df[~df.index.isin(ignore)]['x'], df[~df.index.isin(ignore)]['y'], alpha=1, s=4, color=df[~df.index.isin(ignore)]['z'].map(colors),rasterized=True)
        if threshold != None and max(df['y'])>=threshold:
            df_annote = df[df['y']>=threshold]
            ax.scatter(df_annote[~df_annote.index.isin(ignore)]['x'], df_annote[~df_annote.index.isin(ignore)]['y'], alpha=1, s=8, color='red',rasterized=True)
            ax.hlines(y=threshold, xmin=0, xmax=max(df['x']),color='grey', linewidth=1, alpha=1, linestyles='--')
        ax.set_xticks(self.ticks_loc, self.chrlist)
        ax.set_xlim([0-self.interval,max(df['x'])+self.interval])
        ax.set_ylim([0,max(df['y'])+0.1*max(df['y'])])
        ax.set_xlabel('Chromosome')
        ax.set_ylabel(r'-log$_\mathdefault{10}$(p-value)')
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        return ax
    def qq(self, ax:plt.Axes = None, model:int='qbeta',ci:int=95):
        '''
        可选: bootstrap 抽样次数, ci置信区间(分位数)
        '''
        df = self.df.copy()
        if ax == None:
            plt.rc('font', family='Times New Roman')
            fig = plt.figure(figsize=[12,6], dpi=600)
            gs = GridSpec(12, 1, figure=fig)
            ax = fig.add_subplot(gs[0:12,0])
        p = df['y'].dropna()
        n = len(p)
        p_sorted = np.sort(np.asarray(p, dtype=np.float64))
        draw_idx = _qq_sample_draw_indices(n)
        o = -np.log10(p_sorted[draw_idx])
        e_theoretical = -np.log10((draw_idx.astype(np.float64) + 1.0) / (n + 1.0))
        if model is not None:
            e_theoretical, lower, upper = _qq_confidence_band_logp(n, ci=ci)
            # 绘制置信区间
            ax.fill_between(e_theoretical, lower, upper, color='grey', alpha=0.4)
        # 绘制理论线（y=x）和观测点
        ax.plot([0, min(o.max(),e_theoretical.max())], [0, min(o.max(),e_theoretical.max())], lw=1,)
        ax.scatter(e_theoretical, o, s=1, alpha=0.6,rasterized=True)
        ax.set_xlabel(r'Expected -log$_\mathdefault{10}$(p-value)')
        ax.set_ylabel(r'Observed -log$_\mathdefault{10}$(p-value)')
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
