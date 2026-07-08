import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from functools import lru_cache
from scipy.stats import beta

_PVALUE_EPS = float(np.nextafter(0.0, 1.0))
_QQ_SAMPLE_MAX_POINTS = 120_000
_QQ_BAND_MAX_POINTS = 20_000


def ppoints(n)->np.ndarray:
    '''
    生成理论均匀分布分位数
    '''
    return np.arange(1,n+1)/(n+1)#(np.arange(1, n+1) - 0.5) / n


@lru_cache(maxsize=64)
def _qq_confidence_band_logp_cached(
    n: int,
    ci: float,
    limit: int,
    rank_max: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    band_n = int(max(1, int(n)))
    rank_cap = band_n if rank_max is None else int(max(1, min(int(rank_max), band_n)))
    if rank_cap <= limit:
        ranks = np.arange(1, rank_cap + 1, dtype=np.int64)
    else:
        ranks = np.unique(
            np.round(np.geomspace(1.0, float(rank_cap), num=int(max(2, limit)))).astype(np.int64)
        )
        if ranks[0] != 1:
            ranks = np.insert(ranks, 0, 1)
        if ranks[-1] != rank_cap:
            ranks = np.append(ranks, rank_cap)
    ranks = np.sort(ranks)[::-1]

    ci_frac = float(ci)
    if ci_frac > 1.0:
        ci_frac /= 100.0
    ci_frac = float(np.clip(ci_frac, 1e-12, 1.0 - 1e-12))
    alpha = 1.0 - ci_frac
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0

    x_arr = -np.log10(ranks.astype(np.float64) / (band_n + 1.0))
    lo_arr = -np.log10(beta.ppf(q_hi, ranks, band_n - ranks + 1))
    hi_arr = -np.log10(beta.ppf(q_lo, ranks, band_n - ranks + 1))
    keep = np.isfinite(x_arr) & np.isfinite(lo_arr) & np.isfinite(hi_arr)
    x_arr = np.asarray(x_arr[keep], dtype=np.float64)
    lo_arr = np.asarray(lo_arr[keep], dtype=np.float64)
    hi_arr = np.asarray(hi_arr[keep], dtype=np.float64)
    x_arr.setflags(write=False)
    lo_arr.setflags(write=False)
    hi_arr.setflags(write=False)
    return (x_arr, lo_arr, hi_arr)


def _qq_confidence_band_logp(
    n_total: int,
    *,
    ci: float,
    max_points: int | None = None,
    rank_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(max(1, int(n_total)))
    limit = n if max_points is None else int(max(1, int(max_points)))
    rank_cap = None if rank_max is None else int(max(1, min(int(rank_max), n)))
    return _qq_confidence_band_logp_cached(n, float(ci), limit, rank_cap)


@lru_cache(maxsize=64)
def _qq_sample_draw_indices_cached(
    n: int,
    limit: int,
) -> np.ndarray:
    n_total = int(max(1, int(n)))
    keep_n = int(max(1, min(int(limit), n_total)))
    if n_total <= keep_n:
        arr = np.arange(n_total, dtype=np.int64)
        arr.setflags(write=False)
        return arr

    arr = np.linspace(0, n_total - 1, num=keep_n, dtype=np.int64)
    arr = np.unique(arr.astype(np.int64, copy=False))
    if arr.size < keep_n:
        fill_pool = np.setdiff1d(np.arange(n_total, dtype=np.int64), arr, assume_unique=True)
        fill_need = keep_n - arr.size
        if fill_pool.size > fill_need:
            fill_take = np.linspace(0, fill_pool.size - 1, num=fill_need, dtype=np.int64)
            fill_pool = fill_pool[fill_take]
        arr = np.sort(np.concatenate([arr, fill_pool[:fill_need]])).astype(np.int64, copy=False)
    elif arr.size > keep_n:
        take = np.linspace(0, arr.size - 1, num=keep_n, dtype=np.int64)
        arr = arr[np.unique(take)]
    arr.setflags(write=False)
    return arr


def _qq_sample_draw_indices(
    n_total: int,
    *,
    max_points: int | None = None,
) -> np.ndarray:
    n = int(max(1, int(n_total)))
    limit = _QQ_SAMPLE_MAX_POINTS if max_points is None else int(max(1, int(max_points)))
    return _qq_sample_draw_indices_cached(n, limit)

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
        p_sorted = np.sort(np.clip(np.asarray(p, dtype=np.float64), _PVALUE_EPS, 1.0))
        draw_idx = _qq_sample_draw_indices(n)
        obs_scatter = -np.log10(p_sorted[draw_idx])
        exp_scatter = -np.log10((draw_idx.astype(np.float64) + 1.0) / (n + 1.0))
        if model is not None:
            x_band, lower, upper = _qq_confidence_band_logp(n, ci=ci)
            # 绘制置信区间
            ax.fill_between(x_band, lower, upper, color='grey', alpha=0.4)
        # 绘制理论线（y=x）和观测点
        ax.plot([0, min(obs_scatter.max(),exp_scatter.max())], [0, min(obs_scatter.max(),exp_scatter.max())], lw=1,)
        ax.scatter(exp_scatter, obs_scatter, s=1, alpha=0.6,rasterized=True)
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
