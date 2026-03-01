import numbers
import sys
import time
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Literal, List, Union
from scipy.stats import linregress

try:
    from rich.progress import track as rich_track
except Exception:
    rich_track = None


def _progress_iter(iterable, *, total=None, desc:str='', enable:bool=True):
    if not enable:
        return iterable
    if rich_track is not None and sys.stderr.isatty():
        try:
            return rich_track(iterable, total=total, description=desc)
        except Exception:
            pass
    return tqdm(iterable, total=total, desc=desc)


## covariation matrix
def cor(expr:np.ndarray, cortype:Literal['signed', 'unsigned']='unsigned'):
    '''
    matrix: raw data, correlationship of row variation will be calculated.
    model: 'signed' or 'unsigned', default: 'unsigned'
    '''
    gcor = np.corrcoef(expr, dtype=np.float32)
    if cortype == 'signed':
        gcor = .5*gcor+.5
    elif cortype == 'unsigned':
        gcor = np.absolute(gcor)
    return gcor.astype(np.float32, copy=False)

## adjacent matrix
def adj(cov:np.ndarray, sft:Union[List[int], int],
        *,
        thr:float=0.8,eps:float=1e-8,bins:int=10,
        show_progress:bool=True):
    if isinstance(sft, list) or isinstance(sft, range):
        sft_values = [int(i) for i in sft]
        if len(sft_values) == 0:
            raise ValueError('sft list/range is empty')
        if min(sft_values) < 0:
            raise ValueError(f'sft powers should be >= 0, got min={min(sft_values)}')

        cov = np.asarray(cov, dtype=np.float32)
        unique_powers = sorted(set(sft_values))
        stats_by_power = {}

        # Iterative multiplication reuses A^(p-1) -> A^p and avoids repeated np.power calls.
        cur_power = 0
        cur_adj = np.ones_like(cov, dtype=np.float32)
        cov2 = (cov * cov).astype(np.float32, copy=False)
        for power in _progress_iter(unique_powers, total=len(unique_powers), desc='Scan powers', enable=show_progress):
            while cur_power < power:
                if cur_power + 2 <= power:
                    np.multiply(cur_adj, cov2, out=cur_adj)
                    cur_power += 2
                else:
                    np.multiply(cur_adj, cov, out=cur_adj)
                    cur_power += 1

            sumcor = np.sum(cur_adj, axis=0, dtype=np.float32) - np.float32(1.0)
            meank = float(np.mean(sumcor))
            mediank = float(np.median(sumcor))
            maxk = float(np.max(sumcor))

            freq, r = np.histogram(sumcor, bins=bins)
            r = (r[1:]+r[:-1])/2
            valid = (freq > 0) & (r > 0)
            if valid.sum() < 2:
                stats_by_power[power] = (np.nan, np.nan, meank, mediank, maxk)
                continue

            x = np.log(r[valid])
            y = np.log(freq[valid]+eps)
            res = linregress(x, y)
            stats_by_power[power] = (float(np.power(res.rvalue, 2)), float(res.slope), meank, mediank, maxk)

        rsqr = [stats_by_power[p][0] for p in sft_values]
        slope = [stats_by_power[p][1] for p in sft_values]
        meank = [stats_by_power[p][2] for p in sft_values]
        mediank = [stats_by_power[p][3] for p in sft_values]
        maxk = [stats_by_power[p][4] for p in sft_values]

        sftresult = pd.DataFrame({'sft':sft_values, 'rsqr':rsqr, 'slope':slope, 'meank':meank, 'mediank':mediank, 'maxk':maxk})
        print(sftresult)
        passed = sftresult.loc[sftresult['rsqr']>=thr,'sft']
        if len(passed) > 0:
            sft = passed.iloc[0]
        else:
            valid_rsqr = sftresult['rsqr'].dropna()
            if valid_rsqr.empty:
                sft = sft_values[0]
                warnings.warn(
                    f'No valid rsqr estimated from bins={bins}; fallback to first power {sft}.',
                    RuntimeWarning
                )
            else:
                fallback_row = sftresult.loc[valid_rsqr.idxmax()]
                sft = int(fallback_row['sft'])
                warnings.warn(
                    f'No power reached rsqr>={thr}; fallback to power {sft} (max rsqr={fallback_row["rsqr"]:.4f}).',
                    RuntimeWarning
                )
        print(f'Chosen power: {sft}')
        return adj(cov, sft)
    elif isinstance(sft, numbers.Integral):
        matrix_adj = np.power(cov,sft).astype(np.float32, copy=False)
        return matrix_adj
    else:
        raise TypeError(f'sft should be int or list of int, now is {type(sft)}')

## 共轭矩阵
# Ω=[wij],wij=(lij+aij)/(minki,kj+1−aij)
def TOM(adj:np.ndarray):
    adj = np.asarray(adj, dtype=np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f'Adjacency matrix must be square 2D, got shape={adj.shape}')
    ngenes = adj.shape[0]
    # Keep diagonal fixed to 1 for the standard unsigned TOM definition.
    np.fill_diagonal(adj, np.float32(1.0))

    # k_i = sum_u a_iu - 1
    k = np.sum(adj, axis=1, dtype=np.float32) - np.float32(1.0)

    # Numerator: l_ij + a_ij = (A @ A - 2A) + A = A @ A - A
    # Memory peak keeps only two n×n matrices: adj and tom.
    tom = (adj @ adj).astype(np.float32, copy=False)
    tom -= adj

    # Denominator is computed row-wise as vectors to avoid an extra n×n matrix.
    denom_row = np.empty(ngenes, dtype=np.float32)
    for i in range(ngenes):
        np.minimum(k[i], k, out=denom_row)
        denom_row += np.float32(1.0)
        denom_row -= adj[i, :]
        np.divide(tom[i, :], denom_row, out=tom[i, :], where=(denom_row != 0))

    # TOM diagonal is defined as 1.
    np.fill_diagonal(tom, np.float32(1.0))
    return tom

if __name__ == "__main__":
    tpm = pd.read_csv('~/Public/test.tpm.tsv',sep='\t',index_col=0)
    tpm = tpm.loc[tpm.mean(axis=1)>1]
    cv = (tpm.std(axis=1)/tpm.mean(axis=1)).sort_values(ascending=False)
    tpm = tpm.loc[cv.index]
    print(tpm.shape)
    print(tpm.iloc[:4,:4])
    mcor = cor(tpm)
    madj = adj(mcor,list(range(2,10,1))+list(range(10,21,2)))
    tst = time.time()
    mtom = TOM(madj)
    print(mtom[:4,:4])
    print(f'TOM computed in {time.time()-tst:.3f} seconds')
    tst = time.time()
