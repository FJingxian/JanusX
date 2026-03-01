"""WGCNA helper functions for correlation, adjacency, and TOM construction.

This module implements a lightweight WGCNA-style workflow:
1. Build gene-gene correlation matrix (`cor`)
2. Convert correlation to adjacency matrix (`adj`)
3. Compute topological overlap matrix (`TOM`)

All heavy matrix operations are kept in `float32` to reduce memory footprint.
"""

import numbers
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Literal, List, Union
from scipy.stats import linregress
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

try:
    from rich.progress import track as rich_track
except Exception:
    rich_track = None

try:
    from dynamicTreeCut import cutreeHybrid as _cutree_hybrid
except Exception:
    _cutree_hybrid = None


def _progress_iter(iterable, *, total=None, desc:str='', enable:bool=True):
    """Return a progress-enabled iterator.

    Rich progress is preferred when available on a TTY. Otherwise it falls back
    to tqdm. When `enable=False`, the raw iterable is returned.

    Parameters
    ----------
    iterable : iterable
        Source iterable.
    total : int, optional
        Expected number of iterations.
    desc : str, default=''
        Progress description text.
    enable : bool, default=True
        Whether to enable progress rendering.

    Returns
    -------
    iterable
        Wrapped iterator with progress behavior when enabled.
    """
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
    """Compute correlation matrix from expression matrix.

    Parameters
    ----------
    expr : np.ndarray
        Input matrix with genes on rows and samples on columns.
    cortype : {'signed', 'unsigned'}, default='unsigned'
        Correlation transformation mode:
        - 'signed': transform r -> 0.5 * r + 0.5
        - 'unsigned': use absolute correlation |r|

    Returns
    -------
    np.ndarray
        Correlation matrix in float32.
    """
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
    """Build adjacency matrix or select soft-threshold power.

    Parameters
    ----------
    cov : np.ndarray
        Correlation matrix (typically output of :func:`cor`).
    sft : list[int] | range | int
        Soft-threshold power specification.
        - If `int`, return `cov ** sft` directly.
        - If list/range, evaluate candidate powers and select one by `rsqr`.
    thr : float, default=0.8
        Minimum R^2 threshold for choosing soft-threshold power.
    eps : float, default=1e-8
        Small value added to histogram frequency before `log`.
    bins : int, default=10
        Histogram bins used for scale-free topology fit.
    show_progress : bool, default=True
        Whether to display progress while scanning candidate powers.

    Returns
    -------
    np.ndarray
        Adjacency matrix in float32 for the selected power.

    Notes
    -----
    When no candidate reaches `thr`, the function falls back to the power with
    maximal `rsqr` and emits a runtime warning.
    """
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
        return np.power(cov,sft).astype(np.float32, copy=False), sftresult
    elif isinstance(sft, numbers.Integral):
        return np.power(cov,sft).astype(np.float32, copy=False), None
    else:
        raise TypeError(f'sft should be int or list of int, now is {type(sft)}')

## 共轭矩阵
# Ω=[wij],wij=(lij+aij)/(minki,kj+1−aij)
def tom(adj:np.ndarray):
    """Compute topological overlap matrix (TOM) from adjacency matrix.

    Parameters
    ----------
    adj : np.ndarray
        Square adjacency matrix (float-like). Diagonal is forced to 1.

    Returns
    -------
    np.ndarray
        TOM matrix in float32 with diagonal fixed to 1.

    Notes
    -----
    This implementation keeps memory usage low by:
    - reusing `adj`
    - allocating one TOM matrix (`n x n`)
    - computing denominator row-wise with a 1D buffer
    """
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

def cluster(
    tom: np.ndarray,
    *,
    method: str = "average",
    min_cluster_size: int = 30,
    deep_split: int = 2,
    pam_stage: bool = False,
    cut_height: Union[float, None] = None,
    num_modules: Union[int, None] = None,
    return_linkage: bool = False,
):
    """Cluster genes from TOM matrix.

    Parameters
    ----------
    tom : np.ndarray
        Topological overlap matrix (square).
    method : str, default='average'
        Linkage method passed to `scipy.cluster.hierarchy.linkage`.
    min_cluster_size : int, default=30
        Minimum module size for dynamic tree cut.
    deep_split : int, default=2
        Dynamic split level for dynamic tree cut.
    pam_stage : bool, default=False
        Whether to enable PAM stage in dynamic tree cut.
    cut_height : float | None, default=None
        Optional dynamicTreeCut `cutHeight`. If None, dynamicTreeCut chooses a
        default value and may print a message about the inferred cut height.
    num_modules : int | None, default=None
        Target number of final modules (excluding label 0). When provided,
        `cut_height` is ignored and the function automatically searches a
        suitable `cutHeight` value.
    return_linkage : bool, default=False
        If True, return `(labels, Z)`; otherwise return `labels`.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray]
        Cluster labels from dynamic tree cut, optionally with linkage.
    """
    if _cutree_hybrid is None:
        raise ImportError(
            "dynamicTreeCut is required for cluster(). Please install dynamicTreeCut."
        )

    tom = np.asarray(tom, dtype=np.float32)
    if tom.ndim != 2 or tom.shape[0] != tom.shape[1]:
        raise ValueError(f"tom must be square 2D, got shape={tom.shape}")
    n = tom.shape[0]

    # Keep float32 to reduce memory; assume TOM is already symmetric.
    dist = np.subtract(np.float32(1.0), tom, dtype=np.float32)
    np.fill_diagonal(dist, np.float32(0.0))
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)

    def _labels_from_result(dynamic_res):
        if isinstance(dynamic_res, dict):
            labels_local = None
            for key in ("labels", "label", "clusters", "cluster"):
                if key in dynamic_res:
                    labels_local = np.asarray(dynamic_res[key], dtype=np.int32)
                    break
            if labels_local is None:
                raise RuntimeError(
                    "dynamicTreeCut returned a dict without labels/cluster fields."
                )
        else:
            labels_local = np.asarray(dynamic_res, dtype=np.int32)
        if labels_local.shape[0] != n:
            raise RuntimeError(
                f"dynamicTreeCut returned invalid label length {labels_local.shape[0]} (expect {n})."
            )
        return labels_local

    def _run_dynamic(ch):
        dynamic_res = _cutree_hybrid(
            Z,
            distM=dist,
            deepSplit=deep_split,
            minClusterSize=min_cluster_size,
            pamStage=pam_stage,
            cutHeight=ch,
            verbose=0,
        )
        labels_local = _labels_from_result(dynamic_res)
        n_mod_local = int(np.unique(labels_local[labels_local > 0]).size)
        return labels_local, n_mod_local

    if num_modules is not None:
        target = int(num_modules)
        if target <= 0:
            raise ValueError(f"num_modules must be positive, got {num_modules}")
        if cut_height is not None:
            warnings.warn(
                "num_modules is set; ignoring cut_height and auto-searching cutHeight.",
                RuntimeWarning,
            )

        heights = Z[:, 2]
        lo = float(np.min(heights))
        hi = float(np.max(heights))
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise RuntimeError("Invalid linkage heights for cutHeight search.")
        if hi <= lo:
            hi = lo + 1e-8

        cache = {}

        def _eval(ch):
            key = round(float(ch), 12)
            if key not in cache:
                cache[key] = _run_dynamic(float(ch))
            return cache[key]

        labels_lo, n_lo = _eval(lo)
        labels_hi, n_hi = _eval(hi)

        # Track the closest solution found.
        best_labels = labels_lo
        best_n_mod = n_lo
        best_cut_height = lo
        best_diff = abs(n_lo - target)
        if abs(n_hi - target) < best_diff:
            best_labels = labels_hi
            best_n_mod = n_hi
            best_cut_height = hi
            best_diff = abs(n_hi - target)

        # Determine monotonic direction: as cutHeight increases, module count
        # is usually non-increasing, but we infer from boundary evaluations.
        decreasing = n_lo >= n_hi

        # If target is outside reachable boundary counts, return closest bound.
        lo_n, hi_n = (n_lo, n_hi) if decreasing else (n_hi, n_lo)
        if target < hi_n or target > lo_n:
            labels = best_labels
        else:
            # Binary search on cutHeight.
            for _ in range(16):
                mid = (lo + hi) / 2.0
                cand_labels, cand_n_mod = _eval(mid)
                diff = abs(cand_n_mod - target)
                if diff < best_diff:
                    best_diff = diff
                    best_labels = cand_labels
                    best_n_mod = cand_n_mod
                    best_cut_height = mid
                if cand_n_mod == target or (hi - lo) < 1e-6:
                    break

                if decreasing:
                    # Higher cutHeight => fewer modules.
                    if cand_n_mod > target:
                        lo = mid
                    else:
                        hi = mid
                else:
                    # Fallback for rare non-decreasing response.
                    if cand_n_mod < target:
                        lo = mid
                    else:
                        hi = mid

            labels = best_labels

        print(
            f"[cluster] num_modules target={target}, obtained={best_n_mod}, cutHeight={best_cut_height:.6g}"
        )
        if best_n_mod != target:
            warnings.warn(
                f"Could not hit num_modules={target} exactly; closest={best_n_mod}, cutHeight={best_cut_height:.6g}.",
                RuntimeWarning,
            )
    else:
        dynamic_res = _cutree_hybrid(
            Z,
            distM=dist,
            deepSplit=deep_split,
            minClusterSize=min_cluster_size,
            pamStage=pam_stage,
            cutHeight=cut_height,
            verbose=0,
        )
        labels = _labels_from_result(dynamic_res)

    return (labels, Z) if return_linkage else labels

if __name__ == "__main__":
    import numpy as np
    from janusx.gtools.wgcna import cor, adj, tom, cluster
    import pandas as pd
    tpm = pd.read_csv('~/Public/test.tpm.tsv',sep='\t',index_col=0)
    tpm = tpm.loc[tpm.mean(axis=1)>1]
    cv = (tpm.std(axis=1)/tpm.mean(axis=1)).sort_values(ascending=False).iloc[:15000]
    tpm = tpm.loc[cv.index]
    print(tpm.shape)
    mcor = cor(tpm,'signed')
    madj,sftresult = adj(mcor,list(range(1,10,1))+list(range(10,21,2)))
    if sftresult is not None:
        print(sftresult.set_index('sft'))
    mtom = tom(madj)
    labels = cluster(mtom, num_modules=10,min_cluster_size=30)
    for i in range(1,11): # 计算ME
        print(f"Module {i}:")
        tpm_module = tpm.loc[labels==i].T
        gene = tpm_module.T.iloc[29].values
        tpm_module = (tpm_module - tpm_module.values.mean(axis=1, keepdims=True)).values / tpm_module.values.std(axis=1, keepdims=True)
        eigval,eigvec = np.linalg.eigh(tpm_module@tpm_module.T)
        me = eigvec[:,-1]
        print(np.corrcoef(me,gene)[0,1])
        # break
    
    