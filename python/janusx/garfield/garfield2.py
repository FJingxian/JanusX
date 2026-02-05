import gc
import os
from typing import Any, List, Literal, Tuple, Union
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from janusx.gfreader import load_genotype_chunks
from janusx.garfield.logreg import logreg
from tqdm import tqdm

def ldprune(
    M: np.ndarray,
    sites,
    thr: float = 0.8,
    max_snps: int = 600,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    轻量 LD 过滤：
      1) 按方差从大到小排序 保留信息量大的 SNP
      2) 依次尝试加入 kept 列表，只与已有 kept 算相关，>thr 则丢弃
      3) 最多保留 max_snps 个 SNP

    参数
    ----
    M : (m_snps, n_samples) 基因型矩阵
    sites : 与 SNP 对应的位点信息数组
    thr : 相关系数阈值（|r| >= thr 视为 LD 过高）
    max_snps : 最多保留 SNP 数量

    返回
    ----
    M_kept, sites_kept
    """
    m, n = M.shape
    if m <= 1:
        return M, sites

    # 1) 去掉方差为 0 的 SNP（完全无多态）
    var = M.var(axis=1)
    non_mono = var > 0
    M = M[non_mono]
    sites = np.array(sites)[non_mono]
    var = var[non_mono]
    m = M.shape[0]
    if m <= 1:
        return M, sites

    # 2) 按方差排序（方差越大越优先）
    order = np.argsort(var)[::-1]
    M = M[order]
    sites = sites[order]

    # 3) 标准化：每个 SNP 变成 0 均值、单位方差
    mu = M.mean(axis=1, keepdims=True)
    sigma = M.std(axis=1, keepdims=True)
    sigma[sigma == 0] = 1.0
    Z = (M - mu) / sigma  # (m, n)

    kept_idx = []
    for i in range(m):
        if len(kept_idx) == 0:
            kept_idx.append(i)
        else:
            # 只与已保留 SNP 算相关
            z_i = Z[i]  # (n,)
            Z_kept = Z[kept_idx]  # (k, n)
            # corr = dot / n，因为已经标准化了
            corr = (Z_kept @ z_i) / n
            if np.all(np.abs(corr) < thr):
                kept_idx.append(i)
        if len(kept_idx) >= max_snps:
            break

    kept_idx = np.array(kept_idx, dtype=int)
    return M[kept_idx], sites[kept_idx]

def getLogicgate(
    y: np.ndarray,
    M: np.ndarray,
    nsnp: int = 5,
    n_estimators: int = 200,
    sites: Any = None,
    core:Literal['rf','gbdt']='gbdt'
):
    """
    通过随机森林 + permutation importance + 逻辑回归
    在一段基因型矩阵中寻找逻辑组合 (xcombine)
    """
    if sites is not None:
        if len(sites) == 0:
            sites = np.arange(M.shape[0])
        else:
            first = sites[0]
            if hasattr(first, "chrom") and hasattr(first, "pos"):
                sites = np.array([f"{i.chrom}_{i.pos}" for i in sites])
            else:
                sites = np.array([str(i) for i in sites])
    else:
        sites = np.arange(M.shape[0])

    M,sites = ldprune(M,sites)
    if core == 'rf':
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=nsnp,
            min_samples_leaf=nsnp,
            bootstrap=True,
            n_jobs=1,
            random_state=0,
        )
    elif core == 'gbdt':
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=nsnp,
            learning_rate=0.05,
            subsample=0.7,
            random_state=0,
        )
    model.fit(M.T, y)
    Imp = model.feature_importances_

    topk = nsnp
    idx = np.argsort(Imp)[::-1][:topk]

    # 0/1/2 → 0/1
    Mchoice = (M[idx] / 2).astype(int).T

    resdict = logreg(Mchoice, y, response="continuous", tags=sites[idx])
    if resdict is None:
        return None
    indices = resdict.get("indices", [])
    if indices is None or len(indices) <= 1:
        return None
    if resdict.get("expression", "") == "1":
        return None
    return resdict


# ---------- 遍历全基因组 ----------
def _window_task(
    task: Tuple[str, int, int, np.ndarray, List[str]],
    y: np.ndarray,
    nsnp: int,
    n_estimators: int,
) -> Union[dict[str,Any], None]:
    chrom, start, end, M_win, tags = task
    try:
        resdict = getLogicgate(
            y,
            M_win,
            nsnp=nsnp,
            n_estimators=n_estimators,
            sites=tags,
        )
    except Exception as e:
        print(f"[WARN] window {chrom}:{start}-{end} failed: {e}")
        return None
    if resdict is None:
        return None
    return resdict


def window(
    genofile: str,
    sampleid: np.ndarray,
    y: np.ndarray,
    step: int,
    windowsize: int,
    chunk_size: Union[int,float] = 1e5,
    maf: float = 0.02,
    missing_rate: float = 0.05,
    nsnp: int = 5,
    n_estimators: int = 200,
    threads: int = 4,
    batch_size: int = 128,
):
    """
    全基因组滑窗遍历 (按 chunksize 顺序加载，减少重复 IO)。

    窗口定义：
      - 窗口起点每次推进 loc
      - 每个窗口区间为 [start, start + windowsize)

    读取逻辑：
      - 保持最多两个窗口的数据
      - 读完两个窗口后丢弃第一个窗口，再加载第三个窗口

    Returns:
      list of dict
    """
    y = np.asarray(y, dtype=float).ravel()
    tasks: List[Tuple[str, int, int, np.ndarray, List[str]]] = []
    results: List[dict[str,Any]] = []

    total_windows = None
    step_int = int(step) if step else 0
    if step_int > 0 and not (str(genofile).endswith(".vcf") or str(genofile).endswith(".vcf.gz")):
        bim_path = f"{genofile}.bim"
        if os.path.isfile(bim_path):
            try:
                bim = pd.read_csv(bim_path, sep=r"\s+", header=None, usecols=[0, 3])
                if not bim.empty:
                    chr_minmax = bim.groupby(0)[3].agg(["min", "max"])
                    counts = (chr_minmax["max"] - chr_minmax["min"]) // step_int + 1
                    total_windows = int(counts.sum())
            except Exception:
                total_windows = None
    pbar = tqdm(total=total_windows, desc="Windows", unit="win")

    buffer_M = None
    buffer_sites: List[Any] = []
    buffer_pos = None
    current_chrom = None
    window_start = None
    next_start = None

    def _append_segment(seg_M, seg_sites):
        nonlocal buffer_M, buffer_sites, buffer_pos
        if buffer_M is None:
            buffer_M = seg_M
            buffer_sites = list(seg_sites)
        else:
            buffer_M = np.vstack([buffer_M, seg_M])
            buffer_sites.extend(seg_sites)
        pos_arr = np.array([int(s.pos) for s in buffer_sites], dtype=np.int64)
        buffer_pos = pos_arr

    def _drop_before(pos_thr):
        nonlocal buffer_M, buffer_sites, buffer_pos
        if buffer_M is None:
            return
        mask = buffer_pos >= pos_thr
        if mask.sum() == 0:
            buffer_M = None
            buffer_sites = []
            buffer_pos = None
            return
        buffer_M = buffer_M[mask]
        buffer_sites = [s for i, s in enumerate(buffer_sites) if mask[i]]
        buffer_pos = buffer_pos[mask]

    def _emit_window(start_pos):
        if buffer_M is None:
            return None
        end_pos = start_pos + windowsize
        mask = (buffer_pos >= start_pos) & (buffer_pos < end_pos)
        if mask.sum() < 2:
            return None
        M_win = buffer_M[mask]
        sites_win = [s for i, s in enumerate(buffer_sites) if mask[i]]
        tags = [f"{s.chrom}_{s.pos}" for s in sites_win]
        return current_chrom, start_pos, end_pos, M_win, tags

    chunks = load_genotype_chunks(
        genofile,
        chunk_size=int(chunk_size),
        maf=maf,
        missing_rate=missing_rate,
        impute=True,
        sample_ids=sampleid,
    )

    try:
        for chunk, sites in chunks:
            if not sites:
                continue
            # split by chromosome boundaries within the chunk
            idx = 0
            while idx < len(sites):
                chrom = str(sites[idx].chrom)
                j = idx + 1
                while j < len(sites) and str(sites[j].chrom) == chrom:
                    j += 1
                seg_sites = sites[idx:j]
                seg_M = chunk[idx:j]

                if current_chrom is None or chrom != current_chrom:
                    # flush remaining windows for previous chromosome
                    if current_chrom is not None and buffer_pos is not None:
                        max_pos = int(buffer_pos.max())
                        while window_start is not None and window_start <= max_pos:
                            res = _emit_window(window_start)
                            pbar.update(1)
                            if res is not None:
                                tasks.append(res)
                                if len(tasks) >= batch_size:
                                    if threads <= 1:
                                        batch = [
                                            out for out in (_window_task(t, y, nsnp, n_estimators) for t in tasks)
                                            if out is not None
                                        ]
                                    else:
                                        batch = Parallel(n_jobs=threads, backend="loky")(
                                            delayed(_window_task)(t, y, nsnp, n_estimators) for t in tasks
                                        )
                                        batch = [r for r in batch if r is not None]
                                    results.extend(batch)
                                    tasks.clear()
                            window_start += step
                    # reset buffers for new chromosome
                    buffer_M = None
                    buffer_sites = []
                    buffer_pos = None
                    current_chrom = chrom
                    window_start = int(seg_sites[0].pos)
                    next_start = window_start + step

                _append_segment(seg_M, seg_sites)

                # process windows when we have two windows worth of data
                while buffer_pos is not None and buffer_pos.max() >= window_start + windowsize + step:
                    res = _emit_window(window_start)
                    pbar.update(1)
                    if res is not None:
                        tasks.append(res)
                        if len(tasks) >= batch_size:
                            if threads <= 1:
                                batch = [
                                    out for out in (_window_task(t, y, nsnp, n_estimators) for t in tasks)
                                    if out is not None
                                ]
                            else:
                                batch = Parallel(n_jobs=threads, backend="loky")(
                                    delayed(_window_task)(t, y, nsnp, n_estimators) for t in tasks
                                )
                                batch = [r for r in batch if r is not None]
                            results.extend(batch)
                            tasks.clear()
                    # drop the first window, advance
                    _drop_before(next_start)
                    window_start = next_start
                    next_start = window_start + step

                idx = j

        # flush remaining windows at end
        if current_chrom is not None and buffer_pos is not None:
            max_pos = int(buffer_pos.max())
            while window_start is not None and window_start <= max_pos:
                res = _emit_window(window_start)
                pbar.update(1)
                if res is not None:
                    tasks.append(res)
                    if len(tasks) >= batch_size:
                        if threads <= 1:
                            batch = [
                                out for out in (_window_task(t, y, nsnp, n_estimators) for t in tasks)
                                if out is not None
                            ]
                        else:
                            batch = Parallel(n_jobs=threads, backend="loky")(
                                delayed(_window_task)(t, y, nsnp, n_estimators) for t in tasks
                            )
                            batch = [r for r in batch if r is not None]
                        results.extend(batch)
                        tasks.clear()
                window_start += step

        if tasks:
            if threads <= 1:
                batch = [
                    out for out in (_window_task(t, y, nsnp, n_estimators) for t in tasks)
                    if out is not None
                ]
            else:
                batch = Parallel(n_jobs=threads, backend="loky")(
                    delayed(_window_task)(t, y, nsnp, n_estimators) for t in tasks
                )
                batch = [r for r in batch if r is not None]
            results.extend(batch)
    finally:
        pbar.close()

    return results


# ---------- 自定义遍历 ----------
chrposTuple = Tuple[str, int, int]
def _process(
    ChromPos: Union[chrposTuple,List[chrposTuple]],
    genofile: str,
    sampleid: np.ndarray,
    y: np.ndarray,
    maf: float = 0.02,
    missing_rate: float = 0.05,
    nsnp: int = 5,
    n_estimators: int = 200,
):
    """
    对一个 bed 区间执行:
      1) 从 genofile 取出该区间所有 SNP 的基因型矩阵
      2) 调用 getLogicgate
      3) 返回 (xcombine, site_tuple) 或 None
    """
    if isinstance(ChromPos,tuple):
        chrom, start, end = ChromPos
        chunks = load_genotype_chunks(
            genofile,
            chunk_size=int(1e6),
            maf=maf,
            missing_rate=missing_rate,
            impute=True,
            bim_range=(str(chrom), int(start), int(end)),
            sample_ids=sampleid,
        )

        M_list = []
        sites_list = []

        for chunk, sites in chunks:
            # chunk: (m_chunk, n_samples)
            M_list.append(chunk)
            sites_list.extend(sites)
        if not M_list:
            return None
    elif isinstance(ChromPos,list):
        M_list = []
        sites_list = []
        for chrom, start, end in ChromPos:
            chunks = load_genotype_chunks(
                genofile,
                chunk_size=int(1e6),
                maf=maf,
                missing_rate=missing_rate,
                impute=True,
                bim_range=(str(chrom), int(start), int(end)),
                sample_ids=sampleid,
            )
            for chunk, sites in chunks:
                # chunk: (m_chunk, n_samples)
                M_list.append(chunk)
                sites_list.extend(sites)
        if not M_list:
            return None

    M = np.vstack(M_list)
    if M.shape[0] < 2:
        return None
    try:
        resdict = getLogicgate(
            y,
            M,
            nsnp=nsnp,
            n_estimators=n_estimators,
            sites=sites_list,
        )
    except Exception as e:
        # 某个区间出错时，不让整个并行崩掉
        print(f"[WARN] process {ChromPos} failed: {e}")
        return None
    if resdict is None:
        return None
    return resdict

# ---------- 一次性收集结果的 main ----------

def main(genofile, sampleid, y, bedlist,
    maf: float = 0.02,missing_rate: float = 0.05, 
    threads=4, nsnp=5, n_estimators=200):
    y = np.asarray(y, dtype=float).ravel()
    results = Parallel(n_jobs=threads)(
        delayed(_process)(ChromPos, genofile, sampleid, y, maf, missing_rate, nsnp, n_estimators)
        for ChromPos in tqdm(bedlist)
    )
    return results
