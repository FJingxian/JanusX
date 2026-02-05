import gc
from typing import Any, List, Tuple, Union
from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from janusx.gfreader import load_genotype_chunks
from janusx.pyBLUP.assoc import SUPER
from janusx.garfield.logreg import logreg
from tqdm import tqdm

def getLogicgate(
    y: np.ndarray,
    M: np.ndarray,
    nsnp: int = 5,
    n_estimators: int = 200,
    sites: Any = None,
):
    """
    通过随机森林 + permutation importance + 逻辑回归
    在一段基因型矩阵中寻找逻辑组合 (xcombine)
    """
    if sites is not None:
        sites = np.array([f"{i.chrom}_{i.pos}" for i in sites])
    else:
        sites = np.arange(M.shape[0])

    # 相关性过滤（例如去掉高度共线 SNP）
    keep = SUPER(np.corrcoef(M), np.ones(M.shape[0]), thr=0.8)
    M = M[keep]
    sites = sites[keep]

    y = y.ravel()

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=nsnp,
        min_samples_leaf=nsnp,
        bootstrap=True,
        n_jobs=1,
        random_state=0,
    )
    rf.fit(M.T, y)
    Imp = rf.feature_importances_

    topk = nsnp
    idx = np.argsort(Imp)[::-1][:topk]

    # 0/1/2 → 0/1
    Mchoice = (M[idx] / 2).astype(int).T

    resdict = logreg(Mchoice, y, response="continuous", tags=sites[idx])
    return resdict


# ---------- 单区间 worker ----------
chrposTuple = Tuple[str, int, int]
def _process(
    ChromPos: Union[chrposTuple,List[chrposTuple]],
    genofile: str,
    sampleid: np.ndarray,
    y: np.ndarray,
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
            maf=0.02,
            missing_rate=0.05,
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
                maf=0.02,
                missing_rate=0.05,
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

    M = np.vstack(M_list).astype(np.float32)
    if M.shape[0] < 2:
        return None
    y_vec = y[:, 0].astype(float)
    try:
        resdict = getLogicgate(
            y_vec,
            M,
            nsnp=nsnp,
            n_estimators=n_estimators,
            sites=sites_list,
        )
    except Exception as e:
        # 某个区间出错时，不让整个并行崩掉
        print(f"[WARN] process {ChromPos} failed: {e}")
        return None
    return resdict

# ---------- 一次性收集结果的 main ----------

def main(genofile, sampleid, y, bedlist, threads=4, nsnp=5, n_estimators=200):
    results = Parallel(n_jobs=threads)(
        delayed(_process)(ChromPos, genofile, sampleid, y, nsnp, n_estimators)
        for ChromPos in tqdm(bedlist)
    )
    return results
