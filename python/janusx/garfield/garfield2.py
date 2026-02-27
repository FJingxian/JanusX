import os
from typing import Any, List, Literal, Tuple, Union, Optional
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from janusx.gfreader import load_genotype_chunks, inspect_genotype_file
from janusx.garfield.logreg import logreg
from tqdm import tqdm

def load_all_genotype_int8(
    genofile: str,
    sampleid: np.ndarray,
    chunk_size: int = int(1e6),
    maf: float = 0.02,
    missing_rate: float = 0.05,
    mmap_window_mb: Union[int, None] = None,
    total_snps: Union[int, None] = None,
) -> tuple[np.ndarray, List[Any]]:
    """
    Load all filtered genotypes once into memory as SNP-major int8 matrix.

    Supported input: PLINK / VCF(.gz) / TXT matrix.
    """
    sample_arr = np.asarray(sampleid, dtype=str)
    low = str(genofile).lower()
    if low.endswith(".vcf") or low.endswith(".vcf.gz"):
        all_ids, n_all_snps = inspect_genotype_file(genofile)
        all_ids = np.asarray(all_ids, dtype=str)
        sameidx = np.isin(all_ids, sample_arr)
        if not np.any(sameidx):
            raise ValueError(
                "No overlapping samples between genotype input and selected sample IDs."
            )
        sample_sub = None
        n_use = int(np.sum(sameidx))
    else:
        _all_ids, n_all_snps = inspect_genotype_file(genofile)
        sameidx = None
        sample_sub = sample_arr.tolist()
        n_use = len(sample_sub)

    chunk_size = max(1, int(chunk_size))
    if total_snps is None:
        total_snps = int(n_all_snps)
    total_chunks = None
    if total_snps is not None and int(total_snps) > 0:
        total_chunks = (int(total_snps) + chunk_size - 1) // chunk_size

    chunks = load_genotype_chunks(
        genofile,
        chunk_size=chunk_size,
        maf=maf,
        missing_rate=missing_rate,
        impute=True,
        sample_ids=sample_sub,
        mmap_window_mb=mmap_window_mb,
    )
    M_list: List[np.ndarray] = []
    sites_list: List[Any] = []
    for chunk, sites in tqdm(
        chunks,
        total=total_chunks,
        desc="Loading genotype",
        unit="chunk",
    ):
        if chunk.size == 0 or len(sites) == 0:
            continue
        if sameidx is not None:
            chunk = chunk[:, sameidx]
        M_list.append(np.asarray(np.rint(chunk), dtype=np.int8))
        sites_list.extend(sites)

    if not M_list:
        return np.empty((0, n_use), dtype=np.int8), []
    M = np.vstack(M_list)
    return np.ascontiguousarray(M, dtype=np.int8), sites_list

def ldprune(
    M: np.ndarray,
    sites,
    thr: float = 0.8,
    max_snps: int = 600,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight LD pruning:
      1) Sort SNPs by variance in descending order (prefer informative SNPs)
      2) Iteratively add candidates and compute correlation only vs kept SNPs
      3) If LD(r^2) >= thr with kept SNPs, keep the SNP with larger MAF
      4) Keep at most ``max_snps`` SNPs

    Parameters
    ----------
    M : (m_snps, n_samples) genotype matrix
    sites : site metadata aligned with SNP rows
    thr : LD threshold on r^2 (r^2 >= thr is considered high LD)
    max_snps : maximum number of SNPs to keep

    Returns
    -------
    M_kept, sites_kept
    """
    m, n = M.shape
    if m <= 1:
        return M, sites

    # 1) Remove zero-variance SNPs (monomorphic).
    var = M.var(axis=1)
    non_mono = var > 0
    M = M[non_mono]
    sites = np.array(sites)[non_mono]
    var = var[non_mono]
    m = M.shape[0]
    if m <= 1:
        return M, sites

    # 2) Compute MAF (larger means closer to 0.5).
    af = np.clip(M.mean(axis=1, dtype=np.float64) / 2.0, 0.0, 1.0)
    maf = np.minimum(af, 1.0 - af)

    # 3) Sort by variance (larger variance first).
    order = np.argsort(var)[::-1]
    M = M[order]
    sites = sites[order]
    maf = maf[order]

    # 4) Standardize each SNP row to zero mean and unit variance.
    mu = M.mean(axis=1, keepdims=True)
    sigma = M.std(axis=1, keepdims=True)
    sigma[sigma == 0] = 1.0
    Z = (M - mu) / sigma  # (m, n)

    kept_idx = []
    for i in range(m):
        if len(kept_idx) == 0:
            kept_idx.append(i)
        else:
            # Compute correlation only against already kept SNPs.
            z_i = Z[i]  # (n,)
            Z_kept = Z[kept_idx]  # (k, n)
            # corr = dot / n because rows are standardized.
            corr = (Z_kept @ z_i) / n
            r2 = corr * corr
            conflict_mask = r2 >= thr
            if not np.any(conflict_mask):
                kept_idx.append(i)
            else:
                # For high-LD redundancy, keep the SNP with larger MAF.
                conflict_pos = np.flatnonzero(conflict_mask)
                conflict_kept_idx = np.array([kept_idx[p] for p in conflict_pos], dtype=int)
                if conflict_kept_idx.size > 0 and float(maf[i]) > float(np.max(maf[conflict_kept_idx])):
                    for p in sorted(conflict_pos.tolist(), reverse=True):
                        kept_idx.pop(p)
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
    core:Literal['rf','gbdt']='gbdt',
    response:Literal['binary', 'continuous']="continuous",
    gsetmode:bool = True,
    gset_groups: Union[None, List[np.ndarray], List[list[int]]] = None,
):
    """
    Find logic combinations (xcombine) in a genotype block using
    random forest + permutation importance + logistic regression.
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

    # If gset_groups are provided, normalize and apply ldprune within each group.
    if gset_groups is not None:
        group_indices: List[np.ndarray] = []
        for g in gset_groups:
            arr = np.asarray(g, dtype=int)
            if arr.size > 0:
                group_indices.append(arr)
        if len(group_indices) == 0:
            return None

        kept_union: List[int] = []
        for g_idx in group_indices:
            g_idx = g_idx[(g_idx >= 0) & (g_idx < M.shape[0])]
            if g_idx.size == 0:
                continue
            M_sub = M[g_idx]
            sites_sub = sites[g_idx]
            M_kept, sites_kept = ldprune(M_sub, sites_sub)
            # Map kept indices back to global indices via site strings
            kept_set = set(sites_kept.tolist())
            kept_global = [int(i) for i, s in enumerate(sites) if s in kept_set]
            kept_union.extend(kept_global)

        kept_union = sorted(set(kept_union))
        if len(kept_union) == 0:
            return None
        M = M[kept_union]
        sites = sites[kept_union]
        # Remap group indices to new local indices
        index_map = {old: new for new, old in enumerate(kept_union)}
        gset_groups = [
            np.array([index_map[i] for i in g if i in index_map], dtype=int)
            for g in gset_groups
        ]
        gset_groups = [g for g in gset_groups if g.size > 0]
        if len(gset_groups) == 0:
            return None
    else:
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
    if gsetmode and gset_groups is not None:
        order = np.argsort(Imp)[::-1]
        idx = []
        for g in gset_groups:
            if g.size == 0:
                continue
            # pick highest-importance SNP within this group
            best = g[np.argmax(Imp[g])]
            idx.append(best)
        idx = np.array(sorted(set(idx)), dtype=int)[:topk]  # at most k interaction tests
        if idx.size == 0:
            return None
    else:
        idx = np.argsort(Imp)[::-1][:topk]
    # 0/1/2 → 0/1
    Mchoice = (M[idx] / 2).astype(int).T

    resdict = logreg(Mchoice, y, response=response, tags=sites[idx])
    if resdict is None:
        return None
    indices = resdict.get("indices", [])
    if indices is None or len(indices) <= 1:
        return None
    if resdict.get("expression", "") == "1":
        return None
    return resdict


# ---------- Whole-genome traversal ----------
def _window_task(
    task: Tuple[str, int, int, np.ndarray, List[str]],
    y: np.ndarray,
    nsnp: int,
    n_estimators: int,
    response: Literal['binary', 'continuous'],
    gsetmode: bool,
) -> Union[dict[str,Any], None]:
    chrom, start, end, M_win, tags = task
    try:
        resdict = getLogicgate(
            y,
            M_win,
            nsnp=nsnp,
            n_estimators=n_estimators,
            sites=tags,
            response=response,
            gsetmode=gsetmode,
        )
    except Exception as e:
        print(f"[WARN] window {chrom}:{start}-{end} failed: {e}")
        return None
    if resdict is None:
        return None
    return resdict


def _resolve_stream_sample_subset(
    genofile: str,
    sampleid: np.ndarray,
) -> tuple[Union[list[str], None], Union[np.ndarray, None]]:
    """
    Keep streaming sample handling consistent with gwas.py:
    - For VCF input: read all samples in chunk reader, then subset by boolean mask.
    - For non-VCF input: pass sample_ids directly to chunk reader.
    """
    sample_arr = np.asarray(sampleid, dtype=str)
    low = str(genofile).lower()
    if low.endswith(".vcf") or low.endswith(".vcf.gz"):
        all_ids, _ = inspect_genotype_file(genofile)
        all_ids = np.asarray(all_ids, dtype=str)
        sameidx = np.isin(all_ids, sample_arr)
        if not np.any(sameidx):
            raise ValueError(
                "No overlapping samples between genotype input and selected sample IDs."
            )
        return None, sameidx
    return sample_arr.tolist(), None


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
    response: Literal['binary', 'continuous'] = "continuous",
    gsetmode: bool = False,
    threads: int = 4,
    batch_size: int = 128,
    mmap_window_mb: Union[int, None] = None,
):
    """
    Whole-genome sliding-window traversal (load by chunksize order to reduce repeated IO).

    Window definition:
      - Move window start by ``step`` each time
      - Each window interval is [start, start + windowsize)

    Streaming strategy:
      - Keep at most two windows of data in memory
      - After processing two windows, drop the first and load the next

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
    sample_sub, sameidx = _resolve_stream_sample_subset(genofile, sampleid)

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
        sample_ids=sample_sub,
        mmap_window_mb=mmap_window_mb,
    )

    try:
        for chunk, sites in chunks:
            if not sites:
                continue
            if sameidx is not None:
                chunk = chunk[:, sameidx]
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
                                            out for out in (_window_task(t, y, nsnp, n_estimators, response, gsetmode) for t in tasks)
                                            if out is not None
                                        ]
                                    else:
                                        batch = Parallel(n_jobs=threads, backend="loky")(
                                            delayed(_window_task)(t, y, nsnp, n_estimators, response, gsetmode) for t in tasks
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
                                    out for out in (_window_task(t, y, nsnp, n_estimators, response, gsetmode) for t in tasks)
                                    if out is not None
                                ]
                            else:
                                batch = Parallel(n_jobs=threads, backend="loky")(
                                    delayed(_window_task)(t, y, nsnp, n_estimators, response, gsetmode) for t in tasks
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
                                out for out in (_window_task(t, y, nsnp, n_estimators, response, gsetmode) for t in tasks)
                                if out is not None
                            ]
                        else:
                            batch = Parallel(n_jobs=threads, backend="loky")(
                                delayed(_window_task)(t, y, nsnp, n_estimators, response, gsetmode) for t in tasks
                            )
                            batch = [r for r in batch if r is not None]
                        results.extend(batch)
                        tasks.clear()
                window_start += step

        if tasks:
            if threads <= 1:
                batch = [
                    out for out in (_window_task(t, y, nsnp, n_estimators, response, gsetmode) for t in tasks)
                    if out is not None
                ]
            else:
                batch = Parallel(n_jobs=threads, backend="loky")(
                    delayed(_window_task)(t, y, nsnp, n_estimators, response, gsetmode) for t in tasks
                )
                batch = [r for r in batch if r is not None]
            results.extend(batch)
    finally:
        pbar.close()

    return results


# ---------- Custom traversal ----------
chrposTuple = Tuple[str, int, int]
def _process(
    ChromPos: Union[chrposTuple,List[chrposTuple]],
    genofile: str,
    sample_sub: Union[list[str], None],
    sameidx: Union[np.ndarray, None],
    y: np.ndarray,
    maf: float = 0.02,
    missing_rate: float = 0.05,
    nsnp: int = 5,
    n_estimators: int = 200,
    response: Literal['binary', 'continuous'] = "continuous",
    gsetmode: bool = True,
    mmap_window_mb: Union[int, None] = None,
):
    """
    Run on one BED region:
      1) Extract all SNP genotypes from ``genofile`` in the region
      2) Call ``getLogicgate``
      3) Return ``(xcombine, site_tuple)`` or ``None``
    """
    # mmap_window_mb is not supported with bim_range; disable it here
    mmap_window_mb = None
    if isinstance(ChromPos,tuple):
        chrom, start, end = ChromPos
        chunks = load_genotype_chunks(
            genofile,
            chunk_size=int(1e6),
            maf=maf,
            missing_rate=missing_rate,
            impute=True,
            bim_range=(str(chrom), int(start), int(end)),
            sample_ids=sample_sub,
        )

        M_list = []
        sites_list = []

        for chunk, sites in chunks:
            # chunk: (m_chunk, n_samples)
            if sameidx is not None:
                chunk = chunk[:, sameidx]
            M_list.append(chunk)
            sites_list.extend(sites)
        if not M_list:
            return None
    elif isinstance(ChromPos,list):
        M_list = []
        sites_list = []
        gset_groups: List[np.ndarray] = []
        offset = 0
        # limit per-gene SNPs to bound memory
        max_gene_snps = max(200, int(nsnp) * 60)
        for chrom, start, end in ChromPos:
            chunks = load_genotype_chunks(
                genofile,
                chunk_size=int(1e6),
                maf=maf,
                missing_rate=missing_rate,
                impute=True,
                bim_range=(str(chrom), int(start), int(end)),
                sample_ids=sample_sub,
            )
            M_gene = None
            sites_gene: List[Any] = []
            for chunk, sites in chunks:
                if chunk.size == 0:
                    continue
                if sameidx is not None:
                    chunk = chunk[:, sameidx]
                if M_gene is None:
                    M_gene = chunk
                    sites_gene = list(sites)
                else:
                    M_gene = np.vstack([M_gene, chunk])
                    sites_gene.extend(sites)
                # prune periodically to bound memory per gene
                if M_gene.shape[0] > max_gene_snps:
                    M_gene, sites_kept = ldprune(
                        M_gene, np.array(sites_gene), max_snps=max_gene_snps
                    )
                    sites_gene = list(sites_kept)
            if M_gene is None or M_gene.shape[0] == 0:
                continue
            # final prune to limit per-gene SNPs
            if M_gene.shape[0] > max_gene_snps:
                M_gene, sites_kept = ldprune(
                    M_gene, np.array(sites_gene), max_snps=max_gene_snps
                )
                sites_gene = list(sites_kept)
            if M_gene.shape[0] == 0:
                continue

            M_list.append(M_gene)
            sites_list.extend(sites_gene)
            gset_groups.append(np.arange(offset, offset + M_gene.shape[0], dtype=int))
            offset += M_gene.shape[0]
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
            response=response,
            gsetmode=gsetmode,
            gset_groups=gset_groups if gsetmode and isinstance(ChromPos, list) else None,
        )
    except Exception as e:
        # If one region fails, do not crash the whole parallel job.
        print(f"[WARN] process {ChromPos} failed: {e}")
        return None
    if resdict is None:
        return None
    return resdict

def _build_site_index(
    sites: List[Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build per-chromosome positional index for fast range query.
    Returns {chrom: (pos, global_indices)} in original SNP order.
    """
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for i, site in enumerate(sites):
        chrom = str(site.chrom)
        pos = int(site.pos)
        if chrom not in by_chrom:
            by_chrom[chrom] = []
        by_chrom[chrom].append((pos, i))

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, pairs in by_chrom.items():
        pos = np.fromiter((p for p, _ in pairs), dtype=np.int64, count=len(pairs))
        idx = np.fromiter((j for _, j in pairs), dtype=np.int64, count=len(pairs))
        out[chrom] = (pos, idx)
    return out

def _select_range_indices(
    site_index: dict[str, tuple[np.ndarray, np.ndarray]],
    chrom: str,
    start: int,
    end: int,
) -> np.ndarray:
    item = site_index.get(str(chrom))
    if item is None:
        return np.empty(0, dtype=np.int64)
    pos, idx = item
    if pos.size > 1 and np.any(pos[1:] < pos[:-1]):
        mask = (pos >= int(start)) & (pos <= int(end))
        return idx[mask]
    lo = int(np.searchsorted(pos, int(start), side="left"))
    hi = int(np.searchsorted(pos, int(end), side="right"))
    if hi <= lo:
        return np.empty(0, dtype=np.int64)
    return idx[lo:hi]

def _process_inmemory(
    ChromPos: Union[chrposTuple, List[chrposTuple]],
    all_genotype_i8: np.ndarray,
    all_sites: List[Any],
    site_index: dict[str, tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    nsnp: int = 5,
    n_estimators: int = 200,
    response: Literal['binary', 'continuous'] = "continuous",
    gsetmode: bool = True,
):
    if isinstance(ChromPos, tuple):
        chrom, start, end = ChromPos
        idx = _select_range_indices(site_index, str(chrom), int(start), int(end))
        if idx.size == 0:
            return None
        M = all_genotype_i8[idx]
        sites_list = [all_sites[int(i)] for i in idx]
        gset_groups = None
    elif isinstance(ChromPos, list):
        M_list: List[np.ndarray] = []
        sites_list: List[Any] = []
        gset_groups: List[np.ndarray] = []
        offset = 0
        max_gene_snps = max(200, int(nsnp) * 60)

        for chrom, start, end in ChromPos:
            idx = _select_range_indices(site_index, str(chrom), int(start), int(end))
            if idx.size == 0:
                continue
            M_gene = all_genotype_i8[idx]
            sites_gene: List[Any] = [all_sites[int(i)] for i in idx]
            if M_gene.shape[0] > max_gene_snps:
                M_gene, sites_kept = ldprune(
                    M_gene, np.array(sites_gene), max_snps=max_gene_snps
                )
                sites_gene = list(sites_kept)
            if M_gene.shape[0] == 0:
                continue

            M_list.append(M_gene)
            sites_list.extend(sites_gene)
            gset_groups.append(np.arange(offset, offset + M_gene.shape[0], dtype=int))
            offset += M_gene.shape[0]

        if not M_list:
            return None
        M = np.vstack(M_list)
    else:
        return None

    if M.shape[0] < 2:
        return None
    try:
        resdict = getLogicgate(
            y,
            M,
            nsnp=nsnp,
            n_estimators=n_estimators,
            sites=sites_list,
            response=response,
            gsetmode=gsetmode,
            gset_groups=gset_groups if gsetmode and isinstance(ChromPos, list) else None,
        )
    except Exception as e:
        print(f"[WARN] process {ChromPos} failed: {e}")
        return None
    if resdict is None:
        return None
    return resdict

# ---------- Main (collect results in one pass) ----------

def main(
    genofile,
    sampleid,
    y,
    bedlist,
    maf: float = 0.02,
    missing_rate: float = 0.05,
    nsnp=5,
    n_estimators=200,
    threads=4,
    response: Literal['binary', 'continuous'] = "continuous",
    gsetmode: bool = True,
    gsetmodes: Optional[List[bool]] = None,
    mmap_window_mb: Union[int, None] = None,
):
    y = np.asarray(y, dtype=float).ravel()
    if gsetmodes is not None and len(gsetmodes) != len(bedlist):
        raise ValueError("gsetmodes length must match bedlist length")
    sample_sub, sameidx = _resolve_stream_sample_subset(genofile, sampleid)
    results = Parallel(n_jobs=threads)(
        delayed(_process)(
            ChromPos,
            genofile,
            sample_sub,
            sameidx,
            y,
            maf,
            missing_rate,
            nsnp,
            n_estimators,
            response,
            gsetmodes[i] if gsetmodes is not None else gsetmode,
            mmap_window_mb,
        )
        for i, ChromPos in enumerate(tqdm(bedlist))
    )
    return results

def main_inmemory(
    all_genotype_i8: np.ndarray,
    all_sites: List[Any],
    y: np.ndarray,
    bedlist,
    nsnp: int = 5,
    n_estimators: int = 200,
    threads: int = 4,
    response: Literal['binary', 'continuous'] = "continuous",
    gsetmode: bool = True,
    gsetmodes: Optional[List[bool]] = None,
):
    """
    Run GARFIELD search from preloaded in-memory int8 genotype matrix.
    """
    y = np.asarray(y, dtype=float).ravel()
    all_genotype_i8 = np.asarray(all_genotype_i8, dtype=np.int8)
    if all_genotype_i8.ndim != 2:
        raise ValueError("all_genotype_i8 must be a 2D SNP-major matrix.")
    if len(all_sites) != all_genotype_i8.shape[0]:
        raise ValueError("all_sites length must match all_genotype_i8.shape[0].")
    if gsetmodes is not None and len(gsetmodes) != len(bedlist):
        raise ValueError("gsetmodes length must match bedlist length")

    site_index = _build_site_index(all_sites)
    results = Parallel(n_jobs=threads, backend="threading")(
        delayed(_process_inmemory)(
            ChromPos,
            all_genotype_i8,
            all_sites,
            site_index,
            y,
            nsnp,
            n_estimators,
            response,
            gsetmodes[i] if gsetmodes is not None else gsetmode,
        )
        for i, ChromPos in enumerate(tqdm(bedlist))
    )
    return results
