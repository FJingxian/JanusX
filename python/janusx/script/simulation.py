"""
JanusX Simulation CLI

Mode
- Simulate phenotype from existing genotype: use --vcf, --file or --bfile

Outputs
- Phenotypes (always):
  <out>/<prefix>.pheno        : 3 columns (FID, IID, PHENO)
  <out>/<prefix>.pheno.txt    : 2 columns (IID, PHENO)
  <out>/<prefix>.pheno.NA.txt : 2 columns (IID, PHENO) with ~10% NA

- Causal sites (optional):
  <out>/<prefix>.causal.sites.tsv
"""

import os
import time
import socket
import argparse
from typing import Literal, Optional, List, Tuple, Any

import numpy as np
from janusx.garfield.garfield2 import ldprune
from janusx.gfreader import load_genotype_chunks
from janusx.gfreader import inspect_genotype_file
from ._common.log import setup_logging
from ._common.config_render import emit_cli_configuration
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    format_path_for_display,
    ensure_plink_prefix_exists,
)
from ._common.genocache import configure_genotype_cache_from_out
from ._common.genoio import determine_genotype_source_from_args as determine_genotype_source
from ._common.status import log_success


def _normalize_chrom(chrom: Any) -> str:
    s = str(chrom).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    return s.upper()


def _site_to_chr_pos(site: Any) -> Tuple[str, int]:
    if hasattr(site, "chrom") and hasattr(site, "pos"):
        return str(site.chrom), int(site.pos)
    if isinstance(site, (tuple, list)) and len(site) >= 2:
        return str(site[0]), int(site[1])
    s = str(site)
    if "_" in s:
        chrom, pos = s.rsplit("_", 1)
        try:
            return str(chrom), int(pos)
        except Exception:
            pass
    return str(site), 0


def _extract_window_matrix(
    gfile: str,
    *,
    window: Tuple[str, int, int],
    chunk_size: int,
    maf: float,
    missing_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    m_list: List[np.ndarray] = []
    sites_list: List[Any] = []
    for m_chunk, sites in load_genotype_chunks(
        gfile,
        chunk_size=chunk_size,
        maf=maf,
        missing_rate=missing_rate,
        bim_range=window,
    ):
        if m_chunk.size == 0:
            continue
        m_list.append(np.asarray(m_chunk, dtype=np.float32))
        sites_list.extend(list(sites))
    if not m_list:
        return np.empty((0, 0), dtype=np.float32), np.asarray([], dtype=object)
    m = np.vstack(m_list)
    s = np.asarray(sites_list, dtype=object)
    return m, s


def _select_low_ld_indices(
    g01: np.ndarray,
    *,
    k: int,
    ld_max: float,
    rng: np.random.Generator,
    n_trials: int = 64,
) -> Optional[np.ndarray]:
    """
    Randomized greedy selection under pairwise r^2 <= ld_max.
    Returns row indices on g01 or None.
    """
    m, n = g01.shape
    if k <= 0 or m < k or n <= 1:
        return None

    x = np.asarray(g01, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True, ddof=0)
    keep = np.asarray(sd[:, 0] > 1e-8, dtype=bool)
    if int(np.sum(keep)) < k:
        return None

    idx_map = np.flatnonzero(keep)
    z = (x[keep] - mu[keep]) / (sd[keep] + 1e-8)
    mm = z.shape[0]

    if ld_max >= 0.999:
        pick = rng.choice(np.arange(mm), size=k, replace=False)
        return idx_map[np.asarray(pick, dtype=int)]

    for _ in range(max(1, int(n_trials))):
        order = rng.permutation(mm)
        chosen: List[int] = []
        for cand in order:
            if not chosen:
                chosen.append(int(cand))
            else:
                # corr of candidate against chosen rows; rows are standardized.
                corr = np.dot(z[np.asarray(chosen, dtype=int)], z[cand]) / float(n)
                r2 = np.asarray(corr, dtype=np.float64) ** 2
                if float(np.max(r2)) <= float(ld_max) + 1e-12:
                    chosen.append(int(cand))
            if len(chosen) >= k:
                return idx_map[np.asarray(chosen[:k], dtype=int)]
    return None


def _collapse_to_bin02_with_het_filter(
    m: np.ndarray,
    *,
    het_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert genotype rows to BIN02 style for logic simulation:
    1) drop rows with heterozygote rate > het_max;
    2) for kept rows, replace genotype==1 with row-wise mode in {0,2}.
    Returns
    -------
    g02 : np.ndarray (m_keep, n), dtype=np.uint8 with values in {0,2}
    keep_idx : np.ndarray (m_keep,), indices on original row axis.
    """
    x = np.asarray(m, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.empty((0, 0), dtype=np.uint8), np.asarray([], dtype=int)

    keep_rows: List[np.ndarray] = []
    keep_idx: List[int] = []
    het_cap = float(max(0.0, min(1.0, het_max)))

    for ri in range(int(x.shape[0])):
        row = np.asarray(x[ri, :], dtype=np.float32).reshape(-1)
        valid = np.isfinite(row) & (row >= 0.0)
        if int(np.sum(valid)) == 0:
            continue
        gv = np.asarray(np.rint(row[valid]), dtype=np.int16)
        gv = np.where(gv <= 0, 0, np.where(gv >= 2, 2, 1)).astype(np.int16, copy=False)
        het_rate = float(np.mean(gv == 1))
        if het_rate > het_cap:
            continue
        c0 = int(np.sum(gv == 0))
        c2 = int(np.sum(gv == 2))
        mode02 = 2 if c2 > c0 else 0

        out_row = np.full(row.shape[0], mode02, dtype=np.uint8)
        gv_fill = np.where(gv == 1, mode02, gv).astype(np.uint8, copy=False)
        out_row[valid] = gv_fill
        keep_rows.append(out_row)
        keep_idx.append(ri)

    if len(keep_rows) == 0:
        return np.empty((0, x.shape[1]), dtype=np.uint8), np.asarray([], dtype=int)
    return np.vstack(keep_rows), np.asarray(keep_idx, dtype=int)


def _collect_low_het_pool_global(
    gfile: str,
    *,
    chunk_size: int,
    maf: float,
    missing_rate: float,
    het_max: float,
    max_sites: int = 20000,
) -> Tuple[List[np.ndarray], List[Tuple[str, int, int]]]:
    rows: List[np.ndarray] = []
    sites_out: List[Tuple[str, int, int]] = []
    seen: set[Tuple[str, int]] = set()

    for m_chunk, sites in load_genotype_chunks(
        gfile,
        chunk_size=int(chunk_size),
        maf=float(maf),
        missing_rate=float(missing_rate),
        impute=False,
    ):
        g02, keep_idx = _collapse_to_bin02_with_het_filter(
            np.asarray(m_chunk, dtype=np.float32),
            het_max=float(het_max),
        )
        if g02.size == 0 or len(keep_idx) == 0:
            continue
        g01 = (np.asarray(g02, dtype=np.uint8) > 0).astype(np.uint8)
        s_arr = np.asarray(sites, dtype=object)
        for li in range(int(g01.shape[0])):
            oi = int(np.asarray(keep_idx, dtype=int)[li])
            chrom, pos = _site_to_chr_pos(s_arr[oi])
            key = (str(chrom), int(pos))
            if key in seen:
                continue
            seen.add(key)
            rows.append(np.asarray(g01[li, :], dtype=np.uint8).copy())
            sites_out.append((str(chrom), int(pos), int(pos)))
            if len(rows) >= int(max_sites):
                return rows, sites_out
    return rows, sites_out


def _sample_gate_from_pool(
    g01_pool: np.ndarray,
    pool_sites: List[Tuple[str, int, int]],
    *,
    rng: np.random.Generator,
    k_min: int,
    k_max: int,
    ld_max: float,
    af_min: float,
    af_max: float,
    same_chrom: bool = True,
    n_trials: int = 256,
) -> Optional[Tuple[np.ndarray, List[Tuple[str, int, int]], float]]:
    m_pool = int(g01_pool.shape[0])
    if m_pool < max(2, int(k_min)):
        return None

    af_center = 0.5 * (float(af_min) + float(af_max))
    best: Optional[Tuple[np.ndarray, List[Tuple[str, int, int]], float]] = None

    # Enforce same-chromosome gate by sampling within one chromosome pool.
    chrom_to_idx: dict[str, np.ndarray] = {}
    if bool(same_chrom):
        tmp: dict[str, List[int]] = {}
        for i, s in enumerate(pool_sites):
            c = _normalize_chrom(s[0])
            tmp.setdefault(c, []).append(int(i))
        for c, idxs in tmp.items():
            arr = np.asarray(idxs, dtype=int)
            if int(arr.size) >= max(2, int(k_min)):
                chrom_to_idx[c] = arr
        if len(chrom_to_idx) == 0:
            return None

    for _ in range(max(1, int(n_trials))):
        if bool(same_chrom):
            chrom = str(rng.choice(list(chrom_to_idx.keys())))
            idx_pool = np.asarray(chrom_to_idx[chrom], dtype=int)
            g_pool = np.asarray(g01_pool[idx_pool, :], dtype=np.uint8)
            k_hi = min(int(k_max), int(idx_pool.size))
            k_lo = max(2, min(int(k_min), k_hi))
            if k_lo > k_hi:
                continue
            k = int(rng.integers(k_lo, k_hi + 1))
            sel_local = _select_low_ld_indices(
                g_pool,
                k=k,
                ld_max=float(ld_max),
                rng=rng,
                n_trials=64,
            )
            if sel_local is None:
                sel_local = np.asarray(rng.choice(np.arange(int(idx_pool.size)), size=k, replace=False), dtype=int)
            sel = np.asarray(idx_pool[np.asarray(sel_local, dtype=int)], dtype=int)
        else:
            k_hi = min(int(k_max), m_pool)
            k_lo = max(2, min(int(k_min), k_hi))
            if k_lo > k_hi:
                return None
            k = int(rng.integers(k_lo, k_hi + 1))
            sel = _select_low_ld_indices(
                g01_pool,
                k=k,
                ld_max=float(ld_max),
                rng=rng,
                n_trials=64,
            )
            if sel is None:
                sel = np.asarray(rng.choice(np.arange(m_pool), size=k, replace=False), dtype=int)
        gate = np.all(g01_pool[np.asarray(sel, dtype=int), :] > 0, axis=0).astype(np.float32).reshape(-1, 1)
        var_gate = float(np.var(gate, ddof=0))
        if var_gate <= 1e-12:
            continue
        af = float(np.mean(gate))
        sel_sites = [pool_sites[int(i)] for i in np.asarray(sel, dtype=int)]
        cand = (gate, sel_sites, af)
        if float(af_min) <= af <= float(af_max):
            return cand
        if best is None or abs(af - af_center) < abs(best[2] - af_center):
            best = cand
    return best


# -----------------------------
# Phenotype simulator from genotype stream
# -----------------------------
def simulate_phenotype_from_genofile(
    gfile: str,
    mode: Literal["single", "garfield"] = "single",
    chunk_size: int = 100_000,
    seed: int = 1,
    maf: float = 0.02,
    missing_rate: float = 0.05,
    pve: float = 0.5,
    ve: float = 1.0,
    windows: int = 50_000,
    and_k_min: int = 2,
    and_k_max: int = 4,
    and_ld_max: float = 0.2,
    and_het_max: float = 0.05,
    and_af_min: float = 0.02,
    and_af_max: float = 0.98,
    and_target_pve: float = 0.2,
    and_max_iter: int = 100,
) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
    """
    Simulate phenotype y (n,1) from an existing genotype file.

    Model
    - load_genotype_chunks(...) yields Mchunk with shape (m, n) = (SNPs, samples)
    - Polygenic background:
        g = sum_j beta_j * Z_j
      where Z_j is standardized per SNP and Var(beta_j) ≈ vg/nsnp
    - Fixed parameters:
        mean = 10.0
      pve and ve are configurable via CLI (defaults: 0.5 and 1.0).
    """
    rng = np.random.default_rng(seed)

    # Fixed parameters (not exposed via CLI)
    mean = 10.0

    sampleid, nsnp = inspect_genotype_file(gfile)
    n = len(sampleid)

    # Residuals: e ~ N(0, ve) plus constant mean
    y = rng.normal(0.0, np.sqrt(ve), size=(n, 1)) + mean

    vg = pve * ve / (1.0 - pve)

    siteslist: List[Tuple[str, int, int]] = []

    # Polygenic background
    for Mchunk, sites in load_genotype_chunks(
        gfile,
        chunk_size=chunk_size,
        maf=maf,
        missing_rate=missing_rate,
    ):
        # Mchunk: (m, n)
        Mchunk = Mchunk.astype(np.float32, copy=False)

        # Standardize each SNP across samples
        mu = Mchunk.mean(axis=1, keepdims=True)
        sd = Mchunk.std(axis=1, keepdims=True, ddof=0) + 1e-6
        Z = (Mchunk - mu) / sd

        m = Z.shape[0]
        # beta_j ~ N(0, vg/nsnp)
        beta = rng.normal(0.0, np.sqrt(vg / nsnp), size=(m, 1)).astype(np.float32)
        g = (beta.T @ Z).T  # (n,1)
        y += g

        if mode == "single":
            siteslist.extend([(str(i.chrom), int(i.pos), int(i.pos)) for i in sites])
        else:
            siteslist.extend([(str(i.chrom), int(i.pos) - windows, int(i.pos) + windows) for i in sites])

    # Pick a target SNP or window
    arr = np.array(siteslist, dtype=object)
    siteslistrc = tuple(arr[rng.integers(len(arr))])
    outsites: List[Tuple[str, int, int]] = []

    if mode == "single":
        # Add the target SNP effect (simple: effect size = 1)
        for Mchunk, sites in load_genotype_chunks(
            gfile,
            chunk_size=chunk_size,
            maf=maf,
            missing_rate=missing_rate,
            bim_range=siteslistrc,
        ):
            Mchunk = Mchunk.astype(np.float32, copy=False)  # shape (m, n) or (1, n)
            y += Mchunk.T  # -> (n,1)
            outsites.extend([(str(i.chrom), int(i.pos), int(i.pos)) for i in sites])

    else:
        # Strict AND logic simulation:
        # 1) sample multiple low-LD loci in a random window
        # 2) construct gate = X1 & X2 & ... & Xk
        # 3) scale effect size to target PVE
        outsites = []
        best_candidate: Optional[Tuple[np.ndarray, List[Tuple[str, int, int]], float]] = None
        af_target_center = 0.5 * (float(and_af_min) + float(and_af_max))
        low_het_pool_rows: List[np.ndarray] = []
        low_het_pool_sites: List[Tuple[str, int, int]] = []
        low_het_pool_seen: set[Tuple[str, int]] = set()

        if arr.shape[0] == 0:
            return y.astype(np.float32), outsites

        for _ in range(max(1, int(and_max_iter))):
            win = tuple(arr[rng.integers(len(arr))])  # (chrom, start, end)
            m_raw, s_raw = _extract_window_matrix(
                gfile,
                window=win,  # type: ignore[arg-type]
                chunk_size=chunk_size,
                maf=maf,
                missing_rate=missing_rate,
            )
            if m_raw.size == 0 or s_raw.size == 0:
                continue

            # Keep only loosely non-redundant variants before strict low-LD sampling.
            m_keep, s_keep = ldprune(
                np.asarray(m_raw, dtype=np.float32),
                np.asarray(s_raw, dtype=object),
                float(min(max(and_ld_max, 0.01), 0.95)),
            )
            g02, keep_idx = _collapse_to_bin02_with_het_filter(
                np.asarray(m_keep, dtype=np.float32),
                het_max=float(and_het_max),
            )
            g01 = (np.asarray(g02, dtype=np.uint8) > 0).astype(np.uint8)
            m = int(g01.shape[0])
            if m > 0:
                s_keep_arr = np.asarray(s_keep, dtype=object)
                for li in range(m):
                    oi = int(np.asarray(keep_idx, dtype=int)[li])
                    chrom, pos = _site_to_chr_pos(s_keep_arr[oi])
                    key = (str(chrom), int(pos))
                    if key in low_het_pool_seen:
                        continue
                    low_het_pool_seen.add(key)
                    low_het_pool_rows.append(np.asarray(g01[li, :], dtype=np.uint8).copy())
                    low_het_pool_sites.append((str(chrom), int(pos), int(pos)))
            if m < max(2, int(and_k_min)):
                continue

            k_hi = min(int(and_k_max), m)
            k_lo = max(2, min(int(and_k_min), k_hi))
            if k_lo > k_hi:
                continue
            k = int(rng.integers(k_lo, k_hi + 1))

            sel = _select_low_ld_indices(
                g01,
                k=k,
                ld_max=float(and_ld_max),
                rng=rng,
                n_trials=64,
            )
            if sel is None:
                # Fallback: keep strict AND but relax low-LD selection when window is too tight.
                sel = np.asarray(rng.choice(np.arange(m), size=k, replace=False), dtype=int)

            gate = np.all(g01[np.asarray(sel, dtype=int), :] > 0, axis=0).astype(np.float32).reshape(-1, 1)
            af = float(np.mean(gate))
            sel_sites: List[Tuple[str, int, int]] = []
            for si in np.asarray(sel, dtype=int):
                orig_i = int(np.asarray(keep_idx, dtype=int)[int(si)])
                chrom, pos = _site_to_chr_pos(np.asarray(s_keep, dtype=object)[orig_i])
                sel_sites.append((str(chrom), int(pos), int(pos)))

            if best_candidate is None or abs(af - af_target_center) < abs(best_candidate[2] - af_target_center):
                best_candidate = (gate, sel_sites, af)

            if float(and_af_min) <= af <= float(and_af_max):
                var_gate = float(np.var(gate, ddof=0))
                if var_gate <= 1e-12:
                    continue
                var_base = float(np.var(y, ddof=0))
                target = float(and_target_pve)
                target_var = (target / max(1e-12, 1.0 - target)) * max(var_base, 1e-12)
                beta = float(np.sqrt(target_var / var_gate))
                y += beta * (gate - float(np.mean(gate)))
                outsites = sel_sites
                break
        else:
            # If no gate meets AF constraints, use the best available strict-AND candidate.
            if best_candidate is not None:
                gate, sel_sites, _af = best_candidate
                var_gate = float(np.var(gate, ddof=0))
                if var_gate > 1e-12:
                    var_base = float(np.var(y, ddof=0))
                    target = float(and_target_pve)
                    target_var = (target / max(1e-12, 1.0 - target)) * max(var_base, 1e-12)
                    beta = float(np.sqrt(target_var / var_gate))
                    y += beta * (gate - float(np.mean(gate)))
                    outsites = sel_sites
            elif len(low_het_pool_rows) >= max(2, int(and_k_min)):
                # Fallback: still enforce low-het BIN02 logic, but combine from pooled
                # low-het variants when no sampled window can satisfy all constraints.
                pool = np.vstack(low_het_pool_rows).astype(np.uint8, copy=False)
                cand = _sample_gate_from_pool(
                    pool,
                    low_het_pool_sites,
                    rng=rng,
                    k_min=int(and_k_min),
                    k_max=int(and_k_max),
                    ld_max=float(and_ld_max),
                    af_min=float(and_af_min),
                    af_max=float(and_af_max),
                    same_chrom=True,
                    n_trials=256,
                )
                if cand is not None:
                    gate, sel_sites, _af = cand
                    var_gate = float(np.var(gate, ddof=0))
                    if var_gate > 1e-12:
                        var_base = float(np.var(y, ddof=0))
                        target = float(and_target_pve)
                        target_var = (target / max(1e-12, 1.0 - target)) * max(var_base, 1e-12)
                        beta = float(np.sqrt(target_var / var_gate))
                        y += beta * (gate - float(np.mean(gate)))
                        outsites = sel_sites

        if len(outsites) == 0:
            # Final fallback: global streaming pool under the same low-het BIN02 rule.
            g_rows, g_sites = _collect_low_het_pool_global(
                gfile,
                chunk_size=int(chunk_size),
                maf=float(maf),
                missing_rate=float(missing_rate),
                het_max=float(and_het_max),
                max_sites=20000,
            )
            if len(g_rows) >= max(2, int(and_k_min)):
                pool = np.vstack(g_rows).astype(np.uint8, copy=False)
                cand = _sample_gate_from_pool(
                    pool,
                    g_sites,
                    rng=rng,
                    k_min=int(and_k_min),
                    k_max=int(and_k_max),
                    ld_max=float(and_ld_max),
                    af_min=float(and_af_min),
                    af_max=float(and_af_max),
                    same_chrom=True,
                    n_trials=512,
                )
                if cand is not None:
                    gate, sel_sites, _af = cand
                    var_gate = float(np.var(gate, ddof=0))
                    if var_gate > 1e-12:
                        var_base = float(np.var(y, ddof=0))
                        target = float(and_target_pve)
                        target_var = (target / max(1e-12, 1.0 - target)) * max(var_base, 1e-12)
                        beta = float(np.sqrt(target_var / var_gate))
                        y += beta * (gate - float(np.mean(gate)))
                        outsites = sel_sites

    return y.astype(np.float32), outsites


# -----------------------------
# Phenotype writer
# -----------------------------
def write_phenotypes(
    outprefix: str,
    sample_ids: np.ndarray,
    y: np.ndarray,
    seed: int = 1,
):
    """
    Write phenotype files:
    - <outprefix>.pheno        (FID IID PHENO)
    - <outprefix>.pheno.txt    (IID PHENO)
    - <outprefix>.pheno.NA.txt (IID PHENO, ~10% NA)
    """
    rng = np.random.default_rng(seed)

    sample_ids = np.asarray(sample_ids, dtype=str)
    y = y.reshape(-1)

    # 3 columns: PLINK-style
    pheno3 = np.empty((len(sample_ids), 3), dtype=object)
    pheno3[:, 0] = sample_ids
    pheno3[:, 1] = sample_ids
    pheno3[:, 2] = y
    np.savetxt(
        f"{outprefix}.pheno",
        pheno3,
        delimiter="\t",
        fmt=["%s", "%s", "%.6f"],
    )

    # 2 columns: simple format
    pheno2 = np.column_stack([sample_ids, y.astype(object)])
    np.savetxt(
        f"{outprefix}.pheno.txt",
        pheno2,
        delimiter="\t",
        fmt=["%s", "%.6f"],
        header="IID\tPHENO",
        comments="",
    )

    # Fixed 10% NA
    na_rate = 0.1
    pheno2_na = pheno2.astype(object, copy=True)
    n = len(sample_ids)
    k = int(round(n * na_rate))
    if k > 0:
        idx = rng.choice(n, size=k, replace=False)
        pheno2_na[idx, 1] = "NA"
    np.savetxt(
        f"{outprefix}.pheno.NA.txt",
        pheno2_na,
        delimiter="\t",
        fmt=["%s", "%s"],
        header="IID\tPHENO",
        comments="",
    )


def write_sites(outprefix: str, sites: List[Tuple[str, int, int]]):
    if not sites:
        return
    path = f"{outprefix}.causal.sites.tsv"
    arr = np.array(sites, dtype=object)
    np.savetxt(path, arr, delimiter="\t", fmt=["%s", "%d", "%d"])


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx sim",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx sim -bfile geno_prefix -o out -prefix demo",
            "jx sim -file geno_prefix -o out -prefix demo",
            "jx sim -hmp geno.hmp.gz -o out -prefix demo",
            "jx simulation -vcf geno.vcf.gz -mode single -o out",
        ]),
        description="JanusX simulation: phenotype from existing genotype",
    )

    required_group = parser.add_argument_group("Required arguments")
    optional_group = parser.add_argument_group("Optional arguments")

    # Genotype input (required)
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument(
        "-vcf", "--vcf", type=str,
        help="Input genotype file in VCF format (.vcf or .vcf.gz).",
    )
    geno_group.add_argument(
        "-hmp", "--hmp", type=str,
        help="Input genotype file in HMP format (.hmp or .hmp.gz).",
    )
    geno_group.add_argument(
        "-bfile", "--bfile", type=str,
        help="Input genotype in PLINK binary format (prefix for .bed, .bim, .fam).",
    )
    geno_group.add_argument(
        "-file", "--file", type=str,
        help=(
            "Input genotype numeric matrix (.txt/.tsv/.csv/.npy) or prefix. "
            "Requires sibling prefix.id. Optional site metadata: prefix.site or prefix.bim."
        ),
    )

    # Common arguments
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory for results (default: %(default)s).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Prefix for output files (default: %(default)s).",
    )
    optional_group.add_argument(
        "-chunksize", "--chunksize", type=int, default=100_000,
        help="Number of SNPs per chunk for streaming (default: %(default)s).",
    )

    # Filters when reading genotype
    optional_group.add_argument(
        "-maf", "--maf", type=float, default=0.02,
        help="Exclude variants with minor allele frequency lower than a threshold "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-geno", "--geno", type=float, default=0.05,
        help="Exclude variants with missing call frequencies greater than a threshold "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-ve", "--ve", type=float, default=1.0,
        help="Environmental variance for phenotype simulation "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-pve", "--pve", type=float, default=0.5,
        help="Proportion of variance explained by genetic component "
             "(default: %(default)s).",
    )

    # Causal signal mode
    optional_group.add_argument(
        "-sm", "--sim-mode", type=str, default="single",
        choices=["single", "garfield"],
        help="Causal signal mode (default: %(default)s).",
    )
    optional_group.add_argument(
        "-windows", "--windows", type=int, default=50_000,
        help="Window size for garfield mode "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "--and-k-min", type=int, default=2,
        help="Minimum number of loci in strict AND gate for garfield mode (default: %(default)s).",
    )
    optional_group.add_argument(
        "--and-k-max", type=int, default=4,
        help="Maximum number of loci in strict AND gate for garfield mode (default: %(default)s).",
    )
    optional_group.add_argument(
        "--and-ld-max", type=float, default=0.2,
        help="Maximum pairwise r^2 among selected loci for strict AND gate (default: %(default)s).",
    )
    optional_group.add_argument(
        "--and-het-max", type=float, default=0.05,
        help="Maximum heterozygosity rate allowed before BIN02 collapse in strict AND gate (default: %(default)s).",
    )
    optional_group.add_argument(
        "--and-af-min", type=float, default=0.02,
        help="Minimum gate frequency for strict AND gate acceptance (default: %(default)s).",
    )
    optional_group.add_argument(
        "--and-af-max", type=float, default=0.98,
        help="Maximum gate frequency for strict AND gate acceptance (default: %(default)s).",
    )
    optional_group.add_argument(
        "--and-target-pve", type=float, default=0.2,
        help="Target PVE contributed by strict AND gate effect (default: %(default)s).",
    )
    optional_group.add_argument(
        "--and-max-iter", type=int, default=100,
        help="Maximum attempts to sample strict AND gate in garfield mode (default: %(default)s).",
    )
    optional_group.add_argument(
        "-write-sites", "--write-sites", action="store_true",
        help="Write causal sites file "
             "(default: %(default)s).",
    )
    optional_group.add_argument("--seed", type=int, default=None, help="Random seed.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # -------------------------
    # Existing genotype file
    # -------------------------
    gfile, prefix = determine_genotype_source(args)
    args.out = os.path.normpath(args.out if args.out is not None else ".")
    outprefix = os.path.join(args.out, prefix)
    os.makedirs(args.out, exist_ok=True, mode=0o755)
    configure_genotype_cache_from_out(args.out)

    log_path = f"{outprefix}.sim.log"
    logger = setup_logging(log_path)

    emit_cli_configuration(
        logger,
        app_title="JanusX - Simulation",
        config_title="SIMULATION CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "General",
                [
                    ("Genotype file", gfile),
                    ("Chunk size", args.chunksize),
                    ("MAF threshold", args.maf),
                    ("Missing threshold", args.geno),
                    ("Sim mode", args.sim_mode),
                    ("Windows", args.windows),
                    ("PVE", args.pve),
                    ("VE", args.ve),
                    ("AND k[min,max]", f"{args.and_k_min},{args.and_k_max}"),
                    ("AND ld max", args.and_ld_max),
                    ("AND het max", args.and_het_max),
                    ("AND af[min,max]", f"{args.and_af_min},{args.and_af_max}"),
                    ("AND target PVE", args.and_target_pve),
                    ("AND max iter", args.and_max_iter),
                    ("Write sites", args.write_sites),
                    ("Seed", args.seed),
                ],
            )
        ],
        footer_rows=[("Output prefix", outprefix)],
        line_max_chars=60,
    )

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    if not (0.0 < args.pve < 1.0):
        logger.error("--pve must be in (0, 1).")
        raise SystemExit(1)
    if int(args.and_k_min) < 2:
        logger.error("--and-k-min must be >= 2 for strict AND simulation.")
        raise SystemExit(1)
    if int(args.and_k_max) < int(args.and_k_min):
        logger.error("--and-k-max must be >= --and-k-min.")
        raise SystemExit(1)
    if not (0.0 <= float(args.and_ld_max) <= 1.0):
        logger.error("--and-ld-max must be in [0, 1].")
        raise SystemExit(1)
    if not (0.0 <= float(args.and_het_max) <= 1.0):
        logger.error("--and-het-max must be in [0, 1].")
        raise SystemExit(1)
    if not (0.0 <= float(args.and_af_min) <= 1.0) or not (0.0 <= float(args.and_af_max) <= 1.0):
        logger.error("--and-af-min/--and-af-max must be in [0, 1].")
        raise SystemExit(1)
    if float(args.and_af_min) > float(args.and_af_max):
        logger.error("--and-af-min must be <= --and-af-max.")
        raise SystemExit(1)
    if not (0.0 < float(args.and_target_pve) < 1.0):
        logger.error("--and-target-pve must be in (0, 1).")
        raise SystemExit(1)
    if int(args.and_max_iter) <= 0:
        logger.error("--and-max-iter must be > 0.")
        raise SystemExit(1)

    t_start = time.time()
    logger.info(f"Simulating phenotype from genotype: {gfile}")
    y, outsites = simulate_phenotype_from_genofile(
        gfile=gfile,
        mode=args.sim_mode,
        chunk_size=args.chunksize,
        seed=args.seed,
        maf=args.maf,
        missing_rate=args.geno,
        pve=args.pve,
        ve=args.ve,
        windows=args.windows,
        and_k_min=args.and_k_min,
        and_k_max=args.and_k_max,
        and_ld_max=args.and_ld_max,
        and_het_max=args.and_het_max,
        and_af_min=args.and_af_min,
        and_af_max=args.and_af_max,
        and_target_pve=args.and_target_pve,
        and_max_iter=args.and_max_iter,
    )

    sampleid, _nsnp = inspect_genotype_file(gfile)
    sampleid = np.asarray(sampleid, dtype=str)

    write_phenotypes(outprefix, sampleid, y, seed=args.seed)
    sites_written = False
    if args.write_sites:
        write_sites(outprefix, outsites)
        sites_written = len(outsites) > 0

    log_success(logger, f"Finished in {time.time() - t_start:.2f} s")
    logger.info("Done. Outputs:")
    logger.info(f"  {format_path_for_display(f'{outprefix}.pheno')}")
    logger.info(f"  {format_path_for_display(f'{outprefix}.pheno.txt')}")
    logger.info(f"  {format_path_for_display(f'{outprefix}.pheno.NA.txt')}")
    if args.write_sites and sites_written:
        logger.info(f"  {format_path_for_display(f'{outprefix}.causal.sites.tsv')}")
    elif args.write_sites:
        logger.warning("  causal sites were not generated under current constraints.")

    return 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    raise SystemExit(main())
