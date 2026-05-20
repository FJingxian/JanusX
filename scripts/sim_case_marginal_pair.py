#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from janusx import janusx as jxrs
from janusx.janusx import bed_ldblock_r2_rust, gwas_lmm_lm_null_lrt_decision, load_bed_u8_matrix
from janusx.pyBLUP.assoc import FastLMM


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CHUNK_PREFIX = (
    REPO_ROOT
    / "test.Tgarfield_out/chunks/cubic_All.maf0.02.chunk.chr1.L1.R1.36299099_36799099"
)
DEFAULT_OUT_DIR = REPO_ROOT / "sim.case"
DEFAULT_GRM = Path("~/cubic/cubic_All.maf0.02.cGRM.npy").expanduser()
DEFAULT_GRM_ID = Path("~/cubic/cubic_All.maf0.02.cGRM.npy.id").expanduser()
DEFAULT_SITE_STATS = None
DEFAULT_LDSC_WINDOW = "100kb"


@dataclass(frozen=True)
class SiteInfo:
    row_index: int
    chrom: str
    snp: str
    cm: float
    pos: int
    ref: str
    alt: str
    maf: float
    ldsc: float


@dataclass(frozen=True)
class PairCandidate:
    site1: SiteInfo
    site2: SiteInfo
    block1: int
    block2: int
    distance: int
    pair_r2: float
    dosage_corr: float
    pseudo_freq: float
    score: float


@dataclass(frozen=True)
class LdscWindowSpec:
    kind: str
    value: float
    label: str


def _out_path(prefix: Path, suffix: str) -> Path:
    return Path(f"{prefix}{suffix}")


def _env_positive_int(name: str) -> int | None:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _default_threads() -> int:
    for key in ("JANUSX_THREADS", "OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "NSLOTS", "LSB_DJOB_NUMPROC", "PBS_NP"):
        value = _env_positive_int(key)
        if value is not None:
            return value
    return 1


def _read_fam_ids(prefix: Path) -> list[str]:
    ids: list[str] = []
    with Path(f"{prefix}.fam").open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            parts = raw.strip().split()
            if parts:
                ids.append(parts[0])
    if not ids:
        raise ValueError(f"No sample IDs found in {prefix}.fam")
    return ids


def _read_bim_rows(prefix: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(f"{prefix}.bim").open("r", encoding="utf-8", errors="replace") as fh:
        for idx, raw in enumerate(fh):
            chrom, snp, cm, pos, ref, alt = raw.strip().split()[:6]
            rows.append(
                {
                    "row_index": idx,
                    "chrom": chrom,
                    "snp": snp,
                    "cm": float(cm),
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                }
            )
    if not rows:
        raise ValueError(f"No BIM rows found in {prefix}.bim")
    return rows


def _read_site_stats(path: Path) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            pos = int(str(row["pos"]))
            out[pos] = row
    if not out:
        raise ValueError(f"No site stats found in {path}")
    return out


def _parse_ldsc_window(text: str | None) -> LdscWindowSpec:
    raw = DEFAULT_LDSC_WINDOW if text is None or str(text).strip() == "" else str(text).strip().lower()
    compact = raw.replace(" ", "")
    import re

    m = re.fullmatch(r"([0-9]*\.?[0-9]+)([a-z]*)", compact)
    if m is None:
        raise ValueError(
            f"Invalid --ldsc-window: {text!r}. Use forms like 100, 100kb, 0.1mb, 100000b, or 10cm."
        )
    value = float(m.group(1))
    unit = str(m.group(2) or "")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"--ldsc-window must be > 0, got {text!r}.")

    if unit in {"", "snp", "snps"}:
        if not float(value).is_integer():
            raise ValueError(f"SNP-count LD-score window must be an integer, got {text!r}.")
        v = int(round(value))
        return LdscWindowSpec(kind="variants", value=float(v), label=f"{v}snp")
    if unit in {"b", "bp"}:
        bp = int(round(value))
        return LdscWindowSpec(kind="bp", value=float(bp), label=f"{bp}b")
    if unit == "kb":
        bp = int(round(value * 1000.0))
        return LdscWindowSpec(kind="bp", value=float(bp), label=compact)
    if unit == "mb":
        bp = int(round(value * 1_000_000.0))
        return LdscWindowSpec(kind="bp", value=float(bp), label=compact)
    if unit == "cm":
        return LdscWindowSpec(kind="cm", value=float(value), label=compact)
    raise ValueError(
        f"Unsupported --ldsc-window unit in {text!r}. Use forms like 100, 100kb, 0.1mb, 100000b, or 10cm."
    )


def _candidate_site_stats_paths(chunk_prefix: Path, explicit: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(Path(f"{chunk_prefix}.site_stats.tsv"))
    if chunk_prefix.parent.name == "chunks":
        candidates.append(chunk_prefix.parent.parent / "summary" / f"{chunk_prefix.name}.site_stats.tsv")
    else:
        candidates.append(chunk_prefix.parent / "summary" / f"{chunk_prefix.name}.site_stats.tsv")
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = path.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def _resolve_site_stats_path(chunk_prefix: Path, explicit: Path | None) -> Path | None:
    for path in _candidate_site_stats_paths(chunk_prefix, explicit):
        if path.is_file():
            return path.resolve()
    return None


def _load_grm_matrix(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"GRM must be square, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"GRM contains non-finite values: {path}")
    return arr


def _read_grm_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                ids.append(line.split()[0])
    if not ids:
        raise ValueError(f"No IDs found in {path}")
    return ids


def _load_and_align_grm(grm_path: Path, grm_id_path: Path, target_ids: list[str]) -> np.ndarray:
    grm = _load_grm_matrix(grm_path)
    grm_ids = _read_grm_ids(grm_id_path)
    if len(grm_ids) != grm.shape[0]:
        raise ValueError(
            f"GRM/ID length mismatch: matrix n={grm.shape[0]}, ids={len(grm_ids)}"
        )
    index = {sid: i for i, sid in enumerate(grm_ids)}
    missing = [sid for sid in target_ids if sid not in index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"GRM is missing target IDs, first few: {preview}")
    order = np.asarray([index[sid] for sid in target_ids], dtype=np.intp)
    return np.asarray(grm[np.ix_(order, order)], dtype=np.float64)


def _variance(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return 0.0
    return float(np.var(arr, ddof=1))


def _maf_from_dosage(row: np.ndarray) -> float:
    arr = np.asarray(row, dtype=np.float64).reshape(-1)
    valid = np.isfinite(arr) & (arr >= 0.0)
    if not np.any(valid):
        return float("nan")
    alt_freq = float(np.mean(arr[valid]) / 2.0)
    alt_freq = min(max(alt_freq, 0.0), 1.0)
    return float(min(alt_freq, 1.0 - alt_freq))


def _collapse_to_logic_bin01(row: np.ndarray) -> np.ndarray:
    arr = np.asarray(row, dtype=np.float64).reshape(-1)
    valid = np.isfinite(arr) & (arr >= 0.0)
    if not np.any(valid):
        raise ValueError("Genotype row has no valid values.")
    rounded = np.rint(arr[valid])
    geno = np.where(rounded <= 0.0, 0, np.where(rounded >= 2.0, 2, 1)).astype(np.uint8)
    c0 = int(np.sum(geno == 0))
    c2 = int(np.sum(geno == 2))
    mode02 = 2 if c2 > c0 else 0
    out = np.full(arr.shape[0], mode02, dtype=np.uint8)
    out[np.where(valid)[0]] = np.where(geno == 1, mode02, geno)
    return (out > 0).astype(np.float64)


def _sample_background_from_grm(grm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    evals, evecs = np.linalg.eigh(np.asarray(grm, dtype=np.float64))
    z = rng.standard_normal(grm.shape[0])
    raw = evecs @ (np.sqrt(np.clip(evals, 0.0, None)) * z)
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    raw -= float(np.mean(raw))
    return raw


def _ols_assoc(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    design = np.column_stack([np.ones(xv.shape[0], dtype=np.float64), xv])
    beta = np.linalg.lstsq(design, yv, rcond=None)[0]
    resid = yv - design @ beta
    df = int(xv.shape[0] - design.shape[1])
    rss = float(resid @ resid)
    s2 = rss / max(df, 1)
    xtx_inv = np.linalg.inv(design.T @ design)
    se = float(np.sqrt(max(s2, 0.0) * xtx_inv[1, 1]))
    tval = float(beta[1] / se) if se > 0.0 else float("nan")
    pval = float(2.0 * stats.t.sf(abs(tval), df=max(df, 1))) if np.isfinite(tval) else float("nan")
    return {
        "beta": float(beta[1]),
        "se": se,
        "t": tval,
        "p": pval,
        "n": int(xv.shape[0]),
    }


def _write_pheno_outputs(out_prefix: Path, sample_ids: list[str], y: np.ndarray, seed: int, trait_name: str) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    with _out_path(out_prefix, ".pheno").open("w", encoding="utf-8") as fh:
        for sid, value in zip(sample_ids, yv):
            fh.write(f"{sid}\t{sid}\t{value:.6f}\n")
    with _out_path(out_prefix, ".pheno.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"IID\t{trait_name}\n")
        for sid, value in zip(sample_ids, yv):
            fh.write(f"{sid}\t{value:.6f}\n")
    rng = np.random.default_rng(int(seed) ^ 0xD9A35C714B1E208D)
    na_idx = set(rng.choice(len(sample_ids), size=int(round(0.1 * len(sample_ids))), replace=False).tolist())
    with _out_path(out_prefix, ".pheno.NA.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"IID\t{trait_name}\n")
        for i, (sid, value) in enumerate(zip(sample_ids, yv)):
            if i in na_idx:
                fh.write(f"{sid}\tNA\n")
            else:
                fh.write(f"{sid}\t{value:.6f}\n")


def _connected_components(r2: np.ndarray, threshold: float) -> tuple[np.ndarray, dict[int, int]]:
    n = int(r2.shape[0])
    comp = np.full(n, -1, dtype=np.int64)
    comp_id = 0
    for i in range(n):
        if comp[i] >= 0:
            continue
        stack = [i]
        comp[i] = comp_id
        while stack:
            u = stack.pop()
            nbr = np.where(r2[u] >= threshold)[0]
            for v in nbr:
                if comp[v] < 0:
                    comp[v] = comp_id
                    stack.append(int(v))
        comp_id += 1
    sizes: dict[int, int] = {}
    for cid in comp.tolist():
        sizes[cid] = sizes.get(cid, 0) + 1
    return (comp, sizes)


def _repair_grm_to_psd(grm: np.ndarray, floor: float = 1e-8) -> tuple[np.ndarray, dict[str, float | int]]:
    sym = 0.5 * (np.asarray(grm, dtype=np.float64) + np.asarray(grm, dtype=np.float64).T)
    evals, evecs = np.linalg.eigh(sym)
    min_eig = float(np.min(evals))
    neg_count = int(np.sum(evals < 0.0))
    evals_clip = np.clip(evals, floor, None)
    repaired = (evecs * evals_clip) @ evecs.T
    target_diag = float(np.mean(np.diag(sym)))
    current_diag = float(np.mean(np.diag(repaired)))
    scale = target_diag / current_diag if current_diag > 0.0 else 1.0
    repaired *= scale
    return repaired, {
        "min_eig_before": min_eig,
        "neg_eig_count_before": neg_count,
        "mean_diag_before": target_diag,
        "mean_diag_after": float(np.mean(np.diag(repaired))),
        "scale_applied": float(scale),
    }


def _validate_fastlmm_switch(grm: np.ndarray, y: np.ndarray) -> dict[str, object]:
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    try:
        model = FastLMM(yv, None, np.asarray(grm, dtype=np.float64).copy())
        switch_to_lm, lrt_stat, pval, _lm_ml0 = gwas_lmm_lm_null_lrt_decision(
            yv,
            np.ones((yv.shape[0], 1), dtype=np.float64),
            float(model.ML0),
            0.05,
            True,
        )
        return {
            "status": "ok",
            "switch_to_lm": bool(switch_to_lm),
            "lrt_stat": float(lrt_stat),
            "pval": float(pval),
            "lmm_ml0": float(model.ML0),
            "lmm_pve": float(model.pve),
        }
    except Exception as exc:
        return {
            "status": "error",
            "switch_to_lm": True,
            "error": repr(exc),
        }


def _make_site_infos(
    bim_rows: list[dict[str, object]],
    site_stats: dict[int, dict[str, object]],
) -> list[SiteInfo]:
    out: list[SiteInfo] = []
    for row in bim_rows:
        pos = int(row["pos"])
        stats_row = site_stats.get(pos)
        if stats_row is None:
            continue
        try:
            maf = float(str(stats_row["freq"]))
        except Exception:
            continue
        try:
            ldsc = float(str(stats_row["ldsc"]))
        except Exception:
            ldsc = float("nan")
        out.append(
            SiteInfo(
                row_index=int(row["row_index"]),
                chrom=str(row["chrom"]),
                snp=str(row["snp"]),
                cm=float(row["cm"]),
                pos=pos,
                ref=str(row["ref"]),
                alt=str(row["alt"]),
                maf=maf,
                ldsc=ldsc,
            )
        )
    if not out:
        raise ValueError("No overlapping BIM/site-stats rows found.")
    return out


def _derive_site_stats_from_chunk(
    *,
    chunk_prefix: Path,
    bim_rows: list[dict[str, object]],
    geno: np.ndarray,
    ldsc_window: str,
    threads: int,
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    if len(bim_rows) != int(geno.shape[0]):
        raise ValueError(f"BIM/genotype row mismatch: bim={len(bim_rows)}, geno={geno.shape[0]}")

    out: dict[int, dict[str, object]] = {}
    for row in bim_rows:
        row_index = int(row["row_index"])
        pos = int(row["pos"])
        out[pos] = {
            "chrom": str(row["chrom"]),
            "pos": pos,
            "freq": _maf_from_dosage(geno[row_index]),
            "M": "",
            "ldsc": float("nan"),
        }

    meta: dict[str, object] = {
        "mode": "chunk-derived",
        "path": None,
        "freq_source": "genotype-dosage",
        "ldsc_source": "missing",
    }

    if hasattr(jxrs, "gstats_bed_ldscore"):
        window = _parse_ldsc_window(ldsc_window)
        try:
            m_raw, ld_raw, ld_n_samples = jxrs.gstats_bed_ldscore(
                str(chunk_prefix),
                str(window.kind),
                float(window.value),
                threads=int(max(1, threads)),
            )
            m_arr = np.asarray(m_raw, dtype=np.int64).reshape(-1)
            ld_arr = np.asarray(ld_raw, dtype=np.float64).reshape(-1)
            if len(m_arr) != len(bim_rows) or len(ld_arr) != len(bim_rows):
                raise ValueError(
                    f"BIM/ldscore length mismatch: bim={len(bim_rows)}, M={len(m_arr)}, ldsc={len(ld_arr)}"
                )
            for row, m_value, ld_value in zip(bim_rows, m_arr, ld_arr):
                out[int(row["pos"])]["M"] = int(m_value)
                out[int(row["pos"])]["ldsc"] = float(ld_value)
            meta["ldsc_source"] = f"janusx.gstats_bed_ldscore[{window.label}]"
            meta["ldsc_n_samples"] = int(ld_n_samples)
        except Exception as exc:
            meta["ldsc_error"] = repr(exc)

    return out, meta


def _load_site_stats(
    *,
    chunk_prefix: Path,
    requested_path: Path | None,
    bim_rows: list[dict[str, object]],
    geno: np.ndarray,
    ldsc_window: str,
    threads: int,
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    resolved = _resolve_site_stats_path(chunk_prefix, requested_path)
    if resolved is not None:
        return _read_site_stats(resolved), {
            "mode": "file",
            "path": str(resolved),
            "freq_source": "site_stats.tsv",
            "ldsc_source": "site_stats.tsv",
        }
    site_stats, meta = _derive_site_stats_from_chunk(
        chunk_prefix=chunk_prefix,
        bim_rows=bim_rows,
        geno=geno,
        ldsc_window=ldsc_window,
        threads=threads,
    )
    if requested_path is not None:
        meta["requested_path"] = str(requested_path)
    return site_stats, meta


def _choose_pair(
    *,
    chunk_prefix: Path,
    site_infos: list[SiteInfo],
    geno: np.ndarray,
    max_pair_distance: int,
    min_site_maf: float,
    max_site_maf: float,
    min_pseudo_freq: float,
    ld_block_r2_threshold: float,
    prefer_distinct_blocks: bool,
    target_pseudo_freq: float,
    threads: int,
) -> tuple[PairCandidate, dict[str, object]]:
    positions = [s.pos for s in site_infos]
    chrom = str(site_infos[0].chrom)
    r2_mat, _out_chrom, _out_pos = bed_ldblock_r2_rust(
        str(chunk_prefix),
        [chrom],
        [int(min(positions))],
        [int(max(positions))],
        [chrom] * len(positions),
        [int(p) for p in positions],
        threads=int(max(1, threads)),
    )
    r2_arr = np.asarray(r2_mat, dtype=np.float64)
    comp, comp_sizes = _connected_components(r2_arr, float(ld_block_r2_threshold))
    best: PairCandidate | None = None
    seen = 0
    kept = 0
    for i, site1 in enumerate(site_infos):
        if not (min_site_maf <= site1.maf <= max_site_maf):
            continue
        g1 = geno[site1.row_index].copy()
        b1 = _collapse_to_logic_bin01(g1)
        for j in range(i + 1, len(site_infos)):
            site2 = site_infos[j]
            dist = int(site2.pos - site1.pos)
            if dist > max_pair_distance:
                break
            seen += 1
            if not (min_site_maf <= site2.maf <= max_site_maf):
                continue
            block1 = int(comp[i])
            block2 = int(comp[j])
            if prefer_distinct_blocks and block1 == block2:
                continue
            g2 = geno[site2.row_index].copy()
            b2 = _collapse_to_logic_bin01(g2)
            pseudo = ((b1 > 0.0) & (b2 > 0.0)).astype(np.float64)
            pseudo_freq = float(np.mean(pseudo))
            if pseudo_freq < min_pseudo_freq:
                continue
            mask = np.isfinite(g1) & np.isfinite(g2)
            dosage_corr = float(np.corrcoef(g1[mask], g2[mask])[0, 1]) if int(np.sum(mask)) > 3 else float("nan")
            pair_r2 = float(r2_arr[i, j])
            maf_gap = abs(site1.maf - site2.maf)
            score = (
                abs(pseudo_freq - target_pseudo_freq)
                + 0.25 * abs(dosage_corr if np.isfinite(dosage_corr) else 1.0)
                + 0.25 * maf_gap
                + 0.10 * (dist / float(max_pair_distance))
            )
            cand = PairCandidate(
                site1=site1,
                site2=site2,
                block1=block1,
                block2=block2,
                distance=dist,
                pair_r2=pair_r2,
                dosage_corr=dosage_corr,
                pseudo_freq=pseudo_freq,
                score=float(score),
            )
            kept += 1
            if best is None or cand.score < best.score:
                best = cand
    if best is None:
        raise ValueError("No pair candidate passed the requested filters.")
    meta = {
        "pairs_seen_within_distance": int(seen),
        "pairs_kept_after_filters": int(kept),
        "ld_block_r2_threshold": float(ld_block_r2_threshold),
        "block_sizes": {
            "site1": int(comp_sizes.get(best.block1, 0)),
            "site2": int(comp_sizes.get(best.block2, 0)),
        },
    }
    return best, meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Simulate one two-site raw-AND marginal phenotype case from an existing chunk, "
            "and optionally repair an indefinite external GRM to a PSD copy for stable FastLMM validation."
        )
    )
    p.add_argument("--chunk-prefix", default=str(DEFAULT_CHUNK_PREFIX), help="PLINK prefix for the chosen chunk.")
    p.add_argument(
        "--site-stats",
        default=DEFAULT_SITE_STATS,
        help=(
            "Optional chunk site_stats.tsv path. If missing, the script will recover site statistics from the chunk."
        ),
    )
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    p.add_argument("--prefix", default=None, help="Output prefix stem.")
    p.add_argument("--grm", default=str(DEFAULT_GRM), help="External GRM .npy path.")
    p.add_argument("--grm-id", default=str(DEFAULT_GRM_ID), help="External GRM ID file.")
    p.add_argument(
        "--ldsc-window",
        default=DEFAULT_LDSC_WINDOW,
        help="LD-score window used only when site stats must be recovered from the chunk.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=_default_threads(),
        help="Worker threads for LD/stat kernels. Defaults to scheduler env if available, else 1.",
    )
    p.add_argument("--bg-pve", type=float, default=0.60, help="Background variance target.")
    p.add_argument("--gate-pve", type=float, default=0.04, help="Raw AND gate variance target.")
    p.add_argument("--seed", type=int, default=20260520, help="Random seed.")
    p.add_argument("--max-pair-distance", type=int, default=200000, help="Maximum pair distance in bp.")
    p.add_argument("--min-site-maf", type=float, default=0.05, help="Minimum site MAF for pair search.")
    p.add_argument("--max-site-maf", type=float, default=0.20, help="Maximum site MAF for pair search.")
    p.add_argument("--min-pseudo-freq", type=float, default=0.025, help="Minimum raw-AND carrier frequency.")
    p.add_argument(
        "--target-pseudo-freq",
        type=float,
        default=0.026,
        help="Pair search prefers pseudo frequencies near this value.",
    )
    p.add_argument(
        "--ld-block-r2-threshold",
        type=float,
        default=0.20,
        help="r2 threshold used to define heuristic LD blocks.",
    )
    p.add_argument(
        "--allow-same-ld-block",
        action="store_true",
        help="Allow pair selection from the same LD block.",
    )
    p.add_argument("--site1", type=int, default=None, help="Optional fixed first site position.")
    p.add_argument("--site2", type=int, default=None, help="Optional fixed second site position.")
    p.add_argument(
        "--psd-grm-prefix",
        default=None,
        help="Optional prefix for repaired PSD GRM copy (without .npy suffix). Defaults to <out-dir>/<grm-stem>.psd.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    chunk_prefix = Path(args.chunk_prefix).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    site_stats_path = Path(args.site_stats).expanduser() if args.site_stats else None
    chunk_id = chunk_prefix.name
    prefix_stem = str(args.prefix or f"{chunk_id}.marginal_pair.raw.and")
    out_prefix = out_dir / prefix_stem
    threads = int(max(1, int(args.threads)))
    bg_pve = float(args.bg_pve)
    gate_pve = float(args.gate_pve)
    residual_var = 1.0 - bg_pve - gate_pve
    if residual_var < 0.0:
        raise ValueError("bg_pve + gate_pve must be <= 1.0")

    fam_ids = _read_fam_ids(chunk_prefix)
    bim_rows = _read_bim_rows(chunk_prefix)
    geno = load_bed_u8_matrix(str(chunk_prefix)).astype(np.float64)
    geno[geno == 3.0] = np.nan
    site_stats, site_stats_meta = _load_site_stats(
        chunk_prefix=chunk_prefix,
        requested_path=site_stats_path,
        bim_rows=bim_rows,
        geno=geno,
        ldsc_window=str(args.ldsc_window),
        threads=threads,
    )
    site_infos = _make_site_infos(bim_rows, site_stats)

    pos_to_site = {s.pos: s for s in site_infos}
    pair_meta: dict[str, object]
    if args.site1 is not None and args.site2 is not None:
        site1 = pos_to_site.get(int(args.site1))
        site2 = pos_to_site.get(int(args.site2))
        if site1 is None or site2 is None:
            raise KeyError("Requested site1/site2 are not both present in the chunk metadata.")
        if site1.pos > site2.pos:
            site1, site2 = site2, site1
        positions = [s.pos for s in site_infos]
        chrom = str(site_infos[0].chrom)
        r2_mat, _out_chrom, _out_pos = bed_ldblock_r2_rust(
            str(chunk_prefix),
            [chrom],
            [int(min(positions))],
            [int(max(positions))],
            [chrom] * len(positions),
            [int(p) for p in positions],
            threads=threads,
        )
        r2_arr = np.asarray(r2_mat, dtype=np.float64)
        comp, comp_sizes = _connected_components(r2_arr, float(args.ld_block_r2_threshold))
        pos_to_idx = {s.pos: i for i, s in enumerate(site_infos)}
        i = pos_to_idx[site1.pos]
        j = pos_to_idx[site2.pos]
        b1 = _collapse_to_logic_bin01(geno[site1.row_index])
        b2 = _collapse_to_logic_bin01(geno[site2.row_index])
        pseudo = ((b1 > 0.0) & (b2 > 0.0)).astype(np.float64)
        mask = np.isfinite(geno[site1.row_index]) & np.isfinite(geno[site2.row_index])
        dosage_corr = float(np.corrcoef(geno[site1.row_index][mask], geno[site2.row_index][mask])[0, 1])
        pair = PairCandidate(
            site1=site1,
            site2=site2,
            block1=int(comp[i]),
            block2=int(comp[j]),
            distance=int(site2.pos - site1.pos),
            pair_r2=float(r2_arr[i, j]),
            dosage_corr=dosage_corr,
            pseudo_freq=float(np.mean(pseudo)),
            score=float("nan"),
        )
        pair_meta = {
            "pairs_seen_within_distance": None,
            "pairs_kept_after_filters": None,
            "ld_block_r2_threshold": float(args.ld_block_r2_threshold),
            "block_sizes": {
                "site1": int(comp_sizes.get(pair.block1, 0)),
                "site2": int(comp_sizes.get(pair.block2, 0)),
            },
        }
    else:
        pair, pair_meta = _choose_pair(
            chunk_prefix=chunk_prefix,
            site_infos=site_infos,
            geno=geno,
            max_pair_distance=int(args.max_pair_distance),
            min_site_maf=float(args.min_site_maf),
            max_site_maf=float(args.max_site_maf),
            min_pseudo_freq=float(args.min_pseudo_freq),
            ld_block_r2_threshold=float(args.ld_block_r2_threshold),
            prefer_distinct_blocks=not bool(args.allow_same_ld_block),
            target_pseudo_freq=float(args.target_pseudo_freq),
            threads=threads,
        )

    g1 = geno[pair.site1.row_index].copy()
    g2 = geno[pair.site2.row_index].copy()
    b1 = _collapse_to_logic_bin01(g1)
    b2 = _collapse_to_logic_bin01(g2)
    pseudo = ((b1 > 0.0) & (b2 > 0.0)).astype(np.float64)

    grm_path = Path(args.grm).expanduser().resolve()
    grm_id_path = Path(args.grm_id).expanduser().resolve()
    grm = _load_and_align_grm(grm_path, grm_id_path, fam_ids)
    repaired_grm, grm_repair = _repair_grm_to_psd(grm)
    psd_prefix = (
        Path(args.psd_grm_prefix).expanduser().resolve()
        if args.psd_grm_prefix
        else out_dir / f"{grm_path.stem}.psd"
    )
    np.save(Path(f"{psd_prefix}.npy"), np.asarray(repaired_grm, dtype=np.float64))
    shutil.copyfile(grm_id_path, Path(f"{psd_prefix}.npy.id"))

    rng = np.random.default_rng(int(args.seed))
    bg_raw = _sample_background_from_grm(repaired_grm, rng)
    bg_scale = np.sqrt(bg_pve / _variance(bg_raw)) if bg_pve > 0.0 and _variance(bg_raw) > 0.0 else 0.0
    bg_effect = bg_raw * bg_scale
    noise = np.asarray(rng.standard_normal(len(fam_ids)) * np.sqrt(max(residual_var, 0.0)), dtype=np.float64)
    coef0 = float(rng.uniform(-1.0, 1.0))
    gate_raw = pseudo * coef0
    gate_scale = np.sqrt(gate_pve / _variance(gate_raw)) if gate_pve > 0.0 and _variance(gate_raw) > 0.0 else 0.0
    gate_effect = gate_raw * gate_scale
    y = noise + bg_effect + gate_effect

    trait_name = f"margpair_bg{bg_pve:.3f}_gate{gate_pve:.3f}_grm_rawand"
    _write_pheno_outputs(out_prefix, fam_ids, y, int(args.seed), trait_name)

    with _out_path(out_prefix, ".fixed.effects.tsv").open("w", encoding="utf-8") as fh:
        fh.write("term_id\tkind\tlogic\tsites\tlabel\teffect\n")
        fh.write(
            "1\tlogic_gate\tand\t"
            f"{pair.site1.chrom}:{pair.site1.pos};{pair.site2.chrom}:{pair.site2.pos}\t"
            f"{pair.site1.chrom}_{pair.site1.pos}[{pair.site1.ref}>{pair.site1.alt}]&"
            f"{pair.site2.chrom}_{pair.site2.pos}[{pair.site2.ref}>{pair.site2.alt}]\t"
            f"{(coef0 * gate_scale):.10f}\n"
        )

    with _out_path(out_prefix, ".causal.sites.tsv").open("w", encoding="utf-8") as fh:
        fh.write(f"{pair.site1.chrom}\t{pair.site1.pos}\t{pair.site1.pos}\n")
        fh.write(f"{pair.site2.chrom}\t{pair.site2.pos}\t{pair.site2.pos}\n")

    with _out_path(out_prefix, ".random.effects.tsv").open("w", encoding="utf-8") as fh:
        fh.write("sample_index\tsample_id\tsource\trole\teffect\n")
        for i, (sid, value) in enumerate(zip(fam_ids, bg_effect), start=1):
            fh.write(f"{i}\t{sid}\tgrm_psd\tbackground\t{value:.10f}\n")

    with _out_path(out_prefix, ".pseudo.tsv").open("w", encoding="utf-8") as fh:
        fh.write("IID\tdosage1\tdosage2\tbinary1\tbinary2\tpseudo_and_raw\n")
        for sid, d1, d2, x1, x2, pp in zip(fam_ids, g1, g2, b1, b2, pseudo):
            d1_txt = "NA" if not np.isfinite(d1) else f"{d1:.0f}"
            d2_txt = "NA" if not np.isfinite(d2) else f"{d2:.0f}"
            fh.write(f"{sid}\t{d1_txt}\t{d2_txt}\t{x1:.0f}\t{x2:.0f}\t{pp:.0f}\n")

    assoc_rows = [
        {"feature": "dosage1", **_ols_assoc(g1, y)},
        {"feature": "dosage2", **_ols_assoc(g2, y)},
        {"feature": "binary1", **_ols_assoc(b1, y)},
        {"feature": "binary2", **_ols_assoc(b2, y)},
        {"feature": "pseudo_and_raw", **_ols_assoc(pseudo, y)},
    ]
    with _out_path(out_prefix, ".assoc.tsv").open("w", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["feature", "beta", "se", "t", "p", "n"], delimiter="\t")
        writer.writeheader()
        for row in assoc_rows:
            writer.writerow(row)

    rng_na = np.random.default_rng(int(args.seed) ^ 0xD9A35C714B1E208D)
    na_idx = set(rng_na.choice(len(fam_ids), size=int(round(0.1 * len(fam_ids))), replace=False).tolist())
    keep = np.asarray([i not in na_idx for i in range(len(fam_ids))], dtype=bool)
    validation = {
        "full_orig_grm": _validate_fastlmm_switch(grm, y),
        "full_psd_grm": _validate_fastlmm_switch(repaired_grm, y),
        "na_orig_grm": _validate_fastlmm_switch(grm[np.ix_(keep, keep)], y[keep]),
        "na_psd_grm": _validate_fastlmm_switch(repaired_grm[np.ix_(keep, keep)], y[keep]),
    }

    summary = {
        "chunk_prefix": str(chunk_prefix),
        "chunk_id": chunk_id,
        "trait_name": trait_name,
        "seed": int(args.seed),
        "logic": "and",
        "effect_model": "raw_and",
        "bg_pve": float(bg_pve),
        "gate_pve": float(gate_pve),
        "residual_var": float(residual_var),
        "site1": {
            "chrom": pair.site1.chrom,
            "pos": int(pair.site1.pos),
            "snp": pair.site1.snp,
            "ref": pair.site1.ref,
            "alt": pair.site1.alt,
            "maf": float(pair.site1.maf),
            "ldsc": float(pair.site1.ldsc),
            "binary_freq": float(np.mean(b1)),
            "ld_block_id": int(pair.block1),
        },
        "site2": {
            "chrom": pair.site2.chrom,
            "pos": int(pair.site2.pos),
            "snp": pair.site2.snp,
            "ref": pair.site2.ref,
            "alt": pair.site2.alt,
            "maf": float(pair.site2.maf),
            "ldsc": float(pair.site2.ldsc),
            "binary_freq": float(np.mean(b2)),
            "ld_block_id": int(pair.block2),
        },
        "distance_bp": int(pair.distance),
        "pair_r2": float(pair.pair_r2),
        "dosage_corr": float(pair.dosage_corr),
        "pseudo_and_raw_freq": float(np.mean(pseudo)),
        "background_var": _variance(bg_effect),
        "gate_var": _variance(gate_effect),
        "phenotype_var": _variance(y),
        "raw_gate_coef_before_scaling": float(coef0),
        "scaled_gate_effect": float(coef0 * gate_scale),
        "pair_search": pair_meta,
        "site_stats": site_stats_meta,
        "grm": {
            "original_path": str(grm_path),
            "original_id_path": str(grm_id_path),
            "psd_path": str(Path(f"{psd_prefix}.npy")),
            "psd_id_path": str(Path(f"{psd_prefix}.npy.id")),
            "repair": grm_repair,
        },
        "assoc": assoc_rows,
        "validation": validation,
    }
    with _out_path(out_prefix, ".case.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=True, indent=2)

    print(f"Chunk        : {chunk_prefix}")
    print(f"Output prefix: {out_prefix}")
    print(f"Site stats   : {site_stats_meta['mode']}")
    print(f"Chosen sites : {pair.site1.pos}, {pair.site2.pos}")
    print(f"Distance bp  : {pair.distance}")
    print(f"Pair r2      : {pair.pair_r2:.6f}")
    print(f"Pseudo freq  : {np.mean(pseudo):.6f}")
    for row in assoc_rows:
        print(f"{row['feature']:14s} beta={row['beta']:.6f} p={row['p']:.6g}")
    print(f"Orig GRM eig : min={grm_repair['min_eig_before']:.6f}, neg={grm_repair['neg_eig_count_before']}")
    print(f"Orig -> LM   : full={validation['full_orig_grm'].get('switch_to_lm')}  na={validation['na_orig_grm'].get('switch_to_lm')}")
    print(f"PSD  -> LM   : full={validation['full_psd_grm'].get('switch_to_lm')}  na={validation['na_psd_grm'].get('switch_to_lm')}")
    print(f"Wrote: {_out_path(out_prefix, '.pheno.txt')}")
    print(f"Wrote: {_out_path(out_prefix, '.fixed.effects.tsv')}")
    print(f"Wrote: {_out_path(out_prefix, '.assoc.tsv')}")
    print(f"Wrote: {_out_path(out_prefix, '.case.json')}")
    print(f"Wrote: {Path(f'{psd_prefix}.npy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
