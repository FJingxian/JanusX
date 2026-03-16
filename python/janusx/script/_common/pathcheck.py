from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _norm(path: str) -> str:
    s = str(path)
    if os.name == "nt":
        return s.replace("/", "\\")
    return s.replace("\\", "/")


def _safe_expanduser(path: str | Path) -> Path:
    """
    Expand '~' when possible.
    On some HPC/batch environments, pathlib.expanduser() can raise
    RuntimeError("Could not determine home directory.").
    In that case, keep the original path unchanged.
    """
    p = Path(path)
    try:
        return p.expanduser()
    except RuntimeError:
        return p


def ensure_file_exists(logger, path: str, label: str) -> bool:
    p = _safe_expanduser(path)
    if p.is_file():
        return True
    # Accept PLINK prefix paths (prefix + .bed/.bim/.fam) for
    # genotype inputs that are normalized to cache-backed BED prefixes.
    if p.suffix == "":
        required = [Path(f"{p}{ext}") for ext in (".bed", ".bim", ".fam")]
        if all(r.is_file() for r in required):
            return True
    logger.error(f"{label} not found: {_norm(str(p))}")
    return False


def ensure_dir_exists(logger, path: str, label: str) -> bool:
    p = _safe_expanduser(path)
    if p.is_dir():
        return True
    logger.error(f"{label} not found: {_norm(str(p))}")
    return False


def ensure_file_input_exists(logger, path: str, label: str = "Genotype file input") -> bool:
    p = _safe_expanduser(path)
    low = str(p).lower()
    if low.endswith(".npy"):
        prefix = Path(str(p)[: -len(".npy")])
        matrix_candidates = [p]
        id_candidates = [Path(f"{prefix}.id")]
    elif low.endswith((".txt", ".tsv", ".csv")):
        for ext in (".txt", ".tsv", ".csv"):
            if low.endswith(ext):
                prefix = Path(str(p)[: -len(ext)])
                break
        matrix_candidates = [p]
        id_candidates = [Path(f"{prefix}.id")]
    else:
        prefix = p
        matrix_candidates = [Path(f"{prefix}{ext}") for ext in (".npy", ".txt", ".tsv", ".csv")]
        id_candidates = [Path(f"{prefix}.id")]

    matrix_found = next((cand for cand in matrix_candidates if cand.is_file()), None)
    id_found = next((cand for cand in id_candidates if cand.is_file()), None)
    if matrix_found is not None and id_found is not None:
        return True

    logger.error(f"{label} is incomplete: {_norm(str(p))}")
    if matrix_found is None:
        for cand in matrix_candidates:
            logger.error(f"Missing matrix file: {_norm(str(cand))}")
    if id_found is None:
        for cand in id_candidates:
            logger.error(f"Missing sample ID sidecar: {_norm(str(cand))}")
    return False


def ensure_file_input_site_metadata_exists(
    logger,
    path: str,
    label: str = "Genotype FILE site metadata",
) -> bool:
    p = _safe_expanduser(path)
    low = str(p).lower()
    if low.endswith(".npy"):
        prefix = Path(str(p)[: -len(".npy")])
    elif low.endswith((".txt", ".tsv", ".csv")):
        for ext in (".txt", ".tsv", ".csv"):
            if low.endswith(ext):
                prefix = Path(str(p)[: -len(ext)])
                break
    else:
        prefix = p

    site_candidates = [
        Path(f"{prefix}.site"),
        Path(f"{prefix}.site.tsv"),
        Path(f"{prefix}.site.txt"),
        Path(f"{prefix}.site.csv"),
        Path(f"{prefix}.sites.tsv"),
        Path(f"{prefix}.sites.txt"),
        Path(f"{prefix}.sites.csv"),
        Path(f"{prefix}.bim"),
    ]
    site_found = next((cand for cand in site_candidates if cand.is_file()), None)
    if site_found is not None:
        return True

    logger.error(f"{label} not found: {_norm(str(p))}")
    for cand in site_candidates:
        logger.error(f"Missing site metadata candidate: {_norm(str(cand))}")
    return False


def ensure_plink_prefix_exists(logger, prefix: str, label: str = "PLINK prefix") -> bool:
    pfx = _safe_expanduser(prefix)
    required = [Path(f"{pfx}{ext}") for ext in (".bed", ".bim", ".fam")]
    missing = [p for p in required if not p.is_file()]
    if not missing:
        return True
    logger.error(f"{label} is incomplete: {_norm(str(pfx))}")
    for p in missing:
        logger.error(f"Missing file: {_norm(str(p))}")
    return False


def ensure_all_true(results: Iterable[bool]) -> bool:
    return all(bool(x) for x in results)
