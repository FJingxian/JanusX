from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def format_path_for_display(path: str | Path) -> str:
    s = str(path)
    if s == "" or "://" in s:
        return s
    if os.name == "nt":
        return s.replace("/", "\\")
    return s.replace("\\", "/")


def format_output_display(
    out_fmt: str,
    out_prefix: str | Path,
    out_path: str | Path,
) -> str:
    fmt = str(out_fmt).lower()
    prefix_disp = format_path_for_display(out_prefix)
    path_disp = format_path_for_display(out_path)
    if fmt == "plink":
        return f"{prefix_disp} (.bed/.bim/.fam)"
    if fmt == "npy":
        return f"{prefix_disp} (.npy/.site/.id)"
    if fmt == "txt":
        return f"{prefix_disp} (.txt/.site/.id)"
    if fmt == "bin":
        return f"{prefix_disp} (.bin/.bin.site/.bin.id or .bim/.fam)"
    return f"{path_disp} ({fmt})"


def _norm(path: str | Path) -> str:
    return format_path_for_display(path)


def safe_expanduser(path: str | Path) -> Path:
    """
    Expand only unambiguous home-directory syntax.
    On some HPC/batch environments, pathlib.expanduser() can raise
    RuntimeError("Could not determine home directory.").
    In that case, keep the original path unchanged.

    JanusX also uses cache prefixes like "~panel" / "~mouse_hs1940".
    Those must remain literal relative paths instead of being interpreted
    as a home-directory reference or "~user".
    """
    p = Path(path)
    raw = str(p)
    if not (raw == "~" or raw.startswith("~/") or raw.startswith("~\\")):
        return p
    try:
        return p.expanduser()
    except RuntimeError:
        return p


def safe_home() -> Path:
    """
    Best-effort home directory.
    Prefer explicit environment variables when available and
    gracefully fall back when Path.home() is unavailable.
    """
    env_home = str(os.environ.get("HOME", "")).strip()
    if env_home:
        return Path(env_home)
    env_userprofile = str(os.environ.get("USERPROFILE", "")).strip()
    if env_userprofile:
        return Path(env_userprofile)
    home_drive = str(os.environ.get("HOMEDRIVE", "")).strip()
    home_path = str(os.environ.get("HOMEPATH", "")).strip()
    if home_drive and home_path:
        return Path(f"{home_drive}{home_path}")
    try:
        return Path.home()
    except RuntimeError:
        return Path(".")


def safe_resolve(path: str | Path, *, strict: bool = False) -> Path:
    p = safe_expanduser(path)
    try:
        return p.resolve(strict=strict)
    except Exception:
        return p


# Backward-compatible alias for existing internal usage.
_safe_expanduser = safe_expanduser


def ensure_file_exists(logger, path: str, label: str) -> bool:
    p = safe_expanduser(path)
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
    p = safe_expanduser(path)
    if p.is_dir():
        return True
    logger.error(f"{label} not found: {_norm(str(p))}")
    return False


def ensure_file_input_exists(logger, path: str, label: str = "Genotype file input") -> bool:
    p = safe_expanduser(path)
    low = str(p).lower()
    if low.endswith(".npy"):
        prefix = Path(str(p)[: -len(".npy")])
        matrix_candidates = [p]
        id_candidates = [Path(f"{prefix}.id"), Path(f"{prefix}.fam")]
    elif low.endswith(".bin"):
        prefix = Path(str(p)[: -len(".bin")])
        matrix_candidates = [p]
        id_candidates = [Path(f"{prefix}.bin.id"), Path(f"{prefix}.id"), Path(f"{prefix}.fam")]
    elif low.endswith((".txt", ".tsv", ".csv")):
        for ext in (".txt", ".tsv", ".csv"):
            if low.endswith(ext):
                prefix = Path(str(p)[: -len(ext)])
                break
        matrix_candidates = [p]
        id_candidates = [Path(f"{prefix}.id"), Path(f"{prefix}.fam")]
    else:
        prefix = p
        matrix_candidates = [
            Path(f"{prefix}{ext}") for ext in (".npy", ".bin", ".txt", ".tsv", ".csv")
        ]
        id_candidates = [Path(f"{prefix}.bin.id"), Path(f"{prefix}.id"), Path(f"{prefix}.fam")]

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
    p = safe_expanduser(path)
    low = str(p).lower()
    if low.endswith(".npy"):
        prefix = Path(str(p)[: -len(".npy")])
    elif low.endswith(".bin"):
        prefix = Path(str(p)[: -len(".bin")])
    elif low.endswith((".txt", ".tsv", ".csv")):
        for ext in (".txt", ".tsv", ".csv"):
            if low.endswith(ext):
                prefix = Path(str(p)[: -len(ext)])
                break
    else:
        prefix = p

    site_candidates = [
        Path(f"{prefix}.bin.site"),
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
    pfx = safe_expanduser(prefix)
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
