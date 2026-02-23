from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _norm(path: str) -> str:
    return str(path).replace("\\", "/")


def ensure_file_exists(logger, path: str, label: str) -> bool:
    p = Path(path).expanduser()
    if p.is_file():
        return True
    logger.error(f"{label} not found: {_norm(str(p))}")
    return False


def ensure_dir_exists(logger, path: str, label: str) -> bool:
    p = Path(path).expanduser()
    if p.is_dir():
        return True
    logger.error(f"{label} not found: {_norm(str(p))}")
    return False


def ensure_plink_prefix_exists(logger, prefix: str, label: str = "PLINK prefix") -> bool:
    pfx = Path(prefix).expanduser()
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
