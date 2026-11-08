"""Helpers for keeping PLINK 1 and PLINK 2 input selection explicit."""

from __future__ import annotations

import os


_BFILE_SUFFIXES = (".bed", ".bim", ".fam")
_PFILE_SUFFIXES = (".pgen", ".pvar", ".psam")


def explicit_plink_kind(path_or_prefix: str) -> str | None:
    """Return the PLINK family named by an explicit sidecar suffix."""
    low = str(path_or_prefix).strip().lower()
    if low.endswith(_PFILE_SUFFIXES):
        return "pfile"
    if low.endswith(_BFILE_SUFFIXES):
        return "bfile"
    return None


def normalize_plink_prefix(path_or_prefix: str) -> str:
    """Remove a recognized BED/BIM/FAM/PGEN/PVAR/PSAM suffix."""
    raw = str(path_or_prefix).strip()
    low = raw.lower()
    for suffix in (*_BFILE_SUFFIXES, *_PFILE_SUFFIXES):
        if low.endswith(suffix):
            return raw[: -len(suffix)]
    return raw


def canonical_plink_input_path(path_or_prefix: str, kind: str) -> str:
    """Return an explicit `.bed` or `.pgen` path for the requested family."""
    key = str(kind).strip().lower()
    if key in {"pgen", "pfile"}:
        key = "pfile"
        suffix = ".pgen"
    elif key in {"bed", "bfile"}:
        key = "bfile"
        suffix = ".bed"
    else:
        raise ValueError(f"Unknown PLINK input kind: {kind!r}")

    explicit_kind = explicit_plink_kind(path_or_prefix)
    if explicit_kind is not None and explicit_kind != key:
        raise ValueError(
            f"PLINK input suffix selects {explicit_kind}, but the requested option selects {key}: "
            f"{path_or_prefix}"
        )
    return f"{normalize_plink_prefix(path_or_prefix)}{suffix}"


def detect_existing_plink_kind(path_or_prefix: str) -> tuple[str, str] | None:
    """Resolve a PLINK prefix when its family is explicit or unambiguous.

    A bare prefix with both complete PLINK families is ambiguous and raises
    instead of silently selecting BED. Callers can disambiguate with `.bed`
    or `.pgen`.
    """
    explicit_kind = explicit_plink_kind(path_or_prefix)
    prefix = normalize_plink_prefix(path_or_prefix)
    if prefix == "":
        return None

    bfile_complete = all(
        os.path.isfile(f"{prefix}{suffix}") for suffix in _BFILE_SUFFIXES
    )
    pfile_complete = all(
        os.path.isfile(f"{prefix}{suffix}") for suffix in _PFILE_SUFFIXES
    )
    if explicit_kind is not None:
        return (explicit_kind, prefix)
    if bfile_complete and pfile_complete:
        raise ValueError(
            f"Ambiguous PLINK prefix has both BED and PGEN files: {prefix}; "
            "pass a `.bed` or `.pgen` path to select the input family."
        )
    if pfile_complete:
        return "pfile", prefix
    if bfile_complete:
        return "bfile", prefix
    return None
