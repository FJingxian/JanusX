"""Helpers for keeping PLINK 1 and PLINK 2 input selection explicit."""

from __future__ import annotations

import io
import os
from contextlib import contextmanager
from typing import Iterator, TextIO

import zstandard


_BFILE_SUFFIXES = (".bed", ".bim", ".fam")
_PFILE_SUFFIXES = (".pgen", ".pvar.zst", ".pvar", ".psam")


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
    pfile_complete = (
        os.path.isfile(f"{prefix}.pgen")
        and (os.path.isfile(f"{prefix}.pvar") or os.path.isfile(f"{prefix}.pvar.zst"))
        and os.path.isfile(f"{prefix}.psam")
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


def pvar_path_for_prefix(prefix: str) -> str:
    """Return the existing zstd or plain PVAR sidecar for a pfile prefix."""
    normalized = normalize_plink_prefix(str(prefix))
    compressed = f"{normalized}.pvar.zst"
    if os.path.isfile(compressed):
        return compressed
    return f"{normalized}.pvar"


@contextmanager
def open_text_auto(
    path: str,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    buffering: int = -1,
) -> Iterator[TextIO]:
    """Open plain or zstd-compressed text with the same streaming interface."""
    if str(path).lower().endswith(".zst"):
        with open(path, "rb", buffering=buffering) as raw:
            with zstandard.ZstdDecompressor().stream_reader(
                raw, read_across_frames=True
            ) as decoder:
                with io.TextIOWrapper(
                    decoder, encoding=encoding, errors=errors
                ) as text:
                    yield text
        return
    with open(
        path, "r", encoding=encoding, errors=errors, buffering=buffering
    ) as text:
        yield text
