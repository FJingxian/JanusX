from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from janusx.gfreader import SiteInfo
from .status import CliStatus


GENOTYPE_TEXT_SUFFIXES: tuple[str, ...] = (".txt", ".tsv", ".csv")
GENOTYPE_SUFFIXES: tuple[str, ...] = GENOTYPE_TEXT_SUFFIXES + (".npy",)


def find_duplicates(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    dup_seen: set[str] = set()
    dup: list[str] = []
    for item in items:
        if item in seen and item not in dup_seen:
            dup.append(item)
            dup_seen.add(item)
        seen.add(item)
    return dup


def strip_known_suffix(path: str, exts: Sequence[str] | None = None) -> str:
    p = str(path)
    low = p.lower()
    known = tuple(exts) if exts is not None else (
        ".vcf.gz",
        ".vcf",
        ".bed",
        ".bim",
        ".fam",
        ".txt",
        ".tsv",
        ".csv",
        ".npy",
    )
    for ext in known:
        if low.endswith(ext):
            return p[: -len(ext)]
    return p


def strip_default_prefix_suffix(name: str) -> str:
    base = os.path.basename(str(name).rstrip("/\\"))
    low = base.lower()
    if low.endswith(".vcf.gz"):
        base = base[: -len(".vcf.gz")]
    else:
        for ext in (".vcf", ".txt", ".tsv", ".csv", ".npy"):
            if low.endswith(ext):
                base = base[: -len(ext)]
                break
    if base.startswith("~"):
        base = base[1:]
    return base


def determine_genotype_source(
    *,
    vcf: str | None,
    file: str | None,
    bfile: str | None,
    prefix: str | None = None,
) -> tuple[str, str]:
    if vcf:
        gfile = str(vcf)
        auto_prefix = strip_default_prefix_suffix(gfile)
    elif file:
        gfile = str(file)
        auto_prefix = strip_default_prefix_suffix(gfile)
    elif bfile:
        gfile = str(bfile)
        auto_prefix = os.path.basename(gfile.rstrip("/\\"))
    else:
        raise ValueError("No genotype input specified. Use -vcf, -file or -bfile.")

    resolved_prefix = str(prefix) if prefix is not None else auto_prefix
    return gfile.replace("\\", "/"), resolved_prefix


def determine_genotype_source_from_args(args: Any) -> tuple[str, str]:
    return determine_genotype_source(
        vcf=getattr(args, "vcf", None),
        file=getattr(args, "file", None),
        bfile=getattr(args, "bfile", None),
        prefix=getattr(args, "prefix", None),
    )


def build_prefix_candidates(
    file_arg: str,
    *,
    text_suffixes: Sequence[str] = GENOTYPE_TEXT_SUFFIXES,
) -> tuple[str, list[str]]:
    raw = Path(file_arg).expanduser()
    text_suffix_set = {str(x).lower() for x in text_suffixes}

    if raw.suffix.lower() in text_suffix_set:
        raw_prefix = strip_known_suffix(str(raw), exts=tuple(text_suffixes))
        cache_prefix = str(Path(raw_prefix).parent / f"~{Path(raw_prefix).name}")
        return raw_prefix, [raw_prefix, cache_prefix]

    if raw.suffix.lower() == ".npy":
        raw_prefix = strip_known_suffix(str(raw), exts=(".npy",))
        noncache_prefix = str(Path(raw_prefix).parent / Path(raw_prefix).name.lstrip("~"))
        return noncache_prefix, [noncache_prefix, raw_prefix]

    raw_prefix = str(raw)
    cache_prefix = str(raw.parent / f"~{raw.name}")
    return raw_prefix, [raw_prefix, cache_prefix]


def discover_site_path(prefixes: Sequence[str]) -> str | None:
    for prefix in prefixes:
        for cand in (
            f"{prefix}.site",
            f"{prefix}.site.tsv",
            f"{prefix}.site.txt",
            f"{prefix}.site.csv",
            f"{prefix}.sites.tsv",
            f"{prefix}.sites.txt",
            f"{prefix}.sites.csv",
            f"{prefix}.bim",
        ):
            if Path(cand).is_file():
                return cand
    return None


def discover_id_sidecar_path(matrix_path: str, prefixes: Sequence[str]) -> str | None:
    candidates = [f"{matrix_path}.id"]
    for prefix in prefixes:
        candidates.append(f"{prefix}.id")

    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if Path(cand).is_file():
            return cand
    return None


def output_prefix_from_path(path: str) -> str:
    p = str(path)
    low = p.lower()
    if low.endswith(".vcf.gz"):
        return p[:-7]
    for ext in (".vcf", ".txt", ".tsv", ".csv", ".npy"):
        if low.endswith(ext):
            return p[: -len(ext)]
    return p


def basename_only(path: str) -> str:
    p = str(path).replace("\\", "/").rstrip("/")
    b = os.path.basename(p)
    return b if b else p


def read_id_file(
    path: str,
    logger: Any,
    label: str,
    *,
    use_spinner: bool = False,
    show_status: bool = True,
) -> np.ndarray | None:
    if not os.path.isfile(path):
        logger.warning(f"{label} ID file not found: {path}")
        return None
    src = basename_only(path)

    def _load_ids() -> np.ndarray | None:
        try:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                header=None,
                usecols=[0],
                dtype=str,
                keep_default_na=False,
            )
        except Exception:
            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
                header=None,
                usecols=[0],
                dtype=str,
                keep_default_na=False,
            )
        if not df.empty and df.iloc[0, 0] == "":
            df = pd.read_csv(
                path,
                sep=r"\s+",
                header=None,
                usecols=[0],
                dtype=str,
                keep_default_na=False,
            )
        if df.empty:
            logger.warning(f"{label} ID file is empty: {path}")
            return None
        ids0 = df.iloc[:, 0].astype(str).str.strip().to_numpy()
        if ids0.size == 0:
            logger.warning(f"{label} ID file has no usable IDs: {path}")
            return None
        return ids0

    if not show_status:
        return _load_ids()

    with CliStatus(f"Loading {label} ID from {src}...", enabled=bool(use_spinner)) as task:
        try:
            ids = _load_ids()
        except Exception:
            task.fail(f"Loading {label} ID from {src} ...Failed")
            raise
        if ids is None:
            task.complete(f"Loading {label} ID from {src} shape=(0,)")
            return None
        task.complete(f"Loading {label} ID from {src} shape=({ids.size},)")
    return ids


def read_matrix_with_ids(
    path: str,
    _logger: Any,
    label: str,
) -> tuple[np.ndarray | None, np.ndarray]:
    try:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            header=None,
            dtype={0: str},
            keep_default_na=False,
        )
    except Exception:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            dtype={0: str},
            keep_default_na=False,
        )
    if df.shape[1] < 2:
        raise ValueError(f"{label} file must have IDs in column 1 and data in columns 2+.")
    ids = df.iloc[:, 0].astype(str).str.strip().to_numpy()
    data = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
    return ids, data


def write_site_file(handle, sites: Sequence[SiteInfo]) -> None:
    for site in sites:
        handle.write(
            f"{str(site.chrom)}\t{int(site.pos)}\t{str(site.ref_allele)}\t{str(site.alt_allele)}\n"
        )


def write_id_file(path: str, sample_ids: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as fid:
        fid.write("\n".join(map(str, sample_ids)) + "\n")


def write_text_output(
    out_path: str,
    sample_ids: Sequence[str],
    chunks: Iterator[tuple[np.ndarray, Sequence[SiteInfo]]],
    *,
    total_sites: int,
) -> None:
    prefix = output_prefix_from_path(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_id_file(f"{prefix}.id", sample_ids)
    written = 0
    with open(out_path, "w", encoding="utf-8") as fw, open(
        f"{prefix}.site", "w", encoding="utf-8"
    ) as fsite:
        for block, sites in chunks:
            arr = np.asarray(block, dtype=np.float32)
            np.savetxt(fw, arr, fmt="%.6g", delimiter="\t")
            write_site_file(fsite, sites)
            written += int(arr.shape[0])
    if written != int(total_sites):
        raise RuntimeError(
            f"Written site count mismatch for text output: {written} vs {total_sites}"
        )


def write_npy_output(
    out_path: str,
    sample_ids: Sequence[str],
    chunks: Iterable[tuple[np.ndarray, Sequence[SiteInfo]]],
    *,
    total_sites: int,
) -> None:
    prefix = output_prefix_from_path(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_mm = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(total_sites), len(sample_ids)),
    )
    offset = 0
    write_id_file(f"{prefix}.id", sample_ids)
    with open(f"{prefix}.site", "w", encoding="utf-8") as fsite:
        for block, sites in chunks:
            arr = np.asarray(block, dtype=np.float32)
            n_rows = int(arr.shape[0])
            out_mm[offset : offset + n_rows, :] = arr
            write_site_file(fsite, sites)
            offset += n_rows
    out_mm.flush()
    if offset != int(total_sites):
        raise RuntimeError(
            f"Written site count mismatch for npy output: {offset} vs {total_sites}"
        )
