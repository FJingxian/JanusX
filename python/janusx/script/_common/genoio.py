from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from janusx.gfreader import SiteInfo, prepare_cli_input_cache
from .pathcheck import safe_expanduser
from .progress import build_rich_progress, rich_progress_available
from .status import CliStatus, stdout_is_tty

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # type: ignore[assignment]


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
        ".hmp.gz",
        ".vcf",
        ".hmp",
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
    elif low.endswith(".hmp.gz"):
        base = base[: -len(".hmp.gz")]
    else:
        for ext in (".vcf", ".hmp", ".txt", ".tsv", ".csv", ".npy"):
            if low.endswith(ext):
                base = base[: -len(ext)]
                break
    if base.startswith("~"):
        base = base[1:]
    return base


def determine_genotype_source(
    *,
    vcf: str | None,
    hmp: str | None,
    file: str | None,
    bfile: str | None,
    prefix: str | None = None,
    snps_only: bool = False,
    delimiter: str | None = None,
    apply_cache: bool = False,
) -> tuple[str, str]:
    if vcf:
        gfile = str(vcf)
        auto_prefix = strip_default_prefix_suffix(gfile)
    elif hmp:
        gfile = str(hmp)
        auto_prefix = strip_default_prefix_suffix(gfile)
    elif file:
        gfile = str(file)
        auto_prefix = strip_default_prefix_suffix(gfile)
    elif bfile:
        gfile = str(bfile)
        auto_prefix = os.path.basename(gfile.rstrip("/\\"))
    else:
        raise ValueError("No genotype input specified. Use -vcf, -hmp, -file or -bfile.")

    gfile_norm = gfile.replace("\\", "/")
    if apply_cache:
        gfile_norm = prepare_cli_input_cache(
            gfile_norm,
            snps_only=bool(snps_only),
            delimiter=delimiter,
        )
        gfile_norm = str(gfile_norm).replace("\\", "/")

    resolved_prefix = str(prefix) if prefix is not None else auto_prefix
    return gfile_norm, resolved_prefix


def determine_genotype_source_from_args(args: Any) -> tuple[str, str]:
    return determine_genotype_source(
        vcf=getattr(args, "vcf", None),
        hmp=getattr(args, "hmp", None),
        file=getattr(args, "file", None),
        bfile=getattr(args, "bfile", None),
        prefix=getattr(args, "prefix", None),
        snps_only=bool(getattr(args, "snps_only", False)),
        delimiter=getattr(args, "delimiter", None),
        # Cache conversion is delayed to first real genotype read so CLI can
        # print config/help promptly instead of blocking at startup.
        apply_cache=bool(getattr(args, "cache_input", False)),
    )


def build_prefix_candidates(
    file_arg: str,
    *,
    text_suffixes: Sequence[str] = GENOTYPE_TEXT_SUFFIXES,
) -> tuple[str, list[str]]:
    raw = safe_expanduser(file_arg)
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
    if low.endswith(".hmp.gz"):
        return p[:-7]
    for ext in (".vcf", ".hmp", ".txt", ".tsv", ".csv", ".npy"):
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


def _open_site_progress(desc: str, total_sites: int):
    use_tty = stdout_is_tty()
    if rich_progress_available():
        progress = build_rich_progress(
            description_template="[progress.description]{task.description}",
            show_remaining=False,
            bar_width=40,
            finished_text=" ",
            transient=True,
        )
        if progress is not None:
            task_id = progress.add_task(str(desc), total=float(max(0, int(total_sites))))
            progress.start()
            return progress, task_id, None
    if tqdm is None:
        return None, None, None
    pbar = tqdm(
        total=max(0, int(total_sites)),
        unit="SNP",
        desc=str(desc),
        disable=not use_tty,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    return None, None, pbar


def _advance_site_progress(progress, task_id, pbar, n_sites: int) -> None:
    n = int(max(0, n_sites))
    if progress is not None and task_id is not None:
        progress.update(task_id, advance=n)
    elif pbar is not None:
        pbar.update(n)


def _close_site_progress(progress, pbar) -> None:
    if progress is not None:
        progress.stop()
    if pbar is not None:
        pbar.close()


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
    progress, task_id, pbar = _open_site_progress("Writing TXT", int(total_sites))
    with open(out_path, "w", encoding="utf-8") as fw, open(
        f"{prefix}.site", "w", encoding="utf-8"
    ) as fsite:
        try:
            for block, sites in chunks:
                arr = np.asarray(block, dtype=np.float32)
                np.savetxt(fw, arr, fmt="%.6g", delimiter="\t")
                write_site_file(fsite, sites)
                n_rows = int(arr.shape[0])
                written += n_rows
                _advance_site_progress(progress, task_id, pbar, n_rows)
        finally:
            _close_site_progress(progress, pbar)
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
    progress, task_id, pbar = _open_site_progress("Writing NPY", int(total_sites))
    write_id_file(f"{prefix}.id", sample_ids)
    with open(f"{prefix}.site", "w", encoding="utf-8") as fsite:
        try:
            for block, sites in chunks:
                arr = np.asarray(block, dtype=np.float32)
                n_rows = int(arr.shape[0])
                out_mm[offset : offset + n_rows, :] = arr
                write_site_file(fsite, sites)
                offset += n_rows
                _advance_site_progress(progress, task_id, pbar, n_rows)
        finally:
            _close_site_progress(progress, pbar)
    out_mm.flush()
    if offset != int(total_sites):
        raise RuntimeError(
            f"Written site count mismatch for npy output: {offset} vs {total_sites}"
        )
