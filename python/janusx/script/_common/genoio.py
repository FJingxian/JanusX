from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from janusx.gfreader import (
    SiteInfo,
    inspect_genotype_file,
    prepare_bed_logic_meta_selected,
    prepare_cli_input_cache,
)
try:
    from janusx.gfreader import scan_bed_2bit_packed_stats
except Exception:
    scan_bed_2bit_packed_stats = None  # type: ignore[assignment]
try:
    from janusx.gfreader import prepare_bed_2bit_packed
except Exception:
    prepare_bed_2bit_packed = None  # type: ignore[assignment]

from .grmio import (
    format_grm_cache_num,
    genotype_cache_prefix as _shared_genotype_cache_prefix,
    grm_cache_lock,
    latest_genotype_mtime,
)
from .binsidecar import LEGACY_BSITE_HEADER_SIZE, LEGACY_BSITE_MAGIC, LEGACY_BSITE_VERSION
from .pathcheck import safe_expanduser
from .progress import (
    CliStatus,
    build_rich_progress,
    rich_progress_available,
    should_animate_status,
    stdout_is_tty,
)

try:
    from janusx.janusx import gfd_packbits_from_dosage_block as _gfd_packbits_from_dosage_block
except Exception:
    _gfd_packbits_from_dosage_block = None

try:
    from janusx.janusx import Bin01StreamWriter as _Bin01StreamWriter
except Exception:
    _Bin01StreamWriter = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # type: ignore[assignment]


GENOTYPE_TEXT_SUFFIXES: tuple[str, ...] = (".txt", ".tsv", ".csv")
GENOTYPE_SUFFIXES: tuple[str, ...] = GENOTYPE_TEXT_SUFFIXES + (".npy", ".bin")
PACKED_META_CACHE_VERSION = 1

_BYTE_LUT = np.arange(256, dtype=np.uint16)
_NONMISS_LUT = np.zeros((256,), dtype=np.uint8)
_HET_LUT = np.zeros((256,), dtype=np.uint8)
_ALT_SUM_LUT = np.zeros((256,), dtype=np.uint8)
for _within in range(4):
    _code = (_BYTE_LUT >> (_within * 2)) & 0b11
    _NONMISS_LUT += (_code != 0b01).astype(np.uint8)
    _HET_LUT += (_code == 0b10).astype(np.uint8)
    _ALT_SUM_LUT += ((_code == 0b10).astype(np.uint8) + (2 * (_code == 0b11).astype(np.uint8)))


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
        ".bin",
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
        for ext in (".vcf", ".hmp", ".txt", ".tsv", ".csv", ".npy", ".bin"):
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
    threads: int = 0,
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
        force_kind = determine_genotype_source_force_kind(
            vcf=vcf,
            hmp=hmp,
            file=file,
            bfile=bfile,
        )
        gfile_norm = prepare_cli_input_cache(
            gfile_norm,
            snps_only=bool(snps_only),
            delimiter=delimiter,
            force_kind=force_kind,
            threads=int(threads),
        )
        gfile_norm = str(gfile_norm).replace("\\", "/")

    resolved_prefix = str(prefix) if prefix is not None else auto_prefix
    return gfile_norm, resolved_prefix


def determine_genotype_source_force_kind(
    *,
    vcf: str | None,
    hmp: str | None,
    file: str | None,
    bfile: str | None,
) -> str | None:
    if vcf:
        return "vcf"
    if hmp:
        return "hmp"
    if file:
        return "file"
    if bfile:
        return "plink"
    return None


def determine_genotype_source_force_kind_from_args(args: Any) -> str | None:
    return determine_genotype_source_force_kind(
        vcf=getattr(args, "vcf", None),
        hmp=getattr(args, "hmp", None),
        file=getattr(args, "file", None),
        bfile=getattr(args, "bfile", None),
    )


def determine_genotype_source_from_args(args: Any) -> tuple[str, str]:
    return determine_genotype_source(
        vcf=getattr(args, "vcf", None),
        hmp=getattr(args, "hmp", None),
        file=getattr(args, "file", None),
        bfile=getattr(args, "bfile", None),
        prefix=getattr(args, "prefix", None),
        snps_only=bool(getattr(args, "snps_only", False)),
        delimiter=getattr(args, "delimiter", None),
        threads=int(getattr(args, "thread", 0) or 0),
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
    if raw.suffix.lower() == ".bin":
        raw_prefix = strip_known_suffix(str(raw), exts=(".bin",))
        noncache_prefix = str(Path(raw_prefix).parent / Path(raw_prefix).name.lstrip("~"))
        return noncache_prefix, [noncache_prefix, raw_prefix]

    raw_prefix = str(raw)
    cache_prefix = str(raw.parent / f"~{raw.name}")
    return raw_prefix, [raw_prefix, cache_prefix]


def discover_site_path(prefixes: Sequence[str]) -> str | None:
    for prefix in prefixes:
        for cand in (
            f"{prefix}.bsite",
            f"{prefix}.bin.site",
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
    candidates = [f"{matrix_path}.id", f"{matrix_path}.fam"]
    for prefix in prefixes:
        candidates.append(f"{prefix}.bin.id")
        candidates.append(f"{prefix}.id")
        candidates.append(f"{prefix}.fam")

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
    for ext in (".vcf", ".hmp", ".txt", ".tsv", ".csv", ".npy", ".bin"):
        if low.endswith(ext):
            return p[: -len(ext)]
    return p


def basename_only(path: str) -> str:
    p = str(path).replace("\\", "/").rstrip("/")
    b = os.path.basename(p)
    return b if b else p


def genotype_load_status_open(src: str) -> str:
    return f"Loading genotype from {src}..."


def genotype_load_status_fail(src: str) -> str:
    return f"Loading genotype from {src} ...Failed"


def genotype_load_status_progress(src: str, detail: str) -> str:
    return f"Loading genotype from {src}... {str(detail).strip()}"


def genotype_load_status_done(
    src: str,
    *,
    n_samples: int,
    n_snps: int,
    mode: str | None = None,
) -> str:
    base = f"Loading genotype from {src} (n={int(n_samples)}, nSNP={int(n_snps)}"
    if mode is not None and str(mode).strip() != "":
        return f"{base}, {str(mode).strip()})"
    return f"{base})"


def packed_prepare_failed_message(
    plink_prefix: str,
    attempt_errors: list[tuple[str, Exception]] | None = None,
) -> str:
    if not attempt_errors:
        return f"Packed BED preparation failed for {plink_prefix}."
    detail = "; ".join(
        f"{name}: {type(ex).__name__}: {ex}" for name, ex in attempt_errors
    )
    return f"Packed BED preparation failed for {plink_prefix}. attempts={detail}"


def packed_sample_size_mismatch_message(
    *,
    expected_n: int,
    got_n: int,
) -> str:
    return f"Packed sample size mismatch: packed n={int(got_n)}, expected {int(expected_n)}"


def packed_expected_sample_size_mismatch_message(
    *,
    expected_n: int,
    got_n: int,
    source_prefix: str,
) -> str:
    return (
        f"Packed sample size mismatch: expected {int(expected_n)}, "
        f"got {int(got_n)} from {source_prefix}."
    )


def packed_no_snps_after_filter_message() -> str:
    return "No SNPs left after packed BED filtering. Please relax --maf/--geno thresholds."


def normalize_plink_prefix(path_or_prefix: str) -> str:
    p = str(path_or_prefix).strip()
    low = p.lower()
    if low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam"):
        return p[:-4]
    return p


def _is_simple_snp_allele(allele: str) -> bool:
    a = str(allele).strip().upper()
    return len(a) == 1 and a in {"A", "C", "G", "T"}


def plink_snp_mask(prefix: str) -> np.ndarray:
    mask: list[bool] = []
    bim_path = f"{normalize_plink_prefix(prefix)}.bim"
    with open(bim_path, "r", encoding="utf-8", errors="ignore") as fh:
        for ln, line in enumerate(fh, start=1):
            s = line.strip()
            if s == "":
                continue
            toks = s.split()
            if len(toks) < 6:
                raise ValueError(f"Malformed BIM line at {bim_path}:{ln}")
            a0 = str(toks[4])
            a1 = str(toks[5])
            mask.append(_is_simple_snp_allele(a0) and _is_simple_snp_allele(a1))
    return np.asarray(mask, dtype=np.bool_)


def open_plink_bed_payload_memmap(
    prefix: str,
    *,
    n_samples: int,
    n_snps: int,
) -> np.memmap:
    bed_path, bytes_per_snp, _expected_size = inspect_plink_bed_payload_layout(
        prefix,
        n_samples=n_samples,
        n_snps=n_snps,
    )
    return np.memmap(
        bed_path,
        dtype=np.uint8,
        mode="r",
        offset=3,
        shape=(int(n_snps), int(bytes_per_snp)),
        order="C",
    )


def inspect_plink_bed_payload_layout(
    prefix: str,
    *,
    n_samples: int,
    n_snps: int,
) -> tuple[str, int, int]:
    plink_prefix = normalize_plink_prefix(prefix)
    bed_path = f"{plink_prefix}.bed"
    if not os.path.isfile(bed_path):
        raise ValueError(f"Cannot find BED file for PLINK prefix: {plink_prefix}")
    bytes_per_snp = (int(n_samples) + 3) // 4
    expected_size = 3 + int(n_snps) * int(bytes_per_snp)
    actual_size = int(os.path.getsize(bed_path))
    if actual_size != expected_size:
        raise ValueError(
            f"BED payload size mismatch: file={actual_size}, expected={expected_size} "
            f"(n_samples={int(n_samples)}, n_snps={int(n_snps)}, bytes_per_snp={int(bytes_per_snp)})"
        )
    return bed_path, int(bytes_per_snp), int(expected_size)


def load_plink_bed_payload_owned(
    prefix: str,
    *,
    n_samples: int,
    n_snps: int,
) -> np.ndarray:
    bed_path, bytes_per_snp, expected_size = inspect_plink_bed_payload_layout(
        prefix,
        n_samples=n_samples,
        n_snps=n_snps,
    )
    payload = np.empty((int(n_snps), int(bytes_per_snp)), dtype=np.uint8, order="C")
    payload_view = memoryview(payload).cast("B")
    payload_bytes = int(payload_view.nbytes)
    if payload_bytes != (expected_size - 3):
        raise ValueError(
            f"BED payload byte mismatch: array={payload_bytes}, expected={expected_size - 3}"
        )

    chunk_mb_raw = os.environ.get("JX_PACKED_OWNED_READ_CHUNK_MB", "64").strip()
    try:
        chunk_mb = float(chunk_mb_raw)
    except Exception:
        chunk_mb = 64.0
    if (not np.isfinite(chunk_mb)) or (chunk_mb <= 0.0):
        chunk_mb = 64.0
    chunk_bytes = max(1, int(chunk_mb * 1024.0 * 1024.0))

    with open(bed_path, "rb") as fh:
        fh.seek(3, os.SEEK_SET)
        offset = 0
        while offset < payload_bytes:
            end = min(payload_bytes, offset + chunk_bytes)
            n_read = fh.readinto(payload_view[offset:end])
            if n_read is None or int(n_read) <= 0:
                raise ValueError(
                    f"Unexpected EOF while reading BED payload from {bed_path}: "
                    f"read={offset}, expected={payload_bytes}"
                )
            offset += int(n_read)
    return payload


def packed_row_het_rate(packed: np.ndarray, n_samples: int) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(packed, dtype=np.uint8))
    if arr.ndim != 2:
        raise ValueError(f"packed matrix must be 2D, got shape={arr.shape}")
    n_rows, bytes_per_row = arr.shape
    expected = int((int(n_samples) + 3) // 4)
    if bytes_per_row != expected:
        raise ValueError(
            f"packed byte-width mismatch: got {bytes_per_row}, expected {expected} for n_samples={n_samples}"
        )
    if n_rows == 0:
        return np.zeros((0,), dtype=np.float32)

    full = int(n_samples) // 4
    rem = int(n_samples) % 4
    non_missing = np.zeros((n_rows,), dtype=np.int32)
    het = np.zeros((n_rows,), dtype=np.int32)
    if full > 0:
        base = arr[:, :full]
        non_missing += _NONMISS_LUT[base].sum(axis=1, dtype=np.int64).astype(np.int32, copy=False)
        het += _HET_LUT[base].sum(axis=1, dtype=np.int64).astype(np.int32, copy=False)

    if rem > 0:
        tail = arr[:, full]
        for within in range(rem):
            code = (tail >> (within * 2)) & 0b11
            non_missing += (code != 0b01).astype(np.int32, copy=False)
            het += (code == 0b10).astype(np.int32, copy=False)

    out = np.zeros((n_rows,), dtype=np.float32)
    ok = non_missing > 0
    out[ok] = het[ok] / non_missing[ok]
    return out


def _scan_bed_2bit_packed_stats_fallback(
    prefix: str,
    *,
    n_expected_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Stream packed BED row statistics from a memmap when the Rust-side scan
    helper is unavailable in the installed extension.
    """
    plink_prefix = normalize_plink_prefix(prefix)
    _sample_ids_check, n_snps = inspect_genotype_file(str(plink_prefix))
    n_samples = int(n_expected_samples)
    if n_samples <= 0:
        raise ValueError("Packed BED stats fallback requires n_samples > 0.")
    packed_memmap = open_plink_bed_payload_memmap(
        str(plink_prefix),
        n_samples=n_samples,
        n_snps=int(n_snps),
    )
    bytes_per_snp = int(packed_memmap.shape[1]) if packed_memmap.ndim == 2 else 0
    if bytes_per_snp <= 0:
        raise ValueError("Packed BED stats fallback found invalid bytes_per_snp.")

    chunk_mb_raw = os.environ.get("JX_PACKED_META_SCAN_CHUNK_MB", "64").strip()
    try:
        chunk_mb = float(chunk_mb_raw)
    except Exception:
        chunk_mb = 64.0
    if (not np.isfinite(chunk_mb)) or (chunk_mb <= 0.0):
        chunk_mb = 64.0
    chunk_target_bytes = max(1, int(chunk_mb * 1024.0 * 1024.0))
    chunk_rows = max(1, min(int(n_snps), chunk_target_bytes // max(1, bytes_per_snp)))

    full = n_samples // 4
    rem = n_samples % 4
    n_snps_i = int(n_snps)
    miss_arr = np.zeros((n_snps_i,), dtype=np.float32)
    maf_arr = np.zeros((n_snps_i,), dtype=np.float32)
    std_arr = np.zeros((n_snps_i,), dtype=np.float32)
    row_flip_arr = np.zeros((n_snps_i,), dtype=np.bool_)
    het_arr = np.zeros((n_snps_i,), dtype=np.float32)

    for row_start in range(0, n_snps_i, int(chunk_rows)):
        row_end = min(n_snps_i, row_start + int(chunk_rows))
        chunk = packed_memmap[row_start:row_end]
        n_rows = int(row_end - row_start)
        non_missing = np.zeros((n_rows,), dtype=np.int64)
        het = np.zeros((n_rows,), dtype=np.int64)
        alt_sum = np.zeros((n_rows,), dtype=np.int64)

        if full > 0:
            base = np.asarray(chunk[:, :full], dtype=np.uint8)
            non_missing += _NONMISS_LUT[base].sum(axis=1, dtype=np.int64)
            het += _HET_LUT[base].sum(axis=1, dtype=np.int64)
            alt_sum += _ALT_SUM_LUT[base].sum(axis=1, dtype=np.int64)

        if rem > 0:
            tail = np.asarray(chunk[:, full], dtype=np.uint8)
            for within in range(rem):
                code = (tail >> (within * 2)) & 0b11
                het_mask = (code == 0b10)
                hom_alt_mask = (code == 0b11)
                non_missing += (code != 0b01).astype(np.int64, copy=False)
                het += het_mask.astype(np.int64, copy=False)
                alt_sum += (
                    het_mask.astype(np.int64, copy=False)
                    + (2 * hom_alt_mask.astype(np.int64, copy=False))
                )

        missing = int(n_samples) - non_missing
        miss_arr[row_start:row_end] = (
            missing.astype(np.float32, copy=False) / float(n_samples)
        )
        ok = non_missing > 0
        if np.any(ok):
            p_alt = np.zeros((n_rows,), dtype=np.float32)
            het_chunk = np.zeros((n_rows,), dtype=np.float32)
            p_alt[ok] = alt_sum[ok].astype(np.float32, copy=False) / (
                2.0 * non_missing[ok].astype(np.float32, copy=False)
            )
            het_chunk[ok] = (
                het[ok].astype(np.float32, copy=False)
                / non_missing[ok].astype(np.float32, copy=False)
            )
            maf_arr[row_start:row_end] = np.minimum(
                p_alt,
                1.0 - p_alt,
            ).astype(np.float32, copy=False)
            std_arr[row_start:row_end] = np.sqrt(
                np.maximum(0.0, 2.0 * p_alt * (1.0 - p_alt))
            ).astype(np.float32, copy=False)
            row_flip_arr[row_start:row_end] = p_alt > 0.5
            het_arr[row_start:row_end] = het_chunk

    return (
        miss_arr,
        maf_arr,
        std_arr,
        row_flip_arr,
        het_arr,
        int(n_samples),
    )


def normalize_packed_filter_mode(mode: str) -> str:
    key = str(mode or "compact").strip().lower()
    if key in {"", "compact", "filtered", "subset"}:
        return "compact"
    if key in {"lazy_owned", "owned_full", "full_owned"}:
        return "lazy_owned"
    if key in {"lazy", "lazy_full", "full", "mask"}:
        return "lazy"
    if key == "auto":
        return "auto"
    raise ValueError(f"Unsupported packed filter_mode: {mode!r}")


def packed_preload_failure_state(prefix: str | None, reason: object) -> dict[str, Any]:
    return {
        "_packed_preload_disabled": True,
        "prefix": "" if prefix is None else str(prefix),
        "reason": str(reason),
    }


def packed_preload_is_disabled(obj: object) -> bool:
    return isinstance(obj, dict) and bool(obj.get("_packed_preload_disabled", False))


def packed_preload_is_ready(obj: object) -> bool:
    return (
        isinstance(obj, dict)
        and (not packed_preload_is_disabled(obj))
        and isinstance(obj.get("packed_ctx"), dict)
    )


def packed_preload_reason(obj: object) -> str:
    if isinstance(obj, dict):
        return str(obj.get("reason", ""))
    return ""


def _raise_packed_prepare_error(
    plink_prefix: str,
    attempt_errors: list[tuple[str, Exception]],
) -> None:
    if len(attempt_errors) == 0:
        raise RuntimeError(packed_prepare_failed_message(plink_prefix, None))
    last_ex = attempt_errors[-1][1]
    raise RuntimeError(packed_prepare_failed_message(plink_prefix, attempt_errors)) from last_ex


def packed_meta_active_row_idx(meta: dict[str, Any]) -> np.ndarray:
    raw = meta.get("active_row_idx", None)
    if raw is not None:
        arr = np.ascontiguousarray(np.asarray(raw, dtype=np.int64).reshape(-1), dtype=np.int64)
        if arr.size > 0:
            return arr
    site_keep_raw = meta.get("site_keep", None)
    if site_keep_raw is not None:
        site_keep = np.ascontiguousarray(
            np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        return np.ascontiguousarray(
            np.flatnonzero(site_keep).astype(np.int64, copy=False),
            dtype=np.int64,
        )
    n = int(meta.get("n_active_sites", 0) or 0)
    return np.ascontiguousarray(np.arange(max(0, n), dtype=np.int64), dtype=np.int64)


def packed_meta_filter_param_tag(
    *,
    maf: float,
    missing_rate: float,
    het_threshold: float,
    snps_only: bool,
) -> str:
    return (
        f".maf{format_grm_cache_num(float(maf))}"
        f".geno{format_grm_cache_num(float(missing_rate))}"
        f".het{format_grm_cache_num(float(het_threshold))}"
        f".snp{1 if bool(snps_only) else 0}"
    )


def packed_meta_cache_prefix(
    prefix: str,
    *,
    maf: float,
    missing_rate: float,
    het_threshold: float = 0.0,
    snps_only: bool = False,
    cache_dir: str | None = None,
    logger: Any = None,
    warning_collector: list[str] | None = None,
) -> str:
    plink_prefix = normalize_plink_prefix(prefix)
    base = _shared_genotype_cache_prefix(
        plink_prefix,
        snps_only=bool(snps_only),
        cache_dir=cache_dir,
        logger=logger,
        warning_collector=warning_collector,
    )
    return (
        f"{base}"
        f"{packed_meta_filter_param_tag(maf=float(maf), missing_rate=float(missing_rate), het_threshold=float(het_threshold), snps_only=bool(snps_only))}"
    )


def _packed_meta_cache_paths(cache_prefix: str) -> dict[str, str]:
    stem = f"{cache_prefix}.pmeta"
    return {
        "json": f"{stem}.json",
        "site_keep": f"{stem}.site_keep.npy",
        "row_missing": f"{stem}.row_missing.npy",
        "row_maf": f"{stem}.row_maf.npy",
        "row_flip": f"{stem}.row_flip.npy",
        "std_denom": f"{stem}.std_denom.npy",
        "dom_af": f"{stem}.dom_af.npy",
    }


def _packed_meta_required_keys(layer: str) -> tuple[str, ...]:
    key = str(layer).strip().lower()
    if key == "basic":
        return ("site_keep", "row_missing", "row_maf", "row_flip")
    if key == "std":
        return ("site_keep", "row_missing", "row_maf", "row_flip", "std_denom")
    if key == "dom":
        return ("site_keep", "row_missing", "row_maf", "row_flip", "std_denom", "dom_af")
    raise ValueError(f"Unsupported packed meta layer: {layer!r}")


def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
    tmp = f"{path}.tmp.{os.getpid()}.npy"
    np.save(tmp, np.asarray(arr))
    os.replace(tmp, path)


def _atomic_save_json(path: str, payload: dict[str, Any]) -> None:
    tmp = f"{path}.tmp.{os.getpid()}.json"
    with open(tmp, "w", encoding="utf-8") as fw:
        json.dump(payload, fw, indent=2, sort_keys=True)
        fw.write("\n")
    os.replace(tmp, path)


def _load_npy_checked(path: str, *, dtype: np.dtype | type) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(np.load(path), dtype=dtype).reshape(-1), dtype=dtype)


def _build_packed_meta_context(
    *,
    source_prefix: str,
    n_samples: int,
    row_missing: np.ndarray,
    row_maf: np.ndarray,
    row_flip: np.ndarray,
    site_keep: np.ndarray,
    std_denom: np.ndarray | None = None,
    dom_af: np.ndarray | None = None,
) -> dict[str, Any]:
    active_row_idx = np.ascontiguousarray(
        np.flatnonzero(site_keep).astype(np.int64, copy=False),
        dtype=np.int64,
    )
    ctx: dict[str, Any] = {
        "missing_rate": np.ascontiguousarray(np.asarray(row_missing, dtype=np.float32).reshape(-1), dtype=np.float32),
        "af": np.ascontiguousarray(np.asarray(row_maf, dtype=np.float32).reshape(-1), dtype=np.float32),
        "maf": np.ascontiguousarray(np.asarray(row_maf, dtype=np.float32).reshape(-1), dtype=np.float32),
        "row_flip": np.ascontiguousarray(np.asarray(row_flip, dtype=np.bool_).reshape(-1), dtype=np.bool_),
        "site_keep": np.ascontiguousarray(np.asarray(site_keep, dtype=np.bool_).reshape(-1), dtype=np.bool_),
        "active_row_idx": active_row_idx,
        "n_samples": int(n_samples),
        "n_total_sites": int(np.asarray(site_keep).reshape(-1).shape[0]),
        "n_active_sites": int(active_row_idx.shape[0]),
        "packed_filter_mode": "stats_only",
        "packed_storage": "metadata",
        "source_prefix": str(source_prefix),
    }
    if std_denom is not None:
        ctx["std_denom"] = np.ascontiguousarray(
            np.asarray(std_denom, dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
    if dom_af is not None:
        ctx["dom_af"] = np.ascontiguousarray(
            np.asarray(dom_af, dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
    return ctx


def _load_cached_packed_meta(
    *,
    cache_prefix: str,
    layer: str,
    source_prefix: str,
    expected_n_samples: int | None,
) -> dict[str, Any] | None:
    paths = _packed_meta_cache_paths(cache_prefix)
    json_path = paths["json"]
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as fr:
            meta_info = json.load(fr)
    except Exception:
        return None
    if int(meta_info.get("version", -1)) != int(PACKED_META_CACHE_VERSION):
        return None
    current_source_mtime = latest_genotype_mtime(str(source_prefix))
    cached_source_mtime = meta_info.get("source_mtime", None)
    if (
        current_source_mtime is not None
        and cached_source_mtime is not None
        and float(cached_source_mtime) + 1e-9 < float(current_source_mtime)
    ):
        return None
    n_samples_cached = meta_info.get("n_samples", None)
    if expected_n_samples is not None and n_samples_cached is not None:
        if int(n_samples_cached) != int(expected_n_samples):
            return None
    required = _packed_meta_required_keys(layer)
    if any(not os.path.isfile(paths[key]) for key in required):
        return None
    try:
        row_missing = _load_npy_checked(paths["row_missing"], dtype=np.float32)
        row_maf = _load_npy_checked(paths["row_maf"], dtype=np.float32)
        row_flip = _load_npy_checked(paths["row_flip"], dtype=np.bool_)
        site_keep = _load_npy_checked(paths["site_keep"], dtype=np.bool_)
        std_denom = (
            _load_npy_checked(paths["std_denom"], dtype=np.float32)
            if "std_denom" in required
            else None
        )
        dom_af = (
            _load_npy_checked(paths["dom_af"], dtype=np.float32)
            if "dom_af" in required
            else None
        )
    except Exception:
        return None

    n_total = int(site_keep.shape[0])
    for _arr_name, arr in (
        ("row_missing", row_missing),
        ("row_maf", row_maf),
        ("row_flip", row_flip),
        ("std_denom", std_denom),
        ("dom_af", dom_af),
    ):
        if arr is None:
            continue
        if int(arr.shape[0]) != n_total:
            return None
    return _build_packed_meta_context(
        source_prefix=str(source_prefix),
        n_samples=int(meta_info.get("n_samples", expected_n_samples or 0)),
        row_missing=row_missing,
        row_maf=row_maf,
        row_flip=row_flip,
        site_keep=site_keep,
        std_denom=std_denom,
        dom_af=dom_af,
    )


def _save_packed_meta_cache(
    *,
    cache_prefix: str,
    layer: str,
    source_prefix: str,
    ctx: dict[str, Any],
) -> None:
    paths = _packed_meta_cache_paths(cache_prefix)
    required = _packed_meta_required_keys(layer)
    _atomic_save_npy(paths["site_keep"], np.asarray(ctx["site_keep"], dtype=np.bool_))
    _atomic_save_npy(paths["row_missing"], np.asarray(ctx["missing_rate"], dtype=np.float32))
    _atomic_save_npy(paths["row_maf"], np.asarray(ctx["maf"], dtype=np.float32))
    _atomic_save_npy(paths["row_flip"], np.asarray(ctx["row_flip"], dtype=np.bool_))
    if "std_denom" in required:
        _atomic_save_npy(paths["std_denom"], np.asarray(ctx["std_denom"], dtype=np.float32))
    if "dom_af" in required:
        _atomic_save_npy(paths["dom_af"], np.asarray(ctx["dom_af"], dtype=np.float32))
    layer_order = ["basic"]
    if layer in {"std", "dom"}:
        layer_order.append("std")
    if layer == "dom":
        layer_order.append("dom")
    _atomic_save_json(
        paths["json"],
        {
            "version": int(PACKED_META_CACHE_VERSION),
            "source_prefix": str(source_prefix),
            "source_mtime": latest_genotype_mtime(str(source_prefix)),
            "n_samples": int(ctx.get("n_samples", 0) or 0),
            "n_total_sites": int(ctx.get("n_total_sites", 0) or 0),
            "n_active_sites": int(ctx.get("n_active_sites", 0) or 0),
            "layers": layer_order,
        },
    )


def _build_packed_meta_layer(
    prefix: str,
    *,
    sample_ids_arr: np.ndarray,
    maf: float,
    missing_rate: float,
    het_threshold: float,
    snps_only: bool,
    layer: str,
) -> dict[str, Any]:
    plink_prefix = normalize_plink_prefix(prefix)
    n_expected = int(sample_ids_arr.shape[0])
    want_std = str(layer).strip().lower() in {"std", "dom"}
    want_dom = str(layer).strip().lower() == "dom"

    if scan_bed_2bit_packed_stats is not None:
        miss_raw, maf_raw, std_raw, row_flip_raw, het_raw, packed_n = scan_bed_2bit_packed_stats(
            str(plink_prefix)
        )
    else:
        miss_raw, maf_raw, std_raw, row_flip_raw, het_raw, packed_n = _scan_bed_2bit_packed_stats_fallback(
            str(plink_prefix),
            n_expected_samples=int(n_expected),
        )
    if int(packed_n) != n_expected:
        raise ValueError(
            packed_sample_size_mismatch_message(
                expected_n=int(n_expected),
                got_n=int(packed_n),
            )
        )
    miss_arr = np.ascontiguousarray(np.asarray(miss_raw, dtype=np.float32).reshape(-1), dtype=np.float32)
    maf_minor = np.ascontiguousarray(np.asarray(maf_raw, dtype=np.float32).reshape(-1), dtype=np.float32)
    row_flip_input = np.ascontiguousarray(
        np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1),
        dtype=np.bool_,
    )
    row_maf = np.ascontiguousarray(
        np.where(row_flip_input, 1.0 - maf_minor, maf_minor).astype(np.float32, copy=False),
        dtype=np.float32,
    )
    row_flip = np.zeros_like(row_flip_input, dtype=np.bool_)
    std_arr = (
        np.ascontiguousarray(np.asarray(std_raw, dtype=np.float32).reshape(-1), dtype=np.float32)
        if want_std
        else None
    )
    het_arr = np.ascontiguousarray(np.asarray(het_raw, dtype=np.float32).reshape(-1), dtype=np.float32)
    keep = np.ones((int(maf_minor.shape[0]),), dtype=np.bool_)
    maf_thr = float(maf)
    if maf_thr > 0.0:
        keep &= (maf_minor >= maf_thr) & (maf_minor <= (1.0 - maf_thr))
    miss_thr = float(missing_rate)
    if miss_thr < 1.0:
        keep &= miss_arr <= miss_thr
    het_thr = float(het_threshold)
    if het_thr > 0.0:
        keep &= het_arr <= het_thr
    if bool(snps_only):
        snp_mask = plink_snp_mask(str(plink_prefix))
        if snp_mask.shape[0] != keep.shape[0]:
            raise ValueError(
                f"BIM SNP mask length mismatch: got {snp_mask.shape[0]}, expected {keep.shape[0]}"
            )
        keep &= snp_mask
    if not np.any(keep):
        raise ValueError(packed_no_snps_after_filter_message())
    return _build_packed_meta_context(
        source_prefix=str(plink_prefix),
        n_samples=int(packed_n),
        row_missing=miss_arr,
        row_maf=row_maf,
        row_flip=row_flip,
        site_keep=np.ascontiguousarray(keep, dtype=np.bool_),
        std_denom=std_arr,
        dom_af=het_arr if want_dom else None,
    )


def _load_or_build_packed_meta_layer(
    prefix: str,
    *,
    maf: float,
    missing_rate: float,
    het_threshold: float = 0.0,
    snps_only: bool = False,
    expected_n_samples: int | None = None,
    layer: str,
    cache_dir: str | None = None,
    logger: Any = None,
    warning_collector: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    plink_prefix = normalize_plink_prefix(prefix)
    sample_ids, _ = inspect_genotype_file(str(plink_prefix))
    sample_ids_arr = np.asarray(sample_ids, dtype=str)
    if expected_n_samples is not None and int(expected_n_samples) != int(sample_ids_arr.shape[0]):
        raise ValueError(
            packed_expected_sample_size_mismatch_message(
                expected_n=int(expected_n_samples),
                got_n=int(sample_ids_arr.shape[0]),
                source_prefix=str(plink_prefix),
            )
        )
    cache_prefix = packed_meta_cache_prefix(
        str(plink_prefix),
        maf=float(maf),
        missing_rate=float(missing_rate),
        het_threshold=float(het_threshold),
        snps_only=bool(snps_only),
        cache_dir=cache_dir,
        logger=logger,
        warning_collector=warning_collector,
    )
    cached = _load_cached_packed_meta(
        cache_prefix=str(cache_prefix),
        layer=str(layer),
        source_prefix=str(plink_prefix),
        expected_n_samples=int(sample_ids_arr.shape[0]),
    )
    if cached is not None:
        return sample_ids_arr, cached
    with grm_cache_lock(str(cache_prefix)):
        cached = _load_cached_packed_meta(
            cache_prefix=str(cache_prefix),
            layer=str(layer),
            source_prefix=str(plink_prefix),
            expected_n_samples=int(sample_ids_arr.shape[0]),
        )
        if cached is not None:
            return sample_ids_arr, cached
        built = _build_packed_meta_layer(
            str(plink_prefix),
            sample_ids_arr=sample_ids_arr,
            maf=float(maf),
            missing_rate=float(missing_rate),
            het_threshold=float(het_threshold),
            snps_only=bool(snps_only),
            layer=str(layer),
        )
        _save_packed_meta_cache(
            cache_prefix=str(cache_prefix),
            layer=str(layer),
            source_prefix=str(plink_prefix),
            ctx=built,
        )
        return sample_ids_arr, built


def load_or_build_packed_meta_basic(
    prefix: str,
    *,
    maf: float,
    missing_rate: float,
    het_threshold: float = 0.0,
    snps_only: bool = False,
    expected_n_samples: int | None = None,
    cache_dir: str | None = None,
    logger: Any = None,
    warning_collector: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _load_or_build_packed_meta_layer(
        prefix,
        maf=float(maf),
        missing_rate=float(missing_rate),
        het_threshold=float(het_threshold),
        snps_only=bool(snps_only),
        expected_n_samples=expected_n_samples,
        layer="basic",
        cache_dir=cache_dir,
        logger=logger,
        warning_collector=warning_collector,
    )


def build_packed_meta_basic_uncached(
    prefix: str,
    *,
    maf: float,
    missing_rate: float,
    het_threshold: float = 0.0,
    snps_only: bool = False,
    expected_n_samples: int | None = None,
    mmap_window_mb: int | None = None,
    threads: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Prepare basic packed BED filter metadata entirely in memory.

    This avoids the `.pmeta.*` cache files and is intended for workflows that
    prefer a fresh precompute per run.
    """
    plink_prefix = normalize_plink_prefix(prefix)
    sample_ids, _ = inspect_genotype_file(str(plink_prefix))
    sample_ids_arr = np.asarray(sample_ids, dtype=str)
    if expected_n_samples is not None and int(expected_n_samples) != int(sample_ids_arr.shape[0]):
        raise ValueError(
            packed_expected_sample_size_mismatch_message(
                expected_n=int(expected_n_samples),
                got_n=int(sample_ids_arr.shape[0]),
                source_prefix=str(plink_prefix),
            )
        )

    try:
        row_idx, miss, maf_arr, row_flip, site_keep, packed_n, n_total_sites = prepare_bed_logic_meta_selected(
            str(plink_prefix),
            sample_indices=None,
            maf_threshold=float(maf),
            max_missing_rate=float(missing_rate),
            het_threshold=float(het_threshold),
            snps_only=bool(snps_only),
            mmap_window_mb=(
                None if mmap_window_mb is None else int(max(1, int(mmap_window_mb)))
            ),
            threads=(
                1
                if threads is None
                else int(max(1, int(threads)))
            ),
        )
        if int(packed_n) != int(sample_ids_arr.shape[0]):
            raise ValueError(
                packed_sample_size_mismatch_message(
                    expected_n=int(sample_ids_arr.shape[0]),
                    got_n=int(packed_n),
                )
            )

        active_row_idx = np.ascontiguousarray(
            np.asarray(row_idx, dtype=np.int64).reshape(-1),
            dtype=np.int64,
        )
        missing_arr = np.ascontiguousarray(
            np.asarray(miss, dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
        maf_selected = np.ascontiguousarray(
            np.asarray(maf_arr, dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
        row_flip_arr = np.ascontiguousarray(
            np.asarray(row_flip, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        site_keep_arr = np.ascontiguousarray(
            np.asarray(site_keep, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        n_total = int(n_total_sites)
        if int(site_keep_arr.shape[0]) != n_total:
            raise ValueError(
                f"Packed metadata site_keep length mismatch: got {int(site_keep_arr.shape[0])}, "
                f"expected {n_total}."
            )
        n_active = int(active_row_idx.shape[0])
        if (
            int(missing_arr.shape[0]) != n_active
            or int(maf_selected.shape[0]) != n_active
            or int(row_flip_arr.shape[0]) != n_active
        ):
            raise ValueError(
                "Packed metadata length mismatch: "
                f"row_indices={n_active}, missing={int(missing_arr.shape[0])}, "
                f"maf={int(maf_selected.shape[0])}, row_flip={int(row_flip_arr.shape[0])}."
            )
        if n_active <= 0:
            raise ValueError(packed_no_snps_after_filter_message())
        if int(np.count_nonzero(site_keep_arr)) != n_active:
            raise ValueError(
                "Packed metadata keep-mask mismatch: "
                f"row_indices={n_active}, site_keep_true={int(np.count_nonzero(site_keep_arr))}."
            )
        return sample_ids_arr, {
            "missing_rate": missing_arr,
            "af": maf_selected,
            "maf": maf_selected,
            "row_flip": row_flip_arr,
            "site_keep": site_keep_arr,
            "active_row_idx": active_row_idx,
            "n_samples": int(packed_n),
            "n_total_sites": n_total,
            "n_active_sites": n_active,
            "packed_filter_mode": "stats_only",
            "packed_storage": "metadata",
            "source_prefix": str(plink_prefix),
        }
    except Exception:
        built = _build_packed_meta_layer(
            str(plink_prefix),
            sample_ids_arr=sample_ids_arr,
            maf=float(maf),
            missing_rate=float(missing_rate),
            het_threshold=float(het_threshold),
            snps_only=bool(snps_only),
            layer="basic",
        )
        return sample_ids_arr, built


def load_or_build_packed_meta_std(
    prefix: str,
    *,
    maf: float,
    missing_rate: float,
    het_threshold: float = 0.0,
    snps_only: bool = False,
    expected_n_samples: int | None = None,
    cache_dir: str | None = None,
    logger: Any = None,
    warning_collector: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _load_or_build_packed_meta_layer(
        prefix,
        maf=float(maf),
        missing_rate=float(missing_rate),
        het_threshold=float(het_threshold),
        snps_only=bool(snps_only),
        expected_n_samples=expected_n_samples,
        layer="std",
        cache_dir=cache_dir,
        logger=logger,
        warning_collector=warning_collector,
    )


def load_or_build_packed_meta_dom(
    prefix: str,
    *,
    maf: float,
    missing_rate: float,
    het_threshold: float = 0.0,
    snps_only: bool = False,
    expected_n_samples: int | None = None,
    cache_dir: str | None = None,
    logger: Any = None,
    warning_collector: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _load_or_build_packed_meta_layer(
        prefix,
        maf=float(maf),
        missing_rate=float(missing_rate),
        het_threshold=float(het_threshold),
        snps_only=bool(snps_only),
        expected_n_samples=expected_n_samples,
        layer="dom",
        cache_dir=cache_dir,
        logger=logger,
        warning_collector=warning_collector,
    )


def prepare_packed_ctx_from_plink(
    prefix: str,
    *,
    maf: float,
    missing_rate: float,
    het_threshold: float = 0.0,
    snps_only: bool = False,
    expected_n_samples: int | None = None,
    filter_mode: str = "compact",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load/filter PLINK BED into a packed context for packed/full-Rust GWAS routes.

    The metadata/filter semantics are shared with the layered packed-meta cache in
    this module. When the Rust-side compact loader is unavailable or when lazy
    payload modes are requested, this function materializes the packed payload on
    top of the same cached metadata layer.
    """
    filter_mode_key = normalize_packed_filter_mode(filter_mode)
    plink_prefix = normalize_plink_prefix(prefix)
    sample_ids, _ = inspect_genotype_file(str(plink_prefix))
    sample_ids_arr = np.asarray(sample_ids, dtype=str)
    attempt_errors: list[tuple[str, Exception]] = []

    if expected_n_samples is not None and int(expected_n_samples) != int(sample_ids_arr.shape[0]):
        raise ValueError(
            packed_expected_sample_size_mismatch_message(
                expected_n=int(expected_n_samples),
                got_n=int(sample_ids_arr.shape[0]),
                source_prefix=str(plink_prefix),
            )
        )

    if (filter_mode_key == "compact") and (prepare_bed_2bit_packed is not None):
        try:
            (
                packed_raw,
                miss_raw,
                maf_raw,
                std_raw,
                row_flip_raw,
                site_keep_raw,
                packed_n,
                total_sites,
            ) = prepare_bed_2bit_packed(
                str(plink_prefix),
                maf_threshold=float(maf),
                max_missing_rate=float(missing_rate),
                het_threshold=float(het_threshold),
                snps_only=bool(snps_only),
            )
            if int(packed_n) != int(sample_ids_arr.shape[0]):
                raise ValueError(
                    packed_sample_size_mismatch_message(
                        expected_n=int(sample_ids_arr.shape[0]),
                        got_n=int(packed_n),
                    )
                )
            packed_ctx: dict[str, Any] = {
                "packed": np.ascontiguousarray(np.asarray(packed_raw, dtype=np.uint8)),
                "missing_rate": np.ascontiguousarray(
                    np.asarray(miss_raw, dtype=np.float32).reshape(-1),
                    dtype=np.float32,
                ),
                "af": np.ascontiguousarray(
                    np.asarray(maf_raw, dtype=np.float32).reshape(-1),
                    dtype=np.float32,
                ),
                "maf": np.ascontiguousarray(
                    np.asarray(maf_raw, dtype=np.float32).reshape(-1),
                    dtype=np.float32,
                ),
                "std_denom": np.ascontiguousarray(
                    np.asarray(std_raw, dtype=np.float32).reshape(-1),
                    dtype=np.float32,
                ),
                "dom_af": np.ascontiguousarray(
                    packed_row_het_rate(
                        np.ascontiguousarray(np.asarray(packed_raw, dtype=np.uint8)),
                        int(packed_n),
                    ),
                    dtype=np.float32,
                ),
                "row_flip": np.ascontiguousarray(
                    np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1),
                    dtype=np.bool_,
                ),
                "site_keep": np.ascontiguousarray(
                    np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1),
                    dtype=np.bool_,
                ),
                "n_samples": int(packed_n),
                "n_total_sites": int(total_sites),
                "n_active_sites": int(np.asarray(maf_raw).reshape(-1).shape[0]),
                "active_row_idx": np.ascontiguousarray(
                    np.flatnonzero(
                        np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1)
                    ).astype(np.int64, copy=False),
                    dtype=np.int64,
                ),
                "packed_filter_mode": "compact",
                "packed_storage": "owned",
                "source_prefix": str(plink_prefix),
            }
            return sample_ids_arr, packed_ctx
        except Exception as ex:
            attempt_errors.append(("prepare_bed_2bit_packed", ex))

    try:
        _sample_ids_meta, meta_ctx = load_or_build_packed_meta_dom(
            str(plink_prefix),
            maf=float(maf),
            missing_rate=float(missing_rate),
            het_threshold=float(het_threshold),
            snps_only=bool(snps_only),
            expected_n_samples=int(sample_ids_arr.shape[0]),
        )
        site_keep = np.ascontiguousarray(
            np.asarray(meta_ctx["site_keep"], dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        keep_idx = packed_meta_active_row_idx(meta_ctx)
        keep_ratio = (
            float(keep_idx.shape[0]) / float(site_keep.shape[0])
            if int(site_keep.shape[0]) > 0
            else 0.0
        )
        use_lazy = False
        if filter_mode_key in {"lazy", "lazy_owned"}:
            use_lazy = True
        elif filter_mode_key == "auto":
            lazy_keep_ratio = float(
                os.environ.get("JX_PACKED_LAZY_KEEP_RATIO", "0.98").strip() or "0.98"
            )
            if not np.isfinite(lazy_keep_ratio):
                lazy_keep_ratio = 0.98
            lazy_keep_ratio = min(1.0, max(0.0, lazy_keep_ratio))
            use_lazy = bool(keep_ratio >= lazy_keep_ratio)

        packed_memmap: np.memmap | None = None
        if filter_mode_key != "lazy_owned":
            packed_memmap = open_plink_bed_payload_memmap(
                str(plink_prefix),
                n_samples=int(meta_ctx["n_samples"]),
                n_snps=int(meta_ctx["n_total_sites"]),
            )
        if use_lazy:
            if filter_mode_key == "lazy_owned":
                packed_payload = load_plink_bed_payload_owned(
                    str(plink_prefix),
                    n_samples=int(meta_ctx["n_samples"]),
                    n_snps=int(meta_ctx["n_total_sites"]),
                )
                packed_storage = "owned"
            else:
                assert packed_memmap is not None
                packed_payload = packed_memmap
                packed_storage = "memmap"
            packed_ctx = dict(meta_ctx)
            packed_ctx["packed"] = packed_payload
            packed_ctx["packed_filter_mode"] = "lazy_full"
            packed_ctx["packed_storage"] = packed_storage
            return sample_ids_arr, packed_ctx

        if packed_memmap is None:
            packed_memmap = open_plink_bed_payload_memmap(
                str(plink_prefix),
                n_samples=int(meta_ctx["n_samples"]),
                n_snps=int(meta_ctx["n_total_sites"]),
            )
        if bool(np.all(site_keep)):
            packed = load_plink_bed_payload_owned(
                str(plink_prefix),
                n_samples=int(meta_ctx["n_samples"]),
                n_snps=int(meta_ctx["n_total_sites"]),
            )
            packed_ctx = dict(meta_ctx)
        else:
            packed = np.ascontiguousarray(
                np.asarray(packed_memmap[site_keep], dtype=np.uint8),
                dtype=np.uint8,
            )
            packed_ctx = dict(meta_ctx)
            packed_ctx["missing_rate"] = np.ascontiguousarray(
                np.asarray(meta_ctx["missing_rate"], dtype=np.float32).reshape(-1)[site_keep],
                dtype=np.float32,
            )
            packed_ctx["af"] = np.ascontiguousarray(
                np.asarray(meta_ctx["af"], dtype=np.float32).reshape(-1)[site_keep],
                dtype=np.float32,
            )
            packed_ctx["maf"] = np.ascontiguousarray(
                np.asarray(meta_ctx["maf"], dtype=np.float32).reshape(-1)[site_keep],
                dtype=np.float32,
            )
            packed_ctx["row_flip"] = np.ascontiguousarray(
                np.asarray(meta_ctx["row_flip"], dtype=np.bool_).reshape(-1)[site_keep],
                dtype=np.bool_,
            )
            if "std_denom" in meta_ctx:
                packed_ctx["std_denom"] = np.ascontiguousarray(
                    np.asarray(meta_ctx["std_denom"], dtype=np.float32).reshape(-1)[site_keep],
                    dtype=np.float32,
                )
            if "dom_af" in meta_ctx:
                packed_ctx["dom_af"] = np.ascontiguousarray(
                    np.asarray(meta_ctx["dom_af"], dtype=np.float32).reshape(-1)[site_keep],
                    dtype=np.float32,
                )
        packed_ctx["packed"] = packed
        packed_ctx["active_row_idx"] = keep_idx
        packed_ctx["packed_filter_mode"] = "compact"
        packed_ctx["packed_storage"] = "owned"
        return sample_ids_arr, packed_ctx
    except Exception as ex:
        attempt_errors.append(("load_or_build_packed_meta_dom", ex))
        _raise_packed_prepare_error(str(plink_prefix), attempt_errors)


def prepare_packed_stats_ctx_from_plink(
    prefix: str,
    *,
    maf: float,
    missing_rate: float,
    het_threshold: float = 0.0,
    snps_only: bool = False,
    expected_n_samples: int | None = None,
    use_cache: bool = True,
    mmap_window_mb: int | None = None,
    threads: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Prepare packed BED filter metadata without attaching the packed payload.
    """
    if bool(use_cache):
        return load_or_build_packed_meta_basic(
            str(normalize_plink_prefix(prefix)),
            maf=float(maf),
            missing_rate=float(missing_rate),
            het_threshold=float(het_threshold),
            snps_only=bool(snps_only),
            expected_n_samples=expected_n_samples,
        )
    return build_packed_meta_basic_uncached(
        str(normalize_plink_prefix(prefix)),
        maf=float(maf),
        missing_rate=float(missing_rate),
        het_threshold=float(het_threshold),
        snps_only=bool(snps_only),
        expected_n_samples=expected_n_samples,
        mmap_window_mb=mmap_window_mb,
        threads=threads,
    )


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
    use_fam_iid = str(path).lower().endswith(".fam")

    def _load_ids() -> np.ndarray | None:
        usecols = [1] if use_fam_iid else [0]
        try:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                header=None,
                usecols=usecols,
                dtype=str,
                keep_default_na=False,
            )
        except Exception:
            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
                header=None,
                usecols=usecols,
                dtype=str,
                keep_default_na=False,
            )
        if not df.empty and df.iloc[0, 0] == "":
            df = pd.read_csv(
                path,
                sep=r"\s+",
                header=None,
                usecols=usecols,
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


def write_fam_file(path: str, sample_ids: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as ffam:
        for sid in sample_ids:
            s = str(sid)
            ffam.write(f"{s}\t{s}\t0\t0\t0\t-9\n")


def write_bim_file(path: str, sites: Sequence[SiteInfo]) -> None:
    with open(path, "w", encoding="utf-8") as fbim:
        for site in sites:
            chrom = str(site.chrom)
            try:
                pos = int(site.pos)
            except Exception:
                pos = 0
            sid = f"{chrom}_{pos}"
            ref = str(site.ref_allele)
            alt = str(site.alt_allele)
            fbim.write(f"{chrom}\t{sid}\t0\t{pos}\t{ref}\t{alt}\n")

def _require_bin01_writer():
    if _Bin01StreamWriter is None:
        raise RuntimeError(
            "Bin01StreamWriter is unavailable in janusx extension. "
            "Rebuild/install the Rust extension before writing BIN01 outputs."
        )
    return _Bin01StreamWriter


def _site_columns_for_bin_writer(
    sites: Sequence[SiteInfo],
) -> tuple[list[str], list[int], list[str], list[str]]:
    chrom: list[str] = []
    pos: list[int] = []
    ref: list[str] = []
    alt: list[str] = []
    for site in sites:
        chrom.append(str(site.chrom))
        try:
            pos.append(int(site.pos))
        except Exception:
            pos.append(0)
        ref.append(str(site.ref_allele))
        alt.append(str(site.alt_allele))
    return chrom, pos, ref, alt

_ALLELE_CHAR_TO_CODE: dict[str, int] = {
    "A": 0,
    "T": 1,
    "C": 2,
    "G": 3,
    "N": 4,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
}

def _split_chrom_strand(chrom: str) -> tuple[str, int]:
    s = str(chrom).strip()
    if s.endswith("_1"):
        return s[:-2], 0
    if s.endswith("_2"):
        return s[:-2], 1
    if s.endswith("-"):
        return s[:-1], 0
    if s.endswith("+"):
        return s[:-1], 1
    return s, 1


def _encode_allele_nibbles(seq: str) -> bytes:
    text = str(seq).strip().upper()
    if text == "":
        text = "N"
    raw = bytearray((len(text) + 1) // 2)
    for i, ch in enumerate(text):
        code = _ALLELE_CHAR_TO_CODE.get(ch, 4)
        bi = i >> 1
        shift = 0 if (i & 1) == 0 else 4
        raw[bi] |= (code & 0x0F) << shift
    return bytes(raw)


def _write_bsite_header(
    handle,
    *,
    n_sites: int,
    n_chrom: int,
    chrom_dict_offset: int,
    flags: int = 0,
) -> None:
    handle.seek(0)
    handle.write(LEGACY_BSITE_MAGIC)
    handle.write(struct.pack("<H", int(LEGACY_BSITE_VERSION)))
    handle.write(struct.pack("<H", int(flags)))
    handle.write(struct.pack("<Q", int(n_sites)))
    handle.write(struct.pack("<I", int(n_chrom)))
    handle.write(struct.pack("<Q", int(chrom_dict_offset)))
    handle.write(struct.pack("<I", 0))


def write_bsite_records(
    handle,
    sites: Sequence[SiteInfo],
    *,
    chrom_codes: dict[str, int],
    chrom_names: list[str],
    cm_value: float = 0.0,
) -> None:
    for site in sites:
        chrom_raw = str(getattr(site, "chrom", ""))
        chrom_key, strand = _split_chrom_strand(chrom_raw)
        if chrom_key not in chrom_codes:
            chrom_codes[chrom_key] = len(chrom_names)
            chrom_names.append(chrom_key)
        chrom_code = chrom_codes[chrom_key]

        try:
            bp = int(getattr(site, "pos", 0))
        except Exception:
            bp = 0

        allele0 = str(getattr(site, "ref_allele", "N"))
        allele1 = str(getattr(site, "alt_allele", "N"))
        enc0 = _encode_allele_nibbles(allele0)
        enc1 = _encode_allele_nibbles(allele1)
        n0 = max(1, len(str(allele0).strip()))
        n1 = max(1, len(str(allele1).strip()))
        if n0 > 0xFFFF or n1 > 0xFFFF:
            raise ValueError("Allele length exceeds u16 limit for .bsite output.")

        handle.write(struct.pack("<IBfi", int(chrom_code), int(strand), float(cm_value), int(bp)))
        handle.write(struct.pack("<H", int(n0)))
        handle.write(enc0)
        handle.write(struct.pack("<H", int(n1)))
        handle.write(enc1)


def write_bsite_chrom_dict(handle, chrom_names: Sequence[str]) -> int:
    offset = int(handle.tell())
    for chrom in chrom_names:
        raw = str(chrom).encode("utf-8")
        if len(raw) > 0xFFFF:
            raise ValueError("Chromosome label too long for .bsite output.")
        handle.write(struct.pack("<H", int(len(raw))))
        handle.write(raw)
    return offset

def write_bin01_output(
    out_path: str,
    sample_ids: Sequence[str],
    chunks: Iterable[tuple[np.ndarray, Sequence[SiteInfo]]],
    *,
    total_sites: int,
    write_plink_sidecars: bool = False,
) -> None:
    """
    Write SNP-major bit-packed 0/1 matrix:
      - {prefix}.bin : header + packed rows
      - {prefix}.bin.id  : sample IDs (text)
      - {prefix}.bin.site: k-mer site metadata (2-bit encoded binary)
    Optional:
      - {prefix}.fam / {prefix}.bim when write_plink_sidecars=True
    """
    sample_ids = [str(x) for x in sample_ids]
    n_samples = len(sample_ids)
    if n_samples <= 0:
        raise ValueError("sample_ids is empty")

    prefix = output_prefix_from_path(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_id_file(f"{prefix}.bin.id", sample_ids)
    if write_plink_sidecars:
        write_fam_file(f"{prefix}.fam", sample_ids)

    written = 0
    writer_cls = _require_bin01_writer()
    progress, task_id, pbar = _open_site_progress("Writing BIN", int(total_sites))
    writer = writer_cls(str(out_path), n_samples, "kmer")
    fbim = open(f"{prefix}.bim", "w", encoding="utf-8") if write_plink_sidecars else None
    try:
        for block, sites in chunks:
            arr = np.asarray(block, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError("BIN chunk must be 2D (n_sites, n_samples)")
            if int(arr.shape[1]) != n_samples:
                raise ValueError(
                    f"BIN chunk sample mismatch: got {arr.shape[1]}, expected {n_samples}"
                )
            sites_list = list(sites)
            if int(arr.shape[0]) != len(sites_list):
                raise ValueError(
                    f"BIN chunk rows/sites mismatch: rows={arr.shape[0]}, sites={len(sites_list)}"
                )
            chrom, pos, ref, alt = _site_columns_for_bin_writer(sites_list)
            writer.write_chunk_f32(arr, chrom, pos, ref, alt)
            if fbim is not None:
                for site in sites_list:
                    chrom0 = str(site.chrom)
                    try:
                        pos0 = int(site.pos)
                    except Exception:
                        pos0 = 0
                    sid = f"{chrom0}_{pos0}"
                    ref0 = str(site.ref_allele)
                    alt0 = str(site.alt_allele)
                    fbim.write(f"{chrom0}\t{sid}\t0\t{pos0}\t{ref0}\t{alt0}\n")
            n_rows = int(arr.shape[0])
            written += n_rows
            _advance_site_progress(progress, task_id, pbar, n_rows)
    finally:
        writer.close()
        _close_site_progress(progress, pbar)
        if fbim is not None:
            fbim.close()

    if written != int(total_sites):
        print(
            f"! Warning: Written site count mismatch for BIN output: {written} vs {total_sites}. "
            f"Header was written with n_sites={written}.",
            file=sys.stderr,
        )


def write_gfd_output(
    out_path: str,
    sample_ids: Sequence[str],
    chunks: Iterable[tuple[np.ndarray, Sequence[SiteInfo]]],
    *,
    total_sites: int | None,
    source_is_dosage012: bool = False,
) -> None:
    """
    Write Garfield dedicated binary matrix output:
      - {prefix}.bin : header + packed 0/1 rows (JXBIN001)
      - {prefix}.id  : sample IDs (text)
      - {prefix}.bsite: binary site metadata
        (chrom-code, strand[0='-',1='+'], cM, bp, allele0, allele1)
    """
    sample_ids = [str(x) for x in sample_ids]
    n_samples = len(sample_ids)
    if n_samples <= 0:
        raise ValueError("sample_ids is empty")

    prefix = output_prefix_from_path(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_id_file(f"{prefix}.id", sample_ids)

    written = 0
    packed_cols = (n_samples + 7) // 8
    writer_cls = _require_bin01_writer()
    progress, task_id, pbar = _open_site_progress("Writing GFD", total_sites)
    writer = writer_cls(str(out_path), n_samples, "none")
    with open(f"{prefix}.bsite", "wb") as fsite:
        try:
            _write_bsite_header(
                fsite,
                n_sites=0,
                n_chrom=0,
                chrom_dict_offset=LEGACY_BSITE_HEADER_SIZE,
                flags=0,
            )
            chrom_codes: dict[str, int] = {}
            chrom_names: list[str] = []
            for block, sites in chunks:
                arr = np.asarray(block)
                if arr.ndim != 2:
                    raise ValueError("GFD chunk must be 2D (n_sites, n_samples)")
                if int(arr.shape[1]) != n_samples:
                    raise ValueError(
                        f"GFD chunk sample mismatch: got {arr.shape[1]}, expected {n_samples}"
                    )
                sites_list = list(sites)
                n_rows_in = int(arr.shape[0])
                if n_rows_in != len(sites_list):
                    raise ValueError(
                        f"GFD chunk rows/sites mismatch: rows={n_rows_in}, sites={len(sites_list)}"
                    )

                if source_is_dosage012:
                    if _gfd_packbits_from_dosage_block is not None:
                        packed = np.asarray(
                            _gfd_packbits_from_dosage_block(np.asarray(arr, dtype=np.float32))
                        )
                    else:
                        x = np.asarray(arr, dtype=np.float32)
                        g = np.full(x.shape, -9, dtype=np.int8)
                        valid = np.isfinite(x) & (x >= 0.0)
                        if np.any(valid):
                            vals = np.rint(x[valid]).astype(np.int16, copy=False)
                            vals = np.clip(vals, 0, 2).astype(np.int8, copy=False)
                            g[valid] = vals
                        c0 = np.sum(g == 0, axis=1, dtype=np.int64)
                        c1 = np.sum(g == 1, axis=1, dtype=np.int64)
                        c2 = np.sum(g == 2, axis=1, dtype=np.int64)
                        modes = np.argmax(np.stack([c0, c1, c2], axis=1), axis=1).astype(
                            np.int8, copy=False
                        )
                        miss = g < 0
                        if np.any(miss):
                            rows, cols = np.where(miss)
                            g[rows, cols] = modes[rows]
                        out = np.empty((n_rows_in * 2, n_samples), dtype=np.uint8)
                        out[0::2, :] = (g != 0).astype(np.uint8, copy=False)
                        out[1::2, :] = (g != 2).astype(np.uint8, copy=False)
                        packed = np.packbits(out, axis=1, bitorder="little")

                    if int(packed.shape[1]) != packed_cols:
                        raise ValueError(
                            f"GFD packbits column mismatch: got {packed.shape[1]}, expected {packed_cols}"
                        )
                    writer.write_chunk_packed(np.asarray(packed, dtype=np.uint8))

                    out_sites: list[SiteInfo] = []
                    for s in sites_list:
                        chrom0 = str(getattr(s, "chrom", ""))
                        try:
                            pos0 = int(getattr(s, "pos", 0))
                        except Exception:
                            pos0 = 0
                        out_sites.append(SiteInfo(f"{chrom0}_1", pos0, "0", "1"))
                        out_sites.append(SiteInfo(f"{chrom0}_2", pos0, "0", "1"))
                    write_bsite_records(
                        fsite,
                        out_sites,
                        chrom_codes=chrom_codes,
                        chrom_names=chrom_names,
                    )
                    n_rows_out = int(packed.shape[0])
                    written += n_rows_out
                    _advance_site_progress(progress, task_id, pbar, n_rows_out)
                else:
                    writer.write_chunk_f32(np.asarray(arr, dtype=np.float32))
                    write_bsite_records(
                        fsite,
                        sites_list,
                        chrom_codes=chrom_codes,
                        chrom_names=chrom_names,
                    )
                    n_rows_out = int(arr.shape[0])
                    written += n_rows_out
                    _advance_site_progress(progress, task_id, pbar, n_rows_out)

            chrom_dict_offset = write_bsite_chrom_dict(fsite, chrom_names)
            _write_bsite_header(
                fsite,
                n_sites=written,
                n_chrom=len(chrom_names),
                chrom_dict_offset=chrom_dict_offset,
                flags=0,
            )
            fsite.flush()
        finally:
            writer.close()
            _close_site_progress(progress, pbar)

    if total_sites is not None and written != int(total_sites):
        print(
            f"! Warning: Written site count mismatch for GFD output: {written} vs {total_sites}. "
            f"Header was written with n_sites={written}.",
            file=sys.stderr,
        )


def _open_site_progress(desc: str, total_sites: int | None):
    use_tty = stdout_is_tty()
    animate = bool(should_animate_status(desc))
    total_known = total_sites is not None
    if animate and rich_progress_available():
        progress = build_rich_progress(
            description_template="[progress.description]{task.description}",
            show_bar=bool(total_known),
            show_percentage=bool(total_known),
            show_remaining=False,
            bar_width=(40 if total_known else None),
            finished_text=" ",
            transient=True,
        )
        if progress is not None:
            task_id = progress.add_task(
                str(desc),
                total=(float(max(0, int(total_sites))) if total_known else None),
            )
            progress.start()
            return progress, task_id, None
    if (not animate) or tqdm is None:
        return None, None, None
    pbar = tqdm(
        total=(None if total_sites is None else max(0, int(total_sites))),
        unit="SNP",
        desc=str(desc),
        disable=(not use_tty) or (total_sites is None),
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
    total_sites: int | None,
) -> None:
    prefix = output_prefix_from_path(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_id_file(f"{prefix}.id", sample_ids)
    written = 0
    progress, task_id, pbar = _open_site_progress("Writing TXT", total_sites)
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
    if total_sites is not None and written != int(total_sites):
        print(
            f"! Warning: Written site count mismatch for text output: {written} vs {total_sites}. "
            f"Kept written rows={written}.",
            file=sys.stderr,
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
    expected = int(total_sites)
    if offset != expected:
        # Keep conversion usable when upstream site counting is conservative:
        # rewrite the npy file to the true written row count.
        if offset < expected:
            print(
                f"! Warning: Written site count mismatch for npy output: {offset} vs {total_sites}. "
                f"Adjusting output shape to written rows={offset}.",
                file=sys.stderr,
            )
            tmp_out = f"{out_path}.tmp"
            fixed_mm = np.lib.format.open_memmap(
                tmp_out,
                mode="w+",
                dtype=np.float32,
                shape=(int(offset), len(sample_ids)),
            )
            if offset > 0:
                step = 10_000
                for i in range(0, int(offset), step):
                    j = min(int(offset), i + step)
                    fixed_mm[i:j, :] = out_mm[i:j, :]
            fixed_mm.flush()
            del fixed_mm
            del out_mm
            os.replace(tmp_out, out_path)
            return
        raise RuntimeError(
            f"Written site count mismatch for npy output: {offset} vs {total_sites}"
        )
