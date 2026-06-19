# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import typing

import numpy as np

from janusx.gfreader import (
    inspect_genotype_file,
    load_bed_2bit_packed,
)
try:
    from janusx.gfreader import scan_bed_2bit_packed_stats
except Exception:
    scan_bed_2bit_packed_stats = None  # type: ignore[assignment]

try:
    from janusx.gfreader import prepare_bed_2bit_packed
except Exception:
    prepare_bed_2bit_packed = None  # type: ignore[assignment]

try:
    from janusx import janusx as _jxrs
except Exception:
    _jxrs = None

_BYTE_LUT = np.arange(256, dtype=np.uint16)
_NONMISS_LUT = np.zeros((256,), dtype=np.uint8)
_HET_LUT = np.zeros((256,), dtype=np.uint8)
for _within in range(4):
    _code = (_BYTE_LUT >> (_within * 2)) & 0b11
    _NONMISS_LUT += (_code != 0b01).astype(np.uint8)
    _HET_LUT += (_code == 0b10).astype(np.uint8)


def _normalize_plink_prefix(path_or_prefix: str) -> str:
    p = str(path_or_prefix).strip()
    low = p.lower()
    if low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam"):
        return p[:-4]
    return p


def _is_simple_snp_allele(allele: str) -> bool:
    a = str(allele).strip().upper()
    return len(a) == 1 and a in {"A", "C", "G", "T"}


def _plink_snp_mask(prefix: str) -> np.ndarray:
    mask: list[bool] = []
    bim_path = f"{prefix}.bim"
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


def _open_plink_bed_payload_memmap(
    prefix: str,
    *,
    n_samples: int,
    n_snps: int,
) -> np.memmap:
    plink_prefix = _normalize_plink_prefix(prefix)
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
    return np.memmap(
        bed_path,
        dtype=np.uint8,
        mode="r",
        offset=3,
        shape=(int(n_snps), int(bytes_per_snp)),
        order="C",
    )


def _packed_row_het_rate(packed: np.ndarray, n_samples: int) -> np.ndarray:
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


def _normalize_filter_mode(mode: str) -> str:
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


def packed_preload_failure_state(prefix: str | None, reason: object) -> dict[str, typing.Any]:
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
) -> typing.NoReturn:
    if len(attempt_errors) == 0:
        raise RuntimeError(f"Packed BED preparation failed for {plink_prefix}.")
    detail = "; ".join(
        f"{name}: {type(ex).__name__}: {ex}" for name, ex in attempt_errors
    )
    last_ex = attempt_errors[-1][1]
    raise RuntimeError(
        f"Packed BED preparation failed for {plink_prefix}. attempts={detail}"
    ) from last_ex


def prepare_packed_ctx_from_plink(
    prefix: str,
    *,
    maf: float,
    missing_rate: float,
    het_threshold: float = 0.0,
    snps_only: bool = False,
    expected_n_samples: int | None = None,
    filter_mode: str = "compact",
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    """
    Unified packed BED loading/filtering entry for Python callers.

    Prefer Rust-side preprocessing when available; fallback to legacy Python
    filtering path for compatibility.
    """
    filter_mode_key = _normalize_filter_mode(filter_mode)
    plink_prefix = _normalize_plink_prefix(prefix)
    sample_ids, _ = inspect_genotype_file(str(plink_prefix))
    sample_ids_arr = np.asarray(sample_ids, dtype=str)
    attempt_errors: list[tuple[str, Exception]] = []

    if expected_n_samples is not None and int(expected_n_samples) != int(sample_ids_arr.shape[0]):
        raise ValueError(
            f"Packed sample size mismatch: expected {int(expected_n_samples)}, "
            f"got {int(sample_ids_arr.shape[0])} from {plink_prefix}."
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
                _total_sites,
            ) = prepare_bed_2bit_packed(
                str(plink_prefix),
                maf_threshold=float(maf),
                max_missing_rate=float(missing_rate),
                het_threshold=float(het_threshold),
                snps_only=bool(snps_only),
            )
            if int(packed_n) != int(sample_ids_arr.shape[0]):
                raise ValueError(
                    f"Packed sample size mismatch: packed n={int(packed_n)}, "
                    f"expected {sample_ids_arr.shape[0]}"
                )
            packed_ctx: dict[str, typing.Any] = {
                "packed": np.ascontiguousarray(np.asarray(packed_raw, dtype=np.uint8)),
                "missing_rate": np.ascontiguousarray(
                    np.asarray(miss_raw, dtype=np.float32).reshape(-1), dtype=np.float32
                ),
                "af": np.ascontiguousarray(
                    np.asarray(maf_raw, dtype=np.float32).reshape(-1), dtype=np.float32
                ),
                "maf": np.ascontiguousarray(
                    np.asarray(maf_raw, dtype=np.float32).reshape(-1), dtype=np.float32
                ),
                "std_denom": np.ascontiguousarray(
                    np.asarray(std_raw, dtype=np.float32).reshape(-1), dtype=np.float32
                ),
                "dom_af": np.ascontiguousarray(
                    _packed_row_het_rate(
                        np.ascontiguousarray(np.asarray(packed_raw, dtype=np.uint8)),
                        int(packed_n),
                    ),
                    dtype=np.float32,
                ),
                "row_flip": np.ascontiguousarray(
                    np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1), dtype=np.bool_
                ),
                "site_keep": np.ascontiguousarray(
                    np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1), dtype=np.bool_
                ),
                "n_samples": int(packed_n),
                "n_total_sites": int(_total_sites),
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

    if scan_bed_2bit_packed_stats is not None:
        try:
            miss_raw, maf_raw, std_raw, row_flip_raw, het_raw, packed_n = scan_bed_2bit_packed_stats(
                str(plink_prefix)
            )
            if int(packed_n) != int(sample_ids_arr.shape[0]):
                raise ValueError(
                    f"Packed sample size mismatch: packed n={int(packed_n)}, expected {sample_ids_arr.shape[0]}"
                )

            miss_arr = np.ascontiguousarray(np.asarray(miss_raw, dtype=np.float32).reshape(-1))
            maf_arr = np.ascontiguousarray(np.asarray(maf_raw, dtype=np.float32).reshape(-1))
            std_arr = np.ascontiguousarray(np.asarray(std_raw, dtype=np.float32).reshape(-1))
            row_flip_input = np.ascontiguousarray(
                np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1),
                dtype=np.bool_,
            )
            af_arr = np.ascontiguousarray(
                np.where(row_flip_input, 1.0 - maf_arr, maf_arr).astype(np.float32, copy=False),
                dtype=np.float32,
            )
            row_flip_full = np.zeros_like(row_flip_input, dtype=np.bool_)
            het_arr = np.ascontiguousarray(np.asarray(het_raw, dtype=np.float32).reshape(-1))
            n_total_sites = int(maf_arr.shape[0])

            keep = np.ones((n_total_sites,), dtype=np.bool_)
            maf_thr = float(maf)
            if maf_thr > 0.0:
                keep &= (maf_arr >= maf_thr) & (maf_arr <= (1.0 - maf_thr))
            miss_thr = float(missing_rate)
            if miss_thr < 1.0:
                keep &= miss_arr <= miss_thr
            het_thr = float(het_threshold)
            if het_thr > 0.0:
                keep &= het_arr <= het_thr
            if bool(snps_only):
                snp_mask = _plink_snp_mask(str(plink_prefix))
                if snp_mask.shape[0] != keep.shape[0]:
                    raise ValueError(
                        f"BIM SNP mask length mismatch: got {snp_mask.shape[0]}, expected {keep.shape[0]}"
                    )
                keep &= snp_mask
            if not np.any(keep):
                raise ValueError(
                    "No SNPs left after packed BED filtering. Please relax --maf/--geno thresholds."
                )

            site_keep = np.ascontiguousarray(
                np.asarray(keep, dtype=np.bool_).reshape(-1),
                dtype=np.bool_,
            )
            keep_idx = np.ascontiguousarray(
                np.flatnonzero(site_keep).astype(np.int64, copy=False),
                dtype=np.int64,
            )
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

            if use_lazy:
                packed_memmap = _open_plink_bed_payload_memmap(
                    str(plink_prefix),
                    n_samples=int(packed_n),
                    n_snps=n_total_sites,
                )
                if filter_mode_key == "lazy_owned":
                    packed_payload: np.ndarray = np.array(
                        packed_memmap,
                        dtype=np.uint8,
                        order="C",
                        copy=True,
                    )
                    packed_storage = "owned"
                else:
                    packed_payload = packed_memmap
                    packed_storage = "memmap"
                packed_ctx = {
                    "packed": packed_payload,
                    "missing_rate": miss_arr,
                    "af": af_arr,
                    "maf": af_arr,
                    "std_denom": std_arr,
                    "dom_af": het_arr,
                    "row_flip": row_flip_full,
                    "site_keep": site_keep,
                    "active_row_idx": keep_idx,
                    "n_samples": int(packed_n),
                    "n_total_sites": int(site_keep.shape[0]),
                    "n_active_sites": int(keep_idx.shape[0]),
                    "packed_filter_mode": "lazy_full",
                    "packed_storage": packed_storage,
                    "source_prefix": str(plink_prefix),
                }
                return sample_ids_arr, packed_ctx

            packed_memmap = _open_plink_bed_payload_memmap(
                str(plink_prefix),
                n_samples=int(packed_n),
                n_snps=n_total_sites,
            )
            if not np.all(keep):
                packed = np.ascontiguousarray(
                    np.asarray(packed_memmap[keep], dtype=np.uint8),
                    dtype=np.uint8,
                )
                miss_keep = np.ascontiguousarray(miss_arr[keep], dtype=np.float32)
                maf_keep = np.ascontiguousarray(af_arr[keep], dtype=np.float32)
                std_keep = np.ascontiguousarray(std_arr[keep], dtype=np.float32)
                dom_keep = np.ascontiguousarray(het_arr[keep], dtype=np.float32)
                row_flip = np.ascontiguousarray(row_flip_full[keep], dtype=np.bool_)
            else:
                packed = np.ascontiguousarray(np.asarray(packed_memmap, dtype=np.uint8), dtype=np.uint8)
                miss_keep = miss_arr
                maf_keep = af_arr
                std_keep = std_arr
                dom_keep = het_arr
                row_flip = row_flip_full

            packed_ctx = {
                "packed": packed,
                "missing_rate": miss_keep,
                "af": maf_keep,
                "maf": maf_keep,
                "std_denom": std_keep,
                "dom_af": dom_keep,
                "row_flip": row_flip,
                "site_keep": site_keep,
                "active_row_idx": keep_idx,
                "n_samples": int(packed_n),
                "n_total_sites": int(site_keep.shape[0]),
                "n_active_sites": int(keep_idx.shape[0]),
                "packed_filter_mode": "compact",
                "packed_storage": "owned",
                "source_prefix": str(plink_prefix),
            }
            return sample_ids_arr, packed_ctx
        except Exception as ex:
            attempt_errors.append(("scan_bed_2bit_packed_stats", ex))

    try:
        packed_raw, miss_raw, maf_raw, std_raw, packed_n = load_bed_2bit_packed(str(plink_prefix))
    except Exception as ex:
        attempt_errors.append(("load_bed_2bit_packed", ex))
        _raise_packed_prepare_error(str(plink_prefix), attempt_errors)
    if int(packed_n) != int(sample_ids_arr.shape[0]):
        raise ValueError(
            f"Packed sample size mismatch: packed n={int(packed_n)}, expected {sample_ids_arr.shape[0]}"
        )

    packed = np.ascontiguousarray(np.asarray(packed_raw, dtype=np.uint8))
    miss_arr = np.ascontiguousarray(np.asarray(miss_raw, dtype=np.float32).reshape(-1))
    maf_arr = np.ascontiguousarray(np.asarray(maf_raw, dtype=np.float32).reshape(-1))
    std_arr = np.ascontiguousarray(np.asarray(std_raw, dtype=np.float32).reshape(-1))
    dom_arr = _packed_row_het_rate(packed, int(packed_n))

    keep = np.ones(maf_arr.shape[0], dtype=np.bool_)
    maf_thr = float(maf)
    if maf_thr > 0.0:
        keep &= (maf_arr >= maf_thr) & (maf_arr <= (1.0 - maf_thr))
    miss_thr = float(missing_rate)
    if miss_thr < 1.0:
        keep &= miss_arr <= miss_thr
    het_thr = float(het_threshold)
    if het_thr > 0.0:
        het_arr = _packed_row_het_rate(packed, int(packed_n))
        keep &= het_arr <= het_thr
    if bool(snps_only):
        snp_mask = _plink_snp_mask(str(plink_prefix))
        if snp_mask.shape[0] != keep.shape[0]:
            raise ValueError(
                f"BIM SNP mask length mismatch: got {snp_mask.shape[0]}, expected {keep.shape[0]}"
            )
        keep &= snp_mask
    if not np.any(keep):
        raise ValueError(
            "No SNPs left after packed BED filtering. Please relax --maf/--geno thresholds."
        )
    site_keep = np.ascontiguousarray(np.asarray(keep, dtype=np.bool_).reshape(-1), dtype=np.bool_)

    row_flip_full = np.zeros((int(packed.shape[0]),), dtype=np.bool_)

    keep_idx = np.ascontiguousarray(
        np.flatnonzero(site_keep).astype(np.int64, copy=False),
        dtype=np.int64,
    )
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

    if use_lazy:
        packed_ctx = {
            "packed": packed,
            "missing_rate": miss_arr,
            "af": maf_arr,
            "maf": maf_arr,
            "std_denom": std_arr,
            "dom_af": dom_arr,
            "row_flip": row_flip_full,
            "site_keep": site_keep,
            "active_row_idx": keep_idx,
            "n_samples": int(packed_n),
            "n_total_sites": int(site_keep.shape[0]),
            "n_active_sites": int(keep_idx.shape[0]),
            "packed_filter_mode": "lazy_full",
            "packed_storage": "owned",
            "source_prefix": str(plink_prefix),
        }
        return sample_ids_arr, packed_ctx

    if not np.all(keep):
        packed = np.ascontiguousarray(packed[keep], dtype=np.uint8)
        miss_arr = np.ascontiguousarray(miss_arr[keep], dtype=np.float32)
        maf_arr = np.ascontiguousarray(maf_arr[keep], dtype=np.float32)
        std_arr = np.ascontiguousarray(std_arr[keep], dtype=np.float32)
        dom_arr = np.ascontiguousarray(dom_arr[keep], dtype=np.float32)
        row_flip = np.ascontiguousarray(row_flip_full[keep], dtype=np.bool_)
    else:
        row_flip = row_flip_full

    packed_ctx = {
        "packed": packed,
        "missing_rate": miss_arr,
        "af": maf_arr,
        "maf": maf_arr,
        "std_denom": std_arr,
        "dom_af": dom_arr,
        "row_flip": row_flip,
        "site_keep": site_keep,
        "active_row_idx": keep_idx,
        "n_samples": int(packed_n),
        "n_total_sites": int(site_keep.shape[0]),
        "n_active_sites": int(keep_idx.shape[0]),
        "packed_filter_mode": "compact",
        "packed_storage": "owned",
        "source_prefix": str(plink_prefix),
    }
    return sample_ids_arr, packed_ctx


def prepare_packed_stats_ctx_from_plink(
    prefix: str,
    *,
    maf: float,
    missing_rate: float,
    het_threshold: float = 0.0,
    snps_only: bool = False,
    expected_n_samples: int | None = None,
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    """
    Prepare PLINK BED filter metadata without materializing or attaching packed rows.

    The returned context keeps the same filtering/allele-orientation semantics as
    GWAS packed-BED preprocessing, but omits the `packed` payload so callers can
    defer actual BED access to later streaming kernels.
    """
    plink_prefix = _normalize_plink_prefix(prefix)
    sample_ids, _ = inspect_genotype_file(str(plink_prefix))
    sample_ids_arr = np.asarray(sample_ids, dtype=str)
    if expected_n_samples is not None and int(expected_n_samples) != int(sample_ids_arr.shape[0]):
        raise ValueError(
            f"Packed sample size mismatch: expected {int(expected_n_samples)}, "
            f"got {int(sample_ids_arr.shape[0])} from {plink_prefix}."
        )

    if scan_bed_2bit_packed_stats is not None:
        miss_raw, maf_raw, std_raw, row_flip_raw, het_raw, packed_n = scan_bed_2bit_packed_stats(
            str(plink_prefix)
        )
        if int(packed_n) != int(sample_ids_arr.shape[0]):
            raise ValueError(
                f"Packed sample size mismatch: packed n={int(packed_n)}, expected {sample_ids_arr.shape[0]}"
            )

        miss_arr = np.ascontiguousarray(np.asarray(miss_raw, dtype=np.float32).reshape(-1))
        maf_arr = np.ascontiguousarray(np.asarray(maf_raw, dtype=np.float32).reshape(-1))
        std_arr = np.ascontiguousarray(np.asarray(std_raw, dtype=np.float32).reshape(-1))
        row_flip_input = np.ascontiguousarray(
            np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        af_arr = np.ascontiguousarray(
            np.where(row_flip_input, 1.0 - maf_arr, maf_arr).astype(np.float32, copy=False),
            dtype=np.float32,
        )
        row_flip_full = np.zeros_like(row_flip_input, dtype=np.bool_)
        het_arr = np.ascontiguousarray(np.asarray(het_raw, dtype=np.float32).reshape(-1))
        n_total_sites = int(maf_arr.shape[0])

        keep = np.ones((n_total_sites,), dtype=np.bool_)
        maf_thr = float(maf)
        if maf_thr > 0.0:
            keep &= (maf_arr >= maf_thr) & (maf_arr <= (1.0 - maf_thr))
        miss_thr = float(missing_rate)
        if miss_thr < 1.0:
            keep &= miss_arr <= miss_thr
        het_thr = float(het_threshold)
        if het_thr > 0.0:
            keep &= het_arr <= het_thr
        if bool(snps_only):
            snp_mask = _plink_snp_mask(str(plink_prefix))
            if snp_mask.shape[0] != keep.shape[0]:
                raise ValueError(
                    f"BIM SNP mask length mismatch: got {snp_mask.shape[0]}, expected {keep.shape[0]}"
                )
            keep &= snp_mask
        if not np.any(keep):
            raise ValueError(
                "No SNPs left after packed BED filtering. Please relax --maf/--geno thresholds."
            )

        site_keep = np.ascontiguousarray(
            np.asarray(keep, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        keep_idx = np.ascontiguousarray(
            np.flatnonzero(site_keep).astype(np.int64, copy=False),
            dtype=np.int64,
        )
        stats_ctx: dict[str, typing.Any] = {
            "missing_rate": miss_arr,
            "af": af_arr,
            "maf": af_arr,
            "std_denom": std_arr,
            "dom_af": het_arr,
            "row_flip": row_flip_full,
            "site_keep": site_keep,
            "active_row_idx": keep_idx,
            "n_samples": int(packed_n),
            "n_total_sites": int(site_keep.shape[0]),
            "n_active_sites": int(keep_idx.shape[0]),
            "packed_filter_mode": "stats_only",
            "packed_storage": "metadata",
            "source_prefix": str(plink_prefix),
        }
        return sample_ids_arr, stats_ctx

    sample_ids_arr, packed_ctx = prepare_packed_ctx_from_plink(
        str(plink_prefix),
        maf=float(maf),
        missing_rate=float(missing_rate),
        het_threshold=float(het_threshold),
        snps_only=bool(snps_only),
        expected_n_samples=expected_n_samples,
        filter_mode="lazy",
    )
    stats_ctx = {
        key: value
        for key, value in packed_ctx.items()
        if key != "packed"
    }
    stats_ctx["packed_filter_mode"] = "stats_only"
    stats_ctx["packed_storage"] = "metadata"
    return sample_ids_arr, stats_ctx
