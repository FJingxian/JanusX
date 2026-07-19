from __future__ import annotations

import os

import numpy as np

from janusx.gfreader import prepare_bed_2bit_packed
from janusx import janusx as jxrs


def stable_grm_builder_available() -> bool:
    return bool(
        hasattr(jxrs, "grm_packed_f64_with_stats")
        and callable(getattr(jxrs, "grm_packed_f64_with_stats"))
    )


def prefer_stable_packed_grm() -> bool:
    """
    Opt-in switch for the packed in-memory stable f64 GRM path.

    Default is False so CLI workflows stay on the original streaming/memmap
    backend and avoid eagerly materializing the filtered packed BED in memory.
    """
    raw = os.environ.get("JX_GRM_PREFER_STABLE_PACKED", "")
    key = str(raw).strip().lower()
    if key in {"1", "true", "yes", "on"}:
        return stable_grm_builder_available()
    return False


def build_stable_packed_grm_f64(
    *,
    prefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    method: int,
    block_cols: int,
    threads: int,
    snps_only: bool = False,
    progress_callback=None,
    progress_every: int = 0,
) -> tuple[np.ndarray, int]:
    if not stable_grm_builder_available():
        raise RuntimeError(
            "Stable packed GRM builder requires Rust symbol `grm_packed_f64_with_stats`."
        )

    (
        packed_arr,
        _miss_arr,
        maf_arr,
        _std_arr,
        row_flip_arr,
        _site_keep_arr,
        n_samples,
        _n_total_sites,
    ) = prepare_bed_2bit_packed(
        str(prefix),
        maf_threshold=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
        het_threshold=1.0,
        snps_only=bool(snps_only),
    )

    eff_m = int(np.asarray(packed_arr).shape[0])
    if eff_m <= 0:
        raise RuntimeError("No SNPs remained after filtering; stable GRM is empty.")

    grm_raw, _row_sum, _varsum = jxrs.grm_packed_f64_with_stats(
        packed_arr,
        int(n_samples),
        row_flip_arr,
        maf_arr,
        sample_indices=None,
        method=int(method),
        block_cols=max(1, int(block_cols)),
        threads=max(1, int(threads)),
        progress_callback=progress_callback,
        progress_every=max(0, int(progress_every)),
    )
    grm = np.ascontiguousarray(np.asarray(grm_raw, dtype=np.float64), dtype=np.float64)
    if grm.ndim != 2 or grm.shape[0] != grm.shape[1]:
        raise RuntimeError(f"Stable GRM must be square; got shape={grm.shape}")
    return grm, eff_m


def build_stable_packed_ctx_grm_f64(
    *,
    packed_ctx: dict[str, object],
    sample_indices: np.ndarray | None,
    block_cols: int,
    threads: int,
    progress_callback=None,
    progress_every: int = 0,
) -> tuple[np.ndarray, int]:
    if not stable_grm_builder_available():
        raise RuntimeError(
            "Stable packed-context GRM builder requires Rust symbol "
            "`grm_packed_f64_with_stats`."
        )

    packed_filter_mode = str(packed_ctx.get("packed_filter_mode", "compact")).strip().lower()
    packed_raw = np.asarray(packed_ctx["packed"], dtype=np.uint8)
    maf_raw = np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1)
    row_flip_raw = np.asarray(packed_ctx["row_flip"], dtype=np.bool_).reshape(-1)
    n_samples = int(packed_ctx["n_samples"])

    if packed_raw.ndim != 2:
        raise RuntimeError(f"Packed GRM input must be 2D; got shape={packed_raw.shape}")

    if packed_filter_mode == "lazy_full":
        active_row_idx_raw = packed_ctx.get("active_row_idx", None)
        if active_row_idx_raw is not None:
            active_row_idx = np.ascontiguousarray(
                np.asarray(active_row_idx_raw, dtype=np.int64).reshape(-1),
                dtype=np.int64,
            )
        else:
            site_keep_raw = packed_ctx.get("site_keep", None)
            if site_keep_raw is None:
                raise RuntimeError(
                    "Lazy packed GRM input requires `active_row_idx` or `site_keep`."
                )
            site_keep = np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1)
            active_row_idx = np.ascontiguousarray(
                np.flatnonzero(site_keep).astype(np.int64, copy=False),
                dtype=np.int64,
            )
        packed = np.ascontiguousarray(
            np.asarray(packed_raw[active_row_idx, :], dtype=np.uint8),
            dtype=np.uint8,
        )
        maf = np.ascontiguousarray(maf_raw[active_row_idx], dtype=np.float32)
        row_flip = np.ascontiguousarray(row_flip_raw[active_row_idx], dtype=np.bool_)
    else:
        packed = np.ascontiguousarray(np.asarray(packed_raw, dtype=np.uint8), dtype=np.uint8)
        maf = np.ascontiguousarray(maf_raw, dtype=np.float32)
        row_flip = np.ascontiguousarray(row_flip_raw, dtype=np.bool_)

    eff_m = int(packed.shape[0])
    if eff_m <= 0:
        raise RuntimeError("No SNPs remained in packed-context GRM input.")
    if maf.shape[0] != eff_m or row_flip.shape[0] != eff_m:
        raise RuntimeError(
            "Packed-context GRM payload mismatch: "
            f"packed_rows={eff_m}, maf={maf.shape[0]}, row_flip={row_flip.shape[0]}"
        )

    sidx_arg = None
    if sample_indices is not None:
        sidx = np.ascontiguousarray(
            np.asarray(sample_indices, dtype=np.int64).reshape(-1),
            dtype=np.int64,
        )
        full_identity = bool(
            int(sidx.shape[0]) == int(n_samples)
            and np.array_equal(sidx, np.arange(n_samples, dtype=np.int64))
        )
        if not full_identity:
            sidx_arg = sidx

    grm_raw, _row_sum, _varsum = jxrs.grm_packed_f64_with_stats(
        packed,
        int(n_samples),
        row_flip,
        maf,
        sample_indices=sidx_arg,
        method=1,
        block_cols=max(1, int(block_cols)),
        threads=max(1, int(threads)),
        progress_callback=progress_callback,
        progress_every=max(0, int(progress_every)),
    )
    grm = np.ascontiguousarray(np.asarray(grm_raw, dtype=np.float64), dtype=np.float64)
    if grm.ndim != 2 or grm.shape[0] != grm.shape[1]:
        raise RuntimeError(f"Stable packed-context GRM must be square; got shape={grm.shape}")
    return grm, eff_m


def build_dense_grm_f64(
    marker_matrix: np.ndarray,
    *,
    block_rows: int = 4096,
) -> np.ndarray:
    mat = np.asarray(marker_matrix)
    if mat.ndim != 2:
        raise RuntimeError(f"Dense GRM input must be 2D; got shape={mat.shape}")

    m, n = int(mat.shape[0]), int(mat.shape[1])
    if m <= 0 or n <= 0:
        raise RuntimeError(f"Dense GRM input must be non-empty; got shape={mat.shape}")

    mean = np.mean(mat, axis=1, dtype=np.float64, keepdims=True)
    var = np.var(mat, axis=1, dtype=np.float64, keepdims=True)
    var_sum = float(np.sum(var, dtype=np.float64))
    if (not np.isfinite(var_sum)) or (var_sum <= 0.0):
        raise RuntimeError("Invalid dense GRM denominator (sum(var)<=0).")

    grm = np.zeros((n, n), dtype=np.float64)
    step = max(1, int(block_rows))
    for st in range(0, m, step):
        ed = min(m, st + step)
        blk = np.asarray(mat[st:ed, :], dtype=np.float64)
        blk -= mean[st:ed, :]
        grm += blk.T @ blk
    grm /= var_sum
    grm = 0.5 * (grm + grm.T)
    return np.ascontiguousarray(grm, dtype=np.float64)


def save_grm_npy_blocked(
    path: str,
    grm: np.ndarray,
    *,
    dtype: np.dtype | type = np.float32,
    block_rows: int = 4096,
) -> None:
    arr = np.asarray(grm)
    if arr.ndim != 2:
        raise RuntimeError(f"GRM must be 2D for NPY save; got shape={arr.shape}")
    out_dtype = np.dtype(dtype)
    rows, cols = int(arr.shape[0]), int(arr.shape[1])
    mm = np.lib.format.open_memmap(
        str(path),
        mode="w+",
        dtype=out_dtype,
        shape=(rows, cols),
    )
    step = max(1, int(block_rows))
    try:
        for i in range(0, rows, step):
            j = min(rows, i + step)
            mm[i:j, :] = arr[i:j, :]
        mm.flush()
    finally:
        del mm
