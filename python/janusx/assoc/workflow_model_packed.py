# -*- coding: utf-8 -*-
"""Packed/full-rust GWAS model runners (extracted from workflow.py)."""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Optional, Union

import numpy as np
import pandas as pd
import psutil

from .workflow import (
    CliStatus,
    LMM,
    _ProgressAdapter,
    _align_pheno_to_sample_order,
    _as_plink_prefix,
    _display_path,
    _emit_trait_header,
    _emit_warning_line,
    _gwas_eigh_from_grm,
    _gwas_evd_stage_ctx,
    _gwas_scan_stage_ctx,
    _log_file_only,
    _log_model_line,
    _read_bim_sites,
    _resolve_stream_scan_chunk_size,
    _rich_success,
    _run_fastplot_from_tsv_with_status,
    _run_result_write_with_status,
    _trait_values_and_mask,
    auto_mmap_window_mb,
    detect_effective_threads,
    format_elapsed,
    jxrs,
    prepare_packed_ctx_from_plink,
)

_WARNED_LM_STREAM_MMAP_LEGACY = False


def _gwas_use_rust_unified_v1() -> bool:
    """
    Toggle packed GWAS unified Rust dispatcher (v1).

    Default enabled when Rust symbol exists; set
    JX_GWAS_USE_RUST_UNIFIED_V1=0 to disable quickly.
    """
    raw = str(os.environ.get("JX_GWAS_USE_RUST_UNIFIED_V1", "")).strip().lower()
    enabled = raw not in {"0", "false", "no", "off"}
    return bool(enabled and hasattr(jxrs, "gwas_packed_unified_to_tsv"))


def _lm_precompute_ixx_qr(x_design: np.ndarray) -> np.ndarray:
    """
    Stable LM precompute for IXX without explicitly forming/inverting X'X.

    Let X = QR (reduced QR). Then
      (X'X)^+ = (R'R)^+ = R^+ (R^+)'
    which is numerically more stable than pinv(X'X) under collinearity.
    """
    x = np.ascontiguousarray(np.asarray(x_design, dtype=np.float64))
    if x.ndim != 2:
        raise ValueError(f"x_design must be 2D, got ndim={x.ndim}")
    n, q0 = int(x.shape[0]), int(x.shape[1])
    if n <= q0:
        raise ValueError(f"n too small for LM design: require n > q0, got n={n}, q0={q0}")

    _q, r = np.linalg.qr(x, mode="reduced")
    r = np.asarray(r, dtype=np.float64)
    eye = np.eye(q0, dtype=np.float64)
    try:
        r_inv = np.linalg.solve(r, eye)
    except np.linalg.LinAlgError:
        # Rank-deficient fallback: still avoid forming X'X explicitly.
        r_inv = np.linalg.pinv(r)
    ixx = r_inv @ r_inv.T
    return np.ascontiguousarray(ixx, dtype=np.float64)


def _packed_ctx_active_view_for_gwas(
    packed_ctx: dict[str, object],
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
    packed_n = int(packed_ctx["n_samples"])
    if packed.ndim != 2:
        raise ValueError("Packed GWAS context requires packed ndim=2.")
    maf_full = np.ascontiguousarray(
        np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1),
        dtype=np.float32,
    )
    if int(maf_full.shape[0]) != int(packed.shape[0]):
        raise ValueError("Packed GWAS context mismatch: maf length != packed rows.")
    row_flip_raw = packed_ctx.get("row_flip", None)
    if row_flip_raw is None:
        if not hasattr(jxrs, "bed_packed_row_flip_mask"):
            raise RuntimeError(
                "Rust packed row-flip kernel is unavailable. Rebuild/install JanusX extension."
            )
        row_flip_full = np.ascontiguousarray(
            np.asarray(jxrs.bed_packed_row_flip_mask(packed, int(packed_n)), dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        packed_ctx["row_flip"] = row_flip_full
    else:
        row_flip_full = np.ascontiguousarray(
            np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
    if int(row_flip_full.shape[0]) != int(packed.shape[0]):
        raise ValueError("Packed GWAS context mismatch: row_flip length != packed rows.")

    site_keep_raw = packed_ctx.get("site_keep", None)
    if site_keep_raw is None:
        active_row_idx = np.ascontiguousarray(np.arange(int(packed.shape[0]), dtype=np.int64), dtype=np.int64)
    else:
        site_keep = np.ascontiguousarray(
            np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        active_row_idx = np.ascontiguousarray(
            np.asarray(
                packed_ctx.get(
                    "active_row_idx",
                    np.flatnonzero(site_keep).astype(np.int64, copy=False),
                ),
                dtype=np.int64,
            ).reshape(-1),
            dtype=np.int64,
        )
        if int(site_keep.shape[0]) < int(active_row_idx.shape[0]):
            raise ValueError("Packed GWAS context mismatch: active_row_idx exceeds site_keep length.")
    maf_active = np.ascontiguousarray(maf_full[active_row_idx], dtype=np.float32)
    row_flip_active = np.ascontiguousarray(row_flip_full[active_row_idx], dtype=np.bool_)
    return packed, packed_n, active_row_idx, maf_active, row_flip_active


def _prepare_packed_bed_once_for_gwas(
    *,
    genofile: str,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    snps_only: bool,
    use_spinner: bool,
    preloaded_packed: Union[dict[str, object], None] = None,
) -> tuple[str, np.ndarray, dict[str, object], list[tuple[str, int, str, str]]]:
    """
    Resolve packed BED context for GWAS full-rust routes.

    Returns
    -------
    prefix, full_ids, packed_ctx, sites_all
    """
    prefix = _as_plink_prefix(genofile)
    if prefix is None:
        raise ValueError(
            "Full-rust packed path requires PLINK BED input "
            "(prefix with .bed/.bim/.fam)."
        )

    pre = preloaded_packed if isinstance(preloaded_packed, dict) else None
    if pre is not None and str(pre.get("prefix", "")) == str(prefix):
        full_ids = np.asarray(pre.get("full_ids", []), dtype=str)
        packed_ctx_obj = pre.get("packed_ctx")
        if not isinstance(packed_ctx_obj, dict):
            raise ValueError("Invalid preloaded packed context: missing packed_ctx dict.")
        packed_ctx = {
            "packed": np.ascontiguousarray(np.asarray(packed_ctx_obj["packed"], dtype=np.uint8)),
            "missing_rate": np.ascontiguousarray(
                np.asarray(packed_ctx_obj["missing_rate"], dtype=np.float32).reshape(-1),
                dtype=np.float32,
            ),
            "maf": np.ascontiguousarray(
                np.asarray(packed_ctx_obj["maf"], dtype=np.float32).reshape(-1),
                dtype=np.float32,
            ),
            "row_flip": np.ascontiguousarray(
                np.asarray(packed_ctx_obj["row_flip"], dtype=np.bool_).reshape(-1),
                dtype=np.bool_,
            ),
            "site_keep": np.ascontiguousarray(
                np.asarray(packed_ctx_obj.get("site_keep"), dtype=np.bool_).reshape(-1),
                dtype=np.bool_,
            ) if packed_ctx_obj.get("site_keep", None) is not None else None,
            "active_row_idx": np.ascontiguousarray(
                np.asarray(
                    packed_ctx_obj.get(
                        "active_row_idx",
                        np.arange(int(np.asarray(packed_ctx_obj["packed"]).shape[0]), dtype=np.int64),
                    ),
                    dtype=np.int64,
                ).reshape(-1),
                dtype=np.int64,
            ),
            "packed_filter_mode": str(packed_ctx_obj.get("packed_filter_mode", "compact")),
            "n_samples": int(packed_ctx_obj["n_samples"]),
            "source_prefix": str(prefix),
        }
        sites_pre = pre.get("sites_all")
        if isinstance(sites_pre, list) and len(sites_pre) > 0:
            sites_all = [
                (str(c), int(p), str(a0), str(a1))
                for (c, p, a0, a1) in sites_pre
            ]
            return str(prefix), full_ids, packed_ctx, sites_all
    else:
        with CliStatus("Loading packed BED genotype...", enabled=bool(use_spinner)) as task:
            try:
                full_ids, packed_ctx = prepare_packed_ctx_from_plink(
                    str(prefix),
                    maf=float(maf_threshold),
                    missing_rate=float(max_missing_rate),
                    het_threshold=float(het_threshold),
                    snps_only=bool(snps_only),
                    filter_mode="lazy",
                )
            except Exception:
                task.fail("Loading packed BED genotype ...Failed")
                raise
            task.complete("Loading packed BED genotype ...Finished")
        full_ids = np.asarray(full_ids, dtype=str)
        packed_ctx = {
            "packed": np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8)),
            "missing_rate": np.ascontiguousarray(
                np.asarray(packed_ctx["missing_rate"], dtype=np.float32).reshape(-1),
                dtype=np.float32,
            ),
            "maf": np.ascontiguousarray(
                np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1),
                dtype=np.float32,
            ),
            "row_flip": np.ascontiguousarray(
                np.asarray(packed_ctx["row_flip"], dtype=np.bool_).reshape(-1),
                dtype=np.bool_,
            ),
            "site_keep": np.ascontiguousarray(
                np.asarray(packed_ctx.get("site_keep"), dtype=np.bool_).reshape(-1),
                dtype=np.bool_,
            ) if packed_ctx.get("site_keep", None) is not None else None,
            "active_row_idx": np.ascontiguousarray(
                np.asarray(
                    packed_ctx.get("active_row_idx", np.arange(int(np.asarray(packed_ctx["packed"]).shape[0]), dtype=np.int64)),
                    dtype=np.int64,
                ).reshape(-1),
                dtype=np.int64,
            ),
            "packed_filter_mode": str(packed_ctx.get("packed_filter_mode", "compact")),
            "n_samples": int(packed_ctx["n_samples"]),
            "source_prefix": str(prefix),
        }

    sites_all_full = _read_bim_sites(str(prefix))
    site_keep_obj = packed_ctx.get("site_keep", None)
    if site_keep_obj is None:
        site_keep = np.ones((len(sites_all_full),), dtype=np.bool_)
    else:
        site_keep = np.asarray(site_keep_obj, dtype=np.bool_).reshape(-1)
    if int(site_keep.shape[0]) != int(len(sites_all_full)):
        raise ValueError("Packed site_keep length mismatch with BIM rows.")
    sites_all = [s for s, keep in zip(sites_all_full, site_keep) if bool(keep)]
    active_row_idx = np.ascontiguousarray(
        np.asarray(
            packed_ctx.get("active_row_idx", np.flatnonzero(site_keep).astype(np.int64, copy=False)),
            dtype=np.int64,
        ).reshape(-1),
        dtype=np.int64,
    )
    if int(len(sites_all)) != int(active_row_idx.shape[0]):
        raise ValueError(
            f"Packed/BIM mismatch after filtering: sites={len(sites_all)} "
            f"active_rows={active_row_idx.shape[0]}"
        )
    return str(prefix), full_ids, packed_ctx, sites_all


def prepare_memmap_filtered_bed_for_gwas(
    *,
    genofile: str,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    block_rows: int,
    threads: int,
    logger: logging.Logger,
    use_spinner: bool = False,
) -> tuple[str, dict[str, int]]:
    """
    Build a temporary filtered PLINK BED prefix via Rust mmap engine.

    This performs one Rust-side BED mmap pass with QC filtering, then emits a
    temporary BED/BIM/FAM prefix for downstream GWAS routes.
    """
    prefix = _as_plink_prefix(genofile)
    if prefix is None:
        raise ValueError("memmap BED route requires PLINK BED input/prefix.")
    if not hasattr(jxrs, "bed_mmap_filter_to_plink_rust"):
        raise RuntimeError(
            "Rust extension missing bed_mmap_filter_to_plink_rust. "
            "Rebuild/install JanusX extension first."
        )

    out_mm_prefix = f"{outprefix}.mmapbed.{os.getpid()}.{uuid.uuid4().hex}"
    block_rows_use = int(max(1024, int(block_rows)))
    scan_threads = int(max(1, int(threads)))
    parallel = bool(scan_threads > 1)

    with CliStatus(
        "Preparing memmap BED cache...",
        enabled=bool(use_spinner),
        use_process=True,
    ) as task:
        try:
            with _gwas_scan_stage_ctx(scan_threads):
                kept, scanned, n_samples, n_blocks = jxrs.bed_mmap_filter_to_plink_rust(
                    str(prefix),
                    str(out_mm_prefix),
                    maf_threshold=float(maf_threshold),
                    max_missing_rate=float(max_missing_rate),
                    het_threshold=float(het_threshold),
                    block_rows=int(block_rows_use),
                    parallel=bool(parallel),
                )
        except Exception:
            task.fail("Preparing memmap BED cache ...Failed")
            raise
        task.complete("Preparing memmap BED cache ...Finished")

    kept_i = int(kept)
    if kept_i <= 0:
        raise ValueError(
            "No SNPs remained after memmap BED filtering. "
            "Please relax --maf/--geno/--het thresholds."
        )
    info = {
        "kept_snps": kept_i,
        "scanned_snps": int(scanned),
        "n_samples": int(n_samples),
        "n_blocks": int(n_blocks),
    }
    _log_file_only(
        logger,
        logging.INFO,
        (
            "Memmap BED cache ready: "
            f"kept={info['kept_snps']}, scanned={info['scanned_snps']}, "
            f"samples={info['n_samples']}, blocks={info['n_blocks']}, "
            f"prefix={_display_path(str(out_mm_prefix))}"
        ),
    )
    return str(out_mm_prefix), info


def run_fastlmm_packed_fullrank(
    *,
    genofile: str,
    pheno: pd.DataFrame,
    ids: np.ndarray,
    grm: Union[np.ndarray, None] = None,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    genetic_model: str,
    het_threshold: float,
    chunk_size: int,
    qmatrix: np.ndarray,
    cov_all: Union[np.ndarray, None],
    plot: bool,
    threads: int,
    logger: logging.Logger,
    use_spinner: bool = False,
    snps_only: bool = True,
    eff_snp_by_trait: Union[dict[str, int], None] = None,
    summary_rows: Union[list[dict[str, object]], None] = None,
    saved_paths: Union[list[str], None] = None,
    trait_names: Union[list[str], None] = None,
    emit_trait_header: bool = True,
    preloaded_packed: Union[dict[str, object], None] = None,
) -> None:
    required_symbols = [
        "grm_packed_f64_with_stats",
        "rust_eigh_from_array_f64",
        "fastlmm_assoc_packed_f32_to_tsv",
    ]
    missing_symbols = [s for s in required_symbols if not hasattr(jxrs, s)]
    if len(missing_symbols) > 0:
        raise RuntimeError(
            "Rust extension missing required GWAS full-rust symbols: "
            + ", ".join(missing_symbols)
        )
    prefix, full_ids, packed_ctx, sites_all = _prepare_packed_bed_once_for_gwas(
        genofile=genofile,
        maf_threshold=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
        het_threshold=float(het_threshold),
        snps_only=bool(snps_only),
        use_spinner=bool(use_spinner),
        preloaded_packed=preloaded_packed,
    )
    id_to_idx = {sid: i for i, sid in enumerate(full_ids)}
    try:
        sample_map = np.asarray([id_to_idx[str(sid)] for sid in np.asarray(ids, dtype=str)], dtype=np.int64)
    except KeyError as e:
        raise ValueError("Some aligned sample IDs are not present in packed BED sample order.") from e

    packed, packed_n, packed_row_idx, maf, row_flip = _packed_ctx_active_view_for_gwas(
        packed_ctx
    )
    if packed_n <= 0:
        raise ValueError("Packed BED reported invalid sample count.")
    if int(packed_row_idx.shape[0]) != int(maf.shape[0]) or int(packed_row_idx.shape[0]) != int(row_flip.shape[0]):
        raise ValueError("Packed BED arrays have inconsistent SNP dimensions.")

    if int(len(sites_all)) != int(packed_row_idx.shape[0]):
        raise ValueError(
            f"Packed/BIM mismatch after filtering: sites={len(sites_all)} active_rows={packed_row_idx.shape[0]}"
        )
    chrom_all = [str(c) for (c, _p, _a0, _a1) in sites_all]
    pos_all = [int(p) for (_c, p, _a0, _a1) in sites_all]
    allele0_all = [str(v) for (_c, _p, v, _a1) in sites_all]
    allele1_all = [str(v) for (_c, _p, _a0, v) in sites_all]

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pheno_aligned, ids = _align_pheno_to_sample_order(pheno, ids)
    trait_iter = list(pheno_aligned.columns) if trait_names is None else [t for t in trait_names if t in pheno_aligned.columns]
    multi_trait_mode = len(trait_iter) > 1

    for trait_idx, pname in enumerate(trait_iter):
        cpu_t0 = process.cpu_times()
        t0 = time.time()
        peak_rss = process.memory_info().rss

        y_full, sameidx = _trait_values_and_mask(pheno_aligned, str(pname))
        keep_idx = np.flatnonzero(sameidx).astype(np.int64, copy=False)
        n_idv = int(keep_idx.shape[0])
        if n_idv == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")
            continue

        if bool(emit_trait_header):
            _emit_trait_header(
                logger,
                str(pname),
                int(n_idv),
                pve=None,
                use_spinner=bool(use_spinner),
                width=60,
            )

        y_vec = np.ascontiguousarray(y_full[keep_idx], dtype=np.float64)
        x_cov = qmatrix[keep_idx]
        if cov_all is not None:
            x_cov = np.concatenate([x_cov, cov_all[keep_idx]], axis=1)
        x_cov = np.ascontiguousarray(x_cov, dtype=np.float64)
        x_arg = x_cov if int(x_cov.shape[1]) > 0 else None
        sample_idx_trait = np.ascontiguousarray(sample_map[keep_idx], dtype=np.int64)

        grm_secs = 0.0
        grm_fit: np.ndarray
        if grm is not None:
            grm_fit = np.ascontiguousarray(
                np.asarray(grm[np.ix_(keep_idx, keep_idx)], dtype=np.float64),
                dtype=np.float64,
            )
            if grm_fit.ndim != 2 or grm_fit.shape[0] != grm_fit.shape[1] or grm_fit.shape[0] != n_idv:
                raise ValueError(
                    f"Trait GRM shape mismatch: got {grm_fit.shape}, expected ({n_idv}, {n_idv})."
                )
        else:
            grm_t0 = time.monotonic()
            with CliStatus(
                "Building packed GRM (Rust)...",
                enabled=bool(use_spinner),
                use_process=True,
            ) as task:
                try:
                    grm_raw, _row_sum, _varsum = jxrs.grm_packed_f64_with_stats(
                        packed,
                        int(packed_n),
                        row_flip,
                        maf,
                        sample_idx_trait,
                        1,
                        max(1, int(chunk_size)),
                        int(threads),
                        None,
                        0,
                    )
                except Exception:
                    task.fail("Packed GRM (Rust) ...Failed")
                    raise
                task.complete("Packed GRM (Rust) ...Finished")
            grm_secs = max(time.monotonic() - grm_t0, 0.0)
            grm_fit = np.ascontiguousarray(np.asarray(grm_raw, dtype=np.float64), dtype=np.float64)
            if grm_fit.ndim != 2 or grm_fit.shape[0] != grm_fit.shape[1] or grm_fit.shape[0] != n_idv:
                raise ValueError(
                    f"Trait GRM shape mismatch: got {grm_fit.shape}, expected ({n_idv}, {n_idv})."
                )
        np.fill_diagonal(grm_fit, np.diag(grm_fit) + 1e-6)

        evd_t0 = time.monotonic()
        with CliStatus(
            "Running FastLMM Eigen-Decomposition...",
            enabled=bool(use_spinner),
            use_process=True,
        ) as task:
            try:
                eigvals, eigvecs, evd_backend, evd_secs = _gwas_eigh_from_grm(
                    grm_fit,
                    threads=max(1, int(threads)),
                    logger=logger,
                    stage_label=f"FastLMM-fullrank:{pname}",
                    require_rust=True,
                )
            except Exception:
                task.fail("FastLMM Eigen-Decomposition ...Failed")
                raise
        # Keep explicit elapsed in case backend timing is unavailable/zero.
        evd_secs = max(float(evd_secs), max(time.monotonic() - evd_t0, 0.0))
        # LAPACK eigh returns ascending eigenvalues. The packed Rust scan kernel
        # consumes leading k components, so reorder to descending first and
        # implicitly drop the smallest component when k = n-1.
        try:
            ord_desc = np.argsort(np.asarray(eigvals, dtype=np.float64))[::-1]
            eigvals = np.asarray(eigvals, dtype=np.float64)[ord_desc]
            eigvecs = np.asarray(eigvecs, dtype=np.float64)[:, ord_desc]
        except Exception:
            eigvals = np.asarray(eigvals, dtype=np.float64)
            eigvecs = np.asarray(eigvecs, dtype=np.float64)
        s_trait = np.ascontiguousarray(np.maximum(eigvals, 0.0), dtype=np.float32)
        u_sub = np.ascontiguousarray(eigvecs, dtype=np.float32)
        if u_sub.shape != (n_idv, n_idv):
            raise ValueError(
                f"Trait eigenvector shape mismatch: got {u_sub.shape}, expected ({n_idv}, {n_idv})."
            )
        # FastLMM fixed-lambda mode on packed path:
        # estimate lambda once from null model, then scan with fixed_lbd.
        fixed_lbd: Optional[float] = None
        fixed_ml0: Optional[float] = None
        if (
            str(genetic_model).lower() == "add"
            and hasattr(jxrs, "fastlmm_reml_null_f32")
        ):
            try:
                k_trait = max(1, int(n_idv) - 1)
                s_null = np.ascontiguousarray(
                    np.maximum(np.asarray(eigvals[:k_trait], dtype=np.float64), 0.0),
                    dtype=np.float64,
                )
                u_null = np.ascontiguousarray(
                    np.asarray(eigvecs[:, :k_trait], dtype=np.float64),
                    dtype=np.float64,
                )
                x_full = np.ascontiguousarray(
                    np.concatenate(
                        [np.ones((int(x_cov.shape[0]), 1), dtype=np.float64), x_cov],
                        axis=1,
                    ),
                    dtype=np.float64,
                )
                with _gwas_evd_stage_ctx(max(1, int(threads))):
                    u1tx = np.ascontiguousarray(u_null.T @ x_full, dtype=np.float64)
                    u1ty = np.ascontiguousarray((u_null.T @ y_vec).reshape(-1), dtype=np.float64)
                    u2tx = np.ascontiguousarray(x_full - (u_null @ u1tx), dtype=np.float64)
                    u2ty = np.ascontiguousarray(y_vec - (u_null @ u1ty), dtype=np.float64)
                lbd0, ml0_0, _reml0_0 = jxrs.fastlmm_reml_null_f32(
                    s_null,
                    u1tx,
                    u2tx,
                    u1ty,
                    u2ty,
                    -5.0,
                    5.0,
                    50,
                    1e-2,
                    "add",
                )
                if np.isfinite(float(lbd0)) and float(lbd0) > 0.0 and np.isfinite(float(ml0_0)):
                    fixed_lbd = float(lbd0)
                    fixed_ml0 = float(ml0_0)
            except Exception as ex:
                _emit_warning_line(
                    logger,
                    f"FastLMM fixed-lambda estimate unavailable; fallback to in-kernel search. reason={ex}",
                    use_spinner=bool(use_spinner),
                )
                fixed_lbd = None
                fixed_ml0 = None

        gwas_total = int(len(sites_all))
        gwas_last_done = 0
        gwas_pbar: Optional[_ProgressAdapter] = None
        if bool(use_spinner):
            gwas_pbar = _ProgressAdapter(
                total=max(1, gwas_total),
                desc="FastLMM",
                force_animate=True,
            )

        def _fastlmm_progress(done: int, total: int) -> None:
            nonlocal gwas_last_done, gwas_total, gwas_pbar
            if gwas_pbar is None:
                return
            try:
                d = int(done)
                t = int(total)
            except Exception:
                return
            if t > 0 and t != gwas_total and gwas_last_done == 0:
                try:
                    gwas_pbar.close(show_done=False)
                except Exception:
                    pass
                gwas_total = int(max(1, t))
                gwas_pbar = _ProgressAdapter(
                    total=gwas_total,
                    desc="FastLMM",
                    force_animate=True,
                )
                gwas_last_done = 0
            d = int(max(0, min(d, max(1, gwas_total))))
            step = int(max(0, d - gwas_last_done))
            if step > 0 and gwas_pbar is not None:
                gwas_pbar.update(step)
                gwas_last_done = d

        gm_tag = str(genetic_model).lower()
        if gm_tag == "add":
            out_tsv = f"{outprefix}.{pname}.fastlmm.tsv"
        else:
            out_tsv = f"{outprefix}.{pname}.{gm_tag}.fastlmm.tsv"

        gwas_ok = False
        lbd = float("nan")
        ml0 = float("nan")
        reml0 = float("nan")
        scan_secs = 0.0
        try:
            u_trait = np.zeros((int(packed_n), int(n_idv)), dtype=np.float32)
            u_trait[sample_idx_trait, :] = u_sub
            scan_t0 = time.monotonic()
            with _gwas_scan_stage_ctx(max(1, int(threads))):
                progress_kwargs: dict[str, object] = {}
                if gwas_pbar is not None:
                    progress_kwargs = {
                        "progress_callback": _fastlmm_progress,
                        "progress_every": int(max(1, min(gwas_total, int(max(1, chunk_size))))),
                    }
                unified_done = False
                if _gwas_use_rust_unified_v1():
                    try:
                        jobs = [
                            {
                                "model": "fastlmm",
                                "trait": str(pname),
                                "out_tsv": str(out_tsv),
                                "y": y_vec,
                                "x": x_arg,
                                "u": u_trait,
                                "s": s_trait,
                                "sample_indices": sample_idx_trait,
                                "low": -5.0,
                                "high": 5.0,
                                "max_iter": 50,
                                "tol": 1e-2,
                                "tau": 0.0,
                                "genetic_model": str(genetic_model),
                                "fixed_lbd": (float(fixed_lbd) if fixed_lbd is not None else None),
                                "fixed_ml0": (float(fixed_ml0) if fixed_ml0 is not None else None),
                                "scan_progress_callback": _fastlmm_progress,
                                "progress_every": int(max(1, int(chunk_size))),
                            }
                        ]
                        _res = jxrs.gwas_packed_unified_to_tsv(
                            jobs,
                            packed,
                            int(packed_n),
                            row_flip,
                            maf,
                            chrom_all,
                            pos_all,
                            allele0_all,
                            allele1_all,
                            int(max(1, int(chunk_size))),
                            int(threads),
                            None,
                            int(max(1, int(chunk_size))),
                            row_indices=packed_row_idx,
                        )
                        r0 = _res[0]
                        lbd = float(r0.get("lbd", np.nan))
                        ml0 = float(r0.get("ml0", np.nan))
                        reml0 = float(r0.get("reml0", np.nan))
                        unified_done = True
                    except Exception as ex:
                        _emit_warning_line(
                            logger,
                            f"FastLMM unified Rust dispatcher unavailable; fallback to legacy packed FastLMM. reason={ex}",
                            use_spinner=bool(use_spinner),
                        )
                        unified_done = False
                if not unified_done:
                    lbd, ml0, reml0 = jxrs.fastlmm_assoc_packed_f32_to_tsv(
                        packed,
                        int(packed_n),
                        row_flip,
                        maf,
                        u_trait,
                        s_trait,
                        y_vec,
                        x_arg,
                        sample_idx_trait,
                        -5.0,
                        5.0,
                        50,
                        1e-2,
                        0.0,
                        int(threads),
                        str(genetic_model),
                        chrom_all,
                        pos_all,
                        allele0_all,
                        allele1_all,
                        out_tsv,
                        row_indices=packed_row_idx,
                        fixed_lbd=(float(fixed_lbd) if fixed_lbd is not None else None),
                        fixed_ml0=(float(fixed_ml0) if fixed_ml0 is not None else None),
                        **progress_kwargs,
                    )
            scan_secs = max(time.monotonic() - scan_t0, 0.0)
            gwas_ok = True
        finally:
            if gwas_pbar is not None:
                try:
                    if gwas_ok:
                        if int(gwas_last_done) < int(max(1, gwas_total)):
                            gwas_pbar.update(int(max(1, gwas_total)) - int(gwas_last_done))
                        gwas_pbar.finish()
                        time.sleep(0.05)
                except Exception:
                    pass
                gwas_pbar.close(show_done=False)

        pve = None
        try:
            vg = float(np.mean(np.asarray(s_trait, dtype=np.float64)))
            lbd_v = float(lbd)
            if np.isfinite(vg) and np.isfinite(lbd_v) and (vg + lbd_v) > 0:
                pve = vg / (vg + lbd_v)
        except Exception:
            pve = None

        saved_paths.append(str(out_tsv))
        viz_secs = 0.0
        if plot:
            viz_secs = _run_fastplot_from_tsv_with_status(
                out_tsv,
                y_vec,
                xlabel=str(pname),
                outpdf=f"{os.path.splitext(out_tsv)[0]}.svg",
                use_spinner=bool(use_spinner),
                emit_done_line=False,
            )

        peak_rss = max(peak_rss, process.memory_info().rss)
        cpu_t1 = process.cpu_times()
        t1 = time.time()
        wall = max(t1 - t0, 1e-12)
        cpu_used = (cpu_t1.user - cpu_t0.user) + (cpu_t1.system - cpu_t0.system)
        avg_cpu = 100.0 * cpu_used / (wall * max(1, n_cores))
        peak_rss_gb = peak_rss / (1024 ** 3)

        eff_snp = int(len(sites_all))
        eff_snp_by_trait[pname] = eff_snp
        summary_rows.append(
            {
                "phenotype": str(pname),
                "model": "FastLMM",
                "nidv": int(n_idv),
                "eff_snp": int(eff_snp),
                "pve": (float(pve) if pve is not None else None),
                "avg_cpu": float(avg_cpu),
                "peak_rss_gb": float(peak_rss_gb),
                "gwas_time_s": float(grm_secs + evd_secs + scan_secs),
                "viz_time_s": float(viz_secs),
                "result_file": str(out_tsv),
            }
        )

        done_times: list[str] = []
        if grm_secs > 0:
            done_times.append(format_elapsed(grm_secs))
        done_times.extend([format_elapsed(evd_secs), format_elapsed(scan_secs)])
        if plot:
            done_times.append(format_elapsed(viz_secs))
        _rich_success(
            logger,
            (
                f"FastLMM ...pve {float(pve):.3f} [{'/'.join(done_times)}]"
                if (pve is not None and np.isfinite(float(pve)))
                else f"FastLMM ...Finished [{'/'.join(done_times)}]"
            ),
            use_spinner=bool(use_spinner),
        )
        if multi_trait_mode:
            logger.info("")


def run_lm_packed_fullrank(
    *,
    genofile: str,
    pheno: pd.DataFrame,
    ids: np.ndarray,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    genetic_model: str,
    het_threshold: float,
    chunk_size: int,
    qmatrix: np.ndarray,
    cov_all: Union[np.ndarray, None],
    plot: bool,
    threads: int,
    logger: logging.Logger,
    use_spinner: bool = False,
    snps_only: bool = True,
    eff_snp_by_trait: Union[dict[str, int], None] = None,
    summary_rows: Union[list[dict[str, object]], None] = None,
    saved_paths: Union[list[str], None] = None,
    trait_names: Union[list[str], None] = None,
    emit_trait_header: bool = True,
    preloaded_packed: Union[dict[str, object], None] = None,
) -> None:
    if not hasattr(jxrs, "glmf32_packed_assoc_to_tsv"):
        raise RuntimeError(
            "Rust extension missing glmf32_packed_assoc_to_tsv. Rebuild/install JanusX extension first."
        )
    if str(genetic_model).lower() != "add":
        raise ValueError(
            "LM full-rust packed route currently supports additive coding only (--model add)."
        )

    prefix, full_ids, packed_ctx, sites_all = _prepare_packed_bed_once_for_gwas(
        genofile=genofile,
        maf_threshold=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
        het_threshold=float(het_threshold),
        snps_only=bool(snps_only),
        use_spinner=bool(use_spinner),
        preloaded_packed=preloaded_packed,
    )
    id_to_idx = {sid: i for i, sid in enumerate(full_ids)}
    try:
        sample_map = np.asarray([id_to_idx[str(sid)] for sid in np.asarray(ids, dtype=str)], dtype=np.int64)
    except KeyError as e:
        raise ValueError("Some aligned sample IDs are not present in packed BED sample order.") from e

    packed, packed_n, packed_row_idx, maf, row_flip = _packed_ctx_active_view_for_gwas(
        packed_ctx
    )
    if packed_n <= 0:
        raise ValueError("Packed BED reported invalid sample count.")
    if int(packed_row_idx.shape[0]) != int(maf.shape[0]) or int(packed_row_idx.shape[0]) != int(row_flip.shape[0]):
        raise ValueError("Packed BED arrays have inconsistent SNP dimensions.")

    if int(len(sites_all)) != int(packed_row_idx.shape[0]):
        raise ValueError(
            f"Packed/BIM mismatch after filtering: sites={len(sites_all)} active_rows={packed_row_idx.shape[0]}"
        )
    chrom_all = [str(c) for (c, _p, _a0, _a1) in sites_all]
    pos_all = [int(p) for (_c, p, _a0, _a1) in sites_all]
    allele0_all = [str(v) for (_c, _p, v, _a1) in sites_all]
    allele1_all = [str(v) for (_c, _p, _a0, v) in sites_all]

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pheno_aligned, ids = _align_pheno_to_sample_order(pheno, ids)
    trait_iter = list(pheno_aligned.columns) if trait_names is None else [t for t in trait_names if t in pheno_aligned.columns]
    multi_trait_mode = len(trait_iter) > 1

    for pname in trait_iter:
        cpu_t0 = process.cpu_times()
        t0 = time.time()
        peak_rss = process.memory_info().rss

        y_full, sameidx = _trait_values_and_mask(pheno_aligned, str(pname))
        keep_idx = np.flatnonzero(sameidx).astype(np.int64, copy=False)
        n_idv = int(keep_idx.shape[0])
        if n_idv == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")
            continue

        if bool(emit_trait_header):
            _emit_trait_header(
                logger,
                str(pname),
                int(n_idv),
                pve=None,
                use_spinner=bool(use_spinner),
                width=60,
            )

        y_vec = np.ascontiguousarray(y_full[keep_idx], dtype=np.float64)
        x_cov = qmatrix[keep_idx]
        if cov_all is not None:
            x_cov = np.concatenate([x_cov, cov_all[keep_idx]], axis=1)
        x_cov = np.ascontiguousarray(x_cov, dtype=np.float64)
        x_design = np.ascontiguousarray(
            np.concatenate(
                [np.ones((int(x_cov.shape[0]), 1), dtype=np.float64), x_cov],
                axis=1,
            ),
            dtype=np.float64,
        )
        sample_idx_trait = np.ascontiguousarray(sample_map[keep_idx], dtype=np.int64)
        gm_tag = str(genetic_model).lower()
        if gm_tag == "add":
            out_tsv = f"{outprefix}.{pname}.lm.tsv"
        else:
            out_tsv = f"{outprefix}.{pname}.{gm_tag}.lm.tsv"

        gwas_t0 = time.monotonic()
        gwas_total = int(len(sites_all))
        gwas_last_done = 0
        gwas_pbar: Optional[_ProgressAdapter] = None
        if bool(use_spinner):
            gwas_pbar = _ProgressAdapter(
                total=max(1, gwas_total),
                desc="LM",
                force_animate=True,
            )

        def _lm_progress(done: int, total: int) -> None:
            nonlocal gwas_last_done, gwas_total, gwas_pbar
            if gwas_pbar is None:
                return
            try:
                d = int(done)
                t = int(total)
            except Exception:
                return
            if t > 0 and t != gwas_total and gwas_last_done == 0:
                try:
                    gwas_pbar.close(show_done=False)
                except Exception:
                    pass
                gwas_total = int(max(1, t))
                gwas_pbar = _ProgressAdapter(
                    total=gwas_total,
                    desc="LM",
                    force_animate=True,
                )
                gwas_last_done = 0
            d = int(max(0, min(d, max(1, gwas_total))))
            stepv = int(max(0, d - gwas_last_done))
            if stepv > 0 and gwas_pbar is not None:
                gwas_pbar.update(stepv)
                gwas_last_done = d

        gwas_ok = False
        try:
            with _gwas_scan_stage_ctx(max(1, int(threads))):
                kwargs_assoc: dict[str, object] = {}
                if gwas_pbar is not None:
                    kwargs_assoc = {
                        "progress_callback": _lm_progress,
                        "progress_every": int(max(1, int(chunk_size))),
                    }
                unified_done = False
                if _gwas_use_rust_unified_v1():
                    try:
                        jobs = [
                            {
                                "model": "lm",
                                "trait": str(pname),
                                "out_tsv": str(out_tsv),
                                "y": y_vec,
                                "x": x_design,
                                "ixx": None,
                                "sample_indices": sample_idx_trait,
                                "step": int(max(1, int(chunk_size))),
                                "scan_progress_callback": _lm_progress,
                                "progress_every": int(max(1, int(chunk_size))),
                            }
                        ]
                        _res = jxrs.gwas_packed_unified_to_tsv(
                            jobs,
                            packed,
                            int(packed_n),
                            row_flip,
                            maf,
                            chrom_all,
                            pos_all,
                            allele0_all,
                            allele1_all,
                            int(max(1, int(chunk_size))),
                            int(threads),
                            None,
                            int(max(1, int(chunk_size))),
                            row_indices=packed_row_idx,
                        )
                        try:
                            _written_rows = int(_res[0].get("written_rows", len(chrom_all)))
                        except Exception:
                            _written_rows = int(len(chrom_all))
                        unified_done = True
                    except Exception as ex:
                        _emit_warning_line(
                            logger,
                            f"LM unified Rust dispatcher unavailable; fallback to legacy packed LM. reason={ex}",
                            use_spinner=bool(use_spinner),
                        )
                        unified_done = False

                if not unified_done:
                    try:
                        _written_rows = jxrs.glmf32_packed_assoc_to_tsv(
                            y_vec,
                            x_design,
                            None,
                            packed,
                            int(packed_n),
                            row_flip,
                            maf,
                            chrom_all,
                            pos_all,
                            allele0_all,
                            allele1_all,
                            out_tsv,
                            sample_idx_trait,
                            row_indices=packed_row_idx,
                            step=int(max(1, int(chunk_size))),
                            threads=int(threads),
                            **kwargs_assoc,
                        )
                    except TypeError:
                        # Backward-compat fallback for older extensions that require ixx.
                        ixx = _lm_precompute_ixx_qr(x_design)
                        _written_rows = jxrs.glmf32_packed_assoc_to_tsv(
                            y_vec,
                            x_design,
                            ixx,
                            packed,
                            int(packed_n),
                            row_flip,
                            maf,
                            chrom_all,
                            pos_all,
                            allele0_all,
                            allele1_all,
                            out_tsv,
                            sample_idx_trait,
                            row_indices=packed_row_idx,
                            step=int(max(1, int(chunk_size))),
                            threads=int(threads),
                            **kwargs_assoc,
                        )
            gwas_ok = True
        finally:
            if gwas_pbar is not None:
                try:
                    if gwas_ok:
                        if int(gwas_last_done) < int(max(1, gwas_total)):
                            gwas_pbar.update(int(max(1, gwas_total)) - int(gwas_last_done))
                        gwas_pbar.finish()
                        time.sleep(0.05)
                except Exception:
                    pass
                gwas_pbar.close(show_done=False)

        gwas_secs = max(time.monotonic() - gwas_t0, 0.0)
        saved_paths.append(str(out_tsv))
        viz_secs = 0.0
        if plot:
            viz_secs = _run_fastplot_from_tsv_with_status(
                out_tsv,
                y_vec,
                xlabel=str(pname),
                outpdf=f"{os.path.splitext(out_tsv)[0]}.svg",
                use_spinner=bool(use_spinner),
                emit_done_line=False,
            )

        peak_rss = max(peak_rss, process.memory_info().rss)
        cpu_t1 = process.cpu_times()
        t1 = time.time()
        wall = max(t1 - t0, 1e-12)
        cpu_used = (cpu_t1.user - cpu_t0.user) + (cpu_t1.system - cpu_t0.system)
        avg_cpu = 100.0 * cpu_used / (wall * max(1, n_cores))
        peak_rss_gb = peak_rss / (1024 ** 3)

        eff_snp = int(len(sites_all))
        eff_snp_by_trait[pname] = eff_snp
        summary_rows.append(
            {
                "phenotype": str(pname),
                "model": "LM",
                "nidv": int(n_idv),
                "eff_snp": int(eff_snp),
                "pve": None,
                "avg_cpu": float(avg_cpu),
                "peak_rss_gb": float(peak_rss_gb),
                "gwas_time_s": float(gwas_secs),
                "viz_time_s": float(viz_secs),
                "result_file": str(out_tsv),
            }
        )

        done_times = [format_elapsed(gwas_secs)]
        if plot:
            done_times.append(format_elapsed(viz_secs))
        _rich_success(
            logger,
            f"LM ...Finished [{'/'.join(done_times)}]",
            use_spinner=bool(use_spinner),
        )
        if multi_trait_mode:
            logger.info("")


def run_lm_stream_bed_single_entry(
    *,
    genofile: str,
    pheno: pd.DataFrame,
    ids: np.ndarray,
    n_snps: int,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    genetic_model: str,
    het_threshold: float,
    chunk_size: int,
    qmatrix: np.ndarray,
    cov_all: Union[np.ndarray, None],
    plot: bool,
    threads: int,
    logger: logging.Logger,
    use_spinner: bool = False,
    snps_only: bool = True,
    eff_snp_by_trait: Union[dict[str, int], None] = None,
    summary_rows: Union[list[dict[str, object]], None] = None,
    saved_paths: Union[list[str], None] = None,
    trait_names: Union[list[str], None] = None,
    emit_trait_header: bool = True,
    chunk_size_user_set: bool = True,
    mmap_limit: bool = False,
) -> None:
    if not hasattr(jxrs, "lm_stream_bed_to_tsv"):
        raise RuntimeError(
            "Rust extension missing lm_stream_bed_to_tsv. Rebuild/install JanusX extension first."
        )
    prefix = _as_plink_prefix(genofile)
    if prefix is None:
        raise ValueError("LM rust streaming single-entry route requires PLINK BED input.")

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pheno_aligned, ids = _align_pheno_to_sample_order(pheno, ids)
    trait_iter = (
        list(pheno_aligned.columns)
        if trait_names is None
        else [t for t in trait_names if t in pheno_aligned.columns]
    )
    multi_trait_mode = len(trait_iter) > 1

    for pname in trait_iter:
        cpu_t0 = process.cpu_times()
        t0 = time.time()
        peak_rss = process.memory_info().rss

        y_full, sameidx = _trait_values_and_mask(pheno_aligned, str(pname))
        keep_idx = np.flatnonzero(sameidx).astype(np.int64, copy=False)
        n_idv = int(keep_idx.shape[0])
        if n_idv == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")
            continue

        if bool(emit_trait_header):
            _emit_trait_header(
                logger,
                str(pname),
                int(n_idv),
                pve=None,
                use_spinner=bool(use_spinner),
                width=60,
            )

        trait_ids = np.asarray(ids[keep_idx], dtype=str)
        y_vec = np.ascontiguousarray(y_full[keep_idx], dtype=np.float64)
        x_cov = qmatrix[keep_idx]
        if cov_all is not None:
            x_cov = np.concatenate([x_cov, cov_all[keep_idx]], axis=1)
        x_cov = np.ascontiguousarray(x_cov, dtype=np.float64)
        x_design = np.ascontiguousarray(
            np.concatenate(
                [np.ones((int(x_cov.shape[0]), 1), dtype=np.float64), x_cov],
                axis=1,
            ),
            dtype=np.float64,
        )

        model_chunk_size = _resolve_stream_scan_chunk_size(
            int(chunk_size),
            int(n_snps),
            use_spinner=bool(use_spinner),
            n_samples_hint=int(n_idv),
            model_keys="lm",
            user_specified=bool(chunk_size_user_set),
        )
        mmap_window_mb = (
            auto_mmap_window_mb(genofile, len(ids), n_snps, int(model_chunk_size))
            if bool(mmap_limit)
            else None
        )
        if (not bool(chunk_size_user_set)) and int(model_chunk_size) != int(chunk_size):
            _log_file_only(
                logger,
                logging.INFO,
                f"LM auto chunk-size: {int(chunk_size)} -> {int(model_chunk_size)} "
                f"(n={int(n_idv)}).",
            )

        gm_tag = str(genetic_model).lower()
        if gm_tag == "add":
            out_tsv = f"{outprefix}.{pname}.lm.tsv"
        else:
            out_tsv = f"{outprefix}.{pname}.{gm_tag}.lm.tsv"
        tmp_tsv = f"{out_tsv}.tmp.{os.getpid()}.{uuid.uuid4().hex}"

        scan_threads = int(threads)
        if scan_threads <= 0:
            scan_threads = int(n_cores)
        scan_t0 = time.monotonic()
        pbar_total_hint = int(eff_snp_by_trait.get(str(pname), n_snps))
        pbar: Optional[_ProgressAdapter] = None
        pbar_done = 0
        warm_task: Optional[CliStatus] = None
        warm_active = False
        if bool(use_spinner):
            warm_task = CliStatus(
                "Waiting for LM GWAS",
                enabled=True,
                use_process=True,
            )
            warm_task.__enter__()
            warm_active = True

        def _lm_progress(done: int, total: int) -> None:
            nonlocal pbar, pbar_done, warm_active
            try:
                d = int(done)
                t = int(total)
            except Exception:
                return
            if pbar is None:
                if warm_active and warm_task is not None:
                    try:
                        warm_task.__exit__(None, None, None)
                    except Exception:
                        pass
                    warm_active = False
                total_use = int(max(1, t if t > 0 else pbar_total_hint))
                pbar = _ProgressAdapter(total=total_use, desc="LM", force_animate=True)
            total_cap = int(max(1, pbar.total))
            d = int(max(0, min(d, total_cap)))
            stepv = int(max(0, d - int(pbar_done)))
            if stepv > 0:
                pbar.update(stepv)
                pbar_done = d

        kept_rows = 0
        scan_all_rows = 0
        scan_ok = False
        try:
            with _gwas_scan_stage_ctx(int(max(1, scan_threads))):
                kwargs_stream: dict[str, object] = {}
                if bool(use_spinner):
                    kwargs_stream = {
                        "progress_callback": _lm_progress,
                        "progress_every": int(max(1, int(model_chunk_size))),
                    }
                lm_base_kwargs = {
                    "sample_ids": trait_ids.tolist(),
                    "maf_threshold": float(maf_threshold),
                    "max_missing_rate": float(max_missing_rate),
                    "genetic_model": str(genetic_model),
                    "het_threshold": float(het_threshold),
                    "snps_only": bool(snps_only),
                    "chunk_size": int(max(1, int(model_chunk_size))),
                    "threads": int(max(1, scan_threads)),
                    **kwargs_stream,
                }

                def _call_lm_stream(ixx_opt: object) -> tuple[int, int]:
                    global _WARNED_LM_STREAM_MMAP_LEGACY
                    kwargs_use = dict(lm_base_kwargs)
                    if mmap_window_mb is not None:
                        kwargs_use["mmap_window_mb"] = int(max(1, int(mmap_window_mb)))
                    try:
                        return jxrs.lm_stream_bed_to_tsv(
                            str(prefix),
                            y_vec,
                            x_design,
                            ixx_opt,
                            str(tmp_tsv),
                            **kwargs_use,
                        )
                    except TypeError as ex:
                        msg = str(ex).lower()
                        mmap_kw_err = (
                            "mmap_window_mb" in msg
                            or "unexpected keyword argument" in msg
                        )
                        if ("mmap_window_mb" in kwargs_use) and mmap_kw_err:
                            kwargs_use.pop("mmap_window_mb", None)
                            if not _WARNED_LM_STREAM_MMAP_LEGACY:
                                _WARNED_LM_STREAM_MMAP_LEGACY = True
                                _emit_warning_line(
                                    logger,
                                    (
                                        "Rust extension missing lm_stream_bed_to_tsv(mmap_window_mb=...); "
                                        "falling back to full-mmap LM stream path. Rebuild JanusX to enable memmap window mode."
                                    ),
                                    use_spinner=bool(use_spinner),
                                )
                            return jxrs.lm_stream_bed_to_tsv(
                                str(prefix),
                                y_vec,
                                x_design,
                                ixx_opt,
                                str(tmp_tsv),
                                **kwargs_use,
                            )
                        raise
                try:
                    kept_rows, scan_all_rows = _call_lm_stream(None)
                except TypeError:
                    # Backward-compat fallback for older extensions that require ixx.
                    ixx = _lm_precompute_ixx_qr(x_design)
                    kept_rows, scan_all_rows = _call_lm_stream(ixx)
            scan_ok = True
        finally:
            if warm_active and warm_task is not None:
                try:
                    warm_task.__exit__(None, None, None)
                except Exception:
                    pass
                warm_active = False
            if pbar is not None:
                try:
                    if scan_ok:
                        if int(pbar_done) < int(max(1, pbar.total)):
                            pbar.update(int(max(1, pbar.total)) - int(pbar_done))
                        pbar.finish()
                        time.sleep(0.05)
                except Exception:
                    pass
                pbar.close(show_done=False)

        gwas_secs = max(time.monotonic() - scan_t0, 0.0)
        peak_rss = max(peak_rss, process.memory_info().rss)
        cpu_t1 = process.cpu_times()
        t1 = time.time()
        wall = max(t1 - t0, 1e-12)
        cpu_used = (cpu_t1.user - cpu_t0.user) + (cpu_t1.system - cpu_t0.system)
        avg_cpu = 100.0 * cpu_used / (wall * max(1, n_cores))
        peak_rss_gb = peak_rss / (1024 ** 3)

        _log_model_line(
            logger,
            "LM",
            "single-entry rust streaming scan",
            use_spinner=bool(use_spinner),
        )
        _log_model_line(
            logger,
            "LM",
            f"avg CPU ~ {avg_cpu:.1f}% of {n_cores} c, peak RSS ~ {peak_rss_gb:.2f} G",
            use_spinner=bool(use_spinner),
        )
        _log_file_only(
            logger,
            logging.INFO,
            f"LM stream source SNP rows scanned: {int(scan_all_rows)}",
        )

        done_snps = int(kept_rows)
        if done_snps <= 0:
            if os.path.exists(tmp_tsv):
                os.remove(tmp_tsv)
            logger.info(f"LM: no SNPs passed filters for trait {pname}.")
            eff_snp_by_trait[str(pname)] = int(done_snps)
            summary_rows.append(
                {
                    "phenotype": str(pname),
                    "model": "LM",
                    "nidv": int(n_idv),
                    "eff_snp": int(done_snps),
                    "pve": None,
                    "avg_cpu": float(avg_cpu),
                    "peak_rss_gb": float(peak_rss_gb),
                    "gwas_time_s": float(gwas_secs),
                    "viz_time_s": 0.0,
                    "result_file": "",
                }
            )
            if multi_trait_mode:
                logger.info("")
            continue

        _run_result_write_with_status(
            lambda: os.replace(tmp_tsv, out_tsv),
            use_spinner=bool(use_spinner),
            emit_done_line=False,
        )
        saved_paths.append(str(out_tsv))
        _log_model_line(
            logger,
            "LM",
            f"Results saved to {_display_path(str(out_tsv))}",
            use_spinner=bool(use_spinner),
        )

        viz_secs = 0.0
        if plot:
            viz_secs = _run_fastplot_from_tsv_with_status(
                out_tsv,
                y_vec,
                xlabel=str(pname),
                outpdf=f"{os.path.splitext(out_tsv)[0]}.svg",
                use_spinner=bool(use_spinner),
                emit_done_line=False,
            )

        eff_snp_by_trait[str(pname)] = int(done_snps)
        summary_rows.append(
            {
                "phenotype": str(pname),
                "model": "LM",
                "nidv": int(n_idv),
                "eff_snp": int(done_snps),
                "pve": None,
                "avg_cpu": float(avg_cpu),
                "peak_rss_gb": float(peak_rss_gb),
                "gwas_time_s": float(gwas_secs),
                "viz_time_s": float(viz_secs),
                "result_file": str(out_tsv),
            }
        )

        done_parts = [format_elapsed(gwas_secs)]
        if plot:
            done_parts.append(format_elapsed(viz_secs))
        _rich_success(
            logger,
            f"LM ...Finished [{'/'.join(done_parts)}]",
            use_spinner=bool(use_spinner),
        )
        if multi_trait_mode:
            logger.info("")


def run_lmm_packed_fullrank(
    *,
    genofile: str,
    pheno: pd.DataFrame,
    ids: np.ndarray,
    grm: np.ndarray,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    genetic_model: str,
    het_threshold: float,
    chunk_size: int,
    qmatrix: np.ndarray,
    cov_all: Union[np.ndarray, None],
    plot: bool,
    threads: int,
    logger: logging.Logger,
    use_spinner: bool = False,
    snps_only: bool = True,
    eff_snp_by_trait: Union[dict[str, int], None] = None,
    summary_rows: Union[list[dict[str, object]], None] = None,
    saved_paths: Union[list[str], None] = None,
    trait_names: Union[list[str], None] = None,
    emit_trait_header: bool = True,
    preloaded_packed: Union[dict[str, object], None] = None,
) -> None:
    if not hasattr(jxrs, "lmm_reml_assoc_packed_f32_to_tsv"):
        raise RuntimeError(
            "Rust extension missing lmm_reml_assoc_packed_f32_to_tsv. Rebuild/install JanusX extension first."
        )
    if str(genetic_model).lower() != "add":
        raise ValueError(
            "LMM full-rust packed route currently supports additive coding only (--model add)."
        )
    if grm is None:
        raise ValueError("LMM full-rust packed route requires a prepared GRM.")

    prefix, full_ids, packed_ctx, sites_all = _prepare_packed_bed_once_for_gwas(
        genofile=genofile,
        maf_threshold=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
        het_threshold=float(het_threshold),
        snps_only=bool(snps_only),
        use_spinner=bool(use_spinner),
        preloaded_packed=preloaded_packed,
    )
    id_to_idx = {sid: i for i, sid in enumerate(full_ids)}
    try:
        sample_map = np.asarray([id_to_idx[str(sid)] for sid in np.asarray(ids, dtype=str)], dtype=np.int64)
    except KeyError as e:
        raise ValueError("Some aligned sample IDs are not present in packed BED sample order.") from e

    packed, packed_n, packed_row_idx, maf, row_flip = _packed_ctx_active_view_for_gwas(
        packed_ctx
    )
    if packed_n <= 0:
        raise ValueError("Packed BED reported invalid sample count.")
    if int(packed_row_idx.shape[0]) != int(maf.shape[0]) or int(packed_row_idx.shape[0]) != int(row_flip.shape[0]):
        raise ValueError("Packed BED arrays have inconsistent SNP dimensions.")

    if int(len(sites_all)) != int(packed_row_idx.shape[0]):
        raise ValueError(
            f"Packed/BIM mismatch after filtering: sites={len(sites_all)} active_rows={packed_row_idx.shape[0]}"
        )

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pheno_aligned, ids = _align_pheno_to_sample_order(pheno, ids)
    trait_iter = list(pheno_aligned.columns) if trait_names is None else [t for t in trait_names if t in pheno_aligned.columns]
    multi_trait_mode = len(trait_iter) > 1

    for pname in trait_iter:
        cpu_t0 = process.cpu_times()
        t0 = time.time()
        peak_rss = process.memory_info().rss

        y_full, sameidx = _trait_values_and_mask(pheno_aligned, str(pname))
        keep_idx = np.flatnonzero(sameidx).astype(np.int64, copy=False)
        n_idv = int(keep_idx.shape[0])
        if n_idv == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")
            continue

        if bool(emit_trait_header):
            _emit_trait_header(
                logger,
                str(pname),
                int(n_idv),
                pve=None,
                use_spinner=bool(use_spinner),
                width=60,
            )

        y_vec = np.ascontiguousarray(y_full[keep_idx], dtype=np.float64)
        x_cov = qmatrix[keep_idx]
        if cov_all is not None:
            x_cov = np.concatenate([x_cov, cov_all[keep_idx]], axis=1)

        Ksub = grm[np.ix_(keep_idx, keep_idx)]
        evd_t0 = time.monotonic()
        with CliStatus(
            "Running LMM Eigen-Decomposition...",
            enabled=bool(use_spinner),
            use_process=True,
        ) as task:
            try:
                stage_threads = max(1, int(threads))
                with _gwas_evd_stage_ctx(stage_threads):
                    mod = LMM(y=y_vec, X=x_cov, kinship=Ksub)
            except Exception:
                task.fail("LMM Eigen-Decomposition ...Failed")
                raise
        evd_secs = max(time.monotonic() - evd_t0, 0.0)
        header_pve: Optional[float] = None
        try:
            pve_tmp = float(mod.pve)
            if np.isfinite(pve_tmp):
                header_pve = pve_tmp
        except Exception:
            header_pve = None
        _log_model_line(
            logger,
            "LMM",
            f"PVE(null) ~ {mod.pve:.3f}; eigen-decomposition [{format_elapsed(evd_secs)}]",
            use_spinner=bool(use_spinner),
        )

        sample_idx_trait = np.ascontiguousarray(sample_map[keep_idx], dtype=np.int64)
        # Exact LMM route: per-SNP REML lambda re-estimation.
        s_trait = np.ascontiguousarray(
            np.maximum(np.asarray(mod.S, dtype=np.float64).reshape(-1), 0.0),
            dtype=np.float64,
        )
        u_t = np.ascontiguousarray(np.asarray(mod.Dh, dtype=np.float32), dtype=np.float32)
        if u_t.shape != (n_idv, n_idv):
            raise ValueError(
                f"Trait eigenvector shape mismatch: got {u_t.shape}, expected ({n_idv}, {n_idv})."
            )
        x_rot = np.ascontiguousarray(np.asarray(mod.Xcov, dtype=np.float64), dtype=np.float64)
        y_rot = np.ascontiguousarray(np.asarray(mod.y, dtype=np.float64).reshape(-1), dtype=np.float64)

        gwas_t0 = time.monotonic()
        gwas_total = int(len(sites_all))
        gwas_last_done = 0
        gwas_pbar: Optional[_ProgressAdapter] = None
        if bool(use_spinner):
            gwas_pbar = _ProgressAdapter(
                total=max(1, gwas_total),
                desc="LMM",
                force_animate=True,
            )

        def _lmm_progress(done: int, total: int) -> None:
            nonlocal gwas_last_done, gwas_total, gwas_pbar
            if gwas_pbar is None:
                return
            try:
                d = int(done)
                t = int(total)
            except Exception:
                return
            if t > 0 and t != gwas_total and gwas_last_done == 0:
                try:
                    gwas_pbar.close(show_done=False)
                except Exception:
                    pass
                gwas_total = int(max(1, t))
                gwas_pbar = _ProgressAdapter(
                    total=gwas_total,
                    desc="LMM",
                    force_animate=True,
                )
                gwas_last_done = 0
            d = int(max(0, min(d, max(1, gwas_total))))
            step = int(max(0, d - gwas_last_done))
            if step > 0 and gwas_pbar is not None:
                gwas_pbar.update(step)
                gwas_last_done = d

        gwas_ok = False
        null_lbd = float("nan")
        chrom_all = [str(c) for (c, _p, _a0, _a1) in sites_all]
        pos_all = [int(p) for (_c, p, _a0, _a1) in sites_all]
        allele0_all = [str(v) for (_c, _p, v, _a1) in sites_all]
        allele1_all = [str(v) for (_c, _p, _a0, v) in sites_all]
        gm_tag = str(genetic_model).lower()
        if gm_tag == "add":
            out_tsv = f"{outprefix}.{pname}.lmm.tsv"
        else:
            out_tsv = f"{outprefix}.{pname}.{gm_tag}.lmm.tsv"
        try:
            progress_kwargs: dict[str, object] = {}
            if gwas_pbar is not None:
                progress_kwargs = {
                    "progress_callback": _lmm_progress,
                    "progress_every": int(max(1, int(chunk_size))),
                }
            with _gwas_scan_stage_ctx(max(1, int(threads))):
                null_ml0: Optional[float] = None
                init_log10_lbd: Optional[float] = None
                try:
                    ml0_tmp = float(getattr(mod, "ML0"))
                    if np.isfinite(ml0_tmp):
                        null_ml0 = ml0_tmp
                except Exception:
                    null_ml0 = None
                try:
                    lbd0_tmp = float(getattr(mod, "lbd_null"))
                    if np.isfinite(lbd0_tmp) and lbd0_tmp > 0.0:
                        init_log10_lbd = float(np.log10(lbd0_tmp))
                except Exception:
                    init_log10_lbd = None

                _written_rows = jxrs.lmm_reml_assoc_packed_f32_to_tsv(
                    packed,
                    int(packed_n),
                    row_flip,
                    maf,
                    s_trait,
                    x_rot,
                    y_rot,
                    u_t,
                    chrom_all,
                    pos_all,
                    allele0_all,
                    allele1_all,
                    out_tsv,
                    sample_idx_trait,
                    row_indices=packed_row_idx,
                    low=-5.0,
                    high=5.0,
                    max_iter=50,
                    tol=1e-2,
                    threads=int(threads),
                    model="add",
                    nullml=null_ml0,
                    init_log10_lbd=init_log10_lbd,
                    **progress_kwargs,
                )
                if int(_written_rows) != int(len(sites_all)):
                    _emit_warning_line(
                        logger,
                        f"LMM Rust writer row mismatch: expected={len(sites_all)} wrote={int(_written_rows)}",
                        use_spinner=bool(use_spinner),
                    )
                try:
                    null_lbd = float(getattr(mod, "lbd_null"))
                except Exception:
                    null_lbd = float("nan")
            gwas_ok = True
        finally:
            if gwas_pbar is not None:
                try:
                    if gwas_ok:
                        if int(gwas_last_done) < int(max(1, gwas_total)):
                            gwas_pbar.update(int(max(1, gwas_total)) - int(gwas_last_done))
                        gwas_pbar.finish()
                        time.sleep(0.05)
                except Exception:
                    pass
                gwas_pbar.close(show_done=False)

        gwas_secs = max(time.monotonic() - gwas_t0, 0.0)
        saved_paths.append(str(out_tsv))
        viz_secs = 0.0
        if plot:
            viz_secs = _run_fastplot_from_tsv_with_status(
                out_tsv,
                y_vec,
                xlabel=str(pname),
                outpdf=f"{os.path.splitext(out_tsv)[0]}.svg",
                use_spinner=bool(use_spinner),
                emit_done_line=False,
            )

        peak_rss = max(peak_rss, process.memory_info().rss)
        cpu_t1 = process.cpu_times()
        t1 = time.time()
        wall = max(t1 - t0, 1e-12)
        cpu_used = (cpu_t1.user - cpu_t0.user) + (cpu_t1.system - cpu_t0.system)
        avg_cpu = 100.0 * cpu_used / (wall * max(1, n_cores))
        peak_rss_gb = peak_rss / (1024 ** 3)
        _log_model_line(
            logger,
            "LMM",
            (
                f"full-rust packed exact REML scan (per-SNP lambda); null lambda={float(null_lbd):.4g}"
                if np.isfinite(float(null_lbd))
                else "full-rust packed exact REML scan (per-SNP lambda)"
            ),
            use_spinner=bool(use_spinner),
        )
        _log_model_line(
            logger,
            "LMM",
            f"avg CPU ~ {avg_cpu:.1f}% of {n_cores} c, peak RSS ~ {peak_rss_gb:.2f} G",
            use_spinner=bool(use_spinner),
        )
        _log_model_line(
            logger,
            "LMM",
            f"Results saved to {_display_path(str(out_tsv))}",
            use_spinner=bool(use_spinner),
        )

        eff_snp = int(len(sites_all))
        eff_snp_by_trait[pname] = eff_snp
        summary_rows.append(
            {
                "phenotype": str(pname),
                "model": "LMM",
                "nidv": int(n_idv),
                "eff_snp": int(eff_snp),
                "pve": (
                    float(header_pve)
                    if header_pve is not None
                    else None
                ),
                "avg_cpu": float(avg_cpu),
                "peak_rss_gb": float(peak_rss_gb),
                "gwas_time_s": float(evd_secs + gwas_secs),
                "viz_time_s": float(viz_secs),
                "result_file": str(out_tsv),
            }
        )

        done_times = [format_elapsed(evd_secs), format_elapsed(gwas_secs)]
        if plot:
            done_times.append(format_elapsed(viz_secs))
        _rich_success(
            logger,
            (
                f"LMM ...pve {float(header_pve):.3f} [{'/'.join(done_times)}]"
                if (header_pve is not None and np.isfinite(float(header_pve)))
                else f"LMM ...Finished [{'/'.join(done_times)}]"
            ),
            use_spinner=bool(use_spinner),
        )
        if multi_trait_mode:
            logger.info("")
