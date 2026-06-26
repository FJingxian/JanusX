# -*- coding: utf-8 -*-
"""Packed/full-rust GWAS model runners (extracted from workflow.py)."""

from __future__ import annotations

import concurrent.futures as cf
import gc
import hashlib
import logging
import os
import time
import uuid
import json
from bisect import bisect_right
from typing import Optional, Union

import numpy as np
import pandas as pd
import psutil
from janusx.assoc.workflow_cache import _is_writable_dir, _resolve_gwas_cache_dir
from janusx.script._common.memory import (
    resolve_decode_mmap_window_mb as _common_resolve_decode_mmap_window_mb,
)
from janusx.script._common.genoio import (
    packed_preload_is_disabled,
    packed_preload_is_ready,
    packed_preload_reason,
    prepare_packed_ctx_from_plink,
)

from .workflow import (
    _BimSiteColumns,
    CliStatus,
    FastLMM,
    FvLMM,
    LMM,
    _ProgressAdapter,
    _start_indeterminate_progress_bar,
    _stop_indeterminate_progress_bar,
    _align_pheno_to_sample_order,
    _as_plink_prefix,
    _cleanup_gwas_result_tmp,
    _display_path,
    _emit_trait_header,
    _emit_plain_info_line,
    _emit_warning_line,
    _fastlmm_should_switch_to_lmm,
    _finalize_gwas_result_tsv,
    _gwas_result_tmp_path,
    _gwas_eigh_from_grm,
    _gwas_evd_stage_ctx,
    _gwas_fvlmm_scan_stage_ctx,
    _gwas_scan_stage_ctx,
    _is_full_identity_index,
    _log_file_only,
    _log_model_line,
    _make_deferred_bim_metadata,
    _mixed_model_switch_to_lm_decision,
    _progress_callback_step,
    _coerce_bim_site_columns,
    _resolve_bim_site_columns_meta,
    _replace_file_with_retry,
    _resolve_trait_iter,
    _resolve_stream_scan_chunk_size,
    _rich_success,
    _run_fastplot_from_tsv_with_status,
    _run_result_write_with_status,
    _safe_trait_file_label,
    _site_tuple_parts,
    _subset_square_matrix_identity_aware,
    _trait_values_and_mask,
    detect_effective_threads,
    format_elapsed,
    jxrs,
    prepare_cli_input_cache,
)

_WARNED_LM_STREAM_MMAP_LEGACY = False
_SPLMM_EXACT_N_MAX = 15_000
_SPLMM_KING_THRESHOLD = 0.05
_SPLMM_SPARSE_GRM_CUTOFF = 0.05
_SPLMM_SPARSE_GRM_METHOD = 2
_SPLMM_SPARSE_REML_GRID_SIZE = 17
_SPLMM_SPARSE_REML_MAX_ITER = 20
_SPLMM_SPGRM_SINGLE_BLOCK_N_MAX = 2048


def _current_bed_memory_mb() -> float:
    try:
        mb = float(os.environ.get("JX_BED_BLOCK_TARGET_MB", "512"))
    except Exception:
        return 512.0
    return mb if np.isfinite(mb) and mb > 0.0 else 512.0
_ALGWAS_STAGE1_EBIC_GAMMA_DEFAULT = 0.5


def _gwas_use_rust_unified_v1() -> bool:
    """
    Toggle packed GWAS unified Rust dispatcher (v1).

    Default enabled when Rust symbol exists; set
    JX_GWAS_USE_RUST_UNIFIED_V1=0 to disable quickly.
    """
    raw = str(os.environ.get("JX_GWAS_USE_RUST_UNIFIED_V1", "")).strip().lower()
    enabled = raw not in {"0", "false", "no", "off"}
    return bool(enabled and hasattr(jxrs, "gwas_packed_unified_to_tsv"))


def _algwas_stage1_ebic_gamma() -> float:
    raw = str(os.environ.get("JX_ALGWAS_STAGE1_EBIC_GAMMA", "")).strip()
    try:
        val = float(raw)
    except Exception:
        return float(_ALGWAS_STAGE1_EBIC_GAMMA_DEFAULT)
    if np.isfinite(val) and val >= 0.0:
        return float(val)
    return float(_ALGWAS_STAGE1_EBIC_GAMMA_DEFAULT)


def _algwas_stage1_select_criterion_label() -> str:
    raw = str(os.environ.get("JX_ALGWAS_STAGE1_SELECT", "")).strip().lower()
    return "EBIC" if raw == "ebic" else "BIC"


def _log_algwas_stage1_model_selection(
    logger: logging.Logger,
    summary_tsv: str,
) -> None:
    try:
        if not os.path.exists(summary_tsv):
            return
        tab = pd.read_csv(summary_tsv, sep="\t")
        if int(tab.shape[0]) <= 0:
            return
        for col in (
            "step",
            "lambda",
            "rss",
            "nnz",
            "bic",
            "ebic",
            "selected_by_bic",
            "selected_by_ebic",
            "selected_by_model",
        ):
            if col not in tab.columns:
                return

        model_rows = tab.loc[pd.to_numeric(tab["selected_by_model"], errors="coerce").fillna(0).astype(int) == 1]
        bic_rows = tab.loc[pd.to_numeric(tab["selected_by_bic"], errors="coerce").fillna(0).astype(int) == 1]
        ebic_rows = tab.loc[pd.to_numeric(tab["selected_by_ebic"], errors="coerce").fillna(0).astype(int) == 1]
        best_row = model_rows.iloc[0] if int(model_rows.shape[0]) > 0 else tab.iloc[int(pd.to_numeric(tab["bic"], errors="coerce").astype(float).idxmin())]
        best_bic_row = bic_rows.iloc[0] if int(bic_rows.shape[0]) > 0 else tab.iloc[int(pd.to_numeric(tab["bic"], errors="coerce").astype(float).idxmin())]
        best_ebic_row = ebic_rows.iloc[0] if int(ebic_rows.shape[0]) > 0 else tab.iloc[int(pd.to_numeric(tab["ebic"], errors="coerce").astype(float).idxmin())]

        criterion = _algwas_stage1_select_criterion_label()
        ebic_gamma = _algwas_stage1_ebic_gamma()
        msg = (
            "ALGWAS stage1 model selection: "
            f"criterion={criterion}, "
            f"best_step={int(best_row['step'])}, "
            f"lambda={float(best_row['lambda']):.6e}, "
            f"qtn={int(best_row['nnz'])}, "
            f"rss={float(best_row['rss']):.6e}, "
            f"bic={float(best_row['bic']):.6e}, "
            f"ebic(gamma={ebic_gamma:.2f})={float(best_row['ebic']):.6e}; "
            f"bic_best_step={int(best_bic_row['step'])}, "
            f"bic_best_qtn={int(best_bic_row['nnz'])}, "
            f"ebic_best_step={int(best_ebic_row['step'])}, "
            f"ebic_best_qtn={int(best_ebic_row['nnz'])}, "
            f"summary={_display_path(str(summary_tsv))}"
        )
        _log_file_only(logger, logging.INFO, msg)
    except Exception:
        return


def _splmm_sparse_meta_path(sparse_path: str) -> str:
    return f"{sparse_path}.meta.json"


def _splmm_normalize_sparse_grm_path(path_or_prefix: str) -> str:
    raw = str(path_or_prefix).strip()
    if raw == "":
        return raw
    if raw.lower().endswith(".spgrm") or raw.lower().endswith(".jxgrm"):
        return raw
    spgrm = f"{raw}.spgrm"
    jxgrm = f"{raw}.jxgrm"
    if os.path.exists(jxgrm) and not os.path.exists(spgrm):
        return jxgrm
    return spgrm


def _splmm_sparse_sample_hash(sample_indices: Union[np.ndarray, None]) -> str:
    if sample_indices is None:
        return "all"
    arr = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1), dtype=np.int64)
    h = hashlib.blake2b(digest_size=12)
    h.update(arr.tobytes(order="C"))
    h.update(str(int(arr.shape[0])).encode("ascii"))
    return h.hexdigest()


def _splmm_sparse_out_prefix(prefix: str, sample_indices: Union[np.ndarray, None]) -> str:
    if sample_indices is None:
        return str(prefix)
    arr = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1), dtype=np.int64)
    return f"{prefix}.splmm.n{int(arr.shape[0])}.{_splmm_sparse_sample_hash(arr)}"


def _splmm_sparse_out_prefix_for_gwas(
    prefix: str,
    sample_indices: Union[np.ndarray, None],
    *,
    outprefix: Optional[str] = None,
    dense_grm_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    dense_path = str(dense_grm_path).strip() if dense_grm_path is not None else ""
    if dense_path != "" and dense_path.lower().endswith(".npy"):
        desired = dense_path[: -len(".npy")]
    else:
        desired = _splmm_sparse_out_prefix(str(prefix), sample_indices)
    desired_dir = os.path.abspath(os.path.dirname(str(desired)) or ".")
    if _is_writable_dir(desired_dir):
        return str(desired)

    cache_dir = None
    if outprefix is not None and str(outprefix).strip() != "":
        cache_dir = os.path.abspath(os.path.dirname(str(outprefix)))
    fallback_root = _resolve_gwas_cache_dir(
        str(prefix),
        cache_dir=cache_dir,
        logger=logger,
    )
    desired_base = os.path.basename(str(desired).rstrip("/\\"))
    if desired_base == "":
        desired_base = os.path.basename(str(prefix).rstrip("/\\")) or "sparse_grm"
    return os.path.join(fallback_root, desired_base).replace("\\", "/")


def _splmm_sparse_layout_rebuild_reason(sparse_path: str) -> Optional[str]:
    try:
        file_size = int(os.path.getsize(sparse_path))
        if file_size < 16:
            return "file too short"
        with open(sparse_path, "rb") as fh:
            header = fh.read(16)
        if len(header) != 16:
            return "incomplete header"
        n_samples = int.from_bytes(header[0:8], "little", signed=False)
        nnz = int.from_bytes(header[8:16], "little", signed=False)
        col_ptr_bytes = (n_samples + 1) * 8
        row_bytes = nnz * 4
        values_bytes = nnz * 8
        values_offset_legacy = 16 + col_ptr_bytes + row_bytes
        values_offset_padded = (values_offset_legacy + 7) & ~7
        expected_legacy = values_offset_legacy + values_bytes
        expected_padded = values_offset_padded + values_bytes
        if file_size == expected_padded:
            return None
        if file_size == expected_legacy:
            if values_offset_legacy % 8 != 0:
                return "legacy unpadded values layout"
            return None
        return "header/layout mismatch"
    except Exception as exc:
        return f"layout probe failed: {exc}"


def _splmm_format_stage_times(stage_times: dict[str, float]) -> str:
    parts: list[str] = []
    for stage, secs in stage_times.items():
        secs_f = float(secs)
        if np.isfinite(secs_f) and secs_f > 0.0:
            parts.append(f"{stage}={format_elapsed(secs_f)}")
    return ", ".join(parts)


def _splmm_sparse_spec(
    *,
    cutoff: float,
    abs_threshold: bool,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    snps_only: bool,
    method: int = 1,
) -> dict[str, object]:
    return {
        "cutoff": float(cutoff),
        "abs_threshold": bool(abs_threshold),
        "maf_threshold": float(maf_threshold),
        "max_missing_rate": float(max_missing_rate),
        "het_threshold": float(het_threshold),
        "snps_only": bool(snps_only),
        "method": int(method),
    }


def _splmm_parse_sparse_cutoff(splmm_source: Optional[str]) -> tuple[float, Optional[str]]:
    if splmm_source is None:
        return float(_SPLMM_SPARSE_GRM_CUTOFF), None
    raw = str(splmm_source).strip()
    if raw in {"", "__SELF__"}:
        return float(_SPLMM_SPARSE_GRM_CUTOFF), None
    try:
        cutoff = float(raw)
    except Exception:
        return float(_SPLMM_SPARSE_GRM_CUTOFF), raw
    if (not np.isfinite(cutoff)) or cutoff < 0.0:
        raise ValueError(f"SparseLMM sparse cutoff must be a finite value >= 0, got {raw}")
    return float(cutoff), None


def _ensure_splmm_sparse_grm(
    prefix: str,
    *,
    sample_indices: Union[np.ndarray, None] = None,
    out_prefix: Optional[str] = None,
    dense_grm_path: Optional[str] = None,
    cutoff: float,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    snps_only: bool,
    threads: int,
    logger: logging.Logger,
    use_spinner: bool,
    method: int = 1,
) -> str:
    threads_use = int(max(1, int(threads) if int(threads) > 0 else detect_effective_threads()))
    sample_idx_arg = None
    if sample_indices is not None:
        sample_idx_arg = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1), dtype=np.int64)
        if int(sample_idx_arg.shape[0]) == 0:
            raise ValueError("SparseLMM sparse GRM sample subset must not be empty.")
    dense_grm_path_use = None
    if dense_grm_path is not None:
        dense_grm_path_use = str(dense_grm_path).strip()
        if dense_grm_path_use == "":
            dense_grm_path_use = None
    out_prefix_use = str(out_prefix) if out_prefix is not None else _splmm_sparse_out_prefix(str(prefix), sample_idx_arg)
    sparse_path = _splmm_normalize_sparse_grm_path(out_prefix_use)
    meta_path = _splmm_sparse_meta_path(sparse_path)
    spec = _splmm_sparse_spec(
        cutoff=float(cutoff),
        abs_threshold=False,
        maf_threshold=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
        het_threshold=float(het_threshold),
        snps_only=bool(snps_only),
        method=method,
    )
    spec["sample_hash"] = _splmm_sparse_sample_hash(sample_idx_arg)
    spec["sample_n"] = int(sample_idx_arg.shape[0]) if sample_idx_arg is not None else None
    if dense_grm_path_use is not None and dense_grm_path_use.lower().endswith(".npy"):
        spec["source"] = "dense_grm_npy"
        spec["dense_grm_path"] = os.path.normpath(dense_grm_path_use)
    else:
        spec["source"] = "bed"
    if os.path.exists(sparse_path):
        existing_spec = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    existing_spec = json.load(fh)
            except Exception:
                existing_spec = None
        layout_reason = _splmm_sparse_layout_rebuild_reason(sparse_path)
        if existing_spec == spec and layout_reason is None:
            src = os.path.basename(str(sparse_path))
            with CliStatus(
                f"Loading sparse GRM from {src}...",
                enabled=bool(use_spinner),
            ) as task:
                task.complete(f"Loading sparse GRM from {src} ...Finished")
            _log_file_only(
                logger,
                logging.INFO,
                f"SparseLMM sparse GRM reused: {_display_path(sparse_path)}",
            )
            return sparse_path
        if layout_reason is not None:
            reason = layout_reason
        else:
            reason = "missing metadata" if existing_spec is None else "cutoff or filter settings changed"
        _emit_warning_line(
            logger,
            f"Rebuilding sparse GRM for SparseLMM: {os.path.basename(str(sparse_path))} ({reason})",
            use_spinner=bool(use_spinner),
        )
        for path_rm in (sparse_path, meta_path):
            try:
                os.remove(path_rm)
            except OSError:
                pass
    have_dense_npy_sparse = hasattr(jxrs, "spgrm_dense_npy_to_jxgrm")
    have_bed_sparse = hasattr(jxrs, "spgrm_bed_to_jxgrm")
    if (not have_dense_npy_sparse) and (not have_bed_sparse):
        raise RuntimeError(
            "Rust extension is missing sparse GRM builders required by SparseLMM. "
            "Rebuild/install JanusX extension first."
        )

    use_dense_extract = bool(
        dense_grm_path_use is not None
        and sample_idx_arg is None
        and dense_grm_path_use.lower().endswith(".npy")
        and os.path.exists(dense_grm_path_use)
        and have_dense_npy_sparse
    )
    progress_desc = (
        "Calculating sparse GRM from dense GRM"
        if use_dense_extract
        else f"Calculating sparse GRM from {os.path.basename(str(prefix))}..."
    )
    pbar: Optional[_ProgressAdapter] = None
    spin_handle = _start_indeterminate_progress_bar(progress_desc) if bool(use_spinner) else None
    last_done = 0
    last_total = 1

    def _progress_cb(done: int, total: int) -> None:
        nonlocal pbar, spin_handle, last_done, last_total
        d = max(0, int(done))
        t = max(1, int(total))
        if spin_handle is not None:
            try:
                _stop_indeterminate_progress_bar(spin_handle)
            except Exception:
                pass
            spin_handle = None
        if pbar is None:
            pbar = _ProgressAdapter(
                total=t,
                desc=progress_desc,
                force_animate=True,
                logger=logger,
            )
            last_total = t
        if t != last_total:
            pbar.set_total(t)
            last_total = t
        d = min(d, t)
        if d > last_done:
            pbar.update(d - last_done)
            last_done = d
        try:
            pbar.set_postfix(memory=f"{psutil.Process().memory_info().rss / 1024**3:.2f} GB")
        except Exception:
            pass

    build_ok = False
    build_result = None
    build_t0 = time.monotonic()
    try:
        if use_dense_extract:
            build_result = jxrs.spgrm_dense_npy_to_jxgrm(
                str(dense_grm_path_use),
                out_prefix=str(out_prefix_use),
                threshold=float(cutoff),
                abs_threshold=False,
                progress_callback=_progress_cb,
                progress_every=1,
            )
        else:
            if not have_bed_sparse:
                raise RuntimeError(
                    "Rust extension missing spgrm_bed_to_jxgrm required to build sparse GRM from genotype. "
                    "Rebuild/install JanusX extension first."
                )
            build_result = jxrs.spgrm_bed_to_jxgrm(
                str(prefix),
                out_prefix=str(out_prefix_use),
                sample_indices=sample_idx_arg,
                method=int(method),
                threshold=float(cutoff),
                abs_threshold=False,
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                het_threshold=float(het_threshold),
                snps_only=bool(snps_only),
                block_rows=0,
                sample_block=0,
                threads=int(threads_use),
                progress_callback=_progress_cb,
                progress_every=1,
            )
        build_ok = True
    finally:
        if spin_handle is not None:
            try:
                _stop_indeterminate_progress_bar(spin_handle)
            except Exception:
                pass
            spin_handle = None
        if pbar is not None:
            try:
                if build_ok and last_done < last_total:
                    pbar.update(last_total - last_done)
                if build_ok:
                    pbar.finish()
            finally:
                pbar.close(show_done=False)
    built_path = sparse_path
    built_n_samples = None
    built_nnz = None
    if build_result is not None:
        try:
            if isinstance(build_result, (tuple, list)) and len(build_result) >= 1:
                built_path_candidate = _splmm_normalize_sparse_grm_path(str(build_result[0]))
                if built_path_candidate:
                    built_path = built_path_candidate
            if isinstance(build_result, (tuple, list)) and len(build_result) >= 3:
                built_n_samples = int(build_result[1])
                built_nnz = int(build_result[2])
        except Exception:
            built_path = sparse_path
            built_n_samples = None
            built_nnz = None
    sparse_path = built_path
    meta_path = _splmm_sparse_meta_path(sparse_path)
    if not os.path.exists(sparse_path):
        wait_deadline = time.monotonic() + 2.0
        while time.monotonic() < wait_deadline and not os.path.exists(sparse_path):
            time.sleep(0.05)
    if not os.path.exists(sparse_path):
        raise RuntimeError(
            "Sparse GRM build returned but output file is missing: "
            f"expected {_display_path(sparse_path)} from out_prefix {_display_path(str(out_prefix_use))}"
        )
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=True, sort_keys=True)
    build_secs = max(time.monotonic() - build_t0, 0.0)
    built_shape = []
    if built_n_samples is not None:
        built_shape.append(f"n={built_n_samples}")
    elif sample_idx_arg is not None:
        built_shape.append(f"n={int(sample_idx_arg.shape[0])}")
    else:
        built_shape.append("n=all")
    if built_nnz is not None:
        built_shape.append(f"nnz={built_nnz}")
    _log_file_only(
        logger,
        logging.INFO,
        (
            f"SparseLMM sparse GRM ready: {_display_path(sparse_path)} "
            f"({', '.join(built_shape)}, "
            f"source={'dense_grm_npy' if use_dense_extract else 'bed'}, "
            f"method={'standardized' if int(method) == 2 else 'centered'}, "
            f"BLAS=1, Rayon={int(threads_use)}, "
            f"sample_block=auto, "
            f"time={format_elapsed(build_secs)})"
        ),
    )
    if bool(use_spinner):
        prefix_name = os.path.basename(str(prefix))
        _rich_success(
            logger,
            f"Calculating sparse GRM from {prefix_name} (n={built_n_samples or 'all'}) "
            f"...Finished [{format_elapsed(build_secs)}]",
            use_spinner=True,
        )
    return sparse_path


def _splmm_bed_logic_meta_selected(
    prefix: str,
    *,
    sample_indices: np.ndarray,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    snps_only: bool,
    mmap_window_mb: Optional[int] = None,
) -> dict[str, object]:
    if not hasattr(jxrs, "prepare_bed_logic_meta_selected"):
        raise RuntimeError(
            "Rust extension missing prepare_bed_logic_meta_selected required by SparseLMM memmap path."
        )
    row_idx, miss, af, row_flip, site_keep, n_samples_full, n_snps_total = jxrs.prepare_bed_logic_meta_selected(
        str(prefix),
        sample_indices=np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64), dtype=np.int64),
        maf_threshold=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
        het_threshold=float(het_threshold),
        snps_only=bool(snps_only),
        mmap_window_mb=(None if mmap_window_mb is None else int(max(1, int(mmap_window_mb)))),
    )
    return {
        "row_indices": np.ascontiguousarray(np.asarray(row_idx, dtype=np.int64), dtype=np.int64),
        "missing_rate": np.ascontiguousarray(np.asarray(miss, dtype=np.float32), dtype=np.float32),
        "af": np.ascontiguousarray(np.asarray(af, dtype=np.float32), dtype=np.float32),
        "maf": np.ascontiguousarray(np.asarray(af, dtype=np.float32), dtype=np.float32),
        "row_flip": np.ascontiguousarray(np.asarray(row_flip, dtype=np.bool_), dtype=np.bool_),
        "site_keep": np.ascontiguousarray(np.asarray(site_keep, dtype=np.bool_), dtype=np.bool_),
        "n_samples_full": int(n_samples_full),
        "n_snps_total": int(n_snps_total),
    }


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
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    packed = np.asarray(packed_ctx["packed"], dtype=np.uint8)
    if packed.ndim != 2:
        raise ValueError("Packed GWAS context requires packed ndim=2.")
    if not packed.flags.c_contiguous:
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
    packed_n = int(packed_ctx["n_samples"])
    miss_full = np.ascontiguousarray(
        np.asarray(packed_ctx["missing_rate"], dtype=np.float32).reshape(-1),
        dtype=np.float32,
    )
    af_full = np.ascontiguousarray(
        np.asarray(packed_ctx.get("af", packed_ctx["maf"]), dtype=np.float32).reshape(-1),
        dtype=np.float32,
    )
    row_flip_raw = packed_ctx.get("row_flip", None)
    if row_flip_raw is None:
        row_flip_full = np.ascontiguousarray(
            np.zeros((int(packed.shape[0]),), dtype=np.bool_),
            dtype=np.bool_,
        )
        packed_ctx["row_flip"] = row_flip_full
    else:
        row_flip_full = np.ascontiguousarray(
            np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
    packed_rows = int(packed.shape[0])

    def _resolve_active_row_idx() -> np.ndarray:
        active_row_idx_obj = packed_ctx.get("active_row_idx")
        if active_row_idx_obj is None:
            active_row_idx_obj = packed_ctx.get("row_indices")
        if active_row_idx_obj is not None:
            return np.ascontiguousarray(
                np.asarray(active_row_idx_obj, dtype=np.int64).reshape(-1),
                dtype=np.int64,
            )
        site_keep_raw = packed_ctx.get("site_keep", None)
        if site_keep_raw is None:
            return np.ascontiguousarray(
                np.arange(packed_rows, dtype=np.int64),
                dtype=np.int64,
            )
        site_keep = np.ascontiguousarray(
            np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        return np.ascontiguousarray(
            np.flatnonzero(site_keep).astype(np.int64, copy=False),
            dtype=np.int64,
        )

    def _validate_active_row_idx(active_row_idx: np.ndarray, label: str) -> None:
        if int(active_row_idx.shape[0]) == 0:
            return
        if int(active_row_idx.min()) < 0 or int(active_row_idx.max()) >= packed_rows:
            raise ValueError(f"Packed GWAS {label} context mismatch: active_row_idx out of packed bounds.")

    active_row_idx = _resolve_active_row_idx()
    active_stats_shape = (
        int(active_row_idx.shape[0]) == int(miss_full.shape[0])
        and int(active_row_idx.shape[0]) == int(af_full.shape[0])
        and int(active_row_idx.shape[0]) == int(row_flip_full.shape[0])
    )
    packed_mode = str(packed_ctx.get("packed_filter_mode", "compact")).strip().lower()
    if packed_mode == "lazy_full":
        if (
            int(miss_full.shape[0]) == packed_rows
            and int(af_full.shape[0]) == packed_rows
            and int(row_flip_full.shape[0]) == packed_rows
        ):
            _validate_active_row_idx(active_row_idx, "lazy-full")
            maf_active = np.ascontiguousarray(af_full[active_row_idx], dtype=np.float32)
            miss_active = np.ascontiguousarray(miss_full[active_row_idx], dtype=np.float32)
            row_flip_active = np.ascontiguousarray(row_flip_full[active_row_idx], dtype=np.bool_)
            return packed, packed_n, active_row_idx, maf_active, miss_active, row_flip_active
        if active_stats_shape:
            # FarmCPU may borrow the full packed BED together with already-filtered
            # per-row statistics. In that case active_row_idx carries the source-row
            # mapping and the statistics are already aligned to it.
            _validate_active_row_idx(active_row_idx, "lazy-indexed")
            return packed, packed_n, active_row_idx, af_full, miss_full, row_flip_full
        if int(miss_full.shape[0]) != packed_rows:
            raise ValueError("Packed GWAS lazy-full context mismatch: missing_rate length != packed rows.")
        if int(af_full.shape[0]) != packed_rows:
            raise ValueError("Packed GWAS lazy-full context mismatch: af length != packed rows.")
        if int(row_flip_full.shape[0]) != packed_rows:
            raise ValueError("Packed GWAS lazy-full context mismatch: row_flip length != packed rows.")
        _validate_active_row_idx(active_row_idx, "lazy-full")
        maf_active = np.ascontiguousarray(af_full[active_row_idx], dtype=np.float32)
        miss_active = np.ascontiguousarray(miss_full[active_row_idx], dtype=np.float32)
        row_flip_active = np.ascontiguousarray(row_flip_full[active_row_idx], dtype=np.bool_)
        return packed, packed_n, active_row_idx, maf_active, miss_active, row_flip_active

    if int(miss_full.shape[0]) != int(packed.shape[0]):
        raise ValueError("Packed GWAS compact context mismatch: missing_rate length != packed rows.")
    if int(af_full.shape[0]) != int(packed.shape[0]):
        raise ValueError("Packed GWAS compact context mismatch: af length != packed rows.")
    if int(row_flip_full.shape[0]) != int(packed.shape[0]):
        raise ValueError("Packed GWAS compact context mismatch: row_flip length != packed rows.")
    local_row_idx = np.ascontiguousarray(
        np.arange(int(packed.shape[0]), dtype=np.int64),
        dtype=np.int64,
    )
    return packed, packed_n, local_row_idx, af_full, miss_full, row_flip_full


def _resolve_packed_sites_all(
    prefix: str,
    packed_row_idx: np.ndarray,
    sites_all: Union[_BimSiteColumns, dict[str, object], None],
) -> _BimSiteColumns:
    meta_obj = sites_all
    if meta_obj is None:
        meta_obj = _make_deferred_bim_metadata(
            str(prefix),
            packed_row_idx,
            n_markers=int(packed_row_idx.shape[0]),
        )
    return _resolve_bim_site_columns_meta(
        meta_obj,
        prefix=str(prefix),
        row_indices=packed_row_idx,
        expected_rows=int(packed_row_idx.shape[0]),
    )


def _prepare_packed_bed_once_for_gwas(
    *,
    genofile: str,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    snps_only: bool,
    use_spinner: bool,
    preloaded_packed: Union[dict[str, object], None] = None,
    load_site_meta: bool = False,
) -> tuple[str, np.ndarray, dict[str, object], Union[_BimSiteColumns, None]]:
    """
    Resolve packed BED context for GWAS full-rust routes.

    Returns
    -------
    prefix, full_ids, packed_ctx, site_meta_or_none
    """
    prefix = _as_plink_prefix(genofile)
    if prefix is None:
        raise ValueError(
            "Full-rust packed path requires PLINK BED input "
            "(prefix with .bed/.bim/.fam)."
        )

    if packed_preload_is_disabled(preloaded_packed):
        reason = packed_preload_reason(preloaded_packed)
        raise RuntimeError(
            "Packed BED preload is disabled for this run. "
            f"reason={reason or 'unknown'}"
        )

    pre = preloaded_packed if packed_preload_is_ready(preloaded_packed) else None
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
            "af": np.ascontiguousarray(
                np.asarray(packed_ctx_obj.get("af", packed_ctx_obj["maf"]), dtype=np.float32).reshape(-1),
                dtype=np.float32,
            ),
            "maf": np.ascontiguousarray(
                np.asarray(packed_ctx_obj.get("af", packed_ctx_obj["maf"]), dtype=np.float32).reshape(-1),
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
        sites_pre = pre.get("site_meta", pre.get("sites_all"))
        site_meta = _coerce_bim_site_columns(sites_pre)
        if load_site_meta and site_meta is not None and len(site_meta) > 0:
            return str(prefix), full_ids, packed_ctx, site_meta
        if not load_site_meta:
            return str(prefix), full_ids, packed_ctx, None
    else:
        with CliStatus("Loading packed BED genotype...", enabled=bool(use_spinner)) as task:
            try:
                full_ids, packed_ctx = prepare_packed_ctx_from_plink(
                    str(prefix),
                    maf=float(maf_threshold),
                    missing_rate=float(max_missing_rate),
                    het_threshold=float(het_threshold),
                    snps_only=bool(snps_only),
                    filter_mode="lazy_owned",
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
            "af": np.ascontiguousarray(
                np.asarray(packed_ctx.get("af", packed_ctx["maf"]), dtype=np.float32).reshape(-1),
                dtype=np.float32,
            ),
            "maf": np.ascontiguousarray(
                np.asarray(packed_ctx.get("af", packed_ctx["maf"]), dtype=np.float32).reshape(-1),
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

    active_row_idx = np.ascontiguousarray(
        np.asarray(
            packed_ctx.get(
                "active_row_idx",
                np.arange(int(np.asarray(packed_ctx["packed"]).shape[0]), dtype=np.int64),
            ),
            dtype=np.int64,
        ).reshape(-1),
        dtype=np.int64,
    )
    if not load_site_meta:
        return str(prefix), full_ids, packed_ctx, None
    site_meta = _resolve_bim_site_columns_meta(
        _make_deferred_bim_metadata(
            str(prefix),
            active_row_idx,
            n_markers=int(active_row_idx.shape[0]),
        ),
        prefix=str(prefix),
        row_indices=active_row_idx,
        expected_rows=int(active_row_idx.shape[0]),
    )
    return str(prefix), full_ids, packed_ctx, site_meta


def _read_bed_fam_iids(prefix: str) -> np.ndarray:
    fam_path = f"{prefix}.fam"
    ids: list[str] = []
    with open(fam_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split()
            if len(parts) >= 2:
                ids.append(str(parts[1]))
    return np.asarray(ids, dtype=str)


def _iter_bim_active_rows(prefix: str, active_row_idx: np.ndarray):
    row_idx = np.asarray(active_row_idx, dtype=np.int64).reshape(-1)
    if int(row_idx.shape[0]) == 0:
        return
    ptr = 0
    target = int(row_idx[ptr])
    with open(f"{prefix}.bim", "r", encoding="utf-8", errors="replace") as fh:
        for source_idx, line in enumerate(fh):
            if source_idx < target:
                continue
            while source_idx == target:
                parts = line.rstrip("\n").split()
                if len(parts) < 6:
                    raise ValueError(f"{prefix}.bim line {source_idx + 1} is malformed.")
                yield (
                    ptr,
                    source_idx,
                    str(parts[0]),
                    int(float(parts[3])),
                    str(parts[1]),
                    str(parts[4]),
                    str(parts[5]),
                )
                ptr += 1
                if ptr >= int(row_idx.shape[0]):
                    return
                target = int(row_idx[ptr])
                if source_idx != target:
                    break


def _read_qtn_rows(qtn_tsv: str) -> pd.DataFrame:
    if not qtn_tsv or not os.path.exists(qtn_tsv):
        return pd.DataFrame(columns=["chrom", "pos", "snp"])
    df = pd.read_csv(qtn_tsv, sep="\t")
    need = {"chrom", "pos", "snp"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"QTN TSV lacks required columns {sorted(need)}: {qtn_tsv}")
    keep_cols = ["chrom", "pos", "snp"]
    for opt_col in ("allele0", "allele1", "source_row"):
        if opt_col in df.columns:
            keep_cols.append(opt_col)
    out = df.loc[:, keep_cols].copy()
    out["chrom"] = out["chrom"].astype(str)
    out["pos"] = out["pos"].astype(np.int64)
    out["snp"] = out["snp"].astype(str)
    if "allele0" in out.columns:
        out["allele0"] = out["allele0"].astype(str)
    if "allele1" in out.columns:
        out["allele1"] = out["allele1"].astype(str)
    if "source_row" in out.columns:
        out["source_row"] = pd.to_numeric(out["source_row"], errors="coerce").astype("Int64")
    return out


def _selected_qtn_active_indices(
    *,
    qtn_prefix: str,
    qtn_active_row_idx: np.ndarray,
    qtn_rows: pd.DataFrame,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if qtn_rows.empty:
        return np.zeros((0,), dtype=np.int64), []
    if "source_row" in qtn_rows.columns:
        source_row_series = qtn_rows["source_row"]
        if bool(source_row_series.notna().all()):
            active_src = np.asarray(qtn_active_row_idx, dtype=np.int64).reshape(-1)
            src = np.asarray(source_row_series, dtype=np.int64).reshape(-1)
            loc = np.searchsorted(active_src, src)
            valid = (loc >= 0) & (loc < int(active_src.shape[0]))
            if np.any(valid):
                matched = np.zeros_like(valid, dtype=np.bool_)
                matched[valid] = active_src[loc[valid]] == src[valid]
                valid = matched
            if bool(np.all(valid)):
                active = np.ascontiguousarray(loc.astype(np.int64, copy=False), dtype=np.int64)
                meta: list[dict[str, object]] = []
                for row in qtn_rows.itertuples(index=False):
                    meta.append(
                        {
                            "chrom": str(row.chrom),
                            "pos": int(row.pos),
                            "snp": str(row.snp),
                            "allele0": str(getattr(row, "allele0", "")),
                            "allele1": str(getattr(row, "allele1", "")),
                        }
                    )
                return active, meta
    snp_to_order: dict[str, list[int]] = {}
    pos_to_order: dict[tuple[str, int], list[int]] = {}
    for i, row in qtn_rows.iterrows():
        snp_to_order.setdefault(str(row["snp"]), []).append(int(i))
        pos_to_order.setdefault((str(row["chrom"]), int(row["pos"])), []).append(int(i))
    selected: dict[int, tuple[int, dict[str, object]]] = {}
    for active_i, _source_i, chrom, pos, snp, a0, a1 in _iter_bim_active_rows(qtn_prefix, qtn_active_row_idx):
        orders = snp_to_order.get(str(snp), [])
        if len(orders) == 0:
            orders = pos_to_order.get((str(chrom), int(pos)), [])
        for order in orders:
            selected.setdefault(
                int(order),
                (
                    int(active_i),
                    {
                        "chrom": str(chrom),
                        "pos": int(pos),
                        "snp": str(snp),
                        "allele0": str(a0),
                        "allele1": str(a1),
                    },
                ),
            )
    active: list[int] = []
    meta: list[dict[str, object]] = []
    for order in range(int(qtn_rows.shape[0])):
        hit = selected.get(order)
        if hit is None:
            continue
        active.append(int(hit[0]))
        meta.append(hit[1])
    return np.asarray(active, dtype=np.int64), meta


def _decode_qtn_covariates(
    *,
    qtn_preloaded_packed: dict[str, object],
    qtn_active_indices: np.ndarray,
    trait_ids: np.ndarray,
) -> np.ndarray:
    active_sel = np.asarray(qtn_active_indices, dtype=np.int64).reshape(-1)
    if int(active_sel.shape[0]) == 0:
        return np.zeros((int(np.asarray(trait_ids).shape[0]), 0), dtype=np.float64)
    if not hasattr(jxrs, "bed_packed_decode_rows_f32"):
        raise RuntimeError("Rust extension missing bed_packed_decode_rows_f32 for QTN covariate decode.")
    qtn_full_ids = np.asarray(qtn_preloaded_packed["full_ids"], dtype=str).reshape(-1)
    qtn_id_map = {str(sid): i for i, sid in enumerate(qtn_full_ids)}
    try:
        sample_idx = np.asarray([qtn_id_map[str(sid)] for sid in np.asarray(trait_ids, dtype=str)], dtype=np.int64)
    except KeyError as e:
        raise ValueError("Some trait sample IDs are not present in QTN genotype sample order.") from e
    packed, packed_n, active_row_idx, maf, _miss, row_flip = _packed_ctx_active_view_for_gwas(
        qtn_preloaded_packed["packed_ctx"]  # type: ignore[arg-type]
    )
    if int(active_sel.min()) < 0 or int(active_sel.max()) >= int(active_row_idx.shape[0]):
        raise ValueError("Selected QTN active index out of bounds.")
    source_rows = np.ascontiguousarray(active_row_idx[active_sel], dtype=np.int64)
    flip_sel = np.ascontiguousarray(row_flip[active_sel], dtype=np.bool_)
    maf_sel = np.ascontiguousarray(maf[active_sel], dtype=np.float32)
    geno = np.asarray(
        jxrs.bed_packed_decode_rows_f32(
            packed,
            int(packed_n),
            source_rows,
            flip_sel,
            maf_sel,
            sample_indices=np.ascontiguousarray(sample_idx, dtype=np.int64),
        ),
        dtype=np.float32,
    )
    return np.ascontiguousarray(geno.T, dtype=np.float64)


def _farmcpu_final_window_bp_from_args(args_obj: object) -> int:
    raw_env = str(os.environ.get("JX_FARMCPU_FINAL_WINDOW_BP", "")).strip()
    if raw_env:
        try:
            val = int(float(raw_env))
        except Exception:
            val = -1
        if val >= 0:
            return int(val)
    vals: list[int] = []
    for x in getattr(args_obj, "farmcpu_bin_size", [5e5, 5e6, 5e7]):
        try:
            v = int(float(x))
        except Exception:
            continue
        if v > 0:
            vals.append(v)
    return int(min(vals) if vals else 500_000)


def _qtn_windows_from_meta(qtn_meta: list[dict[str, object]], qtn_cov: np.ndarray, window_bp: int) -> list[dict[str, object]]:
    k = len(qtn_meta)
    if k == 0 or int(window_bp) < 0:
        return []
    parent = list(range(k))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    starts = [int(m["pos"]) - int(window_bp) for m in qtn_meta]
    ends = [int(m["pos"]) + int(window_bp) for m in qtn_meta]
    for i in range(k):
        for j in range(i + 1, k):
            if str(qtn_meta[i]["chrom"]) == str(qtn_meta[j]["chrom"]) and starts[i] <= ends[j] and starts[j] <= ends[i]:
                union(i, j)
    try:
        ld_thr = float(os.environ.get("JX_FARMCPU_FINAL_LD_MERGE_R2", "0.5"))
    except Exception:
        ld_thr = 0.5
    if k > 1 and np.isfinite(ld_thr) and ld_thr > 0.0 and int(qtn_cov.shape[1]) == k:
        corr = np.corrcoef(np.asarray(qtn_cov, dtype=np.float64), rowvar=False)
        for i in range(k):
            for j in range(i + 1, k):
                if str(qtn_meta[i]["chrom"]) == str(qtn_meta[j]["chrom"]):
                    r = corr[i, j]
                    if np.isfinite(r) and float(r * r) >= float(ld_thr):
                        union(i, j)
    grouped: dict[int, dict[str, object]] = {}
    for i in range(k):
        root = find(i)
        item = grouped.setdefault(
            root,
            {"chrom": str(qtn_meta[i]["chrom"]), "start": starts[i], "end": ends[i], "qtn_positions": []},
        )
        item["start"] = min(int(item["start"]), starts[i])
        item["end"] = max(int(item["end"]), ends[i])
        cast_list = item["qtn_positions"]
        assert isinstance(cast_list, list)
        cast_list.append(i)
    windows = sorted(grouped.values(), key=lambda w: (str(w["chrom"]), int(w["start"]), int(w["end"])))
    merged: list[dict[str, object]] = []
    for w in windows:
        if merged and str(merged[-1]["chrom"]) == str(w["chrom"]) and int(w["start"]) <= int(merged[-1]["end"]):
            merged[-1]["end"] = max(int(merged[-1]["end"]), int(w["end"]))
            merged[-1]["qtn_positions"] = sorted(set(int(x) for x in list(merged[-1]["qtn_positions"]) + list(w["qtn_positions"])))
        else:
            w["qtn_positions"] = sorted(set(int(x) for x in list(w["qtn_positions"])))
            merged.append(w)
    return merged


def _active_segments_for_windows(
    *,
    main_prefix: str,
    active_row_idx: np.ndarray,
    windows: list[dict[str, object]],
) -> list[tuple[int, int, int]]:
    if len(windows) == 0:
        return []
    by_chrom: dict[str, dict[str, list[int]]] = {}
    for win_idx, w in enumerate(windows):
        chrom_w = str(w["chrom"])
        by_chrom.setdefault(chrom_w, {"starts": [], "ends": [], "ctx": []})
        by_chrom[chrom_w]["starts"].append(int(w["start"]))
        by_chrom[chrom_w]["ends"].append(int(w["end"]))
        by_chrom[chrom_w]["ctx"].append(int(win_idx) + 1)
    for chrom_w, rec in by_chrom.items():
        order = sorted(range(len(rec["starts"])), key=lambda i: (rec["starts"][i], rec["ends"][i]))
        by_chrom[chrom_w] = {
            "starts": [rec["starts"][i] for i in order],
            "ends": [rec["ends"][i] for i in order],
            "ctx": [rec["ctx"][i] for i in order],
        }

    bounds: dict[int, list[int]] = {}
    for active_i, _source_i, chrom, pos, _snp, _a0, _a1 in _iter_bim_active_rows(main_prefix, active_row_idx):
        rec = by_chrom.get(str(chrom))
        if rec is None:
            continue
        p = bisect_right(rec["starts"], int(pos)) - 1
        if p < 0 or int(pos) > int(rec["ends"][p]):
            continue
        ctx_id = int(rec["ctx"][p])
        cur = bounds.get(ctx_id)
        if cur is None:
            bounds[ctx_id] = [int(active_i), int(active_i) + 1]
        else:
            cur[1] = int(active_i) + 1

    window_ranges = [(start, end, ctx_id) for ctx_id, (start, end) in bounds.items() if start < end]
    window_ranges.sort(key=lambda x: (x[0], x[1]))
    segments: list[tuple[int, int, int]] = []
    cursor = 0
    for start, end, ctx_id in window_ranges:
        if start > cursor:
            segments.append((cursor, start, 0))
        start_use = max(start, cursor)
        if end > start_use:
            segments.append((start_use, end, ctx_id))
        cursor = max(cursor, end)
    n_active = int(np.asarray(active_row_idx).shape[0])
    if cursor < n_active:
        segments.append((cursor, n_active, 0))
    return segments


def _plink_bed_n_snps_from_prefix(prefix: str, n_samples: int) -> int:
    bytes_per_snp = (int(n_samples) + 3) // 4
    if bytes_per_snp <= 0:
        raise ValueError("Invalid PLINK sample count while estimating BED SNP count.")
    bed_path = f"{str(prefix)}.bed"
    bed_size = int(os.path.getsize(bed_path))
    if bed_size < 3:
        raise ValueError(f"BED file is too small: {bed_path}")
    payload = bed_size - 3
    if payload % bytes_per_snp != 0:
        raise ValueError(
            f"BED payload size is not divisible by bytes_per_snp: {bed_path}"
        )
    return int(payload // bytes_per_snp)


def run_qtn_segmented_lm_stage2(
    *,
    main_prefix: str,
    y_vec: np.ndarray,
    trait_ids: np.ndarray,
    base_cov: np.ndarray,
    qtn_preloaded_packed: dict[str, object],
    qtn_tsv: str,
    tmp_tsv: str,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    snps_only: bool,
    chunk_size: int,
    threads: int,
    progress_callback,
    args_obj: object,
    logger: Optional[logging.Logger] = None,
    use_spinner: bool = False,
    status_prefix: Optional[str] = None,
) -> tuple[int, int, int]:
    if not hasattr(jxrs, "lm_stream_bed_segments_compact_to_tsv"):
        raise RuntimeError("Rust extension missing compact stage2 LM streaming function; rebuild JanusX.")
    stage_status_enabled = status_prefix is not None
    stage_label = str(status_prefix).strip() if status_prefix else "QTN stage2"
    status_tasks: dict[str, CliStatus] = {}
    terminal_status_enabled = bool(stage_status_enabled and use_spinner)

    def _status_begin(desc: str) -> Optional[CliStatus]:
        if not terminal_status_enabled:
            return None
        task = CliStatus(f"{desc} ...", enabled=bool(use_spinner))
        task.__enter__()
        return task

    def _status_complete(task: Optional[CliStatus], message: str) -> None:
        if task is not None:
            task.complete(message)
        if logger is not None:
            _log_file_only(logger, logging.INFO, message)

    def _status_fail(task: Optional[CliStatus], message: str) -> None:
        if task is not None:
            task.fail(message)
        if logger is not None:
            _log_file_only(logger, logging.ERROR, message)

    def _log_stage2_summary(message: str) -> None:
        if logger is not None:
            _log_file_only(logger, logging.INFO, f"{stage_label} summary: {message}")

    qtn_desc = f"Preparing {status_prefix} QTN covariates" if status_prefix else ""
    qtn_task = _status_begin(qtn_desc)
    try:
        qtn_rows = _read_qtn_rows(qtn_tsv)
        if qtn_rows.empty:
            raise ValueError(f"{stage_label} requires at least one pseudo-QTN row, but {qtn_tsv} is empty.")
        _qtn_packed, _qtn_n, qtn_active_row_idx, _qtn_maf, _qtn_miss, _qtn_flip = _packed_ctx_active_view_for_gwas(
            qtn_preloaded_packed["packed_ctx"]  # type: ignore[arg-type]
        )
        qtn_active_sel, qtn_meta = _selected_qtn_active_indices(
            qtn_prefix=str(qtn_preloaded_packed["prefix"]),
            qtn_active_row_idx=qtn_active_row_idx,
            qtn_rows=qtn_rows,
        )
        if int(qtn_active_sel.shape[0]) == 0:
            raise ValueError(
                f"{stage_label} could not map pseudo-QTN rows from {qtn_tsv} back to the packed genotype."
            )
        qtn_cov = _decode_qtn_covariates(
            qtn_preloaded_packed=qtn_preloaded_packed,
            qtn_active_indices=qtn_active_sel,
            trait_ids=np.asarray(trait_ids, dtype=str),
        )
        if int(qtn_cov.shape[1]) != int(len(qtn_meta)):
            raise ValueError(
                f"{stage_label} decoded QTN covariates shape mismatch: qtn_cov={int(qtn_cov.shape[1])}, meta={len(qtn_meta)}"
            )
        base_design = np.ascontiguousarray(
            np.concatenate(
                [
                    np.ones((int(y_vec.shape[0]), 1), dtype=np.float64),
                    np.asarray(base_cov, dtype=np.float64),
                ],
                axis=1,
            ),
            dtype=np.float64,
        )
        windows = _qtn_windows_from_meta(qtn_meta, qtn_cov, _farmcpu_final_window_bp_from_args(args_obj))
        context_exclude_qtn: list[list[int]] = [[]]
        for w in windows:
            qpos = sorted(set(int(x) for x in list(w.get("qtn_positions", []))))
            context_exclude_qtn.append(qpos)
        base_context_count = 1
        exclude_qtn_context_count = int(max(0, len(context_exclude_qtn) - base_context_count))
        total_context_count = int(base_context_count + exclude_qtn_context_count)
        _status_complete(
            qtn_task,
            (
                f"{qtn_desc} ...Finished (n={int(qtn_cov.shape[0])}, "
                f"baseCov={int(np.asarray(base_cov).shape[1])}, "
                f"nQTN={int(qtn_cov.shape[1])}, windows={len(windows)})"
            ),
        )
    except Exception:
        _status_fail(qtn_task, f"{qtn_desc} ...Failed")
        raise

    index_desc = f"Indexing {status_prefix} scan markers" if status_prefix else ""
    index_task = _status_begin(index_desc)
    try:
        main_ids = _read_bed_fam_iids(main_prefix)
        id_map = {str(sid): i for i, sid in enumerate(main_ids)}
        try:
            sample_idx = np.asarray([id_map[str(sid)] for sid in np.asarray(trait_ids, dtype=str)], dtype=np.int64)
        except KeyError as e:
            raise ValueError("Some trait sample IDs are not present in main BED sample order.") from e
        n_snps_total = _plink_bed_n_snps_from_prefix(str(main_prefix), len(main_ids))
        stage2_mmap_window_mb = _common_resolve_decode_mmap_window_mb(
            str(main_prefix),
            int(len(main_ids)),
            int(n_snps_total),
            _current_bed_memory_mb(),
            needs_copy=False,
        )
        segments: list[tuple[int, int, int]] = []
        source_windows = [
            (str(w["chrom"]), int(w["start"]), int(w["end"]), int(idx) + 1)
            for idx, w in enumerate(windows)
        ]
        segment_count = int(len(source_windows))
        progress_denominator = int(max(1, n_snps_total))
        index_done = (
            f"{index_desc} ...Finished (total={int(n_snps_total)}, "
            f"windows={len(source_windows)}, segments={segment_count}, mode=source-window)"
        )
        _status_complete(
            index_task,
            index_done,
        )
    except Exception:
        _status_fail(index_task, f"{index_desc} ...Failed")
        raise
    y_stage2 = np.ascontiguousarray(np.asarray(y_vec, dtype=np.float64).reshape(-1), dtype=np.float64)
    sample_ids_stage2 = [str(x) for x in np.asarray(trait_ids, dtype=str)]
    progress_every = int(max(1, min(int(max(1, int(chunk_size))), _progress_callback_step(int(progress_denominator)))))
    mmap_window_arg = None if stage2_mmap_window_mb is None else int(max(1, int(stage2_mmap_window_mb)))

    def _compact_setup_status(phase: str, state: str, value_a: int, value_b: int) -> None:
        if not stage_status_enabled:
            return
        phase_s = str(phase)
        state_s = str(state)
        if state_s == "start":
            status_tasks[phase_s] = _status_begin(phase_s)  # type: ignore[assignment]
            return
        task = status_tasks.pop(phase_s, None)
        if state_s == "finish":
            _status_complete(
                task,
                f"{phase_s} ...Finished (contexts={int(value_a)}, q={int(value_b)})",
            )
        elif state_s == "fail":
            _status_fail(task, f"{phase_s} ...Failed")

    try:
        kept_rows, scan_rows = jxrs.lm_stream_bed_segments_compact_to_tsv(
            str(main_prefix),
            y_stage2,
            base_design,
            np.ascontiguousarray(np.asarray(qtn_cov, dtype=np.float64), dtype=np.float64),
            context_exclude_qtn,
            segments,
            str(tmp_tsv),
            sample_ids=sample_ids_stage2,
            maf_threshold=float(maf_threshold),
            max_missing_rate=float(max_missing_rate),
            het_threshold=float(het_threshold),
            snps_only=bool(snps_only),
            chunk_size=int(max(1, int(chunk_size))),
            threads=int(max(1, int(threads))),
            progress_callback=progress_callback,
            progress_every=progress_every,
            mmap_window_mb=mmap_window_arg,
            setup_callback=_compact_setup_status,
            source_windows=source_windows,
        )
    except Exception:
        task = status_tasks.pop("Building compact QR contexts", None)
        if task is not None:
            _status_fail(task, "Building compact QR contexts ...Failed")
        raise
    _log_stage2_summary(
        "windows="
        f"{len(windows)}, segments={segment_count}, base_cov_cols={int(np.asarray(base_cov).shape[1])}, "
        f"qtn_used={int(qtn_cov.shape[1])}, kept_rows={int(kept_rows)}, "
        f"scan_rows={int(scan_rows)}, base_contexts={base_context_count}, "
        f"exclude_qtn_contexts={exclude_qtn_context_count}, total_contexts={total_context_count}, "
        "mode=compact-source-window"
    )
    return int(kept_rows), int(scan_rows), int(qtn_cov.shape[1])


def _packed_lowrank_scan_to_tsv(
    *,
    mod: object,
    packed: np.ndarray,
    packed_n: int,
    packed_row_idx: np.ndarray,
    row_flip: np.ndarray,
    maf: np.ndarray,
    miss: np.ndarray,
    sites_all: _BimSiteColumns,
    sample_idx_trait: np.ndarray,
    out_tsv: str,
    chunk_size: int,
    threads: int,
    genetic_model: str,
    progress_callback=None,
) -> int:
    if not hasattr(jxrs, "bed_packed_decode_rows_f32"):
        raise RuntimeError(
            "Rust extension missing bed_packed_decode_rows_f32 required by packed low-rank GWAS scan."
        )
    if not hasattr(jxrs, "GwasAssocTsvWriter") or not hasattr(jxrs, "SiteInfo"):
        raise RuntimeError(
            "Rust extension missing GwasAssocTsvWriter/SiteInfo required by packed low-rank GWAS scan."
        )

    total = int(len(sites_all))
    if total != int(packed_row_idx.shape[0]):
        raise ValueError(
            f"packed low-rank scan metadata mismatch: sites={total}, active_rows={packed_row_idx.shape[0]}"
        )
    sample_idx_arr = np.ascontiguousarray(
        np.asarray(sample_idx_trait, dtype=np.int64).reshape(-1),
        dtype=np.int64,
    )
    writer = jxrs.GwasAssocTsvWriter(str(out_tsv), genetic_model=str(genetic_model))
    step = int(max(1, chunk_size))
    done = 0
    try:
        for start in range(0, total, step):
            end = min(start + step, total)
            src_rows = np.ascontiguousarray(
                np.asarray(packed_row_idx[start:end], dtype=np.int64).reshape(-1),
                dtype=np.int64,
            )
            packed_subset = np.ascontiguousarray(
                np.asarray(packed[src_rows, :], dtype=np.uint8),
                dtype=np.uint8,
            )
            maf_blk = np.ascontiguousarray(
                np.asarray(maf[start:end], dtype=np.float32).reshape(-1),
                dtype=np.float32,
            )
            miss_blk = np.ascontiguousarray(
                np.asarray(miss[start:end], dtype=np.float32).reshape(-1),
                dtype=np.float32,
            )
            flip_blk = np.ascontiguousarray(
                np.asarray(row_flip[start:end], dtype=np.bool_).reshape(-1),
                dtype=np.bool_,
            )
            local_rows = np.ascontiguousarray(
                np.arange(int(end - start), dtype=np.int64),
                dtype=np.int64,
            )
            geno_blk = np.ascontiguousarray(
                np.asarray(
                    jxrs.bed_packed_decode_rows_f32(
                        packed_subset,
                        int(packed_n),
                        local_rows,
                        flip_blk,
                        maf_blk,
                        sample_indices=sample_idx_arr,
                    ),
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
            res_blk = np.ascontiguousarray(
                np.asarray(mod.gwas(geno_blk, threads=max(1, int(threads))), dtype=np.float64),
                dtype=np.float64,
            )
            site_objs: list[object] = []
            snp_blk: list[str] = []
            for site in sites_all[start:end]:
                c, p, a0, a1, sid = _site_tuple_parts(site)
                site_objs.append(jxrs.SiteInfo(str(c), int(p), str(sid), str(a0), str(a1)))
                snp_blk.append(str(sid))
            writer.write_chunk(site_objs, snp_blk, maf_blk, miss_blk, res_blk)
            done = int(end)
            if progress_callback is not None:
                progress_callback(done, total)
        writer.flush()
        return int(done)
    finally:
        try:
            writer.close()
        except Exception:
            pass


def _packed_preload_ready_state(
    *,
    prefix: str,
    full_ids: np.ndarray,
    packed_ctx: dict[str, object],
    sites_all: Union[_BimSiteColumns, None],
) -> dict[str, object]:
    out = {
        "prefix": str(prefix),
        "full_ids": np.asarray(full_ids, dtype=str),
        "packed_ctx": packed_ctx,
    }
    if sites_all is not None:
        out["site_meta"] = sites_all
        out["sites_all"] = sites_all
    return out


def _splmm_null_components_valid(sigma_g2: float, sigma_e2: float) -> bool:
    return bool(
        np.isfinite(float(sigma_g2))
        and np.isfinite(float(sigma_e2))
        and float(sigma_g2) >= 0.0
        and float(sigma_e2) >= 0.0
        and (float(sigma_g2) > 0.0 or float(sigma_e2) > 0.0)
    )


def _splmm_ols_residualize(y_vec: np.ndarray, x_cov: Union[np.ndarray, None]) -> np.ndarray:
    y = np.ascontiguousarray(np.asarray(y_vec, dtype=np.float64).reshape(-1), dtype=np.float64)
    n = int(y.shape[0])
    if n == 0:
        return y
    if x_cov is None:
        x_design = np.ones((n, 1), dtype=np.float64)
    else:
        x = np.ascontiguousarray(np.asarray(x_cov, dtype=np.float64), dtype=np.float64)
        if x.ndim != 2 or int(x.shape[0]) != n:
            raise ValueError(
                f"SparseLMM covariate shape mismatch for residualization: got {x.shape}, expected ({n}, p)."
            )
        x_design = np.ascontiguousarray(
            np.concatenate([np.ones((n, 1), dtype=np.float64), x], axis=1),
            dtype=np.float64,
        )
    beta, *_rest = np.linalg.lstsq(x_design, y, rcond=None)
    resid = y - x_design @ beta
    return np.ascontiguousarray(resid, dtype=np.float64)


def _splmm_component_ratio(sigma_g2: float, sigma_e2: float) -> float:
    denom = float(sigma_g2) + float(sigma_e2)
    if np.isfinite(denom) and denom > 0.0:
        return float(float(sigma_g2) / denom)
    return float("nan")


def _splmm_sparse_grm_diag_stats(
    jxgrm_path: str,
    sample_idx: Union[np.ndarray, None] = None,
) -> dict[str, float]:
    if not hasattr(jxrs, "splmm_sparse_grm_diag_stats"):
        return {
            "mean_diag": float("nan"),
            "min_diag": float("nan"),
            "max_diag": float("nan"),
            "n_samples": float("nan"),
            "nnz": float("nan"),
        }
    sample_idx_arg = None
    if sample_idx is not None:
        sample_idx_arg = np.ascontiguousarray(
            np.asarray(sample_idx, dtype=np.int64).reshape(-1),
            dtype=np.int64,
        )
    mean_diag, min_diag, max_diag, n_samples, nnz = jxrs.splmm_sparse_grm_diag_stats(
        str(jxgrm_path),
        sample_indices=sample_idx_arg,
    )
    return {
        "mean_diag": float(mean_diag),
        "min_diag": float(min_diag),
        "max_diag": float(max_diag),
        "n_samples": float(n_samples),
        "nnz": float(nnz),
    }


def _splmm_load_sparse_grm_subset_dense(
    jxgrm_path: str,
    sample_idx: Union[np.ndarray, None] = None,
) -> np.ndarray:
    if not hasattr(jxrs, "splmm_load_sparse_grm_subset_dense"):
        raise RuntimeError(
            "Rust extension missing splmm_load_sparse_grm_subset_dense required by SparseLMM verification helpers."
        )
    sample_idx_arg = None
    if sample_idx is not None:
        sample_idx_arg = np.ascontiguousarray(
            np.asarray(sample_idx, dtype=np.int64).reshape(-1),
            dtype=np.int64,
        )
    out = jxrs.splmm_load_sparse_grm_subset_dense(
        str(jxgrm_path),
        sample_indices=sample_idx_arg,
    )
    arr = np.ascontiguousarray(np.asarray(out, dtype=np.float64), dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[0]) != int(arr.shape[1]):
        raise ValueError(f"SparseLMM dense sparse-GRM subset must be square, got shape={arr.shape}")
    return arr


def _splmm_sparse_lambda_boundary_flag(
    log10_lambda: float,
    low: float,
    high: float,
) -> Optional[str]:
    x = float(log10_lambda)
    low_f = float(low)
    high_f = float(high)
    if (not np.isfinite(x)) or (not np.isfinite(low_f)) or (not np.isfinite(high_f)) or low_f >= high_f:
        return None
    eps = max(1e-3, 0.01 * abs(high_f - low_f))
    if x <= low_f + eps:
        return "low"
    if x >= high_f - eps:
        return "high"
    return None


def _splmm_sparse_grid_local_summary(null_fit: dict, window: int = 2) -> Optional[str]:
    try:
        grid_log10 = np.asarray(null_fit.get("grid_log10", []), dtype=np.float64).reshape(-1)
        grid_reml = np.asarray(null_fit.get("grid_reml", []), dtype=np.float64).reshape(-1)
        best_log10 = float(null_fit.get("log10_lambda", float("nan")))
    except Exception:
        return None
    if grid_log10.size == 0 or grid_log10.size != grid_reml.size or not np.isfinite(best_log10):
        return None
    best_idx = int(np.argmin(np.abs(grid_log10 - best_log10)))
    if best_idx < 0 or best_idx >= int(grid_log10.size):
        return None
    lo = max(0, best_idx - max(1, int(window)))
    hi = min(int(grid_log10.size), best_idx + max(1, int(window)) + 1)
    parts: list[str] = []
    for idx in range(lo, hi):
        x = float(grid_log10[idx])
        y = float(grid_reml[idx])
        star = "*" if idx == best_idx else ""
        if np.isfinite(x) and np.isfinite(y):
            parts.append(f"{x:.2f}:{y:.6g}{star}")
        elif np.isfinite(x):
            parts.append(f"{x:.2f}:nan{star}")
        else:
            parts.append(f"nan:{y:.6g}{star}" if np.isfinite(y) else f"nan:nan{star}")
    if not parts:
        return None
    return "[" + ", ".join(parts) + "]"


def _plink_fam_sample_ids(prefix: str) -> np.ndarray:
    fam_path = f"{prefix}.fam"
    fam = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        engine="python",
        usecols=[1],
        dtype=str,
    )
    return np.asarray(fam.iloc[:, 0].astype(str), dtype=str)


def _write_splmm_assoc_tsv(
    *,
    out_tsv: str,
    sites_all: _BimSiteColumns,
    maf: np.ndarray,
    miss: np.ndarray,
    results: np.ndarray,
    genetic_model: str,
    chunk_rows: int = 131072,
) -> int:
    if not hasattr(jxrs, "GwasAssocTsvWriter") or not hasattr(jxrs, "SiteInfo"):
        raise RuntimeError(
            "Rust extension missing GwasAssocTsvWriter/SiteInfo required by SparseLMM TSV writer."
        )
    n_rows = int(len(sites_all))
    maf_arr = np.ascontiguousarray(np.asarray(maf, dtype=np.float32).reshape(-1), dtype=np.float32)
    miss_arr = np.ascontiguousarray(np.asarray(miss, dtype=np.float32).reshape(-1), dtype=np.float32)
    res_arr = np.ascontiguousarray(np.asarray(results, dtype=np.float64), dtype=np.float64)
    if maf_arr.shape[0] != n_rows or miss_arr.shape[0] != n_rows:
        raise ValueError(
            f"SparseLMM TSV writer metadata mismatch: sites={n_rows}, af={maf_arr.shape[0]}, miss={miss_arr.shape[0]}"
        )
    if res_arr.shape != (n_rows, 3):
        raise ValueError(
            f"SparseLMM TSV writer result shape mismatch: got {res_arr.shape}, expected ({n_rows}, 3)."
        )
    gm = str(genetic_model).strip().lower()

    def _alleles(a0: str, a1: str) -> tuple[str, str]:
        if gm == "dom":
            return f"{a0}{a0}", f"{a0}{a1}/{a1}{a1}"
        if gm == "rec":
            return f"{a0}{a1}/{a0}{a0}", f"{a1}{a1}"
        if gm == "het":
            return f"{a0}{a0}/{a1}{a1}", f"{a0}{a1}"
        return str(a0), str(a1)

    step = int(max(1, chunk_rows))
    written = 0
    with open(out_tsv, "w", encoding="utf-8", newline="") as fh:
        fh.write("chrom\tpos\tsnp\tallele0\tallele1\taf\tmiss\tbeta\tse\tchisq\tpwald\n")
        for start in range(0, n_rows, step):
            end = min(start + step, n_rows)
            chunk_text: list[str] = []
            for off, site in enumerate(sites_all[start:end]):
                chrom, pos, allele0, allele1, snp = _site_tuple_parts(site)
                idx = start + off
                beta = float(res_arr[idx, 0])
                se = float(res_arr[idx, 1])
                pwald = float(res_arr[idx, 2])
                chisq = float("nan")
                if np.isfinite(beta) and np.isfinite(se) and se > 0.0:
                    chisq = float((beta / se) ** 2)
                a0_txt, a1_txt = _alleles(str(allele0), str(allele1))
                chunk_text.append(
                    f"{chrom}\t{int(pos)}\t{snp}\t{a0_txt}\t{a1_txt}\t"
                    f"{float(maf_arr[idx]):.4f}\t{float(miss_arr[idx]):.4f}\t"
                    f"{beta:.4f}\t{se:.4f}\t{chisq:.6g}\t{pwald:.4e}\n"
                )
            fh.write("".join(chunk_text))
            written += end - start
    return int(written)


def _splmm_fastlmm_profile_vc(
    *,
    s_null: np.ndarray,
    u1tx: np.ndarray,
    u2tx: np.ndarray,
    u1ty: np.ndarray,
    u2ty: np.ndarray,
    lbd: float,
) -> tuple[float, float]:
    lbd_f = float(lbd)
    if not np.isfinite(lbd_f) or lbd_f <= 0.0:
        return float("nan"), float("nan")
    s = np.ascontiguousarray(np.asarray(s_null, dtype=np.float64).reshape(-1), dtype=np.float64)
    x1 = np.ascontiguousarray(np.asarray(u1tx, dtype=np.float64), dtype=np.float64)
    x2 = np.ascontiguousarray(np.asarray(u2tx, dtype=np.float64), dtype=np.float64)
    y1 = np.ascontiguousarray(np.asarray(u1ty, dtype=np.float64).reshape(-1), dtype=np.float64)
    y2 = np.ascontiguousarray(np.asarray(u2ty, dtype=np.float64).reshape(-1), dtype=np.float64)
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("SparseLMM FastLMM profile VC expects 2D u1tx/u2tx.")
    k = int(s.shape[0])
    if x1.shape[0] != k or y1.shape[0] != k:
        raise ValueError(
            f"SparseLMM FastLMM profile VC spectral shape mismatch: len(s)={k}, u1tx={x1.shape}, u1ty={y1.shape}"
        )
    n2, p = int(x2.shape[0]), int(x2.shape[1])
    if int(x1.shape[1]) != p or y2.shape[0] != n2:
        raise ValueError(
            f"SparseLMM FastLMM profile VC residual shape mismatch: u1tx={x1.shape}, u2tx={x2.shape}, u2ty={y2.shape}"
        )
    n_minus_p = n2 - p
    if n_minus_p <= 0:
        return float("nan"), float("nan")

    v1_inv = 1.0 / np.maximum(s + lbd_f, 1e-30)
    xtv_inv_x = (x1.T * v1_inv) @ x1 + (x2.T @ x2) / lbd_f
    xtv_inv_y = (x1.T * v1_inv) @ y1 + (x2.T @ y2) / lbd_f
    try:
        beta = np.linalg.solve(xtv_inv_x, xtv_inv_y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xtv_inv_x, xtv_inv_y, rcond=None)[0]

    r1 = y1 - x1 @ beta
    r2 = y2 - x2 @ beta
    rtv_invr = float(np.dot(v1_inv, np.square(r1)) + np.dot(r2, r2) / lbd_f)
    if not np.isfinite(rtv_invr) or rtv_invr <= 0.0:
        return float("nan"), float("nan")
    sigma_g2 = float(rtv_invr / float(n_minus_p))
    sigma_e2 = float(lbd_f * sigma_g2)
    return sigma_g2, sigma_e2


def _splmm_exact_profile_vc(
    *,
    eigvals: np.ndarray,
    utx: np.ndarray,
    uty: np.ndarray,
    lbd: float,
) -> tuple[float, float]:
    lbd_f = float(lbd)
    if not np.isfinite(lbd_f) or lbd_f <= 0.0:
        return float("nan"), float("nan")
    s = np.ascontiguousarray(np.maximum(np.asarray(eigvals, dtype=np.float64).reshape(-1), 0.0), dtype=np.float64)
    x = np.ascontiguousarray(np.asarray(utx, dtype=np.float64), dtype=np.float64)
    y = np.ascontiguousarray(np.asarray(uty, dtype=np.float64).reshape(-1), dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("SparseLMM exact profile VC expects 2D utx.")
    n, p = int(x.shape[0]), int(x.shape[1])
    if int(s.shape[0]) != n or int(y.shape[0]) != n:
        raise ValueError(
            f"SparseLMM exact profile VC spectral shape mismatch: len(s)={s.shape[0]}, utx={x.shape}, uty={y.shape}"
        )
    n_minus_p = n - p
    if n_minus_p <= 0:
        return float("nan"), float("nan")

    v_inv = 1.0 / np.maximum(s + lbd_f, 1e-30)
    xtv_inv_x = (x.T * v_inv) @ x
    xtv_inv_y = (x.T * v_inv) @ y
    beta = _splmm_solve_linear(xtv_inv_x, xtv_inv_y)
    resid = y - (x @ beta)
    rtv_invr = float(np.dot(v_inv, np.square(resid)))
    if not np.isfinite(rtv_invr) or rtv_invr <= 0.0:
        return float("nan"), float("nan")
    sigma_g2 = float(rtv_invr / float(n_minus_p))
    sigma_e2 = float(lbd_f * sigma_g2)
    return sigma_g2, sigma_e2


def _splmm_exact_reml_objective(
    log10_lambda: float,
    *,
    eigvals: np.ndarray,
    utx: np.ndarray,
    uty: np.ndarray,
) -> float:
    lbd_f = float(10.0 ** float(log10_lambda))
    if (not np.isfinite(lbd_f)) or lbd_f <= 0.0:
        return float("inf")
    s = np.ascontiguousarray(np.asarray(eigvals, dtype=np.float64).reshape(-1), dtype=np.float64)
    x = np.ascontiguousarray(np.asarray(utx, dtype=np.float64), dtype=np.float64)
    y = np.ascontiguousarray(np.asarray(uty, dtype=np.float64).reshape(-1), dtype=np.float64)
    n, p = int(x.shape[0]), int(x.shape[1])
    n_minus_p = n - p
    if n_minus_p <= 0:
        return float("inf")
    v = np.maximum(s + lbd_f, 1e-30)
    v_inv = 1.0 / v
    xtv_inv_x = (x.T * v_inv) @ x
    sign_xt, logdet_xt = np.linalg.slogdet(xtv_inv_x)
    if sign_xt <= 0 or (not np.isfinite(logdet_xt)):
        return float("inf")
    xtv_inv_y = (x.T * v_inv) @ y
    beta = _splmm_solve_linear(xtv_inv_x, xtv_inv_y)
    resid = y - (x @ beta)
    rtv_invr = float(np.dot(v_inv, np.square(resid)))
    if (not np.isfinite(rtv_invr)) or rtv_invr <= 0.0:
        return float("inf")
    sigma_g2 = float(rtv_invr / float(n_minus_p))
    if (not np.isfinite(sigma_g2)) or sigma_g2 <= 0.0:
        return float("inf")
    logdet_v = float(np.sum(np.log(v)))
    return float(logdet_v + logdet_xt + float(n_minus_p) * np.log(sigma_g2))


def _splmm_design_with_intercept(
    x_cov: Union[np.ndarray, None],
    n: int,
) -> np.ndarray:
    if x_cov is None:
        return np.ones((n, 1), dtype=np.float64)
    x = np.ascontiguousarray(np.asarray(x_cov, dtype=np.float64), dtype=np.float64)
    if x.ndim != 2 or int(x.shape[0]) != n:
        raise ValueError(f"SparseLMM covariate shape mismatch: got {x.shape}, expected ({n}, p).")
    return np.ascontiguousarray(
        np.concatenate([np.ones((n, 1), dtype=np.float64), x], axis=1),
        dtype=np.float64,
    )


def _splmm_component_scale_reml(
    *,
    y_vec: np.ndarray,
    x_cov: Union[np.ndarray, None],
    sigma_g2: float,
    sigma_e2: float,
) -> dict[str, float]:
    y = np.ascontiguousarray(np.asarray(y_vec, dtype=np.float64).reshape(-1), dtype=np.float64)
    n = int(y.shape[0])
    x_design = _splmm_design_with_intercept(x_cov, n)
    p = int(x_design.shape[1])
    sigma_sum = float(sigma_g2) + float(sigma_e2)
    if n <= p:
        return {
            "rss_reml": float("nan"),
            "rss_mx_reml": float("nan"),
            "df_reml": float("nan"),
            "resid_var_reml": float("nan"),
            "resid_var_mx_reml": float("nan"),
            "ypy_reml": float("nan"),
            "sigma_sum_profile": sigma_sum,
            "sigma2_profile_reml": float("nan"),
            "sigma2_mx_reml": float("nan"),
            "sigma_scale_p0_to_mx": float("nan"),
            "sigma_scale_reml": float("nan"),
        }
    xtx = np.ascontiguousarray(x_design.T @ x_design, dtype=np.float64)
    xty = np.ascontiguousarray(x_design.T @ y, dtype=np.float64)
    beta = _splmm_solve_linear(xtx, xty)
    resid = np.ascontiguousarray(y - (x_design @ beta), dtype=np.float64)
    rss_mx = float(np.dot(resid, resid))
    df = int(n - p)
    # These are diagnostic scale summaries only. The Rust sparse scan paths now
    # consume lambda directly and reconstruct any scan-scale sigma^2 internally
    # on the same K + lambda I chain, so these values are no longer used to
    # rescale scan inputs.
    #
    # Under the explicit P0 + sigma^2 parameterization:
    #   sigma2_profile = y'P0y / (n - p)
    #   sigma2_mx      = y'M_Xy / (n - p)
    # For standardized K, V = sigma^2 (K + lambda I), the residualized approx
    # scan-scale variance is sigma2_scan = sigma2_mx / (1 + lambda).
    ypy_reml = float(sigma_g2) * float(df) if np.isfinite(sigma_g2) and df > 0 else float("nan")
    resid_var_reml = float(sigma_g2) if np.isfinite(sigma_g2) else float("nan")
    resid_var_mx_reml = float(rss_mx / float(df)) if df > 0 else float("nan")
    sigma2_profile_reml = resid_var_reml
    sigma2_mx_reml = resid_var_mx_reml
    sigma_scale_p0_to_mx = (
        float(rss_mx / ypy_reml)
        if np.isfinite(rss_mx) and np.isfinite(ypy_reml) and ypy_reml > 0.0
        else float("nan")
    )
    sigma_scale_reml = (
        float(sigma2_mx_reml / sigma_sum)
        if np.isfinite(sigma2_mx_reml) and np.isfinite(sigma_sum) and sigma_sum > 0.0
        else float("nan")
    )
    return {
        "rss_reml": ypy_reml,
        "rss_mx_reml": rss_mx,
        "df_reml": float(df),
        "resid_var_reml": resid_var_reml,
        "resid_var_mx_reml": resid_var_mx_reml,
        "ypy_reml": ypy_reml,
        "sigma_sum_profile": sigma_sum,
        "sigma2_profile_reml": sigma2_profile_reml,
        "sigma2_mx_reml": sigma2_mx_reml,
        "sigma_scale_p0_to_mx": sigma_scale_p0_to_mx,
        "sigma_scale_reml": sigma_scale_reml,
    }


def _splmm_solve_linear(lhs: np.ndarray, rhs: np.ndarray, *, ridge: float = 1e-10) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(lhs, dtype=np.float64), dtype=np.float64)
    b = np.ascontiguousarray(np.asarray(rhs, dtype=np.float64), dtype=np.float64)
    try:
        return np.ascontiguousarray(np.linalg.solve(a, b), dtype=np.float64)
    except np.linalg.LinAlgError:
        a_reg = np.array(a, dtype=np.float64, copy=True, order="C")
        if a_reg.ndim != 2 or int(a_reg.shape[0]) != int(a_reg.shape[1]):
            return np.ascontiguousarray(np.linalg.lstsq(a_reg, b, rcond=None)[0], dtype=np.float64)
        diag_idx = np.diag_indices_from(a_reg)
        a_reg[diag_idx] = a_reg[diag_idx] + float(max(ridge, 0.0))
        try:
            return np.ascontiguousarray(np.linalg.solve(a_reg, b), dtype=np.float64)
        except np.linalg.LinAlgError:
            return np.ascontiguousarray(np.linalg.lstsq(a_reg, b, rcond=None)[0], dtype=np.float64)


def _splmm_exact_null_fit_from_grm(
    *,
    grm: np.ndarray,
    y_vec: np.ndarray,
    x_cov: Union[np.ndarray, None] = None,
    low: float = -5.0,
    high: float = 5.0,
    threads: Union[int, None] = None,
    stage_label: Optional[str] = None,
    keep_buffers: bool = False,
) -> dict[str, object]:
    del threads
    y = np.ascontiguousarray(np.asarray(y_vec, dtype=np.float64).reshape(-1), dtype=np.float64)
    k = np.ascontiguousarray(np.asarray(grm, dtype=np.float64), dtype=np.float64)
    if k.ndim != 2 or int(k.shape[0]) != int(k.shape[1]):
        raise ValueError(f"SparseLMM exact null fit GRM must be square, got shape={k.shape}")
    if int(k.shape[0]) != int(y.shape[0]):
        raise ValueError(f"SparseLMM exact null fit GRM/y mismatch: grm={k.shape}, y={y.shape}")
    if not np.all(np.isfinite(k)):
        raise ValueError("SparseLMM exact null fit GRM contains NaN/Inf values.")
    k = np.ascontiguousarray((k + k.T) * 0.5, dtype=np.float64)
    n = int(y.shape[0])
    x_design = _splmm_design_with_intercept(x_cov, n)
    p = int(x_design.shape[1])
    if n <= p:
        raise ValueError(f"SparseLMM exact null fit requires n > p, got n={n}, p={p}")

    evals, u = np.linalg.eigh(k)
    evals = np.ascontiguousarray(np.maximum(evals, 0.0), dtype=np.float64)
    utx = np.ascontiguousarray(u.T @ x_design, dtype=np.float64)
    uty = np.ascontiguousarray(u.T @ y, dtype=np.float64)

    best_log10 = float("nan")
    best_reml = float("inf")
    try:
        from scipy.optimize import minimize_scalar  # type: ignore

        opt = minimize_scalar(
            lambda z: _splmm_exact_reml_objective(z, eigvals=evals, utx=utx, uty=uty),
            bounds=(float(low), float(high)),
            method="bounded",
            options={"xatol": 1e-3, "maxiter": 200},
        )
        if bool(getattr(opt, "success", False)) and np.isfinite(float(opt.x)):
            best_log10 = float(opt.x)
            best_reml = float(opt.fun)
    except Exception:
        pass
    if not np.isfinite(best_log10):
        grid = np.linspace(float(low), float(high), 257, dtype=np.float64)
        vals = np.asarray(
            [_splmm_exact_reml_objective(z, eigvals=evals, utx=utx, uty=uty) for z in grid],
            dtype=np.float64,
        )
        idx = int(np.nanargmin(vals))
        best_log10 = float(grid[idx])
        best_reml = float(vals[idx])

    lbd = float(10.0 ** best_log10)
    sigma_g2, sigma_e2 = _splmm_exact_profile_vc(
        eigvals=evals,
        utx=utx,
        uty=uty,
        lbd=lbd,
    )
    out = {
        "strategy": "dense_grm_exact_reml",
        "backend": "dense_eigh",
        "converged": bool(np.isfinite(best_log10) and np.isfinite(sigma_g2) and np.isfinite(sigma_e2)),
        "used_iter": 0,
        "lambda": lbd,
        "log10_lambda": float(best_log10),
        "sigma_g2": float(sigma_g2),
        "sigma_e2": float(sigma_e2),
        "pve": _splmm_component_ratio(sigma_g2, sigma_e2),
        "reml": float(best_reml),
        "ml": float("nan"),
        "lambda_boundary": _splmm_sparse_lambda_boundary_flag(float(best_log10), float(low), float(high)),
    }
    out.update(
        _splmm_component_scale_reml(
            y_vec=y,
            x_cov=x_cov,
            sigma_g2=sigma_g2,
            sigma_e2=sigma_e2,
        )
    )
    if keep_buffers:
        out["eigvals"] = evals
        out["utx"] = utx
        out["uty"] = uty
    if stage_label:
        logging.getLogger(__name__).info(
            "%s: lambda=%.6g sigma_g2=%.6g sigma_e2=%.6g reml=%.6g",
            str(stage_label),
            float(lbd),
            float(sigma_g2),
            float(sigma_e2),
            float(best_reml),
        )
    return out

def _splmm_sparse_null_fit(
    *,
    jxgrm_path: str,
    sample_idx: Union[np.ndarray, None],
    y_vec: np.ndarray,
    x_cov: Union[np.ndarray, None],
    progress_callback=None,
    residualized_approx: bool = False,
    threads: Union[int, None] = None,
) -> dict[str, object]:
    y_arg = np.ascontiguousarray(np.asarray(y_vec, dtype=np.float64).reshape(-1), dtype=np.float64)
    sample_idx_arg = None
    if sample_idx is not None:
        sample_idx_arg = np.ascontiguousarray(np.asarray(sample_idx, dtype=np.int64).reshape(-1), dtype=np.int64)
        if int(sample_idx_arg.shape[0]) != int(y_arg.shape[0]):
            raise ValueError(
                f"SparseLMM sparse null fit sample/y mismatch: sample_idx={sample_idx_arg.shape[0]} y={y_arg.shape[0]}"
            )
    x_arg = None
    if x_cov is not None:
        x_arg = np.ascontiguousarray(np.asarray(x_cov, dtype=np.float64), dtype=np.float64)
        if x_arg.ndim != 2 or int(x_arg.shape[0]) != int(y_arg.shape[0]):
            raise ValueError(
                f"SparseLMM sparse null covariate shape mismatch: got {x_arg.shape}, expected ({y_arg.shape[0]}, p)."
            )

    n_samples_null = int(y_arg.shape[0])
    sparse_diag = _splmm_sparse_grm_diag_stats(str(jxgrm_path), sample_idx_arg)
    mean_diag_k = float(sparse_diag.get("mean_diag", float("nan")))
    n_samples_k = int(sparse_diag.get("n_samples", float("nan")))
    nnz_k = int(sparse_diag.get("nnz", float("nan")))
    offdiag_nnz = max(0, int(nnz_k) - max(0, int(n_samples_k)))
    offdiag_total = max(1, int(n_samples_k) * max(0, int(n_samples_k) - 1) // 2)
    offdiag_density = float(offdiag_nnz / offdiag_total) if int(n_samples_k) > 1 else float("nan")
    threads_use = int(max(1, int(threads) if threads is not None and int(threads) > 0 else detect_effective_threads()))

    if residualized_approx:
        if not hasattr(jxrs, "splmm_residualized_approx_null_fit_from_jxgrm"):
            raise RuntimeError(
                "Rust extension missing splmm_residualized_approx_null_fit_from_jxgrm required by SparseLMM approx residualized null fit."
            )
        fit_t0 = time.monotonic()
        out = jxrs.splmm_residualized_approx_null_fit_from_jxgrm(
            str(jxgrm_path),
            y_arg,
            x_cov=x_arg,
            sample_indices=sample_idx_arg,
            low=-5.0,
            high=5.0,
            grid_size=int(_SPLMM_SPARSE_REML_GRID_SIZE),
            tol=1e-3,
            max_iter=int(_SPLMM_SPARSE_REML_MAX_ITER),
            threads=int(threads_use),
        )
        fit_secs = max(time.monotonic() - fit_t0, 0.0)
        lbd, sigma_g2, sigma_e2, ml, reml, log10_lambda, df_reml = tuple(out)
        sigma_g2 = float(sigma_g2)
        sigma_e2 = float(sigma_e2)
        sigma_sum = float(sigma_g2 + sigma_e2)
        pve = _splmm_component_ratio(sigma_g2, sigma_e2)
        ypy_reml = float(sigma_g2) * float(df_reml) if np.isfinite(sigma_g2) and np.isfinite(df_reml) and df_reml > 0 else float("nan")
        out_dict = {
            "strategy": "residualized_sparse_reml_brent",
            "sigma_g2": sigma_g2,
            "sigma_e2": sigma_e2,
            "pve": pve,
            "lambda": float(lbd),
            "log10_lambda": float(log10_lambda),
            "ml": float(ml),
            "reml": float(reml),
            "converged": True,
            "used_iter": 0,
            "backend": "sparse_cholesky_residualized",
            "fit_secs": float(fit_secs),
            "grid_log10": [],
            "grid_reml": [],
            "grid_sigma_g2": [],
            "grid_sigma_e2": [],
            "mean_diag_k": mean_diag_k,
            "pve_diag_scaled": float("nan"),
            "min_diag_k": float(sparse_diag.get("min_diag", float("nan"))),
            "max_diag_k": float(sparse_diag.get("max_diag", float("nan"))),
            "n_samples_k": float(n_samples_k),
            "nnz_k": float(nnz_k),
            "offdiag_nnz_k": float(offdiag_nnz),
            "offdiag_density_k": float(offdiag_density),
            "lambda_boundary": _splmm_sparse_lambda_boundary_flag(float(log10_lambda), -5.0, 5.0),
            "rss_reml": ypy_reml,
            "rss_mx_reml": float("nan"),
            "df_reml": float(df_reml),
            "resid_var_reml": float(sigma_g2) if np.isfinite(sigma_g2) else float("nan"),
            "resid_var_mx_reml": float("nan"),
            "ypy_reml": ypy_reml,
            "sigma_sum_profile": sigma_sum,
            "sigma2_profile_reml": float(sigma_g2) if np.isfinite(sigma_g2) else float("nan"),
            "sigma2_mx_reml": float("nan"),
            "sigma_scale_p0_to_mx": float("nan"),
            "sigma_scale_reml": float("nan"),
        }
        if progress_callback is not None:
            progress_callback(1, 1)
        return out_dict

    if not hasattr(jxrs, "spreml_sparse_reml_brent_from_jxgrm"):
        raise RuntimeError(
            "Rust extension missing spreml_sparse_reml_brent_from_jxgrm required by SparseLMM sparse null fit."
        )

    fit_t0 = time.monotonic()
    low = -5.0
    high = 5.0
    out = jxrs.spreml_sparse_reml_brent_from_jxgrm(
        str(jxgrm_path),
        y_arg,
        x_cov=x_arg,
        sample_indices=sample_idx_arg,
        low=low,
        high=high,
        grid_size=int(_SPLMM_SPARSE_REML_GRID_SIZE),
        tol=1e-3,
        max_iter=int(_SPLMM_SPARSE_REML_MAX_ITER),
        progress_callback=progress_callback,
    )
    fit_secs = max(time.monotonic() - fit_t0, 0.0)
    (
        lbd,
        sigma_g2,
        sigma_e2,
        ml,
        reml,
        log10_lambda,
        grid_log10,
        grid_reml,
        grid_sigma_g2,
        grid_sigma_e2,
    ) = tuple(out)
    sigma_g2 = float(sigma_g2)
    sigma_e2 = float(sigma_e2)
    # fastGWA-style sparse h2 is interpreted directly from the profiled
    # variance components. Scaling by mean diag(K) can materially overstate h2
    # for thresholded sparse GRMs whose diagonal is not normalized to 1.
    pve = _splmm_component_ratio(sigma_g2, sigma_e2)
    var_g_diag_scaled = float(sigma_g2) * max(mean_diag_k, 0.0) if np.isfinite(mean_diag_k) else float("nan")
    denom_diag_scaled = var_g_diag_scaled + sigma_e2
    pve_diag_scaled = (
        float(var_g_diag_scaled / denom_diag_scaled)
        if np.isfinite(denom_diag_scaled) and denom_diag_scaled > 0.0
        else float("nan")
    )
    lambda_boundary = _splmm_sparse_lambda_boundary_flag(float(log10_lambda), low, high)
    out_dict = {
        "strategy": "sparse_reml_brent",
        "sigma_g2": sigma_g2,
        "sigma_e2": sigma_e2,
        "pve": pve,
        "lambda": float(lbd),
        "log10_lambda": float(log10_lambda),
        "ml": float(ml),
        "reml": float(reml),
        "converged": True,
        "used_iter": 0,
        "backend": "sparse_cholesky",
        "fit_secs": float(fit_secs),
        "grid_log10": list(np.asarray(grid_log10, dtype=np.float64).reshape(-1)),
        "grid_reml": list(np.asarray(grid_reml, dtype=np.float64).reshape(-1)),
        "grid_sigma_g2": list(np.asarray(grid_sigma_g2, dtype=np.float64).reshape(-1)),
        "grid_sigma_e2": list(np.asarray(grid_sigma_e2, dtype=np.float64).reshape(-1)),
        "mean_diag_k": mean_diag_k,
        "pve_diag_scaled": pve_diag_scaled,
        "min_diag_k": float(sparse_diag.get("min_diag", float("nan"))),
        "max_diag_k": float(sparse_diag.get("max_diag", float("nan"))),
        "n_samples_k": float(n_samples_k),
        "nnz_k": float(nnz_k),
        "offdiag_nnz_k": float(offdiag_nnz),
        "offdiag_density_k": float(offdiag_density),
        "lambda_boundary": lambda_boundary,
    }
    out_dict.update(
        _splmm_component_scale_reml(
            y_vec=y_arg,
            x_cov=x_arg,
            sigma_g2=sigma_g2,
            sigma_e2=sigma_e2,
        )
    )
    return out_dict


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
    force_model: bool = False,
    _route_model_key: str = "fastlmm",
    _route_model_label: str = "FastLMM",
    _route_model_cls: object = FastLMM,
    _allow_pve_switch: bool = True,
) -> None:
    required_symbols = [
        "grm_packed_f64_with_stats",
        "rust_eigh_from_array_f64",
        "fastlmm_assoc_packed_f32_to_tsv",
        "fvlmm_assoc_packed_f32_to_tsv",
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
        load_site_meta=False,
    )
    packed_preload_ready = _packed_preload_ready_state(
        prefix=prefix,
        full_ids=full_ids,
        packed_ctx=packed_ctx,
        sites_all=sites_all,
    )
    id_to_idx = {sid: i for i, sid in enumerate(full_ids)}
    try:
        sample_map = np.asarray([id_to_idx[str(sid)] for sid in np.asarray(ids, dtype=str)], dtype=np.int64)
    except KeyError as e:
        raise ValueError("Some aligned sample IDs are not present in packed BED sample order.") from e

    packed, packed_n, packed_row_idx, maf, miss, row_flip = _packed_ctx_active_view_for_gwas(
        packed_ctx
    )
    if packed_n <= 0:
        raise ValueError("Packed BED reported invalid sample count.")
    if (
        int(packed_row_idx.shape[0]) != int(maf.shape[0])
        or int(packed_row_idx.shape[0]) != int(miss.shape[0])
        or int(packed_row_idx.shape[0]) != int(row_flip.shape[0])
    ):
        raise ValueError("Packed BED arrays have inconsistent SNP dimensions.")
    n_sites_active = int(packed_row_idx.shape[0])

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pheno_aligned, ids = _align_pheno_to_sample_order(pheno, ids)
    trait_iter = _resolve_trait_iter(pheno_aligned, trait_names)
    multi_trait_mode = len(trait_iter) > 1

    for trait_idx, pname in enumerate(trait_iter):
        cpu_t0 = process.cpu_times()
        t0 = time.time()
        peak_rss = process.memory_info().rss

        y_full, sameidx = _trait_values_and_mask(pheno_aligned, str(pname))
        keep_idx_nonmissing = np.flatnonzero(sameidx).astype(np.int64, copy=False)
        n_nonmissing = int(keep_idx_nonmissing.shape[0])
        if n_nonmissing == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")
            continue
        keep_idx = keep_idx_nonmissing
        n_idv = int(keep_idx.shape[0])

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
        grm_fit: Optional[np.ndarray] = None
        if grm is None:
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
        evd_t0 = time.monotonic()
        with CliStatus(
            f"Running {_route_model_label} Eigen-Decomposition...",
            enabled=bool(use_spinner),
            use_process=True,
        ) as task:
            try:
                eigvals, eigvecs, evd_backend, evd_secs = _gwas_eigh_from_grm(
                    grm if grm is not None else grm_fit,
                    threads=max(1, int(threads)),
                    logger=logger,
                    stage_label=f"{_route_model_key}-fullrank:{pname}",
                    require_rust=True,
                    diag_ridge=1e-6,
                    subset_idx=(keep_idx if grm is not None else None),
                )
            except Exception:
                task.fail(f"{_route_model_label} Eigen-Decomposition ...Failed")
                raise
        # Keep explicit elapsed in case backend timing is unavailable/zero.
        evd_secs = max(float(evd_secs), max(time.monotonic() - evd_t0, 0.0))
        # LAPACK eigh returns ascending eigenvalues. Reorder descending so the
        # downstream full-rank spectral scan path sees a stable leading-to-trailing order.
        try:
            ord_desc = np.argsort(np.asarray(eigvals, dtype=np.float64))[::-1]
            eigvals = np.asarray(eigvals, dtype=np.float64)[ord_desc]
            eigvecs = np.asarray(eigvecs, dtype=np.float64)[:, ord_desc]
        except Exception:
            eigvals = np.asarray(eigvals, dtype=np.float64)
            eigvecs = np.asarray(eigvecs, dtype=np.float64)
        if eigvecs.shape != (n_idv, n_idv):
            raise ValueError(
                f"Trait eigenvector shape mismatch: got {eigvecs.shape}, expected ({n_idv}, {n_idv})."
            )
        with _gwas_evd_stage_ctx(max(1, int(threads))):
            shared_lmm_mod = LMM.from_spectral(
                y=y_vec,
                X=x_arg,
                eigvals=eigvals,
                eigvecs=eigvecs,
                evd_secs=float(evd_secs),
            )
            null_fast_mod = _route_model_cls.from_lmm(shared_lmm_mod)

        fixed_lbd: Optional[float] = None
        fixed_ml0: Optional[float] = None
        try:
            lbd0_tmp = float(getattr(null_fast_mod, "lbd_null"))
            if np.isfinite(lbd0_tmp) and lbd0_tmp > 0.0:
                fixed_lbd = float(lbd0_tmp)
        except Exception:
            fixed_lbd = None
        try:
            ml0_tmp = float(getattr(null_fast_mod, "ML0"))
            if np.isfinite(ml0_tmp):
                fixed_ml0 = float(ml0_tmp)
        except Exception:
            fixed_ml0 = None
        try:
            pve_tmp = float(getattr(null_fast_mod, "pve"))
            null_pve = pve_tmp if np.isfinite(pve_tmp) else None
        except Exception:
            null_pve = None
        null_ml0: Optional[float] = None

        if (not bool(force_model)) and bool(_allow_pve_switch) and _fastlmm_should_switch_to_lmm(null_pve):
            prev_pve = float(null_pve) if null_pve is not None else float("nan")
            logger.warning(
                f"Warning: {_route_model_label} switch to LMM for trait {pname}: "
                f"null PVE={prev_pve:.4f} (>0.995)."
            )
            _log_model_line(
                logger,
                "LMM",
                (
                    f"switched from {_route_model_label} null to LMM: "
                    f"PVE(null)={prev_pve:.4f} (>0.995) "
                    f"[{format_elapsed(evd_secs)}]"
                ),
                use_spinner=bool(use_spinner),
            )
            run_lmm_packed_fullrank(
                genofile=genofile,
                pheno=pheno,
                ids=ids,
                grm=grm,
                outprefix=outprefix,
                maf_threshold=maf_threshold,
                max_missing_rate=max_missing_rate,
                genetic_model=genetic_model,
                het_threshold=het_threshold,
                chunk_size=chunk_size,
                qmatrix=qmatrix,
                cov_all=cov_all,
                plot=plot,
                threads=threads,
                logger=logger,
                use_spinner=use_spinner,
                snps_only=bool(snps_only),
                eff_snp_by_trait=eff_snp_by_trait,
                summary_rows=summary_rows,
                saved_paths=saved_paths,
                trait_names=[pname],
                emit_trait_header=False,
                preloaded_packed=packed_preload_ready,
                force_model=bool(force_model),
            )
            if multi_trait_mode:
                logger.info("")
            continue

        if (not bool(force_model)) and (null_ml0 is not None):
            switch_to_lm, lrt_stat, lrt_p = _mixed_model_switch_to_lm_decision(
                y_vec=y_vec,
                x_cov=x_cov,
                lmm_ml0=null_ml0,
                alpha=0.05,
            )
            if switch_to_lm:
                logger.warning(
                    f"Warning: {_route_model_label} switch to LM for trait {pname}: "
                    f"null LRT stat={float(lrt_stat):.4g}, p={float(lrt_p):.4g} (>=0.05)."
                )
                _log_model_line(
                    logger,
                    "LM",
                    (
                        f"switched from {_route_model_label} null to LM "
                        f"(LRT stat={float(lrt_stat):.4g}, p={float(lrt_p):.4g}) "
                        f"[{format_elapsed(evd_secs)}]"
                    ),
                    use_spinner=bool(use_spinner),
                )
                run_lm_packed_fullrank(
                    genofile=genofile,
                    pheno=pheno,
                    ids=ids,
                    outprefix=outprefix,
                    maf_threshold=maf_threshold,
                    max_missing_rate=max_missing_rate,
                    genetic_model=genetic_model,
                    het_threshold=het_threshold,
                    chunk_size=chunk_size,
                    qmatrix=qmatrix,
                    cov_all=cov_all,
                    plot=plot,
                    threads=threads,
                    logger=logger,
                    use_spinner=use_spinner,
                    snps_only=bool(snps_only),
                    eff_snp_by_trait=eff_snp_by_trait,
                    summary_rows=summary_rows,
                    saved_paths=saved_paths,
                    trait_names=[pname],
                    emit_trait_header=False,
                    preloaded_packed=packed_preload_ready,
                    force_model=bool(force_model),
                )
                if multi_trait_mode:
                    logger.info("")
                continue

        if bool(getattr(null_fast_mod, "lowrank", False)):
            eigvals = None
            eigvecs = None
            grm_fit = None
            gc.collect()

        # Packed fixed-lambda routes currently expose only scan-stage progress. Keep a
        # placeholder stage-1 handle so shared cleanup code remains safe.
        stage1_pbar: Optional[_ProgressAdapter] = None
        gwas_total = int(n_sites_active)
        gwas_last_done = 0
        gwas_pbar: Optional[_ProgressAdapter] = None
        gwas_pbar = _ProgressAdapter(
            total=max(1, gwas_total),
            desc=_route_model_label,
            force_animate=True,
            logger=logger,
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
                    desc=_route_model_label,
                    force_animate=True,
                    logger=logger,
                )
                gwas_last_done = 0
            d = int(max(0, min(d, max(1, gwas_total))))
            step = int(max(0, d - gwas_last_done))
            if step > 0 and gwas_pbar is not None:
                gwas_pbar.update(step)
                gwas_last_done = d

        gm_tag = str(genetic_model).lower()
        pname_tag = _safe_trait_file_label(pname)
        if gm_tag == "add":
            out_tsv = f"{outprefix}.{pname_tag}.{_route_model_key}.tsv"
        else:
            out_tsv = f"{outprefix}.{pname_tag}.{gm_tag}.{_route_model_key}.tsv"
        tmp_tsv = _gwas_result_tmp_path(out_tsv)

        gwas_ok = False
        lbd = float("nan")
        ml0 = float("nan")
        reml0 = float("nan")
        scan_secs = 0.0
        try:
            scan_t0 = time.monotonic()
            scan_stage_ctx = (
                _gwas_fvlmm_scan_stage_ctx
                if str(_route_model_key).lower() == "fvlmm"
                else _gwas_scan_stage_ctx
            )
            with scan_stage_ctx(max(1, int(threads))):
                if bool(getattr(null_fast_mod, "lowrank", False)):
                    sites_lowrank = _resolve_packed_sites_all(prefix, packed_row_idx, sites_all)
                    _written_rows = _packed_lowrank_scan_to_tsv(
                        mod=null_fast_mod,
                        packed=packed,
                        packed_n=int(packed_n),
                        packed_row_idx=packed_row_idx,
                        row_flip=row_flip,
                        maf=maf,
                        miss=miss,
                        sites_all=sites_lowrank,
                        sample_idx_trait=sample_idx_trait,
                        out_tsv=str(tmp_tsv),
                        chunk_size=int(max(1, int(chunk_size))),
                        threads=int(max(1, int(threads))),
                        genetic_model=str(genetic_model),
                        progress_callback=_fastlmm_progress if gwas_pbar is not None else None,
                        )
                    if int(_written_rows) != int(n_sites_active):
                        _emit_warning_line(
                            logger,
                            f"{_route_model_label} low-rank writer row mismatch: expected={n_sites_active} wrote={int(_written_rows)}",
                            use_spinner=bool(use_spinner),
                        )
                    lbd = float(getattr(null_fast_mod, "lbd_null", np.nan))
                    ml0 = float(getattr(null_fast_mod, "ML0", np.nan))
                    reml0 = float(getattr(null_fast_mod, "LL0", np.nan))
                else:
                    u_sub = np.ascontiguousarray(np.asarray(eigvecs, dtype=np.float32), dtype=np.float32)
                    s_trait = np.ascontiguousarray(np.maximum(eigvals, 0.0), dtype=np.float32)
                    sample_idx_scan_arg: Optional[np.ndarray]
                    if _is_full_identity_index(sample_idx_trait, int(packed_n)):
                        u_trait = u_sub
                        sample_idx_scan_arg = None
                    else:
                        u_trait = np.zeros((int(packed_n), int(n_idv)), dtype=np.float32)
                        u_trait[sample_idx_trait, :] = u_sub
                        sample_idx_scan_arg = sample_idx_trait
                    progress_kwargs: dict[str, object] = {
                        "progress_callback": None,
                        "progress_every": 0,
                    }
                    if gwas_pbar is not None:
                        progress_kwargs = {
                            "progress_callback": _fastlmm_progress,
                            "progress_every": int(
                                max(
                                    1,
                                    min(
                                        int(max(1, min(gwas_total, int(max(1, chunk_size))))),
                                        _progress_callback_step(int(max(1, gwas_total))),
                                    ),
                                )
                            ),
                        }
                    unified_done = False
                    if _gwas_use_rust_unified_v1():
                        try:
                            jobs = [
                                {
                                    "model": _route_model_key,
                                    "trait": str(pname),
                                    "out_tsv": str(tmp_tsv),
                                    "y": y_vec,
                                    "x": x_arg,
                                    "u": u_trait,
                                    "s": s_trait,
                                    "sample_indices": sample_idx_scan_arg,
                                    "low": -5.0,
                                    "high": 5.0,
                                    "max_iter": 50,
                                    "tol": 1e-2,
                                    "tau": 0.0,
                                    "genetic_model": str(genetic_model),
                                    "fixed_lbd": (float(fixed_lbd) if fixed_lbd is not None else None),
                                    "fixed_ml0": None,
                                    "rotate_block_rows": int(max(1, int(chunk_size))),
                                    "scan_progress_callback": _fastlmm_progress,
                                    "progress_every": int(
                                        max(
                                            1,
                                            min(
                                                int(max(1, int(chunk_size))),
                                                _progress_callback_step(int(max(1, gwas_total))),
                                            ),
                                        )
                                    ),
                                }
                            ]
                            _res = jxrs.gwas_packed_unified_to_tsv(
                                jobs,
                                packed,
                                int(packed_n),
                                row_flip,
                                maf,
                                miss,
                                [],
                                [],
                                [],
                                [],
                                [],
                                int(max(1, int(chunk_size))),
                                int(threads),
                                None,
                                int(max(1, int(chunk_size))),
                                row_indices=packed_row_idx,
                                bed_prefix=str(prefix),
                            )
                            r0 = _res[0]
                            lbd = float(r0.get("lbd", np.nan))
                            ml0 = float(r0.get("ml0", np.nan))
                            reml0 = float(r0.get("reml0", np.nan))
                            unified_done = True
                        except Exception as ex:
                            _emit_warning_line(
                                logger,
                                f"{_route_model_label} unified Rust dispatcher unavailable; fallback to legacy packed scan. reason={ex}",
                                use_spinner=bool(use_spinner),
                            )
                            unified_done = False
                    if not unified_done:
                        packed_scan_fn = (
                            jxrs.fvlmm_assoc_packed_f32_to_tsv
                            if str(_route_model_key) == "fvlmm"
                            else jxrs.fastlmm_assoc_packed_f32_to_tsv
                        )
                        lbd, ml0, reml0 = packed_scan_fn(
                            packed,
                            int(packed_n),
                            row_flip,
                            maf,
                            miss,
                            u_trait,
                            s_trait,
                            y_vec,
                            x_arg,
                            sample_idx_scan_arg,
                            -5.0,
                            5.0,
                            50,
                            1e-2,
                            0.0,
                            int(threads),
                            str(genetic_model),
                            [],
                            [],
                            [],
                            [],
                            [],
                            tmp_tsv,
                            row_indices=packed_row_idx,
                            fixed_lbd=(float(fixed_lbd) if fixed_lbd is not None else None),
                            fixed_ml0=None,
                            rotate_block_rows=int(max(1, int(chunk_size))),
                            bed_prefix=str(prefix),
                            **progress_kwargs,
                        )
            scan_secs = max(time.monotonic() - scan_t0, 0.0)
            gwas_ok = True
        finally:
            if stage1_pbar is not None:
                try:
                    stage1_pbar.finish()
                except Exception:
                    pass
                stage1_pbar.close(show_done=False)
            if gwas_pbar is not None:
                try:
                    if gwas_ok:
                        if int(gwas_last_done) < int(max(1, gwas_total)):
                            gwas_pbar.update(int(max(1, gwas_total)) - int(gwas_last_done))
                        gwas_pbar.finish()
                except Exception:
                    pass
                gwas_pbar.close(show_done=False)
            if not gwas_ok:
                _cleanup_gwas_result_tmp(tmp_tsv)

        pve = None
        if bool(getattr(null_fast_mod, "lowrank", False)):
            try:
                pve_tmp = float(getattr(null_fast_mod, "pve"))
                if np.isfinite(pve_tmp):
                    pve = pve_tmp
            except Exception:
                pve = None
        else:
            try:
                vg = float(np.mean(np.asarray(s_trait, dtype=np.float64)))
                lbd_v = float(lbd)
                if np.isfinite(vg) and np.isfinite(lbd_v) and (vg + lbd_v) > 0:
                    pve = vg / (vg + lbd_v)
            except Exception:
                pve = None

        _finalize_gwas_result_tsv(tmp_tsv, out_tsv, prefix, logger=logger)
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

        eff_snp = int(n_sites_active)
        eff_snp_by_trait[pname] = eff_snp
        summary_rows.append(
            {
                "phenotype": str(pname),
                "model": _route_model_label,
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
                f"{_route_model_label} ...pve {float(pve):.3f} [{'/'.join(done_times)}]"
                if (pve is not None and np.isfinite(float(pve)))
                else f"{_route_model_label} ...Finished [{'/'.join(done_times)}]"
            ),
            use_spinner=bool(use_spinner),
        )
        if multi_trait_mode:
            logger.info("")


def run_fvlmm_packed_fullrank(*args, **kwargs) -> None:
    kwargs.setdefault("_route_model_key", "fvlmm")
    kwargs.setdefault("_route_model_label", "FvLMM")
    kwargs.setdefault("_route_model_cls", FvLMM)
    kwargs.setdefault("_allow_pve_switch", False)
    run_fastlmm_packed_fullrank(*args, **kwargs)


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
    force_model: bool = False,
) -> None:
    if not hasattr(jxrs, "lm_block_assoc_packed_to_tsv"):
        raise RuntimeError(
            "Rust extension missing lm_block_assoc_packed_to_tsv. "
            "Rebuild/install JanusX extension first."
        )
    if str(genetic_model).lower() != "add":
        raise ValueError(
            "LM full-rust packed route supports additive coding only."
        )

    prefix, full_ids, packed_ctx, sites_all = _prepare_packed_bed_once_for_gwas(
        genofile=genofile,
        maf_threshold=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
        het_threshold=float(het_threshold),
        snps_only=bool(snps_only),
        use_spinner=bool(use_spinner),
        preloaded_packed=preloaded_packed,
        load_site_meta=False,
    )
    packed_preload_ready = _packed_preload_ready_state(
        prefix=prefix,
        full_ids=full_ids,
        packed_ctx=packed_ctx,
        sites_all=sites_all,
    )
    id_to_idx = {sid: i for i, sid in enumerate(full_ids)}
    try:
        sample_map = np.asarray([id_to_idx[str(sid)] for sid in np.asarray(ids, dtype=str)], dtype=np.int64)
    except KeyError as e:
        raise ValueError("Some aligned sample IDs are not present in packed BED sample order.") from e

    packed, packed_n, packed_row_idx, maf, miss, row_flip = _packed_ctx_active_view_for_gwas(
        packed_ctx
    )
    if packed_n <= 0:
        raise ValueError("Packed BED reported invalid sample count.")
    if (
        int(packed_row_idx.shape[0]) != int(maf.shape[0])
        or int(packed_row_idx.shape[0]) != int(miss.shape[0])
        or int(packed_row_idx.shape[0]) != int(row_flip.shape[0])
    ):
        raise ValueError("Packed BED arrays have inconsistent SNP dimensions.")
    n_sites_active = int(packed_row_idx.shape[0])

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pheno_aligned, ids = _align_pheno_to_sample_order(pheno, ids)
    trait_iter = _resolve_trait_iter(pheno_aligned, trait_names)
    multi_trait_mode = len(trait_iter) > 1

    for pname in trait_iter:
        cpu_t0 = process.cpu_times()
        t0 = time.time()
        peak_rss = process.memory_info().rss

        y_full, sameidx = _trait_values_and_mask(pheno_aligned, str(pname))
        keep_idx_nonmissing = np.flatnonzero(sameidx).astype(np.int64, copy=False)
        n_nonmissing = int(keep_idx_nonmissing.shape[0])
        if n_nonmissing == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")
            continue
        keep_idx = keep_idx_nonmissing
        n_idv = int(keep_idx.shape[0])

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
        pname_tag = _safe_trait_file_label(pname)
        if gm_tag == "add":
            out_tsv = f"{outprefix}.{pname_tag}.lm.tsv"
        else:
            out_tsv = f"{outprefix}.{pname_tag}.{gm_tag}.lm.tsv"
        tmp_tsv = _gwas_result_tmp_path(out_tsv)

        gwas_t0 = time.monotonic()
        gwas_total = int(n_sites_active)
        gwas_last_done = 0
        gwas_pbar: Optional[_ProgressAdapter] = None
        gwas_pbar = _ProgressAdapter(
            total=max(1, gwas_total),
            desc="LM",
            force_animate=True,
            logger=logger,
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
                    logger=logger,
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
                        "progress_every": int(
                            max(
                                1,
                                min(
                                    int(max(1, int(chunk_size))),
                                    _progress_callback_step(int(max(1, gwas_total))),
                                ),
                            )
                        ),
                    }

                ixx = _lm_precompute_ixx_qr(x_design)
                _kept, _scanned = jxrs.lm_block_assoc_packed_to_tsv(
                    y_vec,
                    x_design,
                    ixx,
                    packed,
                    int(packed_n),
                    row_flip,
                    maf,
                    miss,
                    [],
                    [],
                    [],
                    [],
                    [],
                    tmp_tsv,
                    maf_threshold=float(maf_threshold),
                    max_missing_rate=float(max_missing_rate),
                    het_threshold=float(het_threshold),
                    sample_indices=sample_idx_trait,
                    row_indices=packed_row_idx,
                    chunk_size=int(max(1, int(chunk_size))),
                    threads=int(threads),
                    bed_prefix=str(prefix),
                    **kwargs_assoc,
                )
                _written_rows = int(_kept)
            gwas_ok = True
        finally:
            if gwas_pbar is not None:
                try:
                    if gwas_ok:
                        if int(gwas_last_done) < int(max(1, gwas_total)):
                            gwas_pbar.update(int(max(1, gwas_total)) - int(gwas_last_done))
                        gwas_pbar.finish()
                except Exception:
                    pass
                gwas_pbar.close(show_done=False)
            if not gwas_ok:
                _cleanup_gwas_result_tmp(tmp_tsv)

        gwas_secs = max(stage1_secs + stage2_secs, 0.0)
        if gwas_secs <= 0.0:
            gwas_secs = max(time.monotonic() - gwas_t0, 0.0)
        _finalize_gwas_result_tsv(tmp_tsv, out_tsv, prefix, logger=logger)
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

        eff_snp = int(_written_rows) if use_block_lm else int(n_sites_active)
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
                "stage1_time_s": float(stage1_secs),
                "stage2_time_s": float(stage2_secs),
                "gwas_time_s": float(gwas_secs),
                "viz_time_s": float(viz_secs),
                "result_file": str(out_tsv),
            }
        )

        done_times = [format_elapsed(stage1_secs), format_elapsed(stage2_secs)]
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
    model_key: str = "lm",
    lm2_covariate_indices: Union[np.ndarray, None] = None,
) -> None:
    model_key_norm = str(model_key).strip().lower()
    rust_symbol = "lm2_stream_bed_to_tsv" if model_key_norm == "lm2" else "lm_stream_bed_to_tsv"
    if not hasattr(jxrs, rust_symbol):
        raise RuntimeError(
            f"Rust extension missing {rust_symbol}. Rebuild/install JanusX extension first."
        )
    prefix = _as_plink_prefix(genofile)
    if prefix is None:
        raise ValueError(f"{str(model_key_norm).upper()} rust streaming single-entry route requires PLINK BED input.")

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pheno_aligned, ids = _align_pheno_to_sample_order(pheno, ids)
    trait_iter = _resolve_trait_iter(pheno_aligned, trait_names)
    multi_trait_mode = len(trait_iter) > 1

    for pname in trait_iter:
        cpu_t0 = process.cpu_times()
        t0 = time.time()
        peak_rss = process.memory_info().rss

        y_full, sameidx = _trait_values_and_mask(pheno_aligned, str(pname))
        keep_idx_nonmissing = np.flatnonzero(sameidx).astype(np.int64, copy=False)
        n_nonmissing = int(keep_idx_nonmissing.shape[0])
        n_missing = int(np.asarray(y_full).shape[0] - n_nonmissing)
        if n_nonmissing == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")
            continue
        keep_idx = keep_idx_nonmissing
        n_idv = int(keep_idx.shape[0])

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

        model_label = str(model_key_norm).upper()
        model_keys_for_chunk = ["lm", "lm2"] if model_key_norm == "lm2" else "lm"
        model_chunk_size = _resolve_stream_scan_chunk_size(
            int(chunk_size),
            int(n_snps),
            use_spinner=bool(use_spinner),
            n_samples_hint=int(n_idv),
            model_keys=model_keys_for_chunk,
            user_specified=bool(chunk_size_user_set),
        )
        mmap_window_mb = (
            _common_resolve_decode_mmap_window_mb(
                genofile,
                len(ids),
                n_snps,
                _current_bed_memory_mb(),
                needs_copy=False,
            )
            if bool(mmap_limit)
            else None
        )
        if (not bool(chunk_size_user_set)) and int(model_chunk_size) != int(chunk_size):
            _log_file_only(
                logger,
                logging.INFO,
                f"{model_label} auto chunk-size: {int(chunk_size)} -> {int(model_chunk_size)} "
                f"(n={int(n_idv)}).",
            )

        gm_tag = str(genetic_model).lower()
        pname_tag = _safe_trait_file_label(pname)
        if gm_tag == "add":
            out_tsv = f"{outprefix}.{pname_tag}.{model_key_norm}.tsv"
        else:
            out_tsv = f"{outprefix}.{pname_tag}.{gm_tag}.{model_key_norm}.tsv"
        tmp_tsv = _gwas_result_tmp_path(out_tsv)

        scan_threads = int(threads)
        if scan_threads <= 0:
            scan_threads = int(n_cores)
        scan_t0 = time.monotonic()
        pbar_total_hint = int(eff_snp_by_trait.get(str(pname), n_snps))
        pbar: Optional[_ProgressAdapter] = None
        if not bool(use_spinner):
                pbar = _ProgressAdapter(
                    total=int(max(1, pbar_total_hint)),
                    desc=model_label,
                    force_animate=True,
                    logger=logger,
                )
        pbar_done = 0
        warm_task: Optional[CliStatus] = None
        warm_active = False
        if bool(use_spinner):
            warm_task = CliStatus(
                f"Waiting for {model_label} GWAS",
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
                pbar = _ProgressAdapter(
                    total=total_use,
                    desc=model_label,
                    force_animate=True,
                    logger=logger,
                )
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
                kwargs_stream: dict[str, object] = {
                    "progress_callback": _lm_progress,
                    "progress_every": int(
                        max(
                            1,
                            min(
                                int(max(1, int(model_chunk_size))),
                                _progress_callback_step(int(max(1, pbar_total_hint))),
                            ),
                        )
                    ),
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
                        if model_key_norm == "lm2":
                            if cov_all is None:
                                raise ValueError("LM2 requires cov_all to be prepared.")
                            if lm2_covariate_indices is None:
                                raise ValueError(
                                    "LM2 requires explicit interaction columns; use -lm2 0,3 style selectors or fall back to -lm."
                                )
                            cov_idx = np.asarray(lm2_covariate_indices, dtype=np.int64)
                            if int(cov_idx.size) <= 0:
                                raise ValueError(
                                    "LM2 received an empty interaction-column set; use -lm when no interactions are requested."
                                )
                            return jxrs.lm2_stream_bed_to_tsv(
                                str(prefix),
                                y_vec,
                                x_design,
                                np.ascontiguousarray(cov_all[keep_idx], dtype=np.float64),
                                cov_idx,
                                str(tmp_tsv),
                                **kwargs_use,
                            )
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
                            if model_key_norm == "lm2":
                                if cov_all is None:
                                    raise ValueError("LM2 requires cov_all to be prepared.")
                                if lm2_covariate_indices is None:
                                    raise ValueError(
                                        "LM2 requires explicit interaction columns; use -lm2 0,3 style selectors or fall back to -lm."
                                    )
                                cov_idx = np.asarray(lm2_covariate_indices, dtype=np.int64)
                                if int(cov_idx.size) <= 0:
                                    raise ValueError(
                                        "LM2 received an empty interaction-column set; use -lm when no interactions are requested."
                                    )
                                return jxrs.lm2_stream_bed_to_tsv(
                                    str(prefix),
                                    y_vec,
                                    x_design,
                                    np.ascontiguousarray(cov_all[keep_idx], dtype=np.float64),
                                    cov_idx,
                                    str(tmp_tsv),
                                    **kwargs_use,
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
                    if model_key_norm == "lm2":
                        raise
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
                except Exception:
                    pass
                pbar.close(show_done=False)
            if not scan_ok:
                _cleanup_gwas_result_tmp(tmp_tsv)

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
            str(model_key_norm).upper(),
            "single-entry rust streaming scan",
            use_spinner=bool(use_spinner),
        )
        _log_model_line(
            logger,
            str(model_key_norm).upper(),
            f"avg CPU ~ {avg_cpu:.1f}% of {n_cores} c, peak RSS ~ {peak_rss_gb:.2f} G",
            use_spinner=bool(use_spinner),
        )
        _log_file_only(
            logger,
            logging.INFO,
            f"{str(model_key_norm).upper()} stream source SNP rows scanned: {int(scan_all_rows)}",
        )

        done_snps = int(kept_rows)
        if done_snps <= 0:
            _cleanup_gwas_result_tmp(tmp_tsv)
            logger.info(f"{str(model_key_norm).upper()}: no SNPs passed filters for trait {pname}.")
            eff_snp_by_trait[str(pname)] = int(done_snps)
            summary_rows.append(
                {
                    "phenotype": str(pname),
                    "model": str(model_key_norm).upper(),
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

        _finalize_gwas_result_tsv(tmp_tsv, out_tsv, prefix, logger=logger)
        saved_paths.append(str(out_tsv))
        _log_model_line(
            logger,
            str(model_key_norm).upper(),
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
                "model": str(model_key_norm).upper(),
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
            f"{str(model_key_norm).upper()} ...Finished [{'/'.join(done_parts)}]",
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
    force_model: bool = False,
) -> None:
    if not hasattr(jxrs, "lmm_reml_assoc_packed_f32_to_tsv"):
        raise RuntimeError(
            "Rust extension missing lmm_reml_assoc_packed_f32_to_tsv. Rebuild/install JanusX extension first."
        )
    if str(genetic_model).lower() != "add":
        raise ValueError(
            "LMM full-rust packed route supports additive coding only."
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
        load_site_meta=False,
    )
    packed_preload_ready = _packed_preload_ready_state(
        prefix=prefix,
        full_ids=full_ids,
        packed_ctx=packed_ctx,
        sites_all=sites_all,
    )
    id_to_idx = {sid: i for i, sid in enumerate(full_ids)}
    try:
        sample_map = np.asarray([id_to_idx[str(sid)] for sid in np.asarray(ids, dtype=str)], dtype=np.int64)
    except KeyError as e:
        raise ValueError("Some aligned sample IDs are not present in packed BED sample order.") from e

    packed, packed_n, packed_row_idx, maf, miss, row_flip = _packed_ctx_active_view_for_gwas(
        packed_ctx
    )
    if packed_n <= 0:
        raise ValueError("Packed BED reported invalid sample count.")
    if (
        int(packed_row_idx.shape[0]) != int(maf.shape[0])
        or int(packed_row_idx.shape[0]) != int(miss.shape[0])
        or int(packed_row_idx.shape[0]) != int(row_flip.shape[0])
    ):
        raise ValueError("Packed BED arrays have inconsistent SNP dimensions.")
    n_sites_active = int(packed_row_idx.shape[0])

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pheno_aligned, ids = _align_pheno_to_sample_order(pheno, ids)
    trait_iter = _resolve_trait_iter(pheno_aligned, trait_names)
    multi_trait_mode = len(trait_iter) > 1

    for pname in trait_iter:
        cpu_t0 = process.cpu_times()
        t0 = time.time()
        peak_rss = process.memory_info().rss

        y_full, sameidx = _trait_values_and_mask(pheno_aligned, str(pname))
        keep_idx_nonmissing = np.flatnonzero(sameidx).astype(np.int64, copy=False)
        n_nonmissing = int(keep_idx_nonmissing.shape[0])
        if n_nonmissing == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")
            continue
        keep_idx = keep_idx_nonmissing
        n_idv = int(keep_idx.shape[0])

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

        evd_t0 = time.monotonic()
        with CliStatus(
            "Running LMM Eigen-Decomposition...",
            enabled=bool(use_spinner),
            use_process=True,
        ) as task:
            try:
                stage_threads = max(1, int(threads))
                with _gwas_evd_stage_ctx(stage_threads):
                    eigvals, eigvecs, _evd_backend, _evd_elapsed = _gwas_eigh_from_grm(
                        grm,
                        threads=stage_threads,
                        logger=logger,
                        stage_label=f"lmm-fullrank:{pname}",
                        require_rust=True,
                        diag_ridge=1e-6,
                        subset_idx=keep_idx,
                    )
                    mod = LMM.from_spectral(
                        y=y_vec,
                        X=x_cov,
                        eigvals=eigvals,
                        eigvecs=eigvecs,
                        evd_secs=float(_evd_elapsed),
                    )
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
        if not bool(force_model):
            switch_to_lm, lrt_stat, lrt_p = _mixed_model_switch_to_lm_decision(
                y_vec=y_vec,
                x_cov=x_cov,
                lmm_ml0=getattr(mod, "ML0", None),
                alpha=0.05,
            )
            if switch_to_lm:
                logger.warning(
                    f"Warning: LMM switch to LM for trait {pname}: "
                    f"null LRT stat={float(lrt_stat):.4g}, p={float(lrt_p):.4g} (>=0.05)."
                )
                _log_model_line(
                    logger,
                    "LM",
                    (
                        f"switched from LMM null to LM "
                        f"(LRT stat={float(lrt_stat):.4g}, p={float(lrt_p):.4g}) "
                        f"[{format_elapsed(evd_secs)}]"
                    ),
                    use_spinner=bool(use_spinner),
                )
                run_lm_packed_fullrank(
                    genofile=genofile,
                    pheno=pheno,
                    ids=ids,
                    outprefix=outprefix,
                    maf_threshold=maf_threshold,
                    max_missing_rate=max_missing_rate,
                    genetic_model=genetic_model,
                    het_threshold=het_threshold,
                    chunk_size=chunk_size,
                    qmatrix=qmatrix,
                    cov_all=cov_all,
                    plot=plot,
                    threads=threads,
                    logger=logger,
                    use_spinner=use_spinner,
                    snps_only=bool(snps_only),
                    eff_snp_by_trait=eff_snp_by_trait,
                    summary_rows=summary_rows,
                    saved_paths=saved_paths,
                    trait_names=[str(pname)],
                    emit_trait_header=False,
                    preloaded_packed=packed_preload_ready,
                    force_model=bool(force_model),
                )
                if multi_trait_mode:
                    logger.info("")
                continue
        _log_model_line(
            logger,
            "LMM",
            f"PVE(null) ~ {mod.pve:.3f}; eigen-decomposition [{format_elapsed(evd_secs)}]",
            use_spinner=bool(use_spinner),
        )

        sample_idx_trait = np.ascontiguousarray(sample_map[keep_idx], dtype=np.int64)

        gwas_t0 = time.monotonic()
        gwas_total = int(n_sites_active)
        gwas_last_done = 0
        gwas_pbar: Optional[_ProgressAdapter] = None
        gwas_pbar = _ProgressAdapter(
            total=max(1, gwas_total),
            desc="LMM",
            force_animate=True,
            logger=logger,
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
                    logger=logger,
                )
                gwas_last_done = 0
            d = int(max(0, min(d, max(1, gwas_total))))
            step = int(max(0, d - gwas_last_done))
            if step > 0 and gwas_pbar is not None:
                gwas_pbar.update(step)
                gwas_last_done = d

        gwas_ok = False
        null_lbd = float("nan")
        gm_tag = str(genetic_model).lower()
        pname_tag = _safe_trait_file_label(pname)
        if gm_tag == "add":
            out_tsv = f"{outprefix}.{pname_tag}.lmm.tsv"
        else:
            out_tsv = f"{outprefix}.{pname_tag}.{gm_tag}.lmm.tsv"
        tmp_tsv = _gwas_result_tmp_path(out_tsv)
        try:
            with _gwas_scan_stage_ctx(max(1, int(threads))):
                if bool(getattr(mod, "lowrank", False)):
                    sites_lowrank = _resolve_packed_sites_all(prefix, packed_row_idx, sites_all)
                    _written_rows = _packed_lowrank_scan_to_tsv(
                        mod=mod,
                        packed=packed,
                        packed_n=int(packed_n),
                        packed_row_idx=packed_row_idx,
                        row_flip=row_flip,
                        maf=maf,
                        miss=miss,
                        sites_all=sites_lowrank,
                        sample_idx_trait=sample_idx_trait,
                        out_tsv=str(tmp_tsv),
                        chunk_size=int(max(1, int(chunk_size))),
                        threads=int(max(1, int(threads))),
                        genetic_model=str(genetic_model),
                        progress_callback=_lmm_progress if gwas_pbar is not None else None,
                    )
                else:
                    progress_kwargs: dict[str, object] = {}
                    if gwas_pbar is not None:
                        progress_kwargs = {
                            "progress_callback": _lmm_progress,
                            "progress_every": int(
                                max(
                                    1,
                                    min(
                                        int(max(1, int(chunk_size))),
                                        _progress_callback_step(int(max(1, gwas_total))),
                                    ),
                                )
                            ),
                        }
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

                    _written_rows = jxrs.lmm_reml_assoc_packed_f32_to_tsv(
                        packed,
                        int(packed_n),
                        row_flip,
                        maf,
                        miss,
                        s_trait,
                        x_rot,
                        y_rot,
                        u_t,
                        [],
                        [],
                        [],
                        [],
                        [],
                        tmp_tsv,
                        sample_idx_trait,
                        row_indices=packed_row_idx,
                        low=-5.0,
                        high=5.0,
                        max_iter=50,
                        tol=1e-2,
                        threads=int(threads),
                        model="add",
                        nullml=None,
                        init_log10_lbd=init_log10_lbd,
                        rotate_block_rows=int(max(1, int(chunk_size))),
                        bed_prefix=str(prefix),
                        **progress_kwargs,
                    )
                if int(_written_rows) != int(n_sites_active):
                    _emit_warning_line(
                        logger,
                        f"LMM Rust writer row mismatch: expected={n_sites_active} wrote={int(_written_rows)}",
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
                except Exception:
                    pass
                gwas_pbar.close(show_done=False)
            if not gwas_ok:
                _cleanup_gwas_result_tmp(tmp_tsv)

        gwas_secs = max(stage1_secs + stage2_secs, 0.0)
        if gwas_secs <= 0.0:
            gwas_secs = max(time.monotonic() - gwas_t0, 0.0)
        _finalize_gwas_result_tsv(tmp_tsv, out_tsv, prefix, logger=logger)
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

        eff_snp = int(n_sites_active)
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


def run_algwas_packed_fullrank(
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
    qtn_preloaded_packed: Union[dict[str, object], None] = None,
) -> None:
    if str(genetic_model).lower() != "add":
        raise ValueError("ALGWAS full-rust packed route supports additive coding only.")
    if not hasattr(jxrs, "algwas_packed_to_tsv") and not _gwas_use_rust_unified_v1():
        raise RuntimeError(
            "Rust extension missing ALGWAS packed GWAS symbols. "
            "Rebuild/install JanusX extension first."
        )

    qtn_stage_mode = isinstance(qtn_preloaded_packed, dict)
    main_prefix = _as_plink_prefix(genofile)
    if main_prefix is None:
        raise ValueError("ALGWAS route requires PLINK BED input or BED cache prefix.")
    if qtn_stage_mode:
        qtn_prefix = str(qtn_preloaded_packed.get("prefix", "")).strip()
        if not qtn_prefix:
            raise ValueError("Invalid QTN packed payload: missing PLINK prefix.")
        prefix = qtn_prefix
        full_ids = np.asarray(qtn_preloaded_packed.get("full_ids", []), dtype=str)
        packed_ctx_obj = qtn_preloaded_packed.get("packed_ctx")
        if not isinstance(packed_ctx_obj, dict):
            raise ValueError("Invalid QTN packed payload: missing packed_ctx.")
        packed_ctx = packed_ctx_obj
        sites_all = None
    else:
        prefix, full_ids, packed_ctx, sites_all = _prepare_packed_bed_once_for_gwas(
            genofile=genofile,
            maf_threshold=float(maf_threshold),
            max_missing_rate=float(max_missing_rate),
            het_threshold=float(het_threshold),
            snps_only=bool(snps_only),
            use_spinner=bool(use_spinner),
            preloaded_packed=preloaded_packed,
            load_site_meta=False,
        )
    id_to_idx = {sid: i for i, sid in enumerate(full_ids)}
    try:
        sample_map = np.asarray(
            [id_to_idx[str(sid)] for sid in np.asarray(ids, dtype=str)],
            dtype=np.int64,
        )
    except KeyError as e:
        raise ValueError("Some aligned sample IDs are not present in packed BED sample order.") from e

    packed, packed_n, packed_row_idx, maf, miss, row_flip = _packed_ctx_active_view_for_gwas(
        packed_ctx
    )
    if packed_n <= 0:
        raise ValueError("Packed BED reported invalid sample count.")
    if (
        int(packed_row_idx.shape[0]) != int(maf.shape[0])
        or int(packed_row_idx.shape[0]) != int(miss.shape[0])
        or int(packed_row_idx.shape[0]) != int(row_flip.shape[0])
    ):
        raise ValueError("Packed BED arrays have inconsistent SNP dimensions.")
    n_sites_active = int(packed_row_idx.shape[0])

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pheno_aligned, ids = _align_pheno_to_sample_order(pheno, ids)
    trait_iter = _resolve_trait_iter(pheno_aligned, trait_names)
    multi_trait_mode = len(trait_iter) > 1

    for pname in trait_iter:
        cpu_t0 = process.cpu_times()
        t0 = time.time()
        peak_rss = process.memory_info().rss

        y_full, sameidx = _trait_values_and_mask(pheno_aligned, str(pname))
        keep_idx_nonmissing = np.flatnonzero(sameidx).astype(np.int64, copy=False)
        n_nonmissing = int(keep_idx_nonmissing.shape[0])
        if n_nonmissing == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")
            continue
        keep_idx = keep_idx_nonmissing
        n_idv = int(keep_idx.shape[0])

        if bool(emit_trait_header):
            _emit_trait_header(
                logger,
                str(pname),
                int(n_idv),
                pve=None,
                use_spinner=bool(use_spinner),
                width=60,
            )

        # Keep ALGWAS trait filtering consistent with other GWAS models:
        # non-missing phenotypes are retained, missing values are dropped.
        y_vec = np.array(y_full[keep_idx], dtype=np.float64, order="C", copy=True)
        trait_ids = np.asarray(ids[keep_idx], dtype=str)
        x_cov = np.ascontiguousarray(qmatrix[keep_idx], dtype=np.float64)
        if cov_all is not None:
            x_cov = np.ascontiguousarray(
                np.concatenate([x_cov, cov_all[keep_idx]], axis=1),
                dtype=np.float64,
            )
        sample_idx_trait = np.ascontiguousarray(sample_map[keep_idx], dtype=np.int64)

        pname_tag = _safe_trait_file_label(pname)
        out_tsv = f"{outprefix}.{pname_tag}.algwas.tsv"
        tmp_tsv = _gwas_result_tmp_path(out_tsv)
        pseudo_tsv_hint = f"{outprefix}.{pname_tag}.algwas.qtn"
        stage1_tsv_hint = f"{os.path.splitext(out_tsv)[0]}.stage1"
        tmp_stage1_tsv_hint = f"{tmp_tsv}.stage1.tsv"

        stage1_total = 256
        stage1_last_done = 0
        stage1_pbar: Optional[_ProgressAdapter] = None
        stage1_spin_handle = _start_indeterminate_progress_bar("ALGWAS stage1")
        gwas_total = int(n_sites_active)
        gwas_last_done = 0
        gwas_pbar: Optional[_ProgressAdapter] = None

        def _algwas_stage1_progress(done: int, total: int) -> None:
            nonlocal stage1_last_done, stage1_pbar, stage1_spin_handle, peak_rss, stage1_done_at
            try:
                d = int(done)
                t = int(total)
            except Exception:
                return
            if t > 0 and stage1_spin_handle is not None:
                try:
                    _stop_indeterminate_progress_bar(stage1_spin_handle)
                except Exception:
                    pass
                stage1_spin_handle = None
            if stage1_pbar is None:
                if t <= 0:
                    try:
                        peak_rss = max(peak_rss, process.memory_info().rss)
                    except Exception:
                        pass
                    return
                if t > 0:
                    total_use = int(max(1, t))
                elif d > 0:
                    total_use = int(max(stage1_total, d + max(32, d // 4)))
                else:
                    total_use = int(max(1, stage1_total))
                stage1_pbar = _ProgressAdapter(
                    total=total_use,
                    desc="ALGWAS stage1",
                    force_animate=True,
                    logger=logger,
                )
            total_use = int(max(1, stage1_pbar.total if stage1_pbar is not None else stage1_total))
            if t > 0:
                target_total = int(max(1, t))
            else:
                target_total = int(max(total_use, d + max(32, d // 4)))
            if stage1_pbar is not None and target_total != total_use:
                stage1_pbar.set_total(target_total)
                total_use = int(max(1, stage1_pbar.total))
            d = int(max(0, min(d, total_use)))
            stepv = int(max(0, d - stage1_last_done))
            if stepv > 0 and stage1_pbar is not None:
                stage1_pbar.update(stepv)
                stage1_last_done = d
            if stage1_pbar is not None and t > 0 and d >= t:
                if stage1_done_at is None:
                    stage1_done_at = time.monotonic()
                stage1_pbar.finish()
                stage1_pbar.close(show_done=False)
                stage1_pbar = None
            try:
                peak_rss = max(peak_rss, process.memory_info().rss)
            except Exception:
                pass

        def _algwas_progress(done: int, total: int) -> None:
            nonlocal gwas_last_done, gwas_total, gwas_pbar, peak_rss, stage1_pbar, stage1_spin_handle, stage1_done_at
            try:
                d = int(done)
                t = int(total)
            except Exception:
                return
            if stage1_done_at is None:
                stage1_done_at = time.monotonic()
            if stage1_spin_handle is not None:
                try:
                    _stop_indeterminate_progress_bar(stage1_spin_handle)
                except Exception:
                    pass
                stage1_spin_handle = None
            if stage1_pbar is not None:
                try:
                    stage1_pbar.finish()
                except Exception:
                    pass
                try:
                    stage1_pbar.close(show_done=False)
                except Exception:
                    pass
                stage1_pbar = None
            if gwas_pbar is None:
                total_use = int(max(1, t if t > 0 else gwas_total))
                gwas_pbar = _ProgressAdapter(
                    total=total_use,
                    desc="ALGWAS scan",
                    force_animate=True,
                    logger=logger,
                )
            total_use = int(max(1, gwas_pbar.total if gwas_pbar is not None else gwas_total))
            d = int(max(0, min(d, total_use)))
            stepv = int(max(0, d - gwas_last_done))
            if stepv > 0 and gwas_pbar is not None:
                gwas_pbar.update(stepv)
                gwas_last_done = d
            try:
                peak_rss = max(peak_rss, process.memory_info().rss)
            except Exception:
                pass

        qtn_count = 0
        pseudo_rows = 0
        stage2_eff_snp: Optional[int] = None
        gwas_ok = False
        gwas_t0 = time.monotonic()
        stage1_secs = 0.0
        stage2_secs = 0.0
        stage_clock_t0 = gwas_t0
        stage1_done_at: Optional[float] = None
        try:
            with _gwas_scan_stage_ctx(max(1, int(threads))):
                unified_done = False
                stage1_kernel_t0 = time.monotonic()
                if _gwas_use_rust_unified_v1():
                    try:
                        jobs = [
                            {
                                "model": "algwas",
                                "trait": str(pname),
                                "out_tsv": str(tmp_tsv),
                                "y": y_vec,
                                "x_cov": x_cov,
                                "sample_indices": sample_idx_trait,
                                "qtn_bound": None,
                                "lambda_steps": 64,
                                "lambda_min_ratio": 0.001,
                                "scan_step": int(max(1, int(chunk_size))),
                                "stage1_progress_callback": _algwas_stage1_progress,
                                "scan_progress_callback": (None if qtn_stage_mode else _algwas_progress),
                                "progress_every": int(
                                    max(
                                        1,
                                        min(
                                            int(max(1, int(chunk_size))),
                                            _progress_callback_step(int(max(1, gwas_total))),
                                        ),
                                    )
                                ),
                                "pseudo_tsv": str(pseudo_tsv_hint),
                            }
                        ]
                        _res = jxrs.gwas_packed_unified_to_tsv(
                            jobs,
                            packed,
                            int(packed_n),
                            row_flip,
                            maf,
                            miss,
                            [],
                            [],
                            [],
                            [],
                            [],
                            int(max(1, int(chunk_size))),
                            int(threads),
                            None,
                            int(max(1, int(chunk_size))),
                            row_indices=packed_row_idx,
                            bed_prefix=str(prefix),
                        )
                        r0 = _res[0]
                        qtn_count = int(r0.get("qtn_count", 0))
                        pseudo_rows = int(r0.get("pseudo_rows", 0))
                        unified_done = True
                    except Exception as ex:
                        _emit_warning_line(
                            logger,
                            f"ALGWAS unified Rust dispatcher unavailable; fallback to direct ALGWAS kernel. reason={ex}",
                            use_spinner=bool(use_spinner),
                        )
                        unified_done = False

                if not unified_done:
                    qtn_count, pseudo_rows, _written_rows = jxrs.algwas_packed_to_tsv(
                        y_vec,
                        x_cov,
                        [],
                        [],
                        [],
                        [],
                        [],
                        packed,
                        int(packed_n),
                        row_flip,
                        maf,
                        miss,
                        tmp_tsv,
                        sample_indices=sample_idx_trait,
                        qtn_bound=None,
                        lambda_steps=64,
                        lambda_min_ratio=0.001,
                        scan_step=int(max(1, int(chunk_size))),
                        stage1_progress_callback=_algwas_stage1_progress,
                        threads=int(threads),
                        progress_callback=(None if qtn_stage_mode else _algwas_progress),
                        pseudo_tsv=str(pseudo_tsv_hint),
                        row_indices=packed_row_idx,
                        bed_prefix=str(prefix),
                    )
                stage1_kernel_elapsed = max(time.monotonic() - stage1_kernel_t0, 0.0)
                stage1_kernel_end = stage1_kernel_t0 + stage1_kernel_elapsed
                if stage1_done_at is None:
                    stage1_done_at = stage1_kernel_end
                stage1_secs = max(min(stage1_done_at, stage1_kernel_end) - stage_clock_t0, 0.0)
                if qtn_stage_mode:
                    stage2_t0 = time.monotonic()
                    kept_rows2, scan_rows2, qtn_used = run_qtn_segmented_lm_stage2(
                        main_prefix=str(main_prefix),
                        y_vec=y_vec,
                        trait_ids=trait_ids,
                        base_cov=x_cov,
                        qtn_preloaded_packed=qtn_preloaded_packed,  # type: ignore[arg-type]
                        qtn_tsv=str(pseudo_tsv_hint),
                        tmp_tsv=str(tmp_tsv),
                        maf_threshold=float(maf_threshold),
                        max_missing_rate=float(max_missing_rate),
                        het_threshold=float(het_threshold),
                        snps_only=bool(snps_only),
                        chunk_size=int(max(1, int(chunk_size))),
                        threads=int(threads),
                        progress_callback=_algwas_progress,
                        args_obj=type("_AlgwasQtnArgs", (), {"farmcpu_bin_size": [5e5, 5e6, 5e7]})(),
                        logger=logger,
                        use_spinner=False,
                        status_prefix="ALGWAS stage2",
                    )
                    stage2_secs = max(time.monotonic() - stage2_t0, 0.0)
                    pseudo_rows = int(max(pseudo_rows, qtn_used))
                    gwas_total = int(max(1, scan_rows2))
                    stage2_eff_snp = int(kept_rows2)
                else:
                    stage2_secs = max(stage1_kernel_elapsed - stage1_secs, 0.0)
            gwas_ok = True
        finally:
            if stage1_spin_handle is not None:
                try:
                    _stop_indeterminate_progress_bar(stage1_spin_handle)
                except Exception:
                    pass
            if stage1_pbar is not None:
                try:
                    stage1_pbar.finish()
                except Exception:
                    pass
                stage1_pbar.close(show_done=False)
            if gwas_pbar is not None:
                try:
                    if gwas_ok:
                        if int(gwas_last_done) < int(max(1, gwas_total)):
                            gwas_pbar.update(int(max(1, gwas_total)) - int(gwas_last_done))
                        gwas_pbar.finish()
                except Exception:
                    pass
                gwas_pbar.close(show_done=False)
            if not gwas_ok:
                _cleanup_gwas_result_tmp(tmp_tsv)
                _cleanup_gwas_result_tmp(tmp_stage1_tsv_hint)

        gwas_secs = max(stage1_secs + stage2_secs, 0.0)
        if gwas_secs <= 0.0:
            gwas_secs = max(time.monotonic() - gwas_t0, 0.0)
        _finalize_gwas_result_tsv(tmp_tsv, out_tsv, prefix, logger=logger)
        saved_paths.append(str(out_tsv))
        if os.path.exists(tmp_stage1_tsv_hint):
            _replace_file_with_retry(tmp_stage1_tsv_hint, stage1_tsv_hint)
        if os.path.exists(stage1_tsv_hint):
            saved_paths.append(str(stage1_tsv_hint))
            _log_algwas_stage1_model_selection(logger, str(stage1_tsv_hint))
        if int(pseudo_rows) > 0 and os.path.exists(pseudo_tsv_hint):
            saved_paths.append(str(pseudo_tsv_hint))

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
            "ALGWAS",
            f"avg CPU ~ {avg_cpu:.1f}% of {n_cores} c, peak RSS ~ {peak_rss_gb:.2f} G",
            use_spinner=bool(use_spinner),
        )
        _log_model_line(
            logger,
            "ALGWAS",
            f"Results saved to {_display_path(str(out_tsv))}",
            use_spinner=bool(use_spinner),
        )

        eff_snp = int(stage2_eff_snp if stage2_eff_snp is not None else n_sites_active)
        eff_snp_by_trait[str(pname)] = eff_snp
        summary_rows.append(
            {
                "phenotype": str(pname),
                "model": "ALGWAS",
                "nidv": int(n_idv),
                "eff_snp": int(eff_snp),
                "pve": None,
                "avg_cpu": float(avg_cpu),
                "peak_rss_gb": float(peak_rss_gb),
                "stage1_time_s": float(stage1_secs),
                "stage2_time_s": float(stage2_secs),
                "gwas_time_s": float(gwas_secs),
                "viz_time_s": float(viz_secs),
                "result_file": str(out_tsv),
            }
        )

        done_times = [format_elapsed(stage1_secs), format_elapsed(stage2_secs)]
        if plot:
            done_times.append(format_elapsed(viz_secs))
        _rich_success(
            logger,
            f"ALGWAS ...Found {int(qtn_count)} QTNs [{'/'.join(done_times)}]",
            use_spinner=bool(use_spinner),
        )
        if multi_trait_mode:
            logger.info("")


def run_splmm_windowed_fullrank(
    *,
    genofile: str,
    splmm_source: Optional[str] = None,
    splmm_sparse_cutoff: Optional[float] = None,
    splmm_sparse_jxgrm_path: Optional[str] = None,
    splmm_sparse_method: int = 1,
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
    force_model: bool = False,
    scan_mode: str = "exact",
) -> None:
    if str(genetic_model).lower() != "add":
        raise ValueError("SparseLMM windowed BED route supports additive coding only.")
    if not hasattr(jxrs, "splmm_assoc_pcg_bed"):
        raise RuntimeError(
            "Rust extension missing splmm_assoc_pcg_bed. Rebuild/install JanusX extension first."
        )
    scan_mode_norm = str(scan_mode).strip().lower()
    if scan_mode_norm not in {"approx", "exact"}:
        raise ValueError(f"Unsupported SparseLMM scan mode: {scan_mode}")
    if scan_mode_norm == "approx":
        if not hasattr(jxrs, "splmm_residualized_approx_null_fit_from_jxgrm"):
            raise RuntimeError(
                "Rust extension missing splmm_residualized_approx_null_fit_from_jxgrm required by SparseLMM approx residualized null fit. "
                "Rebuild/install JanusX extension first."
            )
    elif not hasattr(jxrs, "spreml_sparse_reml_brent_from_jxgrm"):
        raise RuntimeError(
            "Rust extension missing spreml_sparse_reml_brent_from_jxgrm required by SparseLMM sparse null fit. "
            "Rebuild/install JanusX extension first."
        )

    pheno_aligned, ids = _align_pheno_to_sample_order(pheno, ids)
    prefix = _as_plink_prefix(genofile)
    if prefix is None:
        raise ValueError("SparseLMM requires PLINK BED input/prefix after genotype streaming context preparation.")
    aligned_ids = np.asarray(ids, dtype=str)
    full_geno_ids = _plink_fam_sample_ids(str(prefix))
    id_to_full_idx = {sid: i for i, sid in enumerate(full_geno_ids)}
    try:
        scan_sample_map = np.asarray(
            [id_to_full_idx[str(sid)] for sid in aligned_ids],
            dtype=np.int64,
        )
    except KeyError as e:
        raise ValueError("Some aligned sample IDs are not present in PLINK FAM sample order.") from e
    scan_sample_map = np.ascontiguousarray(scan_sample_map, dtype=np.int64)
    packed_preload_ready = None
    use_preloaded_scan_meta = False
    packed_ctx = None
    scan_row_idx = None
    scan_maf = None
    scan_miss = None
    scan_row_flip = None
    if packed_preload_is_ready(preloaded_packed):
        prefix_packed, full_ids_packed, packed_ctx_packed, sites_all_packed = _prepare_packed_bed_once_for_gwas(
            genofile=genofile,
            maf_threshold=float(maf_threshold),
            max_missing_rate=float(max_missing_rate),
            het_threshold=float(het_threshold),
            snps_only=bool(snps_only),
            use_spinner=bool(use_spinner),
            preloaded_packed=preloaded_packed,
            load_site_meta=False,
        )
        packed_preload_ready = _packed_preload_ready_state(
            prefix=prefix_packed,
            full_ids=full_ids_packed,
            packed_ctx=packed_ctx_packed,
            sites_all=sites_all_packed,
        )
        packed_ctx = packed_ctx_packed
        (
            _scan_packed_unused,
            scan_packed_n,
            scan_row_idx,
            scan_maf,
            scan_miss,
            scan_row_flip,
        ) = _packed_ctx_active_view_for_gwas(packed_ctx_packed)
        if scan_packed_n <= 0:
            raise ValueError("Packed BED reported invalid sample count.")
        if (
            int(scan_row_idx.shape[0]) != int(scan_maf.shape[0])
            or int(scan_row_idx.shape[0]) != int(scan_miss.shape[0])
            or int(scan_row_idx.shape[0]) != int(scan_row_flip.shape[0])
        ):
            raise ValueError("Packed BED arrays have inconsistent SNP dimensions.")
        use_preloaded_scan_meta = True

    sparse_cutoff_log = "precomputed" if splmm_sparse_jxgrm_path is not None else None
    if splmm_sparse_jxgrm_path is None:
        if splmm_sparse_cutoff is None:
            splmm_sparse_cutoff, ignored_splmm_arg = _splmm_parse_sparse_cutoff(splmm_source)
            if ignored_splmm_arg is not None:
                _emit_warning_line(
                    logger,
                    (
                        "SparseLMM only accepts an optional numeric sparse cutoff via -splmm; "
                        f"ignoring non-numeric argument: {ignored_splmm_arg}"
                    ),
                    use_spinner=bool(use_spinner),
                )
        else:
            splmm_sparse_cutoff = float(splmm_sparse_cutoff)
            if (not np.isfinite(splmm_sparse_cutoff)) or float(splmm_sparse_cutoff) < 0.0:
                raise ValueError(
                    f"SparseLMM sparse cutoff must be a finite value >= 0, got {splmm_sparse_cutoff}"
                )
        sparse_cutoff_log = f"{float(splmm_sparse_cutoff):g}"

    kinship_prefix = str(prefix)
    kinship_sample_map = np.asarray(scan_sample_map, dtype=np.int64)
    kinship_present_mask = kinship_sample_map >= 0

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    trait_iter = _resolve_trait_iter(pheno_aligned, trait_names)
    multi_trait_mode = len(trait_iter) > 1
    block_rows_use = int(max(4096, int(chunk_size)))
    shared_sparse_kinship_path: Optional[str] = None
    if splmm_sparse_jxgrm_path is not None:
        shared_sparse_kinship_path = _splmm_normalize_sparse_grm_path(str(splmm_sparse_jxgrm_path))
        if not os.path.exists(shared_sparse_kinship_path):
            raise RuntimeError(
                "Prebuilt SparseLMM sparse GRM path is missing: "
                f"{_display_path(shared_sparse_kinship_path)}"
            )
    sparse_kinship_cache: dict[str, str] = (
        {"__full_sample__": str(shared_sparse_kinship_path)}
        if shared_sparse_kinship_path is not None
        else {}
    )

    for pname in trait_iter:
        cpu_t0 = process.cpu_times()
        t0 = time.time()
        peak_rss = process.memory_info().rss

        y_full, sameidx = _trait_values_and_mask(pheno_aligned, str(pname))
        keep_idx_nonmissing = np.flatnonzero(
            np.asarray(sameidx, dtype=np.bool_) & kinship_present_mask
        ).astype(np.int64, copy=False)
        n_nonmissing = int(keep_idx_nonmissing.shape[0])
        if n_nonmissing == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")
            continue
        dropped_missing_kin = int(np.count_nonzero(np.asarray(sameidx, dtype=np.bool_))) - n_nonmissing
        if dropped_missing_kin > 0:
            _log_model_line(
                logger,
                "SparseLMM",
                f"Trait {pname}: dropped {dropped_missing_kin} sample(s) absent from kinship source.",
                use_spinner=bool(use_spinner),
            )
        keep_idx = keep_idx_nonmissing
        n_idv = int(keep_idx.shape[0])

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
        x_cov = np.ascontiguousarray(qmatrix[keep_idx], dtype=np.float64)
        if cov_all is not None:
            x_cov = np.ascontiguousarray(
                np.concatenate([x_cov, cov_all[keep_idx]], axis=1),
                dtype=np.float64,
            )
        x_arg = x_cov if int(x_cov.shape[1]) > 0 else None
        scan_sample_idx_trait = np.ascontiguousarray(scan_sample_map[keep_idx], dtype=np.int64)
        kinship_sample_idx_trait = np.ascontiguousarray(kinship_sample_map[keep_idx], dtype=np.int64)
        sparse_cache_key = "__full_sample__"
        trait_sparse_kinship_path = sparse_kinship_cache.get(sparse_cache_key)
        if trait_sparse_kinship_path is None:
            trait_sparse_kinship_path = _ensure_splmm_sparse_grm(
                kinship_prefix,
                sample_indices=None,
                out_prefix=_splmm_sparse_out_prefix_for_gwas(
                    kinship_prefix,
                    None,
                    outprefix=str(outprefix),
                    logger=logger,
                ),
                cutoff=float(splmm_sparse_cutoff),
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                het_threshold=float(het_threshold),
                snps_only=bool(snps_only),
                threads=int(threads),
                logger=logger,
                use_spinner=bool(use_spinner),
                method=int(splmm_sparse_method),
            )
            sparse_kinship_cache[sparse_cache_key] = str(trait_sparse_kinship_path)
        prepare_handle = (
            _start_indeterminate_progress_bar("Prepare for SparseLMM...")
            if bool(use_spinner)
            else None
        )

        def _stop_prepare_handle(
            *,
            show_done: bool = False,
            message: Optional[str] = None,
            elapsed_parts: Optional[list[object]] = None,
        ) -> None:
            nonlocal prepare_handle
            if prepare_handle is not None:
                try:
                    _stop_indeterminate_progress_bar(
                        prepare_handle,
                        show_done=show_done,
                        message=message,
                        elapsed_parts=elapsed_parts,
                    )
                except Exception:
                    pass
                prepare_handle = None

        scan_meta = None
        null_fit = None
        scan_meta_secs = 0.0
        null_fit_secs = 0.0
        null_prepare_t0 = time.monotonic()
        scan_meta_mmap_window_mb = int(max(1, _current_bed_memory_mb()))

        def _run_sparse_scan_meta_task() -> tuple[object, float]:
            task_t0 = time.monotonic()
            out = _splmm_bed_logic_meta_selected(
                str(kinship_prefix),
                sample_indices=kinship_sample_idx_trait,
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                het_threshold=float(het_threshold),
                snps_only=bool(snps_only),
                mmap_window_mb=scan_meta_mmap_window_mb,
            )
            return out, max(time.monotonic() - task_t0, 0.0)

        def _run_sparse_null_fit_task() -> tuple[dict[str, object], float]:
            task_t0 = time.monotonic()
            out = _splmm_sparse_null_fit(
                jxgrm_path=str(trait_sparse_kinship_path),
                sample_idx=kinship_sample_idx_trait,
                y_vec=y_vec,
                x_cov=x_arg,
                progress_callback=None,
                residualized_approx=bool(scan_mode_norm == "approx"),
                threads=int(threads),
            )
            return out, max(time.monotonic() - task_t0, 0.0)

        if not bool(use_preloaded_scan_meta):
            with cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="jx-splmm-prepare") as ex:
                future_map: dict[cf.Future, str] = {
                    ex.submit(_run_sparse_scan_meta_task): "scan_meta",
                    ex.submit(_run_sparse_null_fit_task): "null_fit",
                }
                future_results: dict[str, tuple[object, float]] = {}
                try:
                    for fut in cf.as_completed(future_map):
                        future_results[future_map[fut]] = fut.result()
                except Exception:
                    for fut in future_map:
                        fut.cancel()
                    raise
            scan_meta, scan_meta_secs = future_results["scan_meta"]
            null_fit_obj, null_fit_secs = future_results["null_fit"]
            null_fit = dict(null_fit_obj)
        else:
            null_fit_obj, null_fit_secs = _run_sparse_null_fit_task()
            null_fit = dict(null_fit_obj)
        null_secs = max(time.monotonic() - null_prepare_t0, 0.0)
        _stop_prepare_handle(show_done=False)
        peak_rss = max(peak_rss, process.memory_info().rss)
        if bool(use_preloaded_scan_meta):
            n_scan_sites_hint = int(np.asarray(scan_row_idx, dtype=np.int64).reshape(-1).shape[0])
        else:
            if scan_meta is None:
                raise RuntimeError("SparseLMM memmap scan metadata is missing.")
            n_scan_sites_hint = int(
                np.asarray(scan_meta["row_indices"], dtype=np.int64).reshape(-1).shape[0]
            )

        sigma_g2 = float(null_fit["sigma_g2"])
        sigma_e2 = float(null_fit["sigma_e2"])
        null_pve = float(null_fit.get("pve", float("nan")))
        null_lbd = float(null_fit.get("lambda", float("nan")))
        null_mean_diag_k = float(null_fit.get("mean_diag_k", float("nan")))
        null_n_samples_k = int(null_fit.get("n_samples_k", 0) or 0)
        null_nnz_k = int(null_fit.get("nnz_k", 0) or 0)
        null_offdiag_density_k = float(null_fit.get("offdiag_density_k", float("nan")))
        null_lambda_boundary = null_fit.get("lambda_boundary", None)
        sigma_sum_profile = float(null_fit.get("sigma_sum_profile", sigma_g2 + sigma_e2))
        rss_reml = float(null_fit.get("rss_reml", float("nan")))
        df_reml = float(null_fit.get("df_reml", float("nan")))
        resid_var_reml = float(null_fit.get("resid_var_reml", float("nan")))
        resid_var_mx_reml = float(null_fit.get("resid_var_mx_reml", float("nan")))
        ypy_reml = float(null_fit.get("ypy_reml", float("nan")))
        sigma2_profile_reml = float(null_fit.get("sigma2_profile_reml", resid_var_reml))
        sigma2_mx_reml = float(null_fit.get("sigma2_mx_reml", resid_var_mx_reml))
        sigma_scale_p0_to_mx = float(null_fit.get("sigma_scale_p0_to_mx", float("nan")))
        sigma_scale_reml = float(null_fit.get("sigma_scale_reml", float("nan")))
        residualized_approx_null = bool(
            scan_mode_norm == "approx"
            and str(null_fit.get("strategy", "")).strip().lower() == "residualized_sparse_reml_brent"
        )
        if not _splmm_null_components_valid(sigma_g2, sigma_e2):
            _stop_prepare_handle()
            raise RuntimeError(
                f"SparseLMM null variance estimation returned invalid components for trait {pname}: "
                f"sigma_g2={sigma_g2}, sigma_e2={sigma_e2}"
            )
        scan_lbd_arg = float(null_lbd)
        if (not np.isfinite(scan_lbd_arg)) or float(scan_lbd_arg) < 0.0:
            _stop_prepare_handle()
            raise RuntimeError(
                f"SparseLMM null lambda estimation returned invalid value for trait {pname}: "
                f"lambda={scan_lbd_arg}"
            )
        null_backend = str(null_fit.get("backend", "unknown"))
        null_strategy = str(null_fit.get("strategy", "unknown"))
        _null_fit_msg = (
            f"{null_strategy} null fit h2~{float(null_pve):.3f}, "
            f"mean_diag(K)={null_mean_diag_k:.4g}, "
            f"nnz(K)={null_nnz_k}, "
            f"offdiag_density={null_offdiag_density_k:.4g}, "
            f"sparse_cutoff={sparse_cutoff_log}, "
            f"lambda={null_lbd:.4g}, sigma_g2={sigma_g2:.4g}, "
            f"sigma_e2={sigma_e2:.4g}, "
            f"sigma_sum_profile={sigma_sum_profile:.4g}, "
            f"ypy_reml={ypy_reml:.4g}, "
            f"sigma2_profile={sigma2_profile_reml:.4g}, "
            f"sigma2_mx={sigma2_mx_reml:.4g}, "
            f"df_reml={df_reml:.0f}, "
            f"lambda_boundary={null_lambda_boundary or 'interior'}, "
            f"backend={null_backend} [null_fit={format_elapsed(null_fit_secs)}]"
            if np.isfinite(null_pve)
            else f"{null_strategy} null fit sigma_g2={sigma_g2:.4g}, sigma_e2={sigma_e2:.4g}, "
            f"backend={null_backend} [null_fit={format_elapsed(null_fit_secs)}]"
        )
        if bool(use_spinner):
            _log_file_only(logger, logging.INFO, f"SparseLMM: {_null_fit_msg}")
        else:
            _log_model_line(
                logger,
                "SparseLMM",
                _null_fit_msg,
                use_spinner=False,
            )
        grid_local_summary = _splmm_sparse_grid_local_summary(null_fit, window=2)
        if grid_local_summary is not None:
            if bool(use_spinner):
                _log_file_only(
                    logger,
                    logging.INFO,
                    f"SparseLMM: sparse REML grid local log10(lambda): {grid_local_summary}",
                )
            else:
                _log_model_line(
                    logger,
                    "SparseLMM",
                    f"sparse REML grid local log10(lambda): {grid_local_summary}",
                    use_spinner=False,
                )
        if null_lambda_boundary in {"low", "high"}:
            logger.warning(
                f"Warning: SparseLMM sparse REML optimum for trait {pname} is near the {null_lambda_boundary} "
                f"log10(lambda) search boundary; inspect sparse cutoff or widen the search range."
            )
        if (
            null_strategy == "sparse_reml_brent"
            and null_n_samples_k > 1
            and np.isfinite(null_offdiag_density_k)
            and float(null_offdiag_density_k) < 1e-3
        ):
            _log_file_only(
                logger,
                logging.WARNING,
                f"SparseLMM sparse GRM for trait {pname} is nearly diagonal "
                f"(nnz={null_nnz_k}, offdiag_density={float(null_offdiag_density_k):.4g}). "
                "Under this sparse positive-kinship model, h2 may be understated and, in more extreme cases, "
                "LM switch can occur; this is not directly comparable to dense FastLMM h2."
            )
        pheno_trait_subset = pheno_aligned.iloc[keep_idx][[pname]].copy()
        ids_trait_subset = np.asarray(ids[keep_idx], dtype=str)
        if (
            (not bool(force_model))
            and
            null_strategy == "fastlmm_exact_spectral"
            and _fastlmm_should_switch_to_lmm(null_pve)
        ):
            prev_pve = float(null_pve) if np.isfinite(null_pve) else float("nan")
            logger.warning(
                f"Warning: SparseLMM switch to LMM for trait {pname}: "
                f"null PVE={prev_pve:.4f} (>0.995)."
            )
            _log_model_line(
                logger,
                "LMM",
                (
                    f"switched from SparseLMM null to LMM: "
                    f"PVE(null)={prev_pve:.4f} (>0.995) "
                    f"[{format_elapsed(null_secs)}]"
                ),
                use_spinner=bool(use_spinner),
            )
            _stop_prepare_handle()
            from janusx.assoc.workflow_model_stream import run_chunked_gwas_lmm_lm

            run_chunked_gwas_lmm_lm(
                model_name="lmm",
                genofile=genofile,
                pheno=pheno_trait_subset,
                ids=ids_trait_subset,
                n_snps=int(n_scan_sites_hint),
                outprefix=outprefix,
                maf_threshold=maf_threshold,
                max_missing_rate=max_missing_rate,
                genetic_model=genetic_model,
                het_threshold=het_threshold,
                chunk_size=chunk_size,
                mmap_limit=True,
                grm=np.ascontiguousarray(np.asarray(null_fit["grm"], dtype=np.float64), dtype=np.float64),
                qmatrix=qmatrix[keep_idx],
                cov_all=(None if cov_all is None else np.ascontiguousarray(cov_all[keep_idx], dtype=np.float64)),
                eff_m=int(n_scan_sites_hint),
                plot=plot,
                threads=threads,
                logger=logger,
                use_spinner=use_spinner,
                snps_only=bool(snps_only),
                eff_snp_by_trait=eff_snp_by_trait,
                summary_rows=summary_rows,
                saved_paths=saved_paths,
                trait_names=[str(pname)],
                emit_trait_header=False,
                chunk_size_user_set=True,
                force_model=bool(force_model),
            )
            if multi_trait_mode:
                logger.info("")
            _stop_prepare_handle()
            continue

        null_ml0 = null_fit.get("ml", None)
        if (not bool(force_model)) and null_ml0 is not None and np.isfinite(float(null_ml0)):
            switch_to_lm, lrt_stat, lrt_p = _mixed_model_switch_to_lm_decision(
                y_vec=y_vec,
                x_cov=x_cov,
                lmm_ml0=float(null_ml0),
                alpha=0.05,
            )
            if switch_to_lm:
                logger.warning(
                    f"Warning: SparseLMM switch to LM for trait {pname}: "
                    f"null LRT stat={float(lrt_stat):.4g}, p={float(lrt_p):.4g} (>=0.05)."
                )
                _log_model_line(
                    logger,
                    "LM",
                    (
                        f"switched from SparseLMM null to LM "
                        f"(LRT stat={float(lrt_stat):.4g}, p={float(lrt_p):.4g}) "
                        f"[{format_elapsed(null_secs)}]"
                    ),
                    use_spinner=bool(use_spinner),
                )
                _stop_prepare_handle()
                run_lm_stream_bed_single_entry(
                    genofile=genofile,
                    pheno=pheno_trait_subset,
                    ids=ids_trait_subset,
                    n_snps=int(n_scan_sites_hint),
                    outprefix=outprefix,
                    maf_threshold=maf_threshold,
                    max_missing_rate=max_missing_rate,
                    genetic_model=genetic_model,
                    het_threshold=het_threshold,
                    chunk_size=chunk_size,
                    qmatrix=qmatrix[keep_idx],
                    cov_all=(None if cov_all is None else np.ascontiguousarray(cov_all[keep_idx], dtype=np.float64)),
                    plot=plot,
                    threads=threads,
                    logger=logger,
                    use_spinner=use_spinner,
                    snps_only=bool(snps_only),
                    eff_snp_by_trait=eff_snp_by_trait,
                    summary_rows=summary_rows,
                    saved_paths=saved_paths,
                    trait_names=[str(pname)],
                    emit_trait_header=False,
                )
                if multi_trait_mode:
                    logger.info("")
                _stop_prepare_handle()
                continue

        for _buf_key in ("grm", "eigvals", "eigvecs"):
            null_fit.pop(_buf_key, None)

        def _make_splmm_scan_progress():
            current_done = 0
            current_total = 1
            current_pbar: Optional[_ProgressAdapter] = None

            def _progress(_stage: int, done: int, total: int) -> None:
                nonlocal current_done, current_total, current_pbar, peak_rss
                try:
                    done_i = int(done)
                    total_i = int(total)
                except Exception:
                    return
                _stop_prepare_handle()
                total_use = int(max(1, total_i if total_i > 0 else 1))
                if current_pbar is None:
                    current_total = total_use
                    current_pbar = _ProgressAdapter(
                        total=total_use,
                        desc="SparseLMM",
                        force_animate=True,
                        logger=logger,
                    )
                elif total_use != int(max(1, current_total)):
                    current_total = total_use
                    try:
                        current_pbar.set_total(total_use)
                    except Exception:
                        pass
                done_use = int(max(0, min(done_i, total_use)))
                stepv = int(max(0, done_use - current_done))
                if stepv > 0 and current_pbar is not None:
                    current_pbar.update(stepv)
                current_done = done_use
                try:
                    peak_rss = max(peak_rss, process.memory_info().rss)
                except Exception:
                    pass

            def _finish(success: bool) -> None:
                nonlocal current_pbar
                _stop_prepare_handle()
                if current_pbar is not None:
                    try:
                        total_use = int(max(1, current_total))
                        if success and int(current_done) < total_use:
                            current_pbar.update(total_use - int(current_done))
                        if success:
                            current_pbar.finish()
                    except Exception:
                        pass
                    try:
                        current_pbar.close(show_done=False)
                    except Exception:
                        pass
                    current_pbar = None

            return (_progress if bool(use_spinner) else None), _finish

        if bool(use_preloaded_scan_meta):
            trait_row_idx = np.ascontiguousarray(np.asarray(scan_row_idx, dtype=np.int64).reshape(-1), dtype=np.int64)
            trait_scan_maf = np.ascontiguousarray(np.asarray(scan_maf, dtype=np.float32).reshape(-1), dtype=np.float32)
            trait_scan_miss = np.ascontiguousarray(np.asarray(scan_miss, dtype=np.float32).reshape(-1), dtype=np.float32)
            trait_scan_row_flip = np.ascontiguousarray(np.asarray(scan_row_flip, dtype=np.bool_).reshape(-1), dtype=np.bool_)
            trait_site_keep = None
        else:
            if scan_meta is None:
                raise RuntimeError("SparseLMM memmap scan metadata is missing.")
            trait_row_idx = np.ascontiguousarray(np.asarray(scan_meta["row_indices"], dtype=np.int64).reshape(-1), dtype=np.int64)
            trait_scan_maf = np.ascontiguousarray(np.asarray(scan_meta["maf"], dtype=np.float32).reshape(-1), dtype=np.float32)
            trait_scan_miss = np.ascontiguousarray(np.asarray(scan_meta["missing_rate"], dtype=np.float32).reshape(-1), dtype=np.float32)
            trait_scan_row_flip = np.ascontiguousarray(np.asarray(scan_meta["row_flip"], dtype=np.bool_).reshape(-1), dtype=np.bool_)
            trait_site_keep = np.ascontiguousarray(np.asarray(scan_meta["site_keep"], dtype=np.bool_).reshape(-1), dtype=np.bool_)
        n_trait_sites = int(trait_row_idx.shape[0])
        mmap_window_mb = _common_resolve_decode_mmap_window_mb(
            genofile,
            len(ids),
            n_trait_sites,
            _current_bed_memory_mb(),
            needs_copy=False,
        )
        supports_direct_tsv = bool(hasattr(jxrs, "splmm_assoc_pcg_bed_to_tsv"))
        # BIM metadata (chrom/pos/snp/alleles) is now read inside the Rust
        # splmm_assoc_pcg_bed_to_tsv function when the arrays are empty.
        # Passing [] avoids building five large Python lists per trait.
        chrom_all: list[str] = []
        pos_all: list[int] = []
        allele0_all: list[str] = []
        allele1_all: list[str] = []
        snp_all: list[str] = []

        scan_t0 = time.monotonic()
        scan_ok = False
        r_hat = float("nan")
        y_conv = True
        y_iters = 0
        y_rel_res = 0.0
        x_conv_all = True
        x_max_iters = 0
        x_max_rel_res = 0.0
        rhat_requested = 30
        rhat_used = 30
        rust_prepare_inputs_secs = 0.0
        rust_bim_meta_secs = 0.0
        rust_null_scan_core_secs = 0.0
        rust_writer_wait_secs = 0.0
        rust_detach_wall_secs = 0.0
        rust_other_secs = 0.0
        rust_total_secs = 0.0
        factor_nnz = None
        arr = None
        wrote_direct_tsv = False
        scan_route_desc = (
            "sparse Cholesky scan (memmap; preloaded row stats)"
            if bool(use_preloaded_scan_meta)
            else "sparse Cholesky scan (memmap)"
        )
        sparse_progress, sparse_finish = _make_splmm_scan_progress()
        pname_tag = _safe_trait_file_label(pname)
        out_tsv = f"{outprefix}.{pname_tag}.splmm.tsv"
        tmp_tsv = _gwas_result_tmp_path(out_tsv)
        try:
            with _gwas_scan_stage_ctx(max(1, int(threads))):
                if supports_direct_tsv:
                    wrote_direct_tsv = True
                    scan_route_desc = f"{scan_route_desc}; Rust async TSV writer"
                    splmm_out = jxrs.splmm_assoc_pcg_bed_to_tsv(
                        str(kinship_prefix),
                        y_vec,
                        float(scan_lbd_arg),
                        chrom_all,
                        pos_all,
                        snp_all,
                        allele0_all,
                        allele1_all,
                        str(tmp_tsv),
                        x_cov=x_arg,
                        sample_indices=scan_sample_idx_trait,
                        operator_sample_indices=kinship_sample_idx_trait,
                        site_keep=trait_site_keep,
                        tol=1e-5,
                        max_iter=200,
                        block_rows=block_rows_use,
                        std_eps=1e-12,
                        use_train_maf=True,
                        threads=int(threads),
                        model="add",
                        rhat_markers=1000,
                        rhat_seed=20260527,
                        packed=None,
                        packed_n_samples=0,
                        maf=trait_scan_maf,
                        row_flip=trait_scan_row_flip,
                        row_missing=trait_scan_miss,
                        row_indices=trait_row_idx,
                        sparse_jxgrm_path=str(trait_sparse_kinship_path),
                        stage1_progress_callback=None,
                        scan_progress_callback=sparse_progress,
                        progress_every=int(max(512, min(int(block_rows_use), 4096))),
                        rhat_tol=1e-3,
                        scan_mode=scan_mode_norm,
                        mmap_window_mb=(int(mmap_window_mb) if mmap_window_mb is not None else None),
                    )
                else:
                    splmm_out = jxrs.splmm_assoc_pcg_bed(
                        str(kinship_prefix),
                        y_vec,
                        float(scan_lbd_arg),
                        x_cov=x_arg,
                        sample_indices=scan_sample_idx_trait,
                        operator_sample_indices=kinship_sample_idx_trait,
                        site_keep=trait_site_keep,
                        tol=1e-5,
                        max_iter=200,
                        block_rows=block_rows_use,
                        std_eps=1e-12,
                        use_train_maf=True,
                        threads=int(threads),
                        model="add",
                        rhat_markers=1000,
                        rhat_seed=20260527,
                        packed=None,
                        packed_n_samples=0,
                        maf=trait_scan_maf,
                        row_flip=trait_scan_row_flip,
                        row_missing=trait_scan_miss,
                        row_indices=trait_row_idx,
                        sparse_jxgrm_path=str(trait_sparse_kinship_path),
                        stage1_progress_callback=None,
                        scan_progress_callback=sparse_progress,
                        progress_every=int(max(512, min(int(block_rows_use), 4096))),
                        rhat_tol=1e-3,
                        scan_mode=scan_mode_norm,
                        mmap_window_mb=(int(mmap_window_mb) if mmap_window_mb is not None else None),
                    )
            scan_ok = True
        finally:
            sparse_finish(scan_ok)
            if (not scan_ok) and wrote_direct_tsv:
                _cleanup_gwas_result_tmp(tmp_tsv)

        splmm_t = tuple(splmm_out)
        r_hat = float(splmm_t[0])
        y_conv = bool(splmm_t[1])
        y_iters = int(splmm_t[2])
        y_rel_res = float(splmm_t[3])
        x_conv_all = bool(splmm_t[4])
        x_max_iters = int(splmm_t[5])
        x_max_rel_res = float(splmm_t[6])
        rhat_requested = int(splmm_t[7])
        rhat_used = int(splmm_t[8])
        written_rows = 0
        if wrote_direct_tsv:
            written_rows = int(splmm_t[9])
            if len(splmm_t) >= 11 and isinstance(splmm_t[10], (tuple, list)) and len(splmm_t[10]) >= 4:
                rust_prepare_inputs_secs = float(splmm_t[10][0])
                rust_bim_meta_secs = float(splmm_t[10][1])
                rust_null_scan_core_secs = float(splmm_t[10][2])
                rust_writer_wait_secs = float(splmm_t[10][3])
                if len(splmm_t[10]) >= 7:
                    rust_detach_wall_secs = float(splmm_t[10][4])
                    rust_other_secs = float(splmm_t[10][5])
                    rust_total_secs = float(splmm_t[10][6])
            if len(splmm_t) >= 12:
                factor_nnz = int(splmm_t[11])
        else:
            arr = np.ascontiguousarray(np.asarray(splmm_t[9], dtype=np.float64), dtype=np.float64)
            if len(splmm_t) >= 11:
                factor_nnz = int(splmm_t[10])

        scan_secs = max(time.monotonic() - scan_t0, 0.0)
        peak_rss = max(peak_rss, process.memory_info().rss)

        if not wrote_direct_tsv:
            if arr is None:
                raise RuntimeError("SparseLMM scan produced no result matrix.")
            if arr.shape != (n_trait_sites, 3):
                raise RuntimeError(
                    f"SparseLMM result shape mismatch: got {arr.shape}, expected ({n_trait_sites}, 3)."
                )

            def _write_splmm() -> None:
                nonlocal written_rows, arr
                sites_use = _resolve_bim_site_columns_meta(
                    _make_deferred_bim_metadata(
                        str(kinship_prefix),
                        trait_row_idx,
                        n_markers=int(n_trait_sites),
                    ),
                    prefix=str(kinship_prefix),
                    row_indices=trait_row_idx,
                    expected_rows=int(n_trait_sites),
                )
                written_rows = int(
                    _write_splmm_assoc_tsv(
                        out_tsv=tmp_tsv,
                        sites_all=sites_use,
                        maf=trait_scan_maf,
                        miss=trait_scan_miss,
                        results=arr,
                        genetic_model=str(genetic_model),
                    )
                )
                arr = None

            _run_result_write_with_status(
                _write_splmm,
                use_spinner=False,
                emit_done_line=False,
            )
        if int(written_rows) != int(n_trait_sites):
            _emit_warning_line(
                logger,
                f"SparseLMM writer row mismatch: expected={n_trait_sites} wrote={int(written_rows)}",
                use_spinner=bool(use_spinner),
            )
        _finalize_gwas_result_tsv(tmp_tsv, out_tsv, prefix, logger=logger)
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

        cpu_t1 = process.cpu_times()
        t1 = time.time()
        wall = max(t1 - t0, 1e-12)
        cpu_used = (cpu_t1.user - cpu_t0.user) + (cpu_t1.system - cpu_t0.system)
        avg_cpu = 100.0 * cpu_used / (wall * max(1, n_cores))
        peak_rss_gb = peak_rss / (1024 ** 3)

        _scan_mode_desc = {
            "approx": "approximate GRAMMAR-gamma",
            "exact": "exact g'Pg",
        }[scan_mode_norm]
        _rhat_info = (
            f"r_hat={r_hat:.4g}, sampled {rhat_used}/{rhat_requested}, "
            if int(rhat_requested) > 0
            else ""
        )
        if scan_mode_norm == "approx":
            scan_sigma_mode = "lambda_only_residualized_approx"
            _sigma_mode_info = (
                f"scan_param=lambda_only(lambda={scan_lbd_arg:.4g}); "
                f"Rust residualized approx builds scan-scale sigma2 internally; "
                f"sampled markers={rhat_used}/{rhat_requested}"
            )
        else:
            scan_sigma_mode = "lambda_only_exact_internal_sigma2"
            _sigma_mode_info = (
                f"scan_param=lambda_only(lambda={scan_lbd_arg:.4g}); "
                "Rust exact scan uses K+lambda I and null sigma2=y'P0y/df internally"
            )
        _scan_diag_msg = (
            f"{scan_route_desc}; mode={_scan_mode_desc}; {_rhat_info}"
            f"{_sigma_mode_info}; "
            f"V^-1y converged={str(y_conv).lower()} ({y_iters} iters, relres={y_rel_res:.2e}), "
            f"V^-1X converged={str(x_conv_all).lower()} ({x_max_iters} iters, relres={x_max_rel_res:.2e})"
        )
        _timing_diag_msg = (
            f"trait timing: prepare_wall={format_elapsed(null_secs)}, "
            f"scan_meta={format_elapsed(scan_meta_secs)}, "
            f"null_fit={format_elapsed(null_fit_secs)}, "
            f"scan={format_elapsed(scan_secs)}"
        )
        if not (np.isfinite(rust_total_secs) and rust_total_secs > 0.0):
            rust_total_secs = float(
                rust_prepare_inputs_secs
                + rust_bim_meta_secs
                + rust_null_scan_core_secs
                + rust_writer_wait_secs
            )
        _rust_top_level_timing_msg = None
        if np.isfinite(rust_total_secs) and rust_total_secs > 0.0:
            _rust_top_level_timing_msg = (
                "Rust top-level timing: "
                f"prepare_inputs={format_elapsed(rust_prepare_inputs_secs)}, "
                f"bim_meta={format_elapsed(rust_bim_meta_secs)}, "
                f"null_scan_core={format_elapsed(rust_null_scan_core_secs)}, "
                f"writer_wait={format_elapsed(rust_writer_wait_secs)}, "
                f"detach_wall={format_elapsed(rust_detach_wall_secs)}, "
                f"other={format_elapsed(rust_other_secs)}, "
                f"total={format_elapsed(rust_total_secs)}"
            )
        _factor_nnz_msg = None
        if factor_nnz is not None:
            _factor_nnz_msg = f"sparse Cholesky factor: nnz(L)={int(factor_nnz):,}"
        _assoc_test_msg = (
            "association test uses Wald chi^2(1): chisq=(beta/se)^2, "
            "pwald=Pr[Chi^2_1>=chisq]; SparseLMM does not currently emit a PLRT/LRT column."
        )
        if _rust_top_level_timing_msg is not None:
            if bool(use_spinner):
                _log_file_only(
                    logger,
                    logging.INFO,
                    f"SparseLMM: {_rust_top_level_timing_msg}",
                )
            else:
                _emit_plain_info_line(
                    logger,
                    f"SparseLMM: {_rust_top_level_timing_msg}",
                    use_spinner=False,
                )
        if _factor_nnz_msg is not None:
            if bool(use_spinner):
                _log_file_only(
                    logger,
                    logging.INFO,
                    f"SparseLMM: {_factor_nnz_msg}",
                )
            else:
                _log_model_line(
                    logger,
                    "SparseLMM",
                    _factor_nnz_msg,
                    use_spinner=False,
                )
        if bool(use_spinner):
            _log_file_only(logger, logging.INFO, f"SparseLMM: {_timing_diag_msg}")
            _log_file_only(logger, logging.INFO, f"SparseLMM: {_scan_diag_msg}")
            _log_file_only(logger, logging.INFO, f"SparseLMM: {_assoc_test_msg}")
        else:
            _log_model_line(
                logger,
                "SparseLMM",
                _timing_diag_msg,
                use_spinner=False,
            )
            _log_model_line(
                logger,
                "SparseLMM",
                _scan_diag_msg,
                use_spinner=False,
            )
            _log_model_line(
                logger,
                "SparseLMM",
                _assoc_test_msg,
                use_spinner=False,
            )
        if bool(use_spinner):
            _log_file_only(
                logger,
                logging.INFO,
                f"SparseLMM: avg CPU ~ {avg_cpu:.1f}% of {n_cores} c, peak RSS ~ {peak_rss_gb:.2f} G",
            )
        else:
            _log_model_line(
                logger,
                "SparseLMM",
                f"avg CPU ~ {avg_cpu:.1f}% of {n_cores} c, peak RSS ~ {peak_rss_gb:.2f} G",
                use_spinner=False,
            )
        _log_model_line(
            logger,
            "SparseLMM",
            f"Results saved to {_display_path(str(out_tsv))}",
            use_spinner=bool(use_spinner),
        )

        eff_snp = int(n_trait_sites)
        eff_snp_by_trait[str(pname)] = eff_snp
        summary_rows.append(
            {
                "phenotype": str(pname),
                "model": "SparseLMM",
                "nidv": int(n_idv),
                "eff_snp": int(eff_snp),
                "pve": (float(null_pve) if np.isfinite(null_pve) else None),
                "avg_cpu": float(avg_cpu),
                "peak_rss_gb": float(peak_rss_gb),
                "gwas_time_s": float(null_secs + scan_secs),
                "splmm_prepare_wall_s": float(null_secs),
                "splmm_scan_meta_s": float(scan_meta_secs),
                "splmm_null_fit_s": float(null_fit_secs),
                "splmm_scan_s": float(scan_secs),
                "viz_time_s": float(viz_secs),
                "splmm_sigma_scale_reml": (
                    None if residualized_approx_null else (
                        float(sigma_scale_reml) if np.isfinite(sigma_scale_reml) else None
                    )
                ),
                "splmm_sigma2_profile_reml": (
                    float(sigma2_profile_reml) if np.isfinite(sigma2_profile_reml) else None
                ),
                "splmm_sigma2_mx_reml": (
                    float(sigma2_mx_reml) if np.isfinite(sigma2_mx_reml) else None
                ),
                "splmm_sigma_scale_p0_to_mx": (
                    float(sigma_scale_p0_to_mx) if np.isfinite(sigma_scale_p0_to_mx) else None
                ),
                "splmm_scan_sigma_mode": str(scan_sigma_mode),
                "splmm_scan_lambda": (
                    float(scan_lbd_arg) if np.isfinite(scan_lbd_arg) else None
                ),
                "splmm_scan_sigma2": None,
                "splmm_scan_sigma_g2": None,
                "splmm_scan_sigma_e2": None,
                "splmm_resid_var_reml": (
                    float(resid_var_reml) if np.isfinite(resid_var_reml) else None
                ),
                "splmm_resid_var_mx_reml": (
                    float(resid_var_mx_reml) if np.isfinite(resid_var_mx_reml) else None
                ),
                "splmm_ypy_reml": (
                    float(ypy_reml) if np.isfinite(ypy_reml) else None
                ),
                "result_file": str(out_tsv),
            }
        )

        done_times = [
            format_elapsed(null_secs),
            format_elapsed(scan_secs),
        ]
        if plot:
            done_times.append(format_elapsed(viz_secs))
        _rich_success(
            logger,
            (
                f"SparseLMM ...pve {float(null_pve):.3f} [{'/'.join(done_times)}]"
                # f"SparseLMM ...r_hat {r_hat:.4g} h2 {float(null_pve):.3f} [{'/'.join(done_times)}]"
                if np.isfinite(null_pve)
                else f"SparseLMM ...r_hat {r_hat:.4g} [{'/'.join(done_times)}]"
                # else f"SparseLMM ...r_hat {r_hat:.4g} [{'/'.join(done_times)}]"
            ),
            use_spinner=bool(use_spinner),
        )
        _stop_prepare_handle()
        if multi_trait_mode:
            logger.info("")


def run_splmm_packed_fullrank(*args, **kwargs) -> None:
    """Backward-compatible alias for the maintained windowed SparseLMM route."""
    run_splmm_windowed_fullrank(*args, **kwargs)
