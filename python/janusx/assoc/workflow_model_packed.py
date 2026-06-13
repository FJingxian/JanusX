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
from typing import Optional, Union

import numpy as np
import pandas as pd
import psutil
from janusx.assoc.workflow_cache import _is_writable_dir, _resolve_gwas_cache_dir
from janusx.script._common.packedctx import (
    prepare_packed_ctx_from_plink,
    packed_preload_is_disabled,
    packed_preload_is_ready,
    packed_preload_reason,
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
    _mixed_model_switch_to_lm_decision,
    _progress_callback_step,
    _coerce_bim_site_columns,
    _read_bim_site_columns,
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
    auto_mmap_window_mb,
    detect_effective_threads,
    format_elapsed,
    jxrs,
    prepare_cli_input_cache,
)

_WARNED_LM_STREAM_MMAP_LEGACY = False
_SPLMM_EXACT_N_MAX = 15_000
_SPLMM_SPARSE_NULL_CLIP_EIG_EXACT_N_MAX = 0
_SPLMM_KING_THRESHOLD = 0.05
_SPLMM_SPARSE_GRM_CUTOFF = 0.05
_SPLMM_SPARSE_GRM_METHOD = 2
_SPLMM_SPARSE_REML_GRID_SIZE = 17
_SPLMM_SPARSE_REML_MAX_ITER = 20
_SPLMM_SPGRM_SINGLE_BLOCK_N_MAX = 2048
_JXLMM_EXACT_N_MAX = _SPLMM_EXACT_N_MAX
_JXLMM_SPARSE_NULL_CLIP_EIG_EXACT_N_MAX = _SPLMM_SPARSE_NULL_CLIP_EIG_EXACT_N_MAX


def _current_bed_memory_mb() -> float:
    try:
        mb = float(os.environ.get("JX_BED_BLOCK_TARGET_MB", "512"))
    except Exception:
        return 512.0
    return mb if np.isfinite(mb) and mb > 0.0 else 512.0
_JXLMM_KING_THRESHOLD = _SPLMM_KING_THRESHOLD
_JXLMM_SPARSE_GRM_CUTOFF = _SPLMM_SPARSE_GRM_CUTOFF
_JXLMM_SPARSE_GRM_METHOD = _SPLMM_SPARSE_GRM_METHOD
_JXLMM_SPARSE_REML_GRID_SIZE = _SPLMM_SPARSE_REML_GRID_SIZE
_JXLMM_SPARSE_REML_MAX_ITER = _SPLMM_SPARSE_REML_MAX_ITER
_JXLMM_SPGRM_SINGLE_BLOCK_N_MAX = _SPLMM_SPGRM_SINGLE_BLOCK_N_MAX
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


def _splmm_approx_legacy_backend() -> bool:
    legacy_flag = str(os.environ.get("JX_SPLMM_APPROX_LEGACY", "")).strip().lower()
    if legacy_flag in {"1", "true", "yes", "on"}:
        return True
    raw = str(os.environ.get("JX_SPLMM_APPROX_BACKEND", "")).strip().lower()
    return raw in {"legacy", "old", "p0", "gamma0"}


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
        return float(_JXLMM_SPARSE_GRM_CUTOFF), None
    raw = str(splmm_source).strip()
    if raw in {"", "__SELF__"}:
        return float(_JXLMM_SPARSE_GRM_CUTOFF), None
    try:
        cutoff = float(raw)
    except Exception:
        return float(_JXLMM_SPARSE_GRM_CUTOFF), raw
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


_jxlmm_sparse_meta_path = _splmm_sparse_meta_path
_jxlmm_normalize_jxgrm_path = _splmm_normalize_sparse_grm_path
_jxlmm_sparse_sample_hash = _splmm_sparse_sample_hash
_jxlmm_sparse_out_prefix = _splmm_sparse_out_prefix
_jxlmm_sparse_out_prefix_for_gwas = _splmm_sparse_out_prefix_for_gwas
_jxlmm_sparse_layout_rebuild_reason = _splmm_sparse_layout_rebuild_reason
_jxlmm_format_stage_times = _splmm_format_stage_times
_jxlmm_sparse_spec = _splmm_sparse_spec
_jxlmm_parse_sparse_cutoff = _splmm_parse_sparse_cutoff
_ensure_jxlmm_sparse_grm = _ensure_splmm_sparse_grm


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
    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
    packed_n = int(packed_ctx["n_samples"])
    if packed.ndim != 2:
        raise ValueError("Packed GWAS context requires packed ndim=2.")
    miss_full = np.ascontiguousarray(
        np.asarray(packed_ctx["missing_rate"], dtype=np.float32).reshape(-1),
        dtype=np.float32,
    )
    if int(miss_full.shape[0]) != int(packed.shape[0]):
        raise ValueError("Packed GWAS context mismatch: missing_rate length != packed rows.")
    af_full = np.ascontiguousarray(
        np.asarray(packed_ctx.get("af", packed_ctx["maf"]), dtype=np.float32).reshape(-1),
        dtype=np.float32,
    )
    if int(af_full.shape[0]) != int(packed.shape[0]):
        raise ValueError("Packed GWAS context mismatch: af length != packed rows.")
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
    maf_active = np.ascontiguousarray(af_full[active_row_idx], dtype=np.float32)
    miss_active = np.ascontiguousarray(miss_full[active_row_idx], dtype=np.float32)
    row_flip_active = np.ascontiguousarray(row_flip_full[active_row_idx], dtype=np.bool_)
    return packed, packed_n, active_row_idx, maf_active, miss_active, row_flip_active


def _prepare_packed_bed_once_for_gwas(
    *,
    genofile: str,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    snps_only: bool,
    use_spinner: bool,
    preloaded_packed: Union[dict[str, object], None] = None,
) -> tuple[str, np.ndarray, dict[str, object], _BimSiteColumns]:
    """
    Resolve packed BED context for GWAS full-rust routes.

    Returns
    -------
    prefix, full_ids, packed_ctx, site_meta
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
        if site_meta is not None and len(site_meta) > 0:
            return str(prefix), full_ids, packed_ctx, site_meta
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
    site_meta = _read_bim_site_columns(str(prefix), active_row_idx)
    if int(len(site_meta)) != int(active_row_idx.shape[0]):
        raise ValueError(
            f"Packed/BIM mismatch after filtering: sites={len(site_meta)} "
            f"active_rows={active_row_idx.shape[0]}"
        )
    return str(prefix), full_ids, packed_ctx, site_meta


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
    sites_all: _BimSiteColumns,
) -> dict[str, object]:
    return {
        "prefix": str(prefix),
        "full_ids": np.asarray(full_ids, dtype=str),
        "packed_ctx": packed_ctx,
        "site_meta": sites_all,
        "sites_all": sites_all,
    }


def _splmm_null_components_valid(sigma_g2: float, sigma_e2: float) -> bool:
    return bool(
        np.isfinite(float(sigma_g2))
        and np.isfinite(float(sigma_e2))
        and float(sigma_g2) >= 0.0
        and float(sigma_e2) >= 0.0
        and (float(sigma_g2) > 0.0 or float(sigma_e2) > 0.0)
    )


def _jxlmm_ols_residualize(y_vec: np.ndarray, x_cov: Union[np.ndarray, None]) -> np.ndarray:
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


def _jxlmm_pve_from_components(eigvals: np.ndarray, sigma_g2: float, sigma_e2: float) -> float:
    vals = np.asarray(eigvals, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return float("nan")
    mean_k = float(np.mean(np.maximum(vals, 0.0)))
    var_g = float(sigma_g2) * max(mean_k, 0.0)
    denom = var_g + float(sigma_e2)
    if np.isfinite(denom) and denom > 0.0:
        return float(var_g / denom)
    return float("nan")


def _jxlmm_component_ratio(sigma_g2: float, sigma_e2: float) -> float:
    denom = float(sigma_g2) + float(sigma_e2)
    if np.isfinite(denom) and denom > 0.0:
        return float(float(sigma_g2) / denom)
    return float("nan")


def _jxlmm_load_sparse_grm_subset_dense(
    jxgrm_path: str,
    sample_idx: Union[np.ndarray, None] = None,
    progress_callback=None,
) -> np.ndarray:
    file_size = int(os.path.getsize(jxgrm_path))
    if file_size < 16:
        raise RuntimeError(f"Sparse GRM CSC file is too short: {jxgrm_path}")
    with open(jxgrm_path, "rb") as fh:
        header = fh.read(16)
    if len(header) != 16:
        raise RuntimeError(f"Sparse GRM CSC header is incomplete: {jxgrm_path}")

    n_samples = int.from_bytes(header[0:8], "little", signed=False)
    nnz = int.from_bytes(header[8:16], "little", signed=False)
    col_ptr_offset = 16
    col_ptr_bytes = (n_samples + 1) * 8
    row_idx_offset = col_ptr_offset + col_ptr_bytes
    row_idx_bytes = nnz * 4
    values_offset_legacy = row_idx_offset + row_idx_bytes
    values_offset_padded = (values_offset_legacy + 7) & ~7
    values_bytes = nnz * 8
    expected_legacy = values_offset_legacy + values_bytes
    expected_padded = values_offset_padded + values_bytes
    if file_size == expected_padded:
        values_offset = values_offset_padded
    elif file_size == expected_legacy:
        values_offset = values_offset_legacy
    else:
        raise RuntimeError(
            f"Sparse GRM CSC file size mismatch for {jxgrm_path}: got {file_size}, "
            f"expected {expected_legacy} (legacy) or {expected_padded} (padded)."
        )

    col_ptr = np.memmap(
        jxgrm_path,
        mode="r",
        dtype="<u8",
        offset=col_ptr_offset,
        shape=(n_samples + 1,),
        order="C",
    )
    row_idx = np.memmap(
        jxgrm_path,
        mode="r",
        dtype="<u4",
        offset=row_idx_offset,
        shape=(nnz,),
        order="C",
    )
    values = np.memmap(
        jxgrm_path,
        mode="r",
        dtype="<f8",
        offset=values_offset,
        shape=(nnz,),
        order="C",
    )

    if sample_idx is None:
        sample_idx_arg = np.arange(n_samples, dtype=np.int64)
    else:
        sample_idx_arg = np.ascontiguousarray(
            np.asarray(sample_idx, dtype=np.int64).reshape(-1),
            dtype=np.int64,
        )
        if np.any(sample_idx_arg < 0) or np.any(sample_idx_arg >= n_samples):
            raise ValueError(
                f"SparseLMM sparse GRM subset sample index out of bounds for {jxgrm_path}: "
                f"n_samples={n_samples}"
            )

    n_sub = int(sample_idx_arg.shape[0])
    dense = np.zeros((n_sub, n_sub), dtype=np.float64, order="C")
    pos_map = np.full(n_samples, -1, dtype=np.int64)
    pos_map[sample_idx_arg] = np.arange(n_sub, dtype=np.int64)

    total = max(1, n_sub)
    last_emit = -1
    if progress_callback is not None:
        progress_callback(0, total)
    for j_sub, col_full in enumerate(sample_idx_arg.tolist()):
        start = int(col_ptr[col_full])
        end = int(col_ptr[col_full + 1])
        if end > start:
            rows = row_idx[start:end]
            vals = values[start:end]
            mapped = pos_map[rows]
            keep = mapped >= 0
            if np.any(keep):
                tgt = np.asarray(mapped[keep], dtype=np.intp)
                dat = np.asarray(vals[keep], dtype=np.float64)
                dense[tgt, j_sub] = dat
                dense[j_sub, tgt] = dat
        done = j_sub + 1
        if progress_callback is not None and (done == total or done >= last_emit + 32):
            progress_callback(done, total)
            last_emit = done
    return np.ascontiguousarray(dense, dtype=np.float64)


def _jxlmm_sparse_grm_diag_stats(
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


def _jxlmm_sparse_lambda_boundary_flag(
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


def _jxlmm_exact_null_fit_from_grm(
    *,
    grm: np.ndarray,
    y_vec: np.ndarray,
    x_cov: Union[np.ndarray, None],
    threads: int,
    stage_label: str,
    progress_callback=None,
    grm_secs: float = 0.0,
    strategy: str = "fastlmm_exact_spectral",
    pve_mode: str = "spectral",
    keep_buffers: bool = True,
    extra_fields: Union[dict[str, object], None] = None,
) -> dict[str, object]:
    if not hasattr(jxrs, "lmm_reml_null_f32"):
        raise RuntimeError("Rust extension missing lmm_reml_null_f32 required by SparseLMM exact null fit.")

    y_arg = np.ascontiguousarray(np.asarray(y_vec, dtype=np.float64).reshape(-1), dtype=np.float64)
    grm_arr = np.ascontiguousarray(np.asarray(grm, dtype=np.float64), dtype=np.float64)
    n = int(y_arg.shape[0])
    if grm_arr.shape != (n, n):
        raise ValueError(f"SparseLMM exact null GRM shape mismatch: got {grm_arr.shape}, expected ({n}, {n}).")
    grm_local = np.array(grm_arr, dtype=np.float64, copy=True, order="C")
    np.fill_diagonal(grm_local, np.diag(grm_local) + 1e-6)

    progress_total = 1000
    if progress_callback is not None:
        progress_callback(0, progress_total)

    evd_t0 = time.monotonic()
    eigvals, eigvecs, evd_backend, evd_secs = _gwas_eigh_from_grm(
        grm_local,
        threads=max(1, int(threads)),
        logger=None,
        stage_label=stage_label,
        require_rust=True,
    )
    evd_secs = max(float(evd_secs), max(time.monotonic() - evd_t0, 0.0))
    if progress_callback is not None:
        progress_callback(820, progress_total)
    ord_desc = np.argsort(np.asarray(eigvals, dtype=np.float64))[::-1]
    eigvals = np.ascontiguousarray(np.asarray(eigvals, dtype=np.float64).reshape(-1)[ord_desc], dtype=np.float64)
    eigvecs = np.ascontiguousarray(np.asarray(eigvecs, dtype=np.float64)[:, ord_desc], dtype=np.float64)
    if eigvecs.shape != (n, n):
        raise ValueError(f"SparseLMM exact null eigenvector shape mismatch: got {eigvecs.shape}, expected ({n}, {n}).")

    x_design = _jxlmm_design_with_intercept(x_cov, n)
    s_null = np.ascontiguousarray(np.maximum(np.asarray(eigvals, dtype=np.float64), 0.0), dtype=np.float64)
    with _gwas_evd_stage_ctx(max(1, int(threads))):
        utx = np.ascontiguousarray(eigvecs.T @ x_design, dtype=np.float64)
        uty = np.ascontiguousarray((eigvecs.T @ y_arg).reshape(-1), dtype=np.float64)
    if progress_callback is not None:
        progress_callback(900, progress_total)
    lbd, ml, reml = jxrs.lmm_reml_null_f32(
        s_null,
        utx,
        uty,
        -5.0,
        5.0,
        50,
        1e-3,
    )
    sigma_g2, sigma_e2 = _jxlmm_exact_profile_vc(
        eigvals=s_null,
        utx=utx,
        uty=uty,
        lbd=float(lbd),
    )
    if progress_callback is not None:
        progress_callback(progress_total, progress_total)

    if str(pve_mode).strip().lower() == "components":
        pve = _jxlmm_component_ratio(sigma_g2, sigma_e2)
    else:
        pve = _jxlmm_pve_from_components(eigvals, sigma_g2, sigma_e2)

    out: dict[str, object] = {
        "strategy": str(strategy),
        "sigma_g2": sigma_g2,
        "sigma_e2": sigma_e2,
        "pve": pve,
        "lambda": float(lbd),
        "ml": float(ml),
        "reml": float(reml),
        "converged": True,
        "used_iter": 0,
        "grm_secs": float(max(grm_secs, 0.0)),
        "evd_secs": float(evd_secs),
        "backend": str(evd_backend),
    }
    if keep_buffers:
        out["eigvals"] = eigvals
        out["eigvecs"] = eigvecs
        out["grm"] = grm_local
    if extra_fields:
        out.update(extra_fields)
    return out


def _jxlmm_null_progress_stage(route_desc: str, done: int, total: int) -> str:
    route = str(route_desc).strip() or "null fit"
    total_use = int(max(1, total))
    frac = float(max(0, min(int(done), total_use))) / float(total_use)
    route_l = route.lower()
    if "sparse" in route_l:
        if frac < 0.12:
            stage = "analyze sparse CSC"
        elif frac < 0.55:
            stage = "grid lambda search"
        else:
            stage = "Brent REML refine"
    else:
        if frac < 0.60:
            stage = "build dense GRM"
        elif frac < 0.82:
            stage = "eigendecompose GRM"
        elif frac < 0.90:
            stage = "rotate X/y"
        else:
            stage = "Brent variance fit"
    return stage


def _jxlmm_null_progress_desc(route_desc: str, done: int, total: int) -> str:
    route = str(route_desc).strip() or "null fit"
    stage = _jxlmm_null_progress_stage(route_desc, done, total)
    return f"SparseLMM null: {route} / {stage}"


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


def _jxlmm_fastlmm_profile_vc(
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


def _jxlmm_exact_profile_vc(
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
    beta = _jxlmm_solve_linear(xtv_inv_x, xtv_inv_y)
    resid = y - (x @ beta)
    rtv_invr = float(np.dot(v_inv, np.square(resid)))
    if not np.isfinite(rtv_invr) or rtv_invr <= 0.0:
        return float("nan"), float("nan")
    sigma_g2 = float(rtv_invr / float(n_minus_p))
    sigma_e2 = float(lbd_f * sigma_g2)
    return sigma_g2, sigma_e2


def _jxlmm_design_with_intercept(
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


def _jxlmm_component_scale_reml(
    *,
    y_vec: np.ndarray,
    x_cov: Union[np.ndarray, None],
    sigma_g2: float,
    sigma_e2: float,
) -> dict[str, float]:
    y = np.ascontiguousarray(np.asarray(y_vec, dtype=np.float64).reshape(-1), dtype=np.float64)
    n = int(y.shape[0])
    x_design = _jxlmm_design_with_intercept(x_cov, n)
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
    beta = _jxlmm_solve_linear(xtx, xty)
    resid = np.ascontiguousarray(y - (x_design @ beta), dtype=np.float64)
    rss_mx = float(np.dot(resid, resid))
    df = int(n - p)
    # Under the explicit P0 + sigma^2 parameterization:
    #   sigma2_profile = y'P0y / (n - p)
    #   sigma2_mx      = y'M_Xy / (n - p)
    # For standardized K, V = sigma^2 (K + lambda I), so the approximate scan
    # should use sigma2_scan = sigma2_mx / (1 + lambda)
    #                           = sigma2_mx / ((sigma_g2 + sigma_e2) / sigma_g2).
    # Because Rust still receives (sigma_g2, sigma_e2), the effective factor we
    # apply to both components is:
    #   scan_scale = sigma2_scan / sigma2_profile = sigma2_mx / (sigma_g2 + sigma_e2).
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


def _jxlmm_solve_linear(lhs: np.ndarray, rhs: np.ndarray, *, ridge: float = 1e-10) -> np.ndarray:
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
    sparse_diag = _jxlmm_sparse_grm_diag_stats(str(jxgrm_path), sample_idx_arg)
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
            grid_size=int(_JXLMM_SPARSE_REML_GRID_SIZE),
            tol=1e-3,
            max_iter=int(_JXLMM_SPARSE_REML_MAX_ITER),
            threads=int(threads_use),
        )
        fit_secs = max(time.monotonic() - fit_t0, 0.0)
        lbd, sigma_g2, sigma_e2, ml, reml, log10_lambda, df_reml = tuple(out)
        sigma_g2 = float(sigma_g2)
        sigma_e2 = float(sigma_e2)
        sigma_sum = float(sigma_g2 + sigma_e2)
        pve = _jxlmm_component_ratio(sigma_g2, sigma_e2)
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
            "lambda_boundary": _jxlmm_sparse_lambda_boundary_flag(float(log10_lambda), -5.0, 5.0),
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

    if (
        int(_JXLMM_SPARSE_NULL_CLIP_EIG_EXACT_N_MAX) > 0
        and n_samples_null <= int(_JXLMM_SPARSE_NULL_CLIP_EIG_EXACT_N_MAX)
    ):
        fit_t0 = time.monotonic()
        progress_total = 1000
        if progress_callback is not None:
            progress_callback(0, progress_total)

        def _dense_progress(done: int, total: int) -> None:
            if progress_callback is None:
                return
            try:
                d = int(done)
                t = int(total)
            except Exception:
                return
            if t <= 0:
                return
            mapped = int((max(0, min(d, t)) * 320) / max(1, t))
            progress_callback(mapped, progress_total)

        grm_t0 = time.monotonic()
        grm_dense = _jxlmm_load_sparse_grm_subset_dense(
            str(jxgrm_path),
            sample_idx=sample_idx_arg,
            progress_callback=_dense_progress if progress_callback is not None else None,
        )
        grm_secs = max(time.monotonic() - grm_t0, 0.0)

        def _exact_progress(done: int, total: int) -> None:
            if progress_callback is None:
                return
            try:
                d = int(done)
                t = int(total)
            except Exception:
                return
            if t <= 0:
                return
            mapped = 320 + int((max(0, min(d, t)) * 680) / max(1, t))
            progress_callback(mapped, progress_total)

        out = _jxlmm_exact_null_fit_from_grm(
            grm=grm_dense,
            y_vec=y_arg,
            x_cov=x_arg,
            threads=max(1, int(detect_effective_threads())),
            stage_label="SparseLMM sparse exact null eigendecompose clipped sparse subset",
            progress_callback=_exact_progress if progress_callback is not None else None,
            grm_secs=float(grm_secs),
            strategy="sparse_exact_spectral_clipped",
            pve_mode="components",
            keep_buffers=False,
        )
        fit_secs = max(time.monotonic() - fit_t0, 0.0)
        lbd = float(out.get("lambda", float("nan")))
        log10_lambda = float(np.log10(max(lbd, 1e-30))) if np.isfinite(lbd) and lbd > 0.0 else float("nan")
        out.update(
            _jxlmm_component_scale_reml(
                y_vec=y_arg,
                x_cov=x_arg,
                sigma_g2=float(out.get("sigma_g2", float("nan"))),
                sigma_e2=float(out.get("sigma_e2", float("nan"))),
            )
        )
        out.update(
            {
                "fit_secs": float(fit_secs),
                "mean_diag_k": mean_diag_k,
                "min_diag_k": float(sparse_diag.get("min_diag", float("nan"))),
                "max_diag_k": float(sparse_diag.get("max_diag", float("nan"))),
                "n_samples_k": float(n_samples_k),
                "nnz_k": float(nnz_k),
                "offdiag_nnz_k": float(offdiag_nnz),
                "offdiag_density_k": float(offdiag_density),
                "log10_lambda": log10_lambda,
                "lambda_boundary": _jxlmm_sparse_lambda_boundary_flag(log10_lambda, -5.0, 5.0),
            }
        )
        if progress_callback is not None:
            progress_callback(progress_total, progress_total)
        return out

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
        grid_size=int(_JXLMM_SPARSE_REML_GRID_SIZE),
        tol=1e-3,
        max_iter=int(_JXLMM_SPARSE_REML_MAX_ITER),
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
    pve = _jxlmm_component_ratio(sigma_g2, sigma_e2)
    var_g_diag_scaled = float(sigma_g2) * max(mean_diag_k, 0.0) if np.isfinite(mean_diag_k) else float("nan")
    denom_diag_scaled = var_g_diag_scaled + sigma_e2
    pve_diag_scaled = (
        float(var_g_diag_scaled / denom_diag_scaled)
        if np.isfinite(denom_diag_scaled) and denom_diag_scaled > 0.0
        else float("nan")
    )
    lambda_boundary = _jxlmm_sparse_lambda_boundary_flag(float(log10_lambda), low, high)
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
        _jxlmm_component_scale_reml(
            y_vec=y_arg,
            x_cov=x_arg,
            sigma_g2=sigma_g2,
            sigma_e2=sigma_e2,
        )
    )
    return out_dict


_jxlmm_bed_logic_meta_selected = _splmm_bed_logic_meta_selected
_jxlmm_null_components_valid = _splmm_null_components_valid
_jxlmm_sparse_grid_local_summary = _splmm_sparse_grid_local_summary
_write_jxlmm_assoc_tsv = _write_splmm_assoc_tsv
_jxlmm_sparse_null_fit = _splmm_sparse_null_fit


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

    if int(len(sites_all)) != int(packed_row_idx.shape[0]):
        raise ValueError(
            f"Packed/BIM mismatch after filtering: sites={len(sites_all)} active_rows={packed_row_idx.shape[0]}"
        )
    chrom_all, pos_all, allele0_all, allele1_all, snp_all = sites_all.columns()

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
        gwas_total = int(len(sites_all))
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
                    _written_rows = _packed_lowrank_scan_to_tsv(
                        mod=null_fast_mod,
                        packed=packed,
                        packed_n=int(packed_n),
                        packed_row_idx=packed_row_idx,
                        row_flip=row_flip,
                        maf=maf,
                        miss=miss,
                        sites_all=sites_all,
                        sample_idx_trait=sample_idx_trait,
                        out_tsv=str(tmp_tsv),
                        chunk_size=int(max(1, int(chunk_size))),
                        threads=int(max(1, int(threads))),
                        genetic_model=str(genetic_model),
                        progress_callback=_fastlmm_progress if gwas_pbar is not None else None,
                        )
                    if int(_written_rows) != int(len(sites_all)):
                        _emit_warning_line(
                            logger,
                            f"{_route_model_label} low-rank writer row mismatch: expected={len(sites_all)} wrote={int(_written_rows)}",
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
                                chrom_all,
                                pos_all,
                                snp_all,
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
                            chrom_all,
                            pos_all,
                            snp_all,
                            allele0_all,
                            allele1_all,
                            tmp_tsv,
                            row_indices=packed_row_idx,
                            fixed_lbd=(float(fixed_lbd) if fixed_lbd is not None else None),
                            fixed_ml0=None,
                            rotate_block_rows=int(max(1, int(chunk_size))),
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

        eff_snp = int(len(sites_all))
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
    use_block_lm = hasattr(jxrs, "lm_block_assoc_packed_to_tsv")
    if not hasattr(jxrs, "glmf32_packed_assoc_to_tsv") and not use_block_lm:
        raise RuntimeError(
            "Rust extension missing both glmf32_packed_assoc_to_tsv and "
            "lm_block_assoc_packed_to_tsv. Rebuild/install JanusX extension first."
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

    if int(len(sites_all)) != int(packed_row_idx.shape[0]):
        raise ValueError(
            f"Packed/BIM mismatch after filtering: sites={len(sites_all)} active_rows={packed_row_idx.shape[0]}"
        )
    chrom_all, pos_all, allele0_all, allele1_all, snp_all = sites_all.columns()

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
        gwas_total = int(len(sites_all))
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

                if use_block_lm:
                    # New block formula with per-trait filtering
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
                        chrom_all,
                        pos_all,
                        snp_all,
                        allele0_all,
                        allele1_all,
                        tmp_tsv,
                        maf_threshold=float(maf_threshold),
                        max_missing_rate=float(max_missing_rate),
                        het_threshold=float(het_threshold),
                        sample_indices=sample_idx_trait,
                        row_indices=packed_row_idx,
                        chunk_size=int(max(1, int(chunk_size))),
                        threads=int(threads),
                        **kwargs_assoc,
                    )
                    _written_rows = int(_kept)
                else:
                    # Legacy path: unified dispatcher or glmf32_packed_assoc_to_tsv
                    unified_done = False
                    if _gwas_use_rust_unified_v1():
                        try:
                            jobs = [
                                {
                                    "model": "lm",
                                    "trait": str(pname),
                                    "out_tsv": str(tmp_tsv),
                                    "y": y_vec,
                                    "x": x_design,
                                    "ixx": None,
                                    "sample_indices": sample_idx_trait,
                                    "step": int(max(1, int(chunk_size))),
                                    "scan_progress_callback": _lm_progress,
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
                                chrom_all,
                                pos_all,
                                snp_all,
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
                                miss,
                                chrom_all,
                                pos_all,
                                snp_all,
                                allele0_all,
                                allele1_all,
                                tmp_tsv,
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
                                miss,
                                chrom_all,
                                pos_all,
                                snp_all,
                                allele0_all,
                                allele1_all,
                                tmp_tsv,
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
                except Exception:
                    pass
                gwas_pbar.close(show_done=False)
            if not gwas_ok:
                _cleanup_gwas_result_tmp(tmp_tsv)

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

        eff_snp = int(_written_rows) if use_block_lm else int(len(sites_all))
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

        model_chunk_size = _resolve_stream_scan_chunk_size(
            int(chunk_size),
            int(n_snps),
            use_spinner=bool(use_spinner),
            n_samples_hint=int(n_idv),
            model_keys="lm",
            user_specified=bool(chunk_size_user_set),
        )
        mmap_window_mb = (
            auto_mmap_window_mb(genofile, len(ids), n_snps, _current_bed_memory_mb())
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
        pname_tag = _safe_trait_file_label(pname)
        if gm_tag == "add":
            out_tsv = f"{outprefix}.{pname_tag}.lm.tsv"
        else:
            out_tsv = f"{outprefix}.{pname_tag}.{gm_tag}.lm.tsv"
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
                desc="LM",
                force_animate=True,
                logger=logger,
            )
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
                pbar = _ProgressAdapter(
                    total=total_use,
                    desc="LM",
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
            _cleanup_gwas_result_tmp(tmp_tsv)
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

        _finalize_gwas_result_tsv(tmp_tsv, out_tsv, prefix, logger=logger)
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
    force_model: bool = False,
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
        gwas_total = int(len(sites_all))
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
        chrom_all, pos_all, allele0_all, allele1_all, snp_all = sites_all.columns()
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
                    _written_rows = _packed_lowrank_scan_to_tsv(
                        mod=mod,
                        packed=packed,
                        packed_n=int(packed_n),
                        packed_row_idx=packed_row_idx,
                        row_flip=row_flip,
                        maf=maf,
                        miss=miss,
                        sites_all=sites_all,
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
                        chrom_all,
                        pos_all,
                        snp_all,
                        allele0_all,
                        allele1_all,
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
                except Exception:
                    pass
                gwas_pbar.close(show_done=False)
            if not gwas_ok:
                _cleanup_gwas_result_tmp(tmp_tsv)

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
) -> None:
    if str(genetic_model).lower() != "add":
        raise ValueError("ALGWAS full-rust packed route currently supports additive coding only (--model add).")
    if not hasattr(jxrs, "algwas_packed_to_tsv") and not _gwas_use_rust_unified_v1():
        raise RuntimeError(
            "Rust extension missing ALGWAS packed GWAS symbols. "
            "Rebuild/install JanusX extension first."
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
        sample_map = np.asarray(
            [id_to_idx[str(sid)] for sid in np.asarray(ids, dtype=str)],
            dtype=np.int64,
        )
    except KeyError as e:
        raise ValueError("Some aligned sample IDs are not present in packed BED sample order.") from e

    packed_mode = str(packed_ctx.get("packed_filter_mode", "compact")).strip().lower()
    packed_raw = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
    packed_n = int(packed_ctx["n_samples"])
    maf_raw = np.ascontiguousarray(
        np.asarray(packed_ctx.get("af", packed_ctx["maf"]), dtype=np.float32).reshape(-1),
        dtype=np.float32,
    )
    miss_raw = np.ascontiguousarray(np.asarray(packed_ctx["missing_rate"], dtype=np.float32).reshape(-1), dtype=np.float32)
    row_flip_raw = np.ascontiguousarray(
        np.asarray(
            packed_ctx.get(
                "row_flip",
                np.zeros((int(packed_raw.shape[0]),), dtype=np.bool_),
            ),
            dtype=np.bool_,
        ).reshape(-1),
        dtype=np.bool_,
    )
    active_row_idx_raw = np.ascontiguousarray(
        np.asarray(
            packed_ctx.get(
                "active_row_idx",
                np.arange(int(packed_raw.shape[0]), dtype=np.int64),
            ),
            dtype=np.int64,
        ).reshape(-1),
        dtype=np.int64,
    )
    if packed_mode == "lazy_full":
        packed = np.ascontiguousarray(packed_raw[active_row_idx_raw], dtype=np.uint8)
        maf = np.ascontiguousarray(maf_raw[active_row_idx_raw], dtype=np.float32)
        miss = np.ascontiguousarray(miss_raw[active_row_idx_raw], dtype=np.float32)
        row_flip = np.ascontiguousarray(row_flip_raw[active_row_idx_raw], dtype=np.bool_)
    else:
        packed = packed_raw
        maf = maf_raw
        miss = miss_raw
        row_flip = row_flip_raw
        active_row_idx_raw = np.arange(int(packed.shape[0]), dtype=np.int64)

    packed_row_idx = active_row_idx_raw
    if packed_n <= 0:
        raise ValueError("Packed BED reported invalid sample count.")
    if (
        int(packed_row_idx.shape[0]) != int(maf.shape[0])
        or int(packed_row_idx.shape[0]) != int(miss.shape[0])
        or int(packed_row_idx.shape[0]) != int(row_flip.shape[0])
    ):
        raise ValueError("Packed BED arrays have inconsistent SNP dimensions.")
    if int(len(sites_all)) != int(packed_row_idx.shape[0]):
        raise ValueError(
            f"Packed/BIM mismatch after filtering: sites={len(sites_all)} active_rows={packed_row_idx.shape[0]}"
        )

    chrom_all, pos_all, allele0_all, allele1_all, snp_all = sites_all.columns()

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
        gwas_total = int(len(sites_all))
        gwas_last_done = 0
        gwas_pbar: Optional[_ProgressAdapter] = None

        def _algwas_stage1_progress(done: int, total: int) -> None:
            nonlocal stage1_last_done, stage1_pbar, stage1_spin_handle, peak_rss
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
                stage1_pbar.finish()
                stage1_pbar.close(show_done=False)
                stage1_pbar = None
            try:
                peak_rss = max(peak_rss, process.memory_info().rss)
            except Exception:
                pass

        def _algwas_progress(done: int, total: int) -> None:
            nonlocal gwas_last_done, gwas_total, gwas_pbar, peak_rss, stage1_pbar, stage1_spin_handle
            try:
                d = int(done)
                t = int(total)
            except Exception:
                return
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
        gwas_ok = False
        gwas_t0 = time.monotonic()
        try:
            with _gwas_scan_stage_ctx(max(1, int(threads))):
                unified_done = False
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
                                "scan_progress_callback": _algwas_progress,
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
                            chrom_all,
                            pos_all,
                            snp_all,
                            allele0_all,
                            allele1_all,
                            int(max(1, int(chunk_size))),
                            int(threads),
                            None,
                            int(max(1, int(chunk_size))),
                            row_indices=None,
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
                        chrom_all,
                        pos_all,
                        snp_all,
                        allele0_all,
                        allele1_all,
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
                        progress_callback=_algwas_progress,
                        pseudo_tsv=str(pseudo_tsv_hint),
                        row_indices=None,
                    )
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

        eff_snp = int(len(sites_all))
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
            f"ALGWAS ...Found {int(qtn_count)} QTNs [{'/'.join(done_times)}]",
            use_spinner=bool(use_spinner),
        )
        if multi_trait_mode:
            logger.info("")


def run_splmm_packed_fullrank(
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
        raise ValueError("SparseLMM full-rust packed route currently supports additive coding only (--model add).")
    if not hasattr(jxrs, "splmm_assoc_pcg_bed"):
        raise RuntimeError(
            "Rust extension missing splmm_assoc_pcg_bed. Rebuild/install JanusX extension first."
        )
    scan_mode_norm = str(scan_mode).strip().lower()
    if scan_mode_norm not in {"two_stage", "approx", "exact"}:
        raise ValueError(f"Unsupported SparseLMM scan mode: {scan_mode}")
    approx_legacy_backend = bool(
        scan_mode_norm == "approx" and _splmm_approx_legacy_backend()
    )
    if scan_mode_norm == "approx" and (not approx_legacy_backend):
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
        if int(len(sites_all_packed)) != int(scan_row_idx.shape[0]):
            raise ValueError(
                f"Packed/BIM mismatch after filtering: sites={len(sites_all_packed)} active_rows={scan_row_idx.shape[0]}"
            )
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

        def _stop_prepare_handle() -> None:
            nonlocal prepare_handle
            if prepare_handle is not None:
                try:
                    _stop_indeterminate_progress_bar(prepare_handle)
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
                residualized_approx=bool(scan_mode_norm == "approx" and (not approx_legacy_backend)),
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
            and (not approx_legacy_backend)
            and str(null_fit.get("strategy", "")).strip().lower() == "residualized_sparse_reml_brent"
        )
        scan_scale_active = bool(
            scan_mode_norm == "two_stage"
            or (
                scan_mode_norm == "approx"
                and (not approx_legacy_backend)
                and (not residualized_approx_null)
            )
        )
        if not _splmm_null_components_valid(sigma_g2, sigma_e2):
            _stop_prepare_handle()
            raise RuntimeError(
                f"SparseLMM null variance estimation returned invalid components for trait {pname}: "
                f"sigma_g2={sigma_g2}, sigma_e2={sigma_e2}"
            )
        sigma_g2_scan_preview = float("nan")
        sigma_e2_scan_preview = float("nan")
        scan_sigma_preview_valid = False
        if scan_scale_active and (
            np.isfinite(sigma_scale_reml)
            and sigma_scale_reml > 0.0
        ):
            sigma_g2_scan_preview = float(sigma_g2 * sigma_scale_reml)
            sigma_e2_scan_preview = float(sigma_e2 * sigma_scale_reml)
            scan_sigma_preview_valid = _splmm_null_components_valid(
                sigma_g2_scan_preview, sigma_e2_scan_preview
            )
        use_reml_scan_sigma = bool(scan_scale_active and scan_sigma_preview_valid)
        if use_reml_scan_sigma:
            sigma_g2_scan_arg = float(sigma_g2_scan_preview)
            sigma_e2_scan_arg = float(sigma_e2_scan_preview)
        else:
            sigma_g2_scan_arg = float(sigma_g2)
            sigma_e2_scan_arg = float(sigma_e2)
        if residualized_approx_null:
            sigma_g2_scan_use = float("nan")
            sigma_e2_scan_use = float("nan")
        else:
            sigma_g2_scan_use = float(sigma_g2_scan_arg)
            sigma_e2_scan_use = float(sigma_e2_scan_arg)
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
        if scan_scale_active:
            _null_fit_msg = (
                f"{_null_fit_msg}; "
                f"c_p0_to_mx={sigma_scale_p0_to_mx:.4g}, scan_scale={sigma_scale_reml:.4g}"
            )
        if scan_sigma_preview_valid and scan_scale_active:
            _null_fit_msg = (
                f"{_null_fit_msg}; "
                f"scan_sigma_preview=(sigma2={sigma_g2_scan_preview:.4g}, "
                f"sigma_g2={sigma_g2_scan_preview:.4g}, sigma_e2={sigma_e2_scan_preview:.4g})"
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
            run_lmm_packed_fullrank(
                genofile=genofile,
                pheno=pheno_trait_subset,
                ids=ids_trait_subset,
                grm=np.ascontiguousarray(np.asarray(null_fit["grm"], dtype=np.float64), dtype=np.float64),
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
                preloaded_packed=packed_preload_ready,
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
                if packed_preload_ready is not None:
                    run_lm_packed_fullrank(
                        genofile=genofile,
                        pheno=pheno_trait_subset,
                        ids=ids_trait_subset,
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
                        preloaded_packed=packed_preload_ready,
                        force_model=bool(force_model),
                    )
                else:
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
        mmap_window_mb = auto_mmap_window_mb(
            genofile,
            len(ids),
            n_trait_sites,
            _current_bed_memory_mb(),
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
                        float(sigma_g2_scan_arg),
                        float(sigma_e2_scan_arg),
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
                        float(sigma_g2_scan_arg),
                        float(sigma_e2_scan_arg),
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
        else:
            arr = np.ascontiguousarray(np.asarray(splmm_t[9], dtype=np.float64), dtype=np.float64)

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
                sites_use = _read_bim_site_columns(str(kinship_prefix), trait_row_idx)
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
            "two_stage": "two-stage approx->exact",
            "approx": "approximate GRAMMAR-gamma",
            "exact": "exact g'Pg",
        }[scan_mode_norm]
        _rhat_info = (
            f"r_hat={r_hat:.4g}, sampled {rhat_used}/{rhat_requested}, "
            if int(rhat_requested) > 0
            else ""
        )
        sigma2_scan_use = float("nan")
        scan_sigma_mode = "profile"
        if scan_mode_norm == "approx" and (not approx_legacy_backend) and residualized_approx_null:
            scan_sigma_mode = "residualized_reml_residualized_scan_rust_internal"
            _sigma_mode_info = (
                "sigma_mode=residualized_reml+residualized_scan("
                f"Rust internal scaling; profile sigma_g2={sigma_g2:.4g}, sigma_e2={sigma_e2:.4g}, "
                f"sampled={rhat_used}/{rhat_requested})"
            )
        elif scan_mode_norm == "approx" and (not approx_legacy_backend) and use_reml_scan_sigma:
            sigma2_scan_use = float(sigma_g2_scan_use)
            scan_sigma_mode = "mx_sigma2_over_1plambda_residualized_scan"
            _sigma_mode_info = (
                "sigma_mode=mx_sigma2_over_1plambda+residualized_scan("
                f"c_p0_to_mx={sigma_scale_p0_to_mx:.4g}, "
                f"scan_scale={sigma_scale_reml:.4g}, "
                f"sigma2={sigma2_scan_use:.4g}, sigma_g2={sigma_g2_scan_use:.4g}, "
                f"sigma_e2={sigma_e2_scan_use:.4g}, "
                f"sampled={rhat_used}/{rhat_requested})"
            )
        elif scan_mode_norm == "approx" and (not approx_legacy_backend):
            scan_sigma_mode = "profile_residualized_scan"
            _sigma_mode_info = (
                "sigma_mode=profile+residualized_scan; "
                f"Rust approx residualizes scan vectors with supplied profile null components, "
                f"sampled markers={rhat_used}/{rhat_requested}"
            )
        elif use_reml_scan_sigma:
            sigma2_scan_use = float(sigma_g2_scan_use)
            scan_sigma_mode = "mx_sigma2_over_1plambda"
            _sigma_mode_info = (
                f"sigma_mode=mx_sigma2_over_1plambda(c_p0_to_mx={sigma_scale_p0_to_mx:.4g}, "
                f"scan_scale={sigma_scale_reml:.4g}, "
                f"sigma2={sigma2_scan_use:.4g}, sigma_g2={sigma_g2_scan_use:.4g}, sigma_e2={sigma_e2_scan_use:.4g})"
            )
        elif scan_sigma_preview_valid and scan_scale_active:
            _sigma_mode_info = (
                f"sigma_mode=profile; scan_sigma_preview=mx_sigma2_over_1plambda(c_p0_to_mx={sigma_scale_p0_to_mx:.4g}, "
                f"scan_scale={sigma_scale_reml:.4g}, "
                f"sigma2={sigma_g2_scan_preview:.4g}, sigma_g2={sigma_g2_scan_preview:.4g}, sigma_e2={sigma_e2_scan_preview:.4g})"
            )
        else:
            _sigma_mode_info = "sigma_mode=profile"
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
                "splmm_scan_sigma2": (
                    None if residualized_approx_null else (
                        float(sigma2_scan_use) if np.isfinite(sigma2_scan_use) else None
                    )
                ),
                "splmm_scan_sigma_g2": (
                    None if residualized_approx_null else (
                        float(sigma_g2_scan_use) if np.isfinite(sigma_g2_scan_use) else None
                    )
                ),
                "splmm_scan_sigma_e2": (
                    None if residualized_approx_null else (
                        float(sigma_e2_scan_use) if np.isfinite(sigma_e2_scan_use) else None
                    )
                ),
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
