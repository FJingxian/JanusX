# -*- coding: utf-8 -*-
"""FarmCPU GWAS model runners (extracted from workflow.py)."""

from __future__ import annotations

import logging
import os
import time
from typing import Union

import numpy as np
import pandas as pd
import psutil
from janusx.script._common.genoio import (
    determine_genotype_source_force_kind_from_args,
    inspect_genotype_file,
    packed_preload_is_disabled,
    packed_preload_is_ready,
    prepare_packed_ctx_from_plink,
)
from janusx.script._common.memory import (
    bytes_to_gib,
    finalize_peak_memory_metrics,
    process_memory_info_bytes,
)

from .workflow import (
    CliStatus,
    _ProgressAdapter,
    _align_pheno_to_sample_order,
    _as_plink_prefix,
    _basename_only,
    _cache_lock,
    _cleanup_gwas_result_tmp,
    _emit_plain_info_line,
    _emit_trait_header,
    _emit_warning_line,
    _finalize_gwas_result_tsv,
    _grm_cache_paths,
    _grm_cache_paths_legacy,
    _gwas_cache_prefix_with_params,
    _gwas_eigh_from_grm,
    _gwas_result_tmp_path,
    _inspect_genotype_with_status,
    _load_covariates_for_models,
    _load_pca_cache_with_ids,
    _load_phenotype_with_status,
    _log_file_only,
    _normalize_cov_inputs,
    _parse_qcov_dim,
    _pca_cache_path,
    _pca_cache_path_legacy,
    _write_pca_cache_with_ids,
    _bim_metadata_len,
    _make_deferred_bim_metadata,
    _read_id_file,
    _resolve_trait_iter,
    _resolve_ref_alt_columns,
    _replace_file_with_retry,
    _resolve_file_input_matrix,
    _rich_success,
    _run_fastplot_from_tsv_with_status,
    _safe_trait_file_label,
    _trait_values_and_mask,
    detect_effective_threads,
    format_elapsed,
    genotype_cache_prefix,
    jxrs,
    latest_genotype_mtime,
)

_FARMCPU_PACKED_ONLY_MSG = (
    "FarmCPU now requires packed PLINK BED genotype input; "
    "the dense fallback path has been removed."
)


def _parse_farmcpu_rust_tsv_result(
    value,
) -> tuple[int, int, int, Union[float, None], Union[float, None]]:
    if isinstance(value, (tuple, list)):
        if len(value) >= 5:
            return (
                int(value[0]),
                int(value[1]),
                int(value[2]),
                float(value[3]),
                float(value[4]),
            )
        if len(value) >= 3:
            return int(value[0]), int(value[1]), int(value[2]), None, None
    raise ValueError(f"Unexpected FarmCPU Rust result payload: {type(value).__name__}")


def _format_chisq_scalar(value: float) -> str:
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{float(value):.4e}"


def _format_chisq_output(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.asarray([_format_chisq_scalar(v) for v in arr], dtype=object)


def _format_farmcpu_done_elapsed(seconds: float) -> str:
    s = float(max(0.0, seconds))
    if s < 10.0:
        return f"{s:.2f}s"
    return format_elapsed(s)


def _normalize_packed_ctx_for_farmcpu_cache(
    packed_obj: dict[str, object],
) -> dict[str, object]:
    packed_num = np.ascontiguousarray(
        np.asarray(packed_obj["packed"], dtype=np.uint8)
    )
    packed_rows = int(packed_num.shape[0])
    packed_n = int(packed_obj["n_samples"])

    def _raw_vec(key: str, dtype) -> np.ndarray:
        return np.ascontiguousarray(
            np.asarray(packed_obj[key], dtype=dtype).reshape(-1),
            dtype=dtype,
        )

    miss_raw = _raw_vec("missing_rate", np.float32)
    maf_raw = np.ascontiguousarray(
        np.asarray(packed_obj.get("af", packed_obj["maf"]), dtype=np.float32).reshape(-1),
        dtype=np.float32,
    )

    row_flip_obj = packed_obj.get("row_flip")
    if row_flip_obj is None:
        row_flip_obj = np.zeros(packed_rows, dtype=np.bool_)
    row_flip_raw = np.ascontiguousarray(
        np.asarray(row_flip_obj, dtype=np.bool_).reshape(-1),
        dtype=np.bool_,
    )

    row_idx_obj = packed_obj.get("row_indices")
    if row_idx_obj is None:
        row_idx_obj = packed_obj.get("active_row_idx")
    if row_idx_obj is None:
        site_keep_obj = packed_obj.get("site_keep")
        if site_keep_obj is not None:
            site_keep_arr = np.asarray(site_keep_obj, dtype=np.bool_).reshape(-1)
            if int(site_keep_arr.shape[0]) == packed_rows:
                row_idx_obj = np.flatnonzero(site_keep_arr).astype(np.int64, copy=False)
    if row_idx_obj is None:
        default_rows = int(min(packed_rows, maf_raw.shape[0], miss_raw.shape[0], row_flip_raw.shape[0]))
        row_idx_obj = np.arange(default_rows, dtype=np.int64)

    row_idx = np.ascontiguousarray(
        np.asarray(row_idx_obj, dtype=np.int64).reshape(-1),
        dtype=np.int64,
    )
    if int(row_idx.shape[0]) == 0:
        raise ValueError("FarmCPU packed cache row index set is empty.")
    if int(row_idx.min()) < 0 or int(row_idx.max()) >= packed_rows:
        raise ValueError(
            "FarmCPU packed cache row_indices exceed packed marker bounds."
        )

    active_rows = int(row_idx.shape[0])

    def _active_view(arr: np.ndarray, name: str) -> np.ndarray:
        if int(arr.shape[0]) == active_rows:
            return arr
        if int(arr.shape[0]) == packed_rows:
            return np.ascontiguousarray(arr[row_idx], dtype=arr.dtype)
        raise ValueError(
            f"FarmCPU packed cache {name} length mismatch: "
            f"len={int(arr.shape[0])}, packed_rows={packed_rows}, active_rows={active_rows}."
        )

    miss_arr = _active_view(miss_raw, "missing_rate")
    maf_arr = _active_view(maf_raw, "maf")
    row_flip_arr = _active_view(row_flip_raw, "row_flip")

    site_keep_obj = packed_obj.get("site_keep")
    site_keep_norm: Union[np.ndarray, None]
    if site_keep_obj is not None:
        site_keep_arr = np.ascontiguousarray(
            np.asarray(site_keep_obj, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        if int(site_keep_arr.shape[0]) == packed_rows:
            site_keep_norm = site_keep_arr
        else:
            site_keep_norm = None
    elif active_rows == packed_rows:
        site_keep_norm = np.ones((packed_rows,), dtype=np.bool_)
    else:
        site_keep_norm = np.zeros((packed_rows,), dtype=np.bool_)
        site_keep_norm[row_idx] = True

    return {
        "packed": packed_num,
        "missing_rate": miss_arr,
        "maf": maf_arr,
        "row_flip": row_flip_arr,
        "row_indices": row_idx,
        "active_row_idx": row_idx,
        "site_keep": site_keep_norm,
        "packed_filter_mode": str(
            packed_obj.get(
                "packed_filter_mode",
                "lazy_full" if active_rows != packed_rows else "compact",
            )
        ),
        "n_samples": packed_n,
        "source_prefix": packed_obj.get("source_prefix"),
    }

def prepare_qk_and_filter(
    geno: np.ndarray,
    ref_alt: pd.DataFrame,
    maf_threshold: float,
    max_missing_rate: float,
    logger,
):
    """
    Legacy Python QK path is disabled in Rust-only GWAS mode.
    """
    raise RuntimeError(
        "prepare_qk_and_filter (Python QK fallback) is disabled in Rust-only GWAS mode."
    )


def build_qmatrix_farmcpu(
    genofile: str,
    gfile_prefix: str,
    geno: Union[np.ndarray, None],
    qdim: str,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    cov_inputs: Union[str, list[str], None],
    chunk_size: int,
    logger,
    sample_ids: Union[np.ndarray, None] = None,
    use_spinner: bool = False,
    quiet_terminal: bool = False,
    snps_only: bool = True,
    n_snps_hint: Union[int, None] = None,
    threads: int = 1,
    mmap_limit: bool = False,
    preloaded_packed: Union[dict[str, object], None] = None,
    packed_ctx_preloaded: Union[dict[str, object], None] = None,
    return_ids: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Build or load Q matrix for FarmCPU (PCs + optional covariates).
    Note: external Q file via -q is no longer supported; pass external
    covariate matrices via -c.
    """
    def _farm_log(msg: str) -> None:
        if bool(quiet_terminal):
            _log_file_only(logger, logging.INFO, str(msg))
        else:
            logger.info(str(msg))

    sid_arr = None if sample_ids is None else np.asarray(sample_ids, dtype=str)
    if sid_arr is not None:
        n = int(sid_arr.shape[0])
    elif isinstance(packed_ctx_preloaded, dict):
        n = int(packed_ctx_preloaded["n_samples"])
    else:
        raise ValueError(
            "FarmCPU Q-matrix build requires packed BED sample IDs/context. "
            "Dense fallback has been removed."
        )

    q_direct_state: dict[str, object] = {
        "q": None,
        "evd_backend": "",
        "evd_secs": 0.0,
    }
    q_ids: Union[np.ndarray, None] = None

    def _build_grm_from_packed_ctx_rust(
        packed_ctx_obj: dict[str, object],
        qdim: int = 0,
    ) -> np.ndarray:
        if not hasattr(jxrs, "grm_packed_f32"):
            raise RuntimeError("rust symbol grm_packed_f32 is unavailable.")

        packed = np.ascontiguousarray(
            np.asarray(packed_ctx_obj["packed"], dtype=np.uint8),
            dtype=np.uint8,
        )
        maf = np.ascontiguousarray(
            np.asarray(packed_ctx_obj.get("af", packed_ctx_obj["maf"]), dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
        packed_n = int(packed_ctx_obj["n_samples"])
        if packed_n != int(n):
            raise ValueError(
                f"Packed context sample size mismatch for FarmCPU-Q: packed={packed_n}, expected={n}."
            )

        row_flip_obj = packed_ctx_obj.get("row_flip", None)
        if row_flip_obj is None:
            row_flip_obj = np.zeros((int(packed.shape[0]),), dtype=np.bool_)
        row_flip = np.ascontiguousarray(
            np.asarray(row_flip_obj, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )

        if int(packed.shape[0]) != int(maf.shape[0]) or int(packed.shape[0]) != int(row_flip.shape[0]):
            raise ValueError(
                "Packed context dimension mismatch for FarmCPU-Q "
                f"(packed_rows={packed.shape[0]}, maf={maf.shape[0]}, row_flip={row_flip.shape[0]})."
            )

        pbar = _ProgressAdapter(
            total=int(max(1, int(packed.shape[0]))),
            desc="GRM (packed-bed)",
            force_animate=True,
            logger=logger,
        )
        process = psutil.Process()
        mem_tick_span = max(1, 10 * int(max(1, int(chunk_size))))
        progress_state = {"done": 0}

        def _on_grm_progress(done: int, _total: int) -> None:
            d = int(done)
            delta = d - int(progress_state["done"])
            if delta > 0:
                pbar.update(delta)
                progress_state["done"] = d
            if d % mem_tick_span == 0:
                mem = process.memory_info().rss / 1024**3
                pbar.set_postfix(memory=f"{mem:.2f}GB")

        qdim_use = int(max(0, int(qdim)))
        try:
            if qdim_use > 0 and hasattr(jxrs, "farmcpu_q_packed_grm_pca_f32"):
                grm_raw, q_raw, evd_backend, evd_secs = jxrs.farmcpu_q_packed_grm_pca_f32(
                    packed,
                    int(packed_n),
                    row_flip,
                    maf,
                    int(qdim_use),
                    sample_indices=None,
                    method=1,
                    block_cols=max(1, int(chunk_size)),
                    threads=max(1, int(threads)),
                    progress_callback=_on_grm_progress,
                    progress_every=max(1, int(chunk_size)),
                )
                q_direct = np.asarray(q_raw, dtype="float32")
                if q_direct.ndim != 2 or q_direct.shape != (int(n), int(qdim_use)):
                    raise ValueError(
                        "Rust packed FarmCPU-Q shape mismatch: "
                        f"got {q_direct.shape}, expected ({n},{qdim_use})."
                    )
                q_direct_state["q"] = q_direct
                q_direct_state["evd_backend"] = str(evd_backend)
                q_direct_state["evd_secs"] = float(evd_secs)
            else:
                grm_raw = jxrs.grm_packed_f32(
                    packed,
                    int(packed_n),
                    row_flip,
                    maf,
                    sample_indices=None,
                    method=1,
                    block_cols=max(1, int(chunk_size)),
                    threads=max(1, int(threads)),
                    progress_callback=_on_grm_progress,
                    progress_every=max(1, int(chunk_size)),
                )
            pbar.finish()
        finally:
            pbar.close(show_done=False)

        grm = np.asarray(grm_raw, dtype="float32")
        if grm.ndim != 2 or grm.shape[0] != grm.shape[1] or int(grm.shape[0]) != int(n):
            raise ValueError(
                f"Rust packed GRM shape mismatch for FarmCPU-Q: got {grm.shape}, expected ({n},{n})."
            )
        return grm

    def _load_or_build_grm_cache_for_pca() -> np.ndarray:
        grm_path, id_path = _grm_cache_paths(gfile_prefix, mgrm="1")
        legacy_grm_path, legacy_id_path = _grm_cache_paths_legacy(gfile_prefix, mgrm="1")
        with _cache_lock(grm_path):
            if (not os.path.exists(grm_path)) and os.path.exists(legacy_grm_path):
                try:
                    _replace_file_with_retry(legacy_grm_path, grm_path)
                    if os.path.exists(legacy_id_path) and (not os.path.exists(id_path)):
                        _replace_file_with_retry(legacy_id_path, id_path)
                    _farm_log(
                        f"* Migrated legacy GRM cache name: {legacy_grm_path} -> {grm_path}"
                    )
                except Exception:
                    pass
            cache_ready = os.path.exists(grm_path)
            if cache_ready:
                g_mtime = latest_genotype_mtime(genofile)
                k_mtime = os.path.getmtime(grm_path)
                if g_mtime is not None and g_mtime > k_mtime:
                    cache_ready = False
            if cache_ready:
                _farm_log(f"* Loading GRM cache for FarmCPU PCA from {grm_path}...")
                grm = np.load(grm_path, mmap_mode="r")
                if grm.size != n * n:
                    _farm_log(
                        f"GRM cache shape mismatch ({grm.size} elements) for sample size {n}; rebuilding."
                    )
                    cache_ready = False
                else:
                    grm = grm.reshape(n, n)
                    if sid_arr is not None and os.path.exists(id_path):
                        grm_ids = _read_id_file(id_path, logger, "GRM", use_spinner=use_spinner)
                        if grm_ids is None or len(grm_ids) != n:
                            _farm_log("GRM cache IDs are invalid; rebuilding GRM cache.")
                            cache_ready = False
                        else:
                            grm_ids = np.asarray(grm_ids, dtype=str)
                            sid = sid_arr
                            if np.array_equal(grm_ids, sid):
                                return np.asarray(grm, dtype="float32")
                            index = {s: i for i, s in enumerate(grm_ids)}
                            missing = [s for s in sid if s not in index]
                            if missing:
                                _farm_log(
                                    f"GRM cache missing {len(missing)} sample IDs; rebuilding GRM cache."
                                )
                                cache_ready = False
                            else:
                                ord_idx = [index[s] for s in sid]
                                grm = grm[np.ix_(ord_idx, ord_idx)]
                                return np.asarray(grm, dtype="float32")
                    else:
                        if sid_arr is not None and not os.path.exists(id_path):
                            _farm_log("GRM cache ID file not found; assuming genotype sample order.")
                        return np.asarray(grm, dtype="float32")
            if not cache_ready:
                for p in (grm_path, id_path, legacy_grm_path, legacy_id_path):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

                _farm_log("* Building GRM cache for FarmCPU PCA...")
                if isinstance(packed_ctx_preloaded, dict):
                    try:
                        grm = _build_grm_from_packed_ctx_rust(
                            packed_ctx_preloaded,
                            qdim=int(q_int),
                        )
                    except Exception as ex:
                        raise RuntimeError(
                            f"FarmCPU packed-Rust GRM build failed in Rust-only mode: {ex}"
                        ) from ex
                else:
                    raise RuntimeError(
                        "FarmCPU Q build requires packed BED context. "
                        "Dense fallback has been removed."
                    )
                tmp_grm = f"{grm_path}.tmp.{os.getpid()}.npy"
                np.save(tmp_grm, grm)
                _replace_file_with_retry(tmp_grm, grm_path)
                if sid_arr is not None:
                    tmp_id = f"{id_path}.tmp.{os.getpid()}"
                    pd.Series(sid_arr).to_csv(
                        tmp_id, sep="\t", index=False, header=False
                    )
                    _replace_file_with_retry(tmp_id, id_path)
                _farm_log(f"Cached GRM written to {grm_path}")
                return grm

    q_int = _parse_qcov_dim(qdim)
    if q_int >= n and q_int != 0:
        raise ValueError(
            f"Q/PC dimension out of range for FarmCPU: {q_int}. "
            f"valid=[0..{max(0, n-1)}]"
        )

    if q_int == 0:
        qmatrix = np.zeros((n, 0), dtype="float32")
        q_ids = None if sid_arr is None else np.asarray(sid_arr, dtype=str)
        _emit_warning_line(
            logger,
            "PC dimension set to 0; using empty Q matrix.",
            use_spinner=bool(use_spinner),
        )
    else:
        q_path = _pca_cache_path(gfile_prefix, mgrm="1", qdim=int(q_int))
        legacy_q_path = _pca_cache_path_legacy(gfile_prefix, mgrm="1", qdim=int(q_int))
        with _cache_lock(q_path):
            if (not os.path.isfile(q_path)) and os.path.isfile(legacy_q_path):
                _farm_log(
                    f"* Ignoring legacy PCA cache and rebuilding in new format: {legacy_q_path}"
                )
            cache_ready = os.path.isfile(q_path)
            if cache_ready:
                g_mtime = latest_genotype_mtime(genofile)
                q_mtime = os.path.getmtime(q_path)
                if g_mtime is not None and g_mtime > q_mtime:
                    cache_ready = False
            if cache_ready:
                q_src = _basename_only(q_path)
                with CliStatus(
                    f"Loading Q matrix from {q_src}...",
                    enabled=bool(use_spinner),
                ) as task:
                    try:
                        q_ids, qmatrix = _load_pca_cache_with_ids(
                            q_path,
                            expected_rows=int(n),
                            expected_dim=int(q_int),
                            expected_ids=sid_arr,
                        )
                    except Exception as ex:
                        task.fail(f"Loading Q matrix from {q_src} ...Failed")
                        cache_ready = False
                        _emit_warning_line(
                            logger,
                            f"PCA cache format/content invalid; rebuilding cache. path={q_src}, reason={ex}",
                            use_spinner=bool(use_spinner),
                        )
                    if cache_ready:
                        task.complete(
                            f"Loading Q matrix from {q_src} (n={qmatrix.shape[0]}, nPC={qmatrix.shape[1]}) ...Finished"
                        )
            if not cache_ready:
                _farm_log(f"* PCA dimension for FarmCPU Q matrix: {q_int}")
                grm = _load_or_build_grm_cache_for_pca()
                q_direct_obj = q_direct_state.get("q")
                if isinstance(q_direct_obj, np.ndarray):
                    qmatrix = np.asarray(q_direct_obj, dtype="float32")
                    if qmatrix.shape != (n, int(q_int)):
                        raise ValueError(
                            "FarmCPU-Q packed Rust direct output shape mismatch: "
                            f"got {qmatrix.shape}, expected ({n},{q_int})."
                        )
                    evd_backend = str(q_direct_state.get("evd_backend", "unknown"))
                    evd_secs = float(q_direct_state.get("evd_secs", 0.0))
                    _log_file_only(
                        logger,
                        logging.INFO,
                        f"FarmCPU-Q-build packed Rust single-entry eig backend={evd_backend} "
                        f"elapsed={evd_secs:.3f}s",
                    )
                else:
                    _eigval, eigvec, _evd_backend, _evd_secs = _gwas_eigh_from_grm(
                        grm,
                        threads=max(1, int(threads)),
                        logger=logger,
                        stage_label="FarmCPU-Q-build",
                        require_rust=True,
                    )
                    qmatrix = np.asarray(eigvec[:, -q_int:], dtype="float32")
                q_ids = None if sid_arr is None else np.asarray(sid_arr, dtype=str)
                tmp_q = f"{q_path}.tmp.{os.getpid()}"
                if sid_arr is None:
                    raise ValueError("FarmCPU PCA cache write requires sample IDs.")
                _write_pca_cache_with_ids(tmp_q, sid_arr, qmatrix)
                _replace_file_with_retry(tmp_q, q_path)
                _farm_log(f"Cached PCA written to {q_path}")

    if cov_inputs:
        if sid_arr is None:
            raise ValueError("FarmCPU covariate loading requires sample IDs.")
        cov_arr, cov_ids = _load_covariates_for_models(
            cov_inputs=cov_inputs,
            genofile=genofile,
            sample_ids=sid_arr,
            chunk_size=int(chunk_size),
            logger=logger,
            context="FarmCPU",
            use_spinner=bool(use_spinner),
            snps_only=bool(snps_only),
        )
        if cov_arr is not None:
            if cov_ids is None:
                raise ValueError("Internal error: covariate IDs are missing for FarmCPU.")
            cov_ids_arr = np.asarray(cov_ids, dtype=str)
            if q_ids is None:
                q_ids = np.asarray(sid_arr, dtype=str)
            q_ids_arr = np.asarray(q_ids, dtype=str)
            cov_id_set = set(cov_ids_arr.tolist())
            common_ids = [sid for sid in q_ids_arr.tolist() if sid in cov_id_set]
            if len(common_ids) == 0:
                raise ValueError("No overlapping samples between FarmCPU PCA cache and covariates.")
            q_index = {sid: i for i, sid in enumerate(q_ids_arr.tolist())}
            cov_index = {sid: i for i, sid in enumerate(cov_ids_arr.tolist())}
            q_take = np.asarray([q_index[sid] for sid in common_ids], dtype=np.int64)
            cov_take = np.asarray([cov_index[sid] for sid in common_ids], dtype=np.int64)
            qmatrix = np.ascontiguousarray(qmatrix[q_take], dtype="float32")
            cov_use = np.ascontiguousarray(cov_arr[cov_take], dtype="float32")
            qmatrix = np.concatenate([qmatrix, cov_use], axis=1)
            q_ids = np.asarray(common_ids, dtype=str)
    else:
        _emit_warning_line(
            logger,
            "Loading covariates (streaming) ...Skipped (none)",
            use_spinner=bool(use_spinner),
        )

    _farm_log(f"Q matrix (FarmCPU) shape: {qmatrix.shape}")
    if q_ids is None:
        if sid_arr is None:
            raise ValueError("FarmCPU Q matrix IDs are unavailable.")
        q_ids = np.asarray(sid_arr, dtype=str)
    if bool(return_ids):
        return qmatrix, np.asarray(q_ids, dtype=str)
    return qmatrix


def run_farmcpu_fullmem(
    args,
    gfile: str,
    prefix: str,
    logger: logging.Logger,
    pheno_preloaded: Union[pd.DataFrame , None] = None,
    ids_preloaded: Union[np.ndarray, None] = None,
    n_snps_preloaded: Union[int, None] = None,
    qmatrix_preloaded: Union[np.ndarray, None] = None,
    cov_preloaded: Union[np.ndarray, None] = None,
    use_spinner: bool = False,
    context_prepared: bool = False,
    summary_rows: Union[list[dict[str, object]], None] = None,
    saved_paths: Union[list[str], None] = None,
    trait_names: Union[list[str], None] = None,
    farmcpu_cache: Union[dict[str, object], None] = None,
    prepare_only: bool = False,
    emit_trait_header: bool = True,
    preloaded_packed: Union[dict[str, object], None] = None,
    qtn_preloaded_packed: Union[dict[str, object], None] = None,
    emit_file_dense_warning: bool = True,
    trait_prepared_meta: Union[dict[str, object], None] = None,
) -> dict[str, object]:
    """
    Run FarmCPU via the Rust packed-BED controller (packed genotype + QK + PCA).

    If pheno_preloaded is provided from a non-streaming path, it may be reused.
    When called after streaming GWAS context preparation, FarmCPU must use full
    genotype IDs and therefore reload phenotype/Q/cov on the full ID space.
    """
    phenofile = args.pheno
    outfolder = args.out
    qdim = args.qcov
    cov = args.cov
    snps_only = bool(getattr(args, "snps_only", False))
    effective_qtn_preloaded_packed = (
        qtn_preloaded_packed if isinstance(qtn_preloaded_packed, dict) else None
    )

    if (
        farmcpu_cache is None
        and isinstance(qtn_preloaded_packed, dict)
        and pheno_preloaded is not None
        and ids_preloaded is not None
        and qmatrix_preloaded is not None
    ):
        qtn_ctx_obj = qtn_preloaded_packed.get("packed_ctx")
        if not isinstance(qtn_ctx_obj, dict):
            raise ValueError("Invalid QTN packed payload for FarmCPU: missing packed_ctx.")
        qtn_ctx = _normalize_packed_ctx_for_farmcpu_cache(qtn_ctx_obj)
        qtn_row_idx = np.asarray(qtn_ctx.get("row_indices", []), dtype=np.int64).reshape(-1)
        qtn_ids = np.asarray(qtn_preloaded_packed.get("full_ids", []), dtype=str)
        id_to_qtn = {str(sid): i for i, sid in enumerate(qtn_ids)}
        try:
            qtn_sample_idx = np.asarray([id_to_qtn[str(sid)] for sid in np.asarray(ids_preloaded, dtype=str)], dtype=np.int64)
        except KeyError as e:
            raise ValueError("Some GWAS sample IDs are not present in QTN genotype sample order.") from e
        farmcpu_cache = {
            "pheno": pheno_preloaded,
            "famid": np.asarray(ids_preloaded, dtype=str),
            "geno": None,
            "packed_ctx": qtn_ctx,
            "ref_alt": _make_deferred_bim_metadata(
                str(qtn_preloaded_packed.get("prefix", "")),
                qtn_row_idx,
                n_markers=int(qtn_row_idx.shape[0]),
            ),
            "qmatrix": np.asarray(qmatrix_preloaded, dtype="float32"),
            "packed_sample_idx": qtn_sample_idx,
            "_qtn_preloaded_packed": qtn_preloaded_packed,
            "_qtn_stage_mode": True,
        }

    def _load_bim_ref_alt_filtered(
        prefix: str,
        keep_mask: np.ndarray,
        *,
        snps_only_mode: bool,
        return_metadata: bool = True,
        emit_status: bool = True,
    ) -> tuple[Union[pd.DataFrame, None], np.ndarray]:
        bim_path = f"{prefix}.bim"
        src = _basename_only(bim_path)
        keep = np.ascontiguousarray(np.asarray(keep_mask, dtype=np.bool_).reshape(-1))
        n_total = int(keep.shape[0])
        if n_total == 0:
            return (
                (
                    pd.DataFrame(columns=["chrom", "pos", "snp", "allele0", "allele1"])
                    if bool(return_metadata)
                    else None
                ),
                keep,
            )

        n_pre_keep = int(np.sum(keep))
        if n_pre_keep == 0:
            return (
                (
                    pd.DataFrame(columns=["chrom", "pos", "snp", "allele0", "allele1"])
                    if bool(return_metadata)
                    else None
                ),
                keep,
            )

        # Stream BIM and keep only SNPs that survive numeric filters, optionally
        # applying SNP-only filtering in the same pass.
        if bool(return_metadata):
            chrom = np.empty(n_pre_keep, dtype=object)
            pos = np.empty(n_pre_keep, dtype=np.int64)
            snp = np.empty(n_pre_keep, dtype=object)
            allele0 = np.empty(n_pre_keep, dtype=object)
            allele1 = np.empty(n_pre_keep, dtype=object)
        else:
            chrom = pos = snp = allele0 = allele1 = None

        pbar = (
            _ProgressAdapter(
                total=n_total,
                desc=f"Loading site metadata ({src})",
                logger=logger,
                log_unit="row",
            )
            if bool(emit_status)
            else None
        )
        lines = 0
        done = 0
        out = 0
        try:
            with open(bim_path, "r", encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    if lines >= n_total:
                        raise ValueError(
                            f"BIM has more rows than genotype matrix: bim>{n_total} ({src})"
                        )
                    idx = lines
                    lines += 1

                    if keep[idx]:
                        parts = raw.strip().split()
                        if len(parts) < 6:
                            raise ValueError(
                                f"Invalid BIM row at line {idx + 1}: expected >=6 columns."
                            )
                        a0 = str(parts[4])
                        a1 = str(parts[5])
                        if snps_only_mode and (len(a0) != 1 or len(a1) != 1):
                            keep[idx] = False
                        elif bool(return_metadata):
                            chrom[out] = str(parts[0])
                            raw_snp = str(parts[1]) if len(parts) > 1 else "."
                            try:
                                pval = int(parts[3])
                            except Exception:
                                try:
                                    pval = int(float(parts[3]))
                                except Exception:
                                    pval = 0
                            pos[out] = int(pval)
                            snp[out] = raw_snp if raw_snp not in {"", ".", "nan", "NaN"} else f"{chrom[out]}_{int(pval)}"
                            allele0[out] = a0
                            allele1[out] = a1
                            out += 1

                    if (pbar is not None) and (lines - done >= 200_000):
                        step = lines - done
                        pbar.update(step)
                        done = lines

            if lines != n_total:
                raise ValueError(
                    f"BIM row count mismatch: bim={lines}, genotype={n_total} ({src})"
                )
            if (pbar is not None) and done < n_total:
                pbar.update(n_total - done)
            if pbar is not None:
                pbar.finish()
        finally:
            if pbar is not None:
                pbar.close()

        if not bool(return_metadata):
            return None, keep

        ref_alt_df = pd.DataFrame(
            {
                "chrom": chrom[:out],
                "pos": pos[:out].astype(int, copy=False),
                "snp": snp[:out],
                "allele0": allele0[:out],
                "allele1": allele1[:out],
            }
        )
        return ref_alt_df, keep

    def _load_farmcpu_packed_ctx(
        packed_prefix: str,
        expected_n_samples: int,
        *,
        status_label: str,
        trait_prepared_meta: Union[dict[str, object], None] = None,
        emit_status: bool = True,
    ) -> tuple[dict[str, object], object]:
        if isinstance(trait_prepared_meta, dict):
            from janusx.assoc.workflow_model_packed import _prepare_packed_bed_once_for_gwas

            _prefix_loaded, _full_ids_loaded, packed_ctx_prepared, _sites_meta = _prepare_packed_bed_once_for_gwas(
                genofile=str(packed_prefix),
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
                het_threshold=float(args.het),
                snps_only=bool(snps_only),
                use_spinner=bool(use_spinner),
                preloaded_packed=None,
                load_site_meta=False,
                status_label=str(status_label),
                trait_prepared_meta=trait_prepared_meta,
                emit_status=bool(emit_status),
            )
            row_idx_prepared = np.ascontiguousarray(
                np.asarray(
                    packed_ctx_prepared.get(
                        "row_indices",
                        packed_ctx_prepared.get("active_row_idx", []),
                    ),
                    dtype=np.int64,
                ).reshape(-1),
                dtype=np.int64,
            )
            if int(row_idx_prepared.shape[0]) == 0:
                raise ValueError("FarmCPU trait-prepared packed context produced zero active SNPs.")
            ref_alt_prepared = _make_deferred_bim_metadata(
                str(packed_prefix),
                row_idx_prepared,
                n_markers=int(row_idx_prepared.shape[0]),
            )
            return packed_ctx_prepared, ref_alt_prepared

        packed_load_t0 = time.monotonic()
        if bool(emit_status):
            with CliStatus(f"{status_label}...", enabled=bool(use_spinner)) as task:
                try:
                    _sample_ids_packed, packed_ctx_raw = prepare_packed_ctx_from_plink(
                        str(packed_prefix),
                        maf=float(args.maf),
                        missing_rate=float(args.geno),
                        snps_only=False,
                        expected_n_samples=int(expected_n_samples),
                        filter_mode="lazy_owned",
                    )
                except Exception:
                    task.fail(f"{status_label} ...Failed")
                    raise
                task.complete(f"{status_label} ...Finished")
        else:
            _sample_ids_packed, packed_ctx_raw = prepare_packed_ctx_from_plink(
                str(packed_prefix),
                maf=float(args.maf),
                missing_rate=float(args.geno),
                snps_only=False,
                expected_n_samples=int(expected_n_samples),
                filter_mode="lazy_owned",
            )

        _log_file_only(
            logger,
            logging.INFO,
            f"{status_label} ({int(expected_n_samples)} samples) "
            f"[{format_elapsed(time.monotonic() - packed_load_t0)}]",
        )

        keep_numeric = np.ascontiguousarray(
            np.asarray(packed_ctx_raw["site_keep"], dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        _ref_alt_df, keep_final = _load_bim_ref_alt_filtered(
            str(packed_prefix),
            keep_numeric,
            snps_only_mode=bool(snps_only),
            return_metadata=False,
            emit_status=False,
        )
        if not np.any(keep_final):
            raise ValueError("After filtering, number of SNPs is zero for FarmCPU.")

        packed_num = np.ascontiguousarray(np.asarray(packed_ctx_raw["packed"], dtype=np.uint8))
        miss_num = np.ascontiguousarray(
            np.asarray(packed_ctx_raw["missing_rate"], dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
        maf_num = np.ascontiguousarray(
            np.asarray(packed_ctx_raw.get("af", packed_ctx_raw["maf"]), dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
        active_row_idx = np.ascontiguousarray(
            np.flatnonzero(keep_final).astype(np.int64, copy=False),
            dtype=np.int64,
        )
        miss_arr = np.ascontiguousarray(miss_num[active_row_idx], dtype=np.float32)
        maf_arr = np.ascontiguousarray(maf_num[active_row_idx], dtype=np.float32)

        row_flip_raw = packed_ctx_raw.get("row_flip")
        if row_flip_raw is None:
            row_flip_raw = np.zeros(int(packed_num.shape[0]), dtype=np.bool_)
        row_flip_full = np.ascontiguousarray(
            np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        if int(row_flip_full.shape[0]) != int(packed_num.shape[0]):
            raise ValueError("Packed row_flip length mismatch for FarmCPU packed context.")
        row_flip_arr = np.ascontiguousarray(row_flip_full[active_row_idx], dtype=np.bool_)

        loaded_snps = int(active_row_idx.shape[0])
        ref_alt = _make_deferred_bim_metadata(
            str(packed_prefix),
            active_row_idx,
            n_markers=loaded_snps,
        )
        packed_ctx = {
            "packed": packed_num,
            "missing_rate": miss_arr,
            "af": maf_arr,
            "maf": maf_arr,
            "row_flip": row_flip_arr,
            "row_indices": active_row_idx,
            "site_keep": np.ascontiguousarray(keep_final, dtype=np.bool_),
            "packed_filter_mode": "lazy_full",
            "n_samples": int(packed_ctx_raw["n_samples"]),
            "source_prefix": str(packed_prefix),
        }
        return packed_ctx, ref_alt

    if farmcpu_cache is None:
        t_loading = time.time()
        reuse_shared_context = bool(
            context_prepared
            and pheno_preloaded is not None
            and ids_preloaded is not None
            and qmatrix_preloaded is not None
        )
        # If FarmCPU is invoked after streaming models, pheno_preloaded/ids_preloaded
        # are usually intersection-aligned. FarmCPU must operate on full genotype IDs.
        reuse_preloaded_pheno = bool(pheno_preloaded is not None) and (
            bool(reuse_shared_context) or (not bool(context_prepared))
        )
        pheno = pheno_preloaded if reuse_preloaded_pheno else None
        if pheno is None:
            pheno = _load_phenotype_with_status(
                phenofile,
                args.ncol,
                logger,
                id_col=0,
                use_spinner=use_spinner,
            )
        else:
            if not bool(context_prepared):
                with CliStatus("Loading phenotype from dataframe...", enabled=bool(use_spinner)) as task:
                    task.complete(
                        f"Loading phenotype from dataframe (n={pheno.shape[0]}, npheno={pheno.shape[1]})"
                    )

        preloaded_packed_ready = (
            preloaded_packed if packed_preload_is_ready(preloaded_packed) else None
        )
        packed_prefix = _as_plink_prefix(gfile)
        if packed_prefix is None and isinstance(preloaded_packed_ready, dict):
            packed_prefix = _as_plink_prefix(
                str(preloaded_packed_ready.get("prefix", ""))
            )
        if packed_prefix is None:
            cache_guess = _gwas_cache_prefix_with_params(
                gfile,
                maf=float(args.maf),
                geno=float(args.geno),
                snps_only=bool(snps_only),
                logger=logger,
            )
            packed_prefix = _as_plink_prefix(cache_guess)
        if packed_prefix is None:
            cache_guess_plain = genotype_cache_prefix(
                gfile,
                snps_only=bool(snps_only),
                logger=logger,
            )
            packed_prefix = _as_plink_prefix(cache_guess_plain)

        geno = None
        packed_ctx: Union[dict[str, object], None] = None
        ref_alt = None
        famid_full: Union[np.ndarray, None] = None
        n_snps_full: Union[int, None] = None
        if packed_prefix is not None:
            try:
                famid_full_raw, n_snps_full = inspect_genotype_file(str(packed_prefix))
                famid_full = np.asarray(famid_full_raw, dtype=str)
            except Exception:
                famid_full = None
                n_snps_full = None

        if bool(reuse_shared_context):
            famid = np.asarray(ids_preloaded, dtype=str)
            n_snps = int(n_snps_preloaded) if n_snps_preloaded is not None else int(n_snps_full or 0)
        else:
            # Always inspect full genotype metadata for FarmCPU to avoid reusing
            # streaming-intersection IDs (which can be smaller than packed BED n).
            famid_raw, n_snps = _inspect_genotype_with_status(
                gfile,
                logger,
                use_spinner=use_spinner,
                snps_only=bool(snps_only),
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
                het_threshold=float(args.het),
                force_kind=determine_genotype_source_force_kind_from_args(args),
            )
            famid = np.asarray(famid_raw, dtype=str)
            if famid_full is None:
                famid_full = np.asarray(famid, dtype=str)
                n_snps_full = int(n_snps)

        _matrix_prefix_cli, matrix_path_cli = _resolve_file_input_matrix(str(gfile))
        can_use_packed = (
            bool(getattr(args, "model", "add") == "add")
            and packed_prefix is not None
        )
        defer_packed_trait_load = bool(context_prepared)
        if can_use_packed:
            if not defer_packed_trait_load:
                packed_ctx, ref_alt = _load_farmcpu_packed_ctx(
                    str(packed_prefix),
                    int(
                        famid_full.shape[0]
                        if famid_full is not None
                        else famid.shape[0]
                    ),
                    status_label="Loading genotype (Full)",
                )
            else:
                packed_ctx = None
                ref_alt = None
            geno = None
        else:
            raise ValueError(_FARMCPU_PACKED_ONLY_MSG)

        t_loaded = time.time() - t_loading
        if (not bool(context_prepared)) and (not bool(prepare_only)):
            ns_loaded = _bim_metadata_len(ref_alt)
            _rich_success(
                logger,
                f"FarmCPU input ready (n={len(famid)}, nSNP={ns_loaded}) [{format_elapsed(t_loaded)}]",
                use_spinner=use_spinner,
                log_message=f"Genotype and phenotype loaded in {t_loaded:.2f} seconds",
            )
        else:
            _log_file_only(
                logger,
                logging.INFO,
                f"Genotype and phenotype loaded in {t_loaded:.2f} seconds",
            )
        if (ref_alt is not None) and int(_bim_metadata_len(ref_alt)) == 0:
            msg = "After filtering, number of SNPs is zero for FarmCPU."
            logger.error(msg)
            raise ValueError(msg)

        # Build FarmCPU Q/cov on full genotype IDs; do not reuse streaming-aligned
        # preloaded Q/cov because they may be trimmed by GRM/Q/cov intersections.
        gfile_prefix = _gwas_cache_prefix_with_params(
            gfile,
            maf=float(args.maf),
            geno=float(args.geno),
            snps_only=bool(snps_only),
            logger=logger,
        )
        if famid_full is None:
            famid_full = np.asarray(famid, dtype=str)
        if bool(reuse_shared_context):
            q_base = np.ascontiguousarray(
                np.asarray(qmatrix_preloaded, dtype="float32"),
                dtype="float32",
            )
            if q_base.ndim != 2:
                raise ValueError("FarmCPU shared-context Q matrix must be 2D.")
            cov_base = None
            if cov_preloaded is not None:
                cov_base = np.ascontiguousarray(
                    np.asarray(cov_preloaded, dtype="float32"),
                    dtype="float32",
                )
                if cov_base.ndim != 2:
                    raise ValueError("FarmCPU shared-context covariate matrix must be 2D.")
                if int(cov_base.shape[0]) != int(famid.shape[0]):
                    raise ValueError(
                        "FarmCPU shared-context covariates row count mismatch: "
                        f"cov={int(cov_base.shape[0])}, ids={int(famid.shape[0])}."
                    )
            qmatrix = q_base
            if cov_base is not None and int(cov_base.shape[1]) > 0:
                qmatrix = np.ascontiguousarray(
                    np.concatenate([q_base, cov_base], axis=1),
                    dtype="float32",
                )
            q_ids = np.asarray(famid, dtype=str)
            common_ids = q_ids.tolist()
            full_index = {sid: i for i, sid in enumerate(famid_full.tolist())}
            packed_sample_idx = np.asarray(
                [full_index[sid] for sid in common_ids],
                dtype=np.int64,
            )
            cov_n = (
                int(famid.shape[0])
                if (cov_preloaded is not None and int(np.asarray(cov_preloaded).shape[1]) > 0)
                else "NA"
            )
            geno_n = int(famid_full.shape[0])
            common_n = int(len(common_ids))
            q_n = (
                int(q_ids.shape[0])
                if int(q_base.shape[1]) > 0
                else "NA"
            )
        else:
            famid_full = np.asarray(famid, dtype=str)
            qmatrix, q_ids = build_qmatrix_farmcpu(
                genofile=gfile,
                gfile_prefix=gfile_prefix,
                geno=geno,
                qdim=qdim,
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
                het_threshold=float(args.het),
                cov_inputs=cov,
                chunk_size=args.chunksize,
                logger=logger,
                sample_ids=famid.astype(str),
                use_spinner=use_spinner,
                quiet_terminal=bool(context_prepared or prepare_only),
                snps_only=bool(snps_only),
                n_snps_hint=int(_bim_metadata_len(ref_alt) if ref_alt is not None else int(n_snps)),
                threads=int(args.thread),
                mmap_limit=bool(args.mmap_limit),
                preloaded_packed=preloaded_packed,
                packed_ctx_preloaded=packed_ctx,
                return_ids=True,
            )
            q_ids = np.asarray(q_ids, dtype=str)
            q_id_set = set(q_ids.tolist())
            pheno_ids_set = set(np.asarray(pheno.index, dtype=str))
            common_ids = [
                sid for sid in famid_full.tolist()
                if sid in q_id_set and sid in pheno_ids_set
            ]
            if len(common_ids) == 0:
                raise ValueError("No overlapping samples across FarmCPU genotype/phenotype/Q/cov.")
            q_index = {sid: i for i, sid in enumerate(q_ids.tolist())}
            q_take = np.asarray([q_index[sid] for sid in common_ids], dtype=np.int64)
            qmatrix = np.ascontiguousarray(qmatrix[q_take], dtype="float32")
            full_index = {sid: i for i, sid in enumerate(famid_full.tolist())}
            packed_sample_idx = np.asarray(
                [full_index[sid] for sid in common_ids],
                dtype=np.int64,
            )
            famid = np.asarray(common_ids, dtype=str)
            cov_n = "NA"
            if len(_normalize_cov_inputs(cov)) > 0:
                cov_n = int(len(common_ids))
            geno_n = int(famid_full.shape[0]) if geno is None else int(geno.shape[1])
            common_n = int(len(common_ids))
            q_n = (
                int(q_ids.shape[0]) if (np.asarray(qmatrix).ndim == 2 and int(qmatrix.shape[1]) > 0) else "NA"
            )
        if not bool(context_prepared):
            _emit_plain_info_line(
                logger,
                (
                    f"Sample overlap: geno={geno_n}, pheno={pheno.shape[0]}, "
                    f"q={q_n}, cov={cov_n}, common={common_n}"
                ),
                use_spinner=bool(use_spinner),
            )
        farmcpu_cache = {
            "pheno": pheno,
            "famid": famid,
            "geno": geno,
            "packed_ctx": packed_ctx,
            "ref_alt": ref_alt,
            "qmatrix": qmatrix,
            "packed_sample_idx": packed_sample_idx,
            "packed_prefix": str(packed_prefix),
            "packed_full_ids": famid_full,
            "_defer_packed_trait_load": bool(defer_packed_trait_load),
        }
        if effective_qtn_preloaded_packed is not None:
            farmcpu_cache["_qtn_preloaded_packed"] = effective_qtn_preloaded_packed
            farmcpu_cache["_qtn_stage_mode"] = True
    else:
        pheno = farmcpu_cache["pheno"]  # type: ignore[assignment]
        famid = np.asarray(farmcpu_cache["famid"], dtype=str)
        geno_obj = farmcpu_cache.get("geno")
        geno = (
            None
            if geno_obj is None
            else np.ascontiguousarray(np.asarray(geno_obj, dtype="float32"))
        )
        packed_obj = farmcpu_cache.get("packed_ctx")
        packed_ctx = None
        if isinstance(packed_obj, dict):
            packed_ctx = _normalize_packed_ctx_for_farmcpu_cache(packed_obj)
        defer_packed_trait_load = bool(farmcpu_cache.get("_defer_packed_trait_load", False))
        stage2_qtn_payload_obj = farmcpu_cache.get("_qtn_preloaded_packed")
        if effective_qtn_preloaded_packed is None and isinstance(stage2_qtn_payload_obj, dict):
            effective_qtn_preloaded_packed = stage2_qtn_payload_obj
        if (
            effective_qtn_preloaded_packed is not None
            and not bool(farmcpu_cache.get("_qtn_stage_mode", False))
        ):
            farmcpu_cache["_qtn_preloaded_packed"] = effective_qtn_preloaded_packed
            farmcpu_cache["_qtn_stage_mode"] = True
        packed_sample_idx_obj = farmcpu_cache.get("packed_sample_idx")
        packed_sample_idx: Union[np.ndarray, None]
        if packed_sample_idx_obj is None:
            packed_sample_idx = None
        else:
            packed_sample_idx = np.ascontiguousarray(
                np.asarray(packed_sample_idx_obj, dtype=np.int64).reshape(-1),
                dtype=np.int64,
            )
            if int(packed_sample_idx.shape[0]) != int(famid.shape[0]):
                raise ValueError(
                    "FarmCPU cache packed_sample_idx length mismatch with famid."
                )
        ref_alt = farmcpu_cache.get("ref_alt")
        qmatrix = np.asarray(farmcpu_cache["qmatrix"], dtype="float32")
    if farmcpu_cache is None:
        packed_sample_idx = None
        defer_packed_trait_load = False

    if bool(prepare_only):
        return farmcpu_cache if farmcpu_cache is not None else {}

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pheno_aligned, famid = _align_pheno_to_sample_order(pheno, famid)
    trait_iter = _resolve_trait_iter(pheno_aligned, trait_names)
    multi_trait_mode = len(trait_iter) > 1
    for trait_idx, phename in enumerate(trait_iter):
        y_full, famidretain = _trait_values_and_mask(pheno_aligned, str(phename))
        keep_idx = np.flatnonzero(famidretain).astype(np.int64, copy=False)
        if keep_idx.size == 0:
            logger.warning(f"{phename}: no overlapping samples, skipped.")
            if multi_trait_mode:
                logger.info("")  # single blank line between traits
            continue

        p_sub = np.ascontiguousarray(y_full[keep_idx], dtype=np.float64).reshape(-1, 1)
        q_sub = qmatrix[keep_idx]
        trait_ids = np.asarray(famid[keep_idx], dtype=str)
        n_idv = int(keep_idx.shape[0])
        if packed_ctx is None and bool(defer_packed_trait_load):
            packed_prefix_cached = str(farmcpu_cache.get("packed_prefix", "")).strip()
            if packed_prefix_cached == "":
                packed_prefix_cached = str(_as_plink_prefix(gfile) or "")
            if packed_prefix_cached == "":
                raise ValueError("FarmCPU deferred packed load requires a PLINK BED prefix.")
            full_ids_cached = np.asarray(
                farmcpu_cache.get("packed_full_ids", []),
                dtype=str,
            ).reshape(-1)
            expected_n_samples = int(full_ids_cached.shape[0]) if int(full_ids_cached.shape[0]) > 0 else int(famid.shape[0])
            packed_ctx, ref_alt = _load_farmcpu_packed_ctx(
                packed_prefix_cached,
                expected_n_samples,
                status_label="Loading BED genotype of trait-subset",
                trait_prepared_meta=trait_prepared_meta,
                emit_status=False,
            )
            farmcpu_cache["packed_ctx"] = packed_ctx
            farmcpu_cache["ref_alt"] = ref_alt
            if packed_sample_idx is None and int(full_ids_cached.shape[0]) > 0:
                id_to_full = {str(sid): i for i, sid in enumerate(full_ids_cached.tolist())}
                packed_sample_idx = np.ascontiguousarray(
                    np.asarray([id_to_full[str(sid)] for sid in famid], dtype=np.int64),
                    dtype=np.int64,
                )
                farmcpu_cache["packed_sample_idx"] = packed_sample_idx
        if packed_ctx is None:
            raise ValueError(_FARMCPU_PACKED_ONLY_MSG)
        else:
            m_input = packed_ctx
            if packed_sample_idx is not None:
                sample_idx_arg = np.ascontiguousarray(
                    packed_sample_idx[keep_idx],
                    dtype=np.int64,
                )
            else:
                sample_idx_arg = keep_idx
            maf = np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1)
            tested_n = int(sample_idx_arg.shape[0]) if sample_idx_arg is not None else int(packed_ctx["n_samples"])
            miss_arr = np.ascontiguousarray(
                np.rint(
                    np.asarray(packed_ctx["missing_rate"], dtype=np.float64).reshape(-1)
                    * float(tested_n)
                ).astype(np.float32, copy=False),
                dtype=np.float32,
            )
            eff_snp = int(maf.shape[0])

        if bool(emit_trait_header):
            _emit_trait_header(
                logger,
                phename,
                n_idv,
                pve=None,
                use_spinner=bool(use_spinner),
                width=60,
            )

        cpu_t0 = process.cpu_times()
        t0 = time.time()
        gwas_t0 = time.time()
        rss0, _ = process_memory_info_bytes(process)
        peak_rss = int(rss0 or 0)
        farm_iter = max(1, int(getattr(args, "farmcpu_iter", 30)))
        farm_threshold_raw = getattr(args, "farmcpu_threshold", None)
        farm_threshold = (
            (1.0 / float(max(1, int(eff_snp))))
            if farm_threshold_raw is None
            else float(farm_threshold_raw)
        )
        farm_qtn_bound_raw = getattr(args, "farmcpu_qtn_bound", None)
        farm_qtn_bound = None if farm_qtn_bound_raw is None else int(farm_qtn_bound_raw)
        farm_nbin = max(1, int(getattr(args, "farmcpu_nbin", 5)))
        farm_szbin = [float(x) for x in getattr(args, "farmcpu_bin_size", [5e5, 5e6, 5e7])]
        farm_raw = bool(getattr(args, "farmcpu_raw", False))
        farm_stage1_pbar = _ProgressAdapter(
            total=farm_iter,
            desc="FarmCPU stage1",
            force_animate=bool(use_spinner),
            logger=logger,
            log_unit="iter",
        )
        farm_stage1_state = {"done": 0}
        farm_scan_pbar = None
        farm_scan_state = {"done": 0}
        farm_scan_total_hint = int(max(1, int(eff_snp)))

        def _close_farm_stage1(*, force_fill: bool = False) -> None:
            nonlocal farm_stage1_pbar
            if farm_stage1_pbar is None:
                return
            try:
                total_now = int(max(1, int(farm_stage1_pbar.total)))
                if force_fill and int(farm_stage1_state["done"]) < total_now:
                    farm_stage1_pbar.update(total_now - int(farm_stage1_state["done"]))
                    farm_stage1_state["done"] = total_now
                farm_stage1_pbar.finish()
            except Exception:
                pass
            try:
                farm_stage1_pbar.close(show_done=False)
            except Exception:
                pass
            farm_stage1_pbar = None

        def _close_farm_scan(*, force_fill: bool = False) -> None:
            nonlocal farm_scan_pbar
            if farm_scan_pbar is None:
                return
            try:
                total_now = int(max(1, int(farm_scan_pbar.total)))
                if force_fill and int(farm_scan_state["done"]) < total_now:
                    farm_scan_pbar.update(total_now - int(farm_scan_state["done"]))
                    farm_scan_state["done"] = total_now
                farm_scan_pbar.finish()
            except Exception:
                pass
            try:
                farm_scan_pbar.close(show_done=False)
            except Exception:
                pass
            farm_scan_pbar = None

        def _farmcpu_stage1_progress(done: int, total: int) -> None:
            nonlocal peak_rss, farm_stage1_pbar
            if farm_stage1_pbar is None:
                return
            total_use = int(max(1, int(total) if int(total) > 0 else int(farm_stage1_pbar.total)))
            if int(farm_stage1_pbar.total) != total_use:
                farm_stage1_pbar.set_total(total_use)
            target = int(max(0, min(total_use, int(done))))
            delta = target - int(farm_stage1_state["done"])
            if delta > 0:
                farm_stage1_pbar.update(delta)
                farm_stage1_state["done"] = target
            if int(total) > 0 and int(done) >= int(total):
                _close_farm_stage1(force_fill=False)
            try:
                rss_now, _ = process_memory_info_bytes(process)
                peak_rss = max(peak_rss, int(rss_now or 0))
            except Exception:
                pass

        def _farmcpu_scan_progress(done: int, total: int) -> None:
            nonlocal peak_rss, farm_scan_pbar
            if farm_stage1_pbar is not None:
                _close_farm_stage1(force_fill=True)
            total_use = int(max(1, int(total) if int(total) > 0 else farm_scan_total_hint))
            if farm_scan_pbar is None:
                farm_scan_pbar = _ProgressAdapter(
                    total=total_use,
                    desc="FarmCPU stage2",
                    force_animate=bool(use_spinner),
                    logger=logger,
                )
            elif int(farm_scan_pbar.total) != total_use:
                farm_scan_pbar.set_total(total_use)
            target = int(max(0, min(int(farm_scan_pbar.total), int(done))))
            delta = target - int(farm_scan_state["done"])
            if delta > 0:
                farm_scan_pbar.update(delta)
                farm_scan_state["done"] = target
            if int(total) > 0 and int(done) >= int(total):
                _close_farm_scan(force_fill=False)
            try:
                rss_now, _ = process_memory_info_bytes(process)
                peak_rss = max(peak_rss, int(rss_now or 0))
            except Exception:
                pass

        phename_tag = _safe_trait_file_label(phename)
        out_tsv = os.path.join(outfolder, f"{prefix}.{phename_tag}.farmcpu.tsv")
        tmp_tsv = _gwas_result_tmp_path(out_tsv)
        pseudo_tsv_hint = os.path.join(outfolder, f"{prefix}.{phename_tag}.farmcpu.qtn")
        n_pseudo_qtn = 0
        pseudo_tsv: Union[str, None] = None
        stage1_secs = 0.0
        stage2_secs = 0.0

        use_rust_controller = (
            packed_ctx is not None
            and bool(hasattr(jxrs, "farmcpu_packed_to_tsv"))
        )
        bed_prefix_meta: Union[str, None] = None
        if use_rust_controller:
            packed_payload = m_input
            if not isinstance(packed_payload, dict):
                raise ValueError("Internal error: expected packed payload for Rust FarmCPU controller.")
            rust_qtn_written = 0
            skip_stage1_main_scan = bool(farmcpu_cache.get("_qtn_stage_mode", False))
            stage1_wall_t0 = time.monotonic()
            rust_stage1_secs: Union[float, None] = None
            rust_stage2_secs: Union[float, None] = None
            try:
                packed_arr = packed_payload["packed"]
                row_flip_arr = packed_payload["row_flip"]
                maf_arr = packed_payload["maf"]
                sample_idx_use = (
                    None
                    if sample_idx_arg is None
                    else np.ascontiguousarray(np.asarray(sample_idx_arg, dtype=np.int64).reshape(-1))
                )
                tested_n = int(sample_idx_use.shape[0]) if sample_idx_use is not None else int(packed_payload["n_samples"])
                miss_arr = np.ascontiguousarray(
                    np.rint(
                        np.asarray(packed_payload["missing_rate"], dtype=np.float64).reshape(-1)
                        * float(tested_n)
                    ).astype(np.float32, copy=False),
                    dtype=np.float32,
                )
                row_idx_arr = np.ascontiguousarray(
                    np.asarray(
                        packed_payload.get(
                            "row_indices",
                            np.arange(int(np.asarray(maf_arr).reshape(-1).shape[0]), dtype=np.int64),
                        ),
                        dtype=np.int64,
                    ).reshape(-1),
                    dtype=np.int64,
                )
                packed_rows = int(np.asarray(packed_arr, dtype=np.uint8).shape[0])
                meta_rows = int(row_idx_arr.shape[0])
                if isinstance(ref_alt, dict) and str(ref_alt.get("_kind", "")) == "deferred_bim":
                    bed_prefix_raw = str(ref_alt.get("bed_prefix", "") or "").strip()
                    if bed_prefix_raw != "":
                        bed_prefix_meta = bed_prefix_raw
                if not bed_prefix_meta:
                    bed_prefix_raw = str(packed_payload.get("source_prefix", "") or "").strip()
                    if bed_prefix_raw != "":
                        bed_prefix_meta = bed_prefix_raw
                if not bed_prefix_meta:
                    bed_prefix_detected = _as_plink_prefix(gfile)
                    if bed_prefix_detected is not None:
                        bed_prefix_meta = str(bed_prefix_detected)
                if not bed_prefix_meta:
                    raise ValueError("FarmCPU packed Rust path requires a non-empty PLINK bed_prefix for BIM metadata.")

                fallback_meta_cols: Union[tuple[list[str], list[int], list[str], list[str], list[str]], None] = None

                def _resolve_meta_columns_for_fallback() -> tuple[list[str], list[int], list[str], list[str], list[str]]:
                    nonlocal fallback_meta_cols
                    if fallback_meta_cols is None:
                        fallback_meta_cols = _resolve_ref_alt_columns(
                            ref_alt,
                            expected_rows=meta_rows,
                            prefix=str(bed_prefix_meta),
                            row_indices=row_idx_arr,
                        )
                    return fallback_meta_cols

                def _compact_packed_for_metadata_alignment() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                    row_idx_local = np.ascontiguousarray(
                        np.asarray(row_idx_arr, dtype=np.int64).reshape(-1),
                        dtype=np.int64,
                    )
                    if int(row_idx_local.shape[0]) != int(meta_rows):
                        raise ValueError(
                            "FarmCPU packed compact fallback cannot align metadata: "
                            f"row_indices={int(row_idx_local.shape[0])}, metadata={int(meta_rows)}"
                        )
                    packed_compact = np.ascontiguousarray(
                        np.asarray(packed_arr, dtype=np.uint8)[row_idx_local],
                        dtype=np.uint8,
                    )
                    row_flip_compact = np.ascontiguousarray(
                        np.asarray(row_flip_arr, dtype=np.bool_).reshape(-1),
                        dtype=np.bool_,
                    )
                    maf_compact = np.ascontiguousarray(
                        np.asarray(maf_arr, dtype=np.float32).reshape(-1),
                        dtype=np.float32,
                    )
                    if int(row_flip_compact.shape[0]) != int(meta_rows) or int(maf_compact.shape[0]) != int(meta_rows):
                        raise ValueError(
                            "FarmCPU packed compact fallback metadata mismatch: "
                            f"row_flip={int(row_flip_compact.shape[0])}, maf={int(maf_compact.shape[0])}, metadata={int(meta_rows)}"
                        )
                    return packed_compact, row_flip_compact, maf_compact

                if hasattr(jxrs, "farmcpu_packed_to_tsv"):
                    try:
                        rust_result = jxrs.farmcpu_packed_to_tsv(
                            np.ascontiguousarray(p_sub, dtype=np.float64).reshape(-1),
                            np.ascontiguousarray(q_sub, dtype=np.float64),
                            packed_arr,
                            int(packed_payload["n_samples"]),
                            row_flip_arr,
                            maf_arr,
                            miss_arr,
                            tmp_tsv,
                            sample_idx_use,
                            row_indices=row_idx_arr,
                            source_row_indices=row_idx_arr,
                            threshold=float(farm_threshold),
                            max_iter=int(farm_iter),
                            qtn_bound=farm_qtn_bound,
                            nbin=int(farm_nbin),
                            szbin=[float(x) for x in farm_szbin],
                            threads=int(args.thread),
                            stage1_progress_callback=_farmcpu_stage1_progress,
                            progress_callback=(None if skip_stage1_main_scan else _farmcpu_scan_progress),
                            pseudo_tsv=pseudo_tsv_hint,
                            bed_prefix=bed_prefix_meta,
                            raw=bool(farm_raw),
                            skip_main_scan=bool(skip_stage1_main_scan),
                        )
                        (
                            _m_written,
                            n_pseudo_qtn,
                            rust_qtn_written,
                            rust_stage1_secs,
                            rust_stage2_secs,
                        ) = _parse_farmcpu_rust_tsv_result(rust_result)
                    except Exception as ex:
                        msg = str(ex)
                        can_retry_compact = (
                            ("metadata length mismatch" in msg or "row metadata length mismatch" in msg or "row_indices[" in msg)
                            and int(packed_rows) != int(meta_rows)
                            and int(row_idx_arr.shape[0]) == int(meta_rows)
                        )
                        if not can_retry_compact:
                            raise
                        packed_compact, row_flip_compact, maf_compact = _compact_packed_for_metadata_alignment()
                        rust_result = jxrs.farmcpu_packed_to_tsv(
                            np.ascontiguousarray(p_sub, dtype=np.float64).reshape(-1),
                            np.ascontiguousarray(q_sub, dtype=np.float64),
                            packed_compact,
                            int(packed_payload["n_samples"]),
                            row_flip_compact,
                            maf_compact,
                            miss_arr,
                            tmp_tsv,
                            sample_idx_use,
                            row_indices=None,
                            source_row_indices=row_idx_arr,
                            threshold=float(farm_threshold),
                            max_iter=int(farm_iter),
                            qtn_bound=farm_qtn_bound,
                            nbin=int(farm_nbin),
                            szbin=[float(x) for x in farm_szbin],
                            threads=int(args.thread),
                            stage1_progress_callback=_farmcpu_stage1_progress,
                            progress_callback=(None if skip_stage1_main_scan else _farmcpu_scan_progress),
                            pseudo_tsv=pseudo_tsv_hint,
                            bed_prefix=bed_prefix_meta,
                            raw=bool(farm_raw),
                            skip_main_scan=bool(skip_stage1_main_scan),
                        )
                        (
                            _m_written,
                            n_pseudo_qtn,
                            rust_qtn_written,
                            rust_stage1_secs,
                            rust_stage2_secs,
                        ) = _parse_farmcpu_rust_tsv_result(rust_result)
                elif hasattr(jxrs, "gwas_packed_unified_to_tsv"):
                    jobs = [
                        {
                            "model": "farmcpu",
                            "trait": str(phename),
                            "out_tsv": str(tmp_tsv),
                            "y": np.ascontiguousarray(p_sub, dtype=np.float64).reshape(-1),
                            "x_cov": np.ascontiguousarray(q_sub, dtype=np.float64),
                            "sample_indices": sample_idx_use,
                            "threshold": float(farm_threshold),
                            "max_iter": int(farm_iter),
                            "qtn_bound": farm_qtn_bound,
                            "nbin": int(farm_nbin),
                            "szbin": [float(x) for x in farm_szbin],
                            "raw": bool(farm_raw),
                            "skip_main_scan": bool(skip_stage1_main_scan),
                            "pseudo_tsv": str(pseudo_tsv_hint),
                            "stage1_progress_callback": _farmcpu_stage1_progress,
                            "scan_progress_callback": (None if skip_stage1_main_scan else _farmcpu_scan_progress),
                        }
                    ]
                    chrom_col, pos_col, snp_col, allele0_col, allele1_col = _resolve_meta_columns_for_fallback()
                    try:
                        _res = jxrs.gwas_packed_unified_to_tsv(
                            jobs,
                            packed_arr,
                            int(packed_payload["n_samples"]),
                            row_flip_arr,
                            maf_arr,
                            miss_arr,
                            chrom_col,
                            pos_col,
                            snp_col,
                            allele0_col,
                            allele1_col,
                            int(max(1, int(getattr(args, "chunksize", 10000)))),
                            int(args.thread),
                            None,
                            int(max(1, int(getattr(args, "chunksize", 10000)))),
                            row_indices=row_idx_arr,
                            bed_prefix=bed_prefix_meta,
                        )
                    except Exception as ex:
                        msg = str(ex)
                        can_retry_compact = (
                            ("metadata length mismatch" in msg or "row metadata length mismatch" in msg or "row_indices[" in msg)
                            and int(packed_rows) != int(meta_rows)
                            and int(row_idx_arr.shape[0]) == int(meta_rows)
                        )
                        if not can_retry_compact:
                            raise
                        packed_compact, row_flip_compact, maf_compact = _compact_packed_for_metadata_alignment()
                        _res = jxrs.gwas_packed_unified_to_tsv(
                            jobs,
                            packed_compact,
                            int(packed_payload["n_samples"]),
                            row_flip_compact,
                            maf_compact,
                            miss_arr,
                            chrom_col,
                            pos_col,
                            snp_col,
                            allele0_col,
                            allele1_col,
                            int(max(1, int(getattr(args, "chunksize", 10000)))),
                            int(args.thread),
                            None,
                            int(max(1, int(getattr(args, "chunksize", 10000)))),
                            row_indices=None,
                            bed_prefix=None,
                        )
                    r0 = _res[0]
                    n_pseudo_qtn = int(r0.get("qtn_count", 0))
                    rust_qtn_written = int(r0.get("pseudo_rows", 0))
                    _m_written = int(r0.get("written_rows", 0))
                    if "stage1_time_s" in r0:
                        rust_stage1_secs = float(r0.get("stage1_time_s", 0.0))
                    if "stage2_time_s" in r0:
                        rust_stage2_secs = float(r0.get("stage2_time_s", 0.0))
                else:
                    raise RuntimeError(
                        "Rust FarmCPU controller is unavailable: expected farmcpu_packed_to_tsv."
                    )
            finally:
                _close_farm_stage1(force_fill=True)
                _close_farm_scan(force_fill=True)
            stage1_total_secs = max(time.monotonic() - stage1_wall_t0, 0.0)
            stage1_secs = (
                max(float(rust_stage1_secs), 0.0)
                if rust_stage1_secs is not None
                else stage1_total_secs
            )
            if rust_stage2_secs is not None:
                stage2_secs = max(float(rust_stage2_secs), 0.0)
            n_pseudo_qtn = int(max(0, n_pseudo_qtn))
            if int(rust_qtn_written) > 0 and os.path.exists(pseudo_tsv_hint):
                pseudo_tsv = pseudo_tsv_hint
            if bool(skip_stage1_main_scan) and int(n_pseudo_qtn) > 0 and pseudo_tsv is None:
                raise ValueError(
                    "FarmCPU stage1 selected pseudo-QTNs but did not produce a readable pseudo-QTN TSV for stage2."
                )
            if bool(farmcpu_cache.get("_qtn_stage_mode", False)) and int(n_pseudo_qtn) > 0:
                if not isinstance(effective_qtn_preloaded_packed, dict):
                    raise ValueError("FarmCPU QTN stage2 payload is missing.")
                main_prefix = _as_plink_prefix(gfile)
                if main_prefix is None:
                    raise ValueError("FarmCPU QTN stage2 requires main genotype PLINK BED prefix.")
                from janusx.assoc.workflow_model_packed import run_qtn_segmented_lm_stage2

                stage2_done = {"value": 0}
                stage2_total_hint = int(max(1, int(n_snps_preloaded or eff_snp or 1)))
                stage2_pbar = _ProgressAdapter(
                    total=stage2_total_hint,
                    desc="FarmCPU stage2",
                    force_animate=bool(use_spinner),
                    logger=logger,
                )

                def _farmcpu_stage2_progress(done: int, total: int) -> None:
                    nonlocal peak_rss, stage2_pbar
                    try:
                        d = int(done)
                        t = int(total)
                    except Exception:
                        return
                    if t > 0 and int(stage2_pbar.total) != t and int(stage2_done["value"]) == 0:
                        stage2_pbar.set_total(int(max(1, t)))
                    target = int(max(0, min(int(stage2_pbar.total), d)))
                    delta = target - int(stage2_done["value"])
                    if delta > 0:
                        stage2_pbar.update(delta)
                        stage2_done["value"] = target
                    try:
                        rss_now, _ = process_memory_info_bytes(process)
                        peak_rss = max(peak_rss, int(rss_now or 0))
                    except Exception:
                        pass

                try:
                    stage2_t0 = time.monotonic()
                    kept_rows2, scan_rows2, qtn_used = run_qtn_segmented_lm_stage2(
                        main_prefix=str(main_prefix),
                        y_vec=np.ascontiguousarray(p_sub, dtype=np.float64).reshape(-1),
                        trait_ids=trait_ids,
                        base_cov=np.ascontiguousarray(q_sub, dtype=np.float64),
                        qtn_preloaded_packed=effective_qtn_preloaded_packed,
                        qtn_tsv=str(pseudo_tsv_hint),
                        tmp_tsv=str(tmp_tsv),
                        maf_threshold=float(args.maf),
                        max_missing_rate=float(args.geno),
                        het_threshold=float(args.het),
                        snps_only=bool(snps_only),
                        chunk_size=int(max(1, int(getattr(args, "chunksize", 10000)))),
                        threads=int(args.thread),
                        progress_callback=_farmcpu_stage2_progress,
                        args_obj=args,
                        logger=logger,
                        use_spinner=False,
                        status_prefix="FarmCPU stage2",
                    )
                    stage2_secs = max(time.monotonic() - stage2_t0, 0.0)
                    eff_snp = int(kept_rows2)
                    n_pseudo_qtn = int(max(n_pseudo_qtn, qtn_used))
                finally:
                    try:
                        if int(stage2_done["value"]) < int(stage2_pbar.total):
                            stage2_pbar.update(int(stage2_pbar.total) - int(stage2_done["value"]))
                        stage2_pbar.finish()
                    except Exception:
                        pass
                    stage2_pbar.close(show_done=False)
        else:
            _close_farm_stage1(force_fill=False)
            _close_farm_scan(force_fill=False)
            raise RuntimeError(
                "FarmCPU requires the Rust packed controller (`farmcpu_packed_to_tsv` "
                "or `gwas_packed_unified_to_tsv`). Dense/Python fallback has been removed."
            )

        name_source = str(gfile)
        if isinstance(packed_ctx, dict):
            name_source = str(packed_ctx.get("source_prefix", name_source) or name_source)
        elif matrix_path_cli is not None:
            name_source = str(matrix_path_cli)

        _finalize_gwas_result_tsv(tmp_tsv, out_tsv, name_source, logger=logger)

        gwas_secs = max(stage1_secs + stage2_secs, 0.0)
        if gwas_secs <= 0.0:
            gwas_secs = max(time.time() - gwas_t0, 0.0)
        mem_metrics = finalize_peak_memory_metrics(
            process,
            sampled_peak_rss_bytes=peak_rss,
        )
        cpu_t1 = process.cpu_times()
        wall = max(time.time() - t0, 1e-9)
        cpu_used = (cpu_t1.user + cpu_t1.system) - (cpu_t0.user + cpu_t0.system)
        avg_cpu = 100.0 * cpu_used / (wall * max(1, n_cores))
        peak_rss_gb = float(bytes_to_gib(mem_metrics.get("peak_rss_bytes")) or 0.0)
        peak_footprint_gb = bytes_to_gib(mem_metrics.get("peak_footprint_bytes"))

        viz_secs = 0.0
        if args.plot:
            viz_secs = _run_fastplot_from_tsv_with_status(
                out_tsv,
                p_sub,
                xlabel=phename,
                outpdf=os.path.join(outfolder, f"{prefix}.{phename_tag}.farmcpu.svg"),
                use_spinner=bool(use_spinner),
                emit_done_line=False,
            )
        summary_rows.append(
            {
                "phenotype": str(phename),
                "model": "Farm",
                "nidv": int(n_idv),
                "eff_snp": int(eff_snp),
                "pve": None,
                "avg_cpu": float(avg_cpu),
                "peak_rss_gb": float(peak_rss_gb),
                "peak_footprint_gb": (
                    float(peak_footprint_gb) if peak_footprint_gb is not None else None
                ),
                "stage1_time_s": float(stage1_secs),
                "stage2_time_s": float(stage2_secs),
                "gwas_time_s": float(gwas_secs),
                "viz_time_s": float(viz_secs),
                "result_file": str(out_tsv),
            }
        )
        saved_paths.append(str(out_tsv))
        if pseudo_tsv is not None and os.path.exists(pseudo_tsv):
            saved_paths.append(str(pseudo_tsv))

        farm_times = [
            _format_farmcpu_done_elapsed(stage1_secs),
            _format_farmcpu_done_elapsed(stage2_secs),
        ]
        if args.plot:
            farm_times.append(_format_farmcpu_done_elapsed(viz_secs))
        farm_done_msg = f"FarmCPU ...Found {n_pseudo_qtn} QTNs [{'/'.join(farm_times)}]"
        if (not bool(emit_trait_header)) and multi_trait_mode:
            farm_done_msg = f"FarmCPU({phename}) ...Found {n_pseudo_qtn} QTNs [{'/'.join(farm_times)}]"
        _rich_success(logger, farm_done_msg, use_spinner=use_spinner)
        if multi_trait_mode:
            logger.info("")
        if bool(defer_packed_trait_load):
            packed_ctx = None
            ref_alt = None
            farmcpu_cache["packed_ctx"] = None
            farmcpu_cache["ref_alt"] = None
    return farmcpu_cache


# ======================================================================
# CLI
# ======================================================================
