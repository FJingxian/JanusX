# -*- coding: utf-8 -*-
"""Streaming GWAS model runners (extracted from workflow.py)."""

from __future__ import annotations

import concurrent.futures as cf
import io
import logging
import os
import time
import uuid
from typing import Optional, Union

import numpy as np
import pandas as pd
import psutil
from scipy.stats import chi2

from .workflow import (
    CliStatus,
    FastLMM,
    LM,
    LMM,
    _GWAS_PROGRESS_BAR_WIDTH,
    _ProgressAdapter,
    _align_pheno_to_sample_order,
    _as_plink_prefix,
    _display_path,
    _emit_trait_header,
    _format_progress_metric,
    _gwas_evd_stage_ctx,
    _gwas_scan_stage_ctx,
    _log_file_only,
    _log_model_line,
    _resolve_stream_scan_chunk_size,
    _rich_success,
    _run_fastplot_from_tsv_with_status,
    _run_result_write_with_status,
    _trait_values_and_mask,
    auto_mmap_window_mb,
    build_rich_progress,
    detect_effective_threads,
    format_elapsed,
    jxrs,
    rich_progress_available,
    run_lm_packed_fullrank,
    run_lm_stream_bed_single_entry,
    run_lmm_packed_fullrank,
    should_animate_status,
)

_WARNED_BED_MMAP_LIMIT_LEGACY = False
_WARNED_BED_PREPARED_SNPS_ONLY_LEGACY = False


def _chisq_from_gwas_results(results: np.ndarray) -> np.ndarray:
    """Return the chi-square statistic aligned with GWAS TSV output semantics."""
    res = np.asarray(results, dtype=np.float64)
    chisq = np.full((int(res.shape[0]),), np.nan, dtype=np.float64)
    if res.ndim != 2 or res.shape[1] < 2:
        return chisq

    beta = np.asarray(res[:, 0], dtype=np.float64)
    se = np.asarray(res[:, 1], dtype=np.float64)
    valid_wald = np.isfinite(beta) & np.isfinite(se) & (se > 0.0)
    if np.any(valid_wald):
        z = beta[valid_wald] / se[valid_wald]
        chisq[valid_wald] = np.square(z)

    if res.shape[1] <= 3:
        return chisq

    plrt = np.asarray(res[:, 3], dtype=np.float64)
    valid_plrt = np.isfinite(plrt) & (plrt >= 0.0) & (plrt <= 1.0)
    if not np.any(valid_plrt):
        return chisq

    one_mask = valid_plrt & (plrt >= 1.0)
    if np.any(one_mask):
        chisq[one_mask] = 0.0

    zero_mask = valid_plrt & (plrt <= 0.0)
    if np.any(zero_mask):
        chisq[zero_mask] = np.inf

    mid_mask = valid_plrt & (plrt > 0.0) & (plrt < 1.0)
    if np.any(mid_mask):
        chisq[mid_mask] = chi2.isf(plrt[mid_mask], 1)

    return chisq


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


def _new_bed_chunk_reader_for_stream(
    *,
    bed_prefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    sample_ids: list[str],
    model: str,
    het_threshold: float,
    mmap_window_mb: Union[int, None],
    logger: logging.Logger,
) -> object:
    """
    Construct Rust BedChunkReader for GWAS streaming with optional mmap window.

    Rust-only strict mode:
    - `mmap_window_mb` support must exist when requested.
    - No Python-side fallback is allowed.
    """
    global _WARNED_BED_MMAP_LIMIT_LEGACY

    base_kwargs = dict(
        prefix=str(bed_prefix),
        maf_threshold=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
        fill_missing=True,
        sample_ids=list(sample_ids),
        model=str(model),
        het_threshold=float(het_threshold),
    )
    if mmap_window_mb is not None:
        base_kwargs["mmap_window_mb"] = int(max(1, int(mmap_window_mb)))

    try:
        return jxrs.BedChunkReader(**base_kwargs)
    except TypeError as ex:
        mmap_req = ("mmap_window_mb" in base_kwargs)
        msg = str(ex).lower()
        kw_err = ("mmap_window_mb" in msg) or ("unexpected keyword argument" in msg)
        if mmap_req and kw_err:
            raise RuntimeError(
                "Rust-only GWAS mode requires BedChunkReader(..., mmap_window_mb=...) support. "
                "Rebuild/reinstall JanusX extension."
            ) from ex
        raise


def _bed_next_chunk_prepared_compat(
    *,
    bed_reader: object,
    chunk_size: int,
    coding: str,
    snps_only: bool,
    logger: logging.Logger,
) -> tuple[object, bool]:
    """
    Call BedChunkReader.next_chunk_prepared with snps_only (strict Rust-only).

    Returns (out, False). The second value is kept for API compatibility.
    """
    global _WARNED_BED_PREPARED_SNPS_ONLY_LEGACY
    try:
        out = bed_reader.next_chunk_prepared(
            int(chunk_size),
            coding=str(coding),
            snps_only=bool(snps_only),
        )
        return out, False
    except TypeError as ex:
        msg = str(ex).lower()
        kw_err = ("snps_only" in msg) and (
            "unexpected keyword argument" in msg
            or "got an unexpected keyword argument" in msg
        )
        if kw_err:
            raise RuntimeError(
                "Rust-only GWAS mode requires BedChunkReader.next_chunk_prepared(..., snps_only=...). "
                "Rebuild/reinstall JanusX extension."
            ) from ex
        raise


def _filter_snps_only_chunk_python_fallback(
    *,
    geno_center: np.ndarray,
    maf_chunk: np.ndarray,
    sites: list[object],
) -> tuple[np.ndarray, np.ndarray, list[object]]:
    raise RuntimeError(
        "Rust-only GWAS mode disables Python snps_only chunk fallback."
    )


def _fastlmm_switch_to_lm_decision(
    *,
    y_vec: np.ndarray,
    x_cov: np.ndarray,
    lmm_ml0: Optional[float],
    alpha: float = 0.05,
) -> tuple[bool, Optional[float], Optional[float]]:
    """
    Decide whether FastLMM should switch to LM for a trait.

    Rust null-LRT decision (H0: Va=0 boundary test), Rust-only mode.
    """
    ml0 = None
    try:
        if lmm_ml0 is not None:
            ml0_tmp = float(lmm_ml0)
            if np.isfinite(ml0_tmp):
                ml0 = ml0_tmp
    except Exception:
        ml0 = None

    if ml0 is None:
        raise RuntimeError(
            "FastLMM Rust-only switch decision requires finite LMM null ML (ml0)."
        )
    if not hasattr(jxrs, "gwas_lmm_lm_null_lrt_decision"):
        raise RuntimeError(
            "Rust symbol gwas_lmm_lm_null_lrt_decision is unavailable."
        )
    y_arr = np.ascontiguousarray(np.asarray(y_vec, dtype=np.float64).reshape(-1))
    x_arr = np.ascontiguousarray(np.asarray(x_cov, dtype=np.float64))
    switch_to_lm, lrt_stat, pval, _lm_ml0 = jxrs.gwas_lmm_lm_null_lrt_decision(
        y_arr,
        x_arr,
        float(ml0),
        float(alpha),
        True,  # boundary mixture: 0.5*chi2_1 for H0: Va=0
    )
    return bool(switch_to_lm), float(lrt_stat), float(pval)


def run_chunked_gwas_lmm_lm(
    model_name: str,
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
    mmap_limit: bool,
    grm: Union[np.ndarray, None],
    qmatrix: np.ndarray,
    cov_all: Union[np.ndarray , None],
    eff_m: int,
    plot: bool,
    threads: int,
    logger:logging.Logger,
    use_spinner: bool = False,
    snps_only: bool = True,
    eff_snp_by_trait: Union[dict[str, int], None] = None,
    summary_rows: Union[list[dict[str, object]], None] = None,
    saved_paths: Union[list[str], None] = None,
    trait_names: Union[list[str], None] = None,
    show_npve_line: bool = False,
    emit_trait_header: bool = True,
    chunk_size_user_set: bool = True,
    prefer_packed_fullrust: bool = False,
) -> None:
    """
    Run LMM/FastLMM/LM GWAS using a streaming pipeline.

    Important: This function assumes pheno/ids/grm/q/cov have already been prepared
    once (no repeated "Loading phenotype" / "Loading GRM/Q" logs).
    """
    model_map = {
        "lmm": LMM,
        "lm": LM,
        "fastlmm": FastLMM,
    }
    model_key = model_name.lower()
    ModelCls = model_map[model_key]
    base_model_label = {
        "lmm": "LMM",
        "lm": "LM",
        "fastlmm": "FastLMM",
    }[model_key]
    # Keep output file suffixes consistent and lowercase.
    base_model_tag = base_model_label.lower()

    # FastLMM route: reuse prepared streaming GRM and slice to trait samples.
    # Do NOT rebuild packed GRM here.
    if model_key == "fastlmm":
        _log_file_only(
            logger,
            logging.INFO,
            "FastLMM route: using prepared streaming GRM + trait slicing (packed GRM rebuild disabled).",
        )
    if model_key == "lm":
        can_single_entry = (
            _as_plink_prefix(genofile) is not None
            and hasattr(jxrs, "lm_stream_bed_to_tsv")
        )
        if can_single_entry:
            _log_file_only(
                logger,
                logging.INFO,
                "LM route: using rust single-entry BED streaming pipeline "
                "(mmap window supported by default memmap backend; --mmap-limit keeps this explicit).",
            )
            run_lm_stream_bed_single_entry(
                genofile=genofile,
                pheno=pheno,
                ids=ids,
                n_snps=n_snps,
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
                trait_names=trait_names,
                emit_trait_header=emit_trait_header,
                chunk_size_user_set=bool(chunk_size_user_set),
                mmap_limit=bool(mmap_limit),
            )
            return
        packed_fullrust_enabled = bool(prefer_packed_fullrust)
        can_packed = _as_plink_prefix(genofile) is not None
        has_symbols = hasattr(jxrs, "glmf32_packed_assoc_to_tsv")
        is_add = str(genetic_model).lower() == "add"
        if can_packed and has_symbols and is_add:
            if not packed_fullrust_enabled:
                _log_file_only(
                    logger,
                    logging.INFO,
                    "LM route: using streaming orchestrator (fast mode disabled).",
                )
            else:
                _log_file_only(
                    logger,
                    logging.INFO,
                    "LM route: using full-rust packed pipeline (single-entry scan).",
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
                    trait_names=trait_names,
                    emit_trait_header=emit_trait_header,
                )
                return
        reasons: list[str] = []
        if not packed_fullrust_enabled:
            reasons.append("fast mode disabled")
        if not can_packed:
            reasons.append("input is not PLINK BED prefix")
        if not has_symbols:
            reasons.append("required rust symbols unavailable")
        if not is_add:
            reasons.append("non-additive model is not yet supported by LM full-rust route")
        _log_file_only(
            logger,
            logging.INFO,
            "LM route: using streaming orchestrator "
            f"({'; '.join(reasons) if len(reasons) > 0 else 'unknown reason'}).",
        )
    if model_key == "lmm":
        packed_fullrust_enabled = bool(prefer_packed_fullrust)
        can_packed = _as_plink_prefix(genofile) is not None
        has_symbols = hasattr(jxrs, "lmm_reml_assoc_packed_f32")
        has_grm = grm is not None
        is_add = str(genetic_model).lower() == "add"
        if packed_fullrust_enabled and can_packed and has_symbols and has_grm and is_add:
            _log_file_only(
                logger,
                logging.INFO,
                "LMM route: using full-rust packed pipeline (EVD+single-entry scan).",
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
                trait_names=trait_names,
                emit_trait_header=emit_trait_header,
            )
            return
        reasons: list[str] = []
        if not packed_fullrust_enabled:
            reasons.append("fast mode disabled")
        if not can_packed:
            reasons.append("input is not PLINK BED prefix")
        if not has_symbols:
            reasons.append("required rust symbols unavailable")
        if not has_grm:
            reasons.append("GRM is unavailable")
        if not is_add:
            reasons.append("non-additive model is not yet supported by LMM full-rust route")
        _log_file_only(
            logger,
            logging.INFO,
            "LMM route: using streaming orchestrator "
            f"({'; '.join(reasons) if len(reasons) > 0 else 'unknown reason'}).",
        )

    def _apply_genetic_model(geno_chunk: np.ndarray, model: str) -> np.ndarray:
        m = model.lower()
        if m == "add":
            return geno_chunk
        if m == "dom":
            return (
                np.isclose(geno_chunk, 1.0, atol=1e-6)
                | np.isclose(geno_chunk, 2.0, atol=1e-6)
            ).astype(np.float32, copy=False)
        if m == "rec":
            return np.isclose(geno_chunk, 2.0, atol=1e-6).astype(np.float32, copy=False)
        if m == "het":
            return np.isclose(geno_chunk, 1.0, atol=1e-6).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported genetic model: {model}")

    def _transform_allele_labels(
        allele0_list: list[str], allele1_list: list[str], model: str
    ) -> tuple[list[str], list[str]]:
        m = model.lower()
        if m == "add":
            return allele0_list, allele1_list
        out0: list[str] = []
        out1: list[str] = []
        for a0, a1 in zip(allele0_list, allele1_list):
            hom0 = f"{a0}{a0}"
            het = f"{a0}{a1}"
            hom1 = f"{a1}{a1}"
            if m == "dom":
                out0.append(hom0)
                out1.append(f"{het}/{hom1}")
            elif m == "rec":
                out0.append(f"{het}/{hom0}")
                out1.append(hom1)
            elif m == "het":
                out0.append(f"{hom0}/{hom1}")
                out1.append(het)
            else:
                raise ValueError(f"Unsupported genetic model: {model}")
        return out0, out1

    def _heter_keep_mask(geno_chunk: np.ndarray, het: float) -> np.ndarray:
        valid = geno_chunk >= 0
        non_missing = np.sum(valid, axis=1)
        keep = non_missing > 0
        if not np.any(keep):
            return keep
        het_count = np.sum(np.isclose(geno_chunk, 1.0, atol=1e-6) & valid, axis=1)
        het_rate = np.zeros(geno_chunk.shape[0], dtype=np.float32)
        idx = non_missing > 0
        het_rate[idx] = het_count[idx] / non_missing[idx]
        keep &= (het_rate >= het) & (het_rate <= (1.0 - het))
        return keep

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

    # For single-model LMM/FastLMM on PLINK BED input, reuse the shared
    # prepared-chunk scan bus (same Rust read-block path as multi-model mode)
    # to reduce Python per-chunk orchestration overhead.
    if (
        model_key in {"lmm", "fastlmm"}
        and _as_plink_prefix(genofile) is not None
        and hasattr(jxrs, "BedChunkReader")
        and hasattr(getattr(jxrs, "BedChunkReader"), "next_chunk_prepared")
    ):
        _log_file_only(
            logger,
            logging.INFO,
            f"{base_model_label} route: using shared Rust prepared-chunk scan bus (single-model).",
        )
        for trait_idx, pname in enumerate(trait_iter):
            run_chunked_gwas_streaming_shared(
                model_names=[model_key],
                trait_name=str(pname),
                genofile=genofile,
                pheno=pheno,
                ids=ids,
                n_snps=n_snps,
                outprefix=outprefix,
                maf_threshold=maf_threshold,
                max_missing_rate=max_missing_rate,
                genetic_model=genetic_model,
                het_threshold=het_threshold,
                chunk_size=chunk_size,
                mmap_limit=mmap_limit,
                grm=grm,
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
                chunk_size_user_set=bool(chunk_size_user_set),
            )
            if multi_trait_mode and trait_idx < len(trait_iter) - 1:
                logger.info("")
        return

    for trait_idx, pname in enumerate(trait_iter):
        cpu_t0 = process.cpu_times()
        rss0 = process.memory_info().rss
        t0 = time.time()
        peak_rss = rss0
        evd_secs = 0.0

        y_full, sameidx = _trait_values_and_mask(pheno_aligned, str(pname))
        keep_idx = np.flatnonzero(sameidx).astype(np.int64, copy=False)
        n_idv = int(keep_idx.shape[0])
        if n_idv == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")  # single blank line between traits
            continue

        if bool(emit_trait_header):
            _emit_trait_header(
                logger,
                pname,
                n_idv,
                pve=None,
                use_spinner=bool(use_spinner),
                width=60,
            )

        trait_ids = np.asarray(ids[keep_idx], dtype=str)
        y_vec = np.ascontiguousarray(y_full[keep_idx], dtype=np.float64)
        # Build covariate matrix X_cov for this trait
        X_cov = qmatrix[keep_idx]
        if cov_all is not None:
            X_cov = np.concatenate([X_cov, cov_all[keep_idx]], axis=1)

        model_chunk_size = _resolve_stream_scan_chunk_size(
            int(chunk_size),
            int(n_snps),
            use_spinner=bool(use_spinner),
            n_samples_hint=int(n_idv),
            model_keys=model_key,
            user_specified=bool(chunk_size_user_set),
        )
        if (
            str(model_key).lower() == "lm"
            and (not bool(chunk_size_user_set))
            and int(model_chunk_size) != int(chunk_size)
        ):
            _log_file_only(
                logger,
                logging.INFO,
                f"LM auto chunk-size: {int(chunk_size)} -> {int(model_chunk_size)} "
                f"(n={int(n_idv)}).",
            )

        header_pve: Optional[float] = None
        init_log_message: Optional[str] = None
        effective_model_key = str(model_key)
        effective_model_label = str(base_model_label)
        effective_model_tag = str(base_model_tag)
        if model_key in ("lmm", "fastlmm"):
            if grm is None:
                raise ValueError("LMM/fastLMM requires GRM, but GRM was not prepared.")
            Ksub = grm[np.ix_(keep_idx, keep_idx)]
            evd_t0 = time.monotonic()
            evd_label = "LMM" if model_key == "lmm" else "FastLMM"
            evd_desc = f"{evd_label} Eigen-Decomposition"
            with CliStatus(
                f"Running {evd_desc}...",
                enabled=bool(use_spinner),
                use_process=True,
            ) as task:
                try:
                    stage_threads = max(1, int(threads))
                    with _gwas_evd_stage_ctx(stage_threads):
                        mod = ModelCls(y=y_vec, X=X_cov, kinship=Ksub)
                except Exception:
                    task.fail(f"{evd_desc} ...Failed")
                    raise
            try:
                pve_tmp = float(mod.pve)
                if np.isfinite(pve_tmp):
                    header_pve = pve_tmp
            except Exception:
                header_pve = None
            evd_secs = time.monotonic() - evd_t0
            evd_elapsed = format_elapsed(evd_secs)
            if model_key == "fastlmm":
                switch_to_lm, lrt_stat, lrt_p = _fastlmm_switch_to_lm_decision(
                    y_vec=y_vec,
                    x_cov=X_cov,
                    lmm_ml0=getattr(mod, "ML0", None),
                    alpha=0.05,
                )
                if switch_to_lm:
                    logger.warning(
                        f"Warning: FastLMM switch to LM for trait {pname}: "
                        f"null LRT stat={float(lrt_stat):.4g}, p={float(lrt_p):.4g} (>=0.05)."
                    )
                    effective_model_key = "lm"
                    effective_model_label = "LM"
                    effective_model_tag = "lm"
            init_log_message = f"PVE(null) ~ {mod.pve:.3f}; eigen-decomposition [{evd_elapsed}]"
        else:
            mod = ModelCls(y=y_vec, X=X_cov)
            init_log_message = "streaming scan initialized"
        if init_log_message is not None:
            _log_model_line(
                logger,
                effective_model_label,
                init_log_message,
                use_spinner=bool(use_spinner),
            )

        done_snps = 0
        has_results = False
        gm_tag = str(genetic_model).lower()
        if gm_tag == "add":
            out_tsv = f"{outprefix}.{pname}.{effective_model_tag}.tsv"
        else:
            out_tsv = f"{outprefix}.{pname}.{gm_tag}.{effective_model_tag}.tsv"
        tmp_tsv = f"{out_tsv}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        wrote_header = False
        mmap_window_mb = (
            auto_mmap_window_mb(genofile, len(ids), n_snps, model_chunk_size)
            if mmap_limit else None
        )

        # Always pass trait-specific sample IDs to the reader to keep column
        # dimension consistent across BED/VCF/TXT backends.
        sample_sub = trait_ids
        expected_n = int(sample_sub.shape[0])
        scan_threads = int(threads)
        if scan_threads <= 0:
            scan_threads = int(n_cores)
        # Scan stage policy:
        # - Rust/Rayon uses full `-t`
        # - BLAS stays at 1
        # - Python worker fan-out kept at 1 to avoid nested oversubscription.
        max_inflight = 1
        workers = 1
        threads_per_worker = max(1, scan_threads)
        # LM with small n can become orchestration-bound; allow one queued
        # chunk so decode/filter work overlaps with Rust compute.
        if str(effective_model_key).lower() == "lm" and int(expected_n) <= 2_000:
            max_inflight = 2
        prefetch_depth = 2 if scan_threads >= 2 else 1

        process.cpu_percent(interval=None)
        scan_t0 = time.time()
        pbar_total = int(eff_snp_by_trait.get(pname, n_snps))
        pbar_desc = f"{effective_model_label}"
        pbar: Optional[_ProgressAdapter] = None
        scan_warmup_task: Optional[CliStatus] = None
        scan_warmup_active = False
        if bool(use_spinner):
            scan_warmup_task = CliStatus(
                f"Waiting for {effective_model_label} GWAS",
                enabled=True,
                use_process=True,
            )
            scan_warmup_task.__enter__()
            scan_warmup_active = True

        inflight: dict[int, tuple[cf.Future, int, object, np.ndarray]] = {}
        chunk_seq = 0
        interrupted = False

        class _ChunkBatchWriter:
            """
            Lightweight batch writer for GWAS TSV rows.
            Buffers multiple chunks and flushes in one file append call.
            """

            def __init__(self, path: str, *, batch_chunks: int = 8) -> None:
                self.path = str(path)
                self.batch_chunks = max(1, int(batch_chunks))
                self._parts: list[str] = []
                self._has_plrt: Optional[bool] = None
                self._wrote_header = False
                self.rows_written = 0

            def append(self, text: str, *, has_plrt: bool, rows: int) -> None:
                n_rows = int(rows)
                if n_rows <= 0:
                    return
                hp = bool(has_plrt)
                if self._has_plrt is None:
                    self._has_plrt = hp
                elif self._has_plrt != hp:
                    raise ValueError(
                        "Inconsistent result columns across chunks while writing GWAS TSV."
                    )
                self._parts.append(str(text))
                self.rows_written += n_rows
                if len(self._parts) >= self.batch_chunks:
                    self.flush()

            def flush(self) -> None:
                if len(self._parts) == 0:
                    return
                mode = "a" if self._wrote_header else "w"
                with open(self.path, mode, encoding="utf-8", newline="") as fh:
                    if not self._wrote_header:
                        header = "chrom\tpos\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald"
                        if bool(self._has_plrt):
                            header += "\tplrt"
                        fh.write(header + "\n")
                        self._wrote_header = True
                    fh.write("".join(self._parts))
                self._parts.clear()

            @property
            def wrote_header(self) -> bool:
                return bool(self._wrote_header)

        use_rust_assoc_writer = bool(hasattr(jxrs, "GwasAssocTsvWriter"))
        if not use_rust_assoc_writer:
            raise RuntimeError(
                "Rust-only GWAS mode requires GwasAssocTsvWriter symbol. "
                "Rebuild/reinstall JanusX extension."
            )
        writer: object = jxrs.GwasAssocTsvWriter(
            str(tmp_tsv),
            genetic_model=str(genetic_model),
        )

        def _format_chunk_tsv_text(
            results: np.ndarray,
            info_chunk: list[tuple[str, int, str, str]],
            maf_chunk: np.ndarray,
            miss_chunk: np.ndarray,
        ) -> tuple[str, bool, int]:
            if len(info_chunk) == 0:
                return "", bool(np.asarray(results).shape[1] > 3), 0

            chroms, poss, allele0, allele1 = zip(*info_chunk)
            allele0_list = list(allele0)
            allele1_list = list(allele1)
            if genetic_model != "add":
                allele0_list, allele1_list = _transform_allele_labels(
                    allele0_list, allele1_list, genetic_model
                )
            res = np.asarray(results, dtype=np.float64)
            chisq = _chisq_from_gwas_results(res)
            cols = [
                np.asarray(chroms, dtype=object),
                np.asarray(poss, dtype=np.int64).astype(str),
                np.asarray(allele0_list, dtype=object),
                np.asarray(allele1_list, dtype=object),
                np.char.mod("%.4f", np.asarray(maf_chunk, dtype=np.float64)),
                np.char.mod(
                    "%d",
                    np.rint(np.asarray(miss_chunk, dtype=np.float64)).astype(np.int64),
                ),
                np.char.mod("%.4f", res[:, 0]),
                np.char.mod("%.4f", res[:, 1]),
                _format_chisq_output(chisq),
                np.char.mod("%.4e", res[:, 2]),
            ]
            has_plrt = bool(res.shape[1] > 3)
            if has_plrt:
                cols.append(np.char.mod("%.4e", res[:, 3]))
            rows = np.column_stack(cols)
            text = "\n".join("\t".join(map(str, row)) for row in rows)
            if text:
                text += "\n"
            return text, has_plrt, int(rows.shape[0])

        def _drain_completed(*, wait_for_one: bool) -> None:
            nonlocal done_snps, peak_rss, pbar, scan_warmup_active, scan_warmup_task
            if len(inflight) == 0:
                return
            futures = [x[0] for x in inflight.values()]
            if wait_for_one:
                done_set, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            else:
                done_set = {f for f in futures if f.done()}
                if len(done_set) == 0:
                    return

            done_seq = sorted(
                [
                    seq
                    for seq, (fut, _m, _meta, _maf, _miss) in inflight.items()
                    if fut in done_set
                ]
            )
            for seq in done_seq:
                fut, m_chunk, meta_chunk, maf_chunk, miss_chunk = inflight.pop(seq)
                results = fut.result()
                if use_rust_assoc_writer:
                    writer.write_chunk(
                        meta_chunk,
                        np.asarray(maf_chunk, dtype=np.float32),
                        np.asarray(miss_chunk, dtype=np.float32),
                        np.asarray(results, dtype=np.float64),
                    )
                else:
                    chunk_text, has_plrt, n_rows = _format_chunk_tsv_text(
                        results,
                        meta_chunk,
                        maf_chunk,
                        miss_chunk,
                    )
                    writer.append(chunk_text, has_plrt=has_plrt, rows=n_rows)
                done_snps += int(m_chunk)
                if pbar is None:
                    if scan_warmup_active and scan_warmup_task is not None:
                        try:
                            scan_warmup_task.__exit__(None, None, None)
                        except Exception:
                            pass
                        scan_warmup_active = False
                    pbar = _ProgressAdapter(
                        total=pbar_total,
                        desc=pbar_desc,
                        force_animate=True,
                    )
                pbar.update(int(m_chunk))

                mem_info = process.memory_info()
                peak_rss = max(peak_rss, mem_info.rss)
                if done_snps % (10 * model_chunk_size) == 0:
                    mem_gb = mem_info.rss / 1024**3
                    if pbar is not None:
                        pbar.set_postfix(memory=f"{mem_gb:.2f}GB")

        with _gwas_scan_stage_ctx(scan_threads):
            ex = cf.ThreadPoolExecutor(max_workers=workers)
            try:
                bed_prefix = _as_plink_prefix(genofile)
                if bed_prefix is None:
                    raise RuntimeError(
                        "Rust-only GWAS scan requires PLINK BED input/prefix."
                    )
                if not hasattr(jxrs, "BedChunkReader"):
                    raise RuntimeError(
                        "Rust-only GWAS scan requires rust symbol BedChunkReader."
                    )
                bed_reader = _new_bed_chunk_reader_for_stream(
                    bed_prefix=str(bed_prefix),
                    maf_threshold=float(maf_threshold),
                    max_missing_rate=float(max_missing_rate),
                    sample_ids=sample_sub.tolist(),
                    model=str(genetic_model),
                    het_threshold=float(het_threshold),
                    mmap_window_mb=(int(mmap_window_mb) if mmap_window_mb is not None else None),
                    logger=logger,
                )
                if not hasattr(bed_reader, "next_chunk_prepared"):
                    raise RuntimeError(
                        "Rust-only GWAS scan requires BedChunkReader.next_chunk_prepared."
                    )

                while True:
                    out, legacy_snps_only = _bed_next_chunk_prepared_compat(
                        bed_reader=bed_reader,
                        chunk_size=int(model_chunk_size),
                        coding=str(genetic_model),
                        snps_only=bool(snps_only),
                        logger=logger,
                    )
                    if out is None:
                        break
                    geno_center_raw, sites, maf_chunk_raw, miss_chunk_raw = out
                    geno_center = np.ascontiguousarray(
                        np.asarray(geno_center_raw, dtype=np.float32),
                        dtype=np.float32,
                    )
                    maf_chunk = np.asarray(maf_chunk_raw, dtype=np.float32).reshape(-1)
                    miss_chunk = np.asarray(miss_chunk_raw, dtype=np.float32).reshape(-1)
                    if legacy_snps_only and bool(snps_only):
                        geno_center, maf_chunk, sites = _filter_snps_only_chunk_python_fallback(
                            geno_center=geno_center,
                            maf_chunk=maf_chunk,
                            sites=sites,
                        )
                        miss_chunk = miss_chunk[: int(maf_chunk.shape[0])]

                    m_chunk = int(geno_center.shape[0])
                    if m_chunk == 0:
                        continue

                    if len(sites) == 0:
                        continue
                    meta_chunk: object
                    if use_rust_assoc_writer:
                        meta_chunk = sites
                    else:
                        raise RuntimeError("Rust-only GWAS mode requires Rust TSV writer path.")

                    fut = ex.submit(mod.gwas, geno_center, threads=threads_per_worker)
                    inflight[chunk_seq] = (fut, int(m_chunk), meta_chunk, maf_chunk, miss_chunk)
                    chunk_seq += 1

                    if len(inflight) >= max_inflight:
                        _drain_completed(wait_for_one=True)
                    else:
                        _drain_completed(wait_for_one=False)

                while len(inflight) > 0:
                    _drain_completed(wait_for_one=True)
                writer.flush()
                if use_rust_assoc_writer:
                    has_results = int(writer.rows_written) > 0
                    wrote_header = bool(has_results)
                else:
                    wrote_header = bool(writer.wrote_header)
                    has_results = int(writer.rows_written) > 0
                if pbar is not None:
                    pbar.finish()
            except KeyboardInterrupt:
                interrupted = True
                for fut, _m, _meta, _maf, _miss in inflight.values():
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                raise
            finally:
                # Always stop renderer first, then tear down workers.
                if scan_warmup_active and scan_warmup_task is not None:
                    try:
                        scan_warmup_task.__exit__(None, None, None)
                    except Exception:
                        pass
                    scan_warmup_active = False
                if pbar is not None:
                    pbar.close(show_done=False)
                ex.shutdown(wait=False, cancel_futures=True)
                try:
                    writer.close()
                except Exception:
                    pass
                if interrupted and os.path.exists(tmp_tsv):
                    try:
                        os.remove(tmp_tsv)
                    except Exception:
                        pass

        cpu_t1 = process.cpu_times()
        t1 = time.time()
        scan_secs = max(t1 - scan_t0, 0.0)

        wall = t1 - t0
        user_cpu = cpu_t1.user - cpu_t0.user
        sys_cpu = cpu_t1.system - cpu_t0.system
        total_cpu = user_cpu + sys_cpu

        avg_cpu_pct = 100.0 * total_cpu / wall / (n_cores or 1) if wall > 0 else 0.0
        peak_rss_gb = peak_rss / 1024**3
        _log_model_line(
            logger,
            effective_model_label,
            f"avg CPU ~ {avg_cpu_pct:.1f}% of {n_cores} c, peak RSS ~ {peak_rss_gb:.2f} G",
            use_spinner=bool(use_spinner),
        )

        if not has_results:
            logger.info(f"{effective_model_label}: no SNPs passed filters for trait {pname}.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = int(done_snps)
            summary_rows.append(
                {
                    "phenotype": str(pname),
                    "model": effective_model_label,
                    "nidv": int(n_idv),
                    "eff_snp": int(done_snps),
                    "pve": (float(header_pve) if header_pve is not None else None),
                    "avg_cpu": float(avg_cpu_pct),
                    "peak_rss_gb": float(peak_rss_gb),
                    "gwas_time_s": float(evd_secs + scan_secs),
                    "viz_time_s": 0.0,
                    "result_file": "",
                }
            )
            if os.path.exists(tmp_tsv):
                os.remove(tmp_tsv)
            if multi_trait_mode:
                logger.info("")  # single blank line between traits
            continue

        _run_result_write_with_status(
            lambda: os.replace(tmp_tsv, out_tsv),
            use_spinner=bool(use_spinner),
            emit_done_line=False,
        )
        saved_paths.append(str(out_tsv))
        _log_model_line(
            logger,
            effective_model_label,
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
        if pname not in eff_snp_by_trait:
            eff_snp_by_trait[pname] = int(done_snps)
        summary_rows.append(
            {
                "phenotype": str(pname),
                "model": effective_model_label,
                "nidv": int(n_idv),
                "eff_snp": int(done_snps),
                "pve": (float(header_pve) if header_pve is not None else None),
                "avg_cpu": float(avg_cpu_pct),
                "peak_rss_gb": float(peak_rss_gb),
                "gwas_time_s": float(evd_secs + scan_secs),
                "viz_time_s": float(viz_secs),
                "result_file": str(out_tsv),
            }
        )
        time_parts: list[str] = []
        if evd_secs > 0:
            time_parts.append(format_elapsed(evd_secs))
        time_parts.append(format_elapsed(scan_secs))
        if plot:
            time_parts.append(format_elapsed(viz_secs))
        if (
            header_pve is not None
            and np.isfinite(float(header_pve))
            and str(effective_model_label).lower() in {"lmm", "fastlmm"}
        ):
            done_msg = f"{effective_model_label} ...pve {float(header_pve):.3f} [{'/'.join(time_parts)}]"
        else:
            done_msg = f"{effective_model_label} ...Finished [{'/'.join(time_parts)}]"
        _rich_success(logger, done_msg, use_spinner=use_spinner)
        if multi_trait_mode:
            logger.info("")  # ensure blank line between traits


def run_chunked_gwas_streaming_shared(
    model_names: list[str],
    trait_name: str,
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
    mmap_limit: bool,
    grm: Union[np.ndarray, None],
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
    chunk_size_user_set: bool = True,
) -> None:
    """
    Shared-chunk streaming GWAS for multiple models on one trait.

    Decode/filter each chunk once, then run all selected streaming models on the
    same chunk before moving to the next chunk.
    """
    model_order = [str(m).lower() for m in model_names]
    model_map = {"lmm": LMM, "lm": LM, "fastlmm": FastLMM}
    model_order = [m for m in model_order if m in model_map]
    if len(model_order) == 0:
        return

    process = psutil.Process()
    n_cores = detect_effective_threads()

    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    # Shared streaming path is fed by prepare_streaming_context(), where
    # phenotype has already been aligned to `ids` order.
    pheno_aligned = pheno
    ids = np.asarray(ids, dtype=str).reshape(-1)
    pname = str(trait_name)
    y_full, sameidx = _trait_values_and_mask(pheno_aligned, pname)
    keep_idx = np.flatnonzero(sameidx).astype(np.int64, copy=False)
    n_idv = int(keep_idx.shape[0])
    if n_idv == 0:
        logger.warning(f"{pname}: no overlapping samples, skipped.")
        if pname not in eff_snp_by_trait:
            eff_snp_by_trait[pname] = 0
        return

    trait_ids = np.asarray(ids[keep_idx], dtype=str)
    y_vec = np.ascontiguousarray(y_full[keep_idx], dtype=np.float64)
    X_cov = qmatrix[keep_idx]
    if cov_all is not None:
        X_cov = np.concatenate([X_cov, cov_all[keep_idx]], axis=1)
    _emit_trait_header(
        logger,
        pname,
        n_idv,
        pve=None,
        use_spinner=bool(use_spinner),
        width=60,
    )

    def _apply_genetic_model(geno_chunk: np.ndarray, model: str) -> np.ndarray:
        m = model.lower()
        if m == "add":
            return geno_chunk
        if m == "dom":
            return (
                np.isclose(geno_chunk, 1.0, atol=1e-6)
                | np.isclose(geno_chunk, 2.0, atol=1e-6)
            ).astype(np.float32, copy=False)
        if m == "rec":
            return np.isclose(geno_chunk, 2.0, atol=1e-6).astype(np.float32, copy=False)
        if m == "het":
            return np.isclose(geno_chunk, 1.0, atol=1e-6).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported genetic model: {model}")

    def _transform_allele_labels(
        allele0_list: list[str], allele1_list: list[str], model: str
    ) -> tuple[list[str], list[str]]:
        m = model.lower()
        if m == "add":
            return allele0_list, allele1_list
        out0: list[str] = []
        out1: list[str] = []
        for a0, a1 in zip(allele0_list, allele1_list):
            hom0 = f"{a0}{a0}"
            het = f"{a0}{a1}"
            hom1 = f"{a1}{a1}"
            if m == "dom":
                out0.append(hom0)
                out1.append(f"{het}/{hom1}")
            elif m == "rec":
                out0.append(f"{het}/{hom0}")
                out1.append(hom1)
            elif m == "het":
                out0.append(f"{hom0}/{hom1}")
                out1.append(het)
            else:
                raise ValueError(f"Unsupported genetic model: {model}")
        return out0, out1

    def _heter_keep_mask(geno_chunk: np.ndarray, het: float) -> np.ndarray:
        valid = geno_chunk >= 0
        non_missing = np.sum(valid, axis=1)
        keep = non_missing > 0
        if not np.any(keep):
            return keep
        het_count = np.sum(np.isclose(geno_chunk, 1.0, atol=1e-6) & valid, axis=1)
        het_rate = np.zeros(geno_chunk.shape[0], dtype=np.float32)
        idx = non_missing > 0
        het_rate[idx] = het_count[idx] / non_missing[idx]
        keep &= (het_rate >= het) & (het_rate <= (1.0 - het))
        return keep

    use_rust_assoc_writer = bool(hasattr(jxrs, "GwasAssocTsvWriter"))
    if not use_rust_assoc_writer:
        raise RuntimeError(
            "Rust-only GWAS mode requires GwasAssocTsvWriter symbol. "
            "Rebuild/reinstall JanusX extension."
        )

    class _ChunkBatchWriter:
        """
        Lightweight batch writer for GWAS TSV rows.
        Buffers multiple chunks and flushes in one file append call.
        """

        def __init__(self, path: str, *, batch_chunks: int = 8) -> None:
            self.path = str(path)
            self.batch_chunks = max(1, int(batch_chunks))
            self._parts: list[str] = []
            self._has_plrt: Optional[bool] = None
            self._wrote_header = False
            self.rows_written = 0

        def append(self, text: str, *, has_plrt: bool, rows: int) -> None:
            n_rows = int(rows)
            if n_rows <= 0:
                return
            hp = bool(has_plrt)
            if self._has_plrt is None:
                self._has_plrt = hp
            elif self._has_plrt != hp:
                raise ValueError(
                    "Inconsistent result columns across chunks while writing GWAS TSV."
                )
            self._parts.append(str(text))
            self.rows_written += n_rows
            if len(self._parts) >= self.batch_chunks:
                self.flush()

        def flush(self) -> None:
            if len(self._parts) == 0:
                return
            mode = "a" if self._wrote_header else "w"
            with open(self.path, mode, encoding="utf-8", newline="") as fh:
                if not self._wrote_header:
                    header = "chrom\tpos\tallele0\tallele1\tmaf\tmiss\tbeta\tse\tchisq\tpwald"
                    if bool(self._has_plrt):
                        header += "\tplrt"
                    fh.write(header + "\n")
                    self._wrote_header = True
                fh.write("".join(self._parts))
            self._parts.clear()

        @property
        def wrote_header(self) -> bool:
            return bool(self._wrote_header)

    def _format_chunk_tsv_text(
        results: np.ndarray,
        info_chunk: list[tuple[str, int, str, str]],
        maf_chunk: np.ndarray,
        miss_chunk: np.ndarray,
    ) -> tuple[str, bool]:
        if len(info_chunk) == 0:
            return "", bool(np.asarray(results).shape[1] > 3)

        chroms, poss, allele0, allele1 = zip(*info_chunk)
        allele0_list = list(allele0)
        allele1_list = list(allele1)
        if genetic_model != "add":
            allele0_list, allele1_list = _transform_allele_labels(
                allele0_list, allele1_list, genetic_model
            )

        res = np.asarray(results, dtype=np.float64)
        chisq = _chisq_from_gwas_results(res)
        cols = [
            np.asarray(chroms, dtype=object),
            np.asarray(poss, dtype=np.int64).astype(str),
            np.asarray(allele0_list, dtype=object),
            np.asarray(allele1_list, dtype=object),
            np.char.mod("%.4f", np.asarray(maf_chunk, dtype=np.float64)),
            np.char.mod(
                "%d",
                np.rint(np.asarray(miss_chunk, dtype=np.float64)).astype(np.int64),
            ),
            np.char.mod("%.4f", res[:, 0]),
            np.char.mod("%.4f", res[:, 1]),
            _format_chisq_output(chisq),
            np.char.mod("%.4e", res[:, 2]),
        ]
        has_plrt = bool(res.shape[1] > 3)
        if has_plrt:
            cols.append(np.char.mod("%.4e", res[:, 3]))

        out = np.column_stack(cols)
        buf = io.StringIO()
        np.savetxt(buf, out, fmt="%s", delimiter="\t")
        return buf.getvalue(), has_plrt

    model_label_map = {"lmm": "LMM", "lm": "LM", "fastlmm": "FastLMM"}
    share_evd_lmm_fast = ("lmm" in model_order) and ("fastlmm" in model_order)
    shared_lmm_model: Optional[LMM] = None
    shared_ksub: Optional[np.ndarray] = None
    gm_tag = str(genetic_model).lower()

    def _stream_model_outbase(tag: str) -> str:
        t = str(tag).lower()
        if gm_tag == "add":
            return f"{outprefix}.{pname}.{t}"
        return f"{outprefix}.{pname}.{gm_tag}.{t}"

    model_ctxs: list[dict[str, object]] = []
    for mkey in model_order:
        ModelCls = model_map[mkey]
        model_label = model_label_map[mkey]
        model_tag = model_label.lower()
        effective_model_key = str(mkey)
        effective_model_label = str(model_label)
        effective_model_tag = str(model_tag)
        out_base = _stream_model_outbase(model_tag)
        ctx: dict[str, object] = {
            "model_key": mkey,
            "model_label": model_label,
            "model_tag": model_tag,
            "mod": None,
            "evd_secs": 0.0,
            "scan_secs": 0.0,
            "cpu_used": 0.0,
            "peak_rss": int(process.memory_info().rss),
            "done_snps": 0,
            "wrote_header": False,
            "has_results": False,
            "tick": 0,
            "memory_text": "",
            "memory_until_tick": 0,
            "pbar": None,
            "task_id": None,
            "tmp_tsv": f"{out_base}.tsv.tmp.{os.getpid()}.{uuid.uuid4().hex}",
            "out_tsv": f"{out_base}.tsv",
            "writer": None,
            "init_log": None,
        }
        ctx["writer"] = jxrs.GwasAssocTsvWriter(
            str(ctx["tmp_tsv"]),
            genetic_model=str(genetic_model),
        )

        cpu_before = process.cpu_times()
        init_t0 = time.monotonic()
        if mkey in {"lmm", "fastlmm"}:
            if grm is None:
                raise ValueError("LMM/fastLMM requires GRM, but GRM was not prepared.")
            if (
                mkey == "fastlmm"
                and share_evd_lmm_fast
                and shared_lmm_model is not None
            ):
                mod = FastLMM.from_lmm(shared_lmm_model)
                ctx["evd_secs"] = 0.0
                ctx["init_log"] = (
                    f"PVE(null) ~ {mod.pve:.3f}; reusing shared eigen-decomposition from LMM"
                )
            else:
                if shared_ksub is None:
                    shared_ksub = grm[np.ix_(keep_idx, keep_idx)]
                evd_desc = f"{model_label} Eigen-Decomposition"
                with CliStatus(
                    f"Running {evd_desc}...",
                    enabled=bool(use_spinner),
                    use_process=True,
                ) as task:
                    try:
                        stage_threads = max(1, int(threads))
                        with _gwas_evd_stage_ctx(stage_threads):
                            mod = ModelCls(y=y_vec, X=X_cov, kinship=shared_ksub)
                    except Exception:
                        task.fail(f"{evd_desc} ...Failed")
                        raise
                evd_secs = max(time.monotonic() - init_t0, 0.0)
                evd_elapsed = format_elapsed(evd_secs)
                ctx["evd_secs"] = float(evd_secs)
                ctx["init_log"] = f"PVE(null) ~ {mod.pve:.3f}; eigen-decomposition [{evd_elapsed}]"
                if mkey == "lmm" and share_evd_lmm_fast:
                    shared_lmm_model = mod
        else:
            mod = ModelCls(y=y_vec, X=X_cov)
            ctx["init_log"] = "streaming scan initialized"

        pve_now: Optional[float] = None
        if mkey in {"lmm", "fastlmm"}:
            try:
                pve_tmp = float(getattr(mod, "pve"))
                if np.isfinite(pve_tmp):
                    pve_now = pve_tmp
            except Exception:
                pve_now = None

        if mkey == "fastlmm":
            switch_to_lm, lrt_stat, lrt_p = _fastlmm_switch_to_lm_decision(
                y_vec=y_vec,
                x_cov=X_cov,
                lmm_ml0=getattr(mod, "ML0", None),
                alpha=0.05,
            )
            if switch_to_lm:
                logger.warning(
                    f"Warning: FastLMM switch to LM for trait {pname}: "
                    f"null LRT stat={float(lrt_stat):.4g}, p={float(lrt_p):.4g} (>=0.05)."
                )
                effective_model_key = "lm"
                effective_model_label = "LM"
                effective_model_tag = "lm"

        cpu_after = process.cpu_times()
        ctx["cpu_used"] = float(
            (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
        )
        ctx["model_key"] = effective_model_key
        ctx["model_label"] = effective_model_label
        ctx["model_tag"] = effective_model_tag
        if effective_model_tag != model_tag:
            out_base = _stream_model_outbase(effective_model_tag)
            ctx["tmp_tsv"] = (
                f"{out_base}.tsv.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            )
            ctx["out_tsv"] = f"{out_base}.tsv"
            ctx["writer"] = jxrs.GwasAssocTsvWriter(
                str(ctx["tmp_tsv"]),
                genetic_model=str(genetic_model),
            )
        ctx["mod"] = mod
        model_ctxs.append(ctx)

    pbar_total = int(eff_snp_by_trait.get(pname, n_snps))
    use_rich_multi = bool(
        use_spinner
        and should_animate_status("Loading shared GWAS model progress...")
        and rich_progress_available()
    )
    rich_progress = None
    progress_started = False
    shared_warmup_task: Optional[CliStatus] = None
    shared_warmup_active = False
    if bool(use_spinner):
        shared_warmup_task = CliStatus(
            "Waiting for GWAS scan",
            enabled=True,
            use_process=True,
        )
        shared_warmup_task.__enter__()
        shared_warmup_active = True

    def _ensure_shared_progress_started() -> None:
        nonlocal rich_progress, use_rich_multi, progress_started
        nonlocal shared_warmup_task, shared_warmup_active
        if progress_started:
            return
        if shared_warmup_active and shared_warmup_task is not None:
            try:
                shared_warmup_task.__exit__(None, None, None)
            except Exception:
                pass
            shared_warmup_active = False
        if use_rich_multi:
            rich_progress = build_rich_progress(
                description_template="[green]{task.description:<8}",
                show_remaining=False,
                show_percentage=False,
                field_templates=["{task.fields[metric]}"],
                bar_width=_GWAS_PROGRESS_BAR_WIDTH,
                finished_text=" ",
                transient=True,
            )
            if rich_progress is not None:
                rich_progress.start()
                for ctx in model_ctxs:
                    tid = rich_progress.add_task(
                        str(ctx["model_label"]),
                        total=pbar_total,
                        metric=_format_progress_metric(0, pbar_total, None),
                    )
                    ctx["task_id"] = int(tid)
            else:
                use_rich_multi = False

        if not use_rich_multi:
            for ctx in model_ctxs:
                if ctx.get("pbar") is None:
                    ctx["pbar"] = _ProgressAdapter(
                        total=pbar_total,
                        desc=str(ctx["model_label"]),
                        force_animate=True,
                    )
        progress_started = True

    def _metric_text(ctx: dict[str, object]) -> str:
        mem_text = ""
        if (
            str(ctx.get("memory_text", "")).strip() != ""
            and int(ctx.get("tick", 0)) <= int(ctx.get("memory_until_tick", 0))
        ):
            mem_text = str(ctx.get("memory_text", "")).strip()
        return _format_progress_metric(
            int(ctx.get("done_snps", 0)),
            int(pbar_total),
            mem_text if mem_text else None,
        )

    def _advance_ctx(ctx: dict[str, object], m_chunk: int, mem_text: Union[str, None]) -> None:
        done = int(ctx.get("done_snps", 0)) + int(m_chunk)
        tick = int(ctx.get("tick", 0)) + 1
        ctx["done_snps"] = done
        ctx["tick"] = tick
        if mem_text is not None:
            ctx["memory_text"] = str(mem_text).replace(" ", "")
            ctx["memory_until_tick"] = tick + 5
        metric = _metric_text(ctx)
        _ensure_shared_progress_started()
        if use_rich_multi and rich_progress is not None:
            rich_progress.update(int(ctx["task_id"]), advance=int(m_chunk), metric=metric)
        else:
            pbar_obj = ctx.get("pbar")
            if pbar_obj is not None:
                pbar_obj.update(int(m_chunk))
                if mem_text is not None:
                    pbar_obj.set_postfix(memory=str(mem_text))

    model_chunk_size = _resolve_stream_scan_chunk_size(
        int(chunk_size),
        int(n_snps),
        use_spinner=bool(use_spinner),
        n_samples_hint=int(n_idv),
        model_keys=model_order,
        user_specified=bool(chunk_size_user_set),
    )
    if (
        ("lm" in {str(m).lower() for m in model_order})
        and (not bool(chunk_size_user_set))
        and int(model_chunk_size) != int(chunk_size)
    ):
        _log_file_only(
            logger,
            logging.INFO,
            f"LM auto chunk-size: {int(chunk_size)} -> {int(model_chunk_size)} "
            f"(n={int(n_idv)}).",
        )
    mmap_window_mb = (
        auto_mmap_window_mb(genofile, len(ids), n_snps, model_chunk_size)
        if mmap_limit else None
    )
    sample_sub = trait_ids
    expected_n = int(sample_sub.shape[0])

    scan_threads = int(threads)
    if scan_threads <= 0:
        scan_threads = int(n_cores)
    threads_per_model = max(1, scan_threads)
    prefetch_depth = 2 if scan_threads >= 2 else 1

    def _consume_chunk(
        *,
        geno_center: np.ndarray,
        sites: list[object],
        maf_chunk: np.ndarray,
        miss_chunk: np.ndarray,
    ) -> None:
        m_chunk = int(np.asarray(geno_center).shape[0])
        if m_chunk == 0:
            return
        if len(sites) == 0:
            return

        for ctx in model_ctxs:
            cpu_before = process.cpu_times()
            t0 = time.monotonic()
            results = ctx["mod"].gwas(geno_center, threads=threads_per_model)
            elapsed = max(time.monotonic() - t0, 0.0)
            cpu_after = process.cpu_times()
            ctx["scan_secs"] = float(ctx["scan_secs"]) + float(elapsed)
            ctx["cpu_used"] = float(ctx["cpu_used"]) + float(
                (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
            )

            writer = ctx.get("writer")
            if writer is None:
                raise RuntimeError("Internal error: shared GWAS writer not initialized.")
            if use_rust_assoc_writer:
                writer.write_chunk(
                    sites,
                    np.asarray(maf_chunk, dtype=np.float32),
                    np.asarray(miss_chunk, dtype=np.float32),
                    np.asarray(results, dtype=np.float64),
                )
            else:
                info_chunk = [
                    (str(s.chrom), int(s.pos), str(s.ref_allele), str(s.alt_allele))
                    for s in sites
                ]
                if len(info_chunk) == 0:
                    continue
                chunk_text, has_plrt = _format_chunk_tsv_text(
                    results,
                    info_chunk,
                    maf_chunk,
                    miss_chunk,
                )
                writer.append(chunk_text, has_plrt=bool(has_plrt), rows=m_chunk)
            ctx["has_results"] = True

            mem_info = process.memory_info()
            ctx["peak_rss"] = max(int(ctx["peak_rss"]), int(mem_info.rss))
            done_next = int(ctx["done_snps"]) + m_chunk
            mem_text = None
            if done_next % (10 * chunk_size) == 0:
                mem_text = f"{mem_info.rss / 1024**3:.2f}GB"
            _advance_ctx(ctx, m_chunk, mem_text)

    interrupted = False
    with _gwas_scan_stage_ctx(scan_threads):
        try:
            bed_prefix = _as_plink_prefix(genofile)
            if bed_prefix is None:
                raise RuntimeError(
                    "Rust-only GWAS scan requires PLINK BED input/prefix."
                )
            if not hasattr(jxrs, "BedChunkReader"):
                raise RuntimeError(
                    "Rust-only GWAS scan requires rust symbol BedChunkReader."
                )
            try:
                bed_reader = _new_bed_chunk_reader_for_stream(
                    bed_prefix=str(bed_prefix),
                    maf_threshold=float(maf_threshold),
                    max_missing_rate=float(max_missing_rate),
                    sample_ids=sample_sub.tolist(),
                    model=str(genetic_model),
                    het_threshold=float(het_threshold),
                    mmap_window_mb=(int(mmap_window_mb) if mmap_window_mb is not None else None),
                    logger=logger,
                )
            except Exception as ex:
                raise RuntimeError(
                    f"Failed to initialize Rust BedChunkReader for GWAS scan: {ex}"
                ) from ex
            if not hasattr(bed_reader, "next_chunk_prepared"):
                raise RuntimeError(
                    "Rust-only GWAS scan requires BedChunkReader.next_chunk_prepared."
                )
            _log_file_only(
                logger,
                logging.INFO,
                "Shared GWAS scan: using Rust BED prepared-chunk stream.",
            )

            while True:
                out, legacy_snps_only = _bed_next_chunk_prepared_compat(
                    bed_reader=bed_reader,
                    chunk_size=int(model_chunk_size),
                    coding=str(genetic_model),
                    snps_only=bool(snps_only),
                    logger=logger,
                )
                if out is None:
                    break
                geno_center_raw, sites, maf_chunk_raw, miss_chunk_raw = out
                geno_center = np.asarray(geno_center_raw, dtype=np.float32)
                maf_chunk = np.asarray(maf_chunk_raw, dtype=np.float32).reshape(-1)
                miss_chunk = np.asarray(miss_chunk_raw, dtype=np.float32).reshape(-1)
                if legacy_snps_only and bool(snps_only):
                    geno_center, maf_chunk, sites = _filter_snps_only_chunk_python_fallback(
                        geno_center=geno_center,
                        maf_chunk=maf_chunk,
                        sites=sites,
                    )
                    miss_chunk = miss_chunk[: int(maf_chunk.shape[0])]
                geno_center = np.ascontiguousarray(geno_center, dtype=np.float32)
                _consume_chunk(
                    geno_center=geno_center,
                    sites=sites,
                    maf_chunk=maf_chunk,
                    miss_chunk=miss_chunk,
                )

            for ctx in model_ctxs:
                writer = ctx.get("writer")
                if writer is not None:
                    writer.flush()
                    if use_rust_assoc_writer:
                        ctx["has_results"] = int(writer.rows_written) > 0
                        ctx["wrote_header"] = bool(ctx["has_results"])
                    else:
                        ctx["wrote_header"] = bool(writer.wrote_header)
                        ctx["has_results"] = int(writer.rows_written) > 0

            first_done = 0
            if progress_started:
                for ctx in model_ctxs:
                    first_done = first_done or int(ctx.get("done_snps", 0))
                    if use_rich_multi and rich_progress is not None:
                        rich_progress.update(
                            int(ctx["task_id"]),
                            completed=pbar_total,
                            metric=_metric_text(ctx),
                        )
                    else:
                        pbar_obj = ctx.get("pbar")
                        if pbar_obj is not None:
                            pbar_obj.finish()
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            # Always stop any active progress renderer on Ctrl+C / errors.
            if shared_warmup_active and shared_warmup_task is not None:
                try:
                    shared_warmup_task.__exit__(None, None, None)
                except Exception:
                    pass
                shared_warmup_active = False
            if rich_progress is not None:
                try:
                    rich_progress.stop()
                except Exception:
                    pass
            for ctx in model_ctxs:
                pbar_obj = ctx.get("pbar")
                if pbar_obj is not None:
                    try:
                        pbar_obj.close(show_done=False)
                    except Exception:
                        pass
                writer_obj = ctx.get("writer")
                if writer_obj is not None:
                    try:
                        writer_obj.close()
                    except Exception:
                        pass
                if interrupted:
                    tmp_tsv = str(ctx.get("tmp_tsv", ""))
                    if tmp_tsv and os.path.exists(tmp_tsv):
                        try:
                            os.remove(tmp_tsv)
                        except Exception:
                            pass

    first_done = 0
    for ctx in model_ctxs:
        first_done = first_done or int(ctx.get("done_snps", 0))

    if pname not in eff_snp_by_trait:
        eff_snp_by_trait[pname] = int(first_done)

    for ctx in model_ctxs:
        model_label = str(ctx["model_label"])
        done_snps = int(ctx["done_snps"])
        evd_secs = float(ctx["evd_secs"])
        scan_secs = float(ctx["scan_secs"])
        cpu_used = float(ctx["cpu_used"])
        peak_rss_gb = float(int(ctx["peak_rss"]) / 1024**3)
        denom = max(evd_secs + scan_secs, 1e-9)
        avg_cpu_pct = 100.0 * cpu_used / denom / max(1, int(n_cores))
        init_log = str(ctx.get("init_log", "") or "").strip()
        if init_log != "":
            _log_model_line(
                logger,
                model_label,
                init_log,
                use_spinner=bool(use_spinner),
            )
        _log_model_line(
            logger,
            model_label,
            f"avg CPU ~ {avg_cpu_pct:.1f}% of {n_cores} c, peak RSS ~ {peak_rss_gb:.2f} G",
            use_spinner=bool(use_spinner),
        )

        has_results = bool(ctx["has_results"])
        tmp_tsv = str(ctx["tmp_tsv"])
        out_tsv = str(ctx["out_tsv"])
        viz_secs = 0.0
        ctx_pve: Optional[float] = None
        if str(ctx.get("model_key", "")) in {"lmm", "fastlmm"}:
            mod_obj = ctx.get("mod")
            if mod_obj is not None and hasattr(mod_obj, "pve"):
                try:
                    pve_tmp = float(getattr(mod_obj, "pve"))
                    if np.isfinite(pve_tmp):
                        ctx_pve = pve_tmp
                except Exception:
                    ctx_pve = None
        if not has_results:
            logger.info(f"{model_label}: no SNPs passed filters for trait {pname}.")
            summary_rows.append(
                {
                    "phenotype": str(pname),
                    "model": model_label,
                    "nidv": int(n_idv),
                    "eff_snp": int(done_snps),
                    "pve": (float(ctx_pve) if ctx_pve is not None else None),
                    "avg_cpu": float(avg_cpu_pct),
                    "peak_rss_gb": float(peak_rss_gb),
                    "gwas_time_s": float(evd_secs + scan_secs),
                    "viz_time_s": 0.0,
                    "result_file": "",
                }
            )
            if os.path.exists(tmp_tsv):
                os.remove(tmp_tsv)
        if has_results:
            _run_result_write_with_status(
                lambda: os.replace(tmp_tsv, out_tsv),
                use_spinner=bool(use_spinner),
                emit_done_line=False,
            )
            saved_paths.append(str(out_tsv))
            _log_model_line(
                logger,
                model_label,
                f"Results saved to {_display_path(str(out_tsv))}",
                use_spinner=bool(use_spinner),
            )
            if plot:
                viz_secs = _run_fastplot_from_tsv_with_status(
                    out_tsv,
                    y_vec,
                    xlabel=str(pname),
                    outpdf=f"{os.path.splitext(out_tsv)[0]}.svg",
                    use_spinner=bool(use_spinner),
                    emit_done_line=False,
                )

            summary_rows.append(
                {
                    "phenotype": str(pname),
                    "model": model_label,
                    "nidv": int(n_idv),
                    "eff_snp": int(done_snps),
                    "pve": (float(ctx_pve) if ctx_pve is not None else None),
                    "avg_cpu": float(avg_cpu_pct),
                    "peak_rss_gb": float(peak_rss_gb),
                    "gwas_time_s": float(evd_secs + scan_secs),
                    "viz_time_s": float(viz_secs),
                    "result_file": str(out_tsv),
                }
            )

        time_parts: list[str] = []
        if evd_secs > 0:
            time_parts.append(format_elapsed(evd_secs))
        time_parts.append(format_elapsed(scan_secs))
        if plot and has_results:
            time_parts.append(format_elapsed(viz_secs))
        if (
            ctx_pve is not None
            and np.isfinite(float(ctx_pve))
            and str(model_label).lower() in {"lmm", "fastlmm"}
        ):
            done_msg = f"{model_label} ...pve {float(ctx_pve):.3f} [{'/'.join(time_parts)}]"
        else:
            done_msg = f"{model_label} ...Finished [{'/'.join(time_parts)}]"
        _rich_success(logger, done_msg, use_spinner=use_spinner)

    # Keep per-model completion lines self-contained (including optional PVE)
    # to avoid extra standalone summary lines after the model block.


# ======================================================================
# High-memory FarmCPU: full genotype + QK
# ======================================================================
