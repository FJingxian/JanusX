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

from .workflow import (
    CliStatus,
    _ProgressAdapter,
    _align_pheno_to_sample_order,
    _as_plink_prefix,
    _basename_only,
    _cache_lock,
    _emit_plain_info_line,
    _emit_trait_header,
    _emit_warning_line,
    _grm_cache_paths,
    _gwas_cache_prefix_with_params,
    _gwas_eigh_from_grm,
    _inspect_genotype_with_status,
    _load_covariates_for_models,
    _load_phenotype_with_status,
    _log_file_only,
    _normalize_cov_inputs,
    _parse_qcov_dim,
    _pca_cache_path,
    _read_id_file,
    _rich_success,
    _run_fastplot_from_tsv_with_status,
    _trait_values_and_mask,
    detect_effective_threads,
    format_elapsed,
    genotype_cache_prefix,
    inspect_genotype_file,
    jxrs,
    latest_genotype_mtime,
    prepare_packed_ctx_from_plink,
)


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
) -> np.ndarray:
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
    elif geno is not None:
        n = int(geno.shape[1])
    else:
        raise ValueError(
            "FarmCPU Q-matrix build requires either sample_ids or dense geno matrix."
        )

    q_direct_state: dict[str, object] = {
        "q": None,
        "evd_backend": "",
        "evd_secs": 0.0,
    }

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
            np.asarray(packed_ctx_obj["maf"], dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
        packed_n = int(packed_ctx_obj["n_samples"])
        if packed_n != int(n):
            raise ValueError(
                f"Packed context sample size mismatch for FarmCPU-Q: packed={packed_n}, expected={n}."
            )

        row_flip_obj = packed_ctx_obj.get("row_flip", None)
        if row_flip_obj is None:
            if hasattr(jxrs, "bed_packed_row_flip_mask"):
                row_flip_obj = jxrs.bed_packed_row_flip_mask(packed, int(packed_n))
            else:
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
        with _cache_lock(grm_path):
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
                for p in (grm_path, id_path):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

                _farm_log("* Building GRM cache for FarmCPU PCA (packed Rust)...")
                if not isinstance(packed_ctx_preloaded, dict):
                    raise RuntimeError(
                        "FarmCPU Rust-only Q build requires a preloaded packed BED context."
                    )
                try:
                    grm = _build_grm_from_packed_ctx_rust(
                        packed_ctx_preloaded,
                        qdim=int(q_int),
                    )
                except Exception as ex:
                    raise RuntimeError(
                        f"FarmCPU packed-Rust GRM build failed in Rust-only mode: {ex}"
                    ) from ex
                tmp_grm = f"{grm_path}.tmp.{os.getpid()}.npy"
                np.save(tmp_grm, grm)
                os.replace(tmp_grm, grm_path)
                if sid_arr is not None:
                    tmp_id = f"{id_path}.tmp.{os.getpid()}"
                    pd.Series(sid_arr).to_csv(
                        tmp_id, sep="\t", index=False, header=False
                    )
                    os.replace(tmp_id, id_path)
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
        _emit_warning_line(
            logger,
            "PC dimension set to 0; using empty Q matrix.",
            use_spinner=bool(use_spinner),
        )
    else:
        q_path = _pca_cache_path(gfile_prefix, mgrm="1", qdim=int(q_int))
        with _cache_lock(q_path):
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
                    qmatrix = np.loadtxt(q_path, dtype="float32", delimiter="\t")
                    if qmatrix.ndim == 1:
                        qmatrix = qmatrix.reshape(-1, 1)
                    if qmatrix.shape != (n, int(q_int)):
                        raise ValueError(
                            f"PCA cache shape mismatch: expected ({n},{q_int}), got {qmatrix.shape}"
                        )
                    task.complete(
                        f"Loading Q matrix from {q_src} (n={qmatrix.shape[0]}, nPC={qmatrix.shape[1]}) ...Finished"
                    )
            else:
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
                    )
                    qmatrix = np.asarray(eigvec[:, -q_int:], dtype="float32")
                tmp_q = f"{q_path}.tmp.{os.getpid()}"
                np.savetxt(tmp_q, qmatrix, fmt="%.8g", delimiter="\t")
                os.replace(tmp_q, q_path)
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
            sid = sid_arr
            if cov_arr.shape[0] != sid.shape[0]:
                raise ValueError(
                    f"FarmCPU covariate rows ({cov_arr.shape[0]}) do not match sample count ({sid.shape[0]})."
                )
            if not np.array_equal(np.asarray(cov_ids, dtype=str), sid):
                raise ValueError(
                    "FarmCPU covariate sample order does not match genotype sample order after alignment."
                )
            qmatrix = np.concatenate([qmatrix, cov_arr], axis=1)
    else:
        _emit_warning_line(
            logger,
            "Loading covariates (streaming) ...Skipped (none)",
            use_spinner=bool(use_spinner),
        )

    _farm_log(f"Q matrix (FarmCPU) shape: {qmatrix.shape}")
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
) -> dict[str, object]:
    """
    Run FarmCPU in high-memory mode (full genotype + QK + PCA).

    If pheno_preloaded is provided from a non-streaming path, it may be reused.
    When called after streaming GWAS context preparation, FarmCPU must use full
    genotype IDs and therefore reload phenotype/Q/cov on the full ID space.
    """
    phenofile = args.pheno
    outfolder = args.out
    qdim = args.qcov
    cov = args.cov
    snps_only = bool(getattr(args, "snps_only", False))

    def _load_bim_ref_alt_filtered(
        prefix: str,
        keep_mask: np.ndarray,
        *,
        snps_only_mode: bool,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        bim_path = f"{prefix}.bim"
        src = _basename_only(bim_path)
        keep = np.ascontiguousarray(np.asarray(keep_mask, dtype=np.bool_).reshape(-1))
        n_total = int(keep.shape[0])
        if n_total == 0:
            return (
                pd.DataFrame(columns=["chrom", "pos", "allele0", "allele1"]),
                keep,
            )

        n_pre_keep = int(np.sum(keep))
        if n_pre_keep == 0:
            return (
                pd.DataFrame(columns=["chrom", "pos", "allele0", "allele1"]),
                keep,
            )

        # Stream BIM and keep only SNPs that survive numeric filters, optionally
        # applying SNP-only filtering in the same pass.
        chrom = np.empty(n_pre_keep, dtype=object)
        pos = np.empty(n_pre_keep, dtype=np.int64)
        allele0 = np.empty(n_pre_keep, dtype=object)
        allele1 = np.empty(n_pre_keep, dtype=object)

        pbar = _ProgressAdapter(total=n_total, desc=f"Loading site metadata ({src})")
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
                        else:
                            chrom[out] = str(parts[0])
                            try:
                                pval = int(parts[3])
                            except Exception:
                                try:
                                    pval = int(float(parts[3]))
                                except Exception:
                                    pval = 0
                            pos[out] = int(pval)
                            allele0[out] = a0
                            allele1[out] = a1
                            out += 1

                    if lines - done >= 200_000:
                        step = lines - done
                        pbar.update(step)
                        done = lines

            if lines != n_total:
                raise ValueError(
                    f"BIM row count mismatch: bim={lines}, genotype={n_total} ({src})"
                )
            if done < n_total:
                pbar.update(n_total - done)
            pbar.finish()
        finally:
            pbar.close(success_style=True, show_done=True)

        ref_alt_df = pd.DataFrame(
            {
                "chrom": chrom[:out],
                "pos": pos[:out].astype(int, copy=False),
                "allele0": allele0[:out],
                "allele1": allele1[:out],
            }
        )
        return ref_alt_df, keep

    if farmcpu_cache is None:
        t_loading = time.time()
        # If FarmCPU is invoked after streaming models, pheno_preloaded/ids_preloaded
        # are usually intersection-aligned. FarmCPU must operate on full genotype IDs.
        reuse_preloaded_pheno = bool(pheno_preloaded is not None) and (not bool(context_prepared))
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

        # Always inspect full genotype metadata for FarmCPU to avoid reusing
        # streaming-intersection IDs (which can be smaller than packed BED n).
        famid, n_snps = _inspect_genotype_with_status(
            gfile,
            logger,
            use_spinner=use_spinner,
            snps_only=bool(snps_only),
            maf_threshold=float(args.maf),
            max_missing_rate=float(args.geno),
            het_threshold=float(args.het),
        )
        famid = np.asarray(famid, dtype=str)
        geno = None
        packed_ctx: Union[dict[str, object], None] = None

        packed_prefix = _as_plink_prefix(gfile)
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

        can_use_packed = (
            bool(getattr(args, "model", "add") == "add")
            and packed_prefix is not None
        )
        if not can_use_packed:
            raise RuntimeError(
                "FarmCPU Rust-only mode requires additive model with PLINK BED input."
            )
        packed_load_t0 = time.monotonic()
        packed_ctx_raw = None
        pre = preloaded_packed if isinstance(preloaded_packed, dict) else None
        if pre is not None and str(pre.get("prefix", "")) == str(packed_prefix):
            packed_ctx_obj = pre.get("packed_ctx")
            if isinstance(packed_ctx_obj, dict):
                packed_ctx_raw = packed_ctx_obj
                _log_file_only(
                    logger,
                    logging.INFO,
                    "Reusing preloaded packed BED genotype for FarmCPU.",
                )
        if packed_ctx_raw is None:
            packed_status = CliStatus(
                "Loading genotype (Full)...",
                enabled=bool(use_spinner),
            )
            with packed_status as task:
                try:
                    _sample_ids_packed, packed_ctx_raw = prepare_packed_ctx_from_plink(
                        str(packed_prefix),
                        maf=float(args.maf),
                        missing_rate=float(args.geno),
                        snps_only=False,
                        expected_n_samples=int(famid.shape[0]),
                    )
                except Exception:
                    task.fail("Loading genotype (Full) ...Failed")
                    raise
        _log_file_only(
            logger,
            logging.INFO,
            f"Loading genotype (Full, {int(n_snps)} SNPs) "
            f"[{format_elapsed(time.monotonic() - packed_load_t0)}]",
        )

        keep_numeric = np.ascontiguousarray(
            np.asarray(packed_ctx_raw["site_keep"], dtype=np.bool_).reshape(-1), dtype=np.bool_
        )

        ref_alt, keep_final = _load_bim_ref_alt_filtered(
            str(packed_prefix),
            keep_numeric,
            snps_only_mode=bool(snps_only),
        )
        if not np.any(keep_final):
            raise ValueError("After filtering, number of SNPs is zero for FarmCPU.")

        packed_num = np.ascontiguousarray(np.asarray(packed_ctx_raw["packed"], dtype=np.uint8))
        miss_num = np.ascontiguousarray(
            np.asarray(packed_ctx_raw["missing_rate"], dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
        maf_num = np.ascontiguousarray(
            np.asarray(packed_ctx_raw["maf"], dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
        if np.array_equal(keep_final, keep_numeric):
            packed = np.ascontiguousarray(packed_num, dtype=np.uint8)
            miss_arr = np.ascontiguousarray(miss_num, dtype=np.float32)
            maf_arr = np.ascontiguousarray(maf_num, dtype=np.float32)
        else:
            kept_numeric_idx = np.flatnonzero(keep_numeric).astype(np.int64, copy=False)
            keep_local = np.ascontiguousarray(keep_final[kept_numeric_idx], dtype=np.bool_)
            packed = np.ascontiguousarray(packed_num[keep_local], dtype=np.uint8)
            miss_arr = np.ascontiguousarray(miss_num[keep_local], dtype=np.float32)
            maf_arr = np.ascontiguousarray(maf_num[keep_local], dtype=np.float32)

        row_flip_raw = packed_ctx_raw.get("row_flip")
        if row_flip_raw is None:
            if hasattr(jxrs, "bed_packed_row_flip_mask"):
                row_flip_raw = jxrs.bed_packed_row_flip_mask(
                    packed,
                    int(packed_ctx_raw["n_samples"]),
                )
            else:
                row_flip_raw = np.zeros(int(packed.shape[0]), dtype=np.bool_)
        row_flip_arr = np.ascontiguousarray(
            np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        if int(row_flip_arr.shape[0]) != int(packed.shape[0]):
            raise ValueError(
                "Packed row_flip length mismatch for FarmCPU packed context."
            )

        loaded_snps = int(ref_alt.shape[0])
        packed_ctx = {
            "packed": packed,
            "missing_rate": miss_arr,
            "maf": maf_arr,
            "row_flip": row_flip_arr,
            "site_keep": np.ascontiguousarray(keep_final, dtype=np.bool_),
            "n_samples": int(packed_ctx_raw["n_samples"]),
            "source_prefix": str(packed_prefix),
        }

        t_loaded = time.time() - t_loading
        if (not bool(context_prepared)) and (not bool(prepare_only)):
            ns_loaded = int(ref_alt.shape[0])
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
        if int(ref_alt.shape[0]) == 0:
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
        qmatrix = build_qmatrix_farmcpu(
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
            n_snps_hint=int(ref_alt.shape[0]),
            threads=int(args.thread),
            mmap_limit=bool(args.mmap_limit),
            preloaded_packed=preloaded_packed,
            packed_ctx_preloaded=packed_ctx,
        )

        cov_n: Union[int, str] = "NA"
        if len(_normalize_cov_inputs(cov)) > 0:
            cov_n = int(famid.shape[0])
        geno_n = int(famid.shape[0]) if geno is None else int(geno.shape[1])
        pheno_ids_set = set(np.asarray(pheno.index, dtype=str))
        common_n = int(sum(1 for sid in np.asarray(famid, dtype=str) if sid in pheno_ids_set))
        q_n: Union[int, str] = (
            int(qmatrix.shape[0]) if (np.asarray(qmatrix).ndim == 2 and int(qmatrix.shape[1]) > 0) else "NA"
        )
        _emit_plain_info_line(
            logger,
            f"geno={geno_n}, pheno={pheno.shape[0]}, q={q_n}, cov={cov_n} -> {common_n}",
            use_spinner=bool(use_spinner),
        )
        # For packed+Rust path, keep lightweight metadata lists in cache to avoid
        # repeated large temporary string-array conversions in each trait loop.
        ref_alt_cache_obj: object = ref_alt
        if (
            packed_ctx is not None
            and bool(hasattr(jxrs, "farmcpu_packed_to_tsv"))
            and isinstance(ref_alt, pd.DataFrame)
        ):
            ref_alt_cache_obj = {
                "chrom": ref_alt["chrom"].tolist(),
                "pos": pd.to_numeric(ref_alt["pos"], errors="coerce").fillna(0).astype(np.int64).tolist(),
                "allele0": ref_alt["allele0"].tolist(),
                "allele1": ref_alt["allele1"].tolist(),
            }

        farmcpu_cache = {
            "pheno": pheno,
            "famid": famid,
            "geno": geno,
            "packed_ctx": packed_ctx,
            "ref_alt": ref_alt_cache_obj,
            "qmatrix": qmatrix,
        }
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
            packed_num = np.ascontiguousarray(
                np.asarray(packed_obj["packed"], dtype=np.uint8)
            )
            packed_n = int(packed_obj["n_samples"])
            row_flip_obj = packed_obj.get("row_flip")
            if row_flip_obj is None:
                if hasattr(jxrs, "bed_packed_row_flip_mask"):
                    row_flip_obj = jxrs.bed_packed_row_flip_mask(packed_num, packed_n)
                else:
                    row_flip_obj = np.zeros(int(packed_num.shape[0]), dtype=np.bool_)
            row_flip_arr = np.ascontiguousarray(
                np.asarray(row_flip_obj, dtype=np.bool_).reshape(-1),
                dtype=np.bool_,
            )
            if int(row_flip_arr.shape[0]) != int(packed_num.shape[0]):
                raise ValueError(
                    "FarmCPU cache row_flip length mismatch with packed markers."
                )
            packed_ctx = {
                "packed": packed_num,
                "missing_rate": np.ascontiguousarray(
                    np.asarray(packed_obj["missing_rate"], dtype=np.float32)
                ),
                "maf": np.ascontiguousarray(
                    np.asarray(packed_obj["maf"], dtype=np.float32)
                ),
                "row_flip": row_flip_arr,
                "n_samples": packed_n,
            }
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
        ref_alt = farmcpu_cache["ref_alt"]  # type: ignore[assignment]
        qmatrix = np.asarray(farmcpu_cache["qmatrix"], dtype="float32")
    if farmcpu_cache is None:
        packed_sample_idx = None

    if bool(prepare_only):
        return farmcpu_cache if farmcpu_cache is not None else {}

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pheno_aligned, famid = _align_pheno_to_sample_order(pheno, famid)
    trait_iter = list(pheno_aligned.columns) if trait_names is None else [t for t in trait_names if t in pheno_aligned.columns]
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
        n_idv = int(keep_idx.shape[0])
        if packed_ctx is None:
            if geno is None:
                raise ValueError("FarmCPU genotype payload is missing.")
            m_input = np.ascontiguousarray(geno[:, keep_idx], dtype=np.float32)
            sample_idx_arg = None
            maf = (m_input.mean(axis=1) / 2.0).astype(np.float32, copy=False)
            eff_snp = int(m_input.shape[0])
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
        peak_rss = process.memory_info().rss
        farm_iter = max(1, int(getattr(args, "farmcpu_iter", 20)))
        farm_threshold = float(getattr(args, "farmcpu_threshold", 0.05))
        farm_qtn_bound_raw = getattr(args, "farmcpu_qtn_bound", None)
        farm_qtn_bound = None if farm_qtn_bound_raw is None else int(farm_qtn_bound_raw)
        farm_nbin = max(1, int(getattr(args, "farmcpu_nbin", 5)))
        farm_szbin = [float(x) for x in getattr(args, "farmcpu_bin_size", [5e5, 5e6, 5e7])]
        farm_label = f"FarmCPU-{phename} (n={n_idv})"
        farm_pbar = _ProgressAdapter(
            total=farm_iter,
            desc=farm_label,
            force_animate=bool(use_spinner),
        )
        farm_state = {"done": 0}

        def _farmcpu_progress(done: int, total: int) -> None:
            nonlocal peak_rss
            target = int(max(0, min(int(total), int(done))))
            delta = target - int(farm_state["done"])
            if delta > 0:
                farm_pbar.update(delta)
                farm_state["done"] = target
            try:
                peak_rss = max(peak_rss, process.memory_info().rss)
            except Exception:
                pass

        if isinstance(ref_alt, dict):
            chrom_col = [str(x) for x in ref_alt.get("chrom", [])]
            pos_col = [int(x) for x in ref_alt.get("pos", [])]
            allele0_col = [str(x) for x in ref_alt.get("allele0", [])]
            allele1_col = [str(x) for x in ref_alt.get("allele1", [])]
        else:
            chrom_col = ref_alt["chrom"].tolist()
            pos_col = ref_alt["pos"].astype(np.int64, copy=False).tolist()
            allele0_col = ref_alt["allele0"].tolist()
            allele1_col = ref_alt["allele1"].tolist()
        out_tsv = os.path.join(outfolder, f"{prefix}.{phename}.farmcpu.tsv")
        pseudo_tsv_hint = os.path.join(outfolder, f"{prefix}.{phename}.farmcpu.qtn.tsv")
        n_pseudo_qtn = 0
        pseudo_tsv: Union[str, None] = None

        use_rust_controller = (
            packed_ctx is not None
            and bool(hasattr(jxrs, "farmcpu_packed_to_tsv"))
        )
        if use_rust_controller:
            packed_payload = m_input
            if not isinstance(packed_payload, dict):
                raise ValueError("Internal error: expected packed payload for Rust FarmCPU controller.")
            rust_qtn_written = 0
            try:
                packed_arr = packed_payload["packed"]
                row_flip_arr = packed_payload["row_flip"]
                maf_arr = packed_payload["maf"]
                sample_idx_use = (
                    None
                    if sample_idx_arg is None
                    else np.ascontiguousarray(np.asarray(sample_idx_arg, dtype=np.int64).reshape(-1))
                )
                if hasattr(jxrs, "gwas_packed_unified_to_tsv"):
                    jobs = [
                        {
                            "model": "farmcpu",
                            "trait": str(phename),
                            "out_tsv": str(out_tsv),
                            "y": np.ascontiguousarray(p_sub, dtype=np.float64).reshape(-1),
                            "x_cov": np.ascontiguousarray(q_sub, dtype=np.float64),
                            "sample_indices": sample_idx_use,
                            "threshold": float(farm_threshold),
                            "max_iter": int(farm_iter),
                            "qtn_bound": farm_qtn_bound,
                            "nbin": int(farm_nbin),
                            "szbin": [float(x) for x in farm_szbin],
                            "pseudo_tsv": str(pseudo_tsv_hint),
                            "scan_progress_callback": _farmcpu_progress,
                        }
                    ]
                    _res = jxrs.gwas_packed_unified_to_tsv(
                        jobs,
                        packed_arr,
                        int(packed_payload["n_samples"]),
                        row_flip_arr,
                        maf_arr,
                        chrom_col,
                        pos_col,
                        allele0_col,
                        allele1_col,
                        int(max(1, int(getattr(args, "chunksize", 10000)))),
                        int(args.thread),
                        None,
                        int(max(1, int(getattr(args, "chunksize", 10000)))),
                    )
                    r0 = _res[0]
                    n_pseudo_qtn = int(r0.get("pseudo_rows", 0))
                    rust_qtn_written = int(r0.get("written_rows", 0))
                    _m_written = int(r0.get("written_rows", 0))
                else:
                    _m_written, n_pseudo_qtn, rust_qtn_written = jxrs.farmcpu_packed_to_tsv(
                        np.ascontiguousarray(p_sub, dtype=np.float64).reshape(-1),
                        np.ascontiguousarray(q_sub, dtype=np.float64),
                        chrom_col,
                        pos_col,
                        allele0_col,
                        allele1_col,
                        packed_arr,
                        int(packed_payload["n_samples"]),
                        row_flip_arr,
                        maf_arr,
                        out_tsv,
                        sample_idx_use,
                        float(farm_threshold),
                        int(farm_iter),
                        farm_qtn_bound,
                        int(farm_nbin),
                        [float(x) for x in farm_szbin],
                        int(args.thread),
                        _farmcpu_progress,
                        pseudo_tsv_hint,
                    )
                farm_pbar.finish()
            finally:
                farm_pbar.close(show_done=False)
            n_pseudo_qtn = int(max(0, n_pseudo_qtn))
            if int(rust_qtn_written) > 0 and os.path.exists(pseudo_tsv_hint):
                pseudo_tsv = pseudo_tsv_hint
        else:
            try:
                farm_pbar.close(show_done=False)
            except Exception:
                pass
            raise RuntimeError(
                "FarmCPU Rust-only mode requires farmcpu_packed_to_tsv (or unified packed controller). "
                "Python FarmCPU fallback path is disabled."
            )

        gwas_secs = max(time.time() - gwas_t0, 0.0)
        peak_rss = max(peak_rss, process.memory_info().rss)
        cpu_t1 = process.cpu_times()
        wall = max(time.time() - t0, 1e-9)
        cpu_used = (cpu_t1.user + cpu_t1.system) - (cpu_t0.user + cpu_t0.system)
        avg_cpu = 100.0 * cpu_used / (wall * max(1, n_cores))
        peak_rss_gb = peak_rss / (1024 ** 3)

        viz_secs = 0.0
        if args.plot:
            viz_secs = _run_fastplot_from_tsv_with_status(
                out_tsv,
                p_sub,
                xlabel=phename,
                outpdf=os.path.join(outfolder, f"{prefix}.{phename}.farmcpu.svg"),
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
                "gwas_time_s": float(gwas_secs),
                "viz_time_s": float(viz_secs),
                "result_file": str(out_tsv),
            }
        )
        saved_paths.append(str(out_tsv))
        if pseudo_tsv is not None and os.path.exists(pseudo_tsv):
            saved_paths.append(str(pseudo_tsv))

        farm_times = [format_elapsed(gwas_secs)]
        if args.plot:
            farm_times.append(format_elapsed(viz_secs))
        farm_done_msg = f"FarmCPU ...Found {n_pseudo_qtn} QTNs [{'/'.join(farm_times)}]"
        if (not bool(emit_trait_header)) and multi_trait_mode:
            farm_done_msg = f"FarmCPU({phename}) ...Found {n_pseudo_qtn} QTNs [{'/'.join(farm_times)}]"
        _rich_success(logger, farm_done_msg, use_spinner=use_spinner)
        if multi_trait_mode:
            logger.info("")
    return farmcpu_cache


# ======================================================================
# CLI
# ======================================================================
