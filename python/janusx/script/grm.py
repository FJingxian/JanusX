# -*- coding: utf-8 -*-
"""
JanusX: Efficient Genetic Relationship Matrix (GRM) Calculator

Design overview
---------------
Input:
  - VCF   : genotype in VCF/VCF.GZ format
  - BFILE : genotype in PLINK binary format (.bed/.bim/.fam prefix)
  - FILE  : genotype numeric matrix (.txt/.tsv/.csv/.npy) with sibling prefix.id

Implementation:
  - GRM construction is executed by Rust kernels:
      * `grm_stream_bed_f32` (memmap/windowed BED)
      * `grm_packed_bed_f32` (packed BED)
  - For VCF/HMP/TXT-like input, CLI materializes PLINK BED cache first.
  - SNPs are filtered by MAF and missing rate inside Rust kernels.

Output:
  - {prefix}.cGRM.txt / {prefix}.sGRM.txt    : GRM as plain text (if --npy is not used)
  - {prefix}.cGRM.txt.id / {prefix}.sGRM.txt.id : sample IDs for text GRM
  - {prefix}.cGRM.npy / {prefix}.sGRM.npy    : binary GRM (if --npy is used)
  - {prefix}.cGRM.npy.id / {prefix}.sGRM.npy.id : sample IDs for NPY GRM
"""

import os
import time
import socket
import argparse
import logging
import json
from contextlib import contextmanager
from typing import Union

import numpy as np
import psutil
from janusx.gfreader import (
    calc_decode_block_rows_from_memory_mb,
    inspect_genotype_file,
    auto_mmap_window_mb,
    prepare_cli_input_cache,
)
from ._common.log import setup_logging
from ._common.config_render import emit_cli_configuration
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    format_path_for_display,
    ensure_plink_prefix_exists,
)
from ._common.progress import ProgressAdapter
from ._common.status import CliStatus, format_elapsed, log_success
from ._common.genocache import configure_genotype_cache_from_out
from ._common.genoio import determine_genotype_source as _determine_genotype_source
from ._common.threads import (
    apply_blas_thread_env,
    detect_rust_blas_backend,
    detect_effective_threads,
    format_requested_thread_usage,
    get_rust_blas_threads,
    maybe_warn_non_openblas,
    require_openblas_by_default,
    set_rust_blas_threads,
)
from ._common.grmstable import (
    save_grm_npy_blocked,
)
from ._common.grmio import read_id_file, resolve_grm_id_path

try:
    from janusx.janusx import (
        grm_packed_bed_f32 as _grm_packed_bed_f32,
        grm_stream_bed_f32 as _grm_stream_bed_f32,
        grm_stream_bed_f32_to_npy as _grm_stream_bed_f32_to_npy,
    )
except Exception:
    _grm_packed_bed_f32 = None
    _grm_stream_bed_f32 = None
    _grm_stream_bed_f32_to_npy = None

try:
    from janusx.janusx import (
        spgrm_bed_to_jxgrm as _spgrm_bed_to_jxgrm,
        spgrm_dense_f32_to_jxgrm as _spgrm_dense_f32_to_jxgrm,
        spgrm_dense_npy_to_jxgrm as _spgrm_dense_npy_to_jxgrm,
    )
except Exception:
    _spgrm_bed_to_jxgrm = None
    _spgrm_dense_f32_to_jxgrm = None
    _spgrm_dense_npy_to_jxgrm = None


DEFAULT_BED_MEMORY_MB = 512.0


def _is_plink_prefix_path(path_or_prefix: str) -> bool:
    p = str(path_or_prefix).strip()
    if p == "":
        return False
    low = p.lower()
    if low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam"):
        p = p[:-4]
    return all(os.path.isfile(f"{p}.{ext}") for ext in ("bed", "bim", "fam"))


def _normalize_memory_mb(memory_mb: Union[int, float, None]) -> float:
    if memory_mb is None:
        return float(DEFAULT_BED_MEMORY_MB)
    mb = float(memory_mb)
    if (not np.isfinite(mb)) or mb <= 0.0:
        raise ValueError(f"--memory must be a finite value > 0, got {memory_mb}")
    return float(mb)


def _decode_block_rows_from_memory_mb(
    n_samples: int,
    n_snps: int,
    memory_mb: Union[int, float],
    *,
    streaming: bool,
) -> int:
    rows = calc_decode_block_rows_from_memory_mb(
        int(n_samples),
        float(memory_mb),
        buffers=(2 if streaming else 1),
        max_rows=max(1, int(n_snps)),
    )
    return max(1, int(rows if rows is not None else 1))


@contextmanager
def _bed_block_target_env(memory_mb: Union[int, float, None]) -> None:
    prev = os.environ.get("JX_BED_BLOCK_TARGET_MB")
    if memory_mb is not None:
        os.environ["JX_BED_BLOCK_TARGET_MB"] = f"{float(memory_mb):.6g}"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("JX_BED_BLOCK_TARGET_MB", None)
        else:
            os.environ["JX_BED_BLOCK_TARGET_MB"] = prev


def _resolve_rust_grm_input(
    genofile: str,
    *,
    from_vcf: bool,
    from_hmp: bool,
    from_file: bool,
) -> str:
    if _is_plink_prefix_path(str(genofile)):
        p = str(genofile).strip()
        low = p.lower()
        return p[:-4] if (low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam")) else p

    if not bool(from_vcf or from_hmp or from_file):
        raise RuntimeError(
            f"Rust GRM backend requires PLINK BED input, got: {genofile}"
        )

    delim = "," if (from_file and str(genofile).lower().endswith(".csv")) else None
    cached = prepare_cli_input_cache(
        str(genofile),
        snps_only=bool(from_vcf),
        delimiter=delim,
        prefer_plink_for_txt=True,
    )
    if not _is_plink_prefix_path(str(cached)):
        raise RuntimeError(
            "Rust GRM backend requires PLINK BED-compatible input. "
            f"Failed to materialize BED cache from: {genofile}"
        )
    return str(cached)


def _grm_method_tag(method: int) -> str:
    return "cGRM" if int(method) == 1 else "sGRM"


def _dense_grm_auto_prefix(path: str) -> str:
    base = os.path.basename(str(path).rstrip("/\\"))
    low = base.lower()
    if low.endswith(".npy"):
        base = base[:-4]
    if base.startswith("~"):
        base = base[1:]
    return base


def _infer_dense_grm_method_tag(path: str, fallback_method: int) -> str:
    base = os.path.basename(str(path))
    if ".cGRM" in base or base.endswith("cGRM.npy"):
        return "cGRM"
    if ".sGRM" in base or base.endswith("sGRM.npy"):
        return "sGRM"
    return _grm_method_tag(fallback_method)


def _write_sparse_grm_meta(
    sparse_path: str,
    *,
    cutoff: float,
    source: str,
    method: Union[int, None],
    maf_threshold: Union[float, None],
    max_missing_rate: Union[float, None],
    het_threshold: Union[float, None],
    snps_only: bool,
    dense_grm_path: Union[str, None] = None,
) -> None:
    meta = {
        "abs_threshold": False,
        "cutoff": float(cutoff),
        "dense_grm_path": (
            os.path.normpath(str(dense_grm_path))
            if dense_grm_path is not None and str(dense_grm_path).strip() != ""
            else None
        ),
        "het_threshold": (
            None if het_threshold is None else float(het_threshold)
        ),
        "maf_threshold": (
            None if maf_threshold is None else float(maf_threshold)
        ),
        "max_missing_rate": (
            None if max_missing_rate is None else float(max_missing_rate)
        ),
        "method": None if method is None else int(method),
        "sample_hash": "all",
        "sample_n": None,
        "snps_only": bool(snps_only),
        "source": str(source),
    }
    with open(f"{sparse_path}.meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=True, sort_keys=True)


def _select_cli_grm_backend(*, fast: bool = False) -> tuple[str, str]:
    if bool(fast):
        if _grm_packed_bed_f32 is not None:
            return ("packed-bed", "fast flag: force packed BED backend")
        raise RuntimeError(
            "Packed BED GRM kernel is unavailable. Rebuild JanusX extension to export "
            "`grm_packed_bed_f32`."
        )
    if _grm_stream_bed_f32 is not None:
        return ("memmap-bed", "full-sample GRM build")
    if _grm_packed_bed_f32 is not None:
        return ("packed-bed", "memmap GRM kernel unavailable")
    raise RuntimeError(
        "No Rust BED GRM kernel is available. Rebuild JanusX extension to export "
        "`grm_stream_bed_f32` or `grm_packed_bed_f32`."
    )


def build_grm_streaming(
    genofile: str,
    n_samples: int,
    n_snps: int,
    method: int,
    maf_threshold: float,
    max_missing_rate: float,
    block_rows: int,
    mmap_window_mb: Union[int , None],
    threads: int,
    memory_mb: Union[float, None],
    stage_timing: bool,
    logger,
) -> tuple[np.ndarray, int]:
    """
    Build the GRM in memmap mode using Rust single-entry BED kernel.

    Parameters
    ----------
    genofile : str
        Path or prefix to the genotype file (VCF or PLINK bfile).
    n_samples : int
        Number of samples.
    n_snps : int
        Total SNP count reported by inspect_genotype_file.
    method : int
        GRM method:
          - 1: centered GRM
          - 2: standardized/weighted GRM
    maf_threshold : float
        MAF filter threshold passed to the Rust reader.
    max_missing_rate : float
        Missing-rate filter threshold passed to the Rust reader.
    block_rows : int
        Number of SNP rows per decode block.
    logger : logging.Logger
        Logger for progress messages.

    Returns
    -------
    grm : np.ndarray
        GRM matrix of shape (n_samples, n_samples).
    eff_m : int
        Effective number of SNPs after filtering.
    """
    if _grm_stream_bed_f32 is None:
        raise RuntimeError(
            "Rust memmap GRM kernel is unavailable. Rebuild JanusX extension to export "
            "`grm_stream_bed_f32`."
        )
    pbar = ProgressAdapter(
        total=n_snps,
        desc="GRM (rust-memmap)",
        emit_done=False,
        force_animate=True,
    )
    stream_t0 = time.monotonic()
    process = psutil.Process()
    mem_tick_span = max(1, 10 * int(block_rows))
    last_done = 0

    def _progress_cb(done: int, _total: int) -> None:
        nonlocal last_done
        d = int(done)
        if d > last_done:
            pbar.update(d - last_done)
            last_done = d
        if d % mem_tick_span == 0:
            mem = process.memory_info().rss / 1024**3
            pbar.set_postfix(memory=f"{mem:.2f} GB")

    prev_stage_timing = os.environ.get("JX_GRM_STREAM_STAGE_TIMING")
    stage_timing_set = False
    if bool(stage_timing):
        os.environ["JX_GRM_STREAM_STAGE_TIMING"] = "1"
        stage_timing_set = True

    try:
        try:
            with _bed_block_target_env(memory_mb):
                grm_raw, eff_m, stream_n = _grm_stream_bed_f32(
                    str(genofile),
                    method=int(method),
                    maf_threshold=float(maf_threshold),
                    max_missing_rate=float(max_missing_rate),
                    block_cols=max(1, int(block_rows)),
                    threads=max(1, int(threads)),
                    progress_callback=_progress_cb,
                    progress_every=max(1, int(block_rows)),
                    mmap_window_mb=(int(mmap_window_mb) if mmap_window_mb is not None else None),
                )
        finally:
            if stage_timing_set:
                if prev_stage_timing is None:
                    os.environ.pop("JX_GRM_STREAM_STAGE_TIMING", None)
                else:
                    os.environ["JX_GRM_STREAM_STAGE_TIMING"] = prev_stage_timing
    finally:
        pbar.finish()
        pbar.close()
    stream_elapsed = max(0.0, time.monotonic() - stream_t0)
    if int(stream_n) != int(n_samples):
        raise RuntimeError(
            f"Memmap sample count mismatch: memmap={int(stream_n)}, expected={int(n_samples)}"
        )
    grm = np.ascontiguousarray(np.asarray(grm_raw, dtype=np.float32))
    eff_m = int(eff_m)

    log_success(
        logger,
        f"GRM (Effective SNPs: {eff_m}, rust-memmap kernel) ...Finished [{format_elapsed(stream_elapsed)}]",
        force_color=True,
    )
    return grm, eff_m


def build_grm_streaming_to_npy(
    genofile: str,
    out_npy_path: str,
    n_samples: int,
    n_snps: int,
    method: int,
    maf_threshold: float,
    max_missing_rate: float,
    block_rows: int,
    mmap_window_mb: Union[int, None],
    threads: int,
    memory_mb: Union[float, None],
    stage_timing: bool,
    logger,
) -> int:
    """
    Build the GRM in memmap mode and stream the final dense matrix directly to NPY.

    This avoids materializing the full GRM again in Python just to write the file.
    """
    if _grm_stream_bed_f32_to_npy is None:
        raise RuntimeError(
            "Rust memmap GRM->NPY kernel is unavailable. Rebuild JanusX extension to export "
            "`grm_stream_bed_f32_to_npy`."
        )

    pbar = ProgressAdapter(
        total=n_snps,
        desc="GRM (rust-memmap)",
        emit_done=False,
        force_animate=True,
    )
    stream_t0 = time.monotonic()
    process = psutil.Process()
    mem_tick_span = max(1, 10 * int(block_rows))
    last_done = 0

    def _progress_cb(done: int, _total: int) -> None:
        nonlocal last_done
        d = int(done)
        if d > last_done:
            pbar.update(d - last_done)
            last_done = d
        if d % mem_tick_span == 0:
            mem = process.memory_info().rss / 1024**3
            pbar.set_postfix(memory=f"{mem:.2f} GB")

    prev_stage_timing = os.environ.get("JX_GRM_STREAM_STAGE_TIMING")
    stage_timing_set = False
    if bool(stage_timing):
        os.environ["JX_GRM_STREAM_STAGE_TIMING"] = "1"
        stage_timing_set = True

    try:
        try:
            with _bed_block_target_env(memory_mb):
                eff_m, stream_n = _grm_stream_bed_f32_to_npy(
                    str(genofile),
                    str(out_npy_path),
                    method=int(method),
                    maf_threshold=float(maf_threshold),
                    max_missing_rate=float(max_missing_rate),
                    block_cols=max(1, int(block_rows)),
                    threads=max(1, int(threads)),
                    progress_callback=_progress_cb,
                    progress_every=max(1, int(block_rows)),
                    mmap_window_mb=(int(mmap_window_mb) if mmap_window_mb is not None else None),
                )
        finally:
            if stage_timing_set:
                if prev_stage_timing is None:
                    os.environ.pop("JX_GRM_STREAM_STAGE_TIMING", None)
                else:
                    os.environ["JX_GRM_STREAM_STAGE_TIMING"] = prev_stage_timing
    finally:
        pbar.finish()
        pbar.close()
    stream_elapsed = max(0.0, time.monotonic() - stream_t0)
    if int(stream_n) != int(n_samples):
        raise RuntimeError(
            f"Memmap sample count mismatch: memmap={int(stream_n)}, expected={int(n_samples)}"
        )
    eff_m = int(eff_m)

    log_success(
        logger,
        f"GRM (Effective SNPs: {eff_m}, rust-memmap kernel -> npy) ...Finished [{format_elapsed(stream_elapsed)}]",
        force_color=True,
    )
    return eff_m


def build_grm_packed_bed(
    genofile: str,
    n_samples: int,
    n_snps: int,
    method: int,
    maf_threshold: float,
    max_missing_rate: float,
    block_rows: int,
    threads: int,
    memory_mb: Union[float, None],
    stage_timing: bool,
    logger,
) -> tuple[np.ndarray, int]:
    if _grm_packed_bed_f32 is None:
        raise RuntimeError(
            "Packed BED GRM kernel is unavailable. Rebuild JanusX extension to export "
            "`grm_packed_bed_f32`."
        )

    pbar = ProgressAdapter(
        total=n_snps,
        desc="GRM (packed-bed)",
        emit_done=False,
        force_animate=True,
    )
    stream_t0 = time.monotonic()
    process = psutil.Process()
    mem_tick_span = max(1, 10 * int(block_rows))
    last_done = 0

    def _progress_cb(done: int, _total: int) -> None:
        nonlocal last_done
        d = int(done)
        if d > last_done:
            pbar.update(d - last_done)
            last_done = d
        if d % mem_tick_span == 0:
            mem = process.memory_info().rss / 1024**3
            pbar.set_postfix(memory=f"{mem:.2f} GB")

    prev_stage_timing = os.environ.get("JX_GRM_PACKED_STAGE_TIMING")
    stage_timing_set = False
    if bool(stage_timing):
        os.environ["JX_GRM_PACKED_STAGE_TIMING"] = "1"
        stage_timing_set = True

    try:
        try:
            with _bed_block_target_env(memory_mb):
                grm_raw, eff_m, packed_n = _grm_packed_bed_f32(
                    str(genofile),
                    method=int(method),
                    maf_threshold=float(maf_threshold),
                    max_missing_rate=float(max_missing_rate),
                    block_cols=max(1, int(block_rows)),
                    threads=max(1, int(threads)),
                    progress_callback=_progress_cb,
                    progress_every=max(1, int(block_rows)),
                )
            if int(packed_n) != int(n_samples):
                raise RuntimeError(
                    f"Packed sample count mismatch: packed={int(packed_n)}, expected={int(n_samples)}"
                )
            grm = np.ascontiguousarray(np.asarray(grm_raw, dtype=np.float32))
            eff_m = int(eff_m)
        finally:
            if stage_timing_set:
                if prev_stage_timing is None:
                    os.environ.pop("JX_GRM_PACKED_STAGE_TIMING", None)
                else:
                    os.environ["JX_GRM_PACKED_STAGE_TIMING"] = prev_stage_timing
    finally:
        pbar.finish()
        pbar.close()

    stream_elapsed = max(0.0, time.monotonic() - stream_t0)
    log_success(
        logger,
        f"GRM (Effective SNPs: {eff_m}) ...Finished "
        f"[{format_elapsed(stream_elapsed)}]",
        force_color=True,
    )
    return grm, eff_m


def build_sparse_grm_packed_bed(
    genofile: str,
    out_prefix: str,
    n_samples: int,
    n_snps: int,
    method: int,
    kinship_cutoff: float,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    mmap_window_mb: Union[int, None],
    threads: int,
    block_target_mb: Union[float, None],
    stage_timing: bool,
    logger,
) -> tuple[str, int, int]:
    if _spgrm_bed_to_jxgrm is None:
        raise RuntimeError(
            "Sparse GRM BED kernel is unavailable. Rebuild JanusX extension to export "
            "`spgrm_bed_to_jxgrm`."
        )
    logger.info(
        "Sparse GRM route selected stream-bed: "
        f"n={n_samples}, m={n_snps}, sample-blocked sparse CSC writer."
    )
    pbar = ProgressAdapter(
        total=1,
        desc="Sparse GRM (stream-bed)",
        emit_done=False,
        force_animate=True,
    )
    build_t0 = time.monotonic()
    process = psutil.Process()
    last_done = 0
    last_total = 1

    def _progress_cb(done: int, total: int) -> None:
        nonlocal last_done, last_total
        d = max(0, int(done))
        t = max(1, int(total))
        if t != last_total:
            pbar.set_total(t)
            last_total = t
        d = min(d, t)
        if d > last_done:
            pbar.update(d - last_done)
            last_done = d
        mem = process.memory_info().rss / 1024**3
        pbar.set_postfix(memory=f"{mem:.2f} GB")

    try:
        sparse_path, sparse_n, sparse_nnz = _spgrm_bed_to_jxgrm(
            str(genofile),
            out_prefix=str(out_prefix),
            method=int(method),
            threshold=float(kinship_cutoff),
            maf_threshold=float(maf_threshold),
            max_missing_rate=float(max_missing_rate),
            het_threshold=0.0,
            snps_only=False,
            block_rows=0,
            sample_block=0,
            threads=max(1, int(threads)),
            mmap_window_mb=(int(mmap_window_mb) if mmap_window_mb is not None else None),
            progress_callback=_progress_cb,
            progress_every=1,
        )
    finally:
        pbar.finish()
        pbar.close()

    build_elapsed = max(0.0, time.monotonic() - build_t0)
    if int(sparse_n) != int(n_samples):
        raise RuntimeError(
            f"Sparse GRM sample count mismatch: sparse={int(sparse_n)}, expected={int(n_samples)}"
        )
    log_success(
        logger,
        f"Sparse GRM (NNZ: {int(sparse_nnz)}) ...Finished "
        f"[{format_elapsed(build_elapsed)}]",
        force_color=True,
    )
    return str(sparse_path), int(sparse_n), int(sparse_nnz)


def build_sparse_grm_dense_npy(
    dense_grm_path: str,
    out_prefix: str,
    n_samples: int,
    kinship_cutoff: float,
    logger,
) -> tuple[str, int, int]:
    if _spgrm_dense_npy_to_jxgrm is None:
        raise RuntimeError(
            "Sparse GRM dense-NPY kernel is unavailable. Rebuild JanusX extension to export "
            "`spgrm_dense_npy_to_jxgrm`."
        )
    logger.info(
        "Sparse GRM route selected dense-grm-npy: "
        f"n={n_samples}, threshold-only extraction from precomputed dense GRM."
    )
    pbar = ProgressAdapter(
        total=1,
        desc="Sparse GRM (dense-grm-npy)",
        emit_done=False,
        force_animate=True,
    )
    build_t0 = time.monotonic()
    process = psutil.Process()
    last_done = 0
    last_total = 1

    def _progress_cb(done: int, total: int) -> None:
        nonlocal last_done, last_total
        d = max(0, int(done))
        t = max(1, int(total))
        if t != last_total:
            pbar.set_total(t)
            last_total = t
        d = min(d, t)
        if d > last_done:
            pbar.update(d - last_done)
            last_done = d
        mem = process.memory_info().rss / 1024**3
        pbar.set_postfix(memory=f"{mem:.2f} GB")

    try:
        sparse_path, sparse_n, sparse_nnz = _spgrm_dense_npy_to_jxgrm(
            str(dense_grm_path),
            out_prefix=str(out_prefix),
            threshold=float(kinship_cutoff),
            abs_threshold=False,
            progress_callback=_progress_cb,
            progress_every=1,
        )
    finally:
        pbar.finish()
        pbar.close()

    build_elapsed = max(0.0, time.monotonic() - build_t0)
    if int(sparse_n) != int(n_samples):
        raise RuntimeError(
            f"Sparse GRM sample count mismatch: sparse={int(sparse_n)}, expected={int(n_samples)}"
        )
    log_success(
        logger,
        f"Sparse GRM (NNZ: {int(sparse_nnz)}) ...Finished "
        f"[{format_elapsed(build_elapsed)}]",
        force_color=True,
    )
    return str(sparse_path), int(sparse_n), int(sparse_nnz)


def main(log: bool = True):
    t_start = time.time()

    parser = CliArgumentParser(
        prog="jx grm",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx grm -vcf geno.vcf.gz -o outdir -prefix demo",
            "jx grm -hmp geno.hmp.gz -o outdir -prefix demo",
            "jx grm -bfile geno_prefix -m 1 --npy",
        ]),
    )

    # ------------------------------------------------------------------
    # Required arguments
    # ------------------------------------------------------------------
    required_group = parser.add_argument_group("Required Arguments")
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument(
        "-vcf", "--vcf", type=str,
        help="Input genotype file in VCF format (.vcf or .vcf.gz).",
    )
    geno_group.add_argument(
        "-hmp", "--hmp", type=str,
        help="Input genotype file in HMP format (.hmp or .hmp.gz).",
    )
    geno_group.add_argument(
        "-bfile", "--bfile", type=str,
        help="Input genotype in PLINK binary format "
             "(prefix for .bed, .bim, .fam).",
    )
    geno_group.add_argument(
        "-file", "--file", type=str,
        help=(
            "Input genotype numeric matrix (.txt/.tsv/.csv/.npy) or prefix. "
            "Requires sibling prefix.id. Optional site metadata: prefix.site or prefix.bim."
        ),
    )
    geno_group.add_argument(
        "-k", "--dense-grm", type=str, dest="dense_grm",
        help=(
            "Input precomputed dense GRM in `.npy` format. "
            "Requires sibling `<grm>.id` and must be used with `-sparse` to emit `.spgrm`."
        ),
    )

    # ------------------------------------------------------------------
    # Optional arguments
    # ------------------------------------------------------------------
    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-t", "--thread", type=int, default=detect_effective_threads(),
        help="Number of CPU threads (default: %(default)s).",
    )
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory for results (default: current directory).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Prefix of output files (default: inferred from input file name).",
    )
    optional_group.add_argument(
        "-m", "--method", type=int, default=1,
        help=(
            "GRM calculation method: 1=centered (default), "
            "2=standardized/weighted (default: %(default)s)."
        ),
    )
    optional_group.add_argument(
        "-sparse", "--sparse", nargs="?", const=0.05, default=None, type=float,
        help=(
            "Build sparse GRM in CSC `.spgrm` format and keep only off-diagonal "
            "kinship entries >= cutoff (default cutoff when flag is present: %(const)s)."
        ),
    )
    optional_group.add_argument(
        "-maf", "--maf", type=float, default=0.02,
        help="Exclude variants with minor allele frequency lower than a threshold "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-geno", "--geno", type=float, default=0.05,
        help="Exclude variants with missing call frequencies greater than a threshold "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-memory", "--memory", type=float, default=DEFAULT_BED_MEMORY_MB,
        help=(
            "Target decode working-set size in MB for BED memmap/packed kernels "
            "(default: %(default)s)."
        ),
    )
    optional_group.add_argument(
        "--stage-timing", action="store_true", default=False,
        help=(
            "Print stage timing breakdown (decode/GEMM/other) from the selected "
            "Rust BED GRM backend."
        ),
    )
    optional_group.add_argument(
        "-npy", "--npy", action="store_true", default=False,
        help="Save GRM as binary NPY instead of plain text (default: %(default)s).",
    )
    optional_group.add_argument(
        "-fast", "--fast", action="store_true", default=False,
        help=(
            "Force packed BED backend (full in-memory load). "
            "Faster for f64 GRM; for f32 the default streaming backend "
            "with pipeline overlap is usually faster."
        ),
    )

    args = parser.parse_args()
    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    thread_capped = False
    if int(args.thread) <= 0:
        args.thread = int(detected_threads)
    if int(args.thread) > int(detected_threads):
        thread_capped = True
        args.thread = int(detected_threads)

    # ------------------------------------------------------------------
    # Determine genotype file and output prefix
    # ------------------------------------------------------------------
    if getattr(args, "dense_grm", None):
        gfile = str(args.dense_grm)
        auto_prefix = _dense_grm_auto_prefix(gfile)
    else:
        gfile, auto_prefix = _determine_genotype_source(
            vcf=getattr(args, "vcf", None),
            hmp=getattr(args, "hmp", None),
            file=getattr(args, "file", None),
            bfile=getattr(args, "bfile", None),
            prefix=None,
        )
    args.prefix = auto_prefix if args.prefix is None else args.prefix

    args.out = os.path.normpath(args.out if args.out is not None else ".")
    outprefix = os.path.join(args.out, args.prefix)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    os.makedirs(args.out, 0o755, exist_ok=True)
    configure_genotype_cache_from_out(args.out)
    log_path = f"{outprefix}.grm.log"
    logger = setup_logging(log_path)
    if thread_capped:
        logger.warning(
            f"Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(args.thread)}."
        )
    apply_blas_thread_env(int(args.thread))
    rust_blas_set_ok = set_rust_blas_threads(int(args.thread))
    rust_blas_backend = str(detect_rust_blas_backend()).strip().lower() or "unknown"
    rust_blas_threads = get_rust_blas_threads()
    rust_blas_threads_text = (
        "NA" if rust_blas_threads is None else str(int(rust_blas_threads))
    )
    logger.info(
        "Rust SGEMM backend: %s; requested_threads=%s; rust_blas_threads=%s; direct_set=%s.",
        rust_blas_backend,
        int(args.thread),
        rust_blas_threads_text,
        "ok" if rust_blas_set_ok else "no",
    )
    if require_openblas_by_default() and rust_blas_backend not in {"openblas"}:
        logger.warning(
            "Rust SGEMM backend is '%s', not openblas. Dense GRM build may underutilize threads.",
            rust_blas_backend,
        )
    # maybe_warn_non_openblas(
    #     logger=logger,
    #     strict=require_openblas_by_default(),
    # )

    if log:
        spgrm_timing_env = str(os.environ.get("JANUSX_SPGRM_TIMING", "")).strip().lower()
        spgrm_timing_enabled = bool(
            spgrm_timing_env and spgrm_timing_env not in {"0", "false", "no", "off"}
        )
        bed_backend_policy = "auto: memmap primary, packed fallback"
        general_rows = (
            [
                ("Dense GRM file", gfile),
                ("Sparse GRM cutoff", "disabled" if args.sparse is None else args.sparse),
                (
                    "Threads",
                    format_requested_thread_usage(
                        requested_threads=int(requested_threads),
                        using_threads=int(args.thread),
                        detected_threads=int(detected_threads),
                    ),
                ),
                ("Save as NPY", args.npy),
            ]
            if getattr(args, "dense_grm", None)
            else [
                ("Genotype file", gfile),
                ("GRM method", "Centered" if args.method == 1 else "Standardized/weighted"),
                ("Sparse GRM cutoff", "disabled" if args.sparse is None else args.sparse),
                ("MAF threshold", args.maf),
                ("Missing rate", args.geno),
                ("Memory MB", args.memory),
                ("Stage timing", bool(args.stage_timing or spgrm_timing_enabled)),
                (
                    "Threads",
                    format_requested_thread_usage(
                        requested_threads=int(requested_threads),
                        using_threads=int(args.thread),
                        detected_threads=int(detected_threads),
                    ),
                ),
                ("BED backend", bed_backend_policy),
                ("Save as NPY", args.npy),
            ]
        )
        emit_cli_configuration(
            logger,
            app_title="JanusX - GRM",
            config_title="GRM CONFIG",
            host=socket.gethostname(),
            sections=[
                (
                    "General",
                    general_rows,
                )
            ],
            footer_rows=[("Output prefix", outprefix)],
            line_max_chars=60,
        )

    checks: list[bool] = []
    if getattr(args, "dense_grm", None):
        checks.append(ensure_file_exists(logger, gfile, "Dense GRM file"))
    elif args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    if getattr(args, "dense_grm", None):
        if args.sparse is None:
            raise RuntimeError("Dense GRM input requires `-sparse <cutoff>` to emit `.spgrm`.")
        sparse_cutoff = float(args.sparse)
        if not np.isfinite(sparse_cutoff) or sparse_cutoff < 0.0:
            raise RuntimeError(
                f"Sparse GRM cutoff must be finite and >= 0, got {args.sparse}"
            )
        if args.npy:
            logger.warning("`--npy` is ignored for sparse GRM output; writing `.spgrm` CSC.")
        dense_id_path = resolve_grm_id_path(gfile)
        if dense_id_path is None:
            raise RuntimeError(
                "Dense GRM input requires sibling sample IDs in `<grm>.id`."
            )
        sample_ids = np.asarray(read_id_file(dense_id_path), dtype=str)
        if int(sample_ids.shape[0]) <= 0:
            raise RuntimeError("Dense GRM ID file is empty.")
        dense_arr = np.load(gfile, mmap_mode="r")
        if np.asarray(dense_arr).ndim != 2 or int(dense_arr.shape[0]) != int(dense_arr.shape[1]):
            raise RuntimeError(
                f"Dense GRM must be a square `.npy` matrix, got shape={np.asarray(dense_arr).shape}"
            )
        n_samples = int(dense_arr.shape[0])
        if n_samples != int(sample_ids.shape[0]):
            raise RuntimeError(
                f"Dense GRM ID count mismatch: matrix n={n_samples}, id={int(sample_ids.shape[0])}"
            )
        method_tag = _infer_dense_grm_method_tag(gfile, args.method)
        sparse_prefix = f"{outprefix}.{method_tag}"
        grm_path, sparse_n, sparse_nnz = build_sparse_grm_dense_npy(
            dense_grm_path=gfile,
            out_prefix=sparse_prefix,
            n_samples=n_samples,
            kinship_cutoff=sparse_cutoff,
            logger=logger,
        )
        _write_sparse_grm_meta(
            grm_path,
            cutoff=sparse_cutoff,
            source="dense_grm_npy",
            method=None,
            maf_threshold=None,
            max_missing_rate=None,
            het_threshold=None,
            snps_only=False,
            dense_grm_path=gfile,
        )
        id_path = f"{grm_path}.id"
        np.savetxt(id_path, sample_ids, fmt="%s")
        log_success(
            logger,
            f"Saved sparse GRM in CSC format (NNZ={int(sparse_nnz)}):\n"
            f"  {format_path_for_display(id_path)}\n"
            f"  {format_path_for_display(grm_path)}\n"
            f"  {format_path_for_display(f'{grm_path}.meta.json')}",
        )
        lt = time.localtime()
        endinfo = (
            f"\nFinished GRM calculation. Total wall time: "
            f"{round(time.time() - t_start, 2)} seconds\n"
            f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
            f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
        )
        log_success(logger, endinfo)
        return

    # ------------------------------------------------------------------
    # Resolve Rust GRM input (PLINK BED or cache-converted BED)
    # ------------------------------------------------------------------
    grm_input = str(gfile)
    if (_grm_stream_bed_f32 is not None) or (_grm_packed_bed_f32 is not None):
        try:
            grm_input = _resolve_rust_grm_input(
                str(gfile),
                from_vcf=bool(args.vcf),
                from_hmp=bool(args.hmp),
                from_file=bool(args.file),
            )
            if str(grm_input) != str(gfile):
                logger.info(
                    f"GRM backend input switched to cache BED prefix: "
                    f"{format_path_for_display(str(grm_input))}"
                )
        except Exception as ex:
            raise RuntimeError(
                "Unable to route GRM build to Rust BED backend. "
                f"source={gfile}; reason={ex}"
            ) from ex

    # ------------------------------------------------------------------
    # Inspect genotype and build GRM
    # ------------------------------------------------------------------
    genotype_src = format_path_for_display(str(gfile))
    with CliStatus(
        f"Loading genotype from {genotype_src}...",
        enabled=True,
        use_process=True,
    ) as task:
        try:
            sample_ids, n_snps = inspect_genotype_file(grm_input)
        except Exception:
            task.fail(f"Loading genotype from {genotype_src} ...Failed")
            raise
        sample_ids = np.array(sample_ids, dtype=str)
        n_samples = len(sample_ids)
        task.complete(
            f"Loading genotype from {genotype_src} (n={n_samples}, nSNP={n_snps})"
        )

    # Defaults match GWAS; can be overridden via CLI.
    maf_threshold = args.maf
    max_missing_rate = args.geno
    memory_mb = _normalize_memory_mb(args.memory)
    stream_block_rows = _decode_block_rows_from_memory_mb(
        n_samples,
        n_snps,
        memory_mb,
        streaming=True,
    )
    packed_block_rows = _decode_block_rows_from_memory_mb(
        n_samples,
        n_snps,
        memory_mb,
        streaming=False,
    )
    mmap_window_mb = auto_mmap_window_mb(grm_input, n_samples, n_snps, memory_mb)

    if args.sparse is not None:
        sparse_cutoff = float(args.sparse)
        if not np.isfinite(sparse_cutoff) or sparse_cutoff < 0.0:
            raise RuntimeError(
                f"Sparse GRM cutoff must be finite and >= 0, got {args.sparse}"
            )
        if args.npy:
            logger.warning("`--npy` is ignored for sparse GRM output; writing `.spgrm` CSC.")
        logger.info(
            "GRM auto route selected backend: sparse-stream-bed "
            "(BED pre-scan + blockwise streaming sparse CSC writer)."
        )
        sparse_prefix = f"{outprefix}.{_grm_method_tag(args.method)}"
        grm_path, sparse_n, sparse_nnz = build_sparse_grm_packed_bed(
            genofile=grm_input,
            out_prefix=sparse_prefix,
            n_samples=n_samples,
            n_snps=n_snps,
            method=args.method,
            kinship_cutoff=sparse_cutoff,
            maf_threshold=maf_threshold,
            max_missing_rate=max_missing_rate,
            chunk_size=stream_block_rows,
            mmap_window_mb=mmap_window_mb,
            threads=int(args.thread),
            block_target_mb=memory_mb,
            stage_timing=bool(args.stage_timing),
            logger=logger,
        )
        _write_sparse_grm_meta(
            grm_path,
            cutoff=sparse_cutoff,
            source="bed",
            method=int(args.method),
            maf_threshold=maf_threshold,
            max_missing_rate=max_missing_rate,
            het_threshold=0.0,
            snps_only=False,
            dense_grm_path=None,
        )
        id_path = f"{grm_path}.id"
        np.savetxt(id_path, sample_ids, fmt="%s")
        log_success(
            logger,
            f"Saved sparse GRM in CSC format (NNZ={int(sparse_nnz)}):\n"
            f"  {format_path_for_display(id_path)}\n"
            f"  {format_path_for_display(grm_path)}\n"
            f"  {format_path_for_display(f'{grm_path}.meta.json')}",
        )
    else:
        selected_backend, backend_reason = _select_cli_grm_backend(fast=bool(args.fast))
        logger.info(
            f"GRM auto route selected backend: {selected_backend} ({backend_reason})."
        )
        method_tag = _grm_method_tag(args.method)
        direct_npy_path = (
            f"{outprefix}.{method_tag}.npy"
            if (args.npy and selected_backend == "memmap-bed")
            else None
        )
        grm = None
        grm_path = None
        dense_saved_direct = False

        if selected_backend == "memmap-bed":
            logger.info(f"GRM decode memory target: {float(memory_mb):.6g} MB.")
            if bool(args.stage_timing):
                logger.info("Memmap GRM stage timing is enabled (decode/GEMM/other).")
            try:
                if direct_npy_path is not None:
                    eff_m = build_grm_streaming_to_npy(
                        genofile=grm_input,
                        out_npy_path=direct_npy_path,
                        n_samples=n_samples,
                        n_snps=n_snps,
                        method=args.method,
                        maf_threshold=maf_threshold,
                        max_missing_rate=max_missing_rate,
                        block_rows=stream_block_rows,
                        mmap_window_mb=mmap_window_mb,
                        threads=int(args.thread),
                        memory_mb=memory_mb,
                        stage_timing=bool(args.stage_timing),
                        logger=logger,
                    )
                    grm_path = direct_npy_path
                    dense_saved_direct = True
                else:
                    grm, eff_m = build_grm_streaming(
                        genofile=grm_input,
                        n_samples=n_samples,
                        n_snps=n_snps,
                        method=args.method,
                        maf_threshold=maf_threshold,
                        max_missing_rate=max_missing_rate,
                        block_rows=stream_block_rows,
                        mmap_window_mb=mmap_window_mb,
                        threads=int(args.thread),
                        memory_mb=memory_mb,
                        stage_timing=bool(args.stage_timing),
                        logger=logger,
                    )
            except Exception as ex:
                logger.warning(
                    "Memmap GRM kernel failed; fallback to Packed backend. "
                    f"reason={ex}"
                )
                grm, eff_m = build_grm_packed_bed(
                    genofile=grm_input,
                    n_samples=n_samples,
                    n_snps=n_snps,
                    method=args.method,
                    maf_threshold=maf_threshold,
                    max_missing_rate=max_missing_rate,
                    block_rows=packed_block_rows,
                    threads=int(args.thread),
                    memory_mb=memory_mb,
                    stage_timing=bool(args.stage_timing),
                    logger=logger,
                )
        else:
            logger.info("Memmap GRM kernel unavailable; using Rust Packed BED backend.")
            grm, eff_m = build_grm_packed_bed(
                genofile=grm_input,
                n_samples=n_samples,
                n_snps=n_snps,
                method=args.method,
                maf_threshold=maf_threshold,
                max_missing_rate=max_missing_rate,
                block_rows=packed_block_rows,
                threads=int(args.thread),
                memory_mb=memory_mb,
                stage_timing=bool(args.stage_timing),
                logger=logger,
            )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.sparse is None:
        method_tag = _grm_method_tag(args.method)
        if args.npy:
            if grm_path is None:
                grm_path = f"{outprefix}.{method_tag}.npy"
            if not dense_saved_direct:
                if grm is None:
                    raise RuntimeError("Dense GRM is missing before NPY save.")
                save_grm_npy_blocked(
                    grm_path,
                    grm,
                    dtype=np.float32,
                )
            id_path = f"{grm_path}.id"
            np.savetxt(id_path, sample_ids, fmt="%s")
            log_success(
                logger,
                f"Saved GRM in NPY format:\n"
                f"  {format_path_for_display(id_path)}\n"
                f"  {format_path_for_display(grm_path)}",
            )
        else:
            grm_path = f"{outprefix}.{method_tag}.txt"
            if grm is None:
                raise RuntimeError("Dense GRM is missing before text save.")
            np.savetxt(grm_path, grm, fmt="%.6f")
            id_path = f"{grm_path}.id"
            np.savetxt(id_path, sample_ids, fmt="%s")
            log_success(
                logger,
                f"Saved GRM in text format:\n"
                f"  {format_path_for_display(id_path)}\n"
                f"  {format_path_for_display(grm_path)}",
            )

    # ------------------------------------------------------------------
    # Final logging
    # ------------------------------------------------------------------
    lt = time.localtime()
    endinfo = (
        f"\nFinished GRM calculation. Total wall time: "
        f"{round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    log_success(logger, endinfo)


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
