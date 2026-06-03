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
from typing import Union

import numpy as np
import psutil
from janusx.gfreader import (
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
from ._common.status import format_elapsed, log_success
from ._common.genocache import configure_genotype_cache_from_out
from ._common.genoio import determine_genotype_source as _determine_genotype_source
from ._common.threads import (
    apply_blas_thread_env,
    detect_effective_threads,
    maybe_warn_non_openblas,
    require_openblas_by_default,
)
from ._common.grmstable import (
    save_grm_npy_blocked,
)

try:
    from janusx.janusx import (
        grm_packed_bed_f32 as _grm_packed_bed_f32,
        grm_stream_bed_f32 as _grm_stream_bed_f32,
    )
except Exception:
    _grm_packed_bed_f32 = None
    _grm_stream_bed_f32 = None

try:
    from janusx.janusx import (
        spgrm_bed_to_jxgrm as _spgrm_bed_to_jxgrm,
        spgrm_dense_f32_to_jxgrm as _spgrm_dense_f32_to_jxgrm,
    )
except Exception:
    _spgrm_bed_to_jxgrm = None
    _spgrm_dense_f32_to_jxgrm = None


def _is_plink_prefix_path(path_or_prefix: str) -> bool:
    p = str(path_or_prefix).strip()
    if p == "":
        return False
    low = p.lower()
    if low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam"):
        p = p[:-4]
    return all(os.path.isfile(f"{p}.{ext}") for ext in ("bed", "bim", "fam"))


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
    chunk_size: int,
    mmap_window_mb: Union[int , None],
    threads: int,
    block_target_mb: Union[float, None],
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
    chunk_size : int
        Number of SNPs per chunk.
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
    mem_tick_span = max(1, 10 * int(chunk_size))
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

    prev_block_target = os.environ.get("JX_GRM_BLOCK_TARGET_MB")
    prev_stage_timing = os.environ.get("JX_GRM_STREAM_STAGE_TIMING")
    block_target_set = False
    stage_timing_set = False
    if block_target_mb is not None:
        try:
            bt = float(block_target_mb)
        except Exception:
            logger.warning(
                f"Invalid --block-target-mb={block_target_mb}; fallback to runtime default."
            )
        else:
            if np.isfinite(bt) and bt > 0:
                os.environ["JX_GRM_BLOCK_TARGET_MB"] = f"{bt:.6g}"
                block_target_set = True
            else:
                logger.warning(
                    f"Non-positive --block-target-mb={block_target_mb}; fallback to runtime default."
                )
    if bool(stage_timing):
        os.environ["JX_GRM_STREAM_STAGE_TIMING"] = "1"
        stage_timing_set = True

    try:
        try:
            grm_raw, eff_m, stream_n = _grm_stream_bed_f32(
                str(genofile),
                method=int(method),
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                block_cols=max(1, int(chunk_size)),
                threads=max(1, int(threads)),
                progress_callback=_progress_cb,
                progress_every=max(1, int(chunk_size)),
                mmap_window_mb=(int(mmap_window_mb) if mmap_window_mb is not None else None),
            )
        finally:
            if block_target_set:
                if prev_block_target is None:
                    os.environ.pop("JX_GRM_BLOCK_TARGET_MB", None)
                else:
                    os.environ["JX_GRM_BLOCK_TARGET_MB"] = prev_block_target
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


def build_grm_packed_bed(
    genofile: str,
    n_samples: int,
    n_snps: int,
    method: int,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    threads: int,
    block_target_mb: Union[float, None],
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
    mem_tick_span = max(1, 10 * int(chunk_size))
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

    prev_block_target = os.environ.get("JX_GRM_BLOCK_TARGET_MB")
    prev_stage_timing = os.environ.get("JX_GRM_PACKED_STAGE_TIMING")
    block_target_set = False
    stage_timing_set = False
    if block_target_mb is not None:
        try:
            bt = float(block_target_mb)
        except Exception:
            logger.warning(
                f"Invalid --block-target-mb={block_target_mb}; fallback to runtime default."
            )
        else:
            if np.isfinite(bt) and bt > 0:
                os.environ["JX_GRM_BLOCK_TARGET_MB"] = f"{bt:.6g}"
                block_target_set = True
            else:
                logger.warning(
                    f"Non-positive --block-target-mb={block_target_mb}; fallback to runtime default."
                )
    if bool(stage_timing):
        os.environ["JX_GRM_PACKED_STAGE_TIMING"] = "1"
        stage_timing_set = True

    try:
        try:
            grm_raw, eff_m, packed_n = _grm_packed_bed_f32(
                str(genofile),
                method=int(method),
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                block_cols=max(1, int(chunk_size)),
                threads=max(1, int(threads)),
                progress_callback=_progress_cb,
                progress_every=max(1, int(chunk_size)),
            )
            if int(packed_n) != int(n_samples):
                raise RuntimeError(
                    f"Packed sample count mismatch: packed={int(packed_n)}, expected={int(n_samples)}"
                )
            grm = np.ascontiguousarray(np.asarray(grm_raw, dtype=np.float32))
            eff_m = int(eff_m)
        finally:
            if block_target_set:
                if prev_block_target is None:
                    os.environ.pop("JX_GRM_BLOCK_TARGET_MB", None)
                else:
                    os.environ["JX_GRM_BLOCK_TARGET_MB"] = prev_block_target
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
            "Build sparse GRM in CSC `.jxgrm` format and keep only off-diagonal "
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
        "-chunksize", "--chunksize", type=int, default=100_000,
        help="Number of SNPs per chunk for memmap GRM "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "--block-target-mb", type=float, default=None,
        help=(
            "Target temporary working-set size (MB) for the selected BED GRM backend "
            "auto block tuning. Larger values may improve compute utilization at the "
            "cost of memory."
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
        "-mmap-limit", "--mmap-limit", action="store_true", default=False,
        help="Enable windowed mmap for BED inputs (default backend; auto: 2x chunk size).",
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
        emit_cli_configuration(
            logger,
            app_title="JanusX - GRM",
            config_title="GRM CONFIG",
            host=socket.gethostname(),
            sections=[
                (
                    "General",
                    [
                        ("Genotype file", gfile),
                        ("GRM method", "Centered" if args.method == 1 else "Standardized/weighted"),
                        ("Sparse GRM cutoff", "disabled" if args.sparse is None else args.sparse),
                        ("MAF threshold", args.maf),
                        ("Missing rate", args.geno),
                        ("Chunk size", args.chunksize),
                        ("Block target MB", "backend-default" if args.block_target_mb is None else args.block_target_mb),
                        ("Stage timing", bool(args.stage_timing or spgrm_timing_enabled)),
                        ("Threads", f"{args.thread} ({detected_threads} available)"),
                        ("BED backend", bed_backend_policy),
                        ("Mmap limit flag", args.mmap_limit),
                        ("Save as NPY", args.npy),
                    ],
                )
            ],
            footer_rows=[("Output prefix", outprefix)],
            line_max_chars=60,
        )

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

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
    sample_ids, n_snps = inspect_genotype_file(grm_input)
    sample_ids = np.array(sample_ids, dtype=str)
    n_samples = len(sample_ids)
    logger.info(f"Genotype meta: {n_samples} samples, {n_snps} SNPs.")

    # Defaults match GWAS; can be overridden via CLI.
    maf_threshold = args.maf
    max_missing_rate = args.geno
    chunk_size = int(args.chunksize)
    mmap_limit_effective = bool(args.mmap_limit or _is_plink_prefix_path(str(grm_input)))
    if mmap_limit_effective and (not bool(args.mmap_limit)):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("GRM backend policy: memmap (default).")
    mmap_window_mb = (
        auto_mmap_window_mb(grm_input, n_samples, n_snps, chunk_size)
        if mmap_limit_effective else None
    )

    if args.sparse is not None:
        sparse_cutoff = float(args.sparse)
        if not np.isfinite(sparse_cutoff) or sparse_cutoff < 0.0:
            raise RuntimeError(
                f"Sparse GRM cutoff must be finite and >= 0, got {args.sparse}"
            )
        if args.npy:
            logger.warning("`--npy` is ignored for sparse GRM output; writing `.jxgrm` CSC.")
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
            chunk_size=chunk_size,
            mmap_window_mb=mmap_window_mb,
            threads=int(args.thread),
            block_target_mb=args.block_target_mb,
            stage_timing=bool(args.stage_timing),
            logger=logger,
        )
        id_path = f"{grm_path}.id"
        np.savetxt(id_path, sample_ids, fmt="%s")
        log_success(
            logger,
            f"Saved sparse GRM in CSC format (NNZ={int(sparse_nnz)}):\n"
            f"  {format_path_for_display(id_path)}\n"
            f"  {format_path_for_display(grm_path)}",
        )
    else:
        selected_backend, backend_reason = _select_cli_grm_backend(fast=bool(args.fast))
        logger.info(
            f"GRM auto route selected backend: {selected_backend} ({backend_reason})."
        )

        if selected_backend == "memmap-bed":
            if args.block_target_mb is not None:
                logger.info(
                    f"Memmap GRM block target override: {float(args.block_target_mb):.6g} MB."
                )
            if bool(args.stage_timing):
                logger.info("Memmap GRM stage timing is enabled (decode/GEMM/other).")
            try:
                grm, eff_m = build_grm_streaming(
                    genofile=grm_input,
                    n_samples=n_samples,
                    n_snps=n_snps,
                    method=args.method,
                    maf_threshold=maf_threshold,
                    max_missing_rate=max_missing_rate,
                    chunk_size=chunk_size,
                    mmap_window_mb=mmap_window_mb,
                    threads=int(args.thread),
                    block_target_mb=args.block_target_mb,
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
                    chunk_size=chunk_size,
                    threads=int(args.thread),
                    block_target_mb=args.block_target_mb,
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
                chunk_size=chunk_size,
                threads=int(args.thread),
                block_target_mb=args.block_target_mb,
                stage_timing=bool(args.stage_timing),
                logger=logger,
            )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.sparse is None:
        method_tag = _grm_method_tag(args.method)
        if args.npy:
            grm_path = f"{outprefix}.{method_tag}.npy"
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
