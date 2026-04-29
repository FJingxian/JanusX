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
  - Genotypes are streamed via rust2py.gfreader.load_genotype_chunks.
  - SNPs are filtered by MAF and missing rate inside the Rust reader.
  - GRM is accumulated chunk-by-chunk:
      * method = 1 : centered GRM
      * method = 2 : standardized/weighted GRM
  - Memory usage is low and independent of the total SNP count.

Output:
  - {prefix}.grm.txt     : GRM as plain text (if --npy is not used)
  - {prefix}.grm.txt.id  : sample IDs for text GRM
  - {prefix}.grm.npy     : binary GRM (if --npy is used)
  - {prefix}.grm.npy.id  : sample IDs for NPY GRM
"""

import os
import time
import socket
import argparse
from typing import Union

import numpy as np
import psutil
from janusx.gfreader import (
    load_genotype_chunks,
    inspect_genotype_file,
    auto_mmap_window_mb,
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
from ._common.prefetch import prefetch_iter
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

try:
    from janusx.janusx import (
        grm_packed_bed_f32 as _grm_packed_bed_f32,
    )
except Exception:
    _grm_packed_bed_f32 = None


def build_grm_streaming(
    genofile: str,
    n_samples: int,
    n_snps: int,
    method: int,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    mmap_window_mb: Union[int , None],
    logger,
) -> tuple[np.ndarray, int]:
    """
    Build the GRM in streaming mode using rust2py.gfreader.load_genotype_chunks.

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
    grm = np.zeros((n_samples, n_samples), dtype="float32")
    pbar = ProgressAdapter(
        total=n_snps,
        desc="GRM (streaming)",
        emit_done=False,
        force_animate=True,
    )
    stream_t0 = time.monotonic()
    process = psutil.Process()

    varsum = 0.0
    eff_m = 0
    prefetch_depth = 2
    chunk_iter = load_genotype_chunks(
        genofile,
        chunk_size,
        maf_threshold,
        max_missing_rate,
        mmap_window_mb=mmap_window_mb,
    )
    for genosub, _sites in prefetch_iter(chunk_iter, in_flight=prefetch_depth):
        # genosub: (m_chunk, n_samples)
        maf = genosub.mean(axis=1, dtype="float32", keepdims=True) / 2  # (m_chunk,1)
        genosub = genosub - 2 * maf  # center by 2p

        if method == 1:
            # Standard centered GRM
            grm += genosub.T @ genosub
            varsum += float(np.sum(2 * maf * (1 - maf)))
        elif method == 2:
            # Weighted / standardized GRM
            w = 1.0 / (2 * maf * (1 - maf))      # (m_chunk,1)
            grm += (genosub.T * w.ravel()) @ genosub
        else:
            raise ValueError(f"Unsupported GRM method: {method}")

        m_chunk = genosub.shape[0]
        eff_m += m_chunk
        pbar.update(m_chunk)

        # Show memory usage periodically
        if eff_m % (10 * chunk_size) == 0:
            mem = process.memory_info().rss / 1024**3
            pbar.set_postfix(memory=f"{mem:.2f} GB")

    # Force progress bar to 100%, even if some SNPs were filtered in Rust
    pbar.finish()
    pbar.close()
    stream_elapsed = max(0.0, time.monotonic() - stream_t0)

    if eff_m == 0:
        raise RuntimeError("No SNPs remained after filtering; GRM is empty.")

    # Symmetrize and scale
    if method == 1:
        if varsum == 0:
            raise RuntimeError("Variance sum is zero in method=1; check genotype input.")
        grm = (grm + grm.T) / (2 * varsum)
    else:  # method == 2
        grm = (grm + grm.T) / (2 * eff_m)

    log_success(
        logger,
        f"GRM (Effective SNPs: {eff_m}) ...Finished [{format_elapsed(stream_elapsed)}]",
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

    prev_block_target = os.environ.get("JX_GRM_PACKED_BLOCK_TARGET_MB")
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
                os.environ["JX_GRM_PACKED_BLOCK_TARGET_MB"] = f"{bt:.6g}"
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
                    os.environ.pop("JX_GRM_PACKED_BLOCK_TARGET_MB", None)
                else:
                    os.environ["JX_GRM_PACKED_BLOCK_TARGET_MB"] = prev_block_target
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
        f"GRM (Effective SNPs: {eff_m}, packed-bed kernel) ...Finished "
        f"[{format_elapsed(stream_elapsed)}]",
        force_color=True,
    )
    return grm, eff_m


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
        help="Number of SNPs per chunk for streaming GRM "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "--block-target-mb", type=float, default=None,
        help=(
            "Target temporary working-set size (MB) for packed-BED GRM auto block tuning. "
            "Larger values may improve compute utilization at the cost of memory."
        ),
    )
    optional_group.add_argument(
        "--stage-timing", action="store_true", default=False,
        help=(
            "Print packed-BED GRM stage timing breakdown (decode/GEMM/other) from Rust kernel."
        ),
    )
    optional_group.add_argument(
        "-mmap-limit", "--mmap-limit", action="store_true", default=False,
        help="Enable windowed mmap for BED inputs (auto: 2x chunk size).",
    )
    optional_group.add_argument(
        "-npy", "--npy", action="store_true", default=False,
        help="Save GRM as binary NPY instead of plain text (default: %(default)s).",
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
    maybe_warn_non_openblas(
        logger=logger,
        strict=require_openblas_by_default(),
    )

    if log:
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
                        ("MAF threshold", args.maf),
                        ("Missing rate", args.geno),
                        ("Chunk size", args.chunksize),
                        ("Block target MB", "default(64)" if args.block_target_mb is None else args.block_target_mb),
                        ("Stage timing", args.stage_timing),
                        ("Threads", f"{args.thread} ({detected_threads} available)"),
                        ("Mmap limit", args.mmap_limit),
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
    # Inspect genotype and build GRM in streaming mode
    # ------------------------------------------------------------------
    sample_ids, n_snps = inspect_genotype_file(gfile)
    sample_ids = np.array(sample_ids, dtype=str)
    n_samples = len(sample_ids)
    logger.info(f"Genotype meta: {n_samples} samples, {n_snps} SNPs.")

    # Defaults match GWAS; can be overridden via CLI.
    maf_threshold = args.maf
    max_missing_rate = args.geno
    chunk_size = int(args.chunksize)
    mmap_window_mb = (
        auto_mmap_window_mb(gfile, n_samples, n_snps, chunk_size)
        if args.mmap_limit else None
    )

    use_packed_kernel = bool(
        bool(args.bfile)
        and (_grm_packed_bed_f32 is not None)
    )
    if use_packed_kernel:
        logger.info(
            "GRM backend: packed BED kernel "
            "(Rust grm_packed_bed_f32 + unified prepare_bed_2bit_packed filtering)."
        )
        if args.block_target_mb is not None:
            logger.info(f"Packed GRM block target override: {float(args.block_target_mb):.6g} MB.")
        if bool(args.stage_timing):
            logger.info("Packed GRM stage timing is enabled (decode/GEMM/other).")
        try:
            grm, eff_m = build_grm_packed_bed(
                genofile=gfile,
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
        except Exception as ex:
            logger.warning(
                "Packed GRM kernel failed; fallback to streaming backend. "
                f"reason={ex}"
            )
            grm, eff_m = build_grm_streaming(
                genofile=gfile,
                n_samples=n_samples,
                n_snps=n_snps,
                method=args.method,
                maf_threshold=maf_threshold,
                max_missing_rate=max_missing_rate,
                chunk_size=chunk_size,
                mmap_window_mb=mmap_window_mb,
                logger=logger,
            )
    else:
        if bool(args.bfile):
            logger.info("Packed GRM kernel unavailable; fallback to streaming dense-chunk backend.")
        grm, eff_m = build_grm_streaming(
            genofile=gfile,
            n_samples=n_samples,
            n_snps=n_snps,
            method=args.method,
            maf_threshold=maf_threshold,
            max_missing_rate=max_missing_rate,
            chunk_size=chunk_size,
            mmap_window_mb=mmap_window_mb,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.npy:
        grm_path = f"{outprefix}.grm.npy"
        np.save(grm_path, grm)
        id_path = f"{grm_path}.id"
        np.savetxt(id_path, sample_ids, fmt="%s")
        log_success(
            logger,
            f"Saved GRM in NPY format:\n"
            f"  {format_path_for_display(id_path)}\n"
            f"  {format_path_for_display(grm_path)}",
        )
    else:
        grm_path = f"{outprefix}.grm.txt"
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
