# -*- coding: utf-8 -*-
"""
JanusX: Efficient Genetic Relationship Matrix (GRM) Calculator

Design overview
---------------
Input:
  - VCF   : genotype in VCF/VCF.GZ format
  - BFILE : genotype in PLINK binary format (.bed/.bim/.fam prefix)

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
import sys
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
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_plink_prefix_exists,
)
from ._common.status import get_rich_spinner_name, print_success

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    _HAS_RICH_PROGRESS = True
except Exception:
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    TimeRemainingColumn = None  # type: ignore[assignment]
    _HAS_RICH_PROGRESS = False

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore[assignment]
    _HAS_TQDM = False


class _ProgressAdapter:
    """
    Progress adapter for rich-first rendering with tqdm fallback.
    """
    def __init__(self, total: int, desc: str) -> None:
        self.total = int(max(0, total))
        self.desc = str(desc)
        self._backend = "none"
        self._progress = None
        self._task_id = None
        self._tqdm = None

        if _HAS_RICH_PROGRESS and sys.stdout.isatty():
            try:
                self._progress = Progress(
                    SpinnerColumn(
                        spinner_name=get_rich_spinner_name(),
                        style="cyan",
                    ),
                    TextColumn("[bold green]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    TextColumn("{task.fields[postfix]}"),
                    transient=False,
                )
                self._progress.start()
                self._task_id = self._progress.add_task(
                    self.desc,
                    total=self.total,
                    postfix="",
                )
                self._backend = "rich"
            except Exception:
                self._progress = None
                self._task_id = None

        if self._backend == "none" and _HAS_TQDM:
            self._tqdm = tqdm(total=self.total, desc=self.desc, ascii=True)
            self._backend = "tqdm"

    def update(self, n: int) -> None:
        step = int(max(0, n))
        if step == 0:
            return
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=step)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.update(step)

    def set_postfix(self, **kwargs: object) -> None:
        if len(kwargs) == 0:
            return
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            text = " ".join([f"{k}={v}" for k, v in kwargs.items()])
            self._progress.update(self._task_id, postfix=text)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.set_postfix(kwargs)

    def finish(self) -> None:
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=self.total)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.n = self._tqdm.total
            self._tqdm.refresh()

    def close(self) -> None:
        closed = False
        finished = False
        if self._backend == "rich" and self._progress is not None:
            if self._task_id is not None:
                try:
                    task = self._progress.tasks[self._task_id]
                    finished = bool(task.finished)
                except Exception:
                    finished = False
            self._progress.stop()
            self._progress = None
            self._task_id = None
            closed = True
        elif self._backend == "tqdm" and self._tqdm is not None:
            try:
                total = float(self._tqdm.total or 0)
                finished = float(self._tqdm.n or 0) >= total
            except Exception:
                finished = False
            self._tqdm.close()
            self._tqdm = None
            closed = True
        if closed and finished:
            print_success(f"{self.desc} ...Finished")


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
    logger.info(
        f"* Building GRM in streaming mode (method={method}, "
        f"MAF >= {maf_threshold}, missing rate <= {max_missing_rate})"
    )

    grm = np.zeros((n_samples, n_samples), dtype="float32")
    pbar = _ProgressAdapter(total=n_snps, desc="GRM (streaming)")
    process = psutil.Process()

    varsum = 0.0
    eff_m = 0
    for genosub, _sites in load_genotype_chunks(
        genofile,
        chunk_size,
        maf_threshold,
        max_missing_rate,
        mmap_window_mb=mmap_window_mb,
    ):
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

    if eff_m == 0:
        raise RuntimeError("No SNPs remained after filtering; GRM is empty.")

    # Symmetrize and scale
    if method == 1:
        if varsum == 0:
            raise RuntimeError("Variance sum is zero in method=1; check genotype input.")
        grm = (grm + grm.T) / (2 * varsum)
    else:  # method == 2
        grm = (grm + grm.T) / (2 * eff_m)

    logger.info(f"GRM construction finished. Effective SNPs: {eff_m}")
    logger.info(f"GRM shape: {grm.shape}")
    return grm, eff_m


def main(log: bool = True):
    t_start = time.time()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        "-bfile", "--bfile", type=str,
        help="Input genotype in PLINK binary format "
             "(prefix for .bed, .bim, .fam).",
    )

    # ------------------------------------------------------------------
    # Optional arguments
    # ------------------------------------------------------------------
    optional_group = parser.add_argument_group("Optional Arguments")
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
        "-mmap-limit", "--mmap-limit", action="store_true", default=False,
        help="Enable windowed mmap for BED inputs (auto: 2x chunk size).",
    )
    optional_group.add_argument(
        "-npy", "--npy", action="store_true", default=False,
        help="Save GRM as binary NPY instead of plain text (default: %(default)s).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Determine genotype file and output prefix
    # ------------------------------------------------------------------
    if args.vcf:
        gfile = args.vcf
        args.prefix = (
            os.path.basename(gfile).replace(".gz", "").replace(".vcf", "")
            if args.prefix is None else args.prefix
        )
    elif args.bfile:
        gfile = args.bfile
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    else:
        raise ValueError("One of --vcf or --bfile must be provided.")

    gfile = gfile.replace("\\", "/")
    args.out = args.out if args.out is not None else "."

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    os.makedirs(args.out, 0o755, exist_ok=True)
    log_path = f"{args.out}/{args.prefix}.grm.log".replace("\\", "/").replace("//", "/")
    logger = setup_logging(log_path)

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
                        ("Mmap limit", args.mmap_limit),
                        ("Save as NPY", args.npy),
                    ],
                )
            ],
            footer_rows=[("Output prefix", f"{args.out}/{args.prefix}")],
            line_max_chars=60,
        )

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
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

    t_loading = time.time()
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
    logger.info(
        f"GRM calculation completed in {round(time.time() - t_loading, 3)} seconds"
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.npy:
        grm_path = f"{args.out}/{args.prefix}.grm.npy"
        np.save(grm_path, grm)
        id_path = f"{grm_path}.id"
        np.savetxt(id_path, sample_ids, fmt="%s")
        logger.info(
            f"Saved GRM in NPY format:\n"
            f"  {id_path}\n"
            f"  {grm_path}"
        )
    else:
        grm_path = f"{args.out}/{args.prefix}.grm.txt"
        np.savetxt(grm_path, grm, fmt="%.6f")
        id_path = f"{grm_path}.id"
        np.savetxt(id_path, sample_ids, fmt="%s")
        logger.info(
            f"Saved GRM in text format:\n"
            f"  {id_path}\n"
            f"  {grm_path}"
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
    logger.info(endinfo)


if __name__ == "__main__":
    main()
