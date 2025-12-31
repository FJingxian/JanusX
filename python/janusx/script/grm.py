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
  - {prefix}.grm.id   : sample IDs
  - {prefix}.grm.txt  : GRM as plain text (if --npz is not used)
  - {prefix}.grm.npz  : compressed GRM (if --npz is used; stored as arr_0)
"""

import os
import time
import socket
import argparse

import numpy as np
from tqdm import tqdm
import psutil
from janusx.gfreader import load_genotype_chunks, inspect_genotype_file
from ._common.log import setup_logging


def build_grm_streaming(
    genofile: str,
    n_samples: int,
    n_snps: int,
    method: int,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
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
    pbar = tqdm(total=n_snps, desc="GRM (streaming)", ascii=True)
    process = psutil.Process()

    varsum = 0.0
    eff_m = 0

    for genosub, _sites in load_genotype_chunks(
        genofile,
        chunk_size,
        maf_threshold,
        max_missing_rate,
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
    pbar.n = pbar.total
    pbar.refresh()
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
        "-npz", "--npz", action="store_true", default=False,
        help="Save GRM as compressed NPZ instead of plain text (default: %(default)s).",
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

    logger.info("JanusX: Efficient Genetic Relationship Matrix (GRM) Calculator")
    logger.info(f"Host: {socket.gethostname()}\n")

    if log:
        logger.info("*" * 60)
        logger.info("GRM CONFIGURATION")
        logger.info("*" * 60)
        logger.info(f"Genotype file: {gfile}")
        logger.info(
            f"GRM method: {'Centered' if args.method == 1 else 'Standardized/weighted'}"
        )
        logger.info(f"Save as NPZ: {args.npz}")
        logger.info(f"Output prefix: {args.out}/{args.prefix}")
        logger.info("*" * 60 + "\n")

    # ------------------------------------------------------------------
    # Inspect genotype and build GRM in streaming mode
    # ------------------------------------------------------------------
    sample_ids, n_snps = inspect_genotype_file(gfile)
    sample_ids = np.array(sample_ids, dtype=str)
    n_samples = len(sample_ids)
    logger.info(f"Genotype meta: {n_samples} samples, {n_snps} SNPs.")

    # To keep consistent with GWAS module, we fix MAF/missing filters here.
    maf_threshold = 0.01
    max_missing_rate = 0.05
    chunk_size = 100_000

    t_loading = time.time()
    grm, eff_m = build_grm_streaming(
        genofile=gfile,
        n_samples=n_samples,
        n_snps=n_snps,
        method=args.method,
        maf_threshold=maf_threshold,
        max_missing_rate=max_missing_rate,
        chunk_size=chunk_size,
        logger=logger,
    )
    logger.info(
        f"GRM calculation completed in {round(time.time() - t_loading, 3)} seconds"
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    id_path = f"{args.out}/{args.prefix}.grm.id"
    if args.npz:
        grm_path = f"{args.out}/{args.prefix}.grm.npz"
        np.savetxt(id_path, sample_ids, fmt="%s")
        # Store matrix as arr_0, consistent with PCA module loading
        np.savez_compressed(f"{args.out}/{args.prefix}.grm", grm)
        logger.info(
            f"Saved GRM in NPZ format:\n"
            f"  {id_path}\n"
            f"  {grm_path}"
        )
    else:
        grm_path = f"{args.out}/{args.prefix}.grm.txt"
        np.savetxt(id_path, sample_ids, fmt="%s")
        np.savetxt(grm_path, grm, fmt="%.6f")
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
