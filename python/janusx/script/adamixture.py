# -*- coding: utf-8 -*-
"""
JanusX: ADAMIXTURE ancestry inference (Rust-kernel backend)

Examples
--------
  jx adamixture -bfile data/geno -k 8 -o out -prefix cohort
  jx adamixture -vcf data/geno.vcf.gz -k 6 -threads 16
"""

from __future__ import annotations

import logging
import os
import socket
import time
from pathlib import Path

import numpy as np

from janusx.adamixture import ADAMixtureConfig, train_adamixture
from ._common.config_render import emit_cli_configuration
from ._common.genoio import determine_genotype_source_from_args, strip_default_prefix_suffix
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
)
from ._common.status import print_failure, print_success


def _resolve_input(args, logger) -> tuple[str, str, str]:
    gfile, auto_prefix = determine_genotype_source_from_args(args)

    checks = []
    if args.vcf:
        checks.append(ensure_file_exists(logger, gfile, "VCF file"))
        source_label = "VCF"
    elif args.hmp:
        checks.append(ensure_file_exists(logger, gfile, "HMP file"))
        source_label = "HMP"
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, label="FILE genotype input"))
        source_label = "FILE"
    elif args.bfile:
        bprefix = gfile
        if str(gfile).lower().endswith(".bed"):
            bprefix = str(Path(gfile).with_suffix(""))
        checks.append(ensure_plink_prefix_exists(logger, bprefix, label="PLINK prefix"))
        gfile = bprefix
        auto_prefix = os.path.basename(bprefix.rstrip("/\\"))
        source_label = "BFILE"
    else:
        raise ValueError("One genotype input is required: -vcf / -hmp / -file / -bfile.")

    if not ensure_all_true(checks):
        raise FileNotFoundError("Input validation failed.")

    prefix = args.prefix or strip_default_prefix_suffix(auto_prefix)
    return gfile.replace("\\", "/"), source_label, prefix


def main() -> None:
    parser = CliArgumentParser(
        prog="jx adamixture",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx adamixture -bfile data/geno -k 8 -o out -prefix cohort",
            "jx adamixture -vcf data/geno.vcf.gz -k 6 -threads 16",
        ]),
    )

    required = parser.add_argument_group("Required Arguments")
    geno = required.add_mutually_exclusive_group(required=True)
    geno.add_argument("-bfile", "--bfile", type=str, help="PLINK prefix (.bed/.bim/.fam).")
    geno.add_argument("-vcf", "--vcf", type=str, help="VCF/VCF.GZ genotype file.")
    geno.add_argument("-hmp", "--hmp", type=str, help="HMP/HMP.GZ genotype file.")
    geno.add_argument(
        "-file",
        "--file",
        type=str,
        help="Text/NumPy genotype matrix prefix (requires sidecar .id).",
    )
    required.add_argument("-k", "--k", type=int, required=True, help="Number of ancestry clusters.")

    optional = parser.add_argument_group("Optional Arguments")
    optional.add_argument(
        "-o",
        "--out",
        type=str,
        default=".",
        help="Output directory (default: current directory).",
    )
    optional.add_argument("-prefix", "--prefix", type=str, default=None, help="Output prefix.")
    optional.add_argument(
        "-threads",
        "--threads",
        type=int,
        default=1,
        help="CPU threads for BLAS/OpenMP backends (default: 1).",
    )
    optional.add_argument(
        "-chunksize",
        "--chunksize",
        type=int,
        default=50000,
        help="Number of SNPs per loading chunk (default: 50000).",
    )
    optional.add_argument(
        "-snps-only",
        "--snps-only",
        action="store_true",
        default=False,
        help="Input VCF/HMP: keep SNP variants only during loading.",
    )

    optional.add_argument("-seed", "--seed", type=int, default=42, help="Random seed (default: 42).")
    optional.add_argument("-lr", "--lr", type=float, default=0.005, help="Adam learning rate.")
    optional.add_argument("-min_lr", "--min_lr", type=float, default=1e-6, help="Minimum learning rate.")
    optional.add_argument("-lr_decay", "--lr_decay", type=float, default=0.5, help="Learning rate decay factor.")
    optional.add_argument("-beta1", "--beta1", type=float, default=0.80, help="Adam beta1.")
    optional.add_argument("-beta2", "--beta2", type=float, default=0.88, help="Adam beta2.")
    optional.add_argument("-reg_adam", "--reg_adam", type=float, default=1e-8, help="Adam epsilon.")
    optional.add_argument("-max_iter", "--max_iter", type=int, default=1500, help="Maximum ADAM-EM iterations.")
    optional.add_argument("-check", "--check", type=int, default=5, help="Log-likelihood check interval.")

    optional.add_argument("-max_als", "--max_als", type=int, default=1000, help="Maximum ALS iterations.")
    optional.add_argument("-tole_als", "--tole_als", type=float, default=1e-4, help="ALS convergence tolerance.")
    optional.add_argument("-reg_als", "--reg_als", type=float, default=1e-5, help="ALS regularization.")
    optional.add_argument("-power", "--power", type=int, default=5, help="RSVD power iterations.")
    optional.add_argument("-tole_svd", "--tole_svd", type=float, default=1e-1, help="RSVD convergence tolerance.")

    args = parser.parse_args()

    outdir = str(args.out).replace("\\", "/").rstrip("/") or "."
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    tmp_logger = logging.getLogger("janusx.adamixture.bootstrap")
    if not tmp_logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        tmp_logger.addHandler(h)
    tmp_logger.setLevel(logging.INFO)
    tmp_logger.propagate = False
    try:
        genotype_path, source_label, auto_prefix = _resolve_input(args, tmp_logger)
    except Exception as e:
        print_failure(str(e), force_color=True)
        raise

    prefix = args.prefix or auto_prefix
    log_path = os.path.join(outdir, f"{prefix}.adamixture.log")
    logger = setup_logging(log_path)
    logger.info("")

    q_out = os.path.join(outdir, f"{prefix}.{int(args.k)}.Q").replace("\\", "/")
    p_out = os.path.join(outdir, f"{prefix}.{int(args.k)}.P").replace("\\", "/")

    emit_cli_configuration(
        logger,
        app_title="JanusX - ADAMIXTURE",
        config_title="ADAMIXTURE CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "General",
                [
                    ("Genotype", genotype_path),
                    ("Input type", source_label),
                    ("K", int(args.k)),
                    ("Seed", int(args.seed)),
                    ("Threads", int(args.threads)),
                    ("Chunk size", int(args.chunksize)),
                    ("SNPs only", bool(args.snps_only)),
                ],
            ),
            (
                "Optimization",
                [
                    ("lr", float(args.lr)),
                    ("min_lr", float(args.min_lr)),
                    ("lr_decay", float(args.lr_decay)),
                    ("beta1", float(args.beta1)),
                    ("beta2", float(args.beta2)),
                    ("reg_adam", float(args.reg_adam)),
                    ("max_iter", int(args.max_iter)),
                    ("check", int(args.check)),
                    ("max_als", int(args.max_als)),
                    ("tole_als", float(args.tole_als)),
                    ("reg_als", float(args.reg_als)),
                    ("power", int(args.power)),
                    ("tole_svd", float(args.tole_svd)),
                ],
            ),
        ],
        footer_rows=[
            ("Output prefix", f"{outdir}/{prefix}"),
            ("Q output", q_out),
            ("P output", p_out),
            ("Log file", log_path),
        ],
    )

    cfg = ADAMixtureConfig(
        genotype_path=genotype_path,
        k=int(args.k),
        outdir=outdir,
        prefix=prefix,
        seed=int(args.seed),
        threads=max(1, int(args.threads)),
        chunk_size=max(1000, int(args.chunksize)),
        snps_only=bool(args.snps_only),
        lr=float(args.lr),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        reg_adam=float(args.reg_adam),
        lr_decay=float(args.lr_decay),
        min_lr=float(args.min_lr),
        max_iter=max(1, int(args.max_iter)),
        check=max(1, int(args.check)),
        max_als=max(1, int(args.max_als)),
        tole_als=float(args.tole_als),
        reg_als=float(args.reg_als),
        power=max(1, int(args.power)),
        tole_svd=float(args.tole_svd),
    )

    try:
        p_mat, q_mat, m, n = train_adamixture(cfg, logger)
        np.savetxt(q_out, np.asarray(q_mat, dtype=np.float64), fmt="%.8f", delimiter=" ")
        np.savetxt(p_out, np.asarray(p_mat, dtype=np.float64), fmt="%.8f", delimiter=" ")
    except Exception as e:
        logger.exception(f"ADAMIXTURE failed: {e}")
        print_failure("ADAMIXTURE ...Failed", force_color=True)
        raise

    logger.info(f"Data shape: SNPs={m}, samples={n}")
    logger.info(f"Q output: {q_out}")
    logger.info(f"P output: {p_out}")
    logger.info(f"Total elapsed: {time.time() - t0:.2f}s")
    print_success(f"ADAMIXTURE ...Finished [{time.time() - t0:.1f}s]", force_color=True)


if __name__ == "__main__":
    main()
