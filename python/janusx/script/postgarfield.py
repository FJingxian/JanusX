# -*- coding: utf-8 -*-
"""
JanusX: Post-GARFIELD Pipeline (GWAS -> Decode -> PostGWAS)

Examples
--------
  # Run LMM GWAS on pseudo genotype and plot/annotate decoded results
  -bfile test/example.garfield -p test/pheno.tsv -k test/example.grm.npy

  # Use a custom pseudo mapping file and output prefix
  -bfile test/example.garfield -p test/pheno.tsv -k test/example.grm.npy -o out -prefix demo \\
    --pseudo test/example.garfield.pseudo
"""

from __future__ import annotations

import os
import time
import socket
import argparse
import subprocess
from glob import glob
from typing import List

import pandas as pd

from janusx.garfield.decode import decode
from janusx.script._common.log import setup_logging
from janusx.script._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_plink_prefix_exists,
)


def _determine_genotype(args) -> tuple[str, str]:
    if args.vcf:
        gfile = args.vcf
        prefix = os.path.basename(gfile).replace(".gz", "").replace(".vcf", "")
    elif args.bfile:
        gfile = args.bfile
        prefix = os.path.basename(gfile)
    else:
        raise ValueError("One of --vcf or --bfile must be provided.")
    if args.prefix:
        prefix = args.prefix
    return gfile.replace("\\", "/"), prefix


def _find_gwas_results(outprefix: str, model: str) -> list[str]:
    model = model.lower()
    patterns = [
        f"{outprefix}.*.{model}.tsv",
        f"{outprefix}.{model}.tsv",
    ]
    files: list[str] = []
    for pat in patterns:
        files.extend(glob(pat))
    # Exclude already-decoded outputs
    files = [f for f in files if ".decode." not in os.path.basename(f)]
    return sorted(set(files))


def _run_subprocess(cmd: List[str], logger, desc: str) -> None:
    logger.info(f"* {desc}: {' '.join(cmd)}")
    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if res.returncode != 0:
        stderr = (res.stderr or "").strip()
        if stderr:
            logger.error(stderr)
        raise RuntimeError(f"{desc} failed with exit code {res.returncode}")


def main() -> None:
    t_start = time.time()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ------------------------- Required arguments -------------------------
    required_group = parser.add_argument_group("Required Arguments")
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument("-bfile", "--bfile", type=str, help="Pseudo genotype PLINK prefix.")
    geno_group.add_argument("-vcf", "--vcf", type=str, help="Pseudo genotype VCF/VCF.GZ file.")
    required_group.add_argument(
        "-p", "--pheno", type=str, required=True, help="Phenotype file."
    )
    required_group.add_argument(
        "-k", "--grm", type=str, required=True,
        help="Required GRM file path for LMM (no auto-build).",
    )

    # ------------------------- Optional arguments -------------------------
    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-q", "--qcov", type=str, default=None,
        help="Optional Q matrix file path (PC covariates).",
    )
    optional_group.add_argument(
        "-cov", "--cov", type=str, default=None,
        help="Optional covariate file (first column is sample ID).",
    )
    optional_group.add_argument(
        "-n", "--ncol", action="extend", nargs="*", default=None, type=int,
        help="Zero-based phenotype column indices to analyze.",
    )
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory for results (default: current directory).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Output prefix (default: inferred from genotype file).",
    )
    optional_group.add_argument(
        "-maf", "--maf", type=float, default=0.02,
        help="MAF threshold for GWAS (default: %(default)s).",
    )
    optional_group.add_argument(
        "-geno", "--geno", type=float, default=0.05,
        help="Missing rate threshold for GWAS (default: %(default)s).",
    )
    optional_group.add_argument(
        "-chunksize", "--chunksize", type=int, default=100_000,
        help="Chunk size for streaming GWAS (default: %(default)s).",
    )
    optional_group.add_argument(
        "-t", "--thread", type=int, default=-1,
        help="Threads for GWAS/postGWAS (-1 uses all cores; default: %(default)s).",
    )
    optional_group.add_argument(
        "--pseudo", type=str, default=None,
        help="Pseudo mapping file (default: {genofile}.pseudo).",
    )
    optional_group.add_argument(
        "--pseudochrom", type=str, default="pseudo",
        help="Pseudo chromosome label in GWAS results (default: %(default)s).",
    )
    optional_group.add_argument(
        "-threshold", "--threshold", type=float, default=None,
        help="P-value threshold for postGWAS (default: 0.05 / nSNP).",
    )
    optional_group.add_argument(
        "-format", "--format", type=str, default="png",
        help="Output figure format for postGWAS (default: %(default)s).",
    )
    optional_group.add_argument(
        "-noplot", "--noplot", action="store_true", default=False,
        help="Disable Manhattan/QQ plotting in postGWAS.",
    )
    optional_group.add_argument(
        "-hl", "--highlight", type=str, default=None,
        help="BED-like file for highlighting SNPs in postGWAS.",
    )
    optional_group.add_argument(
        "-a", "--anno", type=str, default=None,
        help="Annotation file for postGWAS (GFF/BED).",
    )
    optional_group.add_argument(
        "-ab", "--annobroaden", type=float, default=None,
        help="Annotation window around SNPs in Kb (postGWAS).",
    )
    optional_group.add_argument(
        "-descItem", "--descItem", type=str, default="description",
        help="GFF attribute key used for description (postGWAS).",
    )
    optional_group.add_argument(
        "-color", "--color", type=int, default=0,
        help="Color set index for postGWAS (0-6; -1 uses auto).",
    )

    args = parser.parse_args()

    gfile, prefix = _determine_genotype(args)
    outprefix = f"{args.out}/{prefix}".replace("//", "/")
    os.makedirs(args.out, mode=0o755, exist_ok=True)

    log_path = f"{args.out}/{prefix}.postGARFIELD.log".replace("//", "/")
    logger = setup_logging(log_path)

    logger.info("JanusX - Post-GARFIELD pipeline")
    logger.info(f"Host: {socket.gethostname()}\n")

    logger.info("*" * 60)
    logger.info("POST-GARFIELD CONFIGURATION")
    logger.info("*" * 60)
    logger.info(f"Genotype file: {gfile}")
    logger.info(f"Phenotype file: {args.pheno}")
    logger.info("Model: lmm")
    logger.info(f"GRM: {args.grm}")
    logger.info(f"Q/PC: {args.qcov}")
    logger.info(f"Covariate: {args.cov}")
    logger.info(f"MAF: {args.maf}")
    logger.info(f"Missing rate: {args.geno}")
    logger.info(f"Chunk size: {args.chunksize}")
    logger.info(f"Threads: {args.thread}")
    logger.info(f"Pseudo map: {args.pseudo or f'{gfile}.pseudo'}")
    logger.info(f"Output prefix: {outprefix}")
    logger.info("*" * 60 + "\n")

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    checks.append(ensure_file_exists(logger, args.pheno, "Phenotype file"))
    checks.append(ensure_file_exists(logger, args.grm, "GRM file"))
    if args.qcov:
        checks.append(ensure_file_exists(logger, args.qcov, "Q matrix file"))
    if args.cov:
        checks.append(ensure_file_exists(logger, args.cov, "Covariate file"))
    pseudo_path = args.pseudo or f"{gfile}.pseudo"
    checks.append(ensure_file_exists(logger, pseudo_path, "Pseudo mapping file"))
    if args.highlight:
        checks.append(ensure_file_exists(logger, args.highlight, "Highlight file"))
    if args.anno:
        checks.append(ensure_file_exists(logger, args.anno, "Annotation file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    # ------------------------- Run GWAS -------------------------
    cmd = [
        "jx", "gwas",
        "-lmm",
        "-p", args.pheno,
        "-k", args.grm,
        "-maf", str(args.maf),
        "-geno", str(args.geno),
        "-chunksize", str(args.chunksize),
        "-t", str(args.thread),
        "-o", args.out,
        "-prefix", prefix,
    ]
    if args.qcov:
        cmd.extend(["-q", args.qcov])
    if args.bfile:
        cmd.extend(["-bfile", args.bfile])
    else:
        cmd.extend(["-vcf", args.vcf])
    if args.cov:
        cmd.extend(["-cov", args.cov])
    if args.ncol:
        for n in args.ncol:
            cmd.extend(["-n", str(n)])

    _run_subprocess(cmd, logger, "Running GWAS")

    # ------------------------- Decode pseudo SNPs -------------------------
    pseudodf = pd.read_csv(pseudo_path, sep="\t")
    gwas_files = _find_gwas_results(outprefix, "lmm")
    if not gwas_files:
        raise FileNotFoundError(
            f"No GWAS result files found under {outprefix} for model lmm"
        )

    decoded_files: list[str] = []
    for gwas_file in gwas_files:
        df = pd.read_csv(gwas_file, sep="\t")
        dfdecode = decode(df, pseudodf, args.pseudochrom)
        decode_path = gwas_file.replace(".lmm.tsv", ".decode.lmm.tsv")
        dfdecode.to_csv(decode_path, sep="\t", index=False)
        decoded_files.append(decode_path)
        logger.info(f"Decoded result saved to {decode_path}")

    # ------------------------- PostGWAS plotting -------------------------
    for decode_path in decoded_files:
        cmd = [
            "jx", "postGWAS",
            "-f", decode_path,
            "-o", args.out,
            "-prefix", prefix,
            "-format", args.format,
            "-t", str(args.thread),
        ]
        if args.threshold is not None:
            cmd.extend(["-threshold", str(args.threshold)])
        if args.noplot:
            cmd.append("-noplot")
        if args.highlight:
            cmd.extend(["-hl", args.highlight])
        if args.anno:
            cmd.extend(["-a", args.anno])
        if args.annobroaden is not None:
            cmd.extend(["-ab", str(args.annobroaden)])
        if args.descItem:
            cmd.extend(["-descItem", args.descItem])
        if args.color is not None:
            cmd.extend(["-color", str(args.color)])

        _run_subprocess(cmd, logger, "Running postGWAS")

    lt = time.localtime()
    endinfo = (
        f"\nFinished post-GARFIELD pipeline. Total wall time: "
        f"{round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    logger.info(endinfo)


if __name__ == "__main__":
    main()
