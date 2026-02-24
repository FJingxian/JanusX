# -*- coding: utf-8 -*-
"""
JanusX: Post-GARFIELD Pipeline (GWAS -> Decode -> postgwas)

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
import re
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


_EXPR_SITE_PATTERN = re.compile(r"(?P<flag>!)?(?P<chrom>\d+)_(?P<pos>\d+)")


def _is_local_pseudo_set(expr: object, max_span_bp: int) -> bool:
    """
    Return True if expression is a multi-site pseudoSNP where all sites:
      - are on the same chromosome, and
      - lie within max_span_bp (max(pos) - min(pos) <= max_span_bp).
    """
    if max_span_bp < 0 or not isinstance(expr, str):
        return False

    matches = _EXPR_SITE_PATTERN.findall(expr)
    if len(matches) < 2:
        return False

    chroms = {chrom for _, chrom, _ in matches}
    if len(chroms) != 1:
        return False

    positions = [int(pos) for _, _, pos in matches]
    return (max(positions) - min(positions)) <= max_span_bp


def _filter_pseudomap_by_set_distance(pseudodf: pd.DataFrame, max_span_bp: int) -> tuple[pd.DataFrame, int]:
    """
    Remove pseudoSNP rows composed of nearby loci on one chromosome.
    """
    if "expression" not in pseudodf.columns:
        raise ValueError("Pseudo mapping file must contain an 'expression' column.")

    local_mask = pseudodf["expression"].map(lambda x: _is_local_pseudo_set(x, max_span_bp))
    removed = int(local_mask.sum())
    return pseudodf.loc[~local_mask].copy(), removed


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
        help="Threads for GWAS/postgwas (-1 uses all cores; default: %(default)s).",
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
        help="P-value threshold for postgwas (default: 0.05 / nSNP).",
    )
    optional_group.add_argument(
        "-bimrange", "--bimrange", type=str, default=None,
        help="Plotting range passed to postgwas in Mb, format chr:start-end.",
    )
    optional_group.add_argument(
        "-format", "--format", type=str, default="png",
        help="Output figure format for postgwas (default: %(default)s).",
    )
    optional_group.add_argument(
        "-noplot", "--noplot", action="store_true", default=False,
        help="Disable Manhattan/QQ plotting in postgwas.",
    )
    optional_group.add_argument(
        "-hl", "--highlight", type=str, default=None,
        help="BED-like file for highlighting SNPs in postgwas.",
    )
    optional_group.add_argument(
        "-a", "--anno", type=str, default=None,
        help="Annotation file for postgwas (GFF/BED).",
    )
    optional_group.add_argument(
        "-ab", "--annobroaden", type=float, default=None,
        help="Annotation window around SNPs in Kb (postgwas).",
    )
    optional_group.add_argument(
        "-descItem", "--descItem", type=str, default="description",
        help="GFF attribute key used for description (postgwas).",
    )
    optional_group.add_argument(
        "-pallete", "--pallete", type=str, default=None,
        help=(
            "Manhattan color palette passed to postgwas. "
            "Supports cmap name or ';'-separated colors."
        ),
    )
    optional_group.add_argument(
        "--only-set",
        type=int,
        nargs="?",
        const=50_000,
        default=None,
        metavar="BP",
        help=(
            "Filter out pseudoSNPs composed of loci within a short same-chromosome span. "
            "If provided without value, uses 50000 bp."
        ),
    )

    args = parser.parse_args()
    if args.only_set is not None and args.only_set < 0:
        raise ValueError("--only-set must be >= 0.")

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
    logger.info(f"Only-set filter (bp): {args.only_set}")
    logger.info(f"Bimrange: {args.bimrange}")
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
    if args.only_set is not None:
        pseudodf, n_removed = _filter_pseudomap_by_set_distance(pseudodf, args.only_set)
        logger.info(
            f"Applied --only-set {args.only_set}: removed {n_removed} pseudoSNP(s), "
            f"kept {pseudodf.shape[0]}."
        )
        if pseudodf.shape[0] == 0:
            raise ValueError(
                f"All pseudoSNPs were removed by --only-set {args.only_set}. "
                "Increase the threshold or disable --only-set."
            )

    pseudo_key_cols = pseudodf.columns[[0, 1]].tolist()
    pseudo_idx = set(map(tuple, pseudodf[pseudo_key_cols].itertuples(index=False, name=None)))

    gwas_files = _find_gwas_results(outprefix, "lmm")
    if not gwas_files:
        raise FileNotFoundError(
            f"No GWAS result files found under {outprefix} for model lmm"
        )

    decoded_files: list[str] = []
    for gwas_file in gwas_files:
        df = pd.read_csv(gwas_file, sep="\t")
        gwas_key_cols = df.columns[[0, 1]].tolist()
        gwas_keys = list(df[gwas_key_cols].itertuples(index=False, name=None))
        keep_mask = pd.Series([key in pseudo_idx for key in gwas_keys], index=df.index)
        dropped_rows = int((~keep_mask).sum())
        if dropped_rows > 0:
            logger.info(
                f"{os.path.basename(gwas_file)}: dropped {dropped_rows} GWAS row(s) "
                "not present in filtered pseudo map."
            )
        df = df.loc[keep_mask].copy()
        if df.empty:
            logger.warning(
                f"{os.path.basename(gwas_file)}: no GWAS rows remain after pseudo filtering; skipped."
            )
            continue

        dfdecode = decode(df, pseudodf, args.pseudochrom)
        decode_path = gwas_file.replace(".lmm.tsv", ".decode.lmm.tsv")
        dfdecode.to_csv(decode_path, sep="\t", index=False)
        decoded_files.append(decode_path)
        logger.info(f"Decoded result saved to {decode_path}")

    # ------------------------- postgwas plotting -------------------------
    for decode_path in decoded_files:
        cmd = [
            "jx", "postgwas",
            "-file", decode_path,
            "-o", args.out,
            "-prefix", prefix,
            "-format", args.format,
            "-t", str(args.thread),
        ]
        if args.threshold is not None:
            cmd.extend(["-threshold", str(args.threshold)])
        if args.bimrange is not None:
            cmd.extend(["-bimrange", args.bimrange])
        if not args.noplot:
            cmd.extend(["--manh", "--qq"])
        if args.highlight:
            cmd.extend(["-hl", args.highlight])
        if args.anno:
            cmd.extend(["-a", args.anno])
        if args.annobroaden is not None:
            cmd.extend(["-ab", str(args.annobroaden)])
        if args.descItem:
            cmd.extend(["-descItem", args.descItem])
        if args.pallete is not None:
            cmd.extend(["-pallete", args.pallete])

        _run_subprocess(cmd, logger, "Running postgwas")

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
