# -*- coding: utf-8 -*-
"""
JanusX: Efficient Genotype Merger (gmerge)

This CLI is a thin wrapper around the high-level Python API:
    janusx.gmerge.merge(inputs, out, out_fmt=..., check_exists=..., return_dict=True)

Input:
  - PLINK bfile prefix (BED/BIM/FAM)
  - VCF / VCF.GZ

Output:
  - VCF  : out endswith .vcf/.vcf.gz OR --out-fmt vcf
  - PLINK: otherwise OR --out-fmt plink

The Rust backend enforces:
  - keep only biallelic SNPs (A/C/G/T)
  - unify alleles (swap/strand-complement) and global MAF reordering
  - union of sites, missing filled as ./.
"""

import os
import time
import json
import socket
import argparse

from ._common.log import setup_logging
from ._common.config_render import emit_cli_configuration
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_plink_prefix_exists,
)
from janusx.gfreader.gmerge import merge


def _is_vcf_out(out: str) -> bool:
    x = out.lower()
    return x.endswith(".vcf") or x.endswith(".vcf.gz")


def _infer_prefix(out: str, out_fmt: str) -> str:
    """
    Decide a prefix name for log/json outputs.
    - If writing VCF: strip .vcf/.vcf.gz
    - Else: basename(out)
    """
    fmt = (out_fmt or "auto").lower()
    if fmt == "auto":
        fmt = "vcf" if _is_vcf_out(out) else "plink"

    base = os.path.basename(out)
    if fmt == "vcf":
        if base.endswith(".vcf.gz"):
            return base[:-7]
        if base.endswith(".vcf"):
            return base[:-4]
    return base


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
    required_group.add_argument(
        "-i", "--inputs", nargs="+", required=True, type=str,
        help=(
            "Input genotype datasets (>=2).\n"
            "Each item is either:\n"
            "  - PLINK prefix (no extension), e.g. geno/QC\n"
            "  - VCF path (.vcf or .vcf.gz), e.g. geno/QC.vcf.gz"
        ),
    )
    required_group.add_argument(
        "-o", "--out", required=True, type=str,
        help=(
            "Output target.\n"
            "  - VCF: provide path ending with .vcf or .vcf.gz\n"
            "  - PLINK: provide prefix (no extension)"
        ),
    )

    # ------------------------------------------------------------------
    # Optional arguments
    # ------------------------------------------------------------------
    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "--out-fmt", type=str, default="auto", choices=["auto", "vcf", "plink"],
        help="Output format (default: %(default)s).",
    )
    optional_group.add_argument(
        "--no-check", action="store_true", default=False,
        help="Skip input existence checks (default: %(default)s).",
    )
    optional_group.add_argument(
        "--report-json", type=str, default=None,
        help=(
            "Write merge statistics to JSON file.\n"
            "Default: <outdir>/<prefix>.merge.json"
        ),
    )
    optional_group.add_argument(
        "--outdir", type=str, default=None,
        help=(
            "Directory to write log/json reports.\n"
            "Default: parent directory of --out (or current directory if none)."
        ),
    )

    args = parser.parse_args()

    inputs = [x.replace("\\", "/") for x in args.inputs]
    out = args.out.replace("\\", "/")
    out_fmt = args.out_fmt

    if len(inputs) < 2:
        raise ValueError("--inputs must contain at least 2 datasets")

    # ------------------------------------------------------------------
    # Decide report dir / prefix
    # ------------------------------------------------------------------
    if args.outdir is not None:
        report_dir = args.outdir.replace("\\", "/")
    else:
        report_dir = os.path.dirname(out) or "."

    os.makedirs(report_dir, 0o755, exist_ok=True)

    prefix = _infer_prefix(out, out_fmt)
    log_path = f"{report_dir}/{prefix}.merge.log".replace("\\", "/").replace("//", "/")
    logger = setup_logging(log_path)

    if log:
        input_rows = [(f"Input {i + 1}", x) for i, x in enumerate(inputs)]
        emit_cli_configuration(
            logger,
            app_title="JanusX - gmerge",
            config_title="GMERGE CONFIG",
            host=socket.gethostname(),
            sections=[
                ("Inputs", input_rows),
                (
                    "General",
                    [
                        ("Output", out),
                        ("Output format", out_fmt),
                        ("Check inputs exist", (not args.no_check)),
                        ("Report dir", report_dir),
                        ("Log file", log_path),
                    ],
                ),
            ],
            line_max_chars=60,
        )

    if args.no_check:
        logger.warning("--no-check is ignored in CLI mode; input existence is always validated.")

    checks: list[bool] = []
    for src in inputs:
        low = src.lower()
        if low.endswith(".vcf") or low.endswith(".vcf.gz"):
            checks.append(ensure_file_exists(logger, src, "Input VCF file"))
        else:
            checks.append(ensure_plink_prefix_exists(logger, src, "Input PLINK prefix"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Run merge (call your Python API)
    # ------------------------------------------------------------------
    t0 = time.time()
    stats, d = merge(
        inputs=inputs,
        out=out,
        out_fmt=out_fmt,
        check_exists=True,
        return_dict=True,
    )
    logger.info(f"Merge completed in {round(time.time() - t0, 3)} seconds")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    logger.info("*" * 60)
    logger.info("GMERGE SUMMARY")
    logger.info("*" * 60)
    logger.info(f"Total samples: {d.get('n_samples_total')}")
    logger.info(f"Sites written: {d.get('n_sites_written')}")
    logger.info(f"Union sites seen: {d.get('n_sites_union_seen')}")
    logger.info(f"Dropped multiallelic: {d.get('n_sites_dropped_multiallelic')}")
    logger.info(f"Dropped non-SNP: {d.get('n_sites_dropped_non_snp')}")
    logger.info("*" * 60 + "\n")

    # ------------------------------------------------------------------
    # Save JSON report
    # ------------------------------------------------------------------
    report_json = args.report_json
    if report_json is None:
        report_json = f"{report_dir}/{prefix}.merge.json"
    report_json = report_json.replace("\\", "/")

    with open(report_json, "w", encoding="utf-8") as fw:
        json.dump(d, fw, ensure_ascii=False, indent=2)
    logger.info(f"Saved merge report JSON:\n  {report_json}")

    # ------------------------------------------------------------------
    # Final logging
    # ------------------------------------------------------------------
    lt = time.localtime()
    endinfo = (
        f"\nFinished genotype merge. Total wall time: "
        f"{round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    logger.info(endinfo)


if __name__ == "__main__":
    main()
