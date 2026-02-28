"""
JanusX Simulation CLI

Mode
- Simulate phenotype from existing genotype: use --vcf or --bfile

Outputs
- Phenotypes (always):
  <out>/<prefix>.pheno        : 3 columns (FID, IID, PHENO)
  <out>/<prefix>.pheno.txt    : 2 columns (IID, PHENO)
  <out>/<prefix>.pheno.NA.txt : 2 columns (IID, PHENO) with ~10% NA

- Causal sites (optional):
  <out>/<prefix>.causal.sites.tsv
"""

import os
import time
import socket
import argparse
from typing import Literal, Optional, List, Tuple

import numpy as np
from janusx.garfield.garfield2 import ldprune
from janusx.gfreader import load_genotype_chunks
from janusx.gfreader import inspect_genotype_file
from ._common.log import setup_logging
from ._common.config_render import emit_cli_configuration
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_plink_prefix_exists,
)


# -----------------------------
# Phenotype simulator from genotype stream
# -----------------------------
def simulate_phenotype_from_genofile(
    gfile: str,
    mode: Literal["single", "garfield"] = "single",
    chunk_size: int = 100_000,
    seed: int = 1,
    maf: float = 0.02,
    missing_rate: float = 0.05,
    pve: float = 0.5,
    ve: float = 1.0,
    windows: int = 50_000,
) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
    """
    Simulate phenotype y (n,1) from an existing genotype file.

    Model
    - load_genotype_chunks(...) yields Mchunk with shape (m, n) = (SNPs, samples)
    - Polygenic background:
        g = sum_j beta_j * Z_j
      where Z_j is standardized per SNP and Var(beta_j) ≈ vg/nsnp
    - Fixed parameters:
        mean = 10.0
      pve and ve are configurable via CLI (defaults: 0.5 and 1.0).
    """
    rng = np.random.default_rng(seed)

    # Fixed parameters (not exposed via CLI)
    mean = 10.0

    sampleid, nsnp = inspect_genotype_file(gfile)
    n = len(sampleid)

    # Residuals: e ~ N(0, ve) plus constant mean
    y = rng.normal(0.0, np.sqrt(ve), size=(n, 1)) + mean

    vg = pve * ve / (1.0 - pve)

    siteslist: List[Tuple[str, int, int]] = []

    # Polygenic background
    for Mchunk, sites in load_genotype_chunks(
        gfile,
        chunk_size=chunk_size,
        maf=maf,
        missing_rate=missing_rate,
    ):
        # Mchunk: (m, n)
        Mchunk = Mchunk.astype(np.float32, copy=False)

        # Standardize each SNP across samples
        mu = Mchunk.mean(axis=1, keepdims=True)
        sd = Mchunk.std(axis=1, keepdims=True, ddof=0) + 1e-6
        Z = (Mchunk - mu) / sd

        m = Z.shape[0]
        # beta_j ~ N(0, vg/nsnp)
        beta = rng.normal(0.0, np.sqrt(vg / nsnp), size=(m, 1)).astype(np.float32)
        g = (beta.T @ Z).T  # (n,1)
        y += g

        if mode == "single":
            siteslist.extend([(str(i.chrom), int(i.pos), int(i.pos)) for i in sites])
        else:
            siteslist.extend([(str(i.chrom), int(i.pos) - windows, int(i.pos) + windows) for i in sites])

    # Pick a target SNP or window
    arr = np.array(siteslist, dtype=object)
    siteslistrc = tuple(arr[rng.integers(len(arr))])
    outsites: List[Tuple[str, int, int]] = []

    if mode == "single":
        # Add the target SNP effect (simple: effect size = 1)
        for Mchunk, sites in load_genotype_chunks(
            gfile,
            chunk_size=chunk_size,
            maf=maf,
            missing_rate=missing_rate,
            bim_range=siteslistrc,
        ):
            Mchunk = Mchunk.astype(np.float32, copy=False)  # shape (m, n) or (1, n)
            y += Mchunk.T  # -> (n,1)
            outsites.extend([(str(i.chrom), int(i.pos), int(i.pos)) for i in sites])

    else:  # Garfield-like: build a burden pseudo-SNP within the window
        npseudo = int(rng.integers(2, 5))
        maxiter = 100

        pseudo = np.zeros((n, 1), dtype=np.int32)
        for _ in range(maxiter):
            outsites.clear()
            pseudo[:] = 0

            for Mchunk, sites in load_genotype_chunks(
                gfile,
                chunk_size=chunk_size,
                maf=maf,
                missing_rate=missing_rate,
                bim_range=siteslistrc,
            ):
                # 0/1/2 -> 0/1
                Mchunk,sites = ldprune(Mchunk,sites,0.75)
                G01 = (Mchunk / 2).astype(np.int32)  # (m,n)
                m = G01.shape[0]
                if m == 0:
                    continue
                idxrc = rng.choice(np.arange(m), size=min(npseudo, m), replace=False)
                burden = (np.sum(G01[idxrc, :], axis=0) >= 1).astype(np.int32).reshape(-1, 1)
                pseudo = burden
                sel_sites = np.array([(str(i.chrom), int(i.pos), int(i.pos)) for i in sites], dtype=object)
                outsites.extend(sel_sites[idxrc].tolist())

            af = pseudo.mean()
            if 0.02 <= af <= 0.98:
                break

        # Add a fixed effect size
        y += 2.0 * pseudo.astype(np.float32)

    return y.astype(np.float32), outsites


# -----------------------------
# Phenotype writer
# -----------------------------
def write_phenotypes(
    outprefix: str,
    sample_ids: np.ndarray,
    y: np.ndarray,
    seed: int = 1,
):
    """
    Write phenotype files:
    - <outprefix>.pheno        (FID IID PHENO)
    - <outprefix>.pheno.txt    (IID PHENO)
    - <outprefix>.pheno.NA.txt (IID PHENO, ~10% NA)
    """
    rng = np.random.default_rng(seed)

    sample_ids = np.asarray(sample_ids, dtype=str)
    y = y.reshape(-1)

    # 3 columns: PLINK-style
    pheno3 = np.empty((len(sample_ids), 3), dtype=object)
    pheno3[:, 0] = sample_ids
    pheno3[:, 1] = sample_ids
    pheno3[:, 2] = y
    np.savetxt(
        f"{outprefix}.pheno",
        pheno3,
        delimiter="\t",
        fmt=["%s", "%s", "%.6f"],
    )

    # 2 columns: simple format
    pheno2 = np.column_stack([sample_ids, y.astype(object)])
    np.savetxt(
        f"{outprefix}.pheno.txt",
        pheno2,
        delimiter="\t",
        fmt=["%s", "%.6f"],
        header="IID\tPHENO",
        comments="",
    )

    # Fixed 10% NA
    na_rate = 0.1
    pheno2_na = pheno2.astype(object, copy=True)
    n = len(sample_ids)
    k = int(round(n * na_rate))
    if k > 0:
        idx = rng.choice(n, size=k, replace=False)
        pheno2_na[idx, 1] = "NA"
    np.savetxt(
        f"{outprefix}.pheno.NA.txt",
        pheno2_na,
        delimiter="\t",
        fmt=["%s", "%s"],
        header="IID\tPHENO",
        comments="",
    )


def write_sites(outprefix: str, sites: List[Tuple[str, int, int]]):
    if not sites:
        return
    path = f"{outprefix}.causal.sites.tsv"
    arr = np.array(sites, dtype=object)
    np.savetxt(path, arr, delimiter="\t", fmt=["%s", "%d", "%d"])


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jx sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        description="JanusX simulation: phenotype from existing genotype",
    )

    required_group = parser.add_argument_group("Required arguments")
    optional_group = parser.add_argument_group("Optional arguments")

    # Genotype input (required)
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument(
        "-vcf", "--vcf", type=str,
        help="Input genotype file in VCF format (.vcf or .vcf.gz).",
    )
    geno_group.add_argument(
        "-bfile", "--bfile", type=str,
        help="Input genotype in PLINK binary format (prefix for .bed, .bim, .fam).",
    )

    # Common arguments
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory for results (default: %(default)s).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Prefix for output files (default: %(default)s).",
    )
    optional_group.add_argument(
        "-chunksize", "--chunksize", type=int, default=100_000,
        help="Number of SNPs per chunk for streaming (default: %(default)s).",
    )

    # Filters when reading genotype
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
        "-ve", "--ve", type=float, default=1.0,
        help="Environmental variance for phenotype simulation "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-pve", "--pve", type=float, default=0.5,
        help="Proportion of variance explained by genetic component "
             "(default: %(default)s).",
    )

    # Causal signal mode
    optional_group.add_argument(
        "-sm", "--sim-mode", type=str, default="single",
        choices=["single", "garfield"],
        help="Causal signal mode (default: %(default)s).",
    )
    optional_group.add_argument(
        "-windows", "--windows", type=int, default=50_000,
        help="Window size for garfield mode "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-write-sites", "--write-sites", action="store_true",
        help="Write causal sites file "
             "(default: %(default)s).",
    )
    optional_group.add_argument("--seed", type=int, default=None, help="Random seed.")

    return parser


def determine_genotype_source(args: argparse.Namespace) -> tuple[str, str]:
    """
    Resolve genotype input and output prefix from CLI arguments.
    """
    if args.vcf:
        gfile = args.vcf
        prefix = os.path.basename(gfile).replace(".gz", "").replace(".vcf", "")
    elif args.bfile:
        gfile = args.bfile
        prefix = os.path.basename(gfile)
    else:
        raise ValueError("No genotype input specified. Use -vcf or -bfile.")

    if args.prefix is not None:
        prefix = args.prefix

    gfile = gfile.replace("\\", "/")
    return gfile, prefix


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # -------------------------
    # Existing genotype file
    # -------------------------
    gfile, prefix = determine_genotype_source(args)
    outprefix = f"{args.out}/{prefix}".replace("\\", "/").replace("//", "/")
    os.makedirs(args.out, exist_ok=True, mode=0o755)

    log_path = f"{outprefix}.sim.log"
    logger = setup_logging(log_path)

    emit_cli_configuration(
        logger,
        app_title="JanusX - Simulation",
        config_title="SIMULATION CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "General",
                [
                    ("Genotype file", gfile),
                    ("Chunk size", args.chunksize),
                    ("MAF threshold", args.maf),
                    ("Missing threshold", args.geno),
                    ("Sim mode", args.sim_mode),
                    ("Windows", args.windows),
                    ("PVE", args.pve),
                    ("VE", args.ve),
                    ("Write sites", args.write_sites),
                    ("Seed", args.seed),
                ],
            )
        ],
        footer_rows=[("Output prefix", outprefix)],
        line_max_chars=60,
    )

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    if not (0.0 < args.pve < 1.0):
        logger.error("--pve must be in (0, 1).")
        raise SystemExit(1)

    t_start = time.time()
    logger.info(f"Simulating phenotype from genotype: {gfile}")
    y, outsites = simulate_phenotype_from_genofile(
        gfile=gfile,
        mode=args.sim_mode,
        chunk_size=args.chunksize,
        seed=args.seed,
        maf=args.maf,
        missing_rate=args.geno,
        pve=args.pve,
        ve=args.ve,
        windows=args.windows,
    )

    sampleid, _nsnp = inspect_genotype_file(gfile)
    sampleid = np.asarray(sampleid, dtype=str)

    write_phenotypes(outprefix, sampleid, y, seed=args.seed)
    if args.write_sites:
        write_sites(outprefix, outsites)

    logger.info(f"Finished in {time.time() - t_start:.2f} s")
    logger.info("Done. Outputs:")
    logger.info(f"  {outprefix}.pheno")
    logger.info(f"  {outprefix}.pheno.txt")
    logger.info(f"  {outprefix}.pheno.NA.txt")
    if args.write_sites:
        logger.info(f"  {outprefix}.causal.sites.tsv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
