import os
import time
import socket
import argparse
import numpy as np
import pandas as pd
from joblib import cpu_count
from janusx.garfield.garfield2 import main as garfield_main, window as garfield_window
from janusx.gfreader import SiteInfo, inspect_genotype_file
from janusx.gfreader.gfreader import save_genotype_streaming
from janusx.script._common.readanno import readanno
from janusx.script.gwas import load_phenotype
from janusx.script._common.log import setup_logging

def write_xcombine_results(
    outprefix: str,
    sample_ids: list[str],
    results: list,
    *,
    chrom: str = "pseudo",
):
    """
    Save xcombine results to PLINK/VCF and write {outprefix}.garfield mapping.
    """
    map_path = f"{outprefix}.pseudo"
    n_samples = len(sample_ids)

    def _iter_chunks():
        pos = 1
        with open(map_path, "w", encoding="utf-8") as f:
            f.write("chrom\tpos\texpression\tscore\n")
            for item in results:
                if item is None:
                    continue
                resdict:dict = item
                xcombine = np.asarray(resdict.get("xcombine", []), dtype=np.int8).ravel()
                if xcombine.size != n_samples:
                    raise ValueError("xcombine length does not match sample_ids")
                expr = resdict.get("expression", "")
                score = resdict.get("score", "")
                f.write(f"{chrom}\t{pos}\t{expr}\t{score}\n")
                geno = xcombine.reshape(1, -1)
                sites = [SiteInfo(str(chrom), int(pos), "A", "T")]
                yield geno, sites
                pos += 1

    save_genotype_streaming(outprefix, sample_ids, _iter_chunks())


def _read_geneset_lines(path: str) -> list[list[str]]:
    """
    Read gene sets from a text file.
    Each line can contain multiple genes separated by whitespace.
    """
    genesets: list[list[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            genes = [g for g in line.strip().split() if g]
            if genes:
                genesets.append(genes)
    return genesets


def main() -> None:
    t_start = time.time()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    required_group = parser.add_argument_group("Required Arguments")
    required_group.add_argument(
        "-bfile", "--bfile", type=str, required=True,
        help="Input genotype in PLINK binary format (prefix for .bed/.bim/.fam).",
    )
    required_group.add_argument(
        "-p", "--pheno", type=str, required=True,
        help="Phenotype file (sample IDs in the first column).",
    )

    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-g", "--genefile", type=str, default=None,
        help="Optional gene-set file (one line per gene set).",
    )
    optional_group.add_argument(
        "-gff", "--gff3", type=str, default=None,
        help="Optional GFF3 file for gene coordinates.",
    )
    optional_group.add_argument(
        "-step", "--step", type=int, default=25_000,
        help="Step size for sliding windows (default: %(default)s).",
    )
    optional_group.add_argument(
        "-ext", "--extension", type=int, default=50_000,
        help="Window extension length (default: %(default)s).",
    )
    optional_group.add_argument(
        "-nsnp", "--nsnp", type=int, default=5,
        help="Top SNPs selected by model (default: %(default)s).",
    )
    optional_group.add_argument(
        "-nestimators", "--nestimators", type=int, default=50,
        help="Number of trees/estimators (default: %(default)s).",
    )
    optional_group.add_argument(
        "-t", "--threads", type=int, default=-1,
        help="Number of threads (-1 uses all available cores; default: %(default)s).",
    )
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory (default: current directory).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Output prefix (default: inferred from bfile).",
    )

    args = parser.parse_args()

    bfile = args.bfile.replace("\\", "/")
    prefix = args.prefix or os.path.basename(bfile)
    outprefix = f"{args.out}/{prefix}".replace("//", "/")
    os.makedirs(args.out, mode=0o755, exist_ok=True)

    log_path = f"{args.out}/{prefix}.garfield.log".replace("//", "/")
    logger = setup_logging(log_path)

    logger.info("JanusX - GARFIELD")
    logger.info(f"Host: {socket.gethostname()}\n")
    logger.info("*" * 60)
    logger.info("GARFIELD CONFIGURATION")
    logger.info("*" * 60)
    logger.info(f"Bfile:         {bfile}")
    logger.info(f"Phenotype:     {args.pheno}")
    logger.info(f"Gene file:     {args.genefile}")
    logger.info(f"GFF3:          {args.gff3}")
    logger.info(f"Step:          {args.step}")
    logger.info(f"Extension:     {args.extension}")
    logger.info(f"Top SNPs:      {args.nsnp}")
    logger.info(f"Estimators:    {args.nestimators}")
    threads = cpu_count() if args.threads == -1 else int(args.threads)
    logger.info(f"Threads:       {threads}")
    logger.info(f"Output prefix: {outprefix}")
    logger.info("*" * 60 + "\n")

    # Load genotype meta
    sample_ids, _n_snps = inspect_genotype_file(bfile)
    sample_ids = np.array(sample_ids, dtype=str)

    # Load phenotype and align to genotype order
    pheno = load_phenotype(args.pheno, None, logger, id_col=0)
    pheno_ids = pheno.index.astype(str).to_numpy()
    common = [sid for sid in sample_ids if sid in set(pheno_ids)]
    if len(common) == 0:
        raise ValueError("No overlapping samples between genotype and phenotype.")
    pheno = pheno.loc[common]
    y = pheno.iloc[:, 0].values.reshape(-1, 1)

    # Gene/gff3 mode
    if args.genefile and args.gff3 and os.path.isfile(args.genefile) and os.path.isfile(args.gff3):
        genesets = _read_geneset_lines(args.genefile)
        dfgff3 = readanno(args.gff3, "ID").iloc[:, :4].set_index(3)
        dupgenemask = dfgff3.index.duplicated()
        if any(dupgenemask):
            logger.warning(f"Duplicated genes in GFF3: {np.unique(dfgff3.index[dupgenemask])}")
        dfgff3 = dfgff3.loc[~dupgenemask]

        bimranges = []
        for geneset in genesets:
            ranges = []
            for gene in geneset:
                if gene not in dfgff3.index:
                    logger.warning(f"Gene not found in GFF3: {gene}")
                    ranges = []
                    break
                chrom = dfgff3.loc[gene, 0]
                start = int(dfgff3.loc[gene, 1]) - args.extension
                end = int(dfgff3.loc[gene, 2]) + args.extension
                ranges.append((chrom, start, end))
            if ranges:
                bimranges.append(ranges)

        results = garfield_main(
            bfile,
            common,
            y,
            bimranges,
            nsnp=args.nsnp,
            n_estimators=args.nestimators,
            threads=threads,
        )
    else:
        # Window mode
        results = garfield_window(
            bfile,
            common,
            y,
            args.step,
            args.extension,
            nsnp=args.nsnp,
            n_estimators=args.nestimators,
            threads=threads,
        )

    write_xcombine_results(f"{outprefix}.garfield", list(common), results)
    lt = time.localtime()
    logger.info(
        f"\nFinished GARFIELD. Total wall time: {round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )


if __name__ == "__main__":
    main()
