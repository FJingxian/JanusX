import os
import time
import socket
import argparse
import numpy as np
from joblib import cpu_count
from janusx.garfield.garfield2 import main as garfield_main, window as garfield_window
from janusx.gfreader import SiteInfo, inspect_genotype_file, auto_mmap_window_mb
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
        "-n", "--ncol", action="extend", nargs="*", default=None, type=int,
        help="Zero-based phenotype column indices to analyze (same as gwas.py).",
    )
    optional_group.add_argument(
        "-forceset", "--forceset", action="store_true", default=False,
        help="Enable gene-set mode when a line has >1 gene (default: off).",
    )
    optional_group.add_argument(
        "-vartype", "--vartype", type=str, default="continuous",
        choices=["continuous", "binary"],
        help="Phenotype type: continuous or binary (default: %(default)s).",
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
    optional_group.add_argument(
        "-mmap-limit", "--mmap-limit", action="store_true", default=False,
        help="Enable windowed mmap for BED inputs (auto: 2x chunk size).",
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
    logger.info(f"VarType:       {args.vartype}")
    if not (args.genefile and args.gff3):
        logger.info(f"Step:          {args.step}")
    logger.info(f"Extension:     {args.extension}")
    logger.info(f"Top SNPs:      {args.nsnp}")
    logger.info(f"Estimators:    {args.nestimators}")
    threads = cpu_count() if args.threads == -1 else int(args.threads)
    logger.info(f"Threads:       {threads}")
    logger.info(f"Output prefix: {outprefix}")
    logger.info(f"Mmap limit:    {args.mmap_limit}")
    logger.info("*" * 60 + "\n")

    # Load genotype meta
    sample_ids, n_snps = inspect_genotype_file(bfile)
    sample_ids = np.array(sample_ids, dtype=str)
    chunk_size = 100_000
    mmap_window_mb = (
        auto_mmap_window_mb(bfile, len(sample_ids), n_snps, chunk_size)
        if args.mmap_limit else None
    )

    # Load phenotype and align to genotype order (drop NaNs)
    pheno = load_phenotype(args.pheno, args.ncol, logger, id_col=0)
    pheno_col = pheno.iloc[:, 0].dropna()
    pheno_ids = pheno_col.index.astype(str).to_numpy()
    common = [sid for sid in sample_ids if sid in set(pheno_ids)]
    if len(common) == 0:
        raise ValueError("No overlapping samples between genotype and phenotype after dropna.")
    y = pheno_col.loc[common].values.reshape(-1, 1)

    gsetmode = False
    # Gene/gff3 mode
    if args.genefile and args.gff3 and os.path.isfile(args.genefile) and os.path.isfile(args.gff3):
        genesets = _read_geneset_lines(args.genefile)
        dfgff3 = readanno(args.gff3, "ID").iloc[:, :4].set_index(3)
        dupgenemask = dfgff3.index.duplicated()
        if any(dupgenemask):
            logger.warning(f"Duplicated genes in GFF3: {np.unique(dfgff3.index[dupgenemask])}")
        dfgff3 = dfgff3.loc[~dupgenemask]

        bimranges = []
        gset_flags = []
        for geneset in genesets:
            ranges = []
            use_gset = args.forceset and len(geneset) > 1
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
                gset_flags.append(use_gset)

        if args.mmap_limit:
            logger.warning("mmap-limit is ignored in gene/gset mode (uses bim_range).")
        results = garfield_main(
            bfile,
            common,
            y,
            bimranges,
            nsnp=args.nsnp,
            n_estimators=args.nestimators,
            threads=threads,
            response=args.vartype,
            gsetmodes=gset_flags,
            mmap_window_mb=None,
        )
    else:
        # Window mode
        gsetmode = False
        results = garfield_window(
            bfile,
            common,
            y,
            args.step,
            args.extension,
            nsnp=args.nsnp,
            n_estimators=args.nestimators,
            response=args.vartype,
            gsetmode=gsetmode,
            threads=threads,
            mmap_window_mb=mmap_window_mb,
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
