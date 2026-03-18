import os
import time
import socket
import argparse
import sys
from collections import Counter
from typing import Optional
import numpy as np
from janusx.garfield.garfield2 import (
    load_all_genotype_int8,
    load_all_genotype_packed_bed,
    main_inmemory as garfield_main_inmemory,
    main_inmemory_packed as garfield_main_inmemory_packed,
    window as garfield_window,
)
from janusx.gfreader import SiteInfo, inspect_genotype_file, auto_mmap_window_mb
from janusx.gfreader.gfreader import save_genotype_streaming
from janusx.gtools.reader import readanno
from janusx.script._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    format_path_for_display,
    ensure_plink_prefix_exists,
)
from janusx.script.gwas import load_phenotype
from janusx.script._common.log import setup_logging
from janusx.script._common.config_render import emit_cli_configuration
from janusx.script._common.status import CliStatus, log_success, stdout_is_tty
from janusx.script._common.helptext import CliArgumentParser, cli_help_formatter
from janusx.script._common.genocache import configure_genotype_cache_from_out
from janusx.script._common.genoio import determine_genotype_source_from_args as determine_genotype_source
from janusx.script._common.colspec import parse_zero_based_index_specs
from janusx.script._common.threads import detect_effective_threads


class _GarfieldPhenoLogger:
    """
    Logger proxy for GARFIELD phenotype loading.
    Suppress duplicate selection line from shared load_phenotype().
    """

    def __init__(self, base_logger):
        self._base = base_logger

    def info(self, message, *args, **kwargs):
        msg = str(message)
        if msg.startswith("Phenotypes to be analyzed: "):
            return
        return self._base.info(message, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)


def _looks_numeric_token(token: object) -> bool:
    s = str(token).strip()
    if s == "":
        return False
    try:
        float(s)
        return True
    except Exception:
        return False


def _split_pheno_line(line: str) -> list[str]:
    if "\t" in line:
        return [x.strip() for x in line.split("\t")]
    if "," in line:
        return [x.strip() for x in line.split(",")]
    return [x.strip() for x in line.split()]


def _read_phenotype_header_names(phenofile: str) -> Optional[list[str]]:
    """
    Best-effort read of phenotype header names (excluding sample ID column).
    Return None when no obvious header is present.
    """
    try:
        with open(phenofile, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = str(raw).rstrip("\r\n")
                if line.strip() == "":
                    continue
                parts = _split_pheno_line(line)
                if len(parts) <= 1:
                    return None
                names = [str(x).strip() for x in parts[1:]]
                if len(names) == 0:
                    return None
                if any((n != "") and (not _looks_numeric_token(n)) for n in names):
                    return names
                return None
    except Exception:
        return None
    return None


def _normalize_trait_names_from_header(
    pheno,
    phenofile: str,
):
    """
    When selected phenotype columns are labeled as numeric indices (e.g., '5'),
    map them back to header names (e.g., 'test5') if header is available.
    """
    trait_names = [str(c) for c in pheno.columns]
    selected_ncol = pheno.attrs.get("selected_ncol", None)
    if not isinstance(selected_ncol, list) or len(selected_ncol) != len(trait_names):
        return pheno, trait_names

    header_names = _read_phenotype_header_names(phenofile)
    if not header_names:
        return pheno, trait_names

    mapped: list[str] = []
    changed = False
    for idx_obj, current in zip(selected_ncol, trait_names):
        cur = str(current).strip()
        numeric_like = cur.lstrip("+-").isdigit()
        if numeric_like:
            try:
                idx = int(idx_obj)
            except Exception:
                idx = -1
            if 0 <= idx < len(header_names):
                cand = str(header_names[idx]).strip()
                if cand != "":
                    mapped.append(cand)
                    if cand != cur:
                        changed = True
                    continue
        mapped.append(cur)

    if changed:
        pheno = pheno.copy()
        pheno.columns = mapped
        return pheno, mapped
    return pheno, trait_names


def _safe_trait_label(label: object) -> str:
    name = str(label).strip()
    if name == "":
        return "trait"
    return name.replace("/", "_").replace("\\", "_")


def _format_numeric_for_log(value: float) -> str:
    val = float(value)
    if np.isfinite(val) and val.is_integer():
        return str(int(val))
    return f"{val:g}"


def _detect_response_mode(y: np.ndarray) -> tuple[str, np.ndarray, str]:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if y_arr.size == 0:
        raise ValueError("Phenotype vector is empty after sample alignment.")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("Phenotype contains non-finite values after sample alignment.")

    uniq = np.unique(y_arr)
    if uniq.size == 2:
        lo = float(uniq[0])
        hi = float(uniq[1])
        lo_mask = np.isclose(y_arr, lo, rtol=0.0, atol=1e-12)
        hi_mask = np.isclose(y_arr, hi, rtol=0.0, atol=1e-12)
        if np.all(lo_mask | hi_mask):
            y_bin = hi_mask.astype(np.float64).reshape(-1, 1)
            return ("binary", y_bin, "")

    return "continuous", y_arr.reshape(-1, 1), ""


def _log_window_warning_summary(
    logger,
    trait_name: str,
    warnings: list[str],
    *,
    max_unique: int = 8,
) -> None:
    msgs = [str(w).strip() for w in warnings if str(w).strip() != ""]
    if len(msgs) == 0:
        return
    counter = Counter(msgs)
    logger.warning(
        f"GARFIELD trait '{trait_name}' window scan warnings: "
        f"{len(msgs)} total, {len(counter)} unique."
    )
    for i, (msg, count) in enumerate(counter.most_common(max_unique), start=1):
        logger.warning(f"[{i}] x{count} {msg}")
    if len(counter) > max_unique:
        logger.warning(f"... and {len(counter) - max_unique} more unique warning(s).")


def _resolve_plink_prefix(path_or_prefix: str):
    """
    Return normalized PLINK prefix if {prefix}.bed/.bim/.fam all exist.
    Accept both bare prefix and explicit .bed/.bim/.fam paths.
    """
    raw = str(path_or_prefix)
    cand = [raw]
    low = raw.lower()
    if low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam"):
        cand.append(raw[:-4])
    for pfx in cand:
        if (
            os.path.isfile(f"{pfx}.bed")
            and os.path.isfile(f"{pfx}.bim")
            and os.path.isfile(f"{pfx}.fam")
        ):
            return pfx
    return None


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
    use_spinner = stdout_is_tty()
    parser = CliArgumentParser(
        prog="jx garfield",
        formatter_class=cli_help_formatter(),
    )

    required_group = parser.add_argument_group("Required Arguments")
    geno_group = required_group.add_mutually_exclusive_group(required=False)
    geno_group.add_argument(
        "-bfile", "--bfile", type=str,
        help="Input genotype in PLINK binary format (prefix for .bed/.bim/.fam).",
    )
    geno_group.add_argument(
        "-vcf", "--vcf", type=str,
        help="Input genotype in VCF format (.vcf or .vcf.gz).",
    )
    geno_group.add_argument(
        "-hmp", "--hmp", type=str,
        help="Input genotype in HMP format (.hmp or .hmp.gz).",
    )
    geno_group.add_argument(
        "-file", "--file", type=str,
        help=(
            "Input genotype numeric matrix (.txt/.tsv/.csv/.npy) or prefix. "
            "Requires sibling prefix.id. Optional site metadata: prefix.site or prefix.bim."
        ),
    )
    required_group.add_argument(
        "-p", "--pheno", type=str, required=False,
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
        "-n", "--n", action="extend", nargs="+", metavar="COL", default=None, type=str, dest="ncol",
        help=(
            "Phenotype column(s), zero-based index (excluding sample ID), "
            "comma list (e.g. 0,2), or numeric range (e.g. 0:2). "
            "Repeat this flag for multiple traits."
        ),
    )
    optional_group.add_argument(
        "--ncol", action="extend", nargs="+", metavar="COL", default=None, type=str, dest="ncol",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-forceset", "--forceset", action="store_true", default=False,
        help="Enable gene-set mode when a line has >1 gene (default: off).",
    )
    optional_group.add_argument(
        "-step", "--step", type=int, default=None,
        help="Step size for sliding windows (default: extension/2).",
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
        "-t", "--thread", dest="thread", type=int, default=detect_effective_threads(),
        help="Number of CPU threads (default: %(default)s).",
    )
    optional_group.add_argument(
        "--threads", dest="thread", type=int, default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory (default: current directory).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Output prefix (default: inferred from genotype input).",
    )
    optional_group.add_argument(
        "-mmap-limit", "--mmap-limit", action="store_true", default=False,
        help="Enable windowed mmap for BED inputs (auto: 2x chunk size).",
    )

    args, extras = parser.parse_known_args()
    has_genotype = bool(args.vcf or args.hmp or args.file or args.bfile)
    has_pheno = bool(args.pheno)
    if (not has_pheno) and (not has_genotype):
        parser.error(
            "the following arguments are required: -p/--pheno & "
            "(-vcf VCF | -hmp HMP | -file FILE | -bfile BFILE)"
        )
    if not has_pheno:
        parser.error("the following arguments are required: -p/--pheno")
    if not has_genotype:
        parser.error(
            "the following arguments are required: "
            "(-vcf VCF | -hmp HMP | -file FILE | -bfile BFILE)"
        )
    if len(extras) > 0:
        parser.error("unrecognized arguments: " + " ".join(extras))
    if int(args.extension) <= 0:
        parser.error("-ext/--extension must be > 0")
    if args.step is None:
        args.step = max(1, int(args.extension) // 2)
    elif int(args.step) <= 0:
        parser.error("-step/--step must be > 0")
    try:
        args.ncol = parse_zero_based_index_specs(args.ncol, label="-n/--n")
    except ValueError as e:
        parser.error(str(e))
    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    thread_capped = False
    if int(args.thread) <= 0:
        args.thread = int(detected_threads)
    if int(args.thread) > int(detected_threads):
        thread_capped = True
        args.thread = int(detected_threads)

    gfile, prefix = determine_genotype_source(args)
    args.out = os.path.normpath(args.out if args.out is not None else ".")
    outprefix = os.path.join(args.out, prefix)
    os.makedirs(args.out, mode=0o755, exist_ok=True)
    configure_genotype_cache_from_out(args.out)

    log_path = os.path.join(args.out, f"{prefix}.garfield.log")
    logger = setup_logging(log_path)
    if thread_capped:
        logger.warning(
            f"Warning: Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(args.thread)}."
        )

    threads = int(args.thread)
    cfg_rows: list[tuple[str, object]] = [
        ("Genotype", gfile),
        ("Phenotype", args.pheno),
        ("Gene file", args.genefile),
        ("GFF3", args.gff3),
    ]
    if not (args.genefile and args.gff3):
        cfg_rows.append(("Step", args.step))
    cfg_rows.extend(
        [
            ("Extension", args.extension),
            ("Top SNPs", args.nsnp),
            ("Estimators", args.nestimators),
            ("Mmap limit", args.mmap_limit),
        ]
    )
    emit_cli_configuration(
        logger,
        app_title="JanusX - GARFIELD",
        config_title="GARFIELD CONFIG",
        host=socket.gethostname(),
        sections=[("General", cfg_rows)],
        footer_rows=[
            ("Threads", f"{threads} ({detected_threads} available)"),
            ("Output prefix", outprefix),
        ],
        line_max_chars=60,
    )

    only_one_gene_arg = bool(args.genefile) ^ bool(args.gff3)
    if only_one_gene_arg:
        logger.warning(
            "Only one of --genefile/-g and --gff3/-gff was provided. "
            "Gene/gff3 mode requires both; falling back to window mode."
        )

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    checks.append(ensure_file_exists(logger, args.pheno, "Phenotype file"))
    if args.genefile:
        checks.append(ensure_file_exists(logger, args.genefile, "Gene file"))
    if args.gff3:
        checks.append(ensure_file_exists(logger, args.gff3, "GFF3 file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    # Load genotype meta
    sample_ids, n_snps = inspect_genotype_file(gfile)
    sample_ids = np.array(sample_ids, dtype=str)
    chunk_size = 100_000
    mmap_window_mb = (
        auto_mmap_window_mb(gfile, len(sample_ids), n_snps, chunk_size)
        if args.mmap_limit else None
    )

    # Load phenotype and align to genotype order (drop NaNs)
    with CliStatus("Loading phenotype...", enabled=use_spinner) as task:
        try:
            pheno = load_phenotype(
                args.pheno,
                args.ncol,
                _GarfieldPhenoLogger(logger),
                id_col=0,
            )
        except Exception:
            task.fail("Loading phenotype ...Failed")
            raise
        task.complete("Loading phenotype ...Finished")
    if pheno.shape[1] == 0:
        raise ValueError("No phenotype columns to analyze.")
    pheno, trait_names = _normalize_trait_names_from_header(pheno, args.pheno)
    logger.info("Phenotype columns: " + ", ".join(trait_names))

    pheno_all_ids = set(pheno.index.astype(str).to_numpy())
    sample_pool = [sid for sid in sample_ids if sid in pheno_all_ids]
    if len(sample_pool) == 0:
        raise ValueError("No overlapping samples between genotype and phenotype.")

    bimranges = []
    gset_flags = []
    all_genotype_i8 = None
    all_packed_ctx = None
    all_sites = None
    sample_pool_arr = np.asarray(sample_pool, dtype=str)
    sample_pool_index = {sid: i for i, sid in enumerate(sample_pool)}
    sample_geno_index = {sid: i for i, sid in enumerate(sample_ids.tolist())}

    # Gene/gff3 mode
    if args.genefile and args.gff3 and os.path.isfile(args.genefile) and os.path.isfile(args.gff3):
        genesets = _read_geneset_lines(args.genefile)
        dfgff3 = readanno(args.gff3, "ID").iloc[:, :4].set_index(3)
        dupgenemask = dfgff3.index.duplicated()
        if any(dupgenemask):
            logger.warning(
                f"Duplicated genes in GFF3: {np.unique(dfgff3.index[dupgenemask])}"
            )
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

        plink_prefix = _resolve_plink_prefix(gfile)
        if plink_prefix is not None:
            logger.info(
                "Gene/gff3 mode: loading all genotype as BED-packed matrix "
                "(memory-optimized; decode by region during GARFIELD scan)."
            )
            with CliStatus("Loading packed genotype matrix...", enabled=use_spinner) as task:
                try:
                    all_packed_ctx, all_sites = load_all_genotype_packed_bed(
                        plink_prefix,
                        maf=0.02,
                        missing_rate=0.05,
                    )
                except Exception:
                    task.fail("Loading packed genotype matrix ...Failed")
                    raise
                task.complete("Loading packed genotype matrix ...Finished")
            packed = np.asarray(all_packed_ctx["packed"])
            logger.info(
                "Loaded packed genotype for gene/gset mode: "
                f"{packed.shape[0]} SNPs x {int(all_packed_ctx['n_samples'])} samples, "
                f"bytes_per_snp={packed.shape[1]}, dtype={packed.dtype}"
            )
        else:
            logger.info(
                "Gene/gff3 mode: loading all genotype as int8 into memory "
                "(supports PLINK/VCF/TXT; VCF/TXT use cache conversion)."
            )
            with CliStatus("Loading genotype matrix...", enabled=use_spinner) as task:
                try:
                    all_genotype_i8, all_sites = load_all_genotype_int8(
                        gfile,
                        sample_pool_arr,
                        chunk_size=chunk_size,
                        maf=0.02,
                        missing_rate=0.05,
                        mmap_window_mb=mmap_window_mb,
                        total_snps=n_snps,
                    )
                except Exception:
                    task.fail("Loading genotype matrix ...Failed")
                    raise
                task.complete("Loading genotype matrix ...Finished")
            logger.info(
                "Loaded genotype matrix for gene/gset mode: "
                f"{all_genotype_i8.shape[0]} SNPs x {all_genotype_i8.shape[1]} samples, "
                f"dtype={all_genotype_i8.dtype}"
            )
    else:
        logger.info("Window mode: streaming genotype loading.")

    logger.info("=" * 60)

    used_trait_labels: dict[str, int] = {}
    saved = 0

    for trait_idx, trait in enumerate(pheno.columns):
        if trait_idx > 0:
            logger.info("")
        trait_name = str(trait)
        pheno_col = pheno[trait].dropna()
        pheno_ids = set(pheno_col.index.astype(str).to_numpy())
        common = [sid for sid in sample_pool if sid in pheno_ids]
        if len(common) == 0:
            logger.warning(
                f"No overlapping samples for trait '{trait_name}' after dropna; skipped."
            )
            continue

        y_raw = pheno_col.loc[common].to_numpy(dtype=float).reshape(-1, 1)
        response_mode, y, response_note = _detect_response_mode(y_raw)
        logger.info(f"{trait_name} (n={len(common)}, response={response_mode}{response_note})")

        with CliStatus(
            f"GARFIELD trait '{trait_name}'...",
            enabled=use_spinner,
            timeout=0.1,
        ) as task:
            window_warnings: list[str] = []
            try:
                if all_packed_ctx is not None and all_sites is not None:
                    idx = np.asarray(
                        [sample_geno_index[sid] for sid in common], dtype=np.int64
                    )
                    results = garfield_main_inmemory_packed(
                        all_packed_ctx,
                        all_sites,
                        idx,
                        y,
                        bimranges,
                        nsnp=args.nsnp,
                        n_estimators=args.nestimators,
                        threads=threads,
                        response=response_mode,
                        gsetmodes=gset_flags,
                    )
                elif all_genotype_i8 is not None and all_sites is not None:
                    idx = np.asarray([sample_pool_index[sid] for sid in common], dtype=np.int64)
                    geno_trait = np.ascontiguousarray(all_genotype_i8[:, idx], dtype=np.int8)
                    results = garfield_main_inmemory(
                        geno_trait,
                        all_sites,
                        y,
                        bimranges,
                        nsnp=args.nsnp,
                        n_estimators=args.nestimators,
                        threads=threads,
                        response=response_mode,
                        gsetmodes=gset_flags,
                    )
                else:
                    results, window_warnings = garfield_window(
                        gfile,
                        common,
                        y,
                        args.step,
                        args.extension,
                        nsnp=args.nsnp,
                        n_estimators=args.nestimators,
                        response=response_mode,
                        gsetmode=False,
                        threads=threads,
                        mmap_window_mb=mmap_window_mb,
                        collect_warnings=True,
                    )
            except Exception:
                task.fail(f"GARFIELD trait '{trait_name}' ...Failed")
                raise
            task.complete(f"GARFIELD trait '{trait_name}' ...Finished")

        _log_window_warning_summary(logger, trait_name, window_warnings)

        base_trait = _safe_trait_label(trait_name)
        count = used_trait_labels.get(base_trait, 0) + 1
        used_trait_labels[base_trait] = count
        suffix = base_trait if count == 1 else f"{base_trait}.{count}"
        trait_outprefix = f"{outprefix}.{suffix}"

        write_xcombine_results(f"{trait_outprefix}.garfield", list(common), results)
        log_success(
            logger,
            f"Saved GARFIELD output for trait '{trait_name}': "
            f"{format_path_for_display(f'{trait_outprefix}.garfield')}",
        )
        saved += 1

    if saved == 0:
        raise ValueError("No GARFIELD outputs were generated for the selected phenotype columns.")

    lt = time.localtime()
    log_success(
        logger,
        f"\nFinished GARFIELD. Total wall time: {round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}",
    )


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
