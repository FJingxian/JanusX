"""Phase-aware cis/trans pair screening and conditional association."""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from ._common.cli_args import (
    add_common_covariate_file_or_site_arg,
    add_common_grm_file_arg,
    add_common_out_arg,
    add_common_pheno_arg,
    add_common_thread_arg,
    add_common_trait_selector_args,
    parse_trait_selector_specs,
)
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog


def build_parser() -> CliArgumentParser:
    parser = CliArgumentParser(
        prog="jx ct",
        formatter_class=cli_help_formatter(),
        description=(
            "Screen nearby phased variant pairs for cis/trans effects, then fit "
            "conditional LM or fixed-covariance FvLMM models."
        ),
        epilog=minimal_help_epilog(
            [
                "jx ct -vcf offspring.phased.vcf.gz -p trait.tsv -n 0 -lm -o out/ct",
                "jx ct -bfile fathers mothers --cross crosses.tsv -p trait.tsv -fvlmm -k offspring.npy",
                "jx ct -bfile parents parent-map.tsv -p trait.tsv -lm",
                "jx ct --bfile-list chromosomes.tsv --cross crosses.tsv -p trait.tsv -fvlmm -k offspring.npy",
                "jx ct -vcf parents.vcf --parents parent-map.csv -p trait.tsv -lm",
                "jx ct -hmp parents.hmp --parents parent-map.tsv -p trait.tsv -lm",
            ]
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-vcf",
        "--vcf",
        type=str,
        help=(
            "VCF/VCF.GZ input. Direct mode uses phased calls; with --cross/--parents, "
            "unphased homozygous parental calls derive child phase."
        ),
    )
    source.add_argument(
        "-hmp",
        "--hmp",
        type=str,
        help="HapMap/HMP parental genotypes; use --cross/--parents to identify crosses.",
    )
    source.add_argument(
        "-bfile",
        "--bfile",
        nargs=2,
        metavar=("BFILE", "PARENTS_OR_BFILE"),
        help=(
            "One PLINK prefix plus a parent map, or two parent PLINK prefixes "
            "followed by --cross."
        ),
    )
    source.add_argument(
        "--bfile-list",
        dest="bfile_list",
        type=str,
        help=(
            "TSV manifest of chromosome-wise parent BFILE prefixes. Each row has a father "
            "and mother prefix (one prefix reuses the same BFILE); combine with --cross."
        ),
    )
    parser.add_argument(
        "--cross",
        "--parents",
        dest="cross",
        type=str,
        help=(
            "Parent map with child, father, and mother IDs. Required for two-BFILE mode; "
            "also accepted with -vcf/-hmp. Delimiters may be tab, comma, or whitespace."
        ),
    )

    phenotype = parser.add_argument_group("Phenotype and Covariates")
    add_common_pheno_arg(phenotype, required=True)
    add_common_trait_selector_args(phenotype)
    add_common_covariate_file_or_site_arg(
        phenotype,
        help_text="Covariate file(s) with sample IDs in column 1; repeat -c to add files.",
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("-lm", dest="model", action="store_const", const="lm", help="Fit conditional linear models.")
    model_group.add_argument(
        "-fvlmm",
        dest="model",
        action="store_const",
        const="fvlmm",
        help="Fit fixed-global-covariance FvLMM conditional tests.",
    )
    add_common_grm_file_arg(
        parser.add_argument_group("FvLMM Arguments"),
        default=None,
        help_text="Aligned offspring GRM .npy with a matching ID sidecar; required by -fvlmm.",
    )

    scan = parser.add_argument_group("Pair Scan and QC")
    scan.add_argument("--window-bp", type=int, default=50_000, help="Maximum same-chromosome pair distance in bp.")
    scan.add_argument("--core-bp", type=int, default=50_000, help="Left-endpoint core width in bp.")
    scan.add_argument("--topk", type=int, default=100, help="Maximum screened pairs retained per core.")
    scan.add_argument(
        "-maf",
        "--maf",
        type=float,
        default=0.0,
        help="Minimum raw cis/trans phase-minor frequency among informative double heterozygotes.",
    )
    scan.add_argument("-geno", "--geno", type=float, default=0.05, help="Maximum raw pair unknown rate before imputation.")
    scan.add_argument("--min-cis", type=int, default=0, help="Additional minimum raw cis support.")
    scan.add_argument("--min-trans", type=int, default=0, help="Additional minimum raw trans support.")
    scan.add_argument("--snp-maf", type=float, default=0.0, help="Minimum raw genotype MAF per endpoint SNP.")
    scan.add_argument("--snp-geno", type=float, default=0.05, help="Maximum raw missing genotype rate per endpoint SNP.")
    scan.add_argument(
        "--condition",
        choices=("add-dom", "genotype"),
        default="add-dom",
        help=(
            "add-dom conditions on per-site additive/dominance effects plus the shared "
            "double-heterozygote indicator H; genotype uses all eight unphased background "
            "terms, including H. Both estimate separate cis/trans deviations; pwald jointly "
            "tests whether both deviations are zero."
        ),
    )
    scan.add_argument("--alpha", type=float, default=0.05, help="Alpha used only for the JSON Bonferroni reference threshold.")

    runtime = parser.add_argument_group("Runtime and Output")
    add_common_thread_arg(
        runtime,
        default_threads=1,
        help_text="Threads for the global FvLMM null fit, window screening, and conditional candidate fits.",
    )
    add_common_out_arg(
        runtime,
        default="ct",
        help_text=(
            "Writes <prefix>.ct.tsv and <prefix>.ct.json; multiple traits add a "
            "numbered trait suffix."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    _resolve_source_args(args)
    if args.hmp and not args.cross:
        raise ValueError("-hmp requires --cross/--parents because HMP records do not encode phase")
    if args.model == "fvlmm" and not args.grm:
        raise ValueError("-fvlmm requires -k/--grm with a GRM ID sidecar")
    if args.model == "lm" and args.grm:
        raise ValueError("-k/--grm is only valid with -fvlmm")
    if int(args.thread) <= 0:
        raise ValueError("-t/--thread must be a positive integer")
    for name, value, lower, upper in (
        ("--maf", args.maf, 0.0, 0.5),
        ("--geno", args.geno, 0.0, 1.0),
        ("--snp-maf", args.snp_maf, 0.0, 0.5),
        ("--snp-geno", args.snp_geno, 0.0, 1.0),
    ):
        if not np.isfinite(value) or not lower <= float(value) <= upper:
            raise ValueError(f"{name} must be finite and between {lower} and {upper}")
    for name, value in (
        ("--window-bp", args.window_bp),
        ("--core-bp", args.core_bp),
        ("--topk", args.topk),
        ("--min-cis", args.min_cis),
        ("--min-trans", args.min_trans),
    ):
        minimum = 0 if name in {"--min-cis", "--min-trans"} else 1
        if int(value) < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    if not np.isfinite(args.alpha) or not 0.0 < float(args.alpha) < 1.0:
        raise ValueError("--alpha must be finite and between 0 and 1")


def _unique_ids(ids: Sequence[object], label: str) -> list[str]:
    values = [str(value) for value in ids]
    if any(not value.strip() for value in values):
        raise ValueError(f"{label} contains an empty sample ID")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate sample IDs")
    return values


def _has_bfile_trio(prefix: str) -> bool:
    value = str(prefix)
    base = value[:-4] if value.lower().endswith(".bed") else value
    return all(Path(f"{base}.{extension}").is_file() for extension in ("bed", "bim", "fam"))


def _resolve_source_args(args: argparse.Namespace) -> dict[str, str | None]:
    if args.bfile_list:
        if not args.cross:
            raise ValueError("--bfile-list requires --cross with the child/father/mother map")
        return {
            "kind": "bfile-list",
            "cross_path": str(args.cross),
            "father_prefix": str(args.bfile_list),
            "mother_prefix": None,
        }
    if args.bfile:
        first, second = (str(value) for value in args.bfile)
        if args.cross:
            if not _has_bfile_trio(second):
                raise ValueError(
                    "with two -bfile prefixes, the second argument must be a BFILE "
                    "and --cross must name the parent map"
                )
            return {
                "kind": "bfile",
                "cross_path": str(args.cross),
                "father_prefix": first,
                "mother_prefix": second,
            }
        if _has_bfile_trio(second):
            raise ValueError("two-parent BFILE input requires --cross with the parent map")
        return {
            "kind": "bfile",
            "cross_path": second,
            "father_prefix": first,
            "mother_prefix": first,
        }
    if args.vcf:
        return {
            "kind": "parent-vcf" if args.cross else "vcf",
            "cross_path": str(args.cross) if args.cross else None,
            "father_prefix": None,
            "mother_prefix": None,
        }
    if args.hmp:
        return {
            "kind": "parent-hmp" if args.cross else "hmp",
            "cross_path": str(args.cross) if args.cross else None,
            "father_prefix": None,
            "mother_prefix": None,
        }
    raise ValueError("one genotype input is required: -bfile, -vcf, or -hmp")


def _parent_map_fields(text: str) -> list[str]:
    if "," in text:
        return [field.strip().strip("\ufeff") for field in next(csv.reader([text]))]
    return text.split()


def _parent_map_header(fields: Sequence[str]) -> bool:
    if len(fields) != 3:
        return False
    child_names = {
        "child",
        "child_id",
        "offspring",
        "offspring_id",
        "progeny",
        "progeny_id",
        "f1",
        "f1_id",
        "f1id",
        "f1name",
        "f1_name",
    }
    father_names = {"father", "father_id", "sire", "p1"}
    mother_names = {"mother", "mother_id", "dam", "p2"}
    values = [str(field).strip().lstrip("#").lower() for field in fields]
    return values[0] in child_names and values[1] in father_names and values[2] in mother_names


def read_parent_map(path: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    seen_children: set[str] = set()
    header_pending = True
    with open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                comment_fields = _parent_map_fields(text[1:].strip())
                if _parent_map_header(comment_fields):
                    header_pending = False
                continue
            fields = _parent_map_fields(text)
            if header_pending and _parent_map_header(fields):
                header_pending = False
                continue
            header_pending = False
            if len(fields) != 3 or any(not field for field in fields):
                raise ValueError(f"malformed parent map row {line_number}: {text}")
            child_id, father_id, mother_id = fields
            if child_id in seen_children:
                raise ValueError(f"duplicate child ID '{child_id}' in parent map")
            seen_children.add(child_id)
            rows.append((child_id, father_id, mother_id))
    if not rows:
        raise ValueError(f"parent map {path} contains no child rows")
    return rows


def _has_plink_fid_iid_header(path: str) -> bool:
    fid_names = {"fid", "family", "family_id", "familyid"}
    iid_names = {"iid", "sample", "sample_id", "individual", "individual_id", "individualid"}
    with open(path, "rt", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            commented = text.startswith("#")
            fields = _parent_map_fields(text.lstrip("#").strip() if commented else text)
            if len(fields) >= 3:
                first, second = (field.lower() for field in fields[:2])
                if first in fid_names and second in iid_names:
                    return True
            if not commented:
                return False
    return False


def read_source_ids(
    source_kind: str,
    vcf_path: str | None = None,
    *,
    hmp_path: str | None = None,
    cross_path: str | None = None,
) -> list[str]:
    kind = str(source_kind).strip().lower()
    if cross_path:
        return _unique_ids([row[0] for row in read_parent_map(cross_path)], "parent map")
    if kind == "vcf":
        if not vcf_path:
            raise ValueError("VCF path is required")
        opener = gzip.open if str(vcf_path).lower().endswith(".gz") else open
        with opener(vcf_path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("#CHROM"):
                    columns = line.rstrip("\r\n").split("\t")
                    if len(columns) < 10 or columns[8] != "FORMAT":
                        raise ValueError("VCF #CHROM header has no sample columns")
                    return _unique_ids(columns[9:], "VCF header")
                if not line.startswith("#"):
                    raise ValueError("VCF data appeared before the #CHROM header")
        raise ValueError(f"VCF has no #CHROM header: {vcf_path}")
    if kind in {"bfile", "parents", "parent-vcf", "parent-hmp"}:
        raise ValueError("parent map path is required for family genotype input")
    if kind == "hmp":
        if not hmp_path:
            raise ValueError("HMP path is required")
        opener = gzip.open if str(hmp_path).lower().endswith(".gz") else open
        with opener(hmp_path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                columns = line.rstrip("\r\n").split("\t")
                if len(columns) < 12:
                    columns = line.split()
                if len(columns) < 12:
                    raise ValueError("HMP header has fewer than 12 columns")
                return _unique_ids(columns[11:], "HMP header")
        raise ValueError(f"HMP has no header: {hmp_path}")
    raise ValueError(f"unsupported CT source kind: {source_kind}")


def select_complete_samples(
    source_ids: Sequence[object],
    phenotype_ids: Sequence[object],
    phenotype_values: np.ndarray,
    covariate_ids: Sequence[object] | None = None,
    covariates: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    source = _unique_ids(source_ids, "genotype source")
    pheno_ids = _unique_ids(phenotype_ids, "phenotype table")
    y = np.asarray(phenotype_values, dtype=np.float64).reshape(-1)
    if y.size != len(pheno_ids):
        raise ValueError("phenotype values and IDs have different lengths")
    if covariates is None:
        cov = np.empty((0, 0), dtype=np.float64)
        cov_ids: list[str] = []
        cov_index: dict[str, int] | None = None
    else:
        cov = np.asarray(covariates, dtype=np.float64)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)
        if cov.ndim != 2:
            raise ValueError("covariates must be a two-dimensional matrix")
        if covariate_ids is None:
            raise ValueError("covariate sample IDs are required when covariates are supplied")
        cov_ids = _unique_ids(covariate_ids, "covariate table")
        if cov.shape[0] != len(cov_ids):
            raise ValueError("covariate values and IDs have different row counts")
        cov_index = {sample_id: index for index, sample_id in enumerate(cov_ids)}

    pheno_index = {sample_id: index for index, sample_id in enumerate(pheno_ids)}
    selected_ids: list[str] = []
    selected_y: list[float] = []
    selected_cov: list[np.ndarray] = []
    for sample_id in source:
        pheno_row = pheno_index.get(sample_id)
        if pheno_row is None or not np.isfinite(y[pheno_row]):
            continue
        if cov_index is None:
            cov_row = np.empty(0, dtype=np.float64)
        else:
            cov_row_index = cov_index.get(sample_id)
            if cov_row_index is None:
                continue
            cov_row = cov[cov_row_index]
            if not np.isfinite(cov_row).all():
                continue
        selected_ids.append(sample_id)
        selected_y.append(float(y[pheno_row]))
        selected_cov.append(np.asarray(cov_row, dtype=np.float64))
    if not selected_ids:
        raise ValueError("no source samples have complete phenotype and covariate data")
    cov_out = np.vstack(selected_cov) if selected_cov else np.empty((len(selected_ids), 0))
    return (
        selected_ids,
        np.ascontiguousarray(selected_y, dtype=np.float64),
        np.ascontiguousarray(cov_out, dtype=np.float64),
    )


def _load_gwas_tables(args: argparse.Namespace, source_ids: list[str], logger: logging.Logger):
    from janusx.assoc.workflow import _load_covariates_for_models

    selectors = parse_trait_selector_specs(args.ncol, label="-n/--ncol")
    phenotype = _load_ct_phenotype(args.pheno, source_ids, selectors, logger)
    if args.cov:
        genotype_source = args.vcf or args.hmp or (
            args.bfile[0] if args.bfile else args.bfile_list
        )
        if any(":" in str(value) and not Path(str(value)).is_file() for value in args.cov):
            raise ValueError(
                "CT v1 accepts covariate files; site-derived covariates would require an offspring genotype conversion"
            )
        cov, cov_ids = _load_covariates_for_models(
            args.cov,
            genotype_source,
            np.asarray(source_ids, dtype=str),
            4096,
            logger,
            "CT",
            use_spinner=False,
            snps_only=True,
            allow_missing=True,
            allow_partial_samples=True,
            preserve_precision=True,
        )
        return phenotype, cov, cov_ids
    return phenotype, None, None


def _load_ct_phenotype(
    path: str,
    source_ids: Sequence[str],
    selectors: Sequence[object] | None,
    logger: logging.Logger,
):
    from janusx.assoc.workflow import load_phenotype

    selector_list = list(selectors) if selectors is not None else None
    plink_ids = _has_plink_fid_iid_header(path)
    phenotype = load_phenotype(
        path,
        selector_list,
        logger,
        id_col=1 if plink_ids else 0,
    )
    phenotype.index = phenotype.index.astype(str)
    if plink_ids:
        logger.info("Using phenotype column 2 (IID) as sample IDs from the FID/IID table.")
        return phenotype
    source_set = set(_unique_ids(source_ids, "genotype source"))
    overlap = len(source_set.intersection(phenotype.index))
    if overlap:
        return phenotype
    try:
        iid_phenotype = load_phenotype(
            path,
            selector_list,
            logger,
            id_col=1,
        )
    except Exception:
        return phenotype
    iid_phenotype.index = iid_phenotype.index.astype(str)
    if not source_set.intersection(iid_phenotype.index):
        return phenotype
    logger.warning("No CT samples matched phenotype column 1; using column 2 (IID) as sample IDs.")
    return iid_phenotype


def _trait_output_prefix(
    prefix: str,
    trait: object,
    multiple_traits: bool,
    trait_index: int | None = None,
) -> str:
    if not multiple_traits:
        return str(prefix)
    if trait_index is None or trait_index < 0:
        raise ValueError("trait_index must be a non-negative index for multi-trait CT output")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(trait)).strip("._") or "trait"
    return f"{prefix}.{trait_index + 1:04d}.{slug}"


def _run(args: argparse.Namespace, logger: logging.Logger) -> int:
    from janusx import janusx as jxrs

    source_args = _resolve_source_args(args)
    source_kind = str(source_args["kind"])
    source_ids = read_source_ids(
        source_kind,
        args.vcf,
        hmp_path=args.hmp,
        cross_path=source_args["cross_path"],
    )
    phenotype, all_covariates, all_covariate_ids = _load_gwas_tables(args, source_ids, logger)
    if phenotype.shape[1] == 0:
        raise ValueError("no phenotype traits were selected")

    if args.model == "fvlmm" and Path(args.grm).suffix.lower() != ".npy":
        raise ValueError("CT FvLMM v1 accepts .npy GRMs only; this avoids creating a conversion cache")

    multiple_traits = phenotype.shape[1] > 1
    for trait_index, trait in enumerate(phenotype.columns):
        if all_covariates is None:
            covariate_ids = None
            covariates = None
        else:
            covariate_ids = np.asarray(all_covariate_ids, dtype=str).tolist()
            covariates = np.asarray(all_covariates, dtype=np.float64)
        sample_ids, y, cov = select_complete_samples(
            source_ids,
            phenotype.index.astype(str).tolist(),
            phenotype[trait].to_numpy(dtype=np.float64),
            covariate_ids,
            covariates,
        )
        aligned_grm = None
        if args.model == "fvlmm":
            from ._common.grmio import load_and_align_grm

            aligned_grm, grm_id_path = load_and_align_grm(args.grm, sample_ids, label="CT GRM")
            if grm_id_path is None:
                raise ValueError(
                    "CT FvLMM requires a GRM ID sidecar so child sample order can be verified"
                )
        output_prefix = _trait_output_prefix(
            args.out, trait, multiple_traits, trait_index=trait_index
        )
        logger.info("CT trait %s: %d fixed samples", trait, len(sample_ids))
        result = jxrs.ct_run(
            source_kind=source_kind,
            sample_ids=sample_ids,
            y=np.ascontiguousarray(y, dtype=np.float64),
            covariates=np.ascontiguousarray(cov, dtype=np.float64),
            output_prefix=output_prefix,
            vcf_path=args.vcf,
            hmp_path=args.hmp,
            father_prefix=source_args["father_prefix"],
            mother_prefix=source_args["mother_prefix"],
            cross_path=source_args["cross_path"],
            grm=aligned_grm,
            grm_ids=sample_ids if aligned_grm is not None else None,
            model=args.model,
            condition=args.condition,
            phenotype=str(trait),
            window_bp=args.window_bp,
            core_bp=args.core_bp,
            topk=args.topk,
            snp_maf=args.snp_maf,
            snp_geno=args.snp_geno,
            phase_minor_freq=args.maf,
            pair_geno=args.geno,
            min_cis=args.min_cis,
            min_trans=args.min_trans,
            alpha=args.alpha,
            threads=args.thread,
        )
        logger.info(
            "CT complete: M=%s, rows=%s, TSV=%s, JSON=%s",
            result.get("M"),
            result.get("rows_written"),
            result.get("tsv_path"),
            result.get("json_path"),
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("janusx.ct")
    try:
        return _run(args, logger)
    except Exception as error:
        logger.error("CT failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
