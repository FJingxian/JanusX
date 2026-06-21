"""
Shared CLI argument builders for JanusX Python entrypoints.

This module centralizes command-line flags whose semantics are intentionally
kept aligned across multiple frontends such as GWAS, GS, GARFIELD, simulation,
and benchmarking scripts.

Organization
------------
1. Input selectors
   - genotype source
   - phenotype file
   - trait selectors
2. Variant QC / decode-budget controls
   - MAF / missing / heterozygosity filters
   - SNP-only toggle
   - decode-memory budget
3. Model-side auxiliary inputs
   - GRM / kernel inputs
   - external covariates
4. Common execution / output controls
   - threads
   - output directory
   - output prefix

Only arguments with stable cross-command meaning should live here. Command-
specific wrappers may still override help text, defaults, destinations, or
option aliases without reimplementing parser boilerplate.
"""

from __future__ import annotations

import argparse
import re
from typing import Sequence


_TRAIT_RANGE_RE = re.compile(r"^\s*(\d+)\s*([:-])\s*(\d+)\s*$")


# ---------------------------------------------------------------------------
# Input selectors
# ---------------------------------------------------------------------------
def _resolve_genotype_source_help(
    *,
    kind: str,
    profile: str,
) -> str:
    key = str(profile).strip().lower()
    table: dict[str, dict[str, str]] = {
        "default": {
            "vcf": "Input genotype file in VCF format (.vcf or .vcf.gz).",
            "hmp": "Input genotype file in HMP format (.hmp or .hmp.gz).",
            "file": (
                "Input genotype numeric matrix (.txt/.tsv/.csv/.npy) or prefix. "
                "Requires sibling prefix.id. Optional site metadata: prefix.site or prefix.bim."
            ),
            "bfile": "Input genotype in PLINK binary format (prefix for .bed, .bim, .fam).",
        },
        "gstats": {
            "vcf": "Input VCF/VCF.GZ; will be cached as BED before statistics.",
            "hmp": "Input HapMap/HMP(.gz); will be cached as BED before statistics.",
            "file": (
                "Input genotype numeric matrix (.txt/.tsv/.csv/.npy/.bin) or prefix. "
                "Requires sibling prefix.id; BED caching also requires prefix.site or prefix.bim."
            ),
            "bfile": "PLINK prefix (.bed/.bim/.fam).",
        },
        "admixture": {
            "vcf": "VCF/VCF.GZ genotype file.",
            "hmp": "HMP/HMP.GZ genotype file.",
            "file": "Text/NumPy genotype matrix prefix (requires sidecar .id).",
            "bfile": "PLINK prefix (.bed/.bim/.fam).",
        },
        "gformat": {
            "vcf": "Input genotype file in VCF format (.vcf or .vcf.gz).",
            "hmp": "Input genotype file in HMP format (.hmp or .hmp.gz).",
            "file": (
                "Input genotype numeric matrix (.txt/.tsv/.csv/.npy) or prefix. "
                "Requires sibling prefix.id. Optional site metadata: prefix.bsite/prefix.site or prefix.bim."
            ),
            "bfile": "Input genotype in PLINK binary format (prefix for .bed/.bim/.fam).",
        },
        "hybrid": {
            "vcf": "Input genotype file in VCF format (.vcf or .vcf.gz).",
            "file": (
                "Input genotype matrix (.txt/.tsv/.csv/.npy) or prefix. "
                "Requires sibling prefix.id. Optional site metadata: prefix.site or prefix.bim. "
                "When prefix-matched .npy exists, it is preferred."
            ),
            "bfile": "Input PLINK prefix (.bed/.bim/.fam).",
        },
        "plink_prefix_short": {
            "bfile": "PLINK prefix (.bed/.bim/.fam).",
        },
    }
    selected = table.get(key, table["default"])
    if str(kind) in selected:
        return str(selected[str(kind)])
    return _resolve_genotype_source_help(kind=str(kind), profile="default")


def add_common_genotype_source_args(
    group: argparse._MutuallyExclusiveGroup | argparse._ArgumentGroup,
    *,
    include_vcf: bool = True,
    include_hmp: bool = True,
    include_file: bool = True,
    include_bfile: bool = True,
    help_profile: str = "default",
) -> None:
    if bool(include_vcf):
        group.add_argument(
            "-vcf",
            "--vcf",
            type=str,
            help=_resolve_genotype_source_help(kind="vcf", profile=help_profile),
        )
    if bool(include_hmp):
        group.add_argument(
            "-hmp",
            "--hmp",
            type=str,
            help=_resolve_genotype_source_help(kind="hmp", profile=help_profile),
        )
    if include_file:
        group.add_argument(
            "-file",
            "--file",
            type=str,
            help=_resolve_genotype_source_help(kind="file", profile=help_profile),
        )
    if bool(include_bfile):
        group.add_argument(
            "-bfile",
            "--bfile",
            type=str,
            help=_resolve_genotype_source_help(kind="bfile", profile=help_profile),
        )


def add_common_pheno_arg(
    group: argparse._ArgumentGroup,
    *,
    required: bool = False,
    help_text: str | None = None,
) -> None:
    group.add_argument(
        "-p",
        "--pheno",
        type=str,
        required=bool(required),
        help=(
            str(help_text)
            if help_text is not None
            else "Phenotype file (tab-delimited, sample IDs in the first column)."
        ),
    )


# ---------------------------------------------------------------------------
# Variant QC / decode-budget controls
# ---------------------------------------------------------------------------
def maf_filter_help_text() -> str:
    return (
        "Exclude variants with minor allele frequency lower than a threshold "
        "(default: %(default)s)."
    )


def missing_rate_filter_help_text() -> str:
    return (
        "Exclude variants with missing call frequencies greater than a threshold "
        "(default: %(default)s)."
    )


def het_filter_help_text() -> str:
    return (
        "Maximum allowed heterozygosity rate threshold. "
        "Sites with het rate greater than this threshold are removed; 0 disables this filter "
        "(default: %(default)s)."
    )


def snps_only_help_text() -> str:
    return "Exclude non-SNP variants."


def _resolve_variant_filter_help(
    *,
    kind: str,
    profile: str,
) -> str:
    key = str(profile).strip().lower()
    table: dict[str, dict[str, str]] = {
        "default": {
            "maf": maf_filter_help_text(),
            "geno": missing_rate_filter_help_text(),
            "het": het_filter_help_text(),
        },
        "short": {
            "maf": "MAF filter (default: %(default)s).",
            "geno": "Missing-rate filter (default: %(default)s).",
        },
        "pureline": {
            "maf": "Minor allele frequency threshold.",
            "geno": (
                "Maximum pure-line missing rate threshold; heterozygotes (1) and true missing (NA) "
                "are both counted as missing."
            ),
        },
        "packed_filter": {
            "maf": "MAF threshold for packed filtering.",
            "geno": "Missing-rate threshold for packed filtering.",
        },
        "loading_stage": {
            "maf": "Minor allele frequency threshold in loading stage (default: 0.02).",
            "geno": "Missing-rate threshold in loading stage (default: 0.05).",
        },
        "gwas_threshold": {
            "maf": "MAF threshold for GWAS (default: %(default)s).",
            "geno": "Missing rate threshold for GWAS (default: %(default)s).",
        },
        "simulation": {
            "maf": "Exclude variants with minor allele frequency lower than a threshold (default: 0.02).",
            "geno": "Exclude variants with missing call frequencies greater than a threshold (default: 0.05).",
            "het": "Optional maximum heterozygosity rate per variant in [0,1]. Disabled by default.",
        },
    }
    selected = table.get(key, table["default"])
    if str(kind) in selected:
        return str(selected[str(kind)])
    return _resolve_variant_filter_help(kind=str(kind), profile="default")


def _resolve_snps_only_help(profile: str) -> str:
    key = str(profile).strip().lower()
    table = {
        "default": snps_only_help_text(),
        "benchmark": "Use SNPs only.",
        "packed_filter": "Restrict to SNP sites when caching text/VCF input.",
        "loading_stage": "Input VCF/HMP: keep SNP variants only during loading.",
    }
    return str(table.get(key, table["default"]))


def _resolve_thread_help(profile: str) -> str:
    key = str(profile).strip().lower()
    table = {
        "default": "Number of CPU threads (default: %(default)s).",
        "short": "Threads.",
        "cpu_short": "CPU threads.",
        "rust_auto": "Thread count for Rust kernels (default: auto-detected).",
        "packed_rust": "Number of CPU threads for packed Rust kernels (default: %(default)s).",
        "glob_jobs": (
            "Number of CPU threads (default: %(default)s). "
            "For glob input, this is also the max parallel chromosome jobs."
        ),
        "tree_auto": "Thread count for tree inference (0=auto; default: %(default)s).",
    }
    return str(table.get(key, table["default"]))


def _resolve_out_help(profile: str) -> str:
    key = str(profile).strip().lower()
    table = {
        "default": "Output directory for results (default: %(default)s).",
        "current_dir": "Output directory (default: current directory).",
        "simple": "Output directory.",
        "converted_results": "Output directory for converted results (default: %(default)s).",
        "plot_annotation": "Output directory for plots and annotation (default: current directory).",
        "pca_results": "Output directory for PCA results (default: current directory).",
    }
    return str(table.get(key, table["default"]))


def _resolve_prefix_help(profile: str) -> str:
    key = str(profile).strip().lower()
    table = {
        "default": "Prefix for output files (default: %(default)s).",
        "simple": "Output prefix.",
        "inferred_input": "Output prefix (default: inferred from input).",
        "inferred_input_filename": "Prefix of output files (default: inferred from input file name).",
        "genotype_basename": "Prefix of output files (default: genotype basename).",
        "input_basename": "Output prefix (default: input basename).",
        "inferred_genotype_input": "Output prefix (default: inferred from genotype input).",
        "input_filename_stem": "Output prefix. Defaults to the input filename stem.",
        "janusx_log": "Prefix of the log file (default: JanusX).",
        "tree_inferred_genotype": "Output file prefix (default: inferred from genotype file name).",
        "first_input": "Prefix for output files (default: inferred from first input).",
    }
    return str(table.get(key, table["default"]))


def _resolve_memory_help(profile: str) -> str:
    key = str(profile).strip().lower()
    table = {
        "gwas_decode": (
            "Target BED decode block size in GB for memmap/packed kernels. "
            "This only controls per-block decode/window sizing, not global process memory "
            "(default: %(default)s)."
        ),
        "gs_decode": (
            "Decode block memory budget in GB for streamed BED kernels in GS. "
            "Applied to rrBLUP-PCG, GBLUP(add), and BayesA/B/Cpi streamed BED paths; "
            "also used as the default Bayes resident-packed promotion cap when "
            "JX_GS_BAYES_PACKED_MAX_MB is not set; "
            "if a kernel keeps two decode buffers, each block uses about half of this budget "
            "(default: %(default)s)."
        ),
    }
    if key not in table:
        raise ValueError(f"Unsupported memory help profile: {profile!r}")
    return str(table[key])


def _resolve_grm_file_help(profile: str) -> str:
    key = str(profile).strip().lower()
    table = {
        "required_lmm": "Required GRM file path for LMM (no auto-build).",
        "garfield_residualization": (
            "Optional precomputed GRM/kernel (.npy or text). If set, GARFIELD residualization "
            "uses this matrix on the aligned sample set instead of rebuilding GRM from BED."
        ),
        "background_kernel": (
            "Optional precomputed GRM/kernel (.npy or text). If set, background effects are "
            "drawn from this kernel instead of marker-wise genotype effects."
        ),
    }
    if key not in table:
        raise ValueError(f"Unsupported GRM file help profile: {profile!r}")
    return str(table[key])


def _resolve_covariate_file_or_site_help(profile: str) -> str:
    key = str(profile).strip().lower()
    table = {
        "default": covariate_file_or_site_help_text(),
        "benchmark": "Additional covariates (JanusX only).",
    }
    return str(table.get(key, table["default"]))


def add_common_maf_arg(
    group: argparse._ArgumentGroup,
    *,
    default: float = 0.02,
    help_profile: str = "default",
    help_text: str | None = None,
) -> None:
    group.add_argument(
        "-maf",
        "--maf",
        type=float,
        default=float(default),
        help=(
            str(help_text)
            if help_text is not None
            else _resolve_variant_filter_help(kind="maf", profile=help_profile)
        ),
    )


def add_common_geno_arg(
    group: argparse._ArgumentGroup,
    *,
    default: float = 0.05,
    help_profile: str = "default",
    help_text: str | None = None,
) -> None:
    group.add_argument(
        "-geno",
        "--geno",
        type=float,
        default=float(default),
        help=(
            str(help_text)
            if help_text is not None
            else _resolve_variant_filter_help(kind="geno", profile=help_profile)
        ),
    )


def add_common_het_arg(
    group: argparse._ArgumentGroup,
    *,
    default: float | None = 0.0,
    help_profile: str = "default",
    help_text: str | None = None,
) -> None:
    group.add_argument(
        "-het",
        "--het",
        type=float,
        default=default,
        help=(
            str(help_text)
            if help_text is not None
            else _resolve_variant_filter_help(kind="het", profile=help_profile)
        ),
    )


def add_common_variant_filter_args(
    group: argparse._ArgumentGroup,
    *,
    help_profile: str = "default",
    include_maf: bool = True,
    include_geno: bool = True,
    include_het: bool = False,
    maf_default: float = 0.02,
    geno_default: float = 0.05,
    het_default: float | None = 0.0,
    maf_help_text: str | None = None,
    geno_help_text: str | None = None,
    het_help_text: str | None = None,
) -> None:
    if bool(include_maf):
        add_common_maf_arg(
            group,
            default=float(maf_default),
            help_profile=help_profile,
            help_text=maf_help_text,
        )
    if bool(include_geno):
        add_common_geno_arg(
            group,
            default=float(geno_default),
            help_profile=help_profile,
            help_text=geno_help_text,
        )
    if bool(include_het):
        add_common_het_arg(
            group,
            default=het_default,
            help_profile=help_profile,
            help_text=het_help_text,
        )


def add_common_snps_only_arg(
    group: argparse._ArgumentGroup,
    *,
    dest: str = "snps_only",
    default: bool = False,
    help_profile: str = "default",
    help_text: str | None = None,
    include_legacy_only_snps_aliases: bool = False,
) -> None:
    option_strings = ["-snps-only", "--snps-only"]
    if bool(include_legacy_only_snps_aliases):
        option_strings.extend(["-only-snps", "--only-snps"])
    group.add_argument(
        *option_strings,
        action="store_true",
        default=bool(default),
        dest=str(dest),
        help=(
            str(help_text)
            if help_text is not None
            else _resolve_snps_only_help(help_profile)
        ),
    )


def add_common_memory_arg(
    group: argparse._ArgumentGroup,
    *,
    default: float,
    help_profile: str | None = None,
    help_text: str | None = None,
    dest: str = "memory",
    include_hidden_legacy_single_dash_alias: bool = False,
) -> None:
    group.add_argument(
        "-mem",
        "--memory",
        type=float,
        default=float(default),
        dest=str(dest),
        help=(
            str(help_text)
            if help_text is not None
            else _resolve_memory_help(
                "gwas_decode" if help_profile is None else str(help_profile)
            )
        ),
    )
    if bool(include_hidden_legacy_single_dash_alias):
        group.add_argument(
            "-memory",
            type=float,
            dest=str(dest),
            help=argparse.SUPPRESS,
        )


# ---------------------------------------------------------------------------
# Model-side auxiliary inputs
# ---------------------------------------------------------------------------
def grm_option_help_text() -> str:
    return (
        "GRM option: 1 (centering), 2 (standardization), "
        "or a path to a precomputed GRM file (default: %(default)s)."
    )


def covariate_file_or_site_help_text() -> str:
    return (
        "Additional covariate input (repeatable). Each -c accepts either: "
        "(1) covariate file path, or "
        "    covariate file format: first column sample ID, remaining columns numeric covariates; or "
        "(2) single-site token chr:pos / chr:start:end (start must equal end, "
        "supports full-width colon). "
        "Examples: -c cov.tsv -c 1:1000 -c 1:1000:1000."
    )


def covariate_file_help_text() -> str:
    return "Optional covariate file (first column is sample ID)."


def add_common_grm_option_arg(
    group: argparse._ArgumentGroup,
    *,
    default: str = "1",
    dest: str = "grm",
    help_text: str | None = None,
) -> None:
    group.add_argument(
        "-k",
        "--grm",
        type=str,
        default=str(default),
        dest=str(dest),
        help=str(help_text) if help_text is not None else grm_option_help_text(),
    )


def add_common_grm_file_arg(
    group: argparse._ArgumentGroup,
    *,
    default: str | None = None,
    required: bool = False,
    dest: str = "grm",
    help_profile: str | None = None,
    help_text: str | None = None,
) -> None:
    group.add_argument(
        "-k",
        "--grm",
        type=str,
        default=default,
        required=bool(required),
        dest=str(dest),
        help=(
            str(help_text)
            if help_text is not None
            else _resolve_grm_file_help(
                "required_lmm" if help_profile is None else str(help_profile)
            )
        ),
    )


def add_common_covariate_file_or_site_arg(
    group: argparse._ArgumentGroup,
    *,
    dest: str = "cov",
    default: list[str] | None = None,
    help_profile: str = "default",
    help_text: str | None = None,
) -> None:
    group.add_argument(
        "-c",
        "--cov",
        action="append",
        type=str,
        default=default,
        dest=str(dest),
        help=(
            str(help_text)
            if help_text is not None
            else _resolve_covariate_file_or_site_help(help_profile)
        ),
    )


def add_common_covariate_file_arg(
    group: argparse._ArgumentGroup,
    *,
    dest: str = "cov",
    default: str | None = None,
    help_text: str | None = None,
    option_strings: Sequence[str] = ("-cov", "--cov"),
) -> None:
    group.add_argument(
        *tuple(str(x) for x in option_strings),
        type=str,
        default=default,
        dest=str(dest),
        help=(
            str(help_text)
            if help_text is not None
            else covariate_file_help_text()
        ),
    )


# ---------------------------------------------------------------------------
# Trait selectors
# ---------------------------------------------------------------------------
def trait_selector_help_text() -> str:
    return (
        "Phenotype column(s), accepted as zero-based index (excluding sample ID), "
        "column name, comma list (e.g. 0,2 or TraitA,TraitB), "
        "or numeric range (e.g. 0:2). "
        "Repeat this flag for multiple traits."
    )


def add_common_trait_selector_args(
    group: argparse._ArgumentGroup,
    *,
    dest: str = "ncol",
    label: str = "COL",
    help_text: str | None = None,
) -> None:
    group.add_argument(
        "-n",
        "--n",
        action="extend",
        nargs="+",
        metavar=str(label),
        type=str,
        default=None,
        dest=str(dest),
        help=str(help_text) if help_text is not None else trait_selector_help_text(),
    )
    group.add_argument(
        "--ncol",
        action="extend",
        nargs="+",
        metavar=str(label),
        type=str,
        default=None,
        dest=str(dest),
        help=argparse.SUPPRESS,
    )


# ---------------------------------------------------------------------------
# Common execution / output controls
# ---------------------------------------------------------------------------
def add_common_thread_arg(
    group: argparse._ArgumentGroup,
    *,
    default_threads: int,
    dest: str = "thread",
    help_profile: str = "default",
    help_text: str | None = None,
) -> None:
    group.add_argument(
        "-t",
        "--thread",
        type=int,
        default=int(default_threads),
        dest=str(dest),
        help=(
            str(help_text)
            if help_text is not None
            else _resolve_thread_help(help_profile)
        ),
    )


def add_common_out_arg(
    group: argparse._ArgumentGroup,
    *,
    default: str = ".",
    help_profile: str = "default",
    help_text: str | None = None,
) -> None:
    group.add_argument(
        "-o",
        "--out",
        type=str,
        default=str(default),
        help=(
            str(help_text)
            if help_text is not None
            else _resolve_out_help(help_profile)
        ),
    )


def add_common_prefix_arg(
    group: argparse._ArgumentGroup,
    *,
    default: str | None = None,
    help_profile: str = "default",
    help_text: str | None = None,
) -> None:
    group.add_argument(
        "-prefix",
        "--prefix",
        type=str,
        default=default,
        help=(
            str(help_text)
            if help_text is not None
            else _resolve_prefix_help(help_profile)
        ),
    )


def parse_trait_selector_specs(
    values: Sequence[object] | None,
    *,
    label: str = "-n/--n",
) -> list[object] | None:
    """
    Parse mixed phenotype selectors from CLI.

    Supported forms:
      - zero-based column index: 0
      - inclusive numeric range: 0:3 / 0-3
      - column name: TraitA
      - mixed use across repeated flags / comma-separated tokens
    """
    if values is None:
        return None

    tokens: list[str] = []
    for raw in values:
        s = str(raw).strip()
        if s == "":
            continue
        for part in s.split(","):
            p = str(part).strip()
            if p != "":
                tokens.append(p)

    if len(tokens) == 0:
        return []

    parsed: list[object] = []
    for tk in tokens:
        m = _TRAIT_RANGE_RE.match(tk)
        if m is not None:
            left = int(m.group(1))
            right = int(m.group(3))
            step = 1 if right >= left else -1
            parsed.extend(int(i) for i in range(left, right + step, step))
            continue
        if tk.isdigit():
            parsed.append(int(tk))
            continue
        parsed.append(tk)

    out: list[object] = []
    seen: set[tuple[str, str]] = set()
    for item in parsed:
        if isinstance(item, int):
            if item < 0:
                raise ValueError(
                    f"Invalid {label} index: {item}. Indices must be >= 0 (zero-based)."
                )
            key = ("i", str(int(item)))
            val: object = int(item)
        else:
            text = str(item).strip()
            if text == "":
                continue
            key = ("s", text)
            val = text
        if key not in seen:
            out.append(val)
            seen.add(key)
    return out


def resolve_trait_selectors(
    columns: Sequence[object],
    selectors: Sequence[object] | None,
    *,
    label: str = "-n/--n",
) -> tuple[list[int], list[str]]:
    """
    Resolve mixed phenotype selectors against a column list.

    Returns
    -------
    selected_indices, invalid_specs
    """
    if selectors is None:
        return list(range(len(columns))), []

    cols = list(columns)
    if len(selectors) == 0:
        return [], []

    by_str: dict[str, list[int]] = {}
    for i, col in enumerate(cols):
        by_str.setdefault(str(col), []).append(i)

    def _resolve_name_to_index(name: object) -> int | None:
        exact = [i for i, col in enumerate(cols) if col == name]
        if len(exact) == 1:
            return int(exact[0])
        if len(exact) > 1:
            raise ValueError(
                f"Ambiguous phenotype selector '{name}' in {label}: multiple columns match exactly."
            )
        matched = by_str.get(str(name), [])
        if len(matched) == 1:
            return int(matched[0])
        if len(matched) > 1:
            raise ValueError(
                f"Ambiguous phenotype selector '{name}' in {label}: multiple columns share this string form."
            )
        return None

    selected: list[int] = []
    invalid: list[str] = []
    seen: set[int] = set()
    for spec in selectors:
        local_idx: int | None = None
        if isinstance(spec, int):
            idx = int(spec)
            if 0 <= idx < len(cols):
                local_idx = idx
            else:
                fallback_idx = _resolve_name_to_index(str(spec))
                if fallback_idx is not None:
                    local_idx = int(fallback_idx)
        else:
            local_idx = _resolve_name_to_index(spec)

        if local_idx is None:
            invalid.append(str(spec))
            continue
        if local_idx in seen:
            continue
        selected.append(int(local_idx))
        seen.add(int(local_idx))
    return selected, invalid
