"""Whole-chromosome PBWT IBD segmentation and multi-parent ancestry painting (`jx ibd`)."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import shlex
import socket
import sys
import time
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from janusx import janusx as jxrs
from janusx._plink import open_text_auto, pvar_path_for_prefix
from ._common.cli_core import (
    CliArgumentParser,
    cli_help_formatter,
    minimal_help_epilog,
)
from ._common.config_render import emit_cli_configuration
from ._common.log import get_log_path, setup_logging
from ._common.progress import (
    CliStatus,
    _emit_to_file_handlers,
    build_rich_progress,
    format_elapsed,
    log_success,
    rich_progress_available,
    stdout_is_tty,
)
from ._common.threads import detect_effective_threads


_REPORT_RULE = "=" * 72
_TERMINAL_SECTION_WIDTH = 60
_IBD_PROGRESS_BAR_WIDTH = 12


def _janusx_version() -> str:
    try:
        return version("janusx")
    except PackageNotFoundError:
        return "unknown"


def _invocation_command() -> str:
    return shlex.join([str(value) for value in sys.argv])


def _report_banner(logger: logging.Logger, title: str) -> None:
    logger.info(_REPORT_RULE)
    logger.info(str(title))
    logger.info(_REPORT_RULE)
    logger.info("")


def _report_block(logger: logging.Logger, title: str) -> None:
    logger.info(f"[ {title} ]")


def _report_kv(logger: logging.Logger, key: str, value: object) -> None:
    logger.info(f"  {str(key):<20} : {value}")


def _logger_handlers(
    logger: logging.Logger,
    name: str,
    *,
    file_handlers: bool,
) -> logging.Logger:
    child = logging.getLogger(name)
    child.handlers.clear()
    child.setLevel(logging.INFO)
    child.propagate = False
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler) == bool(file_handlers):
            child.addHandler(handler)
    return child


def _strip_input_suffix(path: Path, suffixes: tuple[str, ...]) -> Path:
    lower = str(path).lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            return Path(str(path)[: -len(suffix)])
    return path


def _has_complete_pfile(prefix: Path) -> bool:
    return all(
        member.is_file()
        for member in (
            Path(f"{prefix}.pgen"),
            Path(pvar_path_for_prefix(str(prefix))),
            Path(f"{prefix}.psam"),
        )
    )


def _has_complete_bfile(prefix: Path) -> bool:
    return all(
        member.is_file()
        for member in (
            Path(f"{prefix}.bed"),
            Path(f"{prefix}.bim"),
            Path(f"{prefix}.fam"),
        )
    )


def _selected_target(args: argparse.Namespace) -> tuple[str, Path]:
    for kind in ("bfile", "pfile", "vcf", "hmp"):
        value = getattr(args, kind)
        if value is not None:
            path = Path(value).expanduser()
            if kind == "bfile":
                prefix = _strip_input_suffix(path, (".bed", ".bim", ".fam"))
                if not _has_complete_bfile(prefix):
                    raise ValueError(
                        "bfile input must be a complete PLINK BED prefix with "
                        f".bed/.bim/.fam: {value}"
                    )
                # Pass explicit .bed path so Rust never mis-detects a co-located .pgen.
                return kind, Path(f"{prefix}.bed")
            if kind == "pfile":
                prefix = _strip_input_suffix(
                    path, (".pgen", ".pvar.zst", ".pvar", ".psam")
                )
                if not _has_complete_pfile(prefix):
                    raise ValueError(
                        "pfile input must be a complete PLINK 2 PGEN prefix with "
                        f".pgen/.pvar/.psam: {value}"
                    )
                # Pass explicit .pgen path so Rust never prefers a co-located .bed.
                return kind, Path(f"{prefix}.pgen")
            if not path.is_file():
                raise ValueError(f"{kind} input not found: {path}")
            lower = str(path).lower()
            if kind == "vcf" and not (
                lower.endswith(".vcf") or lower.endswith(".vcf.gz")
            ):
                raise ValueError(f"-vcf expects a .vcf or .vcf.gz filename: {path}")
            if kind == "hmp" and not (
                lower.endswith(".hmp")
                or lower.endswith(".hmp.gz")
                or lower.endswith(".hmp.txt")
                or lower.endswith(".hmp.txt.gz")
            ):
                raise ValueError(f"-hmp expects a .hmp or .hmp.gz filename: {path}")
            return kind, path
    raise ValueError("one target genotype input is required (-bfile/-pfile/-vcf/-hmp)")


def _resolve_reference_arg(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    lower = str(path).lower()
    if lower.endswith((".vcf", ".vcf.gz", ".hmp", ".hmp.gz", ".hmp.txt", ".hmp.txt.gz")):
        if not path.is_file():
            raise ValueError(f"reference genotype file not found: {path}")
        return path
    if lower.endswith((".pgen", ".pvar.zst", ".pvar", ".psam")):
        pfile_prefix = _strip_input_suffix(
            path, (".pgen", ".pvar.zst", ".pvar", ".psam")
        )
        if not _has_complete_pfile(pfile_prefix):
            raise ValueError(
                "reference PFILE must have complete .pgen/.pvar/.psam files: "
                f"{value}"
            )
        return Path(f"{pfile_prefix}.pgen")
    if lower.endswith((".bed", ".bim", ".fam")):
        bfile_prefix = _strip_input_suffix(path, (".bed", ".bim", ".fam"))
        if not _has_complete_bfile(bfile_prefix):
            raise ValueError(
                "reference BFILE must have complete .bed/.bim/.fam files: "
                f"{value}"
            )
        return Path(f"{bfile_prefix}.bed")
    has_pfile = _has_complete_pfile(path)
    has_bfile = _has_complete_bfile(path)
    if has_pfile and has_bfile:
        raise ValueError(
            f"ambiguous reference prefix '{value}': both PFILE (.pgen/.pvar/.psam) "
            "and BFILE (.bed/.bim/.fam) exist; pass the explicit .pgen or .bed filename"
        )
    if has_pfile:
        return Path(f"{path}.pgen")
    if has_bfile:
        return Path(f"{path}.bed")
    raise ValueError(
        "reference must be a VCF/HMP file or a complete PFILE/BFILE prefix: "
        f"{value}"
    )


def _output_prefix(value: Path) -> Path:
    path = Path(value).expanduser()
    lower = str(path).lower()
    if lower.endswith(".jhibd"):
        return Path(str(path)[: -len(".jhibd")])
    return path


def _resolve_mode(
    requested_mode: str, reference: Path | None, ancestry_model: str = "viterbi"
) -> str:
    mode = str(requested_mode).strip().lower()
    model = str(ancestry_model).strip().lower()
    if mode == "pairwise":
        if model == "mott":
            raise ValueError(
                "--ancestry-model mott cannot be used with --mode pairwise; "
                "use --mode ancestry (or omit --mode) with -ref/--reference"
            )
        if reference is not None:
            raise ValueError(
                "-ref/--reference cannot be used with --mode pairwise; "
                "use --mode ancestry (or omit --mode) for multi-parent ancestry painting"
            )
        return "pairwise"
    if mode == "ancestry":
        if reference is None:
            raise ValueError("--mode ancestry requires -ref/--reference")
        return "ancestry"
    if model == "mott" and reference is None:
        raise ValueError("--ancestry-model mott requires -ref/--reference")
    return "ancestry" if reference is not None else "pairwise"


def _validate_parameters(args: argparse.Namespace) -> None:
    if args.threads <= 0:
        raise ValueError("-t/--thread must be a positive integer")
    if args.map is not None and not Path(args.map).expanduser().is_file():
        raise ValueError(f"genetic map not found: {args.map}")
    if not math.isfinite(args.cm_per_mb) or args.cm_per_mb <= 0.0:
        raise ValueError("--cm-per-mb must be a finite positive number")
    if not math.isfinite(args.inbred_het_threshold) or not (
        0.0 <= args.inbred_het_threshold <= 1.0
    ):
        raise ValueError("--inbred-het-threshold must be in [0, 1]")
    if not math.isfinite(args.min_seed_cm) or args.min_seed_cm <= 0.0:
        raise ValueError("--min-seed-cm must be a finite positive number")
    if not math.isfinite(args.min_output_cm) or args.min_output_cm <= 0.0:
        raise ValueError("--min-cm/--min-output-cm must be a finite positive number")
    if not math.isfinite(args.max_gap_cm) or args.max_gap_cm < 0.0:
        raise ValueError("--max-gap-cm must be finite and nonnegative")
    if not math.isfinite(args.max_mismatch_rate) or not (
        0.0 <= args.max_mismatch_rate < 0.5
    ):
        raise ValueError("--max-mismatch-rate must be in [0, 0.5)")
    if args.min_markers < 1:
        raise ValueError("--min-markers must be a positive integer")
    if not math.isfinite(args.ancestry_step_cm) or args.ancestry_step_cm <= 0.0:
        raise ValueError("--ancestry-step-cm must be a finite positive number")
    if args.ancestry_step_max_markers < 0:
        raise ValueError("--ancestry-step-max-markers must be a nonnegative integer")
    if (
        not math.isfinite(args.ancestry_switch_penalty)
        or args.ancestry_switch_penalty <= 0.0
    ):
        raise ValueError("--ancestry-switch-penalty must be a finite positive number")
    if args.ancestry_top_donors < 1:
        raise ValueError("--ancestry-top-donors must be a positive integer")
    if str(args.ancestry_model).strip().lower() == "mott":
        if not math.isfinite(args.mott_generations) or args.mott_generations <= 0.0:
            raise ValueError("--mott-generations must be a finite positive number")
        if not math.isfinite(args.mott_parent_match_prob) or not (
            0.0 < args.mott_parent_match_prob < 1.0
        ):
            raise ValueError("--mott-parent-match-prob must be in (0, 1)")
        if not math.isfinite(args.mott_map_scale_cm) or args.mott_map_scale_cm <= 0.0:
            raise ValueError("--mott-map-scale-cm must be a finite positive number")
        if not math.isfinite(args.mott_unknown_fold) or args.mott_unknown_fold <= 0.0:
            raise ValueError("--mott-unknown-fold must be a finite positive number")
        if args.mott_max_markers_per_chrom < 0:
            raise ValueError("--mott-max-markers-per-chrom must be a nonnegative integer")
        if not math.isfinite(args.mott_ambiguity_delta) or args.mott_ambiguity_delta < 0.0:
            raise ValueError("--mott-ambiguity-delta must be finite and nonnegative")


def build_parser(argv: Sequence[str] | None = None) -> CliArgumentParser:
    argv_tokens = list(sys.argv[1:] if argv is None else argv)
    show_dev_help = "-dev" in argv_tokens or "--dev" in argv_tokens

    def help_text(text: str, *, advanced: bool = False):
        return (str(text), argparse.SUPPRESS)[bool(advanced and not show_dev_help)]

    parser = CliArgumentParser(
        prog="jx ibd",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx ibd -vcf population.vcf.gz -o pop_ibd",
                "jx ibd -pfile dh_lines -ref founders.vcf.gz --mode ancestry -o dh_ancestry",
                "jx ibd -bfile maize_inbreds --ploidy inbred --min-cm 3.0 --write-tsv -o inbred_ibd",
                "jx ibd -h -dev",
            ]
        ),
    )
    parser.add_argument(
        "-dev",
        "--dev",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    genotype = parser.add_argument_group(
        "Genotype Arguments (Required: Select exactly one)"
    )
    target = genotype.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "-bfile",
        "--bfile",
        help="Input PLINK BED prefix (.bed/.bim/.fam).",
    )
    target.add_argument(
        "-pfile",
        "--pfile",
        help="Input PLINK 2 PGEN prefix (.pgen/.pvar/.psam).",
    )
    target.add_argument(
        "-vcf",
        "--vcf",
        help="Input genotype file in VCF format (.vcf or .vcf.gz).",
    )
    target.add_argument(
        "-hmp",
        "--hmp",
        help="Input genotype file in HapMap format (.hmp or .hmp.gz).",
    )

    mode_group = parser.add_argument_group("Mode & Reference Arguments")
    mode_group.add_argument(
        "-mode",
        "--mode",
        choices=("auto", "pairwise", "ancestry"),
        default="auto",
        help=(
            "Execution mode: 'pairwise' for all-vs-all IBD1/IBD2 + sparse kinship CSR, "
            "'ancestry' for multi-parent ancestry painting (requires -ref), or 'auto' "
            "to choose ancestry when -ref is provided and pairwise otherwise "
            "(default: %(default)s)."
        ),
    )
    mode_group.add_argument(
        "-ref",
        "--reference",
        help=(
            "Reference parent panel (VCF/HMP file or PFILE/BFILE prefix) for "
            "multi-parent ancestry painting."
        ),
    )
    mode_group.add_argument(
        "-map",
        "--map",
        help="PLINK genetic map with CHR, SNP, CM, and BP columns.",
    )
    mode_group.add_argument(
        "--cm-per-mb",
        type=float,
        default=1.0,
        help=help_text(
            "Fallback recombination rate in cM/Mb when -map is omitted (default: %(default)s).",
            advanced=True,
        ),
    )

    model = parser.add_argument_group("IBD & Ancestry Model Arguments")
    model.add_argument(
        "--ploidy",
        choices=("auto", "inbred", "diploid"),
        default="auto",
        help=(
            "Sample ploidy mode: auto-detect inbred/DH lines vs diploids, or force "
            "inbred/diploid (default: %(default)s)."
        ),
    )
    model.add_argument(
        "--inbred-het-threshold",
        type=float,
        default=0.02,
        help=help_text(
            "Maximum heterozygosity rate for auto-classifying inbred/DH lines (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "--min-cm",
        "--min-output-cm",
        dest="min_output_cm",
        type=float,
        default=2.0,
        help="Minimum output IBD segment length in cM (default: %(default)s).",
    )
    model.add_argument(
        "--min-seed-cm",
        type=float,
        default=1.0,
        help=help_text(
            "Minimum exact PBWT seed length in cM (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "--max-gap-cm",
        type=float,
        default=0.5,
        help=help_text(
            "Maximum gap in cM bridged during error-tolerant IBD extension (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "--max-mismatch-rate",
        type=float,
        default=0.005,
        help=help_text(
            "Maximum mismatch rate allowed within an IBD segment (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "--min-markers",
        type=int,
        default=30,
        help=help_text(
            "Minimum jointly observed markers required per IBD segment (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "--ancestry-step-cm",
        type=float,
        default=0.05,
        help=help_text(
            "Genetic step size in cM for Viterbi ancestry decoding (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "--ancestry-step-max-markers",
        type=int,
        default=0,
        help=help_text(
            "Optional per-step effective marker cap for high-density WGS ancestry "
            "decoding; 0 disables step scaling (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "--ancestry-switch-penalty",
        type=float,
        default=12.0,
        help=help_text(
            "Viterbi parent-switch penalty for ancestry painting (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "--ancestry-top-donors",
        type=int,
        default=64,
        help=help_text(
            "Maximum candidate reference haplotypes scored per target haplotype (default: %(default)s).",
            advanced=True,
        ),
    )

    mott_group = parser.add_argument_group("Ancestry Model Arguments")
    mott_group.add_argument(
        "--ancestry-model",
        choices=("viterbi", "mott"),
        default="viterbi",
        help=(
            "Multi-parent ancestry inference model: 'viterbi' (default PBWT/Viterbi) "
            "or 'mott' (author-compatible CUBIC Mott HMM) (default: %(default)s)."
        ),
    )
    mott_group.add_argument(
        "--mott-generations",
        type=float,
        default=9.0,
        help=help_text(
            "Effective generations G for Mott HMM transitions (default: %(default)s).",
            advanced=True,
        ),
    )
    mott_group.add_argument(
        "--mott-parent-match-prob",
        type=float,
        default=0.97,
        help=help_text(
            "Founder match emission probability p1 for Mott HMM (default: %(default)s).",
            advanced=True,
        ),
    )
    mott_group.add_argument(
        "--mott-map-scale-cm",
        type=float,
        default=50.0,
        help=help_text(
            "Genetic map scale denominator in cM for Mott HMM lambda (default: %(default)s).",
            advanced=True,
        ),
    )
    mott_group.add_argument(
        "--mott-unknown-fold",
        type=float,
        default=2.0,
        help=help_text(
            "Strict unknown threshold multiplier: max_f P(s,f) > fold / K (default: %(default)s).",
            advanced=True,
        ),
    )
    mott_group.add_argument(
        "--mott-max-markers-per-chrom",
        type=int,
        default=0,
        help=help_text(
            "Deterministic per-chromosome informative marker cap for Mott HMM; 0 keeps all (default: %(default)s).",
            advanced=True,
        ),
    )
    mott_group.add_argument(
        "--mott-write-posterior",
        action="store_true",
        default=False,
        help=help_text(
            "Write <out>.posterior.tsv for single-sample Mott HMM runs (default: %(default)s).",
            advanced=True,
        ),
    )
    mott_group.add_argument(
        "--mott-write-diagnostics",
        action="store_true",
        default=False,
        help=help_text(
            "Write <out>.mott_diagnostics.tsv.gz with segment-level posterior runner-up diagnostics "
            "(default: %(default)s).",
            advanced=True,
        ),
    )
    mott_group.add_argument(
        "--mott-ambiguity-delta",
        type=float,
        default=0.02,
        help=help_text(
            "Posterior-mean delta used for Mott ambiguous-parent diagnostics (default: %(default)s).",
            advanced=True,
        ),
    )

    output = parser.add_argument_group("Output Arguments")
    output.add_argument(
        "-o",
        "--out",
        required=True,
        help="Output prefix (writes <out>.jhibd sparse archive and <out>.log).",
    )
    output.add_argument(
        "--write-tsv",
        action="store_true",
        help="Also export human-readable gzipped TSV segments (<out>.jhibd.tsv.gz).",
    )

    runtime = parser.add_argument_group("Runtime Arguments")
    runtime.add_argument(
        "-t",
        "--thread",
        dest="threads",
        type=int,
        default=None,
        help="Number of CPU threads (default: auto-detected).",
    )
    return parser


def _process_rss_bytes() -> int:
    current_rss = 0
    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            resident_pages = int(statm.read_text().split()[1])
            current_rss = resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            current_rss = 0
    try:
        import resource

        maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform.startswith("linux"):
            maximum_rss *= 1024
    except (ImportError, OSError, ValueError):
        maximum_rss = 0
    return max(current_rss, maximum_rss)


def _open_text_input(path: Path):
    lower = str(path).lower()
    if lower.endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
    if lower.endswith(".zst"):
        return open_text_auto(str(path))
    return path.open(mode="rt", encoding="utf-8", errors="replace")


def _inspect_reference_dimensions(
    reference: Path | None,
) -> tuple[int | None, int | None]:
    """Fast `(ref_samples, ref_max_chrom_markers)` inspection for BED/PGEN/VCF/HMP reference."""
    if reference is None:
        return None, None
    lower = str(reference).lower()
    try:
        if lower.endswith(".bed"):
            samples, _, _, max_chrom = _inspect_target_dimensions("bfile", reference)
            return samples, max_chrom
        if lower.endswith(".pgen"):
            samples, _, _, max_chrom = _inspect_target_dimensions("pfile", reference)
            return samples, max_chrom
        if lower.endswith((".vcf", ".vcf.gz")):
            with _open_text_input(reference) as fh:
                for raw in fh:
                    if raw.startswith("#CHROM"):
                        fields = raw.rstrip("\r\n").split("\t")
                        return (max(0, len(fields) - 9) or None), None
                    if not raw.startswith("#"):
                        break
            return None, None
        if lower.endswith((".hmp", ".hmp.gz", ".hmp.txt", ".hmp.txt.gz")):
            with _open_text_input(reference) as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    fields = line.split("\t") if "\t" in line else line.split()
                    return (max(0, len(fields) - 11) or None), None
            return None, None
    except Exception:
        return None, None
    return None, None


_PREFLIGHT_TEXT_SCAN_MAX_BYTES = 4 * 1024 * 1024


def _read_pgen_header_dimensions(pgen_path: Path) -> tuple[int | None, int | None]:
    """Read `(samples, markers)` from a PLINK 2 `.pgen` header in O(1) time."""
    with pgen_path.open("rb") as fh:
        header = fh.read(12)
    if len(header) < 3 or header[0:2] != b"\x6c\x1b":
        return None, None
    mode = header[2]
    if mode in (0x02, 0x10) and len(header) >= 11:
        markers = int.from_bytes(header[3:7], "little") or None
        samples = int.from_bytes(header[7:11], "little") or None
        return samples, markers
    return None, None


def _inspect_target_dimensions(
    target_kind: str, target: Path
) -> tuple[int | None, int | None, int | None, int | None]:
    """Fast `(samples, markers, chromosomes, max_chrom_markers)` inspection for BED/PGEN."""
    try:
        if target_kind == "bfile":
            prefix = _strip_input_suffix(target, (".bed", ".bim", ".fam"))
            fam = Path(f"{prefix}.fam")
            bim = Path(f"{prefix}.bim")
            bed = Path(f"{prefix}.bed")
            samples = None
            markers = None
            chrom_counts: dict[str, int] = {}
            if fam.is_file():
                with fam.open("rb") as fh:
                    samples = sum(1 for line in fh if line.strip()) or None
            if samples is not None and bed.is_file():
                bed_size = bed.stat().st_size
                bytes_per_variant = (samples + 3) // 4
                if bed_size >= 3 and bytes_per_variant > 0 and (bed_size - 3) % bytes_per_variant == 0:
                    markers = ((bed_size - 3) // bytes_per_variant) or None
            if bim.is_file() and (
                markers is None or bim.stat().st_size <= _PREFLIGHT_TEXT_SCAN_MAX_BYTES
            ):
                with bim.open("rt", encoding="utf-8", errors="replace") as fh:
                    total = 0
                    for raw in fh:
                        line = raw.strip()
                        if not line:
                            continue
                        total += 1
                        chrom = line.split(None, 1)[0]
                        chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
                    markers = total or None
            chroms = len(chrom_counts) or None
            max_chrom = max(chrom_counts.values()) if chrom_counts else None
            return samples, markers, chroms, max_chrom
        if target_kind == "pfile":
            prefix = _strip_input_suffix(target, (".pgen", ".pvar.zst", ".pvar", ".psam"))
            pgen = Path(f"{prefix}.pgen")
            psam = Path(f"{prefix}.psam")
            pvar = Path(pvar_path_for_prefix(str(prefix)))
            samples = None
            markers = None
            chrom_counts = {}
            if pgen.is_file():
                samples, markers = _read_pgen_header_dimensions(pgen)
            if samples is None and psam.is_file():
                with open_text_auto(str(psam)) as fh:
                    samples = sum(1 for line in fh if line.strip() and not line.startswith("#")) or None
            if markers is None and samples is not None and pgen.is_file():
                # Mode 0x01 BED-compatible PGEN: derive marker count directly from payload size.
                with pgen.open("rb") as fh:
                    magic = fh.read(3)
                if magic == b"\x6c\x1b\x01":
                    pgen_size = pgen.stat().st_size
                    bytes_per_variant = (samples + 3) // 4
                    if pgen_size >= 3 and bytes_per_variant > 0 and (pgen_size - 3) % bytes_per_variant == 0:
                        markers = ((pgen_size - 3) // bytes_per_variant) or None
            # Only scan `.pvar` text in Python when small (<= 4 MiB) or when the binary
            # header did not yield a marker count; avoids multi-second single-core pre-scans
            # on multi-million-SNP panels.
            pvar_is_small_plain = (
                pvar.is_file()
                and not str(pvar).lower().endswith(".zst")
                and pvar.stat().st_size <= _PREFLIGHT_TEXT_SCAN_MAX_BYTES
            )
            if pvar.is_file() and (markers is None or pvar_is_small_plain):
                with open_text_auto(str(pvar)) as fh:
                    total = 0
                    for raw in fh:
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        total += 1
                        chrom = line.split(None, 1)[0]
                        chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
                    markers = total or None
            chroms = len(chrom_counts) or None
            max_chrom = max(chrom_counts.values()) if chrom_counts else None
            return samples, markers, chroms, max_chrom
    except Exception:
        return None, None, None, None
    return None, None, None, None


def _estimate_peak_rss_bytes(
    *,
    samples: int,
    max_chrom_markers: int,
    ploidy: str,
    threads: int,
    ref_samples: int | None = None,
    ref_max_chrom_markers: int | None = None,
    target_is_vcf: bool = False,
    ref_is_vcf: bool = False,
) -> int:
    """Estimate peak RSS (bytes) for streaming whole-chromosome IBD / ancestry."""
    n_samples = max(int(samples), 1)
    m_max = max(int(max_chrom_markers), 1)
    n_threads = max(int(threads), 1)
    n_ref = max(int(ref_samples), 0) if ref_samples else 0
    m_ref_max = (
        max(int(ref_max_chrom_markers), 1)
        if ref_max_chrom_markers is not None
        else m_max
    )

    # Peak occurs while encoding the largest chromosome:
    # VariantRow vec(s) + classify chunk scratch (het/PS events) + BitHaplotypeMatrix.
    row_bytes_per_marker = ((n_samples + 1) // 2) + 192
    if target_is_vcf:
        # VCF rows may carry optional per-sample PS tracks (`Option<PhaseSetId>` ~ 8 B/sample).
        row_bytes_per_marker += n_samples * 8
    target_rows_bytes = m_max * row_bytes_per_marker

    ref_rows_bytes = 0
    if n_ref > 0:
        ref_row_bytes_per_marker = ((n_ref + 1) // 2) + 192
        if ref_is_vcf:
            ref_row_bytes_per_marker += n_ref * 8
        ref_rows_bytes = m_ref_max * ref_row_bytes_per_marker

    is_inbred = str(ploidy).lower() == "inbred"
    haps_per_sample = 1 if is_inbred else 2
    total_haps = (n_samples + n_ref) * haps_per_sample
    words_per_marker = (total_haps + 63) // 64
    matrix_bytes = m_max * (2 * words_per_marker * 8 + 16)

    # Parallel classify_and_collect_hets_with chunk scratch + unphased het / PS event buffers.
    n_chunks = max((m_max + 2047) // 2048, 1)
    classify_base_bytes = n_chunks * (n_samples + n_ref) * 64
    het_event_bytes = (
        0
        if is_inbred
        else int(m_max * (n_samples + n_ref) * 0.05 * 4)
    )
    ps_event_bytes = (
        int(m_max * (n_samples + n_ref) * 0.05 * 12)
        if (target_is_vcf or ref_is_vcf) and not is_inbred
        else 0
    )
    classify_scratch_bytes = classify_base_bytes + het_event_bytes + ps_event_bytes

    pbwt_scratch_bytes = n_threads * total_haps * 32 + m_max * 32
    # Ancestry mode transposes `joint_matrix` into 64-bit haplotype-major words during Viterbi.
    ancestry_transpose_bytes = (
        total_haps * ((m_max + 63) // 64) * 16 if n_ref > 0 else 0
    )

    # Calibrate dynamic peak for allocator/reader overhead (similar to imp.py).
    dynamic_peak = max(
        target_rows_bytes + ref_rows_bytes + matrix_bytes + classify_scratch_bytes,
        matrix_bytes + ancestry_transpose_bytes + pbwt_scratch_bytes,
    )
    calibrated_dynamic = int(dynamic_peak * 1.55)
    return max(_process_rss_bytes(), 0) + calibrated_dynamic


def _emit_memory_preflight(
    logger: logging.Logger,
    *,
    chromosomes: int,
    estimated_peak_rss_bytes: int,
) -> None:
    logger.info(f"chromosomes                   {chromosomes:,}")
    logger.info(
        "estimated peak RSS            "
        f"{estimated_peak_rss_bytes / (1024.0**3):.1f} GiB"
    )


class _IbdProgressReporter:
    """Rich progress on TTY and state-change-only progress records elsewhere (aligned with imp.py)."""

    _STAGE_LABELS = {
        "prepare": "Preparing",
        "load": "Loading",
        "encode": "Encoding",
        "phase": "Phasing",
        "ibd": "IBD scan",
        "ancestry": "Ancestry",
        "chrom_done": "Finished chrom",
        "kinship": "Kinship CSR",
        "write": "Writing",
        "complete": "Finished",
    }

    def __init__(
        self,
        logger: logging.Logger,
        *,
        total: int | None = None,
        task_label: str = "IBD",
    ) -> None:
        self.logger = logger
        self._total = None if total is None or total <= 0 else int(total)
        self._task_label = str(task_label).strip() or "IBD"
        self._started = time.monotonic()
        self._progress = None
        self._task_id = None
        self._last_signature = None
        if stdout_is_tty() and rich_progress_available():
            self._progress = build_rich_progress(
                description_template="[green]{task.description:<16}",
                show_spinner=True,
                show_bar=True,
                show_percentage=self._total is not None,
                show_elapsed=True,
                show_remaining=self._total is not None,
                field_templates=[],
                bar_width=_IBD_PROGRESS_BAR_WIDTH,
                transient=True,
            )

    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._started)

    def _ensure_started(self) -> None:
        if self._progress is None or self._task_id is not None:
            return
        self._progress.start()
        self._task_id = self._progress.add_task(self._task_label, total=self._total)

    def __call__(
        self,
        stage: str,
        chrom: str,
        chrom_index: int,
        chrom_markers: int,
        raw_markers_done: int,
        analyzed_markers_done: int,
        segments_found: int,
    ) -> None:
        stage = str(stage)
        chrom = str(chrom)
        chrom_index = int(max(0, chrom_index))
        chrom_markers = int(max(0, chrom_markers))
        raw_markers_done = int(max(0, raw_markers_done))
        analyzed_markers_done = int(max(0, analyzed_markers_done))
        segments_found = int(max(0, segments_found))
        stage_label = self._STAGE_LABELS.get(stage, stage)
        desc = f"{stage_label} {chrom}".strip() if chrom else stage_label

        self._ensure_started()
        if self._progress is not None and self._task_id is not None:
            update_kwargs: dict[str, object] = {"description": desc[:16]}
            if self._total is not None:
                update_kwargs["completed"] = (
                    self._total
                    if stage == "complete"
                    else min(self._total, raw_markers_done)
                )
            self._progress.update(self._task_id, **update_kwargs)

        signature = (
            stage,
            chrom,
            chrom_index,
            chrom_markers,
            raw_markers_done,
            analyzed_markers_done,
            segments_found,
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature

        parts = [f"stage={stage_label}"]
        if chrom:
            parts.append(f"chromosome={chrom}")
        if chrom_markers and stage != "complete":
            parts.append(f"chrom_markers={chrom_markers:,}")
        if self._total is not None:
            done_val = min(self._total, raw_markers_done)
            pct = 100.0 * float(done_val) / float(self._total)
            parts.append(f"scanned={done_val:,}/{self._total:,} ({pct:.1f}%)")
        elif raw_markers_done:
            parts.append(f"scanned={raw_markers_done:,}")
        if analyzed_markers_done:
            parts.append(f"analyzed={analyzed_markers_done:,}")
        if segments_found:
            parts.append(f"segments={segments_found:,}")
        parts.append(f"elapsed={format_elapsed(self.elapsed_seconds())}")
        message = "  Progress             : " + " | ".join(parts)
        if self._progress is not None:
            _emit_to_file_handlers(self.logger, logging.INFO, message)
        else:
            self.logger.info(message)

    def close(self) -> None:
        if self._progress is not None:
            if self._task_id is not None:
                self._progress.stop()
            self._progress = None
            self._task_id = None


def main(argv: Sequence[str] | None = None) -> int:
    wall_started = time.monotonic()
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    if args.threads is None:
        args.threads = max(1, int(detect_effective_threads()))
    try:
        _validate_parameters(args)
        target_kind, target = _selected_target(args)
        reference = _resolve_reference_arg(args.reference)
        resolved_mode = _resolve_mode(args.mode, reference, args.ancestry_model)
        from ._common.outprefix import validate_output_prefix_arg

        args.out = validate_output_prefix_arg(str(args.out), auto_prefix="ibd")
    except ValueError as exc:
        parser.error(str(exc))

    output_prefix = _output_prefix(Path(args.out))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_path = f"{output_prefix}.log"
    logger: logging.Logger = setup_logging(log_path)
    log_path = get_log_path(logger) or log_path
    terminal_rich = bool(stdout_is_tty() and rich_progress_available())
    if terminal_rich:
        report_logger = _logger_handlers(
            logger, "janusx.ibd.file_only", file_handlers=True
        )
        terminal_logger = _logger_handlers(
            logger, "janusx.ibd.stream_only", file_handlers=False
        )
    else:
        report_logger = logger
        terminal_logger = logger

    target_display = str(target)
    progress_target_label = (
        target_display
        if len(target_display) <= 56
        else os.path.basename(target_display.rstrip("/\\"))
    )
    input_format = {
        "bfile": "PLINK BED",
        "pfile": "PLINK 2 PGEN",
        "vcf": "VCF",
        "hmp": "HapMap",
    }.get(target_kind, target_kind.upper())
    mode_label = (
        "Multi-parent ancestry" if resolved_mode == "ancestry" else "Pairwise IBD"
    )
    general_config_rows = [
        ("Genotype file", target_display),
        ("Input format", input_format),
        ("Reference", "None" if reference is None else str(reference)),
        ("Genetic map", "None" if args.map is None else str(Path(args.map).expanduser())),
        ("Mode", mode_label),
        ("Ploidy", str(args.ploidy)),
        ("Min IBD length", f"{float(args.min_output_cm):g} cM"),
        ("Write TSV", "Yes" if bool(args.write_tsv) else "No"),
        ("Threads", int(args.threads)),
    ]
    config_rows = [
        *general_config_rows,
        ("Output", f"{output_prefix}.jhibd"),
        ("Version", _janusx_version()),
    ]
    host = socket.gethostname()
    if terminal_rich:
        emit_cli_configuration(
            terminal_logger,
            app_title="JanusX - IBD",
            config_title="IBD CONFIG",
            host=host,
            sections=[("General", general_config_rows)],
            footer_rows=[("Output prefix", str(output_prefix))],
            line_max_chars=_TERMINAL_SECTION_WIDTH,
        )
    _report_banner(report_logger, "JanusX - IBD Pipeline Analysis")
    _report_block(report_logger, "System Context")
    _report_kv(report_logger, "Host", host)
    _report_kv(report_logger, "Version", _janusx_version())
    _report_kv(report_logger, "Threads", int(args.threads))
    report_logger.info("")
    _report_block(report_logger, "Configuration")
    for key, value in config_rows:
        _report_kv(report_logger, key, value)
    report_logger.info("")
    _report_block(report_logger, "Command")
    report_logger.info(f"  {_invocation_command()}")
    report_logger.info("")
    _report_block(report_logger, "Stage Progress")

    options = {
        "target": str(target),
        "reference": None if reference is None else str(reference),
        "map": None if args.map is None else str(Path(args.map).expanduser()),
        "output": str(output_prefix),
        "threads": int(args.threads),
        "ploidy_mode": str(args.ploidy),
        "inbred_het_threshold": float(args.inbred_het_threshold),
        "min_seed_cm": float(args.min_seed_cm),
        "min_output_cm": float(args.min_output_cm),
        "max_gap_cm": float(args.max_gap_cm),
        "max_mismatch_rate": float(args.max_mismatch_rate),
        "min_markers": int(args.min_markers),
        "ancestry_model": str(args.ancestry_model),
        "ancestry_switch_penalty": float(args.ancestry_switch_penalty),
        "ancestry_top_donors": int(args.ancestry_top_donors),
        "ancestry_step_cm": float(args.ancestry_step_cm),
        "ancestry_step_max_markers": int(args.ancestry_step_max_markers),
        "cm_per_mb": float(args.cm_per_mb),
        "write_tsv": bool(args.write_tsv),
    }
    if str(args.ancestry_model).strip().lower() == "mott":
        options.update(
            {
                "mott_generations": float(args.mott_generations),
                "mott_parent_match_prob": float(args.mott_parent_match_prob),
                "mott_map_scale_cm": float(args.mott_map_scale_cm),
                "mott_unknown_fold": float(args.mott_unknown_fold),
                "mott_max_markers_per_chrom": int(args.mott_max_markers_per_chrom),
                "mott_write_posterior": bool(args.mott_write_posterior),
                "mott_write_diagnostics": bool(args.mott_write_diagnostics),
                "mott_ambiguity_delta": float(args.mott_ambiguity_delta),
            }
        )

    inspected_samples: int | None = None
    progress_total: int | None = None
    inspected_chroms: int | None = None
    max_chrom_markers: int | None = None
    estimated_peak_rss_bytes: int = 0
    count_task = None
    if terminal_rich:
        count_task = CliStatus(
            f"Loading genotype from {progress_target_label}...",
            enabled=True,
            use_process=True,
        )
        count_task.__enter__()
    try:
        (
            inspected_samples,
            progress_total,
            inspected_chroms,
            max_chrom_markers,
        ) = _inspect_target_dimensions(target_kind, target)
        if progress_total is not None:
            report_logger.info(
                f"  Progress denominator : {progress_total:,} target markers"
            )
        if (
            inspected_samples is not None
            and max_chrom_markers is not None
            and inspected_chroms is not None
        ):
            ref_samples, ref_max_chrom_markers = _inspect_reference_dimensions(reference)
            ref_lower = "" if reference is None else str(reference).lower()
            ref_is_vcf = ref_lower.endswith((".vcf", ".vcf.gz"))
            # Only emit peak-RSS preflight when BOTH reference sample count and
            # reference max-chromosome marker count (if -ref is set) are known,
            # avoiding silent underestimates on unindexed VCF/HapMap references.
            if reference is None or (
                ref_samples is not None and ref_max_chrom_markers is not None
            ):
                estimated_peak_rss_bytes = _estimate_peak_rss_bytes(
                    samples=inspected_samples,
                    max_chrom_markers=max_chrom_markers,
                    ploidy=str(args.ploidy),
                    threads=int(args.threads),
                    ref_samples=ref_samples,
                    ref_max_chrom_markers=ref_max_chrom_markers,
                    target_is_vcf=(target_kind == "vcf"),
                    ref_is_vcf=ref_is_vcf,
                )
        if count_task is not None:
            if inspected_samples is not None and progress_total is not None:
                count_task.complete(
                    f"Loading genotype from {progress_target_label} "
                    f"(n={inspected_samples:,}, nSNP={progress_total:,})"
                )
            else:
                count_task.complete(f"Loading genotype from {progress_target_label}")
    finally:
        if count_task is not None:
            count_task.close(show_done=False)
    if estimated_peak_rss_bytes and inspected_chroms is not None:
        _emit_memory_preflight(
            report_logger,
            chromosomes=inspected_chroms,
            estimated_peak_rss_bytes=estimated_peak_rss_bytes,
        )
        if terminal_logger is not report_logger:
            _emit_memory_preflight(
                terminal_logger,
                chromosomes=inspected_chroms,
                estimated_peak_rss_bytes=estimated_peak_rss_bytes,
            )

    reporter = _IbdProgressReporter(
        report_logger if terminal_rich else logger,
        total=progress_total,
        task_label=mode_label,
    )
    try:
        summary = json.loads(jxrs.haploid_ibd_run(json.dumps(options), reporter))
    except Exception as exc:
        reporter.close()
        logger.error(str(exc))
        return 1
    processing_elapsed = reporter.elapsed_seconds()
    reporter.close()

    generated_outputs = [Path(str(summary["output_path"]))]
    if summary.get("tsv_path"):
        generated_outputs.append(Path(str(summary["tsv_path"])))
    if summary.get("posterior_path"):
        generated_outputs.append(Path(str(summary["posterior_path"])))
    if summary.get("mott_diagnostics_path"):
        generated_outputs.append(Path(str(summary["mott_diagnostics_path"])))

    report_logger.info("")
    _report_block(report_logger, "Outputs Generated")
    for path in generated_outputs:
        _report_kv(report_logger, "Output", path)
    _report_kv(report_logger, "Log", log_path)
    report_logger.info("")
    _report_block(report_logger, "Advanced Parameters")
    _report_kv(report_logger, "Min seed cM", float(args.min_seed_cm))
    _report_kv(report_logger, "Min output cM", float(args.min_output_cm))
    _report_kv(report_logger, "Max gap cM", float(args.max_gap_cm))
    _report_kv(report_logger, "Max mismatch rate", float(args.max_mismatch_rate))
    _report_kv(report_logger, "Min markers", int(args.min_markers))
    _report_kv(report_logger, "Inbred het threshold", float(args.inbred_het_threshold))
    _report_kv(report_logger, "cM per Mb", float(args.cm_per_mb))
    if resolved_mode == "ancestry":
        _report_kv(report_logger, "Ancestry model", str(args.ancestry_model))
        if str(args.ancestry_model) == "mott":
            _report_kv(report_logger, "Mott generations", float(args.mott_generations))
            _report_kv(
                report_logger,
                "Mott parent match",
                float(args.mott_parent_match_prob),
            )
            _report_kv(report_logger, "Mott map scale cM", float(args.mott_map_scale_cm))
            _report_kv(report_logger, "Mott unknown fold", float(args.mott_unknown_fold))
            _report_kv(
                report_logger,
                "Mott max markers",
                int(args.mott_max_markers_per_chrom),
            )
            _report_kv(
                report_logger,
                "Mott posterior TSV",
                "Yes" if bool(args.mott_write_posterior) else "No",
            )
            _report_kv(
                report_logger,
                "Mott diagnostics",
                "Yes" if bool(args.mott_write_diagnostics) else "No",
            )
            _report_kv(
                report_logger,
                "Mott ambiguity delta",
                float(args.mott_ambiguity_delta),
            )
        else:
            _report_kv(report_logger, "Ancestry step cM", float(args.ancestry_step_cm))
            if int(args.ancestry_step_max_markers) > 0:
                _report_kv(
                    report_logger,
                    "Step max markers",
                    int(args.ancestry_step_max_markers),
                )
            _report_kv(report_logger, "Switch penalty", float(args.ancestry_switch_penalty))
            _report_kv(report_logger, "Top donors", int(args.ancestry_top_donors))
    report_logger.info("")
    _report_block(report_logger, "Execution Summary")
    _report_kv(report_logger, "Status", "done")
    _report_kv(report_logger, "Mode", str(summary["mode"]))
    _report_kv(report_logger, "Chromosomes", f"{int(summary['chromosomes']):,}")
    _report_kv(report_logger, "Markers analyzed", f"{int(summary['markers_processed']):,}")
    if resolved_mode == "ancestry":
        exact_n = int(summary.get("aligned_exact_markers", 0))
        swapped_n = int(summary.get("aligned_swapped_markers", 0))
        incompat_n = int(summary.get("aligned_incompatible_markers", 0))
        if exact_n or swapped_n or incompat_n:
            _report_kv(
                report_logger,
                "Aligned markers",
                f"exact={exact_n:,}, swapped={swapped_n:,}, incompatible={incompat_n:,}",
            )
        if str(args.ancestry_model) == "mott":
            info_n = int(summary.get("mott_informative_markers", 0))
            sel_n = int(summary.get("mott_selected_markers", 0))
            _report_kv(
                report_logger,
                "Mott marker pool",
                f"informative={info_n:,}, selected={sel_n:,}",
            )
    _report_kv(
        report_logger,
        "Target samples",
        f"{int(summary['target_samples']):,} "
        f"(inbred={int(summary['inbred_samples']):,}, diploid={int(summary['diploid_samples']):,})",
    )
    if int(summary["reference_samples"]) > 0:
        _report_kv(
            report_logger,
            "Reference samples",
            f"{int(summary['reference_samples']):,}",
        )
    _report_kv(report_logger, "Segments", f"{int(summary['segments_written']):,}")
    if int(summary["sparse_bins_written"]) > 0:
        _report_kv(
            report_logger, "Sparse bins", f"{int(summary['sparse_bins_written']):,}"
        )
    _report_kv(report_logger, "Sparse nnz", f"{int(summary['sparse_nnz']):,}")
    elapsed_seconds = max(time.monotonic() - wall_started, 0.0)
    _report_kv(report_logger, "Total wall time", f"{elapsed_seconds:.2f}s")
    report_logger.info(f"IBD run status: done; log: {log_path}")

    if terminal_rich:
        log_success(
            terminal_logger,
            f"{mode_label} ...Finished "
            f"({int(summary['segments_written']):,} segments, "
            f"{int(summary['markers_processed']):,} markers) "
            f"[{format_elapsed(processing_elapsed)}]",
            force_color=True,
        )
        saved_lines = "\n".join(f"  {path}" for path in generated_outputs)
        terminal_logger.info(f"Saved IBD outputs:\n{saved_lines}")
        lt = time.localtime()
        endinfo = (
            f"\nFinished IBD calculation. Total wall time: {elapsed_seconds:.2f} seconds\n"
            f"{lt.tm_year}-{lt.tm_mon:02d}-{lt.tm_mday:02d} "
            f"{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}"
        )
        _emit_to_file_handlers(report_logger, logging.INFO, endinfo)
        log_success(terminal_logger, endinfo)
    else:
        log_success(
            report_logger,
            f"Finished {mode_label}: {int(summary['segments_written']):,} segments "
            f"across {int(summary['chromosomes']):,} chromosomes "
            f"({int(summary['markers_processed']):,} markers, "
            f"{int(summary['target_samples']):,} samples).",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
