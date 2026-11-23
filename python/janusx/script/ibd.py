"""Whole-chromosome PBWT IBD segmentation and multi-parent ancestry painting (`jx ibd`)."""

from __future__ import annotations

import argparse
import json
import logging
import math
import shlex
import socket
import sys
import time
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from janusx import janusx as jxrs
from janusx._plink import pvar_path_for_prefix
from ._common.cli_core import (
    CliArgumentParser,
    cli_help_formatter,
    minimal_help_epilog,
)
from ._common.config_render import emit_cli_configuration
from ._common.log import get_log_path, setup_logging
from ._common.progress import (
    log_success,
    rich_progress_available,
    stdout_is_tty,
)
from ._common.threads import detect_effective_threads


_REPORT_RULE = "=" * 72
_TERMINAL_SECTION_WIDTH = 60


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
    logger.info(f"  {str(key):<24} : {value}")


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


def _resolve_mode(requested_mode: str, reference: Path | None) -> str:
    mode = str(requested_mode).strip().lower()
    if mode == "pairwise":
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
    if (
        not math.isfinite(args.ancestry_switch_penalty)
        or args.ancestry_switch_penalty <= 0.0
    ):
        raise ValueError("--ancestry-switch-penalty must be a finite positive number")
    if args.ancestry_top_donors < 1:
        raise ValueError("--ancestry-top-donors must be a positive integer")


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
        resolved_mode = _resolve_mode(args.mode, reference)
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
        ("Genotype file", str(target)),
        ("Input format", input_format),
        ("Reference", "None" if reference is None else str(reference)),
        ("Mode", mode_label),
        ("Ploidy", str(args.ploidy)),
        ("Min IBD length", f"{float(args.min_output_cm):g} cM"),
        ("Write TSV", "Yes" if bool(args.write_tsv) else "No"),
        ("Threads", int(args.threads)),
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
    _report_banner(report_logger, "JanusX - Whole-Chromosome PBWT IBD Analysis")
    _report_block(report_logger, "System Context")
    _report_kv(report_logger, "Host", host)
    _report_kv(report_logger, "Version", _janusx_version())
    _report_kv(report_logger, "Threads", int(args.threads))
    report_logger.info("")
    _report_block(report_logger, "Configuration")
    for key, value in general_config_rows:
        _report_kv(report_logger, key, value)
    _report_kv(report_logger, "Output archive", f"{output_prefix}.jhibd")
    report_logger.info("")
    _report_block(report_logger, "Command")
    report_logger.info(f"  {_invocation_command()}")
    report_logger.info("")

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
        "ancestry_switch_penalty": float(args.ancestry_switch_penalty),
        "ancestry_top_donors": int(args.ancestry_top_donors),
        "ancestry_step_cm": float(args.ancestry_step_cm),
        "cm_per_mb": float(args.cm_per_mb),
        "write_tsv": bool(args.write_tsv),
    }
    try:
        summary = json.loads(jxrs.haploid_ibd_run(json.dumps(options)))
    except Exception as exc:
        logger.error(str(exc))
        return 1

    elapsed = max(time.monotonic() - wall_started, 0.0)
    _report_block(report_logger, "Execution Summary")
    _report_kv(report_logger, "Mode", str(summary["mode"]))
    _report_kv(report_logger, "Chromosomes", f"{int(summary['chromosomes']):,}")
    _report_kv(report_logger, "Markers", f"{int(summary['markers_processed']):,}")
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
    _report_kv(report_logger, "Sparse archive", str(summary["output_path"]))
    if summary.get("tsv_path"):
        _report_kv(report_logger, "TSV export", str(summary["tsv_path"]))
    _report_kv(report_logger, "Log", log_path)
    _report_kv(report_logger, "Wall time", f"{elapsed:.2f} s")

    completion_msg = (
        f"Finished {mode_label}: {int(summary['segments_written']):,} segments "
        f"across {int(summary['chromosomes']):,} chromosomes "
        f"({int(summary['markers_processed']):,} markers) -> {summary['output_path']} "
        f"[{elapsed:.2f}s]"
    )
    log_success(terminal_logger, completion_msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
