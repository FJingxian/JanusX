"""Stream genotype windows through the native Rust Beagle-style imp module."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import os
import shlex
import socket
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

from janusx import janusx as jxrs
from janusx._plink import open_text_auto, pvar_path_for_prefix
from ._common.cli_core import (
    CliArgumentParser,
    cli_help_formatter,
    minimal_help_epilog,
)
from ._common.cli_ranges import (
    format_normalized_range,
    normalize_chromosome_key,
    parse_cli_range,
)
from ._common.config_render import emit_cli_configuration
from ._common.grmio import genotype_cache_prefix
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
_IMP_PROGRESS_BAR_WIDTH = 12


def _report_banner(logger: logging.Logger, title: str) -> None:
    logger.info(_REPORT_RULE)
    logger.info(str(title))
    logger.info(_REPORT_RULE)
    logger.info("")


def _report_block(logger: logging.Logger, title: str) -> None:
    logger.info(f"[ {title} ]")


def _report_kv(logger: logging.Logger, key: str, value: object) -> None:
    logger.info(f"  {str(key):<20} : {value}")


def _format_mode(mode: str) -> str:
    return {
        "phase": "Phasing",
        "phasing-imputation": "Phasing&Imputing",
        "reference-imputation": "Reference imputation",
    }.get(str(mode), str(mode).replace("-", " ").title())


def _open_text_input(path: Path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    if str(path).lower().endswith(".zst"):
        return open_text_auto(str(path))
    return path.open(mode="rt", encoding="utf-8", newline="")


def _marker_in_region(
    chrom: object,
    position: object,
    region,
) -> bool:
    if region is None:
        return True
    if normalize_chromosome_key(chrom) != region.chrom_key:
        return False
    parsed_position = int(position)
    return (
        (region.start is None or parsed_position >= int(region.start))
        and (region.end is None or parsed_position <= int(region.end))
    )


def _count_marker_rows(
    path: Path,
    *,
    chrom_index: int,
    position_index: int,
    delimiter: str | None,
    region,
    skip_first_data_row: bool = False,
) -> int:
    count = 0
    data_row_seen = False
    with _open_text_input(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if skip_first_data_row and not data_row_seen:
                data_row_seen = True
                continue
            data_row_seen = True
            if region is None:
                count += 1
                continue
            fields = line.split(delimiter) if delimiter is not None else line.split()
            if len(fields) <= max(chrom_index, position_index):
                raise ValueError(f"{path}: malformed marker row")
            if _marker_in_region(
                fields[chrom_index],
                fields[position_index],
                region,
            ):
                count += 1
    return count


def _count_sample_rows(path: Path) -> int:
    count = 0
    with _open_text_input(path) as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


def _inspect_target_dimensions(
    target_kind: str,
    target: Path,
    region,
) -> tuple[int, int]:
    if target_kind == "bfile":
        samples = _count_sample_rows(Path(f"{target}.fam"))
        markers = _count_marker_rows(
            Path(f"{target}.bim"),
            chrom_index=0,
            position_index=3,
            delimiter=None,
            region=region,
        )
        return samples, markers
    if target_kind == "pfile":
        samples = _count_sample_rows(Path(f"{target}.psam"))
        markers = _count_marker_rows(
            Path(pvar_path_for_prefix(str(target))),
            chrom_index=0,
            position_index=1,
            delimiter=None,
            region=region,
        )
        return samples, markers
    if target_kind == "vcf":
        samples = 0
        markers = 0
        with _open_text_input(target) as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if line.startswith("#CHROM"):
                        samples = max(0, len(line.split("\t")) - 9)
                    continue
                if region is not None:
                    fields = line.split("\t")
                    if len(fields) < 2:
                        raise ValueError(f"{target}: malformed VCF marker row")
                    if not _marker_in_region(fields[0], fields[1], region):
                        continue
                markers += 1
        return samples, markers
    if target_kind == "hmp":
        samples = 0
        markers = 0
        header_seen = False
        with _open_text_input(target) as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if not header_seen:
                    samples = max(0, len(fields) - 11)
                    header_seen = True
                    continue
                if len(fields) < 4:
                    raise ValueError(f"{target}: malformed HapMap marker row")
                if _marker_in_region(fields[2], fields[3], region):
                    markers += 1
        return samples, markers
    raise ValueError(f"unsupported target kind for progress inspection: {target_kind}")


def _janusx_version() -> str:
    try:
        return version("janusx")
    except PackageNotFoundError:
        return "unknown"


def _invocation_command() -> str:
    return shlex.join([str(value) for value in sys.argv])


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


def _output_destination(prefix: Path, output_format: str) -> Path:
    prefix = Path(prefix).expanduser()
    lower = str(prefix).lower()
    if output_format == "vcf":
        if lower.endswith(".vcf.gz") or lower.endswith(".vcf"):
            return prefix
        return Path(f"{prefix}.vcf.gz")
    for suffix in (".pgen", ".pvar.zst", ".pvar", ".psam"):
        if lower.endswith(suffix):
            return Path(str(prefix)[: -len(suffix)])
    return prefix


def _output_prefix(value: Path) -> Path:
    """Return the user-facing run prefix without a genotype output suffix."""
    path = Path(value).expanduser()
    lower = str(path).lower()
    for suffix in (".vcf.gz", ".vcf", ".pgen", ".pvar.zst", ".pvar", ".psam"):
        if lower.endswith(suffix):
            return Path(str(path)[: -len(suffix)])
    return path


def _resolve_reference_arg(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    lower = str(path).lower()
    if lower.endswith(".vcf") or lower.endswith(".vcf.gz"):
        if not path.is_file():
            raise ValueError(f"reference VCF not found: {path}")
        return path
    for suffix in (".pgen", ".pvar.zst", ".pvar", ".psam"):
        if lower.endswith(suffix):
            path = Path(str(path)[: -len(suffix)])
            break
    members = [
        Path(f"{path}.pgen"),
        Path(pvar_path_for_prefix(str(path))),
        Path(f"{path}.psam"),
    ]
    if not all(member.is_file() for member in members):
        raise ValueError(
            "reference must be a complete pfile prefix with .pgen/.pvar/.psam "
            f"or a VCF file: {value}"
        )
    return path


def _hmp_cache_prefix(hmp: Path, cache_dir: str | Path | None = None) -> Path:
    hmp = Path(hmp).expanduser()
    stat = hmp.stat()
    resolved = str(hmp.resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    base = Path(
        genotype_cache_prefix(
            str(hmp),
            cache_dir=None if cache_dir is None else str(cache_dir),
        )
    )
    version = f"v{int(stat.st_size)}-{int(stat.st_mtime_ns)}"
    return Path(f"{base}.{digest}.{version}.imp")


def _strip_input_suffix(path: Path, suffixes: tuple[str, ...]) -> Path:
    lower = str(path).lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            return Path(str(path)[: -len(suffix)])
    return path


def _selected_target(args: argparse.Namespace) -> tuple[str, Path]:
    for kind in ("bfile", "pfile", "vcf", "hmp"):
        value = getattr(args, kind)
        if value is not None:
            path = Path(value).expanduser()
            if kind == "bfile":
                return kind, _strip_input_suffix(path, (".bed", ".bim", ".fam"))
            if kind == "pfile":
                return kind, _strip_input_suffix(
                    path, (".pgen", ".pvar.zst", ".pvar", ".psam")
                )
            if not path.is_file():
                raise ValueError(f"{kind} input not found: {path}")
            lower = str(path).lower()
            if kind == "vcf" and not (
                lower.endswith(".vcf") or lower.endswith(".vcf.gz")
            ):
                raise ValueError(f"-vcf expects a .vcf or .vcf.gz filename: {path}")
            if kind == "hmp" and not (
                lower.endswith(".hmp") or lower.endswith(".hmp.gz")
            ):
                raise ValueError(f"-hmp expects a .hmp or .hmp.gz filename: {path}")
            return kind, path
    raise ValueError("one target input is required")


def _pfile_cache_complete(prefix: Path) -> bool:
    return (
        Path(f"{prefix}.pgen").is_file()
        and Path(pvar_path_for_prefix(str(prefix))).is_file()
        and Path(f"{prefix}.psam").is_file()
    )


class _SingleUseStoreAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, "_bimrange_seen", False):
            raise argparse.ArgumentError(
                self,
                "-bimrange/--bimrange may only be specified once",
            )
        setattr(namespace, self.dest, values)
        setattr(namespace, "_bimrange_seen", True)


def _diagnostic_site(value: str | None) -> dict[str, object] | None:
    if value is None:
        return None
    try:
        chrom, position, sample = value.split(":", 2)
        parsed_position = int(position)
    except (ValueError, TypeError) as exc:
        raise ValueError("--trace-site expects CHROM:POS:SAMPLE") from exc
    if not chrom or not sample or parsed_position <= 0:
        raise ValueError("--trace-site expects nonempty CHROM/SAMPLE and a positive POS")
    return {"chrom": chrom, "position": parsed_position, "sample": sample}


def _validate_modes(
    *,
    reference: str | Path | None,
    only_phase: bool,
    ap: bool,
    gp: bool,
    output_format: str,
) -> str:
    if only_phase:
        if ap or gp:
            raise ValueError("--ap and --gp require reference imputation and cannot be used with -only-phase")
        return "phase"
    if (ap or gp) and reference is None:
        raise ValueError("--ap and --gp require -ref")
    if (ap or gp) and str(output_format).lower() != "vcf":
        raise ValueError("--ap and --gp require VCF output")
    if reference is not None:
        return "reference-imputation"
    return "phasing-imputation"


class _ImpProgressReporter:
    """Rich progress on TTY and state-change-only progress records elsewhere."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        total: int | None = None,
        task_label: str = "IMP",
        imputation_mode: bool = False,
    ) -> None:
        self.logger = logger
        self._total = None if total is None else int(max(0, total))
        self._task_label = str(task_label).strip() or "IMP"
        self._imputation_mode = bool(imputation_mode)
        self._started = time.monotonic()
        self._progress = None
        self._task_id = None
        self._observed_phasing = False
        self._last_progress_signature = None
        if stdout_is_tty() and rich_progress_available():
            self._progress = build_rich_progress(
                description_template="[green]{task.description:<16}",
                show_spinner=True,
                show_bar=True,
                show_percentage=self._total is not None,
                show_elapsed=True,
                show_remaining=self._total is not None,
                field_templates=[],
                bar_width=_IMP_PROGRESS_BAR_WIDTH,
                transient=True,
            )

    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._started)

    def _start_progress(self) -> None:
        if self._progress is None or self._task_id is not None:
            return
        self._progress.start()
        self._task_id = self._progress.add_task(
            self._task_label,
            total=self._total,
        )

    def _log_progress_change(
        self,
        *,
        stage: str,
        stage_label: str,
        chrom: str,
        window_index: int,
        markers: int,
        variants_written: int,
        samples: int,
    ) -> None:
        if stage not in {
            "prepare",
            "direct",
            "phase",
            "impute",
            "window_done",
            "complete",
        }:
            return
        completed = (
            self._total
            if stage == "complete" and self._total is not None
            else min(self._total, variants_written)
            if self._total is not None
            else variants_written
        )
        signature = (
            stage_label,
            chrom,
            window_index,
            markers,
            completed,
            samples,
        )
        if signature == self._last_progress_signature:
            return
        self._last_progress_signature = signature
        parts = [f"stage={stage_label}"]
        if chrom:
            parts.append(f"chromosome={chrom}")
        if stage not in {"prepare", "complete"}:
            parts.append(f"window={window_index + 1:,}")
        if markers and stage != "complete":
            parts.append(f"window_markers={markers:,}")
        if self._total is not None:
            percent = 100.0 * float(completed) / float(self._total)
            parts.append(
                f"completed={int(completed):,}/{self._total:,} "
                f"({percent:.1f}%)"
            )
        elif completed:
            parts.append(f"completed={int(completed):,}")
        if samples and stage in {"prepare", "complete"}:
            parts.append(f"samples={samples:,}")
        message = "  Progress             : " + " | ".join(parts)
        if self._progress is not None:
            _emit_to_file_handlers(self.logger, logging.INFO, message)
        else:
            self.logger.info(message)

    def __call__(
        self,
        stage: str,
        chrom: str,
        window_index: int,
        markers: int,
        variants_written: int,
        samples: int,
    ) -> None:
        stage = str(stage)
        chrom = str(chrom)
        window_index = int(max(0, window_index))
        markers = int(max(0, markers))
        variants_written = int(max(0, variants_written))
        samples = int(max(0, samples))
        if stage in {"cache", "cache_done"}:
            stage_label = "Caching"
        elif stage == "complete":
            stage_label = "Finished"
        elif stage == "phase":
            self._observed_phasing = True
            stage_label = (
                "Phasing&Imputing"
                if self._imputation_mode
                else self._task_label
            )
        elif self._imputation_mode and stage in {
            "direct",
            "impute",
            "write",
            "window_done",
        }:
            stage_label = (
                "Phasing&Imputing"
                if self._observed_phasing
                else "Imputing"
            )
        else:
            stage_label = self._task_label
        self._start_progress()
        if self._progress is not None and self._task_id is not None:
            update_fields = {
                "description": stage_label,
            }
            if self._total is not None:
                cache_stage = stage in {"prepare", "cache", "cache_done"}
                update_fields["completed"] = (
                    self._total
                    if stage == "complete"
                    else (
                        0
                        if cache_stage
                        else min(self._total, variants_written)
                    )
                )
            self._progress.update(self._task_id, **update_fields)
        self._log_progress_change(
            stage=stage,
            stage_label=stage_label,
            chrom=chrom,
            window_index=window_index,
            markers=markers,
            variants_written=variants_written,
            samples=samples,
        )

    def close(self) -> None:
        if self._progress is not None:
            if self._task_id is not None:
                self._progress.stop()
            self._progress = None
            self._task_id = None


def build_parser(argv: Sequence[str] | None = None) -> CliArgumentParser:
    argv_tokens = list(sys.argv[1:] if argv is None else argv)
    show_dev_help = "-dev" in argv_tokens or "--dev" in argv_tokens

    def help_text(text: str, *, advanced: bool = False):
        return (str(text), argparse.SUPPRESS)[bool(advanced and not show_dev_help)]

    parser = CliArgumentParser(
        prog="jx imp",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx imp -vcf target.vcf.gz -o phased -fmt vcf",
                "jx imp -pfile target -ref reference -fmt pgen -o imputed",
                "jx imp -hmp target.hmp.gz -fmt pgen -o phased",
                "jx imp -bfile target -ref reference.vcf.gz -o imputed",
                "jx imp -h -dev",
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
        help=(
            "Input genotype file in HMP format (.hmp or .hmp.gz). "
            "HMP is converted to a reusable PFILE cache before processing."
        ),
    )

    reference = parser.add_argument_group("Reference Arguments")
    reference.add_argument(
        "-ref",
        "--reference",
        help=(
            "Fully phased reference as a VCF filename or PGEN prefix. "
            "The format is detected automatically."
        ),
    )
    reference.add_argument(
        "-map",
        "--map",
        help="PLINK genetic map with CHR, SNP, CM, and BP columns.",
    )
    reference.add_argument(
        "-bimrange",
        "--bimrange",
        action=_SingleUseStoreAction,
        metavar="CHR[:START:END]",
        help=(
            "Restrict output to one chromosome or inclusive physical interval. "
            "Use bp or Mb suffixes when units would be ambiguous."
        ),
    )

    model = parser.add_argument_group("Phasing / Imputation Arguments")
    model.add_argument(
        "-window",
        type=float,
        default=40.0,
        help=help_text("Window length in cM (default: %(default)s).", advanced=True),
    )
    model.add_argument(
        "-overlap",
        type=float,
        default=2.0,
        help=help_text("Window overlap in cM (default: %(default)s).", advanced=True),
    )
    model.add_argument(
        "-window-markers",
        type=int,
        default=4_000_000,
        help=help_text(
            "Maximum target markers per window (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "-phase-states",
        type=int,
        default=280,
        help=help_text(
            "Number of phasing HMM states (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "-imp-states",
        type=int,
        default=1_600,
        help=help_text(
            (
                "Maximum imputation states. Adaptive sizing uses fewer states when "
                "the reference panel is smaller (default: %(default)s)."
            ),
            advanced=True,
        ),
    )
    adaptive = model.add_mutually_exclusive_group()
    adaptive.add_argument(
        "--adaptive-states",
        dest="adaptive_states",
        action="store_true",
        help=help_text(
            "Scale imputation states to the available reference panel.",
            advanced=True,
        ),
    )
    adaptive.add_argument(
        "--no-adaptive-states",
        dest="adaptive_states",
        action="store_false",
        help=help_text("Use the fixed -imp-states value.", advanced=True),
    )
    parser.set_defaults(adaptive_states=True)
    model.add_argument(
        "-burnin",
        type=int,
        default=3,
        help=help_text(
            "Number of EM burn-in rounds per window (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "-iterations",
        type=int,
        default=12,
        help=help_text(
            "Number of EM parameter-update rounds per window (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "-initial-lr",
        type=float,
        default=100_000.0,
        help=help_text(
            "Initial phasing likelihood-ratio threshold (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "-ne",
        type=int,
        default=100_000,
        help=help_text(
            (
                "Effective population size used by phasing transitions "
                "(default: %(default)s)."
            ),
            advanced=True,
        ),
    )
    model.add_argument(
        "-err",
        type=float,
        default=None,
        help=help_text("Optional allele error-rate override.", advanced=True),
    )
    em = model.add_mutually_exclusive_group()
    em.add_argument(
        "--em",
        dest="em",
        action="store_true",
        help=help_text(
            "Enable Beagle-style EM initialization and burn-in updates (default).",
            advanced=True,
        ),
    )
    em.add_argument(
        "--no-em",
        dest="em",
        action="store_false",
        help=help_text("Disable EM parameter updates.", advanced=True),
    )
    parser.set_defaults(em=True)
    model.add_argument(
        "-only-phase",
        "--only-phase",
        dest="only_phase",
        action="store_true",
        help="Phase target genotypes without reference-marker imputation; -ref may still assist phasing.",
    )
    parser.set_defaults(only_phase=False)
    model.add_argument(
        "-imp-segment",
        type=float,
        default=6.0,
        help=help_text(
            (
                "Genetic length in cM of each local PBWT donor segment "
                "(default: %(default)s)."
            ),
            advanced=True,
        ),
    )
    model.add_argument(
        "-imp-step",
        type=float,
        default=0.1,
        help=help_text(
            (
                "Beagle compatibility value (default: %(default)s). "
                "Custom matching steps are not implemented."
            ),
            advanced=True,
        ),
    )
    model.add_argument(
        "-imp-nsteps",
        type=int,
        default=7,
        help=help_text(
            (
                "Beagle compatibility value (default: %(default)s). "
                "Custom matching steps are not implemented."
            ),
            advanced=True,
        ),
    )
    model.add_argument(
        "-cluster",
        type=float,
        default=0.005,
        help=help_text(
            "Maximum genetic distance (cM) within an HMM marker cluster "
            "(default: %(default)s; use 0 to disable clustering).",
            advanced=True,
        ),
    )
    model.add_argument(
        "--assume-global-phase",
        dest="assume_global_phase",
        action="store_true",
        help=help_text(
            (
                "Allow a fully phased PGEN target to bypass phasing when global "
                "phase continuity is known externally."
            ),
            advanced=True,
        ),
    )
    parser.set_defaults(assume_global_phase=False)

    output = parser.add_argument_group("Output Arguments")
    output.add_argument(
        "-o",
        "--out",
        required=True,
        help=(
            "Output prefix. VCF output appends .vcf.gz; PGEN output creates "
            ".pgen, .pvar, and .psam."
        ),
    )
    output.add_argument(
        "-fmt",
        "--format",
        choices=("vcf", "pgen"),
        default="vcf",
        help="Output genotype format (default: %(default)s).",
    )
    output.add_argument(
        "--ap",
        action="store_true",
        help=help_text(
            "Write AP1/AP2 haplotype probabilities when imputing.",
            advanced=True,
        ),
    )
    output.add_argument(
        "--gp",
        action="store_true",
        help=help_text(
            "Write GP genotype probabilities when imputing.",
            advanced=True,
        ),
    )

    runtime = parser.add_argument_group("Runtime Arguments")
    runtime.add_argument(
        "-t",
        "--thread",
        "--threads",
        dest="threads",
        type=int,
        default=None,
        help="Number of CPU threads (default: auto-detected).",
    )
    runtime.add_argument(
        "-seed",
        type=int,
        default=-99_999,
        help=help_text("Random seed (default: %(default)s).", advanced=True),
    )
    diagnostics = parser.add_argument_group("Diagnostics")
    diagnostics.add_argument(
        "--trace-site",
        default=None,
        metavar="CHROM:POS:SAMPLE",
        help=help_text(
            (
                "Write per-round PhaseBaum forward/backward state vectors for one "
                "target site."
            ),
            advanced=True,
        ),
    )
    return parser


def main() -> int:
    wall_started = time.monotonic()
    parser = build_parser()
    args = parser.parse_args()
    if args.threads is None:
        args.threads = max(1, int(detect_effective_threads()))
    if args.threads <= 0:
        parser.error("-t/--threads must be positive")
    try:
        target_kind, target = _selected_target(args)
        reference = _resolve_reference_arg(args.reference)
        mode = _validate_modes(
            reference=reference,
            only_phase=bool(args.only_phase),
            ap=bool(args.ap),
            gp=bool(args.gp),
            output_format=args.format,
        )
        region = (
            None
            if args.bimrange is None
            else parse_cli_range(args.bimrange, allow_chrom_only=True)
        )
        diagnostic_site = _diagnostic_site(args.trace_site)
    except ValueError as exc:
        parser.error(str(exc))

    output_prefix = _output_prefix(Path(args.out))
    output = _output_destination(output_prefix, args.format)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = str(output_prefix) + ".log"
    logger: logging.Logger = setup_logging(log_path)
    log_path = get_log_path(logger) or log_path
    terminal_rich = bool(stdout_is_tty() and rich_progress_available())
    if terminal_rich:
        report_logger = _logger_handlers(
            logger,
            "janusx.imp.file_only",
            file_handlers=True,
        )
        terminal_logger = _logger_handlers(
            logger,
            "janusx.imp.stream_only",
            file_handlers=False,
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
    reference_label = "None" if reference is None else str(reference)
    region_label = (
        "Whole input" if region is None else format_normalized_range(region)
    )
    mode_label = _format_mode(mode)
    progress_label = (
        "Phasing"
        if bool(args.only_phase)
        else "Phasing&Imputing"
    )
    general_config_rows = [
        ("Genotype file", target_display),
        ("Input format", input_format),
        ("Reference", reference_label),
        ("Region", region_label),
        ("Mode", mode_label),
        ("Output format", str(args.format).upper()),
        ("Window", f"{float(args.window):g} cM"),
        ("Overlap", f"{float(args.overlap):g} cM"),
        ("Marker cap", f"{int(args.window_markers):,}"),
        ("Phase states", f"{int(args.phase_states):,}"),
        ("Imputation states", f"{int(args.imp_states):,}"),
        ("Seed", int(args.seed)),
        ("Threads", int(args.threads)),
    ]
    config_rows = [
        *general_config_rows,
        ("Output", str(output)),
        ("Version", _janusx_version()),
    ]
    host = socket.gethostname()
    if terminal_rich:
        emit_cli_configuration(
            terminal_logger,
            app_title="JanusX - IMP",
            config_title="IMP CONFIG",
            host=host,
            sections=[("General", general_config_rows)],
            footer_rows=[("Output prefix", str(output_prefix))],
            line_max_chars=_TERMINAL_SECTION_WIDTH,
        )
    _report_banner(report_logger, "JanusX - IMP Pipeline Analysis")
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
    progress_total = None
    if terminal_rich:
        count_task = CliStatus(
            f"Loading genotype from {progress_target_label}...",
            enabled=True,
            use_process=True,
        )
        count_task.__enter__()
        try:
            samples_hint, counted = _inspect_target_dimensions(
                target_kind,
                target,
                region,
            )
            progress_total = int(counted) if int(counted) > 0 else None
            if progress_total is not None:
                report_logger.info(
                    f"  Progress denominator : {progress_total:,} target markers"
                )
            count_task.complete(
                f"Loading genotype from {progress_target_label} "
                f"(n={int(samples_hint):,}, nSNP={int(counted):,})"
            )
        except Exception as exc:
            message = (
                "Genotype dimensions unavailable for the loading status. "
                f"{exc}"
            )
            report_logger.warning(message)
            terminal_logger.warning(message)
        finally:
            count_task.close(show_done=False)
    reporter = _ImpProgressReporter(
        report_logger if terminal_rich else logger,
        total=progress_total,
        task_label=progress_label,
        imputation_mode=(not bool(args.only_phase)),
    )
    if target_kind == "hmp":
        cache_prefix = _hmp_cache_prefix(target)
        cache_prefix.parent.mkdir(parents=True, exist_ok=True)
        if _pfile_cache_complete(cache_prefix):
            report_logger.info(f"Using HMP PFILE cache: {cache_prefix}")
        else:
            report_logger.info(f"Creating HMP PFILE cache: {cache_prefix}")
            try:
                jxrs.imp_convert_to_pgen(
                    str(target),
                    str(cache_prefix),
                    reporter,
                )
            except Exception as exc:
                reporter.close()
                logger.error(str(exc))
                return 1
        target = cache_prefix

    try:
        options = {
            "target": str(target),
            "reference": None if reference is None else str(reference),
            "map": None if args.map is None else str(Path(args.map).expanduser()),
            "region": (
                None
                if region is None
                else {
                    "chrom": region.chrom_key,
                    "start": region.start,
                    "end": region.end,
                }
            ),
            "output": str(output),
            "format": args.format,
            "threads": int(args.threads),
            "seed": int(args.seed),
            "window": float(args.window),
            "overlap": float(args.overlap),
            "window_markers": int(args.window_markers),
            "phase_states": int(args.phase_states),
            "imp_states": int(args.imp_states),
            "adaptive_states": bool(args.adaptive_states),
            "burnin": int(args.burnin),
            "iterations": int(args.iterations),
            "initial_lr": float(args.initial_lr),
            "ne": int(args.ne),
            "err": args.err,
            "em": bool(args.em),
            "impute": not bool(args.only_phase),
            "imp_segment": float(args.imp_segment),
            "imp_step": float(args.imp_step),
            "imp_nsteps": int(args.imp_nsteps),
            "cluster": float(args.cluster),
            "ap": bool(args.ap),
            "gp": bool(args.gp),
            "assume_global_phase": bool(args.assume_global_phase),
            "diagnostic_site": diagnostic_site,
        }
        summary = json.loads(jxrs.imp_run(json.dumps(options), reporter))
    except Exception as exc:
        reporter.close()
        logger.error(str(exc))
        return 1
    reporter(
        "complete",
        "",
        0,
        int(summary["variants_written"]),
        int(summary["variants_written"]),
        int(summary["samples_written"]),
    )
    processing_elapsed = reporter.elapsed_seconds()
    reporter.close()
    if str(args.format).lower() == "pgen":
        generated_outputs = [
            Path(f"{output}.pgen"),
            Path(f"{output}.pvar"),
            Path(f"{output}.psam"),
        ]
    else:
        generated_outputs = [output]
    if summary.get("hmm_trace"):
        hmm_trace_path = Path(f"{output}.imp.hmm.tsv")
        with hmm_trace_path.open("w", newline="") as handle:
            trace_writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            trace_writer.writerow(
                (
                    "CHROM",
                    "POS",
                    "SAMPLE",
                    "WINDOW",
                    "ROUND",
                    "TRANSITION",
                    "STATE_ALLELES",
                    "EMISSION",
                    "FORWARD",
                    "BACKWARD",
                    "POSTERIOR_NORM",
                    "PHASE_SWAP",
                    "PHASE_LOG_ODDS",
                    "PHASE_THRESHOLD",
                    "SELECTED_ALLELES",
                )
            )
            for row in summary["hmm_trace"]:
                trace_writer.writerow(
                    (
                        row["chrom"],
                        row["position"],
                        row["sample"],
                        row["window_index"],
                        row["round"],
                        row["transition_probability"],
                        json.dumps(row["state_alleles"], separators=(",", ":")),
                        json.dumps(row["emission_probabilities"], separators=(",", ":")),
                        json.dumps(row["forward_probabilities"], separators=(",", ":")),
                        json.dumps(row["backward_probabilities"], separators=(",", ":")),
                        json.dumps(row["posterior_normalization"], separators=(",", ":")),
                        row["phase_swap"],
                        row["phase_log_odds"],
                        row["phase_threshold"],
                        json.dumps(row["selected_alleles"], separators=(",", ":")),
                    )
                )
        generated_outputs.append(hmm_trace_path)
    report_logger.info("")
    _report_block(report_logger, "Outputs Generated")
    for path in generated_outputs:
        _report_kv(report_logger, "Output", path)
    _report_kv(report_logger, "Log", log_path)
    report_logger.info("")
    _report_block(report_logger, "Advanced Parameters")
    _report_kv(report_logger, "Seed", int(args.seed))
    _report_kv(report_logger, "Window cM", float(args.window))
    _report_kv(report_logger, "Overlap cM", float(args.overlap))
    _report_kv(report_logger, "Window markers", int(args.window_markers))
    _report_kv(report_logger, "Phase states", int(args.phase_states))
    _report_kv(report_logger, "Imputation states", int(args.imp_states))
    _report_kv(report_logger, "Adaptive states", bool(args.adaptive_states))
    _report_kv(report_logger, "EM", bool(args.em))
    _report_kv(
        report_logger,
        "Assume global phase",
        bool(args.assume_global_phase),
    )
    _report_kv(
        report_logger,
        "Peak active bytes",
        int(summary["peak_window_bytes"]),
    )
    parameter_trace = summary.get("parameter_trace", [])
    if parameter_trace:
        report_logger.info("")
        _report_block(report_logger, "EM Parameter Trace")
        report_logger.info("  CHROM\tWINDOW\tSTAGE\tROUND\tNE\tERR")
        for row in parameter_trace:
            report_logger.info(
                f"  {row['chrom']}\t{int(row['window_index'])}\t{row['stage']}"
                f"\t{int(row['round'])}\t{int(row['ne'])}\t{float(row['err']):.6g}"
            )
    report_logger.info("")
    _report_block(report_logger, "Execution Summary")
    _report_kv(report_logger, "Status", "done")
    _report_kv(report_logger, "Variants written", int(summary["variants_written"]))
    _report_kv(report_logger, "Samples", int(summary["samples_written"]))
    _report_kv(
        report_logger,
        "Total windows",
        int(summary["total_target_windows"]),
    )
    _report_kv(
        report_logger,
        "Direct-imputation",
        int(summary["direct_imputation_windows"]),
    )
    _report_kv(
        report_logger,
        "Normal phasing",
        int(summary["normal_phasing_windows"]),
    )
    _report_kv(
        report_logger,
        "Reference-assisted phasing",
        int(summary.get("reference_phasing_windows", 0)),
    )
    _report_kv(
        report_logger,
        "Imputation windows",
        int(summary["imputation_windows"]),
    )
    _report_kv(
        report_logger,
        "Direct bypass",
        f"{100.0 * float(summary['direct_bypass_fraction']):.1f}%",
    )
    if summary.get("direct_fallback_reasons"):
        _report_kv(
            report_logger,
            "Fallback reasons",
            json.dumps(summary["direct_fallback_reasons"], sort_keys=True),
        )
    elapsed_seconds = max(time.monotonic() - wall_started, 0.0)
    _report_kv(report_logger, "Total wall time", f"{elapsed_seconds:.2f}s")
    report_logger.info(f"IMP run status: done; log: {log_path}")
    if terminal_rich:
        phase_performed = int(summary["normal_phasing_windows"]) > 0
        imputation_performed = int(summary["imputation_windows"]) > 0
        if phase_performed and imputation_performed:
            completion_label = "Phasing&Imputing"
        elif phase_performed:
            completion_label = "Phasing"
        elif imputation_performed:
            completion_label = "Imputing"
        else:
            completion_label = "Phasing"
        log_success(
            terminal_logger,
            f"{completion_label} ...Finished "
            f"({int(summary['variants_written']):,} markers written) "
            f"[{format_elapsed(processing_elapsed)}]",
            force_color=True,
        )
        saved_lines = "\n".join(f"  {path}" for path in generated_outputs)
        saved_format = "PGEN" if str(args.format).lower() == "pgen" else "VCF"
        terminal_logger.info(
            f"Saved genotype in {saved_format} format:\n{saved_lines}"
        )
        lt = time.localtime()
        endinfo = (
            f"\nFinished IMP calculation. Total wall time: {elapsed_seconds:.2f} seconds\n"
            f"{lt.tm_year}-{lt.tm_mon:02d}-{lt.tm_mday:02d} "
            f"{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}"
        )
        _emit_to_file_handlers(report_logger, logging.INFO, endinfo)
        log_success(terminal_logger, endinfo)
    else:
        log_success(
            report_logger,
            f"Phased {int(summary['variants_written'])} variants for "
            f"{int(summary['samples_written'])} samples.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
