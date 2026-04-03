from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Callable

from .tools import (
    check_bcftools,
    check_beagle,
    check_bgzip,
    check_bwa,
    check_fastp,
    check_featurecounts,
    check_gatk,
    check_hisat2,
    check_hisat2_build,
    check_plink,
    check_python3,
    check_samblaster,
    check_samtools,
    check_tabix,
)
from .tools._probe import ProbeResult

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_WHITE = "\033[37m"
_RESET = "\033[0m"


def _supports_color() -> bool:
    force = str(os.environ.get("JANUSX_FORCE_COLOR", "")).strip().lower()
    if force in {"1", "true", "yes", "on"}:
        return True
    if force in {"0", "false", "no", "off"}:
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _paint(text: str, color: str) -> str:
    if not _supports_color():
        return text
    return f"{color}{text}{_RESET}"


@dataclass(frozen=True)
class PipelineCompatReport:
    pipeline: str
    checks: list[ProbeResult]

    @property
    def passed(self) -> bool:
        return all(item.ok for item in self.checks)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for item in self.checks if item.ok)


def _run_report(
    pipeline: str, probes: list[Callable[[], ProbeResult]]
) -> PipelineCompatReport:
    return PipelineCompatReport(pipeline=pipeline, checks=[probe() for probe in probes])


def run_fastq2vcf_checks() -> PipelineCompatReport:
    probes = [
        check_fastp.probe,
        check_bwa.probe,
        check_samblaster.probe,
        check_gatk.probe,
        check_bcftools.probe,
        check_tabix.probe,
        check_bgzip.probe,
        check_plink.probe,
        check_beagle.probe,
    ]
    return _run_report("fastq2vcf", probes)


def run_fastq2count_checks() -> PipelineCompatReport:
    probes = [
        check_fastp.probe,
        check_hisat2_build.probe,
        check_hisat2.probe,
        check_samtools.probe,
        check_featurecounts.probe,
        check_python3.probe,
    ]
    return _run_report("fastq2count", probes)


def format_report(report: PipelineCompatReport) -> str:
    lines = [
        f"[{report.pipeline}] environment check: {report.passed_count}/{report.total} tools ready"
    ]
    for item in report.checks:
        marker = "✓" if item.ok else "✗"
        detail = item.detail.strip() if item.detail else ""
        color = _GREEN if item.ok else _YELLOW
        if detail:
            lines.append(_paint(f"  {marker} {item.name}: {detail}", color))
        else:
            lines.append(_paint(f"  {marker} {item.name}", color))
    if report.passed:
        lines.append(_paint("All required tools are available. Pipeline can start.", _WHITE))
    else:
        lines.append(
            _paint("Some required tools are missing. Pipeline is blocked.", _WHITE)
        )
    return "\n".join(lines)
