from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .compat import PipelineCompatReport


def run_fastq2vcf_checks() -> "PipelineCompatReport":
    from .compat import run_fastq2vcf_checks as _impl

    return _impl()


def run_fastq2count_checks() -> "PipelineCompatReport":
    from .compat import run_fastq2count_checks as _impl

    return _impl()


def format_report(report: "PipelineCompatReport") -> str:
    from .compat import format_report as _impl

    return _impl(report)

__all__ = [
    "PipelineCompatReport",
    "format_report",
    "run_fastq2count_checks",
    "run_fastq2vcf_checks",
]
