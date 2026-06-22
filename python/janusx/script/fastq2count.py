from __future__ import annotations

import argparse
import sys

from janusx.pipeline import format_report, run_fastq2count_checks

try:
    from janusx import janusx as jxrs
except Exception:
    jxrs = None

from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx fastq2count",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx fastq2count --check-only",
                "jx fastq2count -r ref.fa -a genes.gtf -i fastq_dir -w workdir",
            ]
        ),
        description=(
            "Environment compatibility check for fastq2count workflow. "
            "All required tools must pass before entering pipeline execution."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run compatibility checks and exit (default behavior).",
    )
    return parser


def main() -> int:
    if any(tok in {"-h", "--help"} for tok in sys.argv[1:]):
        if jxrs is None or not hasattr(jxrs, "fastq2count_run"):
            build_parser().print_help()
            return 0
        return int(jxrs.fastq2count_run(list(sys.argv[1:])))

    parser = build_parser()
    parsed, passthrough = parser.parse_known_args()

    if jxrs is None or not hasattr(jxrs, "fastq2count_run"):
        raise RuntimeError(
            "JanusX Rust extension is unavailable or missing `fastq2count_run`. "
            "Please rebuild/install the extension first."
        )

    report = run_fastq2count_checks()
    print(format_report(report))

    if not report.passed:
        return 1
    if parsed.check_only and not passthrough:
        return 0
    return int(jxrs.fastq2count_run(list(sys.argv[1:])))


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
