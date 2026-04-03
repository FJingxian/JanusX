from __future__ import annotations

import argparse

from janusx.pipeline import format_report, run_fastq2count_checks

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
    parser = build_parser()
    _, passthrough = parser.parse_known_args()

    report = run_fastq2count_checks()
    print(format_report(report))

    if not report.passed:
        return 1
    if passthrough:
        print("Compatibility gate passed, but Python `jxpy fastq2count` only performs preflight checks.")
        print("Run launcher pipeline command to execute workflow: `jx fastq2count ...`")
        return 2
    return 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
