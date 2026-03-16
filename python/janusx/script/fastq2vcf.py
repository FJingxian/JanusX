# -*- coding: utf-8 -*-
"""
Compatibility shim for Python-side fastq2vcf.

`jx fastq2vcf` is now implemented in the Rust launcher.
This Python entry is intentionally retired to keep pipeline ownership in launcher.
"""

from __future__ import annotations

import argparse
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx fastq2vcf",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx fastq2vcf -r ref.fa -i fastq_dir -w workdir",
                "jx fastq2vcf -r ref.fa -i 3.gvcf -w workdir -from-step 4 -to-step 6",
            ]
        ),
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Passed through to launcher implementation.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    _ = parser.parse_args()
    print("Error: Python fastq2vcf has been retired.")
    print("Please run launcher command: `jx fastq2vcf ...`")
    return 2


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    raise SystemExit(main())
