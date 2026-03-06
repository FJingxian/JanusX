from __future__ import annotations

import argparse
from typing import Sequence

try:
    from rich_argparse import RichHelpFormatter
except Exception:
    RichHelpFormatter = None  # type: ignore[assignment]

_CITATION_URL = "https://github.com/FJingxian/JanusX/"


def cli_help_formatter() -> type[argparse.HelpFormatter]:
    if RichHelpFormatter is not None:
        return RichHelpFormatter
    return argparse.RawDescriptionHelpFormatter


def minimal_help_epilog(examples: Sequence[str]) -> str:
    lines = ["Examples"]
    lines.append("--------")
    for ex in examples:
        lines.append(f"  {str(ex)}")
    lines.append("")
    lines.append("Citation")
    lines.append("--------")
    lines.append(f"  {_CITATION_URL}")
    return "\n".join(lines)
