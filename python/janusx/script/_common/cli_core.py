"""
Core CLI parser and help-formatting utilities for JanusX entrypoints.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Sequence

try:
    from rich_argparse import RichHelpFormatter, RawDescriptionRichHelpFormatter
except Exception:
    RichHelpFormatter = None  # type: ignore[assignment]
    RawDescriptionRichHelpFormatter = None  # type: ignore[assignment]

_CITATION_URL = "https://github.com/FJingxian/JanusX/"
_ORANGE_ANSI = "\033[38;5;208m"
_RESET_ANSI = "\033[0m"

__all__ = [
    "CliArgumentParser",
    "cli_help_formatter",
    "minimal_help_epilog",
]


def _heading_orange(text: str) -> str:
    if RawDescriptionRichHelpFormatter is not None or RichHelpFormatter is not None:
        return f"[orange3]{text}[/orange3]"
    if os.environ.get("NO_COLOR"):
        return text
    if str(os.environ.get("TERM", "")).lower() == "dumb":
        return text
    if bool(getattr(sys.stdout, "isatty", lambda: False)()):
        return f"{_ORANGE_ANSI}{text}{_RESET_ANSI}"
    return text


def _format_help_group_name(text: str) -> str:
    """
    Title-case plain argparse headings while preserving acronym-heavy tokens.

    Examples:
      required gwas input -> Required Gwas Input
      Required GWAS Input -> Required GWAS Input
      LDBlock Plot -> LDBlock Plot
      Q-Q Plot -> Q-Q Plot
    """

    def _format_token(token: str) -> str:
        if token == "":
            return token
        if any(ch.isupper() for ch in token[1:]) or token.isupper() or any(ch.isdigit() for ch in token):
            return token
        if "-" in token:
            return "-".join(_format_token(part) for part in token.split("-"))
        return token[:1].upper() + token[1:]

    return " ".join(_format_token(part) for part in str(text).split(" "))


def cli_help_formatter() -> type[argparse.HelpFormatter]:
    term_width = max(40, int(shutil.get_terminal_size((100, 20)).columns))
    help_pos = min(32, max(20, term_width // 3))

    if RawDescriptionRichHelpFormatter is not None:

        class _Formatter(RawDescriptionRichHelpFormatter):  # type: ignore[misc, valid-type]
            styles = RawDescriptionRichHelpFormatter.styles.copy()  # type: ignore[attr-defined]
            styles["argparse.groups"] = "orange3"
            group_name_formatter = staticmethod(_format_help_group_name)

            def __init__(self, prog: str) -> None:
                super().__init__(prog, width=term_width, max_help_position=help_pos)

        return _Formatter

    if RichHelpFormatter is not None:

        class _Formatter(RichHelpFormatter):  # type: ignore[misc, valid-type]
            styles = RichHelpFormatter.styles.copy()  # type: ignore[attr-defined]
            styles["argparse.groups"] = "orange3"
            group_name_formatter = staticmethod(_format_help_group_name)

            def __init__(self, prog: str) -> None:
                super().__init__(prog, width=term_width, max_help_position=help_pos)

        return _Formatter

    class _Formatter(argparse.RawDescriptionHelpFormatter):
        def __init__(self, prog: str) -> None:
            super().__init__(prog, width=term_width, max_help_position=help_pos)

    return _Formatter


class CliArgumentParser(argparse.ArgumentParser):
    """
    Keep argparse usage formatting, but standardize error line as:
    Error: <message>
    """

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"Error: {message}\n")


def minimal_help_epilog(examples: Sequence[str]) -> str:
    lines = [_heading_orange("Examples:")]
    for ex in examples:
        lines.append(f"  {str(ex)}")
    lines.append("")
    lines.append(_heading_orange("Citation:"))
    lines.append(f"  {_CITATION_URL}")
    return "\n".join(lines)
