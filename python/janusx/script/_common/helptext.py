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


def cli_help_formatter() -> type[argparse.HelpFormatter]:
    term_width = max(40, int(shutil.get_terminal_size((100, 20)).columns))
    help_pos = min(32, max(20, term_width // 3))

    if RawDescriptionRichHelpFormatter is not None:
        class _Formatter(RawDescriptionRichHelpFormatter):  # type: ignore[misc, valid-type]
            styles = RawDescriptionRichHelpFormatter.styles.copy()  # type: ignore[attr-defined]
            styles["argparse.groups"] = "orange3"

            def __init__(self, prog: str) -> None:
                super().__init__(prog, width=term_width, max_help_position=help_pos)
        return _Formatter

    if RichHelpFormatter is not None:
        class _Formatter(RichHelpFormatter):  # type: ignore[misc, valid-type]
            styles = RichHelpFormatter.styles.copy()  # type: ignore[attr-defined]
            styles["argparse.groups"] = "orange3"

            def __init__(self, prog: str) -> None:
                super().__init__(prog, width=term_width, max_help_position=help_pos)
        return _Formatter

    class _Formatter(argparse.RawDescriptionHelpFormatter):
        def __init__(self, prog: str) -> None:
            super().__init__(prog, width=term_width, max_help_position=help_pos)
    return _Formatter


def minimal_help_epilog(examples: Sequence[str]) -> str:
    lines = [_heading_orange("Examples:")]
    for ex in examples:
        lines.append(f"  {str(ex)}")
    lines.append("")
    lines.append(_heading_orange("Citation:"))
    lines.append(f"  {_CITATION_URL}")
    return "\n".join(lines)
