from __future__ import annotations

from typing import Sequence


_CITATION_URL = "https://github.com/FJingxian/JanusX/"


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

