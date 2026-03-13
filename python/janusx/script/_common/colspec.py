# -*- coding: utf-8 -*-
"""
Column index specification helpers for CLI options.

This module provides shared parsers for index expressions like:
  - single index: "3"
  - range       : "1:10" or "1-10" (inclusive)
  - list        : "1,2,5,10"
"""

from __future__ import annotations

import re
from typing import Iterable


_RANGE_RE = re.compile(r"^\s*(\d+)\s*([:-])\s*(\d+)\s*$")


def _split_spec_tokens(values: Iterable[object] | None) -> list[str]:
    out: list[str] = []
    for raw in list(values or []):
        s = str(raw).strip()
        if s == "":
            continue
        for part in s.split(","):
            p = str(part).strip()
            if p != "":
                out.append(p)
    return out


def parse_zero_based_index_specs(
    values: Iterable[object] | None,
    *,
    label: str = "-n/--ncol",
) -> list[int] | None:
    """
    Parse zero-based column index specs from CLI inputs.

    Supported forms (can be mixed):
      - single: "3"
      - range : "1:10" or "1-10" (inclusive)
      - list  : "1,2,5,10"
    """
    if values is None:
        return None

    tokens = _split_spec_tokens(values)
    if len(tokens) == 0:
        return []

    parsed: list[int] = []
    for tk in tokens:
        m = _RANGE_RE.match(tk)
        if m is not None:
            left = int(m.group(1))
            right = int(m.group(3))
            step = 1 if right >= left else -1
            parsed.extend(range(left, right + step, step))
            continue
        if tk.isdigit():
            parsed.append(int(tk))
            continue
        raise ValueError(
            f"Invalid {label} token: `{tk}`. "
            "Supported forms: `3`, `1:10`, `1-10`, `1,2,5,10` (zero-based)."
        )

    out: list[int] = []
    seen: set[int] = set()
    for idx in parsed:
        if idx < 0:
            raise ValueError(
                f"Invalid {label} index: {idx}. Indices must be >= 0 (zero-based)."
            )
        if idx not in seen:
            out.append(int(idx))
            seen.add(int(idx))
    return out

