#!/usr/bin/env python3
"""Prepare the ADAMIXTURE fork for the JanusX companion distribution.

The source package keeps the import namespace and console entry point named
``adamixture``.  Only the distribution name is changed so it can coexist with
the upstream project on package indexes and be used as an explicit JanusX
dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected one {label} entry, found {count}.")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="ADAMIXTURE source checkout")
    parser.add_argument(
        "--distribution-name",
        default="janusx-adamixture",
        help="PEP 503 distribution name (default: %(default)s)",
    )
    args = parser.parse_args()

    root = args.source.resolve()
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        raise SystemExit(f"ADAMIXTURE pyproject.toml not found: {pyproject}")

    text = pyproject.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        'name = "adamixture"',
        f'name = "{args.distribution_name}"',
        label="project name",
    )
    pyproject.write_text(text, encoding="utf-8")
    print(f"[adamixture-build] distribution={args.distribution_name}")
    print("[adamixture-build] import package and CLI remain named adamixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
