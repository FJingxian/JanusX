#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sync project versions in pyproject.toml and Cargo.toml from a git tag.

Usage:
  python scripts/sync_version_from_tag.py v1.2.3
  python scripts/sync_version_from_tag.py 1.2.3
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CARGO = ROOT / "Cargo.toml"
JANUSX_CLI = ROOT / "python" / "janusx" / "script" / "JanusX.py"


def _normalize_tag(tag: str) -> str:
    t = str(tag).strip()
    if t.startswith("refs/tags/"):
        t = t[len("refs/tags/") :]
    if t.startswith("v"):
        t = t[1:]
    if t == "":
        raise ValueError("Empty tag/version.")
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.\-+]*", t):
        raise ValueError(f"Unsupported version format from tag: {tag}")
    return t


def _replace_section_version(path: Path, section: str, new_version: str) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_section = False
    replaced = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == f"[{section}]"
            continue
        if in_section and stripped.startswith("version"):
            m = re.match(r'^(\s*version\s*=\s*")[^"]*(".*)$', line)
            if m is None:
                raise ValueError(f"Malformed version line in {path}: {line.rstrip()}")
            lines[i] = f'{m.group(1)}{new_version}{m.group(2)}\n'
            replaced = True
            break

    if not replaced:
        raise ValueError(f"Cannot find version under [{section}] in {path}")

    path.write_text("".join(lines), encoding="utf-8")
    return replaced


def _replace_build_date_fallback(path: Path, build_date: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = r'(__BUILD_DATE_FALLBACK__\s*=\s*")[^"]*(")'
    updated, n = re.subn(pattern, rf"\g<1>{build_date}\2", text, count=1)
    if n != 1:
        raise ValueError(
            f"Cannot find __BUILD_DATE_FALLBACK__ in {path}. "
            "Please define it once in JanusX.py."
        )
    path.write_text(updated, encoding="utf-8")


def _git_head_date() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "log", "-1", "--date=format:%Y-%m-%d", "--format=%cd"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return date.today().isoformat()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: python scripts/sync_version_from_tag.py <tag-or-version>")
        return 2

    version = _normalize_tag(args[0])
    build_date = _git_head_date()
    _replace_section_version(PYPROJECT, "project", version)
    _replace_section_version(CARGO, "package", version)
    _replace_build_date_fallback(JANUSX_CLI, build_date)
    print(
        "Synchronized version/build date:\n"
        f"  version    = {version}\n"
        f"  build_date = {build_date}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
