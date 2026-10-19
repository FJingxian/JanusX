#!/usr/bin/env python3
"""Reject macOS wheel Mach-O libraries with a misaligned LINKEDIT string table."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


SYMTAB_RE = re.compile(
    r"cmd LC_SYMTAB\s+cmdsize\s+\d+\s+symoff\s+\d+\s+nsyms\s+\d+"
    r"\s+stroff\s+(\d+)\s+strsize\s+\d+"
)


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def _check_macho(path: Path, label: str) -> list[str]:
    description = _run("file", "-b", str(path))
    if "Mach-O" not in description:
        return []

    architectures = _run("lipo", "-archs", str(path)).split()
    errors: list[str] = []
    for architecture in architectures:
        load_commands = _run("otool", "-arch", architecture, "-l", str(path))
        offsets = [int(value) for value in SYMTAB_RE.findall(load_commands)]
        if not offsets:
            errors.append(f"{label} ({architecture}): missing LC_SYMTAB")
            continue
        for offset in offsets:
            if offset % 8:
                errors.append(
                    f"{label} ({architecture}): LC_SYMTAB.stroff={offset} "
                    f"(0x{offset:x}) is not 8-byte aligned"
                )
    return errors


def _wheel_members(wheel: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(wheel) as archive:
        members = [
            info
            for info in archive.infolist()
            if info.filename.endswith((".so", ".dylib", ".bundle"))
            or "/bin/FastTree" in info.filename
        ]
        return [(info.filename, archive.read(info)) for info in members]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="macOS wheel to inspect")
    args = parser.parse_args()
    if not args.wheel.is_file():
        parser.error(f"wheel does not exist: {args.wheel}")

    errors: list[str] = []
    checked = 0
    with tempfile.TemporaryDirectory(prefix="janusx-linkedit-") as temp:
        root = Path(temp)
        for member, contents in _wheel_members(args.wheel):
            path = root / member.replace("/", "__")
            path.write_bytes(contents)
            if "Mach-O" in _run("file", "-b", str(path)):
                checked += 1
                errors.extend(_check_macho(path, member))

    if checked == 0:
        print(f"[linkedit] no Mach-O libraries found in {args.wheel}", file=sys.stderr)
        return 1
    if errors:
        print("[linkedit] invalid macOS wheel:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"[linkedit] checked {checked} Mach-O file(s); all LINKEDIT string pools are aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
