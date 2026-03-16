#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load/list registry files for JanusX runtime."""

from __future__ import annotations

import argparse
from typing import Sequence

from ._common.gwas_history import (
    list_loaded_files,
    register_loaded_file,
    remove_loaded_file_by_id,
)
from ._common.helptext import CliArgumentParser


def build_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        prog="jx --load",
        description=(
            "List or register files for JanusX runtime.\n"
            "List:  jx --load\n"
            "Load:  jx --load <type> <name> <file>\n"
            "Clean: jx --clean <id>"
        ),
    )
    p.add_argument(
        "--clean-id",
        default="",
        help="Remove one loaded record by unique id.",
    )
    p.add_argument(
        "items",
        nargs="*",
        help="Either empty (list), or: <type> <name> <file>",
    )
    return p


def _print_rows(rows: Sequence[dict]) -> None:
    print("Loaded files")
    print("ID | Type | Name | File | Status")
    print("---|---|---|---|---")
    for r in rows:
        print(
            f"{r.get('id','')} | {r.get('type','')} | {r.get('name','')} | "
            f"{r.get('file_name','')} | {r.get('status',0)}"
        )


def main() -> None:
    args = build_parser().parse_args()
    clean_id = str(args.clean_id or "").strip()
    items = list(args.items or [])
    if clean_id != "":
        if len(items) > 0:
            raise SystemExit("Usage: jx --clean <id>")
        out = remove_loaded_file_by_id(clean_id)
        print(f"Removed load id: {out.get('id','')}")
        print(f"  type: {out.get('type','')}  name: {out.get('name','')}")
        for p in out.get("removed_paths", []):
            print(f"  removed: {p}")
        for p in out.get("failed_paths", []):
            print(f"  failed: {p}")
        return
    if len(items) == 0:
        rows = list_loaded_files(limit=5000)
        _print_rows(rows)
        return
    if len(items) != 3:
        raise SystemExit("Usage: jx --load [<type> <name> <file>]")
    ftype, name, file_path = items
    out = register_loaded_file(ftype, name, file_path)
    chroms = out.get("chroms", [])
    chrom_preview = ",".join([str(x) for x in chroms[:12]])
    if len(chroms) > 12:
        chrom_preview += ",..."
    print(f"Loaded: {out.get('type','')} {out.get('name','')}")
    print(f"  id: {out.get('id','')}")
    print(f"  file: {out.get('file_name','')}")
    print(f"  path: {out.get('stored_path','')}")
    print(f"  md5: {out.get('md5','')}")
    print(f"  chr: {chrom_preview}")
    print(f"  status: {out.get('status','')}")
    if out.get("gzip_started", False):
        print("  gzip: started in background")


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
