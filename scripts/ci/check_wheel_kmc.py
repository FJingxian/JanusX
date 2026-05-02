#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import zipfile


def _is_kmc_entry(name: str) -> bool:
    if not name.startswith("janusx/_kmc_count"):
        return False
    return name.endswith(".so") or name.endswith(".pyd") or ".so." in name


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify wheel contains prebuilt janusx/_kmc_count extension.")
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["dist/*.whl"],
        help="Wheel glob pattern(s). Default: dist/*.whl",
    )
    parser.add_argument(
        "--exactly-one",
        action="store_true",
        default=False,
        help="Require exactly one matched wheel.",
    )
    args = parser.parse_args()

    wheels: list[str] = []
    for pat in args.patterns:
        wheels.extend(glob.glob(pat))
    wheels = sorted(set(os.path.abspath(w) for w in wheels))

    if not wheels:
        raise SystemExit("Strict mode failed: no wheel matched pattern(s)")
    if args.exactly_one and len(wheels) != 1:
        raise SystemExit(f"Strict mode failed: expected exactly one wheel, got {len(wheels)}")

    for wheel in wheels:
        with zipfile.ZipFile(wheel, "r") as zf:
            entries = [n for n in zf.namelist() if _is_kmc_entry(n)]
        print(f"[check_wheel_kmc] wheel={wheel}")
        print(f"[check_wheel_kmc] kmc_entries={entries}")
        if not entries:
            raise SystemExit(
                f"Strict mode failed: wheel lacks prebuilt janusx/_kmc_count extension: {wheel}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
