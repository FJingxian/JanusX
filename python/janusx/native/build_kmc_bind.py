from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_python_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prebuild janusx._kmc_count into cache.")
    parser.add_argument("--kmc-src", type=str, default=None, help="Optional KMC source directory.")
    parser.add_argument("--rebuild", action="store_true", default=False, help="Force rebuild.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose compiler output.")
    args = parser.parse_args()

    py_dir = _repo_python_dir()
    if str(py_dir) not in sys.path:
        sys.path.insert(0, str(py_dir))

    from janusx.kmc_bind import build_kmc_bind_module

    out = build_kmc_bind_module(
        kmc_src=args.kmc_src,
        rebuild=bool(args.rebuild),
        verbose=bool(args.verbose),
    )
    print(f"[janusx] prebuilt KMC bind cache: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
