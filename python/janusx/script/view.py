from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog

BIN01_MAGIC = b"JXBIN001"
BIN01_HEADER_SIZE = 32
BIN_SITE_MAGIC = b"JXBSITE1"
BIN_SITE_HEADER_SIZE = 24

_BITS_LE = tuple("".join("1" if ((b >> i) & 1) else "0" for i in range(8)) for b in range(256))
_DNA_4 = tuple("".join("ATCG"[(b >> s) & 0b11] for s in (0, 2, 4, 6)) for b in range(256))


def _read_exact(handle, n: int) -> bytes:
    data = handle.read(int(n))
    if len(data) != int(n):
        raise RuntimeError(f"Unexpected EOF while reading {int(n)} bytes.")
    return data


def _detect_bin_kind(path: Path) -> str:
    with path.open("rb") as f:
        magic = f.read(8)
    if magic == BIN01_MAGIC:
        return "bin01"
    if magic == BIN_SITE_MAGIC:
        return "binsite"
    raise RuntimeError(
        f"Unsupported binary header in '{path}'. "
        "Expected JXBIN001 (.bin) or JXBSITE1 (.bin.site)."
    )


def _dump_bin01(path: Path) -> None:
    out = sys.stdout
    write = out.write
    with path.open("rb") as f:
        hdr = _read_exact(f, BIN01_HEADER_SIZE)
        if hdr[:8] != BIN01_MAGIC:
            raise RuntimeError(f"Invalid BIN magic in '{path}'.")
        n_sites, n_samples, _reserved = struct.unpack("<QQQ", hdr[8:32])
        row_nbytes = (int(n_samples) + 7) // 8
        if row_nbytes == 0 and int(n_sites) > 0:
            raise RuntimeError(f"Invalid BIN header in '{path}': n_samples={n_samples}.")
        trim = int(n_samples) & 7
        for _ in range(int(n_sites)):
            row = _read_exact(f, row_nbytes)
            bits = "".join(_BITS_LE[b] for b in row)
            if trim != 0:
                bits = bits[: int(n_samples)]
            write(bits)
            write("\n")


def _dump_binsite(path: Path) -> None:
    out = sys.stdout
    write = out.write
    with path.open("rb") as f:
        hdr = _read_exact(f, BIN_SITE_HEADER_SIZE)
        if hdr[:8] != BIN_SITE_MAGIC:
            raise RuntimeError(f"Invalid BIN.SITE magic in '{path}'.")
        n_sites, _reserved = struct.unpack("<QQ", hdr[8:24])
        for _ in range(int(n_sites)):
            klen = struct.unpack("<H", _read_exact(f, 2))[0]
            n_packed = (int(klen) + 3) // 4
            packed = _read_exact(f, n_packed)
            seq = "".join(_DNA_4[b] for b in packed)
            if (int(klen) & 3) != 0:
                seq = seq[: int(klen)]
            write(seq)
            write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx view",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx view -bin test/kmerge_new.bin | head",
                "jx view -bin test/kmerge_new.bin.site | head",
                "jx view -bin test/kmerge_new.bin.site | sort | uniq | head",
            ]
        ),
        description=(
            "View JanusX binary files as plain text for shell pipelines. "
            "`.bin` -> rows of 0/1 bit strings; `.bin.site` -> decoded ATCG k-mers."
        ),
    )
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "-bin",
        "--bin",
        nargs="+",
        required=True,
        help="Input file(s): .bin or .bin.site.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    files = [Path(str(x)).expanduser().resolve() for x in args.bin]
    for p in files:
        if not p.is_file():
            parser.error(f"Input file not found: {p}")

    try:
        for p in files:
            kind = _detect_bin_kind(p)
            if kind == "bin01":
                _dump_bin01(p)
            elif kind == "binsite":
                _dump_binsite(p)
            else:
                raise RuntimeError(f"Unsupported file type for '{p}'.")
        return 0
    except BrokenPipeError:
        # Avoid "Exception ignored while flushing sys.stdout" on pipe close.
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except Exception:
            pass
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
