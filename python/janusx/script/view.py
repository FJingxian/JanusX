from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

from ._common.binsidecar import (
    BIN01_MAGIC,
    BIN_SITE_MAGIC,
    LEGACY_BSITE_HEADER_SIZE,
    LEGACY_BSITE_MAGIC,
    LEGACY_BSITE_VERSION,
)
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog

BKMER_MAGIC = b"JXBKMR1\x00"
BKMER_HEADER_SIZE = 64
KMER_BSITE_MAGIC = b"JXBSIT1\x00"
KMER_BSITE_HEADER_SIZE = 80

_BITS_LE = tuple("".join("1" if ((b >> i) & 1) else "0" for i in range(8)) for b in range(256))
_ALLELE_2 = tuple(
    "".join("ATCGN"[(b >> s) & 0x0F] if ((b >> s) & 0x0F) < 5 else "N" for s in (0, 4))
    for b in range(256)
)


def _read_exact(handle, n: int) -> bytes:
    data = handle.read(int(n))
    if len(data) != int(n):
        raise RuntimeError(f"Unexpected EOF while reading {int(n)} bytes.")
    return data


def _detect_bin_kind(path: Path) -> str:
    with path.open("rb") as f:
        magic = f.read(8)
    if magic == BIN01_MAGIC:
        raise RuntimeError(
            f"Deprecated JXBIN001 file in '{path}'. `jx view` no longer supports legacy `.bin`; "
            "please migrate to `.bkmer/.bsite` outputs."
        )
    if magic == BIN_SITE_MAGIC:
        raise RuntimeError(
            f"Deprecated JXBSITE1 file in '{path}'. `jx view` no longer supports legacy "
            "`.bin.site`; please migrate to `.bkmer/.bsite` outputs."
        )
    if magic == BKMER_MAGIC:
        return "bkmer"
    if magic == KMER_BSITE_MAGIC:
        return "kmer_bsite"
    if magic == LEGACY_BSITE_MAGIC:
        return "legacy_bsite"
    raise RuntimeError(
        f"Unsupported binary header in '{path}'. "
        "Expected JXBKMR1\\0 (.bkmer), JXBSIT1\\0 (kmerge .bsite), "
        "or JXBSIT02 (legacy .bsite)."
    )


def _decode_kmer_code_u64(code: int, k: int) -> str:
    out = ["A"] * int(k)
    for idx in range(int(k) - 1, -1, -1):
        base = int(code) & 0b11
        out[idx] = "ACGT"[base]
        code >>= 2
    return "".join(out)


def _dump_bkmer(path: Path) -> None:
    out = sys.stdout
    write = out.write
    chunk_records = 8192
    with path.open("rb") as f:
        hdr = _read_exact(f, BKMER_HEADER_SIZE)
        if hdr[:8] != BKMER_MAGIC:
            raise RuntimeError(f"Invalid BKMER magic in '{path}'.")
        version, k = struct.unpack("<II", hdr[8:16])
        n_kmers = struct.unpack("<Q", hdr[16:24])[0]
        if int(version) != 1:
            raise RuntimeError(f"Unsupported BKMER version in '{path}': {version}.")
        if int(k) <= 0 or int(k) > 31:
            raise RuntimeError(f"Unsupported BKMER k in '{path}': {k}.")

        remaining = int(n_kmers)
        while remaining > 0:
            take = min(remaining, chunk_records)
            raw = _read_exact(f, take * 8)
            for off in range(0, len(raw), 8):
                code = struct.unpack("<Q", raw[off : off + 8])[0]
                write(_decode_kmer_code_u64(code, int(k)))
                write("\n")
            remaining -= take


def _dump_kmer_bsite(path: Path) -> None:
    out = sys.stdout
    write = out.write
    chunk_cols = 4096
    with path.open("rb") as f:
        hdr = _read_exact(f, KMER_BSITE_HEADER_SIZE)
        if hdr[:8] != KMER_BSITE_MAGIC:
            raise RuntimeError(f"Invalid kmerge BSITE magic in '{path}'.")
        version, layout = struct.unpack("<II", hdr[8:16])
        n_samples, n_kmers, bytes_per_col = struct.unpack("<QQQ", hdr[16:40])
        bit_order, compression, block_cols = struct.unpack("<III", hdr[40:52])
        if int(version) != 1:
            raise RuntimeError(f"Unsupported kmerge BSITE version in '{path}': {version}.")
        if int(layout) != 1:
            raise RuntimeError(f"Unsupported kmerge BSITE layout in '{path}': {layout}.")
        if int(bit_order) != 0:
            raise RuntimeError(f"Unsupported kmerge BSITE bit order in '{path}': {bit_order}.")
        if int(compression) != 0 or int(block_cols) != 0:
            raise RuntimeError(
                f"Compressed kmerge BSITE is not supported in view yet for '{path}'."
            )
        if int(bytes_per_col) != (int(n_samples) + 7) // 8:
            raise RuntimeError(
                f"Invalid kmerge BSITE header in '{path}': bytes_per_col={bytes_per_col}, "
                f"n_samples={n_samples}."
            )

        remaining = int(n_kmers)
        col_nbytes = int(bytes_per_col)
        trim = int(n_samples)
        while remaining > 0:
            take = min(remaining, chunk_cols)
            raw = _read_exact(f, take * col_nbytes)
            for off in range(0, len(raw), col_nbytes):
                col = raw[off : off + col_nbytes]
                bits = "".join(_BITS_LE[b] for b in col)
                write(bits[:trim])
                write("\n")
            remaining -= take


def _decode_allele_nibbles(packed: bytes, n_chars: int) -> str:
    if int(n_chars) <= 0:
        return "N"
    pieces: list[str] = []
    remaining = int(n_chars)
    for byte in packed:
        pair = _ALLELE_2[int(byte)]
        if remaining >= 2:
            pieces.append(pair)
            remaining -= 2
        else:
            pieces.append(pair[0])
            remaining = 0
            break
    return "".join(pieces) if pieces else "N"


def _dump_legacy_bsite(path: Path) -> None:
    out = sys.stdout
    write = out.write
    with path.open("rb") as f:
        hdr = _read_exact(f, LEGACY_BSITE_HEADER_SIZE)
        if hdr[:8] != LEGACY_BSITE_MAGIC:
            raise RuntimeError(f"Invalid legacy BSITE magic in '{path}'.")
        version = struct.unpack("<H", hdr[8:10])[0]
        if int(version) != int(LEGACY_BSITE_VERSION):
            raise RuntimeError(f"Unsupported legacy BSITE version in '{path}': {version}.")
        n_sites = struct.unpack("<Q", hdr[12:20])[0]
        n_chrom = struct.unpack("<I", hdr[20:24])[0]
        dict_offset = struct.unpack("<Q", hdr[24:32])[0]
        file_size = path.stat().st_size
        if int(dict_offset) < LEGACY_BSITE_HEADER_SIZE or int(dict_offset) > int(file_size):
            raise RuntimeError(f"Invalid legacy BSITE dictionary offset in '{path}'.")

        body_offset = int(f.tell())
        f.seek(int(dict_offset))
        chrom_names: list[str] = []
        while len(chrom_names) < int(n_chrom):
            n_raw = _read_exact(f, 2)
            n = struct.unpack("<H", n_raw)[0]
            chrom_names.append(_read_exact(f, int(n)).decode("utf-8", errors="replace"))

        f.seek(body_offset)
        for i in range(int(n_sites)):
            raw = _read_exact(f, 13)
            chrom_code, strand, cm, bp = struct.unpack("<IBfi", raw)

            a0_len = struct.unpack("<H", _read_exact(f, 2))[0]
            a0_nbytes = 0 if int(a0_len) <= 0 else (int(a0_len) + 1) // 2
            a0 = _decode_allele_nibbles(_read_exact(f, a0_nbytes), int(a0_len))

            a1_len = struct.unpack("<H", _read_exact(f, 2))[0]
            a1_nbytes = 0 if int(a1_len) <= 0 else (int(a1_len) + 1) // 2
            a1 = _decode_allele_nibbles(_read_exact(f, a1_nbytes), int(a1_len))

            if int(f.tell()) > int(dict_offset):
                raise RuntimeError(f"Truncated legacy BSITE body in '{path}' at row {i + 1}.")
            if chrom_code < 0 or chrom_code >= len(chrom_names):
                raise RuntimeError(f"Invalid legacy BSITE chromosome code in '{path}': {chrom_code}.")
            chrom = chrom_names[chrom_code]
            strand_txt = "-" if int(strand) == 0 else "+"
            write(f"{chrom}\t{strand_txt}\t{cm}\t{bp}\t{a0}\t{a1}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx view",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx view -bin test.kmer/kmerge.bkmer | head",
                "jx view -bin test.kmer/kmerge.bsite | head",
                "jx view -bin test.kmer/kmerge.bsite | awk 'length($0) > 0' | head",
            ]
        ),
        description=(
            "View JanusX binary files as plain text for shell pipelines. "
            "Auto-detects `.bkmer`, kmerge `.bsite`, and legacy `.bsite`."
        ),
    )
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "-bin",
        "--bin",
        nargs="+",
        required=True,
        help="Input file(s): .bkmer or .bsite.",
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
            if kind == "bkmer":
                _dump_bkmer(p)
            elif kind == "kmer_bsite":
                _dump_kmer_bsite(p)
            elif kind == "legacy_bsite":
                _dump_legacy_bsite(p)
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
