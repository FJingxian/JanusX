# -*- coding: utf-8 -*-
"""
JanusX: Genotype format converter (gformat)

Supported inputs
----------------
- `-vcf/--vcf`    : VCF / VCF.GZ genotype file
- `-hmp/--hmp`    : HMP / HMP.GZ genotype file
- `-bfile/--bfile`: PLINK prefix (.bed/.bim/.fam)
- `-file/--file`  : text / npy genotype matrix or prefix

Supported outputs
-----------------
- `-fmt plink` : PLINK bed/bim/fam
- `-fmt vcf`   : bgzipped VCF
- `-fmt hmp`   : HapMap text (.hmp)
- `-fmt txt`   : `prefix.txt` + `prefix.id` + `prefix.site`
- `-fmt npy`   : `prefix.npy` + `prefix.id` + `prefix.site`

Notes
-----
- `-file` inputs must carry sibling `prefix.id`.
- `-file -> vcf/hmp/plink` requires real site metadata via `prefix.site` or `prefix.bim`.
- `txt/npy` outputs always write headerless `prefix.site` as four columns:
  `CHR POS REF ALT`.
"""

from __future__ import annotations

import os
import socket
import time
import re
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from janusx.gfreader import inspect_genotype_file, load_genotype_chunks, save_genotype_streaming, SiteInfo
from janusx.janusx import VcfChunkReader, HmpChunkReader, count_vcf_snps, count_hmp_snps
from janusx.gfreader.gfreader import _resolve_input, _process_txt_like_chunk

from ._common.config_render import emit_cli_configuration
from ._common.genoio import (
    GENOTYPE_TEXT_SUFFIXES,
    build_prefix_candidates,
    determine_genotype_source_from_args as determine_genotype_source,
    discover_id_sidecar_path,
    discover_site_path,
    write_npy_output,
    write_text_output,
)
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    format_output_display,
    format_path_for_display,
    ensure_plink_prefix_exists,
)
from ._common.status import CliStatus, log_success, stdout_is_tty
from ._common.genocache import configure_genotype_cache_from_out


def _normalize_chr_key(chrom: str) -> str:
    s = str(chrom).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    s = s.strip().upper()
    if s == "M":
        return "MT"
    return s


def _split_tokens(line: str) -> list[str]:
    return [x for x in re.split(r"[,\s]+", line.strip()) if x]


def _parse_int_token(token: str, name: str) -> int:
    t = str(token).strip()
    if t == "":
        raise ValueError(f"Empty integer token for {name}.")
    try:
        return int(float(t))
    except Exception as e:
        raise ValueError(f"Invalid integer token for {name}: {token}") from e


def _parse_site_token(token: str) -> tuple[str, int]:
    t = str(token).strip()
    if t == "":
        raise ValueError("Empty site token in --extract file.")
    if ":" in t:
        c, p = t.split(":", 1)
        return _normalize_chr_key(c), _parse_int_token(p, "site position")
    if "_" in t:
        c, p = t.split("_", 1)
        return _normalize_chr_key(c), _parse_int_token(p, "site position")
    raise ValueError(
        f"Unsupported site token '{token}'. Use CHR:POS / CHR_POS, or two columns: CHR POS."
    )


def _read_keep_samples(path: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = _split_tokens(s)
            if not tok:
                continue
            sid = str(tok[0]).strip()
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)
    if not out:
        raise ValueError(f"--keep file is empty or invalid: {path}")
    return out


def _read_extract_sites(path: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = _split_tokens(s)
            if not tok:
                continue
            if len(tok) == 1:
                key = _parse_site_token(tok[0])
            else:
                key = (_normalize_chr_key(tok[0]), _parse_int_token(tok[1], "site position"))
            if key not in seen:
                seen.add(key)
                out.append(key)
    if not out:
        raise ValueError(f"--extract file is empty or invalid: {path}")
    return out


def _read_extract_ranges(path: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = _split_tokens(s)
            if not tok:
                continue
            if len(tok) == 1:
                c, p = _parse_site_token(tok[0])
                out.append((c, p, p))
                continue
            if len(tok) == 2:
                c = _normalize_chr_key(tok[0])
                p = _parse_int_token(tok[1], "range position")
                out.append((c, p, p))
                continue
            c = _normalize_chr_key(tok[0])
            a = _parse_int_token(tok[1], "range start")
            b = _parse_int_token(tok[2], "range end")
            start, end = (a, b) if a <= b else (b, a)
            out.append((c, start, end))
    if not out:
        raise ValueError(f"--extract range file is empty or invalid: {path}")
    return out


def _parse_extract_arg(values: list[str] | None) -> tuple[str | None, str | None]:
    if not values:
        return None, None
    parts = [str(v).strip() for v in values if str(v).strip()]
    if len(parts) == 1:
        return "sites", parts[0]
    if len(parts) == 2 and parts[0].lower() == "range":
        return "range", parts[1]
    raise ValueError("Invalid --extract usage. Use '--extract <file>' or '--extract range <file>'.")


def _expand_chr_selector(tokens: list[str] | None) -> set[str]:
    if not tokens:
        return set()
    out: set[str] = set()
    for tok in tokens:
        for part in str(tok).split(","):
            p = part.strip()
            if not p:
                continue
            if "-" in p:
                left, right = p.split("-", 1)
                left = left.strip()
                right = right.strip()
                if left.isdigit() and right.isdigit():
                    a = int(left)
                    b = int(right)
                    if a > b:
                        raise ValueError(f"Invalid --chr range: {p}")
                    for k in range(a, b + 1):
                        out.add(_normalize_chr_key(str(k)))
                    continue
            out.add(_normalize_chr_key(p))
    return out


@dataclass
class SiteFilterSpec:
    chr_keys: set[str] | None = None
    bp_min: int | None = None
    bp_max: int | None = None
    exact_sites: set[tuple[str, int]] | None = None
    ranges: list[tuple[str, int, int]] | None = None

    def active(self) -> bool:
        return bool(self.chr_keys) or self.bp_min is not None or self.bp_max is not None or bool(self.exact_sites) or bool(self.ranges)


def _site_passes(site: SiteInfo, flt: SiteFilterSpec) -> bool:
    c = _normalize_chr_key(str(site.chrom))
    p = int(site.pos)
    if flt.chr_keys and c not in flt.chr_keys:
        return False
    if flt.bp_min is not None and p < int(flt.bp_min):
        return False
    if flt.bp_max is not None and p > int(flt.bp_max):
        return False
    if flt.exact_sites and (c, p) not in flt.exact_sites:
        return False
    if flt.ranges:
        matched = False
        for rc, rs, re in flt.ranges:
            if c == rc and int(rs) <= p <= int(re):
                matched = True
                break
        if not matched:
            return False
    return True


def _filter_chunk_by_site(
    geno: np.ndarray,
    sites: list[SiteInfo],
    flt: SiteFilterSpec,
) -> tuple[np.ndarray, list[SiteInfo]]:
    if not flt.active():
        return geno, sites
    keep_idx: list[int] = []
    keep_sites: list[SiteInfo] = []
    for i, s in enumerate(sites):
        if _site_passes(s, flt):
            keep_idx.append(i)
            keep_sites.append(s)
    if len(keep_idx) == len(sites):
        return geno, sites
    if not keep_idx:
        return np.empty((0, geno.shape[1]), dtype=np.float32), []
    return np.asarray(geno[keep_idx, :], dtype=np.float32), keep_sites


def _has_real_file_sites(file_arg: str) -> bool:
    raw_prefix, _ = build_prefix_candidates(file_arg, text_suffixes=GENOTYPE_TEXT_SUFFIXES)
    prefixes = [raw_prefix]
    return discover_site_path(prefixes) is not None


def _resolve_output_target(args) -> tuple[str, str, str]:
    _, prefix = determine_genotype_source(args)
    outdir = os.path.normpath(str(args.out or "."))
    fmt = str(args.format).lower()
    base = os.path.join(outdir, prefix)
    if fmt == "vcf":
        return fmt, base, f"{base}.vcf.gz"
    if fmt == "hmp":
        return fmt, base, f"{base}.hmp"
    if fmt == "txt":
        return fmt, base, f"{base}.txt"
    if fmt == "npy":
        return fmt, base, f"{base}.npy"
    if fmt == "plink":
        return fmt, base, base
    raise ValueError("Unsupported output format. Use one of: plink, vcf, hmp, txt, npy.")

def _iter_filtered_chunks(
    chunks: Iterator[tuple[np.ndarray, list[SiteInfo]]],
    flt: SiteFilterSpec,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    for geno, sites in chunks:
        g2, s2 = _filter_chunk_by_site(np.asarray(geno, dtype=np.float32), list(sites), flt)
        if len(s2) == 0:
            continue
        yield g2, s2


def _count_sites_from_chunks(
    chunks: Iterator[tuple[np.ndarray, list[SiteInfo]]],
) -> int:
    cnt = 0
    for block, _sites in chunks:
        cnt += int(np.asarray(block).shape[0])
    return int(cnt)


def _open_text_auto(path: str):
    low = str(path).lower()
    if low.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _read_id_sidecar(id_path: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    with open(id_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = _split_tokens(s)
            if not tok:
                continue
            sid = str(tok[0]).strip()
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)
    if len(out) == 0:
        raise ValueError(f"ID sidecar is empty or invalid: {id_path}")
    return out


def _count_text_matrix_rows(matrix_path: str) -> int:
    n = 0
    with _open_text_auto(matrix_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            n += 1
    return int(n)


def _resolve_txt_direct_paths(file_arg: str) -> tuple[str, str, str | None]:
    raw_prefix, _ = build_prefix_candidates(file_arg, text_suffixes=GENOTYPE_TEXT_SUFFIXES)
    p = str(file_arg)
    low = p.lower()

    matrix_path = None
    if low.endswith((".txt", ".tsv", ".csv")) and Path(p).is_file():
        matrix_path = p
    else:
        for ext in (".txt", ".tsv", ".csv"):
            cand = f"{raw_prefix}{ext}"
            if Path(cand).is_file():
                matrix_path = cand
                break
    if matrix_path is None:
        raise ValueError(
            f"Cannot find FILE text matrix for direct conversion: {file_arg}"
        )

    id_path = discover_id_sidecar_path(matrix_path, [raw_prefix])
    if id_path is None:
        raise ValueError(
            f"Cannot find ID sidecar for text matrix: {matrix_path} (expected {raw_prefix}.id or {matrix_path}.id)"
        )
    site_path = discover_site_path([raw_prefix])
    return matrix_path, id_path, site_path


def _parse_site_line(site_line: str, *, is_bim: bool, row_idx: int, site_path: str) -> SiteInfo:
    tok = _split_tokens(site_line)
    if is_bim:
        if len(tok) < 6:
            raise ValueError(
                f"Invalid BIM row at {site_path}:{row_idx} (need >=6 columns)."
            )
        chrom = str(tok[0])
        pos = int(float(tok[3]))
        ref = str(tok[4])
        alt = str(tok[5])
    else:
        if len(tok) < 4:
            raise ValueError(
                f"Invalid SITE row at {site_path}:{row_idx} (need >=4 columns: CHR POS REF ALT)."
            )
        chrom = str(tok[0])
        pos = int(float(tok[1]))
        ref = str(tok[2])
        alt = str(tok[3])
    return SiteInfo(chrom, int(pos), ref, alt)


def _parse_text_row_values(line: str, *, expected_cols: int, row_idx: int, matrix_path: str) -> np.ndarray:
    tok = _split_tokens(line)
    if len(tok) != int(expected_cols):
        raise ValueError(
            f"Text matrix column mismatch at {matrix_path}:{row_idx}: expected {expected_cols}, got {len(tok)}."
        )
    vals = np.empty(int(expected_cols), dtype=np.float32)
    for i, t in enumerate(tok):
        tt = str(t).strip()
        up = tt.upper()
        if up in {"NA", "NAN", "."}:
            vals[i] = np.float32(-9.0)
            continue
        try:
            vals[i] = np.float32(float(tt))
        except Exception as e:
            raise ValueError(
                f"Invalid numeric token at {matrix_path}:{row_idx}, col={i + 1}: {tt}"
            ) from e
    return vals


def _iter_vcf_direct_chunks(
    vcf_path: str,
    *,
    chunk_size: int,
    sample_ids: list[str] | None,
    maf_threshold: float,
    max_missing_rate: float,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    reader = VcfChunkReader(
        str(vcf_path),
        float(maf_threshold),
        float(max_missing_rate),
        False,
        sample_ids,
        None,
        "add",
        0.02,
    )
    while True:
        out = reader.next_chunk(int(chunk_size))
        if out is None:
            break
        geno, sites = out
        yield np.asarray(geno, dtype=np.float32), list(sites)


def _iter_hmp_direct_chunks(
    hmp_path: str,
    *,
    chunk_size: int,
    sample_ids: list[str] | None,
    maf_threshold: float,
    max_missing_rate: float,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    reader = HmpChunkReader(
        str(hmp_path),
        float(maf_threshold),
        float(max_missing_rate),
        False,
        sample_ids,
        None,
        "add",
        0.02,
    )
    while True:
        out = reader.next_chunk(int(chunk_size))
        if out is None:
            break
        geno, sites = out
        yield np.asarray(geno, dtype=np.float32), list(sites)


def _iter_txt_direct_chunks(
    matrix_path: str,
    *,
    id_path: str,
    site_path: str | None,
    chunk_size: int,
    sample_ids: list[str] | None,
    maf_threshold: float,
    max_missing_rate: float,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    all_sample_ids = _read_id_sidecar(id_path)
    sample_index_map = {sid: i for i, sid in enumerate(all_sample_ids)}
    if sample_ids is None:
        selected_indices = np.arange(len(all_sample_ids), dtype=np.int64)
    else:
        idx: list[int] = []
        for sid in sample_ids:
            if sid not in sample_index_map:
                raise ValueError(f"Sample ID not found in {id_path}: {sid}")
            idx.append(int(sample_index_map[sid]))
        selected_indices = np.asarray(idx, dtype=np.int64)
    n_expected = int(len(all_sample_ids))

    site_fh = None
    is_bim = False
    site_row_idx = 0
    if site_path is not None:
        site_fh = open(site_path, "r", encoding="utf-8", errors="replace")
        is_bim = str(site_path).lower().endswith(".bim")

    def _next_site(matrix_row_idx: int) -> SiteInfo:
        nonlocal site_row_idx
        if site_fh is None:
            return SiteInfo("N", int(matrix_row_idx), "N", "N")
        while True:
            line = site_fh.readline()
            if line == "":
                raise ValueError(
                    f"Site metadata ended early at row {matrix_row_idx} in {site_path}."
                )
            site_row_idx += 1
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            return _parse_site_line(
                s,
                is_bim=bool(is_bim),
                row_idx=site_row_idx,
                site_path=str(site_path),
            )

    rows: list[np.ndarray] = []
    sites: list[SiteInfo] = []
    matrix_row_idx = 0
    try:
        with _open_text_auto(matrix_path) as f:
            for line_idx, line in enumerate(f, start=1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                matrix_row_idx += 1
                row = _parse_text_row_values(
                    s,
                    expected_cols=n_expected,
                    row_idx=line_idx,
                    matrix_path=matrix_path,
                )
                if selected_indices.size != n_expected:
                    row = np.asarray(row[selected_indices], dtype=np.float32)
                rows.append(row)
                sites.append(_next_site(matrix_row_idx))
                if len(rows) >= int(chunk_size):
                    arr = np.asarray(rows, dtype=np.float32)
                    arr, sites_out = _process_txt_like_chunk(
                        arr,
                        sites,
                        maf=float(maf_threshold),
                        missing_rate=float(max_missing_rate),
                        impute=False,
                        model="add",
                        het=0.02,
                    )
                    if arr.shape[0] > 0:
                        yield arr, list(sites_out)
                    rows = []
                    sites = []
        if len(rows) > 0:
            arr = np.asarray(rows, dtype=np.float32)
            arr, sites_out = _process_txt_like_chunk(
                arr,
                sites,
                maf=float(maf_threshold),
                missing_rate=float(max_missing_rate),
                impute=False,
                model="add",
                het=0.02,
            )
            if arr.shape[0] > 0:
                yield arr, list(sites_out)
    finally:
        if site_fh is not None:
            site_fh.close()


def _count_selected_sites(
    gfile: str,
    *,
    sample_ids: list[str] | None,
    snp_sites: list[tuple[str, int]] | None,
    bim_range: tuple[str, int, int] | None,
    site_filter: SiteFilterSpec,
    maf_threshold: float,
    max_missing_rate: float,
) -> int:
    cnt = 0
    chunks = load_genotype_chunks(
        gfile,
        chunk_size=50_000,
        maf=float(maf_threshold),
        missing_rate=float(max_missing_rate),
        impute=False,
        sample_ids=sample_ids,
        snp_sites=snp_sites,
        bim_range=bim_range,
    )
    if site_filter.active():
        chunks = _iter_filtered_chunks(chunks, site_filter)
    for block, _sites in chunks:
        cnt += int(np.asarray(block).shape[0])
    return int(cnt)


def build_parser() -> CliArgumentParser:
    parser = CliArgumentParser(
        prog="jx gformat",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx gformat -vcf geno.vcf.gz -fmt npy",
                "jx gformat -hmp geno.hmp.gz -fmt plink",
                "jx gformat -bfile geno_prefix -fmt txt -o outdir -prefix panel",
                "jx gformat -file geno_prefix -fmt vcf",
            ]
        ),
    )

    req = parser.add_argument_group("Input Arguments")
    src = req.add_mutually_exclusive_group(required=True)
    src.add_argument("-vcf", "--vcf", type=str, help="Input genotype file in VCF format (.vcf or .vcf.gz).")
    src.add_argument("-hmp", "--hmp", type=str, help="Input genotype file in HMP format (.hmp or .hmp.gz).")
    src.add_argument("-bfile", "--bfile", type=str, help="Input genotype in PLINK binary format (prefix for .bed/.bim/.fam).")
    src.add_argument(
        "-file",
        "--file",
        type=str,
        help=(
            "Input genotype numeric matrix (.txt/.tsv/.csv/.npy) or prefix. "
            "Requires sibling prefix.id. Optional site metadata: prefix.site or prefix.bim."
        ),
    )

    opt = parser.add_argument_group("Optional Arguments")
    opt.add_argument(
        "-fmt",
        "--fmt",
        dest="format",
        choices=["plink", "vcf", "hmp", "txt", "npy"],
        default="npy",
        help="Output genotype format: plink, vcf, hmp, txt, npy (default: npy).",
    )
    opt.add_argument(
        "-o",
        "--out",
        default=".",
        help="Output directory for converted results (default: %(default)s).",
    )
    opt.add_argument(
        "-prefix",
        "--prefix",
        default=None,
        help="Output prefix (default: input basename).",
    )
    opt.add_argument(
        "-maf",
        "--maf",
        type=float,
        default=0.0,
        help=(
            "Exclude variants with minor allele frequency lower than a threshold "
            "(default: %(default)s; no filtering)."
        ),
    )
    opt.add_argument(
        "-geno",
        "--geno",
        type=float,
        default=1.0,
        help=(
            "Exclude variants with missing call frequencies greater than a threshold "
            "(default: %(default)s; no filtering)."
        ),
    )
    opt.add_argument(
        "-keep",
        "--keep",
        type=str,
        default=None,
        help="Keep only samples listed in file (one sample ID per line, no header).",
    )
    opt.add_argument(
        "-extract",
        "--extract",
        nargs="+",
        default=None,
        metavar=("MODE_OR_FILE", "FILE"),
        help=(
            "Keep only listed variants. Use '--extract <file>' for site list "
            "(CHR POS or CHR:POS/CHR_POS), or '--extract range <file>' "
            "for range list (CHR START END). No header."
        ),
    )
    opt.add_argument(
        "-chr",
        "--chr",
        dest="chr_filter",
        nargs="+",
        default=None,
        help=(
            "Keep only variants on selected chromosome(s). Supports spaces/commas "
            "and numeric ranges, e.g. '--chr 1-4,22,XY'."
        ),
    )
    bp_help = (
        "Physical position range filter. Must be used with a single chromosome in --chr. "
        "Both --from-bp and --to-bp are inclusive."
    )
    opt.add_argument("-from-bp", "--from-bp", type=int, default=None, help=bp_help)
    opt.add_argument("-to-bp", "--to-bp", type=int, default=None, help=bp_help)
    return parser


def main() -> None:
    parser = build_parser()
    args, extras = parser.parse_known_args()
    if extras:
        parser.error("unrecognized arguments: " + " ".join(extras))
    if not (0.0 <= float(args.maf) <= 0.5):
        parser.error("-maf must be within [0, 0.5].")
    if not (0.0 <= float(args.geno) <= 1.0):
        parser.error("-geno must be within [0, 1.0].")

    gfile, default_prefix = determine_genotype_source(args)
    if args.prefix is None:
        args.prefix = default_prefix
    out_fmt, out_prefix, out_path = _resolve_output_target(args)
    output_display = format_output_display(out_fmt, out_prefix, out_path)

    os.makedirs(str(args.out), exist_ok=True, mode=0o755)
    configure_genotype_cache_from_out(str(args.out))
    log_path = f"{out_prefix}.gformat.log"
    logger = setup_logging(log_path)
    status_enabled = stdout_is_tty()

    extract_mode, extract_file = _parse_extract_arg(args.extract)
    chr_keys = _expand_chr_selector(args.chr_filter)
    if args.from_bp is not None and args.to_bp is not None and int(args.from_bp) > int(args.to_bp):
        raise ValueError("--from-bp cannot be greater than --to-bp.")
    if (args.from_bp is not None or args.to_bp is not None) and len(chr_keys) != 1:
        raise ValueError("--from-bp/--to-bp requires a single chromosome in --chr.")

    emit_cli_configuration(
        logger,
        app_title="JanusX - gformat",
        config_title="GFORMAT CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "General",
                [
                    ("Genotype file", gfile),
                    ("Output format", out_fmt),
                    ("Keep samples", str(args.keep or "None")),
                    ("Extract", f"{extract_mode or 'None'}:{extract_file or 'None'}"),
                    ("Chr filter", ",".join(sorted(chr_keys)) if chr_keys else "None"),
                    ("BP range", f"{args.from_bp if args.from_bp is not None else '-'}..{args.to_bp if args.to_bp is not None else '-'}"),
                    ("MAF threshold", float(args.maf)),
                    ("Miss threshold", float(args.geno)),
                ],
            )
        ],
        footer_rows=[
            ("Output", output_display),
        ],
        line_max_chars=60,
    )

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    if args.keep:
        checks.append(ensure_file_exists(logger, str(args.keep), "--keep file"))
    if extract_file:
        checks.append(ensure_file_exists(logger, str(extract_file), "--extract file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    if args.file and out_fmt in {"vcf", "hmp", "plink"} and (not _has_real_file_sites(gfile)):
        raise ValueError(
            f"{out_fmt.upper()} output from -file input requires real site metadata. "
            "Please provide a matching prefix.site or prefix.bim sidecar."
        )

    source_kind, _source_prefix, source_path = _resolve_input(gfile)
    use_direct_no_cache = bool(source_kind in {"vcf", "hmp", "txt"})
    txt_matrix_path: str | None = None
    txt_id_path: str | None = None
    txt_site_path: str | None = None

    inspect_desc = "Inspecting genotype metadata..."
    if not status_enabled:
        logger.info(inspect_desc)
    with CliStatus(
        inspect_desc,
        enabled=status_enabled,
    ) as task:
        try:
            if use_direct_no_cache and source_kind == "vcf":
                if source_path is None:
                    raise ValueError(f"VCF source path not found: {gfile}")
                vcf_reader = VcfChunkReader(
                    str(source_path),
                    0.0,
                    1.0,
                    False,
                    None,
                    None,
                    "add",
                    0.02,
                )
                sample_ids = [str(x) for x in vcf_reader.sample_ids]
                n_sites = int(count_vcf_snps(str(source_path)))
            elif use_direct_no_cache and source_kind == "hmp":
                if source_path is None:
                    raise ValueError(f"HMP source path not found: {gfile}")
                hmp_reader = HmpChunkReader(
                    str(source_path),
                    0.0,
                    1.0,
                    False,
                    None,
                    None,
                    "add",
                    0.02,
                )
                sample_ids = [str(x) for x in hmp_reader.sample_ids]
                n_sites = int(count_hmp_snps(str(source_path)))
            elif use_direct_no_cache and source_kind == "txt":
                txt_arg = str(args.file or gfile)
                txt_matrix_path, txt_id_path, txt_site_path = _resolve_txt_direct_paths(txt_arg)
                sample_ids = _read_id_sidecar(txt_id_path)
                n_sites = _count_text_matrix_rows(txt_matrix_path)
            else:
                sample_ids, n_sites = inspect_genotype_file(gfile)
        except Exception:
            task.fail("Inspecting genotype metadata ...Failed")
            raise
        task.complete(
            f"Inspecting genotype metadata ...Finished (n={len(sample_ids)}, nSNP={int(n_sites)})"
        )
    if not status_enabled:
        logger.info(
            f"Inspecting genotype metadata ...Finished (n={len(sample_ids)}, nSNP={int(n_sites)})"
        )
    keep_sample_ids: list[str] | None = None
    if args.keep:
        keep_list = _read_keep_samples(str(args.keep))
        keep_set = set(keep_list)
        keep_sample_ids = [str(s) for s in sample_ids if str(s) in keep_set]
        if len(keep_sample_ids) == 0:
            miss_preview = ", ".join(keep_list[:10])
            raise ValueError(f"--keep selected 0 samples. Missing examples: {miss_preview}")

    extract_sites: list[tuple[str, int]] | None = None
    extract_ranges: list[tuple[str, int, int]] | None = None
    if extract_mode == "sites":
        extract_sites = _read_extract_sites(str(extract_file))
    elif extract_mode == "range":
        extract_ranges = _read_extract_ranges(str(extract_file))

    # Reader-side pushdown filters when possible.
    push_snp_sites: list[tuple[str, int]] | None = None
    push_bim_range: tuple[str, int, int] | None = None
    post_filter = SiteFilterSpec()

    if chr_keys:
        post_filter.chr_keys = set(chr_keys)
    if args.from_bp is not None:
        post_filter.bp_min = int(args.from_bp)
    if args.to_bp is not None:
        post_filter.bp_max = int(args.to_bp)

    if extract_sites:
        push_snp_sites = list(extract_sites)
    if extract_ranges:
        if len(extract_ranges) == 1 and not post_filter.active():
            push_bim_range = extract_ranges[0]
        else:
            post_filter.ranges = list(extract_ranges)

    # Optimize simple --chr + --from-bp/--to-bp into a direct bim_range pushdown.
    if (
        push_snp_sites is None
        and push_bim_range is None
        and extract_ranges is None
        and post_filter.chr_keys
        and len(post_filter.chr_keys) == 1
        and (post_filter.bp_min is not None or post_filter.bp_max is not None)
    ):
        c = next(iter(post_filter.chr_keys))
        start = int(post_filter.bp_min) if post_filter.bp_min is not None else 0
        end = int(post_filter.bp_max) if post_filter.bp_max is not None else 2_147_483_647
        push_bim_range = (c, start, end)
        post_filter = SiteFilterSpec()

    def _make_chunks() -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
        if use_direct_no_cache and source_kind == "vcf":
            if source_path is None:
                raise ValueError(f"VCF source path not found: {gfile}")
            if push_snp_sites is not None:
                post_filter.exact_sites = set(push_snp_sites)
            if push_bim_range is not None:
                post_filter.ranges = [push_bim_range]
            c = _iter_vcf_direct_chunks(
                str(source_path),
                chunk_size=50_000,
                sample_ids=keep_sample_ids,
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
            )
        elif use_direct_no_cache and source_kind == "hmp":
            if source_path is None:
                raise ValueError(f"HMP source path not found: {gfile}")
            if push_snp_sites is not None:
                post_filter.exact_sites = set(push_snp_sites)
            if push_bim_range is not None:
                post_filter.ranges = [push_bim_range]
            c = _iter_hmp_direct_chunks(
                str(source_path),
                chunk_size=50_000,
                sample_ids=keep_sample_ids,
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
            )
        elif use_direct_no_cache and source_kind == "txt":
            if txt_matrix_path is None or txt_id_path is None:
                raise ValueError("Internal error: text input paths are unresolved.")
            if push_snp_sites is not None:
                post_filter.exact_sites = set(push_snp_sites)
            if push_bim_range is not None:
                post_filter.ranges = [push_bim_range]
            c = _iter_txt_direct_chunks(
                str(txt_matrix_path),
                id_path=str(txt_id_path),
                site_path=txt_site_path,
                chunk_size=50_000,
                sample_ids=keep_sample_ids,
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
            )
        else:
            c = load_genotype_chunks(
                gfile,
                chunk_size=50_000,
                maf=float(args.maf),
                missing_rate=float(args.geno),
                impute=False,
                sample_ids=keep_sample_ids,
                snp_sites=push_snp_sites,
                bim_range=push_bim_range,
            )
        if post_filter.active():
            c = _iter_filtered_chunks(c, post_filter)
        return c

    need_selected_count = (
        bool(push_snp_sites)
        or (push_bim_range is not None)
        or post_filter.active()
        or (float(args.maf) > 0.0)
        or (float(args.geno) < 1.0)
        or (use_direct_no_cache and out_fmt in {"npy", "txt"})
    )
    selected_n_sites = int(n_sites)
    if need_selected_count:
        selected_n_sites = _count_sites_from_chunks(_make_chunks())
        if selected_n_sites <= 0:
            raise ValueError("All variants were filtered out by filters (-extract/-chr/-from-bp/-to-bp/-maf/-geno).")

    out_sample_ids = keep_sample_ids if keep_sample_ids is not None else [str(s) for s in sample_ids]
    print(f"Genotype source: {format_path_for_display(gfile)}")
    print(f"Samples: {len(out_sample_ids)}, sites: {selected_n_sites}")

    t0 = time.time()
    if out_fmt == "vcf":
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_genotype_streaming(
            out_path,
            out_sample_ids,
            _make_chunks(),
            fmt="vcf",
            total_snps=int(selected_n_sites),
            desc="Writing VCF",
        )
    elif out_fmt == "hmp":
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_genotype_streaming(
            out_path,
            out_sample_ids,
            _make_chunks(),
            fmt="hmp",
            total_snps=int(selected_n_sites),
            desc="Writing HMP",
        )
    elif out_fmt == "plink":
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
        save_genotype_streaming(
            out_prefix,
            out_sample_ids,
            _make_chunks(),
            fmt="plink",
            total_snps=int(selected_n_sites),
            desc="Writing PLINK",
        )
    elif out_fmt == "txt":
        write_text_output(out_path, out_sample_ids, _make_chunks(), total_sites=int(selected_n_sites))
    else:
        write_npy_output(out_path, out_sample_ids, _make_chunks(), total_sites=int(selected_n_sites))

    log_success(logger, f"Format conversion completed in {time.time() - t0:.2f} s")
    log_success(logger, f"Output written: {output_display}")


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
