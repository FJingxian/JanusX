# -*- coding: utf-8 -*-
"""
JanusX: Genotype format converter (gformat)

Supported inputs
----------------
- `-vcf/--vcf`    : VCF / VCF.GZ genotype file
- `-bfile/--bfile`: PLINK prefix (.bed/.bim/.fam)
- `-file/--file`  : text / npy genotype matrix or prefix

Supported outputs
-----------------
- `-format plink` : PLINK bed/bim/fam
- `-format vcf`   : bgzipped VCF
- `-format txt`   : `prefix.txt` + `prefix.id` + `prefix.site`
- `-format npy`   : `prefix.npy` + `prefix.id` + `prefix.site`

Notes
-----
- `-file` inputs must carry sibling `prefix.id`.
- `-file -> vcf/plink` requires real site metadata via `prefix.site` or `prefix.bim`.
- `txt/npy` outputs always write headerless `prefix.site` as four columns:
  `CHR POS REF ALT`.
"""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from janusx.gfreader import inspect_genotype_file, load_genotype_chunks, save_genotype_streaming, SiteInfo

from ._common.config_render import emit_cli_configuration
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
)
from ._common.status import CliStatus


GENOTYPE_TEXT_SUFFIXES = (".txt", ".tsv", ".csv")
GENOTYPE_SUFFIXES = GENOTYPE_TEXT_SUFFIXES + (".npy",)


def _strip_known_suffix(path: str) -> str:
    p = str(path)
    low = p.lower()
    for ext in (".vcf.gz", ".vcf", ".bed", ".bim", ".fam", ".txt", ".tsv", ".csv", ".npy"):
        if low.endswith(ext):
            return p[: -len(ext)]
    return p


def _strip_default_prefix_suffix(name: str) -> str:
    base = os.path.basename(str(name).rstrip("/\\"))
    low = base.lower()
    if low.endswith(".vcf.gz"):
        return base[: -len(".vcf.gz")]
    for ext in (".vcf", ".txt", ".tsv", ".csv", ".npy"):
        if low.endswith(ext):
            return base[: -len(ext)]
    return base


def _build_prefix_candidates(file_arg: str) -> tuple[str, list[str]]:
    raw = Path(file_arg).expanduser()

    if raw.suffix.lower() in GENOTYPE_TEXT_SUFFIXES:
        raw_prefix = _strip_known_suffix(str(raw))
        cache_prefix = str(Path(raw_prefix).parent / f"~{Path(raw_prefix).name}")
        return raw_prefix, [raw_prefix, cache_prefix]

    if raw.suffix.lower() == ".npy":
        raw_prefix = _strip_known_suffix(str(raw))
        noncache_prefix = str(Path(raw_prefix).parent / Path(raw_prefix).name.lstrip("~"))
        return noncache_prefix, [noncache_prefix, raw_prefix]

    raw_prefix = str(raw)
    cache_prefix = str(raw.parent / f"~{raw.name}")
    return raw_prefix, [raw_prefix, cache_prefix]


def _has_real_file_sites(file_arg: str) -> bool:
    _, prefixes = _build_prefix_candidates(file_arg)
    for prefix in prefixes:
        for cand in (
            f"{prefix}.site",
            f"{prefix}.site.tsv",
            f"{prefix}.site.txt",
            f"{prefix}.site.csv",
            f"{prefix}.sites.tsv",
            f"{prefix}.sites.txt",
            f"{prefix}.sites.csv",
            f"{prefix}.bim",
        ):
            if os.path.isfile(cand):
                return True
    return False


def determine_genotype_source(args) -> tuple[str, str]:
    if args.vcf:
        gfile = args.vcf
        prefix = _strip_default_prefix_suffix(gfile)
    elif args.file:
        gfile = args.file
        prefix = _strip_default_prefix_suffix(gfile)
    elif args.bfile:
        gfile = args.bfile
        prefix = os.path.basename(gfile.rstrip("/\\"))
    else:
        raise ValueError("No genotype input specified. Use -vcf, -file or -bfile.")

    if args.prefix is not None:
        prefix = str(args.prefix)

    return gfile.replace("\\", "/"), prefix


def _resolve_output_target(args) -> tuple[str, str, str]:
    _, prefix = determine_genotype_source(args)
    outdir = str(args.out or ".").replace("\\", "/")
    outdir = outdir.rstrip("/") or "."
    fmt = str(args.format).lower()
    base = f"{outdir}/{prefix}".replace("//", "/")
    if fmt == "vcf":
        return fmt, base, f"{base}.vcf.gz"
    if fmt == "txt":
        return fmt, base, f"{base}.txt"
    if fmt == "npy":
        return fmt, base, f"{base}.npy"
    if fmt == "plink":
        return fmt, base, base
    raise ValueError("Unsupported output format. Use one of: plink, vcf, txt, npy.")


def _write_site_file(handle, sites: Sequence[SiteInfo]) -> None:
    for site in sites:
        chrom = str(site.chrom)
        pos = int(site.pos)
        ref = str(site.ref_allele)
        alt = str(site.alt_allele)
        handle.write(f"{chrom}\t{pos}\t{ref}\t{alt}\n")


def _write_id_file(path: str, sample_ids: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as fid:
        fid.write("\n".join(map(str, sample_ids)) + "\n")


def _write_text_output(
    out_path: str,
    sample_ids: Sequence[str],
    chunks: Iterator[tuple[np.ndarray, list[SiteInfo]]],
    *,
    total_sites: int,
) -> None:
    prefix = _strip_known_suffix(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _write_id_file(f"{prefix}.id", sample_ids)
    written = 0
    with open(out_path, "w", encoding="utf-8") as fw, open(f"{prefix}.site", "w", encoding="utf-8") as fsite:
        for block, sites in chunks:
            np.savetxt(fw, np.asarray(block, dtype=np.float32), fmt="%.6g", delimiter="\t")
            _write_site_file(fsite, sites)
            written += int(block.shape[0])
    if written != int(total_sites):
        raise RuntimeError(f"Written site count mismatch for text output: {written} vs {total_sites}")


def _write_npy_output(
    out_path: str,
    sample_ids: Sequence[str],
    chunks: Iterator[tuple[np.ndarray, list[SiteInfo]]],
    *,
    total_sites: int,
) -> None:
    prefix = _strip_known_suffix(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_mm = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(total_sites), len(sample_ids)),
    )
    offset = 0
    _write_id_file(f"{prefix}.id", sample_ids)
    with open(f"{prefix}.site", "w", encoding="utf-8") as fsite:
        for block, sites in chunks:
            n_rows = int(block.shape[0])
            out_mm[offset : offset + n_rows, :] = np.asarray(block, dtype=np.float32)
            _write_site_file(fsite, sites)
            offset += n_rows
    out_mm.flush()
    if offset != int(total_sites):
        raise RuntimeError(f"Written site count mismatch for npy output: {offset} vs {total_sites}")


def build_parser() -> CliArgumentParser:
    parser = CliArgumentParser(
        prog="jx gformat",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx gformat -vcf geno.vcf.gz -format npy",
                "jx gformat -bfile geno_prefix -format txt -o outdir -prefix panel",
                "jx gformat -file geno_prefix -format vcf",
            ]
        ),
    )

    req = parser.add_argument_group("Input Arguments")
    src = req.add_mutually_exclusive_group(required=True)
    src.add_argument("-vcf", "--vcf", type=str, help="Input genotype file in VCF format (.vcf or .vcf.gz).")
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
        "-format",
        "--format",
        choices=["plink", "vcf", "txt", "npy"],
        default="npy",
        help="Output genotype format: plink, vcf, txt, npy (default: npy).",
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
    return parser


def main() -> None:
    parser = build_parser()
    args, extras = parser.parse_known_args()
    if extras:
        parser.error("unrecognized arguments: " + " ".join(extras))

    gfile, default_prefix = determine_genotype_source(args)
    if args.prefix is None:
        args.prefix = default_prefix
    out_fmt, out_prefix, out_path = _resolve_output_target(args)

    os.makedirs(str(args.out), exist_ok=True, mode=0o755)
    log_path = f"{out_prefix}.gformat.log"
    logger = setup_logging(log_path)

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
                ],
            )
        ],
        footer_rows=[
            ("Output prefix", out_prefix),
            ("Output", out_path),
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
    if not ensure_all_true(checks):
        raise SystemExit(1)

    if args.file and out_fmt in {"vcf", "plink"} and (not _has_real_file_sites(gfile)):
        raise ValueError(
            f"{out_fmt.upper()} output from -file input requires real site metadata. "
            "Please provide a matching prefix.site or prefix.bim sidecar."
        )

    sample_ids, n_sites = inspect_genotype_file(gfile)
    print(f"Genotype source: {gfile}")
    print(f"Samples: {len(sample_ids)}, sites: {n_sites}")
    print(f"Output prefix: {out_prefix}")
    print(f"Output: {out_path} ({out_fmt})")

    chunks = load_genotype_chunks(
        gfile,
        chunk_size=50_000,
        maf=0.0,
        missing_rate=1.0,
        impute=False,
    )

    t0 = time.time()
    with CliStatus("Converting genotype format...", enabled=bool(getattr(os.sys.stdout, "isatty", lambda: False)())) as task:
        try:
            if out_fmt == "vcf":
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                save_genotype_streaming(
                    out_path,
                    sample_ids,
                    chunks,
                    fmt="vcf",
                    total_snps=int(n_sites),
                    desc="Writing VCF",
                )
            elif out_fmt == "plink":
                Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
                save_genotype_streaming(
                    out_prefix,
                    sample_ids,
                    chunks,
                    fmt="plink",
                    total_snps=int(n_sites),
                    desc="Writing PLINK",
                )
            elif out_fmt == "txt":
                _write_text_output(out_path, sample_ids, chunks, total_sites=int(n_sites))
            else:
                _write_npy_output(out_path, sample_ids, chunks, total_sites=int(n_sites))
        except Exception:
            task.fail("Converting genotype format ...Failed")
            raise
        task.complete("Converting genotype format ...Finished")

    logger.info(f"Format conversion completed in {time.time() - t0:.2f} s")
    logger.info(f"Output written: {out_path}")


if __name__ == "__main__":
    main()
