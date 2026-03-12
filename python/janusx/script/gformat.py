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
- `-fmt plink` : PLINK bed/bim/fam
- `-fmt vcf`   : bgzipped VCF
- `-fmt txt`   : `prefix.txt` + `prefix.id` + `prefix.site`
- `-fmt npy`   : `prefix.npy` + `prefix.id` + `prefix.site`

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
import re
from dataclasses import dataclass
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


def _iter_filtered_chunks(
    chunks: Iterator[tuple[np.ndarray, list[SiteInfo]]],
    flt: SiteFilterSpec,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    for geno, sites in chunks:
        g2, s2 = _filter_chunk_by_site(np.asarray(geno, dtype=np.float32), list(sites), flt)
        if len(s2) == 0:
            continue
        yield g2, s2


def _count_selected_sites(
    gfile: str,
    *,
    sample_ids: list[str] | None,
    snp_sites: list[tuple[str, int]] | None,
    bim_range: tuple[str, int, int] | None,
    site_filter: SiteFilterSpec,
) -> int:
    cnt = 0
    chunks = load_genotype_chunks(
        gfile,
        chunk_size=50_000,
        maf=0.0,
        missing_rate=1.0,
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
                "jx gformat -bfile geno_prefix -fmt txt -o outdir -prefix panel",
                "jx gformat -file geno_prefix -fmt vcf",
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
        "-fmt",
        "--fmt",
        dest="format",
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
    opt.add_argument(
        "--keep",
        type=str,
        default=None,
        help="Keep only samples listed in file (one sample ID per line, no header).",
    )
    opt.add_argument(
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
    opt.add_argument("--from-bp", type=int, default=None, help=bp_help)
    opt.add_argument("--to-bp", type=int, default=None, help=bp_help)
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
    if args.keep:
        checks.append(ensure_file_exists(logger, str(args.keep), "--keep file"))
    if extract_file:
        checks.append(ensure_file_exists(logger, str(extract_file), "--extract file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    if args.file and out_fmt in {"vcf", "plink"} and (not _has_real_file_sites(gfile)):
        raise ValueError(
            f"{out_fmt.upper()} output from -file input requires real site metadata. "
            "Please provide a matching prefix.site or prefix.bim sidecar."
        )

    sample_ids, n_sites = inspect_genotype_file(gfile)
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

    has_site_filter = bool(push_snp_sites) or (push_bim_range is not None) or post_filter.active()
    selected_n_sites = int(n_sites)
    if has_site_filter:
        selected_n_sites = _count_selected_sites(
            gfile,
            sample_ids=keep_sample_ids,
            snp_sites=push_snp_sites,
            bim_range=push_bim_range,
            site_filter=post_filter,
        )
        if selected_n_sites <= 0:
            raise ValueError("All variants were filtered out by --extract/--chr/--from-bp/--to-bp.")

    out_sample_ids = keep_sample_ids if keep_sample_ids is not None else [str(s) for s in sample_ids]
    print(f"Genotype source: {gfile}")
    print(f"Samples: {len(out_sample_ids)}, sites: {selected_n_sites}")
    print(f"Output prefix: {out_prefix}")
    print(f"Output: {out_path} ({out_fmt})")

    def _make_chunks() -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
        c = load_genotype_chunks(
            gfile,
            chunk_size=50_000,
            maf=0.0,
            missing_rate=1.0,
            impute=False,
            sample_ids=keep_sample_ids,
            snp_sites=push_snp_sites,
            bim_range=push_bim_range,
        )
        if post_filter.active():
            c = _iter_filtered_chunks(c, post_filter)
        return c

    t0 = time.time()
    with CliStatus("Converting genotype format...", enabled=bool(getattr(os.sys.stdout, "isatty", lambda: False)())) as task:
        try:
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
                _write_text_output(out_path, out_sample_ids, _make_chunks(), total_sites=int(selected_n_sites))
            else:
                _write_npy_output(out_path, out_sample_ids, _make_chunks(), total_sites=int(selected_n_sites))
        except Exception:
            task.fail("Converting genotype format ...Failed")
            raise
        task.complete("Converting genotype format ...Finished")

    logger.info(f"Format conversion completed in {time.time() - t0:.2f} s")
    logger.info(f"Output written: {out_path}")


if __name__ == "__main__":
    main()
