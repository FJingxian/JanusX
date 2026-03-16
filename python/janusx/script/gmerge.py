# -*- coding: utf-8 -*-
"""
JanusX: Efficient Genotype Merger (gmerge)

Supported inputs
----------------
- `-vcf/--vcf`   : one or more VCF / VCF.GZ files
- `-bfile/--bfile`: one or more PLINK prefixes
- `-file/--file` : one or more text / npy genotype matrices

Notes
-----
- `-file` inputs must carry `prefix.id`.
- `-file` inputs accept site metadata via `prefix.site` or `prefix.bim`.
- Output can be written as PLINK, VCF.GZ, text matrix, or NPY matrix.
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from ._common.config_render import emit_cli_configuration
from ._common.genoio import (
    GENOTYPE_TEXT_SUFFIXES,
    build_prefix_candidates,
    discover_id_sidecar_path,
    discover_site_path,
    find_duplicates,
    strip_known_suffix,
    write_npy_output,
    write_text_output,
)
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.genocache import configure_genotype_cache_from_out
from ._common.log import setup_logging
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_plink_prefix_exists,
    safe_expanduser,
)
from ._common.status import CliStatus
from janusx.gfreader import SiteInfo, inspect_genotype_file, load_genotype_chunks, save_genotype_streaming
from janusx.gfreader.gmerge import merge


@dataclass
class SiteProvider:
    chrom: np.ndarray
    pos: np.ndarray
    ref: np.ndarray
    alt: np.ndarray

    @classmethod
    def from_table(cls, path: str, n_sites: int) -> "SiteProvider":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Site table not found: {p}")

        if p.suffix.lower() == ".bim":
            df = pd.read_csv(p, sep=r"\s+", header=None, dtype=str)
            if df.shape[1] < 6:
                raise ValueError(f"Invalid BIM file: {p}")
            chrom = df.iloc[:, 0].astype(str).to_numpy(dtype=object)
            pos = pd.to_numeric(df.iloc[:, 3], errors="raise").astype(np.int64).to_numpy()
            ref = df.iloc[:, 4].astype(str).to_numpy(dtype=object)
            alt = df.iloc[:, 5].astype(str).to_numpy(dtype=object)
        else:
            raw = pd.read_csv(p, sep=None, engine="python", header=None, dtype=str)
            if raw.shape[1] < 4:
                raise ValueError(f"Invalid site table: {p}")
            first_row = [str(x).strip().lower() for x in raw.iloc[0].tolist()]
            has_header = any(
                token in {"#chrom", "chrom", "chr", "chromosome", "pos", "bp", "position", "ps", "ref", "alt", "a0", "a1"}
                for token in first_row
            )

            if not has_header:
                chrom = raw.iloc[:, 0].astype(str).to_numpy(dtype=object)
                pos = pd.to_numeric(raw.iloc[:, 1], errors="raise").astype(np.int64).to_numpy()
                ref = raw.iloc[:, 2].astype(str).to_numpy(dtype=object)
                alt = raw.iloc[:, 3].astype(str).to_numpy(dtype=object)
            else:
                df = pd.read_csv(p, sep=None, engine="python")
                col_map = {str(c).strip().lower(): c for c in df.columns}

                def pick(cands: Sequence[str], label: str) -> str:
                    for cand in cands:
                        hit = col_map.get(str(cand).strip().lower())
                        if hit is not None:
                            return str(hit)
                    raise ValueError(
                        f"Unable to find {label} column in {p}. Available columns: {list(df.columns)}"
                    )

                chr_col = pick(
                    ["#CHROM", "CHROM", "chrom", "CHR", "chr", "chromosome"],
                    "chromosome",
                )
                pos_col = pick(
                    ["POS", "pos", "BP", "bp", "Position", "position", "PS", "ps"],
                    "position",
                )
                ref_col = pick(
                    ["REF", "ref", "A0", "a0", "ref_allele", "REF_ALLELE"],
                    "REF/A0",
                )
                alt_col = pick(
                    ["ALT", "alt", "A1", "a1", "alt_allele", "ALT_ALLELE"],
                    "ALT/A1",
                )

                chrom = df[chr_col].astype(str).to_numpy(dtype=object)
                pos = pd.to_numeric(df[pos_col], errors="raise").astype(np.int64).to_numpy()
                ref = df[ref_col].astype(str).to_numpy(dtype=object)
                alt = df[alt_col].astype(str).to_numpy(dtype=object)

        if len(pos) != int(n_sites):
            raise ValueError(
                f"Site row count mismatch: {p} has {len(pos)} rows, expected {n_sites}"
            )
        return cls(chrom=chrom, pos=pos, ref=ref, alt=alt)

    def is_dummy(self) -> bool:
        n = len(self.pos)
        if n == 0:
            return True
        chrom_dummy = bool(np.all(self.chrom.astype(str) == "N"))
        ref_dummy = bool(np.all(self.ref.astype(str) == "N"))
        alt_dummy = bool(np.all(self.alt.astype(str) == "N"))
        pos_dummy = bool(
            np.array_equal(self.pos.astype(np.int64), np.arange(1, n + 1, dtype=np.int64))
        )
        return chrom_dummy and ref_dummy and alt_dummy and pos_dummy

    def slice(self, start: int, end: int) -> list[SiteInfo]:
        return [
            SiteInfo(str(self.chrom[i]), int(self.pos[i]), str(self.ref[i]), str(self.alt[i]))
            for i in range(start, end)
        ]


@dataclass
class PreparedInput:
    original: str
    merge_path: str
    display_kind: str


@dataclass
class FileInputSource:
    matrix_kind: str
    matrix_path: str
    sample_ids: list[str]
    n_sites: int
    site_provider: SiteProvider
    iter_selected_chunks: Callable[[int], Iterator[tuple[np.ndarray, list[SiteInfo]]]]


def _basename_no_suffix(path: str) -> str:
    return os.path.basename(strip_known_suffix(path.rstrip("/")))


def _read_id_sidecar(path: str) -> list[str]:
    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as fr:
        for lineno, raw in enumerate(fr, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 1:
                raise ValueError(f"{path}:{lineno}: expected 1 sample ID per line")
            ids.append(parts[0])
    if not ids:
        raise ValueError(f"Empty sample ID sidecar: {path}")
    dup = find_duplicates(ids)
    if dup:
        raise ValueError(f"Duplicate sample IDs in sidecar: {', '.join(dup[:10])}")
    return ids


def _discover_file_matrix(file_arg: str) -> tuple[str, str, list[str]]:
    raw_prefix, prefixes = build_prefix_candidates(file_arg, text_suffixes=GENOTYPE_TEXT_SUFFIXES)

    npy_candidates = []
    raw = safe_expanduser(file_arg)
    if raw.suffix.lower() == ".npy":
        npy_candidates.append(str(raw))
    npy_candidates.extend([f"{p}.npy" for p in prefixes])

    seen: set[str] = set()
    for cand in npy_candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if Path(cand).exists():
            return "npy", cand, prefixes

    text_candidates = []
    if raw.suffix.lower() in GENOTYPE_TEXT_SUFFIXES:
        text_candidates.append(str(raw))
    for prefix in prefixes:
        for ext in GENOTYPE_TEXT_SUFFIXES:
            text_candidates.append(f"{prefix}{ext}")

    seen.clear()
    for cand in text_candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if Path(cand).exists():
            return "text", cand, prefixes

    raise FileNotFoundError(
        f"Unable to resolve -file input: {file_arg}. "
        f"Checked prefix candidates: {', '.join(prefixes)}"
    )


def _discover_id_path(matrix_path: str, prefixes: Sequence[str]) -> list[str]:
    id_path = discover_id_sidecar_path(matrix_path, prefixes)
    if id_path is not None:
        return _read_id_sidecar(id_path)
    raise FileNotFoundError(
        f"Unable to find sample IDs for matrix {matrix_path}. "
        "Provide a matching prefix.id sidecar."
    )


def _build_file_source(file_arg: str, delimiter: str | None = None) -> FileInputSource:
    matrix_kind, matrix_path, prefixes = _discover_file_matrix(file_arg)
    site_path = discover_site_path(prefixes)
    if site_path is None:
        raise ValueError(
            f"-file input requires real site metadata (.site or .bim): {file_arg}"
        )
    sample_ids = _discover_id_path(matrix_path, prefixes)

    if matrix_kind == "npy":
        matrix = np.load(matrix_path, mmap_mode="r")
        if matrix.ndim != 2:
            raise ValueError(
                f"NumPy genotype matrix must be 2D (n_sites, n_samples); got {matrix.shape}"
            )
        if int(matrix.shape[1]) != len(sample_ids):
            raise ValueError(
                f"Sample ID count mismatch for {matrix_path}: "
                f"matrix columns={int(matrix.shape[1])}, sample IDs={len(sample_ids)}"
            )
        provider = SiteProvider.from_table(site_path, int(matrix.shape[0]))
        if provider.is_dummy():
            raise ValueError(f"Detected dummy site metadata for -file input: {site_path}")
        sample_index = {sid: i for i, sid in enumerate(sample_ids)}

        def iter_chunks(chunk_size: int):
            total = int(matrix.shape[0])
            selected_idx = list(range(len(sample_ids)))
            for start in range(0, total, int(chunk_size)):
                end = min(total, start + int(chunk_size))
                block = np.asarray(matrix[start:end, :][:, selected_idx], dtype=np.float32)
                yield block, provider.slice(start, end)

        return FileInputSource(
            matrix_kind="npy",
            matrix_path=matrix_path,
            sample_ids=list(sample_ids),
            n_sites=int(matrix.shape[0]),
            site_provider=provider,
            iter_selected_chunks=iter_chunks,
        )

    sample_ids2, n_sites = inspect_genotype_file(matrix_path, delimiter=delimiter)
    if list(sample_ids2) != list(sample_ids):
        raise ValueError(
            f"Sample ID sidecar mismatch for {matrix_path}: "
            "prefix.id does not match detected sample order."
        )
    provider = SiteProvider.from_table(site_path, int(n_sites))
    if provider.is_dummy():
        raise ValueError(f"Detected dummy site metadata for -file input: {site_path}")

    def iter_chunks(chunk_size: int):
        offset = 0
        for block, _ in load_genotype_chunks(
            matrix_path,
            chunk_size=int(chunk_size),
            maf=0.0,
            missing_rate=1.0,
            impute=False,
            delimiter=delimiter,
        ):
            n_rows = int(block.shape[0])
            yield np.asarray(block, dtype=np.float32), provider.slice(offset, offset + n_rows)
            offset += n_rows

    return FileInputSource(
        matrix_kind="text",
        matrix_path=matrix_path,
        sample_ids=list(sample_ids),
        n_sites=int(n_sites),
        site_provider=provider,
        iter_selected_chunks=iter_chunks,
    )


def _check_file_input_resolvable(logger, src: str) -> bool:
    try:
        _build_file_source(src)
        return True
    except Exception as exc:
        logger.error(f"Input FILE not found or unresolved: {src}")
        logger.error(str(exc))
        return False


def _infer_default_prefix(vcf_inputs: Sequence[str], bfile_inputs: Sequence[str], file_inputs: Sequence[str]) -> str:
    all_inputs = list(vcf_inputs) + list(bfile_inputs) + list(file_inputs)
    if not all_inputs:
        return "merged"
    return f"{_basename_no_suffix(all_inputs[0])}.merge"


def _resolve_output(format_name: str, outdir: str, prefix: str) -> tuple[str, str]:
    fmt = str(format_name).lower()
    base = os.path.join(outdir, prefix)
    if fmt == "plink":
        return fmt, base
    if fmt == "vcf":
        return fmt, f"{base}.vcf.gz"
    if fmt == "txt":
        return fmt, f"{base}.txt"
    if fmt == "npy":
        return fmt, f"{base}.npy"
    raise ValueError(f"Unsupported format: {format_name}")


def _read_fam_sample_names(prefix: str) -> list[str]:
    fam = f"{prefix}.fam"
    out: list[str] = []
    with open(fam, "r", encoding="utf-8") as fr:
        for lineno, raw in enumerate(fr, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{fam}:{lineno}: malformed FAM line")
            fid, iid = parts[0], parts[1]
            out.append(f"{fid}_{iid}" if fid not in {"", "0"} else iid)
    return out


def _prepare_file_inputs(
    file_inputs: Sequence[str],
    tmpdir: str,
    logger,
) -> list[PreparedInput]:
    prepared: list[PreparedInput] = []
    for idx, src in enumerate(file_inputs, start=1):
        logger.info(f"Preparing -file input {idx}/{len(file_inputs)}: {src}")
        file_source = _build_file_source(src)
        temp_prefix = os.path.join(tmpdir, f"file_input_{idx}")
        desc = f"Converting -file input {idx}/{len(file_inputs)}"
        save_genotype_streaming(
            temp_prefix,
            file_source.sample_ids,
            file_source.iter_selected_chunks(50_000),
            fmt="plink",
            total_snps=int(file_source.n_sites),
            desc=desc,
        )
        logger.info(f"{desc} finished.")
        prepared.append(
            PreparedInput(
                original=src,
                merge_path=temp_prefix,
                display_kind=f"file:{file_source.matrix_kind}",
            )
        )
    return prepared


def build_parser() -> CliArgumentParser:
    parser = CliArgumentParser(
        prog="jx gmerge",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx gmerge -vcf a.vcf.gz b.vcf.gz -fmt vcf",
                "jx gmerge -bfile A B -o merged_dir -prefix panel -fmt plink",
                "jx gmerge -vcf a.vcf.gz -file matrix_prefix -fmt txt",
            ]
        ),
    )

    req = parser.add_argument_group("Input Arguments")
    req.add_argument(
        "-vcf",
        "--vcf",
        nargs="+",
        action="extend",
        default=None,
        help="Input VCF / VCF.GZ files (repeatable).",
    )
    req.add_argument(
        "-bfile",
        "--bfile",
        nargs="+",
        action="extend",
        default=None,
        help="Input PLINK prefixes (repeatable).",
    )
    req.add_argument(
        "-file",
        "--file",
        nargs="+",
        action="extend",
        default=None,
        help=(
            "Input genotype text/NumPy matrices or prefixes (repeatable). "
            "If prefix-matched .npy exists, it is preferred. "
            "Requires prefix.id. Site metadata: prefix.site or prefix.bim."
        ),
    )

    opt = parser.add_argument_group("Optional Arguments")
    opt.add_argument(
        "-fmt",
        "--fmt",
        dest="format_name",
        choices=["plink", "vcf", "txt", "npy"],
        default="vcf",
        help="Output genotype format: plink, vcf, txt, npy (default: vcf.gz).",
    )
    opt.add_argument(
        "-o",
        "--out",
        default=".",
        help="Output directory for merged results (default: %(default)s).",
    )
    opt.add_argument(
        "-prefix",
        "--prefix",
        default=None,
        help="Prefix for merged output files (default: inferred from first input).",
    )
    opt.add_argument(
        "-sample-prefix",
        "--sample-prefix",
        action="store_true",
        default=False,
        help="Prefix sample IDs by dataset index (D1_, D2_, ...). Default: off.",
    )
    opt.add_argument(
        "-maf",
        "--maf",
        type=float,
        default=0.0,
        help="Drop merged sites with MAF below this threshold (range: 0-0.5; default: %(default)s).",
    )
    opt.add_argument(
        "-geno",
        "--geno",
        type=float,
        default=1.0,
        help="Drop merged sites with missing rate above this threshold (range: 0-1; default: %(default)s).",
    )
    return parser


def main(log: bool = True) -> None:
    t_start = time.time()
    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())
    args = build_parser().parse_args()

    vcf_inputs = [x.replace("\\", "/") for x in (args.vcf or [])]
    bfile_inputs = [x.replace("\\", "/") for x in (args.bfile or [])]
    file_inputs = [x.replace("\\", "/") for x in (args.file or [])]

    total_inputs = len(vcf_inputs) + len(bfile_inputs) + len(file_inputs)
    if total_inputs < 2:
        raise ValueError(
            "At least 2 inputs are required across -vcf/-bfile/-file."
        )

    outdir = args.out.replace("\\", "/")
    prefix = (args.prefix or _infer_default_prefix(vcf_inputs, bfile_inputs, file_inputs)).replace("\\", "/")
    os.makedirs(outdir, mode=0o755, exist_ok=True)
    configure_genotype_cache_from_out(outdir)

    log_path = os.path.join(outdir, f"{prefix}.merge.log").replace("\\", "/")
    logger = setup_logging(log_path)

    final_format, final_output = _resolve_output(args.format_name, outdir, prefix)

    if log:
        input_rows: list[tuple[str, str]] = []
        for idx, src in enumerate(vcf_inputs, start=1):
            input_rows.append((f"VCF {idx}", src))
        for idx, src in enumerate(bfile_inputs, start=1):
            input_rows.append((f"BFILE {idx}", src))
        for idx, src in enumerate(file_inputs, start=1):
            input_rows.append((f"FILE {idx}", src))
        emit_cli_configuration(
            logger,
            app_title="JanusX - gmerge",
            config_title="GMERGE CONFIG",
            host=socket.gethostname(),
            sections=[
                ("Inputs", input_rows),
                (
                    "General",
                    [
                        ("Format", final_format),
                        ("Output dir", outdir),
                        ("Prefix", prefix),
                        ("Sample prefix", bool(args.sample_prefix)),
                        ("MAF", args.maf),
                        ("GENO", args.geno),
                        ("Log file", log_path),
                    ],
                ),
            ],
            footer_rows=[("Output", final_output)],
            line_max_chars=60,
        )

    checks: list[bool] = []
    for src in vcf_inputs:
        checks.append(ensure_file_exists(logger, src, "Input VCF file"))
    for src in bfile_inputs:
        checks.append(ensure_plink_prefix_exists(logger, src, "Input PLINK prefix"))
    for src in file_inputs:
        checks.append(_check_file_input_resolvable(logger, src))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    with tempfile.TemporaryDirectory(prefix=f".janusx_gmerge_{prefix}_", dir=outdir) as tmpdir:
        prepared_inputs: list[PreparedInput] = []
        prepared_inputs.extend(
            PreparedInput(original=x, merge_path=x, display_kind="vcf") for x in vcf_inputs
        )
        prepared_inputs.extend(
            PreparedInput(original=x, merge_path=x, display_kind="bfile") for x in bfile_inputs
        )
        prepared_inputs.extend(
            _prepare_file_inputs(
                file_inputs,
                tmpdir,
                logger,
            )
        )

        merge_inputs = [item.merge_path for item in prepared_inputs]

        merge_out = final_output
        merge_fmt = final_format
        postconvert_source: str | None = None
        if final_format in {"txt", "npy"}:
            merge_out = os.path.join(tmpdir, "merged_tmp")
            merge_fmt = "plink"
            postconvert_source = merge_out

        with CliStatus("Merging genotype datasets...", enabled=use_spinner) as task:
            try:
                stats, d = merge(
                    inputs=merge_inputs,
                    out=merge_out,
                    out_fmt=merge_fmt,
                    sample_prefix=bool(args.sample_prefix),
                    maf=float(args.maf),
                    geno=float(args.geno),
                    check_exists=True,
                    return_dict=True,
                )
            except Exception:
                task.fail("Merging genotype datasets ...Failed")
                raise
            task.complete("Merging genotype datasets ...Finished")

        if final_format == "txt":
            sample_ids = _read_fam_sample_names(postconvert_source)
            with CliStatus("Writing merged text matrix...", enabled=use_spinner) as task:
                try:
                    _, n_sites = inspect_genotype_file(postconvert_source)
                    chunks = load_genotype_chunks(
                        postconvert_source,
                        chunk_size=50_000,
                        maf=0.0,
                        missing_rate=1.0,
                        impute=False,
                    )
                    write_text_output(
                        final_output,
                        sample_ids,
                        chunks,
                        total_sites=int(n_sites),
                    )
                except Exception:
                    task.fail("Writing merged text matrix ...Failed")
                    raise
                task.complete("Writing merged text matrix ...Finished")
        elif final_format == "npy":
            sample_ids = _read_fam_sample_names(postconvert_source)
            with CliStatus("Writing merged npy matrix...", enabled=use_spinner) as task:
                try:
                    _, n_sites = inspect_genotype_file(postconvert_source)
                    chunks = load_genotype_chunks(
                        postconvert_source,
                        chunk_size=50_000,
                        maf=0.0,
                        missing_rate=1.0,
                        impute=False,
                    )
                    write_npy_output(
                        final_output,
                        sample_ids,
                        chunks,
                        total_sites=int(n_sites),
                    )
                except Exception:
                    task.fail("Writing merged npy matrix ...Failed")
                    raise
                task.complete("Writing merged npy matrix ...Finished")

        logger.info("*" * 60)
        logger.info("GMERGE SUMMARY")
        logger.info("*" * 60)
        logger.info(f"Final output: {final_output}")
        logger.info(f"Final format: {final_format}")
        logger.info(f"Total inputs: {len(prepared_inputs)}")
        logger.info(f"Sample counts per input: {d.get('sample_counts')}")
        logger.info(f"Total samples: {d.get('n_samples_total')}")
        logger.info(f"Sites written: {d.get('n_sites_written')}")
        logger.info(f"Union sites seen: {d.get('n_sites_union_seen')}")
        logger.info(f"Dropped multiallelic: {d.get('n_sites_dropped_multiallelic')}")
        logger.info(f"Dropped non-SNP: {d.get('n_sites_dropped_non_snp')}")
        logger.info(f"Dropped by MAF: {d.get('n_sites_dropped_maf')}")
        logger.info(f"Dropped by GENO: {d.get('n_sites_dropped_geno')}")
        logger.info(f"Per-input present sites: {d.get('per_input_present_sites')}")
        logger.info(f"Per-input unaligned sites: {d.get('per_input_unaligned_sites')}")
        logger.info(f"Per-input absent sites: {d.get('per_input_absent_sites')}")
        logger.info("*" * 60)

        if final_format == "plink":
            logger.info(
                "Merged PLINK files saved:\n  %s.bed\n  %s.bim\n  %s.fam",
                final_output,
                final_output,
                final_output,
            )
        elif final_format == "vcf":
            logger.info(f"Merged VCF saved:\n  {final_output}")
        elif final_format == "txt":
            logger.info(
                "Merged text matrix saved:\n  %s\n  %s.id\n  %s.site",
                final_output,
                strip_known_suffix(final_output),
                strip_known_suffix(final_output),
            )
        elif final_format == "npy":
            logger.info(
                "Merged npy matrix saved:\n  %s\n  %s.id\n  %s.site",
                final_output,
                strip_known_suffix(final_output),
                strip_known_suffix(final_output),
            )

    lt = time.localtime()
    logger.info(
        "\nFinished genotype merge. Total wall time: %.2f seconds\n%d-%d-%d %d:%d:%d",
        time.time() - t_start,
        lt.tm_year,
        lt.tm_mon,
        lt.tm_mday,
        lt.tm_hour,
        lt.tm_min,
        lt.tm_sec,
    )


if __name__ == "__main__":
    from janusx.script._common.interrupt import force_exit, install_interrupt_handlers
    install_interrupt_handlers()
    try:
        main()
    except BrokenPipeError:
        raise SystemExit(1)
    except KeyboardInterrupt:
        force_exit(130, "Interrupted by user (Ctrl+C).")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
