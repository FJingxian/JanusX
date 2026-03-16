# -*- coding: utf-8 -*-
"""
JanusX: Pairwise hybrid genotype builder

Build all pairwise hybrids from two parent lists using one genotype source.

Supported genotype inputs
-------------------------
- VCF / VCF.GZ
- PLINK prefix (.bed/.bim/.fam)
- Text matrix (.txt/.tsv/.csv), numeric matrix with sibling prefix.id
- NumPy matrix (.npy), shape = (n_sites, n_samples), with sibling prefix.id

Notes
-----
- P1 and P2 must be text files with one sample ID per line.
- Duplicate IDs inside P1/P2 are removed while keeping first occurrence.
- `-file` text/npy inputs require sibling `prefix.id`.
- `prefix.site` or `prefix.bim` is optional for text/npy inputs, but required
  when exporting these inputs to VCF or PLINK.
- VCF / PLINK outputs round hybrid dosages to diploid 0/1/2.
- TXT / NPY outputs preserve floating hybrid dosages such as 0.5 / 1.5.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from janusx.gfreader import SiteInfo, inspect_genotype_file, load_genotype_chunks, save_genotype_streaming

from ._common.genoio import (
    GENOTYPE_SUFFIXES,
    GENOTYPE_TEXT_SUFFIXES,
    determine_genotype_source_from_args as determine_genotype_source,
    find_duplicates,
    output_prefix_from_path,
    write_npy_output,
    write_text_output,
)
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.genocache import configure_genotype_cache_from_out
from ._common.log import setup_logging
from ._common.pathcheck import format_output_display, format_path_for_display, safe_expanduser
from ._common.status import CliStatus, print_success, print_warning, stdout_is_tty


@dataclass
class SiteProvider:
    chrom: np.ndarray
    pos: np.ndarray
    ref: np.ndarray
    alt: np.ndarray

    @classmethod
    def dummy(cls, n_sites: int) -> "SiteProvider":
        return cls(
            chrom=np.repeat("N", n_sites).astype(object),
            pos=np.arange(1, n_sites + 1, dtype=np.int64),
            ref=np.repeat("N", n_sites).astype(object),
            alt=np.repeat("N", n_sites).astype(object),
        )

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

                def pick(candidates: Sequence[str], label: str) -> str:
                    for cand in candidates:
                        hit = col_map.get(str(cand).strip().lower())
                        if hit is not None:
                            return str(hit)
                    raise ValueError(
                        f"Unable to find {label} column in {p}. "
                        f"Available columns: {list(df.columns)}"
                    )

                chr_col = pick(
                    ["#CHROM", "CHROM", "chrom", "CHR", "chr", "chromosome", "Chromosome"],
                    "chromosome",
                )
                pos_col = pick(
                    ["POS", "pos", "BP", "bp", "Position", "position", "PS", "ps"],
                    "position",
                )
                ref_col = pick(
                    ["REF", "ref", "A0", "a0", "ref_allele", "RefAllele", "REF_ALLELE"],
                    "REF/A0",
                )
                alt_col = pick(
                    ["ALT", "alt", "A1", "a1", "alt_allele", "AltAllele", "ALT_ALLELE"],
                    "ALT/A1",
                )
                chrom = df[chr_col].astype(str).to_numpy(dtype=object)
                pos = pd.to_numeric(df[pos_col], errors="raise").astype(np.int64).to_numpy()
                ref = df[ref_col].astype(str).to_numpy(dtype=object)
                alt = df[alt_col].astype(str).to_numpy(dtype=object)

        if len(pos) != int(n_sites):
            raise ValueError(
                f"Site row count mismatch: {path} has {len(pos)} rows, expected {n_sites}"
            )
        return cls(chrom=chrom, pos=pos, ref=ref, alt=alt)

    def slice(self, start: int, end: int) -> list[SiteInfo]:
        return [
            SiteInfo(str(self.chrom[i]), int(self.pos[i]), str(self.ref[i]), str(self.alt[i]))
            for i in range(start, end)
        ]


def _site_provider_is_dummy(provider: SiteProvider) -> bool:
    n = len(provider.pos)
    if n == 0:
        return True
    chrom_dummy = bool(np.all(provider.chrom.astype(str) == "N"))
    ref_dummy = bool(np.all(provider.ref.astype(str) == "N"))
    alt_dummy = bool(np.all(provider.alt.astype(str) == "N"))
    pos_dummy = bool(np.array_equal(provider.pos.astype(np.int64), np.arange(1, n + 1, dtype=np.int64)))
    return chrom_dummy and ref_dummy and alt_dummy and pos_dummy


@dataclass
class HybridInputSource:
    kind: str
    path: str
    label: str
    sample_ids: list[str]
    n_sites: int
    has_real_sites: bool
    iter_selected_chunks: Callable[[list[str], int], Iterator[tuple[np.ndarray, list[SiteInfo]]]]


def _strip_known_matrix_suffix(path: str) -> str:
    p = str(path)
    low = p.lower()
    for ext in GENOTYPE_SUFFIXES:
        if low.endswith(ext):
            return p[: -len(ext)]
    return p


def _read_sidecar_ids(path: str) -> list[str]:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as fr:
        for raw in fr:
            line = raw.strip()
            if not line:
                continue
            lines.append(line)
    if not lines:
        raise ValueError(f"Empty sample ID sidecar: {path}")
    if len(lines) == 1:
        row = lines[0]
        if "," in row and ("\t" not in row):
            ids = [x.strip() for x in row.split(",") if x.strip()]
        elif "\t" in row:
            ids = [x.strip() for x in row.split("\t") if x.strip()]
        else:
            parts = row.split()
            ids = [x.strip() for x in parts if x.strip()]
    else:
        ids = []
        for idx, row in enumerate(lines, start=1):
            parts = row.split()
            if len(parts) != 1:
                raise ValueError(f"{path}:{idx}: expected 1 sample ID per line")
            ids.append(parts[0])
    dup = find_duplicates(ids)
    if dup:
        raise ValueError(f"Duplicate sample IDs in sidecar: {', '.join(dup[:10])}")
    return ids


def _resolve_file_input(file_arg: str) -> tuple[str, str]:
    """
    Resolve `-file` input.

    Returns
    -------
    kind : "text" | "npy"
    matrix_path : actual matrix path
    """
    raw = safe_expanduser(file_arg)
    base = Path(_strip_known_matrix_suffix(str(raw)))

    npy_candidates = []
    if raw.suffix.lower() == ".npy":
        npy_candidates.append(raw)
    npy_candidates.extend(
        [
            Path(f"{base}.npy"),
            base.parent / f"~{base.name}.npy",
        ]
    )
    seen_npy: set[str] = set()
    for cand in npy_candidates:
        key = str(cand)
        if key in seen_npy:
            continue
        seen_npy.add(key)
        if cand.exists():
            return "npy", str(cand)

    text_candidates = []
    if raw.suffix.lower() in GENOTYPE_TEXT_SUFFIXES:
        text_candidates.append(raw)
    text_candidates.extend([Path(f"{base}{ext}") for ext in GENOTYPE_TEXT_SUFFIXES])
    seen_text: set[str] = set()
    for cand in text_candidates:
        key = str(cand)
        if key in seen_text:
            continue
        seen_text.add(key)
        if cand.exists():
            return "text", str(cand)

    raise FileNotFoundError(
        f"Unable to resolve -file input: {file_arg}. "
        f"Checked text matrix and preferred npy cache for prefix {_strip_known_matrix_suffix(file_arg)}"
    )


def _site_prefix_candidates(file_arg: str, matrix_path: str) -> list[str]:
    raw = str(safe_expanduser(file_arg))
    prefixes: list[str] = []
    for cand in [
        _strip_known_matrix_suffix(raw),
        _strip_known_matrix_suffix(matrix_path),
    ]:
        if not cand:
            continue
        prefixes.append(cand)
        p = Path(cand)
        if p.name.startswith("~"):
            prefixes.append(str(p.with_name(p.name[1:])))
    out: list[str] = []
    seen: set[str] = set()
    for cand in prefixes:
        if cand in seen:
            continue
        seen.add(cand)
        out.append(cand)
    return out


def _discover_site_path(file_arg: str, matrix_path: str) -> str | None:
    candidates: list[str] = []
    for prefix in _site_prefix_candidates(file_arg, matrix_path):
        candidates.extend(
            [
                f"{prefix}.site",
                f"{prefix}.bim",
                f"{prefix}.sites.tsv",
                f"{prefix}.sites.txt",
                f"{prefix}.sites.csv",
                f"{prefix}.site.tsv",
                f"{prefix}.site.txt",
                f"{prefix}.site.csv",
            ]
        )
    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if Path(cand).exists():
            return cand
    return None


def _discover_id_path(file_arg: str, matrix_path: str) -> str:
    candidates: list[str] = []
    for prefix in _site_prefix_candidates(file_arg, matrix_path):
        candidates.append(f"{prefix}.id")
    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if Path(cand).exists():
            return cand
    raise FileNotFoundError(
        f"Missing sample sidecar for -file input: {file_arg}. "
        "Required: prefix.id and one numeric matrix file (.npy/.txt/.tsv/.csv)."
    )


def _load_parent_ids(path: str) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    dup_seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as fr:
        for lineno, raw in enumerate(fr, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 1:
                raise ValueError(f"{path}:{lineno}: expected 1 sample ID per line")
            sid = parts[0]
            if sid in seen:
                if sid not in dup_seen:
                    duplicates.append(sid)
                    dup_seen.add(sid)
                continue
            ids.append(sid)
            seen.add(sid)
    if not ids:
        raise ValueError(f"No valid sample IDs found in {path}")
    return ids, duplicates


def _ensure_unique_source_ids(sample_ids: Sequence[str]) -> None:
    dup = find_duplicates(list(sample_ids))
    if dup:
        raise ValueError(
            f"Duplicate sample IDs in genotype source: {', '.join(dup[:10])}"
        )


def _detect_text_delimiter(path: str) -> str | None:
    with open(path, "r", encoding="utf-8") as fr:
        for raw in fr:
            line = raw.strip()
            if not line:
                continue
            if "," in line and ("\t" not in line):
                return ","
            if "\t" in line:
                return "\t"
            return None
    raise ValueError(f"Empty genotype text matrix: {path}")


def _split_matrix_line(line: str, delimiter: str | None) -> list[str]:
    if delimiter is None:
        return line.split()
    return [x.strip() for x in line.split(delimiter)]


def _inspect_numeric_text_matrix(path: str, n_samples: int, delimiter: str | None) -> int:
    n_sites = 0
    with open(path, "r", encoding="utf-8") as fr:
        for lineno, raw in enumerate(fr, start=1):
            line = raw.strip()
            if not line:
                continue
            toks = [x for x in _split_matrix_line(line, delimiter) if x != ""]
            if len(toks) != int(n_samples):
                raise ValueError(
                    f"{path}:{lineno}: expected {n_samples} genotype columns, got {len(toks)}"
                )
            n_sites += 1
    if n_sites <= 0:
        raise ValueError(f"Empty genotype text matrix: {path}")
    return int(n_sites)


def _iter_numeric_text_matrix(
    path: str,
    sample_ids: Sequence[str],
    selected_ids: list[str],
    chunk_size: int,
    delimiter: str | None,
    provider: SiteProvider,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    sample_index = {sid: i for i, sid in enumerate(sample_ids)}
    selected_idx = [sample_index[sid] for sid in selected_ids]
    rows: list[list[float]] = []
    start_site = 0
    with open(path, "r", encoding="utf-8") as fr:
        for lineno, raw in enumerate(fr, start=1):
            line = raw.strip()
            if not line:
                continue
            toks = [x for x in _split_matrix_line(line, delimiter) if x != ""]
            if len(toks) != len(sample_ids):
                raise ValueError(
                    f"{path}:{lineno}: expected {len(sample_ids)} genotype columns, got {len(toks)}"
                )
            try:
                row = [float(toks[i]) for i in selected_idx]
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: invalid numeric genotype value") from exc
            rows.append(row)
            if len(rows) >= int(chunk_size):
                block = np.asarray(rows, dtype=np.float32)
                end_site = start_site + int(block.shape[0])
                yield block, provider.slice(start_site, end_site)
                start_site = end_site
                rows = []
    if rows:
        block = np.asarray(rows, dtype=np.float32)
        end_site = start_site + int(block.shape[0])
        yield block, provider.slice(start_site, end_site)


def _build_vcf_source(path: str) -> HybridInputSource:
    sample_ids, n_sites = inspect_genotype_file(path)
    _ensure_unique_source_ids(sample_ids)

    def iter_selected(selected_ids: list[str], chunk_size: int):
        yield from load_genotype_chunks(
            path,
            chunk_size=int(chunk_size),
            maf=0.0,
            missing_rate=1.0,
            impute=False,
            sample_ids=selected_ids,
        )

    return HybridInputSource(
        kind="vcf",
        path=path,
        label=f"VCF:{path}",
        sample_ids=list(sample_ids),
        n_sites=int(n_sites),
        has_real_sites=True,
        iter_selected_chunks=iter_selected,
    )


def _build_bfile_source(prefix: str) -> HybridInputSource:
    sample_ids, n_sites = inspect_genotype_file(prefix)
    _ensure_unique_source_ids(sample_ids)

    def iter_selected(selected_ids: list[str], chunk_size: int):
        yield from load_genotype_chunks(
            prefix,
            chunk_size=int(chunk_size),
            maf=0.0,
            missing_rate=1.0,
            impute=False,
            sample_ids=selected_ids,
        )

    return HybridInputSource(
        kind="bfile",
        path=prefix,
        label=f"PLINK:{prefix}",
        sample_ids=list(sample_ids),
        n_sites=int(n_sites),
        has_real_sites=True,
        iter_selected_chunks=iter_selected,
    )


def _build_file_source(file_arg: str) -> HybridInputSource:
    kind, matrix_path = _resolve_file_input(file_arg)
    provider_path = _discover_site_path(file_arg, matrix_path)
    id_path = _discover_id_path(file_arg, matrix_path)
    sample_ids = _read_sidecar_ids(id_path)
    _ensure_unique_source_ids(sample_ids)

    if kind == "npy":
        matrix = np.load(matrix_path, mmap_mode="r")
        if matrix.ndim != 2:
            raise ValueError(
                f"NumPy genotype matrix must be 2D (n_sites, n_samples); got shape {matrix.shape}"
            )
        if int(matrix.shape[1]) != len(sample_ids):
            raise ValueError(
                f"Sample ID count mismatch for {matrix_path}: "
                f"matrix has {int(matrix.shape[1])} columns, sample IDs = {len(sample_ids)}"
            )
        provider = SiteProvider.from_table(provider_path, int(matrix.shape[0])) if provider_path else SiteProvider.dummy(int(matrix.shape[0]))
        has_real_sites = not _site_provider_is_dummy(provider)
        sample_index = {sid: i for i, sid in enumerate(sample_ids)}

        def iter_selected(selected_ids: list[str], chunk_size: int):
            selected_idx = [sample_index[sid] for sid in selected_ids]
            total = int(matrix.shape[0])
            for start in range(0, total, int(chunk_size)):
                end = min(total, start + int(chunk_size))
                block = np.asarray(matrix[start:end, selected_idx], dtype=np.float32)
                yield block, provider.slice(start, end)

        return HybridInputSource(
            kind="npy",
            path=matrix_path,
            label=f"NPY:{matrix_path}",
            sample_ids=list(sample_ids),
            n_sites=int(matrix.shape[0]),
            has_real_sites=has_real_sites,
            iter_selected_chunks=iter_selected,
        )

    delimiter = _detect_text_delimiter(matrix_path)
    n_sites = _inspect_numeric_text_matrix(matrix_path, len(sample_ids), delimiter)
    override_provider = SiteProvider.from_table(provider_path, int(n_sites)) if provider_path else None
    has_real_sites = bool(override_provider is not None and (not _site_provider_is_dummy(override_provider)))
    provider = override_provider if override_provider is not None else SiteProvider.dummy(int(n_sites))

    def iter_selected(selected_ids: list[str], chunk_size: int):
        yield from _iter_numeric_text_matrix(
            matrix_path,
            sample_ids,
            selected_ids,
            int(chunk_size),
            delimiter,
            provider,
        )

    return HybridInputSource(
        kind="text",
        path=matrix_path,
        label=f"TEXT:{matrix_path}",
        sample_ids=list(sample_ids),
        n_sites=int(n_sites),
        has_real_sites=has_real_sites,
        iter_selected_chunks=iter_selected,
    )


def _sanitize_hybrid_parent_id(sample_id: str) -> str:
    return str(sample_id).replace("@", "at")


def _make_hybrid_sample_ids(p1_ids: Sequence[str], p2_ids: Sequence[str]) -> list[str]:
    out_ids: list[str] = []
    seen: set[str] = set()
    for p1 in p1_ids:
        left = _sanitize_hybrid_parent_id(p1)
        for p2 in p2_ids:
            right = _sanitize_hybrid_parent_id(p2)
            hybrid_id = f"{left}@{right}"
            if hybrid_id in seen:
                raise ValueError(
                    "Hybrid sample name collision after '@' normalization: "
                    f"{hybrid_id}. Please rename parent samples to unique IDs."
                )
            seen.add(hybrid_id)
            out_ids.append(hybrid_id)
    return out_ids


def _hybridize_chunk(p1_chunk: np.ndarray, p2_chunk: np.ndarray) -> np.ndarray:
    left = np.asarray(p1_chunk, dtype=np.float32)
    right = np.asarray(p2_chunk, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Hybrid chunk inputs must be 2D")
    if left.shape[0] != right.shape[0]:
        raise ValueError("P1 and P2 chunks must have the same number of rows")

    miss = (left < 0.0)[:, :, None] | (right < 0.0)[:, None, :]
    left_clip = np.clip(left, 0.0, 2.0)[:, :, None]
    right_clip = np.clip(right, 0.0, 2.0)[:, None, :]
    hybrid = (left_clip + right_clip) * np.float32(0.5)
    hybrid = hybrid.reshape(left.shape[0], left.shape[1] * right.shape[1]).astype(np.float32, copy=False)
    if np.any(miss):
        hybrid = hybrid.copy()
        hybrid[miss.reshape(left.shape[0], left.shape[1] * right.shape[1])] = np.float32(-9.0)
    return hybrid


def _make_hybrid_chunk_iterator(
    source: HybridInputSource,
    p1_ids: list[str],
    p2_ids: list[str],
    chunk_size: int,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    selected_ids = list(dict.fromkeys(list(p1_ids) + list(p2_ids)))
    selected_index = {sid: i for i, sid in enumerate(selected_ids)}
    p1_idx = [selected_index[sid] for sid in p1_ids]
    p2_idx = [selected_index[sid] for sid in p2_ids]

    for block, sites in source.iter_selected_chunks(selected_ids, int(chunk_size)):
        left = np.asarray(block[:, p1_idx], dtype=np.float32)
        right = np.asarray(block[:, p2_idx], dtype=np.float32)
        hybrid = _hybridize_chunk(left, right)
        yield hybrid, sites


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

def _validate_parent_ids(
    source: HybridInputSource,
    p1_ids: Sequence[str],
    p2_ids: Sequence[str],
    *,
    logger=None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    available = set(source.sample_ids)
    missing_p1 = [sid for sid in p1_ids if sid not in available]
    missing_p2 = [sid for sid in p2_ids if sid not in available]
    if missing_p1 or missing_p2:
        msg_parts: list[str] = []
        if missing_p1:
            msg_parts.append(f"P1 missing={', '.join(missing_p1[:10])}")
        if missing_p2:
            msg_parts.append(f"P2 missing={', '.join(missing_p2[:10])}")
        warn_msg = "Parent IDs not found in genotype source; they will be skipped. " + " | ".join(msg_parts)
        if logger is not None:
            logger.warning(warn_msg)
        else:
            print_warning(warn_msg)

    kept_p1 = [sid for sid in p1_ids if sid in available]
    kept_p2 = [sid for sid in p2_ids if sid in available]
    if not kept_p1:
        raise ValueError("No valid P1 parent IDs remain after filtering by genotype samples.")
    if not kept_p2:
        raise ValueError("No valid P2 parent IDs remain after filtering by genotype samples.")
    return kept_p1, kept_p2, missing_p1, missing_p2


def build_parser() -> CliArgumentParser:
    parser = CliArgumentParser(
        prog="jx hybrid",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx hybrid -vcf parents.vcf.gz -p1 p1.txt -p2 p2.txt -fmt txt",
                "jx hybrid -bfile geno/QC -p1 tester.txt -p2 female.txt -o outdir -prefix hybrids -fmt vcf",
                "jx hybrid -file geno_prefix -p1 p1.txt -p2 p2.txt -fmt npy",
            ]
        ),
    )

    in_group = parser.add_argument_group("Genotype Input")
    src = in_group.add_mutually_exclusive_group(required=True)
    src.add_argument("-vcf", "--vcf", type=str, help="Input VCF/VCF.GZ genotype file.")
    src.add_argument("-bfile", "--bfile", type=str, help="Input PLINK prefix (.bed/.bim/.fam).")
    src.add_argument(
        "-file",
        "--file",
        type=str,
        help=(
            "Input genotype matrix (.txt/.tsv/.csv/.npy) or prefix. "
            "Requires sibling prefix.id. Optional site metadata: prefix.site or prefix.bim. "
            "When prefix-matched .npy exists, it is preferred."
        ),
    )

    req = parser.add_argument_group("Required Arguments")
    req.add_argument("-p1", "--p1", required=True, type=str, help="Parent-1 sample list (one ID per line).")
    req.add_argument("-p2", "--p2", required=True, type=str, help="Parent-2 sample list (one ID per line).")

    opt = parser.add_argument_group("Optional Arguments")
    opt.add_argument(
        "-o",
        "--out",
        default=".",
        type=str,
        help="Output directory (default: current directory).",
    )
    opt.add_argument(
        "-prefix",
        "--prefix",
        type=str,
        default=None,
        help="Output prefix. Default: inferred from genotype input.",
    )
    opt.add_argument(
        "-fmt",
        "--fmt",
        dest="format",
        default="npy",
        choices=["plink", "vcf", "txt", "npy"],
        help="Output format: plink, vcf, txt, npy (default: npy).",
    )
    opt.add_argument(
        "-chunksize",
        "--chunksize",
        type=int,
        default=2000,
        help=(
            "Number of SNPs per chunk for streaming LMM/LM "
            "(affects GRM and GWAS; default: %(default)s)."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if int(args.chunksize) <= 0:
        raise ValueError("--chunksize must be > 0")

    out_fmt, out_prefix, out_path = _resolve_output_target(args)
    output_display = format_output_display(out_fmt, out_prefix, out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    configure_genotype_cache_from_out(str(Path(out_prefix).parent))
    log_path = f"{out_prefix}.hybrid.log"
    logger = setup_logging(log_path)
    status_enabled = stdout_is_tty()

    with CliStatus("Inspecting genotype input...", enabled=status_enabled) as task:
        try:
            if args.vcf:
                source = _build_vcf_source(args.vcf)
            elif args.bfile:
                source = _build_bfile_source(args.bfile)
            else:
                source = _build_file_source(args.file)
        except Exception:
            task.fail("Inspecting genotype input ...Failed")
            raise
        task.complete("Inspecting genotype input ...Finished")

    p1_ids, p1_dup = _load_parent_ids(args.p1)
    p2_ids, p2_dup = _load_parent_ids(args.p2)
    p1_ids, p2_ids, missing_p1, missing_p2 = _validate_parent_ids(
        source, p1_ids, p2_ids, logger=logger
    )
    hybrid_ids = _make_hybrid_sample_ids(p1_ids, p2_ids)
    require_real_sites = out_fmt in {"vcf", "plink"}
    if require_real_sites and (not source.has_real_sites):
        raise ValueError(
            f"{out_fmt.upper()} output from text/npy input requires real site metadata. "
            "Please provide a matching prefix.site or prefix.bim sidecar."
        )

    print(f"Genotype source: {format_path_for_display(source.path)}")
    print(f"Source samples: {len(source.sample_ids)}, sites: {source.n_sites}")
    if p1_dup:
        print(f"P1 duplicates removed: {', '.join(p1_dup[:10])}")
    if p2_dup:
        print(f"P2 duplicates removed: {', '.join(p2_dup[:10])}")
    if missing_p1:
        print(f"P1 missing skipped: {', '.join(missing_p1[:10])}")
    if missing_p2:
        print(f"P2 missing skipped: {', '.join(missing_p2[:10])}")
    overlap = sorted(set(p1_ids) & set(p2_ids))
    if overlap:
        shown = ", ".join(overlap[:10])
        print(f"P1/P2 overlap retained: {shown}")
    print(f"P1 parents: {len(p1_ids)}")
    print(f"P2 parents: {len(p2_ids)}")
    print(f"Hybrid progeny: {len(hybrid_ids)}")
    print(f"Output: {output_display}")

    chunks = _make_hybrid_chunk_iterator(
        source,
        p1_ids,
        p2_ids,
        int(args.chunksize),
    )

    with CliStatus("Building hybrid genotypes...", enabled=status_enabled) as task:
        try:
            if out_fmt == "vcf":
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                save_genotype_streaming(
                    out_path,
                    hybrid_ids,
                    chunks,
                    fmt="vcf",
                    total_snps=int(source.n_sites),
                    desc="Writing hybrid VCF",
                )
            elif out_fmt == "plink":
                Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
                save_genotype_streaming(
                    out_prefix,
                    hybrid_ids,
                    chunks,
                    fmt="plink",
                    total_snps=int(source.n_sites),
                    desc="Writing hybrid PLINK",
                )
            elif out_fmt == "txt":
                write_text_output(
                    out_path,
                    hybrid_ids,
                    chunks,
                    total_sites=int(source.n_sites),
                )
            else:
                write_npy_output(
                    out_path,
                    hybrid_ids,
                    chunks,
                    total_sites=int(source.n_sites),
                )
        except Exception:
            task.fail("Building hybrid genotypes ...Failed")
            raise
        task.complete("Building hybrid genotypes ...Finished")

    print_success(
        f"Hybrid build completed. Hybrids written: {len(hybrid_ids)}",
        force_color=True,
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
