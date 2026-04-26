#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from janusx.gfreader import inspect_genotype_file, load_genotype_chunks


def _parse_col_spec(spec: str | None) -> str | int | None:
    if spec is None:
        return None
    s = str(spec).strip()
    if s == "":
        return None
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    return s


def _read_table(path: str, sep: str | None, header_mode: str) -> pd.DataFrame:
    p = str(path)
    if header_mode == "none":
        return pd.read_csv(p, sep=sep or r"\s+", header=None, comment="#", engine="python")
    if header_mode == "infer":
        return pd.read_csv(p, sep=sep or None, header="infer", comment="#", engine="python")

    # auto: try infer first; if it degenerates to empty/one-line, fallback to no-header
    try:
        df = pd.read_csv(p, sep=sep or None, header="infer", comment="#", engine="python")
        if df.shape[0] >= 1 and df.shape[1] >= 1:
            return df
    except Exception:
        pass
    return pd.read_csv(p, sep=sep or r"\s+", header=None, comment="#", engine="python")


def _resolve_column(df: pd.DataFrame, spec: str | int | None, default_idx: int, label: str) -> Any:
    if spec is None:
        idx = int(default_idx)
        if idx < 0:
            idx += df.shape[1]
        if idx < 0 or idx >= df.shape[1]:
            raise ValueError(f"Default {label} index out of range: {default_idx}")
        return df.columns[idx]

    if isinstance(spec, int):
        idx = int(spec)
        if idx < 0:
            idx += df.shape[1]
        if idx < 0 or idx >= df.shape[1]:
            raise ValueError(f"{label} column index out of range: {spec}")
        return df.columns[idx]

    if spec not in df.columns:
        raise ValueError(f"{label} column not found: {spec}")
    return spec


def _to_binary01(y: np.ndarray) -> tuple[np.ndarray, str | None]:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NA/NaN/Inf after phenotype parsing")

    uniq = np.unique(y)
    if uniq.size == 0:
        raise ValueError("y is empty")

    if uniq.size <= 2 and set(np.round(uniq, 12).tolist()).issubset({0.0, 1.0}):
        return (y > 0.5).astype(np.uint8, copy=False), None

    if uniq.size == 2:
        lo, hi = float(uniq[0]), float(uniq[1])
        yb = (y == hi).astype(np.uint8, copy=False)
        msg = f"Mapped binary labels to 0/1 automatically: {lo} -> 0, {hi} -> 1"
        return yb, msg

    raise ValueError(
        f"Binary beam search expects 2-class y. Found {uniq.size} unique values: {uniq[:8]}"
    )


def _align_y_to_samples(
    sample_ids: list[str],
    pheno_df: pd.DataFrame,
    id_col_spec: str | int | None,
    y_col_spec: str | int | None,
) -> tuple[np.ndarray, str | None]:
    n_samples = len(sample_ids)
    if n_samples == 0:
        raise ValueError("No samples found in BIN input")

    if (id_col_spec is None) ^ (y_col_spec is None):
        raise ValueError("Please provide both --id-col and --y-col, or provide neither.")

    # Case 1: no explicit id/y columns
    if id_col_spec is None and y_col_spec is None:
        if pheno_df.shape[1] == 1:
            y_raw = pd.to_numeric(pheno_df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
            if y_raw.shape[0] < n_samples:
                raise ValueError(
                    f"Phenotype row count ({y_raw.shape[0]}) is smaller than BIN sample count ({n_samples})"
                )
            y_raw = y_raw[:n_samples]
            return _to_binary01(y_raw)

        # Auto-detect common formats
        if pheno_df.shape[1] >= 3:
            id_col = _resolve_column(pheno_df, 1, 1, "id")
            y_col = _resolve_column(pheno_df, 2, 2, "y")
        else:
            id_col = _resolve_column(pheno_df, 0, 0, "id")
            y_col = _resolve_column(pheno_df, 1, 1, "y")
    else:
        id_col = _resolve_column(pheno_df, id_col_spec, 0, "id")
        y_col = _resolve_column(pheno_df, y_col_spec, 1, "y")

    ids = pheno_df[id_col].astype(str).str.strip()
    yv = pd.to_numeric(pheno_df[y_col], errors="coerce")

    y_map: dict[str, float] = {}
    for sid, val in zip(ids.tolist(), yv.tolist()):
        if sid == "" or pd.isna(val):
            continue
        if sid not in y_map:
            y_map[sid] = float(val)

    missing = [sid for sid in sample_ids if sid not in y_map]
    if missing:
        preview = ", ".join(missing[:8])
        more = "" if len(missing) <= 8 else f" ... (+{len(missing)-8})"
        raise ValueError(
            "Phenotype is missing samples from BIN .id/.fam order: "
            f"{preview}{more}. Please provide complete phenotype IDs."
        )

    y_raw = np.array([y_map[sid] for sid in sample_ids], dtype=float)
    return _to_binary01(y_raw)


def _extract_sites(bin_path: str, selected: list[int]) -> list[Any]:
    if not selected:
        return []
    try:
        out = []
        it = load_genotype_chunks(
            bin_path,
            chunk_size=max(1, len(selected)),
            maf=0.0,
            missing_rate=1.0,
            impute=False,
            snp_indices=[int(i) for i in selected],
        )
        for _geno, sites in it:
            out.extend(list(sites))
        return out
    except Exception:
        return []


BSITE_MAGIC = b"JXBSIT02"
BSITE_VERSION = 1
BSITE_HEADER_SIZE = 36


@dataclass
class _SiteRow:
    index: int
    chrom: str
    strand: int
    cm: float
    bp: int
    allele0: str
    allele1: str


@dataclass
class _Window:
    chrom: str
    bp_start: int
    bp_end: int
    indices: list[int]


def _split_chrom_strand(chrom: str) -> tuple[str, int]:
    s = str(chrom).strip()
    if s.endswith("_1"):
        return s[:-2], 0
    if s.endswith("_2"):
        return s[:-2], 1
    if s.endswith("-"):
        return s[:-1], 0
    if s.endswith("+"):
        return s[:-1], 1
    return s, 1


def _decode_allele_nibbles(packed: bytes, n_chars: int) -> str:
    lut = "ATCGN"
    out = []
    for i in range(int(max(1, n_chars))):
        b = packed[i >> 1]
        code = (b & 0x0F) if ((i & 1) == 0) else ((b >> 4) & 0x0F)
        if 0 <= code < len(lut):
            out.append(lut[code])
        else:
            out.append("N")
    return "".join(out)


def _read_bsite_rows(path: str) -> list[_SiteRow]:
    data = Path(path).read_bytes()
    if len(data) < BSITE_HEADER_SIZE:
        raise ValueError(f"Invalid .bsite file (too small): {path}")
    if data[:8] != BSITE_MAGIC:
        raise ValueError(f"Invalid .bsite magic: {path}")
    ver = struct.unpack_from("<H", data, 8)[0]
    if int(ver) != int(BSITE_VERSION):
        raise ValueError(f"Unsupported .bsite version {ver} in {path}")
    n_sites = int(struct.unpack_from("<Q", data, 12)[0])
    n_chrom = int(struct.unpack_from("<I", data, 20)[0])
    dict_offset = int(struct.unpack_from("<Q", data, 24)[0])
    if dict_offset < BSITE_HEADER_SIZE or dict_offset > len(data):
        raise ValueError(f"Invalid .bsite dictionary offset in {path}")

    rows_raw: list[tuple[int, int, float, int, str, str]] = []
    off = BSITE_HEADER_SIZE
    for i in range(n_sites):
        if off + 13 > dict_offset:
            raise ValueError(f"Truncated .bsite row header at record {i+1} in {path}")
        chrom_code, strand, cm, bp = struct.unpack_from("<IBfi", data, off)
        off += 13
        if off + 2 > dict_offset:
            raise ValueError(f"Truncated .bsite allele0 length at record {i+1} in {path}")
        a0_len = int(struct.unpack_from("<H", data, off)[0])
        off += 2
        a0_bytes = 0 if a0_len <= 0 else (a0_len + 1) // 2
        if off + a0_bytes > dict_offset:
            raise ValueError(f"Truncated .bsite allele0 payload at record {i+1} in {path}")
        a0 = "N" if a0_len <= 0 else _decode_allele_nibbles(data[off : off + a0_bytes], a0_len)
        off += a0_bytes
        if off + 2 > dict_offset:
            raise ValueError(f"Truncated .bsite allele1 length at record {i+1} in {path}")
        a1_len = int(struct.unpack_from("<H", data, off)[0])
        off += 2
        a1_bytes = 0 if a1_len <= 0 else (a1_len + 1) // 2
        if off + a1_bytes > dict_offset:
            raise ValueError(f"Truncated .bsite allele1 payload at record {i+1} in {path}")
        a1 = "N" if a1_len <= 0 else _decode_allele_nibbles(data[off : off + a1_bytes], a1_len)
        off += a1_bytes
        rows_raw.append((int(chrom_code), int(strand), float(cm), int(bp), a0, a1))

    chrom_names: list[str] = []
    doff = dict_offset
    for i in range(n_chrom):
        if doff + 2 > len(data):
            raise ValueError(f"Truncated .bsite chromosome length at entry {i} in {path}")
        n = int(struct.unpack_from("<H", data, doff)[0])
        doff += 2
        if doff + n > len(data):
            raise ValueError(f"Truncated .bsite chromosome payload at entry {i} in {path}")
        chrom_names.append(data[doff : doff + n].decode("utf-8", errors="replace"))
        doff += n

    rows: list[_SiteRow] = []
    for i, (chrom_code, strand, cm, bp, a0, a1) in enumerate(rows_raw):
        if chrom_code < 0 or chrom_code >= len(chrom_names):
            raise ValueError(f"Invalid .bsite chromosome code at row {i+1} in {path}")
        rows.append(
            _SiteRow(
                index=i,
                chrom=chrom_names[chrom_code],
                strand=0 if int(strand) == 0 else 1,
                cm=float(cm),
                bp=int(bp),
                allele0=str(a0),
                allele1=str(a1),
            )
        )
    return rows


def _read_site_rows_text(path: str) -> list[_SiteRow]:
    rows: list[_SiteRow] = []
    with open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.replace(",", " ").split()
            if len(toks) < 2:
                continue
            if len(rows) == 0 and (
                toks[1].lower() in {"pos", "bp", "position", "ps"}
                or toks[0].lower() in {"#chrom", "chrom", "chr", "chromosome"}
            ):
                continue
            if len(toks) >= 6 and toks[1] in {"-", "+", "0", "1"}:
                chrom = str(toks[0]).strip()
                strand = 0 if str(toks[1]) in {"-", "0"} else 1
                try:
                    cm = float(toks[2])
                except Exception:
                    cm = 0.0
                try:
                    bp = int(float(toks[3]))
                except Exception:
                    continue
                a0 = str(toks[4]) if len(toks) >= 5 else "N"
                a1 = str(toks[5]) if len(toks) >= 6 else "N"
            else:
                chrom_raw = str(toks[0])
                chrom, strand = _split_chrom_strand(chrom_raw)
                cm = 0.0
                try:
                    bp = int(float(toks[1]))
                except Exception:
                    continue
                a0 = str(toks[2]) if len(toks) >= 3 else "N"
                a1 = str(toks[3]) if len(toks) >= 4 else "N"
            rows.append(
                _SiteRow(
                    index=len(rows),
                    chrom=chrom,
                    strand=strand,
                    cm=float(cm),
                    bp=bp,
                    allele0=a0,
                    allele1=a1,
                )
            )
    return rows


def _discover_site_sidecar(bin_path: str) -> str | None:
    p = Path(bin_path)
    if p.suffix.lower() == ".bin":
        prefix = p.with_suffix("")
    else:
        prefix = p
    candidates = (
        prefix.with_suffix(".bsite"),
        prefix.with_suffix(".site"),
        prefix.with_suffix(".site.tsv"),
        prefix.with_suffix(".site.txt"),
        prefix.with_suffix(".site.csv"),
        prefix.with_suffix(".sites.tsv"),
        prefix.with_suffix(".sites.txt"),
        prefix.with_suffix(".sites.csv"),
        prefix.with_suffix(".bin.site"),
    )
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None


def _load_site_rows(bin_path: str, n_sites: int) -> tuple[list[_SiteRow], str | None]:
    site_path = _discover_site_sidecar(bin_path)
    if not site_path:
        return [], None
    low = site_path.lower()
    if low.endswith(".bin.site"):
        # Legacy k-mer sidecar has no CHR/BP metadata for genomic windows.
        return [], site_path

    # Preferred path: Rust-side decoding (covers .bsite/.site/.bim consistently).
    try:
        from janusx.janusx import load_site_info

        sites = list(load_site_info(bin_path, None))
        rows: list[_SiteRow] = []
        for i, s in enumerate(sites):
            if i >= int(n_sites):
                break
            chrom_raw = str(getattr(s, "chrom", ""))
            chrom, strand = _split_chrom_strand(chrom_raw)
            try:
                bp = int(getattr(s, "pos", 0))
            except Exception:
                bp = 0
            a0 = str(getattr(s, "ref_allele", "N"))
            a1 = str(getattr(s, "alt_allele", "N"))
            rows.append(
                _SiteRow(
                    index=i,
                    chrom=chrom,
                    strand=strand,
                    cm=0.0,
                    bp=bp,
                    allele0=a0,
                    allele1=a1,
                )
            )
        return rows, site_path
    except Exception:
        pass

    # Fallback path: Python parser (for older extensions without load_site_info).
    low = site_path.lower()
    if low.endswith(".bsite"):
        rows = _read_bsite_rows(site_path)
    else:
        rows = _read_site_rows_text(site_path)
    if len(rows) > n_sites:
        rows = rows[:n_sites]
    for i, row in enumerate(rows):
        row.index = i
    return rows, site_path


def _build_windows(
    rows: list[_SiteRow],
    n_sites: int,
    extension: int,
    step: int,
) -> list[_Window]:
    if n_sites <= 0:
        return []
    if extension <= 0:
        raise ValueError("--extension must be > 0")
    if step <= 0:
        raise ValueError("--step must be > 0")
    if not rows:
        return [_Window(chrom="ALL", bp_start=0, bp_end=0, indices=list(range(int(n_sites))))]

    groups: dict[str, list[tuple[int, int]]] = {}
    chrom_order: list[str] = []
    for r in rows:
        if r.index < 0 or r.index >= n_sites:
            continue
        chrom = str(r.chrom)
        if chrom not in groups:
            groups[chrom] = []
            chrom_order.append(chrom)
        groups[chrom].append((int(r.bp), int(r.index)))

    windows: list[_Window] = []
    for chrom in chrom_order:
        pairs = sorted(groups[chrom], key=lambda t: (t[0], t[1]))
        if not pairs:
            continue
        bps = [p[0] for p in pairs]
        idxs = [p[1] for p in pairs]
        n = len(pairs)
        min_bp = int(bps[0])
        max_bp = int(bps[-1])
        l = 0
        r = 0
        start = min_bp
        prev_sig: tuple[int, int, int] | None = None
        while start <= max_bp:
            while l < n and bps[l] < start:
                l += 1
            if r < l:
                r = l
            end = start + extension
            while r < n and bps[r] < end:
                r += 1
            if r > l:
                chunk = idxs[l:r]
                sig = (chunk[0], chunk[-1], len(chunk))
                if sig != prev_sig:
                    windows.append(
                        _Window(chrom=chrom, bp_start=int(start), bp_end=int(end), indices=chunk)
                    )
                    prev_sig = sig
            start += step
        if not windows or windows[-1].chrom != chrom:
            windows.append(
                _Window(
                    chrom=chrom,
                    bp_start=min_bp,
                    bp_end=max_bp + 1,
                    indices=[int(i) for i in idxs],
                )
            )
    if not windows:
        windows.append(
            _Window(chrom="ALL", bp_start=0, bp_end=0, indices=list(range(int(n_sites))))
        )
    return windows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jx beam",
        description=(
            "Beam search over AND-combinations of JXBIN001 rows (from `jx gformat ... -fmt bin`) "
            "to maximize MCC with binary phenotype y."
        ),
    )
    p.add_argument("-bin", "--bin", required=True, help="Input JXBIN001 file path (.bin)")
    p.add_argument("-pheno", "--pheno", required=True, help="Phenotype table path")
    p.add_argument("--id-col", default=None, help="Phenotype sample-id column (name or 0-based index)")
    p.add_argument("--y-col", default=None, help="Phenotype y column (name or 0-based index)")
    p.add_argument("--sep", default=None, help="Phenotype delimiter (default: auto)")
    p.add_argument(
        "--header",
        choices=("auto", "infer", "none"),
        default="auto",
        help="Phenotype header mode (default: auto)",
    )
    p.add_argument("-m", "--max-pick", type=int, default=3, help="Maximum vectors to AND (default: 3)")
    p.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "-nsnp",
        "--nsnp",
        type=int,
        default=None,
        help="Top SNPs selected by model / beam width (default: 5)",
    )
    p.add_argument(
        "-ext",
        "--extension",
        type=int,
        default=50000,
        help="Window extension length in bp (default: 50000)",
    )
    p.add_argument(
        "-step",
        "--step",
        type=int,
        default=None,
        help="Step size for sliding windows (default: extension/2)",
    )
    p.add_argument(
        "-o",
        "--out",
        default=None,
        help="Optional output prefix; writes <prefix>.beam.tsv and <prefix>.beam.windows.tsv",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        from janusx.janusx import beam_search_and_binary_mcc_bin, beam_search_and_binary_mcc_bin_indices
    except Exception as exc:
        raise SystemExit(
            "Missing Rust symbols `beam_search_and_binary_mcc_bin` / "
            "`beam_search_and_binary_mcc_bin_indices` in janusx extension. "
            "Please rebuild/reinstall JanusX (e.g. `pip install -e .` or your normal build flow)."
        ) from exc
    try:
        from janusx.janusx import beam_scan_windows_binary_mcc_bin  # type: ignore[assignment]
    except Exception:
        beam_scan_windows_binary_mcc_bin = None

    bin_path = str(Path(args.bin).expanduser())
    sample_ids, n_sites = inspect_genotype_file(bin_path)
    sample_ids = [str(s) for s in sample_ids]
    if len(sample_ids) == 0:
        raise SystemExit("No sample IDs found for BIN input.")

    pheno_df = _read_table(str(Path(args.pheno).expanduser()), args.sep, args.header)
    id_col = _parse_col_spec(args.id_col)
    y_col = _parse_col_spec(args.y_col)
    y_bin, map_msg = _align_y_to_samples(sample_ids, pheno_df, id_col, y_col)
    if map_msg:
        print(f"[beam] {map_msg}")

    beam_width = int(args.nsnp if args.nsnp is not None else (args.beam_width or 5))
    if beam_width <= 0:
        raise SystemExit("--nsnp/beam_width must be > 0")
    extension = int(args.extension)
    if extension <= 0:
        raise SystemExit("--extension must be > 0")
    step = int(args.step) if args.step is not None else max(1, extension // 2)
    if step <= 0:
        raise SystemExit("--step must be > 0")

    site_sidecar = _discover_site_sidecar(bin_path)
    best_score = float("-inf")
    selected: list[int] = []
    best_window_chrom = "ALL"
    best_window_bp_start = 0
    best_window_bp_end = 0
    best_window_n_candidates = int(n_sites)
    n_windows_eval = 0
    # (window_id, chrom, bp_start, bp_end, n_candidates, score, selected_indices)
    window_results: list[tuple[int, str, int, int, int, float, list[int]]] = []

    if beam_scan_windows_binary_mcc_bin is not None:
        if site_sidecar is None:
            print("[beam] warning: no site sidecar found, fallback to one global window")
        elif site_sidecar.lower().endswith(".bin.site"):
            print(
                "[beam] warning: legacy .bin.site has no CHR/BP metadata, fallback to one global window"
            )
        else:
            print(
                f"[beam] scanning windows in Rust from {Path(site_sidecar).name}: "
                f"extension={extension}, step={step}"
            )
        (
            sel,
            score,
            w_chrom,
            w_start,
            w_end,
            w_n_candidates,
            w_n_windows,
            w_results,
        ) = beam_scan_windows_binary_mcc_bin(
            bin_path,
            y_bin,
            max_pick=int(args.max_pick),
            beam_width=int(beam_width),
            extension=int(extension),
            step=int(step),
            return_window_results=bool(args.out),
        )
        selected = [int(i) for i in sel]
        best_score = float(score)
        best_window_chrom = str(w_chrom)
        best_window_bp_start = int(w_start)
        best_window_bp_end = int(w_end)
        best_window_n_candidates = int(w_n_candidates)
        n_windows_eval = int(w_n_windows)
        if bool(args.out):
            for item in w_results:
                wi, chrom, bp_start, bp_end, n_cand, sc, sel_idx = item
                window_results.append(
                    (
                        int(wi),
                        str(chrom),
                        int(bp_start),
                        int(bp_end),
                        int(n_cand),
                        float(sc),
                        [int(x) for x in sel_idx],
                    )
                )
    else:
        site_rows, _site_path = _load_site_rows(bin_path, int(n_sites))
        windows = _build_windows(site_rows, int(n_sites), extension, step)
        if site_sidecar is None:
            print("[beam] warning: no site sidecar found, fallback to one global window")
        elif site_sidecar.lower().endswith(".bin.site"):
            print(
                "[beam] warning: legacy .bin.site has no CHR/BP metadata, fallback to one global window"
            )
        else:
            print(
                f"[beam] scanning windows from {Path(site_sidecar).name}: "
                f"n_windows={len(windows)}, extension={extension}, step={step}"
            )

        best_selected: list[int] = []
        best_window: _Window | None = None
        if len(windows) == 1 and len(windows[0].indices) == int(n_sites):
            sel, score = beam_search_and_binary_mcc_bin(
                bin_path,
                y_bin,
                max_pick=int(args.max_pick),
                beam_width=int(beam_width),
            )
            selected = [int(i) for i in sel]
            best_score = float(score)
            best_selected = selected
            best_window = windows[0]
            n_windows_eval = 1
            if bool(args.out):
                window_results.append(
                    (
                        1,
                        best_window.chrom,
                        int(best_window.bp_start),
                        int(best_window.bp_end),
                        len(best_window.indices),
                        best_score,
                        selected,
                    )
                )
        else:
            for wi, win in enumerate(windows, start=1):
                if not win.indices:
                    continue
                sel, score = beam_search_and_binary_mcc_bin_indices(
                    bin_path,
                    y_bin,
                    [int(i) for i in win.indices],
                    max_pick=int(args.max_pick),
                    beam_width=int(beam_width),
                )
                selected_i = [int(i) for i in sel]
                score_f = float(score)
                n_windows_eval += 1
                if bool(args.out):
                    window_results.append(
                        (
                            wi,
                            win.chrom,
                            int(win.bp_start),
                            int(win.bp_end),
                            len(win.indices),
                            score_f,
                            selected_i,
                        )
                    )
                if (score_f > best_score) or (
                    np.isclose(score_f, best_score) and len(selected_i) < len(best_selected)
                ):
                    best_score = score_f
                    best_selected = selected_i
                    best_window = win
                if wi % 20 == 0 or wi == len(windows):
                    print(
                        f"[beam] progress: {wi}/{len(windows)} windows, current_best_mcc={best_score:.8f}"
                    )

        selected = [int(i) for i in best_selected]
        if best_window is None:
            raise SystemExit("No valid window/candidate was found for beam search.")
        best_window_chrom = str(best_window.chrom)
        best_window_bp_start = int(best_window.bp_start)
        best_window_bp_end = int(best_window.bp_end)
        best_window_n_candidates = int(len(best_window.indices))

    print(
        f"[beam] done: n_samples={len(sample_ids)}, n_sites={int(n_sites)}, "
        f"max_pick={int(args.max_pick)}, beam_width={int(beam_width)}, "
        f"n_windows={int(n_windows_eval)}"
    )
    print(
        f"[beam] best_window={best_window_chrom}:{best_window_bp_start}-{best_window_bp_end}, "
        f"window_candidates={best_window_n_candidates}"
    )
    print(f"[beam] best_mcc={float(best_score):.8f}")
    print("[beam] selected_indices=" + ",".join(map(str, selected)))

    if args.out:
        out_prefix = str(Path(args.out).expanduser())
        out_tsv = f"{out_prefix}.beam.tsv"
        out_windows_tsv = f"{out_prefix}.beam.windows.tsv"
        sites = _extract_sites(bin_path, selected)

        with open(out_tsv, "w", encoding="utf-8") as fw:
            fw.write(
                "rank\tsnp_index\tchrom\tpos\tref\talt\tbest_mcc\tmax_pick\tbeam_width\tn_samples\tn_sites\twindow_chrom\twindow_start\twindow_end\twindow_n_candidates\tn_windows\n"
            )
            for rank, idx in enumerate(selected, start=1):
                chrom = pos = ref = alt = ""
                if rank - 1 < len(sites):
                    s = sites[rank - 1]
                    chrom = str(getattr(s, "chrom", ""))
                    pos = str(getattr(s, "pos", ""))
                    ref = str(getattr(s, "ref_allele", ""))
                    alt = str(getattr(s, "alt_allele", ""))
                fw.write(
                    f"{rank}\t{idx}\t{chrom}\t{pos}\t{ref}\t{alt}\t"
                    f"{float(best_score):.12g}\t{int(args.max_pick)}\t{int(beam_width)}\t"
                    f"{len(sample_ids)}\t{int(n_sites)}\t{best_window_chrom}\t{best_window_bp_start}\t"
                    f"{best_window_bp_end}\t{best_window_n_candidates}\t{int(n_windows_eval)}\n"
                )
        print(f"[beam] wrote: {out_tsv}")

        with open(out_windows_tsv, "w", encoding="utf-8") as fw:
            fw.write(
                "window_id\tchrom\tbp_start\tbp_end\tn_candidates\tbest_mcc\tselected_indices\n"
            )
            for wi, chrom, bp_start, bp_end, n_candidates, score, sel in window_results:
                fw.write(
                    f"{wi}\t{chrom}\t{bp_start}\t{bp_end}\t{n_candidates}\t"
                    f"{float(score):.12g}\t{','.join(map(str, sel))}\n"
                )
        print(f"[beam] wrote: {out_windows_tsv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
