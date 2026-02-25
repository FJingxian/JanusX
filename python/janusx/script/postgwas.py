# -*- coding: utf-8 -*-
"""
JanusX: Post-GWAS Visualization and Annotation

Examples
--------
  # Basic usage with default column names (#CHROM, POS, p)
  -gwasfile result.assoc.txt

  # Specify alternative column names
  -gwasfile result.assoc.txt -chr "chr" -pos "pos" -pvalue "P_wald"

  # Specify output path and format
  -gwasfile result.assoc.txt -chr "chr" -pos "pos" -pvalue "P_wald" \
    --out test --format pdf
  # Results will be saved as:
  #   test/result.assoc.manh.pdf
  #   test/result.assoc.qq.pdf

Citation
--------
  https://github.com/FJingxian/JanusX/
"""

import logging
import os
from ._common.log import setup_logging
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_plink_prefix_exists,
)

# Ensure matplotlib uses a non-interactive backend.
for key in ["MPLBACKEND"]:
    if key in os.environ:
        del os.environ[key]

import matplotlib as mpl
mpl.use("Agg")
from janusx.bioplotkit import GWASPLOT, LDblock
from janusx.gfreader import load_genotype_chunks

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd
import numpy as np
import argparse
import re
import time
import socket
from typing import Any, Optional, Tuple
from janusx.gtools.reader import readanno
from joblib import Parallel, delayed
import warnings


def _parse_ratio(value: object, name: str) -> float:
    """Parse aspect ratio string/number: supports '2', '1.25', '5/4'."""
    if value is None:
        raise ValueError(f"{name} ratio is None.")
    text = str(value).strip()
    if text == "":
        raise ValueError(f"{name} ratio is empty.")
    if "/" in text:
        parts = text.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"{name} ratio format error: {value}")
        num = float(parts[0].strip())
        den = float(parts[1].strip())
        if den == 0:
            raise ValueError(f"{name} ratio denominator cannot be zero.")
        ratio = num / den
    else:
        ratio = float(text)
    if ratio <= 0:
        raise ValueError(f"{name} ratio must be > 0.")
    return ratio


def _parse_rgb_triplet(token: str) -> str:
    text = token.strip()
    if not (text.startswith("(") and text.endswith(")")):
        raise ValueError(f"Invalid RGB tuple token: {token}")
    parts = [p.strip() for p in text[1:-1].split(",")]
    if len(parts) != 3:
        raise ValueError(f"RGB tuple must have 3 values: {token}")
    try:
        rgb = [int(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"RGB tuple must be integers: {token}") from e
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError(f"RGB tuple values must be in [0, 255]: {token}")
    return mcolors.to_hex([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0])


def _parse_custom_pallete(text: str) -> list[str]:
    colors: list[str] = []
    for token in text.split(";"):
        tok = token.strip()
        if tok == "":
            continue
        if tok.startswith("(") and tok.endswith(")"):
            colors.append(_parse_rgb_triplet(tok))
            continue
        try:
            colors.append(mcolors.to_hex(mcolors.to_rgba(tok)))
        except ValueError as e:
            raise ValueError(
                f"Invalid --pallete color token: {tok}. "
                "Use #RRGGBB or (R,G,B)."
            ) from e
    if len(colors) == 0:
        raise ValueError("Invalid --pallete: empty color list.")
    return colors


def _parse_pallete_spec(value: object) -> Optional[Tuple[str, Any]]:
    """
    Parse --pallete into:
      - ("cmap", "<matplotlib cmap name>")
      - ("list", ["#hex1", "#hex2", ...])
      - None (use default black/grey)
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        raise ValueError("Invalid --pallete: value is empty.")
    if ";" in text:
        return ("list", _parse_custom_pallete(text))
    try:
        plt.get_cmap(text)
        return ("cmap", text)
    except ValueError:
        # Allow single explicit color token (hex/name/rgb tuple) as shorthand.
        if text.startswith("(") and text.endswith(")"):
            return ("list", [_parse_rgb_triplet(text)])
        try:
            return ("list", [mcolors.to_hex(mcolors.to_rgba(text))])
        except ValueError as e:
            raise ValueError(
                f"Invalid --pallete: {text}. "
                "Use a cmap name (e.g. tab10) or ';'-separated colors."
            ) from e


def _resolve_manhattan_colors(spec: Optional[Tuple[str, Any]], n_chr: int) -> Optional[list[str]]:
    if spec is None:
        return None
    mode, payload = spec
    if mode == "list":
        return list(payload)
    cmap = plt.get_cmap(str(payload))
    # For discrete palettes (tab10/tab20/Set*, etc.), cycle native bins directly.
    # This avoids adjacent duplicate colors when n_chr > number of bins.
    if getattr(cmap, "N", 256) <= 32:
        n_bins = max(1, int(cmap.N))
        return [mcolors.to_hex(cmap(i % n_bins)) for i in range(n_chr)]
    return [mcolors.to_hex(cmap(i / max(1, n_chr - 1))) for i in range(n_chr)]


def _natural_tokens(text: str) -> list[tuple[int, object]]:
    tokens: list[tuple[int, object]] = []
    for part in re.split(r"(\d+)", text):
        if not part:
            continue
        if part.isdigit():
            tokens.append((0, int(part)))
        else:
            tokens.append((1, part.lower()))
    return tokens


def _chrom_sort_key(label: object) -> tuple[int, object]:
    if pd.isna(label):
        return (3, "")

    if isinstance(label, (int, np.integer)):
        return (0, int(label))
    if isinstance(label, (float, np.floating)) and float(label).is_integer():
        return (0, int(label))

    text = str(label).strip()
    no_chr_prefix = text[3:] if text.lower().startswith("chr") else text
    upper = no_chr_prefix.upper()

    if no_chr_prefix.isdigit():
        return (0, int(no_chr_prefix))

    special_chr = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    if upper in special_chr:
        return (1, special_chr[upper])

    return (2, _natural_tokens(no_chr_prefix))


def _manhattan_colors_for_subset(
    spec: Optional[Tuple[str, Any]],
    full_chr_labels: list[object],
    subset_chr_labels: list[object],
) -> list[str]:
    full_order = sorted(pd.unique(pd.Series(full_chr_labels)).tolist(), key=_chrom_sort_key)
    subset_order = sorted(pd.unique(pd.Series(subset_chr_labels)).tolist(), key=_chrom_sort_key)
    if len(full_order) == 0 or len(subset_order) == 0:
        return ["black", "grey"]

    full_colors = _resolve_manhattan_colors(spec, len(full_order))
    if full_colors is None:
        full_colors = [("black" if i % 2 == 0 else "grey") for i in range(len(full_order))]

    color_by_chr = {chrom: full_colors[i % len(full_colors)] for i, chrom in enumerate(full_order)}
    return [color_by_chr[c] for c in subset_order]


def _normalize_chr(value: object) -> str:
    s = str(value).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    return s


def _parse_bimrange(value: object, logger: logging.Logger) -> tuple[str, int, int]:
    """
    Parse --bimrange.
    Supported formats:
      - chr:start-end
      - chr:start:end
    start/end are interpreted as Mb.
    """
    text = str(value).strip()
    m = re.match(r"^([^:]+):([0-9]*\.?[0-9]+)(?:-|:)([0-9]*\.?[0-9]+)$", text)
    if m is None:
        raise ValueError(
            f"Invalid --bimrange format: {value}. "
            "Use chr:start-end (or chr:start:end)."
        )
    chrom = m.group(1)
    start_mb = float(m.group(2))
    end_mb = float(m.group(3))
    if start_mb < 0 or end_mb < 0:
        raise ValueError("Invalid --bimrange: start/end must be >= 0 (Mb).")
    if start_mb > end_mb:
        logger.warning(
            f"bimrange start > end ({start_mb} > {end_mb}); swapped to {end_mb}-{start_mb} Mb."
        )
        start_mb, end_mb = end_mb, start_mb
    start = int(round(start_mb * 1_000_000))
    end = int(round(end_mb * 1_000_000))
    return chrom, start, end


def _format_bimrange_tuple(item: tuple[str, int, int]) -> str:
    chrom, start, end = item
    return f"{chrom}:{start / 1_000_000:g}-{end / 1_000_000:g} Mb"


def _merge_overlapping_bimranges(
    bimranges: list[tuple[str, int, int]],
    logger: logging.Logger,
) -> list[tuple[str, int, int]]:
    """
    Merge overlapping bimranges on the same chromosome.

    Input is assumed to be sorted by chromosome/start/end.
    """
    if len(bimranges) <= 1:
        return bimranges

    merged: list[list[object]] = []
    for chrom, start, end in bimranges:
        chrom = str(chrom)
        start_i = int(start)
        end_i = int(end)
        chrom_norm = _normalize_chr(chrom)
        if len(merged) == 0:
            merged.append([chrom, chrom_norm, start_i, end_i])
            continue

        prev = merged[-1]
        prev_chrom = str(prev[0])
        prev_norm = str(prev[1])
        prev_start = int(prev[2])
        prev_end = int(prev[3])

        if prev_norm == chrom_norm and start_i <= prev_end:
            new_start = min(prev_start, start_i)
            new_end = max(prev_end, end_i)
            logger.warning(
                "Warning: Overlapping bimrange detected; merged "
                f"{_format_bimrange_tuple((prev_chrom, prev_start, prev_end))} and "
                f"{_format_bimrange_tuple((chrom, start_i, end_i))} to "
                f"{_format_bimrange_tuple((prev_chrom, new_start, new_end))}."
            )
            prev[2] = new_start
            prev[3] = new_end
            continue

        merged.append([chrom, chrom_norm, start_i, end_i])

    return [(str(x[0]), int(x[2]), int(x[3])) for x in merged]


def _filter_df_by_bimranges(
    df: pd.DataFrame,
    chr_col: str,
    pos_col: str,
    bimranges: list[tuple[str, int, int]],
    logger: logging.Logger,
    file: str,
) -> tuple[pd.DataFrame, list[dict[str, object]], int]:
    """
    Keep SNPs in one or more bimranges.

    - Multiple ranges are supported.
    - Overlapping ranges should be merged before calling this function.
    """
    n_before = int(df.shape[0])
    if n_before == 0 or len(bimranges) == 0:
        out = df.iloc[0:0].copy()
        out["__seg_id"] = np.array([], dtype=int)
        return out, [], n_before

    chr_norm = df[chr_col].astype(str).map(_normalize_chr)
    pos_num = pd.to_numeric(df[pos_col], errors="coerce")
    seg_id = np.full(n_before, -1, dtype=int)
    seg_defs: list[dict[str, object]] = []

    for i, (chrom, start, end) in enumerate(bimranges):
        target_chr = _normalize_chr(chrom)
        mask = (chr_norm == target_chr) & (pos_num >= start) & (pos_num <= end)
        n_hit = int(mask.sum())
        if n_hit == 0:
            logger.warning(
                f"No SNPs in bimrange {_format_bimrange_tuple((chrom, start, end))} for file {file}."
            )
        else:
            logger.info(
                f"Bimrange {_format_bimrange_tuple((chrom, start, end))}: matched {n_hit} SNPs."
            )
        assign = (seg_id < 0) & mask.to_numpy()
        seg_id[assign] = i
        seg_defs.append(
            {
                "id": i,
                "chrom": str(chrom),
                "chrom_norm": target_chr,
                "start": int(start),
                "end": int(end),
                "length": float(max(1, int(end) - int(start))),
            }
        )

    keep = seg_id >= 0
    out = df.loc[keep].copy()
    out["__seg_id"] = seg_id[keep]
    return out, seg_defs, n_before


def _build_bimrange_layout(seg_defs: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(seg_defs) == 0:
        return []
    max_len = max(float(s["length"]) for s in seg_defs)
    gap = (0.06 * max_len) if len(seg_defs) > 1 else 0.0
    offset = 0.0
    layout: list[dict[str, object]] = []
    for s in seg_defs:
        seg = dict(s)
        seg["offset"] = offset
        seg["x_start"] = offset
        seg["x_end"] = offset + float(seg["length"])
        seg["label"] = (
            f"{seg['chrom']}:{int(seg['start']) / 1_000_000:g}-"
            f"{int(seg['end']) / 1_000_000:g}Mb"
        )
        layout.append(seg)
        offset = float(seg["x_end"]) + gap
    return layout


def _apply_segmented_x_to_plotmodel(
    plotmodel: GWASPLOT,
    filtered_df: pd.DataFrame,
    chr_col: str,
    pos_col: str,
    layout: list[dict[str, object]],
) -> None:
    if len(layout) == 0 or filtered_df.shape[0] == 0:
        return

    # (chrom_norm, pos) -> seg_id
    chr_vals = filtered_df[chr_col].astype(str).map(_normalize_chr).to_numpy()
    pos_vals = pd.to_numeric(filtered_df[pos_col], errors="coerce").to_numpy()
    seg_vals = filtered_df["__seg_id"].to_numpy()
    key_to_seg: dict[tuple[str, int], int] = {}
    for c, p, sid in zip(chr_vals, pos_vals, seg_vals):
        if not np.isfinite(p):
            continue
        key = (str(c), int(round(float(p))))
        if key not in key_to_seg:
            key_to_seg[key] = int(sid)

    layout_by_id = {int(seg["id"]): seg for seg in layout}

    # plotmodel uses integer chr IDs; map them back to normalized chr labels.
    id_to_chr_norm = {
        i + 1: _normalize_chr(label) for i, label in enumerate(plotmodel.chr_labels)
    }

    idx_chr = np.asarray(plotmodel.df.index.get_level_values(0), dtype=np.int64)
    idx_pos = np.asarray(plotmodel.df.index.get_level_values(1), dtype=np.int64)
    x_new = np.asarray(plotmodel.df["x"], dtype=float).copy()
    for j, (cid, pos) in enumerate(zip(idx_chr, idx_pos)):
        chr_norm = id_to_chr_norm.get(int(cid))
        if chr_norm is None:
            continue
        sid = key_to_seg.get((chr_norm, int(pos)))
        if sid is None:
            continue
        seg = layout_by_id.get(int(sid))
        if seg is None:
            continue
        start = int(seg["start"])
        offset = float(seg["offset"])
        length = float(seg["length"])
        rel = float(pos - start)
        rel = min(max(rel, 0.0), length)
        x_new[j] = offset + rel

    plotmodel.df["x"] = x_new
    plotmodel._janusx_bim_layout = layout  # type: ignore[attr-defined]


def _apply_bimrange_manhattan_axis(
    ax: plt.Axes,
    chrom_label: object,
    start_bp: int,
    end_bp: int,
) -> None:
    left = float(min(start_bp, end_bp))
    right = float(max(start_bp, end_bp))
    if right <= left:
        return
    ax.set_xlim(left, right)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1_000_000:g}Mb"))
    ax.set_xlabel(str(chrom_label))
    ax._janusx_loc_left_label = f"{left / 1_000_000:g}Mb"   # type: ignore[attr-defined]
    ax._janusx_loc_right_label = f"{right / 1_000_000:g}Mb"  # type: ignore[attr-defined]


def _apply_multi_bimrange_manhattan_axis(
    ax: plt.Axes,
    layout: list[dict[str, object]],
) -> None:
    if len(layout) == 0:
        return
    left = float(layout[0]["x_start"])
    right = float(layout[-1]["x_end"])
    if not right > left:
        return
    ax.set_xlim(left, right)
    centers = [0.5 * (float(seg["x_start"]) + float(seg["x_end"])) for seg in layout]
    labels = [str(seg["label"]) for seg in layout]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    for i in range(len(layout) - 1):
        x_end = float(layout[i]["x_end"])
        x_next = float(layout[i + 1]["x_start"])
        xb = 0.5 * (x_end + x_next)
        ax.axvline(x=xb, color="black", linestyle=":", linewidth=0.7, alpha=0.9)
        prev_end_lab = f"{layout[i]['chrom']}:{int(layout[i]['end']) / 1_000_000:g}Mb"
        next_start_lab = (
            f"{layout[i + 1]['chrom']}:{int(layout[i + 1]['start']) / 1_000_000:g}Mb"
        )
        x_shift = 0.006 * (right - left)
        trans = ax.get_xaxis_transform()
        ax.text(
            xb - x_shift,
            -0.08,
            prev_end_lab,
            transform=trans,
            ha="right",
            va="top",
            fontsize=5,
            clip_on=False,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 0.2,
            },
        )
        ax.text(
            xb + x_shift,
            -0.08,
            next_start_lab,
            transform=trans,
            ha="left",
            va="top",
            fontsize=5,
            clip_on=False,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 0.2,
            },
        )
    first = layout[0]
    last = layout[-1]
    ax._janusx_loc_left_label = f"{first['chrom']}:{int(first['start']) / 1_000_000:g}Mb"   # type: ignore[attr-defined]
    ax._janusx_loc_right_label = f"{last['chrom']}:{int(last['end']) / 1_000_000:g}Mb"      # type: ignore[attr-defined]
    ax.set_xlabel(None)


def _extract_ld_site_set(
    df: pd.DataFrame,
    chr_col: str,
    pos_col: str,
    p_col: str,
    threshold: float,
    use_all_sites: bool,
) -> set[tuple[str, int]]:
    pvals = pd.to_numeric(df[p_col], errors="coerce")
    pos = pd.to_numeric(df[pos_col], errors="coerce")
    mask_valid = pos.notna() & pvals.notna() & np.isfinite(pvals) & (pvals > 0.0)
    if use_all_sites:
        mask = mask_valid
    else:
        mask = mask_valid & (pvals <= threshold)
    out: set[tuple[str, int]] = set()
    if not bool(mask.any()):
        return out
    for c, p in zip(df.loc[mask, chr_col], pos.loc[mask]):
        out.add((_normalize_chr(c), int(round(float(p)))))
    return out


def _collect_genotypes_for_sites(
    genofile: str,
    bimrange_tuples: list[tuple[str, int, int]],
    site_set: set[tuple[str, int]],
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    if len(site_set) == 0:
        return np.zeros((0, 0), dtype=np.float32), []
    if len(bimrange_tuples) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    rows: list[np.ndarray] = []
    keys: list[tuple[str, int]] = []

    for bchrom, bstart, bend in bimrange_tuples:
        for chunk, sites in load_genotype_chunks(
            genofile,
            chunk_size=50_000,
            maf=0.0,
            missing_rate=1.0,
            impute=True,
            bim_range=(str(bchrom), int(bstart), int(bend)),
        ):
            chunk_np = np.asarray(chunk, dtype=np.float32)
            for i, s in enumerate(sites):
                key = (_normalize_chr(s.chrom), int(s.pos))
                if key in site_set:
                    rows.append(chunk_np[i, :].copy())
                    keys.append(key)

    if len(rows) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    # keep deterministic genomic order and deduplicate same-site entries
    uniq: dict[tuple[str, int], np.ndarray] = {}
    for k, r in zip(keys, rows):
        if k not in uniq:
            uniq[k] = r
    sorted_keys = sorted(uniq.keys(), key=lambda x: (x[0], x[1]))
    mat = np.stack([uniq[k] for k in sorted_keys], axis=0).astype(np.float32)
    return mat, sorted_keys


def _compute_ld_from_genotypes(geno_sig: np.ndarray) -> np.ndarray:
    if geno_sig.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if geno_sig.shape[0] == 1:
        return np.ones((1, 1), dtype=np.float32)
    corr = np.corrcoef(geno_sig)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr).astype(np.float32)
    np.fill_diagonal(corr, 1.0)
    return corr


def _draw_empty_ldblock(
    ax: plt.Axes,
    *,
    n_sites: int = 2,
    text: Optional[str] = None,
) -> None:
    n = max(2, int(n_sites))
    LDblock(np.zeros((n, n), dtype=np.float32), ax=ax, vmin=0, vmax=1, cmap="Greys")
    if text:
        ax.text(n / 2.0, -n / 2.0, text, ha="center", va="center", fontsize=6)


def _format_input_files(files: list[str]) -> str:
    if len(files) <= 1:
        return files[0]
    return f"{files[0]},...({len(files)} files)"


def _format_bimrange_summary(
    bimranges: Optional[list[tuple[str, int, int]]]
) -> str:
    if bimranges is None or len(bimranges) == 0:
        return "None"
    if len(bimranges) == 1:
        return _format_bimrange_tuple(bimranges[0])
    return f"{_format_bimrange_tuple(bimranges[0])},...({len(bimranges)} ranges)"


def _overlay_manhattan_threshold_points(
    ax: plt.Axes,
    plotmodel: GWASPLOT,
    *,
    threshold: float,
    base_size: float,
    rasterized: bool,
    min_logp: float = 0.5,
    ignore: Optional[list[object]] = None,
) -> None:
    """
    Overlay threshold line and significant points on top of Manhattan base points.
    Significant points are enlarged by 1.5x.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        return

    if ignore is None:
        ignore = []
    ignore_set = set(ignore)

    dfp = plotmodel.df.iloc[plotmodel.minidx, -3:].copy()
    pvals = pd.to_numeric(dfp["y"], errors="coerce")
    keep = pvals.notna() & np.isfinite(pvals) & (pvals > 0.0)
    if not bool(keep.any()):
        return
    dfp = dfp.loc[keep].copy()
    dfp["ylog"] = -np.log10(pd.to_numeric(dfp["y"], errors="coerce"))
    dfp = dfp[dfp["ylog"] >= float(min_logp)]
    if dfp.shape[0] == 0:
        return

    thr_log = float(-np.log10(threshold))
    if not np.isfinite(thr_log):
        return

    sig_mask = dfp["ylog"] >= thr_log
    if len(ignore_set) > 0:
        sig_mask = sig_mask & (~dfp.index.isin(ignore_set))

    if bool(sig_mask.any()):
        ax.scatter(
            dfp.loc[sig_mask, "x"],
            dfp.loc[sig_mask, "ylog"],
            color="red",
            s=float(base_size) * 1.5,
            rasterized=rasterized,
            zorder=6,
        )
    ax.axhline(
        y=thr_log,
        linestyle="dashed",
        color="grey",
        linewidth=1.0,
    )


def GWASplot(file: str, args, logger:logging.Logger) -> None:
    """
    Plot Manhattan/QQ figures and optionally annotate significant hits
    for a single GWAS result file.
    """
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.size"] = 6
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = False

    # Silence pandas chained-assignment warnings in this script
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*ChainedAssignmentError.*",
    )

    args.prefix = (
        os.path.basename(file)
        .replace(".tsv", "")
        .replace(".txt", "")
    )

    chr_col, pos_col, p_col = args.chr, args.pos, args.pvalue

    df_all = pd.read_csv(file, sep="\t", usecols=[chr_col, pos_col, p_col])
    full_chr_labels = df_all[chr_col].drop_duplicates().tolist()
    df = df_all
    bim_layout: list[dict[str, object]] = []
    if args.bimrange_tuples is not None:
        df_sel, seg_defs, n_before = _filter_df_by_bimranges(
            df_all,
            chr_col,
            pos_col,
            args.bimrange_tuples,
            logger,
            file,
        )
        n_after = int(df_sel.shape[0])
        if n_after == 0:
            logger.warning(
                f"No SNPs found in all bimrange settings for file {file}; skipped."
            )
            # Keep LD-only no-genotype flow alive: draw a zero-correlation block.
            if not (
                args.ldblock_ratio is not None
                and args.genofile is None
                and args.manh_ratio is None
                and args.qq_ratio is None
            ):
                return
            df = df_sel
        else:
            df = df_sel
            bim_layout = _build_bimrange_layout(seg_defs)
            logger.info(
                f"Applied {len(args.bimrange_tuples)} bimrange settings: kept {n_after}/{n_before} SNPs."
            )
    # Keep positional index contiguous to avoid iloc out-of-bounds in plot model.
    df = df.reset_index(drop=True)

    # Bonferroni-style default threshold if not provided
    threshold = (
        args.threshold
        if args.threshold is not None
        else (0.05 / df.shape[0] if df.shape[0] > 0 else np.nan)
    )
    effective_ldblock_ratio = args.ldblock_ratio
    if effective_ldblock_ratio is not None and args.bimrange_tuples is None:
        logger.warning(
            "Warning: --ldblock/--ldblock-all requires --bimrange; LD block and Manhattan+LD plotting are skipped."
        )
        effective_ldblock_ratio = None

    # ------------------------------------------------------------------
    # 1. Visualization: Manhattan & QQ
    # ------------------------------------------------------------------
    if args.manh_ratio is not None or args.qq_ratio is not None or effective_ldblock_ratio is not None:
        t_plot = time.time()
        logger.info("* Visualizing GWAS results...")
        ld_use_all_sites = bool(args.ldblock_all is not None)

        need_manh_panel = args.manh_ratio is not None or effective_ldblock_ratio is not None
        plotmodel = None
        if (need_manh_panel or args.qq_ratio is not None) and df.shape[0] > 0:
            plotmodel = GWASPLOT(
                df,
                chr_col,
                pos_col,
                p_col,
                0.1,
                compression=(not args.disable_compression),
            )
            if args.bimrange_tuples is not None and len(bim_layout) > 1:
                _apply_segmented_x_to_plotmodel(plotmodel, df, chr_col, pos_col, bim_layout)
        width_in = 8.0
        dpi = 300
        manh_ylim = None
        manh_height_in = None
        manh_fontsize = None
        manh_yticks = None
        manh_axes_bounds = None
        hide_axis_labels = args.manh_ratio is not None and args.manh_ratio > 4.0
        manh_min_logp = 0.0 if args.bimrange_tuples is not None else 0.5
        manh_ymin = 0.0 if args.bimrange_tuples is not None else 0.5

        plot_colors = None
        if plotmodel is not None:
            plot_colors = _manhattan_colors_for_subset(
                args.pallete_spec,
                full_chr_labels,
                df[chr_col].drop_duplicates().tolist(),
            )

        def _draw_manhattan_axis(ax: plt.Axes) -> tuple[tuple[float, float], np.ndarray, float]:
            if plotmodel is None:
                ax.text(0.5, 0.5, "No SNPs", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                if hide_axis_labels:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                return ax.get_ylim(), np.asarray(ax.get_yticks(), dtype=float), float(ax.xaxis.label.get_size())

            rasterized = False if args.format == "pdf" else True

            if args.highlight:
                # Highlight specific SNPs (bed-like file: chr, start, end, gene, desc)
                df_hl = pd.read_csv(args.highlight, sep="\t", header=None)
                gene_mask = df_hl[3].isna()
                df_hl.loc[gene_mask, 3] = (
                    df_hl.loc[gene_mask, 0].astype(str)
                    + "_"
                    + df_hl.loc[gene_mask, 1].astype(str)
                )
                df_hl = df_hl.set_index([0, 1])

                # Intersect highlight positions with SNPs in the plot model
                df_hl_idx = df_hl.index[df_hl.index.isin(plotmodel.df.index)]
                if len(df_hl_idx) == 0:
                    logger.warning("Nothing to highlight. Check the BED file.")
                    plotmodel.manhattan(
                        None,
                        ax=ax,
                        color_set=plot_colors,
                        min_logp=manh_min_logp,
                        y_min=manh_ymin,
                        s=args.scatter_size,
                        rasterized=rasterized,
                    )
                    _overlay_manhattan_threshold_points(
                        ax,
                        plotmodel,
                        threshold=threshold,
                        base_size=args.scatter_size,
                        rasterized=rasterized,
                        min_logp=manh_min_logp,
                    )
                else:
                    ax.scatter(
                        plotmodel.df.loc[df_hl_idx, "x"],
                        -np.log10(plotmodel.df.loc[df_hl_idx, "y"]),
                        marker="D",
                        color="red",
                        zorder=10,
                        s=args.scatter_size,
                        edgecolors="black",
                    )
                    for idx in df_hl_idx:
                        text = df_hl.loc[idx, 3]
                        ax.text(
                            plotmodel.df.loc[idx, "x"],
                            -np.log10(plotmodel.df.loc[idx, "y"]),
                            s=text,
                            ha="center",
                            zorder=11,
                        )

                    plotmodel.manhattan(
                        None,
                        ax=ax,
                        color_set=plot_colors,
                        min_logp=manh_min_logp,
                        y_min=manh_ymin,
                        s=args.scatter_size,
                        ignore=df_hl_idx,
                        rasterized=rasterized,
                    )
                    _overlay_manhattan_threshold_points(
                        ax,
                        plotmodel,
                        threshold=threshold,
                        base_size=args.scatter_size,
                        rasterized=rasterized,
                        min_logp=manh_min_logp,
                        ignore=list(df_hl_idx),
                    )
            else:
                plotmodel.manhattan(
                    None,
                    ax=ax,
                    color_set=plot_colors,
                    min_logp=manh_min_logp,
                    y_min=manh_ymin,
                    s=args.scatter_size,
                    rasterized=rasterized,
                )
                _overlay_manhattan_threshold_points(
                    ax,
                    plotmodel,
                    threshold=threshold,
                    base_size=args.scatter_size,
                    rasterized=rasterized,
                    min_logp=manh_min_logp,
                )

            if args.bimrange_tuples is not None:
                if len(bim_layout) > 1:
                    _apply_multi_bimrange_manhattan_axis(ax, bim_layout)
                elif len(args.bimrange_tuples) == 1:
                    bchrom, bstart, bend = args.bimrange_tuples[0]
                    _apply_bimrange_manhattan_axis(ax, bchrom, bstart, bend)
            if args.manh_ylim is not None:
                ymin_now, _ = ax.get_ylim()
                if args.manh_ylim <= ymin_now:
                    logger.warning(
                        f"Warning: manh-ylim ({args.manh_ylim}) <= current ymin ({ymin_now:.4g}); ignored."
                    )
                else:
                    ax.set_ylim(ymin_now, args.manh_ylim)
            if hide_axis_labels:
                ax.set_xlabel("")
                ax.set_ylabel("")

            return (
                ax.get_ylim(),
                np.asarray(ax.get_yticks(), dtype=float),
                float(ax.xaxis.label.get_size()),
            )

        # ----------------- Manhattan plot -----------------
        if args.manh_ratio is not None:
            manh_h_in = width_in / args.manh_ratio
            fig = plt.figure(
                figsize=(width_in, manh_h_in),
                dpi=dpi,
            )
            ax = fig.add_subplot(111)
            manh_ylim, manh_yticks, manh_fontsize = _draw_manhattan_axis(ax)
            manh_height_in = fig.get_figheight()
            fig.tight_layout()
            manh_axes_bounds = ax.get_position().bounds
            manh_path = f"{args.out}/{args.prefix}.manh.{args.format}"
            fig.savefig(manh_path, transparent=True)
            plt.close(fig)
        else:
            manh_path = None

        # ----------------- QQ plot -----------------
        if args.qq_ratio is not None:
            if plotmodel is None:
                logger.warning("Warning: QQ plotting skipped because no SNPs are available.")
                qq_path = None
            else:
                qq_y_in = 4.0
                if manh_height_in is not None:
                    qq_y_in = manh_height_in
                qq_x_in = qq_y_in * args.qq_ratio
                fig = plt.figure(
                    figsize=(qq_x_in, qq_y_in),
                    dpi=dpi,
                )
                ax2 = fig.add_subplot(111)
                plotmodel.qq(
                    ax=ax2,
                    color_set=["black", "grey"],
                    scatter_size=args.scatter_size,
                )
                qq_xmax = float(np.log10(plotmodel.df.shape[0] + 1))
                if np.isfinite(qq_xmax) and qq_xmax > 0:
                    cur_left, _ = ax2.get_xlim()
                    ax2.set_xlim(cur_left, qq_xmax)
                if manh_ylim is not None:
                    ax2.set_ylim(manh_ylim)
                xlim_now = ax2.get_xlim()
                ylim_now = ax2.get_ylim()
                shared_min = float(max(xlim_now[0], ylim_now[0]))
                ax2.set_xlim(shared_min, xlim_now[1])
                ax2.set_ylim(shared_min, ylim_now[1])
                if manh_yticks is not None:
                    y0, y1 = ax2.get_ylim()
                    lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
                    yticks = [
                        float(t) for t in manh_yticks
                        if (t >= lo - 1e-12) and (t <= hi + 1e-12)
                    ]
                    if not np.isclose(shared_min, 0.0):
                        yticks = [t for t in yticks if not np.isclose(t, 0.0)]
                    ax2.set_yticks(yticks)
                if not np.isclose(shared_min, 0.0):
                    xticks = [
                        t for t in ax2.get_xticks()
                        if (not np.isclose(t, shared_min)) and (not np.isclose(t, 0.0))
                    ]
                    ax2.set_xticks(xticks)
                    if manh_yticks is None:
                        yticks = [
                            t for t in ax2.get_yticks()
                            if (not np.isclose(t, shared_min)) and (not np.isclose(t, 0.0))
                        ]
                        ax2.set_yticks(yticks)
                if manh_fontsize is not None:
                    ax2.xaxis.label.set_size(manh_fontsize)
                    ax2.yaxis.label.set_size(manh_fontsize)
                    ax2.tick_params(axis="both", labelsize=manh_fontsize)
                if hide_axis_labels:
                    ax2.set_xlabel("")
                    ax2.set_ylabel("")
                fig.tight_layout()
                qq_path = f"{args.out}/{args.prefix}.qq.{args.format}"
                fig.savefig(qq_path, transparent=True)
                plt.close(fig)
        else:
            qq_path = None

        # ----------------- LD block -----------------
        if effective_ldblock_ratio is not None:
            ld_sites = _extract_ld_site_set(
                df,
                chr_col,
                pos_col,
                p_col,
                threshold,
                use_all_sites=ld_use_all_sites,
            )
            n_sig_sites = max(2, len(ld_sites))
            ld_overlay_text = None
            ld_site_keys: list[tuple[str, int]] = sorted(ld_sites, key=lambda x: (x[0], x[1]))

            if args.genofile is None:
                logger.warning(
                    "Warning: --ldblock/--ldblock-all enabled but no genotype file provided; drawing zero-correlation LD block."
                )
                ld_mat = np.zeros((n_sig_sites, n_sig_sites), dtype=np.float32)
                ld_overlay_text = "No genotype"
            else:
                if len(ld_sites) < 2:
                    if ld_use_all_sites:
                        logger.warning(
                            "Warning: Fewer than 2 valid SNPs in selected region; drawing empty LD block."
                        )
                    else:
                        logger.warning(
                            "Warning: Fewer than 2 threshold-passing SNPs in selected region; drawing empty LD block."
                        )
                    ld_mat = np.zeros((n_sig_sites, n_sig_sites), dtype=np.float32)
                    ld_overlay_text = "Not enough SNPs"
                else:
                    geno_sig, sig_keys = _collect_genotypes_for_sites(
                        args.genofile,
                        args.bimrange_tuples if args.bimrange_tuples is not None else [],
                        ld_sites,
                    )
                    ld_site_keys = sig_keys
                    if geno_sig.shape[0] < 2:
                        logger.warning(
                            "Warning: Requested SNPs were not found in genotype data; drawing empty LD block."
                        )
                        ld_mat = np.zeros((n_sig_sites, n_sig_sites), dtype=np.float32)
                        ld_overlay_text = "No matched SNPs"
                    else:
                        ld_mat = _compute_ld_from_genotypes(geno_sig)
                        mode_text = "all SNPs" if ld_use_all_sites else "threshold-passing SNPs"
                        logger.info(f"LD block built from {len(sig_keys)} {mode_text}.")

            def _draw_ld_axis(ax: plt.Axes) -> None:
                LDblock(ld_mat.copy(), ax=ax, vmin=0, vmax=1, cmap="Greys")
                if ld_overlay_text:
                    n_ld = int(ld_mat.shape[0])
                    ax.text(n_ld / 2.0, -n_ld / 2.0, ld_overlay_text, ha="center", va="center", fontsize=6)

            def _build_manh_ld_pairs(ax_manh: plt.Axes, ax_ld: plt.Axes) -> list[tuple[float, float]]:
                if plotmodel is None or len(ld_site_keys) == 0:
                    return []

                # Keep mapping behavior consistent with Manhattan plotting:
                # use compressed subset (minidx), then select by ld mode.
                map_df = plotmodel.df.iloc[plotmodel.minidx, -3:].copy()
                p_vals = pd.to_numeric(map_df["y"], errors="coerce").to_numpy(dtype=float)
                keep_mask = np.isfinite(p_vals) & (p_vals > 0.0)
                if not ld_use_all_sites:
                    keep_mask = keep_mask & (p_vals <= threshold)
                if not bool(np.any(keep_mask)):
                    return []

                chr_id_to_norm = {
                    i + 1: _normalize_chr(label) for i, label in enumerate(plotmodel.chr_labels)
                }
                chr_ids = np.asarray(map_df.index.get_level_values(0), dtype=np.int64)
                pos_vals = np.asarray(map_df.index.get_level_values(1), dtype=np.int64)
                x_vals = np.asarray(map_df["x"], dtype=float)
                key_to_x: dict[tuple[str, int], float] = {}
                for cid, p, x, keep in zip(chr_ids, pos_vals, x_vals, keep_mask):
                    if not keep:
                        continue
                    chrom_norm = chr_id_to_norm.get(int(cid))
                    if chrom_norm is None:
                        continue
                    key = (chrom_norm, int(p))
                    if key not in key_to_x:
                        key_to_x[key] = float(x)

                n_ld = int(ld_mat.shape[0])
                keys = ld_site_keys[:n_ld]
                pairs: list[tuple[float, float]] = []
                for i, key in enumerate(keys):
                    x_top = key_to_x.get((str(key[0]), int(key[1])))
                    if x_top is None:
                        continue
                    # In LDblock, SNP i is centered around x=i+0.5 on the top border.
                    x_ld = float(i) + 0.5
                    pairs.append((x_top, x_ld))
                return pairs

            def _draw_bridge_axis(
                ax_bridge: plt.Axes,
                ax_manh: plt.Axes,
                ax_ld: plt.Axes,
                pairs: list[tuple[float, float]],
            ) -> None:
                ax_bridge.set_xlim(0.0, 1.0)
                ax_bridge.set_ylim(0.0, 1.0)
                ax_bridge.set_xticks([])
                ax_bridge.set_yticks([])
                for spine in ax_bridge.spines.values():
                    spine.set_visible(False)
                if len(pairs) == 0:
                    return

                manh_x0, manh_x1 = ax_manh.get_xlim()
                ld_x0, ld_x1 = ax_ld.get_xlim()
                d_manh = float(manh_x1 - manh_x0)
                d_ld = float(ld_x1 - ld_x0)
                if np.isclose(d_manh, 0.0) or np.isclose(d_ld, 0.0):
                    return

                slashes_x: list[float] = []
                edge_margin_n = 0.006
                for x_top, x_ld in pairs:
                    x_top_n = (float(x_top) - manh_x0) / d_manh
                    x_ld_n = (float(x_ld) - ld_x0) / d_ld
                    if not (np.isfinite(x_top_n) and np.isfinite(x_ld_n)):
                        continue
                    x_top_n = float(np.clip(x_top_n, edge_margin_n, 1.0 - edge_margin_n))
                    x_ld_n = float(np.clip(x_ld_n, edge_margin_n, 1.0 - edge_margin_n))
                    joint_y = 0.84
                    bottom_y = 0.06
                    # Draw as one polyline to avoid tiny rendering gap at the joint.
                    ax_bridge.plot(
                        [x_top_n, x_top_n, x_ld_n],
                        [1.04, joint_y, bottom_y],
                        color="black",
                        linewidth=0.5,
                        alpha=0.8,
                        clip_on=False,
                        solid_joinstyle="round",
                    )
                    slashes_x.append(float(x_top))

            def _show_end_locs_without_xticks(ax: plt.Axes) -> None:
                x0, x1 = ax.get_xlim()
                fmt = ax.xaxis.get_major_formatter()

                def _fmt(v: float, i: int) -> str:
                    if callable(fmt):
                        try:
                            s = str(fmt(v, i)).strip()
                            if s != "":
                                return s
                        except Exception:
                            pass
                    return f"{v:g}"

                left_lab = getattr(ax, "_janusx_loc_left_label", None)
                right_lab = getattr(ax, "_janusx_loc_right_label", None)
                if left_lab is None:
                    left_lab = _fmt(float(x0), 0)
                if right_lab is None:
                    right_lab = _fmt(float(x1), 1)

                ax.set_xticks([])
                ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
                trans = ax.get_xaxis_transform()
                ax.text(
                    float(x0),
                    -0.06,
                    left_lab,
                    transform=trans,
                    ha="left",
                    va="top",
                    clip_on=False,
                    zorder=40,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.55,
                        "pad": 0.2,
                    },
                )
                if not np.isclose(float(x0), float(x1)):
                    ax.text(
                        float(x1),
                        -0.06,
                        right_lab,
                        transform=trans,
                        ha="right",
                        va="top",
                        clip_on=False,
                        zorder=40,
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.55,
                            "pad": 0.2,
                        },
                    )

            ld_h_in = width_in / effective_ldblock_ratio
            fig_ld = plt.figure(
                figsize=(width_in, ld_h_in),
                dpi=dpi,
            )
            ax_ld = fig_ld.add_subplot(111)
            ld_path = f"{args.out}/{args.prefix}.ldblock.{args.format}"
            _draw_ld_axis(ax_ld)

            fig_ld.tight_layout()
            # Keep LD block x-axis drawable length consistent with Manhattan.
            if manh_axes_bounds is not None:
                mx0, _my0, mw, _mh = manh_axes_bounds
                _cx0, cy0, _cw, ch = ax_ld.get_position().bounds
                ax_ld.set_position([mx0, cy0, mw, ch])
            fig_ld.savefig(ld_path, transparent=True)
            plt.close(fig_ld)

            # Combined Manhattan + LD panel
            manhld_manh_ratio = args.manh_ratio if args.manh_ratio is not None else 2.0
            manhld_manh_h_in = width_in / manhld_manh_ratio
            bridge_h_in = 0.3
            manhld_total_h_in = manhld_manh_h_in + bridge_h_in + ld_h_in
            fig_manhld = plt.figure(
                figsize=(width_in, manhld_total_h_in),
                dpi=dpi,
            )
            gs_manhld = fig_manhld.add_gridspec(
                3,
                1,
                height_ratios=[manhld_manh_h_in, bridge_h_in, ld_h_in],
                hspace=0.04,
            )
            ax_manhld_top = fig_manhld.add_subplot(gs_manhld[0, 0])
            ax_manhld_mid = fig_manhld.add_subplot(gs_manhld[1, 0])
            ax_manhld_bot = fig_manhld.add_subplot(gs_manhld[2, 0])
            # Keep transition axis below Manhattan axis so loc labels remain visible.
            ax_manhld_top.set_zorder(5)
            ax_manhld_mid.set_zorder(2)
            ax_manhld_bot.set_zorder(1)
            ax_manhld_mid.patch.set_visible(False)

            _draw_manhattan_axis(ax_manhld_top)
            _draw_ld_axis(ax_manhld_bot)
            fig_manhld.subplots_adjust(
                left=0.08,
                right=0.98,
                top=0.98,
                bottom=0.07,
                hspace=0.04,
            )

            # Keep the two panels strictly left/right aligned.
            # LD axis uses fixed aspect and may shrink during draw; align Manhattan to LD.
            fig_manhld.canvas.draw()
            _tx0, ty0, _tw, th = ax_manhld_top.get_position().bounds
            _mx0, my0, _mw, mh = ax_manhld_mid.get_position().bounds
            bx0, _by0, bw, _bh = ax_manhld_bot.get_position().bounds
            ax_manhld_top.set_position([bx0, ty0, bw, th])
            ax_manhld_mid.set_position([bx0, my0, bw, mh])

            bridge_pairs = _build_manh_ld_pairs(ax_manhld_top, ax_manhld_bot)
            _draw_bridge_axis(ax_manhld_mid, ax_manhld_top, ax_manhld_bot, bridge_pairs)
            _show_end_locs_without_xticks(ax_manhld_top)

            manhld_path = f"{args.out}/{args.prefix}.manhld.{args.format}"
            fig_manhld.savefig(manhld_path, transparent=True)
            plt.close(fig_manhld)
        else:
            ld_path = None
            manhld_path = None

        saved_paths: list[tuple[str, str]] = []
        if manh_path is not None:
            saved_paths.append(("Manhattan", manh_path))
        if qq_path is not None:
            saved_paths.append(("QQ", qq_path))
        if ld_path is not None:
            saved_paths.append(("LD block", ld_path))
        if manhld_path is not None:
            saved_paths.append(("Manhattan+LD", manhld_path))

        if len(saved_paths) == 1:
            logger.info(f"{saved_paths[0][0]} plot saved to:\n  {saved_paths[0][1]}")
        elif len(saved_paths) > 1:
            title = ", ".join([x[0] for x in saved_paths])
            body = "\n".join([f"  {x[1]}" for x in saved_paths])
            logger.info(f"{title} plots saved to:\n{body}")
        logger.info(f"Visualization completed in {round(time.time() - t_plot, 2)} seconds.\n")

    # ------------------------------------------------------------------
    # 2. Annotation of significant loci
    # ------------------------------------------------------------------
    if args.anno:
        logger.info("* Annotating significant SNPs...")
        if os.path.exists(args.anno):
            t_anno = time.time()

            # Keep SNPs passing threshold
            df_filter = df.loc[
                df[p_col] <= threshold,
                [chr_col, pos_col, p_col],
            ].set_index([chr_col, pos_col])

            # Read annotation (GFF/bed) into unified annotation table
            # After readanno:
            #   anno[0] = chr
            #   anno[1] = start
            #   anno[2] = end
            #   anno[3] = gene ID
            #   anno[4], anno[5] = description fields
            anno = readanno(args.anno, args.descItem)
            anno_chr = anno[0].astype(str)

            # Exact overlap annotation
            desc_exact = [
                anno.loc[
                    (anno_chr == str(idx[0]))
                    & (anno[1] <= idx[1])
                    & (anno[2] >= idx[1])
                ]
                for idx in df_filter.index
            ]
            df_filter["desc"] = [
                (
                    f"{x.iloc[0, 3]};{x.iloc[0, 4]};{x.iloc[0, 5]}"
                    if not x.empty
                    else "NA;NA;NA"
                )
                for x in desc_exact
            ]

            # Optional broadened window around SNP (± annobroaden kb)
            if args.annobroaden is not None:
                kb = args.annobroaden * 1_000
                desc_broad = [
                    anno.loc[
                        (anno_chr == str(idx[0]))
                        & (anno[1] <= idx[1] + kb)
                        & (anno[2] >= idx[1] - kb)
                    ]
                    for idx in df_filter.index
                ]
                df_filter["broaden"] = [
                    (
                        f"{'|'.join(x.iloc[:, 3])};"
                        f"{'|'.join(x.iloc[:, 4])};"
                        f"{'|'.join(x.iloc[:, 5])}"
                        if not x.empty
                        else "NA;NA;NA"
                    )
                    for x in desc_broad
                ]
            else:
                if "broaden" in df_filter.columns:
                    df_filter = df_filter.drop(columns=["broaden"])

            logger.info(df_filter)

            anno_path = f"{args.out}/{args.prefix}.{threshold}.anno.tsv"
            df_filter.to_csv(anno_path, sep="\t")
            logger.info(f"Annotation table saved to {anno_path}")
            logger.info(f"Annotation completed in {round(time.time() - t_anno, 2)} seconds.\n")
        else:
            logger.info(f"Annotation file not found: {args.anno}\n")


def main():
    t_start = time.time()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ------------------------------------------------------------------
    # Required arguments
    # ------------------------------------------------------------------
    required_group = parser.add_argument_group("Required Arguments")
    required_group.add_argument(
        "-gwasfile", "--gwasfile", nargs="+", type=str, required=True,
        help="One or more GWAS result files (tab-delimited).",
    )

    # ------------------------------------------------------------------
    # Optional arguments
    # ------------------------------------------------------------------
    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-chr", "--chr", type=str, default="chrom",
        help="Column name for chromosome (default: %(default)s).",
    )
    optional_group.add_argument(
        "-pos", "--pos", type=str, default="pos",
        help="Column name for base position (default: %(default)s).",
    )
    optional_group.add_argument(
        "-pvalue", "--pvalue", type=str, default="pwald",
        help="Column name for p-value (default: %(default)s).",
    )
    optional_group.add_argument(
        "-threshold", "--threshold", type=float, default=None,
        help="P-value threshold; if not set, use 0.05 / nSNP (default: %(default)s).",
    )
    optional_group.add_argument(
        "-bimrange", "--bimrange", type=str, action="append", default=None,
        help=(
            "Plotting range filter in Mb, format chr:start-end "
            "(also accepts chr:start:end). Can be specified multiple times."
        ),
    )
    optional_group.add_argument(
        "-manh", "--manh", type=str, nargs="?", const="2", default=None,
        help=(
            "Enable Manhattan plotting with aspect ratio (width/height). "
            "Examples: --manh (default 2), --manh 2, --manh 3/2."
        ),
    )
    ldblock_group = optional_group.add_mutually_exclusive_group(required=False)
    ldblock_group.add_argument(
        "-ldblock", "--ldblock", type=str, nargs="?", const="2", default=None,
        help=(
            "Enable LD block inverted triangle plotting with aspect ratio (width/height), "
            "using only threshold-passing SNPs. Requires --bimrange."
        ),
    )
    ldblock_group.add_argument(
        "-ldblock-all", "--ldblock-all", dest="ldblock_all", type=str, nargs="?", const="2", default=None,
        help=(
            "Enable LD block inverted triangle plotting with aspect ratio (width/height), "
            "using all SNPs in selected bimrange. Requires --bimrange."
        ),
    )
    optional_group.add_argument(
        "-manh-ylim", "--manh-ylim", type=float, default=None,
        help="Upper limit of Manhattan y-axis (default: auto).",
    )
    geno_group = optional_group.add_mutually_exclusive_group(required=False)
    geno_group.add_argument(
        "-bfile", "--bfile", type=str, default=None,
        help="Optional genotype PLINK prefix for --ldblock/--ldblock-all.",
    )
    geno_group.add_argument(
        "-vcf", "--vcf", type=str, default=None,
        help="Optional genotype VCF/VCF.GZ file for --ldblock/--ldblock-all.",
    )
    geno_group.add_argument(
        "-file", "--file", dest="geno", type=str, default=None,
        help="Optional genotype TXT file for --ldblock/--ldblock-all.",
    )
    optional_group.add_argument(
        "-qq", "--qq", type=str, nargs="?", const="5/4", default=None,
        help=(
            "Enable QQ plotting with aspect ratio (width/height). "
            "Examples: --qq (default 5/4), --qq 1.25, --qq 4/3."
        ),
    )
    optional_group.add_argument(
        "-pallete", "--pallete", type=str, default=None,
        help=(
            "Manhattan color palette (QQ keeps black/grey). "
            "Supports cmap names (e.g. tab10, tab20) or ';'-separated colors "
            "(e.g. #1f77b4;#ff7f0e or (215,123,254);(1,1,1)). "
            "If omitted, use default black/grey."
        ),
    )
    optional_group.add_argument(
        "-scatter-size", "--scatter-size", type=float, default=8.0,
        help="Scatter marker size for Manhattan and QQ plots (default: %(default)s).",
    )
    optional_group.add_argument(
        "-fullscatter", "--fullscatter", action="store_true", default=False,
        help="Disable GWASPLOT compression and draw all points.",
    )
    optional_group.add_argument(
        "-hl", "--highlight", type=str, default=None,
        help=(
            "BED-like file of SNPs to highlight, e.g.:\n"
            "  chr\\tpos\\tpos\\tgene\\tfunction"
        ),
    )
    optional_group.add_argument(
        "-format", "--format", type=str, default="png",
        help="Output figure format: pdf, png, svg, tif (default: %(default)s).",
    )
    optional_group.add_argument(
        "-a", "--anno", type=str, default=None,
        help="Annotation file (.gff or .bed) for SNP annotation (default: %(default)s).",
    )
    optional_group.add_argument(
        "-ab", "--annobroaden", type=float, default=None,
        help="Broaden the annotation window around SNPs (Kb) (default: %(default)s).",
    )
    optional_group.add_argument(
        "-descItem", "--descItem", type=str, default="description",
        help="Attribute key used as description in the GFF file (default: %(default)s).",
    )
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory for plots and annotation (default: current directory).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Prefix of the log file (default: JanusX).",
    )
    optional_group.add_argument(
        "-t", "--thread", type=int, default=-1,
        help="Number of CPU threads (-1 uses all available cores; default: %(default)s).",
    )

    args = parser.parse_args()

    args.out = args.out if args.out is not None else "."
    args.prefix = "JanusX" if args.prefix is None else args.prefix

    # Create output directory if needed
    if args.out != "":
        os.makedirs(args.out, mode=0o755, exist_ok=True)
    else:
        args.out = "."

    log_path = f"{args.out}/{args.prefix}.postGWAS.log".replace("//", "/")
    logger = setup_logging(log_path)

    # ------------------------------------------------------------------
    # Basic checks and configuration
    # ------------------------------------------------------------------
    args.format = str(args.format).lower()
    if args.format not in ["pdf", "png", "svg", "tif"]:
        logger.error(
            f"Unsupported figure format: {args.format} "
            "(choose from: pdf, png, svg, tif)"
        )
        raise SystemExit(1)
    if args.scatter_size <= 0:
        logger.error("scatter-size must be > 0.")
        raise SystemExit(1)
    if args.manh_ylim is not None and args.manh_ylim <= 0:
        logger.error("manh-ylim must be > 0.")
        raise SystemExit(1)

    try:
        args.pallete_spec = _parse_pallete_spec(args.pallete)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)

    try:
        args.manh_ratio = _parse_ratio(args.manh, "Manhattan") if args.manh is not None else None
        args.qq_ratio = _parse_ratio(args.qq, "QQ") if args.qq is not None else None
        if args.ldblock_all is not None:
            args.ldblock_ratio = _parse_ratio(args.ldblock_all, "LDBlock-all")
            args.ldblock_mode = "all"
        elif args.ldblock is not None:
            args.ldblock_ratio = _parse_ratio(args.ldblock, "LDBlock")
            args.ldblock_mode = "threshold"
        else:
            args.ldblock_ratio = None
            args.ldblock_mode = None
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    if args.bimrange is not None:
        try:
            args.bimrange_tuples = [
                _parse_bimrange(x, logger) for x in args.bimrange
            ]
            args.bimrange_tuples = sorted(
                args.bimrange_tuples,
                key=lambda x: (_chrom_sort_key(x[0]), int(x[1]), int(x[2])),
            )
            args.bimrange_tuples = _merge_overlapping_bimranges(
                args.bimrange_tuples, logger
            )
        except ValueError as e:
            logger.error(str(e))
            raise SystemExit(1)
    else:
        args.bimrange_tuples = None
    if args.ldblock_ratio is not None and args.bimrange_tuples is None:
        logger.warning(
            "Warning: --ldblock/--ldblock-all requires --bimrange; LD block and Manhattan+LD plotting are skipped."
        )
        args.ldblock_ratio = None
        args.ldblock_mode = None

    args.genofile = args.bfile or args.vcf or args.geno
    if args.ldblock_ratio is not None and args.genofile is None:
        logger.warning(
            "Warning: --ldblock/--ldblock-all enabled but no genotype file provided; zero-correlation LD block will be drawn."
        )

    no_plot_or_anno = (
        args.manh_ratio is None
        and args.qq_ratio is None
        and args.ldblock_ratio is None
        and args.anno is None
    )
    args.disable_compression = bool(args.fullscatter or (args.bimrange_tuples is not None))

    # ------------------------------------------------------------------
    # Configuration summary
    # ------------------------------------------------------------------
    logger.info("JanusX - Post-GWAS visualization and annotation")
    logger.info(f"Host: {socket.gethostname()}\n")

    logger.info("*" * 60)
    logger.info("POST-GWAS CONFIGURATION")
    logger.info("*" * 60)
    logger.info(f"Input files:   {_format_input_files(args.gwasfile)}")
    logger.info(f"Chr column:    {args.chr}")
    logger.info(f"Pos column:    {args.pos}")
    logger.info(f"P-value column:{args.pvalue}")
    logger.info(f"Genotype file: {args.genofile}")
    logger.info(
        f"Threshold:     {args.threshold if args.threshold is not None else '0.05 / nSNP'}"
    )
    logger.info(
        f"Bimrange:      "
        f"{_format_bimrange_summary(args.bimrange_tuples)}"
    )
    if (
        args.manh_ratio is not None
        or args.qq_ratio is not None
        or args.ldblock_ratio is not None
    ):
        logger.info("Visualization:")
        logger.info(
            f"  Manhattan pallete: "
            f"{'default (black/grey)' if args.pallete_spec is None else args.pallete}"
        )
        logger.info("  QQ pallete:        default (black/grey)")
        logger.info(f"  Scatter size:      {args.scatter_size}")
        if args.disable_compression:
            if args.fullscatter and args.bimrange_tuples is not None:
                comp_text = "off (--fullscatter, auto for --bimrange)"
            elif args.fullscatter:
                comp_text = "off (--fullscatter)"
            else:
                comp_text = "off (auto for --bimrange)"
        else:
            comp_text = "on"
        logger.info(f"  Compression:       {comp_text}")
        logger.info(f"  Highlight:   {args.highlight}")
        logger.info(f"  Format:      {args.format}")
        logger.info(f"  Manhattan:   {args.manh_ratio if args.manh_ratio is not None else 'off'}")
        logger.info(
            f"  Manhattan ylim max: {args.manh_ylim if args.manh_ylim is not None else 'auto'}"
        )
        logger.info(f"  QQ:          {args.qq_ratio if args.qq_ratio is not None else 'off'}")
        if args.ldblock_ratio is None:
            logger.info("  LDBlock:     off")
        else:
            ld_mode_text = "all SNPs" if args.ldblock_mode == "all" else "threshold SNPs"
            logger.info(f"  LDBlock:     {args.ldblock_ratio} ({ld_mode_text})")
    else:
        logger.info("Visualization: disabled")
    if args.anno:
        logger.info("Annotation:")
        logger.info(f"  Anno file:   {args.anno}")
        logger.info(f"  Window (kb): {args.annobroaden}")
    logger.info(f"Output prefix: {args.out}/{args.prefix}")
    logger.info(
        f"Threads:       {args.thread} "
        f"({'All cores' if args.thread == -1 else 'User-specified'})"
    )
    logger.info("*" * 60 + "\n")
    if no_plot_or_anno:
        logger.warning(
            "Warning: No --manh/--qq/--ldblock/--ldblock-all/--anno provided. Nothing will be plotted or annotated."
        )

    checks: list[bool] = [ensure_file_exists(logger, f, "GWAS result file") for f in args.gwasfile]
    if args.highlight:
        checks.append(ensure_file_exists(logger, args.highlight, "Highlight file"))
    if args.anno:
        checks.append(ensure_file_exists(logger, args.anno, "Annotation file"))
    if args.vcf:
        checks.append(ensure_file_exists(logger, args.vcf, "Genotype VCF file"))
    if args.geno:
        checks.append(ensure_file_exists(logger, args.geno, "Genotype TXT file"))
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, args.bfile, "Genotype PLINK prefix"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Parallel processing of all input files
    # ------------------------------------------------------------------
    Parallel(n_jobs=args.thread, backend="loky")(
        delayed(GWASplot)(file, args, logger) for file in args.gwasfile
    )

    # ------------------------------------------------------------------
    # Final logging
    # ------------------------------------------------------------------
    lt = time.localtime()
    endinfo = (
        f"\nFinished post-GWAS analysis. Total wall time: "
        f"{round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    logger.info(endinfo)


if __name__ == "__main__":
    main()
