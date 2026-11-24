# -*- coding: utf-8 -*-
"""
JanusX: Post-IBD Ancestry Visualization and Summary

Directly reads `.jhibd` sparse binary archives produced by `jx ibd -ref ...`
via Rust-native aggregation and generates publication-ready plots and tables:
  1. Recombination rate & hotspots (-recomb): empirical population genetic map,
     windowed recombination rate, and hotspot / Marey-curve visualization.
  2. Founder contribution (-segregation): genome-wide and per-chromosome
     founder contribution proportions.
  3. Chromosome Mosaic painting (-mosaic): whole-genome all-sample clustered
     multi-chromosome mosaic heatmap (columns proportional to physical length,
     legend on the right), representative sample karyotype mosaic, and optional
     per-chromosome heatmaps. When both -recomb and -mosaic are enabled, the
     two recombination tracks are aligned directly above the genome mosaic.

Examples
--------
  # Run all visualizations (combined recomb+mosaic + founder contribution)
  jx postibd -i ./cubic_All.maf0.02.imp.jhibd -o ./cubic_post -fmt png

  # Run recombination + mosaic with NAM map comparison (PDF)
  jx postibd -i ./cubic_All.maf0.02.imp.jhibd -o ./cubic_post \\
    -map ./NAM_joint_CN_US_B73v4_beagle.map -recomb -mosaic -fmt pdf

  # Render whole-genome clustered mosaic + specific lines (SVG)
  jx postibd -i ./cubic_All.maf0.02.imp.jhibd -o ./cubic_post \\
    -mosaic --samples L001,L002,L003 -fmt svg
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

for _key in ("MPLBACKEND",):
    if _key in os.environ:
        del os.environ[_key]

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
import numpy as np

from janusx import janusx as jxrs
from janusx.gtools.cleaner import chrom_sort_key
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.config_render import emit_cli_configuration
from ._common.log import setup_logging
from ._common.progress import CliStatus, format_elapsed, log_success

_SUPPORTED_FORMATS = ("png", "pdf", "svg", "tif")
_UNCALLED_COLOR = "#E5E7EB"
_GAP_COLOR = "#FFFFFF"

# 26 perceptually distinct founder colors (covers 24-founder CUBIC/NAM panels
# without repeats; cycles cleanly with slight lightness variation beyond 26).
_FOUNDER_PALETTE = (
    "#1F77B4",  # steel blue
    "#D62728",  # crimson
    "#2CA02C",  # forest green
    "#FF7F0E",  # vivid orange
    "#9467BD",  # purple
    "#17BECF",  # cyan
    "#E377C2",  # rose pink
    "#8C564B",  # chestnut
    "#BCBD22",  # olive gold
    "#393B79",  # deep indigo
    "#637939",  # moss green
    "#8C6D31",  # ochre
    "#843C39",  # brick red
    "#7B4173",  # plum
    "#3182BD",  # sky blue
    "#E6550D",  # burnt orange
    "#31A354",  # emerald
    "#756BB1",  # lavender indigo
    "#6BAED6",  # light azure
    "#FD8D3C",  # tangerine
    "#74C476",  # mint green
    "#9E9AC8",  # soft violet
    "#D6616B",  # coral red
    "#CE6DBD",  # orchid
    "#E7BA52",  # amber
    "#5254A3",  # cobalt
)


def _normalize_chrom(name: str) -> str:
    text = str(name).strip()
    if text.lower().startswith("chr"):
        return text[3:]
    return text


def _resolve_input_jhibd(raw: str) -> Path:
    p = Path(str(raw).strip()).expanduser()
    if p.is_file():
        return p
    if not str(p).endswith(".jhibd"):
        candidate = Path(f"{p}.jhibd")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Input .jhibd archive not found: {raw}")


def _resolve_output_prefix(out_arg: Optional[str], input_path: Path) -> Path:
    stem = input_path.name
    if stem.endswith(".jhibd"):
        stem = stem[: -len(".jhibd")]
    if not out_arg or str(out_arg).strip() == "":
        return Path.cwd() / f"{stem}.postibd"
    out_path = Path(str(out_arg).strip()).expanduser()
    if out_path.exists() and out_path.is_dir():
        return out_path / stem
    if str(out_arg).endswith(("/", os.sep)):
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path / stem
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _build_parent_colors(n_parents: int) -> list[str]:
    if n_parents <= len(_FOUNDER_PALETTE):
        return list(_FOUNDER_PALETTE[:n_parents])
    colors = list(_FOUNDER_PALETTE)
    cmap = plt.get_cmap("gist_ncar")
    extra = n_parents - len(colors)
    for i in range(extra):
        rgba = cmap((i + 0.5) / max(1, extra))
        colors.append(mcolors.to_hex(rgba))
    return colors


def _read_plink_map(map_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read a 4-column PLINK .map file (CHR ID CM BP) -> {norm_chrom: (bp_arr, cm_arr)}."""
    raw: dict[str, list[tuple[int, float]]] = {}
    with map_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                cm = float(parts[2])
                bp = int(float(parts[3]))
            except ValueError:
                continue
            if not math.isfinite(cm):
                continue
            key = _normalize_chrom(parts[0])
            raw.setdefault(key, []).append((bp, cm))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, pts in raw.items():
        pts.sort(key=lambda x: x[0])
        bps = np.asarray([p[0] for p in pts], dtype=np.float64)
        cms = np.asarray([p[1] for p in pts], dtype=np.float64)
        out[key] = (bps, cms)
    return out


def _interp_external_map_cm(
    ext_map: dict[str, tuple[np.ndarray, np.ndarray]],
    chrom_name: str,
    bp: float,
) -> Optional[float]:
    key = _normalize_chrom(chrom_name)
    entry = ext_map.get(key)
    if entry is None or entry[0].size == 0:
        return None
    bps, cms = entry
    return float(np.interp(bp, bps, cms))


_GROUP_HEADER_COL1 = {"id", "iid", "fid", "sample", "sample_id", "parent", "parent_id", "founder", "line"}
_GROUP_HEADER_COL2 = {"group", "ancestry", "pop", "population", "cluster", "label", "type", "subpop"}


def _read_parent_group_file(group_path: Path) -> dict[str, str]:
    """Read a 2+ column whitespace/tab-delimited parent->ancestry-group mapping file."""
    mapping: dict[str, str] = {}
    first_data = True
    with group_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            col1, col2 = parts[0].strip(), parts[1].strip()
            if first_data:
                first_data = False
                if col1.lower() in _GROUP_HEADER_COL1 and col2.lower() in _GROUP_HEADER_COL2:
                    continue
            if col1 and col2:
                mapping[col1] = col2
    if not mapping:
        raise ValueError(
            f"Ancestry group file contains no valid 2-column mappings: {group_path}"
        )
    return mapping


def _save_figure(fig: plt.Figure, out_path: Path, fmt: str, dpi: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {
        "format": "tiff" if fmt == "tif" else fmt,
        "bbox_inches": "tight",
        "facecolor": "white",
    }
    if fmt in {"png", "tif"}:
        save_kwargs["dpi"] = int(dpi)
    fig.savefig(str(out_path), **save_kwargs)
    plt.close(fig)
    return out_path


def _ordered_chrom_indices(chrom_names: Sequence[str]) -> list[int]:
    return sorted(range(len(chrom_names)), key=lambda i: chrom_sort_key(chrom_names[i]))


def _compute_exact_genome_spans(
    chrom_lengths_bp: Sequence[int],
    ordered_indices: Sequence[int],
    gap_frac: float = 0.012,
) -> tuple[dict[int, float], list[tuple[int, float, float, float]], list[float], float, float]:
    """Compute exact physical-Mb x-offsets, spans, and inter-chrom separators."""
    lengths_mb = [max(float(chrom_lengths_bp[c]) / 1_000_000.0, 1e-6) for c in ordered_indices]
    total_mb = max(sum(lengths_mb), 1e-6)
    gap_mb = total_mb * gap_frac if len(ordered_indices) > 1 else 0.0
    offsets: dict[int, float] = {}
    spans: list[tuple[int, float, float, float]] = []
    sep_positions: list[float] = []
    cursor = 0.0
    for idx_c, (c, l_mb) in enumerate(zip(ordered_indices, lengths_mb)):
        offsets[c] = cursor
        spans.append((c, cursor, cursor + l_mb, cursor + 0.5 * l_mb))
        cursor += l_mb
        if idx_c + 1 < len(ordered_indices) and gap_mb > 0.0:
            sep_positions.append(cursor + 0.5 * gap_mb)
            cursor += gap_mb
    total_x = max(cursor, 1e-6)
    return offsets, spans, sep_positions, total_x, total_mb


def _compute_genome_layout(
    chrom_names: Sequence[str],
    chrom_lengths_bp: Sequence[int],
    ordered_indices: Sequence[int],
    gap_frac: float = 0.015,
) -> tuple[dict[int, float], list[tuple[int, float, float, float]], float]:
    """Compute cumulative Mb x-offsets for multi-chromosome genome-wide plots."""
    offsets, spans, _seps, total_x, _total_mb = _compute_exact_genome_spans(
        chrom_lengths_bp, ordered_indices, gap_frac=gap_frac
    )
    return offsets, spans, total_x


def _style_clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#374151")
    ax.spines["bottom"].set_color("#374151")
    ax.tick_params(colors="#1F2937", labelsize=9)


def _ancestry_count_label(data: dict[str, Any]) -> str:
    n_founders = int(data["n_reference_samples"])
    n_categories = len(data["reference_sample_ids"])
    if bool(data.get("grouped_parents", False)):
        return f"{n_categories} ancestry categories ({n_founders} founders)"
    return f"K={n_founders} founders"


def _attach_right_founder_legend(
    ax: plt.Axes,
    parent_ids: Sequence[str],
    parent_colors: Sequence[str],
    *,
    title: str = "Founders",
    order: Optional[Sequence[int]] = None,
) -> None:
    indices = list(order) if order is not None else list(range(len(parent_ids)))
    handles = [Patch(facecolor=parent_colors[p], label=parent_ids[p]) for p in indices]
    handles.append(Patch(facecolor=_UNCALLED_COLOR, label="Uncalled"))
    ncol = 1 if len(handles) <= 28 else 2
    fontsize = 7.6 if len(handles) <= 26 else 6.8
    ax.legend(
        handles=handles,
        title=title,
        title_fontsize=8.5,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        handlelength=1.1,
        labelspacing=0.32,
        columnspacing=0.8,
    )


# ---------------------------------------------------------------------------
# 1. Recombination Rate & Hotspots
# ---------------------------------------------------------------------------


def _prepare_recomb_data(
    data: dict[str, Any],
    ext_map: Optional[dict[str, tuple[np.ndarray, np.ndarray]]],
    generations: Optional[float],
    logger: logging.Logger,
) -> Optional[dict[str, Any]]:
    chrom_names: list[str] = data["chrom_names"]
    raw_rows: list[dict[str, Any]] = data["recomb_windows"]
    if not raw_rows:
        logger.warning("No recombination windows found in archive; skipping -recomb outputs.")
        return None

    ordered_c = _ordered_chrom_indices(chrom_names)
    by_chrom: dict[int, list[dict[str, Any]]] = {c: [] for c in ordered_c}
    for row in raw_rows:
        c = int(row["chrom_idx"])
        if c in by_chrom:
            by_chrom[c].append(row)
    for c in ordered_c:
        by_chrom[c].sort(key=lambda r: int(r["start_bp"]))

    # Attach reference map rate (from explicit -map or embedded .jhibd map).
    has_ref_map = False
    enriched_by_chrom: dict[int, list[dict[str, Any]]] = {}
    for c in ordered_c:
        cname = chrom_names[c]
        rows = by_chrom[c]
        out_rows = []
        for r in rows:
            s_bp = float(r["start_bp"])
            e_bp = float(r["end_bp"])
            win_mb = max((e_bp - s_bp + 1.0) / 1_000_000.0, 1e-9)
            if ext_map is not None:
                in_s = _interp_external_map_cm(ext_map, cname, s_bp)
                in_e = _interp_external_map_cm(ext_map, cname, e_bp)
                if in_s is not None and in_e is not None:
                    ref_cm = max(0.0, in_e - in_s)
                    ref_cum = in_e
                    has_ref_map = True
                else:
                    ref_cm = float("nan")
                    ref_cum = float("nan")
            else:
                in_s = float(r["input_start_cm"])
                in_e = float(r["input_end_cm"])
                ref_cm = max(0.0, in_e - in_s)
                ref_cum = in_e
                if in_e > 0.0:
                    has_ref_map = True
            ref_rate = ref_cm / win_mb if math.isfinite(ref_cm) else float("nan")
            callable_win = bool(r.get("callable", True))
            raw_eff_rate = r.get("effective_cm_per_mb", None)
            eff_rate = (
                float(raw_eff_rate)
                if (callable_win and raw_eff_rate is not None and math.isfinite(float(raw_eff_rate)))
                else float("nan")
            )
            per_meiosis_rate = (
                eff_rate / float(generations)
                if (generations is not None and generations > 0.0 and math.isfinite(eff_rate))
                else float("nan")
            )
            item = dict(r)
            item["callable"] = callable_win
            item["eff_rate_val"] = eff_rate
            item["ref_cm_per_mb"] = ref_rate
            item["ref_cum_cm"] = ref_cum
            item["per_meiosis_cm_per_mb"] = per_meiosis_rate
            out_rows.append(item)
        enriched_by_chrom[c] = out_rows

    scale_div = float(generations) if (generations is not None and generations > 0.0) else 1.0
    callable_rates = [
        float(r["per_meiosis_cm_per_mb"])
        if (generations is not None and generations > 0.0)
        else float(r["eff_rate_val"])
        for c in ordered_c
        for r in enriched_by_chrom[c]
        if r["callable"]
    ]
    all_rates = np.asarray([v for v in callable_rates if math.isfinite(v)], dtype=np.float64)
    hotspot_thr = float(np.percentile(all_rates, 95.0)) if all_rates.size > 0 else 0.0

    return {
        "ordered_c": ordered_c,
        "enriched_by_chrom": enriched_by_chrom,
        "has_ref_map": has_ref_map,
        "scale_div": scale_div,
        "hotspot_thr": hotspot_thr,
    }


def _write_recomb_tables(
    chrom_names: Sequence[str],
    recomb_ctx: dict[str, Any],
    out_prefix: Path,
    generations: Optional[float],
) -> tuple[Path, Path]:
    ordered_c: list[int] = recomb_ctx["ordered_c"]
    enriched_by_chrom: dict[int, list[dict[str, Any]]] = recomb_ctx["enriched_by_chrom"]
    scale_div: float = recomb_ctx["scale_div"]

    # Write <prefix>.recomb.window.tsv
    tsv_path = Path(f"{out_prefix}.recomb.window.tsv")
    with tsv_path.open("w", encoding="utf-8") as fh:
        header = [
            "chrom",
            "start_bp",
            "end_bp",
            "mid_bp",
            "breakpoints",
            "obs_haps",
            "window_coverage",
            "callable",
            "effective_cm",
            "effective_cm_per_mb",
            "cum_effective_cm",
            "ref_cm_per_mb",
            "ref_cum_cm",
        ]
        if generations is not None and generations > 0.0:
            header.append("per_meiosis_cm_per_mb")
        fh.write("\t".join(header) + "\n")
        for c in ordered_c:
            cname = chrom_names[c]
            for r in enriched_by_chrom[c]:
                eff_r_str = f"{r['eff_rate_val']:.6f}" if math.isfinite(r["eff_rate_val"]) else "NA"
                ref_r_str = f"{r['ref_cm_per_mb']:.6f}" if math.isfinite(r["ref_cm_per_mb"]) else "NA"
                ref_c_str = f"{r['ref_cum_cm']:.6f}" if math.isfinite(r["ref_cum_cm"]) else "NA"
                cols = [
                    cname,
                    str(int(r["start_bp"])),
                    str(int(r["end_bp"])),
                    str(int(r["mid_bp"])),
                    str(int(r["breakpoints"])),
                    f"{float(r['obs_haps']):.2f}",
                    f"{float(r.get('window_coverage', 1.0)):.4f}",
                    "1" if r["callable"] else "0",
                    f"{float(r['effective_cm']):.6f}",
                    eff_r_str,
                    f"{float(r['cum_effective_cm']):.6f}",
                    ref_r_str,
                    ref_c_str,
                ]
                if generations is not None and generations > 0.0:
                    pm_str = (
                        f"{float(r['per_meiosis_cm_per_mb']):.6f}"
                        if math.isfinite(r["per_meiosis_cm_per_mb"])
                        else "NA"
                    )
                    cols.append(pm_str)
                fh.write("\t".join(cols) + "\n")

    # Write standard 4-column PLINK empirical genetic map: <prefix>.empirical.map
    map_out_path = Path(f"{out_prefix}.empirical.map")
    with map_out_path.open("w", encoding="utf-8") as fh:
        for c in ordered_c:
            cname = chrom_names[c]
            rows = enriched_by_chrom[c]
            if not rows:
                continue
            fh.write(f"{cname}\t{cname}_start\t0.000000\t{int(rows[0]['start_bp'])}\n")
            for idx, r in enumerate(rows, start=1):
                cm_val = float(r["cum_effective_cm"]) / scale_div
                fh.write(f"{cname}\t{cname}_win{idx}\t{cm_val:.6f}\t{int(r['end_bp'])}\n")

    return tsv_path, map_out_path


def _plot_recomb_panels(
    ax1: plt.Axes,
    ax2: plt.Axes,
    chrom_names: Sequence[str],
    plot_chroms: Sequence[int],
    offsets: dict[int, float],
    spans: Sequence[tuple[int, float, float, float]],
    sep_positions: Sequence[float],
    total_x: float,
    recomb_ctx: dict[str, Any],
    ext_map: Optional[dict[str, tuple[np.ndarray, np.ndarray]]],
    generations: Optional[float],
    *,
    show_bottom_xticks: bool = True,
    title: Optional[str] = None,
) -> None:
    enriched_by_chrom: dict[int, list[dict[str, Any]]] = recomb_ctx["enriched_by_chrom"]
    has_ref_map: bool = recomb_ctx["has_ref_map"]
    scale_div: float = recomb_ctx["scale_div"]
    hotspot_thr: float = recomb_ctx["hotspot_thr"]

    for idx_s, (_c, x0, x1, _xc) in enumerate(spans):
        if idx_s % 2 == 1:
            ax1.axvspan(x0, x1, color="#F3F4F6", alpha=0.85, zorder=0)
            ax2.axvspan(x0, x1, color="#F3F4F6", alpha=0.85, zorder=0)
    for sx in sep_positions:
        ax1.axvline(sx, color="#D1D5DB", linewidth=0.5, alpha=0.7, zorder=1)
        ax2.axvline(sx, color="#D1D5DB", linewidth=0.5, alpha=0.7, zorder=1)

    ax1_ref = ax1.twinx() if has_ref_map else None
    ax2_ref = ax2.twinx() if (has_ref_map and generations is None) else None

    labeled_hot = False
    labeled_cum = False
    labeled_ref_r = False
    labeled_ref_c = False

    for idx_s, c in enumerate(plot_chroms):
        rows = enriched_by_chrom.get(c, [])
        if not rows:
            continue
        off = offsets[c]
        xs_mid = np.asarray([off + float(r["mid_bp"]) / 1_000_000.0 for r in rows], dtype=np.float64)
        ys = np.asarray(
            [
                float(r["per_meiosis_cm_per_mb"])
                if (generations is not None and generations > 0.0)
                else float(r["eff_rate_val"])
                for r in rows
            ],
            dtype=np.float64,
        )
        xs_cum = np.asarray(
            [off + float(rows[0]["start_bp"]) / 1_000_000.0]
            + [off + float(r["end_bp"]) / 1_000_000.0 for r in rows],
            dtype=np.float64,
        )
        cum_y = np.asarray(
            [0.0] + [float(r["cum_effective_cm"]) / scale_div for r in rows],
            dtype=np.float64,
        )
        color = "#1D4ED8" if idx_s % 2 == 0 else "#0F766E"
        valid_rate = np.isfinite(ys)
        if np.any(valid_rate):
            ax1.fill_between(
                xs_mid, 0.0, ys, where=valid_rate, color=color, alpha=0.22, linewidth=0.0, zorder=2
            )
            ax1.plot(xs_mid, ys, color=color, linewidth=1.15, zorder=3)

        if hotspot_thr > 0.0:
            hot = valid_rate & (ys >= hotspot_thr)
            if np.any(hot):
                ax1.scatter(
                    xs_mid[hot],
                    ys[hot],
                    s=14,
                    color="#DC2626",
                    zorder=4,
                    label="Hotspot (>=95th pct)" if not labeled_hot else None,
                )
                labeled_hot = True

        ax2.plot(
            xs_cum,
            cum_y,
            color=color,
            linewidth=1.4,
            label="Empirical cumulative cM" if not labeled_cum else None,
            zorder=3,
        )
        labeled_cum = True

        if has_ref_map:
            ref_r = np.asarray([float(r["ref_cm_per_mb"]) for r in rows], dtype=np.float64)
            ref_c_start = (
                float(rows[0]["input_start_cm"])
                if ext_map is None
                else (_interp_external_map_cm(ext_map, chrom_names[c], float(rows[0]["start_bp"])) or 0.0)
            )
            ref_c = np.asarray(
                [ref_c_start] + [float(r["ref_cum_cm"]) for r in rows],
                dtype=np.float64,
            )
            if ax1_ref is not None and np.any(np.isfinite(ref_r)):
                ax1_ref.plot(
                    xs_mid,
                    ref_r,
                    color="#6B7280",
                    linestyle="--",
                    linewidth=0.85,
                    alpha=0.7,
                    label="Reference map (cM/Mb)" if not labeled_ref_r else None,
                    zorder=2,
                )
                labeled_ref_r = True
            target_ax2 = ax2_ref if ax2_ref is not None else ax2
            if np.any(np.isfinite(ref_c)):
                target_ax2.plot(
                    xs_cum,
                    ref_c,
                    color="#6B7280",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                    label="Reference map (cM)" if not labeled_ref_c else None,
                    zorder=2,
                )
                labeled_ref_c = True

    if hotspot_thr > 0.0:
        ax1.axhline(hotspot_thr, color="#DC2626", linestyle=":", linewidth=0.9, alpha=0.8)

    rate_ylabel = (
        f"Empirical rate (cM/Mb, G={generations:g})"
        if (generations is not None and generations > 0.0)
        else "Empirical eff. rate (cM/Mb)"
    )
    cum_ylabel = (
        f"Empirical map (cM, G={generations:g})"
        if (generations is not None and generations > 0.0)
        else "Empirical cum. (eff. cM)"
    )
    ax1.set_ylabel(rate_ylabel, fontsize=9.5, color="#1F2937")
    ax2.set_ylabel(cum_ylabel, fontsize=9.5, color="#1F2937")
    ax1.set_ylim(bottom=0.0)
    ax2.set_ylim(bottom=0.0)
    _style_clean_axes(ax1)
    _style_clean_axes(ax2)
    ax1.tick_params(labelbottom=False)
    if ax1_ref is not None:
        ax1_ref.set_ylabel("Reference (cM/Mb)", fontsize=8.5, color="#4B5563")
        ax1_ref.spines["top"].set_visible(False)
        ax1_ref.spines["right"].set_color("#6B7280")
        ax1_ref.tick_params(colors="#4B5563", labelsize=8.0)
        ax1_ref.set_ylim(bottom=0.0)
    if ax2_ref is not None:
        ax2_ref.set_ylabel("Reference (cM)", fontsize=8.5, color="#4B5563")
        ax2_ref.spines["top"].set_visible(False)
        ax2_ref.spines["right"].set_color("#6B7280")
        ax2_ref.tick_params(colors="#4B5563", labelsize=8.0)
        ax2_ref.set_ylim(bottom=0.0)

    ax2.set_xlim(0.0, total_x)
    if show_bottom_xticks:
        ax2.set_xticks([xc for (_c, _x0, _x1, xc) in spans])
        ax2.set_xticklabels([chrom_names[c] for (c, _x0, _x1, _xc) in spans])
        ax2.set_xlabel("Chromosome (physical position)", fontsize=10, color="#1F2937")
    else:
        ax2.tick_params(labelbottom=False)

    if title:
        ax1.set_title(title, fontsize=11.5, fontweight="bold", pad=10, color="#111827")

    handles, labels = ax1.get_legend_handles_labels()
    if ax1_ref is not None:
        h2, l2 = ax1_ref.get_legend_handles_labels()
        handles.extend(h2)
        labels.extend(l2)
    if handles:
        ax1.legend(
            handles,
            labels,
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.85,
            fontsize=8.2,
        )


def _run_recomb_outputs(
    data: dict[str, Any],
    out_prefix: Path,
    fmt: str,
    dpi: int,
    ext_map: Optional[dict[str, tuple[np.ndarray, np.ndarray]]],
    generations: Optional[float],
    logger: logging.Logger,
    *,
    write_plot: bool = True,
) -> tuple[list[Path], Optional[dict[str, Any]]]:
    recomb_ctx = _prepare_recomb_data(data, ext_map, generations, logger)
    if recomb_ctx is None:
        return [], None

    chrom_names: list[str] = data["chrom_names"]
    chrom_lengths_bp: list[int] = data["chrom_lengths_bp"]
    ordered_c: list[int] = recomb_ctx["ordered_c"]

    tsv_path, map_out_path = _write_recomb_tables(
        chrom_names, recomb_ctx, out_prefix, generations
    )
    written: list[Path] = [tsv_path, map_out_path]

    if write_plot:
        offsets, spans, sep_positions, total_x, _total_mb = _compute_exact_genome_spans(
            chrom_lengths_bp, ordered_c
        )
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(max(11.0, min(18.0, 2.2 + 0.9 * len(ordered_c))), 6.8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.14},
        )
        title = (
            f"Population Recombination Rate & Hotspots "
            f"(N={data['n_target_samples']} lines, {_ancestry_count_label(data)})"
        )
        _plot_recomb_panels(
            ax1,
            ax2,
            chrom_names,
            ordered_c,
            offsets,
            spans,
            sep_positions,
            total_x,
            recomb_ctx,
            ext_map,
            generations,
            show_bottom_xticks=True,
            title=title,
        )
        fig_path = _save_figure(fig, Path(f"{out_prefix}.recomb_hotspots.{fmt}"), fmt, dpi)
        written.append(fig_path)
        logger.info("Recombination outputs written: %s, %s, %s", tsv_path, map_out_path, fig_path)
    else:
        logger.info("Recombination tables written: %s, %s", tsv_path, map_out_path)

    return written, recomb_ctx


# ---------------------------------------------------------------------------
# 2. Parent Contribution & Segregation Distortion
# ---------------------------------------------------------------------------


def _run_segregation_outputs(
    data: dict[str, Any],
    out_prefix: Path,
    fmt: str,
    dpi: int,
    parent_colors: Sequence[str],
    logger: logging.Logger,
) -> list[Path]:
    chrom_names: list[str] = data["chrom_names"]
    parent_ids: list[str] = data["reference_sample_ids"]
    n_parents = len(parent_ids)
    contrib_chroms: list[dict[str, Any]] = data["parent_contrib_chroms"]

    if not contrib_chroms:
        logger.warning("No founder contribution data found in archive; skipping -segregation outputs.")
        return []

    ordered_c = _ordered_chrom_indices(chrom_names)
    contrib_map = {int(c["chrom_idx"]): c for c in contrib_chroms}

    # Aggregate genome-wide parent contribution.
    gw_bp = np.zeros(n_parents, dtype=np.float64)
    gw_cm = np.zeros(n_parents, dtype=np.float64)
    gw_called_bp = 0.0
    chrom_props = np.zeros((len(ordered_c), n_parents), dtype=np.float64)
    for row_i, c in enumerate(ordered_c):
        entry = contrib_map.get(c)
        if entry is None:
            continue
        p_bp = np.asarray(entry["parent_bp"], dtype=np.float64)
        p_cm = np.asarray(entry["parent_cm"], dtype=np.float64)
        c_bp = float(entry["called_bp"])
        gw_bp += p_bp
        gw_cm += p_cm
        gw_called_bp += c_bp
        if c_bp > 0.0:
            chrom_props[row_i, :] = p_bp / c_bp

    gw_props = gw_bp / gw_called_bp if gw_called_bp > 0.0 else np.zeros(n_parents)
    gw_cm_total = float(np.sum(gw_cm))
    gw_cm_props = gw_cm / gw_cm_total if gw_cm_total > 0.0 else np.zeros(n_parents)
    parent_order = np.argsort(-gw_props, kind="stable").tolist()

    # Write <prefix>.parent_contribution.tsv
    contrib_tsv = Path(f"{out_prefix}.parent_contribution.tsv")
    with contrib_tsv.open("w", encoding="utf-8") as fh:
        chrom_cols = [f"chr_{chrom_names[c]}_prop" for c in ordered_c]
        fh.write(
            "\t".join(
                ["parent", "genome_bp_prop", "genome_cm_prop", "expected_uniform_prop", "fold_enrichment"]
                + chrom_cols
            )
            + "\n"
        )
        exp_prop = 1.0 / max(1, n_parents)
        for p in parent_order:
            fold = gw_props[p] / exp_prop if exp_prop > 0.0 else 0.0
            c_vals = [f"{chrom_props[i, p]:.6f}" for i in range(len(ordered_c))]
            fh.write(
                "\t".join(
                    [
                        parent_ids[p],
                        f"{gw_props[p]:.6f}",
                        f"{gw_cm_props[p]:.6f}",
                        f"{exp_prop:.6f}",
                        f"{fold:.4f}",
                    ]
                    + c_vals
                )
                + "\n"
            )

    # Plot <prefix>.parent_contribution.<fmt>
    fig_h = max(5.5, min(14.0, 2.2 + 0.28 * n_parents))
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(13.5, fig_h),
        gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.28},
    )

    y_pos = np.arange(n_parents)
    bar_vals = [gw_props[p] * 100.0 for p in parent_order]
    bar_colors = [parent_colors[p] for p in parent_order]
    bar_labels = [parent_ids[p] for p in parent_order]
    ax_l.barh(y_pos, bar_vals, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.78)
    ax_l.axvline(
        100.0 / max(1, n_parents),
        color="#111827",
        linestyle="--",
        linewidth=1.0,
        label=f"Uniform ({100.0 / max(1, n_parents):.2f}%)",
    )
    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(bar_labels, fontsize=8.5)
    ax_l.invert_yaxis()
    ax_l.set_xlabel("Genome-wide contribution (%)", fontsize=10)
    grouped = bool(data.get("grouped_parents", False))
    ax_l.set_title(
        "Genome-Wide Ancestry Contribution" if grouped else "Genome-Wide Founder Contribution",
        fontsize=11,
        fontweight="bold",
    )
    max_bar = max(bar_vals) if bar_vals else 10.0
    ax_l.set_xlim(0.0, max_bar * 1.18)
    for y_i, val in zip(y_pos, bar_vals):
        ax_l.text(
            val + max_bar * 0.015, y_i, f"{val:.2f}%", va="center", ha="left", fontsize=7.5, color="#374151"
        )
    _style_clean_axes(ax_l)
    ax_l.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.5)

    x_pos = np.arange(len(ordered_c))
    bottom = np.zeros(len(ordered_c), dtype=np.float64)
    for p in parent_order:
        vals = chrom_props[:, p] * 100.0
        ax_r.bar(
            x_pos,
            vals,
            bottom=bottom,
            color=parent_colors[p],
            width=0.72,
            edgecolor="white",
            linewidth=0.25,
            label=parent_ids[p],
        )
        bottom += vals
    ax_r.set_xticks(x_pos)
    ax_r.set_xticklabels([chrom_names[c] for c in ordered_c], fontsize=9)
    ax_r.set_ylim(0.0, 100.0)
    ax_r.set_ylabel("Called ancestry proportion (%)", fontsize=10)
    ax_r.set_xlabel("Chromosome", fontsize=10)
    ax_r.set_title(
        "Per-Chromosome Ancestry Composition" if grouped else "Per-Chromosome Founder Composition",
        fontsize=11,
        fontweight="bold",
    )
    _style_clean_axes(ax_r)

    contrib_fig = _save_figure(fig, Path(f"{out_prefix}.parent_contribution.{fmt}"), fmt, dpi)
    logger.info(
        "Founder contribution outputs written: %s, %s",
        contrib_tsv,
        contrib_fig,
    )
    return [contrib_tsv, contrib_fig]


# ---------------------------------------------------------------------------
# 3. Chromosome Mosaic Painting
# ---------------------------------------------------------------------------


def _cluster_pop_mosaic_rows(mat: np.ndarray) -> tuple[np.ndarray, str]:
    """Cluster population mosaic rows and return `(row_order, method_label)`."""
    n_samples, n_bins = mat.shape
    if n_samples <= 2:
        return np.arange(n_samples), "sample order"

    # Hierarchical clustering on full-column Hamming distance (O(N^2) in C via SciPy).
    if n_samples <= 3000:
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import pdist

            dists = pdist(mat, metric="hamming")
            if dists.size > 0 and np.any(dists > 0.0):
                z = linkage(dists, method="average")
                return np.asarray(leaves_list(z), dtype=int), "hierarchically clustered"
            return np.arange(n_samples), "hierarchically clustered"
        except Exception:
            pass

    # Fast O(N log N) deterministic fallback for very large cohorts (>3000 lines).
    anchors = np.linspace(0, n_bins - 1, min(16, n_bins), dtype=int)
    keys = []
    for i in range(n_samples):
        row = mat[i]
        valid = row[row >= 0]
        if valid.size > 0:
            vals, counts = np.unique(valid, return_counts=True)
            dom = int(vals[int(np.argmax(counts))])
        else:
            dom = -1
        key = (dom,) + tuple(int(row[a]) for a in anchors)
        keys.append((key, i))
    keys.sort(key=lambda x: x[0])
    return np.asarray([k[1] for k in keys], dtype=int), "sorted by ancestry anchors"


def _prepare_pop_mosaic_genome(
    chrom_lengths_bp: Sequence[int],
    pop_mosaics: Sequence[dict[str, Any]],
    ordered_c: Sequence[int],
) -> Optional[dict[str, Any]]:
    pop_by_c = {int(pm["chrom_idx"]): pm for pm in pop_mosaics}
    active_c = [c for c in ordered_c if c in pop_by_c]
    if not active_c:
        return None
    n_s_genome = int(pop_by_c[active_c[0]]["n_samples"])
    valid_c = [c for c in active_c if int(pop_by_c[c]["n_samples"]) == n_s_genome]
    cluster_blocks: list[np.ndarray] = []
    blocks_by_c: dict[int, np.ndarray] = {}
    for c in valid_c:
        pm = pop_by_c[c]
        n_b = int(pm["n_bins"])
        block = np.asarray(pm["parent_matrix"], dtype=np.int16).reshape((n_s_genome, n_b))
        cluster_blocks.append(block)
        blocks_by_c[c] = block
    if not cluster_blocks:
        return None

    cluster_mat = np.concatenate(cluster_blocks, axis=1)
    row_order, cluster_label = _cluster_pop_mosaic_rows(cluster_mat)
    offsets, spans_mb, sep_positions, total_x, total_mb = _compute_exact_genome_spans(
        chrom_lengths_bp, valid_c
    )
    return {
        "pop_by_c": pop_by_c,
        "active_c": active_c,
        "valid_c": valid_c,
        "n_s_genome": n_s_genome,
        "blocks_by_c": blocks_by_c,
        "row_order": row_order,
        "cluster_label": cluster_label,
        "offsets": offsets,
        "spans_mb": spans_mb,
        "sep_positions": sep_positions,
        "total_x": total_x,
        "total_mb": total_mb,
    }


def _plot_pop_mosaic_panel(
    ax: plt.Axes,
    chrom_names: Sequence[str],
    genome_ctx: dict[str, Any],
    parent_ids: Sequence[str],
    parent_colors: Sequence[str],
    *,
    title: Optional[str] = None,
    legend_title: str = "Founders",
) -> None:
    n_parents = len(parent_ids)
    n_s_genome: int = genome_ctx["n_s_genome"]
    spans_mb: list[tuple[int, float, float, float]] = genome_ctx["spans_mb"]
    sep_positions: list[float] = genome_ctx["sep_positions"]
    total_x: float = genome_ctx["total_x"]
    blocks_by_c: dict[int, np.ndarray] = genome_ctx["blocks_by_c"]
    row_order: np.ndarray = genome_ctx["row_order"]
    cluster_label: str = genome_ctx["cluster_label"]

    cmap_genome = mcolors.ListedColormap([_UNCALLED_COLOR] + list(parent_colors))
    norm_genome = mcolors.BoundaryNorm(np.arange(-1.5, n_parents + 0.5, 1.0), cmap_genome.N)

    for c, x0, x1, _xc in spans_mb:
        block_sorted = blocks_by_c[c][row_order, :]
        ax.imshow(
            block_sorted,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap_genome,
            norm=norm_genome,
            extent=(x0, x1, float(n_s_genome), 0.0),
        )
    for sx in sep_positions:
        ax.axvline(sx, color="#D1D5DB", linewidth=0.5, alpha=0.8)

    ax.set_xlim(0.0, total_x)
    ax.set_ylim(float(n_s_genome), 0.0)
    ax.set_xticks([xc for (_c, _x0, _x1, xc) in spans_mb])
    ax.set_xticklabels([chrom_names[c] for (c, _x0, _x1, _xc) in spans_mb], fontsize=9.5)
    ax.set_xlabel("Chromosome (proportional to physical length)", fontsize=10.5)
    ax.set_ylabel(f"Target lines ({cluster_label}, n={n_s_genome})", fontsize=10.5)
    if title:
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
    _style_clean_axes(ax)
    _attach_right_founder_legend(ax, parent_ids, parent_colors, title=legend_title)


def _run_mosaic_outputs(
    data: dict[str, Any],
    out_prefix: Path,
    fmt: str,
    dpi: int,
    parent_colors: Sequence[str],
    per_chrom_mosaic: bool,
    logger: logging.Logger,
    *,
    recomb_ctx: Optional[dict[str, Any]] = None,
    ext_map: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
    generations: Optional[float] = None,
) -> list[Path]:
    chrom_names: list[str] = data["chrom_names"]
    chrom_lengths_bp: list[int] = data["chrom_lengths_bp"]
    parent_ids: list[str] = data["reference_sample_ids"]
    n_parents = len(parent_ids)
    ancestry_label = _ancestry_count_label(data)
    legend_title = "Ancestry" if bool(data.get("grouped_parents", False)) else "Founders"
    mosaic_samples: list[dict[str, Any]] = data["mosaic_samples"]
    mosaic_segments: list[dict[str, Any]] = data["mosaic_segments"]
    pop_mosaics: list[dict[str, Any]] = data["pop_mosaics"]
    ordered_c = _ordered_chrom_indices(chrom_names)

    written: list[Path] = []

    # 3A: Whole-genome multi-chromosome clustered population mosaic (either
    # combined with the two recombination tracks aligned above it when both
    # -recomb and -mosaic are requested, or standalone mosaic_genome).
    genome_ctx = _prepare_pop_mosaic_genome(chrom_lengths_bp, pop_mosaics, ordered_c)
    active_c = genome_ctx["active_c"] if genome_ctx is not None else []
    pop_by_c = genome_ctx["pop_by_c"] if genome_ctx is not None else {}

    if genome_ctx is not None:
        valid_c = genome_ctx["valid_c"]
        n_s_genome = genome_ctx["n_s_genome"]
        total_mb = genome_ctx["total_mb"]
        fig_w = max(13.0, min(20.0, 4.5 + 1.0 * len(valid_c)))

        if recomb_ctx is not None:
            mosaic_h = max(5.6, min(9.5, 4.2 + 0.0028 * n_s_genome))
            fig_h = mosaic_h + 4.4
            fig, (ax_rate, ax_cum, ax_mosaic) = plt.subplots(
                3,
                1,
                figsize=(fig_w, fig_h),
                sharex=True,
                gridspec_kw={"height_ratios": [1.15, 0.90, 2.75], "hspace": 0.10},
            )
            combined_title = (
                f"Population Recombination & Ancestry Mosaic "
                f"(n={n_s_genome} lines, {len(valid_c)} chroms, {total_mb:.1f} Mb, {ancestry_label})"
            )
            _plot_recomb_panels(
                ax_rate,
                ax_cum,
                chrom_names,
                valid_c,
                genome_ctx["offsets"],
                genome_ctx["spans_mb"],
                genome_ctx["sep_positions"],
                genome_ctx["total_x"],
                recomb_ctx,
                ext_map,
                generations,
                show_bottom_xticks=False,
                title=combined_title,
            )
            _plot_pop_mosaic_panel(
                ax_mosaic,
                chrom_names,
                genome_ctx,
                parent_ids,
                parent_colors,
                title=None,
                legend_title=legend_title,
            )
            combined_path = _save_figure(
                fig, Path(f"{out_prefix}.recomb_mosaic.{fmt}"), fmt, dpi
            )
            written.append(combined_path)
        else:
            fig_h = max(6.2, min(11.0, 4.5 + 0.003 * n_s_genome))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            _plot_pop_mosaic_panel(
                ax,
                chrom_names,
                genome_ctx,
                parent_ids,
                parent_colors,
                title=(
                    f"Whole-Genome Population Ancestry Mosaic "
                    f"(n={n_s_genome} lines, {len(valid_c)} chroms, {total_mb:.1f} Mb, {ancestry_label})"
                ),
                legend_title=legend_title,
            )
            genome_path = _save_figure(fig, Path(f"{out_prefix}.mosaic_genome.{fmt}"), fmt, dpi)
            written.append(genome_path)

    # 3B: Representative / selected whole-genome sample karyotype mosaic
    # Uses `active_c` (when available) so `--chrom` filters both population & sample mosaics.
    sample_chroms = active_c if active_c else ordered_c
    if mosaic_samples and sample_chroms:
        offsets, spans, total_x = _compute_genome_layout(
            chrom_names, chrom_lengths_bp, sample_chroms, gap_frac=0.02
        )
        n_show = len(mosaic_samples)
        fig_h = max(3.8, min(14.0, 1.8 + 0.52 * n_show))
        fig, ax = plt.subplots(figsize=(max(12.5, min(18.0, 3.0 + 0.9 * len(sample_chroms))), fig_h))

        bg_patches: list[Rectangle] = []
        for s_local, s_info in enumerate(mosaic_samples):
            n_haps = int(s_info["n_haps"])
            y_center = float(s_local)
            for _c, x0, x1, _xc in spans:
                if n_haps <= 1:
                    bg_patches.append(Rectangle((x0, y_center - 0.34), x1 - x0, 0.68))
                else:
                    bg_patches.append(Rectangle((x0, y_center - 0.36), x1 - x0, 0.33))
                    bg_patches.append(Rectangle((x0, y_center + 0.03), x1 - x0, 0.33))
        ax.add_collection(
            PatchCollection(bg_patches, facecolor=_UNCALLED_COLOR, edgecolor="none", zorder=1)
        )

        rects_by_parent: dict[int, list[Rectangle]] = {}
        for seg in mosaic_segments:
            c = int(seg["chrom_idx"])
            if c not in offsets:
                continue
            s_local = int(seg["local_sample_idx"])
            if s_local >= n_show:
                continue
            p = int(seg["parent_sample"])
            if p >= n_parents:
                continue
            n_haps = int(mosaic_samples[s_local]["n_haps"])
            hap = int(seg["hap"])
            x0 = offsets[c] + float(seg["start_bp"]) / 1_000_000.0
            w = max(float(seg["end_bp"] - seg["start_bp"] + 1) / 1_000_000.0, 0.05)
            if n_haps <= 1:
                y0, h = float(s_local) - 0.34, 0.68
            elif hap == 0:
                y0, h = float(s_local) - 0.36, 0.33
            else:
                y0, h = float(s_local) + 0.03, 0.33
            rects_by_parent.setdefault(p, []).append(Rectangle((x0, y0), w, h))

        for p, rects in rects_by_parent.items():
            ax.add_collection(
                PatchCollection(rects, facecolor=parent_colors[p], edgecolor="none", zorder=2)
            )

        ax.set_xlim(0.0, total_x)
        ax.set_ylim(-0.6, n_show - 0.4)
        ax.invert_yaxis()
        ax.set_yticks(np.arange(n_show))
        ax.set_yticklabels(
            [f"{s['sample_id']} ({int(s['total_breakpoints'])} bp)" for s in mosaic_samples],
            fontsize=8.5,
        )
        ax.set_xticks([xc for (_c, _x0, _x1, xc) in spans])
        ax.set_xticklabels([chrom_names[c] for (c, _x0, _x1, _xc) in spans], fontsize=9.5)
        ax.set_xlabel("Chromosome (physical position)", fontsize=10)
        ax.set_title(
            f"Representative Sample Ancestry Mosaic ({n_show} lines, {ancestry_label})",
            fontsize=11.5,
            fontweight="bold",
            pad=10,
        )
        _style_clean_axes(ax)
        _attach_right_founder_legend(ax, parent_ids, parent_colors, title=legend_title)

        karyo_path = _save_figure(fig, Path(f"{out_prefix}.mosaic_samples.{fmt}"), fmt, dpi)
        written.append(karyo_path)

    # 3C: Optional per-chromosome population mosaic heatmaps
    if per_chrom_mosaic and active_c:
        cmap = mcolors.ListedColormap([_UNCALLED_COLOR] + list(parent_colors))
        norm = mcolors.BoundaryNorm(np.arange(-1.5, n_parents + 0.5, 1.0), cmap.N)
        for c in active_c:
            pm = pop_by_c[c]
            cname = chrom_names[c]
            n_s = int(pm["n_samples"])
            n_b = int(pm["n_bins"])
            chrom_mb = max(float(pm["chrom_length_bp"]) / 1_000_000.0, 1e-6)
            mat = np.asarray(pm["parent_matrix"], dtype=np.int16).reshape((n_s, n_b))
            order, chrom_cluster_label = _cluster_pop_mosaic_rows(mat)
            mat_sorted = mat[order, :]

            fig, ax = plt.subplots(figsize=(12.0, 6.2))
            ax.imshow(
                mat_sorted,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
                extent=(0.0, chrom_mb, n_s, 0.0),
            )
            ax.set_xlabel(f"Chromosome {cname} physical position (Mb)", fontsize=10)
            ax.set_ylabel(f"Target lines ({chrom_cluster_label}, n={n_s})", fontsize=10)
            ax.set_title(
                f"Population Ancestry Mosaic — {cname} ({ancestry_label})",
                fontsize=11.5,
                fontweight="bold",
                pad=10,
            )
            _style_clean_axes(ax)
            _attach_right_founder_legend(ax, parent_ids, parent_colors, title=legend_title)

            safe_cname = str(cname).replace("/", "_").replace(" ", "_")
            chrom_tag = safe_cname if safe_cname.lower().startswith("chr") else f"chr{safe_cname}"
            pop_path = _save_figure(
                fig, Path(f"{out_prefix}.mosaic_{chrom_tag}.{fmt}"), fmt, dpi
            )
            written.append(pop_path)

    if written:
        logger.info("Mosaic outputs written (%d files): %s", len(written), ", ".join(str(p) for p in written))
    return written


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Main Entry
# ---------------------------------------------------------------------------


def build_parser() -> CliArgumentParser:
    parser = CliArgumentParser(
        prog="jx postibd",
        description="Post-processing & visualization directly from .jhibd sparse ancestry archives.",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx postibd -i cubic.jhibd -o cubic_post -fmt png",
                "jx postibd -i cubic.jhibd -o cubic_post -map NAM.map -recomb -mosaic -fmt pdf",
                "jx postibd -i cubic.jhibd -o cubic_post -mosaic -fmt pdf",
            ]
        ),
    )

    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument(
        "-i",
        "--input",
        "-ibd",
        dest="input",
        required=True,
        help="Input .jhibd sparse archive (or prefix without .jhibd).",
    )
    io_group.add_argument(
        "-o",
        "--out",
        dest="out",
        default=None,
        help="Output prefix or directory (default: <input_stem>.postibd in current directory).",
    )
    io_group.add_argument(
        "-fmt",
        "--fmt",
        dest="fmt",
        default="png",
        choices=list(_SUPPORTED_FORMATS),
        help="Output figure format: png, pdf, svg, or tif (default: png).",
    )
    io_group.add_argument(
        "-dpi",
        "--dpi",
        dest="dpi",
        type=int,
        default=300,
        help="Figure resolution in DPI for raster formats (default: 300).",
    )

    plot_group = parser.add_argument_group(
        "Plot Switches (default: run all three when none is specified)"
    )
    plot_group.add_argument(
        "-recomb",
        "--recomb",
        action="store_true",
        help="Compute empirical recombination rate, write .recomb.window.tsv & .empirical.map, and plot hotspots (combined above mosaic when -mosaic is also set).",
    )
    plot_group.add_argument(
        "-segregation",
        "--segregation",
        action="store_true",
        help="Compute genome-wide and per-chromosome founder contribution table and stacked bar plot.",
    )
    plot_group.add_argument(
        "-mosaic",
        "--mosaic",
        action="store_true",
        help="Plot whole-genome clustered multi-chromosome population mosaic and representative sample karyotype (aligned below recombination tracks when -recomb is also set).",
    )

    opt_group = parser.add_argument_group("Analysis & Display Options")
    opt_group.add_argument(
        "-window",
        "--window-mb",
        dest="window_mb",
        type=float,
        default=1.0,
        help="Physical window size in Mb for recombination curves (default: 1.0).",
    )
    opt_group.add_argument(
        "--min-window-coverage",
        dest="min_window_coverage",
        type=float,
        default=0.10,
        help="Minimum fraction of callable haplotype-bp in a window for recombination rate estimation (default: 0.10).",
    )
    opt_group.add_argument(
        "-map",
        "--map",
        dest="map",
        default=None,
        help="Optional 4-column PLINK genetic map (.map) to overlay against empirical recombination rate.",
    )
    opt_group.add_argument(
        "-group",
        "--group",
        "--parent-group",
        dest="group",
        default=None,
        help="Optional 2-column whitespace/tab file mapping reference parent IDs to ancestry group names (e.g. A, B) for grouped coloring and contributions.",
    )
    opt_group.add_argument(
        "--generations",
        dest="generations",
        type=float,
        default=None,
        help="Optional effective meiosis generations G to scale effective cM/Mb to per-meiosis cM/Mb.",
    )
    opt_group.add_argument(
        "--samples",
        dest="samples",
        default=None,
        help="Comma-separated target sample IDs for the karyotype mosaic (default: representative quantiles).",
    )
    opt_group.add_argument(
        "--max-samples",
        dest="max_samples",
        type=int,
        default=12,
        help="Maximum representative samples in karyotype mosaic when --samples is omitted (default: 12).",
    )
    opt_group.add_argument(
        "--chrom",
        dest="chrom",
        default=None,
        help="Comma-separated chromosome(s) to restrict mosaic plots and also emit per-chromosome heatmaps.",
    )
    opt_group.add_argument(
        "--per-chrom-mosaic",
        dest="per_chrom_mosaic",
        action="store_true",
        help="Also write separate per-chromosome population mosaic figures alongside mosaic_genome.",
    )
    opt_group.add_argument(
        "--max-pop-samples",
        dest="max_pop_samples",
        type=int,
        default=0,
        help="Maximum lines plotted in population mosaic heatmaps, 0 = all samples (default: 0).",
    )
    opt_group.add_argument(
        "--pop-bins",
        dest="pop_bins",
        type=int,
        default=2000,
        help="Total horizontal bin resolution distributed across chromosomes by physical length (default: 2000).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())

    if args.window_mb <= 0.0 or not math.isfinite(args.window_mb):
        parser.error("--window-mb must be a positive finite number.")
    if not (0.0 <= args.min_window_coverage <= 1.0) or not math.isfinite(args.min_window_coverage):
        parser.error("--min-window-coverage must be within [0, 1].")
    if args.dpi <= 0:
        parser.error("--dpi must be > 0.")
    if args.generations is not None and (args.generations <= 0.0 or not math.isfinite(args.generations)):
        parser.error("--generations must be a positive finite number.")

    input_path = _resolve_input_jhibd(args.input)
    out_prefix = _resolve_output_prefix(args.out, input_path)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Default to all three plots if no switch was explicitly passed.
    any_switch = bool(args.recomb or args.segregation or args.mosaic)
    want_recomb = bool(args.recomb) if any_switch else True
    want_segregation = bool(args.segregation) if any_switch else True
    want_mosaic = bool(args.mosaic) if any_switch else True

    log_path = Path(f"{out_prefix}.postibd.log")
    logger = setup_logging(str(log_path))

    enabled_modes = [
        name
        for name, flag in (
            ("recomb", want_recomb),
            ("segregation", want_segregation),
            ("mosaic", want_mosaic),
        )
        if flag
    ]
    emit_cli_configuration(
        logger,
        app_title="JanusX - Post-IBD",
        config_title="POST-IBD CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "General",
                [
                    ("Input archive", str(input_path)),
                    ("Format / DPI", f"{args.fmt} ({args.dpi} dpi)"),
                    ("Modules", ", ".join(enabled_modes)),
                    ("Window (Mb)", f"{args.window_mb:g}"),
                    ("Min window cov", f"{args.min_window_coverage:g}"),
                    ("Reference map", str(args.map) if args.map else "embedded / none"),
                    ("Ancestry group", str(args.group) if args.group else "none"),
                ],
            )
        ],
        footer_rows=[("Output prefix", str(out_prefix))],
        line_max_chars=60,
    )

    t0 = time.perf_counter()
    parent_groups: dict[str, str] = {}
    if args.group:
        group_path = Path(str(args.group)).expanduser()
        if not group_path.is_file():
            raise FileNotFoundError(f"Ancestry group file not found: {args.group}")
        parent_groups = _read_parent_group_file(group_path)

    mosaic_samples = (
        [s.strip() for s in str(args.samples).split(",") if s.strip()]
        if args.samples
        else []
    )
    mosaic_chroms = (
        [s.strip() for s in str(args.chrom).split(",") if s.strip()]
        if args.chrom
        else []
    )

    rust_opts = {
        "input": str(input_path),
        "window_bp": max(1, int(round(float(args.window_mb) * 1_000_000.0))),
        "min_window_coverage": float(args.min_window_coverage),
        "want_recomb": want_recomb,
        "want_segregation": want_segregation,
        "want_mosaic": want_mosaic,
        "mosaic_samples": mosaic_samples,
        "max_mosaic_samples": max(1, int(args.max_samples)),
        "mosaic_chroms": mosaic_chroms,
        "pop_mosaic_bins": max(20, int(args.pop_bins)),
        "max_pop_samples": max(0, int(args.max_pop_samples)),
        "parent_groups": parent_groups,
    }

    with CliStatus("Reading .jhibd sparse archive & aggregating metrics...", enabled=use_spinner) as task:
        try:
            if not hasattr(jxrs, "haploid_ibd_post_extract"):
                raise RuntimeError(
                    "Compiled Rust extension is missing `haploid_ibd_post_extract`; "
                    "please rebuild the JanusX Rust extension."
                )
            raw_json = jxrs.haploid_ibd_post_extract(json.dumps(rust_opts))
            data = json.loads(raw_json)
            if parent_groups:
                if "grouped_parents" not in data:
                    raise RuntimeError(
                        "Compiled Rust extension does not support -group / parent_groups; "
                        "please rebuild the JanusX Rust extension."
                    )
                if not data["grouped_parents"]:
                    logger.warning(
                        "None of the %d reference parent IDs matched entries in the group file; "
                        "keeping individual founder labels.",
                        int(data["n_reference_samples"]),
                    )
            if want_mosaic and mosaic_chroms and not data["pop_mosaics"]:
                raise ValueError(
                    f"No ancestry data to plot for --chrom {','.join(mosaic_chroms)}."
                )
        except Exception:
            task.fail("Reading .jhibd sparse archive ...Failed")
            raise
        task.complete("Reading .jhibd sparse archive ...Finished")

    if bool(data.get("grouped_parents", False)):
        logger.info(
            "Loaded %s: %d lines, %d founders (%d ancestry categories), %d chroms, %d ancestry segments.",
            input_path.name,
            int(data["n_target_samples"]),
            int(data["n_reference_samples"]),
            len(data["reference_sample_ids"]),
            len(data["chrom_names"]),
            int(data["total_segments"]),
        )
    else:
        logger.info(
            "Loaded %s: %d lines, %d founders, %d chroms, %d ancestry segments.",
            input_path.name,
            int(data["n_target_samples"]),
            int(data["n_reference_samples"]),
            len(data["chrom_names"]),
            int(data["total_segments"]),
        )

    ext_map = None
    if args.map:
        map_path = Path(str(args.map)).expanduser()
        if not map_path.is_file():
            raise FileNotFoundError(f"Genetic map file not found: {args.map}")
        ext_map = _read_plink_map(map_path)

    parent_colors = _build_parent_colors(len(data["reference_sample_ids"]))
    all_outputs: list[Path] = []
    combine_recomb_mosaic = bool(
        want_recomb and want_mosaic and data.get("recomb_windows") and data.get("pop_mosaics")
    )
    recomb_ctx: Optional[dict[str, Any]] = None

    if want_recomb:
        status_msg = (
            "Computing recombination rate tables & empirical map..."
            if combine_recomb_mosaic
            else "Generating recombination rate map & hotspot plot..."
        )
        with CliStatus(status_msg, enabled=use_spinner) as task:
            try:
                recomb_paths, recomb_ctx = _run_recomb_outputs(
                    data=data,
                    out_prefix=out_prefix,
                    fmt=args.fmt,
                    dpi=args.dpi,
                    ext_map=ext_map,
                    generations=args.generations,
                    logger=logger,
                    write_plot=not combine_recomb_mosaic,
                )
                all_outputs.extend(recomb_paths)
            except Exception:
                task.fail("Generating recombination outputs ...Failed")
                raise
            task.complete("Generating recombination outputs ...Finished")

    if want_segregation:
        with CliStatus("Generating founder contribution table & plot...", enabled=use_spinner) as task:
            try:
                all_outputs.extend(
                    _run_segregation_outputs(
                        data=data,
                        out_prefix=out_prefix,
                        fmt=args.fmt,
                        dpi=args.dpi,
                        parent_colors=parent_colors,
                        logger=logger,
                    )
                )
            except Exception:
                task.fail("Generating founder contribution outputs ...Failed")
                raise
            task.complete("Generating founder contribution outputs ...Finished")

    if want_mosaic:
        mosaic_status = (
            "Rendering aligned recombination + chromosome mosaic figure..."
            if combine_recomb_mosaic
            else "Rendering chromosome mosaic paintings..."
        )
        with CliStatus(mosaic_status, enabled=use_spinner) as task:
            try:
                all_outputs.extend(
                    _run_mosaic_outputs(
                        data=data,
                        out_prefix=out_prefix,
                        fmt=args.fmt,
                        dpi=args.dpi,
                        parent_colors=parent_colors,
                        per_chrom_mosaic=bool(args.per_chrom_mosaic or len(mosaic_chroms) > 0),
                        logger=logger,
                        recomb_ctx=recomb_ctx if combine_recomb_mosaic else None,
                        ext_map=ext_map,
                        generations=args.generations,
                    )
                )
            except Exception:
                task.fail("Rendering chromosome mosaic paintings ...Failed")
                raise
            task.complete("Rendering chromosome mosaic paintings ...Finished")

    elapsed = time.perf_counter() - t0
    log_success(
        logger,
        f"postibd completed in {format_elapsed(elapsed)} ({len(all_outputs)} output files).",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
