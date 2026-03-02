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
from ._common.status import get_rich_spinner_name

# Ensure matplotlib uses a non-interactive backend.
for key in ["MPLBACKEND"]:
    if key in os.environ:
        del os.environ[key]

import matplotlib as mpl
mpl.use("Agg")
from janusx.bioplotkit import GWASPLOT, LDblock
from janusx.bioplotkit.geneplot import draw_gene_structure_records
from janusx.gfreader import load_genotype_chunks

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd
import numpy as np
import argparse
import re
import time
import socket
import sys
import colorsys
from contextlib import nullcontext
from typing import Any, Optional, Tuple
from janusx.gtools.reader import GFFQuery, bedreader, readanno
from joblib import Parallel, delayed
import warnings

_LEAD_SNP_INFO_COLS = ["allele0", "allele1", "maf", "beta", "se"]
_QQ_FIXED_RATIO = 5.0 / 4.0
_CONFIG_LINE_MAX_CHARS = 60
_CONFIG_OVERFLOW_MARK = "***"

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    _HAS_RICH_PROGRESS = True
except Exception:
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    TimeRemainingColumn = None  # type: ignore[assignment]
    _HAS_RICH_PROGRESS = False

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore[assignment]
    _HAS_TQDM = False

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    _HAS_RICH_CONSOLE = True
except Exception:
    Console = None  # type: ignore[assignment]
    Group = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]
    box = None  # type: ignore[assignment]
    _HAS_RICH_CONSOLE = False


def _emit_info_to_file_handlers(logger: logging.Logger, message: str) -> None:
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        message,
        args=(),
        exc_info=None,
    )
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.handle(record)


def _emit_info_to_stream_handlers(logger: logging.Logger, message: str) -> None:
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        message,
        args=(),
        exc_info=None,
    )
    for handler in logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            handler.handle(record)


def _truncate_config_line(
    text: object,
    *,
    max_chars: int = _CONFIG_LINE_MAX_CHARS,
    overflow_mark: str = _CONFIG_OVERFLOW_MARK,
) -> str:
    s = str(text)
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    mark = str(overflow_mark)
    if mark == "":
        return s[:max_chars]
    if max_chars <= len(mark):
        return mark[:max_chars]
    keep = max_chars - len(mark)
    return s[:keep] + mark


def _render_postgwas_config_rich(
    *,
    title: str,
    host: str,
    base_rows: list[tuple[str, str]],
    merge_map_rows: list[tuple[str, str]],
    vis_rows: Optional[list[tuple[str, str]]],
    anno_rows: Optional[list[tuple[str, str]]],
    output_prefix: str,
    threads_text: str,
    key_width: int = 20,
) -> bool:
    if not (_HAS_RICH_CONSOLE and sys.stdout.isatty()):
        return False

    try:
        assert Console is not None
        assert Group is not None
        assert Panel is not None
        assert Table is not None
        assert Text is not None
        assert box is not None

        def _kv_table(rows: list[tuple[str, str]]) -> Any:
            t = Table(
                show_header=False,
                box=box.SIMPLE,
                pad_edge=False,
                expand=False,
            )
            key_w = max(8, int(key_width))
            t.add_column(style="bold cyan", no_wrap=True, width=key_w, justify="left")
            t.add_column(style="white", no_wrap=True, justify="left")
            for k, v in rows:
                k_txt = str(k)
                v_max = max(1, _CONFIG_LINE_MAX_CHARS - key_w - 2)
                v_txt = _truncate_config_line(v, max_chars=v_max)
                t.add_row(k_txt, v_txt)
            return t

        parts: list[Any] = [
            Text(title, style="bold"),
            Text(f"Host: {host}"),
            Text(""),
            Text("General", style="bold cyan"),
            _kv_table(base_rows),
        ]

        if len(merge_map_rows) > 0:
            map_table = Table(
                show_header=True,
                box=box.SIMPLE,
                pad_edge=False,
                expand=True,
            )
            map_table.add_column("ID", style="bold cyan", no_wrap=True, width=6)
            map_table.add_column("GWAS file", style="white")
            for idx, file in merge_map_rows:
                idx_txt = str(idx)
                file_max = max(1, _CONFIG_LINE_MAX_CHARS - len(idx_txt) - 1)
                map_table.add_row(idx_txt, _truncate_config_line(file, max_chars=file_max))
            parts.extend([Text(""), Text("Merge trait-id mapping", style="bold cyan"), map_table])

        parts.append(Text(""))
        if vis_rows is None:
            parts.extend([Text("Visualization", style="bold cyan"), Text("disabled", style="white")])
        else:
            parts.extend([Text("Visualization", style="bold cyan"), _kv_table(vis_rows)])

        if anno_rows is not None:
            parts.extend([Text(""), Text("Annotation", style="bold cyan"), _kv_table(anno_rows)])

        parts.extend(
            [
                Text(""),
                Text(_truncate_config_line(f"Output prefix: {output_prefix}"), style="white"),
                Text(_truncate_config_line(f"Threads: {threads_text}"), style="white"),
            ]
        )

        panel = Panel(
            Group(*parts),
            title="POST-GWAS CONFIG",
            border_style="green",
            expand=False,
        )
        Console().print(panel)
        return True
    except Exception:
        return False


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


def _scaled_fontsize_for_manhattan(
    manh_ratio: Optional[float],
    *,
    width_in: float = 8.0,
    base_font: float = 6.0,
    ref_ratio: float = 2.0,
    min_scale: float = 0.55,
) -> float:
    """
    Scale font size with Manhattan panel height.
    As panel height decreases (ratio increases), font size decreases.
    """
    if manh_ratio is None:
        return float(base_font)
    try:
        ratio = float(manh_ratio)
    except Exception:
        return float(base_font)
    if not np.isfinite(ratio) or ratio <= 0:
        return float(base_font)

    ref_h = float(width_in) / float(ref_ratio)
    now_h = float(width_in) / float(ratio)
    if not (np.isfinite(ref_h) and ref_h > 0 and np.isfinite(now_h) and now_h > 0):
        return float(base_font)
    scale = float(np.sqrt(now_h / ref_h))
    scale = float(np.clip(scale, float(min_scale), 1.0))
    return float(base_font) * scale


def _parse_ldblock_spec(
    value: object,
    name: str,
    logger: logging.Logger,
) -> tuple[float, Optional[tuple[float, float]]]:
    """
    Parse ldblock option payload.
    Supports:
      - ratio only: "2", "5/4"
        -> x-span defaults to 0:1 (full Manhattan width)
      - x-span only (fraction of Manhattan width): "0.2:0.8" or "0.2-0.8"
        -> ratio defaults to 2.0
    """
    if value is None:
        raise ValueError(f"{name} is None.")
    text = str(value).strip()
    if text == "":
        raise ValueError(f"{name} is empty.")

    m = re.match(r"^([0-9]*\.?[0-9]+)\s*(?:-|:)\s*([0-9]*\.?[0-9]+)$", text)
    if m is not None:
        x0 = float(m.group(1))
        x1 = float(m.group(2))
        if x0 < 0 or x1 < 0 or x0 > 1 or x1 > 1:
            raise ValueError(
                f"{name} x-span must be within [0, 1]: {text}."
            )
        if np.isclose(x0, x1):
            raise ValueError(f"{name} x-span start and end cannot be equal: {text}.")
        if x0 > x1:
            logger.warning(
                f"Warning: {name} x-span start > end ({x0} > {x1}); swapped to {x1}-{x0}."
            )
            x0, x1 = x1, x0
        return 2.0, (float(x0), float(x1))

    return _parse_ratio(text, name), (0.0, 1.0)


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


def _expand_single_pallete_color(base_color: str) -> list[str]:
    """
    Expand one color to two colors by grayscale(lightness) direction.
    - dark input  -> generate a darker mate
    - light input -> generate a lighter mate
    Returns [light, dark].
    """
    base = mcolors.to_hex(mcolors.to_rgba(base_color))
    gray = _relative_luminance(base)
    # Grayscale threshold for light/dark split.
    # Use stronger contrast so generated light/dark are visually distinct.
    if gray < 0.5:
        # dark input: push a clearly lighter mate + a deeper dark mate
        light = _blend_hex_color(base, "#ffffff", 0.85)
        dark = _blend_hex_color(base, "#000000", 0.35)
    else:
        # light input: push a clearly darker mate + an even lighter mate
        light = _blend_hex_color(base, "#ffffff", 0.35)
        dark = _blend_hex_color(base, "#000000", 0.85)
    return [light, dark]


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


def _blend_hex_color(c1: str, c2: str, ratio_to_c2: float) -> str:
    """Blend c1->c2 with ratio in [0, 1], return hex color."""
    t = float(np.clip(float(ratio_to_c2), 0.0, 1.0))
    rgb1 = np.asarray(mcolors.to_rgb(c1), dtype=float)
    rgb2 = np.asarray(mcolors.to_rgb(c2), dtype=float)
    out = (1.0 - t) * rgb1 + t * rgb2
    return mcolors.to_hex(out)


def _relative_luminance(color: str) -> float:
    """Simple RGB luminance in [0, 1] for light/dark ordering."""
    r, g, b = mcolors.to_rgb(color)
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def _resolve_two_color_style(
    spec: Optional[Tuple[str, Any]],
) -> Optional[dict[str, Any]]:
    """
    Return two-color style for LD/gene:
      - exactly two colors -> use as-is
      - single color       -> auto-expand by grayscale direction
    """
    if spec is None:
        return None

    mode, payload = spec
    colors: list[str]
    if mode == "list":
        colors = [mcolors.to_hex(mcolors.to_rgba(c)) for c in list(payload)]
        if len(colors) == 1:
            colors = _expand_single_pallete_color(colors[0])
        elif len(colors) != 2:
            return None
    elif mode == "cmap":
        cmap = plt.get_cmap(str(payload))
        if int(getattr(cmap, "N", 256)) != 2:
            return None
        colors = [mcolors.to_hex(cmap(0)), mcolors.to_hex(cmap(1))]
    else:
        return None

    lum0 = _relative_luminance(colors[0])
    lum1 = _relative_luminance(colors[1])
    if lum0 >= lum1:
        light, dark = colors[0], colors[1]
    else:
        light, dark = colors[1], colors[0]

    # Avoid fully-white fill in gene rectangles by nudging it 20% toward dark.
    if _relative_luminance(light) >= 0.98:
        light = _blend_hex_color(light, dark, 0.2)

    ld_cmap = mcolors.LinearSegmentedColormap.from_list(
        "janusx_ld_bicolor",
        [colors[0], colors[1]],
    )
    return {
        "colors": colors,
        "ld_cmap": ld_cmap,
        "gene_block_color": light,
        "gene_line_color": dark,
    }


def _resolve_manhattan_colors(spec: Optional[Tuple[str, Any]], n_chr: int) -> Optional[list[str]]:
    if spec is None:
        return None
    mode, payload = spec
    if mode == "list":
        return list(payload)
    cmap_name = str(payload).strip()
    cmap = plt.get_cmap(cmap_name)
    # For discrete palettes (tab10/tab20/Set*, etc.), cycle native bins directly.
    # This avoids adjacent duplicate colors when n_chr > number of bins.
    if getattr(cmap, "N", 256) <= 32:
        n_bins = max(1, int(cmap.N))
        colors = [mcolors.to_hex(cmap(i % n_bins)) for i in range(n_chr)]
    else:
        colors = [mcolors.to_hex(cmap(i / max(1, n_chr - 1))) for i in range(n_chr)]
    if cmap_name.lower() == "tab10":
        return [_desaturate_color(c, 0.80) for c in colors]
    return colors


def _desaturate_color(color: str, sat_scale: float = 0.80) -> str:
    r, g, b = mcolors.to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s2 = float(np.clip(float(s) * float(sat_scale), 0.0, 1.0))
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s2, v)
    return mcolors.to_hex((r2, g2, b2))


def _resolve_merge_series_colors(
    spec: Optional[Tuple[str, Any]],
    n_series: int,
) -> list[str]:
    if n_series <= 0:
        return []

    if spec is None:
        cmap_name = "tab10" if n_series <= 10 else "tab20"
        cmap = plt.get_cmap(cmap_name)
        n_bins = max(1, int(getattr(cmap, "N", 10 if n_series <= 10 else 20)))
        colors = [mcolors.to_hex(cmap(i % n_bins)) for i in range(n_series)]
        if cmap_name.lower() == "tab10":
            colors = [_desaturate_color(c, 0.90) for c in colors]
        return colors

    mode, payload = spec
    if mode == "list":
        colors = [mcolors.to_hex(mcolors.to_rgba(c)) for c in list(payload)]
        if len(colors) == 0:
            cmap = plt.get_cmap("tab20")
            n_bins = max(1, int(getattr(cmap, "N", 20)))
            return [mcolors.to_hex(cmap(i % n_bins)) for i in range(n_series)]
        return [colors[i % len(colors)] for i in range(n_series)]

    cmap_name = str(payload).strip()
    cmap = plt.get_cmap(cmap_name)
    if getattr(cmap, "N", 256) <= 32:
        n_bins = max(1, int(cmap.N))
        colors = [mcolors.to_hex(cmap(i % n_bins)) for i in range(n_series)]
    else:
        colors = [mcolors.to_hex(cmap(i / max(1, n_series - 1))) for i in range(n_series)]
    if cmap_name.lower() == "tab10":
        colors = [_desaturate_color(c, 0.90) for c in colors]
    return colors


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


def _parse_ldclump_window_bp(value: object) -> int:
    """
    Parse LD clump window size to bp.

    Supported units:
      - kb / k (default when unit omitted)
      - mb / m
      - bp / b
    """
    text = str(value).strip().lower()
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([a-z]*)$", text)
    if m is None:
        raise ValueError(
            f"Invalid --LDclump window: {value}. "
            "Use formats like 500kb, 0.5mb, or 200000bp."
        )
    raw_val = float(m.group(1))
    unit = m.group(2)
    if raw_val <= 0:
        raise ValueError("--LDclump window must be > 0.")
    if unit in {"", "kb", "k"}:
        factor = 1_000.0
    elif unit in {"mb", "m"}:
        factor = 1_000_000.0
    elif unit in {"bp", "b"}:
        factor = 1.0
    else:
        raise ValueError(
            f"Unsupported --LDclump window unit: {unit}. "
            "Use kb/mb/bp."
        )
    out = int(round(raw_val * factor))
    if out <= 0:
        raise ValueError("--LDclump window must be > 0 bp.")
    return out


def _parse_ldclump_spec(value: object) -> tuple[int, float]:
    """
    Parse --LDclump payload: [window, r2].
    """
    if value is None:
        raise ValueError("--LDclump is None.")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "Invalid --LDclump format. Use: --LDclump <window> <r2> "
            "(e.g. --LDclump 500kb 0.8)."
        )
    window_bp = _parse_ldclump_window_bp(value[0])
    try:
        r2_thr = float(value[1])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid --LDclump r2 threshold: {value[1]}") from e
    if not (0.0 <= r2_thr <= 1.0):
        raise ValueError("--LDclump r2 threshold must be within [0, 1].")
    return window_bp, r2_thr


def _parse_ylim_spec(value: object) -> tuple[Optional[float], Optional[float]]:
    """
    Parse y-range spec for Manhattan plotting.
    Supports:
      - "6" -> (0.0, 6.0)
      - "0:6" / "0-6" / "[0:6]" -> (0.0, 6.0)
      - "2:" / "2-" -> (2.0, None)     # upper auto
      - ":6" / "-6" -> (None, 6.0)     # lower auto
    """
    if value is None:
        raise ValueError("--ylim is None.")
    text = str(value).strip()
    if text == "":
        raise ValueError("--ylim is empty.")
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    # Range form: <lo:hi>, <lo:>, <:hi> (also accepts '-')
    if (":" in text) or ("-" in text):
        m = re.match(r"^\s*([0-9]*\.?[0-9]*)\s*(?:-|:)\s*([0-9]*\.?[0-9]*)\s*$", text)
        if m is None:
            raise ValueError(
                f"Invalid --ylim format: {value}. Use <max>, <min:max>, <min:>, or <:max> "
                "(e.g. 6, 0:6, 2:, :6)."
            )
        lo_txt = m.group(1).strip()
        hi_txt = m.group(2).strip()
        if lo_txt == "" and hi_txt == "":
            raise ValueError("Invalid --ylim: at least one bound must be provided.")
        lo = float(lo_txt) if lo_txt != "" else None
        hi = float(hi_txt) if hi_txt != "" else None
    else:
        # Single number means [0, max]
        try:
            hi = float(text)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid --ylim format: {value}. Use <max>, <min:max>, <min:>, or <:max>."
            ) from e
        lo = 0.0

    if lo is not None:
        if not np.isfinite(lo):
            raise ValueError("--ylim lower bound must be finite.")
        if lo < 0:
            raise ValueError("--ylim lower bound must be >= 0.")
    if hi is not None:
        if not np.isfinite(hi):
            raise ValueError("--ylim upper bound must be finite.")
        if hi <= 0:
            raise ValueError("--ylim upper bound must be > 0.")
    if lo is not None and hi is not None and hi <= lo:
        raise ValueError("--ylim upper bound must be greater than lower bound.")
    return (float(lo) if lo is not None else None, float(hi) if hi is not None else None)


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
    ax.set_xlabel(None)
    c = str(chrom_label)
    ax._janusx_loc_left_label = f"{c}:{left / 1_000_000:g}Mb"   # type: ignore[attr-defined]
    ax._janusx_loc_right_label = f"{c}:{right / 1_000_000:g}Mb"  # type: ignore[attr-defined]


def _apply_multi_bimrange_manhattan_axis(
    ax: plt.Axes,
    layout: list[dict[str, object]],
    *,
    label_fontsize: Optional[float] = None,
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
    loc_fontsize = 5.0 if label_fontsize is None else float(label_fontsize)
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
            -0.06,
            prev_end_lab,
            transform=trans,
            ha="right",
            va="top",
            fontsize=loc_fontsize,
            clip_on=False,
            zorder=40,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 0.2,
            },
        )
        ax.text(
            xb + x_shift,
            -0.06,
            next_start_lab,
            transform=trans,
            ha="left",
            va="top",
            fontsize=loc_fontsize,
            clip_on=False,
            zorder=40,
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


def _show_end_locs_without_xticks(
    ax: plt.Axes,
    *,
    label_fontsize: Optional[float] = None,
) -> None:
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

    ax.set_xlabel(None)
    ax.set_xticks([])
    ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
    trans = ax.transAxes
    loc_fontsize = 5.0 if label_fontsize is None else float(label_fontsize)
    ax.text(
        0.0,
        -0.06,
        left_lab,
        transform=trans,
        ha="left",
        va="top",
        fontsize=loc_fontsize,
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
            1.0,
            -0.06,
            right_lab,
            transform=trans,
            ha="right",
            va="top",
            fontsize=loc_fontsize,
            clip_on=False,
            zorder=40,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 0.2,
            },
        )


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


def _lead_vs_all_r2(geno_block: np.ndarray) -> np.ndarray:
    """
    Compute r^2 between the lead SNP (row 0) and all SNP rows.

    This avoids building a full correlation matrix and is much cheaper for
    LD clumping where only lead-vs-window correlations are needed.
    """
    x = np.asarray(geno_block, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    m, n = x.shape
    if n <= 1:
        out = np.zeros((m,), dtype=np.float32)
        out[0] = 1.0
        return out

    x0 = x[0]  # lead SNP
    n_f = float(n)

    # Pearson correlation via sum products:
    # r = (n*sum(xy)-sum(x)sum(y)) / sqrt((n*sum(x^2)-sum(x)^2)*(n*sum(y^2)-sum(y)^2))
    sx = np.sum(x, axis=1, dtype=np.float64)
    sx2 = np.einsum("ij,ij->i", x, x, dtype=np.float64, optimize=True)
    s0 = float(sx[0])
    s02 = float(sx2[0])
    sxx0 = np.einsum("ij,j->i", x, x0, dtype=np.float64, optimize=True)

    num = n_f * sxx0 - sx * s0
    den_left = n_f * sx2 - np.square(sx)
    den_right = max(0.0, n_f * s02 - s0 * s0)
    den = np.sqrt(np.maximum(den_left, 0.0) * den_right)

    corr = np.zeros((m,), dtype=np.float64)
    valid = den > 0.0
    corr[valid] = num[valid] / den[valid]
    corr = np.clip(corr, -1.0, 1.0)
    corr[0] = 1.0

    r2 = np.square(corr)
    r2 = np.nan_to_num(r2, nan=0.0, posinf=0.0, neginf=0.0)
    return r2.astype(np.float32, copy=False)


def _clean_anno_token(value: object) -> str:
    if value is None:
        return "NA"
    if pd.isna(value):
        return "NA"
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return "NA"
    return text


def _merge_anno_value(base: str, value: str) -> str:
    if value == "NA":
        return base
    if base == "NA":
        return value
    existing = base.split("|")
    if value in existing:
        return base
    return f"{base}|{value}"


def _format_gene_annotation_dict(hits: pd.DataFrame) -> str:
    """
    Format annotation hits as:
      {gene1:[description,additionaldesc], gene2:[description,additionaldesc]}
    """
    if hits is None or hits.shape[0] == 0:
        return "{}"
    out: dict[str, list[str]] = {}
    for _, row in hits.iterrows():
        gene = _clean_anno_token(row.iloc[3] if hits.shape[1] > 3 else "NA")
        desc = _clean_anno_token(row.iloc[4] if hits.shape[1] > 4 else "NA")
        add_desc = _clean_anno_token(row.iloc[5] if hits.shape[1] > 5 else "NA")
        if gene not in out:
            out[gene] = [desc, add_desc]
        else:
            out[gene][0] = _merge_anno_value(out[gene][0], desc)
            out[gene][1] = _merge_anno_value(out[gene][1], add_desc)
    return str(out)


def _format_clump_sites(sites: list[tuple[str, int]]) -> str:
    if sites is None or len(sites) == 0:
        return ""
    return ";".join([f"{str(chrom)}_{int(pos)}" for chrom, pos in sites])


def _ldclump_significant_snps(
    df_sig: pd.DataFrame,
    *,
    chr_col: str,
    pos_col: str,
    p_col: str,
    genofile: str,
    window_bp: int,
    r2_thr: float,
    logger: logging.Logger,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[str, int], list[tuple[str, int]]]]:
    """
    LD-clump threshold-passing SNPs and keep lead SNPs only in annotation output.
    """
    if df_sig.shape[0] == 0:
        out_empty = df_sig.set_index([chr_col, pos_col], drop=True)
        out_empty["start"] = pd.Series(dtype=int)
        out_empty["end"] = pd.Series(dtype=int)
        out_empty["nsnps"] = pd.Series(dtype=int)
        out_empty["MeanR2"] = pd.Series(dtype=float)
        out_empty["LDclump"] = pd.Series(dtype=str)
        return out_empty, {}

    work = df_sig[[chr_col, pos_col, p_col]].copy()
    work[chr_col] = work[chr_col].astype(str)
    work[pos_col] = pd.to_numeric(work[pos_col], errors="coerce")
    work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
    work = work.dropna(subset=[pos_col, p_col]).copy()
    if work.shape[0] == 0:
        out_empty = work.set_index([chr_col, pos_col], drop=True)
        out_empty["start"] = pd.Series(dtype=int)
        out_empty["end"] = pd.Series(dtype=int)
        out_empty["nsnps"] = pd.Series(dtype=int)
        out_empty["MeanR2"] = pd.Series(dtype=float)
        out_empty["LDclump"] = pd.Series(dtype=str)
        return out_empty, {}

    work[pos_col] = work[pos_col].astype(int)
    work = (
        work.sort_values(p_col, ascending=True)
        .drop_duplicates(subset=[chr_col, pos_col], keep="first")
        .reset_index(drop=True)
    )
    if work.shape[0] == 0:
        out_empty = work.set_index([chr_col, pos_col], drop=True)
        out_empty["start"] = pd.Series(dtype=int)
        out_empty["end"] = pd.Series(dtype=int)
        out_empty["nsnps"] = pd.Series(dtype=int)
        out_empty["MeanR2"] = pd.Series(dtype=float)
        out_empty["LDclump"] = pd.Series(dtype=str)
        return out_empty, {}

    all_keys = [
        (str(c), int(p))
        for c, p in zip(work[chr_col].tolist(), work[pos_col].tolist())
    ]
    chrom_arr = work[chr_col].to_numpy(dtype=str)
    pos_arr = work[pos_col].to_numpy(dtype=np.int64)

    # Preload all threshold-passing SNP genotypes once, then reuse in memory.
    # This avoids repeated random-access reads for each lead SNP.
    preloaded_geno: Optional[np.ndarray] = None
    key_to_row: dict[tuple[str, int], int] = {}
    try:
        logger.info(
            f"Preloading genotype rows for LD clump: {len(all_keys)} SNPs..."
        )
        preloaded_chunks: list[np.ndarray] = []
        for chunk, _sites in load_genotype_chunks(
            genofile,
            chunk_size=max(1, min(20_000, len(all_keys))),
            maf=0.0,
            missing_rate=1.0,
            impute=True,
            snp_sites=all_keys,
        ):
            preloaded_chunks.append(np.asarray(chunk, dtype=np.float32))
        if len(preloaded_chunks) > 0:
            preloaded_geno = np.vstack(preloaded_chunks).astype(np.float32, copy=False)
        if preloaded_geno is None or preloaded_geno.shape[0] != len(all_keys):
            raise RuntimeError(
                "preloaded genotype row count does not match requested SNPs"
            )
        key_to_row = {k: i for i, k in enumerate(all_keys)}
    except Exception as e:
        preloaded_geno = None
        key_to_row = {}
        logger.warning(
            "Warning: Failed to preload all LD-clump genotypes; "
            "falling back to per-lead genotype loading. "
            f"Reason: {e}"
        )

    remaining: set[tuple[str, int]] = set(all_keys)
    key_to_p: dict[tuple[str, int], float] = {
        (str(c), int(p)): float(v)
        for c, p, v in zip(work[chr_col].tolist(), work[pos_col].tolist(), work[p_col].tolist())
    }
    kept_rows: list[tuple[str, int, float, int, int, int, float, str]] = []
    clump_dict: dict[tuple[str, int], list[tuple[str, int]]] = {}

    warn_count = 0
    warn_limit = 5
    use_rich_progress = bool(show_progress and _HAS_RICH_PROGRESS and sys.stdout.isatty())
    progress = None
    task_id = None
    progress_tqdm = None
    if use_rich_progress:
        try:
            progress = Progress(
                SpinnerColumn(
                    spinner_name=get_rich_spinner_name(),
                    style="cyan",
                ),
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
            )
        except Exception:
            progress = None
            use_rich_progress = False

    with (progress if progress is not None else nullcontext()):
        if progress is not None:
            task_id = progress.add_task("LD clumping...", total=int(work.shape[0]))
        elif bool(show_progress and _HAS_TQDM):
            progress_tqdm = tqdm(
                total=int(work.shape[0]),
                desc="LD clumping",
                unit="snp",
                leave=True,
                dynamic_ncols=True,
            )

        try:
            for _, row in work.iterrows():
                try:
                    lead_chr = str(row[chr_col])
                    lead_pos = int(row[pos_col])
                    lead_key = (lead_chr, lead_pos)
                    if lead_key not in remaining:
                        continue

                    start = int(lead_pos - window_bp)
                    end = int(lead_pos + window_bp)
                    win_mask = (
                        (chrom_arr == lead_chr)
                        & (pos_arr >= start)
                        & (pos_arr <= end)
                    )
                    win_idx = np.flatnonzero(win_mask)
                    snps = [
                        all_keys[int(i)]
                        for i in win_idx
                        if all_keys[int(i)] in remaining
                    ]
                    if lead_key in snps:
                        snps = [lead_key] + [x for x in snps if x != lead_key]
                    else:
                        snps = [lead_key] + snps

                    clumped = [lead_key]
                    mean_r2 = 1.0
                    if len(snps) > 1:
                        try:
                            if preloaded_geno is not None:
                                idx_arr = np.fromiter(
                                    (key_to_row[k] for k in snps),
                                    dtype=np.int64,
                                    count=len(snps),
                                )
                                geno_block = preloaded_geno[idx_arr, :]
                            else:
                                geno_block = None
                                for chunk, _sites in load_genotype_chunks(
                                    genofile,
                                    chunk_size=max(1, len(snps)),
                                    maf=0.0,
                                    missing_rate=1.0,
                                    impute=True,
                                    snp_sites=snps,
                                ):
                                    geno_block = np.asarray(chunk, dtype=np.float32)
                                    break
                            if geno_block is not None and geno_block.shape[0] == len(snps):
                                r2 = _lead_vs_all_r2(geno_block)
                                clumped = [
                                    snps[i]
                                    for i, keep in enumerate(r2 >= float(r2_thr))
                                    if bool(keep)
                                ]
                                if lead_key not in clumped:
                                    clumped.insert(0, lead_key)
                                r2_map = {snps[i]: float(r2[i]) for i in range(len(snps))}
                                mean_r2 = float(
                                    np.mean([float(r2_map.get(k, 1.0)) for k in clumped])
                                )
                            else:
                                if warn_count < warn_limit:
                                    logger.warning(
                                        "Warning: LDclump genotype rows do not match requested SNP count; "
                                        f"fallback to keep lead SNP only for {lead_chr}:{lead_pos}."
                                    )
                                warn_count += 1
                        except Exception as e:
                            if warn_count < warn_limit:
                                logger.warning(
                                    "Warning: LDclump genotype lookup failed for "
                                    f"{lead_chr}:{lead_pos}; fallback to keep lead SNP only. "
                                    f"Reason: {e}"
                                )
                            warn_count += 1
                            clumped = [lead_key]

                    clumped = sorted(
                        clumped,
                        key=lambda k: (float(key_to_p.get(k, np.inf)), int(k[1])),
                    )
                    clump_dict[lead_key] = clumped
                    ld_start = int(min([int(x[1]) for x in clumped])) if len(clumped) > 0 else int(lead_pos)
                    ld_end = int(max([int(x[1]) for x in clumped])) if len(clumped) > 0 else int(lead_pos)
                    nsnps = int(len(clumped))
                    kept_rows.append(
                        (
                            lead_chr,
                            lead_pos,
                            float(row[p_col]),
                            ld_start,
                            ld_end,
                            nsnps,
                            mean_r2,
                            _format_clump_sites(clumped),
                        )
                    )
                    for key in clumped:
                        if key in remaining:
                            remaining.remove(key)
                finally:
                    if progress is not None and task_id is not None:
                        progress.advance(task_id, 1)
                    if progress_tqdm is not None:
                        progress_tqdm.update(1)
        finally:
            if progress_tqdm is not None:
                progress_tqdm.close()

    if warn_count > warn_limit:
        logger.warning(
            f"Warning: LDclump warnings truncated; {warn_count - warn_limit} more similar warnings omitted."
        )

    out_df = pd.DataFrame(
        kept_rows,
        columns=[chr_col, pos_col, p_col, "start", "end", "nsnps", "MeanR2", "LDclump"],
    )
    out_df = out_df.set_index([chr_col, pos_col], drop=True)
    return out_df, clump_dict


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


def _build_layout_from_bimrange_tuples(
    bimrange_tuples: list[tuple[str, int, int]],
) -> list[dict[str, object]]:
    if len(bimrange_tuples) == 0:
        return []
    seg_defs: list[dict[str, object]] = []
    for i, (chrom, start, end) in enumerate(bimrange_tuples):
        seg_defs.append(
            {
                "id": int(i),
                "chrom": str(chrom),
                "chrom_norm": _normalize_chr(chrom),
                "start": int(start),
                "end": int(end),
                "length": float(max(1, int(end) - int(start))),
            }
        )
    return _build_bimrange_layout(seg_defs)


def _load_gene_like_records_from_anno(
    annofile: str,
    bimrange_tuples: list[tuple[str, int, int]],
    logger: logging.Logger,
    gff_query: Optional[GFFQuery] = None,
) -> pd.DataFrame:
    """
    Load gene-structure-like records from GFF/BED for selected bimranges.

    Output columns:
      chrom_norm, feature, start, end, strand, attribute
    """
    out_cols = ["chrom_norm", "feature", "start", "end", "strand", "attribute"]
    if not annofile or len(bimrange_tuples) == 0:
        return pd.DataFrame(columns=out_cols)

    suffix = str(annofile).replace(".gz", "").split(".")[-1].lower()
    features = ["gene", "five_prime_UTR", "three_prime_UTR", "CDS"]

    if suffix in {"gff", "gff3"}:
        q = gff_query if gff_query is not None else GFFQuery.from_file(annofile)
        chunks: list[pd.DataFrame] = []
        for chrom, start, end in bimrange_tuples:
            hit = q.query_range(
                chrom=chrom,
                start=int(start),
                end=int(end),
                features=features,
                attr="ID",
            )
            if hit.shape[0] == 0:
                continue
            chunk = hit.loc[:, ["chrom_norm", "feature", "start", "end", "strand", "attribute"]].copy()
            chunks.append(chunk)
        if len(chunks) == 0:
            return pd.DataFrame(columns=out_cols)
        out = pd.concat(chunks, axis=0, ignore_index=True)
        out["start"] = pd.to_numeric(out["start"], errors="coerce").astype("Int64")
        out["end"] = pd.to_numeric(out["end"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["start", "end"]).copy()
        out["start"] = out["start"].astype(int)
        out["end"] = out["end"].astype(int)
        return out[out_cols]

    if suffix == "bed":
        bed = bedreader(annofile)
        if bed.shape[0] == 0:
            return pd.DataFrame(columns=out_cols)
        s = pd.to_numeric(bed[1], errors="coerce")
        e = pd.to_numeric(bed[2], errors="coerce")
        valid = s.notna() & e.notna()
        if not bool(valid.any()):
            return pd.DataFrame(columns=out_cols)

        bed_v = bed.loc[valid].copy()
        s_v = s.loc[valid].astype(int)
        e_v = e.loc[valid].astype(int)
        starts = np.minimum(s_v.to_numpy(dtype=np.int64), e_v.to_numpy(dtype=np.int64))
        ends = np.maximum(s_v.to_numpy(dtype=np.int64), e_v.to_numpy(dtype=np.int64))
        chroms = bed_v[0].astype(str).map(_normalize_chr).to_numpy(dtype=object)
        if 3 in bed_v.columns:
            names = bed_v[3].astype(str).to_numpy(dtype=object)
        else:
            names = np.array([""] * len(bed_v), dtype=object)

        rows: list[dict[str, object]] = []
        for chrom, start, end, name in zip(chroms, starts, ends, names):
            gene_id = str(name).strip()
            if gene_id == "" or gene_id.lower() == "nan":
                gene_id = f"{chrom}:{int(start)}-{int(end)}"
            attr = [gene_id]
            # BED has no canonical feature segmentation; build gene+CDS proxy.
            # Strand is explicitly kept as '.' per requirement.
            rows.append(
                {
                    "chrom_norm": str(chrom),
                    "feature": "gene",
                    "start": int(start),
                    "end": int(end),
                    "strand": ".",
                    "attribute": attr,
                }
            )
            rows.append(
                {
                    "chrom_norm": str(chrom),
                    "feature": "CDS",
                    "start": int(start),
                    "end": int(end),
                    "strand": ".",
                    "attribute": attr,
                }
            )
        return pd.DataFrame(rows, columns=out_cols)

    logger.warning(
        f"Warning: Unsupported annotation format for gene-structure plotting: {annofile}. "
        "Only .gff/.gff3/.bed are supported."
    )
    return pd.DataFrame(columns=out_cols)


def _project_gene_records_to_plot_x(
    records: pd.DataFrame,
    bimrange_tuples: list[tuple[str, int, int]],
    layout: list[dict[str, object]],
    *,
    use_segmented_x: bool,
) -> pd.DataFrame:
    """
    Clip records to selected bimranges and project start/end into plot-x coordinates.
    """
    out_cols = ["feature", "strand", "attribute", "x_start", "x_end"]
    if records.shape[0] == 0 or len(bimrange_tuples) == 0:
        return pd.DataFrame(columns=out_cols)

    seg_by_key: dict[tuple[str, int], dict[str, object]] = {}
    if use_segmented_x:
        for seg in layout:
            sid = int(seg["id"])
            seg_by_key[(str(seg["chrom_norm"]), sid)] = seg

    rows: list[dict[str, object]] = []
    for _, row in records.iterrows():
        chrom = _normalize_chr(row["chrom_norm"])
        r_start = int(min(int(row["start"]), int(row["end"])))
        r_end = int(max(int(row["start"]), int(row["end"])))
        feature = str(row["feature"])
        strand = str(row["strand"])
        attr = row["attribute"]

        for sid, (bchrom, bstart, bend) in enumerate(bimrange_tuples):
            bchrom_norm = _normalize_chr(bchrom)
            if chrom != bchrom_norm:
                continue
            ov_start = max(r_start, int(bstart))
            ov_end = min(r_end, int(bend))
            if ov_end < ov_start:
                continue
            if use_segmented_x:
                seg = seg_by_key.get((bchrom_norm, int(sid)))
                if seg is None:
                    continue
                offset = float(seg["offset"])
                seg_start = int(seg["start"])
                x_start = offset + float(ov_start - seg_start)
                x_end = offset + float(ov_end - seg_start)
            else:
                x_start = float(ov_start)
                x_end = float(ov_end)
            rows.append(
                {
                    "feature": feature,
                    "strand": strand,
                    "attribute": attr,
                    "x_start": float(min(x_start, x_end)),
                    "x_end": float(max(x_start, x_end)),
                }
            )
    if len(rows) == 0:
        return pd.DataFrame(columns=out_cols)
    return pd.DataFrame(rows, columns=out_cols)


def _draw_gene_structure_axis(
    ax: plt.Axes,
    gene_df: pd.DataFrame,
    *,
    arrow_color: str = "black",
    block_color: str = "grey",
    line_width: float = 0.5,
    arrow_step: float = 1_000.0,
    thickness_scale: float = 1.0,
    y_offset: float = 0.0,
) -> None:
    """
    Draw gene/CDS/UTR structure into `ax` using projected x coordinates.
    """
    draw_gene_structure_records(
        ax,
        gene_df,
        arrow_color=arrow_color,
        block_color=block_color,
        line_width=line_width,
        arrow_step=arrow_step,
        gene_text_size=5,
        thickness_scale=thickness_scale,
        y_offset=y_offset,
        unknown_strand_as_plus=False,
        label_bbox={
            "facecolor": "white",
            "alpha": 0.55,
            "edgecolor": "none",
            "pad": 0.2,
        },
    )


def _draw_manh_gene_ld_links(
    fig: plt.Figure,
    ax_gene: plt.Axes,
    ax_manh: plt.Axes,
    ax_ld: plt.Axes,
    pairs: list[tuple[float, float, bool]],
    *,
    gene_cds_bottom: float = -0.05,
    force_line_color: Optional[str] = None,
) -> None:
    """
    Draw connectors:
      1) y=0.2 -> y=gene_cds_bottom vertical line at Manhattan SNP x
      2) y=gene_cds_bottom -> y=-0.2 diagonal line from gene x to LD-mapped x
    Significant SNP lines are red; others are black.
    """
    if len(pairs) == 0:
        return
    gx0, gx1 = ax_gene.get_xlim()
    tx0, tx1 = ax_manh.get_xlim()
    dt = float(tx1 - tx0)
    if np.isclose(dt, 0.0):
        return

    edge_margin_n = 0.004
    for x_top, x_ld, is_sig in pairs:
        if not (np.isfinite(x_top) and np.isfinite(x_ld)):
            continue
        # Map Manhattan x into gene-axis data coordinates.
        x_top_n = (float(x_top) - tx0) / dt
        if not np.isfinite(x_top_n):
            continue
        x_top_n = float(np.clip(x_top_n, edge_margin_n, 1.0 - edge_margin_n))
        x_top_g = gx0 + x_top_n * float(gx1 - gx0)
        if force_line_color is not None:
            line_color = str(force_line_color)
        else:
            line_color = "red" if bool(is_sig) else "black"
        ax_gene.plot(
            [x_top_g, x_top_g],
            [0.2, float(gene_cds_bottom)],
            color=line_color,
            linewidth=0.25,
            alpha=0.8,
            clip_on=False,
            zorder=-20,
        )
        # Draw diagonal segment in real cross-axes geometry so mapping
        # remains correct even if LD panel width is narrower than middle panel.
        diag = ConnectionPatch(
            xyA=(x_top_g, float(gene_cds_bottom)),
            xyB=(float(x_ld), 0.0),
            coordsA=ax_gene.transData,
            coordsB=ax_ld.transData,
            color=line_color,
            linewidth=0.25,
            alpha=0.8,
            clip_on=False,
            zorder=-20,
        )
        fig.add_artist(diag)


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
    max_logp: Optional[float] = None,
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
    if max_logp is not None:
        dfp = dfp[dfp["ylog"] <= float(max_logp)]
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
            alpha=0.85,
            edgecolors="none",
            linewidths=0.0,
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
    anno_suffix = (
        str(args.anno).replace(".gz", "").split(".")[-1].lower()
        if args.anno
        else None
    )
    gff_query_cache: Optional[GFFQuery] = None

    try:
        header_cols = pd.read_csv(file, sep="\t", nrows=0).columns.tolist()
    except Exception:
        header_cols = [chr_col, pos_col, p_col]
    lead_info_cols = [c for c in _LEAD_SNP_INFO_COLS if c in header_cols]
    read_cols = [chr_col, pos_col, p_col] + lead_info_cols
    read_cols = list(dict.fromkeys(read_cols))
    df_all = pd.read_csv(file, sep="\t", usecols=read_cols)
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
        plotmodel_qq = None
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
            if args.qq_ratio is not None:
                # QQ should keep all SNPs to preserve expected quantiles.
                plotmodel_qq = GWASPLOT(
                    df,
                    chr_col,
                    pos_col,
                    p_col,
                    0.1,
                    compression=False,
                )
        width_in = 8.0
        gene_panel_h_in = width_in / 20.0
        dpi = 300
        manh_ratio_for_font = float(args.manh_ratio) if args.manh_ratio is not None else 2.0
        manh_fontsize_target = _scaled_fontsize_for_manhattan(
            manh_ratio_for_font,
            width_in=width_in,
        )
        manh_loc_fontsize = max(3.0, float(manh_fontsize_target) * 0.85)
        manh_ylim = None
        manh_height_in = None
        manh_fontsize = None
        manh_yticks = None
        manh_xlim = None
        manh_axes_bounds = None
        hide_axis_labels = args.manh_ratio is not None and args.manh_ratio > 4.0
        default_manh_min = 0.0
        manh_min_logp = args.ylim_min if args.ylim_min is not None else default_manh_min
        manh_max_logp = args.ylim_max
        manh_ymin = manh_min_logp

        plot_colors = None
        two_color_style = _resolve_two_color_style(args.pallete_spec)
        if plotmodel is not None:
            plot_colors = _manhattan_colors_for_subset(
                args.pallete_spec,
                full_chr_labels,
                df[chr_col].drop_duplicates().tolist(),
            )
        qq_point_color = "black"
        qq_band_color = "grey"
        if two_color_style is not None:
            qq_point_color = str(two_color_style["gene_line_color"])   # dark
            qq_band_color = str(two_color_style["gene_block_color"])   # light

        def _draw_manhattan_axis(
            ax: plt.Axes,
        ) -> tuple[tuple[float, float], np.ndarray, float, tuple[float, float]]:
            if plotmodel is None:
                ax.text(0.5, 0.5, "No SNPs", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.xaxis.label.set_size(manh_fontsize_target)
                ax.yaxis.label.set_size(manh_fontsize_target)
                ax.tick_params(axis="both", labelsize=manh_fontsize_target)
                if hide_axis_labels:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                return (
                    ax.get_ylim(),
                    np.asarray(ax.get_yticks(), dtype=float),
                    float(manh_fontsize_target),
                    ax.get_xlim(),
                )

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
                        max_logp=manh_max_logp,
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
                        max_logp=manh_max_logp,
                    )
                else:
                    y_hl = -np.log10(
                        pd.to_numeric(
                            plotmodel.df.loc[df_hl_idx, "y"],
                            errors="coerce",
                        ).to_numpy(dtype=float)
                    )
                    keep_hl = np.isfinite(y_hl) & (y_hl >= float(manh_min_logp))
                    if manh_max_logp is not None:
                        keep_hl = keep_hl & (y_hl <= float(manh_max_logp))
                    draw_hl_idx = df_hl_idx[keep_hl]
                    draw_hl_y = y_hl[keep_hl]
                    ax.scatter(
                        plotmodel.df.loc[draw_hl_idx, "x"],
                        draw_hl_y,
                        marker="D",
                        color="red",
                        alpha=0.85,
                        zorder=10,
                        s=args.scatter_size,
                        edgecolors="none",
                        linewidths=0.0,
                    )
                    for idx in draw_hl_idx:
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
                        max_logp=manh_max_logp,
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
                        max_logp=manh_max_logp,
                        ignore=list(df_hl_idx),
                    )
            else:
                plotmodel.manhattan(
                    None,
                    ax=ax,
                    color_set=plot_colors,
                    min_logp=manh_min_logp,
                    max_logp=manh_max_logp,
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
                    max_logp=manh_max_logp,
                )

            if args.bimrange_tuples is not None:
                if len(bim_layout) > 1:
                    _apply_multi_bimrange_manhattan_axis(
                        ax,
                        bim_layout,
                        label_fontsize=manh_loc_fontsize,
                    )
                elif len(args.bimrange_tuples) == 1:
                    bchrom, bstart, bend = args.bimrange_tuples[0]
                    _apply_bimrange_manhattan_axis(ax, bchrom, bstart, bend)
                    _show_end_locs_without_xticks(
                        ax,
                        label_fontsize=manh_loc_fontsize,
                    )
            if args.ylim_min is not None or args.ylim_max is not None:
                _y0, _y1 = ax.get_ylim()
                lo = float(args.ylim_min) if args.ylim_min is not None else 0.0
                hi = float(args.ylim_max) if args.ylim_max is not None else float(_y1)
                if not (hi > lo):
                    hi = lo + max(1e-9, abs(lo) * 1e-9)
                ax.set_ylim(lo, hi)
            ax.xaxis.label.set_size(manh_fontsize_target)
            ax.yaxis.label.set_size(manh_fontsize_target)
            ax.tick_params(axis="both", labelsize=manh_fontsize_target)
            if hide_axis_labels:
                ax.set_xlabel("")
                ax.set_ylabel("")

            return (
                ax.get_ylim(),
                np.asarray(ax.get_yticks(), dtype=float),
                float(manh_fontsize_target),
                ax.get_xlim(),
            )

        # ----------------- Manhattan plot -----------------
        if args.manh_ratio is not None:
            manh_h_in = width_in / args.manh_ratio
            fig = plt.figure(
                figsize=(width_in, manh_h_in),
                dpi=dpi,
            )
            ax = fig.add_subplot(111)
            manh_ylim, manh_yticks, manh_fontsize, manh_xlim = _draw_manhattan_axis(ax)
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
            if plotmodel_qq is None:
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
                plotmodel_qq.qq(
                    ax=ax2,
                    color_set=[qq_point_color, qq_band_color],
                    line_color="black",
                    scatter_size=args.scatter_size,
                )
                qq_xmax = float(np.log10(plotmodel_qq.df.shape[0] + 1))
                if np.isfinite(qq_xmax) and qq_xmax > 0:
                    x_left = 0.0
                    x_right = float(qq_xmax)
                else:
                    x_left, x_right = ax2.get_xlim()
                    x_left = float(x_left)
                    x_right = float(x_right)
                if x_right <= x_left:
                    x_left, x_right = (0.0, max(1.0, x_right))
                x_pad = max(1e-9, 0.02 * float(x_right - x_left))
                ax2.set_xlim(x_left - x_pad, x_right + x_pad)
                if manh_ylim is not None:
                    ax2.set_ylim(manh_ylim)
                if manh_yticks is not None and manh_ylim is not None:
                    y0, y1 = ax2.get_ylim()
                    lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
                    yticks = [
                        float(t) for t in manh_yticks
                        if (t >= lo - 1e-12) and (t <= hi + 1e-12)
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
        gene_path = None
        if effective_ldblock_ratio is not None:
            ld_panel_xspan = args.ldblock_xspan
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
                    # Keep Manhattan->gene/LD mapping lines even when genotype lookup fails:
                    # fallback to GWAS-derived ld_site_keys if no matched genotype SNP is returned.
                    if len(sig_keys) > 0:
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

            region_ranges = args.bimrange_tuples if args.bimrange_tuples is not None else []
            region_layout = bim_layout if len(bim_layout) > 0 else _build_layout_from_bimrange_tuples(region_ranges)
            use_segmented_gene_x = len(region_layout) > 1
            gene_track_df = pd.DataFrame(
                columns=["feature", "strand", "attribute", "x_start", "x_end"]
            )
            use_gene_bridge = False
            if args.anno:
                if anno_suffix in {"gff", "gff3"} and gff_query_cache is None:
                    gff_query_cache = GFFQuery.from_file(args.anno)
                gene_raw = _load_gene_like_records_from_anno(
                    args.anno,
                    region_ranges,
                    logger,
                    gff_query=gff_query_cache,
                )
                gene_track_df = _project_gene_records_to_plot_x(
                    gene_raw,
                    region_ranges,
                    region_layout,
                    use_segmented_x=use_segmented_gene_x,
                )
                if gene_track_df.shape[0] == 0:
                    logger.warning(
                        "Warning: No gene-structure records found in selected --bimrange; "
                        "gene panel and gene-overlaid Manhattan+LD transition are skipped."
                    )
                else:
                    use_gene_bridge = True

            ld_cmap = "Greys"
            gene_block_color = "grey"
            gene_line_color = "black"
            if two_color_style is not None:
                ld_cmap = two_color_style["ld_cmap"]
                gene_block_color = str(two_color_style["gene_block_color"])
                gene_line_color = str(two_color_style["gene_line_color"])

            def _draw_ld_axis(ax: plt.Axes) -> None:
                LDblock(
                    ld_mat.copy(),
                    ax=ax,
                    vmin=0,
                    vmax=1,
                    cmap=ld_cmap,
                    rasterize_threshold=100,
                )
                if ld_overlay_text:
                    n_ld = int(ld_mat.shape[0])
                    ax.text(n_ld / 2.0, -n_ld / 2.0, ld_overlay_text, ha="center", va="center", fontsize=6)

            def _build_manh_ld_pairs(
                ax_manh: plt.Axes,
                ax_ld: plt.Axes,
            ) -> list[tuple[float, float, bool]]:
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
                key_to_meta: dict[tuple[str, int], tuple[float, bool]] = {}
                for cid, p, x, keep, pv in zip(chr_ids, pos_vals, x_vals, keep_mask, p_vals):
                    if not keep:
                        continue
                    chrom_norm = chr_id_to_norm.get(int(cid))
                    if chrom_norm is None:
                        continue
                    key = (chrom_norm, int(p))
                    is_sig = bool(np.isfinite(pv) and (pv <= threshold))
                    if key not in key_to_meta:
                        key_to_meta[key] = (float(x), is_sig)
                    else:
                        old_x, old_sig = key_to_meta[key]
                        # For duplicated SNP keys, keep x and merge significance with OR.
                        key_to_meta[key] = (old_x, bool(old_sig or is_sig))

                n_ld = int(ld_mat.shape[0])
                keys = ld_site_keys[:n_ld]
                pairs: list[tuple[float, float, bool]] = []
                for i, key in enumerate(keys):
                    meta = key_to_meta.get((str(key[0]), int(key[1])))
                    if meta is None:
                        continue
                    x_top, is_sig = meta
                    # In LDblock, SNP i is centered around x=i+0.5 on the top border.
                    x_ld = float(i) + 0.5
                    pairs.append((x_top, x_ld, bool(is_sig)))
                return pairs

            def _draw_bridge_axis(
                ax_bridge: plt.Axes,
                ax_manh: plt.Axes,
                ax_ld: plt.Axes,
                pairs: list[tuple[float, float]],
                *,
                line_color: str = "black",
            ) -> None:
                ax_bridge.set_xlim(0.0, 1.0)
                ax_bridge.set_ylim(0.0, 1.0)
                ax_bridge.set_xticks([])
                ax_bridge.set_yticks([])
                for spine in ax_bridge.spines.values():
                    spine.set_visible(False)
                if len(pairs) == 0:
                    return

                slashes_x: list[float] = []
                edge_margin_n = 0.006
                y_manh_ref = float(ax_manh.get_ylim()[0])
                y_ld_ref = 0.0
                to_bridge = ax_bridge.transAxes.inverted()
                for x_top, x_ld in pairs:
                    p_top_disp = ax_manh.transData.transform((float(x_top), y_manh_ref))
                    p_ld_disp = ax_ld.transData.transform((float(x_ld), y_ld_ref))
                    x_top_n = float(to_bridge.transform(p_top_disp)[0])
                    x_ld_n = float(to_bridge.transform(p_ld_disp)[0])
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
                        color=line_color,
                        linewidth=0.25,
                        alpha=0.8,
                        clip_on=False,
                        solid_joinstyle="round",
                    )
                    slashes_x.append(float(x_top))

            if use_segmented_gene_x and len(region_layout) > 0:
                gene_xlim = (
                    float(region_layout[0]["x_start"]),
                    float(region_layout[-1]["x_end"]),
                )
            elif len(region_ranges) > 0:
                gene_xlim = (
                    float(min(int(x[1]) for x in region_ranges)),
                    float(max(int(x[2]) for x in region_ranges)),
                )
            else:
                gene_xlim = (0.0, 1.0)
            gene_plot_xlim = manh_xlim if manh_xlim is not None else gene_xlim

            ld_h_in = width_in / effective_ldblock_ratio
            fig_ld = plt.figure(
                figsize=(width_in, ld_h_in),
                dpi=dpi,
            )
            ax_ld = fig_ld.add_subplot(111)
            ld_path = f"{args.out}/{args.prefix}.ldblock.{args.format}"
            _draw_ld_axis(ax_ld)

            fig_ld.subplots_adjust(
                left=0.08,
                right=0.98,
                top=0.98,
                bottom=0.08,
            )
            # Keep LD block x-axis drawable length consistent with Manhattan;
            # optionally constrain to user-specified x-span of Manhattan width.
            if manh_axes_bounds is not None or ld_panel_xspan is not None:
                _cx0, cy0, _cw, ch = ax_ld.get_position().bounds
                if manh_axes_bounds is not None:
                    mx0, _my0, mw, _mh = manh_axes_bounds
                    ref_x0 = float(mx0)
                    ref_w = float(mw)
                else:
                    ref_x0 = float(_cx0)
                    ref_w = float(_cw)
                if ld_panel_xspan is None:
                    fx0, fx1 = (0.0, 1.0)
                else:
                    fx0, fx1 = ld_panel_xspan
                new_x0 = ref_x0 + ref_w * float(fx0)
                new_w = ref_w * float(fx1 - fx0)
                ax_ld.set_position([new_x0, cy0, new_w, ch])
                ax_ld.set_anchor("N")
            fig_ld.savefig(ld_path, transparent=True)
            plt.close(fig_ld)

            if use_gene_bridge:
                gene_h_in = gene_panel_h_in
                fig_gene = plt.figure(
                    figsize=(width_in, gene_h_in),
                    dpi=dpi,
                )
                ax_gene = fig_gene.add_subplot(111)
                _draw_gene_structure_axis(
                    ax_gene,
                    gene_track_df,
                    arrow_color=gene_line_color,
                    block_color=gene_block_color,
                    line_width=1.0,
                    arrow_step=1_000.0,
                )
                ax_gene.set_xlim(gene_plot_xlim)
                fig_gene.tight_layout()
                if manh_axes_bounds is not None:
                    mx0, _my0, mw, _mh = manh_axes_bounds
                    _cx0, cy0, _cw, ch = ax_gene.get_position().bounds
                    ax_gene.set_position([mx0, cy0, mw, ch])
                gene_path = f"{args.out}/{args.prefix}.gene.{args.format}"
                fig_gene.savefig(gene_path, transparent=True)
                plt.close(fig_gene)
            else:
                gene_path = None

            # Combined Manhattan + LD panel
            manhld_manh_ratio = args.manh_ratio if args.manh_ratio is not None else 2.0
            manhld_manh_h_in = width_in / manhld_manh_ratio
            gene_bridge_scale = 0.5
            mid_gene_y_offset = 0.03
            mid_gene_ymin = -0.15
            mid_gene_ymax = 0.2
            mid_h_in = gene_panel_h_in if use_gene_bridge else 0.22
            # LD row height should follow actual LD panel width;
            # otherwise narrowed x-span leaves large vertical blanks.
            sub_left = 0.08
            sub_right = 0.98
            sub_top = 0.98
            sub_bottom = 0.07
            sub_hspace = 0.04
            manh_drawable_frac = float(sub_right - sub_left)
            if ld_panel_xspan is None:
                ld_panel_frac_in_manh = 1.0
            else:
                ld_panel_frac_in_manh = float(ld_panel_xspan[1] - ld_panel_xspan[0])
            # Compensate GridSpec/subplots spacing so LD axis keeps intended geometry
            # under LDblock's fixed aspect (isosceles right triangle).
            n_rows_combo = 3.0
            ld_row_layout_factor = (
                float(sub_top - sub_bottom) * n_rows_combo
            ) / (
                n_rows_combo + float(sub_hspace) * (n_rows_combo - 1.0)
            )
            ld_h_in_combo = (
                width_in
                * manh_drawable_frac
                * ld_panel_frac_in_manh
                / effective_ldblock_ratio
                / ld_row_layout_factor
            )
            ld_h_in_combo = max(0.5, float(ld_h_in_combo))
            manhld_total_h_in = manhld_manh_h_in + mid_h_in + ld_h_in_combo
            fig_manhld = plt.figure(
                figsize=(width_in, manhld_total_h_in),
                dpi=dpi,
            )
            gs_manhld = fig_manhld.add_gridspec(
                3,
                1,
                height_ratios=[manhld_manh_h_in, mid_h_in, ld_h_in_combo],
                hspace=0.04,
            )
            ax_manhld_top = fig_manhld.add_subplot(gs_manhld[0, 0])
            ax_manhld_mid = fig_manhld.add_subplot(gs_manhld[1, 0])
            ax_manhld_bot = fig_manhld.add_subplot(gs_manhld[2, 0])
            # Keep transition axis below Manhattan axis so loc labels remain visible.
            ax_manhld_top.set_zorder(5)
            ax_manhld_mid.set_zorder(2)
            ax_manhld_bot.set_zorder(1)
            if not use_gene_bridge:
                ax_manhld_mid.patch.set_visible(False)

            _tmp_ylim, _tmp_yticks, _tmp_fontsize, manhld_top_xlim = _draw_manhattan_axis(ax_manhld_top)
            _draw_ld_axis(ax_manhld_bot)
            if use_gene_bridge:
                _draw_gene_structure_axis(
                    ax_manhld_mid,
                    gene_track_df,
                    arrow_color=gene_line_color,
                    block_color=gene_block_color,
                    line_width=0.8,
                    arrow_step=1_000.0,
                    thickness_scale=gene_bridge_scale,
                    y_offset=mid_gene_y_offset,
                )
                ax_manhld_mid.set_xlim(manhld_top_xlim)
                ax_manhld_mid.set_ylim(mid_gene_ymin, mid_gene_ymax)
            fig_manhld.subplots_adjust(
                left=sub_left,
                right=sub_right,
                top=sub_top,
                bottom=sub_bottom,
                hspace=sub_hspace,
            )

            # Keep the two panels strictly left/right aligned.
            # Manh/Gene keep full width; optionally narrow only LD panel width.
            fig_manhld.canvas.draw()
            bx0, by0, bw, bh = ax_manhld_bot.get_position().bounds
            tx0, _ty0, tw, _th = ax_manhld_top.get_position().bounds
            if ld_panel_xspan is None:
                ld_panel_width_scale = 1.0
                new_bw = bw * float(ld_panel_width_scale)
                new_bx0 = bx0 + 0.5 * (bw - new_bw)
            else:
                fx0, fx1 = ld_panel_xspan
                new_bw = float(tw) * float(fx1 - fx0)
                new_bx0 = float(tx0) + float(tw) * float(fx0)
            ax_manhld_bot.set_position([new_bx0, by0, new_bw, bh])
            ax_manhld_bot.set_anchor("N")
            fig_manhld.canvas.draw()

            bridge_pairs = _build_manh_ld_pairs(ax_manhld_top, ax_manhld_bot)
            transition_line_color = "grey"
            if use_gene_bridge:
                ax_manhld_mid.set_xlim(ax_manhld_top.get_xlim())
                ax_manhld_mid.set_ylim(mid_gene_ymin, mid_gene_ymax)
                _draw_manh_gene_ld_links(
                    fig_manhld,
                    ax_manhld_mid,
                    ax_manhld_top,
                    ax_manhld_bot,
                    bridge_pairs,
                    gene_cds_bottom=-0.1 * gene_bridge_scale + mid_gene_y_offset,
                    force_line_color=transition_line_color,
                )
            else:
                bridge_pairs_plain = [(x1, x2) for x1, x2, _ in bridge_pairs]
                _draw_bridge_axis(
                    ax_manhld_mid,
                    ax_manhld_top,
                    ax_manhld_bot,
                    bridge_pairs_plain,
                    line_color="grey",
                )
            if not (args.bimrange_tuples is not None and len(args.bimrange_tuples) == 1):
                _show_end_locs_without_xticks(
                    ax_manhld_top,
                    label_fontsize=max(3.0, float(_tmp_fontsize) * 0.85),
                )

            manhld_path = f"{args.out}/{args.prefix}.manhld.{args.format}"
            fig_manhld.savefig(manhld_path, transparent=True)
            plt.close(fig_manhld)
        else:
            ld_path = None
            gene_path = None
            manhld_path = None

        saved_paths: list[tuple[str, str]] = []
        if manh_path is not None:
            saved_paths.append(("Manhattan", manh_path))
        if qq_path is not None:
            saved_paths.append(("QQ", qq_path))
        if ld_path is not None:
            saved_paths.append(("LD block", ld_path))
        if gene_path is not None:
            saved_paths.append(("Gene structure", gene_path))
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
            lead_cols_present = [c for c in _LEAD_SNP_INFO_COLS if c in df.columns]
            df_filter_raw = df.loc[
                df[p_col] <= threshold,
                [chr_col, pos_col, p_col] + lead_cols_present,
            ].copy()
            df_filter = df_filter_raw.set_index([chr_col, pos_col])

            if args.ldclump_window_bp is not None:
                n_before = int(df_filter_raw.shape[0])
                logger.info(
                    "Applying LD clump on threshold-passing SNPs for annotation: "
                    f"window={args.ldclump_window_bp} bp, r2>={args.ldclump_r2:g}"
                )
                df_filter, _clump_dict = _ldclump_significant_snps(
                    df_filter_raw,
                    chr_col=chr_col,
                    pos_col=pos_col,
                    p_col=p_col,
                    genofile=args.genofile,
                    window_bp=int(args.ldclump_window_bp),
                    r2_thr=float(args.ldclump_r2),
                    logger=logger,
                    show_progress=(len(args.gwasfile) == 1),
                )
                n_after = int(df_filter.shape[0])
                logger.info(
                    f"LD clump completed: kept {n_after}/{n_before} threshold-passing SNPs."
                )

            # Attach lead SNP info columns (allele0/allele1/maf/beta/se) by lead index.
            lead_map = df_filter_raw.loc[:, [chr_col, pos_col] + lead_cols_present].copy()
            if lead_map.shape[0] > 0:
                lead_map[chr_col] = lead_map[chr_col].astype(str)
                lead_map[pos_col] = pd.to_numeric(lead_map[pos_col], errors="coerce")
                lead_map = lead_map.dropna(subset=[pos_col]).copy()
                lead_map[pos_col] = lead_map[pos_col].astype(int)
                lead_map = lead_map.drop_duplicates(subset=[chr_col, pos_col], keep="first")
                lead_map = lead_map.set_index([chr_col, pos_col], drop=True)

                idx_chr = pd.Index(df_filter.index.get_level_values(0).astype(str))
                idx_pos = pd.to_numeric(
                    df_filter.index.get_level_values(1),
                    errors="coerce",
                ).fillna(0).astype(int)
                df_filter.index = pd.MultiIndex.from_arrays(
                    [idx_chr, idx_pos],
                    names=[chr_col, pos_col],
                )

                for col in lead_cols_present:
                    df_filter[col] = lead_map.reindex(df_filter.index)[col].values

            for col in _LEAD_SNP_INFO_COLS:
                if col not in df_filter.columns:
                    df_filter[col] = "NA"

            # Read annotation (GFF/bed) into unified annotation table
            # After readanno:
            #   anno[0] = chr
            #   anno[1] = start
            #   anno[2] = end
            #   anno[3] = gene ID
            #   anno[4], anno[5] = description fields
            if anno_suffix in {"gff", "gff3"} and gff_query_cache is not None:
                anno = readanno(args.anno, args.descItem, gff_data=gff_query_cache.gff)
            else:
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
                _format_gene_annotation_dict(x)
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
                    _format_gene_annotation_dict(x)
                    for x in desc_broad
                ]
            else:
                if "broaden" in df_filter.columns:
                    df_filter = df_filter.drop(columns=["broaden"])

            df_out = df_filter.reset_index()
            if pos_col in df_out.columns:
                df_out[pos_col] = pd.to_numeric(df_out[pos_col], errors="coerce").fillna(0).astype(int)
            if "start" in df_out.columns:
                df_out["start"] = pd.to_numeric(df_out["start"], errors="coerce").fillna(df_out[pos_col]).astype(int)
            if "end" in df_out.columns:
                df_out["end"] = pd.to_numeric(df_out["end"], errors="coerce").fillna(df_out[pos_col]).astype(int)
            if "nsnps" in df_out.columns:
                df_out["nsnps"] = pd.to_numeric(df_out["nsnps"], errors="coerce").fillna(0).astype(int)
            if "MeanR2" in df_out.columns:
                df_out["MeanR2"] = (
                    pd.to_numeric(df_out["MeanR2"], errors="coerce")
                    .fillna(0.0)
                    .map(lambda x: f"{float(x):.2f}")
                )

            # Output file is sorted by chromosome and position.
            if chr_col in df_out.columns and pos_col in df_out.columns:
                df_out["_chr_sort_key"] = df_out[chr_col].map(_chrom_sort_key)
                df_out = df_out.sort_values(
                    by=["_chr_sort_key", pos_col],
                    ascending=[True, True],
                    kind="mergesort",
                ).drop(columns=["_chr_sort_key"])

            col_order: list[str] = [chr_col, pos_col]
            for col in _LEAD_SNP_INFO_COLS:
                if col in df_out.columns:
                    col_order.append(col)
            if "start" in df_out.columns:
                col_order.append("start")
            if "end" in df_out.columns:
                col_order.append("end")
            if "nsnps" in df_out.columns:
                col_order.append("nsnps")
            if "MeanR2" in df_out.columns:
                col_order.append("MeanR2")
            if p_col in df_out.columns:
                col_order.append(p_col)
            if "desc" in df_out.columns:
                col_order.append("desc")
            if "broaden" in df_out.columns:
                col_order.append("broaden")
            remain_cols = [c for c in df_out.columns if c not in col_order and c != "LDclump"]
            if "LDclump" in df_out.columns:
                df_out = df_out[col_order + remain_cols + ["LDclump"]]
            else:
                df_out = df_out[col_order + remain_cols]

            logger.info(df_out)

            anno_path = f"{args.out}/{args.prefix}.{threshold}.anno.tsv"
            df_out.to_csv(anno_path, sep="\t", index=False)
            logger.info(f"Annotation table saved to {anno_path}")
            logger.info(f"Annotation completed in {round(time.time() - t_anno, 2)} seconds.\n")
        else:
            logger.info(f"Annotation file not found: {args.anno}\n")


def _read_merge_gwas_table(
    file: str,
    chr_col: str,
    pos_col: str,
    p_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    try:
        df = pd.read_csv(file, sep="\t", usecols=[chr_col, pos_col, p_col])
    except Exception as e:
        logger.error(
            f"Failed to read required columns from {file}: "
            f"{chr_col}, {pos_col}, {p_col}"
        )
        raise SystemExit(1) from e

    if df.shape[0] == 0:
        return df

    df = df.loc[:, [chr_col, pos_col, p_col]].copy()
    df[chr_col] = df[chr_col].astype(str)
    pos_num = pd.to_numeric(df[pos_col], errors="coerce")
    p_num = pd.to_numeric(df[p_col], errors="coerce")
    mask = (
        pos_num.notna()
        & np.isfinite(pos_num.to_numpy(dtype=float))
        & p_num.notna()
        & np.isfinite(p_num.to_numpy(dtype=float))
        & (p_num > 0.0)
    )
    df = df.loc[mask, [chr_col, pos_col, p_col]].copy()
    if df.shape[0] == 0:
        return df

    df[pos_col] = pd.to_numeric(df[pos_col], errors="coerce").astype(int)
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce").astype(float)
    return df


def _run_postgwas_merge_manhattan(args, logger: logging.Logger) -> None:
    files = [str(f) for f in args.merge_files]
    if len(files) < 2:
        logger.warning(
            "Warning: --merge requires at least two input GWAS files in total; "
            "falling back to normal postgwas flow."
        )
        _run_postgwas_tasks(args, logger)
        return

    chr_col, pos_col, p_col = args.chr, args.pos, args.pvalue
    logger.info("* Visualizing merged Manhattan plot...")

    series_colors = _resolve_merge_series_colors(args.pallete_spec, len(files))
    two_color_style = _resolve_two_color_style(args.pallete_spec)

    frames_raw: list[pd.DataFrame] = []
    chrom_sets: list[tuple[int, str, set[str]]] = []
    pvals_by_series: dict[int, np.ndarray] = {}
    for i, file in enumerate(files):
        df = _read_merge_gwas_table(file, chr_col, pos_col, p_col, logger)
        if df.shape[0] == 0:
            logger.warning(f"Warning: no valid SNP rows in merged file {i}: {file}; skipped.")
            continue

        chrom_set = set(df[chr_col].astype(str).tolist())
        chrom_sets.append((i, file, chrom_set))
        pvals = np.asarray(df[p_col], dtype=float)
        pvals = np.clip(pvals, np.nextafter(0.0, 1.0), np.inf)
        pvals_by_series[i] = pvals

        dfi = pd.DataFrame(
            {
                chr_col: df[chr_col].astype(str).to_numpy(),
                pos_col: np.asarray(df[pos_col], dtype=np.int64),
                p_col: pvals,
                "_series_idx": i,
            }
        )
        frames_raw.append(dfi)

    if len(frames_raw) == 0:
        logger.error("No valid SNPs available for merged plotting.")
        return

    if len(chrom_sets) >= 2:
        ref_i, ref_file, ref_chroms = chrom_sets[0]
        mismatch_items: list[tuple[int, str, int, int]] = []
        for i, file, chroms in chrom_sets[1:]:
            if chroms != ref_chroms:
                n_missing = int(len(ref_chroms - chroms))
                n_extra = int(len(chroms - ref_chroms))
                mismatch_items.append((i, file, n_missing, n_extra))
        if len(mismatch_items) > 0:
            ref_label = f"{ref_i}-{ref_file}"
            logger.warning(
                "Warning: Chromosome sets are inconsistent across merge GWAS files; "
                "merge mode is ignored and fallback to single-GWAS plotting."
            )
            for i, file, n_missing, n_extra in mismatch_items[:3]:
                logger.warning(
                    f"  File {i}-{file}: missing {n_missing} / extra {n_extra} chromosomes "
                    f"vs reference {ref_label}."
                )
            if len(mismatch_items) > 3:
                logger.warning(
                    f"  ... {len(mismatch_items) - 3} more mismatched files omitted."
                )
            fallback_file = str(args.gwasfile[0]) if len(args.gwasfile) > 0 else files[0]
            logger.info(f"Fallback single GWAS plotting file: {fallback_file}")
            GWASplot(fallback_file, args, logger)
            return

    plot_df = pd.concat(frames_raw, axis=0, ignore_index=True)
    bim_layout: list[dict[str, object]] = []
    if args.bimrange_tuples is not None:
        df_sel, seg_defs, n_before = _filter_df_by_bimranges(
            plot_df,
            chr_col,
            pos_col,
            args.bimrange_tuples,
            logger,
            "merge",
        )
        n_after = int(df_sel.shape[0])
        if n_after == 0:
            logger.warning(
                "No SNPs found in all bimrange settings for merged GWAS; merged Manhattan/LD/Gene plotting may be empty."
            )
            plot_df = df_sel
        else:
            plot_df = df_sel
            bim_layout = _build_bimrange_layout(seg_defs)
            logger.info(
                f"Applied {len(args.bimrange_tuples)} bimrange settings in merge mode: kept {n_after}/{n_before} SNPs."
            )

    threshold_merge = (
        args.threshold
        if args.threshold is not None
        else (0.05 / plot_df.shape[0] if plot_df.shape[0] > 0 else np.nan)
    )

    xticks: list[float] = []
    xticklabels: list[str] = []
    x_separators: list[float] = []
    use_segmented_layout = len(bim_layout) > 0

    if use_segmented_layout:
        plot_df = plot_df.copy()
        plot_df["_x"] = np.nan
        for seg in bim_layout:
            sid = int(seg["id"])
            start = int(seg["start"])
            offset = float(seg["offset"])
            length = float(seg["length"])
            mask = plot_df["__seg_id"].to_numpy(dtype=np.int64) == sid
            if not bool(np.any(mask)):
                continue
            posv = pd.to_numeric(plot_df.loc[mask, pos_col], errors="coerce").to_numpy(dtype=float)
            rel = np.clip(posv - float(start), 0.0, float(length))
            plot_df.loc[mask, "_x"] = offset + rel
        plot_df = plot_df[np.isfinite(pd.to_numeric(plot_df["_x"], errors="coerce"))].copy()
        xticks = [0.5 * (float(seg["x_start"]) + float(seg["x_end"])) for seg in bim_layout]
        xticklabels = [str(seg["label"]) for seg in bim_layout]
        for i in range(len(bim_layout) - 1):
            x_end = float(bim_layout[i]["x_end"])
            x_next = float(bim_layout[i + 1]["x_start"])
            x_separators.append(0.5 * (x_end + x_next))
    else:
        max_pos_by_chr: dict[str, int] = {}
        if plot_df.shape[0] > 0:
            chr_max = plot_df.groupby(chr_col)[pos_col].max()
            for chrom, max_pos in chr_max.items():
                max_pos_by_chr[str(chrom)] = int(max_pos)
        chrom_order = sorted(max_pos_by_chr.keys(), key=_chrom_sort_key)
        if len(chrom_order) == 0:
            logger.error("No chromosome labels available for merged Manhattan plotting.")
            return
        chr_lens = np.asarray(
            [max(1, int(max_pos_by_chr[c])) for c in chrom_order],
            dtype=float,
        )
        gap = int(max(1.0, float(np.nanmedian(chr_lens)) * 0.02))
        offsets: dict[str, int] = {}
        cursor = 0
        for i, chrom in enumerate(chrom_order):
            length = max(1, int(max_pos_by_chr[chrom]))
            offsets[chrom] = cursor
            xticks.append(float(cursor) + float(length) / 2.0)
            xticklabels.append(str(chrom))
            if i < len(chrom_order) - 1:
                x_separators.append(float(cursor) + float(length) + float(gap) / 2.0)
            cursor += length + gap
        plot_df = plot_df.copy()
        plot_df["_x"] = (
            plot_df[chr_col].astype(str).map(offsets).fillna(0).astype(float)
            + pd.to_numeric(plot_df[pos_col], errors="coerce").fillna(0.0).astype(float)
        )

    if plot_df.shape[0] > 0:
        pvals_draw = np.asarray(plot_df[p_col], dtype=float)
        pvals_draw = np.clip(pvals_draw, np.nextafter(0.0, 1.0), np.inf)
        plot_df["_ylog"] = -np.log10(pvals_draw)
    else:
        plot_df["_ylog"] = np.asarray([], dtype=float)

    width_in = 8.0
    # In merge mode, always rasterize dense scatter layers to keep output files
    # lightweight and easier to edit in vector tools (e.g. Adobe Illustrator).
    rasterized = True
    manh_path = None
    manh_height_in = None
    manh_fontsize: Optional[float] = None
    manh_ylim_pair: Optional[tuple[float, float]] = None
    manh_xlim_pair: Optional[tuple[float, float]] = None
    manh_axes_bounds: Optional[tuple[float, float, float, float]] = None

    if args.manh_ratio is not None:
        manh_ratio = float(args.manh_ratio)
        manh_fontsize = _scaled_fontsize_for_manhattan(
            manh_ratio,
            width_in=width_in,
        )
        manh_loc_fontsize = max(3.0, float(manh_fontsize) * 0.85)
        fig = plt.figure(figsize=(width_in, width_in / manh_ratio), dpi=300)
        ax = fig.add_subplot(111)
        draw_xmins: list[float] = []
        draw_xmaxs: list[float] = []

        for i, _file in enumerate(files):
            dfi = plot_df.loc[plot_df["_series_idx"] == i]
            if dfi.shape[0] == 0:
                continue
            yy = np.asarray(dfi["_ylog"], dtype=float)
            keep = np.isfinite(yy)
            if args.ylim_min is not None:
                keep = keep & (yy >= float(args.ylim_min))
            if args.ylim_max is not None:
                keep = keep & (yy <= float(args.ylim_max))
            if not bool(np.any(keep)):
                continue
            x_keep = np.asarray(dfi.loc[keep, "_x"], dtype=float)
            ax.scatter(
                x_keep,
                yy[keep],
                color=series_colors[i],
                alpha=0.5,
                s=args.scatter_size,
                label=str(i),
                edgecolors="none",
                linewidths=0.0,
                rasterized=rasterized,
            )
            if x_keep.size > 0:
                draw_xmins.append(float(np.nanmin(x_keep)))
                draw_xmaxs.append(float(np.nanmax(x_keep)))

        ax.set_xlabel("chrom")
        ax.set_ylabel("-log10(p)")
        if len(draw_xmins) > 0 and len(draw_xmaxs) > 0:
            xmin = float(np.nanmin(np.asarray(draw_xmins, dtype=float)))
            xmax = float(np.nanmax(np.asarray(draw_xmaxs, dtype=float)))
        else:
            xall = pd.to_numeric(plot_df["_x"], errors="coerce").to_numpy(dtype=float)
            xall = xall[np.isfinite(xall)]
            if xall.size > 0:
                xmin = float(np.nanmin(xall))
                xmax = float(np.nanmax(xall))
            else:
                xmin, xmax = (0.0, 1.0)
        if xmax > xmin:
            ax.set_xlim(xmin, xmax)
        else:
            eps = max(1e-9, abs(xmin) * 1e-9)
            ax.set_xlim(xmin - eps, xmax + eps)
        ax.margins(x=0.0)

        if use_segmented_layout and len(bim_layout) > 0:
            _apply_multi_bimrange_manhattan_axis(
                ax,
                bim_layout,
                label_fontsize=manh_loc_fontsize,
            )
        else:
            for xsep in x_separators:
                ax.axvline(
                    xsep,
                    ymin=0.0,
                    ymax=1.0 / 3.0,
                    linestyle="--",
                    color="lightgrey",
                    linewidth=0.6,
                    alpha=0.8,
                    zorder=8,
                )
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=0)
        ax.xaxis.label.set_size(manh_fontsize)
        ax.yaxis.label.set_size(manh_fontsize)
        ax.tick_params(axis="both", labelsize=manh_fontsize)

        ax.legend(
            loc="upper center",
            frameon=False,
            ncol=min(4, max(1, len(files))),
            markerscale=2.0,
            fontsize=manh_fontsize,
        )
        _y0, _y1 = ax.get_ylim()
        lo = float(args.ylim_min) if args.ylim_min is not None else 0.0
        hi = float(args.ylim_max) if args.ylim_max is not None else float(_y1)
        if not (hi > lo):
            hi = lo + max(1e-9, abs(lo) * 1e-9)
        ax.set_ylim(lo, hi)
        _yy0, _yy1 = ax.get_ylim()
        manh_ylim_pair = (float(_yy0), float(_yy1))
        manh_xlim_pair = (float(ax.get_xlim()[0]), float(ax.get_xlim()[1]))

        fig.tight_layout()
        manh_axes_bounds = ax.get_position().bounds
        manh_path = f"{args.out}/{args.prefix}.merge.manh.{args.format}".replace("//", "/")
        fig.savefig(manh_path, transparent=True)
        manh_height_in = fig.get_figheight()
        plt.close(fig)

    qq_path = None
    if args.qq_ratio is not None:
        qq_fontsize = float(manh_fontsize) if manh_fontsize is not None else 6.0
        qq_y_in = manh_height_in if manh_height_in is not None else 4.0
        qq_x_in = qq_y_in * float(args.qq_ratio)
        fig = plt.figure(figsize=(qq_x_in, qq_y_in), dpi=300)
        ax = fig.add_subplot(111)

        exp_xmin = np.inf
        exp_xmax = -np.inf
        for i, _file in enumerate(files):
            pvals = pvals_by_series.get(i)
            if pvals is None or pvals.size == 0:
                continue
            pvals = np.asarray(pvals, dtype=float)
            pvals = pvals[np.isfinite(pvals) & (pvals > 0.0)]
            if pvals.size == 0:
                continue
            pvals = np.sort(pvals)
            n = int(pvals.size)
            exp = -np.log10(np.arange(1, n + 1, dtype=float) / (n + 1.0))
            obs = -np.log10(pvals)
            if exp.size > 0:
                exp_xmin = min(exp_xmin, float(np.nanmin(exp)))
                exp_xmax = max(exp_xmax, float(np.nanmax(exp)))
            ax.scatter(
                exp,
                obs,
                s=args.scatter_size,
                alpha=0.5,
                rasterized=rasterized,
                color=series_colors[i],
                label=str(i),
                edgecolors="none",
                linewidths=0.0,
            )

        if not np.isfinite(exp_xmin) or not np.isfinite(exp_xmax):
            exp_xmin, exp_xmax = (0.0, 1.0)
        if exp_xmax > exp_xmin:
            x_pad = max(1e-9, 0.02 * float(exp_xmax - exp_xmin))
            ax.set_xlim(exp_xmin - x_pad, exp_xmax + x_pad)
        else:
            eps = max(1e-9, abs(exp_xmin) * 1e-9)
            x_pad = max(1e-9, 0.02 * float(eps))
            ax.set_xlim(exp_xmin - eps - x_pad, exp_xmax + eps + x_pad)
        ax.margins(x=0.0)
        if manh_ylim_pair is not None:
            ax.set_ylim(manh_ylim_pair)
        line_left, line_right = ax.get_xlim()
        ax.plot([line_left, line_right], [line_left, line_right], lw=1.0, color="black")
        ax.set_xlabel("Expected -log10(p-value)")
        ax.set_ylabel("Observed -log10(p-value)")
        ax.xaxis.label.set_size(qq_fontsize)
        ax.yaxis.label.set_size(qq_fontsize)
        ax.tick_params(axis="both", labelsize=qq_fontsize)
        ax.legend(
            loc="upper center",
            frameon=False,
            ncol=min(4, max(1, len(files))),
            markerscale=2.0,
            fontsize=qq_fontsize,
        )
        fig.tight_layout()
        qq_path = f"{args.out}/{args.prefix}.merge.qq.{args.format}".replace("//", "/")
        fig.savefig(qq_path, transparent=True)
        plt.close(fig)

    ld_path = None
    gene_path = None
    effective_ldblock_ratio = args.ldblock_ratio
    region_ranges = args.bimrange_tuples if args.bimrange_tuples is not None else []
    if effective_ldblock_ratio is not None:
        ld_use_all_sites = bool(args.ldblock_all is not None)
        ld_sites = _extract_ld_site_set(
            plot_df,
            chr_col,
            pos_col,
            p_col,
            threshold_merge,
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
                    region_ranges,
                    ld_sites,
                )
                missing_n = int(len(ld_sites) - len(set(sig_keys)))
                if missing_n > 0:
                    logger.warning(
                        f"Warning: {missing_n} requested LD sites are not in genotype data and were ignored."
                    )
                if len(sig_keys) > 0:
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
                    logger.info(f"Merge LD block built from {len(sig_keys)} {mode_text}.")

        ld_cmap = "Greys"
        gene_block_color = "grey"
        gene_line_color = "black"
        if two_color_style is not None:
            ld_cmap = two_color_style["ld_cmap"]
            gene_block_color = str(two_color_style["gene_block_color"])
            gene_line_color = str(two_color_style["gene_line_color"])

        ld_h_in = width_in / effective_ldblock_ratio
        fig_ld = plt.figure(figsize=(width_in, ld_h_in), dpi=300)
        ax_ld = fig_ld.add_subplot(111)
        LDblock(ld_mat.copy(), ax=ax_ld, vmin=0, vmax=1, cmap=ld_cmap, rasterize_threshold=100)
        if ld_overlay_text:
            n_ld = int(ld_mat.shape[0])
            ax_ld.text(n_ld / 2.0, -n_ld / 2.0, ld_overlay_text, ha="center", va="center", fontsize=6)
        fig_ld.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.08)
        if manh_axes_bounds is not None:
            _cx0, cy0, _cw, ch = ax_ld.get_position().bounds
            mx0, _my0, mw, _mh = manh_axes_bounds
            if args.ldblock_xspan is None:
                fx0, fx1 = (0.0, 1.0)
            else:
                fx0, fx1 = args.ldblock_xspan
            new_x0 = float(mx0) + float(mw) * float(fx0)
            new_w = float(mw) * float(fx1 - fx0)
            ax_ld.set_position([new_x0, cy0, new_w, ch])
            ax_ld.set_anchor("N")
        ld_path = f"{args.out}/{args.prefix}.merge.ldblock.{args.format}".replace("//", "/")
        fig_ld.savefig(ld_path, transparent=True)
        plt.close(fig_ld)

        if args.anno:
            if len(region_ranges) == 0:
                logger.warning(
                    "Warning: --anno in merge mode requires --bimrange for gene-structure plotting; gene plot skipped."
                )
            else:
                anno_suffix = str(args.anno).replace(".gz", "").split(".")[-1].lower()
                gff_query_cache: Optional[GFFQuery] = None
                if anno_suffix in {"gff", "gff3"}:
                    gff_query_cache = GFFQuery.from_file(args.anno)
                gene_raw = _load_gene_like_records_from_anno(
                    args.anno,
                    region_ranges,
                    logger,
                    gff_query=gff_query_cache,
                )
                region_layout = (
                    bim_layout
                    if len(bim_layout) > 0
                    else _build_layout_from_bimrange_tuples(region_ranges)
                )
                use_segmented_gene_x = len(region_layout) > 1
                gene_track_df = _project_gene_records_to_plot_x(
                    gene_raw,
                    region_ranges,
                    region_layout,
                    use_segmented_x=use_segmented_gene_x,
                )
                if gene_track_df.shape[0] == 0:
                    logger.warning(
                        "Warning: No gene-structure records found in selected --bimrange; gene plot is skipped."
                    )
                else:
                    if len(region_layout) > 0:
                        gene_xlim = (
                            float(region_layout[0]["x_start"]),
                            float(region_layout[-1]["x_end"]),
                        )
                    else:
                        gene_xlim = (
                            float(min(int(x[1]) for x in region_ranges)),
                            float(max(int(x[2]) for x in region_ranges)),
                        )
                    gene_plot_xlim = manh_xlim_pair if manh_xlim_pair is not None else gene_xlim

                    fig_gene = plt.figure(
                        figsize=(width_in, width_in / 20.0),
                        dpi=300,
                    )
                    ax_gene = fig_gene.add_subplot(111)
                    _draw_gene_structure_axis(
                        ax_gene,
                        gene_track_df,
                        arrow_color=gene_line_color,
                        block_color=gene_block_color,
                        line_width=1.0,
                        arrow_step=1_000.0,
                    )
                    ax_gene.set_xlim(gene_plot_xlim)
                    fig_gene.tight_layout()
                    if manh_axes_bounds is not None:
                        mx0, _my0, mw, _mh = manh_axes_bounds
                        _cx0, cy0, _cw, ch = ax_gene.get_position().bounds
                        ax_gene.set_position([mx0, cy0, mw, ch])
                    gene_path = f"{args.out}/{args.prefix}.merge.gene.{args.format}".replace("//", "/")
                    fig_gene.savefig(gene_path, transparent=True)
                    plt.close(fig_gene)

    saved_paths: list[tuple[str, str]] = []
    if manh_path is not None:
        saved_paths.append(("Merged Manhattan", manh_path))
    if qq_path is not None:
        saved_paths.append(("Merged QQ", qq_path))
    if ld_path is not None:
        saved_paths.append(("Merged LD block", ld_path))
    if gene_path is not None:
        saved_paths.append(("Merged Gene structure", gene_path))

    if len(saved_paths) == 0:
        logger.warning("Warning: no merged figure was generated (both --manh and --qq are off).")
    elif len(saved_paths) == 1:
        logger.info(f"{saved_paths[0][0]} plot saved to:\n  {saved_paths[0][1]}\n")
    else:
        title = ", ".join([x[0] for x in saved_paths])
        body = "\n".join([f"  {x[1]}" for x in saved_paths])
        logger.info(f"{title} plots saved to:\n{body}\n")


def _run_one_postgwas_task(file: str, args, logger: logging.Logger) -> str:
    GWASplot(file, args, logger)
    return str(file)


def _run_postgwas_tasks(args, logger: logging.Logger) -> None:
    files = [str(f) for f in args.gwasfile]
    if len(files) == 0:
        return
    if len(files) == 1:
        GWASplot(files[0], args, logger)
        return

    if _HAS_RICH_PROGRESS and sys.stdout.isatty():
        n_total = len(files)
        basenames = [os.path.basename(f) for f in files]
        name_width = max((len(x) for x in basenames), default=0)
        idx_width = len(str(n_total))
        progress = Progress(
            SpinnerColumn(
                spinner_name=get_rich_spinner_name(),
                style="cyan",
            ),
            TextColumn(
                "Task {task.fields[task_label]}: [bold]{task.fields[file_pad]}[/bold]{task.fields[suffix]}"
            ),
            transient=False,
        )
        with progress:
            task_map: dict[str, int] = {}
            for i, f in enumerate(files, start=1):
                task_map[f] = progress.add_task(
                    description="",
                    total=1,
                    task_label=f"{i:>{idx_width}}/{n_total}",
                    file_pad=os.path.basename(f).ljust(name_width),
                    suffix="",
                )
            try:
                done_iter = Parallel(
                    n_jobs=args.thread,
                    backend="loky",
                    return_as="generator_unordered",
                )(
                    delayed(_run_one_postgwas_task)(f, args, logger) for f in files
                )
                for done_file in done_iter:
                    tid = task_map.get(str(done_file))
                    if tid is not None:
                        progress.update(tid, advance=1, suffix=" ...Finished")
            except Exception:
                for f, tid in task_map.items():
                    task = progress.tasks[tid]
                    if not task.finished:
                        progress.update(tid, completed=1, suffix=" ...Failed")
                raise
        return

    if _HAS_TQDM:
        pbar = tqdm(
            total=len(files),
            desc="PostGWAS tasks",
            unit="file",
            leave=True,
            dynamic_ncols=True,
        )
        try:
            done_iter = Parallel(
                n_jobs=args.thread,
                backend="loky",
                return_as="generator_unordered",
            )(
                delayed(_run_one_postgwas_task)(f, args, logger) for f in files
            )
            for done_file in done_iter:
                pbar.update(1)
                pbar.set_postfix(file=os.path.basename(str(done_file)))
        finally:
            pbar.close()
        return

    Parallel(n_jobs=args.thread, backend="loky")(
        delayed(GWASplot)(file, args, logger) for file in files
    )


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
        "-merge", "--merge", nargs="+", action="append", type=str, default=None,
        help=(
            "Merge mode: add one or more GWAS files (option can be repeated) and plot all files in one Manhattan plot. "
            "Column names follow --chr/--pos/--pvalue."
        ),
    )
    optional_group.add_argument(
        "-LDclump", "--LDclump", dest="ldclump", nargs=2, default=None,
        metavar=("WINDOW", "R2"),
        help=(
            "Enable LD clumping for annotation output using threshold-passing SNPs only. "
            "Format: --LDclump <window> <r2>, e.g. --LDclump 500kb 0.8. "
            "Window supports kb/mb/bp (no unit defaults to kb). "
            "Requires genotype input via --bfile/--vcf/--file."
        ),
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
            "using only threshold-passing SNPs. Requires --bimrange. "
            "You can also pass x-span in Manhattan-width fraction, e.g. 0.2:0.8 or 0.2-0.8 "
            "(then ratio defaults to 2). "
            "If only ratio is given, x-span defaults to 0:1."
        ),
    )
    ldblock_group.add_argument(
        "-ldblock-all", "--ldblock-all", dest="ldblock_all", type=str, nargs="?", const="2", default=None,
        help=(
            "Enable LD block inverted triangle plotting with aspect ratio (width/height), "
            "using all SNPs in selected bimrange. Requires --bimrange. "
            "You can also pass x-span in Manhattan-width fraction, e.g. 0.2:0.8 or 0.2-0.8 "
            "(then ratio defaults to 2). "
            "If only ratio is given, x-span defaults to 0:1."
        ),
    )
    optional_group.add_argument(
        "-ylim", "--ylim", type=str, default=None,
        help=(
            "Y range for Manhattan as <max>, <min:max>, <min:>, or <:max> "
            "(also accepts '-' as separator), e.g. --ylim 6, --ylim 0:6, "
            "--ylim 2:, --ylim :6. Missing bound is auto-determined. "
            "Points outside this range are filtered before Manhattan scatter "
            "to speed plotting. QQ always keeps all points."
        ),
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
        "-qq", "--qq", type=str, nargs="?", const="on", default=None,
        help=(
            "Enable QQ plotting in auto mode. "
            "QQ width/height ratio is fixed at 5:4; user-provided ratio is ignored."
        ),
    )
    optional_group.add_argument(
        "-pallete", "--pallete", type=str, default=None,
        help=(
            "Manhattan color palette (QQ keeps black/grey). "
            "Supports cmap names (e.g. tab10, tab20) or ';'-separated colors "
            "(e.g. #1f77b4;#ff7f0e or (215,123,254);(1,1,1)). "
            "A single color is also accepted and auto-expanded to light/dark pair "
            "by grayscale direction for LD/gene (Manhattan keeps single-color). "
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
    try:
        if args.ylim is not None:
            args.ylim_min, args.ylim_max = _parse_ylim_spec(args.ylim)
        else:
            args.ylim_min, args.ylim_max = (None, None)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)

    try:
        args.pallete_spec = _parse_pallete_spec(args.pallete)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)

    try:
        args.manh_ratio = _parse_ratio(args.manh, "Manhattan") if args.manh is not None else None
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    if args.qq is None:
        args.qq_ratio = None
    else:
        qq_text = str(args.qq).strip().lower()
        if qq_text not in {"", "on"}:
            logger.info(
                f"QQ ratio input '{args.qq}' is ignored; fixed ratio 5:4 is used."
            )
        args.qq_ratio = float(_QQ_FIXED_RATIO)

    try:
        if args.ldblock_all is not None:
            args.ldblock_ratio, args.ldblock_xspan = _parse_ldblock_spec(
                args.ldblock_all, "LDBlock-all", logger
            )
            args.ldblock_mode = "all"
        elif args.ldblock is not None:
            args.ldblock_ratio, args.ldblock_xspan = _parse_ldblock_spec(
                args.ldblock, "LDBlock", logger
            )
            args.ldblock_mode = "threshold"
        else:
            args.ldblock_ratio = None
            args.ldblock_xspan = None
            args.ldblock_mode = None
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)

    try:
        if args.ldclump is not None:
            args.ldclump_window_bp, args.ldclump_r2 = _parse_ldclump_spec(args.ldclump)
        else:
            args.ldclump_window_bp = None
            args.ldclump_r2 = None
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
        args.ldblock_xspan = None
        args.ldblock_mode = None

    args.genofile = args.bfile or args.vcf or args.geno
    if args.ldblock_ratio is not None and args.genofile is None:
        logger.warning(
            "Warning: --ldblock/--ldblock-all enabled but no genotype file provided; zero-correlation LD block will be drawn."
        )
    if args.ldclump_window_bp is not None and args.genofile is None:
        logger.warning(
            "Warning: --LDclump enabled but no genotype file provided; LD clump is skipped."
        )
        args.ldclump_window_bp = None
        args.ldclump_r2 = None
    if args.ldclump_window_bp is not None and args.anno is None:
        logger.warning(
            "Warning: --LDclump only affects --anno output; --LDclump is ignored."
        )
        args.ldclump_window_bp = None
        args.ldclump_r2 = None
    merge_files_raw: list[str] = []
    if args.merge is not None:
        for item in args.merge:
            if isinstance(item, (list, tuple)):
                merge_files_raw.extend([str(x) for x in item])
            elif item is not None:
                merge_files_raw.append(str(item))

    args.merge_mode = bool(len(merge_files_raw) > 0)
    if args.merge_mode:
        args.merge_files = [str(x) for x in list(args.gwasfile)] + merge_files_raw
        if len(args.merge_files) < 2:
            logger.warning(
                "Warning: --merge requires at least two GWAS files in total; merge mode is disabled."
            )
            args.merge_mode = False
            args.merge_files = []
        else:
            if args.manh_ratio is None:
                logger.info("Merge mode detected; forcing --manh ratio to 2.")
                args.manh_ratio = 2.0
            if args.ldclump_window_bp is not None:
                logger.warning("Warning: --LDclump is ignored in --merge mode.")
                args.ldclump_window_bp = None
                args.ldclump_r2 = None
            if args.highlight is not None:
                logger.warning("Warning: --highlight is ignored in --merge mode.")
                args.highlight = None
    else:
        args.merge_files = []

    if (not args.merge_mode) and len(args.gwasfile) == 1 and int(args.thread) != 1:
        logger.info(
            "Single GWAS input detected; forcing --thread to 1."
        )
        args.thread = 1

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
    config_title = "JanusX - Post-GWAS"
    host_text = socket.gethostname()
    input_files = args.merge_files if args.merge_mode else args.gwasfile
    input_files_text = _format_input_files(input_files)
    threshold_text = str(args.threshold if args.threshold is not None else "0.05 / nSNP")
    bimrange_text = _format_bimrange_summary(args.bimrange_tuples)
    merge_map_rows = [(str(i), str(file)) for i, file in enumerate(args.merge_files)] if args.merge_mode else []
    output_prefix_text = f"{args.out}/{args.prefix}"
    threads_text = (
        f"{args.thread} "
        f"({'All cores' if args.thread == -1 else 'User-specified'})"
    )

    base_rows: list[tuple[str, str]] = [
        ("Input files", input_files_text),
        ("Chr|Pos|Pvalue", f"{args.chr}|{args.pos}|{args.pvalue}"),
        ("Genotype file", str(args.genofile)),
        ("Threshold", threshold_text),
        ("Bimrange", bimrange_text),
    ]

    vis_rows: Optional[list[tuple[str, str]]] = None
    if (
        args.manh_ratio is not None
        or args.qq_ratio is not None
        or args.ldblock_ratio is not None
    ):
        if args.merge_mode:
            if args.pallete_spec is None:
                merge_default_cmap = "tab10" if len(args.merge_files) <= 10 else "tab20"
                manh_pal_text = f"default ({merge_default_cmap})"
                qq_pal_text = f"default ({merge_default_cmap})"
            else:
                manh_pal_text = str(args.pallete)
                qq_pal_text = str(args.pallete)
        else:
            manh_pal_text = (
                "default (black/grey)" if args.pallete_spec is None else str(args.pallete)
            )
            qq_pal_text = "default (black/grey)"
        if args.disable_compression:
            if args.fullscatter and args.bimrange_tuples is not None:
                comp_text = "off (--fullscatter, auto for --bimrange)"
            elif args.fullscatter:
                comp_text = "off (--fullscatter)"
            else:
                comp_text = "off (auto for --bimrange)"
        else:
            comp_text = "on"
        manh_text = (
            f"ratio={args.manh_ratio if args.manh_ratio is not None else 'off'}, "
            f"pallete={manh_pal_text}, "
            f"ylim={args.ylim if args.ylim is not None else 'auto'}, "
            f"compression={comp_text}"
        )
        qq_text = f"auto, pallete={qq_pal_text}"
        vis_rows = [
            ("Manhattan", manh_text),
            ("QQ", qq_text),
            ("Scatter", f"size={args.scatter_size}, highlight={args.highlight}"),
            ("Format", str(args.format)),
        ]
        if args.ldblock_ratio is None:
            vis_rows.append(("LDBlock", "off"))
        else:
            ld_mode_text = "all SNPs" if args.ldblock_mode == "all" else "threshold SNPs"
            ld_xspan_text = (
                "full width (default)"
                if args.ldblock_xspan is None
                else f"{args.ldblock_xspan[0]:g}-{args.ldblock_xspan[1]:g} (fraction of Manhattan width)"
            )
            vis_rows.append(
                ("LDBlock", f"ratio={args.ldblock_ratio}, mode={ld_mode_text}, x-span={ld_xspan_text}")
            )
    anno_rows: Optional[list[tuple[str, str]]] = None
    if args.anno:
        anno_rows = [
            ("Anno file", str(args.anno)),
            ("Window (kb)", str(args.annobroaden)),
        ]
        if args.ldclump_window_bp is None:
            anno_rows.append(("LD clump", "off"))
        else:
            anno_rows.append(
                (
                    "LD clump",
                    f"on ({args.ldclump_window_bp / 1000.0:g} kb, r2>={args.ldclump_r2:g})",
                )
            )

    key_width = max(
        [len(k) for k, _ in base_rows]
        + ([len(k) for k, _ in vis_rows] if vis_rows is not None else [])
        + ([len(k) for k, _ in anno_rows] if anno_rows is not None else [])
    )
    key_width = max(8, int(key_width))

    def _fmt_kv(name: str, value: str, *, truncate: bool) -> str:
        pad = max(1, key_width - len(name))
        line = f"  {name}:{' ' * pad}{value}"
        return _truncate_config_line(line) if truncate else line

    divider_full = "*" * 60
    divider = "*" * _CONFIG_LINE_MAX_CHARS

    config_lines_full: list[str] = [
        divider_full,
        "POST-GWAS CONFIG",
        divider_full,
        "General:",
    ]
    config_lines_terminal: list[str] = [
        divider,
        "POST-GWAS CONFIG",
        divider,
        "General:",
    ]
    for name, value in base_rows:
        config_lines_full.append(_fmt_kv(str(name), str(value), truncate=False))
        config_lines_terminal.append(_fmt_kv(str(name), str(value), truncate=True))
    if len(merge_map_rows) > 0:
        config_lines_full.append("Merge trait-id mapping:")
        config_lines_terminal.append("Merge trait-id mapping:")
        for idx, file in merge_map_rows:
            config_lines_full.append(f"  {idx}-{file}")
            config_lines_terminal.append(_truncate_config_line(f"  {idx}-{file}"))
    if vis_rows is None:
        config_lines_full.append("Visualization: disabled")
        config_lines_terminal.append(_truncate_config_line("Visualization: disabled"))
    else:
        config_lines_full.append("Visualization:")
        config_lines_terminal.append("Visualization:")
        for name, value in vis_rows:
            config_lines_full.append(_fmt_kv(str(name), str(value), truncate=False))
            config_lines_terminal.append(_fmt_kv(str(name), str(value), truncate=True))
    if anno_rows is not None:
        config_lines_full.append("Annotation:")
        config_lines_terminal.append("Annotation:")
        for name, value in anno_rows:
            config_lines_full.append(_fmt_kv(str(name), str(value), truncate=False))
            config_lines_terminal.append(_fmt_kv(str(name), str(value), truncate=True))
    config_lines_full.append(f"Output prefix: {output_prefix_text}")
    config_lines_full.append(f"Threads:       {threads_text}")
    config_lines_full.append(divider_full + "\n")
    config_lines_terminal.append(_truncate_config_line(f"Output prefix: {output_prefix_text}"))
    config_lines_terminal.append(_truncate_config_line(f"Threads:       {threads_text}"))
    config_lines_terminal.append(divider + "\n")

    rich_rendered = _render_postgwas_config_rich(
        title=config_title,
        host=host_text,
        base_rows=base_rows,
        merge_map_rows=merge_map_rows,
        vis_rows=vis_rows,
        anno_rows=anno_rows,
        output_prefix=output_prefix_text,
        threads_text=threads_text,
        key_width=key_width,
    )
    if rich_rendered:
        _emit_info_to_file_handlers(logger, config_title)
        _emit_info_to_file_handlers(logger, f"Host: {host_text}\n")
        for line in config_lines_full:
            _emit_info_to_file_handlers(logger, line)
    else:
        _emit_info_to_stream_handlers(logger, config_title)
        _emit_info_to_stream_handlers(logger, f"Host: {host_text}\n")
        for line in config_lines_terminal:
            _emit_info_to_stream_handlers(logger, line)
        _emit_info_to_file_handlers(logger, config_title)
        _emit_info_to_file_handlers(logger, f"Host: {host_text}\n")
        for line in config_lines_full:
            _emit_info_to_file_handlers(logger, line)
    if no_plot_or_anno:
        logger.warning(
            "Warning: No --manh/--qq/--ldblock/--ldblock-all/--anno provided. Nothing will be plotted or annotated."
        )

    check_gwas_files = args.merge_files if args.merge_mode else args.gwasfile
    checks: list[bool] = [
        ensure_file_exists(logger, f, "GWAS result file") for f in check_gwas_files
    ]
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
    if args.merge_mode:
        _run_postgwas_merge_manhattan(args, logger)
    else:
        _run_postgwas_tasks(args, logger)

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
