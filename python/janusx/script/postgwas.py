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
    --out test --fmt pdf
  # Results will be saved as:
  #   test/result.assoc.manh.pdf
  #   test/result.assoc.qq.pdf

Citation
--------
  https://github.com/FJingxian/JanusX/
"""

import logging
import os
from ._common.cli_args import add_common_out_arg, add_common_prefix_arg, add_common_thread_arg
from ._common.log import setup_logging
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_file_input_site_metadata_exists,
    format_path_for_display,
    ensure_plink_prefix_exists,
)
from ._common.progress import (
    CliStatus,
    log_success,
    print_success,
    print_failure,
    print_warning,
    format_elapsed,
    should_animate_status,
    stdout_is_tty,
    warn_deprecated_alias_usage,
)
from ._common.progress import build_rich_progress, rich_progress_available
from ._common.genocache import configure_genotype_cache_from_out
from ._common.config_render import emit_cli_configuration
from ._common.threads import (
    apply_blas_thread_env,
    detect_effective_threads,
    format_requested_thread_usage,
    maybe_warn_non_openblas,
    require_openblas_by_default,
    runtime_thread_stage,
)

# Ensure matplotlib uses a non-interactive backend.
for key in ["MPLBACKEND"]:
    if key in os.environ:
        del os.environ[key]

import matplotlib as mpl
mpl.use("Agg")
from janusx.bioplotkit import GWASPLOT, LDblock, apply_integer_xticks, apply_integer_yticks
from janusx.bioplotkit.geneplot import draw_gene_structure_records
from janusx.gfreader import load_genotype_chunks, prepare_cli_input_cache

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.markers import MarkerStyle
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd
import numpy as np
from scipy.stats import beta
import argparse
import re
import shlex
import time
import socket
import sys
import colorsys
import concurrent.futures as cf
import multiprocessing as mp
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from typing import Any, Optional, Tuple
from janusx import janusx as jxrs
from janusx.gtools.reader import GFFQuery, bedreader, readanno
import warnings
from ._common.cjk import contains_cjk as _contains_cjk, ensure_cjk_font as _ensure_cjk_font

_LEAD_SNP_INFO_COLS = ["allele0", "allele1", "af", "maf", "beta", "se"]
_QQ_FIXED_RATIO = 5.0 / 4.0
_QQ_FAST_MAX_POINTS = 120_000
_QQ_BAND_MAX_POINTS = 20_000
_QQ_BAND_COLOR = "grey"
_CONFIG_LINE_MAX_CHARS = 60
_CONFIG_OVERFLOW_MARK = "***"
_ANNO_DESC_KEY = "description"
_DEFAULT_SINGLE_MARKER = "o"
_DEFAULT_MERGE_MARKERS = ("1", "2", "3", "4", "*", "+", "x")
_PANEL_WIDTH_IN = 8.0
_PANEL_LEFT_IN = 0.95
_PANEL_RIGHT_IN = 0.20
_PANEL_TOP_IN = 0.20
_PANEL_BOTTOM_IN = 0.65
_PANEL_LEGEND_RIGHT_IN = 2.25
_PANEL_STACK_VSPACE_IN = 0.10
_POSTGWAS_PDF_BACKEND_SENTINEL = object()
_POSTGWAS_PREFERRED_PDF_BACKEND: object = _POSTGWAS_PDF_BACKEND_SENTINEL

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore[assignment]
    _HAS_TQDM = False

def _postgwas_status_enabled(args) -> bool:
    return not bool(getattr(args, "_postgwas_worker_mute_stream", False))


def _postgwas_invocation_command(argv: Optional[list[str]] = None) -> str:
    tokens = [str(x) for x in (sys.argv[1:] if argv is None else argv)]
    prog_raw = str(sys.argv[0]).strip() if len(sys.argv) > 0 else ""
    if prog_raw == "":
        prog_raw = "jx postgwas"
    try:
        prog_parts = shlex.split(prog_raw)
    except Exception:
        prog_parts = [prog_raw]
    if len(prog_parts) == 0:
        prog_parts = ["jx", "postgwas"]
    return shlex.join([str(x) for x in (prog_parts + tokens)])


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _allow_windows_postgwas_process_pool() -> bool:
    """
    Windows matplotlib multi-process plotting has proven unstable in some
    environments (observed fail-fast 0xC0000409 crashes during ggval/postgwas).
    Keep a conservative serial default and allow explicit opt-in for further
    diagnostics.
    """
    if os.name != "nt":
        return True
    return _env_truthy("JANUSX_POSTGWAS_WINDOWS_PROCESS_POOL", False)


class _PostgwasFilePrefixFormatter(logging.Formatter):
    """Prefix warning/error records in worker append-mode log handlers."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if record.levelno >= logging.ERROR:
            return msg if msg.startswith("Error: ") else f"Error: {msg}"
        if record.levelno == logging.WARNING:
            return msg if msg.startswith("Warning: ") else f"Warning: {msg}"
        return msg


def _select_postgwas_mp_context():
    """
    Choose a multiprocessing start method for postgwas workers.

    Prefer spawn/forkserver to avoid Python 3.12+ warnings and potential
    deadlocks when forking from a multi-threaded parent after matplotlib /
    native runtimes have already been initialized. Allow manual override via
    JANUSX_POSTGWAS_MP_START_METHOD or the broader JANUSX_MP_START_METHOD.
    """
    try:
        methods = [str(m).strip().lower() for m in mp.get_all_start_methods()]
    except Exception:
        methods = []
    if len(methods) == 0:
        return None

    env_method = str(os.environ.get("JANUSX_POSTGWAS_MP_START_METHOD", "")).strip().lower()
    if env_method == "":
        env_method = str(os.environ.get("JANUSX_MP_START_METHOD", "")).strip().lower()
    if env_method != "" and env_method in methods:
        return mp.get_context(env_method)

    for name in ("spawn", "forkserver", "fork"):
        if name in methods:
            return mp.get_context(name)
    return None


def _build_postgwas_process_pool(max_workers: int) -> cf.ProcessPoolExecutor:
    kwargs: dict[str, Any] = {"max_workers": max(1, int(max_workers))}
    mp_ctx = _select_postgwas_mp_context()
    if mp_ctx is not None:
        kwargs["mp_context"] = mp_ctx
    return cf.ProcessPoolExecutor(**kwargs)


def _resolve_postgwas_worker_count(requested_threads: int, n_files: int) -> int:
    """
    Decide outer postgwas process count.

    Plotting workers are memory-heavy (pandas + matplotlib + optional LD/gene
    structures). Keep a conservative default cap, but allow explicit override.
    """
    req = max(1, int(requested_threads))
    total = max(1, int(n_files))

    env_raw = str(os.environ.get("JANUSX_POSTGWAS_MAX_WORKERS", "")).strip()
    if env_raw != "":
        try:
            env_cap = max(1, int(env_raw))
        except Exception:
            env_cap = None
        if env_cap is not None:
            return max(1, min(req, total, env_cap))

    default_cap = 4
    return max(1, min(req, total, default_cap))


def _log_postgwas_broken_pool_hint(
    logger: logging.Logger,
    *,
    n_workers: int,
    req_threads: int,
    n_files: int,
) -> None:
    mp_ctx = _select_postgwas_mp_context()
    method = ""
    try:
        if mp_ctx is not None:
            method = str(mp_ctx.get_start_method()).strip()
    except Exception:
        method = ""
    method_text = method if method != "" else "default"
    logger.error(
        "PostGWAS worker process exited unexpectedly. "
        f"Likely causes: native crash in matplotlib/numpy stack, OOM kill, or unsafe fork-style startup. "
        f"Current workers={int(n_workers)} (requested threads={int(req_threads)}, files={int(n_files)}), "
        f"mp_start={method_text}. "
        "Try rerunning with fewer outer workers, for example `-t 1` or "
        "`JANUSX_POSTGWAS_MAX_WORKERS=1`, and optionally force "
        "`JANUSX_POSTGWAS_MP_START_METHOD=spawn`."
    )


def _ensure_postgwas_worker_file_logging(args, logger: logging.Logger) -> logging.Logger:
    """
    Spawned workers do not inherit the parent's file handlers. Reattach an
    append-mode file handler so per-task logs are preserved in the main log.
    """
    log_path = str(getattr(args, "_postgwas_log_path", "")).strip()
    if log_path == "":
        return logger
    target = os.path.abspath(log_path)
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        try:
            base = os.path.abspath(str(handler.baseFilename))
        except Exception:
            base = ""
        if base == target:
            return logger
    try:
        file_handler = logging.FileHandler(target, mode="a", encoding="utf-8")
    except Exception:
        return logger
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(_PostgwasFilePrefixFormatter())
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logging.captureWarnings(True)
    return logger


def _sanitize_plot_text(text: object) -> str:
    s = str(text)
    if not _contains_cjk(s):
        return s
    if _ensure_cjk_font():
        return s
    # No CJK-capable font: fallback to ASCII to avoid glyph warnings.
    fallback = re.sub(r"[^\x00-\x7F]+", " ", s)
    fallback = re.sub(r"\s+", " ", fallback).strip()
    return fallback if fallback != "" else "NA"


def _strip_postgwas_input_suffix(path: object) -> str:
    name = os.path.basename(str(path).rstrip("/\\"))
    lower = name.lower()
    for ext in (".tsv.gz", ".txt.gz", ".csv.gz", ".tsv", ".txt", ".csv", ".gz"):
        if lower.endswith(ext):
            stem = name[: -len(ext)]
            return stem if stem != "" else name
    stem = os.path.splitext(name)[0]
    return stem if stem != "" else name


def _resolve_postgwas_output_stem(file: str, plot_prefix: object | None) -> str:
    base_stem = _strip_postgwas_input_suffix(file)
    prefix_text = str(plot_prefix).strip() if plot_prefix is not None else ""
    if prefix_text == "":
        return base_stem
    return f"{prefix_text}.{base_stem}"


def _prepare_cjk_plotting() -> None:
    # Try to enable a CJK font. If unavailable, silence glyph warnings and
    # fallback labels to ASCII where possible.
    if not _ensure_cjk_font():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"Glyph .* missing from font\(s\).*",
        )


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


def _detach_stream_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """
    Temporarily detach non-file handlers from logger.
    Used by parallel workers to prevent progress-line corruption on TTY.
    """
    removed: list[logging.Handler] = []
    try:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                continue
            try:
                logger.removeHandler(handler)
                removed.append(handler)
            except Exception:
                continue
    except Exception:
        return []
    return removed


def _restore_handlers(logger: logging.Logger, handlers: list[logging.Handler]) -> None:
    if len(handlers) == 0:
        return
    try:
        for handler in handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)
    except Exception:
        return


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


def _split_palette_tokens(text: str) -> list[str]:
    """
    Split palette string by ';' or ',' while preserving '(R,G,B)' tuples.
    """
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in str(text):
        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue
        if depth == 0 and ch in {";", ","}:
            tok = "".join(buf).strip()
            if tok != "":
                out.append(tok)
            buf = []
            continue
        buf.append(ch)
    tok = "".join(buf).strip()
    if tok != "":
        out.append(tok)
    return out


def _parse_custom_palette(text: str) -> list[str]:
    colors: list[str] = []
    for token in _split_palette_tokens(text):
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
                f"Invalid --palette color token: {tok}. "
                "Use #RRGGBB or (R,G,B)."
            ) from e
    if len(colors) == 0:
        raise ValueError("Invalid --palette: empty color list.")
    return colors


def _expand_single_palette_color(base_color: str) -> list[str]:
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


def _parse_palette_spec(value: object) -> Optional[Tuple[str, Any]]:
    """
    Parse --palette into:
      - ("cmap", "<matplotlib cmap name>")
      - ("list", ["#hex1", "#hex2", ...])
      - None (use default black/grey)
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        raise ValueError("Invalid --palette: value is empty.")
    if (";" in text) or ("," in text):
        toks = _split_palette_tokens(text)
        if len(toks) >= 2:
            return ("list", _parse_custom_palette(text))
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
                f"Invalid --palette: {text}. "
                "Use a cmap name (e.g. tab10) or ';' / ',' separated colors."
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
    Return two-color style for QQ:
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
            colors = _expand_single_palette_color(colors[0])
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


def _resolve_ldblock_style(
    spec: Optional[Tuple[str, Any]],
) -> Optional[dict[str, Any]]:
    """
    Resolve LD-block colormap + matched gene colors.
    Supports cmap names and custom color lists (2+ colors).
    """
    if spec is None:
        return None

    mode, payload = spec
    if mode == "list":
        colors = [mcolors.to_hex(mcolors.to_rgba(c)) for c in list(payload)]
        if len(colors) == 0:
            return None
        if len(colors) == 1:
            colors = _expand_single_palette_color(colors[0])
        ld_cmap = mcolors.LinearSegmentedColormap.from_list(
            "janusx_ld_custom",
            colors,
        )
    elif mode == "cmap":
        cmap = plt.get_cmap(str(payload))
        ld_cmap = cmap
        n_bins = int(getattr(cmap, "N", 256))
        if n_bins <= 32:
            colors = [
                mcolors.to_hex(cmap(i / max(1, n_bins - 1)))
                for i in range(max(2, n_bins))
            ]
        else:
            colors = [mcolors.to_hex(cmap(x)) for x in (0.0, 0.5, 1.0)]
    else:
        return None

    lums = np.asarray([_relative_luminance(c) for c in colors], dtype=float)
    i_light = int(np.argmax(lums))
    i_dark = int(np.argmin(lums))
    light = colors[i_light]
    dark = colors[i_dark]
    if _relative_luminance(light) >= 0.98:
        light = _blend_hex_color(light, dark, 0.2)

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


def _resolve_qq_point_color(spec: Optional[Tuple[str, Any]]) -> str:
    colors = _resolve_merge_series_colors(spec, 1)
    if len(colors) == 0:
        return "black"
    return str(colors[0])


def _parse_marker_spec(value: object) -> Optional[list[str]]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        raise ValueError("Invalid --marker: value is empty.")
    raw_tokens = [tok.strip() for tok in re.split(r"[;,]", text) if tok.strip() != ""]
    if len(raw_tokens) == 0:
        raise ValueError("Invalid --marker: no marker token found.")
    out: list[str] = []
    for tok in raw_tokens:
        try:
            MarkerStyle(tok)
        except Exception as e:
            raise ValueError(
                f"Invalid --marker token: {tok}. "
                "Examples: o, x, +, *, 1, 2, 3, 4."
            ) from e
        out.append(str(tok))
    return out


def _resolve_single_marker(spec: Optional[list[str]]) -> str:
    if spec is None or len(spec) == 0:
        return str(_DEFAULT_SINGLE_MARKER)
    return str(spec[0])


def _resolve_merge_markers(spec: Optional[list[str]], n_series: int) -> list[str]:
    if n_series <= 0:
        return []
    base = list(spec) if spec is not None and len(spec) > 0 else list(_DEFAULT_MERGE_MARKERS)
    return [str(base[i % len(base)]) for i in range(n_series)]


def _marker_scatter_style(marker: str) -> dict[str, object]:
    try:
        marker_obj = MarkerStyle(str(marker))
        is_filled = bool(marker_obj.is_filled())
    except Exception:
        is_filled = True
    if is_filled:
        return {
            "edgecolors": "none",
            "linewidths": 0.0,
        }
    return {
        "linewidths": 0.8,
    }


def _natural_tokens(text: str) -> tuple[tuple[int, object], ...]:
    tokens: list[tuple[int, object]] = []
    for part in re.split(r"(\d+)", text):
        if not part:
            continue
        if part.isdigit():
            tokens.append((0, int(part)))
        else:
            tokens.append((1, part.lower()))
    # Keep sort tokens hashable so they can be used safely in pandas sort keys.
    return tuple(tokens)


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
    Numeric interpretation:
      - default: Mb
      - if start/end look like bp-scale integers (>6 digits), they are
        interpreted as bp (and axis labels remain in Mb).
    """
    text = str(value).strip()
    m = re.match(r"^([^:]+):([0-9]*\.?[0-9]+)(?:-|:)([0-9]*\.?[0-9]+)$", text)
    if m is None:
        raise ValueError(
            f"Invalid --bimrange format: {value}. "
            "Use chr:start-end (or chr:start:end)."
        )
    chrom = m.group(1)
    start_raw = m.group(2).strip()
    end_raw = m.group(3).strip()
    start_num = float(start_raw)
    end_num = float(end_raw)

    def _looks_like_bp_integer(token: str) -> bool:
        tok = str(token).strip()
        if "." in tok:
            return False
        tok = tok.lstrip("0")
        if tok == "":
            tok = "0"
        return len(tok) > 6

    use_bp_input = _looks_like_bp_integer(start_raw) or _looks_like_bp_integer(end_raw)
    if use_bp_input:
        logger.warning(
            "Warning: --bimrange looks like bp coordinates (>6 digits); "
            "interpreting start/end as bp and showing axis in Mb."
        )
        start = int(round(start_num))
        end = int(round(end_num))
        if start < 0 or end < 0:
            raise ValueError("Invalid --bimrange: start/end must be >= 0 (bp).")
        if start > end:
            logger.warning(
                f"bimrange start > end ({start} > {end}); swapped to {end}-{start} bp."
            )
            start, end = end, start
    else:
        if start_num < 0 or end_num < 0:
            raise ValueError("Invalid --bimrange: start/end must be >= 0 (Mb).")
        if start_num > end_num:
            logger.warning(
                f"bimrange start > end ({start_num} > {end_num}); swapped to {end_num}-{start_num} Mb."
            )
            start_num, end_num = end_num, start_num
        start = int(round(start_num * 1_000_000))
        end = int(round(end_num * 1_000_000))
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
    chrom_max_map: dict[str, int] = {}
    valid_pos_mask = pos_num.notna() & np.isfinite(pos_num.to_numpy(dtype=float))
    if bool(valid_pos_mask.any()):
        chrom_pos_df = pd.DataFrame(
            {
                "chrom_norm": chr_norm.loc[valid_pos_mask].to_numpy(dtype=object),
                "pos_num": pos_num.loc[valid_pos_mask].astype(np.int64).to_numpy(),
            }
        )
        if chrom_pos_df.shape[0] > 0:
            for chrom_key, max_pos in chrom_pos_df.groupby("chrom_norm")["pos_num"].max().items():
                chrom_max_map[str(chrom_key)] = int(max_pos)
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
        display_end = int(end)
        chrom_max = chrom_max_map.get(str(target_chr))
        if chrom_max is not None:
            display_end = min(int(end), int(chrom_max))
        display_end = max(int(start), int(display_end))
        seg_defs.append(
            {
                "id": i,
                "chrom": str(chrom),
                "chrom_norm": target_chr,
                "start": int(start),
                "end": int(display_end),
                "query_end": int(end),
                "length": float(max(1, int(display_end) - int(start))),
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
    labels = [_sanitize_plot_text(seg["label"]) for seg in layout]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    loc_fontsize = 5.0 if label_fontsize is None else float(label_fontsize)
    for i in range(len(layout) - 1):
        x_end = float(layout[i]["x_end"])
        x_next = float(layout[i + 1]["x_start"])
        xb = 0.5 * (x_end + x_next)
        ax.axvline(x=xb, color="black", linestyle=":", linewidth=0.7, alpha=1)
        prev_end_lab = _sanitize_plot_text(
            f"{layout[i]['chrom']}:{int(layout[i]['end']) / 1_000_000:g}Mb"
        )
        next_start_lab = (
            f"{layout[i + 1]['chrom']}:{int(layout[i + 1]['start']) / 1_000_000:g}Mb"
        )
        next_start_lab = _sanitize_plot_text(next_start_lab)
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
    left_lab = _sanitize_plot_text(left_lab)
    right_lab = _sanitize_plot_text(right_lab)

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


def _normalize_plink_prefix(path_or_prefix: object) -> str:
    s = str(path_or_prefix).strip()
    low = s.lower()
    for ext in (".bed", ".bim", ".fam"):
        if low.endswith(ext):
            return s[: -len(ext)]
    return s


def _is_existing_plink_prefix(path_or_prefix: object) -> bool:
    prefix = _normalize_plink_prefix(path_or_prefix)
    if prefix == "":
        return False
    return (
        os.path.isfile(f"{prefix}.bed")
        and os.path.isfile(f"{prefix}.bim")
        and os.path.isfile(f"{prefix}.fam")
    )


def _build_bimrange_lookup(
    bimrange_tuples: list[tuple[str, int, int]],
) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for chrom, start, end in bimrange_tuples:
        ck = _normalize_chr(chrom)
        s = int(min(start, end))
        e = int(max(start, end))
        out.setdefault(ck, []).append((s, e))
    return out


def _site_in_bimrange_lookup(
    chrom_norm: str,
    pos: int,
    bim_lookup: dict[str, list[tuple[int, int]]],
) -> bool:
    spans = bim_lookup.get(str(chrom_norm), [])
    for s, e in spans:
        if int(s) <= int(pos) <= int(e):
            return True
    return False


def _expand_bimranges_for_reader(
    bimrange_tuples: list[tuple[str, int, int]],
) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for chrom, start, end in bimrange_tuples:
        s = int(min(start, end))
        e = int(max(start, end))
        raw = str(chrom).strip()
        norm = _normalize_chr(raw)
        candidates = [raw, norm]
        if norm != "":
            candidates.append(f"chr{norm}")
        for c in candidates:
            cc = str(c).strip()
            if cc == "":
                continue
            key = (cc, s, e)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _ld_r2_from_geno_rows(geno_rows: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(np.asarray(geno_rows, dtype=np.float32))
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    m = int(x.shape[0])
    if m == 1:
        return np.ones((1, 1), dtype=np.float32)

    x64 = np.asarray(x, dtype=np.float64)
    centered = x64 - x64.mean(axis=1, keepdims=True)
    gram = centered @ centered.T
    ss = np.einsum("ij,ij->i", centered, centered, dtype=np.float64, optimize=True)
    den = np.sqrt(np.outer(ss, ss))

    corr = np.zeros_like(gram, dtype=np.float64)
    valid = den > 0.0
    corr[valid] = gram[valid] / den[valid]
    corr = np.clip(corr, -1.0, 1.0)
    r2 = np.square(corr)
    np.fill_diagonal(r2, 1.0)
    r2 = np.nan_to_num(r2, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(r2.astype(np.float32, copy=False))


def _compute_ld_from_genotype_generic(
    genofile: str,
    bimrange_tuples: list[tuple[str, int, int]],
    *,
    selected_sites: Optional[set[tuple[str, int]]] = None,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    if len(bimrange_tuples) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    bim_lookup = _build_bimrange_lookup(bimrange_tuples)
    wanted: Optional[set[tuple[str, int]]] = None
    if selected_sites is not None:
        picked = {
            (_normalize_chr(c), int(p))
            for c, p in selected_sites
            if _site_in_bimrange_lookup(_normalize_chr(c), int(p), bim_lookup)
        }
        if len(picked) == 0:
            return np.zeros((0, 0), dtype=np.float32), []
        wanted = picked

    reader_ranges = _expand_bimranges_for_reader(bimrange_tuples)
    if len(reader_ranges) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    chunk_size = 20_000
    if wanted is not None:
        chunk_size = max(1, min(20_000, len(wanted)))

    geno_chunks: list[np.ndarray] = []
    keys: list[tuple[str, int]] = []
    for chunk, sites in load_genotype_chunks(
        str(genofile),
        chunk_size=chunk_size,
        maf=0.0,
        missing_rate=1.0,
        impute=True,
        ranges=reader_ranges,
    ):
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue

        keep_idx: list[int] = []
        keep_keys: list[tuple[str, int]] = []
        for ridx, site in enumerate(sites):
            chrom_norm = _normalize_chr(getattr(site, "chrom", ""))
            try:
                pos = int(getattr(site, "pos"))
            except Exception:
                continue
            if not _site_in_bimrange_lookup(chrom_norm, pos, bim_lookup):
                continue
            key = (chrom_norm, pos)
            if wanted is not None and key not in wanted:
                continue
            keep_idx.append(int(ridx))
            keep_keys.append(key)

        if len(keep_idx) == 0:
            continue
        if len(keep_idx) == int(arr.shape[0]):
            geno_chunks.append(arr)
        else:
            idx_arr = np.asarray(keep_idx, dtype=np.int64)
            geno_chunks.append(arr[idx_arr, :])
        keys.extend(keep_keys)

    if len(keys) == 0 or len(geno_chunks) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    x = geno_chunks[0] if len(geno_chunks) == 1 else np.vstack(geno_chunks)
    if len(keys) > 1:
        seen: set[tuple[str, int]] = set()
        keep_rows: list[int] = []
        for i, key in enumerate(keys):
            if key in seen:
                continue
            seen.add(key)
            keep_rows.append(i)
        if len(keep_rows) != len(keys):
            idx_arr = np.asarray(keep_rows, dtype=np.int64)
            x = x[idx_arr, :]
            keys = [keys[i] for i in keep_rows]

    ld_mat = _ld_r2_from_geno_rows(x)
    return ld_mat, keys


def _compute_ld_from_bed_rust(
    genofile: str,
    bimrange_tuples: list[tuple[str, int, int]],
    *,
    selected_sites: Optional[set[tuple[str, int]]] = None,
    threads: int = 0,
    logger: Optional[logging.Logger] = None,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """
    Compute LD r2 matrix with automatic backend routing.

    Routing:
      1) If an existing PLINK prefix can be resolved and Rust API is available,
         use Rust backend (BLAS-first, bitwise fallback).
      2) Otherwise, fallback to generic genotype reader + NumPy correlation.

    Returns:
      ld_r2 (float32, m x m), ld_keys [(chrom_norm, pos), ...] in matrix order.
    """
    if len(bimrange_tuples) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    source = str(genofile).strip()
    plink_prefix = _normalize_plink_prefix(source)
    if not _is_existing_plink_prefix(plink_prefix):
        plink_prefix = ""
        source_low = source.lower()
        delim = None
        if source_low.endswith(".csv") or os.path.isfile(f"{source}.csv"):
            delim = ","
        try:
            cached = prepare_cli_input_cache(
                source,
                snps_only=True,
                delimiter=delim,
                prefer_plink_for_txt=True,
                threads=int(threads),
            )
            cached_prefix = _normalize_plink_prefix(cached)
            if _is_existing_plink_prefix(cached_prefix):
                plink_prefix = cached_prefix
                if logger is not None and cached_prefix != source:
                    logger.info(
                        "LD backend: using cached PLINK prefix for Rust path: "
                        f"{format_path_for_display(cached_prefix)}"
                    )
        except Exception as ex:
            if logger is not None:
                logger.warning(
                    "Warning: failed to materialize PLINK cache for LD backend; "
                    f"falling back to generic path. Reason: {ex}"
                )
            plink_prefix = ""

    if plink_prefix != "" and hasattr(jxrs, "bed_ldblock_r2_rust"):
        chrom_ranges = [str(x[0]) for x in bimrange_tuples]
        start_bp = [int(x[1]) for x in bimrange_tuples]
        end_bp = [int(x[2]) for x in bimrange_tuples]

        kwargs: dict[str, object] = {}
        if selected_sites is not None:
            sel = sorted(
                [(_normalize_chr(c), int(p)) for (c, p) in selected_sites],
                key=lambda z: (str(z[0]), int(z[1])),
            )
            kwargs["selected_chrom"] = [str(x[0]) for x in sel]
            kwargs["selected_pos"] = [int(x[1]) for x in sel]

        ld_raw, chr_raw, pos_raw = jxrs.bed_ldblock_r2_rust(
            str(plink_prefix),
            chrom_ranges,
            start_bp,
            end_bp,
            threads=int(max(0, int(threads))),
            **kwargs,
        )
        ld_mat = np.ascontiguousarray(np.asarray(ld_raw, dtype=np.float32))
        ld_keys = [(_normalize_chr(c), int(p)) for c, p in zip(list(chr_raw), list(pos_raw))]
        if ld_mat.ndim != 2:
            return np.zeros((0, 0), dtype=np.float32), []
        if int(ld_mat.shape[0]) != int(ld_mat.shape[1]):
            raise RuntimeError(f"Rust LD matrix is not square: shape={ld_mat.shape}")
        if int(ld_mat.shape[0]) != int(len(ld_keys)):
            raise RuntimeError(
                f"Rust LD key count mismatch: matrix_n={ld_mat.shape[0]}, keys={len(ld_keys)}"
            )
        return ld_mat, ld_keys

    if logger is not None:
        logger.warning(
            "Warning: Rust LD matrix backend requires PLINK BED/BIM/FAM input; "
            "falling back to generic genotype LD computation."
        )
    return _compute_ld_from_genotype_generic(
        source,
        bimrange_tuples,
        selected_sites=selected_sites,
    )


def _format_bimrange_title(item: tuple[str, int, int]) -> str:
    chrom, start, end = item
    # return _sanitize_plot_text(f"{chrom}:{start / 1_000_000:g}-{end / 1_000_000:g}")
    return _sanitize_plot_text(f"")


def _ld_bimrange_spans(
    ld_keys: list[tuple[str, int]],
    bimrange_tuples: list[tuple[str, int, int]],
) -> list[dict[str, object]]:
    """
    Build LD index spans for each bimrange.
    x coordinate in LD panel:
      SNP i center -> x = i + 0.5
    """
    if len(ld_keys) == 0 or len(bimrange_tuples) == 0:
        return []

    spans_meta: list[dict[str, object]] = []
    for sid, (chrom, start, end) in enumerate(bimrange_tuples):
        spans_meta.append(
            {
                "sid": int(sid),
                "chrom_norm": _normalize_chr(chrom),
                "chrom": str(chrom),
                "start": int(start),
                "end": int(end),
                "first": None,
                "last": None,
                "count": 0,
            }
        )

    for idx, (k_chrom, k_pos) in enumerate(ld_keys):
        kk = str(k_chrom)
        pp = int(k_pos)
        hit_sid: Optional[int] = None
        for s in spans_meta:
            if kk != str(s["chrom_norm"]):
                continue
            if int(s["start"]) <= pp <= int(s["end"]):
                hit_sid = int(s["sid"])
                break
        if hit_sid is None:
            continue
        seg = spans_meta[hit_sid]
        if seg["first"] is None:
            seg["first"] = int(idx)
        seg["last"] = int(idx)
        seg["count"] = int(seg["count"]) + 1

    out: list[dict[str, object]] = []
    for s in spans_meta:
        if int(s["count"]) <= 0:
            continue
        first = int(s["first"])
        last = int(s["last"])
        x_start = float(first) + 0.5
        x_end = float(last) + 0.5
        out.append(
            {
                "sid": int(s["sid"]),
                "chrom": str(s["chrom"]),
                "start": int(s["start"]),
                "end": int(s["end"]),
                "count": int(s["count"]),
                "first": int(first),
                "last": int(last),
                "x_start": float(x_start),
                "x_end": float(x_end),
                "x_center": 0.5 * (float(x_start) + float(x_end)),
                "label": _format_bimrange_title(
                    (str(s["chrom"]), int(s["start"]), int(s["end"]))
                ),
            }
        )
    return out


def _project_gene_records_to_ld_spans(
    records: pd.DataFrame,
    bimrange_tuples: list[tuple[str, int, int]],
    ld_spans: list[dict[str, object]],
) -> pd.DataFrame:
    """
    Project gene records to LD-index x space so gene track aligns with LD triangle.
    """
    out_cols = ["feature", "strand", "attribute", "x_start", "x_end"]
    if records.shape[0] == 0 or len(bimrange_tuples) == 0 or len(ld_spans) == 0:
        return pd.DataFrame(columns=out_cols)

    span_by_sid: dict[int, dict[str, object]] = {
        int(s["sid"]): s for s in ld_spans if int(s.get("count", 0)) > 0
    }
    rows: list[dict[str, object]] = []
    for _, row in records.iterrows():
        chrom = _normalize_chr(row["chrom_norm"])
        r_start = int(min(int(row["start"]), int(row["end"])))
        r_end = int(max(int(row["start"]), int(row["end"])))
        feature = str(row["feature"])
        strand = str(row["strand"])
        attr = row["attribute"]
        for sid, (bchrom, bstart, bend) in enumerate(bimrange_tuples):
            if chrom != _normalize_chr(bchrom):
                continue
            ov_start = max(r_start, int(bstart))
            ov_end = min(r_end, int(bend))
            if ov_end < ov_start:
                continue
            span = span_by_sid.get(int(sid))
            if span is None:
                continue

            seg_start_bp = int(bstart)
            seg_end_bp = int(bend)
            seg_len_bp = max(1, int(seg_end_bp - seg_start_bp))
            x0 = float(span["x_start"])
            x1 = float(span["x_end"])
            # Keep one-SNP segments drawable.
            if np.isclose(x0, x1):
                x0 -= 0.45
                x1 += 0.45
            scale = (x1 - x0) / float(seg_len_bp)
            gx0 = x0 + float(ov_start - seg_start_bp) * scale
            gx1 = x0 + float(ov_end - seg_start_bp) * scale
            rows.append(
                {
                    "feature": feature,
                    "strand": strand,
                    "attribute": attr,
                    "x_start": float(min(gx0, gx1)),
                    "x_end": float(max(gx0, gx1)),
                }
            )
    if len(rows) == 0:
        return pd.DataFrame(columns=out_cols)
    return pd.DataFrame(rows, columns=out_cols)


def _draw_ld_bimrange_titles(
    ax: plt.Axes,
    spans: list[dict[str, object]],
    *,
    enabled: bool,
    font_size: float = 5.5,
) -> None:
    if not enabled or len(spans) == 0:
        return
    trans = ax.get_xaxis_transform()
    for i, seg in enumerate(spans):
        ax.text(
            float(seg["x_center"]),
            1.015,
            _sanitize_plot_text(seg["label"]),
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=float(font_size),
            clip_on=False,
            zorder=40,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 0.2,
            },
        )
        if i > 0:
            prev = spans[i - 1]
            xb = 0.5 * (float(prev["x_end"]) + float(seg["x_start"]))
            ax.axvline(
                x=float(xb),
                color="black",
                linestyle=":",
                linewidth=0.7,
                alpha=0.75,
                zorder=5,
            )


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
    # Normalize simple serialized list-like wrappers, e.g. "['geneA']".
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if inner != "" and "," not in inner:
            text = inner
    if len(text) >= 2 and ((text[0] == "'" and text[-1] == "'") or (text[0] == '"' and text[-1] == '"')):
        text = text[1:-1].strip()
    text = re.sub(r"\s+", " ", text)
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
      gene1:description/additionaldesc;gene2:description/additionaldesc
    """
    if hits is None or hits.shape[0] == 0:
        return "NA"
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
    if len(out) == 0:
        return "NA"
    return ";".join(
        [f"{gene}:{vals[0]}/{vals[1]}" for gene, vals in out.items()]
    )


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
    animate_progress = bool(show_progress and should_animate_status("LD clumping..."))
    use_rich_progress = bool(animate_progress and rich_progress_available())
    progress = None
    task_id = None
    progress_tqdm = None
    clump_start_ts = time.monotonic()
    clump_success = False
    if use_rich_progress:
        try:
            progress = build_rich_progress(
                show_remaining=True,
                finished_text=" ",
                transient=True,
            )
        except Exception:
            progress = None
            use_rich_progress = False

    with (progress if progress is not None else nullcontext()):
        if progress is not None:
            task_id = progress.add_task("LD clumping...", total=int(work.shape[0]))
        elif bool(animate_progress and _HAS_TQDM and stdout_is_tty()):
            progress_tqdm = tqdm(
                total=int(work.shape[0]),
                desc="LD clumping",
                unit="snp",
                leave=False,
                dynamic_ncols=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                           "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
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
        clump_success = True
    clump_elapsed = format_elapsed(time.monotonic() - clump_start_ts)
    if bool(show_progress):
        if clump_success:
            print_success(f"LD clumping ...Finished [{clump_elapsed}]", force_color=True)
        else:
            print_failure(f"LD clumping ...Failed [{clump_elapsed}]")

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
    # Keep LD triangle body filling the whole panel width.
    ax.set_xlim(0.5, float(n) - 0.5)
    ax.margins(x=0.0)
    if text:
        ax.text(n / 2.0, -n / 2.0, text, ha="center", va="center", fontsize=6)


def _ld_min_height_over_width(n_sites: int) -> float:
    """
    Minimal LD panel height/width ratio that keeps LDblock (aspect=0.5)
    from shrinking panel width for small SNP counts under xlim=0.5..n-0.5.
    """
    n = max(2, int(n_sites))
    return 0.5 * (float(n) / float(n - 1))


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
    gene_df_plot = gene_df.copy()
    if "attribute" in gene_df_plot.columns:
        gene_df_plot["attribute"] = gene_df_plot["attribute"].map(_sanitize_plot_text)

    draw_gene_structure_records(
        ax,
        gene_df_plot,
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
    nonsig_line_color: str = "grey",
) -> None:
    """
    Draw connectors:
      1) y=0.2 -> y=gene_cds_bottom vertical line at Manhattan SNP x
      2) y=gene_cds_bottom -> y=-0.2 diagonal line from gene x to LD-mapped x
    Significant SNP lines are red; non-significant SNP lines use
    `nonsig_line_color`.
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
            line_color = "red" if bool(is_sig) else str(nonsig_line_color)
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
    if len(files) == 0:
        return "0 files"
    if len(files) == 1:
        return str(files[0])
    return f"{len(files)} files"


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
    marker: str,
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
    dfp["ylog"] = _safe_neglog10_p(dfp["y"])
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
            marker=str(marker),
            s=float(base_size) * 1.5,
            alpha=1,
            rasterized=rasterized,
            zorder=6,
            **_marker_scatter_style(str(marker)),
        )
    ax.axhline(
        y=thr_log,
        linestyle="dashed",
        color="grey",
        linewidth=1.0,
    )


def _safe_neglog10_p(values: object) -> np.ndarray:
    """
    Safe -log10 transform for p-values:
    - coerce non-numeric to NaN
    - replace non-finite with 1.0
    - clamp to (0, 1]
    """
    p = pd.to_numeric(values, errors="coerce")
    if isinstance(p, pd.Series):
        arr = p.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(p, dtype=float)
    arr = np.array(arr, dtype=float, copy=True)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr[~np.isfinite(arr)] = 1.0
    arr = np.clip(arr, np.nextafter(0.0, 1.0), 1.0)
    return -np.log10(arr)


def _postgwas_output_format_from_path(path: str) -> str:
    _, ext = os.path.splitext(str(path))
    return ext.lstrip(".").strip().lower()


def _postgwas_should_rasterize_dense_layers(output_format: object) -> bool:
    fmt = str(output_format).strip().lower()
    # Single-file PDF stays vector-friendly by default; merge mode overrides
    # this per artist and rasterizes dense layers explicitly.
    return fmt != "pdf"


def _postgwas_savefig_kwargs(output_format: object) -> dict[str, object]:
    fmt = str(output_format).strip().lower()
    return {
        "transparent": False,
        "facecolor": "white",
        "edgecolor": "white",
    }


def _postgwas_resolve_pdf_backend() -> Optional[str]:
    global _POSTGWAS_PREFERRED_PDF_BACKEND
    cached = _POSTGWAS_PREFERRED_PDF_BACKEND
    if cached is not _POSTGWAS_PDF_BACKEND_SENTINEL:
        return cached if isinstance(cached, str) else None
    backend_name: Optional[str]
    try:
        from matplotlib.backends import backend_cairo  # noqa: F401
    except Exception:
        backend_name = None
    else:
        backend_name = "cairo"
    _POSTGWAS_PREFERRED_PDF_BACKEND = backend_name
    return backend_name


def _qq_select_points_with_threshold(
    pvals: np.ndarray,
    *,
    sig_p_threshold: Optional[float],
    max_points: int = _QQ_FAST_MAX_POINTS,
    keep_all: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select QQ scatter points with deterministic down-sampling:
    - always keep all points with p <= sig_p_threshold
    - for remaining points, keep an evenly spaced rank grid up to max_points
    """
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p) & (p > 0.0)]
    if p.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    p = np.clip(p, np.nextafter(0.0, 1.0), 1.0)
    p_sorted = np.sort(p, kind="mergesort")
    n = int(p_sorted.size)

    if sig_p_threshold is None or (not np.isfinite(sig_p_threshold)):
        sig_thr = 1.0 / float(max(1, n))
    else:
        sig_thr = float(sig_p_threshold)
    sig_thr = float(np.clip(sig_thr, np.nextafter(0.0, 1.0), 1.0))

    if keep_all or n <= int(max_points):
        draw_idx = np.arange(n, dtype=np.int64)
    else:
        base_idx = np.linspace(0, n - 1, int(max_points), dtype=np.int64)
        sig_n = int(np.searchsorted(p_sorted, sig_thr, side="right"))
        if sig_n > 0:
            sig_idx = np.arange(sig_n, dtype=np.int64)
            draw_idx = np.unique(np.concatenate([base_idx, sig_idx]))
        else:
            draw_idx = np.unique(base_idx)

    ranks = draw_idx.astype(float) + 1.0
    exp = -np.log10(ranks / (n + 1.0))
    obs = -np.log10(p_sorted[draw_idx])
    keep = np.isfinite(exp) & np.isfinite(obs)
    return exp[keep], obs[keep]


def _qq_confidence_band_from_n(
    n_points: int,
    *,
    ci: int = 95,
    max_points: Optional[int] = _QQ_BAND_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    band_n = int(max(0, n_points))
    if band_n <= 0:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )
    if max_points is not None and band_n > int(max_points):
        n_band = max(2, int(max_points))
        ranks = np.unique(
            np.round(np.geomspace(1.0, float(band_n), num=n_band)).astype(np.int64)
        )
        if ranks[0] != 1:
            ranks = np.insert(ranks, 0, 1)
        if ranks[-1] != band_n:
            ranks = np.append(ranks, band_n)
    else:
        ranks = np.arange(1, band_n + 1, dtype=np.int64)
    ranks = np.sort(ranks)[::-1]
    ci_frac = float(ci)
    if ci_frac > 1.0:
        ci_frac /= 100.0
    ci_frac = float(np.clip(ci_frac, 1e-12, 1.0 - 1e-12))
    alpha = 1.0 - ci_frac
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0
    x_band = -np.log10(ranks.astype(np.float64) / (band_n + 1.0))
    p_upper = beta.ppf(q_hi, ranks, band_n - ranks + 1)
    p_lower = beta.ppf(q_lo, ranks, band_n - ranks + 1)
    lower = -np.log10(p_upper)
    upper = -np.log10(p_lower)
    keep = np.isfinite(x_band) & np.isfinite(lower) & np.isfinite(upper)
    return x_band[keep], lower[keep], upper[keep]


def _resolve_qq_ylim(
    ax: plt.Axes,
    *,
    lower: float,
    upper: Optional[float],
) -> tuple[float, float]:
    lo = float(lower)
    _y0, _y1 = ax.get_ylim()
    if upper is not None and np.isfinite(float(upper)):
        hi = float(upper)
    else:
        hi = float(_y1)
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + max(1.0, abs(lo) * 0.1, 1e-9)
    return lo, hi


def _apply_qq_axes(
    ax: plt.Axes,
    *,
    y_lower: float,
    y_upper: float,
    x_right: float,
    y_ticks: Optional[np.ndarray] = None,
) -> None:
    lo = float(y_lower)
    hi = float(y_upper)
    xr = float(x_right)
    if not np.isfinite(xr) or xr <= lo:
        xr = lo + 1.0
    x_pad = max(1e-9, 0.02 * float(max(1e-9, xr - lo)))
    x_upper = xr + x_pad
    if x_upper <= lo:
        x_upper = lo + 1.0
    ax.set_ylim(lo, hi)
    ax.set_xlim(lo, x_upper)
    ax.margins(x=0.0)
    if y_ticks is not None:
        ticks = np.asarray(y_ticks, dtype=float)
        keep = np.isfinite(ticks) & (ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)
        kept = ticks[keep]
        if kept.size > 0:
            apply_integer_yticks(ax, ticks=kept)
            return
    apply_integer_yticks(ax)


def _create_ratio_panel_figure(
    *,
    ratio: float,
    dpi: int,
    panel_width_in: Optional[float] = None,
    panel_height_in: Optional[float] = None,
    reserve_right_in: float = 0.0,
    left_in: float = _PANEL_LEFT_IN,
    right_in: float = _PANEL_RIGHT_IN,
    top_in: float = _PANEL_TOP_IN,
    bottom_in: float = _PANEL_BOTTOM_IN,
) -> tuple[plt.Figure, plt.Axes, float, float]:
    use_ratio = max(0.2, float(ratio))
    if panel_width_in is None and panel_height_in is None:
        panel_width_in = float(_PANEL_WIDTH_IN)
    if panel_width_in is None:
        plot_h = max(1e-6, float(panel_height_in))
        plot_w = plot_h * use_ratio
    elif panel_height_in is None:
        plot_w = max(1e-6, float(panel_width_in))
        plot_h = plot_w / use_ratio
    else:
        plot_w = max(1e-6, float(panel_width_in))
        plot_h = max(1e-6, float(panel_height_in))
    fig_w = float(left_in) + float(plot_w) + float(right_in) + float(reserve_right_in)
    fig_h = float(bottom_in) + float(plot_h) + float(top_in)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(dpi))
    ax = fig.add_axes(
        [
            float(left_in) / float(fig_w),
            float(bottom_in) / float(fig_h),
            float(plot_w) / float(fig_w),
            float(plot_h) / float(fig_h),
        ]
    )
    try:
        ax.set_box_aspect(float(plot_h) / float(plot_w))
    except Exception:
        pass
    return fig, ax, float(plot_w), float(plot_h)


def _create_stacked_panel_figure(
    *,
    panel_width_in: float,
    panel_heights_in: list[float],
    dpi: int,
    reserve_right_in: float = 0.0,
    left_in: float = _PANEL_LEFT_IN,
    right_in: float = _PANEL_RIGHT_IN,
    top_in: float = _PANEL_TOP_IN,
    bottom_in: float = _PANEL_BOTTOM_IN,
    vspace_in: float = _PANEL_STACK_VSPACE_IN,
) -> tuple[plt.Figure, list[plt.Axes], float, list[float]]:
    heights = [max(1e-6, float(h)) for h in panel_heights_in]
    plot_w = max(1e-6, float(panel_width_in))
    gaps_h = max(0, len(heights) - 1) * max(0.0, float(vspace_in))
    fig_w = float(left_in) + plot_w + float(right_in) + float(reserve_right_in)
    fig_h = float(bottom_in) + float(top_in) + float(sum(heights)) + float(gaps_h)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(dpi))
    axes: list[plt.Axes] = []
    current_top = fig_h - float(top_in)
    for h_in in heights:
        y0_in = current_top - float(h_in)
        ax = fig.add_axes(
            [
                float(left_in) / float(fig_w),
                float(y0_in) / float(fig_h),
                float(plot_w) / float(fig_w),
                float(h_in) / float(fig_h),
            ]
        )
        axes.append(ax)
        current_top = y0_in - float(vspace_in)
    return fig, axes, float(plot_w), heights


def _save_figure(fig: plt.Figure, path: str) -> None:
    fmt = _postgwas_output_format_from_path(path)
    save_kwargs = _postgwas_savefig_kwargs(fmt)
    if fmt == "pdf":
        backend_name = _postgwas_resolve_pdf_backend()
        if backend_name is not None:
            try:
                fig.savefig(path, backend=backend_name, **save_kwargs)
                return
            except Exception:
                pass
    fig.savefig(path, **save_kwargs)


def _save_figure_and_close(fig: plt.Figure, path: str) -> None:
    _save_figure(fig, path)
    plt.close(fig)


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
    _prepare_cjk_plotting()

    # Silence pandas chained-assignment warnings in this script
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*ChainedAssignmentError.*",
    )

    output_stem = _resolve_postgwas_output_stem(
        file,
        getattr(args, "_postgwas_plot_prefix", None),
    )

    chr_col, pos_col, p_col = args.chr, args.pos, args.pvalue
    anno_suffix = (
        str(args.anno).replace(".gz", "").split(".")[-1].lower()
        if args.anno
        else None
    )
    gff_query_cache: Optional[GFFQuery] = None
    status_enabled = _postgwas_status_enabled(args)
    src = os.path.basename(str(file))
    task_label = os.path.basename(str(output_stem))
    single_marker = str(getattr(args, "_postgwas_single_marker", _DEFAULT_SINGLE_MARKER))

    if not status_enabled:
        logger.info(f"Loading GWAS results from {src}... [{format_elapsed(0.0)}]")
    with CliStatus(
        f"Loading GWAS results from {src}...",
        enabled=status_enabled,
    ) as load_status:
        try:
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
        except Exception:
            load_status.fail(f"Loading GWAS results from {src} ...Failed")
            raise
        load_status.complete(
            f"Loading GWAS results from {src} (nSNP={int(df_all.shape[0])})"
        )
    if status_enabled:
        logger.info(f"* {task_label} (nSNP={int(df_all.shape[0])})")

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
        args.thr
        if args.thr is not None
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
        logger.info(f"Visualizing GWAS results from {src}...")
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
                # QQ keeps all threshold-passing SNPs and can down-sample
                # only sub-threshold points in fast mode.
                plotmodel_qq = GWASPLOT(
                    df,
                    chr_col,
                    pos_col,
                    p_col,
                    0.1,
                    compression=False,
                )
        width_in = float(_PANEL_WIDTH_IN)
        gene_panel_h_in = width_in / 20.0
        dpi = 300
        rasterized = _postgwas_should_rasterize_dense_layers(args.format)
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
        postgwas_job_workers = max(1, int(getattr(args, "_postgwas_job_workers", 1)))
        enable_pure_manhqq_parallel = (
            postgwas_job_workers == 1
            and args.manh_ratio is not None
            and args.qq_ratio is not None
            and effective_ldblock_ratio is None
        )
        pending_manh_fig: Optional[plt.Figure] = None
        pending_manh_path: Optional[str] = None

        plot_colors = None
        ldblock_style = _resolve_ldblock_style(args.ldblock_palette_spec)
        if plotmodel is not None:
            plot_colors = _manhattan_colors_for_subset(
                args.palette_spec,
                full_chr_labels,
                df[chr_col].drop_duplicates().tolist(),
            )
        qq_point_color = _resolve_qq_point_color(args.palette_spec)
        qq_band_color = str(_QQ_BAND_COLOR)

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
                    np.asarray([], dtype=float),
                    float(manh_fontsize_target),
                    ax.get_xlim(),
                )

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
                        marker=single_marker,
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
                        marker=single_marker,
                        rasterized=rasterized,
                        min_logp=manh_min_logp,
                        max_logp=manh_max_logp,
                    )
                else:
                    y_hl = _safe_neglog10_p(plotmodel.df.loc[df_hl_idx, "y"])
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
                        alpha=1,
                        zorder=10,
                        s=args.scatter_size,
                        **_marker_scatter_style("D"),
                    )
                    for idx in draw_hl_idx:
                        text = _sanitize_plot_text(df_hl.loc[idx, 3])
                        ax.text(
                            plotmodel.df.loc[idx, "x"],
                            float(_safe_neglog10_p(plotmodel.df.loc[idx, "y"])[0]),
                            s=text,
                            ha="center",
                            zorder=11,
                        )

                    plotmodel.manhattan(
                        None,
                        ax=ax,
                        color_set=plot_colors,
                        marker=single_marker,
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
                        marker=single_marker,
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
                    marker=single_marker,
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
                    marker=single_marker,
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
                    if len(bim_layout) == 1:
                        seg = bim_layout[0]
                        _apply_bimrange_manhattan_axis(
                            ax,
                            seg["chrom"],
                            int(seg["start"]),
                            int(seg["end"]),
                        )
                    else:
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
            manh_tick_values = apply_integer_yticks(ax)

            return (
                ax.get_ylim(),
                np.asarray(manh_tick_values, dtype=float),
                float(manh_fontsize_target),
                ax.get_xlim(),
            )

        # ----------------- Manhattan plot -----------------
        if args.manh_ratio is not None:
            fig, ax, _panel_w_in, manh_panel_h_in = _create_ratio_panel_figure(
                ratio=float(args.manh_ratio),
                dpi=dpi,
                panel_width_in=width_in,
            )
            manh_ylim, manh_yticks, manh_fontsize, manh_xlim = _draw_manhattan_axis(ax)
            manh_height_in = float(manh_panel_h_in)
            manh_axes_bounds = ax.get_position().bounds
            manh_path = os.path.join(args.out, f"{output_stem}.manh.{args.format}")
            if enable_pure_manhqq_parallel:
                pending_manh_fig = fig
                pending_manh_path = manh_path
            else:
                _save_figure_and_close(fig, manh_path)
        else:
            manh_path = None

        # ----------------- QQ plot -----------------
        if args.qq_ratio is not None:
            manh_save_executor: Optional[cf.ThreadPoolExecutor] = None
            manh_save_future: Optional[cf.Future] = None
            if (
                enable_pure_manhqq_parallel
                and pending_manh_fig is not None
                and pending_manh_path is not None
            ):
                manh_save_executor = cf.ThreadPoolExecutor(max_workers=1)
                manh_save_future = manh_save_executor.submit(
                    _save_figure_and_close,
                    pending_manh_fig,
                    pending_manh_path,
                )
                pending_manh_fig = None
                pending_manh_path = None
            try:
                if plotmodel_qq is None:
                    logger.warning("Warning: QQ plotting skipped because no SNPs are available.")
                    qq_path = None
                else:
                    qq_lower = (
                        float(manh_ylim[0])
                        if (manh_ylim is not None and len(manh_ylim) >= 1)
                        else (
                            float(args.ylim_min)
                            if args.ylim_min is not None
                            else 0.0
                        )
                    )
                    qq_upper_target = (
                        float(manh_ylim[1])
                        if (manh_ylim is not None and len(manh_ylim) >= 2)
                        else (
                            float(args.ylim_max)
                            if args.ylim_max is not None
                            else None
                        )
                    )
                    qq_y_in = 4.0
                    if manh_height_in is not None:
                        qq_y_in = manh_height_in
                    fig, ax2, _qq_panel_w_in, _qq_panel_h_in = _create_ratio_panel_figure(
                        ratio=float(args.qq_ratio),
                        dpi=dpi,
                        panel_height_in=qq_y_in,
                    )
                    plotmodel_qq.qq(
                        ax=ax2,
                        color_set=[qq_point_color, qq_band_color],
                        marker=single_marker,
                        line_color="black",
                        scatter_size=args.scatter_size,
                        qq_mode=("full" if args.fullscatter else "auto"),
                        qq_fast_max_points=_QQ_FAST_MAX_POINTS,
                        sig_p_threshold=(
                            float(threshold)
                            if (np.isfinite(threshold) and float(threshold) > 0.0)
                            else None
                        ),
                        axis_min=qq_lower,
                        axis_max=qq_upper_target,
                        band_color=qq_band_color,
                        rasterized=rasterized,
                    )
                    qq_xmax = float(np.log10(plotmodel_qq.df.shape[0] + 1))
                    if np.isfinite(qq_xmax) and qq_xmax > 0:
                        x_right = float(qq_xmax)
                    else:
                        _x_right = ax2.get_xlim()[1]
                        x_right = float(_x_right)
                    x_right = max(1.0, x_right)
                    qq_lower, qq_upper = _resolve_qq_ylim(
                        ax2,
                        lower=qq_lower,
                        upper=qq_upper_target,
                    )
                    _apply_qq_axes(
                        ax2,
                        y_lower=qq_lower,
                        y_upper=qq_upper,
                        x_right=x_right,
                        y_ticks=(
                            manh_yticks
                            if (manh_yticks is not None and manh_ylim is not None)
                            else None
                        ),
                    )
                    apply_integer_xticks(ax2)
                    if manh_fontsize is not None:
                        ax2.xaxis.label.set_size(manh_fontsize)
                        ax2.yaxis.label.set_size(manh_fontsize)
                        ax2.tick_params(axis="both", labelsize=manh_fontsize)
                    if hide_axis_labels:
                        ax2.set_xlabel("")
                        ax2.set_ylabel("")
                    qq_path = os.path.join(args.out, f"{output_stem}.qq.{args.format}")
                    _save_figure_and_close(fig, qq_path)
            finally:
                if manh_save_future is not None:
                    manh_save_future.result()
                if manh_save_executor is not None:
                    manh_save_executor.shutdown(wait=True)
        else:
            qq_path = None
        if pending_manh_fig is not None and pending_manh_path is not None:
            _save_figure_and_close(pending_manh_fig, pending_manh_path)
            pending_manh_fig = None
            pending_manh_path = None

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
                    ld_mat, sig_keys = _compute_ld_from_bed_rust(
                        str(args.genofile),
                        args.bimrange_tuples if args.bimrange_tuples is not None else [],
                        selected_sites=ld_sites,
                        threads=int(max(0, int(getattr(args, "thread", 0)))),
                        logger=logger,
                    )
                    # Keep Manhattan->gene/LD mapping lines even when genotype lookup fails:
                    # fallback to GWAS-derived ld_site_keys if no matched genotype SNP is returned.
                    if len(sig_keys) > 0:
                        ld_site_keys = sig_keys
                    if ld_mat.shape[0] < 2:
                        logger.warning(
                            "Warning: Requested SNPs were not found in genotype data; drawing empty LD block."
                        )
                        ld_mat = np.zeros((n_sig_sites, n_sig_sites), dtype=np.float32)
                        ld_overlay_text = "No matched SNPs"
                    else:
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
            if ldblock_style is not None:
                ld_cmap = ldblock_style["ld_cmap"]
                gene_block_color = str(ldblock_style["gene_block_color"])
                gene_line_color = str(ldblock_style["gene_line_color"])

            def _draw_ld_axis(ax: plt.Axes) -> None:
                LDblock(
                    ld_mat,
                    ax=ax,
                    vmin=0,
                    vmax=1,
                    cmap=ld_cmap,
                    rasterize_threshold=100,
                )
                n_ld = max(2, int(ld_mat.shape[0]))
                # Keep LD triangle body filling the whole panel width.
                ax.set_xlim(0.5, float(n_ld) - 0.5)
                ax.margins(x=0.0)
                if ld_overlay_text:
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
                pairs: list[tuple[float, float, bool]],
                *,
                line_color: str = "grey",
                sig_line_color: str = "red",
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
                for x_top, x_ld, is_sig in pairs:
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
                    poly_color = str(sig_line_color) if bool(is_sig) else str(line_color)
                    ax_bridge.plot(
                        [x_top_n, x_top_n, x_ld_n],
                        [1.04, joint_y, bottom_y],
                        color=poly_color,
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

            ld_n_sites = max(2, int(ld_mat.shape[0]))
            ld_h_in = width_in / effective_ldblock_ratio
            ld_min_h_in = width_in * _ld_min_height_over_width(ld_n_sites)
            ld_h_in = max(float(ld_h_in), float(ld_min_h_in))
            fig_ld = plt.figure(
                figsize=(width_in, ld_h_in),
                dpi=dpi,
            )
            ax_ld = fig_ld.add_subplot(111)
            ld_path = os.path.join(args.out, f"{output_stem}.ldblock.{args.format}")
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
            _save_figure_and_close(fig_ld, ld_path)

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
                gene_path = os.path.join(args.out, f"{output_stem}.gene.{args.format}")
                _save_figure_and_close(fig_gene, gene_path)
            else:
                gene_path = None

            # Combined Manhattan + LD panel
            manhld_manh_ratio = args.manh_ratio if args.manh_ratio is not None else 2.0
            manhld_manh_h_in = width_in / manhld_manh_ratio
            gene_bridge_scale = 0.5
            mid_gene_y_offset = 0.03
            mid_gene_ymin = -0.15
            mid_gene_ymax = 0.2
            # Keep Manhattan<->LD transition layer from becoming too tall:
            # cap middle layer height at width/15.
            mid_h_raw = gene_panel_h_in if use_gene_bridge else 0.22
            mid_h_cap = float(width_in) / 15.0
            mid_h_in = min(float(mid_h_raw), float(mid_h_cap))
            if ld_panel_xspan is None:
                ld_panel_frac_in_manh = 1.0
            else:
                ld_panel_frac_in_manh = float(ld_panel_xspan[1] - ld_panel_xspan[0])
            ld_h_in_combo = (
                width_in
                * ld_panel_frac_in_manh
                / effective_ldblock_ratio
            )
            ld_h_in_combo_min = (
                width_in
                * ld_panel_frac_in_manh
                * _ld_min_height_over_width(max(2, int(ld_mat.shape[0])))
            )
            ld_h_in_combo = max(0.5, float(ld_h_in_combo), float(ld_h_in_combo_min))
            fig_manhld, combo_axes, _combo_panel_w_in, _combo_panel_heights = _create_stacked_panel_figure(
                panel_width_in=width_in,
                panel_heights_in=[manhld_manh_h_in, mid_h_in, ld_h_in_combo],
                dpi=dpi,
                vspace_in=_PANEL_STACK_VSPACE_IN,
            )
            ax_manhld_top, ax_manhld_mid, ax_manhld_bot = combo_axes
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
                    nonsig_line_color="grey",
                )
            else:
                _draw_bridge_axis(
                    ax_manhld_mid,
                    ax_manhld_top,
                    ax_manhld_bot,
                    bridge_pairs,
                    line_color="grey",
                    sig_line_color="red",
                )
            if not (args.bimrange_tuples is not None and len(args.bimrange_tuples) == 1):
                _show_end_locs_without_xticks(
                    ax_manhld_top,
                    label_fontsize=max(3.0, float(_tmp_fontsize) * 0.85),
                )

            manhld_path = os.path.join(args.out, f"{output_stem}.manhld.{args.format}")
            _save_figure_and_close(fig_manhld, manhld_path)
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
            log_success(
                logger,
                f"{saved_paths[0][0]} plot saved to:\n  {format_path_for_display(saved_paths[0][1])}",
            )
        elif len(saved_paths) > 1:
            title = ", ".join([x[0] for x in saved_paths])
            body = "\n".join([f"  {format_path_for_display(x[1])}" for x in saved_paths])
            log_success(logger, f"{title} plots saved to:\n{body}")
        log_success(logger, f"Visualizing GWAS results from {src} ...Finished [{format_elapsed(time.time() - t_plot)}]")

    # ------------------------------------------------------------------
    # 2. Annotation of significant loci
    # ------------------------------------------------------------------
    if args.anno:
        if not status_enabled:
            logger.info(f"Annotating significant SNPs from {src}... [{format_elapsed(0.0)}]")
        with CliStatus(
            f"Annotating significant SNPs from {src}...",
            enabled=status_enabled,
        ) as anno_status:
            if not os.path.exists(args.anno):
                anno_status.complete(
                    "Annotating significant SNPs ...Skipped (annotation file not found)"
                )
            else:
                try:
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
                        anno = readanno(args.anno, _ANNO_DESC_KEY, gff_data=gff_query_cache.gff)
                    else:
                        anno = readanno(args.anno, _ANNO_DESC_KEY)
                    anno_chr = anno[0].astype(str).map(_normalize_chr)

                    # Exact overlap annotation
                    desc_exact = [
                        anno.loc[
                            (anno_chr == _normalize_chr(idx[0]))
                            & (anno[1] <= idx[1])
                            & (anno[2] >= idx[1])
                        ]
                        for idx in df_filter.index
                    ]
                    df_filter["desc"] = [
                        _format_gene_annotation_dict(x)
                        for x in desc_exact
                    ]

                    # Optional broadened window around SNP (卤 annobroaden kb)
                    if args.annobroaden is not None:
                        kb = args.annobroaden * 1_000
                        desc_broad = [
                            anno.loc[
                                (anno_chr == _normalize_chr(idx[0]))
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

                    anno_path = os.path.join(args.out, f"{output_stem}.{threshold}.anno.tsv")
                    df_out.to_csv(anno_path, sep="\t", index=False)
                except Exception:
                    anno_status.fail(f"Annotating significant SNPs from {src} ...Failed")
                    raise
                anno_status.complete(
                    f"Annotating significant SNPs from {src} ...Finished (nLead={int(df_out.shape[0])})"
                )
                log_success(logger, f"Annotation table saved to {format_path_for_display(anno_path)}")


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
    _prepare_cjk_plotting()
    files = [str(f) for f in args.merge_files]
    if len(files) == 0:
        logger.warning("Warning: merged plotting requested but no GWAS input file is available.")
        return

    chr_col, pos_col, p_col = args.chr, args.pos, args.pvalue
    t_merge = time.time()
    logger.info("Visualizing merged post-GWAS results...")
    merge_manh_ratio = args._merge_manh_ratio
    merge_qq_ratio = args._merge_qq_ratio
    needs_positional_merge = bool(
        (merge_manh_ratio is not None)
        or (args.ldblock_ratio is not None)
    )

    series_colors = _resolve_merge_series_colors(args.palette_spec, len(files))
    series_markers = (
        list(getattr(args, "_postgwas_merge_markers", []))
        if len(getattr(args, "_postgwas_merge_markers", [])) == len(files)
        else _resolve_merge_markers(args.marker_spec, len(files))
    )
    series_labels = [f"{i + 1}:{os.path.basename(path)}" for i, path in enumerate(files)]
    ldblock_style = _resolve_ldblock_style(args.ldblock_palette_spec)

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

    if needs_positional_merge and len(chrom_sets) >= 2:
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
            if bool(getattr(args, "_postgwas_single_requested", False)):
                logger.info(
                    "Visualizing merged post-GWAS results ...Skipped (per-file plotting will still run)"
                )
                return
            fallback_file = str(args.gwasfile[0]) if len(args.gwasfile) > 0 else files[0]
            logger.info("Visualizing merged post-GWAS results ...Skipped (fallback single-file plotting)")
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
        args.thr
        if args.thr is not None
        else (0.05 / plot_df.shape[0] if plot_df.shape[0] > 0 else np.nan)
    )

    xticks: list[float] = []
    xticklabels: list[str] = []
    x_separators: list[float] = []
    use_segmented_layout = len(bim_layout) > 0

    if needs_positional_merge:
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
            xticklabels = [_sanitize_plot_text(seg["label"]) for seg in bim_layout]
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
                xticklabels.append(_sanitize_plot_text(chrom))
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
        plot_df["_ylog"] = _safe_neglog10_p(pvals_draw)
    else:
        plot_df["_ylog"] = np.asarray([], dtype=float)

    width_in = float(_PANEL_WIDTH_IN)
    # Merged plots can contain multiple dense layers; always rasterize point/band
    # artists while keeping axes/text/legend vector-friendly.
    rasterized = True
    manh_path = None
    manh_height_in = None
    manh_fontsize: Optional[float] = None
    manh_ylim_pair: Optional[tuple[float, float]] = None
    manh_yticks_pair: Optional[np.ndarray] = None
    manh_xlim_pair: Optional[tuple[float, float]] = None
    manh_axes_bounds: Optional[tuple[float, float, float, float]] = None

    if merge_manh_ratio is not None:
        manh_ratio = float(merge_manh_ratio)
        manh_fontsize = _scaled_fontsize_for_manhattan(
            manh_ratio,
            width_in=width_in,
        )
        manh_loc_fontsize = max(3.0, float(manh_fontsize) * 0.85)
        fig, ax, _manh_panel_w_in, manh_panel_h_in = _create_ratio_panel_figure(
            ratio=manh_ratio,
            dpi=300,
            panel_width_in=width_in,
            reserve_right_in=_PANEL_LEGEND_RIGHT_IN,
        )
        legend_handles: list[object] = []
        draw_xmins: list[float] = []
        draw_xmaxs: list[float] = []
        thr_log = (
            float(-np.log10(float(threshold_merge)))
            if (np.isfinite(threshold_merge) and float(threshold_merge) > 0.0)
            else None
        )
        if thr_log is not None and np.isfinite(thr_log):
            ax.axhline(
                y=thr_log,
                color="grey",
                linewidth=1.0,
                linestyle="--",
                zorder=0,
            )

        for i, _file in enumerate(files):
            dfi = plot_df.loc[plot_df["_series_idx"] == i]
            if dfi.shape[0] == 0:
                continue
            yy = np.asarray(dfi["_ylog"], dtype=float)
            pvals_i = np.asarray(dfi[p_col], dtype=float)
            keep = np.isfinite(yy)
            if args.ylim_min is not None:
                keep = keep & (yy >= float(args.ylim_min))
            if args.ylim_max is not None:
                keep = keep & (yy <= float(args.ylim_max))
            if not bool(np.any(keep)):
                continue
            x_keep = np.asarray(dfi.loc[keep, "_x"], dtype=float)
            y_keep = yy[keep]
            p_keep = pvals_i[keep]
            sig_keep = np.isfinite(p_keep) & (p_keep <= float(threshold_merge))
            nonsig_keep = ~sig_keep
            if bool(np.any(nonsig_keep)):
                ax.scatter(
                    x_keep[nonsig_keep],
                    y_keep[nonsig_keep],
                    color="lightgrey",
                    marker=series_markers[i],
                    alpha=1,
                    s=args.scatter_size,
                    rasterized=rasterized,
                    **_marker_scatter_style(series_markers[i]),
                )
            if bool(np.any(sig_keep)):
                ax.scatter(
                    x_keep[sig_keep],
                    y_keep[sig_keep],
                    color=series_colors[i],
                    marker=series_markers[i],
                    alpha=1,
                    s=args.scatter_size,
                    rasterized=rasterized,
                    **_marker_scatter_style(series_markers[i]),
                )
            legend_handles.append(
                ax.scatter(
                    [],
                    [],
                    color=series_colors[i],
                    marker=series_markers[i],
                    alpha=1,
                    s=args.scatter_size,
                    label=series_labels[i],
                    **_marker_scatter_style(series_markers[i]),
                )
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

        if len(legend_handles) > 0:
            ax.legend(
                handles=legend_handles,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                borderaxespad=0.0,
                ncol=1,
                markerscale=2.0,
                fontsize=manh_fontsize,
            )
        _y0, _y1 = ax.get_ylim()
        lo = float(args.ylim_min) if args.ylim_min is not None else 0.0
        hi = float(args.ylim_max) if args.ylim_max is not None else float(_y1)
        if not (hi > lo):
            hi = lo + max(1e-9, abs(lo) * 1e-9)
        ax.set_ylim(lo, hi)
        manh_yticks_pair = apply_integer_yticks(ax)
        _yy0, _yy1 = ax.get_ylim()
        manh_ylim_pair = (float(_yy0), float(_yy1))
        manh_xlim_pair = (float(ax.get_xlim()[0]), float(ax.get_xlim()[1]))

        manh_axes_bounds = ax.get_position().bounds
        manh_path = os.path.join(args.out, f"{args.prefix}.merge.manh.{args.format}")
        _save_figure(fig, manh_path)
        manh_height_in = float(manh_panel_h_in)
        plt.close(fig)

    qq_path = None
    if merge_qq_ratio is not None:
        qq_fontsize = float(manh_fontsize) if manh_fontsize is not None else 6.0
        qq_y_in = manh_height_in if manh_height_in is not None else 4.0
        fig, ax, _qq_panel_w_in, _qq_panel_h_in = _create_ratio_panel_figure(
            ratio=float(merge_qq_ratio),
            dpi=300,
            panel_height_in=qq_y_in,
            reserve_right_in=_PANEL_LEGEND_RIGHT_IN,
        )
        legend_handles: list[object] = []

        exp_xmin = np.inf
        exp_xmax = -np.inf
        qq_band_n = 0
        for i, _file in enumerate(files):
            pvals = pvals_by_series.get(i)
            if pvals is None or pvals.size == 0:
                continue
            pvals_arr = np.asarray(pvals, dtype=float)
            qq_band_n = max(
                int(qq_band_n),
                int(np.sum(np.isfinite(pvals_arr) & (pvals_arr > 0.0))),
            )
            exp, obs = _qq_select_points_with_threshold(
                pvals_arr,
                sig_p_threshold=(
                    float(threshold_merge)
                    if (np.isfinite(threshold_merge) and float(threshold_merge) > 0.0)
                    else None
                ),
                max_points=_QQ_FAST_MAX_POINTS,
                keep_all=bool(args.fullscatter),
            )
            if exp.size == 0 or obs.size == 0:
                continue
            if exp.size > 0:
                exp_xmin = min(exp_xmin, float(np.nanmin(exp)))
                exp_xmax = max(exp_xmax, float(np.nanmax(exp)))
            ax.scatter(
                exp,
                obs,
                s=args.scatter_size,
                marker=series_markers[i],
                alpha=1,
                rasterized=rasterized,
                color=series_colors[i],
                **_marker_scatter_style(series_markers[i]),
            )
            legend_handles.append(
                ax.scatter(
                    [],
                    [],
                    s=args.scatter_size,
                    marker=series_markers[i],
                    alpha=1,
                    color=series_colors[i],
                    label=series_labels[i],
                    **_marker_scatter_style(series_markers[i]),
                )
            )

        if qq_band_n > 0:
            x_band, lower_band, upper_band = _qq_confidence_band_from_n(
                int(qq_band_n),
                max_points=_QQ_BAND_MAX_POINTS,
            )
            if x_band.size > 0:
                exp_xmin = min(exp_xmin, float(np.nanmin(x_band)))
                exp_xmax = max(exp_xmax, float(np.nanmax(x_band)))
                ax.fill_between(
                    x_band,
                    lower_band,
                    upper_band,
                    color=str(_QQ_BAND_COLOR),
                    alpha=0.25,
                    rasterized=rasterized,
                    zorder=0,
                )

        if not np.isfinite(exp_xmin) or not np.isfinite(exp_xmax):
            exp_xmin, exp_xmax = (0.0, 1.0)
        qq_lower = (
            float(manh_ylim_pair[0])
            if manh_ylim_pair is not None
            else (
                float(args.ylim_min)
                if args.ylim_min is not None
                else 0.0
            )
        )
        qq_upper_target = (
            float(manh_ylim_pair[1])
            if manh_ylim_pair is not None
            else (
                float(args.ylim_max)
                if args.ylim_max is not None
                else None
            )
        )
        if exp_xmax > exp_xmin:
            x_right = float(exp_xmax)
        else:
            eps = max(1e-9, abs(exp_xmin) * 1e-9)
            x_right = float(exp_xmax + eps)
        qq_lower, qq_upper = _resolve_qq_ylim(
            ax,
            lower=qq_lower,
            upper=qq_upper_target,
        )
        _apply_qq_axes(
            ax,
            y_lower=qq_lower,
            y_upper=qq_upper,
            x_right=x_right,
            y_ticks=manh_yticks_pair,
        )
        apply_integer_xticks(ax)
        line_left, line_right = ax.get_xlim()
        ax.plot([line_left, line_right], [line_left, line_right], lw=1.0, color="black")
        ax.set_xlabel("Expected -log10(p-value)")
        ax.set_ylabel("Observed -log10(p-value)")
        ax.xaxis.label.set_size(qq_fontsize)
        ax.yaxis.label.set_size(qq_fontsize)
        ax.tick_params(axis="both", labelsize=qq_fontsize)
        if len(legend_handles) > 0:
            ax.legend(
                handles=legend_handles,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                borderaxespad=0.0,
                ncol=1,
                markerscale=2.0,
                fontsize=qq_fontsize,
            )
        qq_path = os.path.join(args.out, f"{args.prefix}.merge.qq.{args.format}")
        _save_figure(fig, qq_path)
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
                ld_mat, sig_keys = _compute_ld_from_bed_rust(
                    str(args.genofile),
                    region_ranges,
                    selected_sites=ld_sites,
                    threads=int(max(0, int(getattr(args, "thread", 0)))),
                    logger=logger,
                )
                missing_n = int(len(ld_sites) - len(set(sig_keys)))
                if missing_n > 0:
                    logger.warning(
                        f"Warning: {missing_n} requested LD sites are not in genotype data and were ignored."
                    )
                if len(sig_keys) > 0:
                    ld_site_keys = sig_keys
                if ld_mat.shape[0] < 2:
                    logger.warning(
                        "Warning: Requested SNPs were not found in genotype data; drawing empty LD block."
                    )
                    ld_mat = np.zeros((n_sig_sites, n_sig_sites), dtype=np.float32)
                    ld_overlay_text = "No matched SNPs"
                else:
                    mode_text = "all SNPs" if ld_use_all_sites else "threshold-passing SNPs"
                    logger.info(f"Merge LD block built from {len(sig_keys)} {mode_text}.")

        ld_cmap = "Greys"
        gene_block_color = "grey"
        gene_line_color = "black"
        if ldblock_style is not None:
            ld_cmap = ldblock_style["ld_cmap"]
            gene_block_color = str(ldblock_style["gene_block_color"])
            gene_line_color = str(ldblock_style["gene_line_color"])

        ld_h_in = width_in / effective_ldblock_ratio
        ld_h_in = max(float(ld_h_in), float(width_in * _ld_min_height_over_width(max(2, int(ld_mat.shape[0])))))
        fig_ld = plt.figure(figsize=(width_in, ld_h_in), dpi=300)
        ax_ld = fig_ld.add_subplot(111)
        LDblock(ld_mat, ax=ax_ld, vmin=0, vmax=1, cmap=ld_cmap, rasterize_threshold=100)
        n_ld = max(2, int(ld_mat.shape[0]))
        # Keep LD triangle body filling the whole panel width.
        ax_ld.set_xlim(0.5, float(n_ld) - 0.5)
        ax_ld.margins(x=0.0)
        if ld_overlay_text:
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
        ld_path = os.path.join(args.out, f"{args.prefix}.merge.ldblock.{args.format}")
        _save_figure_and_close(fig_ld, ld_path)

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
                    gene_path = os.path.join(args.out, f"{args.prefix}.merge.gene.{args.format}")
                    _save_figure_and_close(fig_gene, gene_path)

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
        log_success(
            logger,
            f"{saved_paths[0][0]} plot saved to:\n  {format_path_for_display(saved_paths[0][1])}\n",
        )
    else:
        title = ", ".join([x[0] for x in saved_paths])
        body = "\n".join([f"  {format_path_for_display(x[1])}" for x in saved_paths])
        log_success(logger, f"{title} plots saved to:\n{body}\n")
    log_success(
        logger,
        f"Visualizing merged post-GWAS results ...Finished [{format_elapsed(time.time() - t_merge)}]",
    )


def _run_one_postgwas_task(file: str, args, logger: logging.Logger) -> str:
    logger = _ensure_postgwas_worker_file_logging(args, logger)
    mute_stream = bool(getattr(args, "_postgwas_worker_mute_stream", False))
    detached_handlers: list[logging.Handler] = []
    thread_ctx = (
        runtime_thread_stage(blas_threads=1, rayon_threads=1)
        if int(getattr(args, "_postgwas_job_workers", 1)) > 1
        else nullcontext()
    )
    if mute_stream:
        detached_handlers = _detach_stream_handlers(logger)
    try:
        with thread_ctx:
            if mute_stream:
                # In parallel worker mode, fully silence worker stdout/stderr to
                # avoid corrupting parent-side progress/spinner rendering.
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    with redirect_stdout(devnull), redirect_stderr(devnull):
                        GWASplot(file, args, logger)
            else:
                GWASplot(file, args, logger)
    finally:
        if mute_stream:
            _restore_handlers(logger, detached_handlers)
    return str(file)


def _run_postgwas_ldblock_only(args, logger: logging.Logger) -> None:
    """
    Draw LD block using genotype input only (no GWAS table).
    """
    if args.ldblock_ratio is None:
        logger.warning("Warning: LD-only mode requires --ldblock/--ldblock-all.")
        return
    if args.bimrange_tuples is None or len(args.bimrange_tuples) == 0:
        logger.warning("Warning: LD-only mode requires --bimrange; skipped.")
        return

    if args.ldblock_mode == "threshold":
        logger.warning(
            "Warning: --ldblock (threshold mode) requires GWAS p-values; "
            "without --gwasfile it falls back to all SNPs in --bimrange."
        )

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.size"] = 6
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = False
    _prepare_cjk_plotting()

    t_ld = time.time()
    logger.info("Visualizing LD block...")
    width_in = 8.0
    ld_overlay_text: Optional[str] = None

    if args.genofile is None:
        logger.warning(
            "Warning: --ldblock/--ldblock-all enabled but no genotype file provided; "
            "drawing zero-correlation LD block."
        )
        ld_mat = np.zeros((2, 2), dtype=np.float32)
        ld_site_keys: list[tuple[str, int]] = []
        ld_overlay_text = "No genotype"
        n_sites = 0
    else:
        ld_mat, ld_site_keys = _compute_ld_from_bed_rust(
            str(args.genofile),
            list(args.bimrange_tuples),
            selected_sites=None,
            threads=int(max(0, int(getattr(args, "thread", 0)))),
            logger=logger,
        )
        n_sites = int(ld_mat.shape[0])
        if n_sites < 2:
            logger.warning(
                "Warning: Fewer than 2 SNPs were found in selected --bimrange; "
                "drawing empty LD block."
            )
            ld_mat = np.zeros((max(2, n_sites), max(2, n_sites)), dtype=np.float32)
            ld_overlay_text = "Not enough SNPs"
        else:
            logger.info(f"LD-only block built from {n_sites} SNPs.")
    if args.genofile is None:
        ld_site_keys = []

    ldblock_style = _resolve_ldblock_style(args.ldblock_palette_spec)
    ld_cmap = "Greys"
    gene_block_color = "grey"
    gene_line_color = "black"
    if ldblock_style is not None:
        ld_cmap = ldblock_style["ld_cmap"]
        gene_block_color = str(ldblock_style["gene_block_color"])
        gene_line_color = str(ldblock_style["gene_line_color"])

    region_ranges = list(args.bimrange_tuples)
    ld_spans = _ld_bimrange_spans(ld_site_keys, region_ranges)
    show_ld_titles = True  # LD-only has no Manhattan panel above.

    gene_track_df = pd.DataFrame(columns=["feature", "strand", "attribute", "x_start", "x_end"])
    use_gene_panel = False
    if args.anno:
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
        gene_track_df = _project_gene_records_to_ld_spans(
            gene_raw,
            region_ranges,
            ld_spans,
        )
        if gene_track_df.shape[0] == 0:
            logger.warning(
                "Warning: No gene-structure records found in selected --bimrange; "
                "gene track above LD block is skipped."
            )
        else:
            use_gene_panel = True

    ld_h_in = width_in / float(args.ldblock_ratio)
    ld_h_in = max(
        float(ld_h_in),
        float(width_in * _ld_min_height_over_width(max(2, int(ld_mat.shape[0])))),
    )
    if use_gene_panel:
        gene_h_in = width_in / 20.0
        fig_ld = plt.figure(figsize=(width_in, gene_h_in + ld_h_in), dpi=300)
        gs = fig_ld.add_gridspec(2, 1, height_ratios=[gene_h_in, ld_h_in], hspace=0.03)
        ax_gene = fig_ld.add_subplot(gs[0, 0])
        ax_ld = fig_ld.add_subplot(gs[1, 0])
        _draw_gene_structure_axis(
            ax_gene,
            gene_track_df,
            arrow_color=gene_line_color,
            block_color=gene_block_color,
            line_width=0.9,
            arrow_step=1.0,
        )
    else:
        fig_ld = plt.figure(figsize=(width_in, ld_h_in), dpi=300)
        ax_ld = fig_ld.add_subplot(111)
        ax_gene = None

    LDblock(ld_mat, ax=ax_ld, vmin=0, vmax=1, cmap=ld_cmap, rasterize_threshold=100)
    n_ld = max(2, int(ld_mat.shape[0]))
    ax_ld.set_xlim(0.5, float(n_ld) - 0.5)
    ax_ld.margins(x=0.0)
    if ld_overlay_text:
        ax_ld.text(n_ld / 2.0, -n_ld / 2.0, ld_overlay_text, ha="center", va="center", fontsize=6)
    _draw_ld_bimrange_titles(ax_ld, ld_spans, enabled=show_ld_titles)

    if ax_gene is not None:
        ax_gene.set_xlim(0.5, float(n_ld) - 0.5)

    fig_ld.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.08)
    if args.ldblock_xspan is not None:
        fx0, fx1 = args.ldblock_xspan
        if ax_gene is not None:
            gx0, gy0, gw, gh = ax_gene.get_position().bounds
            new_gx0 = float(gx0) + float(gw) * float(fx0)
            new_gw = float(gw) * float(fx1 - fx0)
            ax_gene.set_position([new_gx0, gy0, new_gw, gh])
        cx0, cy0, cw, ch = ax_ld.get_position().bounds
        new_x0 = float(cx0) + float(cw) * float(fx0)
        new_w = float(cw) * float(fx1 - fx0)
        ax_ld.set_position([new_x0, cy0, new_w, ch])
        ax_ld.set_anchor("N")
        if ax_gene is not None:
            ax_gene.set_anchor("N")

    ld_path = os.path.join(args.out, f"{args.prefix}.ldblock.{args.format}")
    _save_figure_and_close(fig_ld, ld_path)
    log_success(
        logger,
        f"LD block plot saved to:\n  {format_path_for_display(ld_path)}\n",
    )
    log_success(
        logger,
        f"Visualizing LD block ...Finished [{format_elapsed(time.time() - t_ld)}]",
    )


def _run_postgwas_tasks_serial(
    files: list[str],
    args,
    logger: logging.Logger,
    *,
    total_start_ts: float,
    done_count: int = 0,
    file_to_idx: Optional[dict[str, int]] = None,
    skip_files: Optional[set[str]] = None,
    emit_final_success: bool = True,
) -> int:
    n_total = len(files)
    idx_map = (
        {str(k): int(v) for k, v in file_to_idx.items()}
        if file_to_idx is not None
        else {str(f): i for i, f in enumerate(files, start=1)}
    )
    skip = {str(x) for x in (skip_files or set())}
    setattr(args, "_postgwas_job_workers", 1)
    setattr(args, "_postgwas_worker_mute_stream", False)
    for file_path in files:
        if file_path in skip:
            continue
        idx = idx_map.get(file_path, 0)
        task_start_ts = time.monotonic()
        try:
            GWASplot(file_path, args, logger)
        except Exception:
            elapsed = format_elapsed(time.monotonic() - task_start_ts)
            print_failure(
                f"Task {idx}/{n_total}: "
                f"{os.path.basename(file_path)} ...Failed [{elapsed}]"
            )
            raise
        done_count += 1
    if emit_final_success:
        total_elapsed = format_elapsed(time.monotonic() - total_start_ts)
        print_success(
            f"Task {done_count}/{n_total} ...Finished [{total_elapsed}]",
            force_color=True,
        )
    return int(done_count)


def _run_postgwas_tasks(args, logger: logging.Logger) -> None:
    files = [str(f) for f in args.gwasfile]
    if len(files) == 0:
        return
    if len(files) == 1:
        setattr(args, "_postgwas_job_workers", 1)
        setattr(args, "_postgwas_worker_mute_stream", False)
        GWASplot(files[0], args, logger)
        return

    total_start_ts = time.monotonic()
    done_count = 0
    req_threads = int(args.thread)
    logical_workers = _resolve_postgwas_worker_count(req_threads, len(files))

    if (len(files) > 1) and (os.name == "nt") and (not _allow_windows_postgwas_process_pool()):
        logger.warning(
            "Warning: Windows multi-process postgwas plotting is unstable in this "
            "environment; falling back to serial execution. "
            "Set JANUSX_POSTGWAS_WINDOWS_PROCESS_POOL=1 to force experimental "
            "process-pool mode."
        )
        _run_postgwas_tasks_serial(
            files,
            args,
            logger,
            total_start_ts=total_start_ts,
            done_count=done_count,
        )
        return

    # In parallel mode, keep worker logs in file only to avoid spinner/pbar corruption.
    setattr(args, "_postgwas_worker_mute_stream", True)

    if rich_progress_available():
        n_total = len(files)
        basenames = [os.path.basename(f) for f in files]
        name_width = max((len(x) for x in basenames), default=0)
        idx_width = len(str(n_total))
        max_visible = min(5, n_total)
        n_workers = int(logical_workers)
        setattr(args, "_postgwas_job_workers", int(n_workers))
        file_to_idx = {f: i for i, f in enumerate(files, start=1)}
        progress = build_rich_progress(
            description_template="Task {task.fields[task_label]}: {task.fields[file_pad]}",
            show_bar=False,
            show_percentage=False,
            show_elapsed=False,
            show_remaining=False,
            finished_text=" ",
            transient=True,
        ) if should_animate_status("Loading merged post-GWAS tasks...") else None
        with (progress if progress is not None else nullcontext()):
            task_start_ts: dict[str, float] = {}
            task_map: dict[str, int] = {}
            future_map: dict[cf.Future[str], str] = {}
            completed_files: set[str] = set()
            pending_iter = iter(files)

            def _add_visible_task(file_path: str) -> None:
                if file_path in task_map:
                    return
                if len(task_map) >= max_visible:
                    return
                idx = file_to_idx[file_path]
                if progress is None:
                    return
                task_map[file_path] = progress.add_task(
                    description="",
                    total=None,
                    task_label=f"{idx:>{idx_width}}/{n_total}",
                    file_pad=os.path.basename(file_path).ljust(name_width),
                )

            def _submit_next(executor: cf.ProcessPoolExecutor) -> bool:
                try:
                    f_next = next(pending_iter)
                except StopIteration:
                    return False
                fut = executor.submit(_run_one_postgwas_task, f_next, args, logger)
                future_map[fut] = f_next
                task_start_ts[f_next] = time.monotonic()
                return True

            def _fill_visible_from_running() -> None:
                if len(task_map) >= max_visible:
                    return
                for running_file in list(future_map.values()):
                    if len(task_map) >= max_visible:
                        break
                    _add_visible_task(running_file)

            try:
                with _build_postgwas_process_pool(n_workers) as ex:
                    for _ in range(min(n_workers, n_total)):
                        if not _submit_next(ex):
                            break
                    _fill_visible_from_running()

                    while len(future_map) > 0:
                        done, _ = cf.wait(
                            list(future_map.keys()),
                            timeout=0.1,
                            return_when=cf.FIRST_COMPLETED,
                        )
                        if len(done) == 0:
                            _fill_visible_from_running()
                            continue
                        fut = min(
                            done,
                            key=lambda x: file_to_idx.get(
                                future_map.get(x, ""),
                                10**9,
                            ),
                        )
                        file_path = future_map.pop(fut)
                        tid = task_map.pop(file_path, None)
                        if tid is not None:
                            try:
                                progress.remove_task(tid)
                            except Exception:
                                pass
                        elapsed = format_elapsed(
                            time.monotonic()
                            - task_start_ts.get(file_path, time.monotonic())
                        )
                        try:
                            done_file = str(fut.result())
                        except BrokenProcessPool:
                            raise
                        except Exception:
                            idx = file_to_idx.get(file_path, 0)
                            print_failure(
                                f"Task {idx}/{n_total}: "
                                f"{os.path.basename(file_path)} ...Failed [{elapsed}]"
                            )
                            raise
                        _ = done_file
                        completed_files.add(file_path)
                        done_count += 1
                        _submit_next(ex)
                        _fill_visible_from_running()
            except BrokenProcessPool:
                _log_postgwas_broken_pool_hint(
                    logger,
                    n_workers=n_workers,
                    req_threads=req_threads,
                    n_files=n_total,
                )
                print_warning(
                    f"PostGWAS worker pool broke after {done_count}/{n_total} tasks; "
                    "retrying remaining tasks serially."
                )
                for f, tid in list(task_map.items()):
                    try:
                        progress.remove_task(tid)
                    except Exception:
                        pass
                    task_map.pop(f, None)
                _run_postgwas_tasks_serial(
                    files,
                    args,
                    logger,
                    total_start_ts=total_start_ts,
                    done_count=done_count,
                    file_to_idx=file_to_idx,
                    skip_files=completed_files,
                )
                return
            except PermissionError as exc:
                print_warning(
                    "PostGWAS worker pool is unavailable in this environment; "
                    f"retrying serially ({exc})."
                )
                for f, tid in list(task_map.items()):
                    try:
                        progress.remove_task(tid)
                    except Exception:
                        pass
                    task_map.pop(f, None)
                _run_postgwas_tasks_serial(
                    files,
                    args,
                    logger,
                    total_start_ts=total_start_ts,
                    done_count=done_count,
                    file_to_idx=file_to_idx,
                    skip_files=completed_files,
                )
                return
            except Exception:
                for f, tid in list(task_map.items()):
                    try:
                        progress.remove_task(tid)
                    except Exception:
                        pass
                    task_map.pop(f, None)
                raise
        total_elapsed = format_elapsed(time.monotonic() - total_start_ts)
        print_success(f"Task {done_count}/{n_total} ...Finished [{total_elapsed}]", force_color=True)
        return

    if _HAS_TQDM and stdout_is_tty() and should_animate_status("Loading merged post-GWAS tasks..."):
        setattr(args, "_postgwas_job_workers", int(logical_workers))
        pbar = tqdm(
            total=len(files),
            desc="PostGWAS tasks",
            unit="file",
            leave=False,
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                       "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        task_start_ts: dict[str, float] = {}
        file_to_idx = {f: i for i, f in enumerate(files, start=1)}
        completed_files: set[str] = set()
        try:
            future_map: dict[cf.Future[str], str] = {}
            with _build_postgwas_process_pool(logical_workers) as ex:
                for file_path in files:
                    future_map[ex.submit(_run_one_postgwas_task, file_path, args, logger)] = file_path
                    task_start_ts[file_path] = time.monotonic()
                for fut in cf.as_completed(future_map):
                    file_path = future_map[fut]
                    elapsed = format_elapsed(
                        time.monotonic()
                        - task_start_ts.get(file_path, time.monotonic())
                    )
                    try:
                        done_file = str(fut.result())
                    except BrokenProcessPool:
                        raise
                    except Exception:
                        idx = file_to_idx.get(file_path, 0)
                        print_failure(
                            f"Task {idx}/{len(files)}: "
                            f"{os.path.basename(file_path)} ...Failed [{elapsed}]"
                        )
                        raise
                    pbar.update(1)
                    pbar.set_postfix(file=os.path.basename(str(done_file)))
                    completed_files.add(file_path)
                    done_count += 1
        except BrokenProcessPool:
            _log_postgwas_broken_pool_hint(
                logger,
                n_workers=logical_workers,
                req_threads=req_threads,
                n_files=len(files),
            )
            print_warning(
                f"PostGWAS worker pool broke after {done_count}/{len(files)} tasks; "
                "retrying remaining tasks serially."
            )
            _run_postgwas_tasks_serial(
                files,
                args,
                logger,
                total_start_ts=total_start_ts,
                done_count=done_count,
                file_to_idx=file_to_idx,
                skip_files=completed_files,
            )
            return
        except PermissionError as exc:
            print_warning(
                "PostGWAS worker pool is unavailable in this environment; "
                f"retrying serially ({exc})."
            )
            _run_postgwas_tasks_serial(
                files,
                args,
                logger,
                total_start_ts=total_start_ts,
                done_count=done_count,
                file_to_idx=file_to_idx,
                skip_files=completed_files,
            )
            return
        finally:
            pbar.close()
        total_elapsed = format_elapsed(time.monotonic() - total_start_ts)
        print_success(f"Task {done_count}/{len(files)} ...Finished [{total_elapsed}]", force_color=True)
        return

    setattr(args, "_postgwas_job_workers", int(logical_workers))
    task_start_ts: dict[str, float] = {}
    file_to_idx = {f: i for i, f in enumerate(files, start=1)}
    future_map: dict[cf.Future[str], str] = {}
    completed_files: set[str] = set()
    try:
        with _build_postgwas_process_pool(logical_workers) as ex:
            for file_path in files:
                future_map[ex.submit(_run_one_postgwas_task, file_path, args, logger)] = file_path
                task_start_ts[file_path] = time.monotonic()
            for fut in cf.as_completed(future_map):
                file_path = future_map[fut]
                elapsed = format_elapsed(
                    time.monotonic()
                    - task_start_ts.get(file_path, time.monotonic())
                )
                try:
                    _ = str(fut.result())
                except BrokenProcessPool:
                    raise
                except Exception:
                    idx = file_to_idx.get(file_path, 0)
                    print_failure(
                        f"Task {idx}/{len(files)}: "
                        f"{os.path.basename(file_path)} ...Failed [{elapsed}]"
                    )
                    raise
                completed_files.add(file_path)
                done_count += 1
    except BrokenProcessPool:
        _log_postgwas_broken_pool_hint(
            logger,
            n_workers=logical_workers,
            req_threads=req_threads,
            n_files=len(files),
        )
        print_warning(
            f"PostGWAS worker pool broke after {done_count}/{len(files)} tasks; "
            "retrying remaining tasks serially."
        )
        _run_postgwas_tasks_serial(
            files,
            args,
            logger,
            total_start_ts=total_start_ts,
            done_count=done_count,
            file_to_idx=file_to_idx,
            skip_files=completed_files,
        )
        return
    except PermissionError as exc:
        print_warning(
            "PostGWAS worker pool is unavailable in this environment; "
            f"retrying serially ({exc})."
        )
        _run_postgwas_tasks_serial(
            files,
            args,
            logger,
            total_start_ts=total_start_ts,
            done_count=done_count,
            file_to_idx=file_to_idx,
            skip_files=completed_files,
        )
        return
    total_elapsed = format_elapsed(time.monotonic() - total_start_ts)
    print_success(f"Task {done_count}/{len(files)} ...Finished [{total_elapsed}]", force_color=True)


def main(argv: Optional[list[str]] = None):
    warn_deprecated_alias_usage(("-threshold", "--threshold"), replacement="-thr/--thr")
    t_start = time.time()

    parser = CliArgumentParser(
        prog="jx postgwas",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx postgwas -gwasfile result.lmm.tsv -manh -qq",
            "jx postgwas -i a.tsv b.tsv -manh-merge -qq-merge -marker '1,o,x'",
            "jx postgwas -gwasfile result.lmm.tsv -a genes.bed -ab 50",
            "jx postgwas -bfile test/geno -bimrange 1:1-2 -ldblock-all",
        ]),
    )

    # ------------------------------------------------------------------
    # Input arguments
    # ------------------------------------------------------------------
    input_group = parser.add_argument_group("Input Arguments")
    input_group.add_argument(
        "-i", "-gwasfile", "--gwasfile", nargs="+", type=str, required=False, default=None,
        help=(
            "One or more GWAS result files (tab-delimited). "
            "Optional only when running LD block only with genotype input."
        ),
    )
    geno_group = input_group.add_mutually_exclusive_group(required=False)
    geno_group.add_argument(
        "-bfile", "--bfile", type=str, default=None,
        help="Genotype PLINK prefix (for LD block/LD clump).",
    )
    geno_group.add_argument(
        "-vcf", "--vcf", type=str, default=None,
        help="Genotype VCF/VCF.GZ file (for LD block/LD clump).",
    )
    geno_group.add_argument(
        "-hmp", "--hmp", type=str, default=None,
        help="Genotype HMP/HMP.GZ file (for LD block/LD clump).",
    )
    geno_group.add_argument(
        "-file", "--file", dest="geno", type=str, default=None,
        help=(
            "Genotype numeric matrix (.txt/.tsv/.csv/.npy) or prefix. "
            "Requires sibling prefix.id. For LD/LDclump, also requires real prefix.site or prefix.bim."
        ),
    )
    input_group.add_argument(
        "-a", "--anno", type=str, default=None,
        help="Annotation file: .gff/.gff3 or .bed.",
    )

    # ------------------------------------------------------------------
    # Plot arguments
    # ------------------------------------------------------------------
    plot_group = parser.add_argument_group("Plot Arguments")
    plot_group.add_argument(
        "-manh", "--manh", type=str, nargs="?", const="2", default=None,
        help=(
            "Enable Manhattan plotting with aspect ratio (width/height). "
            "Examples: --manh (default 2), --manh 2, --manh 3/2."
        ),
    )
    ldblock_group = plot_group.add_mutually_exclusive_group(required=False)
    ldblock_group.add_argument(
        "-ldblock", "--ldblock", type=str, nargs="?", const="2", default=None,
        help=(
            "Enable LD block inverted triangle plotting with aspect ratio (width/height), "
            "using only threshold-passing SNPs. Requires --bimrange. "
            "You can also pass x-span in Manhattan-width fraction, e.g. 0.2:0.8 or 0.2-0.8 "
            "(then ratio defaults to 2). "
            "If only ratio is given, x-span defaults to 0:1. "
            "You may also pass a colormap/palette token here (e.g. tab10 or white;yellow;red), "
            "which keeps ratio=2 by default."
        ),
    )
    ldblock_group.add_argument(
        "-ldblock-all", "--ldblock-all", dest="ldblock_all", type=str, nargs="?", const="2", default=None,
        help=(
            "Enable LD block inverted triangle plotting with aspect ratio (width/height), "
            "using all SNPs in selected bimrange. Requires --bimrange. "
            "You can also pass x-span in Manhattan-width fraction, e.g. 0.2:0.8 or 0.2-0.8 "
            "(then ratio defaults to 2). "
            "If only ratio is given, x-span defaults to 0:1. "
            "You may also pass a colormap/palette token here (e.g. tab10 or white;yellow;red), "
            "which keeps ratio=2 by default."
        ),
    )
    plot_group.add_argument(
        "-qq", "--qq", type=str, nargs="?", const="5/4", default=None,
        help=(
            "Enable QQ plotting in auto mode with aspect ratio (width/height). "
            "Examples: --qq (default 5/4), --qq 5/4, --qq 2."
        ),
    )
    plot_group.add_argument(
        "-manh-merge", "--manh-merge", dest="manh_merge", type=str, nargs="?", const="2", default=None,
        help=(
            "Draw one merged Manhattan plot from the GWAS files passed by -i. "
            "Ratio parsing matches --manh."
        ),
    )
    plot_group.add_argument(
        "-qq-merge", "--qq-merge", dest="qq_merge", type=str, nargs="?", const="5/4", default=None,
        help=(
            "Draw one merged QQ plot from the GWAS files passed by -i. "
            "Ratio parsing matches --qq."
        ),
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
        "-thr", "--thr", dest="thr", type=float, default=None,
        help="P-value threshold; if not set, use 0.05 / nSNP (default: %(default)s).",
    )
    optional_group.add_argument(
        "-threshold", "--threshold", dest="thr", type=float, default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-LDclump", "--LDclump", dest="ldclump", nargs=2, default=None,
        metavar=("WINDOW", "R2"),
        help=(
            "Enable LD clumping for annotation output using threshold-passing SNPs only. "
            "Format: --LDclump <window> <r2>, e.g. --LDclump 500kb 0.8. "
            "Window supports kb/mb/bp (no unit defaults to kb). "
            "Requires genotype input via --bfile/--vcf/--hmp/--file."
        ),
    )
    optional_group.add_argument(
        "-bimrange", "--bimrange", type=str, action="append", default=None,
        help=(
            "Plotting range filter in Mb, format chr:start-end "
            "(also accepts chr:start:end). "
            "If start/end are integer-like and >6 digits, they are auto-treated as bp "
            "(with warning) and axis labels are shown in Mb. "
            "Can be specified multiple times."
        ),
    )
    optional_group.add_argument(
        "-ylim", "--ylim", type=str, default=None,
        help=(
            "Y range for Manhattan as <max>, <min:max>, <min:>, or <:max> "
            "(also accepts '-' as separator), e.g. --ylim 6, --ylim 0:6, "
            "--ylim 2:, --ylim :6. Missing bound is auto-determined. "
            "Points outside this range are filtered before Manhattan scatter "
            "to speed plotting. QQ keeps all threshold-passing points and "
            "can down-sample sub-threshold points for speed."
        ),
    )
    optional_group.add_argument(
        "-palette", "--palette", dest="palette", type=str, default=None,
        help=(
            "Manhattan color palette and QQ scatter color palette "
            "(QQ confidence band always stays grey). "
            "Supports cmap names (e.g. tab10, tab20) or ';'-separated colors "
            "(e.g. #1f77b4;#ff7f0e or (215,123,254);(1,1,1)). "
            "A single color is also accepted. "
            "If omitted, use default black for QQ scatter and black/grey for Manhattan."
        ),
    )
    optional_group.add_argument(
        "-ldblock-palette", "--ldblock-palette",
        dest="ldblock_palette",
        type=str,
        default=None,
        help=(
            "LD block colormap only (independent from --palette). "
            "Supports cmap names (e.g. tab10/tab20) or color lists "
            "like 'white;yellow;red' or 'white,yellow,red'. "
            "If omitted, LD block keeps default greyscale."
        ),
    )
    optional_group.add_argument(
        "-scatter-size", "--scatter-size", type=float, default=8.0,
        help="Scatter marker size for Manhattan and QQ plots (default: %(default)s).",
    )
    optional_group.add_argument(
        "-marker", "--marker", type=str, default=None,
        help=(
            "Scatter marker(s). Single/non-merge plotting uses the first marker only "
            "(default: o). Merge plotting cycles markers by input GWAS file; "
            "default cycle is 1,2,3,4,*,+,x. "
            "Example: --marker '1,o,x'."
        ),
    )
    optional_group.add_argument(
        "-full", "--full", "-fullscatter", "--fullscatter",
        dest="fullscatter",
        action="store_true",
        default=False,
        help=(
            "Disable scatter compression for both Manhattan and QQ and draw all points "
            "(no fast optimization; no 0.5 cut)."
        ),
    )
    optional_group.add_argument(
        "-fmt", "--fmt", dest="format", type=str, default="png",
        help="Output figure format: pdf, png, svg, tif (default: %(default)s).",
    )
    optional_group.add_argument(
        "-ab", "--annobroaden", type=float, default=None,
        help="Broaden the annotation window around SNPs (Kb) (default: %(default)s).",
    )
    add_common_out_arg(optional_group, default=".", help_profile="plot_annotation")
    add_common_prefix_arg(
        optional_group,
        default=None,
        help_text=(
            "Output prefix. For single-GWAS plotting, figures are saved as "
            "<prefix>.<input-stem>.* ; if omitted, use the input filename stem. "
            "The same prefix is also used for the run log stem."
        ),
    )
    add_common_thread_arg(
        optional_group,
        default_threads=detect_effective_threads(),
        help_profile="default",
    )

    args = parser.parse_args(argv)
    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    thread_capped = False
    if int(args.thread) <= 0:
        args.thread = int(detected_threads)
    if int(args.thread) > int(detected_threads):
        thread_capped = True
        args.thread = int(detected_threads)
    # `--highlight` was removed from CLI; keep a disabled attribute for
    # internal legacy branches that still check it.
    args.highlight = None
    args.gwasfile = (
        [str(x) for x in list(args.gwasfile)]
        if args.gwasfile is not None
        else []
    )

    args.out = os.path.normpath(args.out if args.out is not None else ".")
    user_prefix = str(args.prefix).strip() if args.prefix is not None else ""
    args._postgwas_plot_prefix = user_prefix
    args.prefix = "JanusX" if user_prefix == "" else user_prefix

    # Create output directory if needed
    if args.out != "":
        os.makedirs(args.out, mode=0o755, exist_ok=True)
        configure_genotype_cache_from_out(args.out)
    else:
        args.out = "."

    outprefix_base = os.path.join(args.out, args.prefix)
    log_path = f"{outprefix_base}.postGWAS.log"
    logger = setup_logging(log_path)
    args._postgwas_log_path = str(log_path)
    if thread_capped:
        logger.warning(
            f"Warning: Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(args.thread)}."
        )
    apply_blas_thread_env(int(args.thread))
    # maybe_warn_non_openblas(
    #     logger=logger,
    #     strict=require_openblas_by_default(),
    # )

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
        args.palette_spec = _parse_palette_spec(args.palette)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    try:
        args.ldblock_palette_spec = _parse_palette_spec(args.ldblock_palette)
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
        try:
            args.qq_ratio = _parse_ratio(args.qq, "QQ")
        except ValueError as e:
            logger.error(str(e))
            raise SystemExit(1)
    try:
        args.manh_merge_ratio = (
            _parse_ratio(args.manh_merge, "Merged Manhattan")
            if args.manh_merge is not None
            else None
        )
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    if args.qq_merge is None:
        args.qq_merge_ratio = None
    else:
        try:
            args.qq_merge_ratio = _parse_ratio(args.qq_merge, "Merged QQ")
        except ValueError as e:
            logger.error(str(e))
            raise SystemExit(1)
    try:
        args.marker_spec = _parse_marker_spec(args.marker)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    args._postgwas_single_marker = _resolve_single_marker(args.marker_spec)
    args._postgwas_merge_markers = []

    try:
        args.ldblock_ratio = None
        args.ldblock_xspan = None
        args.ldblock_mode = None
        if args.ldblock_all is not None:
            try:
                args.ldblock_ratio, args.ldblock_xspan = _parse_ldblock_spec(
                    args.ldblock_all, "LDBlock-all", logger
                )
            except ValueError as ratio_err:
                try:
                    pal_spec = _parse_palette_spec(args.ldblock_all)
                except ValueError:
                    logger.error(str(ratio_err))
                    raise SystemExit(1)
                if pal_spec is None:
                    logger.error(str(ratio_err))
                    raise SystemExit(1)
                args.ldblock_ratio, args.ldblock_xspan = 2.0, (0.0, 1.0)
                if args.ldblock_palette_spec is None:
                    args.ldblock_palette_spec = pal_spec
                    args.ldblock_palette = str(args.ldblock_all)
            args.ldblock_mode = "all"
        elif args.ldblock is not None:
            try:
                args.ldblock_ratio, args.ldblock_xspan = _parse_ldblock_spec(
                    args.ldblock, "LDBlock", logger
                )
            except ValueError as ratio_err:
                try:
                    pal_spec = _parse_palette_spec(args.ldblock)
                except ValueError:
                    logger.error(str(ratio_err))
                    raise SystemExit(1)
                if pal_spec is None:
                    logger.error(str(ratio_err))
                    raise SystemExit(1)
                args.ldblock_ratio, args.ldblock_xspan = 2.0, (0.0, 1.0)
                if args.ldblock_palette_spec is None:
                    args.ldblock_palette_spec = pal_spec
                    args.ldblock_palette = str(args.ldblock)
            args.ldblock_mode = "threshold"
    except SystemExit:
        raise
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
    if args.bimrange_tuples is not None and (
        args.qq_ratio is not None or args.qq_merge_ratio is not None
    ):
        logger.info(
            "QQ is disabled when --bimrange is set."
        )
        args.qq_ratio = None
        args.qq_merge_ratio = None
    if args.ldblock_ratio is not None and args.bimrange_tuples is None:
        logger.warning(
            "Warning: --ldblock/--ldblock-all requires --bimrange; LD block and Manhattan+LD plotting are skipped."
        )
        args.ldblock_ratio = None
        args.ldblock_xspan = None
        args.ldblock_mode = None

    args.genofile = args.bfile or args.vcf or args.hmp or args.geno
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
    merge_plot_requested = bool(
        (args.manh_merge_ratio is not None)
        or (args.qq_merge_ratio is not None)
    )
    single_plot_requested = bool(
        (args.manh_ratio is not None)
        or (args.qq_ratio is not None)
        or (args.anno is not None)
        or ((args.ldblock_ratio is not None) and (len(args.gwasfile) > 0))
    )
    args.merge_mode = bool(merge_plot_requested)
    args._postgwas_merge_requested = bool(merge_plot_requested)
    args._postgwas_single_requested = bool(single_plot_requested)
    if merge_plot_requested:
        args.merge_files = [str(x) for x in list(args.gwasfile)]
        if len(args.merge_files) == 0:
            logger.error(
                "Merged plotting requires at least one GWAS file from -i/--gwasfile."
            )
            raise SystemExit(1)
        args._merge_manh_ratio = args.manh_merge_ratio
        args._merge_qq_ratio = args.qq_merge_ratio
        if args._merge_manh_ratio is None and args._merge_qq_ratio is None:
            logger.info("Merge plotting detected; forcing merged Manhattan ratio to 2.")
            args._merge_manh_ratio = 2.0
        args._postgwas_merge_markers = _resolve_merge_markers(
            args.marker_spec,
            len(args.merge_files),
        )
    else:
        args.merge_files = []
        args._merge_manh_ratio = None
        args._merge_qq_ratio = None

    args._postgwas_outer_workers = int(
        _resolve_postgwas_worker_count(int(args.thread), len(args.gwasfile))
    ) if (single_plot_requested and len(args.gwasfile) > 1) else 1

    args.ldblock_only_mode = bool(
        (len(args.gwasfile) == 0) and (args.ldblock_ratio is not None)
    )
    if len(args.gwasfile) == 0:
        if args.manh_ratio is not None:
            logger.warning(
                "Warning: --manh requires GWAS result file(s); it is ignored in LD-only mode."
            )
            args.manh_ratio = None
        if args.qq_ratio is not None:
            logger.warning(
                "Warning: --qq requires GWAS result file(s); it is ignored in LD-only mode."
            )
            args.qq_ratio = None
        if args.anno is not None:
            logger.info(
                "LD-only mode: --anno will be used for gene-structure track above LD block."
            )
        if args.ldclump_window_bp is not None:
            logger.warning(
                "Warning: --LDclump requires GWAS result file(s); it is ignored in LD-only mode."
            )
            args.ldclump_window_bp = None
            args.ldclump_r2 = None
        if args.ldblock_ratio is None:
            logger.error(
                "No GWAS input file provided. "
                "Use --gwasfile, or run LD-only mode with --ldblock/--ldblock-all + genotype input."
            )
            raise SystemExit(1)
        single_plot_requested = False
        args._postgwas_single_requested = False

    if not hasattr(args, "fullscatter"):
        args.fullscatter = bool(getattr(args, "full", False))

    no_plot_or_anno = (
        (not merge_plot_requested)
        and (not single_plot_requested)
        and (not bool(getattr(args, "ldblock_only_mode", False)))
    )
    args.disable_compression = bool(args.fullscatter or (args.bimrange_tuples is not None))

    # ------------------------------------------------------------------
    # Configuration summary
    # ------------------------------------------------------------------
    config_title = "JanusX - Post-GWAS"
    host_text = socket.gethostname()
    input_files = list(args.gwasfile or [])
    input_files_text = _format_input_files(input_files)
    if bool(getattr(args, "ldblock_only_mode", False)):
        mode_text = "ldblock-only"
    elif merge_plot_requested and single_plot_requested:
        mode_text = "single+merge"
    elif merge_plot_requested:
        mode_text = "merge"
    else:
        mode_text = "single-file" if len(input_files) <= 1 else "multi-file"
    threshold_text = str(args.thr if args.thr is not None else "0.05 / nSNP")
    bimrange_text = _format_bimrange_summary(args.bimrange_tuples)
    merge_map_rows = (
        [(str(i), str(file)) for i, file in enumerate(args.merge_files)]
        if merge_plot_requested
        else []
    )
    threads_text = format_requested_thread_usage(
        requested_threads=int(requested_threads),
        using_threads=int(args.thread),
        detected_threads=int(detected_threads),
    )

    base_rows: list[tuple[str, str]] = [
        ("Mode", mode_text),
        ("GWAS files", input_files_text),
        ("Chr|Pos|Pvalue", f"{args.chr}|{args.pos}|{args.pvalue}"),
        ("Genotype file", str(args.genofile) if args.genofile is not None else "NA"),
        ("Threshold", threshold_text),
        ("Bimrange", bimrange_text),
    ]

    vis_rows: Optional[list[tuple[str, str]]] = None
    if (
        args.manh_ratio is not None
        or args.qq_ratio is not None
        or args._merge_manh_ratio is not None
        or args._merge_qq_ratio is not None
        or args.ldblock_ratio is not None
    ):
        single_manh_pal_text = (
            "default (black/grey)" if args.palette_spec is None else str(args.palette)
        )
        single_qq_pal_text = (
            "default (black; band=grey)"
            if args.palette_spec is None
            else f"{args.palette} (band=grey)"
        )
        if args.palette_spec is None:
            merge_default_cmap = "tab10" if len(args.merge_files) <= 10 else "tab20"
            merge_manh_pal_text = f"default ({merge_default_cmap})"
            merge_qq_pal_text = f"default ({merge_default_cmap}; band=grey)"
        else:
            merge_manh_pal_text = str(args.palette)
            merge_qq_pal_text = f"{args.palette} (band=grey)"
        if args.disable_compression:
            if args.fullscatter and args.bimrange_tuples is not None:
                comp_text = "off (--full, auto for --bimrange)"
            elif args.fullscatter:
                comp_text = "off (--full)"
            else:
                comp_text = "off (auto for --bimrange)"
        else:
            comp_text = "on"
        vis_rows = [("Format", str(args.format))]
        if args.manh_ratio is not None:
            vis_rows.append(
                (
                    "Single Manhattan",
                    f"ratio={args.manh_ratio}, palette={single_manh_pal_text}, "
                    f"ylim={args.ylim if args.ylim is not None else 'auto'}, "
                    f"compression={comp_text}",
                )
            )
        if args.qq_ratio is not None:
            vis_rows.append(
                (
                    "Single QQ",
                    f"ratio={args.qq_ratio}, palette={single_qq_pal_text}, "
                    f"ylim={args.ylim if args.ylim is not None else 'auto'}",
                )
            )
        if args.manh_ratio is not None or args.qq_ratio is not None:
            vis_rows.append(
                (
                    "Single Scatter",
                    f"size={args.scatter_size}, marker={args._postgwas_single_marker}",
                )
            )
        if args._merge_manh_ratio is not None:
            vis_rows.append(
                (
                    "Merged Manhattan",
                    f"ratio={args._merge_manh_ratio}, palette={merge_manh_pal_text}, "
                    f"ylim={args.ylim if args.ylim is not None else 'auto'}",
                )
            )
        if args._merge_qq_ratio is not None:
            vis_rows.append(
                (
                    "Merged QQ",
                    f"ratio={args._merge_qq_ratio}, palette={merge_qq_pal_text}, "
                    f"ylim={args.ylim if args.ylim is not None else 'auto'}",
                )
            )
        if merge_plot_requested:
            vis_rows.append(
                (
                    "Merged Scatter",
                    f"size={args.scatter_size}, markers={','.join(args._postgwas_merge_markers)}",
                )
            )
        if args.ldblock_ratio is None:
            vis_rows.append(("LDBlock", "off"))
        else:
            ld_mode_text = "all SNPs" if args.ldblock_mode == "all" else "threshold SNPs"
            ld_xspan_text = (
                "full width (default)"
                if args.ldblock_xspan is None
                else f"{args.ldblock_xspan[0]:g}-{args.ldblock_xspan[1]:g} (fraction of Manhattan width)"
            )
            ld_pal_text = (
                "default (greys)"
                if args.ldblock_palette_spec is None
                else str(args.ldblock_palette)
            )
            vis_rows.append(
                (
                    "LDBlock",
                    f"ratio={args.ldblock_ratio}, mode={ld_mode_text}, "
                    f"x-span={ld_xspan_text}, palette={ld_pal_text}",
                ),
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

    sections: list[tuple[str, list[tuple[str, object]]]] = [("General", base_rows)]
    if len(merge_map_rows) > 0:
        sections.append(("Merge files", merge_map_rows))
    if vis_rows is None:
        sections.append(("Visualization", [("Status", "disabled")]))
    else:
        sections.append(("Visualization", vis_rows))
    if anno_rows is not None:
        sections.append(("Annotation", anno_rows))
    emit_cli_configuration(
        logger,
        app_title=config_title,
        config_title="POST-GWAS CONFIG",
        host=host_text,
        sections=sections,
        footer_rows=[
            ("Threads", threads_text),
            ("PostGWAS workers", int(getattr(args, "_postgwas_outer_workers", 1))),
        ],
        emit_to_stdout=True,
        line_max_chars=_CONFIG_LINE_MAX_CHARS,
        overflow_mark=_CONFIG_OVERFLOW_MARK,
    )
    _emit_info_to_file_handlers(logger, "")
    _emit_info_to_file_handlers(logger, "[ Command ]")
    _emit_info_to_file_handlers(logger, f"  {_postgwas_invocation_command(argv)}")
    if no_plot_or_anno:
        logger.warning(
            "Warning: No --manh/--qq/--manh-merge/--qq-merge/--ldblock/--ldblock-all/--anno provided. Nothing will be plotted or annotated."
        )

    check_gwas_files = list(args.gwasfile or [])
    checks: list[bool] = [
        ensure_file_exists(logger, f, "GWAS result file") for f in check_gwas_files
    ]
    if args.anno:
        checks.append(ensure_file_exists(logger, args.anno, "Annotation file"))
    if args.vcf:
        checks.append(ensure_file_exists(logger, args.vcf, "Genotype VCF file"))
    if args.hmp:
        checks.append(ensure_file_exists(logger, args.hmp, "Genotype HMP file"))
    if args.geno:
        checks.append(ensure_file_input_exists(logger, args.geno, "Genotype FILE input"))
        if args.ldblock_ratio is not None or args.ldclump_window_bp is not None:
            checks.append(
                ensure_file_input_site_metadata_exists(
                    logger,
                    args.geno,
                    "Genotype FILE site metadata for LD/LDclump",
                )
            )
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, args.bfile, "Genotype PLINK prefix"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Parallel processing of all input files
    # ------------------------------------------------------------------
    if bool(getattr(args, "ldblock_only_mode", False)):
        _run_postgwas_ldblock_only(args, logger)
    else:
        if merge_plot_requested:
            _run_postgwas_merge_manhattan(args, logger)
        if single_plot_requested:
            _run_postgwas_tasks(args, logger)

    # ------------------------------------------------------------------
    # Final logging
    # ------------------------------------------------------------------
    lt = time.localtime()
    endinfo = (
        f"\nFinished. Total wall time: "
        f"{round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    log_success(logger, endinfo)


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
