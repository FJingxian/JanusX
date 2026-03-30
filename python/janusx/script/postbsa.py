# -*- coding: utf-8 -*-
"""
JanusX: Post-BSA Visualization

Examples
--------
  jx postbsa -file bsa.tsv -b1 Bulk1 -b2 Bulk2

  jx postbsa -file bsa.tsv -b1 Bulk1 -b2 Bulk2 -o results -prefix case1

  jx postbsa -file bsa.tsv -b1 Bulk1 -b2 Bulk2 --window 1 --step 0.25 --fmt pdf

Input table
-----------
The input file must contain these columns:
  - CHROM
  - POS
  - {bulk1}.DP / {bulk1}.AD / {bulk1}.GQ
  - {bulk2}.DP / {bulk2}.AD / {bulk2}.GQ
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import logging
import os
import re
import socket
import tempfile
import time
import sys
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Optional
from janusx.gtools.cleaner import chrom_sort_key as _chrom_sort_key

from ._common.log import setup_logging
from ._common.config_render import emit_cli_configuration
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.pathcheck import ensure_all_true, ensure_file_exists, format_path_for_display
from ._common.status import (
    CliStatus,
    format_elapsed,
    log_success,
    print_failure,
    print_success,
    should_animate_status,
)
from ._common.progress import build_rich_progress, rich_progress_available
from ._common.threads import detect_effective_threads

for key in ["MPLBACKEND"]:
    if key in os.environ:
        del os.environ[key]

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["svg.hashsalt"] = "hello"

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

try:
    from janusx.janusx import preprocess_bsa as _preprocess_bsa
except ImportError:
    _preprocess_bsa = None


DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_SUBPLOT_RATIO = 3.0
DEFAULT_WINDOW_MB = 1.0
DEFAULT_STEP_RATIO = 0.5
DEFAULT_TOTAL_DP = (30, 300)
DEFAULT_MIN_DP = 15
DEFAULT_MIN_GQ = 90
DEFAULT_REF_ALLELE_FREQ = 0.2
DEFAULT_DEPTH_DIFFERENCE = 150
DEFAULT_ED_POWER = 4
DEFAULT_CI_LEVELS = (95.0,)
DEFAULT_SNPIDX_SCATTER_MAX_POINTS_PER_CHR = 5000
DEFAULT_RICH_ACTIVE_TASKS = 5
DEFAULT_MIN_CONTIG_LOCI = 500
DEFAULT_CHR_BICOLOR = ("#2ca25f", "#fd8d3c")
SUBPLOT_HEIGHT = 4.5


def parse_ratio(value: str) -> float:
    raw = str(value).strip()
    try:
        if ":" in raw:
            width, height = raw.split(":", 1)
            ratio = float(width) / float(height)
        elif "/" in raw:
            width, height = raw.split("/", 1)
            ratio = float(width) / float(height)
        else:
            ratio = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ratio: {value}") from exc

    if ratio <= 0:
        raise argparse.ArgumentTypeError("ratio must be > 0")
    return ratio


def parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid value: {value}") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_total_dp(value: str) -> tuple[int, int]:
    text = str(value).strip()
    for sep in (":", "-", ","):
        if sep in text:
            left, right = text.split(sep, 1)
            try:
                low = int(left.strip())
                high = int(right.strip())
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid total depth range: {value}"
                ) from exc
            if low < 0 or high < 0 or high < low:
                raise argparse.ArgumentTypeError(
                    "total depth range must satisfy 0 <= low <= high"
                )
            return low, high
    raise argparse.ArgumentTypeError(
        "total depth range must be like 30:300, 30-300 or 30,300"
    )


def parse_ci_percentile(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid CI percentile: {value}") from exc
    if not (0.0 < parsed < 100.0):
        raise argparse.ArgumentTypeError("CI percentile must be within (0, 100).")
    return parsed


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


def _parse_custom_palette(text: str) -> list[str]:
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
                f"Invalid --palette color token: {tok}. "
                "Use #RRGGBB or (R,G,B)."
            ) from e
    if len(colors) == 0:
        raise ValueError("Invalid --palette: empty color list.")
    return colors


def _blend_hex_color(c1: str, c2: str, ratio_to_c2: float) -> str:
    t = float(np.clip(float(ratio_to_c2), 0.0, 1.0))
    rgb1 = np.asarray(mcolors.to_rgb(c1), dtype=float)
    rgb2 = np.asarray(mcolors.to_rgb(c2), dtype=float)
    out = (1.0 - t) * rgb1 + t * rgb2
    return mcolors.to_hex(out)


def _relative_luminance(color: str) -> float:
    r, g, b = mcolors.to_rgb(color)
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def _expand_single_palette_color(base_color: str) -> list[str]:
    base = mcolors.to_hex(mcolors.to_rgba(base_color))
    gray = _relative_luminance(base)
    if gray < 0.5:
        light = _blend_hex_color(base, "#ffffff", 0.85)
        dark = _blend_hex_color(base, "#000000", 0.35)
    else:
        light = _blend_hex_color(base, "#ffffff", 0.35)
        dark = _blend_hex_color(base, "#000000", 0.85)
    return [light, dark]


def _parse_palette_spec(value: object) -> Optional[tuple[str, Any]]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        raise ValueError("Invalid --palette: value is empty.")
    if ";" in text:
        return ("list", _parse_custom_palette(text))
    try:
        plt.get_cmap(text)
        return ("cmap", text)
    except ValueError:
        if text.startswith("(") and text.endswith(")"):
            return ("list", [_parse_rgb_triplet(text)])
        try:
            return ("list", [mcolors.to_hex(mcolors.to_rgba(text))])
        except ValueError as e:
            raise ValueError(
                f"Invalid --palette: {text}. "
                "Use a cmap name (e.g. tab10) or ';'-separated colors."
            ) from e


def _resolve_chr_colors(
    palette_spec: Optional[tuple[str, Any]],
    n_chr: int,
) -> list[str]:
    if n_chr <= 0:
        return []
    if palette_spec is None:
        defaults = [mcolors.to_hex(c) for c in DEFAULT_CHR_BICOLOR]
        return [defaults[i % len(defaults)] for i in range(n_chr)]

    mode, payload = palette_spec
    if mode == "list":
        colors = [mcolors.to_hex(mcolors.to_rgba(c)) for c in list(payload)]
        if len(colors) == 0:
            defaults = [mcolors.to_hex(c) for c in DEFAULT_CHR_BICOLOR]
            return [defaults[i % len(defaults)] for i in range(n_chr)]
        if len(colors) == 1:
            colors = _expand_single_palette_color(colors[0])
        return [colors[i % len(colors)] for i in range(n_chr)]

    cmap_name = str(payload).strip()
    cmap = plt.get_cmap(cmap_name)
    if int(getattr(cmap, "N", 256)) <= 32:
        bins = max(1, int(cmap.N))
        return [mcolors.to_hex(cmap(i % bins)) for i in range(n_chr)]
    return [mcolors.to_hex(cmap(i / max(1, n_chr - 1))) for i in range(n_chr)]


def _fmt_percentile(p: float) -> str:
    v = float(p)
    if np.isclose(v, round(v)):
        return str(int(round(v)))
    return f"{v:g}"


def _log_filter_stage(
    logger: logging.Logger,
    stage: str,
    before: int,
    after: int,
) -> None:
    removed = max(0, int(before) - int(after))
    logger.info(
        f"Filter[{stage}]: kept {int(after)}/{int(before)} (removed {removed})"
    )


def _log_rust_filter_summary(
    logger: logging.Logger,
    stage_rows: Any,
) -> None:
    if not isinstance(stage_rows, (list, tuple)):
        return
    for row in stage_rows:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        stage = str(row[0])
        try:
            kept = int(row[1])
        except Exception:
            continue
        try:
            removed = int(row[2])
        except Exception:
            removed = 0
        logger.info(f"Filter[{stage}]: kept {kept}, removed {removed}")


def _mute_stdout_info_logs(
    logger: logging.Logger,
) -> list[tuple[logging.Handler, int]]:
    muted: list[tuple[logging.Handler, int]] = []
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout:
            muted.append((h, h.level))
            h.setLevel(logging.WARNING)
    return muted


def _restore_handler_levels(muted: list[tuple[logging.Handler, int]]) -> None:
    for h, level in muted:
        h.setLevel(level)


def configure_export_format(output_format: str) -> None:
    output_format = output_format.lower()
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    if output_format == "svg":
        mpl.rcParams["svg.fonttype"] = "none"


def figure_size_from_ratio(rows: int, subplot_ratio: float) -> tuple[float, float]:
    width = SUBPLOT_HEIGHT * subplot_ratio
    height = SUBPLOT_HEIGHT * rows
    return width, height


def normalize_output_prefix(prefix: str, bulk1: str, bulk2: str) -> str:
    text = str(prefix).strip().rstrip(".-_")
    if text == "":
        return text

    candidates = [
        f"{bulk1}vs{bulk2}",
        f"{bulk1}_vs_{bulk2}",
        f"{bulk1}-{bulk2}",
        f"{bulk2}vs{bulk1}",
        f"{bulk2}_vs_{bulk1}",
        f"{bulk2}-{bulk1}",
    ]
    lower_text = text.lower()
    for candidate in candidates:
        lower_candidate = candidate.lower()
        for sep in (".", "_", "-"):
            suffix = f"{sep}{lower_candidate}"
            if lower_text.endswith(suffix):
                return text[: -len(suffix)].rstrip(".-_")
        if lower_text == lower_candidate:
            return ""
    return text


def build_output_stem(out_dir: str, prefix: str, bulk1: str, bulk2: str) -> str:
    clean_prefix = normalize_output_prefix(prefix, bulk1, bulk2)
    if clean_prefix:
        stem = f"{clean_prefix}.{bulk1}vs{bulk2}"
    else:
        stem = f"{bulk1}vs{bulk2}"
    return os.path.join(out_dir, stem)


def strip_known_suffix(path: str) -> str:
    name = Path(path).name
    lower = name.lower()
    if lower.endswith(".tsv.gz"):
        return name[:-7]
    if lower.endswith(".csv.gz"):
        return name[:-7]
    for ext in (".tsv", ".txt", ".csv", ".gz"):
        if lower.endswith(ext):
            return name[: -len(ext)]
    return Path(path).stem


def has_glob_magic(text: str) -> bool:
    return any(ch in text for ch in ["*", "?", "["])


def resolve_input_tables(file_arg: str) -> list[str]:
    raw = str(file_arg).strip()
    if raw == "":
        return []
    if has_glob_magic(raw):
        paths = sorted(glob.glob(raw), key=path_sort_key)
    else:
        paths = [raw]
    out = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            out.append(str(path))
    return out


def dedup_paths_keep_order(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for p in paths:
        key = str(Path(p).resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(str(Path(p)))
    return out


def default_prefix_for_inputs(file_arg: str, files: list[str]) -> str:
    if len(files) <= 1 and not has_glob_magic(file_arg):
        return strip_known_suffix(file_arg)
    return "allchr"


def clean_chr(chr_series: pd.Series) -> pd.Series:
    chr_str = (
        chr_series.astype(str)
        .str.strip()
        .str.replace(r"^(?i:chr)", "", regex=True)
        .str.strip()
    )

    def try_convert(x: str):
        if pd.isna(x) or x == "" or str(x).lower() in ["nan", "na", "null"]:
            return np.nan
        try:
            return int(float(x))
        except (ValueError, TypeError):
            try:
                return float(x)
            except (ValueError, TypeError):
                return x

    return chr_str.apply(try_convert)


def _freeze_sort_key(obj):
    if isinstance(obj, list):
        return tuple(_freeze_sort_key(x) for x in obj)
    if isinstance(obj, tuple):
        return tuple(_freeze_sort_key(x) for x in obj)
    return obj


def chr_sort_key(x) -> tuple[int, object]:
    # Keep postbsa chromosome ordering fully aligned with manhanden.py.
    return _freeze_sort_key(_chrom_sort_key(x))


def is_contig_chr(label: object) -> bool:
    if pd.isna(label):
        return False
    text = str(label).strip().upper()
    compact = text.replace("-", "").replace("_", "")
    if compact.startswith("CONTIG") or compact.startswith("SCAFFOLD"):
        return True
    if compact.startswith("CHRUN") or compact.startswith("UNPLACED"):
        return True
    if "RANDOM" in compact:
        return True
    if text.startswith("NW_") or text.startswith("NT_") or text.startswith("GL"):
        return True
    return False


def is_primary_chr_label(label: object) -> bool:
    if pd.isna(label):
        return False
    text = str(label).strip().upper()
    text = re.sub(r"^CHR", "", text)
    if text in {"X", "Y", "M", "MT"}:
        return True
    return bool(re.fullmatch(r"\d+", text))


def path_sort_key(path_text: str) -> tuple[tuple[int, object], tuple[int, object]]:
    name = Path(path_text).name
    stem = strip_known_suffix(name)
    m = re.search(r"merge\.(.+?)\.snp", stem, flags=re.IGNORECASE)
    token = m.group(1) if m is not None else stem
    return chr_sort_key(token), chr_sort_key(name)


def change_chr_loc(
    df: pd.DataFrame,
    chr_col: str = "chr",
    loc_col: str = "pos",
    interval: float = 1 / 50,
) -> tuple[pd.DataFrame, list[float]]:
    if df.empty:
        return df.copy(), []
    if chr_col not in df.columns or loc_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{chr_col}' and '{loc_col}' columns")

    out = df.copy()
    chr_sorted = sorted(out[chr_col].dropna().unique(), key=chr_sort_key)
    chr_max = out.groupby(chr_col, sort=False)[loc_col].max().reindex(chr_sorted)
    total_loc = chr_max.fillna(0).sum()
    chr_interval = int(total_loc * interval)

    offsets = {}
    running_offset = 0
    for chr_id in chr_sorted:
        offsets[chr_id] = running_offset
        running_offset += chr_max.loc[chr_id] + chr_interval

    out[f"{loc_col}_raw"] = out[loc_col]
    out[loc_col] = out[loc_col] + out[chr_col].map(offsets)
    chr_loc = (
        out.groupby(chr_col, sort=False)[loc_col]
        .mean()
        .reindex(chr_sorted)
        .tolist()
    )
    return out, chr_loc


def compute_g_statistic(df: pd.DataFrame) -> pd.Series:
    eps = 1e-10
    observed = df.to_numpy(dtype=float, copy=False)
    if observed.shape[1] != 4:
        raise ValueError(f"G-test requires 4 columns, got {observed.shape[1]}")

    totals = observed.sum(axis=1)
    valid_rows = totals > 0
    if not np.any(valid_rows):
        return pd.Series(np.nan, index=df.index, dtype=float)

    observed_valid = observed[valid_rows].reshape(-1, 2, 2)
    totals_valid = totals[valid_rows][:, None, None]
    row_sums = observed_valid.sum(axis=2, keepdims=True)
    col_sums = observed_valid.sum(axis=1, keepdims=True)
    expected = (row_sums * col_sums) / (totals_valid + eps)

    observed_valid = np.clip(observed_valid, eps, None)
    expected = np.clip(expected, eps, None)
    g_values = 2 * np.sum(observed_valid * np.log(observed_valid / expected), axis=(1, 2))

    full_result = np.full(len(df), np.nan, dtype=float)
    full_result[valid_rows] = g_values
    return pd.Series(full_result, index=df.index, dtype=float)


def window_nanmean(
    values: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)

    prefix_sum = np.empty(values.size + 1, dtype=float)
    prefix_sum[0] = 0.0
    prefix_sum[1:] = np.cumsum(np.where(finite, values, 0.0))

    prefix_count = np.empty(values.size + 1, dtype=np.int64)
    prefix_count[0] = 0
    prefix_count[1:] = np.cumsum(finite.astype(np.int64))

    totals = prefix_sum[right_idx] - prefix_sum[left_idx]
    counts = prefix_count[right_idx] - prefix_count[left_idx]
    means = np.full(left_idx.size, np.nan, dtype=float)
    valid = counts > 0
    means[valid] = totals[valid] / counts[valid]
    return means, counts


def window_gprime(
    pos: np.ndarray,
    g_values: np.ndarray,
    centers: np.ndarray,
    half_window: float,
) -> np.ndarray:
    gprime = np.full(centers.size, np.nan, dtype=float)
    finite = np.isfinite(g_values)
    if not np.any(finite):
        return gprime

    pos_valid = pos[finite]
    g_valid = g_values[finite]
    left_idx = np.searchsorted(pos_valid, centers - half_window, side="left")
    right_idx = np.searchsorted(pos_valid, centers + half_window, side="right")

    for idx, center in enumerate(centers):
        left = left_idx[idx]
        right = right_idx[idx]
        if right <= left:
            continue
        distances = np.abs(pos_valid[left:right] - center) / half_window
        within = distances <= 1
        if not np.any(within):
            continue
        distances = distances[within]
        weights = (1 - distances**3) ** 3
        weights_sum = weights.sum()
        if weights_sum > 0:
            gprime[idx] = np.dot(weights, g_valid[left:right][within]) / weights_sum

    return gprime


def try_rust_preprocess(
    table_path: str,
    bulk1: str,
    bulk2: str,
    total_dp_threshold: tuple[int, int],
    min_dp: int,
    min_gq: int,
    ref_allele_freq: float,
    depth_difference: int,
    window_mb: float,
    step_mb: float,
    ed_power: int,
    logger: logging.Logger,
) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    if _preprocess_bsa is None:
        logger.warning(
            "Rust BSA backend is not available in the current extension; "
            "falling back to the Python implementation."
        )
        return None

    try:
        with tempfile.TemporaryDirectory(prefix="janusx_postbsa_") as tmpdir:
            tmp_prefix = os.path.join(tmpdir, "postbsa")
            rust_out = _preprocess_bsa(
                input_path=table_path,
                bulk1=bulk1,
                bulk2=bulk2,
                out_prefix=tmp_prefix,
                min_dp=min_dp,
                min_gq=min_gq,
                total_dp_min=total_dp_threshold[0],
                total_dp_max=total_dp_threshold[1],
                ref_allele_freq=ref_allele_freq,
                depth_difference=depth_difference,
                window_mb=window_mb,
                step_mb=step_mb,
                ed_power=ed_power,
            )
            stage_rows = None
            if isinstance(rust_out, (list, tuple)):
                if len(rust_out) < 2:
                    raise RuntimeError("Rust preprocess_bsa returned invalid output.")
                raw_path = rust_out[0]
                smooth_path = rust_out[1]
                if len(rust_out) >= 3:
                    stage_rows = rust_out[2]
                else:
                    logger.info(
                        "Rust backend lacks per-condition filter stats; "
                        "falling back to Python preprocessing for detailed filter logs."
                    )
                    return None
            else:
                raise RuntimeError("Rust preprocess_bsa returned non-tuple output.")
            raw_df = pd.read_csv(raw_path, sep="\t")
            smooth_df = pd.read_csv(smooth_path, sep="\t")
        if stage_rows is not None:
            _log_rust_filter_summary(logger, stage_rows)
        logger.debug(
            f"Rust preprocessing finished: raw={raw_df.shape}, smooth={smooth_df.shape}"
        )
        return raw_df, smooth_df
    except Exception as exc:
        logger.warning(f"Rust preprocessing failed; falling back to Python: {exc}")
        return None


def preprocess_single_table(
    table_path: str,
    bulk1: str,
    bulk2: str,
    total_dp_threshold: tuple[int, int],
    min_dp: int,
    min_gq: int,
    ref_allele_freq: float,
    depth_difference: int,
    window_mb: float,
    step_mb: float,
    ed_power: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rust_result = try_rust_preprocess(
        table_path=table_path,
        bulk1=bulk1,
        bulk2=bulk2,
        total_dp_threshold=total_dp_threshold,
        min_dp=min_dp,
        min_gq=min_gq,
        ref_allele_freq=ref_allele_freq,
        depth_difference=depth_difference,
        window_mb=window_mb,
        step_mb=step_mb,
        ed_power=ed_power,
        logger=logger,
    )
    if rust_result is not None:
        raw_df, smooth_df = rust_result
    else:
        raw_df = load_bsa_in_python(
            table_path=table_path,
            bulk1=bulk1,
            bulk2=bulk2,
            total_dp_threshold=total_dp_threshold,
            min_dp=min_dp,
            min_gq=min_gq,
            ref_allele_freq=ref_allele_freq,
            depth_difference=depth_difference,
            logger=logger,
        )
        bulk1_name = f"{bulk1}.SNPindex"
        bulk2_name = f"{bulk2}.SNPindex"
        deltaindex_name = f"Delta.SNPindex({bulk2}-{bulk1})"
        smooth_df = compute_smooth_df(
            raw_df=raw_df,
            bulk1_name=bulk1_name,
            bulk2_name=bulk2_name,
            deltaindex_name=deltaindex_name,
            ed_power=ed_power,
            window_mb=window_mb,
            step_mb=step_mb,
        )
    raw_df = normalize_position_columns(raw_df)
    smooth_df = normalize_position_columns(smooth_df)
    return raw_df, smooth_df


def load_bsa_in_python(
    table_path: str,
    bulk1: str,
    bulk2: str,
    total_dp_threshold: tuple[int, int],
    min_dp: int,
    min_gq: int,
    ref_allele_freq: float,
    depth_difference: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    logger.info(f"BSA table source: {table_path}")
    header = pd.read_csv(table_path, sep="\t", nrows=1)
    required_cols = [
        "CHROM",
        "POS",
        f"{bulk1}.DP",
        f"{bulk1}.AD",
        f"{bulk1}.GQ",
        f"{bulk2}.DP",
        f"{bulk2}.AD",
        f"{bulk2}.GQ",
    ]
    missing = [col for col in required_cols if col not in header.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = pd.read_csv(
        table_path,
        sep="\t",
        usecols=required_cols,
        low_memory=False,
    )
    logger.info(f"BSA table shape: {df.shape}")

    initial_rows = len(df)
    df = df.dropna(subset=required_cols)
    _log_filter_stage(logger, "drop_na_required_cols", initial_rows, len(df))

    df["CHROM"] = clean_chr(df["CHROM"])
    before = len(df)
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    df = df.dropna(subset=["CHROM", "POS"]).reset_index(drop=True)
    _log_filter_stage(logger, "valid_chr_pos", before, len(df))

    for bulk in [bulk1, bulk2]:
        ad_col = f"{bulk}.AD"
        ad_series = df[ad_col].astype(str)
        alt_values = np.where(
            ad_series.str.contains(",", regex=False),
            ad_series.str.rsplit(",", n=1).str[-1],
            ad_series,
        )
        df[ad_col] = pd.to_numeric(alt_values, errors="coerce").fillna(0).astype(int)

        for suffix in [".DP", ".GQ"]:
            col = f"{bulk}{suffix}"
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    before = len(df)
    df = df.loc[df[f"{bulk1}.DP"].to_numpy() >= min_dp].reset_index(drop=True)
    _log_filter_stage(logger, f"{bulk1}.DP>=minDP({min_dp})", before, len(df))

    before = len(df)
    df = df.loc[df[f"{bulk2}.DP"].to_numpy() >= min_dp].reset_index(drop=True)
    _log_filter_stage(logger, f"{bulk2}.DP>=minDP({min_dp})", before, len(df))

    before = len(df)
    df = df.loc[df[f"{bulk1}.GQ"].to_numpy() >= min_gq].reset_index(drop=True)
    _log_filter_stage(logger, f"{bulk1}.GQ>=minGQ({min_gq})", before, len(df))

    before = len(df)
    df = df.loc[df[f"{bulk2}.GQ"].to_numpy() >= min_gq].reset_index(drop=True)
    _log_filter_stage(logger, f"{bulk2}.GQ>=minGQ({min_gq})", before, len(df))

    total_dp = df[f"{bulk1}.DP"].to_numpy() + df[f"{bulk2}.DP"].to_numpy()
    before = len(df)
    df = df.loc[total_dp >= total_dp_threshold[0]].reset_index(drop=True)
    _log_filter_stage(logger, f"totalDP>=min({total_dp_threshold[0]})", before, len(df))

    total_dp = df[f"{bulk1}.DP"].to_numpy() + df[f"{bulk2}.DP"].to_numpy()
    before = len(df)
    df = df.loc[total_dp <= total_dp_threshold[1]].reset_index(drop=True)
    _log_filter_stage(logger, f"totalDP<=max({total_dp_threshold[1]})", before, len(df))

    dp_diff = np.abs(df[f"{bulk1}.DP"].to_numpy() - df[f"{bulk2}.DP"].to_numpy())
    before = len(df)
    df = df.loc[dp_diff <= depth_difference].reset_index(drop=True)
    _log_filter_stage(logger, f"|DPdiff|<=depthDifference({depth_difference})", before, len(df))

    if df.empty:
        raise ValueError("No loci remain after DP/GQ filtering.")

    bulk1_name = f"{bulk1}.SNPindex"
    bulk2_name = f"{bulk2}.SNPindex"
    deltaindex_name = f"Delta.SNPindex({bulk2}-{bulk1})"

    for bulk in [bulk1, bulk2]:
        dp_col = f"{bulk}.DP"
        ad_col = f"{bulk}.AD"
        divisor = df[dp_col].replace(0, np.nan)
        df[f"{bulk}.SNPindex"] = df[ad_col] / divisor

    freq_keep = ~(
        (
            (df[bulk1_name] < ref_allele_freq)
            & (df[bulk2_name] < ref_allele_freq)
        )
        | (
            (df[bulk1_name] > 1 - ref_allele_freq)
            & (df[bulk2_name] > 1 - ref_allele_freq)
        )
    )
    before = len(df)
    df = df.loc[freq_keep].reset_index(drop=True)
    _log_filter_stage(logger, f"refAlleleFreq({ref_allele_freq})", before, len(df))

    if df.empty:
        raise ValueError("No loci remain after allele-frequency filtering.")

    df[deltaindex_name] = df[bulk2_name] - df[bulk1_name]
    df["ED"] = np.sqrt(2 * (df[bulk2_name] - df[bulk1_name]) ** 2)

    gtest_df = pd.DataFrame(
        {
            "bulk1_ref": df[f"{bulk1}.DP"] - df[f"{bulk1}.AD"],
            "bulk1_alt": df[f"{bulk1}.AD"],
            "bulk2_ref": df[f"{bulk2}.DP"] - df[f"{bulk2}.AD"],
            "bulk2_alt": df[f"{bulk2}.AD"],
        }
    )
    df["G"] = compute_g_statistic(gtest_df)

    out = pd.DataFrame(
        {
            "chr": df["CHROM"],
            "pos": df["POS"].astype(float),
            bulk1_name: df[bulk1_name].astype(float),
            bulk2_name: df[bulk2_name].astype(float),
            deltaindex_name: df[deltaindex_name].astype(float),
            "ED": df["ED"].astype(float),
            "G": df["G"].astype(float),
        }
    )
    out, _ = change_chr_loc(out, chr_col="chr", loc_col="pos")
    out = out.sort_values(["chr", "pos_raw"], key=lambda s: s.map(chr_sort_key) if s.name == "chr" else s)
    out = out.reset_index(drop=True)
    logger.info(f"Filter[kept_rows]: kept {len(out)}")
    return out


def compute_smooth_df(
    raw_df: pd.DataFrame,
    bulk1_name: str,
    bulk2_name: str,
    deltaindex_name: str,
    ed_power: int,
    window_mb: float,
    step_mb: float,
) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "chr",
                "pos",
                "pos_raw",
                bulk1_name,
                bulk2_name,
                deltaindex_name,
                "ED_power",
                "Gprime",
            ]
        )

    window_size = window_mb * 1e6
    half_window = window_size / 2.0
    step_size = step_mb * 1e6
    min_window_snps = max(5, int(window_size * 1e-4))

    frames: list[pd.DataFrame] = []
    chr_list = sorted(raw_df["chr"].dropna().unique(), key=chr_sort_key)
    for chr_id in chr_list:
        chr_df = raw_df.loc[raw_df["chr"] == chr_id].sort_values("pos").copy()
        if chr_df.empty:
            continue

        pos = chr_df["pos"].to_numpy(dtype=float)
        pos_raw = chr_df["pos_raw"].to_numpy(dtype=float)
        if pos[-1] - pos[0] < window_size:
            continue

        centers = np.arange(pos[0] + step_size, pos[-1], step_size, dtype=float)
        if centers.size == 0:
            continue

        left_idx = np.searchsorted(pos, centers - half_window, side="left")
        right_idx = np.searchsorted(pos, centers + half_window, side="right")
        window_counts = right_idx - left_idx
        valid_windows = window_counts >= min_window_snps
        if not np.any(valid_windows):
            continue

        smoothed: dict[str, np.ndarray] = {
            "chr": np.repeat(chr_id, centers.size),
            "pos": centers,
            "pos_raw": centers - (pos[0] - pos_raw[0]),
        }
        for col in [bulk1_name, bulk2_name, deltaindex_name]:
            means, _ = window_nanmean(chr_df[col].to_numpy(dtype=float), left_idx, right_idx)
            means[~valid_windows] = np.nan
            smoothed[col] = means

        ed_means, _ = window_nanmean(
            np.power(chr_df["ED"].to_numpy(dtype=float), ed_power),
            left_idx,
            right_idx,
        )
        ed_means[~valid_windows] = np.nan
        smoothed["ED_power"] = ed_means

        gprime = window_gprime(
            pos,
            chr_df["G"].to_numpy(dtype=float),
            centers,
            half_window,
        )
        gprime[~valid_windows] = np.nan
        smoothed["Gprime"] = gprime

        y_df = pd.DataFrame(smoothed)
        signal_cols = [bulk1_name, bulk2_name, deltaindex_name, "ED_power", "Gprime"]
        y_df = y_df.dropna(subset=signal_cols, how="all")
        if not y_df.empty:
            frames.append(y_df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "chr",
                "pos",
                "pos_raw",
                bulk1_name,
                bulk2_name,
                deltaindex_name,
                "ED_power",
                "Gprime",
            ]
        )
    return pd.concat(frames, axis=0, ignore_index=True)


def save_table(df: pd.DataFrame, path: str, logger: logging.Logger, label: str) -> None:
    df.to_csv(path, sep="\t", index=False)
    log_success(logger, f"{label} saved: {format_path_for_display(path)}")


def normalize_position_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["pos", "pos_raw"]:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            if values.notna().all():
                out[col] = np.rint(values).astype(np.int64)
            else:
                out[col] = pd.array(np.rint(values), dtype="Int64")
    return out


def sort_by_chr_pos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "chr" not in df.columns:
        return df
    out = df.copy()
    chr_values = [x for x in pd.unique(out["chr"]) if not pd.isna(x)]
    chr_sorted = sorted(chr_values, key=chr_sort_key)
    chr_rank = {label: idx for idx, label in enumerate(chr_sorted)}
    out["_chr_rank"] = out["chr"].map(chr_rank)
    out["_chr_rank"] = pd.to_numeric(out["_chr_rank"], errors="coerce").fillna(len(chr_sorted)).astype(np.int64)
    pos_col = "pos_raw" if "pos_raw" in out.columns else "pos" if "pos" in out.columns else None
    order_cols = ["_chr_rank"]
    if pos_col is not None:
        order_cols.append(pos_col)
    out = out.sort_values(order_cols, kind="mergesort").drop(columns=["_chr_rank"])
    return out.reset_index(drop=True)


def filter_low_loci_contigs(
    raw_df: pd.DataFrame,
    smooth_df: pd.DataFrame,
    logger: logging.Logger,
    min_loci: int = DEFAULT_MIN_CONTIG_LOCI,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_df.empty or "chr" not in raw_df.columns:
        return raw_df, smooth_df

    counts = raw_df.groupby("chr", sort=False).size()
    if counts.empty:
        return raw_df, smooth_df

    sorted_counts = counts.sort_values(ascending=False)
    top_n = max(3, min(10, len(sorted_counts)))
    baseline = float(np.median(sorted_counts.iloc[:top_n].to_numpy(dtype=float)))
    dynamic_threshold = max(int(min_loci), int(baseline * 0.1))

    drop_set: set[object] = set()
    for chr_id, n_loci in counts.items():
        if is_contig_chr(chr_id):
            drop_set.add(chr_id)
            continue
        # Non-primary labels (e.g., scaffold/NW-like) are treated as contig-like
        # when they are clearly below autosome-scale loci counts.
        if (not is_primary_chr_label(chr_id)) and int(n_loci) < dynamic_threshold:
            drop_set.add(chr_id)

    if len(drop_set) == 0:
        logger.info(
            "Contig loci filter: no contig-like chromosomes removed before plotting "
            f"(threshold={dynamic_threshold}, baseline={int(baseline)})."
        )
        return raw_df, smooth_df

    drop_counts = counts[counts.index.isin(drop_set)]
    raw_out = raw_df.loc[~raw_df["chr"].isin(drop_set)].reset_index(drop=True)
    if "chr" in smooth_df.columns:
        smooth_out = smooth_df.loc[~smooth_df["chr"].isin(drop_set)].reset_index(drop=True)
    else:
        smooth_out = smooth_df
    logger.info(
        "Contig loci filter: removed "
        f"{len(drop_set)} contig-like chromosomes ({int(drop_counts.sum())} raw loci), "
        f"threshold={dynamic_threshold}, baseline={int(baseline)}."
    )
    preview = ", ".join([str(x) for x in list(drop_counts.sort_values().index[:8])])
    if preview:
        logger.info(f"Filtered contig-like labels (first 8): {preview}")
    return raw_out, smooth_out


def reoffset_global_chr_positions(
    raw_df: pd.DataFrame,
    smooth_df: pd.DataFrame,
    interval: float = 1 / 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_df.empty or "chr" not in raw_df.columns:
        return raw_df, smooth_df

    raw_base_col = "pos_raw" if "pos_raw" in raw_df.columns else "pos"
    if raw_base_col not in raw_df.columns:
        return raw_df, smooth_df

    raw_base = pd.to_numeric(raw_df[raw_base_col], errors="coerce")
    key_df = pd.DataFrame({"chr": raw_df["chr"], "base": raw_base}).dropna()
    if key_df.empty:
        return raw_df, smooth_df

    chr_sorted = sorted(key_df["chr"].dropna().unique(), key=chr_sort_key)
    chr_max = key_df.groupby("chr", sort=False)["base"].max().reindex(chr_sorted).fillna(0.0)
    total_loc = float(chr_max.sum())
    chr_interval = int(total_loc * interval)

    offsets: dict[object, float] = {}
    running = 0.0
    for chr_id in chr_sorted:
        offsets[chr_id] = running
        running += float(chr_max.loc[chr_id]) + chr_interval

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "chr" not in df.columns:
            return df
        out = df.copy()
        base_col = "pos_raw" if "pos_raw" in out.columns else "pos"
        if base_col not in out.columns:
            return out
        base = pd.to_numeric(out[base_col], errors="coerce")
        if "pos_raw" not in out.columns:
            out["pos_raw"] = base
        mapped = out["chr"].map(offsets)
        new_pos = base + mapped
        old_pos = pd.to_numeric(out["pos"], errors="coerce") if "pos" in out.columns else base
        out["pos"] = new_pos.where(new_pos.notna(), old_pos)
        return out

    return _apply(raw_df), _apply(smooth_df)


def preprocess_tables_parallel(
    input_files: list[str],
    worker_count: int,
    bulk1: str,
    bulk2: str,
    total_dp_threshold: tuple[int, int],
    min_dp: int,
    min_gq: int,
    ref_allele_freq: float,
    depth_difference: int,
    window_mb: float,
    step_mb: float,
    ed_power: int,
    logger: logging.Logger,
    use_spinner: bool,
    bulk1_name: str,
    bulk2_name: str,
    deltaindex_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_parts: list[pd.DataFrame] = []
    smooth_parts: list[pd.DataFrame] = []
    errors: list[str] = []

    def _submit(executor: cf.ThreadPoolExecutor, table_path: str) -> cf.Future:
        return executor.submit(
            preprocess_single_table,
            table_path,
            bulk1,
            bulk2,
            total_dp_threshold,
            min_dp,
            min_gq,
            ref_allele_freq,
            depth_difference,
            window_mb,
            step_mb,
            ed_power,
            logger,
        )

    def _is_valid_concat_part(df: pd.DataFrame) -> bool:
        if not isinstance(df, pd.DataFrame):
            return False
        if df.empty:
            return False
        return not bool(df.isna().all(axis=None))

    if rich_progress_available():
        muted_console_handlers: list[tuple[logging.Handler, int]] = []
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout:
                muted_console_handlers.append((h, h.level))
                h.setLevel(logging.WARNING)
        total_start_ts = time.monotonic()
        n_total = len(input_files)
        file_to_idx = {f: i for i, f in enumerate(input_files, start=1)}
        basenames = [os.path.basename(f) for f in input_files]
        idx_width = len(str(n_total))
        name_width = max((len(x) for x in basenames), default=0)
        task_start_ts: dict[str, float] = {}
        progress = build_rich_progress(
            description_template="Task {task.fields[task_label]}: {task.fields[file_pad]}",
            show_bar=False,
            show_percentage=False,
            show_elapsed=False,
            show_remaining=False,
            finished_text=" ",
            transient=True,
        ) if should_animate_status("Loading chromosome tables...") else None
        try:
            with (progress if progress is not None else nullcontext()):
                with cf.ThreadPoolExecutor(max_workers=worker_count) as ex:
                    fut_map: dict[cf.Future, str] = {}
                    task_map: dict[cf.Future, int] = {}
                    display_limit = max(1, min(DEFAULT_RICH_ACTIVE_TASKS, n_total))

                    def _submit_only(table_path: str) -> None:
                        fut = _submit(ex, table_path)
                        fut_map[fut] = table_path
                        task_start_ts[table_path] = time.monotonic()

                    def _fill_visible_slots() -> None:
                        if len(task_map) >= display_limit:
                            return
                        pending = sorted(
                            [
                                fut
                                for fut in fut_map.keys()
                                if (fut not in task_map) and (not fut.done())
                            ],
                            key=lambda x: file_to_idx.get(fut_map[x], 0),
                        )
                        for fut in pending[: display_limit - len(task_map)]:
                            table_path = fut_map[fut]
                            idx = file_to_idx.get(table_path, 0)
                            if progress is None:
                                continue
                            task_map[fut] = progress.add_task(
                                description="",
                                total=None,
                                task_label=f"{idx:>{idx_width}}/{n_total}",
                                file_pad=os.path.basename(table_path).ljust(name_width),
                            )

                    for table_path in input_files:
                        _submit_only(table_path)
                    _fill_visible_slots()

                    while len(fut_map) > 0:
                        done, _ = cf.wait(
                            list(fut_map.keys()),
                            timeout=0.1,
                            return_when=cf.FIRST_COMPLETED,
                        )
                        if len(done) == 0:
                            continue
                        for fut in sorted(done, key=lambda x: file_to_idx.get(fut_map[x], 0)):
                            table_path = fut_map.pop(fut)
                            tid = task_map.pop(fut, None)
                            if tid is not None:
                                try:
                                    if progress is not None:
                                        progress.remove_task(tid)
                                except Exception:
                                    pass
                            elapsed = format_elapsed(
                                time.monotonic()
                                - task_start_ts.get(table_path, time.monotonic())
                            )
                            try:
                                raw_part, smooth_part = fut.result()
                                raw_parts.append(raw_part)
                                smooth_parts.append(smooth_part)
                                logger.info(
                                    f"Finished {table_path}: "
                                    f"raw={raw_part.shape}, smooth={smooth_part.shape}"
                                )
                            except Exception as exc:
                                errors.append(f"{table_path}: {exc}")
                                idx = file_to_idx.get(table_path, 0)
                                print_failure(
                                    f"Task {idx}/{n_total}: "
                                    f"{os.path.basename(table_path)} ...Failed [{elapsed}]"
                                )
                        _fill_visible_slots()
        finally:
            for h, level in muted_console_handlers:
                h.setLevel(level)

        if errors:
            raise RuntimeError("Failed chromosome tables:\n- " + "\n- ".join(errors))
        total_elapsed = format_elapsed(time.monotonic() - total_start_ts)
        print_success(
            f"Task {len(raw_parts)}/{n_total} ...Finished [{total_elapsed}]",
            force_color=True,
        )
    else:
        with CliStatus(
            f"Preprocessing {len(input_files)} chromosome tables...",
            enabled=use_spinner,
        ) as task:
            try:
                with cf.ThreadPoolExecutor(max_workers=worker_count) as ex:
                    fut_map = {_submit(ex, table_path): table_path for table_path in input_files}
                    for fut in cf.as_completed(fut_map):
                        table_path = fut_map[fut]
                        try:
                            raw_part, smooth_part = fut.result()
                            raw_parts.append(raw_part)
                            smooth_parts.append(smooth_part)
                            logger.info(
                                f"Finished {table_path}: "
                                f"raw={raw_part.shape}, smooth={smooth_part.shape}"
                            )
                        except Exception as exc:
                            errors.append(f"{table_path}: {exc}")
                if errors:
                    raise RuntimeError("Failed chromosome tables:\n- " + "\n- ".join(errors))
            except Exception:
                task.fail("Preprocessing chromosome tables ...Failed")
                raise
            task.complete("Preprocessing chromosome tables ...Finished")

    raw_parts = [x for x in raw_parts if _is_valid_concat_part(x)]
    smooth_parts = [x for x in smooth_parts if _is_valid_concat_part(x)]

    if not raw_parts:
        raise RuntimeError("No valid loci remained across all chromosome tables.")

    raw_df = pd.concat(raw_parts, axis=0, ignore_index=True)
    smooth_df = (
        pd.concat(smooth_parts, axis=0, ignore_index=True)
        if smooth_parts
        else pd.DataFrame(
            columns=[
                "chr",
                "pos",
                "pos_raw",
                bulk1_name,
                bulk2_name,
                deltaindex_name,
                "ED_power",
                "Gprime",
            ]
        )
    )
    return raw_df, smooth_df


def downsample_scatter_points(
    df: pd.DataFrame,
    max_points: int = DEFAULT_SNPIDX_SCATTER_MAX_POINTS_PER_CHR,
) -> pd.DataFrame:
    if df.shape[0] <= max_points:
        return df
    keep_idx = np.linspace(0, df.shape[0] - 1, num=max_points, dtype=int)
    return df.iloc[keep_idx].copy()


def plot_bsa(
    raw_df: pd.DataFrame,
    smooth_df: pd.DataFrame,
    bulk1_name: str,
    bulk2_name: str,
    deltaindex_name: str,
    output_stem: str,
    output_format: str,
    subplot_ratio: float,
    ed_power: int,
    window_mb: float,
    ci_levels: list[float],
    palette_spec: Optional[tuple[str, Any]],
    logger: logging.Logger,
) -> None:
    if raw_df.empty:
        logger.warning("No valid BSA loci remain after filtering; skip plotting.")
        return
    if smooth_df.empty:
        logger.warning("No sliding-window results remain; skip plotting.")
        return

    raw_df = raw_df.copy()
    smooth_df = smooth_df.copy()
    if "chr" in raw_df.columns:
        contig_mask_raw = raw_df["chr"].map(is_contig_chr)
        removed_contig_n = int(contig_mask_raw.sum())
        if removed_contig_n > 0:
            raw_df = raw_df.loc[~contig_mask_raw].reset_index(drop=True)
            if "chr" in smooth_df.columns:
                smooth_df = smooth_df.loc[~smooth_df["chr"].map(is_contig_chr)].reset_index(
                    drop=True
                )
            logger.info(
                f"Excluded contigs from final figures: {removed_contig_n} raw loci "
                "from Contig* chromosomes."
            )
    if raw_df.empty:
        logger.warning("Only contig loci remain after filtering; skip plotting.")
        return
    if smooth_df.empty:
        logger.warning("No non-contig sliding-window results remain; skip plotting.")
        return
    raw_df["ED_power"] = np.power(raw_df["ED"].to_numpy(dtype=float), ed_power)

    chr_list = sorted(raw_df["chr"].dropna().unique(), key=chr_sort_key)
    if not chr_list:
        logger.warning("No chromosome information available after preprocessing.")
        return

    chrloc = (
        raw_df.groupby("chr", sort=False)["pos"]
        .mean()
        .reindex(chr_list)
        .tolist()
    )
    raw_groups = {
        chr_id: chr_df.sort_values("pos").copy()
        for chr_id, chr_df in raw_df.groupby("chr", sort=False)
    }
    smooth_groups = {
        chr_id: chr_df.sort_values("pos").copy()
        for chr_id, chr_df in smooth_df.groupby("chr", sort=False)
    }

    ci_levels = sorted({float(x) for x in ci_levels if 0.0 < float(x) < 100.0})
    if len(ci_levels) == 0:
        ci_levels = [float(DEFAULT_CI_LEVELS[0])]
    max_ci = float(max(ci_levels))

    ed_thresholds: dict[float, float] = {
        ci: float(np.nanpercentile(raw_df["ED_power"], ci)) for ci in ci_levels
    }
    delta_upper_thresholds: dict[float, float] = {
        ci: float(np.nanpercentile(raw_df[deltaindex_name], ci)) for ci in ci_levels
    }
    delta_lower_thresholds: dict[float, float] = {
        ci: float(np.nanpercentile(raw_df[deltaindex_name], 100.0 - ci)) for ci in ci_levels
    }

    finite_gprime = (
        smooth_df["Gprime"]
        .replace([-np.inf, np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    gprime_thresholds: dict[float, float] = {}
    for ci in ci_levels:
        if finite_gprime.size > 0:
            gprime_thresholds[ci] = float(np.nanpercentile(finite_gprime, ci))
        else:
            gprime_thresholds[ci] = float("nan")

    logger.info(
        "Threshold-region filtering uses max CI percentile: "
        f"P{_fmt_percentile(max_ci)}"
    )
    for ci in ci_levels:
        ci_txt = _fmt_percentile(ci)
        low_txt = _fmt_percentile(100.0 - ci)
        logger.info(f"Threshold of ED^{ed_power} (P{ci_txt}): {ed_thresholds[ci]:.4f}")
        logger.info(
            f"Threshold of Delta-SNPindex (P{low_txt},P{ci_txt}): "
            f"{delta_lower_thresholds[ci]:.4f}, {delta_upper_thresholds[ci]:.4f}"
        )
        if np.isfinite(gprime_thresholds[ci]):
            logger.info(f"Threshold of Gprime (P{ci_txt}): {gprime_thresholds[ci]:.4f}")
        else:
            logger.info(f"Threshold of Gprime (P{ci_txt}): NA")

    window_half = int(window_mb * 1e6 / 2.0)
    above_threshold = []
    ed_cut = ed_thresholds[max_ci]
    delta_upper_cut = delta_upper_thresholds[max_ci]
    delta_lower_cut = delta_lower_thresholds[max_ci]
    for chr_id in chr_list:
        chr_smooth = smooth_groups.get(chr_id)
        if chr_smooth is None or chr_smooth.empty:
            continue
        threshold_mask = (
            np.isfinite(chr_smooth["ED_power"])
            & np.isfinite(chr_smooth[deltaindex_name])
            & (
                (chr_smooth["ED_power"] >= ed_cut)
                | (chr_smooth[deltaindex_name] >= delta_upper_cut)
                | (chr_smooth[deltaindex_name] <= delta_lower_cut)
            )
        )
        for center_raw, ed_val, delta_val in zip(
            chr_smooth.loc[threshold_mask, "pos_raw"].to_numpy(dtype=float),
            chr_smooth.loc[threshold_mask, "ED_power"].to_numpy(dtype=float),
            chr_smooth.loc[threshold_mask, deltaindex_name].to_numpy(dtype=float),
        ):
            direction = "upper" if delta_val >= delta_upper_cut else "lower"
            center = int(center_raw)
            above_threshold.append(
                [chr_id, center - window_half, center + window_half, ed_val, delta_val, direction]
            )

    if above_threshold:
        threshold_df = pd.DataFrame(
            above_threshold,
            columns=["Chr", "start", "end", f"ED{ed_power}", "deltaSNPindex", "direction"],
        ).round(4)
        threshold_path = f"{output_stem}.thr.tsv"
        threshold_df.to_csv(threshold_path, sep="\t", index=False)
        log_success(logger, f"Threshold regions saved: {format_path_for_display(threshold_path)}")

    x_max = raw_df["pos"].max()
    bulk1_short = bulk1_name.split(".")[0]
    bulk2_short = bulk2_name.split(".")[0]

    plt.rc("font", family="DejaVu Sans")
    fig_snp, (ax_bulk1, ax_bulk2) = plt.subplots(
        2, 1, figsize=figure_size_from_ratio(2, subplot_ratio), dpi=300, sharex=True
    )
    fig_stats, (ax_delta, ax_ed, ax_gprime) = plt.subplots(
        3, 1, figsize=figure_size_from_ratio(3, subplot_ratio), dpi=300, sharex=True
    )

    chr_colors = _resolve_chr_colors(palette_spec, len(chr_list))
    plotted = False
    scatter_total = 0
    scatter_kept = 0
    for idx, chr_id in enumerate(chr_list):
        chr_raw = raw_groups.get(chr_id)
        chr_smooth = smooth_groups.get(chr_id)
        if chr_raw is None or chr_raw.empty or chr_smooth is None or chr_smooth.empty:
            continue
        plotted = True
        color = chr_colors[idx % len(chr_colors)] if len(chr_colors) > 0 else "black"
        chr_raw_scatter = downsample_scatter_points(chr_raw)
        scatter_total += chr_raw.shape[0]
        scatter_kept += chr_raw_scatter.shape[0]

        ax_bulk1.scatter(
            chr_raw_scatter["pos"],
            chr_raw_scatter[bulk1_name],
            s=1,
            color=color,
            alpha=0.3,
            edgecolors="none",
            rasterized=True,
        )
        ax_bulk1.plot(chr_smooth["pos"], chr_smooth[bulk1_name], color="black", linewidth=1.2)

        ax_bulk2.scatter(
            chr_raw_scatter["pos"],
            chr_raw_scatter[bulk2_name],
            s=1,
            color=color,
            alpha=0.3,
            edgecolors="none",
            rasterized=True,
        )
        ax_bulk2.plot(chr_smooth["pos"], chr_smooth[bulk2_name], color="black", linewidth=1.2)

        ax_delta.plot(chr_smooth["pos"], chr_smooth[deltaindex_name], color=color, linewidth=1)
        ax_ed.plot(chr_smooth["pos"], chr_smooth["ED_power"], color=color, linewidth=1)
        valid_gprime = chr_smooth["Gprime"].replace([-np.inf, np.inf], np.nan)
        ax_gprime.plot(chr_smooth["pos"], valid_gprime, color=color, linewidth=1)

    if not plotted:
        plt.close(fig_snp)
        plt.close(fig_stats)
        logger.warning("No chromosome retained enough windows for plotting.")
        return
    if scatter_kept < scatter_total:
        logger.info(
            "Downsampled SNP-index scatter points: "
            f"kept {scatter_kept} / {scatter_total} raw SNPs."
        )

    ax_bulk1.set_title(f"BSA of {bulk1_short} and {bulk2_short}", fontsize=20, fontweight="bold")
    ax_bulk1.axhline(y=0.5, color="grey", linestyle="dashed", linewidth=1)
    ax_bulk2.axhline(y=0.5, color="grey", linestyle="dashed", linewidth=1)
    ax_bulk1.set_ylabel(f"SNP-index of {bulk1_short}", fontsize=14, fontweight="bold")
    ax_bulk2.set_ylabel(f"SNP-index of {bulk2_short}", fontsize=14, fontweight="bold")
    ax_bulk1.set_ylim(0, 1)
    ax_bulk2.set_ylim(0, 1)
    ax_bulk1.set_xlim(0, x_max)
    ax_bulk2.set_xlim(0, x_max)
    ax_bulk1.set_xticks([])
    ax_bulk2.set_xticks(chrloc)
    ax_bulk2.set_xticklabels(chr_list)
    ax_bulk2.set_xlabel("Chromosome", fontsize=14, fontweight="bold")
    fig_snp.align_labels()
    fig_snp.tight_layout()

    ci_line_colors = [mcolors.to_hex(plt.get_cmap("tab10")(i % 10)) for i in range(len(ci_levels))]
    ax_delta.axhline(y=0, color="grey", linestyle="dashed", linewidth=1)
    for idx, ci in enumerate(ci_levels):
        line_color = ci_line_colors[idx]
        ax_delta.axhline(y=delta_upper_thresholds[ci], color=line_color, linewidth=0.9, alpha=0.95)
        ax_delta.axhline(y=delta_lower_thresholds[ci], color=line_color, linewidth=0.9, alpha=0.95)
        ax_ed.axhline(y=ed_thresholds[ci], color=line_color, linewidth=0.9, alpha=0.95)
        if np.isfinite(gprime_thresholds[ci]):
            ax_gprime.axhline(y=gprime_thresholds[ci], color=line_color, linewidth=0.9, alpha=0.95)
    ax_delta.set_ylabel(f"Delta SNP-index ({bulk2_short}-{bulk1_short})", fontsize=14, fontweight="bold")
    ax_ed.set_ylabel(f"ED^{ed_power}", fontsize=14, fontweight="bold")
    ax_gprime.set_ylabel("Gprime", fontsize=14, fontweight="bold")
    for ax in [ax_delta, ax_ed, ax_gprime]:
        ax.set_xlim(0, x_max)
    ax_delta.set_xticks([])
    ax_ed.set_xticks([])
    ax_gprime.set_xticks(chrloc)
    ax_gprime.set_xticklabels(chr_list)
    ax_gprime.set_xlabel("Chromosome", fontsize=14, fontweight="bold")
    fig_stats.align_labels()
    fig_stats.tight_layout()

    configure_export_format(output_format)
    snp_out = f"{output_stem}.snpidx.{output_format}"
    stats_out = f"{output_stem}.bsa.{output_format}"
    fig_snp.savefig(snp_out, dpi=300, bbox_inches="tight", transparent=False, facecolor="white")
    fig_stats.savefig(stats_out, dpi=300, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig_snp)
    plt.close(fig_stats)
    log_success(logger, f"Figure saved: {format_path_for_display(snp_out)}")
    log_success(logger, f"Figure saved: {format_path_for_display(stats_out)}")


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx postbsa",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx postbsa -file bsa.tsv -b1 Bulk1 -b2 Bulk2",
            "jx postbsa -file '4.merge/Merge.*.SNP.tsv' -b1 Bulk1 -b2 Bulk2 -t 8",
        ]),
    )

    required_group = parser.add_argument_group("Required Arguments")
    required_group.add_argument(
        "-file",
        "--file",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Input BSA table (tab-delimited), or a glob pattern such as "
            "`Merge.*.SNP.tsv` for per-chromosome files."
        ),
    )
    required_group.add_argument(
        "-b1",
        "--bulk1",
        type=str,
        required=True,
        help="Bulk-1 sample prefix used in input columns.",
    )
    required_group.add_argument(
        "-b2",
        "--bulk2",
        type=str,
        required=True,
        help="Bulk-2 sample prefix used in input columns.",
    )

    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-window",
        "--window",
        type=parse_positive_float,
        default=DEFAULT_WINDOW_MB,
        help="Sliding-window size in Mb (default: %(default)s).",
    )
    optional_group.add_argument(
        "-step",
        "--step",
        type=parse_positive_float,
        default=None,
        help="Sliding-window step in Mb (default: window / 2).",
    )
    optional_group.add_argument(
        "-ratio",
        "--ratio",
        type=parse_ratio,
        default=DEFAULT_SUBPLOT_RATIO,
        help="Subplot width/height ratio, e.g. 3, 3:1, 16/5 (default: %(default)s).",
    )
    optional_group.add_argument(
        "-fmt",
        "--fmt",
        dest="format",
        type=str,
        choices=["png", "pdf", "svg", "tif"],
        default=DEFAULT_OUTPUT_FORMAT,
        help="Output figure format (default: %(default)s).",
    )
    optional_group.add_argument(
        "-minDP",
        "--min-dp",
        dest="min_dp",
        type=parse_positive_int,
        default=DEFAULT_MIN_DP,
        help="Minimum per-bulk DP (default: %(default)s).",
    )
    optional_group.add_argument(
        "-minGQ",
        "--min-gq",
        dest="min_gq",
        type=parse_positive_int,
        default=DEFAULT_MIN_GQ,
        help="Minimum per-bulk GQ (default: %(default)s).",
    )
    optional_group.add_argument(
        "-totalDP",
        "--total-dp",
        dest="total_dp",
        type=parse_total_dp,
        default=DEFAULT_TOTAL_DP,
        help="Total DP range, e.g. 30:300 (default: %(default)s).",
    )
    optional_group.add_argument(
        "-refAlleleFreq",
        "--ref-allele-freq",
        dest="ref_allele_freq",
        type=float,
        default=DEFAULT_REF_ALLELE_FREQ,
        help="Reference-allele frequency filter threshold (default: %(default)s).",
    )
    optional_group.add_argument(
        "-depthDifference",
        "--depth-difference",
        dest="depth_difference",
        type=parse_positive_int,
        default=DEFAULT_DEPTH_DIFFERENCE,
        help="Maximum depth difference between bulks (default: %(default)s).",
    )
    optional_group.add_argument(
        "-ed",
        "--ed-power",
        dest="ed_power",
        type=parse_positive_int,
        default=DEFAULT_ED_POWER,
        help="Power used for ED thresholding and smoothing (default: %(default)s).",
    )
    optional_group.add_argument(
        "-ci",
        "--ci",
        dest="ci",
        action="append",
        type=parse_ci_percentile,
        default=None,
        help=(
            "Percentile threshold line(s) for Delta-SNPindex/ED/Gprime. "
            "Can be provided multiple times, e.g. -ci 95 -ci 99 "
            "(default: 95). Filtering of threshold regions uses the maximum CI value."
        ),
    )
    optional_group.add_argument(
        "-palette",
        "--palette",
        type=str,
        default=None,
        help=(
            "Chromosome color palette. Supports cmap names (e.g. tab10, tab20) "
            "or ';'-separated colors (e.g. #1f77b4;#ff7f0e). "
            "A single color is auto-expanded to a two-color style."
        ),
    )
    optional_group.add_argument(
        "-o",
        "--out",
        type=str,
        default=".",
        help="Output directory (default: %(default)s).",
    )
    optional_group.add_argument(
        "-prefix",
        "--prefix",
        type=str,
        default=None,
        help="Output prefix. Defaults to the input filename stem.",
    )
    optional_group.add_argument(
        "-t",
        "--thread",
        type=int,
        default=detect_effective_threads(),
        help=(
            "Number of CPU threads (default: %(default)s). "
            "For glob input, this is also the max parallel chromosome jobs."
        ),
    )
    return parser


def main() -> None:
    t_start = time.time()
    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())
    parser = build_parser()
    args = parser.parse_args()

    args.format = str(args.format).lower()
    if not (0.0 <= args.ref_allele_freq <= 0.5):
        raise ValueError("--ref-allele-freq must be within [0, 0.5].")
    if args.step is None:
        args.step = args.window * DEFAULT_STEP_RATIO
    if args.ci is None or len(args.ci) == 0:
        args.ci = [float(DEFAULT_CI_LEVELS[0])]
    args.ci = sorted({float(v) for v in args.ci if 0.0 < float(v) < 100.0})
    if len(args.ci) == 0:
        args.ci = [float(DEFAULT_CI_LEVELS[0])]

    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    thread_capped = False
    if args.thread <= 0:
        args.thread = int(detected_threads)
    if int(args.thread) > int(detected_threads):
        thread_capped = True
        args.thread = int(detected_threads)

    file_args = list(args.file) if isinstance(args.file, list) else [str(args.file)]
    input_files: list[str] = []
    if len(file_args) == 1:
        input_files = resolve_input_tables(file_args[0])
    else:
        for token in file_args:
            token_matches = resolve_input_tables(token)
            if len(token_matches) == 0 and Path(token).is_file():
                token_matches = [token]
            input_files.extend(token_matches)
        input_files = dedup_paths_keep_order(input_files)
    if len(input_files) > 1:
        input_files = sorted(input_files, key=path_sort_key)
    input_hint = " ".join(file_args)

    args.out = args.out if args.out else "."
    os.makedirs(args.out, mode=0o755, exist_ok=True)

    if args.prefix is None:
        args.prefix = default_prefix_for_inputs(input_hint, input_files)
    output_stem = build_output_stem(args.out, args.prefix, args.bulk1, args.bulk2)

    log_path = f"{output_stem}.postbsa.log"
    logger = setup_logging(log_path)
    if thread_capped:
        logger.warning(
            f"Warning: Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(args.thread)}."
        )
    try:
        args.palette_spec = _parse_palette_spec(args.palette)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)

    if len(input_files) == 0:
        logger.error(f"No input BSA table matched: {input_hint}")
        raise SystemExit(1)

    if len(input_files) > 1:
        worker_count = max(1, min(int(args.thread), len(input_files)))
        os.environ["RAYON_NUM_THREADS"] = "1"
    else:
        worker_count = 1
        os.environ["RAYON_NUM_THREADS"] = str(args.thread)

    emit_cli_configuration(
        logger,
        app_title="JanusX - Post-BSA",
        config_title="POST-BSA CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "General",
                [
                    ("Input file", input_hint),
                    ("Input tables", len(input_files)),
                    ("Bulk1", args.bulk1),
                    ("Bulk2", args.bulk2),
                    ("Window (Mb)", args.window),
                    ("Step (Mb)", args.step),
                    ("Plot ratio", args.ratio),
                    ("Format", args.format),
                    ("Min DP", args.min_dp),
                    ("Min GQ", args.min_gq),
                    ("Total DP range", f"{args.total_dp[0]}-{args.total_dp[1]}"),
                    ("Ref allele freq", args.ref_allele_freq),
                    ("Depth difference", args.depth_difference),
                    ("ED power", args.ed_power),
                    ("CI percentiles", ",".join(_fmt_percentile(v) for v in args.ci)),
                    ("Palette", str(args.palette) if args.palette is not None else "default bicolor"),
                ],
            )
        ],
        footer_rows=[
            ("Threads", f"{args.thread} ({detected_threads} available)"),
            ("Parallel jobs", worker_count if len(input_files) > 1 else 1),
            ("Output stem", output_stem),
        ],
        line_max_chars=60,
    )

    checks = [ensure_file_exists(logger, p, "Input BSA table") for p in input_files]
    if not ensure_all_true(checks):
        raise SystemExit(1)
    if len(input_files) > 1:
        logger.info(
            f"Glob mode: detected {len(input_files)} chromosome tables, "
            f"running with {worker_count} parallel jobs."
        )
        if args.thread > 1:
            logger.info("Glob mode sets RAYON_NUM_THREADS=1 to avoid nested oversubscription.")

    bulk1_name = f"{args.bulk1}.SNPindex"
    bulk2_name = f"{args.bulk2}.SNPindex"
    deltaindex_name = f"Delta.SNPindex({args.bulk2}-{args.bulk1})"

    raw_df: pd.DataFrame
    smooth_df: pd.DataFrame
    preprocess_muted_handlers: list[tuple[logging.Handler, int]] = []
    if use_spinner:
        preprocess_muted_handlers = _mute_stdout_info_logs(logger)
    if len(input_files) == 1:
        try:
            with CliStatus("Preprocessing BSA table...", enabled=use_spinner) as task:
                try:
                    raw_df, smooth_df = preprocess_single_table(
                        table_path=input_files[0],
                        bulk1=args.bulk1,
                        bulk2=args.bulk2,
                        total_dp_threshold=args.total_dp,
                        min_dp=args.min_dp,
                        min_gq=args.min_gq,
                        ref_allele_freq=args.ref_allele_freq,
                        depth_difference=args.depth_difference,
                        window_mb=args.window,
                        step_mb=args.step,
                        ed_power=args.ed_power,
                        logger=logger,
                    )
                except Exception:
                    task.fail("Preprocessing BSA table ...Failed")
                    raise
                task.complete("Preprocessing BSA table ...Finished")
        finally:
            _restore_handler_levels(preprocess_muted_handlers)
    else:
        try:
            raw_df, smooth_df = preprocess_tables_parallel(
                input_files=input_files,
                worker_count=worker_count,
                bulk1=args.bulk1,
                bulk2=args.bulk2,
                total_dp_threshold=args.total_dp,
                min_dp=args.min_dp,
                min_gq=args.min_gq,
                ref_allele_freq=args.ref_allele_freq,
                depth_difference=args.depth_difference,
                window_mb=args.window,
                step_mb=args.step,
                ed_power=args.ed_power,
                logger=logger,
                use_spinner=use_spinner,
                bulk1_name=bulk1_name,
                bulk2_name=bulk2_name,
                deltaindex_name=deltaindex_name,
            )
        finally:
            _restore_handler_levels(preprocess_muted_handlers)

    log_success(logger, f"Filter details saved in log: {format_path_for_display(log_path)}")

    raw_df, smooth_df = reoffset_global_chr_positions(raw_df, smooth_df)
    raw_df, smooth_df = filter_low_loci_contigs(raw_df, smooth_df, logger)
    raw_df = sort_by_chr_pos(normalize_position_columns(raw_df))
    smooth_df = sort_by_chr_pos(normalize_position_columns(smooth_df))
    save_table(smooth_df, f"{output_stem}.smooth.tsv", logger, "Smoothed table")

    plot_info_lines: list[str] = []
    muted_console_handlers: list[tuple[logging.Handler, int]] = []
    capture_handler: Optional[logging.Handler] = None
    if use_spinner:
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout:
                muted_console_handlers.append((h, h.level))
                h.setLevel(logging.WARNING)

        class _PlotInfoBufferHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if record.levelno != logging.INFO:
                    return
                try:
                    msg = self.format(record)
                except Exception:
                    msg = record.getMessage()
                text = str(msg).strip()
                if text:
                    plot_info_lines.append(text)

        capture_handler = _PlotInfoBufferHandler(level=logging.INFO)
        capture_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(capture_handler)

    try:
        with CliStatus("Plotting BSA figures...", enabled=use_spinner) as task:
            try:
                plot_bsa(
                    raw_df=raw_df,
                    smooth_df=smooth_df,
                    bulk1_name=bulk1_name,
                    bulk2_name=bulk2_name,
                    deltaindex_name=deltaindex_name,
                    output_stem=output_stem,
                    output_format=args.format,
                    subplot_ratio=args.ratio,
                    ed_power=args.ed_power,
                    window_mb=args.window,
                    ci_levels=args.ci,
                    palette_spec=args.palette_spec,
                    logger=logger,
                )
            except Exception:
                task.fail("Plotting BSA figures ...Failed")
                raise
            task.complete("Plotting BSA figures ...Finished")
    finally:
        if capture_handler is not None:
            logger.removeHandler(capture_handler)
        for h, level in muted_console_handlers:
            h.setLevel(level)
        if plot_info_lines:
            print("\n".join(plot_info_lines))

    lt = time.localtime()
    log_success(
        logger,
        "\nFinished post-BSA analysis. Total wall time: "
        f"{round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
