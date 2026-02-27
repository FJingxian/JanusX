# -*- coding: utf-8 -*-
"""
JanusX: Post-BSA Visualization

Examples
--------
  jx postbsa -file bsa.tsv -b1 Bulk1 -b2 Bulk2

  jx postbsa -file bsa.tsv -b1 Bulk1 -b2 Bulk2 -o results -prefix case1

  jx postbsa -file bsa.tsv -b1 Bulk1 -b2 Bulk2 --window 1 --step 0.25 --format pdf

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
import logging
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Optional

from ._common.log import setup_logging
from ._common.pathcheck import ensure_all_true, ensure_file_exists

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
import numpy as np
import pandas as pd
from joblib import cpu_count

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
DEFAULT_SNPIDX_SCATTER_MAX_POINTS_PER_CHR = 5000
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
    return f"{out_dir}/{stem}".replace("\\", "/").replace("//", "/")


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


def clean_chr(chr_series: pd.Series) -> pd.Series:
    chr_str = chr_series.astype(str).str.upper().str.replace("CHR", "", regex=False).str.strip()

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


def chr_sort_key(x) -> tuple[int, object]:
    try:
        if isinstance(x, (int, float)):
            return (0, float(x))
        if str(x).replace(".", "", 1).isdigit():
            return (0, float(x))
        return (1, str(x))
    except Exception:
        return (2, str(x))


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
    chr_unique = []
    for chr_val in out[chr_col].dropna().unique():
        try:
            num_val = float(chr_val) if "." in str(chr_val) else int(chr_val)
            chr_unique.append((0, num_val, chr_val))
        except (ValueError, TypeError):
            chr_unique.append((1, str(chr_val), chr_val))

    chr_unique.sort()
    chr_sorted = [item[2] for item in chr_unique]
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
            tmp_prefix = f"{tmpdir}/postbsa"
            raw_path, smooth_path = _preprocess_bsa(
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
            raw_df = pd.read_csv(raw_path, sep="\t")
            smooth_df = pd.read_csv(smooth_path, sep="\t")
        logger.info(
            f"Rust preprocessing finished: raw={raw_df.shape}, smooth={smooth_df.shape}"
        )
        return raw_df, smooth_df
    except Exception as exc:
        logger.warning(f"Rust preprocessing failed; falling back to Python: {exc}")
        return None


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
    logger.info(f"Loading BSA table: {table_path}")
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
    logger.info(f"Input shape: {df.shape}")

    initial_rows = len(df)
    df = df.dropna(subset=required_cols)
    logger.info(f"After dropping NA rows: {df.shape} (removed {initial_rows - len(df)})")

    df["CHROM"] = clean_chr(df["CHROM"])
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    df = df.dropna(subset=["CHROM", "POS"]).reset_index(drop=True)

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

    keep = np.ones(len(df), dtype=bool)
    keep &= df[f"{bulk1}.DP"].to_numpy() >= min_dp
    keep &= df[f"{bulk2}.DP"].to_numpy() >= min_dp
    keep &= df[f"{bulk1}.GQ"].to_numpy() >= min_gq
    keep &= df[f"{bulk2}.GQ"].to_numpy() >= min_gq

    total_dp = df[f"{bulk1}.DP"].to_numpy() + df[f"{bulk2}.DP"].to_numpy()
    keep &= total_dp >= total_dp_threshold[0]
    keep &= total_dp <= total_dp_threshold[1]

    dp_diff = np.abs(df[f"{bulk1}.DP"].to_numpy() - df[f"{bulk2}.DP"].to_numpy())
    keep &= dp_diff <= depth_difference
    df = df.loc[keep].reset_index(drop=True)
    logger.info(f"After depth/GQ filtering: {df.shape}")

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
    df = df.loc[freq_keep].reset_index(drop=True)
    logger.info(f"After allele-frequency filtering: {df.shape}")

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
    logger.info(f"{label} saved: {path}")


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

    ed_threshold_90 = np.nanpercentile(raw_df["ED_power"], 90)
    ed_threshold_95 = np.nanpercentile(raw_df["ED_power"], 95)
    delta_95 = np.nanpercentile(raw_df[deltaindex_name], 95)
    delta_90 = np.nanpercentile(raw_df[deltaindex_name], 90)
    delta_10 = np.nanpercentile(raw_df[deltaindex_name], 10)
    delta_5 = np.nanpercentile(raw_df[deltaindex_name], 5)

    finite_gprime = smooth_df["Gprime"].replace([-np.inf, np.inf], np.nan).dropna().to_numpy(dtype=float)
    gprime_threshold_90 = np.nan
    gprime_threshold_95 = np.nan
    if finite_gprime.size > 0:
        gprime_threshold_90 = np.nanpercentile(finite_gprime, 90)
        gprime_threshold_95 = np.nanpercentile(finite_gprime, 95)

    logger.info(f"Threshold of ED^{ed_power} (90,95): {ed_threshold_90:.4f}, {ed_threshold_95:.4f}")
    logger.info(f"Threshold of Delta-SNPindex (5,95): {delta_5:.4f}, {delta_95:.4f}")
    logger.info(f"Threshold of Delta-SNPindex (10,90): {delta_10:.4f}, {delta_90:.4f}")
    if np.isfinite(gprime_threshold_90):
        logger.info(f"Threshold of Gprime (90,95): {gprime_threshold_90:.4f}, {gprime_threshold_95:.4f}")
    else:
        logger.info("Threshold of Gprime (90,95): NA, NA")

    window_half = int(window_mb * 1e6 / 2.0)
    above_threshold = []
    for chr_id in chr_list:
        chr_smooth = smooth_groups.get(chr_id)
        if chr_smooth is None or chr_smooth.empty:
            continue
        threshold_mask = (
            np.isfinite(chr_smooth["ED_power"])
            & np.isfinite(chr_smooth[deltaindex_name])
            & (
                (chr_smooth["ED_power"] >= ed_threshold_90)
                | (chr_smooth[deltaindex_name] >= delta_90)
                | (chr_smooth[deltaindex_name] <= delta_10)
            )
        )
        for center_raw, ed_val, delta_val in zip(
            chr_smooth.loc[threshold_mask, "pos_raw"].to_numpy(dtype=float),
            chr_smooth.loc[threshold_mask, "ED_power"].to_numpy(dtype=float),
            chr_smooth.loc[threshold_mask, deltaindex_name].to_numpy(dtype=float),
        ):
            direction = "upper" if delta_val >= delta_90 else "lower"
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
        logger.info(f"Threshold regions saved: {threshold_path}")

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

    plotted = False
    scatter_total = 0
    scatter_kept = 0
    for idx, chr_id in enumerate(chr_list):
        chr_raw = raw_groups.get(chr_id)
        chr_smooth = smooth_groups.get(chr_id)
        if chr_raw is None or chr_raw.empty or chr_smooth is None or chr_smooth.empty:
            continue
        plotted = True
        color = "green" if idx % 2 == 0 else "orange"
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

    ax_delta.axhline(y=0, color="grey", linestyle="dashed", linewidth=1)
    ax_delta.axhline(y=delta_95, color="blue", linewidth=0.8)
    ax_delta.axhline(y=delta_5, color="blue", linewidth=0.8)
    ax_delta.axhline(y=delta_90, color="yellow", linewidth=0.8)
    ax_delta.axhline(y=delta_10, color="yellow", linewidth=0.8)
    ax_ed.axhline(y=ed_threshold_90, color="yellow", linewidth=0.8)
    ax_ed.axhline(y=ed_threshold_95, color="blue", linewidth=0.8)
    if np.isfinite(gprime_threshold_90):
        ax_gprime.axhline(y=gprime_threshold_90, color="yellow", linewidth=0.8)
        ax_gprime.axhline(y=gprime_threshold_95, color="blue", linewidth=0.8)
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
    logger.info(f"Figure saved: {snp_out}")
    logger.info(f"Figure saved: {stats_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    required_group = parser.add_argument_group("Required Arguments")
    required_group.add_argument(
        "-file",
        "--file",
        type=str,
        required=True,
        help="Input BSA table (tab-delimited).",
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
        "-format",
        "--format",
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
        default=-1,
        help="Number of CPU threads (-1 uses all available cores; default: %(default)s).",
    )
    return parser


def main() -> None:
    t_start = time.time()
    parser = build_parser()
    args = parser.parse_args()

    args.format = str(args.format).lower()
    if not (0.0 <= args.ref_allele_freq <= 0.5):
        raise ValueError("--ref-allele-freq must be within [0, 0.5].")
    if args.step is None:
        args.step = args.window * DEFAULT_STEP_RATIO

    if args.thread <= 0:
        args.thread = cpu_count()
    os.environ["RAYON_NUM_THREADS"] = str(args.thread)

    args.out = args.out if args.out else "."
    os.makedirs(args.out, mode=0o755, exist_ok=True)

    if args.prefix is None:
        args.prefix = strip_known_suffix(args.file)
    output_stem = build_output_stem(args.out, args.prefix, args.bulk1, args.bulk2)

    log_path = f"{output_stem}.postbsa.log"
    logger = setup_logging(log_path)

    logger.info("JanusX - Post-BSA visualization")
    logger.info(f"Host: {socket.gethostname()}\n")
    logger.info("*" * 60)
    logger.info("POST-BSA CONFIGURATION")
    logger.info("*" * 60)
    logger.info(f"Input file:        {args.file}")
    logger.info(f"Bulk1:             {args.bulk1}")
    logger.info(f"Bulk2:             {args.bulk2}")
    logger.info(f"Window (Mb):       {args.window}")
    logger.info(f"Step (Mb):         {args.step}")
    logger.info(f"Plot ratio:        {args.ratio}")
    logger.info(f"Format:            {args.format}")
    logger.info(f"Min DP:            {args.min_dp}")
    logger.info(f"Min GQ:            {args.min_gq}")
    logger.info(f"Total DP range:    {args.total_dp[0]}-{args.total_dp[1]}")
    logger.info(f"Ref allele freq:   {args.ref_allele_freq}")
    logger.info(f"Depth difference:  {args.depth_difference}")
    logger.info(f"ED power:          {args.ed_power}")
    logger.info(f"Threads:           {args.thread} ({cpu_count()} available)")
    logger.info(f"Output stem:       {output_stem}")
    logger.info("*" * 60 + "\n")

    checks = [ensure_file_exists(logger, args.file, "Input BSA table")]
    if not ensure_all_true(checks):
        raise SystemExit(1)

    rust_result = try_rust_preprocess(
        table_path=args.file,
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

    bulk1_name = f"{args.bulk1}.SNPindex"
    bulk2_name = f"{args.bulk2}.SNPindex"
    deltaindex_name = f"Delta.SNPindex({args.bulk2}-{args.bulk1})"

    if rust_result is not None:
        raw_df, smooth_df = rust_result
    else:
        raw_df = load_bsa_in_python(
            table_path=args.file,
            bulk1=args.bulk1,
            bulk2=args.bulk2,
            total_dp_threshold=args.total_dp,
            min_dp=args.min_dp,
            min_gq=args.min_gq,
            ref_allele_freq=args.ref_allele_freq,
            depth_difference=args.depth_difference,
            logger=logger,
        )
        smooth_df = compute_smooth_df(
            raw_df=raw_df,
            bulk1_name=bulk1_name,
            bulk2_name=bulk2_name,
            deltaindex_name=deltaindex_name,
            ed_power=args.ed_power,
            window_mb=args.window,
            step_mb=args.step,
        )

    raw_df = normalize_position_columns(raw_df)
    smooth_df = normalize_position_columns(smooth_df)
    save_table(smooth_df, f"{output_stem}.smooth.tsv", logger, "Smoothed table")

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
        logger=logger,
    )

    lt = time.localtime()
    logger.info(
        "\nFinished post-BSA analysis. Total wall time: "
        f"{round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )


if __name__ == "__main__":
    main()
