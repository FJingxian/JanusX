import time
from functools import lru_cache
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from scipy.stats import beta
from janusx.gtools.cleaner import chrom_sort_key

_PVALUE_EPS = float(np.nextafter(0.0, 1.0))
_QQ_SAMPLE_TRIGGER = 1_000_000
_QQ_SAMPLE_MAX_POINTS = 120_000
_QQ_BAND_MAX_POINTS = 20_000
_CIRCLE_LINK_TENSION = 0.28
_CIRCLE_EDGE_GAP_MULTIPLIER = 1.85
_CIRCLE_IDEOGRAM_WIDTH_DEFAULT = 0.026


def ppoints(n: int) -> np.ndarray:
    """
    Approximate expected quantiles of the Uniform(0,1) distribution.

    Equivalent to R's ppoints(n) with a simple n/(n+1) scheme:
        q_i = i / (n + 1), i = 1..n
    """
    return np.arange(1, n + 1) / (n + 1)


def _sanitize_pvalues(values) -> np.ndarray:
    """
    Convert p-values to finite float64 and clamp to (_PVALUE_EPS, 1.0].
    This prevents -log10 warnings from zeros/NaN/inf/out-of-range values.
    """
    p = pd.to_numeric(values, errors="coerce")
    if isinstance(p, pd.Series):
        arr = p.to_numpy(dtype=np.float64, copy=False)
    else:
        arr = np.asarray(p, dtype=np.float64)
    arr = np.array(arr, dtype=np.float64, copy=True)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr[~np.isfinite(arr)] = 1.0
    return np.clip(arr, _PVALUE_EPS, 1.0)


def _finite_positive(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr) & (arr > 0.0)]


@lru_cache(maxsize=64)
def _qq_confidence_band_logp_cached(
    n: int,
    ci: float,
    limit: int,
    rank_max: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    band_n = int(max(1, int(n)))
    rank_cap = band_n if rank_max is None else int(max(1, min(int(rank_max), band_n)))

    if rank_cap <= limit:
        ranks = np.arange(1, rank_cap + 1, dtype=np.int64)
    else:
        ranks = np.unique(
            np.round(np.geomspace(1.0, float(rank_cap), num=int(max(2, limit)))).astype(np.int64)
        )
        if ranks[0] != 1:
            ranks = np.insert(ranks, 0, 1)
        if ranks[-1] != rank_cap:
            ranks = np.append(ranks, rank_cap)
    ranks = np.sort(ranks)[::-1]

    ci_frac = float(ci)
    if ci_frac > 1.0:
        ci_frac /= 100.0
    ci_frac = float(np.clip(ci_frac, 1e-12, 1.0 - 1e-12))
    alpha = 1.0 - ci_frac
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0

    x_arr = -np.log10(ranks.astype(np.float64) / (band_n + 1.0))
    lo_arr = -np.log10(beta.ppf(q_hi, ranks, band_n - ranks + 1))
    hi_arr = -np.log10(beta.ppf(q_lo, ranks, band_n - ranks + 1))
    keep = np.isfinite(x_arr) & np.isfinite(lo_arr) & np.isfinite(hi_arr)
    x_arr = np.asarray(x_arr[keep], dtype=np.float64)
    lo_arr = np.asarray(lo_arr[keep], dtype=np.float64)
    hi_arr = np.asarray(hi_arr[keep], dtype=np.float64)
    x_arr.setflags(write=False)
    lo_arr.setflags(write=False)
    hi_arr.setflags(write=False)
    return (x_arr, lo_arr, hi_arr)


def _qq_confidence_band_logp(
    n_total: int,
    *,
    ci: float,
    max_points: Union[int, None],
    rank_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(max(1, int(n_total)))
    limit = n if max_points is None else int(max(1, int(max_points)))
    rank_cap = None if rank_max is None else int(max(1, min(int(rank_max), n)))
    return _qq_confidence_band_logp_cached(n, float(ci), limit, rank_cap)


@lru_cache(maxsize=64)
def _qq_sample_draw_indices_cached(
    n: int,
    limit: int,
) -> np.ndarray:
    n_total = int(max(1, int(n)))
    keep_n = int(max(1, min(int(limit), n_total)))
    if n_total <= keep_n:
        arr = np.arange(n_total, dtype=np.int64)
        arr.setflags(write=False)
        return arr

    arr = np.linspace(0, n_total - 1, num=keep_n, dtype=np.int64)
    arr = np.unique(arr.astype(np.int64, copy=False))
    if arr.size < keep_n:
        fill_pool = np.setdiff1d(np.arange(n_total, dtype=np.int64), arr, assume_unique=True)
        fill_need = keep_n - arr.size
        if fill_pool.size > fill_need:
            fill_take = np.linspace(0, fill_pool.size - 1, num=fill_need, dtype=np.int64)
            fill_pool = fill_pool[fill_take]
        arr = np.sort(np.concatenate([arr, fill_pool[:fill_need]])).astype(np.int64, copy=False)
    elif arr.size > keep_n:
        take = np.linspace(0, arr.size - 1, num=keep_n, dtype=np.int64)
        arr = arr[np.unique(take)]
    arr.setflags(write=False)
    return arr


def _qq_sample_draw_indices(
    n_total: int,
    *,
    max_points: Union[int, None],
) -> np.ndarray:
    n = int(max(1, int(n_total)))
    limit = _QQ_SAMPLE_MAX_POINTS if max_points is None else int(max(1, int(max_points)))
    return _qq_sample_draw_indices_cached(n, limit)


def resolve_manhattan_chr_gap(
    lengths: Union[np.ndarray, list[float], list[int]],
    *,
    interval_ratio: float = 0.5,
) -> float:
    """
    Compute chromosome-gap width for Manhattan plots.

    Gap = interval_ratio * median(chromosome_length) / 3
    where interval_ratio is expected in [0, 1].
    """
    positive = _finite_positive(lengths)
    if positive.size == 0:
        return 0.0
    ratio = float(interval_ratio)
    if not np.isfinite(ratio):
        ratio = 0.0
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if ratio <= 0.0:
        return 0.0
    median_len = float(np.median(positive))
    if not np.isfinite(median_len) or median_len <= 0.0:
        return 0.0
    return float(ratio * median_len / 3.0)


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


def _normalize_chrom_token(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    return text


def _repeat_palette(
    n: int,
    colors: Union[list[str], None],
    *,
    cmap_name: str = "tab20",
) -> list[object]:
    n_items = int(max(0, int(n)))
    if n_items == 0:
        return []
    if colors is not None and len(colors) > 0:
        return [colors[i % len(colors)] for i in range(n_items)]
    cmap = plt.get_cmap(str(cmap_name))
    if n_items == 1:
        return [cmap(0.15)]
    return [cmap(i / float(max(1, n_items - 1))) for i in range(n_items)]


def _scale_values_to_range(
    values: object,
    dst_min: float,
    dst_max: float,
    *,
    src_min: Union[float, None] = None,
    src_max: Union[float, None] = None,
    default_frac: float = 0.5,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr)) if src_min is None else float(src_min)
    hi = float(np.nanmax(arr)) if src_max is None else float(src_max)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        frac = np.full(arr.shape, float(np.clip(default_frac, 0.0, 1.0)), dtype=np.float64)
    else:
        frac = (arr - lo) / (hi - lo)
        frac = np.clip(frac, 0.0, 1.0)
    return float(dst_min) + frac * (float(dst_max) - float(dst_min))


def _polar_to_cartesian(theta: object, radius: object) -> tuple[np.ndarray, np.ndarray]:
    theta_arr = np.asarray(theta, dtype=np.float64)
    radius_arr = np.asarray(radius, dtype=np.float64)
    return radius_arr * np.cos(theta_arr), radius_arr * np.sin(theta_arr)


def _wrap_angle_pi(theta: float) -> float:
    wrapped = (float(theta) + np.pi) % (2.0 * np.pi) - np.pi
    if wrapped <= -np.pi:
        wrapped += 2.0 * np.pi
    return float(wrapped)


def _circular_link_path(
    theta1: float,
    theta2: float,
    *,
    anchor_radius: float,
    control_radius: float,
    tension: float = _CIRCLE_LINK_TENSION,
) -> Path:
    delta = _wrap_angle_pi(float(theta2) - float(theta1))
    tension_clamped = float(np.clip(float(tension), 0.05, 0.45))
    ctrl1_theta = float(theta1) + delta * tension_clamped
    ctrl2_theta = float(theta2) - delta * tension_clamped
    sx, sy = _polar_to_cartesian(float(theta1), float(anchor_radius))
    c1x, c1y = _polar_to_cartesian(ctrl1_theta, float(control_radius))
    c2x, c2y = _polar_to_cartesian(ctrl2_theta, float(control_radius))
    ex, ey = _polar_to_cartesian(float(theta2), float(anchor_radius))
    return Path(
        [
            (float(sx), float(sy)),
            (float(c1x), float(c1y)),
            (float(c2x), float(c2y)),
            (float(ex), float(ey)),
        ],
        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
    )


def _interpolate_rgb(
    color_lo: object,
    color_hi: object,
    frac: object,
) -> tuple[float, float, float]:
    lo = np.asarray(mcolors.to_rgb(color_lo), dtype=np.float64)
    hi = np.asarray(mcolors.to_rgb(color_hi), dtype=np.float64)
    alpha = float(np.clip(float(frac), 0.0, 1.0))
    out = lo * (1.0 - alpha) + hi * alpha
    return float(out[0]), float(out[1]), float(out[2])


def _circle_link_palette(logic_key: object) -> tuple[str, str, str]:
    token = str(logic_key).strip().upper()
    if token == "OR":
        return ("#F6C6CC", "#C1121F", "OR")
    return ("#C7DAF3", "#1D4E89", "AND")


def _circle_link_colorbar_axes(
    ax: plt.Axes,
    n_bars: int,
) -> list[plt.Axes]:
    n = max(1, int(n_bars))
    fig = ax.figure
    ax_bbox = ax.get_position()
    width = 0.022
    height = 0.18 if n == 1 else 0.13
    gap = 0.07 if n > 1 else 0.0
    x0 = min(0.965 - width, float(ax_bbox.x1) + 0.018)
    top = min(0.965, float(ax_bbox.y1) - 0.01)
    total_height = height * float(n) + gap * float(max(0, n - 1))
    y0 = top - total_height
    return [
        fig.add_axes(
            [x0, y0 + (n - 1 - i) * (height + gap), width, height],
        )
        for i in range(n)
    ]


def _infer_logic_link_type(value: object) -> str:
    token = str(value).strip().upper()
    if len(token) == 0 or token == "NAN":
        return "OTHER"
    if token in {"AND", "&"} or "&" in token:
        if "!" in token and token not in {"AND", "&"}:
            return "NOT"
        return "AND"
    if token in {"NOT", "!"}:
        return "NOT"
    if "!" in token:
        return "NOT"
    if token in {"OR", "|"} or "|" in token:
        return "OR"
    if token in {"INTERACTION", "EPISTASIS", "*"} or "*" in token:
        return "INTERACTION"
    return token


def _normalize_circle_direction(value: object) -> str:
    token = str(value).strip().lower()
    if token in {"in", "inner", "inward", "circle-in"}:
        return "in"
    return "out"


def _integer_tick_values(
    ymin: float,
    ymax: float,
    *,
    max_ticks: int = 7,
) -> np.ndarray:
    """
    Build integer y-tick locations within the current axis limits.
    """
    lo = float(min(ymin, ymax))
    hi = float(max(ymin, ymax))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)):
        return np.asarray([], dtype=np.float64)
    if hi < lo:
        lo, hi = hi, lo

    start = int(np.ceil(lo - 1e-12))
    stop = int(np.floor(hi + 1e-12))
    if stop < start:
        center = int(np.round((lo + hi) * 0.5))
        if (center < lo - 1e-12) or (center > hi + 1e-12):
            return np.asarray([], dtype=np.float64)
        return np.asarray([float(center)], dtype=np.float64)

    tick_cap = max(1, int(max_ticks))
    count = int(stop - start + 1)
    if count <= tick_cap:
        step = 1
    else:
        step = int(np.ceil(float(count - 1) / float(max(1, tick_cap - 1))))

    ticks = np.arange(start, stop + 1, step, dtype=np.float64)
    if ticks.size == 0 or ticks[0] != float(start):
        ticks = np.insert(ticks, 0, float(start))
    if ticks[-1] != float(stop):
        ticks = np.append(ticks, float(stop))
    return np.unique(ticks.astype(np.float64, copy=False))


def _integer_colorbar_ticks(
    vmin: float,
    vmax: float,
    *,
    max_ticks: int = 4,
) -> tuple[np.ndarray, list[str]]:
    lo = float(min(vmin, vmax))
    hi = float(max(vmin, vmax))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)):
        return np.asarray([], dtype=np.float64), []

    start = int(np.ceil(lo - 1e-12))
    stop = int(np.floor(hi + 1e-12))
    if stop >= start:
        ticks = np.arange(start, stop + 1, dtype=np.int64)
        tick_cap = max(1, int(max_ticks))
        if ticks.size > tick_cap:
            take = np.unique(
                np.round(
                    np.linspace(0, ticks.size - 1, num=tick_cap)
                ).astype(np.int64)
            )
            ticks = ticks[take]
        return (
            ticks.astype(np.float64, copy=False),
            [str(int(x)) for x in ticks.tolist()],
        )

    fallback = np.asarray(
        [lo] if abs(hi - lo) <= 1e-12 else [lo, hi],
        dtype=np.float64,
    )
    keep: list[float] = []
    labels: list[str] = []
    seen: set[str] = set()
    for pos in fallback.tolist():
        label = str(int(round(float(pos))))
        if label in seen:
            continue
        seen.add(label)
        keep.append(float(pos))
        labels.append(label)
    if len(keep) == 0:
        keep = [float(lo)]
        labels = [str(int(round(float(lo))))]
    return np.asarray(keep, dtype=np.float64), labels


def _apply_integer_ticks(
    ax: plt.Axes,
    *,
    axis: str,
    ticks: Union[np.ndarray, list[float], None] = None,
    max_ticks: int = 7,
) -> np.ndarray:
    if str(axis).lower() == "x":
        a0, a1 = ax.get_xlim()
        setter = ax.set_xticks
        label_setter = ax.set_xticklabels
        restore = lambda: ax.set_xlim(a0, a1)
    else:
        a0, a1 = ax.get_ylim()
        setter = ax.set_yticks
        label_setter = ax.set_yticklabels
        restore = lambda: ax.set_ylim(a0, a1)
    lo = float(min(a0, a1))
    hi = float(max(a0, a1))

    if ticks is None:
        tick_values = _integer_tick_values(lo, hi, max_ticks=max_ticks)
    else:
        tick_values = np.asarray(ticks, dtype=np.float64).reshape(-1)
        tick_values = tick_values[np.isfinite(tick_values)]
        tick_values = tick_values[
            (tick_values >= lo - 1e-12) & (tick_values <= hi + 1e-12)
        ]
        tick_values = np.unique(np.round(tick_values).astype(np.int64)).astype(np.float64)
        if tick_values.size == 0:
            tick_values = _integer_tick_values(lo, hi, max_ticks=max_ticks)

    if tick_values.size > 0:
        labels = [str(int(round(float(v)))) for v in tick_values]
        setter(tick_values)
        label_setter(labels)
        restore()
    return tick_values


def apply_integer_yticks(
    ax: plt.Axes,
    *,
    ticks: Union[np.ndarray, list[float], None] = None,
    max_ticks: int = 7,
) -> np.ndarray:
    """
    Force y-axis ticks to integer values while preserving current limits.
    """
    return _apply_integer_ticks(ax, axis="y", ticks=ticks, max_ticks=max_ticks)


def apply_integer_xticks(
    ax: plt.Axes,
    *,
    ticks: Union[np.ndarray, list[float], None] = None,
    max_ticks: int = 7,
) -> np.ndarray:
    """
    Force x-axis ticks to integer values while preserving current limits.
    """
    return _apply_integer_ticks(ax, axis="x", ticks=ticks, max_ticks=max_ticks)


class GWASPLOT:
    """
    Lightweight helper for GWAS visualization (Manhattan & QQ plots).

    Parameters
    ----------
    df : pd.DataFrame
        GWAS result table. Must contain chromosome, position and p-value columns.
    chr : str, default '#CHROM'
        Column name for chromosome.
    pos : str, default 'POS'
        Column name for base-pair position.
    pvalue : str, default 'p'
        Column name for p-values.
    interval_rate : float, default 0.1
        Spacing ratio between chromosomes on the x-axis.
        Absolute spacing = interval_rate * median(chromosome_length) / 3.
    compression : bool, default True
        If True, down-sample SNPs to roughly ~100,000 points for plotting:
          - keep all highly significant points (p <= 10000 / n),
          - for the rest, partition into chunks and keep only the most significant
            SNP per chunk.
    chr_order : list or None, default None
        Optional chromosome order. If None, chromosome labels are ordered
        automatically by natural chromosome order (1..N, X/Y/M/MT, then others).

    Notes
    -----
    - Internally, chromosomes are mapped to integer codes 1..K for plotting.
    - The main DataFrame is stored as self.df with a MultiIndex (chr, pos).
    - A global index list self.minidx encodes which SNPs are retained when
      compression is enabled (used by both Manhattan and QQ plots).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        chr: str = "chrom",
        pos: str = "pos",
        pvalue: str = "pwald",
        interval_rate: float = 0.1,
        compression: bool = True,
        chr_order: Union[list[object], None] = None,
    ) -> None:
        self.t_start = time.time()
        # Ensure positional indices are contiguous for downstream iloc usage.
        df = df.reset_index(drop=True)
        # Global p-value sanitization: avoid log10 warnings in all plotting paths.
        if pvalue in df.columns:
            df[pvalue] = _sanitize_pvalues(df[pvalue])

        # ---- (1) Optional down-sampling for plotting ----
        # minidx is first computed on the original row order (after reset_index).
        # We later sort df by (chr, pos), so we must remap these row labels to
        # positional indices in the sorted table before using iloc.
        if compression:
            n_snp = int(df.shape[0])
            if n_snp == 0:
                self.minidx = []
            else:
                # target: ~100,000 SNPs for display
                chunk_size = n_snp // 100_000 if n_snp >= 100_000 else 1

                # Use numpy arrays to avoid expensive pandas sort/copy chains.
                pvals = _sanitize_pvalues(df[pvalue])

                # p-value threshold: small p => highly significant
                # we keep all SNPs with p <= p_thresh without compression
                p_thresh = 10_000.0 / float(n_snp)
                mask_large_p = pvals > p_thresh

                idx_large = np.flatnonzero(mask_large_p)
                n_large = int(idx_large.size)
                n_chunks = n_large // chunk_size

                if n_chunks > 0:
                    n_take = n_chunks * chunk_size
                    large_vals = pvals[idx_large]
                    # Descending by p-value among "large p", then keep top n_take.
                    ord_desc = np.argsort(-large_vals, kind="mergesort")
                    idx_take = idx_large[ord_desc[:n_take]]
                    # Reorder by original row index (equivalent to previous sort_index()).
                    idx_take.sort()
                    p_take = pvals[idx_take].reshape(n_chunks, chunk_size)
                    local_min_pos = np.argmin(p_take, axis=1)
                    row_base = np.arange(n_chunks, dtype=np.int64) * chunk_size
                    keep_idx_large = idx_take[row_base + local_min_pos]
                else:
                    keep_idx_large = np.empty(0, dtype=np.int64)

                # always keep all highly significant SNPs
                keep_idx_small = np.flatnonzero(~mask_large_p)

                # final kept SNP indices for plotting
                self.minidx = np.concatenate([keep_idx_large, keep_idx_small]).tolist()
        else:
            self.minidx = df.index.to_list()

        # ---- (2) Core columns copy to avoid side effects ----
        df = df[[chr, pos, pvalue]].copy()

        # ---- (3) Map chromosome labels to integers 1..K ----
        observed_labels = df[chr].drop_duplicates().tolist()

        if chr_order is None:
            self.chr_labels = sorted(observed_labels, key=chrom_sort_key)
        else:
            observed_set = set(observed_labels)
            ordered_present = [label for label in chr_order if label in observed_set]
            ordered_set = set(ordered_present)
            missing = [label for label in observed_labels if label not in ordered_set]
            self.chr_labels = ordered_present + sorted(missing, key=chrom_sort_key)

        chr_map = dict(zip(self.chr_labels, range(1, 1 + len(self.chr_labels))))
        df[chr] = df[chr].map(chr_map).astype(int)
        self._chr_label_lookup = {
            _normalize_chrom_token(label): int(code)
            for label, code in chr_map.items()
        }
        self._chr_label_by_id = {
            int(code): str(label)
            for label, code in chr_map.items()
        }

        # Sort by chromosome and position
        df = df.sort_values(by=[chr, pos])
        # Remap kept-row labels -> positional indices in sorted df to avoid
        # iloc selecting wrong rows after sorting.
        if len(self.minidx) > 0:
            orig_labels = np.asarray(self.minidx, dtype=np.int64)
            label_to_pos = np.empty(df.shape[0], dtype=np.int64)
            sorted_labels = df.index.to_numpy(dtype=np.int64, copy=False)
            label_to_pos[sorted_labels] = np.arange(df.shape[0], dtype=np.int64)
            self.minidx = label_to_pos[orig_labels].tolist()
        self.chr_ids = df[chr].unique()

        # ---- (4) Build cumulative genomic x-coordinate (vectorized) ----
        if df.shape[0] > 0:
            chr_arr = df[chr].to_numpy(dtype=np.int64, copy=False)
            pos_arr = pd.to_numeric(df[pos], errors="coerce").fillna(0).to_numpy(
                dtype=np.int64,
                copy=False,
            )

            # boundaries of each chromosome block in the sorted table
            starts = np.r_[0, np.flatnonzero(np.diff(chr_arr) != 0) + 1]
            ends = np.r_[starts[1:], chr_arr.size] - 1

            # cumulative offsets: sum(max_pos of previous chromosomes)
            max_pos_per_chr = np.maximum.reduceat(pos_arr, starts)
            self.interval = int(round(resolve_manhattan_chr_gap(
                max_pos_per_chr,
                interval_ratio=float(interval_rate),
            )))
            chr_offsets = np.zeros_like(max_pos_per_chr, dtype=np.int64)
            if chr_offsets.size > 1:
                chr_offsets[1:] = np.cumsum(max_pos_per_chr[:-1], dtype=np.int64)
            min_pos_per_chr = pos_arr[starts].astype(np.int64, copy=False)

            # chr IDs are remapped to 1..K; direct index lookup is safe.
            offset_by_chr = np.zeros(int(chr_arr.max()) + 1, dtype=np.int64)
            offset_by_chr[self.chr_ids.astype(np.int64)] = chr_offsets
            x_arr = pos_arr + offset_by_chr[chr_arr] + (chr_arr - 1) * self.interval
            df["x"] = x_arr
            self._chr_offset_by_id = {
                int(chr_id): int(chr_offsets[i])
                for i, chr_id in enumerate(self.chr_ids.astype(np.int64))
            }
            self._chr_min_pos_by_id = {
                int(chr_id): int(min_pos_per_chr[i])
                for i, chr_id in enumerate(self.chr_ids.astype(np.int64))
            }
            self._chr_max_pos_by_id = {
                int(chr_id): int(max_pos_per_chr[i])
                for i, chr_id in enumerate(self.chr_ids.astype(np.int64))
            }

            # Cache chromosome bounds / separators / ticks for repeated plotting.
            chr_min_x = x_arr[starts].astype(np.float64)
            chr_max_x = x_arr[ends].astype(np.float64)
            self._chr_bounds_min = chr_min_x
            self._chr_bounds_max = chr_max_x
            if chr_min_x.size > 1:
                self._chr_separators = (chr_max_x[:-1] + chr_min_x[1:]) / 2.0
            else:
                self._chr_separators = np.empty(0, dtype=np.float64)
            chr_gaps = _finite_positive(chr_min_x[1:] - chr_max_x[:-1])
            if chr_gaps.size > 0:
                self._edge_padding_x = 0.5 * float(np.median(chr_gaps))
            elif self.interval > 0:
                self._edge_padding_x = 0.5 * float(self.interval)
            else:
                self._edge_padding_x = 0.0
            self.ticks_loc = (chr_min_x + chr_max_x) / 2.0
            self._plot_xmin = float(chr_min_x[0]) - float(self._edge_padding_x)
            self._plot_xmax = float(chr_max_x[-1]) + float(self._edge_padding_x)
        else:
            self.interval = 0
            df["x"] = np.array([], dtype=np.int64)
            self._chr_bounds_min = np.empty(0, dtype=np.float64)
            self._chr_bounds_max = np.empty(0, dtype=np.float64)
            self._chr_separators = np.empty(0, dtype=np.float64)
            self._edge_padding_x = 0.0
            self.ticks_loc = np.empty(0, dtype=np.float64)
            self._plot_xmin = 0.0
            self._plot_xmax = 1.0
            self._chr_offset_by_id = {}
            self._chr_min_pos_by_id = {}
            self._chr_max_pos_by_id = {}

        # rename standardized y/z columns
        df["y"] = df[pvalue]        # raw p-values
        df["z"] = df[chr]           # integer chromosome ID

        # store as MultiIndex (chr, pos) for easier masking later
        self.df = df.set_index([chr, pos])

    # ------------------------------------------------------------------
    # Manhattan plot
    # ------------------------------------------------------------------
    def _x_to_circle_theta(
        self,
        x: object,
        *,
        start_angle: float = np.pi / 2.0,
        clockwise: bool = True,
        x_min: Union[float, None] = None,
        x_max: Union[float, None] = None,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        use_xmin = float(self._plot_xmin) if x_min is None else float(x_min)
        use_xmax = float(self._plot_xmax) if x_max is None else float(x_max)
        span = float(use_xmax) - float(use_xmin)
        if (not np.isfinite(span)) or span <= 0.0:
            return np.full(x_arr.shape, float(start_angle), dtype=np.float64)
        frac = (x_arr - float(use_xmin)) / span
        frac = np.clip(frac, 0.0, 1.0)
        direction = -1.0 if bool(clockwise) else 1.0
        return float(start_angle) + direction * (2.0 * np.pi * frac)

    def _map_external_loci_to_x(
        self,
        chrom_values: object,
        pos_values: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        chrom_tokens = pd.Series(chrom_values, copy=False).map(_normalize_chrom_token)
        pos_numeric = pd.to_numeric(pos_values, errors="coerce")
        chr_codes = chrom_tokens.map(self._chr_label_lookup)
        keep = chr_codes.notna() & pos_numeric.notna()
        if not bool(keep.any()):
            return (
                np.asarray([], dtype=np.float64),
                np.asarray([], dtype=np.int64),
            )
        chr_arr = chr_codes.loc[keep].to_numpy(dtype=np.int64, copy=False)
        pos_arr = pos_numeric.loc[keep].to_numpy(dtype=np.float64, copy=False)
        offsets = np.asarray(
            [self._chr_offset_by_id.get(int(chr_id), 0) for chr_id in chr_arr],
            dtype=np.float64,
        )
        x_arr = pos_arr + offsets + (chr_arr.astype(np.float64) - 1.0) * float(self.interval)
        return x_arr, chr_arr

    def manhattan(
        self,
        threshold: Union[float,None] = None,
        color_set: Union[list[str],None] = None,
        ax: Union[plt.Axes,None] = None,
        ignore: Union[list[str],None] = None,
        marker: str = "o",
        min_logp: Union[float, None] = 0.5,
        max_logp: Union[float, None] = None,
        y_min: Union[float, None] = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Draw a Manhattan plot.

        Parameters
        ----------
        threshold : float or None
            Genome-wide significance threshold on -log10(p).
            If provided, significant loci are highlighted in red.
        color_set : list[str] or None
            Colors for alternating chromosomes, e.g. ['black','grey'].
        ax : matplotlib.axes.Axes or None
            Existing Axes to draw on. If None, a new figure is created.
        ignore : list
            List of index keys (MultiIndex entries) to ignore in highlighting.
        **kwargs :
            Additional keyword arguments passed to ax.scatter.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if color_set is None or len(color_set) == 0:
            color_set = ["black", "grey"]
        if ignore is None:
            ignore = []

        # subset to plotting SNPs
        df = self.df.iloc[self.minidx, -3:].copy()
        df["y"] = -np.log10(_sanitize_pvalues(df["y"]))
        if min_logp is not None:
            df = df[df["y"] >= float(min_logp)]
        if max_logp is not None:
            df = df[df["y"] <= float(max_logp)]

        if ax is None:
            fig = plt.figure(figsize=[12, 6], dpi=300)
            gs = GridSpec(12, 1, figure=fig)
            ax = fig.add_subplot(gs[0:12, 0])

        if df.shape[0] == 0:
            ax.text(0.5, 0.5, "No SNPs", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks(self.ticks_loc, self.chr_labels)
            ax.set_xlabel("Chromosome")
            ax.set_ylabel("-log10(p-value)")
            return ax

        # color per chromosome (alternating colors)
        chr_color_map = dict(
            zip(
                self.chr_ids,
                [color_set[i % len(color_set)] for i in range(len(self.chr_ids))],
            )
        )
        plot_kwargs = dict(kwargs)
        plot_kwargs.setdefault("alpha", 0.78)
        for key, value in _marker_scatter_style(str(marker)).items():
            plot_kwargs.setdefault(key, value)

        mask_ignore = ~df.index.isin(ignore)
        draw_df = df.loc[mask_ignore, ["x", "y", "z"]]
        for chr_id in self.chr_ids:
            chr_mask = draw_df["z"] == chr_id
            if not bool(np.any(chr_mask)):
                continue
            ax.scatter(
                draw_df.loc[chr_mask, "x"],
                draw_df.loc[chr_mask, "y"],
                color=chr_color_map[chr_id],
                marker=str(marker),
                **plot_kwargs,
            )

        # Highlight SNPs above genome-wide threshold
        if threshold is not None and df["y"].max() >= threshold:
            df_sig = df[df["y"] >= threshold]
            sig_mask = ~df_sig.index.isin(ignore)
            sig_kwargs = dict(kwargs)
            sig_kwargs.setdefault("alpha", 0.9)
            for key, value in _marker_scatter_style(str(marker)).items():
                sig_kwargs.setdefault(key, value)
            ax.scatter(
                df_sig.loc[sig_mask, "x"],
                df_sig.loc[sig_mask, "y"],
                color="red",
                marker=str(marker),
                **sig_kwargs,
            )
            ax.axhline(
                y=threshold,
                color="grey",
                linewidth=1,
                linestyle="--",
            )

        # axis cosmetics (reuse cached chromosome separators)
        if self._chr_separators.size > 0:
            for xsep in self._chr_separators:
                if np.isfinite(xsep):
                    ax.axvline(
                        float(xsep),
                        ymin=0.0,
                        ymax=1.0 / 3.0,
                        linestyle="--",
                        color="lightgrey",
                        linewidth=0.6,
                        alpha=0.8,
                        zorder=0,
                    )

        ax.set_xticks(self.ticks_loc, self.chr_labels)
        if self._chr_bounds_min.size > 0 and self._chr_bounds_max.size > 0:
            xmin = float(self._chr_bounds_min[0]) - float(self._edge_padding_x)
            xmax = float(self._chr_bounds_max[-1]) + float(self._edge_padding_x)
        else:
            xmin = float(df["x"].min())
            xmax = float(df["x"].max())
        if xmax > xmin:
            ax.set_xlim(xmin, xmax)
        else:
            eps = max(1e-9, abs(xmin) * 1e-9)
            ax.set_xlim(xmin - eps, xmax + eps)
        ax.margins(x=0.0)
        ymax = df["y"].max()
        base_ymin = float(min_logp) if min_logp is not None else 0.0
        if y_min is not None:
            base_ymin = float(y_min)
        top = ymax + 0.1 * ymax if ymax > 0 else (base_ymin + 1.0)
        if top <= base_ymin:
            top = base_ymin + 1.0
        ax.set_ylim([base_ymin, top])
        ax.set_xlabel("Chromosome")
        ax.set_ylabel("-log10(p-value)")
        apply_integer_yticks(ax)
        return ax

    def circle_manhattan(
        self,
        threshold: Union[float, None] = None,
        color_set: Union[list[str], None] = None,
        ax: Union[plt.Axes, None] = None,
        links_df: Union[pd.DataFrame, None] = None,
        link_chr1: str = "chrom1",
        link_pos1: str = "pos1",
        link_chr2: str = "chrom2",
        link_pos2: str = "pos2",
        link_type_col: Union[str, None] = None,
        link_pvalue_col: Union[str, None] = None,
        marker: str = "o",
        scatter_size: float = 8.0,
        scatter_alpha: float = 0.76,
        min_logp: Union[float, None] = 0.5,
        max_logp: Union[float, None] = None,
        y_min: Union[float, None] = None,
        outer_radius: float = 1.0,
        ideogram_width: float = _CIRCLE_IDEOGRAM_WIDTH_DEFAULT,
        track_width: float = 0.34,
        track_ratio: float = 0.5,
        link_radius: float = 0.34,
        link_center_radius: float = 0.05,
        link_interval: float = 1.0,
        link_alpha: float = 0.92,
        link_linewidth: float = 1.0,
        show_link_legend: bool = True,
        show_link_colorbar: bool = False,
        draw_background: bool = True,
        draw_scatter: bool = True,
        draw_links: bool = True,
        circle_direction: str = "out",
        label_fontsize: float = 10.0,
        radial_tick_fontsize: float = 8.0,
        rasterized: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """
        Draw a circular Manhattan / Circos-style plot.

        The outer region shows single-locus Manhattan scatter points, a
        chromosome ideogram transition ring separates scatter and
        interaction layers, and the inner core overlays smooth
        interaction arcs.
        """
        df = self.df.iloc[self.minidx, -3:].copy()
        df["y"] = -np.log10(_sanitize_pvalues(df["y"]))
        if min_logp is not None:
            df = df[df["y"] >= float(min_logp)]
        if max_logp is not None:
            df = df[df["y"] <= float(max_logp)]
        df = df[np.isfinite(df["x"]) & np.isfinite(df["y"]) & np.isfinite(df["z"])]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=300)
        ax.set_aspect("equal")
        ax.axis("off")
        dynamic_artists: list[object] = []
        dynamic_axes: list[plt.Axes] = []

        chr_ids_int = np.asarray(self.chr_ids, dtype=np.int64)
        chr_palette = _repeat_palette(chr_ids_int.size, color_set, cmap_name="tab20")
        chr_color_map = {
            int(chr_id): chr_palette[i]
            for i, chr_id in enumerate(chr_ids_int.tolist())
        }
        direction = _normalize_circle_direction(circle_direction)
        circle_inward = direction == "in"

        plot_outer_r = max(0.35, float(outer_radius))
        ideogram_width_raw = float(ideogram_width)
        if abs(ideogram_width_raw - float(_CIRCLE_IDEOGRAM_WIDTH_DEFAULT)) <= 1e-12:
            # Keep the chromosome transition ring just slightly wider than
            # the interaction lines so it stays readable without dominating
            # the radial layout.
            ideogram_w = max(
                0.014,
                min(0.028, 0.014 + 0.006 * float(link_linewidth)),
            )
        else:
            ideogram_w = ideogram_width_raw
        ideogram_w = float(np.clip(ideogram_w, 0.012, plot_outer_r * 0.20))
        track_outer_r = max(0.12, plot_outer_r - 0.06)
        track_ratio = float(np.clip(float(track_ratio), 0.0, 1.0))
        inner_reserve = max(0.07, ideogram_w + 0.035)
        max_track_width = max(0.08, track_outer_r - inner_reserve)
        resolved_track_width = max(0.08, max_track_width * track_ratio)
        track_inner_r = max(0.08, track_outer_r - resolved_track_width)
        ideogram_gap = max(0.006, min(0.016, 0.06 * resolved_track_width))
        ideogram_outer_r = max(0.08, track_inner_r - ideogram_gap)
        ideogram_inner_r = max(0.05, ideogram_outer_r - ideogram_w)
        base_link_anchor_r = float(np.clip(float(link_radius), 0.02, max(0.02, ideogram_inner_r - 0.03)))
        base_link_core_r = float(np.clip(float(link_center_radius), 0.0, max(0.0, base_link_anchor_r - 0.02)))
        link_interval = float(np.clip(float(link_interval), 0.0, 1.0))
        near_link_anchor_r = max(base_link_core_r + 0.02, ideogram_inner_r - 0.015)
        anchor_shift = max(0.0, near_link_anchor_r - base_link_anchor_r) * (1.0 - link_interval)
        link_anchor_r = float(
            np.clip(
                base_link_anchor_r + anchor_shift,
                max(base_link_core_r + 0.02, 0.02),
                max(base_link_core_r + 0.02, ideogram_inner_r - 0.01),
            )
        )
        link_core_r = float(
            np.clip(
                base_link_core_r + anchor_shift,
                0.0,
                max(0.0, link_anchor_r - 0.02),
            )
        )

        finite_y = df["y"].to_numpy(dtype=np.float64, copy=False)
        base_ymin = float(min_logp) if min_logp is not None else 0.0
        if y_min is not None:
            base_ymin = float(y_min)
        plot_ymax = float(np.nanmax(finite_y)) if finite_y.size > 0 else 1.0
        if threshold is not None and np.isfinite(float(threshold)):
            plot_ymax = max(plot_ymax, float(threshold))
        if not np.isfinite(plot_ymax) or plot_ymax <= base_ymin:
            plot_ymax = base_ymin + 1.0
        scatter_radius_start = float(track_outer_r if circle_inward else track_inner_r)
        scatter_radius_end = float(track_inner_r if circle_inward else track_outer_r)
        zero_radius = float(
            np.clip(
                _scale_values_to_range(
                    [0.0],
                    scatter_radius_start,
                    scatter_radius_end,
                    src_min=base_ymin,
                    src_max=plot_ymax,
                )[0],
                track_inner_r,
                track_outer_r,
            )
        )

        circle_edge_padding_x = max(
            float(self._edge_padding_x) * float(_CIRCLE_EDGE_GAP_MULTIPLIER),
            float(self._edge_padding_x) + max(0.0, 0.35 * float(self.interval)),
        )
        circle_plot_xmin = float(self._chr_bounds_min[0]) - circle_edge_padding_x if self._chr_bounds_min.size > 0 else float(self._plot_xmin)
        circle_plot_xmax = float(self._chr_bounds_max[-1]) + circle_edge_padding_x if self._chr_bounds_max.size > 0 else float(self._plot_xmax)

        theta_grid = np.linspace(0.0, 2.0 * np.pi, num=721, endpoint=True, dtype=np.float64)
        lead_gap_theta = float(
            self._x_to_circle_theta(
                [
                    0.5 * (float(circle_plot_xmin) + float(self._chr_bounds_min[0]))
                    if self._chr_bounds_min.size > 0
                    else float(circle_plot_xmin)
                ],
                x_min=circle_plot_xmin,
                x_max=circle_plot_xmax,
            )[0]
        )
        if draw_background:
            for idx, chr_id in enumerate(chr_ids_int.tolist()):
                theta_start = float(
                    self._x_to_circle_theta(
                        [self._chr_bounds_min[idx]],
                        x_min=circle_plot_xmin,
                        x_max=circle_plot_xmax,
                    )[0]
                )
                theta_end = float(
                    self._x_to_circle_theta(
                        [self._chr_bounds_max[idx]],
                        x_min=circle_plot_xmin,
                        x_max=circle_plot_xmax,
                    )[0]
                )
                wedge = Wedge(
                    (0.0, 0.0),
                    r=ideogram_outer_r,
                    theta1=np.degrees(theta_end),
                    theta2=np.degrees(theta_start),
                    width=ideogram_w,
                    facecolor=chr_color_map[int(chr_id)],
                    edgecolor="white",
                    linewidth=1.0,
                    zorder=4,
                )
                ax.add_patch(wedge)
                label_theta = float(
                    self._x_to_circle_theta(
                        [self.ticks_loc[idx]],
                        x_min=circle_plot_xmin,
                        x_max=circle_plot_xmax,
                    )[0]
                )
                label_radius = plot_outer_r + 0.055
                delta_to_gap = abs(float(np.arctan2(np.sin(label_theta - lead_gap_theta), np.cos(label_theta - lead_gap_theta))))
                if delta_to_gap <= 0.42:
                    label_radius = plot_outer_r + 0.085
                lx, ly = _polar_to_cartesian(label_theta, label_radius)
                text_rotation = float(np.degrees(label_theta) - 90.0)
                if text_rotation < -90.0:
                    text_rotation += 180.0
                elif text_rotation > 90.0:
                    text_rotation -= 180.0
                ax.text(
                    float(lx),
                    float(ly),
                    self._chr_label_by_id.get(int(chr_id), str(chr_id)),
                    ha="center",
                    va="center",
                    rotation=text_rotation,
                    rotation_mode="anchor",
                    fontsize=float(label_fontsize),
                    zorder=5,
                )

        if draw_background:
            for radius in (track_inner_r, track_outer_r):
                cx, cy = _polar_to_cartesian(
                    theta_grid,
                    np.full(theta_grid.shape, radius, dtype=np.float64),
                )
                ax.plot(cx, cy, color="#D6D6D6", linewidth=0.7, alpha=0.9, zorder=1)
            zero_cx, zero_cy = _polar_to_cartesian(
                theta_grid,
                np.full(theta_grid.shape, zero_radius, dtype=np.float64),
            )
            ax.plot(
                zero_cx,
                zero_cy,
                color="#111111",
                linewidth=1.2,
                alpha=0.98,
                zorder=2.1,
            )

        tick_floor = max(0.0, float(base_ymin))
        radial_ticks = _integer_tick_values(tick_floor, plot_ymax, max_ticks=5)
        radial_ticks = radial_ticks[
            (radial_ticks >= tick_floor - 1e-12) & (radial_ticks <= plot_ymax + 1e-12)
        ]
        if tick_floor <= 1e-12:
            radial_ticks = radial_ticks[radial_ticks > 1e-12]
        tick_angle = lead_gap_theta
        axis_line_start = float(zero_radius)
        axis_line_end = (
            float(track_inner_r)
            if circle_inward
            else float(track_outer_r + 0.03)
        )
        axis_sx, axis_sy = _polar_to_cartesian(tick_angle, axis_line_start)
        axis_ex, axis_ey = _polar_to_cartesian(tick_angle, axis_line_end)
        if draw_background:
            ax.plot(
                [float(axis_sx), float(axis_ex)],
                [float(axis_sy), float(axis_ey)],
                color="#7A7A7A",
                linewidth=1.0,
                alpha=0.95,
                zorder=4.2,
            )
            tangent_ux = -float(np.sin(tick_angle))
            tangent_uy = float(np.cos(tick_angle))
            tick_mark_len = max(0.018, 0.10 * float(track_outer_r - track_inner_r))
            tick_label_tangent_offset = tick_mark_len + 0.008
            tick_label_radial_offset = 0.003
            for tick in radial_ticks:
                tick_radius = float(
                    _scale_values_to_range(
                        [tick],
                        scatter_radius_start,
                        scatter_radius_end,
                        src_min=base_ymin,
                        src_max=plot_ymax,
                    )[0]
                )
                tick_cx, tick_cy = _polar_to_cartesian(tick_angle, tick_radius)
                mark_x = np.asarray(
                    [
                        float(tick_cx),
                        float(tick_cx) - tangent_ux * tick_mark_len,
                    ],
                    dtype=np.float64,
                )
                mark_y = np.asarray(
                    [
                        float(tick_cy),
                        float(tick_cy) - tangent_uy * tick_mark_len,
                    ],
                    dtype=np.float64,
                )
                ax.plot(
                    mark_x,
                    mark_y,
                    color="#7A7A7A",
                    linewidth=0.9,
                    alpha=0.95,
                    zorder=4.2,
                )
                tick_text_x, tick_text_y = _polar_to_cartesian(
                    tick_angle,
                    tick_radius + tick_label_radial_offset,
                )
                tick_text = (
                    str(int(round(float(tick))))
                    if abs(float(tick) - round(float(tick))) < 1e-8
                    else f"{float(tick):g}"
                )
                ax.text(
                    float(tick_text_x) + tangent_ux * tick_label_tangent_offset,
                    float(tick_text_y) + tangent_uy * tick_label_tangent_offset,
                    tick_text,
                    fontsize=float(radial_tick_fontsize),
                    ha="center",
                    va="center",
                    color="#666666",
                    zorder=5,
                )

        if draw_background and threshold is not None and np.isfinite(float(threshold)):
            thr_radius = float(
                _scale_values_to_range(
                    [float(threshold)],
                    scatter_radius_start,
                    scatter_radius_end,
                    src_min=base_ymin,
                    src_max=plot_ymax,
                )[0]
            )
            tx, ty = _polar_to_cartesian(
                theta_grid,
                np.full(theta_grid.shape, thr_radius, dtype=np.float64),
            )
            ax.plot(tx, ty, color="#C1121F", linewidth=1.0, linestyle="--", alpha=0.95, zorder=2)

        if draw_background and df.shape[0] == 0 and (links_df is None or links_df.shape[0] == 0):
            ax.text(0.0, 0.0, "No SNPs", ha="center", va="center", fontsize=12)
            lim = outer_r + 0.28
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax._janusx_circle_dynamic_artists = dynamic_artists
            ax._janusx_circle_dynamic_axes = dynamic_axes
            return ax

        if draw_scatter and df.shape[0] > 0:
            y_arr = df["y"].to_numpy(dtype=np.float64, copy=False)
            theta_points = self._x_to_circle_theta(
                df["x"].to_numpy(dtype=np.float64, copy=False),
                x_min=circle_plot_xmin,
                x_max=circle_plot_xmax,
            )
            radial_points = _scale_values_to_range(
                y_arr,
                scatter_radius_start,
                scatter_radius_end,
                src_min=base_ymin,
                src_max=plot_ymax,
            )
            radial_points = (
                np.minimum(radial_points, zero_radius)
                if circle_inward
                else np.maximum(radial_points, zero_radius)
            )
            px, py = _polar_to_cartesian(theta_points, radial_points)
            plot_kwargs = dict(kwargs)
            plot_kwargs.setdefault("s", float(scatter_size))
            plot_kwargs.setdefault("alpha", float(scatter_alpha))
            plot_kwargs.setdefault("rasterized", bool(rasterized))
            for key, value in _marker_scatter_style(str(marker)).items():
                plot_kwargs.setdefault(key, value)
            clip_outer_radius = float(zero_radius if circle_inward else track_outer_r)
            clip_inner_radius = float(track_inner_r if circle_inward else zero_radius)
            scatter_clip_patch = Wedge(
                (0.0, 0.0),
                r=clip_outer_radius + 1e-3,
                theta1=0.0,
                theta2=360.0,
                width=max(clip_outer_radius - clip_inner_radius + 1e-3, 1e-6),
                facecolor="none",
                edgecolor="none",
            )
            scatter_clip_patch.set_transform(ax.transData)
            z_arr = df["z"].to_numpy(dtype=np.int64, copy=False)
            sig_mask = np.zeros(df.shape[0], dtype=bool)
            if threshold is not None and np.isfinite(float(threshold)):
                sig_mask = y_arr >= float(threshold) - 1e-12
            for chr_id in chr_ids_int.tolist():
                chr_mask = z_arr == int(chr_id)
                if not bool(np.any(chr_mask)):
                    continue
                base_mask = chr_mask & (~sig_mask)
                if bool(np.any(base_mask)):
                    collection = ax.scatter(
                        px[base_mask],
                        py[base_mask],
                        color=chr_color_map[int(chr_id)],
                        marker=str(marker),
                        zorder=3,
                        **plot_kwargs,
                    )
                    collection.set_clip_path(scatter_clip_patch)
                hit_mask = chr_mask & sig_mask
                if bool(np.any(hit_mask)):
                    hit_kwargs = dict(plot_kwargs)
                    hit_kwargs["alpha"] = max(float(scatter_alpha), 0.9)
                    hit_collection = ax.scatter(
                        px[hit_mask],
                        py[hit_mask],
                        color="#C1121F",
                        marker=str(marker),
                        zorder=3.25,
                        **hit_kwargs,
                    )
                    hit_collection.set_clip_path(scatter_clip_patch)

        legend_handles: list[Line2D] = []
        red_count = 0
        blue_count = 0
        strength_lo = None
        strength_hi = None
        if draw_links and links_df is not None and links_df.shape[0] > 0:
            missing_cols = [
                col for col in (link_chr1, link_pos1, link_chr2, link_pos2)
                if col not in links_df.columns
            ]
            if len(missing_cols) > 0:
                raise ValueError(
                    "links_df is missing required columns: " + ", ".join(missing_cols)
                )
            link_table = links_df.copy()
            resolved_type_col = link_type_col
            if resolved_type_col is None:
                for candidate in ("gate", "logic", "type", "combo_id", "combo", "interaction"):
                    if candidate in link_table.columns:
                        resolved_type_col = candidate
                        break
            resolved_p_col = link_pvalue_col
            if resolved_p_col is None:
                for candidate in ("p_combo_joint", "p_combo_marginal", "pwald", "p"):
                    if candidate in link_table.columns:
                        resolved_p_col = candidate
                        break
            if resolved_p_col is not None and resolved_p_col in link_table.columns:
                link_p = _sanitize_pvalues(link_table[resolved_p_col])
                link_table = link_table.assign(__link_p=link_p)

            keep_mask = np.zeros(link_table.shape[0], dtype=bool)
            chrom1_series = pd.Series(link_table[link_chr1], copy=False).map(_normalize_chrom_token)
            chrom2_series = pd.Series(link_table[link_chr2], copy=False).map(_normalize_chrom_token)
            pos1_series = pd.to_numeric(link_table[link_pos1], errors="coerce")
            pos2_series = pd.to_numeric(link_table[link_pos2], errors="coerce")
            chr1_keep = chrom1_series.map(self._chr_label_lookup).notna() & pos1_series.notna()
            chr2_keep = chrom2_series.map(self._chr_label_lookup).notna() & pos2_series.notna()
            keep_mask[:] = (chr1_keep & chr2_keep).to_numpy(dtype=bool, copy=False)
            link_table = link_table.loc[keep_mask].copy()
            if link_table.shape[0] > 0:
                x1, _chr1_codes = self._map_external_loci_to_x(
                    link_table[link_chr1],
                    link_table[link_pos1],
                )
                x2, _chr2_codes = self._map_external_loci_to_x(
                    link_table[link_chr2],
                    link_table[link_pos2],
                )
                theta1 = self._x_to_circle_theta(
                    x1,
                    x_min=circle_plot_xmin,
                    x_max=circle_plot_xmax,
                )
                theta2 = self._x_to_circle_theta(
                    x2,
                    x_min=circle_plot_xmin,
                    x_max=circle_plot_xmax,
                )
                if resolved_type_col is not None and resolved_type_col in link_table.columns:
                    link_types = (
                        link_table[resolved_type_col]
                        .map(_infer_logic_link_type)
                        .to_numpy(dtype=object, copy=False)
                    )
                else:
                    link_types = np.full(link_table.shape[0], "OTHER", dtype=object)
                if "__link_p" in link_table.columns:
                    strength_arr = -np.log10(
                        link_table["__link_p"].to_numpy(dtype=np.float64, copy=False)
                    )
                else:
                    strength_arr = np.full(link_table.shape[0], float("nan"), dtype=np.float64)
                strength_arr[~np.isfinite(strength_arr)] = np.nan
                finite_strength = strength_arr[np.isfinite(strength_arr)]
                if finite_strength.size == 0:
                    strength_lo = (
                        float(threshold)
                        if threshold is not None and np.isfinite(float(threshold))
                        else 0.0
                    )
                    strength_hi = strength_lo + 1.0
                else:
                    strength_lo = (
                        float(threshold)
                        if threshold is not None and np.isfinite(float(threshold))
                        else float(np.nanmin(finite_strength))
                    )
                    if not np.isfinite(strength_lo):
                        strength_lo = float(np.nanmin(finite_strength))
                    strength_hi = float(np.nanmax(finite_strength))
                    if (not np.isfinite(strength_hi)) or strength_hi <= strength_lo:
                        strength_hi = strength_lo + 1.0
                curve_control_r = max(link_core_r, 0.58 * link_anchor_r)
                draw_strength = np.where(np.isfinite(strength_arr), strength_arr, -np.inf)
                draw_order = np.argsort(draw_strength, kind="stable")
                for idx in draw_order.tolist():
                    t1 = float(theta1[idx])
                    t2 = float(theta2[idx])
                    sig_frac = (
                        float(
                            np.clip(
                                (float(strength_arr[idx]) - float(strength_lo))
                                / (float(strength_hi) - float(strength_lo)),
                                0.0,
                                1.0,
                            )
                        )
                        if np.isfinite(strength_arr[idx])
                        else 0.9
                    )
                    logic_key = str(link_types[idx]).strip().upper()
                    is_or = logic_key == "OR"
                    color_lo, color_hi, _group_label = _circle_link_palette(
                        "OR" if is_or else "AND"
                    )
                    edge_color = _interpolate_rgb(color_lo, color_hi, sig_frac)
                    patch = PathPatch(
                        _circular_link_path(
                            t1,
                            t2,
                            anchor_radius=link_anchor_r,
                            control_radius=curve_control_r,
                            tension=_CIRCLE_LINK_TENSION,
                        ),
                        facecolor="none",
                        edgecolor=edge_color,
                        linewidth=float(link_linewidth),
                        alpha=float(link_alpha),
                        capstyle="round",
                        joinstyle="round",
                        zorder=2.2 + 0.002 * float(sig_frac),
                    )
                    ax.add_patch(patch)
                    dynamic_artists.append(patch)
                    if is_or:
                        red_count += 1
                    else:
                        blue_count += 1
                if np.isfinite(float(strength_lo)) and np.isfinite(float(strength_hi)):
                    strength_lo = float(strength_lo)
                    strength_hi = float(strength_hi)
                if red_count > 0:
                    legend_handles.append(
                        Line2D([0], [0], color="#C1121F", lw=2.4, label=f"OR ({red_count})")
                    )
                if blue_count > 0:
                    legend_handles.append(
                        Line2D([0], [0], color="#1D4E89", lw=2.4, label=f"AND ({blue_count})")
                    )

        if show_link_colorbar and (red_count > 0 or blue_count > 0) and strength_lo is not None and strength_hi is not None:
            bar_specs: list[tuple[str, int, str, str]] = []
            if red_count > 0:
                color_lo, color_hi, label = _circle_link_palette("OR")
                bar_specs.append((label, red_count, color_lo, color_hi))
            if blue_count > 0:
                color_lo, color_hi, label = _circle_link_palette("AND")
                bar_specs.append((label, blue_count, color_lo, color_hi))
            norm = mcolors.Normalize(vmin=float(strength_lo), vmax=float(strength_hi))
            for cax, (label, count, color_lo, color_hi) in zip(
                _circle_link_colorbar_axes(ax, len(bar_specs)),
                bar_specs,
            ):
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    f"janusx_{label.lower()}_link",
                    [color_lo, color_hi],
                )
                sm = ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cbar = ax.figure.colorbar(sm, cax=cax)
                tick_pos, tick_labels = _integer_colorbar_ticks(
                    float(strength_lo),
                    float(strength_hi),
                    max_ticks=4,
                )
                if tick_pos.size > 0:
                    cbar.set_ticks(tick_pos.tolist())
                    cbar.set_ticklabels(tick_labels)
                cbar.ax.tick_params(labelsize=max(6.0, float(radial_tick_fontsize) - 1.0), length=2.0, pad=1.5)
                cbar.outline.set_linewidth(0.5)
                cbar.set_label("-log10(p)", fontsize=max(6.5, float(radial_tick_fontsize) - 0.5), labelpad=1.5)
                cax.set_title(f"{label} ({count})", fontsize=max(6.5, float(radial_tick_fontsize) - 0.25), pad=2.0)
                dynamic_axes.append(cax)
        elif show_link_legend and len(legend_handles) > 0:
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                bbox_to_anchor=(0.93, 0.88),
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=0.9,
                title="Links",
                borderaxespad=0.2,
                labelspacing=0.35,
                handlelength=1.8,
            )

        lim = plot_outer_r + 0.28
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax._janusx_circle_dynamic_artists = dynamic_artists
        ax._janusx_circle_dynamic_axes = dynamic_axes
        return ax

    # ------------------------------------------------------------------
    # QQ plot
    # ------------------------------------------------------------------
    def qq(
        self,
        ax: plt.Axes = None,
        ci: int = 95,
        color_set: list[str] = None,
        marker: str = "o",
        scatter_size: float = 8.0,
        scatter_alpha: float = 0.75,
        line_color: str = "black",
        qq_mode: str = "auto",
        qq_auto_threshold: int = _QQ_SAMPLE_TRIGGER,
        qq_fast_max_points: int = _QQ_SAMPLE_MAX_POINTS,
        qq_band_max_points: Union[int, None] = _QQ_BAND_MAX_POINTS,
        sig_p_threshold: Union[float, None] = None,
        axis_min: Union[float, None] = None,
        axis_max: Union[float, None] = None,
        band_color: Union[str, None] = None,
        rasterized: bool = True,
    ) -> plt.Axes:
        """
        Draw a QQ plot of observed vs expected -log10(p) with a confidence band.

        The band is computed from SciPy Beta order-statistic quantiles.
        Point selection and drawing stay in Python to keep plotting isolated
        from the native GWAS extension.

        QQ computation is independent from Manhattan compression (`self.minidx`)
        to avoid significance-enriched bias in QQ shape.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new figure is created.
        ci : int, default 95
            Confidence interval (in percent) for the null band.
        color_set : list[str], optional
            QQ colors. If >=2 colors are given:
            - color_set[0]: scatter points
            - color_set[1]: confidence band
            If only one color is provided, it is used for both.
        scatter_size : float, default 8.0
            Marker size for QQ scatter points.
        scatter_alpha : float, default 0.75
            Marker alpha for QQ scatter points.
        line_color : str, default "black"
            Color of the ideal diagonal line y=x.
        qq_mode : {"auto","full","fast"}, default "auto"
            QQ computation mode:
            - auto: use fast mode when SNP count > qq_auto_threshold
            - full: full-rank QQ using all SNPs (exact)
            - fast: rank-grid approximation from full QQ (deterministic)
        qq_auto_threshold : int, default 1_000_000
            Threshold used when qq_mode="auto".
        qq_fast_max_points : int, default 120_000
            Max points in fast-mode QQ scatter.
        qq_band_max_points : int or None, default 20_000
            Max points used for confidence band (both modes).
            If None, use all ranks.
        sig_p_threshold : float or None, default None
            Significance threshold to always preserve in QQ scatter.
            If None, Bonferroni-like threshold 1/n is used.
        axis_min : float or None, default None
            If provided, force both QQ axis lower bounds (x/y) to this value.
        axis_max : float or None, default None
            If provided, force QQ y-axis upper bound to this value.
        band_color : str or None, default None
            Override the confidence-band fill color.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if color_set is None or len(color_set) == 0:
            color_set = ["black", "grey"]
        if len(color_set) >= 2:
            point_color = color_set[0]
            band_fill_color = color_set[1]
        else:
            point_color = color_set[0]
            band_fill_color = color_set[0]
        if band_color is not None:
            band_fill_color = str(band_color)

        mode_key = str(qq_mode).strip().lower()
        if mode_key not in {"auto", "full", "fast"}:
            raise ValueError("qq_mode must be one of: auto, full, fast")

        if ax is None:
            fig = plt.figure(figsize=[12, 6], dpi=600)
            gs = GridSpec(12, 1, figure=fig)
            ax = fig.add_subplot(gs[0:12, 0])

        p = self.df["y"].to_numpy(dtype=np.float64, copy=True)
        n = p.size
        if n == 0:
            raise ValueError("No p-values found for QQ plot.")
        p[~np.isfinite(p)] = 1.0
        p = np.clip(p, np.finfo(np.float64).tiny, 1.0)
        if sig_p_threshold is None:
            sig_thr = 1.0 / float(n)
        else:
            sig_thr = float(sig_p_threshold)
        if not np.isfinite(sig_thr):
            sig_thr = 1.0 / float(n)
        sig_thr = float(np.clip(sig_thr, np.finfo(np.float64).tiny, 1.0))

        resolved_mode = mode_key
        if mode_key == "auto":
            resolved_mode = "fast" if n > int(qq_auto_threshold) else "full"

        # Full sorted p-values are the canonical QQ backbone.
        p_sorted = np.sort(p, kind="mergesort")
        sig_n = int(np.searchsorted(p_sorted, sig_thr, side="right"))

        if resolved_mode == "full":
            # Exact QQ: all SNPs
            draw_idx = np.arange(n, dtype=np.int64)
        else:
            # Fast QQ uses a dense linear-rank grid and always preserves the
            # most significant tail to avoid visible gaps near the top.
            max_points = int(max(1, qq_fast_max_points))
            if n > max_points:
                base_idx = _qq_sample_draw_indices(n, max_points=max_points)
                if sig_n > 0:
                    sig_idx = np.arange(sig_n, dtype=np.int64)
                    draw_idx = np.unique(np.concatenate([base_idx, sig_idx]))
                else:
                    draw_idx = base_idx
            else:
                draw_idx = np.arange(n, dtype=np.int64)

        p_draw = p_sorted[draw_idx]
        ranks_draw = draw_idx.astype(np.float64) + 1.0
        obs_scatter = -np.log10(p_draw)
        exp_scatter = -np.log10(ranks_draw / (n + 1.0))

        x_band, lower, upper = _qq_confidence_band_logp(
            n,
            ci=ci,
            max_points=qq_band_max_points,
        )
        band_mask = np.isfinite(x_band) & np.isfinite(lower) & np.isfinite(upper)

        # Draw confidence interval
        if np.any(band_mask):
            ax.fill_between(
                x_band[band_mask],
                lower[band_mask],
                upper[band_mask],
                color=band_fill_color,
                alpha=0.3,
                rasterized=bool(rasterized),
            )

        # Diagonal reference line y = x
        finite_obs = obs_scatter[np.isfinite(obs_scatter)]
        finite_exp = exp_scatter[np.isfinite(exp_scatter)]
        if finite_obs.size == 0 or finite_exp.size == 0:
            max_lim = 1.0
        else:
            max_lim = float(min(np.max(finite_obs), np.max(finite_exp)))
            if not np.isfinite(max_lim) or max_lim <= 0:
                max_lim = 1.0
        ax.plot([0.0, max_lim], [0.0, max_lim], lw=1, color=line_color)

        scatter_mask = np.isfinite(exp_scatter) & np.isfinite(obs_scatter)
        ax.scatter(
            exp_scatter[scatter_mask],
            obs_scatter[scatter_mask],
            marker=str(marker),
            s=scatter_size,
            alpha=float(scatter_alpha),
            rasterized=bool(rasterized),
            color=point_color,
            **_marker_scatter_style(str(marker)),
        )

        # Keep QQ axis lower bounds consistent between X/Y.
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x_data_min = 0.0
        x_data_max = float(max_lim)
        if scatter_mask.any():
            exp_valid = exp_scatter[scatter_mask]
            exp_valid = exp_valid[np.isfinite(exp_valid)]
            if exp_valid.size > 0:
                x_data_min = min(x_data_min, float(np.min(exp_valid)))
                x_data_max = max(x_data_max, float(np.max(exp_valid)))
        if np.any(band_mask):
            x_band_valid = x_band[band_mask]
            x_band_valid = x_band_valid[np.isfinite(x_band_valid)]
            if x_band_valid.size > 0:
                x_data_min = min(x_data_min, float(np.min(x_band_valid)))
                x_data_max = max(x_data_max, float(np.max(x_band_valid)))
        if np.isfinite([x0, x1, y0, y1]).all():
            if axis_min is not None and np.isfinite(float(axis_min)):
                lo = float(axis_min)
            else:
                right_pad = max(0.0, float(x1) - x_data_max)
                lo = x_data_min - right_pad
            x_hi = float(x1) if float(x1) > lo else (lo + 1.0)
            if axis_max is not None and np.isfinite(float(axis_max)) and float(axis_max) > lo:
                y_hi = float(axis_max)
            else:
                y_hi = float(y1) if float(y1) > lo else (lo + 1.0)
            ax.set_xlim(lo, x_hi)
            ax.set_ylim(lo, y_hi)

        ax.set_xlabel("Expected -log10(p-value)")
        ax.set_ylabel("Observed -log10(p-value)")
        apply_integer_xticks(ax)
        apply_integer_yticks(ax)
        return ax
