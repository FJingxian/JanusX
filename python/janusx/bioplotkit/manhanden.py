import time
from functools import lru_cache
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.gridspec import GridSpec
from scipy.stats import beta
from janusx.gtools.cleaner import chrom_sort_key

_PVALUE_EPS = float(np.nextafter(0.0, 1.0))
_QQ_SAMPLE_TRIGGER = 20_000
_QQ_SAMPLE_KEEP_HEAD = 25_000
_QQ_SAMPLE_MAX_POINTS = 50_000
_QQ_BAND_MAX_POINTS = 512


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
    limit = _QQ_BAND_MAX_POINTS if max_points is None else int(max(1, int(max_points)))
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

    keep_head = min(n_total, keep_n, int(_QQ_SAMPLE_KEEP_HEAD))
    head_idx = np.arange(keep_head, dtype=np.int64)
    tail_need = max(0, keep_n - keep_head)

    if tail_need <= 0 or keep_head >= n_total:
        arr = head_idx[:keep_n]
        arr.setflags(write=False)
        return arr

    tail_idx = np.unique(
        np.round(
            np.geomspace(float(keep_head), float(n_total - 1), num=int(max(2, tail_need + 1)))
        ).astype(np.int64)
    )
    tail_idx = tail_idx[(tail_idx >= keep_head) & (tail_idx < n_total)]
    if tail_idx.size == 0 or tail_idx[-1] != n_total - 1:
        tail_idx = np.append(tail_idx, n_total - 1)
    if tail_idx.size > tail_need:
        select = np.linspace(0, tail_idx.size - 1, num=tail_need, dtype=np.int64)
        tail_idx = tail_idx[np.unique(select)]

    arr = np.unique(np.concatenate([head_idx, tail_idx])).astype(np.int64, copy=False)
    if arr.size > keep_n:
        arr = arr[:keep_n]
    elif arr.size < keep_n:
        fill_need = keep_n - arr.size
        fill_idx = np.linspace(keep_head, n_total - 1, num=fill_need + 2, dtype=np.int64)[1:-1]
        arr = np.unique(np.concatenate([arr, fill_idx])).astype(np.int64, copy=False)
        if arr.size > keep_n:
            arr = arr[:keep_n]
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

            # chr IDs are remapped to 1..K; direct index lookup is safe.
            offset_by_chr = np.zeros(int(chr_arr.max()) + 1, dtype=np.int64)
            offset_by_chr[self.chr_ids.astype(np.int64)] = chr_offsets
            x_arr = pos_arr + offset_by_chr[chr_arr] + (chr_arr - 1) * self.interval
            df["x"] = x_arr

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
        else:
            self.interval = 0
            df["x"] = np.array([], dtype=np.int64)
            self._chr_bounds_min = np.empty(0, dtype=np.float64)
            self._chr_bounds_max = np.empty(0, dtype=np.float64)
            self._chr_separators = np.empty(0, dtype=np.float64)
            self._edge_padding_x = 0.0
            self.ticks_loc = np.empty(0, dtype=np.float64)

        # rename standardized y/z columns
        df["y"] = df[pvalue]        # raw p-values
        df["z"] = df[chr]           # integer chromosome ID

        # store as MultiIndex (chr, pos) for easier masking later
        self.df = df.set_index([chr, pos])

    # ------------------------------------------------------------------
    # Manhattan plot
    # ------------------------------------------------------------------
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
        line_color : str, default "black"
            Color of the ideal diagonal line y=x.
        qq_mode : {"auto","full","fast"}, default "auto"
            QQ computation mode:
            - auto: use fast mode when SNP count > qq_auto_threshold
            - full: full-rank QQ using all SNPs (exact)
            - fast: rank-grid approximation from full QQ (deterministic)
        qq_auto_threshold : int, default 20_000
            Threshold used when qq_mode="auto".
        qq_fast_max_points : int, default 50_000
            Max points in fast-mode QQ scatter.
        qq_band_max_points : int or None, default 512
            Max points used for the exact confidence band (both modes).
            Band sampling keeps only a small exact head and log-samples the
            remaining ranks so multi-million-SNP QQ plots stay responsive.
        sig_p_threshold : float or None, default None
            Reserved for backward compatibility with older QQ APIs.
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

        resolved_mode = mode_key
        if mode_key == "auto":
            resolved_mode = "fast" if n > int(qq_auto_threshold) else "full"

        # Full sorted p-values are the canonical QQ backbone.
        p_sorted = np.sort(p, kind="mergesort")

        if resolved_mode == "full":
            # Exact QQ: all SNPs
            draw_idx = np.arange(n, dtype=np.int64)
        else:
            # Non-full QQ keeps the first 25k ranks and log-samples the rest.
            draw_idx = _qq_sample_draw_indices(n, max_points=qq_fast_max_points)

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
            alpha=0.75,
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
