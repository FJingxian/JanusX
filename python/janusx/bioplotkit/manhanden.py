import time
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import beta
from janusx.gtools.cleaner import chrom_sort_key

_PVALUE_EPS = float(np.nextafter(0.0, 1.0))


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
        Spacing ratio between chromosomes on the x-axis
        (absolute spacing = interval_rate * max_position).
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

            # base spacing between chromosomes
            self.interval = int(interval_rate * int(pos_arr.max()))

            # boundaries of each chromosome block in the sorted table
            starts = np.r_[0, np.flatnonzero(np.diff(chr_arr) != 0) + 1]
            ends = np.r_[starts[1:], chr_arr.size] - 1

            # cumulative offsets: sum(max_pos of previous chromosomes)
            max_pos_per_chr = np.maximum.reduceat(pos_arr, starts)
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
            self.ticks_loc = (chr_min_x + chr_max_x) / 2.0
        else:
            self.interval = 0
            df["x"] = np.array([], dtype=np.int64)
            self._chr_bounds_min = np.empty(0, dtype=np.float64)
            self._chr_bounds_max = np.empty(0, dtype=np.float64)
            self._chr_separators = np.empty(0, dtype=np.float64)
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
        plot_kwargs.setdefault("edgecolors", "none")
        plot_kwargs.setdefault("linewidths", 0.0)

        mask_ignore = ~df.index.isin(ignore)
        ax.scatter(
            df.loc[mask_ignore, "x"],
            df.loc[mask_ignore, "y"],
            color=df.loc[mask_ignore, "z"].map(chr_color_map),
            **plot_kwargs,
        )

        # Highlight SNPs above genome-wide threshold
        if threshold is not None and df["y"].max() >= threshold:
            df_sig = df[df["y"] >= threshold]
            sig_mask = ~df_sig.index.isin(ignore)
            sig_kwargs = dict(kwargs)
            sig_kwargs.setdefault("alpha", 0.9)
            sig_kwargs.setdefault("edgecolors", "none")
            sig_kwargs.setdefault("linewidths", 0.0)
            ax.scatter(
                df_sig.loc[sig_mask, "x"],
                df_sig.loc[sig_mask, "y"],
                color="red",
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
        return ax

    # ------------------------------------------------------------------
    # QQ plot
    # ------------------------------------------------------------------
    def qq(
        self,
        ax: plt.Axes = None,
        ci: int = 95,
        color_set: list[str] = None,
        scatter_size: float = 8.0,
        line_color: str = "black",
        qq_mode: str = "auto",
        qq_auto_threshold: int = 1_000_000,
        qq_fast_max_points: int = 120_000,
        qq_band_max_points: Union[int, None] = 20_000,
        sig_p_threshold: Union[float, None] = None,
        axis_min: Union[float, None] = None,
    ) -> plt.Axes:
        """
        Draw a QQ plot of observed vs expected -log10(p) with a beta-based
        confidence band.

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

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if color_set is None or len(color_set) == 0:
            color_set = ["black", "grey"]
        if len(color_set) >= 2:
            point_color = color_set[0]
            band_color = color_set[1]
        else:
            point_color = color_set[0]
            band_color = color_set[0]

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
            # Fast QQ: deterministic rank-grid decimation from full QQ,
            # while keeping all significant points.
            max_points = int(max(1, qq_fast_max_points))
            if n > max_points:
                base_idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
                if sig_n > 0:
                    sig_idx = np.arange(sig_n, dtype=np.int64)
                    draw_idx = np.unique(np.concatenate([base_idx, sig_idx]))
                else:
                    draw_idx = np.unique(base_idx)
            else:
                draw_idx = np.arange(n, dtype=np.int64)

        p_draw = p_sorted[draw_idx]
        ranks_draw = draw_idx.astype(np.float64) + 1.0
        obs_scatter = -np.log10(p_draw)
        exp_scatter = -np.log10(ranks_draw / (n + 1.0))

        # Confidence band is always based on the full null sample size n.
        band_n = n
        if qq_band_max_points is not None and band_n > int(qq_band_max_points):
            n_band = max(2, int(qq_band_max_points))
            # Log-dense rank sampling: denser at small ranks (QQ right tail),
            # smoother tail shape than uniform linear-rank sampling.
            i = np.unique(np.round(np.geomspace(1.0, float(band_n), num=n_band)).astype(np.int64))
            if i[0] != 1:
                i = np.insert(i, 0, 1)
            if i[-1] != band_n:
                i = np.append(i, band_n)
        else:
            i = np.arange(1, band_n + 1, dtype=np.int64)

        # Build confidence band on a rank grid in ascending x
        i = np.sort(i)[::-1]
        ci_frac = float(ci)
        if ci_frac > 1.0:
            ci_frac /= 100.0
        ci_frac = float(np.clip(ci_frac, 1e-12, 1.0 - 1e-12))
        alpha = 1.0 - ci_frac
        q_lo = alpha / 2.0
        q_hi = 1.0 - alpha / 2.0
        x_band = -np.log10(i.astype(np.float64) / (band_n + 1.0))
        # True two-sided CI on p-scale, then transformed to -log10 scale:
        # lower_y uses upper p-quantile; upper_y uses lower p-quantile.
        p_upper = beta.ppf(q_hi, i, band_n - i + 1)
        p_lower = beta.ppf(q_lo, i, band_n - i + 1)
        lower = -np.log10(p_upper)
        upper = -np.log10(p_lower)
        band_mask = np.isfinite(x_band) & np.isfinite(lower) & np.isfinite(upper)

        # Draw confidence interval
        if np.any(band_mask):
            ax.fill_between(
                x_band[band_mask],
                lower[band_mask],
                upper[band_mask],
                color=band_color,
                alpha=0.3,
                rasterized=True,
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
            s=scatter_size,
            alpha=0.75,
            rasterized=True,
            color=point_color,
            edgecolors="none",
            linewidths=0.0,
        )

        # Keep QQ axis lower bounds consistent between X/Y.
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if np.isfinite([x0, x1, y0, y1]).all():
            if axis_min is not None and np.isfinite(float(axis_min)):
                lo = float(axis_min)
            else:
                lo = float(min(x0, y0))
            x_hi = float(x1) if float(x1) > lo else (lo + 1.0)
            y_hi = float(y1) if float(y1) > lo else (lo + 1.0)
            ax.set_xlim(lo, x_hi)
            ax.set_ylim(lo, y_hi)

        ax.set_xlabel("Expected -log10(p-value)")
        ax.set_ylabel("Observed -log10(p-value)")
        return ax
