import time
import re
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import beta


def ppoints(n: int) -> np.ndarray:
    """
    Approximate expected quantiles of the Uniform(0,1) distribution.

    Equivalent to R's ppoints(n) with a simple n/(n+1) scheme:
        q_i = i / (n + 1), i = 1..n
    """
    return np.arange(1, n + 1) / (n + 1)


def _natural_tokens(text: str) -> list[tuple[int, Union[int, str]]]:
    """
    Tokenize string for natural sorting with comparable typed tokens.
    """
    tokens: list[tuple[int, Union[int, str]]] = []
    for part in re.split(r"(\d+)", text):
        if not part:
            continue
        if part.isdigit():
            tokens.append((0, int(part)))
        else:
            tokens.append((1, part.lower()))
    return tokens


def _chrom_sort_key(label: object) -> tuple[int, object]:
    """
    Natural ordering for chromosome labels:
      1) numeric chromosomes (1,2,...)
      2) X/Y/M/MT
      3) other contigs by natural sort
      4) missing values last
    """
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

        # ---- (1) Optional down-sampling for plotting ----
        if compression:
            n_snp = df.shape[0]
            # target: ~100,000 SNPs for display
            chunk_size = n_snp // 100_000 if n_snp >= 100_000 else 1

            # p-value threshold: small p => highly significant
            # we keep all SNPs with p <= p_thresh without compression
            p_thresh = 10_000 / n_snp

            # Among the "less significant" SNPs (p > p_thresh), we down-sample:
            #   - sort by p descending (largest p first, least significant),
            #   - take an integer multiple of chunk_size rows,
            #   - then in each chunk keep the smallest p (most significant in that chunk).
            mask_large_p = df[pvalue] > p_thresh
            n_large = int(mask_large_p.sum())
            n_chunks = n_large // chunk_size

            if n_chunks > 0:
                # indices of "large p" SNPs, sorted by p descending
                idx_large_sorted = (
                    df.loc[mask_large_p]
                    .sort_values(pvalue, ascending=False)
                    .iloc[: n_chunks * chunk_size, :]
                    .index
                )
                df_large = df.loc[idx_large_sorted].sort_index()

                # convert to array to find minima in each chunk of size chunk_size
                pvals_large = df_large[pvalue].values.reshape(n_chunks, chunk_size)
                local_min_pos = np.argmin(pvals_large, axis=1)  # position within each chunk

                # map chunk index + local min position back to global index
                keep_idx_large = df_large.iloc[
                    [i * chunk_size + j for i, j in enumerate(local_min_pos)]
                ].index.tolist()
            else:
                keep_idx_large = []

            # always keep all highly significant SNPs
            keep_idx_small = df.loc[~mask_large_p].index.tolist()

            # final kept SNP indices for plotting
            self.minidx = keep_idx_large + keep_idx_small
        else:
            self.minidx = df.index.to_list()

        # ---- (2) Core columns copy to avoid side effects ----
        df = df[[chr, pos, pvalue]].copy()

        # ---- (3) Map chromosome labels to integers 1..K ----
        observed_labels = df[chr].drop_duplicates().tolist()

        if chr_order is None:
            self.chr_labels = sorted(observed_labels, key=_chrom_sort_key)
        else:
            observed_set = set(observed_labels)
            ordered_present = [label for label in chr_order if label in observed_set]
            ordered_set = set(ordered_present)
            missing = [label for label in observed_labels if label not in ordered_set]
            self.chr_labels = ordered_present + sorted(missing, key=_chrom_sort_key)

        chr_map = dict(zip(self.chr_labels, range(1, 1 + len(self.chr_labels))))
        df[chr] = df[chr].map(chr_map).astype(int)

        # Sort by chromosome and position
        df = df.sort_values(by=[chr, pos])
        self.chr_ids = df[chr].unique()

        # ---- (4) Build cumulative genomic x-coordinate ----
        # base spacing between chromosomes
        self.interval = int(interval_rate * df[pos].max())
        df["x"] = df[pos]

        if len(self.chr_ids) > 1:
            for cid in self.chr_ids:
                if cid > 1:
                    prev_max = df.loc[df[chr] == cid - 1, "x"].max()
                    df.loc[df[chr] == cid, "x"] += prev_max

        # add extra spacing between chromosomes
        df["x"] = (df[chr] - 1) * self.interval + df["x"]

        # rename standardized y/z columns
        df["y"] = df[pvalue]        # raw p-values
        df["z"] = df[chr]           # integer chromosome ID

        # chromosome tick positions on x-axis
        self.ticks_loc = df.groupby("z")["x"].mean()

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
        df["y"] = -np.log10(df["y"])
        df = df[df["y"] >= 0.5]  # basic cutoff to avoid extreme low points

        if ax is None:
            fig = plt.figure(figsize=[12, 6], dpi=300)
            gs = GridSpec(12, 1, figure=fig)
            ax = fig.add_subplot(gs[0:12, 0])

        # color per chromosome (alternating colors)
        chr_color_map = dict(
            zip(
                self.chr_ids,
                [color_set[i % len(color_set)] for i in range(len(self.chr_ids))],
            )
        )

        mask_ignore = ~df.index.isin(ignore)
        ax.scatter(
            df.loc[mask_ignore, "x"],
            df.loc[mask_ignore, "y"],
            color=df.loc[mask_ignore, "z"].map(chr_color_map),
            **kwargs,
        )

        # Highlight SNPs above genome-wide threshold
        if threshold is not None and df["y"].max() >= threshold:
            df_sig = df[df["y"] >= threshold]
            sig_mask = ~df_sig.index.isin(ignore)
            ax.scatter(
                df_sig.loc[sig_mask, "x"],
                df_sig.loc[sig_mask, "y"],
                color="red",
                **kwargs,
            )
            ax.axhline(
                y=threshold,
                color="grey",
                linewidth=1,
                linestyle="--",
            )

        # axis cosmetics
        ax.set_xticks(self.ticks_loc, self.chr_labels)
        ax.set_xlim([-self.interval, df["x"].max() + self.interval])
        ymax = df["y"].max()
        ax.set_ylim([0.5, ymax + 0.1 * ymax])
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
    ) -> plt.Axes:
        """
        Draw a QQ plot of observed vs expected -log10(p) with a beta-based
        confidence band.

        This implementation matches the original logic you used:
        - Use all SNPs (no extra down-sampling beyond self.minidx indices)
        - Map back to the original index before combining observed/expected
        - Use self.minidx for consistency with the Manhattan down-sampling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new figure is created.
        ci : int, default 95
            Confidence interval (in percent) for the null band.
        color_set : list[str], optional
            Color settings; first color is used for band and points.
        scatter_size : float, default 8.0
            Marker size for QQ scatter points.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if color_set is None or len(color_set) == 0:
            color_set = ["black", "grey"]

        # Use a fresh copy so we don't accidentally modify self.df
        df = self.df.copy()

        if ax is None:
            fig = plt.figure(figsize=[12, 6], dpi=600)
            gs = GridSpec(12, 1, figure=fig)
            ax = fig.add_subplot(gs[0:12, 0])

        # Raw p-values (full set, not compressed)
        p = df["y"].fillna(1)
        n = len(p)
        if n == 0:
            raise ValueError("No p-values found for QQ plot.")

        # Step 1: sort observed p-values and assign expected uniform quantiles
        #   o_e: index = SNP index (from df),
        #        columns 'o' (observed p), 'e' (expected p under Uniform)
        o_e = p.sort_values().to_frame()
        o_e["e"] = ppoints(n)
        o_e.columns = ["o", "e"]

        # Step 2: reorder back by the original index order,
        #         then transform both observed and expected to -log10 scale
        #         so that indices align with df (this is key for self.minidx).
        o_e = -np.log10(o_e.sort_index())

        # Theoretical expected -log10(p) values sorted ascending
        # (used for drawing confidence band and diagonal reference line)
        e_theoretical = o_e["e"].sort_values().values

        # Step 3: beta-based confidence band under the null
        # For each theoretical point with -log10(p) = e, the underlying p is 10^-e,
        # so expected rank is ~p * n, and we use Beta(r, n-r+1) to get a band.
        xi = np.ceil(10 ** (-e_theoretical) * n)  # ranks
        # Make sure indices are valid for Beta; your original code did not clip,
        # but xi is always in [1, n] here, so we keep behavior the same.
        lower = -np.log10(beta.ppf(1 - ci / 100, xi, n - xi + 1))
        upper = -np.log10(beta.ppf(ci / 100, xi, n - xi + 1))

        # Draw confidence interval
        ax.fill_between(
            e_theoretical,
            lower,
            upper,
            color=color_set[0],
            alpha=0.3,
            rasterized=True,
        )

        # Diagonal reference line y = x
        max_lim = np.min(o_e.max(axis=0))
        ax.plot([0, max_lim], [0, max_lim], lw=1, color="grey")

        # Scatter observed vs expected for the same subset used in Manhattan plot
        #   o_e.iloc[self.minidx, 1] -> expected -log10(p)
        #   o_e.iloc[self.minidx, 0] -> observed -log10(p)
        ax.scatter(
            o_e.iloc[self.minidx, 1],
            o_e.iloc[self.minidx, 0],
            s=scatter_size,
            alpha=0.8,
            rasterized=True,
            color=color_set[0],
        )

        ax.set_xlabel("Expected -log10(p-value)")
        ax.set_ylabel("Observed -log10(p-value)")
        return ax
