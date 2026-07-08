import re
import colorsys
from typing import Any, Iterable, Optional

import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def mse(tparray: np.ndarray) -> float:
    se = np.asarray(tparray, dtype=float)[:, 0] - np.asarray(tparray, dtype=float)[:, 1]
    return float(np.mean(se * se))


def mae(tparray: np.ndarray) -> float:
    ae = np.abs(np.asarray(tparray, dtype=float)[:, 0] - np.asarray(tparray, dtype=float)[:, 1])
    return float(np.mean(ae))


def r2(tparray: np.ndarray) -> float:
    arr = np.asarray(tparray, dtype=float)
    sse1 = arr[:, 0] - arr[:, 1]
    sse2 = arr[:, 0] - np.mean(arr[:, 0])
    denom = float(sse2.T @ sse2)
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - (sse1.T @ sse1) / denom)


def _split_palette_tokens(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in str(text):
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        if ch in (";", ",") and depth == 0:
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


def _resolve_colors(palette: Optional[str], n: int) -> list[Any]:
    n = max(1, int(n))
    if palette is None or str(palette).strip() == "":
        cmap = plt.get_cmap("tab10")
        return [cmap(i % cmap.N) for i in range(n)]

    p = str(palette).strip()
    toks = _split_palette_tokens(p)
    if len(toks) > 1:
        return [toks[i % len(toks)] for i in range(n)]

    try:
        cmap = plt.get_cmap(p)
        # Prefer categorical sampling on listed colormaps to preserve contrast.
        if hasattr(cmap, "colors") and getattr(cmap, "N", 0) > 0:
            n_base = int(cmap.N)
            # For tab20-like paired palettes, interleave even/odd to avoid adjacent same-hue pairs.
            order = list(range(0, n_base, 2)) + list(range(1, n_base, 2))
            return [cmap.colors[order[i % n_base]] for i in range(n)]  # type: ignore[index]
        return [cmap(i / max(1, n - 1)) for i in range(n)]
    except Exception:
        return [p for _ in range(n)]


def _adjust_lightness(color: Any, factor: float) -> tuple[float, float, float, float]:
    r, g, b, a = mcolors.to_rgba(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l2 = max(0.0, min(1.0, l * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l2, s)
    return (r2, g2, b2, a)


def _gaussian_kde_eval_1d(values: np.ndarray, y_grid: np.ndarray, *, span_hint: float) -> np.ndarray:
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    grid = np.asarray(y_grid, dtype=float).reshape(-1)
    if vals.size == 0:
        return np.zeros_like(grid, dtype=float)
    if vals.size == 1:
        bw = max(0.05 * max(float(span_hint), 1e-6), 1e-3)
        z = (grid - float(vals[0])) / bw
        return np.exp(-0.5 * z * z) / (bw * np.sqrt(2.0 * np.pi))

    sd = float(np.nanstd(vals, ddof=1))
    if (not np.isfinite(sd)) or sd < 1e-12:
        bw = max(0.05 * max(float(span_hint), 1e-6), 1e-3)
        mu = float(np.nanmean(vals))
        z = (grid - mu) / bw
        return np.exp(-0.5 * z * z) / (bw * np.sqrt(2.0 * np.pi))

    # Silverman's rule-of-thumb bandwidth, with a conservative floor so
    # small-CV-fold violins remain smooth without depending on SciPy kernels.
    bw = 1.06 * sd * float(vals.size) ** (-0.2)
    bw = max(float(bw), max(0.01 * max(float(span_hint), 1e-6), 1e-3))
    z = (grid[:, None] - vals[None, :]) / bw
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (float(vals.size) * bw * np.sqrt(2.0 * np.pi))
    dens = np.asarray(dens, dtype=float)
    dens[~np.isfinite(dens)] = 0.0
    return dens


def scatterh(
    ttest: np.ndarray,
    ttrain: np.ndarray = None,
    color_set: list = ["black", "grey"],
    fig: plt.Figure = None,
    rasterized: bool | None = None,
):
    if not fig:
        fig = plt.figure(figsize=(5, 4), dpi=300)
    layout = np.array([["A"] * 4 + ["."] + (["C"] * 4 + ["D"]) * 4]).reshape(5, 5)
    axe = fig.subplot_mosaic(layout)
    ax1: plt.Axes = axe["A"]
    ax2: plt.Axes = axe["C"]
    ax3: plt.Axes = axe["D"]
    if rasterized is None:
        n_train = int(ttrain.shape[0]) if (ttrain is not None) else 0
        rasterized_flag = True if n_train > 2000 else False  # reduce vector size on very large plots
    else:
        rasterized_flag = bool(rasterized)

    ttest = np.asarray(ttest, dtype=float)
    tmin = float(np.nanmin(ttest))
    tmax = float(np.nanmax(ttest))
    ax2.plot([tmin, tmax], [tmin, tmax], linestyle="--", color=color_set[0], alpha=0.8, label="y = x (Ideal)")
    if ttrain is not None:
        ttrain = np.asarray(ttrain, dtype=float)
        ax2.scatter(
            ttrain[:, 0],
            ttrain[:, 1],
            color=color_set[1],
            alpha=0.4,
            marker="+",
            label="Train data",
            rasterized=rasterized_flag,
        )
    ax2.scatter(
        ttest[:, 0],
        ttest[:, 1],
        color=color_set[0],
        alpha=0.8,
        marker="*",
        label="Test data",
        rasterized=rasterized_flag,
    )
    ax2.set_xlabel("Observed Value")
    ax2.set_ylabel("Predicted Value")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3, axis="both")
    if ttrain is not None:
        stat_txt = (
            f"Train MAE: {mae(ttrain):.2f}\n"
            f"Test MAE: {mae(ttest):.2f}\n"
            f"Train R2: {r2(ttrain):.2f}\n"
            f"Test R2: {r2(ttest):.2f}"
        )
    else:
        stat_txt = f"Test MAE: {mae(ttest):.2f}\nTest R2: {r2(ttest):.2f}"
    ax2.text(
        -0.2,
        0.04,
        stat_txt,
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        color="gray",
        alpha=0.8,
        multialignment="left",
    )

    if ttrain is not None:
        ax1.hist(ttrain[:, 0], color=color_set[1], density=True, alpha=0.4, bins=20)
    ax1.hist(ttest[:, 0], color=color_set[0], density=True, alpha=0.8, bins=20)
    if ttrain is not None:
        ax3.hist(ttrain[:, 1], color=color_set[1], density=True, alpha=0.4, orientation="horizontal", bins=20)
    ax3.hist(ttest[:, 1], color=color_set[0], density=True, alpha=0.8, orientation="horizontal", bins=20)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15, wspace=0.1, hspace=0.1)


def plot_accuracy_runtime_scatter(
    ranking_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    palette: Optional[str] = "tab10",
    x_col: str = "time_cv_mean_sec",
    y_col: str = "pearsonr_cv_mean",
    label_col: str = "model",
    point_size: float = 250.0,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 7.0), dpi=300)
    df = ranking_df.copy()
    df = df[
        np.isfinite(pd.to_numeric(df[x_col], errors="coerce"))
        & np.isfinite(pd.to_numeric(df[y_col], errors="coerce"))
    ]
    if df.shape[0] == 0:
        ax.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=ax.transAxes)
        return ax

    # Style: soft dashboard-like background with white grid.
    fig = ax.figure
    if fig is not None:
        fig.patch.set_facecolor("#E8ECEF")
    ax.set_facecolor("#E8ECEF")
    ax.grid(True, linestyle="-", color="white", linewidth=1.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B0B8C1")
    ax.spines["bottom"].set_color("#B0B8C1")

    labels = df[label_col].astype(str).tolist()
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    keep_pos = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    if not np.any(keep_pos):
        ax.text(0.5, 0.5, "No positive runtime values for log-scale plotting", ha="center", va="center", transform=ax.transAxes)
        return ax
    x = x[keep_pos]
    y = y[keep_pos]
    labels = [labels[i] for i in range(len(labels)) if bool(keep_pos[i])]

    # Prefer softer palette for this figure.
    palette_used = palette
    if palette_used is None or str(palette_used).strip().lower() == "tab10":
        palette_used = "Set2"
    colors = _resolve_colors(palette_used, len(labels))

    x_min_raw = float(np.nanmin(x))
    x_max_raw = float(np.nanmax(x))
    ratio = x_max_raw / max(x_min_raw, 1e-12)
    use_log_time = ratio >= 20.0

    # Pareto-like upper envelope (runtime min, accuracy max), fitted in time domain.
    # Use log-time only when runtime span is wide.
    u_runtime = np.unique(x)
    runtime_sorted = np.sort(u_runtime)
    best_acc_by_runtime = np.array([np.nanmax(y[x == xv]) for xv in runtime_sorted], dtype=float)
    envelope_y = np.maximum.accumulate(best_acc_by_runtime)

    if runtime_sorted.size >= 2:
        fx = runtime_sorted.astype(float)
        fy = envelope_y.astype(float)
        if use_log_time:
            tx = np.log10(fx.clip(min=1e-12))
            t_smooth = np.linspace(float(np.min(tx)), float(np.max(tx)), 300)
            x_smooth = np.power(10.0, t_smooth)
        else:
            tx = fx
            t_smooth = np.linspace(float(np.min(tx)), float(np.max(tx)), 300)
            x_smooth = t_smooth
        y_smooth = np.interp(t_smooth, tx, fy)

        ax.plot(x_smooth, y_smooth, color="#457B9D", linewidth=2.2, zorder=1, alpha=0.85)

    # Glow + white-edge points.
    psize = max(float(point_size), 1.0)
    ax.scatter(
        x,
        y,
        s=psize * 1.6,
        c=colors,
        alpha=0.18,
        linewidths=0.0,
        zorder=2,
    )
    ax.scatter(
        x,
        y,
        s=psize,
        c=colors,
        edgecolors="white",
        linewidths=2.5,
        alpha=0.90,
        zorder=3,
    )

    # CV-based error bars (if available from caller).
    xerr: Optional[np.ndarray] = None
    yerr: Optional[np.ndarray] = None
    if (not use_log_time) and ("time_cv_sd" in df.columns):
        xe_all = pd.to_numeric(df["time_cv_sd"], errors="coerce").to_numpy(dtype=float)
        xe = xe_all[keep_pos]
        xe = np.where(np.isfinite(xe) & (xe > 0.0), xe, np.nan)
        if np.isfinite(xe).any():
            xerr = xe
    if "pearsonr_cv_sd" in df.columns:
        ye_all = pd.to_numeric(df["pearsonr_cv_sd"], errors="coerce").to_numpy(dtype=float)
        ye = ye_all[keep_pos]
        ye = np.where(np.isfinite(ye) & (ye > 0.0), ye, np.nan)
        if np.isfinite(ye).any():
            yerr = ye
    if xerr is not None or yerr is not None:
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor="#4C5C68",
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            alpha=0.65,
            zorder=2.7,
        )

    y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1e-12)
    dy = max(0.02 * y_span, 3e-4)
    for i, txt in enumerate(labels):
        oy = dy if (i % 2 == 0) else (0.45 * dy)
        ax.annotate(
            txt,
            (x[i], y[i]),
            xytext=(7, 6 if (i % 2 == 0) else 3),
            textcoords="offset points",
            fontsize=12,
            fontfamily="sans-serif",
            color="#1D3557",
            zorder=4,
        )

    ax.set_title(
        "Accuracy-Runtime Pareto Trade-off",
        fontsize=16,
        pad=15,
        fontweight="bold",
        color="#1D3557",
    )
    ax.set_xlabel("Runtime (sec, log scale)" if use_log_time else "Runtime (sec)", fontsize=14, labelpad=10, color="#1D3557")
    ax.set_ylabel("Accuracy (Pearson r)", fontsize=14, labelpad=10, color="#1D3557")
    ax.tick_params(axis="both", which="major", labelsize=12, colors="#1D3557")
    if use_log_time:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:g}"))
        ax.grid(True, which="minor", axis="x", linestyle="-", linewidth=0.8, color="white", alpha=0.55, zorder=0)

    # Tight but breathable limits.
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if use_log_time:
        x_lo = x_min / 1.35
        x_hi = x_max * 1.35
        if x_lo <= 0.0:
            x_lo = x_min * 0.75
        x_lo = max(x_lo, 1e-8)
    else:
        x_span = max(x_max - x_min, 1e-9)
        x_lo = x_min - max(0.08 * x_span, 0.05)
        x_hi = x_max + max(0.08 * x_span, 0.05)
    y_pad = max(0.08 * y_span, 0.0015)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(float(np.nanmin(y)) - y_pad, float(np.nanmax(y)) + y_pad)
    return ax


def plot_pearson_bar(
    ranking_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    palette: Optional[str] = "tab10",
    y_col: str = "pearsonr_cv_mean",
    label_col: str = "model",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8.2, 4.8), dpi=300)
    df = ranking_df.copy()
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df[np.isfinite(df[y_col].to_numpy(dtype=float))]
    if df.shape[0] == 0:
        ax.text(0.5, 0.5, "No valid bars", ha="center", va="center", transform=ax.transAxes)
        return ax
    df = df.sort_values(by=y_col, ascending=False).reset_index(drop=True)
    names = df[label_col].astype(str).tolist()
    vals = df[y_col].to_numpy(dtype=float)
    colors = _resolve_colors(palette, len(names))
    ax.bar(np.arange(len(names)), vals, color=colors, alpha=0.95, edgecolor="black", linewidth=0.25)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("Model Accuracy (Pearson r)")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    return ax


def _kde_density_1d(values: np.ndarray, y_grid: np.ndarray, *, span_hint: float) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(y_grid, dtype=float)
    if vals.size == 1 or float(np.nanstd(vals)) < 1e-12:
        bw = max(0.05 * max(span_hint, 1e-6), 1e-3)
        z = (y_grid - float(vals[0])) / bw
        return np.exp(-0.5 * z * z) / (bw * np.sqrt(2.0 * np.pi))
    try:
        return _gaussian_kde_eval_1d(vals, y_grid, span_hint=span_hint)
    except Exception:
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals))
        sd = max(sd, 1e-3)
        z = (y_grid - mu) / sd
        return np.exp(-0.5 * z * z) / (sd * np.sqrt(2.0 * np.pi))


def plot_accuracy_split_violin(
    cv_long_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    palette: Optional[str] = None,
    model_order: Optional[Iterable[str]] = None,
    model_col: str = "Model",
    metric_col: str = "Metric",
    value_col: str = "Accuracy",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10.0, 6.0), dpi=300)

    df = cv_long_df.copy()
    if df.shape[0] == 0:
        ax.text(0.5, 0.5, "No CV fold-level accuracy available", ha="center", va="center", transform=ax.transAxes)
        return ax

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[np.isfinite(df[value_col].to_numpy(dtype=float))]
    df[model_col] = df[model_col].astype(str)
    df[metric_col] = df[metric_col].astype(str)
    df = df[df[metric_col].isin(["Pearson r", "Spearman r"])]
    if df.shape[0] == 0:
        ax.text(0.5, 0.5, "No Pearson/Spearman fold-level values", ha="center", va="center", transform=ax.transAxes)
        return ax

    if model_order is None:
        models = sorted(df[model_col].unique().tolist(), key=_natkey)
    else:
        ordered = [str(m) for m in model_order]
        models = [m for m in ordered if m in set(df[model_col].tolist())]
        extra = [m for m in sorted(df[model_col].unique().tolist(), key=_natkey) if m not in set(models)]
        models.extend(extra)

    if len(models) == 0:
        ax.text(0.5, 0.5, "No valid model labels", ha="center", va="center", transform=ax.transAxes)
        return ax

    metric_palette: dict[str, Any] = {"Pearson r": "#72A08A", "Spearman r": "#E68C6A"}
    if palette is not None and str(palette).strip() != "":
        ptxt = str(palette).strip()
        toks = _split_palette_tokens(ptxt)
        if len(toks) >= 2:
            metric_palette = {"Pearson r": toks[0], "Spearman r": toks[1]}
        elif len(toks) == 1:
            tok = toks[0]
            if mcolors.is_color_like(tok):
                r, g, b, _ = mcolors.to_rgba(tok)
                metric_palette = {
                    "Pearson r": (r, g, b, 0.90),
                    "Spearman r": (r, g, b, 0.52),
                }
            else:
                c2 = _resolve_colors(tok, 2)
                metric_palette = {"Pearson r": c2[0], "Spearman r": c2[1]}

    y_vals_all = df[value_col].to_numpy(dtype=float)
    y_min = float(np.nanmin(y_vals_all))
    y_max = float(np.nanmax(y_vals_all))
    y_span = max(y_max - y_min, 1e-6)
    # Leave extra headroom/tailroom so KDE tails do not look clipped.
    y_pad = max(0.25 * y_span, 0.010)
    y_lo = y_min - y_pad
    y_hi = y_max + y_pad
    y_grid = np.linspace(y_lo, y_hi, 256)
    half_width = 0.42

    for i, model in enumerate(models):
        x0 = float(i)
        sub = df[df[model_col] == model]
        vals_left = sub.loc[sub[metric_col] == "Pearson r", value_col].to_numpy(dtype=float)
        vals_right = sub.loc[sub[metric_col] == "Spearman r", value_col].to_numpy(dtype=float)
        dens_left = _kde_density_1d(vals_left, y_grid, span_hint=y_span)
        dens_right = _kde_density_1d(vals_right, y_grid, span_hint=y_span)
        dmax = max(float(np.nanmax(dens_left)), float(np.nanmax(dens_right)), 1e-12)
        w_left = (dens_left / dmax) * half_width
        w_right = (dens_right / dmax) * half_width

        ax.fill_betweenx(
            y_grid,
            x0 - w_left,
            x0,
            facecolor=metric_palette["Pearson r"],
            edgecolor=metric_palette["Pearson r"],
            linewidth=1.5,
            alpha=1.0,
            zorder=2,
        )
        ax.fill_betweenx(
            y_grid,
            x0,
            x0 + w_right,
            facecolor=metric_palette["Spearman r"],
            edgecolor=metric_palette["Spearman r"],
            linewidth=1.5,
            alpha=1.0,
            zorder=2,
        )

        for vals, side in ((vals_left, "left"), (vals_right, "right")):
            if vals.size == 0:
                continue
            q1, q2, q3 = np.nanpercentile(vals, [25.0, 50.0, 75.0])
            if side == "left":
                xa, xb = x0 - half_width * 0.85, x0 - half_width * 0.10
            else:
                xa, xb = x0 + half_width * 0.10, x0 + half_width * 0.85
            for q in (q1, q2, q3):
                ax.plot(
                    [xa, xb],
                    [q, q],
                    linestyle="--",
                    linewidth=1.2,
                    color="white",
                    alpha=0.95,
                    zorder=3,
                )

    ax.set_title("Model Accuracy Comparison\n(Pearson r vs Spearman r)", fontsize=16, pad=15, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xticks(np.arange(len(models), dtype=float))
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlim(-0.7, float(len(models) - 1) + 0.7)
    ax.set_ylim(y_lo, y_hi)

    handles = [
        Patch(facecolor=metric_palette["Pearson r"], edgecolor=metric_palette["Pearson r"], alpha=1.0, label="Pearson r"),
        Patch(facecolor=metric_palette["Spearman r"], edgecolor=metric_palette["Spearman r"], alpha=1.0, label="Spearman r"),
    ]
    ax.legend(handles=handles, title="", loc="lower left", framealpha=0.9, edgecolor="gray", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    return ax


def _guess_effect_col(df: pd.DataFrame, hint: Optional[str] = None) -> str:
    if hint is not None and str(hint).strip() != "":
        h = str(hint).strip()
        if h in df.columns:
            return h
    meta_cols = {
        "chrom",
        "chr",
        "pos",
        "bp",
        "position",
        "id",
        "snp",
        "marker",
        "allele0",
        "allele1",
        "maf",
    }
    num_cols: list[str] = []
    for c in df.columns:
        if str(c).strip().lower() in meta_cols:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if np.isfinite(s.to_numpy(dtype=float)).sum() > 0:
            num_cols.append(str(c))
    if len(num_cols) == 0:
        raise ValueError("No numeric effect columns found.")
    return num_cols[0]


def _natkey(text: str) -> tuple:
    s = str(text)
    parts = re.split(r"(\d+)", s)
    key: list[Any] = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return tuple(key)


def _resolve_chr_pos_cols(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    chr_cands = ["chrom", "chr", "CHROM", "CHR", "Chromosome"]
    pos_cands = ["pos", "bp", "position", "POS", "BP"]
    chr_col = next((c for c in chr_cands if c in df.columns), None)
    pos_col = next((c for c in pos_cands if c in df.columns), None)
    return chr_col, pos_col


def _signed_effect_xy(
    df: pd.DataFrame,
    effect_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[float, str]], int]:
    chr_col, pos_col = _resolve_chr_pos_cols(df)
    eff = pd.to_numeric(df[effect_col], errors="coerce")
    keep = np.isfinite(eff.to_numpy(dtype=float))
    if chr_col is not None and pos_col is not None:
        chr_v = df[chr_col].astype(str)
        pos_v = pd.to_numeric(df[pos_col], errors="coerce")
        keep = keep & np.isfinite(pos_v.to_numpy(dtype=float))
        d = pd.DataFrame(
            {
                "_chr": chr_v[keep].astype(str).to_numpy(),
                "_pos": pos_v[keep].to_numpy(dtype=float),
                "_eff": eff[keep].to_numpy(dtype=float),
            }
        )
        if d.shape[0] == 0:
            return np.array([]), np.array([]), np.array([]), np.array([], dtype=int), [], 0
        chrs = sorted(d["_chr"].unique().tolist(), key=_natkey)
        chr_to_idx = {c: i for i, c in enumerate(chrs)}
        ticks: list[tuple[float, str]] = []
        x = np.zeros(d.shape[0], dtype=float)
        chr_idx = np.zeros(d.shape[0], dtype=int)
        offset = 0.0
        for c in chrs:
            sel = d["_chr"] == c
            sub = d.loc[sel].sort_values(by="_pos")
            loc = sub["_pos"].to_numpy(dtype=float)
            loc = loc - float(np.nanmin(loc))
            x_vals = loc + offset
            out_idx = sub.index.to_numpy(dtype=int)
            x[out_idx] = x_vals
            chr_idx[out_idx] = int(chr_to_idx[c])
            ticks.append((float(np.nanmean(x_vals)), str(c)))
            offset = float(np.nanmax(x_vals) + 1.0)
        y = d["_eff"].to_numpy(dtype=float)
        sign = np.where(y >= 0.0, 1.0, -1.0)
        return x, y, sign, chr_idx, ticks, len(chrs)
    y = eff[keep].to_numpy(dtype=float)
    x = np.arange(y.shape[0], dtype=float)
    sign = np.where(y >= 0.0, 1.0, -1.0)
    chr_idx = np.zeros(y.shape[0], dtype=int)
    return x, y, sign, chr_idx, [], 1


def plot_signed_effect(
    effect_df: pd.DataFrame,
    *,
    effect_col: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    palette: Optional[str] = "tab10",
    rasterized: bool = True,
    point_size: float = 4.0,
) -> tuple[plt.Axes, str]:
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 4.6), dpi=300)
    col = _guess_effect_col(effect_df, hint=effect_col)
    x, y, sign, chr_idx, ticks, n_chr = _signed_effect_xy(effect_df, col)
    if y.size == 0:
        ax.text(0.5, 0.5, "No valid effect points", ha="center", va="center", transform=ax.transAxes)
        return ax, col
    abs_y = np.abs(y)
    if abs_y.size <= 1 or np.all(abs_y <= 0.0):
        sizes = np.full(abs_y.shape, max(2.0 * float(point_size), 7.0))
    else:
        max_abs = float(np.nanmax(abs_y))
        if not np.isfinite(max_abs) or max_abs <= 1e-12:
            sizes = np.full(abs_y.shape, max(2.0 * float(point_size), 7.0))
        else:
            n_bins = 7
            norm = np.clip(abs_y / max_abs, 0.0, 1.0)
            bins = np.floor(norm * float(n_bins - 1)).astype(int)
            bins = np.clip(bins, 0, n_bins - 1)
            s_min = max(2.0 * float(point_size), 7.0)
            s_max = max(12.0 * float(point_size), 48.0)
            sizes = s_min + (s_max - s_min) * (bins.astype(float) / max(1.0, float(n_bins - 1)))

    mask_pos = sign >= 0
    mask_neg = ~mask_pos
    if len(ticks) > 0 and n_chr > 1:
        chr_colors = _resolve_colors(palette, n_chr)
        rgba_chr = np.asarray([mcolors.to_rgba(chr_colors[int(i)]) for i in chr_idx], dtype=float)
        # Use original chromosome color for positive effects; darken for negatives.
        rgba_pos = rgba_chr.copy()
        rgba_neg = np.asarray([_adjust_lightness(c, 0.68) for c in rgba_chr], dtype=float)
        mixed = np.where(mask_pos[:, None], rgba_pos, rgba_neg)
        mixed[:, 3] = 0.96
        ax.scatter(
            x,
            y,
            s=sizes,
            c=mixed,
            rasterized=bool(rasterized),
            linewidths=0.0,
            edgecolors="none",
        )
    else:
        base = _resolve_colors(palette, 1)[0]
        cpos = mcolors.to_rgba(base)
        cneg = _adjust_lightness(cpos, 0.68)
        if np.any(mask_neg):
            ax.scatter(
                x[mask_neg],
                y[mask_neg],
                s=sizes[mask_neg],
                c=[cneg],
                alpha=0.96,
                rasterized=bool(rasterized),
                linewidths=0.0,
                edgecolors="none",
            )
        if np.any(mask_pos):
            ax.scatter(
                x[mask_pos],
                y[mask_pos],
                s=sizes[mask_pos],
                c=[cpos],
                alpha=0.96,
                rasterized=bool(rasterized),
                linewidths=0.0,
                edgecolors="none",
            )
    ax.axhline(0.0, linestyle="--", color="black", linewidth=0.8, alpha=0.6)
    ax.set_ylabel(f"Effect ({col})")
    ax.set_xlabel("Genome Position")
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    if len(ticks) > 0:
        xt = [t[0] for t in ticks]
        xl = [t[1] for t in ticks]
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, rotation=0, fontsize=8)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    xr = max(x_max - x_min, 1e-9)
    x_pad = max(0.01 * xr, 1.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    return ax, col


def plot_effect_models_layered(
    effect_long_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    palette: Optional[str] = "tab10",
    model_col: str = "Model",
    chr_col: Optional[str] = None,
    pos_col: Optional[str] = None,
    effect_col: str = "Effect",
    rasterized: bool = True,
    point_size: float = 4.0,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300)
    df = effect_long_df.copy()
    if df.shape[0] == 0:
        ax.text(0.5, 0.5, "No effect points", ha="center", va="center", transform=ax.transAxes)
        return ax

    if chr_col is None or pos_col is None:
        c0, p0 = _resolve_chr_pos_cols(df)
        chr_col = c0
        pos_col = p0
    if chr_col is None or pos_col is None or chr_col not in df.columns or pos_col not in df.columns:
        ax.text(0.5, 0.5, "No chrom/pos columns in merged effect table", ha="center", va="center", transform=ax.transAxes)
        return ax

    df[model_col] = df[model_col].astype(str)
    df["_chr"] = df[chr_col].astype(str)
    df["_pos"] = pd.to_numeric(df[pos_col], errors="coerce")
    df["_eff"] = pd.to_numeric(df[effect_col], errors="coerce")
    df = df[np.isfinite(df["_pos"].to_numpy(dtype=float)) & np.isfinite(df["_eff"].to_numpy(dtype=float))]
    if df.shape[0] == 0:
        ax.text(0.5, 0.5, "No valid merged effect points", ha="center", va="center", transform=ax.transAxes)
        return ax

    chr_order = sorted(df["_chr"].unique().tolist(), key=_natkey)
    df["_pos_local"] = df["_pos"] - df.groupby("_chr")["_pos"].transform("min")

    chr_offset: dict[str, float] = {}
    ticks: list[tuple[float, str]] = []
    offset = 0.0
    for c in chr_order:
        sub = df[df["_chr"] == c]
        loc = sub["_pos_local"].to_numpy(dtype=float)
        chr_offset[c] = offset
        ticks.append((offset + float(np.nanmean(loc)), str(c)))
        offset += float(np.nanmax(loc) + 1.0)
    df["_x"] = df["_pos_local"] + df["_chr"].map(chr_offset).astype(float)

    abs_eff = np.abs(df["_eff"].to_numpy(dtype=float))
    max_abs = float(np.nanmax(abs_eff)) if abs_eff.size > 0 else 0.0
    if not np.isfinite(max_abs) or max_abs <= 1e-12:
        max_abs = 1.0
    norm = np.clip(abs_eff / max_abs, 0.0, 1.0)

    n_bins = 7
    bins = np.floor(norm * float(n_bins - 1)).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    s_min = max(2.0 * float(point_size), 7.0)
    s_max = max(12.0 * float(point_size), 48.0)
    df["_size"] = s_min + (s_max - s_min) * (bins.astype(float) / max(1.0, float(n_bins - 1)))

    t1 = max_abs / 3.0
    t2 = (2.0 * max_abs) / 3.0
    df["_abs_eff"] = abs_eff
    df["_tier"] = np.where(df["_abs_eff"] <= t1, "low", np.where(df["_abs_eff"] <= t2, "mid", "high"))

    d_low = df[df["_tier"] == "low"]
    if d_low.shape[0] > 0:
        ax.scatter(
            d_low["_x"].to_numpy(dtype=float),
            d_low["_eff"].to_numpy(dtype=float),
            s=d_low["_size"].to_numpy(dtype=float),
            c="#B9BEC5",
            alpha=0.35,
            rasterized=bool(rasterized),
            linewidths=0.0,
            edgecolors="none",
        )

    models = sorted(df[model_col].unique().tolist(), key=_natkey)
    colors = _resolve_colors(palette, len(models))
    model_to_color = {m: colors[i] for i, m in enumerate(models)}
    for m in models:
        d_mid = df[(df[model_col] == m) & (df["_tier"] == "mid")]
        if d_mid.shape[0] > 0:
            ax.scatter(
                d_mid["_x"].to_numpy(dtype=float),
                d_mid["_eff"].to_numpy(dtype=float),
                s=d_mid["_size"].to_numpy(dtype=float),
                c=[model_to_color[m]],
                alpha=0.55,
                rasterized=bool(rasterized),
                linewidths=0.0,
                edgecolors="none",
            )
        d_hi = df[(df[model_col] == m) & (df["_tier"] == "high")]
        if d_hi.shape[0] > 0:
            ax.scatter(
                d_hi["_x"].to_numpy(dtype=float),
                d_hi["_eff"].to_numpy(dtype=float),
                s=d_hi["_size"].to_numpy(dtype=float),
                c=[model_to_color[m]],
                alpha=0.96,
                rasterized=bool(rasterized),
                linewidths=0.0,
                edgecolors="none",
            )

    ax.axhline(0.0, linestyle="--", color="black", linewidth=0.8, alpha=0.55)
    ax.set_ylabel("Effect")
    ax.set_xlabel("Genome Position")
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    if len(ticks) > 0:
        ax.set_xticks([t[0] for t in ticks])
        ax.set_xticklabels([t[1] for t in ticks], rotation=0, fontsize=8)
    x_min = float(np.nanmin(df["_x"].to_numpy(dtype=float)))
    x_max = float(np.nanmax(df["_x"].to_numpy(dtype=float)))
    xr = max(x_max - x_min, 1e-9)
    x_pad = max(0.01 * xr, 1.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    handles = [Patch(facecolor=model_to_color[m], edgecolor="none", alpha=0.90, label=str(m)) for m in models]
    if len(handles) > 0:
        ax.legend(
            handles=handles,
            title="Model",
            loc="upper right",
            framealpha=0.90,
            edgecolor="#999999",
            fontsize=9,
            title_fontsize=9,
            ncol=1 if len(handles) <= 8 else 2,
        )
    return ax
