#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from scipy.stats import wilcoxon

from janusx.bioplotkit.sci_set import sci_set


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "test.Tgarfield_out" / "summary" / "test_breakdown"
DEFAULT_OUT_DIR = REPO_ROOT / "test.figure"

LD_SINGLE_ORDER = ["low", "high"]
LD_PAIR_ORDER = ["LL", "LH", "HH"]
MAF_SINGLE_ORDER = ["low", "mid", "high"]
MAF_PAIR_ORDER = ["LL", "LM", "LH", "MM", "MH", "HH"]


def _cmap_colors(cmap: str, n: int) -> list[tuple[float, float, float, float]]:
    cm = plt.get_cmap(str(cmap))
    if n <= 1:
        return [to_rgba(cm(0.55))]
    vals = np.linspace(0.10, 0.90, n)
    return [to_rgba(cm(float(v))) for v in vals]


def _mix(c1: tuple[float, float, float, float], c2: tuple[float, float, float, float], w1: float, w2: float) -> tuple[float, float, float, float]:
    total = max(w1 + w2, 1e-9)
    return (
        (c1[0] * w1 + c2[0] * w2) / total,
        (c1[1] * w1 + c2[1] * w2) / total,
        (c1[2] * w1 + c2[2] * w2) / total,
        1.0,
    )


def _load_ratio_values(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep="\t")
    sim = pd.to_numeric(df["sim_pwald"], errors="coerce")
    pred = pd.to_numeric(df["pred_pwald"], errors="coerce")
    mask = (sim > 0.0) & (pred > 0.0) & np.isfinite(sim) & np.isfinite(pred)
    vals = -np.log10((pred.loc[mask] / sim.loc[mask]).to_numpy(dtype=float))
    vals = vals[np.isfinite(vals)]
    return np.asarray(vals, dtype=float)


def _load_group_series(base_dir: Path, specs: list[tuple[str, str]]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for _label, fname in specs:
        path = base_dir / fname
        if not path.is_file():
            out.append(np.asarray([], dtype=float))
            continue
        out.append(_load_ratio_values(path))
    return out


def _holm_adjust(pvals: list[float]) -> list[float]:
    arr = np.asarray(pvals, dtype=float)
    if arr.size == 0:
        return []
    order = np.argsort(arr)
    out = np.empty(arr.size, dtype=float)
    prev = 0.0
    m = int(arr.size)
    for i, idx in enumerate(order):
        val = float(arr[idx]) * float(m - i)
        val = min(1.0, max(val, prev))
        prev = val
        out[idx] = val
    return out.tolist()


def _p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def _test_vs_zero(vals: np.ndarray) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if np.allclose(arr, 0.0):
        return 1.0
    try:
        return float(wilcoxon(arr, alternative="two-sided", zero_method="pratt", method="auto").pvalue)
    except Exception:
        return float("nan")


def _group_annotations(values: list[np.ndarray]) -> list[str]:
    raw_p = [_test_vs_zero(v) for v in values]
    finite_idx = [i for i, p in enumerate(raw_p) if np.isfinite(p)]
    finite_p = [raw_p[i] for i in finite_idx]
    adj_p = [float("nan")] * len(raw_p)
    if finite_p:
        finite_adj = _holm_adjust(finite_p)
        for i, p in zip(finite_idx, finite_adj):
            adj_p[i] = p
    labels: list[str] = []
    for vals, p in zip(values, adj_p):
        labels.append(f"n={len(vals)}\n{_p_to_stars(float(p))}")
    return labels


def _positions(n_single: int, n_pair: int, gap: float = 1.2) -> tuple[list[float], list[float]]:
    single = [float(i + 1) for i in range(n_single)]
    start_pair = float(n_single) + gap + 1.0
    pair = [start_pair + float(i) for i in range(n_pair)]
    return single, pair


def _draw_violin_box_scatter(
    ax: plt.Axes,
    values: list[np.ndarray],
    positions: list[float],
    colors: list[tuple[float, float, float, float]],
) -> None:
    if len(values) == 0:
        return

    vp = ax.violinplot(
        values,
        positions=positions,
        widths=0.85,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for body, color in zip(vp.get("bodies", []), colors):
        body.set_facecolor(_mix(color, (1.0, 1.0, 1.0, 1.0), 0.55, 0.45))
        body.set_edgecolor("none")
        body.set_linewidth(0.0)
        body.set_alpha(0.75)
        body.set_zorder(1.0)

    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        boxprops={
            "facecolor": (1.0, 1.0, 1.0, 0.95),
            "edgecolor": (0.0, 0.0, 0.0, 1.0),
            "linewidth": 1.0,
        },
        whiskerprops={"color": (0.0, 0.0, 0.0, 1.0), "linewidth": 1.0},
        capprops={"color": (0.0, 0.0, 0.0, 1.0), "linewidth": 1.0},
        medianprops={"color": (0.0, 0.0, 0.0, 1.0), "linewidth": 1.2},
    )
    for patch in bp.get("boxes", []):
        patch.set_zorder(2.0)
    for key in ("whiskers", "caps", "medians"):
        for artist in bp.get(key, []):
            artist.set_zorder(2.1)

    rng = np.random.default_rng(42)
    for vals, pos, color in zip(values, positions, colors):
        if len(vals) == 0:
            continue
        jitter = rng.normal(loc=pos, scale=0.05, size=len(vals))
        point_color = _mix(color, (0.0, 0.0, 0.0, 1.0), 0.8, 0.2)
        ax.scatter(
            jitter,
            vals,
            s=15,
            c=[point_color],
            edgecolors=[point_color],
            linewidths=0.3,
            alpha=0.85,
            zorder=3.0,
        )


def _decorate_axis(
    ax: plt.Axes,
    *,
    xticks: list[float],
    xticklabels: list[str],
    values: list[np.ndarray],
    split_x: float,
    single_center: float,
    pair_center: float,
    title: str,
) -> None:
    ax.axhline(0.0, color="0.40", linestyle="--", linewidth=1.0, zorder=0.5)
    ax.axvline(split_x, color="0.85", linestyle="-", linewidth=1.0, zorder=0.4)
    ax.set_xticks(xticks, xticklabels)
    ax.set_ylabel(r"$-\log_{10}(\mathrm{Pred}/\mathrm{Sim})$")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)

    ymin, ymax = ax.get_ylim()
    y_span = max(ymax - ymin, 1e-6)
    annos = _group_annotations(values)
    y_text = ymax + y_span * 0.06
    ax.text(single_center, y_text, "1-site", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(pair_center, y_text, "2-site", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for x, vals, txt in zip(xticks, values, annos):
        top = float(np.nanmax(vals)) if len(vals) > 0 else ymax
        ax.text(
            x,
            top + y_span * 0.05,
            txt,
            ha="center",
            va="bottom",
            fontsize=9,
            linespacing=0.95,
        )
    ax.text(
        0.99,
        0.99,
        "Wilcoxon vs 0\nHolm-adjusted",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="0.35",
    )
    ax.set_ylim(ymin, ymax + y_span * 0.22)


def _plot_panel(
    *,
    base_dir: Path,
    out_dir: Path,
    title: str,
    out_name: str,
    single_specs: list[tuple[str, str]],
    pair_specs: list[tuple[str, str]],
    cmap: str,
) -> Path:
    single_vals = _load_group_series(base_dir, single_specs)
    pair_vals = _load_group_series(base_dir, pair_specs)
    all_vals = single_vals + pair_vals

    n_single = len(single_specs)
    n_pair = len(pair_specs)
    single_pos, pair_pos = _positions(n_single, n_pair)
    xticks = single_pos + pair_pos
    xticklabels = [x[0] for x in single_specs] + [x[0] for x in pair_specs]

    colors = _cmap_colors(cmap, len(all_vals))
    fig, ax = plt.subplots(figsize=(max(8.2, 0.9 * len(xticks) + 3.0), 5.8))
    _draw_violin_box_scatter(ax, all_vals, xticks, colors)

    split_x = (single_pos[-1] + pair_pos[0]) / 2.0 if n_single > 0 and n_pair > 0 else float(n_single) + 0.5
    single_center = float(np.mean(single_pos)) if len(single_pos) > 0 else 0.0
    pair_center = float(np.mean(pair_pos)) if len(pair_pos) > 0 else 0.0
    _decorate_axis(
        ax,
        xticks=xticks,
        xticklabels=xticklabels,
        values=all_vals,
        split_x=split_x,
        single_center=single_center,
        pair_center=pair_center,
        title=title,
    )

    fig.tight_layout()
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Draw LD/MAF violin plots using -log10(Pred/Sim).")
    p.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing test_breakdown TSVs.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for output figures.")
    p.add_argument("--cmap", default="viridis", help="Matplotlib colormap for category colors.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        sci_set()
    except Exception as exc:
        print(f"[warn] sci_set() failed, fallback to default matplotlib style: {exc}")

    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ld_path = _plot_panel(
        base_dir=input_dir,
        out_dir=out_dir,
        title="LDscore tests",
        out_name="ld_ratio_violin.pdf",
        single_specs=[(x, f"ld_single.{x}.tsv") for x in LD_SINGLE_ORDER],
        pair_specs=[(x, f"ld_pair.{x}.tsv") for x in LD_PAIR_ORDER],
        cmap=str(args.cmap),
    )
    maf_path = _plot_panel(
        base_dir=input_dir,
        out_dir=out_dir,
        title="MAF tests",
        out_name="maf_ratio_violin.pdf",
        single_specs=[(x, f"maf_single.{x}.tsv") for x in MAF_SINGLE_ORDER],
        pair_specs=[(x, f"maf_pair.{x}.tsv") for x in MAF_PAIR_ORDER],
        cmap=str(args.cmap),
    )

    print(f"Input dir : {input_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Colormap  : {args.cmap}")
    print(f"Created   : 2 figure(s)")
    print(ld_path)
    print(maf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
