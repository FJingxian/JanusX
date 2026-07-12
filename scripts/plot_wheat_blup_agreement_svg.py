from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


METHOD_ORDER = [
    "HIBLUP_GBLUP",
    "sommer_GBLUP",
    "rrBLUP_GBLUP",
    "JanusX_GBLUP",
    "JanusX_rrBLUP_exact",
    "JanusX_rrBLUP_PCG",
]

DISPLAY_NAME = {
    "HIBLUP_GBLUP": "HIBLUP",
    "sommer_GBLUP": "sommer",
    "rrBLUP_GBLUP": "rrBLUP",
    "JanusX_GBLUP": "JanusX\nGBLUP",
    "JanusX_rrBLUP_exact": "JanusX\nexact",
    "JanusX_rrBLUP_PCG": "JanusX\nPCG",
}

COLOR_MAP = {
    "HIBLUP_GBLUP": "#2D7FB8",
    "sommer_GBLUP": "#F28E2B",
    "rrBLUP_GBLUP": "#3BAA3D",
    "JanusX_GBLUP": "#3B3B98",
    "JanusX_rrBLUP_exact": "#D45087",
    "JanusX_rrBLUP_PCG": "#7A5CFA",
}


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_agreement_figure(root: Path, out_svg: Path, out_png: Path | None = None) -> None:
    plots_dir = root / "plots"
    pair_rows = _read_tsv(plots_dir / "pair_metrics.tsv")
    cv_rows = _read_tsv(plots_dir / "cv_accuracy.tsv")

    methods = [m for m in METHOD_ORDER if any(r["label"] == m for r in cv_rows)]
    if len(methods) < 2:
        raise ValueError("Need at least two methods to plot agreement.")

    n_methods = len(methods)
    corr_lookup: dict[tuple[str, str], float] = {}
    min_pair_r = 1.0
    n_overlap = None
    max_abs_diff = 0.0
    max_mae = 0.0
    for row in pair_rows:
        a = row["method_a"]
        b = row["method_b"]
        if a not in methods or b not in methods:
            continue
        r = float(row["pearson_r"])
        corr_lookup[(a, b)] = r
        corr_lookup[(b, a)] = r
        min_pair_r = min(min_pair_r, r)
        if n_overlap is None:
            n_overlap = int(row["n_overlap"])
        parsed = {
            "pearson_r": r,
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "max_abs_diff": float(row["max_abs_diff"]),
        }
        max_abs_diff = max(max_abs_diff, parsed["max_abs_diff"])
        max_mae = max(max_mae, parsed["mae"])

    corr_matrix = np.full((n_methods, n_methods), np.nan, dtype=float)
    annot_matrix: list[list[str | None]] = [[None] * n_methods for _ in range(n_methods)]
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i == j:
                corr_matrix[i, j] = 1.0
                continue
            r = corr_lookup[(a, b)]
            corr_matrix[i, j] = r
            if i > j:
                annot_matrix[i][j] = f"{r:.6f}"

    fold_lookup: dict[int, dict[str, float]] = {}
    for row in cv_rows:
        fold = int(row["fold"])
        label = row["label"]
        if label not in methods:
            continue
        fold_lookup.setdefault(fold, {})[label] = float(row["pearson_r"])

    fold_ids = sorted(fold_lookup)
    max_fold_gap = 0.0
    series_by_method: dict[str, list[float]] = {m: [] for m in methods}
    fold_min: list[float] = []
    fold_max: list[float] = []
    fold_mean: list[float] = []
    for fold in fold_ids:
        fold_values = [fold_lookup[fold][method] for method in methods]
        best = max(fold_values)
        fold_min.append(min(fold_values))
        fold_max.append(max(fold_values))
        fold_mean.append(float(np.mean(fold_values)))
        for method in methods:
            value = fold_lookup[fold][method]
            series_by_method[method].append(value)
            gap = best - value
            max_fold_gap = max(max_fold_gap, gap)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "agreement_corr",
        ["#EAF2FB", "#9BBDE0", "#3F6EA6", "#153B6B"],
    )

    fig = plt.figure(figsize=(14.8, 7.2), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[15, 2.7],
        width_ratios=[1.18, 1.0],
        hspace=0.06,
        wspace=0.22,
    )
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_note = fig.add_subplot(gs[1, :])

    x = np.asarray(fold_ids, dtype=float)
    ax_curve.fill_between(
        x,
        np.asarray(fold_min, dtype=float),
        np.asarray(fold_max, dtype=float),
        color="#1F2937",
        alpha=0.10,
        zorder=1,
    )
    ax_curve.plot(
        x,
        np.asarray(fold_mean, dtype=float),
        color="#111827",
        linewidth=2.8,
        zorder=2,
    )
    for method in methods:
        ax_curve.plot(
            x,
            np.asarray(series_by_method[method], dtype=float),
            color=COLOR_MAP[method],
            linewidth=2.0,
            marker="o",
            markersize=6.2,
            alpha=0.82,
            zorder=3,
        )
    ax_curve.set_title("A. Foldwise CV curves overlap visually", loc="left", fontsize=17, pad=10)
    ax_curve.text(
        0.0,
        1.01,
        "All six method-specific curves collapse into one visible trajectory.",
        transform=ax_curve.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.2,
        color="#475569",
    )
    ax_curve.set_ylabel("CV Pearson r", fontsize=12.5)
    ax_curve.set_xticks(fold_ids)
    y_all = np.asarray([v for vals in series_by_method.values() for v in vals], dtype=float)
    y_pad = max(0.004, 0.08 * float(y_all.max() - y_all.min()))
    ax_curve.set_ylim(float(y_all.min() - y_pad), float(y_all.max() + y_pad))
    ax_curve.grid(axis="y", color="#D7DEE7", linewidth=1.0, alpha=0.8)
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)
    ax_curve.spines["left"].set_linewidth(1.8)
    ax_curve.spines["bottom"].set_linewidth(1.8)
    ax_curve.tick_params(axis="both", width=1.5, length=5, labelsize=11.5)

    mask = np.ma.masked_where(~np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1), corr_matrix)
    im = ax_heat.imshow(mask, cmap=cmap, vmin=0.9999990, vmax=1.0)

    ax_heat.set_xticks(range(n_methods))
    ax_heat.set_yticks(range(n_methods))
    ax_heat.set_xticklabels([DISPLAY_NAME[m] for m in methods], fontsize=10.0)
    ax_heat.set_yticklabels([DISPLAY_NAME[m] for m in methods], fontsize=11)
    plt.setp(ax_heat.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    ax_heat.set_title(
        "B. Pairwise correlations are all 1.000000 at six decimals",
        loc="left",
        fontsize=17,
        pad=10,
    )
    ax_heat.text(
        0.0,
        1.01,
        f"Minimum pairwise r = {min_pair_r:.9f}",
        transform=ax_heat.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.2,
        color="#475569",
    )
    ax_heat.set_xlim(-0.5, n_methods - 0.5)
    ax_heat.set_ylim(n_methods - 0.5, -0.5)
    ax_heat.set_xticks(np.arange(-0.5, n_methods, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, n_methods, 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.8)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    for i in range(n_methods):
        for j in range(n_methods):
            if i <= j or annot_matrix[i][j] is None:
                continue
            ax_heat.text(
                j,
                i,
                annot_matrix[i][j],
                ha="center",
                va="center",
                fontsize=10.4,
                color="white",
                fontweight="bold",
            )

    ax_note.axis("off")
    card_style = {
        "boxstyle": "round,pad=0.45",
        "facecolor": "#F8FAFC",
        "edgecolor": "#CBD5E1",
        "linewidth": 1.2,
    }
    cards = [
        ("Minimum pairwise r", f"{min_pair_r:.9f}"),
        ("Largest pairwise MAE", f"{max_mae:.2e}"),
        ("Largest |prediction difference|", f"{max_abs_diff:.2e}"),
        ("Largest within-fold CV gap", f"{max_fold_gap:.2e}"),
    ]
    x_pos = [0.03, 0.285, 0.54, 0.795]
    for (title, value), xpos in zip(cards, x_pos):
        ax_note.text(
            xpos,
            0.56,
            f"{title}\n{value}",
            ha="left",
            va="center",
            fontsize=11.4,
            color="#0F172A",
            fontweight="bold",
            linespacing=1.45,
            bbox=card_style,
            transform=ax_note.transAxes,
        )
    ax_note.text(
        0.03,
        0.05,
        f"n = {n_overlap or 'NA'} overlapping predictions. The practical conclusion is equivalence: "
        "all methods track the same fold pattern and their pairwise predictions are effectively identical.",
        ha="left",
        va="bottom",
        fontsize=10.8,
        color="#475569",
        transform=ax_note.transAxes,
    )

    fig.suptitle(
        "Wheat trait1 BLUP implementations are practically indistinguishable",
        fontsize=19.2,
        fontweight="bold",
        x=0.5,
        y=0.982,
    )

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    if out_png is not None:
        fig.savefig(out_png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an editable SVG summarizing near-identical BLUP predictions."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".janusx_test/1result/wheat_trait1_blup_cv_consistency"),
        help="Consistency result root containing plots/*.tsv",
    )
    parser.add_argument(
        "--out-svg",
        type=Path,
        default=None,
        help="Output SVG path. Defaults to <root>/plots/wheat_trait1_blup_agreement.svg",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Optional PNG preview path. Defaults to <root>/plots/wheat_trait1_blup_agreement.png",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out_svg = args.out_svg or root / "plots" / "wheat_trait1_blup_agreement.svg"
    out_png = args.out_png if args.out_png is not None else root / "plots" / "wheat_trait1_blup_agreement.png"
    build_agreement_figure(root=root, out_svg=out_svg, out_png=out_png)


if __name__ == "__main__":
    main()
