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
from scipy.stats import gaussian_kde

from janusx.bioplotkit.sci_set import sci_set


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "test.figure"
DEFAULT_CMAP = "viridis"


def _better_mask(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["pred_pwald"], errors="coerce") <= pd.to_numeric(df["sim_pwald"], errors="coerce")


def _plot_ready_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sim_pwald"] = pd.to_numeric(out["sim_pwald"], errors="coerce")
    out["pred_pwald"] = pd.to_numeric(out["pred_pwald"], errors="coerce")
    out["exact_match"] = out["exact_match"].astype(bool)
    out = out.loc[(out["sim_pwald"] > 0.0) & (out["pred_pwald"] > 0.0)].copy()
    out["sim_logp"] = -np.log10(out["sim_pwald"])
    out["pred_logp"] = -np.log10(out["pred_pwald"])
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["sim_logp", "pred_logp"])
    return out


def _density_values(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return np.ones(len(x), dtype=float)
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)
    except Exception:
        z = np.ones(len(x), dtype=float)
    if not np.all(np.isfinite(z)):
        z = np.ones(len(x), dtype=float)
    return np.asarray(z, dtype=float)


def _sort_by_density(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(z) == 0:
        return x, y, z
    idx = np.argsort(z)
    return x[idx], y[idx], z[idx]


def _annotation_text(df: pd.DataFrame) -> str:
    exact_rate = float(df["exact_match"].mean()) if len(df) > 0 else float("nan")
    better = _better_mask(df)
    exact_or_better = float((df["exact_match"] | better).mean()) if len(df) > 0 else float("nan")
    return "\n".join(
        [
            f"n = {len(df)}",
            f"exact = {exact_rate:.1%}",
            f"exact|better = {exact_or_better:.1%}",
        ]
    )


def _draw_one(df: pd.DataFrame, *, title: str, out_path: Path, cmap: str) -> None:
    plot_df = _plot_ready_frame(df)
    if plot_df.empty:
        return

    x = plot_df["sim_logp"].to_numpy()
    y = plot_df["pred_logp"].to_numpy()
    z = _density_values(x, y)
    x, y, z = _sort_by_density(x, y, z)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sc = ax.scatter(
        x,
        y,
        c=z,
        s=28,
        cmap=cmap,
        alpha=0.88,
        edgecolors="none",
    )
    fig.colorbar(sc, ax=ax, label="Point density")

    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    pad = max(0.3, (hi - lo) * 0.04)
    lims = [lo - pad, hi + pad]
    ax.plot(lims, lims, "--", linewidth=1.1, color="0.35")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(r"Simulated rule $-\log_{10}(p)$")
    ax.set_ylabel(r"Predicted rule $-\log_{10}(p)$")
    ax.set_title(title)
    ax.text(
        0.02,
        0.98,
        _annotation_text(plot_df),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_groups(path: Path, cmap: str, out_dir: Path) -> list[Path]:
    df = pd.read_csv(path, sep="\t")
    required = {"sim_pwald", "pred_pwald", "exact_match"}
    if not required.issubset(df.columns):
        return []

    stem = path.stem
    created: list[Path] = []
    logic_modes: list[str] = []
    if "logic_mode" in df.columns:
        logic_modes = sorted({str(x).strip() for x in df["logic_mode"].fillna("").tolist() if str(x).strip() != ""})

    groups: list[tuple[str, pd.DataFrame, str]] = [(stem, df.copy(), stem.replace("_", " "))]
    if len(logic_modes) > 1:
        for mode in logic_modes:
            sub = df.loc[df["logic_mode"].astype(str).str.strip() == mode].copy()
            if not sub.empty:
                groups.append((f"{stem}.{mode}", sub, f"{stem.replace('_', ' ')} [{mode}]"))

    for name, sub, title in groups:
        out_path = out_dir / f"{name}.pdf"
        _draw_one(sub, title=title, out_path=out_path, cmap=cmap)
        if out_path.exists():
            created.append(out_path)
    return created


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot all test_breakdown TSVs as density scatters.")
    p.add_argument("--input-dir", default=str(BASE_DIR), help="Directory containing test_breakdown TSVs.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for output figures.")
    p.add_argument("--cmap", default=DEFAULT_CMAP, help="Matplotlib colormap for point density.")
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

    created: list[Path] = []
    for path in sorted(input_dir.glob("*.tsv")):
        created.extend(_plot_groups(path, cmap=str(args.cmap), out_dir=out_dir))

    print(f"Input dir : {input_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Colormap  : {args.cmap}")
    print(f"Created   : {len(created)} figure(s)")
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
