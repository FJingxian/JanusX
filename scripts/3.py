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

from janusx.bioplotkit.sci_set import sci_set


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "test.Tgarfield_out3" / "summary" / "test_breakdown"
DEFAULT_OUT_DIR = REPO_ROOT / "test.figure3"
LAYER_ORDER = ["1 site", "2 sites AND", "2 sites OR", "3 sites"]


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).map({"True": True, "False": False}).fillna(False)


def _family_from_name(name: str) -> str:
    if name.startswith("chunk_size"):
        return "chunk_random"
    if name.startswith("maf_pair"):
        return "maf_pair"
    if name.startswith("ld_pair"):
        return "ld_pair"
    if name.startswith("maf_single"):
        return "maf_single"
    if name.startswith("ld_single"):
        return "ld_single"
    return "other"


def _layer_from_row(row: pd.Series) -> str:
    logic = str(row.get("logic", "")).strip().lower()
    causal_size = pd.to_numeric(row.get("causal_size"), errors="coerce")
    if logic == "single" or causal_size == 1:
        return "1 site"
    if causal_size == 2 and logic == "and":
        return "2 sites AND"
    if causal_size == 2 and logic == "or":
        return "2 sites OR"
    if causal_size == 3:
        return "3 sites"
    return "other"


def _load_results(summary_dir: Path) -> pd.DataFrame:
    path = summary_dir / "results.tsv"
    df = pd.read_csv(path, sep="\t")
    for col in ["top_rule_match", "exact_rule_match", "simbench_pwald_le_best_window"]:
        if col in df.columns:
            df[col] = _coerce_bool(df[col])
    df["logic"] = df["logic_mode"].fillna("").replace({"": "single"}).astype(str)
    if "causal_size" in df.columns:
        df["causal_size"] = pd.to_numeric(df["causal_size"], errors="coerce").fillna(0).astype(int)
    else:
        df["causal_size"] = 0
    df["best_window_site_count"] = pd.to_numeric(df["best_window_site_count"], errors="coerce").fillna(0).astype(int)
    df["layer"] = df.apply(_layer_from_row, axis=1)
    return df


def _load_breakdown_pseudo(base: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not base.is_dir():
        return pd.DataFrame()
    for path in sorted(base.glob("*.tsv")):
        df = pd.read_csv(path, sep="\t")
        if "logic_mode" not in df.columns or "pred_pseudo_maf" not in df.columns:
            continue
        df = df.copy()
        df["family"] = _family_from_name(path.stem)
        df["logic"] = df["logic_mode"].fillna("").astype(str)
        df["pred_pseudo_maf"] = pd.to_numeric(df["pred_pseudo_maf"], errors="coerce")
        df["pred_site_count"] = pd.to_numeric(df["pred_site_count"], errors="coerce").fillna(0).astype(int)
        df["exact_match"] = _coerce_bool(df["exact_match"])
        rows.append(df[["family", "logic", "pred_pseudo_maf", "pred_site_count", "exact_match"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _site_count_distribution(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = [1, 2, 3, 4, 5]

    overall = (
        df["best_window_site_count"]
        .value_counts()
        .reindex(bins, fill_value=0)
        .rename_axis("site_count")
        .reset_index(name="n")
    )
    overall["prop"] = overall["n"] / max(len(df), 1)

    layer_rows: list[dict[str, object]] = []
    layer_df = df[df["layer"].isin(LAYER_ORDER)].copy()
    for layer, sub in layer_df.groupby("layer"):
        vc = sub["best_window_site_count"].value_counts().reindex(bins, fill_value=0)
        denom = max(len(sub), 1)
        for site_count, n in vc.items():
            layer_rows.append(
                {
                    "layer": layer,
                    "site_count": int(site_count),
                    "n": int(n),
                    "prop": float(n) / float(denom),
                }
            )
    by_layer = pd.DataFrame(layer_rows)
    return overall, by_layer


def _family_layer_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["layer"].isin(LAYER_ORDER)]
        .groupby(["family", "layer"], dropna=False)
        .agg(
            n=("experiment_id", "size"),
            top_rule_match=("top_rule_match", "mean"),
            exact_rule_match=("exact_rule_match", "mean"),
            simbench_pwald_le_best_window=("simbench_pwald_le_best_window", "mean"),
            mean_pred_sites=("best_window_site_count", "mean"),
            median_pred_sites=("best_window_site_count", "median"),
        )
        .reset_index()
    )


def _layer_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["layer"].isin(LAYER_ORDER)].copy()
    return (
        sub.groupby(["layer"], dropna=False)
        .agg(
            n=("experiment_id", "size"),
            top_rule_match=("top_rule_match", "mean"),
            exact_rule_match=("exact_rule_match", "mean"),
            simbench_pwald_le_best_window=("simbench_pwald_le_best_window", "mean"),
            mean_pred_sites=("best_window_site_count", "mean"),
            median_pred_sites=("best_window_site_count", "median"),
        )
        .reset_index()
    )


def _pseudo_summary(pseudo_df: pd.DataFrame) -> pd.DataFrame:
    if pseudo_df.empty:
        return pd.DataFrame()
    return (
        pseudo_df.groupby(["family", "logic"], dropna=False)
        .agg(
            n=("pred_pseudo_maf", "size"),
            median_pseudo_maf=("pred_pseudo_maf", "median"),
            mean_pseudo_maf=("pred_pseudo_maf", "mean"),
            median_pred_sites=("pred_site_count", "median"),
            exact_match=("exact_match", "mean"),
        )
        .reset_index()
    )


def _plot_distribution(overall: pd.DataFrame, by_layer: pd.DataFrame, out_path: Path) -> None:
    try:
        sci_set()
    except Exception as exc:
        print(f"[warn] sci_set() failed, fallback to default matplotlib style: {exc}")

    count_plot_bins = [1, 2, 3, 4, 5]
    prop_plot_bins = [1, 2, 3, 4, 5]
    overall_plot = overall[overall["site_count"].isin(count_plot_bins)].copy()
    by_layer_plot = by_layer[by_layer["site_count"].isin(prop_plot_bins)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    x_all = overall_plot["site_count"].to_numpy(dtype=int)
    axes[0].bar(
        x_all,
        overall_plot["n"].to_numpy(dtype=float),
        color="#4C78A8",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0].set_title("All results")
    axes[0].set_xlabel("GARFIELD found site count")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(x_all)
    axes[0].grid(axis="y", linestyle=":", alpha=0.35)

    layer_order = [x for x in LAYER_ORDER if x in set(by_layer_plot["layer"].tolist())]
    x_logic = prop_plot_bins
    width = 0.18 if len(layer_order) >= 4 else 0.24
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(layer_order)) if len(layer_order) > 1 else np.asarray([0.0])
    colors = {
        "1 site": "#9E9E9E",
        "2 sites AND": "#1F77B4",
        "2 sites OR": "#E45756",
        "3 sites": "#59A14F",
    }
    for layer, offset in zip(layer_order, offsets):
        sub = (
            by_layer_plot[by_layer_plot["layer"] == layer]
            .set_index("site_count")
            .reindex(prop_plot_bins, fill_value=0)
            .reset_index()
        )
        axes[1].bar(
            np.asarray(x_logic, dtype=float) + float(offset),
            sub["prop"].to_numpy(dtype=float),
            width=width,
            label=layer,
            color=colors.get(layer, None),
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
        )
    axes[1].set_title("By causal layer")
    axes[1].set_xlabel("GARFIELD found site count")
    axes[1].set_ylabel("Proportion")
    axes[1].set_xticks(x_logic)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", linestyle=":", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnose GARFIELD site-count distribution and and/or behavior.")
    p.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Summary/test_breakdown directory.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Figure output directory.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).resolve()
    summary_dir = input_dir.parent
    out_root = summary_dir.parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(summary_dir)
    pseudo_df = _load_breakdown_pseudo(input_dir)

    overall, by_layer = _site_count_distribution(results)
    fam_layer = _family_layer_summary(results)
    layer_summary = _layer_summary(results)
    pseudo_summary = _pseudo_summary(pseudo_df)

    overall_path = summary_dir / "garfield_site_count_distribution.overall.tsv"
    logic_path = summary_dir / "garfield_site_count_distribution.by_layer.tsv"
    fam_logic_path = summary_dir / "garfield_logic_diagnostic.by_family_layer.tsv"
    causal_logic_path = summary_dir / "garfield_logic_diagnostic.by_layer.tsv"
    pseudo_path = summary_dir / "garfield_logic_diagnostic.pseudo_maf.tsv"

    overall.to_csv(overall_path, sep="\t", index=False)
    by_layer.to_csv(logic_path, sep="\t", index=False)
    fam_layer.to_csv(fam_logic_path, sep="\t", index=False)
    layer_summary.to_csv(causal_logic_path, sep="\t", index=False)
    if not pseudo_summary.empty:
        pseudo_summary.to_csv(pseudo_path, sep="\t", index=False)

    fig_path = out_dir / f"{out_root.name}.garfield_site_count_distribution.pdf"
    _plot_distribution(overall, by_layer, fig_path)

    print(f"Out root   : {out_root}")
    print(f"Input dir  : {input_dir}")
    print(f"Figure dir : {out_dir}")
    print(f"Rows       : {len(results)}")
    print("Layer counts:")
    print(results["layer"].value_counts().to_string())
    print("\nBy family / layer:")
    print(fam_layer.to_csv(sep='\t', index=False))
    print("By layer:")
    print(layer_summary.to_csv(sep='\t', index=False))
    if not pseudo_summary.empty:
        print("Pseudo MAF summary:")
        print(pseudo_summary.to_csv(sep='\t', index=False))
    print(f"Wrote: {overall_path}")
    print(f"Wrote: {logic_path}")
    print(f"Wrote: {fam_logic_path}")
    print(f"Wrote: {causal_logic_path}")
    if not pseudo_summary.empty:
        print(f"Wrote: {pseudo_path}")
    print(f"Wrote: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
