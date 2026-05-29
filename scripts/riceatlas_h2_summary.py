#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch GREML for RiceAtlas and merge with existing REML broad-sense H2 summary."
    )
    p.add_argument(
        "--inputs-dir",
        default="test.atlas/reml_atlas/inputs",
        help="Directory containing per-trait *.reml.tsv inputs.",
    )
    p.add_argument(
        "--broad-summary",
        default="test.atlas/reml_atlas/riceatlas.reml.summary.tsv",
        help="Existing combined REML broad-sense heritability summary.",
    )
    p.add_argument(
        "--grm",
        default="test.atlas/Rice6048.cGRM.npy",
        help="GRM matrix used for GREML.",
    )
    p.add_argument(
        "--out-dir",
        default="test.atlas/greml_atlas",
        help="Output directory for GREML runs and merged summaries.",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke janusx.script.greml.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-trait GREML summaries if present.",
    )
    return p


def _read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")


def _run_one_trait(
    python_exe: str,
    input_file: Path,
    grm_file: Path,
    runs_dir: Path,
    trait: str,
    *,
    skip_existing: bool,
) -> tuple[Path, float]:
    prefix = trait
    out_summary = runs_dir / f"{prefix}.greml.summary.tsv"
    if skip_existing and out_summary.is_file():
        return out_summary, 0.0

    cmd = [
        str(python_exe),
        "-m",
        "janusx.script.greml",
        "-file",
        str(input_file),
        "-k",
        str(grm_file),
        "-n",
        str(trait),
        "-o",
        str(runs_dir),
        "-prefix",
        str(prefix),
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return out_summary, float(time.time() - t0)


def main() -> int:
    args = build_parser().parse_args()

    inputs_dir = Path(args.inputs_dir).resolve()
    broad_summary_path = Path(args.broad_summary).resolve()
    grm_file = Path(args.grm).resolve()
    out_dir = Path(args.out_dir).resolve()
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(inputs_dir.glob("*.reml.tsv"))
    if len(input_files) == 0:
        raise FileNotFoundError(f"No *.reml.tsv files found under {inputs_dir}")

    rows: list[dict[str, object]] = []
    for path in input_files:
        df_in = _read_table(path)
        if df_in.shape[1] < 2:
            raise ValueError(f"Input has <2 columns: {path}")
        trait = str(df_in.columns[1])
        summary_path, elapsed = _run_one_trait(
            python_exe=str(args.python),
            input_file=path,
            grm_file=grm_file,
            runs_dir=runs_dir,
            trait=trait,
            skip_existing=bool(args.skip_existing),
        )
        if not summary_path.is_file():
            raise FileNotFoundError(f"Expected GREML summary not found: {summary_path}")
        one = pd.read_csv(summary_path, sep="\t")
        if one.shape[0] != 1:
            raise ValueError(f"Expected one-row GREML summary, got {one.shape[0]} rows: {summary_path}")
        rec = one.iloc[0].to_dict()
        rec["input_file"] = str(path)
        rec["summary_file"] = str(summary_path)
        rec["elapsed_driver_sec"] = float(elapsed)
        rows.append(rec)

    greml_df = pd.DataFrame(rows)
    greml_df = greml_df.sort_values("trait").reset_index(drop=True)
    greml_summary_path = out_dir / "riceatlas.greml.summary.tsv"
    greml_df.to_csv(greml_summary_path, sep="\t", index=False)

    broad_df = pd.read_csv(broad_summary_path, sep="\t")
    broad_cols = {
        "trait": "trait",
        "hsqr": "H2_broad",
        "status": "broad_status",
        "e": "broad_e",
        "h_plot": "broad_h_plot",
        "r": "broad_r",
        "vg": "broad_vg",
        "vge": "broad_vge",
        "ve": "broad_ve",
        "used": "broad_used_obs",
        "total": "broad_total_obs",
        "input_file": "broad_input_file",
        "summary_file": "broad_summary_file",
        "blup_file": "broad_blup_file",
    }
    broad_keep = [c for c in broad_cols if c in broad_df.columns]
    broad_view = broad_df.loc[:, broad_keep].rename(columns={c: broad_cols[c] for c in broad_keep})

    narrow_cols = {
        "trait": "trait",
        "hsqr": "h2_narrow",
        "pve": "h2_pve",
        "lambda": "narrow_lambda",
        "vg": "narrow_vg",
        "ve": "narrow_ve",
        "e": "narrow_e",
        "r": "narrow_r",
        "used_obs": "narrow_used_obs",
        "used_ids": "narrow_used_ids",
        "total_obs": "narrow_total_obs",
        "total_ids": "narrow_total_ids",
        "missing_grm_ids": "missing_grm_ids",
        "stage1_fixed": "stage1_fixed",
        "stage1_random": "stage1_random",
        "stage2_fixed": "stage2_fixed",
        "status": "narrow_status",
        "input_file": "narrow_input_file",
        "summary_file": "narrow_summary_file",
    }
    narrow_keep = [c for c in narrow_cols if c in greml_df.columns]
    narrow_view = greml_df.loc[:, narrow_keep].rename(columns={c: narrow_cols[c] for c in narrow_keep})

    merged = broad_view.merge(narrow_view, on="trait", how="outer", validate="one_to_one")
    merged["H2_minus_h2"] = merged["H2_broad"] - merged["h2_narrow"]
    merged["h2_over_H2"] = np.divide(
        merged["h2_narrow"],
        merged["H2_broad"],
        out=np.full(merged.shape[0], np.nan, dtype=float),
        where=np.isfinite(pd.to_numeric(merged["H2_broad"], errors="coerce"))
        & (pd.to_numeric(merged["H2_broad"], errors="coerce") != 0.0),
    )
    merged = merged.sort_values("trait").reset_index(drop=True)
    merged_summary_path = out_dir / "riceatlas.h2.summary.tsv"
    merged.to_csv(merged_summary_path, sep="\t", index=False)

    finite_broad = int(pd.to_numeric(merged["H2_broad"], errors="coerce").notna().sum())
    finite_narrow = int(pd.to_numeric(merged["h2_narrow"], errors="coerce").notna().sum())
    finite_both = int(
        pd.to_numeric(merged["H2_broad"], errors="coerce").notna()
        .mul(pd.to_numeric(merged["h2_narrow"], errors="coerce").notna())
        .sum()
    )
    stats = pd.DataFrame(
        [
            {"metric": "n_traits_total", "value": int(merged.shape[0])},
            {"metric": "n_traits_broad_finite", "value": finite_broad},
            {"metric": "n_traits_narrow_finite", "value": finite_narrow},
            {"metric": "n_traits_both_finite", "value": finite_both},
            {
                "metric": "mean_H2_broad",
                "value": float(pd.to_numeric(merged["H2_broad"], errors="coerce").mean()),
            },
            {
                "metric": "mean_h2_narrow",
                "value": float(pd.to_numeric(merged["h2_narrow"], errors="coerce").mean()),
            },
            {
                "metric": "mean_H2_minus_h2",
                "value": float(pd.to_numeric(merged["H2_minus_h2"], errors="coerce").mean()),
            },
        ]
    )
    stats_path = out_dir / "riceatlas.h2.stats.tsv"
    stats.to_csv(stats_path, sep="\t", index=False)

    manifest = pd.DataFrame(
        [
            {"file": greml_summary_path.name, "description": "Combined GREML narrow-sense heritability summary across RiceAtlas traits."},
            {"file": merged_summary_path.name, "description": "Merged RiceAtlas broad-sense (REML) and narrow-sense (GREML) heritability summary."},
            {"file": stats_path.name, "description": "Aggregate summary statistics for RiceAtlas H2/h2 tables."},
            {"file": runs_dir.name, "description": "Per-trait GREML outputs (.greml.summary.tsv, .blup.txt, .adjusted_mean.txt, .greml.log)."},
        ]
    )
    manifest_path = out_dir / "riceatlas.h2.manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)

    print(f"Wrote: {greml_summary_path}")
    print(f"Wrote: {merged_summary_path}")
    print(f"Wrote: {stats_path}")
    print(f"Wrote: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
