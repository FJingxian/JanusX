#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from janusx.bioplotkit.sci_set import sci_set
from janusx.janusx import bed_ldblock_r2_rust


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT_ROOT = REPO_ROOT / "test.Tgarfield_out"
DEFAULT_FIGURE_DIR = REPO_ROOT / "test.figure4"

CHUNK_LABELS = {"L1": "front", "L2": "middle", "L3": "back"}
CATEGORY_LABELS = {
    "exact_rule": "Exact rule",
    "high_ld_proxy": "High-LD proxy",
    "overlap_gt1_snp": ">1 SNP overlap",
    "overlap_1_snp": "1 SNP overlap",
    "other": "Other",
}
CATEGORY_COLORS = {
    "exact_rule": "#4CAF50",
    "high_ld_proxy": "#F2C14E",
    "overlap_gt1_snp": "#E07A5F",
    "overlap_1_snp": "#7AA6C2",
    "other": "#9E9E9E",
}


@dataclass(frozen=True)
class ChunkInfo:
    chunk_id: str
    chrom: str
    start_bp: int
    end_bp: int
    chunk_prefix: str


def _parse_expr(expr: object) -> tuple[str, tuple[str, ...]]:
    raw = str(expr or "").strip()
    if raw == "":
        return ("", tuple())
    op = "SINGLE"
    if " AND " in raw:
        op = "AND"
    elif " OR " in raw:
        op = "OR"
    sites = tuple(sorted(str(site).strip() for site in re.findall(r"BIN\(([^)]+)\)", raw)))
    return (op, sites)


def _parse_chunk_position(chunk_id: object) -> tuple[str, str]:
    raw = str(chunk_id or "").strip()
    m = re.search(r"\.chr([^.]+)\.(L[123])\.", raw)
    if m is None:
        return ("", "")
    return (f"chr{m.group(1)}", CHUNK_LABELS.get(str(m.group(2)), ""))


def _site_token_to_pair(site: str) -> tuple[str, int]:
    txt = str(site).strip()
    if "_" not in txt:
        raise ValueError(f"Unexpected site token: {site}")
    chrom, pos = txt.split("_", 1)
    return (str(chrom), int(pos))


def _site_pair_to_token(chrom: str, pos: int) -> str:
    return f"{chrom}_{int(pos)}"


def _load_chunk_map(summary_dir: Path) -> dict[str, ChunkInfo]:
    path = summary_dir / "chunks.tsv"
    df = pd.read_csv(path, sep="\t")
    out: dict[str, ChunkInfo] = {}
    for row in df[["chunk_id", "chrom", "start_bp", "end_bp", "chunk_prefix"]].itertuples(index=False):
        out[str(row.chunk_id)] = ChunkInfo(
            chunk_id=str(row.chunk_id),
            chrom=str(row.chrom),
            start_bp=int(row.start_bp),
            end_bp=int(row.end_bp),
            chunk_prefix=str(row.chunk_prefix),
        )
    return out


def _load_chunk_results(summary_dir: Path) -> pd.DataFrame:
    path = summary_dir / "results.tsv"
    df = pd.read_csv(path, sep="\t")
    df = df.loc[df["family"].astype(str) == "chunk_random"].copy()
    df["causal_size"] = pd.to_numeric(df["causal_size"], errors="coerce").fillna(0).astype(int)
    df = df.loc[df["causal_size"].isin([1, 2, 3])].copy()
    df["top_rule_match"] = df["top_rule_match"].astype(str).map({"True": True, "False": False}).fillna(False)
    return df.reset_index(drop=True)


def _category_order_for_size(causal_size: int) -> list[str]:
    if int(causal_size) == 1:
        return ["exact_rule", "high_ld_proxy", "other"]
    return ["exact_rule", "overlap_gt1_snp", "overlap_1_snp", "other"]


def _build_chunk_site_unions(df: pd.DataFrame) -> dict[str, set[tuple[str, int]]]:
    chunk_sites: dict[str, set[tuple[str, int]]] = {}
    for row in df[["chunk_id", "simbench_best_expr", "best_window_expr"]].itertuples(index=False):
        chunk_id = str(row.chunk_id)
        site_set = chunk_sites.setdefault(chunk_id, set())
        for expr in (row.simbench_best_expr, row.best_window_expr):
            _op, sites = _parse_expr(expr)
            for site in sites:
                site_set.add(_site_token_to_pair(site))
    return chunk_sites


def _compute_chunk_ld_map(
    *,
    chunk_info: ChunkInfo,
    selected_sites: set[tuple[str, int]],
    threads: int,
) -> dict[tuple[str, str], float]:
    if not selected_sites:
        return {}
    ordered = sorted(selected_sites, key=lambda x: (str(x[0]), int(x[1])))
    selected_chrom = [str(chrom) for chrom, _ in ordered]
    selected_pos = [int(pos) for _, pos in ordered]
    r2, out_chrom, out_pos = bed_ldblock_r2_rust(
        chunk_info.chunk_prefix,
        [str(chunk_info.chrom)],
        [int(chunk_info.start_bp)],
        [int(chunk_info.end_bp)],
        selected_chrom,
        selected_pos,
        threads=max(1, int(threads)),
    )
    mat = np.asarray(r2, dtype=np.float64)
    tokens = [_site_pair_to_token(str(ch), int(pos)) for ch, pos in zip(out_chrom, out_pos)]
    lookup: dict[tuple[str, str], float] = {}
    for i, left in enumerate(tokens):
        for j, right in enumerate(tokens):
            lookup[(left, right)] = float(mat[i, j])
    return lookup


def _best_ld_assignment(
    sim_sites: tuple[str, ...],
    pred_sites: tuple[str, ...],
    ld_lookup: dict[tuple[str, str], float],
) -> tuple[float, float, tuple[tuple[str, str, float], ...]]:
    if len(sim_sites) == 0 or len(pred_sites) == 0 or len(sim_sites) != len(pred_sites):
        return (float("nan"), float("nan"), tuple())
    n = len(sim_sites)
    best_key: tuple[float, float] | None = None
    best_pairs: tuple[tuple[str, str, float], ...] = tuple()
    for perm in itertools.permutations(range(n)):
        pairs: list[tuple[str, str, float]] = []
        vals: list[float] = []
        ok = True
        for i, pred_idx in enumerate(perm):
            sim_site = sim_sites[i]
            pred_site = pred_sites[pred_idx]
            r2 = float(ld_lookup.get((sim_site, pred_site), float("nan")))
            if not np.isfinite(r2):
                ok = False
                break
            pairs.append((sim_site, pred_site, r2))
            vals.append(r2)
        if not ok:
            continue
        key = (min(vals), float(np.mean(vals)))
        if best_key is None or key > best_key:
            best_key = key
            best_pairs = tuple(pairs)
    if best_key is None:
        return (float("nan"), float("nan"), tuple())
    return (float(best_key[0]), float(best_key[1]), best_pairs)


def _classify_chunk_hits(
    *,
    df: pd.DataFrame,
    chunk_map: dict[str, ChunkInfo],
    ld_threshold: float,
    ld_threads: int,
) -> pd.DataFrame:
    df_size1 = df.loc[df["causal_size"] == 1].copy()
    chunk_sites = _build_chunk_site_unions(df_size1)
    chunk_ld_lookup: dict[str, dict[tuple[str, str], float]] = {}
    for chunk_id, selected_sites in chunk_sites.items():
        info = chunk_map.get(chunk_id)
        if info is None:
            raise KeyError(f"Missing chunk info for {chunk_id}")
        chunk_ld_lookup[chunk_id] = _compute_chunk_ld_map(
            chunk_info=info,
            selected_sites=selected_sites,
            threads=ld_threads,
        )

    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        sim_op, sim_sites = _parse_expr(row.simbench_best_expr)
        pred_op, pred_sites = _parse_expr(row.best_window_expr)
        same_op = bool(sim_op != "" and pred_op != "" and sim_op == pred_op)
        same_site_count = len(sim_sites) == len(pred_sites)
        overlap_sites = tuple(sorted(set(sim_sites) & set(pred_sites)))
        overlap_site_count = len(overlap_sites)
        if int(row.causal_size) == 1:
            min_r2, mean_r2, matched_pairs = _best_ld_assignment(
                sim_sites,
                pred_sites,
                chunk_ld_lookup.get(str(row.chunk_id), {}),
            )
        else:
            min_r2, mean_r2, matched_pairs = (float("nan"), float("nan"), tuple())
        high_ld_proxy = bool(
            (not bool(row.top_rule_match))
            and same_op
            and same_site_count
            and len(sim_sites) > 0
            and np.isfinite(min_r2)
            and float(min_r2) >= float(ld_threshold)
        )
        if bool(row.top_rule_match):
            category = "exact_rule"
        elif int(row.causal_size) == 1 and high_ld_proxy:
            category = "high_ld_proxy"
        elif int(row.causal_size) in (2, 3) and overlap_site_count > 1:
            category = "overlap_gt1_snp"
        elif int(row.causal_size) in (2, 3) and overlap_site_count == 1:
            category = "overlap_1_snp"
        else:
            category = "other"
        chrom, chunk_position = _parse_chunk_position(row.chunk_id)
        rows.append(
            {
                "experiment_id": str(row.experiment_id),
                "chunk_id": str(row.chunk_id),
                "chrom": chrom,
                "chunk_position": chunk_position,
                "replicate": int(row.replicate),
                "causal_size": int(row.causal_size),
                "logic_mode": str(row.logic_mode or ""),
                "sim_expr": str(row.simbench_best_expr),
                "pred_expr": str(row.best_window_expr),
                "sim_sites": ";".join(sim_sites),
                "pred_sites": ";".join(pred_sites),
                "sim_site_count": int(len(sim_sites)),
                "pred_site_count": int(len(pred_sites)),
                "overlap_site_count": int(overlap_site_count),
                "overlap_sites": ";".join(overlap_sites),
                "exact_rule_match": bool(row.top_rule_match),
                "same_logic_gate": same_op,
                "same_site_count": bool(same_site_count),
                "best_match_min_r2": min_r2,
                "best_match_mean_r2": mean_r2,
                "ld_proxy_match": high_ld_proxy,
                "ld_threshold": float(ld_threshold),
                "matched_ld_pairs": ";".join(
                    f"{sim_site}~{pred_site}:{r2:.4f}" for sim_site, pred_site, r2 in matched_pairs
                ),
                "category": category,
            }
        )
    return pd.DataFrame(rows)


def _write_outputs(out_dir: Path, classified: pd.DataFrame) -> tuple[list[Path], Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_paths: list[Path] = []
    for causal_size in (1, 2, 3):
        sub = classified.loc[classified["causal_size"] == causal_size].copy()
        sub = sub.sort_values(["chrom", "chunk_position", "replicate", "logic_mode", "experiment_id"], kind="stable")
        path = out_dir / f"chunk_size{causal_size}.ld_proxy.tsv"
        sub.to_csv(path, sep="\t", index=False)
        detail_paths.append(path)
    summary = (
        classified.groupby(["causal_size", "category"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = classified.groupby("causal_size").size().rename("total").reset_index()
    summary = summary.merge(totals, on="causal_size", how="left")
    summary["prop"] = summary["n"] / summary["total"]
    summary["category_label"] = summary["category"].map(CATEGORY_LABELS)
    summary["category_rank"] = summary.apply(
        lambda row: _category_order_for_size(int(row["causal_size"])).index(str(row["category"]))
        if str(row["category"]) in _category_order_for_size(int(row["causal_size"]))
        else 999,
        axis=1,
    )
    summary = summary.sort_values(["causal_size", "category_rank"], kind="stable").reset_index(drop=True)
    summary = summary.drop(columns=["category_rank"])
    summary_path = out_dir / "chunk_ld_proxy.summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    return detail_paths, summary_path


def _plot_pies(summary: pd.DataFrame, out_path: Path, ld_threshold: float) -> None:
    try:
        sci_set()
    except Exception as exc:
        print(f"[warn] sci_set() failed, fallback to default matplotlib style: {exc}")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1))
    for ax, causal_size in zip(axes, (1, 2, 3)):
        sub = summary.loc[summary["causal_size"] == causal_size].copy()
        category_order = _category_order_for_size(causal_size)
        counts = [int(sub.loc[sub["category"] == cat, "n"].sum()) for cat in category_order]
        total = max(1, int(sum(counts)))
        labels = [
            f"{CATEGORY_LABELS[cat]}\n{counts[i]}/{total} ({100.0 * counts[i] / total:.1f}%)"
            for i, cat in enumerate(category_order)
        ]
        colors = [CATEGORY_COLORS[cat] for cat in category_order]
        if sum(counts) == 0:
            ax.axis("off")
            continue
        ax.pie(
            counts,
            labels=labels,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
            textprops={"fontsize": 10},
        )
        ax.set_title(f"{causal_size}-site\nn={total}", fontsize=12, pad=12)
    fig.suptitle(
        f"Chunk-random hit categories (1-site: LD proxy r² >= {ld_threshold:.2f}; 2/3-site: SNP overlap)",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Classify chunk-random GARFIELD hits: 1-site uses exact-rule / high-LD proxy / other; "
            "2/3-site use exact-rule / >1 SNP overlap / 1 SNP overlap / other."
        )
    )
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Tgarfield output root.")
    p.add_argument(
        "--summary-subdir",
        default="chunk_ld_proxy",
        help="Subdirectory under <out-root>/summary to store detail/summary TSVs.",
    )
    p.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR), help="Output directory for pie-chart PDF.")
    p.add_argument("--ld-threshold", type=float, default=0.8, help="High-LD proxy threshold on r².")
    p.add_argument("--ld-threads", type=int, default=1, help="Threads per chunk LD call.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_root = Path(args.out_root).resolve()
    summary_dir = out_root / "summary"
    figure_dir = Path(args.figure_dir).resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_dir = summary_dir / str(args.summary_subdir)
    ld_threshold = float(args.ld_threshold)

    chunk_map = _load_chunk_map(summary_dir)
    chunk_results = _load_chunk_results(summary_dir)
    classified = _classify_chunk_hits(
        df=chunk_results,
        chunk_map=chunk_map,
        ld_threshold=ld_threshold,
        ld_threads=max(1, int(args.ld_threads)),
    )
    detail_paths, summary_path = _write_outputs(out_dir, classified)
    fig_path = figure_dir / f"{out_root.name}.chunk_ld_proxy_pies.pdf"
    summary = pd.read_csv(summary_path, sep="\t")
    _plot_pies(summary, fig_path, ld_threshold)

    print(f"Out root   : {out_root}")
    print(f"Summary dir: {out_dir}")
    print(f"Figure dir : {figure_dir}")
    print(f"Rows       : {len(classified)}")
    print(f"LD thr     : {ld_threshold:.4g}")
    print(summary.to_csv(sep='\t', index=False))
    for path in detail_paths:
        print(f"Wrote: {path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
