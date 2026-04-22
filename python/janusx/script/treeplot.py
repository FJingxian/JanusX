# -*- coding: utf-8 -*-
"""
JanusX: Tree visualization with toytree.

Input modes
-----------
1) Newick file:
   -nwk / --newick  *.nwk

2) GRM matrix:
   -k / --grm       *.npy | *.txt
   -kid / --grm-id  optional sample-id file

Outputs
-------
- <out>/<prefix>.tree.svg
- <out>/<prefix>.tree.html (when --hover is enabled)
- <out>/<prefix>.tree.nwk (when input is GRM and a tree is inferred)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog


def _normalize_prefix(path_or_prefix: str) -> str:
    p = str(path_or_prefix).strip()
    low = p.lower()
    for ext in (".nwk", ".newick", ".svg", ".html", ".pdf", ".png"):
        if low.endswith(ext):
            p = p[: -len(ext)]
            break
    p = p.strip()
    if p == "":
        return "treeplot"
    return p


def _default_prefix_from_path(path: str, fallback: str) -> str:
    name = Path(str(path)).name
    low = name.lower()
    for ext in (".grm.npy", ".grm.txt", ".npy", ".txt", ".tsv", ".csv", ".nwk", ".newick"):
        if low.endswith(ext):
            name = name[: -len(ext)]
            break
    name = re.sub(r"[^\w.\-]+", "_", name).strip("._-")
    return name if name else fallback


def _read_ids_from_file(path: str) -> list[str]:
    out: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fr:
        for raw in fr:
            s = str(raw).strip()
            if s == "":
                continue
            out.append(s.split()[0])
    return out


def _resolve_grm_id_path(grm_path: str, explicit: str | None) -> str | None:
    if explicit is not None and str(explicit).strip() != "":
        p = str(Path(str(explicit).strip()).expanduser())
        if not Path(p).is_file():
            raise FileNotFoundError(f"GRM ID file not found: {p}")
        return p

    direct = f"{grm_path}.id"
    if Path(direct).is_file():
        return direct

    p = Path(grm_path)
    txt_like = {".txt", ".tsv", ".csv", ".npy"}
    if p.suffix.lower() in txt_like:
        cand = f"{str(p)}.id"
        if Path(cand).is_file():
            return cand
        stem_cand = f"{str(p.with_suffix(''))}.id"
        if Path(stem_cand).is_file():
            return stem_cand
    return None


def _load_grm_matrix(path: str) -> np.ndarray:
    low = str(path).lower()
    if low.endswith(".npy"):
        arr = np.asarray(np.load(path), dtype=np.float64)
    else:
        arr = np.asarray(np.genfromtxt(path, dtype=np.float64), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"GRM must be a square matrix, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("GRM matrix contains NaN/Inf values.")
    return arr


def _grm_to_distance(grm: np.ndarray) -> np.ndarray:
    # Kernel-space Euclidean distance: d(i,j)^2 = Kii + Kjj - 2*Kij
    diag = np.asarray(np.diag(grm), dtype=np.float64)
    d2 = diag[:, None] + diag[None, :] - 2.0 * np.asarray(grm, dtype=np.float64)
    np.maximum(d2, 0.0, out=d2)
    dist = np.sqrt(d2, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    return dist


def _to_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s == "":
        return bool(default)
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _load_meta(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        dtype=str,
        keep_default_na=False,
    )
    cols_map = {str(c).strip().lower(): c for c in df.columns}
    if "sample" not in cols_map:
        if "id" in cols_map:
            df = df.rename(columns={cols_map["id"]: "sample"})
        else:
            raise ValueError(
                "Meta file must contain a 'sample' column (or 'id' alias)."
            )
    else:
        df = df.rename(columns={cols_map["sample"]: "sample"})
    for c in ("label", "show_label", "group", "label_color", "node_color", "node_size"):
        if c not in df.columns:
            df[c] = ""
    return df


def _build_color_map(groups: list[str]) -> dict[str, str]:
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    ]
    out: dict[str, str] = {}
    uniq = []
    seen = set()
    for g in groups:
        gg = str(g).strip()
        if gg == "" or gg in seen:
            continue
        seen.add(gg)
        uniq.append(gg)
    for i, g in enumerate(uniq):
        out[g] = palette[i % len(palette)]
    return out


def _resolve_root_query(root_raw: str, tip_labels: list[str]) -> str:
    s = str(root_raw).strip()
    if s == "":
        raise ValueError("--root cannot be empty.")
    try:
        idx = int(s)
    except Exception:
        return s
    n = len(tip_labels)
    if idx < 0 or idx >= n:
        raise ValueError(f"--root index out of range: {idx} (n_tips={n})")
    return str(tip_labels[idx])


def _infer_nj_tree_from_distance(
    dist: np.ndarray,
    sample_ids: list[str],
    nj_backend: str,
) -> tuple[Any, str, str]:
    import toytree

    backend = str(nj_backend).strip().lower()
    if backend not in {"auto", "rust", "toytree"}:
        raise ValueError(f"Unsupported NJ backend: {nj_backend}")

    rust_err: Exception | None = None
    if backend in {"auto", "rust"}:
        try:
            from janusx.janusx import nj_newick_from_distance_matrix as _nj_rust

            d = np.ascontiguousarray(np.asarray(dist, dtype=np.float64))
            nwk = str(
                _nj_rust(
                    d,
                    list(sample_ids),
                    max_taxa=max(20_000, int(d.shape[0]) + 16),
                )
            )
            return toytree.tree(nwk), nwk, "rust"
        except Exception as exc:
            rust_err = exc
            if backend == "rust":
                raise RuntimeError(
                    "Rust NJ backend failed. Rebuild JanusX extension and ensure "
                    "`nj_newick_from_distance_matrix` is available."
                ) from exc

    dist_df = pd.DataFrame(dist, index=sample_ids, columns=sample_ids)
    tree = toytree.infer.neighbor_joining_tree(dist_df)
    return tree, str(tree.write()), ("toytree-fallback" if rust_err is not None else "toytree")


def _load_tree_from_inputs(args) -> tuple[Any, str | None, list[str], str]:
    import toytree

    if args.newick:
        tree = toytree.tree(str(args.newick))
        tips = list(tree.get_tip_labels())
        if len(tips) < 2:
            raise ValueError("Newick tree must contain at least 2 tips.")
        return tree, None, tips, "newick"

    grm_path = str(args.grm)
    if not Path(grm_path).is_file():
        raise FileNotFoundError(f"GRM file not found: {grm_path}")
    grm = _load_grm_matrix(grm_path)
    n = int(grm.shape[0])

    id_path = _resolve_grm_id_path(grm_path, args.grm_id)
    if id_path is not None:
        sample_ids = _read_ids_from_file(id_path)
        if len(sample_ids) != n:
            raise ValueError(
                f"GRM size and ID count mismatch: matrix={n}, ids={len(sample_ids)} ({id_path})"
            )
    else:
        sample_ids = [f"sample_{i+1}" for i in range(n)]

    dist = _grm_to_distance(grm)

    method = str(args.method).strip().lower()
    if method == "nj":
        tree, nwk, backend_used = _infer_nj_tree_from_distance(
            dist=dist,
            sample_ids=sample_ids,
            nj_backend=str(args.nj_backend),
        )
    elif method == "upgma":
        dist_df = pd.DataFrame(dist, index=sample_ids, columns=sample_ids)
        tree = toytree.infer.upgma_tree(dist_df)
        nwk = str(tree.write())
        backend_used = "toytree-upgma"
    else:
        raise ValueError(f"Unsupported --method: {args.method}")

    tips = list(tree.get_tip_labels())
    return tree, nwk, tips, backend_used


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx treeplot",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx treeplot -nwk result.tree.nwk -o out -prefix fig1 --layout c --showlabels -fmt svg",
                "jx treeplot -k panel.grm.npy -kid panel.grm.npy.id -o out -prefix panel_tree --method nj -fmt pdf",
                "jx treeplot -k panel.grm.txt --meta sample_meta.tsv --layout r --root B73 --scale-bar -fmt svg,html",
            ]
        ),
        description="Visualize phylogenetic trees from Newick or GRM using toytree.",
    )

    req = parser.add_argument_group("Required arguments")
    g = req.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "-nwk",
        "--newick",
        type=str,
        help="Input Newick tree file (*.nwk/*.newick).",
    )
    g.add_argument(
        "-k",
        "--grm",
        type=str,
        help="Input GRM matrix file (*.npy|*.txt).",
    )

    opt = parser.add_argument_group("Optional arguments")
    opt.add_argument(
        "-kid",
        "--grm-id",
        type=str,
        default=None,
        help=(
            "Optional GRM sample ID file (default auto-detect: <grm>.id). "
            "Used only when -k/--grm is provided."
        ),
    )
    opt.add_argument(
        "-layout",
        "--layout",
        choices=["r", "l", "u", "d", "c", "w"],
        default="c",
        help=(
            "Tree layout: r/l/u/d directional, c circular, w unrooted. "
            "Default: c (unrooted circular-like view)."
        ),
    )
    opt.add_argument(
        "-root",
        "--root",
        type=str,
        default=None,
        help=(
            "Optional rooting target: tip index (0-based) or sample label. "
            "If omitted, keep tree unrooted."
        ),
    )
    opt.add_argument(
        "-showlabels",
        "--showlabels",
        action="store_true",
        help="Show tip labels (default: off).",
    )
    opt.add_argument(
        "-regexlabels",
        "--regexlabels",
        type=str,
        default=None,
        help="Only show labels matching this regex pattern.",
    )
    opt.add_argument(
        "--shrink",
        type=float,
        default=None,
        help="Add extra spacing for labels (toytree draw option `shrink`).",
    )
    opt.add_argument(
        "--node-size",
        type=float,
        default=4.0,
        help="Default node size (pixels).",
    )
    opt.add_argument(
        "--edge-width",
        type=float,
        default=1.5,
        help="Branch edge width.",
    )
    opt.add_argument(
        "--scale-bar",
        action="store_true",
        help="Show scale bar on tree.",
    )
    opt.add_argument(
        "--hover",
        action="store_true",
        help="Enable node hover tooltip in rendered tree.",
    )
    opt.add_argument(
        "-fmt",
        "--fmt",
        type=str,
        default="svg",
        help=(
            "Output format(s): png/svg/pdf/html. "
            "Use one value (e.g. svg) or comma-separated list (e.g. svg,html). "
            "Default: svg."
        ),
    )
    opt.add_argument(
        "-ratio",
        "--ratio",
        type=float,
        default=1.6,
        help="Canvas width/height ratio (default: 1.6).",
    )
    opt.add_argument(
        "-fontsize",
        "--fontsize",
        type=float,
        default=12.0,
        help="Tip label font size.",
    )
    opt.add_argument(
        "-method",
        "--method",
        type=str,
        choices=["nj", "upgma"],
        default="nj",
        help="Tree inference method from GRM (default: nj).",
    )
    opt.add_argument(
        "--nj-backend",
        type=str,
        choices=["auto", "rust", "toytree"],
        default="auto",
        help=(
            "Backend for method=nj on GRM input: auto (prefer Rust), rust (force), "
            "toytree (Python fallback). Default: auto."
        ),
    )
    opt.add_argument(
        "-meta",
        "--meta",
        type=str,
        default=None,
        help=(
            "Optional sample meta table (*.csv/*.tsv/*.txt), columns:\n"
            "sample,label,show_label,group,label_color,node_color,node_size"
        ),
    )
    opt.add_argument(
        "-o",
        "--out",
        type=str,
        default=".",
        help="Output directory (default: .).",
    )
    opt.add_argument(
        "-prefix",
        "--prefix",
        type=str,
        default=None,
        help="Output prefix (default: inferred from input).",
    )
    opt.add_argument(
        "--height",
        type=float,
        default=700.0,
        help="Canvas height in pixels (default: 700).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if float(args.node_size) < 0:
        parser.error("--node-size must be >= 0.")
    if float(args.edge_width) <= 0:
        parser.error("--edge-width must be > 0.")
    if float(args.ratio) <= 0:
        parser.error("-ratio/--ratio must be > 0.")
    if float(args.fontsize) <= 0:
        parser.error("-fontsize/--fontsize must be > 0.")
    if float(args.height) <= 0:
        parser.error("--height must be > 0.")

    raw_fmt = str(args.fmt).strip().lower()
    if raw_fmt == "":
        parser.error("-fmt/--fmt cannot be empty.")
    fmt_tokens = [x.strip().lower() for x in raw_fmt.split(",") if x.strip() != ""]
    if len(fmt_tokens) == 0:
        parser.error("-fmt/--fmt cannot be empty.")
    allowed_fmts = {"png", "svg", "pdf", "html"}
    out_fmts: list[str] = []
    seen_fmts: set[str] = set()
    for tok in fmt_tokens:
        if tok not in allowed_fmts:
            parser.error(
                f"Unsupported -fmt value: {tok}. Allowed: png, svg, pdf, html."
            )
        if tok not in seen_fmts:
            seen_fmts.add(tok)
            out_fmts.append(tok)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prefix is not None and str(args.prefix).strip() != "":
        prefix = _normalize_prefix(str(args.prefix))
    elif args.newick:
        prefix = _default_prefix_from_path(str(args.newick), "treeplot")
    else:
        prefix = _default_prefix_from_path(str(args.grm), "treeplot")

    nwk_path = out_dir / f"{prefix}.tree.nwk"

    tree, inferred_newick, tip_labels, backend_used = _load_tree_from_inputs(args)

    if args.root is not None:
        root_q = _resolve_root_query(str(args.root), tip_labels)
        try:
            tree = tree.mod.root(root_q)
        except Exception:
            tree = tree.root(root_q)

    tip_labels_now = list(tree.get_tip_labels())
    ntips = len(tip_labels_now)
    nnodes = int(getattr(tree, "nnodes", ntips))
    show_mask = np.full(ntips, bool(args.showlabels), dtype=bool)
    label_list = [str(x) for x in tip_labels_now]
    tip_label_colors: list[str | None] = [None] * ntips

    node_sizes = np.full(nnodes, float(args.node_size), dtype=float)
    node_colors: list[str] = ["#5f7f9f"] * nnodes

    if args.regexlabels is not None and str(args.regexlabels).strip() != "":
        pat = re.compile(str(args.regexlabels))
        regex_mask = np.asarray([bool(pat.search(x)) for x in tip_labels_now], dtype=bool)
        show_mask = regex_mask if not bool(args.showlabels) else (show_mask & regex_mask)

    if args.meta is not None and str(args.meta).strip() != "":
        meta_path = str(Path(str(args.meta).strip()).expanduser())
        if not Path(meta_path).is_file():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        mdf = _load_meta(meta_path)

        tip_pos = {name: i for i, name in enumerate(tip_labels_now)}
        has_group = "group" in mdf.columns
        gmap: dict[str, str] = {}
        if has_group:
            gmap = _build_color_map(mdf["group"].astype(str).tolist())

        for _, row in mdf.iterrows():
            sample = str(row.get("sample", "")).strip()
            if sample == "":
                continue
            if sample not in tip_pos:
                continue
            i = int(tip_pos[sample])

            label = str(row.get("label", "")).strip()
            if label != "":
                label_list[i] = label

            sl = row.get("show_label", "")
            if str(sl).strip() != "":
                show_mask[i] = _to_bool(sl, default=bool(show_mask[i]))

            grp = str(row.get("group", "")).strip()
            default_group_color = gmap.get(grp, None) if grp != "" else None

            lcol = str(row.get("label_color", "")).strip() or default_group_color
            if lcol:
                tip_label_colors[i] = str(lcol)

            ncol = str(row.get("node_color", "")).strip() or default_group_color
            if ncol:
                node_colors[i] = str(ncol)

            nsize = str(row.get("node_size", "")).strip()
            if nsize != "":
                try:
                    val = float(nsize)
                except Exception:
                    val = float(args.node_size)
                if val < 0:
                    val = float(args.node_size)
                node_sizes[i] = float(val)

    if bool(np.any(show_mask)):
        tip_labels_arg: bool | list[str] = [
            label_list[i] if bool(show_mask[i]) else "" for i in range(ntips)
        ]
    else:
        tip_labels_arg = False

    draw_layout = {"w": "unr"}.get(str(args.layout), str(args.layout))
    draw_kwargs: dict[str, Any] = {
        "layout": draw_layout,
        "edge_type": "c",
        "tip_labels": tip_labels_arg,
        "node_sizes": node_sizes,
        "node_colors": node_colors,
        "edge_style": {"stroke-width": float(args.edge_width)},
        "tip_labels_style": {"font-size": f"{float(args.fontsize):.2f}px"},
        "scale_bar": bool(args.scale_bar),
        "node_hover": bool(args.hover),
        "height": float(args.height),
        "width": float(args.height) * float(args.ratio),
    }
    if args.shrink is not None:
        draw_kwargs["shrink"] = float(args.shrink)

    if any(c is not None and str(c).strip() != "" for c in tip_label_colors):
        filled = [str(c) if c is not None and str(c).strip() != "" else "#262626" for c in tip_label_colors]
        draw_kwargs["tip_labels_colors"] = filled

    canvas, _axes, _mark = tree.draw(**draw_kwargs)

    import toytree

    out_paths: list[Path] = []
    for fmt in out_fmts:
        p = out_dir / f"{prefix}.tree.{fmt}"
        try:
            toytree.save(canvas, str(p))
        except Exception as exc:
            msg = str(exc)
            if fmt == "png" and "ghostscript" in msg.lower():
                raise RuntimeError(
                    "Failed to export PNG because ghostscript is missing in PATH. "
                    "Please install ghostscript (e.g. `brew install ghostscript`) "
                    "or use -fmt svg/pdf/html."
                ) from exc
            raise RuntimeError(
                f"Failed to export {fmt} file: {p}\nOriginal error: {exc}"
            ) from exc
        out_paths.append(p)

    if inferred_newick is not None:
        with open(nwk_path, "w", encoding="utf-8") as fw:
            fw.write(str(inferred_newick).strip())
            fw.write("\n")
        out_paths.append(nwk_path)

    print("TreePlot finished.")
    if args.grm is not None:
        print(f"NJ backend: {backend_used}")
    print("Output files:")
    for p in out_paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
