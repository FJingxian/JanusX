# -*- coding: utf-8 -*-
"""
JanusX: Post-GARFIELD interaction annotation and circular plotting.

Examples
--------
  jx postgarfield -i trait.garfield.fvlmm.tsv -gff genes.gff3.gz
  jx postgarfield -i trait.garfield.fvlmm.tsv -gff genes.gff3.gz -gwasfile trait.gwas.tsv --circle 6 0.4
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import socket
import sys
import time
from functools import lru_cache
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from janusx.assoc.workflow import (
    _format_cli_finished_timestamp,
    _gwas_terminal_config_line_max_chars,
    _terminal_saved_result_paths,
)
from janusx.assoc.workflow_ui import _rich_success
from janusx.script._common.cli_args import (
    add_common_out_arg,
    add_common_prefix_arg,
    add_common_thread_arg,
)
from janusx.script._common.cli_core import (
    CliArgumentParser,
    cli_help_formatter,
    minimal_help_epilog,
)
from janusx.script._common.config_render import emit_cli_configuration
from janusx.script._common.log import setup_logging
from janusx.script._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    format_path_for_display,
)
from janusx.script._common.progress import warn_deprecated_alias_usage
from janusx.script._common.threads import (
    apply_blas_thread_env,
    detect_effective_threads,
    format_requested_thread_usage,
)
from janusx.script.postgwas import (
    GWASPLOT,
    _DEFAULT_CIRCLE_INTERVAL,
    _DEFAULT_CIRCLE_LW,
    _DEFAULT_CIRCLE_SIZE_IN,
    _DEFAULT_CIRCLE_TRACK_RATIO,
    _DEFAULT_SCATTER_SIZE,
    _POSTGWAS_DEFAULT_FONT_SIZE,
    _apply_postgwas_matplotlib_style,
    _chrom_sort_key,
    _format_postgwas_gff_site_desc_direct,
    _format_postgwas_gff_site_desc_many_rust,
    _load_postgwas_input_table,
    _manhattan_colors_for_subset,
    _parse_alpha_spec,
    _parse_marker_spec,
    _parse_palette_spec,
    _parse_scatter_size_spec,
    _parse_ylim_spec,
    _postgwas_build_circle_link_table_from_groups,
    _postgwas_get_gff_query,
    _postgwas_get_gff_rust_index,
    _postgwas_should_rasterize_dense_layers,
    _resolve_postgwas_fontstyle,
    _resolve_postgwas_output_stem,
    _resolve_single_marker,
    _resolve_single_series_value,
    _safe_neglog10_p,
    _save_figure,
    _save_figure_and_close,
    _strip_postgwas_input_suffix,
)

_GARFIELD_GROUP_COL = "snp"
_GARFIELD_CHR_COL = "chrom"
_GARFIELD_POS_COL = "pos"
_GARFIELD_P_COL = "pwald"
_GARFIELD_PADJ_COL = "padj"
_GARFIELD_ROW_ROLE_COL = "row_role"
_GARFIELD_COMBO_ROLE = "combo"
_GARFIELD_SIG_PADJ_DEFAULT = 0.05
_GARFIELD_LABEL_ORDER = {
    "CDS": 0,
    "FivePrimeUTR": 1,
    "ThreePrimeUTR": 2,
    "Exon": 3,
    "Intron": 4,
    "Upstream2kb": 5,
    "Downstream2kb": 6,
    "Intergenic": 7,
}
_GARFIELD_CATEGORY_SPECS = (
    ("all", "all significant interactions"),
    ("cds", "all interaction endpoints include CDS"),
    ("cds_upstream2kb", "all interaction endpoints are CDS/Upstream2kb and include both labels"),
)


def _postgarfield_split_desc_labels(desc: object) -> tuple[str, ...]:
    text = str(desc).strip()
    if text == "" or text == "NA":
        return tuple()
    out: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"\s+\|\s+", text):
        label = str(token).split(";", 1)[0].strip()
        if label == "" or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return tuple(out)


def _postgarfield_join_labels(labels: set[str]) -> str:
    if labels is None or len(labels) == 0:
        return "Intergenic"
    return "|".join(
        sorted(
            [str(x) for x in labels if str(x).strip() != ""],
            key=lambda x: (_GARFIELD_LABEL_ORDER.get(str(x), 999), str(x)),
        )
    )


def _postgarfield_format_decimal_text(value: object) -> str:
    try:
        value_f = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(value_f):
        return "NA"
    return f"{value_f:.4f}"


def _postgarfield_format_sci_text(value: object) -> str:
    try:
        value_f = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(value_f):
        return "NA"
    text = f"{value_f:.4e}"
    return text.replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")


def _postgarfield_format_output_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [x for x in out.columns if str(x).startswith("_")]:
        out = out.drop(columns=[col])
    for col in ("af", "miss", "beta", "se"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(_postgarfield_format_decimal_text)
    for col in ("chisq", "pwald", "padj"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(_postgarfield_format_sci_text)
    return out


def _postgarfield_dedupe_saved_paths(paths: list[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        text = str(path or "").strip()
        if text == "" or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _postgarfield_emit_saved_paths(logger: logging.Logger, paths: list[object]) -> None:
    saved = _terminal_saved_result_paths(_postgarfield_dedupe_saved_paths(paths))
    if len(saved) == 0:
        return
    if len(saved) == 1:
        _rich_success(
            logger,
            f"Results saved to {format_path_for_display(saved[0])}",
            use_spinner=False,
        )
        return
    body = "\n".join(f"  {format_path_for_display(path)}" for path in saved)
    _rich_success(
        logger,
        f"Results saved:\n{body}",
        use_spinner=False,
    )


def _postgarfield_emit_finished_lines(
    logger: logging.Logger,
    *,
    elapsed_sec: float,
    finished_unix: float | None = None,
) -> None:
    _rich_success(
        logger,
        "\n".join(
            [
                "",
                f"Finished. Total wall time: {float(max(elapsed_sec, 0.0)):.2f} seconds",
                _format_cli_finished_timestamp(finished_unix),
            ]
        ),
        use_spinner=False,
    )


def _postgarfield_invocation_command(argv: Optional[list[str]] = None) -> str:
    tokens = [str(x) for x in (sys.argv[1:] if argv is None else argv)]
    prog_raw = str(sys.argv[0]).strip() if len(sys.argv) > 0 else ""
    if prog_raw == "":
        prog_raw = "jx postgarfield"
    try:
        prog_parts = shlex.split(prog_raw)
    except Exception:
        prog_parts = [prog_raw]
    if len(prog_parts) == 0:
        prog_parts = ["jx", "postgarfield"]
    return shlex.join([str(x) for x in (prog_parts + tokens)])


def _emit_postgarfield_command_to_log(
    logger: logging.Logger,
    argv: Optional[list[str]] = None,
) -> None:
    lines = ["", "[ Command ]", f"  {_postgarfield_invocation_command(argv)}"]
    for message in lines:
        record = logger.makeRecord(
            logger.name,
            logging.INFO,
            __file__,
            0,
            message,
            args=(),
            exc_info=None,
        )
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.handle(record)


def _postgarfield_expand_optional_files(
    files: Optional[list[str]],
    n_inputs: int,
    *,
    label: str,
) -> list[Optional[str]]:
    if files is None or len(files) == 0:
        return [None] * int(n_inputs)
    if len(files) == 1:
        return [str(files[0])] * int(n_inputs)
    if len(files) != int(n_inputs):
        raise ValueError(
            f"{label} expects either 1 file or exactly {int(n_inputs)} files to match -i."
        )
    return [str(x) for x in files]


def _postgarfield_parse_circle(args, logger: logging.Logger) -> None:
    if args.circle is None:
        args.circle_size = float(_DEFAULT_CIRCLE_SIZE_IN)
        args.circle_track_ratio = float(_DEFAULT_CIRCLE_TRACK_RATIO)
        return
    circle_items = list(args.circle)
    if len(circle_items) > 2:
        logger.error("circle accepts at most two values: <size_in> <track_ratio>.")
        raise SystemExit(1)
    if len(circle_items) == 0:
        args.circle_size = float(_DEFAULT_CIRCLE_SIZE_IN)
        args.circle_track_ratio = float(_DEFAULT_CIRCLE_TRACK_RATIO)
        return
    try:
        args.circle_size = float(circle_items[0])
    except Exception:
        logger.error("circle size must be a finite number > 0.")
        raise SystemExit(1)
    if (not np.isfinite(args.circle_size)) or float(args.circle_size) <= 0.0:
        logger.error("circle size must be > 0.")
        raise SystemExit(1)
    if len(circle_items) >= 2:
        try:
            args.circle_track_ratio = float(circle_items[1])
        except Exception:
            logger.error("circle track ratio must be a finite number in [0,1].")
            raise SystemExit(1)
    else:
        args.circle_track_ratio = float(_DEFAULT_CIRCLE_TRACK_RATIO)
    if (
        (not np.isfinite(args.circle_track_ratio))
        or float(args.circle_track_ratio) < 0.0
        or float(args.circle_track_ratio) > 1.0
    ):
        logger.error("circle track ratio must be in [0,1].")
        raise SystemExit(1)


def _postgarfield_prepare_plot_args(args, logger: logging.Logger) -> None:
    try:
        args.palette_spec = _parse_palette_spec(args.palette)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)

    try:
        args.scatter_size_spec = _parse_scatter_size_spec(args.scatter_size)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    args._postgwas_single_scatter_size = _resolve_single_series_value(
        args.scatter_size_spec,
        default=float(_DEFAULT_SCATTER_SIZE),
    )

    try:
        args.alpha_spec = _parse_alpha_spec(args.alpha)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    args._postgwas_single_alpha = (
        float(args.alpha_spec[0])
        if args.alpha_spec is not None and len(args.alpha_spec) > 0
        else None
    )

    try:
        args.marker_spec = _parse_marker_spec(args.marker)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    args._postgwas_single_marker = _resolve_single_marker(args.marker_spec)

    if args.fontsize is not None:
        try:
            args.fontsize = float(args.fontsize)
        except Exception:
            logger.error("fontsize must be a finite number > 0.")
            raise SystemExit(1)
        if (not np.isfinite(args.fontsize)) or float(args.fontsize) <= 0.0:
            logger.error("fontsize must be > 0.")
            raise SystemExit(1)
    args._postgwas_base_fontsize = (
        float(args.fontsize)
        if args.fontsize is not None
        else float(_POSTGWAS_DEFAULT_FONT_SIZE)
    )

    if args.fontstyle is not None:
        try:
            font_family, font_display, _match_mode = _resolve_postgwas_fontstyle(args.fontstyle)
        except ValueError as e:
            logger.error(str(e))
            raise SystemExit(1)
        args._postgwas_font_family = font_family
        args._postgwas_font_display = font_display
    else:
        args._postgwas_font_family = ""
        args._postgwas_font_display = "auto"

    if args.ylim is not None:
        try:
            args.ylim_min, args.ylim_max = _parse_ylim_spec(args.ylim)
        except ValueError as e:
            logger.error(str(e))
            raise SystemExit(1)
    else:
        args.ylim_min, args.ylim_max = (None, None)

    if args.circle_interval is None:
        args.circle_interval = float(_DEFAULT_CIRCLE_INTERVAL)
    try:
        args.circle_interval = float(args.circle_interval)
    except Exception:
        logger.error("circle-interval must be a finite number in [0,1].")
        raise SystemExit(1)
    if (
        (not np.isfinite(args.circle_interval))
        or float(args.circle_interval) < 0.0
        or float(args.circle_interval) > 1.0
    ):
        logger.error("circle-interval must be in [0,1].")
        raise SystemExit(1)

    if args.circle_lw is None:
        args.circle_lw = float(_DEFAULT_CIRCLE_LW)
    try:
        args.circle_lw = float(args.circle_lw)
    except Exception:
        logger.error("circle-lw must be a finite number > 0.")
        raise SystemExit(1)
    if (not np.isfinite(args.circle_lw)) or float(args.circle_lw) <= 0.0:
        logger.error("circle-lw must be > 0.")
        raise SystemExit(1)
    args.circle_direction = str(getattr(args, "circle_direction", "out") or "out").strip().lower()
    if args.circle_direction not in {"in", "out"}:
        logger.error("circle direction must be either circle-in or circle-out.")
        raise SystemExit(1)

    args.fullscatter = bool(args.fullscatter)


def _postgarfield_extract_combo_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[0] == 0:
        return df.iloc[0:0].copy()
    if _GARFIELD_ROW_ROLE_COL in df.columns:
        role = df[_GARFIELD_ROW_ROLE_COL].astype(str).str.strip().str.lower()
        combo_mask = role.eq(_GARFIELD_COMBO_ROLE)
        if bool(combo_mask.any()):
            return df.loc[combo_mask].copy()
    if _GARFIELD_GROUP_COL not in df.columns:
        return df.iloc[0:0].copy()
    group_sizes = df.groupby(_GARFIELD_GROUP_COL, sort=False).size()
    multi_groups = set(group_sizes[group_sizes >= 2].index.tolist())
    if len(multi_groups) == 0:
        return df.iloc[0:0].copy()
    group_series = df[_GARFIELD_GROUP_COL].astype(str)
    token_mask = group_series.str.contains(r"[|&*!]", regex=True)
    return df.loc[group_series.isin(multi_groups) & token_mask].copy()


def _postgarfield_build_combo_summary(combo_df: pd.DataFrame) -> pd.DataFrame:
    work = combo_df.copy()
    work[_GARFIELD_GROUP_COL] = work[_GARFIELD_GROUP_COL].astype(str)
    work[_GARFIELD_P_COL] = pd.to_numeric(work[_GARFIELD_P_COL], errors="coerce")
    if _GARFIELD_PADJ_COL in work.columns:
        work[_GARFIELD_PADJ_COL] = pd.to_numeric(work[_GARFIELD_PADJ_COL], errors="coerce")
    else:
        work[_GARFIELD_PADJ_COL] = np.nan
    summary = work.groupby(_GARFIELD_GROUP_COL, sort=False).agg(
        combo_pwald=(_GARFIELD_P_COL, "min"),
        combo_padj=(_GARFIELD_PADJ_COL, "min"),
        combo_n_loci=(_GARFIELD_POS_COL, "size"),
    )
    return summary


def _postgarfield_resolve_sig_groups(
    summary_df: pd.DataFrame,
    *,
    thr: Optional[float],
) -> tuple[set[str], Optional[float], str]:
    if summary_df.shape[0] == 0:
        return set(), None, "no combo rows"
    pwald = pd.to_numeric(summary_df["combo_pwald"], errors="coerce")
    if thr is not None:
        sig_mask = pwald <= float(thr)
        return (
            set(summary_df.index[sig_mask].tolist()),
            float(thr),
            f"combo_pwald<={float(thr):.4g}",
        )

    padj = pd.to_numeric(summary_df["combo_padj"], errors="coerce")
    sig_mask = padj <= float(_GARFIELD_SIG_PADJ_DEFAULT)
    sig_groups = set(summary_df.index[sig_mask].tolist())
    if len(sig_groups) == 0:
        return set(), None, f"combo_padj<={float(_GARFIELD_SIG_PADJ_DEFAULT):.4g} (none)"
    thr_p = float(pd.to_numeric(summary_df.loc[list(sig_groups), "combo_pwald"], errors="coerce").max())
    return sig_groups, thr_p, f"combo_padj<={float(_GARFIELD_SIG_PADJ_DEFAULT):.4g}"


def _postgarfield_annotate_sig_rows(
    sig_rows: pd.DataFrame,
    *,
    gff_query,
    gff_rust_index,
    summary_df: pd.DataFrame,
    sig_rule: str,
    sig_thr_p: Optional[float],
) -> pd.DataFrame:
    out = sig_rows.copy()
    out[_GARFIELD_CHR_COL] = out[_GARFIELD_CHR_COL].astype(str)
    out[_GARFIELD_POS_COL] = pd.to_numeric(out[_GARFIELD_POS_COL], errors="coerce")
    out = out.dropna(subset=[_GARFIELD_POS_COL]).copy()
    out[_GARFIELD_POS_COL] = out[_GARFIELD_POS_COL].astype(int)

    desc_values = ["Intergenic;NA;NA"] * int(out.shape[0])
    if out.shape[0] > 0 and gff_rust_index is not None:
        site_index = pd.MultiIndex.from_arrays(
            [
                out[_GARFIELD_CHR_COL].astype(str).tolist(),
                out[_GARFIELD_POS_COL].astype(int).tolist(),
            ]
        )
        desc_values = _format_postgwas_gff_site_desc_many_rust(
            site_index,
            gff_rust_index=gff_rust_index,
            flank_bp=2_000,
        )
    elif out.shape[0] > 0 and gff_query is not None:
        @lru_cache(maxsize=None)
        def _site_desc(chrom: str, pos: int) -> str:
            return str(
                _format_postgwas_gff_site_desc_direct(
                    chrom=chrom,
                    pos=int(pos),
                    gff_query=gff_query,
                    flank_bp=2_000,
                )
            )

        desc_values = [
            _site_desc(str(chrom), int(pos))
            for chrom, pos in zip(
                out[_GARFIELD_CHR_COL].astype(str).tolist(),
                out[_GARFIELD_POS_COL].astype(int).tolist(),
            )
        ]
    out["desc"] = [str(x) for x in desc_values]

    row_label_sets = [set(_postgarfield_split_desc_labels(x)) for x in out["desc"].tolist()]
    group_endpoint_has_cds: dict[str, list[bool]] = {}
    group_endpoint_has_upstream2kb: dict[str, list[bool]] = {}
    for group_id, labels in zip(out[_GARFIELD_GROUP_COL].astype(str).tolist(), row_label_sets):
        group_key = str(group_id)
        group_endpoint_has_cds.setdefault(group_key, []).append("CDS" in labels)
        group_endpoint_has_upstream2kb.setdefault(group_key, []).append("Upstream2kb" in labels)

    group_all_cds: dict[str, bool] = {}
    group_cds_upstream2kb: dict[str, bool] = {}
    for group_id, cds_flags in group_endpoint_has_cds.items():
        up_flags = group_endpoint_has_upstream2kb.get(group_id, [False] * int(len(cds_flags)))
        endpoint_n = int(len(cds_flags))
        all_cds = endpoint_n >= 2 and all(bool(x) for x in cds_flags)
        all_cds_or_up = endpoint_n >= 2 and all(bool(c) or bool(u) for c, u in zip(cds_flags, up_flags))
        any_cds = any(bool(x) for x in cds_flags)
        any_up = any(bool(x) for x in up_flags)
        group_all_cds[group_id] = bool(all_cds)
        group_cds_upstream2kb[group_id] = bool(all_cds_or_up and any_cds and any_up)

    summary_work = summary_df.copy()
    out["_group_all_cds"] = out[_GARFIELD_GROUP_COL].map(group_all_cds).fillna(False).astype(bool)
    out["_group_cds_upstream2kb"] = (
        out[_GARFIELD_GROUP_COL].map(group_cds_upstream2kb).fillna(False).astype(bool)
    )
    out["_combo_pwald_sort"] = (
        out[_GARFIELD_GROUP_COL].map(summary_work["combo_pwald"].to_dict())
    )
    out["_combo_pwald_sort"] = pd.to_numeric(out["_combo_pwald_sort"], errors="coerce").fillna(np.inf)
    out["_chr_sort_key"] = out[_GARFIELD_CHR_COL].map(_chrom_sort_key)
    out = out.sort_values(
        by=["_combo_pwald_sort", _GARFIELD_GROUP_COL, "_chr_sort_key", _GARFIELD_POS_COL],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).drop(columns=["_combo_pwald_sort", "_chr_sort_key"])
    return out


def _postgarfield_build_category_frames(annotated_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "all": annotated_df.copy(),
        "cds": annotated_df.loc[annotated_df["_group_all_cds"]].copy(),
        "cds_upstream2kb": annotated_df.loc[
            annotated_df["_group_cds_upstream2kb"]
        ].copy(),
    }


def _postgarfield_background_from_garfield(df: pd.DataFrame) -> tuple[pd.DataFrame, list[object]]:
    work = df.copy()
    if _GARFIELD_ROW_ROLE_COL in work.columns:
        role = work[_GARFIELD_ROW_ROLE_COL].astype(str).str.strip().str.lower()
        non_combo = work.loc[~role.eq(_GARFIELD_COMBO_ROLE)].copy()
        if non_combo.shape[0] > 0:
            work = non_combo
    work[_GARFIELD_CHR_COL] = work[_GARFIELD_CHR_COL].astype(str)
    work[_GARFIELD_POS_COL] = pd.to_numeric(work[_GARFIELD_POS_COL], errors="coerce")
    work[_GARFIELD_P_COL] = pd.to_numeric(work[_GARFIELD_P_COL], errors="coerce")
    work = work.dropna(subset=[_GARFIELD_POS_COL, _GARFIELD_P_COL]).copy()
    if work.shape[0] == 0:
        return pd.DataFrame(columns=[_GARFIELD_CHR_COL, _GARFIELD_POS_COL, _GARFIELD_P_COL]), []
    work[_GARFIELD_POS_COL] = work[_GARFIELD_POS_COL].astype(int)
    full_chr_labels = work[_GARFIELD_CHR_COL].drop_duplicates().tolist()
    work = (
        work.groupby([_GARFIELD_CHR_COL, _GARFIELD_POS_COL], as_index=False)[_GARFIELD_P_COL]
        .min()
        .copy()
    )
    return work, full_chr_labels


def _postgarfield_load_background_table(
    garfield_df: pd.DataFrame,
    *,
    gwasfile: Optional[str],
    chr_col: str,
    pos_col: str,
    p_col: str,
) -> tuple[pd.DataFrame, list[object], str]:
    if gwasfile is None:
        bg_df, full_chr_labels = _postgarfield_background_from_garfield(garfield_df)
        return bg_df, full_chr_labels, "GARFIELD singleton/background"
    bg_df, full_chr_labels = _load_postgwas_input_table(
        str(gwasfile),
        chr_col=chr_col,
        pos_col=pos_col,
        p_col=p_col,
        keep_all_columns=False,
    )
    return bg_df, full_chr_labels, os.path.basename(str(gwasfile))


def _postgarfield_build_circle_links_df(
    links_source_df: pd.DataFrame,
    *,
    thr_p: Optional[float],
) -> pd.DataFrame:
    links_df, _meta = _postgwas_build_circle_link_table_from_groups(
        links_source_df,
        group_col=_GARFIELD_GROUP_COL,
        chr_col=_GARFIELD_CHR_COL,
        pos_col=_GARFIELD_POS_COL,
        p_col=_GARFIELD_P_COL,
        type_col=_GARFIELD_GROUP_COL,
        group_tokens=["|", "&", "*"],
    )
    if (
        links_df is not None
        and links_df.shape[0] > 0
        and thr_p is not None
        and np.isfinite(float(thr_p))
        and float(thr_p) > 0.0
        and "link_pvalue" in links_df.columns
    ):
        links_df = links_df.loc[
            pd.to_numeric(links_df["link_pvalue"], errors="coerce") <= float(thr_p)
        ].copy()
    if links_df is None:
        return pd.DataFrame()
    return links_df


def _postgarfield_clear_circle_overlay(ax: plt.Axes) -> None:
    for artist in list(getattr(ax, "_janusx_circle_dynamic_artists", []) or []):
        try:
            artist.remove()
        except Exception:
            pass
    for child_ax in list(getattr(ax, "_janusx_circle_dynamic_axes", []) or []):
        try:
            child_ax.remove()
        except Exception:
            pass
    ax._janusx_circle_dynamic_artists = []
    ax._janusx_circle_dynamic_axes = []


def _postgarfield_create_circle_background(
    *,
    background_df: pd.DataFrame,
    full_chr_labels: list[object],
    chr_col: str,
    pos_col: str,
    p_col: str,
    thr_p: Optional[float],
    args,
) -> tuple[plt.Figure, plt.Axes, GWASPLOT]:
    plotmodel = GWASPLOT(
        background_df,
        chr_col,
        pos_col,
        p_col,
        float(args.interval),
        compression=(not bool(args.fullscatter)),
    )
    plot_colors = _manhattan_colors_for_subset(
        args.palette_spec,
        full_chr_labels,
        background_df[chr_col].drop_duplicates().tolist(),
    )

    circle_threshold = (
        float(_safe_neglog10_p([thr_p])[0])
        if thr_p is not None and np.isfinite(float(thr_p)) and float(thr_p) > 0.0
        else None
    )
    min_logp = float(args.ylim_min) if args.ylim_min is not None else 0.0
    max_logp = float(args.ylim_max) if args.ylim_max is not None else None
    rasterized = _postgwas_should_rasterize_dense_layers(args.format)

    fig_circle, ax_circle = plt.subplots(
        figsize=(float(args.circle_size), float(args.circle_size)),
        dpi=300,
    )
    plotmodel.circle_manhattan(
        threshold=circle_threshold,
        color_set=plot_colors,
        ax=ax_circle,
        links_df=None,
        link_type_col="link_type",
        link_pvalue_col="link_pvalue",
        marker=str(args._postgwas_single_marker),
        scatter_size=float(args._postgwas_single_scatter_size),
        scatter_alpha=(
            float(args._postgwas_single_alpha)
            if args._postgwas_single_alpha is not None
            else 0.76
        ),
        track_ratio=float(args.circle_track_ratio),
        link_interval=float(args.circle_interval),
        link_linewidth=float(args.circle_lw),
        min_logp=min_logp,
        max_logp=max_logp,
        y_min=min_logp,
        circle_direction=str(args.circle_direction),
        rasterized=rasterized,
        draw_links=False,
        show_link_colorbar=False,
    )
    return fig_circle, ax_circle, plotmodel


def _postgarfield_snapshot_circle_background(
    *,
    background_df: pd.DataFrame,
    full_chr_labels: list[object],
    chr_col: str,
    pos_col: str,
    p_col: str,
    thr_p: Optional[float],
    args,
) -> tuple[dict[str, object], GWASPLOT]:
    fig_circle, ax_circle, plotmodel = _postgarfield_create_circle_background(
        background_df=background_df,
        full_chr_labels=full_chr_labels,
        chr_col=chr_col,
        pos_col=pos_col,
        p_col=p_col,
        thr_p=thr_p,
        args=args,
    )
    fig_circle.canvas.draw()
    snapshot = {
        "rgba": np.asarray(fig_circle.canvas.buffer_rgba()).copy(),
        "figsize": tuple(float(x) for x in fig_circle.get_size_inches()),
        "dpi": float(fig_circle.dpi),
        "ax_bounds": tuple(float(x) for x in ax_circle.get_position().bounds),
        "xlim": tuple(float(x) for x in ax_circle.get_xlim()),
        "ylim": tuple(float(x) for x in ax_circle.get_ylim()),
    }
    plt.close(fig_circle)
    return snapshot, plotmodel


def _postgarfield_create_circle_overlay_figure(
    background_snapshot: dict[str, object],
) -> tuple[plt.Figure, plt.Axes]:
    fig_circle = plt.figure(
        figsize=tuple(background_snapshot["figsize"]),
        dpi=float(background_snapshot["dpi"]),
    )
    bg_ax = fig_circle.add_axes([0.0, 0.0, 1.0, 1.0], zorder=0)
    bg_ax.imshow(
        np.asarray(background_snapshot["rgba"]),
        origin="upper",
        aspect="auto",
    )
    bg_ax.axis("off")
    ax_circle = fig_circle.add_axes(list(background_snapshot["ax_bounds"]))
    ax_circle.set_zorder(1)
    ax_circle.set_facecolor("none")
    ax_circle.patch.set_alpha(0.0)
    ax_circle.set_xlim(*tuple(background_snapshot["xlim"]))
    ax_circle.set_ylim(*tuple(background_snapshot["ylim"]))
    ax_circle.set_aspect("equal")
    ax_circle.axis("off")
    return fig_circle, ax_circle


def _postgarfield_write_category_outputs(
    category_frames: dict[str, pd.DataFrame],
    *,
    output_stem: str,
    outdir: str,
    fmt: str,
    background_df: pd.DataFrame,
    full_chr_labels: list[object],
    chr_col: str,
    pos_col: str,
    p_col: str,
    thr_p: Optional[float],
    args,
    logger: logging.Logger,
) -> list[str]:
    saved_paths: list[str] = []
    background_snapshot, plotmodel = _postgarfield_snapshot_circle_background(
        background_df=background_df,
        full_chr_labels=full_chr_labels,
        chr_col=chr_col,
        pos_col=pos_col,
        p_col=p_col,
        thr_p=thr_p,
        args=args,
    )
    plot_colors = _manhattan_colors_for_subset(
        args.palette_spec,
        full_chr_labels,
        background_df[chr_col].drop_duplicates().tolist(),
    )
    circle_threshold = (
        float(_safe_neglog10_p([thr_p])[0])
        if thr_p is not None and np.isfinite(float(thr_p)) and float(thr_p) > 0.0
        else None
    )
    min_logp = float(args.ylim_min) if args.ylim_min is not None else 0.0
    max_logp = float(args.ylim_max) if args.ylim_max is not None else None
    rasterized = _postgwas_should_rasterize_dense_layers(args.format)
    for category, category_desc in _GARFIELD_CATEGORY_SPECS:
        df_cat = category_frames.get(category, pd.DataFrame()).copy()
        out_tsv = os.path.join(outdir, f"{output_stem}.sig.{category}.tsv")
        circle_path = os.path.join(outdir, f"{output_stem}.sig.{category}.circle.{fmt}")
        df_cat_out = _postgarfield_format_output_frame(df_cat)
        df_cat_out.to_csv(out_tsv, sep="\t", index=False)
        saved_paths.append(out_tsv)
        logger.info(
            f"{os.path.basename(out_tsv)}: wrote {int(df_cat_out.shape[0])} row(s) for {category_desc}."
        )
        fig_circle, ax_circle = _postgarfield_create_circle_overlay_figure(
            background_snapshot,
        )
        links_df = _postgarfield_build_circle_links_df(
            df_cat,
            thr_p=thr_p,
        )
        plotmodel.circle_manhattan(
            threshold=circle_threshold,
            color_set=plot_colors,
            ax=ax_circle,
            links_df=links_df,
            link_type_col="link_type",
            link_pvalue_col="link_pvalue",
            marker=str(args._postgwas_single_marker),
            scatter_size=float(args._postgwas_single_scatter_size),
            scatter_alpha=(
                float(args._postgwas_single_alpha)
                if args._postgwas_single_alpha is not None
                else 0.76
            ),
            track_ratio=float(args.circle_track_ratio),
            link_interval=float(args.circle_interval),
            link_linewidth=float(args.circle_lw),
            min_logp=min_logp,
            max_logp=max_logp,
            y_min=min_logp,
            circle_direction=str(args.circle_direction),
            rasterized=rasterized,
            draw_background=False,
            draw_scatter=False,
            draw_links=True,
            show_link_legend=False,
            show_link_colorbar=True,
        )
        _save_figure(fig_circle, circle_path)
        saved_paths.append(circle_path)
        _postgarfield_clear_circle_overlay(ax_circle)
        plt.close(fig_circle)
        logger.info(f"{os.path.basename(circle_path)}: circle plot saved.")
    return saved_paths


def main(argv: Optional[list[str]] = None) -> None:
    warn_deprecated_alias_usage(("-threshold", "--threshold"), replacement="-thr/--thr")
    t_start = time.time()

    parser = CliArgumentParser(
        prog="jx postgarfield",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx postgarfield -i trait.garfield.fvlmm.tsv -gff genes.gff3.gz",
            "jx postgarfield -i trait.garfield.fvlmm.tsv -gff genes.gff3.gz -gwasfile trait.gwas.tsv --circle 6 0.4",
        ]),
    )
    required_group = parser.add_argument_group("Required Arguments")
    required_group.add_argument(
        "-i", "--input", dest="garfield", nargs="+", required=True,
        help="GARFIELD interaction result file(s), typically *.fvlmm.tsv.",
    )
    required_group.add_argument(
        "-gff", "--gff", dest="gff", required=True,
        help="GFF/GFF3 annotation file for interaction endpoint annotation.",
    )

    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-gwasfile", "--gwasfile", nargs="+", default=None,
        help=(
            "Optional GWAS result file(s) used only as circular Manhattan scatter background. "
            "Provide either 1 file for all inputs or one file per -i input."
        ),
    )
    optional_group.add_argument(
        "-thr", "--thr", dest="thr", type=float, default=None,
        help=(
            "Interaction p-value threshold for GARFIELD links. "
            "If omitted, significant interactions are defined by group-level padj<=0.05, "
            "and the circle threshold line uses the largest pwald among those significant groups."
        ),
    )
    optional_group.add_argument(
        "-threshold", "--threshold", dest="thr", type=float, default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-chr", "--chr", dest="chr", type=str, default="chrom",
        help="Background GWAS chromosome column (default: %(default)s).",
    )
    optional_group.add_argument(
        "-pos", "--pos", dest="pos", type=str, default="pos",
        help="Background GWAS position column (default: %(default)s).",
    )
    optional_group.add_argument(
        "-pvalue", "--pvalue", dest="pvalue", type=str, default="pwald",
        help="Background GWAS p-value column (default: %(default)s).",
    )
    optional_group.add_argument(
        "-interval", "--interval", type=float, default=0.5,
        help="Chromosome-gap ratio for circular Manhattan x-axis spacing in [0,1]. Default: %(default)s.",
    )
    optional_group.add_argument(
        "-circle", "--circle", dest="circle", type=float, nargs="*", default=None,
        help=(
            "Enable circular Manhattan plotting. Accepts 0, 1, or 2 values: "
            "--circle, --circle <size_in>, or --circle <size_in> <track_ratio>. "
            "Defaults: size=8.5 in, track_ratio=0.5."
        ),
    )
    optional_group.add_argument(
        "-circle-interval", "--circle-interval", dest="circle_interval", type=float,
        default=_DEFAULT_CIRCLE_INTERVAL,
        help="Relative gap between the scatter ring and interaction links in [0,1]. Default: %(default)s.",
    )
    optional_group.add_argument(
        "-circle-lw", "--circle-lw", dest="circle_lw", type=float,
        default=_DEFAULT_CIRCLE_LW,
        help="Interaction curve line width for circular Manhattan. Default: %(default)s.",
    )
    circle_dir_group = optional_group.add_mutually_exclusive_group(required=False)
    circle_dir_group.add_argument(
        "-circle-in", "--circle-in",
        dest="circle_direction",
        action="store_const",
        const="in",
        help="Draw circular Manhattan values toward the center (0 at the outer ring edge).",
    )
    circle_dir_group.add_argument(
        "-circle-out", "--circle-out",
        dest="circle_direction",
        action="store_const",
        const="out",
        help="Draw circular Manhattan values away from the center (current default).",
    )
    optional_group.add_argument(
        "-palette", "--palette", dest="palette", type=str, default=None,
        help="Circle Manhattan palette, compatible with postgwas --palette.",
    )
    optional_group.add_argument(
        "-scatter-size", "--scatter-size", nargs="+", default=[str(_DEFAULT_SCATTER_SIZE)],
        help="Scatter marker size, compatible with postgwas --scatter-size.",
    )
    optional_group.add_argument(
        "-alpha", "--alpha", nargs="+", default=None,
        help="Scatter alpha in [0,1], compatible with postgwas --alpha.",
    )
    optional_group.add_argument(
        "-marker", "--marker", type=str, default="o",
        help="Scatter marker, compatible with postgwas --marker.",
    )
    optional_group.add_argument(
        "-ylim", "--ylim", nargs="+", default=None,
        help="Shared y-range for circle Manhattan, compatible with postgwas --ylim.",
    )
    optional_group.add_argument(
        "-fontsize", "--fontsize", type=float, default=None,
        help="Unified plot font size, compatible with postgwas --fontsize.",
    )
    optional_group.add_argument(
        "-fontstyle", "--fontstyle", "-fontstype", "--fontstype",
        dest="fontstyle", type=str, default=None,
        help="Unified plot font family or font file path, compatible with postgwas.",
    )
    optional_group.add_argument(
        "-full", "--full", "-fullscatter", "--fullscatter",
        dest="fullscatter", action="store_true", default=False,
        help="Disable scatter compression for the circle background.",
    )
    optional_group.add_argument(
        "-fmt", "--fmt", dest="format", type=str, default="png",
        help="Output figure format: pdf, png, svg, tif (default: %(default)s).",
    )
    add_common_out_arg(optional_group, default=".", help_profile="plot_annotation")
    add_common_prefix_arg(optional_group, default=None, help_profile="inferred_input")
    add_common_thread_arg(optional_group, default_threads=detect_effective_threads())

    args = parser.parse_args(argv)
    args.out = os.path.normpath(args.out if args.out is not None else ".")
    os.makedirs(args.out, mode=0o755, exist_ok=True)

    log_stem = (
        str(args.prefix).strip()
        if args.prefix is not None and str(args.prefix).strip() != ""
        else (_strip_postgwas_input_suffix(args.garfield[0]) if len(args.garfield) == 1 else "postgarfield")
    )
    log_path = os.path.join(args.out, f"{log_stem}.postGARFIELD.log")
    logger = setup_logging(log_path)

    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    if int(args.thread) <= 0:
        args.thread = int(detected_threads)
    if int(args.thread) > int(detected_threads):
        logger.warning(
            f"Warning: Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(detected_threads)}."
        )
        args.thread = int(detected_threads)
    apply_blas_thread_env(int(args.thread))

    if (not np.isfinite(float(args.interval))) or float(args.interval) < 0.0 or float(args.interval) > 1.0:
        logger.error("interval must be in [0,1].")
        raise SystemExit(1)

    _postgarfield_parse_circle(args, logger)
    _postgarfield_prepare_plot_args(args, logger)

    try:
        bg_files = _postgarfield_expand_optional_files(
            args.gwasfile,
            len(args.garfield),
            label="-gwasfile/--gwasfile",
        )
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)

    checks: list[bool] = [
        ensure_file_exists(logger, str(path), "GARFIELD result file")
        for path in list(args.garfield)
    ]
    checks.append(ensure_file_exists(logger, str(args.gff), "GFF annotation file"))
    for bg in bg_files:
        if bg is not None:
            checks.append(ensure_file_exists(logger, str(bg), "GWAS background file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    emit_cli_configuration(
        logger,
        app_title="JanusX - Post-GARFIELD",
        config_title="POST-GARFIELD CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "General",
                [
                    ("GARFIELD files", ", ".join([os.path.basename(str(x)) for x in list(args.garfield)])),
                    ("GFF", str(args.gff)),
                    ("GWAS background", "auto(singleton fallback)" if args.gwasfile is None else ", ".join([os.path.basename(str(x)) for x in list(args.gwasfile)])),
                    ("Background Chr|Pos|P", f"{args.chr}|{args.pos}|{args.pvalue}"),
                    ("Threshold", str(args.thr) if args.thr is not None else "auto from combo padj<=0.05"),
                    ("Circle", f"size={float(args.circle_size):g}, track_ratio={float(args.circle_track_ratio):g}, gap={float(args.circle_interval):g}, lw={float(args.circle_lw):g}, dir={str(args.circle_direction)}"),
                    ("Output format", str(args.format)),
                    (
                        "Threads",
                        format_requested_thread_usage(
                            requested_threads=int(requested_threads),
                            using_threads=int(args.thread),
                            detected_threads=int(detected_threads),
                        ),
                    ),
                ],
            )
        ],
        footer_rows=[("Output dir", args.out)],
        line_max_chars=_gwas_terminal_config_line_max_chars(60),
    )
    _emit_postgarfield_command_to_log(logger, argv)

    _apply_postgwas_matplotlib_style(args)
    gff_rust_index = _postgwas_get_gff_rust_index(
        str(args.gff),
        use_shared=True,
    )
    gff_query = None if gff_rust_index is not None else _postgwas_get_gff_query(
        str(args.gff),
        use_shared=True,
    )
    if gff_rust_index is None and gff_query is None:
        logger.error(f"Failed to build GFF annotation index: {args.gff}")
        raise SystemExit(1)

    saved_paths: list[str] = []
    for idx, garfield_file in enumerate(list(args.garfield), start=1):
        background_file = bg_files[idx - 1]
        output_stem = _resolve_postgwas_output_stem(str(garfield_file), args.prefix)
        logger.info(f"[{idx}/{len(args.garfield)}] Processing {os.path.basename(str(garfield_file))} ...")

        garfield_df = pd.read_csv(str(garfield_file), sep="\t")
        required_cols = {_GARFIELD_GROUP_COL, _GARFIELD_CHR_COL, _GARFIELD_POS_COL, _GARFIELD_P_COL}
        missing_cols = [col for col in required_cols if col not in garfield_df.columns]
        if len(missing_cols) > 0:
            logger.error(
                f"{os.path.basename(str(garfield_file))}: missing required GARFIELD columns: "
                + ", ".join(missing_cols)
            )
            raise SystemExit(1)

        combo_df = _postgarfield_extract_combo_rows(garfield_df)
        summary_df = _postgarfield_build_combo_summary(combo_df)
        sig_groups, sig_thr_p, sig_rule = _postgarfield_resolve_sig_groups(
            summary_df,
            thr=args.thr,
        )
        sig_rows = combo_df.loc[combo_df[_GARFIELD_GROUP_COL].astype(str).isin(sig_groups)].copy()
        annotated_df = _postgarfield_annotate_sig_rows(
            sig_rows,
            gff_query=gff_query,
            gff_rust_index=gff_rust_index,
            summary_df=summary_df,
            sig_rule=sig_rule,
            sig_thr_p=sig_thr_p,
        )
        category_frames = _postgarfield_build_category_frames(annotated_df)

        background_df, full_chr_labels, bg_desc = _postgarfield_load_background_table(
            garfield_df,
            gwasfile=background_file,
            chr_col=str(args.chr),
            pos_col=str(args.pos),
            p_col=str(args.pvalue),
        )
        if background_df.shape[0] == 0:
            logger.error(
                f"{os.path.basename(str(garfield_file))}: background scatter table is empty "
                f"(source={bg_desc})."
            )
            raise SystemExit(1)

        logger.info(
            f"{os.path.basename(str(garfield_file))}: "
            f"combo groups={int(summary_df.shape[0])}, "
            f"significant groups={int(len(sig_groups))}, "
            f"sig_rule={sig_rule}, "
            f"sig_pthr={f'{float(sig_thr_p):.4g}' if sig_thr_p is not None else 'NA'}, "
            f"background={bg_desc}."
        )

        saved_paths.extend(
            _postgarfield_write_category_outputs(
                category_frames,
                output_stem=output_stem,
                outdir=args.out,
            fmt=str(args.format),
            background_df=background_df,
            full_chr_labels=full_chr_labels,
            chr_col=str(args.chr),
            pos_col=str(args.pos),
            p_col=str(args.pvalue),
                thr_p=sig_thr_p,
                args=args,
                logger=logger,
            )
        )

    _postgarfield_emit_saved_paths(logger, saved_paths)
    _postgarfield_emit_finished_lines(
        logger,
        elapsed_sec=(time.time() - t_start),
        finished_unix=time.time(),
    )


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    main()
