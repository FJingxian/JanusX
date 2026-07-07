# -*- coding: utf-8 -*-
"""
JanusX: Post-GS summary and visualization

Examples
--------
  jx postgs -json summary.json
  jx postgs -json summary.json -effect test/mouse_hs1940.gs.model/test0.rrBLUP.jxmodel -effect-col beta
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from janusx.bioplotkit import gsplot
from janusx.script._common.cli_core import (
    CliArgumentParser,
    cli_help_formatter,
    minimal_help_epilog,
)
from janusx.script._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
)
from janusx.script._common.log import setup_logging


class _SilentLogger:
    def error(self, msg: str) -> None:
        _ = msg


def _enable_windows_ansi_stream(stream: Any) -> bool:
    if os.name != "nt":
        return True
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle in (0, -1):
            return False
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return False
        enable_vt = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if mode.value & enable_vt:
            return True
        return kernel32.SetConsoleMode(handle, mode.value | enable_vt) != 0
    except Exception:
        return False


def _supports_live_ansi(stream: Any) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.name == "nt":
        return _enable_windows_ansi_stream(stream)
    return True


class _PostGsPhasePrinter:
    def __init__(self, ui: "_PostGsCliUI", title: str) -> None:
        self.ui = ui
        self.title = str(title)
        self.child_lines = 0
        self.ui.plain(self.title)

    def ok(self, label: str, sec: float) -> None:
        line = f"  ✔︎ {str(label):<34} [{float(sec):.1f}s]"
        self.ui.green(line)
        self.child_lines += 1

    def note(self, text: str) -> None:
        self.ui.plain(str(text))
        self.child_lines += 1

    def complete(self) -> None:
        self.ui.complete_parent(self.title, self.child_lines)


class _PostGsCliUI:
    _GREEN = "\033[32m"
    _RESET = "\033[0m"
    _CLEAR_LINE = "\033[2K"

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.live = bool(_supports_live_ansi(self.stream))
        self.color = self.live

    def _write(self, text: str) -> None:
        self.stream.write(str(text))
        self.stream.flush()

    def _greenize(self, text: str) -> str:
        if not self.color:
            return str(text)
        return f"{self._GREEN}{text}{self._RESET}"

    def plain(self, text: str = "") -> None:
        self._write(f"{text}\n")

    def green(self, text: str) -> None:
        self._write(f"{self._greenize(text)}\n")

    def start_phase(self, title: str) -> _PostGsPhasePrinter:
        return _PostGsPhasePrinter(self, title)

    def complete_parent(self, title: str, child_lines: int) -> None:
        if not self.live:
            return
        n = max(int(child_lines), 0)
        total_up = n + 1
        # Move to parent line, repaint parent in green, then return to current location.
        self._write(f"\033[{total_up}F")
        self._write(self._CLEAR_LINE)
        self._write(self._greenize(title))
        self._write("\n")
        if n > 0:
            self._write(f"\033[{n}E")


class _PostGsNoopPhase:
    def ok(self, label: str, sec: float) -> None:
        _ = (label, sec)

    def note(self, text: str) -> None:
        _ = text

    def complete(self) -> None:
        return


class _PostGsNoopUI:
    def plain(self, text: str = "") -> None:
        _ = text

    def green(self, text: str) -> None:
        _ = text

    def start_phase(self, title: str) -> _PostGsNoopPhase:
        _ = title
        return _PostGsNoopPhase()

    def complete_parent(self, title: str, child_lines: int) -> None:
        _ = (title, child_lines)
        return


def _sanitize_token(text: str) -> str:
    s = re.sub(r"\s+", "_", str(text).strip())
    s = re.sub(r"[^0-9A-Za-z._+-]+", "_", s)
    return s.strip("_") or "item"


def _norm_method_token(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(text).strip().lower())


def _parse_formats(text: str) -> list[str]:
    raw = str(text).replace(";", ",")
    items = [x.strip().lower() for x in raw.split(",") if x.strip() != ""]
    if len(items) == 0:
        return ["png"]
    return items


def _parse_ratio_palette_spec(
    raw: Optional[Any],
    *,
    default_ratio: float,
) -> tuple[float, Optional[str]]:
    ratio = float(default_ratio)
    palette: Optional[str] = None
    if raw is None:
        return ratio, palette

    def _try_positive_float(text: str) -> Optional[float]:
        try:
            rv = float(str(text).strip())
            if np.isfinite(rv) and rv > 0.0:
                return float(rv)
        except Exception:
            return None
        return None

    if isinstance(raw, (list, tuple)):
        items = [str(x).strip() for x in raw if str(x).strip() != ""]
        if len(items) == 0:
            return ratio, palette
        if len(items) >= 2:
            rv = _try_positive_float(items[0])
            if rv is not None:
                ratio = rv
                palette = ",".join(items[1:]) if len(items) > 1 else None
                return ratio, palette
            palette = ",".join(items)
            return ratio, palette
        # single-item list fallback: allow legacy "1,tab10" or just "tab10"/"1"
        raw = items[0]

    s = str(raw).strip()
    if s == "":
        return ratio, palette
    toks = [t.strip() for t in re.split(r"[;,]", s) if str(t).strip() != ""]
    if len(toks) == 0:
        return ratio, palette
    if len(toks) >= 2:
        rv = _try_positive_float(toks[0])
        if rv is not None:
            ratio = rv
            palette = ",".join(toks[1:])
        else:
            palette = ",".join(toks)
        return ratio, palette

    rv = _try_positive_float(toks[0])
    if rv is not None:
        ratio = rv
    else:
        palette = toks[0]
    return ratio, palette


def _figsize_from_ratio(
    ratio: float,
    *,
    base_height: float,
    min_ratio: float = 0.25,
    max_ratio: float = 6.0,
) -> tuple[float, float]:
    r = float(ratio)
    if not np.isfinite(r) or r <= 0.0:
        r = 1.0
    r = float(min(max(r, min_ratio), max_ratio))
    h = max(float(base_height), 2.0)
    w = h * r
    return (w, h)


def _save_fig(fig: plt.Figure, out_path: str, fmt: str) -> None:
    fmt_l = str(fmt).strip().lower()
    rc = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    with mpl.rc_context(rc):
        fig.savefig(
            out_path,
            format=fmt_l,
            dpi=(300 if fmt_l not in {"pdf", "svg"} else None),
            transparent=False,
            facecolor="white",
            bbox_inches="tight",
        )


def _max_abs_scale(values: pd.Series) -> tuple[np.ndarray, float]:
    arr = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr, float("nan")
    max_abs = float(np.nanmax(np.abs(arr[finite])))
    if (not np.isfinite(max_abs)) or (max_abs <= 0.0):
        out = arr.copy()
        out[finite] = 0.0
        return out, max_abs
    out = arr.copy()
    out[finite] = out[finite] / max_abs
    return out, max_abs


def _resolve_existing_path(path_like: str, *, base_dir: str) -> Optional[str]:
    if path_like is None:
        return None
    p = str(path_like).strip()
    if p == "":
        return None
    p0 = Path(p)
    if p0.is_file():
        return str(p0)
    p1 = Path(base_dir) / p
    if p1.is_file():
        return str(p1)
    return None


def _infer_prefix(summary: dict[str, Any], summary_path: str) -> str:
    pref = str(summary.get("prefix", "")).strip()
    if pref != "":
        return _sanitize_token(pref)
    sp = Path(summary_path)
    if sp.name.lower() == "summary.json":
        parent = sp.parent.name.strip()
        if parent != "":
            return _sanitize_token(parent)
    return _sanitize_token(sp.stem)


def _summary_rows_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows = summary.get("summary_rows", [])
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("summary.json has no `summary_rows`.")
    df = pd.DataFrame(rows)
    if "trait" not in df.columns:
        df["trait"] = "trait0"
    if "model" not in df.columns:
        raise ValueError("summary_rows missing `model` column.")

    num_cols = [
        "pearsonr_cv_mean",
        "spearmanr_cv_mean",
        "r2_cv_mean",
        "mse_cv_mean",
        "mae_cv_mean",
        "pve_final",
        "time_cv_mean_sec",
    ]
    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trait"] = df["trait"].astype(str)
    df["model"] = df["model"].astype(str)
    return df


def _cv_accuracy_long_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tmd = summary.get("trait_method_details", {})
    if not isinstance(tmd, dict):
        return pd.DataFrame(columns=["Trait", "Model", "Metric", "Accuracy", "Fold"])

    for trait, trait_rec in tmd.items():
        if not isinstance(trait_rec, dict):
            continue
        for mk, mv in trait_rec.items():
            if not isinstance(mv, dict):
                continue
            model_display = str(mv.get("method_display", mk))
            folds = mv.get("fold_rows", [])
            if not isinstance(folds, list):
                continue
            for fr in folds:
                if not isinstance(fr, dict):
                    continue
                fold_id = fr.get("fold", None)
                p = pd.to_numeric(fr.get("pearson", np.nan), errors="coerce")
                s = pd.to_numeric(fr.get("spearman", np.nan), errors="coerce")
                if np.isfinite(float(p)):
                    rows.append(
                        {
                            "Trait": str(trait),
                            "Model": model_display,
                            "Metric": "Pearson r",
                            "Accuracy": float(p),
                            "Fold": fold_id,
                        }
                    )
                if np.isfinite(float(s)):
                    rows.append(
                        {
                            "Trait": str(trait),
                            "Model": model_display,
                            "Metric": "Spearman r",
                            "Accuracy": float(s),
                            "Fold": fold_id,
                        }
                    )

    if len(rows) == 0:
        return pd.DataFrame(columns=["Trait", "Model", "Metric", "Accuracy", "Fold"])
    return pd.DataFrame(rows)


def _model_cv_error_frame(
    summary: dict[str, Any],
    *,
    trait: Optional[str] = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tmd = summary.get("trait_method_details", {})
    if not isinstance(tmd, dict):
        return pd.DataFrame(columns=["model", "pearsonr_cv_sd", "time_cv_sd"])

    trait_filter = str(trait) if trait is not None else None
    for _trait, trait_rec in tmd.items():
        if trait_filter is not None and str(_trait) != trait_filter:
            continue
        if not isinstance(trait_rec, dict):
            continue
        for mk, mv in trait_rec.items():
            if not isinstance(mv, dict):
                continue
            model_display = str(mv.get("method_display", mk))
            folds = mv.get("fold_rows", [])
            if not isinstance(folds, list):
                continue
            for fr in folds:
                if not isinstance(fr, dict):
                    continue
                p = pd.to_numeric(fr.get("pearson", np.nan), errors="coerce")
                t = pd.to_numeric(fr.get("elapsed_sec", np.nan), errors="coerce")
                rows.append(
                    {
                        "model": model_display,
                        "pearson": float(p) if np.isfinite(float(p)) else np.nan,
                        "time": float(t) if np.isfinite(float(t)) else np.nan,
                    }
                )

    if len(rows) == 0:
        return pd.DataFrame(columns=["model", "pearsonr_cv_sd", "time_cv_sd"])

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("model", as_index=False)
        .agg(
            pearsonr_cv_sd=("pearson", "std"),
            time_cv_sd=("time", "std"),
        )
        .reset_index(drop=True)
    )
    return agg


def _rank_by_trait(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_pearson"] = pd.to_numeric(work["pearsonr_cv_mean"], errors="coerce").fillna(-np.inf)
    work["_r2"] = pd.to_numeric(work["r2_cv_mean"], errors="coerce").fillna(-np.inf)
    work["_time"] = pd.to_numeric(work["time_cv_mean_sec"], errors="coerce").fillna(np.inf)
    work = work.sort_values(
        by=["trait", "_pearson", "_r2", "_time", "model"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    work["rank_in_trait"] = work.groupby("trait").cumcount() + 1
    return work.drop(columns=["_pearson", "_r2", "_time"])


def _aggregate_model_ranking(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("model", as_index=False)
        .agg(
            n_traits=("trait", "nunique"),
            pearsonr_cv_mean=("pearsonr_cv_mean", "mean"),
            spearmanr_cv_mean=("spearmanr_cv_mean", "mean"),
            r2_cv_mean=("r2_cv_mean", "mean"),
            pve_final=("pve_final", "mean"),
            time_cv_mean_sec=("time_cv_mean_sec", "mean"),
        )
        .sort_values(
            by=["pearsonr_cv_mean", "r2_cv_mean", "time_cv_mean_sec", "model"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )
    agg.insert(0, "rank", np.arange(1, agg.shape[0] + 1, dtype=int))
    return agg


def _best_models_by_trait(ranked_by_trait: pd.DataFrame) -> pd.DataFrame:
    best = ranked_by_trait[ranked_by_trait["rank_in_trait"] == 1].copy()
    best = best.sort_values(by=["trait"]).reset_index(drop=True)
    return best


def _find_method_rec_by_display(
    summary: dict[str, Any],
    trait: str,
    model_display: str,
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    tmd = summary.get("trait_method_details", {})
    if not isinstance(tmd, dict):
        return None, None
    trait_rec = tmd.get(str(trait), {})
    if not isinstance(trait_rec, dict):
        return None, None
    tgt = _norm_method_token(model_display)
    for mk, mv in trait_rec.items():
        if not isinstance(mv, dict):
            continue
        md = str(mv.get("method_display", mk))
        if _norm_method_token(mk) == tgt or _norm_method_token(md) == tgt:
            return str(mk), mv
    return None, None


def _find_effect_path_and_col_for_best(
    summary: dict[str, Any],
    *,
    trait: str,
    model_display: str,
    summary_dir: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    mk, mv = _find_method_rec_by_display(summary, trait, model_display)
    aliases = {_norm_method_token(model_display)}
    if mk is not None:
        aliases.add(_norm_method_token(mk))
    if isinstance(mv, dict):
        aliases.add(_norm_method_token(mv.get("method_display", model_display)))

    artifacts = summary.get("model_artifacts", [])
    if isinstance(artifacts, list):
        for rec in artifacts:
            if not isinstance(rec, dict):
                continue
            if str(rec.get("trait", "")) != str(trait):
                continue
            m = str(rec.get("method", ""))
            if _norm_method_token(m) in aliases:
                ef = _resolve_existing_path(str(rec.get("effect_file", "")), base_dir=summary_dir)
                if ef is None:
                    ef = _resolve_existing_path(str(rec.get("model_file", "")), base_dir=summary_dir)
                ec = str(rec.get("effect_column", "")).strip() or None
                if ef is not None:
                    return ef, ec, mk

    if isinstance(mv, dict):
        ef = _resolve_existing_path(str(mv.get("effect_file", "")), base_dir=summary_dir)
        if ef is not None:
            return ef, None, mk

    result_files = summary.get("result_files", [])
    if isinstance(result_files, list):
        for rf in result_files:
            p = str(rf)
            if p.endswith(".jxmodel"):
                ef = _resolve_existing_path(p, base_dir=summary_dir)
                if ef is not None:
                    return ef, None, mk
    return None, None, mk


def _resolve_chr_pos_cols(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    chr_cands = ["chrom", "chr", "CHROM", "CHR", "Chromosome"]
    pos_cands = ["pos", "bp", "position", "POS", "BP"]
    chr_col = next((c for c in chr_cands if c in df.columns), None)
    pos_col = next((c for c in pos_cands if c in df.columns), None)
    return chr_col, pos_col


def _collect_trait_effect_specs(
    summary: dict[str, Any],
    *,
    summary_dir: str,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    tmd = summary.get("trait_method_details", {})
    tmd = tmd if isinstance(tmd, dict) else {}
    artifacts = summary.get("model_artifacts", [])
    if isinstance(artifacts, list):
        for rec in artifacts:
            if not isinstance(rec, dict):
                continue
            trait = str(rec.get("trait", "")).strip()
            if trait == "":
                continue
            ef = _resolve_existing_path(str(rec.get("effect_file", "")), base_dir=summary_dir)
            if ef is None:
                ef = _resolve_existing_path(str(rec.get("model_file", "")), base_dir=summary_dir)
            if ef is None:
                continue
            method_key = str(rec.get("method", "")).strip()
            effect_col = str(rec.get("effect_column", "")).strip() or None
            model_display = method_key
            tr = tmd.get(trait, {})
            if isinstance(tr, dict) and method_key in tr and isinstance(tr[method_key], dict):
                model_display = str(tr[method_key].get("method_display", method_key))
            item = {
                "trait": trait,
                "method_key": method_key,
                "model_display": model_display,
                "effect_file": ef,
                "effect_col": effect_col,
            }
            out.setdefault(trait, []).append(item)

    # Fallback path when model_artifacts is unavailable.
    if len(out) == 0 and isinstance(tmd, dict):
        for trait, tr in tmd.items():
            if not isinstance(tr, dict):
                continue
            for mk, mv in tr.items():
                if not isinstance(mv, dict):
                    continue
                ef = _resolve_existing_path(str(mv.get("effect_file", "")), base_dir=summary_dir)
                if ef is None:
                    ef = _resolve_existing_path(str(mv.get("model_file", "")), base_dir=summary_dir)
                if ef is None:
                    continue
                out.setdefault(str(trait), []).append(
                    {
                        "trait": str(trait),
                        "method_key": str(mk),
                        "model_display": str(mv.get("method_display", mk)),
                        "effect_file": ef,
                        "effect_col": None,
                    }
                )

    # Deduplicate and stable sort.
    for trait, items in list(out.items()):
        seen: set[tuple[str, str, str, str]] = set()
        uniq: list[dict[str, Any]] = []
        for it in items:
            k = (
                str(it.get("model_display", "")),
                str(it.get("method_key", "")),
                str(it.get("effect_file", "")),
                str(it.get("effect_col", "")),
            )
            if k in seen:
                continue
            seen.add(k)
            uniq.append(it)
        uniq.sort(key=lambda z: (_norm_method_token(z.get("model_display", "")), str(z.get("model_display", ""))))
        out[trait] = uniq
    return out


def _infer_trait_n_m(summary: dict[str, Any], trait: str) -> tuple[Optional[int], Optional[int]]:
    n_candidates: list[int] = []
    m_candidates: list[int] = []

    rows = summary.get("summary_rows", [])
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            if str(r.get("trait", "")) != str(trait):
                continue
            n_train = pd.to_numeric(r.get("n_train", np.nan), errors="coerce")
            n_test = pd.to_numeric(r.get("n_test", np.nan), errors="coerce")
            if np.isfinite(float(n_train)) and np.isfinite(float(n_test)):
                n_candidates.append(int(round(float(n_train) + float(n_test))))
            for k in ("m", "m_effective", "n_snp"):
                v = pd.to_numeric(r.get(k, np.nan), errors="coerce")
                if np.isfinite(float(v)):
                    m_candidates.append(int(round(float(v))))

    tmd = summary.get("trait_method_details", {})
    if isinstance(tmd, dict):
        tr = tmd.get(str(trait), {})
        if isinstance(tr, dict):
            for _mk, mv in tr.items():
                if not isinstance(mv, dict):
                    continue
                gfs = mv.get("gblup_final_state", {})
                if isinstance(gfs, dict):
                    v = pd.to_numeric(gfs.get("m_effective", np.nan), errors="coerce")
                    if np.isfinite(float(v)):
                        m_candidates.append(int(round(float(v))))
                em = mv.get("effect_meta", {})
                if isinstance(em, dict):
                    v = pd.to_numeric(em.get("rows", np.nan), errors="coerce")
                    if np.isfinite(float(v)):
                        m_candidates.append(int(round(float(v))))

    artifacts = summary.get("model_artifacts", [])
    if isinstance(artifacts, list):
        for rec in artifacts:
            if not isinstance(rec, dict):
                continue
            if str(rec.get("trait", "")) != str(trait):
                continue
            em = rec.get("effect_meta", {})
            if isinstance(em, dict):
                v = pd.to_numeric(em.get("rows", np.nan), errors="coerce")
                if np.isfinite(float(v)):
                    m_candidates.append(int(round(float(v))))

    n_val = max(n_candidates) if len(n_candidates) > 0 else None
    m_val = max(m_candidates) if len(m_candidates) > 0 else None
    return n_val, m_val


def _trait_xlabel_text(summary: dict[str, Any], trait: str) -> str:
    n_val, m_val = _infer_trait_n_m(summary, trait)
    n_txt = str(n_val) if n_val is not None else "NA"
    m_txt = str(m_val) if m_val is not None else "NA"
    return f"{trait} (n={n_txt}, m={m_txt})"


def _guess_effect_col_by_hints(
    df: pd.DataFrame,
    *,
    hint_cols: list[str],
) -> Optional[str]:
    for h in hint_cols:
        if h is None:
            continue
        hs = str(h).strip()
        if hs != "" and hs in df.columns:
            return hs
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for h in hint_cols:
        if h is None:
            continue
        hs = str(h).strip().lower()
        if hs in lower_map:
            return lower_map[hs]
    return None


def _resolve_effect_col(
    df: pd.DataFrame,
    *,
    hint_cols: list[str],
) -> str:
    hit = _guess_effect_col_by_hints(df, hint_cols=hint_cols)
    if hit is not None:
        return str(hit)
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
    for c in df.columns:
        if str(c).strip().lower() in meta_cols:
            continue
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if int(np.isfinite(vals).sum()) > 0:
            return str(c)
    raise ValueError("No numeric effect columns found.")


def _load_effect_table(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".jxmodel":
        try:
            df = pd.read_csv(path, sep="\t")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
        try:
            from janusx.gs.workflow import _build_method_effect_table, _load_jxmodel

            payload = _load_jxmodel(path)
            raw_state = payload.get("model_state", None)
            if isinstance(raw_state, dict) and len(raw_state) > 0:
                prefix_hint_raw = payload.get(
                    "genotype_prefix_hint",
                    raw_state.get(
                        "genotype_prefix_hint",
                        raw_state.get("source_prefix", None),
                    ),
                )
                prefix_hint = (
                    None
                    if prefix_hint_raw is None or str(prefix_hint_raw).strip() == ""
                    else str(prefix_hint_raw)
                )
                fallback_marker_count_raw = payload.get("fallback_marker_count", None)
                fallback_marker_count = (
                    None
                    if fallback_marker_count_raw is None
                    else int(fallback_marker_count_raw)
                )
                table, _meta = _build_method_effect_table(
                    model_state=dict(raw_state),
                    packed_ctx=None,
                    genotype_prefix_hint=prefix_hint,
                    fallback_marker_count=fallback_marker_count,
                )
                return table
        except Exception:
            pass
    if ext == ".csv":
        return pd.read_csv(path)
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=None, engine="python")


_POSTGS_EFFECT_PANEL_TARGET_POINTS = 20_000
_POSTGS_EFFECT_MERGED_TARGET_POINTS = 30_000
_POSTGS_EFFECT_STRONG_KEEP_RATIO = 0.25


def _natkey(text: Any) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(text))
    key: list[Any] = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return tuple(key)


def _chrom_sort_key(value: Any) -> tuple[Any, Any]:
    if value is None:
        return (3, "")
    if isinstance(value, (int, np.integer)):
        return (0, int(value))
    if isinstance(value, (float, np.floating)):
        fv = float(value)
        if np.isfinite(fv):
            iv = int(round(fv))
            if abs(fv - float(iv)) <= 1e-9:
                return (0, iv)
            return (0, fv)
        return (3, "")

    text = str(value).strip()
    if text == "":
        return (3, "")
    upper = text.upper()
    if upper.startswith("CHR"):
        upper = upper[3:].strip()
    try:
        fv = float(upper)
        if np.isfinite(fv):
            iv = int(round(fv))
            if abs(fv - float(iv)) <= 1e-9:
                return (0, iv)
            return (0, fv)
    except Exception:
        pass

    special_rank = {
        "X": 1,
        "Y": 2,
        "XY": 3,
        "M": 4,
        "MT": 5,
    }
    if upper in special_rank:
        return (1, int(special_rank[upper]))
    return (2, _natkey(upper))


def _compress_effect_table_core(
    effect_df: pd.DataFrame,
    *,
    effect_col: str,
    chr_col: Optional[str],
    pos_col: Optional[str],
    max_points: int,
) -> pd.DataFrame:
    target = max(1, int(max_points))
    if effect_df.shape[0] <= target:
        return effect_df.reset_index(drop=True)

    effect_vals = pd.to_numeric(effect_df[effect_col], errors="coerce")
    valid_mask = np.isfinite(effect_vals.to_numpy(dtype=float))
    pos_vals_series: Optional[pd.Series] = None
    if (
        chr_col is not None
        and pos_col is not None
        and chr_col in effect_df.columns
        and pos_col in effect_df.columns
    ):
        pos_vals_series = pd.to_numeric(effect_df[pos_col], errors="coerce")
        valid_mask &= np.isfinite(pos_vals_series.to_numpy(dtype=float))

    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        return effect_df.iloc[0:0].copy()
    if valid_idx.size <= target:
        return effect_df.iloc[valid_idx].reset_index(drop=True)

    effect_arr = effect_vals.to_numpy(dtype=float, copy=False)[valid_mask]
    abs_effect = np.abs(effect_arr)
    strong_target = int(round(float(target) * float(_POSTGS_EFFECT_STRONG_KEEP_RATIO)))
    if target >= 4096:
        strong_target = max(strong_target, 1024)
    strong_target = max(1, min(target, strong_target))

    if strong_target >= valid_idx.size:
        return effect_df.iloc[valid_idx].reset_index(drop=True)

    strong_rel = np.argpartition(-abs_effect, strong_target - 1)[:strong_target]
    keep_rel_mask = np.zeros(valid_idx.size, dtype=bool)
    keep_rel_mask[strong_rel] = True

    if pos_vals_series is not None:
        chr_series = effect_df[chr_col]
        chr_levels = list(pd.unique(chr_series))
        chr_levels.sort(key=_chrom_sort_key)
        chr_map = {level: i for i, level in enumerate(chr_levels)}
        chr_codes = (
            pd.to_numeric(chr_series.map(chr_map), errors="coerce")
            .fillna(-1)
            .to_numpy(dtype=np.int32, copy=False)[valid_mask]
        )
        pos_vals = pos_vals_series.to_numpy(dtype=float, copy=False)[valid_mask]
        order = np.lexsort((pos_vals, chr_codes))
    else:
        order = np.arange(valid_idx.size, dtype=np.int64)

    remainder_target = max(0, target - strong_rel.size)
    if remainder_target > 0:
        ordered_remainder = order[~keep_rel_mask[order]]
        if ordered_remainder.size <= remainder_target:
            keep_rel_mask[ordered_remainder] = True
        elif ordered_remainder.size > 0:
            edges = np.linspace(
                0,
                int(ordered_remainder.size),
                num=int(remainder_target) + 1,
                dtype=np.int64,
            )
            for start, stop in zip(edges[:-1], edges[1:]):
                if stop <= start:
                    continue
                chunk = ordered_remainder[start:stop]
                winner = chunk[int(np.argmax(abs_effect[chunk]))]
                keep_rel_mask[int(winner)] = True

    final_rel = order[keep_rel_mask[order]]
    final_idx = valid_idx[final_rel]
    return effect_df.iloc[final_idx].reset_index(drop=True)


def _compress_effect_table(
    effect_df: pd.DataFrame,
    *,
    effect_col: str,
    max_points: int,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    target = max(1, int(max_points))
    if effect_df.shape[0] <= target:
        return effect_df.reset_index(drop=True)

    chr_col, pos_col = _resolve_chr_pos_cols(effect_df)
    if (
        group_col is not None
        and group_col in effect_df.columns
        and effect_df[group_col].nunique(dropna=False) > 1
    ):
        groups = pd.unique(effect_df[group_col])
        per_group_target = max(1, int(np.ceil(float(target) / max(1, len(groups)))))
        parts: list[pd.DataFrame] = []
        for group_value in groups:
            sub = effect_df.loc[effect_df[group_col] == group_value]
            if sub.shape[0] == 0:
                continue
            parts.append(
                _compress_effect_table_core(
                    sub,
                    effect_col=effect_col,
                    chr_col=chr_col,
                    pos_col=pos_col,
                    max_points=per_group_target,
                )
            )
        if len(parts) == 0:
            return effect_df.iloc[0:0].copy()
        merged = pd.concat(parts, axis=0, ignore_index=True)
        if merged.shape[0] <= target:
            return merged.reset_index(drop=True)
        return _compress_effect_table_core(
            merged,
            effect_col=effect_col,
            chr_col=chr_col,
            pos_col=pos_col,
            max_points=target,
        )

    return _compress_effect_table_core(
        effect_df,
        effect_col=effect_col,
        chr_col=chr_col,
        pos_col=pos_col,
        max_points=target,
    )


def _plot_and_save_effect(
    *,
    effect_df: pd.DataFrame,
    effect_hint: Optional[str],
    palette: Optional[str],
    point_size: float,
    fig_size: tuple[float, float],
    out_prefix: str,
    fmts: list[str],
    saved_paths: list[str],
    x_label: Optional[str] = None,
    fullscatter: bool = False,
) -> str:
    effect_col = _resolve_effect_col(
        effect_df,
        hint_cols=[effect_hint] if effect_hint is not None else [],
    )
    plot_df = (
        effect_df.reset_index(drop=True)
        if bool(fullscatter)
        else _compress_effect_table(
            effect_df,
            effect_col=effect_col,
            max_points=_POSTGS_EFFECT_PANEL_TARGET_POINTS,
        )
    )
    fig = plt.figure(figsize=fig_size, dpi=300)
    ax = fig.add_subplot(111)
    ax, _effect_col = gsplot.plot_signed_effect(
        plot_df,
        effect_col=effect_col,
        ax=ax,
        palette=palette,
        rasterized=True,
        point_size=point_size,
    )
    ax.set_title("")
    if x_label is not None and str(x_label).strip() != "":
        ax.set_xlabel(str(x_label))
    for fmt in fmts:
        out_path = f"{out_prefix}.effect_signed.{fmt}"
        _save_fig(fig, out_path, fmt)
        saved_paths.append(str(out_path))
    plt.close(fig)
    return effect_col


def _plot_and_save_trait_effect_panels(
    *,
    trait: str,
    signed_panels: list[dict[str, Any]],
    merged_df: Optional[pd.DataFrame],
    palette: Optional[str],
    point_size: float,
    subplot_size: tuple[float, float],
    out_prefix: str,
    fmts: list[str],
    saved_paths: list[str],
    x_label: str,
    fullscatter: bool = False,
) -> bool:
    n_signed = len(signed_panels)
    has_merged = merged_df is not None and isinstance(merged_df, pd.DataFrame) and merged_df.shape[0] > 0
    n_total = n_signed + (1 if has_merged else 0)
    if n_total <= 0:
        return False

    cols = min(3, max(1, n_total))
    rows = int(np.ceil(n_total / cols))
    sub_w, sub_h = subplot_size
    fig_w = max(2.0, float(sub_w) * cols)
    fig_h = max(2.0, float(sub_h) * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=300, squeeze=False)
    ax_list = [ax for row in axes for ax in row]

    idx = 0
    for panel in signed_panels:
        ax = ax_list[idx]
        model_name = str(panel.get("model_name", "")).strip() or f"model{idx + 1}"
        eff_df = panel.get("effect_df", None)
        eff_col = panel.get("effect_col", None)
        if isinstance(eff_df, pd.DataFrame):
            plot_df = (
                eff_df.reset_index(drop=True)
                if bool(fullscatter)
                else _compress_effect_table(
                    eff_df,
                    effect_col=str(eff_col) if eff_col is not None else _resolve_effect_col(eff_df, hint_cols=[]),
                    max_points=_POSTGS_EFFECT_PANEL_TARGET_POINTS,
                )
            )
            gsplot.plot_signed_effect(
                plot_df,
                effect_col=str(eff_col) if eff_col is not None else None,
                ax=ax,
                palette=palette,
                rasterized=True,
                point_size=point_size,
            )
            ax.set_title("")
            ax.text(
                0.01,
                0.98,
                model_name,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.65),
            )
            ax.set_xlabel(x_label)
        else:
            ax.text(0.5, 0.5, f"{model_name}\nNo data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        idx += 1

    if has_merged:
        ax = ax_list[idx]
        merged_plot_df = (
            merged_df.reset_index(drop=True)
            if bool(fullscatter)
            else _compress_effect_table(
                merged_df,
                effect_col="Effect",
                max_points=_POSTGS_EFFECT_MERGED_TARGET_POINTS,
                group_col="Model",
            )
        )
        gsplot.plot_effect_models_layered(
            merged_plot_df,
            ax=ax,
            palette=palette,
            model_col="Model",
            chr_col="chrom",
            pos_col="pos",
            effect_col="Effect",
            rasterized=True,
            point_size=point_size,
        )
        ax.set_title("")
        ax.text(
            0.01,
            0.98,
            "Merged",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.65),
        )
        ax.set_xlabel(x_label)
        idx += 1

    for j in range(idx, len(ax_list)):
        ax_list[j].set_axis_off()

    trait_tag = _sanitize_token(trait)
    for fmt in fmts:
        out_path = f"{out_prefix}.{trait_tag}.effects.{fmt}"
        _save_fig(fig, out_path, fmt)
        saved_paths.append(str(out_path))
    plt.close(fig)
    return True


def _log_step_ok(logger: Any, label: str, sec: float) -> None:
    logger.info(f"  ✔︎ {str(label):<34} [{float(sec):.1f}s]")


def _log_step_skip(logger: Any, label: str, reason: str) -> None:
    logger.info(f"  - {str(label):<34} [{reason}]")


def _log_results_overview(
    logger: Any,
    *,
    out_dir: str,
    prefix: str,
    saved_paths: list[str],
) -> None:
    uniq = sorted({str(p) for p in saved_paths})
    _ = out_dir
    _ = prefix
    logger.info("Results saved:")
    if len(uniq) == 0:
        logger.info("  (none)")
        return
    cwd = os.getcwd()
    for p in uniq:
        abs_p = os.path.abspath(str(p))
        rel_p = os.path.relpath(abs_p, start=cwd)
        logger.info(f"  {rel_p}")


def _log_missing_effect_hint(logger: Any, summary: dict[str, Any]) -> None:
    usage = summary.get("usage", {})
    effect_pattern = str(usage.get("effect_file_pattern", "{trait}.{method}.jxmodel")).strip()
    if effect_pattern == "":
        effect_pattern = "{trait}.{method}.jxmodel"
    logger.warning("No effect files were found in summary.json; effect plotting was skipped.")
    logger.warning(
        "Rerun `jx gs` with `-save-model` to export `%s`, or pass `-effect <file>` to `jx postgs`.",
        effect_pattern,
    )


def main() -> None:
    parser = CliArgumentParser(
        prog="jx postgs",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx postgs -json test/mouse_hs1940.gs.model/summary.json",
            "jx postgs -json summary.json -palette tab10 -fmt png",
            "jx postgs -json summary.json -pcctime 1 tab10 -violin 1 '#5ca68d' -manh 2 '#4c78a8,#f58518,#54a24b,#e45756'",
            "jx postgs -json summary.json -effect test/mouse_hs1940.gs.model/test0.BayesB.jxmodel -effect-col beta",
        ]),
    )
    req = parser.add_argument_group("Required Arguments")
    req.add_argument(
        "-json",
        "--json",
        type=str,
        required=True,
        help="GS summary.json path (from jx gs output model directory).",
    )

    opt = parser.add_argument_group("Optional Arguments")
    opt.add_argument("-o", "--out", type=str, default=".", help="Output directory.")
    opt.add_argument(
        "-prefix",
        "--prefix",
        type=str,
        default=None,
        help="Output prefix (default: inferred from summary.json).",
    )
    opt.add_argument(
        "-fmt",
        "--fmt",
        type=str,
        default="png",
        help="Figure format(s), e.g. png or png,pdf,svg (default: %(default)s).",
    )
    opt.add_argument(
        "-palette",
        "--palette",
        "-pallete",
        "--pallete",
        dest="palette",
        type=str,
        default="tab10",
        help="Unified plotting palette (e.g. tab10, tab20, #1f77b4;#ff7f0e).",
    )
    opt.add_argument(
        "-effect",
        "--effect",
        type=str,
        default=None,
        help="Optional effect file for standalone signed-effect plotting.",
    )
    opt.add_argument(
        "-effect-col",
        "--effect-col",
        type=str,
        default=None,
        help="Optional effect column for standalone effect plotting.",
    )
    opt.add_argument(
        "-manh",
        "--manh",
        nargs="*",
        default=None,
        help=(
            "Enable effect Manhattan plotting (JSON-driven). Optional spec style: "
            "'ratio palette' (e.g. -manh 2 tab10) or legacy 'ratio,palette'. "
            "Default ratio=2."
        ),
    )
    opt.add_argument(
        "-violin",
        "--violin",
        nargs="*",
        default=None,
        help=(
            "Enable split violin plotting (JSON-driven). Optional spec style: "
            "'ratio palette' (e.g. -violin 1 '#5ca68d') or legacy 'ratio,palette'. "
            "Default ratio=1."
        ),
    )
    opt.add_argument(
        "-pcctime",
        "--pcctime",
        nargs="*",
        default=None,
        help=(
            "Enable Pearson-vs-runtime plotting (JSON-driven). Optional spec style: "
            "'ratio palette' (e.g. -pcctime 1 tab10) or legacy 'ratio,palette'. "
            "Default ratio=1."
        ),
    )
    opt.add_argument(
        "-scatter-size",
        "--scatter-size",
        type=float,
        default=4.0,
        help="Base scatter size control for effect plots (runtime scatter uses scaled size).",
    )
    opt.add_argument(
        "-full",
        "--full",
        "-fullscatter",
        "--fullscatter",
        dest="fullscatter",
        action="store_true",
        default=False,
        help=(
            "Disable effect scatter compression and draw all marker-effect points "
            "(single panels, merged panels, and standalone -effect plots)."
        ),
    )
    args, extras = parser.parse_known_args()
    if len(extras) > 0:
        parser.error("unrecognized arguments: " + " ".join(extras))

    args.out = os.path.normpath(str(args.out) if args.out is not None else ".")
    os.makedirs(args.out, mode=0o755, exist_ok=True)

    check_logger: Any = _SilentLogger()
    if args.effect is not None:
        checks = [
            ensure_file_exists(check_logger, args.json, "summary.json"),
            ensure_file_exists(check_logger, args.effect, "effect file"),
        ]
    else:
        checks = [ensure_file_exists(check_logger, args.json, "summary.json")]
    if not ensure_all_true(checks):
        logging.error("Input file(s) not found. Please check -json/--json and -effect/--effect paths.")
        raise SystemExit(1)

    with open(args.json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    prefix = str(args.prefix).strip() if args.prefix is not None else _infer_prefix(summary, args.json)
    outprefix = os.path.join(args.out, prefix)
    log_path = os.path.join(args.out, f"{prefix}.postgs.log")
    logger = setup_logging(log_path)

    fmts = _parse_formats(args.fmt)
    saved_paths: list[str] = []
    summary_dir = str(Path(args.json).resolve().parent)
    effect_specs_by_trait = _collect_trait_effect_specs(summary, summary_dir=summary_dir)
    effect_table_cache: dict[str, pd.DataFrame] = {}

    has_selector = any(x is not None for x in (args.manh, args.violin, args.pcctime))
    do_manh = bool(args.effect is not None) or (not has_selector) or (args.manh is not None)
    do_violin = (not has_selector) or (args.violin is not None)
    do_pcctime = (not has_selector) or (args.pcctime is not None)
    manh_ratio, manh_palette = _parse_ratio_palette_spec(args.manh, default_ratio=2.0)
    violin_ratio, violin_palette = _parse_ratio_palette_spec(args.violin, default_ratio=1.0)
    pcctime_ratio, pcctime_palette = _parse_ratio_palette_spec(args.pcctime, default_ratio=1.0)
    if manh_palette is None:
        manh_palette = args.palette
    if violin_palette is None:
        violin_palette = args.palette
    if pcctime_palette is None:
        pcctime_palette = args.palette

    effect_point_size = max(float(args.scatter_size), 0.5)
    runtime_point_size = max(10.0, effect_point_size * 62.5)
    manh_fig_size = _figsize_from_ratio(manh_ratio, base_height=4.8)
    merged_manh_fig_size = _figsize_from_ratio(manh_ratio, base_height=5.0)
    violin_fig_size = _figsize_from_ratio(violin_ratio, base_height=6.0)
    pcctime_fig_size = _figsize_from_ratio(pcctime_ratio, base_height=6.2)

    df = _summary_rows_frame(summary)
    cv_long = _cv_accuracy_long_frame(summary)
    trait_eval_items = sorted(df["trait"].astype(str).unique().tolist(), key=str)
    ui = _PostGsCliUI(stream=sys.stdout)

    if do_pcctime or do_violin:
        phase1 = ui.start_phase("[1/2] Generating trait-wise evaluation plots...")
        for trait in trait_eval_items:
            trait_s = str(trait)
            trait_df = df[df["trait"].astype(str) == trait_s].copy()
            if trait_df.shape[0] == 0:
                continue
            trait_model_rank = _aggregate_model_ranking(trait_df)
            trait_cv_err = _model_cv_error_frame(summary, trait=trait_s)
            if trait_cv_err.shape[0] > 0:
                trait_model_rank = trait_model_rank.merge(trait_cv_err, on="model", how="left")

            if cv_long.shape[0] > 0:
                trait_cv_long = cv_long[cv_long["Trait"].astype(str) == trait_s].copy()
            else:
                trait_cv_long = cv_long

            trait_out_dir = os.path.join(args.out, _sanitize_token(trait_s))
            os.makedirs(trait_out_dir, mode=0o755, exist_ok=True)
            trait_outprefix = os.path.join(trait_out_dir, prefix)

            # Accuracy-runtime
            if do_pcctime:
                step_t0 = time.time()
                fig = plt.figure(figsize=pcctime_fig_size, dpi=300)
                ax = fig.add_subplot(111)
                gsplot.plot_accuracy_runtime_scatter(
                    trait_model_rank,
                    ax=ax,
                    palette=pcctime_palette,
                    x_col="time_cv_mean_sec",
                    y_col="pearsonr_cv_mean",
                    label_col="model",
                    point_size=runtime_point_size,
                )
                ax.set_title("")
                for fmt in fmts:
                    p = f"{trait_outprefix}.accuracy_runtime.{fmt}"
                    _save_fig(fig, p, fmt)
                    saved_paths.append(str(p))
                plt.close(fig)
                phase1.ok(f"{trait_s} accuracy-runtime scatter", time.time() - step_t0)

            # Split violin
            if do_violin and trait_cv_long.shape[0] > 0:
                step_t0 = time.time()
                fig = plt.figure(figsize=violin_fig_size, dpi=300)
                ax = fig.add_subplot(111)
                gsplot.plot_accuracy_split_violin(
                    trait_cv_long,
                    ax=ax,
                    palette=violin_palette,
                    model_order=trait_model_rank["model"].astype(str).tolist(),
                    model_col="Model",
                    metric_col="Metric",
                    value_col="Accuracy",
                )
                ax.set_title("")
                for fmt in fmts:
                    p = f"{trait_outprefix}.accuracy_violin.{fmt}"
                    _save_fig(fig, p, fmt)
                    saved_paths.append(str(p))
                plt.close(fig)
                phase1.ok(f"{trait_s} split violin (Pearson/Spearman)", time.time() - step_t0)
        phase1.complete()

    trait_items = sorted(effect_specs_by_trait.keys(), key=str) if do_manh else []
    if len(trait_items) == 0:
        ui.plain("")
        phase2_empty = ui.start_phase("[2/2] Plotting marker effects...")
        if do_manh:
            phase2_empty.note("  - no effect files")
            if args.effect is None:
                _log_missing_effect_hint(logger, summary)
        else:
            phase2_empty.note("  - disabled")
        phase2_empty.complete()

    for trait in trait_items:
        specs = effect_specs_by_trait[trait]
        if len(specs) == 0:
            continue
        ui.plain("")
        phase2 = ui.start_phase(f"[2/2] Plotting marker effects (Trait: {trait})...")
        trait_xlabel = _trait_xlabel_text(summary, trait)
        signed_panels: list[dict[str, Any]] = []
        merged_parts: list[pd.DataFrame] = []

        for spec in specs:
            model_name = str(spec.get("model_display", "")).strip() or str(spec.get("method_key", "model"))
            method_key = str(spec.get("method_key", "")).strip()
            eff_path = str(spec.get("effect_file", "")).strip()
            eff_col_hint = spec.get("effect_col", None)
            label = f"{model_name} (signed effect)"
            if eff_path == "":
                phase2.note(f"  - {label:<34} [no effect path]")
                continue

            step_t0 = time.time()
            if eff_path not in effect_table_cache:
                effect_table_cache[eff_path] = _load_effect_table(eff_path)
            df_eff = effect_table_cache[eff_path]
            hint_cols = [eff_col_hint, model_name, method_key]
            guess_col = _guess_effect_col_by_hints(df_eff, hint_cols=hint_cols)
            if guess_col is None:
                phase2.note(f"  - {label:<34} [effect column not found]")
                continue
            used_col = str(guess_col)
            signed_panels.append(
                {
                    "model_name": model_name,
                    "effect_df": df_eff,
                    "effect_col": used_col,
                }
            )
            chr_col, pos_col = _resolve_chr_pos_cols(df_eff)
            if chr_col is not None and pos_col is not None and used_col in df_eff.columns:
                part = pd.DataFrame(
                    {
                        "Model": model_name,
                        "chrom": df_eff[chr_col].astype(str),
                        "pos": pd.to_numeric(df_eff[pos_col], errors="coerce"),
                        "Effect": pd.to_numeric(df_eff[used_col], errors="coerce"),
                    }
                )
                part = part[
                    np.isfinite(part["pos"].to_numpy(dtype=float))
                    & np.isfinite(part["Effect"].to_numpy(dtype=float))
                ]
                if part.shape[0] > 0:
                    scaled_eff, max_abs = _max_abs_scale(part["Effect"])
                    part["Effect"] = scaled_eff
                    if np.isfinite(max_abs):
                        part["ScaleMaxAbs"] = float(max_abs)
                    merged_parts.append(part)
            phase2.ok(label, time.time() - step_t0)

        merged_df: Optional[pd.DataFrame] = None
        if len(merged_parts) > 0:
            step_t0 = time.time()
            merged_df = pd.concat(merged_parts, axis=0, ignore_index=True)
            phase2.ok("Merged layered effect", time.time() - step_t0)
        else:
            phase2.note("  - Merged layered effect              [no valid points]")

        if len(signed_panels) > 0 or (merged_df is not None and merged_df.shape[0] > 0):
            step_t0 = time.time()
            _plot_and_save_trait_effect_panels(
                trait=trait,
                signed_panels=signed_panels,
                merged_df=merged_df,
                palette=manh_palette,
                point_size=effect_point_size,
                subplot_size=manh_fig_size,
                out_prefix=outprefix,
                fmts=fmts,
                saved_paths=saved_paths,
                x_label=trait_xlabel,
                fullscatter=bool(args.fullscatter),
            )
            phase2.ok("Combined effect panel", time.time() - step_t0)
        else:
            phase2.note("  - Combined effect panel              [no valid panels]")
        phase2.complete()

    if args.effect is not None and do_manh:
        logger.info("")
        logger.info("[extra] Plotting standalone effect file...")
        step_t0 = time.time()
        eff_path = _resolve_existing_path(str(args.effect), base_dir=os.getcwd())
        if eff_path is None:
            raise FileNotFoundError(f"effect file not found: {args.effect}")
        df_eff = _load_effect_table(eff_path)
        tag = f"{outprefix}.manual"
        _plot_and_save_effect(
            effect_df=df_eff,
            effect_hint=args.effect_col,
            palette=manh_palette,
            point_size=effect_point_size,
            fig_size=manh_fig_size,
            out_prefix=tag,
            fmts=fmts,
            saved_paths=saved_paths,
            x_label="manual (n=NA, m=NA)",
            fullscatter=bool(args.fullscatter),
        )
        _log_step_ok(logger, "Standalone signed effect", time.time() - step_t0)
    elif args.effect is not None and not do_manh:
        logger.info("")
        _log_step_skip(logger, "Standalone signed effect", "disabled by plot selector")

    logger.info("")
    _log_results_overview(
        logger,
        out_dir=args.out,
        prefix=prefix,
        saved_paths=saved_paths,
    )


if __name__ == "__main__":
    main()
