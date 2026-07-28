# -*- coding: utf-8 -*-
"""UI/status helpers for GWAS workflow output and plotting."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from janusx.script._common.cjk import contains_cjk as _contains_cjk, ensure_cjk_font as _ensure_cjk_font
from janusx.script._common.progress import (
    ProgressBarAdapter as _BaseProgressAdapter,
    SpinnerStatusAdapter,
    CliStatus,
    format_elapsed,
    is_skip_status_text,
    print_success,
    print_warning,
    stdout_is_tty,
)
from janusx.script._common.threads import runtime_thread_stage


_GWAS_PROGRESS_BAR_WIDTH = 30
_GWAS_LOG_PROGRESS_MIN_INTERVAL_SEC = 2.0
_GWAS_MODEL_LINE_WIDTH = 80
_GWAS_MODEL_DONE_RE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z0-9]*(?:\([^)]*\))?)\s+\.\.\.(?P<body>.*?)(?:\s+\[(?P<times>[^\]]+)\])?$"
)
_GWAS_MODEL_LABELS = {
    "LM",
    "LMM",
    "FvLMM",
    "SparseLMM",
    "ALGWAS",
    "FarmCPU",
}
_FASTPLOT_FIGSIZE = (16.0, 4.0)
_FASTPLOT_DPI = 300
_FASTPLOT_SCATTER_SIZE = 8.0
_FASTPLOT_HIST_BINS = 15
_FASTPLOT_MANHATTAN_INTERVAL_RATE = 0.5
_FASTPLOT_SINGLETON_COLOR = "#C7CCD3"
_FASTPLOT_SIG_COMBO_COLOR = "#D62728"
_FASTPLOT_COMBO_FDR_THRESHOLD = 0.05
_FASTPLOT_COMBO_FDR_COLUMNS = ("padj", "pwald_fdr", "p_combo_joint_fdr")


def _histogram_kde_count_curve(
    values: np.ndarray,
    *,
    edges: np.ndarray,
    n_grid: int = 256,
) -> tuple[np.ndarray, np.ndarray] | None:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return None
    if np.unique(vals).size < 2:
        return None

    q75, q25 = np.percentile(vals, [75.0, 25.0])
    iqr = float(q75 - q25)
    std = float(np.std(vals, ddof=1))
    robust_scale = iqr / 1.34 if iqr > 0.0 else 0.0
    if robust_scale > 0.0 and np.isfinite(robust_scale):
        scale = min(std, robust_scale) if std > 0.0 and np.isfinite(std) else robust_scale
    else:
        scale = std
    if (not np.isfinite(scale)) or scale <= 0.0:
        scale = float(np.ptp(vals)) / 6.0
    if (not np.isfinite(scale)) or scale <= 0.0:
        return None

    bandwidth = 0.9 * scale * (float(vals.size) ** (-0.2))
    if (not np.isfinite(bandwidth)) or bandwidth <= 0.0:
        return None

    x_min = float(np.min(vals))
    x_max = float(np.max(vals))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return None
    if x_max <= x_min:
        x_max = x_min + bandwidth

    x_grid = np.linspace(x_min, x_max, int(max(32, n_grid)), dtype=np.float64)
    z = (x_grid[:, None] - vals[None, :]) / bandwidth
    kernel = np.exp(-0.5 * np.square(z)) / np.sqrt(2.0 * np.pi)
    density = np.mean(kernel, axis=1) / bandwidth

    edge_arr = np.asarray(edges, dtype=np.float64).reshape(-1)
    if edge_arr.size >= 2:
        bin_width = float(np.mean(np.diff(edge_arr)))
    else:
        bin_width = bandwidth
    if (not np.isfinite(bin_width)) or bin_width <= 0.0:
        bin_width = bandwidth

    y_count = density * float(vals.size) * bin_width
    return x_grid, y_count


def _fastplot_sanitize_pvalues(values: object) -> np.ndarray:
    p = pd.to_numeric(values, errors="coerce")
    if isinstance(p, pd.Series):
        arr = p.to_numpy(dtype=np.float64, copy=False)
    else:
        arr = np.asarray(p, dtype=np.float64)
    arr = np.array(arr, dtype=np.float64, copy=True).reshape(-1)
    arr[~np.isfinite(arr)] = 1.0
    return np.clip(arr, np.nextafter(0.0, 1.0), 1.0)


def _resolve_fastplot_combo_fdr_column(columns: object) -> str | None:
    colset = {str(c) for c in columns}
    for name in _FASTPLOT_COMBO_FDR_COLUMNS:
        if name in colset:
            return str(name)
    return None


def _normalize_fastplot_style(style: object) -> str:
    text = str(style or "").strip().lower()
    if text == "garfield":
        return "garfield"
    return "gwas"


def _prepare_role_layered_manhattan_df(
    results: pd.DataFrame,
    gwasplot,
) -> pd.DataFrame | None:
    results_df = results.reset_index(drop=True).copy()
    if "row_role" not in results_df.columns and "snp" in results_df.columns:
        snp_text = results_df["snp"].astype(str)
        inferred_combo = snp_text.str.contains(r"[&|*]", regex=True, na=False)
        results_df["row_role"] = np.where(inferred_combo, "combo", "singleton")
    if "row_role" not in results_df.columns:
        return None
    if results_df.shape[0] == 0:
        return pd.DataFrame(columns=["x", "y", "z", "row_role"])
    results_df["chrom"] = results_df["chrom"].astype(str)
    results_df["pos"] = pd.to_numeric(results_df["pos"], errors="coerce").fillna(0).astype(np.int64)
    results_df["row_role"] = results_df["row_role"].astype(str).str.strip().str.lower()
    combo_fdr_col = _resolve_fastplot_combo_fdr_column(results_df.columns)
    results_df["__combo_sig"] = False
    if combo_fdr_col is not None:
        combo_fdr = _fastplot_sanitize_pvalues(results_df[combo_fdr_col])
        results_df["__combo_sig"] = (
            results_df["row_role"].eq("combo")
            & (combo_fdr < float(_FASTPLOT_COMBO_FDR_THRESHOLD))
        )
    chr_map = {str(label): i + 1 for i, label in enumerate(getattr(gwasplot, "chr_labels", []))}
    results_df["__chr_code"] = results_df["chrom"].map(chr_map)
    if results_df["__chr_code"].isna().any():
        return None
    results_sorted = (
        results_df.sort_values(by=["__chr_code", "pos"], kind="mergesort")
        .reset_index(drop=True)
    )
    if results_sorted.shape[0] != int(gwasplot.df.shape[0]):
        return None
    keep_idx = np.asarray(getattr(gwasplot, "minidx", []), dtype=np.int64)
    if keep_idx.size == 0:
        return pd.DataFrame(columns=["x", "y", "z", "row_role"])
    kept_roles = results_sorted.iloc[keep_idx]["row_role"].to_numpy(dtype=object, copy=False)
    kept_combo_sig = results_sorted.iloc[keep_idx]["__combo_sig"].to_numpy(dtype=bool, copy=False)
    plot_df = gwasplot.df.iloc[keep_idx].copy().reset_index(drop=True)
    plot_df = plot_df.loc[:, ["x", "y", "z"]].copy()
    plot_df["y"] = -np.log10(_fastplot_sanitize_pvalues(plot_df["y"]))
    plot_df["row_role"] = kept_roles
    plot_df["combo_sig"] = kept_combo_sig
    plot_df = plot_df[np.isfinite(plot_df["x"]) & np.isfinite(plot_df["y"]) & np.isfinite(plot_df["z"])]
    plot_df = plot_df.reset_index(drop=True)
    plot_df.attrs["combo_fdr_col"] = combo_fdr_col
    return plot_df


def _prepare_fastplot_qq_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.shape[0] == 0:
        return results.copy()
    qq_df = results.reset_index(drop=True).copy()
    if "row_role" not in qq_df.columns and "snp" in qq_df.columns:
        snp_text = qq_df["snp"].astype(str)
        inferred_combo = snp_text.str.contains(r"[&|*]", regex=True, na=False)
        qq_df["row_role"] = np.where(inferred_combo, "combo", "singleton")
    if "snp" not in qq_df.columns or "row_role" not in qq_df.columns:
        return qq_df
    row_role = qq_df["row_role"].astype(str).str.strip().str.lower()
    combo_mask = row_role.eq("combo")
    if not bool(combo_mask.any()):
        return qq_df
    keep_combo = ~qq_df.loc[combo_mask].duplicated(subset=["snp"], keep="first")
    combo_mask_np = combo_mask.to_numpy(dtype=bool, copy=False)
    keep_mask = np.array(~combo_mask_np, dtype=bool, copy=True)
    keep_mask[combo_mask_np] = keep_combo.to_numpy(
        dtype=bool,
        copy=False,
    )
    return qq_df.loc[keep_mask].reset_index(drop=True)


def _draw_role_layered_manhattan(
    results: pd.DataFrame,
    gwasplot,
    *,
    ax: plt.Axes,
    threshold: float | None,
    scatter_size: float,
    min_logp: float = 0.5,
) -> bool:
    layered_df = _prepare_role_layered_manhattan_df(results, gwasplot)
    if layered_df is None:
        return False

    from janusx.bioplotkit import apply_integer_yticks, color_set as _bioplotkit_color_set

    plot_df = layered_df.copy()
    plot_df = plot_df[plot_df["y"] >= float(min_logp)]
    if plot_df.shape[0] == 0:
        ax.text(0.5, 0.5, "No SNPs", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks(gwasplot.ticks_loc, gwasplot.chr_labels)
        ax.set_xlabel("Chromosome")
        ax.set_ylabel("-log10(p-value)")
        return True

    combo_fdr_col = str(layered_df.attrs.get("combo_fdr_col") or "").strip()
    has_combo_fdr = combo_fdr_col != ""
    singleton_mask = plot_df["row_role"] == "singleton"
    combo_mask = plot_df["row_role"] == "combo"
    combo_sig_mask = combo_mask & plot_df.get("combo_sig", False).astype(bool)
    combo_bg_mask = combo_mask & ~combo_sig_mask
    combo_palette = list(_bioplotkit_color_set.get(6, []))
    if len(combo_palette) == 0:
        combo_palette = ["#3E4F94", "#3E90BF"]
    chr_color_map = dict(
        zip(
            gwasplot.chr_ids,
            [combo_palette[i % len(combo_palette)] for i in range(len(gwasplot.chr_ids))],
        )
    )

    base_mask = singleton_mask | combo_bg_mask if has_combo_fdr else singleton_mask
    if bool(np.any(base_mask)):
        single_df = plot_df.loc[base_mask, ["x", "y"]]
        ax.scatter(
            single_df["x"],
            single_df["y"],
            color=_FASTPLOT_SINGLETON_COLOR,
            s=max(1.0, float(scatter_size) * 0.95),
            alpha=0.55,
            linewidths=0.0,
            rasterized=True,
            zorder=1,
        )

    if has_combo_fdr:
        if bool(np.any(combo_sig_mask)):
            combo_sig_df = plot_df.loc[combo_sig_mask, ["x", "y"]]
            ax.scatter(
                combo_sig_df["x"],
                combo_sig_df["y"],
                color=_FASTPLOT_SIG_COMBO_COLOR,
                s=max(1.0, float(scatter_size) * 1.2),
                alpha=0.95,
                linewidths=0.0,
                rasterized=True,
                zorder=4,
            )
    elif bool(np.any(combo_mask)):
        combo_df = plot_df.loc[combo_mask, ["x", "y", "z"]]
        for chr_id in gwasplot.chr_ids:
            chr_mask = combo_df["z"] == chr_id
            if not bool(np.any(chr_mask)):
                continue
            ax.scatter(
                combo_df.loc[chr_mask, "x"],
                combo_df.loc[chr_mask, "y"],
                color=chr_color_map[chr_id],
                s=max(1.0, float(scatter_size) * 1.05),
                alpha=0.88,
                linewidths=0.0,
                rasterized=True,
                zorder=3,
            )

    if threshold is not None:
        ax.axhline(
            y=float(threshold),
            color="grey",
            linewidth=1,
            linestyle="--",
        )

    if gwasplot._chr_separators.size > 0:
        for xsep in gwasplot._chr_separators:
            if np.isfinite(xsep):
                ax.axvline(
                    float(xsep),
                    ymin=0.0,
                    ymax=1.0 / 3.0,
                    linestyle="--",
                    color="lightgrey",
                    linewidth=0.6,
                    alpha=0.8,
                    zorder=0,
                )

    ax.set_xticks(gwasplot.ticks_loc, gwasplot.chr_labels)
    if gwasplot._chr_bounds_min.size > 0 and gwasplot._chr_bounds_max.size > 0:
        xmin = float(gwasplot._chr_bounds_min[0]) - float(gwasplot._edge_padding_x)
        xmax = float(gwasplot._chr_bounds_max[-1]) + float(gwasplot._edge_padding_x)
    else:
        xmin = float(plot_df["x"].min())
        xmax = float(plot_df["x"].max())
    if xmax > xmin:
        ax.set_xlim(xmin, xmax)
    else:
        eps = max(1e-9, abs(xmin) * 1e-9)
        ax.set_xlim(xmin - eps, xmax + eps)
    ax.margins(x=0.0)
    ymax = float(plot_df["y"].max())
    top = ymax + 0.1 * ymax if ymax > 0.0 else (float(min_logp) + 1.0)
    if top <= float(min_logp):
        top = float(min_logp) + 1.0
    ax.set_ylim([float(min_logp), top])
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("-log10(p-value)")
    apply_integer_yticks(ax)
    return True


def _logger_flag(logger: Optional[logging.Logger], name: str, default: bool = False) -> bool:
    if logger is None:
        return bool(default)
    try:
        return bool(getattr(logger, name))
    except Exception:
        return bool(default)


def _gwas_progress_enabled(logger: Optional[logging.Logger]) -> bool:
    env_raw = str(os.environ.get("JX_GWAS_PROGRESS", "0")).strip().lower()
    env_enabled = env_raw in {"1", "true", "yes", "y", "on"}
    return bool(_logger_flag(logger, "_janusx_gwas_progress_enabled", env_enabled))


def _gwas_verbose_enabled(logger: Optional[logging.Logger]) -> bool:
    return bool(_logger_flag(logger, "_janusx_gwas_verbose", False))


def _format_gwas_model_completion_line(message: object) -> Optional[str]:
    msg = str(message).strip()
    if msg == "":
        return None
    m = _GWAS_MODEL_DONE_RE.match(msg)
    if m is None:
        return None
    label_raw = str(m.group("label") or "").strip()
    label = re.sub(r"\([^)]*\)$", "", label_raw).strip()
    if label not in _GWAS_MODEL_LABELS:
        return None
    body = str(m.group("body") or "").strip()
    if body == "":
        body = "Finished"
    if body.lower().startswith("pve "):
        body = f"PVE={body[4:].strip()}"
    times = str(m.group("times") or "").strip()
    prefix = f"  > {label:<14}: "
    suffix = f"[ {times} ]" if times != "" else ""
    if suffix != "":
        dots = max(2, _GWAS_MODEL_LINE_WIDTH - len(prefix) - len(body) - len(suffix) - 1)
        return f"{prefix}{body} {'.' * dots} {suffix}"
    return f"{prefix}{body}"


def _progress_callback_step(total: int, *, updates: int = _GWAS_PROGRESS_BAR_WIDTH) -> int:
    tot = int(max(1, total))
    n_updates = int(max(8, updates))
    return int(max(1, (tot + n_updates - 1) // n_updates))


def _format_progress_metric(
    done: int,
    total: int,
    memory_text: Optional[str] = None,
) -> str:
    tot = int(max(0, total))
    dn = int(max(0, done))
    if tot <= 0:
        pct = 0.0
    else:
        pct = 100.0 * float(min(dn, tot)) / float(tot)
    mem_raw = str(memory_text or "").strip()
    if mem_raw != "":
        return f"[{mem_raw:>7}]"
    return f"[{pct:5.1f}% ]"


def _log_file_only(logger: logging.Logger, level: int, message: str) -> None:
    """
    Log to file handlers only; avoid echoing to terminal stream handlers.
    """
    rec = logger.makeRecord(
        logger.name,
        level,
        fn="",
        lno=0,
        msg=str(message),
        args=(),
        exc_info=None,
    )
    handled = False
    for h in logger.handlers:
        try:
            if isinstance(h, logging.FileHandler):
                h.handle(rec)
                handled = True
        except Exception:
            continue
    if (not handled) and logger.propagate:
        parent = logger.parent
        while parent is not None:
            for h in parent.handlers:
                try:
                    if isinstance(h, logging.FileHandler):
                        h.handle(rec)
                        handled = True
                except Exception:
                    continue
            if handled:
                break
            parent = parent.parent


def _log_info(logger: logging.Logger, message: str, *, use_spinner: bool = False) -> None:
    if use_spinner:
        _log_file_only(logger, logging.INFO, str(message))
    else:
        logger.info(str(message))


def _emit_plain_info_line(
    logger: logging.Logger,
    message: str,
    *,
    use_spinner: bool = False,
) -> None:
    """
    Emit a plain (non-success-styled) info line.
    In spinner mode we print to terminal and also keep file logging.
    """
    msg = str(message)
    if use_spinner:
        _log_file_only(logger, logging.INFO, msg)
        print(msg, flush=True)
    else:
        logger.info(msg)


def _emit_warning_line(
    logger: logging.Logger,
    message: str,
    *,
    use_spinner: bool = False,
) -> None:
    msg = str(message)
    if use_spinner or stdout_is_tty():
        _log_file_only(logger, logging.WARNING, msg)
        print_warning(msg, force_color=bool(use_spinner))
    else:
        logger.warning(msg)


def _log_model_line(
    logger: logging.Logger,
    model_label: str,
    message: str,
    *,
    use_spinner: bool = False,
) -> None:
    msg = f"{str(model_label)}: {str(message)}"
    if (not _gwas_verbose_enabled(logger)) and (not use_spinner) and (not stdout_is_tty()):
        _log_file_only(logger, logging.INFO, msg)
        return
    _log_info(logger, msg, use_spinner=use_spinner)


def _rich_success(
    logger: logging.Logger,
    message: str,
    *,
    use_spinner: bool = False,
    log_message: Optional[str] = None,
) -> None:
    msg = str(message)
    file_msg = str(msg if log_message is None else log_message)
    model_line = _format_gwas_model_completion_line(msg)
    if model_line is not None and (not use_spinner) and (not stdout_is_tty()):
        logger.info(model_line)
        return
    force_color = bool(use_spinner)
    if is_skip_status_text(msg):
        if use_spinner or stdout_is_tty():
            _log_file_only(logger, logging.WARNING, file_msg)
            print_warning(msg, force_color=force_color)
        else:
            logger.warning(file_msg)
        return
    if use_spinner or stdout_is_tty():
        _log_file_only(logger, logging.INFO, file_msg)
        print_success(msg, force_color=force_color)
    else:
        logger.info(file_msg)


class _ProgressAdapter(_BaseProgressAdapter):
    """GWAS progress wrapper over the shared progress-bar adapter."""

    def __init__(
        self,
        total: int,
        desc: str,
        *,
        force_animate: bool = False,
        logger: Optional[logging.Logger] = None,
        log_unit: str = "SNP",
    ) -> None:
        self._log_progress_enabled = _gwas_progress_enabled(logger)
        self._logger = logger
        self._log_unit = str(log_unit).strip() or "item"
        self._done = 0
        self._tick = 0
        self._memory_text = ""
        self._memory_until_tick = 0
        self._log_last_cells = -1
        self._log_last_desc = str(desc)
        self._log_last_memory = ""
        self._log_last_emit_ts = 0.0
        self._log_finished = False
        super().__init__(
            total=total,
            desc=desc,
            show_spinner=True,
            show_postfix=True,
            keep_display=False,
            show_remaining=True,
            emit_done=True,
            bar_width=_GWAS_PROGRESS_BAR_WIDTH,
            force_animate=force_animate,
            logger=logger,
            log_unit=log_unit,
        )
        self._emit_log_progress(force=True)

    def _active_memory_text(self) -> str:
        if self._memory_text and self._tick <= self._memory_until_tick:
            return str(self._memory_text).strip()
        return ""

    def _emit_progress_log_line(self, line: str) -> None:
        if self._logger is None:
            return
        if stdout_is_tty():
            _log_file_only(self._logger, logging.INFO, str(line))
        else:
            self._logger.info(str(line))

    def _emit_log_progress(self, *, force: bool = False) -> None:
        if (not self._log_progress_enabled) or self._logger is None:
            return
        total = int(max(1, self.total))
        done = int(max(0, min(self._done, total)))
        frac = float(done) / float(total) if total > 0 else 0.0
        width = int(max(8, _GWAS_PROGRESS_BAR_WIDTH))
        filled = width if done >= total else int(frac * float(width))
        filled = max(0, min(width, filled))
        mem_text = self._active_memory_text()
        now = time.monotonic()
        if not force:
            if (
                filled == self._log_last_cells
                and self.desc == self._log_last_desc
                and mem_text == self._log_last_memory
                and (now - self._log_last_emit_ts) < _GWAS_LOG_PROGRESS_MIN_INTERVAL_SEC
            ):
                return
        elif (
            done == total
            and self._log_finished
            and filled == self._log_last_cells
            and self.desc == self._log_last_desc
            and mem_text == self._log_last_memory
        ):
            return
        bar = "*" * filled
        unit = str(self._log_unit).strip() or "item"
        if done == total and total > 0:
            self._log_finished = True
        elapsed = format_elapsed(now - self._start_ts)
        details = [f"{done:,}/{total:,} {unit}", elapsed]
        if mem_text != "":
            details.append(f"RSS={mem_text}")
        line = f"{self.desc} progress [{bar}] [{', '.join(details)}]"
        self._emit_progress_log_line(line)
        self._log_last_cells = filled
        self._log_last_desc = self.desc
        self._log_last_memory = mem_text
        self._log_last_emit_ts = now

    def update(self, step: int = 1) -> None:
        if self._finished:
            return
        step = int(max(0, step))
        if step == 0:
            return
        super().update(step)
        self._done = int(min(self.total, self._done + step))
        self._tick += 1
        self._emit_log_progress(force=False)

    def set_postfix(self, *, memory: Optional[str] = None) -> None:
        if memory is not None:
            self._memory_text = str(memory).replace(" ", "")
            self._memory_until_tick = self._tick + 5
        mem = self._active_memory_text()
        super().set_postfix(memory=mem) if mem != "" else super().set_postfix()
        self._emit_log_progress(force=False)

    def set_description(self, desc: str) -> None:
        super().set_description(desc)
        self._emit_log_progress(force=True)

    def set_total(self, total: int) -> None:
        super().set_total(total)
        self._done = int(min(self._done, self.total))
        self._emit_log_progress(force=True)

    def finish(self, *, message=None, elapsed_parts=None) -> None:
        self._done = self.total
        super().finish(message=message, elapsed_parts=elapsed_parts)
        self._emit_log_progress(force=True)

    def close(self, *, show_done: bool = True, message=None, elapsed_parts=None) -> None:
        if show_done and self._finished and (not self._log_finished):
            self._emit_log_progress(force=True)
        super().close(show_done=show_done, message=message, elapsed_parts=elapsed_parts)


def _start_indeterminate_progress_bar(
    desc: str,
    *,
    enabled: bool = True,
    timeout: float = 0.08,
    use_process: bool = False,
) -> Optional[SpinnerStatusAdapter]:
    handle = SpinnerStatusAdapter(
        str(desc),
        enabled=bool(enabled),
        timeout=float(timeout),
        use_process=bool(use_process),
    )
    return handle.start()


def _stop_indeterminate_progress_bar(
    handle: Optional[SpinnerStatusAdapter],
    *,
    show_done: bool = False,
    message: Optional[str] = None,
    elapsed_parts: Optional[list[object]] = None,
) -> None:
    if handle is None:
        return
    handle.close(show_done=show_done, message=message, elapsed_parts=elapsed_parts)


def fastplot(
    gwasresult: pd.DataFrame,
    phenosub: np.ndarray,
    xlabel: str = "",
    outpdf: str = "fastplot.pdf",
    threshold_n_tests: Optional[int] = None,
    plot_style: str = "gwas",
) -> None:
    """
    Generate diagnostic plots for GWAS results: phenotype histogram, Manhattan, and QQ.
    """
    mpl.rcParams["font.size"] = 12
    results = gwasresult
    plot_style_norm = _normalize_fastplot_style(plot_style)
    if "pos" in results.columns and not pd.api.types.is_integer_dtype(results["pos"]):
        results = results.copy()
        results["pos"] = pd.to_numeric(results["pos"], errors="coerce").fillna(0).astype(np.int64)
    fig = plt.figure(figsize=_FASTPLOT_FIGSIZE, dpi=_FASTPLOT_DPI)
    try:
        layout = [["A", "B", "B", "C"]]
        axes: dict[str, plt.Axes] = fig.subplot_mosaic(mosaic=layout)

        from janusx.bioplotkit import GWASPLOT, apply_integer_yticks

        gwasplot = GWASPLOT(results, interval_rate=_FASTPLOT_MANHATTAN_INTERVAL_RATE)
        qq_results = (
            _prepare_fastplot_qq_results(results)
            if plot_style_norm == "garfield"
            else results
        )
        gwasplot_qq = GWASPLOT(
            qq_results,
            interval_rate=_FASTPLOT_MANHATTAN_INTERVAL_RATE,
            compression=False,
        )
        scatter_size = _FASTPLOT_SCATTER_SIZE

        # A: phenotype distribution
        pheno = np.asarray(phenosub, dtype="float64").reshape(-1)
        pheno = pheno[np.isfinite(pheno)]
        n_samples = int(pheno.size)
        label_base = str(xlabel).strip() if str(xlabel).strip() else "phenotype"

        if n_samples > 0:
            counts, edges, _ = axes["A"].hist(
                pheno,
                bins=_FASTPLOT_HIST_BINS,
                color="black",
                edgecolor="none",
                alpha=1.0,
            )
            kde_curve = _histogram_kde_count_curve(pheno, edges=edges)
            if counts.size > 1 and kde_curve is not None:
                x_grid, y_count = kde_curve
                axes["A"].plot(x_grid, y_count, color="#B3B3B3", linewidth=1.6)
        else:
            axes["A"].text(
                0.5,
                0.5,
                "No valid phenotype values",
                ha="center",
                va="center",
                transform=axes["A"].transAxes,
            )

        x_label_text = f"{label_base} (n={n_samples})"
        if _contains_cjk(x_label_text) and not _ensure_cjk_font():
            # No CJK font found on this host; fallback to ASCII to avoid glyph warnings.
            x_label_text = f"phenotype (n={n_samples})"
        axes["A"].set_xlabel(x_label_text)
        axes["A"].set_ylabel("Count")

        # B: Manhattan plot
        if threshold_n_tests is None:
            threshold_base_n = int(results.shape[0])
        else:
            try:
                threshold_base_n = int(threshold_n_tests)
            except Exception:
                threshold_base_n = int(results.shape[0])
        threshold = -np.log10(1 / max(1, threshold_base_n))
        drew_manhattan = False
        if plot_style_norm == "garfield":
            drew_manhattan = _draw_role_layered_manhattan(
                results,
                gwasplot,
                ax=axes["B"],
                threshold=threshold,
                scatter_size=scatter_size,
            )
        if not drew_manhattan:
            gwasplot.manhattan(
                threshold,
                ax=axes["B"],
                rasterized=True,
                s=scatter_size,
            )
        manh_yticks = apply_integer_yticks(axes["B"])
        snp_n = int(results.shape[0])
        if snp_n >= 1_000_000:
            snp_val = snp_n / 1_000_000.0
            snp_suffix = "M"
        else:
            snp_val = snp_n / 1_000.0
            snp_suffix = "K"
        snp_text = f"{snp_val:.3f}".rstrip("0").rstrip(".")
        axes["B"].set_xlabel(f"Chromosome (SNP={snp_text}{snp_suffix})")

        # C: QQ plot
        manh_ymin, manh_ymax = axes["B"].get_ylim()
        gwasplot_qq.qq(
            ax=axes["C"],
            scatter_size=scatter_size,
            axis_min=0.0,
        )

        # Align QQ with Manhattan:
        # - QQ shares the Manhattan upper y bound for visual comparability
        # - but both QQ axes still start from 0 to avoid artificial upward shifts
        qq_xmin, qq_xmax = axes["C"].get_xlim()
        qq_left = 0.0
        qq_bottom = 0.0
        if np.isfinite(manh_ymax) and float(manh_ymax) > qq_bottom:
            axes["C"].set_ylim(qq_bottom, float(manh_ymax))
        if np.isfinite(qq_xmin) and np.isfinite(qq_xmax) and qq_xmax > qq_xmin:
            x_right = max(float(qq_xmax), qq_left + 1e-9)
            x_span = max(1e-9, x_right - qq_left)
            x_pad = max(1e-9, 0.02 * x_span)
            x_upper = x_right + x_pad
            if x_upper <= qq_left:
                x_upper = qq_left + 1.0
            axes["C"].set_xlim(qq_left, x_upper)
        else:
            axes["C"].set_xlim(qq_left, qq_left + 1.0)
        apply_integer_yticks(axes["C"], ticks=manh_yticks)

        fig.tight_layout()
        fig.savefig(outpdf, transparent=False, facecolor="white")
    finally:
        plt.close(fig)


def _run_fastplot_with_status(
    gwasresult: pd.DataFrame,
    phenosub: np.ndarray,
    *,
    xlabel: str,
    outpdf: str,
    threshold_n_tests: Optional[int] = None,
    plot_style: str = "gwas",
    use_spinner: bool = False,
    emit_done_line: bool = False,
) -> float:
    viz_t0 = time.time()
    if bool(use_spinner):
        with CliStatus("Visualizing ...", enabled=True, use_process=False) as task:
            try:
                with runtime_thread_stage(blas_threads=1, rayon_threads=1):
                    fastplot(
                        gwasresult,
                        phenosub,
                        xlabel=str(xlabel),
                        outpdf=str(outpdf),
                        threshold_n_tests=threshold_n_tests,
                        plot_style=str(plot_style),
                    )
            except Exception:
                task.fail("Visualizing ...Failed")
                raise
            if bool(emit_done_line):
                task.complete("Visualizing ...Finished")
    else:
        with runtime_thread_stage(blas_threads=1, rayon_threads=1):
            fastplot(
                gwasresult,
                phenosub,
                xlabel=str(xlabel),
                outpdf=str(outpdf),
                threshold_n_tests=threshold_n_tests,
                plot_style=str(plot_style),
            )
    return max(time.time() - viz_t0, 0.0)


def _run_fastplot_from_tsv_with_status(
    out_tsv: str,
    phenosub: np.ndarray,
    *,
    xlabel: str,
    outpdf: str,
    threshold_n_tests: Optional[int] = None,
    plot_style: str = "gwas",
    use_spinner: bool = False,
    emit_done_line: bool = False,
) -> float:
    viz_t0 = time.time()
    plot_style_norm = _normalize_fastplot_style(plot_style)
    header_df = pd.read_csv(out_tsv, sep="\t", nrows=0)
    usecols = ["chrom", "pos", "pwald"]
    if plot_style_norm == "garfield":
        if "row_role" in header_df.columns:
            usecols.append("row_role")
        if "snp" in header_df.columns:
            usecols.append("snp")
        for fdr_col in _FASTPLOT_COMBO_FDR_COLUMNS:
            if fdr_col in header_df.columns:
                usecols.append(str(fdr_col))
    dtype_map: dict[str, object] = {"chrom": str, "pos": "int64"}
    if "row_role" in usecols:
        dtype_map["row_role"] = str
    if "snp" in usecols:
        dtype_map["snp"] = str
    if bool(use_spinner):
        with CliStatus("Visualizing ...", enabled=True, use_process=False) as task:
            try:
                plot_df = pd.read_csv(
                    out_tsv,
                    sep="\t",
                    usecols=usecols,
                    dtype=dtype_map,
                )
                plot_df["pwald"] = pd.to_numeric(plot_df["pwald"], errors="coerce")
                if plot_style_norm == "garfield":
                    for fdr_col in _FASTPLOT_COMBO_FDR_COLUMNS:
                        if fdr_col in plot_df.columns:
                            plot_df[fdr_col] = pd.to_numeric(plot_df[fdr_col], errors="coerce")
                with runtime_thread_stage(blas_threads=1, rayon_threads=1):
                    fastplot(
                        plot_df,
                        phenosub,
                        xlabel=str(xlabel),
                        outpdf=str(outpdf),
                        threshold_n_tests=threshold_n_tests,
                        plot_style=plot_style_norm,
                    )
            except Exception:
                task.fail("Visualizing ...Failed")
                raise
            if bool(emit_done_line):
                task.complete("Visualizing ...Finished")
    else:
        plot_df = pd.read_csv(
            out_tsv,
            sep="\t",
            usecols=usecols,
            dtype=dtype_map,
        )
        plot_df["pwald"] = pd.to_numeric(plot_df["pwald"], errors="coerce")
        if plot_style_norm == "garfield":
            for fdr_col in _FASTPLOT_COMBO_FDR_COLUMNS:
                if fdr_col in plot_df.columns:
                    plot_df[fdr_col] = pd.to_numeric(plot_df[fdr_col], errors="coerce")
        with runtime_thread_stage(blas_threads=1, rayon_threads=1):
            fastplot(
                plot_df,
                phenosub,
                xlabel=str(xlabel),
                outpdf=str(outpdf),
                threshold_n_tests=threshold_n_tests,
                plot_style=plot_style_norm,
            )
    return max(time.time() - viz_t0, 0.0)


def _run_result_write_with_status(
    write_fn,
    *,
    use_spinner: bool = False,
    emit_done_line: bool = False,
) -> float:
    w_t0 = time.time()
    if bool(use_spinner):
        with CliStatus("Writing results ...", enabled=True, use_process=False) as task:
            try:
                write_fn()
            except Exception:
                task.fail("Writing results ...Failed")
                raise
            if bool(emit_done_line):
                task.complete("Writing results ...Finished")
    else:
        write_fn()
    return max(time.time() - w_t0, 0.0)
