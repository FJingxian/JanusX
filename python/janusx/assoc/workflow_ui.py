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
_FASTPLOT_FIGSIZE = (14.0, 3.4)
_FASTPLOT_DPI = 180
_FASTPLOT_SCATTER_SIZE = 5.0
_FASTPLOT_HIST_BINS = 12
_FASTPLOT_MANH_TARGET_POINTS = 60_000
_FASTPLOT_QQ_FAST_MAX_POINTS = 1_024
_FASTPLOT_QQ_BAND_MAX_POINTS = 256


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


def _fastplot_sanitize_pvalues(values) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce")
    if isinstance(arr, pd.Series):
        out = arr.to_numpy(dtype=np.float64, copy=False)
    else:
        out = np.asarray(arr, dtype=np.float64)
    out = np.array(out, dtype=np.float64, copy=True).reshape(-1)
    out[~np.isfinite(out)] = 1.0
    return np.clip(out, np.nextafter(0.0, 1.0), 1.0)


def _fastplot_downsample_results(
    gwasresult: pd.DataFrame,
    *,
    p_col: str = "pwald",
    target_points: int = _FASTPLOT_MANH_TARGET_POINTS,
) -> pd.DataFrame:
    n = int(gwasresult.shape[0])
    target = int(max(1, target_points))
    if n <= target:
        return gwasresult

    pvals = _fastplot_sanitize_pvalues(gwasresult[p_col])
    sig_thresh = 10_000.0 / float(max(1, n))
    sig_idx = np.flatnonzero(pvals <= sig_thresh)
    nonsig_idx = np.flatnonzero(pvals > sig_thresh)

    if sig_idx.size >= target or nonsig_idx.size == 0:
        keep_idx = sig_idx if sig_idx.size > 0 else np.arange(0, n, max(1, n // target), dtype=np.int64)
        return gwasresult.iloc[np.sort(np.unique(keep_idx[:target]))]

    nonsig_budget = max(1, target - int(sig_idx.size))
    step = max(1, int(np.ceil(float(nonsig_idx.size) / float(nonsig_budget))))
    sample_idx = nonsig_idx[::step]
    if sample_idx.size == 0 or sample_idx[-1] != nonsig_idx[-1]:
        sample_idx = np.append(sample_idx, nonsig_idx[-1])
    if sample_idx.size > nonsig_budget:
        step2 = max(1, int(np.ceil(float(sample_idx.size) / float(nonsig_budget))))
        sample_idx = sample_idx[::step2]
        if sample_idx[-1] != nonsig_idx[-1]:
            sample_idx = np.append(sample_idx, nonsig_idx[-1])

    keep_idx = np.sort(np.unique(np.concatenate([sig_idx, sample_idx]).astype(np.int64, copy=False)))
    return gwasresult.iloc[keep_idx]


def _fastplot_draw_qq(
    ax: plt.Axes,
    pvalues,
    *,
    scatter_size: float,
    axis_min: float,
    qq_fast_max_points: int = _FASTPLOT_QQ_FAST_MAX_POINTS,
    qq_band_max_points: int = _FASTPLOT_QQ_BAND_MAX_POINTS,
) -> None:
    from janusx.bioplotkit.manhanden import (
        _marker_scatter_style,
        _qq_confidence_band_logp,
        _qq_sample_draw_indices,
    )

    p = _fastplot_sanitize_pvalues(pvalues)
    n = int(p.size)
    if n <= 0:
        raise ValueError("No p-values found for QQ plot.")

    if np.isfinite(float(axis_min)) and float(axis_min) > 0.0:
        visible_rank_cap = int(np.floor((n + 1.0) * (10.0 ** (-float(axis_min)))))
        visible_rank_cap = max(1, min(n, visible_rank_cap))
    else:
        visible_rank_cap = n

    draw_idx = _qq_sample_draw_indices(
        visible_rank_cap,
        max_points=int(max(1, qq_fast_max_points)),
    )
    draw_idx = np.asarray(draw_idx, dtype=np.int64).reshape(-1)
    draw_idx_asc = np.sort(draw_idx)
    if draw_idx_asc.size == visible_rank_cap:
        p_ordered = np.sort(p, kind="mergesort")
    else:
        p_ordered = np.array(p, dtype=np.float64, copy=True)
        p_ordered.partition(draw_idx_asc)
    p_draw = p_ordered[draw_idx]
    ranks_draw = draw_idx.astype(np.float64) + 1.0
    obs_scatter = -np.log10(p_draw)
    exp_scatter = -np.log10(ranks_draw / (n + 1.0))

    x_band, lower, upper = _qq_confidence_band_logp(
        n,
        ci=95,
        max_points=int(max(1, qq_band_max_points)),
        rank_max=visible_rank_cap,
    )
    band_mask = np.isfinite(x_band) & np.isfinite(lower) & np.isfinite(upper)
    if np.any(band_mask):
        ax.fill_between(
            x_band[band_mask],
            lower[band_mask],
            upper[band_mask],
            color="grey",
            alpha=0.3,
            rasterized=True,
        )

    finite_obs = obs_scatter[np.isfinite(obs_scatter)]
    finite_exp = exp_scatter[np.isfinite(exp_scatter)]
    if finite_obs.size == 0 or finite_exp.size == 0:
        max_lim = 1.0
    else:
        max_lim = float(min(np.max(finite_obs), np.max(finite_exp)))
        if not np.isfinite(max_lim) or max_lim <= 0.0:
            max_lim = 1.0
    ax.plot([0.0, max_lim], [0.0, max_lim], lw=1, color="black")

    scatter_mask = np.isfinite(exp_scatter) & np.isfinite(obs_scatter)
    ax.scatter(
        exp_scatter[scatter_mask],
        obs_scatter[scatter_mask],
        marker="o",
        s=float(scatter_size),
        alpha=0.75,
        rasterized=True,
        color="black",
        **_marker_scatter_style("o"),
    )

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_data_min = 0.0
    x_data_max = float(max_lim)
    if scatter_mask.any():
        exp_valid = exp_scatter[scatter_mask]
        exp_valid = exp_valid[np.isfinite(exp_valid)]
        if exp_valid.size > 0:
            x_data_min = min(x_data_min, float(np.min(exp_valid)))
            x_data_max = max(x_data_max, float(np.max(exp_valid)))
    if np.any(band_mask):
        x_band_valid = x_band[band_mask]
        x_band_valid = x_band_valid[np.isfinite(x_band_valid)]
        if x_band_valid.size > 0:
            x_data_min = min(x_data_min, float(np.min(x_band_valid)))
            x_data_max = max(x_data_max, float(np.max(x_band_valid)))
    if np.isfinite([x0, x1, y0, y1]).all():
        lo = float(axis_min) if np.isfinite(float(axis_min)) else x_data_min
        x_hi = float(x1) if float(x1) > lo else (lo + 1.0)
        y_hi = float(y1) if float(y1) > lo else (lo + 1.0)
        ax.set_xlim(lo, x_hi)
        ax.set_ylim(lo, y_hi)

    ax.set_xlabel("Expected -log10(p-value)")
    ax.set_ylabel("Observed -log10(p-value)")


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
) -> None:
    """
    Generate diagnostic plots for GWAS results: phenotype histogram, Manhattan, and QQ.
    """
    mpl.rcParams["font.size"] = 11
    results = gwasresult
    if "pos" in results.columns and not pd.api.types.is_integer_dtype(results["pos"]):
        results = results.copy()
        results["pos"] = pd.to_numeric(results["pos"], errors="coerce").fillna(0).astype(np.int64)
    plot_results = _fastplot_downsample_results(results)
    fig = plt.figure(figsize=_FASTPLOT_FIGSIZE, dpi=_FASTPLOT_DPI)
    try:
        layout = [["A", "B", "B", "C"]]
        axes: dict[str, plt.Axes] = fig.subplot_mosaic(mosaic=layout)

        from janusx.bioplotkit import GWASPLOT, apply_integer_yticks

        gwasplot = GWASPLOT(plot_results)
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
        gwasplot.manhattan(
            -np.log10(1 / max(1, results.shape[0])),
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
        _fastplot_draw_qq(
            ax=axes["C"],
            pvalues=results["pwald"],
            scatter_size=scatter_size,
            axis_min=float(manh_ymin),
            qq_fast_max_points=_FASTPLOT_QQ_FAST_MAX_POINTS,
            qq_band_max_points=_FASTPLOT_QQ_BAND_MAX_POINTS,
        )

        # Align QQ with Manhattan:
        # - QQ ylim follows Manhattan ylim
        # - QQ xlim minimum equals Manhattan ylim minimum
        qq_xmin, qq_xmax = axes["C"].get_xlim()
        axes["C"].set_ylim(manh_ymin, manh_ymax)
        qq_left = float(manh_ymin)
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

        fig.subplots_adjust(
            left=0.055,
            right=0.995,
            bottom=0.23,
            top=0.94,
            wspace=0.26,
        )
        fig.savefig(outpdf, transparent=False, facecolor="white")
    finally:
        plt.close(fig)


def _run_fastplot_with_status(
    gwasresult: pd.DataFrame,
    phenosub: np.ndarray,
    *,
    xlabel: str,
    outpdf: str,
    use_spinner: bool = False,
    emit_done_line: bool = False,
) -> float:
    viz_t0 = time.time()
    if bool(use_spinner):
        with CliStatus("Visualizing ...", enabled=True, use_process=False) as task:
            try:
                with runtime_thread_stage(blas_threads=1, rayon_threads=1):
                    fastplot(gwasresult, phenosub, xlabel=str(xlabel), outpdf=str(outpdf))
            except Exception:
                task.fail("Visualizing ...Failed")
                raise
            if bool(emit_done_line):
                task.complete("Visualizing ...Finished")
    else:
        with runtime_thread_stage(blas_threads=1, rayon_threads=1):
            fastplot(gwasresult, phenosub, xlabel=str(xlabel), outpdf=str(outpdf))
    return max(time.time() - viz_t0, 0.0)


def _run_fastplot_from_tsv_with_status(
    out_tsv: str,
    phenosub: np.ndarray,
    *,
    xlabel: str,
    outpdf: str,
    use_spinner: bool = False,
    emit_done_line: bool = False,
) -> float:
    viz_t0 = time.time()
    if bool(use_spinner):
        with CliStatus("Visualizing ...", enabled=True, use_process=False) as task:
            try:
                plot_df = pd.read_csv(
                    out_tsv,
                    sep="\t",
                    usecols=["chrom", "pos", "pwald"],
                    dtype={"chrom": str, "pos": "int64"},
                )
                plot_df["pwald"] = pd.to_numeric(plot_df["pwald"], errors="coerce")
                with runtime_thread_stage(blas_threads=1, rayon_threads=1):
                    fastplot(plot_df, phenosub, xlabel=str(xlabel), outpdf=str(outpdf))
            except Exception:
                task.fail("Visualizing ...Failed")
                raise
            if bool(emit_done_line):
                task.complete("Visualizing ...Finished")
    else:
        plot_df = pd.read_csv(
            out_tsv,
            sep="\t",
            usecols=["chrom", "pos", "pwald"],
            dtype={"chrom": str, "pos": "int64"},
        )
        plot_df["pwald"] = pd.to_numeric(plot_df["pwald"], errors="coerce")
        with runtime_thread_stage(blas_threads=1, rayon_threads=1):
            fastplot(plot_df, phenosub, xlabel=str(xlabel), outpdf=str(outpdf))
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
