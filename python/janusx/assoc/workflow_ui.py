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

from janusx.gtools.cleaner import chrom_sort_key
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
_GWAS_LOG_PROGRESS_MIN_INTERVAL_SEC = 10.0
_GWAS_MODEL_LINE_WIDTH = 80
_GWAS_MODEL_DONE_RE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z0-9]*(?:\([^)]*\))?)\s+\.\.\.(?P<body>.*?)(?:\s+\[(?P<times>[^\]]+)\])?$"
)
_GWAS_MODEL_LABELS = {
    "LM",
    "LMM",
    "LMM2",
    "FvLMM",
    "SparseLMM",
    "ALGWAS",
    "FarmCPU",
}
_FASTPLOT_FIGSIZE = (16.0, 4.0)
# The GWAS fastplot Manhattan panel occupies two of the four mosaic columns
# in a 16x4 inch canvas.  A standalone population plot therefore uses an
# 8x4 inch canvas so its physical plotting area has the same aspect ratio.
_FASTPLOT_MANHATTAN_FIGSIZE = (8.0, 4.0)
_FASTPLOT_DPI = 300
_FASTPLOT_SCATTER_SIZE = 8.0
_FASTPLOT_HIST_BINS = 15
_FASTPLOT_MANHATTAN_INTERVAL_RATE = 0.5


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


def _normalize_fastplot_style(style: object) -> str:
    text = str(style or "").strip().lower()
    if text == "garfield":
        return "garfield"
    return "gwas"


def _prepare_fastplot_manhattan_results(results: pd.DataFrame) -> pd.DataFrame:
    """Put GARFIELD rows in genomic order before Manhattan compression.

    GARFIELD result tables are intentionally written in p-value order so that
    the most significant rules appear first in text output.  ``GWASPLOT``
    performs its optional down-sampling in input order, however, so passing
    that table directly makes each compression block contain similarly ranked
    p-values rather than neighbouring loci.  Sort a plotting-only copy by
    natural chromosome order and position; the published result table is left
    unchanged.
    """
    if not isinstance(results, pd.DataFrame):
        return results
    if results.shape[0] <= 1 or "chrom" not in results.columns or "pos" not in results.columns:
        return results

    out = results.reset_index(drop=True).copy()
    chrom_text = out["chrom"].map(str)
    labels = [str(value) for value in pd.unique(chrom_text)]
    ordered_labels = sorted(labels, key=chrom_sort_key)
    rank_by_label = {label: index for index, label in enumerate(ordered_labels)}
    out["__fastplot_chr_order"] = chrom_text.map(rank_by_label).fillna(len(ordered_labels))
    out["__fastplot_pos_order"] = pd.to_numeric(out["pos"], errors="coerce").fillna(0)
    out = out.sort_values(
        by=["__fastplot_chr_order", "__fastplot_pos_order"],
        kind="mergesort",
    )
    return out.drop(
        columns=["__fastplot_chr_order", "__fastplot_pos_order"],
        errors="ignore",
    ).reset_index(drop=True)


def _logger_flag(logger: Optional[logging.Logger], name: str, default: bool = False) -> bool:
    if logger is None:
        return bool(default)
    try:
        return bool(getattr(logger, name))
    except Exception:
        return bool(default)


def _gwas_progress_enabled(logger: Optional[logging.Logger]) -> bool:
    env_raw = str(os.environ.get("JX_GWAS_PROGRESS", "1")).strip().lower()
    env_enabled = env_raw in {"1", "true", "yes", "y", "on"}
    return bool(_logger_flag(logger, "_janusx_gwas_progress_enabled", env_enabled))


class _GwasProgressLog:
    """Plain progress snapshots shared by animated and non-interactive scans."""

    def __init__(self, logger, model: str, total: int, *, trait: str = "", unit: str = "SNP"):
        self.logger = logger
        # Keep the run's console mode even if live rendering wraps stdout.
        terminal_rich = getattr(logger, "_janusx_gwas_terminal_rich", None)
        self._file_only = stdout_is_tty() if terminal_rich is None else bool(terminal_rich)
        self.model = " ".join(str(model).split())
        self.trait = " ".join(str(trait).split())
        self.total = max(0, int(total))
        self.unit = str(unit)
        self.started = time.monotonic()
        self.last_emit = self.started
        self.finished = False
        self.update(0, force=True)

    def update(self, done: int, *, total=None, memory: str = "", force: bool = False):
        if self.finished or self.logger is None:
            return
        if total is not None:
            self.total = max(0, int(total))
        done = max(0, int(done))
        if self.total:
            done = min(done, self.total)
        complete = self.total > 0 and done == self.total
        now = time.monotonic()
        if not force and not complete:
            if not _gwas_progress_enabled(self.logger):
                return
            if now - self.last_emit < _GWAS_LOG_PROGRESS_MIN_INTERVAL_SEC:
                return
        elapsed = max(0.0, now - self.started)
        percent = 100.0 * done / self.total if self.total else 0.0
        eta = "--"
        if complete:
            eta = "0s"
        elif done > 0 and self.total > done and elapsed >= 1.0:
            eta = format_elapsed(elapsed * (self.total - done) / done)
        trait = f" [{self.trait}]" if self.trait else ""
        rss = f" RSS={memory}" if memory else ""
        line = (
            f"GWAS{trait} {self.model:<8} {percent:5.1f}%"
            f" | {self.unit}s={done:,}/{self.total:,}"
            f" | elapsed={format_elapsed(elapsed)} ETA={eta}{rss}"
        )
        # Advance the throttle before emitting so a failing handler does not
        # get retried on every native progress callback.
        self.last_emit = now
        try:
            if self._file_only:
                _log_file_only(self.logger, logging.INFO, line)
            else:
                self.logger.info(line)
        except OSError:
            # Progress is advisory; a local spool or console write failure
            # must not abort the GWAS native scan or flood callbacks.
            pass
        self.finished = complete


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
    pve_match = re.match(r"^pve(?:\(pheno\))?\s+(.+)$", body, re.IGNORECASE)
    if pve_match is not None:
        body = f"Finished; PVE={pve_match.group(1).strip()}"
    times = str(m.group("times") or "").strip().replace("/", " | ")
    prefix = f"  > {label:<10}: "
    suffix = f"[ {times} ]" if times != "" else ""
    if suffix != "":
        dots = max(2, _GWAS_MODEL_LINE_WIDTH - len(prefix) - len(body) - len(suffix) - 2)
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
        trait: str = "",
        progress_log: Optional[_GwasProgressLog] = None,
    ) -> None:
        self._done = 0
        self._tick = 0
        self._memory_text = ""
        self._memory_until_tick = 0
        self._log_finished = False
        self._progress_log = progress_log or _GwasProgressLog(
            logger, desc, total, trait=trait, unit=log_unit,
        )
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
            description_template="[green]{task.description:<10}",
        )

    def _active_memory_text(self) -> str:
        if self._memory_text and self._tick <= self._memory_until_tick:
            return str(self._memory_text).strip()
        return ""

    def _emit_log_progress(self, *, force: bool = False) -> None:
        self._progress_log.model = self.desc
        self._progress_log.update(
            self._done, total=self.total, memory=self._active_memory_text(), force=force,
        )
        self._log_finished = self._progress_log.finished

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
    plot_results = (
        _prepare_fastplot_manhattan_results(results)
        if plot_style_norm == "garfield"
        else results
    )
    fig = plt.figure(figsize=_FASTPLOT_FIGSIZE, dpi=_FASTPLOT_DPI)
    try:
        layout = [["A", "B", "B", "C"]]
        axes: dict[str, plt.Axes] = fig.subplot_mosaic(mosaic=layout)

        from janusx.bioplotkit import GWASPLOT, apply_integer_yticks

        # One plot model is sufficient for both panels.  ``GWASPLOT.qq``
        # deliberately computes its ranks from the complete ``self.df`` and
        # does not use ``self.minidx`` (the Manhattan compression index), so a
        # second ``GWASPLOT(..., compression=False)`` would retain another
        # full copy of every locus for no statistical or visual benefit.
        gwasplot = GWASPLOT(plot_results, interval_rate=_FASTPLOT_MANHATTAN_INTERVAL_RATE)
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
        gwasplot.qq(
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


def _read_fastplot_tsv(out_tsv: str) -> pd.DataFrame:
    """Read only the three columns needed by the GWAS diagnostic plot.

    GWAS result files can contain many effect/model columns and millions of
    rows.  Keep chromosome labels categorical and parse the numeric columns at
    the CSV boundary so the plot path does not retain an object-backed copy of
    the chromosome column or perform a second p-value conversion.  A fallback
    keeps the historical tolerant handling for a non-numeric p-value token.
    """
    with open(out_tsv, "r", encoding="utf-8", errors="replace") as fh:
        header_cols = fh.readline().rstrip("\r\n").split("\t")
    p_col = "pwald" if "pwald" in header_cols else ("plrt" if "plrt" in header_cols else "pwald")
    usecols = ["chrom", "pos", p_col]
    dtype_map: dict[str, object] = {
        "chrom": "category",
        "pos": "int64",
        p_col: "float64",
    }
    try:
        plot_df = pd.read_csv(
            out_tsv,
            sep="\t",
            usecols=usecols,
            dtype=dtype_map,
        )
    except (TypeError, ValueError):
        # Preserve the old permissive behavior for files containing tokens
        # such as ``NA`` or other non-numeric p-value text.
        plot_df = pd.read_csv(
            out_tsv,
            sep="\t",
            usecols=usecols,
            dtype={"chrom": "category", "pos": "int64"},
        )
        plot_df[p_col] = pd.to_numeric(plot_df[p_col], errors="coerce")
    if p_col != "pwald":
        plot_df = plot_df.rename(columns={p_col: "pwald"})
    return plot_df


def _resolve_population_plot_column(columns, requested: str) -> str:
    """Resolve a population-scan column while tolerating header case."""

    names = [str(column) for column in columns]
    wanted = str(requested)
    if wanted in names:
        return wanted
    wanted_lower = wanted.lower()
    for name in names:
        if name.lower() == wanted_lower:
            return name
    raise ValueError(
        f"population scan table is missing column {wanted!r}; "
        f"available columns: {', '.join(names)}"
    )


def _read_population_manhattan_tsv(
    out_tsv: str,
    *,
    chrom_col: str,
    start_col: str,
    end_col: str,
    score_col: str,
    group_cols: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Read the compact window columns needed by a population Manhattan plot."""

    header = pd.read_csv(out_tsv, sep="\t", nrows=0)
    resolved = {
        "chrom": _resolve_population_plot_column(header.columns, chrom_col),
        "start": _resolve_population_plot_column(header.columns, start_col),
        "end": _resolve_population_plot_column(header.columns, end_col),
        "score": _resolve_population_plot_column(header.columns, score_col),
    }
    resolved_groups = tuple(
        _resolve_population_plot_column(header.columns, column)
        for column in group_cols
    )
    usecols = list(dict.fromkeys([*resolved.values(), *resolved_groups]))
    table = pd.read_csv(out_tsv, sep="\t", usecols=usecols)
    rename_map = {value: key for key, value in resolved.items()}
    table = table.rename(columns=rename_map)

    start = pd.to_numeric(table["start"], errors="coerce")
    end = pd.to_numeric(table["end"], errors="coerce")
    score = pd.to_numeric(table["score"], errors="coerce")
    chrom = table["chrom"].astype("string")
    finite = (
        chrom.notna()
        & start.notna()
        & end.notna()
        & score.notna()
        & np.isfinite(start.to_numpy(dtype=np.float64, copy=False))
        & np.isfinite(end.to_numpy(dtype=np.float64, copy=False))
        & np.isfinite(score.to_numpy(dtype=np.float64, copy=False))
    )
    out = pd.DataFrame(
        {
            "chrom": chrom,
            # Window statistics are placed at the window midpoint.  This
            # matches the genomic placement of a locus while avoiding a
            # systematic left-edge shift for wide windows.
            "pos": np.rint((start + end) / 2.0),
            "score": score,
        }
    )
    for column in resolved_groups:
        out[column] = table[column].astype("string")
        finite &= out[column].notna().to_numpy(dtype=bool, copy=False)
    out = out.loc[np.asarray(finite, dtype=bool)].reset_index(drop=True)
    if not out.empty:
        # Pandas 3/4 deprecates the Series ``copy=`` keyword under
        # copy-on-write; assignment already materializes the compact columns.
        out["pos"] = out["pos"].astype(np.int64)
        out["score"] = out["score"].astype(np.float64)
    return out, resolved_groups


def _safe_population_plot_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = token.strip("._")
    return token or "group"


def _fast_population_manhattan(
    results: pd.DataFrame,
    *,
    outsvg: str,
    score_label: str,
    title: str = "",
) -> None:
    """Render one fast, GWAS-style Manhattan SVG for window statistics."""

    from janusx.bioplotkit import GWASPLOT, apply_integer_yticks

    plot_df = results[["chrom", "pos", "score"]].copy()
    if plot_df.empty:
        fig = plt.figure(figsize=_FASTPLOT_MANHATTAN_FIGSIZE, dpi=_FASTPLOT_DPI)
        try:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No valid windows", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Chromosome")
            ax.set_ylabel(str(score_label))
            if str(title).strip():
                ax.set_title(str(title))
            fig.tight_layout()
            fig.savefig(outsvg, format="svg", transparent=False, facecolor="white")
        finally:
            plt.close(fig)
        return

    # GWASPLOT already owns the chromosome ordering, gap calculation,
    # compression and axis styling used by GWAS fastplot.  Treat the scan
    # statistic as a precomputed -log10-like score so no p-value transform is
    # applied; the y-axis remains labelled with the actual statistic.
    gwasplot = GWASPLOT(
        plot_df,
        chr="chrom",
        pos="pos",
        pvalue="score",
        interval_rate=_FASTPLOT_MANHATTAN_INTERVAL_RATE,
        pvalue_is_log10=True,
    )
    fig = plt.figure(figsize=_FASTPLOT_MANHATTAN_FIGSIZE, dpi=_FASTPLOT_DPI)
    try:
        ax = fig.add_subplot(111)
        gwasplot.manhattan(
            None,
            ax=ax,
            color_set=["black", "grey"],
            marker="o",
            min_logp=None,
            y_label=str(score_label),
            s=_FASTPLOT_SCATTER_SIZE,
            alpha=0.78,
            rasterized=True,
        )
        finite_scores = pd.to_numeric(plot_df["score"], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=False,
        )
        finite_scores = finite_scores[np.isfinite(finite_scores)]
        if finite_scores.size > 0:
            low = float(np.min(finite_scores))
            high = float(np.max(finite_scores))
            span = max(high - low, 1e-12)
            pad = max(0.05 * span, 1e-6)
            # Keep the conventional zero baseline for non-negative FST, while
            # retaining negative XP-CLR-normalized values when they occur.
            bottom = 0.0 if low >= 0.0 else low - pad
            top = high + pad
            if top <= bottom:
                top = bottom + 1.0
            ax.set_ylim(bottom, top)
            apply_integer_yticks(ax)
        if str(title).strip():
            ax.set_title(str(title))
        fig.tight_layout()
        fig.savefig(outsvg, format="svg", transparent=False, facecolor="white")
    finally:
        plt.close(fig)


def _run_fast_manhattan_from_tsv_with_status(
    out_tsv: str,
    *,
    outsvg: str,
    score_col: str,
    score_label: str,
    chrom_col: str = "chrom",
    start_col: str = "start",
    end_col: str = "end",
    group_cols: tuple[str, ...] = (),
    use_spinner: bool = False,
) -> tuple[float, list[str]]:
    """Render population-scan Manhattan SVG(s) with the GWAS fastplot style."""

    viz_t0 = time.time()
    plot_df, resolved_groups = _read_population_manhattan_tsv(
        out_tsv,
        chrom_col=chrom_col,
        start_col=start_col,
        end_col=end_col,
        score_col=score_col,
        group_cols=group_cols,
    )
    group_columns = tuple(resolved_groups)
    base_svg = str(outsvg)
    if base_svg.lower().endswith(".svg"):
        base_stem = base_svg[:-4]
    else:
        base_stem = base_svg

    jobs: list[tuple[pd.DataFrame, str, str]] = []
    if group_columns:
        if plot_df.empty:
            jobs.append((plot_df, base_svg, ""))
        else:
            for key, group in plot_df.groupby(list(group_columns), sort=False, dropna=False):
                key_values = key if isinstance(key, tuple) else (key,)
                tokens = [_safe_population_plot_token(value) for value in key_values]
                path = f"{base_stem}.{'_vs_'.join(tokens)}.svg"
                title = " vs ".join(str(value) for value in key_values)
                jobs.append((group, path, title))
    else:
        jobs.append((plot_df, base_svg, ""))

    with CliStatus(
        "Rendering Manhattan plot...",
        enabled=True,
        use_process=bool(use_spinner),
    ) as task:
        try:
            with runtime_thread_stage(blas_threads=1, rayon_threads=1):
                for group, path, title in jobs:
                    _fast_population_manhattan(
                        group,
                        outsvg=path,
                        score_label=str(score_label),
                        title=title,
                    )
        except Exception:
            task.fail("Rendering Manhattan plot ...Failed")
            raise
        task.complete("Rendering Manhattan plot ...Finished")
    return max(time.time() - viz_t0, 0.0), [path for _group, path, _title in jobs]


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
    if bool(use_spinner):
        with CliStatus("Visualizing ...", enabled=True, use_process=False) as task:
            try:
                plot_df = _read_fastplot_tsv(out_tsv)
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
        plot_df = _read_fastplot_tsv(out_tsv)
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
