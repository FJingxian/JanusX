# -*- coding: utf-8 -*-
"""UI/status helpers for GWAS workflow output and plotting."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from janusx.script._common.cjk import contains_cjk as _contains_cjk, ensure_cjk_font as _ensure_cjk_font
from janusx.script._common.progress import build_rich_progress, rich_progress_available
from janusx.script._common.status import (
    CliStatus,
    format_elapsed,
    is_skip_status_text,
    print_success,
    print_warning,
    should_animate_status,
    stdout_is_tty,
)

try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore[assignment]
    _HAS_TQDM = False


_GWAS_PROGRESS_BAR_WIDTH = 30
_GWAS_LOG_PROGRESS_MIN_INTERVAL_SEC = 2.0
_GWAS_MODEL_LINE_WIDTH = 80
_GWAS_MODEL_DONE_RE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z0-9]*(?:\([^)]*\))?)\s+\.\.\.(?P<body>.*?)(?:\s+\[(?P<times>[^\]]+)\])?$"
)
_GWAS_MODEL_LABELS = {
    "LM",
    "LMM",
    "FastLMM",
    "FvLMM",
    "SparseLMM",
    "ALGWAS",
    "FarmCPU",
}


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
    if (not _gwas_verbose_enabled(logger)) and (not use_spinner) and (not stdout_is_tty()):
        return
    _log_info(logger, f"{str(model_label)}: {str(message)}", use_spinner=use_spinner)


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


class _ProgressAdapter:
    """
    Progress bar adapter with rich-first rendering and tqdm fallback.
    """

    def __init__(
        self,
        total: int,
        desc: str,
        *,
        force_animate: bool = False,
        logger: Optional[logging.Logger] = None,
        log_unit: str = "SNP",
    ) -> None:
        self.total = int(max(0, total))
        self.desc = str(desc)
        self._log_progress_enabled = _gwas_progress_enabled(logger)
        self._animate = bool((force_animate or should_animate_status(self.desc)) and stdout_is_tty())
        self._logger = logger
        self._log_unit = str(log_unit).strip() or "item"
        self._backend = "none"
        self._progress = None
        self._task_id = None
        self._tqdm = None
        self._start_ts = time.monotonic()
        self._finished = False
        self._done = 0
        self._tick = 0
        self._memory_text = ""
        self._memory_until_tick = 0
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        self._log_last_cells = -1
        self._log_last_desc = self.desc
        self._log_last_memory = ""
        self._log_last_emit_ts = 0.0
        self._log_finished = False

        if self._animate and rich_progress_available():
            try:
                self._progress = build_rich_progress(
                    description_template="[green]{task.description:<8}",
                    show_remaining=False,
                    show_percentage=False,
                    field_templates=["{task.fields[metric]}"],
                    bar_width=_GWAS_PROGRESS_BAR_WIDTH,
                    finished_text=" ",
                    transient=True,
                )
                if self._progress is None:
                    raise RuntimeError("rich progress unavailable")
                self._progress.start()
                self._task_id = self._progress.add_task(
                    self.desc,
                    total=self.total,
                    metric=_format_progress_metric(0, self.total, None),
                )
                self._backend = "rich"
            except Exception:
                self._progress = None
                self._task_id = None

        if self._animate and self._backend == "none" and _HAS_TQDM and stdout_is_tty():
            self._tqdm = tqdm(
                total=self.total,
                desc=self.desc,
                leave=False,
                dynamic_ncols=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            )
            self._backend = "tqdm"

        if self._animate and self._backend in {"rich", "tqdm"}:
            self._hb_thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._hb_thread.start()
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

    def _metric_text(self) -> str:
        mem = self._active_memory_text()
        return _format_progress_metric(self._done, self.total, mem if mem else None)

    def _refresh_metric(self) -> None:
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, metric=self._metric_text())
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.set_postfix_str(self._metric_text())

    def _heartbeat(self) -> None:
        while not self._hb_stop.wait(0.25):
            try:
                if self._backend == "rich" and self._progress is not None and self._task_id is not None:
                    self._progress.update(self._task_id, metric=self._metric_text())
                elif self._backend == "tqdm" and self._tqdm is not None:
                    self._tqdm.refresh()
            except Exception:
                continue

    def update(self, step: int = 1) -> None:
        if self._finished:
            return
        step = int(max(0, step))
        if step == 0:
            return
        self._done = int(min(self.total, self._done + step))
        self._tick += 1
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=step, metric=self._metric_text())
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.update(step)
            self._tqdm.set_postfix_str(self._metric_text())
        self._emit_log_progress(force=False)

    def set_postfix(self, *, memory: Optional[str] = None) -> None:
        if memory is not None:
            self._memory_text = str(memory).replace(" ", "")
            self._memory_until_tick = self._tick + 5
        self._refresh_metric()
        self._emit_log_progress(force=False)

    def set_description(self, desc: str) -> None:
        self.desc = str(desc)
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=self.desc, metric=self._metric_text())
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.set_description_str(self.desc)
        self._emit_log_progress(force=True)

    def set_total(self, total: int) -> None:
        self.total = int(max(0, total))
        self._done = int(min(self._done, self.total))
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, total=self.total, metric=self._metric_text())
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.total = self.total
            self._tqdm.refresh()
        self._emit_log_progress(force=True)

    def finish(self) -> None:
        if self._finished:
            return
        self._done = self.total
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=self.total, metric=self._metric_text())
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.n = self._tqdm.total
            self._tqdm.set_postfix_str(self._metric_text())
            self._tqdm.refresh()
        self._finished = True
        self._emit_log_progress(force=True)

    def close(self, *, show_done: bool = True) -> None:
        if self._hb_thread is not None:
            self._hb_stop.set()
            try:
                self._hb_thread.join(timeout=0.2)
            except Exception:
                pass
            self._hb_thread = None
        if self._backend == "rich" and self._progress is not None:
            try:
                if show_done and self._task_id is not None:
                    self._progress.update(self._task_id, metric=self._metric_text())
            finally:
                self._progress.stop()
                self._progress = None
                self._task_id = None
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None
        if show_done and self._finished and (not self._log_finished):
            self._emit_log_progress(force=True)
        self._backend = "none"
        self._finished = True


def _start_indeterminate_progress_bar(
    desc: str,
    *,
    total: int = 100,
    tick_interval: float = 0.08,
) -> Optional[tuple[_ProgressAdapter, threading.Event, threading.Thread]]:
    pbar = _ProgressAdapter(total=total, desc=str(desc), force_animate=True)
    if getattr(pbar, "_backend", "none") == "none":
        return None
    stop_evt = threading.Event()

    def _runner() -> None:
        while not stop_evt.wait(max(0.01, float(tick_interval))):
            try:
                pbar.update(1)
            except Exception:
                return

    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    return (pbar, stop_evt, th)


def _stop_indeterminate_progress_bar(
    handle: Optional[tuple[_ProgressAdapter, threading.Event, threading.Thread]],
) -> None:
    if handle is None:
        return
    pbar, stop_evt, th = handle
    stop_evt.set()
    try:
        th.join(timeout=0.2)
    except Exception:
        pass
    try:
        pbar.finish()
    except Exception:
        pass
    try:
        pbar.close(show_done=False)
    except Exception:
        pass


def fastplot(
    gwasresult: pd.DataFrame,
    phenosub: np.ndarray,
    xlabel: str = "",
    outpdf: str = "fastplot.pdf",
) -> None:
    """
    Generate diagnostic plots for GWAS results: phenotype histogram, Manhattan, and QQ.
    """
    mpl.rcParams["font.size"] = 12
    results = gwasresult.astype({"pos": "int64"})
    fig = plt.figure(figsize=(16, 4), dpi=300)
    try:
        layout = [["A", "B", "B", "C"]]
        axes: dict[str, plt.Axes] = fig.subplot_mosaic(mosaic=layout)

        from janusx.bioplotkit import GWASPLOT, apply_integer_yticks

        gwasplot = GWASPLOT(results)
        scatter_size = 8.0

        # A: phenotype distribution
        pheno = np.asarray(phenosub, dtype="float64").reshape(-1)
        pheno = pheno[np.isfinite(pheno)]
        n_samples = int(pheno.size)
        label_base = str(xlabel).strip() if str(xlabel).strip() else "phenotype"

        if n_samples > 0:
            counts, edges, _ = axes["A"].hist(
                pheno,
                bins=15,
                color="black",
                edgecolor="none",
                alpha=1.0,
            )
            # Overlay seaborn-like KDE curve, scaled to histogram "Count" axis.
            if counts.size > 1 and np.unique(pheno).size > 1:
                try:
                    from scipy.stats import gaussian_kde

                    kde = gaussian_kde(pheno)
                    x_grid = np.linspace(float(np.min(pheno)), float(np.max(pheno)), 256)
                    y_density = kde(x_grid)
                    bin_width = float(np.mean(np.diff(edges)))
                    y_count = y_density * float(n_samples) * bin_width
                    axes["A"].plot(x_grid, y_count, color="#B3B3B3", linewidth=1.6)
                except Exception:
                    pass
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
            -np.log10(1 / results.shape[0]),
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
            axis_min=float(manh_ymin),
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
    use_spinner: bool = False,
    emit_done_line: bool = False,
) -> float:
    viz_t0 = time.time()
    if bool(use_spinner):
        with CliStatus("Visualizing ...", enabled=True, use_process=False) as task:
            try:
                fastplot(gwasresult, phenosub, xlabel=str(xlabel), outpdf=str(outpdf))
            except Exception:
                task.fail("Visualizing ...Failed")
                raise
            if bool(emit_done_line):
                task.complete("Visualizing ...Finished")
    else:
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
