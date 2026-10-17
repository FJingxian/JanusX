"""GS progress rendering and plain snapshots, independent of model fitting."""

from __future__ import annotations

import logging
import os
import re
import time
from functools import wraps

from janusx.script._common.progress import (
    ProgressAdapter,
    _emit_to_file_handlers,
    format_elapsed,
    stdout_is_tty,
    log_success as _base_log_success,
    print_success as _base_print_success,
)


def format_completion(message: str) -> str:
    match = re.match(r"^(.*?)\s+\.\.\.(Finished|Skipped)(?:\s+(\(.*\)))?(?:\s+\[([^]]+)\])?$", str(message))
    if match is None:
        return str(message)
    label, status, details, elapsed = match.groups()
    label = {"Fitting": "Fit", "Predicting": "Predict", "Cross-validation": "CV"}.get(label, label)
    prefix = f"  > {label:<10}: {status}" + (f" {details}" if details else "") + " "
    suffix = f"[ {elapsed} ]" if elapsed else ""
    return f"{prefix}{'.' * max(2, 80 - len(prefix) - len(suffix) - 1)} {suffix}".rstrip()


def print_success(message: str, *, force_color=False):
    rendered = str(message) if stdout_is_tty() else format_completion(message)
    _base_print_success(rendered, force_color=bool(force_color and stdout_is_tty()))


def log_success(logger, message: str, *, log_message=None, force_color=False):
    terminal_message = str(message) if stdout_is_tty() else format_completion(message)
    raw_log_message = message if log_message is None else log_message
    _base_log_success(
        logger,
        terminal_message,
        log_message=format_completion(raw_log_message),
        force_color=bool(force_color and stdout_is_tty()),
    )


class GSProgressAdapter(ProgressAdapter):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("description_template", "[green]{task.description:<10}")
        kwargs.setdefault("bar_width", 30)
        super().__init__(*args, **kwargs)


class GSProgressLog:
    """Record real stage/fold events in both interactive and batch runs.

    Updates are throttled at callback boundaries; start/end/failure events are
    always recorded. Unknown or convergence-limited totals have no ETA.
    """

    interval = 10.0

    def __init__(self, logger, model: str, trait: str, folds: int, *, start_method=True):
        self.logger = logger
        self.model = " ".join(str(model).split())
        self.trait = " ".join(str(trait).split())
        self.folds = max(0, int(folds))
        self.cv_done = 0
        self.search_done = 0
        self.search_total = None
        self.search_stage = "Search"
        self.states = {}
        if start_method:
            self.update("Method", status="started")

    def update(self, stage, *, done=None, total=None, status="running", eta=True):
        if self.logger is None:
            return
        now = time.monotonic()
        state = self.states.get(stage)
        if state is None or status == "started":
            state = {"start": now, "last": float("-inf"), "closed": False}
            self.states[stage] = state
        if state["closed"]:
            return
        terminal = status in {"finished", "skipped", "failed"}
        enabled = os.environ.get("JX_GS_PROGRESS", "1").strip().lower() in {"1", "true", "yes", "on"}
        if status == "running" and (not enabled or now - state["last"] < self.interval):
            return
        elapsed = max(0.0, now - state["start"])
        count = ""
        remaining = "--"
        if done is not None:
            done = max(0, int(done))
            if total is not None and int(total) > 0:
                total = int(total)
                done = min(done, total)
                count = f" {done:,}/{total:,}"
                if eta:
                    count += f" {100.0 * done / total:5.1f}%"
                    if done == total:
                        remaining = "0s"
                    elif done > 0 and elapsed >= 1.0:
                        remaining = format_elapsed(elapsed * (total - done) / done)
            else:
                count = f" done={done:,}"
        line = (
            f"GS [{self.trait}] {self.model:<10} {stage:<7}{count}"
            f" | elapsed={format_elapsed(elapsed)} ETA={remaining} {status}"
        )
        if stdout_is_tty():
            _emit_to_file_handlers(self.logger, logging.INFO, line)
        else:
            self.logger.info(line)
        for handler in self.logger.handlers:
            handler.flush()
        state["last"] = now
        state["closed"] = terminal

    def wrap_stage(self, hook):
        def callback(event, payload):
            stage, _, action = str(event).rpartition("_")
            if stage in {"cv", "fit", "predict"} and action in {"start", "end"}:
                if action == "start" and self.search_stage in self.states and not self.states[self.search_stage]["closed"]:
                    self.update(self.search_stage, done=self.search_done, total=self.search_total, status="finished")
                name = {"cv": "CV", "fit": "Fit", "predict": "Predict"}[stage]
                status = "started" if action == "start" else ("skipped" if payload.get("skipped") else "finished")
                self.update(name, done=self.cv_done if stage == "cv" else None,
                            total=self.folds if stage == "cv" else None, status=status)
            if hook is not None:
                hook(event, payload)
        return callback

    def wrap_cv(self, hook):
        def callback(method, increment):
            self.cv_done = min(self.folds, self.cv_done + max(0, int(increment)))
            self.update("CV", done=self.cv_done, total=self.folds)
            if hook is not None:
                hook(method, increment)
        return callback

    def wrap_search(self, hook):
        def callback(event, payload):
            if event == "search_plan":
                self.search_total = max(0, int(payload.get("count", 0)))
                self.search_done = 0
                self.search_stage = "Search " + str(payload.get("stage", "configs"))
                self.update(self.search_stage, done=0, total=self.search_total, status="started")
            elif event == "search_advance":
                self.search_done += max(0, int(payload.get("inc", 0)))
                complete = self.search_total is not None and self.search_total > 0 and self.search_done >= self.search_total
                self.update(self.search_stage, done=self.search_done, total=self.search_total,
                            status="finished" if complete else "running")
            if hook is not None:
                hook(event, payload)
        return callback

    def wrap_iterations(self, hook):
        def callback(event, payload):
            # PCG iteration limits are ceilings, not a known amount of work.
            if event in {"pcg_start", "pcg_iter", "pcg_end"}:
                self.update("PCG", done=payload.get("iter", payload.get("iters", 0)),
                            status="started" if event == "pcg_start" else "finished" if event == "pcg_end" else "running",
                            eta=False)
            if hook is not None:
                hook(event, payload)
        return callback

    def finish(self, *, failed=False):
        for stage, state in list(self.states.items()):
            if stage != "Method" and not state["closed"]:
                # A successful method return closes stages without inventing counts.
                self.update(stage, status="failed" if failed else "finished")
        if "Method" in self.states:
            self.update("Method", status="failed" if failed else "finished")


def trace_preparation(stage, *, counted=False):
    """Add optional logging around a preparation kernel without changing its work."""
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            logger = kwargs.pop("progress_logger", None)
            trait = kwargs.pop("progress_trait", "genotype")
            if logger is None:
                return function(*args, **kwargs)
            reporter = GSProgressLog(logger, "Prepare", trait, 0, start_method=False)
            done = None
            total = None
            if counted:
                total = int(kwargs["packed_ctx"]["packed"].shape[0])
                original_callback = kwargs.get("progress_callback")

                def callback(completed, backend_total):
                    nonlocal done, total
                    done, total = int(completed), int(backend_total)
                    reporter.update(stage, done=done, total=total)
                    if original_callback is not None:
                        original_callback(completed, backend_total)

                kwargs["progress_callback"] = callback
                if not kwargs.get("progress_every"):
                    kwargs["progress_every"] = max(1, (total + 199) // 200)
            reporter.update(stage, done=0 if counted else None, total=total, status="started")
            try:
                result = function(*args, **kwargs)
            except Exception:
                reporter.finish(failed=True)
                raise
            reporter.update(stage, done=done, total=total, status="finished")
            return result
        return run
    return decorate
