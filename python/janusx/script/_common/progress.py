from __future__ import annotations

import time
from typing import Iterable, Optional, Sequence

from .status import (
    CliStatus,
    format_elapsed_parts,
    get_rich_spinner_name,
    print_success,
    should_animate_status,
    stdout_is_tty,
)

try:
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        Task,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text

    _HAS_RICH_PROGRESS = True
except Exception:
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    Task = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    TimeRemainingColumn = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]
    _HAS_RICH_PROGRESS = False

try:
    from tqdm.auto import tqdm as _tqdm_auto

    _HAS_TQDM = True
except Exception:
    _tqdm_auto = None  # type: ignore[assignment]
    _HAS_TQDM = False


def rich_progress_available() -> bool:
    return bool(_HAS_RICH_PROGRESS and stdout_is_tty())


def _normalize_status_base(desc: str) -> str:
    base = str(desc).strip()
    while base.endswith("."):
        base = base[:-1].rstrip()
    return base if base != "" else "Task"


def running_status_text(desc: str) -> str:
    return f"{_normalize_status_base(desc)}..."


def finished_status_text(desc: str) -> str:
    return f"{_normalize_status_base(desc)} ...Finished"


def failed_status_text(desc: str) -> str:
    return f"{_normalize_status_base(desc)} ...Failed"


class MaybeHiddenBarColumn(BarColumn):
    """Rich bar column that can be hidden per-task via `hide_bar=True`."""

    def render(self, task: "Task"):  # type: ignore[override]
        if bool(getattr(task, "fields", {}).get("hide_bar", False)):
            assert Text is not None
            return Text("")
        return super().render(task)


def build_rich_progress(
    *,
    description_template: str = "[green]{task.description}",
    show_spinner: bool = True,
    show_bar: bool = True,
    show_percentage: bool = True,
    show_elapsed: bool = True,
    show_remaining: bool = False,
    field_templates_before_elapsed: Sequence[str] | None = None,
    field_templates: Sequence[str] | None = None,
    bar_width: int | None = None,
    finished_text: str = " ",
    transient: bool = True,
):
    if not rich_progress_available():
        return None
    assert Progress is not None
    assert SpinnerColumn is not None
    assert TextColumn is not None

    columns: list[object] = []
    if show_spinner:
        spinner_name = get_rich_spinner_name()
        try:
            spinner_col = SpinnerColumn(
                spinner_name=spinner_name,
                style="cyan",
                finished_text=str(finished_text),
            )
        except Exception:
            spinner_col = SpinnerColumn(
                spinner_name="dots",
                style="cyan",
                finished_text=str(finished_text),
            )
        columns.append(spinner_col)
    columns.append(TextColumn(str(description_template)))
    if show_bar:
        assert BarColumn is not None
        columns.append(
            MaybeHiddenBarColumn(bar_width=bar_width)
            if bar_width is not None
            else MaybeHiddenBarColumn()
        )
    if show_percentage:
        columns.append(TextColumn("{task.percentage:>6.1f}%"))
    for template in (field_templates_before_elapsed or []):
        columns.append(TextColumn(str(template)))
    if show_elapsed:
        assert TimeElapsedColumn is not None
        columns.append(TimeElapsedColumn())
    if show_remaining:
        assert TimeRemainingColumn is not None
        columns.append(TimeRemainingColumn())
    for template in (field_templates or []):
        columns.append(TextColumn(str(template)))
    return Progress(*columns, transient=bool(transient))


class SpinnerStatusAdapter:
    """Spinner-only status line: running elapsed, final `...Finished [time]`."""

    def __init__(
        self,
        desc: str,
        *,
        enabled: bool = True,
        timeout: float = 0.08,
        use_process: bool = False,
        force_animate: bool = True,
    ) -> None:
        self.desc = _normalize_status_base(desc)
        self._status = CliStatus(
            running_status_text(self.desc),
            enabled=bool(enabled),
            timeout=float(timeout),
            show_elapsed=True,
            use_process=bool(use_process),
            force_animate=bool(force_animate),
        )
        self._started = False
        self._closed = False
        self._done_message: Optional[str] = None
        self._done_elapsed_parts: Optional[list[object]] = None

    def start(self) -> "SpinnerStatusAdapter":
        if not self._started:
            self._status.__enter__()
            self._started = True
        return self

    def __enter__(self) -> "SpinnerStatusAdapter":
        return self.start()

    def finish(
        self,
        *,
        message: Optional[str] = None,
        elapsed_parts: Optional[Iterable[object]] = None,
    ) -> None:
        if self._closed:
            return
        self.start()
        self._done_message = str(message) if message is not None else finished_status_text(self.desc)
        self._done_elapsed_parts = list(elapsed_parts) if elapsed_parts is not None else None
        self._status.complete(
            self._done_message,
            elapsed_parts=self._done_elapsed_parts,
        )
        self._closed = True

    def fail(
        self,
        *,
        message: Optional[str] = None,
        elapsed_parts: Optional[Iterable[object]] = None,
    ) -> None:
        if self._closed:
            return
        self.start()
        self._status.fail(
            str(message) if message is not None else failed_status_text(self.desc),
            elapsed_parts=elapsed_parts,
        )
        self._closed = True

    def close(
        self,
        *,
        show_done: bool = False,
        message: Optional[str] = None,
        elapsed_parts: Optional[Iterable[object]] = None,
    ) -> None:
        if self._closed:
            return
        self.start()
        if show_done:
            self.finish(message=message, elapsed_parts=elapsed_parts)
            return
        self._status.close(show_done=False)
        self._closed = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed:
            self.close(show_done=False)


class ProgressBarAdapter:
    """Spinner + progress bar + elapsed/ETA, with unified final done line."""

    def __init__(
        self,
        total: int,
        desc: str,
        *,
        show_spinner: bool = True,
        show_postfix: bool = True,
        keep_display: bool = False,
        show_remaining: bool = True,
        emit_done: bool = True,
        bar_width: int | None = None,
        force_animate: bool = False,
        logger=None,
        log_unit: str = "item",
    ) -> None:
        self.total = int(max(0, total))
        self.desc = _normalize_status_base(desc)
        self.emit_done = bool(emit_done)
        self.logger = logger
        self.log_unit = str(log_unit).strip() or "item"
        self._progress = None
        self._task_id = None
        self._tqdm = None
        self._backend = "none"
        self._finished = False
        self._start_ts = time.monotonic()
        self._animate = bool((force_animate or should_animate_status(self.desc)) and stdout_is_tty())
        self._show_postfix = bool(show_postfix)
        self._keep_display = bool(keep_display)
        self._done_message: Optional[str] = None
        self._done_elapsed_parts: Optional[list[object]] = None

        if self._animate:
            progress = build_rich_progress(
                show_spinner=bool(show_spinner),
                show_remaining=bool(show_remaining),
                field_templates=["{task.fields[postfix]}"] if self._show_postfix else [],
                bar_width=bar_width,
                finished_text=" ",
                transient=(not self._keep_display),
            )
            if progress is not None:
                try:
                    self._progress = progress
                    self._progress.start()
                    self._task_id = self._progress.add_task(
                        self.desc,
                        total=self.total,
                        postfix="",
                    )
                    self._backend = "rich"
                except Exception:
                    self._progress = None
                    self._task_id = None
                    self._backend = "none"

            if self._backend == "none" and _HAS_TQDM and stdout_is_tty():
                assert _tqdm_auto is not None
                self._tqdm = _tqdm_auto(
                    total=self.total,
                    desc=self.desc,
                    ascii=True,
                    leave=bool(self._keep_display),
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                )
                self._backend = "tqdm"

    def update(self, n: int) -> None:
        step = int(max(0, n))
        if step == 0:
            return
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=step)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.update(step)

    def set_postfix(self, **kwargs: object) -> None:
        if not self._show_postfix or len(kwargs) == 0:
            return
        text = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, postfix=text)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.set_postfix(kwargs)

    def set_desc(self, desc: str) -> None:
        self.desc = _normalize_status_base(desc)
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=self.desc)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.set_description_str(self.desc)

    def set_description(self, desc: str) -> None:
        self.set_desc(desc)

    def set_total(self, total: int) -> None:
        new_total = int(max(0, total))
        self.total = new_total
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, total=self.total)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.total = self.total
            self._tqdm.refresh()

    def finish(
        self,
        *,
        message: Optional[str] = None,
        elapsed_parts: Optional[Iterable[object]] = None,
    ) -> None:
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=self.total)
        elif self._backend == "tqdm" and self._tqdm is not None and self._tqdm.total is not None:
            self._tqdm.n = self._tqdm.total
            self._tqdm.refresh()
        self._finished = True
        if message is not None:
            self._done_message = str(message)
        if elapsed_parts is not None:
            self._done_elapsed_parts = list(elapsed_parts)

    def complete(
        self,
        *,
        message: Optional[str] = None,
        elapsed_parts: Optional[Iterable[object]] = None,
    ) -> None:
        self.finish(message=message, elapsed_parts=elapsed_parts)
        self.close(show_done=True)

    def close(
        self,
        *,
        show_done: bool = True,
        message: Optional[str] = None,
        elapsed_parts: Optional[Iterable[object]] = None,
    ) -> None:
        elapsed_total = time.monotonic() - self._start_ts
        if self._backend == "rich" and self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None
        self._backend = "none"

        if not show_done:
            return
        if not self._finished or not self.emit_done:
            return

        done_msg = (
            str(message)
            if message is not None
            else (
                self._done_message
                if self._done_message is not None
                else finished_status_text(self.desc)
            )
        )
        done_parts = (
            list(elapsed_parts)
            if elapsed_parts is not None
            else self._done_elapsed_parts
        )
        elapsed_text = format_elapsed_parts(done_parts, fallback_seconds=elapsed_total)
        msg = f"{done_msg} [{elapsed_text}]"
        if self._animate:
            print_success(msg, force_color=True)
        else:
            print(msg, flush=True)


ProgressAdapter = ProgressBarAdapter
