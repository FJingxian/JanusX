from __future__ import annotations

import time
from typing import Iterable, Sequence

from .status import (
    format_elapsed,
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
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _HAS_RICH_PROGRESS = True
except Exception:
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    TimeRemainingColumn = None  # type: ignore[assignment]
    _HAS_RICH_PROGRESS = False

try:
    from tqdm.auto import tqdm as _tqdm_auto

    _HAS_TQDM = True
except Exception:
    _tqdm_auto = None  # type: ignore[assignment]
    _HAS_TQDM = False


def rich_progress_available() -> bool:
    return bool(_HAS_RICH_PROGRESS and stdout_is_tty())


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
            # Absolute fallback for environment-specific Rich spinner registries.
            spinner_col = SpinnerColumn(
                spinner_name="line",
                style="cyan",
                finished_text=str(finished_text),
            )
        columns.append(spinner_col)
    columns.append(TextColumn(str(description_template)))
    if show_bar:
        assert BarColumn is not None
        columns.append(BarColumn(bar_width=bar_width) if bar_width is not None else BarColumn())
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


class ProgressAdapter:
    """Rich-first progress bar with tqdm fallback."""

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
    ) -> None:
        self.total = int(max(0, total))
        self.desc = str(desc)
        self.emit_done = bool(emit_done)
        self._progress = None
        self._task_id = None
        self._tqdm = None
        self._backend = "none"
        self._finished = False
        self._start_ts = time.monotonic()
        self._animate = bool(force_animate or should_animate_status(self.desc))
        self._show_postfix = bool(show_postfix)
        self._keep_display = bool(keep_display)

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
        if not self._show_postfix:
            return
        if len(kwargs) == 0:
            return
        text = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, postfix=text)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.set_postfix(kwargs)

    def set_desc(self, desc: str) -> None:
        self.desc = str(desc)
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=self.desc)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.set_description_str(self.desc)

    def finish(self) -> None:
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=self.total)
        elif self._backend == "tqdm" and self._tqdm is not None and self._tqdm.total is not None:
            self._tqdm.n = self._tqdm.total
            self._tqdm.refresh()
        self._finished = True

    def close(self) -> None:
        elapsed = format_elapsed(time.monotonic() - self._start_ts)
        if self._backend == "rich" and self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None
        if self._finished and self.emit_done:
            msg = f"{self.desc} ...Finished [{elapsed}]"
            if self._animate:
                print_success(msg, force_color=True)
            else:
                print(msg, flush=True)
