from __future__ import annotations

import logging
import math
import os
import sys
import time
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import monotonic, sleep
from typing import Iterable, Optional, Sequence

try:
    from rich.console import Console
    from rich.spinner import SPINNERS
    from rich.text import Text

    _HAS_RICH = True
except Exception:
    Console = None  # type: ignore[assignment]
    SPINNERS = {}  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]
    _HAS_RICH = False

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
    if not _HAS_RICH:
        Text = None  # type: ignore[assignment]
    _HAS_RICH_PROGRESS = False

try:
    from tqdm.auto import tqdm as _tqdm_auto

    _HAS_TQDM = True
except Exception:
    _tqdm_auto = None  # type: ignore[assignment]
    _HAS_TQDM = False

_SPINNER_NAME = "dots"
_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_SPINNER_REFRESH_SEC = 0.2
_SPINNER_FALLBACKS = ("dots", "simpleDots", "line", "bouncingBar")
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"
JANUSX_SPINNER_NAME = _SPINNER_NAME
_LOADING_PREFIXES = (
    "loading ",
    "inspecting ",
    "loaded ",
    "running ",
    "waiting ",
    "computing ",
    "building ",
    "calculating ",
    "visualizing ",
    "writing ",
    "converting ",
)
_COMPLETION_TOKENS = (
    "...finished",
    "...found",
    "...pve",
    "...r_hat",
)
_SKIP_TOKENS = (
    "skip",
    "skipped",
    "skipping",
    "ignored",
    "ignore",
)


def stdout_is_tty() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _encoding_supports(text: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        str(text).encode(enc)
        return True
    except Exception:
        return False


def _symbol_or_ascii(preferred: str, fallback: str) -> str:
    return str(preferred) if _encoding_supports(preferred) else str(fallback)


_SUCCESS_SYMBOL = _symbol_or_ascii("\u2714\ufe0e", "[OK]")
_FAIL_SYMBOL = _symbol_or_ascii("\u2718\ufe0e", "[X]")
_WARN_SYMBOL = "!"


def success_symbol() -> str:
    return _SUCCESS_SYMBOL


def failure_symbol() -> str:
    return _FAIL_SYMBOL


def _safe_console_text(text: str) -> str:
    s = str(text)
    enc = getattr(sys.stdout, "encoding", None)
    if not enc:
        return s
    try:
        s.encode(enc)
        return s
    except Exception:
        return s.encode(enc, errors="replace").decode(enc, errors="replace")


def _safe_print(text: str, *, end: str = "\n", flush: bool = True) -> None:
    print(_safe_console_text(text), end=end, flush=flush)


def _safe_write(text: str, *, flush: bool = True) -> None:
    sys.stdout.write(_safe_console_text(text))
    if flush:
        sys.stdout.flush()


def _supports_live_ansi(stream) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.name != "nt":
        return True
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


def _clear_current_console_line() -> None:
    # Prefer ANSI erase-in-line to avoid writing terminal-width spaces, which
    # can themselves wrap after a resize and leave a blank line artifact.
    if _supports_live_ansi(sys.stdout):
        _safe_write("\r\033[2K\r")
        return
    # Conservative fallback for non-ANSI environments.
    try:
        cols = int(get_terminal_size((80, 20)).columns)
    except Exception:
        cols = 80
    cols = max(1, int(cols))
    _safe_write("\r" + (" " * max(1, cols - 1)) + "\r")


def _terminal_columns(default: int = 80) -> int:
    try:
        return max(1, int(get_terminal_size((default, 20)).columns))
    except Exception:
        return max(1, int(default))


def _truncate_console_text(text: str, *, reserve: int = 0) -> str:
    s = str(text).replace("\r", " ").replace("\n", " ")
    if not stdout_is_tty():
        return s
    width = max(1, _terminal_columns() - max(0, int(reserve)))
    if len(s) <= width:
        return s
    if width <= 3:
        return s[:width]
    return s[: width - 3].rstrip() + "..."


def _rich_status_text(text: str, *, style: str | None = None):
    rendered = _truncate_console_text(text)
    if Text is None:
        return rendered
    return Text(
        rendered,
        style="" if style is None else str(style),
        no_wrap=True,
        overflow="ellipsis",
        end="",
    )


def _split_leading_newlines(text: str) -> tuple[str, str]:
    s = str(text)
    i = 0
    while i < len(s) and s[i] in "\r\n":
        i += 1
    return s[:i], s[i:]


def is_loading_status_text(text: object) -> bool:
    _, msg = _split_leading_newlines(text)
    t = str(msg).strip().lower()
    return any(t.startswith(prefix) for prefix in _LOADING_PREFIXES)


def is_completed_status_text(text: object) -> bool:
    _, msg = _split_leading_newlines(text)
    t = str(msg).strip().lower()
    if t == "":
        return False
    return any(tok in t for tok in _COMPLETION_TOKENS)


def is_skip_status_text(text: object) -> bool:
    _, msg = _split_leading_newlines(text)
    t = str(msg).strip().lower()
    return any(tok in t for tok in _SKIP_TOKENS)


def should_animate_status(text: object) -> bool:
    return is_loading_status_text(text) and (not is_skip_status_text(text))


def format_elapsed(seconds: Optional[float]) -> str:
    if seconds is None:
        return "0.0s"
    s = float(max(0.0, seconds))
    if s < 60.0:
        tenths = math.floor(s * 10.0) / 10.0
        return f"{tenths:.1f}s"
    total_seconds = int(math.floor(s))
    if total_seconds < 3600:
        minutes, rem = divmod(total_seconds, 60)
        return f"{minutes}m{rem:02d}s"
    total_minutes = int(math.floor(s / 60.0))
    hours, mins = divmod(total_minutes, 60)
    return f"{hours}h{mins:02d}m"


def format_elapsed_parts(
    parts: Optional[Iterable[object]],
    *,
    fallback_seconds: Optional[float] = None,
) -> str:
    out: list[str] = []
    if parts is not None:
        for item in parts:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                label = str(item[0]).strip()
                value = item[1]
                if isinstance(value, (int, float)):
                    value_text = format_elapsed(float(value))
                else:
                    value_text = str(value).strip()
                if value_text == "":
                    continue
                out.append(f"{label} {value_text}" if label != "" else value_text)
                continue
            if isinstance(item, (int, float)):
                out.append(format_elapsed(float(item)))
                continue
            text = str(item).strip()
            if text != "":
                out.append(text)
    if out:
        return "/".join(out)
    return format_elapsed(fallback_seconds)


def spinner_refresh_interval(seconds: Optional[float]) -> float:
    _ = seconds
    # Keep spinner animation cadence independent from elapsed-time formatting.
    return float(_SPINNER_REFRESH_SEC)


def plain_spinner_frames() -> tuple[str, ...]:
    return tuple(str(frame) for frame in _SPINNER_FRAMES)


def ensure_rich_spinner_registered() -> None:
    if not _HAS_RICH:
        return
    if _SPINNER_NAME != "janusx_ascii":
        return
    try:
        if _SPINNER_NAME not in SPINNERS:
            SPINNERS[_SPINNER_NAME] = {
                "interval": 80,
                "frames": _SPINNER_FRAMES,
            }
    except Exception:
        # Some Rich builds may expose a non-mutable spinner registry.
        # In that case we silently fall back to a built-in spinner name.
        return


def _resolve_spinner_name(preferred: str) -> str:
    if (not _HAS_RICH) or (not hasattr(SPINNERS, "__contains__")):
        return "dots"
    if str(preferred) in SPINNERS:
        return str(preferred)
    for name in _SPINNER_FALLBACKS:
        if name in SPINNERS:
            return name
    # Last-resort built-in name; Rich will usually have it.
    return "dots"


def get_rich_spinner_name() -> str:
    ensure_rich_spinner_registered()
    return _resolve_spinner_name(JANUSX_SPINNER_NAME)


def _console_for_print(*, force_color: bool) -> Optional["Console"]:
    if not _HAS_RICH:
        return None
    is_tty = stdout_is_tty()
    if not (is_tty or force_color):
        return None
    try:
        return Console(force_terminal=bool(force_color), no_color=False)
    except Exception:
        return None


def print_success(message: str, *, force_color: bool = False) -> None:
    leading, msg = _split_leading_newlines(message)
    if leading:
        _safe_write(leading)
    if is_skip_status_text(msg):
        print_skipped(msg, force_color=bool(force_color))
        return
    if (not is_loading_status_text(msg)) and (not is_completed_status_text(msg)):
        _safe_print(msg)
        return
    line = f"{_SUCCESS_SYMBOL} {msg}" if msg != "" else f"{_SUCCESS_SYMBOL}"
    console = _console_for_print(force_color=bool(force_color))
    if console is not None:
        try:
            console.print(f"[green]{line}[/green]")
            return
        except Exception:
            pass
    if stdout_is_tty() or bool(force_color):
        _safe_print(f"{_GREEN}{line}{_RESET}")
    else:
        _safe_print(line)


def print_failure(message: str, *, force_color: bool = False) -> None:
    leading, msg = _split_leading_newlines(message)
    if leading:
        _safe_write(leading)
    if is_skip_status_text(msg):
        print_skipped(msg, force_color=bool(force_color))
        return
    line = f"{_FAIL_SYMBOL} {msg}" if msg != "" else f"{_FAIL_SYMBOL}"
    console = _console_for_print(force_color=bool(force_color))
    if console is not None:
        try:
            console.print(f"[red]{line}[/red]")
            return
        except Exception:
            pass
    if stdout_is_tty() or bool(force_color):
        _safe_print(f"{_RED}{line}{_RESET}")
    else:
        _safe_print(line)


def print_warning(message: str, *, force_color: bool = False) -> None:
    leading, msg = _split_leading_newlines(message)
    if leading:
        _safe_write(leading)
    msg = str(msg)
    body = msg if msg.startswith("Warning: ") else f"Warning: {msg}"
    line = f"{_WARN_SYMBOL} {body}"
    console = _console_for_print(force_color=bool(force_color))
    if console is not None:
        try:
            console.print(f"[yellow]{line}[/yellow]")
            return
        except Exception:
            pass
    if stdout_is_tty() or bool(force_color):
        _safe_print(f"{_YELLOW}{line}{_RESET}")
    else:
        _safe_print(line)


def print_skipped(message: str, *, force_color: bool = False) -> None:
    leading, msg = _split_leading_newlines(message)
    if leading:
        _safe_write(leading)
    line = f"{_SUCCESS_SYMBOL} {msg}" if msg != "" else f"{_SUCCESS_SYMBOL}"
    console = _console_for_print(force_color=bool(force_color))
    if console is not None:
        try:
            console.print(f"[yellow]{line}[/yellow]")
            return
        except Exception:
            pass
    if stdout_is_tty() or bool(force_color):
        _safe_print(f"{_YELLOW}{line}{_RESET}")
    else:
        _safe_print(line)


def _emit_to_file_handlers(logger, level: int, message: str) -> bool:
    handled = False
    try:
        for handler in getattr(logger, "handlers", []):
            if isinstance(handler, logging.FileHandler):
                record = logger.makeRecord(
                    logger.name,
                    int(level),
                    __file__,
                    0,
                    str(message),
                    args=(),
                    exc_info=None,
                    func=None,
                    extra=None,
                )
                handler.handle(record)
                handled = True
    except Exception:
        handled = False
    return handled


def log_success(
    logger,
    message: str,
    *,
    log_message: str | None = None,
    force_color: bool = False,
) -> None:
    msg = str(message)
    file_msg = str(msg if log_message is None else log_message)
    if is_skip_status_text(msg):
        if stdout_is_tty() or bool(force_color):
            _emit_to_file_handlers(logger, logging.INFO, file_msg)
            print_skipped(msg, force_color=bool(force_color))
        else:
            logger.info(file_msg)
        return
    if stdout_is_tty() or bool(force_color):
        _emit_to_file_handlers(logger, logging.INFO, file_msg)
        print_success(msg, force_color=bool(force_color))
    else:
        logger.info(file_msg)


def warn_deprecated_alias_usage(
    aliases: Iterable[str],
    *,
    replacement: str,
    argv: Sequence[str] | None = None,
) -> bool:
    alias_list = [str(x).strip() for x in aliases if str(x).strip()]
    if len(alias_list) == 0:
        return False
    alias_set = set(alias_list)
    for tok in (argv if argv is not None else sys.argv[1:]):
        key = str(tok).split("=", 1)[0].strip()
        if key in alias_set:
            joined = "/".join(alias_list)
            print_warning(f"`{joined}` is deprecated; use `{replacement}`.")
            return True
    return False


def _animate_plain_process(
    desc: str,
    timeout: float,
    show_elapsed: bool,
    start_ts: float,
    stop_evt,
    parent_pid: int,
    use_color: bool,
) -> None:
    """
    Process-based spinner loop.

    Runs in a child process so animation/elapsed can keep updating even when
    the parent process is blocked inside a long C/Rust call that holds GIL.
    """
    idx = 0
    desc_text = str(desc)
    while True:
        try:
            if bool(stop_evt.is_set()):
                break
        except Exception:
            break
        if int(parent_pid) > 0:
            alive = True
            try:
                import psutil  # type: ignore

                alive = bool(psutil.pid_exists(int(parent_pid)))
            except Exception:
                try:
                    os.kill(int(parent_pid), 0)
                except Exception:
                    alive = False
            if not alive:
                break
        frame = _SPINNER_FRAMES[idx % len(_SPINNER_FRAMES)]
        idx += 1
        elapsed = max(0.0, float(monotonic() - float(start_ts)))
        if bool(show_elapsed):
            body = f"{desc_text} [{format_elapsed(elapsed)}]"
        else:
            body = f"{desc_text}"
        body = _truncate_console_text(body, reserve=2)
        frame_text = (
            f"{_CYAN}{frame}{_RESET}"
            if bool(use_color)
            else str(frame)
        )
        line = f"\r{frame_text} {body}"
        try:
            _safe_print(line, flush=True, end="")
        except Exception:
            break
        sleep(max(float(timeout), spinner_refresh_interval(elapsed)))


class CliStatus:
    """CLI status helper with rich-first spinner and plain fallback."""

    def __init__(
        self,
        desc: str,
        *,
        enabled: bool = True,
        timeout: float = 0.08,
        show_elapsed: bool = True,
        use_process: bool = False,
        force_animate: bool = False,
    ) -> None:
        self.desc = str(desc)
        self.enabled = bool(enabled)
        self.timeout = float(timeout)
        self.show_elapsed = bool(show_elapsed)
        self.use_process = bool(use_process)
        # Restrict spinner animation to interactive TTY only; in non-TTY
        # contexts keep output clean and emit only completion lines.
        self._animate = bool(
            self.enabled
            and stdout_is_tty()
            and (bool(force_animate) or should_animate_status(self.desc))
        )
        self._backend = "none"
        self._done = False
        self._thread: Optional[Thread] = None
        self._proc = None
        self._proc_stop_evt = None
        self._plain_done = False
        self._console = None
        self._rich_progress = None
        self._rich_task_id = None
        self._start_ts: Optional[float] = None

    def _animate_plain(self) -> None:
        for frame in cycle(_SPINNER_FRAMES):
            if self._plain_done:
                break
            body = _truncate_console_text(self._running_line(), reserve=2)
            frame_text = (
                f"{_CYAN}{frame}{_RESET}"
                if stdout_is_tty()
                else str(frame)
            )
            line = f"\r{frame_text} {body}"
            _safe_print(line, flush=True, end="")
            sleep(max(self.timeout, spinner_refresh_interval(self._elapsed_seconds())))

    def _animate_rich(self) -> None:
        while not self._plain_done:
            if self._rich_progress is None or self._rich_task_id is None:
                break
            sleep(max(self.timeout, spinner_refresh_interval(self._elapsed_seconds())))

    def __enter__(self) -> "CliStatus":
        self._start_ts = monotonic()
        if not self._animate:
            self._backend = "none"
            return self

        if self.use_process:
            try:
                main_mod = sys.modules.get("__main__")
                main_file = getattr(main_mod, "__file__", "")
                main_file_s = str(main_file or "").strip()
                # Windows spawn cannot bootstrap safely from `<stdin>` / `<string>`.
                if (main_file_s != "") and (not main_file_s.startswith("<")):
                    import multiprocessing as mp

                    ctx = mp.get_context("spawn")
                    self._proc_stop_evt = ctx.Event()
                    self._proc = ctx.Process(
                        target=_animate_plain_process,
                        args=(
                            self.desc,
                            float(self.timeout),
                            bool(self.show_elapsed),
                            float(self._start_ts),
                            self._proc_stop_evt,
                            int(os.getpid()),
                            bool(self.enabled and stdout_is_tty()),
                        ),
                        daemon=True,
                    )
                    self._proc.start()
                    self._backend = "process"
                    return self
            except Exception:
                self._proc = None
                self._proc_stop_evt = None

        if _HAS_RICH and stdout_is_tty():
            try:
                ensure_rich_spinner_registered()
                assert Console is not None
                self._console = Console()
                self._rich_progress = build_rich_progress(
                    description_template="[green]{task.description}",
                    show_spinner=True,
                    show_bar=False,
                    show_percentage=False,
                    show_elapsed=bool(self.show_elapsed),
                    show_remaining=False,
                    finished_text=" ",
                    transient=True,
                )
                if self._rich_progress is None:
                    raise RuntimeError("rich progress unavailable")
                self._rich_progress.start()
                self._rich_task_id = self._rich_progress.add_task(
                    _truncate_console_text(self.desc),
                    total=None,
                )
                self._backend = "rich"
                return self
            except Exception:
                try:
                    if self._rich_progress is not None:
                        self._rich_progress.stop()
                except Exception:
                    pass
                self._rich_progress = None
                self._rich_task_id = None
                self._console = None

        self._backend = "plain"
        self._plain_done = False
        self._thread = Thread(target=self._animate_plain, daemon=True)
        self._thread.start()
        return self

    def _stop_spinner(self) -> None:
        self._plain_done = True
        if self._thread is not None:
            self._thread.join(timeout=self.timeout * 4)
            self._thread = None

        if self._backend == "process":
            try:
                if self._proc_stop_evt is not None:
                    self._proc_stop_evt.set()
            except Exception:
                pass
            try:
                if self._proc is not None:
                    self._proc.join(timeout=max(0.5, self.timeout * 8))
                    if self._proc.is_alive():
                        self._proc.terminate()
                        self._proc.join(timeout=max(0.5, self.timeout * 8))
            except Exception:
                pass
            finally:
                self._proc = None
                self._proc_stop_evt = None

        if self._backend == "rich" and self._rich_progress is not None:
            try:
                self._rich_progress.stop()
            finally:
                self._rich_progress = None
                self._rich_task_id = None
            return

        if self._backend in ("plain", "process"):
            # No-op teardown for plain/process fallback to avoid introducing
            # extra blank lines in captured/pseudo-terminal outputs.
            return

    def _elapsed_seconds(self) -> Optional[float]:
        if self._start_ts is None:
            return None
        return max(0.0, float(monotonic() - self._start_ts))

    @staticmethod
    def _format_elapsed(seconds: Optional[float]) -> str:
        return format_elapsed(seconds)

    def _compose_line(
        self,
        symbol: str,
        message: str,
        *,
        elapsed_parts: Optional[Iterable[object]] = None,
    ) -> str:
        base = f"{symbol} {str(message)}"
        if not self.show_elapsed:
            return base
        elapsed_text = format_elapsed_parts(
            elapsed_parts,
            fallback_seconds=self._elapsed_seconds(),
        )
        return f"{base} [{elapsed_text}]"

    def _running_line(self) -> str:
        if not self.show_elapsed:
            return self.desc
        return f"{self.desc} [{self._format_elapsed(self._elapsed_seconds())}]"

    def complete(
        self,
        message: str,
        *,
        elapsed_parts: Optional[Iterable[object]] = None,
    ) -> None:
        if self._done:
            return
        self._stop_spinner()
        if self._backend in ("plain", "process") and stdout_is_tty():
            # Clear the in-place spinner line before printing completion text,
            # otherwise outputs like "- Computing ...✔︎ ...Finished" may glue
            # together on terminals that keep cursor at line tail.
            _clear_current_console_line()
        symbol = _SUCCESS_SYMBOL
        line = self._compose_line(symbol, str(message), elapsed_parts=elapsed_parts)
        if should_animate_status(message) or is_completed_status_text(message):
            if self._backend == "rich" and self._console is not None:
                self._console.print(_rich_status_text(line, style="green"))
            elif self.enabled:
                _safe_print(f"{_GREEN}{_truncate_console_text(line)}{_RESET}")
            else:
                _safe_print(line)
        elif is_skip_status_text(message):
            plain = str(message)
            if self.show_elapsed:
                plain = (
                    f"{plain} ["
                    f"{format_elapsed_parts(elapsed_parts, fallback_seconds=self._elapsed_seconds())}"
                    f"]"
                )
            print_skipped(plain, force_color=bool(self.enabled))
        else:
            plain = str(message)
            if self.show_elapsed:
                plain = (
                    f"{plain} ["
                    f"{format_elapsed_parts(elapsed_parts, fallback_seconds=self._elapsed_seconds())}"
                    f"]"
                )
            _safe_print(plain)
        self._done = True

    def fail(
        self,
        message: str,
        *,
        elapsed_parts: Optional[Iterable[object]] = None,
    ) -> None:
        if self._done:
            return
        self._stop_spinner()
        if self._backend in ("plain", "process") and stdout_is_tty():
            # Keep failure output aligned with complete(): remove stale spinner line.
            _clear_current_console_line()
        if is_skip_status_text(message):
            plain = str(message)
            if self.show_elapsed:
                plain = (
                    f"{plain} ["
                    f"{format_elapsed_parts(elapsed_parts, fallback_seconds=self._elapsed_seconds())}"
                    f"]"
                )
            print_skipped(plain, force_color=bool(self.enabled))
            self._done = True
            return
        symbol = _FAIL_SYMBOL
        line = self._compose_line(symbol, str(message), elapsed_parts=elapsed_parts)
        if self._backend == "rich" and self._console is not None:
            self._console.print(_rich_status_text(line, style="red"))
        elif self.enabled:
            _safe_print(f"{_RED}{_truncate_console_text(line)}{_RESET}")
        else:
            _safe_print(line)
        self._done = True

    def close(
        self,
        *,
        show_done: bool = False,
        message: Optional[str] = None,
        elapsed_parts: Optional[Iterable[object]] = None,
    ) -> None:
        if self._done:
            return
        if show_done:
            self.complete(
                str(self.desc if message is None else message),
                elapsed_parts=elapsed_parts,
            )
            return
        self._stop_spinner()
        if self._backend in ("plain", "process") and stdout_is_tty():
            _clear_current_console_line()
        self._done = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._done:
            self._stop_spinner()
            if self._backend in ("plain", "process") and stdout_is_tty():
                # When status context exits without complete()/fail(), ensure
                # the in-place spinner line is cleared before next log line.
                # This avoids glued outputs like:
                #   "- Waiting ... [1.4s]position... [10.5s]"
                _clear_current_console_line()


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


class FixedRateSpinnerColumn(SpinnerColumn):
    """Spinner column with a fixed wall-clock animation cadence."""

    def __init__(
        self,
        *,
        frame_interval_sec: float,
        spinner_name: str = "dots",
        style: str | None = "progress.spinner",
        finished_text: str | object = " ",
    ) -> None:
        super().__init__(
            spinner_name=str(spinner_name),
            style=style,
            speed=1.0,
            finished_text=finished_text,
        )
        self._frame_interval_sec = max(0.01, float(frame_interval_sec))
        self._anim_start: float | None = None

    def render(self, task: "Task"):  # type: ignore[override]
        if task.finished:
            return self.finished_text
        if self._anim_start is None:
            self._anim_start = monotonic()
        elapsed = max(0.0, monotonic() - self._anim_start)
        frames = list(getattr(self.spinner, "frames", ()) or ["."])
        idx = int(elapsed / self._frame_interval_sec) % len(frames)
        frame = frames[idx]
        if Text is None:
            return str(frame)
        return Text(str(frame), style=self.spinner.style or "")


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
            spinner_col = FixedRateSpinnerColumn(
                frame_interval_sec=_SPINNER_REFRESH_SEC,
                spinner_name=spinner_name,
                style="cyan",
                finished_text=str(finished_text),
            )
        except Exception:
            spinner_col = FixedRateSpinnerColumn(
                frame_interval_sec=_SPINNER_REFRESH_SEC,
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
    return Progress(
        *columns,
        auto_refresh=True,
        refresh_per_second=max(1.0, 1.0 / _SPINNER_REFRESH_SEC),
        transient=bool(transient),
    )


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
        self._done_message = (
            str(message) if message is not None else finished_status_text(self.desc)
        )
        self._done_elapsed_parts = (
            list(elapsed_parts) if elapsed_parts is not None else None
        )
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
        self._animate = bool(
            (force_animate or should_animate_status(self.desc)) and stdout_is_tty()
        )
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

__all__ = [
    "CliStatus",
    "JANUSX_SPINNER_NAME",
    "MaybeHiddenBarColumn",
    "ProgressAdapter",
    "ProgressBarAdapter",
    "SpinnerStatusAdapter",
    "_emit_to_file_handlers",
    "build_rich_progress",
    "ensure_rich_spinner_registered",
    "failed_status_text",
    "failure_symbol",
    "format_elapsed",
    "format_elapsed_parts",
    "get_rich_spinner_name",
    "is_completed_status_text",
    "is_loading_status_text",
    "is_skip_status_text",
    "log_success",
    "print_failure",
    "print_skipped",
    "print_success",
    "print_warning",
    "plain_spinner_frames",
    "rich_progress_available",
    "running_status_text",
    "should_animate_status",
    "spinner_refresh_interval",
    "stdout_is_tty",
    "success_symbol",
    "warn_deprecated_alias_usage",
]
