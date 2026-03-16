from __future__ import annotations

import logging
import math
import sys
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import monotonic, sleep
from typing import Iterable, Optional, Sequence

try:
    from rich.console import Console
    from rich.spinner import SPINNERS

    _HAS_RICH = True
except Exception:
    Console = None  # type: ignore[assignment]
    SPINNERS = {}  # type: ignore[assignment]
    _HAS_RICH = False

_SPINNER_NAME = "janusx_ascii"
_SPINNER_FRAMES = ["|", "/", "-", "\\"]
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"
JANUSX_SPINNER_NAME = _SPINNER_NAME


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


def _split_leading_newlines(text: str) -> tuple[str, str]:
    s = str(text)
    i = 0
    while i < len(s) and s[i] in "\r\n":
        i += 1
    return s[:i], s[i:]


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


def spinner_refresh_interval(seconds: Optional[float]) -> float:
    s = float(max(0.0, float(seconds or 0.0)))
    if s < 60.0:
        return 0.1
    return 1.0


def ensure_rich_spinner_registered() -> None:
    if not _HAS_RICH:
        return
    if _SPINNER_NAME not in SPINNERS:
        SPINNERS[_SPINNER_NAME] = {
            "interval": 80,
            "frames": _SPINNER_FRAMES,
        }


def get_rich_spinner_name() -> str:
    ensure_rich_spinner_registered()
    return JANUSX_SPINNER_NAME


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
    msg = str(message)
    line = msg if msg.startswith("Warning: ") else f"Warning: {msg}"
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


class CliStatus:
    """CLI status helper with rich-first spinner and plain fallback."""

    def __init__(
        self,
        desc: str,
        *,
        enabled: bool = True,
        timeout: float = 0.08,
        show_elapsed: bool = True,
    ) -> None:
        self.desc = str(desc)
        self.enabled = bool(enabled)
        self.timeout = float(timeout)
        self.show_elapsed = bool(show_elapsed)
        self._backend = "none"
        self._done = False
        self._thread: Optional[Thread] = None
        self._plain_done = False
        self._status_cm = None
        self._console = None
        self._start_ts: Optional[float] = None

    def _animate_plain(self) -> None:
        for frame in cycle(_SPINNER_FRAMES):
            if self._plain_done:
                break
            _safe_print(f"\r{frame} {self._running_line()}", flush=True, end="")
            sleep(max(self.timeout, spinner_refresh_interval(self._elapsed_seconds())))

    def _animate_rich(self) -> None:
        while not self._plain_done:
            if self._status_cm is None:
                break
            try:
                self._status_cm.update(self._running_line())
            except Exception:
                break
            sleep(max(self.timeout, spinner_refresh_interval(self._elapsed_seconds())))

    def __enter__(self) -> "CliStatus":
        self._start_ts = monotonic()
        if not self.enabled:
            self._backend = "none"
            return self

        if _HAS_RICH and stdout_is_tty():
            try:
                ensure_rich_spinner_registered()
                assert Console is not None
                self._console = Console()
                self._status_cm = self._console.status(
                    self._running_line(),
                    spinner=_SPINNER_NAME,
                    spinner_style="cyan",
                )
                self._status_cm.__enter__()
                self._backend = "rich"
                if self.show_elapsed:
                    self._plain_done = False
                    self._thread = Thread(target=self._animate_rich, daemon=True)
                    self._thread.start()
                return self
            except Exception:
                self._status_cm = None
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

        if self._backend == "rich" and self._status_cm is not None:
            self._status_cm.__exit__(None, None, None)
            self._status_cm = None
            return

        if self._backend == "plain":
            cols = get_terminal_size((80, 20)).columns
            _safe_print("\r" + " " * cols, end="", flush=True)
            _safe_print("\r", end="", flush=True)

    def _elapsed_seconds(self) -> Optional[float]:
        if self._start_ts is None:
            return None
        return max(0.0, float(monotonic() - self._start_ts))

    @staticmethod
    def _format_elapsed(seconds: Optional[float]) -> str:
        return format_elapsed(seconds)

    def _compose_line(self, symbol: str, message: str) -> str:
        base = f"{symbol} {str(message)}"
        if not self.show_elapsed:
            return base
        elapsed_text = self._format_elapsed(self._elapsed_seconds())
        return f"{base} [{elapsed_text}]"

    def _running_line(self) -> str:
        if not self.show_elapsed:
            return self.desc
        return f"{self.desc} [{self._format_elapsed(self._elapsed_seconds())}]"

    def complete(self, message: str) -> None:
        if self._done:
            return
        self._stop_spinner()
        symbol = _SUCCESS_SYMBOL
        line = self._compose_line(symbol, str(message))
        if self._backend == "rich" and self._console is not None:
            self._console.print(f"[green]{line}[/green]")
        elif self.enabled and stdout_is_tty():
            _safe_print(f"{_GREEN}{line}{_RESET}")
        else:
            _safe_print(line)
        self._done = True

    def fail(self, message: str) -> None:
        if self._done:
            return
        self._stop_spinner()
        symbol = _FAIL_SYMBOL
        line = self._compose_line(symbol, str(message))
        tail = line[len(symbol):]
        if self._backend == "rich" and self._console is not None:
            self._console.print(f"[red]{symbol}[/red]{tail}")
        elif self.enabled and stdout_is_tty():
            _safe_print(f"{_RED}{symbol}{_RESET}{tail}")
        else:
            _safe_print(line)
        self._done = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._done:
            self._stop_spinner()
