from __future__ import annotations

import sys
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep, monotonic
from typing import Optional

try:
    from rich.console import Console
    from rich.spinner import SPINNERS

    _HAS_RICH = True
except Exception:
    Console = None  # type: ignore[assignment]
    SPINNERS = {}  # type: ignore[assignment]
    _HAS_RICH = False

_SPINNER_NAME = "janusx_braille"
_SPINNER_FRAMES = [
    "⠋",
    "⠙",
    "⠹",
    "⠸",
    "⠼",
    "⠴",
    "⠦",
    "⠧",
    "⠇",
    "⠏",
    "⠞",
]
_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"
JANUSX_SPINNER_NAME = _SPINNER_NAME


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


def print_success(message: str) -> None:
    msg = str(message)
    if _HAS_RICH and getattr(sys.stdout, "isatty", lambda: False)():
        try:
            Console().print(f"[green]✔︎[/green] {msg}")
            return
        except Exception:
            pass
    if getattr(sys.stdout, "isatty", lambda: False)():
        print(f"{_GREEN}✔︎{_RESET} {msg}", flush=True)
    else:
        print(f"✔︎ {msg}", flush=True)


def print_failure(message: str) -> None:
    msg = str(message)
    if _HAS_RICH and getattr(sys.stdout, "isatty", lambda: False)():
        try:
            Console().print(f"[red]✘[/red] {msg}")
            return
        except Exception:
            pass
    if getattr(sys.stdout, "isatty", lambda: False)():
        print(f"{_RED}✘{_RESET} {msg}", flush=True)
    else:
        print(f"✘ {msg}", flush=True)


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
            print(f"\r{frame} {self._running_line()}", flush=True, end="")
            sleep(self.timeout)

    def _animate_rich(self) -> None:
        while not self._plain_done:
            if self._status_cm is None:
                break
            try:
                self._status_cm.update(self._running_line())
            except Exception:
                break
            sleep(self.timeout)

    def __enter__(self) -> "CliStatus":
        self._start_ts = monotonic()
        if not self.enabled:
            self._backend = "none"
            return self

        if _HAS_RICH and getattr(sys.stdout, "isatty", lambda: False)():
            try:
                ensure_rich_spinner_registered()
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
            print("\r" + " " * cols, end="", flush=True)
            print("\r", end="", flush=True)

    def _elapsed_seconds(self) -> Optional[float]:
        if self._start_ts is None:
            return None
        return max(0.0, float(monotonic() - self._start_ts))

    @staticmethod
    def _format_elapsed(seconds: Optional[float]) -> str:
        if seconds is None:
            return "0.0s"
        s = float(max(0.0, seconds))
        if s < 60.0:
            return f"{s:.1f}s"
        total_seconds = int(round(s))
        if total_seconds < 3600:
            minutes, rem = divmod(total_seconds, 60)
            return f"{minutes}m{rem:02d}s"
        hours, rem_seconds = divmod(total_seconds, 3600)
        mins = rem_seconds // 60
        return f"{hours}h{mins:02d}m"

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
        symbol = "✔︎"
        line = self._compose_line(symbol, str(message))
        tail = line[len(symbol):]
        if self._backend == "rich" and self._console is not None:
            self._console.print(f"[green]{symbol}[/green]{tail}")
        elif self.enabled and getattr(sys.stdout, "isatty", lambda: False)():
            print(f"{_GREEN}{symbol}{_RESET}{tail}", flush=True)
        else:
            print(line, flush=True)
        self._done = True

    def fail(self, message: str) -> None:
        if self._done:
            return
        self._stop_spinner()
        symbol = "✘"
        line = self._compose_line(symbol, str(message))
        tail = line[len(symbol):]
        if self._backend == "rich" and self._console is not None:
            self._console.print(f"[red]{symbol}[/red]{tail}")
        elif self.enabled and getattr(sys.stdout, "isatty", lambda: False)():
            print(f"{_RED}{symbol}{_RESET}{tail}", flush=True)
        else:
            print(line, flush=True)
        self._done = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._done:
            self._stop_spinner()
