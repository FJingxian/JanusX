from __future__ import annotations

import sys
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
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

    def __init__(self, desc: str, *, enabled: bool = True, timeout: float = 0.08) -> None:
        self.desc = str(desc)
        self.enabled = bool(enabled)
        self.timeout = float(timeout)
        self._backend = "none"
        self._done = False
        self._thread: Optional[Thread] = None
        self._plain_done = False
        self._status_cm = None
        self._console = None

    def _animate_plain(self) -> None:
        for frame in cycle(_SPINNER_FRAMES):
            if self._plain_done:
                break
            print(f"\r{frame} {self.desc}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self) -> "CliStatus":
        if not self.enabled:
            self._backend = "none"
            return self

        if _HAS_RICH and getattr(sys.stdout, "isatty", lambda: False)():
            try:
                ensure_rich_spinner_registered()
                self._console = Console()
                self._status_cm = self._console.status(
                    self.desc,
                    spinner=_SPINNER_NAME,
                    spinner_style="cyan",
                )
                self._status_cm.__enter__()
                self._backend = "rich"
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
        if self._backend == "rich" and self._status_cm is not None:
            self._status_cm.__exit__(None, None, None)
            self._status_cm = None
            return

        if self._backend == "plain":
            self._plain_done = True
            if self._thread is not None:
                self._thread.join(timeout=self.timeout * 4)
            cols = get_terminal_size((80, 20)).columns
            print("\r" + " " * cols, end="", flush=True)
            print("\r", end="", flush=True)

    def complete(self, message: str) -> None:
        if self._done:
            return
        self._stop_spinner()
        print_success(message)
        self._done = True

    def fail(self, message: str) -> None:
        if self._done:
            return
        self._stop_spinner()
        print_failure(message)
        self._done = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._done:
            self._stop_spinner()
