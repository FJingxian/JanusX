# -*- coding: utf-8 -*-
"""
JanusX updater.

Examples
--------
  jx --update
"""

from __future__ import annotations

import subprocess
import sys
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep

try:
    from rich.console import Console
except Exception:
    Console = None


_GITHUB_PROXY_SPEC = "git+https://gh-proxy.org/https://github.com/FJingxian/JanusX.git"
_GITHUB_SPEC = "git+https://github.com/FJingxian/JanusX.git"
_PIP_TIMEOUT_SECONDS = 30
_RICH_CONSOLE = Console() if Console is not None else None


class Loader:
    def __init__(
        self,
        desc: str = "Updating...",
        end: str = "",
        timeout: float = 0.1,
        enabled: bool = True,
    ) -> None:
        self.desc = desc
        self.end = end
        self.timeout = timeout
        self.enabled = enabled
        self.steps = ("|", "/", "-", "\\")
        self.done = False
        self._thread = Thread(target=self._animate, daemon=True)

    def _animate(self) -> None:
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r[{c}] {self.desc}", flush=True, end="")
            sleep(self.timeout)

    def start(self) -> "Loader":
        if not self.enabled:
            print(self.desc)
            return self
        self._thread.start()
        return self

    def stop(self) -> None:
        if self.enabled:
            self.done = True
            self._thread.join(timeout=self.timeout * 4)
            cols = get_terminal_size((80, 20)).columns
            print("\r" + " " * cols, end="", flush=True)
            print("\r", end="", flush=True)
        if self.end:
            print(self.end, flush=True)

    def __enter__(self) -> "Loader":
        return self.start()

    def __exit__(self, exc_type, exc_value, tb) -> None:
        self.stop()


def _run_update(spec: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-cache-dir",
        "--disable-pip-version-check",
        "--default-timeout",
        str(_PIP_TIMEOUT_SECONDS),
        spec,
    ]
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _looks_like_timeout(output: str) -> bool:
    text = str(output).lower()
    timeout_tokens = (
        "timed out",
        "readtimeout",
        "connect timeout",
        "connection timed out",
        "operation timed out",
        "timeouterror",
    )
    return any(token in text for token in timeout_tokens)


def _extract_error_reason(output: str) -> str:
    lines = [line.strip() for line in str(output).splitlines() if line.strip()]
    if not lines:
        return "No pip output captured."
    error_lines = [line for line in lines if "error" in line.lower()]
    if error_lines:
        return error_lines[-1]
    return lines[-1]


def _print_failure(label: str, proc: subprocess.CompletedProcess[str]) -> None:
    cmd = " ".join(proc.args) if isinstance(proc.args, (list, tuple)) else str(proc.args)
    print(f"{label}")
    print(f"Command: {cmd}")
    print("-" * 60)
    if proc.stdout:
        print(proc.stdout.rstrip())
    else:
        print("(no pip output)")
    print("-" * 60)
    print(f"Reason: {_extract_error_reason(proc.stdout)}")


def _status(desc: str, enabled: bool):
    if enabled and _RICH_CONSOLE is not None:
        return _RICH_CONSOLE.status(f"[bold green]{desc}")
    return Loader(desc, enabled=enabled)


def main() -> None:
    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())
    with _status("Updating...", enabled=use_spinner):
        direct = _run_update(_GITHUB_SPEC)
    if direct.returncode == 0:
        print("JanusX update completed.")
        return

    if _looks_like_timeout(direct.stdout):
        print("Direct GitHub update timed out, retrying with proxy...")
    else:
        print("Direct GitHub update failed, retrying with proxy...")

    with _status("Updating via proxy...", enabled=use_spinner):
        proxied = _run_update(_GITHUB_PROXY_SPEC)
    if proxied.returncode == 0:
        print("JanusX update completed (via proxy).")
        return

    _print_failure("GitHub attempt failed.", direct)
    _print_failure("Proxy retry failed.", proxied)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
