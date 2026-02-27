# -*- coding: utf-8 -*-
"""
JanusX updater.

Examples
--------
  jx update
"""

from __future__ import annotations

import subprocess
import sys


_GITHUB_PROXY_SPEC = "git+https://gh-proxy.org/https://github.com/FJingxian/JanusX.git"
_GITHUB_SPEC = "git+https://github.com/FJingxian/JanusX.git"
_PIP_TIMEOUT_SECONDS = 30


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


def main() -> None:
    print("Updating JanusX from GitHub...")
    direct = _run_update(_GITHUB_SPEC)
    if direct.returncode == 0:
        print("JanusX update completed.")
        return

    if _looks_like_timeout(direct.stdout):
        print("Direct GitHub update timed out, retrying with proxy...")
        proxied = _run_update(_GITHUB_PROXY_SPEC)
        if proxied.returncode == 0:
            print("JanusX update completed (via proxy).")
            return
        _print_failure("GitHub attempt timed out.", direct)
        _print_failure("Proxy retry failed.", proxied)
        raise SystemExit(1)

    _print_failure("GitHub update failed.", direct)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
