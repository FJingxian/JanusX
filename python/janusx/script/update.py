# -*- coding: utf-8 -*-
"""
JanusX updater.

Examples
--------
  jx --update
"""

from __future__ import annotations

import os
import subprocess
import sys
from janusx.script._common.status import CliStatus


_GITHUB_PROXY_SPEC = "git+https://gh-proxy.org/https://github.com/FJingxian/JanusX.git"
_GITHUB_SPEC = "git+https://github.com/FJingxian/JanusX.git"
_PIP_TIMEOUT_SECONDS = 30
_WIN_STAGE2_FLAG = "--janusx-update-stage2"


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


def _wait_for_parent_exit_on_windows(parent_pid: int, timeout_seconds: int = 180) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
    except Exception:
        return
    if parent_pid <= 0:
        return

    SYNCHRONIZE = 0x00100000
    WAIT_TIMEOUT = 0x00000102
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, int(parent_pid))
    if not handle:
        return
    try:
        wait_ms = max(0, int(timeout_seconds)) * 1000
        ret = kernel32.WaitForSingleObject(handle, wait_ms)
        if ret == WAIT_TIMEOUT:
            return
    finally:
        kernel32.CloseHandle(handle)


def _maybe_spawn_windows_stage2() -> bool:
    """
    Windows self-update workaround:
    the currently running jx.exe cannot be replaced by pip.
    Spawn a stage-2 updater process that waits for this process to exit.
    """
    if os.name != "nt":
        return False
    if _WIN_STAGE2_FLAG in sys.argv:
        return False

    cmd = [
        sys.executable,
        "-m",
        "janusx.script.update",
        _WIN_STAGE2_FLAG,
        str(os.getpid()),
    ]
    try:
        subprocess.Popen(cmd)
    except Exception:
        return False
    print("Windows update launcher started. Waiting for current jx process to exit...")
    return True


def _handle_windows_stage2_args() -> None:
    if os.name != "nt":
        return
    if _WIN_STAGE2_FLAG not in sys.argv:
        return
    try:
        idx = sys.argv.index(_WIN_STAGE2_FLAG)
        parent_pid = int(sys.argv[idx + 1]) if (idx + 1) < len(sys.argv) else -1
    except Exception:
        parent_pid = -1
    _wait_for_parent_exit_on_windows(parent_pid, timeout_seconds=180)


def main() -> None:
    if _maybe_spawn_windows_stage2():
        return
    _handle_windows_stage2_args()

    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())
    with CliStatus("Updating...", enabled=use_spinner) as task:
        direct = _run_update(_GITHUB_SPEC)
        if direct.returncode == 0:
            task.complete("JanusX update completed.")
    if direct.returncode == 0:
        return

    if _looks_like_timeout(direct.stdout):
        print("Direct GitHub update timed out, retrying with proxy...")
    else:
        print("Direct GitHub update failed, retrying with proxy...")

    with CliStatus("Updating via proxy...", enabled=use_spinner) as task:
        proxied = _run_update(_GITHUB_PROXY_SPEC)
        if proxied.returncode == 0:
            task.complete("JanusX update completed (via proxy).")
    if proxied.returncode == 0:
        return

    _print_failure("GitHub attempt failed.", direct)
    _print_failure("Proxy retry failed.", proxied)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
