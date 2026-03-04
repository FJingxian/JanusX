# -*- coding: utf-8 -*-
"""
JanusX updater.

Usage:
  jx --update
  jx --update latest
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from importlib import metadata as importlib_metadata
from typing import List, Tuple

from janusx.script._common.status import CliStatus


_PYPI_SPEC = "janusx"
_GITHUB_SPEC = "git+https://github.com/FJingxian/JanusX.git"
_GITHUB_PROXY_SPEC = "git+https://gh-proxy.org/https://github.com/FJingxian/JanusX.git"
_PIP_TIMEOUT_SECONDS = 30
_WIN_STAGE2_FLAG = "--janusx-update-stage2"


def _run_update(spec: str, *, force_reinstall: bool) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--disable-pip-version-check",
        "--default-timeout",
        str(_PIP_TIMEOUT_SECONDS),
    ]
    if force_reinstall:
        cmd.extend(["--force-reinstall", "--no-cache-dir"])
    cmd.append(spec)
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _installed_version() -> str | None:
    try:
        v = str(importlib_metadata.version("janusx")).strip()
        return v if v else None
    except Exception:
        return None


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
    print(label)
    print(f"Command: {cmd}")
    print("-" * 60)
    if proc.stdout:
        print(proc.stdout.rstrip())
    else:
        print("(no pip output)")
    print("-" * 60)
    print(f"Reason: {_extract_error_reason(proc.stdout)}")


def _git_install_hint() -> str:
    if os.name == "nt" or sys.platform.lower().startswith("win"):
        return (
            "Git is required for `jx --update latest`, but it was not found in PATH.\n"
            "Windows install options:\n"
            "  1) winget: winget install --id Git.Git -e\n"
            "  2) choco:  choco install git -y\n"
            "  3) installer: https://git-scm.com/download/win\n"
            "Then reopen terminal and run: jx --update latest"
        )
    if sys.platform.lower().startswith("darwin"):
        return (
            "Git is required for `jx --update latest`, but it was not found in PATH.\n"
            "macOS install options:\n"
            "  1) xcode-select --install\n"
            "  2) brew install git\n"
            "Then reopen terminal and run: jx --update latest"
        )
    return (
        "Git is required for `jx --update latest`, but it was not found in PATH.\n"
        "Linux install options:\n"
        "  Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y git\n"
        "  RHEL/CentOS/Fedora: sudo dnf install -y git (or: sudo yum install -y git)\n"
        "  openSUSE: sudo zypper install -y git\n"
        "Then reopen terminal and run: jx --update latest"
    )


def _ensure_git_available_or_exit() -> None:
    if shutil.which("git") is not None:
        return
    print(_git_install_hint(), flush=True)
    raise SystemExit(1)


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


def _split_stage2_and_user_args(argv: List[str]) -> Tuple[bool, int, List[str]]:
    args = list(argv)
    if _WIN_STAGE2_FLAG not in args:
        return False, -1, args
    idx = args.index(_WIN_STAGE2_FLAG)
    parent_pid = -1
    if idx + 1 < len(args):
        try:
            parent_pid = int(args[idx + 1])
        except Exception:
            parent_pid = -1
        del args[idx : idx + 2]
    else:
        del args[idx]
    return True, parent_pid, args


def _maybe_spawn_windows_stage2(user_args: List[str], *, is_stage2: bool) -> bool:
    if os.name != "nt":
        return False
    if is_stage2:
        return False

    cmd = [
        sys.executable,
        "-m",
        "janusx.script.update",
        _WIN_STAGE2_FLAG,
        str(os.getpid()),
    ]
    if user_args:
        cmd.extend(user_args)
    try:
        # Keep update in the same terminal (no popup window).
        # Parent returns immediately to release jx.exe lock; stage2 continues.
        subprocess.Popen(
            cmd,
            close_fds=False,
        )
        print("Windows update launcher started in this terminal.")
    except Exception:
        return False
    return True


def _pause_windows_stage2_exit(*, is_stage2: bool) -> None:
    if os.name != "nt" or not is_stage2:
        return
    if not bool(getattr(sys.stdin, "isatty", lambda: False)()):
        return
    try:
        print("Update finished. Press Enter to return...", flush=True)
        input()
    except Exception:
        return


def _parse_args(user_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jx --update",
        description=(
            "Update JanusX. Default channel is PyPI latest; "
            "use optional `latest` to update from GitHub."
        ),
    )
    parser.add_argument(
        "channel",
        nargs="?",
        default=None,
        choices=["latest"],
        help="Use `latest` to update from GitHub (requires git).",
    )
    parser.add_argument(
        "--force-reinstall",
        "--reinstall",
        "--full",
        dest="force_reinstall",
        action="store_true",
        help="Force reinstall package files.",
    )
    return parser.parse_args(user_args)


def _run_pypi_update(spec: str, *, force_reinstall: bool, use_spinner: bool) -> None:
    before = _installed_version()
    show_latest_hint = False
    with CliStatus("Updating from PyPI...", enabled=use_spinner) as task:
        proc = _run_update(spec, force_reinstall=force_reinstall)
        if proc.returncode == 0:
            after = _installed_version()
            if (not force_reinstall) and before is not None and after is not None and before == after:
                task.complete(f"Already the latest PyPI version ({after})")
                show_latest_hint = True
            else:
                task.complete("JanusX update completed.")
    if proc.returncode != 0:
        _print_failure("PyPI update failed.", proc)
        raise SystemExit(1)
    if show_latest_hint:
        print("Use `jx --update latest` for GitHub latest.")


def _run_github_update(*, force_reinstall: bool, use_spinner: bool) -> None:
    _ensure_git_available_or_exit()
    with CliStatus("Updating from GitHub...", enabled=use_spinner) as task:
        direct = _run_update(_GITHUB_SPEC, force_reinstall=force_reinstall)
        if direct.returncode == 0:
            task.complete("JanusX update completed.")
    if direct.returncode == 0:
        return

    if _looks_like_timeout(direct.stdout):
        print("Direct GitHub update timed out, retrying with proxy...")
    else:
        print("Direct GitHub update failed, retrying with proxy...")

    with CliStatus("Updating from proxy...", enabled=use_spinner) as task:
        proxied = _run_update(_GITHUB_PROXY_SPEC, force_reinstall=force_reinstall)
        if proxied.returncode == 0:
            task.complete("JanusX update completed (via proxy).")
    if proxied.returncode == 0:
        return

    _print_failure("GitHub attempt failed.", direct)
    _print_failure("Proxy retry failed.", proxied)
    raise SystemExit(1)


def main() -> None:
    is_stage2, parent_pid, user_args = _split_stage2_and_user_args(sys.argv[1:])
    args = _parse_args(user_args)

    if _maybe_spawn_windows_stage2(user_args, is_stage2=is_stage2):
        return
    if is_stage2:
        _wait_for_parent_exit_on_windows(parent_pid, timeout_seconds=180)

    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())

    if args.channel == "latest":
        _run_github_update(force_reinstall=bool(args.force_reinstall), use_spinner=use_spinner)
    else:
        _run_pypi_update(_PYPI_SPEC, force_reinstall=bool(args.force_reinstall), use_spinner=use_spinner)
    _pause_windows_stage2_exit(is_stage2=is_stage2)


if __name__ == "__main__":
    main()
