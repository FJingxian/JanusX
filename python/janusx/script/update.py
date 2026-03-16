# -*- coding: utf-8 -*-
"""
JanusX updater.

Usage:
  jx --update
  jx --update latest
  jx --update --verbose
  jx --update latest --verbose
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib import metadata as importlib_metadata
from time import monotonic
from typing import List, NamedTuple, Tuple

from janusx.script._common.status import CliStatus, format_elapsed


_PYPI_SPEC = "janusx"
_GITHUB_SPEC_CN = "git+https://gh-proxy.org/https://github.com/FJingxian/JanusX.git"
_GITHUB_SPEC_ORIGIN = "git+https://github.com/FJingxian/JanusX.git"
_PIP_TIMEOUT_SECONDS = 30
_WIN_STAGE2_FLAG = "--janusx-update-stage2"
_FORCE_FLAGS = {"--force-reinstall", "--reinstall", "--full"}
_VERBOSE_FLAGS = {"--verbose"}
_LATEST_FLAGS = {"latest", "--latest"}
_HELP_FLAGS = {"help", "-h", "--help"}


class _UpdateArgs(NamedTuple):
    latest: bool
    force_reinstall: bool
    verbose: bool


def _build_pip_cmd(spec: str, *, force_reinstall: bool) -> List[str]:
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
    return cmd


def _build_update_env(*, use_cn_mirror: bool) -> dict:
    env = os.environ.copy()
    if use_cn_mirror:
        env["RUSTUP_DIST_SERVER"] = "https://rsproxy.cn"
        env["RUSTUP_UPDATE_ROOT"] = "https://rsproxy.cn/rustup"
    else:
        env["RUSTUP_DIST_SERVER"] = "https://static.rust-lang.org"
        env["RUSTUP_UPDATE_ROOT"] = "https://static.rust-lang.org/rustup"
    env["CARGO_REGISTRIES_CRATES_IO_PROTOCOL"] = "sparse"
    env["CARGO_NET_GIT_FETCH_WITH_CLI"] = "true"
    return env


def _run_update(
    spec: str,
    *,
    force_reinstall: bool,
    use_cn_mirror: bool,
) -> subprocess.CompletedProcess[str]:
    cmd = _build_pip_cmd(spec, force_reinstall=force_reinstall)
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_build_update_env(use_cn_mirror=use_cn_mirror),
    )


def _run_update_verbose(
    spec: str,
    *,
    force_reinstall: bool,
    use_cn_mirror: bool,
    desc: str,
) -> Tuple[subprocess.CompletedProcess[str], float]:
    cmd = _build_pip_cmd(spec, force_reinstall=force_reinstall)
    print(desc)
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=_build_update_env(use_cn_mirror=use_cn_mirror),
    )

    start_ts = monotonic()
    all_lines: List[str] = []
    if proc.stdout is not None:
        for raw in proc.stdout:
            line = str(raw).rstrip("\n")
            all_lines.append(line)
            print(line, flush=True)
    returncode = proc.wait()

    elapsed = max(0.0, monotonic() - start_ts)
    output = "\n".join(all_lines)
    return subprocess.CompletedProcess(cmd, returncode, output), elapsed


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
        print("\nUpdate finished. Press Enter to return...", flush=True)
        input()
    except Exception:
        return


def _update_usage() -> str:
    return (
        "Update usage:\n"
        "  jx --update [latest] [--verbose] [--force-reinstall]\n\n"
        "Channels:\n"
        "  (default)  PyPI latest\n"
        "  latest     GitHub latest (requires git)\n\n"
        "Options:\n"
        "  --verbose  Print full installer details (no rich animation)\n"
    )


def _parse_args(user_args: List[str]) -> _UpdateArgs:
    latest = False
    force_reinstall = False
    verbose = False
    unknown: List[str] = []

    for raw in user_args:
        token = str(raw).strip()
        if not token:
            continue
        if token in _HELP_FLAGS:
            print(_update_usage())
            raise SystemExit(0)
        if token in _LATEST_FLAGS:
            latest = True
            continue
        if token in _FORCE_FLAGS:
            force_reinstall = True
            continue
        if token in _VERBOSE_FLAGS:
            verbose = True
            continue
        unknown.append(raw)

    if unknown:
        print(f"Unknown update option(s): {' '.join(unknown)}")
        print(_update_usage())
        raise SystemExit(2)

    return _UpdateArgs(
        latest=latest,
        force_reinstall=force_reinstall,
        verbose=verbose,
    )


def _run_pypi_update(
    spec: str,
    *,
    force_reinstall: bool,
    use_spinner: bool,
    verbose: bool,
) -> None:
    before = _installed_version()
    show_latest_hint = False
    if verbose:
        proc, elapsed = _run_update_verbose(
            spec,
            force_reinstall=force_reinstall,
            use_cn_mirror=True,
            desc="Updating from PyPI...",
        )
        if proc.returncode == 0:
            after = _installed_version()
            if (not force_reinstall) and before is not None and after is not None and before == after:
                print(f"It is the latest PyPI version ({after}) [{format_elapsed(elapsed)}]")
                show_latest_hint = True
            else:
                print(f"JanusX update completed. [{format_elapsed(elapsed)}]")
    else:
        with CliStatus("Updating from PyPI...", enabled=use_spinner) as task:
            proc = _run_update(
                spec,
                force_reinstall=force_reinstall,
                use_cn_mirror=True,
            )
            if proc.returncode == 0:
                after = _installed_version()
                if (not force_reinstall) and before is not None and after is not None and before == after:
                    task.complete(f"It is the latest PyPI version ({after})")
                    show_latest_hint = True
                else:
                    task.complete("JanusX update completed.")
    if proc.returncode != 0:
        _print_failure("PyPI update failed.", proc)
        raise SystemExit(1)
    if show_latest_hint:
        print("Use `jx --update latest` for GitHub latest.")


def _run_github_update(
    *,
    force_reinstall: bool,
    use_spinner: bool,
    verbose: bool,
) -> None:
    _ensure_git_available_or_exit()
    if verbose:
        direct, elapsed = _run_update_verbose(
            _GITHUB_SPEC_CN,
            force_reinstall=force_reinstall,
            use_cn_mirror=True,
            desc="Updating from GitHub (CN mirror)...",
        )
        if direct.returncode == 0:
            print(f"JanusX update completed. [{format_elapsed(elapsed)}]")
    else:
        with CliStatus("Updating from GitHub (CN mirror)...", enabled=use_spinner) as task:
            direct = _run_update(
                _GITHUB_SPEC_CN,
                force_reinstall=force_reinstall,
                use_cn_mirror=True,
            )
            if direct.returncode == 0:
                task.complete("JanusX update completed.")
    if direct.returncode == 0:
        return

    if _looks_like_timeout(direct.stdout):
        print("GitHub CN mirror timed out, retrying with source...")
    else:
        print("GitHub CN mirror failed, retrying with source...")

    if verbose:
        proxied, elapsed = _run_update_verbose(
            _GITHUB_SPEC_ORIGIN,
            force_reinstall=force_reinstall,
            use_cn_mirror=False,
            desc="Updating from GitHub (source)...",
        )
        if proxied.returncode == 0:
            print(f"JanusX update completed (via source). [{format_elapsed(elapsed)}]")
    else:
        with CliStatus("Updating from GitHub (source)...", enabled=use_spinner) as task:
            proxied = _run_update(
                _GITHUB_SPEC_ORIGIN,
                force_reinstall=force_reinstall,
                use_cn_mirror=False,
            )
            if proxied.returncode == 0:
                task.complete("JanusX update completed (via source).")
    if proxied.returncode == 0:
        return

    _print_failure("GitHub CN mirror attempt failed.", direct)
    _print_failure("Source retry failed.", proxied)
    raise SystemExit(1)


def main() -> None:
    is_stage2, parent_pid, user_args = _split_stage2_and_user_args(sys.argv[1:])
    args = _parse_args(user_args)

    if _maybe_spawn_windows_stage2(user_args, is_stage2=is_stage2):
        return
    if is_stage2:
        _wait_for_parent_exit_on_windows(parent_pid, timeout_seconds=180)

    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())

    if args.latest:
        _run_github_update(
            force_reinstall=bool(args.force_reinstall),
            use_spinner=use_spinner,
            verbose=bool(args.verbose),
        )
    else:
        _run_pypi_update(
            _PYPI_SPEC,
            force_reinstall=bool(args.force_reinstall),
            use_spinner=use_spinner,
            verbose=bool(args.verbose),
        )
    _pause_windows_stage2_exit(is_stage2=is_stage2)


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
