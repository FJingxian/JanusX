# -*- coding: utf-8 -*-
"""
JanusX updater.

Examples
--------
  jx --update
  jx --update --force-reinstall
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata as importlib_metadata
import importlib.util
from typing import List, Optional, Tuple
from janusx.script._common.status import CliStatus


_GITHUB_PROXY_SPEC = "git+https://gh-proxy.org/https://github.com/FJingxian/JanusX.git"
_GITHUB_SPEC = "git+https://github.com/FJingxian/JanusX.git"
_PIP_TIMEOUT_SECONDS = 30
_WIN_STAGE2_FLAG = "--janusx-update-stage2"
_FORCE_FLAGS = {"--force-reinstall", "--reinstall", "--full"}
_FAST_COMMIT_MARKER = ".janusx_fastupdate_commit"


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


def _git_install_hint() -> str:
    if os.name == "nt" or sys.platform.lower().startswith("win"):
        return (
            "Git is required for jx --update, but it was not found in PATH.\n"
            "Windows install options:\n"
            "  1) winget: winget install --id Git.Git -e\n"
            "  2) choco:  choco install git -y\n"
            "  3) installer: https://git-scm.com/download/win\n"
            "Then reopen terminal and run: jx --update"
        )
    if sys.platform.lower().startswith("darwin"):
        return (
            "Git is required for jx --update, but it was not found in PATH.\n"
            "macOS install options:\n"
            "  1) xcode-select --install\n"
            "  2) brew install git\n"
            "Then reopen terminal and run: jx --update"
        )
    return (
        "Git is required for jx --update, but it was not found in PATH.\n"
        "Linux install options:\n"
        "  Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y git\n"
        "  RHEL/CentOS/Fedora: sudo dnf install -y git (or: sudo yum install -y git)\n"
        "  openSUSE: sudo zypper install -y git\n"
        "Then reopen terminal and run: jx --update"
    )


def _ensure_git_available_or_exit() -> None:
    if shutil.which("git") is not None:
        return
    print(_git_install_hint(), flush=True)
    raise SystemExit(1)


def _spec_to_repo_url(spec: str) -> str:
    text = str(spec).strip()
    if text.startswith("git+"):
        return text[4:]
    return text


def _run_git(args: List[str], *, cwd: Optional[str] = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def _resolve_installed_package_dir() -> Optional[Path]:
    # 1) Namespace/package-spec path (works for janusx without __init__.py).
    try:
        spec = importlib.util.find_spec("janusx")
        if spec is not None and spec.submodule_search_locations is not None:
            for loc in spec.submodule_search_locations:
                p = Path(str(loc)).resolve()
                if p.exists() and p.is_dir():
                    return p
    except Exception:
        pass

    # 2) Distribution metadata location fallback.
    try:
        dist = importlib_metadata.distribution("janusx")
        candidate = Path(dist.locate_file("janusx")).resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate
    except Exception:
        pass

    # 3) Traditional package __file__ fallback.
    try:
        import janusx as _janusx  # type: ignore
        pkg_file = getattr(_janusx, "__file__", None)
        if pkg_file:
            p = Path(str(pkg_file)).resolve().parent
            if p.exists() and p.is_dir():
                return p
    except Exception:
        pass
    return None


def _get_local_commit_marker(pkg_dir: Optional[Path]) -> Optional[str]:
    if pkg_dir is None:
        return None
    marker = pkg_dir / _FAST_COMMIT_MARKER
    if not marker.exists():
        return None
    try:
        text = marker.read_text(encoding="utf-8").strip()
        return text if text else None
    except Exception:
        return None


def _write_local_commit_marker(pkg_dir: Optional[Path], commit: str) -> None:
    if pkg_dir is None:
        return
    marker = pkg_dir / _FAST_COMMIT_MARKER
    try:
        marker.write_text(str(commit).strip() + "\n", encoding="utf-8")
    except Exception:
        return


def _load_direct_url_json_path() -> Optional[Path]:
    try:
        dist = importlib_metadata.distribution("janusx")
        p = Path(dist.locate_file("direct_url.json"))
        if p.exists():
            return p
    except Exception:
        return None
    return None


def _get_local_commit_from_direct_url() -> Optional[str]:
    p = _load_direct_url_json_path()
    if p is None:
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        vcs = data.get("vcs_info", {})
        commit = str(vcs.get("commit_id", "")).strip()
        return commit if commit else None
    except Exception:
        return None


def _update_direct_url_commit(commit: str) -> None:
    p = _load_direct_url_json_path()
    if p is None:
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        vcs = data.get("vcs_info")
        if not isinstance(vcs, dict):
            return
        vcs["commit_id"] = str(commit).strip()
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        return


def _get_local_effective_commit(pkg_dir: Optional[Path]) -> Optional[str]:
    marker_commit = _get_local_commit_marker(pkg_dir)
    if marker_commit:
        return marker_commit
    return _get_local_commit_from_direct_url()


def _get_remote_head_commit(repo_url: str) -> Optional[str]:
    try:
        proc = _run_git(["ls-remote", repo_url, "HEAD"], timeout=60)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    line = str(proc.stdout).strip().splitlines()
    if len(line) == 0:
        return None
    fields = line[0].split()
    if len(fields) == 0:
        return None
    commit = fields[0].strip()
    if len(commit) >= 7:
        return commit
    return None


def _is_python_only_change(paths: List[str]) -> bool:
    if len(paths) == 0:
        return False
    allowed_single = {
        "README.md",
        "README.zh-CN.md",
        "README_cn.md",
        "LICENSE",
        "MANIFEST.in",
    }
    for p in paths:
        rel = str(p).strip().lstrip("./")
        if not rel:
            continue
        if rel.startswith("python/"):
            continue
        if rel in allowed_single:
            continue
        # Any Rust/build/packaging/meta change falls back to full update.
        return False
    return True


def _overlay_tree(src_dir: Path, dst_dir: Path) -> None:
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        root_path = Path(root)
        rel = root_path.relative_to(src_dir)
        out_dir = dst_dir / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            src_file = root_path / name
            dst_file = out_dir / name
            shutil.copy2(src_file, dst_file)


def _try_python_only_fast_update() -> Tuple[str, str]:
    """
    Returns:
      - ("updated", reason)
      - ("up_to_date", reason)
      - ("fallback", reason)
    """
    pkg_dir = _resolve_installed_package_dir()
    if pkg_dir is None:
        return ("fallback", "cannot resolve installed janusx package path")
    local_commit = _get_local_effective_commit(pkg_dir)
    if not local_commit:
        return ("fallback", "cannot detect installed commit metadata")

    candidates = [
        ("GitHub", _spec_to_repo_url(_GITHUB_SPEC)),
        ("Proxy", _spec_to_repo_url(_GITHUB_PROXY_SPEC)),
    ]
    last_reason = "remote query failed"
    for label, repo_url in candidates:
        remote_commit = _get_remote_head_commit(repo_url)
        if not remote_commit:
            last_reason = f"{label} HEAD unavailable"
            continue
        if remote_commit == local_commit:
            return ("up_to_date", f"{label} HEAD == local commit")

        try:
            with tempfile.TemporaryDirectory(prefix="janusx_fastupdate_") as td:
                # Fetch both commits and diff changed files.
                init = _run_git(["init"], cwd=td, timeout=60)
                if init.returncode != 0:
                    last_reason = f"{label} git init failed"
                    continue
                remote_add = _run_git(["remote", "add", "origin", repo_url], cwd=td, timeout=60)
                if remote_add.returncode != 0:
                    last_reason = f"{label} git remote add failed"
                    continue
                fetch_old = _run_git(["fetch", "--depth=1", "origin", local_commit], cwd=td, timeout=120)
                if fetch_old.returncode != 0:
                    last_reason = f"{label} cannot fetch local commit"
                    continue
                fetch_new = _run_git(["fetch", "--depth=1", "origin", remote_commit], cwd=td, timeout=120)
                if fetch_new.returncode != 0:
                    last_reason = f"{label} cannot fetch remote commit"
                    continue
                diff = _run_git(["diff", "--name-only", local_commit, remote_commit], cwd=td, timeout=120)
                if diff.returncode != 0:
                    last_reason = f"{label} git diff failed"
                    continue
                changed_files = [x.strip() for x in str(diff.stdout).splitlines() if x.strip()]
                if not _is_python_only_change(changed_files):
                    return ("fallback", f"{label} detected non-python changes")

                checkout = _run_git(["checkout", "--detach", remote_commit], cwd=td, timeout=120)
                if checkout.returncode != 0:
                    last_reason = f"{label} checkout failed"
                    continue
                src_python = Path(td) / "python" / "janusx"
                if not src_python.exists():
                    last_reason = f"{label} python/janusx not found in repo"
                    continue
                _overlay_tree(src_python, pkg_dir)
                _write_local_commit_marker(pkg_dir, remote_commit)
                _update_direct_url_commit(remote_commit)
                return ("updated", f"{label} python-only overlay applied")
        except Exception as e:
            last_reason = f"{label} fast path error: {e}"
            continue

    return ("fallback", last_reason)


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
    """
    Split stage2 internal args from user-facing update args.
    Returns: (is_stage2, parent_pid, user_args)
    """
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
        del args[idx:idx + 2]
    else:
        del args[idx]
    return True, parent_pid, args


def _is_force_reinstall_requested(user_args: List[str]) -> bool:
    return any(str(x).strip().lower() in _FORCE_FLAGS for x in user_args)


def _maybe_spawn_windows_stage2(user_args: List[str], *, is_stage2: bool) -> bool:
    """
    Windows self-update workaround:
    the currently running jx.exe cannot be replaced by pip.
    Spawn a stage-2 updater process that waits for this process to exit.
    """
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
    if len(user_args) > 0:
        cmd.extend(user_args)
    try:
        # Start updater in a separate console so users can still see progress
        # and elapsed time, while current jx process exits to release jx.exe lock.
        if hasattr(subprocess, "CREATE_NEW_CONSOLE"):
            subprocess.Popen(
                cmd,
                close_fds=True,
                creationflags=int(subprocess.CREATE_NEW_CONSOLE),  # type: ignore[arg-type]
            )
            print("Windows update launcher started in a new console window (progress shown there).")
        else:
            subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            print("Windows update launcher started in background. This terminal will return immediately.")
    except Exception:
        return False
    return True


def _print_windows_stage2_exit_hint() -> None:
    if os.name != "nt":
        return
    print("Update finished. Returning to terminal prompt...", flush=True)
    print("", flush=True)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass


def main() -> None:
    is_stage2, parent_pid, user_args = _split_stage2_and_user_args(sys.argv[1:])
    force_reinstall = _is_force_reinstall_requested(user_args)
    _ensure_git_available_or_exit()

    if _maybe_spawn_windows_stage2(user_args, is_stage2=is_stage2):
        return
    if is_stage2:
        _wait_for_parent_exit_on_windows(parent_pid, timeout_seconds=180)

    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())
    if os.name == "nt" and is_stage2:
        # Avoid dynamic terminal control sequences in stage2 updater process on
        # Windows cmd/PowerShell; they can leave prompt redraw in a bad state.
        use_spinner = False
    if not force_reinstall:
        with CliStatus("Checking fast-update path...", enabled=use_spinner) as task:
            fast_status, fast_reason = _try_python_only_fast_update()
            if fast_status == "updated":
                task.complete("JanusX fast update completed (Python-only).")
            elif fast_status == "up_to_date":
                task.complete("JanusX is already up to date.")
        if fast_status in {"updated", "up_to_date"}:
            if is_stage2:
                _print_windows_stage2_exit_hint()
            return
        print(f"Fast Python-only update skipped: {fast_reason}. Falling back to full pip update...")

    with CliStatus("Updating...", enabled=use_spinner) as task:
        direct = _run_update(_GITHUB_SPEC, force_reinstall=force_reinstall)
        if direct.returncode == 0:
            task.complete("JanusX update completed.")
    if direct.returncode == 0:
        if is_stage2:
            _print_windows_stage2_exit_hint()
        return

    if _looks_like_timeout(direct.stdout):
        print("Direct GitHub update timed out, retrying with proxy...")
    else:
        print("Direct GitHub update failed, retrying with proxy...")

    with CliStatus("Updating via proxy...", enabled=use_spinner) as task:
        proxied = _run_update(_GITHUB_PROXY_SPEC, force_reinstall=force_reinstall)
        if proxied.returncode == 0:
            task.complete("JanusX update completed (via proxy).")
    if proxied.returncode == 0:
        if is_stage2:
            _print_windows_stage2_exit_hint()
        return

    _print_failure("GitHub attempt failed.", direct)
    _print_failure("Proxy retry failed.", proxied)
    if is_stage2:
        _print_windows_stage2_exit_hint()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
