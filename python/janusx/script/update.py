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
        "--default-timeout",
        str(_PIP_TIMEOUT_SECONDS),
        spec,
    ]
    print(f"Running: {' '.join(cmd)}")
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


def main() -> None:
    direct = _run_update(_GITHUB_SPEC)
    if direct.stdout:
        print(direct.stdout, end="")

    if direct.returncode == 0:
        print("JanusX update completed.")
        return

    if _looks_like_timeout(direct.stdout):
        print("Direct GitHub update timed out, retrying with proxy...")
        proxied = _run_update(_GITHUB_PROXY_SPEC)
        if proxied.stdout:
            print(proxied.stdout, end="")
        if proxied.returncode == 0:
            print("JanusX update completed.")
            return
        raise subprocess.CalledProcessError(
            proxied.returncode, proxied.args, output=proxied.stdout
        )

    raise subprocess.CalledProcessError(
        direct.returncode, direct.args, output=direct.stdout
    )


if __name__ == "__main__":
    main()
