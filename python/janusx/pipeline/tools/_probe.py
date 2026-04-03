from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Sequence
import shlex
import shutil
import subprocess


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    command: str
    detail: str


def _clip_line(text: str, limit: int = 56) -> str:
    raw = str(os.environ.get("JANUSX_CHECK_DETAIL_MAX", "")).strip()
    try:
        max_len = int(raw) if raw else int(limit)
    except Exception:
        max_len = int(limit)
    max_len = max(24, max_len)
    first = text.strip().splitlines()[0] if text.strip() else ""
    line = " ".join(first.replace("\t", " ").split())
    if len(line) <= max_len:
        return line
    return line[: max_len - 3] + "..."


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    hay = text.lower()
    return any(needle.lower() in hay for needle in needles if needle)


def _format_cmd(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(token)) for token in argv)


def probe_command(
    name: str,
    commands: Sequence[Sequence[str]],
    expected_tokens: Sequence[str] = (),
    timeout_sec: float = 6.0,
) -> ProbeResult:
    binaries: list[str] = []
    missing_bins: list[str] = []
    last_error = "probe failed"

    for argv in commands:
        if not argv:
            continue
        bin_name = str(argv[0]).strip()
        if not bin_name:
            continue

        if bin_name not in binaries:
            binaries.append(bin_name)

        bin_path = shutil.which(bin_name)
        if bin_path is None:
            if bin_name not in missing_bins:
                missing_bins.append(bin_name)
            continue

        cmd = [str(x) for x in argv]
        cmd_text = _format_cmd(cmd)
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            last_error = f"timeout (> {timeout_sec:.1f}s) running `{cmd_text}`"
            continue
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = f"error running `{cmd_text}`: {exc}"
            continue

        merged = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        summary = _clip_line(merged)
        if proc.returncode == 0 or _contains_any(merged, expected_tokens):
            detail = summary or f"found: {bin_path}"
            return ProbeResult(name=name, ok=True, command=cmd_text, detail=detail)

        if summary:
            last_error = f"exit={proc.returncode}; {summary}"
        else:
            last_error = f"exit={proc.returncode}"

    if binaries and len(missing_bins) == len(binaries):
        return ProbeResult(
            name=name,
            ok=False,
            command="",
            detail=f"not found in PATH ({', '.join(missing_bins)})",
        )
    return ProbeResult(name=name, ok=False, command="", detail=last_error)


def run_probe_cli(probe_fn: Callable[[], ProbeResult]) -> int:
    result = probe_fn()
    marker = "✓" if result.ok else "✗"
    cmd = f" ({result.command})" if result.command else ""
    if result.detail:
        print(f"{marker} {result.name}{cmd}: {result.detail}")
    else:
        print(f"{marker} {result.name}{cmd}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    print("Use this module via specific checks, e.g. `python -m janusx.pipeline.tools.check_fastp`.")
    raise SystemExit(2)
