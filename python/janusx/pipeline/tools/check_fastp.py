from __future__ import annotations

from ._probe import ProbeResult, probe_command, run_probe_cli


def probe() -> ProbeResult:
    return probe_command(
        name="fastp",
        commands=[["fastp", "--version"], ["fastp", "-v"]],
        expected_tokens=("fastp",),
    )


if __name__ == "__main__":
    raise SystemExit(run_probe_cli(probe))

