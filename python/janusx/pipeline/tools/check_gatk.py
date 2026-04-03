from __future__ import annotations

from ._probe import ProbeResult, probe_command, run_probe_cli


def probe() -> ProbeResult:
    return probe_command(
        name="gatk",
        commands=[["gatk", "--version"], ["gatk", "--help"]],
        expected_tokens=("gatk",),
    )


if __name__ == "__main__":
    raise SystemExit(run_probe_cli(probe))

