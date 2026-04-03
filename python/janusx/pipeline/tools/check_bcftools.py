from __future__ import annotations

from ._probe import ProbeResult, probe_command, run_probe_cli


def probe() -> ProbeResult:
    return probe_command(
        name="bcftools",
        commands=[["bcftools", "--version"], ["bcftools", "-v"]],
        expected_tokens=("bcftools",),
    )


if __name__ == "__main__":
    raise SystemExit(run_probe_cli(probe))

