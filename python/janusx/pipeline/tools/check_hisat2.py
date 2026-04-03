from __future__ import annotations

from ._probe import ProbeResult, probe_command, run_probe_cli


def probe() -> ProbeResult:
    return probe_command(
        name="hisat2",
        commands=[["hisat2", "--version"], ["hisat2", "-h"]],
        expected_tokens=("hisat2",),
    )


if __name__ == "__main__":
    raise SystemExit(run_probe_cli(probe))

