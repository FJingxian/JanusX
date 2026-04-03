from __future__ import annotations

from ._probe import ProbeResult, probe_command, run_probe_cli


def probe() -> ProbeResult:
    return probe_command(
        name="hisat2-build",
        commands=[["hisat2-build", "--version"], ["hisat2-build", "-h"]],
        expected_tokens=("hisat2-build", "hisat2"),
    )


if __name__ == "__main__":
    raise SystemExit(run_probe_cli(probe))

