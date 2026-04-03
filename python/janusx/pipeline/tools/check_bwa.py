from __future__ import annotations

from ._probe import ProbeResult, probe_command, run_probe_cli


def probe() -> ProbeResult:
    return probe_command(
        name="bwa",
        commands=[
            ["bwa", "--version"],
            ["bwa", "mem"],
            ["bwa-mem2", "version"],
            ["bwa-mem2", "--version"],
        ],
        expected_tokens=("bwa", "mem2"),
    )


if __name__ == "__main__":
    raise SystemExit(run_probe_cli(probe))

