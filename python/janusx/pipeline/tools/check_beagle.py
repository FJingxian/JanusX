from __future__ import annotations

from ._probe import ProbeResult, probe_command, run_probe_cli


def probe() -> ProbeResult:
    return probe_command(
        name="beagle",
        commands=[
            ["beagle"],
            ["beagle", "help"],
            ["beagle", "--help"],
            ["beagle", "-h"],
        ],
        expected_tokens=(
            "beagle",
            "usage",
            "illegalargumentexception",
            "missing delimiter character",
        ),
    )


if __name__ == "__main__":
    raise SystemExit(run_probe_cli(probe))
