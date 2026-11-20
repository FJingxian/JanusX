"""Process adapter for the upstream ADAMIXTURE command line.

The actual fitting process is delegated to the installed ADAMIXTURE
executable; this module intentionally does not import its private Cython
implementation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np


class ExternalAdmixtureError(RuntimeError):
    """Raised when the ADAMIXTURE process cannot complete a run."""


def resolve_executable() -> str:
    requested = str(os.environ.get("JANUSX_ADMIXTURE_BIN", "")).strip()
    if requested:
        candidate = shutil.which(requested) or requested
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        raise ExternalAdmixtureError(
            "JANUSX_ADMIXTURE_BIN does not point to an executable: "
            f"{requested}"
        )
    candidate = shutil.which("adamixture")
    if candidate is None:
        raise ExternalAdmixtureError(
            "ADAMIXTURE backend requires the 'adamixture' executable. "
            "Install adamixture or set JANUSX_ADMIXTURE_BIN."
        )
    return str(candidate)


def run(
    *,
    genotype_path: str,
    outdir: str,
    prefix: str,
    k: int,
    threads: int,
    seed: int,
    max_iter: int,
    tol: float,
    algorithm: str = "adamem",
    device: str = "cpu",
    n_inits: int = 1,
    cv_folds: int = 0,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run ADAMIXTURE for one K and return its Q/P matrices."""

    executable = resolve_executable()
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(
        prefix=f".{prefix}.{int(k)}.adamixture.", dir=str(output_dir)
    ) as temp_dir:
        cmd = [
            executable,
            "--data_path",
            str(genotype_path),
            "--save_dir",
            str(temp_dir),
            "--name",
            "run",
            "--k",
            str(int(k)),
            "--threads",
            str(max(1, int(threads))),
            "--seed",
            str(int(seed)),
            "--device",
            str(device),
            "--algorithm",
            str(algorithm),
            "--max_iter",
            str(max(1, int(max_iter))),
            "--tol",
            str(float(tol)),
            "--n_inits",
            str(max(1, int(n_inits))),
        ]
        if int(cv_folds) >= 2:
            cmd.extend(["--cv", str(int(cv_folds))])
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as ex:
            raise ExternalAdmixtureError(
                f"Failed to start ADAMIXTURE executable {executable!r}: {ex}"
            ) from ex
        assert proc.stdout is not None
        for line in proc.stdout:
            if stream_callback is not None:
                stream_callback(line.rstrip("\n"))
        proc.stdout.close()
        return_code = int(proc.wait())
        elapsed = float(time.time() - started)

        if return_code != 0:
            raise ExternalAdmixtureError(
                f"ADAMIXTURE failed with exit code {return_code}"
            )
        q_path = Path(temp_dir) / f"run.{int(k)}.Q"
        p_path = Path(temp_dir) / f"run.{int(k)}.P"
        if not q_path.is_file() or not p_path.is_file():
            missing = ", ".join(
                str(path) for path in (q_path, p_path) if not path.is_file()
            )
            raise ExternalAdmixtureError(
                f"ADAMIXTURE completed without expected output: {missing}"
            )
        try:
            q_mat = np.loadtxt(q_path, dtype=np.float64, ndmin=2)
            p_mat = np.loadtxt(p_path, dtype=np.float32, ndmin=2)
        except Exception as ex:
            raise ExternalAdmixtureError(
                f"Failed to load ADAMIXTURE Q/P output: {ex}"
            ) from ex
    return q_mat, p_mat, float(time.time() - started)
