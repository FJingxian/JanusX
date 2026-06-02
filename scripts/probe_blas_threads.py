#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(f"NumPy is required: {exc}") from exc

try:
    from threadpoolctl import threadpool_info  # type: ignore
except Exception:  # pragma: no cover - optional dependency on some nodes
    threadpool_info = None

from janusx import janusx as _jx
from janusx.script._common.threads import (
    apply_blas_thread_env,
    detect_blas_backend,
    detect_effective_threads,
    get_rust_blas_threads,
    set_rust_blas_threads,
)


BLAS_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_MAX_THREADS",
    "MKL_NUM_THREADS",
    "MKL_MAX_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
)

SCHED_ENV_KEYS = (
    "SLURM_CPUS_PER_TASK",
    "PBS_NP",
    "LSB_DJOB_NUMPROC",
    "NSLOTS",
    "NCPUS",
)


def _parse_requests(text: str) -> list[int]:
    out: list[int] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        val = int(item)
        if val <= 0:
            raise ValueError(f"thread request must be > 0, got {val}")
        out.append(val)
    if not out:
        raise ValueError("no valid thread requests provided")
    return sorted(set(out))


def _env_snapshot(keys: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in keys:
        raw = str(os.environ.get(key, "")).strip()
        if raw:
            out[key] = raw
    return out


def _collect_threadpool_records() -> list[dict[str, Any]]:
    if threadpool_info is None:
        return []
    try:
        z = np.zeros((1, 1), dtype=np.float64)
        _ = z @ z
    except Exception:
        pass
    try:
        infos = list(threadpool_info())
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for rec in infos:
        row = {
            "user_api": rec.get("user_api"),
            "internal_api": rec.get("internal_api"),
            "num_threads": rec.get("num_threads"),
            "prefix": rec.get("prefix"),
            "threading_layer": rec.get("threading_layer"),
            "architecture": rec.get("architecture"),
            "filepath": rec.get("filepath"),
            "version": rec.get("version"),
        }
        out.append(row)
    return out


def _pick_openblas_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for rec in records:
        text = " ".join(
            str(rec.get(key, ""))
            for key in ("internal_api", "user_api", "prefix", "filepath", "version")
        ).lower()
        if "openblas" in text:
            return rec
    return None


def _probe_request(request: int) -> dict[str, Any]:
    applied = int(apply_blas_thread_env(request, set_jx_threads=False, set_max_keys=True))
    setter_ok = bool(set_rust_blas_threads(applied))
    rust_now = get_rust_blas_threads()
    records = _collect_threadpool_records()
    openblas_rec = _pick_openblas_record(records)
    return {
        "requested": int(request),
        "applied_env": int(applied),
        "setter_ok": bool(setter_ok),
        "rust_blas_threads": None if rust_now is None else int(rust_now),
        "openblas_threadpool_threads": (
            None
            if openblas_rec is None or openblas_rec.get("num_threads") is None
            else int(openblas_rec["num_threads"])
        ),
        "openblas_record": openblas_rec,
        "threadpools": records,
        "env": _env_snapshot(BLAS_ENV_KEYS),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Probe requested vs actual BLAS thread settings in the current Python/Rust runtime."
        )
    )
    ap.add_argument(
        "--requests",
        default="1,2,4,8,12,24,48",
        help="Comma-separated thread requests to test.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable report.",
    )
    ap.add_argument(
        "--show-all-threadpools",
        action="store_true",
        help="Include every threadpoolctl record in text output.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requests = _parse_requests(args.requests)

    header = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "effective_threads": int(detect_effective_threads()),
        "python_blas_backend": str(detect_blas_backend()),
        "rust_sgemm_backend": str(_jx.rust_sgemm_backend()),
        "rust_eigh_lapack_backend": str(_jx.rust_eigh_lapack_backend()),
        "scheduler_env": _env_snapshot(SCHED_ENV_KEYS),
        "initial_env": _env_snapshot(BLAS_ENV_KEYS),
        "initial_rust_blas_threads": get_rust_blas_threads(),
    }
    probes = [_probe_request(req) for req in requests]

    if bool(args.json):
        payload = {
            "header": header,
            "probes": probes,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"host={header['host']}")
    print(f"platform={header['platform']}")
    print(f"python={header['python']}")
    print(f"effective_threads={header['effective_threads']}")
    print(f"python_blas_backend={header['python_blas_backend']}")
    print(f"rust_sgemm_backend={header['rust_sgemm_backend']}")
    print(f"rust_eigh_lapack_backend={header['rust_eigh_lapack_backend']}")
    if header["scheduler_env"]:
        print("scheduler_env:")
        for key, val in header["scheduler_env"].items():
            print(f"  {key}={val}")
    if header["initial_env"]:
        print("initial_env:")
        for key, val in header["initial_env"].items():
            print(f"  {key}={val}")
    print(f"initial_rust_blas_threads={header['initial_rust_blas_threads']}")
    print()

    for probe in probes:
        print(
            "request={requested} applied_env={applied_env} setter_ok={setter_ok} "
            "rust_blas_threads={rust_blas_threads} openblas_threadpool_threads={openblas_threadpool_threads}".format(
                **probe
            )
        )
        for key, val in probe["env"].items():
            print(f"  {key}={val}")
        if probe["openblas_record"] is not None:
            rec = probe["openblas_record"]
            print(
                "  openblas_record="
                f"user_api={rec.get('user_api')} internal_api={rec.get('internal_api')} "
                f"num_threads={rec.get('num_threads')} prefix={rec.get('prefix')} "
                f"threading_layer={rec.get('threading_layer')} architecture={rec.get('architecture')}"
            )
        elif threadpool_info is None:
            print("  openblas_record=threadpoolctl unavailable")
        else:
            print("  openblas_record=not found")
        if bool(args.show_all_threadpools):
            for idx, rec in enumerate(probe["threadpools"], start=1):
                print(
                    f"  threadpool[{idx}] "
                    f"user_api={rec.get('user_api')} internal_api={rec.get('internal_api')} "
                    f"num_threads={rec.get('num_threads')} prefix={rec.get('prefix')} "
                    f"threading_layer={rec.get('threading_layer')} architecture={rec.get('architecture')}"
                )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
