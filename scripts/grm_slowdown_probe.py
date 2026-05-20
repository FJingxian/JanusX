#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_CRITERION = REPO_ROOT / "target" / "criterion"
_STAGE_RE = re.compile(
    r"GRM stream timing:\s*decode=(?P<decode>[0-9.]+)s.*?"
    r"gemm=(?P<gemm>[0-9.]+)s.*?"
    r"other=(?P<other>[0-9.]+)s.*?"
    r"total=(?P<total>[0-9.]+)s",
    flags=re.IGNORECASE,
)
_BENCH_RE = re.compile(
    r"(?P<kernel>full_packed|accum_syrk|accum_gemm)_(?P<dtype>f32|f64)_t(?P<threads>\d+)"
)


def _print(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_conda_sh() -> Path:
    candidates: list[Path] = []
    conda_exe = os.environ.get("CONDA_EXE", "").strip()
    if conda_exe:
        candidates.append(Path(conda_exe).resolve().parents[1] / "etc" / "profile.d" / "conda.sh")
    conda_which = shutil.which("conda")
    if conda_which:
        candidates.append(Path(conda_which).resolve().parents[1] / "etc" / "profile.d" / "conda.sh")
    candidates.append(Path.home() / "miniconda3" / "etc" / "profile.d" / "conda.sh")
    candidates.append(Path.home() / "anaconda3" / "etc" / "profile.d" / "conda.sh")
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError("Unable to locate conda.sh. Set CONDA_EXE or install conda in a standard location.")


def _bash_with_conda(env_name: str, command: str) -> list[str]:
    conda_sh = _find_conda_sh()
    cmd = (
        f"source {shlex_quote(str(conda_sh))} && "
        f"conda activate {shlex_quote(env_name)} && "
        f"{command}"
    )
    return ["bash", "-lc", cmd]


def shlex_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if log_path is not None:
        log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
        )
    return proc


def collect_env_info(env_name: str) -> dict[str, Any]:
    py = r"""
import json
import hashlib
from pathlib import Path
import janusx
import janusx.janusx as jx

pkg_dir = Path(janusx.__file__).resolve().parent
grm_py = pkg_dir / "script" / "grm.py"
native = Path(jx.__file__).resolve()
grm_text = grm_py.read_text(encoding="utf-8", errors="replace")
raw_openblas = str(__import__("os").environ.get("JX_OPENBLAS_LIB_PATH", "")).strip()
openblas = Path(raw_openblas).resolve() if raw_openblas else None

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

out = {
    "env": __import__("os").environ.get("CONDA_DEFAULT_ENV", ""),
    "conda_prefix": __import__("os").environ.get("CONDA_PREFIX", ""),
    "grm_py": str(grm_py),
    "native": str(native),
    "openblas": (str(openblas) if openblas is not None else ""),
    "uses_stream_f32": "_grm_stream_bed_f32" in grm_text,
    "uses_stream_f64": "_grm_stream_bed_f64" in grm_text,
    "npy_float64": "dtype=np.float64" in grm_text,
    "native_sha256": sha256(native),
    "openblas_sha256": (sha256(openblas) if openblas is not None and openblas.is_file() else ""),
    "rust_sgemm_backend": str(jx.rust_sgemm_backend()).strip(),
    "rust_eigh_lapack_backend": str(jx.rust_eigh_lapack_backend()).strip(),
}
print(json.dumps(out))
"""
    proc = run_command(
        _bash_with_conda(env_name, f"python - <<'PY'\n{py}\nPY"),
        cwd=Path.home(),
    )
    return json.loads(proc.stdout.strip())


def simulate_bed(sim_env: str, outdir: Path, prefix: str, nsnp_k: int, n_individuals: int, seed: int) -> Path:
    out_prefix = outdir / prefix
    cmd = (
        f"jx sim {int(nsnp_k)} {int(n_individuals)} {shlex_quote(str(out_prefix))} "
        f"--seed {int(seed)}"
    )
    run_command(_bash_with_conda(sim_env, cmd), cwd=REPO_ROOT, log_path=outdir / f"{prefix}.sim.log")
    for ext in ("bed", "bim", "fam"):
        path = out_prefix.with_suffix(f".{ext}")
        if not path.is_file():
            raise FileNotFoundError(f"Simulation did not produce {path}")
    return out_prefix


def parse_stage_timing(log_text: str) -> dict[str, float]:
    m = _STAGE_RE.search(log_text)
    if not m:
        return {"decode": float("nan"), "gemm": float("nan"), "other": float("nan"), "total": float("nan")}
    return {k: float(v) for k, v in m.groupdict().items()}


def parse_grm_backend(log_text: str) -> str:
    m = re.search(
        r"GRM \(Effective SNPs:\s*\d+,\s*([^)]+)\)\s*\.\.\.Finished",
        log_text,
        flags=re.IGNORECASE,
    )
    if m:
        return str(m.group(1)).strip()
    return "unknown"


def run_grm_once(env_name: str, bfile_prefix: Path, outdir: Path, prefix: str, threads: int) -> dict[str, Any]:
    log_path = outdir / f"{prefix}.log"
    env = os.environ.copy()
    env["JX_GRM_STREAM_STAGE_TIMING"] = "1"
    cmd = (
        f"jx grm -bfile {shlex_quote(str(bfile_prefix))} -t {int(threads)} "
        f"-o {shlex_quote(str(outdir))} -prefix {shlex_quote(prefix)} -npy"
    )
    t0 = time.perf_counter()
    proc = run_command(
        _bash_with_conda(env_name, cmd),
        cwd=REPO_ROOT,
        env=env,
        log_path=log_path,
    )
    elapsed = time.perf_counter() - t0
    log_text = proc.stdout + proc.stderr
    return {
        "env": env_name,
        "threads": int(threads),
        "elapsed": elapsed,
        "backend": parse_grm_backend(log_text),
        "stage": parse_stage_timing(log_text),
        "log_path": str(log_path),
    }


def resolve_cargo_blas_backend(requested: str) -> str:
    norm = requested.strip().lower()
    if norm in ("", "auto"):
        return "accelerate" if sys.platform == "darwin" else "openblas"
    if norm == "accelerate":
        if sys.platform != "darwin":
            raise ValueError("cargo BLAS backend 'accelerate' is only supported on macOS")
        return norm
    if norm == "openblas":
        return norm
    raise ValueError(f"Unsupported cargo BLAS backend: {requested}")


def _find_env_libpython(cargo_env: str) -> Path:
    py = r"""
from pathlib import Path
import sys
lib = Path(sys.executable).resolve().parents[1] / "lib" / f"libpython{sys.version_info[0]}.{sys.version_info[1]}.dylib"
print(str(lib))
"""
    proc = run_command(
        _bash_with_conda(cargo_env, f"python - <<'PY'\n{py}\nPY"),
        cwd=REPO_ROOT,
    )
    path = Path(proc.stdout.strip())
    if not path.is_file():
        raise FileNotFoundError(f"libpython not found for cargo env {cargo_env}: {path}")
    return path


def run_cargo_bench(
    cargo_env: str,
    nsnp_k: int,
    n_individuals: int,
    seed: int,
    sample_size: int,
    threads_max: int,
    blas_backend: str,
) -> str:
    actual_backend = resolve_cargo_blas_backend(blas_backend)
    bench_env = os.environ.copy()
    bench_env["JX_GRM_DTYPE_BENCH_NSNP_K"] = str(int(nsnp_k))
    bench_env["JX_GRM_DTYPE_BENCH_NIND"] = str(int(n_individuals))
    bench_env["JX_GRM_DTYPE_BENCH_SEED"] = str(int(seed))
    bench_env["JX_GRM_DTYPE_BENCH_SAMPLE_SIZE"] = str(int(sample_size))
    bench_env["JX_GRM_DTYPE_BENCH_THREADS"] = f"1,{max(1, int(threads_max))}"
    bench_env["JX_RUST_BLAS_BACKEND"] = actual_backend
    if TARGET_CRITERION.exists():
        shutil.rmtree(TARGET_CRITERION)
    cargo_args = "cargo bench --bench grm_dtype --no-run --no-default-features"
    if actual_backend == "openblas":
        cargo_args += " --features blas-openblas"
    build_command = (
        "export PYO3_PYTHON=\"$CONDA_PREFIX/bin/python\" && "
        "export DYLD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:${DYLD_LIBRARY_PATH:-}\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"$CONDA_PREFIX/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}\" && "
        f"{cargo_args}"
    )
    proc = run_command(_bash_with_conda(cargo_env, build_command), cwd=REPO_ROOT, env=bench_env)
    text = proc.stdout + proc.stderr
    m = re.search(r"Executable .* \(([^)]+)\)", text)
    if not m:
        raise RuntimeError(f"Unable to locate bench executable from cargo output:\n{text}")
    bench_bin = Path(m.group(1)).resolve()
    if not bench_bin.is_file():
        raise FileNotFoundError(f"Bench executable missing: {bench_bin}")

    libpython = _find_env_libpython(cargo_env)
    link_path = bench_bin.parent / libpython.name
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() != libpython.resolve():
            link_path.unlink()
            link_path.symlink_to(libpython)
    else:
        link_path.symlink_to(libpython)

    bench_env["DYLD_LIBRARY_PATH"] = f"{libpython.parent}:{bench_env.get('DYLD_LIBRARY_PATH', '')}"
    bench_env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{libpython.parent}:{bench_env.get('DYLD_FALLBACK_LIBRARY_PATH', '')}"
    run_command(
        [str(bench_bin), "--noplot", "--bench"],
        cwd=REPO_ROOT,
        env=bench_env,
    )
    return actual_backend


def collect_criterion_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not TARGET_CRITERION.is_dir():
        return rows
    for est_path in TARGET_CRITERION.glob("*/new/estimates.json"):
        name = est_path.parent.parent.name
        m = _BENCH_RE.fullmatch(name)
        if not m:
            continue
        payload = json.loads(est_path.read_text(encoding="utf-8"))
        mean_ns = float(payload["mean"]["point_estimate"])
        rows.append(
            {
                "name": name,
                "kernel": m.group("kernel"),
                "dtype": m.group("dtype"),
                "threads": int(m.group("threads")),
                "mean_sec": mean_ns / 1e9,
            }
        )
    rows.sort(key=lambda r: (r["kernel"], r["threads"], r["dtype"]))
    return rows


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    return "\n".join([line, sep] + body)


def print_env_summary(info_rows: list[dict[str, Any]]) -> None:
    rows = [
        [
            row["env"],
            str(row["threads"]),
            f"{row['elapsed']:.3f}s",
            f"{row['stage']['decode']:.3f}s",
            f"{row['stage']['gemm']:.3f}s",
            f"{row['stage']['other']:.3f}s",
            row["backend"],
        ]
        for row in info_rows
    ]
    _print("\nEnvironment reproduction")
    _print(
        format_table(
            ["env", "threads", "wall", "decode", "gemm", "other", "backend"],
            rows,
        )
    )


def print_binary_summary(info_rows: list[dict[str, Any]]) -> None:
    rows = [
        [
            info["env"],
            "f32" if info["uses_stream_f32"] else ("f64" if info["uses_stream_f64"] else "unknown"),
            "yes" if info["npy_float64"] else "no",
            info["rust_sgemm_backend"],
            info["rust_eigh_lapack_backend"],
            info["native_sha256"][:12],
            info["openblas_sha256"][:12],
        ]
        for info in info_rows
    ]
    _print("\nInstalled binary identity")
    _print(
        format_table(
            ["env", "stream_path", "npy_f64", "sgemm", "eigh", "native_sha", "openblas_sha"],
            rows,
        )
    )


def print_criterion_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        _print("\nCriterion results not found.")
        return
    grouped: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        grouped.setdefault((row["kernel"], row["threads"]), {})[row["dtype"]] = row["mean_sec"]
    out_rows: list[list[str]] = []
    for (kernel, threads), vals in sorted(grouped.items()):
        f32 = vals.get("f32", float("nan"))
        f64 = vals.get("f64", float("nan"))
        ratio = f64 / f32 if f32 and f32 == f32 and f64 == f64 else float("nan")
        out_rows.append(
            [
                kernel,
                str(int(threads)),
                f"{f32:.3f}s",
                f"{f64:.3f}s",
                f"{ratio:.3f}x" if ratio == ratio else "nan",
            ]
        )
    _print("\nCriterion dtype isolation")
    _print(format_table(["kernel", "threads", "f32", "f64", "f64/f32"], out_rows))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="grm_slowdown_probe.py",
        description="Reproduce JanusX GRM slowdown across conda environments and summarize Criterion dtype benchmarks.",
    )
    p.add_argument("--env-old", default="janusx", help="Old/reference conda environment.")
    p.add_argument("--env-new", default="jxfu", help="New/regressed conda environment.")
    p.add_argument("--sim-env", default="jxfu", help="Conda environment used to generate the synthetic BED once.")
    p.add_argument("--outdir", default=str(REPO_ROOT / "tmp_grm_probe"), help="Output directory.")
    p.add_argument("--prefix", default="grm_probe", help="Prefix for the synthetic dataset.")
    p.add_argument("--nsnp-k", type=int, default=250, help="Thousands of SNPs to simulate.")
    p.add_argument("--n-individuals", type=int, default=5000, help="Number of individuals.")
    p.add_argument("--seed", type=int, default=20260520, help="Simulation / bench seed.")
    p.add_argument("--threads-max", type=int, default=8, help="All-core comparison target.")
    p.add_argument("--criterion-sample-size", type=int, default=10, help="Criterion sample size.")
    p.add_argument("--cargo-env", default="jxfu", help="Conda environment used to run cargo bench.")
    p.add_argument(
        "--cargo-blas-backend",
        default="auto",
        help="BLAS backend for the Rust dtype bench: auto, accelerate, or openblas.",
    )
    p.add_argument("--skip-env-runs", action="store_true", help="Skip end-to-end conda environment reproduction.")
    p.add_argument("--skip-cargo-bench", action="store_true", help="Skip cargo Criterion benchmark.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    binary_info = [collect_env_info(args.env_old), collect_env_info(args.env_new)]
    print_binary_summary(binary_info)

    env_rows: list[dict[str, Any]] = []
    if not args.skip_env_runs:
        dataset_prefix = simulate_bed(
            args.sim_env,
            outdir,
            args.prefix,
            int(args.nsnp_k),
            int(args.n_individuals),
            int(args.seed),
        )
        for env_name in (args.env_old, args.env_new):
            for threads in (1, int(args.threads_max)):
                run_prefix = f"{args.prefix}.{env_name}.t{threads}"
                env_rows.append(run_grm_once(env_name, dataset_prefix, outdir, run_prefix, threads))
        print_env_summary(env_rows)

    if not args.skip_cargo_bench:
        cargo_backend = run_cargo_bench(
            args.cargo_env,
            int(args.nsnp_k),
            int(args.n_individuals),
            int(args.seed),
            int(args.criterion_sample_size),
            int(args.threads_max),
            str(args.cargo_blas_backend),
        )
        _print(f"\nCargo bench BLAS backend: {cargo_backend}")
        print_criterion_summary(collect_criterion_rows())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
