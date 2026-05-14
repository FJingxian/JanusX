#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT_DIR = project_root()


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int, min_value: int | None = None) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        fail(f"{name} must be an integer, got {raw!r}")
    if min_value is not None and value < min_value:
        fail(f"{name} must be >= {min_value}, got {value}")
    return value


def env_float(name: str, default: float, min_value: float | None = None) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        fail(f"{name} must be a float, got {raw!r}")
    if min_value is not None and value < min_value:
        fail(f"{name} must be >= {min_value}, got {value}")
    return value


def env_truthy(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _parse_positive_int(raw: object) -> int | None:
    try:
        value = int(str(raw).strip())
    except Exception:
        return None
    return value if value > 0 else None


def _parse_positive_env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return _parse_positive_int(raw)


def _detect_cgroup_cpu_quota() -> int | None:
    try:
        quota_path = "/sys/fs/cgroup/cpu.max"
        if os.path.isfile(quota_path):
            raw = Path(quota_path).read_text(encoding="utf-8").strip().split()
            if len(raw) >= 2 and raw[0] != "max":
                quota = int(raw[0])
                period = int(raw[1])
                if quota > 0 and period > 0:
                    return max(1, int(math.ceil(float(quota) / float(period))))
    except Exception:
        pass

    try:
        quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
        period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
        if os.path.isfile(quota_path) and os.path.isfile(period_path):
            quota = int(Path(quota_path).read_text(encoding="utf-8").strip())
            period = int(Path(period_path).read_text(encoding="utf-8").strip())
            if quota > 0 and period > 0:
                return max(1, int(math.ceil(float(quota) / float(period))))
    except Exception:
        pass
    return None


def _detect_joblib_cpu_count() -> int | None:
    try:
        from joblib import cpu_count as joblib_cpu_count

        return _parse_positive_int(joblib_cpu_count())
    except Exception:
        return None


def detect_effective_threads(override_threads: int = 0) -> tuple[int, str]:
    if override_threads > 0:
        return int(override_threads), "override (--bench-threads / JX_GGVAL_BENCH_THREADS)"

    scheduler_envs = [
        "SLURM_CPUS_PER_TASK",
        "PBS_NP",
        "LSB_DJOB_NUMPROC",
        "NSLOTS",
        "NCPUS",
    ]
    detected: int | None = None
    source = ""
    for name in scheduler_envs:
        value = _parse_positive_env_int(name)
        if value is not None:
            detected = value
            source = f"scheduler:{name}"
            break

    if detected is None:
        try:
            if hasattr(os, "sched_getaffinity"):
                affinity = os.sched_getaffinity(0)  # type: ignore[attr-defined]
                if affinity is not None and len(affinity) > 0:
                    detected = int(len(affinity))
                    source = "sched_getaffinity"
        except Exception:
            pass

    if detected is None:
        quota_threads = _detect_cgroup_cpu_quota()
        if quota_threads is not None:
            detected = quota_threads
            source = "cgroup"

    if detected is None:
        joblib_threads = _detect_joblib_cpu_count()
        if joblib_threads is not None:
            detected = joblib_threads
            source = "joblib.cpu_count"

    if detected is None:
        detected = int(os.cpu_count() or 1)
        source = "os.cpu_count"

    cap_envs = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    caps: list[tuple[str, int]] = []
    for name in cap_envs:
        value = _parse_positive_env_int(name)
        if value is not None:
            caps.append((name, value))
    if caps:
        cap_name, cap_value = min(caps, key=lambda item: item[1])
        if cap_value < detected:
            detected = cap_value
            source = f"{source}, capped:{cap_name}"

    return max(1, int(detected)), source


def _default_he_thread_policy() -> str:
    # Benchmarks on macOS/Accelerate consistently favored keeping BLAS serial
    # and letting Rayon own the outer HE parallelism.
    if sys.platform == "darwin":
        return "rayon_parallel_blas_serial"
    # Keep the current default unchanged on other platforms until Linux/Windows
    # benchmark results settle.
    return "rayon_parallel_blas_serial"


_HE_THREAD_POLICY_DEFAULT = _default_he_thread_policy()
_HE_THREAD_POLICY_SWEEP_DEFAULT = [
    "rayon_parallel_blas_serial",
    "blas_parallel_rayon_serial",
    "split_half",
]


def _default_pcg_thread_policy() -> str:
    # Preserve the current rrBLUP-PCG default: BLAS serial, Rayon owns the
    # outer loop. Benchmark sweeps can still compare split_half explicitly.
    return "rayon_parallel_blas_serial"


_PCG_THREAD_POLICY_DEFAULT = _default_pcg_thread_policy()
_PCG_THREAD_POLICY_SWEEP_DEFAULT = [
    "rayon_parallel_blas_serial",
    "blas_parallel_rayon_serial",
    "split_half",
]
_ROWMAJOR_KERNEL_DEFAULT = "blas"


def _default_rowmajor_kernel_compare_policies() -> list[str]:
    # Benchmarks should compare both kernels on every platform so Windows runs
    # can surface whether the slowdown/hang is kernel-specific or more general.
    return ["blas", "rayon"]


_ROWMAJOR_KERNEL_COMPARE_DEFAULT = _default_rowmajor_kernel_compare_policies()


def _normalize_he_thread_policy_name(
    raw: object,
    *,
    default: str = _HE_THREAD_POLICY_DEFAULT,
) -> str:
    txt = str(raw).strip().lower()
    if txt == "":
        txt = str(default).strip().lower()
    norm = re.sub(r"[^a-z0-9]+", "", txt)
    if norm in {
        "",
        "default",
        "auto",
        "rayon",
        "rayonparallelblasserial",
        "rayononly",
    }:
        return "rayon_parallel_blas_serial"
    if norm in {"blas", "blasparallelrayonserial", "blasonly"}:
        return "blas_parallel_rayon_serial"
    if norm in {"split", "splithalf", "halfhalf", "balanced"}:
        return "split_half"
    if norm in {"serial", "single", "singlethread", "one"}:
        return "serial"
    if norm in {"compare", "all", "sweep"}:
        return "compare"
    raise ValueError(
        "Unsupported HE thread policy "
        f"{raw!r}. Use one of: rayon_parallel_blas_serial, "
        "blas_parallel_rayon_serial, split_half."
    )


def _normalize_rowmajor_kernel_name(
    raw: object,
    *,
    default: str = _ROWMAJOR_KERNEL_DEFAULT,
) -> str:
    txt = str(raw).strip().lower()
    if txt == "":
        txt = str(default).strip().lower()
    norm = re.sub(r"[^a-z0-9]+", "", txt)
    if norm in {"", "default", "auto", "blas", "gemm", "serial", "macos", "macoslike"}:
        return "blas"
    if norm in {"rayon", "parallel", "custom", "rowmajor", "rayonrowmajor"}:
        return "rayon"
    if norm in {"compare", "all", "sweep", "both"}:
        return "compare"
    raise ValueError(
        "Unsupported row-major kernel strategy "
        f"{raw!r}. Use one of: blas, rayon."
    )


def _expand_rowmajor_kernel_names(raw: object) -> list[str]:
    txt = str(raw).strip()
    if txt == "":
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[\s,;:+|]+", txt):
        token = str(token).strip()
        if token == "":
            continue
        name = _normalize_rowmajor_kernel_name(token)
        names = (
            list(_ROWMAJOR_KERNEL_COMPARE_DEFAULT)
            if name == "compare"
            else [name]
        )
        for item in names:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


def _rowmajor_kernel_label(name: object) -> str:
    norm = _normalize_rowmajor_kernel_name(name)
    if norm == "blas":
        return "macos_like_blas_kernel"
    if norm == "rayon":
        return "rayon_rowmajor_kernel"
    return str(norm)


def _expand_he_thread_policy_names(raw: object) -> list[str]:
    txt = str(raw).strip()
    if txt == "":
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[\s,;:+|]+", txt):
        token = str(token).strip()
        if token == "":
            continue
        policy = _normalize_he_thread_policy_name(token)
        names = (
            list(_HE_THREAD_POLICY_SWEEP_DEFAULT)
            if policy == "compare"
            else [policy]
        )
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


def _resolve_he_thread_policy(policy_name: object, total_threads: int) -> dict[str, object]:
    total = max(1, int(total_threads))
    policy = _normalize_he_thread_policy_name(policy_name)
    if policy == "compare":
        raise ValueError("compare/all/sweep must be expanded before resolving HE thread policy.")
    if (policy == "serial") or (total <= 1):
        blas_threads = 1
        rayon_threads = 1
        he_threads = 1
    elif policy == "rayon_parallel_blas_serial":
        blas_threads = 1
        rayon_threads = total
        he_threads = total
    elif policy == "blas_parallel_rayon_serial":
        blas_threads = total
        rayon_threads = 1
        he_threads = 1
    elif policy == "split_half":
        blas_threads = max(1, total // 2)
        rayon_threads = max(1, total - blas_threads)
        he_threads = rayon_threads
    else:
        raise ValueError(f"Unsupported HE thread policy {policy_name!r}")
    return {
        "policy": str(policy),
        "target_threads": int(total),
        "blas_threads": int(blas_threads),
        "rayon_threads": int(rayon_threads),
        "he_threads": int(he_threads),
        "label": (
            f"{policy} "
            f"(blas={int(blas_threads)}, rayon={int(rayon_threads)}, he={int(he_threads)})"
        ),
    }


def _normalize_pcg_thread_policy_name(
    raw: object,
    *,
    default: str = _PCG_THREAD_POLICY_DEFAULT,
) -> str:
    txt = str(raw).strip().lower()
    if txt == "":
        txt = str(default).strip().lower()
    norm = re.sub(r"[^a-z0-9]+", "", txt)
    if norm in {
        "",
        "default",
        "auto",
        "rayon",
        "rayonparallelblasserial",
        "rayononly",
    }:
        return "rayon_parallel_blas_serial"
    if norm in {"blas", "blasparallelrayonserial", "blasonly"}:
        return "blas_parallel_rayon_serial"
    if norm in {"split", "splithalf", "halfhalf", "balanced"}:
        return "split_half"
    if norm in {"serial", "single", "singlethread", "one"}:
        return "serial"
    if norm in {"compare", "all", "sweep"}:
        return "compare"
    raise ValueError(
        "Unsupported PCG thread policy "
        f"{raw!r}. Use one of: rayon_parallel_blas_serial, "
        "blas_parallel_rayon_serial, split_half."
    )


def _expand_pcg_thread_policy_names(raw: object) -> list[str]:
    txt = str(raw).strip()
    if txt == "":
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[\s,;:+|]+", txt):
        token = str(token).strip()
        if token == "":
            continue
        policy = _normalize_pcg_thread_policy_name(token)
        names = (
            list(_PCG_THREAD_POLICY_SWEEP_DEFAULT)
            if policy == "compare"
            else [policy]
        )
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


def _resolve_pcg_thread_policy(policy_name: object, total_threads: int) -> dict[str, object]:
    total = max(1, int(total_threads))
    policy = _normalize_pcg_thread_policy_name(policy_name)
    if (policy == "serial") or (total <= 1):
        blas_threads = 1
        rayon_threads = 1
        pcg_threads = 1
    elif policy == "rayon_parallel_blas_serial":
        blas_threads = 1
        rayon_threads = total
        pcg_threads = total
    elif policy == "blas_parallel_rayon_serial":
        blas_threads = total
        rayon_threads = 1
        pcg_threads = 1
    elif policy == "split_half":
        blas_threads = max(1, total // 2)
        rayon_threads = max(1, total - blas_threads)
        pcg_threads = rayon_threads
    else:
        raise ValueError(f"Unsupported PCG thread policy {policy_name!r}")
    return {
        "policy": str(policy),
        "target_threads": int(total),
        "blas_threads": int(blas_threads),
        "rayon_threads": int(rayon_threads),
        "pcg_threads": int(pcg_threads),
        "label": (
            f"{policy} "
            f"(blas={int(blas_threads)}, rayon={int(rayon_threads)}, pcg={int(pcg_threads)})"
        ),
    }


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def step(message: str) -> None:
    print()
    print(f"==> {message}")


def sep() -> None:
    print("=" * 44)


def report_special_progress(done: int, total: int, label: str) -> None:
    total_n = max(1, int(total))
    done_n = max(0, min(int(done), total_n))
    print(
        "Benchmark progress      : "
        f"{done_n}/{total_n} ({str(label)} done)"
    )


def report_benchmark_phase(label: str, phase: str, elapsed: float, extra: str = "") -> None:
    line = (
        f"{str(label)} benchmark progress: "
        f"{str(phase)} done [{float(max(0.0, elapsed)):.3f}s]"
    )
    extra_txt = str(extra).strip()
    if extra_txt != "":
        line += f" {extra_txt}"
    print(line)


def cmd_to_text(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return " ".join(sh_quote(x) for x in cmd)


def sh_quote(s: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+@%-]+", s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def run(
    cmd: list[str],
    *,
    cwd: Path = ROOT_DIR,
    stdout_path: Path | None = None,
    stderr_to_stdout: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"+ {cmd_to_text(cmd)}")
    proc_env = os.environ.copy()

    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as out:
            return subprocess.run(
                cmd,
                cwd=str(cwd),
                text=True,
                env=proc_env,
                stdout=out,
                stderr=subprocess.STDOUT if stderr_to_stdout else None,
                check=True,
            )

    return subprocess.run(cmd, cwd=str(cwd), text=True, env=proc_env, check=True)


def require_file(path: Path, message: str = "required file not found") -> None:
    if not path.is_file():
        fail(f"{message}: {path}")


def require_any_file(message: str, candidates: Iterable[Path]) -> Path:
    candidates = list(candidates)
    for path in candidates:
        if path.is_file():
            return path

    print("Missing candidates:", file=sys.stderr)
    for path in candidates:
        print(f"  {path}", file=sys.stderr)
    fail(message)


def require_log_match(log_path: Path, pattern: str, message: str) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if not re.search(pattern, text, flags=re.MULTILINE):
        print_log(log_path)
        fail(f"{message}: {pattern}")


def require_log_not_match(log_path: Path, pattern: str, message: str) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
        print_log(log_path)
        fail(f"{message}: {pattern}")


def print_log(log_path: Path) -> None:
    print(f"----- LOG BEGIN: {log_path} -----", file=sys.stderr)
    try:
        print(log_path.read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
    except FileNotFoundError:
        print("<log file missing>", file=sys.stderr)
    print(f"----- LOG END: {log_path} -----", file=sys.stderr)


def remove_if_exists(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def find_grm(outdir: Path) -> Path:
    candidates = [
        outdir / "mouse_hs1940.cGRM.txt",
        outdir / "mouse_hs1940.grm.txt",
    ]
    return require_any_file("GRM file not found for GWAS (-k).", candidates)


def find_gs_summary(outdir: Path) -> Path:
    return require_any_file(
        "GS summary output missing",
        [
            outdir / "mouse_hs1940.gs.model" / "summary.json",
            outdir / "mouse_hs1940.gs" / "summary.json",
        ],
    )


def _tsv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            first = fh.readline().rstrip("\n\r")
    except OSError:
        return []
    if not first:
        return []
    return first.split("\t")


def _looks_like_postgwas_input(path: Path) -> bool:
    """Return True only for GWAS result tables accepted by `jx postgwas`.

    The validation directory can contain other TSV files with similar names,
    for example postGS `*.effect_top.tsv` tables, GS prediction tables, or
    FarmCPU `*.qtn.tsv` helper outputs. Those are not the intended postgwas
    inputs for this validation step. Valid JanusX GWAS result tables are
    identified by chrom/pos/pwald-style columns.
    """
    name = path.name
    if name.endswith(".gs.tsv"):
        return False
    if name.endswith(".effect_top.tsv"):
        return False
    if name.endswith(".qtn.tsv"):
        return False
    if ".effect" in name:
        return False

    header_raw = _tsv_header(path)
    header = {h.strip().lower() for h in header_raw if h.strip()}

    has_chrom = bool({"chrom", "chr", "chromosome"} & header)
    has_pos = bool({"pos", "bp", "position"} & header)
    has_pvalue = "pwald" in header

    # JanusX GWAS result tables typically use:
    #   chrom, pos, allele0, allele1, maf, beta, se, pwald
    # They may not contain an explicit SNP/marker ID column, so requiring
    # `snp` here would incorrectly reject valid GWAS outputs.
    return has_chrom and has_pos and has_pvalue


def find_gwas_tsv_files(outdir: Path) -> list[Path]:
    candidates = sorted(p for p in outdir.glob("mouse_hs1940.test*.tsv") if p.is_file())
    files = [p for p in candidates if _looks_like_postgwas_input(p)]
    if not files:
        print("Scanned TSV candidates:", file=sys.stderr)
        for p in candidates:
            header = "\t".join(_tsv_header(p)[:8])
            print(f"  {p}\t[{header}]", file=sys.stderr)
        fail("No valid GWAS result TSV files found for postgwas.")
    return files


def check_jx_available() -> None:
    if shutil.which("jx") is None:
        fail("jx command not found in PATH")
    run(["jx", "--version"])


def import_janusx_runtime() -> tuple[object, object]:
    use_local_py = env_truthy("JX_GGVAL_USE_LOCAL_PYTHON", False)
    if use_local_py:
        py_root = ROOT_DIR / "python"
        if str(py_root) not in sys.path:
            sys.path.insert(0, str(py_root))
    else:
        root_resolved = str(ROOT_DIR.resolve())
        py_root_resolved = str((ROOT_DIR / "python").resolve())
        cleaned: list[str] = []
        for p in sys.path:
            p_norm = p if p != "" else os.getcwd()
            try:
                p_resolved = str(Path(p_norm).resolve())
            except Exception:
                p_resolved = str(p_norm)
            if p_resolved in {root_resolved, py_root_resolved}:
                continue
            cleaned.append(p)
        sys.path[:] = cleaned

    try:
        import janusx  # type: ignore
        from janusx import janusx as jxrs  # type: ignore
        return janusx, jxrs
    except Exception as ex:  # pragma: no cover - runtime environment dependent
        if not use_local_py:
            py_root = ROOT_DIR / "python"
            if str(py_root) not in sys.path:
                sys.path.insert(0, str(py_root))
            try:
                import janusx  # type: ignore
                from janusx import janusx as jxrs  # type: ignore
                print("NOTE: janusx site-packages import failed; fallback to local python/janusx.")
                return janusx, jxrs
            except Exception as ex2:
                fail(f"Unable to import janusx Rust module for backend checks: {ex2} (initial error: {ex})")
        fail(f"Unable to import janusx Rust module for backend checks: {ex}")


def resolve_runtime_thread_stage():
    try:
        from janusx.script._common.threads import runtime_thread_stage as _runtime_thread_stage

        return _runtime_thread_stage
    except Exception:
        def _fallback_runtime_thread_stage(
            *,
            blas_threads: int | None = None,
            rayon_threads: int | None = None,
        ):
            _ = blas_threads
            _ = rayon_threads
            return nullcontext()

        return _fallback_runtime_thread_stage


@contextmanager
def temporary_env_value(name: str, value: str | None):
    prev = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def report_janusx_runtime(janusx_module: object, jxrs_module: object) -> None:
    print(f"JANUSX module path       : {str(getattr(jxrs_module, '__file__', '?'))}")
    print(f"JANUSX package path      : {str(getattr(janusx_module, '__file__', '?'))}")


def cleanup_tmp_outputs(outdir: Path, prefix_name: str) -> None:
    pattern = f"{prefix_name}*"
    for p in outdir.glob(pattern):
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
        except FileNotFoundError:
            pass


def assert_he_outputs_close(ref: tuple[object, ...], cur: tuple[object, ...], full_cores: int) -> None:
    if len(ref) != len(cur):
        fail(
            "HE return length mismatch between 1-core and "
            f"{full_cores}-core runs: {len(ref)} vs {len(cur)}"
        )

    exact_fields = {
        3: "converged",
        4: "iters",
        6: "m_effective",
    }
    for idx, name in exact_fields.items():
        if idx >= len(ref):
            continue
        if ref[idx] != cur[idx]:
            fail(
                f"HE output mismatch for {name}: "
                f"1-core={ref[idx]!r}, {full_cores}-core={cur[idx]!r}"
            )

    float_fields = {
        0: "sigma_g2",
        1: "sigma_e2",
        2: "h2",
        5: "rel_res",
        7: "tr_k2",
        8: "y_ky",
        9: "y_y",
        10: "lambda",
        11: "tr_k2_solve",
    }
    for idx, name in float_fields.items():
        if idx >= len(ref):
            continue
        a = float(ref[idx])
        b = float(cur[idx])
        if not np.allclose([a], [b], rtol=5e-4, atol=1e-8, equal_nan=True):
            fail(
                f"HE output drifted across thread counts for {name}: "
                f"1-core={a}, {full_cores}-core={b}"
            )


def assert_pcg_outputs_close(ref: tuple[object, ...], cur: tuple[object, ...], full_cores: int) -> None:
    if len(ref) != len(cur):
        fail(
            "PCG return length mismatch between 1-core and "
            f"{full_cores}-core runs: {len(ref)} vs {len(cur)}"
        )

    scalar_fields = {
        2: ("pve_trainvar", 5e-4, 1e-8),
        5: ("rel_res", 5e-4, 1e-8),
        7: ("pve_lambda_vc", 5e-4, 1e-8),
        8: ("k_trace_mean", 5e-4, 1e-8),
    }
    for idx, (name, rtol, atol) in scalar_fields.items():
        if idx >= len(ref):
            continue
        a = float(ref[idx])
        b = float(cur[idx])
        if not np.allclose([a], [b], rtol=rtol, atol=atol, equal_nan=True):
            fail(
                f"PCG output drifted across thread counts for {name}: "
                f"1-core={a}, {full_cores}-core={b}"
            )

    exact_fields = {
        3: "converged",
        6: "m_effective",
    }
    for idx, name in exact_fields.items():
        if idx >= len(ref):
            continue
        if ref[idx] != cur[idx]:
            fail(
                f"PCG output mismatch for {name}: "
                f"1-core={ref[idx]!r}, {full_cores}-core={cur[idx]!r}"
            )

    if len(ref) >= 10:
        beta_ref = np.asarray(ref[9], dtype=np.float32).reshape(-1)
        beta_cur = np.asarray(cur[9], dtype=np.float32).reshape(-1)
        if beta_ref.shape != beta_cur.shape:
            fail(
                f"PCG beta shape mismatch: 1-core={beta_ref.shape}, {full_cores}-core={beta_cur.shape}"
            )
        if not np.allclose(beta_ref, beta_cur, rtol=5e-4, atol=1e-6, equal_nan=True):
            diff = float(np.max(np.abs(beta_ref - beta_cur))) if beta_ref.size > 0 else 0.0
            fail(
                "PCG beta drifted across thread counts: "
                f"max_abs_diff={diff}, size={beta_ref.size}"
            )


def smoke_flow(outdir: Path, logdir: Path, threads: int, cv_folds: int) -> None:
    step("SMOKE 1. Build PLINK cache from example VCF")
    run(["jx", "gformat", "-vcf", "example/mouse_hs1940.vcf.gz", "-fmt", "plink", "-o", str(outdir)])
    require_file(outdir / "mouse_hs1940.bed", "PLINK BED output missing")
    require_file(outdir / "mouse_hs1940.bim", "PLINK BIM output missing")
    require_file(outdir / "mouse_hs1940.fam", "PLINK FAM output missing")
    sep()

    step("SMOKE 1b. VCF cache split regression (-snps-only vs default)")
    vcf_src = "example/mouse_hs1940.vcf.gz"
    vcf_pheno = "example/mouse_hs1940.pheno"
    vcf_cache_base = Path("example/~mouse_hs1940")
    log_snp0 = logdir / "jx_gwas_vcf_snp0.log"
    log_snp1 = logdir / "jx_gwas_vcf_snp1.log"

    remove_if_exists(
        [
            vcf_cache_base.with_suffix(".snp0.bed"),
            vcf_cache_base.with_suffix(".snp0.bim"),
            vcf_cache_base.with_suffix(".snp0.fam"),
            vcf_cache_base.with_suffix(".snp1.bed"),
            vcf_cache_base.with_suffix(".snp1.bim"),
            vcf_cache_base.with_suffix(".snp1.fam"),
        ]
    )

    run(
        [
            "jx",
            "gwas",
            "-vcf",
            vcf_src,
            "-p",
            vcf_pheno,
            "-lm",
            "-n",
            "0",
            "-t",
            str(threads),
            "-o",
            str(outdir),
        ],
        stdout_path=log_snp0,
        stderr_to_stdout=True,
    )
    run(
        [
            "jx",
            "gwas",
            "-vcf",
            vcf_src,
            "-p",
            vcf_pheno,
            "-lm",
            "-n",
            "0",
            "-snps-only",
            "-t",
            str(threads),
            "-o",
            str(outdir),
        ],
        stdout_path=log_snp1,
        stderr_to_stdout=True,
    )

    require_log_match(log_snp0, r"Cache prefix: .*\.snp0", "default VCF GWAS did not use snp0 cache prefix")
    require_log_match(log_snp1, r"Cache prefix: .*\.snp1", "-snps-only VCF GWAS did not use snp1 cache prefix")

    for suffix in [".snp0.bed", ".snp0.bim", ".snp0.fam", ".snp1.bed", ".snp1.bim", ".snp1.fam"]:
        require_file(Path(f"{vcf_cache_base}{suffix}"), f"missing VCF cache file {suffix}")

    remove_if_exists(Path(f"{vcf_cache_base}{suffix}") for suffix in [".snp0.bed", ".snp0.bim", ".snp0.fam", ".snp1.bed", ".snp1.bim", ".snp1.fam"])
    sep()

    step("SMOKE 2. GWAS runtime check (LM/LMM/FastLMM/FarmCPU)")
    run(["jx", "grm", "-bfile", str(outdir / "mouse_hs1940"), "-o", str(outdir)])
    run_pca_with_backend_report(
        ["jx", "pca", "-bfile", str(outdir / "mouse_hs1940"), "-o", str(outdir)],
        log_path=logdir / "jx_pca_smoke.log",
    )
    grm_k = find_grm(outdir)
    require_file(outdir / "mouse_hs1940.eigenvec", "PCA eigenvec output missing")
    run(
        [
            "jx",
            "gwas",
            "-bfile",
            str(outdir / "mouse_hs1940"),
            "-p",
            "example/mouse_hs1940.pheno",
            "-farmcpu",
            "-lmm",
            "-lm",
            "-fastlmm",
            "-k",
            str(grm_k),
            "-c",
            str(outdir / "mouse_hs1940.eigenvec"),
            "-t",
            str(threads),
            "-o",
            str(outdir),
        ]
    )
    sep()

    step("SMOKE 3. rrBLUP runtime check")
    run(
        [
            "jx",
            "gs",
            "-bfile",
            str(outdir / "mouse_hs1940"),
            "-p",
            "example/mouse_hs1940.pheno",
            "-n",
            "0",
            "-rrBLUP",
            "-cv",
            str(cv_folds),
            "-t",
            str(threads),
            "-o",
            str(outdir),
        ]
    )
    find_gs_summary(outdir)
    sep()


def full_flow(outdir: Path, logdir: Path, postgs_enabled: bool) -> None:
    step("STEP 1. Simulation for validation flow")
    run(["jx", "sim", "10", "1000", str(outdir / "mouse_hs1940")])
    run(["jx", "gformat", "-vcf", "example/mouse_hs1940.vcf.gz", "-fmt", "plink", "-o", str(outdir)])
    require_file(outdir / "mouse_hs1940.bed", "PLINK BED output missing")
    require_file(outdir / "mouse_hs1940.bim", "PLINK BIM output missing")
    require_file(outdir / "mouse_hs1940.fam", "PLINK FAM output missing")
    run(["jx", "gformat", "-vcf", "example/mouse_hs1940.vcf.gz", "-fmt", "hmp", "-o", str(outdir)])
    run(["jx", "gformat", "-vcf", "example/mouse_hs1940.vcf.gz", "-fmt", "txt", "-o", str(outdir)])
    sep()

    step("STEP 2. Validation of GWAS-related modules")
    run(["jx", "grm", "-bfile", str(outdir / "mouse_hs1940"), "-o", str(outdir), "-m", "1"])
    run(["jx", "grm", "-bfile", str(outdir / "mouse_hs1940"), "-o", str(outdir), "-m", "2"])
    run_pca_with_backend_report(
        ["jx", "pca", "-bfile", str(outdir / "mouse_hs1940"), "-o", str(outdir)],
        log_path=logdir / "jx_pca_full.log",
    )
    run(["jx", "pca", "-bfile", str(outdir / "mouse_hs1940"), "-rsvd", "-o", str(outdir)])
    grm_k = find_grm(outdir)
    require_file(outdir / "mouse_hs1940.eigenvec", "PCA eigenvec output missing")

    run(
        [
            "jx",
            "gwas",
            "-bfile",
            str(outdir / "mouse_hs1940"),
            "-p",
            "example/mouse_hs1940.pheno",
            "-farmcpu",
            "-lmm",
            "-lm",
            "-fastlmm",
            "-k",
            str(grm_k),
            "-c",
            str(outdir / "mouse_hs1940.eigenvec"),
            "-o",
            str(outdir),
        ]
    )

    gwas_files = find_gwas_tsv_files(outdir)
    run(
        [
            "jx",
            "postgwas",
            "-gwasfile",
            *[str(p) for p in gwas_files],
            "-bfile",
            str(outdir / "mouse_hs1940"),
            "-manh",
            "4",
            "-qq",
            "-scatter-size",
            "16",
            "-fmt",
            "pdf",
            "-full",
            "-palette",
            "tab10",
            "-o",
            str(outdir),
        ]
    )
    sep()

    step("STEP 3. Validation of GS-related modules")
    run(
        [
            "jx",
            "gs",
            "-bfile",
            str(outdir / "mouse_hs1940"),
            "-p",
            "example/mouse_hs1940.pheno",
            "-n",
            "0",
            "-GBLUP",
            "-GBLUP",
            "ad",
            "-rrBLUP",
            "-BayesA",
            "-BayesB",
            "-BayesCpi",
            "-cv",
            "5",
            "-o",
            str(outdir),
        ]
    )
    gs_summary = find_gs_summary(outdir)

    if postgs_enabled:
        step("STEP 3b. Validation of postGS-related modules")
        run(["jx", "postgs", "-json", str(gs_summary), "-o", str(outdir)])
        sep()

    run(
        [
            "jx",
            "reml",
            "-file",
            "example/rice6048.reml.tsv",
            "-n",
            "Plant_height",
            "-rh",
            "year",
            "-rh",
            "loc",
            "-o",
            str(outdir),
        ]
    )
    sep()


def _parse_grm_kernel_backend(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"GRM \(Effective SNPs:\s*\d+,\s*([^)]+)\)\s*\.\.\.Finished",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return str(m.group(1)).strip()
    if re.search(r"memmap.*fallback.*packed", text, flags=re.IGNORECASE):
        return "packed-bed kernel (fallback)"
    if re.search(r"memmap", text, flags=re.IGNORECASE):
        return "rust-memmap kernel"
    if re.search(r"packed", text, flags=re.IGNORECASE):
        return "packed-bed kernel"
    return "unknown"


def _parse_pca_eigh_backend(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = [
        r"EIGH backend:\s*([A-Za-z0-9_:+.-]+)",
        r"Eigen decomposition finished \(backend=([^),\s]+)",
        r"Rust eigh did not use LAPACK backend \(backend=([^)\s]+)\)",
        r"backend=([A-Za-z0-9_:+.-]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return str(m.group(1)).strip()
    if re.search(r"\bnalgebra\b", text, flags=re.IGNORECASE):
        return "nalgebra"
    if re.search(r"\blapack_dsyevd\b", text, flags=re.IGNORECASE):
        return "lapack_dsyevd"
    if re.search(r"\blapack_dsyevr\b", text, flags=re.IGNORECASE):
        return "lapack_dsyevr"
    return "unknown"


def run_pca_with_backend_report(
    cmd: list[str],
    *,
    log_path: Path,
    label: str = "PCA/EIGH",
) -> str:
    try:
        run(
            cmd,
            stdout_path=log_path,
            stderr_to_stdout=True,
        )
    except subprocess.CalledProcessError:
        print_log(log_path)
        raise

    backend = _parse_pca_eigh_backend(log_path)
    print(f"{label} backend         : {backend}")
    if backend == "unknown":
        print(f"{label} backend log     : {log_path}")
    return backend


def backend_thread_checks(outdir: Path, *, bench_threads: int = 0) -> None:
    nsnp_k = env_int("JX_GGVAL_BENCH_NSNP_K", 500, min_value=1)
    n_individuals = env_int("JX_GGVAL_BENCH_NIND", 5000, min_value=2)
    step(
        "STEP 4. Backend report and 1-core/all-core timing checks "
        f"(simulated {n_individuals} x {nsnp_k * 1000} genotype)"
    )
    sim_prefix = outdir / "__ggval_tmp_large_sim"
    full_cores, full_cores_source = detect_effective_threads(bench_threads)

    janusx, jxrs = import_janusx_runtime()
    runtime_thread_stage = resolve_runtime_thread_stage()
    report_janusx_runtime(janusx, jxrs)
    print(
        "Benchmark thread target  : "
        f"1 vs {full_cores} ({full_cores_source})"
    )

    def _find_grm_for_prefix(prefix_path: Path) -> Path:
        base = str(prefix_path)
        candidates = [
            Path(f"{base}.cGRM.npy"),
            Path(f"{base}.grm.npy"),
            Path(f"{base}.cGRM.txt"),
            Path(f"{base}.grm.txt"),
        ]
        return require_any_file("GRM benchmark output missing", candidates)

    cleanup_tmp_outputs(outdir, sim_prefix.name)
    try:
        run(["jx", "sim", str(nsnp_k), str(n_individuals), str(sim_prefix)])
        bfile_prefix = sim_prefix
        require_file(bfile_prefix.with_suffix(".bed"), "Simulated BED output missing")
        require_file(bfile_prefix.with_suffix(".bim"), "Simulated BIM output missing")
        require_file(bfile_prefix.with_suffix(".fam"), "Simulated FAM output missing")

        def run_grm_once(threads: int) -> tuple[float, str, Path]:
            prefix = f"{sim_prefix.name}.grmbench.t{int(threads)}"
            t0 = time.perf_counter()
            run(
                [
                    "jx",
                    "grm",
                    "-bfile",
                    str(bfile_prefix),
                    "-t",
                    str(int(threads)),
                    "-o",
                    str(outdir),
                    "-prefix",
                    prefix,
                    "-npy",
                ]
            )
            elapsed = time.perf_counter() - t0
            log_path = outdir / f"{prefix}.grm.log"
            require_file(log_path, "GRM benchmark log missing")
            backend = _parse_grm_kernel_backend(log_path)
            grm_file = _find_grm_for_prefix(outdir / prefix)
            return elapsed, backend, grm_file

        grm_t1, grm_backend_1, grm_file = run_grm_once(1)
        grm_tn, grm_backend_n, _ = run_grm_once(full_cores)
        if grm_backend_1 == grm_backend_n:
            print(f"GRM build backend        : {grm_backend_n}")
        else:
            print(
                "GRM build backend        : "
                f"1-core={grm_backend_1}, {full_cores}-core={grm_backend_n}"
            )

        eigh_backend = str(jxrs.rust_eigh_lapack_backend()).strip()
        print(f"EIGH backend             : {eigh_backend}")

        def run_eigh_once(threads: int) -> tuple[float, str]:
            t0 = time.perf_counter()
            ret = jxrs.rust_eigh_from_matrix_file_f64(
                str(grm_file),
                threads=int(threads),
                driver="auto",
                jobz="N",
                require_lapack=False,
            )
            elapsed = time.perf_counter() - t0
            evd_backend = str(ret[3]).strip()
            return elapsed, evd_backend

        eigh_t1, eigh_run_backend_1 = run_eigh_once(1)
        eigh_tn, eigh_run_backend_n = run_eigh_once(full_cores)
        if eigh_run_backend_1 == eigh_run_backend_n:
            print(f"EIGH solve backend       : {eigh_run_backend_1}")
        else:
            print(
                "EIGH solve backend       : "
                f"1-core={eigh_run_backend_1}, {full_cores}-core={eigh_run_backend_n}"
            )

        print(f"EIGH runtime  (1 core)   : {eigh_t1:.3f}s")
        print(f"EIGH runtime  ({full_cores} cores): {eigh_tn:.3f}s")
        print(f"GRM runtime   (1 core)   : {grm_t1:.3f}s")
        print(f"GRM runtime   ({full_cores} cores): {grm_tn:.3f}s")
    finally:
        cleanup_tmp_outputs(outdir, sim_prefix.name)
    sep()


def he_thread_checks(
    outdir: Path,
    *,
    bench_threads: int = 0,
    thread_policy: str = _HE_THREAD_POLICY_DEFAULT,
    thread_policies: str = "",
    rowmajor_kernel_policy: str = _ROWMAJOR_KERNEL_DEFAULT,
    rowmajor_kernel_policies: str = "blas,rayon",
    stage_timing: bool = False,
    stage_log_every: int = 8,
) -> None:
    default_nsnp_k = env_int("JX_GGVAL_BENCH_NSNP_K", 500, min_value=1)
    default_n_individuals = env_int("JX_GGVAL_BENCH_NIND", 5000, min_value=2)
    nsnp_k = env_int("JX_GGVAL_HE_BENCH_NSNP_K", default_nsnp_k, min_value=1)
    n_individuals = env_int("JX_GGVAL_HE_BENCH_NIND", default_n_individuals, min_value=2)
    trace_samples = env_int("JX_GGVAL_HE_TRACE_SAMPLES", 64, min_value=8)
    trace_probe_batch = env_int(
        "JX_GGVAL_HE_TRACE_PROBE_BATCH",
        min(64, trace_samples),
        min_value=1,
    )
    block_rows = env_int("JX_GGVAL_HE_BLOCK_ROWS", 4096, min_value=1)
    max_iter = env_int("JX_GGVAL_HE_MAX_ITER", 32, min_value=1)
    seed = env_int("JX_GGVAL_HE_SEED", 20260514, min_value=0)

    step(
        "STEP HE. HE equation 1-core/all-core timing check "
        f"(simulated {n_individuals} x {nsnp_k * 1000} genotype)"
    )
    sim_prefix = outdir / "__ggval_tmp_he_sim"
    full_cores, full_cores_source = detect_effective_threads(bench_threads)

    janusx, jxrs = import_janusx_runtime()
    runtime_thread_stage = resolve_runtime_thread_stage()
    policy_names = _expand_he_thread_policy_names(thread_policies)
    if len(policy_names) == 0:
        policy_names = _expand_he_thread_policy_names(thread_policy)
    if len(policy_names) == 0:
        policy_names = [_normalize_he_thread_policy_name(thread_policy)]
    kernel_names = _expand_rowmajor_kernel_names(rowmajor_kernel_policies)
    if len(kernel_names) == 0:
        kernel_names = _expand_rowmajor_kernel_names(rowmajor_kernel_policy)
    if len(kernel_names) == 0:
        kernel_names = [_normalize_rowmajor_kernel_name(rowmajor_kernel_policy)]
    report_janusx_runtime(janusx, jxrs)
    print(
        "HE benchmark threads     : "
        f"1 vs {full_cores} ({full_cores_source})"
    )
    print(f"HE BLAS backend          : {str(jxrs.rust_sgemm_backend()).strip()}")
    if len(kernel_names) == 1:
        print(
            "HE kernel strategy      : "
            f"{_rowmajor_kernel_label(kernel_names[0])} ({kernel_names[0]})"
        )
    else:
        print(
            "HE kernel compare       : "
            + " vs ".join(
                f"{_rowmajor_kernel_label(name)} ({name})" for name in kernel_names
            )
        )
    if len(policy_names) == 1:
        policy_preview = _resolve_he_thread_policy(policy_names[0], full_cores)
        print(f"HE thread policy         : {str(policy_preview['label'])}")
    else:
        print(
            "HE thread sweep          : "
            + ", ".join(str(name) for name in policy_names)
        )
    print(
        "HE trace config         : "
        f"samples={int(trace_samples)}, probe_batch={int(trace_probe_batch)}"
    )
    if stage_timing:
        print(
            "HE stage timing         : "
            "enabled for a separate diagnostic run "
            f"(JX_GS_HE_STAGE_LOG_EVERY={int(max(1, stage_log_every))})"
        )

    cleanup_tmp_outputs(outdir, sim_prefix.name)
    try:
        run(["jx", "sim", str(nsnp_k), str(n_individuals), str(sim_prefix)])
        bfile_prefix = sim_prefix
        require_file(bfile_prefix.with_suffix(".bed"), "Simulated BED output missing")
        require_file(bfile_prefix.with_suffix(".bim"), "Simulated BIM output missing")
        require_file(bfile_prefix.with_suffix(".fam"), "Simulated FAM output missing")

        packed_arr, _miss_arr, maf_arr, _std_arr, n_samples = jxrs.load_bed_2bit_packed(str(bfile_prefix))
        packed = np.ascontiguousarray(np.asarray(packed_arr, dtype=np.uint8), dtype=np.uint8)
        maf = np.ascontiguousarray(np.asarray(maf_arr, dtype=np.float32).reshape(-1), dtype=np.float32)
        row_flip = np.ascontiguousarray(
            np.asarray(
                jxrs.bed_packed_row_flip_mask(packed, int(n_samples)),
                dtype=np.bool_,
            ).reshape(-1),
            dtype=np.bool_,
        )
        train_idx = np.arange(int(n_samples), dtype=np.int64)
        y = np.asarray(np.random.default_rng(int(seed)).standard_normal(int(n_samples)), dtype=np.float64)
        y -= float(y.mean())

        print(
            "HE benchmark payload     : "
            f"{int(packed.shape[0])} SNP x {int(n_samples)} samples"
        )

        def run_he_once(
            policy_spec: dict[str, object],
            kernel_name: str,
            *,
            enable_stage_timing: bool = False,
        ) -> tuple[float, tuple[object, ...]]:
            threads = int(policy_spec["he_threads"])
            blas_threads = int(policy_spec["blas_threads"])
            rayon_threads = int(policy_spec["rayon_threads"])
            prev_stage_timing = os.environ.get("JX_GS_HE_STAGE_TIMING")
            prev_stage_log_every = os.environ.get("JX_GS_HE_STAGE_LOG_EVERY")
            if enable_stage_timing:
                print(
                    "HE timing run            : "
                    f"{str(policy_spec['label'])}"
                )
                os.environ["JX_GS_HE_STAGE_TIMING"] = "1"
                os.environ["JX_GS_HE_STAGE_LOG_EVERY"] = str(int(max(1, stage_log_every)))
            t0 = time.perf_counter()
            try:
                with temporary_env_value("JX_ROWMAJOR_F32_KERNEL", str(kernel_name)):
                    with runtime_thread_stage(
                        blas_threads=int(blas_threads),
                        rayon_threads=int(rayon_threads),
                    ):
                        call_kwargs = dict(
                            trace_samples=int(trace_samples),
                            trace_probe_batch=int(trace_probe_batch),
                            tol=1e-6,
                            max_iter=int(max_iter),
                            block_rows=int(block_rows),
                            std_eps=1e-12,
                            use_train_maf=True,
                            exact_trace_debug=False,
                            exact_trace_max_n=256,
                            threads=int(threads),
                            blas_threads=int(blas_threads),
                            seed=int(seed),
                            packed=packed,
                            packed_n_samples=int(n_samples),
                            maf=maf,
                            row_flip=row_flip,
                        )
                        try:
                            ret = jxrs.he_pcg_bed(
                                str(bfile_prefix),
                                train_idx,
                                y,
                                **call_kwargs,
                            )
                        except TypeError:
                            call_kwargs.pop("blas_threads", None)
                            ret = jxrs.he_pcg_bed(
                                str(bfile_prefix),
                                train_idx,
                                y,
                                **call_kwargs,
                            )
                elapsed = time.perf_counter() - t0
                return elapsed, tuple(ret)
            finally:
                if prev_stage_timing is None:
                    os.environ.pop("JX_GS_HE_STAGE_TIMING", None)
                else:
                    os.environ["JX_GS_HE_STAGE_TIMING"] = prev_stage_timing
                if prev_stage_log_every is None:
                    os.environ.pop("JX_GS_HE_STAGE_LOG_EVERY", None)
                else:
                    os.environ["JX_GS_HE_STAGE_LOG_EVERY"] = prev_stage_log_every

        he_base_spec = _resolve_he_thread_policy("serial", 1)
        he_base_kernel = kernel_names[0]
        he_t1, he_out_1 = run_he_once(
            he_base_spec,
            he_base_kernel,
            enable_stage_timing=False,
        )
        report_benchmark_phase("HE", "1-core baseline", he_t1)
        he_results: list[tuple[str, dict[str, object], float, tuple[object, ...]]] = []
        for kernel_name in kernel_names:
            for policy_name in policy_names:
                policy_spec = _resolve_he_thread_policy(policy_name, full_cores)
                he_tn, he_out_n = run_he_once(
                    policy_spec,
                    kernel_name,
                    enable_stage_timing=False,
                )
                assert_he_outputs_close(he_out_1, he_out_n, full_cores)
                he_results.append((kernel_name, policy_spec, he_tn, he_out_n))
        if len(he_results) > 0:
            best_multi_t = min(float(item[2]) for item in he_results)
            phase_text = (
                f"{full_cores}-core run"
                if len(he_results) == 1
                else f"{full_cores}-core strategy sweep"
            )
            report_benchmark_phase(
                "HE",
                phase_text,
                best_multi_t,
                extra=f"(completed {len(he_results)} run{'s' if len(he_results) != 1 else ''})",
            )

        best_kernel, best_spec, best_time, best_out = min(he_results, key=lambda item: item[2])
        print(f"HE runtime    (1 core)   : {he_t1:.3f}s")
        if len(he_results) == 1:
            only_kernel, only_spec, only_time, only_out = he_results[0]
            print(f"HE runtime    ({full_cores} cores): {only_time:.3f}s")
            if only_time > 0.0:
                print(f"HE speedup               : {he_t1 / only_time:.2f}x")
            he_out_to_report = only_out
            diag_kernel = only_kernel
            diag_spec = only_spec
        else:
            for kernel_name, policy_spec, elapsed, _he_out in he_results:
                speedup = (he_t1 / elapsed) if elapsed > 0.0 else float("nan")
                print(
                    "HE strategy result       : "
                    f"{_rowmajor_kernel_label(kernel_name)} ({kernel_name}), "
                    f"{str(policy_spec['label'])} -> "
                    f"{elapsed:.3f}s ({speedup:.2f}x)"
                )
            print(
                "HE best strategy         : "
                f"{_rowmajor_kernel_label(best_kernel)} ({best_kernel}), "
                f"{str(best_spec['label'])} -> "
                f"{best_time:.3f}s ({(he_t1 / best_time):.2f}x)"
            )
            he_out_to_report = best_out
            diag_kernel = best_kernel
            diag_spec = best_spec
        if len(he_out_to_report) >= 11:
            print(
                "HE estimate              : "
                f"h2={float(he_out_to_report[2]):.6f}, "
                f"lambda={float(he_out_to_report[10]):.6f}, "
                f"converged={bool(he_out_to_report[3])}, "
                f"m_eff={int(he_out_to_report[6])}"
            )
        if stage_timing:
            print(
                "HE stage timing note     : "
                "benchmark timings above exclude diagnostic logging overhead"
            )
            print(
                "HE stage timing policy   : "
                f"{str(diag_spec['label'])}"
            )
            print(
                "HE stage timing kernel   : "
                f"{_rowmajor_kernel_label(diag_kernel)} ({diag_kernel})"
            )
            he_diag_t, _he_diag_out = run_he_once(
                diag_spec,
                diag_kernel,
                enable_stage_timing=True,
            )
            print(f"HE stage diag runtime    : {he_diag_t:.3f}s")
    finally:
        cleanup_tmp_outputs(outdir, sim_prefix.name)
    sep()


def pcg_thread_checks(
    outdir: Path,
    *,
    bench_threads: int = 0,
    thread_policy: str = _PCG_THREAD_POLICY_DEFAULT,
    thread_policies: str = "",
    rowmajor_kernel_policy: str = _ROWMAJOR_KERNEL_DEFAULT,
    rowmajor_kernel_policies: str = "blas,rayon",
    stage_timing: bool = False,
    stage_log_every: int = 100,
) -> None:
    default_nsnp_k = env_int("JX_GGVAL_BENCH_NSNP_K", 500, min_value=1)
    default_n_individuals = env_int("JX_GGVAL_BENCH_NIND", 5000, min_value=2)
    nsnp_k = env_int("JX_GGVAL_PCG_BENCH_NSNP_K", default_nsnp_k, min_value=1)
    n_individuals = env_int("JX_GGVAL_PCG_BENCH_NIND", default_n_individuals, min_value=2)
    lambda_value = env_float("JX_GGVAL_PCG_LAMBDA", 1.0, min_value=0.0)
    tol = env_float("JX_GGVAL_PCG_TOL", 1e-4, min_value=0.0)
    max_iter = env_int("JX_GGVAL_PCG_MAX_ITER", 100, min_value=1)
    block_rows = env_int("JX_GGVAL_PCG_BLOCK_ROWS", 4096, min_value=1)
    seed = env_int("JX_GGVAL_PCG_SEED", 20260514, min_value=0)

    step(
        "STEP PCG. rrBLUP-PCG 1-core/all-core timing check "
        f"(simulated {n_individuals} x {nsnp_k * 1000} genotype)"
    )
    sim_prefix = outdir / "__ggval_tmp_pcg_sim"
    full_cores, full_cores_source = detect_effective_threads(bench_threads)

    janusx, jxrs = import_janusx_runtime()
    runtime_thread_stage = resolve_runtime_thread_stage()
    policy_names = _expand_pcg_thread_policy_names(thread_policies)
    if len(policy_names) == 0:
        if _normalize_pcg_thread_policy_name(thread_policy) == _PCG_THREAD_POLICY_DEFAULT:
            policy_names = list(_PCG_THREAD_POLICY_SWEEP_DEFAULT)
        else:
            policy_names = _expand_pcg_thread_policy_names(thread_policy)
    if len(policy_names) == 0:
        policy_names = [_normalize_pcg_thread_policy_name(thread_policy)]
    kernel_names = _expand_rowmajor_kernel_names(rowmajor_kernel_policies)
    if len(kernel_names) == 0:
        kernel_names = _expand_rowmajor_kernel_names(rowmajor_kernel_policy)
    if len(kernel_names) == 0:
        kernel_names = [_normalize_rowmajor_kernel_name(rowmajor_kernel_policy)]
    report_janusx_runtime(janusx, jxrs)
    print(
        "PCG benchmark threads    : "
        f"1 vs {full_cores} ({full_cores_source})"
    )
    print(f"PCG BLAS backend         : {str(jxrs.rust_sgemm_backend()).strip()}")
    if len(kernel_names) == 1:
        print(
            "PCG kernel strategy     : "
            f"{_rowmajor_kernel_label(kernel_names[0])} ({kernel_names[0]})"
        )
    else:
        print(
            "PCG kernel compare      : "
            + " vs ".join(
                f"{_rowmajor_kernel_label(name)} ({name})" for name in kernel_names
            )
        )
    if len(policy_names) == 1:
        policy_preview = _resolve_pcg_thread_policy(policy_names[0], full_cores)
        print(f"PCG thread policy       : {str(policy_preview['label'])}")
    else:
        print(
            "PCG thread sweep        : "
            + ", ".join(str(name) for name in policy_names)
        )
    if stage_timing:
        print(
            "PCG stage timing        : "
            f"enabled (JX_GS_PCG_STAGE_LOG_EVERY={int(max(1, stage_log_every))})"
        )

    cleanup_tmp_outputs(outdir, sim_prefix.name)
    try:
        run(["jx", "sim", str(nsnp_k), str(n_individuals), str(sim_prefix)])
        bfile_prefix = sim_prefix
        require_file(bfile_prefix.with_suffix(".bed"), "Simulated BED output missing")
        require_file(bfile_prefix.with_suffix(".bim"), "Simulated BIM output missing")
        require_file(bfile_prefix.with_suffix(".fam"), "Simulated FAM output missing")

        packed_arr, _miss_arr, maf_arr, _std_arr, n_samples = jxrs.load_bed_2bit_packed(str(bfile_prefix))
        packed = np.ascontiguousarray(np.asarray(packed_arr, dtype=np.uint8), dtype=np.uint8)
        maf = np.ascontiguousarray(np.asarray(maf_arr, dtype=np.float32).reshape(-1), dtype=np.float32)
        row_flip = np.ascontiguousarray(
            np.asarray(
                jxrs.bed_packed_row_flip_mask(packed, int(n_samples)),
                dtype=np.bool_,
            ).reshape(-1),
            dtype=np.bool_,
        )
        train_idx = np.arange(int(n_samples), dtype=np.int64)
        y = np.asarray(np.random.default_rng(int(seed)).standard_normal(int(n_samples)), dtype=np.float64)
        y -= float(y.mean())
        empty_idx = np.zeros((0,), dtype=np.int64)

        print(
            "PCG benchmark payload    : "
            f"{int(packed.shape[0])} SNP x {int(n_samples)} samples"
        )

        def run_pcg_once(
            policy_spec: dict[str, object],
            kernel_name: str,
        ) -> tuple[float, tuple[object, ...]]:
            threads = int(policy_spec["pcg_threads"])
            blas_threads = int(policy_spec["blas_threads"])
            rayon_threads = int(policy_spec["rayon_threads"])
            prev_stage_timing = os.environ.get("JX_GS_PCG_STAGE_TIMING")
            prev_stage_log_every = os.environ.get("JX_GS_PCG_STAGE_LOG_EVERY")
            if stage_timing:
                print(
                    "PCG timing run           : "
                    f"{str(policy_spec['label'])}, "
                    f"kernel={_rowmajor_kernel_label(kernel_name)} ({kernel_name})"
                )
                os.environ["JX_GS_PCG_STAGE_TIMING"] = "1"
                os.environ["JX_GS_PCG_STAGE_LOG_EVERY"] = str(int(max(1, stage_log_every)))
            t0 = time.perf_counter()
            try:
                with temporary_env_value("JX_ROWMAJOR_F32_KERNEL", str(kernel_name)):
                    with runtime_thread_stage(
                        blas_threads=int(blas_threads),
                        rayon_threads=int(rayon_threads),
                    ):
                        call_kwargs = dict(
                            test_sample_indices=empty_idx,
                            train_pred_local_indices=empty_idx,
                            site_keep=None,
                            lambda_value=float(lambda_value),
                            tol=float(tol),
                            max_iter=int(max_iter),
                            block_rows=int(block_rows),
                            std_eps=1e-12,
                            threads=int(threads),
                            progress_callback=None,
                            progress_every=0,
                            compute_trainvar=False,
                            packed=packed,
                            packed_n_samples=int(n_samples),
                            maf=maf,
                            row_flip=row_flip,
                            blas_threads=int(blas_threads),
                        )
                        try:
                            ret = jxrs.rrblup_pcg_bed(
                                str(bfile_prefix),
                                train_idx,
                                y,
                                **call_kwargs,
                            )
                        except TypeError:
                            call_kwargs.pop("blas_threads", None)
                            ret = jxrs.rrblup_pcg_bed(
                                str(bfile_prefix),
                                train_idx,
                                y,
                                **call_kwargs,
                            )
                elapsed = time.perf_counter() - t0
                return elapsed, tuple(ret)
            finally:
                if prev_stage_timing is None:
                    os.environ.pop("JX_GS_PCG_STAGE_TIMING", None)
                else:
                    os.environ["JX_GS_PCG_STAGE_TIMING"] = prev_stage_timing
                if prev_stage_log_every is None:
                    os.environ.pop("JX_GS_PCG_STAGE_LOG_EVERY", None)
                else:
                    os.environ["JX_GS_PCG_STAGE_LOG_EVERY"] = prev_stage_log_every

        pcg_base_spec = _resolve_pcg_thread_policy("serial", 1)
        pcg_base_kernel = kernel_names[0]
        pcg_t1, pcg_out_1 = run_pcg_once(pcg_base_spec, pcg_base_kernel)
        report_benchmark_phase("PCG", "1-core baseline", pcg_t1)
        pcg_results: list[tuple[str, dict[str, object], float, tuple[object, ...]]] = []
        for kernel_name in kernel_names:
            for policy_name in policy_names:
                policy_spec = _resolve_pcg_thread_policy(policy_name, full_cores)
                pcg_tn, pcg_out_n = run_pcg_once(policy_spec, kernel_name)
                assert_pcg_outputs_close(pcg_out_1, pcg_out_n, full_cores)
                pcg_results.append((kernel_name, policy_spec, pcg_tn, pcg_out_n))
        if len(pcg_results) > 0:
            best_multi_t = min(float(item[2]) for item in pcg_results)
            phase_text = (
                f"{full_cores}-core run"
                if len(pcg_results) == 1
                else f"{full_cores}-core strategy sweep"
            )
            report_benchmark_phase(
                "PCG",
                phase_text,
                best_multi_t,
                extra=f"(completed {len(pcg_results)} run{'s' if len(pcg_results) != 1 else ''})",
            )

        print(f"PCG runtime   (1 core)   : {pcg_t1:.3f}s")
        if len(pcg_results) == 1:
            only_kernel, only_spec, only_time, only_out = pcg_results[0]
            print(f"PCG runtime   ({full_cores} cores): {only_time:.3f}s")
            if only_time > 0.0:
                print(f"PCG speedup              : {pcg_t1 / only_time:.2f}x")
            pcg_out_to_report = only_out
            diag_kernel = only_kernel
            diag_spec = only_spec
        else:
            for kernel_name, policy_spec, elapsed, _pcg_out in pcg_results:
                speedup = (pcg_t1 / elapsed) if elapsed > 0.0 else float("nan")
                print(
                    "PCG strategy result      : "
                    f"{_rowmajor_kernel_label(kernel_name)} ({kernel_name}), "
                    f"{str(policy_spec['label'])} -> "
                    f"{elapsed:.3f}s ({speedup:.2f}x)"
                )
            best_kernel, best_spec, best_time, best_out = min(
                pcg_results,
                key=lambda item: item[2],
            )
            print(
                "PCG best strategy        : "
                f"{_rowmajor_kernel_label(best_kernel)} ({best_kernel}), "
                f"{str(best_spec['label'])} -> "
                f"{best_time:.3f}s ({(pcg_t1 / best_time):.2f}x)"
            )
            pcg_out_to_report = best_out
            diag_kernel = best_kernel
            diag_spec = best_spec
        if len(pcg_out_to_report) >= 9:
            print(
                "PCG estimate             : "
                f"converged={bool(pcg_out_to_report[3])}, "
                f"iters={int(pcg_out_to_report[4])}, "
                f"rel_res={float(pcg_out_to_report[5]):.6g}, "
                f"m_eff={int(pcg_out_to_report[6])}, "
                f"k_trace={float(pcg_out_to_report[8]):.6g}"
            )
        if stage_timing:
            print(
                "PCG stage timing note    : "
                "benchmark timings above exclude diagnostic logging overhead"
            )
            print(
                "PCG stage timing kernel  : "
                f"{_rowmajor_kernel_label(diag_kernel)} ({diag_kernel})"
            )
            print(
                "PCG stage timing policy  : "
                f"{str(diag_spec['label'])}"
            )
            pcg_diag_t, _pcg_diag_out = run_pcg_once(diag_spec, diag_kernel)
            print(f"PCG stage diag runtime   : {pcg_diag_t:.3f}s")
    finally:
        cleanup_tmp_outputs(outdir, sim_prefix.name)
    sep()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-platform JanusX validation runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default=env_str("JX_GGVAL_MODE", "full"),
        help="Validation mode. Can also be set by JX_GGVAL_MODE.",
    )
    parser.add_argument(
        "--outdir",
        default=env_str("JX_GGVAL_OUTDIR", "test"),
        help="Output directory. Can also be set by JX_GGVAL_OUTDIR.",
    )
    parser.add_argument(
        "--logdir",
        default=None,
        help="Log directory. Default is <outdir>/logs or JX_GGVAL_LOGDIR.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=env_int("JX_GGVAL_THREADS", 2, min_value=1),
        help="Thread count for selected commands. Can also be set by JX_GGVAL_THREADS.",
    )
    parser.add_argument(
        "--bench-threads",
        type=int,
        default=env_int("JX_GGVAL_BENCH_THREADS", 0, min_value=0),
        help=(
            "Thread count for the multi-core side of benchmark checks "
            "(GRM/EIGH/HE/PCG). Use 0 to auto-detect scheduler/affinity limits."
        ),
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=env_int("JX_GGVAL_CV", 2, min_value=2),
        help="CV folds for smoke GS. Can also be set by JX_GGVAL_CV.",
    )
    parser.add_argument(
        "--no-postgs",
        action="store_true",
        default=env_str("JX_GGVAL_POSTGS", "1") != "1",
        help="Skip postGS validation in full mode.",
    )
    parser.add_argument(
        "--testHE",
        "--test-he",
        dest="test_he",
        action="store_true",
        help="Run the HE thread benchmark branch (1 core vs all cores).",
    )
    parser.add_argument(
        "--he-thread-policy",
        type=str,
        default=env_str(
            "JX_GGVAL_HE_THREAD_POLICY",
            env_str("JX_GS_HE_THREAD_POLICY", _HE_THREAD_POLICY_DEFAULT),
        ),
        help=(
            "HE benchmark multi-core thread policy. "
            "Aliases: rayon, blas, split."
        ),
    )
    parser.add_argument(
        "--he-thread-policies",
        type=str,
        default=env_str("JX_GGVAL_HE_THREAD_POLICIES", ""),
        help=(
            "Comma-separated HE policies to compare on the same payload. "
            "Special token compare expands to rayon,blas,split."
        ),
    )
    parser.add_argument(
        "--testPCG",
        "--test-pcg",
        dest="test_pcg",
        action="store_true",
        help="Run the rrBLUP-PCG thread benchmark branch (1 core vs all cores).",
    )
    parser.add_argument(
        "--pcg-thread-policy",
        type=str,
        default=env_str(
            "JX_GGVAL_PCG_THREAD_POLICY",
            _PCG_THREAD_POLICY_DEFAULT,
        ),
        help=(
            "PCG benchmark multi-core thread policy. "
            "Aliases: rayon, blas, split."
        ),
    )
    parser.add_argument(
        "--pcg-thread-policies",
        type=str,
        default=env_str("JX_GGVAL_PCG_THREAD_POLICIES", ""),
        help=(
            "Comma-separated PCG policies to compare on the same payload. "
            "Special token compare expands to rayon,blas,split. "
            "When omitted, the benchmark auto-sweeps all three if the single-policy "
            "setting stays at its default."
        ),
    )
    parser.add_argument(
        "--rowmajor-kernel-policy",
        type=str,
        default=env_str("JX_GGVAL_ROWMAJOR_F32_KERNEL", env_str("JX_ROWMAJOR_F32_KERNEL", _ROWMAJOR_KERNEL_DEFAULT)),
        help=(
            "Row-major f32 kernel strategy for HE/PCG benchmark runs. "
            "Choices: blas or rayon."
        ),
    )
    parser.add_argument(
        "--rowmajor-kernel-policies",
        type=str,
        default=env_str(
            "JX_GGVAL_ROWMAJOR_F32_KERNEL_POLICIES",
            ",".join(_ROWMAJOR_KERNEL_COMPARE_DEFAULT),
        ),
        help=(
            "Comma-separated row-major f32 kernel strategies to compare in HE/PCG benchmarks. "
            "Use empty string to disable comparison."
        ),
    )
    parser.add_argument(
        "--he-stage-timing",
        action="store_true",
        default=env_truthy("JX_GGVAL_HE_STAGE_TIMING", False),
        help="Enable Rust HE stage timing logs during --testHE runs.",
    )
    parser.add_argument(
        "--he-stage-log-every",
        type=int,
        default=env_int("JX_GGVAL_HE_STAGE_LOG_EVERY", 8, min_value=1),
        help="Emit HE trace timing log every N trace batches during --testHE.",
    )
    parser.add_argument(
        "--pcg-stage-timing",
        action="store_true",
        default=env_truthy("JX_GGVAL_PCG_STAGE_TIMING", False),
        help="Enable Rust rrBLUP-PCG stage timing logs during --testPCG runs.",
    )
    parser.add_argument(
        "--pcg-stage-log-every",
        type=int,
        default=env_int("JX_GGVAL_PCG_STAGE_LOG_EVERY", 100, min_value=1),
        help="Emit rrBLUP-PCG timing log every N PCG iterations during --testPCG.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir)
    logdir = Path(args.logdir or env_str("JX_GGVAL_LOGDIR", str(outdir / "logs")))

    os.chdir(ROOT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    check_jx_available()

    ran_special = False
    special_total = int(bool(args.test_he)) + int(bool(args.test_pcg))
    special_done = 0
    if args.test_he:
        he_thread_checks(
            outdir,
            bench_threads=int(args.bench_threads),
            thread_policy=str(args.he_thread_policy),
            thread_policies=str(args.he_thread_policies),
            rowmajor_kernel_policy=str(args.rowmajor_kernel_policy),
            rowmajor_kernel_policies=str(args.rowmajor_kernel_policies),
            stage_timing=bool(args.he_stage_timing),
            stage_log_every=int(args.he_stage_log_every),
        )
        special_done += 1
        report_special_progress(special_done, special_total, "HE")
        ran_special = True
    if args.test_pcg:
        pcg_thread_checks(
            outdir,
            bench_threads=int(args.bench_threads),
            thread_policy=str(args.pcg_thread_policy),
            thread_policies=str(args.pcg_thread_policies),
            rowmajor_kernel_policy=str(args.rowmajor_kernel_policy),
            rowmajor_kernel_policies=str(args.rowmajor_kernel_policies),
            stage_timing=bool(args.pcg_stage_timing),
            stage_log_every=int(args.pcg_stage_log_every),
        )
        special_done += 1
        report_special_progress(special_done, special_total, "PCG")
        ran_special = True
    if not ran_special and args.mode == "smoke":
        smoke_flow(outdir, logdir, args.threads, args.cv)
        backend_thread_checks(outdir, bench_threads=int(args.bench_threads))
    elif not ran_special:
        full_flow(outdir, logdir, postgs_enabled=not args.no_postgs)
        backend_thread_checks(outdir, bench_threads=int(args.bench_threads))

    step("Validation completed")
    if args.test_he or args.test_pcg:
        parts: list[str] = []
        if args.test_he:
            parts.append("testHE")
        if args.test_pcg:
            parts.append("testPCG")
        mode_text = ",".join(parts)
    else:
        mode_text = args.mode
    print(f"Mode      : {mode_text}")
    print(f"Output dir: {outdir}")
    print(f"Log dir   : {logdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
