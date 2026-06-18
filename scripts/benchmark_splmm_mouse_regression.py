#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import tarfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import psutil


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "test" / "splmm_mouse_regression"
DEFAULT_BFILE = REPO_ROOT / "test" / "mouse_hs1940"
DEFAULT_PHENO = REPO_ROOT / "example" / "mouse_hs1940.pheno"
DEFAULT_GCTA = Path("/Users/jingxianfu/Downloads/gcta-1.95.1-macOS-arm64/bin/gcta64")
DEFAULT_PYTHON = Path("/Users/jingxianfu/miniconda3/envs/jxfu/bin/python")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark JanusX SparseLMM new-vs-old exact/approx paths on mouse_hs1940, "
            "with GCTA fastGWA / fastGWA-exact reference runs."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--old-tree",
        type=Path,
        default=None,
        help=(
            "Optional existing JanusX worktree/snapshot used as the `old` baseline. "
            "If omitted, the script snapshots the requested old SHA into <out-dir>/head_worktree."
        ),
    )
    parser.add_argument(
        "--old-sha",
        type=str,
        default=None,
        help=(
            "Optional git revision to snapshot as the `old` baseline when --old-tree is not provided. "
            "Defaults to the current HEAD for backwards compatibility."
        ),
    )
    parser.add_argument("--bfile", type=Path, default=DEFAULT_BFILE)
    parser.add_argument("--pheno", type=Path, default=DEFAULT_PHENO)
    parser.add_argument(
        "--jx-spgrm",
        type=Path,
        default=None,
        help=(
            "Optional prebuilt JanusX sparse GRM (.spgrm). If omitted, the benchmark "
            "builds one from the mouse BED using the current tree."
        ),
    )
    parser.add_argument(
        "--jx-sparse-method",
        type=int,
        default=2,
        help="JanusX sparse GRM method: 1=centered, 2=standardized/weighted (default: %(default)s).",
    )
    parser.add_argument("--gcta-bin", type=Path, default=DEFAULT_GCTA)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--accuracy-traits",
        default="0:5",
        help="Traits for one-pass accuracy comparison. Zero-based phenotype indices.",
    )
    parser.add_argument(
        "--perf-traits",
        default="0,3,5",
        help="Traits for repeated timing/memory runs. Zero-based phenotype indices.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--maf", type=float, default=0.02)
    parser.add_argument("--geno", type=float, default=0.05)
    parser.add_argument("--sparse-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--gcta-autosome-num",
        type=int,
        default=19,
        help="Autosome count passed to GCTA for mouse_hs1940.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip `maturin develop --release` rebuilds and assume the env already matches each tree.",
    )
    return parser.parse_args()


def die(msg: str) -> NoReturn:
    raise SystemExit(msg)


def prefix_path(prefix: Path, suffix: str) -> Path:
    return Path(f"{prefix}{suffix}")


def parse_trait_spec(spec: str, max_idx: int) -> list[int]:
    out: list[int] = []
    for raw_piece in str(spec).split(","):
        piece = raw_piece.strip()
        if piece == "":
            continue
        if ":" in piece:
            left_s, right_s = piece.split(":", 1)
            left = int(left_s)
            right = int(right_s)
            if right < left:
                die(f"invalid trait range: {piece}")
            out.extend(range(left, right + 1))
        else:
            out.append(int(piece))
    uniq = sorted(set(out))
    for idx in uniq:
        if idx < 0 or idx > max_idx:
            die(f"trait index out of range: {idx}; valid range is 0..{max_idx}")
    return uniq


def read_trait_names(pheno_path: Path) -> list[str]:
    with pheno_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    if len(header) < 2:
        die(f"invalid phenotype header in {pheno_path}")
    return [str(x).strip() for x in header[1:]]


def run_checked(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=False,
        text=True,
        capture_output=capture_output,
    )
    if proc.returncode != 0:
        cmd_txt = " ".join(cmd)
        raise RuntimeError(
            f"command failed ({proc.returncode}): {cmd_txt}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def ensure_old_worktree(repo_root: Path, head_sha: str, old_tree: Path) -> None:
    marker = old_tree / ".snapshot_head"
    if old_tree.exists():
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == head_sha:
            return
        shutil.rmtree(old_tree)
    old_tree.parent.mkdir(parents=True, exist_ok=True)
    old_tree.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "archive", "--format=tar", head_sha],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git archive failed for {head_sha}\nstdout:\n{proc.stdout.decode(errors='replace')}\n"
            f"stderr:\n{proc.stderr.decode(errors='replace')}"
        )
    root_resolved = old_tree.resolve()
    with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as tf:
        for member in tf.getmembers():
            target = (old_tree / member.name).resolve()
            try:
                target.relative_to(root_resolved)
            except Exception as exc:
                raise RuntimeError(f"refusing to extract unsafe path from archive: {member.name}") from exc
        tf.extractall(path=str(old_tree))
    marker.write_text(head_sha + "\n", encoding="utf-8")
    print(f"[snapshot] extracted HEAD {head_sha[:12]} to {old_tree}")


def resolve_old_tree(args: argparse.Namespace, current_head_sha: str) -> tuple[Path, str]:
    if args.old_tree is not None:
        old_tree = args.old_tree.resolve()
        if not old_tree.exists():
            die(f"missing old-tree: {old_tree}")
        marker = old_tree / ".snapshot_head"
        old_sha = marker.read_text(encoding="utf-8").strip() if marker.exists() else "unknown"
        return old_tree, old_sha

    old_sha = str(args.old_sha).strip() if args.old_sha is not None else current_head_sha
    old_tree = args.out_dir / "head_worktree"
    ensure_old_worktree(REPO_ROOT, old_sha, old_tree)
    return old_tree, old_sha


def build_extension(tree_root: Path, python_bin: Path) -> None:
    build_out = tree_root / ".benchmark_build"
    if build_out.exists():
        shutil.rmtree(build_out)
    build_out.mkdir(parents=True, exist_ok=True)
    print(f"[build] {tree_root}")
    env = os.environ.copy()
    env["CARGO_NET_OFFLINE"] = "true"
    maturin_bin = python_bin.parent / "maturin"
    maturin_cmd = str(maturin_bin if maturin_bin.exists() else "maturin")
    run_checked(
        [
            maturin_cmd,
            "build",
            "--release",
            "--interpreter",
            str(python_bin),
            "--out",
            str(build_out),
        ],
        cwd=tree_root,
        env=env,
        capture_output=True,
    )
    wheels = sorted(build_out.glob("janusx-*.whl"))
    if not wheels:
        raise RuntimeError(f"maturin build produced no wheel in {build_out}")
    wheel_path = wheels[-1]
    _install_wheel_native_assets(tree_root=tree_root, wheel_path=wheel_path)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _install_wheel_native_assets(*, tree_root: Path, wheel_path: Path) -> None:
    pkg_root = tree_root / "python"
    cleanup_paths = [
        pkg_root / "janusx" / "janusx.abi3.so",
        pkg_root / "janusx" / ".dylibs",
        pkg_root / "janusx" / ".libs",
        pkg_root / "janusx.libs",
    ]
    for path in cleanup_paths:
        if path.exists():
            _remove_path(path)

    wanted_members: list[str] = []
    with zipfile.ZipFile(wheel_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            keep = False
            if name.startswith("janusx/") and (
                name.endswith(".so")
                or name.endswith(".pyd")
                or name.endswith(".dylib")
                or "/.dylibs/" in name
                or "/.libs/" in name
            ):
                keep = True
            if name.startswith("janusx.libs/"):
                keep = True
            if not keep:
                continue
            wanted_members.append(name)
        if not wanted_members:
            raise RuntimeError(f"no native JanusX assets found in wheel {wheel_path}")
        for name in wanted_members:
            dst = pkg_root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(zf.read(name))


def ensure_local_native_assets(tree_root: Path) -> None:
    so_path = tree_root / "python" / "janusx" / "janusx.abi3.so"
    if not so_path.exists():
        raise RuntimeError(
            f"missing local native extension for {tree_root}. "
            "Run without --skip-build first so the benchmark can build/extract the wheel."
        )


def base_runtime_env(threads: int, python_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_path)
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    env["XDG_CACHE_HOME"] = "/tmp/xdg"
    env["OMP_NUM_THREADS"] = str(int(threads))
    env["OPENBLAS_NUM_THREADS"] = str(int(threads))
    env["MKL_NUM_THREADS"] = str(int(threads))
    env["VECLIB_MAXIMUM_THREADS"] = str(int(threads))
    env["NUMEXPR_NUM_THREADS"] = str(int(threads))
    return env


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _proc_tree_rss_bytes(pid: int) -> int:
    try:
        root = psutil.Process(pid)
    except Exception:
        return 0
    total = 0
    procs = [root]
    try:
        procs.extend(root.children(recursive=True))
    except Exception:
        pass
    for proc in procs:
        try:
            total += int(proc.memory_info().rss)
        except Exception:
            continue
    return total


def run_monitored(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    sample_interval_s: float = 0.05,
) -> tuple[subprocess.CompletedProcess[str], float, float | None]:
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak_rss = 0
    while True:
        peak_rss = max(peak_rss, _proc_tree_rss_bytes(proc.pid))
        if proc.poll() is not None:
            break
        time.sleep(sample_interval_s)
    stdout, stderr = proc.communicate()
    peak_rss = max(peak_rss, _proc_tree_rss_bytes(proc.pid))
    wall_s = time.perf_counter() - t0
    completed = subprocess.CompletedProcess(
        args=cmd,
        returncode=int(proc.returncode),
        stdout=stdout,
        stderr=stderr,
    )
    peak_rss_mb = (float(peak_rss) / (1024.0 * 1024.0)) if peak_rss > 0 else None
    return completed, float(wall_s), peak_rss_mb


def load_fam_ids(fam_path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with fam_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split()
            if len(fields) >= 2:
                out.append((fields[0], fields[1]))
    return out


def write_gcta_pheno(example_pheno: Path, fam_path: Path, out_path: Path) -> None:
    fam_ids = load_fam_ids(fam_path)
    with example_pheno.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if len(header) < 2:
            die(f"invalid phenotype header in {example_pheno}")
        pheno_by_iid: dict[str, list[str]] = {}
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if not fields:
                continue
            iid = str(fields[0]).strip()
            if iid == "":
                continue
            vals = fields[1:]
            if len(vals) < len(header) - 1:
                vals.extend(["NA"] * (len(header) - 1 - len(vals)))
            pheno_by_iid[iid] = vals

    rows: list[str] = []
    for fid, iid in fam_ids:
        vals = pheno_by_iid.get(iid)
        if vals is None:
            vals = ["-9"] * (len(header) - 1)
        else:
            vals = [("-9" if str(v).strip().upper() == "NA" or str(v).strip() == "" else str(v).strip()) for v in vals]
        rows.append("\t".join([fid, iid, *vals]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def ensure_gcta_sparse_grm(
    *,
    gcta_bin: Path,
    bfile: Path,
    out_prefix: Path,
    threads: int,
    sparse_cutoff: float,
    autosome_num: int,
    maf: float,
    geno: float,
) -> None:
    sparse_path = prefix_path(out_prefix, ".grm.sp")
    id_path = prefix_path(out_prefix, ".grm.id")
    if sparse_path.exists() and id_path.exists():
        print(f"[reuse] GCTA sparse GRM: {out_prefix}")
        return
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f"[gcta] make sparse GRM: {out_prefix}")
    run_checked(
        [
            str(gcta_bin),
            "--bfile",
            str(bfile),
            "--autosome-num",
            str(int(autosome_num)),
            "--make-grm",
            "--sparse-cutoff",
            str(float(sparse_cutoff)),
            "--maf",
            str(float(maf)),
            "--geno",
            str(float(geno)),
            "--thread-num",
            str(int(threads)),
            "--out",
            str(out_prefix),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
    )


def jx_sparse_method_tag(method: int) -> str:
    return "cGRM" if int(method) == 1 else "sGRM"


def ensure_jx_sparse_grm(
    *,
    python_bin: Path,
    tree_root: Path,
    bfile: Path,
    out_prefix: Path,
    threads: int,
    sparse_cutoff: float,
    maf: float,
    geno: float,
    method: int,
) -> Path:
    sparse_path = Path(f"{out_prefix}.{jx_sparse_method_tag(method)}.spgrm")
    id_path = Path(f"{sparse_path}.id")
    if sparse_path.exists() and id_path.exists():
        print(f"[reuse] JanusX sparse GRM: {sparse_path}")
        return sparse_path
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f"[janusx] make sparse GRM: {sparse_path}")
    env = base_runtime_env(threads=threads, python_path=tree_root / "python")
    run_checked(
        [
            str(python_bin),
            "-m",
            "janusx.script.JanusX",
            "grm",
            "-bfile",
            str(bfile),
            "-m",
            str(int(method)),
            "-sparse",
            str(float(sparse_cutoff)),
            "-maf",
            str(float(maf)),
            "-geno",
            str(float(geno)),
            "-t",
            str(int(threads)),
            "-o",
            str(out_prefix.parent),
            "-prefix",
            out_prefix.name,
        ],
        cwd=tree_root,
        env=env,
        capture_output=True,
    )
    if not sparse_path.exists():
        raise RuntimeError(f"expected JanusX sparse GRM missing after build: {sparse_path}")
    if not id_path.exists():
        raise RuntimeError(f"expected JanusX sparse GRM id file missing after build: {id_path}")
    return sparse_path


def find_gcta_assoc_file(out_prefix: Path) -> Path:
    candidates = [
        prefix_path(out_prefix, ".fastGWA"),
        prefix_path(out_prefix, ".fastGWA.gz"),
        prefix_path(out_prefix, ".mlma"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise RuntimeError(f"cannot locate GCTA association output for prefix {out_prefix}")


def trait_tag(trait_names: list[str], trait_idx: int) -> str:
    return str(trait_names[int(trait_idx)])


def run_janusx_trait(
    *,
    version_label: str,
    tree_root: Path,
    python_bin: Path,
    bfile: Path,
    pheno: Path,
    jx_spgrm: Path,
    out_dir: Path,
    trait_names: list[str],
    trait_idx: int,
    mode: str,
    threads: int,
    maf: float,
    geno: float,
    rep: int,
) -> dict[str, Any]:
    mode_flag = "-splmm" if mode == "exact" else "-splmm-approx"
    prefix = f"{version_label}_{mode}_t{trait_idx}_r{rep}"
    trait_name = trait_tag(trait_names, trait_idx)
    method_dir = out_dir / "runs" / version_label / mode
    method_dir.mkdir(parents=True, exist_ok=True)
    env = base_runtime_env(threads=threads, python_path=tree_root / "python")
    cmd = [
        str(python_bin),
        "-m",
        "janusx.script.JanusX",
        "gwas",
        "-bfile",
        str(bfile),
        "-p",
        str(pheno),
        "-n",
        str(int(trait_idx)),
        mode_flag,
        "-spk",
        str(jx_spgrm),
        "-maf",
        str(float(maf)),
        "-geno",
        str(float(geno)),
        "-t",
        str(int(threads)),
        "-o",
        str(method_dir),
        "-prefix",
        prefix,
    ]
    print(f"[run] JanusX {version_label} {mode} trait={trait_name} rep={rep}")
    proc, wall_s, peak_rss_mb = run_monitored(
        cmd,
        cwd=str(tree_root),
        env=env,
    )
    stdout_path = method_dir / f"{prefix}.stdout.log"
    stderr_path = method_dir / f"{prefix}.stderr.log"
    write_text(stdout_path, proc.stdout)
    write_text(stderr_path, proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"JanusX run failed ({version_label}, {mode}, trait={trait_idx}, rep={rep})\n"
            f"stdout: {stdout_path}\nstderr: {stderr_path}"
        )
    assoc_path = method_dir / f"{prefix}.{trait_name}.splmm.tsv"
    if not assoc_path.exists():
        raise RuntimeError(f"expected JanusX output missing: {assoc_path}")
    return {
        "tool": "janusx",
        "version": version_label,
        "mode": mode,
        "trait_idx": int(trait_idx),
        "trait_name": trait_name,
        "rep": int(rep),
        "wall_s": float(wall_s),
        "max_rss_mb": peak_rss_mb,
        "assoc_path": str(assoc_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def run_gcta_trait(
    *,
    mode: str,
    gcta_bin: Path,
    bfile: Path,
    sparse_prefix: Path,
    pheno_path: Path,
    out_dir: Path,
    trait_names: list[str],
    trait_idx: int,
    threads: int,
    maf: float,
    geno: float,
    autosome_num: int,
    rep: int,
) -> dict[str, Any]:
    trait_name = trait_tag(trait_names, trait_idx)
    prefix = f"gcta_{mode}_t{trait_idx}_r{rep}"
    method_dir = out_dir / "runs" / "gcta" / mode
    method_dir.mkdir(parents=True, exist_ok=True)
    scan_flag = "--fastGWA-mlm-exact" if mode == "exact" else "--fastGWA-mlm"
    cmd = [
        str(gcta_bin),
        "--bfile",
        str(bfile),
        "--autosome-num",
        str(int(autosome_num)),
        "--grm-sparse",
        str(sparse_prefix),
        scan_flag,
        "--pheno",
        str(pheno_path),
        "--mpheno",
        str(int(trait_idx) + 1),
        "--maf",
        str(float(maf)),
        "--geno",
        str(float(geno)),
        "--thread-num",
        str(int(threads)),
        "--out",
        str(method_dir / prefix),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(int(threads))
    print(f"[run] GCTA {mode} trait={trait_name} rep={rep}")
    proc, wall_s, peak_rss_mb = run_monitored(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
    )
    out_prefix = method_dir / prefix
    stdout_path = method_dir / f"{prefix}.stdout.log"
    stderr_path = method_dir / f"{prefix}.stderr.log"
    write_text(stdout_path, proc.stdout)
    write_text(stderr_path, proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"GCTA run failed ({mode}, trait={trait_idx}, rep={rep})\n"
            f"stdout: {stdout_path}\nstderr: {stderr_path}"
        )
    assoc_path = find_gcta_assoc_file(out_prefix)
    return {
        "tool": "gcta",
        "version": "gcta",
        "mode": mode,
        "trait_idx": int(trait_idx),
        "trait_name": trait_name,
        "rep": int(rep),
        "wall_s": float(wall_s),
        "max_rss_mb": peak_rss_mb,
        "assoc_path": str(assoc_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def load_janusx_assoc(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            snp = str(row["snp"])
            out[snp] = {
                "a0": str(row["allele0"]),
                "a1": str(row["allele1"]),
                "beta": float(row["beta"]),
                "se": float(row["se"]),
                "p": float(row["pwald"]),
                "chisq": float(row["chisq"]),
            }
    return out


def load_fastgwa_assoc(path: Path) -> dict[str, dict[str, float]]:
    opener = open
    if str(path).endswith(".gz"):
        import gzip

        opener = gzip.open
    out: dict[str, dict[str, float]] = {}
    with opener(path, "rt", encoding="utf-8") as fh:
        header = fh.readline().strip().split()
        idx = {name.upper(): i for i, name in enumerate(header)}
        snp_i = idx.get("SNP")
        beta_i = idx.get("BETA")
        se_i = idx.get("SE")
        p_i = idx.get("P")
        if None in {snp_i, beta_i, se_i, p_i}:
            raise RuntimeError(f"unexpected fastGWA header in {path}: {header}")
        for line in fh:
            fields = line.strip().split()
            if len(fields) <= max(int(snp_i), int(beta_i), int(se_i), int(p_i)):
                continue
            snp = fields[int(snp_i)]
            out[snp] = {
                "a1": str(fields[idx["A1"]]),
                "a2": str(fields[idx["A2"]]),
                "beta": float(fields[int(beta_i)]),
                "se": float(fields[int(se_i)]),
                "p": float(fields[int(p_i)]),
                "chisq": float("nan"),
            }
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size == 0 or b.size == 0:
        return None
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    return float(np.corrcoef(a, b)[0, 1])


def finite_median(arr: np.ndarray) -> float | None:
    if arr.size == 0:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.median(arr))


def compare_assoc_maps(
    lhs_label: str,
    lhs_map: dict[str, dict[str, float]],
    rhs_label: str,
    rhs_map: dict[str, dict[str, float]],
) -> dict[str, Any]:
    common = sorted(set(lhs_map).intersection(rhs_map))
    if not common:
        return {
            "lhs": lhs_label,
            "rhs": rhs_label,
            "n_common": 0,
        }
    lhs_beta = np.asarray([lhs_map[s]["beta"] for s in common], dtype=np.float64)
    rhs_beta = np.asarray([rhs_map[s]["beta"] for s in common], dtype=np.float64)
    n_beta_sign_aligned = 0
    if common and {"a0", "a1"} <= set(lhs_map[common[0]].keys()) and {"a1", "a2"} <= set(rhs_map[common[0]].keys()):
        rhs_beta_aligned = rhs_beta.copy()
        for i, snp in enumerate(common):
            lhs = lhs_map[snp]
            rhs = rhs_map[snp]
            if str(lhs["a0"]) == str(rhs["a1"]) and str(lhs["a1"]) == str(rhs["a2"]):
                rhs_beta_aligned[i] = -rhs_beta_aligned[i]
                n_beta_sign_aligned += 1
            elif str(lhs["a0"]) == str(rhs["a2"]) and str(lhs["a1"]) == str(rhs["a1"]):
                n_beta_sign_aligned += 1
        rhs_beta = rhs_beta_aligned
    lhs_se = np.asarray([lhs_map[s]["se"] for s in common], dtype=np.float64)
    rhs_se = np.asarray([rhs_map[s]["se"] for s in common], dtype=np.float64)
    lhs_p = np.asarray([max(lhs_map[s]["p"], 1e-300) for s in common], dtype=np.float64)
    rhs_p = np.asarray([max(rhs_map[s]["p"], 1e-300) for s in common], dtype=np.float64)
    se_ratio = lhs_se / rhs_se
    beta_diff = lhs_beta - rhs_beta
    logp_lhs = -np.log10(lhs_p)
    logp_rhs = -np.log10(rhs_p)
    k = min(100, len(common))
    lhs_top = {common[i] for i in np.argsort(lhs_p)[:k]}
    rhs_top = {common[i] for i in np.argsort(rhs_p)[:k]}
    return {
        "lhs": lhs_label,
        "rhs": rhs_label,
        "n_common": int(len(common)),
        "beta_corr": corr(lhs_beta, rhs_beta),
        "logp_corr": corr(logp_lhs, logp_rhs),
        "median_se_ratio": finite_median(se_ratio),
        "median_abs_beta_diff": finite_median(np.abs(beta_diff)),
        "max_abs_beta_diff": float(np.max(np.abs(beta_diff))),
        "median_abs_logp_diff": finite_median(np.abs(logp_lhs - logp_rhs)),
        "top100_overlap": int(len(lhs_top.intersection(rhs_top))),
        "n_beta_sign_aligned": int(n_beta_sign_aligned),
    }


def summarize_perf(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in run_records:
        key = (str(rec["tool"]), str(rec["version"]), str(rec["mode"]), str(rec["trait_name"]))
        grouped[key].append(rec)

    summary_rows: list[dict[str, Any]] = []
    for (tool, version, mode, trait_name), rows in sorted(grouped.items()):
        wall = [float(r["wall_s"]) for r in rows]
        rss = [float(r["max_rss_mb"]) for r in rows if r["max_rss_mb"] is not None]
        summary_rows.append(
            {
                "tool": tool,
                "version": version,
                "mode": mode,
                "trait_name": trait_name,
                "n_runs": len(rows),
                "wall_median_s": float(statistics.median(wall)),
                "wall_min_s": float(min(wall)),
                "wall_max_s": float(max(wall)),
                "rss_median_mb": (float(statistics.median(rss)) if rss else None),
            }
        )

    def pick(version: str, mode: str, trait_name: str) -> dict[str, Any] | None:
        for row in summary_rows:
            if row["tool"] != "janusx":
                continue
            if row["mode"] != mode or row["trait_name"] != trait_name:
                continue
            if row["version"] == version:
                return row
        return None

    regressions: list[dict[str, Any]] = []
    trait_names = sorted({str(r["trait_name"]) for r in run_records})
    for mode in ("exact", "approx"):
        for trait_name in trait_names:
            new_row = pick("new", mode, trait_name)
            old_row = pick("old", mode, trait_name)
            if new_row is None or old_row is None:
                continue
            regressions.append(
                {
                    "mode": mode,
                    "trait_name": trait_name,
                    "new_over_old_wall": float(new_row["wall_median_s"]) / float(old_row["wall_median_s"]),
                    "new_over_old_rss": (
                        None
                        if new_row["rss_median_mb"] is None or old_row["rss_median_mb"] in (None, 0.0)
                        else float(new_row["rss_median_mb"]) / float(old_row["rss_median_mb"])
                    ),
                }
            )

    return {
        "groups": summary_rows,
        "new_vs_old": regressions,
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_perf_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = ["tool", "version", "mode", "trait_idx", "trait_name", "rep", "wall_s", "max_rss_mb", "assoc_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in cols})


def write_accuracy_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "mode",
        "trait_idx",
        "trait_name",
        "lhs",
        "rhs",
        "n_common",
        "beta_corr",
        "logp_corr",
        "median_se_ratio",
        "median_abs_beta_diff",
        "max_abs_beta_diff",
        "median_abs_logp_diff",
        "top100_overlap",
        "n_beta_sign_aligned",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in cols})


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.old_tree is not None:
        args.old_tree = args.old_tree.resolve()
    args.bfile = args.bfile.resolve()
    args.pheno = args.pheno.resolve()
    if args.jx_spgrm is not None:
        args.jx_spgrm = args.jx_spgrm.resolve()
    args.gcta_bin = args.gcta_bin.resolve()
    args.python_bin = args.python_bin.resolve()
    if not prefix_path(args.bfile, ".bed").exists():
        die(f"missing BED file: {prefix_path(args.bfile, '.bed')}")
    if not args.pheno.exists():
        die(f"missing phenotype file: {args.pheno}")
    if args.jx_spgrm is not None and not args.jx_spgrm.exists():
        die(f"missing JanusX sparse GRM: {args.jx_spgrm}")
    if not args.gcta_bin.exists():
        die(f"missing GCTA binary: {args.gcta_bin}")
    if not args.python_bin.exists():
        die(f"missing Python binary: {args.python_bin}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trait_names = read_trait_names(args.pheno)
    max_trait_idx = len(trait_names) - 1
    accuracy_traits = parse_trait_spec(args.accuracy_traits, max_trait_idx)
    perf_traits = parse_trait_spec(args.perf_traits, max_trait_idx)
    all_traits = sorted(set(accuracy_traits).union(perf_traits))

    head_sha = run_checked(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]).stdout.strip()
    old_tree, old_sha = resolve_old_tree(args, head_sha)
    if args.skip_build:
        ensure_local_native_assets(REPO_ROOT)
    else:
        build_extension(REPO_ROOT, args.python_bin)

    if args.jx_spgrm is not None:
        jx_spgrm = args.jx_spgrm
    else:
        jx_sparse_prefix = args.out_dir / "jx_sparse" / f"{args.bfile.name}_sparse_m{int(args.jx_sparse_method)}"
        jx_spgrm = ensure_jx_sparse_grm(
            python_bin=args.python_bin,
            tree_root=REPO_ROOT,
            bfile=args.bfile,
            out_prefix=jx_sparse_prefix,
            threads=args.threads,
            sparse_cutoff=args.sparse_cutoff,
            maf=args.maf,
            geno=args.geno,
            method=args.jx_sparse_method,
        )

    gcta_dir = args.out_dir / "gcta"
    gcta_sparse_prefix = gcta_dir / "mouse_hs1940_sparse"
    gcta_pheno = gcta_dir / "mouse_hs1940.gcta.pheno"
    write_gcta_pheno(args.pheno, prefix_path(args.bfile, ".fam"), gcta_pheno)
    ensure_gcta_sparse_grm(
        gcta_bin=args.gcta_bin,
        bfile=args.bfile,
        out_prefix=gcta_sparse_prefix,
        threads=args.threads,
        sparse_cutoff=args.sparse_cutoff,
        autosome_num=args.gcta_autosome_num,
        maf=args.maf,
        geno=args.geno,
    )

    meta = {
        "head_sha": head_sha,
        "old_sha": old_sha,
        "repo_root": str(REPO_ROOT),
        "old_tree": str(old_tree),
        "bfile": str(args.bfile),
        "pheno": str(args.pheno),
        "jx_spgrm": str(jx_spgrm),
        "jx_sparse_method": int(args.jx_sparse_method),
        "gcta_sparse_prefix": str(gcta_sparse_prefix),
        "accuracy_traits": accuracy_traits,
        "perf_traits": perf_traits,
        "threads": int(args.threads),
        "repeats": int(args.repeats),
        "maf": float(args.maf),
        "geno": float(args.geno),
        "trait_names": trait_names,
    }
    write_json(args.out_dir / "meta.json", meta)

    accuracy_runs: dict[tuple[str, str, int], dict[str, Any]] = {}
    perf_runs: list[dict[str, Any]] = []

    for mode in ("exact", "approx"):
        for trait_idx in all_traits:
            rec = run_janusx_trait(
                version_label="new",
                tree_root=REPO_ROOT,
                python_bin=args.python_bin,
                bfile=args.bfile,
                pheno=args.pheno,
                jx_spgrm=jx_spgrm,
                out_dir=args.out_dir,
                trait_names=trait_names,
                trait_idx=trait_idx,
                mode=mode,
                threads=args.threads,
                maf=args.maf,
                geno=args.geno,
                rep=1,
            )
            if trait_idx in accuracy_traits:
                accuracy_runs[("new", mode, trait_idx)] = rec
            if trait_idx in perf_traits:
                perf_runs.append(rec)
        for rep in range(2, int(args.repeats) + 1):
            for trait_idx in perf_traits:
                perf_runs.append(
                    run_janusx_trait(
                        version_label="new",
                        tree_root=REPO_ROOT,
                        python_bin=args.python_bin,
                        bfile=args.bfile,
                        pheno=args.pheno,
                        jx_spgrm=jx_spgrm,
                        out_dir=args.out_dir,
                        trait_names=trait_names,
                        trait_idx=trait_idx,
                        mode=mode,
                        threads=args.threads,
                        maf=args.maf,
                        geno=args.geno,
                        rep=rep,
                    )
                )

    if args.skip_build:
        ensure_local_native_assets(old_tree)
    else:
        build_extension(old_tree, args.python_bin)

    for mode in ("exact", "approx"):
        for trait_idx in all_traits:
            rec = run_janusx_trait(
                version_label="old",
                tree_root=old_tree,
                python_bin=args.python_bin,
                bfile=args.bfile,
                pheno=args.pheno,
                jx_spgrm=jx_spgrm,
                out_dir=args.out_dir,
                trait_names=trait_names,
                trait_idx=trait_idx,
                mode=mode,
                threads=args.threads,
                maf=args.maf,
                geno=args.geno,
                rep=1,
            )
            if trait_idx in accuracy_traits:
                accuracy_runs[("old", mode, trait_idx)] = rec
            if trait_idx in perf_traits:
                perf_runs.append(rec)
        for rep in range(2, int(args.repeats) + 1):
            for trait_idx in perf_traits:
                perf_runs.append(
                    run_janusx_trait(
                        version_label="old",
                        tree_root=old_tree,
                        python_bin=args.python_bin,
                        bfile=args.bfile,
                        pheno=args.pheno,
                        jx_spgrm=jx_spgrm,
                        out_dir=args.out_dir,
                        trait_names=trait_names,
                        trait_idx=trait_idx,
                        mode=mode,
                        threads=args.threads,
                        maf=args.maf,
                        geno=args.geno,
                        rep=rep,
                    )
                )

    if not args.skip_build:
        build_extension(REPO_ROOT, args.python_bin)

    for mode in ("exact", "approx"):
        for trait_idx in all_traits:
            rec = run_gcta_trait(
                mode=mode,
                gcta_bin=args.gcta_bin,
                bfile=args.bfile,
                sparse_prefix=gcta_sparse_prefix,
                pheno_path=gcta_pheno,
                out_dir=args.out_dir,
                trait_names=trait_names,
                trait_idx=trait_idx,
                threads=args.threads,
                maf=args.maf,
                geno=args.geno,
                autosome_num=args.gcta_autosome_num,
                rep=1,
            )
            if trait_idx in accuracy_traits:
                accuracy_runs[("gcta", mode, trait_idx)] = rec
            if trait_idx in perf_traits:
                perf_runs.append(rec)
        for rep in range(2, int(args.repeats) + 1):
            for trait_idx in perf_traits:
                perf_runs.append(
                    run_gcta_trait(
                        mode=mode,
                        gcta_bin=args.gcta_bin,
                        bfile=args.bfile,
                        sparse_prefix=gcta_sparse_prefix,
                        pheno_path=gcta_pheno,
                        out_dir=args.out_dir,
                        trait_names=trait_names,
                        trait_idx=trait_idx,
                        threads=args.threads,
                        maf=args.maf,
                        geno=args.geno,
                        autosome_num=args.gcta_autosome_num,
                        rep=rep,
                    )
                )

    accuracy_rows: list[dict[str, Any]] = []
    for mode in ("exact", "approx"):
        for trait_idx in accuracy_traits:
            rec_new = accuracy_runs[("new", mode, trait_idx)]
            rec_old = accuracy_runs[("old", mode, trait_idx)]
            rec_gcta = accuracy_runs[("gcta", mode, trait_idx)]
            map_new = load_janusx_assoc(Path(rec_new["assoc_path"]))
            map_old = load_janusx_assoc(Path(rec_old["assoc_path"]))
            map_gcta = load_fastgwa_assoc(Path(rec_gcta["assoc_path"]))
            for lhs, rhs, lhs_map, rhs_map in [
                ("new", "old", map_new, map_old),
                ("new", "gcta", map_new, map_gcta),
                ("old", "gcta", map_old, map_gcta),
            ]:
                row = compare_assoc_maps(lhs, lhs_map, rhs, rhs_map)
                row["mode"] = mode
                row["trait_idx"] = int(trait_idx)
                row["trait_name"] = trait_tag(trait_names, trait_idx)
                accuracy_rows.append(row)

    perf_summary = summarize_perf(perf_runs)
    write_perf_tsv(args.out_dir / "perf_runs.tsv", perf_runs)
    write_accuracy_tsv(args.out_dir / "accuracy.tsv", accuracy_rows)
    write_json(args.out_dir / "perf_summary.json", perf_summary)
    write_json(args.out_dir / "accuracy.json", accuracy_rows)

    print(f"[done] outputs written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
