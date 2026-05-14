#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
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


def env_truthy(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def step(message: str) -> None:
    print()
    print(f"==> {message}")


def sep() -> None:
    print("=" * 44)


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
    run(["jx", "pca", "-bfile", str(outdir / "mouse_hs1940"), "-o", str(outdir)])
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


def full_flow(outdir: Path, postgs_enabled: bool) -> None:
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
    run(["jx", "pca", "-bfile", str(outdir / "mouse_hs1940"), "-o", str(outdir)])
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


def backend_thread_checks(outdir: Path) -> None:
    nsnp_k = env_int("JX_GGVAL_BENCH_NSNP_K", 500, min_value=1)
    n_individuals = env_int("JX_GGVAL_BENCH_NIND", 5000, min_value=2)
    step(
        "STEP 4. Backend report and 1-core/all-core timing checks "
        f"(simulated {n_individuals} x {nsnp_k * 1000} genotype)"
    )
    sim_prefix = outdir / "__ggval_tmp_large_sim"
    full_cores = max(1, int(os.cpu_count() or 1))

    janusx, jxrs = import_janusx_runtime()
    report_janusx_runtime(janusx, jxrs)

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
        min(8, trace_samples),
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
    full_cores = max(1, int(os.cpu_count() or 1))

    janusx, jxrs = import_janusx_runtime()
    report_janusx_runtime(janusx, jxrs)
    print(f"HE BLAS backend          : {str(jxrs.rust_sgemm_backend()).strip()}")
    if stage_timing:
        print(
            "HE stage timing         : "
            f"enabled (JX_GS_HE_STAGE_LOG_EVERY={int(max(1, stage_log_every))})"
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

        def run_he_once(threads: int) -> tuple[float, tuple[object, ...]]:
            prev_stage_timing = os.environ.get("JX_GS_HE_STAGE_TIMING")
            prev_stage_log_every = os.environ.get("JX_GS_HE_STAGE_LOG_EVERY")
            if stage_timing:
                print(f"HE timing run            : threads={int(threads)}")
                os.environ["JX_GS_HE_STAGE_TIMING"] = "1"
                os.environ["JX_GS_HE_STAGE_LOG_EVERY"] = str(int(max(1, stage_log_every)))
            t0 = time.perf_counter()
            try:
                ret = jxrs.he_pcg_bed(
                    str(bfile_prefix),
                    train_idx,
                    y,
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
                    seed=int(seed),
                    packed=packed,
                    packed_n_samples=int(n_samples),
                    maf=maf,
                    row_flip=row_flip,
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

        he_t1, he_out_1 = run_he_once(1)
        he_tn, he_out_n = run_he_once(full_cores)
        assert_he_outputs_close(he_out_1, he_out_n, full_cores)

        print(f"HE runtime    (1 core)   : {he_t1:.3f}s")
        print(f"HE runtime    ({full_cores} cores): {he_tn:.3f}s")
        if he_tn > 0.0:
            print(f"HE speedup               : {he_t1 / he_tn:.2f}x")
        if len(he_out_n) >= 11:
            print(
                "HE estimate              : "
                f"h2={float(he_out_n[2]):.6f}, "
                f"lambda={float(he_out_n[10]):.6f}, "
                f"converged={bool(he_out_n[3])}, "
                f"m_eff={int(he_out_n[6])}"
            )
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
        help="Run only the HE thread benchmark branch (1 core vs all cores).",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir)
    logdir = Path(args.logdir or env_str("JX_GGVAL_LOGDIR", str(outdir / "logs")))

    os.chdir(ROOT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    check_jx_available()

    if args.test_he:
        he_thread_checks(
            outdir,
            stage_timing=bool(args.he_stage_timing),
            stage_log_every=int(args.he_stage_log_every),
        )
    elif args.mode == "smoke":
        smoke_flow(outdir, logdir, args.threads, args.cv)
        backend_thread_checks(outdir)
    else:
        full_flow(outdir, postgs_enabled=not args.no_postgs)
        backend_thread_checks(outdir)

    step("Validation completed")
    print(f"Mode      : {'testHE' if args.test_he else args.mode}")
    print(f"Output dir: {outdir}")
    print(f"Log dir   : {logdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
