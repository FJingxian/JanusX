#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


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
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as out:
            return subprocess.run(
                cmd,
                cwd=str(cwd),
                text=True,
                stdout=out,
                stderr=subprocess.STDOUT if stderr_to_stdout else None,
                check=True,
            )

    return subprocess.run(cmd, cwd=str(cwd), text=True, check=True)


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir)
    logdir = Path(args.logdir or env_str("JX_GGVAL_LOGDIR", str(outdir / "logs")))

    os.chdir(ROOT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    check_jx_available()

    if args.mode == "smoke":
        smoke_flow(outdir, logdir, args.threads, args.cv)
    else:
        full_flow(outdir, postgs_enabled=not args.no_postgs)

    step("Validation completed")
    print(f"Mode      : {args.mode}")
    print(f"Output dir: {outdir}")
    print(f"Log dir   : {logdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())