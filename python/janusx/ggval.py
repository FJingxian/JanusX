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

SIM_PREFIX_NAME = "ggval_sim"
SIM_TXT_PREFIX_NAME = "ggval_sim_txt"
SIM_HMP_PREFIX_NAME = "ggval_sim_hmp"
SIM_VCF_PREFIX_NAME = "ggval_sim_vcf"
SIM_ROUNDTRIP_PREFIX_NAME = "ggval_roundtrip"


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
    cwd: Path | None = None,
    stdout_path: Path | None = None,
    stderr_to_stdout: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"+ {cmd_to_text(cmd)}")
    proc_env = os.environ.copy()
    cwd_use = Path.cwd() if cwd is None else Path(cwd)

    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as out:
            return subprocess.run(
                cmd,
                cwd=str(cwd_use),
                text=True,
                env=proc_env,
                stdout=out,
                stderr=subprocess.STDOUT if stderr_to_stdout else None,
                check=True,
            )

    return subprocess.run(cmd, cwd=str(cwd_use), text=True, env=proc_env, check=True)


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


def print_log(log_path: Path) -> None:
    print(f"----- LOG BEGIN: {log_path} -----", file=sys.stderr)
    try:
        print(log_path.read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
    except FileNotFoundError:
        print("<log file missing>", file=sys.stderr)
    print(f"----- LOG END: {log_path} -----", file=sys.stderr)


def find_grm(prefix: Path) -> Path:
    base = str(prefix)
    candidates = [
        Path(f"{base}.cGRM.npy"),
        Path(f"{base}.grm.npy"),
    ]
    return require_any_file("GRM file not found for GWAS (-k).", candidates)


def find_gs_summary(outdir: Path, prefix_name: str) -> Path:
    return require_any_file(
        "GS summary output missing",
        [
            outdir / f"{prefix_name}.gs.model" / "summary.json",
            outdir / f"{prefix_name}.gs" / "summary.json",
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
    if name.endswith(".gs.tsv") or name.endswith(".gebv.tsv"):
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


def find_gwas_tsv_files(outdir: Path, prefix_name: str) -> list[Path]:
    candidates = sorted(p for p in outdir.glob(f"{prefix_name}.test*.tsv") if p.is_file())
    files = [p for p in candidates if _looks_like_postgwas_input(p)]
    if not files:
        print("Scanned TSV candidates:", file=sys.stderr)
        for p in candidates:
            header = "\t".join(_tsv_header(p)[:8])
            print(f"  {p}\t[{header}]", file=sys.stderr)
        fail("No valid GWAS result TSV files found for postgwas.")
    return files


def require_no_matching_files(
    base_dir: Path,
    pattern: str,
    message: str,
) -> None:
    matches = sorted(p for p in base_dir.glob(pattern) if p.exists())
    if len(matches) == 0:
        return
    print("Unexpected files:", file=sys.stderr)
    for path in matches:
        print(f"  {path}", file=sys.stderr)
    fail(message)


def require_tsv_columns(path: Path, expected: list[str], message: str) -> None:
    header = _tsv_header(path)
    if header != list(expected):
        fail(
            f"{message}: expected header={expected}, got={header} ({path})"
        )


def _resolve_sim_config(mode: str) -> tuple[int, int, int]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "smoke":
        nsnp_k = env_int("JX_GGVAL_SMOKE_NSNP_K", 2, min_value=1)
        n_individuals = env_int("JX_GGVAL_SMOKE_NIND", 80, min_value=20)
    else:
        nsnp_k = env_int("JX_GGVAL_FULL_NSNP_K", 10, min_value=1)
        n_individuals = env_int("JX_GGVAL_FULL_NIND", 240, min_value=40)
    seed = env_int("JX_GGVAL_SIM_SEED", 20260707, min_value=0)
    return int(nsnp_k), int(n_individuals), int(seed)


def cleanup_validation_artifacts(outdir: Path) -> None:
    patterns = [
        f"{SIM_PREFIX_NAME}*",
        f"{SIM_TXT_PREFIX_NAME}*",
        f"{SIM_HMP_PREFIX_NAME}*",
        f"{SIM_VCF_PREFIX_NAME}*",
        f"{SIM_ROUNDTRIP_PREFIX_NAME}*",
        f"~{SIM_PREFIX_NAME}*",
        f"~{SIM_TXT_PREFIX_NAME}*",
        f"~{SIM_HMP_PREFIX_NAME}*",
        f"~{SIM_VCF_PREFIX_NAME}*",
        f"~{SIM_ROUNDTRIP_PREFIX_NAME}*",
    ]
    for pattern in patterns:
        for path in sorted(outdir.glob(pattern)):
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink()
            except FileNotFoundError:
                pass


def build_validation_dataset(
    outdir: Path,
    *,
    nsnp_k: int,
    n_individuals: int,
    seed: int,
    export_debug_formats: bool,
) -> dict[str, Path]:
    sim_prefix = outdir / SIM_PREFIX_NAME
    txt_prefix = outdir / SIM_TXT_PREFIX_NAME
    hmp_prefix = outdir / SIM_HMP_PREFIX_NAME
    vcf_prefix = outdir / SIM_VCF_PREFIX_NAME
    roundtrip_prefix = outdir / SIM_ROUNDTRIP_PREFIX_NAME

    run(
        [
            "jx",
            "sim",
            str(int(nsnp_k)),
            str(int(n_individuals)),
            str(sim_prefix),
            "-seed",
            str(int(seed)),
            "-trait-name",
            "test",
        ]
    )
    for suffix, label in [
        (".bed", "Simulated BED output missing"),
        (".bim", "Simulated BIM output missing"),
        (".fam", "Simulated FAM output missing"),
        (".pheno", "Simulated phenotype output missing"),
        (".pheno.txt", "Simulated phenotype table missing"),
        (".pheno.NA.txt", "Simulated phenotype NA table missing"),
    ]:
        require_file(Path(f"{sim_prefix}{suffix}"), label)

    run(
        [
            "jx",
            "gformat",
            "-bfile",
            str(sim_prefix),
            "-fmt",
            "txt",
            "-o",
            str(outdir),
            "-prefix",
            SIM_TXT_PREFIX_NAME,
        ]
    )
    for suffix, label in [
        (".txt", "TXT export missing"),
        (".id", "TXT export sample-id sidecar missing"),
        (".site", "TXT export site sidecar missing"),
    ]:
        require_file(Path(f"{txt_prefix}{suffix}"), label)

    run(
        [
            "jx",
            "gformat",
            "-file",
            str(txt_prefix.with_suffix(".txt")),
            "-fmt",
            "plink",
            "-o",
            str(outdir),
            "-prefix",
            SIM_ROUNDTRIP_PREFIX_NAME,
        ]
    )
    for suffix, label in [
        (".bed", "Roundtrip PLINK BED missing"),
        (".bim", "Roundtrip PLINK BIM missing"),
        (".fam", "Roundtrip PLINK FAM missing"),
    ]:
        require_file(Path(f"{roundtrip_prefix}{suffix}"), label)

    if export_debug_formats:
        run(
            [
                "jx",
                "gformat",
                "-bfile",
                str(sim_prefix),
                "-fmt",
                "hmp",
                "-o",
                str(outdir),
                "-prefix",
                SIM_HMP_PREFIX_NAME,
            ]
        )
        require_file(hmp_prefix.with_suffix(".hmp"), "HMP export missing")
        run(
            [
                "jx",
                "gformat",
                "-bfile",
                str(sim_prefix),
                "-fmt",
                "vcf",
                "-o",
                str(outdir),
                "-prefix",
                SIM_VCF_PREFIX_NAME,
            ]
        )
        require_file(Path(f"{vcf_prefix}.vcf.gz"), "VCF export missing")

    return {
        "sim_prefix": sim_prefix,
        "pheno_plain": Path(f"{sim_prefix}.pheno"),
        "pheno_txt": Path(f"{sim_prefix}.pheno.txt"),
        "pheno_na_txt": Path(f"{sim_prefix}.pheno.NA.txt"),
        "txt_prefix": txt_prefix,
        "txt_matrix": Path(f"{txt_prefix}.txt"),
        "txt_id": Path(f"{txt_prefix}.id"),
        "txt_site": Path(f"{txt_prefix}.site"),
        "roundtrip_prefix": roundtrip_prefix,
        "hmp_prefix": hmp_prefix,
        "vcf_prefix": vcf_prefix,
    }


def validate_gs_file_input_debug_outputs(
    outdir: Path,
    *,
    prefix_name: str,
    trait_name: str,
    expected_models: Iterable[str] | None = None,
) -> Path:
    gebv_path = outdir / f"{prefix_name}.{trait_name}.gebv.tsv"
    require_file(gebv_path, "GS GEBV output missing")
    if (outdir / f"{prefix_name}.{trait_name}.gs.tsv").exists():
        fail("Legacy GS prediction output (.gs.tsv) should not be generated.")
    require_no_matching_files(
        outdir,
        f"{prefix_name}.{trait_name}.gs.effect*",
        "Legacy GS effect side output should not be generated.",
    )

    summary_path = find_gs_summary(outdir, prefix_name)
    model_dir = outdir / f"{prefix_name}.gs.model"
    model_headers = {
        "BLUP": ["chr", "pos", "snp", "beta"],
        "BayesA": ["chr", "pos", "snp", "beta"],
        "BayesB": ["chr", "pos", "snp", "beta", "pip"],
        "BayesCpi": ["chr", "pos", "snp", "beta", "pip"],
    }
    if expected_models is None:
        expected_model_names = ["BLUP", "BayesA", "BayesB", "BayesCpi"]
    else:
        expected_model_names = [str(x).strip() for x in expected_models if str(x).strip()]
    for model_name in expected_model_names:
        expected_header = model_headers.get(model_name)
        if expected_header is None:
            fail(f"Unsupported GS model expectation: {model_name!r}")
        model_path = model_dir / f"{trait_name}.{model_name}.jxmodel"
        require_file(model_path, f"{model_name} text .jxmodel missing")
        require_tsv_columns(
            model_path,
            expected_header,
            f"{model_name} text .jxmodel header mismatch",
        )
    require_file(
        outdir / f"~{prefix_name}.snp0.bed",
        "Expected GS file-input cached PLINK BED is missing.",
    )
    require_any_file(
        "Unified GS/GWAS GRM cache missing for file-input route.",
        outdir.glob(f"~{prefix_name}.maf*.geno*.snp0.cGRM.npy"),
    )
    require_no_matching_files(
        outdir,
        f"~{prefix_name}.snp0.maf*.cGRM.npy*",
        "Deprecated duplicated snp0-prefixed GRM cache should not be generated.",
    )
    require_no_matching_files(
        outdir,
        f"~~{prefix_name}*",
        "Double-tilde genotype/GRM cache should not be generated.",
    )
    require_no_matching_files(
        outdir,
        f"~{prefix_name}*.pmeta*",
        "Legacy packed metadata sidecars should not be generated for GS file-input route.",
    )
    return summary_path


def validate_reml_outputs(
    outdir: Path,
    *,
    prefix_name: str = SIM_PREFIX_NAME,
) -> None:
    base = outdir / f"{prefix_name}.pheno"
    require_file(Path(f"{base}.reml.summary.tsv"), "REML summary output missing")
    require_file(Path(f"{base}.blue.txt"), "REML BLUE output missing")
    require_file(Path(f"{base}.gblup.txt"), "REML GBLUP output missing")


def check_jx_available() -> None:
    if shutil.which("jx") is None:
        fail("jx command not found in PATH")
    run(["jx", "--version"])


def smoke_flow(outdir: Path, logdir: Path, threads: int, cv_folds: int) -> None:
    cleanup_validation_artifacts(outdir)
    nsnp_k, n_individuals, seed = _resolve_sim_config("smoke")

    step("SMOKE 1. Build simulated validation dataset")
    paths = build_validation_dataset(
        outdir,
        nsnp_k=nsnp_k,
        n_individuals=n_individuals,
        seed=seed,
        export_debug_formats=False,
    )
    sim_prefix = paths["sim_prefix"]
    pheno_txt = paths["pheno_txt"]
    txt_matrix = paths["txt_matrix"]
    sep()

    step("SMOKE 2. GWAS runtime check on simulated PLINK input")
    run(["jx", "grm", "-bfile", str(sim_prefix), "-o", str(outdir)])
    run_pca_with_backend_report(
        ["jx", "pca", "-bfile", str(sim_prefix), "-o", str(outdir)],
        log_path=logdir / "jx_pca_smoke.log",
    )
    grm_k = find_grm(sim_prefix)
    require_file(sim_prefix.with_suffix(".eigenvec"), "PCA eigenvec output missing")
    run(
        [
            "jx",
            "gwas",
            "-bfile",
            str(sim_prefix),
            "-p",
            str(pheno_txt),
            "-lmm",
            "-lm",
            "-k",
            str(grm_k),
            "-c",
            str(sim_prefix.with_suffix(".eigenvec")),
            "-n",
            "test",
            "-t",
            str(threads),
            "-o",
            str(outdir),
        ]
    )
    find_gwas_tsv_files(outdir, SIM_PREFIX_NAME)
    sep()

    step("SMOKE 3. GS file-input runtime check")
    run(
        [
            "jx",
            "gs",
            "-file",
            str(txt_matrix),
            "-p",
            str(pheno_txt),
            "-n",
            "test",
            "-BLUP",
            "-cv",
            str(cv_folds),
            "-save-model",
            "-t",
            str(threads),
            "-o",
            str(outdir),
        ]
    )
    validate_gs_file_input_debug_outputs(
        outdir,
        prefix_name=SIM_TXT_PREFIX_NAME,
        trait_name="test",
        expected_models=["BLUP"],
    )
    sep()


def full_flow(
    outdir: Path,
    logdir: Path,
    threads: int,
    cv_folds: int,
    postgs_enabled: bool,
) -> None:
    cleanup_validation_artifacts(outdir)
    nsnp_k, n_individuals, seed = _resolve_sim_config("full")

    step("STEP 1. Build simulated validation dataset")
    paths = build_validation_dataset(
        outdir,
        nsnp_k=nsnp_k,
        n_individuals=n_individuals,
        seed=seed,
        export_debug_formats=True,
    )
    sim_prefix = paths["sim_prefix"]
    pheno_txt = paths["pheno_txt"]
    txt_matrix = paths["txt_matrix"]
    sep()

    step("STEP 2. Validate GWAS and postGWAS on simulated data")
    run(["jx", "grm", "-bfile", str(sim_prefix), "-o", str(outdir), "-m", "1"])
    run(["jx", "grm", "-bfile", str(sim_prefix), "-o", str(outdir), "-m", "2"])
    run_pca_with_backend_report(
        ["jx", "pca", "-bfile", str(sim_prefix), "-o", str(outdir)],
        log_path=logdir / "jx_pca_full.log",
    )
    run(["jx", "pca", "-bfile", str(sim_prefix), "-rsvd", "-o", str(outdir)])
    grm_k = find_grm(sim_prefix)
    require_file(sim_prefix.with_suffix(".eigenvec"), "PCA eigenvec output missing")

    run(
        [
            "jx",
            "gwas",
            "-bfile",
            str(sim_prefix),
            "-p",
            str(pheno_txt),
            "-farmcpu",
            "-lmm",
            "-lmm2",
            "-lm",
            "-fvlmm",
            "-splmm",
            "-k",
            str(grm_k),
            "-c",
            str(sim_prefix.with_suffix(".eigenvec")),
            "-n",
            "test",
            "-t",
            str(threads),
            "-o",
            str(outdir),
        ]
    )

    gwas_files = find_gwas_tsv_files(outdir, SIM_PREFIX_NAME)
    run(
        [
            "jx",
            "postgwas",
            "-gwasfile",
            *[str(p) for p in gwas_files],
            "-bfile",
            str(sim_prefix),
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
    require_any_file(
        "postGWAS Manhattan output missing",
        outdir.glob(f"{SIM_PREFIX_NAME}.test*.manh.pdf"),
    )
    require_any_file(
        "postGWAS QQ output missing",
        outdir.glob(f"{SIM_PREFIX_NAME}.test*.qq.pdf"),
    )
    sep()

    step("STEP 3. Validate GS and postGS on simulated TXT input")
    run(
        [
            "jx",
            "gs",
            "-file",
            str(txt_matrix),
            "-p",
            str(pheno_txt),
            "-n",
            "test",
            "-BLUP",
            "-BayesA",
            "-BayesB",
            "-BayesCpi",
            "-cv",
            str(cv_folds),
            "-save-model",
            "-t",
            str(threads),
            "-o",
            str(outdir),
        ]
    )
    gs_summary = validate_gs_file_input_debug_outputs(
        outdir,
        prefix_name=SIM_TXT_PREFIX_NAME,
        trait_name="test",
    )

    if postgs_enabled:
        step("STEP 3b. Validate postGS direct-read from .jxmodel")
        run(["jx", "postgs", "-json", str(gs_summary), "-o", str(outdir)])
        require_file(
            outdir / f"{SIM_TXT_PREFIX_NAME}.test.effects.png",
            "postGS effect plot missing",
        )
        sep()

    step("STEP 4. Validate REML on simulated phenotype table")
    run(
        [
            "jx",
            "reml",
            "-file",
            str(pheno_txt),
            "-l",
            "IID",
            "-p",
            "test",
            "-k",
            str(grm_k),
            "-o",
            str(outdir),
        ]
    )
    validate_reml_outputs(outdir, prefix_name=SIM_PREFIX_NAME)
    sep()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JanusX validation runner based on simulated data.",
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
        help="Thread count for validation commands. Can also be set by JX_GGVAL_THREADS.",
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
        default=not env_truthy("JX_GGVAL_POSTGS", True),
        help="Skip postGS validation in full mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path.cwd().resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (workdir / outdir).resolve()
    logdir = Path(args.logdir or env_str("JX_GGVAL_LOGDIR", str(outdir / "logs")))
    if not logdir.is_absolute():
        logdir = (workdir / logdir).resolve()

    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    check_jx_available()

    if args.mode == "smoke":
        smoke_flow(outdir, logdir, args.threads, args.cv)
    else:
        full_flow(
            outdir,
            logdir,
            args.threads,
            args.cv,
            postgs_enabled=not args.no_postgs,
        )

    step("Validation completed")
    print(f"Mode      : {args.mode}")
    print(f"Output dir: {outdir}")
    print(f"Log dir   : {logdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
