#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

TRAIT_NAME = "test"
SIM_PREFIX_NAME = "ggval_sim"
SIM_TXT_PREFIX_NAME = "ggval_sim_txt"
SIM_HMP_PREFIX_NAME = "ggval_sim_hmp"
SIM_VCF_PREFIX_NAME = "ggval_sim_vcf"
SIM_ROUNDTRIP_PREFIX_NAME = "ggval_roundtrip"
BENCH_PREFIX_NAME = "ggval_bench"
GS_BFILE_PREFIX_NAME = "ggval_gs_bfile"
GS_VCF_GS_PREFIX_NAME = "ggval_gs_vcf"
GS_HMP_GS_PREFIX_NAME = "ggval_gs_hmp"
GS_ML_PREFIX_NAME = "ggval_gs_ml"
ALL_SUITES = [
    "bench",
    "gwas",
    "gs-file",
    "gs-bfile",
    "gs-vcf",
    "gs-hmp",
    "gs-ml",
    "reml",
]
DEFAULT_FULL_SUITES = ["gwas", "gs-file", "gs-bfile", "gs-vcf", "gs-hmp", "gs-ml", "reml"]
DEFAULT_SMOKE_SUITES = ["gwas", "gs-file"]
TEXT_EFFECT_HEADERS = {
    "BLUP": ["chr", "pos", "snp", "beta"],
    "BayesA": ["chr", "pos", "snp", "beta"],
    "BayesB": ["chr", "pos", "snp", "beta", "pip"],
    "BayesCpi": ["chr", "pos", "snp", "beta", "pip"],
}
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


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
    items = list(candidates)
    for path in items:
        if path.is_file():
            return path

    print("Missing candidates:", file=sys.stderr)
    for path in items:
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
        Path(f"{base}.sGRM.npy"),
    ]
    return require_any_file("GRM file not found.", candidates)


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
    return has_chrom and has_pos and has_pvalue


def find_gwas_tsv_files(outdir: Path, prefix_name: str) -> list[Path]:
    candidates = sorted(p for p in outdir.glob(f"{prefix_name}.test*.tsv") if p.is_file())
    files = [p for p in candidates if _looks_like_postgwas_input(p)]
    if not files:
        print("Scanned TSV candidates:", file=sys.stderr)
        for path in candidates:
            header = "\t".join(_tsv_header(path)[:8])
            print(f"  {path}\t[{header}]", file=sys.stderr)
        fail("No valid GWAS result TSV files found for postgwas.")
    return files


def require_no_matching_files(base_dir: Path, pattern: str, message: str) -> None:
    matches = sorted(path for path in base_dir.glob(pattern) if path.exists())
    if not matches:
        return
    print("Unexpected files:", file=sys.stderr)
    for path in matches:
        print(f"  {path}", file=sys.stderr)
    fail(message)


def require_tsv_columns(path: Path, expected: list[str], message: str) -> None:
    header = _tsv_header(path)
    if header != list(expected):
        fail(f"{message}: expected header={expected}, got={header} ({path})")


def _parse_suite_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    tokens = [tok.strip().lower() for tok in re.split(r"[\s,]+", str(raw)) if tok.strip()]
    invalid = [tok for tok in tokens if tok not in ALL_SUITES]
    if invalid:
        fail(f"Unknown suites: {', '.join(invalid)}. Supported: {', '.join(ALL_SUITES)}")
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _order_suites(suites: Iterable[str]) -> list[str]:
    ordered = [suite for suite in suites if suite != "bench"]
    if any(str(suite) == "bench" for suite in suites):
        ordered.append("bench")
    return ordered


def resolve_requested_suites(mode: str, only: str | None, skip: str | None) -> list[str]:
    only_suites = _parse_suite_list(only)
    if only_suites:
        base = only_suites
    else:
        base = list(DEFAULT_FULL_SUITES if str(mode).strip().lower() == "full" else DEFAULT_SMOKE_SUITES)

    skip_set = set(_parse_suite_list(skip))
    selected = _order_suites([suite for suite in base if suite not in skip_set])
    if not selected:
        fail("No validation suites selected after applying --only/--skip.")
    return selected


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


def _resolve_bench_config(mode: str) -> tuple[int, int, int]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "smoke":
        nsnp_k = env_int("JX_GGVAL_BENCH_SMOKE_NSNP_K", 500, min_value=1)
        n_individuals = env_int("JX_GGVAL_BENCH_SMOKE_NIND", 5000, min_value=40)
    else:
        nsnp_k = env_int("JX_GGVAL_BENCH_FULL_NSNP_K", 500, min_value=1)
        n_individuals = env_int("JX_GGVAL_BENCH_FULL_NIND", 5000, min_value=80)
    seed = env_int("JX_GGVAL_BENCH_SEED", 20260708, min_value=0)
    return int(nsnp_k), int(n_individuals), int(seed)


def _remove_path(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink()
    except FileNotFoundError:
        pass


def cleanup_validation_artifacts(outdir: Path) -> None:
    prefix_names = [
        SIM_PREFIX_NAME,
        SIM_TXT_PREFIX_NAME,
        SIM_HMP_PREFIX_NAME,
        SIM_VCF_PREFIX_NAME,
        SIM_ROUNDTRIP_PREFIX_NAME,
        BENCH_PREFIX_NAME,
        GS_BFILE_PREFIX_NAME,
        GS_VCF_GS_PREFIX_NAME,
        GS_HMP_GS_PREFIX_NAME,
        GS_ML_PREFIX_NAME,
    ]
    for prefix_name in prefix_names:
        for pattern in [f"{prefix_name}*", f"~{prefix_name}*", f"~~{prefix_name}*"]:
            for path in sorted(outdir.glob(pattern)):
                _remove_path(path)

    trait_dir = outdir / TRAIT_NAME
    if trait_dir.is_dir():
        for prefix_name in prefix_names:
            for path in sorted(trait_dir.glob(f"{prefix_name}*")):
                _remove_path(path)
        try:
            next(trait_dir.iterdir())
        except StopIteration:
            trait_dir.rmdir()
        except FileNotFoundError:
            pass


def build_bench_dataset(
    outdir: Path,
    *,
    nsnp_k: int,
    n_individuals: int,
    seed: int,
) -> dict[str, Path]:
    bench_prefix = outdir / BENCH_PREFIX_NAME
    run(
        [
            "jx",
            "sim",
            str(int(nsnp_k)),
            str(int(n_individuals)),
            str(bench_prefix),
            "-seed",
            str(int(seed)),
            "-trait-name",
            TRAIT_NAME,
        ]
    )
    for suffix, label in [
        (".bed", "Benchmark BED output missing"),
        (".bim", "Benchmark BIM output missing"),
        (".fam", "Benchmark FAM output missing"),
        (".pheno.txt", "Benchmark phenotype table missing"),
    ]:
        require_file(Path(f"{bench_prefix}{suffix}"), label)
    return {
        "bench_prefix": bench_prefix,
        "pheno_txt": Path(f"{bench_prefix}.pheno.txt"),
    }


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
            TRAIT_NAME,
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


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def detect_effective_threads() -> int:
    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    return max(1, int(os.cpu_count() or 1))


def _resolve_bench_full_threads() -> int:
    return env_int("JX_GGVAL_BENCH_FULL_THREADS", detect_effective_threads(), min_value=1)


def _benchmark_thread_plan(full_threads: int) -> list[tuple[str, int]]:
    desired = [
        ("1c", 1),
        ("0.25x", max(1, int(round(full_threads * 0.25)))),
        ("0.50x", max(1, int(round(full_threads * 0.50)))),
        ("0.75x", max(1, int(round(full_threads * 0.75)))),
        ("full", max(1, int(full_threads))),
    ]
    labels_by_threads: dict[int, list[str]] = {}
    ordered_threads: list[int] = []
    for label, threads in desired:
        threads = max(1, min(int(full_threads), int(threads)))
        if threads not in labels_by_threads:
            labels_by_threads[threads] = []
            ordered_threads.append(threads)
        labels_by_threads[threads].append(label)
    return [("/".join(labels_by_threads[t]), t) for t in ordered_threads]


def _parse_grm_stage_timing(log_path: Path) -> dict[str, float]:
    text = _strip_ansi(log_path.read_text(encoding="utf-8", errors="replace"))
    match = re.search(
        r"GRM stream timing:\s*decode=([0-9.]+)s.*?gemm=([0-9.]+)s.*?other=([0-9.]+)s.*?total=([0-9.]+)s",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return {}
    return {
        "decode_sec": float(match.group(1)),
        "gemm_sec": float(match.group(2)),
        "other_sec": float(match.group(3)),
        "total_sec": float(match.group(4)),
    }


def _parse_grm_kernel_backend(log_path: Path) -> str:
    text = _strip_ansi(log_path.read_text(encoding="utf-8", errors="replace"))
    patterns = [
        r"GRM auto route selected backend:\s*([^\n(]+)",
        r"GRM\s+\(Effective SNPs:\s*\d+,\s*([^)]+?)\)",
        r"Rust SGEMM backend:\s*([A-Za-z0-9_:+.-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).strip()
    return "unknown"


def _load_json(path: Path) -> dict[str, Any]:
    require_file(path, "JSON file missing")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as ex:
        fail(f"Failed to parse JSON {path}: {ex}")
    if not isinstance(loaded, dict):
        fail(f"Expected JSON object in {path}")
    return loaded


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return int(round(numeric))


def _count_nonempty_lines(path: Path) -> int:
    require_file(path, "Line-count input missing")
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _count_bim_rows(path: Path) -> int:
    return _count_nonempty_lines(path)


def _resolve_summary_artifact_path(summary_path: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    token = str(raw_path).strip()
    if token == "":
        return None
    path = Path(token)
    if path.is_absolute():
        return path
    return (summary_path.parent / path).resolve()


def _summary_artifact_by_method(summary_path: Path, *, trait_name: str = TRAIT_NAME) -> dict[str, dict[str, Any]]:
    summary = _load_json(summary_path)
    out: dict[str, dict[str, Any]] = {}
    for rec in list(summary.get("model_artifacts", []) or []):
        if not isinstance(rec, dict):
            continue
        rec_trait = str(rec.get("trait", "")).strip()
        if rec_trait not in {"", str(trait_name)}:
            continue
        method = str(rec.get("method", "")).strip()
        if method == "":
            continue
        out[method] = rec
    return out


def _validate_text_effect_table(
    path: Path,
    *,
    expected_header: list[str],
    expected_rows: int | None,
    method_name: str,
) -> dict[str, int]:
    require_tsv_columns(path, expected_header, f"{method_name} effect header mismatch")
    row_count = 0
    beta_non_nan_rows = 0
    pip_non_nan_rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if list(reader.fieldnames or []) != list(expected_header):
            fail(f"{method_name} effect header mismatch after DictReader parse: {path}")
        for lineno, row in enumerate(reader, start=2):
            if row is None:
                continue
            values = list(row.values())
            if not any(str(v or "").strip() for v in values):
                continue
            row_count += 1

            chr_raw = str(row.get("chr", "") or "").strip()
            pos_raw = str(row.get("pos", "") or "").strip()
            snp_raw = str(row.get("snp", "") or "").strip()
            if chr_raw == "":
                fail(f"{method_name} effect chr is empty at line {lineno}: {path}")
            if snp_raw == "":
                fail(f"{method_name} effect snp is empty at line {lineno}: {path}")
            try:
                pos_val = float(pos_raw)
            except ValueError:
                fail(f"{method_name} effect pos is not numeric at line {lineno}: {path}")
            if not math.isfinite(pos_val):
                fail(f"{method_name} effect pos is not finite at line {lineno}: {path}")

            beta_raw = str(row.get("beta", "") or "").strip()
            try:
                beta_val = float(beta_raw)
            except ValueError:
                fail(f"{method_name} effect beta is not numeric at line {lineno}: {path}")
            if not math.isfinite(beta_val):
                fail(f"{method_name} effect beta is not finite at line {lineno}: {path}")
            beta_non_nan_rows += 1

            if "pip" in expected_header:
                pip_raw = str(row.get("pip", "") or "").strip()
                try:
                    pip_val = float(pip_raw)
                except ValueError:
                    fail(f"{method_name} effect pip is not numeric at line {lineno}: {path}")
                if not math.isfinite(pip_val) or pip_val < -1e-12 or pip_val > 1.0 + 1e-12:
                    fail(f"{method_name} effect pip is outside [0, 1] at line {lineno}: {path}")
                pip_non_nan_rows += 1

    if row_count <= 0:
        fail(f"{method_name} effect file has no data rows: {path}")
    if beta_non_nan_rows != row_count:
        fail(f"{method_name} beta rows mismatch: expected {row_count}, got {beta_non_nan_rows}")
    if "pip" in expected_header and pip_non_nan_rows != row_count:
        fail(f"{method_name} pip rows mismatch: expected {row_count}, got {pip_non_nan_rows}")
    if expected_rows is not None and row_count != int(expected_rows):
        fail(f"{method_name} effect row count mismatch: expected {expected_rows}, got {row_count} ({path})")
    return {
        "rows": row_count,
        "beta_non_nan_rows": beta_non_nan_rows,
        "pip_non_nan_rows": pip_non_nan_rows,
    }


def validate_gs_output_basics(outdir: Path, *, prefix_name: str, trait_name: str) -> Path:
    gebv_path = outdir / f"{prefix_name}.{trait_name}.gebv.tsv"
    require_file(gebv_path, "GS GEBV output missing")
    if (outdir / f"{prefix_name}.{trait_name}.gs.tsv").exists():
        fail("Legacy GS prediction output (.gs.tsv) should not be generated.")
    require_no_matching_files(
        outdir,
        f"{prefix_name}.{trait_name}.gs.effect*",
        "Legacy GS effect side output should not be generated.",
    )
    return find_gs_summary(outdir, prefix_name)


def validate_text_effect_artifact(
    summary_path: Path,
    *,
    method_name: str,
    expected_rows: int | None = None,
) -> None:
    artifacts = _summary_artifact_by_method(summary_path)
    if method_name not in artifacts:
        fail(f"{method_name} artifact missing from summary: {summary_path}")
    rec = artifacts[method_name]
    artifact_format = str(rec.get("artifact_format", "")).strip()
    if artifact_format != "text_effect":
        fail(f"{method_name} artifact format mismatch: expected text_effect, got {artifact_format!r}")

    model_path = _resolve_summary_artifact_path(summary_path, rec.get("model_file"))
    effect_path = _resolve_summary_artifact_path(summary_path, rec.get("effect_file")) or model_path
    if model_path is None or effect_path is None:
        fail(f"{method_name} artifact paths are missing in {summary_path}")
    require_file(model_path, f"{method_name} model file missing")
    require_file(effect_path, f"{method_name} effect file missing")

    expected_header = TEXT_EFFECT_HEADERS[method_name]
    stats = _validate_text_effect_table(
        effect_path,
        expected_header=expected_header,
        expected_rows=expected_rows,
        method_name=method_name,
    )
    effect_col = str(rec.get("effect_column", "") or "").strip().lower()
    if effect_col not in {"", "beta"}:
        fail(f"{method_name} effect_column should be beta, got {effect_col!r}")

    effect_meta = dict(rec.get("effect_meta", {}) or {})
    meta_rows = _coerce_optional_int(effect_meta.get("rows"))
    if meta_rows is not None and meta_rows != stats["rows"]:
        fail(f"{method_name} summary effect_meta.rows mismatch: {meta_rows} vs {stats['rows']}")

    meta_beta_rows = _coerce_optional_int(effect_meta.get("beta_non_nan_rows"))
    if meta_beta_rows is not None and meta_beta_rows != stats["beta_non_nan_rows"]:
        fail(
            f"{method_name} summary beta_non_nan_rows mismatch: "
            f"{meta_beta_rows} vs {stats['beta_non_nan_rows']}"
        )

    meta_pip_rows = _coerce_optional_int(effect_meta.get("pip_non_nan_rows"))
    if "pip" in expected_header:
        if meta_pip_rows is not None and meta_pip_rows != stats["pip_non_nan_rows"]:
            fail(
                f"{method_name} summary pip_non_nan_rows mismatch: "
                f"{meta_pip_rows} vs {stats['pip_non_nan_rows']}"
            )
    elif meta_pip_rows not in {None, 0}:
        fail(f"{method_name} should not report pip rows, got {meta_pip_rows}")


def validate_binary_jxmodel_artifact(summary_path: Path, *, method_name: str) -> None:
    artifacts = _summary_artifact_by_method(summary_path)
    if method_name not in artifacts:
        fail(f"{method_name} artifact missing from summary: {summary_path}")
    rec = artifacts[method_name]
    artifact_format = str(rec.get("artifact_format", "")).strip()
    if artifact_format != "binary_jxmodel":
        fail(f"{method_name} artifact format mismatch: expected binary_jxmodel, got {artifact_format!r}")

    model_path = _resolve_summary_artifact_path(summary_path, rec.get("model_file"))
    if model_path is None:
        fail(f"{method_name} model_file missing from summary: {summary_path}")
    require_file(model_path, f"{method_name} binary .jxmodel missing")
    if model_path.suffix.lower() != ".jxmodel":
        fail(f"{method_name} model_file is not .jxmodel: {model_path}")

    effect_meta = dict(rec.get("effect_meta", {}) or {})
    meta_rows = _coerce_optional_int(effect_meta.get("rows"))
    meta_beta_rows = _coerce_optional_int(effect_meta.get("beta_non_nan_rows"))
    if meta_rows is not None and meta_rows <= 0:
        fail(f"{method_name} effect_meta.rows should be positive, got {meta_rows}")
    if meta_beta_rows is not None and meta_beta_rows <= 0:
        fail(f"{method_name} effect_meta.beta_non_nan_rows should be positive, got {meta_beta_rows}")


def validate_postgs_outputs(
    outdir: Path,
    *,
    prefix_name: str,
    trait_name: str,
    expect_effect_plot: bool,
) -> None:
    require_file(
        outdir / trait_name / f"{prefix_name}.accuracy_runtime.png",
        "postGS accuracy-runtime plot missing",
    )
    require_file(
        outdir / trait_name / f"{prefix_name}.accuracy_violin.png",
        "postGS accuracy-violin plot missing",
    )
    if expect_effect_plot:
        require_file(
            outdir / f"{prefix_name}.{trait_name}.effects.png",
            "postGS effect plot missing",
        )


def validate_gs_file_input_debug_outputs(
    outdir: Path,
    *,
    prefix_name: str,
    trait_name: str,
    expected_models: Iterable[str] | None = None,
) -> Path:
    summary_path = validate_gs_output_basics(outdir, prefix_name=prefix_name, trait_name=trait_name)
    if expected_models is None:
        expected_model_names = ["BLUP", "BayesA", "BayesB", "BayesCpi"]
    else:
        expected_model_names = [str(x).strip() for x in expected_models if str(x).strip()]

    for method_name in expected_model_names:
        if method_name not in TEXT_EFFECT_HEADERS:
            fail(f"Unsupported GS model expectation: {method_name!r}")
        validate_text_effect_artifact(summary_path, method_name=method_name, expected_rows=None)

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


def validate_reml_outputs(outdir: Path, *, prefix_name: str = SIM_PREFIX_NAME) -> None:
    base = outdir / f"{prefix_name}.pheno"
    require_file(Path(f"{base}.reml.summary.tsv"), "REML summary output missing")
    require_file(Path(f"{base}.blue.txt"), "REML BLUE output missing")
    require_file(Path(f"{base}.gblup.txt"), "REML GBLUP output missing")


def check_jx_available() -> None:
    if shutil.which("jx") is None:
        fail("jx command not found in PATH")
    run(["jx", "--version"])


def _parse_pca_eigh_backend(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = [
        r"EIGH backend:\s*([A-Za-z0-9_:+.-]+)",
        r"Eigen decomposition finished \(backend=([^),\s]+)",
        r"Rust eigh did not use LAPACK backend \(backend=([^)\s]+)\)",
        r"backend=([A-Za-z0-9_:+.-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).strip()
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
        run(cmd, stdout_path=log_path, stderr_to_stdout=True)
    except subprocess.CalledProcessError:
        print_log(log_path)
        raise

    backend = _parse_pca_eigh_backend(log_path)
    print(f"{label} backend         : {backend}")
    if backend == "unknown":
        print(f"{label} backend log     : {log_path}")
    return backend


def _print_benchmark_table(title: str, rows: list[dict[str, Any]], *, time_key: str, time_label: str) -> None:
    print(title)
    print("-" * 78)
    print(f"{'Target':<14}{'Threads':>8}{time_label:>14}{'Speedup':>12}{'Wall(s)':>12}  Backend")
    print("-" * 78)
    for row in rows:
        speed = row.get("speedup_vs_1c", float("nan"))
        speed_text = "NA" if not math.isfinite(float(speed)) else f"{float(speed):.2f}x"
        metric = float(row.get(time_key, float("nan")))
        wall = float(row.get("wall_sec", float("nan")))
        backend = str(row.get("backend", "unknown"))
        print(
            f"{str(row.get('target_label', '')):<14}"
            f"{int(row.get('threads', 0)):>8}"
            f"{metric:>14.4f}"
            f"{speed_text:>12}"
            f"{wall:>12.4f}  "
            f"{backend}"
        )
    print("-" * 78)


def run_multicore_benchmark_suite(outdir: Path, logdir: Path, *, mode: str) -> None:
    nsnp_k, n_individuals, seed = _resolve_bench_config(mode)
    full_threads = _resolve_bench_full_threads()
    thread_plan = _benchmark_thread_plan(full_threads)

    step("BENCH. Build simulated benchmark dataset")
    bench_paths = build_bench_dataset(
        outdir,
        nsnp_k=nsnp_k,
        n_individuals=n_individuals,
        seed=seed,
    )
    bench_prefix = bench_paths["bench_prefix"]
    print(
        f"Benchmark dataset      : {bench_prefix} "
        f"(n={n_individuals}, m≈{nsnp_k * 1000}, full_threads={full_threads})"
    )
    sep()

    step("BENCH. Measure GRM/GEMM and PCA(-grm) multicore efficiency")
    grm_rows: list[dict[str, Any]] = []
    baseline_metric: float | None = None
    baseline_grm: Path | None = None
    for target_label, threads in thread_plan:
        prefix_name = f"{BENCH_PREFIX_NAME}.t{threads}"
        prefix_path = outdir / prefix_name
        log_path = logdir / f"{prefix_name}.grm.log"
        t0 = time.perf_counter()
        try:
            run(
                [
                    "jx",
                    "grm",
                    "-bfile",
                    str(bench_prefix),
                    "-maf",
                    "0",
                    "-geno",
                    "1",
                    "--stage-timing",
                    "-v",
                    "-t",
                    str(int(threads)),
                    "-o",
                    str(outdir),
                    "-prefix",
                    prefix_name,
                ],
                stdout_path=log_path,
                stderr_to_stdout=True,
            )
        except subprocess.CalledProcessError:
            print_log(log_path)
            raise
        wall_sec = time.perf_counter() - t0
        timing = _parse_grm_stage_timing(log_path)
        backend = _parse_grm_kernel_backend(log_path)
        gemm_sec = float(timing.get("gemm_sec", float("nan")))
        metric_sec = gemm_sec if math.isfinite(gemm_sec) and gemm_sec > 0 else float(
            timing.get("total_sec", wall_sec)
        )
        if baseline_metric is None:
            baseline_metric = metric_sec
            baseline_grm = find_grm(prefix_path)
        speedup = (
            float(baseline_metric) / metric_sec
            if math.isfinite(metric_sec) and metric_sec > 0 and baseline_metric is not None
            else float("nan")
        )
        grm_rows.append(
            {
                "task": "grm_gemm",
                "target_label": target_label,
                "threads": int(threads),
                "metric_sec": float(metric_sec),
                "stage_total_sec": float(timing.get("total_sec", wall_sec)),
                "wall_sec": float(wall_sec),
                "speedup_vs_1c": float(speedup),
                "backend": backend,
                "log_file": str(log_path),
                "matrix_file": str(find_grm(prefix_path)),
            }
        )

    _print_benchmark_table("GRM/GEMM speedup vs 1c", grm_rows, time_key="metric_sec", time_label="GEMM(s)")

    if baseline_grm is None:
        fail("Benchmark GRM baseline file is missing.")

    pca_rows: list[dict[str, Any]] = []
    baseline_pca: float | None = None
    for target_label, threads in thread_plan:
        pca_prefix_name = f"{BENCH_PREFIX_NAME}.pca.t{threads}"
        log_path = logdir / f"{pca_prefix_name}.log"
        t0 = time.perf_counter()
        try:
            run(
                [
                    "jx",
                    "pca",
                    "-k",
                    str(baseline_grm),
                    "-v",
                    "-t",
                    str(int(threads)),
                    "-o",
                    str(outdir),
                    "-prefix",
                    pca_prefix_name,
                ],
                stdout_path=log_path,
                stderr_to_stdout=True,
            )
        except subprocess.CalledProcessError:
            print_log(log_path)
            raise
        wall_sec = time.perf_counter() - t0
        backend = _parse_pca_eigh_backend(log_path)
        metric_sec = float(wall_sec)
        if baseline_pca is None:
            baseline_pca = metric_sec
        speedup = (
            float(baseline_pca) / metric_sec
            if math.isfinite(metric_sec) and metric_sec > 0 and baseline_pca is not None
            else float("nan")
        )
        require_file(outdir / f"{pca_prefix_name}.eigenvec", "PCA eigenvec output missing")
        require_file(outdir / f"{pca_prefix_name}.eigenval", "PCA eigenval output missing")
        pca_rows.append(
            {
                "task": "pca_grm",
                "target_label": target_label,
                "threads": int(threads),
                "metric_sec": float(metric_sec),
                "stage_total_sec": float(metric_sec),
                "wall_sec": float(wall_sec),
                "speedup_vs_1c": float(speedup),
                "backend": backend,
                "log_file": str(log_path),
                "matrix_file": str(baseline_grm),
            }
        )

    _print_benchmark_table("PCA(-grm) speedup vs 1c", pca_rows, time_key="metric_sec", time_label="PCA(s)")

    bench_tsv = outdir / f"{BENCH_PREFIX_NAME}.multicore.tsv"
    with bench_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "task",
                "target_label",
                "threads",
                "metric_sec",
                "stage_total_sec",
                "wall_sec",
                "speedup_vs_1c",
                "backend",
                "log_file",
                "matrix_file",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in grm_rows + pca_rows:
            writer.writerow(row)
    require_file(bench_tsv, "Benchmark summary TSV missing")
    sep()


def run_gwas_suite(paths: dict[str, Path], outdir: Path, logdir: Path, *, threads: int) -> None:
    sim_prefix = paths["sim_prefix"]
    pheno_txt = paths["pheno_txt"]

    step("GWAS. Validate GWAS and postGWAS on simulated data")
    run(["jx", "grm", "-bfile", str(sim_prefix), "-o", str(outdir), "-m", "1"])
    run(["jx", "grm", "-bfile", str(sim_prefix), "-o", str(outdir), "-m", "2"])
    run_pca_with_backend_report(
        ["jx", "pca", "-bfile", str(sim_prefix), "-o", str(outdir)],
        log_path=logdir / "jx_pca_gwas.log",
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
            "-splmm-exact",
            "-k",
            str(grm_k),
            "-c",
            str(sim_prefix.with_suffix(".eigenvec")),
            "-n",
            TRAIT_NAME,
            "-t",
            str(threads),
            "-o",
            str(outdir),
        ]
    )
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
            "-splmm-exact",
            "-k",
            str(grm_k),
            "-c",
            str(sim_prefix.with_suffix(".eigenvec")),
            "-n",
            TRAIT_NAME,
            "-t",
            str(threads),
            "-o",
            str(outdir),
            "-force-model",
        ]
    )
    gwas_files = find_gwas_tsv_files(outdir, SIM_PREFIX_NAME)
    run(
        [
            "jx",
            "postgwas",
            "-gwasfile",
            *[str(path) for path in gwas_files],
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
    require_any_file("postGWAS Manhattan output missing", outdir.glob(f"{SIM_PREFIX_NAME}.test*.manh.pdf"))
    require_any_file("postGWAS QQ output missing", outdir.glob(f"{SIM_PREFIX_NAME}.test*.qq.pdf"))
    sep()


def run_gs_file_suite(
    paths: dict[str, Path],
    outdir: Path,
    *,
    threads: int,
    cv_folds: int,
    postgs_enabled: bool,
) -> None:
    txt_matrix = paths["txt_matrix"]
    pheno_txt = paths["pheno_txt"]

    step("GS-FILE. Validate GS text-input route and text .jxmodel outputs")
    run(
        [
            "jx",
            "gs",
            "-file",
            str(txt_matrix),
            "-p",
            str(pheno_txt),
            "-n",
            TRAIT_NAME,
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
        trait_name=TRAIT_NAME,
    )
    if postgs_enabled:
        run(["jx", "postgs", "-json", str(gs_summary), "-o", str(outdir)])
        validate_postgs_outputs(
            outdir,
            prefix_name=SIM_TXT_PREFIX_NAME,
            trait_name=TRAIT_NAME,
            expect_effect_plot=True,
        )
    sep()


def run_gs_bfile_suite(
    paths: dict[str, Path],
    outdir: Path,
    *,
    threads: int,
    cv_folds: int,
) -> None:
    sim_prefix = paths["sim_prefix"]
    pheno_txt = paths["pheno_txt"]
    expected_rows = _count_bim_rows(sim_prefix.with_suffix(".bim"))

    step("GS-BFILE. Validate GS PLINK-input route and numeric effect outputs")
    run(
        [
            "jx",
            "gs",
            "-bfile",
            str(sim_prefix),
            "-p",
            str(pheno_txt),
            "-n",
            TRAIT_NAME,
            "-BLUP",
            "-BayesA",
            "-BayesB",
            "-BayesCpi",
            "-maf",
            "0",
            "-geno",
            "1",
            "-cv",
            str(cv_folds),
            "-save-model",
            "-t",
            str(threads),
            "-o",
            str(outdir),
            "-prefix",
            GS_BFILE_PREFIX_NAME,
        ]
    )
    summary_path = validate_gs_output_basics(
        outdir,
        prefix_name=GS_BFILE_PREFIX_NAME,
        trait_name=TRAIT_NAME,
    )
    for method_name in ["BLUP", "BayesA", "BayesB", "BayesCpi"]:
        validate_text_effect_artifact(summary_path, method_name=method_name, expected_rows=expected_rows)
    sep()


def run_gs_vcf_suite(
    paths: dict[str, Path],
    outdir: Path,
    *,
    threads: int,
    cv_folds: int,
) -> None:
    vcf_path = Path(f"{paths['vcf_prefix']}.vcf.gz")
    pheno_txt = paths["pheno_txt"]
    sim_prefix = paths["sim_prefix"]
    expected_rows = _count_bim_rows(sim_prefix.with_suffix(".bim"))

    step("GS-VCF. Validate GS VCF-input regression")
    require_file(vcf_path, "VCF regression input missing")
    run(
        [
            "jx",
            "gs",
            "-vcf",
            str(vcf_path),
            "-p",
            str(pheno_txt),
            "-n",
            TRAIT_NAME,
            "-BLUP",
            "-maf",
            "0",
            "-geno",
            "1",
            "-cv",
            str(cv_folds),
            "-save-model",
            "-t",
            str(threads),
            "-o",
            str(outdir),
            "-prefix",
            GS_VCF_GS_PREFIX_NAME,
        ]
    )
    summary_path = validate_gs_output_basics(
        outdir,
        prefix_name=GS_VCF_GS_PREFIX_NAME,
        trait_name=TRAIT_NAME,
    )
    validate_text_effect_artifact(summary_path, method_name="BLUP", expected_rows=expected_rows)
    sep()


def run_gs_hmp_suite(
    paths: dict[str, Path],
    outdir: Path,
    *,
    threads: int,
    cv_folds: int,
) -> None:
    hmp_path = paths["hmp_prefix"].with_suffix(".hmp")
    pheno_txt = paths["pheno_txt"]
    sim_prefix = paths["sim_prefix"]
    expected_rows = _count_bim_rows(sim_prefix.with_suffix(".bim"))

    step("GS-HMP. Validate GS HapMap-input regression")
    require_file(hmp_path, "HMP regression input missing")
    run(
        [
            "jx",
            "gs",
            "-hmp",
            str(hmp_path),
            "-p",
            str(pheno_txt),
            "-n",
            TRAIT_NAME,
            "-BLUP",
            "-maf",
            "0",
            "-geno",
            "1",
            "-cv",
            str(cv_folds),
            "-save-model",
            "-t",
            str(threads),
            "-o",
            str(outdir),
            "-prefix",
            GS_HMP_GS_PREFIX_NAME,
        ]
    )
    summary_path = validate_gs_output_basics(
        outdir,
        prefix_name=GS_HMP_GS_PREFIX_NAME,
        trait_name=TRAIT_NAME,
    )
    validate_text_effect_artifact(summary_path, method_name="BLUP", expected_rows=expected_rows)
    sep()


def run_gs_ml_suite(
    paths: dict[str, Path],
    outdir: Path,
    *,
    threads: int,
    cv_folds: int,
    postgs_enabled: bool,
) -> str:
    if importlib.util.find_spec("sklearn") is None:
        print("GS-ML skipped        : scikit-learn is unavailable in current Python environment")
        sep()
        return "skipped (scikit-learn unavailable)"

    sim_prefix = paths["sim_prefix"]
    pheno_txt = paths["pheno_txt"]

    step("GS-ML. Validate scikit-learn GS save-model/postgs compatibility")
    run(
        [
            "jx",
            "gs",
            "-bfile",
            str(sim_prefix),
            "-p",
            str(pheno_txt),
            "-n",
            TRAIT_NAME,
            "-RF",
            "-ENET",
            "-maf",
            "0",
            "-geno",
            "1",
            "-cv",
            str(cv_folds),
            "-save-model",
            "-t",
            str(threads),
            "-o",
            str(outdir),
            "-prefix",
            GS_ML_PREFIX_NAME,
        ]
    )
    summary_path = validate_gs_output_basics(
        outdir,
        prefix_name=GS_ML_PREFIX_NAME,
        trait_name=TRAIT_NAME,
    )
    for method_name in ["RF", "ENET"]:
        validate_binary_jxmodel_artifact(summary_path, method_name=method_name)
    if postgs_enabled:
        run(["jx", "postgs", "-json", str(summary_path), "-o", str(outdir)])
        validate_postgs_outputs(
            outdir,
            prefix_name=GS_ML_PREFIX_NAME,
            trait_name=TRAIT_NAME,
            expect_effect_plot=True,
        )
    sep()
    return "ok"


def run_reml_suite(paths: dict[str, Path], outdir: Path) -> None:
    sim_prefix = paths["sim_prefix"]
    pheno_txt = paths["pheno_txt"]
    grm_k = find_grm(sim_prefix)

    step("REML. Validate REML outputs on simulated phenotype table")
    run(
        [
            "jx",
            "reml",
            "-file",
            str(pheno_txt),
            "-l",
            "IID",
            "-p",
            TRAIT_NAME,
            "-k",
            str(grm_k),
            "-o",
            str(outdir),
        ]
    )
    validate_reml_outputs(outdir, prefix_name=SIM_PREFIX_NAME)
    sep()


def run_validation_flow(
    outdir: Path,
    logdir: Path,
    *,
    mode: str,
    suites: list[str],
    threads: int,
    cv_folds: int,
    postgs_enabled: bool,
) -> dict[str, str]:
    cleanup_validation_artifacts(outdir)
    suite_status: dict[str, str] = {}
    main_paths: dict[str, Path] | None = None
    sklearn_available = importlib.util.find_spec("sklearn") is not None

    def ensure_main_paths() -> dict[str, Path]:
        nonlocal main_paths
        if main_paths is None:
            nsnp_k, n_individuals, seed = _resolve_sim_config(mode)
            export_debug_formats = any(suite in {"gs-vcf", "gs-hmp"} for suite in suites)
            step("DATA. Build simulated validation dataset")
            main_paths = build_validation_dataset(
                outdir,
                nsnp_k=nsnp_k,
                n_individuals=n_individuals,
                seed=seed,
                export_debug_formats=export_debug_formats,
            )
            sep()
        return main_paths

    for suite in suites:
        if suite == "bench":
            run_multicore_benchmark_suite(outdir, logdir, mode=mode)
            suite_status[suite] = "ok"
        elif suite == "gwas":
            run_gwas_suite(ensure_main_paths(), outdir, logdir, threads=threads)
            suite_status[suite] = "ok"
        elif suite == "gs-file":
            run_gs_file_suite(
                ensure_main_paths(),
                outdir,
                threads=threads,
                cv_folds=cv_folds,
                postgs_enabled=postgs_enabled,
            )
            suite_status[suite] = "ok"
        elif suite == "gs-bfile":
            run_gs_bfile_suite(ensure_main_paths(), outdir, threads=threads, cv_folds=cv_folds)
            suite_status[suite] = "ok"
        elif suite == "gs-vcf":
            run_gs_vcf_suite(ensure_main_paths(), outdir, threads=threads, cv_folds=cv_folds)
            suite_status[suite] = "ok"
        elif suite == "gs-hmp":
            run_gs_hmp_suite(ensure_main_paths(), outdir, threads=threads, cv_folds=cv_folds)
            suite_status[suite] = "ok"
        elif suite == "gs-ml":
            if not sklearn_available:
                print("GS-ML skipped        : scikit-learn is unavailable in current Python environment")
                sep()
                suite_status[suite] = "skipped (scikit-learn unavailable)"
            else:
                suite_status[suite] = run_gs_ml_suite(
                    ensure_main_paths(),
                    outdir,
                    threads=threads,
                    cv_folds=cv_folds,
                    postgs_enabled=postgs_enabled,
                )
        elif suite == "reml":
            run_reml_suite(ensure_main_paths(), outdir)
            suite_status[suite] = "ok"
        else:
            fail(f"Unsupported suite: {suite}")

    return suite_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JanusX validation runner based on simulated data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default=env_str("JX_GGVAL_MODE", "full"),
        help="Validation mode. Controls default suite set and simulated dataset size.",
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
        help="Thread count for validation commands except multicore benchmark.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=env_int("JX_GGVAL_CV", 2, min_value=2),
        help="CV folds for GS validation suites.",
    )
    parser.add_argument(
        "--no-postgs",
        action="store_true",
        default=not env_truthy("JX_GGVAL_POSTGS", True),
        help="Skip postGS validation steps.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=f"Run only the named suites (comma or space separated). Supported: {', '.join(ALL_SUITES)}.",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help=f"Skip the named suites (comma or space separated). Supported: {', '.join(ALL_SUITES)}.",
    )
    parser.add_argument(
        "--multicore",
        action="store_true",
        help="Run only the multicore GRM/EIGH benchmark suite with a larger benchmark dataset.",
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
    if args.multicore:
        if args.only is not None or args.skip is not None:
            fail("--multicore cannot be combined with --only/--skip; it runs bench as a standalone suite.")
        suites = ["bench"]
    else:
        suites = resolve_requested_suites(args.mode, args.only, args.skip)
    suite_status = run_validation_flow(
        outdir,
        logdir,
        mode=args.mode,
        suites=suites,
        threads=args.threads,
        cv_folds=args.cv,
        postgs_enabled=not args.no_postgs,
    )

    step("Validation completed")
    print(f"Mode      : {args.mode}")
    print(f"Suites    : {', '.join(suites)}")
    print(f"Output dir: {outdir}")
    print(f"Log dir   : {logdir}")
    for suite in suites:
        print(f"Status    : {suite} -> {suite_status.get(suite, 'ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
