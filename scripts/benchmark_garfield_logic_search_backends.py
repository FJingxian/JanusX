#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _split_fields(line: str) -> list[str]:
    return line.strip().replace(",", "\t").split()


def _read_fam_iids(prefix: str) -> list[str]:
    fam = Path(f"{prefix}.fam").expanduser()
    if not fam.is_file():
        raise FileNotFoundError(f"FAM not found: {fam}")
    out: list[str] = []
    with fam.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            toks = _split_fields(line)
            if len(toks) < 2:
                raise ValueError(f"Malformed FAM at {fam}:{line_no}")
            out.append(str(toks[1]))
    if not out:
        raise ValueError(f"No samples found in {fam}")
    return out


def _read_pheno(pheno: str, trait: str | None) -> tuple[str, dict[str, float]]:
    path = Path(pheno).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Phenotype file not found: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        header = _split_fields(fh.readline())
        if len(header) < 2:
            raise ValueError(f"Phenotype header must contain IID and at least one trait: {path}")
        trait_names = header[1:]
        trait_name = trait or trait_names[0]
        if trait_name in trait_names:
            trait_idx = trait_names.index(trait_name) + 1
        else:
            try:
                trait_idx = int(trait_name)
            except ValueError as exc:
                raise ValueError(f"Trait '{trait_name}' not found in {path}") from exc
            if trait_idx < 0 or trait_idx >= len(header):
                raise ValueError(f"Trait index out of range: {trait_idx}")
            trait_name = header[trait_idx]

        values: dict[str, float] = {}
        for line_no, line in enumerate(fh, start=2):
            if not line.strip():
                continue
            toks = _split_fields(line)
            if len(toks) <= trait_idx:
                raise ValueError(f"Malformed phenotype row at {path}:{line_no}")
            raw = toks[trait_idx]
            if raw.upper() in {"NA", "NAN", "."}:
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if np.isfinite(val):
                values[str(toks[0])] = val
    if not values:
        raise ValueError(f"No finite phenotype values loaded for trait '{trait_name}'")
    return trait_name, values


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value
    return value * 1024


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "n_rules",
        "n_train",
        "n_test",
        "n_samples",
        "units_total",
        "units_scanned",
        "timing_total_wall_s",
        "timing_scan_wall_s",
        "timing_scan_beam_wall_s",
        "timing_scan_literal_score_wall_s",
        "timing_scan_beam_calls",
        "timing_literal_score_share_of_total_pct",
        "timing_literal_score_share_of_scan_pct",
        "timing_literal_score_share_of_beam_pct",
        "timing_beam_share_of_total_pct",
        "timing_beam_share_of_scan_pct",
        "split_applied",
        "train_pve",
        "test_pve",
        "snp_names",
        "expressions",
        "unit_names",
        "unit_kinds",
        "scores",
        "positions",
        "selected_row_indices",
        "ml_ranks",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        val = result.get(key)
        if isinstance(val, np.ndarray):
            val = val.tolist()
        out[key] = val
    return out


def _rule_hash(payload: dict[str, Any]) -> str:
    rows = []
    n = int(payload.get("n_rules") or 0)
    for i in range(n):
        rows.append(
            {
                "snp": (payload.get("snp_names") or [None] * n)[i],
                "expr": (payload.get("expressions") or [None] * n)[i],
                "unit": (payload.get("unit_names") or [None] * n)[i],
                "kind": (payload.get("unit_kinds") or [None] * n)[i],
                "pos": (payload.get("positions") or [None] * n)[i],
                "selected": (payload.get("selected_row_indices") or [None] * n)[i],
                "ml_rank": (payload.get("ml_ranks") or [None] * n)[i],
            }
        )
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _probe_singleton_backends(jx: Any) -> dict[str, Any]:
    try:
        y = np.linspace(-1.0, 1.0, 64, dtype=np.float64)
        bits = np.asarray(
            [
                0xAAAAAAAAAAAAAAAA,
                0xCCCCCCCCCCCCCCCC,
                0xF0F0F0F0F0F0F0F0,
                0xFF00FF00FF00FF00,
            ],
            dtype=np.uint64,
        ).reshape(4, 1)
        report = jx.garfield_compare_score_cont_centered_gain_singleton_backends(
            y,
            bits,
            64,
            repeats=1,
            warmup=0,
        )
        return _json_safe(dict(report))
    except Exception as exc:
        return {"probe_error": str(exc)}


def _probe_metal_runtime_status(jx: Any) -> dict[str, Any]:
    try:
        return _json_safe(dict(jx.garfield_metal_runtime_status()))
    except Exception as exc:
        return {"probe_error": str(exc)}


def _score_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    sa = np.asarray(a.get("scores") or [], dtype=np.float64)
    sb = np.asarray(b.get("scores") or [], dtype=np.float64)
    if sa.shape != sb.shape:
        return {"same_score_shape": False, "shape_a": list(sa.shape), "shape_b": list(sb.shape)}
    if sa.size == 0:
        return {"same_score_shape": True, "max_abs_score_diff": 0.0, "mean_abs_score_diff": 0.0}
    diff = np.abs(sa - sb)
    return {
        "same_score_shape": True,
        "max_abs_score_diff": float(np.max(diff)),
        "mean_abs_score_diff": float(np.mean(diff)),
    }


def _worker(args: argparse.Namespace) -> int:
    os.environ["JX_GARFIELD_SCORE_BACKEND"] = str(args.backend)

    import janusx.janusx as jx

    prefix = str(Path(args.bfile).expanduser())
    fam_ids = _read_fam_iids(prefix)
    trait_name, pheno = _read_pheno(args.pheno, args.trait)
    sample_ids = [sid for sid in fam_ids if sid in pheno]
    if args.sample_limit and args.sample_limit > 0:
        sample_ids = sample_ids[: int(args.sample_limit)]
    if len(sample_ids) < 8:
        raise ValueError(f"Too few overlapping samples: {len(sample_ids)}")
    y = np.asarray([pheno[sid] for sid in sample_ids], dtype=np.float64)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = str(out_dir / "garfield_logic")
    scan_bimranges = [x for x in str(args.scan_bimranges).split(",") if x.strip()] or None
    backend_probe = _probe_singleton_backends(jx)
    metal_runtime_status = _probe_metal_runtime_status(jx)

    if str(args.backend).lower() == "gpu":
        if metal_runtime_status.get("probe_error"):
            raise RuntimeError(
                "Failed to query garfield Metal runtime status: "
                f"{metal_runtime_status['probe_error']}"
            )
        if not bool(metal_runtime_status.get("score_gpu_ready")):
            detail = metal_runtime_status.get("error") or "unknown Metal runtime error"
            raise RuntimeError(
                "Explicit Garfield GPU backend requested, but score_gpu is not ready: "
                f"{detail}"
            )

    t0 = time.perf_counter()
    result = jx.garfield_logic_search_bed(
        prefix,
        y,
        grm=None,
        x_cov=None,
        sample_ids=list(sample_ids),
        unit_kind=str(args.unit_kind),
        groups=None,
        group_names=None,
        extension=int(args.extension),
        step=None if args.step is None else int(args.step),
        scan_bimranges=scan_bimranges,
        bin_mode=str(args.bin_mode),
        ml_method=str(args.ml_method),
        ml_importance="imp",
        ml_top_k=int(args.topk),
        ml_top_frac=0.0,
        permutation_repeats=int(args.permutation_repeats),
        permutation_scoring="auto",
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        min_samples_leaf=1,
        min_samples_split=2,
        bootstrap=True,
        feature_subsample=0.0,
        fold=0,
        seed=int(args.seed),
        max_pick=int(args.layer),
        exhaustive_depth=int(args.exhaustive_depth),
        beam_width=int(args.beam_width),
        rank_score=str(args.rank_score),
        maf_threshold=float(args.maf),
        max_missing_rate=float(args.geno),
        snps_only=False,
        block_cols=int(args.block_cols),
        threads=int(args.threads),
        low=-5.0,
        high=5.0,
        max_iter=50,
        tol=1e-3,
        add_intercept=True,
        exact_n_max=15000,
        require_lapack=False,
        out_prefix=out_prefix,
        simbench_path=None,
        top_rules_per_unit=int(args.top_rules_per_unit),
        max_output_rules=int(args.max_output_rules),
        max_output_ratio=0.0,
        rule_permutation=bool(args.rule_permutation),
        prior_len=None,
        no_clean=bool(args.no_clean),
        progress_callback=None,
        progress_every=0,
    )
    wall_s = time.perf_counter() - t0

    normalized = _normalize_result(dict(result))
    payload = {
        "backend": str(args.backend),
        "trait": trait_name,
        "wall_s": float(wall_s),
        "peak_rss_bytes": int(_peak_rss_bytes()),
        "peak_rss_gb": float(_peak_rss_bytes() / (1024 ** 3)),
        "backend_probe": backend_probe,
        "metal_runtime_status": metal_runtime_status,
        "sample_overlap": int(len(sample_ids)),
        "result": normalized,
        "rule_hash": _rule_hash(normalized),
        "out_prefix": out_prefix,
    }
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _run_child(args: argparse.Namespace, backend: str, root: Path) -> dict[str, Any]:
    out_dir = root / f"backend_{backend}"
    out_json = out_dir / "result.json"
    log_path = out_dir / "worker.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--out-dir",
        str(out_dir),
        "--out-json",
        str(out_json),
        "--bfile",
        str(args.bfile),
        "--pheno",
        str(args.pheno),
        "--trait",
        str(args.trait or ""),
        "--unit-kind",
        str(args.unit_kind),
        "--extension",
        str(args.extension),
        "--bin-mode",
        str(args.bin_mode),
        "--ml-method",
        str(args.ml_method),
        "--topk",
        str(args.topk),
        "--permutation-repeats",
        str(args.permutation_repeats),
        "--n-estimators",
        str(args.n_estimators),
        "--max-depth",
        str(args.max_depth),
        "--seed",
        str(args.seed),
        "--layer",
        str(args.layer),
        "--exhaustive-depth",
        str(args.exhaustive_depth),
        "--beam-width",
        str(args.beam_width),
        "--logic-gate",
        "--rank-score",
        str(args.rank_score),
        "--maf",
        str(args.maf),
        "--geno",
        str(args.geno),
        "--block-cols",
        str(args.block_cols),
        "--threads",
        str(args.threads),
        "--top-rules-per-unit",
        str(args.top_rules_per_unit),
        "--max-output-rules",
        str(args.max_output_rules),
        "--sample-limit",
        str(args.sample_limit),
        "--scan-bimranges",
        str(args.scan_bimranges or ""),
    ]
    if args.step is not None:
        cmd.extend(["--step", str(args.step)])
    if args.rule_permutation:
        cmd.append("--rule-permutation")
    if args.no_clean:
        cmd.append("--no-clean")

    env = os.environ.copy()
    env["JX_GARFIELD_SCORE_BACKEND"] = backend
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=Path.cwd(), env=env, stdout=log, stderr=log, check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        hint = ""
        if "No SNPs left after pure-line BED filtering" in log_text:
            hint = "No SNPs survived filtering; retry with lower --maf and/or higher --geno."
        return {
            "backend": backend,
            "status": "error",
            "returncode": int(proc.returncode),
            "elapsed_s": float(elapsed),
            "log_path": str(log_path),
            "error_hint": hint,
        }
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    payload["status"] = "ok"
    payload["log_path"] = str(log_path)
    return payload


def _compare(results: list[dict[str, Any]]) -> dict[str, Any]:
    oks = [r for r in results if r.get("status") == "ok"]
    if not oks:
        return {}
    base = oks[0]
    comparisons: dict[str, Any] = {}
    for item in oks[1:]:
        comparisons[str(item["backend"])] = {
            "baseline": str(base["backend"]),
            "same_rule_hash": bool(item.get("rule_hash") == base.get("rule_hash")),
            "rule_hash": item.get("rule_hash"),
            "baseline_rule_hash": base.get("rule_hash"),
            "n_rules": item.get("result", {}).get("n_rules"),
            "baseline_n_rules": base.get("result", {}).get("n_rules"),
            **_score_diff(base.get("result", {}), item.get("result", {})),
        }
    return comparisons


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "End-to-end Garfield logic-search benchmark for centered-gain score backends. "
            "Runs garfield_logic_search_bed in one child process per backend."
        )
    )
    ap.add_argument("--bfile", default="test/mouse_hs1940")
    ap.add_argument("--pheno", default="test/mouse_hs1940.pheno.txt")
    ap.add_argument("--trait", default="test")
    ap.add_argument("--backends", default="legacy,cpu,gpu")
    ap.add_argument("--out", default="test/garfield_logic_backend_bench")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--unit-kind", default="window")
    ap.add_argument("--extension", type=int, default=100000)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--scan-bimranges", default="")
    ap.add_argument("--bin-mode", default="bin")
    ap.add_argument("--ml-method", default="rf")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--permutation-repeats", type=int, default=0)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--exhaustive-depth", type=int, default=1)
    ap.add_argument("--beam-width", type=int, default=100)
    ap.add_argument("--logic-gate", default="ao")
    ap.add_argument("--rank-score", default="interaction_gain")
    ap.add_argument("--maf", type=float, default=0.001)
    ap.add_argument("--geno", type=float, default=0.2)
    ap.add_argument("--block-cols", type=int, default=65536)
    ap.add_argument("--top-rules-per-unit", type=int, default=1)
    ap.add_argument("--max-output-rules", type=int, default=200)
    ap.add_argument("--sample-limit", type=int, default=0)
    ap.add_argument("--rule-permutation", action="store_true")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--backend", default="", help=argparse.SUPPRESS)
    ap.add_argument("--out-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--out-json", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        return _worker(args)

    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    backends = [x.strip() for x in str(args.backends).split(",") if x.strip()]
    if not backends:
        raise ValueError("--backends cannot be empty")

    results = [_run_child(args, backend, root) for backend in backends]
    summary = {
        "bfile": str(args.bfile),
        "pheno": str(args.pheno),
        "trait": str(args.trait),
        "backends": backends,
        "results": results,
        "comparisons": _compare(results),
        "note": (
            "Use metal_runtime_status as the primary signal for Garfield GPU readiness. "
            "Explicit backend=gpu now fails fast if score_gpu is unavailable; backend=auto "
            "may still choose a non-GPU path."
        ),
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    compact = {
        "summary_json": str(summary_path),
        "runs": [
            {
                "backend": r.get("backend"),
                "status": r.get("status"),
                "wall_s": r.get("wall_s"),
                "peak_rss_gb": r.get("peak_rss_gb"),
                "n_rules": (r.get("result") or {}).get("n_rules"),
                "units_scanned": (r.get("result") or {}).get("units_scanned"),
                "score_wall_s": (r.get("result") or {}).get("timing_scan_literal_score_wall_s"),
                "score_share_total_pct": (r.get("result") or {}).get(
                    "timing_literal_score_share_of_total_pct"
                ),
                "score_share_scan_pct": (r.get("result") or {}).get(
                    "timing_literal_score_share_of_scan_pct"
                ),
                "beam_share_total_pct": (r.get("result") or {}).get(
                    "timing_beam_share_of_total_pct"
                ),
                "gpu_available_probe": (r.get("backend_probe") or {}).get("gpu_available"),
                "metal_score_gpu_ready": (r.get("metal_runtime_status") or {}).get("score_gpu_ready"),
                "metal_device_name": (r.get("metal_runtime_status") or {}).get("device_name"),
                "log_path": r.get("log_path"),
            }
            for r in results
        ],
        "comparisons": summary["comparisons"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
