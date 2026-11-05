#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GARFIELD logic benchmark workflow.

Pipeline per run:
1) simulate phenotype from input genotype (strict AND mode by default);
2) derive a target interval from simulated causal loci (+/- flank);
3) extract interval genotype only (avoid full-genome benchmark scan);
4) run GARFIELD on the extracted interval;
5) evaluate whether top rules recover simulated causal loci.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from janusx.gfreader import inspect_genotype_file, load_genotype_chunks
from janusx.gfreader.gfreader import save_genotype_streaming
from janusx.script._common.cli_args import add_common_out_arg, add_common_thread_arg
from janusx.script._common.config_render import emit_cli_configuration
from janusx.script._common.genoio import determine_genotype_source_from_args as determine_genotype_source
from janusx.script._common.genocache import configure_genotype_cache_from_out
from janusx.script._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from janusx.script._common.log import local_log_capture, setup_logging
from janusx.script._common.outprefix import apply_output_prefix_compat
from janusx.script._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
    format_path_for_display,
)
from janusx.script._common.progress import CliStatus, log_success
from janusx.script._common.threads import detect_effective_threads, format_requested_thread_usage
from janusx.script.simulation import simulate_phenotype_from_genofile, write_phenotypes, write_sites


def _normalize_chrom(chrom: object) -> str:
    s = str(chrom).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    return s.upper()


def _safe_prefix(name: str) -> str:
    out = str(name).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("._")
    return out if out else "garfieldbench"


def _parse_expression_sites(expr: str) -> set[tuple[str, int]]:
    pat = re.compile(r"([^\s&]+)_([0-9]+)\[")
    out: set[tuple[str, int]] = set()
    for m in pat.finditer(str(expr)):
        out.add((_normalize_chrom(m.group(1)), int(m.group(2))))
    return out


def _parse_window_unit_name(unit_name: str) -> Optional[tuple[str, int, int]]:
    """Parse a GARFIELD window label (``chrom:start-end``).

    GARFIELD writes chromosome labels with and without a ``chr`` prefix.  The
    benchmark keeps the normalized chromosome representation used by the LD
    helpers, so a label such as ``chr2:10-20`` becomes ``("2", 10, 20)``.
    ``None`` is returned for the legacy/global unit labels that do not encode a
    genomic interval.
    """
    m = re.match(r"^\s*([^:]+):(\d+)-(\d+)\s*$", str(unit_name))
    if m is None:
        return None
    chrom = _normalize_chrom(m.group(1))
    start = int(m.group(2))
    end = int(m.group(3))
    if start > end:
        start, end = end, start
    return chrom, start, end


def _rule_sites(row: dict[str, Any]) -> set[tuple[str, int]]:
    """Extract sites from both historical and current rules TSV schemas."""
    expr = row.get("expression", None)
    if expr in (None, ""):
        expr = row.get("snp_name", None)
    if expr in (None, ""):
        expr = row.get("rule", "")
    return _parse_expression_sites(str(expr))


def _build_window_credit_sets(
    rows: list[dict[str, Any]],
    causal_set: set[tuple[str, int]],
    site_genotypes: dict[tuple[str, int], np.ndarray],
    ld_r2_threshold: float,
) -> dict[str, dict[str, Any]]:
    """Build LD credit sets independently for every scanned GARFIELD window.

    Only SNPs retained in a rule for the same window are representatives.  A
    causal SNP receives a representative when the two sites are on the same
    chromosome and their dosage ``r^2`` reaches ``ld_r2_threshold``.  Keeping
    this mapping per window prevents a proxy from a different window from
    turning an untested causal combination into a hit.

    The returned mapping is intentionally explicit so callers can report both
    the representative set and the causal-to-proxy map in a sidecar file.
    """
    grouped: dict[str, set[tuple[str, int]]] = {}
    for row in rows:
        unit = str(row.get("unit_name", "__all__") or "__all__")
        grouped.setdefault(unit, set()).update(_rule_sites(row))

    out: dict[str, dict[str, Any]] = {}
    threshold = float(ld_r2_threshold)
    for unit, representatives in grouped.items():
        parsed = _parse_window_unit_name(unit)

        def _in_unit(site: tuple[str, int]) -> bool:
            if parsed is None:
                return True
            return (
                _normalize_chrom(site[0]) == parsed[0]
                and parsed[1] <= int(site[1]) <= parsed[2]
            )

        causal_in_unit = {c for c in causal_set if _in_unit(c)}
        if len(causal_in_unit) == 0:
            continue
        proxy_map: dict[tuple[str, int], set[tuple[str, int]]] = {}
        for causal in sorted(causal_in_unit):
            gx = site_genotypes.get(causal)
            proxies: set[tuple[str, int]] = set()
            for rep in representatives:
                if _normalize_chrom(rep[0]) != _normalize_chrom(causal[0]):
                    continue
                if rep == causal:
                    proxies.add(rep)
                    continue
                gy = site_genotypes.get(rep)
                if gx is None or gy is None:
                    continue
                if _site_vec_r2(gx, gy) >= threshold:
                    proxies.add(rep)
            proxy_map[causal] = proxies
        out[unit] = {
            "representatives": set(representatives),
            "causal_sites": causal_in_unit,
            "causal_proxy_map": proxy_map,
        }
    return out


def _site_vec_r2(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(m)) < 3:
        return 0.0
    a = a[m]
    b = b[m]
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    va = float(np.dot(a, a))
    vb = float(np.dot(b, b))
    if va <= 1e-12 or vb <= 1e-12:
        return 0.0
    r = float(np.dot(a, b) / np.sqrt(va * vb))
    r2 = r * r
    if r2 < 0.0:
        return 0.0
    if r2 > 1.0:
        return 1.0
    return float(r2)


def _load_site_genotypes(gfile: str, chunk_size: int) -> dict[tuple[str, int], np.ndarray]:
    out: dict[tuple[str, int], np.ndarray] = {}
    for m, sites in load_genotype_chunks(
        gfile,
        chunk_size=int(chunk_size),
        maf=0.0,
        missing_rate=1.0,
        impute=True,
    ):
        mm = np.asarray(m, dtype=np.float32)
        for i, s in enumerate(sites):
            chrom = _normalize_chrom(getattr(s, "chrom", ""))
            pos = int(getattr(s, "pos", 0))
            if pos <= 0:
                continue
            key = (chrom, pos)
            if key not in out:
                out[key] = np.asarray(mm[i, :], dtype=np.float32).copy()
    return out


def _covers_all_causal_with_ld(
    seen: set[tuple[str, int]],
    causal_set: set[tuple[str, int]],
    ld_proxy_map: dict[tuple[str, int], set[tuple[str, int]]],
) -> bool:
    if len(causal_set) == 0:
        return False
    causal_order = sorted(list(causal_set), key=lambda c: len(seen.intersection(ld_proxy_map.get(c, set()))))
    used: set[tuple[str, int]] = set()

    def _dfs(i: int) -> bool:
        if i >= len(causal_order):
            return True
        c = causal_order[i]
        cands = sorted(list(seen.intersection(ld_proxy_map.get(c, set()))))
        if len(cands) == 0:
            return False
        for s in cands:
            if s in used:
                continue
            used.add(s)
            if _dfs(i + 1):
                return True
            used.remove(s)
        return False

    return _dfs(0)


def _causal_site_set(causal_sites: list[tuple[str, int, int]]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for chrom, start, _end in causal_sites:
        out.add((_normalize_chrom(chrom), int(start)))
    return out


def _derive_region_from_causal(
    causal_sites: list[tuple[str, int, int]],
    flank_bp: int,
) -> tuple[str, int, int]:
    if len(causal_sites) == 0:
        raise ValueError("No causal sites generated; unable to derive target region.")

    chrom0 = _normalize_chrom(causal_sites[0][0])
    pos = [int(x[1]) for x in causal_sites if _normalize_chrom(x[0]) == chrom0]
    if len(pos) == 0:
        raise ValueError("Causal sites are empty after chromosome normalization.")
    lo = min(pos)
    hi = max(pos)
    start = max(1, lo - int(flank_bp))
    end = hi + int(flank_bp)
    if end < start:
        end = start
    return chrom0, start, end


def _dynamic_window_from_causal(
    causal_sites: list[tuple[str, int, int]],
    *,
    base_extension: int,
    base_step: Optional[int],
) -> tuple[int, int, int]:
    """
    Build GARFIELD window params from causal-site span on the selected chromosome.
    Returns (effective_extension, effective_step, causal_span_bp).
    """
    ext0 = max(1, int(base_extension))
    step0 = int(base_step) if base_step is not None else max(1, ext0 // 2)
    if len(causal_sites) == 0:
        return ext0, max(1, step0), 0

    chrom0 = _normalize_chrom(causal_sites[0][0])
    pos = sorted([int(x[1]) for x in causal_sites if _normalize_chrom(x[0]) == chrom0])
    if len(pos) == 0:
        return ext0, max(1, step0), 0
    span = int(max(pos) - min(pos)) if len(pos) >= 2 else 0
    # Ensure one window centered near one endpoint can still cover the other.
    eff_ext = max(ext0, span)
    # Use span-adaptive step to match enlarged windows.
    eff_step = max(1, eff_ext // 2)
    return int(eff_ext), int(eff_step), int(span)


def _extract_region_plink(
    gfile: str,
    out_prefix: str,
    chrom: str,
    start: int,
    end: int,
    chunk_size: int,
) -> tuple[int, int]:
    sample_ids, _ = inspect_genotype_file(gfile)
    sample_ids = [str(x) for x in sample_ids]
    stream = load_genotype_chunks(
        gfile,
        chunk_size=int(chunk_size),
        maf=0.0,
        missing_rate=1.0,
        impute=False,
        bim_range=(str(chrom), int(start), int(end)),
    )
    first = next(stream, None)
    if first is None:
        raise ValueError(f"No SNPs found in target region {chrom}:{start}-{end}.")

    n_sites = int(first[0].shape[0])

    def _iter():
        nonlocal n_sites
        m0, s0 = first
        yield np.asarray(m0, dtype=np.float32), list(s0)
        for m, s in stream:
            n_sites += int(m.shape[0])
            yield np.asarray(m, dtype=np.float32), list(s)

    save_genotype_streaming(out_prefix, sample_ids, _iter(), fmt="plink")
    return len(sample_ids), int(n_sites)


def _run_garfield_subprocess(
    *,
    bfile_prefix: str,
    pheno_path: str,
    out_dir: str,
    out_prefix: str,
    feature_source: str,
    input_maf: float,
    input_geno: float,
    extension: int,
    step: Optional[int],
    nsnp: int,
    max_pick: int,
    top_k_validate: int,
    val_frac: float,
    seed: int,
    thread: int,
    log_path: str,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "janusx.script.garfield",
        "-bfile",
        str(bfile_prefix),
        "-p",
        str(pheno_path),
        "-n",
        "0",
        "-maf",
        str(float(input_maf)),
        "-geno",
        str(float(input_geno)),
        "-w",
        str(int(extension)),
    ]
    # The current GARFIELD interface takes window extension and step as the
    # optional values following ``-w``.  ``nsnp`` is the historical benchmark
    # name for the unified ``-width`` parameter; validation options are kept
    # in garfieldbench's result schema for compatibility but are no longer
    # passed to the main CLI because that stage is now fixed in GARFIELD.
    if step is not None:
        cmd.append(str(int(step)))
    else:
        cmd.append(str(max(1, int(extension) // 2)))
    cmd.extend(
        [
            "-width",
            str(int(nsnp)),
            "-m",
            str(int(max_pick)),
            "--seed",
            str(int(seed)),
            "-t",
            str(int(thread)),
            "-o",
            str(Path(out_dir) / str(out_prefix)),
        ]
    )
    with local_log_capture(log_path, mode="w") as fw:
        rc = subprocess.run(cmd, stdout=fw, stderr=subprocess.STDOUT, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"GARFIELD command failed (exit={rc}). See log: {log_path}")


def _evaluate_rules(
    rules_path: str,
    causal_set: set[tuple[str, int]],
    top_k_hit: int,
    *,
    hit_mode: str = "all-ld",
    ld_r2_threshold: float = 0.8,
    ld_geno_source: Optional[str] = None,
    ld_chunk_size: int = 100_000,
    credit_sets_path: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate exact and per-window LD-credit recovery separately.

    ``top*_hit_*`` remains the metric selected by ``hit_mode`` for backwards
    compatibility.  The explicit ``exact_*`` and ``credit_*`` fields are
    always returned so a benchmark cannot confuse a global LD proxy match
    with an exact causal recovery.
    """
    empty = {
        "top1_hit_any": False,
        "top1_hit_all": False,
        "topk_hit_any": False,
        "topk_hit_all": False,
        "first_hit_any_rank": np.nan,
        "first_hit_all_rank": np.nan,
        "exact_top1_hit_any": False,
        "exact_top1_hit_all": False,
        "exact_topk_hit_any": False,
        "exact_topk_hit_all": False,
        "exact_first_hit_any_rank": np.nan,
        "exact_first_hit_all_rank": np.nan,
        "credit_top1_hit_any": False,
        "credit_top1_hit_all": False,
        "credit_topk_hit_any": False,
        "credit_topk_hit_all": False,
        "credit_first_hit_any_rank": np.nan,
        "credit_first_hit_all_rank": np.nan,
        "credit_status": "not_tested",
        "credit_window_count": 0,
        "credit_representative_count": 0,
        "credit_complete_window": False,
        "best_val_score": np.nan,
        "best_rule": "",
    }
    if len(causal_set) == 0:
        return empty
    with open(rules_path, "r", encoding="utf-8") as fr:
        rd = csv.DictReader(fr, delimiter="\t")
        rows = list(rd)
    if len(rows) == 0:
        return empty

    if str(hit_mode) not in {"all", "all-ld"}:
        raise ValueError(f"Unsupported hit mode: {hit_mode}")
    if str(hit_mode) == "all-ld" and ld_geno_source is None:
        raise ValueError("ld_geno_source is required when --hit-mode=all-ld")

    # GARFIELD's current rules schema uses ``snp_name`` and ``MLrank`` while
    # older benchmark outputs used ``expression`` and numeric ``rank``.  The
    # file order is the authoritative output ranking for the current schema;
    # use a numeric rank only when it is actually present.
    normalized_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        rank_text = str(row.get("rank", "") or "").strip()
        try:
            rank = int(rank_text) if rank_text else i
        except ValueError:
            rank = i
        sites = _rule_sites(row)
        unit = str(row.get("unit_name", "__all__") or "__all__")
        expression = str(row.get("expression", "") or row.get("snp_name", ""))
        normalized_rows.append({"row": row, "rank": rank, "sites": sites, "unit": unit, "expression": expression})

    site_genotypes: dict[tuple[str, int], np.ndarray] = {}
    credit_sets: dict[str, dict[str, Any]] = {}
    if ld_geno_source is not None:
        site_genotypes = _load_site_genotypes(str(ld_geno_source), int(ld_chunk_size))
        credit_sets = _build_window_credit_sets(
            rows=rows,
            causal_set=causal_set,
            site_genotypes=site_genotypes,
            ld_r2_threshold=float(ld_r2_threshold),
        )

    if credit_sets_path is not None:
        p = Path(credit_sets_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as fw:
            wr = csv.writer(fw, delimiter="\t")
            wr.writerow(
                [
                    "unit_name",
                    "representative",
                    "causal",
                    "r2",
                    "causal_is_representative",
                ]
            )
            for unit in sorted(credit_sets):
                data = credit_sets[unit]
                mapped_representatives: set[tuple[str, int]] = set()
                for causal in sorted(data["causal_proxy_map"]):
                    for rep in sorted(data["causal_proxy_map"][causal]):
                        mapped_representatives.add(rep)
                        r2 = 1.0 if rep == causal else _site_vec_r2(
                            site_genotypes.get(causal, np.array([], dtype=np.float32)),
                            site_genotypes.get(rep, np.array([], dtype=np.float32)),
                        )
                        wr.writerow(
                            [
                                unit,
                                f"{rep[0]}:{rep[1]}",
                                f"{causal[0]}:{causal[1]}",
                                f"{float(r2):.8g}",
                                str(bool(rep == causal)),
                            ]
                        )
                # Keep representatives with no causal proxy in the sidecar as
                # well.  They are still part of the per-window credit set and
                # make the set size auditable rather than silently dropping
                # uninformative retained SNPs.
                for rep in sorted(set(data["representatives"]).difference(mapped_representatives)):
                    wr.writerow([unit, f"{rep[0]}:{rep[1]}", "", "", "False"])

    exact_top1_any = False
    exact_top1_all = False
    exact_topk_any = False
    exact_topk_all = False
    exact_first_any: float = np.nan
    exact_first_all: float = np.nan
    credit_top1_any = False
    credit_top1_all = False
    credit_topk_any = False
    credit_topk_all = False
    credit_first_any: float = np.nan
    credit_first_all: float = np.nan

    for i, item in enumerate(normalized_rows, start=1):
        rank = int(item["rank"])
        seen = set(item["sites"])
        exact_any = len(seen.intersection(causal_set)) > 0
        exact_all = causal_set.issubset(seen)

        data = credit_sets.get(str(item["unit"]))
        credit_any = False
        credit_all = False
        if data is not None:
            proxy_map = data["causal_proxy_map"]
            credit_any = any(len(seen.intersection(proxy_map.get(c, set()))) > 0 for c in causal_set)
            credit_all = _covers_all_causal_with_ld(seen, causal_set, proxy_map)

        if i == 1:
            exact_top1_any, exact_top1_all = exact_any, exact_all
            credit_top1_any, credit_top1_all = credit_any, credit_all
        if i <= int(top_k_hit):
            exact_topk_any = exact_topk_any or exact_any
            exact_topk_all = exact_topk_all or exact_all
            credit_topk_any = credit_topk_any or credit_any
            credit_topk_all = credit_topk_all or credit_all
        if exact_any and np.isnan(exact_first_any):
            exact_first_any = float(rank)
        if exact_all and np.isnan(exact_first_all):
            exact_first_all = float(rank)
        if credit_any and np.isnan(credit_first_any):
            credit_first_any = float(rank)
        if credit_all and np.isnan(credit_first_all):
            credit_first_all = float(rank)

    complete_windows = [
        unit
        for unit, data in credit_sets.items()
        if causal_set.issubset(set(data["causal_sites"]))
    ]
    if len(complete_windows) == 0:
        credit_status = "not_tested"
    elif exact_topk_all:
        credit_status = "recovered_exact"
    elif credit_topk_all:
        credit_status = "recovered_credit"
    else:
        credit_status = "tested_but_not_recovered"

    best_row = normalized_rows[0]
    raw_best = best_row["row"]
    score_text = str(raw_best.get("val_score", "") or raw_best.get("score", "nan"))
    try:
        best_val = float(score_text)
    except ValueError:
        best_val = float("nan")
    result = dict(empty)
    result.update(
        {
            "exact_top1_hit_any": bool(exact_top1_any),
            "exact_top1_hit_all": bool(exact_top1_all),
            "exact_topk_hit_any": bool(exact_topk_any),
            "exact_topk_hit_all": bool(exact_topk_all),
            "exact_first_hit_any_rank": exact_first_any,
            "exact_first_hit_all_rank": exact_first_all,
            "credit_top1_hit_any": bool(credit_top1_any),
            "credit_top1_hit_all": bool(credit_top1_all),
            "credit_topk_hit_any": bool(credit_topk_any),
            "credit_topk_hit_all": bool(credit_topk_all),
            "credit_first_hit_any_rank": credit_first_any,
            "credit_first_hit_all_rank": credit_first_all,
            "credit_status": credit_status,
            "credit_window_count": int(len(credit_sets)),
            "credit_representative_count": int(
                len(set().union(*(set(x["representatives"]) for x in credit_sets.values())))
                if credit_sets
                else 0
            ),
            "credit_complete_window": bool(len(complete_windows) > 0),
            "best_val_score": best_val,
            "best_rule": str(best_row["expression"]),
        }
    )
    if str(hit_mode) == "all-ld":
        result.update(
            {
                "top1_hit_any": bool(credit_top1_any),
                "top1_hit_all": bool(credit_top1_all),
                "topk_hit_any": bool(credit_topk_any),
                "topk_hit_all": bool(credit_topk_all),
                "first_hit_any_rank": credit_first_any,
                "first_hit_all_rank": credit_first_all,
            }
        )
    else:
        result.update(
            {
                "top1_hit_any": bool(exact_top1_any),
                "top1_hit_all": bool(exact_top1_all),
                "topk_hit_any": bool(exact_topk_any),
                "topk_hit_all": bool(exact_topk_all),
                "first_hit_any_rank": exact_first_any,
                "first_hit_all_rank": exact_first_all,
            }
        )
    return result


def _format_site_set(sites: set[tuple[str, int]]) -> str:
    if len(sites) == 0:
        return ""
    items = sorted(list(sites), key=lambda x: (x[0], x[1]))
    return ";".join([f"{c}:{p}" for c, p in items])


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx garfieldbench",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx garfieldbench -vcf example/mouse_hs1940.vcf.gz -o bench",
                "jx garfieldbench -bfile data/mouse -seed 2026 -n-runs 20",
            ]
        ),
        description="GARFIELD benchmark: simulate logic phenotype, extract local interval, run GARFIELD, score hit rate.",
    )

    required_group = parser.add_argument_group("Required arguments")
    optional_group = parser.add_argument_group("Optional arguments")

    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument("-vcf", "--vcf", type=str, help="Input genotype VCF(.gz).")
    geno_group.add_argument("-hmp", "--hmp", type=str, help="Input genotype HMP(.gz).")
    geno_group.add_argument("-bfile", "--bfile", type=str, help="Input PLINK prefix.")
    geno_group.add_argument("-file", "--file", type=str, help="Input numeric matrix prefix/path.")

    add_common_out_arg(optional_group, default=".", help_profile="default")
    optional_group.add_argument("-prefix", "--prefix", type=str, default=None, help=argparse.SUPPRESS)
    optional_group.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    optional_group.add_argument("--n-runs", type=int, default=1, help="Number of benchmark runs.")
    optional_group.add_argument(
        "--region-flank-mb",
        type=float,
        default=20.0,
        help="Region flank size in Mb around causal loci (default: 20).",
    )
    optional_group.add_argument(
        "-chunksize",
        "--chunksize",
        type=int,
        default=100_000,
        help="Chunk size for simulation and extraction.",
    )

    # Simulation parameters
    optional_group.add_argument("-maf", "--maf", type=float, default=0.02, help="Simulation MAF threshold.")
    optional_group.add_argument("-geno", "--geno", type=float, default=0.05, help="Simulation missing threshold.")
    optional_group.add_argument("-ve", "--ve", type=float, default=1.0, help="Simulation residual variance.")
    optional_group.add_argument("-pve", "--pve", type=float, default=0.1, help="Simulation polygenic PVE.")
    optional_group.add_argument("-windows", "--windows", type=int, default=50_000, help="Simulation window size.")
    optional_group.add_argument("--and-k-min", type=int, default=2, help="Minimum AND gate loci.")
    optional_group.add_argument("--and-k-max", type=int, default=2, help="Maximum AND gate loci.")
    optional_group.add_argument("--and-ld-max", type=float, default=0.3, help="Maximum LD (r^2) in AND gate.")
    optional_group.add_argument("--and-het-max", type=float, default=0.05, help="Maximum heterozygosity before BIN02 collapse in AND gate.")
    optional_group.add_argument("--and-af-min", type=float, default=0.02, help="Minimum AND gate frequency.")
    optional_group.add_argument("--and-af-max", type=float, default=0.90, help="Maximum AND gate frequency.")
    optional_group.add_argument("--and-target-pve", type=float, default=0.45, help="Target PVE from AND gate.")
    optional_group.add_argument("--and-max-iter", type=int, default=200, help="Max attempts to sample AND gate.")

    # GARFIELD parameters
    optional_group.add_argument("--feature-source", choices=["bin", "mbin"], default="mbin")
    optional_group.add_argument("-ext", "--extension", type=int, default=50_000, help="GARFIELD window extension.")
    optional_group.add_argument("-step", "--step", type=int, default=None, help="GARFIELD window step.")
    optional_group.add_argument(
        "--dynamic-window-from-causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-adjust GARFIELD ext/step by simulated causal-site distance (default: enabled).",
    )
    optional_group.add_argument("-nsnp", "--nsnp", type=int, default=12, help="GARFIELD beam width.")
    optional_group.add_argument("-m", "--max-pick", type=int, default=2, help="GARFIELD max literals.")
    optional_group.add_argument("--top-k-validate", type=int, default=20, help="GARFIELD top-k validation.")
    optional_group.add_argument("--val-frac", type=float, default=0.25, help="GARFIELD validation fraction.")
    add_common_thread_arg(
        optional_group,
        default_threads=detect_effective_threads(),
        help_text="CPU threads.",
    )
    optional_group.add_argument("--top-k-hit", type=int, default=10, help="Hit criterion on top-K rules.")
    optional_group.add_argument(
        "--hit-mode",
        choices=["all", "all-ld"],
        default="all-ld",
        help=(
            "all: exact causal sites; all-ld: use retained representatives from the same "
            "window as high-LD proxies (default: all-ld)."
        ),
    )
    optional_group.add_argument(
        "--hit-ld-r2",
        type=float,
        default=0.8,
        help="r^2 threshold for LD-proxy hit when --hit-mode=all-ld (default: 0.8).",
    )

    return parser


def _run_one(
    *,
    run_idx: int,
    seed: int,
    gfile: str,
    out_root: Path,
    args,
    logger,
) -> dict[str, Any]:
    run_name = f"seed{seed}"
    run_dir = out_root / run_name
    sim_dir = run_dir / "sim"
    region_dir = run_dir / "region"
    gar_dir = run_dir / "garfield"
    for d in [sim_dir, region_dir, gar_dir]:
        d.mkdir(parents=True, exist_ok=True)

    sim_prefix = str(sim_dir / "sim")
    with CliStatus(f"[{run_name}] Simulating phenotype...", enabled=True) as task:
        try:
            y, causal_sites = simulate_phenotype_from_genofile(
                gfile=gfile,
                mode="garfield",
                chunk_size=int(args.chunksize),
                seed=int(seed),
                maf=float(args.maf),
                missing_rate=float(args.geno),
                pve=float(args.pve),
                ve=float(args.ve),
                windows=int(args.windows),
                and_k_min=int(args.and_k_min),
                and_k_max=int(args.and_k_max),
                and_ld_max=float(args.and_ld_max),
                and_het_max=float(args.and_het_max),
                and_af_min=float(args.and_af_min),
                and_af_max=float(args.and_af_max),
                and_target_pve=float(args.and_target_pve),
                and_max_iter=int(args.and_max_iter),
            )
        except Exception:
            task.fail(f"[{run_name}] Simulating phenotype ...Failed")
            raise
        task.complete(f"[{run_name}] Simulating phenotype ...Finished")

    sample_ids, _ = inspect_genotype_file(gfile)
    sample_ids = np.asarray(sample_ids, dtype=str)
    write_phenotypes(sim_prefix, sample_ids, y, seed=int(seed))
    write_sites(sim_prefix, causal_sites)

    causal_set = _causal_site_set(causal_sites)
    flank_bp = int(round(float(args.region_flank_mb) * 1_000_000.0))
    chrom, region_start, region_end = _derive_region_from_causal(causal_sites, flank_bp)
    if bool(args.dynamic_window_from_causal):
        run_ext, run_step, causal_span_bp = _dynamic_window_from_causal(
            causal_sites,
            base_extension=int(args.extension),
            base_step=args.step,
        )
    else:
        run_ext = int(args.extension)
        run_step = args.step if args.step is not None else max(1, int(args.extension) // 2)
        chrom0 = _normalize_chrom(causal_sites[0][0]) if len(causal_sites) > 0 else ""
        pos0 = [int(x[1]) for x in causal_sites if _normalize_chrom(x[0]) == chrom0]
        causal_span_bp = int(max(pos0) - min(pos0)) if len(pos0) >= 2 else 0

    region_prefix = str(region_dir / "region")
    with CliStatus(f"[{run_name}] Extracting target region...", enabled=True) as task:
        try:
            n_samples_region, n_sites_region = _extract_region_plink(
                gfile=gfile,
                out_prefix=region_prefix,
                chrom=chrom,
                start=region_start,
                end=region_end,
                chunk_size=int(args.chunksize),
            )
        except Exception:
            task.fail(f"[{run_name}] Extracting target region ...Failed")
            raise
        task.complete(f"[{run_name}] Extracting target region ...Finished")

    gf_prefix = f"gf_{seed}"
    gf_log = str(gar_dir / "garfield.run.log")
    with CliStatus(f"[{run_name}] Running GARFIELD...", enabled=True) as task:
        try:
            _run_garfield_subprocess(
                bfile_prefix=region_prefix,
                pheno_path=f"{sim_prefix}.pheno.txt",
                out_dir=str(gar_dir),
                out_prefix=gf_prefix,
                feature_source=str(args.feature_source),
                input_maf=float(args.maf),
                input_geno=float(args.geno),
                extension=int(run_ext),
                step=int(run_step),
                nsnp=int(args.nsnp),
                max_pick=int(args.max_pick),
                top_k_validate=int(args.top_k_validate),
                val_frac=float(args.val_frac),
                seed=int(seed),
                thread=int(args.thread),
                log_path=gf_log,
            )
        except Exception:
            task.fail(f"[{run_name}] Running GARFIELD ...Failed")
            raise
        task.complete(f"[{run_name}] Running GARFIELD ...Finished")

    rules_files = sorted(list(gar_dir.glob("*.garfield.rules.tsv")))
    if len(rules_files) == 0:
        raise FileNotFoundError(f"[{run_name}] rules file not found under {gar_dir}")
    rules_path = str(rules_files[0])
    credit_sets_path = str(gar_dir / f"{gf_prefix}.garfield.credit_sets.tsv")

    hit = _evaluate_rules(
        rules_path,
        causal_set,
        int(args.top_k_hit),
        hit_mode=str(args.hit_mode),
        ld_r2_threshold=float(args.hit_ld_r2),
        ld_geno_source=region_prefix,
        ld_chunk_size=int(min(max(1_000, int(args.chunksize)), 250_000)),
        credit_sets_path=credit_sets_path,
    )
    out = {
        "run_index": int(run_idx),
        "seed": int(seed),
        "region_chrom": str(chrom),
        "region_start": int(region_start),
        "region_end": int(region_end),
        "region_span_mb": (int(region_end) - int(region_start) + 1) / 1_000_000.0,
        "n_samples_region": int(n_samples_region),
        "n_sites_region": int(n_sites_region),
        "causal_span_bp": int(causal_span_bp),
        "effective_extension": int(run_ext),
        "effective_step": int(run_step),
        "n_causal_sites": int(len(causal_set)),
        "causal_sites": _format_site_set(causal_set),
        "rules_path": rules_path,
        "credit_sets_path": credit_sets_path,
        "sim_prefix": sim_prefix,
        "garfield_log": gf_log,
        "hit_mode": str(args.hit_mode),
        "hit_ld_r2": float(args.hit_ld_r2),
        **hit,
    }
    logger.info(
        f"[{run_name}] span={int(causal_span_bp)}bp, ext={int(run_ext)}, step={int(run_step)}, "
        f"exact_top1_hit_all={bool(hit['exact_top1_hit_all'])}, "
        f"exact_top{int(args.top_k_hit)}_hit_all={bool(hit['exact_topk_hit_all'])}, "
        f"credit_top{int(args.top_k_hit)}_hit_all={bool(hit['credit_topk_hit_all'])}, "
        f"credit_status={hit['credit_status']}, "
        f"best_val={float(hit['best_val_score']):.6g}"
    )
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    gfile, auto_prefix = determine_genotype_source(args)
    out_dir_str, outprefix, prefix = apply_output_prefix_compat(
        args,
        _safe_prefix(auto_prefix),
        argv=argv,
        fallback_prefix="garfieldbench",
    )
    out_dir = Path(out_dir_str).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_genotype_cache_from_out(str(out_dir))

    log_path = f"{outprefix}.garfieldbench.log"
    logger = setup_logging(log_path)

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype input"))
    if not ensure_all_true(checks):
        return 1

    if int(args.n_runs) <= 0:
        logger.error("--n-runs must be > 0")
        return 1
    if float(args.region_flank_mb) <= 0:
        logger.error("--region-flank-mb must be > 0")
        return 1
    detected_threads = int(detect_effective_threads())
    requested_threads = int(args.thread)
    if int(args.thread) <= 0:
        args.thread = int(detected_threads)
    if args.step is not None and int(args.step) <= 0:
        logger.error("-step/--step must be > 0")
        return 1
    if not (0.0 <= float(args.and_het_max) <= 1.0):
        logger.error("--and-het-max must be in [0, 1].")
        return 1
    if not (0.0 <= float(args.hit_ld_r2) <= 1.0):
        logger.error("--hit-ld-r2 must be in [0, 1].")
        return 1

    emit_cli_configuration(
        logger,
        app_title="JanusX - GARFIELD Benchmark",
        config_title="GARFIELDBENCH CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "General",
                [
                    ("Genotype input", gfile),
                    ("Runs", int(args.n_runs)),
                    ("Seed(base)", int(args.seed)),
                    ("Region flank(Mb)", float(args.region_flank_mb)),
                    ("Sim mode", "garfield"),
                    ("AND k[min,max]", f"{int(args.and_k_min)},{int(args.and_k_max)}"),
                    ("AND het max", float(args.and_het_max)),
                    ("AND target PVE", float(args.and_target_pve)),
                    ("GARFIELD feature", str(args.feature_source)),
                    ("GARFIELD ext", int(args.extension)),
                    ("GARFIELD step", args.step if args.step is not None else "ext/2"),
                    ("Dynamic window", bool(args.dynamic_window_from_causal)),
                    ("GARFIELD beam width", int(args.nsnp)),
                    ("GARFIELD max pick", int(args.max_pick)),
                    ("GARFIELD top-K validate", int(args.top_k_validate)),
                    ("GARFIELD val frac", float(args.val_frac)),
                    ("Hit top-K", int(args.top_k_hit)),
                    ("Hit mode", str(args.hit_mode)),
                    ("Hit LD r2", float(args.hit_ld_r2)),
                    (
                        "Threads",
                        format_requested_thread_usage(
                            requested_threads=int(requested_threads),
                            using_threads=int(args.thread),
                            detected_threads=int(detected_threads),
                        ),
                    ),
                ],
            )
        ],
        footer_rows=[("Output root", outprefix)],
        line_max_chars=60,
    )

    t0 = time.time()
    bench_root = Path(outprefix)
    bench_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for i in range(int(args.n_runs)):
        run_seed = int(args.seed) + i
        try:
            rec = _run_one(
                run_idx=i + 1,
                seed=run_seed,
                gfile=gfile,
                out_root=bench_root,
                args=args,
                logger=logger,
            )
            rec["status"] = "ok"
        except Exception as e:
            rec = {
                "run_index": int(i + 1),
                "seed": int(run_seed),
                "status": "failed",
                "error": str(e),
            }
            logger.exception(f"[seed{run_seed}] failed: {e}")
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    summary_path = bench_root / f"{prefix}.garfieldbench.summary.tsv"
    df.to_csv(summary_path, sep="\t", index=False)

    ok_df = df[df.get("status", pd.Series([], dtype=str)) == "ok"] if "status" in df.columns else pd.DataFrame()
    if len(ok_df) > 0:
        top1_all = float(np.mean(ok_df["top1_hit_all"].astype(float))) if "top1_hit_all" in ok_df.columns else float("nan")
        topk_all = float(np.mean(ok_df["topk_hit_all"].astype(float))) if "topk_hit_all" in ok_df.columns else float("nan")
        top1_any = float(np.mean(ok_df["top1_hit_any"].astype(float))) if "top1_hit_any" in ok_df.columns else float("nan")
        topk_any = float(np.mean(ok_df["topk_hit_any"].astype(float))) if "topk_hit_any" in ok_df.columns else float("nan")
        logger.info(
            f"Hit rates (n_ok={len(ok_df)}): "
            f"credit_top1_any={top1_any:.3f}, credit_top1_all={top1_all:.3f}, "
            f"credit_top{int(args.top_k_hit)}_any={topk_any:.3f}, "
            f"credit_top{int(args.top_k_hit)}_all={topk_all:.3f}"
        )
        if "exact_topk_hit_all" in ok_df.columns:
            exact_topk_all = float(np.mean(ok_df["exact_topk_hit_all"].astype(float)))
            logger.info(
                f"Exact recovery (top{int(args.top_k_hit)}_all)={exact_topk_all:.3f}; "
                "credit metrics are evaluated per window."
            )
    else:
        logger.warning("No successful runs; hit-rate summary skipped.")

    log_success(
        logger,
        f"Finished GARFIELDBENCH in {time.time() - t0:.2f}s. "
        f"Summary: {format_path_for_display(str(summary_path))}",
    )
    return 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
