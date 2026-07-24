"""
JanusX Simulation CLI

Rust-first phenotype simulation from an existing genotype input.
Python is kept as the orchestration / CLI layer and optional plotting hook.
"""

from __future__ import annotations

import argparse
import bisect
import logging
import os
import socket
import time
from datetime import datetime
from typing import Any, Callable, Literal, Optional

import numpy as np

from janusx.assoc.workflow import load_or_build_grm_with_cache
from janusx.assoc.workflow_cache import _gwas_cache_prefix_with_params
from janusx.gfreader import (
    inspect_genotype_file,
    prepare_bed_logic_keep_mask_pure_line,
    prepare_cli_input_cache,
)
from janusx.gtools.reader import readanno
from janusx.janusx import g2p_simulate

from ._common.cli_args import (
    add_common_genotype_source_args,
    add_common_grm_file_arg,
    add_common_out_arg,
    add_common_prefix_arg,
    add_common_variant_filter_args,
)
from ._common.config_render import emit_cli_configuration
from ._common.genocache import configure_genotype_cache_from_out
from ._common.genoio import determine_genotype_source_from_args as determine_genotype_source
from ._common.grmio import load_and_align_grm
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
    format_path_for_display,
)
from ._common.progress import CliStatus, ProgressAdapter, format_elapsed, log_success, stdout_is_tty
from ._common.threads import detect_effective_threads


def _parse_bimrange(text: str) -> tuple[str, int, int]:
    raw = str(text).strip()
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid bimrange '{raw}'. Expected format like chr1:1000:2000."
        )
    chrom = str(parts[0]).strip()
    start = int(parts[1])
    end = int(parts[2])
    if chrom == "":
        raise ValueError(f"Invalid bimrange '{raw}': chromosome is empty.")
    if end < start:
        raise ValueError(f"Invalid bimrange '{raw}': end < start.")
    return chrom, start, end


def _parse_nonnegative_int(text: str, *, label: str) -> int:
    try:
        value = int(str(text).strip())
    except Exception as exc:
        raise ValueError(f"{label} must be an integer, got: {text}") from exc
    if value < 0:
        raise ValueError(f"{label} must be >= 0.")
    return value


def _resolve_gff_sampling_spec(
    raw_args: Optional[list[str]],
    *,
    default_extension: int = 100_000,
) -> tuple[Optional[str], Optional[int]]:
    values = [] if raw_args is None else [str(x).strip() for x in raw_args if str(x).strip() != ""]
    if len(values) == 0:
        return None, None
    if len(values) > 2:
        raise ValueError("`-gff/--gff3` accepts GFF3 [EXT].")
    gff3_path = values[0]
    extension = (
        int(default_extension)
        if len(values) == 1
        else _parse_nonnegative_int(values[1], label="`-gff/--gff3 EXT`")
    )
    return gff3_path, int(extension)


def _load_gene_catalog_from_gff(
    gff3_path: str,
    extension: int,
) -> list[tuple[str, tuple[str, int, int]]]:
    dfgff3 = readanno(str(gff3_path), "ID").iloc[:, :4].set_index(3)
    dfgff3 = dfgff3.loc[~dfgff3.index.duplicated()]
    out: list[tuple[str, tuple[str, int, int]]] = []
    for gene_id, row in dfgff3.iterrows():
        gene = str(gene_id).strip()
        if gene == "" or gene.lower() == "nan":
            continue
        chrom = str(row[0]).strip()
        if chrom == "" or chrom.lower() == "nan":
            continue
        start = int(row[1]) - int(extension)
        end = int(row[2]) + int(extension)
        out.append((gene, (chrom, int(start), int(end))))
    if len(out) == 0:
        raise ValueError(f"No valid gene intervals were parsed from GFF3: {gff3_path}")
    return out


def _normalize_bim_chrom(chrom: object) -> str:
    text = str(chrom).strip()
    if len(text) >= 3 and text[:3].lower() == "chr":
        text = text[3:].strip()
    if len(text) > 2 and (text.endswith("_1") or text.endswith("_2")):
        text = text[:-2]
    if len(text) > 1 and (text.endswith("-") or text.endswith("+")):
        text = text[:-1]
    return text.upper()


def _prepare_simulation_site_keep(
    *,
    bed_prefix: str,
    n_samples: int,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    threads: int,
) -> np.ndarray:
    sample_indices = np.arange(int(n_samples), dtype=np.int64)
    site_keep_raw, _n_samples_seen, n_total_sites = prepare_bed_logic_keep_mask_pure_line(
        str(bed_prefix),
        sample_indices=sample_indices,
        maf_threshold=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
        het_threshold=float(het_threshold),
        snps_only=False,
        mmap_window_mb=None,
        threads=max(1, int(threads)),
    )
    site_keep = np.ascontiguousarray(
        np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1),
        dtype=np.bool_,
    )
    if int(site_keep.shape[0]) != int(n_total_sites):
        raise ValueError(
            "Simulation site_keep length mismatch: "
            f"mask={int(site_keep.shape[0])}, total={int(n_total_sites)}."
        )
    if int(np.count_nonzero(site_keep)) <= 0:
        raise ValueError("No SNPs remain after applying simulation QC filters.")
    return site_keep


def _build_active_bim_position_index(
    *,
    bed_prefix: str,
    site_keep: np.ndarray,
) -> dict[str, list[int]]:
    bim_path = f"{bed_prefix}.bim"
    if not os.path.exists(bim_path):
        raise ValueError(f"GFF-constrained simulation requires BIM metadata: {bim_path}")
    keep_mask = np.asarray(site_keep, dtype=np.bool_).reshape(-1)
    out: dict[str, list[int]] = {}
    with open(bim_path, "r", encoding="utf-8") as fh:
        for row_idx, line in enumerate(fh):
            if row_idx >= int(keep_mask.shape[0]) or not bool(keep_mask[row_idx]):
                continue
            toks = line.rstrip("\n").split()
            if len(toks) < 4:
                continue
            chrom = _normalize_bim_chrom(toks[0])
            try:
                pos = int(toks[3])
            except Exception:
                continue
            out.setdefault(chrom, []).append(int(pos))
    for chrom in list(out.keys()):
        out[chrom].sort()
    return out


def _filter_gene_catalog_by_active_sites(
    gene_catalog: list[tuple[str, tuple[str, int, int]]],
    *,
    active_positions: dict[str, list[int]],
) -> list[tuple[str, tuple[str, int, int]]]:
    out: list[tuple[str, tuple[str, int, int]]] = []
    for gene, (chrom, start, end) in gene_catalog:
        pos_vec = active_positions.get(_normalize_bim_chrom(chrom), [])
        if len(pos_vec) == 0:
            continue
        lo = bisect.bisect_left(pos_vec, int(start))
        if lo < len(pos_vec) and int(pos_vec[lo]) <= int(end):
            out.append((gene, (chrom, int(start), int(end))))
    return out


def _sample_causal_gene_units(
    gene_catalog: list[tuple[str, tuple[str, int, int]]],
    *,
    causal_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if int(causal_count) <= 0:
        return []
    if len(gene_catalog) < int(causal_count):
        raise ValueError(
            "Not enough genes in GFF3 to sample the requested number of causal units "
            f"without replacement: genes={len(gene_catalog)}, causal={int(causal_count)}."
        )
    remaining = list(gene_catalog)
    rng = np.random.default_rng(int(seed) ^ 0x5EED_91A7)
    units: list[dict[str, Any]] = []
    while len(units) < int(causal_count):
        units_left = int(causal_count) - len(units)
        genes_left = len(remaining)
        feasible_sizes = [1]
        if genes_left - 2 >= (units_left - 1):
            feasible_sizes.append(2)
        unit_size = int(feasible_sizes[int(rng.integers(0, len(feasible_sizes)))])
        pick_idx = np.sort(rng.choice(len(remaining), size=unit_size, replace=False)).tolist()
        chosen = [remaining[i] for i in pick_idx]
        for i in reversed(pick_idx):
            remaining.pop(int(i))
        chosen = sorted(chosen, key=lambda x: x[0])
        genes = [gene for gene, _iv in chosen]
        intervals = [iv for _gene, iv in chosen]
        units.append(
            {
                "unit_index": len(units) + 1,
                "unit_kind": "geneset" if len(genes) > 1 else "gene",
                "genes": genes,
                "unit_name": "|".join(genes),
                "intervals": intervals,
            }
        )
    return units


def _format_unit_intervals(intervals: list[tuple[str, int, int]]) -> str:
    return ";".join(f"{chrom}:{start}:{end}" for chrom, start, end in intervals)


def _write_causal_unit_files(
    *,
    outprefix: str,
    units: list[dict[str, Any]],
    fixed_rows: list[tuple[int, str, str, str, str, float]],
) -> tuple[str, str]:
    units_path = f"{outprefix}.causal.units.txt"
    truth_path = f"{outprefix}.causal.unit_truth.tsv"
    with open(units_path, "w", encoding="utf-8") as fh:
        for unit in units:
            fh.write("\t".join(str(g) for g in unit["genes"]) + "\n")
    if len(fixed_rows) != len(units):
        raise ValueError(
            "Causal unit count does not match simulated causal term count: "
            f"units={len(units)}, terms={len(fixed_rows)}."
        )
    with open(truth_path, "w", encoding="utf-8") as fh:
        fh.write(
            "term_id\tunit_index\tunit_kind\tunit_name\tgenes\tintervals\tterm_kind\tlogic\tsites\tlabel\teffect\n"
        )
        for unit, row in zip(units, fixed_rows):
            term_id, term_kind, logic, site_text, label, effect = row
            fh.write(
                "\t".join(
                    [
                        str(term_id),
                        str(unit["unit_index"]),
                        str(unit["unit_kind"]),
                        str(unit["unit_name"]),
                        str("|".join(str(g) for g in unit["genes"])),
                        _format_unit_intervals(list(unit["intervals"])),
                        str(term_kind),
                        str(logic),
                        str(site_text),
                        str(label),
                        f"{float(effect):.10f}",
                    ]
                )
                + "\n"
            )
    return units_path, truth_path


_LOGIC_GATE_MODES = {"a", "na", "an", "nan", "r"}


def _parse_logic_size_weights(text: str) -> list[float]:
    raw = str(text).strip()
    if raw == "":
        raise ValueError(
            "Invalid logic size weights: expected a comma-separated list like '3,1,0.5'."
        )
    weights: list[float] = []
    for i, token in enumerate(raw.split(","), start=1):
        field = str(token).strip()
        if field == "":
            raise ValueError(f"Invalid logic size weights: empty entry at position {i}.")
        try:
            weight = float(field)
        except ValueError as exc:
            raise ValueError(
                f"Invalid logic size weights: '{field}' at position {i} is not a number."
            ) from exc
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"Invalid logic size weights: entry {i} must be finite and >= 0, got {field}."
            )
        weights.append(weight)
    if not any(weight > 0.0 for weight in weights):
        raise ValueError("Invalid logic size weights: at least one entry must be > 0.")
    return weights


def _weights_from_gate_size_range(k_min: int, k_max: int) -> list[float]:
    if int(k_min) <= 0:
        raise ValueError("k_min must be > 0 when building logic size weights.")
    if int(k_max) < int(k_min):
        raise ValueError("k_max must be >= k_min when building logic size weights.")
    weights = [0.0] * int(max(1, k_max))
    for size in range(int(k_min), int(k_max) + 1):
        weights[size - 1] = 1.0
    return weights


def _resolve_logic_config(
    args: argparse.Namespace,
) -> tuple[Optional[str], Optional[list[float]], int, int, Optional[int]]:
    if args.logic_gate is not None:
        if int(args.causal) <= 0:
            raise ValueError("`--causal` must be > 0 when `--logic-gate` is enabled.")
        if len(args.logic_gate) != 2:
            raise ValueError(
                "`--logic-gate` expects two arguments: MODE and WEIGHTS "
                "(for example: `--logic-gate r 3,1,0.5`)."
            )
        logic_mode = str(args.logic_gate[0]).strip().lower()
        if logic_mode not in _LOGIC_GATE_MODES:
            allowed = "/".join(sorted(_LOGIC_GATE_MODES))
            raise ValueError(f"`--logic-gate` MODE must be one of {allowed}.")
        logic_size_weights = _parse_logic_size_weights(args.logic_gate[1])
        gate_sizes = [i + 1 for i, weight in enumerate(logic_size_weights) if weight > 0.0 and i >= 1]
        logic_k_min = min(gate_sizes) if gate_sizes else 2
        logic_k_max = max(gate_sizes) if gate_sizes else 2
        return logic_mode, logic_size_weights, logic_k_min, logic_k_max, None
    return None, None, 2, 2, None


def _estimate_simulation_scan_passes(
    *,
    causal_count: int,
    cs_pve: Optional[float],
    bimranges: list[tuple[str, int, int]],
    logic_mode: Optional[str],
    logic_size_weights: Optional[list[float]],
    logic_gate_count: Optional[int],
) -> int:
    logic_requested = logic_mode is not None and str(logic_mode).strip() != ""
    base_term_count = (
        (
            int(causal_count)
            if logic_size_weights is not None
            else (int(logic_gate_count) if logic_gate_count is not None else max(1, int(causal_count)))
        )
        if logic_requested
        else int(causal_count)
    )
    effective_term_count = int(base_term_count)
    causal_pve_target = (
        float(cs_pve)
        if cs_pve is not None
        else (min(0.05 * effective_term_count, 0.95) if effective_term_count > 0 else 0.0)
    )
    needs_causal_scan = effective_term_count > 0 and causal_pve_target > 0.0
    return 1 + int(needs_causal_scan)


def _align_square_matrix_to_ids(
    matrix: np.ndarray,
    source_ids: Optional[list[str] | np.ndarray],
    target_ids: list[str] | np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    target = [str(x) for x in target_ids]
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{label} must be square, got shape={arr.shape}")
    if source_ids is None:
        if arr.shape[0] != len(target):
            raise ValueError(
                f"{label} shape {arr.shape} does not match target sample count {len(target)}."
            )
        return np.asarray(arr, dtype=np.float64, order="C")

    source = [str(x) for x in source_ids]
    if len(source) != arr.shape[0]:
        raise ValueError(
            f"{label} ID count mismatch: matrix n={arr.shape[0]} but ids={len(source)}."
        )
    if source == target:
        return np.asarray(arr, dtype=np.float64, order="C")

    index = {sid: i for i, sid in enumerate(source)}
    missing = [sid for sid in target if sid not in index]
    if missing:
        preview = ", ".join(missing[:5])
        extra = "" if len(missing) <= 5 else f" ... (+{len(missing) - 5} more)"
        raise ValueError(f"{label} is missing target sample IDs: {preview}{extra}")
    order = np.asarray([index[sid] for sid in target], dtype=np.intp)
    return np.asarray(arr[np.ix_(order, order)], dtype=np.float64, order="C")


def _load_or_build_background_grm_auto(
    *,
    gfile: str,
    sample_ids: np.ndarray,
    n_sites: int,
    maf: float,
    geno: float,
    het: float,
    out_dir: str | None,
    logger: logging.Logger | None,
    prefer_plink_source: bool,
    threads: int,
) -> tuple[np.ndarray, Optional[str], str]:
    logger_use = logger if logger is not None else logging.getLogger(__name__)
    grm_input = str(gfile)
    if not bool(prefer_plink_source):
        delim = "," if str(grm_input).lower().endswith(".csv") else None
        grm_input = str(
            prepare_cli_input_cache(
                str(grm_input),
                snps_only=False,
                delimiter=delim,
                prefer_plink_for_txt=True,
                threads=int(max(1, int(threads))),
            )
        )
    cache_prefix = _gwas_cache_prefix_with_params(
        str(gfile),
        maf=float(maf),
        geno=float(geno),
        snps_only=False,
        cache_dir=(None if out_dir is None else str(out_dir)),
        logger=logger_use,
    )
    grm_all, _eff_m, grm_ids, grm_cache_path = load_or_build_grm_with_cache(
        genofile=str(grm_input),
        cache_prefix=cache_prefix,
        mgrm="1",
        maf_threshold=float(maf),
        max_missing_rate=float(geno),
        het_threshold=float(het),
        chunk_size=65536,
        memory_mb=1024.0,
        threads=int(max(1, int(threads))),
        logger=logger_use,
        use_spinner=bool(stdout_is_tty()),
        ids_preloaded=np.asarray(sample_ids, dtype=str),
        n_snps_preloaded=int(n_sites),
        snps_only=False,
        allow_packed_full_load=True,
    )
    aligned = _align_square_matrix_to_ids(
        np.asarray(grm_all, dtype=np.float64),
        grm_ids,
        sample_ids,
        label="Simulation background GRM",
    )
    return aligned, grm_cache_path, grm_input


def _run_rust_simulation(
    *,
    gfile: str,
    seed: int,
    maf: float,
    causal_maf_min: float,
    missing_rate: float,
    het_threshold: float | None,
    bg_pve: float,
    residual_var: float,
    causal: int,
    cs_pve: Optional[float],
    bimranges: list[tuple[str, int, int]],
    bimrange_groups: Optional[list[list[tuple[str, int, int]]]],
    logic_mode: Optional[str],
    logic_size_weights: Optional[list[float]],
    logic_gate_count: Optional[int],
    logic_k_min: int,
    logic_k_max: int,
    logic_ld_max: float,
    logic_het_max: float,
    logic_af_min: float,
    logic_af_max: float,
    logic_max_iter: int,
    logic_window_bp: Optional[int],
    logic_effect_model: str,
    background_dist: str,
    gamma_shape: float,
    gamma_scale: float,
    laplace_scale: float,
    outprefix: Optional[str] = None,
    trait_name: Optional[str] = None,
    write_effect_tables: bool = False,
    grm: np.ndarray | None = None,
    snps_only: bool = False,
    progress_callback: Any | None = None,
    progress_total_hint: Optional[int] = None,
    progress_every: int = 10_000,
) -> dict[str, Any]:
    # Keep passing `residual_var` for API compatibility. Rust derives the
    # residual variance target as `1 - bg_pve - causal_pve` under the
    # final-variance PVE definition and samples background / residual terms on
    # the expectation scale from that target.
    fixed_path = f"{outprefix}.fixed.effects.tsv" if (outprefix and write_effect_tables) else None
    random_path = (
        f"{outprefix}.random.effects.tsv" if (outprefix and write_effect_tables) else None
    )
    return dict(
        g2p_simulate(
            gfile,
            chunk_size=100_000,
            maf_threshold=float(maf),
            causal_maf_min=float(causal_maf_min),
            max_missing_rate=float(missing_rate),
            het_threshold=None if het_threshold is None else float(het_threshold),
            seed=int(seed),
            residual_var=float(residual_var),
            bg_pve=float(bg_pve),
            background_dist=str(background_dist),
            gamma_shape=float(gamma_shape),
            gamma_scale=float(gamma_scale),
            laplace_scale=float(laplace_scale),
            causal_count=int(max(0, causal)),
            causal_pve=None if cs_pve is None else float(cs_pve),
            bim_ranges=list(bimranges),
            bim_range_groups=(
                None
                if bimrange_groups is None
                else [
                    [(str(chrom), int(start), int(end)) for chrom, start, end in group]
                    for group in bimrange_groups
                ]
            ),
            logic_mode=logic_mode,
            logic_size_weights=(
                None if logic_size_weights is None else [float(x) for x in logic_size_weights]
            ),
            logic_gate_count=None if logic_gate_count is None else int(logic_gate_count),
            logic_k_min=int(logic_k_min),
            logic_k_max=int(logic_k_max),
            logic_ld_max=float(logic_ld_max),
            logic_het_max=float(logic_het_max),
            logic_af_min=float(logic_af_min),
            logic_af_max=float(logic_af_max),
            logic_max_iter=int(logic_max_iter),
            logic_window_bp=logic_window_bp,
            logic_effect_model=str(logic_effect_model),
            delimiter=None,
            snps_only=bool(snps_only),
            pheno_prefix=outprefix,
            fixed_effects_path=fixed_path,
            random_effects_path=random_path,
            causal_sites_path=None,
            trait_name=trait_name,
            na_rate=0.1,
            grm=grm,
            progress_callback=progress_callback,
            progress_total_hint=(
                None if progress_total_hint is None else int(max(0, progress_total_hint))
            ),
            progress_every=int(max(1, progress_every)),
        )
    )


def simulate_phenotype_from_genofile(
    gfile: str,
    mode: Literal["single", "garfield"] = "single",
    chunk_size: int = 100_000,
    seed: int = 1,
    maf: float = 0.02,
    missing_rate: float = 0.05,
    het: float | None = None,
    pve: float = 0.5,
    ve: float = 1.0,
    windows: int = 50_000,
    and_k_min: int = 2,
    and_k_max: int = 4,
    and_ld_max: float = 0.2,
    and_het_max: float = 0.05,
    and_af_min: float = 0.02,
    and_af_max: float = 0.98,
    and_target_pve: float = 0.2,
    and_max_iter: int = 100,
    logic_effect_model: Literal["gate", "centered_interaction"] = "gate",
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    logic_mode = "a" if str(mode).lower() == "garfield" else None
    logic_size_weights = (
        _weights_from_gate_size_range(int(and_k_min), int(and_k_max))
        if logic_mode is not None
        else None
    )
    grm = None
    if float(pve) > 0.0:
        sample_ids, n_sites = inspect_genotype_file(
            gfile,
            snps_only=False,
            maf=float(maf),
            missing_rate=float(missing_rate),
            het=1.0 if het is None else float(het),
        )
        sample_ids = np.asarray(sample_ids, dtype=str)
        prefer_plink_source = bool(
            str(gfile).lower().endswith(".bed")
            or os.path.exists(f"{gfile}.bed")
        )
        grm, _grm_cache_path, _grm_input = _load_or_build_background_grm_auto(
            gfile=str(gfile),
            sample_ids=sample_ids,
            n_sites=int(n_sites),
            maf=float(maf),
            geno=float(missing_rate),
            het=1.0 if het is None else float(het),
            out_dir=None,
            logger=None,
            prefer_plink_source=prefer_plink_source,
            threads=int(detect_effective_threads()),
        )
    res = _run_rust_simulation(
        gfile=gfile,
        seed=int(seed),
        maf=float(maf),
        causal_maf_min=float(maf),
        missing_rate=float(missing_rate),
        het_threshold=None if het is None else float(het),
        bg_pve=float(pve),
        residual_var=float(ve),
        causal=1,
        cs_pve=float(and_target_pve) if logic_mode is not None else None,
        bimranges=[],
        bimrange_groups=None,
        logic_mode=logic_mode,
        logic_size_weights=logic_size_weights,
        logic_gate_count=None,
        logic_k_min=int(and_k_min),
        logic_k_max=int(and_k_max),
        logic_ld_max=float(and_ld_max),
        logic_het_max=float(and_het_max),
        logic_af_min=float(and_af_min),
        logic_af_max=float(and_af_max),
        logic_max_iter=int(and_max_iter),
        logic_window_bp=int(windows) if logic_mode is not None else None,
        logic_effect_model=str(logic_effect_model),
        background_dist="normal",
        gamma_shape=1.0,
        gamma_scale=1.0,
        laplace_scale=1.0,
        outprefix=None,
        trait_name=None,
        write_effect_tables=False,
        grm=grm,
        snps_only=False,
    )
    y = np.asarray(res["phenotype"], dtype=np.float64).reshape(-1, 1)
    outsites = [(str(c), int(s), int(e)) for c, s, e in list(res.get("causal_sites", []))]
    return y, outsites


def write_phenotypes(outprefix: str, sample_ids: np.ndarray, y: np.ndarray, seed: int = 1):
    sample_ids = np.asarray(sample_ids, dtype=str).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)

    pheno3 = np.empty((len(sample_ids), 3), dtype=object)
    pheno3[:, 0] = sample_ids
    pheno3[:, 1] = sample_ids
    pheno3[:, 2] = yv
    np.savetxt(
        f"{outprefix}.pheno",
        pheno3,
        delimiter="\t",
        fmt=["%s", "%s", "%.6f"],
    )

    pheno2 = np.column_stack([sample_ids, yv.astype(object)])
    np.savetxt(
        f"{outprefix}.pheno.txt",
        pheno2,
        delimiter="\t",
        fmt=["%s", "%.6f"],
        header="IID\tPHENO",
        comments="",
    )

    rng = np.random.default_rng(int(seed))
    pheno2_na = pheno2.astype(object, copy=True)
    k = int(round(len(sample_ids) * 0.1))
    if k > 0:
        idx = rng.choice(len(sample_ids), size=k, replace=False)
        pheno2_na[idx, 1] = "NA"
    np.savetxt(
        f"{outprefix}.pheno.NA.txt",
        pheno2_na,
        delimiter="\t",
        fmt=["%s", "%s"],
        header="IID\tPHENO",
        comments="",
    )


def write_sites(outprefix: str, sites: list[tuple[str, int, int]]):
    if not sites:
        return
    arr = np.asarray(sites, dtype=object)
    np.savetxt(f"{outprefix}.causal.sites.tsv", arr, delimiter="\t", fmt=["%s", "%d", "%d"])


def _histogram_edges(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return np.asarray([-0.5, 0.5], dtype=np.float64)
    lo = float(np.min(data))
    hi = float(np.max(data))
    if data.size == 1 or np.isclose(lo, hi):
        pad = max(1e-6, abs(lo) * 0.1 + 1e-6)
        return np.asarray([lo - pad, hi + pad], dtype=np.float64)
    try:
        edges = np.histogram_bin_edges(data, bins="fd")
    except ValueError:
        edges = np.histogram_bin_edges(data, bins="auto")
    if np.asarray(edges).size < 2:
        pad = max(1e-6, (hi - lo) * 0.1)
        return np.asarray([lo - pad, hi + pad], dtype=np.float64)
    return np.asarray(edges, dtype=np.float64)


def _background_effect_label(background_source: str) -> str:
    low = str(background_source).strip().lower()
    if "grm" in low or "kernel" in low or "sample" in low:
        return "breeding values"
    return "background effects"


def _background_effect_axis_label(background_source: str) -> str:
    low = str(background_source).strip().lower()
    if "grm" in low or "kernel" in low or "sample" in low:
        return "Breeding value"
    return "Effect"


def _plot_random_effect_distribution(
    *,
    effects_tsv: str,
    out_pdf: str,
    trait_name: str,
    background_dist: str,
    background_source: str = "sample_kernel",
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> None:
    phase_total = 5
    source_low = str(background_source).strip().lower()
    is_sample_space = ("grm" in source_low) or ("kernel" in source_low) or ("sample" in source_low)

    def _report(phase: str, step: int) -> None:
        if progress_callback is not None:
            progress_callback(str(phase), int(step), phase_total)

    if not os.path.exists(effects_tsv):
        raise FileNotFoundError(f"Random effects table not found: {effects_tsv}")

    effects = np.atleast_1d(
        np.loadtxt(
            effects_tsv,
            delimiter="\t",
            skiprows=1,
            usecols=[4],
            dtype=np.float64,
        )
    )
    effects = np.asarray(effects, dtype=np.float64)
    effects = effects[np.isfinite(effects)]
    if effects.size == 0:
        raise ValueError("No finite random effects found for plotting.")
    _report("load-table", 1)

    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    import matplotlib.pyplot as plt
    _report("init-plotting", 2)

    fig, ax_hist = plt.subplots(1, 1, figsize=(7.4, 4.8))
    fig.patch.set_facecolor("white")

    edges = _histogram_edges(effects)
    curve_xs = None
    curve_ys = None
    curve_color = "#C44E52"
    curve_label = None
    if is_sample_space:
        _report("fit-normal-curve", 2)
        sigma = float(np.std(effects))
        if effects.size >= 2 and sigma > 1e-12:
            mu = float(np.mean(effects))
            curve_xs = np.linspace(float(edges[0]), float(edges[-1]), 256)
            z = (curve_xs - mu) / sigma
            curve_ys = np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))
            curve_label = "normal fit"
        _report("fit-normal-curve", 3)
    else:
        _report("fit-kde-curve", 2)
        try:
            from scipy.stats import gaussian_kde
        except Exception:
            gaussian_kde = None
        if gaussian_kde is not None and effects.size >= 8 and float(np.std(effects)) > 1e-12:
            curve_xs = np.linspace(float(edges[0]), float(edges[-1]), 256)
            kde = gaussian_kde(effects)
            curve_ys = kde(curve_xs)
            curve_label = "kde"
        _report("fit-kde-curve", 3)

    ax_hist.hist(
        effects,
        bins=edges,
        density=True,
        color="#4C78A8",
        alpha=0.80,
        edgecolor="white",
        linewidth=0.8,
    )
    ax_hist.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.85)

    if curve_xs is not None and curve_ys is not None:
        ax_hist.plot(curve_xs, curve_ys, color=curve_color, linewidth=1.8, label=curve_label)

    ax_hist.set_title(f"{trait_name} {_background_effect_label(background_source)}", fontsize=12)
    ax_hist.set_xlabel(_background_effect_axis_label(background_source))
    ax_hist.set_ylabel("Density")
    ax_hist.grid(axis="y", color="#D1D5DB", alpha=0.55, linewidth=0.8)
    ax_hist.text(
        0.98,
        0.97,
        (
            f"gaussian sample-space, entries={effects.size:,}"
            if is_sample_space
            else f"{background_dist}, entries={effects.size:,}"
        ),
        transform=ax_hist.transAxes,
        va="top",
        ha="right",
        fontsize=9.2,
        color="#374151",
    )
    if curve_label is not None:
        ax_hist.legend(loc="upper left", frameon=False, fontsize=9.0)
    _report("render-figure", 4)

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    _report("write-pdf", 5)


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx sim",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx sim -bfile geno_prefix -o out -prefix demo",
                "jx sim -vcf geno.vcf.gz -causal 3 -cs-pve 0.15 -o out",
                "jx sim -bfile geno_prefix -logic-gate r 3,1,0.5 -causal 100 -bg-pve 0.4 -o out",
                "jx sim -bfile geno_prefix -k panel.grm.npy -o out -prefix demo",
            ]
        ),
        description="JanusX simulation: phenotype from existing genotype (Rust-first)",
    )

    required_group = parser.add_argument_group("Required arguments")
    optional_group = parser.add_argument_group("Optional arguments")
    filter_group = parser.add_argument_group("Genotype filtering")
    pve_group = parser.add_argument_group("Variance / PVE model")
    causal_group = parser.add_argument_group("Causal terms")

    geno_group = required_group.add_mutually_exclusive_group(required=True)
    add_common_genotype_source_args(geno_group, include_file=True, help_profile="default")

    add_common_out_arg(optional_group, default=".")
    add_common_prefix_arg(optional_group, default=None)
    add_common_grm_file_arg(
        optional_group,
        default=None,
        dest="grm",
        help_profile="background_kernel",
    )
    optional_group.add_argument("--seed", type=int, default=None, help="Random seed. If omitted, use current time.")

    filter_group.add_argument(
        "-chunksize",
        "--chunksize",
        type=int,
        default=100_000,
        help="Compatibility placeholder; simulation core streams in Rust (default: 100,000).",
    )
    add_common_variant_filter_args(
        filter_group,
        help_profile="simulation",
        include_maf=True,
        include_geno=True,
        include_het=True,
        maf_default=0.02,
        geno_default=0.05,
        het_default=None,
    )

    pve_group.add_argument(
        "-bg-pve",
        "--bg-pve",
        dest="bg_pve",
        type=float,
        default=0.5,
        help=(
            "Background/polygenic variance contribution Var(u_bg) in the final phenotype. "
            "Together with --cs-pve, this determines effective residual variance as "
            "1 - bg_pve - cs_pve. Default: %(default)s."
        ),
    )
    causal_group.add_argument(
        "-causal",
        "--causal",
        type=int,
        default=1,
        help=(
            "Total number of causal terms. Without --logic-gate these are additive single-site "
            "terms. With --logic-gate, each term size is sampled from the supplied weight vector; "
            "size 1 denotes a single-site additive term."
        ),
    )
    causal_group.add_argument(
        "-cs-pve",
        "--cs-pve",
        type=float,
        default=None,
        help=(
            "Overall causal variance contribution Var(Qγ) in the final phenotype. "
            "If omitted, Rust uses Garfield default: min(0.05 * number_of_terms, cap)."
        ),
    )
    causal_group.add_argument(
        "-lmaf",
        "--lmaf",
        type=float,
        default=None,
        help=(
            "Minimum MAF for selected causal terms. For additive terms this filters chosen sites; "
            "for logic-gate terms it also enforces a minimum gate MAF. "
            "Defaults to --maf."
        ),
    )
    causal_group.add_argument(
        "-logic-gate",
        "--logic-gate",
        nargs=2,
        metavar=("MODE", "WEIGHTS"),
        default=None,
        help=(
            "Mixed causal-term sampler. MODE is one of a|na|an|nan|r. WEIGHTS is a comma list "
            "whose i-th entry controls the relative probability of sampling term size i "
            "(1=additive single-site, 2=two-site gate, 3=three-site gate, ...). "
            "Example: `--logic-gate r 3,1,0.5 --causal 100` samples 100 causal terms with "
            "sizes 1/2/3 in proportion to 3:1:0.5."
        ),
    )
    causal_group.add_argument(
        "--logic-effect-model",
        type=str,
        choices=["gate", "centered_interaction"],
        default="gate",
        help=(
            "Logic-term effect model. 'gate' keeps the current mean-centered gate effect; "
            "'centered_interaction' removes member main effects and builds a weak-marginal "
            "interaction benchmark."
        ),
    )
    causal_group.add_argument(
        "-bimrange",
        "--bimrange",
        action="append",
        default=[],
        help="Repeatable causal region: chr:start:end. Can be specified multiple times.",
    )
    causal_group.add_argument(
        "-gff",
        "--gff3",
        nargs="+",
        metavar="ARG",
        default=None,
        help=(
            "Sample causal gene/gene-set units from GFF3. Use `-gff GFF3 [EXT]`. "
            "When enabled, simulation draws --causal units without replacement; each unit "
            "contains one gene or a two-gene set, and every causal term is constrained to "
            "the selected unit window(s)."
        ),
    )
    # bg_group.add_argument(
    #     "-normal",
    #     "--normal",
    #     action="store_true",
    #     help="Use normal background effects g₀ᵢ ~ N(0,1). This is the default.",
    # )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    gfile, prefix = determine_genotype_source(args)
    args.out = os.path.normpath(args.out if args.out is not None else ".")
    outstem = str(args.prefix).strip() if args.prefix is not None else prefix
    outprefix = os.path.join(args.out, outstem)
    os.makedirs(args.out, exist_ok=True, mode=0o755)
    cache_dir = configure_genotype_cache_from_out(args.out)

    log_path = f"{outprefix}.sim.log"
    logger = setup_logging(log_path)

    seed = int(args.seed) if args.seed is not None else int(time.time()) & 0x7FFFFFFF
    bimranges = [_parse_bimrange(x) for x in list(args.bimrange or [])]
    try:
        logic_mode, logic_size_weights, logic_k_min, logic_k_max, logic_window_bp = (
            _resolve_logic_config(args)
        )
        gff3_path, gff_extension = _resolve_gff_sampling_spec(args.gff3)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc
    logic_gate_count = None
    cs_pve = float(args.cs_pve) if args.cs_pve is not None else None
    causal_maf_min = max(
        float(args.maf),
        float(args.lmaf) if args.lmaf is not None else float(args.maf),
    )
    logic_ld_max = 1.0
    logic_het_max = 1.0
    logic_af_min = 0.0
    logic_af_max = 1.0
    logic_max_iter = 256
    logic_effect_model = str(args.logic_effect_model).strip().lower()
    logic_enabled_sizes = (
        ",".join(str(i + 1) for i, weight in enumerate(logic_size_weights) if weight > 0.0)
        if logic_size_weights is not None
        else "None"
    )
    logic_weight_text = (
        ",".join(f"{float(weight):g}" for weight in logic_size_weights)
        if logic_size_weights is not None
        else "None"
    )

    emit_cli_configuration(
        logger,
        app_title="JanusX - Simulation",
        config_title="SIMULATION CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "General",
                [
                    ("Genotype file", gfile),
                    ("MAF threshold", args.maf),
                    ("Missing threshold", args.geno),
                    ("Het threshold", "None" if args.het is None else args.het),
                    ("Background PVE", args.bg_pve),
                    ("Background GRM", args.grm),
                    ("Causal lMAF", causal_maf_min),
                    ("Causal GFF", gff3_path),
                    ("Causal GFF ext", gff_extension),
                    (
                        "Background path",
                        (
                            "external GRM"
                            if args.grm
                            else (
                                "auto cached GRM"
                                if float(args.bg_pve) > 0.0
                                else "none (bg_pve=0)"
                            )
                        ),
                    ),
                    ("Causal count", args.causal),
                    ("Causal PVE", cs_pve),
                    ("Logic gate", "None" if logic_mode is None else logic_mode),
                    ("Logic sizes", logic_enabled_sizes),
                    ("Logic size weights", logic_weight_text),
                    ("Logic window bp", logic_window_bp),
                    ("Logic effect model", logic_effect_model if logic_mode is not None else "None"),
                    ("Background dist", "gaussian sample-space"),
                    ("Sampling scale", "expectation-scale"),
                    ("SNPs only", False),
                    ("BIM ranges", len(bimranges)),
                    ("Seed", seed),
                ],
            ),
        ],
        footer_rows=[("Output prefix", outprefix)],
        line_max_chars=60,
    )

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    if args.grm:
        checks.append(ensure_file_exists(logger, args.grm, "Background GRM"))
    if gff3_path is not None:
        checks.append(ensure_file_exists(logger, gff3_path, "Causal GFF3"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    if not (0.0 <= float(args.bg_pve) <= 1.0):
        logger.error("--bg-pve must be in [0, 1].")
        raise SystemExit(1)
    if args.het is not None and not (0.0 <= float(args.het) <= 1.0):
        logger.error("--het must be in [0, 1].")
        raise SystemExit(1)
    if args.lmaf is not None and not (0.0 <= float(args.lmaf) <= 0.5):
        logger.error("--lmaf must be in [0, 0.5].")
        raise SystemExit(1)
    if int(args.causal) < len(bimranges):
        logger.error(
            "--causal must be >= number of --bimrange constraints. "
            "Got causal=%d and bimranges=%d.",
            int(args.causal),
            len(bimranges),
        )
        raise SystemExit(1)
    if gff3_path is not None and len(bimranges) > 0:
        logger.error("--bimrange cannot be combined with -gff/--gff3 causal-unit sampling.")
        raise SystemExit(1)
    if cs_pve is not None and not (0.0 <= float(cs_pve) <= 1.0):
        logger.error("--cs-pve must be in [0, 1].")
        raise SystemExit(1)
    if cs_pve is not None and float(args.bg_pve) + float(cs_pve) > 1.0:
        logger.error(
            "--bg-pve + --cs-pve must be <= 1 under the final-variance PVE definition. "
            "Got bg_pve=%.6g and cs_pve=%.6g.",
            float(args.bg_pve),
            float(cs_pve),
        )
        raise SystemExit(1)

    selected_causal_units: list[dict[str, Any]] = []
    bimrange_groups: Optional[list[list[tuple[str, int, int]]]] = None

    t_start = time.time()
    logger.info(f"Simulating phenotype from genotype: {gfile}")
    with CliStatus("Inspecting genotype input...", enabled=True) as task:
        sample_ids, n_sites = inspect_genotype_file(
            gfile,
            snps_only=False,
            maf=float(args.maf),
            missing_rate=float(args.geno),
            het=1.0 if args.het is None else float(args.het),
        )
        task.complete("Inspecting genotype input ...Finished")
    sample_ids = np.asarray(sample_ids, dtype=str)
    detected_threads = int(detect_effective_threads())

    if gff3_path is not None:
        if not os.path.exists(f"{gfile}.bim"):
            logger.error(
                "GFF-constrained simulation currently requires PLINK BIM metadata; expected %s.bim.",
                gfile,
            )
            raise SystemExit(1)
        try:
            gene_catalog = _load_gene_catalog_from_gff(str(gff3_path), int(gff_extension or 0))
            with CliStatus("Computing active-site mask for causal GFF units...", enabled=True) as task:
                site_keep = _prepare_simulation_site_keep(
                    bed_prefix=str(gfile),
                    n_samples=int(sample_ids.shape[0]),
                    maf_threshold=float(args.maf),
                    max_missing_rate=float(args.geno),
                    het_threshold=1.0 if args.het is None else float(args.het),
                    threads=int(detected_threads),
                )
                active_pos_index = _build_active_bim_position_index(
                    bed_prefix=str(gfile),
                    site_keep=site_keep,
                )
                filtered_gene_catalog = _filter_gene_catalog_by_active_sites(
                    gene_catalog,
                    active_positions=active_pos_index,
                )
                task.complete(
                    "Computing active-site mask for causal GFF units ...Finished "
                    f"(genes={len(filtered_gene_catalog)}/{len(gene_catalog)})"
                )
            selected_causal_units = _sample_causal_gene_units(
                filtered_gene_catalog,
                causal_count=int(args.causal),
                seed=seed,
            )
        except ValueError as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc
        bimrange_groups = [list(unit["intervals"]) for unit in selected_causal_units]
        logger.info(
            "Sampled %d causal gene/gene-set units from %s after QC prefilter (ext=%d).",
            len(selected_causal_units),
            format_path_for_display(str(gff3_path)),
            int(gff_extension or 0),
        )

    aligned_grm = None
    if args.grm:
        with CliStatus("Loading background GRM...", enabled=True) as task:
            aligned_grm, resolved_grm_id = load_and_align_grm(
                str(args.grm),
                sample_ids.tolist(),
                grm_id_path=None,
                label="Background GRM",
            )
            task.complete("Loading background GRM ...Finished")
        logger.info(
            "Background GRM mode: using %s%s",
            format_path_for_display(str(args.grm)),
            (
                f" (ID: {format_path_for_display(str(resolved_grm_id))})"
                if resolved_grm_id is not None
                else " (sample order assumed to match genotype input)"
            ),
        )
    elif float(args.bg_pve) > 0.0:
        aligned_grm, grm_cache_path, grm_input = _load_or_build_background_grm_auto(
            gfile=str(gfile),
            sample_ids=sample_ids,
            n_sites=int(n_sites),
            maf=float(args.maf),
            geno=float(args.geno),
            het=1.0 if args.het is None else float(args.het),
            out_dir=str(cache_dir or args.out),
            logger=logger,
            prefer_plink_source=bool(args.bfile),
            threads=int(detected_threads),
        )
        logger.info(
            "Background GRM mode: using auto cached cGRM from %s (source=%s, threads=%d).",
            format_path_for_display(str(grm_cache_path or "[memory]")),
            format_path_for_display(str(grm_input)),
            int(detected_threads),
        )
    scan_passes = _estimate_simulation_scan_passes(
        causal_count=int(args.causal),
        cs_pve=cs_pve,
        bimranges=bimranges,
        logic_mode=logic_mode,
        logic_size_weights=logic_size_weights,
        logic_gate_count=logic_gate_count,
    )
    progress_total_hint = int(max(0, n_sites))
    progress_total = int(max(1, progress_total_hint * max(1, scan_passes)))
    sim_pbar = ProgressAdapter(
        total=progress_total,
        desc="Simulation work",
        emit_done=False,
        force_animate=True,
    )
    progress_state = {"last_done": 0, "stage": "background"}

    def _simulation_progress(stage: str, done: int, total: int) -> None:
        stage_key = str(stage)
        total_now = int(total)
        done_now = int(done)
        if stage_key != progress_state["stage"]:
            progress_state["stage"] = stage_key
        if total_now > 0:
            if int(getattr(sim_pbar, "total", 0)) != total_now:
                sim_pbar.set_total(total_now)
            done_now = max(0, min(done_now, total_now))
        delta = done_now - int(progress_state["last_done"])
        if delta > 0:
            sim_pbar.update(delta)
            progress_state["last_done"] = done_now
        first_pass_total = (
            min(progress_total_hint, total_now) if (progress_total_hint > 0 and total_now > 0) else 0
        )
        if stage_key == "background":
            stage_total_now = first_pass_total if first_pass_total > 0 else total_now
            stage_done_now = min(done_now, stage_total_now) if stage_total_now > 0 else done_now
        elif stage_key in ("causal_additive", "causal_logic"):
            stage_done_now = max(0, done_now - first_pass_total)
            stage_total_now = max(0, total_now - first_pass_total)
        elif stage_key == "finalize":
            stage_done_now = 1
            stage_total_now = 1
        else:
            stage_done_now = done_now
            stage_total_now = total_now
        if stage_total_now > 0:
            sim_pbar.set_postfix(
                sites=f"{stage_done_now:,}/{stage_total_now:,}",
            )
        else:
            sim_pbar.set_postfix(
                sites=f"{stage_done_now:,}",
            )

    sim_start = time.monotonic()
    try:
        res = _run_rust_simulation(
            gfile=gfile,
            seed=seed,
            maf=float(args.maf),
            causal_maf_min=float(causal_maf_min),
            missing_rate=float(args.geno),
            het_threshold=None if args.het is None else float(args.het),
            bg_pve=float(args.bg_pve),
            residual_var=1.0,
            causal=int(args.causal),
            cs_pve=cs_pve,
            bimranges=bimranges,
            bimrange_groups=bimrange_groups,
            logic_mode=logic_mode,
            logic_size_weights=logic_size_weights,
            logic_gate_count=logic_gate_count,
            logic_k_min=int(logic_k_min),
            logic_k_max=int(logic_k_max),
            logic_ld_max=float(logic_ld_max),
            logic_het_max=float(logic_het_max),
            logic_af_min=float(logic_af_min),
            logic_af_max=float(logic_af_max),
            logic_max_iter=int(logic_max_iter),
            logic_window_bp=logic_window_bp,
            logic_effect_model=logic_effect_model,
            background_dist="normal",
            gamma_shape=1.0,
            gamma_scale=1.0,
            laplace_scale=1.0,
            outprefix=outprefix,
            trait_name=None,
            write_effect_tables=True,
            grm=aligned_grm,
            snps_only=False,
            progress_callback=_simulation_progress,
            progress_total_hint=progress_total_hint,
            progress_every=max(1, min(10_000, progress_total_hint // 200 if progress_total_hint > 0 else 10_000)),
        )
    finally:
        sim_pbar.finish()
        sim_pbar.close()
    sim_elapsed = max(0.0, time.monotonic() - sim_start)
    log_success(logger, f"Simulation ...Finished [{format_elapsed(sim_elapsed)}]")
    random_effects_tsv = f"{outprefix}.random.effects.tsv"
    random_effects_pdf = f"{outprefix}.random.effects.pdf"
    try:
        vis_source = str(res.get("background_source", "none"))
        vis_label = (
            "GRM breeding values"
            if vis_source.strip().lower() == "grm"
            else _background_effect_label(vis_source)
        )
        vis_total = 5
        vis_pbar = ProgressAdapter(
            total=vis_total,
            desc=f"Visualizing {vis_label}",
            show_remaining=False,
            force_animate=True,
        )
        vis_phase_labels = {
            "load-table": "loading effects",
            "init-plotting": "initializing matplotlib",
            "fit-normal-curve": "fitting normal curve",
            "fit-kde-curve": "estimating density curve",
            "render-figure": "rendering figure",
            "write-pdf": "writing pdf",
        }

        def _visualization_progress(phase: str, done: int, total: int) -> None:
            total_now = max(1, int(total))
            done_now = max(0, min(int(done), total_now))
            vis_pbar.update(done_now - getattr(_visualization_progress, "last_done", 0))
            _visualization_progress.last_done = done_now
            vis_pbar.set_postfix(
                stage=vis_phase_labels.get(str(phase), str(phase)),
                step=f"{done_now}/{total_now}",
            )

        _visualization_progress.last_done = 0  # type: ignore[attr-defined]
        vis_pbar.set_postfix(stage="queued", step=f"0/{vis_total}")
        vis_success = False
        try:
            _plot_random_effect_distribution(
                effects_tsv=random_effects_tsv,
                out_pdf=random_effects_pdf,
                trait_name=str(res.get("trait_name", "PHENO")),
                background_dist="normal",
                background_source=str(res.get("background_source", "none")),
                progress_callback=_visualization_progress,
            )
            vis_success = True
        finally:
            if vis_success:
                vis_pbar.finish()
            vis_pbar.close()
    except Exception as exc:
        logger.warning("Random effect distribution PDF was skipped: %s", exc)

    logger.info(
        "Final-variance PVE runtime: bg_pve=%s, causal_pve=%s, ve=%s.",
        res.get("bg_pve"),
        res.get("causal_pve"),
        res.get("ve"),
    )
    realized_summary = res.get("realized_summary")
    if isinstance(realized_summary, dict):
        logger.info(
            "mean(y)=%.6g, var(y)=%.6g.",
            float(realized_summary.get("mean_y", 0.0)),
            float(realized_summary.get("var_y", 0.0)),
        )
    if len(selected_causal_units) > 0:
        fixed_rows = [
            (
                int(term_id),
                str(term_kind),
                str(logic),
                str(site_text),
                str(label),
                float(effect),
            )
            for term_id, term_kind, logic, site_text, label, effect in list(res.get("fixed_rows", []))
        ]
        units_path, truth_path = _write_causal_unit_files(
            outprefix=outprefix,
            units=selected_causal_units,
            fixed_rows=fixed_rows,
        )
        logger.info(
            "Saved causal unit file: %s",
            format_path_for_display(str(units_path)),
        )
        logger.info(
            "Saved causal truth file: %s",
            format_path_for_display(str(truth_path)),
        )
        logger.info(
            "mean(c)=%.6g, mean(u)=%.6g, mean(e)=%.6g.",
            float(realized_summary.get("mean_causal", 0.0)),
            float(realized_summary.get("mean_background", 0.0)),
            float(realized_summary.get("mean_residual", 0.0)),
        )
        logger.info(
            "var(c)=%.6g, var(u)=%.6g, var(e)=%.6g.",
            float(realized_summary.get("var_causal", 0.0)),
            float(realized_summary.get("var_background", 0.0)),
            float(realized_summary.get("var_residual", 0.0)),
        )
        logger.info(
            "cov(c,u)=%.6g, cov(c,e)=%.6g, cov(u,e)=%.6g.",
            float(realized_summary.get("cov_causal_background", 0.0)),
            float(realized_summary.get("cov_causal_residual", 0.0)),
            float(realized_summary.get("cov_background_residual", 0.0)),
        )
        logger.info(
            "cs=%.6g, bg=%.6g, res=%.6g, sum=%.6g.",
            float(realized_summary.get("pve_causal", 0.0)),
            float(realized_summary.get("pve_background", 0.0)),
            float(realized_summary.get("pve_residual", 0.0)),
            float(realized_summary.get("pve_causal", 0.0))
            + float(realized_summary.get("pve_background", 0.0))
            + float(realized_summary.get("pve_residual", 0.0)),
        )
    if len(sample_ids) != int(np.asarray(res["phenotype"]).reshape(-1).shape[0]):
        logger.warning("Sample count from inspection differs from Rust phenotype length.")

    logger.info("Outputs:")
    logger.info(f"  {format_path_for_display(f'{outprefix}.pheno')}")
    logger.info(f"  {format_path_for_display(f'{outprefix}.pheno.txt')}")
    logger.info(f"  {format_path_for_display(f'{outprefix}.pheno.NA.txt')}")
    logger.info(f"  {format_path_for_display(f'{outprefix}.fixed.effects.tsv')}")
    logger.info(f"  {format_path_for_display(f'{outprefix}.random.effects.tsv')}")
    if os.path.exists(random_effects_pdf):
        logger.info(f"  {format_path_for_display(random_effects_pdf)}")
    total_elapsed = max(0.0, time.time() - t_start)
    logger.info("Finished, total time: %.2f secs", total_elapsed)
    now = datetime.now()
    logger.info(
        f"{now.year}-{now.month}-{now.day} {now.hour:02d}:{now.minute:02d}:{now.second:02d}"
    )
    return 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
