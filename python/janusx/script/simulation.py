"""
JanusX Simulation CLI

Rust-first phenotype simulation from an existing genotype input.
Python is kept as the orchestration / CLI layer and optional plotting hook.
"""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
import logging
import os
import re
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


@dataclass(frozen=True)
class CausalSpec:
    count: int
    effect_model: Literal["equal", "geometric"]


@dataclass(frozen=True)
class GffSamplingSpec:
    path: str
    extension: int
    unit_sizes: tuple[int, ...]
    mode_label: str


DEFAULT_LOGIC_GFF_UNIT_GAP_BP = 50_000_000
DEFAULT_GFF_LOGIC_REDRAW_ATTEMPTS = 2_000


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


def _resolve_causal_spec(raw_args: Optional[list[str]]) -> CausalSpec:
    values = [] if raw_args is None else [str(x).strip() for x in raw_args if str(x).strip() != ""]
    if len(values) == 0:
        return CausalSpec(count=1, effect_model="equal")
    if len(values) > 2:
        raise ValueError("`-causal/--causal` accepts NUMBER [equal|geometric].")
    count = _parse_nonnegative_int(values[0], label="`-causal/--causal NUMBER`")
    effect_model = "equal" if len(values) == 1 else str(values[1]).strip().lower()
    if effect_model not in {"equal", "geometric"}:
        raise ValueError(
            "`-causal/--causal` effect model must be one of: equal, geometric."
        )
    return CausalSpec(count=int(count), effect_model=effect_model)  # type: ignore[arg-type]


def _resolve_gff_sampling_spec(
    raw_args: Optional[list[str]],
    *,
    default_extension: int = 100_000,
) -> Optional[GffSamplingSpec]:
    values = [] if raw_args is None else [str(x).strip() for x in raw_args if str(x).strip() != ""]
    if len(values) == 0:
        return None
    if len(values) > 3:
        raise ValueError("`-gff/--gff3` accepts GFFFILE [EXTENSION] [g1|g2|g3].")
    gff3_path = values[0]
    extension = int(default_extension)
    unit_sizes: tuple[int, ...] = (1, 2)
    mode_label = "g1/g2"
    saw_extension = False
    saw_mode = False
    for token in values[1:]:
        token_lc = str(token).strip().lower()
        if token_lc in {"g1", "g2", "g3"}:
            if saw_mode:
                raise ValueError("`-gff/--gff3` accepts at most one of g1/g2/g3.")
            unit_sizes = (int(token_lc[1:]),)
            mode_label = token_lc
            saw_mode = True
            continue
        if saw_extension:
            raise ValueError(
                "`-gff/--gff3` accepts at most one EXTENSION plus one optional g1/g2/g3 mode."
            )
        extension = _parse_nonnegative_int(token, label="`-gff/--gff3 EXTENSION`")
        saw_extension = True
    return GffSamplingSpec(
        path=gff3_path,
        extension=int(extension),
        unit_sizes=tuple(sorted(set(int(x) for x in unit_sizes if int(x) > 0))),
        mode_label=mode_label,
    )


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


def _filter_gene_catalog_by_min_active_sites(
    gene_catalog: list[tuple[str, tuple[str, int, int]]],
    *,
    active_positions: dict[str, list[int]],
    min_active_sites: int,
) -> list[tuple[str, tuple[str, int, int]]]:
    if int(min_active_sites) <= 1:
        return list(gene_catalog)
    out: list[tuple[str, tuple[str, int, int]]] = []
    for gene, (chrom, start, end) in gene_catalog:
        if (
            _count_active_sites_in_interval(
                chrom,
                int(start),
                int(end),
                active_positions=active_positions,
            )
            >= int(min_active_sites)
        ):
            out.append((gene, (chrom, int(start), int(end))))
    return out


def _count_active_sites_in_interval(
    chrom: str,
    start: int,
    end: int,
    *,
    active_positions: dict[str, list[int]],
) -> int:
    pos_vec = active_positions.get(_normalize_bim_chrom(chrom), [])
    if len(pos_vec) == 0:
        return 0
    lo = bisect.bisect_left(pos_vec, int(start))
    hi = bisect.bisect_right(pos_vec, int(end))
    return max(0, int(hi - lo))


def _count_active_sites_in_intervals(
    intervals: list[tuple[str, int, int]],
    *,
    active_positions: dict[str, list[int]],
) -> int:
    merged: dict[str, list[tuple[int, int]]] = {}
    for chrom, start, end in intervals:
        chrom_norm = _normalize_bim_chrom(chrom)
        merged.setdefault(chrom_norm, []).append((int(start), int(end)))
    total = 0
    for chrom_norm, spans in merged.items():
        spans_sorted = sorted(spans, key=lambda x: (int(x[0]), int(x[1])))
        collapsed: list[tuple[int, int]] = []
        for start, end in spans_sorted:
            if len(collapsed) == 0 or int(start) > int(collapsed[-1][1]):
                collapsed.append((int(start), int(end)))
            else:
                collapsed[-1] = (collapsed[-1][0], max(int(collapsed[-1][1]), int(end)))
        pos_vec = active_positions.get(chrom_norm, [])
        if len(pos_vec) == 0:
            continue
        for start, end in collapsed:
            lo = bisect.bisect_left(pos_vec, int(start))
            hi = bisect.bisect_right(pos_vec, int(end))
            total += max(0, int(hi - lo))
    return int(total)


def _gff_logic_unit_min_active_sites(
    logic_mode: Optional[str],
    logic_size_weights: Optional[list[float]],
    logic_k_min: int,
) -> int:
    if logic_mode is None or str(logic_mode).strip() == "":
        return 1
    weights = [] if logic_size_weights is None else [float(x) for x in logic_size_weights]
    size1_positive = len(weights) >= 1 and float(weights[0]) > 0.0
    if size1_positive:
        return 1
    return max(1, int(logic_k_min))


def _merge_unit_intervals_by_chrom(
    intervals: list[tuple[str, int, int]],
) -> dict[str, list[tuple[int, int]]]:
    merged: dict[str, list[tuple[int, int]]] = {}
    for chrom, start, end in intervals:
        chrom_norm = _normalize_bim_chrom(chrom)
        merged.setdefault(chrom_norm, []).append((int(start), int(end)))
    out: dict[str, list[tuple[int, int]]] = {}
    for chrom_norm, spans in merged.items():
        spans_sorted = sorted(spans, key=lambda x: (int(x[0]), int(x[1])))
        collapsed: list[tuple[int, int]] = []
        for start, end in spans_sorted:
            if len(collapsed) == 0 or int(start) > int(collapsed[-1][1]):
                collapsed.append((int(start), int(end)))
            else:
                collapsed[-1] = (collapsed[-1][0], max(int(collapsed[-1][1]), int(end)))
        out[chrom_norm] = collapsed
    return out


def _unit_is_physically_isolated(
    intervals: list[tuple[str, int, int]],
    *,
    existing_units: list[dict[str, Any]],
    min_unit_gap_bp: int,
) -> bool:
    if int(min_unit_gap_bp) <= 0 or len(existing_units) == 0:
        return True
    lhs_by_chrom = _merge_unit_intervals_by_chrom(intervals)
    for unit in existing_units:
        rhs_by_chrom = _merge_unit_intervals_by_chrom(list(unit["intervals"]))
        for chrom_norm, lhs_spans in lhs_by_chrom.items():
            rhs_spans = rhs_by_chrom.get(chrom_norm, [])
            if len(rhs_spans) == 0:
                continue
            for lhs_start, lhs_end in lhs_spans:
                for rhs_start, rhs_end in rhs_spans:
                    separated = (
                        int(lhs_end) + int(min_unit_gap_bp) < int(rhs_start)
                        or int(rhs_end) + int(min_unit_gap_bp) < int(lhs_start)
                    )
                    if not separated:
                        return False
    return True


def _draw_single_causal_gene_unit(
    *,
    gene_catalog: list[tuple[str, tuple[str, int, int]]],
    rng: np.random.Generator,
    used_genes: set[str],
    feasible_sizes: list[int],
    active_positions: Optional[dict[str, list[int]]],
    min_unit_active_sites: int,
    blocked_unit_names: set[str],
    existing_units: list[dict[str, Any]],
    min_unit_gap_bp: int,
    inner_budget: int,
    gene_sampling_weights: Optional[dict[str, float]] = None,
) -> Optional[dict[str, Any]]:
    remaining = [item for item in gene_catalog if str(item[0]) not in used_genes]
    if len(remaining) == 0 or len(feasible_sizes) == 0:
        return None
    attempts = int(max(1, inner_budget))
    while attempts > 0:
        attempts -= 1
        unit_size = int(feasible_sizes[int(rng.integers(0, len(feasible_sizes)))])
        weight_map = {} if gene_sampling_weights is None else gene_sampling_weights
        weight_vec = np.asarray(
            [max(0.0, float(weight_map.get(str(gene), 1.0))) for gene, _iv in remaining],
            dtype=float,
        )
        positive_count = int(np.count_nonzero(weight_vec > 0.0))
        if positive_count >= int(unit_size) and float(weight_vec.sum()) > 0.0:
            prob = weight_vec / float(weight_vec.sum())
            pick_idx = np.sort(
                rng.choice(len(remaining), size=unit_size, replace=False, p=prob)
            ).tolist()
        else:
            pick_idx = np.sort(
                rng.choice(len(remaining), size=unit_size, replace=False)
            ).tolist()
        chosen = sorted((remaining[i] for i in pick_idx), key=lambda x: x[0])
        genes = [str(gene) for gene, _iv in chosen]
        intervals = [iv for _gene, iv in chosen]
        unit_name = "|".join(genes)
        if unit_name in blocked_unit_names:
            continue
        if int(min_unit_active_sites) > 1:
            if active_positions is None:
                raise ValueError(
                    "Internal error: active_positions is required when min_unit_active_sites > 1."
                )
            if (
                _count_active_sites_in_intervals(
                    intervals,
                    active_positions=active_positions,
                )
                < int(min_unit_active_sites)
            ):
                continue
        if not _unit_is_physically_isolated(
            intervals,
            existing_units=existing_units,
            min_unit_gap_bp=int(min_unit_gap_bp),
        ):
            continue
        return {
            "unit_kind": "geneset" if len(genes) > 1 else "gene",
            "genes": genes,
            "unit_name": unit_name,
            "intervals": intervals,
        }
    return None


def _sample_causal_gene_units(
    gene_catalog: list[tuple[str, tuple[str, int, int]]],
    *,
    causal_count: int,
    seed: int,
    unit_sizes: tuple[int, ...] = (1, 2),
    active_positions: Optional[dict[str, list[int]]] = None,
    min_unit_active_sites: int = 1,
    blocked_unit_names: Optional[set[str]] = None,
    min_unit_gap_bp: int = 0,
    gene_sampling_weights: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    if int(causal_count) <= 0:
        return []
    allowed_sizes = tuple(sorted(set(int(x) for x in unit_sizes if int(x) > 0)))
    if len(allowed_sizes) == 0:
        raise ValueError("Internal error: GFF unit_sizes must contain at least one positive size.")
    min_unit_size = int(min(allowed_sizes))
    if len(gene_catalog) < int(causal_count) * int(min_unit_size):
        raise ValueError(
            "Not enough genes in GFF3 to sample the requested number of causal units without "
            "replacement: "
            f"genes={len(gene_catalog)}, causal={int(causal_count)}, "
            f"min_unit_size={int(min_unit_size)}."
        )
    rng = np.random.default_rng(int(seed) ^ 0x5EED_91A7)
    blocked = set() if blocked_unit_names is None else {str(x) for x in blocked_unit_names}
    used_genes: set[str] = set()
    units: list[dict[str, Any]] = []
    draw_budget = max(512, int(causal_count) * 64)
    stalled_draws = 0
    while len(units) < int(causal_count):
        remaining = [item for item in gene_catalog if str(item[0]) not in used_genes]
        units_left = int(causal_count) - len(units)
        genes_left = len(remaining)
        if genes_left < units_left * int(min_unit_size):
            raise ValueError(
                "Not enough unused genes remain to finish GFF causal-unit sampling after "
                f"filtering / redraw: remaining_genes={genes_left}, units_left={units_left}, "
                f"blocked_units={len(blocked)}, min_unit_size={int(min_unit_size)}."
            )
        accepted = False
        feasible_sizes = [
            int(size)
            for size in allowed_sizes
            if int(size) <= genes_left
            and (genes_left - int(size)) >= (units_left - 1) * int(min_unit_size)
        ]
        if len(feasible_sizes) == 0:
            raise ValueError(
                "No feasible GFF unit size remains under the current without-replacement "
                f"constraints: remaining_genes={genes_left}, units_left={units_left}, "
                f"allowed_sizes={','.join(str(x) for x in allowed_sizes)}."
            )
        inner_budget = max(64, len(remaining) * 8)
        stalled_draws += 1
        candidate = _draw_single_causal_gene_unit(
            gene_catalog=gene_catalog,
            rng=rng,
            used_genes=used_genes,
            feasible_sizes=feasible_sizes,
            active_positions=active_positions,
            min_unit_active_sites=int(min_unit_active_sites),
            blocked_unit_names=blocked,
            existing_units=units,
            min_unit_gap_bp=int(min_unit_gap_bp),
            inner_budget=int(inner_budget),
            gene_sampling_weights=gene_sampling_weights,
        )
        if candidate is not None:
            used_genes.update(str(g) for g in candidate["genes"])
            candidate["unit_index"] = len(units) + 1
            units.append(candidate)
            accepted = True
            stalled_draws = 0
        if accepted:
            continue
        if stalled_draws >= draw_budget:
            raise ValueError(
                "Unable to sample enough valid GFF causal units after redraw filtering: "
                f"requested={int(causal_count)}, built={len(units)}, blocked_units={len(blocked)}, "
                f"min_unit_active_sites={int(min_unit_active_sites)}, min_unit_gap_bp={int(min_unit_gap_bp)}."
            )
        raise ValueError(
            "Unable to sample a valid GFF causal unit from remaining genes under the current "
            f"constraints: remaining_genes={genes_left}, min_unit_active_sites={int(min_unit_active_sites)}, "
            f"min_unit_gap_bp={int(min_unit_gap_bp)}, blocked_units={len(blocked)}."
        )
    return units


def _extract_failed_gff_logic_unit_index(message: str) -> Optional[int]:
    text = str(message)
    patterns = (
        r"unable to realize a benchmarkable logic gate for term (\d+)",
        r"unable to build a valid logic gate for term (\d+)",
        r"unable to build a valid logic gate for pool (\d+)",
        r"causal_group\[(\d+)\] has no eligible sites after QC",
        r"causal constraint group\[(\d+)\]",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is not None:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def _format_unit_intervals(intervals: list[tuple[str, int, int]]) -> str:
    return ";".join(f"{chrom}:{start}:{end}" for chrom, start, end in intervals)


def _write_causal_units_txt(
    *,
    outprefix: str,
    units: list[dict[str, Any]],
) -> str:
    units_path = f"{outprefix}.causal.units.txt"
    with open(units_path, "w", encoding="utf-8") as fh:
        for unit in units:
            fh.write("\t".join(str(g) for g in unit["genes"]) + "\n")
    return units_path


def _write_fixed_effects_table(
    *,
    outprefix: str,
    fixed_rows: list[tuple[int, str, str, str, str, float]],
    units: Optional[list[dict[str, Any]]] = None,
) -> str:
    fixed_path = f"{outprefix}.fixed.effects.tsv"
    unit_rows = [] if units is None else list(units)
    if len(unit_rows) not in {0, len(fixed_rows)}:
        raise ValueError(
            "Causal unit count does not match simulated causal term count: "
            f"units={len(unit_rows)}, terms={len(fixed_rows)}."
        )
    with open(fixed_path, "w", encoding="utf-8") as fh:
        fh.write("unit_kind\tunit_name\tkind\tsites\tlabel\teffect\n")
        for row_idx, row in enumerate(fixed_rows):
            term_id, _term_kind, logic, site_text, label, effect = row
            if row_idx < len(unit_rows):
                unit = unit_rows[row_idx]
                unit_kind = str(unit["unit_kind"])
                unit_name = str(unit["unit_name"])
            else:
                unit_kind = "term"
                unit_name = str(label).strip() if str(label).strip() != "" else str(site_text)
            kind = "s" if str(logic).strip().lower() in {"", "single"} else str(logic).strip().lower()
            label_text = str(label).strip() if str(label).strip() != "" else str(site_text)
            fh.write(
                "\t".join(
                    [
                        unit_kind,
                        unit_name,
                        kind,
                        str(site_text),
                        label_text,
                        f"{float(effect):.10f}",
                    ]
                )
                + "\n"
            )
    return fixed_path


def _remove_optional_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


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
    *,
    causal_count: int,
) -> tuple[Optional[str], Optional[list[float]], int, int, Optional[int]]:
    if args.logic_gate is not None:
        if int(causal_count) <= 0:
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
    causal_effect_model: str,
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
    logic_delta: float,
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
    grm_cache_key = None
    if grm is not None:
        grm_arr = np.asarray(grm)
        grm_ptr = int(grm_arr.__array_interface__.get("data", (0, False))[0])
        grm_cache_key = None if grm_ptr == 0 else int(grm_ptr)
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
            causal_effect_model=str(causal_effect_model),
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
            logic_delta=float(logic_delta),
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
            grm_cache_key=grm_cache_key,
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
    logic_delta: float = 1e-6,
    and_target_pve: float = 0.2,
    and_max_iter: int = 100,
    causal_effect_model: Literal["equal", "geometric"] = "equal",
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
        causal_effect_model=str(causal_effect_model),
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
        logic_delta=float(logic_delta),
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


_SIM_SECTION_RULE = "-" * 72


def _simulation_section(logger: logging.Logger, title: str) -> None:
    logger.info("")
    logger.info(_SIM_SECTION_RULE)
    logger.info("[ %s ]", str(title).strip())


def _simulation_stage_view(
    stage: str,
    done: int,
    total: int,
    *,
    progress_total_hint: int,
    logic_mode: Optional[str],
) -> tuple[str, int, int, dict[str, str]]:
    stage_key = str(stage).strip().lower()
    total_now = max(0, int(total))
    done_now = max(0, int(done))
    first_pass_total = (
        min(int(progress_total_hint), total_now)
        if (int(progress_total_hint) > 0 and total_now > 0)
        else total_now
    )
    if stage_key == "background":
        stage_total = first_pass_total
        stage_done = min(done_now, stage_total) if stage_total > 0 else done_now
        return (
            "Loading eligible genotype variants",
            int(stage_done),
            int(stage_total),
            {"sites": f"{int(stage_done):,}/{int(stage_total):,}"} if stage_total > 0 else {},
        )
    if stage_key == "grm_factor":
        stage_total = max(1, total_now)
        stage_done = min(done_now, stage_total)
        return (
            "Preparing GRM sampling factor",
            int(stage_done),
            int(stage_total),
            {"step": f"{int(stage_done):,}/{int(stage_total):,}"},
        )
    if stage_key in {"causal_additive", "causal_logic"}:
        stage_total = max(0, total_now - first_pass_total)
        stage_done = max(0, done_now - first_pass_total)
        stage_done = min(stage_done, stage_total) if stage_total > 0 else stage_done
        desc = (
            "Sampling causal sites"
            if stage_key == "causal_additive" and (logic_mode is None or str(logic_mode).strip() == "")
            else "Sampling causal terms"
        )
        return (
            desc,
            int(stage_done),
            int(stage_total),
            {"sites": f"{int(stage_done):,}/{int(stage_total):,}"} if stage_total > 0 else {},
        )
    if stage_key == "finalize":
        stage_total = 1
        stage_done = 1 if done_now > 0 else 0
        return (
            "Finalizing phenotype outputs",
            int(stage_done),
            int(stage_total),
            {"step": f"{int(stage_done):,}/{int(stage_total):,}"},
        )
    stage_total = max(1, total_now)
    stage_done = min(done_now, stage_total)
    return (
        "Simulation",
        int(stage_done),
        int(stage_total),
        {"step": f"{int(stage_done):,}/{int(stage_total):,}"},
    )


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
        prog="jx simulation",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx simulation -bfile geno_prefix -o out -prefix demo",
                "jx simulation -vcf geno.vcf.gz -causal 3 -cs-pve 0.15 -o out",
                "jx simulation -bfile geno_prefix -logic-gate r 3,1,0.5 -causal 100 -bg-pve 0.4 -o out",
                "jx simulation -bfile geno_prefix -k panel.grm.npy -o out -prefix demo",
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
        nargs="+",
        metavar=("NUMBER", "MODEL"),
        default=["1"],
        help=(
            "Causal term specification: NUMBER, with an optional equal or geometric effect model. "
            "NUMBER is the total number "
            "of causal terms. Without --logic-gate these are additive single-site terms. With "
            "--logic-gate, each term size is sampled from the supplied weight vector; size 1 "
            "denotes a single-site additive term. Effect allocation defaults to equal."
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
        "-logic-delta",
        "--logic-delta",
        type=float,
        default=1e-6,
        help=(
            "Minimum realized score margin required for a simulated logic gate over its best "
            "parent literal. Gates with realized delta below this threshold are redrawn. "
            "Default: %(default)g."
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
        metavar=("GFFFILE", "EXTENSION"),
        default=None,
        help=(
            "Sample causal gene/gene-set units from GFF3. Use `-gff GFFFILE`, optionally "
            "followed by EXTENSION and an optional g1/g2/g3 mode in either order. Default mode "
            "mixes g1/g2, matching the current random single-gene / two-gene strategy. "
            "g1 forces single-gene units; g2 and g3 force exact two-gene or three-gene units."
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
        causal_spec = _resolve_causal_spec(args.causal)
        logic_mode, logic_size_weights, logic_k_min, logic_k_max, logic_window_bp = (
            _resolve_logic_config(args, causal_count=int(causal_spec.count))
        )
        gff_spec = _resolve_gff_sampling_spec(args.gff3)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc
    causal_count = int(causal_spec.count)
    causal_effect_model = str(causal_spec.effect_model)
    gff3_path = None if gff_spec is None else str(gff_spec.path)
    gff_extension = None if gff_spec is None else int(gff_spec.extension)
    gff_mode_label = None if gff_spec is None else str(gff_spec.mode_label)
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
    logic_delta = float(args.logic_delta)
    logic_max_iter = 256
    logic_effect_model = "gate"
    logic_has_multi_site_terms = logic_mode is not None and logic_size_weights is not None and any(
        float(weight) > 0.0 for weight in logic_size_weights[1:]
    )
    gff_logic_min_unit_gap_bp = (
        int(DEFAULT_LOGIC_GFF_UNIT_GAP_BP) if logic_has_multi_site_terms else 0
    )
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
                    ("Causal GFF mode", gff_mode_label),
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
                    ("Causal count", causal_count),
                    ("Causal effect model", causal_effect_model),
                    ("Causal PVE", cs_pve),
                    ("Logic gate", "None" if logic_mode is None else logic_mode),
                    ("Logic sizes", logic_enabled_sizes),
                    ("Logic size weights", logic_weight_text),
                    ("Logic realized delta", logic_delta),
                    ("Logic unit gap bp", gff_logic_min_unit_gap_bp),
                    ("Logic window bp", logic_window_bp),
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
    if not np.isfinite(logic_delta) or float(logic_delta) < 0.0:
        logger.error("--logic-delta must be finite and >= 0.")
        raise SystemExit(1)
    if int(causal_count) < len(bimranges):
        logger.error(
            "--causal must be >= number of --bimrange constraints. "
            "Got causal=%d and bimranges=%d.",
            int(causal_count),
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
    filtered_gene_catalog: list[tuple[str, tuple[str, int, int]]] = []
    active_pos_index: dict[str, list[int]] | None = None
    gene_sampling_weights: dict[str, float] | None = None
    gff_logic_min_unit_sites = 1
    selected_unit_sizes = (1, 2) if gff_spec is None else tuple(gff_spec.unit_sizes)

    t_start = time.time()
    _simulation_section(logger, "Genotype")
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

    aligned_grm = None
    _simulation_section(logger, "Background GRM")
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
            "Using external GRM: %s",
            format_path_for_display(str(args.grm)),
        )
        if resolved_grm_id is not None:
            logger.info("  GRM ID: %s", format_path_for_display(str(resolved_grm_id)))
        else:
            logger.info("  Sample alignment: assumed to match genotype input.")
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
            "Using cached cGRM: %s (source=%s, threads=%d).",
            format_path_for_display(str(grm_cache_path or "[memory]")),
            format_path_for_display(str(grm_input)),
            int(detected_threads),
        )

    if gff3_path is not None:
        _simulation_section(logger, "Causal Units")
        if not os.path.exists(f"{gfile}.bim"):
            logger.error(
                "GFF-constrained simulation currently requires PLINK BIM metadata; expected %s.bim.",
                gfile,
            )
            raise SystemExit(1)
        try:
            gene_catalog = _load_gene_catalog_from_gff(str(gff3_path), int(gff_extension or 0))
            with CliStatus("Preparing causal GFF unit candidates...", enabled=True) as task:
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
                    "Preparing causal GFF unit candidates ...Finished "
                    f"(genes={len(filtered_gene_catalog)}/{len(gene_catalog)})"
                )
            gff_logic_min_unit_sites = _gff_logic_unit_min_active_sites(
                logic_mode,
                logic_size_weights,
                int(logic_k_min),
            )
            if active_pos_index is not None:
                if tuple(selected_unit_sizes) == (1,) and int(gff_logic_min_unit_sites) > 1:
                    filtered_gene_catalog = _filter_gene_catalog_by_min_active_sites(
                        filtered_gene_catalog,
                        active_positions=active_pos_index,
                        min_active_sites=int(gff_logic_min_unit_sites),
                    )
                gene_sampling_weights = {}
                for gene, (chrom, start, end) in filtered_gene_catalog:
                    active_n = _count_active_sites_in_interval(
                        chrom,
                        int(start),
                        int(end),
                        active_positions=active_pos_index,
                    )
                    if tuple(selected_unit_sizes) == (1,) and int(gff_logic_min_unit_sites) > 1:
                        weight = float(max(1, active_n * max(0, active_n - 1) // 2))
                    else:
                        weight = float(max(1, active_n))
                    gene_sampling_weights[str(gene)] = float(weight)
            selected_causal_units = _sample_causal_gene_units(
                filtered_gene_catalog,
                causal_count=int(causal_count),
                seed=seed,
                unit_sizes=selected_unit_sizes,
                active_positions=active_pos_index,
                min_unit_active_sites=int(gff_logic_min_unit_sites),
                min_unit_gap_bp=int(gff_logic_min_unit_gap_bp),
                gene_sampling_weights=gene_sampling_weights,
            )
        except ValueError as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc
        bimrange_groups = [list(unit["intervals"]) for unit in selected_causal_units]
        logger.info(
            "Selected %d causal GFF units from %s.",
            len(selected_causal_units),
            format_path_for_display(str(gff3_path)),
        )
        logger.info(
            "  ext=%d, mode=%s",
            int(gff_extension or 0),
            str(gff_mode_label or "g1/g2"),
        )
        if int(gff_logic_min_unit_gap_bp) > 0:
            logger.info("  isolated_unit_gap_bp=%d", int(gff_logic_min_unit_gap_bp))
    scan_passes = _estimate_simulation_scan_passes(
        causal_count=int(causal_count),
        cs_pve=cs_pve,
        bimranges=bimranges,
        logic_mode=logic_mode,
        logic_size_weights=logic_size_weights,
        logic_gate_count=logic_gate_count,
    )
    progress_total_hint = int(max(0, n_sites))
    blocked_gff_unit_names: set[str] = set()
    max_gff_logic_redraw_attempts = int(DEFAULT_GFF_LOGIC_REDRAW_ATTEMPTS)
    redraw_attempt = 0
    _simulation_section(logger, "Phenotype Simulation")
    logger.info(
        "Stage order: eligible variants -> GRM factorization -> %s -> finalize.",
        "causal terms" if logic_mode is not None else "causal sites",
    )
    while True:
        if len(selected_causal_units) > 0:
            bimrange_groups = [list(unit["intervals"]) for unit in selected_causal_units]
        sim_pbar = ProgressAdapter(
            total=max(1, progress_total_hint if progress_total_hint > 0 else scan_passes),
            desc="Preparing phenotype simulation",
            emit_done=False,
            force_animate=True,
        )
        progress_state = {
            "raw_stage": None,
            "desc": "Preparing phenotype simulation",
            "last_done": 0,
            "stage_total": max(1, progress_total_hint if progress_total_hint > 0 else scan_passes),
        }

        def _simulation_progress(stage: str, done: int, total: int) -> None:
            desc, stage_done, stage_total, postfix = _simulation_stage_view(
                stage,
                int(done),
                int(total),
                progress_total_hint=int(progress_total_hint),
                logic_mode=logic_mode,
            )
            stage_total_norm = max(1, int(stage_total))
            if str(stage) != progress_state["raw_stage"]:
                progress_state["raw_stage"] = str(stage)
                progress_state["desc"] = str(desc)
                progress_state["last_done"] = 0
                progress_state["stage_total"] = stage_total_norm
                sim_pbar.set_desc(str(desc))
                sim_pbar.set_total(stage_total_norm)
            elif int(progress_state["stage_total"]) != stage_total_norm:
                progress_state["stage_total"] = stage_total_norm
                sim_pbar.set_total(stage_total_norm)
            delta = int(stage_done) - int(progress_state["last_done"])
            if delta > 0:
                sim_pbar.update(delta)
                progress_state["last_done"] = int(stage_done)
            sim_pbar.set_postfix(**postfix)

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
                causal=int(causal_count),
                causal_effect_model=causal_effect_model,
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
                logic_delta=float(logic_delta),
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
            sim_elapsed = max(0.0, time.monotonic() - sim_start)
            break
        except RuntimeError as exc:
            failed_unit_idx = _extract_failed_gff_logic_unit_index(str(exc))
            has_failed_gff_logic_unit = (
                gff3_path is not None
                and logic_mode is not None
                and len(filtered_gene_catalog) > 0
                and len(selected_causal_units) == int(causal_count)
                and failed_unit_idx is not None
                and 0 <= int(failed_unit_idx) < len(selected_causal_units)
            )
            can_redraw_gff_unit = (
                has_failed_gff_logic_unit
                and redraw_attempt < int(max_gff_logic_redraw_attempts)
            )
            if not can_redraw_gff_unit:
                if has_failed_gff_logic_unit:
                    raise RuntimeError(
                        f"{exc} [GFF redraw budget exhausted: attempts={redraw_attempt}/{max_gff_logic_redraw_attempts}, "
                        f"blocked_units={len(blocked_gff_unit_names)}]"
                    ) from exc
                raise
            failed_unit = selected_causal_units[int(failed_unit_idx)]
            blocked_gff_unit_names.add(str(failed_unit["unit_name"]))
            redraw_attempt += 1
            locked_units = [
                dict(unit)
                for unit_idx, unit in enumerate(selected_causal_units)
                if int(unit_idx) != int(failed_unit_idx)
            ]
            locked_genes = {
                str(gene)
                for unit in locked_units
                for gene in list(unit["genes"])
            }
            remaining_gene_count = sum(
                1
                for gene, _iv in filtered_gene_catalog
                if str(gene) not in locked_genes
            )
            feasible_sizes = [
                int(size) for size in selected_unit_sizes if int(size) <= int(remaining_gene_count)
            ]
            replacement_rng = np.random.default_rng(
                (int(seed) + int(redraw_attempt) * 1009) ^ 0x5EED_91A7
            )
            replacement = _draw_single_causal_gene_unit(
                gene_catalog=filtered_gene_catalog,
                rng=replacement_rng,
                used_genes=locked_genes,
                feasible_sizes=feasible_sizes,
                active_positions=active_pos_index,
                min_unit_active_sites=int(gff_logic_min_unit_sites),
                blocked_unit_names=blocked_gff_unit_names,
                existing_units=locked_units,
                min_unit_gap_bp=int(gff_logic_min_unit_gap_bp),
                inner_budget=max(128, len(filtered_gene_catalog) * 4),
                gene_sampling_weights=gene_sampling_weights,
            )
            if replacement is None:
                raise RuntimeError(
                    "unable to replace failed causal GFF unit under the current isolation / QC "
                    f"constraints: failed_unit={failed_unit['unit_name']}, blocked_units={len(blocked_gff_unit_names)}, "
                    f"min_unit_gap_bp={int(gff_logic_min_unit_gap_bp)}"
                ) from exc
            replacement["unit_index"] = int(failed_unit_idx) + 1
            selected_causal_units[int(failed_unit_idx)] = replacement
            continue
        finally:
            sim_pbar.finish()
            sim_pbar.close()

    log_success(logger, f"Simulation ...Finished [{format_elapsed(sim_elapsed)}]")
    if redraw_attempt > 0:
        logger.info(
            "Causal-unit redraws: retried %d times; skipped %d incompatible GFF unit signatures.",
            int(redraw_attempt),
            len(blocked_gff_unit_names),
        )
    random_effects_tsv = f"{outprefix}.random.effects.tsv"
    random_effects_pdf = f"{outprefix}.random.effects.pdf"
    try:
        _simulation_section(logger, "Visualization")
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

    realized_summary = res.get("realized_summary")
    _simulation_section(logger, "Summary")
    logger.info(
        "Targets: bg_pve=%s, causal_pve=%s, residual=%s.",
        res.get("bg_pve"),
        res.get("causal_pve"),
        res.get("ve"),
    )
    background_factorization = str(res.get("background_factorization", "none")).strip().lower()
    if background_factorization not in {"", "none"}:
        logger.info("GRM factorization: %s.", background_factorization)
    if isinstance(realized_summary, dict):
        logger.info(
            "Phenotype: mean(y)=%.6g, var(y)=%.6g.",
            float(realized_summary.get("mean_y", 0.0)),
            float(realized_summary.get("var_y", 0.0)),
        )
        logger.info(
            "Components (mean): c=%.6g, u=%.6g, e=%.6g.",
            float(realized_summary.get("mean_causal", 0.0)),
            float(realized_summary.get("mean_background", 0.0)),
            float(realized_summary.get("mean_residual", 0.0)),
        )
        logger.info(
            "Components (var):  c=%.6g, u=%.6g, e=%.6g.",
            float(realized_summary.get("var_causal", 0.0)),
            float(realized_summary.get("var_background", 0.0)),
            float(realized_summary.get("var_residual", 0.0)),
        )
        logger.info(
            "Components (share): cs=%.6g, bg=%.6g, res=%.6g, sum=%.6g.",
            float(realized_summary.get("pve_causal", 0.0)),
            float(realized_summary.get("pve_background", 0.0)),
            float(realized_summary.get("pve_residual", 0.0)),
            float(realized_summary.get("pve_causal", 0.0))
            + float(realized_summary.get("pve_background", 0.0))
            + float(realized_summary.get("pve_residual", 0.0)),
        )
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
    _write_fixed_effects_table(
        outprefix=outprefix,
        fixed_rows=fixed_rows,
        units=(selected_causal_units if len(selected_causal_units) > 0 else None),
    )
    truth_path = f"{outprefix}.causal.unit_truth.tsv"
    _remove_optional_file(truth_path)
    if len(selected_causal_units) > 0:
        units_path = _write_causal_units_txt(
            outprefix=outprefix,
            units=selected_causal_units,
        )
        logger.info(
            "Causal units: %s",
            format_path_for_display(str(units_path)),
        )
    if len(sample_ids) != int(np.asarray(res["phenotype"]).reshape(-1).shape[0]):
        logger.warning("Sample count from inspection differs from Rust phenotype length.")

    _simulation_section(logger, "Outputs")
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
