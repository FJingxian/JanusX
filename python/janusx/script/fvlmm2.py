# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from janusx.assoc.workflow import (
    _align_pheno_to_sample_order,
    _inspect_genotype_with_status,
    _load_covariates_for_models,
    _load_phenotype_with_status,
    _resolve_trait_iter,
    _trait_values_and_mask,
    load_or_build_grm_with_cache,
)
from janusx.assoc.workflow_cache import _gwas_cache_prefix_with_params
from janusx.pyBLUP.assoc import FvLMM
from janusx.script._common.cli_args import (
    add_common_covariate_file_or_site_arg,
    add_common_genotype_source_args,
    add_common_grm_file_arg,
    add_common_out_arg,
    add_common_pheno_arg,
    add_common_prefix_arg,
    add_common_snps_only_arg,
    add_common_thread_arg,
    add_common_trait_selector_args,
    add_common_variant_filter_args,
    parse_trait_selector_specs,
)
from janusx.script._common.cli_core import (
    CliArgumentParser,
    cli_help_formatter,
    minimal_help_epilog,
)
from janusx.script._common.config_render import emit_cli_configuration
from janusx.script._common.genoio import (
    determine_genotype_source,
    prepare_packed_ctx_from_plink,
)
from janusx.script._common.genocache import configure_genotype_cache_from_out
from janusx.script._common.grmio import load_and_align_grm
from janusx.script._common.log import setup_logging
from janusx.script._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_file_input_site_metadata_exists,
    ensure_plink_prefix_exists,
    format_path_for_display,
)
from janusx.script._common.progress import stdout_is_tty
from janusx.script._common.threads import (
    apply_outer_thread_cap,
    detect_effective_threads,
    format_requested_thread_usage,
)

try:
    from janusx import janusx as jxrs
except Exception:
    jxrs = None


_INTERACTION_RE = re.compile(r"^\s*([^\s&|*]+)\s*([&|*])\s*([^\s&|*]+)\s*$")


@dataclass(frozen=True)
class ActiveSite:
    local_row: int
    source_row: int
    chrom: str
    pos: int
    snp: str
    allele0: str
    allele1: str


@dataclass(frozen=True)
class InteractionSpec:
    expr: str
    snp1: str
    op: str
    snp2: str
    row1: int
    row2: int
    neg1: bool = False
    neg2: bool = False


def _require_rust_backend() -> None:
    if jxrs is None:
        raise RuntimeError(
            "JanusX Rust extension is unavailable. Please rebuild/install the extension first."
        )
    for name in (
        "bed_packed_decode_rows_f32",
        "fvlmm2_assoc_chunk_f32",
        "load_bim_columns",
    ):
        if not hasattr(jxrs, name):
            raise RuntimeError(
                f"Rust extension is missing `{name}`. Please rebuild JanusX so fvlmm2 kernels are exported."
            )


def _normalize_snp_name(raw_name: object, chrom: object, pos: object) -> str:
    name = str(raw_name).strip()
    if name == "" or name == "." or name.lower() == "nan":
        return f"{str(chrom).strip()}_{int(pos)}"
    return name


def _load_active_sites(prefix: str, packed_ctx: dict[str, object]) -> tuple[list[ActiveSite], dict[str, list[int]]]:
    active_row_idx = np.ascontiguousarray(
        np.asarray(
            packed_ctx.get("active_row_idx", packed_ctx.get("row_indices")),
            dtype=np.int64,
        ).reshape(-1),
        dtype=np.int64,
    )
    row_flip = np.ascontiguousarray(
        np.asarray(packed_ctx["row_flip"], dtype=np.bool_).reshape(-1),
        dtype=np.bool_,
    )
    chrom, pos, snp, allele0, allele1 = jxrs.load_bim_columns(str(prefix), active_row_idx)
    chrom_col = [str(v) for v in list(chrom)]
    pos_col = [int(v) for v in list(pos)]
    snp_col = list(snp)
    allele0_col = [str(v) for v in list(allele0)]
    allele1_col = [str(v) for v in list(allele1)]
    if not (
        len(chrom_col)
        == len(pos_col)
        == len(snp_col)
        == len(allele0_col)
        == len(allele1_col)
        == int(active_row_idx.shape[0])
        == int(row_flip.shape[0])
    ):
        raise ValueError("Active site metadata length mismatch in fvlmm2 loader.")

    sites: list[ActiveSite] = []
    name_map: dict[str, list[int]] = {}
    for local_row in range(int(active_row_idx.shape[0])):
        snp_name = _normalize_snp_name(snp_col[local_row], chrom_col[local_row], pos_col[local_row])
        a0 = allele0_col[local_row]
        a1 = allele1_col[local_row]
        if bool(row_flip[local_row]):
            a0, a1 = a1, a0
        site = ActiveSite(
            local_row=int(local_row),
            source_row=int(active_row_idx[local_row]),
            chrom=str(chrom_col[local_row]),
            pos=int(pos_col[local_row]),
            snp=str(snp_name),
            allele0=str(a0),
            allele1=str(a1),
        )
        sites.append(site)
        for key in {site.snp, f"{site.chrom}_{site.pos}"}:
            if key == "":
                continue
            name_map.setdefault(key, []).append(site.local_row)
    return sites, name_map


def _resolve_snp_token(token: str, name_map: dict[str, list[int]]) -> int:
    hits = name_map.get(str(token).strip(), [])
    if len(hits) == 1:
        return int(hits[0])
    if len(hits) > 1:
        raise ValueError(
            f"SNP token '{token}' is ambiguous after filtering; matched {len(hits)} active rows."
        )
    raise KeyError(f"SNP token '{token}' was not found among active filtered variants.")


def _split_literal_token(token: str) -> tuple[str, bool]:
    text = str(token).strip()
    if text == "":
        raise ValueError("Literal token is empty.")
    negated = False
    while text.startswith("!"):
        negated = not negated
        text = text[1:].strip()
    if text == "":
        raise ValueError("Literal token has no SNP name after '!'.")
    return text, bool(negated)


def _literal_display_name(snp: str, negated: bool) -> str:
    base = str(snp).strip()
    return f"!{base}" if bool(negated) else base


def _literal_alleles(site: ActiveSite, negated: bool) -> tuple[str, str]:
    if bool(negated):
        return (str(site.allele1), str(site.allele0))
    return (str(site.allele0), str(site.allele1))


def _parse_interaction_file(
    path: str,
    name_map: dict[str, list[int]],
) -> tuple[list[InteractionSpec], list[dict[str, str]]]:
    specs: list[InteractionSpec] = []
    skipped: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = str(raw).strip()
            if line == "" or line.startswith("#"):
                continue
            token = line.split()[0]
            m = _INTERACTION_RE.match(token)
            if m is None:
                skipped.append(
                    {
                        "line": str(line_no),
                        "expr": token,
                        "reason": "invalid_expression",
                    }
                )
                continue
            token1 = str(m.group(1)).strip()
            op = str(m.group(2)).strip()
            token2 = str(m.group(3)).strip()
            try:
                snp1, neg1 = _split_literal_token(token1)
                snp2, neg2 = _split_literal_token(token2)
            except Exception as ex:
                skipped.append(
                    {
                        "line": str(line_no),
                        "expr": f"{token1}{op}{token2}",
                        "reason": str(ex),
                    }
                )
                continue
            expr = f"{_literal_display_name(snp1, neg1)}{op}{_literal_display_name(snp2, neg2)}"
            if op == "*" and (neg1 or neg2):
                skipped.append(
                    {
                        "line": str(line_no),
                        "expr": expr,
                        "reason": "negated_literals_not_supported_for_multiplicative_interaction",
                    }
                )
                continue
            try:
                row1 = _resolve_snp_token(snp1, name_map)
                row2 = _resolve_snp_token(snp2, name_map)
            except Exception as ex:
                skipped.append(
                    {
                        "line": str(line_no),
                        "expr": expr,
                        "reason": str(ex),
                    }
                )
                continue
            specs.append(
                InteractionSpec(
                    expr=expr,
                    snp1=snp1,
                    op=op,
                    snp2=snp2,
                    row1=int(row1),
                    row2=int(row2),
                    neg1=bool(neg1),
                    neg2=bool(neg2),
                )
            )
    return specs, skipped


def _decode_rows(
    packed_ctx: dict[str, object],
    row_indices_local: list[int] | np.ndarray,
    sample_indices_full: np.ndarray,
) -> np.ndarray:
    rows = np.ascontiguousarray(np.asarray(row_indices_local, dtype=np.int64).reshape(-1), dtype=np.int64)
    sidx = np.ascontiguousarray(np.asarray(sample_indices_full, dtype=np.int64).reshape(-1), dtype=np.int64)
    if int(rows.shape[0]) == 0:
        return np.empty((0, int(sidx.shape[0])), dtype=np.float32)
    decoded = jxrs.bed_packed_decode_rows_f32(
        packed_ctx["packed"],
        int(packed_ctx["n_samples"]),
        rows,
        packed_ctx["row_flip"],
        packed_ctx["maf"],
        sidx,
    )
    return np.ascontiguousarray(np.asarray(decoded, dtype=np.float32), dtype=np.float32)


def _make_combo_chunk(
    g1: np.ndarray,
    g2: np.ndarray,
    ops: list[str],
    neg1_flags: list[bool] | np.ndarray | None = None,
    neg2_flags: list[bool] | np.ndarray | None = None,
) -> np.ndarray:
    literal1 = _literalize_chunk(g1, neg1_flags)
    literal2 = _literalize_chunk(g2, neg2_flags)
    out = np.empty_like(np.asarray(g1, dtype=np.float32), dtype=np.float32)
    neg1_arr = (
        np.zeros((int(out.shape[0]),), dtype=np.bool_)
        if neg1_flags is None
        else np.ascontiguousarray(np.asarray(neg1_flags, dtype=np.bool_).reshape(-1), dtype=np.bool_)
    )
    neg2_arr = (
        np.zeros((int(out.shape[0]),), dtype=np.bool_)
        if neg2_flags is None
        else np.ascontiguousarray(np.asarray(neg2_flags, dtype=np.bool_).reshape(-1), dtype=np.bool_)
    )
    for i, op in enumerate(ops):
        if op == "*":
            if bool(neg1_arr[i]) or bool(neg2_arr[i]):
                raise ValueError(
                    "Negated literals are not supported for multiplicative interaction '*'."
                )
            out[i] = np.asarray(g1[i] * g2[i], dtype=np.float32)
        elif op == "&":
            out[i] = np.asarray(
                2.0 * ((literal1[i] > 0.0) & (literal2[i] > 0.0)),
                dtype=np.float32,
            )
        elif op == "|":
            out[i] = np.asarray(
                2.0 * ((literal1[i] > 0.0) | (literal2[i] > 0.0)),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unsupported interaction operator: {op}")
    return np.ascontiguousarray(out, dtype=np.float32)


def _literalize_chunk(
    g: np.ndarray,
    neg_flags: list[bool] | np.ndarray | None = None,
) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(g, dtype=np.float32), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Literal chunk must be 2D, got shape={arr.shape}.")
    neg = (
        np.zeros((int(arr.shape[0]),), dtype=np.bool_)
        if neg_flags is None
        else np.ascontiguousarray(np.asarray(neg_flags, dtype=np.bool_).reshape(-1), dtype=np.bool_)
    )
    if int(neg.shape[0]) != int(arr.shape[0]):
        raise ValueError(
            f"Literal negation flag length mismatch: flags={int(neg.shape[0])}, rows={int(arr.shape[0])}."
        )
    hit = arr > 0.0
    if bool(np.any(neg)):
        hit = np.where(neg.reshape(-1, 1), ~hit, hit)
    return np.ascontiguousarray((hit.astype(np.float32, copy=False) * 2.0), dtype=np.float32)


def _trait_label(trait_ref: object) -> str:
    text = str(trait_ref).strip()
    if text == "":
        return "trait"
    return text.replace("/", "_").replace("\\", "_")


def _delta_pwald_orders_of_magnitude(
    parent_p1: object,
    parent_p2: object,
    child_p: object,
) -> float:
    vals: list[float] = []
    for raw in (parent_p1, parent_p2):
        try:
            v = float(raw)
        except Exception:
            continue
        if np.isfinite(v) and v > 0.0:
            vals.append(v)
    try:
        child = float(child_p)
    except Exception:
        return float("nan")
    if len(vals) == 0 or (not np.isfinite(child)) or child <= 0.0:
        return float("nan")
    best_parent = min(vals)
    return float(np.log10(best_parent) - np.log10(child))


def _compact_fvlmm2_output_df(full_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "chrom",
        "pos",
        "combo_id",
        "combo_af",
        "unit_name",
        "beta_combo_joint",
        "se_combo_joint",
        "p_combo_joint",
        "p_lit1_joint",
        "p_lit2_joint",
    ]
    if full_df.shape[0] == 0:
        return pd.DataFrame(columns=columns)

    unit_name = (
        full_df["garfield_unit_name"].fillna("").astype(str)
        if "garfield_unit_name" in full_df.columns
        else pd.Series([""] * int(full_df.shape[0]), index=full_df.index, dtype="object")
    )
    compact = pd.DataFrame(
        {
            "chrom": full_df["chrom1"].astype(str),
            "pos": pd.to_numeric(full_df["pos1"], errors="coerce"),
            "combo_id": full_df["combo"].astype(str),
            "combo_af": pd.to_numeric(full_df["combo_af"], errors="coerce"),
            "unit_name": unit_name,
            "beta_combo_joint": pd.to_numeric(full_df["beta_combo_joint"], errors="coerce"),
            "se_combo_joint": pd.to_numeric(full_df["se_combo_joint"], errors="coerce"),
            "p_combo_joint": pd.to_numeric(full_df["p_combo_joint"], errors="coerce"),
            "p_lit1_joint": pd.to_numeric(full_df["p1_joint"], errors="coerce"),
            "p_lit2_joint": pd.to_numeric(full_df["p2_joint"], errors="coerce"),
        },
        index=full_df.index,
    )
    return compact.loc[:, columns]


def _format_fixed4(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.4f}"


def _format_sci4(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.4e}"


def _format_pos_int(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return str(int(round(v)))


def _format_fvlmm2_output_df_for_tsv(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[0] == 0:
        return df.copy()
    out = df.copy()
    out["chrom"] = out["chrom"].astype(str)
    out["pos"] = [
        _format_pos_int(v) for v in out["pos"]
    ]
    out["combo_id"] = out["combo_id"].astype(str)
    out["unit_name"] = [
        "." if str(v).strip() == "" or str(v).strip().lower() == "nan" else str(v).strip()
        for v in out["unit_name"]
    ]
    for col in ("combo_af", "beta_combo_joint", "se_combo_joint"):
        out[col] = [_format_fixed4(v) for v in out[col]]
    for col in (
        "p_combo_joint",
        "p_lit1_joint",
        "p_lit2_joint",
    ):
        out[col] = [_format_sci4(v) for v in out[col]]
    return out


def _build_grm_for_scan(
    *,
    grm_path: str | None,
    gfile: str,
    sample_ids_full: np.ndarray,
    n_snps: int,
    maf: float,
    geno: float,
    het: float,
    snps_only: bool,
    threads: int,
    out_dir: str,
    logger,
    packed_ctx: dict[str, object],
) -> np.ndarray:
    if grm_path:
        grm_all, _ = load_and_align_grm(
            str(grm_path),
            sample_ids_full.astype(str).tolist(),
            label="FvLMM2 GRM",
        )
        return np.asarray(grm_all, dtype=np.float64, order="C")

    cache_prefix = _gwas_cache_prefix_with_params(
        str(gfile),
        maf=float(maf),
        geno=float(geno),
        snps_only=bool(snps_only),
        cache_dir=str(out_dir),
        logger=logger,
    )
    grm_all, _eff_m, grm_ids, _cache_path = load_or_build_grm_with_cache(
        genofile=str(gfile),
        cache_prefix=cache_prefix,
        mgrm="1",
        maf_threshold=float(maf),
        max_missing_rate=float(geno),
        het_threshold=float(het),
        chunk_size=65536,
        memory_mb=1024.0,
        threads=int(threads),
        logger=logger,
        use_spinner=bool(stdout_is_tty()),
        ids_preloaded=np.asarray(sample_ids_full, dtype=str),
        n_snps_preloaded=int(n_snps),
        snps_only=bool(snps_only),
        allow_packed_full_load=True,
        preloaded_packed={
            "prefix": str(gfile),
            "full_ids": np.asarray(sample_ids_full, dtype=str),
            "packed_ctx": packed_ctx,
        },
    )
    if grm_ids is None:
        return np.asarray(grm_all, dtype=np.float64, order="C")
    grm_ids_arr = np.asarray(grm_ids, dtype=str)
    if np.array_equal(grm_ids_arr, np.asarray(sample_ids_full, dtype=str)):
        return np.asarray(grm_all, dtype=np.float64, order="C")
    idx = {sid: i for i, sid in enumerate(grm_ids_arr.tolist())}
    order = np.asarray([idx[str(sid)] for sid in sample_ids_full.astype(str).tolist()], dtype=np.int64)
    return np.asarray(grm_all[np.ix_(order, order)], dtype=np.float64, order="C")


def _run_trait_scan(
    *,
    trait_ref: object,
    pheno_base: pd.DataFrame,
    base_ids: np.ndarray,
    base_full_indices: np.ndarray,
    base_grm: np.ndarray,
    base_cov: np.ndarray | None,
    packed_ctx: dict[str, object],
    sites: list[ActiveSite],
    specs: list[InteractionSpec],
    threads: int,
    batch_size: int,
) -> pd.DataFrame:
    y_base, keep_pheno = _trait_values_and_mask(pheno_base, trait_ref)
    keep = np.asarray(keep_pheno, dtype=np.bool_).reshape(-1)
    if base_cov is not None:
        keep &= np.all(np.isfinite(base_cov), axis=1)
    keep &= np.isfinite(y_base)
    n_keep = int(np.count_nonzero(keep))
    cov_dim = 0 if base_cov is None else int(base_cov.shape[1])
    if n_keep <= cov_dim + 4:
        raise ValueError(
            f"Trait {trait_ref} has too few usable samples after alignment/filtering: n={n_keep}, cov={cov_dim}."
        )

    y_trait = np.ascontiguousarray(np.asarray(y_base[keep], dtype=np.float64).reshape(-1), dtype=np.float64)
    cov_trait = (
        None
        if base_cov is None
        else np.ascontiguousarray(np.asarray(base_cov[keep], dtype=np.float64), dtype=np.float64)
    )
    trait_ids = np.asarray(base_ids[keep], dtype=str)
    trait_sample_indices = np.ascontiguousarray(
        np.asarray(base_full_indices[keep], dtype=np.int64).reshape(-1),
        dtype=np.int64,
    )
    grm_trait = np.asarray(base_grm[np.ix_(keep, keep)], dtype=np.float64, order="C")

    model = FvLMM(y_trait, cov_trait, grm_trait)
    lbd_null = float(getattr(model, "lbd_null", np.nan))
    if not np.isfinite(lbd_null) or lbd_null <= 0.0:
        raise RuntimeError(f"Trait {trait_ref} produced invalid null lambda: {lbd_null}")
    log10_lbd = float(np.log10(lbd_null))
    s_vec = np.ascontiguousarray(np.asarray(model.S, dtype=np.float64).reshape(-1), dtype=np.float64)
    x_rot = np.ascontiguousarray(np.asarray(model.Xcov, dtype=np.float64), dtype=np.float64)
    y_rot = np.ascontiguousarray(np.asarray(model.y, dtype=np.float64).reshape(-1), dtype=np.float64)

    results: list[dict[str, object]] = []
    for start in range(0, len(specs), max(1, int(batch_size))):
        batch = specs[start : start + max(1, int(batch_size))]
        unique_rows = sorted({int(spec.row1) for spec in batch} | {int(spec.row2) for spec in batch})
        row_to_offset = {row: idx for idx, row in enumerate(unique_rows)}
        decoded_unique = _decode_rows(
            packed_ctx=packed_ctx,
            row_indices_local=unique_rows,
            sample_indices_full=trait_sample_indices,
        )
        unique_stats = np.asarray(model.gwas(decoded_unique, threads=int(threads)), dtype=np.float64)
        g1 = np.ascontiguousarray(
            decoded_unique[[row_to_offset[int(spec.row1)] for spec in batch], :],
            dtype=np.float32,
        )
        g2 = np.ascontiguousarray(
            decoded_unique[[row_to_offset[int(spec.row2)] for spec in batch], :],
            dtype=np.float32,
        )
        neg1_flags = [bool(spec.neg1) for spec in batch]
        neg2_flags = [bool(spec.neg2) for spec in batch]
        literal1 = _literalize_chunk(g1, neg1_flags)
        literal2 = _literalize_chunk(g2, neg2_flags)
        af1 = np.ascontiguousarray(np.mean(literal1 > 0.0, axis=1, dtype=np.float64), dtype=np.float64)
        af2 = np.ascontiguousarray(np.mean(literal2 > 0.0, axis=1, dtype=np.float64), dtype=np.float64)
        literal1_marg = np.asarray(model.gwas(literal1, threads=int(threads)), dtype=np.float64)
        literal2_marg = np.asarray(model.gwas(literal2, threads=int(threads)), dtype=np.float64)
        ops = [str(spec.op) for spec in batch]
        combo = _make_combo_chunk(g1, g2, ops, neg1_flags=neg1_flags, neg2_flags=neg2_flags)
        combo_af = np.ascontiguousarray(np.mean(combo > 0.0, axis=1, dtype=np.float64), dtype=np.float64)
        combo_marg = np.asarray(model.gwas(combo, threads=int(threads)), dtype=np.float64)
        joint = np.asarray(
            jxrs.fvlmm2_assoc_chunk_f32(
                s_vec,
                x_rot,
                y_rot,
                float(log10_lbd),
                g1,
                g2,
                combo,
                int(threads),
            ),
            dtype=np.float64,
        )
        for i, spec in enumerate(batch):
            site1 = sites[int(spec.row1)]
            site2 = sites[int(spec.row2)]
            stat1 = unique_stats[row_to_offset[int(spec.row1)]]
            stat2 = unique_stats[row_to_offset[int(spec.row2)]]
            lit1 = literal1_marg[i]
            lit2 = literal2_marg[i]
            statc = combo_marg[i]
            joint_row = joint[i]
            literal1_name = _literal_display_name(site1.snp, bool(spec.neg1))
            literal2_name = _literal_display_name(site2.snp, bool(spec.neg2))
            literal1_allele0, literal1_allele1 = _literal_alleles(site1, bool(spec.neg1))
            literal2_allele0, literal2_allele1 = _literal_alleles(site2, bool(spec.neg2))
            results.append(
                {
                    "trait": str(trait_ref),
                    "n_samples": int(trait_ids.shape[0]),
                    "chrom1": site1.chrom,
                    "pos1": int(site1.pos),
                    "snp1": site1.snp,
                    "allele0_1": site1.allele0,
                    "allele1_1": site1.allele1,
                    "literal1": literal1_name,
                    "literal1_negated": bool(spec.neg1),
                    "allele0_literal1": literal1_allele0,
                    "allele1_literal1": literal1_allele1,
                    "chrom2": site2.chrom,
                    "pos2": int(site2.pos),
                    "snp2": site2.snp,
                    "allele0_2": site2.allele0,
                    "allele1_2": site2.allele1,
                    "literal2": literal2_name,
                    "literal2_negated": bool(spec.neg2),
                    "allele0_literal2": literal2_allele0,
                    "allele1_literal2": literal2_allele1,
                    "op": str(spec.op),
                    "combo": str(spec.expr),
                    "af1": float(af1[i]),
                    "af2": float(af2[i]),
                    "combo_af": float(combo_af[i]),
                    "beta1_marginal": float(stat1[0]),
                    "se1_marginal": float(stat1[1]),
                    "p1_marginal": float(stat1[2]),
                    "beta_literal1_marginal": float(lit1[0]),
                    "se_literal1_marginal": float(lit1[1]),
                    "p_literal1_marginal": float(lit1[2]),
                    "beta2_marginal": float(stat2[0]),
                    "se2_marginal": float(stat2[1]),
                    "p2_marginal": float(stat2[2]),
                    "beta_literal2_marginal": float(lit2[0]),
                    "se_literal2_marginal": float(lit2[1]),
                    "p_literal2_marginal": float(lit2[2]),
                    "beta_combo_marginal": float(statc[0]),
                    "se_combo_marginal": float(statc[1]),
                    "p_combo_marginal": float(statc[2]),
                    "beta1_joint": float(joint_row[0]),
                    "se1_joint": float(joint_row[1]),
                    "p1_joint": float(joint_row[2]),
                    "beta2_joint": float(joint_row[3]),
                    "se2_joint": float(joint_row[4]),
                    "p2_joint": float(joint_row[5]),
                    "beta_combo_joint": float(joint_row[6]),
                    "se_combo_joint": float(joint_row[7]),
                    "p_combo_joint": float(joint_row[8]),
                }
            )
    return pd.DataFrame(results)


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx fvlmm2",
        description=(
            "Joint FvLMM recheck for specified pseudo/interaction loci.\n"
            "Model: y = covariates + SNP1 + SNP2 + combo + Zu + e"
        ),
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            ["jx fvlmm2 -bfile geno -p pheno.tsv -n TraitA -i pairs.txt -k geno.cGRM.npy"]
        ),
    )
    src = parser.add_argument_group("Input")
    src_mx = src.add_mutually_exclusive_group(required=True)
    add_common_genotype_source_args(src_mx, help_profile="default")
    add_common_pheno_arg(src, required=True)
    add_common_trait_selector_args(src)
    src.add_argument(
        "-i",
        "--interaction",
        type=str,
        required=True,
        help=(
            "Interaction file; one expression per line. Supported forms: "
            "snp1&snp2, snp1|snp2, snp1*snp2."
        ),
    )

    model = parser.add_argument_group("Model")
    add_common_covariate_file_or_site_arg(model)
    add_common_grm_file_arg(
        model,
        default=None,
        required=False,
        help_profile="garfield_residualization",
        help_text=(
            "Optional precomputed GRM/kernel (.npy or text). "
            "If omitted, build a GRM from the genotype input."
        ),
    )

    filt = parser.add_argument_group("Variant Filter")
    add_common_variant_filter_args(
        filt,
        help_profile="gwas_threshold",
        include_maf=True,
        include_geno=True,
        include_het=True,
        maf_default=0.02,
        geno_default=0.05,
        het_default=0.0,
    )
    add_common_snps_only_arg(filt, default=False)

    run = parser.add_argument_group("Runtime")
    add_common_thread_arg(run, default_threads=int(detect_effective_threads()), help_profile="default")
    run.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Number of interaction rows processed per batch (default: %(default)s).",
    )

    out = parser.add_argument_group("Output")
    add_common_out_arg(out, default=".", help_profile="current_dir")
    add_common_prefix_arg(out, default=None, help_profile="genotype_basename")
    return parser


def main(argv: list[str] | None = None) -> int:
    _require_rust_backend()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.ncol = parse_trait_selector_specs(args.ncol, label="-n/--n")
    except ValueError as ex:
        parser.error(str(ex))

    detected_threads = int(detect_effective_threads())
    requested_threads = int(args.thread)
    if int(args.thread) <= 0:
        args.thread = detected_threads
    if int(args.thread) > detected_threads:
        args.thread = detected_threads

    args.out = os.path.normpath(args.out if args.out is not None else ".")
    os.makedirs(args.out, mode=0o755, exist_ok=True)
    configure_genotype_cache_from_out(args.out)

    raw_gfile, auto_prefix = determine_genotype_source(
        vcf=getattr(args, "vcf", None),
        hmp=getattr(args, "hmp", None),
        file=getattr(args, "file", None),
        bfile=getattr(args, "bfile", None),
        prefix=None,
        apply_cache=False,
    )
    prefix = str(args.prefix) if args.prefix else str(auto_prefix)
    outprefix = os.path.join(str(args.out), str(prefix))
    logger = setup_logging(f"{outprefix}.fvlmm2.log")
    apply_outer_thread_cap(int(args.thread))

    checks: list[bool] = [
        ensure_file_exists(logger, str(args.pheno), "Phenotype file"),
        ensure_file_exists(logger, str(args.interaction), "Interaction file"),
    ]
    if args.vcf:
        checks.append(ensure_file_exists(logger, raw_gfile, "Genotype VCF input"))
    elif args.hmp:
        checks.append(ensure_file_exists(logger, raw_gfile, "Genotype HMP input"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, raw_gfile, "Genotype FILE input"))
        checks.append(
            ensure_file_input_site_metadata_exists(logger, raw_gfile, "Genotype FILE site metadata")
        )
    elif args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, raw_gfile, "Genotype PLINK prefix"))
    if args.grm:
        checks.append(ensure_file_exists(logger, str(args.grm), "GRM"))
    if not ensure_all_true(checks):
        return 1

    gfile, _resolved_prefix = determine_genotype_source(
        vcf=getattr(args, "vcf", None),
        hmp=getattr(args, "hmp", None),
        file=getattr(args, "file", None),
        bfile=getattr(args, "bfile", None),
        prefix=prefix,
        snps_only=bool(args.snps_only),
        threads=int(args.thread),
        apply_cache=True,
    )
    if not ensure_plink_prefix_exists(logger, gfile, "Cached PLINK prefix for fvlmm2"):
        return 1

    emit_cli_configuration(
        logger,
        app_title="JanusX fvlmm2",
        config_title="Joint FvLMM interaction recheck",
        host=socket.gethostname(),
        sections=[
            (
                "Input",
                [
                    ("Raw genotype", raw_gfile),
                    ("PLINK prefix", gfile),
                    ("Phenotype", str(args.pheno)),
                    ("Interaction", str(args.interaction)),
                    ("GRM", (str(args.grm) if args.grm else "auto-build")),
                ],
            ),
            (
                "Filter",
                [
                    ("MAF", float(args.maf)),
                    ("Missing", float(args.geno)),
                    ("Het", float(args.het)),
                    ("SNP-only", bool(args.snps_only)),
                ],
            ),
            (
                "Runtime",
                [
                    (
                        "Threads",
                        format_requested_thread_usage(
                            requested_threads=int(requested_threads),
                            using_threads=int(args.thread),
                            detected_threads=int(detected_threads),
                        ),
                    ),
                    ("Batch size", int(args.batch_size)),
                ],
            ),
        ],
    )

    pheno = _load_phenotype_with_status(
        str(args.pheno),
        args.ncol,
        logger,
        use_spinner=bool(stdout_is_tty()),
    )
    sample_ids_full, n_snps = _inspect_genotype_with_status(
        str(gfile),
        logger,
        use_spinner=bool(stdout_is_tty()),
        snps_only=bool(args.snps_only),
        maf_threshold=float(args.maf),
        max_missing_rate=float(args.geno),
        het_threshold=float(args.het),
    )
    pheno_aligned_full, sample_ids_full = _align_pheno_to_sample_order(pheno, sample_ids_full)
    cov_all, cov_ids = _load_covariates_for_models(
        getattr(args, "cov", None),
        str(gfile),
        sample_ids_full,
        65536,
        logger,
        "fvlmm2",
        use_spinner=bool(stdout_is_tty()),
        snps_only=bool(args.snps_only),
    )

    if cov_all is None or cov_ids is None:
        base_ids = np.asarray(sample_ids_full, dtype=str)
        base_full_indices = np.arange(int(sample_ids_full.shape[0]), dtype=np.int64)
        base_cov = None
        pheno_base = pheno_aligned_full
    else:
        base_ids = np.asarray(cov_ids, dtype=str)
        full_index_map = {str(sid): i for i, sid in enumerate(np.asarray(sample_ids_full, dtype=str).tolist())}
        base_full_indices = np.asarray([full_index_map[str(sid)] for sid in base_ids.tolist()], dtype=np.int64)
        base_cov = np.ascontiguousarray(np.asarray(cov_all, dtype=np.float64), dtype=np.float64)
        pheno_base = pheno_aligned_full.reindex(pd.Index(base_ids, dtype=str))

    logger.info(
        "Loading packed genotype context for requested interaction loci (use_cache=False, no pmeta sidecars)."
    )
    _full_ids_packed, packed_ctx = prepare_packed_ctx_from_plink(
        str(gfile),
        maf=float(args.maf),
        missing_rate=float(args.geno),
        het_threshold=float(args.het),
        snps_only=bool(args.snps_only),
        filter_mode="compact",
        use_cache=False,
    )
    sites, name_map = _load_active_sites(str(gfile), packed_ctx)
    specs, skipped = _parse_interaction_file(str(args.interaction), name_map)
    if len(specs) == 0:
        raise ValueError("No valid interaction expressions remain after variant lookup/filtering.")
    logger.info(
        "Resolved %d interaction rows against %d active filtered variants; skipped=%d.",
        len(specs),
        len(sites),
        len(skipped),
    )
    if len(skipped) > 0:
        skipped_path = f"{outprefix}.fvlmm2.skip"
        pd.DataFrame(skipped).to_csv(skipped_path, sep="\t", index=False)
        logger.warning("Skipped interaction rows were written to %s", format_path_for_display(skipped_path))

    grm_full = _build_grm_for_scan(
        grm_path=getattr(args, "grm", None),
        gfile=str(gfile),
        sample_ids_full=np.asarray(sample_ids_full, dtype=str),
        n_snps=int(n_snps),
        maf=float(args.maf),
        geno=float(args.geno),
        het=float(args.het),
        snps_only=bool(args.snps_only),
        threads=int(args.thread),
        out_dir=str(args.out),
        logger=logger,
        packed_ctx=packed_ctx,
    )
    base_grm = np.asarray(grm_full[np.ix_(base_full_indices, base_full_indices)], dtype=np.float64, order="C")

    trait_refs = _resolve_trait_iter(pheno_base, args.ncol)
    saved_paths: list[str] = []
    for trait_ref in trait_refs:
        trait_tag = _trait_label(trait_ref)
        logger.info("Running joint FvLMM recheck for trait %s ...", trait_ref)
        full_df = _run_trait_scan(
            trait_ref=trait_ref,
            pheno_base=pheno_base,
            base_ids=np.asarray(base_ids, dtype=str),
            base_full_indices=np.asarray(base_full_indices, dtype=np.int64),
            base_grm=np.asarray(base_grm, dtype=np.float64, order="C"),
            base_cov=base_cov,
            packed_ctx=packed_ctx,
            sites=sites,
            specs=specs,
            threads=int(args.thread),
            batch_size=int(args.batch_size),
        )
        df = _format_fvlmm2_output_df_for_tsv(_compact_fvlmm2_output_df(full_df))
        tsv_path = f"{outprefix}.{trait_tag}.fvlmm2.tsv"
        df.to_csv(tsv_path, sep="\t", index=False)
        saved_paths.append(tsv_path)
        logger.info(
            "Saved %d joint recheck rows for trait %s to %s",
            int(df.shape[0]),
            trait_ref,
            format_path_for_display(tsv_path),
        )

    if len(saved_paths) == 1:
        logger.info("Result saved to %s", format_path_for_display(saved_paths[0]))
    else:
        logger.info("Results saved:\n%s", "\n".join(f"  {format_path_for_display(p)}" for p in saved_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
