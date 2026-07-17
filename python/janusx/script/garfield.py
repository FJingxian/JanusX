import argparse
import json
import logging
import os
from pathlib import Path
import re
import socket
import tempfile
import threading
import time
from typing import Optional

import numpy as np
import pandas as pd
import psutil

from janusx.assoc.workflow import (
    _format_cli_finished_timestamp,
    _gwas_terminal_config_line_max_chars,
    _inspect_genotype_with_status,
    _load_covariates_for_models,
    _load_phenotype_with_status,
    _terminal_saved_result_paths,
    load_or_build_grm_with_cache,
)
from janusx.assoc.workflow_cache import _gwas_cache_prefix_with_params
from janusx.gfreader import prepare_bed_logic_keep_mask_pure_line
from janusx.gtools.reader import readanno
from janusx.script._common.cli_args import (
    add_common_covariate_file_or_site_arg,
    add_common_genotype_source_args,
    add_common_grm_file_arg,
    add_common_out_arg,
    add_common_pheno_arg,
    add_common_prefix_arg,
    add_common_thread_arg,
    add_common_trait_selector_args,
    add_common_variant_filter_args,
    parse_trait_selector_specs,
)
from janusx.script._common.config_render import emit_cli_configuration
from janusx.script._common.genoio import (
    determine_genotype_source_from_args as determine_genotype_source,
    packed_meta_cache_prefix,
    prepare_packed_ctx_from_plink,
)
from janusx.script._common.genocache import configure_genotype_cache_from_out
from janusx.script._common.grmio import load_and_align_grm
from janusx.script._common.cli_core import CliArgumentParser, cli_help_formatter
from janusx.script._common.log import setup_logging
from janusx.script._common.memory import (
    resolve_decode_mmap_window_mb as _common_resolve_decode_mmap_window_mb,
)
from janusx.script._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_plink_prefix_exists,
    format_path_for_display,
)
from janusx.script._common.progress import (
    CliStatus,
    ProgressAdapter,
    build_rich_progress,
    print_failure,
    rich_progress_available,
    stdout_is_tty,
    success_symbol,
)
from janusx.script._common.threads import (
    apply_outer_thread_cap,
    detect_effective_threads,
    format_requested_thread_usage,
)
from janusx.assoc.workflow_ui import _emit_plain_info_line, _rich_success
from janusx.assoc.workflow_ui import _run_fastplot_from_tsv_with_status
from janusx.script.fvlmm2 import (
    InteractionSpec as _Fvlmm2InteractionSpec,
    _compact_fvlmm2_output_df as _fvlmm2_compact_output_df,
    _format_fvlmm2_output_df_for_tsv as _fvlmm2_format_output_df_for_tsv,
    _load_active_sites as _fvlmm2_load_active_sites,
    _require_rust_backend as _fvlmm2_require_rust_backend,
    _run_trait_scan as _fvlmm2_run_trait_scan,
    _split_literal_token as _fvlmm2_split_literal_token,
)


try:
    from janusx.janusx import garfield_logic_search_bed
except Exception as _exc:  # pragma: no cover
    garfield_logic_search_bed = None  # type: ignore[assignment]
    _RUST_IMPORT_ERROR = _exc
else:
    _RUST_IMPORT_ERROR = None

# Python 仅负责 CLI 调度；训练/测试划分、null-model 残差化、ML 候选筛选、
# beam search 与最终导出都由 Rust 端统一完成。


def _current_bed_memory_mb() -> float:
    try:
        mb = float(os.environ.get("JX_BED_BLOCK_TARGET_MB", "512"))
    except Exception:
        return 512.0
    return mb if np.isfinite(mb) and mb > 0.0 else 512.0


def _env_truthy(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _garfield_rss_debug_enabled() -> bool:
    return _env_truthy("JX_GARFIELD_RSS_DEBUG")


def _format_mem_bytes(value: object) -> str:
    if value is None:
        return "NA"
    try:
        b = int(value)
    except Exception:
        return "NA"
    if b <= 0:
        return "NA"
    kib = 1024.0
    mib = kib * 1024.0
    gib = mib * 1024.0
    if b >= gib:
        return f"{b / gib:.2f} GiB"
    if b >= mib:
        return f"{b / mib:.1f} MiB"
    if b >= kib:
        return f"{b / kib:.1f} KiB"
    return f"{b} B"


class _GarfieldRssSampler:
    def __init__(self, interval_s: float = 0.05):
        self._process = psutil.Process(os.getpid())
        self._interval_s = max(0.01, float(interval_s))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_rss_bytes: Optional[int] = None
        self._end_rss_bytes: Optional[int] = None
        self._observed_peak_rss_bytes: Optional[int] = None
        self._samples = 0

    def _sample_once(self) -> None:
        try:
            rss_now = int(self._process.memory_info().rss)
        except Exception:
            return
        if rss_now <= 0:
            return
        self._samples += 1
        if self._start_rss_bytes is None:
            self._start_rss_bytes = rss_now
        self._end_rss_bytes = rss_now
        self._observed_peak_rss_bytes = max(
            int(self._observed_peak_rss_bytes or 0),
            rss_now,
        )

    def __enter__(self):
        self._sample_once()

        def _worker() -> None:
            while not self._stop.wait(self._interval_s):
                self._sample_once()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, self._interval_s * 4.0))
            self._thread = None
        self._sample_once()

    def summary(self) -> dict[str, object]:
        return {
            "metric": "rss",
            "samples": int(self._samples),
            "start_current_bytes": self._start_rss_bytes,
            "start_rss_bytes": self._start_rss_bytes,
            "start_footprint_bytes": None,
            "end_current_bytes": self._end_rss_bytes,
            "end_rss_bytes": self._end_rss_bytes,
            "end_footprint_bytes": None,
            "observed_peak_current_bytes": self._observed_peak_rss_bytes,
            "observed_peak_rss_bytes": self._observed_peak_rss_bytes,
            "observed_peak_footprint_bytes": None,
        }


def _emit_garfield_rss_checkpoint(
    logger,
    stage: str,
    payload: object,
) -> None:
    if not isinstance(payload, dict):
        return
    samples = int(payload.get("samples") or 0)
    start_rss = payload.get("start_rss_bytes") or payload.get("start_current_bytes")
    end_rss = payload.get("end_rss_bytes") or payload.get("end_current_bytes")
    peak_rss = payload.get("observed_peak_rss_bytes") or payload.get(
        "observed_peak_current_bytes"
    )
    peak_footprint = payload.get("observed_peak_footprint_bytes")
    if start_rss is None and end_rss is None and peak_rss is None and peak_footprint is None:
        return
    metric = payload.get("metric") or "rss"
    parts = [
        f"[GARFIELD-RSS] stage={stage}",
        f"metric={metric}",
        f"rss_start={_format_mem_bytes(start_rss)}",
        f"rss_end={_format_mem_bytes(end_rss)}",
        f"rss_peak={_format_mem_bytes(peak_rss)}",
    ]
    if peak_footprint is not None:
        parts.append(f"footprint_peak={_format_mem_bytes(peak_footprint)}")
    parts.append(f"samples={samples}")
    logger.info(" ".join(parts))


class _GarfieldPhenoLogger:
    """Suppress duplicate phenotype selection line from shared loader."""

    def __init__(self, base_logger):
        self._base = base_logger

    def info(self, message, *args, **kwargs):
        msg = str(message)
        if msg.startswith("Phenotypes to be analyzed: "):
            return
        return self._base.info(message, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)


def _require_rust_backend() -> None:
    if garfield_logic_search_bed is None:
        raise ImportError(
            "janusx Rust extension does not provide the GARFIELD Rust pipeline API. "
            "Please rebuild/reinstall JanusX extension."
        ) from _RUST_IMPORT_ERROR


def _looks_numeric_token(token: object) -> bool:
    s = str(token).strip()
    if s == "":
        return False
    try:
        float(s)
        return True
    except Exception:
        return False


def _looks_sample_header_token(token: object) -> bool:
    text = str(token).strip().lower()
    if text == "":
        return False
    norm = "".join(ch for ch in text if ch.isalnum())
    return norm in {
        "sampleid",
        "sample",
        "id",
        "iid",
        "fid",
        "taxa",
        "accession",
        "line",
    }


def _split_pheno_line(line: str) -> list[str]:
    if "\t" in line:
        return [x.strip() for x in line.split("\t")]
    if "," in line:
        return [x.strip() for x in line.split(",")]
    return [x.strip() for x in line.split()]


def _read_phenotype_header_names(phenofile: str) -> Optional[list[str]]:
    try:
        with open(phenofile, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = str(raw).rstrip("\r\n")
                if line.strip() == "":
                    continue
                parts = _split_pheno_line(line)
                if len(parts) <= 1:
                    return None
                first_token = str(parts[0]).strip()
                names = [str(x).strip() for x in parts[1:]]
                if len(names) == 0:
                    return None
                if _looks_sample_header_token(first_token):
                    return names
                if any((n != "") and (not _looks_numeric_token(n)) for n in names):
                    return names
                return None
    except Exception:
        return None
    return None


def _normalize_trait_names_from_header(pheno, phenofile: str):
    trait_names = [str(c) for c in pheno.columns]
    selected_ncol = pheno.attrs.get("selected_ncol", None)
    if not isinstance(selected_ncol, list) or len(selected_ncol) != len(trait_names):
        return pheno, trait_names

    header_names = _read_phenotype_header_names(phenofile)
    if not header_names:
        return pheno, trait_names

    mapped: list[str] = []
    changed = False
    for idx_obj, current in zip(selected_ncol, trait_names):
        cur = str(current).strip()
        numeric_like = cur.lstrip("+-").isdigit()
        if numeric_like:
            try:
                idx = int(idx_obj)
            except Exception:
                idx = -1
            if 0 <= idx < len(header_names):
                cand = str(header_names[idx]).strip()
                if cand != "":
                    mapped.append(cand)
                    if cand != cur:
                        changed = True
                    continue
        mapped.append(cur)

    if changed:
        pheno = pheno.copy()
        pheno.columns = mapped
        return pheno, mapped
    return pheno, trait_names


def _safe_trait_label(label: object) -> str:
    name = str(label).strip()
    if name == "":
        return "trait"
    return name.replace("/", "_").replace("\\", "_")


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


def _ensure_followup_grm(
    *,
    existing_grm: Optional[np.ndarray],
    genofile: str,
    sample_ids: np.ndarray,
    n_snps: int,
    maf_threshold: float,
    max_missing_rate: float,
    threads: int,
    cache_dir: str,
    logger,
    use_spinner: bool,
) -> np.ndarray:
    if existing_grm is not None:
        return np.asarray(existing_grm, dtype=np.float64, order="C")

    cache_prefix = _gwas_cache_prefix_with_params(
        genofile,
        maf=float(maf_threshold),
        geno=float(max_missing_rate),
        snps_only=False,
        cache_dir=cache_dir,
        logger=logger,
    )
    grm_all, _eff_m, grm_ids, _grm_cache_path = load_or_build_grm_with_cache(
        genofile=genofile,
        cache_prefix=cache_prefix,
        mgrm="1",
        maf_threshold=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
        het_threshold=0.0,
        chunk_size=65536,
        threads=int(threads),
        memory_mb=1024.0,
        logger=logger,
        use_spinner=bool(use_spinner),
        ids_preloaded=np.asarray(sample_ids, dtype=str),
        n_snps_preloaded=int(n_snps),
        snps_only=False,
        allow_packed_full_load=True,
    )
    return _align_square_matrix_to_ids(
        np.asarray(grm_all, dtype=np.float64),
        grm_ids,
        sample_ids,
        label="GARFIELD follow-up GRM",
    )


def _prepare_site_keep(
    *,
    genofile: str,
    sample_ids: list[str],
    sample_index_map: dict[str, int],
    n_snps: int,
    maf_threshold: float,
    max_missing_rate: float,
    snps_only: bool,
    threads: int,
    use_spinner: bool,
    global_stats: bool,
) -> np.ndarray:
    if len(sample_ids) == 0:
        raise ValueError("GARFIELD metadata statistics require at least one sample.")
    sample_indices = np.asarray(
        [int(sample_index_map[str(sid)]) for sid in sample_ids],
        dtype=np.int64,
    )
    if bool(global_stats):
        task_label = "Computing global row statistics..."
        fail_label = "Computing global row statistics ...Failed"
        done_label = "Computing global row statistics ...Finished"
    else:
        task_label = "Computing trait-subset row statistics..."
        fail_label = "Computing trait-subset row statistics ...Failed"
        done_label = "Computing trait-subset row statistics ...Finished"
    mmap_window_mb = _common_resolve_decode_mmap_window_mb(
        str(genofile),
        int(len(sample_ids)),
        int(max(1, int(n_snps))),
        _current_bed_memory_mb(),
        needs_copy=False,
        buffers=1,
    )
    with CliStatus(task_label, enabled=bool(use_spinner), use_process=True) as task:
        try:
            site_keep_raw, _n_samples, n_total_sites = prepare_bed_logic_keep_mask_pure_line(
                str(genofile),
                sample_indices=sample_indices,
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                snps_only=bool(snps_only),
                mmap_window_mb=(int(mmap_window_mb) if mmap_window_mb is not None else None),
                threads=max(1, int(threads)),
            )
            site_keep = np.ascontiguousarray(
                np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1),
                dtype=np.bool_,
            )
            if int(site_keep.shape[0]) != int(n_total_sites):
                raise ValueError(
                    "GARFIELD site_keep length mismatch: "
                    f"mask={int(site_keep.shape[0])}, total={int(n_total_sites)}"
                )
            kept_n = int(np.count_nonzero(site_keep))
            if kept_n <= 0:
                raise ValueError("GARFIELD metadata statistics produced zero active SNPs.")
        except BaseException:
            task.fail(fail_label)
            raise
        suffix = "; reused across traits" if bool(global_stats) else ""
        task.complete(f"{done_label} (nSNP={kept_n}{suffix})")
    return site_keep


_GARFIELD_FOLLOWUP_PAIR_RE = re.compile(
    r"^\s*(!?[^\s&|*]+)\s*([&|*])\s*(!?[^\s&|*]+)\s*$"
)


def _garfield_followup_key(chrom: object, pos: object, snp: object) -> tuple[str, int, str]:
    return (str(chrom).strip(), int(pos), str(snp).strip())


def _garfield_followup_chisq(beta: object, se: object) -> float:
    try:
        beta_f = float(beta)
        se_f = float(se)
    except Exception:
        return float("nan")
    if (not np.isfinite(beta_f)) or (not np.isfinite(se_f)) or abs(se_f) <= 0.0:
        return float("nan")
    z = beta_f / se_f
    return float(z * z)


def _resolve_garfield_followup_snp_token(
    token: str,
    name_map: dict[str, list[int]],
) -> int:
    hits = name_map.get(str(token).strip(), [])
    if len(hits) == 1:
        return int(hits[0])
    if len(hits) > 1:
        raise ValueError(
            f"SNP token '{token}' is ambiguous after filtering; matched {len(hits)} active rows."
        )
    raise KeyError(f"SNP token '{token}' was not found among active filtered variants.")


def _load_garfield_pseudo_row_meta(
    *,
    pseudo_prefix: str,
    expected_sample_ids: list[str],
    maf_threshold: float,
    max_missing_rate: float,
) -> dict[tuple[str, int, str], dict[str, object]]:
    pseudo_ids, pseudo_ctx = prepare_packed_ctx_from_plink(
        str(pseudo_prefix),
        maf=float(maf_threshold),
        missing_rate=float(max_missing_rate),
        het_threshold=0.0,
        snps_only=False,
        filter_mode="compact",
        use_cache=False,
    )
    pseudo_ids_arr = np.asarray(pseudo_ids, dtype=str)
    expected_ids_arr = np.asarray(expected_sample_ids, dtype=str)
    if (
        int(pseudo_ids_arr.shape[0]) != int(expected_ids_arr.shape[0])
        or not np.array_equal(pseudo_ids_arr, expected_ids_arr)
    ):
        raise ValueError(
            "GARFIELD pseudo follow-up sample mismatch between pseudo BED and aligned trait samples."
        )
    pseudo_sites, _ = _fvlmm2_load_active_sites(str(pseudo_prefix), pseudo_ctx)
    af = np.ascontiguousarray(
        np.asarray(pseudo_ctx.get("af", pseudo_ctx["maf"]), dtype=np.float32).reshape(-1),
        dtype=np.float32,
    )
    miss = np.ascontiguousarray(
        np.asarray(pseudo_ctx["missing_rate"], dtype=np.float32).reshape(-1),
        dtype=np.float32,
    )
    if int(af.shape[0]) != len(pseudo_sites) or int(miss.shape[0]) != len(pseudo_sites):
        raise ValueError(
            "GARFIELD pseudo follow-up metadata length mismatch: "
            f"sites={len(pseudo_sites)}, af={int(af.shape[0])}, miss={int(miss.shape[0])}."
        )
    out: dict[tuple[str, int, str], dict[str, object]] = {}
    for idx, site in enumerate(pseudo_sites):
        out[_garfield_followup_key(site.chrom, site.pos, site.snp)] = {
            "chrom": str(site.chrom),
            "pos": int(site.pos),
            "snp": str(site.snp),
            "allele0": str(site.allele0),
            "allele1": str(site.allele1),
            "af": float(af[idx]),
            "miss": float(miss[idx]),
        }
    return out


def _collect_garfield_fvlmm2_specs(
    *,
    rules_tsv: str,
    name_map: dict[str, list[int]],
) -> tuple[list[_Fvlmm2InteractionSpec], pd.DataFrame, list[dict[str, str]]]:
    rules = pd.read_csv(
        str(rules_tsv),
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    specs: list[_Fvlmm2InteractionSpec] = []
    supported_meta: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    seen_combo: set[str] = set()

    for row_no, row in enumerate(rules.to_dict(orient="records"), start=2):
        snp_name = str(row.get("snp_name", "")).strip()
        expr = str(row.get("expr", "")).strip()
        if snp_name == "":
            skipped.append(
                {
                    "line": str(row_no),
                    "expr": expr,
                    "snp_name": snp_name,
                    "reason": "missing_snp_name",
                }
            )
            continue
        if snp_name in seen_combo:
            skipped.append(
                {
                    "line": str(row_no),
                    "expr": expr,
                    "snp_name": snp_name,
                    "reason": "duplicate_combo",
                }
            )
            continue
        seen_combo.add(snp_name)
        match = _GARFIELD_FOLLOWUP_PAIR_RE.match(snp_name)
        if match is None:
            skipped.append(
                {
                    "line": str(row_no),
                    "expr": expr,
                    "snp_name": snp_name,
                    "reason": (
                        "singleton_rule"
                        if not any(op in snp_name for op in ("|", "&", "*"))
                        else "requires_exactly_two_literals"
                    ),
                }
            )
            continue
        token1 = str(match.group(1)).strip()
        op = str(match.group(2)).strip()
        token2 = str(match.group(3)).strip()
        try:
            snp1, neg1 = _fvlmm2_split_literal_token(token1)
            snp2, neg2 = _fvlmm2_split_literal_token(token2)
        except Exception as ex:
            skipped.append(
                {
                    "line": str(row_no),
                    "expr": expr,
                    "snp_name": snp_name,
                    "reason": str(ex),
                }
            )
            continue
        combo = f"{'!' if neg1 else ''}{snp1}{op}{'!' if neg2 else ''}{snp2}"
        if op == "*" and (neg1 or neg2):
            skipped.append(
                {
                    "line": str(row_no),
                    "expr": expr,
                    "snp_name": snp_name,
                    "reason": "negated_literals_not_supported_for_multiplicative_interaction",
                }
            )
            continue
        try:
            row1 = _resolve_garfield_followup_snp_token(snp1, name_map)
            row2 = _resolve_garfield_followup_snp_token(snp2, name_map)
        except Exception as ex:
            skipped.append(
                {
                    "line": str(row_no),
                    "expr": expr,
                    "snp_name": snp_name,
                    "reason": str(ex),
                }
            )
            continue
        specs.append(
            _Fvlmm2InteractionSpec(
                expr=combo,
                snp1=snp1,
                op=op,
                snp2=snp2,
                row1=int(row1),
                row2=int(row2),
                neg1=bool(neg1),
                neg2=bool(neg2),
            )
        )
        supported_meta.append(
            {
                "combo": combo,
                "garfield_unit_kind": str(row.get("unit_kind", "")).strip(),
                "garfield_unit_index": str(row.get("unit_index", "")).strip(),
                "garfield_unit_name": str(row.get("unit_name", "")).strip(),
                "garfield_region_size": str(row.get("region_size", "")).strip(),
                "garfield_ml_feature_count": str(row.get("ml_feature_count", "")).strip(),
                "garfield_ml_rank": str(row.get("MLrank", row.get("ml_rank", ""))).strip(),
                "garfield_rule_snp_name": snp_name,
                "garfield_rule_expr": expr,
                "garfield_rule_beta": str(row.get("beta", "")).strip(),
                "garfield_rule_se": str(row.get("se", "")).strip(),
                "garfield_rule_chisq": str(row.get("chisq", "")).strip(),
                "garfield_rule_pwald": str(row.get("pwald", "")).strip(),
                "garfield_rule_score": str(row.get("score", "")).strip(),
                "garfield_rule_delta_score": str(row.get("delta_score", "")).strip(),
                "garfield_rule_delta_pwald": str(row.get("delta_pwald", "")).strip(),
            }
        )

    return specs, pd.DataFrame(supported_meta), skipped


def _garfield_followup_meta_or_fallback(
    *,
    row_meta: dict[tuple[str, int, str], dict[str, object]],
    key: tuple[str, int, str],
    allele0: str,
    allele1: str,
) -> dict[str, object]:
    meta = row_meta.get(key)
    if meta is not None:
        return meta
    chrom, pos, snp = key
    return {
        "chrom": chrom,
        "pos": int(pos),
        "snp": snp,
        "allele0": str(allele0),
        "allele1": str(allele1),
        "af": float("nan"),
        "miss": float("nan"),
    }


def _build_garfield_fvlmm2_expanded_df(
    *,
    raw_df: pd.DataFrame,
    pseudo_row_meta: dict[tuple[str, int, str], dict[str, object]],
) -> pd.DataFrame:
    if raw_df.shape[0] == 0:
        return raw_df.copy()

    emitted_singletons: set[tuple[str, int, str]] = set()
    out_rows: list[dict[str, object]] = []
    for rec in raw_df.itertuples(index=False):
        rec_dict = rec._asdict()
        single_specs = [
            (
                1,
                _garfield_followup_key(rec.chrom1, rec.pos1, rec.literal1),
                str(rec.allele0_literal1),
                str(rec.allele1_literal1),
                float(rec.beta_literal1_marginal),
                float(rec.se_literal1_marginal),
                float(rec.p_literal1_marginal),
                float(rec.beta1_marginal),
                float(rec.se1_marginal),
                float(rec.p1_marginal),
                float(rec.beta1_joint),
                float(rec.se1_joint),
                float(rec.p1_joint),
            ),
            (
                2,
                _garfield_followup_key(rec.chrom2, rec.pos2, rec.literal2),
                str(rec.allele0_literal2),
                str(rec.allele1_literal2),
                float(rec.beta_literal2_marginal),
                float(rec.se_literal2_marginal),
                float(rec.p_literal2_marginal),
                float(rec.beta2_marginal),
                float(rec.se2_marginal),
                float(rec.p2_marginal),
                float(rec.beta2_joint),
                float(rec.se2_joint),
                float(rec.p2_joint),
            ),
        ]
        for (
            component_index,
            key,
            allele0,
            allele1,
            beta_lit,
            se_lit,
            p_lit,
            beta_main,
            se_main,
            p_main,
            beta_j,
            se_j,
            p_j,
        ) in single_specs:
            if key in emitted_singletons:
                continue
            meta = _garfield_followup_meta_or_fallback(
                row_meta=pseudo_row_meta,
                key=key,
                allele0=allele0,
                allele1=allele1,
            )
            row = dict(rec_dict)
            row.update(
                {
                    "chrom": str(meta["chrom"]),
                    "pos": int(meta["pos"]),
                    "snp": str(meta["snp"]),
                    "allele0": str(meta["allele0"]),
                    "allele1": str(meta["allele1"]),
                    "af": float(meta["af"]),
                    "miss": float(meta["miss"]),
                    "beta": float(beta_lit),
                    "se": float(se_lit),
                    "chisq": _garfield_followup_chisq(beta_lit, se_lit),
                    "pwald": float(p_lit),
                    "row_role": "singleton",
                    "component_index": int(component_index),
                    "parent_combo": str(rec.combo),
                    "main_beta_marginal": float(beta_main),
                    "main_se_marginal": float(se_main),
                    "main_pwald_marginal": float(p_main),
                    "main_beta_joint_in_parent": float(beta_j),
                    "main_se_joint_in_parent": float(se_j),
                    "main_pwald_joint_in_parent": float(p_j),
                    "combo_beta_joint": float(rec.beta_combo_joint),
                    "combo_se_joint": float(rec.se_combo_joint),
                    "combo_pwald_joint": float(rec.p_combo_joint),
                }
            )
            out_rows.append(row)
            emitted_singletons.add(key)

        combo_specs = [
            (
                1,
                _garfield_followup_key(rec.chrom1, rec.pos1, rec.combo),
                f"{str(rec.allele0_literal1)},{str(rec.allele0_literal2)}",
                f"{str(rec.allele1_literal1)},{str(rec.allele1_literal2)}",
            ),
            (
                2,
                _garfield_followup_key(rec.chrom2, rec.pos2, rec.combo),
                f"{str(rec.allele0_literal1)},{str(rec.allele0_literal2)}",
                f"{str(rec.allele1_literal1)},{str(rec.allele1_literal2)}",
            ),
        ]
        for component_index, key, allele0, allele1 in combo_specs:
            meta = _garfield_followup_meta_or_fallback(
                row_meta=pseudo_row_meta,
                key=key,
                allele0=allele0,
                allele1=allele1,
            )
            row = dict(rec_dict)
            row.update(
                {
                    "chrom": str(meta["chrom"]),
                    "pos": int(meta["pos"]),
                    "snp": str(meta["snp"]),
                    "allele0": str(meta["allele0"]),
                    "allele1": str(meta["allele1"]),
                    "af": float(meta["af"]),
                    "miss": float(meta["miss"]),
                    "beta": float(rec.beta_combo_joint),
                    "se": float(rec.se_combo_joint),
                    "chisq": _garfield_followup_chisq(rec.beta_combo_joint, rec.se_combo_joint),
                    "pwald": float(rec.p_combo_joint),
                    "row_role": "combo",
                    "component_index": int(component_index),
                    "parent_combo": str(rec.combo),
                    "main_beta_marginal": float("nan"),
                    "main_se_marginal": float("nan"),
                    "main_pwald_marginal": float("nan"),
                    "main_beta_joint_in_parent": float("nan"),
                    "main_se_joint_in_parent": float("nan"),
                    "main_pwald_joint_in_parent": float("nan"),
                    "combo_beta_joint": float(rec.beta_combo_joint),
                    "combo_se_joint": float(rec.se_combo_joint),
                    "combo_pwald_joint": float(rec.p_combo_joint),
                }
            )
            out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    preferred = [
        "chrom",
        "pos",
        "snp",
        "allele0",
        "allele1",
        "af",
        "miss",
        "beta",
        "se",
        "chisq",
        "pwald",
        "row_role",
        "component_index",
        "parent_combo",
        "main_beta_marginal",
        "main_se_marginal",
        "main_pwald_marginal",
        "main_beta_joint_in_parent",
        "main_se_joint_in_parent",
        "main_pwald_joint_in_parent",
        "combo_beta_joint",
        "combo_se_joint",
        "combo_pwald_joint",
    ]
    ordered = [col for col in preferred if col in out_df.columns]
    ordered.extend([col for col in out_df.columns if col not in ordered])
    return out_df.loc[:, ordered]


def _run_garfield_pseudo_fvlmm2(
    *,
    genofile: str,
    pseudo_prefix: str,
    rules_tsv: str,
    trait_label: str,
    pheno_values: np.ndarray,
    sample_ids: list[str],
    sample_indices_full: np.ndarray,
    trait_grm: np.ndarray,
    trait_cov: Optional[np.ndarray],
    source_packed_ctx: dict[str, object],
    source_sites: list[object],
    source_name_map: dict[str, list[int]],
    maf_threshold: float,
    max_missing_rate: float,
    threads: int,
    logger,
    use_spinner: bool,
) -> dict[str, object]:
    _fvlmm2_require_rust_backend()
    specs, rule_meta_df, skipped = _collect_garfield_fvlmm2_specs(
        rules_tsv=str(rules_tsv),
        name_map=source_name_map,
    )
    out_base = f"{pseudo_prefix}.{trait_label}.fvlmm2"
    tsv_path = f"{out_base}.tsv"
    skipped_tsv_path = f"{out_base}.skip"
    figure_path = f"{out_base}.svg"
    saved_paths: list[str] = []

    if len(skipped) > 0:
        pd.DataFrame(skipped).to_csv(skipped_tsv_path, sep="\t", index=False)
        saved_paths.append(skipped_tsv_path)
        logger.warning(
            "GARFIELD pseudo FvLMM2 skipped %d rule(s); details saved to %s",
            len(skipped),
            format_path_for_display(skipped_tsv_path),
        )

    if len(specs) == 0:
        logger.warning(
            "GARFIELD pseudo FvLMM2 found no pairwise rules that can be jointly rechecked for '%s'.",
            trait_label,
        )
        return {
            "saved_paths": saved_paths,
            "tsv_paths": [p for p in saved_paths if str(p).lower().endswith(".tsv")],
            "figure_paths": [],
            "summary_rows": [],
            "tsv_path": None,
            "raw_tsv_path": None,
            "expanded_tsv_path": None,
            "skipped_tsv_path": skipped_tsv_path if len(skipped) > 0 else None,
            "figure_path": None,
            "n_rules_supported": 0,
            "n_rules_skipped": len(skipped),
        }

    pheno_df = pd.DataFrame(
        {str(trait_label): np.asarray(pheno_values, dtype=np.float64)},
        index=pd.Index(np.asarray(sample_ids, dtype=str), dtype=str),
    )
    full_df = _fvlmm2_run_trait_scan(
        trait_ref=str(trait_label),
        pheno_base=pheno_df,
        base_ids=np.asarray(sample_ids, dtype=str),
        base_full_indices=np.asarray(sample_indices_full, dtype=np.int64),
        base_grm=np.asarray(trait_grm, dtype=np.float64, order="C"),
        base_cov=(
            None
            if trait_cov is None
            else np.asarray(trait_cov, dtype=np.float64, order="C")
        ),
        packed_ctx=source_packed_ctx,
        sites=source_sites,
        specs=specs,
        threads=int(threads),
        batch_size=4096,
    )
    if rule_meta_df.shape[0] > 0:
        full_df = full_df.merge(rule_meta_df, on="combo", how="left", sort=False)
    tsv_df = _fvlmm2_format_output_df_for_tsv(_fvlmm2_compact_output_df(full_df))
    tsv_df.to_csv(tsv_path, sep="\t", index=False)
    saved_paths.append(tsv_path)

    pseudo_row_meta = _load_garfield_pseudo_row_meta(
        pseudo_prefix=str(pseudo_prefix),
        expected_sample_ids=list(sample_ids),
        maf_threshold=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
    )
    expanded_df = _build_garfield_fvlmm2_expanded_df(
        raw_df=full_df,
        pseudo_row_meta=pseudo_row_meta,
    )
    plot_tsv_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"{Path(str(pseudo_prefix)).name}.{str(trait_label)}.",
            suffix=".fvlmm2.plot.tsv",
            delete=False,
            encoding="utf-8",
            newline="",
        ) as tmp_handle:
            plot_tsv_path = str(tmp_handle.name)
        expanded_df.to_csv(plot_tsv_path, sep="\t", index=False)
        _run_fastplot_from_tsv_with_status(
            plot_tsv_path,
            np.asarray(pheno_values, dtype=np.float64),
            xlabel=str(trait_label),
            outpdf=figure_path,
            use_spinner=bool(use_spinner),
            emit_done_line=False,
        )
    finally:
        _remove_file_if_exists(plot_tsv_path)
    saved_paths.append(figure_path)

    summary_rows: list[dict[str, object]] = []
    best_combo = None
    best_combo_joint_p = float("nan")
    best_combo_marginal_p = float("nan")
    if full_df.shape[0] > 0 and "p_combo_joint" in full_df.columns:
        p_joint = pd.to_numeric(full_df["p_combo_joint"], errors="coerce")
        if p_joint.notna().any():
            best_idx = int(p_joint.idxmin())
            best_row = full_df.loc[best_idx]
            best_combo = str(best_row.get("combo", ""))
            best_combo_joint_p = float(best_row.get("p_combo_joint", float("nan")))
            best_combo_marginal_p = float(best_row.get("p_combo_marginal", float("nan")))
    summary_rows.append(
        {
            "trait": str(trait_label),
            "n_pairs_tested": int(full_df.shape[0]),
            "n_rules_skipped": int(len(skipped)),
            "best_combo": best_combo,
            "best_combo_joint_p": best_combo_joint_p,
            "best_combo_marginal_p": best_combo_marginal_p,
            "route": "fvlmm2",
        }
    )
    return {
        "saved_paths": saved_paths,
        "tsv_paths": [p for p in saved_paths if str(p).lower().endswith(".tsv")],
        "figure_paths": [figure_path],
        "summary_rows": summary_rows,
        "tsv_path": tsv_path,
        "raw_tsv_path": None,
        "expanded_tsv_path": None,
        "skipped_tsv_path": skipped_tsv_path if len(skipped) > 0 else None,
        "figure_path": figure_path,
        "n_rules_supported": int(full_df.shape[0]),
        "n_rules_skipped": int(len(skipped)),
    }


def _detect_response_mode(y: np.ndarray) -> tuple[str, np.ndarray, str]:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if y_arr.size == 0:
        raise ValueError("Phenotype vector is empty after sample alignment.")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("Phenotype contains non-finite values after sample alignment.")

    uniq = np.unique(y_arr)
    if uniq.size == 2:
        lo = float(uniq[0])
        hi = float(uniq[1])
        lo_mask = np.isclose(y_arr, lo, rtol=0.0, atol=1e-12)
        hi_mask = np.isclose(y_arr, hi, rtol=0.0, atol=1e-12)
        if np.all(lo_mask | hi_mask):
            y_bin = hi_mask.astype(np.float64)
            note = f", mapped({lo:g}->0,{hi:g}->1)"
            return ("binary", y_bin, note)

    return "continuous", y_arr.astype(np.float64, copy=False), ""


def _read_geneset_lines(path: str) -> list[list[str]]:
    genesets: list[list[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            genes = [g for g in re.split(r"[\s,;]+", line.strip()) if g]
            if genes:
                genesets.append(genes)
    return genesets


def _infer_scan_mode_from_genefile(path: str) -> str:
    genesets = _read_geneset_lines(path)
    if len(genesets) == 0:
        raise ValueError(f"Gene file is empty: {path}")
    return "geneset" if any(len(genes) > 1 for genes in genesets) else "gene"


def _build_interval_groups(
    genefile: str,
    gff3: str,
    extension: int,
    scan_mode: str,
) -> tuple[list[str], list[list[tuple[str, int, int]]]]:
    genesets = _read_geneset_lines(genefile)
    dfgff3 = readanno(gff3, "ID").iloc[:, :4].set_index(3)
    dfgff3 = dfgff3.loc[~dfgff3.index.duplicated()]

    labels: list[str] = []
    groups: list[list[tuple[str, int, int]]] = []

    def _iv(gene: str) -> Optional[tuple[str, int, int]]:
        if gene not in dfgff3.index:
            return None
        chrom = str(dfgff3.loc[gene, 0])
        start = int(dfgff3.loc[gene, 1]) - int(extension)
        end = int(dfgff3.loc[gene, 2]) + int(extension)
        return (chrom, start, end)

    if scan_mode == "gene":
        for genes in genesets:
            for g in genes:
                iv = _iv(g)
                if iv is None:
                    continue
                labels.append(g)
                groups.append([iv])
    elif scan_mode in {"genepair", "geneset"}:
        for genes in genesets:
            if len(genes) <= 1:
                gene = genes[0] if len(genes) == 1 else None
                if not gene:
                    continue
                iv = _iv(gene)
                if iv is None:
                    continue
                labels.append(gene)
                groups.append([iv])
            else:
                ivs = [iv for iv in (_iv(g) for g in genes) if iv is not None]
                if len(ivs) < 2:
                    continue
                labels.append("|".join(genes))
                groups.append(ivs)
    else:
        raise ValueError(f"unsupported scan_mode: {scan_mode}")

    return labels, groups


def _scan_mode_to_logic_unit_kind(scan_mode: str) -> str:
    mode = str(scan_mode).strip().lower()
    if mode == "window":
        return "window"
    if mode == "gene":
        return "gene"
    if mode in {"genepair", "geneset"}:
        return "geneset"
    raise ValueError(f"unsupported scan_mode: {scan_mode}")


def _describe_rank_schedule(rank_score_runtime: str) -> str:
    mode = str(rank_score_runtime).strip().lower()
    if mode == "raw":
        return "all raw"
    m = re.fullmatch(r"gain_from_layer:(\d+)", mode)
    if m is not None:
        gain_start = int(m.group(1))
        if gain_start <= 1:
            return "gain from layer 1 (all gain)"
        return f"raw through layer {gain_start - 1}, gain from layer {gain_start}"
    return mode


def _dedupe_saved_paths(paths: list[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        text = str(path or "").strip()
        if text == "" or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _emit_garfield_saved_paths(
    logger,
    paths: list[object],
    *,
    use_spinner: bool,
) -> None:
    saved = _terminal_saved_result_paths(_dedupe_saved_paths(paths))
    if len(saved) == 0:
        return
    if len(saved) == 1:
        _rich_success(
            logger,
            f"Results saved to {format_path_for_display(saved[0])}",
            use_spinner=bool(use_spinner),
        )
        return
    body = "\n".join(f"  {format_path_for_display(path)}" for path in saved)
    _rich_success(
        logger,
        f"Results saved:\n{body}",
        use_spinner=bool(use_spinner),
    )


def _emit_garfield_summary_to_log(logger, summary_rows: list[dict[str, object]]) -> None:
    if len(summary_rows) == 0:
        return
    headers = ["trait", "n_samples", "best_score", "route"]
    rows: list[list[str]] = []
    for row in summary_rows:
        rows.append(
            [
                str(row.get("trait", "")),
                f"{int(row.get('n_samples', 0))}",
                f"{float(row.get('best_score', float('nan'))):.12g}",
                str(row.get("route", "")),
            ]
        )
    widths = [len(x) for x in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    logger.info("")
    logger.info("[ GARFIELD Summary ]")
    logger.info("  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    for row in rows:
        logger.info("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(row))))


def _load_json_if_exists(path: Optional[str]):
    if path is None:
        return None
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _remove_file_if_exists(path: Optional[str]) -> None:
    if path is None:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        return


def _garfield_pseudo_pmeta_paths(
    *,
    pseudo_prefix: str,
    maf_threshold: float,
    max_missing_rate: float,
) -> list[str]:
    cache_prefix = packed_meta_cache_prefix(
        str(pseudo_prefix),
        maf=float(maf_threshold),
        missing_rate=float(max_missing_rate),
        het_threshold=0.0,
        snps_only=False,
    )
    stem = f"{cache_prefix}.pmeta"
    return [
        f"{stem}.json",
        f"{stem}.site_keep.npy",
        f"{stem}.row_missing.npy",
        f"{stem}.row_maf.npy",
        f"{stem}.row_flip.npy",
        f"{stem}.std_denom.npy",
        f"{stem}.dom_af.npy",
    ]


def _split_structure_prior_payload(payload):
    if not isinstance(payload, dict):
        return (None, payload)
    prior_payload = payload.get("prior")
    posterior_payload = dict(payload)
    posterior_payload.pop("prior", None)
    return (prior_payload, posterior_payload)


class _GarfieldStageProgress:
    _HIDDEN_STAGES = {"null_prep", "structure_prep"}

    @classmethod
    def is_hidden_stage(cls, stage: object) -> bool:
        return str(stage).strip().lower() in cls._HIDDEN_STAGES

    def __init__(self, *, scan_desc: str, enabled: bool) -> None:
        self.scan_desc = str(scan_desc)
        self.enabled = bool(enabled)
        self._rich = None
        self._tasks: dict[str, int] = {}
        self._current_stage: str | None = None
        self._adapter: ProgressAdapter | None = None
        self._adapter_done = 0
        if self.enabled and rich_progress_available():
            self._rich = build_rich_progress(
                show_spinner=True,
                show_bar=True,
                show_percentage=True,
                show_elapsed=True,
                show_remaining=True,
                field_templates=["{task.fields[postfix]}"],
                finished_text=f"[green]{success_symbol()}[/green]",
                transient=False,
            )
            if self._rich is not None:
                self._rich.start()

    def _label(self, stage: str) -> str:
        if stage == "null_prep":
            return "Preparing Bg Noise"
        if stage == "null_penalty":
            return "Estimating Bg Noise"
        if stage == "structure_prep":
            return "Preparing Priors"
        if stage == "structure_prior":
            return "Learning Priors"
        return self.scan_desc

    def update(self, stage: str, done: int, total: int, meta: object | None = None) -> None:
        stage_key = str(stage).strip().lower()
        if stage_key in self._HIDDEN_STAGES:
            return
        done_i = max(0, int(done))
        total_i = max(done_i, int(total))
        postfix = "" if meta in {None, ""} else str(meta)
        label = self._label(stage_key)

        if self._rich is not None:
            task_id = self._tasks.get(stage_key)
            if task_id is None:
                task_id = self._rich.add_task(
                    label,
                    total=max(1, total_i),
                    completed=min(done_i, max(1, total_i)),
                    postfix=postfix,
                )
                self._tasks[stage_key] = task_id
            else:
                self._rich.update(
                    task_id,
                    description=label,
                    total=max(1, total_i),
                    completed=min(done_i, max(1, total_i)),
                    postfix=postfix,
                )
            return

        if not self.enabled:
            return
        if self._current_stage != stage_key or self._adapter is None:
            if self._adapter is not None:
                self._adapter.finish()
                self._adapter.close()
            self._adapter = ProgressAdapter(
                total=max(1, total_i),
                desc=label,
                show_spinner=True,
                show_postfix=True,
                show_remaining=True,
                emit_done=True,
                force_animate=True,
            )
            self._current_stage = stage_key
            self._adapter_done = 0
        else:
            self._adapter.set_total(max(1, total_i))

        delta = max(0, done_i - self._adapter_done)
        if delta > 0:
            self._adapter.update(delta)
        if postfix != "":
            self._adapter.set_postfix(progress=f"{done_i}/{total_i}", detail=postfix)
        else:
            self._adapter.set_postfix(progress=f"{done_i}/{total_i}")
        self._adapter_done = done_i

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.finish()
            self._adapter.close()
            self._adapter = None
        if self._rich is not None:
            self._rich.stop()
            self._rich = None


def _run_scan_with_progress(
    desc: str,
    *,
    use_spinner: bool,
    invoke,
):
    if not bool(use_spinner):
        with CliStatus(f"{desc}...", enabled=False, timeout=0.1):
            return invoke(None)

    prepare_desc = "Preparing GARFIELD input"
    prepare_status = CliStatus(
        f"{prepare_desc}...",
        enabled=bool(use_spinner),
        timeout=0.08,
        show_elapsed=True,
        force_animate=True,
    )
    stage_progress: _GarfieldStageProgress | None = None
    prepare_done = False

    def _finish_prepare() -> None:
        nonlocal prepare_done
        if prepare_done:
            return
        prepare_status.complete(f"{prepare_desc} ...Finished")
        prepare_done = True

    def _ensure_stage_progress() -> _GarfieldStageProgress:
        nonlocal stage_progress
        if stage_progress is None:
            _finish_prepare()
            stage_progress = _GarfieldStageProgress(
                scan_desc=desc,
                enabled=bool(use_spinner),
            )
        return stage_progress

    def _progress_cb(*event) -> None:
        if len(event) == 4:
            stage, done, total, meta = event
            stage_name = str(stage)
            if stage_progress is None and _GarfieldStageProgress.is_hidden_stage(stage_name):
                return
            _ensure_stage_progress().update(stage_name, int(done), int(total), meta)
            return
        if len(event) == 2:
            done, total = event
            _ensure_stage_progress().update("scan", int(done), int(total), None)
            return
        raise ValueError(f"unexpected GARFIELD progress event: {event!r}")

    try:
        with prepare_status:
            out = invoke(_progress_cb)
            _finish_prepare()
    except BaseException:
        if stage_progress is not None:
            stage_progress.close()
            print_failure(f"{desc} ...Failed", force_color=True)
        else:
            prepare_status.fail(f"{prepare_desc} ...Failed")
        raise

    if stage_progress is not None:
        stage_progress.close()
    return out


def main() -> None:
    _require_rust_backend()

    t_start = time.time()
    use_spinner = stdout_is_tty()

    parser = CliArgumentParser(prog="jx garfield", formatter_class=cli_help_formatter())

    required_group = parser.add_argument_group("Required Arguments")
    add_common_genotype_source_args(
        required_group,
        include_vcf=False,
        include_hmp=False,
        include_file=False,
        include_bfile=True,
        help_profile="plink_prefix_short",
    )
    add_common_pheno_arg(required_group, required=False, help_text="Phenotype file.")

    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-Window",
        "--Window",
        action="store_true",
        dest="mode_window",
        help="Window scan mode.",
    )
    optional_group.add_argument(
        "-g",
        "--genefile",
        type=str,
        default=None,
        help=(
            "Gene or gene-set file. If provided together with -gff/--gff3, GARFIELD "
            "auto-switches out of window mode: single-gene lines are treated as Gene units, "
            "multi-gene lines as GeneSet units."
        ),
    )
    optional_group.add_argument("-gff", "--gff3", type=str, default=None, help="GFF3 annotation.")
    optional_group.add_argument(
        "--scan-mode",
        type=str,
        choices=["window", "gene", "genepair", "geneset"],
        default=None,
        help=argparse.SUPPRESS,
    )
    add_common_trait_selector_args(optional_group, dest="ncol")
    optional_group.add_argument("-ext", "--extension", type=int, default=100_000, help="Window extension in bp.")
    optional_group.add_argument("-step", "--step", dest="step", type=int, default=None, help="Window step size (default: extension/2).")
    add_common_variant_filter_args(
        optional_group,
        help_profile="pureline",
        include_maf=True,
        include_geno=True,
        include_het=False,
        maf_default=0.02,
        geno_default=0.05,
    )
    optional_group.add_argument(
        "-dev",
        "--dev",
        action="store_true",
        help="Deprecated compatibility flag. Under pure-line GARFIELD it aliases BIN encoding.",
    )
    add_common_grm_file_arg(
        optional_group,
        default=None,
        dest="grm",
        help_profile="garfield_residualization",
    )
    add_common_covariate_file_or_site_arg(
        optional_group,
        dest="cov_inputs",
        default=None,
    )
    optional_group.add_argument(
        "-engine",
        "--engine",
        type=str.upper,
        choices=["AUTO", "CORR", "RF", "GBDT", "GBDT2", "NONE", "SKIP", "DIRECT"],
        default="CORR",
        help=(
            "ML engine for candidate search. Default: CORR. "
            "Use NONE/SKIP/DIRECT to bypass ML and pass all unit variants directly to beam search."
        ),
    )
    optional_group.add_argument(
        "-width",
        "--width",
        type=int,
        default=None,
        help="Unified width controlling both ML top-k and beam width (default: 100).",
    )
    optional_group.add_argument(
        "-permutation",
        "--permutation",
        action="store_true",
        dest="permutation",
        help=(
            "Enable rule-level permutation null calibration on representative scan units "
            "(up to 32 units; if fewer are available, use all of them)."
        ),
    )
    optional_group.add_argument(
        "--fold",
        type=int,
        default=0,
        help=(
            "Holdout split used to separate GARFIELD rule discovery from pseudo-GWAS evaluation. "
            "Use values >=2 to enable a 1/fold test split; 0 or 1 keeps the legacy full-data mode "
            "and may substantially inflate pseudo-GWAS statistics."
        ),
    )
    optional_group.add_argument(
        "-no-clean",
        "--no-clean",
        action="store_true",
        dest="no_clean",
        help="Disable structured beam pruning and fall back to the legacy unfiltered fixed-width search.",
    )
    optional_group.add_argument(
        "-global",
        "--global",
        dest="global_stats",
        action="store_true",
        default=False,
        help=(
            "Compute GARFIELD pure-line row statistics once on the full overlapping sample pool "
            "and reuse the keep mask across traits. Default is per-trait row statistics."
        ),
    )
    optional_group.add_argument("--prior-not", type=float, default=None, help=argparse.SUPPRESS)
    optional_group.add_argument("-layer", "--layer", type=int, default=None, help="Maximum beam-search rule depth (default: 4).")
    optional_group.add_argument(
        "-bimrange",
        "--bimrange",
        type=str,
        action="append",
        default=None,
        help=(
            "Restrict only the final scan stage to one or more genomic bp intervals. "
            "Repeat the flag or use comma-separated items, e.g. "
            "--bimrange 10:110800000-111200000,10:112000000-112200000. "
            "Background-noise calibration remains genome-wide."
        ),
    )
    optional_group.add_argument("-m", "--max-pick", type=int, default=None, dest="layer_compat", help=argparse.SUPPRESS)
    optional_group.add_argument(
        "--feature-source",
        type=str,
        choices=["bin", "mbin"],
        default=None,
        dest="feature_source_compat",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument("--seed", type=int, default=42, help="Random seed.")
    add_common_thread_arg(optional_group, default_threads=detect_effective_threads(), help_profile="cpu_short")
    optional_group.add_argument("--threads", dest="thread", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    add_common_out_arg(optional_group, default=".", help_profile="simple")
    add_common_prefix_arg(optional_group, default=None, help_profile="simple")
    optional_group.add_argument(
        "-simbench",
        "--simbench",
        type=str,
        default=None,
        help="Optional simulation benchmark TSV (<prefix>.fixed.effects.tsv). Matching simulation combinations will be appended to GARFIELD rules output using the same residualized-phenotype statistics.",
    )

    args, extras = parser.parse_known_args()

    has_genotype = bool(args.bfile)
    has_pheno = bool(args.pheno)
    if not has_genotype and not has_pheno:
        parser.error("the following arguments are required: -p/--pheno and -bfile/--bfile")
    if not has_genotype:
        parser.error("the following arguments are required: -bfile/--bfile")
    if not has_pheno:
        parser.error("the following arguments are required: -p/--pheno")
    if len(extras) > 0:
        parser.error("unrecognized arguments: " + " ".join(extras))

    if int(args.extension) <= 0:
        parser.error("-ext/--extension must be > 0")
    if args.step is None:
        args.step = max(1, int(args.extension) // 2)
    elif int(args.step) <= 0:
        parser.error("-step/--step must be > 0")
    if not (0.0 <= float(args.maf) <= 0.5):
        parser.error("-maf/--maf must be in [0, 0.5]")
    if not (0.0 <= float(args.geno) <= 1.0):
        parser.error("-geno/--geno must be in [0, 1]")
    if (
        args.grm is not None
        and str(args.grm).strip().isdigit()
        and not os.path.exists(str(args.grm).strip())
    ):
        parser.error("-k/--grm now expects a GRM path.")

    if args.mode_window and args.genefile is not None:
        parser.error("-Window cannot be combined with -g/--genefile.")

    if args.genefile is not None:
        if not args.gff3:
            parser.error("-g/--genefile requires -gff/--gff3.")
        try:
            args.scan_mode = _infer_scan_mode_from_genefile(args.genefile)
        except ValueError as e:
            parser.error(str(e))
    else:
        args.scan_mode = "window"

    args.width = int(args.width) if args.width is not None else 100
    if int(args.width) <= 0:
        parser.error("-width/--width must be > 0")
    args.beam_width = int(args.width)

    args.layer = (
        int(args.layer) if args.layer is not None
        else int(args.layer_compat) if args.layer_compat is not None
        else 4
    )
    if int(args.layer) <= 0:
        parser.error("-layer must be > 0")
    args.exhaustive_depth_runtime = 2 if int(args.layer) >= 2 else 1
    if args.prior_not is not None:
        try:
            if not np.isfinite(float(args.prior_not)):
                parser.error("--prior-not must be finite when provided")
        except Exception:
            parser.error("--prior-not must be finite when provided")

    if args.engine is not None:
        args.engine = str(args.engine).upper()
        if args.engine == "AUTO":
            args.engine = "CORR"

    ml_skip_tokens = {"NONE", "SKIP", "DIRECT"}
    args.topk = int(args.width) if args.engine not in ml_skip_tokens else 0
    args.top_rules_runtime = 1
    args.max_output_rules_runtime = 0
    args.max_output_ratio_runtime = 0.0

    args.feature_source = (
        str(args.feature_source_compat).lower()
        if args.feature_source_compat is not None
        else ("mbin" if bool(args.dev) else "bin")
    )
    if args.feature_source not in {"bin", "mbin"}:
        parser.error("--feature-source must be one of: bin, mbin")
    args.feature_source_requested = str(args.feature_source)
    if args.feature_source == "mbin":
        args.feature_source = "bin"

    args.rank_score = "gain_from_layer:2"
    args.rank_schedule_source = "fixed-combination-gain"
    args.gain_start_layer_runtime = 2

    try:
        args.ncol = parse_trait_selector_specs(args.ncol, label="-n/--n")
    except ValueError as e:
        parser.error(str(e))

    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    if int(args.thread) <= 0:
        args.thread = int(detected_threads)
    if int(args.thread) > int(detected_threads):
        args.thread = int(detected_threads)

    gfile, prefix = determine_genotype_source(args)
    input_kind = "bfile"

    args.out = os.path.normpath(args.out if args.out is not None else ".")
    outprefix = os.path.join(args.out, prefix)
    os.makedirs(args.out, mode=0o755, exist_ok=True)
    configure_genotype_cache_from_out(args.out)

    log_path = os.path.join(args.out, f"{prefix}.garfield.log")
    logger = setup_logging(log_path)
    apply_outer_thread_cap(int(args.thread))
    if args.feature_source_requested == "mbin":
        logger.warning(
            "MBIN encoding is deprecated for GARFIELD pure-line mode; using BIN instead "
            "(heterozygotes and NA are both treated as missing before binary decoding)."
        )
    if int(args.fold) < 2:
        logger.warning(
            "GARFIELD is running with fold < 2, so rule discovery and pseudo-GWAS follow-up "
            "reuse the same samples. This can substantially inflate pseudo-GWAS statistics; "
            "use --fold >= 2 to enable a holdout split."
        )

    ml_skipped = args.engine in ml_skip_tokens
    engine_runtime = "none" if ml_skipped else str(args.engine)
    rank_score_runtime = str(args.rank_score)
    rank_schedule_source = str(args.rank_schedule_source)
    gain_start_layer_runtime = (
        None if args.gain_start_layer_runtime is None else int(args.gain_start_layer_runtime)
    )
    rank_schedule_runtime = _describe_rank_schedule(rank_score_runtime)

    gff3_effective = args.gff3 if args.genefile else None

    checks: list[bool] = []
    checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    checks.append(ensure_file_exists(logger, args.pheno, "Phenotype file"))
    if args.grm:
        checks.append(ensure_file_exists(logger, args.grm, "GARFIELD GRM"))
    if args.genefile:
        checks.append(ensure_file_exists(logger, args.genefile, "Gene file"))
    if gff3_effective:
        checks.append(ensure_file_exists(logger, gff3_effective, "GFF3 file"))
    if args.simbench:
        checks.append(ensure_file_exists(logger, args.simbench, "Simulation benchmark TSV"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    feature_source = str(args.feature_source).lower()
    if feature_source not in {"bin", "mbin"}:
        raise ValueError(f"Unsupported feature-source: {args.feature_source}")

    general_rows = [
        ("Genotype input", gfile),
        ("Input kind", input_kind),
        ("Execution route", "direct Rust BED pipeline"),
        (
            "Prepare reuse",
            (
                "global site_keep once + trait-local load"
                if bool(args.global_stats)
                else "trait-specific site_keep + trait-local load"
            ),
        ),
        ("Encoding", feature_source),
        ("Residualization GRM", args.grm if args.grm else "auto from genotype"),
        ("Covariates", None if not args.cov_inputs else ",".join(str(x) for x in args.cov_inputs)),
        ("Phenotype", args.pheno),
        ("Scan mode", args.scan_mode),
        ("Gene file", args.genefile),
        ("GFF3", gff3_effective if gff3_effective else ("ignored" if args.gff3 else None)),
        ("MAF", float(args.maf)),
        ("Missing max (1+NA)", float(args.geno)),
        ("Extension", int(args.extension)),
        ("Step", int(args.step)),
        ("Bimrange", None if not args.bimrange else ",".join(str(x) for x in args.bimrange)),
        ("Split", "none (full data)"),
        ("Engine", "none (skip ML)" if ml_skipped else args.engine),
        ("Permutation", bool(args.permutation)),
        ("Pseudo GWAS", "FvLMM follow-up"),
        ("Structured pruning", not bool(args.no_clean)),
        ("NOT control", "null penalty only"),
        ("Width", int(args.width)),
        ("Rules/unit", int(args.top_rules_runtime)),
        ("Layer", int(args.layer)),
        ("Pair seed depth", int(args.exhaustive_depth_runtime)),
        ("Rule ranking", rank_schedule_runtime),
        ("Sim bench", args.simbench),
        ("Seed", int(args.seed)),
    ]

    emit_cli_configuration(
        logger,
        app_title="JanusX - GARFIELD",
        config_title="GARFIELD CONFIG",
        host=socket.gethostname(),
        sections=[("General", general_rows)],
        footer_rows=[
            (
                "Threads",
                format_requested_thread_usage(
                    requested_threads=int(requested_threads),
                    using_threads=int(args.thread),
                    detected_threads=int(detected_threads),
                ),
            ),
            ("Output prefix", outprefix),
        ],
        line_max_chars=_gwas_terminal_config_line_max_chars(60),
    )
    # logger.info(
    #     "Rule-ranking resolution: logic_gate=%s, source=%s -> %s",
    #     logic_gate_runtime,
    #     rank_schedule_source,
    #     rank_schedule_runtime,
    # )

    pheno = _load_phenotype_with_status(
        args.pheno,
        args.ncol,
        _GarfieldPhenoLogger(logger),
        id_col=0,
        use_spinner=use_spinner,
    )

    sample_ids, _n_snps = _inspect_genotype_with_status(
        gfile,
        logger,
        use_spinner=use_spinner,
        snps_only=False,
        maf_threshold=float(args.maf),
        max_missing_rate=float(args.geno),
        het_threshold=0.0,
    )
    sample_ids = np.asarray(sample_ids, dtype=str)
    if len(sample_ids) == 0:
        raise ValueError("No sample IDs found in genotype input.")
    sample_index_map = {sid: i for i, sid in enumerate(sample_ids.tolist())}

    aligned_grm = None
    resolved_grm_id = None
    if args.grm:
        grm_src = os.path.basename(str(args.grm))
        with CliStatus(f"Loading GRM from {grm_src}...", enabled=use_spinner) as task:
            try:
                aligned_grm, resolved_grm_id = load_and_align_grm(
                    str(args.grm),
                    sample_ids.tolist(),
                    grm_id_path=None,
                    label="GARFIELD GRM",
                )
            except BaseException:
                task.fail(f"Loading GRM from {grm_src} ...Failed")
                raise
            task.complete(f"Loading GRM from {grm_src} (n={aligned_grm.shape[0]})")

    cov_all, cov_ids = _load_covariates_for_models(
        cov_inputs=args.cov_inputs,
        genofile=gfile,
        sample_ids=sample_ids,
        chunk_size=65536,
        logger=logger,
        context="streaming",
        use_spinner=use_spinner,
        snps_only=False,
    )
    if cov_all is not None:
        cov_all = np.asarray(cov_all, dtype=np.float64, order="C")
    if cov_ids is not None:
        cov_ids = np.asarray(cov_ids, dtype=str)

    if pheno.shape[1] == 0:
        raise ValueError("No phenotype columns to analyze.")
    pheno, trait_names = _normalize_trait_names_from_header(pheno, args.pheno)
    geno_ids = sample_ids.astype(str)
    pheno_ids_all = pheno.index.astype(str).to_numpy()
    common = set(geno_ids) & set(pheno_ids_all)
    if aligned_grm is not None:
        common &= set(geno_ids)
    if cov_ids is not None:
        common &= set(cov_ids.astype(str))
    sample_pool = [sid for sid in geno_ids.tolist() if sid in common]
    if len(sample_pool) == 0:
        raise ValueError("No overlapping samples across genotype/phenotype/GRM/cov.")

    grm_n: int | str = "NA" if aligned_grm is None else int(aligned_grm.shape[0])
    cov_n: int | str = "NA" if cov_ids is None else int(len(cov_ids))
    split_line = "-"*60
    _emit_plain_info_line(
        logger,
        (
            f"geno={len(geno_ids)}, pheno={len(pheno_ids_all)}, "
            f"grm={grm_n}, q=NA, cov={cov_n} -> {len(sample_pool)}"
            f"\n{split_line}"
        ),
        use_spinner=use_spinner,
    )

    cov_index = (
        None
        if cov_ids is None
        else {sid: i for i, sid in enumerate(cov_ids.astype(str).tolist())}
    )
    followup_grm_full = (
        None
        if aligned_grm is None
        else np.asarray(aligned_grm, dtype=np.float64, order="C")
    )
    followup_source_packed_ctx: dict[str, object] | None = None
    followup_source_sites: list[object] | None = None
    followup_source_name_map: dict[str, list[int]] | None = None

    group_labels: list[str] = []
    group_intervals: list[list[tuple[str, int, int]]] = []
    global_site_keep: Optional[np.ndarray] = None
    if bool(args.global_stats):
        global_site_keep = _prepare_site_keep(
            genofile=gfile,
            sample_ids=list(sample_pool),
            sample_index_map=sample_index_map,
            n_snps=int(_n_snps),
            maf_threshold=float(args.maf),
            max_missing_rate=float(args.geno),
            snps_only=False,
            threads=int(args.thread),
            use_spinner=use_spinner,
            global_stats=True,
        )
    if args.scan_mode in {"gene", "geneset"}:
        if not args.genefile or not args.gff3:
            raise ValueError(f"scan-mode={args.scan_mode} requires both --genefile and --gff3.")
        group_labels, group_intervals = _build_interval_groups(
            args.genefile,
            args.gff3,
            int(args.extension),
            args.scan_mode,
        )
        if len(group_intervals) == 0:
            raise ValueError(f"No valid groups built for scan-mode={args.scan_mode}.")
        logger.info(f"Prepared {len(group_intervals)} interval groups for {args.scan_mode} scan.")

    used_trait_labels: dict[str, int] = {}
    saved = 0
    summary_rows: list[dict[str, object]] = []
    garfield_manifest_traits: list[dict[str, object]] = []

    for trait_idx, trait in enumerate(pheno.columns):
        if trait_idx > 0:
            logger.info("")

        trait_name = str(trait)
        pheno_col = pheno[trait].dropna()
        pheno_ids = set(pheno_col.index.astype(str).to_numpy())
        common_ids = [sid for sid in sample_pool if sid in pheno_ids]
        if len(common_ids) == 0:
            logger.warning(f"No overlapping samples for trait '{trait_name}' after dropna; skipped.")
            continue

        y_raw = pheno_col.loc[common_ids].to_numpy(dtype=float)
        response_mode, y_common, response_note = _detect_response_mode(y_raw)
        _emit_plain_info_line(
            logger,
            f"{trait_name} (n={len(common_ids)}, response={response_mode}{response_note})",
            use_spinner=use_spinner,
        )
        site_keep_trait = global_site_keep
        if site_keep_trait is None:
            site_keep_trait = _prepare_site_keep(
                genofile=gfile,
                sample_ids=list(common_ids),
                sample_index_map=sample_index_map,
                n_snps=int(_n_snps),
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
                snps_only=False,
                threads=int(args.thread),
                use_spinner=use_spinner,
                global_stats=False,
            )
        # Binary traits are accepted: LMM residualization produces continuous residuals,
        # and centered-gain scoring is valid on any finite y (including 0/1).
        base_trait = _safe_trait_label(trait_name)
        count = used_trait_labels.get(base_trait, 0) + 1
        used_trait_labels[base_trait] = count
        suffix = base_trait if count == 1 else f"{base_trait}.{count}"
        trait_outprefix = f"{outprefix}.{suffix}"
        trait_seed = int(args.seed) + trait_idx
        trait_grm = None
        if aligned_grm is not None:
            common_positions = np.asarray(
                [sample_index_map[sid] for sid in common_ids],
                dtype=np.intp,
            )
            trait_grm = np.asarray(
                aligned_grm[np.ix_(common_positions, common_positions)],
                dtype=np.float64,
            )
        trait_cov = None
        if cov_all is not None and cov_index is not None:
            cov_take = np.asarray([cov_index[sid] for sid in common_ids], dtype=np.intp)
            trait_cov = np.asarray(cov_all[cov_take, :], dtype=np.float64, order="C")
        scan_desc = {
            "window": "Scan Windows",
            "gene": "Scan Genes",
            "genepair": "Scan Gene Pairs",
            "geneset": "Scan Gene Sets",
        }.get(str(args.scan_mode), f"Rust GARFIELD search for '{trait_name}'")
        logic_unit_kind = _scan_mode_to_logic_unit_kind(args.scan_mode)
        rust_groups = group_intervals if args.scan_mode != "window" else None
        rust_group_names = group_labels if args.scan_mode != "window" else None
        trait_logic_prefix = f"{trait_outprefix}.garfield"
        # Rust handles full-data residualization before ML candidate search
        # and beam search are executed.
        result = _run_scan_with_progress(
            scan_desc,
            use_spinner=use_spinner,
            invoke=lambda progress_cb: garfield_logic_search_bed(
                gfile,
                np.asarray(y_common, dtype=np.float64),
                grm=trait_grm,
                x_cov=trait_cov,
                sample_ids=list(common_ids),
                site_keep=site_keep_trait,
                unit_kind=logic_unit_kind,
                groups=rust_groups,
                group_names=rust_group_names,
                extension=int(args.extension),
                step=int(args.step),
                scan_bimranges=args.bimrange,
                bin_mode=feature_source,
                ml_method=str(engine_runtime).lower(),
                ml_importance="imp",
                ml_top_k=int(args.topk),
                ml_top_frac=0.0,
                permutation_repeats=20,
                permutation_scoring="auto",
                n_estimators=100,
                max_depth=int(args.layer) + 1,
                min_samples_leaf=1,
                min_samples_split=2,
                bootstrap=True,
                feature_subsample=0.0,
                fold=max(0, int(args.fold)),
                seed=trait_seed,
                max_pick=int(args.layer),
                exhaustive_depth=int(args.exhaustive_depth_runtime),
                beam_width=int(args.beam_width),
                rank_score=str(args.rank_score),
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
                snps_only=False,
                block_cols=65536,
                threads=int(args.thread),
                low=-5.0,
                high=5.0,
                max_iter=50,
                tol=1e-3,
                add_intercept=True,
                exact_n_max=15000,
                require_lapack=False,
                out_prefix=trait_logic_prefix,
                simbench_path=args.simbench,
                top_rules_per_unit=int(args.top_rules_runtime),
                max_output_rules=int(args.max_output_rules_runtime),
                max_output_ratio=float(args.max_output_ratio_runtime),
                rule_permutation=bool(args.permutation),
                prior_len=None,
                no_clean=bool(args.no_clean),
                progress_callback=progress_cb,
                progress_every=0,
            ),
        )
        rust_memory_debug = result.get("memory_debug")
        if _garfield_rss_debug_enabled() and isinstance(rust_memory_debug, dict):
            for stage_name in (
                "global_bits_loaded",
                "scan",
                "null_penalty",
                "structure_prior",
            ):
                _emit_garfield_rss_checkpoint(
                    logger,
                    stage_name,
                    rust_memory_debug.get(stage_name),
                )

        pseudo_path = f"{trait_logic_prefix}.pseudo"
        posterior_tsv_path = f"{trait_logic_prefix}.posterior.tsv"
        posterior_json_path = result.get("posterior_json")
        run_config_path = f"{trait_outprefix}.garfield.run_config.json"
        skipped_messages = result.get("skipped_messages") or []
        if len(skipped_messages) > 0:
            logger.warning(
                f"GARFIELD skipped {len(skipped_messages)} unit(s) for trait '{trait_name}'."
            )
            for msg in skipped_messages:
                logger.warning(str(msg))
        n_rules = int(result.get("n_rules", 0))
        if n_rules <= 0:
            _remove_file_if_exists(pseudo_path)
            _remove_file_if_exists(posterior_tsv_path)
            _remove_file_if_exists(run_config_path)
            _remove_file_if_exists(posterior_json_path)
            logger.warning(f"No GARFIELD rules survived Rust search for trait '{trait_name}', skipped.")
            continue

        _remove_file_if_exists(pseudo_path)
        _remove_file_if_exists(posterior_tsv_path)
        split_applied = bool(result.get("split_applied", False))
        prior_payload, posterior_payload = _split_structure_prior_payload(
            _load_json_if_exists(posterior_json_path)
        )
        pseudo_gwas_payload = None
        followup_memory_debug = None
        pseudo_prefix = result.get("pseudo_prefix")
        rules_tsv = result.get("rules_tsv")
        if pseudo_prefix and rules_tsv:
            def _run_followup() -> dict[str, object]:
                nonlocal followup_grm_full
                nonlocal followup_source_packed_ctx
                nonlocal followup_source_sites
                nonlocal followup_source_name_map
                if followup_grm_full is None:
                    followup_grm_full = _ensure_followup_grm(
                        existing_grm=None,
                        genofile=gfile,
                        sample_ids=sample_ids,
                        n_snps=_n_snps,
                        maf_threshold=float(args.maf),
                        max_missing_rate=float(args.geno),
                        threads=int(args.thread),
                        cache_dir=args.out,
                        logger=logger,
                        use_spinner=use_spinner,
                    )
                if (
                    followup_source_packed_ctx is None
                    or followup_source_sites is None
                    or followup_source_name_map is None
                ):
                    _followup_ids, followup_source_packed_ctx = prepare_packed_ctx_from_plink(
                        str(gfile),
                        maf=float(args.maf),
                        missing_rate=float(args.geno),
                        het_threshold=0.0,
                        snps_only=False,
                        filter_mode="compact",
                        use_cache=False,
                    )
                    followup_source_sites, followup_source_name_map = _fvlmm2_load_active_sites(
                        str(gfile),
                        followup_source_packed_ctx,
                    )
                common_positions = np.asarray(
                    [sample_index_map[sid] for sid in common_ids],
                    dtype=np.intp,
                )
                trait_grm_followup = np.asarray(
                    followup_grm_full[np.ix_(common_positions, common_positions)],
                    dtype=np.float64,
                    order="C",
                )
                logger.info(
                    f"Running pseudo FvLMM2 follow-up for '{trait_name}' on {int(n_rules)} GARFIELD rule(s)."
                )
                return _run_garfield_pseudo_fvlmm2(
                    genofile=str(gfile),
                    pseudo_prefix=str(pseudo_prefix),
                    rules_tsv=str(rules_tsv),
                    trait_label=suffix,
                    pheno_values=np.asarray(y_common, dtype=np.float64),
                    sample_ids=list(common_ids),
                    sample_indices_full=common_positions,
                    trait_grm=trait_grm_followup,
                    trait_cov=trait_cov,
                    source_packed_ctx=followup_source_packed_ctx,
                    source_sites=followup_source_sites,
                    source_name_map=followup_source_name_map,
                    maf_threshold=float(args.maf),
                    max_missing_rate=float(args.geno),
                    threads=int(args.thread),
                    logger=logger,
                    use_spinner=use_spinner,
                )
            followup_sampler = (
                _GarfieldRssSampler() if _garfield_rss_debug_enabled() else None
            )
            if followup_sampler is None:
                pseudo_gwas_payload = _run_followup()
            else:
                with followup_sampler:
                    pseudo_gwas_payload = _run_followup()
                followup_memory_debug = followup_sampler.summary()
                _emit_garfield_rss_checkpoint(
                    logger,
                    "follow-up",
                    followup_memory_debug,
                )
        elif pseudo_prefix and not rules_tsv:
            logger.warning(
                "GARFIELD pseudo follow-up skipped for '%s' because rules.tsv is missing.",
                trait_name,
            )
        _remove_file_if_exists(run_config_path)
        _remove_file_if_exists(posterior_json_path)
        trait_memory_debug = (
            dict(rust_memory_debug) if isinstance(rust_memory_debug, dict) else None
        )
        if followup_memory_debug is not None:
            if trait_memory_debug is None:
                trait_memory_debug = {}
            trait_memory_debug["follow_up"] = followup_memory_debug
        trait_manifest = {
            "trait": trait_name,
            "trait_output_prefix": trait_outprefix,
            "garfield_prefix": trait_logic_prefix,
            "pseudo_prefix": pseudo_prefix,
            "route": "rust-bed",
            "split_applied": split_applied,
            "scan_mode": args.scan_mode,
            "unit_kind": logic_unit_kind,
            "response": response_mode,
            "engine": args.engine,
            "engine_runtime": engine_runtime,
            "ml_skipped": ml_skipped,
            "permutation": bool(args.permutation),
            "rule_permutation_active": bool(result.get("rule_permutation_active", False)),
            "null_chunk_bp": int(result.get("null_chunk_bp", 0)),
            "null_chunk_min_snps": int(result.get("null_chunk_min_snps", 0)),
            "null_chunk_target": int(result.get("null_chunk_target", 0)),
            "null_chunk_valid_total": int(result.get("null_chunk_valid_total", 0)),
            "null_chunk_selected": int(result.get("null_chunk_selected", 0)),
            "representative_units_target": int(result.get("representative_units_target", 0)),
            "representative_units_used": int(result.get("representative_units_used", 0)),
            "permutation_null_repeats": int(result.get("permutation_null_repeats", 0)),
            "permutation_bootstrap_repeats": int(
                result.get("permutation_bootstrap_repeats", 0)
            ),
            "no_clean_requested": bool(args.no_clean),
            "structured_pruning": not bool(args.no_clean),
            "width": int(args.width),
            "top_rules_per_unit": int(args.top_rules_runtime),
            "layer": int(args.layer),
            "pair_seed_depth": int(args.exhaustive_depth_runtime),
            "beam_width": int(args.beam_width),
            "not_control": "null_penalty_only",
            "prior_not_ignored": (
                None if args.prior_not is None else float(args.prior_not)
            ),
            "rank_schedule_source": rank_schedule_source,
            "gain_start_layer_runtime": gain_start_layer_runtime,
            "rank_schedule_runtime": rank_schedule_runtime,
            "rank_score": rank_score_runtime,
            "rank_score_runtime": rank_score_runtime,
            "ml_top_k": (None if ml_skipped else int(args.topk)),
            "extension": int(args.extension),
            "step": int(args.step),
            "bimrange": (list(args.bimrange) if args.bimrange else None),
            "ranking_dataset": "full",
            "feature_source": feature_source,
            "grm_path": args.grm,
            "grm_id_path": resolved_grm_id,
            "maf": float(args.maf),
            "geno": float(args.geno),
            "pure_line_missing_rule": "heterozygote_or_na",
            "simbench_path": args.simbench,
            "simbench_rows": int(result.get("n_simbench", 0)),
            "seed": trait_seed,
            "n_samples": len(common_ids),
            "n_train": int(result.get("n_train", 0)),
            "n_test": int(result.get("n_test", 0)),
            "n_rules": int(result.get("n_rules", 0)),
            "units_total": int(result.get("units_total", 0)),
            "units_scanned": int(result.get("units_scanned", 0)),
            "train_pve": float(result.get("train_pve", float("nan"))),
            "test_pve": float(result.get("test_pve", float("nan"))),
            "train_sigma_g2": float(result.get("train_sigma_g2", float("nan"))),
            "train_sigma_e2": float(result.get("train_sigma_e2", float("nan"))),
            "test_sigma_g2": float(result.get("test_sigma_g2", float("nan"))),
            "test_sigma_e2": float(result.get("test_sigma_e2", float("nan"))),
            "memory_debug": trait_memory_debug,
            "outputs": {
                "rules_tsv": result.get("rules_tsv"),
                "pseudo_fvlmm2_raw_tsv": None,
                "pseudo_fvlmm2_tsv": (
                    None
                    if pseudo_gwas_payload is None
                    else pseudo_gwas_payload.get("tsv_path")
                ),
                "pseudo_fvlmm2_skipped_tsv": (
                    None
                    if pseudo_gwas_payload is None
                    else pseudo_gwas_payload.get("skipped_tsv_path")
                ),
                "pseudo_fvlmm2_figure": (
                    None
                    if pseudo_gwas_payload is None
                    else pseudo_gwas_payload.get("figure_path")
                ),
                "pseudo_fastlmm_tsv": (
                    None
                    if pseudo_gwas_payload is None
                    else pseudo_gwas_payload.get("tsv_path")
                ),
                "pseudo_fastlmm_figure": (
                    None
                    if pseudo_gwas_payload is None
                    else pseudo_gwas_payload.get("figure_path")
                ),
            },
            "prior": prior_payload,
            "posterior": posterior_payload,
            "pseudo_fvlmm2": pseudo_gwas_payload,
            "pseudo_fastlmm": pseudo_gwas_payload,
        }
        garfield_manifest_traits.append(trait_manifest)
        _emit_plain_info_line(
            logger,
            (
                f"GARFIELD null-model PVE for '{trait_name}': "
                f"full={float(result.get('train_pve', float('nan'))):.4g}"
            ),
            use_spinner=use_spinner,
        )
        if args.simbench:
            _emit_plain_info_line(
                logger,
                f"GARFIELD simbench rows appended for '{trait_name}': "
                f"{int(result.get('n_simbench', 0))}",
                use_spinner=use_spinner,
            )
        rank_scores = [
            float(x)
            for x in (
                result.get("scores")
                or result.get("test_scores")
                or result.get("train_scores")
                or []
            )
        ]
        best_score = rank_scores[0] if len(rank_scores) > 0 else float("nan")
        summary_rows.append(
            {
                "trait": trait_name,
                "n_samples": len(common_ids),
                "best_score": best_score,
                "route": "rust-bed",
            }
        )
        trait_saved_paths: list[object] = [result.get("rules_tsv")]
        if pseudo_gwas_payload is not None:
            trait_saved_paths.extend(pseudo_gwas_payload.get("tsv_paths") or [])
            trait_saved_paths.extend(pseudo_gwas_payload.get("figure_paths") or [])
        _emit_garfield_saved_paths(
            logger,
            trait_saved_paths,
            use_spinner=use_spinner,
        )
        saved += 1

    if saved == 0:
        raise ValueError("No GARFIELD outputs were generated for the selected phenotype columns.")

    aggregate_json_path = f"{outprefix}.garfield.json"
    with open(aggregate_json_path, "w", encoding="utf-8") as fw:
        json.dump(
            {
                "format": "janusx.garfield.summary.v1",
                "log_file": log_path,
                "output_prefix": outprefix,
                "genotype_input": gfile,
                "input_kind": input_kind,
                "execution_route": "direct Rust BED pipeline",
                "encoding": feature_source,
                "phenotype_file": args.pheno,
                "grm_path": args.grm,
                "grm_id_path": resolved_grm_id,
                "scan_mode": args.scan_mode,
                "rank_score_runtime": rank_score_runtime,
                "rank_schedule_runtime": rank_schedule_runtime,
                "rank_schedule_source": rank_schedule_source,
                "permutation": bool(args.permutation),
                "null_chunk_bp": int(args.extension) * 2,
                "null_chunk_target": 150 if bool(args.permutation) else 0,
                "null_chunk_min_snps": 50 if bool(args.permutation) else 0,
                "bimrange": (list(args.bimrange) if args.bimrange else None),
                "top_rules_per_unit": int(args.top_rules_runtime),
                "pair_seed_depth": int(args.exhaustive_depth_runtime),
                "not_control": "null_penalty_only",
                "prior_not_ignored": (
                    None if args.prior_not is None else float(args.prior_not)
                ),
                "thread": int(args.thread),
                "seed": int(args.seed),
                "summary_rows": summary_rows,
                "traits": garfield_manifest_traits,
            },
            fw,
            indent=2,
            ensure_ascii=False,
        )
    _emit_garfield_saved_paths(
        logger,
        [aggregate_json_path],
        use_spinner=use_spinner,
    )
    _emit_garfield_summary_to_log(logger, summary_rows)

    _rich_success(
        logger,
        f"\nFinished. Total wall time: {round(time.time() - t_start, 2)} seconds\n"
        f"{_format_cli_finished_timestamp()}",
        use_spinner=use_spinner,
    )


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger().info("Interrupted by user (Ctrl+C).")
        if os.name == "nt":
            try:
                logging.getLogger().info(
                    "Terminating all GARFIELD workers immediately on Windows."
                )
            except Exception:
                pass
            try:
                logging.shutdown()
            finally:
                os._exit(130)
        raise SystemExit(130)
