import argparse
import json
import logging
import os
from pathlib import Path
import re
import shlex
import socket
import sys
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
    prepare_packed_ctx_from_plink,
)
from janusx.script._common.genocache import configure_genotype_cache_from_out
from janusx.script._common.outprefix import apply_output_prefix_compat
from janusx.script._common.grmio import (
    format_grm_cache_num, load_and_align_grm, read_id_file, resolve_grm_id_path,
)
from janusx.script._common.cli_core import CliArgumentParser, cli_help_formatter
from janusx.script._common.log import setup_logging
from janusx.script._common.memory import (
    resolve_decode_mmap_window_mb as _common_resolve_decode_mmap_window_mb,
)
from janusx.script._common.max6_display import (
    MAX6_BONFERRONI_LOADING_DESC,
    MAX6_LD_TAG_LOADING_DESC,
    format_max6_count,
    pwald_bonferroni_reference,
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
    format_elapsed,
    print_failure,
    print_success,
    rich_progress_available,
    stdout_is_tty,
)
from janusx.script._common.threads import (
    apply_outer_thread_cap,
    detect_effective_threads,
    format_requested_thread_usage,
)
from janusx.assoc.workflow_ui import _emit_plain_info_line, _rich_success
from janusx.assoc.workflow_ui import _run_fastplot_from_tsv_with_status
from janusx.pyBLUP.assoc import FvLMM
from janusx.script.fvlmm2 import (
    _decode_rows as _fvlmm2_decode_rows,
    _load_active_sites as _fvlmm2_load_active_sites,
)


try:
    from janusx.janusx import (
        finalize_max6_genotype_universe,
        garfield_logic_search_bed,
        prepare_max6_genotype_tags,
    )
except Exception as _exc:  # pragma: no cover
    garfield_logic_search_bed = None  # type: ignore[assignment]
    finalize_max6_genotype_universe = None  # type: ignore[assignment]
    prepare_max6_genotype_tags = None  # type: ignore[assignment]
    _RUST_IMPORT_ERROR = _exc
else:
    _RUST_IMPORT_ERROR = None

# Python 仅负责 CLI 调度；训练/测试划分、null-model 残差化、LD-tag
# exhaustive Max6/GRM-Max6 扫描与最终导出都由 Rust 端统一完成。


def _current_bed_memory_mb() -> float:
    try:
        mb = float(os.environ.get("JX_BED_BLOCK_TARGET_MB", "512"))
    except Exception:
        return 512.0
    return mb if np.isfinite(mb) and mb > 0.0 else 512.0


def _env_truthy(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _resolve_garfield_xor_search(requested: bool) -> bool:
    """Resolve the explicit XOR-search opt-in and legacy force-off override."""
    return bool(requested) and not _env_truthy("JX_GARFIELD_DISABLE_XOR_SEARCH")


def _is_max6_grm_backend(search_backend: object) -> bool:
    return str(search_backend).strip().lower() in {
        "max6-grmmax6",
        "max6-grmmax6-legacy",
    }


def _is_ld_tag_max6_backend(search_backend: object) -> bool:
    return str(search_backend).strip().lower() == "max6-grmmax6"


def _resolve_garfield_grm_int_mode(
    requested: bool | None,
    search_backend: object,
) -> tuple[bool, str]:
    if requested is None:
        enabled = _is_ld_tag_max6_backend(search_backend)
        return enabled, "default_max6_grmmax6" if enabled else "default_off_for_route"
    return bool(requested), "explicit_enabled" if requested else "explicit_disabled"


def _resolve_garfield_null_resampling(
    search_backend: object,
    explicit_permutation_spec: object,
    max6_audit_requested: bool,
) -> bool:
    if _is_ld_tag_max6_backend(search_backend):
        if explicit_permutation_spec is not None:
            raise ValueError(
                "max6-grmmax6 uses an analytical Bonferroni cutoff; -pm/--permutation is unsupported"
            )
        return bool(max6_audit_requested)
    return True


def _garfield_rss_debug_enabled() -> bool:
    return _env_truthy("JX_GARFIELD_RSS_DEBUG")


def _resolve_logic_maf_threshold(
    requested_lmaf: Optional[float],
    n_effective_samples: int,
) -> tuple[float, str]:
    if requested_lmaf is not None:
        value = float(requested_lmaf)
        return value, "manual"
    if int(n_effective_samples) <= 0:
        return 0.5, "auto(30/n_valid)"
    value = min(0.5, max(0.0, 30.0 / float(n_effective_samples)))
    return value, f"auto(30/{int(n_effective_samples)})"


def _copy_prefixed_result_fields(
    payload: object,
    prefixes: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    copied: dict[str, object] = {}
    for key, value in payload.items():
        text = str(key)
        if any(text.startswith(prefix) for prefix in prefixes):
            copied[text] = value
    return copied


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


def _format_max6_score_value(value: object) -> str:
    """Format Max6 output scores without hiding very small values."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(score):
        return "NA"
    if abs(score) < 0.0001:
        mantissa, exponent = f"{score:.4e}".split("e", 1)
        return f"{mantissa}e{int(exponent)}"
    return f"{score:.4f}"


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
    het_threshold: float,
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
    cache_prefix = f"{cache_prefix}.het{format_grm_cache_num(float(het_threshold))}"
    grm_all, _eff_m, grm_ids, _grm_cache_path = load_or_build_grm_with_cache(
        genofile=genofile,
        cache_prefix=cache_prefix,
        mgrm="1",
        maf_threshold=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
        het_threshold=float(het_threshold),
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
    het_threshold: float,
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
                het_threshold=float(het_threshold),
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


def _garfield_followup_row_role(snp: object) -> str:
    token = str(snp).strip()
    return "combo" if any(op in token for op in ("&", "|", "*", "^")) else "singleton"


def _garfield_format_fixed4(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.4f}"


def _garfield_normalize_sci_text(text: str) -> str:
    mantissa, exp = text.split("e", 1)
    mantissa = mantissa.rstrip("0").rstrip(".")
    if mantissa in {"", "-0", "+0"}:
        mantissa = "0"
    return f"{mantissa}e{int(exp)}"


def _garfield_format_sci4(value: object, *, blank_if_nan: bool = False) -> str:
    try:
        v = float(value)
    except Exception:
        return "" if blank_if_nan else "NA"
    if not np.isfinite(v):
        return "" if blank_if_nan else "NA"
    return _garfield_normalize_sci_text(f"{v:.4e}")


def _garfield_format_metric4(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    if v == 0.0:
        return "0"
    if 1e-4 <= abs(v) < 1e4:
        return f"{v:.4f}"
    return _garfield_normalize_sci_text(f"{v:.4e}")


def _garfield_format_pos_int(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return str(int(round(v)))


def _format_garfield_fvlmm_output_df_for_tsv(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [
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
    ]
    if df.shape[0] == 0:
        return pd.DataFrame(columns=ordered)
    out = df.copy()
    out = out.drop(columns=["row_role", "padj"], errors="ignore")
    if "chrom" in out.columns:
        out["chrom"] = out["chrom"].astype(str)
    if "pos" in out.columns:
        out["pos"] = [_garfield_format_pos_int(v) for v in out["pos"]]
    for col in ("af", "miss", "beta", "se"):
        if col in out.columns:
            out[col] = [_garfield_format_fixed4(v) for v in out[col]]
    for col in ("chisq", "pwald"):
        if col in out.columns:
            out[col] = [_garfield_format_sci4(v) for v in out[col]]
    ordered_present = [col for col in ordered if col in out.columns]
    ordered_present.extend([col for col in out.columns if col not in ordered_present])
    return out.loc[:, ordered_present]


def _sort_garfield_fvlmm_rows_for_tsv(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[0] == 0:
        return df.copy()
    out = df.copy()
    sort_cols: list[str] = []
    if "pwald" in out.columns:
        out["__sort_pwald"] = pd.to_numeric(out["pwald"], errors="coerce")
        sort_cols.append("__sort_pwald")
    if "chrom" in out.columns:
        sort_cols.append("chrom")
    if "pos" in out.columns:
        out["__sort_pos"] = pd.to_numeric(out["pos"], errors="coerce")
        sort_cols.append("__sort_pos")
    if "snp" in out.columns:
        sort_cols.append("snp")
    if len(sort_cols) == 0:
        return out
    out = out.sort_values(by=sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)
    return out.drop(columns=["__sort_pwald", "__sort_pos"], errors="ignore")


def _garfield_invocation_command() -> str:
    tokens = [str(x) for x in sys.argv[1:]]
    prog_raw = str(sys.argv[0]).strip() if len(sys.argv) > 0 else ""
    if prog_raw == "":
        prog_raw = "jx garfield"
    try:
        prog_parts = shlex.split(prog_raw)
    except Exception:
        prog_parts = [prog_raw]
    if len(prog_parts) == 0:
        prog_parts = ["jx", "garfield"]
    return shlex.join([str(x) for x in (prog_parts + tokens)])


def _emit_garfield_command_to_log(logger: logging.Logger) -> None:
    lines = ["", "[ Command ]", f"  {_garfield_invocation_command()}"]
    for message in lines:
        record = logger.makeRecord(
            logger.name,
            logging.INFO,
            __file__,
            0,
            message,
            args=(),
            exc_info=None,
        )
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.handle(record)


def _emit_garfield_file_only_line(logger: logging.Logger, message: str) -> None:
    _emit_garfield_file_only_record(logger, logging.INFO, message)


def _emit_garfield_file_only_warning(logger: logging.Logger, message: str) -> None:
    _emit_garfield_file_only_record(logger, logging.WARNING, message)


def _emit_garfield_file_only_record(
    logger: logging.Logger, level: int, message: str
) -> None:
    record = logger.makeRecord(
        logger.name,
        int(level),
        __file__,
        0,
        str(message),
        args=(),
        exc_info=None,
    )
    handled = False
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.handle(record)
            handled = True
    if (not handled) and logger.propagate:
        parent = logger.parent
        while parent is not None:
            for handler in parent.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.handle(record)
                    handled = True
            if handled:
                break
            parent = parent.parent


def _run_garfield_pseudo_fvlmm(
    *,
    pseudo_prefix: str,
    trait_label: str,
    pheno_values: np.ndarray,
    sample_ids: list[str],
    trait_grm: np.ndarray,
    trait_cov: Optional[np.ndarray],
    logic_maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    threads: int,
    logger,
    use_spinner: bool,
    bonferroni_n_tests: int | None = None,
    batch_size: int = 4096,
) -> dict[str, object]:
    pseudo_ids, pseudo_ctx = prepare_packed_ctx_from_plink(
        str(pseudo_prefix),
        maf=float(logic_maf_threshold),
        missing_rate=float(max_missing_rate),
        het_threshold=float(het_threshold),
        snps_only=False,
        filter_mode="compact",
        use_cache=False,
    )
    pseudo_ids_arr = np.asarray(pseudo_ids, dtype=str)
    expected_ids_arr = np.asarray(sample_ids, dtype=str)
    if (
        int(pseudo_ids_arr.shape[0]) != int(expected_ids_arr.shape[0])
        or not np.array_equal(pseudo_ids_arr, expected_ids_arr)
    ):
        raise ValueError(
            "GARFIELD pseudo follow-up sample mismatch between pseudo BED and aligned trait samples."
        )

    keep = np.isfinite(np.asarray(pheno_values, dtype=np.float64).reshape(-1))
    if trait_cov is not None:
        keep &= np.all(np.isfinite(np.asarray(trait_cov, dtype=np.float64)), axis=1)
    keep_idx = np.flatnonzero(keep).astype(np.int64, copy=False)
    n_keep = int(np.count_nonzero(keep))
    cov_dim = 0 if trait_cov is None else int(np.asarray(trait_cov).shape[1])
    if n_keep <= cov_dim + 4:
        raise ValueError(
            f"GARFIELD pseudo FvLMM has too few usable samples after alignment/filtering: n={n_keep}, cov={cov_dim}."
        )

    y_trait = np.ascontiguousarray(
        np.asarray(pheno_values, dtype=np.float64).reshape(-1)[keep],
        dtype=np.float64,
    )
    cov_trait = (
        None
        if trait_cov is None
        else np.ascontiguousarray(np.asarray(trait_cov, dtype=np.float64)[keep, :], dtype=np.float64)
    )
    grm_trait = np.asarray(
        np.asarray(trait_grm, dtype=np.float64)[np.ix_(keep_idx, keep_idx)],
        dtype=np.float64,
        order="C",
    )
    sample_indices_full = np.ascontiguousarray(keep_idx, dtype=np.int64)

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
            "GARFIELD pseudo FvLMM metadata length mismatch: "
            f"sites={len(pseudo_sites)}, af={int(af.shape[0])}, miss={int(miss.shape[0])}."
        )

    model = FvLMM(y_trait, cov_trait, grm_trait)
    rows: list[dict[str, object]] = []
    step = max(1, int(batch_size))
    n_sites = len(pseudo_sites)
    for start in range(0, n_sites, step):
        end = min(start + step, n_sites)
        local_rows = np.arange(start, end, dtype=np.int64)
        decoded = _fvlmm2_decode_rows(
            pseudo_ctx,
            row_indices_local=local_rows,
            sample_indices_full=sample_indices_full,
        )
        stats = np.asarray(model.gwas(decoded, threads=int(threads)), dtype=np.float64)
        for idx_local, stat in enumerate(stats, start=start):
            site = pseudo_sites[idx_local]
            snp = str(site.snp)
            rows.append(
                {
                    "chrom": str(site.chrom),
                    "pos": int(site.pos),
                    "snp": snp,
                    "allele0": str(site.allele0),
                    "allele1": str(site.allele1),
                    "af": float(af[idx_local]),
                    "miss": float(miss[idx_local]),
                    "beta": float(stat[0]),
                    "se": float(stat[1]),
                    "chisq": _garfield_followup_chisq(stat[0], stat[1]),
                    "pwald": float(stat[2]),
                    "row_role": _garfield_followup_row_role(snp),
                }
            )

    full_df = pd.DataFrame(rows)
    tsv_df = _format_garfield_fvlmm_output_df_for_tsv(
        _sort_garfield_fvlmm_rows_for_tsv(full_df)
    )

    out_base = f"{pseudo_prefix}.fvlmm"
    tsv_path = f"{out_base}.tsv"
    figure_path = f"{out_base}.svg"
    tsv_df.to_csv(tsv_path, sep="\t", index=False)
    saved_paths = [tsv_path]

    # GARFIELD's ``-m/--meff`` is the effective number of independent SNP
    # tests, not the number of expanded pseudo-rule rows.  Keep the fallback
    # local to this follow-up so old callers which do not pass ``meff`` still
    # get a valid Bonferroni threshold.
    bonferroni_tests = int(bonferroni_n_tests or 0)
    if bonferroni_tests <= 0:
        bonferroni_tests = max(1, int(full_df.shape[0]))

    if tsv_df.shape[0] > 0:
        _run_fastplot_from_tsv_with_status(
            tsv_path,
            y_trait,
            xlabel=str(trait_label),
            outpdf=figure_path,
            threshold_n_tests=bonferroni_tests,
            plot_style="garfield",
            use_spinner=bool(use_spinner),
            emit_done_line=False,
        )
        saved_paths.append(figure_path)

    combo_unique = pd.DataFrame(columns=["snp", "pwald"])
    if full_df.shape[0] > 0:
        combo_unique = (
            full_df.loc[full_df["row_role"].astype(str).str.lower().eq("combo"), ["snp", "pwald"]]
            .drop_duplicates(subset=["snp"], keep="first")
            .copy()
        )
        combo_unique["pwald"] = pd.to_numeric(combo_unique["pwald"], errors="coerce")
        combo_unique = combo_unique.sort_values(by=["pwald", "snp"], kind="mergesort").reset_index(drop=True)

    best_combo = None
    best_combo_p = float("nan")
    if combo_unique.shape[0] > 0:
        best_row = combo_unique.iloc[0]
        best_combo = str(best_row.get("snp", ""))
        best_combo_p = float(best_row.get("pwald", float("nan")))

    summary_rows = [
        {
            "trait": str(trait_label),
            "n_rows_tested": int(full_df.shape[0]),
            "n_combo_tested": int(combo_unique.shape[0]),
            "bonferroni_n_tests": bonferroni_tests,
            "best_combo": best_combo,
            "best_combo_p": best_combo_p,
            "route": "fvlmm",
        }
    ]
    return {
        "saved_paths": saved_paths,
        "tsv_paths": [tsv_path],
        "figure_paths": [figure_path] if tsv_df.shape[0] > 0 else [],
        "summary_rows": summary_rows,
        "tsv_path": tsv_path,
        "raw_tsv_path": None,
        "expanded_tsv_path": None,
        "skipped_tsv_path": None,
        "figure_path": figure_path if tsv_df.shape[0] > 0 else None,
        "n_rules_supported": int(combo_unique.shape[0]),
        "n_rules_skipped": 0,
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


def _coerce_genefile_paths(paths: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(paths, str):
        out = [paths]
    else:
        out = [str(x) for x in paths]
    return [str(x).strip() for x in out if str(x).strip() != ""]


def _infer_scan_mode_from_genefile(paths: str | list[str] | tuple[str, ...]) -> str:
    scan_mode = "gene"
    path_list = _coerce_genefile_paths(paths)
    if len(path_list) == 0:
        raise ValueError("Gene file list is empty.")
    for path in path_list:
        genesets = _read_geneset_lines(path)
        if len(genesets) == 0:
            raise ValueError(f"Gene file is empty: {path}")
        if any(len(genes) > 1 for genes in genesets):
            scan_mode = "geneset"
    return scan_mode


def _resolve_scan_cli_positive_int(parser: argparse.ArgumentParser, raw: str, flag_desc: str) -> int:
    try:
        value = int(str(raw).strip())
    except Exception:
        parser.error(f"{flag_desc} must be an integer, got: {raw}")
    if value <= 0:
        parser.error(f"{flag_desc} must be > 0")
    return value


def _resolve_window_scan_args(
    parser: argparse.ArgumentParser,
    raw_args: list[str] | None,
    *,
    default_extension: int,
) -> tuple[int, int]:
    values = [] if raw_args is None else [str(x).strip() for x in raw_args if str(x).strip() != ""]
    if len(values) > 2:
        parser.error("-w/--window accepts at most [ext] [step]")
    extension = (
        int(default_extension)
        if len(values) <= 0
        else _resolve_scan_cli_positive_int(parser, values[0], "-w/--window ext")
    )
    step = (
        max(1, int(extension) // 2)
        if len(values) <= 1
        else _resolve_scan_cli_positive_int(parser, values[1], "-w/--window step")
    )
    return int(extension), int(step)


def _resolve_genefile_scan_args(
    parser: argparse.ArgumentParser,
    raw_specs: list[list[str]] | None,
    *,
    default_extension: int,
) -> tuple[list[str], int, int]:
    specs = [] if raw_specs is None else raw_specs
    if len(specs) == 0:
        parser.error("-g/--genefile requires at least one file")
    files: list[str] = []
    explicit_extensions: list[int] = []
    explicit_steps: list[int] = []
    for spec in specs:
        values = [str(x).strip() for x in spec if str(x).strip() != ""]
        if len(values) <= 0 or len(values) > 3:
            parser.error("-g/--genefile accepts FILE [ext] [step]")
        files.append(values[0])
        if len(values) >= 2:
            explicit_extensions.append(
                _resolve_scan_cli_positive_int(parser, values[1], "-g/--genefile ext")
            )
        if len(values) >= 3:
            explicit_steps.append(
                _resolve_scan_cli_positive_int(parser, values[2], "-g/--genefile step")
            )
    if len(set(explicit_extensions)) > 1:
        parser.error("All -g/--genefile occurrences must use the same ext when specified.")
    if len(set(explicit_steps)) > 1:
        parser.error("All -g/--genefile occurrences must use the same step when specified.")
    extension = int(explicit_extensions[0]) if len(explicit_extensions) > 0 else int(default_extension)
    step = int(explicit_steps[0]) if len(explicit_steps) > 0 else max(1, int(extension) // 2)
    return files, extension, step


def _format_genefile_display(paths: list[str]) -> Optional[str]:
    if len(paths) <= 0:
        return None
    return "; ".join(str(x) for x in paths)


def _build_interval_groups(
    genefile: str | list[str] | tuple[str, ...],
    gff3: str,
    extension: int,
    scan_mode: str,
) -> tuple[list[str], list[list[tuple[str, int, int]]]]:
    all_genesets: list[list[str]] = []
    for path in _coerce_genefile_paths(genefile):
        all_genesets.extend(_read_geneset_lines(path))
    genesets = all_genesets
    if len(genesets) == 0:
        raise ValueError("Gene file list is empty.")
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


def _build_all_gene_interval_groups(
    gff3: str,
    extension: int,
) -> list[list[tuple[str, int, int]]]:
    dfgff3 = readanno(gff3, "ID").iloc[:, :4].set_index(3)
    dfgff3 = dfgff3.loc[~dfgff3.index.duplicated()]
    groups: list[list[tuple[str, int, int]]] = []
    ext = max(1, int(extension))
    for gene_id, row in dfgff3.iterrows():
        chrom = str(row[0])
        gene_start = int(row[1])
        gene_end = int(row[2])
        center = (gene_start + gene_end) // 2
        start = center - ext
        end = center + ext
        if not gene_id:
            continue
        groups.append([(chrom, start, end)])
    return groups


def _normalize_scan_chrom(chrom: object) -> str:
    text = str(chrom).strip()
    if len(text) > 2 and (text.endswith("_1") or text.endswith("_2")):
        return text[:-2]
    if len(text) > 1 and (text.endswith("-") or text.endswith("+")):
        return text[:-1]
    return text


def _count_window_scan_units_from_bim(
    prefix: str,
    *,
    extension: int,
    step: int,
    site_keep: Optional[np.ndarray] = None,
) -> int:
    bim_path = f"{prefix}.bim"
    if not os.path.exists(bim_path):
        return 0

    groups: dict[str, list[int]] = {}
    chrom_order: list[str] = []
    keep_mask = None if site_keep is None else np.asarray(site_keep, dtype=np.bool_).reshape(-1)

    with open(bim_path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if keep_mask is not None:
                if row_idx >= int(keep_mask.shape[0]) or not bool(keep_mask[row_idx]):
                    continue
            tok = line.rstrip("\n").split()
            if len(tok) < 4:
                continue
            chrom = _normalize_scan_chrom(tok[0])
            try:
                pos = int(tok[3])
            except Exception:
                continue
            if chrom not in groups:
                groups[chrom] = []
                chrom_order.append(chrom)
            groups[chrom].append(pos)

    total_windows = 0
    ext = max(1, int(extension))
    step_bp = max(1, int(step))
    for chrom in chrom_order:
        positions = groups.get(chrom, [])
        if len(positions) == 0:
            continue
        positions.sort()
        min_bp = int(positions[0])
        max_bp = int(positions[-1])
        l = 0
        r = 0
        center = int(min_bp)
        prev_sig = None
        n = len(positions)

        while True:
            left_bp = max(center - ext, min_bp)
            right_bp = min(center + ext, max_bp)
            while l < n and int(positions[l]) < left_bp:
                l += 1
            if r < l:
                r = l
            while r < n and int(positions[r]) <= right_bp:
                r += 1
            if r > l:
                sig = (l, r - 1, r - l)
                if prev_sig != sig:
                    total_windows += 1
                    prev_sig = sig
            if center >= max_bp:
                break
            center += step_bp
            if center > np.iinfo(np.int32).max:
                break

        if prev_sig is None:
            total_windows += 1

    return int(total_windows)


def _scan_mode_to_logic_unit_kind(scan_mode: str) -> str:
    mode = str(scan_mode).strip().lower()
    if mode == "window":
        return "window"
    if mode == "wholegenome":
        return "wholegenome"
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
            return "gain from layer 1 (layer 1 gain = score; interaction gain thereafter)"
        return f"raw through layer {gain_start - 1}, gain from layer {gain_start}"
    return mode


def _dev_help_requested(argv: Optional[list[str]] = None) -> bool:
    tokens = list(sys.argv[1:] if argv is None else argv)
    return ("-dev" in tokens) or ("--dev" in tokens)


def _diagnostics_requested(argv: Optional[list[str]] = None) -> bool:
    """Return whether the opt-in GARFIELD P0 diagnostics flag is active.

    Diagnostics are deliberately development-only: keeping ``--diagnostics``
    behind ``-dev`` prevents a normal production scan from materializing the
    per-window candidate/beam trace.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    return "--diagnostics" in tokens and _dev_help_requested(tokens)


def _parse_rule_null_penalty_spec(
    spec: object,
) -> tuple[str, float, bool, Optional[str]]:
    if spec is None:
        return "gev", 0.99, False, None
    text = str(spec).strip().lower()
    if text == "":
        raise ValueError(
            "-pm/--permutation requires one of: gev, bayes, g99, g99.9, q99, q99.9, or a float in (0, 1)."
        )
    if text in {"gev", "gumbel", "auto"}:
        return "gev", 0.99, False, text
    if text in {"bayes", "bayesian", "hbayes", "hierarchical"}:
        return "bayes", 0.99, False, text
    if text.startswith("g"):
        digits = text[1:]
        try:
            pct = float(digits)
        except Exception as exc:
            raise ValueError(
                "-pm/--permutation GEV mode must look like g99 or g99.9."
            ) from exc
        quantile = pct / 100.0
        if not np.isfinite(quantile) or not (0.0 < float(quantile) < 1.0):
            raise ValueError("-pm/--permutation GEV quantile must be in (0, 1).")
        return "gev", float(quantile), False, text
    if text.startswith("q"):
        digits = text[1:]
        try:
            pct = float(digits)
        except Exception as exc:
            raise ValueError(
                "-pm/--permutation empirical-quantile mode must look like q99 or q99.9."
            ) from exc
        quantile = pct / 100.0
    else:
        try:
            quantile = float(text)
        except Exception as exc:
            raise ValueError(
                "-pm/--permutation must look like gev, bayes, g99, g99.9, q99, q99.9, or a float in (0, 1)."
            ) from exc
    if not np.isfinite(quantile) or not (0.0 < float(quantile) < 1.0):
        raise ValueError("-pm/--permutation quantile must be in (0, 1).")
    return "quantile", float(quantile), False, text


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


def _emit_garfield_bg_noise_summary_to_log(
    logger,
    *,
    trait_name: str,
    bg_noise_summary: object,
    use_spinner: bool,
) -> None:
    if not isinstance(bg_noise_summary, dict):
        return
    buckets = bg_noise_summary.get("buckets")
    if not isinstance(buckets, list) or len(buckets) == 0:
        return
    dataset = str(bg_noise_summary.get("dataset", "")).strip() or "full"
    tsv_rows: list[str] = [
        "\t".join(
            [
                "trait",
                "dataset",
                "bucket",
                "kind",
                "method",
                "quantile",
                "penalty",
                "mean",
                "variance",
                "n",
                "min",
                "q25",
                "median",
                "q75",
                "max",
            ]
        )
    ]
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        label = str(bucket.get("label", "")).strip() or "layer?"
        search = bucket.get("search") if isinstance(bucket.get("search"), dict) else {}
        output = bucket.get("output") if isinstance(bucket.get("output"), dict) else {}
        if search:
            tsv_rows.append(
                "\t".join(
                    [
                        str(trait_name),
                        dataset,
                        label,
                        "search",
                        str(search.get("method", "")),
                        _garfield_format_metric4(search.get("quantile")),
                        _garfield_format_metric4(search.get("penalty")),
                        _garfield_format_metric4(search.get("mean")),
                        _garfield_format_metric4(search.get("variance")),
                        str(int(search.get("n", 0) or 0)),
                        _garfield_format_metric4(search.get("min")),
                        _garfield_format_metric4(search.get("q25")),
                        _garfield_format_metric4(search.get("median")),
                        _garfield_format_metric4(search.get("q75")),
                        _garfield_format_metric4(search.get("max")),
                    ]
                )
            )
        tsv_rows.append(
            "\t".join(
                [
                    str(trait_name),
                    dataset,
                    label,
                    "output",
                    str(output.get("method", "")),
                    _garfield_format_metric4(output.get("quantile")),
                    _garfield_format_metric4(output.get("penalty")),
                    _garfield_format_metric4(output.get("mean")),
                    _garfield_format_metric4(output.get("variance")),
                    str(int(output.get("n", 0) or 0)),
                    _garfield_format_metric4(output.get("min")),
                    _garfield_format_metric4(output.get("q25")),
                    _garfield_format_metric4(output.get("median")),
                    _garfield_format_metric4(output.get("q75")),
                    _garfield_format_metric4(output.get("max")),
                ]
            )
        )
    _emit_garfield_file_only_line(
        logger,
        "GARFIELD bg-noise table:",
    )
    for row in tsv_rows:
        _emit_garfield_file_only_line(logger, row)


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


def _split_structure_prior_payload(payload):
    if not isinstance(payload, dict):
        return (None, payload)
    prior_payload = payload.get("prior")
    posterior_payload = dict(payload)
    posterior_payload.pop("prior", None)
    return (prior_payload, posterior_payload)


class _GarfieldStageProgress:
    _HIDDEN_STAGES = {"null_prep", "structure_prep", "garfield_null_summary"}

    @classmethod
    def is_hidden_stage(cls, stage: object) -> bool:
        return str(stage).strip().lower() in cls._HIDDEN_STAGES

    def __init__(self, *, scan_desc: str, enabled: bool) -> None:
        self.scan_desc = str(scan_desc)
        self.enabled = bool(enabled)
        self._start_ts = time.monotonic()
        self._closed = False
        self._rich = None
        self._task_id: int | None = None
        self._task_total = 1
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
                field_templates=[],
                bar_width=30,
                finished_text=" ",
                transient=True,
            )

    def _label(self, stage: str) -> str:
        if stage == "garfield_marker_qc":
            return "Reading genotype metadata / applying QC"
        if stage == "garfield_auto_grm":
            return "Building additive GRM"
        if stage == "garfield_residual_fit":
            return "Fitting observed-y LMM null"
        if stage == "garfield_projection":
            return "Building exact GRM projection"
        if stage == "garfield_logic_bits":
            return "Packing genotype bitsets"
        if stage == "max6_setup_margins":
            return "Preparing Max6 marker margins"
        if stage == "max6_setup_ld_tags":
            return "Preparing Max6 LD-tag universe"
        if stage == "max6_pair_count":
            return "Counting Bonferroni pair universe"
        if stage == "max6_grmmax6_scan_pairs":
            return "Screening Max6 pair chunks"
        if stage == "max6_screen_calibration":
            return "Max6 screen null calibration"
        if stage == "max6_screen_validation":
            return "Independent Max6 screen validation"
        if stage == "max6_screen_sample_selection":
            return "Selecting sampled Max6 null universe"
        if stage == "max6_screen_sample_scoring":
            return "Scoring sampled Max6 null maxima"
        if stage == "max6_exceedance_pair_census":
            return "Census unique pairs by support stratum"
        if stage == "max6_exceedance_pair_preflight":
            return "Preflighting exceedance sample memory"
        if stage == "max6_exceedance_pair_sample_selection":
            return "Selecting deterministic nested pair samples"
        if stage == "max6_screen_exceedance_full_scan":
            return "Exact full-universe exceedance audit"
        if stage == "max6_grmmax6_scan":
            return self.scan_desc
        if stage == "null_prep":
            return "Preparing Null Penalty"
        if stage == "null_penalty":
            return "Estimating Null Penalty"
        if stage == "structure_prep":
            return self.scan_desc
        if stage == "structure_prior":
            return self.scan_desc
        if stage == "max6_grmmax6_permutation":
            return "Calibrating Max6 Null Threshold"
        return self.scan_desc

    def update(self, stage: str, done: int, total: int, meta: object | None = None) -> None:
        stage_key = str(stage).strip().lower()
        if stage_key in self._HIDDEN_STAGES:
            return
        done_i = max(0, int(done))
        total_i = max(done_i, int(total))
        self._task_total = max(1, total_i)
        label = self._label(stage_key)

        if self._rich is not None:
            if self._task_id is None:
                self._rich.start()
                self._task_id = self._rich.add_task(
                    label,
                    total=max(1, total_i),
                    completed=min(done_i, max(1, total_i)),
                )
            else:
                self._rich.update(
                    self._task_id,
                    description=label,
                    total=max(1, total_i),
                    completed=min(done_i, max(1, total_i)),
                )
            self._current_stage = stage_key
            return

        if not self.enabled:
            return
        if self._current_stage != stage_key or self._adapter is None:
            if self._adapter is not None:
                self._adapter.close(show_done=False)
            self._adapter = ProgressAdapter(
                total=max(1, total_i),
                desc=label,
                show_spinner=True,
                show_postfix=False,
                show_remaining=True,
                keep_display=False,
                emit_done=False,
                bar_width=30,
                force_animate=True,
            )
            self._current_stage = stage_key
            self._adapter_done = 0
        else:
            self._adapter.set_total(max(1, total_i))

        delta = max(0, done_i - self._adapter_done)
        if delta > 0:
            self._adapter.update(delta)
        self._adapter_done = done_i

    def close(self, *, success: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        if self._adapter is not None:
            if success:
                self._adapter.finish()
            self._adapter.close(show_done=False)
            self._adapter = None
        if self._rich is not None:
            if success and self._task_id is not None:
                self._rich.update(self._task_id, completed=self._task_total)
            if self._task_id is not None:
                self._rich.stop()
            self._rich = None
        if success and self.enabled:
            elapsed = format_elapsed(time.monotonic() - self._start_ts)
            print_success(f"{self.scan_desc} ...Finished [{elapsed}]", force_color=True)

    @property
    def elapsed_text(self) -> str:
        return format_elapsed(time.monotonic() - self._start_ts)


def _run_scan_with_progress(
    desc: str,
    *,
    use_spinner: bool,
    invoke,
):
    if not bool(use_spinner):
        with CliStatus(f"{desc}...", enabled=False, timeout=0.1):
            return invoke(None)

    stage_progress = _GarfieldStageProgress(scan_desc=desc, enabled=bool(use_spinner))

    def _progress_cb(*event) -> None:
        if len(event) == 4:
            stage, done, total, meta = event
            stage_name = str(stage)
            if _GarfieldStageProgress.is_hidden_stage(stage_name):
                return
            stage_progress.update(stage_name, int(done), int(total), meta)
            return
        if len(event) == 2:
            done, total = event
            stage_progress.update("scan", int(done), int(total), None)
            return
        raise ValueError(f"unexpected GARFIELD progress event: {event!r}")

    try:
        out = invoke(_progress_cb)
    except BaseException:
        stage_progress.close(success=False)
        print_failure(f"{desc} ...Failed [{stage_progress.elapsed_text}]", force_color=True)
        raise

    stage_progress.close()
    return out


def main() -> None:
    _require_rust_backend()

    t_start = time.time()
    use_spinner = stdout_is_tty()
    show_dev_help = _dev_help_requested()

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
    add_common_pheno_arg(required_group, required=False, help_text="Phenotype file for the scan.")
    scan_mode_group = required_group.add_mutually_exclusive_group(required=True)
    scan_mode_group.add_argument(
        "-w",
        "--window",
        dest="window_args",
        nargs="*",
        default=None,
        metavar=("EXT", "STEP"),
        help="Window scan mode. Use `-w`, optionally followed by EXT and STEP.",
    )
    scan_mode_group.add_argument(
        "-g",
        "--genefile",
        dest="genefile_args",
        nargs="+",
        action="append",
        default=None,
        metavar="FILE",
        help=(
            "Gene or gene-set scan file. Use `-g FILE`, optionally followed by EXT and STEP; "
            "repeat `-g` for multiple files. "
            "Requires -gff/--gff3."
        ),
    )
    scan_mode_group.add_argument(
        "-wg",
        "--whole-genome",
        dest="whole_genome",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument("-gff", "--gff3", type=str, default=None, help="GFF3 annotation.")
    optional_group.add_argument(
        "--scan-mode",
        type=str,
        choices=["window", "gene", "genepair", "geneset", "wholegenome"],
        default=None,
        help=argparse.SUPPRESS,
    )
    add_common_trait_selector_args(optional_group, dest="ncol")
    add_common_variant_filter_args(
        optional_group,
        help_profile="pureline",
        include_maf=True,
        include_geno=True,
        include_het=True,
        maf_default=0.02,
        geno_default=0.05,
        het_default=1.0,
    )
    optional_group.add_argument(
        "-lmaf",
        "--lmaf",
        type=float,
        default=None,
        help=(
            "MAF threshold for logic/pseudo SNPs generated by GARFIELD "
            "(default: auto = 30 / trait valid sample size). "
            "Input genotype filtering still uses -maf/--maf."
        ) if show_dev_help else argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-mem",
        "--memory",
        dest="logic_memory_mb",
        type=float,
        default=0.0,
        help=(
            "Decoded GARFIELD logic-bit working budget in MiB. "
            "0 keeps the resident path; a positive value enables the mmap-backed path "
            "when the decoded bitset exceeds this budget."
        ),
    )
    optional_group.add_argument(
        "-dev",
        "--dev",
        action="store_true",
        help=argparse.SUPPRESS,
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
        "--residual-method",
        dest="residual_method",
        choices=["exact", "rrblup_pcg"],
        default="exact",
        help=(
            "Residualization method: exact GRM-eigh/REML (default) or "
            "rrBLUP-PCG additive projection."
        ),
    )
    optional_group.add_argument(
        "--pcg-max-iter",
        dest="pcg_max_iter",
        type=int,
        default=100,
        help="Maximum marker-space PCG iterations for --residual-method rrblup_pcg.",
    )
    optional_group.add_argument(
        "--pcg-tol",
        dest="pcg_tol",
        type=float,
        default=1e-4,
        help="Relative residual tolerance for rrBLUP-PCG (default: 1e-4).",
    )
    optional_group.add_argument(
        "--pcg-small-n-max",
        dest="pcg_small_n_max",
        type=int,
        default=15000,
        help=(
            "Use GRM-eigh/Brent-REML for PCG lambda selection when n is at most this "
            "value; larger n uses HE (default: 15000)."
        ),
    )
    optional_group.add_argument(
        "-engine",
        "--engine",
        type=str.upper,
        choices=["NONE", "CORR", "RF", "GBDT"],
        default="NONE",
        help=(
            "Legacy ML engine for candidate search. The default max6-grmmax6 backend uses "
            "LD-tag exhaustive Max6 screening followed by GRM-INT Top-K scoring and does "
            "not call ML/CORR."
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
        "-pm",
        "--permutation",
        dest="rule_permutation_quantile",
        type=str,
        default=None,
        help=(
            "Set the GARFIELD bucket null-penalty method for legacy backends. "
            "The analytical `max6-grmmax6` backend does not accept this option. "
            "Legacy backends accept `gev`, `gumbel`, `bayes`, `g99`, `g99.9`, "
            "or an empirical quantile such as `q99`, `q99.9`, or `0.99`. "
            "This option only controls the null-penalty threshold; it does not append "
            "permutation p-value or FDR columns."
        ),
    )
    optional_group.add_argument(
        "-no-clean",
        "--no-clean",
        action="store_true",
        dest="no_clean",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--raw-design",
        action="store_true",
        dest="raw_design",
        default=False,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-nf-xor",
        "--nf-xor",
        action="store_true",
        dest="disable_xor_substate_filter",
        default=False,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--xor-search",
        action="store_true",
        dest="xor_search",
        default=False,
        help=(
            "Enable XOR logic-gate expansion during GARFIELD search (legacy Beam, "
            "or the order-2 AllPair stage of pair-triple; default: off)."
        ),
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
    optional_group.add_argument("-layer", "--layer", type=int, default=None, help="Maximum beam-search rule depth (default: 2).")
    dev_group = parser.add_argument_group("Development Arguments (show with -dev)")
    dev_group.add_argument(
        "--diagnostics",
        action="store_true",
        default=False,
        help=(
            "Write per-window stage1/selected/beam recall diagnostics TSV. "
            "Requires -dev."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--sampled-screen-null-diagnostic",
        action="store_true",
        default=False,
        help=(
            "Run diagnostic-only exact Max6 null maxima over nested uniform hash-priority samples "
            "of 100k/1m/10m/50m unique canonical pairs. Requires max6-grmmax6, -dev, and exactly "
            "100 existing null draws; never changes the production screen threshold."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--sampled-screen-pair-seed",
        type=int,
        default=0x4A5853414D504C45,
        help=(
            "Independent deterministic pair-priority seed for the sampled-screen diagnostic."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--sampled-screen-pair-seeds",
        default=None,
        help=(
            "Optional comma-separated list of 3–5 independent deterministic pair-priority seeds; "
            "sampled runs are sequential and share the same full-null maxima. If omitted, use the "
            "single --sampled-screen-pair-seed value."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--sampled-screen-maf-cutpoints",
        default="0,0.05,0.10,0.20,0.30,0.50",
        help=(
            "Configurable diagnostic min-binary-MAF bin edges (comma-separated; provisional defaults, "
            "not yet canonical)."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--sampled-screen-r2-cutpoints",
        default="0,0.2,0.4,0.6,0.8,1.0",
        help=(
            "Configurable diagnostic binary pairwise-r2 bin edges (comma-separated; provisional defaults, "
            "not yet canonical)."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--max6-screen-tail-audit",
        action="store_true",
        default=False,
        help=(
            "Run an opt-in exact full-MaxT tail audit, retaining bounded Top-4096 pairs per null. "
            "Diagnostic only; does not change the production threshold. Requires -dev and "
            "max6-grmmax6."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--max6-screen-exceedance-audit",
        action="store_true",
        default=False,
        help=(
            "Run the opt-in weighted Max6 exceedance-count audit on the exact full-MaxT calibration nulls; "
            "diagnostic only, with no threshold or FWER-claim changes. Requires -dev, "
            "--search-backend max6-grmmax6, exact residualization, and 100 or 500 null repeats."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--max6-screen-pot-audit",
        action="store_true",
        default=False,
        help=(
            "Run the opt-in exact sampled-pair Max6 marginal POT/GPD diagnostic using 5 nested uniform samples; "
            "diagnostic only, never changes production thresholds. Requires -dev, max6-grmmax6, exact residualization, "
            "and 100 or 500 null repeats."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--max6-screen-tail-audit-repeats",
        type=int,
        choices=(100, 250, 500),
        default=500,
        help=(
            "Independent audit null draw count (use 500 for the main region and 100 for geometry replications)."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--max6-screen-tail-pair-seed",
        type=int,
        default=0x4A5853414D504C45,
        help=(
            "Deterministic hash-priority pair sample seed, independent of the null stream."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--max6-screen-tail-support-aware",
        action="store_true",
        default=False,
        help=(
            "Compare uniform sampling with diagnostic 40/50/60% allocations to min_cell 5–19; "
            "does not alter production thresholds or FWER claims."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--max6-screen-tail-support-shares",
        default="40,50,60",
        help=(
            "Comma-separated support-aware min_cell 5–19 shares (allowed: 40,50,60); "
            "use one selected share for replication regions."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--search-backend",
        dest="search_backend",
        choices=["legacy", "pair-triple", "max6-grmmax6", "max6-grmmax6-legacy"],
        default="max6-grmmax6",
        help=(
            "GARFIELD search backend. `max6-grmmax6` runs deterministic LD-tag "
            "exhaustive packed Max6 screening followed by one-df GRM-INT Top-K ranking "
            "by default (`--no-grm-int` selects exact GRM-Max6 reranking); "
            "`max6-grmmax6-legacy` retains the former ML/CORR-prefiltered path; "
            "`pair-triple` remains development-only."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "-gain",
        "--gain-layer",
        dest="gain_layer",
        type=int,
        default=1,
        help=(
            "Start ranking beam candidates by interaction gain from this layer onward "
            "(default: 1; layer 1 gain is its own score, later layers use interaction "
            "gain, and the bucket null penalty remains active)."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    dev_group.add_argument(
        "--null-repeats",
        dest="null_repeats",
        type=int,
        default=300,
        help=(
            "Number of full-scan null replicates used for rule-tail calibration "
            "(default: 300; minimum: 50; 100 or 300 allowed outside -dev; "
            "300-500 recommended for stable 4/5-way tails)."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    optional_group.add_argument(
        "--grm-int",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use one-df GRM interaction-only FWL scoring for the Max6 Top-K pairs "
            "(g1×g2 adjusted for g1, g2, covariates, and GRM), write chi-square/pwald, "
            "and rank within Top-K by interaction chi-square. Defaults on for "
            "--search-backend max6-grmmax6 and off for other routes; --no-grm-int "
            "selects exact GRM-Max6 scoring."
        ),
    )
    optional_group.add_argument(
        "--min-cell",
        type=int,
        default=None,
        help=(
            "Before Max6 scoring, require at least this many samples in each of the four "
            "binary genotype cells for a marker pair (default: 10 for Max6; 0 disables "
            "the count floor). Combined with --min-cell-frac by taking the maximum."
        ),
    )
    optional_group.add_argument(
        "--ld-tag-r2",
        type=float,
        default=1.0,
        help=(
            "Max6 pre-screen genotype compression (default: 1, identical encoded "
            "genotypes only). Values in (0, 1) enable approximate LD compression; "
            "0.95 reproduces the previous faster, lossy search."
        ),
    )
    optional_group.add_argument(
        "--min-cell-frac",
        "--min-cell-fraction",
        type=float,
        default=None,
        help=(
            "Minimum fraction of effective samples in each of the four genotype cells "
            "(default: 0.01 for Max6; range: 0 to 0.25). Set 0 for the count floor only. "
            "The effective threshold "
            "is max(min-cell, ceil(min-cell-frac * Neff)), using the aligned samples "
            "actually scored for each phenotype. Max6 backends only."
        ),
    )
    optional_group.add_argument(
        "-topk",
        "--topk",
        dest="rule_topk",
        type=int,
        default=1,
        help=(
            "Keep up to top-k de-duplicated Max6 pairs per scan unit for GRM-INT scoring "
            "(default: 1)."
        ),
    )
    optional_group.add_argument(
        "-bimrange",
        "--bimrange",
        type=str,
        action="append",
        default=None,
        help=(
            "Restrict scan units to one or more genomic bp intervals. "
            "Repeat the flag or use comma-separated items, e.g. "
            "--bimrange 10:110800000-111200000,10:112000000-112200000. "
            "The analytical Max6 Bonferroni pair count uses this selected unique-pair "
            "universe; other backend calibration scopes are backend-specific."
        ),
    )
    optional_group.add_argument(
        "-m",
        "--meff",
        type=int,
        default=None,
        help=(
            "Effective SNP count used for GARFIELD Manhattan Bonferroni thresholding. "
            "Default uses the input genotype SNP count."
        ),
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
        help=argparse.SUPPRESS,
    )

    args, extras = parser.parse_known_args()
    args.grm_int, args.grm_int_mode_source = _resolve_garfield_grm_int_mode(
        args.grm_int,
        args.search_backend,
    )

    has_genotype = bool(args.bfile)
    has_pheno = bool(args.pheno)
    if bool(args.grm_int) and str(args.search_backend).strip().lower() != "max6-grmmax6":
        parser.error("--grm-int requires --search-backend max6-grmmax6")
    if args.min_cell is not None:
        if int(args.min_cell) < 0:
            parser.error("--min-cell must be >= 0")
        if not _is_max6_grm_backend(args.search_backend):
            parser.error("--min-cell requires a Max6 GARFIELD search backend")
    if args.min_cell_frac is None:
        args.min_cell_frac = 0.01 if _is_max6_grm_backend(args.search_backend) else 0.0
    if not np.isfinite(args.min_cell_frac) or not 0.0 <= args.min_cell_frac <= 0.25:
        parser.error("--min-cell-frac must be finite and between 0 and 0.25")
    if args.min_cell_frac > 0.0:
        if not _is_max6_grm_backend(args.search_backend):
            parser.error("--min-cell-frac requires a Max6 GARFIELD search backend")
    if not np.isfinite(args.ld_tag_r2) or not 0.0 < args.ld_tag_r2 <= 1.0:
        parser.error("--ld-tag-r2 must be finite and in (0, 1]")
    if args.ld_tag_r2 != 1.0 and not _is_ld_tag_max6_backend(args.search_backend):
        parser.error("--ld-tag-r2 requires the max6-grmmax6 search backend")
    args.min_cell_runtime = (
        int(args.min_cell)
        if args.min_cell is not None
        else (10 if _is_max6_grm_backend(args.search_backend) else 0)
    )
    if not has_genotype and not has_pheno:
        parser.error("the following arguments are required: -p/--pheno and -bfile/--bfile")
    if not has_genotype:
        parser.error("the following arguments are required: -bfile/--bfile")
    if not has_pheno:
        parser.error("the following arguments are required: -p/--pheno")
    if len(extras) > 0:
        parser.error("unrecognized arguments: " + " ".join(extras))
    if bool(args.diagnostics) and not bool(args.dev):
        parser.error("--diagnostics requires -dev")
    if bool(args.sampled_screen_null_diagnostic):
        if not bool(args.dev):
            parser.error("--sampled-screen-null-diagnostic requires -dev")
        if str(args.search_backend).lower() != "max6-grmmax6":
            parser.error("--sampled-screen-null-diagnostic requires --search-backend max6-grmmax6")
        if int(args.null_repeats) != 100:
            parser.error("--sampled-screen-null-diagnostic requires --null-repeats 100")
        seed_values_raw = (
            [value.strip() for value in str(args.sampled_screen_pair_seeds).split(",")]
            if args.sampled_screen_pair_seeds is not None
            else [str(args.sampled_screen_pair_seed)]
        )
        try:
            pair_sample_seeds = [int(value) for value in seed_values_raw]
        except ValueError as exc:
            parser.error(f"invalid --sampled-screen-pair-seeds value: {exc}")
        if not pair_sample_seeds or (
            len(pair_sample_seeds) != 1 and not 3 <= len(pair_sample_seeds) <= 5
        ):
            parser.error("provide one sampled pair seed or 3–5 independent pair seeds")
        if len(set(pair_sample_seeds)) != len(pair_sample_seeds):
            parser.error("--sampled-screen-pair-seeds must not contain duplicates")
        if any(seed < 0 or seed > 0xFFFFFFFFFFFFFFFF for seed in pair_sample_seeds):
            parser.error("sampled pair seeds must be unsigned 64-bit integers")
        if int(args.seed) in pair_sample_seeds:
            parser.error("pair sample seeds must be separate from --seed/null stream")
        args.sampled_screen_pair_seeds_normalized = ",".join(
            str(seed) for seed in pair_sample_seeds
        )
    else:
        args.sampled_screen_pair_seeds_normalized = ""
    if bool(args.max6_screen_tail_audit):
        if not bool(args.dev):
            parser.error("--max6-screen-tail-audit requires -dev")
        if str(args.search_backend).lower() != "max6-grmmax6":
            parser.error("--max6-screen-tail-audit requires --search-backend max6-grmmax6")
        if str(args.residual_method).lower() != "exact":
            parser.error("--max6-screen-tail-audit requires exact LMM residualization")
        if int(args.max6_screen_tail_pair_seed) == int(args.seed):
            parser.error("tail pair-sample seed must be separate from --seed")
    if bool(args.max6_screen_exceedance_audit):
        if not bool(args.dev):
            parser.error("--max6-screen-exceedance-audit requires -dev")
        if str(args.search_backend).lower() != "max6-grmmax6":
            parser.error("--max6-screen-exceedance-audit requires --search-backend max6-grmmax6")
        if str(args.residual_method).lower() != "exact":
            parser.error("--max6-screen-exceedance-audit requires exact LMM residualization")
        if int(args.null_repeats) not in (100, 500):
            parser.error("--max6-screen-exceedance-audit requires --null-repeats 100 or 500")
        if bool(args.sampled_screen_null_diagnostic) or bool(args.max6_screen_tail_audit):
            parser.error(
                "--max6-screen-exceedance-audit cannot be combined with sampled-max or Top-4096 tail-audit modes"
            )
    if bool(args.max6_screen_pot_audit):
        if not bool(args.dev):
            parser.error("--max6-screen-pot-audit requires -dev")
        if str(args.search_backend).lower() != "max6-grmmax6":
            parser.error("--max6-screen-pot-audit requires --search-backend max6-grmmax6")
        if str(args.residual_method).lower() != "exact":
            parser.error("--max6-screen-pot-audit requires exact LMM residualization")
        if int(args.null_repeats) not in (100, 500):
            parser.error("--max6-screen-pot-audit requires --null-repeats 100 or 500")
        pot_pair_sample_seeds = {
            0x2F6E2B1D6F6E0A11,
            0x79D23B474A5C19E3,
            0xA8C590D12E736B45,
            0xC4E18F2A0B6D9357,
            0xE75A13C98D42B60F,
        }
        if int(args.seed) in pot_pair_sample_seeds:
            parser.error("POT pair-sample seeds must be separate from --seed/null stream")
        if (
            bool(args.sampled_screen_null_diagnostic)
            or bool(args.max6_screen_tail_audit)
        ):
            parser.error(
                "--max6-screen-pot-audit cannot be combined with sampled-max or Top-4096 tail-audit modes"
            )
    if bool(args.max6_screen_tail_support_aware):
        if not bool(args.max6_screen_tail_audit):
            parser.error("--max6-screen-tail-support-aware requires --max6-screen-tail-audit")
        try:
            support_shares = [
                int(value.strip())
                for value in str(args.max6_screen_tail_support_shares).split(",")
                if value.strip()
            ]
        except ValueError as exc:
            parser.error(f"invalid --max6-screen-tail-support-shares: {exc}")
        if not support_shares or any(value not in (40, 50, 60) for value in support_shares):
            parser.error("support-aware shares must be a non-empty subset of 40,50,60")
        if len(set(support_shares)) != len(support_shares):
            parser.error("--max6-screen-tail-support-shares must not contain duplicates")
        args.max6_screen_tail_support_shares_normalized = support_shares
    else:
        args.max6_screen_tail_support_shares_normalized = []
    if str(args.search_backend).lower() == "pair-triple" and not bool(args.dev):
        parser.error("--search-backend pair-triple requires -dev")
    if int(args.null_repeats) < 1:
        parser.error("--null-repeats must be at least 1")
    if int(args.null_repeats) < 50 and not bool(args.dev):
        parser.error("--null-repeats below 50 is allowed only with -dev")
    if int(args.null_repeats) not in (100, 300) and not bool(args.dev):
        parser.error("--null-repeats must be 100 or 300 outside -dev")
    if int(args.pcg_max_iter) <= 0:
        parser.error("--pcg-max-iter must be > 0")
    if not np.isfinite(float(args.pcg_tol)) or float(args.pcg_tol) <= 0.0:
        parser.error("--pcg-tol must be finite and > 0")
    if int(args.pcg_small_n_max) <= 0:
        parser.error("--pcg-small-n-max must be > 0")
    args.xor_search_requested = bool(args.xor_search)
    args.xor_search = _resolve_garfield_xor_search(args.xor_search_requested)
    try:
        (
            args.rule_null_penalty_method_runtime,
            args.rule_null_quantile_runtime,
            args.rule_null_report_pvalue_runtime,
            args.rule_null_quantile_spec_runtime,
        ) = _parse_rule_null_penalty_spec(args.rule_permutation_quantile)
    except ValueError as e:
        parser.error(str(e))

    max6_audit_requested = any(
        (
            args.sampled_screen_null_diagnostic,
            args.max6_screen_exceedance_audit,
            args.max6_screen_pot_audit,
            args.max6_screen_tail_audit,
        )
    )
    try:
        args.rule_permutation_runtime = _resolve_garfield_null_resampling(
            args.search_backend,
            args.rule_permutation_quantile,
            max6_audit_requested,
        )
    except ValueError as exc:
        parser.error(str(exc))
    # Explicit development audits retain their historical null stream.
    if _is_ld_tag_max6_backend(args.search_backend) and max6_audit_requested:
        args.rule_null_penalty_method_runtime = "empirical"
        args.rule_null_quantile_runtime = 0.95
        args.rule_null_quantile_spec_runtime = "q95"

    default_extension = 50_000
    if bool(args.whole_genome):
        args.genefiles = []
        args.extension = int(default_extension)
        args.step = max(1, int(default_extension) // 2)
    elif args.window_args is not None:
        args.extension, args.step = _resolve_window_scan_args(
            parser,
            args.window_args,
            default_extension=default_extension,
        )
        args.genefiles = []
    else:
        args.genefiles, args.extension, args.step = _resolve_genefile_scan_args(
            parser,
            args.genefile_args,
            default_extension=default_extension,
        )
    if not (0.0 <= float(args.maf) <= 0.5):
        parser.error("-maf/--maf must be in [0, 0.5]")
    if args.lmaf is not None and not (0.0 <= float(args.lmaf) <= 0.5):
        parser.error("-lmaf/--lmaf must be in [0, 0.5]")
    if args.lmaf is not None and _is_max6_grm_backend(args.search_backend):
        parser.error("-lmaf/--lmaf applies only to legacy search; use --min-cell/--min-cell-frac for Max6")
    if not (0.0 <= float(args.geno) <= 1.0):
        parser.error("-geno/--geno must be in [0, 1]")
    if not (0.0 <= float(args.het) <= 1.0):
        parser.error("-het/--het must be in [0, 1]")
    if args.meff is not None:
        try:
            args.meff = int(args.meff)
        except Exception:
            parser.error("-m/--meff must be an integer")
        if int(args.meff) <= 0:
            parser.error("-m/--meff must be > 0")
    if (
        args.grm is not None
        and str(args.grm).strip().isdigit()
        and not os.path.exists(str(args.grm).strip())
    ):
        parser.error("-k/--grm now expects a GRM path.")

    if bool(args.whole_genome):
        args.scan_mode = "wholegenome"
    elif len(args.genefiles) > 0:
        if not args.gff3:
            parser.error("-g/--genefile requires -gff/--gff3.")
        try:
            args.scan_mode = _infer_scan_mode_from_genefile(args.genefiles)
        except ValueError as e:
            parser.error(str(e))
    else:
        args.scan_mode = "window"

    args.width = int(args.width) if args.width is not None else 100
    if int(args.width) <= 0:
        parser.error("-width/--width must be > 0")
    if not np.isfinite(float(args.logic_memory_mb)) or float(args.logic_memory_mb) < 0.0:
        parser.error("-mem/--memory must be >= 0")
    args.beam_width = int(args.width)

    args.layer = (
        int(args.layer) if args.layer is not None
        else 2
    )
    if int(args.layer) <= 0:
        parser.error("-layer must be > 0")
    if (
        bool(args.xor_search)
        and str(args.search_backend).lower() == "pair-triple"
        and int(args.layer) >= 3
    ):
        parser.error(
            "--xor-search with --search-backend pair-triple is currently supported "
            "only for order-2 AllPair; use the legacy backend for order-3 search"
        )
    if int(args.gain_layer) < 1:
        parser.error("-gain/--gain-layer must be >= 1")
    if int(args.rule_topk) <= 0:
        parser.error("-topk/--topk must be > 0")
    args.exhaustive_depth_runtime = (
        1
        if str(args.scan_mode).lower() == "wholegenome"
        else 2 if int(args.layer) >= 2 else 1
    )
    if args.engine is not None:
        args.engine = str(args.engine).upper()
    if str(args.scan_mode).lower() == "wholegenome":
        args.engine = "NONE"
    if _is_ld_tag_max6_backend(args.search_backend):
        # The production Max6 path is deliberately exhaustive after
        # genotype-only LD-tag compression. Do not let a stale --engine flag
        # claim ML prefiltering when Rust bypasses it by construction.
        args.engine = "NONE"

    ml_skip_tokens = {"NONE", "SKIP", "DIRECT"}
    args.ml_top_k_runtime = int(args.width) if args.engine not in ml_skip_tokens else 0
    args.top_rules_runtime = int(args.rule_topk)
    args.max_output_rules_runtime = 0
    args.max_output_ratio_runtime = 0.0

    args.rank_score = f"gain_from_layer:{int(args.gain_layer)}"
    args.rank_schedule_source = "cli-gain-layer"
    args.gain_start_layer_runtime = int(args.gain_layer)

    try:
        args.ncol = parse_trait_selector_specs(args.ncol, label="-n/--ncol")
    except ValueError as e:
        parser.error(str(e))

    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    if int(args.thread) <= 0:
        args.thread = int(detected_threads)
    if int(args.thread) > int(detected_threads):
        args.thread = int(detected_threads)

    gfile, prefix = determine_genotype_source(args)
    out_dir, outprefix, out_stem = apply_output_prefix_compat(args, prefix)
    if bool(args.max6_screen_exceedance_audit) or bool(args.max6_screen_pot_audit):
        project_root = Path(__file__).resolve().parents[3]
        resolved_output_prefix = Path(outprefix).expanduser().resolve()
        try:
            resolved_output_prefix.relative_to(project_root)
        except ValueError:
            pass
        else:
            raise SystemExit(
                "Max6 null-audit outputs must be outside the JanusX project; "
                "use /Volumes/data/... or /share/home/jxfu26/benchmark/..."
            )
    os.makedirs(out_dir, mode=0o755, exist_ok=True)
    configure_genotype_cache_from_out(out_dir)

    log_path = f"{outprefix}.garfield.log"
    logger = setup_logging(log_path)
    apply_outer_thread_cap(int(args.thread))
    ml_skipped = args.engine in ml_skip_tokens
    engine_runtime = "none" if ml_skipped else str(args.engine)
    rank_score_runtime = str(args.rank_score)
    rank_schedule_source = str(args.rank_schedule_source)
    gain_start_layer_runtime = (
        None if args.gain_start_layer_runtime is None else int(args.gain_start_layer_runtime)
    )
    rank_schedule_runtime = _describe_rank_schedule(rank_score_runtime)

    genefile_display = _format_genefile_display(args.genefiles)
    gff3_effective = args.gff3 if len(args.genefiles) > 0 else None

    checks: list[bool] = []
    checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    checks.append(ensure_file_exists(logger, args.pheno, "Phenotype file"))
    if args.grm:
        checks.append(ensure_file_exists(logger, args.grm, "GARFIELD GRM"))
    for genefile_path in args.genefiles:
        checks.append(ensure_file_exists(logger, genefile_path, "Gene file"))
    if gff3_effective:
        checks.append(ensure_file_exists(logger, gff3_effective, "GFF3 file"))
    if args.simbench:
        checks.append(ensure_file_exists(logger, args.simbench, "Simulation benchmark TSV"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    general_rows = [
        ("Genotype input", gfile),
        ("Residualization GRM", args.grm if args.grm else "auto from genotype"),
        ("Covariates", None if not args.cov_inputs else ",".join(str(x) for x in args.cov_inputs)),
        ("Phenotype", args.pheno),
        ("Scan mode", args.scan_mode),
        ("Gene files", genefile_display),
        ("GFF3", gff3_effective if gff3_effective else ("ignored" if args.gff3 else None)),
        ("Input MAF", float(args.maf)),
        ("Missing max (NA only)", float(args.geno)),
        ("Het max", float(args.het)),
        ("Extension", int(args.extension)),
        ("Step", int(args.step)),
        ("Bimrange", None if not args.bimrange else ",".join(str(x) for x in args.bimrange)),
        ("Engine", "none (skip ML)" if ml_skipped else args.engine),
        ("Structured pruning", not bool(args.no_clean)),
        ("Width", int(args.width)),
        (
            "Max6 minimum cell",
            int(args.min_cell_runtime) if _is_max6_grm_backend(args.search_backend) else None,
        ),
        (
            "Max6 minimum cell fraction",
            float(args.min_cell_frac) if _is_max6_grm_backend(args.search_backend) else None,
        ),
        (
            "Max6 genotype compression",
            "exact identity" if args.ld_tag_r2 == 1.0
            else f"approximate LD r2 >= {args.ld_tag_r2:g}",
        ),
        (
            "Post-screen test",
            (
                f"GRM-INT 1df; Max6 Top-{int(args.top_rules_runtime)} per unit; rank by interaction p"
                if args.grm_int
                else "GRM-Max6 exact candidate rerank"
            ),
        ),
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
    _emit_garfield_command_to_log(logger)
    if bool(args.dev) and int(args.null_repeats) < 50:
        logger.warning(
            "Development run uses fewer than 50 Max6 null replicates; "
            "threshold/FWER outputs are timing diagnostics only and are not statistically interpretable."
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
        het_threshold=float(args.het),
    )
    sample_ids = np.asarray(sample_ids, dtype=str)
    if len(sample_ids) == 0:
        raise ValueError("No sample IDs found in genotype input.")
    sample_index_map = {sid: i for i, sid in enumerate(sample_ids.tolist())}

    aligned_grm = None
    resolved_grm_id = None
    grm_sample_ids = sample_ids.tolist()
    grm_index_map = sample_index_map
    if args.grm:
        grm_src = os.path.basename(str(args.grm))
        with CliStatus(f"Loading GRM from {grm_src}...", enabled=use_spinner) as task:
            try:
                resolved_grm_id = resolve_grm_id_path(str(args.grm))
                if resolved_grm_id is not None:
                    available_grm_ids = set(read_id_file(resolved_grm_id))
                    grm_sample_ids = [sid for sid in sample_ids.tolist() if sid in available_grm_ids]
                    if not grm_sample_ids:
                        raise ValueError("No overlapping sample IDs across genotype and GARFIELD GRM.")
                grm_index_map = {sid: i for i, sid in enumerate(grm_sample_ids)}
                aligned_grm, resolved_grm_id = load_and_align_grm(
                    str(args.grm),
                    grm_sample_ids,
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
        common &= set(grm_sample_ids)
    if cov_ids is not None:
        common &= set(cov_ids.astype(str))
    sample_pool = [sid for sid in geno_ids.tolist() if sid in common]
    if len(sample_pool) == 0:
        raise ValueError("No overlapping samples across genotype/phenotype/GRM/cov.")
    _emit_garfield_file_only_line(
        logger,
        f"Sample intersection: retained {len(sample_pool)}/{len(geno_ids)} genotype samples; "
        f"missing phenotype ID={len(set(geno_ids) - set(pheno_ids_all))}, "
        f"missing GRM ID={len(set(geno_ids) - set(grm_sample_ids))}, "
        f"missing covariate ID={0 if cov_ids is None else len(set(geno_ids) - set(cov_ids))} "
        "(exclusion counts may overlap; nonfinite phenotypes are filtered per trait).",
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

    group_labels: list[str] = []
    group_intervals: list[list[tuple[str, int, int]]] = []
    null_group_intervals: Optional[list[list[tuple[str, int, int]]]] = None
    scan_unit_total: Optional[int] = None
    if args.scan_mode == "window":
        scan_unit_total = _count_window_scan_units_from_bim(
            str(gfile),
            extension=int(args.extension),
            step=int(args.step),
        )
    elif args.scan_mode == "wholegenome":
        scan_unit_total = 1
    global_site_keep: Optional[np.ndarray] = None
    use_prepared_max6 = bool(
        _is_ld_tag_max6_backend(args.search_backend) and not max6_audit_requested
    )
    if bool(args.global_stats) and not use_prepared_max6:
        global_site_keep = _prepare_site_keep(
            genofile=gfile,
            sample_ids=list(sample_pool),
            sample_index_map=sample_index_map,
            n_snps=int(_n_snps),
            maf_threshold=float(args.maf),
            max_missing_rate=float(args.geno),
            het_threshold=float(args.het),
            snps_only=False,
            threads=int(args.thread),
            use_spinner=use_spinner,
            global_stats=True,
        )
    if args.scan_mode in {"gene", "geneset"}:
        if len(args.genefiles) <= 0 or not args.gff3:
            raise ValueError(
                f"scan-mode={args.scan_mode} requires one or more -g/--genefile and -gff/--gff3."
            )
        group_labels, group_intervals = _build_interval_groups(
            args.genefiles,
            args.gff3,
            int(args.extension),
            args.scan_mode,
        )
        if len(group_intervals) == 0:
            raise ValueError(f"No valid groups built for scan-mode={args.scan_mode}.")
        if args.scan_mode in {"gene", "geneset"}:
            null_group_intervals = _build_all_gene_interval_groups(
                args.gff3,
                int(args.extension),
            )
        scan_unit_total = len(group_intervals)

    grm_n: int | str = "NA" if aligned_grm is None else int(aligned_grm.shape[0])
    cov_n: int | str = "NA" if cov_ids is None else int(len(cov_ids))
    prepared_max6_tags = None
    prepared_max6_universe = None
    prepared_sample_ids: Optional[list[str]] = None
    if use_prepared_max6:
        if (
            prepare_max6_genotype_tags is None
            or finalize_max6_genotype_universe is None
        ):
            raise RuntimeError(
                "The installed JanusX native extension does not expose the shared "
                "Max6 genotype-preparation API; rebuild the extension before using "
                "the prepared max6-grmmax6 path."
            ) from _RUST_IMPORT_ERROR
        prepared_sample_ids = list(sample_pool)
        prepared_sample_indices = [
            int(sample_index_map[sid]) for sid in prepared_sample_ids
        ]
        prep_groups = (
            group_intervals
            if args.scan_mode not in {"window", "wholegenome"}
            else None
        )
        prep_group_names = (
            group_labels
            if args.scan_mode not in {"window", "wholegenome"}
            else None
        )
        with CliStatus(
            MAX6_LD_TAG_LOADING_DESC,
            enabled=use_spinner,
            use_process=True,
        ) as task:
            try:
                prepared_max6_tags = prepare_max6_genotype_tags(
                    str(gfile),
                    sample_ids=prepared_sample_ids,
                    sample_indices=prepared_sample_indices,
                    unit_kind=_scan_mode_to_logic_unit_kind(args.scan_mode),
                    groups=prep_groups,
                    group_names=prep_group_names,
                    ld_tag_r2=float(args.ld_tag_r2),
                    extension=int(args.extension),
                    step=int(args.step),
                    scan_bimranges=args.bimrange,
                    maf_threshold=float(args.maf),
                    max_missing_rate=float(args.geno),
                    het_threshold=float(args.het),
                    snps_only=False,
                    threads=int(args.thread),
                    logic_mem_budget_bytes=int(
                        max(0.0, float(args.logic_memory_mb)) * 1024.0 * 1024.0
                    ),
                )
            except BaseException:
                task.fail("Loading Max6 LD tags ...Failed")
                raise
            task.complete(
                "Loading Max6 LD tags "
                f"(units={int(prepared_max6_tags.unit_count)}, "
                f"tags={format_max6_count(prepared_max6_tags.tag_count)})...Finished"
            )
        with CliStatus(
            MAX6_BONFERRONI_LOADING_DESC,
            enabled=use_spinner,
            use_process=True,
        ) as task:
            try:
                prepared_max6_universe = finalize_max6_genotype_universe(
                    prepared_max6_tags
                )
            except BaseException:
                task.fail("Preparing the unique-pair Bonferroni family...Failed")
                raise
            task.complete(
                "Unique-pair family ready "
                f"(N_unique={format_max6_count(prepared_max6_universe.unique_pair_count)})...Finished"
            )
        if bool(args.global_stats) and aligned_grm is None:
            global_site_keep = np.asarray(
                prepared_max6_universe.global_site_keep,
                dtype=np.bool_,
            )
        prepared_max6_tags = None

    _emit_plain_info_line(
        logger,
        (
            f"geno={len(geno_ids)}, pheno={len(pheno_ids_all)}, "
            f"grm={grm_n}, q=NA, cov={cov_n} -> {len(sample_pool)}"
        ),
        use_spinner=use_spinner,
    )

    split_line = "-" * 60
    if scan_unit_total is not None and int(scan_unit_total) > 0:
        _emit_plain_info_line(
            logger,
            f"Prepared {int(scan_unit_total)} scan unit(s) for {args.scan_mode} scan.",
            use_spinner=use_spinner,
        )
    _emit_plain_info_line(
        logger,
        split_line,
        use_spinner=use_spinner,
    )

    used_trait_labels: dict[str, int] = {}
    saved = 0
    summary_rows: list[dict[str, object]] = []
    garfield_manifest_traits: list[dict[str, object]] = []

    for trait_idx, trait in enumerate(pheno.columns):
        if trait_idx > 0:
            logger.info("")

        trait_name = str(trait)
        pheno_col = pheno[trait].dropna()
        pheno_col = pheno_col[np.isfinite(pheno_col.to_numpy(dtype=float))]
        pheno_ids = set(pheno_col.index.astype(str).to_numpy())
        common_ids = [sid for sid in sample_pool if sid in pheno_ids]
        if len(common_ids) == 0:
            logger.warning(f"No overlapping samples for trait '{trait_name}' after dropna; skipped.")
            continue

        if _is_max6_grm_backend(args.search_backend):
            logic_maf_threshold, logic_maf_source = 0.0, "not_applicable_max6"
        else:
            logic_maf_threshold, logic_maf_source = _resolve_logic_maf_threshold(
                args.lmaf,
                len(common_ids),
            )
        y_raw = pheno_col.loc[common_ids].to_numpy(dtype=float)
        response_mode, y_common, response_note = _detect_response_mode(y_raw)
        _emit_plain_info_line(
            logger,
            f"{trait_name} (n={len(common_ids)})",
            use_spinner=use_spinner,
        )
        site_keep_trait = global_site_keep
        if prepared_max6_universe is not None and aligned_grm is not None:
            # External-GRM Max6 only needs trait-sample QC for the frozen tag
            # rows; the native consumer performs that bounded source-row QC.
            site_keep_trait = None
        elif site_keep_trait is None:
            site_keep_trait = _prepare_site_keep(
                genofile=gfile,
                sample_ids=list(common_ids),
                sample_index_map=sample_index_map,
                n_snps=int(_n_snps),
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
                het_threshold=float(args.het),
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
                [grm_index_map[sid] for sid in common_ids],
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
            "wholegenome": "Scan Whole Genome",
            "gene": "Scan Genes",
            "genepair": "Scan Gene Pairs",
            "geneset": "Scan Gene Sets",
        }.get(str(args.scan_mode), f"Rust GARFIELD search for '{trait_name}'")
        logic_unit_kind = _scan_mode_to_logic_unit_kind(args.scan_mode)
        rust_groups = group_intervals if args.scan_mode not in {"window", "wholegenome"} else None
        rust_group_names = group_labels if args.scan_mode not in {"window", "wholegenome"} else None
        rust_null_groups = null_group_intervals if args.scan_mode in {"gene", "geneset"} else None
        trait_logic_prefix = f"{trait_outprefix}.garfield"
        if _is_max6_grm_backend(args.search_backend):
            min_cell_effective = max(
                int(args.min_cell_runtime),
                int(np.ceil(float(args.min_cell_frac) * len(common_ids))),
            )
            _emit_plain_info_line(
                logger,
                f"Max6 cell threshold: {min_cell_effective} "
                f"(count={args.min_cell_runtime}, fraction={args.min_cell_frac:g}, "
                f"Neff={len(common_ids)})",
                use_spinner=use_spinner,
            )

        def _on_scan_stage(stage, done, total, meta):
            if stage == "garfield_null_summary":
                null_summary = json.loads(meta)
                pve = null_summary.get("full_pve")
                _emit_plain_info_line(
                    logger,
                    f"GARFIELD null-model PVE for '{trait_name}': "
                    f"full={float(pve) if pve is not None else float('nan'):.4g}",
                    use_spinner=use_spinner,
                )

        # Rust handles residualization, genotype-only LD-tag compression,
        # exhaustive Max6 screening, and the selected GRM-INT or exact
        # GRM-Max6 scoring route.
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
                null_groups=rust_null_groups,
                group_names=rust_group_names,
                extension=int(args.extension),
                step=int(args.step),
                scan_bimranges=args.bimrange,
                bin_mode="bin",
                ml_method=str(engine_runtime).lower(),
                ml_importance="imp",
                ml_top_k=int(args.ml_top_k_runtime),
                ml_top_frac=0.0,
                # Rust also enforces this lower bound, but normalizing it at
                # the CLI keeps the requested and effective repeat counts
                # aligned in the result manifest.
                permutation_repeats=max(50, int(args.null_repeats)),
                permutation_scoring="auto",
                rule_null_penalty_method=str(args.rule_null_penalty_method_runtime),
                rule_null_quantile=float(args.rule_null_quantile_runtime),
                rule_null_report_pvalue=bool(args.rule_null_report_pvalue_runtime),
                n_estimators=100,
                max_depth=int(args.layer) + 1,
                min_samples_leaf=1,
                min_samples_split=2,
                bootstrap=True,
                feature_subsample=0.0,
                seed=trait_seed,
                max_pick=int(args.layer),
                exhaustive_depth=int(args.exhaustive_depth_runtime),
                beam_width=int(args.beam_width),
                rank_score=str(args.rank_score),
                maf_threshold=float(args.maf),
                logic_maf_threshold=float(logic_maf_threshold),
                max_missing_rate=float(args.geno),
                het_threshold=float(args.het),
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
                residual_method=str(args.residual_method),
                pcg_max_iter=int(args.pcg_max_iter),
                pcg_tol=float(args.pcg_tol),
                pcg_small_n_max=int(args.pcg_small_n_max),
                out_prefix=trait_logic_prefix,
                simbench_path=args.simbench,
                top_rules_per_unit=int(args.top_rules_runtime),
                max_output_rules=int(args.max_output_rules_runtime),
                max_output_ratio=float(args.max_output_ratio_runtime),
                search_backend=str(args.search_backend),
                rule_permutation=bool(args.rule_permutation_runtime),
                prior_len=None,
                no_clean=bool(args.no_clean),
                raw_design=bool(args.raw_design),
                filter_xor_substates=not bool(args.disable_xor_substate_filter),
                xor_search=bool(args.xor_search),
                whole_genome_dev_mode=bool(str(args.scan_mode).lower() == "wholegenome"),
                progress_callback=progress_cb,
                progress_every=0,
                logic_mem_budget_bytes=int(
                    max(0.0, float(args.logic_memory_mb)) * 1024.0 * 1024.0
                ),
                diagnostics=bool(args.diagnostics),
                sampled_screen_null_diagnostic=bool(args.sampled_screen_null_diagnostic),
                max6_screen_exceedance_audit=bool(args.max6_screen_exceedance_audit),
                max6_screen_pot_audit=bool(args.max6_screen_pot_audit),
                sampled_screen_pair_seed=int(args.sampled_screen_pair_seed),
                sampled_screen_pair_seeds=str(args.sampled_screen_pair_seeds_normalized),
                sampled_screen_maf_cutpoints=str(args.sampled_screen_maf_cutpoints),
                sampled_screen_r2_cutpoints=str(args.sampled_screen_r2_cutpoints),
                max6_screen_tail_audit=bool(args.max6_screen_tail_audit),
                max6_screen_tail_audit_repeats=int(args.max6_screen_tail_audit_repeats),
                max6_screen_tail_pair_seed=int(args.max6_screen_tail_pair_seed),
                max6_screen_tail_support_aware=bool(args.max6_screen_tail_support_aware),
                max6_screen_tail_support_shares=list(args.max6_screen_tail_support_shares_normalized),
                prepared_max6_universe=prepared_max6_universe,
                prepared_sample_ids=prepared_sample_ids,
                grm_int_only=bool(args.grm_int),
                min_cell=int(args.min_cell_runtime),
                min_cell_fraction=float(args.min_cell_frac),
                ld_tag_r2=float(args.ld_tag_r2),
                summary_callback=_on_scan_stage,
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
        _emit_garfield_bg_noise_summary_to_log(
            logger,
            trait_name=trait_name,
            bg_noise_summary=result.get("bg_noise_summary"),
            use_spinner=use_spinner,
        )
        grm_int_pwald_reference = (
            pwald_bonferroni_reference(result.get("max6_pair_universe_size", 0))
            if args.grm_int
            else None
        )
        if _is_max6_grm_backend(args.search_backend):
            screen_method = result.get("max6_screen_threshold_method")
            if _is_ld_tag_max6_backend(args.search_backend) and (
                args.grm_int or not screen_method
            ):
                if args.grm_int:
                    _emit_garfield_file_only_line(
                        logger,
                        (
                            f"Max6 -> GRM-INT: topk={int(args.top_rules_runtime)} per scan unit; "
                            "interaction-only one-df FWL score; output ranked by grm_int_chisq; "
                            "pwald=chi2_1; Bonferroni threshold is a reference only"
                        ),
                    )
                    if grm_int_pwald_reference is not None:
                        _emit_garfield_file_only_line(
                            logger,
                            (
                                "GRM-INT pwald Bonferroni reference line: "
                                f"N_unique={format_max6_count(grm_int_pwald_reference['n_unique_pairs'])}, "
                                f"alpha={grm_int_pwald_reference['alpha']:.3g}, "
                                f"p_cut={grm_int_pwald_reference['p_cut']:.6g}, "
                                f"-log10(p_cut)={grm_int_pwald_reference['minus_log10_p_cut']:.3f}; "
                                "one chi-square(1) test per unique pair; not applied to output"
                            ),
                        )
                else:
                    _emit_garfield_file_only_line(
                        logger,
                        (
                            "GRM-Max6 pwald Bonferroni reference line: "
                            f"N_unique={format_max6_count(result.get('max6_pair_universe_size', 0))}, "
                            f"m={int(result.get('max6_bonferroni_channel_count', 0))}, "
                            f"alpha={result.get('max6_bonferroni_alpha')}, "
                            f"p_cut={result.get('max6_bonferroni_p_cut')}; "
                            "pair column=pwald; "
                            "screen=deterministic Top-5000; null calibration=inactive"
                        ),
                    )
            elif screen_method:
                _emit_garfield_file_only_line(
                    logger,
                    (
                        "Max6 screen operational threshold: "
                        f"quantile={_format_max6_score_value(args.rule_null_quantile_runtime)}, "
                        f"empirical_q={_format_max6_score_value(result.get('max6_screen_threshold'))}, "
                        f"gumbel_diagnostic_q={_format_max6_score_value(result.get('max6_gumbel_score_threshold'))}, "
                        f"null_repeats={int(result.get('max6_screen_null_repeats', 0))}, "
                        f"pair_universe_unique={format_max6_count(result.get('max6_screen_pair_universe_size', 0))}, "
                        f"window_pair_occurrences={int(result.get('max6_screen_window_pair_occurrences', 0))}, "
                        f"calibration={float(result.get('timing_max6_screen_calibration_wall_s', 0.0)):.3f}s "
                        f"({float(result.get('timing_max6_screen_calibration_wall_s', 0.0)) / max(1, int(result.get('max6_screen_null_repeats', 0))):.6f}s/null), "
                        f"validation={float(result.get('timing_max6_screen_validation_wall_s', 0.0)):.3f}s "
                        f"({float(result.get('timing_max6_screen_validation_wall_s', 0.0)) / max(1, int(result.get('max6_screen_validation_repeats', 0))):.6f}s/null), "
                        f"screen_candidates={int(result.get('max6_screen_candidate_count', 0))}; "
                        f"validation={int(result.get('max6_screen_validation_rejections', 0))}/"
                        f"{int(result.get('max6_screen_validation_repeats', 0))} null maxima, "
                        f"FWER={_format_max6_score_value(result.get('max6_screen_validation_fwer'))}, "
                        f"95% exact CI=[{_format_max6_score_value(result.get('max6_screen_validation_ci_lower'))}, "
                        f"{_format_max6_score_value(result.get('max6_screen_validation_ci_upper'))}]; "
                        f"D_final subset audit={'PASS' if result.get('max6_final_screen_subset_audit_passed') else 'NOT PASS'}, "
                        f"checked={int(result.get('max6_final_screen_subset_audit_pairs_checked', 0))}, "
                        f"violations={int(result.get('max6_final_screen_subset_audit_violations', 0))}; "
                        "scope=weak_global_null; final_output=subset_of_screen_pass; "
                        f"calibration_maxima={result.get('max6_screen_maxima_tsv') or 'not saved (enable --diagnostics)'}, "
                        f"validation_maxima={result.get('max6_screen_validation_maxima_tsv') or 'not saved (enable --diagnostics)'}"
                    ),
                )
            if result.get("max6_screen_exceedance_audit_manifest_json"):
                _emit_garfield_file_only_line(
                    logger,
                    (
                        "Max6 weighted exceedance audit completed using paired full-MaxT calibration nulls; "
                        f"wall={float(result.get('timing_max6_screen_exceedance_audit_wall_s', 0.0)):.3f}s, "
                        f"manifest={result.get('max6_screen_exceedance_audit_manifest_json')}"
                    ),
                )
            if result.get("max6_screen_pot_audit_manifest_json"):
                _emit_garfield_file_only_line(
                    logger,
                    (
                        "Max6 marginal POT sample audit completed using calibration nulls; "
                        f"wall={float(result.get('timing_max6_screen_pot_audit_wall_s', 0.0)):.3f}s, "
                        f"manifest={result.get('max6_screen_pot_audit_manifest_json')}"
                    ),
                )
            if not _is_ld_tag_max6_backend(args.search_backend):
                _emit_garfield_file_only_line(
                    logger,
                    (
                        "Max6 -> GRM-Max6 legacy threshold: "
                        f"quantile={_format_max6_score_value(args.rule_null_quantile_runtime)}, "
                        f"empirical_q={_format_max6_score_value(result.get('max6_score_threshold'))}, "
                        f"gumbel_moment_q={_format_max6_score_value(result.get('max6_gumbel_score_threshold'))}, "
                        f"n_permutations={int(result.get('permutation_null_repeats', 0))}, "
                        f"pairs_per_permutation={int(result.get('max6_permutation_pairs_per_repeat', 0))}, "
                        f"pair_pool={int(result.get('max6_permutation_pair_pool_size', 0))}, "
                        f"n_max_scores={int(result.get('max6_permutation_score_count', 0))}; "
                        "legacy threshold=empirical order statistic; "
                        f"maxima={result.get('max6_permutation_maxima_tsv') or 'not saved (enable --diagnostics)'}"
                    ),
                )
            if _is_ld_tag_max6_backend(args.search_backend):
                _emit_garfield_file_only_line(
                    logger,
                    (
                        "Max6 pair universe: genotype-tag exhaustive; "
                        f"compression={result.get('max6_ld_tag_mode')}; "
                        f"r2_threshold={result.get('max6_ld_tag_r2')}, "
                        f"tag_map={result.get('max6_ld_tags_tsv') or 'not written'}"
                    ),
                )
            if _is_max6_grm_backend(args.search_backend) and int(
                result.get("max6_observed_units_timed", 0)
            ):
                observed_rerank_stages = [
                    (
                        "endpoint_prepare",
                        float(result.get("timing_max6_observed_rerank_endpoint_prepare_task_sum_s", 0.0)),
                    ),
                    (
                        "PCX",
                        float(result.get("timing_max6_observed_rerank_pcx_task_sum_s", 0.0)),
                    ),
                    (
                        "candidate_geometry",
                        float(result.get("timing_max6_observed_rerank_geometry_task_sum_s", 0.0)),
                    ),
                    (
                        "mincell_q",
                        float(result.get("timing_max6_observed_rerank_q_task_sum_s", 0.0)),
                    ),
                    (
                        "score_assembly",
                        float(result.get("timing_max6_observed_rerank_score_task_sum_s", 0.0)),
                    ),
                    (
                        "Top-K",
                        float(result.get("timing_max6_observed_rerank_topk_task_sum_s", 0.0)),
                    ),
                    (
                        "other",
                        float(result.get("timing_max6_observed_rerank_other_task_sum_s", 0.0)),
                    ),
                ]
                dominant_stage, dominant_seconds = max(
                    observed_rerank_stages, key=lambda item: item[1]
                )
                rerank_task_sum = sum(value for _, value in observed_rerank_stages)
                dominant_share = (
                    100.0 * dominant_seconds / rerank_task_sum
                    if rerank_task_sum > 0.0
                    else 0.0
                )
                if args.grm_int:
                    observed_score_summary = (
                        "GRM-INT_score_task_sum="
                        f"{float(result.get('timing_max6_observed_grm_int_task_sum_s', 0.0)):.3f}s, "
                        f"batches={int(result.get('max6_observed_grm_int_batches', 0))}, "
                        "invalid_singular="
                        f"{int(result.get('max6_observed_grm_int_singular_endpoint_design_pairs', 0))}, "
                        "invalid_no_interaction_info="
                        f"{int(result.get('max6_observed_grm_int_negligible_interaction_information_pairs', 0))}, "
                        "invalid_nonfinite="
                        f"{int(result.get('max6_observed_grm_int_nonfinite_pairs', 0))}"
                    )
                    scored_candidate_count = int(
                        result.get("max6_observed_grm_int_pair_occurrences", 0)
                    )
                    scored_candidate_label = "GRM-INT_scored_occurrences"
                else:
                    observed_score_summary = (
                        f"exact_rerank_task_sum={float(result.get('timing_max6_observed_rerank_kernel_task_sum_s', 0.0)):.3f}s "
                        f"[endpoint={observed_rerank_stages[0][1]:.3f}s, "
                        f"PCX={observed_rerank_stages[1][1]:.3f}s, "
                        f"geometry={observed_rerank_stages[2][1]:.3f}s, "
                        f"q={observed_rerank_stages[3][1]:.3f}s "
                        f"(P_C validation={float(result.get('timing_max6_observed_rerank_q_validation_task_sum_s', 0.0)):.3f}s, "
                        f"quadratic={float(result.get('timing_max6_observed_rerank_q_quadratic_task_sum_s', 0.0)):.3f}s), "
                        f"geometry_detail(marker_moments={float(result.get('timing_max6_observed_rerank_geometry_marker_moments_task_sum_s', 0.0)):.3f}s, "
                        f"pairwise={float(result.get('timing_max6_observed_rerank_geometry_pairwise_task_sum_s', 0.0)):.3f}s, "
                        f"lookup={float(result.get('timing_max6_observed_rerank_geometry_lookup_task_sum_s', 0.0)):.3f}s, "
                        f"interaction_py={float(result.get('timing_max6_observed_rerank_geometry_interaction_py_task_sum_s', 0.0)):.3f}s, "
                        f"fused_AU_Py={float(result.get('timing_max6_observed_au_fused_py_task_sum_s', 0.0)):.3f}s (included in pairwise if selected), "
                        f"unclassified={float(result.get('timing_max6_observed_rerank_geometry_unclassified_task_sum_s', 0.0)):.3f}s), "
                        f"score={observed_rerank_stages[4][1]:.3f}s, "
                        f"Top-K={observed_rerank_stages[5][1]:.3f}s, "
                        f"other={observed_rerank_stages[6][1]:.3f}s]; "
                        f"largest={dominant_stage} ({dominant_share:.1f}% of measured rerank stages)"
                    )
                    scored_candidate_count = int(
                        result.get("max6_observed_exact_candidate_pairs", 0)
                    )
                    scored_candidate_label = "exact_scored_occurrences"
                _emit_garfield_file_only_line(
                    logger,
                    (
                        f"Observed Max6 -> {'GRM-INT' if args.grm_int else 'GRM-Max6'} timing (summed window-task elapsed time; "
                        "parallel stages overlap): "
                        f"scan_wall={float(result.get('timing_scan_wall_s', 0.0)):.3f}s, "
                        f"windows={int(result.get('max6_observed_units_timed', 0))}, "
                        f"screen_candidates={int(result.get('max6_observed_screen_candidate_pairs', 0))}, "
                        f"screen_batches={int(result.get('max6_observed_screen_batch_count', 0))}, "
                        f"batch_unique_pairs={int(result.get('max6_observed_screen_batch_unique_scored_pairs', 0))}, "
                        f"batch_score_offers={int(result.get('max6_observed_screen_batch_score_offers', 0))}, "
                        f"batch_direct_fallbacks={int(result.get('max6_observed_screen_batch_direct_fallbacks', 0))}; "
                        f"{scored_candidate_label}={scored_candidate_count}; "
                        f"screen_input={float(result.get('timing_max6_observed_screen_input_prepare_task_sum_s', 0.0)):.3f}s, "
                        f"Max6_screen={float(result.get('timing_max6_observed_screen_kernel_task_sum_s', 0.0)):.3f}s, "
                        f"candidate_setup={float(result.get('timing_max6_observed_candidate_prepare_task_sum_s', 0.0)):.3f}s, "
                        f"{observed_score_summary}; "
                        f"output_finalize={float(result.get('timing_max6_observed_output_finalize_task_sum_s', 0.0)):.3f}s"
                    ),
                )
                batch_count = int(result.get("max6_observed_screen_batch_count", 0))
                if batch_count and "timing_max6_batch_pair_scan_merge_wall_s" in result:
                    wall_stages = {
                        name: float(result.get(f"timing_max6_batch_{name}_wall_s", 0.0))
                        for name in (
                            "batch_prepare", "membership_prepare", "pair_scan_merge",
                            "finalists", "grm_int", "result_drain",
                        )
                    }
                    accounted_wall = sum(wall_stages.values())
                    scan_wall = float(result.get("timing_scan_wall_s", 0.0))
                    _emit_garfield_file_only_line(
                        logger,
                        "Max6 batch phase timing (disjoint wall spans): "
                        + ", ".join(f"{name}={seconds:.3f}s" for name, seconds in wall_stages.items())
                        + f"; accounted={accounted_wall:.3f}s, scan_wall={scan_wall:.3f}s, "
                        + f"unaccounted={scan_wall - accounted_wall:.3f}s; "
                        + "heap_merge_task_sum="
                        + f"{float(result.get('timing_max6_batch_heap_merge_task_sum_s', 0.0)):.3f}s "
                        + "(overlaps pair_scan_merge; not additive); "
                        + f"parallel_batches={int(result.get('max6_batch_parallel_count', 0))}, "
                        + f"serial_batches={int(result.get('max6_batch_serial_count', 0))}, "
                        + f"shared_path_source_rows_sum={int(result.get('max6_batch_source_rows_sum', 0))}, "
                        + f"source_rows_max={int(result.get('max6_batch_source_rows_max', 0))}, "
                        + f"memberships_sum={int(result.get('max6_batch_memberships_sum', 0))}; "
                        + "direct fallback timing includes its per-unit prepare/finalize work; "
                        + "grm_int wall includes unit setup/finalization; "
                        + "result_drain includes four-cell recount, TSV writes and progress callbacks",
                    )
                batch_profiles = result.get("max6_batch_scan_profiles") or []
                if batch_profiles:
                    def _batch_quantile(values, probability):
                        ordered = sorted(float(value) for value in values)
                        if not ordered:
                            return float("nan")
                        index = max(
                            0,
                            min(len(ordered) - 1, int(np.ceil(probability * len(ordered))) - 1),
                        )
                        return ordered[index]

                    tail_values = [row[11] for row in batch_profiles]
                    skew_values = [row[10] for row in batch_profiles]
                    fold_max_values = [row[3] for row in batch_profiles]
                    merge_path_values = [row[15] for row in batch_profiles]
                    scan_reduce_path_values = [row[16] for row in batch_profiles]
                    anchor_imbalance = [row[5] / row[4] for row in batch_profiles if row[4] > 0]
                    worker_imbalance = [row[9] / row[7] for row in batch_profiles if row[7] > 0]
                    topk_merge_task_sum = sum(row[12] for row in batch_profiles)
                    topk_merge_wall_max = max(row[13] for row in batch_profiles)
                    topk_merge_count = sum(int(row[14]) for row in batch_profiles)
                    _emit_garfield_file_only_line(
                        logger,
                        "Max6 batch worker/Top-K diagnostics (per-batch p50/p95/max): "
                        + f"post_last_fold_tail={_batch_quantile(tail_values, .50):.4f}/"
                        + f"{_batch_quantile(tail_values, .95):.4f}/{max(tail_values):.4f}s, "
                        + f"worker_finish_skew={_batch_quantile(skew_values, .50):.4f}/"
                        + f"{_batch_quantile(skew_values, .95):.4f}/{max(skew_values):.4f}s, "
                        + f"fold_task_max={_batch_quantile(fold_max_values, .50):.4f}/"
                        + f"{_batch_quantile(fold_max_values, .95):.4f}/{max(fold_max_values):.4f}s, "
                        + f"fold_anchor_max/min_ratio={_batch_quantile(anchor_imbalance, .50):.2f}/"
                        + f"{_batch_quantile(anchor_imbalance, .95):.2f}/{max(anchor_imbalance, default=float('nan')):.2f}, "
                        + f"worker_task_wall_max/min_ratio={_batch_quantile(worker_imbalance, .50):.2f}/"
                        + f"{_batch_quantile(worker_imbalance, .95):.2f}/{max(worker_imbalance, default=float('nan')):.2f}, "
                        + f"TopK_merge_critical_path={_batch_quantile(merge_path_values, .50):.4f}/"
                        + f"{_batch_quantile(merge_path_values, .95):.4f}/{max(merge_path_values):.4f}s, "
                        + f"scan_plus_merge_critical_path={_batch_quantile(scan_reduce_path_values, .50):.4f}/"
                        + f"{_batch_quantile(scan_reduce_path_values, .95):.4f}/{max(scan_reduce_path_values):.4f}s; "
                        + f"TopK_merge_task_sum={topk_merge_task_sum:.3f}s, "
                        + f"max_merge_call={topk_merge_wall_max:.6f}s, calls={topk_merge_count}; "
                        + "fold and Top-K task sums overlap wall time",
                    )
                reuse_diagnostic = result.get("max6_candidate_reuse_diagnostic")
                if isinstance(reuse_diagnostic, dict) and reuse_diagnostic.get("enabled"):
                    screen_profile = reuse_diagnostic.get("screen_microprofile") or {}
                    _emit_garfield_file_only_line(
                        logger,
                        (
                            "Max6 candidate-reuse diagnostic: "
                            f"screen_pairs={int(reuse_diagnostic.get('screen_pairs_evaluated', 0))}, "
                            f"screen={float(reuse_diagnostic.get('screen_kernel_task_seconds', 0.0)):.3f}s "
                            f"({float(reuse_diagnostic.get('screen_kernel_nanoseconds_per_evaluated_pair', 0.0)):.1f} ns/evaluated pair); "
                            f"scored_occurrences={int(reuse_diagnostic.get('scored_candidate_pair_occurrences', 0))}, "
                            f"global_unique_pairs={int(reuse_diagnostic.get('global_unique_scored_pairs', 0))}, "
                            f"cross_window_duplicate_fraction={float(reuse_diagnostic.get('duplicate_fraction_of_scored_occurrences', 0.0)):.3%}, "
                            f"pair_multiplicity p50/p90/p99/max="
                            f"{int(reuse_diagnostic.get('pair_multiplicity_p50', 0))}/"
                            f"{int(reuse_diagnostic.get('pair_multiplicity_p90', 0))}/"
                            f"{int(reuse_diagnostic.get('pair_multiplicity_p99', 0))}/"
                            f"{int(reuse_diagnostic.get('pair_multiplicity_max', 0))}; "
                            f"endpoint reuse={float(reuse_diagnostic.get('endpoint_reuse_factor', 0.0)):.3f}x, "
                            f"pair-key buffer={int(reuse_diagnostic.get('candidate_pair_key_buffer_bytes', 0)) / (1024 * 1024):.1f} MiB, "
                            f"dedup={float(reuse_diagnostic.get('candidate_key_sort_and_summary_wall_seconds', 0.0)):.3f}s; "
                            f"screen profile={int(screen_profile.get('profiled_windows', 0))} windows/"
                            f"{int(screen_profile.get('sampled_pairs', 0))} sampled pairs, "
                            f"bitset={float((screen_profile.get('stage_seconds') or {}).get('bitset_intersection_popcount_seconds', 0.0)):.3f}s, "
                            f"weighted-y={float((screen_profile.get('stage_seconds') or {}).get('weighted_y_lookup_seconds', 0.0)):.3f}s, "
                            f"score={float((screen_profile.get('stage_seconds') or {}).get('score_assembly_seconds', 0.0)):.3f}s, "
                            f"heap={float((screen_profile.get('stage_seconds') or {}).get('heap_topk_seconds', 0.0)):.3f}s"
                        ),
                    )
                au_kernel = os.environ.get(
                    "JX_GARFIELD_EXPERIMENTAL_CANDIDATE_AU_KERNEL", "reference"
                )
                if au_kernel != "reference":
                    _emit_garfield_file_only_line(
                        logger,
                        (
                            f"Experimental candidate A/U kernel={au_kernel}; "
                            f"allocation={float(result.get('timing_max6_observed_au_allocation_task_sum_s', 0.0)):.3f}s, "
                            f"X_times_Z={float(result.get('timing_max6_observed_au_materialization_task_sum_s', 0.0)):.3f}s, "
                            f"A/H_GEMM={float(result.get('timing_max6_observed_au_gemm_task_sum_s', 0.0)):.3f}s, "
                            f"candidate_lookup={float(result.get('timing_max6_observed_au_candidate_lookup_task_sum_s', 0.0)):.3f}s, "
                            f"copy={float(result.get('timing_max6_observed_au_copy_task_sum_s', 0.0)):.3f}s, "
                            f"validation={float(result.get('timing_max6_observed_au_validation_task_sum_s', 0.0)):.3f}s, "
                            f"max_extra_workspace={int(result.get('timing_max6_observed_au_max_workspace_bytes', 0))} bytes"
                        ),
                    )
            if not screen_method and not _is_ld_tag_max6_backend(args.search_backend):
                _emit_garfield_file_only_line(
                    logger,
                    (
                        "Max6 permutation timing (wall / summed task time): "
                        f"setup={_format_max6_score_value(result.get('timing_max6_permutation_setup_wall_s'))}s, "
                        f"score_wall={_format_max6_score_value(result.get('timing_max6_permutation_score_wall_s'))}s, "
                        f"project_PCy={_format_max6_score_value(result.get('timing_max6_permutation_projected_y_task_sum_s'))}s, "
                        f"endpoint_prepare={_format_max6_score_value(result.get('timing_max6_permutation_endpoint_prepare_task_sum_s'))}s, "
                        f"rerank={_format_max6_score_value(result.get('timing_max6_permutation_rerank_kernel_task_sum_s'))}s "
                        "[PCX="
                        f"{_format_max6_score_value(result.get('timing_max6_permutation_rerank_pcx_task_sum_s'))}s, "
                        f"geometry={_format_max6_score_value(result.get('timing_max6_permutation_rerank_geometry_task_sum_s'))}s, "
                        f"q={_format_max6_score_value(result.get('timing_max6_permutation_rerank_q_task_sum_s'))}s, "
                        f"score={_format_max6_score_value(result.get('timing_max6_permutation_rerank_score_task_sum_s'))}s, "
                        f"topk={_format_max6_score_value(result.get('timing_max6_permutation_rerank_topk_task_sum_s'))}s, "
                        f"other={_format_max6_score_value(result.get('timing_max6_permutation_rerank_other_task_sum_s'))}s], "
                        f"threshold={_format_max6_score_value(result.get('timing_max6_permutation_threshold_wall_s'))}s, "
                        f"maxima_write={_format_max6_score_value(result.get('timing_max6_permutation_maxima_write_wall_s'))}s; "
                        f"pair_score_calls={int(result.get('timing_max6_permutation_pair_score_count', 0))}"
                    ),
                )
            _emit_garfield_file_only_line(
                logger,
                (
                    "Max6 -> GRM-Max6 score scale: "
                    f"sigma_g2={_format_max6_score_value(result.get('full_sigma_g2'))}, "
                    f"scale=1/sigma_g2={_format_max6_score_value(result.get('max6_score_scale'))}, "
                    "source=inverse_sigma_g2; "
                    f"scope={'conditional_interaction' if args.grm_int else 'exact_channels'}; "
                    "screen=raw_Max6_proposal_score"
                ),
            )

        pseudo_path = f"{trait_logic_prefix}.pseudo"
        posterior_tsv_path = f"{trait_logic_prefix}.posterior.tsv"
        posterior_json_path = result.get("posterior_json")
        run_config_path = f"{trait_outprefix}.garfield.run_config.json"
        skipped_units = result.get("skipped_units") or []
        skipped_messages = result.get("skipped_messages") or []
        if len(skipped_units) > 0:
            _emit_garfield_file_only_warning(
                logger,
                f"GARFIELD skipped {len(skipped_units)} unit(s) for trait '{trait_name}' because no valid initial literals remained."
            )
            _emit_garfield_file_only_line(
                logger,
                "Skipped scan units (unit_name -> max_singleton_dosage_maf):",
            )
            for item in skipped_units:
                if not isinstance(item, dict):
                    continue
                unit_name = str(item.get("unit_name", "NA"))
                max_lmaf = item.get("max_singleton_dosage_maf")
                try:
                    maf_txt = f"{float(max_lmaf):.4f}"
                except Exception:
                    maf_txt = "NA"
                _emit_garfield_file_only_line(logger, f"  {unit_name} -> {maf_txt}")
        elif len(skipped_messages) > 0:
            _emit_garfield_file_only_warning(
                logger,
                f"GARFIELD skipped {len(skipped_messages)} unit(s) for trait '{trait_name}'."
            )
            for msg in skipped_messages:
                _emit_garfield_file_only_line(logger, str(msg))
        n_rules = int(result.get("n_rules", 0))
        # The Max6 -> GRM-Max6 backend emits pair results rather than legacy
        # beam-rule records.  Keep the existing control flow alive for log
        # and manifest generation without pretending that pair rows are beam
        # rules.
        if n_rules <= 0 and result.get("max6_pairs_tsv"):
            n_rules = int(result.get("max6_pair_count", 0))
        if n_rules <= 0:
            _remove_file_if_exists(pseudo_path)
            _remove_file_if_exists(posterior_tsv_path)
            _remove_file_if_exists(run_config_path)
            _remove_file_if_exists(posterior_json_path)
            sampled_maxima_path = result.get("max6_screen_sampled_maxima_tsv")
            if (
                (bool(args.sampled_screen_null_diagnostic) and sampled_maxima_path)
                or bool(args.max6_screen_tail_audit)
                or bool(args.max6_screen_exceedance_audit)
                or bool(args.max6_screen_pot_audit)
            ):
                sampled_paths = [
                    sampled_maxima_path,
                    result.get("max6_screen_sample_manifest_json"),
                    result.get("max6_screen_sample_strata_tsv"),
                    result.get("max6_screen_full_maximizer_geometry_tsv"),
                    result.get("max6_screen_tail_top_pairs_tsv"),
                    result.get("max6_screen_tail_sampled_ranks_tsv"),
                    result.get("max6_screen_tail_summary_json"),
                    result.get("max6_screen_tail_support_aware_maxima_tsv"),
                    result.get("max6_screen_tail_support_aware_strata_tsv"),
                    result.get("max6_screen_tail_support_aware_summary_json"),
                    result.get("max6_screen_exceedance_audit_per_null_tsv"),
                    result.get("max6_screen_exceedance_audit_summary_tsv"),
                    result.get("max6_screen_exceedance_audit_top_pairs_tsv"),
                    result.get("max6_screen_exceedance_audit_manifest_json"),
                    result.get("max6_screen_pot_audit_manifest_json"),
                    result.get("max6_screen_pot_sample_scores_manifest_json"),
                    result.get("max6_screen_maxima_tsv"),
                    result.get("max6_screen_validation_maxima_tsv"),
                ]
                _emit_garfield_saved_paths(
                    logger,
                    sampled_paths,
                    use_spinner=use_spinner,
                )
                _emit_garfield_file_only_line(
                    logger,
                    "No observed pair passed the unchanged production screen threshold; "
                    "the opt-in null audit completed independently.",
                )
                summary_rows.append(
                    {
                        "trait": trait_name,
                        "n_samples": len(common_ids),
                        "best_score": float("nan"),
                        "route": "rust-bed-sampled-diagnostic",
                    }
                )
                saved += 1
                continue
            logger.warning(f"No GARFIELD rules survived Rust search for trait '{trait_name}', skipped.")
            continue

        _remove_file_if_exists(pseudo_path)
        _remove_file_if_exists(posterior_tsv_path)
        prior_payload, posterior_payload = _split_structure_prior_payload(
            _load_json_if_exists(posterior_json_path)
        )
        pseudo_gwas_payload = None
        followup_memory_debug = None
        pseudo_prefix = result.get("pseudo_prefix")
        rules_tsv = result.get("rules_tsv")
        bonferroni_n_tests = (
            int(args.meff)
            if args.meff is not None
            else max(1, int(_n_snps))
        )
        if pseudo_prefix:
            def _run_followup() -> dict[str, object]:
                nonlocal followup_grm_full
                if followup_grm_full is None:
                    followup_grm_full = _ensure_followup_grm(
                        existing_grm=None,
                        genofile=gfile,
                        sample_ids=sample_ids,
                        n_snps=_n_snps,
                        maf_threshold=float(args.maf),
                        max_missing_rate=float(args.geno),
                        het_threshold=float(args.het),
                        threads=int(args.thread),
                        cache_dir=args.out,
                        logger=logger,
                        use_spinner=use_spinner,
                    )
                common_positions = np.asarray(
                    [(grm_index_map if aligned_grm is not None else sample_index_map)[sid]
                     for sid in common_ids],
                    dtype=np.intp,
                )
                trait_grm_followup = np.asarray(
                    followup_grm_full[np.ix_(common_positions, common_positions)],
                    dtype=np.float64,
                    order="C",
                )
                logger.info(
                    f"Running pseudo FvLMM follow-up for '{trait_name}' on {int(n_rules)} GARFIELD rule(s)."
                )
                return _run_garfield_pseudo_fvlmm(
                    pseudo_prefix=str(pseudo_prefix),
                    trait_label=suffix,
                    pheno_values=np.asarray(y_common, dtype=np.float64),
                    sample_ids=list(common_ids),
                    trait_grm=trait_grm_followup,
                    trait_cov=trait_cov,
                    logic_maf_threshold=float(logic_maf_threshold),
                    max_missing_rate=float(args.geno),
                    het_threshold=float(args.het),
                    threads=int(args.thread),
                    logger=logger,
                    use_spinner=use_spinner,
                    bonferroni_n_tests=bonferroni_n_tests,
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
            "min_cell": int(args.min_cell_runtime),
            "min_cell_fraction": float(args.min_cell_frac),
            "min_cell_effective": result.get("min_cell_effective"),
            "min_cell_neff": result.get("min_cell_neff"),
            "evaluation_mode": str(result.get("evaluation_mode", "full_exploratory")),
            "scan_mode": args.scan_mode,
            "unit_kind": logic_unit_kind,
            "response": response_mode,
            "engine": args.engine,
            "engine_runtime": engine_runtime,
            "ml_skipped": ml_skipped,
            # Keep the historical flag for downstream readers.
            "permutation": (
                bool(result.get("rule_permutation_active", False))
                and not _is_ld_tag_max6_backend(args.search_backend)
                and not args.grm_int
            ),
            "permutation_scope": (
                None
                if _is_ld_tag_max6_backend(args.search_backend) or args.grm_int
                else "global_pair_score_threshold"
                if _is_max6_grm_backend(args.search_backend)
                else "rule_null_penalty"
            ),
            "null_calibration_scope": (
                "screen_maxT_weak_global_null"
                if _is_ld_tag_max6_backend(args.search_backend)
                and args.rule_permutation_runtime
                else None
            ),
            "null_generation": (
                (
                    "lmm_parametric_bootstrap_max6_screen"
                    if args.rule_permutation_runtime
                    else None
                )
                if _is_ld_tag_max6_backend(args.search_backend)
                else None
                if _is_max6_grm_backend(args.search_backend) and args.grm_int
                else "global_pair_score_permutation_fixed_pc"
                if _is_max6_grm_backend(args.search_backend)
                else str(result.get("rule_null_generation", "lmm_parametric_bootstrap"))
            ),
            "operational_fwer_scope": (
                "weak_global_null_screen_subset"
                if _is_ld_tag_max6_backend(args.search_backend)
                and args.rule_permutation_runtime
                else None
            ),
            "residual_method": str(result.get("residualization_method", args.residual_method)),
            "lambda_source": str(result.get("lambda_source", "NA")),
            "no_grm_boundary": bool(result.get("no_grm_boundary", False)),
            "lambda_boundary_policy": str(
                result.get("lambda_boundary_policy", "finite_reml_lambda")
            ),
            "lambda_vc": float(result.get("lambda_vc", float("nan"))),
            "lambda_marker": float(result.get("lambda_marker", float("nan"))),
            "residual_marker_count": int(result.get("residual_marker_count", 0)),
            "pcg_converged": bool(result.get("pcg_converged", True)),
            "pcg_iters": int(result.get("pcg_iters", 0)),
            "pcg_rel_res": float(result.get("pcg_rel_res", 0.0)),
            "lambda_converged": bool(result.get("lambda_converged", True)),
            "lambda_iters": int(result.get("lambda_iters", 0)),
            "lambda_rel_res": float(result.get("lambda_rel_res", 0.0)),
            "rule_null_penalty_method": (
                None if _is_ld_tag_max6_backend(args.search_backend)
                and not args.rule_permutation_runtime
                else str(args.rule_null_penalty_method_runtime)
            ),
            "rule_null_quantile": (
                None if _is_ld_tag_max6_backend(args.search_backend)
                and not args.rule_permutation_runtime
                else float(args.rule_null_quantile_runtime)
            ),
            "rule_null_report_pvalue": bool(args.rule_null_report_pvalue_runtime),
            "rule_null_quantile_spec": args.rule_null_quantile_spec_runtime,
            "bg_noise_summary": result.get("bg_noise_summary"),
            "scheduler_scan": result.get("scheduler_scan"),
            "scheduler_permutation": result.get("scheduler_permutation"),
            "rule_permutation_active": bool(result.get("rule_permutation_active", False)),
            "null_calibration_active": bool(
                result.get("max6_screen_null_repeats", 0)
                or result.get("rule_permutation_active", False)
            ),
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
            "search_backend": str(args.search_backend),
            "grm_int_mode_source": str(args.grm_int_mode_source),
            **(
                {
                    "grm_int_only": True,
                    "grm_int_test": "one_df_FWL_g1xg2_adjusted_for_g1_g2_covariates_GRM",
                    "grm_int_pvalue_calibration": "nominal_chi2_1_after_Max6_selection_unadjusted",
                    "pwald_bonferroni_reference": grm_int_pwald_reference,
                    "pwald_bonferroni_reference_applied": False,
                    "pwald_bonferroni_reference_method": (
                        "alpha_over_full_unique_pair_universe_one_df_chi2"
                    ),
                    "grm_int_screen_source": "Max6 proposal score; exact requested Top-K per scan unit",
                    "grm_int_best_channel_source": "winning gate from the Max6 screen",
                    "grm_int_output_order": "conditional GRM-INT chi-square(1) within selected Top-K",
                    "grm_int_screen_threshold_applied": False,
                }
                if args.grm_int
                else {}
            ),
            "max6_pairs_tsv": result.get("max6_pairs_tsv"),
            "max6_pairs_provenance_tsv": result.get("max6_pairs_provenance_tsv"),
            "max6_ld_tags_tsv": result.get("max6_ld_tags_tsv"),
            "max6_ld_tag_r2": result.get("max6_ld_tag_r2"),
            "max6_ld_tag_mode": result.get("max6_ld_tag_mode"),
            "max6_pair_universe": (
                "ld_tag_pairs"
                if _is_ld_tag_max6_backend(args.search_backend)
                else "legacy_ml_prefilter_pairs"
            ),
            "max6_threshold_tsv": result.get("max6_threshold_tsv"),
            "max6_screen_maxima_tsv": result.get("max6_screen_maxima_tsv"),
            "max6_screen_validation_maxima_tsv": result.get(
                "max6_screen_validation_maxima_tsv"
            ),
            "max6_screen_exceedance_audit": bool(args.max6_screen_exceedance_audit),
            "max6_screen_exceedance_audit_per_null_tsv": result.get(
                "max6_screen_exceedance_audit_per_null_tsv"
            ),
            "max6_screen_exceedance_audit_summary_tsv": result.get(
                "max6_screen_exceedance_audit_summary_tsv"
            ),
            "max6_screen_exceedance_audit_top_pairs_tsv": result.get(
                "max6_screen_exceedance_audit_top_pairs_tsv"
            ),
            "max6_screen_exceedance_audit_manifest_json": result.get(
                "max6_screen_exceedance_audit_manifest_json"
            ),
            "max6_screen_pot_audit": bool(args.max6_screen_pot_audit),
            "max6_screen_pot_audit_manifest_json": result.get(
                "max6_screen_pot_audit_manifest_json"
            ),
            "max6_screen_pot_sample_scores_manifest_json": result.get(
                "max6_screen_pot_sample_scores_manifest_json"
            ),
            "timing_max6_screen_pot_audit_wall_s": float(
                result.get("timing_max6_screen_pot_audit_wall_s", 0.0)
            ),
            "timing_max6_screen_exceedance_audit_wall_s": float(
                result.get("timing_max6_screen_exceedance_audit_wall_s", 0.0)
            ),
            "max6_permutation_maxima_tsv": result.get("max6_permutation_maxima_tsv"),
            "max6_pair_pve_total_quadratic": result.get("max6_pair_pve_total_quadratic"),
            "max6_pair_pve_definition": result.get("max6_pair_pve_definition"),
            "max6_pair_universe_size": int(result.get("max6_pair_universe_size", 0)),
            **(
                {
                    "max6_bonferroni_alpha": result.get("max6_bonferroni_alpha"),
                    "max6_bonferroni_channel_count": result.get(
                        "max6_bonferroni_channel_count"
                    ),
                    "max6_bonferroni_p_cut": result.get("max6_bonferroni_p_cut"),
                }
                if not args.grm_int
                else {}
            ),
            "max6_pvalue_method": result.get("max6_pvalue_method"),
            "max6_score_threshold": result.get("max6_score_threshold"),
            "max6_gumbel_score_threshold": result.get("max6_gumbel_score_threshold"),
            "max6_screen_threshold": result.get("max6_screen_threshold"),
            "max6_screen_threshold_method": result.get("max6_screen_threshold_method"),
            "max6_screen_null_repeats": int(result.get("max6_screen_null_repeats", 0)),
            "max6_screen_validation_method": result.get("max6_screen_validation_method"),
            "max6_screen_validation_seed_policy": result.get(
                "max6_screen_validation_seed_policy"
            ),
            "max6_screen_validation_repeats": int(
                result.get("max6_screen_validation_repeats", 0)
            ),
            "max6_screen_validation_rejections": int(
                result.get("max6_screen_validation_rejections", 0)
            ),
            "max6_screen_validation_fwer": result.get("max6_screen_validation_fwer"),
            "max6_screen_validation_ci_lower": result.get(
                "max6_screen_validation_ci_lower"
            ),
            "max6_screen_validation_ci_upper": result.get(
                "max6_screen_validation_ci_upper"
            ),
            "max6_final_screen_subset_audit_passed": result.get(
                "max6_final_screen_subset_audit_passed"
            ),
            "max6_final_screen_subset_audit_pairs_checked": int(
                result.get("max6_final_screen_subset_audit_pairs_checked", 0)
            ),
            "max6_final_screen_subset_audit_violations": int(
                result.get("max6_final_screen_subset_audit_violations", 0)
            ),
            "max6_screen_pair_universe_size": int(
                result.get("max6_screen_pair_universe_size", 0)
            ),
            "max6_screen_window_pair_occurrences": int(
                result.get("max6_screen_window_pair_occurrences", 0)
            ),
            "timing_max6_screen_calibration_wall_s": float(
                result.get("timing_max6_screen_calibration_wall_s", 0.0)
            ),
            "timing_max6_screen_validation_wall_s": float(
                result.get("timing_max6_screen_validation_wall_s", 0.0)
            ),
            "max6_screen_candidate_count": int(
                result.get("max6_screen_candidate_count", 0)
            ),
            "max6_score_scale": result.get("max6_score_scale"),
            "max6_score_scale_source": result.get("max6_score_scale_source"),
            "max6_threshold_method": result.get("max6_threshold_method"),
            "max6_permutation_pair_pool_size": int(
                result.get("max6_permutation_pair_pool_size", 0)
            ),
            "max6_permutation_pairs_per_repeat": int(
                result.get("max6_permutation_pairs_per_repeat", 0)
            ),
            "max6_permutation_pair_draw_count": int(
                result.get("max6_permutation_pair_draw_count", 0)
            ),
            "timing_max6_permutation_setup_wall_s": float(
                result.get("timing_max6_permutation_setup_wall_s", 0.0)
            ),
            "timing_max6_permutation_score_wall_s": float(
                result.get("timing_max6_permutation_score_wall_s", 0.0)
            ),
            "timing_max6_permutation_threshold_wall_s": float(
                result.get("timing_max6_permutation_threshold_wall_s", 0.0)
            ),
            "timing_max6_permutation_maxima_write_wall_s": float(
                result.get("timing_max6_permutation_maxima_write_wall_s", 0.0)
            ),
            "timing_max6_permutation_projected_y_task_sum_s": float(
                result.get("timing_max6_permutation_projected_y_task_sum_s", 0.0)
            ),
            "timing_max6_permutation_endpoint_prepare_task_sum_s": float(
                result.get("timing_max6_permutation_endpoint_prepare_task_sum_s", 0.0)
            ),
            "timing_max6_permutation_rerank_kernel_task_sum_s": float(
                result.get("timing_max6_permutation_rerank_kernel_task_sum_s", 0.0)
            ),
            "timing_max6_permutation_rerank_pcx_task_sum_s": float(
                result.get("timing_max6_permutation_rerank_pcx_task_sum_s", 0.0)
            ),
            "timing_max6_permutation_rerank_geometry_task_sum_s": float(
                result.get("timing_max6_permutation_rerank_geometry_task_sum_s", 0.0)
            ),
            "timing_max6_permutation_rerank_q_task_sum_s": float(
                result.get("timing_max6_permutation_rerank_q_task_sum_s", 0.0)
            ),
            "timing_max6_permutation_rerank_score_task_sum_s": float(
                result.get("timing_max6_permutation_rerank_score_task_sum_s", 0.0)
            ),
            "timing_max6_permutation_rerank_topk_task_sum_s": float(
                result.get("timing_max6_permutation_rerank_topk_task_sum_s", 0.0)
            ),
            "timing_max6_permutation_rerank_other_task_sum_s": float(
                result.get("timing_max6_permutation_rerank_other_task_sum_s", 0.0)
            ),
            "timing_max6_permutation_pair_score_count": int(
                result.get("timing_max6_permutation_pair_score_count", 0)
            ),
            "max6_observed_units_timed": int(
                result.get("max6_observed_units_timed", 0)
            ),
            "max6_observed_screen_candidate_pairs": int(
                result.get("max6_observed_screen_candidate_pairs", 0)
            ),
            "max6_observed_screen_pairs_evaluated": int(
                result.get("max6_observed_screen_pairs_evaluated", 0)
            ),
            "max6_observed_screen_batch_count": int(
                result.get("max6_observed_screen_batch_count", 0)
            ),
            "max6_observed_screen_batch_unique_scored_pairs": int(
                result.get("max6_observed_screen_batch_unique_scored_pairs", 0)
            ),
            "max6_observed_screen_batch_score_offers": int(
                result.get("max6_observed_screen_batch_score_offers", 0)
            ),
            "max6_observed_screen_batch_direct_fallbacks": int(
                result.get("max6_observed_screen_batch_direct_fallbacks", 0)
            ),
            "max6_observed_exact_candidate_pairs": int(
                result.get("max6_observed_exact_candidate_pairs", 0)
            ),
            "max6_candidate_reuse_diagnostic": result.get(
                "max6_candidate_reuse_diagnostic"
            ),
            "timing_max6_observed_screen_input_prepare_task_sum_s": float(
                result.get("timing_max6_observed_screen_input_prepare_task_sum_s", 0.0)
            ),
            "timing_max6_observed_screen_kernel_task_sum_s": float(
                result.get("timing_max6_observed_screen_kernel_task_sum_s", 0.0)
            ),
            "timing_max6_observed_candidate_prepare_task_sum_s": float(
                result.get("timing_max6_observed_candidate_prepare_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_endpoint_prepare_task_sum_s": float(
                result.get("timing_max6_observed_rerank_endpoint_prepare_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_kernel_task_sum_s": float(
                result.get("timing_max6_observed_rerank_kernel_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_pcx_task_sum_s": float(
                result.get("timing_max6_observed_rerank_pcx_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_geometry_task_sum_s": float(
                result.get("timing_max6_observed_rerank_geometry_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_geometry_marker_moments_task_sum_s": float(
                result.get(
                    "timing_max6_observed_rerank_geometry_marker_moments_task_sum_s", 0.0
                )
            ),
            "timing_max6_observed_rerank_geometry_pairwise_task_sum_s": float(
                result.get("timing_max6_observed_rerank_geometry_pairwise_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_geometry_lookup_task_sum_s": float(
                result.get("timing_max6_observed_rerank_geometry_lookup_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_geometry_interaction_py_task_sum_s": float(
                result.get(
                    "timing_max6_observed_rerank_geometry_interaction_py_task_sum_s", 0.0
                )
            ),
            "timing_max6_observed_au_fused_py_task_sum_s": float(
                result.get("timing_max6_observed_au_fused_py_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_geometry_unclassified_task_sum_s": float(
                result.get(
                    "timing_max6_observed_rerank_geometry_unclassified_task_sum_s", 0.0
                )
            ),
            "timing_max6_observed_rerank_q_task_sum_s": float(
                result.get("timing_max6_observed_rerank_q_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_q_validation_task_sum_s": float(
                result.get("timing_max6_observed_rerank_q_validation_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_q_bitset_task_sum_s": float(
                result.get("timing_max6_observed_rerank_q_bitset_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_q_cell_task_sum_s": float(
                result.get("timing_max6_observed_rerank_q_cell_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_q_enumerate_task_sum_s": float(
                result.get("timing_max6_observed_rerank_q_enumerate_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_q_quadratic_task_sum_s": float(
                result.get("timing_max6_observed_rerank_q_quadratic_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_q_reconstruct_task_sum_s": float(
                result.get("timing_max6_observed_rerank_q_reconstruct_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_score_task_sum_s": float(
                result.get("timing_max6_observed_rerank_score_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_topk_task_sum_s": float(
                result.get("timing_max6_observed_rerank_topk_task_sum_s", 0.0)
            ),
            "timing_max6_observed_rerank_other_task_sum_s": float(
                result.get("timing_max6_observed_rerank_other_task_sum_s", 0.0)
            ),
            "timing_max6_observed_output_finalize_task_sum_s": float(
                result.get("timing_max6_observed_output_finalize_task_sum_s", 0.0)
            ),
            "max6_pair_count": int(result.get("max6_pair_count", 0)),
            "beam_width": int(args.beam_width),
            "not_control": (
                "score_threshold_only"
                if _is_max6_grm_backend(args.search_backend)
                else "null_penalty_only"
            ),
            "rank_schedule_source": rank_schedule_source,
            "gain_start_layer_runtime": gain_start_layer_runtime,
            "rank_schedule_runtime": rank_schedule_runtime,
            "rank_score": rank_score_runtime,
            "rank_score_runtime": rank_score_runtime,
            "xor_substate_lmaf_filter": not bool(args.disable_xor_substate_filter),
            "xor_search_requested": bool(args.xor_search_requested),
            "xor_search_enabled": bool(args.xor_search),
            "ml_top_k": (None if ml_skipped else int(args.ml_top_k_runtime)),
            "extension": int(args.extension),
            "step": int(args.step),
            "bimrange": (list(args.bimrange) if args.bimrange else None),
            "meff": (None if args.meff is None else int(args.meff)),
            "ranking_dataset": "full",
            "feature_source": "bin",
            "gene_files": list(args.genefiles),
            "gff3": gff3_effective,
            "grm_path": args.grm,
            "grm_id_path": resolved_grm_id,
            "maf": float(args.maf),
            **({
                "logic_maf": float(logic_maf_threshold),
                "logic_maf_source": logic_maf_source,
            } if not _is_max6_grm_backend(args.search_backend) else {}),
            "geno": float(args.geno),
            "het": float(args.het),
            "logic_memory_mb": float(args.logic_memory_mb),
            "logic_bits_backend": str(result.get("logic_bits_backend", "resident")),
            "logic_bits_bytes": int(result.get("logic_bits_bytes", 0)),
            "ld_clump_r2": float(result.get("ld_clump_r2", 0.8)),
            "ld_clump_rows_compressed": int(result.get("ld_clump_rows_compressed", 0)),
            "ld_clump_units": int(result.get("ld_clump_units", 0)),
            "pure_line_missing_rule": "na_only",
            "pure_line_het_rule": "drop_if_het_rate_gt_threshold",
            "simbench_path": args.simbench,
            "simbench_rows": int(result.get("n_simbench", 0)),
            "diagnostics_enabled": bool(args.diagnostics),
            "seed": trait_seed,
            "n_samples": len(common_ids),
            "full_n": int(result.get("full_n", result.get("n_samples", 0))),
            "n_rules": int(n_rules),
            "units_total": int(result.get("units_total", 0)),
            "units_scanned": int(result.get("units_scanned", 0)),
            "full_pve": float(result.get("full_pve", float("nan"))),
            "full_sigma_g2": float(result.get("full_sigma_g2", float("nan"))),
            "full_sigma_e2": float(result.get("full_sigma_e2", float("nan"))),
            "memory_debug": trait_memory_debug,
            "outputs": {
                "rules_tsv": result.get("rules_tsv"),
                "max6_pairs_tsv": result.get("max6_pairs_tsv"),
                "max6_pairs_provenance_tsv": result.get("max6_pairs_provenance_tsv"),
                "max6_threshold_tsv": result.get("max6_threshold_tsv"),
                "diagnostics_tsv": result.get("diagnostics_tsv"),
                "frontier_diagnostics_tsv": result.get("frontier_diagnostics_tsv"),
                "pseudo_fvlmm_raw_tsv": None,
                "pseudo_fvlmm_tsv": (
                    None
                    if pseudo_gwas_payload is None
                    else pseudo_gwas_payload.get("tsv_path")
                ),
                "pseudo_fvlmm_skipped_tsv": (
                    None
                    if pseudo_gwas_payload is None
                    else pseudo_gwas_payload.get("skipped_tsv_path")
                ),
                "pseudo_fvlmm_figure": (
                    None
                    if pseudo_gwas_payload is None
                    else pseudo_gwas_payload.get("figure_path")
                ),
                "pseudo_fvlmm2_raw_tsv": None,
                "pseudo_fvlmm2_tsv": None,
                "pseudo_fvlmm2_skipped_tsv": None,
                "pseudo_fvlmm2_figure": None,
                "pseudo_fastlmm_tsv": None,
                "pseudo_fastlmm_figure": None,
            },
            "prior": prior_payload,
            "posterior": posterior_payload,
            "pseudo_fvlmm": pseudo_gwas_payload,
            "pseudo_fvlmm2": None,
            "pseudo_fastlmm": None,
        }
        trait_manifest.update(_copy_prefixed_result_fields(result, ("timing_", "ld_")))
        garfield_manifest_traits.append(trait_manifest)
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
        trait_saved_paths: list[object] = [
            result.get("rules_tsv"),
            result.get("max6_pairs_tsv"),
            result.get("max6_pairs_provenance_tsv"),
            result.get("max6_screen_maxima_tsv"),
            result.get("max6_screen_validation_maxima_tsv"),
            result.get("max6_screen_sampled_maxima_tsv"),
            result.get("max6_screen_sample_manifest_json"),
            result.get("max6_screen_sample_strata_tsv"),
            result.get("max6_screen_full_maximizer_geometry_tsv"),
            result.get("max6_screen_tail_top_pairs_tsv"),
            result.get("max6_screen_tail_sampled_ranks_tsv"),
            result.get("max6_screen_tail_summary_json"),
            result.get("max6_screen_tail_support_aware_maxima_tsv"),
            result.get("max6_screen_tail_support_aware_strata_tsv"),
            result.get("max6_screen_tail_support_aware_summary_json"),
            result.get("max6_threshold_tsv"),
            result.get("diagnostics_tsv"),
            result.get("frontier_diagnostics_tsv"),
        ]
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
                "input_kind": "bfile",
                "execution_route": "direct Rust BED pipeline",
                "encoding": "bin",
                "phenotype_file": args.pheno,
                "gene_files": list(args.genefiles),
                "gff3": gff3_effective,
                "grm_path": args.grm,
                "grm_id_path": resolved_grm_id,
                "scan_mode": args.scan_mode,
                "rank_score_runtime": rank_score_runtime,
                "rank_schedule_runtime": rank_schedule_runtime,
                "rank_schedule_source": rank_schedule_source,
                "search_backend": str(args.search_backend),
                "grm_int_mode_source": str(args.grm_int_mode_source),
                **(
                    {
                        "grm_int_only": True,
                        "grm_int_test": "one_df_FWL_g1xg2_adjusted_for_g1_g2_covariates_GRM",
                        "grm_int_pvalue_calibration": "nominal_chi2_1_after_Max6_selection_unadjusted",
                        "grm_int_screen_source": "Max6 proposal score; exact requested Top-K per scan unit",
                        "grm_int_best_channel_source": "winning gate from the Max6 screen",
                        "grm_int_output_order": "conditional GRM-INT chi-square(1) within selected Top-K",
                        "grm_int_screen_threshold_applied": False,
                    }
                    if args.grm_int
                    else {}
                ),
                "max6_pairs_tsv": [
                    item.get("max6_pairs_tsv")
                    for item in garfield_manifest_traits
                    if isinstance(item, dict) and item.get("max6_pairs_tsv")
                ],
                "max6_pairs_provenance_tsv": [
                    item.get("max6_pairs_provenance_tsv")
                    for item in garfield_manifest_traits
                    if isinstance(item, dict) and item.get("max6_pairs_provenance_tsv")
                ],
                "max6_ld_tags_tsv": [
                    item.get("max6_ld_tags_tsv")
                    for item in garfield_manifest_traits
                    if isinstance(item, dict) and item.get("max6_ld_tags_tsv")
                ],
                "max6_threshold_tsv": [
                    item.get("max6_threshold_tsv")
                    for item in garfield_manifest_traits
                    if isinstance(item, dict) and item.get("max6_threshold_tsv")
                ],
                "max6_screen_maxima_tsv": [
                    item.get("max6_screen_maxima_tsv")
                    for item in garfield_manifest_traits
                    if isinstance(item, dict) and item.get("max6_screen_maxima_tsv")
                ],
                "max6_screen_validation_maxima_tsv": [
                    item.get("max6_screen_validation_maxima_tsv")
                    for item in garfield_manifest_traits
                    if isinstance(item, dict) and item.get("max6_screen_validation_maxima_tsv")
                ],
                "max6_screen_validation_fwer": [
                    item.get("max6_screen_validation_fwer")
                    for item in garfield_manifest_traits
                    if isinstance(item, dict)
                    and item.get("max6_screen_validation_fwer") is not None
                ],
                "max6_screen_validation": [
                    {
                        "trait": item.get("trait"),
                        "scope": item.get("operational_fwer_scope"),
                        "method": item.get("max6_screen_validation_method"),
                        "threshold": item.get("max6_screen_threshold"),
                        "threshold_method": item.get("max6_screen_threshold_method"),
                        "pair_universe_size": item.get("max6_screen_pair_universe_size"),
                        "window_pair_occurrences": item.get(
                            "max6_screen_window_pair_occurrences"
                        ),
                        "calibration_repeats": item.get("max6_screen_null_repeats"),
                        "validation_repeats": item.get("max6_screen_validation_repeats"),
                        "validation_rejections": item.get("max6_screen_validation_rejections"),
                        "validation_fwer": item.get("max6_screen_validation_fwer"),
                        "validation_ci95": [
                            item.get("max6_screen_validation_ci_lower"),
                            item.get("max6_screen_validation_ci_upper"),
                        ],
                        "final_subset_audit": {
                            "passed": item.get("max6_final_screen_subset_audit_passed"),
                            "pairs_checked": item.get(
                                "max6_final_screen_subset_audit_pairs_checked", 0
                            ),
                            "violations": item.get(
                                "max6_final_screen_subset_audit_violations", 0
                            ),
                        },
                    }
                    for item in garfield_manifest_traits
                    if isinstance(item, dict)
                    and item.get("max6_screen_validation_repeats", 0) > 0
                ],
                "max6_final_screen_subset_audit": [
                    {
                        "passed": item.get("max6_final_screen_subset_audit_passed"),
                        "pairs_checked": item.get(
                            "max6_final_screen_subset_audit_pairs_checked", 0
                        ),
                        "violations": item.get(
                            "max6_final_screen_subset_audit_violations", 0
                        ),
                    }
                    for item in garfield_manifest_traits
                    if isinstance(item, dict)
                    and item.get("max6_final_screen_subset_audit_passed") is not None
                ],
                "max6_pair_pve_definition": (
                    "scaled_winner_score / scaled_y_quadratic; GRM-metric partial-R2"
                    if any(
                        isinstance(item, dict) and item.get("max6_pair_pve_total_quadratic") is not None
                        for item in garfield_manifest_traits
                    )
                    else None
                ),
                "max6_pair_pve_total_quadratic": [
                    item.get("max6_pair_pve_total_quadratic")
                    for item in garfield_manifest_traits
                    if isinstance(item, dict) and item.get("max6_pair_pve_total_quadratic") is not None
                ],
                "max6_bonferroni_thresholds": (
                    []
                    if args.grm_int
                    else [
                        {
                            "trait": item.get("trait"),
                            "N_unique": item.get("max6_pair_universe_size"),
                            "alpha": item.get("max6_bonferroni_alpha"),
                            "m": item.get("max6_bonferroni_channel_count"),
                            "p_cut": item.get("max6_bonferroni_p_cut"),
                        }
                        for item in garfield_manifest_traits
                        if isinstance(item, dict)
                        and item.get("max6_bonferroni_p_cut") is not None
                    ]
                ),
                "pwald_bonferroni_thresholds": [
                    {
                        "trait": item.get("trait"),
                        "N_unique_pairs": item["pwald_bonferroni_reference"][
                            "n_unique_pairs"
                        ],
                        "alpha": item["pwald_bonferroni_reference"]["alpha"],
                        "p_cut": item["pwald_bonferroni_reference"]["p_cut"],
                        "minus_log10_p_cut": item["pwald_bonferroni_reference"][
                            "minus_log10_p_cut"
                        ],
                        "method": item.get("pwald_bonferroni_reference_method"),
                        "applied": item.get("pwald_bonferroni_reference_applied"),
                    }
                    for item in garfield_manifest_traits
                    if isinstance(item, dict)
                    and isinstance(item.get("pwald_bonferroni_reference"), dict)
                ],
                "xor_search_requested": bool(args.xor_search_requested),
                "xor_search_enabled": bool(args.xor_search),
                "evaluation_mode": "full_exploratory",
                "permutation": (
                    not _is_ld_tag_max6_backend(args.search_backend) and not args.grm_int
                ),
                "permutation_scope": (
                    None
                    if _is_ld_tag_max6_backend(args.search_backend) or args.grm_int
                    else "global_pair_score_threshold"
                    if _is_max6_grm_backend(args.search_backend)
                    else "rule_null_penalty"
                ),
                "null_calibration_scope": (
                    "screen_maxT_weak_global_null"
                    if _is_ld_tag_max6_backend(args.search_backend)
                    and args.rule_permutation_runtime
                    else None
                ),
                "null_generation": (
                    (
                        "lmm_parametric_bootstrap_max6_screen"
                        if args.rule_permutation_runtime
                        else None
                )
                if _is_ld_tag_max6_backend(args.search_backend)
                else None
                if _is_max6_grm_backend(args.search_backend) and args.grm_int
                else "global_pair_score_permutation_fixed_pc"
                if _is_max6_grm_backend(args.search_backend)
                    else "lmm_parametric_bootstrap"
                ),
                "operational_fwer_scope": (
                    "weak_global_null_screen_subset"
                    if _is_ld_tag_max6_backend(args.search_backend)
                    and args.rule_permutation_runtime
                    else None
                ),
                "rule_null_penalty_method": (
                    None if _is_ld_tag_max6_backend(args.search_backend)
                    and not args.rule_permutation_runtime
                    else str(args.rule_null_penalty_method_runtime)
                ),
                "rule_null_quantile": (
                    None if _is_ld_tag_max6_backend(args.search_backend)
                    and not args.rule_permutation_runtime
                    else float(args.rule_null_quantile_runtime)
                ),
                "rule_null_report_pvalue": bool(args.rule_null_report_pvalue_runtime),
                "rule_null_quantile_spec": args.rule_null_quantile_spec_runtime,
                "null_chunk_bp": int(args.extension) * 2,
                "null_chunk_target": 150,
                "null_chunk_min_snps": 50,
                "bimrange": (list(args.bimrange) if args.bimrange else None),
                "top_rules_per_unit": int(args.top_rules_runtime),
                "pair_seed_depth": int(args.exhaustive_depth_runtime),
                "not_control": (
                    "score_threshold_only"
                    if _is_max6_grm_backend(args.search_backend)
                    else "null_penalty_only"
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
