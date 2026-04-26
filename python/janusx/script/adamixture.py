# -*- coding: utf-8 -*-
"""
JanusX: ADAMIXTURE ancestry inference (Rust-kernel backend)

Examples
--------
  jx adamixture -bfile data/geno -k 8 -o out -prefix cohort
  jx adamixture -vcf data/geno.vcf.gz -k 6 -t 16
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from janusx.adamixture import ADAMixtureConfig, evaluate_adamixture_cverror, train_adamixture
from janusx.adamixture.core import load_genotype_u8_matrix
from janusx.gfreader import inspect_genotype_file
from ._common.config_render import emit_cli_configuration
from ._common.genoio import determine_genotype_source_from_args, strip_default_prefix_suffix
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    format_path_for_display,
    ensure_plink_prefix_exists,
)
from ._common.progress import build_rich_progress, rich_progress_available
from ._common.status import CliStatus, log_success, print_failure, print_success, stdout_is_tty
from ._common.threads import (
    apply_blas_thread_env,
    detect_effective_threads,
    maybe_warn_non_openblas,
    require_openblas_by_default,
)

_DEFAULT_CVERROR_FOLDS = 5
_ADMX_PROGRESS_BAR_WIDTH = 24


def _parse_cv_spec(cv_raw: Optional[str]) -> tuple[bool, int, str]:
    """
    Parse CV option:
      - None        -> disabled
      - ""/omitted  -> argparse const handles as default folds
      - "N>=2"      -> N-fold CVerror
      - "0"         -> disabled
    Returns: (enabled, cv_value_for_cfg, label)
    """
    if cv_raw is None:
        return False, 0, "off"
    raw = str(cv_raw).strip().lower()
    if raw in {"", "none"}:
        return False, 0, "off"
    try:
        n = int(raw)
    except Exception as ex:
        raise ValueError(
            f"Invalid -cv/--cv value: {cv_raw!r}. Use no arg or integer folds."
        ) from ex
    if n < 0:
        raise ValueError(f"-cv/--cv must be >= 0, got {n}.")
    if n == 0:
        return False, 0, "off"
    if n < 2:
        raise ValueError(f"-cv/--cv must be >= 2 when enabled, got {n}.")
    return True, int(n), f"{int(n)}-fold"


def _bool_env(name: str, default: bool = True) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if raw == "":
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def _build_k_warm_start(
    prev_model: Optional[dict[str, Any]],
    *,
    target_k: int,
    seed: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if prev_model is None:
        return None
    prev_k = int(prev_model.get("k", 0))
    if int(target_k) <= int(prev_k):
        return None
    p_prev = np.ascontiguousarray(np.asarray(prev_model.get("p"), dtype=np.float32))
    q_prev = np.ascontiguousarray(np.asarray(prev_model.get("q"), dtype=np.float32))
    if p_prev.ndim != 2 or q_prev.ndim != 2:
        return None
    if p_prev.shape[1] != int(prev_k) or q_prev.shape[1] != int(prev_k):
        return None

    m = int(p_prev.shape[0])
    n = int(q_prev.shape[0])
    dk = int(target_k) - int(prev_k)
    if dk <= 0:
        return None

    rng = np.random.default_rng(int(seed) + int(target_k) * 7919)
    p_extra = rng.random(size=(m, dk), dtype=np.float32)
    p0 = np.ascontiguousarray(
        np.clip(np.concatenate([p_prev, p_extra], axis=1), 1e-5, 1.0 - 1e-5),
        dtype=np.float32,
    )

    mass_raw = str(os.environ.get("JANUSX_ADMX_K_WARM_QMASS", "0.02")).strip()
    try:
        new_mass = float(mass_raw)
    except Exception:
        new_mass = 0.02
    new_mass = min(0.20, max(1e-4, float(new_mass)))

    q_prev = np.clip(q_prev, 1e-8, None)
    s_prev = q_prev.sum(axis=1, keepdims=True)
    s_prev[s_prev <= 0] = 1.0
    q_prev = q_prev / s_prev

    q_extra = rng.random(size=(n, dk), dtype=np.float32)
    s_extra = q_extra.sum(axis=1, keepdims=True)
    s_extra[s_extra <= 0] = 1.0
    q_extra = q_extra / s_extra

    q_old = q_prev * float(1.0 - new_mass)
    q_new = q_extra * float(new_mass)
    q0 = np.ascontiguousarray(
        np.clip(np.concatenate([q_old, q_new], axis=1), 1e-8, None),
        dtype=np.float32,
    )
    return p0, q0


def _mute_stdout_info_logs(
    logger: logging.Logger,
) -> list[tuple[logging.Handler, int]]:
    muted: list[tuple[logging.Handler, int]] = []
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout:
            muted.append((handler, handler.level))
            handler.setLevel(logging.WARNING)
    return muted


def _restore_handler_levels(muted: list[tuple[logging.Handler, int]]) -> None:
    for handler, level in muted:
        handler.setLevel(level)


class _AdmixtureBatchProgress:
    """Two-line rich progress for multi-K mode."""

    def __init__(
        self,
        *,
        enabled: bool,
        total_k: int,
        cv_enabled: bool,
        cv_label: str,
    ) -> None:
        self.enabled = bool(enabled and rich_progress_available())
        self.total_k = int(max(1, int(total_k)))
        self.cv_enabled = bool(cv_enabled)
        self.cv_label = str(cv_label)
        self._progress = None
        self._task_k = None
        self._task_cv = None

    def __enter__(self) -> "_AdmixtureBatchProgress":
        if not self.enabled:
            return self
        self._progress = build_rich_progress(
            description_template="[green]{task.description}",
            show_spinner=False,
            show_bar=True,
            show_percentage=True,
            show_elapsed=True,
            show_remaining=False,
            bar_width=int(_ADMX_PROGRESS_BAR_WIDTH),
            transient=False,
        )
        if self._progress is None:
            self.enabled = False
            return self
        self._progress.__enter__()
        self._task_k = self._progress.add_task("K progress", total=self.total_k)
        self._task_cv = self._progress.add_task("CV progress", total=1)
        self._progress.update(self._task_cv, completed=0, total=1)
        return self

    def start_k(self, *, k: int) -> None:
        if not self.enabled or self._progress is None or self._task_cv is None:
            return
        if self.cv_enabled:
            self._progress.reset(
                self._task_cv,
                description=f"CV folds (K={int(k)})",
                total=1,
                completed=0,
            )
        else:
            self._progress.reset(
                self._task_cv,
                description=f"CV off (K={int(k)})",
                total=1,
                completed=1,
            )

    def cv_start(self, *, k: int, folds: int) -> None:
        if not self.enabled or self._progress is None or self._task_cv is None:
            return
        total = max(1, int(folds))
        self._progress.reset(
            self._task_cv,
            description=f"CV folds (K={int(k)})",
            total=total,
            completed=0,
        )

    def cv_advance(self, *, steps: int = 1) -> None:
        if not self.enabled or self._progress is None or self._task_cv is None:
            return
        self._progress.advance(self._task_cv, int(max(1, int(steps))))

    def cv_complete(self) -> None:
        if not self.enabled or self._progress is None or self._task_cv is None:
            return
        task = self._progress.tasks[self._task_cv]
        self._progress.update(self._task_cv, completed=task.total)

    def advance_k(self, *, steps: int = 1) -> None:
        if not self.enabled or self._progress is None or self._task_k is None:
            return
        self._progress.advance(self._task_k, int(max(1, int(steps))))

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._progress is not None:
            self._progress.__exit__(exc_type, exc, tb)
            self._progress = None


class _AdmixtureCliProgress:
    def __init__(self, logger: logging.Logger, *, enabled: bool) -> None:
        self.logger = logger
        self.enabled = bool(enabled)
        self._status: Optional[CliStatus] = None
        self._adam_max_iter = 0

    def _start(self, desc: str) -> None:
        self.close()
        self._status = CliStatus(str(desc), enabled=self.enabled)
        self._status.__enter__()

    def _update_desc(self, desc: str) -> None:
        if self._status is not None:
            self._status.desc = str(desc)

    def _complete(self, message: str) -> None:
        if self._status is not None:
            self._status.complete(str(message))
            self._status = None

    def fail(self, message: str) -> None:
        if self._status is not None:
            self._status.fail(str(message))
            self._status = None

    def close(self) -> None:
        if self._status is not None:
            self._status.__exit__(None, None, None)
            self._status = None

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        e = str(event)
        if e == "data_loaded":
            m = int(payload.get("m", 0))
            n = int(payload.get("n", 0))
            if self.enabled:
                print_success(
                    f"Loaded genotype matrix (SNPs={m}, samples={n})",
                    force_color=True,
                )
                print(f"Data shape: SNPs={m}, samples={n}", flush=True)
            return

        if e == "rsvd_start":
            self._start("RSVD")
            return
        if e == "rsvd_done":
            self._complete("RSVD ...Finished")
            return

        if e == "als_start":
            self._start("ALS initialization")
            return
        if e == "als_done":
            self._complete("ALS initialization ...Finished")
            return

        if e == "adam_start":
            self._adam_max_iter = int(payload.get("max_iter", 0))
            self._start(f"ADAM-EM iter 0/{self._adam_max_iter}")
            return
        if e == "adam_iter":
            it = int(payload.get("iteration", 0))
            ll = float(payload.get("ll", float("nan")))
            lr = float(payload.get("lr", float("nan")))
            if np.isfinite(ll) and np.isfinite(lr):
                self._update_desc(
                    f"ADAM-EM iter {it}/{self._adam_max_iter} "
                    f"ll={ll:.3f} lr={lr:.3e}"
                )
            return
        if e == "adam_done":
            self._complete("ADAM-EM optimization ...Finished")
            return


def _resolve_input(args, logger) -> tuple[str, str, str]:
    gfile, auto_prefix = determine_genotype_source_from_args(args)

    checks = []
    if args.vcf:
        checks.append(ensure_file_exists(logger, gfile, "VCF file"))
        source_label = "VCF"
    elif args.hmp:
        checks.append(ensure_file_exists(logger, gfile, "HMP file"))
        source_label = "HMP"
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, label="FILE genotype input"))
        source_label = "FILE"
    elif args.bfile:
        bprefix = gfile
        if str(gfile).lower().endswith(".bed"):
            bprefix = str(Path(gfile).with_suffix(""))
        checks.append(ensure_plink_prefix_exists(logger, bprefix, label="PLINK prefix"))
        gfile = bprefix
        auto_prefix = os.path.basename(bprefix.rstrip("/\\"))
        source_label = "BFILE"
    else:
        raise ValueError("One genotype input is required: -vcf / -hmp / -file / -bfile.")

    if not ensure_all_true(checks):
        raise FileNotFoundError("Input validation failed.")

    prefix = args.prefix or strip_default_prefix_suffix(auto_prefix)
    return gfile, source_label, prefix


def _site_line_from_obj(site: Any) -> str:
    chrom = "0"
    pos = 0
    ref = "N"
    alt = "N"
    try:
        if hasattr(site, "chrom") and hasattr(site, "pos"):
            chrom = str(getattr(site, "chrom"))
            pos = int(getattr(site, "pos"))
            if hasattr(site, "ref_allele"):
                ref = str(getattr(site, "ref_allele"))
            if hasattr(site, "alt_allele"):
                alt = str(getattr(site, "alt_allele"))
            return f"{chrom}\t{pos}\t{ref}\t{alt}"
        if isinstance(site, (tuple, list)):
            if len(site) >= 4:
                chrom = str(site[0])
                pos = int(site[1])
                ref = str(site[2])
                alt = str(site[3])
                return f"{chrom}\t{pos}\t{ref}\t{alt}"
            if len(site) >= 2:
                chrom = str(site[0])
                pos = int(site[1])
                return f"{chrom}\t{pos}\t{ref}\t{alt}"
    except Exception:
        pass
    return f"{chrom}\t{pos}\t{ref}\t{alt}"


def _safe_sample_ids_and_site_lines(
    genotype_path: str,
    *,
    snps_only: bool,
    maf: float,
    missing_rate: float,
    expected_n: int,
    expected_m: int,
    logger: logging.Logger,
) -> tuple[list[str], list[str]]:
    sample_ids: list[str] = []
    site_lines: list[str] = []
    try:
        sample_ids_raw, site_rows_raw = inspect_genotype_file(
            genotype_path,
            snps_only=bool(snps_only),
            maf=float(maf),
            missing_rate=float(missing_rate),
        )
        sample_ids = [str(x) for x in sample_ids_raw]
        if isinstance(site_rows_raw, (list, tuple, np.ndarray)):
            site_lines = [_site_line_from_obj(x) for x in site_rows_raw]
        else:
            site_lines = []
    except Exception as ex:
        logger.warning(f"Failed to inspect sample/site IDs for Q/P output: {ex}")
        sample_ids = []
        site_lines = []

    if len(sample_ids) != int(expected_n):
        if len(sample_ids) > 0:
            logger.warning(
                "Sample ID count mismatch for Q output "
                f"(ids={len(sample_ids)}, expected={int(expected_n)}). "
                "Fallback to generated sample IDs."
            )
        sample_ids = [f"S{i+1}" for i in range(int(expected_n))]

    if len(site_lines) != int(expected_m):
        if len(site_lines) > 0:
            logger.warning(
                "Site row count mismatch for P.site output "
                f"(sites={len(site_lines)}, expected={int(expected_m)}). "
                "Fallback to generated site rows."
            )
        site_lines = [f"0\t{i+1}\tN\tN" for i in range(int(expected_m))]

    return sample_ids, site_lines


def _write_q_txt_with_row_ids(
    out_path: str,
    row_ids: list[str],
    mat: np.ndarray,
) -> None:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got shape={arr.shape}")
    if len(row_ids) != int(arr.shape[0]):
        raise ValueError(
            f"Row ID length mismatch: ids={len(row_ids)}, rows={int(arr.shape[0])}"
        )
    with open(out_path, "w", encoding="utf-8") as fw:
        for rid, row in zip(row_ids, arr):
            vals = "\t".join(f"{float(v):.8f}" for v in row)
            fw.write(f"{rid}\t{vals}\n")


def _resolve_p_npy_dtype() -> np.dtype:
    raw = str(os.environ.get("JANUSX_ADMX_P_DTYPE", "float32")).strip().lower()
    if raw in {"float32", "f4", "single"}:
        return np.dtype(np.float32)
    if raw in {"float64", "f8", "double"}:
        return np.dtype(np.float64)
    raise ValueError(
        "Invalid JANUSX_ADMX_P_DTYPE. Use float32 or float64."
    )


def _stable_unique(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for x in items:
        s = str(x)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _parse_tag_samples(
    tag_raw: Optional[str],
    *,
    logger: logging.Logger,
) -> list[str]:
    raw = ("" if tag_raw is None else str(tag_raw)).strip()
    if raw == "":
        return []

    p = Path(raw)
    if p.exists() and p.is_file():
        tags: list[str] = []
        with open(p, "r", encoding="utf-8") as fr:
            for line in fr:
                x = str(line).strip()
                if x == "":
                    continue
                tags.append(x)
        return _stable_unique(tags)

    parts = [x.strip() for x in raw.split(",") if x.strip() != ""]
    if len(parts) == 0:
        logger.warning(
            f"--tag specification is empty after parsing: {tag_raw!r}. "
            "No x-axis labels will be displayed."
        )
        return []
    return _stable_unique(parts)


def _render_structure_plot(
    *,
    q_mat: np.ndarray,
    q_ids: list[str],
    out_path: str,
    tag_samples: list[str],
    logger: logging.Logger,
) -> tuple[Optional[str], dict[str, Any]]:
    try:
        from janusx.bioplotkit.popstructure import plot_admixture_structure
    except Exception as ex:
        logger.warning(f"Failed to import popstructure plotting helper: {ex}")
        return None, {}

    try:
        meta = plot_admixture_structure(
            np.asarray(q_mat, dtype=np.float64),
            sample_ids=list(q_ids),
            out_path=str(out_path),
            tag_samples=list(tag_samples),
            show_xticks=False,
            dpi=300,
        )
        missing = list(meta.get("tag_missing", []))
        if len(missing) > 0:
            logger.warning(
                f"{len(missing)} tagged sample(s) were not found in Q sample IDs: "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )
        return str(out_path), meta
    except Exception as ex:
        logger.warning(f"Failed to render structure plot: {ex}")
        return None, {}


def _write_p_npy_and_site(
    out_npy_path: str,
    out_site_path: str,
    site_lines: list[str],
    mat: np.ndarray,
) -> None:
    arr = np.asarray(mat)
    if arr.ndim != 2:
        raise ValueError(f"P matrix must be 2D, got shape={arr.shape}")
    if len(site_lines) != int(arr.shape[0]):
        raise ValueError(
            f"P.site row count mismatch: sites={len(site_lines)}, rows={int(arr.shape[0])}"
        )

    with open(out_site_path, "w", encoding="utf-8") as fw:
        for line in site_lines:
            text = str(line).rstrip("\n")
            fw.write(f"{text}\n")

    dtype = _resolve_p_npy_dtype()
    chunk_rows_raw = str(os.environ.get("JANUSX_ADMX_P_MEMMAP_CHUNK", "4096")).strip()
    try:
        chunk_rows = max(1, int(chunk_rows_raw))
    except Exception:
        chunk_rows = 4096

    mm = np.lib.format.open_memmap(
        out_npy_path,
        mode="w+",
        dtype=dtype,
        shape=(int(arr.shape[0]), int(arr.shape[1])),
    )
    try:
        for start in range(0, int(arr.shape[0]), int(chunk_rows)):
            end = min(int(arr.shape[0]), int(start + int(chunk_rows)))
            mm[start:end, :] = np.asarray(arr[start:end, :], dtype=dtype)
    finally:
        del mm


def _parse_k_spec(k_spec: str) -> list[int]:
    """
    Parse K specification:
      - single value: "8"
      - inclusive range: "2..10" or "2:10"
      - inclusive range with step: "2..10..3", "2:10:3", or "2..10:3"
      - list: "2,5,8"
      - mixed: "2..10:3,5,8"
    """
    raw = str(k_spec or "").strip()
    if raw == "":
        raise ValueError("Empty -k/--k specification.")
    if ";" in raw:
        raise ValueError(
            "Semicolon-separated K list is not supported for shell compatibility. "
            "Use comma-separated form, e.g. -k 2,5,8."
        )

    tokens = []
    for seg in raw.split(","):
        x = seg.strip()
        if x:
            tokens.append(x)
    if len(tokens) == 0:
        raise ValueError(f"Invalid -k/--k specification: {k_spec!r}")

    values: list[int] = []
    for tok in tokens:
        if ".." in tok:
            norm = tok.replace("..", ":")
            parts = [x.strip() for x in norm.split(":") if x.strip() != ""]
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid K range token: {tok!r}")
            lo, hi = int(parts[0]), int(parts[1])
            step_abs = abs(int(parts[2])) if len(parts) == 3 else 1
            if step_abs <= 0:
                raise ValueError(f"K range step must be non-zero: {tok!r}")
            step = int(step_abs if hi >= lo else -step_abs)
            stop = int(hi + (1 if step > 0 else -1))
            values.extend(list(range(int(lo), stop, int(step))))
            continue
        if ":" in tok:
            parts = tok.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid K range token: {tok!r}")
            lo, hi = int(parts[0]), int(parts[1])
            step_abs = abs(int(parts[2])) if len(parts) == 3 else 1
            if step_abs <= 0:
                raise ValueError(f"K range step must be non-zero: {tok!r}")
            step = int(step_abs if hi >= lo else -step_abs)
            stop = int(hi + (1 if step > 0 else -1))
            values.extend(list(range(int(lo), stop, int(step))))
            continue
        values.append(int(tok))

    if len(values) == 0:
        raise ValueError(f"Invalid -k/--k specification: {k_spec!r}")

    deduped: list[int] = []
    seen = set()
    for v in values:
        if int(v) < 2:
            raise ValueError(f"K must be >= 2, got {v}.")
        if int(v) not in seen:
            deduped.append(int(v))
            seen.add(int(v))
    return deduped


def _write_cverror_summary_tsv(
    out_tsv: str,
    rows: list[dict[str, Any]],
) -> None:
    with open(out_tsv, "w", encoding="utf-8") as fw:
        fw.write(
            "\t".join(
                [
                    "K",
                    "CVerror",
                    "CVerror_SE",
                    "CV_folds",
                    "WallTime_sec",
                    "Q_output",
                    "P_npy_output",
                    "P_site_output",
                    "Structure_plot",
                    "Log_file",
                ]
            )
            + "\n"
        )
        for r in rows:
            fw.write(
                "\t".join(
                    [
                        str(int(r["k"])),
                        f"{float(r['cverror']):.6f}",
                        f"{float(r['cverror_se']):.6f}",
                        str(int(r["cv_folds"])),
                        f"{float(r['wall']):.3f}",
                        str(r["q_out"]),
                        str(r["p_npy_out"]),
                        str(r["p_site_out"]),
                        str(r.get("structure_plot", "")),
                        str(r["log_path"]),
                    ]
                )
                + "\n"
            )


def _log_cverror_summary(
    logger: logging.Logger,
    rows: list[dict[str, Any]],
    summary_tsv: str,
) -> None:
    if len(rows) == 0:
        return
    logger.info("-" * 60)
    row_fmt = "{k:<4} {cv:<10} {se:<10} {folds:<8} {time:<10}"
    logger.info(
        row_fmt.format(
            k="K",
            cv="CVerror",
            se="SE",
            folds="CVfolds",
            time="time(sec)",
        )
    )
    for r in rows:
        logger.info(
            row_fmt.format(
                k=int(r["k"]),
                cv=f"{float(r['cverror']):.6f}",
                se=f"{float(r['cverror_se']):.6f}",
                folds=int(r["cv_folds"]),
                time=f"{float(r['wall']):.2f}",
            )
        )
    best = min(rows, key=lambda x: float(x["cverror"]))
    logger.info("-" * 60)
    logger.info(
        f"Best K by CVerror: K={int(best['k'])} "
        f"(CVerror={float(best['cverror']):.6f}, SE={float(best['cverror_se']):.6f})"
    )
    logger.info(f"CVerror summary TSV: {format_path_for_display(summary_tsv)}")


def _log_outputs_summary(
    logger: logging.Logger,
    rows: list[dict[str, Any]],
) -> None:
    if len(rows) == 0:
        return
    logger.info("-" * 60)
    logger.info("Output files:")
    for r in rows:
        k = int(r["k"])
        logger.info(f"K={k:<4} Q: {format_path_for_display(str(r['q_out']))}")
        logger.info(f"K={k:<4} P.npy: {format_path_for_display(str(r['p_npy_out']))}")
        logger.info(f"K={k:<4} P.site: {format_path_for_display(str(r['p_site_out']))}")
        if r.get("structure_plot"):
            logger.info(
                f"K={k:<4} STRUCT: "
                f"{format_path_for_display(str(r['structure_plot']))}"
            )
        logger.info(f"K={k:<4} LOG: {format_path_for_display(str(r['log_path']))}")


def _run_single_k(
    *,
    genotype_path: str,
    source_label: str,
    prefix: str,
    outdir: str,
    k: int,
    k_spec: str,
    k_index: int,
    total_k: int,
    args: argparse.Namespace,
    detected_threads: int,
    resolved_threads: int,
    enable_spinner: bool,
    emit_config_to_stdout: bool,
    batch_progress: Optional[_AdmixtureBatchProgress],
    cv_enabled: bool,
    cv_value: int,
    cv_label: str,
    tag_samples: list[str],
    shared_g: Optional[np.ndarray],
    warm_start: Optional[tuple[np.ndarray, np.ndarray]],
    keep_model_for_warmstart: bool,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    use_batch_progress = bool(batch_progress is not None and batch_progress.enabled)
    log_path = os.path.join(outdir, f"{prefix}.{int(k)}.adamix.log")
    logger = setup_logging(log_path)
    apply_blas_thread_env(int(resolved_threads))
    if not use_batch_progress:
        logger.info("")
    if (int(total_k) > 1) and (not use_batch_progress):
        logger.info(
            f"K scan progress: {int(k_index)}/{int(total_k)} "
            f"(K={int(k)}, spec={k_spec})"
        )

    q_out = os.path.join(outdir, f"{prefix}.{int(k)}.Q.txt")
    p_npy_out = os.path.join(outdir, f"{prefix}.{int(k)}.P.npy")
    p_site_out = os.path.join(outdir, f"{prefix}.{int(k)}.P.site")
    structure_fmt = "svg"
    structure_out = os.path.join(outdir, f"{prefix}.{int(k)}.admix.{structure_fmt}")

    model_rows = [
        ("K", int(k)),
        ("Solver", str(args.solver)),
        ("max_iter", int(args.max_iter)),
        ("tol", float(args.tol)),
        ("CVerror", (str(cv_label) if cv_enabled else "off")),
        ("Tag labels", ("off" if len(tag_samples) == 0 else f"{len(tag_samples)} sample(s)")),
    ]
    if int(total_k) > 1:
        model_rows.append(("K spec", str(k_spec)))
        model_rows.append(("K warmstart", ("on" if warm_start is not None else "off")))

    cfg_mute: list[tuple[logging.Handler, int]] = []
    if not emit_config_to_stdout:
        cfg_mute = _mute_stdout_info_logs(logger)
    try:
        emit_cli_configuration(
            logger,
            app_title="JanusX - ADAMIXTURE",
            config_title="ADAMIXTURE CONFIG",
            host=socket.gethostname(),
            sections=[
                (
                    "Input/Output",
                    [
                        ("Genotype", genotype_path),
                        ("Input type", source_label),
                        ("Chunk size", int(args.chunksize)),
                        ("SNPs only", bool(args.snps_only)),
                        ("MAF threshold", float(args.maf)),
                        ("Miss threshold", float(args.geno)),
                        ("Output dir", outdir),
                        ("Prefix", prefix),
                    ],
                ),
                (
                    "Runtime",
                    [
                        ("Threads", f"{int(resolved_threads)} ({detected_threads} available)"),
                        ("Seed", int(args.seed)),
                    ],
                ),
                ("Model", model_rows),
            ],
            footer_rows=[
                ("Output prefix", os.path.join(outdir, prefix)),
                ("Q output", q_out),
                ("P output (.npy)", p_npy_out),
                ("P site (.site)", p_site_out),
                ("Structure plot", structure_out),
                ("Log file", log_path),
            ],
            emit_to_stdout=bool(emit_config_to_stdout),
        )
    finally:
        _restore_handler_levels(cfg_mute)

    cfg = ADAMixtureConfig(
        genotype_path=genotype_path,
        k=int(k),
        outdir=outdir,
        prefix=prefix,
        seed=int(args.seed),
        threads=max(1, int(resolved_threads)),
        chunk_size=max(1000, int(args.chunksize)),
        snps_only=bool(args.snps_only),
        maf=float(args.maf),
        geno=float(args.geno),
        solver=str(args.solver),
        max_iter=max(1, int(args.max_iter)),
        tol=float(args.tol),
        cv=int(cv_value if cv_enabled else 0),
    )

    progress_ui: Optional[_AdmixtureCliProgress] = None
    callback = None
    muted: list[tuple[logging.Handler, int]] = []
    cverror_result: Optional[dict[str, float]] = None
    structure_plot_path: Optional[str] = None
    t_one = time.time()
    try:
        if use_batch_progress:
            muted = _mute_stdout_info_logs(logger)
        elif enable_spinner:
            progress_ui = _AdmixtureCliProgress(logger, enabled=True)
            callback = progress_ui.on_event
            muted = _mute_stdout_info_logs(logger)
        try:
            p_mat, q_mat, m, n = train_adamixture(
                cfg,
                logger,
                callback=callback,
                g_matrix=shared_g,
                warm_start=warm_start,
            )
        finally:
            if progress_ui is not None:
                progress_ui.close()
            if not use_batch_progress:
                _restore_handler_levels(muted)
        q_ids, p_site_lines = _safe_sample_ids_and_site_lines(
            genotype_path,
            snps_only=bool(args.snps_only),
            maf=float(args.maf),
            missing_rate=float(args.geno),
            expected_n=int(n),
            expected_m=int(m),
            logger=logger,
        )
        _write_q_txt_with_row_ids(q_out, q_ids, np.asarray(q_mat, dtype=np.float64))
        _write_p_npy_and_site(
            p_npy_out,
            p_site_out,
            p_site_lines,
            np.asarray(p_mat, dtype=np.float32),
        )
        structure_plot_path, _plot_meta = _render_structure_plot(
            q_mat=np.asarray(q_mat, dtype=np.float64),
            q_ids=list(q_ids),
            out_path=str(structure_out),
            tag_samples=list(tag_samples),
            logger=logger,
        )
        if cv_enabled:
            cv_progress_callback = None
            if use_batch_progress:
                def _cv_progress(event: str, payload: dict[str, Any]) -> None:
                    if batch_progress is None:
                        return
                    e = str(event)
                    if e == "cv_start":
                        batch_progress.cv_start(
                            k=int(k),
                            folds=int(payload.get("folds", 1)),
                        )
                        return
                    if e == "cv_fold_done":
                        batch_progress.cv_advance(steps=1)
                        return
                    if e == "cv_done":
                        batch_progress.cv_complete()
                        return
                cv_progress_callback = _cv_progress
            cv_status: Optional[CliStatus] = None
            cv_muted: list[tuple[logging.Handler, int]] = []
            if (not use_batch_progress) and enable_spinner:
                cv_muted = _mute_stdout_info_logs(logger)
                cv_status = CliStatus("CVerror estimation", enabled=True)
                cv_status.__enter__()
            try:
                cverror_result = evaluate_adamixture_cverror(
                    cfg,
                    logger,
                    progress_callback=cv_progress_callback,
                    g_full=shared_g,
                )
                if cv_status is not None:
                    cv_status.complete("CVerror estimation ...Finished")
            except Exception:
                if cv_status is not None:
                    cv_status.fail("CVerror estimation ...Failed")
                raise
            finally:
                if cv_status is not None:
                    cv_status.__exit__(None, None, None)
                _restore_handler_levels(cv_muted)
    except Exception as e:
        _restore_handler_levels(muted)
        if progress_ui is not None:
            try:
                progress_ui.fail("ADAMIXTURE ...Failed")
            except Exception:
                pass
        logger.exception(f"ADAMIXTURE failed: {e}")
        print_failure("ADAMIXTURE ...Failed", force_color=True)
        raise

    _restore_handler_levels(muted)
    wall = float(time.time() - t_one)
    end_mute: list[tuple[logging.Handler, int]] = []
    if use_batch_progress:
        end_mute = _mute_stdout_info_logs(logger)
    try:
        logger.info(f"Q output: {format_path_for_display(q_out)}")
        logger.info(f"P output (.npy): {format_path_for_display(p_npy_out)}")
        logger.info(f"P site (.site): {format_path_for_display(p_site_out)}")
        if structure_plot_path is not None:
            logger.info(
                f"Structure plot: {format_path_for_display(str(structure_plot_path))}"
            )
        if cv_enabled:
            if cverror_result is None:
                raise RuntimeError("Internal error: CVerror result is missing.")
            logger.info(
                "CVerror: "
                f"{float(cverror_result['cverror']):.6f} "
                f"(SE={float(cverror_result['cverror_se']):.6f}, "
                f"folds={int(cverror_result['cv_folds'])})"
            )
        now = datetime.now()
        logger.info("")
        log_success(logger, f"K={int(k)} finished in {wall:.2f} seconds")
        logger.info(now.strftime("%Y-%m-%d %H:%M:%S"))
    finally:
        _restore_handler_levels(end_mute)
    row = {
        "k": int(k),
        "cverror": (None if cverror_result is None else float(cverror_result["cverror"])),
        "cverror_se": (None if cverror_result is None else float(cverror_result["cverror_se"])),
        "cv_folds": (0 if cverror_result is None else int(cverror_result["cv_folds"])),
        "wall": float(wall),
        "q_out": q_out,
        "p_npy_out": p_npy_out,
        "p_site_out": p_site_out,
        "structure_plot": (None if structure_plot_path is None else str(structure_plot_path)),
        "log_path": log_path,
    }
    model_state: Optional[dict[str, Any]] = None
    if keep_model_for_warmstart:
        model_state = {
            "k": int(k),
            "p": np.ascontiguousarray(np.asarray(p_mat, dtype=np.float32)),
            "q": np.ascontiguousarray(np.asarray(q_mat, dtype=np.float32)),
        }
    return row, model_state


def _build_parser() -> CliArgumentParser:
    parser = CliArgumentParser(
        prog="jx adamixture",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx adamixture -bfile data/geno -k 8 -o out -prefix cohort",
            "jx adamixture -vcf data/geno.vcf.gz -k 2..10 -t 16",
            "jx adamixture -vcf data/geno.vcf.gz -k 2..10..3 -t 16",
            "jx adamixture -vcf data/geno.vcf.gz -k 2,5,8 -t 16",
            "jx adamixture -vcf data/geno.vcf.gz -k 2..10 -cv",
            "jx adamixture -vcf data/geno.vcf.gz -k 2..10 -cv 5",
            "jx adamixture -vcf data/geno.vcf.gz -k 6 -tag sample1,sample2",
            "jx adamixture -vcf data/geno.vcf.gz -k 6 --tag tag_samples.txt",
            "jx adamixture -vcf data/geno.vcf.gz -k 2..10 -cv --k-warmstart",
        ]),
    )

    req_geno = parser.add_argument_group("Required Genotype Input (Choose one)")
    geno = req_geno.add_mutually_exclusive_group(required=True)
    geno.add_argument("-bfile", "--bfile", type=str, help="PLINK prefix (.bed/.bim/.fam).")
    geno.add_argument("-vcf", "--vcf", type=str, help="VCF/VCF.GZ genotype file.")
    geno.add_argument("-hmp", "--hmp", type=str, help="HMP/HMP.GZ genotype file.")
    geno.add_argument(
        "-file",
        "--file",
        type=str,
        help="Text/NumPy genotype matrix prefix (requires sidecar .id).",
    )
    req_model = parser.add_argument_group("Required Nclusters")
    req_model.add_argument(
        "-k",
        "--k",
        type=str,
        required=True,
        help=(
            "K spec: single (8), range (2..10 or 2:10), "
            "stepped range (2..10..3, 2:10:3, or 2..10:3), or list (2,5,8)."
        ),
    )
    input_output = parser.add_argument_group("Input/Output Arguments")
    input_output.add_argument(
        "-o",
        "--out",
        type=str,
        default=".",
        help="Output directory (default: current directory).",
    )
    input_output.add_argument("-prefix", "--prefix", type=str, default=None, help="Output prefix.")
    input_output.add_argument(
        "-tag",
        "--tag",
        type=str,
        default=None,
        help=(
            "Optional sample labels for structure x-axis. "
            "Use a file path (one sample ID per line) or comma-separated sample IDs."
        ),
    )
    input_output.add_argument(
        "-chunksize",
        "--chunksize",
        type=int,
        default=50000,
        help="Number of SNPs per loading chunk (default: 50000).",
    )
    input_output.add_argument(
        "-snps-only",
        "--snps-only",
        action="store_true",
        default=False,
        help="Input VCF/HMP: keep SNP variants only during loading.",
    )
    input_output.add_argument(
        "-maf",
        "--maf",
        type=float,
        default=0.02,
        help="Minor allele frequency threshold in loading stage (default: 0.02).",
    )
    input_output.add_argument(
        "-geno",
        "--geno",
        type=float,
        default=0.05,
        help="Missing-rate threshold in loading stage (default: 0.05).",
    )
    runtime = parser.add_argument_group("Runtime Arguments")
    runtime.add_argument(
        "-t",
        "--thread",
        dest="thread",
        type=int,
        default=detect_effective_threads(),
        help="Number of CPU threads (default: %(default)s).",
    )
    runtime.add_argument(
        "-threads",
        "--threads",
        dest="thread",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    runtime.add_argument("-seed", "--seed", type=int, default=42, help="Random seed (default: 42).")

    model = parser.add_argument_group("Model/Optimization Arguments")
    model.add_argument(
        "-solver",
        "--solver",
        type=str,
        default="adam-em",
        choices=["auto", "adam", "adam-em"],
        help="Optimization solver (default: adam-em).",
    )
    model.add_argument(
        "-max-iter",
        "--max-iter",
        dest="max_iter",
        type=int,
        default=500,
        help="Maximum optimization iterations (default: 500).",
    )
    model.add_argument(
        "-tol",
        "--tol",
        type=float,
        default=1e-5,
        help="Convergence tolerance (default: 1e-5).",
    )
    model.add_argument(
        "-cv",
        "--cv",
        nargs="?",
        default=None,
        const=str(_DEFAULT_CVERROR_FOLDS),
        metavar="N",
        help=(
            "Enable CVerror. No value -> 5-fold; integer N -> N-fold. "
            "Omit -cv to disable CVerror."
        ),
    )
    model.add_argument(
        "--k-warmstart",
        action="store_true",
        default=False,
        help=(
            "Enable warm-start between consecutive K values in K-scan mode "
            "(default: disabled, for consistency)."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not (0.0 <= float(args.maf) <= 0.5):
        parser.error("-maf must be within [0, 0.5].")
    if not (0.0 <= float(args.geno) <= 1.0):
        parser.error("-geno must be within [0, 1.0].")
    if float(args.tol) <= 0:
        parser.error("-tol/--tol must be > 0.")
    if int(args.max_iter) <= 0:
        parser.error("-max-iter/--max-iter must be a positive integer.")
    if int(args.thread) <= 0:
        parser.error("-t/--thread must be a positive integer.")
    try:
        k_values = _parse_k_spec(str(args.k))
    except Exception as ex:
        parser.error(str(ex))
    try:
        cv_enabled, cv_value, cv_label = _parse_cv_spec(args.cv)
    except Exception as ex:
        parser.error(str(ex))
    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    thread_capped = False
    resolved_threads = int(args.thread)
    if int(resolved_threads) > int(detected_threads):
        thread_capped = True
        resolved_threads = int(detected_threads)

    outdir = os.path.normpath(str(args.out or "."))
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    tmp_logger = logging.getLogger("janusx.adamixture.bootstrap")
    if not tmp_logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        tmp_logger.addHandler(h)
    tmp_logger.setLevel(logging.INFO)
    tmp_logger.propagate = False
    try:
        genotype_path, source_label, auto_prefix = _resolve_input(args, tmp_logger)
    except Exception as e:
        print_failure(str(e), force_color=True)
        raise
    tag_samples = _parse_tag_samples(args.tag, logger=tmp_logger)

    prefix = args.prefix or auto_prefix
    multi_k = len(k_values) > 1
    enable_spinner = bool(stdout_is_tty())
    k_warmstart_enabled = bool(multi_k and bool(args.k_warmstart))
    use_geno_cache = _bool_env("JANUSX_ADMX_CACHE_GENO", True)
    if multi_k:
        if cv_enabled:
            cv_msg = str(cv_label)
        else:
            cv_msg = "off"
        print_success(
            f"K-scan mode: {len(k_values)} K values ({args.k})",
            force_color=True,
        )
        print(
            f"CVerror={cv_msg}, solver={args.solver}, "
            f"max_iter={int(args.max_iter)}, threads={int(resolved_threads)}, "
            f"k_warmstart={'on' if k_warmstart_enabled else 'off'}, "
            f"geno_cache={'on' if use_geno_cache else 'off'}, "
            f"tag={'on' if len(tag_samples) > 0 else 'off'}",
            flush=True,
        )
        if k_warmstart_enabled:
            tmp_logger.warning(
                "Warning: --k-warmstart is experimental and may alter convergence; "
                "for strict reproducibility use default (warm-start off)."
            )
    if thread_capped:
        tmp_logger.warning(
            f"Warning: Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(resolved_threads)}."
        )
    apply_blas_thread_env(int(resolved_threads))
    maybe_warn_non_openblas(
        logger=tmp_logger,
        strict=require_openblas_by_default(),
    )
    shared_g: Optional[np.ndarray] = None
    if use_geno_cache:
        shared_g = load_genotype_u8_matrix(
            genotype_path,
            chunk_size=max(1000, int(args.chunksize)),
            snps_only=bool(args.snps_only),
            maf=float(args.maf),
            missing_rate=float(args.geno),
        )
    prev_model_state: Optional[dict[str, Any]] = None
    summary_rows: list[dict[str, Any]] = []
    with _AdmixtureBatchProgress(
        enabled=bool(multi_k),
        total_k=int(len(k_values)),
        cv_enabled=bool(cv_enabled),
        cv_label=str(cv_label),
    ) as batch_progress:
        for idx, k in enumerate(k_values, start=1):
            batch_progress.start_k(k=int(k))
            warm_start = None
            if k_warmstart_enabled:
                warm_start = _build_k_warm_start(
                    prev_model_state,
                    target_k=int(k),
                    seed=int(args.seed),
                )
            row, model_state = _run_single_k(
                genotype_path=genotype_path,
                source_label=source_label,
                prefix=prefix,
                outdir=outdir,
                k=int(k),
                k_spec=str(args.k),
                k_index=int(idx),
                total_k=int(len(k_values)),
                args=args,
                detected_threads=int(detected_threads),
                resolved_threads=int(resolved_threads),
                enable_spinner=bool(enable_spinner),
                emit_config_to_stdout=bool(not multi_k),
                batch_progress=batch_progress,
                cv_enabled=bool(cv_enabled),
                cv_value=int(cv_value),
                cv_label=str(cv_label),
                tag_samples=list(tag_samples),
                shared_g=shared_g,
                warm_start=warm_start,
                keep_model_for_warmstart=bool(k_warmstart_enabled),
            )
            summary_rows.append(row)
            if k_warmstart_enabled:
                prev_model_state = model_state
            batch_progress.advance_k(steps=1)

    summary_log = os.path.join(outdir, f"{prefix}.adamixture.summary.log")
    logger = setup_logging(summary_log)
    if cv_enabled:
        cv_rows = [r for r in summary_rows if r.get("cverror") is not None]
        summary_tsv = os.path.join(outdir, f"{prefix}.adamixture.cverror.summary.tsv")
        _write_cverror_summary_tsv(summary_tsv, cv_rows)
        logger.info("")
        _log_cverror_summary(logger, cv_rows, summary_tsv)
    logger.info("")
    _log_outputs_summary(logger, summary_rows)

    wall = float(time.time() - t0)
    now = datetime.now()
    logger.info("")
    logger.info(f"Summary log: {format_path_for_display(summary_log)}")
    logger.info(f"Output dir: {format_path_for_display(outdir)}")
    log_success(logger, f"Finished. Total wall time: {wall:.2f} seconds")
    logger.info(now.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
