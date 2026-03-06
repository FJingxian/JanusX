# -*- coding: utf-8 -*-
"""
JanusX: High-Performance GWAS Command-Line Interface

Design overview
---------------
Models:
  - LMM     : streaming, low-memory implementation (slim.LMM)
  - LM      : streaming, low-memory implementation (slim.LM)
  - FarmCPU : in-memory implementation (pyBLUP.farmcpu) that loads the
              full genotype matrix

Execution mode (automatic)
--------------------------
  - No explicit "low-memory" flag is required.
  - LMM/LM always run in streaming mode via rust2py.gfreader.load_genotype_chunks.
  - FarmCPU always runs on the full in-memory genotype matrix.

Caching
-------
  - GRM (kinship) and PCA (Q matrix) are cached in the genotype directory
    for streaming LMM/LM runs:
      * GRM: {geno_prefix}.k.{method}.npy
      * Q   : {geno_prefix}.q.{pcdim}.txt

Covariates
----------
  - The --cov option is shared by LMM, LM, and FarmCPU.
  - For LMM/LM, the covariate file must match the genotype sample order
    (inspect_genotype_file IDs).
  - For FarmCPU, the covariate file must match the genotype sample order
    (famid from the genotype matrix).

Citation
--------
  https://github.com/FJingxian/JanusX/
"""

import os
import time
import socket
import argparse
import logging
import sys
import threading
import warnings
import multiprocessing as mp
import concurrent.futures as cf
import textwrap
from datetime import datetime
from typing import Union, Optional
import uuid

from janusx.pyBLUP.QK2 import GRM
from janusx.pyBLUP.stream_grm import (
    auto_stream_grm_chunk_size,
    build_streaming_grm_from_chunks,
)

# ---- Matplotlib backend configuration (non-interactive, server-safe) ----
for key in ["MPLBACKEND"]:
    if key in os.environ:
        del os.environ[key]

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['svg.hashsalt'] = 'hello'

import matplotlib.pyplot as plt
from matplotlib import font_manager as mpl_font_manager
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from joblib import cpu_count
import psutil
from janusx.bioplotkit import GWASPLOT
from janusx.pyBLUP import QK
from janusx.gfreader import (
    load_genotype_chunks,
    inspect_genotype_file,
    auto_mmap_window_mb,
)
from janusx.pyBLUP import LMM, LM, FastLMM, farmcpu
from ._common.log import setup_logging
from ._common.config_render import emit_cli_configuration
from ._common.helptext import cli_help_formatter, minimal_help_epilog
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_plink_prefix_exists,
)
from ._common.prefetch import prefetch_iter
from ._common.status import (
    CliStatus,
    get_rich_spinner_name,
    print_success,
    format_elapsed,
)
from ._common.gwas_history import record_gwas_run

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    _HAS_RICH_PROGRESS = True
except Exception:
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    TimeRemainingColumn = None  # type: ignore[assignment]
    _HAS_RICH_PROGRESS = False

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore[assignment]
    _HAS_TQDM = False


# ======================================================================
# Basic utilities
# ======================================================================

def _section(logger:logging.Logger, title: str) -> None:
    """Emit a formatted log section header with a leading blank line."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def _emit_trait_header(
    logger: logging.Logger,
    trait_name: str,
    n_idv: int,
    *,
    pve: Optional[float] = None,
    use_spinner: bool = False,
    width: int = 60,
) -> None:
    """
    Emit trait header as one line: "<trait> (n=<idv>)".
    If it exceeds section width, wrap at trait/name boundary.
    """
    trait = str(trait_name)
    pve_val: Optional[float] = None
    if pve is not None:
        try:
            pve_tmp = float(pve)
            if np.isfinite(pve_tmp):
                pve_val = pve_tmp
        except Exception:
            pve_val = None
    if pve_val is None:
        n_line = f"(n={int(n_idv)})"
    else:
        n_line = f"(n={int(n_idv)}; pve={pve_val:.3f})"
    full = f"{trait} {n_line}"
    if len(full) <= int(width):
        lines = [full]
    else:
        if len(trait) <= int(width):
            lines = [trait, n_line]
        else:
            trait_lines = textwrap.wrap(
                trait,
                width=max(10, int(width)),
                break_long_words=True,
                break_on_hyphens=False,
            )
            lines = trait_lines + [n_line]
    for ln in lines:
        _log_file_only(logger, logging.INFO, ln)
        if use_spinner:
            print(ln, flush=True)
        else:
            logger.info(ln)


def _emit_gwas_summary(
    logger: logging.Logger,
    rows: list[dict[str, object]],
) -> None:
    if len(rows) == 0:
        return

    _section(logger, "Summary")
    headers = [
        "pheno",
        "model",
        "nidv",
        "nsnp",
        "pve",
        "mem(G)",
        "Ctime(s)",
        "Vtime(s)",
    ]

    out_rows: list[list[str]] = []
    for r in rows:
        pve_raw = r.get("pve", None)
        pve_text = "NA"
        try:
            if pve_raw is not None:
                pve_val = float(pve_raw)
                if np.isfinite(pve_val):
                    pve_text = f"{pve_val:.3f}"
        except Exception:
            pve_text = "NA"
        out_rows.append(
            [
                str(r.get("phenotype", "")),
                str(r.get("model", "")),
                f"{int(r.get('nidv', 0))}",
                f"{int(r.get('eff_snp', 0))}",
                pve_text,
                f"{float(r.get('peak_rss_gb', 0.0)):.2f}",
                f"{float(r.get('gwas_time_s', 0.0)):.1f}",
                f"{float(r.get('viz_time_s', 0.0)):.1f}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in out_rows:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))

    header_line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    logger.info(header_line)
    for row in out_rows:
        logger.info("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def _log_file_only(logger: logging.Logger, level: int, message: str) -> None:
    """
    Emit a log record to file handlers only.
    Used to keep rich terminal output clean while preserving full log files.
    """
    msg = str(message)
    handled = False
    try:
        for handler in getattr(logger, "handlers", []):
            if isinstance(handler, logging.FileHandler):
                record = logger.makeRecord(
                    logger.name,
                    level,
                    __file__,
                    0,
                    msg,
                    args=(),
                    exc_info=None,
                    func=None,
                    extra=None,
                )
                handler.handle(record)
                handled = True
    except Exception:
        handled = False
    if not handled:
        logger.log(level, msg)


def _log_info(logger: logging.Logger, message: str, *, use_spinner: bool = False) -> None:
    if use_spinner:
        _log_file_only(logger, logging.INFO, str(message))
    else:
        logger.info(str(message))


def _rich_success(
    logger: logging.Logger,
    message: str,
    *,
    use_spinner: bool = False,
    log_message: Optional[str] = None,
) -> None:
    msg = str(message)
    file_msg = str(msg if log_message is None else log_message)
    if use_spinner:
        _log_file_only(logger, logging.INFO, file_msg)
        print_success(msg)
    else:
        logger.info(file_msg)


class _ProgressAdapter:
    """
    Progress bar adapter with rich-first rendering and tqdm fallback.
    """
    def __init__(self, total: int, desc: str) -> None:
        self.total = int(max(0, total))
        self.desc = str(desc)
        self._backend = "none"
        self._progress = None
        self._task_id = None
        self._tqdm = None
        self._start_ts = time.monotonic()
        self._finished = False
        self._done = 0
        self._tick = 0
        self._memory_text = ""
        self._memory_until_tick = 0

        if _HAS_RICH_PROGRESS and sys.stdout.isatty():
            try:
                self._progress = Progress(
                    SpinnerColumn(
                        spinner_name=get_rich_spinner_name(),
                        style="cyan",
                        finished_text="[green]✔︎[/green]",
                    ),
                    TextColumn("[green]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.fields[metric]}"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    transient=True,
                )
                self._progress.start()
                self._task_id = self._progress.add_task(
                    self.desc,
                    total=self.total,
                    metric="0.0%",
                )
                self._backend = "rich"
            except Exception:
                self._progress = None
                self._task_id = None

        if self._backend == "none" and _HAS_TQDM:
            self._tqdm = tqdm(
                total=self.total,
                desc=self.desc,
                ascii=False,
                leave=False,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                           "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
            self._backend = "tqdm"

    def _metric_text(self) -> str:
        if self._memory_text and self._tick <= self._memory_until_tick:
            return self._memory_text
        if self.total <= 0:
            pct = 0.0
        else:
            pct = 100.0 * float(self._done) / float(self.total)
        return f"{pct:>6.1f}%"

    def update(self, n: int) -> None:
        step = int(max(0, n))
        if step == 0:
            return
        self._done += step
        self._tick += 1
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                advance=step,
                metric=self._metric_text(),
            )
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.update(step)
            self._tqdm.set_postfix_str(self._metric_text())

    def set_postfix(self, **kwargs: object) -> None:
        if len(kwargs) == 0:
            return
        if len(kwargs) == 1 and "memory" in kwargs:
            self._memory_text = str(kwargs["memory"]).replace(" ", "")
            self._memory_until_tick = self._tick + 5
            if self._backend == "rich" and self._progress is not None and self._task_id is not None:
                self._progress.update(self._task_id, metric=self._metric_text())
            elif self._backend == "tqdm" and self._tqdm is not None:
                self._tqdm.set_postfix_str(self._metric_text())
            return
        text = " ".join([f"{k} ~ {v}" for k, v in kwargs.items()])
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, metric=text)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.set_postfix_str(text)

    def set_desc(self, desc: str) -> None:
        self.desc = str(desc)
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=self.desc)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.set_description_str(self.desc)

    def finish(self) -> None:
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=self.total)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.n = self._tqdm.total
            self._tqdm.refresh()
        self._finished = True

    def close(
        self,
        done_text: str = "Finished",
        show_done: bool = True,
        success_style: bool = True,
    ) -> None:
        elapsed = format_elapsed(time.monotonic() - self._start_ts)
        if self._backend == "rich" and self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None
        if self._finished and show_done:
            msg = f"{self.desc} ...{done_text} [{elapsed}]"
            if success_style:
                print_success(msg)
            else:
                print(f"✔︎ {msg}", flush=True)


def fastplot(
    gwasresult: pd.DataFrame,
    phenosub: np.ndarray,
    xlabel: str = "",
    outpdf: str = "fastplot.pdf",
) -> None:
    """
    Generate diagnostic plots for GWAS results: phenotype histogram, Manhattan, and QQ.
    """
    mpl.rcParams["font.size"] = 12
    results = gwasresult.astype({"pos": "int64"})
    fig = plt.figure(figsize=(16, 4), dpi=300)
    try:
        layout = [["A", "B", "B", "C"]]
        axes:dict[str,plt.Axes] = fig.subplot_mosaic(mosaic=layout)

        gwasplot = GWASPLOT(results)
        scatter_size = 8.0

        # A: phenotype distribution
        pheno = np.asarray(phenosub, dtype="float64").reshape(-1)
        pheno = pheno[np.isfinite(pheno)]
        n_samples = int(pheno.size)
        label_base = str(xlabel).strip() if str(xlabel).strip() else "phenotype"

        if n_samples > 0:
            counts, edges, _ = axes["A"].hist(
                pheno,
                bins=15,
                color="black",
                edgecolor="none",
                alpha=1.0,
            )
            # Overlay seaborn-like KDE curve, scaled to histogram "Count" axis.
            if counts.size > 1 and np.unique(pheno).size > 1:
                try:
                    kde = gaussian_kde(pheno)
                    x_grid = np.linspace(float(np.min(pheno)), float(np.max(pheno)), 256)
                    y_density = kde(x_grid)
                    bin_width = float(np.mean(np.diff(edges)))
                    y_count = y_density * float(n_samples) * bin_width
                    axes["A"].plot(x_grid, y_count, color="#B3B3B3", linewidth=1.6)
                except Exception:
                    pass
        else:
            axes["A"].text(
                0.5,
                0.5,
                "No valid phenotype values",
                ha="center",
                va="center",
                transform=axes["A"].transAxes,
            )

        x_label_text = f"{label_base} (n={n_samples})"
        if _contains_cjk(x_label_text) and not _ensure_cjk_font():
            # No CJK font found on this host; fallback to ASCII to avoid glyph warnings.
            x_label_text = f"phenotype (n={n_samples})"
        axes["A"].set_xlabel(x_label_text)
        axes["A"].set_ylabel("Count")

        # B: Manhattan plot
        gwasplot.manhattan(
            -np.log10(1 / results.shape[0]),
            ax=axes["B"],
            rasterized=True,
            s=scatter_size,
        )
        snp_n = int(results.shape[0])
        if snp_n >= 1_000_000:
            snp_val = snp_n / 1_000_000.0
            snp_suffix = "M"
        else:
            snp_val = snp_n / 1_000.0
            snp_suffix = "K"
        snp_text = f"{snp_val:.3f}".rstrip("0").rstrip(".")
        axes["B"].set_xlabel(f"Chromosome (SNP={snp_text}{snp_suffix})")

        # C: QQ plot
        gwasplot.qq(ax=axes["C"], scatter_size=scatter_size)

        # Align QQ with Manhattan:
        # - QQ ylim follows Manhattan ylim
        # - QQ xlim lower bound follows Manhattan ylim lower bound
        # - QQ xlim upper bound keeps current auto-scaled maximum
        manh_ymin, manh_ymax = axes["B"].get_ylim()
        qq_xmin, qq_xmax = axes["C"].get_xlim()
        axes["C"].set_ylim(manh_ymin, manh_ymax)
        axes["C"].set_xlim(left=manh_ymin, right=max(qq_xmax, manh_ymin + 1e-9))

        fig.tight_layout()
        fig.savefig(outpdf, transparent=False, facecolor="white")
    finally:
        plt.close(fig)


def determine_genotype_source(args) -> tuple[str, str]:
    """
    Resolve genotype input and output prefix from CLI arguments.
    """
    def _strip_geno_suffix(name: str) -> str:
        low = name.lower()
        if low.endswith(".vcf.gz"):
            return name[: -len(".vcf.gz")]
        for ext in (".vcf", ".txt", ".tsv", ".csv", ".npy"):
            if low.endswith(ext):
                return name[: -len(ext)]
        return name

    if args.vcf:
        gfile = args.vcf
        prefix = _strip_geno_suffix(os.path.basename(gfile))
    elif args.file:
        gfile = args.file
        prefix = _strip_geno_suffix(os.path.basename(gfile))
    elif args.bfile:
        gfile = args.bfile
        prefix = os.path.basename(gfile)
    else:
        raise ValueError("No genotype input specified. Use -vcf, -file or -bfile.")

    if args.prefix is not None:
        prefix = args.prefix

    gfile = gfile.replace("\\", "/")
    return gfile, prefix


def genotype_cache_prefix(genofile: str) -> str:
    """
    Construct a cache prefix within the genotype directory.
    """
    base = os.path.basename(genofile)
    low = base.lower()
    if low.endswith(".vcf.gz"):
        base = base[: -len(".vcf.gz")]
    else:
        for ext in (".vcf", ".txt", ".tsv", ".csv", ".npy"):
            if low.endswith(ext):
                base = base[: -len(ext)]
                break
    cache_dir = os.path.dirname(genofile) or "."
    return os.path.join(cache_dir, base).replace("\\", "/")

def latest_genotype_mtime(genofile: str) -> Union[float, None]:
    """
    Return the latest modification time of genotype input files.

    - PLINK prefix: max mtime of .bed/.bim/.fam (if all exist)
    - File input  : mtime of the file path itself
    """
    bed = f"{genofile}.bed"
    bim = f"{genofile}.bim"
    fam = f"{genofile}.fam"
    if all(os.path.isfile(p) for p in (bed, bim, fam)):
        return max(os.path.getmtime(bed), os.path.getmtime(bim), os.path.getmtime(fam))
    if os.path.isfile(genofile):
        return os.path.getmtime(genofile)
    return None


def _basename_only(path: str) -> str:
    p = str(path).replace("\\", "/").rstrip("/")
    b = os.path.basename(p)
    return b if b else p


def _read_id_file(
    path: str,
    logger,
    label: str,
    *,
    use_spinner: bool = False,
    show_status: bool = True,
) -> Union[np.ndarray, None]:
    if not os.path.isfile(path):
        logger.warning(f"{label} ID file not found: {path}")
        return None
    src = _basename_only(path)

    def _load_ids() -> Union[np.ndarray, None]:
        try:
            df = pd.read_csv(
                path, sep=r"\s+", header=None, usecols=[0],
                dtype=str, keep_default_na=False
            )
        except Exception:
            df = pd.read_csv(
                path, sep=None, engine="python", header=None, usecols=[0],
                dtype=str, keep_default_na=False
            )
        if not df.empty and df.iloc[0, 0] == "":
            # sep=None can mis-parse single-column ID files; fallback to whitespace
            df = pd.read_csv(
                path, sep=r"\s+", header=None, usecols=[0],
                dtype=str, keep_default_na=False
            )
        if df.empty:
            logger.warning(f"{label} ID file is empty: {path}")
            return None
        ids0 = df.iloc[:, 0].astype(str).str.strip().to_numpy()
        if ids0.size == 0:
            logger.warning(f"{label} ID file has no usable IDs: {path}")
            return None
        return ids0

    if not show_status:
        return _load_ids()

    with CliStatus(f"Loading {label} ID from {src}...", enabled=bool(use_spinner)) as task:
        try:
            ids = _load_ids()
        except Exception:
            task.fail(f"Loading {label} ID from {src} ...Failed")
            raise
        if ids is None:
            task.complete(f"Loading {label} ID from {src} shape=(0,)")
            return None
        task.complete(f"Loading {label} ID from {src} shape=({ids.size},)")
    return ids


def _read_matrix_with_ids(path: str, logger, label: str) -> tuple[Union[np.ndarray, None], np.ndarray]:
    try:
        df = pd.read_csv(
            path, sep=None, engine="python", header=None,
            dtype={0: str}, keep_default_na=False
        )
    except Exception:
        df = pd.read_csv(
            path, sep=r"\s+", header=None,
            dtype={0: str}, keep_default_na=False
        )
    if df.shape[1] < 2:
        raise ValueError(f"{label} file must have IDs in column 1 and data in columns 2+.")
    ids = df.iloc[:, 0].astype(str).str.strip().to_numpy()
    data = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
    return ids, data


def _normalize_cov_inputs(cov_arg: Union[str, list[str], None]) -> list[str]:
    if cov_arg is None:
        return []
    if isinstance(cov_arg, str):
        raw = [cov_arg]
    else:
        raw = [str(x) for x in cov_arg]
    out: list[str] = []
    for x in raw:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def _parse_cov_site_token(token: str) -> Union[tuple[str, int], None]:
    """
    Parse site-style covariate token:
      - chr:pos
      - chr:start:end  (single-site only, requires start == end)
    Also accepts full-width colon '：'.
    """
    t = str(token).strip().replace("：", ":")
    parts = [p.strip() for p in t.split(":")]
    if len(parts) not in (2, 3):
        return None

    chrom = parts[0]
    if chrom == "":
        return None

    try:
        start = int(float(parts[1]))
    except Exception:
        return None
    if start <= 0:
        raise ValueError(f"Invalid site position in --cov: {token}")

    if len(parts) == 3:
        try:
            end = int(float(parts[2]))
        except Exception:
            return None
        if end <= 0:
            raise ValueError(f"Invalid site position in --cov: {token}")
        if end != start:
            raise ValueError(
                f"--cov site token must specify a single site (start=end), got: {token}"
            )

    return chrom, start


def _split_cov_sources(cov_inputs: list[str]) -> tuple[list[str], list[tuple[str, int]]]:
    cov_files: list[str] = []
    cov_sites: list[tuple[str, int]] = []
    for item in cov_inputs:
        parsed = _parse_cov_site_token(item)
        if parsed is None:
            cov_files.append(item)
        else:
            cov_sites.append(parsed)
    return cov_files, cov_sites


def _canon_site_key(chrom: str, pos: int) -> tuple[str, int]:
    c = str(chrom).strip().lower()
    if c.startswith("chr"):
        c = c[3:]
    return c, int(pos)


def _read_cov_file_flexible(
    path: str,
    sample_ids: Union[np.ndarray, None],
    logger,
    label: str = "Covariate",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read covariate file with flexible format:
      1) ID + cov columns
      2) numeric-only matrix (sample order must match genotype)
    """
    try:
        df = pd.read_csv(
            path, sep=None, engine="python", header=None,
            dtype=str, keep_default_na=False
        )
    except Exception:
        df = pd.read_csv(
            path, sep=r"\s+", header=None,
            dtype=str, keep_default_na=False
        )

    if df.empty:
        raise ValueError(f"{label} file is empty: {path}")

    sid = None if sample_ids is None else np.asarray(sample_ids, dtype=str)
    if sid is not None and df.shape[1] > 1:
        col0 = df.iloc[:, 0].astype(str).str.strip().to_numpy()
        overlap = len(set(col0) & set(sid))
        if overlap >= max(1, int(0.9 * len(sid))):
            data = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return col0, data

    data = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if sid is not None:
        if data.shape[0] != len(sid):
            raise ValueError(
                f"{label} rows ({data.shape[0]}) do not match genotype sample count ({len(sid)}): {path}"
            )
        return sid.copy(), data
    return np.arange(data.shape[0]).astype(str), data


def _load_site_covariates(
    genofile: str,
    site_specs: list[tuple[str, int]],
    sample_ids: np.ndarray,
    chunk_size: int,
    logger,
    use_spinner: bool = False,
) -> np.ndarray:
    """
    Load additive genotype values for requested SNP sites as covariates.
    Returns matrix shape (n_samples, n_sites).
    """
    if len(site_specs) == 0:
        return np.zeros((len(sample_ids), 0), dtype="float32")

    sample_ids = np.asarray(sample_ids, dtype=str)
    unique_sites: list[tuple[str, int]] = []
    seen_keys: set[tuple[str, int]] = set()
    for c, p in site_specs:
        k = _canon_site_key(c, p)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        unique_sites.append((c, p))
    need_keys: set[tuple[str, int]] = {_canon_site_key(c, p) for c, p in unique_sites}

    picked_rows: list[np.ndarray] = []
    picked_sites: list[tuple[str, int]] = []
    query_chunk = max(1, min(int(max(1, chunk_size)), max(1, len(site_specs))))
    for geno_chunk, site_chunk in load_genotype_chunks(
        genofile,
        chunk_size=query_chunk,
        maf=0.0,
        missing_rate=1.0,
        impute=True,
        model="add",
        snp_sites=unique_sites,
        sample_ids=sample_ids.tolist(),
    ):
        if geno_chunk.shape[0] == 0:
            continue
        for i, s in enumerate(site_chunk):
            key = _canon_site_key(str(s.chrom), int(s.pos))
            if key not in need_keys:
                continue
            picked_rows.append(np.asarray(geno_chunk[i], dtype="float32"))
            picked_sites.append((str(s.chrom), int(s.pos)))
            need_keys.remove(key)
        if len(need_keys) == 0:
            break

    missing_tokens = []
    for c, p in unique_sites:
        key = _canon_site_key(c, p)
        if key in need_keys:
            missing_tokens.append(f"{c}:{p}")
    if missing_tokens:
        show = ", ".join(missing_tokens[:10])
        if len(missing_tokens) > 10:
            show += f", ... ({len(missing_tokens)} missing)"
        raise ValueError(f"Some --cov SNP site(s) were not found in genotype: {show}")

    idx_map: dict[tuple[str, int], list[int]] = {}
    for i, (c, p) in enumerate(picked_sites):
        k = _canon_site_key(c, p)
        idx_map.setdefault(k, []).append(i)

    ordered_rows: list[np.ndarray] = []
    for c, p in site_specs:
        k = _canon_site_key(c, p)
        pool = idx_map.get(k, [])
        if len(pool) == 0:
            raise ValueError(f"--cov site lookup failed for {c}:{p}")
        ordered_rows.append(picked_rows[pool[0]])

    cov = np.vstack(ordered_rows).astype("float32", copy=False).T
    _log_info(
        logger,
        f"Loaded SNP-site covariates: {len(site_specs)} site(s), shape={cov.shape}",
        use_spinner=use_spinner,
    )
    return cov


def _load_covariates_for_models(
    cov_inputs: Union[str, list[str], None],
    genofile: str,
    sample_ids: np.ndarray,
    chunk_size: int,
    logger,
    context: str,
    use_spinner: bool = False,
) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    inputs = _normalize_cov_inputs(cov_inputs)
    if len(inputs) == 0:
        _rich_success(
            logger,
            f"Loading covariates ({context}) ...Skipped (none)",
            use_spinner=use_spinner,
            log_message=f"No covariate input provided for {context}; skipping covariates.",
        )
        return None, None

    cov_files, cov_sites = _split_cov_sources(inputs)
    parts: list[tuple[np.ndarray, np.ndarray, str]] = []

    for path in cov_files:
        if not os.path.isfile(path):
            logger.warning(f"Covariate file not found: {path}; skipped.")
            continue
        src = _basename_only(path)
        with CliStatus(f"Loading covariate from {src}...", enabled=bool(use_spinner)) as task:
            try:
                ids_i, cov_i = _read_cov_file_flexible(path, sample_ids, logger, label="Covariate")
                if cov_i.ndim == 1:
                    cov_i = cov_i.reshape(-1, 1)
            except Exception:
                task.fail(f"Loading covariate from {src} ...Failed")
                raise
            task.complete(
                f"Loading covariate from {src} (n={cov_i.shape[0]}, ncov={cov_i.shape[1]})"
            )
        parts.append((np.asarray(ids_i, dtype=str), np.asarray(cov_i, dtype="float32"), src))

    if len(cov_sites) > 0:
        for chrom, pos in cov_sites:
            token = f"{chrom}:{pos}"
            with CliStatus(f"Loading covariate from {token}...", enabled=bool(use_spinner)) as task:
                try:
                    cov_site = _load_site_covariates(
                        genofile=genofile,
                        site_specs=[(chrom, pos)],
                        sample_ids=np.asarray(sample_ids, dtype=str),
                        chunk_size=chunk_size,
                        logger=logger,
                        use_spinner=use_spinner,
                    )
                except Exception:
                    task.fail(f"Loading covariate from {token} ...Failed")
                    raise
                task.complete(
                    f"Loading covariate from {token} (n={cov_site.shape[0]}, ncov={cov_site.shape[1]})"
                )
            parts.append((np.asarray(sample_ids, dtype=str), cov_site, token))

    if len(parts) == 0:
        logger.warning(f"All covariate inputs are empty/unavailable for {context}; skipping.")
        return None, None

    common = set(parts[0][0].astype(str))
    for ids_i, _cov_i, _name in parts[1:]:
        common &= set(ids_i.astype(str))

    ordered_ids = [sid for sid in np.asarray(sample_ids, dtype=str) if sid in common]
    if len(ordered_ids) == 0:
        raise ValueError(f"No overlapping samples between genotype and covariates ({context}).")

    mats: list[np.ndarray] = []
    for ids_i, cov_i, _name in parts:
        idx_map = {sid: i for i, sid in enumerate(ids_i.astype(str))}
        take = [idx_map[sid] for sid in ordered_ids]
        sub = cov_i[take]
        mats.append(sub.astype("float32", copy=False))

    cov_all = np.concatenate(mats, axis=1).astype("float32", copy=False)
    cov_ids = np.asarray(ordered_ids, dtype=str)
    return cov_all, cov_ids


def load_phenotype(
    phenofile: str,
    ncol: Union[list[int] , None],
    logger,
    id_col: int = 0,
    use_spinner: bool = False,
) -> pd.DataFrame:
    """
    Load and preprocess phenotype table.

    Assumptions
    -----------
      - By default, the first column contains sample IDs.
      - If needed, set id_col=1 to use the second column as IDs (PLINK FID/IID).
    - Duplicated IDs are averaged.
    """
    try:
        df = pd.read_csv(phenofile, sep=None, engine="python", header=None)
    except Exception:
        df = pd.read_csv(phenofile, sep=r"\s+", header=None)

    if df.empty:
        raise ValueError("Phenotype file is empty.")
    if id_col >= df.shape[1]:
        raise ValueError(f"Phenotype file has no column {id_col + 1} for sample IDs.")

    # Detect header-like first row (non-numeric phenotype columns).
    header_like = False
    header_names = None
    if df.shape[0] > 1 and df.shape[1] > 1:
        row0 = pd.to_numeric(df.iloc[0, 1:], errors="coerce")
        row1 = pd.to_numeric(df.iloc[1, 1:], errors="coerce")
        if row0.isna().all() and row1.notna().any():
            header_like = True
            header_names = df.iloc[0, 1:].astype(str).tolist()
            df = df.iloc[1:, :].reset_index(drop=True)

    ids = df.iloc[:, id_col].astype(str)
    data = df.drop(columns=[id_col])
    # If using IID (column 2), drop FID (column 1) as well.
    if id_col == 1 and data.shape[1] >= 2:
        data = data.drop(columns=[0])
    if header_like and header_names is not None and len(header_names) == data.shape[1]:
        data.columns = header_names

    data = data.apply(pd.to_numeric, errors="coerce")
    pheno = data
    pheno.index = ids
    pheno = pheno.groupby(pheno.index).mean()

    if pheno.shape[1] <= 0:
        msg = (
            "No phenotype data found. Please check the phenotype file format.\n"
            f"{pheno.head()}"
        )
        logger.error(msg)
        raise ValueError(msg)

    if ncol is not None:
        if len(ncol) == 0 or np.min(ncol) >= pheno.shape[1]:
            msg = "Phenotype column index out of range."
            logger.error(msg)
            raise ValueError(msg)
        ncol = [i for i in ncol if i in range(pheno.shape[1])]
        _log_info(
            logger,
            "Phenotypes to be analyzed: " + "\t".join(map(str, pheno.columns[ncol])),
            use_spinner=use_spinner,
        )
        pheno = pheno.iloc[:, ncol]

    return pheno


# ======================================================================
# Low-memory LMM/LM: streaming GRM + PCA with caching
# ======================================================================

def _cache_prefix_tilde(genofile: str) -> str:
    """
    Cache prefix used by gfreader for VCF/TXT temporary converted files.
    """
    p = str(genofile).replace("\\", "/")
    base = os.path.basename(p)
    low = base.lower()
    if low.endswith(".vcf.gz"):
        stem = base[: -len(".vcf.gz")]
    elif low.endswith((".vcf", ".txt", ".tsv", ".csv", ".npy")):
        stem = os.path.splitext(base)[0]
    else:
        stem = base
    return os.path.join(os.path.dirname(p) or ".", f"~{stem}").replace("\\", "/")


def _detect_cache_need(genofile: str) -> tuple[bool, str, list[str]]:
    """
    Detect whether genotype cache build is expected before inspect/load.
    """
    low = str(genofile).lower()
    if low.endswith(".vcf.gz") or low.endswith(".vcf"):
        cprefix = _cache_prefix_tilde(genofile)
        targets = [f"{cprefix}.bed", f"{cprefix}.bim", f"{cprefix}.fam"]
        all_exist = all(os.path.isfile(p) for p in targets)
        stale = False
        if all_exist and os.path.isfile(genofile):
            src_mtime = os.path.getmtime(genofile)
            cache_mtime = min(os.path.getmtime(p) for p in targets)
            stale = cache_mtime < src_mtime
        return (not all_exist) or stale, "vcf", targets

    if low.endswith((".txt", ".tsv", ".csv")):
        cprefix = _cache_prefix_tilde(genofile)
        targets = [f"{cprefix}.npy"]
        return (not os.path.isfile(targets[0])), "txt", targets

    return False, "", []


def _format_cache_size(nbytes: int) -> str:
    x = float(max(0, int(nbytes)))
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while x >= 1024.0 and idx < len(units) - 1:
        x /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(x)}{units[idx]}"
    return f"{x:.1f}{units[idx]}"


def _load_phenotype_with_status(
    phenofile: str,
    ncol: Union[list[int], None],
    logger: logging.Logger,
    *,
    id_col: int = 0,
    use_spinner: bool = False,
) -> pd.DataFrame:
    """
    Load phenotype with rich/plain CLI status.
    """
    src = _basename_only(phenofile)
    with CliStatus(f"Loading phenotype from {src}...", enabled=bool(use_spinner)) as task:
        try:
            pheno = load_phenotype(
                phenofile,
                ncol,
                logger,
                id_col=id_col,
                use_spinner=use_spinner,
            )
        except Exception:
            task.fail(f"Loading phenotype from {src} ...Failed")
            raise
        task.complete(
            f"Loading phenotype from {src} (n={pheno.shape[0]}, npheno={pheno.shape[1]})"
        )
    return pheno


def _inspect_genotype_file_with_warnings(
    genofile: str,
) -> tuple[np.ndarray, int, list[str]]:
    """
    Run inspect_genotype_file and capture warning messages.
    Designed for subprocess execution to avoid GIL stalls during cache build.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ids0, ns0 = inspect_genotype_file(genofile)
    warn_msgs: list[str] = []
    for w in caught:
        try:
            msg = str(w.message).strip()
        except Exception:
            msg = ""
        if msg:
            warn_msgs.append(msg)
    return np.asarray(ids0, dtype=str), int(ns0), warn_msgs


def _inspect_genotype_with_status(
    genofile: str,
    logger: logging.Logger,
    *,
    use_spinner: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Inspect genotype metadata with optional cache-status spinner.
    """
    src = _basename_only(genofile)
    need_cache, cache_kind, cache_targets = _detect_cache_need(genofile)
    status_enabled = bool(use_spinner)
    plain_progress = (not status_enabled) and (cache_kind in {"vcf", "txt"})

    # For direct PLINK prefixes (no cache-target metadata), inspect directly.
    # For VCF/TXT sources, always run the threaded+monitored path below so that
    # cache rebuilds triggered inside inspect_genotype_file() are still visible.
    if cache_kind == "":
        with CliStatus(f"Loading genotype from {src}...", enabled=status_enabled) as task:
            try:
                ids, n_snps = inspect_genotype_file(genofile)
            except Exception:
                task.fail(f"Loading genotype from {src} ...Failed")
                raise
            task.complete(f"Loading genotype from {src} (n={len(ids)}, nSNP={n_snps})")
        return np.asarray(ids, dtype=str), int(n_snps)

    with CliStatus(f"Loading genotype from {src}...", enabled=status_enabled) as task:
        out: dict[str, tuple[np.ndarray, int]] = {}
        err: dict[str, Exception] = {}
        warn_msgs: list[str] = []
        last_plain_msg = ""
        last_plain_emit = 0.0
        plain_t0 = time.monotonic()
        last_wait_beat = plain_t0
        if plain_progress:
            logger.info(f"Loading genotype from {src}... [{format_elapsed(0.0)}]")
            last_plain_msg = f"Loading genotype from {src}..."
            last_plain_emit = plain_t0

        def _emit_plain(msg: str, *, allow_same_after: float = 5.0) -> None:
            nonlocal last_plain_msg, last_plain_emit
            if not plain_progress:
                return
            m = str(msg)
            now = time.monotonic()
            elapsed = format_elapsed(now - plain_t0)
            changed = (m != last_plain_msg)
            # Emit on change with a small throttle; also emit heartbeat when unchanged.
            if (
                (changed and (now - last_plain_emit >= 0.5))
                or ((not changed) and (now - last_plain_emit >= float(max(0.5, allow_same_after))))
            ):
                logger.info(f"{m} [{elapsed}]")
                last_plain_msg = m
                last_plain_emit = now

        # Use subprocess first so UI updates are not blocked by GIL while Rust builds cache.
        fut = None
        executor = None
        done_evt: Union[threading.Event, None] = None
        t: Union[threading.Thread, None] = None
        use_subproc = cache_kind in {"vcf", "txt"}
        if use_subproc:
            try:
                mp_ctx = mp.get_context("spawn")
                executor = cf.ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx)
                fut = executor.submit(_inspect_genotype_file_with_warnings, genofile)
            except Exception:
                use_subproc = False

        if not use_subproc:
            done_evt = threading.Event()

            def _worker() -> None:
                try:
                    ids0, ns0, warns0 = _inspect_genotype_file_with_warnings(genofile)
                    out["value"] = (np.asarray(ids0, dtype=str), int(ns0))
                    warn_msgs.extend(warns0)
                except Exception as ex:
                    err["value"] = ex
                finally:
                    done_evt.set()

            t = threading.Thread(target=_worker, daemon=True)
            t.start()

        bim_path = cache_targets[1] if (cache_kind == "vcf" and len(cache_targets) >= 2) else ""
        bed_path = cache_targets[0] if (cache_kind == "vcf" and len(cache_targets) >= 1) else ""
        npy_path = cache_targets[0] if (cache_kind == "txt" and len(cache_targets) >= 1) else ""
        bim_fp = None
        bim_dev_ino: tuple[int, int] = (-1, -1)
        bim_seen = 0
        bim_count = 0
        last_count = -1
        last_size = -1
        while True:
            if use_subproc:
                if fut is not None and fut.done():
                    break
                time.sleep(0.25)
            else:
                if done_evt is not None and done_evt.wait(timeout=0.25):
                    break
            if bim_path and os.path.isfile(bim_path):
                bim_progressed = False
                try:
                    st = os.stat(bim_path)
                    dev_ino = (int(st.st_dev), int(st.st_ino))
                    if (
                        bim_fp is None
                        or dev_ino != bim_dev_ino
                        or int(st.st_size) < int(bim_seen)
                    ):
                        if bim_fp is not None:
                            try:
                                bim_fp.close()
                            except Exception:
                                pass
                        bim_fp = open(bim_path, "rb")
                        bim_dev_ino = dev_ino
                        bim_seen = 0
                        bim_count = 0
                    if int(st.st_size) > int(bim_seen):
                        bim_fp.seek(int(bim_seen))
                        chunk = bim_fp.read(int(st.st_size) - int(bim_seen))
                        bim_seen = int(st.st_size)
                        bim_count += int(chunk.count(b"\n"))
                        bim_progressed = True
                        if bim_count != last_count:
                            last_count = bim_count
                            msg = f"Loading genotype from {src}... SNP={bim_count}"
                            task.desc = msg
                            _emit_plain(msg)
                except Exception:
                    pass
                if (not bim_progressed) and bed_path and os.path.isfile(bed_path):
                    # .bim may not flush frequently; keep progress alive via .bed growth.
                    try:
                        bsz = int(os.path.getsize(bed_path))
                        if bsz != last_size:
                            last_size = bsz
                            msg = f"Loading genotype from {src}... cache={_format_cache_size(bsz)}"
                            task.desc = msg
                            _emit_plain(msg)
                    except Exception:
                        pass
            elif bed_path and os.path.isfile(bed_path):
                try:
                    sz = int(os.path.getsize(bed_path))
                    if sz != last_size:
                        last_size = sz
                        msg = f"Loading genotype from {src}... cache={_format_cache_size(sz)}"
                        task.desc = msg
                        _emit_plain(msg)
                except Exception:
                    pass
            elif npy_path and os.path.isfile(npy_path):
                try:
                    sz = int(os.path.getsize(npy_path))
                    if sz != last_size:
                        last_size = sz
                        msg = f"Loading genotype from {src}... cache={_format_cache_size(sz)}"
                        task.desc = msg
                        _emit_plain(msg)
                except Exception:
                    pass
            now = time.monotonic()
            if now - last_wait_beat >= 3.0:
                dots = "." * (1 + (int((now - plain_t0) // 3) % 3))
                wait_msg = f"Loading genotype from {src}... waiting{dots}"
                task.desc = wait_msg
                _emit_plain(wait_msg, allow_same_after=3.0)
                last_wait_beat = now

        if use_subproc:
            try:
                if fut is not None:
                    ids0, ns0, warns0 = fut.result()
                    out["value"] = (np.asarray(ids0, dtype=str), int(ns0))
                    warn_msgs.extend(list(warns0))
            except Exception as ex:
                err["value"] = ex
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=False)
        elif t is not None:
            t.join()
        if bim_fp is not None:
            try:
                bim_fp.close()
            except Exception:
                pass
        if "value" in err:
            task.fail(f"Loading genotype from {src} ...Failed")
            raise err["value"]

        ids, n_snps = out["value"]
        for wmsg in warn_msgs:
            if use_spinner:
                print(f"Warning: {wmsg}", flush=True)
            else:
                logger.warning(wmsg)
        task.desc = f"Loading genotype from {src}... SNP={n_snps}"
        task.complete(f"Loading genotype from {src} (n={len(ids)}, nSNP={n_snps})")
        _log_info(logger, f"Cached genotype sites: {n_snps}", use_spinner=use_spinner)
        return ids, n_snps

def build_grm_streaming(
    genofile: str,
    n_samples: int,
    n_snps: int,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    method: int,
    mmap_window_mb: Union[int , None],
    threads: int,
    logger,
    use_spinner: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Build GRM in a streaming fashion using rust2py.gfreader.load_genotype_chunks.
    """
    _log_info(logger, f"Building GRM (streaming), method={method}", use_spinner=use_spinner)
    pbar = _ProgressAdapter(total=n_snps, desc="GRM (streaming)")
    process = psutil.Process()

    prefetch_depth = 2
    tuned_chunk_size = auto_stream_grm_chunk_size(
        n_samples=n_samples,
        requested_chunk_size=chunk_size,
        threads=threads,
        prefetch_depth=prefetch_depth,
    )
    if tuned_chunk_size != int(chunk_size):
        _log_file_only(
            logger,
            logging.INFO,
            (
                "Auto-tuned GRM chunk size: "
                f"{chunk_size} -> {tuned_chunk_size} "
                f"(n_samples={n_samples}, threads={threads})"
            ),
        )
    mem_tick_span = max(1, 10 * int(tuned_chunk_size))

    chunk_iter = load_genotype_chunks(
        genofile,
        tuned_chunk_size,
        maf_threshold,
        max_missing_rate,
        model="add",
        mmap_window_mb=mmap_window_mb,
    )

    def _on_grm_chunk(added_snps: int, total_eff: int) -> None:
        pbar.update(int(added_snps))
        if total_eff % mem_tick_span == 0:
            mem = process.memory_info().rss / 1024**3
            pbar.set_postfix(memory=f"{mem:.2f}GB")

    grm, grm_stats = build_streaming_grm_from_chunks(
        prefetch_iter(chunk_iter, in_flight=prefetch_depth),
        n_samples=n_samples,
        method=method,
        accumulate="gemm",
        on_chunk=_on_grm_chunk,
    )

    # force bar to 100% even if SNPs were filtered in Rust
    pbar.finish()
    pbar.close()

    _log_info(logger, "GRM construction finished.", use_spinner=use_spinner)
    return grm, int(grm_stats.eff_m)

_CJK_FONT_READY: Optional[bool] = None


def _contains_cjk(text: str) -> bool:
    for ch in str(text):
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x3000 <= code <= 0x303F
            or 0xFF00 <= code <= 0xFFEF
        ):
            return True
    return False


def _ensure_cjk_font() -> bool:
    """
    Ensure matplotlib has at least one CJK-capable sans font configured.
    Returns True when a candidate font is found, otherwise False.
    """
    global _CJK_FONT_READY
    if _CJK_FONT_READY is not None:
        return _CJK_FONT_READY

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "PingFang SC",
        "Heiti SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in mpl_font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), None)
    if selected is None:
        _CJK_FONT_READY = False
        return False

    current = mpl.rcParams.get("font.sans-serif", [])
    if not isinstance(current, list):
        current = [str(current)]
    mpl.rcParams["font.sans-serif"] = [selected] + [x for x in current if x != selected]
    mpl.rcParams["axes.unicode_minus"] = False
    _CJK_FONT_READY = True
    return True


def load_or_build_grm_with_cache(
    genofile: str,
    cache_prefix: str,
    mgrm: str,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    threads: int,
    mmap_limit: bool,
    logger:logging.Logger,
    use_spinner: bool = False,
) -> tuple[np.ndarray, int, Union[np.ndarray, None]]:
    """
    Load or build a GRM with caching for streaming LMM/LM runs.
    """
    ids, n_snps = inspect_genotype_file(genofile)
    n_samples = len(ids)
    method_is_builtin = mgrm in ["1", "2"]

    grm_ids = None
    if method_is_builtin:
        # GRM is always additive and shared by add/dom/rec/het scans.
        km_path = f"{cache_prefix}.k.{mgrm}"
        id_path = f"{km_path}.npy.id"
        cache_npy = f"{km_path}.npy"
        cache_is_stale = False
        if os.path.exists(cache_npy):
            g_mtime = latest_genotype_mtime(genofile)
            k_mtime = os.path.getmtime(cache_npy)
            if g_mtime is not None and g_mtime > k_mtime:
                cache_is_stale = True
                logger.warning(
                    "Warning: Genotype input is newer than cached GRM; rebuilding GRM cache."
                )
        if os.path.exists(cache_npy) and (not cache_is_stale):
            src = _basename_only(cache_npy)
            with CliStatus(f"Loading GRM from {src}...", enabled=bool(use_spinner)) as task:
                try:
                    grm = np.load(f'{km_path}.npy',mmap_mode='r')
                    grm = grm.reshape(n_samples, n_samples)
                    grm_ids = _read_id_file(
                        id_path,
                        logger,
                        "GRM",
                        use_spinner=use_spinner,
                        show_status=False,
                    )
                    if grm_ids is not None and len(grm_ids) != n_samples:
                        raise ValueError(
                            f"GRM ID count ({len(grm_ids)}) does not match GRM shape ({n_samples})."
                        )
                    if grm.ndim != 2 or grm.shape[0] != grm.shape[1]:
                        raise ValueError(f"GRM must be square; got shape={grm.shape}")
                    eff_m = n_snps  # approximate; exact effective M not critical here
                except Exception:
                    task.fail(f"Loading GRM from {src} ...Failed")
                    raise
                task.complete(f"Loading GRM from {src} (n={grm.shape[0]})")
        else:
            method_int = int(mgrm)
            grm, eff_m = build_grm_streaming(
                genofile=genofile,
                n_samples=n_samples,
                n_snps=n_snps,
                maf_threshold=maf_threshold,
                max_missing_rate=max_missing_rate,
                chunk_size=chunk_size,
                method=method_int,
                mmap_window_mb=auto_mmap_window_mb(
                    genofile, n_samples, n_snps, chunk_size
                ) if mmap_limit else None,
                threads=threads,
                logger=logger,
                use_spinner=use_spinner,
            )
            np.save(f'{km_path}.npy', grm)
            pd.Series(ids).to_csv(id_path, sep="\t", index=False, header=False)
            grm_ids = ids
            grm = np.load(f'{km_path}.npy',mmap_mode='r')
            if grm.ndim != 2 or grm.shape[0] != grm.shape[1]:
                raise ValueError(f"GRM must be square; got shape={grm.shape}")
            _log_file_only(logger, logging.INFO, f"Cached GRM written to {km_path}.npy")
    else:
        if not os.path.isfile(mgrm):
            msg = f"GRM file not found: {mgrm}"
            logger.error(msg)
            raise ValueError(msg)
        src = _basename_only(mgrm)
        with CliStatus(f"Loading GRM from {src}...", enabled=bool(use_spinner)) as task:
            try:
                if mgrm.endswith('.npy'):
                    grm = np.load(mgrm,mmap_mode='r')
                else:
                    grm = np.genfromtxt(mgrm, dtype="float32")
                grm_ids = _read_id_file(
                    f"{mgrm}.id",
                    logger,
                    "GRM",
                    use_spinner=use_spinner,
                    show_status=False,
                )
                if grm_ids is None:
                    if grm.size != n_samples * n_samples:
                        msg = f"GRM size mismatch: expected {n_samples*n_samples}, got {grm.size}"
                        logger.error(msg)
                        raise ValueError(msg)
                    grm = grm.reshape(n_samples, n_samples)
                else:
                    if grm.size != len(grm_ids) * len(grm_ids):
                        msg = (
                            f"GRM size mismatch: expected {len(grm_ids)*len(grm_ids)}, "
                            f"got {grm.size}"
                        )
                        logger.error(msg)
                        raise ValueError(msg)
                    grm = grm.reshape(len(grm_ids), len(grm_ids))
                if grm.ndim != 2 or grm.shape[0] != grm.shape[1]:
                    raise ValueError(f"GRM must be square; got shape={grm.shape}")
                eff_m = n_snps
            except Exception:
                task.fail(f"Loading GRM from {src} ...Failed")
                raise
            task.complete(f"Loading GRM from {src} (n={grm.shape[0]})")

    _log_info(logger, f"GRM shape: {grm.shape}", use_spinner=use_spinner)
    return grm, eff_m, grm_ids


def build_pcs_from_grm(
    grm: np.ndarray,
    dim: int,
    logger: logging.Logger,
    *,
    use_spinner: bool = False,
) -> np.ndarray:
    """
    Compute leading principal components from GRM.
    """
    if use_spinner:
        with CliStatus(f"Computing top {dim} PCs from GRM...", enabled=True) as task:
            try:
                _eigval, eigvec = np.linalg.eigh(grm)
                pcs = eigvec[:, -dim:]
            except Exception:
                task.fail(f"Computing top {dim} PCs from GRM ...Failed")
                raise
            task.complete(
                f"Computing top {dim} PCs from GRM (n={pcs.shape[0]}, nPC={pcs.shape[1]})"
            )
        _log_file_only(logger, logging.INFO, f"Computing top {dim} PCs from GRM...")
        _log_file_only(logger, logging.INFO, "PC computation finished.")
        return pcs

    logger.info(f"Computing top {dim} PCs from GRM...")
    _eigval, eigvec = np.linalg.eigh(grm)
    pcs = eigvec[:, -dim:]
    logger.info("PC computation finished.")
    return pcs


def load_or_build_q_with_cache(
    grm: Union[np.ndarray, None],
    n_samples: int,
    cache_prefix: str,
    pcdim: str,
    ids: np.ndarray,
    logger,
    use_spinner: bool = False,
) -> tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Load or build Q matrix (PCs) with caching for streaming LMM/LM.
    When loading from file, the first column is treated as sample IDs and
    the remaining columns are PCs.
    """
    n = int(n_samples)

    q_ids = None
    if pcdim in np.arange(1, n).astype(str):
        dim = int(pcdim)
        q_path = f"{cache_prefix}.q.{pcdim}.txt"
        if os.path.exists(q_path):
            src = _basename_only(q_path)
            with CliStatus(f"Loading Q matrix from {src}...", enabled=bool(use_spinner)) as task:
                try:
                    try:
                        q_ids, qmatrix = _read_matrix_with_ids(q_path, logger, "Q")
                    except Exception:
                        qmatrix = np.genfromtxt(q_path, dtype="float32")
                        q_ids = None
                        logger.warning("Q cache has no IDs; assuming genotype order.")
                except Exception:
                    task.fail(f"Loading Q matrix from {src} ...Failed")
                    raise
                task.complete(
                    f"Loading Q matrix from {src} (n={qmatrix.shape[0]}, nPC={qmatrix.shape[1]})"
                )
        else:
            if grm is None:
                raise ValueError(
                    "Q cache not found and GRM is unavailable; cannot generate PCA covariates."
                )
            qmatrix = build_pcs_from_grm(grm, dim, logger, use_spinner=use_spinner)
            df = pd.DataFrame(np.column_stack([ids.astype(str), qmatrix]))
            df.to_csv(q_path, sep="\t", header=False, index=False)
            q_ids = ids
            _log_file_only(logger, logging.INFO, f"Cached Q matrix written to {q_path}")
    elif pcdim == "0":
        _rich_success(
            logger,
            "PC dimension set to 0; using empty Q matrix.",
            use_spinner=use_spinner,
        )
        qmatrix = np.zeros((n, 0), dtype="float32")
        q_ids = ids
    elif os.path.isfile(pcdim):
        src = _basename_only(pcdim)
        with CliStatus(f"Loading Q matrix from {src}...", enabled=bool(use_spinner)) as task:
            try:
                q_ids, qmatrix = _read_matrix_with_ids(pcdim, logger, "Q")
            except Exception:
                task.fail(f"Loading Q matrix from {src} ...Failed")
                raise
            task.complete(
                f"Loading Q matrix from {src} (n={qmatrix.shape[0]}, nPC={qmatrix.shape[1]})"
            )
    else:
        raise ValueError(f"Unknown Q/PC option: {pcdim}")

    _log_info(logger, f"Q matrix shape: {qmatrix.shape}", use_spinner=use_spinner)
    return qmatrix, q_ids


def _load_covariate_for_streaming(
    cov_inputs: Union[str, list[str], None],
    genofile: str,
    sample_ids: np.ndarray,
    chunk_size: int,
    logger,
    use_spinner: bool = False,
) -> tuple[Union[np.ndarray , None], Union[np.ndarray, None]]:
    """
    Backward-compatible wrapper for streaming covariate loading.
    """
    return _load_covariates_for_models(
        cov_inputs=cov_inputs,
        genofile=genofile,
        sample_ids=np.asarray(sample_ids, dtype=str),
        chunk_size=int(chunk_size),
        logger=logger,
        context="streaming",
        use_spinner=use_spinner,
    )


def prepare_streaming_context(
    genofile: str,
    phenofile: str,
    pheno_cols: Union[list[int] , None],
    maf_threshold: float,
    max_missing_rate: float,
    genetic_model: str,
    het_threshold: float,
    chunk_size: int,
    mgrm: str,
    pcdim: str,
    cov_inputs: Union[str, list[str], None],
    threads: int,
    mmap_limit: bool,
    require_kinship: bool,
    logger,
    use_spinner: bool = False,
):
    """
    Prepare all shared resources for streaming LMM/LM once:
      - phenotype
      - genotype metadata (ids, n_snps)
      - GRM + Q (cached)
      - covariates (optional)
    """
    pheno = _load_phenotype_with_status(
        phenofile,
        pheno_cols,
        logger,
        id_col=0,
        use_spinner=use_spinner,
    )

    ids, n_snps = _inspect_genotype_with_status(
        genofile,
        logger,
        use_spinner=use_spinner,
    )
    n_samples = len(ids)
    _log_info(
        logger,
        f"Genotype meta: {n_samples} samples, {n_snps} SNPs.",
        use_spinner=use_spinner,
    )

    cache_prefix = genotype_cache_prefix(genofile)
    _log_info(
        logger,
        f"Cache prefix (genotype folder): {cache_prefix}",
        use_spinner=use_spinner,
    )
    need_generate_q = (
        pcdim in np.arange(1, n_samples).astype(str)
        and not os.path.exists(f"{cache_prefix}.q.{pcdim}.txt")
    )
    need_grm = bool(require_kinship or need_generate_q)

    grm: Union[np.ndarray, None] = None
    eff_m = n_snps
    grm_ids = None
    if need_grm:
        # GRM stream...
        grm, eff_m, grm_ids = load_or_build_grm_with_cache(
            genofile=genofile,
            cache_prefix=cache_prefix,
            mgrm=mgrm,
            maf_threshold=maf_threshold,
            max_missing_rate=max_missing_rate,
            chunk_size=chunk_size,
            threads=threads,
            mmap_limit=mmap_limit,
            logger=logger,
            use_spinner=use_spinner,
        )
    else:
        _rich_success(
            logger,
            "Skipping GRM construction: no LMM/fastLMM and no PCA generation needed.",
            use_spinner=use_spinner,
        )

    # PCA stream...
    qmatrix, q_ids = load_or_build_q_with_cache(
        grm=grm,
        n_samples=n_samples,
        cache_prefix=cache_prefix,
        pcdim=pcdim,
        ids=ids,
        logger=logger,
        use_spinner=use_spinner,
    )

    cov_all, cov_ids = _load_covariate_for_streaming(
        cov_inputs,
        genofile,
        ids,
        chunk_size,
        logger,
        use_spinner=use_spinner,
    )

    # -----------------------------------------
    # Align all data sources to shared IDs
    # -----------------------------------------
    geno_ids = ids.astype(str)
    pheno_ids = pheno.index.astype(str).to_numpy()

    # If optional ID files are missing, inherit genotype order
    if grm is not None:
        if grm_ids is None:
            if grm.shape[0] != n_samples:
                raise ValueError(
                    f"GRM size mismatch: {grm.shape[0]} != genotype samples {n_samples} "
                    "and no GRM ID file was provided."
                )
            logger.warning("GRM IDs not provided; assuming genotype order.")
            grm_ids = ids
        else:
            grm_ids = np.asarray(grm_ids, dtype=str)
    if q_ids is None:
        if qmatrix.shape[0] != n_samples:
            raise ValueError(
                f"Q matrix size mismatch: {qmatrix.shape[0]} != genotype samples {n_samples} "
                "and no Q ID file was provided."
            )
        logger.warning("Q IDs not provided; assuming genotype order.")
        q_ids = ids
    else:
        q_ids = np.asarray(q_ids, dtype=str)
    if cov_ids is None and cov_all is not None:
        if cov_all.shape[0] != n_samples:
            raise ValueError(
                f"Covariate size mismatch: {cov_all.shape[0]} != genotype samples {n_samples} "
                "and no covariate ID file was provided."
            )
        logger.warning("Covariate IDs not provided; assuming genotype order.")
        cov_ids = ids
    elif cov_ids is not None:
        cov_ids = np.asarray(cov_ids, dtype=str)

    common = set(geno_ids) & set(pheno_ids)
    if grm_ids is not None:
        common &= set(grm_ids.astype(str))
    if q_ids is not None:
        common &= set(q_ids.astype(str))
    if cov_ids is not None:
        common &= set(cov_ids.astype(str))

    common_ids = [i for i in geno_ids if i in common]
    if len(common_ids) == 0:
        # Try using IID (second column) for PLINK-style phenotype files
        try:
            pheno_alt = load_phenotype(
                phenofile,
                pheno_cols,
                logger,
                id_col=1,
                use_spinner=use_spinner,
            )
            pheno_ids_alt = pheno_alt.index.astype(str).to_numpy()
            common_alt = set(geno_ids) & set(pheno_ids_alt)
            if grm_ids is not None:
                common_alt &= set(grm_ids.astype(str))
            if q_ids is not None:
                common_alt &= set(q_ids.astype(str))
            if cov_ids is not None:
                common_alt &= set(cov_ids.astype(str))
            if len(common_alt) > 0:
                logger.warning("Using phenotype column 2 (IID) as sample IDs.")
                pheno = pheno_alt
                pheno_ids = pheno_ids_alt
                common = common_alt
        except Exception as e:
            logger.warning(f"Failed to parse phenotype with IID column: {e}")

    common_ids = [i for i in geno_ids if i in common]
    if len(common_ids) == 0:
        logger.error("No overlapping samples across genotype/phenotype/GRM/Q/cov.")
        logger.error(f"Genotype IDs (first 5): {list(geno_ids[:5])}")
        logger.error(f"Phenotype IDs (first 5): {list(pheno_ids[:5])}")
        if grm_ids is not None:
            logger.error(f"GRM IDs (first 5): {list(grm_ids[:5])}")
        if q_ids is not None:
            logger.error(f"Q IDs (first 5): {list(q_ids[:5])}")
        if cov_ids is not None:
            logger.error(f"Covariate IDs (first 5): {list(cov_ids[:5])}")
        raise ValueError("No overlapping samples across genotype/phenotype/GRM/Q/cov.")

    logger.info(
        f"geno={len(geno_ids)}, pheno={len(pheno_ids)}, "
        f"grm={'NA' if grm_ids is None else len(grm_ids)}, "
        f"q={'NA' if q_ids is None else len(q_ids)}, "
        f"cov={'NA' if cov_ids is None else len(cov_ids)} -> {len(common_ids)}"
    )

    # index maps
    geno_index = {sid: i for i, sid in enumerate(geno_ids)}
    if grm_ids is not None:
        grm_index = {sid: i for i, sid in enumerate(grm_ids.astype(str))}
    else:
        grm_index = geno_index
    if q_ids is not None:
        q_index = {sid: i for i, sid in enumerate(q_ids.astype(str))}
    else:
        q_index = geno_index
    if cov_ids is not None:
        cov_index = {sid: i for i, sid in enumerate(cov_ids.astype(str))}
    else:
        cov_index = geno_index

    # reorder/trim
    ids = np.array(common_ids)
    pheno = pheno.loc[ids]

    if grm is not None:
        grm_idx = [grm_index[sid] for sid in ids]
        grm = grm[np.ix_(grm_idx, grm_idx)]

    q_idx = [q_index[sid] for sid in ids]
    qmatrix = qmatrix[q_idx]

    if cov_all is not None:
        cov_idx = [cov_index[sid] for sid in ids]
        cov_all = cov_all[cov_idx]

    return pheno, ids, n_snps, grm, qmatrix, cov_all, eff_m


def run_chunked_gwas_lmm_lm(
    model_name: str,
    genofile: str,
    pheno: pd.DataFrame,
    ids: np.ndarray,
    n_snps: int,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    genetic_model: str,
    het_threshold: float,
    chunk_size: int,
    mmap_limit: bool,
    grm: Union[np.ndarray, None],
    qmatrix: np.ndarray,
    cov_all: Union[np.ndarray , None],
    eff_m: int,
    plot: bool,
    threads: int,
    logger:logging.Logger,
    use_spinner: bool = False,
    eff_snp_by_trait: Union[dict[str, int], None] = None,
    summary_rows: Union[list[dict[str, object]], None] = None,
    saved_paths: Union[list[str], None] = None,
    trait_names: Union[list[str], None] = None,
    show_npve_line: bool = False,
) -> None:
    """
    Run LMM or LM GWAS using a streaming, low-memory pipeline.

    Important: This function assumes pheno/ids/grm/q/cov have already been prepared
    once (no repeated "Loading phenotype" / "Loading GRM/Q" logs).
    """
    model_map = {"lmm": LMM, "lm": LM, "fastlmm": FastLMM}
    model_key = model_name.lower()
    ModelCls = model_map[model_key]
    model_label = {"lmm": "LMM", "lm": "LM", "fastlmm": "FastLMM"}[model_key]
    # Keep output file suffixes consistent and lowercase.
    model_tag = model_label.lower()

    def _apply_genetic_model(geno_chunk: np.ndarray, model: str) -> np.ndarray:
        m = model.lower()
        if m == "add":
            return geno_chunk
        if m == "dom":
            return (
                np.isclose(geno_chunk, 1.0, atol=1e-6)
                | np.isclose(geno_chunk, 2.0, atol=1e-6)
            ).astype(np.float32, copy=False)
        if m == "rec":
            return np.isclose(geno_chunk, 2.0, atol=1e-6).astype(np.float32, copy=False)
        if m == "het":
            return np.isclose(geno_chunk, 1.0, atol=1e-6).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported genetic model: {model}")

    def _transform_allele_labels(
        allele0_list: list[str], allele1_list: list[str], model: str
    ) -> tuple[list[str], list[str]]:
        m = model.lower()
        if m == "add":
            return allele0_list, allele1_list
        out0: list[str] = []
        out1: list[str] = []
        for a0, a1 in zip(allele0_list, allele1_list):
            hom0 = f"{a0}{a0}"
            het = f"{a0}{a1}"
            hom1 = f"{a1}{a1}"
            if m == "dom":
                out0.append(hom0)
                out1.append(f"{het}/{hom1}")
            elif m == "rec":
                out0.append(f"{het}/{hom0}")
                out1.append(hom1)
            elif m == "het":
                out0.append(f"{hom0}/{hom1}")
                out1.append(het)
            else:
                raise ValueError(f"Unsupported genetic model: {model}")
        return out0, out1

    def _heter_keep_mask(geno_chunk: np.ndarray, het: float) -> np.ndarray:
        valid = geno_chunk >= 0
        non_missing = np.sum(valid, axis=1)
        keep = non_missing > 0
        if not np.any(keep):
            return keep
        het_count = np.sum(np.isclose(geno_chunk, 1.0, atol=1e-6) & valid, axis=1)
        het_rate = np.zeros(geno_chunk.shape[0], dtype=np.float32)
        idx = non_missing > 0
        het_rate[idx] = het_count[idx] / non_missing[idx]
        keep &= (het_rate >= het) & (het_rate <= (1.0 - het))
        return keep

    process = psutil.Process()
    n_cores = psutil.cpu_count(logical=True) or cpu_count()

    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    trait_iter = list(pheno.columns) if trait_names is None else [t for t in trait_names if t in pheno.columns]
    multi_trait_mode = len(trait_iter) > 1
    for trait_idx, pname in enumerate(trait_iter):
        cpu_t0 = process.cpu_times()
        rss0 = process.memory_info().rss
        t0 = time.time()
        peak_rss = rss0
        evd_secs = 0.0

        pheno_sub = pheno[pname].dropna()
        sameidx = np.isin(ids, pheno_sub.index)
        n_idv = int(np.sum(sameidx))
        if n_idv == 0:
            logger.info(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")  # single blank line between traits
            continue

        trait_ids = np.asarray(ids[sameidx], dtype=str)
        y_vec = pheno_sub.loc[trait_ids].values
        # Build covariate matrix X_cov for this trait
        X_cov = qmatrix[sameidx]
        if cov_all is not None:
            X_cov = np.concatenate([X_cov, cov_all[sameidx]], axis=1)

        header_pve: Optional[float] = None
        if model_key in ("lmm", "fastlmm"):
            if grm is None:
                raise ValueError("LMM/fastLMM requires GRM, but GRM was not prepared.")
            Ksub = grm[np.ix_(sameidx, sameidx)]
            evd_t0 = time.monotonic()
            with CliStatus("Eigen-Decomposition...", enabled=bool(use_spinner)):
                mod = ModelCls(y=y_vec, X=X_cov, kinship=Ksub)
            try:
                pve_tmp = float(mod.pve)
                if np.isfinite(pve_tmp):
                    header_pve = pve_tmp
            except Exception:
                header_pve = None
            evd_secs = time.monotonic() - evd_t0
            evd_elapsed = format_elapsed(evd_secs)
            _log_file_only(
                logger,
                logging.INFO,
                f"{model_label}, trait: {pname}, PVE(null): {mod.pve:.3f}",
            )
            _log_file_only(
                logger,
                logging.INFO,
                f"Eigen-Decomposition ...Finished [{evd_elapsed}]",
            )
        else:
            mod = ModelCls(y=y_vec, X=X_cov)
            _log_file_only(logger, logging.INFO, f"{model_label}, trait: {pname}")

        _emit_trait_header(
            logger,
            pname,
            n_idv,
            pve=header_pve,
            use_spinner=bool(use_spinner),
            width=60,
        )

        done_snps = 0
        has_results = False
        out_tsv = f"{outprefix}.{pname}.{genetic_model}.{model_tag}.tsv"
        tmp_tsv = f"{out_tsv}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        wrote_header = False
        mmap_window_mb = (
            auto_mmap_window_mb(genofile, len(ids), n_snps, chunk_size)
            if mmap_limit else None
        )

        # Always pass trait-specific sample IDs to the reader to keep column
        # dimension consistent across BED/VCF/TXT backends.
        sample_sub = trait_ids
        expected_n = int(sample_sub.shape[0])
        scan_threads = int(threads)
        if scan_threads <= 0:
            scan_threads = int(n_cores)
        max_inflight = 2 if scan_threads >= 2 else 1
        workers = max(1, max_inflight)
        threads_per_worker = max(1, scan_threads // workers)

        process.cpu_percent(interval=None)
        scan_t0 = time.time()
        pbar_total = int(eff_snp_by_trait.get(pname, n_snps))
        pbar_desc = f"{model_label}"
        pbar = _ProgressAdapter(total=pbar_total, desc=pbar_desc)

        inflight: dict[
            int,
            tuple[cf.Future, int, list[tuple[str, int, str, str]], np.ndarray],
        ] = {}
        ready_rows: dict[int, tuple[pd.DataFrame, int]] = {}
        chunk_seq = 0
        next_write_seq = 0

        def _build_chunk_df(
            results: np.ndarray,
            info_chunk: list[tuple[str, int, str, str]],
            maf_chunk: np.ndarray,
        ) -> pd.DataFrame:
            chroms, poss, allele0, allele1 = zip(*info_chunk)
            allele0_list = list(allele0)
            allele1_list = list(allele1)
            if genetic_model != "add":
                allele0_list, allele1_list = _transform_allele_labels(
                    allele0_list, allele1_list, genetic_model
                )
            chunk_df = pd.DataFrame(
                {
                    "chrom": chroms,
                    "pos": poss,
                    "allele0": allele0_list,
                    "allele1": allele1_list,
                    "maf": maf_chunk,
                    "beta": results[:, 0],
                    "se": results[:, 1],
                    "pwald": results[:, 2],
                }
            )
            if results.shape[1] > 3:
                chunk_df["plrt"] = results[:, 3]
                chunk_df["plrt"] = chunk_df["plrt"].map(lambda x: f"{x:.4e}")
            chunk_df["pos"] = chunk_df["pos"].astype(int)
            chunk_df["pwald"] = chunk_df["pwald"].map(lambda x: f"{x:.4e}")
            return chunk_df

        def _write_ready_chunks() -> None:
            nonlocal next_write_seq, wrote_header, has_results, done_snps, peak_rss
            while next_write_seq in ready_rows:
                chunk_df, m_chunk = ready_rows.pop(next_write_seq)
                chunk_df.to_csv(
                    tmp_tsv,
                    sep="\t",
                    float_format="%.4f",
                    index=False,
                    header=not wrote_header,
                    mode="w" if not wrote_header else "a",
                )
                wrote_header = True
                has_results = True

                done_snps += m_chunk
                pbar.update(m_chunk)

                mem_info = process.memory_info()
                peak_rss = max(peak_rss, mem_info.rss)
                if done_snps % (10 * chunk_size) == 0:
                    mem_gb = mem_info.rss / 1024**3
                    pbar.set_postfix(memory=f"{mem_gb:.2f}GB")
                next_write_seq += 1

        def _drain_completed(*, wait_for_one: bool) -> None:
            if len(inflight) == 0:
                return
            futures = [x[0] for x in inflight.values()]
            if wait_for_one:
                done_set, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            else:
                done_set = {f for f in futures if f.done()}
                if len(done_set) == 0:
                    return

            done_seq = sorted(
                [
                    seq
                    for seq, (fut, _m, _info, _maf) in inflight.items()
                    if fut in done_set
                ]
            )
            for seq in done_seq:
                fut, m_chunk, info_chunk, maf_chunk = inflight.pop(seq)
                results = fut.result()
                ready_rows[seq] = (_build_chunk_df(results, info_chunk, maf_chunk), m_chunk)
            _write_ready_chunks()

        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for genosub, sites in load_genotype_chunks(
                genofile,
                chunk_size,
                maf_threshold,
                max_missing_rate,
                model=genetic_model,
                het=het_threshold,
                sample_ids=sample_sub,
                mmap_window_mb=mmap_window_mb,
            ):
                genosub: np.ndarray
                if genosub.shape[1] != expected_n:
                    # Backward-compatible fallback: if backend ignored sample_ids and
                    # returned columns for full aligned IDs, apply sameidx here.
                    if genosub.shape[1] == int(sameidx.shape[0]):
                        genosub = genosub[:, sameidx]
                    else:
                        raise ValueError(
                            f"Genotype sample dimension mismatch for trait {pname}: "
                            f"chunk has {genosub.shape[1]} columns, "
                            f"expected {expected_n} (or {sameidx.shape[0]} full aligned IDs)."
                        )
                if genetic_model != "add":
                    keep_mask = _heter_keep_mask(genosub, het_threshold)
                    if not np.any(keep_mask):
                        continue
                    genosub = genosub[keep_mask]
                    sites = [s for s, k in zip(sites, keep_mask) if k]
                m_chunk = genosub.shape[0]
                if m_chunk == 0:
                    continue

                info_chunk = [
                    (str(s.chrom), int(s.pos), str(s.ref_allele), str(s.alt_allele))
                    for s in sites
                ]
                if len(info_chunk) == 0:
                    continue

                geno_model = _apply_genetic_model(genosub, genetic_model)
                if genetic_model == "add":
                    maf_chunk = np.mean(genosub, axis=1)
                    maf_chunk = (maf_chunk / 2).astype(np.float32, copy=False)
                else:
                    maf_chunk = np.mean(geno_model, axis=1).astype(np.float32, copy=False)
                geno_center = geno_model - np.mean(
                    geno_model, axis=1, dtype=np.float32, keepdims=True
                )

                fut = ex.submit(mod.gwas, geno_center, threads=threads_per_worker)
                inflight[chunk_seq] = (fut, int(m_chunk), info_chunk, maf_chunk)
                chunk_seq += 1

                if len(inflight) >= max_inflight:
                    _drain_completed(wait_for_one=True)
                else:
                    _drain_completed(wait_for_one=False)

            while len(inflight) > 0:
                _drain_completed(wait_for_one=True)
            _write_ready_chunks()

        pbar.finish()
        pbar.close(show_done=False)

        cpu_t1 = process.cpu_times()
        t1 = time.time()
        scan_secs = max(t1 - scan_t0, 0.0)

        wall = t1 - t0
        user_cpu = cpu_t1.user - cpu_t0.user
        sys_cpu = cpu_t1.system - cpu_t0.system
        total_cpu = user_cpu + sys_cpu

        avg_cpu_pct = 100.0 * total_cpu / wall / (n_cores or 1) if wall > 0 else 0.0
        peak_rss_gb = peak_rss / 1024**3
        _log_file_only(
            logger,
            logging.INFO,
            f"avg CPU ~ {avg_cpu_pct:.1f}% of {n_cores} c, "
            f"peak RSS ~ {peak_rss_gb:.2f} G",
        )

        if not has_results:
            logger.info(f"No SNPs passed filters for trait {pname}.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = int(done_snps)
            summary_rows.append(
                {
                    "phenotype": str(pname),
                    "model": model_label,
                    "nidv": int(n_idv),
                    "eff_snp": int(done_snps),
                    "pve": (float(header_pve) if header_pve is not None else None),
                    "avg_cpu": float(avg_cpu_pct),
                    "peak_rss_gb": float(peak_rss_gb),
                    "gwas_time_s": float(evd_secs + scan_secs),
                    "viz_time_s": 0.0,
                }
            )
            if os.path.exists(tmp_tsv):
                os.remove(tmp_tsv)
            if multi_trait_mode:
                logger.info("")  # single blank line between traits
            continue

        viz_secs = 0.0
        if plot:
            viz_t0 = time.time()
            def _run_plot() -> None:
                plot_df = pd.read_csv(
                    tmp_tsv,
                    sep="\t",
                    usecols=["chrom", "pos", "pwald"],
                    dtype={"chrom": str, "pos": "int64"},
                )
                plot_df["pwald"] = pd.to_numeric(plot_df["pwald"], errors="coerce")
                fastplot(
                    plot_df,
                    y_vec,
                    xlabel=pname,
                    outpdf=f"{outprefix}.{pname}.{genetic_model}.{model_tag}.svg",
                )

            _run_plot()
            viz_secs = max(time.time() - viz_t0, 0.0)

        if pname not in eff_snp_by_trait:
            eff_snp_by_trait[pname] = int(done_snps)
        summary_rows.append(
            {
                "phenotype": str(pname),
                "model": model_label,
                "nidv": int(n_idv),
                "eff_snp": int(done_snps),
                "pve": (float(header_pve) if header_pve is not None else None),
                "avg_cpu": float(avg_cpu_pct),
                "peak_rss_gb": float(peak_rss_gb),
                "gwas_time_s": float(evd_secs + scan_secs),
                "viz_time_s": float(viz_secs),
            }
        )

        os.replace(tmp_tsv, out_tsv)
        saved_paths.append(str(out_tsv).replace("//", "/"))
        _log_file_only(
            logger,
            logging.INFO,
            f"Results saved to {str(out_tsv).replace('//', '/')}",
        )
        time_parts: list[str] = []
        if evd_secs > 0:
            time_parts.append(format_elapsed(evd_secs))
        time_parts.append(format_elapsed(scan_secs))
        if plot:
            time_parts.append(format_elapsed(viz_secs))
        done_msg = f"{model_label} ...Finished [{'/'.join(time_parts)}]"
        if use_spinner:
            print_success(done_msg)
        else:
            logger.info(done_msg)
        if multi_trait_mode:
            logger.info("")  # ensure blank line between traits


def run_chunked_gwas_streaming_shared(
    model_names: list[str],
    trait_name: str,
    genofile: str,
    pheno: pd.DataFrame,
    ids: np.ndarray,
    n_snps: int,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    genetic_model: str,
    het_threshold: float,
    chunk_size: int,
    mmap_limit: bool,
    grm: Union[np.ndarray, None],
    qmatrix: np.ndarray,
    cov_all: Union[np.ndarray, None],
    plot: bool,
    threads: int,
    logger: logging.Logger,
    use_spinner: bool = False,
    eff_snp_by_trait: Union[dict[str, int], None] = None,
    summary_rows: Union[list[dict[str, object]], None] = None,
    saved_paths: Union[list[str], None] = None,
) -> None:
    """
    Shared-chunk streaming GWAS for multiple models on one trait.

    Decode/filter each chunk once, then run all selected streaming models on the
    same chunk before moving to the next chunk.
    """
    model_order = [str(m).lower() for m in model_names]
    model_map = {"lmm": LMM, "lm": LM, "fastlmm": FastLMM}
    model_order = [m for m in model_order if m in model_map]
    if len(model_order) == 0:
        return

    process = psutil.Process()
    n_cores = psutil.cpu_count(logical=True) or cpu_count()

    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pname = str(trait_name)
    pheno_sub = pheno[pname].dropna()
    sameidx = np.isin(ids, pheno_sub.index)
    n_idv = int(np.sum(sameidx))
    if n_idv == 0:
        logger.info(f"{pname}: no overlapping samples, skipped.")
        if pname not in eff_snp_by_trait:
            eff_snp_by_trait[pname] = 0
        return

    trait_ids = np.asarray(ids[sameidx], dtype=str)
    y_vec = pheno_sub.loc[trait_ids].values
    X_cov = qmatrix[sameidx]
    if cov_all is not None:
        X_cov = np.concatenate([X_cov, cov_all[sameidx]], axis=1)

    def _apply_genetic_model(geno_chunk: np.ndarray, model: str) -> np.ndarray:
        m = model.lower()
        if m == "add":
            return geno_chunk
        if m == "dom":
            return (
                np.isclose(geno_chunk, 1.0, atol=1e-6)
                | np.isclose(geno_chunk, 2.0, atol=1e-6)
            ).astype(np.float32, copy=False)
        if m == "rec":
            return np.isclose(geno_chunk, 2.0, atol=1e-6).astype(np.float32, copy=False)
        if m == "het":
            return np.isclose(geno_chunk, 1.0, atol=1e-6).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported genetic model: {model}")

    def _transform_allele_labels(
        allele0_list: list[str], allele1_list: list[str], model: str
    ) -> tuple[list[str], list[str]]:
        m = model.lower()
        if m == "add":
            return allele0_list, allele1_list
        out0: list[str] = []
        out1: list[str] = []
        for a0, a1 in zip(allele0_list, allele1_list):
            hom0 = f"{a0}{a0}"
            het = f"{a0}{a1}"
            hom1 = f"{a1}{a1}"
            if m == "dom":
                out0.append(hom0)
                out1.append(f"{het}/{hom1}")
            elif m == "rec":
                out0.append(f"{het}/{hom0}")
                out1.append(hom1)
            elif m == "het":
                out0.append(f"{hom0}/{hom1}")
                out1.append(het)
            else:
                raise ValueError(f"Unsupported genetic model: {model}")
        return out0, out1

    def _heter_keep_mask(geno_chunk: np.ndarray, het: float) -> np.ndarray:
        valid = geno_chunk >= 0
        non_missing = np.sum(valid, axis=1)
        keep = non_missing > 0
        if not np.any(keep):
            return keep
        het_count = np.sum(np.isclose(geno_chunk, 1.0, atol=1e-6) & valid, axis=1)
        het_rate = np.zeros(geno_chunk.shape[0], dtype=np.float32)
        idx = non_missing > 0
        het_rate[idx] = het_count[idx] / non_missing[idx]
        keep &= (het_rate >= het) & (het_rate <= (1.0 - het))
        return keep

    def _build_chunk_df(
        results: np.ndarray,
        info_chunk: list[tuple[str, int, str, str]],
        maf_chunk: np.ndarray,
    ) -> pd.DataFrame:
        chroms, poss, allele0, allele1 = zip(*info_chunk)
        allele0_list = list(allele0)
        allele1_list = list(allele1)
        if genetic_model != "add":
            allele0_list, allele1_list = _transform_allele_labels(
                allele0_list, allele1_list, genetic_model
            )
        chunk_df = pd.DataFrame(
            {
                "chrom": chroms,
                "pos": poss,
                "allele0": allele0_list,
                "allele1": allele1_list,
                "maf": maf_chunk,
                "beta": results[:, 0],
                "se": results[:, 1],
                "pwald": results[:, 2],
            }
        )
        if results.shape[1] > 3:
            chunk_df["plrt"] = results[:, 3]
            chunk_df["plrt"] = chunk_df["plrt"].map(lambda x: f"{x:.4e}")
        chunk_df["pos"] = chunk_df["pos"].astype(int)
        chunk_df["pwald"] = chunk_df["pwald"].map(lambda x: f"{x:.4e}")
        return chunk_df

    model_label_map = {"lmm": "LMM", "lm": "LM", "fastlmm": "FastLMM"}
    model_ctxs: list[dict[str, object]] = []
    for mkey in model_order:
        ModelCls = model_map[mkey]
        model_label = model_label_map[mkey]
        model_tag = model_label.lower()
        ctx: dict[str, object] = {
            "model_key": mkey,
            "model_label": model_label,
            "model_tag": model_tag,
            "mod": None,
            "evd_secs": 0.0,
            "scan_secs": 0.0,
            "cpu_used": 0.0,
            "peak_rss": int(process.memory_info().rss),
            "done_snps": 0,
            "wrote_header": False,
            "has_results": False,
            "tick": 0,
            "memory_text": "",
            "memory_until_tick": 0,
            "pbar": None,
            "task_id": None,
            "tmp_tsv": f"{outprefix}.{pname}.{genetic_model}.{model_tag}.tsv.tmp.{os.getpid()}.{uuid.uuid4().hex}",
            "out_tsv": f"{outprefix}.{pname}.{genetic_model}.{model_tag}.tsv",
        }

        cpu_before = process.cpu_times()
        init_t0 = time.monotonic()
        if mkey in {"lmm", "fastlmm"}:
            if grm is None:
                raise ValueError("LMM/fastLMM requires GRM, but GRM was not prepared.")
            Ksub = grm[np.ix_(sameidx, sameidx)]
            with CliStatus("Eigen-Decomposition...", enabled=bool(use_spinner)):
                mod = ModelCls(y=y_vec, X=X_cov, kinship=Ksub)
            evd_secs = max(time.monotonic() - init_t0, 0.0)
            evd_elapsed = format_elapsed(evd_secs)
            _log_file_only(
                logger,
                logging.INFO,
                f"{model_label}, trait: {pname}, PVE(null): {mod.pve:.3f}",
            )
            _log_file_only(
                logger,
                logging.INFO,
                f"Eigen-Decomposition ...Finished [{evd_elapsed}]",
            )
            ctx["evd_secs"] = float(evd_secs)
        else:
            mod = ModelCls(y=y_vec, X=X_cov)
            _log_file_only(logger, logging.INFO, f"{model_label}, trait: {pname}")

        cpu_after = process.cpu_times()
        ctx["cpu_used"] = float(
            (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
        )
        ctx["mod"] = mod
        model_ctxs.append(ctx)

    header_pve: Optional[float] = None
    for ctx in model_ctxs:
        if str(ctx.get("model_key", "")) in {"lmm", "fastlmm"}:
            mod_obj = ctx.get("mod")
            if mod_obj is not None and hasattr(mod_obj, "pve"):
                try:
                    pve_tmp = float(getattr(mod_obj, "pve"))
                    if np.isfinite(pve_tmp):
                        header_pve = pve_tmp
                        break
                except Exception:
                    pass

    _emit_trait_header(
        logger,
        pname,
        n_idv,
        pve=header_pve,
        use_spinner=bool(use_spinner),
        width=60,
    )

    pbar_total = int(eff_snp_by_trait.get(pname, n_snps))
    use_rich_multi = bool(use_spinner and _HAS_RICH_PROGRESS and sys.stdout.isatty())
    rich_progress = None
    if use_rich_multi:
        rich_progress = Progress(
            SpinnerColumn(
                spinner_name=get_rich_spinner_name(),
                style="cyan",
                finished_text="[green]✔︎[/green]",
            ),
            TextColumn("[green]{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[metric]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        )
        rich_progress.start()
        for ctx in model_ctxs:
            tid = rich_progress.add_task(
                str(ctx["model_label"]),
                total=pbar_total,
                metric="0.0%",
            )
            ctx["task_id"] = int(tid)
    else:
        for ctx in model_ctxs:
            ctx["pbar"] = _ProgressAdapter(total=pbar_total, desc=str(ctx["model_label"]))

    def _metric_text(ctx: dict[str, object]) -> str:
        tick = int(ctx.get("tick", 0))
        mem_until = int(ctx.get("memory_until_tick", 0))
        mem_text = str(ctx.get("memory_text", ""))
        if mem_text and tick <= mem_until:
            return mem_text
        if pbar_total <= 0:
            pct = 0.0
        else:
            pct = 100.0 * float(int(ctx.get("done_snps", 0))) / float(pbar_total)
        return f"{pct:>6.1f}%"

    def _advance_ctx(ctx: dict[str, object], m_chunk: int, mem_text: Union[str, None]) -> None:
        done = int(ctx.get("done_snps", 0)) + int(m_chunk)
        tick = int(ctx.get("tick", 0)) + 1
        ctx["done_snps"] = done
        ctx["tick"] = tick
        if mem_text is not None:
            ctx["memory_text"] = str(mem_text).replace(" ", "")
            ctx["memory_until_tick"] = tick + 5
        metric = _metric_text(ctx)
        if use_rich_multi and rich_progress is not None:
            rich_progress.update(int(ctx["task_id"]), advance=int(m_chunk), metric=metric)
        else:
            pbar_obj = ctx.get("pbar")
            if pbar_obj is not None:
                pbar_obj.update(int(m_chunk))
                if mem_text is not None:
                    pbar_obj.set_postfix(memory=str(mem_text))

    mmap_window_mb = (
        auto_mmap_window_mb(genofile, len(ids), n_snps, chunk_size)
        if mmap_limit else None
    )
    sample_sub = trait_ids
    expected_n = int(sample_sub.shape[0])

    scan_threads = int(threads)
    if scan_threads <= 0:
        scan_threads = int(n_cores)
    threads_per_model = max(1, scan_threads)

    for genosub, sites in load_genotype_chunks(
        genofile,
        chunk_size,
        maf_threshold,
        max_missing_rate,
        model=genetic_model,
        het=het_threshold,
        sample_ids=sample_sub,
        mmap_window_mb=mmap_window_mb,
    ):
        genosub: np.ndarray
        if genosub.shape[1] != expected_n:
            if genosub.shape[1] == int(sameidx.shape[0]):
                genosub = genosub[:, sameidx]
            else:
                raise ValueError(
                    f"Genotype sample dimension mismatch for trait {pname}: "
                    f"chunk has {genosub.shape[1]} columns, "
                    f"expected {expected_n} (or {sameidx.shape[0]} full aligned IDs)."
                )
        if genetic_model != "add":
            keep_mask = _heter_keep_mask(genosub, het_threshold)
            if not np.any(keep_mask):
                continue
            genosub = genosub[keep_mask]
            sites = [s for s, k in zip(sites, keep_mask) if k]
        m_chunk = int(genosub.shape[0])
        if m_chunk == 0:
            continue

        info_chunk = [
            (str(s.chrom), int(s.pos), str(s.ref_allele), str(s.alt_allele))
            for s in sites
        ]
        if len(info_chunk) == 0:
            continue

        geno_model = _apply_genetic_model(genosub, genetic_model)
        if genetic_model == "add":
            maf_chunk = (np.mean(genosub, axis=1) / 2).astype(np.float32, copy=False)
        else:
            maf_chunk = np.mean(geno_model, axis=1).astype(np.float32, copy=False)
        geno_center = geno_model - np.mean(
            geno_model, axis=1, dtype=np.float32, keepdims=True
        )

        for ctx in model_ctxs:
            cpu_before = process.cpu_times()
            t0 = time.monotonic()
            results = ctx["mod"].gwas(geno_center, threads=threads_per_model)
            elapsed = max(time.monotonic() - t0, 0.0)
            cpu_after = process.cpu_times()
            ctx["scan_secs"] = float(ctx["scan_secs"]) + float(elapsed)
            ctx["cpu_used"] = float(ctx["cpu_used"]) + float(
                (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
            )

            chunk_df = _build_chunk_df(results, info_chunk, maf_chunk)
            tmp_tsv = str(ctx["tmp_tsv"])
            wrote_header = bool(ctx["wrote_header"])
            chunk_df.to_csv(
                tmp_tsv,
                sep="\t",
                float_format="%.4f",
                index=False,
                header=not wrote_header,
                mode="w" if not wrote_header else "a",
            )
            ctx["wrote_header"] = True
            ctx["has_results"] = True

            mem_info = process.memory_info()
            ctx["peak_rss"] = max(int(ctx["peak_rss"]), int(mem_info.rss))
            done_next = int(ctx["done_snps"]) + m_chunk
            mem_text = None
            if done_next % (10 * chunk_size) == 0:
                mem_text = f"{mem_info.rss / 1024**3:.2f}GB"
            _advance_ctx(ctx, m_chunk, mem_text)

    first_done = 0
    for ctx in model_ctxs:
        first_done = first_done or int(ctx.get("done_snps", 0))
        if use_rich_multi and rich_progress is not None:
            rich_progress.update(
                int(ctx["task_id"]),
                completed=pbar_total,
                metric=_metric_text(ctx),
            )
        else:
            pbar_obj = ctx.get("pbar")
            if pbar_obj is not None:
                pbar_obj.finish()
                pbar_obj.close(show_done=False)
    if rich_progress is not None:
        rich_progress.stop()

    if pname not in eff_snp_by_trait:
        eff_snp_by_trait[pname] = int(first_done)

    for ctx in model_ctxs:
        model_label = str(ctx["model_label"])
        done_snps = int(ctx["done_snps"])
        evd_secs = float(ctx["evd_secs"])
        scan_secs = float(ctx["scan_secs"])
        cpu_used = float(ctx["cpu_used"])
        peak_rss_gb = float(int(ctx["peak_rss"]) / 1024**3)
        denom = max(evd_secs + scan_secs, 1e-9)
        avg_cpu_pct = 100.0 * cpu_used / denom / max(1, int(n_cores))
        _log_file_only(
            logger,
            logging.INFO,
            f"avg CPU ~ {avg_cpu_pct:.1f}% of {n_cores} c, "
            f"peak RSS ~ {peak_rss_gb:.2f} G",
        )

        has_results = bool(ctx["has_results"])
        tmp_tsv = str(ctx["tmp_tsv"])
        out_tsv = str(ctx["out_tsv"])
        ctx_pve: Optional[float] = None
        if str(ctx.get("model_key", "")) in {"lmm", "fastlmm"}:
            mod_obj = ctx.get("mod")
            if mod_obj is not None and hasattr(mod_obj, "pve"):
                try:
                    pve_tmp = float(getattr(mod_obj, "pve"))
                    if np.isfinite(pve_tmp):
                        ctx_pve = pve_tmp
                except Exception:
                    ctx_pve = None
        if not has_results:
            logger.info(f"No SNPs passed filters for trait {pname} ({model_label}).")
            summary_rows.append(
                {
                    "phenotype": str(pname),
                    "model": model_label,
                    "nidv": int(n_idv),
                    "eff_snp": int(done_snps),
                    "pve": (float(ctx_pve) if ctx_pve is not None else None),
                    "avg_cpu": float(avg_cpu_pct),
                    "peak_rss_gb": float(peak_rss_gb),
                    "gwas_time_s": float(evd_secs + scan_secs),
                    "viz_time_s": 0.0,
                }
            )
            if os.path.exists(tmp_tsv):
                os.remove(tmp_tsv)
        viz_secs = 0.0
        if has_results:
            if plot:
                viz_t0 = time.time()

                def _run_plot() -> None:
                    plot_df = pd.read_csv(
                        tmp_tsv,
                        sep="\t",
                        usecols=["chrom", "pos", "pwald"],
                        dtype={"chrom": str, "pos": "int64"},
                    )
                    plot_df["pwald"] = pd.to_numeric(plot_df["pwald"], errors="coerce")
                    fastplot(
                        plot_df,
                        y_vec,
                        xlabel=pname,
                        outpdf=f"{outprefix}.{pname}.{genetic_model}.{str(ctx['model_tag'])}.svg",
                    )

                _run_plot()
                viz_secs = max(time.time() - viz_t0, 0.0)

            summary_rows.append(
                {
                    "phenotype": str(pname),
                    "model": model_label,
                    "nidv": int(n_idv),
                    "eff_snp": int(done_snps),
                    "pve": (float(ctx_pve) if ctx_pve is not None else None),
                    "avg_cpu": float(avg_cpu_pct),
                    "peak_rss_gb": float(peak_rss_gb),
                    "gwas_time_s": float(evd_secs + scan_secs),
                    "viz_time_s": float(viz_secs),
                }
            )

            os.replace(tmp_tsv, out_tsv)
            saved_paths.append(str(out_tsv).replace("//", "/"))
            _log_file_only(
                logger,
                logging.INFO,
                f"Results saved to {str(out_tsv).replace('//', '/')}",
            )

        time_parts: list[str] = []
        if evd_secs > 0:
            time_parts.append(format_elapsed(evd_secs))
        time_parts.append(format_elapsed(scan_secs))
        if plot and has_results:
            time_parts.append(format_elapsed(viz_secs))
        done_msg = f"{model_label} ...Finished [{'/'.join(time_parts)}]"
        if use_spinner:
            print_success(done_msg)
        else:
            logger.info(done_msg)


# ======================================================================
# High-memory FarmCPU: full genotype + QK
# ======================================================================

def prepare_qk_and_filter(
    geno: np.ndarray,
    ref_alt: pd.DataFrame,
    maf_threshold: float,
    max_missing_rate: float,
    logger,
):
    """
    Filter SNPs and impute missing values using QK, then update ref_alt.
    """
    logger.info(
        "* Filtering SNPs (MAF < "
        f"{maf_threshold} or missing rate > {max_missing_rate}; mode imputation)..."
    )
    logger.info("  Tip: if available, use pre-imputed genotypes from BEAGLE/IMPUTE2.")
    qkmodel = QK(geno, maff=maf_threshold, missf=max_missing_rate)
    geno_filt = qkmodel.M

    ref_alt_filt = ref_alt.loc[qkmodel.SNPretain].copy()
    # Swap REF/ALT for extremely rare alleles
    ref_alt_filt.iloc[qkmodel.maftmark, [0, 1]] = ref_alt_filt.iloc[
        qkmodel.maftmark, [1, 0]
    ]
    ref_alt_filt["maf"] = qkmodel.maf
    logger.info("Filtering and imputation finished.")
    return geno_filt, ref_alt_filt, qkmodel


def build_qmatrix_farmcpu(
    genofile: str,
    gfile_prefix: str,
    geno: np.ndarray,
    qdim: str,
    cov_inputs: Union[str, list[str], None],
    chunk_size: int,
    logger,
    sample_ids: Union[np.ndarray, None] = None,
    use_spinner: bool = False,
    quiet_terminal: bool = False,
) -> np.ndarray:
    """
    Build or load Q matrix for FarmCPU (PCs + optional covariates).
    """
    def _farm_log(msg: str) -> None:
        if bool(quiet_terminal):
            _log_file_only(logger, logging.INFO, str(msg))
        else:
            logger.info(str(msg))

    def _load_or_build_grm_cache_for_pca() -> np.ndarray:
        grm_path = f"{gfile_prefix}.k.1.npy"
        id_path = f"{grm_path}.id"
        n = geno.shape[1]

        if os.path.exists(grm_path):
            _farm_log(f"* Loading GRM cache for FarmCPU PCA from {grm_path}...")
            grm = np.load(grm_path, mmap_mode="r")
            if grm.size != n * n:
                _farm_log(
                    f"GRM cache shape mismatch ({grm.size} elements) for sample size {n}; rebuilding."
                )
            else:
                grm = grm.reshape(n, n)
                if sample_ids is not None and os.path.exists(id_path):
                    grm_ids = _read_id_file(id_path, logger, "GRM", use_spinner=use_spinner)
                    if grm_ids is None or len(grm_ids) != n:
                        _farm_log("GRM cache IDs are invalid; rebuilding GRM cache.")
                    else:
                        grm_ids = np.asarray(grm_ids, dtype=str)
                        sid = sample_ids.astype(str)
                        if np.array_equal(grm_ids, sid):
                            return np.asarray(grm, dtype="float32")
                        index = {s: i for i, s in enumerate(grm_ids)}
                        missing = [s for s in sid if s not in index]
                        if missing:
                            _farm_log(
                                f"GRM cache missing {len(missing)} sample IDs; rebuilding GRM cache."
                            )
                        else:
                            ord_idx = [index[s] for s in sid]
                            grm = grm[np.ix_(ord_idx, ord_idx)]
                            return np.asarray(grm, dtype="float32")
                else:
                    if sample_ids is not None and not os.path.exists(id_path):
                        _farm_log("GRM cache ID file not found; assuming genotype sample order.")
                    return np.asarray(grm, dtype="float32")

        _farm_log("* Building GRM cache for FarmCPU PCA...")
        grm = GRM(geno).astype("float32")
        np.save(grm_path, grm)
        if sample_ids is not None:
            pd.Series(sample_ids.astype(str)).to_csv(
                id_path, sep="\t", index=False, header=False
            )
        _farm_log(f"Cached GRM written to {grm_path}")
        return grm

    def _maybe_load_with_ids(path: str, expect_rows: int):
        df = pd.read_csv(
            path, sep=None, engine="python", header=None,
            dtype=str, keep_default_na=False
        )
        if df.shape[0] != expect_rows:
            raise ValueError(
                f"Q matrix rows ({df.shape[0]}) do not match sample count ({expect_rows})."
            )
        if sample_ids is not None:
            col0 = df.iloc[:, 0].astype(str).str.strip()
            overlap = len(set(col0) & set(sample_ids))
            if overlap >= int(0.9 * len(sample_ids)) and df.shape[1] > 1:
                data = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
                index = {sid: i for i, sid in enumerate(col0)}
                missing = [sid for sid in sample_ids if sid not in index]
                if missing:
                    raise ValueError(f"Q matrix missing {len(missing)} sample IDs (e.g. {missing[:5]}).")
                return data[[index[sid] for sid in sample_ids]]
        # Fallback: treat all columns as numeric Q matrix
        q = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
        return q

    if qdim in np.arange(0, 30).astype(str):
        q_path = f"{gfile_prefix}.q.{qdim}.txt"
        if os.path.exists(q_path):
            q_src = _basename_only(q_path)
            with CliStatus(
                f"Loading Q matrix from {q_src}...",
                enabled=bool(use_spinner and (not quiet_terminal)),
            ) as task:
                qmatrix = _maybe_load_with_ids(q_path, geno.shape[1])
                task.complete(f"Loading Q matrix from {q_src}")
        elif qdim == "0":
            qmatrix = np.array([]).reshape(geno.shape[1], 0)
        else:
            _farm_log(f"* PCA dimension for FarmCPU Q matrix: {qdim}")
            grm = _load_or_build_grm_cache_for_pca()
            _eigval, eigvec = np.linalg.eigh(grm)
            qmatrix = eigvec[:, -int(qdim):]
            if sample_ids is not None:
                df = pd.DataFrame(np.column_stack([sample_ids.astype(str), qmatrix]))
                df.to_csv(q_path, sep="\t", header=False, index=False)
            else:
                np.savetxt(q_path, qmatrix, fmt="%.6f")
            _farm_log(f"Cached Q matrix written to {q_path}")
    else:
        q_src = _basename_only(qdim)
        with CliStatus(
            f"Loading Q matrix from {q_src}...",
            enabled=bool(use_spinner and (not quiet_terminal)),
        ) as task:
            qmatrix = _maybe_load_with_ids(qdim, geno.shape[1])
            task.complete(f"Loading Q matrix from {q_src}")

    if cov_inputs:
        if sample_ids is None:
            raise ValueError("FarmCPU covariate loading requires sample IDs.")
        cov_arr, cov_ids = _load_covariates_for_models(
            cov_inputs=cov_inputs,
            genofile=genofile,
            sample_ids=np.asarray(sample_ids, dtype=str),
            chunk_size=int(chunk_size),
            logger=logger,
            context="FarmCPU",
            use_spinner=bool(use_spinner and (not quiet_terminal)),
        )
        if cov_arr is not None:
            if cov_ids is None:
                raise ValueError("Internal error: covariate IDs are missing for FarmCPU.")
            sid = np.asarray(sample_ids, dtype=str)
            if cov_arr.shape[0] != sid.shape[0]:
                raise ValueError(
                    f"FarmCPU covariate rows ({cov_arr.shape[0]}) do not match sample count ({sid.shape[0]})."
                )
            if not np.array_equal(np.asarray(cov_ids, dtype=str), sid):
                raise ValueError(
                    "FarmCPU covariate sample order does not match genotype sample order after alignment."
                )
            qmatrix = np.concatenate([qmatrix, cov_arr], axis=1)

    _farm_log(f"Q matrix (FarmCPU) shape: {qmatrix.shape}")
    return qmatrix


def run_farmcpu_fullmem(
    args,
    gfile: str,
    prefix: str,
    logger: logging.Logger,
    pheno_preloaded: Union[pd.DataFrame , None] = None,
    ids_preloaded: Union[np.ndarray, None] = None,
    n_snps_preloaded: Union[int, None] = None,
    qmatrix_preloaded: Union[np.ndarray, None] = None,
    cov_preloaded: Union[np.ndarray, None] = None,
    use_spinner: bool = False,
    context_prepared: bool = False,
    summary_rows: Union[list[dict[str, object]], None] = None,
    saved_paths: Union[list[str], None] = None,
    trait_names: Union[list[str], None] = None,
    farmcpu_cache: Union[dict[str, object], None] = None,
) -> dict[str, object]:
    """
    Run FarmCPU in high-memory mode (full genotype + QK + PCA).

    If pheno_preloaded is provided, it will reuse that phenotype table to avoid
    repeated "Loading phenotype ..." logs and repeated I/O.
    """
    phenofile = args.pheno
    outfolder = args.out
    qdim = args.qcov
    cov = args.cov

    if farmcpu_cache is None:
        t_loading = time.time()
        pheno = pheno_preloaded
        if pheno is None:
            pheno = _load_phenotype_with_status(
                phenofile,
                args.ncol,
                logger,
                id_col=0,
                use_spinner=use_spinner,
            )
        else:
            if not bool(context_prepared):
                with CliStatus("Loading phenotype from dataframe...", enabled=bool(use_spinner)) as task:
                    task.complete(
                        f"Loading phenotype from dataframe (n={pheno.shape[0]}, npheno={pheno.shape[1]})"
                    )

        if ids_preloaded is not None and n_snps_preloaded is not None:
            famid = np.asarray(ids_preloaded, dtype=str)
            n_snps = int(n_snps_preloaded)
        else:
            famid, n_snps = _inspect_genotype_with_status(
                gfile,
                logger,
                use_spinner=use_spinner,
            )
            famid = np.asarray(famid, dtype=str)
        geno_chunks = []
        site_rows = []
        pbar = _ProgressAdapter(total=n_snps, desc="Loading genotype (Full)")
        for chunk, sites in load_genotype_chunks(
            gfile,
            chunk_size=args.chunksize,
            maf=args.maf,
            missing_rate=args.geno,
            impute=True,
            sample_ids=famid.tolist(),
        ):
            if chunk.shape[0] == 0:
                continue
            geno_chunks.append(np.asarray(chunk, dtype="float32"))
            site_rows.extend(
                [(s.chrom, int(s.pos), s.ref_allele, s.alt_allele) for s in sites]
            )
            pbar.update(chunk.shape[0])
        loaded_snps = int(sum(c.shape[0] for c in geno_chunks))
        pbar.set_desc(f"Loading genotype (Full, {loaded_snps} SNPs)")
        pbar.finish()
        pbar.close(success_style=False)

        if len(geno_chunks) == 0:
            msg = "After filtering, number of SNPs is zero for FarmCPU."
            logger.error(msg)
            raise ValueError(msg)
        geno = np.concatenate(geno_chunks, axis=0)
        ref_alt = pd.DataFrame(site_rows, columns=["chrom", "pos", "allele0", "allele1"])
        ref_alt["pos"] = pd.to_numeric(ref_alt["pos"], errors="coerce").fillna(0).astype(int)

        t_loaded = time.time() - t_loading
        if not bool(context_prepared):
            _rich_success(
                logger,
                f"FarmCPU input ready (n={len(famid)}, nSNP={geno.shape[0]}) [{format_elapsed(t_loaded)}]",
                use_spinner=use_spinner,
                log_message=f"Genotype and phenotype loaded in {t_loaded:.2f} seconds",
            )
        else:
            _log_file_only(
                logger,
                logging.INFO,
                f"Genotype and phenotype loaded in {t_loaded:.2f} seconds",
            )
        if geno.size == 0:
            msg = "After filtering, number of SNPs is zero for FarmCPU."
            logger.error(msg)
            raise ValueError(msg)

        if bool(context_prepared) and qmatrix_preloaded is not None:
            qmatrix = np.asarray(qmatrix_preloaded, dtype="float32")
            if cov_preloaded is not None:
                cov_arr = np.asarray(cov_preloaded, dtype="float32")
                if cov_arr.shape[0] != qmatrix.shape[0]:
                    raise ValueError(
                        f"FarmCPU preloaded covariate rows ({cov_arr.shape[0]}) "
                        f"do not match preloaded Q rows ({qmatrix.shape[0]})."
                    )
                qmatrix = np.concatenate([qmatrix, cov_arr], axis=1)
        else:
            gfile_prefix = genotype_cache_prefix(gfile)
            qmatrix = build_qmatrix_farmcpu(
                genofile=gfile,
                gfile_prefix=gfile_prefix,
                geno=geno,
                qdim=qdim,
                cov_inputs=cov,
                chunk_size=args.chunksize,
                logger=logger,
                sample_ids=famid.astype(str),
                use_spinner=use_spinner,
                quiet_terminal=bool(context_prepared),
            )

        if bool(context_prepared):
            cov_n = "NA" if cov_preloaded is None else int(np.asarray(cov_preloaded).shape[0])
            _log_file_only(
                logger,
                logging.INFO,
                f"geno={geno.shape[1]}, pheno={pheno.shape[0]}, "
                f"q={qmatrix.shape[0]}, cov={cov_n} -> {famid.shape[0]}"
            )
        farmcpu_cache = {
            "pheno": pheno,
            "famid": famid,
            "geno": geno,
            "ref_alt": ref_alt,
            "qmatrix": qmatrix,
        }
    else:
        pheno = farmcpu_cache["pheno"]  # type: ignore[assignment]
        famid = np.asarray(farmcpu_cache["famid"], dtype=str)
        geno = np.asarray(farmcpu_cache["geno"], dtype="float32")
        ref_alt = farmcpu_cache["ref_alt"]  # type: ignore[assignment]
        qmatrix = np.asarray(farmcpu_cache["qmatrix"], dtype="float32")

    process = psutil.Process()
    n_cores = psutil.cpu_count(logical=True) or cpu_count()
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    trait_iter = list(pheno.columns) if trait_names is None else [t for t in trait_names if t in pheno.columns]
    multi_trait_mode = len(trait_iter) > 1
    for trait_idx, phename in enumerate(trait_iter):
        p = pheno[phename].dropna()
        famidretain = np.isin(famid, p.index)
        if np.sum(famidretain) == 0:
            logger.info(f"{phename}: no overlapping samples, skipped.")
            if multi_trait_mode:
                logger.info("")  # single blank line between traits
            continue

        snp_sub = geno[:, famidretain]
        p_sub = p.loc[famid[famidretain]].values.reshape(-1, 1)
        q_sub = qmatrix[famidretain]
        n_idv = int(np.sum(famidretain))
        _log_file_only(
            logger,
            logging.INFO,
            f"FarmCPU, trait: {phename}, Samples: {n_idv}",
        )
        trait_line = f"{phename} (n={n_idv})"
        if use_spinner:
            print(trait_line, flush=True)
        else:
            logger.info(trait_line)

        cpu_t0 = process.cpu_times()
        t0 = time.time()
        gwas_t0 = time.time()
        peak_rss = process.memory_info().rss
        maf = snp_sub.mean(axis=1)/2
        farm_iter = 20
        farm_label = f"FarmCPU-{phename} (n={n_idv})"
        farm_pbar = _ProgressAdapter(total=farm_iter, desc=farm_label)
        farm_state = {"done": 0}

        def _farmcpu_progress(done: int, total: int) -> None:
            nonlocal peak_rss
            target = int(max(0, min(int(total), int(done))))
            delta = target - int(farm_state["done"])
            if delta > 0:
                farm_pbar.update(delta)
                farm_state["done"] = target
            try:
                peak_rss = max(peak_rss, process.memory_info().rss)
            except Exception:
                pass

        try:
            farm_out = farmcpu(
                y=p_sub,
                M=snp_sub,
                X=q_sub,
                chrlist=ref_alt["chrom"].values,
                poslist=ref_alt["pos"].values,
                iter=farm_iter,
                threads=args.thread,
                progress_cb=_farmcpu_progress,
                return_info=True,
            )
        except Exception:
            farm_pbar.close()
            raise
        else:
            farm_pbar.finish()
        if isinstance(farm_out, tuple):
            res, _farm_info = farm_out
            n_pseudo_qtn = int(_farm_info.get("n_pseudo_qtn", 0))
        else:
            res = farm_out
            n_pseudo_qtn = 0
        farm_pbar.close(show_done=False)
        gwas_secs = max(time.time() - gwas_t0, 0.0)
        res_df = pd.DataFrame(res, columns=["beta", "se", "pwald"])
        res_df['maf'] = maf
        res_df = pd.concat([ref_alt.reset_index(drop=True), res_df], axis=1)
        res_df = res_df[['chrom','pos','allele0','allele1','maf','beta','se','pwald']]

        viz_secs = 0.0
        if args.plot:
            viz_t0 = time.time()
            fastplot(
                res_df,
                p_sub,
                xlabel=phename,
                outpdf=f"{outfolder}/{prefix}.{phename}.farmcpu.svg",
            )
            viz_secs = max(time.time() - viz_t0, 0.0)

        peak_rss = max(peak_rss, process.memory_info().rss)
        cpu_t1 = process.cpu_times()
        wall = max(time.time() - t0, 1e-9)
        cpu_used = (cpu_t1.user + cpu_t1.system) - (cpu_t0.user + cpu_t0.system)
        avg_cpu = 100.0 * cpu_used / (wall * max(1, n_cores))
        peak_rss_gb = peak_rss / (1024 ** 3)
        _log_file_only(
            logger,
            logging.INFO,
            f"avg CPU ~ {avg_cpu:.1f}% of {n_cores} c, "
            f"peak RSS ~ {peak_rss_gb:.2f} G, pseudoQTNs ~ {n_pseudo_qtn}",
        )
        summary_rows.append(
            {
                "phenotype": str(phename),
                "model": "Farm",
                "nidv": int(n_idv),
                "eff_snp": int(snp_sub.shape[0]),
                "pve": None,
                "avg_cpu": float(avg_cpu),
                "peak_rss_gb": float(peak_rss_gb),
                "gwas_time_s": float(gwas_secs),
                "viz_time_s": float(viz_secs),
            }
        )

        res_df = res_df.astype({"pwald": "object","pos":int})
        res_df.loc[:, "pwald"] = res_df["pwald"].map(lambda x: f"{x:.4e}")
        out_tsv = f"{outfolder}/{prefix}.{phename}.farmcpu.tsv"
        res_df.to_csv(out_tsv, sep="\t", float_format="%.4f", index=None)
        saved_paths.append(str(out_tsv).replace("//", "/"))
        _log_file_only(
            logger,
            logging.INFO,
            f"Results saved to {str(out_tsv).replace('//', '/')}",
        )
        farm_times = [format_elapsed(gwas_secs)]
        if args.plot:
            farm_times.append(format_elapsed(viz_secs))
        farm_done_msg = f"FarmCPU ...Found {n_pseudo_qtn} QTNs [{'/'.join(farm_times)}]"
        if use_spinner:
            print_success(farm_done_msg)
        else:
            logger.info(farm_done_msg)
        if multi_trait_mode:
            logger.info("")
    return farmcpu_cache


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        prog="jx gwas",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx gwas -vcf example.vcf.gz -p pheno.tsv -lmm",
            "jx gwas -bfile example_prefix -p pheno.tsv -lm",
        ]),
    )

    required_group = parser.add_argument_group("Required arguments")

    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument(
        "-vcf", "--vcf", type=str,
        help="Input genotype file in VCF format (.vcf or .vcf.gz).",
    )
    geno_group.add_argument(
        "-file", "--file", type=str,
        help="Input genotype text matrix (.txt/.tsv/.csv), header row is sample IDs.",
    )
    geno_group.add_argument(
        "-bfile", "--bfile", type=str,
        help="Input genotype in PLINK binary format "
             "(prefix for .bed, .bim, .fam).",
    )

    required_group.add_argument(
        "-p", "--pheno", type=str, required=True,
        help="Phenotype file (tab-delimited, sample IDs in the first column).",
    )

    models_group = parser.add_argument_group("Model Arguments")
    models_group.add_argument(
        "-lmm", "--lmm", action="store_true", default=False,
        help="Run the linear mixed model (streaming, low-memory; default: %(default)s).",
    )
    models_group.add_argument(
        "-fastlmm", "--fastlmm", action="store_true", default=False,
        help="Run the linear mixed model with fixed lambda estimated in null model (streaming, low-memory; default: %(default)s).",
    )
    models_group.add_argument(
        "-farmcpu", "--farmcpu", action="store_true", default=False,
        help="Run FarmCPU (full genotype in memory; default: %(default)s).",
    )
    models_group.add_argument(
        "-lm", "--lm", action="store_true", default=False,
        help="Run the linear model (streaming, low-memory; default: %(default)s).",
    )
    models_group.add_argument(
        "-model", "--model", type=str, choices=["add", "dom", "rec", "het"], default="add",
        help="Genetic effect coding model for streaming LM/LMM/fastLMM (default: %(default)s).",
    )

    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-n", "--ncol", action="extend", nargs="*",
        default=None, type=int,
        help="Zero-based phenotype column indices to analyze. "
             'E.g., "-n 0 -n 3" to analyze the 1st and 4th traits '
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-k", "--grm", type=str, default="1",
        help="GRM option: 1 (centering), 2 (standardization), "
             "or a path to a precomputed GRM file (default: %(default)s).",
    )
    optional_group.add_argument(
        "-q", "--qcov", type=str, default="0",
        help="Number of principal components for Q matrix or path to Q file "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-c", "--cov", action="append", type=str, default=None,
        help=(
            "Additional covariate input (repeatable). Each -c accepts either: "
            "(1) covariate file path, or "
            "(2) single-site token chr:pos / chr:start:end (start must equal end, "
            "supports full-width colon). "
            "Examples: -c cov.tsv -c 1:1000 -c 1:1000:1000."
        ),
    )
    optional_group.add_argument(
        "-maf", "--maf", type=float, default=0.02,
        help="Exclude variants with minor allele frequency lower than a threshold "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-geno", "--geno", type=float, default=0.05,
        help="Exclude variants with missing call frequencies greater than a threshold "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-het", "--het", type=float, default=0.02,
        help="Heterozygosity filter threshold for non-additive models. "
             "Sites with het rate outside [het, 1-het] are removed (default: %(default)s).",
    )
    optional_group.add_argument(
        "-chunksize", "--chunksize", type=int, default=100_000,
        help="Number of SNPs per chunk for streaming LMM/LM "
             "(affects GRM and GWAS; default: %(default)s).",
    )
    optional_group.add_argument(
        "-mmap-limit", "--mmap-limit", action="store_true", default=False,
        help="Enable windowed mmap for BED inputs (auto: 2x chunk size).",
    )
    optional_group.add_argument(
        "-t", "--thread", type=int, default=-1,
        help="Number of CPU threads (-1 uses all available cores; default: %(default)s).",
    )
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory for results (default: %(default)s).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Prefix for output files (default: %(default)s).",
    )

    return parser.parse_args()


def main(log: bool = True):
    t_start = time.time()
    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_status = "done"
    run_error = ""
    args = parse_args()
    # Plotting is always enabled for GWAS CLI.
    args.plot = True
    args.cov = _normalize_cov_inputs(args.cov)

    if args.thread <= 0:
        args.thread = cpu_count()
    args.model = args.model.lower()
    if not (0.0 <= args.het <= 0.5):
        raise ValueError("--het must be within [0, 0.5].")

    gfile, prefix = determine_genotype_source(args)

    args.out = os.path.normpath(args.out if args.out is not None else ".")
    os.makedirs(args.out, 0o755, exist_ok=True)
    outprefix = f"{args.out}/{prefix}".replace("\\", "/").replace("//", "/")
    log_path = f"{outprefix}.gwas.log"
    logger = setup_logging(log_path)
    gwas_summary_rows: list[dict[str, object]] = []
    saved_result_paths: list[str] = []
    trait_order: list[str] = []

    if log:
        model_tokens: list[str] = []
        if args.lmm:
            model_tokens.append("LMM")
        if args.fastlmm:
            model_tokens.append("fastLMM")
        if args.lm:
            model_tokens.append("LM")
        if args.farmcpu:
            model_tokens.append("FarmCPU")
        cfg_rows: list[tuple[str, object]] = [
            ("Genotype file", gfile),
            ("Phenotype file", args.pheno),
            ("Phenotype cols", args.ncol if args.ncol is not None else "All"),
            ("Mmap limit", args.mmap_limit),
            ("Models", " ".join(model_tokens) if len(model_tokens) > 0 else "None"),
            ("Genetic model", args.model),
            ("GRM option", args.grm),
            ("Q option", args.qcov),
            ("MAF threshold", args.maf),
            ("Miss threshold", args.geno),
            ("Chunk size", args.chunksize),
        ]
        if args.model != "add":
            cfg_rows.append(("Het filter", f"{args.het} (keep [{args.het}, {1.0 - args.het}])"))
        if args.cov:
            cfg_rows.append(("Covariates", "; ".join(args.cov)))
        emit_cli_configuration(
            logger,
            app_title="JanusX - GWAS",
            config_title="GWAS CONFIG",
            host=socket.gethostname(),
            sections=[("General", cfg_rows)],
            footer_rows=[
                ("Threads", f"{args.thread} ({cpu_count()} available)"),
                ("Output prefix", outprefix),
            ],
            line_max_chars=60,
        )

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    checks.append(ensure_file_exists(logger, args.pheno, "Phenotype file"))
    if args.grm not in ["1", "2"]:
        checks.append(ensure_file_exists(logger, args.grm, "GRM file"))
    if args.qcov not in np.arange(0, 30).astype(str):
        checks.append(ensure_file_exists(logger, args.qcov, "Q matrix file"))
    if args.cov:
        for cov_item in args.cov:
            try:
                site_token = _parse_cov_site_token(cov_item)
            except Exception as e:
                logger.error(str(e))
                raise SystemExit(1)
            if site_token is None:
                checks.append(ensure_file_exists(logger, cov_item, "Covariate file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    if not (args.lm or args.lmm or args.fastlmm or args.farmcpu):
        logger.error("No model selected. Use -lm, -lmm, -fastlmm, and/or -farmcpu.")
        raise SystemExit(1)
    if args.farmcpu and args.model != "add":
        logger.warning(
            "Warning: --model/--het currently apply to streaming LM/LMM/fastLMM; "
            "FarmCPU keeps additive coding."
        )

    try:

        # --- prepare streaming context once if needed ---
        pheno = None
        ids = None
        n_snps = None
        grm = None
        qmatrix = None
        cov_all = None
        eff_m = None
        eff_snp_by_trait: dict[str, int] = {}

        stream_selected = bool(args.lmm or args.lm or args.fastlmm)
        if stream_selected:
            _section(logger, "Metadata preparation")
            _log_file_only(
                logger,
                logging.INFO,
                "Prepare streaming context (phenotype/genotype meta/GRM/Q/cov)",
            )
            pheno, ids, n_snps, grm, qmatrix, cov_all, eff_m = prepare_streaming_context(
                genofile=gfile,
                phenofile=args.pheno,
                pheno_cols=args.ncol,
                maf_threshold=args.maf,
                max_missing_rate=args.geno,
                genetic_model=args.model,
                het_threshold=args.het,
                chunk_size=args.chunksize,
                mgrm=args.grm,
                pcdim=args.qcov,
                cov_inputs=args.cov,
                threads=args.thread,
                mmap_limit=args.mmap_limit,
                require_kinship=(args.lmm or args.fastlmm),
                logger=logger,
                use_spinner=use_spinner,
            )
            logger.info("")
        else:
            if args.farmcpu:
                pheno = _load_phenotype_with_status(
                    args.pheno,
                    args.ncol,
                    logger,
                    id_col=0,
                    use_spinner=use_spinner,
                )

        stream_models: list[str] = []
        if args.lmm:
            stream_models.append("lmm")
        if args.fastlmm:
            stream_models.append("fastlmm")
        if args.lm:
            stream_models.append("lm")
        has_farmcpu = bool(args.farmcpu)

        trait_order = list(pheno.columns) if pheno is not None else []
        if len(stream_models) > 0:
            _section(logger, "Streaming task")

        # -------------------------------
        # 1) Streaming models first
        # -------------------------------
        if len(stream_models) > 0:
            for trait_idx, pname in enumerate(trait_order):
                if len(stream_models) == 1:
                    model_key = stream_models[0]
                    pve_line_model = model_key if model_key in {"lmm", "fastlmm"} else None
                    run_chunked_gwas_lmm_lm(
                        model_name=model_key,
                        genofile=gfile,
                        pheno=pheno,
                        ids=ids,
                        n_snps=n_snps,
                        outprefix=outprefix,
                        maf_threshold=args.maf,
                        max_missing_rate=args.geno,
                        genetic_model=args.model,
                        het_threshold=args.het,
                        chunk_size=args.chunksize,
                        mmap_limit=args.mmap_limit,
                        grm=grm,
                        qmatrix=qmatrix,
                        cov_all=cov_all,
                        eff_m=eff_m,
                        plot=args.plot,
                        threads=args.thread,
                        logger=logger,
                        use_spinner=use_spinner,
                        eff_snp_by_trait=eff_snp_by_trait,
                        summary_rows=gwas_summary_rows,
                        saved_paths=saved_result_paths,
                        trait_names=[str(pname)],
                        show_npve_line=True if pve_line_model is None else (model_key == pve_line_model),
                    )
                else:
                    run_chunked_gwas_streaming_shared(
                        model_names=stream_models,
                        trait_name=str(pname),
                        genofile=gfile,
                        pheno=pheno,
                        ids=ids,
                        n_snps=n_snps,
                        outprefix=outprefix,
                        maf_threshold=args.maf,
                        max_missing_rate=args.geno,
                        genetic_model=args.model,
                        het_threshold=args.het,
                        chunk_size=args.chunksize,
                        mmap_limit=args.mmap_limit,
                        grm=grm,
                        qmatrix=qmatrix,
                        cov_all=cov_all,
                        plot=args.plot,
                        threads=args.thread,
                        logger=logger,
                        use_spinner=use_spinner,
                        eff_snp_by_trait=eff_snp_by_trait,
                        summary_rows=gwas_summary_rows,
                        saved_paths=saved_result_paths,
                    )

                if trait_idx < len(trait_order) - 1:
                    logger.info("")

        # -------------------------------
        # 2) FarmCPU full-memory last
        # -------------------------------
        if has_farmcpu:
            if len(stream_models) > 0:
                logger.info("")
            _section(logger, "Full genotype task")
            context_prepared = bool(pheno is not None and ids is not None and n_snps is not None)
            _ = run_farmcpu_fullmem(
                args=args,
                gfile=gfile,
                prefix=prefix,
                logger=logger,
                pheno_preloaded=pheno,  # Reuse phenotype loaded in streaming stage.
                ids_preloaded=ids,
                n_snps_preloaded=n_snps,
                qmatrix_preloaded=qmatrix,
                cov_preloaded=cov_all,
                use_spinner=use_spinner,
                context_prepared=context_prepared,
                summary_rows=gwas_summary_rows,
                saved_paths=saved_result_paths,
                trait_names=[str(t) for t in trait_order],
                farmcpu_cache=None,
            )

        if len(gwas_summary_rows) > 0:
            _emit_gwas_summary(logger, gwas_summary_rows)
        if len(saved_result_paths) > 0:
            logger.info("")
            logger.info("Results saved:")
            for p in saved_result_paths:
                logger.info(f"  {p}")
        logger.info(f"Log saved in {str(log_path).replace('//', '/')}")

    except Exception as e:
        run_status = "failed"
        run_error = str(e)
        logger.exception(f"Error in JanusX GWAS pipeline: {e}")

    try:
        genofile_kind = "bfile" if bool(args.bfile) else ("vcf" if bool(args.vcf) else "file")
        args_data = {
            "models": {
                "lmm": bool(args.lmm),
                "fastlmm": bool(args.fastlmm),
                "lm": bool(args.lm),
                "farmcpu": bool(args.farmcpu),
            },
            "model": str(args.model),
            "maf": float(args.maf),
            "geno": float(args.geno),
            "het": float(args.het),
            "chunksize": int(args.chunksize),
            "thread": int(args.thread),
            "ncol": (list(args.ncol) if args.ncol is not None else None),
            "cov": (list(args.cov) if args.cov is not None else []),
            "grm": str(args.grm),
            "qcov": str(args.qcov),
            "traits": [str(x) for x in trait_order],
        }
        record_gwas_run(
            run_id=run_id,
            status=run_status,
            genofile=str(gfile),
            genofile_kind=genofile_kind,
            phenofile=str(args.pheno),
            outprefix=str(outprefix),
            log_file=str(log_path),
            result_files=[str(x) for x in saved_result_paths],
            summary_rows=[dict(x) for x in gwas_summary_rows],
            args_data=args_data,
            error_text=run_error,
            created_at=run_created_at,
        )
    except Exception as e:
        logger.warning(f"Failed to write GWAS history DB: {e}")

    lt = time.localtime()
    endinfo = (
        f"\n\n\nFinished. Total wall time: {round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    logger.info(endinfo)


if __name__ == "__main__":
    main()
