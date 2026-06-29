# -*- coding: utf-8 -*-
"""
JanusX: Principal Component Analysis (PCA) Command-Line Interface

Design overview
---------------
Input modes:
  - VCF    : genotype in VCF/VCF.GZ format
  - BFILE  : genotype in PLINK binary format (.bed/.bim/.fam prefix)
  - FILE   : genotype numeric matrix (.txt/.tsv/.csv/.npy) with sibling prefix.id
  - GRM    : precomputed genetic relationship matrix (GRM) plus ID file
  - QCOV : precomputed PCA results (eigenvec/eigenval) for
             visualization only

PCA computation strategy
------------------------
  - For VCF/BFILE:
      * Build GRM via Rust kernels from PLINK BED/cache.
      * Perform eigendecomposition of GRM via Rust LAPACK backend.
      * Optional: --rsvd uses Rust memmap-aware randomized SVD directly.

  - For GRM:
      * Read the precomputed GRM from .cGRM/.sGRM (.txt/.npy), or legacy .grm.*.
      * Perform eigendecomposition via Rust LAPACK backend.

  - For QCOV:
      * Load PC coordinates and eigenvalues directly.
      * Produce only visualization outputs.

Output:
  - {prefix}.eigenvec      : sample IDs + PC coordinates (first column is ID)
  - {prefix}.eigenval      : eigenvalues plus explained-variance ratio per PC
  - Optional plots:
      * {prefix}.eigenvec.2D.pdf
      * {prefix}.eigenvec.3D.gif
"""

import os
from typing import Optional, Union
for key in ["MPLBACKEND"]:
    if key in os.environ:
        del os.environ[key]

import time
import socket
import argparse
import logging
import re
import colorsys
import multiprocessing as mp
import tempfile
import shutil
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psutil
import janusx as jx_pkg
from janusx import janusx as jxrs
from janusx.gfreader import (
    inspect_genotype_file,
    prepare_cli_input_cache,
)
from ._common.log import setup_logging
from ._common.cli_args import (
    add_common_genotype_source_args,
    add_common_memory_arg,
    add_common_out_arg,
    add_common_prefix_arg,
    add_common_thread_arg,
    add_common_variant_filter_args,
)
from ._common.config_render import emit_cli_configuration
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
    format_path_for_display,
)
from ._common.progress import CliStatus, ProgressAdapter, format_elapsed, log_success, stdout_is_tty
from ._common.genocache import configure_genotype_cache_from_out
from ._common.genoio import (
    determine_genotype_source as _determine_genotype_source,
    genotype_load_status_done,
    genotype_load_status_fail,
    genotype_load_status_open,
    read_id_file as _read_id_file,
    read_matrix_with_ids as _read_matrix_with_ids,
    strip_default_prefix_suffix,
)
from ._common.threads import (
    apply_blas_thread_env,
    detect_effective_threads,
    format_requested_thread_usage,
    set_rust_blas_threads,
)
from ._common.memory import (
    bed_block_target_env as _common_bed_block_target_env,
    decode_memory_gb_to_mb as _common_decode_memory_gb_to_mb,
    normalize_decode_memory_gb as _common_normalize_decode_memory_gb,
    resolve_decode_block_rows as _common_resolve_decode_block_rows,
    resolve_decode_mmap_window_mb as _common_resolve_decode_mmap_window_mb,
)

# ======================================================================
# Helpers: GRM-based PCA (aligned with GWAS module)
# ======================================================================

DEFAULT_BED_MEMORY_GB = 1.0
_PCA_WORKING_BUFFERS_STREAM = 2
_PCA_PLOT_BACKEND = None


def _pca_logger_verbose(logger) -> bool:
    if logger is None:
        return False
    try:
        return bool(getattr(logger, "_janusx_pca_verbose"))
    except Exception:
        return False


def _get_pca_plot_backend():
    global _PCA_PLOT_BACKEND
    if _PCA_PLOT_BACKEND is not None:
        return _PCA_PLOT_BACKEND
    import matplotlib as mpl

    mpl.use("Agg")
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from janusx.bioplotkit.sci_set import color_set
    from janusx.bioplotkit.pcshow import PCSHOW

    _PCA_PLOT_BACKEND = (mpl, plt, PCSHOW, color_set)
    return _PCA_PLOT_BACKEND


def _normalize_memory_gb(memory_gb: Union[int, float, None]) -> float:
    return float(
        _common_normalize_decode_memory_gb(
            memory_gb,
            default_gb=float(DEFAULT_BED_MEMORY_GB),
        )
    )


def _memory_gb_to_mb(memory_gb: Union[int, float, None]) -> float:
    return float(
        _common_decode_memory_gb_to_mb(
            memory_gb,
            default_gb=float(DEFAULT_BED_MEMORY_GB),
        )
    )


def _get_palette_helpers():
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib import colors as mcolors

    return mpl, mcolors


def _get_palette_cmap(name: str):
    mpl, _mcolors = _get_palette_helpers()
    try:
        return mpl.colormaps.get_cmap(str(name))
    except Exception as exc:
        raise ValueError(f"Invalid --palette colormap: {name}") from exc


def _parse_rgb_triplet(token: str) -> str:
    _mpl, mcolors = _get_palette_helpers()
    text = str(token).strip()
    if not (text.startswith("(") and text.endswith(")")):
        raise ValueError(f"Invalid RGB tuple token: {token}")
    parts = [p.strip() for p in text[1:-1].split(",")]
    if len(parts) != 3:
        raise ValueError(f"RGB tuple must have 3 values: {token}")
    try:
        rgb = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"RGB tuple must be integers: {token}") from exc
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError(f"RGB tuple values must be in [0, 255]: {token}")
    return mcolors.to_hex([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0])


def _split_palette_tokens(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in str(text):
        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue
        if depth == 0 and ch in {";", ","}:
            tok = "".join(buf).strip()
            if tok != "":
                out.append(tok)
            buf = []
            continue
        buf.append(ch)
    tok = "".join(buf).strip()
    if tok != "":
        out.append(tok)
    return out


def _parse_custom_palette(text: str) -> list[str]:
    _mpl, mcolors = _get_palette_helpers()
    colors: list[str] = []
    for token in _split_palette_tokens(text):
        tok = token.strip()
        if tok == "":
            continue
        if tok.startswith("(") and tok.endswith(")"):
            colors.append(_parse_rgb_triplet(tok))
            continue
        try:
            colors.append(mcolors.to_hex(mcolors.to_rgba(tok)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid --palette color token: {tok}. Use #RRGGBB, color names, or (R,G,B)."
            ) from exc
    if len(colors) == 0:
        raise ValueError("Invalid --palette: empty color list.")
    return colors


def _parse_palette_spec(value: object) -> Optional[tuple[str, object]]:
    _mpl, mcolors = _get_palette_helpers()
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        raise ValueError("Invalid --palette: value is empty.")
    if (";" in text) or ("," in text):
        toks = _split_palette_tokens(text)
        if len(toks) >= 2:
            return ("list", _parse_custom_palette(text))
    try:
        _get_palette_cmap(text)
        return ("cmap", text)
    except ValueError:
        if text.startswith("(") and text.endswith(")"):
            return ("list", [_parse_rgb_triplet(text)])
        try:
            return ("list", [mcolors.to_hex(mcolors.to_rgba(text))])
        except ValueError as exc:
            raise ValueError(
                f"Invalid --palette: {text}. Use a cmap name (e.g. tab10) or ',' / ';' separated colors."
            ) from exc


def _desaturate_color(color: str, sat_scale: float = 0.90) -> str:
    _mpl, mcolors = _get_palette_helpers()
    r, g, b = mcolors.to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s2 = float(np.clip(float(s) * float(sat_scale), 0.0, 1.0))
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s2, v)
    return mcolors.to_hex((r2, g2, b2))


def _resolve_palette_colors(spec: Optional[tuple[str, object]], n_series: int) -> list[str]:
    _mpl, mcolors = _get_palette_helpers()
    if n_series <= 0:
        return []
    if spec is None:
        spec = ("cmap", "tab10")

    mode, payload = spec
    if mode == "list":
        colors = [mcolors.to_hex(mcolors.to_rgba(c)) for c in list(payload)]
        if len(colors) == 0:
            colors = [mcolors.to_hex(_get_palette_cmap("tab10")(i / max(1, n_series - 1))) for i in range(n_series)]
        return [colors[i % len(colors)] for i in range(n_series)]

    cmap_name = str(payload).strip()
    cmap = _get_palette_cmap(cmap_name)
    if getattr(cmap, "N", 256) <= 32:
        n_bins = max(1, int(cmap.N))
        colors = [mcolors.to_hex(cmap(i % n_bins)) for i in range(n_series)]
    else:
        colors = [mcolors.to_hex(cmap(i / max(1, n_series - 1))) for i in range(n_series)]
    if cmap_name.lower() == "tab10":
        colors = [_desaturate_color(c, 0.90) for c in colors]
    return colors


@contextmanager
def _bed_block_target_env(memory_mb: Union[int, float, None]) -> None:
    with _common_bed_block_target_env(
        memory_mb,
        needs_copy=False,
        buffers=_PCA_WORKING_BUFFERS_STREAM,
    ):
        yield

def load_group_table(group_path: str) -> tuple[pd.DataFrame, Union[str , None], Union[str , None]]:
    group_df = pd.read_csv(group_path, sep="\t", header=None, index_col=0)
    if group_df.shape[1] == 0:
        raise ValueError(f"Group file has no columns: {group_path}")
    if group_df.shape[1] == 1:
        group_df.columns = ["group"]
        return group_df, "group", None
    group_df = group_df.iloc[:, :2]
    group_df.columns = ["group", "label"]
    return group_df, "group", "label"


def extract_group_order(group_df: pd.DataFrame, group_col: Union[str, None]) -> Union[list, None]:
    if group_col is None or group_col not in group_df.columns:
        return None
    return pd.Index(group_df[group_col].dropna()).unique().tolist()


def resolve_group_levels(data: pd.DataFrame, group_col: Union[str, None], group_order: Union[list, None]) -> list:
    if group_col is None or group_col not in data.columns:
        return []
    present_groups = pd.Index(data[group_col].dropna().unique()).tolist()
    if group_order is None:
        return present_groups

    present_set = set(present_groups)
    ordered_groups = []
    seen = set()
    for g in group_order:
        if pd.isna(g) or g not in present_set or g in seen:
            continue
        ordered_groups.append(g)
        seen.add(g)
    for g in present_groups:
        if g not in seen:
            ordered_groups.append(g)
    return ordered_groups


def _is_plink_prefix_path(path_or_prefix: str) -> bool:
    p = str(path_or_prefix)
    return all(os.path.isfile(f"{p}.{ext}") for ext in ("bed", "bim", "fam"))


def _is_txt_like_input_path(path_or_prefix: str) -> bool:
    p = str(path_or_prefix)
    low = p.lower()
    if low.endswith((".txt", ".tsv", ".csv")):
        return True
    return any(os.path.isfile(f"{p}{ext}") for ext in (".txt", ".tsv", ".csv"))


def _resolve_txt_matrix_path(path_or_prefix: str) -> Union[str, None]:
    p = str(path_or_prefix)
    low = p.lower()
    if low.endswith((".txt", ".tsv", ".csv")) and os.path.isfile(p):
        return p
    for ext in (".txt", ".tsv", ".csv"):
        cand = f"{p}{ext}"
        if os.path.isfile(cand):
            return cand
    return None


def _resolve_rust_grm_input_for_pca(
    genofile: str,
    *,
    snps_only: bool,
    threads: int = 0,
) -> str:
    p = str(genofile).strip()
    if _is_plink_prefix_path(p):
        return p
    delim = "," if p.lower().endswith(".csv") else None
    cached = prepare_cli_input_cache(
        p,
        snps_only=bool(snps_only),
        delimiter=delim,
        prefer_plink_for_txt=True,
        threads=int(threads),
    )
    if not _is_plink_prefix_path(str(cached)):
        raise RuntimeError(
            "PCA Rust GRM backend requires PLINK BED input/cache prefix. "
            f"failed source={genofile}"
        )
    return str(cached)


def _txt_is_discrete_012_matrix(matrix_path: str) -> bool:
    """
    Return True only when all non-missing numeric entries are in {0,1,2}.

    Missing tokens accepted: NA/NAN/. and -9.
    """
    if not os.path.isfile(matrix_path):
        return False

    def _split_tokens(line: str) -> list[str]:
        return [x.lstrip("\ufeff") for x in re.split(r"[,\s]+", line.strip().lstrip("\ufeff")) if x]

    def _is_missing_token(tok: str) -> bool:
        t = str(tok).strip().upper()
        return t in {"NA", "NAN", "."}

    seen_numeric = False
    with open(matrix_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = _split_tokens(s)
            if not toks:
                continue

            vals: list[float] = []
            parse_fail = 0
            for t in toks:
                if _is_missing_token(t):
                    continue
                try:
                    vals.append(float(t))
                except Exception:
                    parse_fail += 1

            if parse_fail > 0:
                # tolerate a possible leading row label column
                vals2: list[float] = []
                parse_fail2 = 0
                for t in toks[1:]:
                    if _is_missing_token(t):
                        continue
                    try:
                        vals2.append(float(t))
                    except Exception:
                        parse_fail2 += 1
                if parse_fail2 == 0 and len(vals2) > 0:
                    vals = vals2
                elif not seen_numeric:
                    # likely header line, skip once/early lines
                    continue
                else:
                    return False

            if len(vals) == 0:
                continue
            seen_numeric = True
            for v in vals:
                if abs(v + 9.0) <= 1e-8:
                    continue
                if abs(v - 0.0) <= 1e-8 or abs(v - 1.0) <= 1e-8 or abs(v - 2.0) <= 1e-8:
                    continue
                return False
    return seen_numeric


def _rsvd_worker(
    q,
    genotype_path: str,
    dim: int,
    seed: int,
    power: int,
    tol: float,
    snps_only: bool,
    maf: float,
    missing_rate: float,
    mmap_window_mb: int,
    force_packed_bed: bool,
    threads: int,
    eval_path: str,
    evec_path: str,
    trace_path: str,
) -> None:
    try:
        if int(threads) > 0 and hasattr(jxrs, "admx_set_threads"):
            jxrs.admx_set_threads(int(threads))
        if bool(force_packed_bed):
            os.environ["JANUSX_RSVD_BED_PACKED"] = "1"
        eval_raw, evec_raw, total_variance = jxrs.admx_rsvd_stream_sample(
            str(genotype_path),
            int(dim),
            int(seed),
            int(power),
            float(tol),
            bool(snps_only),
            float(maf),
            float(missing_rate),
            None,
            int(mmap_window_mb),
        )
        eval_np = np.asarray(eval_raw, dtype=np.float32)
        evec_np = np.asarray(evec_raw, dtype=np.float32)
        np.save(eval_path, eval_np)
        np.save(evec_path, evec_np)
        np.save(trace_path, np.asarray([float(total_variance)], dtype=np.float64))
        q.put(("ok", ""))
    except Exception as e:
        q.put(("err", str(e)))
        raise


def _select_rsvd_mp_context():
    """
    Select a multiprocessing context for RSVD subprocesses.

    Prefer spawn to avoid Python 3.12+ fork warnings in multi-threaded processes
    on Unix-like systems. Allow manual override via JANUSX_MP_START_METHOD.
    """
    try:
        methods = [str(m).strip().lower() for m in mp.get_all_start_methods()]
    except Exception:
        methods = []

    env_method = str(os.environ.get("JANUSX_MP_START_METHOD", "")).strip().lower()
    if env_method and env_method in methods:
        return mp.get_context(env_method)

    for name in ("spawn", "forkserver", "fork"):
        if name in methods:
            return mp.get_context(name)
    return mp.get_context()


def _run_rsvd_subprocess(
    *,
    genotype_path: str,
    dim: int,
    seed: int,
    power: int,
    tol: float,
    snps_only: bool,
    maf: float,
    missing_rate: float,
    mmap_window_mb: int,
    force_packed_bed: bool,
    threads: int,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    ctx = _select_rsvd_mp_context()
    q = ctx.Queue(maxsize=1)
    tmpdir = tempfile.mkdtemp(prefix="janusx_pca_rsvd_")
    eval_path = os.path.join(tmpdir, "eval.npy")
    evec_path = os.path.join(tmpdir, "evec.npy")
    trace_path = os.path.join(tmpdir, "trace.npy")
    proc = ctx.Process(
        target=_rsvd_worker,
        args=(
            q,
            str(genotype_path),
            int(dim),
            int(seed),
            int(power),
            float(tol),
            bool(snps_only),
            float(maf),
            float(missing_rate),
            int(mmap_window_mb),
            bool(force_packed_bed),
            int(threads),
            str(eval_path),
            str(evec_path),
            str(trace_path),
        ),
        daemon=True,
    )
    try:
        proc.start()
        t0 = time.monotonic()
        while proc.is_alive():
            proc.join(timeout=0.1)
            if progress_callback is not None:
                try:
                    progress_callback(max(0.0, time.monotonic() - t0))
                except Exception:
                    pass
        proc.join(timeout=0.1)

        state = None
        msg = ""
        try:
            if not q.empty():
                state, msg = q.get_nowait()
        except Exception:
            state = None

        if proc.exitcode != 0:
            if state == "err" and msg:
                raise RuntimeError(f"Random-SVD worker failed: {msg}")
            raise RuntimeError("Random-SVD worker failed or was interrupted.")
        if state != "ok":
            raise RuntimeError("Random-SVD worker did not return completion state.")
        eval_np = np.load(eval_path)
        evec_np = np.load(evec_path)
        trace_np = np.asarray(np.load(trace_path), dtype=np.float64).reshape(-1)
        if trace_np.size == 0 or (not np.isfinite(trace_np[0])) or trace_np[0] <= 0.0:
            raise RuntimeError("Random-SVD worker returned invalid total variance.")
        total_variance = float(trace_np[0])
        return (
            np.asarray(eval_np, dtype=np.float32),
            np.asarray(evec_np, dtype=np.float32),
            total_variance,
        )
    finally:
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=0.2)
        except Exception:
            pass
        try:
            q.close()
        except Exception:
            pass
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_rsvd_direct(
    *,
    genotype_path: str,
    dim: int,
    seed: int,
    power: int,
    tol: float,
    snps_only: bool,
    maf: float,
    missing_rate: float,
    mmap_window_mb: int,
    force_packed_bed: bool,
    threads: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    prev_pack = os.environ.get("JANUSX_RSVD_BED_PACKED")
    try:
        if int(threads) > 0 and hasattr(jxrs, "admx_set_threads"):
            jxrs.admx_set_threads(int(threads))
        if bool(force_packed_bed):
            os.environ["JANUSX_RSVD_BED_PACKED"] = "1"
        eval_raw, evec_raw, total_variance = jxrs.admx_rsvd_stream_sample(
            str(genotype_path),
            int(dim),
            int(seed),
            int(power),
            float(tol),
            bool(snps_only),
            float(maf),
            float(missing_rate),
            None,
            int(mmap_window_mb),
        )
    finally:
        if prev_pack is None:
            os.environ.pop("JANUSX_RSVD_BED_PACKED", None)
        else:
            os.environ["JANUSX_RSVD_BED_PACKED"] = prev_pack
    total_variance = float(total_variance)
    if (not np.isfinite(total_variance)) or total_variance <= 0.0:
        raise RuntimeError("Random-SVD returned invalid total variance.")
    return (
        np.asarray(eval_raw, dtype=np.float32),
        np.asarray(evec_raw, dtype=np.float32),
        total_variance,
    )


def _use_rsvd_subprocess() -> bool:
    raw = str(
        os.environ.get(
            "JX_PCA_RSVD_SUBPROCESS",
            os.environ.get("JANUSX_PCA_RSVD_SUBPROCESS", ""),
        )
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _algo_spinner_symbol(elapsed_s: float) -> str:
    # Keep the first frame as "\" on Windows for consistency with prior UX.
    frames = ("\\", "|", "/", "-") if os.name == "nt" else ("/", "-", "\\", "|")
    idx = int(max(0.0, float(elapsed_s)) * 10.0) % len(frames)
    return frames[idx]


def _algo_metrics_suffix(
    n_samples: Optional[int] = None,
    n_snps: Optional[int] = None,
) -> str:
    if n_samples is None or n_snps is None:
        return ""
    return f" (n={int(n_samples)}, snp={int(n_snps)})"


def _algo_progress_text(
    algo_name: str,
    elapsed_s: float,
    n_samples: Optional[int] = None,
    n_snps: Optional[int] = None,
) -> str:
    symbol = _algo_spinner_symbol(elapsed_s)
    suffix = _algo_metrics_suffix(n_samples=n_samples, n_snps=n_snps)
    return f"{symbol} Process of {algo_name}{suffix} ... [{format_elapsed(elapsed_s)}]"


def _print_algo_progress(
    algo_name: str,
    elapsed_s: float,
    n_samples: Optional[int] = None,
    n_snps: Optional[int] = None,
) -> None:
    if not stdout_is_tty():
        return
    text = _algo_progress_text(
        algo_name,
        elapsed_s,
        n_samples=n_samples,
        n_snps=n_snps,
    )
    sys.stdout.write("\r" + text)
    sys.stdout.flush()


def _clear_algo_progress_line() -> None:
    if not stdout_is_tty():
        return
    cols = shutil.get_terminal_size((80, 20)).columns
    sys.stdout.write("\r" + (" " * cols) + "\r")
    sys.stdout.flush()


def _run_with_algo_progress(
    algo_name: str,
    fn,
    logger=None,
    n_samples: Optional[int] = None,
    n_snps: Optional[int] = None,
):
    t0 = time.monotonic()
    suffix = _algo_metrics_suffix(n_samples=n_samples, n_snps=n_snps)
    # CliStatus animates only for loading/computing-like prefixes.
    status_desc_raw = f"Computing {algo_name}{suffix} ..."
    status_desc = status_desc_raw
    if stdout_is_tty():
        try:
            cols = int(shutil.get_terminal_size((80, 20)).columns)
        except Exception:
            cols = 80
        # Keep room for spinner frame + elapsed suffix to avoid hard wraps.
        max_desc = max(24, int(cols) - 18)
        if len(status_desc_raw) > max_desc:
            if max_desc <= 3:
                status_desc = status_desc_raw[:max_desc]
            else:
                status_desc = status_desc_raw[: max_desc - 3] + "..."

    if stdout_is_tty():
        with CliStatus(status_desc, enabled=True, use_process=True) as task:
            try:
                result = fn()
            except Exception:
                task.fail(f"Computing {algo_name}{suffix} ...Failed")
                raise
    else:
        result = fn()
    return result, max(0.0, time.monotonic() - t0)


def _log_saved_eigen_results(logger, outprefix: str) -> None:
    log_success(
        logger,
        "Results saved:\n"
        f"   {str(outprefix)}.eigenvec\n"
        f"   {str(outprefix)}.eigenval",
    )


def _compute_explained_ratio(
    eigenval: np.ndarray,
    total_variance: float | None = None,
) -> np.ndarray:
    evals = np.asarray(eigenval, dtype=np.float64).reshape(-1)
    if total_variance is None:
        total = float(np.sum(evals))
    else:
        total = float(total_variance)
    if not np.isfinite(total) or total <= 0.0:
        return np.zeros_like(evals, dtype=np.float64)
    return evals / total


def _write_eigenval_table(
    out_path: str,
    eigenval: np.ndarray,
    total_variance: float | None = None,
) -> None:
    evals = np.asarray(eigenval, dtype=np.float64).reshape(-1)
    explained_ratio = _compute_explained_ratio(evals, total_variance=total_variance)
    table = np.column_stack([evals, explained_ratio])
    np.savetxt(
        out_path,
        table,
        fmt=["%.8f", "%.8f"],
        delimiter="\t",
    )


def _load_eigenval_table(path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(np.genfromtxt(path, ndmin=2), dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"Empty eigenval table: {path}")
    if arr.ndim == 2 and arr.shape[1] >= 1:
        evals = np.asarray(arr[:, 0], dtype=np.float64).reshape(-1)
        if arr.shape[1] >= 2:
            ratio = np.asarray(arr[:, 1], dtype=np.float64).reshape(-1)
            if ratio.shape[0] == evals.shape[0] and np.all(np.isfinite(ratio)):
                return evals, ratio
        return evals, _compute_explained_ratio(evals)
    raise ValueError(f"Invalid eigenval table shape: {arr.shape}")


def _log_pca_eigh_backend(logger, evd_backend: str | None) -> None:
    backend = str(evd_backend).strip()
    if backend == "" or (not _pca_logger_verbose(logger)):
        return
    try:
        logger.info(f"EIGH backend: {backend}")
    except Exception:
        pass


def build_grm_streaming_for_pca(
    genofile: str,
    maf_threshold: float = 0.02,
    max_missing_rate: float = 0.05,
    memory_mb: float = DEFAULT_BED_MEMORY_GB * 1024.0,
    inspected_meta: Optional[tuple[list[str], int]] = None,
    logger=None,
    emit_progress_done: bool = True,
    snps_only: bool = False,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Construct the GRM in memmap mode via Rust BED kernel
    and return (GRM, sample_ids, effective_snp_count).
    """
    if not hasattr(jxrs, "grm_stream_bed_f32"):
        raise RuntimeError(
            "Rust extension is missing grm_stream_bed_f32. "
            "Rebuild/install JanusX extension first."
        )

    rust_input = _resolve_rust_grm_input_for_pca(
        str(genofile),
        snps_only=bool(snps_only),
        threads=int(threads),
    )

    # Inspect genotype meta information
    if inspected_meta is None or (str(rust_input) != str(genofile)):
        sample_ids_raw, n_snps = inspect_genotype_file(
            rust_input,
            snps_only=bool(snps_only),
            maf=float(maf_threshold),
            missing_rate=float(max_missing_rate),
        )
    else:
        sample_ids_raw, n_snps = inspected_meta
    sample_ids = np.array(sample_ids_raw, dtype=str)
    n_samples = len(sample_ids)
    block_rows = _common_resolve_decode_block_rows(
        n_samples,
        float(memory_mb),
        max_rows=max(1, int(n_snps)),
        buffers=_PCA_WORKING_BUFFERS_STREAM,
    )
    block_rows = max(1, int(block_rows))
    pbar = ProgressAdapter(
        total=n_snps,
        desc="GRM (rust-memmap)",
        emit_done=bool(emit_progress_done),
        force_animate=True,
    )
    process = psutil.Process()

    mem_tick_span = max(1, 10 * int(block_rows))
    last_done = 0

    def _progress_cb(done: int, _total: int) -> None:
        nonlocal last_done
        d = int(done)
        if d > last_done:
            pbar.update(d - last_done)
            last_done = d
        if d % mem_tick_span == 0:
            mem = process.memory_info().rss / 1024**3
            pbar.set_postfix(memory=f"{mem:.2f} GB")

    mmap_window_mb = _common_resolve_decode_mmap_window_mb(
        rust_input,
        n_samples,
        n_snps,
        float(memory_mb),
        buffers=_PCA_WORKING_BUFFERS_STREAM,
    )
    try:
        with _bed_block_target_env(memory_mb):
            grm_raw, eff_m_raw, stream_n_raw = jxrs.grm_stream_bed_f32(
                str(rust_input),
                method=1,
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                block_cols=max(1, int(block_rows)),
                threads=max(1, int(threads)) if int(threads) > 0 else 0,
                progress_callback=_progress_cb,
                progress_every=max(1, int(block_rows)),
                mmap_window_mb=(int(mmap_window_mb) if mmap_window_mb is not None else None),
            )
    finally:
        pbar.finish()
        pbar.close()

    if int(stream_n_raw) != int(n_samples):
        raise RuntimeError(
            f"PCA GRM sample count mismatch: memmap={int(stream_n_raw)}, expected={int(n_samples)}"
        )
    grm = np.ascontiguousarray(np.asarray(grm_raw, dtype=np.float32))
    eff_m = int(eff_m_raw)
    if eff_m <= 0:
        raise RuntimeError("No SNPs remained after filtering; GRM for PCA is empty.")

    if logger is not None:
        logger.info(f"GRM construction finished. Effective SNPs: {eff_m}")

    return grm, sample_ids, int(eff_m)


def eigendecompose_grm(
    grm: np.ndarray,
    logger=None,
    *,
    threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Perform eigen decomposition of a symmetric GRM with Rust LAPACK backend:
      GRM = V Λ V^T

    Returns:
      eigenvec : columns are eigenvectors ordered by descending eigenvalue
      eigenval : corresponding eigenvalues (1D array)
    """
    if logger is not None:
        logger.info("* Performing eigen decomposition of GRM...")

    if not hasattr(jxrs, "rust_eigh_from_array_f64"):
        raise RuntimeError(
            "Rust extension is missing rust_eigh_from_array_f64. "
            "Rebuild/install JanusX extension first."
        )

    g0 = np.asarray(grm, dtype=np.float64)
    if g0.ndim != 2 or g0.shape[0] != g0.shape[1] or g0.shape[0] == 0:
        raise RuntimeError(f"PCA eigh expects non-empty square GRM, got shape={g0.shape}")
    g = np.ascontiguousarray(g0)
    n = int(g.shape[0])
    threads = max(1, int(detect_effective_threads() if threads is None else threads))

    # Keep PCA aligned with GWAS default strategy:
    # default to Rust "auto", but switch large full-vector problems to dsyevr
    # before hitting dsyevd workspace limits.
    raw_driver = str(
        os.environ.get(
            "JX_PCA_EIGH_DRIVER",
            os.environ.get("JX_GWAS_EIGH_DRIVER", "auto"),
        )
    ).strip().lower()
    driver = raw_driver if raw_driver in {"auto", "dsyevd", "dsyevr"} else "auto"
    if driver == "auto" and n >= 32768:
        driver = "dsyevr"

    def _run(driver_name: str, mat: np.ndarray):
        if hasattr(jxrs, "rust_eigh_from_array_f64_inplace") and bool(mat.flags.c_contiguous) and bool(mat.flags.writeable):
            try:
                return jxrs.rust_eigh_from_array_f64_inplace(
                    mat,
                    threads=int(threads),
                    driver=str(driver_name),
                    jobz="V",
                    require_lapack=False,
                )
            except Exception as ex_inplace:
                mat_retry = np.ascontiguousarray(mat, dtype=np.float64)
                try:
                    return jxrs.rust_eigh_from_array_f64(
                        mat_retry,
                        threads=int(threads),
                        driver=str(driver_name),
                        jobz="V",
                        require_lapack=False,
                    )
                except Exception as ex_copy:
                    raise RuntimeError(
                        f"Rust eigh failed after inplace retry "
                        f"(inplace={ex_inplace}; copied={ex_copy})"
                    ) from ex_copy
        return jxrs.rust_eigh_from_array_f64(
            mat,
            threads=int(threads),
            driver=str(driver_name),
            jobz="V",
            require_lapack=False,
        )

    try:
        eval_raw, evec_raw, _blas_backend, _evd_backend, _n, _tb, _ti, _ta, _lapack_used, _elapsed = _run(driver, g)
    except Exception as ex:
        if str(driver).lower() == "dsyevr":
            g_retry = np.ascontiguousarray(g0)
            eval_raw, evec_raw, _blas_backend, _evd_backend, _n, _tb, _ti, _ta, _lapack_used, _elapsed = _run("dsyevd", g_retry)
        else:
            raise RuntimeError(f"Rust eigh failed (driver={driver}): {ex}") from ex

    if evec_raw is None:
        raise RuntimeError("Rust eigh returned no eigenvectors for jobz=V.")
    eigval = np.asarray(eval_raw, dtype=np.float64).reshape(-1)
    eigvec = np.asarray(evec_raw, dtype=np.float64)
    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    backend = str(_evd_backend).strip()
    if logger is not None and _pca_logger_verbose(logger):
        logger.info(f"Eigen decomposition finished (backend={backend}).")

    return eigvec, eigval, backend


def eigendecompose_grm_file(
    grm_file: str,
    logger=None,
    *,
    threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    """
    Rust-native file-path eigendecomposition for precomputed GRM:
      - .npy: parsed directly in Rust
      - .txt/.tsv/.csv: parsed directly in Rust
    Returns:
      eigenvec, eigenval, backend, n
    """
    if logger is not None:
        logger.info("* Performing eigen decomposition of GRM (Rust file-path backend)...")
    if not hasattr(jxrs, "rust_eigh_from_matrix_file_f64"):
        raise RuntimeError(
            "Rust extension is missing rust_eigh_from_matrix_file_f64. "
            "Rebuild/install JanusX extension first."
        )
    threads = max(1, int(detect_effective_threads() if threads is None else threads))
    raw_driver = str(
        os.environ.get(
            "JX_PCA_EIGH_DRIVER",
            os.environ.get("JX_GWAS_EIGH_DRIVER", "auto"),
        )
    ).strip().lower()
    driver = raw_driver if raw_driver in {"auto", "dsyevd", "dsyevr"} else "auto"

    try:
        ret = jxrs.rust_eigh_from_matrix_file_f64(
            str(grm_file),
            threads=int(threads),
            driver=str(driver),
            jobz="V",
            require_lapack=False,
        )
    except Exception as ex:
        if str(driver).lower() == "dsyevr":
            ret = jxrs.rust_eigh_from_matrix_file_f64(
                str(grm_file),
                threads=int(threads),
                driver="dsyevd",
                jobz="V",
                require_lapack=False,
            )
        else:
            raise RuntimeError(f"Rust file-path eigh failed (driver={driver}): {ex}") from ex

    eval_raw, evec_raw, _blas_backend, _evd_backend, n, _tb, _ti, _ta, _lapack_used, _elapsed = ret
    if evec_raw is None:
        raise RuntimeError("Rust file-path eigh returned no eigenvectors for jobz=V.")
    eigval = np.asarray(eval_raw, dtype=np.float64).reshape(-1)
    eigvec = np.asarray(evec_raw, dtype=np.float64)
    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    backend = str(_evd_backend).strip()
    if logger is not None and _pca_logger_verbose(logger):
        logger.info(f"Eigen decomposition finished (backend={backend}, n={int(n)}).")
    return eigvec, eigval, backend, int(n)


# ======================================================================
# Main CLI
# ======================================================================

def main(log: bool = True):
    t_start = time.time()

    parser = CliArgumentParser(
        prog="jx pca",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx pca -vcf geno.vcf.gz -dim 3 -plot",
            "jx pca -vcf geno.vcf.gz -dim 20 -rsvd 3 0.1",
            "jx pca -hmp geno.hmp.gz -dim 3 -plot",
            "jx pca -k data.grm -dim 3 -plot",
            "jx pca -c data_prefix -plot -plot3D",
        ]),
    )

    # ------------------------- Required arguments -------------------------
    required_group = parser.add_argument_group("Required Arguments")
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    add_common_genotype_source_args(geno_group, help_profile="default")
    geno_group.add_argument(
        "-k", "--grm", type=str,
        help=(
            "GRM prefix for PCA (expects {prefix}.cGRM.* / {prefix}.sGRM.*; "
            "legacy {prefix}.grm.* is also supported)."
        ),
    )
    geno_group.add_argument(
        "-c", "--cov", dest="qcov", type=str,
        help=(
            "Prefix of existing PCA result files for visualization only "
            "({prefix}.eigenval, {prefix}.eigenvec)."
        ),
    )
    geno_group.add_argument(
        "--qcov", dest="qcov", type=str, help=argparse.SUPPRESS
    )

    # ------------------------- Optional arguments -------------------------
    optional_group = parser.add_argument_group("Optional Arguments")
    add_common_out_arg(optional_group, default=".", help_profile="pca_results")
    add_common_prefix_arg(
        optional_group,
        default=None,
        help_profile="inferred_input_filename",
    )
    optional_group.add_argument(
        "-dim", "--dim", type=int, default=3,
        help="Number of leading principal components to output (default: %(default)s).",
    )
    add_common_variant_filter_args(
        optional_group,
        help_profile="default",
        include_het=False,
        maf_default=0.02,
        geno_default=0.05,
    )
    add_common_memory_arg(
        optional_group,
        default=DEFAULT_BED_MEMORY_GB,
        help_text=(
            "Working memory budget in GB for PCA BED/GRM kernels. "
            "Explicit -mem keeps the requested fixed budget."
        ),
        include_hidden_legacy_single_dash_alias=True,
    )
    optional_group.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Show advanced backend and thread diagnostics.",
    )
    add_common_thread_arg(optional_group, default_threads=detect_effective_threads())
    optional_group.add_argument(
        "-plot", "--plot", action="store_true", default=False,
        help="Generate 2D scatter plots for PC1 vs PC2 and PC1 vs PC3 "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-plot3D", "--plot3D", action="store_true", default=False,
        help="Generate a 3D rotating GIF for PC1-PC3 (default: %(default)s).",
    )
    optional_group.add_argument(
        "-group", "--group", type=str, default=None,
        help=(
            "Group file with two columns: sample ID and group label (no header). "
            "Optional third column will be used as a text annotation tag "
            "(default: %(default)s)."
        ),
    )
    optional_group.add_argument(
        "-palette", "--palette", type=str, default="tab10",
        help=(
            "Palette for grouped PCA plots. Supports cmap names (e.g. tab10, tab20) "
            "or ',' / ';' separated colors such as 'red,green,yellow' "
            "(default: %(default)s)."
        ),
    )
    optional_group.add_argument(
        "-rsvd",
        "--rsvd",
        nargs="*",
        default=None,
        metavar="RSVD_ARG",
        help=(
            "Use Rust RSVD for PCA from genotype input (memmap default). "
            "Accepted forms: '-rsvd', '-rsvd 3', or '-rsvd 3 0.1' "
            "(defaults: power=3, tol=0.1)."
        ),
    )

    args = parser.parse_args()
    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    thread_capped = False
    if int(args.thread) <= 0:
        args.thread = int(detected_threads)
    if int(args.thread) > int(detected_threads):
        thread_capped = True
        args.thread = int(detected_threads)

    # Parse -rsvd optional values: -rsvd [power] [tol]
    # Seed is fixed to 42 to keep CLI concise and deterministic.
    raw_rsvd_args = args.rsvd
    args.rsvd = raw_rsvd_args is not None
    args.rsvd_power = 3
    args.rsvd_tol = 1e-1
    args.rsvd_seed = 42
    if raw_rsvd_args is not None:
        if len(raw_rsvd_args) > 2:
            parser.error("-rsvd accepts at most two optional values: [power] [tol].")
        if len(raw_rsvd_args) >= 1:
            try:
                args.rsvd_power = int(raw_rsvd_args[0])
            except Exception:
                parser.error("Invalid RSVD power. Use integer >= 0, e.g. '-rsvd 3' or '-rsvd 3 0.1'.")
        if len(raw_rsvd_args) >= 2:
            try:
                args.rsvd_tol = float(raw_rsvd_args[1])
            except Exception:
                parser.error("Invalid RSVD tol. Use float > 0, e.g. '-rsvd 3 0.1'.")

    if int(args.rsvd_power) < 0:
        parser.error("RSVD power must be >= 0.")
    if not (float(args.rsvd_tol) > 0):
        parser.error("RSVD tol must be > 0.")
    if bool(args.rsvd) and not bool(args.vcf or args.hmp or args.file or args.bfile):
        parser.error("--rsvd is only supported for genotype inputs (-vcf/-hmp/-file/-bfile).")
    # Keep RSVD preprocessing aligned with load_genotype_chunks defaults:
    # only VCF path applies SNP-only allele filter in current pipeline.
    rsvd_snps_only = bool(args.vcf)

    # ------------------------- Resolve input file & output prefix -------------------------
    if args.vcf:
        gfile, auto_prefix = _determine_genotype_source(
            vcf=args.vcf,
            hmp=None,
            file=None,
            bfile=None,
            prefix=None,
        )
        args.prefix = auto_prefix if args.prefix is None else args.prefix
    elif args.hmp:
        gfile, auto_prefix = _determine_genotype_source(
            vcf=None,
            hmp=args.hmp,
            file=None,
            bfile=None,
            prefix=None,
        )
        args.prefix = auto_prefix if args.prefix is None else args.prefix
    elif args.file:
        gfile, auto_prefix = _determine_genotype_source(
            vcf=None,
            hmp=None,
            file=args.file,
            bfile=None,
            prefix=None,
        )
        args.prefix = auto_prefix if args.prefix is None else args.prefix
    elif args.bfile:
        gfile, auto_prefix = _determine_genotype_source(
            vcf=None,
            hmp=None,
            file=None,
            bfile=args.bfile,
            prefix=None,
        )
        args.prefix = auto_prefix if args.prefix is None else args.prefix
    elif args.grm:
        gfile = args.grm
        args.prefix = strip_default_prefix_suffix(os.path.basename(gfile)) if args.prefix is None else args.prefix
    elif args.qcov:
        gfile = args.qcov
        args.prefix = strip_default_prefix_suffix(os.path.basename(gfile)) if args.prefix is None else args.prefix
    else:
        raise ValueError("No valid input found; one of --vcf/--hmp/--bfile/--grm/--cov must be provided.")

    args.out = os.path.normpath(args.out if args.out is not None else ".")
    outprefix = os.path.join(args.out, args.prefix)
    genotype_input_mode = bool(args.vcf or args.hmp or args.file or args.bfile)
    args.memory = _normalize_memory_gb(args.memory)
    memory_mb = _memory_gb_to_mb(args.memory)

    # ------------------------- Logging -------------------------
    os.makedirs(args.out, 0o755, exist_ok=True)
    configure_genotype_cache_from_out(args.out)
    log_path = f"{outprefix}.pca.log"
    logger = setup_logging(log_path)
    setattr(logger, "_janusx_pca_verbose", bool(getattr(args, "verbose", False)))
    if thread_capped:
        logger.warning(
            f"Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(args.thread)}."
        )
    apply_blas_thread_env(int(args.thread))
    set_rust_blas_threads(int(args.thread))
    if hasattr(jxrs, "admx_set_threads"):
        try:
            jxrs.admx_set_threads(int(args.thread))
        except Exception:
            pass
    # Keep algorithm status visible immediately.
    use_spinner = True
    # if genotype_input_mode and (not bool(args.mmap_limit)):
    #     logger.info("PCA backend policy: memmap (default).")

    try:
        args.palette_spec = _parse_palette_spec(args.palette)
    except ValueError as exc:
        logger.error(str(exc))
        raise SystemExit(1)

    if log:
        cfg_rows: list[tuple[str, object]] = []
        rsvd_bed_cache_build = bool(
            args.rsvd and (args.vcf or args.hmp or (args.file and _is_txt_like_input_path(gfile)))
        )
        rsvd_packed_bed_backend = bool(
            args.rsvd and (
                _is_plink_prefix_path(str(gfile))
                or bool(args.vcf)
                or bool(args.hmp)
                or bool(args.file and _is_txt_like_input_path(gfile))
            )
        )
        if args.vcf or args.hmp or args.file or args.bfile:
            cfg_rows.extend(
                [
                    ("Genotype file", gfile),
                    ("Output PCs", f"top {args.dim}"),
                    ("MAF threshold", args.maf),
                    ("Missing rate", args.geno),
                    (
                        "Threads",
                        format_requested_thread_usage(
                            requested_threads=int(requested_threads),
                            using_threads=int(args.thread),
                            detected_threads=int(detected_threads),
                        ),
                    ),
                    ("Memory", args.memory),
                    ("BED backend", "memmap (default)"),
                    ("RSVD mode", args.rsvd),
                ]
            )
            if bool(args.rsvd):
                cfg_rows.extend(
                    [
                        ("RSVD power", args.rsvd_power),
                        ("RSVD tol", args.rsvd_tol),
                        ("RSVD BED cache build", rsvd_bed_cache_build),
                        ("RSVD packed BED backend", rsvd_packed_bed_backend),
                        ("RSVD SNP-only filter", rsvd_snps_only),
                    ]
                )
        elif args.grm:
            cfg_rows.extend(
                [
                    ("GRM prefix", gfile),
                    ("Output PCs", f"top {args.dim}"),
                    (
                        "Threads",
                        format_requested_thread_usage(
                            requested_threads=int(requested_threads),
                            using_threads=int(args.thread),
                            detected_threads=int(detected_threads),
                        ),
                    ),
                ]
            )
        elif args.qcov:
            cfg_rows.append(("PCA prefix", f"{gfile} (visualization only)"))
        if args.plot or args.plot3D:
            cfg_rows.extend(
                [
                    ("2D visualization", args.plot),
                    ("3D visualization (GIF)", args.plot3D),
                ]
            )
        if args.group:
            cfg_rows.extend(
                [
                    ("Group file", args.group),
                    ("Palette", str(args.palette)),
                ]
            )
        emit_cli_configuration(
            logger,
            app_title="JanusX - PCA",
            config_title="PCA CONFIG",
            host=socket.gethostname(),
            sections=[("General", cfg_rows)],
            footer_rows=[("Output prefix", outprefix)],
            line_max_chars=60,
        )

    checks: list[bool] = []
    if args.vcf or args.hmp:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
    elif args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.grm:
        if os.path.isfile(gfile):
            grm_input = gfile
        else:
            grm_input = None
            for cand in (
                f"{gfile}.cGRM.txt",
                f"{gfile}.sGRM.txt",
                f"{gfile}.cGRM.npy",
                f"{gfile}.sGRM.npy",
                f"{gfile}.grm.txt",
                f"{gfile}.grm.npy",
            ):
                if os.path.isfile(cand):
                    grm_input = cand
                    break
        if grm_input is None:
            logger.error(
                "GRM matrix not found: "
                f"{gfile} (or {gfile}.cGRM/.sGRM/.grm with .txt/.npy)"
            )
            checks.append(False)
        if grm_input is not None:
            checks.append(ensure_file_exists(logger, f"{grm_input}.id", "GRM ID file"))
    elif args.qcov:
        checks.append(ensure_file_exists(logger, f"{gfile}.eigenvec", "Eigenvec file"))
        checks.append(ensure_file_exists(logger, f"{gfile}.eigenval", "Eigenval file"))

    if args.group:
        checks.append(ensure_file_exists(logger, args.group, "Group file"))

    if not ensure_all_true(checks):
        raise SystemExit(1)

    # ------------------------- PCA core logic -------------------------
    t_loading = time.time()

    eigenvec = None
    eigenval = None
    explained_ratio = None
    samples = None

    # --- Case 1: VCF / BFILE -> memmap GRM -> PCA (aligned with GWAS) ---
    if args.vcf or args.hmp or args.file or args.bfile:
        rsvd_total_variance = None
        load_src_disp = os.path.basename(str(gfile).rstrip("/\\")) or format_path_for_display(str(gfile))
        if bool(args.rsvd):
            rsvd_input = str(gfile)
            txt_force_eig = False
            txt_delim = "," if str(gfile).lower().endswith(".csv") else None
            if args.file and _is_txt_like_input_path(gfile):
                txt_matrix_path = _resolve_txt_matrix_path(gfile)
                txt_is_012 = bool(
                    txt_matrix_path is not None and _txt_is_discrete_012_matrix(txt_matrix_path)
                )
                if not txt_is_012:
                    txt_force_eig = True
                    rsvd_input = prepare_cli_input_cache(
                        str(gfile),
                        snps_only=bool(rsvd_snps_only),
                        delimiter=txt_delim,
                        prefer_plink_for_txt=False,
                        threads=int(args.thread),
                    )

            force_cache_bed = bool(args.vcf or args.hmp or (args.file and _is_txt_like_input_path(gfile) and not txt_force_eig))
            if force_cache_bed:
                rsvd_input = prepare_cli_input_cache(
                    str(gfile),
                    snps_only=bool(rsvd_snps_only),
                    delimiter=txt_delim,
                    prefer_plink_for_txt=True,
                    threads=int(args.thread),
                )

            algo_name = "Eigen-Decomposition" if txt_force_eig else "Random-SVD"
            if txt_force_eig:
                meta_sample_ids, meta_n_snps = inspect_genotype_file(
                    rsvd_input,
                    snps_only=bool(rsvd_snps_only),
                    maf=float(args.maf),
                    missing_rate=float(args.geno),
                )
                with CliStatus(
                    genotype_load_status_open(load_src_disp),
                    enabled=use_spinner,
                    use_process=True,
                ) as task:
                    try:
                        grm, samples, algo_snp = build_grm_streaming_for_pca(
                            genofile=rsvd_input,
                            maf_threshold=args.maf,
                            max_missing_rate=args.geno,
                            memory_mb=memory_mb,
                            inspected_meta=(meta_sample_ids, meta_n_snps),
                            logger=None,
                            emit_progress_done=False,
                            snps_only=bool(rsvd_snps_only),
                            threads=int(args.thread),
                        )
                    except Exception:
                        task.fail(genotype_load_status_fail(load_src_disp))
                        raise
                    algo_n = int(len(samples))
                    task.complete(
                        genotype_load_status_done(
                            load_src_disp,
                            n_samples=algo_n,
                            n_snps=int(algo_snp),
                        )
                    )

                def _run_txt_force_eig_only():
                    eigenvec_local, eigenval_local, evd_backend_local = eigendecompose_grm(
                        grm,
                        logger=None,
                        threads=int(args.thread),
                    )
                    return eigenvec_local, eigenval_local, evd_backend_local

                (eigenvec, eigenval, evd_backend), algo_elapsed_s = _run_with_algo_progress(
                    algo_name,
                    _run_txt_force_eig_only,
                    logger=logger,
                    n_samples=algo_n,
                    n_snps=int(algo_snp),
                )
                jx_pkg.maybe_emit_macos_eigh_fallback_hint(
                    evd_backend=str(evd_backend),
                    logger=logger,
                )
                _log_pca_eigh_backend(logger, str(evd_backend))
            else:
                if not hasattr(jxrs, "admx_rsvd_stream_sample"):
                    raise RuntimeError(
                        "Rust extension is missing admx_rsvd_stream_sample. "
                        "Rebuild/install JanusX extension first."
                    )
                with CliStatus(
                    genotype_load_status_open(load_src_disp),
                    enabled=use_spinner,
                    use_process=True,
                ) as task:
                    try:
                        sample_ids, algo_snp = inspect_genotype_file(
                            rsvd_input,
                            snps_only=bool(rsvd_snps_only),
                            maf=float(args.maf),
                            missing_rate=float(args.geno),
                        )
                        samples = np.asarray(sample_ids, dtype=str)
                    except Exception:
                        task.fail(genotype_load_status_fail(load_src_disp))
                        raise
                    algo_n = int(samples.shape[0])
                    task.complete(
                        genotype_load_status_done(
                            load_src_disp,
                            n_samples=algo_n,
                            n_snps=int(algo_snp),
                        )
                    )

                mmap_window_mb = _common_resolve_decode_mmap_window_mb(
                    rsvd_input,
                    len(samples),
                    int(algo_snp),
                    memory_mb,
                    buffers=_PCA_WORKING_BUFFERS_STREAM,
                )

                def _run_rsvd_only():
                    with _bed_block_target_env(memory_mb):
                        run_fn = _run_rsvd_subprocess if _use_rsvd_subprocess() else _run_rsvd_direct
                        call_kwargs = dict(
                            genotype_path=str(rsvd_input),
                            dim=int(args.dim),
                            seed=int(args.rsvd_seed),
                            power=int(args.rsvd_power),
                            tol=float(args.rsvd_tol),
                            snps_only=bool(rsvd_snps_only),
                            maf=float(args.maf),
                            missing_rate=float(args.geno),
                            mmap_window_mb=int(mmap_window_mb) if mmap_window_mb is not None else 0,
                            force_packed_bed=bool(_is_plink_prefix_path(rsvd_input)),
                            threads=int(args.thread),
                        )
                        if run_fn is _run_rsvd_subprocess:
                            call_kwargs["progress_callback"] = None
                        return run_fn(**call_kwargs)

                (eigenval, eigenvec, rsvd_total_variance), algo_elapsed_s = _run_with_algo_progress(
                    algo_name,
                    _run_rsvd_only,
                    logger=logger,
                    n_samples=algo_n,
                    n_snps=int(algo_snp),
                )
                if eigenvec.shape[0] != samples.shape[0]:
                    samples = samples[: eigenvec.shape[0]]
            log_success(
                logger,
                f"{algo_name} (n={algo_n}, snp={int(algo_snp)}) "
                f"...Finished [{format_elapsed(algo_elapsed_s)}]",
                force_color=True,
            )
        else:
            with CliStatus(
                genotype_load_status_open(load_src_disp),
                enabled=use_spinner,
                use_process=True,
            ) as task:
                try:
                    grm, samples, algo_snp = build_grm_streaming_for_pca(
                        genofile=gfile,
                        maf_threshold=args.maf,
                        max_missing_rate=args.geno,
                        memory_mb=memory_mb,
                        inspected_meta=None,
                        logger=None,
                        emit_progress_done=False,
                        snps_only=bool(args.vcf),
                        threads=int(args.thread),
                    )
                except Exception:
                    task.fail(genotype_load_status_fail(load_src_disp))
                    raise
                algo_n = int(len(samples))
                task.complete(
                    genotype_load_status_done(
                        load_src_disp,
                        n_samples=algo_n,
                        n_snps=int(algo_snp),
                    )
                )

            def _run_memmap_eig_only():
                eigenvec_local, eigenval_local, evd_backend_local = eigendecompose_grm(
                    grm,
                    logger=None,
                    threads=int(args.thread),
                )
                return eigenvec_local, eigenval_local, evd_backend_local

            (eigenvec, eigenval, evd_backend), algo_elapsed_s = _run_with_algo_progress(
                "Eigen-Decomposition",
                _run_memmap_eig_only,
                logger=logger,
                n_samples=algo_n,
                n_snps=int(algo_snp),
            )
            jx_pkg.maybe_emit_macos_eigh_fallback_hint(
                evd_backend=str(evd_backend),
                logger=logger,
            )
            _log_pca_eigh_backend(logger, str(evd_backend))
            log_success(
                logger,
                f"Eigen-Decomposition (n={algo_n}, snp={int(algo_snp)}) "
                f"...Finished [{format_elapsed(algo_elapsed_s)}]",
                force_color=True,
            )

        # Save core PCA results
        df_vec = pd.DataFrame(
            np.column_stack([samples.astype(str), eigenvec[:, : args.dim]])
        )
        df_vec.to_csv(
            f"{outprefix}.eigenvec",
            sep="\t",
            header=False,
            index=False,
            float_format="%.6f",
        )
        explained_ratio = _compute_explained_ratio(
            eigenval,
            total_variance=rsvd_total_variance,
        )
        _write_eigenval_table(
            f"{outprefix}.eigenval",
            eigenval,
            total_variance=rsvd_total_variance,
        )
        _log_saved_eigen_results(logger, outprefix)

    # --- Case 2: GRM prefix -> load GRM -> PCA ---
    elif args.grm:
        logger.info("* PCA from precomputed GRM.")
        # Support both prefix-based GRM and direct GRM file paths
        grm_file = None
        if os.path.isfile(gfile):
            grm_file = gfile
        else:
            for cand in (
                f"{gfile}.cGRM.txt",
                f"{gfile}.sGRM.txt",
                f"{gfile}.cGRM.npy",
                f"{gfile}.sGRM.npy",
                f"{gfile}.grm.txt",
                f"{gfile}.grm.npy",
            ):
                if os.path.exists(cand):
                    grm_file = cand
                    break

        if grm_file is None:
            raise ValueError(
                "GRM matrix (.cGRM/.sGRM/.grm with .txt/.npy, or direct path) not found."
            )

        grm_src = os.path.basename(str(grm_file).rstrip("/\\")) or str(grm_file)
        use_rust_file_eigh = bool(hasattr(jxrs, "rust_eigh_from_matrix_file_f64"))
        grm = None
        with CliStatus(f"Loading GRM from {grm_src}...", enabled=use_spinner) as task:
            try:
                id_path = f"{grm_file}.id"
                samples = _read_id_file(id_path, logger, "GRM", show_status=False)
                if samples is None:
                    raise ValueError(f"GRM ID file not found: {id_path}")
                if not use_rust_file_eigh:
                    if grm_file.endswith(".npy"):
                        grm = np.load(grm_file)
                    else:
                        grm = np.genfromtxt(grm_file)
                    if grm.shape[0] != len(samples):
                        raise ValueError(
                            f"GRM size mismatch: {grm.shape[0]} != {len(samples)} (ID count)."
                        )
            except Exception:
                task.fail(f"Loading GRM from {grm_src} ...Failed")
                raise
            task.complete(f"Loading GRM from {grm_src} (n={len(samples)})")

        # Eigen decomposition: keep one dynamic progress line only.
        with CliStatus(
            "Computing Eigen-Decomposition...",
            enabled=use_spinner,
            use_process=True,
        ) as task:
            try:
                if use_rust_file_eigh:
                    eigenvec, eigenval, evd_backend, n_rust = eigendecompose_grm_file(
                        grm_file,
                        logger=None,
                        threads=int(args.thread),
                    )
                    if int(n_rust) != int(len(samples)):
                        raise ValueError(
                            f"GRM size mismatch: Rust n={int(n_rust)} != {len(samples)} (ID count)."
                        )
                else:
                    if grm is None:
                        raise RuntimeError("GRM matrix is not loaded for PCA eigendecomposition.")
                    eigenvec, eigenval, evd_backend = eigendecompose_grm(
                        grm,
                        logger=None,
                        threads=int(args.thread),
                    )
            except Exception:
                task.fail("Computing Eigen-Decomposition ...Failed")
                raise
            task.complete("Computing Eigen-Decomposition ...Finished")
        jx_pkg.maybe_emit_macos_eigh_fallback_hint(
            evd_backend=str(evd_backend),
            logger=logger,
        )
        _log_pca_eigh_backend(logger, str(evd_backend))

        # Save core PCA results
        df_vec = pd.DataFrame(
            np.column_stack([samples.astype(str), eigenvec[:, : args.dim]])
        )
        df_vec.to_csv(
            f"{outprefix}.eigenvec",
            sep="\t",
            header=False,
            index=False,
            float_format="%.6f",
        )
        explained_ratio = _compute_explained_ratio(eigenval)
        _write_eigenval_table(f"{outprefix}.eigenval", eigenval)
        _log_saved_eigen_results(logger, outprefix)

    # --- Case 3: qcov prefix -> load PC results only for plotting ---
    elif args.qcov:
        logger.info("* Using existing PC results for visualization only.")
        qsrc = os.path.basename(str(gfile).rstrip("/\\")) or str(gfile)
        with CliStatus(f"Loading existing PC results from {qsrc}...", enabled=use_spinner) as task:
            try:
                samples, eigenvec = _read_matrix_with_ids(f"{gfile}.eigenvec", logger, "Eigenvec")
                eigenval, explained_ratio = _load_eigenval_table(f"{gfile}.eigenval")
            except Exception:
                task.fail(f"Loading existing PC results from {qsrc} ...Failed")
                raise
            n_samples = int(eigenvec.shape[0]) if isinstance(eigenvec, np.ndarray) else 0
            n_comp = int(eigenvec.shape[1]) if isinstance(eigenvec, np.ndarray) and eigenvec.ndim >= 2 else 0
            task.complete(
                f"Loading existing PC results from {qsrc} (n={n_samples}, nPC={n_comp})"
            )
        logger.info(
            f"Loaded PC results from {gfile}.eigenvec(.id/.eigenval) in "
            f"{round(time.time() - t_loading, 3)} seconds"
        )

    # Safety check
    if eigenvec is None or eigenval is None or samples is None:
        raise RuntimeError("PCA results are not available; check input arguments.")

    # ------------------------- Visualization -------------------------
    if args.plot or args.plot3D:
        _, plt, PCSHOW, _legacy_color_set = _get_pca_plot_backend()
        if explained_ratio is None:
            explained_ratio = _compute_explained_ratio(eigenval)
        exp = 100.0 * np.asarray(explained_ratio, dtype=np.float64).reshape(-1)
        df_pc = pd.DataFrame(
            eigenvec[:, :3],
            index=samples,
            columns=[
                f"PC{i + 1}({round(float(exp[i]), 2)}%)" for i in range(3)
            ],
        )

        group = None
        group_order = None
        textanno = None
        plot_colors = None
        if args.group:
            group_df, group, textanno = load_group_table(args.group)
            group_order = extract_group_order(group_df, group)
            df_pc = df_pc.join(group_df, how="left")
            plot_colors = _resolve_palette_colors(
                args.palette_spec,
                len(resolve_group_levels(df_pc, group, group_order)),
            )

    if args.plot:
        logger.info("* Generating 2D PCA scatter plots...")

        pcshow = PCSHOW(df_pc)
        fig = plt.figure(figsize=(10, 4), dpi=300)
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(df_pc.columns[0])
        ax1.set_ylabel(df_pc.columns[1])
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(df_pc.columns[0])
        ax2.set_ylabel(df_pc.columns[2])

        pcshow.pcplot(
            df_pc.columns[0],
            df_pc.columns[1],
            group=group,
            group_order=group_order,
            ax=ax1,
            color_set=plot_colors,
            anno_tag=textanno,
        )
        pcshow.pcplot(
            df_pc.columns[0],
            df_pc.columns[2],
            group=group,
            group_order=group_order,
            ax=ax2,
            color_set=plot_colors,
            anno_tag=textanno,
        )

        plt.tight_layout()
        out_pdf = f"{outprefix}.eigenvec.2D.pdf"
        plt.savefig(out_pdf, transparent=True)
        plt.close()
        log_success(logger, f"2D PCA figure saved to {format_path_for_display(out_pdf)}")

    if args.plot3D:
        logger.info("* Generating 3D PCA rotating GIF...")
        pcshow = PCSHOW(df_pc)
        out_gif = f"{outprefix}.eigenvec.3D.gif"
        pcshow.pcplot3D_gif(
            df_pc.columns[0],
            df_pc.columns[1],
            df_pc.columns[2],
            group=group,
            group_order=group_order,
            anno_tag=textanno,
            color_set=plot_colors,
            out_gif=out_gif,
        )
        log_success(logger, f"3D PCA GIF saved to {format_path_for_display(out_gif)}")

    # ------------------------- Final logging -------------------------
    lt = time.localtime()
    endinfo = (
        f"\nFinished PCA. Total wall time: {round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    log_success(logger, endinfo)


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
