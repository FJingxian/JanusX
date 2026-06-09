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
  - {prefix}.eigenval      : eigenvalues (variance along each PC)
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
import multiprocessing as mp
import tempfile
import shutil
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import psutil
import janusx as jx_pkg
from janusx import janusx as jxrs

mpl.use("Agg")
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
from janusx.bioplotkit.sci_set import color_set
from janusx.bioplotkit.pcshow import PCSHOW
from janusx.gfreader import (
    calc_decode_block_rows_from_memory_mb,
    load_genotype_chunks,
    inspect_genotype_file,
    auto_mmap_window_mb,
    prepare_cli_input_cache,
)
from ._common.log import setup_logging
from ._common.config_render import emit_cli_configuration
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
    format_path_for_display,
)
from ._common.progress import ProgressAdapter
from ._common.status import CliStatus, format_elapsed, log_success, stdout_is_tty
from ._common.genocache import configure_genotype_cache_from_out
from ._common.genoio import (
    determine_genotype_source as _determine_genotype_source,
    read_id_file as _read_id_file,
    read_matrix_with_ids as _read_matrix_with_ids,
    strip_default_prefix_suffix,
)
from ._common.threads import detect_effective_threads

# ======================================================================
# Helpers: GRM-based PCA (aligned with GWAS module)
# ======================================================================

DEFAULT_BED_MEMORY_MB = 512.0


def _normalize_memory_mb(memory_mb: Union[int, float, None]) -> float:
    if memory_mb is None:
        return float(DEFAULT_BED_MEMORY_MB)
    mb = float(memory_mb)
    if (not np.isfinite(mb)) or mb <= 0.0:
        raise ValueError(f"--memory must be a finite value > 0, got {memory_mb}")
    return float(mb)


@contextmanager
def _bed_block_target_env(memory_mb: Union[int, float, None]) -> None:
    prev = os.environ.get("JX_BED_BLOCK_TARGET_MB")
    if memory_mb is not None:
        os.environ["JX_BED_BLOCK_TARGET_MB"] = f"{float(memory_mb):.6g}"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("JX_BED_BLOCK_TARGET_MB", None)
        else:
            os.environ["JX_BED_BLOCK_TARGET_MB"] = prev

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
    eval_path: str,
    evec_path: str,
) -> None:
    try:
        if bool(force_packed_bed):
            os.environ["JANUSX_RSVD_BED_PACKED"] = "1"
        eval_raw, evec_raw = jxrs.admx_rsvd_stream_sample(
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
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    ctx = _select_rsvd_mp_context()
    q = ctx.Queue(maxsize=1)
    tmpdir = tempfile.mkdtemp(prefix="janusx_pca_rsvd_")
    eval_path = os.path.join(tmpdir, "eval.npy")
    evec_path = os.path.join(tmpdir, "evec.npy")
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
            str(eval_path),
            str(evec_path),
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
        return np.asarray(eval_np, dtype=np.float32), np.asarray(evec_np, dtype=np.float32)
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


def _log_pca_eigh_backend(logger, evd_backend: str | None) -> None:
    backend = str(evd_backend).strip()
    if backend == "":
        return
    try:
        logger.info(f"EIGH backend: {backend}")
    except Exception:
        pass


def _count_effective_snps_from_input(
    genofile: str,
    *,
    maf_threshold: float,
    max_missing_rate: float,
    snps_only: bool,
    chunk_size: int = 100_000,
) -> int:
    """
    Count effective SNP rows after MAF/missing filtering from current input.

    For RSVD with VCF/HMP/TXT inputs, `genofile` is already a PLINK BED cache prefix,
    so this count is read directly from cached BED chunks under the same filter setup.
    """
    eff_m = 0
    scan_chunk = int(max(10_000, int(chunk_size)))
    for genosub, _sites in load_genotype_chunks(
        genofile,
        chunk_size=scan_chunk,
        maf=float(maf_threshold),
        missing_rate=float(max_missing_rate),
        impute=False,
        snps_only=bool(snps_only),
    ):
        eff_m += int(genosub.shape[0])
    return int(eff_m)


def build_grm_streaming_for_pca(
    genofile: str,
    maf_threshold: float = 0.02,
    max_missing_rate: float = 0.05,
    memory_mb: float = DEFAULT_BED_MEMORY_MB,
    inspected_meta: Optional[tuple[list[str], int]] = None,
    logger=None,
    emit_progress_done: bool = True,
    snps_only: bool = False,
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
    block_rows = calc_decode_block_rows_from_memory_mb(
        n_samples,
        float(memory_mb),
        buffers=2,
        max_rows=max(1, int(n_snps)),
    )
    block_rows = max(1, int(block_rows if block_rows is not None else 1))
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

    mmap_window_mb = auto_mmap_window_mb(rust_input, n_samples, n_snps, float(memory_mb))
    try:
        with _bed_block_target_env(memory_mb):
            grm_raw, eff_m_raw, stream_n_raw = jxrs.grm_stream_bed_f32(
                str(rust_input),
                method=1,
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                block_cols=max(1, int(block_rows)),
                threads=0,
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


def eigendecompose_grm(grm: np.ndarray, logger=None) -> tuple[np.ndarray, np.ndarray, str]:
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
    threads = max(1, int(detect_effective_threads()))

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
    if logger is not None:
        logger.info(f"Eigen decomposition finished (backend={backend}).")

    return eigvec, eigval, backend


def eigendecompose_grm_file(grm_file: str, logger=None) -> tuple[np.ndarray, np.ndarray, str, int]:
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
    threads = max(1, int(detect_effective_threads()))
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
    if logger is not None:
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
    geno_group.add_argument(
        "-vcf", "--vcf", type=str,
        help="Input genotype file in VCF format (.vcf or .vcf.gz).",
    )
    geno_group.add_argument(
        "-hmp", "--hmp", type=str,
        help="Input genotype file in HMP format (.hmp or .hmp.gz).",
    )
    geno_group.add_argument(
        "-bfile", "--bfile", type=str,
        help=(
            "Input genotype in PLINK binary format "
            "(prefix for .bed, .bim, .fam)."
        ),
    )
    geno_group.add_argument(
        "-file", "--file", type=str,
        help=(
            "Input genotype numeric matrix (.txt/.tsv/.csv/.npy) or prefix. "
            "Requires sibling prefix.id. Optional site metadata: prefix.site or prefix.bim."
        ),
    )
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
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory for PCA results (default: current directory).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Prefix of output files (default: inferred from input file name).",
    )
    optional_group.add_argument(
        "-dim", "--dim", type=int, default=3,
        help="Number of leading principal components to output (default: %(default)s).",
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
        "-memory", "--memory", type=float, default=DEFAULT_BED_MEMORY_MB,
        help=(
            "Target decode working-set size in MB for BED memmap/packed kernels "
            "(default: %(default)s)."
        ),
    )
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
        "-color", "--color", type=int, default=-1,
        help="Color palette index for PCA plots, 0-6; -1 uses auto palette "
             "(default: %(default)s).",
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
    args.memory = _normalize_memory_mb(args.memory)

    # Keep index for logging and validate after logger is ready
    palette_idx = int(args.color)

    # ------------------------- Logging -------------------------
    os.makedirs(args.out, 0o755, exist_ok=True)
    configure_genotype_cache_from_out(args.out)
    log_path = f"{outprefix}.pca.log"
    logger = setup_logging(log_path)
    # Keep algorithm status visible immediately.
    use_spinner = True
    # if genotype_input_mode and (not bool(args.mmap_limit)):
    #     logger.info("PCA backend policy: memmap (default).")

    if palette_idx == -1:
        args.color = None
    elif 0 <= palette_idx <= 6:
        args.color = color_set[palette_idx]
    else:
        logger.error("Color set index out of range; please use 0-6 or -1.")
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
                    ("Memory MB", args.memory),
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
            cfg_rows.extend([("GRM prefix", gfile), ("Output PCs", f"top {args.dim}")])
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
                    ("Color palette", "auto" if palette_idx == -1 else f"index {palette_idx}"),
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
    samples = None

    # --- Case 1: VCF / BFILE -> memmap GRM -> PCA (aligned with GWAS) ---
    if args.vcf or args.hmp or args.file or args.bfile:
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
                    )

            force_cache_bed = bool(args.vcf or args.hmp or (args.file and _is_txt_like_input_path(gfile) and not txt_force_eig))
            if force_cache_bed:
                rsvd_input = prepare_cli_input_cache(
                    str(gfile),
                    snps_only=bool(rsvd_snps_only),
                    delimiter=txt_delim,
                    prefer_plink_for_txt=True,
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
                    f"Loading genotype from {load_src_disp}...",
                    enabled=use_spinner,
                    use_process=True,
                ) as task:
                    try:
                        grm, samples, algo_snp = build_grm_streaming_for_pca(
                            genofile=rsvd_input,
                            maf_threshold=args.maf,
                            max_missing_rate=args.geno,
                            memory_mb=float(args.memory),
                            inspected_meta=(meta_sample_ids, meta_n_snps),
                            logger=None,
                            emit_progress_done=False,
                            snps_only=bool(rsvd_snps_only),
                        )
                    except Exception:
                        task.fail(f"Loading genotype from {load_src_disp} ...Failed")
                        raise
                    algo_n = int(len(samples))
                    task.complete(
                        f"Loading genotype from {load_src_disp} (n={algo_n}, nSNP={int(algo_snp)})"
                    )

                def _run_txt_force_eig_only():
                    eigenvec_local, eigenval_local, evd_backend_local = eigendecompose_grm(grm, logger=None)
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
                    f"Loading genotype from {load_src_disp}...",
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
                        task.fail(f"Loading genotype from {load_src_disp} ...Failed")
                        raise
                    algo_n = int(samples.shape[0])
                    task.complete(
                        f"Loading genotype from {load_src_disp} (n={algo_n}, nSNP={int(algo_snp)})"
                    )

                mmap_window_mb = auto_mmap_window_mb(
                    rsvd_input,
                    len(samples),
                    int(algo_snp),
                    float(args.memory),
                )

                def _run_rsvd_only():
                    with _bed_block_target_env(float(args.memory)):
                        return _run_rsvd_subprocess(
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
                            progress_callback=None,
                        )

                (eigenval, eigenvec), algo_elapsed_s = _run_with_algo_progress(
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
                f"Loading genotype from {load_src_disp}...",
                enabled=use_spinner,
                use_process=True,
            ) as task:
                try:
                    grm, samples, algo_snp = build_grm_streaming_for_pca(
                        genofile=gfile,
                        maf_threshold=args.maf,
                        max_missing_rate=args.geno,
                        memory_mb=float(args.memory),
                        inspected_meta=None,
                        logger=None,
                        emit_progress_done=False,
                        snps_only=bool(args.vcf),
                    )
                except Exception:
                    task.fail(f"Loading genotype from {load_src_disp} ...Failed")
                    raise
                algo_n = int(len(samples))
                task.complete(
                    f"Loading genotype from {load_src_disp} (n={algo_n}, nSNP={int(algo_snp)})"
                )

            def _run_memmap_eig_only():
                eigenvec_local, eigenval_local, evd_backend_local = eigendecompose_grm(grm, logger=None)
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
        np.savetxt(
            f"{outprefix}.eigenval",
            eigenval,
            fmt="%.2f",
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
                    )
                    if int(n_rust) != int(len(samples)):
                        raise ValueError(
                            f"GRM size mismatch: Rust n={int(n_rust)} != {len(samples)} (ID count)."
                        )
                else:
                    if grm is None:
                        raise RuntimeError("GRM matrix is not loaded for PCA eigendecomposition.")
                    eigenvec, eigenval, evd_backend = eigendecompose_grm(grm, logger=None)
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
        np.savetxt(
            f"{outprefix}.eigenval",
            eigenval,
            fmt="%.2f",
        )
        _log_saved_eigen_results(logger, outprefix)

    # --- Case 3: qcov prefix -> load PC results only for plotting ---
    elif args.qcov:
        logger.info("* Using existing PC results for visualization only.")
        qsrc = os.path.basename(str(gfile).rstrip("/\\")) or str(gfile)
        with CliStatus(f"Loading existing PC results from {qsrc}...", enabled=use_spinner) as task:
            try:
                samples, eigenvec = _read_matrix_with_ids(f"{gfile}.eigenvec", logger, "Eigenvec")
                eigenval = np.genfromtxt(f"{gfile}.eigenval")
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
        exp = 100 * eigenval / np.sum(eigenval)
        df_pc = pd.DataFrame(
            eigenvec[:, :3],
            index=samples,
            columns=[
                f"PC{i + 1}({round(float(exp[i]), 2)}%)" for i in range(3)
            ],
        )

        group = None
        textanno = None
        if args.group:
            group_df, group, textanno = load_group_table(args.group)
            df_pc = df_pc.join(group_df, how="left")

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
            ax=ax1,
            color_set=args.color,
            anno_tag=textanno,
        )
        pcshow.pcplot(
            df_pc.columns[0],
            df_pc.columns[2],
            group=group,
            ax=ax2,
            color_set=args.color,
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
            anno_tag=textanno,
            color_set=args.color,
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
