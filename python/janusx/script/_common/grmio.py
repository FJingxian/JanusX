from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np

from .pathcheck import safe_expanduser, safe_resolve


_WARNED_GENOTYPE_CACHE_FALLBACK_KEYS: set[str] = set()

GENOTYPE_CACHE_FALLBACK_PREFIX = "No write permission for genotype-side cache directory:"
GENOTYPE_CACHE_PERMISSION_RETRY_PREFIX = "Failed to build/read genotype-side cache at"
GENOTYPE_CACHE_STAGED_PREFIX = "Using staged input under JANUSX_CACHE_DIR:"


def grm_cache_src_name(path: str) -> str:
    raw = str(path or "").replace("\\", "/").rstrip("/")
    if raw == "":
        return ""
    base = os.path.basename(raw)
    return base if base != "" else raw


def grm_load_status_open(path: str) -> str:
    return f"Loading GRM from {grm_cache_src_name(path)}..."


def grm_load_status_fail(path: str) -> str:
    return f"Loading GRM from {grm_cache_src_name(path)} ...Failed"


def grm_load_status_done(path: str, n_samples: int) -> str:
    return f"Loading GRM from {grm_cache_src_name(path)} (n={int(n_samples)})"


def grm_text_materialized_message(path: str) -> str:
    return f"Materialized GRM text cache to memmap NPY: {path}"


def genotype_cache_fallback_message(geno_dir: str, fallback_dir: str) -> str:
    return (
        f"{GENOTYPE_CACHE_FALLBACK_PREFIX} "
        f"{geno_dir}. Falling back to cache directory from "
        f"JANUSX_CACHE_DIR: {fallback_dir}"
    )


def genotype_cache_permission_retry_message(primary_path: str, fallback_dir: str) -> str:
    return (
        f"{GENOTYPE_CACHE_PERMISSION_RETRY_PREFIX} '{primary_path}' due to permission issue. "
        f"Falling back to JANUSX_CACHE_DIR: {fallback_dir}"
    )


def genotype_cache_staged_input_message(geno_dir: str, stage_root: str) -> str:
    return (
        f"{GENOTYPE_CACHE_FALLBACK_PREFIX} "
        f"{geno_dir}. {GENOTYPE_CACHE_STAGED_PREFIX} {stage_root}"
    )


def is_writable_dir(path: str) -> bool:
    d = os.path.abspath(path or ".")
    return os.path.isdir(d) and os.access(d, os.W_OK | os.X_OK)


def _warn_genotype_cache_fallback_once(
    key: str,
    msg: str,
    logger: logging.Logger | None = None,
) -> None:
    if key in _WARNED_GENOTYPE_CACHE_FALLBACK_KEYS:
        return
    _WARNED_GENOTYPE_CACHE_FALLBACK_KEYS.add(key)
    if logger is not None:
        logger.warning(msg)
    else:
        warnings.warn(msg, RuntimeWarning, stacklevel=2)


def resolve_genotype_cache_dir(
    genofile: str,
    *,
    cache_dir: str | None = None,
    logger: logging.Logger | None = None,
    emit_warning: bool = True,
    warning_collector: list[str] | None = None,
) -> str:
    geno_dir = os.path.abspath(os.path.dirname(str(genofile)) or ".")
    if is_writable_dir(geno_dir):
        return geno_dir

    preferred = (str(cache_dir).strip() if cache_dir is not None else "")
    if preferred == "":
        preferred = os.environ.get("JANUSX_CACHE_DIR", "").strip()
    if preferred:
        fallback_dir = str(safe_resolve(preferred))
        try:
            os.makedirs(fallback_dir, mode=0o755, exist_ok=True)
        except Exception:
            pass
        if is_writable_dir(fallback_dir):
            if bool(emit_warning):
                msg = genotype_cache_fallback_message(geno_dir, fallback_dir)
                if warning_collector is not None:
                    warning_collector.append(msg)
                else:
                    _warn_genotype_cache_fallback_once(
                        f"{geno_dir}->{fallback_dir}",
                        msg,
                        logger=logger,
                    )
            return fallback_dir

    return geno_dir


def genotype_cache_prefix(
    genofile: str,
    *,
    snps_only: bool = True,
    cache_dir: str | None = None,
    logger: logging.Logger | None = None,
    warning_collector: list[str] | None = None,
) -> str:
    """
    Construct a genotype-side cache prefix.

    The cache root prefers the genotype directory itself; if that location is
    not writable, it falls back to `cache_dir` or `JANUSX_CACHE_DIR`.
    """
    del snps_only  # kept for signature compatibility with existing callers
    base = os.path.basename(str(genofile))
    low = base.lower()
    if low.endswith(".vcf.gz") or low.endswith(".hmp.gz"):
        base = base[: -len(".vcf.gz")] if low.endswith(".vcf.gz") else base[: -len(".hmp.gz")]
    else:
        for ext in (".vcf", ".hmp", ".txt", ".tsv", ".csv", ".npy"):
            if low.endswith(ext):
                base = base[: -len(ext)]
                break
    cache_root = resolve_genotype_cache_dir(
        genofile,
        cache_dir=cache_dir,
        logger=logger,
        warning_collector=warning_collector,
    )
    return os.path.join(cache_root, f"~{base}").replace("\\", "/")


def format_grm_cache_num(v: float | int) -> str:
    x = float(v)
    s = f"{x:.6g}"
    if "e" in s or "E" in s:
        s = f"{x:.12f}".rstrip("0").rstrip(".")
    if s in {"", "-0"}:
        s = "0"
    return s


def grm_filter_param_tag(
    *,
    maf: float,
    geno: float,
    snps_only: bool,
) -> str:
    return (
        f".maf{format_grm_cache_num(maf)}"
        f".geno{format_grm_cache_num(geno)}"
        f".snp{1 if bool(snps_only) else 0}"
    )


def grm_cache_prefix_with_params(
    genofile: str,
    *,
    maf: float,
    geno: float,
    snps_only: bool,
    cache_dir: str | None = None,
    logger: logging.Logger | None = None,
    warning_collector: list[str] | None = None,
) -> str:
    base = genotype_cache_prefix(
        genofile,
        snps_only=bool(snps_only),
        cache_dir=cache_dir,
        logger=logger,
        warning_collector=warning_collector,
    )
    return f"{base}{grm_filter_param_tag(maf=float(maf), geno=float(geno), snps_only=bool(snps_only))}"


def grm_method_cache_tag(mgrm: str, *, legacy: bool = False) -> str:
    s = str(mgrm).strip()
    if s == "1":
        return "grm1" if legacy else "cGRM"
    if s == "2":
        return "grm2" if legacy else "sGRM"
    raw = str(s).replace("\\", "/").rstrip("/")
    base = os.path.basename(raw)
    low = base.lower()
    if ".cgrm" in low:
        return "grm1" if legacy else "cGRM"
    if ".sgrm" in low:
        return "grm2" if legacy else "sGRM"
    stem = base
    while True:
        stem_low = stem.lower()
        matched = False
        for ext in (".spgrm", ".jxgrm", ".npy", ".txt", ".tsv", ".csv", ".bin", ".gz"):
            if stem_low.endswith(ext):
                stem = stem[: -len(ext)]
                matched = True
                break
        if not matched:
            break
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if stem == "":
        stem = "grm"
    try:
        resolved = str(safe_resolve(raw))
    except Exception:
        resolved = raw
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:10]
    return f"{stem}.{digest}"


def grm_cache_paths(
    cache_prefix_with_params: str,
    *,
    mgrm: str,
) -> tuple[str, str]:
    tag = grm_method_cache_tag(str(mgrm), legacy=False)
    grm_path = f"{cache_prefix_with_params}.{tag}.npy"
    return grm_path, f"{grm_path}.id"


def grm_cache_paths_legacy(
    cache_prefix_with_params: str,
    *,
    mgrm: str,
) -> tuple[str, str]:
    tag = grm_method_cache_tag(str(mgrm), legacy=True)
    grm_path = f"{cache_prefix_with_params}.{tag}.npy"
    return grm_path, f"{grm_path}.id"


@contextmanager
def grm_cache_lock(lock_key: str, timeout_s: float = 7200.0, poll_s: float = 0.2):
    lock_path = f"{lock_key}.lock"
    start = time.monotonic()
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
            break
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > timeout_s:
                    os.remove(lock_path)
                    continue
            except Exception:
                pass
            if (time.monotonic() - start) > timeout_s:
                raise TimeoutError(f"Timeout waiting cache lock: {lock_path}")
            time.sleep(poll_s)
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        except Exception:
            pass


def _resolve_file_input_matrix(genofile: str) -> tuple[str | None, str | None]:
    path = str(safe_expanduser(str(genofile)))
    low = path.lower()
    if low.endswith(".npy"):
        return path[: -len(".npy")], path
    if low.endswith(".bin"):
        return path[: -len(".bin")], path
    for ext in (".txt", ".tsv", ".csv"):
        if low.endswith(ext):
            return path[: -len(ext)], path
    for ext in (".npy", ".bin", ".txt", ".tsv", ".csv"):
        cand = f"{path}{ext}"
        if os.path.isfile(cand):
            return path, cand
    return None, None


def _file_input_sidecars(prefix: str) -> list[str]:
    return [
        f"{prefix}.bin.id",
        f"{prefix}.id",
        f"{prefix}.bin.site",
        f"{prefix}.site",
        f"{prefix}.site.tsv",
        f"{prefix}.site.txt",
        f"{prefix}.site.csv",
    ]


def latest_genotype_mtime(genofile: str) -> float | None:
    """
    Return the latest modification time of genotype input files.

    - PLINK prefix: max mtime of .bed/.bim/.fam (if all exist)
    - FILE prefix : max mtime of matrix file, prefix.id, and optional site metadata
    - File input  : mtime of the file path itself
    """
    bed = f"{genofile}.bed"
    bim = f"{genofile}.bim"
    fam = f"{genofile}.fam"
    if all(os.path.isfile(p) for p in (bed, bim, fam)):
        return max(os.path.getmtime(bed), os.path.getmtime(bim), os.path.getmtime(fam))

    file_prefix, matrix_path = _resolve_file_input_matrix(genofile)
    if file_prefix and matrix_path:
        mtimes = [os.path.getmtime(matrix_path)]
        for cand in _file_input_sidecars(file_prefix):
            if os.path.isfile(cand):
                mtimes.append(os.path.getmtime(cand))
        return max(mtimes)

    if os.path.isfile(genofile):
        return os.path.getmtime(genofile)
    return None


def read_id_file(path: str) -> list[str]:
    out: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fr:
        for raw in fr:
            s = str(raw).strip()
            if s == "":
                continue
            out.append(s.split()[0])
    return out


def resolve_grm_id_path(grm_path: str, explicit: str | None = None) -> str | None:
    if explicit is not None and str(explicit).strip() != "":
        p = str(Path(str(explicit).strip()).expanduser())
        if not Path(p).is_file():
            raise FileNotFoundError(f"GRM ID file not found: {p}")
        return p

    direct = f"{grm_path}.id"
    if Path(direct).is_file():
        return direct

    p = Path(grm_path)
    txt_like = {".txt", ".tsv", ".csv", ".npy"}
    if p.suffix.lower() in txt_like:
        stem_cand = f"{str(p.with_suffix(''))}.id"
        if Path(stem_cand).is_file():
            return stem_cand
    return None


def load_grm_matrix(path: str) -> np.ndarray:
    low = str(path).lower()
    if low.endswith(".npy"):
        arr = np.asarray(np.load(path), dtype=np.float64)
    else:
        arr = np.asarray(np.genfromtxt(path, dtype=np.float64), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"GRM must be a square matrix, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("GRM matrix contains NaN/Inf values.")
    return arr


def grm_cache_text_path(npy_path: str) -> str:
    path = str(npy_path)
    if path.lower().endswith(".npy"):
        return f"{path[:-4]}.txt"
    return f"{path}.txt"


def load_square_grm_npy_memmap(
    path: str,
    *,
    expected_n: int,
    mmap_mode: str = "r",
    expected_dtype: np.dtype | type | None = None,
) -> np.ndarray:
    arr = np.load(str(path), mmap_mode=str(mmap_mode))
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"GRM must be square; got shape={arr.shape}")
    if int(arr.shape[0]) != int(expected_n):
        raise ValueError(
            f"GRM shape mismatch: got {arr.shape}, expected ({int(expected_n)}, {int(expected_n)})."
        )
    if expected_dtype is not None and np.dtype(arr.dtype) != np.dtype(expected_dtype):
        raise ValueError(
            f"GRM dtype mismatch: got {np.dtype(arr.dtype)}, expected {np.dtype(expected_dtype)}."
        )
    return arr


def convert_square_grm_txt_to_npy_memmap(
    *,
    txt_path: str,
    npy_path: str,
    n_samples: int,
    dtype: np.dtype | type = np.float64,
) -> None:
    rows_expected = int(max(1, int(n_samples)))
    out_dtype = np.dtype(dtype)
    tmp_path = f"{str(npy_path)}.tmp.{os.getpid()}"
    mm = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=out_dtype,
        shape=(rows_expected, rows_expected),
    )
    row_idx = 0
    try:
        with open(str(txt_path), "r", encoding="utf-8", errors="replace") as fr:
            for raw in fr:
                line = str(raw).strip()
                if line == "":
                    continue
                vals = np.fromstring(line.replace(",", " "), sep=" ", dtype=np.float64)
                if int(vals.size) != rows_expected:
                    raise ValueError(
                        f"GRM txt row width mismatch at row={row_idx}: got {int(vals.size)}, "
                        f"expected {rows_expected}"
                    )
                if not np.all(np.isfinite(vals)):
                    raise ValueError(f"GRM txt row contains NaN/Inf at row={row_idx}")
                if row_idx >= rows_expected:
                    raise ValueError(
                        f"GRM txt row count exceeds expected square size {rows_expected}"
                    )
                mm[row_idx, :] = vals
                row_idx += 1
        if row_idx != rows_expected:
            raise ValueError(
                f"GRM txt row count mismatch: got {row_idx}, expected {rows_expected}"
            )
        mm.flush()
    finally:
        del mm
    os.replace(tmp_path, str(npy_path))


def load_or_materialize_square_grm_cache(
    cache_npy_path: str,
    *,
    expected_n: int,
    mmap_mode: str = "r",
    expected_dtype: np.dtype | type | None = None,
    txt_fallback_path: str | None = None,
    npy_dtype: np.dtype | type = np.float64,
    cache_id_path: str | None = None,
    txt_id_path: str | None = None,
) -> tuple[np.ndarray | None, str | None]:
    path = str(cache_npy_path)
    if os.path.exists(path):
        try:
            return (
                load_square_grm_npy_memmap(
                    path,
                    expected_n=int(expected_n),
                    mmap_mode=str(mmap_mode),
                    expected_dtype=expected_dtype,
                ),
                "npy",
            )
        except Exception:
            try:
                os.remove(path)
            except Exception:
                pass

    txt_path = str(txt_fallback_path or grm_cache_text_path(path))
    if not os.path.exists(txt_path):
        return None, None

    convert_square_grm_txt_to_npy_memmap(
        txt_path=txt_path,
        npy_path=path,
        n_samples=int(expected_n),
        dtype=npy_dtype,
    )
    if cache_id_path is not None and str(cache_id_path).strip() != "":
        src_id = str(txt_id_path).strip() if txt_id_path is not None else ""
        if src_id == "":
            resolved = resolve_grm_id_path(txt_path, None)
            src_id = "" if resolved is None else str(resolved)
        if src_id != "" and os.path.isfile(src_id) and (not os.path.exists(str(cache_id_path))):
            shutil.copyfile(src_id, str(cache_id_path))
    return (
        load_square_grm_npy_memmap(
            path,
            expected_n=int(expected_n),
            mmap_mode=str(mmap_mode),
            expected_dtype=expected_dtype,
        ),
        "txt->npy",
    )


def load_and_align_grm(
    grm_path: str,
    target_ids: Sequence[str],
    *,
    grm_id_path: str | None = None,
    label: str = "GRM",
) -> tuple[np.ndarray, str | None]:
    target = [str(x) for x in target_ids]
    if len(target) == 0:
        raise ValueError(f"{label} alignment requires at least one target sample ID.")
    if len(set(target)) != len(target):
        raise ValueError(f"{label} alignment target sample IDs must be unique.")

    grm = load_grm_matrix(grm_path)
    id_path = resolve_grm_id_path(grm_path, grm_id_path)
    if id_path is None:
        if grm.shape[0] != len(target):
            raise ValueError(
                f"{label} shape {grm.shape} does not match target sample count {len(target)}, "
                "and no GRM ID file was found for reordering."
            )
        return np.asarray(grm, dtype=np.float64), None

    grm_ids = read_id_file(id_path)
    if len(grm_ids) != grm.shape[0]:
        raise ValueError(
            f"{label} ID count mismatch: matrix n={grm.shape[0]} but ID file has {len(grm_ids)} rows."
        )

    index: dict[str, int] = {}
    for i, sid in enumerate(grm_ids):
        if sid in index:
            raise ValueError(f"{label} ID file contains duplicate sample ID: {sid}")
        index[sid] = i

    missing = [sid for sid in target if sid not in index]
    if missing:
        preview = ", ".join(missing[:5])
        extra = "" if len(missing) <= 5 else f" ... (+{len(missing) - 5} more)"
        raise ValueError(f"{label} is missing target sample IDs: {preview}{extra}")

    order = np.asarray([index[sid] for sid in target], dtype=np.intp)
    if np.array_equal(order, np.arange(len(target), dtype=np.intp)) and grm.shape[0] == len(target):
        return np.asarray(grm, dtype=np.float64), id_path
    return np.asarray(grm[np.ix_(order, order)], dtype=np.float64), id_path
