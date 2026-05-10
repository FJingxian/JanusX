# -*- coding: utf-8 -*-
"""Cache/path helpers for GWAS workflow."""

from __future__ import annotations

import logging
import os
import time
import warnings
from contextlib import contextmanager
from typing import Optional, Union

from janusx.script._common.pathcheck import safe_resolve


_WARNED_GWAS_CACHE_FALLBACK_KEYS: set[str] = set()


def _is_writable_dir(path: str) -> bool:
    d = os.path.abspath(path or ".")
    return os.path.isdir(d) and os.access(d, os.W_OK | os.X_OK)


def _warn_gwas_cache_fallback_once(
    key: str,
    msg: str,
    logger: Union[logging.Logger, None] = None,
) -> None:
    if key in _WARNED_GWAS_CACHE_FALLBACK_KEYS:
        return
    _WARNED_GWAS_CACHE_FALLBACK_KEYS.add(key)
    if logger is not None:
        logger.warning(msg)
    else:
        warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _normalize_cache_warning_message(msg: str) -> str:
    s = str(msg).strip()
    prefix = "No write permission for genotype-side cache directory:"
    marker = ". Falling back to cache directory from JANUSX_CACHE_DIR:"
    if s.startswith(prefix) and marker in s:
        try:
            left, right = s.split(marker, 1)
            src = left[len(prefix):].strip()
            dst = right.strip()
            src_abs = str(safe_resolve(src))
            dst_abs = str(safe_resolve(dst))
            return f"{prefix} {src_abs}{marker} {dst_abs}"
        except Exception:
            return s
    return s


def _dedupe_cache_warning_messages(msgs: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in msgs:
        s = _normalize_cache_warning_message(str(m))
        if s == "" or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _is_cache_warning_message(msg: str) -> bool:
    s = str(msg).lower()
    if s == "":
        return False
    return (
        "cache" in s
        or "no write permission for genotype-side cache directory" in s
        or "janusx_cache_dir" in s
    )


def _resolve_gwas_cache_dir(
    genofile: str,
    *,
    cache_dir: Union[str, None] = None,
    logger: Union[logging.Logger, None] = None,
    emit_warning: bool = True,
    warning_collector: Union[list[str], None] = None,
) -> str:
    geno_dir = os.path.abspath(os.path.dirname(str(genofile)) or ".")
    if _is_writable_dir(geno_dir):
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
        if _is_writable_dir(fallback_dir):
            if bool(emit_warning):
                msg = (
                    "No write permission for genotype-side cache directory: "
                    f"{geno_dir}. Falling back to cache directory from "
                    f"JANUSX_CACHE_DIR: {fallback_dir}"
                )
                if warning_collector is not None:
                    warning_collector.append(msg)
                else:
                    _warn_gwas_cache_fallback_once(
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
    cache_dir: Union[str, None] = None,
    logger: Union[logging.Logger, None] = None,
    warning_collector: Union[list[str], None] = None,
) -> str:
    """
    Construct a cache prefix for GWAS GRM/Q caches.
    Prefer genotype directory; if not writable, fallback to JANUSX_CACHE_DIR.
    """
    base = os.path.basename(genofile)
    low = base.lower()
    if low.endswith(".vcf.gz") or low.endswith(".hmp.gz"):
        base = base[: -len(".vcf.gz")] if low.endswith(".vcf.gz") else base[: -len(".hmp.gz")]
    else:
        for ext in (".vcf", ".hmp", ".txt", ".tsv", ".csv", ".npy"):
            if low.endswith(ext):
                base = base[: -len(ext)]
                break
    cache_root = _resolve_gwas_cache_dir(
        genofile,
        cache_dir=cache_dir,
        logger=logger,
        warning_collector=warning_collector,
    )
    return os.path.join(cache_root, f"~{base}").replace("\\", "/")


def _fmt_cache_num(v: Union[float, int]) -> str:
    x = float(v)
    s = f"{x:.6g}"
    if "e" in s or "E" in s:
        s = f"{x:.12f}".rstrip("0").rstrip(".")
    if s in {"", "-0"}:
        s = "0"
    return s


def _param_tag(
    *,
    maf: float,
    geno: float,
    snps_only: bool,
) -> str:
    tag = f".maf{_fmt_cache_num(maf)}.geno{_fmt_cache_num(geno)}.snp{1 if bool(snps_only) else 0}"
    return tag


def _gwas_cache_prefix_with_params(
    genofile: str,
    *,
    maf: float,
    geno: float,
    snps_only: bool,
    cache_dir: Union[str, None] = None,
    logger: Union[logging.Logger, None] = None,
    warning_collector: Union[list[str], None] = None,
) -> str:
    base = genotype_cache_prefix(
        genofile,
        snps_only=bool(snps_only),
        cache_dir=cache_dir,
        logger=logger,
        warning_collector=warning_collector,
    )
    return f"{base}{_param_tag(maf=float(maf), geno=float(geno), snps_only=bool(snps_only))}"


def _grm_cache_paths(
    cache_prefix_with_params: str,
    *,
    mgrm: str,
) -> tuple[str, str]:
    tag = _grm_method_cache_tag(str(mgrm), legacy=False)
    grm_path = f"{cache_prefix_with_params}.{tag}.npy"
    id_path = f"{grm_path}.id"
    return grm_path, id_path


def _grm_cache_paths_legacy(
    cache_prefix_with_params: str,
    *,
    mgrm: str,
) -> tuple[str, str]:
    tag = _grm_method_cache_tag(str(mgrm), legacy=True)
    grm_path = f"{cache_prefix_with_params}.{tag}.npy"
    id_path = f"{grm_path}.id"
    return grm_path, id_path


def _pca_cache_path(
    cache_prefix_with_params: str,
    *,
    mgrm: str,
    qdim: Union[int, str],
) -> str:
    tag = _grm_method_cache_tag(str(mgrm), legacy=False)
    return f"{cache_prefix_with_params}.{tag}.pc{qdim}.txt"


def _pca_cache_path_legacy(
    cache_prefix_with_params: str,
    *,
    mgrm: str,
    qdim: Union[int, str],
) -> str:
    tag = _grm_method_cache_tag(str(mgrm), legacy=True)
    return f"{cache_prefix_with_params}.{tag}.pc{qdim}.txt"


def _grm_method_cache_tag(mgrm: str, *, legacy: bool = False) -> str:
    s = str(mgrm).strip()
    if s == "1":
        return "grm1" if legacy else "cGRM"
    if s == "2":
        return "grm2" if legacy else "sGRM"
    return s


@contextmanager
def _cache_lock(lock_key: str, timeout_s: float = 7200.0, poll_s: float = 0.2):
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
