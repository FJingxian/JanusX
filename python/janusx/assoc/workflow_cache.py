# -*- coding: utf-8 -*-
"""Cache/path helpers for GWAS workflow."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Optional, Union

from janusx.script._common.pathcheck import safe_resolve
from janusx.script._common.grmio import (
    GENOTYPE_CACHE_FALLBACK_PREFIX as _GENOTYPE_CACHE_FALLBACK_PREFIX,
    genotype_cache_fallback_message as _common_genotype_cache_fallback_message,
    genotype_cache_prefix as _common_genotype_cache_prefix,
    grm_cache_lock as _common_grm_cache_lock,
    grm_cache_paths as _common_grm_cache_paths,
    grm_cache_paths_legacy as _common_grm_cache_paths_legacy,
    grm_cache_prefix_with_params as _common_grm_cache_prefix_with_params,
    grm_method_cache_tag as _common_grm_method_cache_tag,
    is_writable_dir as _common_is_writable_dir,
    resolve_genotype_cache_dir as _common_resolve_genotype_cache_dir,
)


_WARNED_GWAS_CACHE_FALLBACK_KEYS: set[str] = set()


def _is_writable_dir(path: str) -> bool:
    return bool(_common_is_writable_dir(path))


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
    prefix = _GENOTYPE_CACHE_FALLBACK_PREFIX
    marker = ". Falling back to cache directory from JANUSX_CACHE_DIR:"
    if s.startswith(prefix) and marker in s:
        try:
            left, right = s.split(marker, 1)
            src = left[len(prefix):].strip()
            dst = right.strip()
            src_abs = str(safe_resolve(src))
            dst_abs = str(safe_resolve(dst))
            return _common_genotype_cache_fallback_message(src_abs, dst_abs)
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
    return str(
        _common_resolve_genotype_cache_dir(
            genofile,
            cache_dir=cache_dir,
            logger=logger,
            emit_warning=emit_warning,
            warning_collector=warning_collector,
        )
    )


def genotype_cache_prefix(
    genofile: str,
    *,
    snps_only: bool = True,
    cache_dir: Union[str, None] = None,
    logger: Union[logging.Logger, None] = None,
    warning_collector: Union[list[str], None] = None,
) -> str:
    return str(
        _common_genotype_cache_prefix(
            genofile,
            snps_only=bool(snps_only),
            cache_dir=cache_dir,
            logger=logger,
            warning_collector=warning_collector,
        )
    )


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
    return str(
        _common_grm_cache_prefix_with_params(
            genofile,
            maf=float(maf),
            geno=float(geno),
            snps_only=bool(snps_only),
            cache_dir=cache_dir,
            logger=logger,
            warning_collector=warning_collector,
        )
    )


def _grm_cache_paths(
    cache_prefix_with_params: str,
    *,
    mgrm: str,
) -> tuple[str, str]:
    return _common_grm_cache_paths(cache_prefix_with_params, mgrm=str(mgrm))


def _grm_cache_paths_legacy(
    cache_prefix_with_params: str,
    *,
    mgrm: str,
) -> tuple[str, str]:
    return _common_grm_cache_paths_legacy(cache_prefix_with_params, mgrm=str(mgrm))


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
    return str(_common_grm_method_cache_tag(str(mgrm), legacy=bool(legacy)))


def _cache_lock(lock_key: str, timeout_s: float = 7200.0, poll_s: float = 0.2):
    return _common_grm_cache_lock(lock_key, timeout_s=float(timeout_s), poll_s=float(poll_s))
