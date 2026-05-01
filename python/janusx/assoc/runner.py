# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import threading
from typing import Optional

from .config import AssociationConfig

_GWAS_MODULE_LOCK = threading.Lock()
_GWAS_MODULE = None


def _get_gwas_script():
    global _GWAS_MODULE
    mod = _GWAS_MODULE
    if mod is not None:
        return mod
    with _GWAS_MODULE_LOCK:
        mod = _GWAS_MODULE
        if mod is None:
            mod = importlib.import_module("janusx.assoc.workflow")
            _GWAS_MODULE = mod
    return mod


def prewarm_gwas_runner() -> None:
    """
    Preload and cache GWAS script module for low-latency first call.
    """
    _ = _get_gwas_script()


def run_gwas_cli(
    argv: list[str],
    *,
    log: bool = True,
    return_result: bool = True,
):
    """
    Execute GWAS CLI workflow in-process.

    This is a structural wrapper only; all computation stays in
    `janusx.assoc.workflow`.
    """
    return run_gwas_argv(
        argv=argv,
        log=log,
        return_result=return_result,
    )


def run_gwas_args(
    args,
    *,
    log: bool = True,
    return_result: bool = True,
):
    gwas_script = _get_gwas_script()
    return gwas_script._run_gwas_pipeline(  # type: ignore[attr-defined]
        args=args,
        log=bool(log),
        return_result=bool(return_result),
    )


def run_gwas_argv(
    argv: list[str] | None = None,
    *,
    log: bool = True,
    return_result: bool = True,
):
    gwas_script = _get_gwas_script()
    args = gwas_script.parse_args(argv)  # type: ignore[attr-defined]
    return run_gwas_args(
        args=args,
        log=log,
        return_result=return_result,
    )


def run_gwas_config(
    config: AssociationConfig,
    *,
    model_key: str,
    out: Optional[str] = None,
    prefix: Optional[str] = None,
    log: bool = True,
):
    """
    Build CLI-style args from AssociationConfig and run GWAS.
    """
    argv = config.build_gwas_argv(
        model_key=model_key,
        out_dir=out,
        out_prefix=prefix,
    )
    return run_gwas_cli(argv, log=log, return_result=True)
