# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

from .config import GsConfig
from .result import GsResult
from .runner import prewarm_gs_runner, run_gs_config


class GenomicSelection:
    """
    Thin GS wrapper.

    This class only organizes arguments and delegates execution to
    `janusx.gs.workflow` (existing algorithms and optimizations).
    """

    def __init__(
        self,
        genotype: str,
        phenotype: str,
        *,
        traits: Optional[Sequence[int]] = None,
        maf: float = 0.02,
        miss: float = 0.05,
        cv: Optional[int] = 5,
        threads: int = 0,
        out: str = ".",
        prefix: Optional[str] = None,
        config: Optional[GsConfig] = None,
        prewarm_runner: bool = True,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = GsConfig(
                genotype=str(genotype),
                phenotype=str(phenotype),
                traits=traits,
                maf=maf,
                geno=miss,
                cv=cv,
                threads=threads,
                out=out,
                prefix=prefix,
            )
        self.config.validate()
        if bool(prewarm_runner):
            prewarm_gs_runner()

    def run(
        self,
        *,
        out: Optional[str] = None,
        prefix: Optional[str] = None,
        log: bool = True,
    ) -> GsResult:
        payload = run_gs_config(self.config, out=out, prefix=prefix, log=bool(log))
        return GsResult.from_payload(payload)

    def gblup(
        self,
        *,
        kernels: Sequence[str] = ("a",),
        out: Optional[str] = None,
        prefix: Optional[str] = None,
        log: bool = True,
    ) -> GsResult:
        cfg = replace(
            self.config,
            gblup_kernels=tuple(str(x).strip().lower() for x in kernels),
            rrblup=False,
            bayesa=False,
            bayesb=False,
            bayescpi=False,
            rf=False,
            et=False,
            gbdt=False,
            xgb=False,
            svm=False,
            enet=False,
        )
        payload = run_gs_config(cfg, out=out, prefix=prefix, log=bool(log))
        return GsResult.from_payload(payload)

    def rrblup(
        self,
        *,
        out: Optional[str] = None,
        prefix: Optional[str] = None,
        log: bool = True,
    ) -> GsResult:
        cfg = replace(
            self.config,
            gblup_kernels=tuple(),
            rrblup=True,
            bayesa=False,
            bayesb=False,
            bayescpi=False,
            rf=False,
            et=False,
            gbdt=False,
            xgb=False,
            svm=False,
            enet=False,
        )
        payload = run_gs_config(cfg, out=out, prefix=prefix, log=bool(log))
        return GsResult.from_payload(payload)
