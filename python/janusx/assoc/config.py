# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence
import os
import numpy as np


def _normalize_path_text(path: str) -> str:
    return os.path.normpath(str(path).strip())


def _as_plink_prefix(path_or_prefix: str) -> Optional[str]:
    p = _normalize_path_text(path_or_prefix)
    if p == "":
        return None
    low = p.lower()
    if low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam"):
        p = p[:-4]
    if all(os.path.isfile(f"{p}.{ext}") for ext in ("bed", "bim", "fam")):
        return p
    return None


def _detect_genotype_cli_flag(path_or_prefix: str) -> tuple[str, str]:
    src = _normalize_path_text(path_or_prefix)
    low = src.lower()
    if low.endswith(".vcf") or low.endswith(".vcf.gz"):
        return "-vcf", src
    if low.endswith(".hmp") or low.endswith(".hmp.gz"):
        return "-hmp", src
    prefix = _as_plink_prefix(src)
    if prefix is not None:
        return "-bfile", prefix
    return "-file", src


@dataclass(slots=True)
class AssociationConfig:
    """
    Thin-wrapper config for association workflows.

    `genotype` / `phenotype` can be either path-based file input or in-memory
    matrices. File mode reuses `janusx.assoc.workflow`; matrix mode reuses
    existing `janusx.pyBLUP.assoc` kernels.
    """

    genotype: Any
    phenotype: Any
    sample_ids: Optional[Sequence[str]] = None
    sites: Optional[Any] = None
    covariates: Optional[Any] = None
    traits: Optional[Sequence[int]] = None
    maf: float = 0.02
    geno: float = 0.05
    het: float = 0.02
    snps_only: bool = True
    threads: int = 0
    chunksize: int = 10000
    mmap_limit: bool = False
    grm: str = "1"
    qcov: int = 0
    model: str = "add"
    out: str = "."
    prefix: Optional[str] = None

    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if isinstance(self.genotype, str):
            if str(self.genotype).strip() == "":
                raise ValueError("genotype path must not be empty.")
            if not isinstance(self.phenotype, str):
                raise ValueError("When genotype is file input, phenotype must be a file path.")
            if str(self.phenotype).strip() == "":
                raise ValueError("phenotype path must not be empty.")
        else:
            g = np.asarray(self.genotype)
            y = np.asarray(self.phenotype)
            if g.ndim != 2:
                raise ValueError("Matrix-mode genotype must be 2D.")
            if y.ndim not in (1, 2):
                raise ValueError("Matrix-mode phenotype must be 1D or 2D.")
            n_samples = int(g.shape[0])
            if int(y.shape[0]) != n_samples:
                raise ValueError(
                    f"Matrix-mode sample mismatch: genotype n={n_samples}, phenotype n={y.shape[0]}"
                )

        if not (0.0 <= float(self.maf) <= 0.5):
            raise ValueError("maf must be within [0, 0.5].")
        if not (0.0 <= float(self.geno) <= 1.0):
            raise ValueError("geno must be within [0, 1].")
        if not (0.0 <= float(self.het) <= 0.5):
            raise ValueError("het must be within [0, 0.5].")
        if int(self.chunksize) < 1:
            raise ValueError("chunksize must be >= 1.")

    @property
    def is_file_mode(self) -> bool:
        return isinstance(self.genotype, str)

    def normalize_threads(self, fallback: int) -> int:
        t = int(self.threads)
        return int(fallback) if t <= 0 else t

    def trait_indices(self) -> Optional[list[int]]:
        if self.traits is None:
            return None
        out: list[int] = []
        for x in self.traits:
            out.append(int(x))
        return out

    def build_gwas_argv(
        self,
        *,
        model_key: str,
        out_dir: Optional[str] = None,
        out_prefix: Optional[str] = None,
        extra_covariates: Optional[Iterable[str]] = None,
        threads: Optional[int] = None,
    ) -> list[str]:
        if not self.is_file_mode:
            raise ValueError("build_gwas_argv is only valid for file-mode config.")
        self.validate()
        mk = str(model_key).strip().lower()
        if mk not in {"lm", "lmm", "fastlmm", "farmcpu", "lrlmm"}:
            raise ValueError(f"unsupported model_key: {model_key}")

        flag, src = _detect_genotype_cli_flag(str(self.genotype))
        argv: list[str] = [flag, src, "-p", _normalize_path_text(str(self.phenotype))]
        if mk == "lm":
            argv.append("-lm")
        elif mk == "lmm":
            argv.append("-lmm")
        elif mk == "fastlmm":
            argv.append("-fastlmm")
        elif mk == "farmcpu":
            argv.append("-farmcpu")
        elif mk == "lrlmm":
            argv.append("-lrlmm")

        argv.extend(["-model", str(self.model).strip().lower()])
        argv.extend(["-maf", str(float(self.maf))])
        argv.extend(["-geno", str(float(self.geno))])
        argv.extend(["-het", str(float(self.het))])
        argv.extend(["-chunksize", str(int(self.chunksize))])
        argv.extend(["-k", str(self.grm)])
        argv.extend(["-q", str(int(self.qcov))])
        if bool(self.snps_only):
            argv.append("-snps-only")
        if bool(self.mmap_limit):
            argv.append("-mmap-limit")

        out_use = self.out if out_dir is None else out_dir
        argv.extend(["-o", _normalize_path_text(str(out_use))])
        pref_use = self.prefix if out_prefix is None else out_prefix
        if pref_use is not None and str(pref_use).strip() != "":
            argv.extend(["-prefix", str(pref_use).strip()])

        t_use = self.threads if threads is None else int(threads)
        if int(t_use) > 0:
            argv.extend(["-t", str(int(t_use))])

        if self.traits is not None:
            for tidx in self.trait_indices() or []:
                argv.extend(["-n", str(int(tidx))])

        cov_inputs: list[str] = []
        if self.covariates is not None:
            if isinstance(self.covariates, (list, tuple)):
                cov_inputs.extend([str(x) for x in self.covariates])
            else:
                cov_inputs.append(str(self.covariates))
        if extra_covariates is not None:
            cov_inputs.extend([str(x) for x in extra_covariates])
        for cov in cov_inputs:
            c = str(cov).strip()
            if c != "":
                argv.extend(["-c", c])

        return argv
