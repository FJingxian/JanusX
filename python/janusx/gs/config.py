# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
import os


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
class GsConfig:
    genotype: str
    phenotype: str
    traits: Optional[Sequence[int]] = None
    maf: float = 0.02
    geno: float = 0.05
    cv: Optional[int] = 5
    threads: int = 0
    out: str = "."
    prefix: Optional[str] = None
    gblup_kernels: tuple[str, ...] = ("a",)
    rrblup: bool = False
    bayesa: bool = False
    bayesb: bool = False
    bayescpi: bool = False
    rf: bool = False
    et: bool = False
    gbdt: bool = False
    xgb: bool = False
    svm: bool = False
    enet: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if str(self.genotype).strip() == "":
            raise ValueError("genotype path must not be empty.")
        if str(self.phenotype).strip() == "":
            raise ValueError("phenotype path must not be empty.")
        if not (0.0 <= float(self.maf) <= 0.5):
            raise ValueError("maf must be within [0, 0.5].")
        if not (0.0 <= float(self.geno) <= 1.0):
            raise ValueError("geno must be within [0, 1].")
        if self.cv is not None and int(self.cv) < 2:
            raise ValueError("cv must be >= 2 when set.")
        if self.traits is not None:
            for x in self.traits:
                if int(x) < 0:
                    raise ValueError(f"trait index must be >= 0, got {x}")
        for k in self.gblup_kernels:
            kk = str(k).strip().lower()
            if kk not in {"a", "d", "ad"}:
                raise ValueError(f"Unsupported GBLUP kernel token: {k}")

    def _selected_model_count(self) -> int:
        n = 0
        if len(self.gblup_kernels) > 0:
            n += 1
        n += int(bool(self.rrblup))
        n += int(bool(self.bayesa))
        n += int(bool(self.bayesb))
        n += int(bool(self.bayescpi))
        n += int(bool(self.rf))
        n += int(bool(self.et))
        n += int(bool(self.gbdt))
        n += int(bool(self.xgb))
        n += int(bool(self.svm))
        n += int(bool(self.enet))
        return int(n)

    def build_gs_argv(
        self,
        *,
        out_dir: Optional[str] = None,
        out_prefix: Optional[str] = None,
        threads: Optional[int] = None,
    ) -> list[str]:
        self.validate()
        if self._selected_model_count() == 0:
            raise ValueError("No GS model selected in config.")

        flag, src = _detect_genotype_cli_flag(self.genotype)
        argv: list[str] = [flag, src, "-p", _normalize_path_text(self.phenotype)]
        argv.extend(["-maf", str(float(self.maf))])
        argv.extend(["-geno", str(float(self.geno))])
        if self.cv is not None:
            argv.extend(["-cv", str(int(self.cv))])

        if len(self.gblup_kernels) > 0:
            g_tokens = [str(x).strip().lower() for x in self.gblup_kernels]
            if len(g_tokens) == 1 and g_tokens[0] == "a":
                argv.append("-GBLUP")
            else:
                argv.extend(["-GBLUP"] + g_tokens)
        if bool(self.rrblup):
            argv.append("-rrBLUP")
        if bool(self.bayesa):
            argv.append("-BayesA")
        if bool(self.bayesb):
            argv.append("-BayesB")
        if bool(self.bayescpi):
            argv.append("-BayesCpi")
        if bool(self.rf):
            argv.append("-RF")
        if bool(self.et):
            argv.append("-ET")
        if bool(self.gbdt):
            argv.append("-GBDT")
        if bool(self.xgb):
            argv.append("-XGB")
        if bool(self.svm):
            argv.append("-SVM")
        if bool(self.enet):
            argv.append("-ENET")

        if self.traits is not None:
            for t in self.traits:
                argv.extend(["-n", str(int(t))])

        out_use = self.out if out_dir is None else out_dir
        argv.extend(["-o", _normalize_path_text(str(out_use))])
        pref_use = self.prefix if out_prefix is None else out_prefix
        if pref_use is not None and str(pref_use).strip() != "":
            argv.extend(["-prefix", str(pref_use).strip()])
        t_use = int(self.threads if threads is None else int(threads))
        if t_use > 0:
            argv.extend(["-t", str(t_use)])
        return argv
