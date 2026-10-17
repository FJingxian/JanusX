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
    het: float = 1.0
    cv: Optional[int] = 5
    threads: int = 0
    out: str = "."
    prefix: Optional[str] = None
    gblup_kernels: tuple[str, ...] = ("a",)
    blup: bool = False
    rrblup: bool = False
    bayesa: bool = False
    bayesb: bool | float = False
    bayesc: bool | float = False
    bayesr: bool = False
    rf: bool = False
    et: bool = False
    gbdt: bool = False
    xgb: bool = False
    svm: bool = False
    enet: bool = False
    pls: bool = False
    krr: bool = False
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
        if not (0.0 <= float(self.het) <= 1.0):
            raise ValueError("het must be within [0, 1].")
        if self.cv is not None and int(self.cv) < 2:
            raise ValueError("cv must be >= 2 when set.")
        if self.traits is not None:
            for x in self.traits:
                if int(x) < 0:
                    raise ValueError(f"trait index must be >= 0, got {x}")
        for k in self.gblup_kernels:
            kk = str(k).strip().lower()
            if kk != "a":
                raise ValueError(
                    "GS supports additive GBLUP only; "
                    f"unsupported kernel token: {k!r}."
                )
        for name in ("bayesb", "bayesc"):
            value = getattr(self, name)
            if value is False or value is True or value is None:
                continue
            try:
                pi = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be False/True or a fixed pi in (0, 1).") from exc
            if not (0.0 < pi < 1.0):
                raise ValueError(f"{name} fixed pi must be in (0, 1), got {value!r}.")

    def _selected_model_count(self) -> int:
        n = 0
        if len(self.gblup_kernels) > 0:
            n += 1
        n += int(bool(self.blup))
        n += int(bool(self.rrblup))
        n += int(bool(self.bayesa))
        n += int(self.bayesb is not False and self.bayesb is not None)
        n += int(self.bayesc is not False and self.bayesc is not None)
        n += int(bool(self.bayesr))
        n += int(bool(self.rf))
        n += int(bool(self.et))
        n += int(bool(self.gbdt))
        n += int(bool(self.xgb))
        n += int(bool(self.svm))
        n += int(bool(self.enet))
        n += int(bool(self.pls))
        n += int(bool(self.krr))
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
        argv.extend(["-het", str(float(self.het))])
        if self.cv is not None:
            argv.extend(["-cv", str(int(self.cv))])

        if len(self.gblup_kernels) > 0:
            g_tokens = [str(x).strip().lower() for x in self.gblup_kernels]
            if g_tokens != ["a"]:
                raise ValueError(
                    "The public GS CLI supports additive GBLUP only through -BLUP."
                )
            argv.append("-BLUP")
        if bool(self.blup):
            if "-BLUP" not in argv:
                argv.append("-BLUP")
        if bool(self.rrblup):
            if "-BLUP" not in argv:
                argv.append("-BLUP")
            argv.extend(["--rrblup-solver", "exact"])
        if bool(self.bayesa):
            argv.append("-BayesA")
        if self.bayesb is not False and self.bayesb is not None:
            argv.append("-BayesB")
            if self.bayesb is not True:
                argv.append(str(float(self.bayesb)))
        if self.bayesc is not False and self.bayesc is not None:
            argv.append("-BayesC")
            if self.bayesc is not True:
                argv.append(str(float(self.bayesc)))
        if bool(self.bayesr):
            argv.append("-BayesR")
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
        if bool(self.pls):
            argv.append("-PLS")
        if bool(self.krr):
            argv.append("-KRR")

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
