# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional, Sequence
import os
import time
import numpy as np
import pandas as pd

from .config import AssociationConfig
from .result import AssociationResult
from .runner import prewarm_gwas_runner, run_gwas_config


def _ensure_2d_pheno(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError("phenotype matrix must be 1D or 2D.")


def _resolve_trait_indices(n_traits: int, traits: Optional[Sequence[int]]) -> list[int]:
    if traits is None:
        return list(range(n_traits))
    out: list[int] = []
    for x in traits:
        idx = int(x)
        if idx < 0 or idx >= n_traits:
            raise ValueError(f"trait index out of range: {idx}, n_traits={n_traits}")
        out.append(idx)
    return out


def _resolve_sample_major_genotype(genotype: np.ndarray, n_samples: int) -> np.ndarray:
    g = np.asarray(genotype, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError("genotype matrix must be 2D.")
    r, c = int(g.shape[0]), int(g.shape[1])
    if r == n_samples and c != n_samples:
        # samples x sites -> SNP-major
        return np.ascontiguousarray(g.T, dtype=np.float32)
    if c == n_samples and r != n_samples:
        # already SNP-major
        return np.ascontiguousarray(g, dtype=np.float32)
    if r == n_samples and c == n_samples:
        raise ValueError(
            "Ambiguous square genotype matrix layout (n_samples x n_samples). "
            "Please pass genotype as samples x sites or SNP-major with non-square shape."
        )
    raise ValueError(
        f"Cannot align genotype with phenotype samples. genotype shape={g.shape}, n_samples={n_samples}"
    )


def _sites_arrays(sites: Any, m: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    chrom = np.array(["1"] * m, dtype=object)
    pos = np.arange(1, m + 1, dtype=np.int64)
    a0 = ["N"] * m
    a1 = ["N"] * m

    if sites is None:
        return chrom, pos, a0, a1

    if isinstance(sites, pd.DataFrame):
        if {"chrom", "pos"}.issubset(set(sites.columns)):
            if int(sites.shape[0]) != m:
                raise ValueError(f"sites row count mismatch: {sites.shape[0]} != {m}")
            chrom = sites["chrom"].astype(str).to_numpy(dtype=object)
            pos = pd.to_numeric(sites["pos"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
            if "allele0" in sites.columns:
                a0 = sites["allele0"].astype(str).tolist()
            if "allele1" in sites.columns:
                a1 = sites["allele1"].astype(str).tolist()
            return chrom, pos, a0, a1
        raise ValueError("sites DataFrame must contain at least columns: chrom, pos")

    rows = list(sites)
    if len(rows) != m:
        raise ValueError(f"sites length mismatch: {len(rows)} != {m}")
    for i, row in enumerate(rows):
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            chrom[i] = str(row[0])
            pos[i] = int(row[1])
            if len(row) >= 4:
                a0[i] = str(row[2])
                a1[i] = str(row[3])
        else:
            raise ValueError("sites must be DataFrame or iterable of tuples: (chrom,pos[,a0,a1])")
    return chrom, pos, a0, a1


class LinearModel:
    """
    Thin association wrapper that preserves current algorithms.

    File-mode input:
      Reuses `janusx.assoc.workflow` (Rust-first routes included).
    Matrix-mode input:
      Reuses `janusx.pyBLUP.assoc` implementations directly.
    """

    def __init__(
        self,
        genotype: Any,
        phenotype: Any,
        *,
        sample_ids: Optional[Sequence[str]] = None,
        sites: Optional[Any] = None,
        covariates: Optional[Any] = None,
        traits: Optional[Sequence[int]] = None,
        maf: float = 0.02,
        miss: float = 0.05,
        het: float = 0.02,
        snps_only: bool = True,
        threads: int = 0,
        chunksize: int = 10000,
        mmap_limit: bool = False,
        grm: str = "1",
        qcov: int = 0,
        model: str = "add",
        out: str = ".",
        prefix: Optional[str] = None,
        config: Optional[AssociationConfig] = None,
        extra: Optional[dict[str, Any]] = None,
        prewarm_runner: bool = True,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = AssociationConfig(
                genotype=genotype,
                phenotype=phenotype,
                sample_ids=sample_ids,
                sites=sites,
                covariates=covariates,
                traits=traits,
                maf=maf,
                geno=miss,
                het=het,
                snps_only=snps_only,
                threads=threads,
                chunksize=chunksize,
                mmap_limit=mmap_limit,
                grm=grm,
                qcov=qcov,
                model=model,
                out=out,
                prefix=prefix,
                extra={} if extra is None else dict(extra),
            )
        self.config.validate()
        if bool(prewarm_runner) and bool(self.config.is_file_mode):
            prewarm_gwas_runner()

    def _run_file_workflow(
        self,
        *,
        model_key: str,
        out: Optional[str],
        prefix: Optional[str],
        log: bool,
    ) -> AssociationResult:
        payload = run_gwas_config(
            self.config,
            model_key=model_key,
            out=out,
            prefix=prefix,
            log=bool(log),
        )
        return AssociationResult.from_gwas_payload(payload)

    def _run_matrix_workflow(
        self,
        *,
        model_key: str,
        out: Optional[str],
        prefix: Optional[str],
        write_files: bool,
    ) -> AssociationResult:
        from janusx.pyBLUP.QK2 import GRM as _calc_grm
        from janusx.pyBLUP.assoc import LM as _LM, LMM as _LMM, FastLMM as _FastLMM, farmcpu as _farmcpu

        t0 = time.time()
        cfg = self.config
        g = np.asarray(cfg.genotype, dtype=np.float32)
        y = _ensure_2d_pheno(np.asarray(cfg.phenotype, dtype=np.float64))
        m_snp = _resolve_sample_major_genotype(g, int(y.shape[0]))
        n_samples = int(y.shape[0])
        m = int(m_snp.shape[0])

        x_cov = None
        if cfg.covariates is not None:
            x_cov = np.asarray(cfg.covariates, dtype=np.float64)
            if x_cov.ndim == 1:
                x_cov = x_cov.reshape(-1, 1)
            if int(x_cov.shape[0]) != n_samples:
                raise ValueError(
                    f"covariate sample mismatch: cov n={x_cov.shape[0]}, expected n={n_samples}"
                )

        trait_idx = _resolve_trait_indices(int(y.shape[1]), cfg.traits)
        chrom, pos, allele0, allele1 = _sites_arrays(cfg.sites, m)

        out_dir = cfg.out if out is None else out
        out_prefix = cfg.prefix if prefix is None else prefix
        if out_prefix is None or str(out_prefix).strip() == "":
            out_prefix = "janusx_assoc"
        out_dir = os.path.normpath(str(out_dir))
        if bool(write_files):
            os.makedirs(out_dir, mode=0o755, exist_ok=True)

        threads = int(max(1, cfg.threads if int(cfg.threads) > 0 else 1))
        summary_rows: list[dict[str, Any]] = []
        result_files: list[str] = []
        tables: dict[str, pd.DataFrame] = {}

        kinship = None
        if model_key in {"lmm", "fastlmm"}:
            kinship = cfg.extra.get("kinship") if isinstance(cfg.extra, dict) else None
            if kinship is None:
                kinship = _calc_grm(m_snp, log=False, chunksize=max(1, int(cfg.chunksize)))
            kinship = np.asarray(kinship, dtype=np.float64)
            if kinship.shape != (n_samples, n_samples):
                raise ValueError(
                    f"kinship shape mismatch: got {kinship.shape}, expected ({n_samples}, {n_samples})"
                )

        for tidx in trait_idx:
            y_vec = np.asarray(y[:, tidx], dtype=np.float64)
            trait_name = f"trait{tidx}"

            if model_key == "lm":
                mod = _LM(y=y_vec, X=x_cov)
                arr = np.asarray(mod.gwas(m_snp, threads=threads), dtype=np.float64)
                df = pd.DataFrame(
                    {
                        "chrom": chrom,
                        "pos": pos,
                        "allele0": allele0,
                        "allele1": allele1,
                        "beta": arr[:, 0],
                        "se": arr[:, 1],
                        "pwald": arr[:, 2],
                    }
                )
                pve = None
            elif model_key == "lmm":
                mod = _LMM(y=y_vec, X=x_cov, kinship=np.array(kinship, copy=True))
                arr = np.asarray(mod.gwas(m_snp, threads=threads), dtype=np.float64)
                cols = {
                    "chrom": chrom,
                    "pos": pos,
                    "allele0": allele0,
                    "allele1": allele1,
                    "beta": arr[:, 0],
                    "se": arr[:, 1],
                    "pwald": arr[:, 2],
                }
                if arr.shape[1] > 3:
                    cols["plrt"] = arr[:, 3]
                df = pd.DataFrame(cols)
                pve = float(mod.pve) if np.isfinite(float(mod.pve)) else None
            elif model_key == "fastlmm":
                mod = _FastLMM(y=y_vec, X=x_cov, kinship=np.array(kinship, copy=True))
                arr = np.asarray(mod.gwas(m_snp, threads=threads), dtype=np.float64)
                cols = {
                    "chrom": chrom,
                    "pos": pos,
                    "allele0": allele0,
                    "allele1": allele1,
                    "beta": arr[:, 0],
                    "se": arr[:, 1],
                    "pwald": arr[:, 2],
                }
                if arr.shape[1] > 3:
                    cols["plrt"] = arr[:, 3]
                df = pd.DataFrame(cols)
                pve = float(mod.pve) if np.isfinite(float(mod.pve)) else None
            elif model_key == "farmcpu":
                arr = np.asarray(
                    _farmcpu(
                        y=y_vec,
                        M=m_snp,
                        X=x_cov,
                        chrlist=np.asarray(chrom),
                        poslist=np.asarray(pos, dtype=np.int64),
                        threads=threads,
                    ),
                    dtype=np.float64,
                )
                df = pd.DataFrame(
                    {
                        "chrom": chrom,
                        "pos": pos,
                        "allele0": allele0,
                        "allele1": allele1,
                        "beta": arr[:, 0],
                        "se": arr[:, 1],
                        "pwald": arr[:, 2],
                    }
                )
                pve = None
            else:
                raise ValueError(f"Unsupported matrix-mode model: {model_key}")

            tables[trait_name] = df
            out_file = ""
            if bool(write_files):
                out_file = os.path.join(out_dir, f"{out_prefix}.{trait_name}.{model_key}.tsv")
                w = df.copy()
                w["pwald"] = pd.to_numeric(w["pwald"], errors="coerce").map(lambda x: f"{x:.4e}")
                if "plrt" in w.columns:
                    w["plrt"] = pd.to_numeric(w["plrt"], errors="coerce").map(lambda x: f"{x:.4e}")
                w.to_csv(out_file, sep="\t", index=False)
                result_files.append(out_file)

            summary_rows.append(
                {
                    "phenotype": trait_name,
                    "model": model_key.upper() if model_key != "fastlmm" else "FastLMM",
                    "nidv": int(n_samples),
                    "eff_snp": int(m),
                    "pve": pve,
                    "result_file": out_file,
                }
            )

        payload = {
            "status": "done",
            "error": "",
            "outprefix": os.path.join(out_dir, out_prefix),
            "log_file": "",
            "elapsed_sec": float(max(time.time() - t0, 0.0)),
            "result_files": result_files,
            "summary_rows": summary_rows,
            "traits": [f"trait{x}" for x in trait_idx],
            "tables": tables,
            "config": asdict(cfg),
        }
        res = AssociationResult.from_gwas_payload(payload)
        res.payload = payload
        return res

    def _run(
        self,
        *,
        model_key: str,
        out: Optional[str] = None,
        prefix: Optional[str] = None,
        log: bool = True,
        write_files: bool = False,
    ) -> AssociationResult:
        if self.config.is_file_mode:
            return self._run_file_workflow(
                model_key=model_key,
                out=out,
                prefix=prefix,
                log=log,
            )
        return self._run_matrix_workflow(
            model_key=model_key,
            out=out,
            prefix=prefix,
            write_files=bool(write_files),
        )

    def lm(
        self,
        *,
        out: Optional[str] = None,
        prefix: Optional[str] = None,
        log: bool = True,
        write_files: bool = False,
    ) -> AssociationResult:
        return self._run(
            model_key="lm",
            out=out,
            prefix=prefix,
            log=log,
            write_files=write_files,
        )

    def lmm(
        self,
        *,
        out: Optional[str] = None,
        prefix: Optional[str] = None,
        log: bool = True,
        write_files: bool = False,
    ) -> AssociationResult:
        return self._run(
            model_key="lmm",
            out=out,
            prefix=prefix,
            log=log,
            write_files=write_files,
        )

    def fastlmm(
        self,
        *,
        out: Optional[str] = None,
        prefix: Optional[str] = None,
        log: bool = True,
        write_files: bool = False,
    ) -> AssociationResult:
        return self._run(
            model_key="fastlmm",
            out=out,
            prefix=prefix,
            log=log,
            write_files=write_files,
        )

    def farmcpu(
        self,
        *,
        out: Optional[str] = None,
        prefix: Optional[str] = None,
        log: bool = True,
        write_files: bool = False,
    ) -> AssociationResult:
        return self._run(
            model_key="farmcpu",
            out=out,
            prefix=prefix,
            log=log,
            write_files=write_files,
        )
