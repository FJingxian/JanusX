from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Union, Optional, Dict, Any, List, Tuple

from ..janusx import merge_genotypes


PathLike = Union[str, os.PathLike]


def _is_vcf(p: str) -> bool:
    x = p.lower()
    return x.endswith(".vcf") or x.endswith(".vcf.gz")


def _ensure_outdir(out: str) -> None:
    """
    Ensure parent directory exists for out target.
    - out can be "merged" (no parent) -> do nothing
    - out can be "out/merged" or "out/merged.vcf.gz" -> mkdir out/
    """
    parent = os.path.dirname(os.fspath(out))
    if parent:
        os.makedirs(parent, mode=0o755, exist_ok=True)


def _normalize_inputs(inputs: Iterable[PathLike]) -> List[str]:
    if inputs is None:
        raise ValueError("inputs cannot be None")
    xs = [os.fspath(x) for x in inputs]
    if len(xs) == 0:
        raise ValueError("inputs must be a non-empty list")
    return xs


def _check_inputs_exist(inputs: List[str]) -> None:
    """
    Best-effort validation:
    - VCF: file must exist
    - PLINK prefix: prefix.bed/.bim/.fam must exist
    """
    missing: List[str] = []
    for x in inputs:
        if _is_vcf(x):
            if not os.path.exists(x):
                missing.append(x)
        else:
            bed = x + ".bed"
            bim = x + ".bim"
            fam = x + ".fam"
            if not (os.path.exists(bed) and os.path.exists(bim) and os.path.exists(fam)):
                missing.append(f"{x} (need {bed},{bim},{fam})")
    if missing:
        msg = "Some inputs are missing:\n  - " + "\n  - ".join(missing)
        raise FileNotFoundError(msg)


def merge(
    inputs: Iterable[PathLike],
    out: PathLike,
    *,
    out_fmt: str = "auto",
    sample_prefix: bool = False,
    maf: float = 0.0,
    geno: float = 1.0,
    check_exists: bool = True,
    return_dict: bool = True,
) -> Tuple[object, Dict[str, Any]] | object:
    """
    Merge multiple genotype datasets by *sample concatenation* (union of sites),
    powered by the Rust `merge_genotypes` backend.

    Supported input formats
    -----------------------
    - PLINK bfile (BED/BIM/FAM): pass the prefix without extension, e.g. "data/QC"
    - VCF / VCF.GZ: pass the full path, e.g. "data/QC.vcf.gz"

    Output formats
    --------------
    - out_fmt="auto" (default):
        * if `out` endswith ".vcf" or ".vcf.gz" -> write VCF
        * otherwise -> write PLINK bfile (out as prefix)
    - out_fmt="vcf": force VCF output (out should endwith .vcf or .vcf.gz)
    - out_fmt="plink": force PLINK bfile output (out is prefix)

    Merge rules (Rust side)
    -----------------------
    1) Drop all multi-allelic / non-SNP sites; keep only biallelic A/C/G/T
    2) Same (chrom,pos) and same REF/ALT -> direct merge
    3) Same (chrom,pos) but REF/ALT differs -> align via swap/strand-complement,
       then apply global MAF-based REF/ALT reordering (ALT freq <= 0.5)
    4) Different positions -> union merge; missing filled as ./.

    Parameters
    ----------
    inputs : iterable of path/prefix
        Genotype sources (>=2 items).
    out : path-like
        Output target:
        - PLINK: prefix without extension, e.g. "out/merged"
        - VCF  : file path, e.g. "out/merged.vcf.gz"
    out_fmt : {"auto","vcf","plink"}
        Output format selection.
    sample_prefix : bool
        If True, prepend dataset prefix in merged sample IDs (e.g. D1_, D2_).
        Default False.
    maf : float
        Drop merged sites with minor allele frequency lower than this threshold.
        Valid range: [0, 0.5]. Default 0.0.
    geno : float
        Drop merged sites with missing rate greater than this threshold.
        Valid range: [0, 1]. Default 1.0.
    check_exists : bool
        If True, validate inputs exist before calling Rust.
    return_dict : bool
        If True, also return a pure-python dict version of stats.

    Returns
    -------
    stats : janusx.PyMergeStats
        Rust stats object, with fields like:
          - n_sites_written
          - n_samples_total
          - as_dict()
    stats_dict : dict
        Only returned when return_dict=True. Convenience JSON-friendly dict.

    Examples
    --------
    >>> stats, d = merge(["geno/A", "geno/B.vcf.gz"], "out/merged", out_fmt="plink")
    >>> print(stats.n_sites_written, stats.n_samples_total)
    >>> print(d["per_input_present_sites"])

    >>> stats, d = merge(["geno/A", "geno/B"], "out/merged.vcf", out_fmt="auto")
    """
    xs = _normalize_inputs(inputs)
    out_s = os.fspath(out)

    if check_exists:
        _check_inputs_exist(xs)

    _ensure_outdir(out_s)

    # Call Rust
    stats = merge_genotypes(xs, out_s, out_fmt, sample_prefix, float(maf), float(geno))

    if not return_dict:
        return stats

    # Make JSON-friendly dict
    d = stats.as_dict()
    return stats, d
