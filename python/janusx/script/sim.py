import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import numpy as np

from janusx.gfreader import SiteInfo, save_genotype_streaming

DEFAULT_MAF_LOW = 0.02
DEFAULT_MAF_HIGH = 0.45
AUTO_CHUNK_MEM_MB = 512
AUTO_FAMILY_FRAC = 0.80
AUTO_FAMILY_SIZE = 4


@dataclass
class FamilyLayout:
    structure: str
    family_size: int
    n_families: int
    n_family_samples: int
    n_unrelated: int
    sire_idx: np.ndarray
    dam_idx: np.ndarray
    child_groups: List[np.ndarray]
    unrelated_idx: np.ndarray


def _auto_chunk_size(n_individuals: int) -> int:
    """
    Infer a safer chunk size from sample size.

    Approximation uses ~10 bytes per genotype entry to account for transient arrays.
    """
    if n_individuals <= 0:
        raise ValueError("n_individuals must be positive.")
    per_entry_bytes = 10
    budget = int(AUTO_CHUNK_MEM_MB) * 1024 * 1024
    guess = max(256, budget // max(1, n_individuals * per_entry_bytes))
    return int(max(256, min(50_000, guess)))


def _draw_hwe_genotypes(mafs: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw unrelated diploid genotypes under HWE.
    Returns matrix of shape (m_snps, n_samples), dtype int8 in {0,1,2}.
    """
    m = int(mafs.shape[0])
    if n_samples <= 0:
        return np.empty((m, 0), dtype=np.int8)

    u = rng.random((m, n_samples), dtype=np.float32)
    p0 = (1.0 - mafs) ** 2
    p1 = p0 + 2.0 * mafs * (1.0 - mafs)

    g = np.empty((m, n_samples), dtype=np.int8)
    g[u < p0[:, None]] = 0
    mid = (u >= p0[:, None]) & (u < p1[:, None])
    g[mid] = 1
    g[u >= p1[:, None]] = 2
    return g


def _build_family_layout(
    n_individuals: int,
    structure: str,
    family_frac: float,
    family_size: int,
) -> FamilyLayout:
    structure_norm = str(structure).strip().lower()
    if structure_norm not in {"unrelated", "family", "mixed"}:
        raise ValueError("structure must be one of: unrelated, family, mixed")
    if family_size < 3:
        raise ValueError("family_size must be >= 3 (two parents + >=1 offspring).")

    if structure_norm == "unrelated":
        n_family_samples = 0
    elif structure_norm == "family":
        n_family_samples = n_individuals
    else:
        ff = max(0.0, min(1.0, float(family_frac)))
        n_family_samples = int(round(n_individuals * ff))

    n_family_samples = (n_family_samples // family_size) * family_size
    n_families = n_family_samples // family_size
    n_unrelated = n_individuals - n_family_samples

    if n_families > 0:
        fam = np.arange(n_family_samples, dtype=np.int32).reshape(n_families, family_size)
        sire_idx = fam[:, 0]
        dam_idx = fam[:, 1]
        child_groups = [fam[:, i] for i in range(2, family_size)]
    else:
        sire_idx = np.empty((0,), dtype=np.int32)
        dam_idx = np.empty((0,), dtype=np.int32)
        child_groups = []

    unrelated_idx = np.arange(n_family_samples, n_individuals, dtype=np.int32)
    return FamilyLayout(
        structure=structure_norm,
        family_size=int(family_size),
        n_families=int(n_families),
        n_family_samples=int(n_family_samples),
        n_unrelated=int(n_unrelated),
        sire_idx=sire_idx,
        dam_idx=dam_idx,
        child_groups=child_groups,
        unrelated_idx=unrelated_idx,
    )


def simulate_chunks(
    nsnp: int,
    nidv: int,
    chunk_size: int = 50_000,
    maf_low: float = 0.02,
    maf_high: float = 0.45,
    seed: int = 1,
    layout: Optional[FamilyLayout] = None,
) -> Iterator[tuple[np.ndarray, list[SiteInfo]]]:
    rng = np.random.default_rng(seed)
    n_done = 0
    layout = layout or _build_family_layout(nidv, "unrelated", 0.0, 4)

    while n_done < nsnp:
        m = min(chunk_size, nsnp - n_done)
        mafs = rng.uniform(maf_low, maf_high, size=m).astype(np.float32)
        g = np.empty((m, nidv), dtype=np.int8)

        if layout.n_families > 0:
            founders = _draw_hwe_genotypes(mafs, layout.n_families * 2, rng)
            sires = founders[:, 0::2]
            dams = founders[:, 1::2]

            g[:, layout.sire_idx] = sires
            g[:, layout.dam_idx] = dams

            for child_idx in layout.child_groups:
                pat = (rng.random((m, layout.n_families), dtype=np.float32) * 2.0) < sires
                mat = (rng.random((m, layout.n_families), dtype=np.float32) * 2.0) < dams
                g[:, child_idx] = pat.astype(np.int8) + mat.astype(np.int8)

        if layout.n_unrelated > 0:
            g[:, layout.unrelated_idx] = _draw_hwe_genotypes(mafs, layout.n_unrelated, rng)

        sites = [SiteInfo("1", int(i), "A", "T") for i in range(n_done, n_done + m)]
        yield g, sites
        n_done += m


def _simulate_trait(
    chunks: Iterable[tuple[np.ndarray, list[SiteInfo]]],
    nsnp: int,
    nidv: int,
    pve: float,
    ve: float,
    seed: int,
    trait_mean: float = 100.0,
) -> np.ndarray:
    if not (0.0 <= pve < 1.0):
        raise ValueError("pve must satisfy 0 <= pve < 1.")
    if ve <= 0.0:
        raise ValueError("ve must be > 0.")

    rng = np.random.default_rng(seed + 100_003)
    y = rng.normal(loc=trait_mean, scale=np.sqrt(ve), size=(nidv, 1)).astype(np.float32)
    if pve == 0.0:
        return y

    vg = pve * ve / (1.0 - pve)
    beta_sd = np.sqrt(vg / max(1, int(nsnp)))
    for g, _ in chunks:
        beta = rng.normal(0.0, beta_sd, size=(g.shape[0], 1)).astype(np.float32)
        y += g.T.astype(np.float32, copy=False) @ beta
    return y


def _write_phenotypes(
    outprefix: str,
    sample_ids: np.ndarray,
    y: np.ndarray,
    trait_name: str = "test",
    na_rate: float = 0.10,
    seed: int = 1,
) -> None:
    sample_ids = np.asarray(sample_ids, dtype=str)
    yv = y.reshape(-1).astype(np.float32)

    pheno3 = np.empty((sample_ids.size, 3), dtype=object)
    pheno3[:, 0] = sample_ids
    pheno3[:, 1] = sample_ids
    pheno3[:, 2] = yv
    np.savetxt(
        f"{outprefix}.pheno",
        pheno3,
        delimiter="\t",
        fmt=["%s", "%s", "%.6f"],
    )

    pheno2 = np.column_stack([sample_ids, yv.astype(object)])
    np.savetxt(
        f"{outprefix}.pheno.txt",
        pheno2,
        delimiter="\t",
        fmt=["%s", "%.6f"],
        header=f"IID\t{trait_name}",
        comments="",
    )

    rng = np.random.default_rng(seed + 777)
    pheno2_na = np.empty((sample_ids.size, 2), dtype=object)
    pheno2_na[:, 0] = sample_ids
    pheno2_na[:, 1] = [f"{v:.6f}" for v in yv.tolist()]
    k = int(round(sample_ids.size * max(0.0, min(1.0, float(na_rate)))))
    if k > 0:
        idx = rng.choice(sample_ids.size, size=k, replace=False)
        pheno2_na[idx, 1] = "NA"
    np.savetxt(
        f"{outprefix}.pheno.NA.txt",
        pheno2_na,
        delimiter="\t",
        fmt=["%s", "%s"],
        header=f"IID\t{trait_name}",
        comments="",
    )


def _write_family_map(outprefix: str, sample_ids: np.ndarray, layout: FamilyLayout) -> None:
    if layout.n_families == 0:
        return

    path = f"{outprefix}.family.tsv"
    sample_ids = np.asarray(sample_ids, dtype=str)
    with open(path, "w", encoding="utf-8") as f:
        f.write("IID\tFAMILY\tROLE\tSIRE\tDAM\n")
        for fi in range(layout.n_families):
            fam = f"F{fi + 1:06d}"
            sire = sample_ids[layout.sire_idx[fi]]
            dam = sample_ids[layout.dam_idx[fi]]
            f.write(f"{sire}\t{fam}\tsire\t0\t0\n")
            f.write(f"{dam}\t{fam}\tdam\t0\t0\n")
            for cg in layout.child_groups:
                kid = sample_ids[cg[fi]]
                f.write(f"{kid}\t{fam}\toffspring\t{sire}\t{dam}\n")
        for i, idx in enumerate(layout.unrelated_idx, start=1):
            iid = sample_ids[idx]
            f.write(f"{iid}\tU{i:06d}\tunrelated\t0\t0\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jx sim",
        description="Quick genotype + phenotype simulator for benchmark workflows.",
    )
    parser.add_argument("nsnp_k", type=int, help="SNP count in thousands (e.g., 500 -> 500k SNPs).")
    parser.add_argument("n_individuals", type=int, help="Number of individuals.")
    parser.add_argument("outprefix", type=str, help="Output prefix for .bed/.bim/.fam and .pheno files.")

    parser.add_argument(
        "-chunksize", "--chunksize",
        dest="chunk_size",
        type=int,
        default=None,
        help=(
            "SNP chunk size for streaming generation. If unset, it is auto-tuned "
            f"from sample size using an internal memory budget (~{AUTO_CHUNK_MEM_MB} MB)."
        ),
    )
    parser.add_argument("--chunk-size", dest="chunk_size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("-seed", "--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "-maf-high", "--maf-high",
        type=float,
        default=DEFAULT_MAF_HIGH,
        help=f"Upper MAF bound for simulation (lower bound fixed at {DEFAULT_MAF_LOW}).",
    )

    parser.add_argument(
        "-structure", "--structure",
        type=str,
        default="unrelated",
        choices=["unrelated", "family", "mixed"],
        help="Sample structure mode.",
    )
    parser.add_argument("-pve", "--pve", type=float, default=0.5, help="Genetic proportion of variance for phenotype simulation.")
    parser.add_argument("-ve", "--ve", type=float, default=1.0, help="Environmental variance for phenotype simulation.")
    parser.add_argument("-trait-name", "--trait-name", type=str, default="test", help="Trait column name in .pheno.txt outputs.")
    parser.add_argument("-na-rate", "--na-rate", type=float, default=0.10, help="Missing-value rate in .pheno.NA.txt.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    nsnp = int(1e3 * int(args.nsnp_k))
    nidv = int(args.n_individuals)
    if nsnp <= 0 or nidv <= 0:
        raise ValueError("nsnp_k and n_individuals must be positive integers.")

    outprefix = str(args.outprefix)
    outdir = os.path.dirname(outprefix)
    if outdir:
        os.makedirs(outdir, exist_ok=True, mode=0o777)

    chunk_size = int(args.chunk_size) if (args.chunk_size is not None and int(args.chunk_size) > 0) else _auto_chunk_size(nidv)
    maf_low = float(DEFAULT_MAF_LOW)
    maf_high = float(args.maf_high)
    if not (maf_low < maf_high <= 0.5):
        raise ValueError(f"maf_high must satisfy {maf_low} < maf_high <= 0.5.")
    layout = _build_family_layout(
        n_individuals=nidv,
        structure=args.structure,
        family_frac=float(AUTO_FAMILY_FRAC),
        family_size=int(AUTO_FAMILY_SIZE),
    )

    sample_ids = np.arange(1, 1 + nidv).astype(str)
    realized_family_frac = 0.0 if nidv == 0 else layout.n_family_samples / float(nidv)
    auto_structure_note = ""
    if layout.structure == "mixed":
        auto_structure_note = f", auto_family_frac={AUTO_FAMILY_FRAC:.0%}, auto_family_size={AUTO_FAMILY_SIZE}"
    elif layout.structure == "family":
        auto_structure_note = f", auto_family_size={AUTO_FAMILY_SIZE}"
    print(
        "Generating data with "
        f"{nsnp:.1e} SNPs, {nidv} individuals, chunk={chunk_size}, "
        f"structure={layout.structure}, families={layout.n_families}, "
        f"family_samples={layout.n_family_samples} ({realized_family_frac:.1%}), "
        f"unrelated={layout.n_unrelated}{auto_structure_note}..."
    )

    chunks_for_trait = simulate_chunks(
        nsnp,
        nidv,
        chunk_size=chunk_size,
        maf_low=maf_low,
        maf_high=maf_high,
        seed=int(args.seed),
        layout=layout,
    )
    y = _simulate_trait(
        chunks_for_trait,
        nsnp=nsnp,
        nidv=nidv,
        pve=float(args.pve),
        ve=float(args.ve),
        seed=int(args.seed),
    )

    chunks_for_write = simulate_chunks(
        nsnp,
        nidv,
        chunk_size=chunk_size,
        maf_low=maf_low,
        maf_high=maf_high,
        seed=int(args.seed),
        layout=layout,
    )
    save_genotype_streaming(outprefix, sample_ids.tolist(), chunks_for_write, total_snps=nsnp)
    _write_phenotypes(
        outprefix,
        sample_ids,
        y,
        trait_name=str(args.trait_name),
        na_rate=float(args.na_rate),
        seed=int(args.seed),
    )
    if layout.n_families > 0:
        _write_family_map(outprefix, sample_ids, layout)

    print(f"[DONE] Prefix: {outprefix}")


if __name__ == "__main__":
    main(sys.argv[1:])
