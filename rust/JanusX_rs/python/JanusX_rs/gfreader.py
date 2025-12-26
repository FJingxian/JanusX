import numpy as np
from tqdm import tqdm
from .JanusX_rs import BedChunkReader, VcfChunkReader, count_vcf_snps, PlinkStreamWriter, VcfStreamWriter, SiteInfo

def load_genotype_chunks(
    path_or_prefix: str,
    chunk_size: int = 50000,
    maf: float = 0.0,
    missing_rate: float = 1.0
):
    """
    Unified high-level Python interface for reading genotype data in chunks
    using the Rust gfreader_rs backend.

    This function provides a *streaming* (iterator-based) interface and never
    loads the full genotype matrix into memory.

    It works for:
      - PLINK BED/BIM/FAM (SNP-major, on-disk)
      - VCF / VCF.GZ (text or compressed)

    Parameters
    ----------
    path_or_prefix : str
        Genotype source.
        - PLINK: prefix without extension, e.g. "data/QC"
        - VCF  : full path, e.g. "data/QC.vcf.gz"

    chunk_size : int
        Number of SNPs (rows) decoded per iteration.

    maf : float
        Minor Allele Frequency threshold.
        SNPs with MAF < maf are filtered out *during decoding* in Rust.

    missing_rate : float
        Maximum allowed missing genotype rate per SNP.
        SNPs exceeding this threshold are filtered out.

    Yields
    ------
    geno : np.ndarray, shape (n_snps_chunk, n_samples), dtype=float32
        SNP-major genotype block.
        Each block is:
          - already MAF-flipped (ALT freq <= 0.5)
          - mean-imputed for missing genotypes
          - dense float32 array backed by Rust memory

    sites : list[SiteInfo]
        Metadata objects for each SNP in this chunk.
        Length == n_snps_chunk.
        Fields:
          - chrom
          - pos
          - ref_allele
          - alt_allele

    Notes
    -----
    - Sample order (columns) is fixed and defined by `reader.sample_ids`.
    - SNP order (rows) follows the original file order after filtering.

    Example
    -------
    >>> for geno, sites in load_genotype_chunks("QC", chunk_size=20000, maf=0.05):
    ...     print(geno.shape)          # (m_chunk, n_samples)
    ...     print(sites[0].chrom, sites[0].pos)
    """

    # 1) Determine file type: BED or VCF
    if path_or_prefix.endswith(".vcf") or path_or_prefix.endswith(".vcf.gz"):
        reader = VcfChunkReader(
            path_or_prefix,
            float(maf),
            float(missing_rate),
        )
    else:
        # Otherwise treat it as PLINK BED prefix
        reader = BedChunkReader(
            path_or_prefix,
            float(maf),
            float(missing_rate),
        )

    # 2) Iterate until exhausted
    while True:
        out = reader.next_chunk(chunk_size)
        if out is None:
            break

        geno, sites = out

        # geno is already float32 C-contiguous memory managed by Rust
        # Convert to numpy array (zero-copy)
        geno_np = np.asarray(geno, dtype=np.float32)

        yield geno_np, sites
    
def inspect_genotype_file(path_or_prefix: str):
    """
    Inspect genotype input and return sample IDs and SNP count.

    This function is cheap and does NOT decode genotypes.

    Returns
    -------
    sample_ids : list[str]
        Sample / individual IDs in genotype column order.

    n_snps : int
        Number of SNPs in the file.
        - For PLINK BED: exact count from BIM.
        - For VCF     : counted by scanning variant lines.

    Notes
    -----
    - sample_ids correspond to:
        * .fam IID column (PLINK)
        * VCF #CHROM header columns
    - sample_ids do NOT contain SNP / site information.
    """
    # BED
    if not (path_or_prefix.endswith(".vcf") or path_or_prefix.endswith(".vcf.gz")):
        reader = BedChunkReader(path_or_prefix, 0.0, 1.0)
        return reader.sample_ids, reader.n_snps
    # VCF
    reader = VcfChunkReader(path_or_prefix, 0.0, 1.0)
    return reader.sample_ids, count_vcf_snps(path_or_prefix)


def _to_i8_snp_major(geno_chunk: np.ndarray) -> np.ndarray:
    """
    Convert a SNP-major genotype chunk to int8 encoding for streaming writers.

    Input
    -----
    geno_chunk : array-like, shape (m_chunk, n_samples)
        float32 or float64 genotypes.
        Missing genotypes are indicated by values < 0.

    Output
    ------
    gi8 : np.ndarray, dtype=int8, shape (m_chunk, n_samples)
        Genotype encoding:
          0 -> homozygous REF
          1 -> heterozygous
          2 -> homozygous ALT
         -9 -> missing

    Notes
    -----
    - Rounds imputed float values to nearest integer.
    - Clips values into [0, 2].
    - Output is C-contiguous and safe to pass into Rust.
    """
    g = np.asarray(geno_chunk)
    if g.ndim != 2:
        raise ValueError("geno_chunk must be 2D (m_chunk, n_samples)")

    if g.dtype == np.int8:
        return np.ascontiguousarray(g)

    miss = g < 0
    gi = np.rint(g).astype(np.int16, copy=False)
    gi = np.clip(gi, 0, 2).astype(np.int8, copy=False)
    gi = np.ascontiguousarray(gi)
    if np.any(miss):
        gi = gi.copy()
        gi[miss] = np.int8(-9)
    return gi

def _infer_format_from_out(out: str) -> str:
    """
    Infer output format from `out`.

    Returns
    -------
    "vcf"   : VCF or VCF.GZ
    "plink" : PLINK prefix (BED/BIM/FAM)
    """
    out = str(out).lower()
    if out.endswith(".vcf") or out.endswith(".vcf.gz"):
        return "vcf"
    return "plink"

def save_genotype_streaming(
    out: str,
    sample_ids: list[str],
    chunks,
    *,
    fmt: str = "auto",
    flush_every_chunks: int = 20,
    total_snps: int | None = None,
    desc: str = "Writing genotypes",
):
    """
    Unified streaming writer for genotype data.

    - If fmt="auto":
        * out endswith .vcf or .vcf.gz  -> write VCF (optionally gz if your Rust writer supports it)
        * otherwise                     -> write PLINK BED/BIM/FAM using `out` as prefix

    This function NEVER materializes the full genotype matrix in memory.

    Parameters
    ----------
    out : str
        Output target.
        - VCF: path like "out.vcf" or "out.vcf.gz"
        - PLINK: prefix like "out_prefix" (writes out_prefix.bed/.bim/.fam)

    sample_ids : list[str]
        Sample IDs (individual IDs). Must match genotype column order.

    chunks : iterator
        Yields:
            geno_chunk : np.ndarray (m_chunk, n_samples) (float32/float64/int8 all ok)
            sites      : list[SiteInfo] of length m_chunk

    fmt : {"auto", "vcf", "plink"}
        Force output format. "auto" infers from `out`.

    flush_every_chunks : int
        Flush periodically to reduce data loss risk for long runs.
        Set 0 to disable periodic flush.

    Notes
    -----
    - Genotype chunks are expected SNP-major: (m_chunk, n_samples)
    - Missing values should be < 0; writer will encode as ./.
    - For PLINK, phenotype is NOT written; Rust side should write -9 in FAM.
    """
    if fmt not in ("auto", "vcf", "plink"):
        raise ValueError("fmt must be one of: auto, vcf, plink")

    if not sample_ids:
        raise ValueError("sample_ids is empty")

    if fmt == "auto":
        fmt = _infer_format_from_out(out)

    sample_ids = list(map(str, sample_ids))

    # pick writer
    if fmt == "plink":
        w = PlinkStreamWriter(str(out), sample_ids, None)
    else:
        w = VcfStreamWriter(str(out), sample_ids)

    pbar = tqdm(
        total=total_snps,
        unit="SNP",
        desc=desc,
        disable=(total_snps is None),
    )

    written = 0
    k = 0
    try:
        for geno_chunk, sites in chunks:
            gi8 = _to_i8_snp_major(geno_chunk)
            w.write_chunk(gi8, list(sites))

            n = len(sites)
            written += n
            pbar.update(n)

            k += 1
            if flush_every_chunks > 0 and (k % flush_every_chunks == 0):
                w.flush()
    finally:
        w.close()
        pbar.close()