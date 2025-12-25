import numpy as np
from .gfreader_rs import BedChunkReader, VcfChunkReader, count_vcf_snps

def load_genotype_chunks(
    path_or_prefix: str,
    chunk_size: int = 50000,
    maf: float = 0.0,
    missing_rate: float = 1.0
):
    """
    Unified high-level Python interface for reading genotype data in chunks
    using the Rust gfreader_rs backend.

    Works for both PLINK BED files (.bed/.bim/.fam) and VCF/VCF.gz files.

    Parameters
    ----------
    path_or_prefix : str
        - For BED input: prefix path, e.g. "data/QC" (without .bed)
        - For VCF input: full file name, e.g. "data/QC.vcf.gz"
    chunk_size : int
        Number of SNPs to decode per iteration.
    maf : float
        MAF threshold. SNPs with MAF < maf are filtered.
    missing_rate : float
        Maximum allowed missing rate. SNPs exceeding this rate are filtered.

    Yields
    ------
    geno : np.ndarray (n_snps_chunk, n_samples), float32
        Decoded and filtered genotype matrix block.
        Genotypes include:
            - MAF flip
            - Mean imputation for missing genotypes
            - float32 dense output
    sites : list[SiteInfo]
        List of Rust SiteInfo objects for SNP metadata.

    Example
    -------
    >>> for geno, sites in load_genotype_chunks("QC", chunk_size=20000, maf=0.05):
    ...     print(geno.shape)
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
    Inspect genotype input file and return (n_samples, n_snps or None).

    For BED:
        returns (samples_id, n_snps)

    For VCF / VCF.gz:
        returns (samples_id, None) because SNP count cannot be known in advance.

    """
    # BED
    if not (path_or_prefix.endswith(".vcf") or path_or_prefix.endswith(".vcf.gz")):
        reader = BedChunkReader(path_or_prefix, 0.0, 1.0)
        return reader.sample_ids, reader.n_snps
    # VCF
    reader = VcfChunkReader(path_or_prefix, 0.0, 1.0)
    return reader.sample_ids, count_vcf_snps(path_or_prefix)