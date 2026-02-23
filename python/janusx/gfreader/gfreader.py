from typing import Union
import os
import numpy as np
from tqdm import tqdm
from janusx.janusx import (
    BedChunkReader,
    VcfChunkReader,
    TxtChunkReader,
    count_vcf_snps,
    PlinkStreamWriter,
    VcfStreamWriter,
    SiteInfo,
)


def _is_vcf_input(path_or_prefix: str) -> bool:
    p = str(path_or_prefix).lower()
    return p.endswith(".vcf") or p.endswith(".vcf.gz")


def _is_plink_prefix(path_or_prefix: str) -> bool:
    return all(
        os.path.exists(f"{path_or_prefix}.{ext}") for ext in ("bed", "bim", "fam")
    )


def _is_text_matrix_input(path_or_prefix: str) -> bool:
    if _is_vcf_input(path_or_prefix):
        return False
    if _is_plink_prefix(path_or_prefix):
        return False
    if os.path.isfile(path_or_prefix):
        return True
    return False

def calc_mmap_window_mb(
    n_samples: int,
    n_snps: int,
    chunk_size: int,
    min_chunks: int = 2,
) -> Union[int,None]:
    """
    Compute a BED mmap window size (MB) to cover at least `min_chunks` chunks.

    Returns None when inputs are invalid.
    """
    if n_samples <= 0 or n_snps <= 0 or chunk_size <= 0:
        return None
    target_snps = min(n_snps, max(1, min_chunks) * chunk_size)
    bytes_per_snp = (n_samples + 3) // 4
    required_bytes = max(1, target_snps * bytes_per_snp)
    mb = (required_bytes + (1024 * 1024 - 1)) // (1024 * 1024)
    return int(max(1, mb))


def auto_mmap_window_mb(
    path_or_prefix: str,
    n_samples: int,
    n_snps: int,
    chunk_size: int,
    min_chunks: int = 2,
) -> Union[int,None]:
    """
    Compute mmap window size for BED inputs; return None for VCF/TXT inputs.
    """
    if _is_vcf_input(path_or_prefix) or _is_text_matrix_input(path_or_prefix):
        return None
    return calc_mmap_window_mb(n_samples, n_snps, chunk_size, min_chunks=min_chunks)

def bed_chunk_reader(
    prefix: str,
    chunk_size: int = 50000,
    maf: float = 0.0,
    missing_rate: float = 1.0,
    impute: bool = True,
    *,
    snp_range: Union[tuple[int, int] , None] = None,
    snp_indices: Union[list[int],None ]= None,
    bim_range: Union[tuple[str, int, int] , None] = None,
    sample_ids: Union[list[str] , None] = None,
    sample_indices: Union[list[int] , None] = None,
    mmap_window_mb: Union[int , None] = None,
):
    """
    Stream PLINK BED/BIM/FAM chunks with optional SNP/sample selection.

    Selection (mutually exclusive):
      - snp_range   : (start, end) with end exclusive (0-based)
      - snp_indices : explicit 0-based SNP indices
      - bim_range   : (chrom, start_pos, end_pos), inclusive on positions

    Samples (mutually exclusive):
      - sample_ids     : list of IID strings from .fam
      - sample_indices : explicit 0-based sample indices
    mmap_window_mb:
      - limit BED mmap window size (MB); disables snp_range/snp_indices/bim_range
    """
    reader = BedChunkReader(
        prefix,
        float(maf),
        float(missing_rate),
        bool(impute),
        snp_range,
        snp_indices,
        bim_range,
        sample_ids,
        sample_indices,
        mmap_window_mb,
    )

    while True:
        out = reader.next_chunk(chunk_size)
        if out is None:
            break
        geno, sites = out
        geno_np = np.asarray(geno, dtype=np.float32)
        yield geno_np, sites


def txt_chunk_reader(
    path: str,
    chunk_size: int = 50000,
    *,
    delimiter: Union[str , None] = None,
    snp_range: Union[tuple[int, int] , None] = None,
    snp_indices: Union[list[int] , None] = None,
    bim_range: Union[tuple[str, int, int] , None] = None,
    sample_ids: Union[list[str] , None] = None,
    sample_indices: Union[list[int] , None] = None,
):
    """
    Stream numeric TXT matrix chunks (float32) via Rust TXT->NPY->mmap backend.

    - The first non-empty row is sample IDs (header).
    - Remaining rows are SNP-major numeric values (float32-compatible).
    - Site metadata is always defaulted by Rust:
      chrom='N', pos=1..m, ref='N', alt='N'.
    """
    reader = TxtChunkReader(
        path,
        delimiter,
        snp_range,
        snp_indices,
        bim_range,
        sample_ids,
        sample_indices,
    )

    while True:
        out = reader.next_chunk(chunk_size)
        if out is None:
            break
        geno, sites = out
        geno_np = np.asarray(geno, dtype=np.float32)
        yield geno_np, sites


def load_genotype_chunks(
    path_or_prefix: str,
    chunk_size: int = 50000,
    maf: float = 0.0,
    missing_rate: float = 1.0,
    impute: bool = True,
    *,
    snp_range: Union[tuple[int, int] , None] = None,
    snp_indices: Union[list[int] , None] = None,
    bim_range: Union[tuple[str, int, int] , None] = None,
    sample_ids: Union[list[str] , None] = None,
    sample_indices: Union[list[int] , None] = None,
    mmap_window_mb: Union[int , None] = None,
    delimiter: Union[str , None] = None,
):
    """
    High-level Python interface for reading genotype data in chunks
    using the Rust gfreader_rs backend.

    This function provides a streaming (iterator-based) interface and never
    loads the full genotype matrix into memory.

    It works for:
      - PLINK BED/BIM/FAM (SNP-major, on-disk)
      - VCF / VCF.GZ (text or compressed)
      - Numeric TXT matrix (float32 via TXT->NPY->mmap)

    Parameters
    ----------
    path_or_prefix : str
        Genotype source.
        - PLINK: prefix without extension, e.g. "data/QC"
        - VCF  : full path, e.g. "data/QC.vcf.gz"
        - TXT  : matrix path, e.g. "data/matrix.txt" (header = sample IDs)

    chunk_size : int
        Number of SNPs (rows) decoded per iteration.

    maf : float
        Minor allele frequency threshold.
        SNPs with MAF < maf are filtered during decoding in Rust.

    missing_rate : float
        Maximum allowed missing genotype rate per SNP.
        SNPs exceeding this threshold are filtered out.

    impute : bool
        Whether to mean-impute missing genotypes.
        If False, missing values remain negative (e.g. -9).

    Yields
    ------
    geno : np.ndarray, shape (n_snps_chunk, n_samples), dtype=float32
        SNP-major genotype block.
        Each block is:
          - already MAF-flipped (ALT freq <= 0.5)
          - mean-imputed for missing genotypes (if impute=True)
          - dense float32 array backed by Rust memory

    sites : list[SiteInfo]
        Metadata objects for each SNP in the chunk.
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

    Selection
    ---------
    - snp_range: (start, end) 0-based, end exclusive (PLINK only)
    - snp_indices: list of 0-based SNP indices (PLINK only)
    - bim_range: (chrom, start_pos, end_pos), inclusive on positions (PLINK only)
    - sample_ids: list of sample IDs (PLINK .fam IID or VCF #CHROM header)
    - sample_indices: list of 0-based sample indices
    - mmap_window_mb: window size (MB) for BED mmap; not supported for VCF/TXT
    - delimiter: optional token delimiter used for TXT matrix inputs

    Example
    -------
    >>> for geno, sites in load_genotype_chunks("QC", chunk_size=20000, maf=0.05):
    ...     print(geno.shape)          # (m_chunk, n_samples)
    ...     print(sites[0].chrom, sites[0].pos)
    """
    chunk_size = int(chunk_size)
    # 1) Determine file type: BED, VCF, or numeric TXT matrix
    if _is_vcf_input(path_or_prefix):
        if mmap_window_mb is not None:
            raise ValueError("mmap_window_mb is only supported for PLINK BED inputs.")
        if any(v is not None for v in (snp_range, snp_indices, bim_range)):
            raise ValueError("SNP selection is only supported for PLINK BED/BIM/FAM inputs.")
        reader = VcfChunkReader(
            path_or_prefix,
            float(maf),
            float(missing_rate),
            bool(impute),
            sample_ids,
            sample_indices,
        )
    elif _is_text_matrix_input(path_or_prefix):
        if mmap_window_mb is not None:
            raise ValueError("mmap_window_mb is only supported for PLINK BED inputs.")
        reader = TxtChunkReader(
            path_or_prefix,
            delimiter,
            snp_range,
            snp_indices,
            bim_range,
            sample_ids,
            sample_indices,
        )
    else:
        # Otherwise treat it as PLINK BED prefix
        yield from bed_chunk_reader(
            path_or_prefix,
            chunk_size=chunk_size,
            maf=maf,
            missing_rate=missing_rate,
            impute=impute,
            snp_range=snp_range,
            snp_indices=snp_indices,
            bim_range=bim_range,
            sample_ids=sample_ids,
            sample_indices=sample_indices,
            mmap_window_mb=mmap_window_mb,
        )
        return

    # 2) Iterate until exhausted
    while True:
        out = reader.next_chunk(chunk_size)
        if out is None:
            break

        geno, sites = out

        # geno is already float32 C-contiguous memory managed by Rust
        # Convert to NumPy array (zero-copy)
        geno_np = np.asarray(geno, dtype=np.float32)

        yield geno_np, sites

def inspect_genotype_file(
    path_or_prefix: str,
    *,
    delimiter: Union[str , None] = None,
):
    """
    Inspect the genotype input and return sample IDs and the SNP count.

    This function is lightweight and does not decode genotypes.

    Returns
    -------
    sample_ids : list[str]
        Sample / individual IDs in genotype column order.

    n_snps : int
        Number of SNPs in the file.
        - For PLINK BED: exact count from BIM.
        - For VCF     : counted by scanning variant lines.
        - For TXT     : matrix row count from cached NPY metadata.

    Notes
    -----
    - sample_ids correspond to:
        * .fam IID column (PLINK)
        * VCF #CHROM header columns
        * TXT header row tokens
    - sample_ids do NOT contain SNP / site information.
    """
    if _is_vcf_input(path_or_prefix):
        reader = VcfChunkReader(path_or_prefix, 0.0, 1.0)
        return reader.sample_ids, count_vcf_snps(path_or_prefix)

    if _is_text_matrix_input(path_or_prefix):
        reader = TxtChunkReader(
            path_or_prefix,
            delimiter,
            None,
            None,
            None,
            None,
            None,
        )
        return reader.sample_ids, reader.n_snps

    # BED
    if _is_plink_prefix(path_or_prefix):
        reader = BedChunkReader(path_or_prefix, 0.0, 1.0)
        return reader.sample_ids, reader.n_snps

    raise ValueError(
        "Unable to infer genotype input type. Provide a VCF path, "
        "a PLINK prefix (.bed/.bim/.fam), or a TXT matrix path (with header sample IDs)."
    )


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
    total_snps: Union[int , None] = None,
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
