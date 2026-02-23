from typing import Union
import os
import warnings
import numpy as np
from tqdm import tqdm
from janusx.janusx import (
    BedChunkReader,
    TxtChunkReader,
    convert_genotypes,
    PlinkStreamWriter,
    VcfStreamWriter,
    SiteInfo,
)


def _cache_prefix(prefix: str) -> str:
    base = os.path.basename(prefix)
    if base.startswith("~"):
        return prefix
    return os.path.join(os.path.dirname(prefix), f"~{base}")


def _is_plink_prefix(path_or_prefix: str) -> bool:
    return all(
        os.path.exists(f"{path_or_prefix}.{ext}") for ext in ("bed", "bim", "fam")
    )


def _strip_known_suffix(path_or_prefix: str) -> str:
    p = str(path_or_prefix)
    low = p.lower()
    if low.endswith(".vcf.gz"):
        return p[: -len(".vcf.gz")]
    for ext in (".vcf", ".txt", ".tsv", ".csv", ".npy"):
        if low.endswith(ext):
            return p[: -len(ext)]
    return p


def _resolve_input(path_or_prefix: str):
    """
    Resolve input kind and prefix.
    Returns (kind, prefix, src_path_or_none).
    kind: "plink" | "vcf" | "txt" | "npy" | "unknown"
    """
    p = str(path_or_prefix)
    low = p.lower()

    if _is_plink_prefix(p):
        return "plink", p, None

    if low.endswith(".vcf.gz") or low.endswith(".vcf"):
        return "vcf", _strip_known_suffix(p), p

    if low.endswith((".txt", ".tsv", ".csv")):
        return "txt", _strip_known_suffix(p), p

    if low.endswith(".npy"):
        return "npy", _strip_known_suffix(p), p

    prefix = p
    if _is_plink_prefix(prefix):
        return "plink", prefix, None
    if os.path.exists(f"{prefix}.vcf.gz"):
        return "vcf", prefix, f"{prefix}.vcf.gz"
    if os.path.exists(f"{prefix}.vcf"):
        return "vcf", prefix, f"{prefix}.vcf"
    if os.path.exists(f"{prefix}.txt"):
        return "txt", prefix, f"{prefix}.txt"
    if os.path.exists(f"{prefix}.tsv"):
        return "txt", prefix, f"{prefix}.tsv"
    if os.path.exists(f"{prefix}.csv"):
        return "txt", prefix, f"{prefix}.csv"
    cache_prefix = _cache_prefix(prefix)
    if os.path.exists(f"{cache_prefix}.npy"):
        return "npy", prefix, f"{cache_prefix}.npy"
    if os.path.isfile(prefix):
        return "txt", prefix, prefix

    return "unknown", prefix, None


def _ensure_plink_cache(prefix: str, vcf_path: str):
    cache_prefix = _cache_prefix(prefix)
    cache_files = [f"{cache_prefix}.bed", f"{cache_prefix}.bim", f"{cache_prefix}.fam"]

    if any(os.path.exists(p) for p in cache_files):
        src_mtime = os.path.getmtime(vcf_path)
        cache_mtime = min(os.path.getmtime(p) for p in cache_files if os.path.exists(p))
        if cache_mtime < src_mtime:
            warnings.warn(
                (
                    f"Detected stale VCF cache for '{prefix}': cache files are older than "
                    f"source VCF '{vcf_path}'. Removing and rebuilding."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            for path in cache_files:
                if os.path.exists(path):
                    os.remove(path)

    if _is_plink_prefix(cache_prefix):
        return
    convert_genotypes(vcf_path, cache_prefix, out_fmt="plink")
    if not _is_plink_prefix(cache_prefix):
        raise RuntimeError(
            f"PLINK cache not created: {cache_prefix}.bed/.bim/.fam"
        )


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
    kind, _, _ = _resolve_input(path_or_prefix)
    if kind != "plink":
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
      - VCF / VCF.GZ (converted once to cached PLINK BED/BIM/FAM: ~prefix.*)
      - Numeric TXT matrix (converted once to cached NPY: ~prefix.npy)

    Parameters
    ----------
    path_or_prefix : str
        Genotype source.
        - PLINK: prefix without extension, e.g. "data/QC"
        - VCF  : full path, e.g. "data/QC.vcf.gz" (cached as "data/~QC.*")
        - TXT  : matrix path, e.g. "data/matrix.txt" (cached as "data/~matrix.npy")

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
    - VCF/TXT caching does not perform filtering or imputation.

    Selection
    ---------
    - snp_range: (start, end) 0-based, end exclusive (PLINK only)
    - snp_indices: list of 0-based SNP indices (PLINK only)
    - bim_range: (chrom, start_pos, end_pos), inclusive on positions (PLINK only)
    - sample_ids: list of sample IDs (PLINK .fam IID or VCF #CHROM header)
    - sample_indices: list of 0-based sample indices
    - mmap_window_mb: window size (MB) for BED mmap; supported for PLINK (incl. cached VCF), not TXT
    - delimiter: optional token delimiter used for TXT matrix inputs

    Example
    -------
    >>> for geno, sites in load_genotype_chunks("QC", chunk_size=20000, maf=0.05):
    ...     print(geno.shape)          # (m_chunk, n_samples)
    ...     print(sites[0].chrom, sites[0].pos)
    """
    chunk_size = int(chunk_size)
    kind, prefix, src_path = _resolve_input(path_or_prefix)
    # 1) Resolve caches and dispatch
    if kind == "vcf":
        if src_path is None:
            raise ValueError("VCF source path not found.")
        _ensure_plink_cache(prefix, src_path)
        cache_prefix = _cache_prefix(prefix)
        yield from bed_chunk_reader(
            cache_prefix,
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

    if kind == "plink":
        yield from bed_chunk_reader(
            prefix,
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

    if kind in ("txt", "npy"):
        if mmap_window_mb is not None:
            raise ValueError("mmap_window_mb is only supported for PLINK BED inputs.")
        txt_input = src_path if src_path is not None else prefix
        reader = TxtChunkReader(
            txt_input,
            delimiter,
            snp_range,
            snp_indices,
            bim_range,
            sample_ids,
            sample_indices,
        )
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
        return

    raise ValueError(
        "Unable to infer genotype input type. Provide a VCF path, "
        "a PLINK prefix (.bed/.bim/.fam), or a TXT matrix path (with header sample IDs)."
    )

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
        - For VCF     : count from cached PLINK BIM after conversion.
        - For TXT     : matrix row count from cached NPY metadata.

    Notes
    -----
    - sample_ids correspond to:
        * .fam IID column (PLINK)
        * VCF #CHROM header columns (after PLINK cache is created)
        * TXT header row tokens
    - sample_ids do NOT contain SNP / site information.
    """
    kind, prefix, src_path = _resolve_input(path_or_prefix)

    if kind == "vcf":
        if src_path is None:
            raise ValueError("VCF source path not found.")
        _ensure_plink_cache(prefix, src_path)
        cache_prefix = _cache_prefix(prefix)
        reader = BedChunkReader(cache_prefix, 0.0, 1.0)
        return reader.sample_ids, reader.n_snps

    if kind in ("txt", "npy"):
        txt_input = src_path if src_path is not None else prefix
        reader = TxtChunkReader(
            txt_input,
            delimiter,
            None,
            None,
            None,
            None,
            None,
        )
        return reader.sample_ids, reader.n_snps

    if kind == "plink":
        reader = BedChunkReader(prefix, 0.0, 1.0)
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
