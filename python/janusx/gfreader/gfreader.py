from typing import Union
import os
import json
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

_WARNED_BED_MODEL_FALLBACK = False
_PLINK_CACHE_META_VERSION = 2


def _cache_prefix(prefix: str) -> str:
    base = os.path.basename(prefix)
    if base.startswith("~"):
        return prefix
    return os.path.join(os.path.dirname(prefix), f"~{base}")


def _vcf_cache_prefix(prefix: str, snps_only: bool = True) -> str:
    """
    VCF->PLINK cache prefix.
    - snps_only=True : legacy SNP-only cache (~prefix.*)
    - snps_only=False: all-biallelic cache (~prefix.all.*)
    """
    base = _cache_prefix(prefix)
    return base if bool(snps_only) else f"{base}.all"


def _plink_cache_meta_path(cache_prefix: str) -> str:
    return f"{cache_prefix}.meta.json"


def _expected_plink_cache_meta(*, snps_only: bool) -> dict:
    return {
        "kind": "vcf_plink_cache",
        "version": int(_PLINK_CACHE_META_VERSION),
        "snps_only": bool(snps_only),
        # Keep REF==ALT rows in conversion; downstream MAF/missing filters decide.
        "keep_ref_eq_alt": True,
    }


def _load_plink_cache_meta(cache_prefix: str) -> dict:
    path = _plink_cache_meta_path(cache_prefix)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def _write_plink_cache_meta(cache_prefix: str, meta: dict) -> None:
    path = _plink_cache_meta_path(cache_prefix)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, sort_keys=True)
    os.replace(tmp, path)


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
        prefix = _strip_known_suffix(p)
        # keep non-cache logical prefix for downstream cache bookkeeping
        base = os.path.basename(prefix)
        if base.startswith("~"):
            prefix = os.path.join(os.path.dirname(prefix), base[1:])
        return "npy", prefix, p

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
    # New FILE layout: prefix.npy + prefix.id (+ optional prefix.site)
    if os.path.exists(f"{prefix}.npy"):
        return "npy", prefix, f"{prefix}.npy"
    cache_prefix = _cache_prefix(prefix)
    if os.path.exists(f"{cache_prefix}.npy"):
        return "npy", prefix, f"{cache_prefix}.npy"
    if os.path.isfile(prefix):
        return "txt", prefix, prefix

    return "unknown", prefix, None


def _ensure_plink_cache(prefix: str, vcf_path: str, *, snps_only: bool = True):
    cache_prefix = _vcf_cache_prefix(prefix, snps_only=snps_only)
    cache_files = [f"{cache_prefix}.bed", f"{cache_prefix}.bim", f"{cache_prefix}.fam"]
    expect_meta = _expected_plink_cache_meta(snps_only=bool(snps_only))

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
            _remove_plink_cache_files(cache_prefix)

    if _is_plink_prefix(cache_prefix):
        meta = _load_plink_cache_meta(cache_prefix)
        if meta == expect_meta:
            return
        warnings.warn(
            (
                f"Detected outdated VCF cache metadata for '{prefix}'. "
                "Removing cache files and rebuilding."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        _remove_plink_cache_files(cache_prefix)

    convert_genotypes(vcf_path, cache_prefix, out_fmt="plink", snps_only=bool(snps_only))
    if not _is_plink_prefix(cache_prefix):
        raise RuntimeError(
            f"PLINK cache not created: {cache_prefix}.bed/.bim/.fam"
        )
    _write_plink_cache_meta(cache_prefix, expect_meta)


def _remove_plink_cache_files(cache_prefix: str) -> None:
    for ext in ("bed", "bim", "fam", "meta.json"):
        path = f"{cache_prefix}.{ext}"
        if os.path.exists(path):
            os.remove(path)


def _is_likely_broken_plink_cache_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keys = (
        "malformed bim line",
        "malformed fam line",
        "cached bim row count mismatch",
        "bim row count mismatch",
        "invalid bed",
        "bed magic",
        "not enough bytes",
        "unexpected eof",
        "truncated",
    )
    return any(k in msg for k in keys)


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
    model: str = "add",
    het: float = 0.02,
    *,
    snp_range: Union[tuple[int, int] , None] = None,
    snp_indices: Union[list[int],None ]= None,
    bim_range: Union[tuple[str, int, int] , None] = None,
    snp_sites: Union[list[tuple[str, int]], None] = None,
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
      - snp_sites   : list of (chrom, pos), preserving input order

    Samples (mutually exclusive):
      - sample_ids     : list of IID strings from .fam
      - sample_indices : explicit 0-based sample indices
    mmap_window_mb:
      - limit BED mmap window size (MB); disables snp_range/snp_indices/bim_range
    """
    global _WARNED_BED_MODEL_FALLBACK
    base_kwargs = dict(
        prefix=prefix,
        maf_threshold=float(maf),
        max_missing_rate=float(missing_rate),
        fill_missing=bool(impute),
        snp_range=snp_range,
        snp_indices=snp_indices,
        bim_range=bim_range,
        snp_sites=snp_sites,
        sample_ids=sample_ids,
        sample_indices=sample_indices,
        mmap_window_mb=mmap_window_mb,
    )
    try:
        reader = BedChunkReader(
            **base_kwargs,
            model=model,
            het_threshold=float(het),
        )
    except TypeError:
        # Backward-compat for older compiled extensions that do not yet
        # expose model/het_threshold in BedChunkReader.
        legacy_kwargs = dict(base_kwargs)
        # Older extensions may not support snp_sites.
        legacy_kwargs.pop("snp_sites", None)
        reader = BedChunkReader(**legacy_kwargs)
        if (snp_sites is not None) and len(snp_sites) > 0:
            warnings.warn(
                (
                    "BedChunkReader extension does not support snp_sites yet; "
                    "falling back to legacy reader signature without snp_sites. "
                    "Please rebuild/reinstall janusx Rust extension to enable "
                    "chr/pos site-list selection."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        if (model != "add") and (not _WARNED_BED_MODEL_FALLBACK):
            warnings.warn(
                (
                    "BedChunkReader extension does not support model/het_threshold yet; "
                    "falling back to legacy reader signature. "
                    "Please rebuild/reinstall janusx Rust extension to enable "
                    "Rust-side heterozygosity filtering."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_BED_MODEL_FALLBACK = True

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
    snp_sites: Union[list[tuple[str, int]], None] = None,
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
        snp_sites,
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


def _process_txt_like_chunk(
    geno_np: np.ndarray,
    sites: list,
    *,
    maf: float,
    missing_rate: float,
    impute: bool,
    model: str,
    het: float,
) -> tuple[np.ndarray, list]:
    """
    Apply PLINK/VCF-like SNP filtering/imputation for TXT/NPY inputs.

    This mirrors Rust `process_snp_row` behavior so streaming GWAS/GRM
    semantics stay consistent across genotype input types.
    """
    g = np.asarray(geno_np, dtype=np.float32)
    if g.ndim != 2 or g.shape[0] == 0:
        return np.empty((0, g.shape[1] if g.ndim == 2 else 0), dtype=np.float32), []

    model_key = str(model).lower()
    if model_key not in {"add", "dom", "rec", "het"}:
        raise ValueError("model must be one of: add, dom, rec, het")
    if not (0.0 <= float(het) <= 0.5):
        raise ValueError("het must be within [0, 0.5]")

    n_samples = int(g.shape[1])
    if n_samples <= 0:
        return np.empty((0, 0), dtype=np.float32), []

    maf_thr = float(maf)
    miss_thr = float(missing_rate)
    fill_missing = bool(impute)

    valid = g >= 0.0
    non_missing = np.sum(valid, axis=1).astype(np.int64, copy=False)
    miss_rate = 1.0 - (non_missing.astype(np.float64) / float(n_samples))
    keep = miss_rate <= miss_thr

    has_obs = non_missing > 0
    alt_sum = np.sum(np.where(valid, g, 0.0), axis=1, dtype=np.float64)
    alt_freq = np.zeros(g.shape[0], dtype=np.float64)
    alt_freq[has_obs] = alt_sum[has_obs] / (
        2.0 * non_missing[has_obs].astype(np.float64)
    )

    # Rust parity: keep all-missing rows only when maf==0.
    if maf_thr > 0.0:
        keep &= has_obs

    apply_het = model_key != "add"
    if apply_het:
        het_count = np.sum(np.isclose(g, 1.0, atol=1e-6) & valid, axis=1).astype(
            np.int64, copy=False
        )
        het_rate = np.zeros(g.shape[0], dtype=np.float64)
        het_rate[has_obs] = (
            het_count[has_obs].astype(np.float64)
            / non_missing[has_obs].astype(np.float64)
        )
        low = float(het)
        keep &= has_obs & (het_rate >= low) & (het_rate <= (1.0 - low))

    # MAF flip for ALT freq > 0.5 and swap REF/ALT labels.
    flip = has_obs & (alt_freq > 0.5)
    if np.any(flip):
        flip_rows = np.where(flip)[0]
        g_sub = g[flip_rows]
        v_sub = valid[flip_rows]
        g_sub[v_sub] = 2.0 - g_sub[v_sub]
        g[flip_rows] = g_sub
        alt_sum[flip] = 2.0 * non_missing[flip].astype(np.float64) - alt_sum[flip]
        alt_freq[flip] = alt_sum[flip] / (
            2.0 * non_missing[flip].astype(np.float64)
        )

    maf_val = np.minimum(alt_freq, 1.0 - alt_freq)
    keep &= maf_val >= maf_thr

    if fill_missing:
        mean_g = np.zeros(g.shape[0], dtype=np.float32)
        mean_g[has_obs] = (
            alt_sum[has_obs] / non_missing[has_obs].astype(np.float64)
        ).astype(np.float32)
        miss_rows, miss_cols = np.where(~valid)
        if miss_rows.size > 0:
            g[miss_rows, miss_cols] = mean_g[miss_rows]
        all_missing_keep = (~has_obs) & keep
        if np.any(all_missing_keep):
            g[all_missing_keep] = 0.0

    keep_idx = np.where(keep)[0]
    if keep_idx.size == 0:
        return np.empty((0, n_samples), dtype=np.float32), []

    g_out = np.ascontiguousarray(g[keep_idx], dtype=np.float32)
    out_sites = []
    for idx in keep_idx.tolist():
        s = sites[idx]
        if flip[idx]:
            out_sites.append(
                SiteInfo(
                    str(s.chrom),
                    int(s.pos),
                    str(s.alt_allele),
                    str(s.ref_allele),
                )
            )
        else:
            out_sites.append(s)
    return g_out, out_sites


def load_genotype_chunks(
    path_or_prefix: str,
    chunk_size: int = 50000,
    maf: float = 0.0,
    missing_rate: float = 1.0,
    impute: bool = True,
    model: str = "add",
    het: float = 0.02,
    *,
    snps_only: bool = True,
    snp_range: Union[tuple[int, int] , None] = None,
    snp_indices: Union[list[int] , None] = None,
    bim_range: Union[tuple[str, int, int] , None] = None,
    snp_sites: Union[list[tuple[str, int]], None] = None,
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

    model : {"add", "dom", "rec", "het"}
        Genetic effect model flag used by Rust-side SNP filtering.
        Non-additive models enable heterozygosity filtering.

    het : float
        Heterozygosity filter threshold for non-additive models.
        SNPs with het rate outside [het, 1-het] are filtered.

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
    - VCF/TXT caching itself does not perform maf/missing filtering or imputation.
      Filters are applied during chunk streaming (Rust for PLINK/VCF, Python for TXT/NPY).
    - VCF cache can be SNP-only or all-biallelic depending on `snps_only`.

    Selection
    ---------
    - snp_range: (start, end) 0-based, end exclusive (PLINK only)
    - snp_indices: list of 0-based SNP indices (PLINK only)
    - bim_range: (chrom, start_pos, end_pos), inclusive on positions (PLINK only)
    - snp_sites: list of (chrom, pos), preserving input order (PLINK/TXT cached backends)
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
        _ensure_plink_cache(prefix, src_path, snps_only=bool(snps_only))
        cache_prefix = _vcf_cache_prefix(prefix, snps_only=bool(snps_only))

        def _iter_cached_bed():
            yield from bed_chunk_reader(
                cache_prefix,
                chunk_size=chunk_size,
                maf=maf,
                missing_rate=missing_rate,
                impute=impute,
                model=model,
                het=het,
                snp_range=snp_range,
                snp_indices=snp_indices,
                bim_range=bim_range,
                snp_sites=snp_sites,
                sample_ids=sample_ids,
                sample_indices=sample_indices,
                mmap_window_mb=mmap_window_mb,
            )

        try:
            yield from _iter_cached_bed()
        except Exception as ex:
            # If a prior run was interrupted, cache files may exist but be incomplete.
            # Rebuild once and retry transparently.
            if not _is_likely_broken_plink_cache_error(ex):
                raise
            warnings.warn(
                (
                    f"Detected broken PLINK cache for '{prefix}': {ex}. "
                    "Removing cache files and rebuilding."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            _remove_plink_cache_files(cache_prefix)
            _ensure_plink_cache(prefix, src_path, snps_only=bool(snps_only))
            yield from _iter_cached_bed()
        return

    if kind == "plink":
        yield from bed_chunk_reader(
            prefix,
            chunk_size=chunk_size,
            maf=maf,
            missing_rate=missing_rate,
            impute=impute,
            model=model,
            het=het,
            snp_range=snp_range,
            snp_indices=snp_indices,
            bim_range=bim_range,
            snp_sites=snp_sites,
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
            snp_sites,
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
            geno_np, sites = _process_txt_like_chunk(
                geno_np,
                list(sites),
                maf=float(maf),
                missing_rate=float(missing_rate),
                impute=bool(impute),
                model=str(model),
                het=float(het),
            )
            if geno_np.shape[0] == 0:
                continue
            yield geno_np, sites
        return

    raise ValueError(
        "Unable to infer genotype input type. Provide a VCF path, "
        "a PLINK prefix (.bed/.bim/.fam), or a FILE prefix/path "
        "(prefix.id + prefix.npy/.txt/.tsv/.csv)."
    )

def inspect_genotype_file(
    path_or_prefix: str,
    *,
    snps_only: bool = True,
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
        _ensure_plink_cache(prefix, src_path, snps_only=bool(snps_only))
        cache_prefix = _vcf_cache_prefix(prefix, snps_only=bool(snps_only))
        try:
            reader = BedChunkReader(cache_prefix, 0.0, 1.0)
            return reader.sample_ids, reader.n_snps
        except Exception as ex:
            if not _is_likely_broken_plink_cache_error(ex):
                raise
            warnings.warn(
                (
                    f"Detected broken PLINK cache for '{prefix}': {ex}. "
                    "Removing cache files and rebuilding."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            _remove_plink_cache_files(cache_prefix)
            _ensure_plink_cache(prefix, src_path, snps_only=bool(snps_only))
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
            None,
        )
        return reader.sample_ids, reader.n_snps

    if kind == "plink":
        reader = BedChunkReader(prefix, 0.0, 1.0)
        return reader.sample_ids, reader.n_snps

    raise ValueError(
        "Unable to infer genotype input type. Provide a VCF path, "
        "a PLINK prefix (.bed/.bim/.fam), or a FILE prefix/path "
        "(prefix.id + prefix.npy/.txt/.tsv/.csv)."
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
        bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                   "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
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
