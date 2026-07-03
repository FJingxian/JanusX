from typing import Union
from contextlib import contextmanager
import gzip
import os
import warnings
import shutil
import sys
import time
import tempfile
import numpy as np
from tqdm import tqdm
from janusx.janusx import (
    BedChunkReader,
    HmpChunkReader,
    TxtChunkReader,
    convert_genotypes,
    count_hmp_snps,
    load_bed_2bit_packed as _load_bed_2bit_packed,
    PlinkStreamWriter,
    HmpStreamWriter,
    VcfStreamWriter,
    SiteInfo,
)
try:
    from janusx.janusx import load_bin01_packed as _load_bin01_packed
except Exception:
    _load_bin01_packed = None
try:
    from janusx.janusx import load_mbin_packed as _load_mbin_packed
except Exception:
    _load_mbin_packed = None
try:
    from janusx.janusx import scan_bed_2bit_packed_stats as _scan_bed_2bit_packed_stats
except Exception:
    _scan_bed_2bit_packed_stats = None
try:
    from janusx.janusx import prepare_bed_2bit_packed as _prepare_bed_2bit_packed
except Exception:
    _prepare_bed_2bit_packed = None

try:
    from janusx.script._common.progress import build_rich_progress as _build_rich_progress
except Exception:
    _build_rich_progress = None

from janusx.script._common.grmio import (
    genotype_cache_fallback_message as _genotype_cache_fallback_message,
    genotype_cache_permission_retry_message as _genotype_cache_permission_retry_message,
    genotype_cache_staged_input_message as _genotype_cache_staged_input_message,
)

_WARNED_BED_MODEL_FALLBACK = False
_GENO_CACHE_ENV = "JANUSX_CACHE_DIR"
_WARNED_CACHE_FALLBACK_KEYS: set[str] = set()


def _expanduser_unambiguous(path: Union[str, os.PathLike[str]]) -> str:
    """
    Expand only true home-directory syntax:
      "~", "~/...", "~\\..."

    Keep JanusX cache-like prefixes such as "~panel" literal.
    """
    s = str(path)
    if not (s == "~" or s.startswith("~/") or s.startswith("~\\")):
        return s
    return os.path.expanduser(s)


def load_bed_2bit_packed(prefix: str):
    """
    Load PLINK BED genotypes as packed 2-bit matrix plus per-SNP statistics.

    Returns
    -------
    packed : np.ndarray[uint8], shape (n_snps, ceil(n_samples/4))
        Raw BED 2-bit payload (header removed), SNP-major.
        Codes: 00->0, 10->1, 11->2, 01->missing.
    missing_rate : np.ndarray[float32], shape (n_snps,)
    maf : np.ndarray[float32], shape (n_snps,)
    std_denom : np.ndarray[float32], shape (n_snps,)
        sqrt(2 * p * (1 - p)), where p is ALT allele frequency on non-missing calls.
    n_samples : int
    """
    return _load_bed_2bit_packed(str(prefix))


def load_bin01_packed(path: str):
    """
    Load JanusX BIN01 cache as packed 0/1 bit matrix plus per-row group ids.

    Returns
    -------
    packed : np.ndarray[uint8], shape (n_rows, ceil(n_samples/8))
        1-bit packed payload, row-major.
    group_ids : np.ndarray[uint64], shape (n_rows,)
        Feature-group ids derived from site sidecar; rows sharing the same
        source SNP get the same id when sidecar metadata is available.
    n_samples : int
    """
    if _load_bin01_packed is None:
        raise RuntimeError(
            "load_bin01_packed is unavailable in Rust extension. "
            "Please rebuild/install JanusX."
        )
    return _load_bin01_packed(str(path))


def load_mbin_packed(path: str):
    """
    Load JanusX MBIN cache as packed 0/1 bit matrix plus per-row group ids.

    Returns
    -------
    packed : np.ndarray[uint8], shape (n_rows, ceil(n_samples/8))
        1-bit packed payload, row-major.
    group_ids : np.ndarray[uint64], shape (n_rows,)
        Feature-group ids where DOM/REC/HET rows from the same SNP share one id.
    n_samples : int
    """
    if _load_mbin_packed is None:
        raise RuntimeError(
            "load_mbin_packed is unavailable in Rust extension. "
            "Please rebuild/install JanusX."
        )
    return _load_mbin_packed(str(path))


def scan_bed_2bit_packed_stats(prefix: str):
    """
    Scan PLINK BED genotypes and return per-SNP statistics without copying
    the packed payload into Python-owned memory.

    Returns
    -------
    missing_rate : np.ndarray[float32], shape (n_snps,)
    maf : np.ndarray[float32], shape (n_snps,)
    std_denom : np.ndarray[float32], shape (n_snps,)
    row_flip : np.ndarray[bool], shape (n_snps,)
    het_rate : np.ndarray[float32], shape (n_snps,)
    n_samples : int
    """
    if _scan_bed_2bit_packed_stats is None:
        raise RuntimeError(
            "scan_bed_2bit_packed_stats is unavailable in Rust extension. "
            "Please rebuild/install JanusX."
        )
    return _scan_bed_2bit_packed_stats(str(prefix))


def prepare_bed_2bit_packed(
    prefix: str,
    *,
    maf_threshold: float = 0.0,
    max_missing_rate: float = 1.0,
    het_threshold: float = 0.0,
    snps_only: bool = False,
):
    """
    Load PLINK BED packed matrix and apply Rust-side filtering once.

    Returns
    -------
    packed : np.ndarray[uint8], shape (n_kept, ceil(n_samples/4))
    missing_rate : np.ndarray[float32], shape (n_kept,)
    maf : np.ndarray[float32], shape (n_kept,)
    std_denom : np.ndarray[float32], shape (n_kept,)
    row_flip : np.ndarray[bool], shape (n_kept,)
    site_keep : np.ndarray[bool], shape (n_total_sites,)
    n_samples : int
    n_total_sites : int
    """
    if _prepare_bed_2bit_packed is None:
        raise RuntimeError(
            "prepare_bed_2bit_packed is unavailable in Rust extension. "
            "Please rebuild/install JanusX."
        )
    try:
        return _prepare_bed_2bit_packed(
            str(prefix),
            maf_threshold=float(maf_threshold),
            max_missing_rate=float(max_missing_rate),
            het_threshold=float(het_threshold),
            snps_only=bool(snps_only),
        )
    except TypeError as ex:
        # Backward compatibility with older extension builds that don't expose
        # het_threshold in prepare_bed_2bit_packed.
        msg = str(ex).lower()
        if ("keyword" not in msg) and ("argument" not in msg) and ("unexpected" not in msg):
            raise
        return _prepare_bed_2bit_packed(
            str(prefix),
            maf_threshold=float(maf_threshold),
            max_missing_rate=float(max_missing_rate),
            snps_only=bool(snps_only),
        )


def set_genotype_cache_dir(cache_dir: Union[str, None]) -> Union[str, None]:
    """
    Set preferred writable cache directory for VCF/TXT/Numpy temporary caches.
    Returns normalized absolute path or None.
    """
    if cache_dir is None:
        os.environ.pop(_GENO_CACHE_ENV, None)
        return None
    s = str(cache_dir).strip()
    if s == "":
        os.environ.pop(_GENO_CACHE_ENV, None)
        return None
    path = os.path.abspath(_expanduser_unambiguous(s))
    os.makedirs(path, mode=0o755, exist_ok=True)
    os.environ[_GENO_CACHE_ENV] = path
    return path


def _cache_dir_from_env() -> Union[str, None]:
    raw = os.environ.get(_GENO_CACHE_ENV, "").strip()
    if raw == "":
        return None
    return os.path.abspath(_expanduser_unambiguous(raw))


def _dir_is_writable(path: str) -> bool:
    d = os.path.abspath(path or ".")
    return os.path.isdir(d) and os.access(d, os.W_OK | os.X_OK)


def _looks_like_permission_error(exc: Exception) -> bool:
    if isinstance(exc, PermissionError):
        return True
    msg = str(exc).lower()
    keys = (
        "permission denied",
        "operation not permitted",
        "read-only file system",
        "errno 13",
        "eacces",
        "access is denied",
    )
    return any(k in msg for k in keys)


def _warn_cache_once(key: str, msg: str) -> None:
    if key in _WARNED_CACHE_FALLBACK_KEYS:
        return
    _WARNED_CACHE_FALLBACK_KEYS.add(key)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _link_or_copy(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst) or ".", mode=0o755, exist_ok=True)
    try:
        os.symlink(src, dst)
        return
    except Exception:
        pass
    try:
        os.link(src, dst)
        return
    except Exception:
        pass
    shutil.copy2(src, dst)


def _cache_prefix(prefix: str, cache_dir: Union[str, None] = None) -> str:
    base = os.path.basename(prefix)
    if base.startswith("~"):
        bare = base[1:]
    else:
        bare = base
    if cache_dir is not None:
        return os.path.join(cache_dir, f"~{bare}")
    if base.startswith("~"):
        return prefix
    return os.path.join(os.path.dirname(prefix), f"~{base}")


def _vcf_cache_prefix(
    prefix: str,
    snps_only: bool = True,
    *,
    maf: Union[float, None] = None,
    missing_rate: Union[float, None] = None,
    cache_dir: Union[str, None] = None,
) -> str:
    """
    VCF->PLINK cache prefix.
    Keep source-cache independent from runtime maf/missing filters, but include
    SNP-only mode to avoid cache-mode stickiness across runs.
    """
    mode_tag = ".snp1" if bool(snps_only) else ".snp0"
    _ = maf
    _ = missing_rate
    return f"{_cache_prefix(prefix, cache_dir=cache_dir)}{mode_tag}"


@contextmanager
def _cache_lock(lock_key: str, timeout_s: float = 7200.0, poll_s: float = 0.2):
    lock_path = f"{lock_key}.lock"
    start = time.monotonic()
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
            break
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > timeout_s:
                    os.remove(lock_path)
                    continue
            except Exception:
                pass
            if (time.monotonic() - start) > timeout_s:
                raise TimeoutError(f"Timeout waiting cache lock: {lock_path}")
            time.sleep(poll_s)
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        except Exception:
            pass


def _is_plink_prefix(path_or_prefix: str) -> bool:
    return all(
        os.path.exists(f"{path_or_prefix}.{ext}") for ext in ("bed", "bim", "fam")
    )


def _strip_known_suffix(path_or_prefix: str) -> str:
    p = str(path_or_prefix)
    low = p.lower()
    if low.endswith(".hmp.gz"):
        return p[: -len(".hmp.gz")]
    if low.endswith(".vcf.gz"):
        return p[: -len(".vcf.gz")]
    for ext in (".hmp", ".vcf", ".txt", ".tsv", ".csv", ".npy", ".bin"):
        if low.endswith(ext):
            return p[: -len(ext)]
    return p


def _matrix_kind_from_path(path: str) -> Union[str, None]:
    low = str(path).lower()
    if low.endswith((".txt", ".tsv", ".csv")):
        return "txt"
    if low.endswith(".npy"):
        return "npy"
    if low.endswith(".bin"):
        return "bin"
    return None


def _normalize_matrix_prefix(prefix: str, kind: str) -> str:
    if kind in {"npy", "bin"}:
        base = os.path.basename(prefix)
        if base.startswith("~"):
            return os.path.join(os.path.dirname(prefix), base[1:])
    return prefix


def _plink_missing_files(prefix: str) -> list[str]:
    return [
        f"{prefix}.{ext}"
        for ext in ("bed", "bim", "fam")
        if not os.path.isfile(f"{prefix}.{ext}")
    ]


def _raise_plink_prefix_error(prefix: str) -> None:
    missing = _plink_missing_files(prefix)
    lines = [f"PLINK BED prefix is incomplete or not found: {prefix}"]
    for path in missing:
        lines.append(f"Missing file: {path}")
    raise ValueError("\n".join(lines))


def _peek_text_lines(path: str, limit: int = 32) -> list[str]:
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    lines: list[str] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for _ in range(max(1, int(limit))):
            line = handle.readline()
            if line == "":
                break
            lines.append(line.rstrip("\r\n"))
    return lines


def _looks_like_vcf_file(path: str) -> bool:
    try:
        lines = _peek_text_lines(path, limit=64)
    except Exception as ex:
        raise ValueError(f"Unable to read VCF header from '{path}': {ex}") from ex
    saw_fileformat = False
    saw_chrom = False
    for line in lines:
        row = line.strip()
        if not row:
            continue
        if row.startswith("##fileformat=VCF"):
            saw_fileformat = True
            continue
        if row.startswith("#CHROM"):
            saw_chrom = True
            break
        if not row.startswith("#"):
            break
    return saw_fileformat or saw_chrom


def _looks_like_hmp_file(path: str) -> bool:
    try:
        lines = _peek_text_lines(path, limit=8)
    except Exception as ex:
        raise ValueError(f"Unable to read HMP header from '{path}': {ex}") from ex
    for line in lines:
        row = line.strip()
        if not row:
            continue
        cols = [str(x).strip().lower() for x in row.split()]
        if len(cols) < 5:
            return False
        return cols[:5] == ["rs#", "alleles", "chrom", "pos", "strand"]
    return False


def _resolve_existing_text_source(
    path_or_prefix: str,
    *,
    suffixes: tuple[str, ...],
) -> tuple[str, Union[str, None]]:
    p = str(path_or_prefix)
    if os.path.isfile(p):
        return _strip_known_suffix(p), p
    prefix = _strip_known_suffix(p)
    for suffix in suffixes:
        cand = f"{prefix}{suffix}"
        if os.path.isfile(cand):
            return _strip_known_suffix(cand), cand
    return prefix, None


def _raise_missing_text_source(kind: str, path_or_prefix: str, *, suffixes: tuple[str, ...]) -> None:
    prefix = _strip_known_suffix(str(path_or_prefix))
    tried: list[str] = []
    p = str(path_or_prefix)
    if p != prefix or os.path.splitext(p)[1]:
        tried.append(p)
    else:
        tried.extend(f"{prefix}{suffix}" for suffix in suffixes)
    lines = [f"{kind.upper()} input not found: {path_or_prefix}"]
    for cand in tried:
        lines.append(f"Tried file: {cand}")
    raise ValueError("\n".join(lines))


def _raise_invalid_text_format(kind: str, path: str) -> None:
    label = "VCF" if kind == "vcf" else "HMP"
    fmt = "VCF" if kind == "vcf" else "HapMap"
    raise ValueError(f"{label} input format check failed: {path} does not look like a {fmt} file.")


def _file_matrix_candidates(
    path_or_prefix: str,
    *,
    expected_kind: Union[str, None],
) -> tuple[str, list[tuple[str, str]]]:
    p = str(path_or_prefix)
    matrix_kind = _matrix_kind_from_path(p)
    if matrix_kind is not None:
        return _strip_known_suffix(p), [(matrix_kind, p)]
    prefix = p
    if expected_kind == "txt":
        return prefix, [("txt", f"{prefix}.txt"), ("txt", f"{prefix}.tsv"), ("txt", f"{prefix}.csv")]
    if expected_kind == "npy":
        return prefix, [("npy", f"{prefix}.npy")]
    if expected_kind == "bin":
        return prefix, [("bin", f"{prefix}.bin")]
    return prefix, [
        ("npy", f"{prefix}.npy"),
        ("bin", f"{prefix}.bin"),
        ("txt", f"{prefix}.txt"),
        ("txt", f"{prefix}.tsv"),
        ("txt", f"{prefix}.csv"),
    ]


def _file_sidecar_candidates(prefix: str, kind: str) -> list[str]:
    if kind == "bin":
        return [f"{prefix}.bin.id", f"{prefix}.id", f"{prefix}.fam"]
    return [f"{prefix}.id", f"{prefix}.fam"]


def _raise_file_input_error(
    path_or_prefix: str,
    *,
    matrix_candidates: list[str],
    sidecar_candidates: list[str],
) -> None:
    lines = [f"FILE input is incomplete or not found: {path_or_prefix}"]
    for cand in matrix_candidates:
        lines.append(f"Missing matrix file: {cand}")
    for cand in sidecar_candidates:
        lines.append(f"Missing sample ID sidecar: {cand}")
    raise ValueError("\n".join(lines))


def _resolve_input(path_or_prefix: str, force_kind: Union[str, None] = None):
    """
    Resolve input kind and prefix.
    Returns (kind, prefix, src_path_or_none).
    kind: "plink" | "vcf" | "hmp" | "txt" | "npy" | "bin" | "unknown"
    """
    p = str(path_or_prefix)
    fk = str(force_kind or "").strip().lower()
    if fk in {"vcf", "hmp", "txt", "file", "npy", "bin", "plink", "bfile"}:
        if fk in {"plink", "bfile"}:
            prefix = p[:-4] if p.lower().endswith((".bed", ".bim", ".fam")) else p
            missing = _plink_missing_files(prefix)
            if missing:
                _raise_plink_prefix_error(prefix)
            return "plink", prefix, None
        if fk == "vcf":
            prefix, src_path = _resolve_existing_text_source(
                p,
                suffixes=(".vcf.gz", ".vcf"),
            )
            if src_path is None:
                _raise_missing_text_source("vcf", p, suffixes=(".vcf.gz", ".vcf"))
            if not _looks_like_vcf_file(src_path):
                _raise_invalid_text_format("vcf", src_path)
            return "vcf", prefix, src_path
        if fk == "hmp":
            prefix, src_path = _resolve_existing_text_source(
                p,
                suffixes=(".hmp.gz", ".hmp"),
            )
            if src_path is None:
                _raise_missing_text_source("hmp", p, suffixes=(".hmp.gz", ".hmp"))
            if not _looks_like_hmp_file(src_path):
                _raise_invalid_text_format("hmp", src_path)
            return "hmp", prefix, src_path
        expected_kind = None if fk == "file" else fk
        prefix, candidates = _file_matrix_candidates(p, expected_kind=expected_kind)
        matrix_path = None
        matrix_kind = None
        for cand_kind, cand_path in candidates:
            if expected_kind is not None and cand_kind != expected_kind:
                continue
            if os.path.isfile(cand_path):
                matrix_kind = cand_kind
                matrix_path = cand_path
                break
        if matrix_path is None or matrix_kind is None:
            _raise_file_input_error(
                p,
                matrix_candidates=[cand_path for _, cand_path in candidates],
                sidecar_candidates=[],
            )
        sidecar_candidates = _file_sidecar_candidates(prefix, matrix_kind)
        if not any(os.path.isfile(cand) for cand in sidecar_candidates):
            _raise_file_input_error(
                p,
                matrix_candidates=[],
                sidecar_candidates=sidecar_candidates,
            )
        return matrix_kind, _normalize_matrix_prefix(prefix, matrix_kind), matrix_path
    low = p.lower()

    if _is_plink_prefix(p):
        return "plink", p, None

    if low.endswith(".vcf.gz") or low.endswith(".vcf"):
        return "vcf", _strip_known_suffix(p), p

    if low.endswith(".hmp.gz") or low.endswith(".hmp"):
        return "hmp", _strip_known_suffix(p), p

    if low.endswith((".txt", ".tsv", ".csv")):
        return "txt", _strip_known_suffix(p), p

    if low.endswith(".npy"):
        prefix = _strip_known_suffix(p)
        # keep non-cache logical prefix for downstream cache bookkeeping
        base = os.path.basename(prefix)
        if base.startswith("~"):
            prefix = os.path.join(os.path.dirname(prefix), base[1:])
        return "npy", prefix, p
    if low.endswith(".bin"):
        prefix = _strip_known_suffix(p)
        base = os.path.basename(prefix)
        if base.startswith("~"):
            prefix = os.path.join(os.path.dirname(prefix), base[1:])
        return "bin", prefix, p

    prefix = p
    if _is_plink_prefix(prefix):
        return "plink", prefix, None
    if os.path.exists(f"{prefix}.vcf.gz"):
        return "vcf", prefix, f"{prefix}.vcf.gz"
    if os.path.exists(f"{prefix}.vcf"):
        return "vcf", prefix, f"{prefix}.vcf"
    if os.path.exists(f"{prefix}.hmp.gz"):
        return "hmp", prefix, f"{prefix}.hmp.gz"
    if os.path.exists(f"{prefix}.hmp"):
        return "hmp", prefix, f"{prefix}.hmp"
    if os.path.exists(f"{prefix}.txt"):
        return "txt", prefix, f"{prefix}.txt"
    if os.path.exists(f"{prefix}.tsv"):
        return "txt", prefix, f"{prefix}.tsv"
    if os.path.exists(f"{prefix}.csv"):
        return "txt", prefix, f"{prefix}.csv"
    # FILE layout: prefix.(npy/txt/tsv/csv/bin) + prefix.(id/fam) (+ optional prefix.site/bim)
    if os.path.exists(f"{prefix}.npy"):
        return "npy", prefix, f"{prefix}.npy"
    cache_prefix = _cache_prefix(prefix)
    if os.path.exists(f"{cache_prefix}.npy"):
        return "npy", prefix, f"{cache_prefix}.npy"
    if os.path.exists(f"{prefix}.bin"):
        return "bin", prefix, f"{prefix}.bin"
    if os.path.exists(f"{cache_prefix}.bin"):
        return "bin", prefix, f"{cache_prefix}.bin"
    if os.path.isfile(prefix):
        return "txt", prefix, prefix

    return "unknown", prefix, None


def _ensure_plink_cache_at(
    cache_prefix: str,
    prefix: str,
    vcf_path: str,
    *,
    snps_only: bool = True,
    maf: Union[float, None] = None,
    missing_rate: Union[float, None] = None,
    het: Union[float, None] = None,
    threads: int = 0,
) -> None:
    _ensure_plink_cache_from_source_at(
        cache_prefix,
        prefix,
        vcf_path,
        source_label="VCF",
        snps_only=snps_only,
        threads=threads,
    )


def _ensure_plink_cache_from_source_at(
    cache_prefix: str,
    prefix: str,
    source_path: str,
    *,
    source_label: str,
    snps_only: bool = True,
    threads: int = 0,
) -> None:
    eff_threads = 0 if threads is None else max(0, int(threads))
    cache_files = [f"{cache_prefix}.bed", f"{cache_prefix}.bim", f"{cache_prefix}.fam"]
    with _cache_lock(cache_prefix):
        cache_present = any(os.path.exists(p) for p in cache_files)
        if cache_present:
            rebuild_reason = None
            if not _is_plink_prefix(cache_prefix):
                rebuild_reason = (
                    f"Incomplete {source_label} cache detected for '{prefix}' at '{cache_prefix}'. "
                    "Removing cache files and rebuilding."
                )
            else:
                src_mtime = os.path.getmtime(source_path)
                cache_mtime = min(os.path.getmtime(p) for p in cache_files)
                if cache_mtime >= src_mtime:
                    return
                rebuild_reason = (
                    f"Detected stale {source_label} cache for '{prefix}': cache files are older than "
                    f"source '{source_path}'. Removing and rebuilding."
                )
            if rebuild_reason is not None:
                warnings.warn(rebuild_reason, RuntimeWarning, stacklevel=2)
                _remove_plink_cache_files(cache_prefix)

        try:
            convert_genotypes(
                source_path,
                cache_prefix,
                out_fmt="plink",
                threads=eff_threads,
                snps_only=bool(snps_only),
            )
        except Exception as ex:
            # Backward-compat fallback:
            # older Rust extension builds may not support HMP as convert_genotypes input.
            # In that case convert_genotypes misinterprets '*.hmp' as a PLINK prefix.
            ex_msg = str(ex).lower()
            source_low = str(source_path).lower()
            hmp_src = (str(source_label).upper() == "HMP") or source_low.endswith((".hmp", ".hmp.gz"))
            txt_src = source_low.endswith((".txt", ".tsv", ".csv", ".npy", ".bin"))
            looks_like_old_hmp_bug = (
                "plink prefix not found or missing .bed/.bim/.fam" in ex_msg
                and hmp_src
            )
            looks_like_txt_unsupported = (
                "plink prefix not found or missing .bed/.bim/.fam" in ex_msg
                and txt_src
            )
            if (str(source_label).upper() == "HMP") and looks_like_old_hmp_bug:
                warnings.warn(
                    (
                        "Current Rust extension does not support direct HMP->PLINK conversion "
                        "in convert_genotypes; falling back to streaming HMP conversion."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                _convert_hmp_to_plink_streaming(
                    source_path=source_path,
                    out_prefix=cache_prefix,
                    snps_only=bool(snps_only),
                )
            elif (str(source_label).upper() == "TXT") and looks_like_txt_unsupported:
                # TXT->PLINK direct conversion is unavailable in some builds.
                # Fall back to a stable streaming conversion path.
                delim = "," if source_low.endswith(".csv") else None
                _convert_txt_to_plink_streaming(
                    source_path=source_path,
                    out_prefix=cache_prefix,
                    snps_only=bool(snps_only),
                    delimiter=delim,
                )
            else:
                raise
        if not _is_plink_prefix(cache_prefix):
            raise RuntimeError(
                f"PLINK cache not created: {cache_prefix}.bed/.bim/.fam"
            )


def _convert_hmp_to_plink_streaming(
    *,
    source_path: str,
    out_prefix: str,
    snps_only: bool,
    chunk_size: int = 50000,
) -> None:
    """
    Python-side HMP -> PLINK streaming converter used as compatibility fallback.

    This path avoids full matrix materialization and works with older Rust wheels.
    """
    reader = HmpChunkReader(
        str(source_path),
        0.0,
        1.0,
        False,
        None,
        None,
        "add",
        0.0,
    )
    sample_ids = [str(x) for x in reader.sample_ids]
    if len(sample_ids) == 0:
        raise RuntimeError(f"HMP has no samples: {source_path}")

    def _iter_chunks():
        while True:
            out = reader.next_chunk(int(chunk_size))
            if out is None:
                break
            geno, sites = out
            geno_np = np.asarray(geno, dtype=np.float32)
            sites_list = list(sites)
            if bool(snps_only):
                keep_idx: list[int] = []
                for i, s in enumerate(sites_list):
                    r = str(getattr(s, "ref_allele", "")).upper()
                    a = str(getattr(s, "alt_allele", "")).upper()
                    if (
                        len(r) == 1
                        and len(a) == 1
                        and r in {"A", "C", "G", "T"}
                        and a in {"A", "C", "G", "T"}
                    ):
                        keep_idx.append(i)
                if len(keep_idx) == 0:
                    continue
                if len(keep_idx) != len(sites_list):
                    geno_np = np.asarray(geno_np[keep_idx, :], dtype=np.float32)
                    sites_list = [sites_list[i] for i in keep_idx]
            if geno_np.shape[0] == 0:
                continue
            yield geno_np, sites_list

    save_genotype_streaming(
        str(out_prefix),
        sample_ids,
        _iter_chunks(),
        fmt="plink",
        total_snps=None,
        desc="Caching HMP as PLINK",
    )


def _convert_txt_to_plink_streaming(
    *,
    source_path: str,
    out_prefix: str,
    snps_only: bool,
    delimiter: Union[str, None] = None,
    chunk_size: int = 50000,
) -> None:
    """
    Python-side TXT/TSV/CSV -> PLINK streaming converter used as compatibility fallback.
    """
    sample_ids, _ = inspect_genotype_file(str(source_path))
    sample_ids = [str(x) for x in sample_ids]
    if len(sample_ids) == 0:
        raise RuntimeError(f"TXT input has no samples: {source_path}")

    def _is_simple_base(a: str) -> bool:
        aa = str(a).strip().upper()
        return len(aa) == 1 and aa in {"A", "C", "G", "T"}

    def _iter_chunks():
        it = load_genotype_chunks(
            str(source_path),
            chunk_size=int(chunk_size),
            maf=0.0,
            missing_rate=1.0,
            impute=False,
            model="add",
            delimiter=delimiter,
        )
        for geno, sites in it:
            geno_np = np.asarray(geno, dtype=np.float32)
            sites_list = list(sites)
            if bool(snps_only):
                keep_idx: list[int] = []
                for i, s in enumerate(sites_list):
                    r = str(getattr(s, "ref_allele", ""))
                    a = str(getattr(s, "alt_allele", ""))
                    if _is_simple_base(r) and _is_simple_base(a):
                        keep_idx.append(i)
                if len(keep_idx) == 0:
                    continue
                if len(keep_idx) != len(sites_list):
                    geno_np = np.asarray(geno_np[keep_idx, :], dtype=np.float32)
                    sites_list = [sites_list[i] for i in keep_idx]
            if geno_np.shape[0] == 0:
                continue
            yield geno_np, sites_list

    save_genotype_streaming(
        str(out_prefix),
        sample_ids,
        _iter_chunks(),
        fmt="plink",
        total_snps=None,
        desc="Caching TXT as PLINK",
    )


def _ensure_plink_cache(
    prefix: str,
    vcf_path: str,
    *,
    snps_only: bool = True,
    maf: Union[float, None] = None,
    missing_rate: Union[float, None] = None,
    het: Union[float, None] = None,
) -> str:
    primary = _vcf_cache_prefix(
        prefix,
        snps_only=snps_only,
        maf=maf,
        missing_rate=missing_rate,
    )
    cache_dir = _cache_dir_from_env()
    fallback = (
        _vcf_cache_prefix(
            prefix,
            snps_only=snps_only,
            maf=maf,
            missing_rate=missing_rate,
            cache_dir=cache_dir,
        )
        if cache_dir
        else None
    )

    # If source-side cache directory is not writable, prefer fallback cache dir.
    if fallback is not None and not _dir_is_writable(os.path.dirname(primary) or "."):
        _warn_cache_once(
            f"vcf:{prefix}:{cache_dir}",
            _genotype_cache_fallback_message(
                os.path.dirname(primary) or ".",
                str(cache_dir),
            ),
        )
        _ensure_plink_cache_at(
            fallback,
            prefix,
            vcf_path,
            snps_only=bool(snps_only),
            maf=maf,
            missing_rate=missing_rate,
            het=het,
        )
        return fallback

    try:
        _ensure_plink_cache_at(
            primary,
            prefix,
            vcf_path,
            snps_only=bool(snps_only),
            maf=maf,
            missing_rate=missing_rate,
            het=het,
        )
        return primary
    except Exception as ex:
        if fallback is None or (fallback == primary) or (not _looks_like_permission_error(ex)):
            raise
        _warn_cache_once(
            f"vcf-perm:{prefix}:{cache_dir}",
            _genotype_cache_permission_retry_message(primary, str(cache_dir)),
        )
        _ensure_plink_cache_at(
            fallback,
            prefix,
            vcf_path,
            snps_only=bool(snps_only),
            maf=maf,
            missing_rate=missing_rate,
            het=het,
        )
        return fallback


def _ensure_plink_cache_for_cli_source(
    prefix: str,
    source_path: str,
    *,
    source_label: str,
    snps_only: bool = False,
    threads: int = 0,
) -> str:
    mode_tag = ".snp1" if bool(snps_only) else ".snp0"
    primary = f"{_cache_prefix(prefix)}{mode_tag}"
    cache_dir = _cache_dir_from_env()
    fallback = (
        f"{_cache_prefix(prefix, cache_dir=cache_dir)}{mode_tag}"
        if cache_dir
        else None
    )
    if fallback is None:
        auto_cache_dir = os.path.join(tempfile.gettempdir(), "janusx_cache")
        os.makedirs(auto_cache_dir, mode=0o755, exist_ok=True)
        fallback = f"{_cache_prefix(prefix, cache_dir=auto_cache_dir)}{mode_tag}"

    if fallback is not None and not _dir_is_writable(os.path.dirname(primary) or "."):
        _warn_cache_once(
            f"{source_label.lower()}:{prefix}:{cache_dir}",
            _genotype_cache_fallback_message(
                os.path.dirname(primary) or ".",
                str(cache_dir),
            ),
        )
        _ensure_plink_cache_from_source_at(
            fallback,
            prefix,
            source_path,
            source_label=source_label,
            snps_only=bool(snps_only),
            threads=threads,
        )
        return fallback

    try:
        _ensure_plink_cache_from_source_at(
            primary,
            prefix,
            source_path,
            source_label=source_label,
            snps_only=bool(snps_only),
            threads=threads,
        )
        return primary
    except Exception as ex:
        if fallback is None or (fallback == primary) or (not _looks_like_permission_error(ex)):
            raise
        _warn_cache_once(
            f"{source_label.lower()}-perm:{prefix}:{cache_dir}",
            _genotype_cache_permission_retry_message(primary, str(cache_dir)),
        )
        _ensure_plink_cache_from_source_at(
            fallback,
            prefix,
            source_path,
            source_label=source_label,
            snps_only=bool(snps_only),
            threads=threads,
        )
        return fallback


def _ensure_txt_npy_cache_for_cli(
    prefix: str,
    src_path: str,
    *,
    delimiter: Union[str, None] = None,
) -> str:
    txt_input = _prepare_txtlike_input_for_cache("txt", prefix, src_path)
    # Constructing TxtChunkReader triggers Rust-side txt->npy cache build/refresh.
    _ = TxtChunkReader(
        txt_input,
        delimiter,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    base_prefix = _strip_known_suffix(txt_input)
    candidates = [
        f"{base_prefix}.npy",
        f"{_cache_prefix(base_prefix)}.npy",
        f"{_cache_prefix(prefix)}.npy",
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return candidates[-1]


def prepare_cli_input_cache(
    path_or_prefix: str,
    *,
    snps_only: bool = False,
    delimiter: Union[str, None] = None,
    prefer_plink_for_txt: bool = False,
    force_kind: Union[str, None] = None,
    threads: int = 0,
) -> str:
    """
    Normalize CLI genotype input to cache-backed paths:
      - VCF/HMP -> PLINK BED prefix cache
      - TXT     -> NPY cache path (default) or PLINK BED prefix cache
      - others  -> unchanged
    """
    kind, prefix, src_path = _resolve_input(path_or_prefix, force_kind=force_kind)
    if kind == "vcf":
        if src_path is None:
            raise ValueError("VCF source path not found.")
        return _ensure_plink_cache_for_cli_source(
            prefix,
            src_path,
            source_label="VCF",
            snps_only=bool(snps_only),
            threads=threads,
        )
    if kind == "hmp":
        if src_path is None:
            raise ValueError("HMP source path not found.")
        return _ensure_plink_cache_for_cli_source(
            prefix,
            src_path,
            source_label="HMP",
            snps_only=bool(snps_only),
            threads=threads,
        )
    if kind == "txt":
        if src_path is None:
            raise ValueError("TXT source path not found.")
        if bool(prefer_plink_for_txt):
            return _ensure_plink_cache_for_cli_source(
                prefix,
                src_path,
                source_label="TXT",
                snps_only=bool(snps_only),
                threads=threads,
            )
        return _ensure_txt_npy_cache_for_cli(prefix, src_path, delimiter=delimiter)
    if kind == "npy":
        if bool(prefer_plink_for_txt):
            src = src_path if src_path is not None else f"{prefix}.npy"
            return _ensure_plink_cache_for_cli_source(
                prefix,
                src,
                source_label="TXT",
                snps_only=bool(snps_only),
                threads=threads,
            )
        if src_path is not None:
            return src_path
        return f"{prefix}.npy"
    if kind == "bin":
        if bool(prefer_plink_for_txt):
            src = src_path if src_path is not None else f"{prefix}.bin"
            return _ensure_plink_cache_for_cli_source(
                prefix,
                src,
                source_label="TXT",
                snps_only=bool(snps_only),
                threads=threads,
            )
        if src_path is not None:
            return src_path
        return f"{prefix}.bin"
    return path_or_prefix


def _remove_plink_cache_files(cache_prefix: str) -> None:
    for ext in ("bed", "bim", "fam"):
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


def _prepare_txtlike_input_for_cache(kind: str, prefix: str, src_path: Union[str, None]) -> str:
    """
    For txt/npy inputs on read-only genotype directories, stage lightweight links
    into JANUSX_CACHE_DIR and read from there so Rust-side cache files are writable.
    """
    txt_input = src_path if src_path is not None else prefix
    cache_dir = _cache_dir_from_env()
    if cache_dir is None:
        return txt_input
    primary_cache_prefix = _cache_prefix(prefix)
    if _dir_is_writable(os.path.dirname(primary_cache_prefix) or "."):
        return txt_input
    if src_path is None:
        return txt_input

    stage_root = os.path.join(cache_dir, "txt_cache_stage")
    os.makedirs(stage_root, mode=0o755, exist_ok=True)
    src = os.path.abspath(src_path)
    staged = os.path.join(stage_root, os.path.basename(src_path))
    try:
        _link_or_copy(src, staged)
    except Exception as e:
        warnings.warn(
            (
                f"Unable to stage read-only input into cache dir ({stage_root}): {e}. "
                "Proceeding with original input path."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return txt_input

    src_prefix = _strip_known_suffix(src_path)
    staged_prefix = _strip_known_suffix(staged)
    sidecar_suffixes = (
        ".bin.id",
        ".id",
        ".fam",
        ".bsite",
        ".bin.site",
        ".site",
        ".site.tsv",
        ".site.txt",
        ".site.csv",
        ".sites.tsv",
        ".sites.txt",
        ".sites.csv",
        ".bim",
    )
    # Also mirror possible cache-side sidecars if they already exist in source dir.
    src_cache_prefix = _cache_prefix(src_prefix)
    for suf in sidecar_suffixes:
        src_cands = [f"{src_prefix}{suf}", f"{src_cache_prefix}{suf}"]
        dst = f"{staged_prefix}{suf}"
        for cand in src_cands:
            if os.path.isfile(cand):
                try:
                    _link_or_copy(os.path.abspath(cand), dst)
                except Exception:
                    pass
                break

    _warn_cache_once(
        f"txt:{prefix}:{cache_dir}",
        _genotype_cache_staged_input_message(
            os.path.dirname(primary_cache_prefix) or ".",
            stage_root,
        ),
    )
    return staged


def _read_plink_fam_sample_ids(prefix: str) -> list[str]:
    fam_path = f"{prefix}.fam"
    sample_ids: list[str] = []
    with open(fam_path, "r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            row = line.strip()
            if not row:
                continue
            cols = row.split()
            if len(cols) < 2:
                raise ValueError(f"Malformed FAM line at {fam_path}:{line_no}: {row}")
            sample_ids.append(cols[1])
    if not sample_ids:
        raise ValueError(f"FAM contains zero samples: {fam_path}")
    return sample_ids


def _count_plink_bim_rows(prefix: str) -> int:
    bim_path = f"{prefix}.bim"
    n_snps = 0
    with open(bim_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                n_snps += 1
    if n_snps <= 0:
        raise ValueError(f"BIM contains zero variants: {bim_path}")
    return n_snps


def calc_decode_block_rows_from_memory_mb(
    n_samples: int,
    memory_mb: Union[int, float],
    *,
    elem_bytes: int = 4,
    buffers: int = 1,
    reserve_bytes: int = 0,
    min_rows: int = 1,
    max_rows: Union[int, None] = None,
) -> Union[int, None]:
    """
    Convert a memory target (MB) into SNP rows per decode block.

    Mirrors the Rust-side `block_rows_from_memory_target_mb(...)` policy.
    """
    if n_samples <= 0 or elem_bytes <= 0 or buffers <= 0:
        return None
    try:
        target_mb = float(memory_mb)
    except Exception:
        return None
    if not np.isfinite(target_mb) or target_mb <= 0.0:
        return None
    target_bytes = int(target_mb * 1024.0 * 1024.0)
    floor_rows = max(1, int(min_rows))
    max_rows_eff = None if max_rows is None else max(1, int(max_rows))
    if max_rows_eff is not None:
        floor_rows = min(floor_rows, max_rows_eff)
    if target_bytes <= 0:
        return int(floor_rows)
    if int(reserve_bytes) >= target_bytes:
        return int(floor_rows)
    bytes_per_row = max(1, int(n_samples)) * max(1, int(elem_bytes)) * max(1, int(buffers))
    usable_bytes = max(0, target_bytes - max(0, int(reserve_bytes)))
    rows = max(1, usable_bytes // max(1, bytes_per_row))
    rows = max(int(floor_rows), int(rows))
    if max_rows_eff is not None:
        rows = min(rows, max_rows_eff)
    return int(rows)


def calc_mmap_window_mb(
    n_samples: int,
    n_snps: int,
    block_rows: int,
    min_chunks: int = 2,
) -> Union[int,None]:
    """
    Compute a BED mmap window size (MB) to cover at least `min_chunks` decode blocks.

    Returns None when inputs are invalid.
    """
    if n_samples <= 0 or n_snps <= 0 or block_rows <= 0:
        return None
    target_snps = min(n_snps, max(1, min_chunks) * block_rows)
    bytes_per_snp = (n_samples + 3) // 4
    required_bytes = max(1, target_snps * bytes_per_snp)
    mb = (required_bytes + (1024 * 1024 - 1)) // (1024 * 1024)
    return int(max(1, mb))


def auto_mmap_window_mb(
    path_or_prefix: str,
    n_samples: int,
    n_snps: int,
    memory_mb: Union[int, float],
    *,
    elem_bytes: int = 4,
    buffers: int = 2,
    reserve_bytes: int = 0,
    min_rows: int = 1,
    max_rows: Union[int, None] = None,
    min_chunks: int = 2,
) -> Union[int,None]:
    """
    Compute mmap window size for BED inputs; return None for VCF/TXT inputs.
    """
    kind, _, _ = _resolve_input(path_or_prefix)
    if kind != "plink":
        return None
    block_rows = calc_decode_block_rows_from_memory_mb(
        n_samples,
        memory_mb,
        elem_bytes=elem_bytes,
        buffers=buffers,
        reserve_bytes=reserve_bytes,
        min_rows=min_rows,
        max_rows=max_rows,
    )
    if block_rows is None:
        return None
    return calc_mmap_window_mb(n_samples, n_snps, block_rows, min_chunks=min_chunks)

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
    chr_keys: Union[list[str], None] = None,
    bp_min: Union[int, None] = None,
    bp_max: Union[int, None] = None,
    ranges: Union[list[tuple[str, int, int]], None] = None,
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
        chr_keys=chr_keys,
        bp_min=bp_min,
        bp_max=bp_max,
        ranges=ranges,
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
        if (snp_sites is not None) and len(snp_sites) > 0:
            raise RuntimeError(
                "BedChunkReader extension does not support snp_sites selection. "
                "Please rebuild/reinstall janusx Rust extension."
            )
        if chr_keys is not None or bp_min is not None or bp_max is not None or ranges:
            raise RuntimeError(
                "BedChunkReader extension does not support chr/bp/ranges filtering. "
                "Please rebuild/reinstall janusx Rust extension."
            )
        if (str(model).strip().lower() != "add") or (abs(float(het) - 0.02) > 1e-12):
            raise RuntimeError(
                "BedChunkReader extension does not support model/het_threshold filtering. "
                "Please rebuild/reinstall janusx Rust extension."
            )
        legacy_kwargs = dict(base_kwargs)
        # Older extensions may not support snp_sites.
        legacy_kwargs.pop("snp_sites", None)
        legacy_kwargs.pop("chr_keys", None)
        legacy_kwargs.pop("bp_min", None)
        legacy_kwargs.pop("bp_max", None)
        legacy_kwargs.pop("ranges", None)
        reader = BedChunkReader(**legacy_kwargs)
        if not _WARNED_BED_MODEL_FALLBACK:
            warnings.warn(
                (
                    "BedChunkReader extension is in legacy compatibility mode; "
                    "please rebuild/reinstall janusx Rust extension."
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
        max_het = float(het)
        keep &= has_obs & (het_rate <= max_het)

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
    chr_keys: Union[list[str], None] = None,
    bp_min: Union[int, None] = None,
    bp_max: Union[int, None] = None,
    ranges: Union[list[tuple[str, int, int]], None] = None,
    sample_ids: Union[list[str] , None] = None,
    sample_indices: Union[list[int] , None] = None,
    mmap_window_mb: Union[int , None] = None,
    delimiter: Union[str , None] = None,
    force_kind: Union[str, None] = None,
):
    """
    High-level Python interface for reading genotype data in chunks
    using the Rust gfreader_rs backend.

    This function provides a streaming (iterator-based) interface and never
    loads the full genotype matrix into memory.

    It works for:
      - PLINK BED/BIM/FAM (SNP-major, on-disk)
      - VCF / VCF.GZ (converted once to cached PLINK BED/BIM/FAM: ~prefix.*)
      - HMP / HMP.GZ (prefer cached PLINK BED/BIM/FAM; fallback direct HMP streaming)
      - Numeric TXT matrix (converted once to cached NPY: ~prefix.npy)
      - FILE NPY/BIN matrix (read directly from prefix.npy or prefix.bin)

    Parameters
    ----------
    path_or_prefix : str
        Genotype source.
        - PLINK: prefix without extension, e.g. "data/QC"
        - VCF  : full path, e.g. "data/QC.vcf.gz" (cached as "data/~QC.*")
        - HMP  : full path, e.g. "data/QC.hmp.gz" (prefer cached as "data/~QC.*")
        - TXT  : matrix path, e.g. "data/matrix.txt" (cached as "data/~matrix.npy")
        - NPY/BIN: matrix path, e.g. "data/matrix.npy" or "data/matrix.bin"

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
    - VCF/HMP/TXT caching itself does not perform maf/missing filtering or imputation.
      Filters are applied during chunk streaming in Rust readers.
    - VCF cache uses a single base filename:
      `~prefix.bed/.bim/.fam`, with mtime-based stale detection and
      lock-based concurrent build safety.

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
    kind, prefix, src_path = _resolve_input(path_or_prefix, force_kind=force_kind)
    # 1) Resolve caches and dispatch
    if kind == "vcf":
        if src_path is None:
            raise ValueError("VCF source path not found.")
        cache_prefix = _ensure_plink_cache(
            prefix,
            src_path,
            snps_only=bool(snps_only),
            maf=float(maf),
            missing_rate=float(missing_rate),
            het=float(het),
        )

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
                chr_keys=chr_keys,
                bp_min=bp_min,
                bp_max=bp_max,
                ranges=ranges,
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
            cache_prefix = _ensure_plink_cache(
                prefix,
                src_path,
                snps_only=bool(snps_only),
                maf=float(maf),
                missing_rate=float(missing_rate),
                het=float(het),
            )
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
            chr_keys=chr_keys,
            bp_min=bp_min,
            bp_max=bp_max,
            ranges=ranges,
            sample_ids=sample_ids,
            sample_indices=sample_indices,
            mmap_window_mb=mmap_window_mb,
        )
        return

    if kind == "hmp":
        if src_path is None:
            raise ValueError("HMP source path not found.")

        # Prefer BED cache for HMP so downstream workloads (e.g. GWAS GRM) can
        # reuse packed-BED kernels. Keep a direct-HMP fallback for compatibility.
        try:
            cache_prefix = _ensure_plink_cache_for_cli_source(
                prefix,
                src_path,
                source_label="HMP",
                snps_only=bool(snps_only),
            )

            def _iter_cached_hmp_bed():
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
                    chr_keys=chr_keys,
                    bp_min=bp_min,
                    bp_max=bp_max,
                    ranges=ranges,
                    sample_ids=sample_ids,
                    sample_indices=sample_indices,
                    mmap_window_mb=mmap_window_mb,
                )

            try:
                yield from _iter_cached_hmp_bed()
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
                cache_prefix = _ensure_plink_cache_for_cli_source(
                    prefix,
                    src_path,
                    source_label="HMP",
                    snps_only=bool(snps_only),
                )
                yield from _iter_cached_hmp_bed()
            return
        except Exception as ex:
            warnings.warn(
                (
                    "HMP->PLINK cache path is unavailable; "
                    f"falling back to direct HMP streaming. reason={ex}"
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        mode_flags = int(snp_range is not None) + int(snp_indices is not None) + int(bim_range is not None) + int(snp_sites is not None)
        if mode_flags > 1:
            raise ValueError(
                "Provide only one of snp_range, snp_indices, bim_range, or snp_sites."
            )

        if snp_range is not None:
            if len(snp_range) != 2:
                raise ValueError("snp_range must be (start, end).")
            _start = int(snp_range[0])
            _end = int(snp_range[1])
            if _start < 0 or _end <= _start:
                raise ValueError("invalid snp_range: start must be >=0 and end>start.")
        else:
            _start = 0
            _end = 0

        index_set = None
        if snp_indices is not None:
            index_set = set(int(i) for i in snp_indices)
            if len(index_set) == 0:
                raise ValueError("snp_indices is empty.")

        site_set = None
        if snp_sites is not None:
            site_set = set((str(c), int(p)) for c, p in snp_sites)
            if len(site_set) == 0:
                raise ValueError("snp_sites is empty.")

        range_chrom = None
        range_start = None
        range_end = None
        if bim_range is not None:
            if len(bim_range) != 3:
                raise ValueError("bim_range must be (chrom, start_pos, end_pos).")
            range_chrom = str(bim_range[0])
            range_start = int(bim_range[1])
            range_end = int(bim_range[2])
            if range_start > range_end:
                raise ValueError("bim_range start > end.")

        reader = HmpChunkReader(
            src_path,
            float(maf),
            float(missing_rate),
            bool(impute),
            sample_ids,
            sample_indices,
            str(model),
            float(het),
        )
        row0 = 0
        while True:
            out = reader.next_chunk(chunk_size)
            if out is None:
                break
            geno, sites = out
            geno_np = np.asarray(geno, dtype=np.float32)
            sites_list = list(sites)
            n_rows = int(geno_np.shape[0])
            if mode_flags > 0:
                keep_idx: list[int] = []
                for i, s in enumerate(sites_list):
                    ridx = row0 + i
                    keep = True
                    if snp_range is not None:
                        keep = (_start <= ridx < _end)
                    elif index_set is not None:
                        keep = (ridx in index_set)
                    elif (range_chrom is not None) and (range_start is not None) and (range_end is not None):
                        keep = (str(s.chrom) == range_chrom) and (int(s.pos) >= range_start) and (int(s.pos) <= range_end)
                    elif site_set is not None:
                        keep = ((str(s.chrom), int(s.pos)) in site_set)
                    if keep:
                        keep_idx.append(i)
                row0 += n_rows
                if len(keep_idx) == 0:
                    continue
                geno_np = np.asarray(geno_np[keep_idx, :], dtype=np.float32)
                sites_list = [sites_list[i] for i in keep_idx]
            else:
                row0 += n_rows
            yield geno_np, sites_list
        return

    if kind in ("txt", "npy", "bin"):
        if mmap_window_mb is not None:
            raise ValueError("mmap_window_mb is only supported for PLINK BED inputs.")
        txt_input = _prepare_txtlike_input_for_cache(kind, prefix, src_path)
        reader = TxtChunkReader(
            txt_input,
            delimiter,
            snp_range,
            snp_indices,
            bim_range,
            snp_sites,
            sample_ids,
            sample_indices,
            float(maf),
            float(missing_rate),
            bool(impute),
            str(model),
            float(het),
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
            if geno_np.shape[0] == 0:
                continue
            yield geno_np, list(sites)
        return

    raise ValueError(
        "Unable to infer genotype input type. Provide a VCF path, "
        "an HMP path (.hmp/.hmp.gz), "
        "a PLINK prefix (.bed/.bim/.fam), or a FILE prefix/path "
        "(prefix.id/.fam + prefix.npy/.txt/.tsv/.csv/.bin)."
    )

def inspect_genotype_file(
    path_or_prefix: str,
    *,
    snps_only: bool = True,
    maf: float = 0.02,
    missing_rate: float = 0.05,
    het: float = 0.01,
    delimiter: Union[str , None] = None,
    force_kind: Union[str, None] = None,
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
        - For HMP     : count from cached PLINK BIM when cache path is available;
                        otherwise direct HMP row count.
        - For TXT     : matrix row count from cached NPY metadata.
        - For NPY/BIN : matrix row count from matrix header metadata.

    Notes
    -----
    - sample_ids correspond to:
        * .fam IID column (PLINK)
        * VCF #CHROM header columns (after PLINK cache is created)
        * HMP sample columns (after PLINK cache is created, or direct HMP reader fallback)
        * FILE sidecar IDs from prefix.id or prefix.fam
    - sample_ids do NOT contain SNP / site information.
    """
    kind, prefix, src_path = _resolve_input(path_or_prefix, force_kind=force_kind)

    if kind == "vcf":
        if src_path is None:
            raise ValueError("VCF source path not found.")
        cache_prefix = _ensure_plink_cache(
            prefix,
            src_path,
            snps_only=bool(snps_only),
            maf=float(maf),
            missing_rate=float(missing_rate),
            het=float(het),
        )
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
            cache_prefix = _ensure_plink_cache(
                prefix,
                src_path,
                snps_only=bool(snps_only),
                maf=float(maf),
                missing_rate=float(missing_rate),
                het=float(het),
            )
            reader = BedChunkReader(cache_prefix, 0.0, 1.0)
            return reader.sample_ids, reader.n_snps

    if kind == "hmp":
        if src_path is None:
            raise ValueError("HMP source path not found.")
        try:
            cache_prefix = _ensure_plink_cache_for_cli_source(
                prefix,
                src_path,
                source_label="HMP",
                snps_only=bool(snps_only),
            )
            reader = BedChunkReader(cache_prefix, 0.0, 1.0)
            return reader.sample_ids, reader.n_snps
        except Exception:
            # Compatibility fallback: keep direct HMP inspect available.
            reader = HmpChunkReader(
                src_path,
                0.0,
                1.0,
                False,
                None,
                None,
                "add",
                float(het),
            )
            return reader.sample_ids, int(count_hmp_snps(src_path))

    if kind in ("txt", "npy", "bin"):
        txt_input = _prepare_txtlike_input_for_cache(kind, prefix, src_path)
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
        return _read_plink_fam_sample_ids(prefix), _count_plink_bim_rows(prefix)

    raise ValueError(
        "Unable to infer genotype input type. Provide a VCF path, "
        "an HMP path (.hmp/.hmp.gz), "
        "a PLINK prefix (.bed/.bim/.fam), or a FILE prefix/path "
        "(prefix.id/.fam + prefix.npy/.txt/.tsv/.csv/.bin)."
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
    "hmp"   : HMP or HMP.GZ
    "plink" : PLINK prefix (BED/BIM/FAM)
    """
    out = str(out).lower()
    if out.endswith(".vcf") or out.endswith(".vcf.gz"):
        return "vcf"
    if out.endswith(".hmp") or out.endswith(".hmp.gz"):
        return "hmp"
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
        * out endswith .vcf or .vcf.gz  -> write VCF
        * out endswith .hmp or .hmp.gz  -> write HMP
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

    fmt : {"auto", "vcf", "hmp", "plink"}
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
    if fmt not in ("auto", "vcf", "hmp", "plink"):
        raise ValueError("fmt must be one of: auto, vcf, hmp, plink")

    if not sample_ids:
        raise ValueError("sample_ids is empty")

    if fmt == "auto":
        fmt = _infer_format_from_out(out)

    sample_ids = list(map(str, sample_ids))

    # pick writer
    if fmt == "plink":
        w = PlinkStreamWriter(str(out), sample_ids, None)
    elif fmt == "hmp":
        w = HmpStreamWriter(str(out), sample_ids)
    else:
        w = VcfStreamWriter(str(out), sample_ids)

    use_rich = bool((_build_rich_progress is not None) and getattr(sys.stdout, "isatty", lambda: False)())
    pbar = None
    progress = None
    task_id = None
    if use_rich:
        progress = _build_rich_progress(
            description_template="[green]{task.description}",
            show_bar=(total_snps is not None),
            show_percentage=(total_snps is not None),
            show_elapsed=True,
            show_remaining=False,
            bar_width=(40 if total_snps is not None else None),
            finished_text=" ",
            transient=True,
        )
        if progress is not None:
            task_id = progress.add_task(
                str(desc),
                total=(None if total_snps is None else float(total_snps)),
            )
            progress.start()
    else:
        pbar = tqdm(
            total=total_snps,
            unit="SNP",
            desc=desc,
            disable=(total_snps is None),
            bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                       "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
    if use_rich and progress is None:
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
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=n)
            elif pbar is not None:
                pbar.update(n)

            k += 1
            if flush_every_chunks > 0 and (k % flush_every_chunks == 0):
                w.flush()
    finally:
        w.close()
        if progress is not None:
            progress.stop()
        if pbar is not None:
            pbar.close()
