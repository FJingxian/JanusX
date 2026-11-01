"""Versioned GWAS null-model sidecar schema and matching helpers.

The sidecar is deliberately implemented with the Python standard library only.
It is embedded in the human-readable GWAS log between versioned sentinels so
future readers can ignore ordinary log text and older logs can remain valid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from importlib import metadata as importlib_metadata
import json
import logging
import math
from pathlib import Path
import re
import struct
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


SIDECAR_SCHEMA_V1 = "janusx.gwas-null-model/v1"
SIDECAR_BEGIN_V1 = "[JANUSX_GWAS_NULL_MODEL_V1_BEGIN]"
SIDECAR_END_V1 = "[JANUSX_GWAS_NULL_MODEL_V1_END]"

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_HASH_LENGTH = struct.Struct(">Q")
_HASH_CHUNK_SIZE = 1024 * 1024


class SidecarError(Exception):
    """Base class for sidecar discovery and format errors."""


class SidecarFormatError(SidecarError, ValueError):
    """Raised when a sidecar block is incomplete or violates its schema."""


class SidecarNotFound(SidecarError, LookupError):
    """Raised when no sidecar matches a result file."""


class SidecarAmbiguous(SidecarError, LookupError):
    """Raised when more than one sidecar matches a result file."""


class FineMapSkip(Exception):
    """Expected sidecar compatibility failure that skips fine-mapping."""


@dataclass(frozen=True)
class FileFingerprintV1:
    canonical_path: str
    basename: str
    size_bytes: int
    sha256: str


@dataclass
class GwasSidecarRunContext:
    genotype_prefix: Path
    phenotype_file: Path
    phenotype_id_column: str
    covariate_file: Path | None
    covariate_columns: tuple[str, ...]
    kinship_file: Path
    kinship_id_file: Path
    genotype_filters: dict[str, object]
    fingerprint_cache: dict[Path, FileFingerprintV1] = field(default_factory=dict)
    # When present, this is the exact numerical Q/PC matrix used by GWAS,
    # persisted in the workflow's versioned NPZ source format.  The sidecar
    # stores only its fingerprint and column identity; values remain external
    # to the file-only log block.
    qmatrix_file: Path | None = None
    qmatrix_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class GwasNullModelSidecarV1:
    schema: str
    created_at: str
    janusx_version: str
    result: FileFingerprintV1
    model: str
    trait: str
    z_columns: tuple[str, str]
    effective_snp_count: int
    genotype_prefix: str
    genotype_files: tuple[FileFingerprintV1, ...]
    phenotype_file: FileFingerprintV1
    phenotype_id_column: str
    phenotype_trait_column: str
    covariate_file: FileFingerprintV1 | None
    covariate_columns: tuple[str, ...]
    kinship_file: FileFingerprintV1
    kinship_id_file: FileFingerprintV1
    kinship_format: str
    kinship_shape: tuple[int, int]
    sample_count: int
    sample_order_sha256: str
    lambda_null: float
    sigma_g2: float | None
    sigma_e2: float | None
    pve: float
    grm_trace_mean: float
    fixed_effect_columns: tuple[str, ...]
    genotype_filters: dict[str, object]
    allele_coding: str
    qmatrix_file: FileFingerprintV1 | None = None
    qmatrix_columns: tuple[str, ...] = ()
    phenotype_trait_index: int | None = None
    # Raw phenotype data-column index before any -n/--ncol subset or reorder.
    # The existing phenotype_trait_index is retained for old records and is
    # populated with this same raw index for newly emitted records.
    phenotype_trait_source_index: int | None = None


__all__ = [
    "FileFingerprintV1",
    "GwasSidecarRunContext",
    "GwasNullModelSidecarV1",
    "SIDECAR_BEGIN_V1",
    "SIDECAR_END_V1",
    "SIDECAR_SCHEMA_V1",
    "FineMapSkip",
    "SidecarAmbiguous",
    "SidecarError",
    "SidecarFormatError",
    "SidecarNotFound",
    "build_fvlmm_sidecar",
    "discover_matching_sidecar",
    "emit_sidecar_to_file_log",
    "find_matching_sidecar",
    "fingerprint_file",
    "hash_ordered_sample_ids",
    "parse_sidecar_blocks",
    "serialize_sidecar_block",
    "validate_sidecar_dependencies",
]


def fingerprint_file(path: str | Path) -> FileFingerprintV1:
    """Return the canonical path, size, and streamed SHA-256 of *path*."""

    canonical = Path(path).expanduser().resolve(strict=True)
    stat_result = canonical.stat()
    digest = hashlib.sha256()
    with canonical.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return FileFingerprintV1(
        canonical_path=str(canonical),
        basename=canonical.name,
        size_bytes=int(stat_result.st_size),
        sha256=digest.hexdigest(),
    )


def hash_ordered_sample_ids(sample_ids: Iterable[str]) -> str:
    """Hash sample IDs using an unambiguous 8-byte length-prefixed encoding."""

    digest = hashlib.sha256()
    for sample_id in sample_ids:
        if not isinstance(sample_id, str):
            raise TypeError("sample IDs must be strings")
        encoded = sample_id.encode("utf-8")
        digest.update(_HASH_LENGTH.pack(len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def build_fvlmm_sidecar(
    context: GwasSidecarRunContext,
    result_file: str | Path,
    trait: str,
    sample_ids: Iterable[str],
    lambda_null: float,
    sigma_g2: float | None,
    sigma_e2: float | None,
    pve: float,
    grm_trace_mean: float,
    effective_snp_count: int,
    phenotype_trait_index: int | None = None,
    phenotype_trait_source_index: int | None = None,
) -> GwasNullModelSidecarV1:
    """Build one FvLMM sidecar from the finalized result and fitted null."""

    if not isinstance(context, GwasSidecarRunContext):
        raise TypeError("context must be GwasSidecarRunContext")

    result_path = Path(result_file).expanduser()
    if not result_path.is_file():
        raise FileNotFoundError(f"FvLMM result file not found: {result_path}")
    if result_path.stat().st_size <= 0:
        raise ValueError(f"FvLMM result file is empty: {result_path}")

    ordered_sample_ids = list(sample_ids)
    if len(ordered_sample_ids) == 0:
        raise ValueError("FvLMM sidecar sample IDs must not be empty")

    genotype_prefix = Path(context.genotype_prefix).expanduser()
    genotype_prefix = _normalize_genotype_prefix(genotype_prefix)
    genotype_suffixes = _plink_source_suffixes(genotype_prefix)
    genotype_paths = tuple(Path(f"{genotype_prefix}{suffix}") for suffix in genotype_suffixes)

    kinship_id_path = Path(context.kinship_id_file).expanduser()
    kinship_shape_n = _count_nonempty_lines(kinship_id_path)
    if kinship_shape_n <= 0:
        raise ValueError(f"kinship ID file is empty: {kinship_id_path}")

    covariate_columns = tuple(str(column) for column in context.covariate_columns)
    qmatrix_columns = tuple(str(column) for column in context.qmatrix_columns)
    fixed_effect_columns = ("Intercept",) + covariate_columns
    result_fingerprint = fingerprint_file(result_path)

    record = GwasNullModelSidecarV1(
        schema=SIDECAR_SCHEMA_V1,
        created_at=_sidecar_created_at(),
        janusx_version=_janusx_version(),
        result=result_fingerprint,
        model="fvlmm",
        trait=str(trait),
        z_columns=("beta", "se"),
        effective_snp_count=int(effective_snp_count),
        genotype_prefix=str(genotype_prefix.resolve()),
        genotype_files=tuple(
            _cached_fingerprint(context, path) for path in genotype_paths
        ),
        phenotype_file=_cached_fingerprint(context, context.phenotype_file),
        phenotype_id_column=str(context.phenotype_id_column),
        phenotype_trait_column=str(trait),
        covariate_file=(
            None
            if context.covariate_file is None
            else _cached_fingerprint(context, context.covariate_file)
        ),
        covariate_columns=covariate_columns,
        kinship_file=_cached_fingerprint(context, context.kinship_file),
        kinship_id_file=_cached_fingerprint(context, kinship_id_path),
        kinship_format="dense",
        kinship_shape=(kinship_shape_n, kinship_shape_n),
        sample_count=len(ordered_sample_ids),
        sample_order_sha256=hash_ordered_sample_ids(ordered_sample_ids),
        lambda_null=float(lambda_null),
        sigma_g2=(None if sigma_g2 is None else float(sigma_g2)),
        sigma_e2=(None if sigma_e2 is None else float(sigma_e2)),
        pve=float(pve),
        grm_trace_mean=float(grm_trace_mean),
        fixed_effect_columns=fixed_effect_columns,
        genotype_filters=dict(context.genotype_filters),
        allele_coding="A1_effect",
        qmatrix_file=(
            None
            if context.qmatrix_file is None
            else _cached_fingerprint(context, context.qmatrix_file)
        ),
        qmatrix_columns=qmatrix_columns,
        phenotype_trait_index=(
            None
            if phenotype_trait_index is None
            else int(phenotype_trait_index)
        ),
        phenotype_trait_source_index=(
            None
            if phenotype_trait_source_index is None
            else int(phenotype_trait_source_index)
        ),
    )
    _validate_record(record)
    return record


def emit_sidecar_to_file_log(
    logger: Any, record: GwasNullModelSidecarV1
) -> None:
    """Write one serialized sidecar block through file handlers only."""

    block = serialize_sidecar_block(record)
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger")
    if not logger.isEnabledFor(logging.INFO):
        return

    handlers: list[logging.FileHandler] = []
    seen_handlers: set[int] = set()
    current: logging.Logger | None = logger
    while current is not None:
        for handler in current.handlers:
            if not isinstance(handler, logging.FileHandler):
                continue
            handler_id = id(handler)
            if handler_id in seen_handlers:
                continue
            seen_handlers.add(handler_id)
            if handler.level <= logging.INFO:
                handlers.append(handler)
        if not current.propagate:
            break
        current = current.parent

    if not handlers:
        raise RuntimeError("no INFO-level file handler available for sidecar log")

    log_record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        block.rstrip("\n"),
        (),
        None,
    )
    for handler in handlers:
        handler.handle(log_record)


def _cached_fingerprint(
    context: GwasSidecarRunContext, path: str | Path
) -> FileFingerprintV1:
    canonical = Path(path).expanduser().resolve()
    cached = context.fingerprint_cache.get(canonical)
    if cached is None:
        cached = fingerprint_file(canonical)
        context.fingerprint_cache[canonical] = cached
    return cached


def _count_nonempty_lines(path: str | Path) -> int:
    count = 0
    with Path(path).expanduser().open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip() != "":
                count += 1
    return count


def _sidecar_created_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _janusx_version() -> str:
    for distribution_name in ("janusx", "JanusX"):
        try:
            return str(importlib_metadata.version(distribution_name))
        except importlib_metadata.PackageNotFoundError:
            continue
    return "unknown"


def serialize_sidecar_block(record: GwasNullModelSidecarV1) -> str:
    """Serialize one v1 record as a deterministic sentinel-delimited block."""

    _validate_record(record)
    try:
        payload = asdict(record)
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SidecarFormatError(f"cannot serialize sidecar: {exc}") from exc
    return f"{SIDECAR_BEGIN_V1}\n{encoded}\n{SIDECAR_END_V1}\n"


def parse_sidecar_blocks(text: str) -> list[GwasNullModelSidecarV1]:
    """Parse all complete v1 blocks from ordinary log text."""

    if not isinstance(text, str):
        raise SidecarFormatError("sidecar log must be text")

    records: list[GwasNullModelSidecarV1] = []
    payload_lines: list[str] = []
    in_block = False
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if not in_block:
            if content == SIDECAR_BEGIN_V1:
                in_block = True
                payload_lines = []
            elif content == SIDECAR_END_V1:
                raise SidecarFormatError("sidecar end sentinel has no begin sentinel")
            continue

        if content == SIDECAR_BEGIN_V1:
            raise SidecarFormatError("sidecar begin sentinel appears before its end")
        if content == SIDECAR_END_V1:
            raw_payload = "".join(payload_lines).strip()
            if not raw_payload:
                raise SidecarFormatError("sidecar block is empty")
            records.append(_parse_record(raw_payload))
            payload_lines = []
            in_block = False
            continue
        payload_lines.append(line)

    if in_block:
        raise SidecarFormatError("sidecar begin sentinel has no end sentinel")
    return records


def find_matching_sidecar(
    result_path: str | Path,
    candidates: Iterable[GwasNullModelSidecarV1],
) -> GwasNullModelSidecarV1:
    """Find one sidecar by canonical path, then moved-file fingerprint."""

    try:
        result_fingerprint = fingerprint_file(result_path)
    except (OSError, ValueError) as exc:
        raise SidecarNotFound(f"result file is unavailable: {result_path}") from exc

    records = list(candidates)
    for record in records:
        _validate_record(record)

    exact = [
        record
        for record in records
        if (
            record.result.canonical_path == result_fingerprint.canonical_path
            and record.result.basename == result_fingerprint.basename
            and record.result.size_bytes == result_fingerprint.size_bytes
            and record.result.sha256.lower() == result_fingerprint.sha256.lower()
        )
    ]
    if exact:
        return _one_match(exact, "canonical result path")

    moved = [
        record
        for record in records
        if (
            record.result.basename == result_fingerprint.basename
            and record.result.size_bytes == result_fingerprint.size_bytes
            and record.result.sha256.lower() == result_fingerprint.sha256.lower()
        )
    ]
    if moved:
        return _one_match(moved, "result basename, size, and SHA-256")
    raise SidecarNotFound(
        f"no sidecar matches result {result_fingerprint.canonical_path}"
    )


def discover_matching_sidecar(
    result_path: str | Path,
) -> GwasNullModelSidecarV1:
    """Discover exactly one matching sidecar from sorted sibling GWAS logs."""

    result = Path(result_path).expanduser()
    try:
        log_paths = sorted(
            result.parent.glob("*.gwas.log"),
            key=lambda path: str(path),
        )
    except OSError as exc:
        raise FineMapSkip(
            f"unable to scan for GWAS sidecar logs beside {result}: {exc}"
        ) from exc

    candidates: list[GwasNullModelSidecarV1] = []
    for log_path in log_paths:
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            candidates.extend(parse_sidecar_blocks(log_text))
        except SidecarFormatError as exc:
            raise FineMapSkip(
                f"sidecar log {log_path} is malformed: {exc}"
            ) from exc
        except OSError as exc:
            raise FineMapSkip(
                f"sidecar log {log_path} is unavailable: {exc}"
            ) from exc

    try:
        return find_matching_sidecar(result, candidates)
    except SidecarAmbiguous as exc:
        raise FineMapSkip(f"matching sidecar is ambiguous: {exc}") from exc
    except SidecarNotFound as exc:
        raise FineMapSkip(f"matching sidecar was not found: {exc}") from exc
    except SidecarFormatError as exc:
        raise FineMapSkip(f"matching sidecar is incompatible: {exc}") from exc


def validate_sidecar_dependencies(
    record: GwasNullModelSidecarV1,
    result_path: str | Path,
    bfile: str | Path,
    *,
    expected_model: str = "fvlmm",
    expected_allele_coding: str = "A1_effect",
) -> None:
    """Validate sidecar metadata and all files needed by PostGWAS."""

    try:
        _validate_record(record)
    except SidecarFormatError as exc:
        raise FineMapSkip(f"sidecar metadata is incompatible: {exc}") from exc

    if record.model != str(expected_model):
        raise FineMapSkip(
            f"sidecar model {record.model!r} does not match the supplied "
            f"mixed-model result {expected_model!r}"
        )
    if record.allele_coding != str(expected_allele_coding):
        raise FineMapSkip(
            "sidecar allele convention "
            f"{record.allele_coding!r} is incompatible with "
            f"{expected_allele_coding!r}"
        )

    _validate_current_fingerprint(
        record.result,
        result_path,
        "GWAS result",
        allow_moved=True,
    )

    requested_prefix = _normalize_genotype_prefix(bfile)

    if len(record.genotype_files) != 3:
        raise FineMapSkip(
            "sidecar genotype fingerprint set must contain a BED/BIM/FAM or PGEN/PVAR/PSAM triplet"
        )
    genotype_suffixes = tuple(
        Path(item.basename).suffix.lower() for item in record.genotype_files
    )
    if genotype_suffixes not in (
        (".bed", ".bim", ".fam"),
        (".pgen", ".pvar", ".psam"),
    ):
        raise FineMapSkip(
            "sidecar genotype fingerprint set must contain a BED/BIM/FAM or PGEN/PVAR/PSAM triplet"
        )
    current_suffixes = _plink_source_suffixes(requested_prefix)
    if genotype_suffixes != current_suffixes:
        raise FineMapSkip("sidecar PLINK source format differs from the current genotype input")
    for suffix, expected in zip(genotype_suffixes, record.genotype_files):
        genotype_path = Path(f"{requested_prefix}{suffix}")
        _validate_current_fingerprint(
            expected,
            genotype_path,
            f"genotype {suffix[1:].upper()}",
            allow_moved=True,
        )

    dependency_roots = _dependency_search_roots(result_path, requested_prefix)
    phenotype_path = _resolve_dependency_path(
        record.phenotype_file,
        dependency_roots,
        "phenotype",
    )
    _validate_current_fingerprint(
        record.phenotype_file,
        phenotype_path,
        "phenotype",
        allow_moved=True,
    )
    if record.covariate_file is not None:
        covariate_path = _resolve_dependency_path(
            record.covariate_file,
            dependency_roots,
            "covariate",
        )
        _validate_current_fingerprint(
            record.covariate_file,
            covariate_path,
            "covariate",
            allow_moved=True,
        )
    kinship_path = _resolve_dependency_path(
        record.kinship_file,
        dependency_roots,
        "GRM",
    )
    _validate_current_fingerprint(
        record.kinship_file,
        kinship_path,
        "GRM",
        allow_moved=True,
    )
    kinship_id_path = _resolve_dependency_path(
        record.kinship_id_file,
        dependency_roots,
        "GRM ID",
    )
    _validate_current_fingerprint(
        record.kinship_id_file,
        kinship_id_path,
        "GRM ID",
        allow_moved=True,
    )
    if record.qmatrix_file is not None:
        qmatrix_path = _resolve_dependency_path(
            record.qmatrix_file,
            dependency_roots,
            "Q/PC matrix",
        )
        _validate_current_fingerprint(
            record.qmatrix_file,
            qmatrix_path,
            "Q/PC matrix",
            allow_moved=True,
        )


def _normalize_genotype_prefix(path: str | Path) -> Path:
    text = str(Path(path).expanduser())
    lower = text.lower()
    for suffix in (".bed", ".bim", ".fam", ".pgen", ".pvar", ".psam"):
        if lower.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return Path(text).resolve()


def _plink_source_suffixes(genotype_prefix: str | Path) -> tuple[str, str, str]:
    prefix = str(genotype_prefix)
    if Path(f"{prefix}.bed").is_file():
        return (".bed", ".bim", ".fam")
    if Path(f"{prefix}.pgen").is_file():
        return (".pgen", ".pvar", ".psam")
    raise FileNotFoundError(f"No PLINK BED or PGEN source found at prefix: {prefix}")


def _dependency_search_roots(
    result_path: str | Path, genotype_prefix: str | Path
) -> tuple[Path, ...]:
    roots = {
        Path(result_path).expanduser().resolve().parent,
        _normalize_genotype_prefix(genotype_prefix).parent,
    }
    return tuple(sorted(roots, key=str))


def _resolve_dependency_path(
    expected: FileFingerprintV1,
    search_roots: Iterable[Path],
    label: str,
) -> Path:
    """Resolve a dependency by exact path, then unique sibling identity."""

    exact_path = Path(expected.canonical_path).expanduser().resolve()
    exact_path_present = exact_path.exists()
    if _fingerprint_matches(expected, exact_path):
        return exact_path

    expected_basename = Path(expected.basename)
    if expected_basename.name != expected.basename:
        raise FineMapSkip(
            f"{label} dependency has an invalid basename: {expected.basename!r}"
        )

    candidates: list[Path] = []
    for root in search_roots:
        candidate = (Path(root) / expected.basename).resolve()
        if candidate == exact_path or candidate in candidates:
            continue
        if _fingerprint_matches(expected, candidate):
            candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        rendered = ", ".join(str(path) for path in candidates)
        raise FineMapSkip(
            f"{label} dependency is ambiguous among matching siblings: {rendered}"
        )
    if exact_path_present:
        raise FineMapSkip(f"{label} fingerprint mismatch: {exact_path}")
    raise FineMapSkip(
        f"{label} dependency matching basename/fingerprint was not found: "
        f"{expected.basename}"
    )


def _fingerprint_matches(expected: FileFingerprintV1, path: str | Path) -> bool:
    try:
        current = fingerprint_file(path)
    except (OSError, ValueError):
        return False
    return (
        current.basename == expected.basename
        and current.size_bytes == expected.size_bytes
        and current.sha256.lower() == expected.sha256.lower()
    )


def _validate_current_fingerprint(
    expected: FileFingerprintV1,
    path: str | Path,
    label: str,
    *,
    allow_moved: bool = False,
) -> None:
    try:
        current = fingerprint_file(path)
    except (OSError, ValueError) as exc:
        raise FineMapSkip(f"{label} is unavailable: {path}") from exc

    same_identity = (
        current.basename == expected.basename
        and current.size_bytes == expected.size_bytes
        and current.sha256.lower() == expected.sha256.lower()
        and (allow_moved or current.canonical_path == expected.canonical_path)
    )
    if not same_identity:
        raise FineMapSkip(f"{label} fingerprint mismatch: {path}")


def _one_match(
    records: list[GwasNullModelSidecarV1], criterion: str
) -> GwasNullModelSidecarV1:
    if len(records) != 1:
        raise SidecarAmbiguous(
            f"{len(records)} sidecars match by {criterion}"
        )
    return records[0]


def _parse_record(raw_payload: str) -> GwasNullModelSidecarV1:
    try:
        payload = json.loads(
            raw_payload,
            object_pairs_hook=_object_pairs_without_duplicates,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except SidecarFormatError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SidecarFormatError(f"invalid sidecar JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SidecarFormatError("sidecar JSON must contain an object")

    try:
        record = GwasNullModelSidecarV1(
            schema=_string_field(payload, "schema"),
            created_at=_string_field(payload, "created_at"),
            janusx_version=_string_field(payload, "janusx_version"),
            result=_fingerprint_field(payload, "result"),
            model=_string_field(payload, "model"),
            trait=_string_field(payload, "trait"),
            z_columns=_string_tuple_field(payload, "z_columns", length=2),
            effective_snp_count=_positive_int_field(payload, "effective_snp_count"),
            genotype_prefix=_string_field(payload, "genotype_prefix"),
            genotype_files=_fingerprint_tuple_field(payload, "genotype_files"),
            phenotype_file=_fingerprint_field(payload, "phenotype_file"),
            phenotype_id_column=_string_field(payload, "phenotype_id_column"),
            phenotype_trait_column=_string_field(payload, "phenotype_trait_column"),
            covariate_file=_optional_fingerprint_field(payload, "covariate_file"),
            covariate_columns=_string_tuple_field(payload, "covariate_columns"),
            kinship_file=_fingerprint_field(payload, "kinship_file"),
            kinship_id_file=_fingerprint_field(payload, "kinship_id_file"),
            kinship_format=_string_field(payload, "kinship_format"),
            kinship_shape=_positive_shape_field(payload, "kinship_shape"),
            sample_count=_positive_int_field(payload, "sample_count"),
            sample_order_sha256=_string_field(payload, "sample_order_sha256"),
            lambda_null=_finite_number_field(payload, "lambda_null"),
            sigma_g2=_optional_finite_number_field(payload, "sigma_g2"),
            sigma_e2=_optional_finite_number_field(payload, "sigma_e2"),
            pve=_finite_number_field(payload, "pve"),
            grm_trace_mean=_finite_number_field(payload, "grm_trace_mean"),
            fixed_effect_columns=_string_tuple_field(payload, "fixed_effect_columns"),
            genotype_filters=_mapping_field(payload, "genotype_filters"),
            allele_coding=_string_field(payload, "allele_coding"),
            qmatrix_file=_optional_fingerprint_field_if_present(
                payload, "qmatrix_file"
            ),
            qmatrix_columns=_optional_string_tuple_field(
                payload, "qmatrix_columns"
            ),
            phenotype_trait_index=_optional_nonnegative_int_field(
                payload, "phenotype_trait_index"
            ),
            phenotype_trait_source_index=_optional_nonnegative_int_field(
                payload, "phenotype_trait_source_index"
            ),
        )
    except SidecarFormatError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise SidecarFormatError(f"invalid v1 sidecar fields: {exc}") from exc
    _validate_record(record)
    return record


def _object_pairs_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SidecarFormatError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise SidecarFormatError(f"non-finite JSON number: {value}")


def _require_field(payload: Mapping[str, Any], name: str) -> Any:
    if name not in payload:
        raise SidecarFormatError(f"missing sidecar field: {name}")
    return payload[name]


def _string_field(payload: Mapping[str, Any], name: str) -> str:
    value = _require_field(payload, name)
    if not isinstance(value, str):
        raise SidecarFormatError(f"sidecar field {name} must be a string")
    return value


def _string_tuple_field(
    payload: Mapping[str, Any], name: str, *, length: int | None = None
) -> tuple[str, ...]:
    value = _require_field(payload, name)
    if not isinstance(value, (list, tuple)):
        raise SidecarFormatError(f"sidecar field {name} must be an array")
    if length is not None and len(value) != length:
        raise SidecarFormatError(
            f"sidecar field {name} must contain exactly {length} values"
        )
    if any(not isinstance(item, str) for item in value):
        raise SidecarFormatError(f"sidecar field {name} must contain strings")
    return tuple(value)


def _optional_string_tuple_field(
    payload: Mapping[str, Any], name: str
) -> tuple[str, ...]:
    if name not in payload:
        return ()
    value = payload[name]
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise SidecarFormatError(f"sidecar field {name} must be an array")
    if any(not isinstance(item, str) for item in value):
        raise SidecarFormatError(f"sidecar field {name} must contain strings")
    return tuple(value)


def _fingerprint_field(
    payload: Mapping[str, Any], name: str
) -> FileFingerprintV1:
    value = _require_field(payload, name)
    if not isinstance(value, dict):
        raise SidecarFormatError(f"sidecar field {name} must be an object")
    try:
        fingerprint = FileFingerprintV1(
            canonical_path=_string_field(value, "canonical_path"),
            basename=_string_field(value, "basename"),
            size_bytes=_nonnegative_int_field(value, "size_bytes"),
            sha256=_string_field(value, "sha256"),
        )
    except SidecarFormatError:
        raise
    _validate_fingerprint(fingerprint, name)
    return fingerprint


def _optional_fingerprint_field(
    payload: Mapping[str, Any], name: str
) -> FileFingerprintV1 | None:
    value = _require_field(payload, name)
    if value is None:
        return None
    return _fingerprint_field(payload, name)


def _optional_fingerprint_field_if_present(
    payload: Mapping[str, Any], name: str
) -> FileFingerprintV1 | None:
    if name not in payload:
        return None
    return _optional_fingerprint_field(payload, name)


def _fingerprint_tuple_field(
    payload: Mapping[str, Any], name: str
) -> tuple[FileFingerprintV1, ...]:
    value = _require_field(payload, name)
    if not isinstance(value, (list, tuple)):
        raise SidecarFormatError(f"sidecar field {name} must be an array")
    if not value:
        raise SidecarFormatError(f"sidecar field {name} must not be empty")
    fingerprints: list[FileFingerprintV1] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise SidecarFormatError(
                f"sidecar field {name}[{index}] must be an object"
            )
        fingerprints.append(_fingerprint_field({name: item}, name))
    return tuple(fingerprints)


def _positive_int_field(payload: Mapping[str, Any], name: str) -> int:
    value = _nonnegative_int_field(payload, name)
    if value <= 0:
        raise SidecarFormatError(f"sidecar field {name} must be positive")
    return value


def _nonnegative_int_field(payload: Mapping[str, Any], name: str) -> int:
    value = _require_field(payload, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SidecarFormatError(f"sidecar field {name} must be an integer")
    if value < 0:
        raise SidecarFormatError(f"sidecar field {name} must not be negative")
    return value


def _optional_nonnegative_int_field(
    payload: Mapping[str, Any], name: str
) -> int | None:
    if name not in payload or payload[name] is None:
        return None
    return _nonnegative_int_field(payload, name)


def _positive_shape_field(
    payload: Mapping[str, Any], name: str
) -> tuple[int, int]:
    value = _require_field(payload, name)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SidecarFormatError(f"sidecar field {name} must have two dimensions")
    dimensions = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise SidecarFormatError(
                f"sidecar field {name} dimensions must be integers"
            )
        if dimension <= 0:
            raise SidecarFormatError(
                f"sidecar field {name} dimensions must be positive"
            )
        dimensions.append(dimension)
    return dimensions[0], dimensions[1]


def _finite_number_field(payload: Mapping[str, Any], name: str) -> float:
    value = _require_field(payload, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SidecarFormatError(f"sidecar field {name} must be numeric")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise SidecarFormatError(f"sidecar field {name} must be finite")
    return value


def _optional_finite_number_field(
    payload: Mapping[str, Any], name: str
) -> float | None:
    value = _require_field(payload, name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SidecarFormatError(f"sidecar field {name} must be numeric or null")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise SidecarFormatError(f"sidecar field {name} must be finite or null")
    return value


def _mapping_field(payload: Mapping[str, Any], name: str) -> dict[str, object]:
    value = _require_field(payload, name)
    if not isinstance(value, dict):
        raise SidecarFormatError(f"sidecar field {name} must be an object")
    return dict(value)


def _validate_fingerprint(
    fingerprint: FileFingerprintV1, field_name: str = "fingerprint"
) -> None:
    if not isinstance(fingerprint, FileFingerprintV1):
        raise SidecarFormatError(f"{field_name} must be a FileFingerprintV1")
    if not isinstance(fingerprint.canonical_path, str) or not fingerprint.canonical_path:
        raise SidecarFormatError(f"{field_name}.canonical_path must be a string")
    if not isinstance(fingerprint.basename, str) or not fingerprint.basename:
        raise SidecarFormatError(f"{field_name}.basename must be a string")
    if (
        isinstance(fingerprint.size_bytes, bool)
        or not isinstance(fingerprint.size_bytes, int)
        or fingerprint.size_bytes < 0
    ):
        raise SidecarFormatError(f"{field_name}.size_bytes must be non-negative")
    if not isinstance(fingerprint.sha256, str) or not _SHA256_RE.fullmatch(
        fingerprint.sha256
    ):
        raise SidecarFormatError(f"{field_name}.sha256 must be a 64-character hex hash")


def _validate_record(record: GwasNullModelSidecarV1) -> None:
    if not isinstance(record, GwasNullModelSidecarV1):
        raise SidecarFormatError("sidecar record must be GwasNullModelSidecarV1")
    if record.schema != SIDECAR_SCHEMA_V1:
        raise SidecarFormatError(f"unsupported sidecar schema: {record.schema!r}")
    for name in (
        "created_at",
        "janusx_version",
        "model",
        "trait",
        "genotype_prefix",
        "phenotype_id_column",
        "phenotype_trait_column",
        "kinship_format",
        "allele_coding",
    ):
        if not isinstance(getattr(record, name), str):
            raise SidecarFormatError(f"sidecar field {name} must be a string")

    _validate_fingerprint(record.result, "result")
    _validate_fingerprint(record.phenotype_file, "phenotype_file")
    _validate_fingerprint(record.kinship_file, "kinship_file")
    _validate_fingerprint(record.kinship_id_file, "kinship_id_file")
    if record.qmatrix_file is not None:
        _validate_fingerprint(record.qmatrix_file, "qmatrix_file")
    if record.covariate_file is not None:
        _validate_fingerprint(record.covariate_file, "covariate_file")
    if not isinstance(record.genotype_files, (tuple, list)) or not record.genotype_files:
        raise SidecarFormatError("sidecar field genotype_files must not be empty")
    for index, fingerprint in enumerate(record.genotype_files):
        _validate_fingerprint(fingerprint, f"genotype_files[{index}]")

    _validate_string_sequence(record.z_columns, "z_columns", length=2)
    _validate_string_sequence(record.covariate_columns, "covariate_columns")
    _validate_string_sequence(record.fixed_effect_columns, "fixed_effect_columns")
    _validate_string_sequence(record.qmatrix_columns, "qmatrix_columns")
    if record.phenotype_trait_column != record.trait:
        raise SidecarFormatError(
            "phenotype_trait_column must match trait"
        )
    if tuple(record.z_columns) != ("beta", "se"):
        raise SidecarFormatError("z columns must be exactly ('beta', 'se')")
    qmatrix_columns = tuple(record.qmatrix_columns)
    if qmatrix_columns and record.qmatrix_file is None:
        raise SidecarFormatError(
            "nonempty qmatrix columns require a qmatrix dependency"
        )
    if record.qmatrix_file is not None and not qmatrix_columns:
        raise SidecarFormatError(
            "qmatrix dependency requires nonempty qmatrix columns"
        )
    if qmatrix_columns and tuple(record.covariate_columns[: len(qmatrix_columns)]) != qmatrix_columns:
        raise SidecarFormatError(
            "qmatrix columns must be the leading covariate columns"
        )
    external_covariate_columns = tuple(record.covariate_columns[len(qmatrix_columns) :])
    if external_covariate_columns and record.covariate_file is None:
        raise SidecarFormatError(
            "nonempty covariate columns require a covariate dependency"
        )
    expected_fixed_effect_columns = ("Intercept",) + tuple(record.covariate_columns)
    if tuple(record.fixed_effect_columns) != expected_fixed_effect_columns:
        raise SidecarFormatError(
            "fixed-effect columns must be Intercept plus covariate columns"
        )
    _validate_positive_int(record.effective_snp_count, "effective_snp_count")
    _validate_positive_int(record.sample_count, "sample_count")
    _validate_positive_shape(record.kinship_shape, "kinship_shape")
    _validate_hash(record.sample_order_sha256, "sample_order_sha256")
    for name in ("lambda_null", "pve", "grm_trace_mean"):
        _validate_finite_number(getattr(record, name), name)
    for name in ("sigma_g2", "sigma_e2"):
        value = getattr(record, name)
        if value is not None:
            _validate_finite_number(value, name)
    if not isinstance(record.genotype_filters, dict):
        raise SidecarFormatError("sidecar field genotype_filters must be an object")
    if record.phenotype_trait_index is not None:
        if (
            isinstance(record.phenotype_trait_index, bool)
            or not isinstance(record.phenotype_trait_index, int)
            or record.phenotype_trait_index < 0
        ):
            raise SidecarFormatError(
                "phenotype_trait_index must be a nonnegative integer or null"
            )
    if record.phenotype_trait_source_index is not None:
        if (
            isinstance(record.phenotype_trait_source_index, bool)
            or not isinstance(record.phenotype_trait_source_index, int)
            or record.phenotype_trait_source_index < 0
        ):
            raise SidecarFormatError(
                "phenotype_trait_source_index must be a nonnegative integer or null"
            )


def _validate_string_sequence(
    value: object, name: str, *, length: int | None = None
) -> None:
    if not isinstance(value, (tuple, list)):
        raise SidecarFormatError(f"sidecar field {name} must be an array")
    if length is not None and len(value) != length:
        raise SidecarFormatError(
            f"sidecar field {name} must contain exactly {length} values"
        )
    if any(not isinstance(item, str) for item in value):
        raise SidecarFormatError(f"sidecar field {name} must contain strings")


def _validate_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SidecarFormatError(f"sidecar field {name} must be a positive integer")


def _validate_positive_shape(value: object, name: str) -> None:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in value
        )
    ):
        raise SidecarFormatError(f"sidecar field {name} must contain positive dimensions")


def _validate_hash(value: object, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise SidecarFormatError(f"sidecar field {name} must be a 64-character hex hash")


def _validate_finite_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SidecarFormatError(f"sidecar field {name} must be numeric")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise SidecarFormatError(f"sidecar field {name} must be finite")
