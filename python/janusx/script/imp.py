"""Stream genotype windows through the native Rust Beagle-style imp module."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import shlex
import socket
import sys
import time
from array import array
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from janusx import janusx as jxrs
from janusx._plink import open_text_auto, pvar_path_for_prefix
from ._common.cli_core import (
    CliArgumentParser,
    cli_help_formatter,
    minimal_help_epilog,
)
from ._common.cli_ranges import (
    format_normalized_range,
    format_normalized_ranges,
    normalize_chromosome_key,
    parse_cli_range,
    parse_cli_ranges,
)
from ._common.config_render import emit_cli_configuration
from ._common.grmio import genotype_cache_prefix
from ._common.log import get_log_path, setup_logging
from ._common.progress import (
    CliStatus,
    _emit_to_file_handlers,
    build_rich_progress,
    format_elapsed,
    log_success,
    rich_progress_available,
    stdout_is_tty,
)
from ._common.threads import detect_effective_threads


_REPORT_RULE = "=" * 72
_TERMINAL_SECTION_WIDTH = 60
_IMP_PROGRESS_BAR_WIDTH = 12
_IMP_PREFLIGHT_RSS_NUMERATOR = 5
_IMP_PREFLIGHT_RSS_DENOMINATOR = 3


def _report_banner(logger: logging.Logger, title: str) -> None:
    logger.info(_REPORT_RULE)
    logger.info(str(title))
    logger.info(_REPORT_RULE)
    logger.info("")


def _report_block(logger: logging.Logger, title: str) -> None:
    logger.info(f"[ {title} ]")


def _report_kv(logger: logging.Logger, key: str, value: object) -> None:
    logger.info(f"  {str(key):<20} : {value}")


def _format_mode(mode: str) -> str:
    return {
        "phase": "Phasing",
        "phasing-imputation": "Phasing&Imputing",
        "reference-imputation": "Reference imputation",
    }.get(str(mode), str(mode).replace("-", " ").title())


def _open_text_input(path: Path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    if str(path).lower().endswith(".zst"):
        return open_text_auto(str(path))
    return path.open(mode="rt", encoding="utf-8", newline="")


class _GeneticMap:
    def __init__(
        self,
        by_chromosome: dict[str, list[tuple[int, float]]] | None = None,
        *,
        use_fallback: bool = False,
    ) -> None:
        self.by_chromosome = by_chromosome or {}
        self.use_fallback = use_fallback

    @classmethod
    def from_plink(cls, path: Path) -> "_GeneticMap":
        by_chromosome: dict[str, list[tuple[int, float]]] = {}
        with path.open(mode="rt", encoding="utf-8") as handle:
            for line_idx, raw in enumerate(handle):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 4:
                    if line_idx == 0 and any(field.lower() == "chr" for field in fields):
                        continue
                    raise ValueError(
                        f"{path}:{line_idx + 1}: expected PLINK map columns CHR SNP CM BP"
                    )
                try:
                    cm = float(fields[2])
                    position = int(fields[3])
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_idx + 1}: invalid genetic-map row"
                    ) from exc
                if position <= 0 or not math.isfinite(cm) or cm < 0.0:
                    raise ValueError(
                        f"{path}:{line_idx + 1}: map positions must be positive "
                        "bp and finite nonnegative cM"
                    )
                points = by_chromosome.setdefault(str(fields[0]), [])
                if points and (position < points[-1][0] or cm < points[-1][1]):
                    raise ValueError(
                        f"{path}:{line_idx + 1}: map positions must be "
                        "nondecreasing within each chromosome"
                    )
                points.append((position, cm))
        if not by_chromosome:
            raise ValueError(f"{path}: genetic map contains no data rows")
        return cls(by_chromosome)

    @classmethod
    def fallback(cls) -> "_GeneticMap":
        return cls(use_fallback=True)

    def cm_at(self, chrom: str, position: int) -> float:
        points = self.by_chromosome.get(chrom)
        if points is None:
            if self.use_fallback:
                return float(position) / 1_000_000.0
            raise ValueError(f"genetic map has no chromosome {chrom}")
        if len(points) == 1:
            return points[0][1] + (position - points[0][0]) / 1_000_000.0
        upper = bisect_left(points, position, key=lambda point: point[0])
        if upper == 0:
            min_cm = points[0][1] + 5.0
            right = bisect_left(points, min_cm, key=lambda point: point[1])
            left, right = points[0], points[min(right, len(points) - 1)]
        elif upper >= len(points):
            last = len(points) - 1
            min_cm = points[last][1] - 5.0
            lower = bisect_left(points, min_cm, key=lambda point: point[1])
            left_index = (
                lower
                if lower < len(points) and points[lower][1] == min_cm
                else lower - 1
            )
            left, right = points[min(left_index, last - 1)], points[last]
        elif points[upper][0] == position:
            return points[upper][1]
        else:
            left, right = points[upper - 1], points[upper]
        if left[0] == right[0]:
            return left[1]
        fraction = (position - left[0]) / (right[0] - left[0])
        return left[1] + fraction * (right[1] - left[1])

    def base_pos_at(self, chrom: str, cm: float) -> int:
        if not math.isfinite(cm):
            raise ValueError("genetic map boundary must be finite")
        if self.use_fallback:
            rounded = cm * 1_000_000.0
        else:
            points = self.by_chromosome.get(chrom)
            if not points:
                raise ValueError(f"genetic map has no positions for chromosome {chrom}")
            if len(points) == 1:
                rounded = points[0][0] + (cm - points[0][1]) * 1_000_000.0
            else:
                last = len(points) - 1
                upper = bisect_left(points, cm, key=lambda point: point[1])
                if upper < len(points) and points[upper][1] == cm:
                    return points[upper][0]
                if cm < points[0][1]:
                    threshold = points[0][1] + 5.0
                    upper_boundary = bisect_left(
                        points, threshold, key=lambda point: point[1]
                    )
                    left, right = 0, min(upper_boundary, last)
                elif cm > points[last][1]:
                    threshold = points[last][1] - 5.0
                    upper_boundary = bisect_left(
                        points, threshold, key=lambda point: point[1]
                    )
                    left = min(max(upper_boundary - 1, 0), last - 1)
                    right = last
                else:
                    left = min(max(upper - 1, 0), last - 1)
                    right = min(upper, last)
                left_bp, left_cm = points[left]
                right_bp, right_cm = points[right]
                if left_cm == right_cm:
                    rounded = float(left_bp)
                else:
                    rounded = left_bp + (
                        (cm - left_cm) / (right_cm - left_cm)
                    ) * (right_bp - left_bp)
        if not math.isfinite(rounded) or rounded < 0.0 or rounded > 2**64 - 1:
            raise ValueError(f"genetic position {cm} cM has no valid base position")
        return math.floor(rounded + 0.5)


def _initial_pbwt_max_chunk_markers(
    cm_positions: array,
    threads: int,
) -> int:
    marker_count = len(cm_positions)
    if marker_count == 0:
        return 0
    logical_chunks = max(int(threads), 1)
    total_cm = float(cm_positions[-1] - cm_positions[0])
    advance_cm = max(total_cm / logical_chunks, 2.0)
    maximum = 0
    start = 0
    end = bisect_right(cm_positions, cm_positions[0] + advance_cm)
    while end < marker_count:
        maximum = max(maximum, end - start)
        start = bisect_left(cm_positions, cm_positions[end] - 0.5)
        next_end = bisect_right(cm_positions, cm_positions[end] + advance_cm)
        if next_end <= end:
            raise ValueError("initial PBWT window partition did not advance")
        end = next_end
    return max(maximum, marker_count - start)


def _build_region_intervals(
    region,
) -> dict[str, list[tuple[int | None, int | None]]] | None:
    if not region:
        return None
    regions = region if isinstance(region, (list, tuple)) else (region,)
    grouped: dict[str, list[tuple[int | None, int | None]]] = {}
    for item in regions:
        grouped.setdefault(item.chrom_key, []).append(
            (
                None if item.start is None else int(item.start),
                None if item.end is None else int(item.end),
            )
        )
    merged_by_chrom: dict[str, list[tuple[int | None, int | None]]] = {}
    for chrom_key, intervals in grouped.items():
        if any(start is None and end is None for start, end in intervals):
            merged_by_chrom[chrom_key] = [(None, None)]
            continue
        concrete = sorted(
            (
                0 if start is None else start,
                sys.maxsize if end is None else end,
            )
            for start, end in intervals
        )
        merged: list[tuple[int | None, int | None]] = []
        for start, end in concrete:
            if merged and merged[-1][1] is not None and start <= merged[-1][1] + 1:
                prev_start, prev_end = merged[-1]
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        merged_by_chrom[chrom_key] = merged
    return merged_by_chrom


def _region_interval_index(
    intervals_by_chrom: dict[str, list[tuple[int | None, int | None]]] | None,
    chrom: str,
    position: int,
) -> int | None:
    if intervals_by_chrom is None:
        return None
    intervals = intervals_by_chrom.get(normalize_chromosome_key(chrom))
    if not intervals:
        return None
    for idx, (start, end) in enumerate(intervals):
        if (start is None or position >= start) and (end is None or position <= end):
            return idx
    return None


class _MetadataWindowCursor:
    def __init__(
        self,
        records: Iterator[tuple[str, int]],
        map_data: _GeneticMap,
        window_cm: float,
        overlap_cm: float,
        window_markers: int,
        threads: int,
        region=None,
    ) -> None:
        self.records = records
        self.map = map_data
        self.window_cm = window_cm
        self.overlap_cm = overlap_cm
        self.window_markers = window_markers
        self.threads = threads
        self.region_intervals = _build_region_intervals(region)
        self.pending: tuple[str, int, float] | None = None
        self.carry_positions = array("Q")
        self.carry_cm = array("d")
        self.carry_chromosome: str | None = None
        self.last_chromosome: str | None = None
        self.last_position = 0
        self.last_cm = -math.inf
        self.exhausted = False
        self.seen_chromosomes: set[str] = set()

    def next_window(self) -> tuple[int, int, float] | None:
        if self.exhausted and not self.carry_positions and self.pending is None:
            return None
        window_positions = self.carry_positions
        window_cm = self.carry_cm
        carry_chromosome = self.carry_chromosome
        self.carry_positions = array("Q")
        self.carry_cm = array("d")
        self.carry_chromosome = None
        retained_overlap_rows = len(window_positions)
        if retained_overlap_rows and self.pending is None and not self.exhausted:
            self.pending = self._read_next()
            self.exhausted = self.pending is None

        if self.pending is not None:
            pending_chrom, position, cm = self.pending
            self.pending = None
            if not window_positions:
                chrom = pending_chrom
                window_positions.append(position)
                window_cm.append(cm)
            else:
                chrom = carry_chromosome or ""
                self.pending = (pending_chrom, position, cm)
        elif window_positions:
            chrom = carry_chromosome or ""
        else:
            first = self._read_next()
            if first is None:
                self.exhausted = True
                return None
            chrom, position, cm = first
            window_positions.append(position)
            window_cm.append(cm)
        if not window_positions:
            return None

        active_interval = _region_interval_index(
            self.region_intervals, chrom, int(window_positions[0])
        )
        start_cm = window_cm[0]
        if retained_overlap_rows:
            pending_same_interval = (
                self.pending is not None
                and self.pending[0] == chrom
                and _region_interval_index(
                    self.region_intervals, self.pending[0], int(self.pending[1])
                )
                == active_interval
            )
            end_cm = (
                self.pending[2] + self.window_cm - self.overlap_cm
                if pending_same_interval
                else start_cm + self.window_cm
            )
        else:
            end_cm = start_cm + self.window_cm
        end_position = self.map.base_pos_at(chrom, end_cm)
        hit_chromosome_boundary = False
        hit_cm_boundary = False
        while len(window_positions) < self.window_markers and not self.exhausted:
            record = self._read_next()
            if record is None:
                self.exhausted = True
                break
            next_chrom, next_position, next_cm = record
            if next_chrom != chrom or (
                self.region_intervals is not None
                and _region_interval_index(
                    self.region_intervals, next_chrom, int(next_position)
                )
                != active_interval
            ):
                self.pending = record
                hit_chromosome_boundary = True
                break
            if next_position >= end_position:
                self.pending = record
                hit_cm_boundary = True
                break
            window_positions.append(next_position)
            window_cm.append(next_cm)

        marker_cap_reached = len(window_positions) >= self.window_markers
        no_new_rows_after_overlap = len(window_positions) == retained_overlap_rows
        terminal = self.exhausted or hit_chromosome_boundary or (
            hit_cm_boundary and no_new_rows_after_overlap
        )
        if terminal:
            carry_start = len(window_positions)
            outgoing_phase_carry_len = 0
        else:
            overlap_start_cm = max(
                window_cm[-1] - self.overlap_cm,
                window_cm[0],
            )
            overlap_start_position = self.map.base_pos_at(chrom, overlap_start_cm)
            low = max(
                len(window_positions) - max(self.window_markers >> 2, 1),
                0,
            )
            insertion = bisect_left(
                window_positions,
                overlap_start_position,
                lo=low,
            )
            candidate = (
                insertion
                if insertion < len(window_positions)
                and window_positions[insertion] == overlap_start_position
                else max(insertion - 1, 0)
            )
            candidate_position = window_positions[candidate]
            overlap_start = bisect_left(window_positions, candidate_position)
            splice = (len(window_positions) + overlap_start) >> 1
            carry_start = overlap_start
            outgoing_phase_carry_len = splice - overlap_start

        retained_positions = window_positions[carry_start:]
        retained_cm = window_cm[carry_start:]
        if marker_cap_reached and len(retained_positions) >= self.window_markers:
            removed = len(retained_positions) - (self.window_markers - 1)
            retained_positions = retained_positions[removed:]
            retained_cm = retained_cm[removed:]
            outgoing_phase_carry_len = max(
                outgoing_phase_carry_len - removed,
                0,
            )
        self.carry_positions = retained_positions
        self.carry_cm = retained_cm
        self.carry_chromosome = chrom if retained_positions else None
        return (
            len(window_positions),
            _initial_pbwt_max_chunk_markers(window_cm, self.threads),
            float(window_cm[-1] - window_cm[0]),
        )

    def _read_next(self) -> tuple[str, int, float] | None:
        if self.pending is not None:
            record = self.pending
            self.pending = None
            return record
        try:
            chrom, position = next(self.records)
        except StopIteration:
            return None
        if self.last_chromosome is not None:
            if chrom == self.last_chromosome:
                if position < self.last_position:
                    raise ValueError(
                        f"input positions are not sorted: {chrom}:{position} "
                        f"follows {self.last_chromosome}:{self.last_position}"
                    )
            else:
                self.seen_chromosomes.add(self.last_chromosome)
                if chrom in self.seen_chromosomes:
                    raise ValueError(
                        f"input chromosome {chrom} reappears after another chromosome"
                    )
                self.last_cm = -math.inf
        cm = self.map.cm_at(chrom, position)
        if cm + sys.float_info.epsilon < self.last_cm:
            raise ValueError(
                f"genetic positions decrease at {chrom}:{position} "
                f"({cm} cM after {self.last_cm} cM)"
            )
        self.last_chromosome = chrom
        self.last_position = position
        self.last_cm = cm
        return chrom, position, cm


@dataclass(frozen=True)
class _WindowInspection:
    samples: int
    markers: int
    windows: int
    max_window_markers: int
    max_initial_pbwt_chunk_markers: int = 0
    max_window_cm_span: float = 0.0
    estimated_peak_rss_bytes: int = 0


def _marker_in_region(
    chrom: object,
    position: object,
    region,
) -> bool:
    if not region:
        return True
    regions = region if isinstance(region, (list, tuple)) else (region,)
    chrom_key = normalize_chromosome_key(chrom)
    parsed_position = int(position)
    for item in regions:
        if chrom_key != item.chrom_key:
            continue
        if (item.start is None or parsed_position >= int(item.start)) and (
            item.end is None or parsed_position <= int(item.end)
        ):
            return True
    return False


def _plain_marker_records(
    path: Path,
    *,
    chrom_index: int,
    position_index: int,
    region,
) -> Iterator[tuple[str, int]]:
    with _open_text_input(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) <= max(chrom_index, position_index):
                raise ValueError(f"{path}: malformed marker row")
            chrom = str(fields[chrom_index])
            position = int(fields[position_index])
            if _marker_in_region(chrom, position, region):
                yield chrom, position


def _count_sample_rows(path: Path) -> int:
    count = 0
    with _open_text_input(path) as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


def _window_layout(
    records: Iterable[tuple[str, int]],
    *,
    map_data: _GeneticMap,
    window_cm: float,
    overlap_cm: float,
    window_markers: int,
    threads: int,
    region=None,
) -> tuple[int, int, int, int, float]:
    marker_count = 0

    def counted_records() -> Iterator[tuple[str, int]]:
        nonlocal marker_count
        for record in records:
            marker_count += 1
            yield record

    cursor = _MetadataWindowCursor(
        counted_records(),
        map_data,
        window_cm,
        overlap_cm,
        window_markers,
        threads,
        region=region,
    )
    windows = 0
    max_window_markers = 0
    max_initial_pbwt_chunk_markers = 0
    max_window_cm_span = 0.0
    while True:
        layout = cursor.next_window()
        if layout is None:
            break
        observed, initial_pbwt_markers, window_cm_span = layout
        windows += 1
        max_window_markers = max(max_window_markers, observed)
        max_initial_pbwt_chunk_markers = max(
            max_initial_pbwt_chunk_markers,
            initial_pbwt_markers,
        )
        max_window_cm_span = max(max_window_cm_span, window_cm_span)
    return (
        marker_count,
        windows,
        max_window_markers,
        max_initial_pbwt_chunk_markers,
        max_window_cm_span,
    )


def _inspect_target_dimensions(
    target_kind: str,
    target: Path,
    region,
    *,
    map_path: Path | None,
    window_cm: float,
    overlap_cm: float,
    window_markers: int,
    threads: int,
) -> _WindowInspection:
    map_data = (
        _GeneticMap.fallback()
        if map_path is None
        else _GeneticMap.from_plink(map_path)
    )
    if target_kind == "bfile":
        samples = _count_sample_rows(Path(f"{target}.fam"))
        (
            markers,
            windows,
            max_window_markers,
            max_initial_pbwt_chunk_markers,
            max_window_cm_span,
        ) = _window_layout(
            _plain_marker_records(
                Path(f"{target}.bim"),
                chrom_index=0,
                position_index=3,
                region=region,
            ),
            map_data=map_data,
            window_cm=window_cm,
            overlap_cm=overlap_cm,
            window_markers=window_markers,
            threads=threads,
            region=region,
        )
        return _WindowInspection(
            samples,
            markers,
            windows,
            max_window_markers,
            max_initial_pbwt_chunk_markers,
            max_window_cm_span,
        )
    if target_kind == "pfile":
        samples = _count_sample_rows(Path(f"{target}.psam"))
        (
            markers,
            windows,
            max_window_markers,
            max_initial_pbwt_chunk_markers,
            max_window_cm_span,
        ) = _window_layout(
            _plain_marker_records(
                Path(pvar_path_for_prefix(str(target))),
                chrom_index=0,
                position_index=1,
                region=region,
            ),
            map_data=map_data,
            window_cm=window_cm,
            overlap_cm=overlap_cm,
            window_markers=window_markers,
            threads=threads,
            region=region,
        )
        return _WindowInspection(
            samples,
            markers,
            windows,
            max_window_markers,
            max_initial_pbwt_chunk_markers,
            max_window_cm_span,
        )
    if target_kind == "vcf":
        samples = 0

        def records() -> Iterator[tuple[str, int]]:
            nonlocal samples
            with _open_text_input(target) as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        if line.startswith("#CHROM"):
                            samples = max(0, len(line.split("\t")) - 9)
                        continue
                    fields = line.split("\t")
                    if len(fields) < 2:
                        raise ValueError(f"{target}: malformed VCF marker row")
                    chrom = str(fields[0])
                    position = int(fields[1])
                    if _marker_in_region(chrom, position, region):
                        yield chrom, position

        (
            markers,
            windows,
            max_window_markers,
            max_initial_pbwt_chunk_markers,
            max_window_cm_span,
        ) = _window_layout(
            records(),
            map_data=map_data,
            window_cm=window_cm,
            overlap_cm=overlap_cm,
            window_markers=window_markers,
            threads=threads,
            region=region,
        )
        return _WindowInspection(
            samples,
            markers,
            windows,
            max_window_markers,
            max_initial_pbwt_chunk_markers,
            max_window_cm_span,
        )
    if target_kind == "hmp":
        samples = 0
        header_seen = False

        def records() -> Iterator[tuple[str, int]]:
            nonlocal samples, header_seen
            with _open_text_input(target) as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    fields = line.split()
                    if not header_seen:
                        samples = max(0, len(fields) - 11)
                        header_seen = True
                        continue
                    if len(fields) < 4:
                        raise ValueError(f"{target}: malformed HapMap marker row")
                    chrom = str(fields[2])
                    position = int(fields[3])
                    if _marker_in_region(chrom, position, region):
                        yield chrom, position

        (
            markers,
            windows,
            max_window_markers,
            max_initial_pbwt_chunk_markers,
            max_window_cm_span,
        ) = _window_layout(
            records(),
            map_data=map_data,
            window_cm=window_cm,
            overlap_cm=overlap_cm,
            window_markers=window_markers,
            threads=threads,
            region=region,
        )
        return _WindowInspection(
            samples,
            markers,
            windows,
            max_window_markers,
            max_initial_pbwt_chunk_markers,
            max_window_cm_span,
        )
    raise ValueError(f"unsupported target kind for progress inspection: {target_kind}")


def _inspect_reference_dimensions(
    reference: Path | None,
    region,
    *,
    map_path: Path | None,
    window_cm: float,
    overlap_cm: float,
    window_markers: int,
    threads: int,
) -> _WindowInspection | None:
    if reference is None:
        return None
    lower = str(reference).lower()
    reference_kind = (
        "vcf"
        if lower.endswith(".vcf") or lower.endswith(".vcf.gz")
        else "pfile"
    )
    return _inspect_target_dimensions(
        reference_kind,
        reference,
        region,
        map_path=map_path,
        window_cm=window_cm,
        overlap_cm=overlap_cm,
        window_markers=window_markers,
        threads=threads,
    )


def _process_rss_bytes() -> int:
    current_rss = 0
    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            resident_pages = int(statm.read_text().split()[1])
            current_rss = resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            current_rss = 0
    try:
        import resource

        maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform.startswith("linux"):
            maximum_rss *= 1024
    except (ImportError, OSError, ValueError):
        maximum_rss = 0
    return max(current_rss, maximum_rss)


def _emit_memory_preflight(
    logger: logging.Logger,
    inspection: _WindowInspection,
) -> None:
    logger.info(f"windows                       {inspection.windows:,}")
    logger.info(
        "estimated peak RSS            "
        f"{inspection.estimated_peak_rss_bytes / (1024.0**3):.1f} GiB"
    )


def _set_preflight_estimate(
    inspection: _WindowInspection,
    *,
    options: dict[str, object],
    reference_inspection: _WindowInspection | None,
) -> _WindowInspection:
    reference_samples = (
        0 if reference_inspection is None else reference_inspection.samples
    )
    reference_markers = (
        0 if reference_inspection is None else reference_inspection.max_window_markers
    )
    union_markers = min(
        int(options["window_markers"]),
        max(inspection.max_window_markers, reference_markers),
    )
    window_cm_span = max(
        inspection.max_window_cm_span,
        0.0
        if reference_inspection is None
        else reference_inspection.max_window_cm_span,
    )
    window_bytes = int(
        jxrs.imp_estimate_window_bytes(
            json.dumps(options),
            inspection.samples,
            reference_samples,
            max(inspection.max_window_markers, 1),
            max(reference_markers, 0),
            max(union_markers, 1),
            max(inspection.max_initial_pbwt_chunk_markers, 1),
            max(float(window_cm_span), 0.0),
        )
    )
    # The active-window model does not include resident allocator, reader, and
    # streaming-pipeline retention. Calibrate only that dynamic component
    # against observed HPC peaks; the process baseline is already included.
    window_bytes = (
        window_bytes * _IMP_PREFLIGHT_RSS_NUMERATOR
        + _IMP_PREFLIGHT_RSS_DENOMINATOR
        - 1
    ) // _IMP_PREFLIGHT_RSS_DENOMINATOR
    return _WindowInspection(
        samples=inspection.samples,
        markers=inspection.markers,
        windows=inspection.windows,
        max_window_markers=inspection.max_window_markers,
        max_initial_pbwt_chunk_markers=(
            inspection.max_initial_pbwt_chunk_markers
        ),
        max_window_cm_span=inspection.max_window_cm_span,
        estimated_peak_rss_bytes=max(_process_rss_bytes(), 0) + window_bytes,
    )

def _janusx_version() -> str:
    try:
        return version("janusx")
    except PackageNotFoundError:
        return "unknown"


def _invocation_command() -> str:
    return shlex.join([str(value) for value in sys.argv])


def _logger_handlers(
    logger: logging.Logger,
    name: str,
    *,
    file_handlers: bool,
) -> logging.Logger:
    child = logging.getLogger(name)
    child.handlers.clear()
    child.setLevel(logging.INFO)
    child.propagate = False
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler) == bool(file_handlers):
            child.addHandler(handler)
    return child


def _output_destination(prefix: Path, output_format: str) -> Path:
    prefix = Path(prefix).expanduser()
    lower = str(prefix).lower()
    if output_format == "vcf":
        if lower.endswith(".vcf.gz") or lower.endswith(".vcf"):
            return prefix
        return Path(f"{prefix}.vcf.gz")
    for suffix in (".pgen", ".pvar.zst", ".pvar", ".psam"):
        if lower.endswith(suffix):
            return Path(str(prefix)[: -len(suffix)])
    return prefix


def _output_prefix(value: Path) -> Path:
    """Return the user-facing run prefix without a genotype output suffix."""
    path = Path(value).expanduser()
    lower = str(path).lower()
    for suffix in (".vcf.gz", ".vcf", ".pgen", ".pvar.zst", ".pvar", ".psam"):
        if lower.endswith(suffix):
            return Path(str(path)[: -len(suffix)])
    return path


def _resolve_reference_arg(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    lower = str(path).lower()
    if lower.endswith(".vcf") or lower.endswith(".vcf.gz"):
        if not path.is_file():
            raise ValueError(f"reference VCF not found: {path}")
        return path
    for suffix in (".pgen", ".pvar.zst", ".pvar", ".psam"):
        if lower.endswith(suffix):
            path = Path(str(path)[: -len(suffix)])
            break
    members = [
        Path(f"{path}.pgen"),
        Path(pvar_path_for_prefix(str(path))),
        Path(f"{path}.psam"),
    ]
    if not all(member.is_file() for member in members):
        raise ValueError(
            "reference must be a complete pfile prefix with .pgen/.pvar/.psam "
            f"or a VCF file: {value}"
        )
    return path


def _hmp_cache_prefix(hmp: Path, cache_dir: str | Path | None = None) -> Path:
    hmp = Path(hmp).expanduser()
    stat = hmp.stat()
    resolved = str(hmp.resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    base = Path(
        genotype_cache_prefix(
            str(hmp),
            cache_dir=None if cache_dir is None else str(cache_dir),
        )
    )
    version = f"v{int(stat.st_size)}-{int(stat.st_mtime_ns)}"
    return Path(f"{base}.{digest}.{version}.imp")


def _strip_input_suffix(path: Path, suffixes: tuple[str, ...]) -> Path:
    lower = str(path).lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            return Path(str(path)[: -len(suffix)])
    return path


def _selected_target(args: argparse.Namespace) -> tuple[str, Path]:
    for kind in ("bfile", "pfile", "vcf", "hmp"):
        value = getattr(args, kind)
        if value is not None:
            path = Path(value).expanduser()
            if kind == "bfile":
                return kind, _strip_input_suffix(path, (".bed", ".bim", ".fam"))
            if kind == "pfile":
                return kind, _strip_input_suffix(
                    path, (".pgen", ".pvar.zst", ".pvar", ".psam")
                )
            if not path.is_file():
                raise ValueError(f"{kind} input not found: {path}")
            lower = str(path).lower()
            if kind == "vcf" and not (
                lower.endswith(".vcf") or lower.endswith(".vcf.gz")
            ):
                raise ValueError(f"-vcf expects a .vcf or .vcf.gz filename: {path}")
            if kind == "hmp" and not (
                lower.endswith(".hmp") or lower.endswith(".hmp.gz")
            ):
                raise ValueError(f"-hmp expects a .hmp or .hmp.gz filename: {path}")
            return kind, path
    raise ValueError("one target input is required")


def _pfile_cache_complete(prefix: Path) -> bool:
    return (
        Path(f"{prefix}.pgen").is_file()
        and Path(pvar_path_for_prefix(str(prefix))).is_file()
        and Path(f"{prefix}.psam").is_file()
    )




def _diagnostic_site(value: str | None) -> dict[str, object] | None:
    if value is None:
        return None
    try:
        chrom, position, sample = value.split(":", 2)
        parsed_position = int(position)
    except (ValueError, TypeError) as exc:
        raise ValueError("--trace-site expects CHROM:POS:SAMPLE") from exc
    if not chrom or not sample or parsed_position <= 0:
        raise ValueError("--trace-site expects nonempty CHROM/SAMPLE and a positive POS")
    return {"chrom": chrom, "position": parsed_position, "sample": sample}


def _validate_modes(
    *,
    reference: str | Path | None,
    only_phase: bool,
    ap: bool,
    gp: bool,
    output_format: str,
) -> str:
    if only_phase:
        if ap or gp:
            raise ValueError("--ap and --gp require reference imputation and cannot be used with -only-phase")
        return "phase"
    if (ap or gp) and reference is None:
        raise ValueError("--ap and --gp require -ref")
    if (ap or gp) and str(output_format).lower() != "vcf":
        raise ValueError("--ap and --gp require VCF output")
    if reference is not None:
        return "reference-imputation"
    return "phasing-imputation"


class _ImpProgressReporter:
    """Rich progress on TTY and state-change-only progress records elsewhere."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        total: int | None = None,
        task_label: str = "IMP",
        imputation_mode: bool = False,
    ) -> None:
        self.logger = logger
        self._total = None if total is None else int(max(0, total))
        self._task_label = str(task_label).strip() or "IMP"
        self._imputation_mode = bool(imputation_mode)
        self._started = time.monotonic()
        self._progress = None
        self._task_id = None
        self._observed_phasing = False
        self._last_progress_signature = None
        if stdout_is_tty() and rich_progress_available():
            self._progress = build_rich_progress(
                description_template="[green]{task.description:<16}",
                show_spinner=True,
                show_bar=True,
                show_percentage=self._total is not None,
                show_elapsed=True,
                show_remaining=self._total is not None,
                field_templates=[],
                bar_width=_IMP_PROGRESS_BAR_WIDTH,
                transient=True,
            )

    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._started)

    def _start_progress(self) -> None:
        if self._progress is None or self._task_id is not None:
            return
        self._progress.start()
        self._task_id = self._progress.add_task(
            self._task_label,
            total=self._total,
        )

    def _log_progress_change(
        self,
        *,
        stage: str,
        stage_label: str,
        chrom: str,
        window_index: int,
        markers: int,
        variants_written: int,
        samples: int,
    ) -> None:
        if stage not in {
            "prepare",
            "direct",
            "phase",
            "impute",
            "window_done",
            "complete",
        }:
            return
        completed = (
            self._total
            if stage == "complete" and self._total is not None
            else min(self._total, variants_written)
            if self._total is not None
            else variants_written
        )
        signature = (
            stage_label,
            chrom,
            window_index,
            markers,
            completed,
            samples,
        )
        if signature == self._last_progress_signature:
            return
        self._last_progress_signature = signature
        parts = [f"stage={stage_label}"]
        if chrom:
            parts.append(f"chromosome={chrom}")
        if stage not in {"prepare", "complete"}:
            parts.append(f"window={window_index + 1:,}")
        if markers and stage != "complete":
            parts.append(f"window_markers={markers:,}")
        if self._total is not None:
            percent = 100.0 * float(completed) / float(self._total)
            parts.append(
                f"completed={int(completed):,}/{self._total:,} "
                f"({percent:.1f}%)"
            )
        elif completed:
            parts.append(f"completed={int(completed):,}")
        if samples and stage in {"prepare", "complete"}:
            parts.append(f"samples={samples:,}")
        message = "  Progress             : " + " | ".join(parts)
        if self._progress is not None:
            _emit_to_file_handlers(self.logger, logging.INFO, message)
        else:
            self.logger.info(message)

    def __call__(
        self,
        stage: str,
        chrom: str,
        window_index: int,
        markers: int,
        variants_written: int,
        samples: int,
    ) -> None:
        stage = str(stage)
        chrom = str(chrom)
        window_index = int(max(0, window_index))
        markers = int(max(0, markers))
        variants_written = int(max(0, variants_written))
        samples = int(max(0, samples))
        if stage in {"cache", "cache_done"}:
            stage_label = "Caching"
        elif stage == "complete":
            stage_label = "Finished"
        elif stage == "phase":
            self._observed_phasing = True
            stage_label = (
                "Phasing&Imputing"
                if self._imputation_mode
                else self._task_label
            )
        elif self._imputation_mode and stage in {
            "direct",
            "impute",
            "write",
            "window_done",
        }:
            stage_label = (
                "Phasing&Imputing"
                if self._observed_phasing
                else "Imputing"
            )
        else:
            stage_label = self._task_label
        self._start_progress()
        if self._progress is not None and self._task_id is not None:
            update_fields = {
                "description": stage_label,
            }
            if self._total is not None:
                cache_stage = stage in {"prepare", "cache", "cache_done"}
                update_fields["completed"] = (
                    self._total
                    if stage == "complete"
                    else (
                        0
                        if cache_stage
                        else min(self._total, variants_written)
                    )
                )
            self._progress.update(self._task_id, **update_fields)
        self._log_progress_change(
            stage=stage,
            stage_label=stage_label,
            chrom=chrom,
            window_index=window_index,
            markers=markers,
            variants_written=variants_written,
            samples=samples,
        )

    def close(self) -> None:
        if self._progress is not None:
            if self._task_id is not None:
                self._progress.stop()
            self._progress = None
            self._task_id = None


def build_parser(argv: Sequence[str] | None = None) -> CliArgumentParser:
    argv_tokens = list(sys.argv[1:] if argv is None else argv)
    show_dev_help = "-dev" in argv_tokens or "--dev" in argv_tokens

    def help_text(text: str, *, advanced: bool = False):
        return (str(text), argparse.SUPPRESS)[bool(advanced and not show_dev_help)]

    parser = CliArgumentParser(
        prog="jx imp",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx imp -vcf target.vcf.gz -o phased -fmt vcf",
                "jx imp -pfile target -ref reference -fmt pfile -o imputed",
                "jx imp -hmp target.hmp.gz -fmt pfile -o phased",
                "jx imp -bfile target -ref reference.vcf.gz -o imputed",
                "jx imp -h -dev",
            ]
        ),
    )
    parser.add_argument(
        "-dev",
        "--dev",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    genotype = parser.add_argument_group(
        "Genotype Arguments (Required: Select exactly one)"
    )
    target = genotype.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "-bfile",
        "--bfile",
        help="Input PLINK BED prefix (.bed/.bim/.fam).",
    )
    target.add_argument(
        "-pfile",
        "--pfile",
        help="Input PLINK 2 PGEN prefix (.pgen/.pvar/.psam).",
    )
    target.add_argument(
        "-vcf",
        "--vcf",
        help="Input genotype file in VCF format (.vcf or .vcf.gz).",
    )
    target.add_argument(
        "-hmp",
        "--hmp",
        help=(
            "Input genotype file in HMP format (.hmp or .hmp.gz). "
            "HMP is converted to a reusable PFILE cache before processing."
        ),
    )

    reference = parser.add_argument_group("Reference Arguments")
    reference.add_argument(
        "-ref",
        "--reference",
        help=(
            "Fully phased reference as a VCF filename or PGEN prefix. "
            "The format is detected automatically."
        ),
    )
    reference.add_argument(
        "-map",
        "--map",
        help="PLINK genetic map with CHR, SNP, CM, and BP columns.",
    )
    reference.add_argument(
        "-bimrange",
        "--bimrange",
        type=str,
        action="append",
        default=None,
        metavar="CHR[:START:END]",
        help=(
            "Restrict output to one or more chromosomes or inclusive physical "
            "intervals. Repeat the flag for multiple entries (e.g. "
            "-bimrange 1 -bimrange 2 -bimrange 3:0:100); comma-separated items "
            "are also accepted. Use bp or Mb suffixes when units would be "
            "ambiguous."
        ),
    )

    model = parser.add_argument_group("Phasing / Imputation Arguments")
    model.add_argument(
        "-w",
        "--window",
        type=float,
        default=40.0,
        help="Window length in cM (default: %(default)s).",
    )
    model.add_argument(
        "-overlap",
        "--overlap",
        type=float,
        default=2.0,
        help="Window overlap in cM (default: %(default)s).",
    )
    model.add_argument(
        "-window-markers",
        type=int,
        default=4_000_000,
        help="Maximum target markers per window (default: %(default)s).",
    )
    model.add_argument(
        "-phase-states",
        type=int,
        default=280,
        help=help_text(
            "Number of phasing HMM states (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "-imp-states",
        type=int,
        default=1_600,
        help=help_text(
            (
                "Maximum imputation states. Adaptive sizing uses fewer states when "
                "the reference panel is smaller (default: %(default)s)."
            ),
            advanced=True,
        ),
    )
    model.add_argument(
        "--no-adaptive-states",
        dest="adaptive_states",
        action="store_false",
        default=True,
        help=help_text("Use the fixed -imp-states value.", advanced=True),
    )
    model.add_argument(
        "-burnin",
        type=int,
        default=3,
        help=help_text(
            "Number of EM burn-in rounds per window (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "-iterations",
        type=int,
        default=12,
        help=help_text(
            "Number of EM parameter-update rounds per window (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "-initial-lr",
        type=float,
        default=100_000.0,
        help=help_text(
            "Initial phasing likelihood-ratio threshold (default: %(default)s).",
            advanced=True,
        ),
    )
    model.add_argument(
        "-ne",
        type=int,
        default=100_000,
        help=help_text(
            (
                "Effective population size used by phasing transitions "
                "(default: %(default)s)."
            ),
        ),
    )
    model.add_argument(
        "-err",
        type=float,
        default=None,
        help=(
            "Allele error-rate override (default: auto, "
            "1/[2+2H(0.5+ln H)], where H is the total target+reference "
            "haplotype count)."
        ),
    )
    model.add_argument(
        "--no-em",
        dest="em",
        action="store_false",
        default=True,
        help=help_text("Disable EM parameter updates.", advanced=True),
    )
    model.add_argument(
        "-only-phase",
        "--only-phase",
        dest="only_phase",
        action="store_true",
        help="Phase target genotypes without reference-marker imputation; -ref may still assist phasing.",
    )
    parser.set_defaults(only_phase=False)
    model.add_argument(
        "-imp-segment",
        type=float,
        default=6.0,
        help=help_text(
            (
                "Genetic length in cM of each local PBWT donor segment "
                "(default: %(default)s)."
            ),
            advanced=True,
        ),
    )
    model.add_argument(
        "-imp-step",
        type=float,
        default=0.1,
        help=help_text(
            (
                "Beagle compatibility value (default: %(default)s). "
                "Custom matching steps are not implemented."
            ),
            advanced=True,
        ),
    )
    model.add_argument(
        "-imp-nsteps",
        type=int,
        default=7,
        help=help_text(
            (
                "Beagle compatibility value (default: %(default)s). "
                "Custom matching steps are not implemented."
            ),
            advanced=True,
        ),
    )
    model.add_argument(
        "-cluster",
        type=float,
        default=0.005,
        help=help_text(
            "Maximum genetic distance (cM) within an HMM marker cluster "
            "(default: %(default)s; use 0 to disable clustering).",
            advanced=True,
        ),
    )
    model.add_argument(
        "--assume-global-phase",
        dest="assume_global_phase",
        action="store_true",
        help=help_text(
            (
                "Allow a fully phased PGEN target to bypass phasing when global "
                "phase continuity is known externally."
            ),
            advanced=True,
        ),
    )
    parser.set_defaults(assume_global_phase=False)

    output = parser.add_argument_group("Output Arguments")
    output.add_argument(
        "-o",
        "--out",
        required=True,
        help=(
            "Output prefix. VCF output appends .vcf.gz; PFILE output creates "
            ".pgen, .pvar, and .psam."
        ),
    )
    output.add_argument(
        "-fmt",
        "--format",
        choices=("vcf", "pfile"),
        default="vcf",
        help="Output genotype format (default: %(default)s).",
    )
    output.add_argument(
        "--ap",
        action="store_true",
        help=help_text(
            "Write AP1/AP2 haplotype probabilities when imputing.",
            advanced=True,
        ),
    )
    output.add_argument(
        "--gp",
        action="store_true",
        help=help_text(
            "Write GP genotype probabilities when imputing.",
            advanced=True,
        ),
    )

    runtime = parser.add_argument_group("Runtime Arguments")
    runtime.add_argument(
        "-t",
        "--thread",
        dest="threads",
        type=int,
        default=None,
        help="Number of CPU threads (default: auto-detected).",
    )
    runtime.add_argument(
        "-seed",
        type=int,
        default=-99_999,
        help=help_text("Random seed (default: %(default)s).", advanced=True),
    )
    diagnostics = parser.add_argument_group("Diagnostics")
    diagnostics.add_argument(
        "--trace-site",
        default=None,
        metavar="CHROM:POS:SAMPLE",
        help=help_text(
            (
                "Write per-round PhaseBaum forward/backward state vectors for one "
                "target site."
            ),
            advanced=True,
        ),
    )
    return parser


def main() -> int:
    wall_started = time.monotonic()
    parser = build_parser()
    args = parser.parse_args()
    if args.threads is None:
        args.threads = max(1, int(detect_effective_threads()))
    if args.threads <= 0:
        parser.error("-t/--thread must be positive")
    try:
        target_kind, target = _selected_target(args)
        reference = _resolve_reference_arg(args.reference)
        mode = _validate_modes(
            reference=reference,
            only_phase=bool(args.only_phase),
            ap=bool(args.ap),
            gp=bool(args.gp),
            output_format=args.format,
        )
        regions = (
            None
            if not args.bimrange
            else parse_cli_ranges(args.bimrange, allow_chrom_only=True)
        )
        diagnostic_site = _diagnostic_site(args.trace_site)
        from ._common.outprefix import validate_output_prefix_arg

        args.out = validate_output_prefix_arg(str(args.out), auto_prefix="imp")
    except ValueError as exc:
        parser.error(str(exc))

    output_prefix = _output_prefix(Path(args.out))
    output = _output_destination(output_prefix, args.format)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = str(output_prefix) + ".log"
    logger: logging.Logger = setup_logging(log_path)
    log_path = get_log_path(logger) or log_path
    terminal_rich = bool(stdout_is_tty() and rich_progress_available())
    if terminal_rich:
        report_logger = _logger_handlers(
            logger,
            "janusx.imp.file_only",
            file_handlers=True,
        )
        terminal_logger = _logger_handlers(
            logger,
            "janusx.imp.stream_only",
            file_handlers=False,
        )
    else:
        report_logger = logger
        terminal_logger = logger
    target_display = str(target)
    progress_target_label = (
        target_display
        if len(target_display) <= 56
        else os.path.basename(target_display.rstrip("/\\"))
    )
    input_format = {
        "bfile": "PLINK BED",
        "pfile": "PLINK 2 PGEN",
        "vcf": "VCF",
        "hmp": "HapMap",
    }.get(target_kind, target_kind.upper())
    reference_label = "None" if reference is None else str(reference)
    region_label = (
        "Whole input" if not regions else format_normalized_ranges(regions)
    )
    mode_label = _format_mode(mode)
    progress_label = (
        "Phasing"
        if bool(args.only_phase)
        else "Phasing&Imputing"
    )
    general_config_rows = [
        ("Genotype file", target_display),
        ("Input format", input_format),
        ("Reference", reference_label),
        ("Region", region_label),
        ("Mode", mode_label),
        ("Output format", str(args.format).upper()),
        ("Window", f"{float(args.window):g} cM"),
        ("Overlap", f"{float(args.overlap):g} cM"),
        ("Marker cap", f"{int(args.window_markers):,}"),
        ("Phase states", f"{int(args.phase_states):,}"),
        ("Imputation states", f"{int(args.imp_states):,}"),
        ("Seed", int(args.seed)),
        ("Threads", int(args.threads)),
    ]
    config_rows = [
        *general_config_rows,
        ("Output", str(output)),
        ("Version", _janusx_version()),
    ]
    host = socket.gethostname()
    if terminal_rich:
        emit_cli_configuration(
            terminal_logger,
            app_title="JanusX - IMP",
            config_title="IMP CONFIG",
            host=host,
            sections=[("General", general_config_rows)],
            footer_rows=[("Output prefix", str(output_prefix))],
            line_max_chars=_TERMINAL_SECTION_WIDTH,
        )
    _report_banner(report_logger, "JanusX - IMP Pipeline Analysis")
    _report_block(report_logger, "System Context")
    _report_kv(report_logger, "Host", host)
    _report_kv(report_logger, "Version", _janusx_version())
    _report_kv(report_logger, "Threads", int(args.threads))
    report_logger.info("")
    _report_block(report_logger, "Configuration")
    for key, value in config_rows:
        _report_kv(report_logger, key, value)
    report_logger.info("")
    _report_block(report_logger, "Command")
    report_logger.info(f"  {_invocation_command()}")
    report_logger.info("")
    _report_block(report_logger, "Stage Progress")
    options = {
        "target": str(target),
        "reference": None if reference is None else str(reference),
        "map": None if args.map is None else str(Path(args.map).expanduser()),
        "region": (
            {
                "chrom": regions[0].chrom_key,
                "start": regions[0].start,
                "end": regions[0].end,
            }
            if regions and len(regions) == 1
            else None
        ),
        "regions": (
            [
                {
                    "chrom": item.chrom_key,
                    "start": item.start,
                    "end": item.end,
                }
                for item in regions
            ]
            if regions
            else []
        ),
        "output": str(output),
        "format": "pgen" if args.format == "pfile" else "vcf",
        "threads": int(args.threads),
        "seed": int(args.seed),
        "window": float(args.window),
        "overlap": float(args.overlap),
        "window_markers": int(args.window_markers),
        "phase_states": int(args.phase_states),
        "imp_states": int(args.imp_states),
        "adaptive_states": bool(args.adaptive_states),
        "burnin": int(args.burnin),
        "iterations": int(args.iterations),
        "initial_lr": float(args.initial_lr),
        "ne": int(args.ne),
        "err": args.err,
        "em": bool(args.em),
        "impute": not bool(args.only_phase),
        "imp_segment": float(args.imp_segment),
        "imp_step": float(args.imp_step),
        "imp_nsteps": int(args.imp_nsteps),
        "cluster": float(args.cluster),
        "ap": bool(args.ap),
        "gp": bool(args.gp),
        "assume_global_phase": bool(args.assume_global_phase),
        "diagnostic_site": diagnostic_site,
    }
    progress_total = None
    inspection: _WindowInspection | None = None
    count_task = None
    if terminal_rich:
        count_task = CliStatus(
            f"Loading genotype from {progress_target_label}...",
            enabled=True,
            use_process=True,
        )
        count_task.__enter__()
    try:
        inspection = _inspect_target_dimensions(
            target_kind,
            target,
            regions,
            map_path=None
            if args.map is None
            else Path(args.map).expanduser(),
            window_cm=float(args.window),
            overlap_cm=float(args.overlap),
            window_markers=int(args.window_markers),
            threads=int(args.threads),
        )
        progress_total = int(inspection.markers) if inspection.markers > 0 else None
        if progress_total is not None:
            report_logger.info(
                f"  Progress denominator : {progress_total:,} target markers"
            )
        try:
            if inspection.markers == 0:
                raise ValueError(
                    "no markers remain inside the requested -bimrange"
                )
            reference_inspection = _inspect_reference_dimensions(
                reference,
                regions,
                map_path=None
                if args.map is None
                else Path(args.map).expanduser(),
                window_cm=float(args.window),
                overlap_cm=float(args.overlap),
                window_markers=int(args.window_markers),
                threads=int(args.threads),
            )
            inspection = _set_preflight_estimate(
                inspection,
                options=options,
                reference_inspection=reference_inspection,
            )
        except Exception as exc:
            message = f"Peak-memory preflight unavailable: {exc}"
            report_logger.warning(message)
            terminal_logger.warning(message)
        if count_task is not None:
            count_task.complete(
                f"Loading genotype from {progress_target_label} "
                f"(n={inspection.samples:,}, nSNP={inspection.markers:,})"
            )
    except Exception as exc:
        message = (
            "Genotype dimensions unavailable for startup planning. "
            f"{exc}"
        )
        report_logger.warning(message)
        terminal_logger.warning(message)
    finally:
        if count_task is not None:
            count_task.close(show_done=False)
    if inspection is not None and inspection.estimated_peak_rss_bytes:
        _emit_memory_preflight(report_logger, inspection)
        if terminal_logger is not report_logger:
            _emit_memory_preflight(terminal_logger, inspection)
    reporter = _ImpProgressReporter(
        report_logger if terminal_rich else logger,
        total=progress_total,
        task_label=progress_label,
        imputation_mode=(not bool(args.only_phase)),
    )
    if target_kind == "hmp":
        cache_prefix = _hmp_cache_prefix(target)
        cache_prefix.parent.mkdir(parents=True, exist_ok=True)
        if _pfile_cache_complete(cache_prefix):
            report_logger.info(f"Using HMP PFILE cache: {cache_prefix}")
        else:
            report_logger.info(f"Creating HMP PFILE cache: {cache_prefix}")
            try:
                jxrs.imp_convert_to_pgen(
                    str(target),
                    str(cache_prefix),
                    reporter,
                )
            except Exception as exc:
                reporter.close()
                logger.error(str(exc))
                return 1
        target = cache_prefix
        options["target"] = str(target)

    try:
        summary = json.loads(jxrs.imp_run(json.dumps(options), reporter))
    except Exception as exc:
        reporter.close()
        logger.error(str(exc))
        return 1
    reporter(
        "complete",
        "",
        0,
        int(summary["variants_written"]),
        int(summary["variants_written"]),
        int(summary["samples_written"]),
    )
    processing_elapsed = reporter.elapsed_seconds()
    reporter.close()
    if str(args.format).lower() == "pfile":
        generated_outputs = [
            Path(f"{output}.pgen"),
            Path(f"{output}.pvar"),
            Path(f"{output}.psam"),
        ]
    else:
        generated_outputs = [output]
    if summary.get("hmm_trace"):
        hmm_trace_path = Path(f"{output}.imp.hmm.tsv")
        with hmm_trace_path.open("w", newline="") as handle:
            trace_writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            trace_writer.writerow(
                (
                    "CHROM",
                    "POS",
                    "SAMPLE",
                    "WINDOW",
                    "ROUND",
                    "TRANSITION",
                    "STATE_ALLELES",
                    "EMISSION",
                    "FORWARD",
                    "BACKWARD",
                    "POSTERIOR_NORM",
                    "PHASE_SWAP",
                    "PHASE_LOG_ODDS",
                    "PHASE_THRESHOLD",
                    "SELECTED_ALLELES",
                )
            )
            for row in summary["hmm_trace"]:
                trace_writer.writerow(
                    (
                        row["chrom"],
                        row["position"],
                        row["sample"],
                        row["window_index"],
                        row["round"],
                        row["transition_probability"],
                        json.dumps(row["state_alleles"], separators=(",", ":")),
                        json.dumps(row["emission_probabilities"], separators=(",", ":")),
                        json.dumps(row["forward_probabilities"], separators=(",", ":")),
                        json.dumps(row["backward_probabilities"], separators=(",", ":")),
                        json.dumps(row["posterior_normalization"], separators=(",", ":")),
                        row["phase_swap"],
                        row["phase_log_odds"],
                        row["phase_threshold"],
                        json.dumps(row["selected_alleles"], separators=(",", ":")),
                    )
                )
        generated_outputs.append(hmm_trace_path)
    report_logger.info("")
    _report_block(report_logger, "Outputs Generated")
    for path in generated_outputs:
        _report_kv(report_logger, "Output", path)
    _report_kv(report_logger, "Log", log_path)
    report_logger.info("")
    _report_block(report_logger, "Advanced Parameters")
    _report_kv(report_logger, "Seed", int(args.seed))
    _report_kv(report_logger, "Window cM", float(args.window))
    _report_kv(report_logger, "Overlap cM", float(args.overlap))
    _report_kv(report_logger, "Window markers", int(args.window_markers))
    _report_kv(report_logger, "Phase states", int(args.phase_states))
    _report_kv(report_logger, "Imputation states", int(args.imp_states))
    _report_kv(report_logger, "Adaptive states", bool(args.adaptive_states))
    _report_kv(report_logger, "EM", bool(args.em))
    _report_kv(
        report_logger,
        "Assume global phase",
        bool(args.assume_global_phase),
    )
    _report_kv(
        report_logger,
        "Peak active bytes",
        int(summary["peak_window_bytes"]),
    )
    parameter_trace = summary.get("parameter_trace", [])
    if parameter_trace:
        report_logger.info("")
        _report_block(report_logger, "EM Parameter Trace")
        report_logger.info("  CHROM\tWINDOW\tSTAGE\tROUND\tNE\tERR")
        for row in parameter_trace:
            report_logger.info(
                f"  {row['chrom']}\t{int(row['window_index'])}\t{row['stage']}"
                f"\t{int(row['round'])}\t{int(row['ne'])}\t{float(row['err']):.6g}"
            )
    report_logger.info("")
    _report_block(report_logger, "Execution Summary")
    _report_kv(report_logger, "Status", "done")
    _report_kv(report_logger, "Variants written", int(summary["variants_written"]))
    _report_kv(report_logger, "Samples", int(summary["samples_written"]))
    _report_kv(
        report_logger,
        "Total windows",
        int(summary["total_target_windows"]),
    )
    _report_kv(
        report_logger,
        "Direct-imputation",
        int(summary["direct_imputation_windows"]),
    )
    _report_kv(
        report_logger,
        "Normal phasing",
        int(summary["normal_phasing_windows"]),
    )
    _report_kv(
        report_logger,
        "Reference-assisted phasing",
        int(summary.get("reference_phasing_windows", 0)),
    )
    _report_kv(
        report_logger,
        "Imputation windows",
        int(summary["imputation_windows"]),
    )
    _report_kv(
        report_logger,
        "Direct bypass",
        f"{100.0 * float(summary['direct_bypass_fraction']):.1f}%",
    )
    if summary.get("direct_fallback_reasons"):
        _report_kv(
            report_logger,
            "Fallback reasons",
            json.dumps(summary["direct_fallback_reasons"], sort_keys=True),
        )
    elapsed_seconds = max(time.monotonic() - wall_started, 0.0)
    _report_kv(report_logger, "Total wall time", f"{elapsed_seconds:.2f}s")
    report_logger.info(f"IMP run status: done; log: {log_path}")
    if terminal_rich:
        phase_performed = int(summary["normal_phasing_windows"]) > 0
        imputation_performed = int(summary["imputation_windows"]) > 0
        if phase_performed and imputation_performed:
            completion_label = "Phasing&Imputing"
        elif phase_performed:
            completion_label = "Phasing"
        elif imputation_performed:
            completion_label = "Imputing"
        else:
            completion_label = "Phasing"
        log_success(
            terminal_logger,
            f"{completion_label} ...Finished "
            f"({int(summary['variants_written']):,} markers written) "
            f"[{format_elapsed(processing_elapsed)}]",
            force_color=True,
        )
        saved_lines = "\n".join(f"  {path}" for path in generated_outputs)
        saved_format = "PFILE" if str(args.format).lower() == "pfile" else "VCF"
        terminal_logger.info(
            f"Saved genotype in {saved_format} format:\n{saved_lines}"
        )
        lt = time.localtime()
        endinfo = (
            f"\nFinished IMP calculation. Total wall time: {elapsed_seconds:.2f} seconds\n"
            f"{lt.tm_year}-{lt.tm_mon:02d}-{lt.tm_mday:02d} "
            f"{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}"
        )
        _emit_to_file_handlers(report_logger, logging.INFO, endinfo)
        log_success(terminal_logger, endinfo)
    else:
        log_success(
            report_logger,
            f"Phased {int(summary['variants_written'])} variants for "
            f"{int(summary['samples_written'])} samples.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
