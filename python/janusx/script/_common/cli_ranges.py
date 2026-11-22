"""Shared chromosome and physical-range parsing for JanusX CLIs."""

from __future__ import annotations

from dataclasses import dataclass
import re


_RANGE_RE = re.compile(r"^(?P<chrom>[^:：]+):(?P<start>[^:：-]+)[：:-](?P<end>[^:：]+)$")
_COORDINATE_RE = re.compile(
    r"^\s*(?P<number>[0-9]*\.?[0-9]+)\s*(?P<unit>bp|mb)?\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class NormalizedRange:
    original: str
    chrom_original: str
    chrom_key: str
    start: int | None
    end: int | None
    unit: str | None


def normalize_chromosome_key(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    text = text.upper()
    if not text:
        raise ValueError("Invalid -bimrange/--bimrange: chromosome must not be empty.")
    return text


def _looks_like_implicit_bp(number_text: str) -> bool:
    if "." in number_text:
        return False
    normalized = number_text.lstrip("0") or "0"
    return len(normalized) > 6


def _parse_coordinate(token: str) -> tuple[float, str | None]:
    match = _COORDINATE_RE.fullmatch(str(token).strip())
    if match is None:
        raise ValueError(
            "Invalid -bimrange/--bimrange coordinate: "
            f"{token!r}. Use a number with optional bp or Mb suffix."
        )
    number = float(match.group("number"))
    if number < 0:
        raise ValueError("Invalid -bimrange/--bimrange: coordinates must be >= 0.")
    unit = match.group("unit")
    return number, unit.lower() if unit is not None else None


def _resolve_unit(
    start_token: str,
    end_token: str,
    start_unit: str | None,
    end_unit: str | None,
    *,
    implicit_unit: str = "auto",
) -> str:
    if (start_unit is None) != (end_unit is None):
        raise ValueError(
            "Invalid -bimrange/--bimrange: use an explicit unit on both bounds "
            "or omit it from both bounds."
        )
    if start_unit is not None and end_unit is not None:
        if start_unit != end_unit:
            raise ValueError(
                "Invalid -bimrange/--bimrange: both bounds must use the same unit."
            )
        return start_unit

    if implicit_unit == "bp":
        if "." in start_token or "." in end_token:
            raise ValueError(
                "Invalid -bimrange/--bimrange: fractional coordinates require "
                "an explicit Mb suffix."
            )
        return "bp"
    if implicit_unit != "auto":
        raise ValueError(f"Unsupported implicit_unit: {implicit_unit!r}")

    start_bp = _looks_like_implicit_bp(start_token)
    end_bp = _looks_like_implicit_bp(end_token)
    if start_bp != end_bp:
        raise ValueError(
            "Invalid -bimrange/--bimrange: implicit bp/Mb units are ambiguous; "
            "add bp or Mb suffixes to both bounds."
        )
    return "bp" if start_bp else "mb"


def _coordinate_to_bp(value: float, unit: str) -> int:
    multiplier = 1_000_000 if unit == "mb" else 1
    return int(round(value * multiplier))


def parse_cli_range(
    value: object,
    *,
    allow_chrom_only: bool,
    implicit_unit: str = "auto",
) -> NormalizedRange:
    text = str(value).strip().replace("：", ":")
    if not text:
        raise ValueError("Invalid -bimrange/--bimrange: value must not be empty.")
    if ":" not in text and "-" not in text:
        if not allow_chrom_only:
            raise ValueError(
                f"Invalid -bimrange/--bimrange format: {value}. "
                "Use chr:start-end (or chr:start:end)."
            )
        chrom = text.strip()
        return NormalizedRange(
            original=str(value),
            chrom_original=chrom,
            chrom_key=normalize_chromosome_key(chrom),
            start=None,
            end=None,
            unit=None,
        )

    normalized = text
    if normalized.count(":") == 1 and "-" in normalized:
        chrom, tail = normalized.split(":", 1)
        start_token, end_token = tail.split("-", 1)
    elif normalized.count(":") >= 2:
        parts = normalized.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid -bimrange/--bimrange format: {value}. "
                "Use chr:start-end (or chr:start:end)."
            )
        chrom, start_token, end_token = parts
    else:
        match = _RANGE_RE.fullmatch(normalized)
        if match is None:
            raise ValueError(
                f"Invalid -bimrange/--bimrange format: {value}. "
                "Use chr:start-end (or chr:start:end)."
            )
        chrom = match.group("chrom")
        start_token = match.group("start")
        end_token = match.group("end")

    chrom = chrom.strip()
    if not chrom:
        raise ValueError("Invalid -bimrange/--bimrange: chromosome must not be empty.")
    start_num, start_unit = _parse_coordinate(start_token)
    end_num, end_unit = _parse_coordinate(end_token)
    unit = _resolve_unit(
        start_token.strip(),
        end_token.strip(),
        start_unit,
        end_unit,
        implicit_unit=implicit_unit,
    )
    start_bp = _coordinate_to_bp(start_num, unit)
    end_bp = _coordinate_to_bp(end_num, unit)
    if start_bp > end_bp:
        start_bp, end_bp = end_bp, start_bp
    return NormalizedRange(
        original=str(value),
        chrom_original=chrom,
        chrom_key=normalize_chromosome_key(chrom),
        start=start_bp,
        end=end_bp,
        unit=unit,
    )


def _flatten_cli_range_tokens(values: object) -> list[str]:
    if values is None:
        return []
    raw_items: list[object]
    if isinstance(values, (str, bytes)):
        raw_items = [values]
    elif isinstance(values, (list, tuple)):
        raw_items = []
        for item in values:
            if isinstance(item, (list, tuple)):
                raw_items.extend(item)
            else:
                raw_items.append(item)
    else:
        raw_items = [values]

    tokens: list[str] = []
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue
        for part in text.split(","):
            piece = part.strip()
            if piece:
                tokens.append(piece)
    return tokens


def parse_cli_ranges(
    values: object,
    *,
    allow_chrom_only: bool,
    implicit_unit: str = "auto",
) -> list[NormalizedRange]:
    tokens = _flatten_cli_range_tokens(values)
    if not tokens:
        raise ValueError("Invalid -bimrange/--bimrange: value must not be empty.")
    return [
        parse_cli_range(
            token,
            allow_chrom_only=allow_chrom_only,
            implicit_unit=implicit_unit,
        )
        for token in tokens
    ]


def format_normalized_range(value: NormalizedRange) -> str:
    if value.start is None or value.end is None:
        return value.chrom_original
    return f"{value.chrom_original}:{int(value.start)}-{int(value.end)}"


def format_normalized_ranges(
    values: list[NormalizedRange] | tuple[NormalizedRange, ...],
) -> str:
    return ",".join(format_normalized_range(item) for item in values)
