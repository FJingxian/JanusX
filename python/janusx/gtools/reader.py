"""
General-purpose readers and range query helpers used across JanusX.
"""

from pathlib import Path
from typing import Iterable, Optional, Union
import gzip
import re
import warnings
from urllib.parse import unquote

import numpy as np
import pandas as pd

pathlike = Union[str, Path]
_BIMRANGE_RE = re.compile(r"^([^:]+):([0-9]*\.?[0-9]+)(?:-|:)([0-9]*\.?[0-9]+)$")
__all__ = [
    "GFFQuery",
    "gffreader",
    "readanno",
    "bedreader",
]


def _normalize_chr(chrom: object) -> str:
    """
    Normalize chromosome label by stripping leading 'chr' (case-insensitive).
    """
    text = str(chrom).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    return text


def _parse_bimrange(bimrange: str) -> tuple[str, int, int]:
    """
    Parse bimrange in Mb:
      - chr:start-end
      - chr:start:end

    Parameters
    ----------
    bimrange : str
        Range string in Mb.

    Returns
    -------
    tuple[str, int, int]
        (chrom, start_bp, end_bp)
    """
    text = str(bimrange).strip()
    m = _BIMRANGE_RE.match(text)
    if m is None:
        raise ValueError(
            f"Invalid bimrange format: {bimrange}. "
            "Use chr:start-end (also accepts chr:start:end)."
        )
    chrom = m.group(1).strip()
    start_v = float(m.group(2))
    end_v = float(m.group(3))
    if start_v > end_v:
        start_v, end_v = end_v, start_v
    if start_v < 0 or end_v < 0:
        raise ValueError("bimrange start/end must be >= 0.")
    start = int(round(start_v * 1_000_000))
    end = int(round(end_v * 1_000_000))
    return chrom, start, end


def _normalize_attr_keys(attr: Optional[Iterable[str]]) -> list[str]:
    """
    Normalize requested GFF attribute keys and keep order (deduplicated).
    """
    if attr is None:
        keys = ["ID", "description"]
    elif isinstance(attr, str):
        keys = [attr]
    else:
        keys = list(attr)

    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        text = str(key).strip()
        if text == "" or text in seen:
            continue
        seen.add(text)
        out.append(text)
    if len(out) == 0:
        raise ValueError("attr must contain at least one non-empty key.")
    return out


def _extract_attr_value(attr_series: pd.Series, key: str) -> pd.Series:
    """
    Extract one attribute value from GFF attribute text by key.

    Supports both:
      - "...;key=value;..."
      - "...;key=value" (line end)
      - "... key=value next_key=..." (space-delimited fallback)
    """
    key_pat = re.escape(str(key))
    # Stop at ';' OR at the next " <token>=" boundary to support
    # malformed space-delimited attribute strings.
    pattern = rf"(?:^|;|\s){key_pat}=([^;]*?)(?=;|\s+[^\s;=]+=|$)"
    out = attr_series.astype(str).str.extract(pattern, expand=False).fillna("NA")

    def _normalize_text(x: object) -> object:
        if not isinstance(x, str):
            return x
        text = unquote(x).strip()
        return re.sub(r"\s+", " ", text)

    return out.map(_normalize_text)


def _extract_attr_values(attr_series: pd.Series, keys: Iterable[str]) -> pd.DataFrame:
    """
    Extract multiple attribute keys from GFF attribute text into a DataFrame.
    """
    keys_list = [str(k) for k in keys]
    data = {k: _extract_attr_value(attr_series, k) for k in keys_list}
    return pd.DataFrame(data, index=attr_series.index)


def _extract_attr_list(attr_series: pd.Series, keys: Iterable[str]) -> pd.Series:
    """
    Extract multiple keys and return list values in key order for each row.
    """
    keys_list = [str(k) for k in keys]
    values = _extract_attr_values(attr_series, keys_list)
    return pd.Series(values[keys_list].values.tolist(), index=attr_series.index)


def _warn_missing_features(
    requested_raw: Iterable[str],
    missing_lc: set[str],
    available_raw: Iterable[str],
    chrom: str,
    start: int,
    end: int,
) -> None:
    """
    Emit warning when requested feature filters are not present in the queried range.
    """
    missing_requested = []
    for x in requested_raw:
        text = str(x).strip()
        if text == "":
            continue
        if text.lower() in missing_lc:
            missing_requested.append(text)
    # Keep input order but remove duplicates.
    missing_requested = list(dict.fromkeys(missing_requested))

    available_list = sorted(list({str(x) for x in available_raw if str(x).strip() != ""}))
    available_text = ", ".join(available_list) if len(available_list) > 0 else "None"
    missing_text = ", ".join(missing_requested) if len(missing_requested) > 0 else ", ".join(sorted(missing_lc))
    warnings.warn(
        f"Warning: feature(s) not found in range {chrom}:{start}-{end}: {missing_text}. "
        f"Available features: {available_text}.",
        RuntimeWarning,
        stacklevel=3,
    )


def gffreader(gffpath: pathlike, attr: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Read GFF/GFF3 (plain text or .gz) into a normalized DataFrame.

    Returns columns:
    chrom, source, feature, start, end, score, strand, phase, attributes, chrom_norm
    If `attr` is provided, also returns:
    attribute

    Parameters
    ----------
    gffpath : pathlike
        GFF/GFF3 path.
    attr : iterable[str] or str or None
        Optional GFF attribute keys extracted into `attribute` list column.
        If None, skip attribute extraction.
    """
    path = Path(gffpath)
    if not path.exists():
        raise FileNotFoundError(f"GFF file not found: {path}")
    attr_keys: Optional[list[str]] = None
    if attr is not None:
        attr_keys = _normalize_attr_keys(attr)

    rows: list[list[str]] = []
    skipped_too_short = 0
    merged_extra_cols = 0
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            line = line.rstrip("\r\n")
            if line == "":
                continue

            parts = line.split("\t")
            if len(parts) < 9:
                parts = re.split(r"\s+", line, maxsplit=8)
            if len(parts) < 9:
                skipped_too_short += 1
                continue
            if len(parts) > 9:
                merged_extra_cols += 1
                attr_head = parts[8].replace(";", " ")
                attr_tail = " ".join(parts[9:]).strip()
                attr = f"{attr_head} {attr_tail}".strip() if attr_tail else attr_head
                attr = re.sub(r"\s+", " ", attr).strip()
                parts = parts[:8] + [attr]
            rows.append(parts[:9])

    if len(rows) == 0:
        raise ValueError(f"No valid GFF rows found in file: {path}")

    gff = pd.DataFrame(rows, columns=list(range(9)))
    gff.columns = [
        "chrom",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "attributes",
    ]
    gff["start"] = pd.to_numeric(gff["start"], errors="coerce")
    gff["end"] = pd.to_numeric(gff["end"], errors="coerce")
    gff = gff.dropna(subset=["chrom", "feature", "start", "end"]).copy()
    gff["start"] = gff["start"].astype(int)
    gff["end"] = gff["end"].astype(int)
    gff["chrom"] = gff["chrom"].astype(str)
    gff["feature"] = gff["feature"].astype(str)
    gff["chrom_norm"] = gff["chrom"].map(_normalize_chr)
    if attr_keys is not None:
        gff["attribute"] = _extract_attr_list(gff["attributes"], attr_keys)
        gff.attrs["attribute_keys"] = attr_keys
    gff = gff.sort_values(["chrom_norm", "start", "end"]).reset_index(drop=True)
    if skipped_too_short > 0 or merged_extra_cols > 0:
        warnings.warn(
            f"Warning: gffreader adjusted malformed rows in {path} "
            f"(skipped <9 cols: {skipped_too_short}, merged >9 cols: {merged_extra_cols}).",
            RuntimeWarning,
            stacklevel=2,
        )
    return gff


def bedreader(annofile: pathlike) -> pd.DataFrame:
    """
    Read BED annotation and guarantee columns 0..5 exist
    (missing columns are filled with 'NA').
    """
    anno = pd.read_csv(annofile, sep="\t", header=None).fillna("NA")
    if anno.shape[1] <= 4:
        anno[4] = "NA"
    if anno.shape[1] <= 5:
        anno[5] = "NA"
    return anno


def readanno(annofile: str, descItem: str = "description") -> pd.DataFrame:
    """
    Read BED/GFF annotation into JanusX unified schema.

    Output columns after processing:
      0 = chr, 1 = start, 2 = end, 3 = geneID, 4 = desc1, 5 = desc2
    """
    suffix = str(annofile).replace(".gz", "").split(".")[-1].lower()

    if suffix == "bed":
        return bedreader(annofile)

    if suffix in {"gff", "gff3"}:
        gff = gffreader(annofile)
        if gff.shape[0] == 0:
            return pd.DataFrame(columns=[0, 1, 2, 3, 4, 5])

        anno = gff.loc[:, ["chrom", "feature", "start", "end", "attributes"]].copy()
        anno.columns = [0, 2, 3, 4, 8]

        anno = anno[anno[2] == "gene"]
        if anno.shape[0] == 0:
            return pd.DataFrame(columns=[0, 1, 2, 3, 4, 5])

        del anno[2]
        anno[0] = anno[0].astype(str).str.strip()
        anno[3] = pd.to_numeric(anno[3], errors="coerce")
        anno[4] = pd.to_numeric(anno[4], errors="coerce")
        anno = anno.dropna(subset=[3, 4])
        anno[3] = anno[3].astype(int)
        anno[4] = anno[4].astype(int)
        anno = anno.sort_values([0, 3])
        anno.columns = range(anno.shape[1])

        attrs = _extract_attr_values(anno[3], ["ID", descItem])
        anno[4] = attrs[descItem]
        anno[3] = attrs["ID"]
        anno[3] = anno[3].str.replace(r"^[^:]*:", "", regex=True)
        anno[5] = "NA"
        return anno

    raise ValueError(f"Unsupported annotation suffix: {suffix} (file: {annofile})")


class GFFQuery:
    """
    Indexed query helper for repeated range lookups on GFF data.
    """

    def __init__(self, gff: pd.DataFrame):
        required = {"chrom_norm", "start", "end", "feature"}
        missing = required.difference(gff.columns)
        if missing:
            raise ValueError(f"Missing required columns in GFF DataFrame: {sorted(missing)}")

        self.gff = gff.copy()
        self._empty = self.gff.iloc[0:0].copy()
        self._use_iloc = (
            isinstance(self.gff.index, pd.RangeIndex)
            and self.gff.index.start == 0
            and self.gff.index.step == 1
        )
        self._chr_index: dict[str, dict[str, np.ndarray]] = {}
        for chrom, block in self.gff.groupby("chrom_norm", sort=False):
            # Build per-chromosome numpy arrays for fast range filtering.
            block = block.sort_values(["start", "end"], kind="mergesort")
            rowids = block.index.to_numpy()
            if self._use_iloc:
                rowids = rowids.astype(np.int64, copy=False)
            self._chr_index[str(chrom)] = {
                "rowids": rowids,
                "starts": block["start"].to_numpy(dtype=np.int64),
                "ends": block["end"].to_numpy(dtype=np.int64),
                "features": block["feature"].astype(str).str.lower().to_numpy(dtype=object),
                "features_raw": block["feature"].astype(str).to_numpy(dtype=object),
            }

    @classmethod
    def from_file(cls, gffpath: pathlike) -> "GFFQuery":
        """
        Build a `GFFQuery` index directly from GFF/GFF3 file.
        """
        return cls(gffreader(gffpath))

    def query_range(
        self,
        chrom: object,
        start: int,
        end: int,
        *,
        features: Optional[Iterable[str]] = None,
        attr: Optional[Iterable[str]] = ("ID", "description"),
        overlap: bool = True,
    ) -> pd.DataFrame:
        """
        Query GFF records on one chromosome range.

        Parameters
        ----------
        chrom : object
            Chromosome label.
        start, end : int
            Range boundaries in bp.
        features : iterable[str] or str or None
            Optional feature filter (e.g. {"gene", "mRNA"}).
        attr : iterable[str] or str or None
            Optional GFF attribute keys extracted into output `attribute` list column.
            Default: ("ID", "description"). Set to None to disable extraction.
        overlap : bool
            If True, return overlapping records; else fully-contained records.
        """
        chrom_norm = _normalize_chr(chrom)
        if start > end:
            start, end = end, start
        attr_keys: Optional[list[str]] = None
        if attr is not None:
            attr_keys = _normalize_attr_keys(attr)
        requested_raw_all: list[str] = []
        requested_set_all: set[str] = set()
        if features is not None:
            if isinstance(features, str):
                requested_raw_all = [features]
            else:
                requested_raw_all = [str(x) for x in features]
            requested_set_all = {
                str(x).strip().lower()
                for x in requested_raw_all
                if str(x).strip() != ""
            }

        hit = self._chr_index.get(chrom_norm)
        if hit is None:
            if len(requested_set_all) > 0:
                _warn_missing_features(
                    requested_raw=requested_raw_all,
                    missing_lc=requested_set_all,
                    available_raw=[],
                    chrom=chrom_norm,
                    start=int(start),
                    end=int(end),
                )
            out = self._empty.copy()
            if attr_keys is not None:
                out["attribute"] = pd.Series(dtype=object)
                out.attrs["attribute_keys"] = attr_keys
            return out
        starts = hit["starts"]
        ends = hit["ends"]
        features_lc = hit["features"]
        features_raw = hit["features_raw"]
        rowids = hit["rowids"]

        hi = int(np.searchsorted(starts, int(end), side="right"))
        if hi <= 0:
            if len(requested_set_all) > 0:
                _warn_missing_features(
                    requested_raw=requested_raw_all,
                    missing_lc=requested_set_all,
                    available_raw=[],
                    chrom=chrom_norm,
                    start=int(start),
                    end=int(end),
                )
            out = self._empty.copy()
            if attr_keys is not None:
                out["attribute"] = pd.Series(dtype=object)
                out.attrs["attribute_keys"] = attr_keys
            return out

        starts_s = starts[:hi]
        ends_s = ends[:hi]
        features_s = features_lc[:hi]
        features_raw_s = features_raw[:hi]
        rowids_s = rowids[:hi]

        if overlap:
            mask = ends_s >= int(start)
        else:
            mask = (starts_s >= int(start)) & (ends_s <= int(end))

        if features is not None:
            requested_raw = requested_raw_all
            feature_set = requested_set_all
            if len(feature_set) == 0:
                feature_set = set()
            available_in_range = {str(x).strip().lower() for x in features_s[mask]}
            missing = feature_set.difference(available_in_range)
            if len(missing) > 0:
                _warn_missing_features(
                    requested_raw=requested_raw,
                    missing_lc=missing,
                    available_raw=features_raw_s[mask],
                    chrom=chrom_norm,
                    start=int(start),
                    end=int(end),
                )
            if len(feature_set) == 1:
                only = next(iter(feature_set))
                mask = mask & (features_s == only)
            else:
                mask = mask & np.isin(features_s, list(feature_set))

        if not bool(np.any(mask)):
            out = self._empty.copy()
            if attr_keys is not None:
                out["attribute"] = pd.Series(dtype=object)
                out.attrs["attribute_keys"] = attr_keys
            return out

        hit_rowids = rowids_s[mask]
        out: pd.DataFrame
        if self._use_iloc:
            out = self.gff.iloc[np.asarray(hit_rowids, dtype=np.int64)].copy()
        else:
            out = self.gff.loc[hit_rowids].copy()
        if attr_keys is not None:
            out["attribute"] = _extract_attr_list(out["attributes"], attr_keys)
            out.attrs["attribute_keys"] = attr_keys
        return out

    def query_bimrange(
        self,
        bimrange: str,
        *,
        features: Optional[Iterable[str]] = None,
        attr: Optional[Iterable[str]] = ("ID", "description"),
        overlap: bool = True,
    ) -> pd.DataFrame:
        """
        Query by bimrange string in Mb.
        Format: chr:start-end (also accepts chr:start:end).
        """
        chrom, start, end = _parse_bimrange(bimrange)
        return self.query_range(
            chrom,
            start,
            end,
            features=features,
            attr=attr,
            overlap=overlap,
        )
