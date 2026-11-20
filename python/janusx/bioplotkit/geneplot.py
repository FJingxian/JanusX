from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from tqdm import trange
from matplotlib.artist import Artist
from matplotlib import colors as mcolors
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import IdentityTransform
from janusx.gtools.reader import GFFQuery, gffreader

pathlike = Union[str, Path]
_BIMRANGE_RE = re.compile(r"^([^:]+):([0-9]*\.?[0-9]+)(?:-|:)([0-9]*\.?[0-9]+)$")


def _parse_bimrange_mbp(bimrange: str) -> tuple[str, int, int]:
    """
    Parse bimrange string and convert Mb to bp.

    Accept:
    - chr:start-end
    - chr:start:end
    """
    text = str(bimrange).strip()
    m = _BIMRANGE_RE.match(text)
    if m is None:
        raise ValueError(
            f"Invalid bimrange format: {bimrange}. "
            "Use chr:start-end (also accepts chr:start:end)."
        )
    chrom = str(m.group(1)).strip()
    start_mb = float(m.group(2))
    end_mb = float(m.group(3))
    if start_mb > end_mb:
        start_mb, end_mb = end_mb, start_mb
    start_bp = int(round(start_mb * 1_000_000))
    end_bp = int(round(end_mb * 1_000_000))
    return chrom, start_bp, end_bp


def _gene_interval_lanes(
    gene_df: pd.DataFrame,
    *,
    gene_text_size: float = 4.0,
) -> dict[object, int]:
    """Assign non-overlapping gene intervals to compact vertical lanes."""
    genes = gene_df[gene_df["feature"].astype(str).str.lower() == "gene"]
    if genes.empty:
        return {}
    items = []
    x_values = pd.to_numeric(
        pd.concat([genes["x_start"], genes["x_end"]]), errors="coerce"
    ).dropna()
    x_span = max(1.0, float(x_values.max() - x_values.min())) if not x_values.empty else 1.0
    # Long IDs need horizontal room for their labels.  Reserve an estimated
    # text width during lane coloring so adjacent labels are separated too.
    label_scale = max(0.0, float(gene_text_size)) / 4.0
    for idx, row in genes.iterrows():
        left = min(float(row["x_start"]), float(row["x_end"]))
        right = max(float(row["x_start"]), float(row["x_end"]))
        attr = row.get("attribute", "")
        if isinstance(attr, (list, tuple, np.ndarray)) and len(attr) > 0:
            label = str(attr[0])
        else:
            label = str(attr)
        label_pad = x_span * min(0.18, 0.004 * max(1, len(label)) * label_scale)
        items.append((left - label_pad, right + label_pad, idx))
    items.sort(key=lambda item: (item[0], item[1], str(item[2])))
    lane_ends: list[float] = []
    result: dict[object, int] = {}
    for left, right, idx in items:
        lane = next((i for i, end in enumerate(lane_ends) if left > end), None)
        if lane is None:
            lane = len(lane_ends)
            lane_ends.append(right)
        else:
            lane_ends[lane] = right
        result[idx] = lane
    return result


def gene_structure_lane_count(
    gene_df: pd.DataFrame,
    *,
    gene_text_size: float = 4.0,
) -> int:
    """Return the number of vertical lanes needed by the gene records."""
    lanes = _gene_interval_lanes(gene_df, gene_text_size=gene_text_size)
    return max(lanes.values(), default=-1) + 1


class _IGVDirectionChevronArtist(Artist):
    """Draw compact font-derived direction glyphs in display units.

    The glyph is rendered from the white ``>``/``<`` character as a filled
    path.  This avoids the open-path seam that appears when two stroked arms
    meet at a very small tip, while still allowing the final height and width
    to be controlled in display units.
    """

    def __init__(
        self,
        ax: plt.Axes,
        spans: Iterable[tuple[float, float]],
        y: float,
        strand: str,
        color: str,
        *,
        triangle_leg_px: Optional[float] = None,
        backbone_line_width: float = 0.5,
        max_symbols: Optional[int] = None,
        fallback_spans: Optional[Iterable[tuple[float, float]]] = None,
        zorder: float = 4.0,
    ) -> None:
        super().__init__()
        self._fallback_axes = ax
        self._spans = tuple(
            (min(float(left), float(right)), max(float(left), float(right)))
            for left, right in spans
            if float(right) > float(left)
        )
        self._y = float(y)
        self._strand = str(strand)
        self._color = color
        self._triangle_leg_px = (
            None
            if triangle_leg_px is None
            else max(0.15, float(triangle_leg_px))
        )
        self._backbone_line_width = max(0.1, float(backbone_line_width))
        self._max_symbols = (
            None
            if max_symbols is None
            else max(1, int(max_symbols))
        )
        self._fallback_spans = tuple(
            (min(float(left), float(right)), max(float(left), float(right)))
            for left, right in (fallback_spans or ())
            if float(right) > float(left)
        )
        self.set_zorder(float(zorder))
        self.set_clip_on(True)

    def draw(self, renderer) -> None:  # pragma: no cover - exercised by savefig
        if (
            not self.get_visible()
            or (len(self._spans) == 0 and len(self._fallback_spans) == 0)
        ):
            return
        ax = self.axes if self.axes is not None else self._fallback_axes
        if ax is None:
            return

        transform = ax.transData
        strand_sign = 1.0 if self._strand == "+" else -1.0
        def _build_path(
            spans: tuple[tuple[float, float], ...],
        ) -> tuple[list[tuple[float, float]], list[int]]:
            vertices: list[tuple[float, float]] = []
            codes: list[int] = []
            symbols_left = self._max_symbols
            font = FontProperties(
                family="Arial",
                style="normal",
                weight="normal",
            )
            for left, right in spans:
                if symbols_left is not None and symbols_left <= 0:
                    break
                x0, y0 = transform.transform((left, self._y))
                x1, _y1 = transform.transform((right, self._y))
                if not np.isfinite([x0, y0, x1]).all():
                    continue

                display_left = min(x0, x1)
                display_right = max(x0, x1)
                span_px = display_right - display_left
                line_width_px = float(renderer.points_to_pixels(self._backbone_line_width))
                line_half_px = max(0.15, 0.5 * line_width_px)
                leg_px = (
                    line_half_px
                    if self._triangle_leg_px is None
                    else min(self._triangle_leg_px, line_half_px)
                )
                if span_px <= 0.0 or leg_px <= 0.0:
                    continue

                # The arrow points in genomic strand direction.  This also
                # keeps the direction correct if a caller later reverses the
                # x-axis.
                data_to_display_sign = 1.0 if x1 >= x0 else -1.0
                direction_sign = strand_sign * data_to_display_sign
                # Keep the chevrons inside the backbone interval.  A stroke
                # is centered on its path, so a chevron placed exactly at a
                # gene edge would protrude by half a line width.
                edge_inset = line_half_px
                safe_left = display_left + edge_inset
                safe_right = display_right - edge_inset
                safe_span_px = safe_right - safe_left
                if safe_span_px <= 0.0:
                    continue

                # The glyph height remains the backbone width (2 * leg_px),
                # while its horizontal width stays compact.  The filled font
                # shape has no stroke expansion, so this is its true visible
                # width rather than the width of a thick V-shaped line.
                chevron_height = 2.0 * leg_px
                chevron_width = max(
                    0.20,
                    min(0.72 * chevron_height, 0.80 * safe_span_px),
                )
                chevron_gap = max(
                    0.5,
                    float(renderer.points_to_pixels(1.25)),
                )
                requested_count = max(
                    1,
                    int(
                        np.floor(
                            (safe_span_px + chevron_gap)
                            / (chevron_width + chevron_gap)
                        )
                    ),
                )
                count = (
                    requested_count
                    if symbols_left is None
                    else min(symbols_left, requested_count)
                )
                if count <= 0:
                    break
                if count == 1:
                    tile_width = min(chevron_width, 0.45 * safe_span_px)
                    tile_starts = [
                        safe_left + 0.5 * (safe_span_px - tile_width)
                    ]
                else:
                    tile_width = min(
                        chevron_width,
                        safe_span_px / float(count),
                    )
                    actual_gap = (
                        safe_span_px - count * tile_width
                    ) / float(count - 1)
                    tile_starts = [
                        safe_left + tile_index * (tile_width + actual_gap)
                        for tile_index in range(count)
                    ]
                glyph_text = ">" if direction_sign > 0 else "<"
                glyph = TextPath((0.0, 0.0), glyph_text, size=1.0, prop=font)
                bbox = glyph.get_extents()
                if bbox.width <= 0.0 or bbox.height <= 0.0:
                    continue
                glyph_vertices = np.asarray(glyph.vertices, dtype=float)
                glyph_codes = np.asarray(glyph.codes, dtype=np.uint8)
                scale_x = tile_width / float(bbox.width)
                scale_y = chevron_height / float(bbox.height)
                for tile_left in tile_starts:
                    glyph_center = tile_left + 0.5 * tile_width
                    transformed = glyph_vertices.copy()
                    transformed[:, 0] = (
                        glyph_center
                        - 0.5 * tile_width
                        + (transformed[:, 0] - float(bbox.x0)) * scale_x
                    )
                    transformed[:, 1] = (
                        y0
                        - 0.5 * chevron_height
                        + (transformed[:, 1] - float(bbox.y0)) * scale_y
                    )
                    vertices.extend(map(tuple, transformed.tolist()))
                    codes.extend(glyph_codes.tolist())
                if symbols_left is not None:
                    symbols_left -= count
            return vertices, codes

        vertices, codes = _build_path(self._spans)
        # Keep the fallback for direct callers that provide only a sparse set
        # of spans.  postgwas passes the complete gene span explicitly.
        if len(vertices) == 0 and len(self._fallback_spans) > 0:
            vertices, codes = _build_path(self._fallback_spans)

        if len(vertices) == 0:
            return

        path = MplPath(np.asarray(vertices, dtype=float), codes)
        gc = renderer.new_gc()
        try:
            gc.set_clip_rectangle(ax.bbox)
            gc.set_linewidth(0.0)
            fill = mcolors.to_rgba(self._color)
            gc.set_foreground(fill, isRGBA=True)
            renderer.draw_path(
                gc,
                path,
                IdentityTransform(),
                rgbFace=fill,
            )
        finally:
            gc.restore()


class _GeneFeatureBlockArtist(Artist):
    """Draw a CDS/UTR block in display units relative to the gene line.

    The gene backbone and direction glyphs are sized in screen units, while
    ordinary Matplotlib rectangles use data-unit heights.  That difference
    made the apparent CDS/UTR/backbone ratio change with the figure layout.
    This artist keeps the feature height tied to the same rendered line width
    for every backend and every ``bimrange``.
    """

    def __init__(
        self,
        ax: plt.Axes,
        left: float,
        right: float,
        y: float,
        color: str,
        *,
        height_ratio: float,
        backbone_line_width: float,
        zorder: float = 3.0,
    ) -> None:
        super().__init__()
        self._fallback_axes = ax
        self._left = float(left)
        self._right = float(right)
        self._y = float(y)
        self._color = color
        self._height_ratio = max(0.0, float(height_ratio))
        self._backbone_line_width = max(0.1, float(backbone_line_width))
        self.set_zorder(float(zorder))
        self.set_clip_on(True)

    def draw(self, renderer) -> None:  # pragma: no cover - exercised by savefig
        if not self.get_visible() or self._right <= self._left:
            return
        ax = self.axes if self.axes is not None else self._fallback_axes
        if ax is None:
            return

        transform = ax.transData
        x0, y_center = transform.transform((self._left, self._y))
        x1, _ = transform.transform((self._right, self._y))
        if not np.isfinite([x0, x1, y_center]).all():
            return

        line_width_px = max(
            0.1,
            float(renderer.points_to_pixels(self._backbone_line_width)),
        )
        height_px = max(0.1, self._height_ratio * line_width_px)
        left = min(float(x0), float(x1))
        right = max(float(x0), float(x1))
        bottom = float(y_center) - 0.5 * height_px
        top = float(y_center) + 0.5 * height_px
        vertices = np.asarray(
            [
                (left, bottom),
                (right, bottom),
                (right, top),
                (left, top),
                (left, bottom),
            ],
            dtype=float,
        )
        codes = [
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ]
        path = MplPath(vertices, codes)
        fill = mcolors.to_rgba(self._color)
        gc = renderer.new_gc()
        try:
            gc.set_clip_rectangle(ax.bbox)
            gc.set_linewidth(0.0)
            gc.set_foreground(fill, isRGBA=True)
            renderer.draw_path(
                gc,
                path,
                IdentityTransform(),
                rgbFace=fill,
            )
        finally:
            gc.restore()


def draw_gene_structure_records(
    ax: plt.Axes,
    gene_df: pd.DataFrame,
    *,
    arrow_color: str = "black",
    triangle_color: Optional[str] = "white",
    block_color: str = "grey",
    utr_color: Optional[str] = None,
    draw_utr: bool = True,
    line_width: float = 0.5,
    arrow_step: float = 5_000.0,
    gene_text_size: float = 4.0,
    thickness_scale: float = 1.0,
    y_offset: float = 0.0,
    label_offset: float = 0.22,
    unknown_strand_as_plus: bool = True,
    label_bbox: Optional[dict[str, object]] = None,
    adaptive_lanes: bool = True,
) -> plt.Axes:
    """
    Draw gene/CDS records with projected x coordinates.

    UTR records remain available for transcript model selection, but can be
    hidden with ``draw_utr=False`` when the annotation contains many
    alternative UTR segments that make the compact track difficult to read.

    Required columns in ``gene_df``:
      feature, strand, attribute, x_start, x_end
    """
    scale = max(0.2, float(thickness_scale))
    lane_gene_df = gene_df[
        gene_df["feature"].astype(str).str.lower().eq("gene")
    ].drop_duplicates(
        subset=["strand", "x_start", "x_end"],
        keep="first",
    )
    lane_by_gene = (
        _gene_interval_lanes(lane_gene_df, gene_text_size=gene_text_size)
        if adaptive_lanes
        else {}
    )
    n_lanes = max(lane_by_gene.values(), default=-1) + 1
    # Treat each lane as one equal-height slot.  The caller controls the
    # physical height of the gene panel, so a three-lane panel gets three
    # equally sized slots instead of compressing all rows into the lower part
    # of the old fixed-gap range.  Feature thickness remains in display
    # pixels and therefore does not change the lane allocation.
    lane_step = 1.0
    lane_y = {
        idx: float(y_offset) + (n_lanes - 0.5 - lane) * lane_step
        for idx, lane in lane_by_gene.items()
    }
    y0 = float(y_offset)
    # Keep the visible feature heights tied to the backbone line width:
    # CDS : UTR : GENE = 4 : 2 : 1.  The block artist below converts these
    # ratios to display pixels at render time, so the ratio survives changes
    # in axes height, figure DPI, and genomic interval width.
    cds_height_ratio = 4.0
    utr_height_ratio = 2.0

    def _display_interval(x1: float, x2: float) -> tuple[float, float]:
        # GFF coordinates are 1-based and inclusive.  Use half-base display
        # edges so adjacent records such as UTR 707-760 and CDS 761-806 meet
        # at 760.5 instead of leaving a one-coordinate rendering seam.
        left = min(x1, x2) - 0.5
        right = max(x1, x2) + 0.5
        return left, right

    x_values = pd.to_numeric(
        pd.concat([gene_df["x_start"], gene_df["x_end"]]), errors="coerce"
    ).dropna()
    x_span = max(1.0, float(x_values.max() - x_values.min())) if not x_values.empty else 1.0
    def _add_feature(
        left: float,
        right: float,
        y: float,
        height_ratio: float,
        color: str,
    ) -> None:
        if right <= left:
            return
        ax.add_artist(
            _GeneFeatureBlockArtist(
                ax,
                left,
                right,
                y,
                color,
                height_ratio=height_ratio,
                backbone_line_width=line_width,
                # IGV renders exon/UTR boxes above the thin transcript
                # backbone.  Keeping this order also makes a touching UTR
                # and CDS read as two clean, non-overlapping feature types.
                zorder=3.0,
            )
        )

    if gene_df.shape[0] > 0:
        # Keep every transcript until its representative model has been
        # selected.  De-duplicating by coordinates first can silently remove
        # the terminal UTR/CDS of the transcript that should be displayed.
        feature_df = gene_df.copy()
        feature_df["feature"] = feature_df["feature"].astype(str).str.strip()
        genes = feature_df[
            feature_df["feature"].str.lower().eq("gene")
        ].drop_duplicates(
            subset=["strand", "x_start", "x_end"],
            keep="first",
        ).copy()

        def _attribute_values(value: object) -> list[str]:
            if isinstance(value, (list, tuple, np.ndarray)):
                values = [str(item).strip() for item in value]
            else:
                values = [str(value).strip()]
            return [
                item
                for item in values
                if item != "" and item.lower() not in {"na", "nan", "none"}
            ]

        def _strip_gff_prefix(value: object) -> str:
            text = str(value).strip()
            for prefix in ("gene:", "transcript:"):
                if text.startswith(prefix):
                    text = text[len(prefix):]
            return text

        def _gene_id(value: object) -> str:
            values = _attribute_values(value)
            return _strip_gff_prefix(values[0]) if values else ""

        def _transcript_key(value: object) -> str:
            values = _attribute_values(value)
            for item in values:
                cleaned = _strip_gff_prefix(item)
                if re.search(r"_T[0-9]+$", cleaned):
                    return cleaned
            return _strip_gff_prefix(values[0]) if values else ""

        def _transcript_gene_id(transcript: str) -> str:
            return re.sub(r"_T[0-9]+$", "", str(transcript))

        gene_ids = {
            gene_idx: _gene_id(row["attribute"])
            for gene_idx, row in genes.iterrows()
        }
        gene_indices_by_id: dict[str, list[object]] = {}
        for gene_idx, gene_id in gene_ids.items():
            if gene_id != "":
                gene_indices_by_id.setdefault(gene_id, []).append(gene_idx)

        def _feature_gene_key(row: pd.Series) -> object:
            transcript = _transcript_key(row["attribute"])
            transcript_gene = _transcript_gene_id(transcript)
            candidates = gene_indices_by_id.get(transcript_gene, [])
            if not candidates:
                candidates = list(genes.index)
            left = min(float(row["x_start"]), float(row["x_end"]))
            right = max(float(row["x_start"]), float(row["x_end"]))
            scored: list[tuple[float, float, str, object]] = []
            for gene_idx in candidates:
                gene_row = genes.loc[gene_idx]
                gene_left = min(float(gene_row["x_start"]), float(gene_row["x_end"]))
                gene_right = max(float(gene_row["x_start"]), float(gene_row["x_end"]))
                overlap = max(0.0, min(right, gene_right) - max(left, gene_left))
                width = max(0.0, gene_right - gene_left)
                if overlap > 0.0 or (left == right and gene_left <= left <= gene_right):
                    scored.append((overlap, -width, str(gene_idx), gene_idx))
            if scored:
                return max(scored)[3]
            return None

        # Convert to half-base display coordinates once.  Every later
        # operation—CDS, UTR, gene backbone, and direction symbols—uses these
        # exact edges, so a touching UTR/CDS pair has one shared boundary.
        feature_rows: list[dict[str, object]] = []
        for _, row in feature_df[
            feature_df["feature"].str.lower().isin(
                ["cds", "five_prime_utr", "three_prime_utr"]
            )
        ].iterrows():
            left, right = _display_interval(
                float(row["x_start"]), float(row["x_end"])
            )
            feature_rows.append(
                {
                    "feature": str(row["feature"]),
                    "strand": str(row["strand"]),
                    "attribute": row["attribute"],
                    "_left": left,
                    "_right": right,
                    "_gene_key": _feature_gene_key(row),
                    "_transcript_key": _transcript_key(row["attribute"]),
                }
            )

        def _merge_intervals(
            intervals: Iterable[tuple[float, float]],
        ) -> list[tuple[float, float]]:
            ordered = sorted(
                (float(left), float(right))
                for left, right in intervals
                if np.isfinite(left) and np.isfinite(right) and right > left
            )
            merged: list[list[float]] = []
            for left, right in ordered:
                if merged and left <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], right)
                else:
                    merged.append([left, right])
            return [(left, right) for left, right in merged]

        selected_rows: list[dict[str, object]] = []
        if feature_rows:
            feature_work = pd.DataFrame(feature_rows)
            transcript_scores: dict[tuple[object, str], tuple[float, float, int, str]] = {}
            for (gene_key, transcript), group in feature_work.groupby(
                ["_gene_key", "_transcript_key"], dropna=False, sort=False
            ):
                left = float(group["_left"].min())
                right = float(group["_right"].max())
                cds_width = float(
                    group.loc[group["feature"].str.lower().eq("cds"), "_right"].sub(
                        group.loc[group["feature"].str.lower().eq("cds"), "_left"]
                    ).sum()
                )
                # Prefer the transcript with the largest coding sequence.
                # A model with a very long alternative UTR can span more of
                # the locus while carrying a shorter CDS; choosing by genomic
                # span first makes those isoforms dominate the compact gene
                # panel and produces an unnecessarily fragmented UTR track.
                transcript_scores[(gene_key, str(transcript))] = (
                    cds_width,
                    right - left,
                    int(group.shape[0]),
                    str(transcript),
                )

            selected_transcripts: dict[object, str] = {}
            for gene_key in feature_work["_gene_key"].drop_duplicates().tolist():
                candidates = [
                    (score, transcript)
                    for (candidate_gene, transcript), score in transcript_scores.items()
                    if candidate_gene == gene_key
                ]
                if candidates:
                    selected_transcripts[gene_key] = max(candidates)[1]

            for row in feature_rows:
                if row["_transcript_key"] == selected_transcripts.get(row["_gene_key"]):
                    selected_rows.append(row)

        merged_feature_rows: list[dict[str, object]] = []
        if selected_rows:
            selected_work = pd.DataFrame(selected_rows)
            for (gene_key, feature, strand), group in selected_work.groupby(
                ["_gene_key", "feature", "strand"], dropna=False, sort=False
            ):
                for left, right in _merge_intervals(
                    zip(group["_left"], group["_right"])
                ):
                    merged_feature_rows.append(
                        {
                            "feature": str(feature),
                            "strand": str(strand),
                            "attribute": group.iloc[0]["attribute"],
                            "x_start": float(left),
                            "x_end": float(right),
                            "_gene_key": gene_key,
                        }
                    )

        merged_features = pd.DataFrame(merged_feature_rows)
        if merged_features.empty:
            merged_features = pd.DataFrame(
                columns=[
                    "feature",
                    "strand",
                    "attribute",
                    "x_start",
                    "x_end",
                    "_gene_key",
                ]
            )
        cds = merged_features[
            merged_features["feature"].str.lower().eq("cds")
        ]
        utr = (
            merged_features[
                merged_features["feature"].str.lower().isin(
                    ["five_prime_utr", "three_prime_utr"]
                )
            ]
            if draw_utr
            else merged_features.iloc[0:0]
        )

        # With UTRs visible, use the representative transcript's outer
        # feature edges so the backbone terminates at the outermost displayed
        # block.  When UTRs are hidden, retain the complete gene record span;
        # the direction line then still represents the annotated gene start
        # and end even when the CDS occupies only its middle section.
        gene_bounds: dict[object, tuple[float, float]] = {}
        for gene_idx, gene_row in genes.iterrows():
            gene_left, gene_right = _display_interval(
                float(gene_row["x_start"]), float(gene_row["x_end"])
            )
            model = merged_features[merged_features["_gene_key"] == gene_idx]
            if draw_utr and not model.empty:
                model_left = float(model["x_start"].min())
                model_right = float(model["x_end"].max())
                # Projected records may be clipped at a bimrange boundary;
                # never let a feature extend outside its gene record.
                model_left = max(gene_left, model_left)
                model_right = min(gene_right, model_right)
                if model_right > model_left:
                    gene_left, gene_right = model_left, model_right
            gene_bounds[gene_idx] = (gene_left, gene_right)

        # Put a white, borderless CDS-height plate behind every gene model.
        # Manhattan-to-LD connectors are drawn below this plate, so dense
        # bridge lines cannot visually cut through the gene structure while
        # the plate still leaves the gene line, UTR, CDS, and labels visible.
        for gene_idx, (gene_left, gene_right) in gene_bounds.items():
            y_gene = lane_y.get(gene_idx, y0)
            ax.add_artist(
                _GeneFeatureBlockArtist(
                    ax,
                    gene_left,
                    gene_right,
                    y_gene,
                    "white",
                    height_ratio=4.0,
                    backbone_line_width=line_width,
                    zorder=0.5,
                )
            )

        # Work in display edges, so adjacent CDS blocks remain exactly
        # touching and cannot acquire a one-base seam.
        cds_by_gene: dict[object, list[tuple[float, float]]] = {
            gene_idx: _merge_intervals(
                zip(
                    group["x_start"],
                    group["x_end"],
                )
            )
            for gene_idx, group in cds.groupby("_gene_key", dropna=False, sort=False)
        }
        utr_parts: list[dict[str, object]] = []
        if draw_utr:
            for _, row in utr.iterrows():
                gene_key = row["_gene_key"]
                left = float(row["x_start"])
                right = float(row["x_end"])
                if gene_key in gene_bounds:
                    bound_left, bound_right = gene_bounds[gene_key]
                    left = max(left, bound_left)
                    right = min(right, bound_right)
                parts = [(left, right)] if right > left else []
                for block_left, block_right in cds_by_gene.get(gene_key, []):
                    next_parts: list[tuple[float, float]] = []
                    for part_left, part_right in parts:
                        if block_right <= part_left or block_left >= part_right:
                            next_parts.append((part_left, part_right))
                            continue
                        if part_left < block_left:
                            next_parts.append((part_left, block_left))
                        if block_right < part_right:
                            next_parts.append((block_right, part_right))
                    parts = next_parts
                for part_left, part_right in parts:
                    if part_right > part_left:
                        utr_parts.append(
                            {
                                "x_start": part_left,
                                "x_end": part_right,
                                "_gene_key": gene_key,
                                "strand": row["strand"],
                            }
                        )

        def _row_y(row: pd.Series) -> float:
            gene_key = row.get("_gene_key")
            if gene_key in lane_y:
                return lane_y[gene_key]
            midpoint = 0.5 * (float(row["x_start"]) + float(row["x_end"]))
            candidates = []
            for gene_idx, gene_row in genes.iterrows():
                gleft = min(float(gene_row["x_start"]), float(gene_row["x_end"]))
                gright = max(float(gene_row["x_start"]), float(gene_row["x_end"]))
                if gleft <= midpoint <= gright:
                    candidates.append((gright - gleft, gene_idx))
            if candidates:
                return lane_y.get(min(candidates)[1], float(y_offset))
            return float(y_offset)

        if not cds.empty:
            for _, row in cds.iterrows():
                _add_feature(
                    float(row["x_start"]),
                    float(row["x_end"]),
                    _row_y(row),
                    cds_height_ratio,
                    block_color,
                )

        if utr_parts:
            utr_fill = block_color if utr_color is None else str(utr_color)
            for row in utr_parts:
                _add_feature(
                    float(row["x_start"]),
                    float(row["x_end"]),
                    _row_y(pd.Series(row)),
                    utr_height_ratio,
                    utr_fill,
                )

        if not genes.empty:
            text_bbox = (
                label_bbox
                if label_bbox is not None
                else {
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none",
                    "boxstyle": "round,pad=0.2",
                }
            )
            for _, row in genes.iterrows():
                left, right = gene_bounds.get(
                    row.name,
                    _display_interval(float(row["x_start"]), float(row["x_end"])),
                )
                strand = str(row["strand"]).strip()
                attr_values = _attribute_values(row["attribute"])
                gene_name = attr_values[0] if attr_values else ""
                gene_name = gene_name.removeprefix("gene:")
                y_gene = lane_y.get(row.name, y0)
                ax.plot(
                    [left, right],
                    [y_gene, y_gene],
                    color=arrow_color,
                    linewidth=line_width,
                    solid_capstyle="butt",
                    solid_joinstyle="miter",
                    zorder=1.0,
                )
                if gene_name.strip() != "":
                    ax.text(
                        0.5 * (left + right),
                        y_gene - float(label_offset) * scale,
                        gene_name,
                        ha="center",
                        va="center",
                        fontsize=gene_text_size,
                        bbox=text_bbox,
                        zorder=4.0,
                    )

                if strand == "-" or strand == "+" or (
                    unknown_strand_as_plus and strand not in {"+", "-"}
                ):
                    ax.add_artist(
                        _IGVDirectionChevronArtist(
                            ax,
                            [(left, right)],
                            y_gene,
                            "+" if strand != "-" else "-",
                            arrow_color if triangle_color is None else triangle_color,
                            backbone_line_width=line_width,
                            max_symbols=None,
                            zorder=2.0,
                        )
                    )

    if n_lanes > 0:
        # The gene rows occupy the full [y_offset, y_offset + n_lanes]
        # interval.  Their centers are at the middle of their respective
        # slots, leaving a small slot boundary on both sides.  postgwas adds
        # a separate bottom routing gutter for Manhattan-to-LD connectors.
        ymin = float(y_offset)
        ymax = float(y_offset) + float(n_lanes) * lane_step
    else:
        ymin = -0.15 * scale + y0
        ymax = 0.15 * scale + y0
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for k in ["right", "left", "bottom", "top"]:
        ax.spines[k].set_visible(False)
    return ax


def plot_gene_structure(
    gffpath: pathlike,
    bimrange: Union[str, Iterable[str]],
    FancyArrowPatchcolor: str = "black",
    Rectanglecolor: str = "grey",
    ax: Optional[plt.Axes] = None,
    step: int = 1000,
    gene_text_size: float = 4.0,
    line_width: float = 0.5,
) -> plt.Axes:
    """
    Plot gene structure track for one or multiple bimrange windows.

    Parameters
    ----------
    gffpath : str | Path
        GFF/GFF3 path.
    bimrange : str | iterable[str]
        One range or multiple ranges, format chr:start-end or chr:start:end (Mb).
    FancyArrowPatchcolor : str
        Arrow/line color.
    Rectanglecolor : str
        CDS/UTR block color.
    ax : matplotlib.axes.Axes | None
        Existing axis. If None, create a new axis.
    step : int
        Arrow segment step in bp.
    gene_text_size : float
        Gene label font size.
    line_width : float
        Line width for structure segments.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 1), dpi=300)

    if isinstance(bimrange, str):
        ranges = [bimrange]
    else:
        ranges = [str(x) for x in bimrange]
    ranges = [x.strip() for x in ranges if x is not None and str(x).strip() != ""]
    if len(ranges) == 0:
        raise ValueError("bimrange is empty.")
    if step <= 0:
        raise ValueError("step must be > 0.")

    q = GFFQuery.from_file(gffpath)
    features = ["gene", "five_prime_UTR", "three_prime_UTR", "CDS"]
    all_hits = []
    xmins: list[int] = []
    xmaxs: list[int] = []
    for r in ranges:
        chrom, start_bp, end_bp = _parse_bimrange_mbp(r)
        xmins.append(start_bp)
        xmaxs.append(end_bp)
        hit = q.query_range(
            chrom=chrom,
            start=start_bp,
            end=end_bp,
            features=features,
            attr="ID",
        )
        if hit.shape[0] > 0:
            all_hits.append(hit)
    if len(all_hits) == 0:
        ax.set_xlim(min(xmins), max(xmaxs))
        draw_gene_structure_records(ax, pd.DataFrame())
        return ax

    gffset = pd.concat(all_hits, axis=0, ignore_index=True)
    gene_df = gffset.loc[:, ["feature", "strand", "attribute", "start", "end"]].copy()
    gene_df["x_start"] = pd.to_numeric(gene_df["start"], errors="coerce")
    gene_df["x_end"] = pd.to_numeric(gene_df["end"], errors="coerce")
    gene_df = gene_df.dropna(subset=["x_start", "x_end"]).copy()
    gene_df["x_start"] = gene_df["x_start"].astype(float)
    gene_df["x_end"] = gene_df["x_end"].astype(float)
    gene_df = gene_df.loc[:, ["feature", "strand", "attribute", "x_start", "x_end"]]
    draw_gene_structure_records(
        ax,
        gene_df,
        arrow_color=FancyArrowPatchcolor,
        block_color=Rectanglecolor,
        line_width=line_width,
        arrow_step=float(step),
        gene_text_size=gene_text_size,
        unknown_strand_as_plus=True,
    )

    ax.set_xlim(min(xmins), max(xmaxs))
    return ax


class CHRPLOT:
    def __init__(self,gff:Union[pd.DataFrame,pathlike]):
        if isinstance(gff, (str, Path)):
            gff = Path(gff)
            assert gff.exists(),"gff file not exists"
            self.gff = self._gffreader(gff)
        else:
            self.gff = gff
        pass
    def plotchr(self,window=1_000_000,color_dict:dict=None,ax:plt.Axes=None):
        _ = []
        chr_list = self.gff[0].unique()
        for i in chr_list:
            gff_chr = self.gff[(self.gff[0]==i)&(self.gff[2]=='gene')]
            chrend = gff_chr[4].max()
            locr = list(range(0,chrend,window))
            for loc in locr:
                gene_num = gff_chr[(gff_chr[3]>=loc)&(gff_chr[4]<=loc+window)].shape[0]
                _.append([i, loc, gene_num])
        _ = pd.DataFrame(_)
        _[2] = _[2]/_[2].max()/2
        _['l'] = _[0] - _[2]
        _['r'] = _[0] + _[2]
        cd = dict(zip(chr_list,['grey' for _ in chr_list]))
        cd.update(color_dict) if color_dict is not None else None
        for i in chr_list:
            ax.fill_betweenx(_.loc[_[0]==i,1],_.loc[_[0]==i,'l'],_.loc[_[0]==i,'r'],color=cd[i])
        ax.set_xticks(chr_list.astype(int),[f'chr{i}' for  i in chr_list])
        return ax
    def _gffreader(self,gffpath:str):
        gff = gffreader(gffpath)
        chr_num = pd.to_numeric(gff["chrom_norm"], errors="coerce")
        gff = gff[chr_num.notna()].copy()
        gff.loc[:, 0] = chr_num.loc[gff.index].astype(int)
        gff.loc[:, 2] = gff["feature"].astype(str)
        gff.loc[:, 3] = gff["start"].astype(int)
        gff.loc[:, 4] = gff["end"].astype(int)
        gff = gff[gff[0].isin(np.arange(1,100).astype(int))]
        gff = gff.sort_values([0, 3])
        return gff[[0, 2, 3, 4]]


if __name__ == "__main__":
    q = GFFQuery.from_file('/Users/jingxianfu/Public/Triticum_aestivum.IWGSC.62.mapped.gff3.gz')
    print(np.array(q.query_bimrange("1:80-100", features=["gene","CDS"],attr=['ID'])['attribute'].tolist()))
