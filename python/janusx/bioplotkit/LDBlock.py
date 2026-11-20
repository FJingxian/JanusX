import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Polygon
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import Affine2D
from dataclasses import dataclass
from typing import Optional, Sequence

from janusx.gfreader.gfreader import load_genotype_chunks


def _rgb_luminance(color: object) -> float:
    r, g, b = mcolors.to_rgb(color)
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


@dataclass(frozen=True)
class LDblockDisplay:
    """Display-only LD representation with a reversible SNP-to-bin map."""

    matrix: np.ndarray
    site_to_bin: np.ndarray
    groups: tuple[tuple[int, int], ...]
    block_intervals: tuple[tuple[int, int], ...] = ()


def _normalise_segment_spans(
    n_sites: int,
    segment_spans: Optional[Sequence[tuple[int, int]]],
) -> list[tuple[int, int]]:
    """Return non-overlapping half-open spans covering all matrix rows."""
    n = max(0, int(n_sites))
    if n == 0:
        return []
    if segment_spans is None:
        return [(0, n)]

    raw_spans: list[tuple[int, int]] = []
    for start, stop in segment_spans:
        lo = max(0, min(n, int(start)))
        hi = max(0, min(n, int(stop)))
        if hi > lo:
            raw_spans.append((lo, hi))
    if len(raw_spans) == 0:
        return [(0, n)]

    raw_spans.sort()
    spans: list[tuple[int, int]] = []
    cursor = 0
    for lo, hi in raw_spans:
        if lo > cursor:
            spans.append((cursor, lo))
        if lo < cursor:
            raise ValueError("LD display segment spans overlap")
        spans.append((lo, hi))
        cursor = hi
    if cursor < n:
        spans.append((cursor, n))
    return spans


def _allocate_segment_bin_counts(
    spans: Sequence[tuple[int, int]],
    target_bins: int,
) -> list[int]:
    lengths = np.asarray([max(1, int(stop) - int(start)) for start, stop in spans], dtype=float)
    n_segments = int(len(lengths))
    if n_segments == 0:
        return []
    target = max(n_segments, min(int(target_bins), int(np.sum(lengths))))
    if target <= n_segments:
        return [1] * n_segments

    ideal = lengths * float(target) / float(np.sum(lengths))
    counts = np.floor(ideal).astype(np.int64)
    counts = np.maximum(counts, 1)
    remainder = int(target - int(np.sum(counts)))
    fractions = ideal - np.floor(ideal)
    if remainder > 0:
        order = np.argsort(-fractions, kind="stable")
        for index in order[:remainder].tolist():
            counts[int(index)] += 1
    elif remainder < 0:
        order = np.argsort(fractions, kind="stable")
        for index in order.tolist():
            if remainder >= 0:
                break
            if counts[int(index)] > 1:
                counts[int(index)] -= 1
                remainder += 1
    return [int(value) for value in counts.tolist()]


def _balanced_groups(
    start: int,
    stop: int,
    n_groups: int,
) -> list[tuple[int, int]]:
    length = max(0, int(stop) - int(start))
    groups = max(1, min(int(n_groups), length))
    base, remainder = divmod(length, groups)
    out: list[tuple[int, int]] = []
    cursor = int(start)
    for index in range(groups):
        width = base + (1 if index < remainder else 0)
        out.append((cursor, cursor + width))
        cursor += width
    return out


def _reduce_ld_values(values: np.ndarray, reducer: str) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    method = str(reducer).strip().lower()
    if method in {"mean", "average", "avg"}:
        return float(np.mean(finite))
    if method == "median":
        return float(np.median(finite))
    if method == "max":
        return float(np.max(finite))
    raise ValueError(
        "Unsupported LD display reducer {!r}; use mean, median, or max.".format(reducer)
    )


def find_ldblock_intervals(
    C: np.ndarray,
    *,
    segment_spans: Optional[Sequence[tuple[int, int]]] = None,
    r2_threshold: float = 0.8,
    min_bins: int = 4,
) -> tuple[tuple[int, int], ...]:
    """Find contiguous high-LD intervals for inverted-triangle outlines.

    The detector uses high adjacent-bin LD as the block backbone, then checks
    that the complete lower-triangle portion of each candidate retains a
    substantial high-LD fraction. Returned intervals are half-open display
    bin ranges and never cross a supplied segment boundary.
    """
    C_arr = np.asarray(np.ma.filled(np.ma.asarray(C), np.nan), dtype=float)
    if C_arr.ndim != 2 or C_arr.shape[0] != C_arr.shape[1]:
        raise ValueError(f"find_ldblock_intervals expects a square matrix, got shape={C_arr.shape}")
    n = int(C_arr.shape[0])
    if n < 2:
        return ()
    threshold = float(r2_threshold)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("LD block r2 threshold must be finite and within [0, 1].")
    minimum = max(2, int(min_bins))
    spans = _normalise_segment_spans(n, segment_spans)
    intervals: list[tuple[int, int]] = []

    for seg_start, seg_stop in spans:
        length = int(seg_stop - seg_start)
        if length < minimum:
            continue
        local = C_arr[seg_start:seg_stop, seg_start:seg_stop]
        adjacent_values = np.diag(local, k=-1)
        adjacent = np.isfinite(adjacent_values) & (adjacent_values >= threshold)
        run_start: Optional[int] = None
        for edge_index, is_high in enumerate(adjacent.tolist()):
            # Only the immediately adjacent lower-triangle cell is used as
            # the backbone; this avoids drawing a triangle around scattered
            # high-LD islands.
            high = bool(is_high)
            if high and run_start is None:
                run_start = int(edge_index)
            is_last = edge_index == int(adjacent.shape[0]) - 1
            if high and not is_last:
                continue
            if run_start is None:
                continue
            run_stop = int(edge_index + (1 if high else 0))
            start = int(run_start)
            stop = int(run_stop + 1)
            run_start = None
            if stop - start < minimum:
                continue
            lower = local[start:stop, start:stop]
            lower_values = lower[np.tril_indices(stop - start, k=-1)]
            finite = lower_values[np.isfinite(lower_values)]
            if finite.size == 0:
                continue
            high_fraction = float(np.mean(finite >= threshold))
            mean_ld = float(np.mean(finite))
            if high_fraction >= 0.55 and mean_ld >= max(0.0, threshold - 0.12):
                intervals.append((seg_start + start, seg_start + stop))

    if len(intervals) == 0:
        return ()
    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, stop in intervals[1:]:
        old_start, old_stop = merged[-1]
        if start <= old_stop:
            merged[-1] = (old_start, max(old_stop, stop))
        else:
            merged.append((start, stop))
    return tuple(merged)


def prepare_ldblock_display(
    C: np.ndarray,
    *,
    target_bins: Optional[int] = 128,
    segment_spans: Optional[Sequence[tuple[int, int]]] = None,
    reducer: str = "mean",
    block_r2_threshold: Optional[float] = 0.8,
    block_min_bins: int = 4,
) -> LDblockDisplay:
    """Aggregate a raw LD matrix for a readable triangular display.

    ``segment_spans`` uses raw matrix row indices and is kept as a hard
    boundary, so two separated ``bimrange`` intervals are never merged into
    the same display bin.  ``site_to_bin`` maps every raw SNP row to its
    display bin and lets callers keep bridge lines and labels tied to the
    original SNP coordinates.
    """
    C_arr = np.asarray(np.ma.filled(np.ma.asarray(C), np.nan), dtype=float)
    if C_arr.ndim != 2 or C_arr.shape[0] != C_arr.shape[1]:
        raise ValueError(f"prepare_ldblock_display expects a square matrix, got shape={C_arr.shape}")
    n = int(C_arr.shape[0])
    if n == 0:
        return LDblockDisplay(
            matrix=np.zeros((0, 0), dtype=np.float32),
            site_to_bin=np.zeros((0,), dtype=np.int64),
            groups=(),
            block_intervals=(),
        )

    spans = _normalise_segment_spans(n, segment_spans)
    requested = n if target_bins is None or int(target_bins) <= 0 else int(target_bins)
    if n <= requested:
        groups = [(index, index + 1) for index in range(n)]
    else:
        counts = _allocate_segment_bin_counts(spans, requested)
        groups = []
        for (start, stop), count in zip(spans, counts):
            groups.extend(_balanced_groups(start, stop, count))

    site_to_bin = np.empty((n,), dtype=np.int64)
    for bin_index, (start, stop) in enumerate(groups):
        site_to_bin[int(start) : int(stop)] = int(bin_index)

    n_bins = int(len(groups))
    display = np.zeros((n_bins, n_bins), dtype=np.float32)
    for row_index, (row_start, row_stop) in enumerate(groups):
        for col_index in range(row_index + 1):
            col_start, col_stop = groups[col_index]
            block = C_arr[row_start:row_stop, col_start:col_stop]
            if row_index == col_index and block.shape[0] > 1:
                block = block[~np.eye(block.shape[0], dtype=bool)]
            value = _reduce_ld_values(block, reducer)
            display[row_index, col_index] = np.float32(value)
            if row_index != col_index:
                display[col_index, row_index] = np.float32(value)

    display_spans: list[tuple[int, int]] = []
    for start, stop in spans:
        display_start = int(site_to_bin[int(start)])
        display_stop = int(site_to_bin[int(stop) - 1]) + 1
        if display_stop > display_start:
            display_spans.append((display_start, display_stop))
    block_intervals = ()
    if block_r2_threshold is not None:
        block_intervals = find_ldblock_intervals(
            display,
            segment_spans=display_spans or None,
            r2_threshold=float(block_r2_threshold),
            min_bins=int(block_min_bins),
        )

    return LDblockDisplay(
        matrix=display,
        site_to_bin=site_to_bin,
        groups=tuple((int(start), int(stop)) for start, stop in groups),
        block_intervals=block_intervals,
    )


def LDblock(
    C: np.ndarray,
    ax: plt.Axes,
    vmin: float = 0,
    vmax: float = 1,
    cmap: str = "Greys",
    cbar_title: str = "LD (r-square)",
    rasterized: Optional[bool] = None,
    rasterize_threshold: Optional[int] = None,
    block_intervals: Optional[Sequence[tuple[int, int]]] = None,
    block_color: object = "black",
    block_linewidth: float = 0.8,
):
    '''
    ax.set_xlim(0,n), 0.5 ~ n-0.5 n points, n=C.shape[0]=C.shape[1]
    '''
    C_arr = np.asarray(C)
    if C_arr.ndim != 2 or C_arr.shape[0] != C_arr.shape[1]:
        raise ValueError(f"LDblock expects a square 2D matrix, got shape={C_arr.shape}")
    n = C_arr.shape[0]
    if rasterized is None:
        do_rasterized = False
        if rasterize_threshold is not None:
            do_rasterized = bool(n > int(rasterize_threshold))
    else:
        do_rasterized = bool(rasterized)
    mask = np.triu(np.ones((n, n), dtype=bool), k=0)  # 不包括对角线的上三角
    C_plot = np.ma.array(C_arr, mask=mask, copy=False)
    # 绘制
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    # Enforce "light -> dark" mapping so 0 is light and 1 is dark.
    try:
        c0 = cmap_obj(0.0)
        c1 = cmap_obj(1.0)
        if _rgb_luminance(c0) < _rgb_luminance(c1):
            cmap_obj = cmap_obj.reversed()
    except Exception:
        pass
    # Use an affine-transformed image instead of building an (n+1)^2 coordinate mesh.
    # This keeps the diamond LD geometry but avoids the large temporary grid allocation.
    cmap_obj = cmap_obj.with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    image_transform = Affine2D.from_values(0.5, 1.0, 0.5, -1.0, 0.0, 0.0) + ax.transData
    ax.imshow(
        C_plot,
        origin="lower",
        extent=(0.0, float(n), 0.0, float(n)),
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        rasterized=do_rasterized,
        interpolation="nearest",
        transform=image_transform,
    )
    if block_intervals:
        for start, stop in block_intervals:
            lo = max(0, min(n, int(start)))
            hi = max(0, min(n, int(stop)))
            if hi - lo < 2:
                continue
            width = float(hi - lo)
            vertices = [
                (float(lo), 0.0),
                (float(hi), 0.0),
                (float(lo) + 0.5 * width, -width),
            ]
            ax.add_patch(
                Polygon(
                    vertices,
                    closed=True,
                    fill=False,
                    edgecolor=block_color,
                    linewidth=float(block_linewidth),
                    alpha=0.9,
                    joinstyle="round",
                    capstyle="round",
                    clip_on=True,
                    zorder=12,
                )
            )
    # Discrete 10-step vertical colorbar in a dedicated inset axis.
    # This avoids clipping/transform issues and guarantees visible tick labels.
    _ = cbar_title  # keep arg for API compatibility; do not render title text.
    edges = np.round(np.arange(vmin, vmax + 1e-9, 0.1), 10)
    if edges.size < 2:
        edges = np.array([vmin, vmax], dtype=float)
    elif not np.isclose(edges[-1], vmax):
        edges = np.append(edges, float(vmax))

    label_ticks = np.round(np.arange(vmin, vmax + 1e-9, 0.2), 10)
    if label_ticks.size == 0:
        label_ticks = np.array([vmin, vmax], dtype=float)
    elif not np.isclose(label_ticks[-1], vmax):
        label_ticks = np.append(label_ticks, float(vmax))

    denom = max(float(vmax) - float(vmin), 1e-12)
    mids = 0.5 * (edges[:-1] + edges[1:])
    seg_colors = [cmap_obj((float(m) - float(vmin)) / denom) for m in mids]
    if len(seg_colors) == 0:
        seg_colors = [cmap_obj(0.0)]
    cmap_disc = mcolors.ListedColormap(seg_colors)
    norm_disc = mcolors.BoundaryNorm(edges, ncolors=max(1, cmap_disc.N), clip=True)

    # Move both cbar and ctick labels to the LEFT side.
    cax = ax.inset_axes([0.035, 0.02, 0.02, 0.45], transform=ax.transAxes)
    cb = ColorbarBase(
        cax,
        cmap=cmap_disc,
        norm=norm_disc,
        boundaries=edges,
        ticks=label_ticks,
        spacing="proportional",
        orientation="vertical",
        drawedges=False,
    )
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.tick_params(
        axis="y",
        which="both",
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
        labelsize=7,
        length=2.0,
        width=0.5,
        pad=1.0,
        colors="black",
    )
    cb.set_ticklabels([f"{float(v):g}" for v in label_ticks])
    # Keep 0.1 tick marks as minor ticks, but label only 0.2 steps.
    minor_ticks = [float(v) for v in edges if not np.any(np.isclose(v, label_ticks))]
    cb.ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))
    cb.ax.tick_params(
        axis="y",
        which="minor",
        left=True,
        right=False,
        length=1.5,
        width=0.4,
        colors="black",
    )
    for t in cb.ax.get_yticklabels():
        t.set_color("black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-n,0)
    ax.set_xlim(0.5,n-0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_aspect(0.5, adjustable="box")  # 对应当前 t 的 x 压缩
    ax.set_anchor("N")

    
if __name__ == '__main__':
    chunks = load_genotype_chunks('example/mouse_hs1940.vcf.gz')
    for chunk,site in chunks:
        pass
    cor = np.corrcoef(chunk[:1000])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    LDblock(cor,ax)
    plt.show()
