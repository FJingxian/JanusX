import numpy as np
import matplotlib.patches as mpathes
import itertools
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import FixedLocator

from janusx.gfreader.gfreader import load_genotype_chunks


def _rgb_luminance(color: object) -> float:
    r, g, b = mcolors.to_rgb(color)
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)

def LDblock(
    C: np.ndarray,
    ax: plt.Axes,
    vmin: float = 0,
    vmax: float = 1,
    cmap: str = "Greys",
    cbar_title: str = "LD (r-square)",
):
    '''
    ax.set_xlim(0,n), 0.5 ~ n-0.5 n points, n=C.shape[0]=C.shape[1]
    '''
    mask = np.triu(np.ones_like(C, dtype=bool), k=0) # 不包括对角线的上三角
    C[mask] = np.nan
    n = C.shape[0]
    # 创建旋转/缩放矩阵 (45度旋转伴随缩放)
    t = np.array([[1, 0.5], [-1, 0.5]])
    # 创建坐标矩阵并变换它
    A = np.dot(np.array([(i[1], i[0]) for i in itertools.product(range(n,-1,-1), range(0,n+1,1))]), t)
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
    ax.pcolormesh(
        A[:, 1].reshape(n + 1, n + 1),
        A[:, 0].reshape(n + 1, n + 1),
        np.flipud(C),
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
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
        drawedges=True,
    )
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("black")
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
