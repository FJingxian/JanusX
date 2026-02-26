from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from tqdm import trange
from matplotlib.patches import FancyArrowPatch, Rectangle
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


def draw_gene_structure_records(
    ax: plt.Axes,
    gene_df: pd.DataFrame,
    *,
    arrow_color: str = "black",
    block_color: str = "grey",
    line_width: float = 0.5,
    arrow_step: float = 1_000.0,
    gene_text_size: float = 4.0,
    thickness_scale: float = 1.0,
    y_offset: float = 0.0,
    unknown_strand_as_plus: bool = True,
    label_bbox: Optional[dict[str, object]] = None,
) -> plt.Axes:
    """
    Draw gene/CDS/UTR records with projected x coordinates.

    Required columns in ``gene_df``:
      feature, strand, attribute, x_start, x_end
    """
    scale = max(0.2, float(thickness_scale))
    y0 = float(y_offset)
    cds_bottom = -0.1 * scale + y0
    cds_height = 0.2 * scale
    utr_bottom = -0.05 * scale + y0
    utr_height = 0.1 * scale

    if gene_df.shape[0] > 0:
        cds = gene_df[gene_df["feature"] == "CDS"]
        utr = gene_df[
            (gene_df["feature"] == "five_prime_UTR")
            | (gene_df["feature"] == "three_prime_UTR")
        ]
        genes = gene_df[gene_df["feature"] == "gene"]

        if not cds.empty:
            for _, r in cds.iterrows():
                x1 = float(r["x_start"])
                x2 = float(r["x_end"])
                left = min(x1, x2)
                right = max(x1, x2)
                rect = Rectangle(
                    (left, cds_bottom),
                    width=max(0.0, right - left),
                    height=cds_height,
                    color=block_color,
                    linewidth=0.0,
                )
                ax.add_patch(rect)

        if not utr.empty:
            for _, r in utr.iterrows():
                x1 = float(r["x_start"])
                x2 = float(r["x_end"])
                left = min(x1, x2)
                right = max(x1, x2)
                rect = Rectangle(
                    (left, utr_bottom),
                    width=max(0.0, right - left),
                    height=utr_height,
                    color=block_color,
                    linewidth=0.0,
                )
                ax.add_patch(rect)

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
            step = max(1.0, float(arrow_step))
            for _, r in genes.iterrows():
                x1 = float(r["x_start"])
                x2 = float(r["x_end"])
                left = min(x1, x2)
                right = max(x1, x2)
                strand = str(r["strand"]).strip()
                attr = r["attribute"]
                if isinstance(attr, (list, tuple, np.ndarray)) and len(attr) > 0:
                    gene_name = str(attr[0])
                else:
                    gene_name = str(attr)

                ax.plot([left, right], [y0, y0], color=arrow_color, linewidth=line_width)
                ax.plot(
                    [left, left],
                    [-0.05 * scale + y0, 0.05 * scale + y0],
                    color=arrow_color,
                    linewidth=line_width,
                )
                ax.plot(
                    [right, right],
                    [-0.05 * scale + y0, 0.05 * scale + y0],
                    color=arrow_color,
                    linewidth=line_width,
                )
                ax.text(
                    0.5 * (left + right),
                    y0,
                    gene_name,
                    ha="center",
                    va="center",
                    fontsize=gene_text_size,
                    bbox=text_bbox,
                )

                if strand == "-":
                    for seg_end in np.arange(right, left, -step):
                        seg_start = max(seg_end - step, left)
                        arrow = FancyArrowPatch(
                            (seg_end, y0),
                            (seg_start, y0),
                            arrowstyle="->",
                            mutation_scale=10,
                            linewidth=line_width,
                            color=arrow_color,
                        )
                        ax.add_patch(arrow)
                elif strand == "+" or (
                    unknown_strand_as_plus and strand not in {"+", "-"}
                ):
                    for seg_start in np.arange(left, right, step):
                        seg_end = min(seg_start + step, right)
                        arrow = FancyArrowPatch(
                            (seg_start, y0),
                            (seg_end, y0),
                            arrowstyle="->",
                            mutation_scale=10,
                            linewidth=line_width,
                            color=arrow_color,
                        )
                        ax.add_patch(arrow)

    ax.set_ylim(-0.15 * scale + y0, 0.15 * scale + y0)
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
