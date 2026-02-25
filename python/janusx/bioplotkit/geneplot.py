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
        ax.set_ylim(-0.15, 0.15)
        ax.set_xticks([])
        ax.set_yticks([])
        for k in ["right", "left", "bottom", "top"]:
            ax.spines[k].set_visible(False)
        return ax

    gffset = pd.concat(all_hits, axis=0, ignore_index=True)
    gffgene = gffset[gffset["feature"] == "gene"]
    gffcds = gffset[gffset["feature"] == "CDS"]
    gffutr = gffset[
        (gffset["feature"] == "five_prime_UTR")
        | (gffset["feature"] == "three_prime_UTR")
    ]

    if not gffcds.empty:
        for i in gffcds.index:
            start = int(gffcds.loc[i, "start"])
            end = int(gffcds.loc[i, "end"])
            rect = Rectangle(
                (start, -0.1),
                width=np.abs(end - start),
                height=0.2,
                color=Rectanglecolor,
            )
            ax.add_patch(rect)

    if not gffutr.empty:
        for i in gffutr.index:
            start = int(gffutr.loc[i, "start"])
            end = int(gffutr.loc[i, "end"])
            rect = Rectangle(
                (start, -0.05),
                width=np.abs(end - start),
                height=0.1,
                color=Rectanglecolor,
            )
            ax.add_patch(rect)

    if not gffgene.empty:
        for i in gffgene.index:
            start = int(gffgene.loc[i, "start"])
            end = int(gffgene.loc[i, "end"])
            gene_left = min(start, end)
            gene_right = max(start, end)
            strand = str(gffgene.loc[i, "strand"])

            attr = gffgene.loc[i, "attribute"]
            if isinstance(attr, (list, tuple, np.ndarray)) and len(attr) > 0:
                genename = str(attr[0])
            else:
                genename = str(attr)

            ax.plot([gene_left, gene_right], [0, 0], color=FancyArrowPatchcolor, linewidth=line_width)
            ax.plot([gene_left, gene_left], [-0.05, 0.05], color=FancyArrowPatchcolor, linewidth=line_width)
            ax.plot([gene_right, gene_right], [-0.05, 0.05], color=FancyArrowPatchcolor, linewidth=line_width)
            ax.text(
                (gene_left + gene_right) / 2,
                0,
                genename,
                ha="center",
                va="center",
                fontsize=gene_text_size,
                bbox=dict(
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                ),
            )

            if strand == "-":
                for seg_end in np.arange(gene_right, gene_left, -step):
                    seg_start = max(seg_end - step, gene_left)
                    arrow = FancyArrowPatch(
                        (seg_end, 0),
                        (seg_start, 0),
                        arrowstyle="->",
                        mutation_scale=10,
                        linewidth=line_width,
                        color=FancyArrowPatchcolor,
                    )
                    ax.add_patch(arrow)
            else:
                for seg_start in np.arange(gene_left, gene_right, step):
                    seg_end = min(seg_start + step, gene_right)
                    arrow = FancyArrowPatch(
                        (seg_start, 0),
                        (seg_end, 0),
                        arrowstyle="->",
                        mutation_scale=10,
                        linewidth=line_width,
                        color=FancyArrowPatchcolor,
                    )
                    ax.add_patch(arrow)

    ax.set_xlim(min(xmins), max(xmaxs))
    ax.set_ylim(-0.15, 0.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
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
