from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import animation
from .sci_set import marker_set

class PCSHOW:
    def __init__(self,data:pd.DataFrame,):
        self.data = data.replace('others',pd.NA)
        self.color = None
        self.marker = None
        pass

    def _auto_colors(self, n: int) -> list[str]:
        if n <= 10:
            cmap = plt.get_cmap("tab10")
        elif n <= 20:
            cmap = plt.get_cmap("tab20")
        else:
            cmap = plt.get_cmap("turbo")
        return [mcolors.to_hex(cmap(i / max(1, n - 1))) for i in range(n)]

    def pcplot(self,x:str, y:str,group:str=None,group_order:list=None,color_set:list=None,marker_set:list=marker_set,anno_tag:str=None,ax:plt.Axes=None,**kwargs):
        ax = ax if ax is not None else plt.gca()
        if group is not None:
            groupmask = self.data[group].isna()
            groups = self.data.loc[~groupmask,group].unique() if group_order is None else group_order
            colors = color_set if color_set is not None else self._auto_colors(len(groups))
            color = dict(zip(groups,[colors[i%len(colors)] for i in range(len(groups))]))
            marker = dict(zip(groups,[marker_set[i%len(marker_set)] for i in range(len(groups))]))
            self.color = color
            self.marker = marker
            for g in groups:
                data_ = self.data[self.data[group]==g]
                ax.scatter(x=data_[x],y=data_[y],s=32,alpha=.8,marker=marker[g],color=color[g],label=g,**kwargs)
            if np.sum(groupmask)>0:
                data_ = self.data.loc[groupmask]
                ax.scatter(x=data_[x],y=data_[y],s=8,alpha=.4,marker='*',color='grey',label='others',**kwargs)
            ax.legend()
        else:
            data = self.data
            ax.scatter(x=data[x],y=data[y],**kwargs)
        if anno_tag:
            data = self.data[[x, y, anno_tag]].dropna()
            if not data.empty:
                for i in data.index:
                    ax.text(data.loc[i,x],data.loc[i,y],data.loc[i,anno_tag],ha='center',**kwargs)


    def pcplot3D_gif(
        self,
        x: str,
        y: str,
        z: str,
        group: Union[str,None] = None,
        anno_tag: Union[str,None] = None,
        color_set: Union[list,None] = None,
        out_gif: str = "pcshow_3d.gif",
        elev: float = 20,
        azim_start: float = 0,
        azim_end: float = 360,
        frames: int = 72,
        interval: int = 200,
        dpi: int = 150,
        s: float = 32,
        alpha: float = 0.85,
    ) -> None:
        """
        Save a rotating 3D scatter plot as a GIF.
        """
        fig = plt.figure(figsize=(6, 6), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        data = self.data.copy()
        if group is not None:
            data[group] = data[group].fillna("others")
            groups = data[group].unique()
            colors = color_set if color_set is not None else self._auto_colors(len(groups))
            color_map = dict(zip(groups, [colors[i % len(colors)] for i in range(len(groups))]))
            for g in groups:
                data_g = data[data[group] == g]
                ax.scatter(
                    data_g[x],
                    data_g[y],
                    data_g[z],
                    s=s,
                    alpha=alpha if g != "others" else 0.4,
                    color=color_map.get(g, "#999999"),
                    label=g,
                )
            ax.legend(loc="best")
        else:
            ax.scatter(data[x], data[y], data[z], s=s, alpha=alpha)

        if anno_tag:
            data_anno = data[[x, y, z, anno_tag]].dropna()
            for i in data_anno.index:
                ax.text(
                    data_anno.loc[i, x],
                    data_anno.loc[i, y],
                    data_anno.loc[i, z],
                    data_anno.loc[i, anno_tag],
                    ha="center",
                )

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)

        def update(frame):
            azim = azim_start + (azim_end - azim_start) * frame / max(1, frames - 1)
            ax.view_init(elev=elev, azim=azim)
            return (ax,)

        anim = animation.FuncAnimation(
            fig, update, frames=frames, interval=interval, blit=False
        )
        plt.tight_layout()
        anim.save(out_gif, writer=animation.PillowWriter(fps=1000 / interval))
        plt.close(fig)
