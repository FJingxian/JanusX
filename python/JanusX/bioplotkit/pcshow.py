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

    def pcplot3D(self,x:str,y:str,z:str,group:str=None,anno_tag:str=None,color_set:list=None,equal_aspect:bool=True):
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            print('It is a beta version for 3Dplot')
        except:
            raise ImportError("Please use pip install plotly or conda install plotly")
        data = self.data.reset_index().copy()
        data['size'] = 5
        if group:
            data[group] = data[group].fillna('others')
            groups = data[group].unique()
            colors = color_set if color_set is not None else self._auto_colors(len(groups))
            color = dict(zip(groups,[colors[i%len(colors)] for i in range(len(groups))]))
            color['others'] = '#DCDCDC'
            data.loc[data[group]=='others','size'] = 1
        fig = px.scatter_3d(data,
                            x=x,
                            y=y,
                            z=z,
                            hover_name=data.columns[0],
                            color=group if group is not None else None,
                            size='size',
                            opacity=0.7,
                            color_discrete_map=color if group is not None else None,)
        # Remove marker outlines to avoid white-edge overlap
        fig.update_traces(marker=dict(line=dict(width=0)))
        if group and anno_tag:
            data_anno = data.loc[~data[anno_tag].isna()]
            if not data_anno.empty:
                fig.add_trace(go.Scatter3d(
                    x=data_anno[x],
                    y=data_anno[y],
                    z=data_anno[z],
                    mode='markers+text',  # show markers and text
                    marker=dict(
                        symbol='diamond',
                        color='#39ff14',
                        size=12,
                        opacity=1.0,
                        line=dict(width=3, color='#39ff14')
                    ),
                    text=data_anno[anno_tag],  # show id text
                    textposition="top center",  # text loc
                    textfont=dict(
                        size=12,
                        color='#39ff14'
                    ),
                    name='Hilighted',
                    hoverinfo='text',
                    showlegend=True
                ))
        fig.update_layout(
        title=None,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene=dict(
            xaxis=dict(title=x, tickfont=dict(color="black"), titlefont=dict(color="black")),
            yaxis=dict(title=y, tickfont=dict(color="black"), titlefont=dict(color="black")),
            zaxis=dict(title=z, tickfont=dict(color="black"), titlefont=dict(color="black")),
            # cube aspect
            aspectmode='cube' if equal_aspect else 'auto'
        ),
        showlegend=True)
        return fig

    def pcplot3D_gif(
        self,
        x: str,
        y: str,
        z: str,
        group: str | None = None,
        anno_tag: str | None = None,
        color_set: list | None = None,
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
