from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _select_palette(k: int) -> list[Any]:
    kk = max(1, int(k))
    if kk <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i % 10) for i in range(kk)]
    if kk <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i % 20) for i in range(kk)]
    cmap = plt.get_cmap("hsv", kk)
    return [cmap(i) for i in range(kk)]


def _stable_unique(items: Sequence[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for x in items:
        s = str(x)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def plot_admixture_structure(
    qmatrix: np.ndarray,
    *,
    sample_ids: Optional[Sequence[str]] = None,
    out_path: Optional[str] = None,
    threshold: float = 0.7,
    tag_samples: Optional[Sequence[str]] = None,
    show_xticks: bool = False,
    figsize: tuple[float, float] = (10.0, 3.0),
    dpi: int = 300,
    ai_editable: bool = True,
) -> dict[str, Any]:
    """
    Plot ADMIXTURE-like structure barplot from Q matrix.

    Parameters
    ----------
    qmatrix
        Array of shape (n_samples, K).
    sample_ids
        Optional sample IDs. If not provided, S1..Sn are used.
    out_path
        Optional output image path. If provided, figure is saved and closed.
    threshold
        Samples with max(Q) < threshold are treated as "unassigned" for ordering.
    tag_samples
        Optional sample IDs to annotate on x-axis.
    show_xticks
        Whether to show x ticks. Default False. If tag_samples is provided and
        non-empty, tagged samples are shown regardless of this flag.
    figsize, dpi
        Figure style.

    Returns
    -------
    dict
        Contains plotting metadata and matching details for tagged samples.
    """
    arr = np.asarray(qmatrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"qmatrix must be 2D, got shape={arr.shape}")
    n, k = int(arr.shape[0]), int(arr.shape[1])
    if n <= 0 or k <= 0:
        raise ValueError(f"qmatrix must be non-empty, got shape={arr.shape}")

    if sample_ids is None:
        sid = [f"S{i + 1}" for i in range(n)]
    else:
        sid = [str(x) for x in sample_ids]
        if len(sid) != n:
            raise ValueError(
                f"sample_ids length mismatch: ids={len(sid)}, rows={n}"
            )

    thr = float(threshold)
    dominant = np.asarray(np.argmax(arr, axis=1), dtype=np.int64)
    pop = np.asarray(dominant, dtype=np.int64)
    mx = np.asarray(np.max(arr, axis=1), dtype=np.float64)
    pop[mx < thr] = -1

    ordered_idx: list[int] = []
    boundaries: list[int] = [0]
    for cluster in range(k):
        idx_main = np.flatnonzero(pop == int(cluster))
        if idx_main.size > 0:
            order_main = np.argsort(arr[idx_main, cluster])[::-1]
            ordered_idx.extend([int(x) for x in idx_main[order_main].tolist()])

        # Mixed samples (max Q < threshold) are merged back by dominant cluster
        # so each cluster keeps visual continuity instead of being split into two blocks.
        idx_mixed = np.flatnonzero((pop < 0) & (dominant == int(cluster)))
        if idx_mixed.size > 0:
            order_mixed = np.argsort(arr[idx_mixed, cluster])[::-1]
            ordered_idx.extend([int(x) for x in idx_mixed[order_mixed].tolist()])

        boundaries.append(int(len(ordered_idx)))

    if len(ordered_idx) != n:
        missing = [i for i in range(n) if i not in set(ordered_idx)]
        ordered_idx.extend(missing)

    ordered = np.asarray(ordered_idx, dtype=np.int64)
    q_ord = np.asarray(arr[ordered, :], dtype=np.float64)
    sid_ord = [sid[int(i)] for i in ordered.tolist()]
    dom_ord = np.asarray(dominant[ordered], dtype=np.int64)

    colors = _select_palette(k)

    rc_ctx = {}
    if bool(ai_editable):
        rc_ctx = {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": False,
        }

    with plt.rc_context(rc=rc_ctx):
        fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
        x = np.arange(n, dtype=np.int64)
        bottom = np.zeros(n, dtype=np.float64)
        # 1) dominant component at the bottom for each sample
        for cluster in range(k):
            idx = np.flatnonzero(dom_ord == int(cluster))
            if idx.size <= 0:
                continue
            y = np.asarray(q_ord[idx, int(cluster)], dtype=np.float64)
            ax.bar(
                x[idx],
                y,
                bottom=bottom[idx],
                width=1.1,
                color=colors[int(cluster)],
                edgecolor="none",
                align="center",
            )
            bottom[idx] += y
        # 2) remaining components stacked in default order
        for cluster in range(k):
            idx = np.flatnonzero(dom_ord != int(cluster))
            if idx.size <= 0:
                continue
            y = np.asarray(q_ord[idx, int(cluster)], dtype=np.float64)
            ax.bar(
                x[idx],
                y,
                bottom=bottom[idx],
                width=1.1,
                color=colors[int(cluster)],
                edgecolor="none",
                align="center",
            )
            bottom[idx] += y

        for b in boundaries:
            if int(b) <= 0 or int(b) >= n:
                continue
            ax.vlines(
                int(b) - 0.5,
                0.0,
                1.0,
                colors="black",
                linewidth=0.8,
            )

        tag_list = _stable_unique([str(x).strip() for x in (tag_samples or []) if str(x).strip() != ""])
        pos_map = {name: i for i, name in enumerate(sid_ord)}
        matched = [x for x in tag_list if x in pos_map]
        missing_tags = [x for x in tag_list if x not in pos_map]

        if len(matched) > 0:
            ticks = [int(pos_map[x]) for x in matched]
            labels = [str(x) for x in matched]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
        elif bool(show_xticks):
            ax.set_xticks(x)
            ax.set_xticklabels(sid_ord, rotation=90, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        ax.set_xlim(-0.5, float(n) - 0.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Ancestry proportion")
        ax.set_xlabel("")
        ax.grid(False)
        for spine in ("top", "right"):
            ax.spines[str(spine)].set_visible(False)
        fig.tight_layout()

        if out_path is not None and str(out_path).strip() != "":
            out = str(out_path)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
    return {
        "n_samples": int(n),
        "k": int(k),
        "palette": ("tab10" if k <= 10 else ("tab20" if k <= 20 else "hsv")),
        "mixed_n": int(np.sum(pop < 0)),
        "ordered_sample_ids": sid_ord,
        "tag_requested": tag_list,
        "tag_matched": matched,
        "tag_missing": missing_tags,
        "out_path": (None if out_path is None else str(out_path)),
    }
