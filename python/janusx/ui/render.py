from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
import re
import gzip
import time
import hashlib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib as mpl
import numpy as np
import pandas as pd
from janusx.bioplotkit.LDBlock import LDblock
from janusx.bioplotkit.manhanden import GWASPLOT
from janusx.gfreader import load_genotype_chunks, inspect_genotype_file
from janusx.gtools.reader import readanno
from janusx.gtools.cleaner import chrom_sort_key as _chrom_sort_key
from ..script._common.pathcheck import safe_expanduser, safe_resolve
try:
    from janusx.janusx import load_gwas_triplet_fast as _load_gwas_triplet_fast
except Exception:
    _load_gwas_triplet_fast = None

# Keep WebUI rendering headless/stable on server side.
mpl.use("Agg")
# Keep vectors/text editable in PDF/SVG exports (AI friendly).
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyArrowPatch


_CHROM_CANDIDATES = [
    "#CHROM",
    "chrom",
    "chr",
    "chromosome",
    "chromosome_id",
    "chrom_id",
    "chrom_name",
    "chr_id",
    "seqid",
    "seqname",
    "contig",
    "scaffold",
]
_POS_CANDIDATES = [
    "POS",
    "pos",
    "position",
    "bp",
    "ps",
    "base_pair",
    "basepair",
    "bp_position",
    "physical_position",
    "physical_pos",
    "coordinate",
    "coord",
    "location",
    "loc",
]
_P_CANDIDATES = [
    "p",
    "pvalue",
    "p_value",
    "pval",
    "p_val",
    "pwald",
    "p_wald",
    "wald_p",
    "waldp",
    "p_lrt",
    "lrt_p",
    "prob",
]
_LD_CACHE: dict[str, tuple[np.ndarray, list[tuple[str, int]]]] = {}
_META_CACHE: dict[str, int] = {}
_ANNO_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_SIG_GWAS_CACHE: dict[str, tuple[float, pd.DataFrame, dict[str, str]]] = {}
_TRACK_ANNO_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_ACTIVE_TASK_CACHE: dict[str, str] = {"gwas": "", "anno": ""}
_LARGE_MATRIX_ROWS = 1_000_000
_LARGE_MATRIX_P_THRESH = 1e-3
_LARGE_MATRIX_LOGP_FLOOR = float(-np.log10(_LARGE_MATRIX_P_THRESH))
_WEBUI_RASTER_DPI = 180
_WEBUI_VECTOR_DPI = 150
_AUTO_SIG_TARGET_POINTS = 100_000
_GWAS_DISK_CACHE_VER = 1
_GWAS_DISK_CACHE_DIR = Path(tempfile.gettempdir()) / "janusx_webui" / "gwas_cache"


@dataclass(frozen=True)
class Bimrange:
    chrom: str
    start_bp: int
    end_bp: int
    label: str


@dataclass(frozen=True)
class Segment:
    br: Bimrange
    x_start: float
    x_end: float


@dataclass(frozen=True)
class HighlightRange:
    chrom: str
    start_bp: int
    end_bp: int
    logp_cap: float | None = None
    ld_keys: tuple[tuple[str, int], ...] = ()


def _column_key(v: object) -> str:
    s = str(v or "").strip().lower()
    if s.startswith("#"):
        s = s[1:]
    # Keep alnum only, so CHR/chr/chr_id/chr-id/chr.id all map consistently.
    return re.sub(r"[^a-z0-9]+", "", s)


def _pick_existing_column(df: pd.DataFrame, candidates: list[str], *, kind: str = "") -> str | None:
    cols = list(df.columns)
    cmap = {_column_key(c): str(c) for c in cols}
    for c in candidates:
        hit = cmap.get(_column_key(c))
        if hit is not None:
            return hit
    # Heuristic fallback for non-standard headers.
    best_col = None
    best_score = 0
    for c in cols:
        name = str(c)
        k = _column_key(name)
        if k == "":
            continue
        score = 0
        if kind == "chrom":
            if ("chromosome" in k) or k.startswith("chrom"):
                score = 100
            elif k.startswith("chr"):
                score = 95
            elif ("contig" in k) or ("scaffold" in k) or ("seqid" in k) or ("seqname" in k):
                score = 90
        elif kind == "pos":
            if k in {"pos", "bp", "ps"}:
                score = 100
            elif "position" in k:
                score = 98
            elif k.startswith("pos") or k.endswith("bp"):
                score = 95
            elif ("basepair" in k) or ("coordinate" in k) or ("location" in k):
                score = 90
        elif kind == "p":
            if k == "p":
                score = 100
            elif k in {"pvalue", "pval", "pwald", "pvaluewald", "waldp", "waldpvalue"}:
                score = 98
            elif k.startswith("pvalue") or k.startswith("pval"):
                score = 96
            elif ("pwald" in k) or ("wald" in k and k.startswith("p")):
                score = 94
            elif ("lrt" in k and k.startswith("p")):
                score = 90
        if score > best_score:
            best_score = score
            best_col = name
    if best_score >= 90:
        return str(best_col)
    return None


def _normalize_chrom(v: object) -> str:
    s = str(v).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    s = s.strip()
    if s == "" or s.lower() in {"nan", "na", "null", "none", "."}:
        return ""
    try:
        f = float(s)
        if np.isfinite(f) and float(f).is_integer():
            return str(int(f))
    except Exception:
        pass
    up = s.upper()
    if up in {"X", "Y", "M", "MT"}:
        return up
    return up


def _freeze_sort_key(obj: object) -> object:
    if isinstance(obj, list):
        return tuple(_freeze_sort_key(x) for x in obj)
    if isinstance(obj, tuple):
        return tuple(_freeze_sort_key(x) for x in obj)
    return obj


def _natural_key(text: object) -> tuple[int, object]:
    return _freeze_sort_key(_chrom_sort_key(_normalize_chrom(text)))  # type: ignore[return-value]


def _mix_color(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    x = float(np.clip(float(t), 0.0, 1.0))
    return (
        float(a[0] * (1.0 - x) + b[0] * x),
        float(a[1] * (1.0 - x) + b[1] * x),
        float(a[2] * (1.0 - x) + b[2] * x),
    )


def _luminance(rgb: tuple[float, float, float]) -> float:
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def _boost_saturation(rgb: tuple[float, float, float], factor: float = 1.25) -> tuple[float, float, float]:
    hsv = mcolors.rgb_to_hsv(np.array([[rgb]], dtype=float))[0, 0]
    hsv[1] = float(np.clip(hsv[1] * float(factor), 0.0, 1.0))
    out = mcolors.hsv_to_rgb(np.array([[hsv]], dtype=float))[0, 0]
    return (float(out[0]), float(out[1]), float(out[2]))


def _derive_tone_palette(base_color: str) -> dict[str, Any]:
    default_dark = (0.25, 0.29, 0.36)
    try:
        base = tuple(float(x) for x in mcolors.to_rgb(str(base_color or "#4b5563")))
    except Exception:
        base = default_dark

    lum = _luminance(base)
    if lum >= 0.55:
        light = base
        dark = _mix_color(base, (0.0, 0.0, 0.0), 0.82)
    else:
        dark = _mix_color(base, (0.0, 0.0, 0.0), 0.25)
        light = _mix_color(base, (1.0, 1.0, 1.0), 0.82)

    # Ensure usable contrast in very gray / mid luminance inputs.
    if abs(_luminance(light) - _luminance(dark)) < 0.45:
        light = _mix_color(light, (1.0, 1.0, 1.0), 0.35)
        dark = _mix_color(dark, (0.0, 0.0, 0.0), 0.35)

    dark = _boost_saturation(dark, 1.20)
    light = _boost_saturation(light, 1.10)
    mid = _boost_saturation(_mix_color(dark, light, 0.54), 1.15)
    cmap = LinearSegmentedColormap.from_list(
        "janusx_ld_tone",
        [
            (0.00, mcolors.to_hex(dark)),
            (0.40, mcolors.to_hex(mid)),
            (1.00, mcolors.to_hex(light)),
        ],
        N=256,
    )
    return {
        "dark": mcolors.to_hex(dark),
        "mid": mcolors.to_hex(mid),
        "light": mcolors.to_hex(light),
        "cmap": cmap,
    }


def _parse_float(raw: Any, default: float, *, min_value: float | None = None) -> float:
    try:
        v = float(raw)
    except Exception:
        v = float(default)
    if not np.isfinite(v):
        v = float(default)
    if min_value is not None and v < float(min_value):
        v = float(min_value)
    return float(v)


def _min_fig_h_for_ld_triangle(
    *,
    fig_w: float,
    left: float,
    right: float,
    top: float,
    bottom: float,
    height_ratios: list[float],
    ld_ratio_index: int = 2,
) -> float:
    """
    Minimum figure height so that LD panel can keep an isosceles-right triangle
    while sharing the exact Manhattan width.
    """
    wr = max(1e-9, float(right) - float(left))
    hr = max(1e-9, float(top) - float(bottom))
    hsum = float(np.sum(np.asarray(height_ratios, dtype=float)))
    if hsum <= 0.0:
        return 1.0
    ld_frac = max(1e-9, float(height_ratios[int(ld_ratio_index)]) / hsum)
    # Triangle in LDblock uses aspect=0.5, so required LD panel height ~= width*0.5.
    need = (float(fig_w) * wr * 0.5) / (hr * ld_frac)
    return float(max(1.0, need))


def _transition_ratio_from_manhattan(top_ratio: float) -> float:
    """
    Transition-layer height follows Manhattan panel height in WebUI.
    """
    return float(np.clip(0.11 * float(top_ratio), 0.025, 0.16))


def _ld_layout_for_webui(
    *,
    fig_w: float,
    left: float,
    right: float,
    top: float,
    bottom: float,
    manh_ratio: float,
) -> tuple[float, list[float]]:
    """
    Compute figure height + GridSpec height ratios for WebUI (with LD block),
    while keeping:
    1) Manhattan panel width/height ratio == manh_ratio
    2) LD block panel suitable for isosceles-right triangle (height ~= width * 0.5)
    3) Transition panel scales with Manhattan height
    """
    wr = max(1e-9, float(right) - float(left))
    hr = max(1e-9, float(top) - float(bottom))
    panel_w = float(fig_w) * wr
    ratio = max(0.2, float(manh_ratio))

    top_h = panel_w / ratio
    ld_h = panel_w * 0.5
    mid_h = float(np.clip(0.11 * top_h, 0.03, 1.20))

    fig_h = (top_h + mid_h + ld_h) / hr
    return float(max(1.0, fig_h)), [float(top_h), float(mid_h), float(ld_h)]


def _sync_qq_panel_with_manhattan_height(
    ax_manh: plt.Axes,
    ax_qq: plt.Axes,
    *,
    qq_box_ratio: float = 1.0,
) -> None:
    """
    Keep QQ panel y-height identical to Manhattan panel and preserve the
    historical QQ box ratio (default 1:1) inside its slot.
    """
    manh_pos = ax_manh.get_position()
    qq_slot = ax_qq.get_position()
    fig_w, fig_h = ax_qq.figure.get_size_inches()
    if not (np.isfinite(fig_w) and np.isfinite(fig_h) and fig_w > 0 and fig_h > 0):
        ax_qq.set_position([qq_slot.x0, manh_pos.y0, qq_slot.width, manh_pos.height])
        return

    target_w = float(manh_pos.height) * (float(fig_h) / float(fig_w)) * float(max(0.2, qq_box_ratio))
    new_w = float(min(float(qq_slot.width), max(1e-6, target_w)))
    x0 = float(qq_slot.x0) + 0.5 * (float(qq_slot.width) - new_w)
    ax_qq.set_position([x0, float(manh_pos.y0), new_w, float(manh_pos.height)])


def _normalize_marker(raw: Any) -> str:
    m = str(raw or "o").strip()
    allowed = {"o", ".", ",", "x", "+", "s", "^", "v", "D", "*", "P", "X", "h", "H", "d", "p", "<", ">"}
    return m if m in allowed else "o"


def _resolve_manh_color_set(palette: Any, tone: dict[str, Any] | None = None) -> list[str]:
    p_raw = str(palette or "").strip()
    p = p_raw.lower() if p_raw != "" else "auto"
    if p == "tab10":
        cmap = plt.get_cmap("tab10")
        vals = getattr(cmap, "colors", None)
        if vals is not None and len(vals) > 0:
            return [mcolors.to_hex(c) for c in vals]
    elif p == "tab20":
        cmap = plt.get_cmap("tab20")
        vals = getattr(cmap, "colors", None)
        if vals is not None and len(vals) > 0:
            return [mcolors.to_hex(c) for c in vals]
    elif p not in {"", "auto"}:
        # Accept explicit single color value (e.g. #1f77b4, rgb name).
        try:
            return [mcolors.to_hex(mcolors.to_rgb(p_raw))]
        except Exception:
            pass

    # auto: use high-contrast tone pair from LD/Gene color picker.
    t = tone or _derive_tone_palette("#4b5563")
    return [str(t.get("dark", "#111111")), str(t.get("light", "#8a8a8a"))]


def _parse_color_rgb(raw: Any) -> tuple[float, float, float] | None:
    try:
        return tuple(float(x) for x in mcolors.to_rgb(str(raw)))
    except Exception:
        return None


def _resolve_single_contrast_manh_colors(
    *,
    manh_palette: Any,
    ld_color: Any,
    tone: dict[str, Any] | None = None,
) -> list[str]:
    """
    Single-GWAS WebUI color policy:
    - keep a same-hue high-contrast pair (dark/light), not complementary inversion
    - if user selected a manhattan color, use it as the base hue
    """
    p_txt = str(manh_palette or "").strip()
    p_lc = p_txt.lower()
    manh_rgb = None if p_lc in {"", "auto", "tab10", "tab20"} else _parse_color_rgb(p_txt)
    ld_rgb = _parse_color_rgb(ld_color)

    if manh_rgb is not None:
        base = manh_rgb
    elif ld_rgb is not None:
        base = ld_rgb
    else:
        t = tone or _derive_tone_palette("#4b5563")
        base = _parse_color_rgb(str(t.get("mid", "#4b5563"))) or (0.30, 0.34, 0.39)

    lum = _luminance(base)
    if lum <= 0.28:
        dark = _mix_color(base, (0.0, 0.0, 0.0), 0.10)
        light = _mix_color(base, (1.0, 1.0, 1.0), 0.62)
    elif lum <= 0.60:
        dark = _mix_color(base, (0.0, 0.0, 0.0), 0.30)
        light = _mix_color(base, (1.0, 1.0, 1.0), 0.45)
    else:
        dark = _mix_color(base, (0.0, 0.0, 0.0), 0.48)
        light = _mix_color(base, (1.0, 1.0, 1.0), 0.18)

    dark = _boost_saturation(dark, 1.18)
    light = _boost_saturation(light, 1.06)

    # Ensure visible contrast for alternating chromosomes while preserving hue.
    if abs(_luminance(dark) - _luminance(light)) < 0.24:
        dark = _mix_color(dark, (0.0, 0.0, 0.0), 0.16)
        light = _mix_color(light, (1.0, 1.0, 1.0), 0.16)

    return [mcolors.to_hex(dark), mcolors.to_hex(light)]


def _parse_bimrange(text: str) -> Bimrange:
    raw = str(text or "").strip()
    if raw == "":
        raise ValueError("empty bimrange")
    m = re.match(r"^([^:]+):([0-9]*\.?[0-9]+)(?:-|:)([0-9]*\.?[0-9]+)$", raw)
    if m is None:
        raise ValueError("Invalid bimrange. Use chr:start-end or chr:start:end.")
    chrom = _normalize_chrom(m.group(1))
    s_txt = m.group(2).strip()
    e_txt = m.group(3).strip()
    s_num = float(s_txt)
    e_num = float(e_txt)
    if s_num < 0 or e_num < 0:
        raise ValueError("Invalid bimrange: start/end must be >= 0.")

    def _looks_bp(tok: str) -> bool:
        t = tok.strip()
        if "." in t:
            return False
        t = t.lstrip("0")
        if t == "":
            t = "0"
        return len(t) > 6

    if _looks_bp(s_txt) or _looks_bp(e_txt):
        start_bp = int(round(s_num))
        end_bp = int(round(e_num))
        unit = "bp"
    else:
        start_bp = int(round(s_num * 1_000_000))
        end_bp = int(round(e_num * 1_000_000))
        unit = "Mb"
    if start_bp > end_bp:
        start_bp, end_bp = end_bp, start_bp
    if unit == "bp":
        label = f"{chrom}:{start_bp}-{end_bp} bp"
    else:
        label = f"{chrom}:{start_bp / 1e6:g}-{end_bp / 1e6:g} Mb"
    return Bimrange(chrom=chrom, start_bp=start_bp, end_bp=end_bp, label=label)


def _parse_bimranges(raw: Any) -> list[Bimrange]:
    vals: list[str] = []
    if isinstance(raw, list):
        vals = [str(x).strip() for x in raw if str(x).strip()]
    else:
        txt = str(raw or "").strip()
        if txt != "":
            vals = [x.strip() for x in re.split(r"[\n,;]+", txt) if x.strip()]
    out: list[Bimrange] = []
    for v in vals:
        out.append(_parse_bimrange(v))
    return out


def _parse_highlight_ranges(raw: Any) -> list[HighlightRange]:
    if raw is None:
        return []
    vals: list[Any]
    if isinstance(raw, list):
        vals = list(raw)
    else:
        vals = [raw]
    out: list[HighlightRange] = []
    for v in vals:
        if isinstance(v, dict):
            c = _normalize_chrom(v.get("chrom", ""))
            try:
                s = int(round(float(v.get("start", np.nan))))
                e = int(round(float(v.get("end", np.nan))))
            except Exception:
                continue
            if c == "":
                continue
            y_cap: float | None = None
            for k in ("logp", "y_cap", "ymax"):
                try:
                    vv = float(v.get(k, np.nan))
                except Exception:
                    continue
                if np.isfinite(vv) and vv > 0.0:
                    y_cap = float(vv)
                    break
            if s > e:
                s, e = e, s
            ld_raw = (
                str(v.get("ldclump", "") or "")
                or str(v.get("LDclump", "") or "")
                or str(v.get("ld_points", "") or "")
            )
            ld_keys: list[tuple[str, int]] = []
            if ld_raw.strip() != "":
                toks = re.split(r"[;, \t\r\n]+", ld_raw.strip())
                for tk in toks:
                    t = str(tk).strip()
                    if t == "":
                        continue
                    m = re.match(r"^(.+)_([0-9]+)$", t)
                    if m is None:
                        continue
                    cc = _normalize_chrom(m.group(1))
                    try:
                        pp = int(m.group(2))
                    except Exception:
                        continue
                    ld_keys.append((cc, pp))
            ld_unique = tuple(sorted(set(ld_keys), key=lambda x: (_natural_key(x[0]), int(x[1]))))
            out.append(
                HighlightRange(
                    chrom=c,
                    start_bp=max(0, s),
                    end_bp=max(0, e),
                    logp_cap=y_cap,
                    ld_keys=ld_unique,
                )
            )
            continue
        txt = str(v or "").strip()
        if txt == "":
            continue
        try:
            br = _parse_bimrange(txt)
        except Exception:
            continue
        out.append(
            HighlightRange(
                chrom=br.chrom,
                start_bp=br.start_bp,
                end_bp=br.end_bp,
                logp_cap=None,
                ld_keys=(),
            )
        )
    return out


def _draw_highlight_spans(
    ax: plt.Axes,
    spans: list[tuple[float, float, float | None]],
    *,
    color: str = "#dc2626",
) -> None:
    if len(spans) == 0:
        return
    y0, y1 = ax.get_ylim()
    dy = float(y1 - y0) if np.isfinite(y1 - y0) and (y1 - y0) != 0 else 1.0
    for x0, x1, y_cap in spans:
        if not (np.isfinite(x0) and np.isfinite(x1)):
            continue
        xa = float(min(x0, x1))
        xb = float(max(x0, x1))
        y_top_frac = 1.0
        if y_cap is not None and np.isfinite(y_cap):
            y_top = float(np.clip(float(y_cap), min(y0, y1), max(y0, y1)))
            y_top_frac = float(np.clip((y_top - y0) / dy, 0.0, 1.0))
            if y_top_frac <= 0.0:
                continue
        if xb <= xa:
            ax.axvline(
                xa,
                color=color,
                linewidth=1.1,
                linestyle="-",
                alpha=0.95,
                ymin=0.0,
                ymax=y_top_frac,
                zorder=12,
            )
            continue
        ax.axvspan(
            xa,
            xb,
            ymin=0.0,
            ymax=y_top_frac,
            facecolor=color,
            alpha=0.14,
            edgecolor="none",
            zorder=3,
        )
        ax.axvline(
            xa,
            color=color,
            linewidth=1.0,
            linestyle="-",
            alpha=0.95,
            ymin=0.0,
            ymax=y_top_frac,
            zorder=12,
        )
        ax.axvline(
            xb,
            color=color,
            linewidth=1.0,
            linestyle="-",
            alpha=0.95,
            ymin=0.0,
            ymax=y_top_frac,
            zorder=12,
        )


def _resolve_result_file(row: dict[str, Any]) -> Path:
    def _base_dirs_for_row(row_data: dict[str, Any]) -> list[Path]:
        bases: list[Path] = []
        for p in [Path.cwd()]:
            if p not in bases:
                bases.append(p)
        for key in ("outprefix", "log_file", "phenofile", "genofile"):
            v = str(row_data.get(key, "")).strip()
            if v == "":
                continue
            try:
                parent = safe_resolve(safe_expanduser(v).parent)
            except Exception:
                parent = safe_expanduser(v).parent
            if parent not in bases:
                bases.append(parent)
        return bases

    def _resolve_existing_path(raw_path: str, base_dirs: list[Path]) -> Path | None:
        txt = str(raw_path or "").strip()
        if txt == "":
            return None
        p = safe_expanduser(txt)
        if p.exists():
            try:
                return safe_resolve(p)
            except Exception:
                return p
        if p.is_absolute():
            return None
        for bd in base_dirs:
            try:
                cand = safe_expanduser(bd / p)
            except Exception:
                continue
            if cand.exists():
                try:
                    return safe_resolve(cand)
                except Exception:
                    return cand
        return None

    base_dirs = _base_dirs_for_row(row)
    p = str(row.get("result_file", "")).strip()
    if p != "":
        resolved = _resolve_existing_path(p, base_dirs)
        if resolved is not None:
            return resolved
    run_files = row.get("result_files", [])
    if isinstance(run_files, list):
        for r in run_files:
            rp = str(r).strip()
            if rp == "":
                continue
            resolved = _resolve_existing_path(rp, base_dirs)
            if resolved is not None:
                return resolved
    raise FileNotFoundError("No valid GWAS result file found in history row.")


def activate_task_cache(row: dict[str, Any], *, anno_file: Any = None, preload: bool = True) -> None:
    """
    Keep GWAS/GFF cache scoped to current selected task.
    On task switch, drop previous task caches to control memory.
    """
    gwas_key = ""
    try:
        gwas_key = str(_resolve_result_file(row))
    except Exception:
        gwas_key = ""

    anno_change_requested = anno_file is not None
    anno_key = ""
    if anno_change_requested:
        anno_txt = str(anno_file or "").strip()
        if anno_txt != "":
            p = safe_expanduser(anno_txt)
            try:
                anno_key = str(safe_resolve(p))
            except Exception:
                anno_key = str(p)

    prev_g = str(_ACTIVE_TASK_CACHE.get("gwas", ""))
    prev_a = str(_ACTIVE_TASK_CACHE.get("anno", ""))

    if gwas_key != prev_g:
        if gwas_key != "":
            for k in list(_SIG_GWAS_CACHE.keys()):
                if str(k) != gwas_key:
                    _SIG_GWAS_CACHE.pop(k, None)
            for k in list(_LD_CACHE.keys()):
                if f"|{gwas_key}|" not in str(k):
                    _LD_CACHE.pop(k, None)
        else:
            _SIG_GWAS_CACHE.clear()
            _LD_CACHE.clear()

    if anno_change_requested and anno_key != prev_a:
        if anno_key != "":
            for k in list(_ANNO_CACHE.keys()):
                if str(k) != anno_key:
                    _ANNO_CACHE.pop(k, None)
            for k in list(_TRACK_ANNO_CACHE.keys()):
                if str(k) != anno_key:
                    _TRACK_ANNO_CACHE.pop(k, None)
        else:
            _ANNO_CACHE.clear()
            _TRACK_ANNO_CACHE.clear()

    _ACTIVE_TASK_CACHE["gwas"] = gwas_key
    if anno_change_requested:
        _ACTIVE_TASK_CACHE["anno"] = anno_key

    if not bool(preload):
        return

    # Preload current task data into memory.
    if gwas_key != "":
        try:
            _load_sig_table_cached(Path(gwas_key))
        except Exception:
            pass
    if anno_change_requested and anno_key != "":
        try:
            _load_anno_table_for_sig(anno_key)
        except Exception:
            pass
        try:
            _load_track_anno_cached(anno_key)
        except Exception:
            pass


def _path_resolved_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _path_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return -1.0


def _path_stat_token(path: Path) -> tuple[int, int]:
    try:
        st = path.stat()
        return int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size)
    except Exception:
        return -1, -1


def _disk_cache_file(path: Path) -> Path:
    rp = _path_resolved_key(path)
    key = hashlib.sha1(rp.encode("utf-8", errors="ignore")).hexdigest()
    return _GWAS_DISK_CACHE_DIR / f"{key}.npz"


def _build_gwas_df(
    chrom_norm: np.ndarray,
    pos_arr: np.ndarray,
    p_arr: np.ndarray,
    y_arr: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chrom_norm": pd.Categorical(np.asarray(chrom_norm, dtype=object)),
            "pos": pos_arr,
            "p": p_arr,
            "y": y_arr,
        }
    )


def _read_gwas_disk_cache(path: Path) -> tuple[pd.DataFrame, dict[str, str]] | None:
    cpath = _disk_cache_file(path)
    if not cpath.exists():
        return None
    mtime_ns, fsize = _path_stat_token(path)
    if mtime_ns < 0 or fsize < 0:
        return None
    try:
        with np.load(cpath, allow_pickle=False) as z:
            ver = int(np.asarray(z["ver"]).item())
            if ver != int(_GWAS_DISK_CACHE_VER):
                return None
            c_mtime = int(np.asarray(z["mtime_ns"]).item())
            c_size = int(np.asarray(z["fsize"]).item())
            if c_mtime != int(mtime_ns) or c_size != int(fsize):
                return None
            chrom_norm = np.asarray(z["chrom_norm"], dtype=object)
            pos_arr = np.asarray(z["pos"])
            p_arr = np.asarray(z["p"], dtype=np.float32)
            y_arr = np.asarray(z["y"], dtype=np.float32)
            if not (chrom_norm.shape[0] == pos_arr.shape[0] == p_arr.shape[0] == y_arr.shape[0]):
                return None
            cols = {
                "chr": str(np.asarray(z["col_chr"]).item()),
                "pos": str(np.asarray(z["col_pos"]).item()),
                "p": str(np.asarray(z["col_p"]).item()),
            }
            return _build_gwas_df(chrom_norm, pos_arr, p_arr, y_arr), cols
    except Exception:
        return None


def _write_gwas_disk_cache(
    path: Path,
    *,
    chrom_norm: np.ndarray,
    pos_arr: np.ndarray,
    p_arr: np.ndarray,
    y_arr: np.ndarray,
    cols: dict[str, str],
) -> None:
    mtime_ns, fsize = _path_stat_token(path)
    if mtime_ns < 0 or fsize < 0:
        return
    try:
        _GWAS_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cpath = _disk_cache_file(path)
        tmp_base = cpath.with_suffix(".tmp")
        np.savez_compressed(
            str(tmp_base),
            ver=np.asarray([int(_GWAS_DISK_CACHE_VER)], dtype=np.int32),
            mtime_ns=np.asarray([int(mtime_ns)], dtype=np.int64),
            fsize=np.asarray([int(fsize)], dtype=np.int64),
            chrom_norm=np.asarray(chrom_norm, dtype=str),
            pos=np.asarray(pos_arr),
            p=np.asarray(p_arr, dtype=np.float32),
            y=np.asarray(y_arr, dtype=np.float32),
            col_chr=np.asarray([str(cols.get("chr", ""))], dtype=str),
            col_pos=np.asarray([str(cols.get("pos", ""))], dtype=str),
            col_p=np.asarray([str(cols.get("p", ""))], dtype=str),
        )
        tmp = Path(str(tmp_base) + ".npz")
        try:
            tmp.replace(cpath)
        except Exception:
            pass
    except Exception:
        pass


def _peek_table_cache(
    path: Path,
    *,
    mtime: float,
    cache: dict[str, tuple[float, pd.DataFrame, dict[str, str]]] | None = None,
) -> tuple[pd.DataFrame | None, dict[str, str], str]:
    key = str(path)
    if cache is not None and key in cache:
        c_mtime, c_df, c_cols = cache[key]
        if float(c_mtime) == float(mtime):
            return c_df, dict(c_cols), "render_cache"
    rp = _path_resolved_key(path)
    cached_sig = _SIG_GWAS_CACHE.get(rp)
    if cached_sig is not None:
        s_mtime, s_df, s_cols = cached_sig
        if float(s_mtime) == float(mtime):
            if cache is not None:
                cache[key] = (float(mtime), s_df, dict(s_cols))
            return s_df, dict(s_cols), "sig_cache"
    return None, {}, "miss"


def _store_table_cache(
    path: Path,
    *,
    mtime: float,
    table: pd.DataFrame,
    cols: dict[str, str],
    cache: dict[str, tuple[float, pd.DataFrame, dict[str, str]]] | None = None,
) -> None:
    key = str(path)
    rp = _path_resolved_key(path)
    norm_cols = {
        "chr": str(cols.get("chr", "")),
        "pos": str(cols.get("pos", "")),
        "p": str(cols.get("p", "")),
    }
    if cache is not None:
        cache[key] = (float(mtime), table, dict(norm_cols))
    _SIG_GWAS_CACHE[rp] = (float(mtime), table, dict(norm_cols))


def _load_table_cached(
    path: Path,
    *,
    cache: dict[str, tuple[float, pd.DataFrame, dict[str, str]]] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], str, float]:
    mtime = _path_mtime(path)
    table, cols, source = _peek_table_cache(path, mtime=mtime, cache=cache)
    if table is not None:
        return table, cols, source, 0.0
    disk_hit = _read_gwas_disk_cache(path)
    if disk_hit is not None:
        table, cols = disk_hit
        _store_table_cache(path, mtime=mtime, table=table, cols=cols, cache=cache)
        return table, cols, "disk_cache", 0.0
    t0 = time.perf_counter()
    table, cols = _load_table(path)
    dt = float(time.perf_counter() - t0)
    _store_table_cache(path, mtime=mtime, table=table, cols=cols, cache=cache)
    return table, cols, "disk", dt


def _load_table(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    cached = _read_gwas_disk_cache(path)
    if cached is not None:
        return cached

    def _sniff_sep(p: Path) -> str:
        sample = ""
        try:
            with p.open("r", encoding="utf-8", errors="replace") as fh:
                for _ in range(16):
                    line = fh.readline()
                    if not line:
                        break
                    s = line.strip()
                    if s:
                        sample = s
                        break
        except Exception:
            sample = ""
        if "\t" in sample:
            return "tab"
        if "," in sample:
            return "comma"
        return "whitespace"

    def _read_fast(p: Path, sep_kind: str, *, nrows: int | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
        kwargs: dict[str, Any] = {}
        if nrows is not None:
            kwargs["nrows"] = int(nrows)
        if usecols is not None:
            kwargs["usecols"] = usecols
        if sep_kind == "tab":
            kwargs["sep"] = "\t"
            kwargs["engine"] = "c"
        elif sep_kind == "comma":
            kwargs["sep"] = ","
            kwargs["engine"] = "c"
        else:
            kwargs["delim_whitespace"] = True
            kwargs["engine"] = "c"
        return pd.read_csv(p, **kwargs)

    df: pd.DataFrame | None = None
    rust_ok = False
    c_chr = c_pos = c_p = None
    if _load_gwas_triplet_fast is not None:
        try:
            chrom_raw_l, pos_raw_l, p_raw_l, c_chr, c_pos, c_p = _load_gwas_triplet_fast(
                str(path),
                list(_CHROM_CANDIDATES),
                list(_POS_CANDIDATES),
                list(_P_CANDIDATES),
            )
            chrom_raw = np.asarray(chrom_raw_l, dtype=object)
            pos_raw = np.asarray(pos_raw_l, dtype=float)
            p_raw = np.asarray(p_raw_l, dtype=float)
            rust_ok = True
        except Exception:
            rust_ok = False
    if rust_ok:
        mask = np.isfinite(pos_raw) & np.isfinite(p_raw) & (p_raw > 0.0) & (p_raw <= 1.0)
        if not np.any(mask):
            raise RuntimeError("No valid SNP rows after filtering (pos/pvalue).")
        chrom_norm = np.asarray([_normalize_chrom(x) for x in chrom_raw[mask]], dtype=object)
        pos_f = pos_raw[mask]
        p_f = p_raw[mask]
        keep_chr = chrom_norm != ""
        if not np.any(keep_chr):
            raise RuntimeError("No valid chromosome labels after normalization.")
        chrom_norm = chrom_norm[keep_chr]
        pos_f = pos_f[keep_chr]
        p_f = p_f[keep_chr]

        pos_u = np.rint(pos_f).astype(np.int64, copy=False)
        if pos_u.size > 0 and np.min(pos_u) >= 0 and np.max(pos_u) <= np.iinfo(np.uint32).max:
            pos_arr = pos_u.astype(np.uint32, copy=False)
        else:
            pos_arr = pos_u.astype(np.int64, copy=False)

        p_arr = p_f.astype(np.float32, copy=False)
        y_arr = (-np.log10(np.clip(p_arr.astype(np.float64, copy=False), np.nextafter(0.0, 1.0), 1.0))).astype(
            np.float32,
            copy=False,
        )

        out = _build_gwas_df(chrom_norm, pos_arr, p_arr, y_arr)
        cols_out = {"chr": str(c_chr), "pos": str(c_pos), "p": str(c_p)}
        _write_gwas_disk_cache(
            path,
            chrom_norm=chrom_norm,
            pos_arr=pos_arr,
            p_arr=p_arr,
            y_arr=y_arr,
            cols=cols_out,
        )
        return out, cols_out

    try:
        sep_kind = _sniff_sep(path)
        head = _read_fast(path, sep_kind, nrows=0)
        c_chr = _pick_existing_column(head, _CHROM_CANDIDATES, kind="chrom")
        c_pos = _pick_existing_column(head, _POS_CANDIDATES, kind="pos")
        c_p = _pick_existing_column(head, _P_CANDIDATES, kind="p")
        if c_chr is not None and c_pos is not None and c_p is not None:
            df = _read_fast(path, sep_kind, usecols=[c_chr, c_pos, c_p])
    except Exception:
        df = None

    if df is None:
        try:
            df_full = pd.read_csv(path, sep=None, engine="python")
        except Exception as exc:
            raise RuntimeError(f"Failed to read GWAS file: {path}") from exc
        c_chr = _pick_existing_column(df_full, _CHROM_CANDIDATES, kind="chrom")
        c_pos = _pick_existing_column(df_full, _POS_CANDIDATES, kind="pos")
        c_p = _pick_existing_column(df_full, _P_CANDIDATES, kind="p")
        if c_chr is None or c_pos is None or c_p is None:
            raise RuntimeError(
                "Cannot detect required columns (chrom/pos/pvalue). "
                f"Found columns: {list(df_full.columns)}"
            )
        df = df_full[[c_chr, c_pos, c_p]].copy()

    if df.shape[0] == 0:
        raise RuntimeError("GWAS file has no rows.")
    if c_chr is None or c_pos is None or c_p is None:
        c_chr = _pick_existing_column(df, _CHROM_CANDIDATES, kind="chrom")
        c_pos = _pick_existing_column(df, _POS_CANDIDATES, kind="pos")
        c_p = _pick_existing_column(df, _P_CANDIDATES, kind="p")
    if c_chr is None or c_pos is None or c_p is None:
        raise RuntimeError("Cannot detect required columns (chrom/pos/pvalue).")

    chrom_raw = df[c_chr].astype(str).to_numpy(dtype=object)
    pos_raw = pd.to_numeric(df[c_pos], errors="coerce").to_numpy(dtype=float)
    p_raw = pd.to_numeric(df[c_p], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(pos_raw) & np.isfinite(p_raw) & (p_raw > 0.0) & (p_raw <= 1.0)
    if not np.any(mask):
        raise RuntimeError("No valid SNP rows after filtering (pos/pvalue).")
    chrom_norm = np.asarray([_normalize_chrom(x) for x in chrom_raw[mask]], dtype=object)
    pos_f = pos_raw[mask]
    p_f = p_raw[mask]
    keep_chr = chrom_norm != ""
    if not np.any(keep_chr):
        raise RuntimeError("No valid chromosome labels after normalization.")
    chrom_norm = chrom_norm[keep_chr]
    pos_f = pos_f[keep_chr]
    p_f = p_f[keep_chr]

    # Position uses compact integer dtype when possible.
    pos_u = np.rint(pos_f).astype(np.int64, copy=False)
    if pos_u.size > 0 and np.min(pos_u) >= 0 and np.max(pos_u) <= np.iinfo(np.uint32).max:
        pos_arr = pos_u.astype(np.uint32, copy=False)
    else:
        pos_arr = pos_u.astype(np.int64, copy=False)

    p_arr = p_f.astype(np.float32, copy=False)
    y_arr = (-np.log10(np.clip(p_arr.astype(np.float64, copy=False), np.nextafter(0.0, 1.0), 1.0))).astype(
        np.float32,
        copy=False,
    )

    out = _build_gwas_df(chrom_norm, pos_arr, p_arr, y_arr)
    cols_out = {"chr": c_chr, "pos": c_pos, "p": c_p}
    _write_gwas_disk_cache(
        path,
        chrom_norm=chrom_norm,
        pos_arr=pos_arr,
        p_arr=p_arr,
        y_arr=y_arr,
        cols=cols_out,
    )
    return out, cols_out


def detect_gwas_columns(path: Path) -> dict[str, Any]:
    """
    Validate GWAS table and report detected key columns for WebUI upload.
    """
    tbl, cols = _load_table(Path(path))
    chroms = sorted(pd.unique(tbl["chrom_norm"]).astype(str).tolist(), key=_natural_key)
    return {
        "chr": str(cols.get("chr", "")),
        "pos": str(cols.get("pos", "")),
        "pvalue": str(cols.get("p", "")),
        "n_rows": int(tbl.shape[0]),
        "n_chrom": int(len(chroms)),
        "chroms": chroms,
    }


def _load_sig_table_cached(path: Path) -> pd.DataFrame:
    """
    Load GWAS table with in-memory cache for sig-site/LD-clump path.
    This avoids re-reading large files when only threshold/LD params change.
    """
    rp = _path_resolved_key(path)
    mtime = _path_mtime(path)
    cached = _SIG_GWAS_CACHE.get(rp)
    if cached is not None and float(cached[0]) == float(mtime):
        return cached[1]
    tbl, cols = _load_table(path)
    _SIG_GWAS_CACHE[rp] = (
        float(mtime),
        tbl,
        {
            "chr": str(cols.get("chr", "")),
            "pos": str(cols.get("pos", "")),
            "p": str(cols.get("p", "")),
        },
    )
    return tbl


def _build_segments(brs: list[Bimrange], gap_mb: float = 0.0) -> list[Segment]:
    segs: list[Segment] = []
    cursor = 0.0
    for br in brs:
        width_mb = max(1e-9, (float(br.end_bp) - float(br.start_bp)) / 1e6)
        x0 = cursor
        x1 = x0 + width_mb
        segs.append(Segment(br=br, x_start=x0, x_end=x1))
        cursor = x1 + float(gap_mb)
    return segs


def _filter_by_bimranges(df: pd.DataFrame, brs: list[Bimrange]) -> pd.DataFrame:
    if len(brs) == 0:
        return df
    chrom = df["chrom_norm"].values
    pos = df["pos"].values.astype(float, copy=False)
    mask = np.zeros(df.shape[0], dtype=bool)
    for br in brs:
        mask |= (
            (chrom == br.chrom)
            & (pos >= float(br.start_bp))
            & (pos <= float(br.end_bp))
        )
    return df.loc[mask].copy()


def _compute_x_for_segments(df: pd.DataFrame, segs: list[Segment]) -> tuple[np.ndarray, np.ndarray]:
    n = df.shape[0]
    x = np.full(n, np.nan, dtype=float)
    seg_idx = np.full(n, -1, dtype=int)
    if n == 0 or len(segs) == 0:
        return x, seg_idx
    chrom = df["chrom_norm"].values
    pos = df["pos"].values.astype(float, copy=False)
    for i, seg in enumerate(segs):
        br = seg.br
        m = (
            (chrom == br.chrom)
            & (pos >= float(br.start_bp))
            & (pos <= float(br.end_bp))
        )
        if not np.any(m):
            continue
        x[m] = (pos[m] - float(br.start_bp)) / 1e6 + float(seg.x_start)
        seg_idx[m] = i
    return x, seg_idx


def _ld_bridge_x_from_index(idx: int, n_ld: int) -> float:
    """
    Bridge anchor x for LD panel.
    Map each LD row/column index directly to the SNP center along
    LDblock x-axis: [0.5, 1.5, ..., n-0.5].
    This keeps transition lines aligned with Manhattan SNP order.
    """
    n = int(max(1, n_ld))
    i = int(max(0, min(int(idx), n - 1)))
    return float(i) + 0.5


def _compute_global_x(df: pd.DataFrame) -> tuple[np.ndarray, list[tuple[float, str]], list[float]]:
    chroms = sorted(pd.unique(df["chrom_norm"]).tolist(), key=_natural_key)
    starts: dict[str, float] = {}
    mids: list[tuple[float, str]] = []
    bounds: list[float] = []
    cursor = 0.0
    for c in chroms:
        sub = df.loc[df["chrom_norm"] == c, "pos"]
        if sub.shape[0] == 0:
            continue
        lo = float(np.nanmin(sub.values))
        hi = float(np.nanmax(sub.values))
        starts[c] = cursor - lo
        mids.append((cursor + (hi - lo) * 0.5, c))
        cursor += (hi - lo)
        bounds.append(cursor)
        cursor += 1_000_000.0
    x = np.asarray(
        [float(pos) + starts.get(str(cn), 0.0) for pos, cn in zip(df["pos"].values, df["chrom_norm"].values)],
        dtype=float,
    )
    return x, mids, bounds[:-1]


def _downsample_for_qq(expected: np.ndarray, observed: np.ndarray, max_points: int = 140_000) -> tuple[np.ndarray, np.ndarray]:
    n = int(expected.size)
    if n <= max_points:
        return expected, observed
    sig_idx = np.where(observed >= 6.0)[0]
    keep_n = max(0, max_points - int(sig_idx.size))
    base_idx = np.linspace(0, n - 1, num=max(2, keep_n), dtype=int)
    idx = np.unique(np.concatenate([base_idx, sig_idx, np.array([n - 1], dtype=int)]))
    return expected[idx], observed[idx]


def _auto_threshold_from_pvals_bisect(
    pvals: np.ndarray,
    *,
    target_points: int = _AUTO_SIG_TARGET_POINTS,
) -> tuple[float, int]:
    pv = np.asarray(pvals, dtype=float)
    pv = pv[np.isfinite(pv) & (pv > 0.0)]
    if pv.size == 0:
        return (np.nan, 0)
    pv = np.clip(pv, np.nextafter(0.0, 1.0), 1.0)
    n = int(pv.size)
    tgt = int(max(1, min(int(target_points), n)))
    if n <= tgt:
        return (1.0, n)

    lo = float(np.nextafter(0.0, 1.0))
    hi = 1.0
    best_thr = hi
    best_cnt = int(np.count_nonzero(pv <= hi))
    best_gap = abs(best_cnt - tgt)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        cnt = int(np.count_nonzero(pv <= mid))
        gap = abs(cnt - tgt)
        if gap < best_gap or (gap == best_gap and cnt >= tgt and best_cnt < tgt):
            best_gap = gap
            best_thr = mid
            best_cnt = cnt
        if cnt < tgt:
            lo = mid
        else:
            hi = mid
    return (float(best_thr), int(best_cnt))


def _resolve_sig_threshold(
    table: pd.DataFrame,
    threshold: Any,
) -> tuple[float, int]:
    raw = str(threshold).strip().lower() if threshold is not None else ""
    if raw in {"", "auto", "none", "nan"}:
        return _auto_threshold_from_pvals_bisect(
            pd.to_numeric(table["p"], errors="coerce").to_numpy(dtype=float, copy=False),
            target_points=_AUTO_SIG_TARGET_POINTS,
        )
    try:
        thr = float(threshold)
    except Exception:
        return _auto_threshold_from_pvals_bisect(
            pd.to_numeric(table["p"], errors="coerce").to_numpy(dtype=float, copy=False),
            target_points=_AUTO_SIG_TARGET_POINTS,
        )
    if not np.isfinite(thr) or thr <= 0.0:
        return _auto_threshold_from_pvals_bisect(
            pd.to_numeric(table["p"], errors="coerce").to_numpy(dtype=float, copy=False),
            target_points=_AUTO_SIG_TARGET_POINTS,
        )
    return (float(min(thr, 1.0)), -1)


def _parse_gff_attrs(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in str(text or "").split(";"):
        t = str(item).strip()
        if t == "":
            continue
        if "=" in t:
            k, v = t.split("=", 1)
            out[k.strip().lower()] = v.strip().strip('"')
            continue
        if " " in t:
            k, v = t.split(" ", 1)
            out[k.strip().lower()] = v.strip().strip('"')
    return out


def _load_track_anno_cached(anno_file: str) -> pd.DataFrame:
    """
    Load full gene-structure annotation into memory cache for current task.
    Columns: chrom_norm,start,end,name,feature,parent,strand
    """
    txt = str(anno_file or "").strip()
    if txt == "":
        return pd.DataFrame(columns=["chrom_norm", "start", "end", "name", "feature", "parent", "strand"])
    p = safe_expanduser(txt)
    if not p.exists() or not p.is_file():
        return pd.DataFrame(columns=["chrom_norm", "start", "end", "name", "feature", "parent", "strand"])

    try:
        rp = str(p.resolve())
        mtime = float(p.stat().st_mtime)
    except Exception:
        rp = str(p)
        mtime = -1.0
    cached = _TRACK_ANNO_CACHE.get(rp)
    if cached is not None and float(cached[0]) == mtime:
        return cached[1]

    name_low = p.name.lower()
    is_bgz = name_low.endswith(".bgz")
    is_gz = name_low.endswith(".gz") or is_bgz
    if is_bgz:
        base_low = name_low[:-4]
    elif name_low.endswith(".gz"):
        base_low = name_low[:-3]
    else:
        base_low = name_low
    if base_low.endswith(".bed"):
        fmt = "bed"
    elif base_low.endswith(".gff") or base_low.endswith(".gff3"):
        fmt = "gff"
    else:
        fmt = "bed" if p.suffix.lower() == ".bed" else "gff"

    opener = gzip.open if is_gz else open
    rows: list[tuple[str, int, int, str, str, str, str]] = []
    try:
        with opener(p, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                arr = line.rstrip("\n").split("\t")
                if fmt == "bed":
                    if len(arr) < 3:
                        continue
                    chrom = _normalize_chrom(arr[0])
                    try:
                        s = int(float(arr[1])) + 1
                        e = int(float(arr[2]))
                    except Exception:
                        continue
                    name = arr[3] if len(arr) > 3 else f"{chrom}:{s}-{e}"
                    feature = "gene"
                    parent = str(name)
                    strand = str(arr[5]).strip() if len(arr) > 5 else "+"
                else:
                    if len(arr) < 5:
                        continue
                    chrom = _normalize_chrom(arr[0])
                    feature = str(arr[2]).strip().lower()
                    if feature not in {"gene", "cds", "utr", "five_prime_utr", "three_prime_utr"}:
                        continue
                    try:
                        s = int(float(arr[3]))
                        e = int(float(arr[4]))
                    except Exception:
                        continue
                    attrs = arr[8] if len(arr) > 8 else ""
                    ad = _parse_gff_attrs(attrs)
                    strand = str(arr[6]).strip() if len(arr) > 6 else "+"
                    name = (
                        ad.get("name")
                        or ad.get("gene")
                        or ad.get("gene_id")
                        or ad.get("id")
                        or f"{chrom}:{s}-{e}"
                    )
                    parent = (
                        ad.get("parent")
                        or ad.get("gene")
                        or ad.get("gene_id")
                        or ad.get("name")
                        or str(name)
                    )
                if e < s:
                    s, e = e, s
                strand = strand if strand in {"+", "-"} else "+"
                rows.append((str(chrom), int(s), int(e), str(name), str(feature), str(parent), str(strand)))
    except Exception:
        out_empty = pd.DataFrame(columns=["chrom_norm", "start", "end", "name", "feature", "parent", "strand"])
        _TRACK_ANNO_CACHE[rp] = (mtime, out_empty)
        return out_empty

    if len(rows) == 0:
        out_empty = pd.DataFrame(columns=["chrom_norm", "start", "end", "name", "feature", "parent", "strand"])
        _TRACK_ANNO_CACHE[rp] = (mtime, out_empty)
        return out_empty

    arr = np.asarray(rows, dtype=object)
    s = arr[:, 1].astype(np.int64, copy=False)
    e = arr[:, 2].astype(np.int64, copy=False)
    if np.min(s) >= 0 and np.max(e) <= np.iinfo(np.uint32).max:
        s = s.astype(np.uint32, copy=False)
        e = e.astype(np.uint32, copy=False)
    out = pd.DataFrame(
        {
            "chrom_norm": pd.Categorical(arr[:, 0].astype(str)),
            "start": s,
            "end": e,
            "name": arr[:, 3].astype(str),
            "feature": pd.Categorical(arr[:, 4].astype(str)),
            "parent": arr[:, 5].astype(str),
            "strand": pd.Categorical(arr[:, 6].astype(str)),
        }
    )
    out = out.sort_values(["chrom_norm", "start", "end"], kind="mergesort").reset_index(drop=True)
    _TRACK_ANNO_CACHE[rp] = (mtime, out)
    return out


def _read_gene_track(anno_file: str, br: Bimrange | None, max_rows: int = 1200) -> list[dict[str, Any]]:
    if br is None:
        return []
    txt = str(anno_file or "").strip()
    if txt == "":
        return []
    df = _load_track_anno_cached(txt)
    if df.shape[0] == 0:
        return []

    m = (
        (df["chrom_norm"].astype(str).values == br.chrom)
        & (df["start"].values.astype(np.int64, copy=False) <= int(br.end_bp))
        & (df["end"].values.astype(np.int64, copy=False) >= int(br.start_bp))
    )
    if not np.any(m):
        return []
    cols = ["start", "end", "name", "feature", "parent"]
    if "strand" in df.columns:
        cols.append("strand")
    sub = df.loc[m, cols]
    if int(max_rows) > 0 and sub.shape[0] > int(max_rows):
        sub = sub.iloc[: int(max_rows)].copy()

    out: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        s = max(int(r["start"]), int(br.start_bp))
        e = min(int(r["end"]), int(br.end_bp))
        if s >= e:
            continue
        out.append(
            {
                "start": int(s),
                "end": int(e),
                "name": str(r["name"]),
                "feature": str(r["feature"]),
                "parent": str(r["parent"]),
                "strand": str(r["strand"]) if "strand" in cols else "+",
            }
        )
    return out


def _nsnp_label(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _coerce_pos_int(v: object) -> int | None:
    try:
        x = int(float(v))
    except Exception:
        return None
    return x if x > 0 else None


def _sniff_sep_file(path: str) -> str:
    sample = ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(16):
                line = fh.readline()
                if not line:
                    break
                s = line.strip()
                if s:
                    sample = s
                    break
    except Exception:
        sample = ""
    if "\t" in sample:
        return "tab"
    if "," in sample:
        return "comma"
    return "whitespace"


def _estimate_n_from_phenofile(phenofile: str) -> int | None:
    p = str(phenofile or "").strip()
    if p == "":
        return None
    cache_key = f"pheno_n::{p}"
    cached = _META_CACHE.get(cache_key)
    if cached is not None and int(cached) > 0:
        return int(cached)
    path = safe_expanduser(p)
    if not path.exists() or not path.is_file():
        return None
    sep_kind = _sniff_sep_file(str(path))
    try:
        kwargs: dict[str, Any] = {"header": None, "usecols": [0]}
        if sep_kind == "tab":
            kwargs["sep"] = "\t"
            kwargs["engine"] = "c"
        elif sep_kind == "comma":
            kwargs["sep"] = ","
            kwargs["engine"] = "c"
        else:
            kwargs["delim_whitespace"] = True
            kwargs["engine"] = "c"
        ids = pd.read_csv(path, **kwargs).iloc[:, 0].astype(str)
        ids = ids[ids.str.strip() != ""]
        n = int(ids.nunique())
        if n > 0:
            _META_CACHE[cache_key] = n
            return n
    except Exception:
        return None
    return None


def _inspect_genotype_meta(genofile: str) -> tuple[int | None, int | None]:
    g = str(genofile or "").strip()
    if g == "":
        return None, None
    ck_n = f"geno_n::{g}"
    ck_s = f"geno_snp::{g}"
    n_cached = _META_CACHE.get(ck_n)
    s_cached = _META_CACHE.get(ck_s)
    if (n_cached is not None and int(n_cached) > 0) or (s_cached is not None and int(s_cached) > 0):
        return (
            int(n_cached) if n_cached is not None and int(n_cached) > 0 else None,
            int(s_cached) if s_cached is not None and int(s_cached) > 0 else None,
        )
    try:
        ids, n_snps = inspect_genotype_file(g)
        n_idv = int(len(ids)) if ids is not None else 0
        n_snp = int(n_snps) if n_snps is not None else 0
        if n_idv > 0:
            _META_CACHE[ck_n] = n_idv
        if n_snp > 0:
            _META_CACHE[ck_s] = n_snp
        return (n_idv if n_idv > 0 else None, n_snp if n_snp > 0 else None)
    except Exception:
        return None, None


def _ld_key(
    genofile: str,
    genokind: str,
    brs: list[Bimrange],
    *,
    gwas_token: str = "",
    ld_p_threshold: float = 1e-4,
) -> str:
    tail = ";".join([f"{b.chrom}:{int(b.start_bp)}:{int(b.end_bp)}" for b in brs])
    return f"{genokind}|{genofile}|{tail}|{gwas_token}|pthr={ld_p_threshold:.3e}"


def _compute_ld_r2(
    row: dict[str, Any],
    brs: list[Bimrange],
    *,
    max_snps: int = 320,
    selected_keys: set[tuple[str, int]] | None = None,
    gwas_token: str = "",
    ld_p_threshold: float = 1e-4,
) -> tuple[np.ndarray | None, list[tuple[str, int]], str]:
    if len(brs) == 0:
        return None, [], "Set bimrange to render LD block"
    if selected_keys is not None and len(selected_keys) == 0:
        return None, [], f"No SNPs pass LD threshold (p<={ld_p_threshold:g}) in selected bimrange"

    genofile = str(row.get("genofile", "")).strip()
    genokind = str(row.get("genofile_kind", "")).strip().lower()
    if genofile == "":
        return None, [], "No genotype in GWAS history row"
    if genokind not in {"bfile", "vcf", "tsv", "txt", "csv", "file"}:
        genokind = "bfile"

    k = _ld_key(
        genofile,
        genokind,
        brs,
        gwas_token=str(gwas_token or ""),
        ld_p_threshold=float(ld_p_threshold),
    )
    cached = _LD_CACHE.get(k)
    if cached is not None:
        mat = np.asarray(cached[0], dtype=float)
        if mat.ndim == 2 and mat.shape[0] >= 2 and mat.shape[0] == mat.shape[1]:
            return mat.copy(), list(cached[1]), ""

    path = safe_expanduser(genofile)
    if genokind == "bfile":
        if not (
            Path(f"{genofile}.bed").exists()
            and Path(f"{genofile}.bim").exists()
            and Path(f"{genofile}.fam").exists()
        ):
            return None, [], "Genotype prefix is incomplete (.bed/.bim/.fam)"
    elif not path.exists():
        return None, [], "Genotype file does not exist"

    geno_parts: list[np.ndarray] = []
    key_parts: list[tuple[str, int]] = []
    err_msgs: list[str] = []
    for br in brs:
        candidates = [br.chrom]
        if not br.chrom.lower().startswith("chr"):
            candidates.append(f"chr{br.chrom}")
        else:
            candidates.append(br.chrom[3:])
        got_for_this_range = False
        for chrom_sel in candidates:
            try:
                for geno, sites in load_genotype_chunks(
                    genofile,
                    chunk_size=50_000,
                    maf=0.0,
                    missing_rate=1.0,
                    impute=True,
                    model="add",
                    bim_range=(chrom_sel, int(br.start_bp), int(br.end_bp)),
                ):
                    if geno is None or geno.shape[0] == 0 or len(sites) == 0:
                        continue
                    keep_idx: list[int] = []
                    keep_keys: list[tuple[str, int]] = []
                    for i, s in enumerate(sites):
                        c = _normalize_chrom(getattr(s, "chrom", ""))
                        p = int(getattr(s, "pos", -1))
                        key = (c, p)
                        if c == br.chrom and (br.start_bp <= p <= br.end_bp):
                            if selected_keys is not None and key not in selected_keys:
                                continue
                            keep_idx.append(i)
                            keep_keys.append(key)
                    if len(keep_idx) == 0:
                        continue
                    geno_parts.append(np.asarray(geno[keep_idx, :], dtype=np.float32))
                    key_parts.extend(keep_keys)
                    got_for_this_range = True
                if got_for_this_range:
                    break
            except Exception as exc:
                err_msgs.append(str(exc))

    if len(geno_parts) == 0:
        if len(err_msgs) > 0:
            return None, [], f"LD unavailable: {err_msgs[-1]}"
        return None, [], "No genotype SNPs in selected bimrange"

    g = np.vstack(geno_parts)
    keys = key_parts
    if g.shape[0] < 2:
        return None, [], "Need >=2 SNPs for LD block"

    # Keep unique (chrom,pos) keys (first appearance) to stabilize matrix.
    seen: set[tuple[str, int]] = set()
    idx_first: list[int] = []
    for i, k2 in enumerate(keys):
        if k2 in seen:
            continue
        seen.add(k2)
        idx_first.append(i)
    if len(idx_first) < 2:
        return None, [], "Need >=2 unique SNP positions for LD block"
    idx_first_arr = np.asarray(idx_first, dtype=int)
    g = g[idx_first_arr, :]
    keys_unique = [keys[int(i)] for i in idx_first_arr]

    # If threshold-selected SNP keys are provided, preserve their correspondence
    # to Manhattan points by avoiding the legacy 320-SNP truncation.
    if selected_keys is not None:
        max_snps_eff = max(int(max_snps), int(len(selected_keys)))
    else:
        max_snps_eff = int(max_snps)

    if g.shape[0] > int(max_snps_eff):
        idx = np.linspace(0, g.shape[0] - 1, num=int(max_snps_eff), dtype=int)
        g = g[idx, :]
        keys_unique = [keys_unique[int(i)] for i in idx]

    c = np.corrcoef(g)
    c = np.asarray(c, dtype=float)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, 1.0)
    r2 = np.clip(c * c, 0.0, 1.0)
    if r2.ndim != 2 or r2.shape[0] < 2 or r2.shape[0] != r2.shape[1]:
        return None, [], "Invalid LD matrix shape"
    _LD_CACHE[k] = (r2.copy(), list(keys_unique))
    return r2, list(keys_unique), ""


def _lead_vs_all_r2(geno_block: np.ndarray) -> np.ndarray:
    """
    Compute lead-vs-all r2 for geno_block where row 0 is the lead SNP.
    """
    g = np.asarray(geno_block, dtype=np.float32)
    if g.ndim != 2 or g.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    lead = g[0].astype(np.float64, copy=False)
    mat = g.astype(np.float64, copy=False)
    lead_c = lead - np.nanmean(lead)
    mat_c = mat - np.nanmean(mat, axis=1, keepdims=True)
    num = np.dot(mat_c, lead_c)
    den = np.sqrt(np.sum(mat_c * mat_c, axis=1) * np.sum(lead_c * lead_c))
    corr = np.zeros(mat.shape[0], dtype=np.float64)
    ok = den > 0
    corr[ok] = num[ok] / den[ok]
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    r2 = np.clip(corr * corr, 0.0, 1.0)
    if r2.shape[0] > 0:
        r2[0] = 1.0
    return r2.astype(np.float32, copy=False)


def _clean_anno_token(v: object) -> str:
    x = str(v).strip()
    # Normalize simple serialized list-like wrappers, e.g. "['geneA']".
    if x.startswith("[") and x.endswith("]"):
        inner = x[1:-1].strip()
        if inner != "" and "," not in inner:
            x = inner
    if len(x) >= 2 and ((x[0] == "'" and x[-1] == "'") or (x[0] == '"' and x[-1] == '"')):
        x = x[1:-1].strip()
    if x == "" or x.lower() in {"na", "nan", "none", "null"}:
        return "NA"
    return re.sub(r"\s+", " ", x)


def _merge_anno_value(base: str, value: str) -> str:
    b = _clean_anno_token(base)
    v = _clean_anno_token(value)
    if b == "NA":
        return v
    if v == "NA":
        return b
    parts = [p for p in b.split("|") if p != ""]
    if v not in parts:
        parts.append(v)
    return "|".join(parts) if len(parts) > 0 else "NA"


def _load_anno_table_for_sig(anno_file: str) -> pd.DataFrame:
    txt = str(anno_file or "").strip()
    if txt == "":
        return pd.DataFrame(columns=["chrom_norm", "start", "end", "gene", "desc", "add"])

    p = safe_expanduser(txt)
    if not p.exists() or not p.is_file():
        return pd.DataFrame(columns=["chrom_norm", "start", "end", "gene", "desc", "add"])

    try:
        rp = str(p.resolve())
        mtime = float(p.stat().st_mtime)
    except Exception:
        rp = str(p)
        mtime = -1.0

    cached = _ANNO_CACHE.get(rp)
    if cached is not None and float(cached[0]) == mtime:
        return cached[1]

    try:
        anno = readanno(str(p), descItem="description")
    except Exception:
        out_empty = pd.DataFrame(columns=["chrom_norm", "start", "end", "gene", "desc", "add"])
        _ANNO_CACHE[rp] = (mtime, out_empty)
        return out_empty

    if anno.shape[0] == 0:
        out_empty = pd.DataFrame(columns=["chrom_norm", "start", "end", "gene", "desc", "add"])
        _ANNO_CACHE[rp] = (mtime, out_empty)
        return out_empty

    arr = anno.copy()
    for c in range(6):
        if c not in arr.columns:
            arr[c] = "NA"
    arr = arr[[0, 1, 2, 3, 4, 5]].copy()
    arr[0] = arr[0].astype(str).map(_normalize_chrom)
    arr[1] = pd.to_numeric(arr[1], errors="coerce")
    arr[2] = pd.to_numeric(arr[2], errors="coerce")
    arr = arr.dropna(subset=[0, 1, 2]).copy()
    if arr.shape[0] == 0:
        out_empty = pd.DataFrame(columns=["chrom_norm", "start", "end", "gene", "desc", "add"])
        _ANNO_CACHE[rp] = (mtime, out_empty)
        return out_empty

    s = np.minimum(arr[1].to_numpy(dtype=np.int64), arr[2].to_numpy(dtype=np.int64))
    e = np.maximum(arr[1].to_numpy(dtype=np.int64), arr[2].to_numpy(dtype=np.int64))
    out = pd.DataFrame(
        {
            "chrom_norm": arr[0].astype(str).values,
            "start": s.astype(np.int64, copy=False),
            "end": e.astype(np.int64, copy=False),
            "gene": arr[3].astype(str).values,
            "desc": arr[4].astype(str).values,
            "add": arr[5].astype(str).values,
        }
    )
    out = out.sort_values(["chrom_norm", "start", "end"], kind="mergesort").reset_index(drop=True)
    _ANNO_CACHE[rp] = (mtime, out)
    return out


def _format_sig_gene_hits(hits: pd.DataFrame) -> str:
    if hits.shape[0] == 0:
        return "NA"
    gene_map: dict[str, list[str]] = {}
    for _, r in hits.iterrows():
        g = _clean_anno_token(r.get("gene", "NA"))
        d = _clean_anno_token(r.get("desc", "NA"))
        a = _clean_anno_token(r.get("add", "NA"))
        if g not in gene_map:
            gene_map[g] = [d, a]
        else:
            gene_map[g][0] = _merge_anno_value(gene_map[g][0], d)
            gene_map[g][1] = _merge_anno_value(gene_map[g][1], a)
    if len(gene_map) == 0:
        return "NA"
    return ";".join([f"{g}:{vals[0]}/{vals[1]}" for g, vals in gene_map.items()])


def _annotate_sig_rows_with_genes(rows: list[dict[str, Any]], anno_file: str) -> None:
    if len(rows) == 0:
        return
    anno = _load_anno_table_for_sig(anno_file)
    if anno.shape[0] == 0:
        for r in rows:
            r["Gene"] = "NA"
        return

    by_chr: dict[str, pd.DataFrame] = {
        str(c): g.reset_index(drop=True)
        for c, g in anno.groupby("chrom_norm", sort=False)
    }
    for r in rows:
        chrom = _normalize_chrom(r.get("chrom", ""))
        if chrom == "":
            r["Gene"] = "NA"
            continue
        try:
            p = r.get("pos", None)
            p_i = int(p) if p is not None else None
        except Exception:
            p_i = None

        try:
            s_raw = r.get("start", None)
            s_i = int(s_raw) if s_raw is not None else None
        except Exception:
            s_i = None
        try:
            e_raw = r.get("end", None)
            e_i = int(e_raw) if e_raw is not None else None
        except Exception:
            e_i = None

        # Sig rows usually carry only `pos`; fallback to point interval.
        if s_i is None and e_i is None:
            if p_i is None:
                r["Gene"] = "NA"
                continue
            s_i, e_i = p_i, p_i
        else:
            if s_i is None:
                s_i = p_i if p_i is not None else e_i
            if e_i is None:
                e_i = p_i if p_i is not None else s_i
            if s_i is None or e_i is None:
                r["Gene"] = "NA"
                continue

        s = int(s_i)
        e = int(e_i)
        if s > e:
            s, e = e, s
        g = by_chr.get(chrom, None)
        if g is None or g.shape[0] == 0:
            r["Gene"] = "NA"
            continue
        mask = (g["start"].values <= int(e)) & (g["end"].values >= int(s))
        if not np.any(mask):
            r["Gene"] = "NA"
            continue
        hits = g.loc[mask, ["gene", "desc", "add"]]
        r["Gene"] = _format_sig_gene_hits(hits)


def annotate_sig_rows_with_genes(rows: list[dict[str, Any]], anno_file: str) -> None:
    """
    Public wrapper for server-side selective annotation.
    """
    _annotate_sig_rows_with_genes(rows, anno_file)


def build_sig_table(
    row: dict[str, Any],
    *,
    threshold: Any = "auto",
    window_bp: int = 10_000,
    r2_thr: float = 0.8,
    anno_file: str = "",
    annotate_rows: bool = True,
) -> dict[str, Any]:
    """
    Build threshold-passing SNP table.
    Threshold defaults to pwald bisection (~100k SNPs).
    """
    _ = window_bp
    _ = r2_thr
    gwas_path = _resolve_result_file(row)
    table = _load_sig_table_cached(gwas_path)
    thr, _auto_cnt = _resolve_sig_threshold(table, threshold)
    if not np.isfinite(thr) or thr <= 0.0:
        return {"threshold": float("nan"), "n_sig": 0, "n_leads": 0, "rows": []}

    work = table.loc[table["p"].values <= float(thr), ["chrom_norm", "pos", "p"]].copy()
    if work.shape[0] == 0:
        return {
            "threshold": float(thr),
            "n_sig": 0,
            "n_leads": 0,
            "rows": [],
        }

    work["chrom_norm"] = work["chrom_norm"].astype(str)
    work["pos"] = pd.to_numeric(work["pos"], errors="coerce").astype("Int64")
    work["p"] = pd.to_numeric(work["p"], errors="coerce")
    work = work.dropna(subset=["pos", "p"]).copy()
    if work.shape[0] == 0:
        return {
            "threshold": float(thr),
            "n_sig": 0,
            "n_leads": 0,
            "rows": [],
        }
    work["pos"] = work["pos"].astype(np.int64)
    work = (
        work.sort_values(["p", "chrom_norm", "pos"], ascending=[True, True, True])
        .drop_duplicates(subset=["chrom_norm", "pos"], keep="first")
        .reset_index(drop=True)
    )
    if int(work.shape[0]) == 0:
        return {
            "threshold": float(thr),
            "n_sig": 0,
            "n_leads": 0,
            "rows": [],
        }
    out_rows = [
        {
            "chrom": str(r["chrom_norm"]),
            "pos": int(r["pos"]),
            "p": float(r["p"]),
            "logp": float(-np.log10(max(float(r["p"]), np.nextafter(0.0, 1.0)))),
        }
        for _, r in work.iterrows()
    ]
    out_rows = sorted(
        out_rows,
        key=lambda x: (_natural_key(x["chrom"]), int(x["pos"]), float(x["p"])),
    )
    anno_txt = str(anno_file or "").strip()
    if bool(annotate_rows) and anno_txt != "":
        _annotate_sig_rows_with_genes(out_rows, anno_txt)
    else:
        for r in out_rows:
            r["Gene"] = ""
    n_sig = int(len(out_rows))
    return {
        "threshold": float(thr),
        "n_sig": int(n_sig),
        "n_leads": int(n_sig),
        "rows": out_rows,
    }


def build_merged_sig_table(
    rows: list[dict[str, Any]],
    *,
    threshold: Any = "auto",
    window_bp: int = 10_000,
    r2_thr: float = 0.8,
    anno_file: str = "",
    annotate_rows: bool = True,
) -> dict[str, Any]:
    """
    Build merged sig table across multiple GWAS results:
    1) keep SNPs that pass threshold in every GWAS (intersection)
    2) aggregate p by min-p across GWAS
    """
    _ = window_bp
    _ = r2_thr
    if not isinstance(rows, list) or len(rows) == 0:
        return {"threshold": float("nan"), "n_sig": 0, "n_leads": 0, "rows": []}

    tables: list[pd.DataFrame] = []
    first_pvals: np.ndarray | None = None
    for row in rows:
        gwas_path = _resolve_result_file(row)
        table = _load_sig_table_cached(gwas_path)
        tables.append(table)
        if first_pvals is None:
            first_pvals = pd.to_numeric(table["p"], errors="coerce").to_numpy(dtype=float, copy=False)
    base_pvals = first_pvals if first_pvals is not None else np.asarray([], dtype=float)
    if str(threshold).strip().lower() in {"", "auto", "none", "nan"}:
        thr, _auto_cnt = _auto_threshold_from_pvals_bisect(
            base_pvals,
            target_points=_AUTO_SIG_TARGET_POINTS,
        )
    else:
        try:
            thr = float(threshold)
        except Exception:
            thr, _auto_cnt = _auto_threshold_from_pvals_bisect(
                base_pvals,
                target_points=_AUTO_SIG_TARGET_POINTS,
            )
        else:
            if not np.isfinite(thr) or thr <= 0.0:
                thr, _auto_cnt = _auto_threshold_from_pvals_bisect(
                    base_pvals,
                    target_points=_AUTO_SIG_TARGET_POINTS,
                )
            else:
                thr = float(min(thr, 1.0))

    if not np.isfinite(thr) or thr <= 0.0:
        return {"threshold": float("nan"), "n_sig": 0, "n_leads": 0, "rows": []}

    per_maps: list[dict[tuple[str, int], float]] = []
    for table in tables:
        work = table.loc[table["p"].values <= float(thr), ["chrom_norm", "pos", "p"]].copy()
        if int(work.shape[0]) == 0:
            return {"threshold": float(thr), "n_sig": 0, "n_leads": 0, "rows": []}
        work["chrom_norm"] = work["chrom_norm"].astype(str)
        work["pos"] = pd.to_numeric(work["pos"], errors="coerce").astype("Int64")
        work["p"] = pd.to_numeric(work["p"], errors="coerce")
        work = work.dropna(subset=["pos", "p"]).copy()
        if int(work.shape[0]) == 0:
            return {"threshold": float(thr), "n_sig": 0, "n_leads": 0, "rows": []}
        work["pos"] = work["pos"].astype(np.int64)
        # per GWAS: duplicated SNP keeps min p
        g = (
            work.groupby(["chrom_norm", "pos"], as_index=False)["p"]
            .min()
            .sort_values(["p", "chrom_norm", "pos"], ascending=[True, True, True])
        )
        m: dict[tuple[str, int], float] = {}
        for _, r in g.iterrows():
            k = (str(r["chrom_norm"]), int(r["pos"]))
            pv = float(r["p"])
            old = m.get(k, None)
            if old is None or pv < old:
                m[k] = pv
        per_maps.append(m)

    if len(per_maps) == 0:
        return {"threshold": float(thr), "n_sig": 0, "n_leads": 0, "rows": []}

    keys_inter = set(per_maps[0].keys())
    for m in per_maps[1:]:
        keys_inter &= set(m.keys())
        if len(keys_inter) == 0:
            break
    if len(keys_inter) == 0:
        return {"threshold": float(thr), "n_sig": 0, "n_leads": 0, "rows": []}

    rows0: list[dict[str, Any]] = []
    for c, p in keys_inter:
        vals = [float(m[(c, p)]) for m in per_maps if (c, p) in m]
        if len(vals) == 0:
            continue
        pmin = float(np.min(np.asarray(vals, dtype=float)))
        rows0.append(
            {
                "chrom": str(c),
                "pos": int(p),
                "p": pmin,
                "logp": float(-np.log10(max(float(pmin), np.nextafter(0.0, 1.0)))),
            }
        )
    if len(rows0) == 0:
        return {"threshold": float(thr), "n_sig": 0, "n_leads": 0, "rows": []}

    out_rows = sorted(rows0, key=lambda x: (_natural_key(x["chrom"]), int(x["pos"]), float(x["p"])))
    anno_txt = str(anno_file or "").strip()
    if bool(annotate_rows) and anno_txt != "":
        _annotate_sig_rows_with_genes(out_rows, anno_txt)
    else:
        for r in out_rows:
            r["Gene"] = ""
    n_sig = int(len(out_rows))
    return {
        "threshold": float(thr),
        "n_sig": int(n_sig),
        "n_leads": int(n_sig),
        "rows": out_rows,
    }


def _merged_series_label(row: dict[str, Any], idx: int) -> str:
    ph = str(row.get("phenotype", "")).strip()
    md = str(row.get("model", "")).strip()
    gt = str(row.get("genotype_type", "")).strip()
    parts = [x for x in [ph, md, gt] if x != ""]
    if len(parts) > 0:
        return ".".join(parts)
    rf = str(row.get("result_file", "")).strip()
    if rf != "":
        return Path(rf).name
    return f"GWAS{idx + 1}"


def render_merged_manhattan_svg(
    rows: list[dict[str, Any]],
    *,
    bimrange_text: Any = "",
    threshold: Any = "auto",
    include_all_points: bool = False,
    manh_ratio: float = 2.0,
    manh_palette: str = "auto",
    manh_alpha: float = 0.7,
    manh_marker: str = "o",
    manh_size: float = 16.0,
    series_styles: Any = None,
    ld_color: str = "#4b5563",
    ld_p_threshold: Any = "auto",
    render_qq: bool = True,
    editable_svg: bool = False,
    cache: dict[str, tuple[float, pd.DataFrame, dict[str, str]]] | None = None,
) -> tuple[str, dict[str, Any]]:
    t_render0 = time.perf_counter()
    if not isinstance(rows, list) or len(rows) == 0:
        raise RuntimeError("No GWAS tasks selected.")
    brs = _parse_bimranges(bimrange_text)
    tone = _derive_tone_palette(ld_color)

    raw_thr = str(ld_p_threshold).strip().lower() if ld_p_threshold is not None else ""
    if raw_thr in {"", "auto", "none", "default"}:
        ld_auto = True
        ld_thr = 1.0
    else:
        ld_auto = False
        try:
            ld_thr = float(ld_p_threshold)
        except Exception:
            ld_thr = 1.0
    if not np.isfinite(ld_thr) or ld_thr <= 0.0:
        ld_thr = 1.0
    if ld_thr > 1.0:
        ld_thr = 1.0

    # Merged mode threshold:
    # keep thr adjustable; points below threshold are rendered in light gray.
    raw_sig_thr = str(threshold).strip().lower() if threshold is not None else ""
    if raw_sig_thr in {"", "auto", "none", "default", "nan"}:
        # In bimrange mode, do not enforce the default cutoff.
        # Keep all points unless user explicitly sets `thr`.
        sig_p_thr = 1.0 if len(brs) > 0 else float(_LARGE_MATRIX_P_THRESH)
    else:
        try:
            sig_p_thr = float(threshold)
        except Exception:
            sig_p_thr = 1.0 if len(brs) > 0 else float(_LARGE_MATRIX_P_THRESH)
    if (not np.isfinite(sig_p_thr)) or sig_p_thr <= 0.0 or sig_p_thr > 1.0:
        sig_p_thr = 1.0 if len(brs) > 0 else float(_LARGE_MATRIX_P_THRESH)
    sig_y_thr = float(-np.log10(max(float(sig_p_thr), np.nextafter(0.0, 1.0))))
    low_sig_color = "#d1d5db"
    # Full-point mode: keep all points in viewport (download path).
    # Coloring still follows `sig_p_thr`.
    full_point_mode = bool(include_all_points) or (
        bool(editable_svg) and float(sig_p_thr) >= (1.0 - 1e-12)
    )

    resolved_rows: list[tuple[int, dict[str, Any], Path]] = []
    for i, row in enumerate(rows):
        resolved_rows.append((i, row, _resolve_result_file(row)))

    table_by_key: dict[str, pd.DataFrame] = {}
    mtime_by_key: dict[str, float] = {}
    cache_src_by_key: dict[str, str] = {}
    miss_paths: dict[str, Path] = {}
    for _, _, gwas_path in resolved_rows:
        key = str(gwas_path)
        if key in table_by_key:
            continue
        mtime = _path_mtime(gwas_path)
        mtime_by_key[key] = mtime
        c_tbl, _c_cols, c_src = _peek_table_cache(gwas_path, mtime=mtime, cache=cache)
        if c_tbl is not None:
            table_by_key[key] = c_tbl
            cache_src_by_key[key] = c_src
        else:
            d_tbl = _read_gwas_disk_cache(gwas_path)
            if d_tbl is not None:
                tbl0, cols0 = d_tbl
                _store_table_cache(
                    gwas_path,
                    mtime=mtime,
                    table=tbl0,
                    cols=cols0,
                    cache=cache,
                )
                table_by_key[key] = tbl0
                cache_src_by_key[key] = "disk_cache"
            else:
                miss_paths[key] = gwas_path

    load_sec_total = 0.0
    if len(miss_paths) > 0:
        def _load_table_timed(path_obj: Path) -> tuple[pd.DataFrame, dict[str, str], float]:
            t0 = time.perf_counter()
            tbl0, cols0 = _load_table(path_obj)
            return tbl0, cols0, float(time.perf_counter() - t0)

        items = list(miss_paths.items())
        if len(items) == 1:
            key0, path0 = items[0]
            tbl0, cols0, dt0 = _load_table_timed(path0)
            load_sec_total += float(dt0)
            _store_table_cache(
                path0,
                mtime=mtime_by_key.get(key0, _path_mtime(path0)),
                table=tbl0,
                cols=cols0,
                cache=cache,
            )
            table_by_key[key0] = tbl0
            cache_src_by_key[key0] = "disk"
        else:
            workers = max(1, min(4, len(items)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fut_map = {pool.submit(_load_table_timed, p0): (k0, p0) for k0, p0 in items}
                for fut in as_completed(fut_map):
                    k0, p0 = fut_map[fut]
                    tbl0, cols0, dt0 = fut.result()
                    load_sec_total += float(dt0)
                    _store_table_cache(
                        p0,
                        mtime=mtime_by_key.get(k0, _path_mtime(p0)),
                        table=tbl0,
                        cols=cols0,
                        cache=cache,
                    )
                    table_by_key[k0] = tbl0
                    cache_src_by_key[k0] = "disk"

    series: list[dict[str, Any]] = []
    all_chroms: set[str] = set()
    for i, row, gwas_path in resolved_rows:
        key = str(gwas_path)
        table = table_by_key.get(key)
        if table is None:
            table, _cols, _src, dt = _load_table_cached(gwas_path, cache=cache)
            load_sec_total += float(dt)
            cache_src_by_key[key] = str(_src)
            table_by_key[key] = table
        if int(table.shape[0]) == 0:
            continue
        draw_df = _filter_by_bimranges(table, brs) if len(brs) > 0 else table
        if not full_point_mode:
            # Preview path: y-axis floor is fixed at default threshold line,
            # so drop invisible points early.
            # points early before coordinate mapping and scatter drawing.
            yv = draw_df["y"].to_numpy(dtype=float, copy=False)
            draw_df = draw_df.loc[np.isfinite(yv) & (yv >= float(_LARGE_MATRIX_LOGP_FLOOR))].copy()
        if int(draw_df.shape[0]) == 0:
            continue
        lbl = _merged_series_label(row, i)
        series.append(
            {
                "path": str(gwas_path),
                "row": row,
                "full": table,
                "draw": draw_df,
                "label": lbl,
            }
        )
        all_chroms.update(draw_df["chrom_norm"].astype(str).unique().tolist())

    if len(series) == 0:
        raise RuntimeError("No valid GWAS rows to draw.")

    # Palette logic:
    # Merge mode defaults to tab10 to distinguish GWAS series.
    n_series = len(series)
    p_in = str(manh_palette or "auto").strip().lower()
    if p_in == "auto":
        p_use = "tab10"
    else:
        p_use = p_in if p_in in {"tab10", "tab20"} else "tab10"
    color_base = _resolve_manh_color_set(p_use, tone)
    if len(color_base) == 0:
        color_base = ["#111111", "#8a8a8a"]

    ratio = _parse_float(manh_ratio, 2.0, min_value=0.2)
    alpha_default = float(np.clip(_parse_float(manh_alpha, 0.7, min_value=0.0), 0.0, 1.0))
    marker = _normalize_marker(manh_marker)
    msize = _parse_float(manh_size, 16.0, min_value=0.1)
    # Keep scatter-heavy artists rasterized for both preview and download.
    # Text/axes/labels remain vector in SVG output.
    rasterize_main = True
    fig_dpi = int(_WEBUI_RASTER_DPI)

    style_by_history: dict[str, dict[str, Any]] = {}
    style_by_index: dict[int, dict[str, Any]] = {}
    if isinstance(series_styles, list):
        for j, raw in enumerate(series_styles):
            if not isinstance(raw, dict):
                continue
            st: dict[str, Any] = {}
            c_raw = str(raw.get("color", "")).strip()
            if c_raw != "":
                try:
                    st["color"] = mcolors.to_hex(mcolors.to_rgb(c_raw))
                except Exception:
                    pass
            st["marker"] = _normalize_marker(raw.get("marker", "o"))
            st["size"] = _parse_float(raw.get("size", 16.0), 16.0, min_value=0.1)
            st["alpha"] = float(np.clip(_parse_float(raw.get("alpha", 0.7), 0.7, min_value=0.0), 0.0, 1.0))
            hid = str(raw.get("history_id", "")).strip()
            if hid != "":
                style_by_history[hid] = dict(st)
            style_by_index[int(j)] = dict(st)

    # Keep merge behavior aligned with single-GWAS:
    # - bimrange mode: Manhattan + LD only (no QQ)
    # - non-bimrange mode: QQ drawn only when render_qq is requested.
    show_ld = len(brs) > 0
    show_qq = (not show_ld) and bool(render_qq)
    if show_ld:
        fig_w = 15.0
        left = 0.05
        right = 0.995
        top = 0.965
        bottom = 0.06
        fig_h, h_ratios = _ld_layout_for_webui(
            fig_w=fig_w,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            manh_ratio=float(ratio),
        )
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=fig_dpi)
        gs = GridSpec(
            3,
            1,
            figure=fig,
            width_ratios=[1.0],
            height_ratios=h_ratios,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            hspace=0.018,
            wspace=0.0,
        )
        ax_manh = fig.add_subplot(gs[0, 0])
        ax_qq = None
        ax_bridge = fig.add_subplot(gs[1, 0])
        ax_ld = fig.add_subplot(gs[2, 0])
    else:
        if show_qq:
            fig = plt.figure(figsize=(15.0, 5.2), dpi=fig_dpi)
            mosaic = fig.subplot_mosaic([["A", "A", "B"]])
            ax_manh = mosaic["A"]
            ax_qq = mosaic["B"]
            ax_ld = None
            fig.subplots_adjust(left=0.05, right=0.995, top=0.96, bottom=0.13, wspace=0.09)
        else:
            fig = plt.figure(figsize=(15.0, 5.2), dpi=fig_dpi)
            ax_manh = fig.add_subplot(111)
            ax_qq = None
            ax_ld = None
            fig.subplots_adjust(left=0.05, right=0.995, top=0.96, bottom=0.13)
        ax_bridge = None

    # Build x mapping.
    xticks: list[tuple[float, str]] = []
    separators: list[float] = []
    segs: list[Segment] = []
    x_offsets: dict[str, float] = {}
    if len(brs) > 0:
        if len(brs) == 1:
            pass
        else:
            segs = _build_segments(brs, gap_mb=0.0)
            separators = [float(seg.x_end) for seg in segs[:-1]]
    else:
        chrom_order = sorted(list(all_chroms), key=_natural_key)
        if len(chrom_order) == 0:
            plt.close(fig)
            raise RuntimeError("No chromosome information found.")
        chr_minmax: dict[str, tuple[float, float]] = {}
        for c in chrom_order:
            lo = np.inf
            hi = -np.inf
            for s in series:
                sub = s["draw"].loc[s["draw"]["chrom_norm"].astype(str) == c, "pos"]
                if int(sub.shape[0]) == 0:
                    continue
                a = float(np.nanmin(sub.values.astype(float, copy=False)))
                b = float(np.nanmax(sub.values.astype(float, copy=False)))
                lo = min(lo, a)
                hi = max(hi, b)
            if np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
                chr_minmax[c] = (lo, hi)
        if len(chr_minmax) == 0:
            plt.close(fig)
            raise RuntimeError("No SNP points available across selected GWAS tasks.")
        cursor = 0.0
        for c in chrom_order:
            mm = chr_minmax.get(c, None)
            if mm is None:
                continue
            lo, hi = mm
            x_offsets[c] = cursor - lo
            xticks.append((cursor + (hi - lo) * 0.5, c))
            cursor += (hi - lo)
            separators.append(cursor)
            cursor += 1_000_000.0
        if len(separators) > 0:
            separators = separators[:-1]

    x_left = np.inf
    x_right = -np.inf
    y_top = -np.inf
    qq_xmax = 1.0
    for i, s in enumerate(series):
        df0 = s["draw"]
        y = df0["y"].to_numpy(dtype=float, copy=False)
        p = df0["p"].to_numpy(dtype=float, copy=False)
        if len(brs) == 0:
            chrom = df0["chrom_norm"].astype(str).to_numpy(dtype=object)
            pos = df0["pos"].to_numpy(dtype=float, copy=False)
            x = np.full(pos.shape[0], np.nan, dtype=float)
            for c in np.unique(chrom):
                off = x_offsets.get(str(c), None)
                if off is None:
                    continue
                m = chrom == c
                x[m] = pos[m] + float(off)
        elif len(brs) == 1:
            x = df0["pos"].to_numpy(dtype=float, copy=False) / 1e6
        else:
            x_all, seg_idx = _compute_x_for_segments(df0, segs)
            keep_seg = seg_idx >= 0
            if not np.any(keep_seg):
                continue
            x = x_all[keep_seg]
            y = y[keep_seg]
            p = p[keep_seg]
        keep = np.isfinite(x) & np.isfinite(y)
        if not np.any(keep):
            continue
        xk = x[keep]
        yk = y[keep]
        pk = p[keep]
        x_left = min(x_left, float(np.nanmin(xk)))
        x_right = max(x_right, float(np.nanmax(xk)))
        y_top = max(y_top, float(np.nanmax(yk)))
        if not full_point_mode:
            # Preview path: y-axis floor is fixed at default threshold line, so points below it are
            # guaranteed to be outside viewport; skip plotting them early.
            vis = yk >= float(_LARGE_MATRIX_LOGP_FLOOR)
            if not np.any(vis):
                continue
            xk = xk[vis]
            yk = yk[vis]
            pk = pk[vis]
        row_hid = str(s.get("row", {}).get("history_id", "")).strip()
        st = style_by_history.get(row_hid, style_by_index.get(i, {}))
        col = str(st.get("color", color_base[i % len(color_base)]))
        alpha_i = float(
            np.clip(
                _parse_float(st.get("alpha", alpha_default), alpha_default, min_value=0.0),
                0.0,
                1.0,
            )
        )
        marker_i = str(st.get("marker", marker))
        size_i = float(st.get("size", msize))
        high_mask = pk <= float(sig_p_thr)
        low_mask = ~high_mask
        if np.any(low_mask):
            ax_manh.scatter(
                xk[low_mask],
                yk[low_mask],
                s=size_i,
                c=low_sig_color,
                alpha=max(0.45, alpha_i * 0.75),
                marker=marker_i,
                linewidths=0.0,
                rasterized=rasterize_main,
            )
        if np.any(high_mask):
            ax_manh.scatter(
                xk[high_mask],
                yk[high_mask],
                s=size_i,
                c=col,
                alpha=alpha_i,
                marker=marker_i,
                linewidths=0.0,
                rasterized=rasterize_main,
                label=str(s["label"]),
            )
        else:
            # Keep series legend entry even if all points are below threshold.
            ax_manh.scatter(
                [],
                [],
                s=size_i,
                c=col,
                alpha=alpha_i,
                marker=marker_i,
                linewidths=0.0,
                rasterized=rasterize_main,
                label=str(s["label"]),
            )

        # QQ series (same color as manhattan)
        if show_qq and (ax_qq is not None):
            pvals = s["full"]["p"].to_numpy(dtype=float, copy=False)
            pmask = np.isfinite(pvals) & (pvals > 0.0) & (pvals <= 1.0)
            pvals = pvals[pmask]
            n = int(pvals.size)
            if n >= 2:
                obs = -np.log10(np.clip(np.sort(pvals), np.nextafter(0.0, 1.0), 1.0))
                exp = -np.log10((np.arange(1, n + 1, dtype=float) - 0.5) / float(n))
                exp, obs = _downsample_for_qq(exp, obs, max_points=120_000)
                if exp.size > 0:
                    qq_xmax = max(qq_xmax, float(np.nanmax(exp)))
                    ax_qq.scatter(
                        exp,
                        obs,
                        s=max(2.0, size_i * 0.55),
                        c=col,
                        alpha=alpha_i,
                        marker=marker_i,
                        linewidths=0.0,
                        rasterized=rasterize_main,
                    )

    if not np.isfinite(x_left) or not np.isfinite(x_right):
        plt.close(fig)
        raise RuntimeError("No points to draw in merged Manhattan plot.")
    if x_right <= x_left:
        eps = max(1e-9, abs(x_left) * 1e-9)
        x_left -= eps
        x_right += eps

    ax_manh.set_xlim(float(x_left), float(x_right))
    ax_manh.margins(x=0.0)
    if full_point_mode:
        y_lo = 0.0
    else:
        y_lo = float(_LARGE_MATRIX_LOGP_FLOOR)
    if not np.isfinite(y_top):
        y_top = y_lo + 0.2
    y_top = max(float(y_top), float(sig_y_thr), y_lo + 0.2)
    y_span = max(0.2, float(y_top) - y_lo)
    y_hi = float(y_top) + 0.06 * y_span
    if not np.isfinite(y_hi) or y_hi <= y_lo:
        y_hi = y_lo + 0.5
    manh_ylim = (y_lo, y_hi)
    ax_manh.set_ylim(manh_ylim)
    ax_manh.set_ylabel("-log10(P)")
    if len(brs) == 0:
        ax_manh.set_xlabel("Chromosome")
        if len(xticks) > 0:
            ax_manh.set_xticks([float(x) for x, _c in xticks])
            ax_manh.set_xticklabels([str(c) for _x, c in xticks], rotation=0)
    else:
        ax_manh.set_xlabel("")
        ax_manh.set_xticks([])
        ax_manh.tick_params(axis="x", which="both", length=0, labelbottom=False)
    for b in separators:
        ax_manh.axvline(
            float(b),
            ymin=0.0,
            ymax=1.0,
            linestyle="--",
            color="#d1d5db",
            linewidth=0.6,
            alpha=0.9,
            zorder=8,
        )
    ax_manh.axhline(
        float(sig_y_thr),
        linestyle="--",
        color="#9ca3af",
        linewidth=0.9,
        alpha=0.95,
        zorder=9,
    )
    if show_ld:
        ax_manh.set_aspect("auto")
    else:
        ax_manh.set_box_aspect(1.0 / float(ratio))

    # QQ axis
    if show_qq and (ax_qq is not None):
        x_left_qq = max(0.0, 0.0 - 0.02 * qq_xmax)
        x_right_qq = float(qq_xmax) * 1.02
        if not np.isfinite(x_right_qq) or x_right_qq <= x_left_qq:
            x_right_qq = x_left_qq + 1.0
        ax_qq.set_xlim(x_left_qq, x_right_qq)
        ax_qq.set_ylim(manh_ylim)
        # Keep QQ panel height aligned with Manhattan while preserving
        # QQ box ratio style (historically square).
        try:
            _sync_qq_panel_with_manhattan_height(ax_manh, ax_qq, qq_box_ratio=1.0)
        except Exception:
            pass
        line_left, line_right = ax_qq.get_xlim()
        ax_qq.plot([line_left, line_right], [line_left, line_right], color="#9ca3af", linewidth=1.0)
        ax_qq.set_xlabel("Expected -log10(P)")
        ax_qq.set_ylabel("Observed -log10(P)")

    # Transition bridge panel in merge+bimrange mode.
    if show_ld and (ax_bridge is not None):
        ax_bridge.set_xticks([])
        ax_bridge.set_yticks([])
        ax_bridge.tick_params(axis="both", which="both", length=0)
        for sp in ax_bridge.spines.values():
            sp.set_visible(False)
        # Label each bimrange at top-center of transition panel.
        x0, x1 = ax_manh.get_xlim()
        span = float(x1 - x0)
        if np.isfinite(span) and span > 0.0:
            if len(brs) == 1:
                br = brs[0]
                centers: list[tuple[float, Bimrange]] = [
                    ((float(br.start_bp) + float(br.end_bp)) / 2.0 / 1e6, br)
                ]
            else:
                centers = [((float(seg.x_start) + float(seg.x_end)) * 0.5, seg.br) for seg in segs]
            for cx, br in centers:
                x_norm = (float(cx) - float(x0)) / span
                if not np.isfinite(x_norm):
                    continue
                x_norm = float(np.clip(x_norm, 0.02, 0.98))
                lbl = f"{br.chrom}:{br.start_bp/1e6:g}Mb:{br.end_bp/1e6:g}Mb"
                ax_bridge.text(
                    x_norm,
                    0.985,
                    lbl,
                    transform=ax_bridge.transAxes,
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    color=str(tone["dark"]),
                )

    # Legend
    leg = ax_manh.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, min(6, len(series))),
        frameon=False,
        fontsize=8,
        markerscale=2.0,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    if leg is not None:
        leg.set_title(None)

    # LD block in merge+bimrange mode.
    ld_n = 0
    ld_has_matrix = False
    if show_ld and ax_ld is not None:
        selected_keys: set[tuple[str, int]] | None = None
        if 0.0 < float(ld_thr) < 1.0:
            sig_sets: list[set[tuple[str, int]]] = []
            for s in series:
                dfx = _filter_by_bimranges(s["full"], brs)
                if int(dfx.shape[0]) == 0:
                    sig_sets.append(set())
                    continue
                m = dfx["p"].values.astype(float) <= float(ld_thr)
                if not np.any(m):
                    sig_sets.append(set())
                    continue
                cs = dfx.loc[m, "chrom_norm"].astype(str).to_numpy(dtype=object)
                ps = dfx.loc[m, "pos"].to_numpy(dtype=np.int64, copy=False)
                sig_sets.append(set((str(c), int(p)) for c, p in zip(cs, ps)))
            if len(sig_sets) > 0:
                inter = set(sig_sets[0])
                for st in sig_sets[1:]:
                    inter &= st
                    if len(inter) == 0:
                        break
                if len(inter) > 0:
                    selected_keys = inter

        ref_row = series[0]["row"] if len(series) > 0 else rows[0]
        ld_r2, ld_keys, ld_msg = _compute_ld_r2(
            ref_row,
            brs,
            selected_keys=selected_keys,
            gwas_token="|".join([str(s["path"]) for s in series]),
            ld_p_threshold=float(ld_thr),
        )
        if ld_r2 is None:
            ax_ld.text(
                0.5,
                0.5,
                ld_msg if ld_msg else "LD block unavailable",
                ha="center",
                va="center",
                fontsize=9,
                color=str(tone["dark"]),
                transform=ax_ld.transAxes,
            )
            ax_ld.set_xticks([])
            ax_ld.set_yticks([])
            for sp in ax_ld.spines.values():
                sp.set_visible(False)
        else:
            LDblock(
                np.array(ld_r2, dtype=float, copy=True),
                ax_ld,
                vmin=0.0,
                vmax=1.0,
                cmap=tone["cmap"],
                rasterized=rasterize_main,
                rasterize_threshold=120 if rasterize_main else None,
            )
            ld_has_matrix = True
            # Draw merge bridge lines: Manhattan SNP position -> LD block order.
            if ax_bridge is not None:
                try:
                    pairs: list[tuple[float, float]] = []
                    # Build x mapping from LD keys to Manhattan x coordinates.
                    if len(brs) == 1:
                        n_ld = int(ld_r2.shape[0])
                        for i, k0 in enumerate(ld_keys[:n_ld]):
                            c0 = _normalize_chrom(k0[0])
                            if c0 != brs[0].chrom:
                                continue
                            x_top = float(k0[1]) / 1e6
                            x_ld = _ld_bridge_x_from_index(i, n_ld)
                            pairs.append((x_top, x_ld))
                    else:
                        seg_by_chrom = {}
                        for seg in segs:
                            seg_by_chrom.setdefault(seg.br.chrom, []).append(seg)
                        n_ld = int(ld_r2.shape[0])
                        for i, k0 in enumerate(ld_keys[:n_ld]):
                            c0 = _normalize_chrom(k0[0])
                            p0 = int(k0[1])
                            x_top = None
                            for seg in seg_by_chrom.get(c0, []):
                                if int(seg.br.start_bp) <= p0 <= int(seg.br.end_bp):
                                    x_top = float(seg.x_start) + (float(p0) - float(seg.br.start_bp)) / 1e6
                                    break
                            if x_top is None:
                                continue
                            x_ld = _ld_bridge_x_from_index(i, n_ld)
                            pairs.append((float(x_top), x_ld))
                    if len(pairs) > 0:
                        to_bridge = ax_bridge.transAxes.inverted()
                        edge_margin_n = 0.006
                        y_top_ref = float(ax_manh.get_ylim()[0])
                        y_ld_ref = 0.0
                        for x_top, x_ld in pairs:
                            p_top_disp = ax_manh.transData.transform((float(x_top), y_top_ref))
                            p_ld_disp = ax_ld.transData.transform((float(x_ld), y_ld_ref))
                            x_top_n = float(to_bridge.transform(p_top_disp)[0])
                            x_ld_n = float(to_bridge.transform(p_ld_disp)[0])
                            if not (np.isfinite(x_top_n) and np.isfinite(x_ld_n)):
                                continue
                            x_top_n = float(np.clip(x_top_n, edge_margin_n, 1.0 - edge_margin_n))
                            x_ld_n = float(np.clip(x_ld_n, edge_margin_n, 1.0 - edge_margin_n))
                            ax_bridge.plot(
                                [x_top_n, x_top_n, x_ld_n],
                                [1.04, 0.82, 0.06],
                                color=str(tone["dark"]),
                                linewidth=0.35,
                                alpha=0.85,
                                transform=ax_bridge.transAxes,
                                clip_on=False,
                                solid_joinstyle="round",
                            )
                except Exception:
                    pass
            ld_n = int(ld_r2.shape[0])

    # Final hard alignment: Manhattan / bridge / LD share exact horizontal box.
    if show_ld and (ax_ld is not None):
        try:
            manh_pos = ax_manh.get_position()
            if ax_bridge is not None:
                bridge_pos = ax_bridge.get_position()
                ax_bridge.set_position([manh_pos.x0, bridge_pos.y0, manh_pos.width, bridge_pos.height])
            ld_pos = ax_ld.get_position()
            ax_ld.set_adjustable("box")
            ax_ld.set_position([manh_pos.x0, ld_pos.y0, manh_pos.width, ld_pos.height])
            if ld_has_matrix:
                ax_ld.set_aspect(0.5, adjustable="box")
                ax_ld.set_anchor("N")
        except Exception:
            pass

    buf = StringIO()
    fig.savefig(buf, format="svg", transparent=True)
    plt.close(fig)
    svg = buf.getvalue()
    if show_ld:
        mode = "merged_manhattan_ld"
    else:
        mode = "merged_manhattan_qq" if show_qq else "merged_manhattan"
    info = {
        "n_tasks": int(len(series)),
        "palette": str(p_use),
        "manh_alpha": float(alpha_default),
        "threshold": float(sig_p_thr),
        "threshold_logp": float(sig_y_thr),
        "include_all_points": bool(full_point_mode),
        "n_total": int(sum(int(s["full"].shape[0]) for s in series)),
        "legend": [str(s["label"]) for s in series],
        "mode": mode,
        "ld_nsnps": int(ld_n),
        "ld_p_threshold": float(ld_thr),
        "ld_p_threshold_auto": bool(ld_auto),
        "bimrange": "; ".join([b.label for b in brs]),
        "load_sec": float(load_sec_total),
        "cache_hits_render": int(sum(1 for v in cache_src_by_key.values() if v == "render_cache")),
        "cache_hits_sig": int(sum(1 for v in cache_src_by_key.values() if v == "sig_cache")),
        "cache_hits_disk": int(sum(1 for v in cache_src_by_key.values() if v == "disk_cache")),
        "cache_miss_disk": int(sum(1 for v in cache_src_by_key.values() if v == "disk")),
        "rust_loader_available": bool(_load_gwas_triplet_fast is not None),
        "render_sec": float(time.perf_counter() - t_render0),
    }
    return svg, info


def render_single_svg(
    row: dict[str, Any],
    *,
    bimrange_text: Any = "",
    anno_file: str = "",
    highlight_ranges: Any = None,
    full_plot: bool = False,
    manh_ratio: float = 2.0,
    manh_palette: str = "auto",
    manh_alpha: float = 0.7,
    manh_marker: str = "o",
    manh_size: float = 16.0,
    ld_color: str = "#4b5563",
    ld_p_threshold: Any = "auto",
    editable_svg: bool = False,
    cache: dict[str, tuple[float, pd.DataFrame, dict[str, str]]] | None = None,
) -> tuple[str, dict[str, Any]]:
    t_render0 = time.perf_counter()
    gwas_path = _resolve_result_file(row)
    table, cols, cache_source, load_sec = _load_table_cached(gwas_path, cache=cache)

    phenotype_name = str(row.get("phenotype", "")).strip() or "Phenotype"
    n_idv = _coerce_pos_int(row.get("nidv", None))
    n_snp = _coerce_pos_int(row.get("eff_snp", None))

    # WebUI rendering path keeps latency low: avoid probing huge genotype files
    # on first render. Only use already-cached genotype meta when available.
    geno_n_idv, geno_n_snp = (None, None)
    genofile_txt = str(row.get("genofile", "")).strip()
    if n_idv is None:
        n_idv = _estimate_n_from_phenofile(str(row.get("phenofile", "")).strip())
    if (n_idv is None or n_snp is None) and genofile_txt != "":
        ck_n = f"geno_n::{genofile_txt}"
        ck_s = f"geno_snp::{genofile_txt}"
        n_cached = _META_CACHE.get(ck_n)
        s_cached = _META_CACHE.get(ck_s)
        geno_n_idv = int(n_cached) if n_cached is not None and int(n_cached) > 0 else None
        geno_n_snp = int(s_cached) if s_cached is not None and int(s_cached) > 0 else None
    if n_idv is None:
        n_idv = geno_n_idv
    if n_snp is None:
        n_snp = geno_n_snp
    if n_snp is None:
        n_snp = int(table.shape[0]) if table.shape[0] > 0 else None
    n_text = str(int(n_idv)) if n_idv is not None else "NA"
    snp_text = _nsnp_label(int(n_snp)) if n_snp is not None else "NA"
    manh_xlabel = f"{phenotype_name} (n={n_text},snp={snp_text})"

    brs = _parse_bimranges(bimrange_text)
    highlights = _parse_highlight_ranges(highlight_ranges)
    df = _filter_by_bimranges(table, brs)
    tone = _derive_tone_palette(ld_color)
    ratio = _parse_float(manh_ratio, 2.0, min_value=0.2)
    alpha_main = float(np.clip(_parse_float(manh_alpha, 0.7, min_value=0.0), 0.0, 1.0))
    marker = _normalize_marker(manh_marker)
    msize = _parse_float(manh_size, 16.0, min_value=0.1)
    # Single-GWAS mode: use same-hue high-contrast (dark/light) colors.
    manh_color_set = _resolve_single_contrast_manh_colors(
        manh_palette=manh_palette,
        ld_color=ld_color,
        tone=tone,
    )
    ld_auto = False
    raw_thr = str(ld_p_threshold).strip().lower() if ld_p_threshold is not None else ""
    if raw_thr in {"", "auto", "none", "default"}:
        ld_auto = True
        if int(df.shape[0]) > 1000:
            ld_thr = 1e-4
        else:
            ld_thr = 1.0
    else:
        try:
            ld_thr = float(ld_p_threshold)
        except Exception:
            ld_thr = 1e-4
    if not np.isfinite(ld_thr) or ld_thr <= 0.0:
        ld_thr = 1e-4
    if ld_thr > 1.0:
        ld_thr = 1.0
    p_filter_text = "full"
    use_threshold_floor = False
    if not bool(full_plot):
        # Single preview mode: draw only SNPs above the default threshold line.
        # Keep full-point rendering for download mode (full_plot=True).
        p_filter_text = f"p<={_LARGE_MATRIX_P_THRESH:g}"
        df = df.loc[df["p"].values <= float(_LARGE_MATRIX_P_THRESH)].copy()
        use_threshold_floor = True
    if df.shape[0] == 0:
        raise RuntimeError("No SNPs left after applying bimrange.")
    n_draw = int(df.shape[0])
    # Keep scatter-heavy artists rasterized for both preview and download.
    # Text/axes/labels remain vector in SVG output.
    rasterize_main = True
    fig_dpi = int(_WEBUI_RASTER_DPI)

    show_track = len(brs) > 0
    show_qq = (len(brs) == 0) and bool(full_plot)
    if len(brs) == 0:
        if show_qq:
            # No bimrange + full mode: Manhattan + QQ with strict layout:
            # [A, A, B] -> manhattan:qq width = 2:1, same row -> same y-axis height.
            fig_w = 15.0
            fig_h = max(3.0, (fig_w * (2.0 / 3.0)) / float(ratio))
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=fig_dpi)
            mosaic = fig.subplot_mosaic([["A", "A", "B"]])
            ax_manh = mosaic["A"]
            ax_qq = mosaic["B"]
            fig.subplots_adjust(
                left=0.045,
                right=0.995,
                top=0.96,
                bottom=0.12,
                wspace=0.085,
            )
        else:
            # No bimrange + non-full mode: Manhattan only (skip QQ for speed).
            fig_w = 10.0
            fig_h = max(3.0, fig_w / float(ratio))
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=fig_dpi)
            ax_manh = fig.add_subplot(111)
            fig.subplots_adjust(
                left=0.055,
                right=0.995,
                top=0.965,
                bottom=0.12,
            )
            ax_qq = None
        ax_gene = None
        ax_empty1 = None
        ax_ld = None
        ax_empty2 = None
    else:
        # With bimrange: render Manhattan + Gene + LD (single-column).
        # Keep transition layer compact: height ~= width / 15.
        # For fixed figure size (9.8 x 12.8), this maps to a GridSpec mid ratio
        # close to 0.11 (previous 0.15 was visibly taller).
        # Keep Manhattan panel responsive to larger ratio values in bimrange mode.
        # 2 -> 1.0, 4 -> 0.5, 8 -> 0.25, 20 -> 0.10
        # Clamp only to avoid collapsing completely.
        fig_w = 9.8
        left = 0.055
        right = 0.995
        top = 0.985
        bottom = 0.05
        fig_h, h_ratios = _ld_layout_for_webui(
            fig_w=fig_w,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            manh_ratio=float(ratio),
        )
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=fig_dpi)
        gs = GridSpec(
            3,
            1,
            figure=fig,
            height_ratios=h_ratios,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            hspace=0.018,
            wspace=0.0,
        )
        ax_manh = fig.add_subplot(gs[0, 0])
        ax_qq = None
        ax_gene = fig.add_subplot(gs[1, 0])
        ax_empty1 = None
        ax_ld = fig.add_subplot(gs[2, 0])
        ax_empty2 = None

    def _strip_panel(ax: plt.Axes) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Manhattan
    y = np.asarray(df["y"].values, dtype=float)
    bounds: list[float] = []
    segs: list[Segment] = []
    plotmodel_ui: GWASPLOT | None = None
    manh_highlight_x: list[np.ndarray] = []
    manh_highlight_y: list[np.ndarray] = []
    highlight_points_n = 0
    if len(brs) == 0:
        # Reuse GWASPLOT compression/minidx pipeline for large GWAS tables.
        # This keeps highly significant SNPs while downsampling the rest.
        df_plot = pd.DataFrame(
            {
                "chrom": df["chrom_norm"].astype(str).values,
                "pos": pd.to_numeric(df["pos"], errors="coerce").fillna(0).astype(np.int64).values,
                "pwald": pd.to_numeric(df["p"], errors="coerce").fillna(1.0).values,
            }
        )
        plotmodel_ui = GWASPLOT(
            df_plot,
            chr="chrom",
            pos="pos",
            pvalue="pwald",
            interval_rate=0.0,
            compression=True,
        )
        plotmodel_ui.manhattan(
            threshold=None,
            color_set=manh_color_set,
            ax=ax_manh,
            min_logp=0.0,
            y_min=0.0,
            s=msize,
            marker=marker,
            rasterized=rasterize_main,
        )
        for coll in ax_manh.collections:
            try:
                coll.set_alpha(alpha_main)
            except Exception:
                pass
        # WebUI: no chromosome gap; use full-height dashed separators.
        for xsep in np.asarray(getattr(plotmodel_ui, "_chr_separators", []), dtype=float):
            if np.isfinite(xsep):
                ax_manh.axvline(
                    float(xsep),
                    ymin=0.0,
                    ymax=1.0,
                    linestyle="--",
                    color="#d1d5db",
                    linewidth=0.6,
                    alpha=0.9,
                    zorder=8,
                )
        ax_manh.set_xlabel(manh_xlabel)
        if len(highlights) > 0:
            try:
                dfx = plotmodel_ui.df.iloc[plotmodel_ui.minidx, -3:].copy()
                dfx["y"] = -np.log10(np.clip(dfx["y"].values.astype(float), np.nextafter(0.0, 1.0), 1.0))
                if dfx.shape[0] > 0:
                    label_to_id = {
                        _normalize_chrom(lbl): int(i + 1)
                        for i, lbl in enumerate(getattr(plotmodel_ui, "chr_labels", []))
                    }
                    chr_vals = dfx["z"].to_numpy(dtype=int, copy=False)
                    x_vals = dfx["x"].to_numpy(dtype=float, copy=False)
                    y_vals = dfx["y"].to_numpy(dtype=float, copy=False)
                    pos_vals = dfx.index.get_level_values(1).to_numpy(dtype=float, copy=False)
                    for hr in highlights:
                        cid = label_to_id.get(_normalize_chrom(hr.chrom))
                        if cid is None:
                            continue
                        mask = (
                            (chr_vals == int(cid))
                            & (pos_vals >= float(hr.start_bp))
                            & (pos_vals <= float(hr.end_bp))
                        )
                        if len(hr.ld_keys) > 0:
                            pos_allow = np.asarray(
                                [int(p) for c0, p in hr.ld_keys if _normalize_chrom(c0) == _normalize_chrom(hr.chrom)],
                                dtype=np.int64,
                            )
                            if pos_allow.size == 0:
                                continue
                            mask &= np.isin(pos_vals.astype(np.int64, copy=False), pos_allow)
                        if not np.any(mask):
                            continue
                        xs = x_vals[mask]
                        ys = y_vals[mask]
                        if xs.size == 0:
                            continue
                        manh_highlight_x.append(xs)
                        manh_highlight_y.append(ys)
            except Exception:
                pass
    elif len(brs) == 1:
        br = brs[0]
        x = np.asarray(df["pos"].values, dtype=float) / 1e6
        ax_manh.scatter(
            x,
            y,
            s=msize,
            c=manh_color_set[0] if len(manh_color_set) > 0 else "#111111",
            alpha=alpha_main,
            marker=marker,
            linewidths=0.0,
            rasterized=rasterize_main,
        )
        ax_manh.set_xlabel("")
        if len(highlights) > 0:
            for hr in highlights:
                if _normalize_chrom(hr.chrom) != br.chrom:
                    continue
                ov_s = max(int(br.start_bp), int(hr.start_bp))
                ov_e = min(int(br.end_bp), int(hr.end_bp))
                if ov_e < ov_s:
                    continue
                mask = (
                    (df["chrom_norm"].values.astype(str) == br.chrom)
                    & (df["pos"].values.astype(float) >= float(ov_s))
                    & (df["pos"].values.astype(float) <= float(ov_e))
                )
                if len(hr.ld_keys) > 0:
                    pos_allow = np.asarray(
                        [int(p) for c0, p in hr.ld_keys if _normalize_chrom(c0) == br.chrom],
                        dtype=np.int64,
                    )
                    if pos_allow.size == 0:
                        continue
                    mask &= np.isin(df["pos"].values.astype(np.int64, copy=False), pos_allow)
                if np.any(mask):
                    manh_highlight_x.append(x[mask])
                    manh_highlight_y.append(y[mask])
    else:
        segs = _build_segments(brs, gap_mb=0.0)
        x_all, seg_idx_all = _compute_x_for_segments(df, segs)
        keep = seg_idx_all >= 0
        if not np.any(keep):
            raise RuntimeError("No SNPs left after applying bimrange.")
        df = df.loc[keep].copy()
        x = x_all[keep]
        seg_idx = seg_idx_all[keep]
        order = np.lexsort((df["pos"].values.astype(float), seg_idx))
        df = df.iloc[order].copy()
        x = x[order]
        y = np.asarray(df["y"].values, dtype=float)
        seg_idx = seg_idx[order]
        for i, seg in enumerate(segs):
            m = seg_idx == i
            if not np.any(m):
                continue
            color = manh_color_set[i % max(1, len(manh_color_set))]
            ax_manh.scatter(
                x[m],
                y[m],
                s=msize,
                c=color,
                alpha=alpha_main,
                marker=marker,
                linewidths=0.0,
                rasterized=rasterize_main,
            )
        mids = [((seg.x_start + seg.x_end) * 0.5, seg.br.chrom) for seg in segs]
        bounds = [seg.x_end for seg in segs[:-1]]
        ax_manh.set_xlabel("")
        if len(highlights) > 0:
            for hr in highlights:
                hchr = _normalize_chrom(hr.chrom)
                for seg in segs:
                    if seg.br.chrom != hchr:
                        continue
                    ov_s = max(int(seg.br.start_bp), int(hr.start_bp))
                    ov_e = min(int(seg.br.end_bp), int(hr.end_bp))
                    if ov_e < ov_s:
                        continue
                    x0 = float(seg.x_start) + (float(ov_s) - float(seg.br.start_bp)) / 1e6
                    x1 = float(seg.x_start) + (float(ov_e) - float(seg.br.start_bp)) / 1e6
                    mask = (
                        (df["chrom_norm"].values.astype(str) == seg.br.chrom)
                        & (df["pos"].values.astype(float) >= float(ov_s))
                        & (df["pos"].values.astype(float) <= float(ov_e))
                    )
                    if len(hr.ld_keys) > 0:
                        pos_allow = np.asarray(
                            [int(p) for c0, p in hr.ld_keys if _normalize_chrom(c0) == seg.br.chrom],
                            dtype=np.int64,
                        )
                        if pos_allow.size == 0:
                            continue
                        mask &= np.isin(df["pos"].values.astype(np.int64, copy=False), pos_allow)
                    if np.any(mask):
                        manh_highlight_x.append(x[mask])
                        manh_highlight_y.append(y[mask])
    if len(brs) > 0:
        x_left = float(np.nanmin(x))
        x_right = float(np.nanmax(x))
        if not np.isfinite(x_left) or not np.isfinite(x_right):
            x_left, x_right = (0.0, 1.0)
        if x_right <= x_left:
            eps = max(1e-9, abs(x_left) * 1e-9)
            x_left -= eps
            x_right += eps
        ax_manh.set_xlim(x_left, x_right)
        ax_manh.margins(x=0.0)

    y_top = float(np.nanmax(y)) if y.size > 0 else 1.0
    if not np.isfinite(y_top) or y_top <= 0:
        y_top = 1.0
    y_floor = 0.0
    if use_threshold_floor:
        y_floor = float(-np.log10(float(_LARGE_MATRIX_P_THRESH)))
    if y_top <= y_floor:
        y_top = y_floor + 0.5
    manh_ylim = (y_floor, y_top * 1.03)
    ax_manh.set_ylim(manh_ylim)
    # LD selection threshold line for bimrange mode.
    if len(brs) > 0 and 0.0 < ld_thr < 1.0:
        y_thr = float(-np.log10(ld_thr))
        if np.isfinite(y_thr):
            ax_manh.axhline(
                y=y_thr,
                linestyle="--",
                color="#9ca3af",
                linewidth=0.8,
                alpha=0.9,
                zorder=9,
            )

    if len(bounds) > 0:
        for b in bounds:
            ax_manh.axvline(
                float(b),
                ymin=0.0,
                ymax=1.0,
                linestyle="--",
                color="#d1d5db",
                linewidth=0.6,
                alpha=0.9,
                zorder=8,
            )
    if len(manh_highlight_x) > 0 and len(manh_highlight_y) > 0:
        try:
            hx = np.concatenate([np.asarray(a, dtype=float) for a in manh_highlight_x if np.asarray(a).size > 0])
            hy = np.concatenate([np.asarray(a, dtype=float) for a in manh_highlight_y if np.asarray(a).size > 0])
            if hx.size > 0 and hy.size > 0:
                order = np.argsort(hx, kind="mergesort")
                hx_s = hx[order]
                hy_s = hy[order]
                ux, idx = np.unique(hx_s, return_index=True)
                uy = np.maximum.reduceat(hy_s, idx)
                ax_manh.scatter(
                    ux,
                    uy,
                    s=max(float(msize) * 2.0, 8.0),
                    c="#dc2626",
                    alpha=0.95,
                    marker=marker,
                    linewidths=0.0,
                    rasterized=False,
                    zorder=13,
                )
                highlight_points_n = int(ux.size)
        except Exception:
            pass

    # Keep Manhattan aspect controllable by user ratio.
    # In bimrange mode, avoid fixed box-aspect shrink causing extra blank gap.
    if len(brs) == 0:
        ax_manh.set_box_aspect(1.0 / float(ratio))
    else:
        ax_manh.set_aspect("auto")
    ax_manh.set_ylabel("-log10(P)")
    if len(brs) > 0:
        ax_manh.set_xlabel("")
        ax_manh.set_xticks([])
        ax_manh.tick_params(axis="x", which="both", length=0, labelbottom=False)

    # QQ (disabled when bimrange is set)
    if show_qq and ax_qq is not None:
        if plotmodel_ui is None:
            df_plot = pd.DataFrame(
                {
                    "chrom": df["chrom_norm"].astype(str).values,
                    "pos": pd.to_numeric(df["pos"], errors="coerce").fillna(0).astype(np.int64).values,
                    "pwald": pd.to_numeric(df["p"], errors="coerce").fillna(1.0).values,
                }
            )
            plotmodel_ui = GWASPLOT(
                df_plot,
                chr="chrom",
                pos="pos",
                pvalue="pwald",
                interval_rate=0.0,
                compression=True,
            )
        # Reuse QQ auto/full/fast path from manhanden.py for large data.
        plotmodel_ui.qq(
            ax=ax_qq,
            color_set=["#111111", "#8a8a8a"],
            line_color="#9ca3af",
            scatter_size=8.0,
            qq_mode="auto",
            qq_auto_threshold=1_000_000,
            qq_fast_max_points=120_000,
            qq_band_max_points=20_000,
        )
        for coll in ax_qq.collections:
            try:
                coll.set_alpha(alpha_main)
            except Exception:
                pass
        # Keep QQ ylim identical to Manhattan ylim.
        # Keep QQ xlim adaptive to QQ expected range.
        qq_xmin, qq_xmax = ax_qq.get_xlim()
        x_left = float(qq_xmin)
        x_right = float(qq_xmax)
        if not np.isfinite(x_left):
            x_left = 0.0
        if not np.isfinite(x_right) or x_right <= x_left:
            x_right = x_left + 1.0
        x_pad = max(1e-9, 0.02 * float(x_right - x_left))
        x_left = max(0.0, x_left - x_pad)
        x_right = x_right + x_pad
        ax_qq.set_xlim(x_left, x_right)
        ax_qq.margins(x=0.0)
        ax_qq.set_ylim(manh_ylim)
        # Keep QQ panel height aligned with Manhattan while preserving
        # QQ box ratio style (historically square).
        try:
            _sync_qq_panel_with_manhattan_height(ax_manh, ax_qq, qq_box_ratio=1.0)
        except Exception:
            pass
        line_left, line_right = ax_qq.get_xlim()
        ax_qq.plot([line_left, line_right], [line_left, line_right], color="#9ca3af", linewidth=1.0)
        ax_qq.set_xlabel("Expected -log10(P)")
        ax_qq.set_ylabel("Observed -log10(P)")

    def _draw_bridge_overlay(
        ax_bridge: plt.Axes,
        ax_top: plt.Axes,
        ax_bottom: plt.Axes,
        pairs: list[tuple[float, float]],
        *,
        line_color: str = "#6b7280",
    ) -> None:
        if len(pairs) == 0:
            return
        to_bridge = ax_bridge.transAxes.inverted()
        edge_margin_n = 0.006
        y_top_ref = float(ax_top.get_ylim()[0])
        y_ld_ref = 0.0
        for x_top, x_ld in pairs:
            p_top_disp = ax_top.transData.transform((float(x_top), y_top_ref))
            p_ld_disp = ax_bottom.transData.transform((float(x_ld), y_ld_ref))
            x_top_n = float(to_bridge.transform(p_top_disp)[0])
            x_ld_n = float(to_bridge.transform(p_ld_disp)[0])
            if not (np.isfinite(x_top_n) and np.isfinite(x_ld_n)):
                continue
            x_top_n = float(np.clip(x_top_n, edge_margin_n, 1.0 - edge_margin_n))
            x_ld_n = float(np.clip(x_ld_n, edge_margin_n, 1.0 - edge_margin_n))
            ax_bridge.plot(
                [x_top_n, x_top_n, x_ld_n],
                [1.04, 0.82, 0.06],
                color=line_color,
                linewidth=0.35,
                alpha=0.85,
                transform=ax_bridge.transAxes,
                clip_on=False,
                solid_joinstyle="round",
            )

    ld_r2 = None
    ld_keys: list[tuple[str, int]] = []
    ld_has_matrix = False
    if show_track and ax_gene is not None and ax_ld is not None:
        # Gene panel
        ax_gene.set_yticks([])
        mapped_records: list[dict[str, Any]] = []
        if len(brs) == 1:
            br = brs[0]
            feats = _read_gene_track(anno_file, br)
            for rec in feats:
                s = int(rec.get("start", 0))
                e = int(rec.get("end", 0))
                mapped_records.append(
                    {
                        "x0": float(s) / 1e6,
                        "x1": float(e) / 1e6,
                        "name": str(rec.get("name", "")),
                        "feature": str(rec.get("feature", "gene")).lower(),
                        "parent": str(rec.get("parent", rec.get("name", ""))),
                        "strand": str(rec.get("strand", "+")),
                    }
                )
        else:
            if len(segs) == 0:
                segs = _build_segments(brs, gap_mb=0.0)
            for seg in segs:
                feats = _read_gene_track(anno_file, seg.br)
                for rec in feats:
                    s = int(rec.get("start", 0))
                    e = int(rec.get("end", 0))
                    xs = float(seg.x_start) + (float(s) - float(seg.br.start_bp)) / 1e6
                    xe = float(seg.x_start) + (float(e) - float(seg.br.start_bp)) / 1e6
                    mapped_records.append(
                        {
                            "x0": xs,
                            "x1": xe,
                            "name": str(rec.get("name", "")),
                            "feature": str(rec.get("feature", "gene")).lower(),
                            "parent": str(rec.get("parent", rec.get("name", ""))),
                            "strand": str(rec.get("strand", "+")),
                        }
                    )

        if len(mapped_records) > 0:
            row_slots = 6
            row_by_gene: dict[str, int] = {}
            gene_segments: list[tuple[float, float, float, float]] = []

            def _norm_gene_key(raw: object) -> str:
                txt = str(raw or "").strip()
                if txt == "":
                    return ""
                # Parent may contain multiple IDs.
                txt = txt.split(",")[0].strip()
                # Normalize common prefixes while preserving stable matching.
                if ":" in txt:
                    prefix, rest = txt.split(":", 1)
                    p = prefix.lower()
                    if p in {"gene", "transcript", "mrna", "rna"} and rest.strip() != "":
                        txt = rest.strip()
                return txt.lower()

            def _pick_row_for_feature(
                *,
                parent_key: str,
                name_key: str,
                x0: float,
                x1: float,
                fallback_row: float,
            ) -> float:
                keys = [
                    str(parent_key or "").strip(),
                    str(name_key or "").strip(),
                    _norm_gene_key(parent_key),
                    _norm_gene_key(name_key),
                ]
                for k in keys:
                    if k and k in row_by_gene:
                        return float(row_by_gene[k])
                if len(gene_segments) == 0:
                    return float(fallback_row)

                c = 0.5 * (float(x0) + float(x1))
                best_overlap = -1.0
                best_dist = float("inf")
                best_row = float(fallback_row)
                for gy, gx0, gx1, gc in gene_segments:
                    ov = max(0.0, min(float(x1), gx1) - max(float(x0), gx0))
                    dist = abs(float(c) - float(gc))
                    if ov > best_overlap + 1e-12:
                        best_overlap = ov
                        best_dist = dist
                        best_row = float(gy)
                    elif abs(ov - best_overlap) <= 1e-12 and dist < best_dist:
                        best_dist = dist
                        best_row = float(gy)
                return float(best_row)

            genes = [r for r in mapped_records if r.get("feature") == "gene"]
            genes.sort(key=lambda r: (float(r.get("x0", 0.0)), float(r.get("x1", 0.0))))
            used_rows = int(max(1, min(int(row_slots), len(genes) if len(genes) > 0 else 1)))
            row_offset = int(max(0, (int(row_slots) - int(used_rows)) // 2))
            for i, r in enumerate(genes):
                yrow = int(row_offset + (i % used_rows))
                g_parent = str(r.get("parent") or "")
                g_name = str(r.get("name") or "")
                gx0 = float(r.get("x0", 0.0))
                gx1 = float(r.get("x1", 0.0))
                gcen = 0.5 * (gx0 + gx1)
                for k in {
                    g_parent,
                    g_name,
                    _norm_gene_key(g_parent),
                    _norm_gene_key(g_name),
                }:
                    if str(k).strip() != "":
                        row_by_gene[str(k).strip()] = int(yrow)
                gene_segments.append((float(yrow), gx0, gx1, gcen))

            n_label = 0
            x_lo, x_hi = ax_manh.get_xlim()
            span = max(1e-6, float(x_hi) - float(x_lo))
            arrow_step = max(0.006, span / 160.0)
            gene_line_w = 1.15
            end_tick_half = 0.09
            cds_h = 0.36
            utr_h = 0.22
            # Draw gene skeleton line + strand arrows (dark).
            for i, r in enumerate(genes):
                yrow = _pick_row_for_feature(
                    parent_key=str(r.get("parent") or ""),
                    name_key=str(r.get("name") or ""),
                    x0=float(r.get("x0", 0.0)),
                    x1=float(r.get("x1", 0.0)),
                    fallback_row=float(row_offset + (i % used_rows)),
                )
                x0 = float(r.get("x0", 0.0))
                x1 = float(r.get("x1", 0.0))
                xs = min(x0, x1)
                xe = max(x0, x1)
                yc = float(yrow) + 0.5
                ax_gene.plot(
                    [xs, xe],
                    [yc, yc],
                    color=str(tone["dark"]),
                    linewidth=gene_line_w,
                    solid_capstyle="round",
                    alpha=0.95,
                )
                ax_gene.plot(
                    [xs, xs],
                    [yc - end_tick_half, yc + end_tick_half],
                    color=str(tone["dark"]),
                    linewidth=gene_line_w,
                    alpha=0.95,
                )
                ax_gene.plot(
                    [xe, xe],
                    [yc - end_tick_half, yc + end_tick_half],
                    color=str(tone["dark"]),
                    linewidth=gene_line_w,
                    alpha=0.95,
                )
                strand = str(r.get("strand", "+")).strip()
                if strand not in {"+", "-"}:
                    strand = "+"
                if (xe - xs) > (arrow_step * 0.66):
                    if strand == "-":
                        for seg_end in np.arange(xe, xs, -arrow_step):
                            seg_start = max(seg_end - arrow_step, xs)
                            arrow = FancyArrowPatch(
                                (float(seg_end), yc),
                                (float(seg_start), yc),
                                arrowstyle="->",
                                mutation_scale=7.0,
                                linewidth=gene_line_w,
                                color=str(tone["dark"]),
                                alpha=0.9,
                            )
                            ax_gene.add_patch(arrow)
                    else:
                        for seg_start in np.arange(xs, xe, arrow_step):
                            seg_end = min(seg_start + arrow_step, xe)
                            arrow = FancyArrowPatch(
                                (float(seg_start), yc),
                                (float(seg_end), yc),
                                arrowstyle="->",
                                mutation_scale=7.0,
                                linewidth=gene_line_w,
                                color=str(tone["dark"]),
                                alpha=0.9,
                            )
                            ax_gene.add_patch(arrow)
                if n_label < 14:
                    ax_gene.text(
                        0.5 * (xs + xe),
                        yc + 0.34,
                        str(r.get("name", "")),
                        fontsize=6,
                        color=str(tone["dark"]),
                        ha="center",
                        va="center",
                        bbox={
                            "facecolor": "white",
                            "alpha": 0.78,
                            "edgecolor": "none",
                            "boxstyle": "round,pad=0.2",
                        },
                    )
                    n_label += 1

            # Draw CDS/UTR blocks (CDS thicker, UTR thinner).
            for i, r in enumerate(mapped_records):
                ft = str(r.get("feature", "gene")).lower()
                if ft not in {"cds", "utr", "five_prime_utr", "three_prime_utr"}:
                    continue
                x0 = float(r.get("x0", 0.0))
                x1 = float(r.get("x1", 0.0))
                xs = min(x0, x1)
                xe = max(x0, x1)
                yrow = _pick_row_for_feature(
                    parent_key=str(r.get("parent") or ""),
                    name_key=str(r.get("name") or ""),
                    x0=xs,
                    x1=xe,
                    fallback_row=float(row_offset + (i % used_rows)),
                )
                yc = float(yrow) + 0.5
                if ft == "cds":
                    block_h = cds_h
                    face_color = str(tone["mid"])
                else:
                    block_h = utr_h
                    face_color = str(tone["light"])
                rect = Rectangle(
                    (xs, yc - 0.5 * block_h),
                    max(xe - xs, 1e-6),
                    block_h,
                    facecolor=face_color,
                    edgecolor=face_color,
                    linewidth=0.0,
                    alpha=0.96,
                )
                ax_gene.add_patch(rect)
            ax_gene.set_xlim(ax_manh.get_xlim())
            ax_gene.set_ylim(-0.15, float(row_slots) + 0.15)
        else:
            # No annotation: keep a transition-only middle layer.
            ax_gene.set_xlim(0.0, 1.0)
            ax_gene.set_ylim(0.0, 1.0)
        _strip_panel(ax_gene)
        # Label each bimrange at the top-center of transition layer.
        x0, x1 = ax_manh.get_xlim()
        span = float(x1 - x0)
        if np.isfinite(span) and span > 0.0:
            if len(brs) == 1:
                br = brs[0]
                centers: list[tuple[float, Bimrange]] = [
                    ((float(br.start_bp) + float(br.end_bp)) / 2.0 / 1e6, br)
                ]
            else:
                if len(segs) == 0:
                    segs = _build_segments(brs, gap_mb=0.0)
                centers = [
                    ((float(seg.x_start) + float(seg.x_end)) * 0.5, seg.br)
                    for seg in segs
                ]
            for cx, br in centers:
                x_norm = (float(cx) - float(x0)) / span
                if not np.isfinite(x_norm):
                    continue
                x_norm = float(np.clip(x_norm, 0.02, 0.98))
                lbl = f"{br.chrom}:{br.start_bp/1e6:g}Mb:{br.end_bp/1e6:g}Mb"
                ax_gene.text(
                    x_norm,
                    0.985,
                    lbl,
                    transform=ax_gene.transAxes,
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    color=str(tone["dark"]),
                )

        # LD panel (real r2 block when genotype+bimrange are available).
        ld_sel = df.loc[df["p"].values <= float(ld_thr), ["chrom_norm", "pos"]].copy()
        ld_selected_keys: set[tuple[str, int]] = set()
        if ld_sel.shape[0] > 0:
            chr_vals = ld_sel["chrom_norm"].astype(str).to_numpy(dtype=object)
            pos_vals = pd.to_numeric(ld_sel["pos"], errors="coerce").to_numpy(dtype=float)
            for c0, p0 in zip(chr_vals, pos_vals):
                if not np.isfinite(p0):
                    continue
                ld_selected_keys.add((_normalize_chrom(c0), int(round(float(p0)))))

        ld_r2, ld_keys, ld_msg = _compute_ld_r2(
            row,
            brs,
            selected_keys=ld_selected_keys,
            gwas_token=str(gwas_path),
            ld_p_threshold=float(ld_thr),
        )
        if ld_r2 is None:
            ax_ld.text(
                0.5,
                0.5,
                ld_msg if ld_msg else "LD block unavailable",
                ha="center",
                va="center",
                fontsize=9,
                color=str(tone["dark"]),
                transform=ax_ld.transAxes,
            )
            ax_ld.set_xlim(ax_manh.get_xlim())
            _strip_panel(ax_ld)
        else:
            LDblock(
                np.array(ld_r2, dtype=float, copy=True),
                ax_ld,
                vmin=0.0,
                vmax=1.0,
                cmap=tone["cmap"],
                rasterized=rasterize_main,
                rasterize_threshold=120 if rasterize_main else None,
            )
            ld_has_matrix = True
            # Keep LD panel width strictly aligned with Manhattan panel in WebUI.
            try:
                manh_pos = ax_manh.get_position()
                ld_pos = ax_ld.get_position()
                ax_ld.set_adjustable("box")
                ax_ld.set_position([manh_pos.x0, ld_pos.y0, manh_pos.width, ld_pos.height])
            except Exception:
                pass
            # Build Manhattan<->LD mapping pairs and draw transition overlay.
            key_to_x: dict[tuple[str, int], float] = {}
            chrom_vals = df["chrom_norm"].values
            pos_vals = df["pos"].values
            for xi, c0, p0 in zip(x, chrom_vals, pos_vals):
                k0 = (str(c0), int(float(p0)))
                if k0 not in key_to_x:
                    key_to_x[k0] = float(xi)
            n_ld = int(ld_r2.shape[0])
            pairs: list[tuple[float, float]] = []
            for i, k0 in enumerate(ld_keys[:n_ld]):
                x_top = key_to_x.get((str(k0[0]), int(k0[1])))
                if x_top is None:
                    continue
                x_ld = _ld_bridge_x_from_index(i, n_ld)
                pairs.append((float(x_top), x_ld))
            _draw_bridge_overlay(ax_gene, ax_manh, ax_ld, pairs, line_color=str(tone["dark"]))

    for ax in (ax_empty1, ax_empty2):
        if ax is not None:
            ax.axis("off")

    # Final hard alignment for single+bimrange mode:
    # force Gene/LD panels to share the exact same horizontal box as Manhattan.
    if show_track and (ax_gene is not None) and (ax_ld is not None):
        try:
            manh_pos = ax_manh.get_position()
            gene_pos = ax_gene.get_position()
            ld_pos = ax_ld.get_position()
            ax_gene.set_position([manh_pos.x0, gene_pos.y0, manh_pos.width, gene_pos.height])
            ax_ld.set_adjustable("box")
            ax_ld.set_position([manh_pos.x0, ld_pos.y0, manh_pos.width, ld_pos.height])
            if ld_has_matrix:
                ax_ld.set_aspect(0.5, adjustable="box")
                ax_ld.set_anchor("N")
        except Exception:
            pass

    buf = StringIO()
    fig.savefig(buf, format="svg", transparent=True)
    plt.close(fig)
    svg = buf.getvalue()
    info = {
        "result_file": str(gwas_path),
        "n_total": int(table.shape[0]),
        "n_draw": int(df.shape[0]),
        "columns": cols,
        "bimrange": "; ".join([b.label for b in brs]),
        "ld_nsnps": int(ld_r2.shape[0]) if ld_r2 is not None else 0,
        "p_filter": p_filter_text,
        "ld_color": str(ld_color or ""),
        "ld_p_threshold": float(ld_thr),
        "ld_p_threshold_auto": bool(ld_auto),
        "manh_ratio": float(ratio),
        "manh_palette": str(manh_palette or "auto"),
        "manh_alpha": float(alpha_main),
        "manh_marker": str(marker),
        "manh_size": float(msize),
        "highlight_n": int(highlight_points_n),
        "load_sec": float(load_sec),
        "cache_source": str(cache_source),
        "rust_loader_available": bool(_load_gwas_triplet_fast is not None),
        "render_sec": float(time.perf_counter() - t_render0),
    }
    return svg, info
