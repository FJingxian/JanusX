from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
import re
import gzip

import matplotlib as mpl
import numpy as np
import pandas as pd
from janusx.bioplotkit.LDBlock import LDblock
from janusx.bioplotkit.manhanden import GWASPLOT
from janusx.gfreader import load_genotype_chunks, inspect_genotype_file
from janusx.gtools.reader import readanno

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
from matplotlib.patches import Rectangle


_CHROM_CANDIDATES = ["chrom", "#CHROM", "chr", "CHR", "Chromosome", "chromosome"]
_POS_CANDIDATES = ["pos", "POS", "bp", "position", "Position", "BP"]
_P_CANDIDATES = ["pwald", "p", "pvalue", "P", "PVALUE", "p_wald", "p_wald"]
_LD_CACHE: dict[str, tuple[np.ndarray, list[tuple[str, int]]]] = {}
_META_CACHE: dict[str, int] = {}
_ANNO_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_SIG_GWAS_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_TRACK_ANNO_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_ACTIVE_TASK_CACHE: dict[str, str] = {"gwas": "", "anno": ""}
_LARGE_MATRIX_ROWS = 1_000_000
_LARGE_MATRIX_P_THRESH = 1e-3
_WEBUI_RASTER_DPI = 180
_WEBUI_VECTOR_DPI = 150


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


def _pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    cmap = {str(c).strip().lower(): str(c) for c in cols}
    for c in candidates:
        hit = cmap.get(str(c).strip().lower())
        if hit is not None:
            return hit
    return None


def _normalize_chrom(v: object) -> str:
    s = str(v).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    return s


def _natural_key(text: object) -> tuple[int, object]:
    s = _normalize_chrom(text)
    if s.isdigit():
        return (0, int(s))
    up = s.upper()
    if up == "X":
        return (1, 23)
    if up == "Y":
        return (1, 24)
    if up in {"M", "MT"}:
        return (1, 25)
    parts = re.split(r"(\d+)", s)
    out: list[tuple[int, object]] = []
    for p in parts:
        if p == "":
            continue
        if p.isdigit():
            out.append((0, int(p)))
        else:
            out.append((1, p.lower()))
    return (2, tuple(out))


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


def _normalize_marker(raw: Any) -> str:
    m = str(raw or "o").strip()
    allowed = {"o", ".", ",", "x", "+", "s", "^", "v", "D", "*", "P", "X", "h", "H", "d", "p", "<", ">"}
    return m if m in allowed else "o"


def _resolve_manh_color_set(palette: Any, tone: dict[str, Any] | None = None) -> list[str]:
    p = str(palette or "auto").strip().lower()
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

    # auto: use high-contrast tone pair from LD/Gene color picker.
    t = tone or _derive_tone_palette("#4b5563")
    return [str(t.get("dark", "#111111")), str(t.get("light", "#8a8a8a"))]


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
    p = str(row.get("result_file", "")).strip()
    if p != "":
        path = Path(p).expanduser()
        if path.exists():
            return path.resolve()
    run_files = row.get("result_files", [])
    if isinstance(run_files, list):
        for r in run_files:
            rp = str(r).strip()
            if rp == "":
                continue
            path = Path(rp).expanduser()
            if path.exists():
                return path.resolve()
    raise FileNotFoundError("No valid GWAS result file found in history row.")


def activate_task_cache(row: dict[str, Any], *, anno_file: Any = None) -> None:
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
            p = Path(anno_txt).expanduser()
            try:
                anno_key = str(p.resolve())
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


def _load_table(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
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
    c_chr = c_pos = c_p = None
    try:
        sep_kind = _sniff_sep(path)
        head = _read_fast(path, sep_kind, nrows=0)
        c_chr = _pick_existing_column(head, _CHROM_CANDIDATES)
        c_pos = _pick_existing_column(head, _POS_CANDIDATES)
        c_p = _pick_existing_column(head, _P_CANDIDATES)
        if c_chr is not None and c_pos is not None and c_p is not None:
            df = _read_fast(path, sep_kind, usecols=[c_chr, c_pos, c_p])
    except Exception:
        df = None

    if df is None:
        try:
            df_full = pd.read_csv(path, sep=None, engine="python")
        except Exception as exc:
            raise RuntimeError(f"Failed to read GWAS file: {path}") from exc
        c_chr = _pick_existing_column(df_full, _CHROM_CANDIDATES)
        c_pos = _pick_existing_column(df_full, _POS_CANDIDATES)
        c_p = _pick_existing_column(df_full, _P_CANDIDATES)
        if c_chr is None or c_pos is None or c_p is None:
            raise RuntimeError(
                "Cannot detect required columns (chrom/pos/pvalue). "
                f"Found columns: {list(df_full.columns)}"
            )
        df = df_full[[c_chr, c_pos, c_p]].copy()

    if df.shape[0] == 0:
        raise RuntimeError("GWAS file has no rows.")
    if c_chr is None or c_pos is None or c_p is None:
        c_chr = _pick_existing_column(df, _CHROM_CANDIDATES)
        c_pos = _pick_existing_column(df, _POS_CANDIDATES)
        c_p = _pick_existing_column(df, _P_CANDIDATES)
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

    out = pd.DataFrame(
        {
            "chrom_norm": pd.Categorical(chrom_norm),
            "pos": pos_arr,
            "p": p_arr,
            "y": y_arr,
        }
    )
    return out, {"chr": c_chr, "pos": c_pos, "p": c_p}


def _load_sig_table_cached(path: Path) -> pd.DataFrame:
    """
    Load GWAS table with in-memory cache for sig-site/LD-clump path.
    This avoids re-reading large files when only threshold/LD params change.
    """
    try:
        rp = str(path.resolve())
        mtime = float(path.stat().st_mtime)
    except Exception:
        rp = str(path)
        mtime = -1.0
    cached = _SIG_GWAS_CACHE.get(rp)
    if cached is not None and float(cached[0]) == mtime:
        return cached[1]
    tbl, _ = _load_table(path)
    _SIG_GWAS_CACHE[rp] = (mtime, tbl)
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
    Columns: chrom_norm,start,end,name,feature,parent
    """
    txt = str(anno_file or "").strip()
    if txt == "":
        return pd.DataFrame(columns=["chrom_norm", "start", "end", "name", "feature", "parent"])
    p = Path(txt).expanduser()
    if not p.exists() or not p.is_file():
        return pd.DataFrame(columns=["chrom_norm", "start", "end", "name", "feature", "parent"])

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
    rows: list[tuple[str, int, int, str, str, str]] = []
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
                rows.append((str(chrom), int(s), int(e), str(name), str(feature), str(parent)))
    except Exception:
        out_empty = pd.DataFrame(columns=["chrom_norm", "start", "end", "name", "feature", "parent"])
        _TRACK_ANNO_CACHE[rp] = (mtime, out_empty)
        return out_empty

    if len(rows) == 0:
        out_empty = pd.DataFrame(columns=["chrom_norm", "start", "end", "name", "feature", "parent"])
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
    sub = df.loc[m, ["start", "end", "name", "feature", "parent"]]
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
    path = Path(p).expanduser()
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

    path = Path(genofile).expanduser()
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

    p = Path(txt).expanduser()
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
        try:
            s = int(r.get("start", 0))
            e = int(r.get("end", 0))
        except Exception:
            r["Gene"] = "NA"
            continue
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


def build_sig_table(
    row: dict[str, Any],
    *,
    threshold: Any = 1e-6,
    window_bp: int = 10_000,
    r2_thr: float = 0.8,
    anno_file: str = "",
) -> dict[str, Any]:
    """
    Build threshold-passing SNP table with LD clump de-redundancy.
    """
    gwas_path = _resolve_result_file(row)
    table = _load_sig_table_cached(gwas_path)

    try:
        thr = float(threshold)
    except Exception:
        thr = 1e-6
    if not np.isfinite(thr) or thr <= 0.0:
        thr = 1e-6
    if thr > 1.0:
        thr = 1.0

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
    n_sig = int(work.shape[0])
    n = int(work.shape[0])
    if n == 0:
        return {
            "threshold": float(thr),
            "n_sig": 0,
            "n_leads": 0,
            "rows": [],
        }

    chrom_arr = work["chrom_norm"].to_numpy(dtype=object)
    pos_arr = work["pos"].to_numpy(dtype=np.int64)
    p_arr = work["p"].to_numpy(dtype=float)
    all_keys = [(str(chrom_arr[i]), int(pos_arr[i])) for i in range(n)]

    genofile = str(row.get("genofile", "")).strip()
    preloaded_geno: np.ndarray | None = None
    if genofile != "":
        try:
            chunks: list[np.ndarray] = []
            for chunk, _sites in load_genotype_chunks(
                genofile,
                chunk_size=max(1, min(20_000, n)),
                maf=0.0,
                missing_rate=1.0,
                impute=True,
                snp_sites=all_keys,
            ):
                chunks.append(np.asarray(chunk, dtype=np.float32))
            if len(chunks) > 0:
                preloaded_geno = np.vstack(chunks).astype(np.float32, copy=False)
            if preloaded_geno is None or int(preloaded_geno.shape[0]) != n:
                preloaded_geno = None
        except Exception:
            preloaded_geno = None

    # Build per-chromosome sorted position index for O(log n) window lookup.
    chr_window_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c in pd.unique(chrom_arr):
        idx = np.flatnonzero(chrom_arr == c).astype(np.int64, copy=False)
        if idx.size == 0:
            continue
        ord_pos = np.argsort(pos_arr[idx], kind="mergesort")
        sorted_idx = idx[ord_pos]
        sorted_pos = pos_arr[sorted_idx]
        chr_window_index[str(c)] = (sorted_idx, sorted_pos)

    remaining = np.ones(n, dtype=bool)
    out_rows: list[dict[str, Any]] = []
    wb = int(window_bp)
    r2_cut = float(r2_thr)

    for lead_i in range(n):
        if not bool(remaining[lead_i]):
            continue
        lead_chr = str(chrom_arr[lead_i])
        lead_pos = int(pos_arr[lead_i])
        lead_p = float(p_arr[lead_i])

        sorted_pack = chr_window_index.get(lead_chr, None)
        if sorted_pack is None:
            cand_idx = np.array([lead_i], dtype=np.int64)
        else:
            sorted_idx, sorted_pos = sorted_pack
            left = int(np.searchsorted(sorted_pos, lead_pos - wb, side="left"))
            right = int(np.searchsorted(sorted_pos, lead_pos + wb, side="right"))
            cand_idx = sorted_idx[left:right]
            if cand_idx.size > 0:
                cand_idx = cand_idx[remaining[cand_idx]]
            if cand_idx.size == 0:
                cand_idx = np.array([lead_i], dtype=np.int64)
            elif int(cand_idx[0]) != int(lead_i):
                cand_idx = np.concatenate(
                    (
                        np.array([lead_i], dtype=np.int64),
                        cand_idx[cand_idx != int(lead_i)],
                    ),
                    axis=0,
                )

        clump_idx = np.array([lead_i], dtype=np.int64)
        mean_r2 = 1.0
        if preloaded_geno is not None and cand_idx.size > 1:
            try:
                geno_block = preloaded_geno[cand_idx, :]
                r2_vec = _lead_vs_all_r2(geno_block)
                keep = np.asarray(r2_vec >= r2_cut, dtype=bool)
                if keep.shape[0] != cand_idx.shape[0]:
                    keep = np.zeros(cand_idx.shape[0], dtype=bool)
                    keep[0] = True
                clump_idx = cand_idx[keep]
                if clump_idx.size == 0:
                    clump_idx = np.array([lead_i], dtype=np.int64)
                elif int(clump_idx[0]) != int(lead_i):
                    clump_idx = np.concatenate(
                        (
                            np.array([lead_i], dtype=np.int64),
                            clump_idx[clump_idx != int(lead_i)],
                        ),
                        axis=0,
                    )
                if np.any(keep):
                    mean_r2 = float(np.mean(r2_vec[keep]))
            except Exception:
                clump_idx = np.array([lead_i], dtype=np.int64)
                mean_r2 = 1.0

        clump_pos = pos_arr[clump_idx]
        order = np.argsort(clump_pos, kind="mergesort")
        clump_idx = clump_idx[order]
        ld_start = int(np.min(clump_pos))
        ld_end = int(np.max(clump_pos))
        ldclump_text = ";".join([f"{str(chrom_arr[j])}_{int(pos_arr[j])}" for j in clump_idx])
        out_rows.append(
            {
                "chrom": str(lead_chr),
                "pos": int(lead_pos),
                "p": float(lead_p),
                "logp": float(-np.log10(max(float(lead_p), np.nextafter(0.0, 1.0)))),
                "start": int(ld_start),
                "end": int(ld_end),
                "nsnps": int(clump_idx.size),
                "MeanR2": float(mean_r2),
                "LDclump": ldclump_text,
            }
        )
        remaining[clump_idx] = False

    out_rows = sorted(out_rows, key=lambda x: (_natural_key(x["chrom"]), int(x["pos"])))
    anno_txt = str(anno_file or "").strip()
    if anno_txt != "":
        _annotate_sig_rows_with_genes(out_rows, anno_txt)
    else:
        for r in out_rows:
            r["Gene"] = ""
    return {
        "threshold": float(thr),
        "n_sig": int(n_sig),
        "n_leads": int(len(out_rows)),
        "rows": out_rows,
    }


def build_merged_sig_table(
    rows: list[dict[str, Any]],
    *,
    threshold: Any = 1e-6,
    window_bp: int = 10_000,
    r2_thr: float = 0.8,
    anno_file: str = "",
) -> dict[str, Any]:
    """
    Build merged sig table across multiple GWAS results:
    1) keep SNPs that pass threshold in every GWAS (intersection)
    2) aggregate p by min-p across GWAS
    3) LD-clump on the shared SNP set
    """
    if not isinstance(rows, list) or len(rows) == 0:
        return {"threshold": 1e-6, "n_sig": 0, "n_leads": 0, "rows": []}

    try:
        thr = float(threshold)
    except Exception:
        thr = 1e-6
    if not np.isfinite(thr) or thr <= 0.0:
        thr = 1e-6
    if thr > 1.0:
        thr = 1.0

    per_maps: list[dict[tuple[str, int], float]] = []
    for row in rows:
        gwas_path = _resolve_result_file(row)
        table = _load_sig_table_cached(gwas_path)
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
        rows0.append({"chrom": str(c), "pos": int(p), "p": pmin})
    if len(rows0) == 0:
        return {"threshold": float(thr), "n_sig": 0, "n_leads": 0, "rows": []}

    rows0 = sorted(rows0, key=lambda x: (float(x["p"]), _natural_key(x["chrom"]), int(x["pos"])))
    n = len(rows0)
    chrom_arr = np.asarray([str(r["chrom"]) for r in rows0], dtype=object)
    pos_arr = np.asarray([int(r["pos"]) for r in rows0], dtype=np.int64)
    p_arr = np.asarray([float(r["p"]) for r in rows0], dtype=float)
    n_sig = int(n)

    # Use first row genotype (server already checks genotype consistency).
    gref = rows[0] if len(rows) > 0 else {}
    genofile = str(gref.get("genofile", "")).strip()
    preloaded_geno: np.ndarray | None = None
    if genofile != "":
        all_keys = [(str(chrom_arr[i]), int(pos_arr[i])) for i in range(n)]
        try:
            chunks: list[np.ndarray] = []
            for chunk, _sites in load_genotype_chunks(
                genofile,
                chunk_size=max(1, min(20_000, n)),
                maf=0.0,
                missing_rate=1.0,
                impute=True,
                snp_sites=all_keys,
            ):
                chunks.append(np.asarray(chunk, dtype=np.float32))
            if len(chunks) > 0:
                preloaded_geno = np.vstack(chunks).astype(np.float32, copy=False)
            if preloaded_geno is None or int(preloaded_geno.shape[0]) != n:
                preloaded_geno = None
        except Exception:
            preloaded_geno = None

    # Build per-chromosome sorted index for window lookup.
    chr_window_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c in pd.unique(chrom_arr):
        idx = np.flatnonzero(chrom_arr == c).astype(np.int64, copy=False)
        if idx.size == 0:
            continue
        ord_pos = np.argsort(pos_arr[idx], kind="mergesort")
        sorted_idx = idx[ord_pos]
        sorted_pos = pos_arr[sorted_idx]
        chr_window_index[str(c)] = (sorted_idx, sorted_pos)

    remaining = np.ones(n, dtype=bool)
    out_rows: list[dict[str, Any]] = []
    wb = int(window_bp)
    r2_cut = float(r2_thr)

    for lead_i in range(n):
        if not bool(remaining[lead_i]):
            continue
        lead_chr = str(chrom_arr[lead_i])
        lead_pos = int(pos_arr[lead_i])
        lead_p = float(p_arr[lead_i])

        sorted_pack = chr_window_index.get(lead_chr, None)
        if sorted_pack is None:
            cand_idx = np.array([lead_i], dtype=np.int64)
        else:
            sorted_idx, sorted_pos = sorted_pack
            left = int(np.searchsorted(sorted_pos, lead_pos - wb, side="left"))
            right = int(np.searchsorted(sorted_pos, lead_pos + wb, side="right"))
            cand_idx = sorted_idx[left:right]
            if cand_idx.size > 0:
                cand_idx = cand_idx[remaining[cand_idx]]
            if cand_idx.size == 0:
                cand_idx = np.array([lead_i], dtype=np.int64)
            elif int(cand_idx[0]) != int(lead_i):
                cand_idx = np.concatenate(
                    (
                        np.array([lead_i], dtype=np.int64),
                        cand_idx[cand_idx != int(lead_i)],
                    ),
                    axis=0,
                )

        clump_idx = np.array([lead_i], dtype=np.int64)
        mean_r2 = 1.0
        if preloaded_geno is not None and cand_idx.size > 1:
            try:
                geno_block = preloaded_geno[cand_idx, :]
                r2_vec = _lead_vs_all_r2(geno_block)
                keep = np.asarray(r2_vec >= r2_cut, dtype=bool)
                if keep.shape[0] != cand_idx.shape[0]:
                    keep = np.zeros(cand_idx.shape[0], dtype=bool)
                    keep[0] = True
                clump_idx = cand_idx[keep]
                if clump_idx.size == 0:
                    clump_idx = np.array([lead_i], dtype=np.int64)
                elif int(clump_idx[0]) != int(lead_i):
                    clump_idx = np.concatenate(
                        (
                            np.array([lead_i], dtype=np.int64),
                            clump_idx[clump_idx != int(lead_i)],
                        ),
                        axis=0,
                    )
                if np.any(keep):
                    mean_r2 = float(np.mean(r2_vec[keep]))
            except Exception:
                clump_idx = np.array([lead_i], dtype=np.int64)
                mean_r2 = 1.0

        clump_pos = pos_arr[clump_idx]
        order = np.argsort(clump_pos, kind="mergesort")
        clump_idx = clump_idx[order]
        ld_start = int(np.min(clump_pos))
        ld_end = int(np.max(clump_pos))
        ldclump_text = ";".join([f"{str(chrom_arr[j])}_{int(pos_arr[j])}" for j in clump_idx])
        out_rows.append(
            {
                "chrom": str(lead_chr),
                "pos": int(lead_pos),
                "p": float(lead_p),
                "logp": float(-np.log10(max(float(lead_p), np.nextafter(0.0, 1.0)))),
                "start": int(ld_start),
                "end": int(ld_end),
                "nsnps": int(clump_idx.size),
                "MeanR2": float(mean_r2),
                "LDclump": ldclump_text,
            }
        )
        remaining[clump_idx] = False

    out_rows = sorted(out_rows, key=lambda x: (_natural_key(x["chrom"]), int(x["pos"])))
    anno_txt = str(anno_file or "").strip()
    if anno_txt != "":
        _annotate_sig_rows_with_genes(out_rows, anno_txt)
    else:
        for r in out_rows:
            r["Gene"] = ""
    return {
        "threshold": float(thr),
        "n_sig": int(n_sig),
        "n_leads": int(len(out_rows)),
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
    manh_ratio: float = 2.0,
    manh_palette: str = "auto",
    manh_marker: str = "o",
    manh_size: float = 16.0,
    ld_color: str = "#4b5563",
    ld_p_threshold: Any = "auto",
    render_qq: bool = True,
    editable_svg: bool = False,
    cache: dict[str, tuple[float, pd.DataFrame, dict[str, str]]] | None = None,
) -> tuple[str, dict[str, Any]]:
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

    series: list[dict[str, Any]] = []
    all_chroms: set[str] = set()
    for i, row in enumerate(rows):
        gwas_path = _resolve_result_file(row)
        key = str(gwas_path)
        mtime = float(gwas_path.stat().st_mtime)
        table: pd.DataFrame
        if cache is not None and key in cache:
            c_mtime, c_df, _c_cols = cache[key]
            if float(c_mtime) == mtime:
                table = c_df
            else:
                table, cols = _load_table(gwas_path)
                cache[key] = (mtime, table, cols)
        else:
            table, cols = _load_table(gwas_path)
            if cache is not None:
                cache[key] = (mtime, table, cols)
        if int(table.shape[0]) == 0:
            continue
        draw_df = _filter_by_bimranges(table, brs) if len(brs) > 0 else table
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
    marker = _normalize_marker(manh_marker)
    msize = _parse_float(manh_size, 16.0, min_value=0.1)
    rasterize_main = (not bool(editable_svg))
    fig_dpi = int(_WEBUI_RASTER_DPI if rasterize_main else _WEBUI_VECTOR_DPI)

    # Keep merge behavior aligned with single-GWAS:
    # - bimrange mode: Manhattan + LD only (no QQ)
    # - non-bimrange mode: QQ drawn only when render_qq is requested.
    show_ld = len(brs) > 0
    show_qq = (not show_ld) and bool(render_qq)
    if show_ld:
        top_ratio = float(np.clip(2.0 / float(ratio), 0.08, 2.50))
        fig = plt.figure(figsize=(15.0, 8.8), dpi=fig_dpi)
        gs = GridSpec(
            3,
            1,
            figure=fig,
            width_ratios=[1.0],
            height_ratios=[top_ratio, 0.11, 1.05],
            left=0.05,
            right=0.995,
            top=0.965,
            bottom=0.06,
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
    y_top = 1.0
    qq_xmax = 1.0
    for i, s in enumerate(series):
        df0 = s["draw"]
        y = df0["y"].to_numpy(dtype=float, copy=False)
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
        keep = np.isfinite(x) & np.isfinite(y)
        if not np.any(keep):
            continue
        xk = x[keep]
        yk = y[keep]
        x_left = min(x_left, float(np.nanmin(xk)))
        x_right = max(x_right, float(np.nanmax(xk)))
        y_top = max(y_top, float(np.nanmax(yk)))
        col = color_base[i % len(color_base)]
        ax_manh.scatter(
            xk,
            yk,
            s=msize,
            c=col,
            alpha=0.68,
            marker=marker,
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
                        s=max(2.0, msize * 0.55),
                        c=col,
                        alpha=0.68,
                        marker=marker,
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
    if not np.isfinite(y_top) or y_top <= 0.0:
        y_top = 1.0
    manh_ylim = (0.0, float(y_top) * 1.03)
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
        ax_qq.set_box_aspect(1.0)
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
            # Draw merge bridge lines: Manhattan SNP position -> LD block order.
            if ax_bridge is not None:
                try:
                    pairs: list[tuple[float, float]] = []
                    # Build x mapping from LD keys to Manhattan x coordinates.
                    if len(brs) == 1:
                        for i, k0 in enumerate(ld_keys[: int(ld_r2.shape[0])]):
                            c0 = _normalize_chrom(k0[0])
                            if c0 != brs[0].chrom:
                                continue
                            x_top = float(k0[1]) / 1e6
                            x_ld = float(i) + 0.5
                            pairs.append((x_top, x_ld))
                    else:
                        seg_by_chrom = {}
                        for seg in segs:
                            seg_by_chrom.setdefault(seg.br.chrom, []).append(seg)
                        for i, k0 in enumerate(ld_keys[: int(ld_r2.shape[0])]):
                            c0 = _normalize_chrom(k0[0])
                            p0 = int(k0[1])
                            x_top = None
                            for seg in seg_by_chrom.get(c0, []):
                                if int(seg.br.start_bp) <= p0 <= int(seg.br.end_bp):
                                    x_top = float(seg.x_start) + (float(p0) - float(seg.br.start_bp)) / 1e6
                                    break
                            if x_top is None:
                                continue
                            x_ld = float(i) + 0.5
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
            ax_ld.set_adjustable("datalim")
            ax_ld.set_position([manh_pos.x0, ld_pos.y0, manh_pos.width, ld_pos.height])
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
        "n_total": int(sum(int(s["full"].shape[0]) for s in series)),
        "legend": [str(s["label"]) for s in series],
        "mode": mode,
        "ld_nsnps": int(ld_n),
        "ld_p_threshold": float(ld_thr),
        "ld_p_threshold_auto": bool(ld_auto),
        "bimrange": "; ".join([b.label for b in brs]),
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
    manh_marker: str = "o",
    manh_size: float = 16.0,
    ld_color: str = "#4b5563",
    ld_p_threshold: Any = "auto",
    editable_svg: bool = False,
    cache: dict[str, tuple[float, pd.DataFrame, dict[str, str]]] | None = None,
) -> tuple[str, dict[str, Any]]:
    gwas_path = _resolve_result_file(row)
    key = str(gwas_path)
    mtime = float(gwas_path.stat().st_mtime)
    table: pd.DataFrame
    cols: dict[str, str]
    if cache is not None and key in cache:
        c_mtime, c_df, c_cols = cache[key]
        if float(c_mtime) == mtime:
            table = c_df
            cols = c_cols
        else:
            table, cols = _load_table(gwas_path)
            cache[key] = (mtime, table, cols)
    else:
        table, cols = _load_table(gwas_path)
        if cache is not None:
            cache[key] = (mtime, table, cols)

    phenotype_name = str(row.get("phenotype", "")).strip() or "Phenotype"
    n_idv = _coerce_pos_int(row.get("nidv", None))
    n_snp = _coerce_pos_int(row.get("eff_snp", None))
    geno_n_idv, geno_n_snp = (None, None)
    if n_idv is None:
        n_idv = _estimate_n_from_phenofile(str(row.get("phenofile", "")).strip())
    if n_idv is None or n_snp is None:
        geno_n_idv, geno_n_snp = _inspect_genotype_meta(str(row.get("genofile", "")).strip())
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
    marker = _normalize_marker(manh_marker)
    msize = _parse_float(manh_size, 16.0, min_value=0.1)
    manh_color_set = _resolve_manh_color_set(manh_palette, tone)
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
    if (not bool(full_plot)) and int(df.shape[0]) >= int(_LARGE_MATRIX_ROWS):
        p_filter_text = f"p<={_LARGE_MATRIX_P_THRESH:g}"
        df_sig = df.loc[df["p"].values <= float(_LARGE_MATRIX_P_THRESH)].copy()
        if df_sig.shape[0] > 0:
            df = df_sig
            use_threshold_floor = True
        else:
            # Keep a minimal informative set when no SNP passes threshold.
            keep_n = int(min(5000, df.shape[0]))
            if keep_n > 0:
                pvals = np.asarray(df["p"].values, dtype=float)
                idx = np.argpartition(pvals, keep_n - 1)[:keep_n]
                df = df.iloc[np.sort(idx)].copy()
            p_filter_text = f"top-{keep_n} by p"
    if df.shape[0] == 0:
        raise RuntimeError("No SNPs left after applying bimrange.")
    n_draw = int(df.shape[0])
    rasterize_main = (not bool(editable_svg))
    fig_dpi = int(_WEBUI_RASTER_DPI if rasterize_main else _WEBUI_VECTOR_DPI)

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
        top_ratio = float(np.clip(2.0 / float(ratio), 0.08, 2.50))
        fig = plt.figure(figsize=(9.8, 12.8), dpi=fig_dpi)
        gs = GridSpec(
            3,
            1,
            figure=fig,
            height_ratios=[top_ratio, 0.11, 1.05],
            left=0.055,
            right=0.995,
            top=0.985,
            bottom=0.05,
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
            alpha=0.72,
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
                alpha=0.72,
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
        # Keep QQ panel physically square (equal width/height on canvas).
        ax_qq.set_box_aspect(1.0)
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
                        }
                    )

        if len(mapped_records) > 0:
            rows = 6
            row_by_gene: dict[str, int] = {}
            genes = [r for r in mapped_records if r.get("feature") == "gene"]
            genes.sort(key=lambda r: (float(r.get("x0", 0.0)), float(r.get("x1", 0.0))))
            for i, r in enumerate(genes):
                gid = str(r.get("parent") or r.get("name") or i)
                row_by_gene[gid] = int(i % rows)

            n_label = 0
            # Draw gene skeleton line (dark).
            for i, r in enumerate(genes):
                gid = str(r.get("parent") or r.get("name") or i)
                yrow = float(row_by_gene.get(gid, i % rows))
                xs = float(r.get("x0", 0.0))
                xe = float(r.get("x1", 0.0))
                ax_gene.plot(
                    [xs, xe],
                    [yrow + 0.5, yrow + 0.5],
                    color=str(tone["dark"]),
                    linewidth=1.1,
                    solid_capstyle="round",
                    alpha=0.95,
                )
                if n_label < 14:
                    ax_gene.text(
                        xs,
                        yrow + 0.86,
                        str(r.get("name", "")),
                        fontsize=6,
                        color=str(tone["dark"]),
                    )
                    n_label += 1

            # Draw CDS/UTR blocks (middle tone).
            for i, r in enumerate(mapped_records):
                ft = str(r.get("feature", "gene")).lower()
                if ft == "gene":
                    continue
                gid = str(r.get("parent") or r.get("name") or i)
                yrow = float(row_by_gene.get(gid, i % rows))
                xs = float(r.get("x0", 0.0))
                xe = float(r.get("x1", 0.0))
                rect = Rectangle(
                    (xs, yrow + 0.34),
                    max(xe - xs, 1e-6),
                    0.32,
                    facecolor=str(tone["mid"]),
                    edgecolor=str(tone["mid"]),
                    linewidth=0.0,
                    alpha=0.95,
                )
                ax_gene.add_patch(rect)
            ax_gene.set_xlim(ax_manh.get_xlim())
            ax_gene.set_ylim(0, float(rows + 0.5))
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
            # Keep LD panel width strictly aligned with Manhattan panel in WebUI.
            try:
                manh_pos = ax_manh.get_position()
                ld_pos = ax_ld.get_position()
                ax_ld.set_adjustable("datalim")
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
                x_ld = float(i) + 0.5
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
            ax_ld.set_adjustable("datalim")
            ax_ld.set_position([manh_pos.x0, ld_pos.y0, manh_pos.width, ld_pos.height])
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
        "manh_marker": str(marker),
        "manh_size": float(msize),
        "highlight_n": int(highlight_points_n),
    }
    return svg, info
