from __future__ import annotations

from typing import Any, Iterable, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb, to_rgba


GenotypeChunks = Iterable[Tuple[np.ndarray, Sequence[object]]]
GenotypeInput = Union[pd.DataFrame, GenotypeChunks, Tuple[GenotypeChunks, Sequence[str]]]

_PALETTE_PRESETS: dict[str, dict[str, object]] = {
    "default": {
        "violin_face": "#4C78A8",
        "violin_edge": "#2F3E4E",
        "violin_alpha": 0.65,
        "box_face": "#FFFFFF",
        "box_edge": "#2F3E4E",
        "box_alpha": 0.65,
        "point_face": "#FFFFFF",
        "point_edge": "#2F3E4E",
        "point_alpha": 0.85,
        "letter_color": "#111111",
        "matrix_na": "#D9D9D9",
        "matrix_nonfav": "#E8E8E8",
        "matrix_hue_step": 0.61803398875,
        "matrix_sat_light": 0.45,
        "matrix_val_light": 0.96,
        "matrix_sat_dark": 0.90,
        "matrix_val_dark": 0.50,
        "matrix_sat_mid": 0.62,
        "matrix_val_mid": 0.74,
    },
    "pastel": {
        "violin_face": "#8EC5FC",
        "violin_edge": "#4A6FA5",
        "violin_alpha": 0.70,
        "box_face": "#F7FBFF",
        "box_edge": "#4A6FA5",
        "box_alpha": 0.70,
        "point_face": "#FFFFFF",
        "point_edge": "#4A6FA5",
        "point_alpha": 0.82,
        "letter_color": "#1B2A41",
        "matrix_na": "#ECECEC",
        "matrix_nonfav": "#F1F1F1",
        "matrix_hue_step": 0.61803398875,
        "matrix_sat_light": 0.32,
        "matrix_val_light": 0.98,
        "matrix_sat_dark": 0.62,
        "matrix_val_dark": 0.72,
        "matrix_sat_mid": 0.45,
        "matrix_val_mid": 0.84,
    },
    "deep": {
        "violin_face": "#1F77B4",
        "violin_edge": "#102A43",
        "violin_alpha": 0.78,
        "box_face": "#F5F7FA",
        "box_edge": "#102A43",
        "box_alpha": 0.72,
        "point_face": "#E9F2FF",
        "point_edge": "#102A43",
        "point_alpha": 0.90,
        "letter_color": "#0B132B",
        "matrix_na": "#CFCFCF",
        "matrix_nonfav": "#E0E0E0",
        "matrix_hue_step": 0.61803398875,
        "matrix_sat_light": 0.55,
        "matrix_val_light": 0.92,
        "matrix_sat_dark": 0.95,
        "matrix_val_dark": 0.44,
        "matrix_sat_mid": 0.75,
        "matrix_val_mid": 0.68,
    },
    "colorblind": {
        "violin_face": "#0072B2",
        "violin_edge": "#003049",
        "violin_alpha": 0.72,
        "box_face": "#F8FAFC",
        "box_edge": "#003049",
        "box_alpha": 0.70,
        "point_face": "#FFFFFF",
        "point_edge": "#003049",
        "point_alpha": 0.88,
        "letter_color": "#111111",
        "matrix_na": "#D8D8D8",
        "matrix_nonfav": "#EAEAEA",
        "matrix_hue_step": 0.38196601125,
        "matrix_sat_light": 0.45,
        "matrix_val_light": 0.96,
        "matrix_sat_dark": 0.85,
        "matrix_val_dark": 0.52,
        "matrix_sat_mid": 0.60,
        "matrix_val_mid": 0.72,
    },
    "mono": {
        "violin_face": "#B0BEC5",
        "violin_edge": "#37474F",
        "violin_alpha": 0.72,
        "box_face": "#FFFFFF",
        "box_edge": "#37474F",
        "box_alpha": 0.72,
        "point_face": "#FFFFFF",
        "point_edge": "#37474F",
        "point_alpha": 0.85,
        "letter_color": "#1A1A1A",
        "matrix_na": "#CCCCCC",
        "matrix_nonfav": "#E5E5E5",
        "matrix_hue_step": 0.0,
        "matrix_sat_light": 0.0,
        "matrix_val_light": 0.92,
        "matrix_sat_dark": 0.0,
        "matrix_val_dark": 0.52,
        "matrix_sat_mid": 0.0,
        "matrix_val_mid": 0.72,
    },
}


def _clamp_float(value: object, default: float, low: float, high: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(f):
        return default
    return float(min(high, max(low, f)))


def _safe_rgba(value: object, fallback: str) -> tuple[float, float, float, float]:
    try:
        return tuple(to_rgba(value))
    except Exception:
        return tuple(to_rgba(fallback))


def _colormap_color_list(cmap_name: object, n: int) -> list[tuple[float, float, float, float]]:
    if cmap_name is None or int(n) <= 0:
        return []
    try:
        cmap = plt.get_cmap(str(cmap_name))
    except Exception:
        return []

    if hasattr(cmap, "colors") and getattr(cmap, "colors", None):
        base = [tuple(to_rgba(c)) for c in cmap.colors]
        if len(base) == 0:
            return []
        return [base[i % len(base)] for i in range(int(n))]

    if int(n) == 1:
        return [tuple(to_rgba(cmap(0.5)))]
    idx = np.linspace(0.0, 1.0, int(n), endpoint=False, dtype=float)
    return [tuple(to_rgba(cmap(float(x)))) for x in idx]


def _mix_rgb(
    c1: tuple[float, float, float, float],
    c2: tuple[float, float, float, float],
    w1: float,
    w2: float,
) -> tuple[float, float, float, float]:
    total = max(float(w1 + w2), 1e-9)
    rgb = (
        (c1[0] * w1 + c2[0] * w2) / total,
        (c1[1] * w1 + c2[1] * w2) / total,
        (c1[2] * w1 + c2[2] * w2) / total,
        1.0,
    )
    return rgb


def _resolve_palette(palette: Optional[Union[str, dict]]) -> dict[str, object]:
    if palette is None:
        palette = "default"
    if isinstance(palette, str):
        key = palette.strip().lower()
        if key in _PALETTE_PRESETS:
            cfg = dict(_PALETTE_PRESETS[key])
        else:
            cmap_colors = _colormap_color_list(palette, 3)
            if not cmap_colors:
                available = ", ".join(sorted(_PALETTE_PRESETS.keys()))
                raise ValueError(
                    f"Unknown palette `{palette}`. Available presets: {available}, "
                    "or any matplotlib colormap (e.g. `tab10`)."
                )
            cfg = dict(_PALETTE_PRESETS["default"])
            cfg["matrix_colormap"] = str(palette)
            cfg["violin_face"] = cmap_colors[0]
            cfg["violin_edge"] = cmap_colors[1]
            cfg["box_edge"] = cmap_colors[1]
            cfg["point_edge"] = cmap_colors[1]
            cfg["letter_color"] = cmap_colors[1]
    elif isinstance(palette, dict):
        cfg = dict(_PALETTE_PRESETS["default"])
        cfg.update(palette)
    else:
        raise TypeError("`palette` must be a preset name or a dict of style overrides.")

    for color_key, default in [
        ("violin_face", "#4C78A8"),
        ("violin_edge", "#2F3E4E"),
        ("box_face", "#FFFFFF"),
        ("box_edge", "#2F3E4E"),
        ("point_face", "#FFFFFF"),
        ("point_edge", "#2F3E4E"),
        ("letter_color", "#111111"),
        ("matrix_na", "#D9D9D9"),
        ("matrix_nonfav", "#E8E8E8"),
    ]:
        cfg[color_key] = _safe_rgba(cfg.get(color_key, default), default)

    for alpha_key, default in [
        ("violin_alpha", 0.65),
        ("box_alpha", 0.65),
        ("point_alpha", 0.85),
    ]:
        cfg[alpha_key] = _clamp_float(cfg.get(alpha_key, default), default, 0.0, 1.0)

    for num_key, default in [
        ("matrix_hue_step", 0.61803398875),
        ("matrix_sat_light", 0.45),
        ("matrix_val_light", 0.96),
        ("matrix_sat_dark", 0.90),
        ("matrix_val_dark", 0.50),
        ("matrix_sat_mid", 0.62),
        ("matrix_val_mid", 0.74),
    ]:
        cfg[num_key] = _clamp_float(cfg.get(num_key, default), default, 0.0, 1.0)
    return cfg


def _normalize_phenotype(phenotype: pd.DataFrame) -> pd.Series:
    if not isinstance(phenotype, pd.DataFrame):
        raise TypeError("`phenotype` must be a pandas DataFrame.")
    if phenotype.empty:
        raise ValueError("`phenotype` is empty.")

    if phenotype.shape[1] == 1:
        value = phenotype.iloc[:, 0].copy()
        if isinstance(phenotype.index, pd.RangeIndex):
            raise ValueError(
                "For one-column phenotype DataFrame, sample IDs must be in the index."
            )
    else:
        if isinstance(phenotype.index, pd.RangeIndex):
            sample_col = phenotype.columns[0]
            value_col = phenotype.columns[1]
            value = phenotype.set_index(sample_col)[value_col].copy()
        else:
            numeric_cols = [
                c for c in phenotype.columns if pd.api.types.is_numeric_dtype(phenotype[c])
            ]
            if not numeric_cols:
                raise ValueError("No numeric phenotype column found.")
            value = phenotype[numeric_cols[0]].copy()

    value = pd.to_numeric(value, errors="coerce")
    value.index = value.index.map(str)
    value = value.dropna()
    value = value[~value.index.duplicated(keep="first")]
    if value.empty:
        raise ValueError("No valid phenotype values after dropping NA/invalid rows.")
    return value


def _detect_genotype_columns(df: pd.DataFrame) -> Tuple[str, str, str, str, Sequence[str]]:
    def pick(candidates: Sequence[str]) -> Optional[str]:
        lower_map = {str(c).lower(): c for c in df.columns}
        for name in candidates:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        return None

    chrom_col = pick(["#CHROM", "CHROM", "chr", "chrom"])
    pos_col = pick(["POS", "pos"])
    ref_col = pick(["A0", "REF", "ref", "ref_allele"])
    alt_col = pick(["A1", "ALT", "alt", "alt_allele"])

    if chrom_col is None or pos_col is None:
        raise ValueError("Genotype DataFrame must contain CHROM/chr and POS columns.")
    if ref_col is None or alt_col is None:
        raise ValueError("Genotype DataFrame must contain REF/A0 and ALT/A1 columns.")

    meta = {chrom_col, pos_col, ref_col, alt_col}
    sample_cols = [c for c in df.columns if c not in meta]
    if not sample_cols:
        raise ValueError("Genotype DataFrame has no sample columns.")
    return chrom_col, pos_col, ref_col, alt_col, sample_cols


def _parse_site_meta(site: object) -> Tuple[str, int, str, str]:
    if isinstance(site, dict):
        chrom = site.get("chrom", site.get("chr", site.get("#CHROM", "NA")))
        pos = site.get("pos", site.get("POS", 0))
        ref = site.get("ref_allele", site.get("ref", site.get("REF", "N")))
        alt = site.get("alt_allele", site.get("alt", site.get("ALT", "N")))
        return str(chrom), int(pos), str(ref), str(alt)

    chrom = getattr(site, "chrom", getattr(site, "chr", "NA"))
    pos = getattr(site, "pos", getattr(site, "POS", 0))
    ref = getattr(site, "ref_allele", getattr(site, "ref", "N"))
    alt = getattr(site, "alt_allele", getattr(site, "alt", "N"))
    return str(chrom), int(pos), str(ref), str(alt)


def _convert_call_to_allele(value: object, ref: str, alt: str) -> str:
    if pd.isna(value):
        return "NA"

    if isinstance(value, (int, np.integer, float, np.floating)) and np.isfinite(value):
        v = int(round(float(value)))
        if v == 0:
            return ref
        if v == 1:
            return "H"
        if v == 2:
            return alt
        return "NA"

    s = str(value).strip()
    if not s:
        return "NA"
    if s in {ref, alt, "H"}:
        return s

    if s in {"0", "0/0", "0|0"}:
        return ref
    if s in {"1", "0/1", "1/0", "0|1", "1|0"}:
        return "H"
    if s in {"2", "1/1", "1|1"}:
        return alt

    if "/" in s or "|" in s:
        sep = "/" if "/" in s else "|"
        alleles = s.split(sep)
        if len(alleles) == 2:
            a, b = alleles
            if a == b:
                if a in {"0", ref}:
                    return ref
                if a in {"1", alt}:
                    return alt
            return "H"

    return s


def _genotype_dataframe_to_sample_site(
    genotype_df: pd.DataFrame, phenotype_samples: Sequence[str]
) -> pd.DataFrame:
    if isinstance(genotype_df.index, pd.MultiIndex) and genotype_df.index.nlevels >= 2:
        chrom_vals = genotype_df.index.get_level_values(0)
        pos_vals = genotype_df.index.get_level_values(1)
        ref_col = None
        alt_col = None
        for cand in ("A0", "REF", "ref", "ref_allele"):
            if cand in genotype_df.columns:
                ref_col = cand
                break
        for cand in ("A1", "ALT", "alt", "alt_allele"):
            if cand in genotype_df.columns:
                alt_col = cand
                break
        if ref_col is None or alt_col is None:
            raise ValueError(
                "MultiIndex genotype DataFrame requires REF/A0 and ALT/A1 columns."
            )
        sample_cols = [c for c in genotype_df.columns if c not in {ref_col, alt_col}]
        if not sample_cols:
            raise ValueError("Genotype DataFrame has no sample columns.")
        work = genotype_df.reset_index(drop=True).copy()
        work["_chrom"] = chrom_vals.astype(str).to_numpy()
        work["_pos"] = pd.to_numeric(pos_vals, errors="coerce").astype("Int64").to_numpy()
        chrom_col, pos_col = "_chrom", "_pos"
    else:
        work = genotype_df.copy()
        chrom_col, pos_col, ref_col, alt_col, sample_cols = _detect_genotype_columns(work)

    sample_cols_map = {str(c): c for c in sample_cols}
    keep_samples = [s for s in phenotype_samples if s in sample_cols_map]
    if not keep_samples:
        raise ValueError("No overlapping samples between phenotype and genotype.")

    site_keys = []
    site_calls = []
    for _, row in work.iterrows():
        chrom = str(row[chrom_col])
        pos = int(row[pos_col])
        ref = str(row[ref_col])
        alt = str(row[alt_col])
        site_key = (chrom, pos)
        calls = [
            _convert_call_to_allele(row[sample_cols_map[s]], ref, alt) for s in keep_samples
        ]
        site_keys.append(site_key)
        site_calls.append(calls)

    if not site_calls:
        raise ValueError("No SNP sites available in genotype input.")

    mat = np.asarray(site_calls, dtype=object).T
    out = pd.DataFrame(mat, index=keep_samples, columns=site_keys)
    return out


def _chunks_to_sample_site(
    chunks: GenotypeChunks, phenotype_samples: Sequence[str], chunk_sample_ids: Optional[Sequence[str]]
) -> pd.DataFrame:
    sample_ids = [str(x) for x in (chunk_sample_ids if chunk_sample_ids is not None else phenotype_samples)]
    if not sample_ids:
        raise ValueError("Sample IDs are required for chunk genotype input.")

    site_keys = []
    site_calls = []
    for chunk, sites in chunks:
        arr = np.asarray(chunk)
        if arr.ndim != 2:
            raise ValueError("Each chunk must be a 2D array with shape [n_sites, n_samples].")
        if arr.shape[0] != len(sites):
            raise ValueError("Chunk site count does not match site metadata length.")
        if arr.shape[1] != len(sample_ids):
            raise ValueError(
                f"Chunk sample count ({arr.shape[1]}) != sample_ids length ({len(sample_ids)})."
            )

        for i, site in enumerate(sites):
            chrom, pos, ref, alt = _parse_site_meta(site)
            site_key = (chrom, pos)
            calls = [_convert_call_to_allele(v, ref, alt) for v in arr[i, :]]
            site_keys.append(site_key)
            site_calls.append(calls)

    if not site_calls:
        raise ValueError("No SNP sites loaded from chunk genotype input.")

    mat = np.asarray(site_calls, dtype=object).T
    df = pd.DataFrame(mat, index=sample_ids, columns=site_keys)
    keep = [s for s in phenotype_samples if s in df.index]
    if not keep:
        raise ValueError("No overlapping samples between phenotype and chunk genotype.")
    return df.loc[keep]


def _group_values_by_index(
    data: pd.DataFrame, value_col: str, group_cols: Sequence[object], group_index: pd.Index
) -> list[np.ndarray]:
    out = []
    for g in group_index:
        key = g if isinstance(g, tuple) else (g,)
        if len(group_cols) == 1:
            mask = data[group_cols[0]].to_numpy() == key[0]
        else:
            mask = np.logical_and.reduce(
                [data[col].to_numpy() == key[i] for i, col in enumerate(group_cols)]
            )
        out.append(data.loc[mask, value_col].astype(float).to_numpy())
    return out


def _wilson_ci(k: float, n: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    delta = z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    low = max(0.0, center - delta)
    high = min(1.0, center + delta)
    return float(low), float(high)


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = int(p.size)
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * float(p[idx])
        if val < prev:
            val = prev
        prev = val
        adjusted[idx] = min(1.0, val)
    return adjusted


def _compact_letter_display_local(
    sig_matrix: pd.DataFrame,
    means: pd.Series,
) -> dict[object, str]:
    groups = list(means.sort_values(ascending=False).index)
    letter_sets: list[set] = []
    group_letters: dict[object, set] = {g: set() for g in groups}

    for g in groups:
        placed = False
        for i, members in enumerate(letter_sets):
            if all(not bool(sig_matrix.at[g, m]) for m in members):
                members.add(g)
                group_letters[g].add(i)
                placed = True
        if not placed:
            letter_sets.append({g})
            group_letters[g].add(len(letter_sets) - 1)

    for i, members in enumerate(letter_sets):
        for g in groups:
            if g in members:
                continue
            if all(not bool(sig_matrix.at[g, m]) for m in members):
                members.add(g)
                group_letters[g].add(i)

    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(letter_sets) > len(letters):
        raise ValueError("Too many significance letter groups (>52).")
    out: dict[object, str] = {}
    for g in groups:
        idxs = sorted(group_letters[g])
        out[g] = "".join(letters[i] for i in idxs)
    return out


def _build_binomial_summary(
    merged: pd.DataFrame,
    site_cols: Sequence[object],
    ascending: bool,
    alpha: float = 0.05,
) -> pd.DataFrame:
    vals = pd.to_numeric(merged["_pheno"], errors="coerce")
    if vals.isna().any():
        raise ValueError("mode='binomial' requires phenotype values encoded as numeric 0/1.")
    uniq = {float(x) for x in pd.unique(vals)}
    if not uniq.issubset({0.0, 1.0}):
        shown = ", ".join(str(x) for x in sorted(uniq))
        raise ValueError(
            "mode='binomial' requires phenotype values in {0,1}; "
            f"got: {shown}."
        )

    merged = merged.copy()
    merged["_pheno"] = vals.astype(float)

    grouped = merged.groupby(list(site_cols))["_pheno"]
    n = grouped.size().astype(int)
    k = grouped.sum().round().astype(int)
    mean = (k / n).astype(float)

    ci_low = []
    ci_high = []
    for kk, nn in zip(k.to_numpy(dtype=float), n.to_numpy(dtype=float)):
        lo, hi = _wilson_ci(kk, nn)
        ci_low.append(lo)
        ci_high.append(hi)

    out = pd.DataFrame(
        {
            "mean": mean,
            "n": n,
            "k": k,
            "ci_low": np.asarray(ci_low, dtype=float),
            "ci_high": np.asarray(ci_high, dtype=float),
        }
    )
    out["label"] = [f"{int(kk)}/{int(nn)}" for kk, nn in zip(out["k"], out["n"])]
    out["letters"] = ""
    out["sem"] = np.nan
    out = out.sort_values("mean", ascending=bool(ascending))

    if out.shape[0] >= 2:
        try:
            from scipy.stats import chi2_contingency, fisher_exact
        except Exception as e:
            raise RuntimeError("mode='binomial' significance test requires scipy.") from e

        gidx = list(out.index)
        succ = out["k"].astype(int)
        fail = (out["n"] - out["k"]).astype(int)

        table = np.column_stack([succ.loc[gidx].to_numpy(), fail.loc[gidx].to_numpy()])
        p_omnibus = np.nan
        try:
            _chi2, p_omnibus, _dof, _exp = chi2_contingency(table, correction=False)
        except Exception:
            p_omnibus = np.nan

        sig = pd.DataFrame(False, index=gidx, columns=gidx, dtype=bool)
        means = out["mean"].astype(float)

        if len(gidx) == 2:
            g1, g2 = gidx
            _, p = fisher_exact(
                [[int(succ.loc[g1]), int(fail.loc[g1])], [int(succ.loc[g2]), int(fail.loc[g2])]]
            )
            reject = bool(np.isfinite(p) and float(p) < float(alpha))
            sig.at[g1, g2] = reject
            sig.at[g2, g1] = reject
        elif np.isfinite(p_omnibus) and float(p_omnibus) < float(alpha):
            pairs: list[tuple[object, object]] = []
            pvals: list[float] = []
            for i in range(len(gidx)):
                for j in range(i + 1, len(gidx)):
                    gi, gj = gidx[i], gidx[j]
                    _, p = fisher_exact(
                        [
                            [int(succ.loc[gi]), int(fail.loc[gi])],
                            [int(succ.loc[gj]), int(fail.loc[gj])],
                        ]
                    )
                    pairs.append((gi, gj))
                    pvals.append(float(p))

            p_adj = _holm_adjust(np.asarray(pvals, dtype=float))
            for (gi, gj), pv in zip(pairs, p_adj):
                reject = bool(np.isfinite(pv) and float(pv) < float(alpha))
                sig.at[gi, gj] = reject
                sig.at[gj, gi] = reject

        letters_map = _compact_letter_display_local(sig, means)
        out["letters"] = [letters_map.get(g, "") for g in out.index]
        out.attrs["p_omnibus"] = float(p_omnibus) if np.isfinite(p_omnibus) else np.nan

    return out


def _format_haplotype_label(g: object) -> str:
    if isinstance(g, tuple):
        return "|".join(str(x) for x in g)
    return str(g)


def _normalize_orient(orient: str) -> str:
    o = str(orient).strip().lower()
    if o in {"h", "horizontal"}:
        return "horizontal"
    if o in {"v", "vertical"}:
        return "vertical"
    raise ValueError("`orient` must be 'horizontal'/'h' or 'vertical'/'v'.")


def _format_site_label(site: object) -> str:
    if isinstance(site, tuple) and len(site) >= 2:
        return f"{site[0]}:{site[1]}"
    return str(site)


def _site_to_tuple(site: object) -> Tuple[str, int]:
    if isinstance(site, tuple) and len(site) >= 2:
        chrom = str(site[0])
        try:
            pos = int(site[1])
        except (TypeError, ValueError):
            pos = 0
        return chrom, pos
    s = str(site)
    if ":" in s:
        chrom, pos = s.split(":", 1)
        try:
            return chrom, int(pos)
        except ValueError:
            return chrom, 0
    return s, 0


def _infer_adv_alleles(
    merged: pd.DataFrame,
    site_cols: Sequence[object],
    na_like: set[str],
) -> dict[object, str]:
    adv: dict[object, str] = {}
    for col in site_cols:
        sub = merged[[col, "_pheno"]].dropna()
        if sub.empty:
            continue
        means = sub.groupby(col)["_pheno"].mean().sort_values(ascending=False)
        if means.empty:
            continue
        picked = None
        for allele in means.index:
            a = str(allele)
            if a not in na_like and a != "H":
                picked = a
                break
        if picked is None:
            picked = str(means.index[0])
        adv[col] = picked
    return adv


def _draw_haplotype_matrix(
    ax: plt.Axes,
    geno_df: pd.DataFrame,
    merged: pd.DataFrame,
    advantage_sites: Optional[Sequence[Tuple[str, int]]] = None,
    palette_cfg: Optional[dict[str, object]] = None,
) -> None:
    palette_cfg = _resolve_palette(palette_cfg)
    matrix_labels = geno_df.astype(str).to_numpy()
    na_like = {"NA", "N", ".", "./.", "-", "nan"}
    nrow, ncol = matrix_labels.shape
    rgba = np.zeros((nrow, ncol, 4), dtype=float)
    rgba[:, :, :] = _safe_rgba("#efefef", "#efefef")

    use_adv = bool(advantage_sites)
    adv_site_set = {
        (str(chrom), int(pos)) for chrom, pos in (advantage_sites or [])
    }
    adv_alleles = _infer_adv_alleles(merged, list(geno_df.columns), na_like) if use_adv else {}

    hue_step = float(palette_cfg.get("matrix_hue_step", 0.61803398875))
    sat_dark = float(palette_cfg.get("matrix_sat_dark", 0.90))
    val_dark = float(palette_cfg.get("matrix_val_dark", 0.50))
    gray = tuple(palette_cfg.get("matrix_nonfav", _safe_rgba("#e8e8e8", "#e8e8e8")))
    na_gray = tuple(palette_cfg.get("matrix_na", _safe_rgba("#d9d9d9", "#d9d9d9")))
    cm_colors = _colormap_color_list(palette_cfg.get("matrix_colormap"), ncol)

    for j, col in enumerate(geno_df.columns):
        if cm_colors:
            allele_color = cm_colors[j]
        else:
            hue = (hue_step * (j + 1)) % 1.0
            allele_color = tuple(hsv_to_rgb((hue, sat_dark, val_dark)).tolist()) + (1.0,)
        het_color = _mix_rgb(allele_color, gray, 0.5, 0.5)

        col_vals = matrix_labels[:, j]
        non_na = [x for x in pd.unique(col_vals) if str(x) not in na_like]
        allele_main = [x for x in non_na if str(x) != "H"]
        ref_like = str(allele_main[0]) if len(allele_main) >= 1 else None
        alt_like = str(allele_main[1]) if len(allele_main) >= 2 else None

        site_key = _site_to_tuple(col)
        col_adv_mode = use_adv and site_key in adv_site_set
        favored = adv_alleles.get(col)
        if not col_adv_mode:
            if "ALT" in {str(x).upper() for x in allele_main}:
                favored = "ALT"
            elif alt_like is not None:
                favored = alt_like

        for i, raw in enumerate(col_vals):
            v = str(raw)
            if v in na_like:
                rgba[i, j, :] = na_gray
                continue
            if v == "H":
                rgba[i, j, :] = het_color
                continue

            # ALT / D-allele -> palette color.
            if favored is not None and v == str(favored):
                rgba[i, j, :] = allele_color
                continue
            if str(v).upper() in {"ALT", "D"}:
                rgba[i, j, :] = allele_color
                continue

            # REF / non-D-allele -> light gray.
            rgba[i, j, :] = gray

    # Keep row order consistent with bar plotting order (first group at the bottom).
    ax.imshow(rgba, aspect="auto", interpolation="nearest", origin="lower")

    for i in range(nrow):
        for j in range(ncol):
            ax.text(
                j,
                i,
                matrix_labels[i, j],
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    ax.set_xticks(
        np.arange(ncol),
        [_format_site_label(c) for c in geno_df.columns],
        # rotation=45,
        # ha="right",
    )
    ax.set_yticks(np.arange(nrow), [str(x) for x in geno_df.index])
    # ax.set_xlabel("Sites")
    # ax.set_ylabel("Haplotype")

    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)


def _build_haplotype_matrix_from_summary(
    summary: pd.DataFrame,
    site_cols: Sequence[object],
) -> pd.DataFrame:
    n_site = len(site_cols)
    rows: list[list[str]] = []
    idx: list[str] = []
    for i, g in enumerate(summary.index, start=1):
        if isinstance(g, tuple):
            vals = [str(x) for x in g]
        else:
            vals = [str(g)]
        if len(vals) < n_site:
            vals.extend(["NA"] * (n_site - len(vals)))
        rows.append(vals[:n_site])
        idx.append(f"Hap {i}")
    return pd.DataFrame(rows, index=idx, columns=list(site_cols))


def plot_haplotype(
    phenotype: pd.DataFrame,
    genotype: GenotypeInput,
    draw_letters: bool = False,
    draw_bar: bool = False,
    draw_matrix: bool = False,
    advantage_sites: Optional[Sequence[Tuple[str, int]]] = None,
    palette: Optional[Union[str, dict]] = "default",
    min_haplotype_n: int = 30,
    ascending: bool = False,
    mode: Literal["continuous", "binomial"] = "continuous",
    alpha: float = 0.05,
    orient: Literal['vertical','horizontal'] = "horizontal",
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> Tuple[plt.Axes, pd.DataFrame]:
    """
    Plot phenotype distribution grouped by haplotype.

    Parameters
    ----------
    phenotype
        DataFrame with sample IDs + phenotype values.
        Preferred format:
          - one value column with sample IDs in index; or
          - first column sample IDs, second column phenotype values.
    genotype
        Either:
          1) DataFrame with columns like [chr, pos, ref, alt, sample...]
             (or MultiIndex index=(chr,pos) + REF/A0 ALT/A1 columns),
          2) chunks iterator from `load_genotype_chunks`,
          3) (chunks, sample_ids) tuple.
    draw_letters
        Draw annotations:
        - continuous mode: compact-letter-display;
        - binomial mode: significance letters + k/n label.
    draw_bar
        Draw bar panel:
        - continuous mode: violin+box+errorbar+scatter;
        - binomial mode: proportion bar+95% CI+scatter.
    draw_matrix
        Draw sample-by-site genotype matrix using imshow and per-cell allele labels.
    advantage_sites
        Optional list of site tuples: [(chr, pos), ...].
        If provided, for listed columns only the inferred advantageous allele is colored;
        non-advantage alleles are rendered in light gray.
        If not provided, each column uses a high-contrast same-hue scheme for ref/alt-like alleles.
    palette
        Overall color palette preset or custom dict.
        Presets: 'default', 'pastel', 'deep', 'colorblind', 'mono'.
    min_haplotype_n
        Minimum sample count per haplotype group to keep for statistics and plotting.
        Groups with n < min_haplotype_n are removed (default: 30).
    ascending
        Sort haplotype plotting order by phenotype mean.
        False: high-to-low (default), True: low-to-high.
    mode
        `continuous`: quantitative phenotype mode.
        `binomial`: binary phenotype mode (0/1).
    alpha
        Significance threshold used by statistical tests (default: 0.05).
    orient
        'horizontal'/'h' or 'vertical'/'v'.
    ax
        Matplotlib Axes. If None, create a new figure and axis.

    Returns
    -------
    (ax, summary_df)
        continuous columns: ['mean', 'n', 'letters', 'sem'].
        binomial adds: ['k', 'ci_low', 'ci_high', 'label'].
    """
    legacy_draw_violin = kwargs.pop("draw_violin", None)
    legacy_draw_box = kwargs.pop("draw_box", None)
    legacy_pallete = kwargs.pop("pallete", None)
    if kwargs:
        bad = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword argument(s): {bad}")

    if legacy_pallete is not None:
        if palette not in (None, "default"):
            raise ValueError("Use only one of `palette` or legacy `pallete`.")
        palette = legacy_pallete

    if legacy_draw_violin is not None or legacy_draw_box is not None:
        draw_bar = bool(draw_bar) or bool(legacy_draw_violin) or bool(legacy_draw_box)

    if not draw_matrix and not draw_bar:
        raise ValueError(
            "At least one of `draw_bar` or `draw_matrix` must be True."
        )
    if int(min_haplotype_n) < 1:
        raise ValueError("`min_haplotype_n` must be >= 1.")
    palette_cfg = _resolve_palette(palette)

    orient = _normalize_orient(orient)
    vert = orient == "vertical"
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    pheno = _normalize_phenotype(phenotype)
    pheno_name = pheno.name if pheno.name is not None else "phenotype"
    phenotype_samples = pheno.index.tolist()

    if isinstance(genotype, pd.DataFrame):
        geno_sample_site = _genotype_dataframe_to_sample_site(genotype, phenotype_samples)
    elif isinstance(genotype, tuple) and len(genotype) == 2:
        geno_sample_site = _chunks_to_sample_site(
            genotype[0], phenotype_samples, genotype[1]
        )
    else:
        geno_sample_site = _chunks_to_sample_site(genotype, phenotype_samples, None)

    merged = pd.concat([pheno.rename("_pheno"), geno_sample_site], axis=1, join="inner")
    merged = merged.dropna()
    if merged.empty:
        raise ValueError("No valid samples after merging phenotype and genotype.")

    site_cols = list(geno_sample_site.columns)
    if len(site_cols) == 1:
        merged["_hap_key"] = merged[site_cols[0]]
    else:
        merged["_hap_key"] = list(merged[site_cols].itertuples(index=False, name=None))

    hap_counts_all = merged.groupby("_hap_key")["_pheno"].size()
    keep_keys = hap_counts_all[hap_counts_all >= int(min_haplotype_n)].index
    removed_counts = hap_counts_all[hap_counts_all < int(min_haplotype_n)]
    merged = merged.loc[merged["_hap_key"].isin(keep_keys)].copy()
    if merged.empty:
        raise ValueError(
            f"No haplotypes remain after filtering by min_haplotype_n={int(min_haplotype_n)}."
        )

    mode_key = str(mode).strip().lower()
    if mode_key in {"continuous", "cont"}:
        mode_key = "continuous"
    elif mode_key in {"binomial", "binary", "bin"}:
        mode_key = "binomial"
    else:
        raise ValueError("`mode` must be 'continuous' or 'binomial'.")

    if mode_key == "continuous":
        from janusx.bioplotkit.stat import multiple_comparison_groupby

        summary = multiple_comparison_groupby(
            merged,
            value_col="_pheno",
            group_cols=site_cols,
            alpha=float(alpha),
        )
        summary["sem"] = merged.groupby(site_cols)["_pheno"].sem()
        summary = summary.sort_values("mean", ascending=bool(ascending))
    else:
        summary = _build_binomial_summary(
            merged,
            site_cols,
            ascending=bool(ascending),
            alpha=float(alpha),
        )

    summary.attrs["min_haplotype_n"] = int(min_haplotype_n)
    summary.attrs["removed_haplotype_groups"] = int(removed_counts.shape[0])
    summary.attrs["removed_haplotype_samples"] = int(removed_counts.sum())
    summary.attrs["mode"] = mode_key

    group_values = _group_values_by_index(
        merged, "_pheno", site_cols, summary.index
    )
    positions = np.arange(1, len(group_values) + 1)

    if draw_bar:
        base_color = tuple(palette_cfg["violin_face"])
        violin_fill = _mix_rgb(base_color, (1.0, 1.0, 1.0, 1.0), 0.45, 0.55)
        scatter_color = _mix_rgb(base_color, (0.0, 0.0, 0.0, 1.0), 0.78, 0.22)
        if mode_key == "binomial":
            mean_vals = summary["mean"].astype(float).to_numpy()
            ci_low = summary["ci_low"].astype(float).to_numpy()
            ci_high = summary["ci_high"].astype(float).to_numpy()
            err_low = np.clip(mean_vals - ci_low, 0.0, 1.0)
            err_high = np.clip(ci_high - mean_vals, 0.0, 1.0)

            bar_fill = _mix_rgb(base_color, (1.0, 1.0, 1.0, 1.0), 0.50, 0.50)
            edge_color = (0.0, 0.0, 0.0, 1.0)
            point_face = scatter_color
            point_edge = scatter_color

            if vert:
                ax.bar(
                    positions,
                    mean_vals,
                    width=0.56,
                    color=bar_fill,
                    edgecolor=edge_color,
                    linewidth=1.0,
                    zorder=1.5,
                )
                ax.errorbar(
                    positions,
                    mean_vals,
                    yerr=np.vstack([err_low, err_high]),
                    fmt="none",
                    ecolor=edge_color,
                    elinewidth=1.0,
                    capsize=2.0,
                    capthick=1.0,
                    zorder=2.2,
                )
            else:
                ax.barh(
                    positions,
                    mean_vals,
                    height=0.56,
                    color=bar_fill,
                    edgecolor=edge_color,
                    linewidth=1.0,
                    zorder=1.5,
                )
                ax.errorbar(
                    mean_vals,
                    positions,
                    xerr=np.vstack([err_low, err_high]),
                    fmt="none",
                    ecolor=edge_color,
                    elinewidth=1.0,
                    capsize=2.0,
                    capthick=1.0,
                    zorder=2.2,
                )

            rng = np.random.default_rng(42)
            for pos, vals in zip(positions, group_values):
                if vals.size == 0:
                    continue
                v = np.clip(vals.astype(float), 0.0, 1.0)
                jitter_a = rng.normal(loc=pos, scale=0.035, size=v.size)
                jitter_b = rng.normal(loc=0.0, scale=0.015, size=v.size)
                if vert:
                    yy = np.clip(v + jitter_b, 0.0, 1.0)
                    ax.scatter(
                        jitter_a,
                        yy,
                        s=8,
                        c=[point_face],
                        edgecolors=[point_edge],
                        linewidths=0.4,
                        alpha=float(palette_cfg["point_alpha"]),
                        zorder=3.0,
                    )
                else:
                    xx = np.clip(v + jitter_b, 0.0, 1.0)
                    ax.scatter(
                        xx,
                        jitter_a,
                        s=8,
                        c=[point_face],
                        edgecolors=[point_edge],
                        linewidths=0.4,
                        alpha=float(palette_cfg["point_alpha"]),
                        zorder=3.0,
                    )

            hap_labels = [_format_haplotype_label(g) for g in summary.index]
            if vert:
                ax.set_xticks(positions, hap_labels, ha="right")
                ax.set_ylabel(f"{pheno_name} (proportion)")
                ax.set_ylim(0.0, 1.06)
            else:
                ax.set_yticks(positions, hap_labels)
                ax.set_xlabel(f"{pheno_name} (proportion)")
                ax.set_xlim(0.0, 1.06)

            if draw_letters:
                labels = summary["label"].astype(str).tolist()
                letters = summary["letters"].astype(str).tolist()
                annos = [f"{lt}\n{lb}" if lt.strip() else lb for lt, lb in zip(letters, labels)]
                if vert:
                    for pos, y, txt in zip(positions, mean_vals, annos):
                        ax.text(
                            pos,
                            min(1.05, float(y) + 0.03),
                            txt,
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            color="black",
                            zorder=4.0,
                        )
                else:
                    for pos, x, txt in zip(positions, mean_vals, annos):
                        ax.text(
                            min(1.05, float(x) + 0.03),
                            pos,
                            txt,
                            ha="left",
                            va="center",
                            fontsize=9,
                            color="black",
                            zorder=4.0,
                        )
        else:
            vp = ax.violinplot(
                group_values,
                positions=positions,
                vert=vert,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            for body in vp.get("bodies", []):
                body.set_facecolor(violin_fill)
                body.set_edgecolor("none")
                body.set_linewidth(0.0)
                body.set_alpha(max(0.35, float(palette_cfg["violin_alpha"]) * 0.75))
                body.set_zorder(1.0)
            for key in ("cmeans", "cbars", "cmins", "cmaxes"):
                artist = vp.get(key, None)
                if artist is not None:
                    artist.set_visible(False)

            bp = ax.boxplot(
                group_values,
                positions=positions,
                vert=vert,
                widths=0.22,
                patch_artist=True,
                showfliers=False,
                boxprops={
                    "facecolor": (1.0, 1.0, 1.0, 0.96),
                    "edgecolor": (0.0, 0.0, 0.0, 1.0),
                    "linewidth": 1.0,
                },
                whiskerprops={"color": (0.0, 0.0, 0.0, 1.0), "linewidth": 1.0},
                capprops={"color": (0.0, 0.0, 0.0, 1.0), "linewidth": 1.0},
                medianprops={"color": (0.0, 0.0, 0.0, 1.0), "linewidth": 1.2},
            )
            for patch in bp.get("boxes", []):
                patch.set_facecolor((1.0, 1.0, 1.0, 0.96))
                patch.set_edgecolor((0.0, 0.0, 0.0, 1.0))
                patch.set_zorder(2.0)
            for key in ("whiskers", "caps", "medians"):
                for artist in bp.get(key, []):
                    artist.set_color((0.0, 0.0, 0.0, 1.0))
                    artist.set_zorder(2.1)

            mean_vals = summary["mean"].astype(float).to_numpy()
            sem_vals = summary["sem"].astype(float).fillna(0.0).to_numpy()
            err_color = (0.0, 0.0, 0.0, 1.0)
            if vert:
                ax.errorbar(
                    positions,
                    mean_vals,
                    yerr=sem_vals,
                    fmt="none",
                    ecolor=err_color,
                    elinewidth=1.0,
                    capsize=0.0,
                    capthick=1.0,
                    zorder=2.2,
                )
            else:
                ax.errorbar(
                    mean_vals,
                    positions,
                    xerr=sem_vals,
                    fmt="none",
                    ecolor=err_color,
                    elinewidth=1.0,
                    capsize=0.0,
                    capthick=1.0,
                    zorder=2.2,
                )

            point_face = scatter_color
            point_edge = scatter_color
            rng = np.random.default_rng(42)
            for pos, vals in zip(positions, group_values):
                if vals.size == 0:
                    continue
                jitter = rng.normal(loc=pos, scale=0.04, size=vals.size)
                if vert:
                    ax.scatter(
                        jitter,
                        vals,
                        s=8,
                        c=[point_face],
                        edgecolors=[point_edge],
                        linewidths=0.4,
                        alpha=float(palette_cfg["point_alpha"]),
                        zorder=3.0,
                    )
                else:
                    ax.scatter(
                        vals,
                        jitter,
                        s=8,
                        c=[point_face],
                        edgecolors=[point_edge],
                        linewidths=0.4,
                        alpha=float(palette_cfg["point_alpha"]),
                        zorder=3.0,
                    )

            hap_labels = [_format_haplotype_label(g) for g in summary.index]
            if vert:
                ax.set_xticks(positions, hap_labels, ha="right")
                ax.set_ylabel(pheno_name)
            else:
                ax.set_yticks(positions, hap_labels)
                ax.set_xlabel(pheno_name)

            if draw_letters:
                all_vals = np.concatenate([v for v in group_values if v.size > 0]) if any(
                    v.size > 0 for v in group_values
                ) else np.array([0.0])
                vmin = float(np.nanmin(all_vals))
                vmax = float(np.nanmax(all_vals))
                span = max(vmax - vmin, 1e-9)
                pad = span * 0.05
                letters_seq = summary["letters"].astype(str).tolist()
                means_seq = summary["mean"].astype(float).tolist()

                for pos, vals, letter, mean_v in zip(
                    positions, group_values, letters_seq, means_seq
                ):
                    anchor = float(np.nanmax(vals)) if vals.size > 0 else float(mean_v)
                    if vert:
                        ax.text(
                            pos,
                            anchor + pad,
                            letter,
                            ha="center",
                            va="bottom",
                            fontsize=10,
                            color="black",
                        )
                    else:
                        ax.text(
                            anchor + pad,
                            pos,
                            letter,
                            ha="left",
                            va="center",
                            fontsize=10,
                            color="black",
                        )

                if vert:
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim(ymin, ymax + pad * 2.0)
                else:
                    xmin, xmax = ax.get_xlim()
                    ax.set_xlim(xmin, xmax + pad * 2.0)

    if draw_matrix:
        matrix_ax = ax
        if draw_bar:
            matrix_ax = ax.inset_axes([0.05, -0.68, 0.9, 0.55], transform=ax.transAxes)
        hap_matrix = _build_haplotype_matrix_from_summary(summary, site_cols)
        _draw_haplotype_matrix(
            matrix_ax,
            hap_matrix,
            merged,
            advantage_sites,
            palette_cfg=palette_cfg,
        )

    return ax, summary


__all__ = ["plot_haplotype"]
