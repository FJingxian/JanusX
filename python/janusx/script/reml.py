# -*- coding: utf-8 -*-
"""
JanusX: REML/BLUP Heritability Estimation from Phenotype Table

Input table
-----------
- First column is always sample ID.
- Remaining columns are candidate phenotype/fixed/random effect columns.
- String/categorical columns selected by `-c/-rc/-gxe/-gxc` are encoded from
  the phenotype table.

Examples
--------
  jx reml -p pheno.tsv -n Yield -c year,loc -o outdir
  jx reml -p pheno.tsv -n Yield -c PCA1,PCA2 -rc block -k data.cGRM.npy
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import typing
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve
from scipy.stats import t as student_t

from janusx.assoc.workflow_model_packed import (
    jxrs,
    _splmm_normalize_sparse_grm_path,
    _splmm_sparse_grm_diag_stats,
    _splmm_sparse_null_fit,
)
from janusx.pyBLUP.blup import BLUP
from ._common.cli_args import (
    add_common_out_arg,
    add_common_prefix_arg,
    add_common_thread_arg,
)
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.outprefix import apply_output_prefix_compat
from ._common.config_render import emit_cli_configuration
from ._common.grmio import load_grm_matrix, read_id_file, resolve_grm_id_path
from ._common.pathcheck import ensure_file_exists, format_path_for_display
from ._common.genoio import strip_default_prefix_suffix
from ._common.progress import log_success, print_failure, format_elapsed, success_symbol
from ._common.threads import apply_outer_thread_cap, detect_effective_threads


@dataclass
class _TermSpec:
    name: str
    force_onehot: bool


@dataclass(frozen=True)
class _EffectSpec:
    """One parsed fixed/random/GxE/GxC effect from the phenotype table."""

    kind: str
    sources: tuple[str, ...]
    source_types: tuple[str, ...]
    label: str
    interaction: str | None = None

    @property
    def result_type(self) -> str:
        if len(self.sources) == 1:
            return self.source_types[0]
        if self.source_types == ("categorical", "categorical"):
            return "categorical"
        return "continuous"


@dataclass
class _CompiledModelTerms:
    fixed_specs: list[_EffectSpec]
    random_specs: list[_EffectSpec]
    gxe_specs: list[_EffectSpec]
    gxc_specs: list[_EffectSpec]
    fixed_matrix: np.ndarray | None
    fixed_names: list[str]
    fixed_labels: list[str]
    line_z: sparse.csr_matrix
    line_names: list[str]
    random_matrices: list[typing.Union[np.ndarray, sparse.spmatrix]]
    random_names: list[str]


@dataclass
class _GrmContext:
    matrix: np.ndarray
    ids: list[str]
    id_path: str | None
    index: dict[str, int]


@dataclass
class _SparseGrmContext:
    path: str
    ids: list[str]
    id_path: str | None
    index: dict[str, int]
    n_samples: int


@dataclass
class _Stage1BlueResult:
    sample_ids: list[str]
    values: np.ndarray
    noise_diag: np.ndarray | None = None


@dataclass
class _JointKernelResult:
    va: float
    vline: float
    h2_raw: float
    beta: np.ndarray
    add_blup: np.ndarray
    line_blup: np.ndarray
    noise_mean: float


_JOINT_VAR_FLOOR = 1e-10
_JOINT_LOG_FLOOR = -24.0
_JOINT_LOG_CEIL = 24.0
_JOINT_OBJ_PENALTY = 1e60


def _split_tokens(values: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    for v in list(values or []):
        s = str(v).strip()
        if s == "":
            continue
        for part in s.split(","):
            p = str(part).strip()
            if p != "":
                out.append(p)
    return out


def _sniff_sep(path: str) -> str:
    sample = ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(16):
                line = fh.readline()
                if not line:
                    break
                s = line.strip()
                if s != "":
                    sample = s
                    break
    except Exception:
        sample = ""

    if "\t" in sample:
        return "tab"
    if "," in sample:
        return "comma"
    return "whitespace"


def _candidate_orders(kind: str) -> list[str]:
    if kind == "tab":
        return ["tab", "comma", "whitespace"]
    if kind == "comma":
        return ["comma", "tab", "whitespace"]
    return ["whitespace", "tab", "comma"]


def _looks_sample_header_token(token: object) -> bool:
    text = str(token).strip().lower()
    if text == "":
        return False
    norm = "".join(ch for ch in text if ch.isalnum())
    return norm in {
        "sampleid",
        "sample",
        "id",
        "iid",
        "fid",
        "taxa",
        "accession",
        "line",
    }


def _read_table_with_optional_header(path: str) -> pd.DataFrame:
    sniffed = _sniff_sep(path)
    read_err: Exception | None = None
    df: pd.DataFrame | None = None
    for mode in _candidate_orders(sniffed):
        try:
            kwargs: dict[str, object] = {
                "header": None,
                "low_memory": False,
            }
            if mode == "tab":
                kwargs["sep"] = "\t"
                kwargs["engine"] = "c"
            elif mode == "comma":
                kwargs["sep"] = ","
                kwargs["engine"] = "c"
            else:
                kwargs["sep"] = r"\s+"
                kwargs["engine"] = "c"
            df_try = pd.read_csv(path, **kwargs)
            if df_try.shape[1] <= 1:
                continue
            df = df_try
            break
        except Exception as ex:
            read_err = ex
            continue

    if df is None:
        if read_err is not None:
            raise read_err
        raise ValueError("Failed to read input table.")
    if df.empty:
        raise ValueError("Input file is empty.")

    header_like = False
    if df.shape[0] > 1 and df.shape[1] > 1:
        row0 = pd.to_numeric(df.iloc[0, 1:], errors="coerce")
        probe_stop = min(int(df.shape[0]), 33)
        probe_rows = df.iloc[1:probe_stop, 1:].apply(pd.to_numeric, errors="coerce")
        probe_has_numeric = bool(probe_rows.notna().to_numpy().any())
        if _looks_sample_header_token(df.iloc[0, 0]) or (row0.isna().all() and probe_has_numeric):
            header_like = True

    if header_like:
        raw_sample_name = df.iloc[0, 0]
        sample_name = "" if pd.isna(raw_sample_name) else str(raw_sample_name).strip()
        sample_name = sample_name or "sample_id"
        data_names: list[str] = []
        for idx, raw_name in enumerate(df.iloc[0, 1:].tolist(), start=1):
            name = "" if pd.isna(raw_name) else str(raw_name).strip()
            data_names.append(name if name != "" else f"V{idx}")
        df = df.iloc[1:, :].reset_index(drop=True)
        df.columns = [sample_name] + data_names
    else:
        df = df.copy()
        df.columns = ["sample_id"] + [f"V{i}" for i in range(1, int(df.shape[1]))]

    return df


def _resolve_columns(
    tokens: list[str],
    data_cols: list[str],
    label: str,
    *,
    index_base: int = 0,
) -> list[str]:
    if len(tokens) == 0:
        return []

    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1.")

    lower_map = {str(c).lower(): str(c) for c in data_cols}
    resolved: list[str] = []
    for tk in tokens:
        t = str(tk).strip()
        if t == "":
            continue
        # Support numeric range syntax like "0:3" / "0-3" (inclusive).
        range_sep = None
        if ":" in t:
            range_sep = ":"
        elif t.count("-") == 1 and not t.startswith("-"):
            range_sep = "-"
        if range_sep is not None:
            left, right = t.split(range_sep, 1)
            if left.lstrip("+-").isdigit() and right.lstrip("+-").isdigit():
                start, end = int(left), int(right)
                step = 1 if end >= start else -1
                for idx in range(start, end + step, step):
                    if index_base == 0:
                        valid = (0 <= idx < len(data_cols))
                        ix = idx
                        valid_msg = f"[0..{max(0, len(data_cols)-1)}]"
                    else:
                        valid = (1 <= idx <= len(data_cols))
                        ix = idx - 1
                        valid_msg = f"[1..{len(data_cols)}]"
                    if not valid:
                        raise ValueError(
                            f"{label} column index out of range: {idx}. valid={valid_msg}"
                        )
                    resolved.append(str(data_cols[ix]))
                continue
        if t.lstrip("+-").isdigit():
            idx = int(t)
            if index_base == 0:
                valid = (0 <= idx < len(data_cols))
                ix = idx
                valid_msg = f"[0..{max(0, len(data_cols)-1)}]"
            else:
                valid = (1 <= idx <= len(data_cols))
                ix = idx - 1
                valid_msg = f"[1..{len(data_cols)}]"
            if not valid:
                raise ValueError(
                    f"{label} column index out of range: {idx}. valid={valid_msg}"
                )
            resolved.append(str(data_cols[ix]))
            continue
        if t in data_cols:
            resolved.append(str(t))
            continue
        t_low = t.lower()
        if t_low in lower_map:
            resolved.append(lower_map[t_low])
            continue
        raise ValueError(f"{label} column not found: {t}")

    # de-duplicate while preserving order
    out: list[str] = []
    seen: set[str] = set()
    for c in resolved:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _is_numeric_series(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce")
    non_na = int(series.notna().sum())
    if non_na == 0:
        return False
    finite = vals.notna() & np.isfinite(vals)
    return int(finite.sum()) == non_na


def _collect_numeric_required_mask(
    df: pd.DataFrame,
    terms: list[_TermSpec],
) -> pd.Series:
    mask = pd.Series(True, index=df.index, dtype=bool)
    for term in terms:
        if term.force_onehot:
            continue
        s = df[term.name]
        if _is_numeric_series(s):
            values = pd.to_numeric(s, errors="coerce")
            mask &= values.notna() & np.isfinite(values)
    return mask


def _collect_numeric_required_mask_specs(
    df: pd.DataFrame,
    specs: list[_EffectSpec],
) -> pd.Series:
    mask = pd.Series(True, index=df.index, dtype=bool)
    for spec in specs:
        for column, column_type in zip(spec.sources, spec.source_types):
            if column_type == "continuous":
                values = pd.to_numeric(df[column], errors="coerce")
                mask &= values.notna() & np.isfinite(values)
    return mask


def _encode_term_matrix(
    df_sub: pd.DataFrame,
    term: _TermSpec,
    *,
    for_random: bool,
    sparse_onehot: bool = False,
) -> tuple[typing.Union[np.ndarray, sparse.csr_matrix], list[str]]:
    s = df_sub[term.name]
    if (not term.force_onehot) and _is_numeric_series(s):
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
        return arr, [term.name]

    # Default rule: string/categorical columns are one-hot encoded.
    return _onehot_encode_series(
        s,
        prefix=term.name,
        drop_first=(not for_random),
        sparse_output=bool(sparse_onehot),
    )


def _onehot_encode_series(
    series: pd.Series,
    *,
    prefix: str,
    drop_first: bool,
    sparse_output: bool,
) -> tuple[typing.Union[np.ndarray, sparse.csr_matrix], list[str]]:
    ss = series.astype("string").fillna("NA").astype(str)
    n = int(ss.shape[0])
    if n == 0:
        empty = sparse.csr_matrix((0, 0), dtype=float)
        return (empty if sparse_output else np.zeros((0, 0), dtype=float)), []

    codes, levels = pd.factorize(ss, sort=True)
    n_levels = int(levels.shape[0])
    if n_levels == 0:
        empty = sparse.csr_matrix((n, 0), dtype=float)
        return (empty if sparse_output else np.zeros((n, 0), dtype=float)), []

    if drop_first:
        kept_levels = [str(x) for x in levels[1:]]
        mask = codes > 0
        cols = (codes[mask] - 1).astype(np.int64, copy=False)
        n_cols = max(0, n_levels - 1)
    else:
        kept_levels = [str(x) for x in levels]
        mask = codes >= 0
        cols = codes[mask].astype(np.int64, copy=False)
        n_cols = n_levels
    if n_cols == 0:
        empty = sparse.csr_matrix((n, 0), dtype=float)
        return (empty if sparse_output else np.zeros((n, 0), dtype=float)), []

    rows = np.nonzero(mask)[0].astype(np.int64, copy=False)
    data = np.ones(rows.shape[0], dtype=float)
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n, n_cols), dtype=float)
    names = [f"{prefix}-{lv}" for lv in kept_levels]
    return (mat if sparse_output else mat.toarray()), names


def _onehot_level_count(series: pd.Series) -> int:
    ss = series.astype("string").fillna("NA").astype(str)
    return int(ss.nunique(dropna=False))


def _format_onehot_terms_with_counts(
    df_sub: pd.DataFrame,
    cols: list[str],
    *,
    dropped: dict[str, int] | None = None,
) -> str:
    if len(cols) == 0:
        return "None"
    out: list[str] = []
    for c in cols:
        if c not in df_sub.columns:
            out.append(f"{c} (?)")
            continue
        lv = _onehot_level_count(df_sub[c])
        if dropped is not None and c in dropped:
            out.append(f"{c} ({lv}, dropped)")
        else:
            out.append(f"{c} ({lv})")
    return ", ".join(out)


def _is_repeat_like(name: str) -> bool:
    s = str(name).strip().lower()
    keys = ("rep", "repeat", "block", "plot")
    return any(k in s for k in keys)


def _is_env_like(name: str) -> bool:
    s = str(name).strip().lower()
    keys = ("env", "environment", "site", "location", "loc", "year", "season", "place")
    return any(k in s for k in keys)


def _combine_key(df_sub: pd.DataFrame, cols: list[str], default_label: str) -> pd.Series:
    if len(cols) == 0:
        return pd.Series([default_label] * len(df_sub), index=df_sub.index, dtype="string")
    z = df_sub[cols].astype("string").fillna("NA")
    if len(cols) == 1:
        return z.iloc[:, 0].astype("string")
    return z.agg("|".join, axis=1).astype("string")


def _infer_env_rep_columns(random_terms: list[_TermSpec]) -> tuple[list[str], list[str]]:
    all_terms = [str(t.name) for t in random_terms]
    rep_cols = [c for c in all_terms if _is_repeat_like(c)]
    env_cols = [c for c in all_terms if _is_env_like(c) and c not in rep_cols]
    return env_cols, rep_cols


def _unique_preserve(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = str(v)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _infer_model_env_rep_columns(
    data_cols: list[str],
    trait_cols: list[str],
    fixed_terms: list[_TermSpec],
    random_terms: list[_TermSpec],
) -> tuple[list[str], list[str]]:
    fixed_names = [str(t.name) for t in fixed_terms]
    random_names = [str(t.name) for t in random_terms]
    explicit = _unique_preserve(fixed_names + random_names)

    env_cols = [c for c in explicit if _is_env_like(c)]
    rep_cols = [c for c in explicit if _is_repeat_like(c) and c not in env_cols]

    if len(env_cols) == 0:
        env_cols = [
            c for c in data_cols
            if c not in trait_cols and _is_env_like(c)
        ]
    if len(rep_cols) == 0:
        rep_cols = [
            c for c in data_cols
            if c not in trait_cols and c not in env_cols and _is_repeat_like(c)
        ]

    return _unique_preserve(env_cols), _unique_preserve(rep_cols)


def _exclude_special_terms(
    terms: list[_TermSpec],
    special_cols: set[str],
) -> list[_TermSpec]:
    return [t for t in terms if str(t.name) not in special_cols]


def _harmonic_mean(values: Iterable[float | int]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return 1.0
    return float(arr.size / np.sum(1.0 / arr))


def _effective_env_plot_counts(
    sample_ids_sub: pd.Series,
    sub: pd.DataFrame,
    env_cols: list[str],
    rep_cols: list[str],
) -> tuple[float, float, float]:
    sid = sample_ids_sub.astype("string").fillna("NA").astype(str)
    env_key = _combine_key(sub, env_cols, "__ENV__").astype(str)
    env_df = pd.DataFrame({"sid": sid, "env": env_key})
    env_per_sid = env_df.drop_duplicates().groupby("sid", sort=False)["env"].nunique()
    h_env = max(1.0, _harmonic_mean(env_per_sid.to_numpy(dtype=float)))

    if len(rep_cols) > 0:
        rep_key = _combine_key(sub, rep_cols, "__REP__").astype(str)
        plot_key = env_key + "@@" + rep_key
        plot_df = pd.DataFrame({"sid": sid, "plot": plot_key})
        plot_per_sid = plot_df.drop_duplicates().groupby("sid", sort=False)["plot"].nunique()
    else:
        plot_per_sid = pd.DataFrame({"sid": sid}).groupby("sid", sort=False).size()
    h_plot = max(1.0, _harmonic_mean(plot_per_sid.to_numpy(dtype=float)))
    r_eff = max(1.0, float(h_plot / h_env))
    return h_env, h_plot, r_eff


def _effective_env_rep_counts(
    sample_ids_sub: pd.Series,
    sub: pd.DataFrame,
    env_cols: list[str],
    rep_cols: list[str],
) -> tuple[float, float]:
    e_eff, _h_plot, r_eff = _effective_env_plot_counts(sample_ids_sub, sub, env_cols, rep_cols)
    return e_eff, r_eff


def _gls_fixed_stats_from_blup(
    model: BLUP,
    z_list: list[typing.Union[np.ndarray, sparse.spmatrix]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta = np.asarray(model.beta, dtype=float).reshape(-1)
    n, p = int(model.X.shape[0]), int(model.X.shape[1])
    cov_beta_cached = getattr(model, "_cov_beta", None)
    if cov_beta_cached is not None:
        cov_beta = np.asarray(cov_beta_cached, dtype=float)
    elif model.theta is None or len(z_list) == 0:
        resid = np.asarray(model.residuals, dtype=float).reshape(-1, 1)
        sigma2 = float((resid.T @ resid)[0, 0]) / max(1, n - p)
        xtx = np.asarray(model.X.T @ model.X, dtype=float)
        cov_beta = np.linalg.pinv(xtx) * sigma2
    else:
        theta = np.asarray(model.theta, dtype=float).reshape(-1)
        if theta.size < (len(z_list) + 1):
            raise ValueError("BLUP theta size is inconsistent with random effects.")

        # Reconstruct the same K-list definition used in BLUP._fit()
        z_standardized = bool(getattr(model, "_z_standardized", True))
        k_list: list[np.ndarray] = []
        for i, z in enumerate(z_list):
            z_arr = np.asarray(z.toarray(), dtype=float) if sparse.issparse(z) else np.asarray(z, dtype=float)
            if z_standardized:
                q, mean_vec, std_vec = model.onehot_info[i]
                z_fit = (
                    (z_arr - np.asarray(mean_vec, dtype=float))
                    / np.asarray(std_vec, dtype=float)
                    / np.sqrt(float(q))
                )
            else:
                z_fit = z_arr
            k_list.append(np.asarray(z_fit @ z_fit.T, dtype=float))

        v = np.eye(n, dtype=float) * float(theta[-1])
        for i, k in enumerate(k_list):
            v += float(theta[i]) * k
        l = np.linalg.cholesky(v)
        vinvx = cho_solve((l, True), np.asarray(model.X, dtype=float))
        xt_vinv_x = np.asarray(model.X.T @ vinvx, dtype=float)
        cov_beta = np.linalg.pinv(xt_vinv_x)

        r = np.asarray(model.residuals, dtype=float).reshape(-1, 1)
        vinvr = cho_solve((l, True), r)
        sigma2 = float((r.T @ vinvr)[0, 0]) / max(1, n - p)
        cov_beta = cov_beta * sigma2

    se = np.sqrt(np.clip(np.diag(cov_beta), a_min=0.0, a_max=None))
    tval = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pval = 2.0 * student_t.sf(np.abs(tval), df=max(1, n - p))
    return beta, se, pval


def _fmt_metric(value: float | int | np.floating | np.integer | None) -> str:
    try:
        v = float(value)  # type: ignore[arg-type]
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.6g}"


def _resolve_trait_columns_auto(
    df: pd.DataFrame,
    candidate_cols: list[str],
) -> list[str]:
    out: list[str] = []
    for c in candidate_cols:
        if _is_numeric_series(df[c]):
            out.append(str(c))
    return out


def _format_design_label(env_cols: list[str], fixed_cols: list[str]) -> str:
    parts: list[str] = []
    if len(env_cols) > 0:
        parts.append("|".join(env_cols))
    if len(fixed_cols) > 0:
        parts.append(", ".join(fixed_cols))
    return " / ".join(parts) if len(parts) > 0 else "None"


def _format_random_label(random_cols: list[str]) -> str:
    return ", ".join(random_cols) if len(random_cols) > 0 else "None"


def _render_summary_table(
    rows: list[dict[str, typing.Any]],
    *,
    log_style: bool,
) -> str:
    if len(rows) == 0:
        return ""

    if log_style:
        headers = ["Trait", "N_Obs (Lines)", "Env / Fixed", "Random", "H2 (Broad)", "h2 (Narrow)"]
    else:
        headers = ["Trait", "N_Obs(Lines)", "Env/Fixed", "Random", "H2", "h2"]

    body: list[list[str]] = []
    for row in rows:
        try:
            nobs_value = float(row.get("used_obs", 0))
            nobs = int(nobs_value) if np.isfinite(nobs_value) else 0
        except Exception:
            nobs = 0
        try:
            nlines_value = float(row.get("used_lines", 0))
            nlines = int(nlines_value) if np.isfinite(nlines_value) else 0
        except Exception:
            nlines = 0
        obs_label = f"{nobs:,} ({nlines:,})" if log_style else f"{nobs:,}({nlines:,})"
        body.append(
            [
                str(row.get("trait", "NA")),
                obs_label,
                str(row.get("env_fixed_label", "None")),
                str(row.get("random_label", "None")),
                _fmt_metric(row.get("hsqr")),
                _fmt_metric(row.get("h2_narrow")),
            ]
        )

    widths = [len(h) for h in headers]
    for rec in body:
        for i, cell in enumerate(rec):
            widths[i] = max(widths[i], len(cell))

    fmt = " ".join("{:<" + str(w) + "}" for w in widths)
    lines: list[str] = []
    if log_style:
        total_w = sum(widths) + (len(widths) - 1)
        lines.append("============================= SUMMARY ==================================")
        lines.append(fmt.format(*headers))
        lines.append("-" * total_w)
        for rec in body:
            lines.append(fmt.format(*rec))
        lines.append("=" * 72)
    else:
        lines.append(fmt.format(*headers))
        lines.append("-" * (sum(widths) + (len(widths) - 1)))
        for rec in body:
            lines.append(fmt.format(*rec))
    return "\n".join(lines)


def _load_grm_context(
    grm_path: str,
    grm_id_path: str | None,
    fallback_ids: list[str],
) -> _GrmContext:
    grm = np.asarray(load_grm_matrix(grm_path), dtype=np.float64)
    id_path = resolve_grm_id_path(grm_path, grm_id_path)
    if id_path is not None:
        ids = [str(x) for x in read_id_file(id_path)]
        if len(ids) != int(grm.shape[0]):
            raise ValueError(
                f"GRM ID count mismatch: matrix n={grm.shape[0]} but ID file has {len(ids)} rows."
            )
    else:
        if int(grm.shape[0]) != len(fallback_ids):
            raise ValueError(
                f"GRM shape {grm.shape} does not match phenotype unique line count {len(fallback_ids)}, "
                "and no GRM ID file was found for reordering."
            )
        ids = [str(x) for x in fallback_ids]

    index: dict[str, int] = {}
    for i, sid in enumerate(ids):
        if sid in index:
            raise ValueError(f"GRM IDs contain duplicate line ID: {sid}")
        index[sid] = i
    return _GrmContext(matrix=grm, ids=ids, id_path=id_path, index=index)


def _load_sparse_grm_context(
    grm_path: str,
    grm_id_path: str | None,
    fallback_ids: list[str],
) -> _SparseGrmContext:
    sparse_path = _splmm_normalize_sparse_grm_path(grm_path)
    diag = _splmm_sparse_grm_diag_stats(sparse_path, None)
    n_samples_raw = float(diag.get("n_samples", float("nan")))
    if (not np.isfinite(n_samples_raw)) or int(n_samples_raw) <= 0:
        raise ValueError(
            f"Sparse GRM sample size is unavailable or invalid: path={sparse_path}, n={n_samples_raw}"
        )
    n_samples = int(n_samples_raw)
    id_path = resolve_grm_id_path(sparse_path, grm_id_path)
    if id_path is not None:
        ids = [str(x) for x in read_id_file(id_path)]
        if len(ids) != n_samples:
            raise ValueError(
                f"Sparse GRM ID count mismatch: sparse n={n_samples} but ID file has {len(ids)} rows."
            )
    else:
        if n_samples != len(fallback_ids):
            raise ValueError(
                f"Sparse GRM shape n={n_samples} does not match phenotype unique line count {len(fallback_ids)}, "
                "and no Sparse GRM ID file was found for reordering."
            )
        ids = [str(x) for x in fallback_ids]

    index: dict[str, int] = {}
    for i, sid in enumerate(ids):
        if sid in index:
            raise ValueError(f"Sparse GRM IDs contain duplicate line ID: {sid}")
        index[sid] = i
    return _SparseGrmContext(
        path=sparse_path,
        ids=ids,
        id_path=id_path,
        index=index,
        n_samples=n_samples,
    )


def _term_constant_within_line(
    sub: pd.DataFrame,
    line_col: str,
    term_name: str,
) -> bool:
    if term_name not in sub.columns:
        return False
    grouped = (
        sub[[line_col, term_name]]
        .groupby(line_col, sort=False)[term_name]
        .nunique(dropna=False)
    )
    if grouped.empty:
        return False
    return bool((grouped <= 1).all())


def _encode_fixed_design(
    df_sub: pd.DataFrame,
    terms: list[_TermSpec],
    *,
    trait: str,
    logger: typing.Any,
) -> tuple[np.ndarray | None, list[str]]:
    x_blocks: list[np.ndarray] = []
    x_names: list[str] = []
    for term in terms:
        arr, names = _encode_term_matrix(df_sub, term, for_random=False, sparse_onehot=False)
        arr = np.asarray(arr, dtype=float)
        if int(arr.shape[1]) == 0:
            logger.warning(f"Trait {trait}: fixed term `{term.name}` expanded to 0 columns; skipped.")
            continue
        x_blocks.append(arr)
        x_names.extend(names)
    if len(x_blocks) == 0:
        return None, []
    return np.concatenate(x_blocks, axis=1), x_names


def _encode_random_design(
    df_sub: pd.DataFrame,
    terms: list[_TermSpec],
    *,
    trait: str,
    logger: typing.Any,
) -> tuple[list[typing.Union[np.ndarray, sparse.spmatrix]], list[str]]:
    z_list: list[typing.Union[np.ndarray, sparse.spmatrix]] = []
    z_names: list[str] = []
    for term in terms:
        arr, _ = _encode_term_matrix(df_sub, term, for_random=True, sparse_onehot=True)
        if int(arr.shape[1]) == 0:
            logger.warning(f"Trait {trait}: random term `{term.name}` expanded to 0 columns; skipped.")
            continue
        z_list.append(arr)
        z_names.append(str(term.name))
    return z_list, z_names


def _build_stage1_blue_terms(
    sub: pd.DataFrame,
    *,
    line_col: str,
    trait: str,
    fixed_terms_all: list[_TermSpec],
    random_terms_all: list[_TermSpec],
    logger: typing.Any,
) -> tuple[list[_TermSpec], list[_TermSpec]]:
    # Keep the complete fixed design.  The former implementation filtered out
    # covariates that varied within line, which changed the BLUE estimand and
    # silently discarded the user's fixed effect.  The compiled path is the
    # normal route, but this compatibility helper must preserve the same
    # invariant for callers that still use the legacy term objects.
    return list(random_terms_all), list(fixed_terms_all)


def _fit_stage1_blue(
    y_obs: np.ndarray,
    sub: pd.DataFrame,
    *,
    line_col: str,
    trait: str,
    env_cols: list[str] | None = None,
    stage1_random_terms: list[_TermSpec] | None = None,
    gxe_var: float | None = None,
    resid_var: float | None = None,
    maxiter: int = 100,
    logger: typing.Any = None,
    compiled: _CompiledModelTerms | None = None,
) -> _Stage1BlueResult:
    if compiled is not None:
        return _fit_stage1_blue_compiled(
            y_obs=y_obs,
            sub=sub,
            line_col=line_col,
            compiled=compiled,
            maxiter=maxiter,
        )

    env_cols = list(env_cols or [])
    stage1_random_terms = list(stage1_random_terms or [])
    if logger is None:
        raise ValueError("A logger is required for the legacy BLUE path.")

    if len(stage1_random_terms) == 0:
        try:
            return _fit_stage1_blue_weighted_ls(
                y_obs=y_obs,
                sub=sub,
                line_col=line_col,
                env_cols=env_cols,
                gxe_var=gxe_var,
                resid_var=resid_var,
            )
        except Exception as ex:
            logger.warning(
                f"Trait {trait}: fast weighted BLUE fallback to BLUP path because {type(ex).__name__}: {ex}"
            )

    line_ids_sub = sub[line_col].astype("string").fillna("NA").astype(str)
    env_key = _combine_key(sub, env_cols, "__ENV__").astype(str)

    x_blocks_sparse: list[sparse.spmatrix] = []
    x_names: list[str] = []

    env_prefix = "ENV"
    env_levels = sorted(pd.unique(env_key.astype(str)).tolist())
    if len(env_cols) > 0:
        env_arr, env_names = _onehot_encode_series(
            env_key,
            prefix=env_prefix,
            drop_first=True,
            sparse_output=True,
        )
        if int(env_arr.shape[1]) > 0:
            x_blocks_sparse.append(env_arr.tocsr())
            x_names.extend(env_names)

    line_prefix = str(line_col)
    line_arr, line_names = _onehot_encode_series(
        line_ids_sub,
        prefix=line_prefix,
        drop_first=True,
        sparse_output=True,
    )
    if int(line_arr.shape[1]) > 0:
        x_blocks_sparse.append(line_arr.tocsr())
        x_names.extend(line_names)

    x_stage1 = (
        sparse.hstack(x_blocks_sparse, format="csr", dtype=float)
        if len(x_blocks_sparse) > 0
        else None
    )

    z_stage1: list[typing.Union[np.ndarray, sparse.spmatrix]] = []
    z_names: list[str] = []
    gxe_name = f"{line_col}xENV"
    if len(env_cols) > 0 and len(env_levels) > 1:
        gxe_key = (line_ids_sub + "@@" + env_key).astype("string")
        gxe_dummies, _ = _onehot_encode_series(
            gxe_key,
            prefix=gxe_name,
            drop_first=False,
            sparse_output=True,
        )
        if int(gxe_dummies.shape[1]) > 0:
            z_stage1.append(gxe_dummies)
            z_names.append(gxe_name)

    extra_random, extra_random_names = _encode_random_design(
        sub,
        stage1_random_terms,
        trait=trait,
        logger=logger,
    )
    z_stage1.extend(extra_random)
    z_names.extend(extra_random_names)

    model = BLUP(
        y=np.asarray(y_obs, dtype=float).reshape(-1, 1),
        X=x_stage1,
        Z=z_stage1 if len(z_stage1) > 0 else None,
        maxiter=max(1, int(maxiter)),
        progress=False,
    )
    beta = np.asarray(model.beta, dtype=float).reshape(-1)
    intercept = float(beta[0])
    beta_map = {x_names[i]: float(beta[i + 1]) for i in range(len(x_names))}

    env_mean = 0.0
    if len(env_levels) > 0:
        env_effects = [0.0]
        for level in env_levels[1:]:
            env_effects.append(float(beta_map.get(f"{env_prefix}-{level}", 0.0)))
        env_mean = float(np.mean(np.asarray(env_effects, dtype=float)))

    line_levels = sorted(pd.unique(line_ids_sub.astype(str)).tolist())
    line_effect_map: dict[str, float] = {}
    if len(line_levels) > 0:
        line_effect_map[line_levels[0]] = 0.0
        for level in line_levels[1:]:
            line_effect_map[level] = float(beta_map.get(f"{line_prefix}-{level}", 0.0))

    blue_vals = np.asarray(
        [intercept + env_mean + float(line_effect_map.get(sid, 0.0)) for sid in line_levels],
        dtype=float,
    )
    return _Stage1BlueResult(sample_ids=line_levels, values=blue_vals)


def _fit_stage1_blue_weighted_ls(
    y_obs: np.ndarray,
    sub: pd.DataFrame,
    *,
    line_col: str,
    env_cols: list[str],
    gxe_var: float | None,
    resid_var: float | None,
) -> _Stage1BlueResult:
    y_vec = np.asarray(y_obs, dtype=float).reshape(-1)
    line_ids_sub = sub[line_col].astype("string").fillna("NA").astype(str)
    env_key = _combine_key(sub, env_cols, "__ENV__").astype(str)

    work = pd.DataFrame(
        {
            "line": line_ids_sub.to_numpy(dtype=object),
            "env": env_key.to_numpy(dtype=object),
            "y": y_vec,
        }
    )
    cell = (
        work.groupby(["line", "env"], sort=False, observed=False)
        .agg(n=("y", "size"), y_mean=("y", "mean"))
        .reset_index()
    )
    if cell.empty:
        raise ValueError("No usable line x env cells for weighted BLUE.")

    line_levels = pd.unique(line_ids_sub.astype(str)).tolist()
    env_levels = pd.unique(env_key.astype(str)).tolist()
    line_index = {sid: i for i, sid in enumerate(line_levels)}
    env_index = {sid: i for i, sid in enumerate(env_levels)}

    line_codes = np.asarray([line_index[str(x)] for x in cell["line"].tolist()], dtype=np.int64)
    env_codes = np.asarray([env_index[str(x)] for x in cell["env"].tolist()], dtype=np.int64)
    n_cell = cell["n"].to_numpy(dtype=float)
    y_mean = cell["y_mean"].to_numpy(dtype=float)

    gxe = 0.0 if gxe_var is None or (not np.isfinite(gxe_var)) or gxe_var < 0.0 else float(gxe_var)
    resid = 1.0 if resid_var is None or (not np.isfinite(resid_var)) or resid_var <= 0.0 else float(resid_var)
    cell_var = gxe + (resid / np.maximum(n_cell, 1.0))
    weights = 1.0 / np.maximum(cell_var, 1e-12)
    sqrt_w = np.sqrt(weights)

    n_rows = int(cell.shape[0])
    n_env = int(len(env_levels))
    n_line = int(len(line_levels))
    p = 1 + max(0, n_env - 1) + max(0, n_line - 1)

    row_parts: list[np.ndarray] = [np.arange(n_rows, dtype=np.int64)]
    col_parts: list[np.ndarray] = [np.zeros(n_rows, dtype=np.int64)]
    data_parts: list[np.ndarray] = [np.ones(n_rows, dtype=float)]

    if n_env > 1:
        keep_env = env_codes > 0
        if np.any(keep_env):
            row_parts.append(np.nonzero(keep_env)[0].astype(np.int64, copy=False))
            col_parts.append(env_codes[keep_env].astype(np.int64, copy=False))
            data_parts.append(np.ones(int(np.sum(keep_env)), dtype=float))

    if n_line > 1:
        keep_line = line_codes > 0
        if np.any(keep_line):
            row_parts.append(np.nonzero(keep_line)[0].astype(np.int64, copy=False))
            col_parts.append(
                (1 + max(0, n_env - 1) + (line_codes[keep_line] - 1)).astype(np.int64, copy=False)
            )
            data_parts.append(np.ones(int(np.sum(keep_line)), dtype=float))

    rows = np.concatenate(row_parts, axis=0)
    cols = np.concatenate(col_parts, axis=0)
    data = np.concatenate(data_parts, axis=0)
    x = sparse.csr_matrix((data, (rows, cols)), shape=(n_rows, p), dtype=float)

    xw = x.multiply(sqrt_w.reshape(-1, 1))
    yw = y_mean * sqrt_w
    xtwx = (xw.T @ xw).tocsc()
    xtwx = xtwx + sparse.eye(p, format="csc", dtype=float) * 1e-10
    xtwy = np.asarray(xw.T @ yw.reshape(-1, 1), dtype=float).reshape(-1)
    beta = np.asarray(spsolve(xtwx, xtwy), dtype=float).reshape(-1)

    intercept = float(beta[0])
    env_effects = np.zeros(n_env, dtype=float)
    if n_env > 1:
        env_effects[1:] = beta[1 : 1 + (n_env - 1)]
    env_mean = float(np.mean(env_effects)) if n_env > 0 else 0.0

    line_effects = np.zeros(n_line, dtype=float)
    line_start = 1 + max(0, n_env - 1)
    if n_line > 1:
        line_effects[1:] = beta[line_start : line_start + (n_line - 1)]

    blue_vals = intercept + env_mean + line_effects
    return _Stage1BlueResult(
        sample_ids=[str(x) for x in line_levels],
        values=np.asarray(blue_vals, dtype=float),
    )


def _fit_stage1_blue_compiled(
    y_obs: np.ndarray,
    sub: pd.DataFrame,
    *,
    line_col: str,
    compiled: _CompiledModelTerms,
    maxiter: int,
) -> _Stage1BlueResult:
    """Fit BLUEs with the exact compiled fixed/random nuisance design."""

    line_ids = sub[line_col].astype("string").fillna("NA").astype(str)
    line_fixed, _line_fixed_names = _onehot_encode_series(
        line_ids,
        prefix=line_col,
        drop_first=True,
        sparse_output=False,
    )
    line_fixed = np.asarray(line_fixed, dtype=float)

    x_blocks: list[np.ndarray] = []
    if compiled.fixed_matrix is not None:
        x_blocks.append(np.asarray(compiled.fixed_matrix, dtype=float))
    if int(line_fixed.shape[1]) > 0:
        x_blocks.append(line_fixed)
    x_stage2 = np.concatenate(x_blocks, axis=1) if x_blocks else None

    model = BLUP(
        y=np.asarray(y_obs, dtype=float).reshape(-1, 1),
        X=x_stage2,
        Z=compiled.random_matrices if compiled.random_matrices else None,
        maxiter=max(1, int(maxiter)),
        progress=False,
    )
    beta = np.asarray(model.beta, dtype=float).reshape(-1)
    fixed_count = (
        int(compiled.fixed_matrix.shape[1])
        if compiled.fixed_matrix is not None
        else 0
    )
    fixed_beta = beta[1 : 1 + fixed_count]
    line_beta = beta[1 + fixed_count :]

    fixed_fitted_mean = 0.0
    if fixed_count > 0:
        fixed_fitted_mean = float(
            np.mean(np.asarray(compiled.fixed_matrix, dtype=float) @ fixed_beta)
        )

    line_levels = sorted(pd.unique(line_ids).tolist())
    line_effects = np.zeros(len(line_levels), dtype=float)
    if line_effects.size > 1:
        line_effects[1:] = line_beta[: line_effects.size - 1]
    blue_values = float(beta[0]) + fixed_fitted_mean + line_effects

    noise_diag: np.ndarray | None = None
    cov_beta = getattr(model, "_cov_beta", None)
    if cov_beta is not None:
        cov_beta_arr = np.asarray(cov_beta, dtype=float)
        if cov_beta_arr.shape == (beta.size, beta.size):
            # A line BLUE is ``intercept + mean(fixed fit)`` for the baseline
            # level and adds one treatment-coded line coefficient for every
            # remaining level.  Compute the covariance diagonal from this
            # structure without allocating an n_line x n_line transform.
            base_transform = np.zeros(beta.size, dtype=float)
            base_transform[0] = 1.0
            if fixed_count > 0:
                base_transform[1 : 1 + fixed_count] = np.mean(
                    np.asarray(compiled.fixed_matrix, dtype=float),
                    axis=0,
                )
            base_variance = float(base_transform @ cov_beta_arr @ base_transform)
            noise_diag = np.full(len(line_levels), base_variance, dtype=float)
            if line_effects.size > 1:
                line_start = 1 + fixed_count
                line_cov = cov_beta_arr[line_start:, :]
                base_cross = np.asarray(line_cov @ base_transform, dtype=float).reshape(-1)
                line_diag = np.diag(cov_beta_arr[line_start:, line_start:])
                noise_diag[1:] = (
                    base_variance
                    + 2.0 * base_cross[: line_effects.size - 1]
                    + line_diag[: line_effects.size - 1]
                )
            noise_diag = np.asarray(noise_diag, dtype=float)
            noise_diag = np.where(
                np.isfinite(noise_diag) & (noise_diag >= 0.0),
                noise_diag,
                np.nan,
            )
    return _Stage1BlueResult(
        sample_ids=[str(value) for value in line_levels],
        values=np.asarray(blue_values, dtype=float),
        noise_diag=noise_diag,
    )


def _random_term_fitted_values(
    model: BLUP,
    z_term: typing.Union[np.ndarray, sparse.spmatrix],
    z_idx: int,
) -> np.ndarray:
    if getattr(model, "u_by_Z", None) is None or z_idx < 0 or z_idx >= len(model.u_by_Z):
        return np.zeros(int(z_term.shape[0]), dtype=float)

    coef = np.asarray(model.u_by_Z[z_idx], dtype=float).reshape(-1)
    q, mean_vec, std_vec = model.onehot_info[z_idx]
    scale = np.sqrt(float(q)) if float(q) > 0.0 else 1.0
    mean_arr = np.asarray(mean_vec, dtype=float)
    std_arr = np.asarray(std_vec, dtype=float)

    if bool(getattr(model, "_z_standardized", False)):
        if mean_arr.ndim == 0 and std_arr.ndim == 0:
            std_scalar = float(std_arr) if abs(float(std_arr)) > 0.0 else 1.0
            scaled_coef = coef / (std_scalar * scale)
            if sparse.issparse(z_term):
                fitted = np.asarray(z_term @ scaled_coef, dtype=float).reshape(-1)
            else:
                fitted = np.asarray(np.asarray(z_term, dtype=float) @ scaled_coef, dtype=float).reshape(-1)
            mean_scalar = float(mean_arr)
            if abs(mean_scalar) > 0.0:
                fitted = fitted - float(mean_scalar * np.sum(scaled_coef))
            return fitted

        mean_vec_arr = mean_arr.reshape(-1)
        std_vec_arr = std_arr.reshape(-1)
        std_vec_arr = np.where(np.abs(std_vec_arr) > 0.0, std_vec_arr, 1.0)
        scaled_coef = coef / (std_vec_arr * scale)
        if sparse.issparse(z_term):
            fitted = np.asarray(z_term @ scaled_coef, dtype=float).reshape(-1)
        else:
            fitted = np.asarray(np.asarray(z_term, dtype=float) @ scaled_coef, dtype=float).reshape(-1)
        return fitted - float(mean_vec_arr @ scaled_coef)

    if sparse.issparse(z_term):
        return np.asarray(z_term @ coef, dtype=float).reshape(-1)
    return np.asarray(np.asarray(z_term, dtype=float) @ coef, dtype=float).reshape(-1)


def _line_level_blup_from_broad_model(
    model: BLUP,
    sub: pd.DataFrame,
    *,
    line_col: str,
    line_z: typing.Union[np.ndarray, sparse.spmatrix],
    line_term_idx: int,
) -> dict[str, float]:
    fixed_fitted = np.asarray(model.X @ model.beta, dtype=float).reshape(-1)
    line_fitted = _random_term_fitted_values(model, line_z, line_term_idx)
    obs_pred = fixed_fitted + line_fitted
    line_ids = sub[line_col].astype("string").fillna("NA").astype(str)
    agg = (
        pd.DataFrame({"line": line_ids.to_numpy(dtype=object), "pred": obs_pred})
        .groupby("line", sort=False, observed=False)["pred"]
        .mean()
    )
    return {str(k): float(v) for k, v in agg.items()}


def _line_level_noise_diag(
    sub: pd.DataFrame,
    *,
    line_col: str,
    env_cols: list[str],
    line_ids: list[str],
    vge: float,
    ve: float,
) -> np.ndarray:
    sid = sub[line_col].astype("string").fillna("NA").astype(str)
    env_key = _combine_key(sub, env_cols, "__ENV__").astype(str)

    env_per_line = (
        pd.DataFrame({"line": sid, "env": env_key})
        .drop_duplicates()
        .groupby("line", sort=False)["env"]
        .nunique()
    )
    plot_per_line = sid.groupby(sid, sort=False).size()

    env_counts = np.asarray(
        [float(env_per_line.get(str(sid_i), 1.0)) for sid_i in line_ids],
        dtype=float,
    )
    plot_counts = np.asarray(
        [float(plot_per_line.get(str(sid_i), 1.0)) for sid_i in line_ids],
        dtype=float,
    )
    vge_use = 0.0 if (not np.isfinite(vge)) or vge < 0.0 else float(vge)
    ve_use = 0.0 if (not np.isfinite(ve)) or ve < 0.0 else float(ve)
    return (vge_use / np.maximum(env_counts, 1.0)) + (ve_use / np.maximum(plot_counts, 1.0))


# Joint additive + line-nonadditive REML is the active dense narrow path.
# Sparse REML uses the same stage-1 uncertainty on the reported phenotype scale.
def _prepare_joint_kernel_inputs(
    y_line: np.ndarray,
    *,
    kinship: np.ndarray,
    noise_diag: np.ndarray,
    x_fixed: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    y = np.asarray(y_line, dtype=float).reshape(-1, 1)
    n = int(y.shape[0])
    if n <= 1:
        raise ValueError("Need at least 2 lines for joint kernel fit.")

    k = np.asarray(kinship, dtype=float)
    if k.shape != (n, n):
        raise ValueError(f"kinship shape mismatch: got {k.shape}, expected {(n, n)}")
    k = (k + k.T) / 2.0
    k_diag_mean = float(np.mean(np.diag(k)))
    if (not np.isfinite(k_diag_mean)) or k_diag_mean <= 0.0:
        raise ValueError(f"Invalid kinship mean diagonal: {k_diag_mean}")
    k = k / k_diag_mean

    d = np.asarray(noise_diag, dtype=float).reshape(-1)
    if d.shape[0] != n:
        raise ValueError(f"noise_diag length mismatch: got {d.shape[0]}, expected {n}")
    d = np.where(np.isfinite(d) & (d >= 0.0), d, 0.0)
    d_mean = float(np.mean(d)) if d.size > 0 else 0.0

    if x_fixed is None:
        x = np.ones((n, 1), dtype=float)
    else:
        xf = np.asarray(x_fixed, dtype=float)
        if xf.shape[0] != n:
            raise ValueError(f"x_fixed row mismatch: got {xf.shape[0]}, expected {n}")
        x = np.concatenate([np.ones((n, 1), dtype=float), xf], axis=1)

    return y, x, k, d, d_mean


def _joint_kernel_state(
    *,
    y: np.ndarray,
    x: np.ndarray,
    kinship: np.ndarray,
    noise_diag: np.ndarray,
    noise_mean: float,
    va: float,
    vline: float,
) -> tuple[float, _JointKernelResult]:
    va_use = float(max(float(va), _JOINT_VAR_FLOOR))
    vline_use = float(max(float(vline), _JOINT_VAR_FLOOR))
    n = int(y.shape[0])

    v = va_use * kinship
    v.flat[:: n + 1] += noise_diag + vline_use
    v = (v + v.T) / 2.0

    l = np.linalg.cholesky(v)
    vinvx = cho_solve((l, True), x)
    vinvy = cho_solve((l, True), y)
    xt_vinv_x = (x.T @ vinvx + (x.T @ vinvx).T) / 2.0
    lx = np.linalg.cholesky(xt_vinv_x)
    beta = np.linalg.solve(xt_vinv_x, x.T @ vinvy)
    r = y - x @ beta
    vinvr = cho_solve((l, True), r)

    quad = float((r.T @ vinvr)[0, 0])
    if (not np.isfinite(quad)) or quad <= 0.0:
        raise ValueError(f"Invalid joint REML quadratic form: {quad}")
    log_det_v = float(2.0 * np.sum(np.log(np.diag(l))))
    log_det_xt = float(2.0 * np.sum(np.log(np.diag(lx))))
    nll = 0.5 * (log_det_v + log_det_xt + quad)
    if not np.isfinite(nll):
        raise ValueError("Joint REML objective became non-finite.")

    add_blup = va_use * (kinship @ vinvr)
    line_blup = vline_use * vinvr
    denom = va_use + vline_use + noise_mean
    h2_raw = float(va_use / denom) if denom > 0.0 else np.nan
    return nll, _JointKernelResult(
        va=va_use,
        vline=vline_use,
        h2_raw=h2_raw,
        beta=np.asarray(beta, dtype=float).reshape(-1),
        add_blup=np.asarray(add_blup, dtype=float).reshape(-1),
        line_blup=np.asarray(line_blup, dtype=float).reshape(-1),
        noise_mean=float(noise_mean),
    )


def _fit_joint_line_kernel_approx(
    y_line: np.ndarray,
    *,
    kinship: np.ndarray,
    noise_diag: np.ndarray,
    x_fixed: np.ndarray | None,
) -> _JointKernelResult:
    y, x, k, d, d_mean = _prepare_joint_kernel_inputs(
        y_line,
        kinship=kinship,
        noise_diag=noise_diag,
        x_fixed=x_fixed,
    )
    n = int(y.shape[0])

    beta_ols, *_ = np.linalg.lstsq(x, y, rcond=None)
    r_ols = (y - x @ beta_ols).reshape(-1)

    kk = float(np.sum(k * k))
    ki = float(np.trace(k))
    ii = float(n)
    ks = float(r_ols @ (k @ r_ols) - np.sum(np.diag(k) * d))
    is_ = float(np.dot(r_ols, r_ols) - np.sum(d))
    a = np.array([[kk, ki], [ki, ii]], dtype=float)
    b = np.array([ks, is_], dtype=float)

    def _loss(va: float, vline: float) -> float:
        return (
            (va * va * kk)
            + (2.0 * va * vline * ki)
            + (vline * vline * ii)
            - (2.0 * va * ks)
            - (2.0 * vline * is_)
        )

    cand: list[tuple[float, float]] = []
    try:
        sol = np.linalg.solve(a + np.eye(2, dtype=float) * 1e-12, b)
        cand.append((max(float(sol[0]), 0.0), max(float(sol[1]), 0.0)))
    except Exception:
        pass
    cand.extend(
        [
            (max(ks / max(kk, 1e-12), 0.0), 0.0),
            (0.0, max(is_ / max(ii, 1e-12), 0.0)),
            (0.0, 0.0),
        ]
    )
    va, vline = min(cand, key=lambda vv: _loss(vv[0], vv[1]))
    _nll, state = _joint_kernel_state(
        y=y,
        x=x,
        kinship=k,
        noise_diag=d,
        noise_mean=d_mean,
        va=max(float(va), _JOINT_VAR_FLOOR),
        vline=max(float(vline), _JOINT_VAR_FLOOR),
    )
    return state


def _fit_joint_line_kernel_exact(
    y_line: np.ndarray,
    *,
    kinship: np.ndarray,
    noise_diag: np.ndarray,
    x_fixed: np.ndarray | None,
    maxiter: int,
) -> _JointKernelResult:
    y, x, k, d, d_mean = _prepare_joint_kernel_inputs(
        y_line,
        kinship=kinship,
        noise_diag=noise_diag,
        x_fixed=x_fixed,
    )

    y_center = y - np.mean(y, axis=0, keepdims=True)
    y_var = float(np.var(y_center.reshape(-1), ddof=1)) if int(y.shape[0]) > 1 else 1.0
    if (not np.isfinite(y_var)) or y_var <= 0.0:
        y_var = 1.0

    starts: list[tuple[float, float]] = []
    try:
        approx = _fit_joint_line_kernel_approx(
            y.reshape(-1),
            kinship=k,
            noise_diag=d,
            x_fixed=x_fixed,
        )
        starts.append((float(approx.va), float(approx.vline)))
    except Exception:
        pass
    starts.extend(
        [
            (max(y_var * 0.50, _JOINT_VAR_FLOOR), max(y_var * 0.50, _JOINT_VAR_FLOOR)),
            (max(y_var, _JOINT_VAR_FLOOR), _JOINT_VAR_FLOOR),
            (_JOINT_VAR_FLOOR, max(y_var, _JOINT_VAR_FLOOR)),
            (max(d_mean, _JOINT_VAR_FLOOR), max(d_mean, _JOINT_VAR_FLOOR)),
        ]
    )

    start_unique: list[tuple[float, float]] = []
    seen_start: set[tuple[int, int]] = set()
    for va0, vline0 in starts:
        key = (
            int(np.round(np.log10(max(float(va0), _JOINT_VAR_FLOOR)) * 1000.0)),
            int(np.round(np.log10(max(float(vline0), _JOINT_VAR_FLOOR)) * 1000.0)),
        )
        if key in seen_start:
            continue
        seen_start.add(key)
        start_unique.append(
            (
                max(float(va0), _JOINT_VAR_FLOOR),
                max(float(vline0), _JOINT_VAR_FLOOR),
            )
        )

    best_fun = np.inf
    best_state: _JointKernelResult | None = None
    best_eta: np.ndarray | None = None

    def _eta_to_var(eta: np.ndarray) -> tuple[float, float]:
        eta = np.asarray(eta, dtype=float).reshape(-1)
        eta = np.clip(eta, _JOINT_LOG_FLOOR, _JOINT_LOG_CEIL)
        return float(np.exp(eta[0])), float(np.exp(eta[1]))

    def _objective(eta: np.ndarray) -> float:
        try:
            va_now, vline_now = _eta_to_var(eta)
            nll, _state = _joint_kernel_state(
                y=y,
                x=x,
                kinship=k,
                noise_diag=d,
                noise_mean=d_mean,
                va=va_now,
                vline=vline_now,
            )
            return float(nll)
        except Exception:
            return _JOINT_OBJ_PENALTY

    for va0, vline0 in start_unique:
        eta0 = np.log(np.asarray([va0, vline0], dtype=float))
        eta0 = np.clip(eta0, _JOINT_LOG_FLOOR, _JOINT_LOG_CEIL)
        res = minimize(
            _objective,
            eta0,
            method="L-BFGS-B",
            bounds=[(_JOINT_LOG_FLOOR, _JOINT_LOG_CEIL), (_JOINT_LOG_FLOOR, _JOINT_LOG_CEIL)],
            options={"maxiter": max(25, int(maxiter))},
        )
        candidates = [np.asarray(res.x, dtype=float).reshape(-1), eta0]
        for eta_try in candidates:
            obj = _objective(eta_try)
            if (not np.isfinite(obj)) or obj >= best_fun:
                continue
            try:
                va_now, vline_now = _eta_to_var(eta_try)
                _nll, state = _joint_kernel_state(
                    y=y,
                    x=x,
                    kinship=k,
                    noise_diag=d,
                    noise_mean=d_mean,
                    va=va_now,
                    vline=vline_now,
                )
            except Exception:
                continue
            best_fun = float(obj)
            best_state = state
            best_eta = np.asarray(eta_try, dtype=float).reshape(-1)

    if best_state is None or best_eta is None or (not np.isfinite(best_fun)):
        raise RuntimeError("Exact joint REML failed to converge to a finite solution.")
    return best_state


def _fit_dense_narrow_corrected(
    y_line: np.ndarray,
    *,
    kinship: np.ndarray,
    noise_diag: np.ndarray,
    x_fixed: np.ndarray | None,
    maxiter: int,
) -> _JointKernelResult:
    """Run dense line-level REML with first-stage BLUE uncertainty included."""

    return _fit_joint_line_kernel_exact(
        y_line,
        kinship=kinship,
        noise_diag=noise_diag,
        x_fixed=x_fixed,
        maxiter=maxiter,
    )


def _fit_sparse_narrow_corrected(
    *,
    jxgrm_path: str,
    sample_idx: np.ndarray | None,
    y_vec: np.ndarray,
    x_cov: np.ndarray | None,
    noise_diag: np.ndarray,
    objective_mode: str,
    threads: int,
) -> dict[str, object]:
    """Run sparse REML and add the first-stage BLUE uncertainty to PVE."""

    result = dict(
        _splmm_sparse_null_fit(
            jxgrm_path=jxgrm_path,
            sample_idx=sample_idx,
            y_vec=y_vec,
            x_cov=x_cov,
            progress_callback=None,
            objective_mode=objective_mode,
            threads=threads,
        )
    )
    d = np.asarray(noise_diag, dtype=float).reshape(-1)
    if d.size == 0 or not np.all(np.isfinite(d) & (d >= 0.0)):
        raise ValueError("Sparse narrow REML requires a finite non-negative stage1 noise diagonal.")
    noise_mean = float(np.mean(d))
    sigma_g2 = float(result.get("sigma_g2", float("nan")))
    sigma_e2 = float(result.get("sigma_e2", float("nan")))
    mean_diag_k = float(
        result.get("mean_diag_k", result.get("grm_mean_diag", float("nan")))
    )
    genetic_diag = sigma_g2 * max(mean_diag_k, 0.0)
    denominator = genetic_diag + sigma_e2 + noise_mean
    backend_pve = float(result.get("pve", float("nan")))
    if np.isfinite(denominator) and denominator > 0.0:
        result["pve"] = float(genetic_diag / denominator)
        result["pve_pheno_scale"] = float(genetic_diag / denominator)
    result["stage1_noise_mean"] = noise_mean
    result["pve_backend_uncorrected"] = backend_pve
    return result


def _sparse_additive_blup_from_subset(
    *,
    jxgrm_path: str,
    sample_idx: np.ndarray,
    y_vec: np.ndarray,
    noise_diag: np.ndarray,
    sigma_g2: float,
    sigma_e2: float,
    x_cov: np.ndarray | None = None,
) -> np.ndarray:
    """Compute additive BLUP from the sparse GRM subset used by sparse REML."""

    k = np.asarray(
        jxrs.splmm_load_sparse_grm_subset_dense(
            str(jxgrm_path),
            sample_indices=np.ascontiguousarray(sample_idx, dtype=np.int64),
        ),
        dtype=float,
    )
    y = np.asarray(y_vec, dtype=float).reshape(-1, 1)
    d = np.asarray(noise_diag, dtype=float).reshape(-1)
    n = int(y.shape[0])
    if k.shape != (n, n) or d.shape[0] != n:
        raise ValueError(
            f"Sparse GBLUP shape mismatch: K={k.shape}, y={y.shape}, noise={d.shape}."
        )
    v = float(sigma_g2) * ((k + k.T) * 0.5)
    v.flat[:: n + 1] += float(sigma_e2) + d
    l = np.linalg.cholesky((v + v.T) * 0.5)
    if x_cov is None:
        x = np.ones((n, 1), dtype=float)
    else:
        x_arg = np.asarray(x_cov, dtype=float)
        x = np.concatenate([np.ones((n, 1), dtype=float), x_arg], axis=1)
    vinvx = cho_solve((l, True), x)
    vinvy = cho_solve((l, True), y)
    xt_vinv_x = (x.T @ vinvx + (x.T @ vinvx).T) * 0.5
    beta = np.linalg.solve(xt_vinv_x, x.T @ vinvy)
    residual = y - x @ beta
    vinvr = cho_solve((l, True), residual)
    return np.asarray(float(sigma_g2) * (k @ vinvr), dtype=float).reshape(-1)


def _resolve_cli_columns(
    values: Iterable[str] | None,
    candidates: list[str],
    label: str,
) -> list[str]:
    return _resolve_columns(_split_tokens(values), candidates, label, index_base=0)


def _infer_column_type_details(series: pd.Series) -> dict[str, typing.Any]:
    """Infer a source type and retain the counts/reason for configuration logs."""

    non_missing = series.dropna()
    valid_count = int(non_missing.shape[0])
    numeric = pd.to_numeric(non_missing, errors="coerce")
    finite_mask = numeric.notna() & np.isfinite(numeric)
    numeric_count = int(finite_mask.sum())
    if valid_count == 0:
        return {
            "type": "categorical",
            "valid_count": 0,
            "unique_count": 0,
            "reason": "empty_or_all_missing",
        }
    if numeric_count != valid_count:
        return {
            "type": "categorical",
            "valid_count": valid_count,
            "unique_count": int(non_missing.astype("string").nunique(dropna=False)),
            "reason": "non_numeric_or_non_finite_values",
        }

    values = np.asarray(numeric, dtype=float).reshape(-1)
    integer_valued = bool(np.all(values == np.floor(values)))
    unique_count = int(pd.Series(values).nunique(dropna=True))
    categorical_limit = max(1, int(np.floor(valid_count * 0.05)))
    if integer_valued and unique_count <= 10 and unique_count <= categorical_limit:
        return {
            "type": "categorical",
            "valid_count": valid_count,
            "unique_count": unique_count,
            "reason": f"low_cardinality_integer(<=10_and_<=5%;limit={categorical_limit})",
        }
    return {
        "type": "continuous",
        "valid_count": valid_count,
        "unique_count": unique_count,
        "reason": "finite_numeric_not_low_cardinality_integer",
    }


def _infer_column_type(series: pd.Series) -> str:
    """Infer categorical/continuous using the approved low-cardinality rule."""

    return str(_infer_column_type_details(series)["type"])


def _split_effect_values(values: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    for raw in list(values or []):
        for token in str(raw).split(","):
            text = token.strip()
            if text:
                out.append(text)
    return out


def _parse_effect_specs(
    values: Iterable[str] | None,
    kind: str,
    candidates: list[str],
    df: pd.DataFrame,
) -> list[_EffectSpec]:
    """Parse effect tokens and validate their source-column types."""

    kind_text = str(kind).strip().lower()
    if kind_text not in {"fixed", "random", "gxe", "gxc"}:
        raise ValueError(f"Unknown effect kind: {kind}")

    specs: list[_EffectSpec] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for token in _split_effect_values(values):
        if token.count(":") > 1:
            raise ValueError(
                f"Invalid {kind_text} interaction {token}: expected A:B."
            )
        if ":" in token:
            left, right = [part.strip() for part in token.split(":", 1)]
            if left == "" or right == "":
                raise ValueError(
                    f"Invalid {kind_text} interaction {token}: both columns are required."
                )
            source_cols = _resolve_cli_columns(
                [left, right], candidates, f"-{kind_text}"
            )
            if len(source_cols) != 2:
                raise ValueError(
                    f"Invalid {kind_text} interaction {token}: expected two columns."
                )
            interaction = ":"
        else:
            source_cols = _resolve_cli_columns(
                [token], candidates, f"-{kind_text}"
            )
            if len(source_cols) != 1:
                raise ValueError(
                    f"Invalid {kind_text} term {token}: expected one column."
                )
            interaction = None

        source_types = tuple(_infer_column_type(df[col]) for col in source_cols)
        if kind_text == "gxe":
            if len(source_cols) == 1 and source_types[0] != "categorical":
                raise ValueError(
                    f"-gxe term {token} requires a categorical environment column; "
                    f"{source_cols[0]} is {source_types[0]}."
                )
            if len(source_cols) == 2 and source_types != (
                "categorical",
                "categorical",
            ):
                raise ValueError(
                    f"-gxe interaction {token} must compile to categorical×categorical; "
                    f"got {source_types}."
                )
        if kind_text == "gxc":
            if len(source_cols) != 1 or source_types[0] != "continuous":
                name = source_cols[0] if source_cols else token
                actual = source_types[0] if source_types else "unknown"
                raise ValueError(
                    f"-gxc term {token} requires a continuous column; "
                    f"{name} is {actual}."
                )

        key = (kind_text, tuple(source_cols))
        if key in seen:
            raise ValueError(f"Duplicate {kind_text} term: {token}")
        seen.add(key)
        specs.append(
            _EffectSpec(
                kind=kind_text,
                sources=tuple(source_cols),
                source_types=source_types,
                label=":".join(source_cols),
                interaction=interaction,
            )
        )
    return specs


def _compile_effect_matrix(
    df: pd.DataFrame,
    spec: _EffectSpec,
    *,
    for_random: bool,
) -> tuple[np.ndarray, list[str]]:
    """Compile one effect specification into a numeric design block."""

    def finalize(
        block: typing.Union[np.ndarray, sparse.spmatrix],
        names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        arr = np.asarray(block.toarray() if sparse.issparse(block) else block, dtype=float)
        if arr.ndim != 2:
            arr = arr.reshape(int(df.shape[0]), -1)
        if arr.shape[1] != len(names):
            raise ValueError(
                f"Effect {spec.label} produced inconsistent design metadata: "
                f"matrix={arr.shape}, names={len(names)}."
            )
        if arr.shape[1] == 0:
            raise ValueError(f"Effect {spec.label} is not estimable after encoding (zero columns).")
        finite = np.isfinite(arr)
        if not bool(finite.all()):
            raise ValueError(f"Effect {spec.label} contains non-finite design values.")
        active = np.any(np.abs(arr) > 1e-12, axis=0)
        if not bool(active.any()):
            raise ValueError(f"Effect {spec.label} is constant/zero after encoding.")
        return arr[:, active], [name for name, keep in zip(names, active) if bool(keep)]

    source_types = spec.source_types
    if len(spec.sources) == 1:
        col = spec.sources[0]
        if source_types[0] == "continuous":
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if np.ptp(values) <= 1e-12:
                raise ValueError(f"Effect {spec.label} is constant and cannot be estimated.")
            return finalize(values.reshape(-1, 1), [spec.label])
        if int(df[col].astype("string").fillna("NA").nunique(dropna=False)) <= 1:
            raise ValueError(f"Effect {spec.label} has only one categorical level and cannot be estimated.")
        arr, names = _onehot_encode_series(
            df[col],
            prefix=spec.label,
            drop_first=not for_random,
            sparse_output=False,
        )
        return finalize(arr, names)

    left, right = spec.sources
    if source_types == ("categorical", "categorical"):
        combo = _combine_key(df, [left, right], spec.label)
        arr, names = _onehot_encode_series(
            combo,
            prefix=spec.label,
            drop_first=not for_random,
            sparse_output=False,
        )
        if int(combo.nunique(dropna=False)) <= 1:
            raise ValueError(f"Effect {spec.label} has only one combined level and cannot be estimated.")
        return finalize(arr, names)

    if source_types == ("continuous", "continuous"):
        left_values = pd.to_numeric(df[left], errors="coerce").to_numpy(dtype=float)
        right_values = pd.to_numeric(df[right], errors="coerce").to_numpy(dtype=float)
        product = left_values * right_values
        if np.ptp(product) <= 1e-12:
            raise ValueError(f"Effect {spec.label} is constant and cannot be estimated.")
        return finalize(product.reshape(-1, 1), [spec.label])

    categorical_col = left if source_types[0] == "categorical" else right
    continuous_col = right if source_types[0] == "categorical" else left
    continuous_values = pd.to_numeric(df[continuous_col], errors="coerce").to_numpy(dtype=float)
    onehot, level_names = _onehot_encode_series(
        df[categorical_col],
        prefix=f"{categorical_col}:{continuous_col}",
        drop_first=False,
        sparse_output=False,
    )
    arr = np.asarray(onehot, dtype=float) * continuous_values.reshape(-1, 1)
    names = [f"{name}:slope" for name in level_names]
    return finalize(arr, names)


def _effect_factor_series(df: pd.DataFrame, spec: _EffectSpec) -> pd.Series:
    if spec.result_type != "categorical":
        raise ValueError(f"Effect {spec.label} does not compile to a categorical factor.")
    return _combine_key(df, list(spec.sources), spec.label).astype("string")


def _compile_line_factor_matrix(
    df: pd.DataFrame,
    *,
    line_col: str,
    factor: pd.Series,
    label: str,
) -> tuple[sparse.csr_matrix, list[str]]:
    line_ids = df[line_col].astype("string").fillna("NA").astype(str)
    factor_ids = factor.astype("string").fillna("NA").astype(str)
    if int(factor_ids.nunique(dropna=False)) <= 1:
        raise ValueError(
            f"GxE factor {label} has only one level after filtering and cannot be estimated."
        )
    key = (line_ids + "@@" + factor_ids).astype("string")
    matrix, _ = _onehot_encode_series(
        key,
        prefix=f"{line_col}×{label}",
        drop_first=False,
        sparse_output=True,
    )
    return matrix.tocsr(), [f"{line_col}×{label}"]


def _compile_line_slope_matrix(
    df: pd.DataFrame,
    *,
    line_col: str,
    column: str,
) -> tuple[sparse.csr_matrix, list[str]]:
    line_ids = df[line_col].astype("string").fillna("NA").astype(str)
    levels = sorted(pd.unique(line_ids).tolist())
    level_index = {str(level): idx for idx, level in enumerate(levels)}
    codes = np.asarray([level_index[str(value)] for value in line_ids], dtype=np.int64)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"GxC column {column} contains non-finite values after filtering.")
    centered = values - float(np.mean(values))
    if not np.any(np.abs(centered) > 0.0):
        raise ValueError(f"GxC column {column} is constant after centering.")
    rows = np.arange(int(df.shape[0]), dtype=np.int64)
    matrix = sparse.csr_matrix(
        (centered, (rows, codes)),
        shape=(int(df.shape[0]), len(levels)),
        dtype=float,
    )
    return matrix, [f"{line_col}×{column}"]


def _compile_model_terms(
    df: pd.DataFrame,
    *,
    line_col: str,
    fixed_specs: list[_EffectSpec],
    random_specs: list[_EffectSpec],
    gxe_specs: list[_EffectSpec],
    gxc_specs: list[_EffectSpec],
) -> _CompiledModelTerms:
    fixed_blocks: list[np.ndarray] = []
    fixed_names: list[str] = []
    fixed_labels: list[str] = []
    for spec in fixed_specs:
        block, names = _compile_effect_matrix(df, spec, for_random=False)
        if int(block.shape[1]) > 0:
            fixed_blocks.append(np.asarray(block, dtype=float))
            fixed_names.extend(names)
            fixed_labels.append(spec.label)

    random_matrices: list[typing.Union[np.ndarray, sparse.spmatrix]] = []
    random_names: list[str] = []
    for spec in random_specs:
        block, _names = _compile_effect_matrix(df, spec, for_random=True)
        if int(block.shape[1]) > 0:
            random_matrices.append(np.asarray(block, dtype=float))
            random_names.append(spec.label)

    line_ids = df[line_col].astype("string").fillna("NA").astype(str)
    line_z, line_names = _onehot_encode_series(
        line_ids,
        prefix=line_col,
        drop_first=False,
        sparse_output=True,
    )
    line_z = line_z.tocsr()

    for spec in gxe_specs:
        factor = _effect_factor_series(df, spec)
        block, names = _compile_line_factor_matrix(
            df,
            line_col=line_col,
            factor=factor,
            label=spec.label,
        )
        if int(block.shape[1]) > 0:
            random_matrices.append(block)
            random_names.extend(names)

    for spec in gxc_specs:
        if len(spec.sources) != 1:
            raise ValueError(f"GxC term {spec.label} must name one continuous column.")
        block, names = _compile_line_slope_matrix(
            df,
            line_col=line_col,
            column=spec.sources[0],
        )
        random_matrices.append(block)
        random_names.extend(names)

    fixed_matrix = (
        np.concatenate(fixed_blocks, axis=1) if len(fixed_blocks) > 0 else None
    )
    return _CompiledModelTerms(
        fixed_specs=list(fixed_specs),
        random_specs=list(random_specs),
        gxe_specs=list(gxe_specs),
        gxc_specs=list(gxc_specs),
        fixed_matrix=fixed_matrix,
        fixed_names=fixed_names,
        fixed_labels=fixed_labels,
        line_z=line_z,
        line_names=line_names,
        random_matrices=random_matrices,
        random_names=random_names,
    )


def _reml_dev_help_requested(argv: list[str] | None = None) -> bool:
    tokens = list(sys.argv[1:] if argv is None else argv)
    return "-dev" in tokens or "--dev" in tokens


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    show_dev_help = _reml_dev_help_requested(argv)
    parser = CliArgumentParser(
        prog="jx reml",
        formatter_class=cli_help_formatter(),
        allow_abbrev=False,
        epilog=minimal_help_epilog(
            [
                "jx reml -p pheno.tsv -n Yield -c year,loc -o outdir",
                "jx reml -p pheno.tsv -n Yield -c PCA1,PCA2 -rc block -k data.cGRM.npy",
                "jx reml -p pheno.tsv -n Yield -gxe loc -gxc temperature -spk data.cGRM.spgrm",
                "jx reml -h -dev",
            ]
        ),
    )

    req = parser.add_argument_group("Required Arguments")
    req.add_argument(
        "-p",
        "--pheno",
        required=True,
        type=str,
        dest="pheno",
        help="Input phenotype table (.tsv/.csv/whitespace); the first column is the sample/line ID.",
    )

    opt = parser.add_argument_group("Optional Arguments")
    opt.add_argument(
        "-n",
        "--ncol",
        action="extend",
        nargs="+",
        default=None,
        metavar="COL",
        dest="ncol",
        help=(
            "Phenotype column(s), selected by name or zero-based index excluding the first sample-ID column. "
            "Comma lists, repeated flags, and numeric ranges are accepted; default: all usable numeric columns."
        ),
    )

    opt.add_argument(
        "-c",
        "--cov",
        action="append",
        default=[],
        metavar="TERM",
        dest="cov",
        help=(
            "Fixed effect column(s) from the phenotype table (name or zero-based index excluding the ID column). Use A:B for an interaction; "
            "categorical×categorical combines levels, numeric×numeric multiplies, and mixed types create slopes."
        ),
    )
    opt.add_argument(
        "-rc",
        "--rcov",
        action="append",
        default=[],
        metavar="TERM",
        dest="rcov",
        help="Random nuisance effect column(s) from the phenotype table (name or zero-based index excluding ID); repeat or comma-separate terms.",
    )
    opt.add_argument(
        "-gxe",
        "--gxe",
        action="append",
        default=[],
        metavar="TERM",
        dest="gxe",
        help="Random Line×discrete-environment term(s) from the phenotype table (name or zero-based index excluding ID).",
    )
    opt.add_argument(
        "-gxc",
        "--gxc",
        action="append",
        default=[],
        metavar="COL",
        dest="gxc",
        help="Random Line×continuous-gradient slope column(s) from the phenotype table (name or zero-based index excluding ID).",
    )

    grm_group = opt.add_mutually_exclusive_group(required=False)
    grm_group.add_argument(
        "-k",
        "--grm",
        dest="grm",
        default=None,
        metavar="FILE",
        help=(
            "Optional dense GRM matrix. When provided, corrected narrow-sense h2 and GBLUP are estimated."
        ),
    )
    grm_group.add_argument(
        "-spk",
        "--grm-sparse",
        dest="grm_sparse",
        default=None,
        metavar="FILE",
        help=(
            "Optional Sparse GRM (.spgrm/.jxgrm or GCTA/fastGWA prefix/.grm.sp). "
            "When provided, corrected sparse narrow-sense h2 and GBLUP are estimated."
        ),
    )

    opt.add_argument(
        "-dev",
        "--dev",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    opt.add_argument(
        "--spk-mode",
        dest="grm_sparse_mode",
        choices=("raw", "fastgwa"),
        default="raw",
        help=(
            "Sparse REML objective for -spk/--grm-sparse. "
            "`raw` uses JanusX profile REML on the sparse K directly; "
            "`fastgwa` uses a fastGWA-compatible fixed-Vp sparse REML objective "
            "matched to GCTA fastGWA-REML behavior (default: %(default)s)."
        ) if show_dev_help else argparse.SUPPRESS,
    )
    add_common_out_arg(opt, default=".", help_profile="current_dir")
    add_common_prefix_arg(opt, default=None, help_profile="inferred_input_filename")
    add_common_thread_arg(
        opt,
        default_threads=detect_effective_threads(),
        help_profile="default",
    )
    opt.add_argument(
        "-maxiter",
        "--maxiter",
        type=int,
        default=100,
        help="Maximum REML iterations (default: %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    t0 = time.time()
    args = build_parser(argv).parse_args(argv)
    if int(args.thread) <= 0:
        raise ValueError("-t/--thread must be a positive integer.")
    args.thread = int(apply_outer_thread_cap(int(args.thread)))
    if args.grm is not None and args.grm_sparse is not None:
        raise ValueError("Please provide only one of -k/--grm or -spk/--grm-sparse.")
    if args.grm_sparse is not None:
        args.grm_sparse = _splmm_normalize_sparse_grm_path(str(args.grm_sparse))
    auto_prefix = strip_default_prefix_suffix(os.path.basename(str(args.pheno)))
    outdir, outprefix, prefix = apply_output_prefix_compat(
        args,
        auto_prefix,
        argv=argv,
        fallback_prefix="reml",
    )
    os.makedirs(outdir, mode=0o755, exist_ok=True)
    log_path = f"{outprefix}.reml.log"
    logger = setup_logging(log_path)

    if not ensure_file_exists(logger, str(args.pheno), "Input phenotype file"):
        return
    if args.grm is not None and not ensure_file_exists(logger, str(args.grm), "GRM file"):
        return
    if args.grm_sparse is not None and not ensure_file_exists(logger, str(args.grm_sparse), "Sparse GRM file"):
        return

    load_t0 = time.time()
    try:
        df = _read_table_with_optional_header(str(args.pheno))
    except Exception:
        raise
    load_elapsed = format_elapsed(time.time() - load_t0)

    if df.shape[1] < 2:
        raise ValueError("Input file must contain at least 2 columns.")

    df = df.copy()
    all_cols = [str(c) for c in df.columns]

    line_col = str(all_cols[0])
    effect_cols = [c for c in all_cols if c != line_col]
    env_cols: list[str] = []
    raw_line = df[line_col].copy()
    if raw_line.isna().any() or raw_line.astype("string").str.strip().eq("").any():
        raise ValueError("The first phenotype-table column must contain non-empty sample/line IDs.")
    df[line_col] = raw_line.astype("string").astype(str)
    fixed_specs = _parse_effect_specs(args.cov, "fixed", effect_cols, df)
    random_specs = _parse_effect_specs(args.rcov, "random", effect_cols, df)
    gxe_specs = _parse_effect_specs(args.gxe, "gxe", effect_cols, df)
    gxc_specs = _parse_effect_specs(args.gxc, "gxc", effect_cols, df)

    source_columns = _unique_preserve(
        source
        for specs in (fixed_specs, random_specs, gxe_specs, gxc_specs)
        for spec in specs
        for source in spec.sources
    )
    for source in source_columns:
        details = _infer_column_type_details(df[source])
        logger.info(
            "Column type [%s] = %s | valid=%s | unique=%s | reason=%s",
            source,
            details["type"],
            details["valid_count"],
            details["unique_count"],
            details["reason"],
        )

    fixed_cols = [spec.label for spec in fixed_specs]
    random_cols = [spec.label for spec in random_specs]
    gxe_cols = [spec.label for spec in gxe_specs]
    gxc_cols = [spec.label for spec in gxc_specs]
    for label, specs in (
        ("-c/--cov", fixed_specs),
        ("-rc/--rcov", random_specs),
        ("-gxe/--gxe", gxe_specs),
        ("-gxc/--gxc", gxc_specs),
    ):
        for spec in specs:
            if line_col in spec.sources:
                raise ValueError(f"{label} cannot reuse the line/sample column: {line_col}")

    reserved_sources = {
        source
        for specs in (fixed_specs, random_specs, gxe_specs, gxc_specs)
        for spec in specs
        for source in spec.sources
    }
    overlap_effects = (
        set(source for spec in fixed_specs for source in spec.sources)
        & set(source for spec in random_specs for source in spec.sources)
    )
    if len(overlap_effects) > 0:
        raise ValueError(
            "Columns cannot be assigned to multiple design groups: "
            + ", ".join(sorted(overlap_effects))
        )

    reserved_cols = set(reserved_sources)
    trait_tokens = _split_tokens(args.ncol)
    if len(trait_tokens) > 0:
        trait_cols = _resolve_cli_columns(trait_tokens, effect_cols, "-n/--ncol")
        conflict = sorted((set(trait_cols) & reserved_cols) | ({line_col} & set(trait_cols)))
        if len(conflict) > 0:
            raise ValueError(
                "Trait column(s) overlap with line/env/fixed/random columns: "
                + ", ".join(conflict)
            )
    else:
        trait_candidates = [c for c in effect_cols if c not in reserved_cols]
        trait_cols = _resolve_trait_columns_auto(df, trait_candidates)
    if len(trait_cols) == 0:
        raise ValueError("No usable trait columns were found.")

    n_obs_total = int(df.shape[0])
    unique_lines = df[line_col].drop_duplicates().reset_index(drop=True)
    n_lines_total = int(unique_lines.shape[0])
    env_fixed_label = _format_design_label(env_cols, fixed_cols)
    random_label = _format_random_label(random_cols + gxe_cols + gxc_cols)

    grm_ctx: _GrmContext | None = None
    sparse_grm_ctx: _SparseGrmContext | None = None
    if args.grm is not None:
        grm_ctx = _load_grm_context(
            str(args.grm),
            None,
            unique_lines.astype(str).tolist(),
        )
    if args.grm_sparse is not None:
        sparse_grm_ctx = _load_sparse_grm_context(
            str(args.grm_sparse),
            None,
            unique_lines.astype(str).tolist(),
        )
        if sparse_grm_ctx.id_path is None:
            logger.warning(
                "Sparse GRM has no sibling ID file; REML is assuming the phenotype unique-line order matches the Sparse GRM sample order."
            )

    narrow_path_label = "None"
    if grm_ctx is not None:
        narrow_path_label = "BLUE -> corrected dense joint REML"
    elif sparse_grm_ctx is not None:
        narrow_path_label = (
            "BLUE -> corrected Sparse REML"
            if str(args.grm_sparse_mode) == "raw"
            else "BLUE -> corrected Sparse REML (fastGWA-compatible)"
        )

    emit_cli_configuration(
        logger,
        app_title="JanusX - REML",
        config_title="REML CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "Input",
                [
                    ("Phenotype file", str(args.pheno)),
                    ("Line column", line_col),
                    ("Rows(total)", n_obs_total),
                    ("Lines(unique)", n_lines_total),
                    ("Load time", load_elapsed),
                ],
            ),
            (
                "Columns",
                [
                    ("Traits", ", ".join(trait_cols)),
                    ("Fixed", ", ".join(fixed_cols) if len(fixed_cols) > 0 else "None"),
                    ("Random", ", ".join(random_cols) if len(random_cols) > 0 else "None"),
                    ("GxE", ", ".join(gxe_cols) if len(gxe_cols) > 0 else "None"),
                    ("GxC", ", ".join(gxc_cols) if len(gxc_cols) > 0 else "None"),
                    ("Model", f"{line_col} (random) + compiled random/fixed terms"),
                    ("BLUE model", f"{line_col} (fixed) + compiled fixed/random nuisance terms"),
                ],
            ),
            (
                "GRM",
                [
                    ("GRM file", str(args.grm) if args.grm is not None else "None"),
                    ("GRM ID file", "auto-detect <grm>.id" if args.grm is not None else "None"),
                    ("GRM n", int(grm_ctx.matrix.shape[0]) if grm_ctx is not None else "NA"),
                    ("Sparse GRM file", str(args.grm_sparse) if args.grm_sparse is not None else "None"),
                    ("Sparse GRM ID file", "auto-detect <spgrm>.id" if args.grm_sparse is not None else "None"),
                    ("Sparse GRM n", int(sparse_grm_ctx.n_samples) if sparse_grm_ctx is not None else "NA"),
                    ("Narrow path", narrow_path_label),
                ],
            ),
            (
                "Output",
                [
                    ("Out dir", outdir),
                    ("Prefix", prefix),
                    ("BLUE file", f"{outprefix}.blue.txt"),
                    ("BLUP file", f"{outprefix}.blup.txt"),
                    ("GBLUP file", f"{outprefix}.gblup.txt" if (grm_ctx is not None or sparse_grm_ctx is not None) else "None"),
                    ("Summary file", f"{outprefix}.reml.summary.tsv"),
                    ("Log file", log_path),
                ],
            ),
        ],
    )

    blue_out = pd.DataFrame({line_col: unique_lines.to_numpy(dtype=object)})
    blup_out = pd.DataFrame({line_col: unique_lines.to_numpy(dtype=object)})
    gblup_out = (
        pd.DataFrame({line_col: unique_lines.to_numpy(dtype=object)})
        if grm_ctx is not None or sparse_grm_ctx is not None
        else None
    )
    summary_rows: list[dict[str, typing.Any]] = []

    for trait in trait_cols:
        step_t0 = time.time()
        try:
            y_all = pd.to_numeric(df[trait], errors="coerce")
            mask = y_all.notna() & np.isfinite(y_all)
            all_effect_specs = fixed_specs + random_specs + gxe_specs + gxc_specs
            mask &= _collect_numeric_required_mask_specs(df, all_effect_specs)

            used_obs = int(mask.sum())
            used_lines = int(df.loc[mask, line_col].astype(str).nunique(dropna=False))
            blue_out[trait] = np.nan
            blup_out[trait] = np.nan
            if gblup_out is not None:
                gblup_out[trait] = np.nan

            if used_obs <= 2 or used_lines <= 1:
                logger.warning(
                    f"Trait {trait}: too few observations after filtering (obs={used_obs}, lines={used_lines}); skipped."
                )
                summary_rows.append(
                    {
                        "trait": trait,
                        "used_obs": used_obs,
                        "used_lines": used_lines,
                        "total_obs": n_obs_total,
                        "total_lines": n_lines_total,
                        "env_fixed_label": env_fixed_label,
                        "random_label": random_label,
                        "hsqr": np.nan,
                        "h2_narrow": np.nan,
                        "h2_narrow_vc_ratio_raw": np.nan,
                        "va_joint": np.nan,
                        "vline_joint": np.nan,
                        "noise_mean_joint": np.nan,
                        "pve": np.nan,
                        "lambda": np.nan,
                        "vg": np.nan,
                        "vge": np.nan,
                        "ve": np.nan,
                        "h_env": np.nan,
                        "h_plot": np.nan,
                        "blue_n": np.nan,
                        "missing_grm": np.nan,
                        "narrow_lambda": np.nan,
                        "narrow_sigma_g2": np.nan,
                        "narrow_sigma_e2": np.nan,
                        "narrow_grm_mean_diag": np.nan,
                        "narrow_nnz_k": np.nan,
                        "narrow_offdiag_density_k": np.nan,
                        "narrow_method": "skipped",
                        "elapsed_sec": float(time.time() - step_t0),
                        "status": "skipped_too_few_observations",
                    }
                )
                continue

            source_cols = [
                source
                for specs in (fixed_specs, random_specs, gxe_specs, gxc_specs)
                for spec in specs
                for source in spec.sources
            ]
            sub_cols = _unique_preserve([line_col, trait, *source_cols])
            sub = df.loc[mask, sub_cols].copy()
            y = pd.to_numeric(sub[trait], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
            line_ids_sub = sub[line_col].astype("string").fillna("NA").astype(str)
            env_key = _combine_key(sub, env_cols, "__ENV__").astype(str)

            compiled = _compile_model_terms(
                sub,
                line_col=line_col,
                fixed_specs=fixed_specs,
                random_specs=random_specs,
                gxe_specs=gxe_specs,
                gxc_specs=gxc_specs,
            )
            x_broad = compiled.fixed_matrix
            line_z = compiled.line_z
            z_list = [line_z, *compiled.random_matrices]
            z_names = [line_col, *compiled.random_names]

            broad_model = BLUP(
                y=y,
                X=x_broad,
                Z=z_list if len(z_list) > 0 else None,
                maxiter=int(args.maxiter),
                progress=False,
            )

            hsqr = np.nan
            pve_line = np.nan
            lbd = np.nan
            vg = np.nan
            vge = 0.0
            ve = np.nan
            pve_e = np.nan
            h_env = 1.0
            h_plot = 1.0
            r_eff = 1.0
            status = "ok"
            random_variances: OrderedDict[str, float] = OrderedDict()
            single_obs_per_line = bool(int(line_ids_sub.shape[0]) == int(line_ids_sub.nunique(dropna=False)))
            line_idx = z_names.index(line_col) if line_col in z_names else -1
            gxe_labels = {f"{line_col}×{spec.label}" for spec in gxe_specs}
            gxe_indices = [
                idx for idx, name in enumerate(z_names) if name in gxe_labels
            ]
            if getattr(broad_model, "var", None) is not None and len(z_names) > 0:
                var_all = np.asarray(broad_model.var, dtype=float).reshape(-1)
                if var_all.size >= (len(z_names) + 1):
                    rand_var = var_all[: len(z_names)]
                    ve = float(var_all[-1])
                    for term_name, term_var in zip(z_names, rand_var):
                        random_variances[str(term_name)] = float(term_var)
                        logger.info(
                            "Trait %s: random variance [%s] = %s",
                            trait,
                            term_name,
                            _fmt_metric(term_var),
                        )
                        if np.isfinite(term_var) and float(term_var) <= 1e-9:
                            logger.warning(
                                "Trait %s: variance component [%s] is at/near the non-negative boundary (%s).",
                                trait,
                                term_name,
                                _fmt_metric(term_var),
                            )
                    logger.info(
                        "Trait %s: residual variance = %s",
                        trait,
                        _fmt_metric(ve),
                    )
                    if np.isfinite(ve) and float(ve) <= 1e-9:
                        logger.warning(
                            "Trait %s: residual variance is at/near the non-negative boundary (%s).",
                            trait,
                            _fmt_metric(ve),
                        )
                    total_var = float(np.sum(rand_var) + ve)
                    vg = float(rand_var[line_idx]) if line_idx >= 0 else np.nan
                    vge = float(np.sum(rand_var[gxe_indices])) if gxe_indices else 0.0
                    if total_var > 0.0 and line_idx >= 0:
                        pve_line = float(rand_var[line_idx] / total_var)
                        pve_e = float(ve / total_var)
                    lbd = (
                        float(ve / vg)
                        if np.isfinite(ve) and np.isfinite(vg) and vg > 0.0
                        else np.nan
                    )
                    h_env, h_plot, r_eff = _effective_env_plot_counts(
                        line_ids_sub,
                        sub,
                        env_cols,
                        [],
                    )
                    denom = vg + (vge / h_env) + (ve / h_plot)
                    if np.isfinite(vg) and np.isfinite(denom) and denom > 0.0:
                        hsqr = float(vg / denom)

            if single_obs_per_line and z_names == [line_col]:
                hsqr = np.nan
                pve_line = np.nan
                lbd = np.nan
                vg = np.nan
                status = "warning_single_obs_nonidentifiable_h2"
                logger.warning(
                    f"Trait {trait}: only one observation per line and no ENV/random replication; broad-sense H2 is non-identifiable."
                )

            blup_map = _line_level_blup_from_broad_model(
                broad_model,
                sub,
                line_col=line_col,
                line_z=line_z,
                line_term_idx=line_idx,
            )
            blup_out[trait] = (
                blup_out[line_col]
                .astype(str)
                .map(blup_map)
                .to_numpy(dtype=float)
            )

            stage1_blue = _fit_stage1_blue(
                y_obs=y.reshape(-1),
                sub=sub,
                line_col=line_col,
                trait=trait,
                gxe_var=vge,
                resid_var=ve,
                maxiter=int(args.maxiter),
                logger=logger,
                compiled=compiled,
            )
            blue_map = {
                str(sid): float(val)
                for sid, val in zip(stage1_blue.sample_ids, stage1_blue.values)
            }
            blue_out[trait] = (
                blue_out[line_col]
                .astype(str)
                .map(blue_map)
                .to_numpy(dtype=float)
            )

            h2_narrow = np.nan
            h2_narrow_vc_ratio_raw = np.nan
            va_joint = np.nan
            vline_joint = np.nan
            noise_mean_joint = np.nan
            missing_grm = np.nan
            narrow_lambda = np.nan
            narrow_sigma_g2 = np.nan
            narrow_sigma_e2 = np.nan
            narrow_grm_mean_diag = np.nan
            narrow_nnz_k = np.nan
            narrow_offdiag_density_k = np.nan
            narrow_method = "none"
            blue_n = int(len(stage1_blue.sample_ids))
            if grm_ctx is not None or sparse_grm_ctx is not None:
                blue_trait_df = pd.DataFrame(
                    {
                        line_col: np.asarray(stage1_blue.sample_ids, dtype=object),
                        trait: np.asarray(stage1_blue.values, dtype=float),
                    }
                )

                sid_series = blue_trait_df[line_col].astype(str)
                kinship_index = grm_ctx.index if grm_ctx is not None else sparse_grm_ctx.index
                keep_mask = sid_series.isin(set(kinship_index.keys()))
                missing_grm = int((~keep_mask).sum())
                if missing_grm > 0:
                    logger.warning(
                        f"Trait {trait}: dropped {missing_grm} BLUE lines absent from kinship input."
                    )
                if int(keep_mask.sum()) > 2:
                    kept = blue_trait_df.loc[keep_mask].reset_index(drop=True)
                    kept_ids = kept[line_col].astype(str).tolist()
                    x_stage2 = None
                    if stage1_blue.noise_diag is not None:
                        blue_noise_map = {
                            str(sid): float(value)
                            for sid, value in zip(
                                stage1_blue.sample_ids,
                                np.asarray(stage1_blue.noise_diag, dtype=float),
                            )
                        }
                        noise_diag = np.asarray(
                            [blue_noise_map.get(sid, np.nan) for sid in kept_ids],
                            dtype=float,
                        )
                        if not np.all(np.isfinite(noise_diag) & (noise_diag >= 0.0)):
                            logger.warning(
                                f"Trait {trait}: BLUE covariance diagonal was incomplete; using conservative line-level fallback for invalid entries."
                            )
                            fallback_noise = _line_level_noise_diag(
                                sub,
                                line_col=line_col,
                                env_cols=env_cols,
                                line_ids=kept_ids,
                                vge=vge,
                                ve=ve,
                            )
                            noise_diag = np.where(
                                np.isfinite(noise_diag) & (noise_diag >= 0.0),
                                noise_diag,
                                fallback_noise,
                            )
                    else:
                        noise_diag = _line_level_noise_diag(
                            sub,
                            line_col=line_col,
                            env_cols=env_cols,
                            line_ids=kept_ids,
                            vge=vge,
                            ve=ve,
                        )
                    try:
                        noise_mean_joint = float(np.mean(noise_diag)) if noise_diag.size > 0 else np.nan
                        if noise_diag.size > 0 and np.all(np.isfinite(noise_diag) & (noise_diag >= 0.0)):
                            logger.info(
                                "Trait %s: stage1 BLUE uncertainty diag min=%s | median=%s | mean=%s | max=%s",
                                trait,
                                _fmt_metric(np.min(noise_diag)),
                                _fmt_metric(np.median(noise_diag)),
                                _fmt_metric(np.mean(noise_diag)),
                                _fmt_metric(np.max(noise_diag)),
                            )
                        if grm_ctx is not None:
                            grm_idx = [grm_ctx.index[sid] for sid in kept_ids]
                            kinship = grm_ctx.matrix[np.ix_(grm_idx, grm_idx)]
                            joint_state = _fit_dense_narrow_corrected(
                                kept[trait].to_numpy(dtype=float),
                                kinship=kinship,
                                noise_diag=noise_diag,
                                x_fixed=x_stage2,
                                maxiter=int(args.maxiter),
                            )
                            va_joint = float(joint_state.va)
                            vline_joint = float(joint_state.vline)
                            noise_mean_joint = float(joint_state.noise_mean)
                            h2_narrow = float(joint_state.h2_raw)
                            latent_denom = float(joint_state.va + joint_state.vline)
                            h2_narrow_vc_ratio_raw = (
                                float(joint_state.va / latent_denom)
                                if latent_denom > 0.0
                                else np.nan
                            )
                            narrow_method = "joint_dense_corrected_reml"
                            if va_joint <= 1e-9 or vline_joint <= 1e-9:
                                logger.warning(
                                    "Trait %s: dense line-level variance component is at/near the non-negative boundary (Va=%s, Vline=%s).",
                                    trait,
                                    _fmt_metric(va_joint),
                                    _fmt_metric(vline_joint),
                                )
                            if np.isfinite(hsqr) and np.isfinite(h2_narrow) and (h2_narrow > hsqr * 1.02):
                                logger.warning(
                                    f"Trait {trait}: corrected narrow h2 ({h2_narrow:.6g}) exceeds broad H2 ({hsqr:.6g}); broad and narrow estimators are on different effective scales."
                                )
                            g_map = {
                                kept_ids[i]: float(joint_state.add_blup[i])
                                for i in range(len(kept_ids))
                            }
                            assert gblup_out is not None
                            gblup_out[trait] = (
                                gblup_out[line_col]
                                .astype(str)
                                .map(g_map)
                                .to_numpy(dtype=float)
                            )
                        else:
                            sparse_idx = np.ascontiguousarray(
                                np.asarray([sparse_grm_ctx.index[sid] for sid in kept_ids], dtype=np.int64),
                                dtype=np.int64,
                            )
                            sparse_null = _fit_sparse_narrow_corrected(
                                jxgrm_path=str(sparse_grm_ctx.path),
                                sample_idx=sparse_idx,
                                y_vec=kept[trait].to_numpy(dtype=float),
                                x_cov=x_stage2,
                                noise_diag=noise_diag,
                                objective_mode=str(args.grm_sparse_mode),
                                threads=int(args.thread),
                            )
                            h2_narrow = float(sparse_null.get("pve", float("nan")))
                            h2_narrow_vc_ratio_raw = float(
                                sparse_null.get(
                                    "pve_vc_ratio_raw",
                                    sparse_null.get("pve", float("nan")),
                                )
                            )
                            narrow_lambda = float(sparse_null.get("lambda", float("nan")))
                            narrow_sigma_g2 = float(sparse_null.get("sigma_g2", float("nan")))
                            narrow_sigma_e2 = float(sparse_null.get("sigma_e2", float("nan")))
                            if (
                                (np.isfinite(narrow_sigma_g2) and narrow_sigma_g2 <= 1e-9)
                                or (np.isfinite(narrow_sigma_e2) and narrow_sigma_e2 <= 1e-9)
                            ):
                                logger.warning(
                                    "Trait %s: sparse variance component is at/near the non-negative boundary (sigma_g2=%s, sigma_e2=%s).",
                                    trait,
                                    _fmt_metric(narrow_sigma_g2),
                                    _fmt_metric(narrow_sigma_e2),
                                )
                            narrow_grm_mean_diag = float(
                                sparse_null.get(
                                    "grm_mean_diag",
                                    sparse_null.get("mean_diag_k", float("nan")),
                                )
                            )
                            narrow_nnz_k = float(sparse_null.get("nnz_k", float("nan")))
                            narrow_offdiag_density_k = float(sparse_null.get("offdiag_density_k", float("nan")))
                            narrow_method = (
                                "blue_corrected_sparse_reml_fastgwa"
                                if str(args.grm_sparse_mode) == "fastgwa"
                                else "blue_corrected_sparse_reml"
                            )
                            noise_mean_joint = float(
                                sparse_null.get("stage1_noise_mean", noise_mean_joint)
                            )
                            if np.isfinite(hsqr) and np.isfinite(h2_narrow) and (h2_narrow > hsqr * 1.02):
                                logger.warning(
                                    f"Trait {trait}: corrected sparse narrow h2 ({h2_narrow:.6g}) exceeds broad H2 ({hsqr:.6g}); broad and narrow estimators are on different effective scales."
                                )
                            if gblup_out is not None:
                                sparse_g = _sparse_additive_blup_from_subset(
                                    jxgrm_path=str(sparse_grm_ctx.path),
                                    sample_idx=sparse_idx,
                                    y_vec=kept[trait].to_numpy(dtype=float),
                                    noise_diag=noise_diag,
                                    sigma_g2=narrow_sigma_g2,
                                    sigma_e2=narrow_sigma_e2,
                                    x_cov=x_stage2,
                                )
                                sparse_g_map = {
                                    kept_ids[i]: float(sparse_g[i])
                                    for i in range(len(kept_ids))
                                }
                                gblup_out[trait] = (
                                    gblup_out[line_col]
                                    .astype(str)
                                    .map(sparse_g_map)
                                    .to_numpy(dtype=float)
                                )
                    except Exception as narrow_exc:
                        if grm_ctx is not None:
                            logger.warning(
                                f"Trait {trait}: corrected dense narrow REML failed ({type(narrow_exc).__name__}: {narrow_exc}); narrow-sense h2 skipped."
                            )
                            narrow_method = "failed_corrected_dense_reml"
                        else:
                            logger.warning(
                                f"Trait {trait}: corrected Sparse REML failed ({type(narrow_exc).__name__}: {narrow_exc}); narrow-sense h2 skipped."
                            )
                            narrow_method = "failed_corrected_sparse_reml"
                else:
                    logger.warning(
                        f"Trait {trait}: too few lines overlap with kinship input after filtering; narrow-sense h2 skipped."
                    )
                    narrow_method = "skipped_grm_overlap" if grm_ctx is not None else "skipped_sparse_grm_overlap"

            # logger.info("-" * 72)
            # logger.info(
            #     f"{success_symbol()} Trait={trait} | obs={used_obs} | lines={used_lines} | H2={_fmt_metric(hsqr)} | h2={_fmt_metric(h2_narrow)} | method={narrow_method} | elapsed={format_elapsed(time.time() - step_t0)}"
            # )
            if np.isfinite(h2_narrow):
                logger.info(
                    f"  narrow(h2)={_fmt_metric(h2_narrow)}"
                )
            if np.isfinite(h2_narrow_vc_ratio_raw) and np.isfinite(h2_narrow):
                logger.info(
                    "  narrow(latent_vc_ratio_raw)=%s%s",
                    _fmt_metric(h2_narrow_vc_ratio_raw),
                    (
                        f" | grm_mean_diag={_fmt_metric(narrow_grm_mean_diag)}"
                        if np.isfinite(narrow_grm_mean_diag)
                        else ""
                    ),
                )
            if np.isfinite(narrow_grm_mean_diag) or np.isfinite(narrow_sigma_g2) or np.isfinite(narrow_sigma_e2):
                logger.info(
                    "  sparse lambda=%s | sigma_g2=%s | sigma_e2=%s | h2=%s | nnz(K)=%s | offdiag_density=%s",
                    _fmt_metric(narrow_lambda),
                    _fmt_metric(narrow_sigma_g2),
                    _fmt_metric(narrow_sigma_e2),
                    _fmt_metric(h2_narrow),
                    _fmt_metric(narrow_nnz_k),
                    _fmt_metric(narrow_offdiag_density_k),
                )
            if np.isfinite(noise_mean_joint) and sparse_grm_ctx is not None:
                logger.info(
                    "  sparse stage1_noise_mean=%s | corrected phenotype-scale h2=%s",
                    _fmt_metric(noise_mean_joint),
                    _fmt_metric(h2_narrow),
                )
            if np.isfinite(va_joint):
                logger.info(
                    f"  joint additive={_fmt_metric(va_joint)} | joint line_nonadd={_fmt_metric(vline_joint)} | joint noise_mean={_fmt_metric(noise_mean_joint)}"
                )

            summary_rows.append(
                {
                    "trait": trait,
                    "used_obs": used_obs,
                    "used_lines": used_lines,
                    "total_obs": n_obs_total,
                    "total_lines": n_lines_total,
                    "env_fixed_label": env_fixed_label,
                    "random_label": random_label,
                    "hsqr": float(hsqr) if np.isfinite(hsqr) else np.nan,
                    "h2_narrow": float(h2_narrow) if np.isfinite(h2_narrow) else np.nan,
                    "h2_narrow_vc_ratio_raw": (
                        float(h2_narrow_vc_ratio_raw)
                        if np.isfinite(h2_narrow_vc_ratio_raw)
                        else np.nan
                    ),
                    "va_joint": float(va_joint) if np.isfinite(va_joint) else np.nan,
                    "vline_joint": float(vline_joint) if np.isfinite(vline_joint) else np.nan,
                    "noise_mean_joint": float(noise_mean_joint) if np.isfinite(noise_mean_joint) else np.nan,
                    "pve": float(pve_line) if np.isfinite(pve_line) else np.nan,
                    "lambda": float(lbd) if np.isfinite(lbd) else np.nan,
                    "vg": float(vg) if np.isfinite(vg) else np.nan,
                    "vge": float(vge) if np.isfinite(vge) else np.nan,
                    "ve": float(ve) if np.isfinite(ve) else np.nan,
                    "h_env": float(h_env),
                    "h_plot": float(h_plot),
                    "r": float(r_eff),
                    "blue_n": float(blue_n),
                    "missing_grm": float(missing_grm) if np.isfinite(missing_grm) else np.nan,
                    "narrow_lambda": float(narrow_lambda) if np.isfinite(narrow_lambda) else np.nan,
                    "narrow_sigma_g2": float(narrow_sigma_g2) if np.isfinite(narrow_sigma_g2) else np.nan,
                    "narrow_sigma_e2": float(narrow_sigma_e2) if np.isfinite(narrow_sigma_e2) else np.nan,
                    "narrow_grm_mean_diag": (
                        float(narrow_grm_mean_diag)
                        if np.isfinite(narrow_grm_mean_diag)
                        else np.nan
                    ),
                    "narrow_nnz_k": float(narrow_nnz_k) if np.isfinite(narrow_nnz_k) else np.nan,
                    "narrow_offdiag_density_k": float(narrow_offdiag_density_k) if np.isfinite(narrow_offdiag_density_k) else np.nan,
                    "narrow_method": narrow_method,
                    "elapsed_sec": float(time.time() - step_t0),
                    "status": status,
                }
            )
        except Exception as exc:
            logger.exception(f"Trait {trait}: REML failed: {exc}")
            blue_out[trait] = np.nan
            blup_out[trait] = np.nan
            if gblup_out is not None:
                gblup_out[trait] = np.nan
            summary_rows.append(
                {
                    "trait": trait,
                    "used_obs": np.nan,
                    "used_lines": np.nan,
                    "total_obs": n_obs_total,
                    "total_lines": n_lines_total,
                    "env_fixed_label": env_fixed_label,
                    "random_label": random_label,
                    "hsqr": np.nan,
                    "h2_narrow": np.nan,
                    "h2_narrow_vc_ratio_raw": np.nan,
                    "va_joint": np.nan,
                    "vline_joint": np.nan,
                    "noise_mean_joint": np.nan,
                    "pve": np.nan,
                    "lambda": np.nan,
                    "vg": np.nan,
                    "vge": np.nan,
                    "ve": np.nan,
                    "h_env": np.nan,
                    "h_plot": np.nan,
                    "blue_n": np.nan,
                    "missing_grm": np.nan,
                    "narrow_lambda": np.nan,
                    "narrow_sigma_g2": np.nan,
                    "narrow_sigma_e2": np.nan,
                    "narrow_grm_mean_diag": np.nan,
                    "narrow_nnz_k": np.nan,
                    "narrow_offdiag_density_k": np.nan,
                    "narrow_method": "failed",
                    "elapsed_sec": float(time.time() - step_t0),
                    "status": f"failed:{type(exc).__name__}",
                }
            )

    out_blue = f"{outprefix}.blue.txt"
    out_blup = f"{outprefix}.blup.txt"
    out_summary = f"{outprefix}.reml.summary.tsv"
    blue_out.to_csv(out_blue, sep="\t", index=False)
    blup_out.to_csv(out_blup, sep="\t", index=False)
    if gblup_out is not None:
        out_gblup = f"{outprefix}.gblup.txt"
        gblup_out.to_csv(out_gblup, sep="\t", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_summary, sep="\t", index=False)

    summary_console = _render_summary_table(summary_rows, log_style=False)
    summary_log = _render_summary_table(summary_rows, log_style=True)
    # if summary_console != "":
    #     print(summary_console)
    if summary_log != "":
        logger.info(summary_log)
    # logger.info("=" * 60)
    log_success(logger, f"BLUE table saved: {format_path_for_display(out_blue)}")
    log_success(logger, f"BLUP table saved: {format_path_for_display(out_blup)}")
    if gblup_out is not None:
        log_success(logger, f"GBLUP table saved: {format_path_for_display(f'{outprefix}.gblup.txt')}")
    log_success(logger, f"Summary table saved: {format_path_for_display(out_summary)}")
    logger.info(f"Total elapsed: {format_elapsed(time.time() - t0)}")


if __name__ == "__main__":
    from janusx.script._common.interrupt import force_exit, install_interrupt_handlers
    install_interrupt_handlers()
    try:
        main()
    except KeyboardInterrupt:
        force_exit(130, "Interrupted by user (Ctrl+C).")
    except Exception as exc:
        print_failure("REML ...Failed")
        print(f"Error: {exc}")
        raise
