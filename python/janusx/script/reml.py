# -*- coding: utf-8 -*-
"""
JanusX: REML/BLUP Heritability Estimation from Phenotype Table

Input table
-----------
- First column is always sample ID.
- Remaining columns are candidate phenotype/fixed/random effect columns.
- String/categorical columns selected by `-f/-r` are one-hot encoded by default.

Examples
--------
  jx reml -file pheno.tsv -l sample_id -e year,loc -o outdir
  jx reml -file pheno.tsv -l sample_id -e year,loc -f PCA1,PCA2 -r block -k data.cGRM.npy -p Yield
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
    _splmm_normalize_sparse_grm_path,
    _splmm_sparse_grm_diag_stats,
    _splmm_sparse_null_fit,
)
from janusx.pyBLUP.assoc import LMM
from janusx.pyBLUP.blup import BLUP
from ._common.cli_args import (
    add_common_out_arg,
    add_common_prefix_arg,
    add_common_thread_arg,
)
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
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
class _LmmNullStats:
    beta: np.ndarray
    se: np.ndarray
    pval: np.ndarray
    g_blup: np.ndarray


@dataclass
class _Stage1BlueResult:
    sample_ids: list[str]
    values: np.ndarray


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
    return int(vals.notna().sum()) == non_na


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
            mask &= pd.to_numeric(s, errors="coerce").notna()
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
        headers = ["Trait", "N_Obs (Lines)", "Env / Fixed", "Random", "H2 (Broad)", "h2 (Narrow)", "PVEc"]
    else:
        headers = ["Trait", "N_Obs(Lines)", "Env/Fixed", "Random", "H2", "h2", "PVEc"]

    body: list[list[str]] = []
    for row in rows:
        nobs = int(row.get("used_obs", 0))
        nlines = int(row.get("used_lines", 0))
        obs_label = f"{nobs:,} ({nlines:,})" if log_style else f"{nobs:,}({nlines:,})"
        body.append(
            [
                str(row.get("trait", "NA")),
                obs_label,
                str(row.get("env_fixed_label", "None")),
                str(row.get("random_label", "None")),
                _fmt_metric(row.get("hsqr")),
                _fmt_metric(row.get("h2_narrow")),
                _fmt_metric(row.get("pvec")),
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


def _lmm_null_stats(model: LMM) -> _LmmNullStats:
    s = np.asarray(model.S, dtype=np.float64).reshape(-1)
    x_rot = np.asarray(model.Xcov, dtype=np.float64)
    y_rot = np.asarray(model.y, dtype=np.float64).reshape(-1, 1)
    lbd = float(model.lbd_null)
    if s.size == 0:
        raise ValueError("LMM null model has zero samples.")

    vinv = 1.0 / (s + lbd)
    xt_vinv = x_rot.T * vinv.reshape(1, -1)
    xt_vinv_x = xt_vinv @ x_rot
    xt_vinv_y = xt_vinv @ y_rot
    beta = np.linalg.solve(xt_vinv_x, xt_vinv_y)
    r_rot = y_rot - x_rot @ beta

    n = int(x_rot.shape[0])
    p = int(x_rot.shape[1])
    sigma2 = float((r_rot.T @ (vinv.reshape(-1, 1) * r_rot))[0, 0]) / max(1, n - p)
    cov_beta = np.linalg.pinv(xt_vinv_x) * sigma2
    se = np.sqrt(np.clip(np.diag(cov_beta), a_min=0.0, a_max=None)).reshape(-1, 1)
    tval = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0.0)
    pval = 2.0 * student_t.sf(np.abs(tval), df=max(1, n - p))

    u = np.asarray(model.Dh.T, dtype=np.float64)
    g_rot = (s.reshape(-1, 1) / (s.reshape(-1, 1) + lbd)) * r_rot
    g_blup = u @ g_rot
    return _LmmNullStats(
        beta=np.asarray(beta, dtype=float).reshape(-1),
        se=np.asarray(se, dtype=float).reshape(-1),
        pval=np.asarray(pval, dtype=float).reshape(-1),
        g_blup=np.asarray(g_blup, dtype=float).reshape(-1),
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
    stage2_fixed_terms: list[_TermSpec] = []
    dropped_varying_fixed: list[str] = []
    for term in fixed_terms_all:
        if _term_constant_within_line(sub, line_col, str(term.name)):
            stage2_fixed_terms.append(term)
        else:
            dropped_varying_fixed.append(str(term.name))

    if len(dropped_varying_fixed) > 0:
        logger.warning(
            f"Trait {trait}: fixed covariates varying within line are not retained for BLUE stage-2 covariates -> {', '.join(dropped_varying_fixed)}"
        )
    return list(random_terms_all), stage2_fixed_terms


def _fit_stage1_blue(
    y_obs: np.ndarray,
    sub: pd.DataFrame,
    *,
    line_col: str,
    trait: str,
    env_cols: list[str],
    stage1_random_terms: list[_TermSpec],
    gxe_var: float | None = None,
    resid_var: float | None = None,
    maxiter: int,
    logger: typing.Any,
) -> _Stage1BlueResult:
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


# Reserved for future re-activation of the joint additive + nonadditive REML path.
# The active narrow-sense workflow is currently:
#   stage-1 BLUE -> stage-2 GWAS null LMM (kinship only) -> h2/prediction.
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


def _resolve_cli_columns(
    values: Iterable[str] | None,
    candidates: list[str],
    label: str,
) -> list[str]:
    return _resolve_columns(_split_tokens(values), candidates, label, index_base=0)


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx reml",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx reml -file pheno.tsv -l sample_id -e year,loc -o outdir",
                "jx reml -file pheno.tsv -l sample_id -e year,loc -f PCA1,PCA2 -r block -k data.cGRM.npy -p Yield",
                "jx reml -file pheno.tsv -l sample_id -spk data.cGRM.spgrm -p Yield -o outdir",
            ]
        ),
    )

    req = parser.add_argument_group("Required Arguments")
    req.add_argument(
        "-file",
        "--file",
        required=True,
        type=str,
        help="Input phenotype table (.tsv/.csv/whitespace).",
    )

    opt = parser.add_argument_group("Optional Arguments")
    opt.add_argument(
        "-l",
        "--line",
        default=None,
        metavar="COL",
        help="Line/sample ID column. Numeric indices are 0-based on the original input table. Default: first column.",
    )
    opt.add_argument(
        "-e",
        "--env",
        action="append",
        default=[],
        metavar="COL",
        help=(
            "Environment factor column(s), e.g. year,loc. Numeric indices are 0-based on the original input table. They are combined as one ENV factor."
        ),
    )
    opt.add_argument(
        "-f",
        "--fixed",
        action="append",
        default=[],
        metavar="COL",
        help=(
            "Additional fixed covariates. Numeric indices are 0-based on the original input table. Numeric columns stay numeric; categorical columns are one-hot encoded."
        ),
    )
    opt.add_argument(
        "-r",
        "--random",
        action="append",
        default=[],
        metavar="COL",
        help=(
            "Additional random nuisance covariates. Numeric indices are 0-based on the original input table. Numeric columns stay numeric; categorical columns are one-hot encoded."
        ),
    )
    opt.add_argument(
        "-p",
        "--pheno",
        action="append",
        default=[],
        metavar="COL",
        help=(
            "Trait column(s). Numeric indices are 0-based on the original input table. If omitted, all numeric non-design columns are used."
        ),
    )
    opt.add_argument(
        "-k",
        "-grm",
        "--grm",
        dest="grm",
        default=None,
        metavar="FILE",
        help=(
            "Optional GRM matrix. When provided, narrow-sense h2 and genomic BLUP are estimated in addition to broad-sense H2 and BLUE."
        ),
    )
    opt.add_argument(
        "-spk",
        "--grm-sparse",
        dest="grm_sparse",
        default=None,
        metavar="FILE",
        help=(
            "Optional precomputed Sparse GRM (.spgrm/.jxgrm). When provided, narrow-sense h2 is estimated via Sparse REML on the BLUE phenotype scale for fast debugging and narrow heritability checks."
        ),
    )
    opt.add_argument(
        "--spk-mode",
        dest="grm_sparse_mode",
        choices=("raw", "fastgwa"),
        default="fastgwa",
        help=(
            "Sparse REML objective for -spk/--grm-sparse. "
            "`raw` uses JanusX profile REML on the sparse K directly; "
            "`fastgwa` uses a fastGWA-compatible fixed-Vp sparse REML objective "
            "matched to GCTA fastGWA-REML behavior (default: %(default)s)."
        ),
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


def main() -> None:
    t0 = time.time()
    args = build_parser().parse_args()
    if int(args.thread) <= 0:
        raise ValueError("-t/--thread must be a positive integer.")
    args.thread = int(apply_outer_thread_cap(int(args.thread)))
    if args.grm is not None and args.grm_sparse is not None:
        raise ValueError("Please provide only one of -k/--grm or -spk/--grm-sparse.")
    if args.grm_sparse is not None:
        args.grm_sparse = _splmm_normalize_sparse_grm_path(str(args.grm_sparse))
    outdir = os.path.normpath(str(args.out or "."))
    os.makedirs(outdir, mode=0o755, exist_ok=True)
    prefix = str(args.prefix or strip_default_prefix_suffix(os.path.basename(str(args.file))))
    outprefix = os.path.join(outdir, prefix)
    log_path = f"{outprefix}.reml.log"
    logger = setup_logging(log_path)

    if not ensure_file_exists(logger, str(args.file), "Input file"):
        return
    if args.grm is not None and not ensure_file_exists(logger, str(args.grm), "GRM file"):
        return
    if args.grm_sparse is not None and not ensure_file_exists(logger, str(args.grm_sparse), "Sparse GRM file"):
        return

    load_t0 = time.time()
    try:
        df = _read_table_with_optional_header(str(args.file))
    except Exception:
        raise
    load_elapsed = format_elapsed(time.time() - load_t0)

    if df.shape[1] < 2:
        raise ValueError("Input file must contain at least 2 columns.")

    df = df.copy()
    all_cols = [str(c) for c in df.columns]

    line_spec = _resolve_cli_columns(
        [args.line] if args.line is not None else [],
        all_cols,
        "-l/--line",
    )
    if len(line_spec) > 1:
        raise ValueError("Please specify exactly one line column with -l/--line.")
    line_col = str(line_spec[0]) if len(line_spec) == 1 else str(all_cols[0])
    df[line_col] = df[line_col].astype("string").fillna("NA").astype(str)

    effect_cols = [c for c in all_cols if c != line_col]
    env_cols = _resolve_cli_columns(args.env, all_cols, "-e/--env")
    fixed_cols = _resolve_cli_columns(args.fixed, all_cols, "-f/--fixed")
    random_cols = _resolve_cli_columns(args.random, all_cols, "-r/--random")

    for label, cols in (
        ("-e/--env", env_cols),
        ("-f/--fixed", fixed_cols),
        ("-r/--random", random_cols),
    ):
        if line_col in cols:
            raise ValueError(f"{label} cannot reuse the line/sample column: {line_col}")

    overlap_effects = (set(env_cols) & set(fixed_cols)) | (set(env_cols) & set(random_cols)) | (set(fixed_cols) & set(random_cols))
    if len(overlap_effects) > 0:
        raise ValueError(
            "Columns cannot be assigned to multiple design groups: "
            + ", ".join(sorted(overlap_effects))
        )

    reserved_cols = set(env_cols) | set(fixed_cols) | set(random_cols)
    trait_tokens = _split_tokens(args.pheno)
    if len(trait_tokens) > 0:
        trait_cols = _resolve_cli_columns(trait_tokens, all_cols, "-p/--pheno")
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

    fixed_terms = [_TermSpec(name=str(c), force_onehot=False) for c in fixed_cols]
    random_terms_all = [_TermSpec(name=str(c), force_onehot=False) for c in random_cols]

    n_obs_total = int(df.shape[0])
    unique_lines = df[line_col].drop_duplicates().reset_index(drop=True)
    n_lines_total = int(unique_lines.shape[0])
    env_fixed_label = _format_design_label(env_cols, fixed_cols)
    random_label = _format_random_label(random_cols)

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
        narrow_path_label = "BLUE -> GWAS null LMM"
    elif sparse_grm_ctx is not None:
        narrow_path_label = (
            "BLUE -> Sparse REML"
            if str(args.grm_sparse_mode) == "raw"
            else "BLUE -> Sparse REML (fastGWA-compatible)"
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
                    ("File", str(args.file)),
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
                    ("ENV", "|".join(env_cols) if len(env_cols) > 0 else "None"),
                    ("Fixed", ", ".join(fixed_cols) if len(fixed_cols) > 0 else "None"),
                    ("Random", ", ".join(random_cols) if len(random_cols) > 0 else "None"),
                    ("Broad model", f"{line_col} (random) + {line_col}xENV (random)"),
                    ("BLUE stage-1", f"{line_col} (fixed) + ENV (fixed) + {line_col}xENV (random)"),
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
                    ("GBLUP file", f"{outprefix}.gblup.txt" if grm_ctx is not None else "None"),
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
        if grm_ctx is not None
        else None
    )
    summary_rows: list[dict[str, typing.Any]] = []

    for trait in trait_cols:
        step_t0 = time.time()
        try:
            y_all = pd.to_numeric(df[trait], errors="coerce")
            mask = y_all.notna()
            mask &= _collect_numeric_required_mask(df, fixed_terms)
            mask &= _collect_numeric_required_mask(df, random_terms_all)

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
                        "pvec": np.nan,
                        "h2_blue_raw": np.nan,
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
                        "narrow_mean_diag_k": np.nan,
                        "narrow_pve_diag_scaled": np.nan,
                        "narrow_nnz_k": np.nan,
                        "narrow_offdiag_density_k": np.nan,
                        "narrow_method": "skipped",
                        "elapsed_sec": float(time.time() - step_t0),
                        "status": "skipped_too_few_observations",
                    }
                )
                continue

            sub_cols = _unique_preserve([line_col, trait, *env_cols, *fixed_cols, *random_cols])
            sub = df.loc[mask, sub_cols].copy()
            y = pd.to_numeric(sub[trait], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
            line_ids_sub = sub[line_col].astype("string").fillna("NA").astype(str)
            env_key = _combine_key(sub, env_cols, "__ENV__").astype(str)

            x_blocks: list[np.ndarray] = []
            x_names: list[str] = []
            fixed_x, fixed_names = _encode_fixed_design(
                sub,
                fixed_terms,
                trait=trait,
                logger=logger,
            )
            if fixed_x is not None:
                x_blocks.append(np.asarray(fixed_x, dtype=float))
                x_names.extend(fixed_names)

            if len(env_cols) > 0:
                env_arr, env_names = _onehot_encode_series(
                    env_key,
                    prefix="ENV",
                    drop_first=True,
                    sparse_output=False,
                )
                if int(env_arr.shape[1]) > 0:
                    x_blocks.append(np.asarray(env_arr, dtype=float))
                    x_names.extend(env_names)

            x_broad = np.concatenate(x_blocks, axis=1) if len(x_blocks) > 0 else None

            z_list: list[typing.Union[np.ndarray, sparse.spmatrix]] = []
            z_names: list[str] = []
            line_z, _line_names = _onehot_encode_series(
                line_ids_sub,
                prefix=line_col,
                drop_first=False,
                sparse_output=True,
            )
            if int(line_z.shape[1]) > 0:
                z_list.append(line_z)
                z_names.append(line_col)

            gxe_name = f"{line_col}xENV"
            if len(env_cols) > 0 and int(env_key.nunique(dropna=False)) > 1:
                gxe_key = (line_ids_sub + "@@" + env_key).astype("string")
                gxe_z, _ = _onehot_encode_series(
                    gxe_key,
                    prefix=gxe_name,
                    drop_first=False,
                    sparse_output=True,
                )
                if int(gxe_z.shape[1]) > 0:
                    z_list.append(gxe_z)
                    z_names.append(gxe_name)

            extra_random, extra_random_names = _encode_random_design(
                sub,
                random_terms_all,
                trait=trait,
                logger=logger,
            )
            z_list.extend(extra_random)
            z_names.extend(extra_random_names)

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
            single_obs_per_line = bool(int(line_ids_sub.shape[0]) == int(line_ids_sub.nunique(dropna=False)))
            line_idx = z_names.index(line_col) if line_col in z_names else -1
            gxe_idx = z_names.index(gxe_name) if gxe_name in z_names else -1
            if getattr(broad_model, "var", None) is not None and len(z_names) > 0:
                var_all = np.asarray(broad_model.var, dtype=float).reshape(-1)
                if var_all.size >= (len(z_names) + 1):
                    rand_var = var_all[: len(z_names)]
                    ve = float(var_all[-1])
                    total_var = float(np.sum(rand_var) + ve)
                    vg = float(rand_var[line_idx]) if line_idx >= 0 else np.nan
                    vge = float(rand_var[gxe_idx]) if gxe_idx >= 0 else 0.0
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

            stage1_random_terms, stage2_fixed_terms = _build_stage1_blue_terms(
                sub,
                line_col=line_col,
                trait=trait,
                fixed_terms_all=fixed_terms,
                random_terms_all=random_terms_all,
                logger=logger,
            )
            stage1_blue = _fit_stage1_blue(
                y_obs=y.reshape(-1),
                sub=sub,
                line_col=line_col,
                trait=trait,
                env_cols=env_cols,
                stage1_random_terms=stage1_random_terms,
                gxe_var=vge,
                resid_var=ve,
                maxiter=int(args.maxiter),
                logger=logger,
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
            pvec = np.nan
            h2_blue_raw = np.nan
            va_joint = np.nan
            vline_joint = np.nan
            noise_mean_joint = np.nan
            missing_grm = np.nan
            narrow_lambda = np.nan
            narrow_sigma_g2 = np.nan
            narrow_sigma_e2 = np.nan
            narrow_mean_diag_k = np.nan
            narrow_pve_diag_scaled = np.nan
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
                if len(stage2_fixed_terms) > 0:
                    fixed_keep = [line_col] + [str(t.name) for t in stage2_fixed_terms]
                    fixed_line_df = (
                        sub[fixed_keep]
                        .drop_duplicates(subset=[line_col], keep="first")
                        .copy()
                    )
                    blue_trait_df = blue_trait_df.merge(
                        fixed_line_df,
                        on=line_col,
                        how="left",
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
                    x_stage2, _ = _encode_fixed_design(
                        kept,
                        stage2_fixed_terms,
                        trait=trait,
                        logger=logger,
                    )
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
                        if grm_ctx is not None:
                            grm_idx = [grm_ctx.index[sid] for sid in kept_ids]
                            kinship = grm_ctx.matrix[np.ix_(grm_idx, grm_idx)]
                            # Future hook:
                            #   exact/approx joint additive + line nonadditive REML
                            # is intentionally disabled for now. We currently use the
                            # BLUE phenotype directly in the GWAS null model.
                            lmm = LMM(
                                y=kept[trait].to_numpy(dtype=float),
                                X=x_stage2,
                                kinship=kinship,
                            )
                            h2_blue_raw = float(lmm.pve) if np.isfinite(lmm.pve) else np.nan
                            h2_narrow = h2_blue_raw
                            pvec = np.nan
                            narrow_lambda = float(lmm.lbd_null) if np.isfinite(float(lmm.lbd_null)) else np.nan
                            narrow_method = "blue_null_lmm"
                            if np.isfinite(hsqr) and np.isfinite(h2_narrow) and (h2_narrow > hsqr * 1.02):
                                logger.warning(
                                    f"Trait {trait}: BLUE-null narrow h2 ({h2_narrow:.6g}) exceeds broad H2 ({hsqr:.6g}); broad and narrow estimators are on different effective scales."
                                )
                            narrow_stats = _lmm_null_stats(lmm)
                            g_map = {
                                kept_ids[i]: float(narrow_stats.g_blup[i])
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
                            sparse_null = _splmm_sparse_null_fit(
                                jxgrm_path=str(sparse_grm_ctx.path),
                                sample_idx=sparse_idx,
                                y_vec=kept[trait].to_numpy(dtype=float),
                                x_cov=x_stage2,
                                progress_callback=None,
                                objective_mode=str(args.grm_sparse_mode),
                                threads=int(args.thread),
                            )
                            h2_blue_raw = float(sparse_null.get("pve", float("nan")))
                            h2_narrow = h2_blue_raw
                            pvec = float(sparse_null.get("pve_diag_scaled", float("nan")))
                            narrow_lambda = float(sparse_null.get("lambda", float("nan")))
                            narrow_sigma_g2 = float(sparse_null.get("sigma_g2", float("nan")))
                            narrow_sigma_e2 = float(sparse_null.get("sigma_e2", float("nan")))
                            narrow_mean_diag_k = float(sparse_null.get("mean_diag_k", float("nan")))
                            narrow_pve_diag_scaled = pvec
                            narrow_nnz_k = float(sparse_null.get("nnz_k", float("nan")))
                            narrow_offdiag_density_k = float(sparse_null.get("offdiag_density_k", float("nan")))
                            narrow_method = (
                                "blue_null_sparse_reml_fastgwa"
                                if str(args.grm_sparse_mode) == "fastgwa"
                                else "blue_null_sparse_reml"
                            )
                            if np.isfinite(hsqr) and np.isfinite(h2_narrow) and (h2_narrow > hsqr * 1.02):
                                logger.warning(
                                    f"Trait {trait}: BLUE-null sparse narrow h2 ({h2_narrow:.6g}) exceeds broad H2 ({hsqr:.6g}); broad and narrow estimators are on different effective scales."
                                )
                    except Exception as narrow_exc:
                        if grm_ctx is not None:
                            logger.warning(
                                f"Trait {trait}: BLUE-null GWAS LMM failed ({type(narrow_exc).__name__}: {narrow_exc}); narrow-sense h2 skipped."
                            )
                            narrow_method = "failed_blue_null_lmm"
                        else:
                            logger.warning(
                                f"Trait {trait}: BLUE-null Sparse REML failed ({type(narrow_exc).__name__}: {narrow_exc}); narrow-sense h2 skipped."
                            )
                            narrow_method = "failed_blue_null_sparse_reml"
                else:
                    logger.warning(
                        f"Trait {trait}: too few lines overlap with kinship input after filtering; narrow-sense h2 skipped."
                    )
                    narrow_method = "skipped_grm_overlap" if grm_ctx is not None else "skipped_sparse_grm_overlap"

            # logger.info("-" * 72)
            # logger.info(
            #     f"{success_symbol()} Trait={trait} | obs={used_obs} | lines={used_lines} | H2={_fmt_metric(hsqr)} | h2={_fmt_metric(h2_narrow)} | method={narrow_method} | elapsed={format_elapsed(time.time() - step_t0)}"
            # )
            if np.isfinite(h2_blue_raw):
                logger.info(
                    f"  narrow(raw BLUE scale)={_fmt_metric(h2_blue_raw)}"
                )
            if np.isfinite(pvec):
                logger.info(
                    f"  narrow(corrected sparse PVEc)={_fmt_metric(pvec)}"
                )
            if np.isfinite(narrow_mean_diag_k) or np.isfinite(narrow_sigma_g2) or np.isfinite(narrow_sigma_e2):
                logger.info(
                    "  sparse lambda=%s | sigma_g2=%s | sigma_e2=%s | mean_diag(K)=%s | PVEc=%s | nnz(K)=%s | offdiag_density=%s",
                    _fmt_metric(narrow_lambda),
                    _fmt_metric(narrow_sigma_g2),
                    _fmt_metric(narrow_sigma_e2),
                    _fmt_metric(narrow_mean_diag_k),
                    _fmt_metric(pvec),
                    _fmt_metric(narrow_nnz_k),
                    _fmt_metric(narrow_offdiag_density_k),
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
                    "pvec": float(pvec) if np.isfinite(pvec) else np.nan,
                    "h2_blue_raw": float(h2_blue_raw) if np.isfinite(h2_blue_raw) else np.nan,
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
                    "narrow_mean_diag_k": float(narrow_mean_diag_k) if np.isfinite(narrow_mean_diag_k) else np.nan,
                    "narrow_pve_diag_scaled": float(narrow_pve_diag_scaled) if np.isfinite(narrow_pve_diag_scaled) else np.nan,
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
                    "pvec": np.nan,
                    "h2_blue_raw": np.nan,
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
                    "narrow_mean_diag_k": np.nan,
                    "narrow_pve_diag_scaled": np.nan,
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
