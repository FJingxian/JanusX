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
  jx reml -file test.reml.txt -rh 1 -rh 2 -rh 3 -n 4 -o test -prefix reml
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
from scipy.stats import t as student_t

from janusx.pyBLUP.blup import BLUP
from ._common.log import setup_logging
from ._common.config_render import emit_cli_configuration
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.pathcheck import ensure_file_exists
from ._common.genoio import strip_default_prefix_suffix
from ._common.status import print_success, print_failure, format_elapsed


@dataclass
class _TermSpec:
    name: str
    force_onehot: bool


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


def _resolve_columns(tokens: list[str], data_cols: list[str], label: str) -> list[str]:
    if len(tokens) == 0:
        return []

    lower_map = {str(c).lower(): str(c) for c in data_cols}
    resolved: list[str] = []
    for tk in tokens:
        t = str(tk).strip()
        if t == "":
            continue
        # Support numeric range syntax like "1:3" / "1-3" (inclusive, 1-based).
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
                    if idx < 1 or idx > len(data_cols):
                        raise ValueError(
                            f"{label} column index out of range: {idx}. valid=[1..{len(data_cols)}]"
                        )
                    resolved.append(str(data_cols[idx - 1]))
                continue
        if t.lstrip("+-").isdigit():
            idx = int(t)
            if idx < 1 or idx > len(data_cols):
                raise ValueError(
                    f"{label} column index out of range: {idx}. valid=[1..{len(data_cols)}]"
                )
            resolved.append(str(data_cols[idx - 1]))
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


def _effective_env_rep_counts(
    sample_ids_sub: pd.Series,
    sub: pd.DataFrame,
    env_cols: list[str],
    rep_cols: list[str],
) -> tuple[float, float]:
    sid = sample_ids_sub.astype("string").fillna("NA").astype(str)
    env_key = _combine_key(sub, env_cols, "__ENV__").astype(str)
    ge = pd.DataFrame({"sid": sid, "env": env_key})
    e_per_sid = ge.drop_duplicates().groupby("sid", sort=False)["env"].nunique()
    e_eff = float(e_per_sid.mean()) if len(e_per_sid) > 0 else 1.0
    e_eff = max(1.0, e_eff)

    if len(rep_cols) > 0:
        rep_key = _combine_key(sub, rep_cols, "__REP__").astype(str)
        ger = pd.DataFrame({"sid": sid, "env": env_key, "rep": rep_key})
        r_per_ge = ger.drop_duplicates().groupby(["sid", "env"], sort=False)["rep"].nunique()
    else:
        r_per_ge = ge.groupby(["sid", "env"], sort=False).size()
    r_eff = float(r_per_ge.mean()) if len(r_per_ge) > 0 else 1.0
    r_eff = max(1.0, r_eff)
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


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx reml",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx reml -file test.reml.txt -rh 1 -rh 2 -rh 3 -n 4 -o test -prefix reml",
                "jx reml -file test.reml.txt -f sex -rh family -n trait1 -n trait2",
            ]
        ),
    )

    req = parser.add_argument_group("Required Arguments")
    req.add_argument(
        "-file",
        "--file",
        required=True,
        type=str,
        help="Input phenotype/effect table. First column is sample ID.",
    )
    req.add_argument(
        "-n",
        "--n",
        action="append",
        required=True,
        metavar="COL",
        help=(
            "Phenotype column(s), 1-based index (excluding sample ID), name, "
            "comma list (e.g. 1,3), or numeric range (e.g. 1:3 or 1-3). "
            "Repeat this flag for multiple traits."
        ),
    )

    opt = parser.add_argument_group("Optional Arguments")
    opt.add_argument(
        "-rh",
        "--rh",
        action="append",
        default=[],
        metavar="COL",
        help="Random-effect categorical column(s), one-hot encoded.",
    )
    opt.add_argument(
        "-fh",
        "--fh",
        action="append",
        default=[],
        metavar="COL",
        help="Fixed-effect categorical column(s), one-hot encoded.",
    )
    opt.add_argument(
        "-r",
        "--r",
        action="append",
        default=[],
        metavar="COL",
        help="Random-effect column(s). String columns are one-hot encoded by default.",
    )
    opt.add_argument(
        "-f",
        "--f",
        action="append",
        default=[],
        metavar="COL",
        help="Fixed-effect column(s). String columns are one-hot encoded by default.",
    )
    opt.add_argument(
        "-o",
        "--out",
        type=str,
        default=".",
        help="Output directory (default: current directory).",
    )
    opt.add_argument(
        "-prefix",
        "--prefix",
        type=str,
        default=None,
        help="Output prefix (default: inferred from input file name).",
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
    outdir = os.path.normpath(str(args.out or "."))
    os.makedirs(outdir, mode=0o755, exist_ok=True)
    prefix = str(args.prefix or strip_default_prefix_suffix(os.path.basename(str(args.file))))
    outprefix = os.path.join(outdir, prefix).replace("\\", "/")
    log_path = f"{outprefix}.reml.log"
    logger = setup_logging(log_path)

    if not ensure_file_exists(logger, str(args.file), "Input file"):
        return

    load_t0 = time.time()
    try:
        df = pd.read_csv(str(args.file), sep=None, engine="python")
    except Exception:
        raise
    load_elapsed = format_elapsed(time.time() - load_t0)

    if df.shape[1] < 2:
        raise ValueError("Input file must contain at least 2 columns: sample_id + data columns.")

    sample_col = str(df.columns[0])
    sample_ids = df.iloc[:, 0].astype(str).str.strip()
    n_samples_total = int(sample_ids.shape[0])
    n_samples_unique = int(sample_ids.nunique(dropna=False))
    df = df.copy()
    df[sample_col] = sample_ids
    dfx = df.iloc[:, 1:].copy()
    data_cols = [str(c) for c in dfx.columns]

    n_cols = _resolve_columns(_split_tokens(args.n), data_cols, "-n")
    rh_cols = _resolve_columns(_split_tokens(args.rh), data_cols, "-rh")
    fh_cols = _resolve_columns(_split_tokens(args.fh), data_cols, "-fh")
    r_cols = _resolve_columns(_split_tokens(args.r), data_cols, "-r")
    f_cols = _resolve_columns(_split_tokens(args.f), data_cols, "-f")

    fixed_map: "OrderedDict[str, bool]" = OrderedDict()
    for c in fh_cols:
        fixed_map[c] = True
    for c in f_cols:
        fixed_map[c] = bool(fixed_map.get(c, False))
    fixed_terms = [_TermSpec(name=k, force_onehot=v) for k, v in fixed_map.items()]

    random_map: "OrderedDict[str, bool]" = OrderedDict()
    for c in rh_cols:
        random_map[c] = True
    for c in r_cols:
        random_map[c] = bool(random_map.get(c, False))
    random_terms = [_TermSpec(name=k, force_onehot=v) for k, v in random_map.items()]
    env_cols_auto, rep_cols_auto = _infer_env_rep_columns(random_terms)

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
                    ("Load table", f"Finished [{load_elapsed}]"),
                    ("Sample column", sample_col),
                    ("Samples(total)", n_samples_total),
                    ("Samples(unique)", n_samples_unique),
                ],
            ),
            (
                "Columns",
                [
                    ("Phenotypes", ", ".join(n_cols)),
                    ("ID random effect", f"{sample_col} (auto one-hot)"),
                    ("Auto env columns", ", ".join(env_cols_auto) if env_cols_auto else "None"),
                    ("Auto rep columns", ", ".join(rep_cols_auto) if rep_cols_auto else "None"),
                    ("Fixed onehot", ", ".join(fh_cols) if fh_cols else "None"),
                    ("Fixed", ", ".join(f_cols) if f_cols else "None"),
                    ("Random onehot", ", ".join(rh_cols) if rh_cols else "None"),
                    ("Random", ", ".join(r_cols) if r_cols else "None"),
                ],
            ),
            (
                "Output",
                [
                    ("Out dir", outdir),
                    ("Prefix", prefix),
                    ("BLUP file", f"{outprefix}.blup.txt"),
                    ("Summary file", f"{outprefix}.reml.summary.tsv"),
                    ("Log file", log_path),
                ],
            ),
        ],
    )

    unique_sample_ids = sample_ids.drop_duplicates().reset_index(drop=True)
    blup_out = pd.DataFrame({sample_col: unique_sample_ids.to_numpy(dtype=object)})
    summary_rows: list[dict[str, typing.Any]] = []

    for trait in n_cols:
        step_t0 = time.time()
        y_all = pd.to_numeric(dfx[trait], errors="coerce")
        mask = y_all.notna()
        mask &= _collect_numeric_required_mask(dfx, fixed_terms)
        mask &= _collect_numeric_required_mask(dfx, random_terms)

        n_used = int(mask.sum())
        if n_used <= 2:
            logger.warning(f"Trait {trait}: too few valid samples after filtering ({n_used}); skip.")
            blup_out[trait] = np.nan
            summary_rows.append(
                {
                    "trait": trait,
                    "used": n_used,
                    "total": int(df.shape[0]),
                    "e": np.nan,
                    "r": np.nan,
                    "hsqr": np.nan,
                    "ve": np.nan,
                    "pve_e": np.nan,
                    "elapsed_sec": float(time.time() - step_t0),
                    "status": "skipped_too_few_samples",
                }
            )
            continue

        sub = dfx.loc[mask, :].copy()
        y = pd.to_numeric(sub[trait], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)

        x_blocks: list[np.ndarray] = []
        x_names: list[str] = []
        for term in fixed_terms:
            arr, names = _encode_term_matrix(sub, term, for_random=False, sparse_onehot=False)
            if arr.shape[1] == 0:
                logger.warning(f"Trait {trait}: fixed term `{term.name}` expanded to 0 columns; skipped.")
                continue
            x_blocks.append(np.asarray(arr, dtype=float))
            x_names.extend(names)
        x_mat = np.concatenate(x_blocks, axis=1) if len(x_blocks) > 0 else None

        z_list: list[typing.Union[np.ndarray, sparse.spmatrix]] = []
        z_names: list[str] = []
        sample_ids_sub = df.loc[mask, sample_col].astype("string").fillna("NA").astype(str)
        # Always include sample-ID random effect from the first column.
        id_series = sample_ids_sub
        id_dummies, id_col_names = _onehot_encode_series(
            id_series,
            prefix=str(sample_col),
            drop_first=False,
            sparse_output=True,
        )
        if id_dummies.shape[1] > 0:
            z_list.append(id_dummies)
            z_names.append(sample_col)
        else:
            logger.warning(
                f"Trait {trait}: sample-ID random effect expanded to 0 columns; H2 will be unavailable."
            )

        gxe_name = f"{sample_col}xenv"
        if len(env_cols_auto) > 0:
            env_key = _combine_key(sub, [c for c in env_cols_auto if c in sub.columns], "__ENV__").astype(str)
            gxe_key = (sample_ids_sub + "@@" + env_key).astype("string")
            gxe_dummies, _ = _onehot_encode_series(
                gxe_key,
                prefix=gxe_name,
                drop_first=False,
                sparse_output=True,
            )
            if gxe_dummies.shape[1] > 0:
                z_list.append(gxe_dummies)
                z_names.append(gxe_name)
            else:
                logger.warning(
                    f"Trait {trait}: sample×env random effect expanded to 0 columns; use fallback H2."
                )

        for term in random_terms:
            arr, _ = _encode_term_matrix(sub, term, for_random=True, sparse_onehot=True)
            if arr.shape[1] == 0:
                logger.warning(f"Trait {trait}: random term `{term.name}` expanded to 0 columns; skipped.")
                continue
            z_list.append(arr)
            z_names.append(term.name)

        model = BLUP(
            y=y,
            X=x_mat,
            Z=z_list if len(z_list) > 0 else None,
            maxiter=int(args.maxiter),
            progress=False,
        )
        # Export sample-level BLUP (ID random effect), one row per unique sample.
        id_u = np.full(blup_out.shape[0], np.nan, dtype=float)
        if getattr(model, "u_by_Z", None) is not None and len(model.u_by_Z) > 0:
            u_id = np.asarray(model.u_by_Z[0], dtype=float).reshape(-1)
            id_cols = [str(c) for c in id_col_names]
            id_prefix = f"{sample_col}-"
            id_levels = [
                c[len(id_prefix):] if c.startswith(id_prefix) else c
                for c in id_cols
            ]
            id_map = {str(k): float(v) for k, v in zip(id_levels, u_id)}
            id_u = (
                blup_out[sample_col]
                .astype(str)
                .map(id_map)
                .to_numpy(dtype=float)
            )
        blup_out[trait] = id_u

        # Variance decomposition and heritability
        h2 = np.nan
        ve = np.nan
        pve_e = np.nan
        e_eff = 1.0
        r_eff = 1.0
        random_rows: list[tuple[str, float, float]] = []
        if getattr(model, "var", None) is not None and len(z_names) > 0:
            var_all = np.asarray(model.var, dtype=float).reshape(-1)
            if var_all.size >= (len(z_names) + 1):
                rand_var = var_all[: len(z_names)]
                ve = float(var_all[-1])
                total = float(np.sum(rand_var) + ve)
                if total > 0.0:
                    pve = rand_var / total
                    # Broad-sense heritability on mean scale:
                    # H2 = Vg / (Vg + Vge/e + Ve/(e*r))
                    id_idx = z_names.index(sample_col) if sample_col in z_names else -1
                    gxe_idx = z_names.index(gxe_name) if gxe_name in z_names else -1
                    vg = float(rand_var[id_idx]) if id_idx >= 0 else np.nan
                    vge = float(rand_var[gxe_idx]) if gxe_idx >= 0 else 0.0
                    e_eff, r_eff = _effective_env_rep_counts(
                        sample_ids_sub,
                        sub,
                        [c for c in env_cols_auto if c in sub.columns],
                        [c for c in rep_cols_auto if c in sub.columns],
                    )
                    denom = vg + (vge / e_eff) + (ve / (e_eff * r_eff))
                    if np.isfinite(denom) and denom > 0.0 and np.isfinite(vg):
                        h2 = float(vg / denom)
                    else:
                        # fallback to observation-scale id-PVE when mean-scale denominator is unstable
                        h2 = float(pve[id_idx]) if id_idx >= 0 else np.nan
                    pve_e = float(ve / total)
                    random_rows = [
                        (z_names[i], float(rand_var[i]), float(pve[i])) for i in range(len(z_names))
                    ]

        # Fixed effects stats
        beta, se, pval = _gls_fixed_stats_from_blup(model, z_list)
        fx_names = ["Intercept"] + x_names

        logger.info("-" * 60)
        logger.info(
            f"Trait: {trait} | used={n_used}/{df.shape[0]} | e={e_eff:.2f} | r={r_eff:.2f} | elapsed={format_elapsed(time.time()-step_t0)}"
        )
        logger.info(f"hsqr={h2:.6g}" if np.isfinite(h2) else "hsqr=NA")
        logger.info("Fixed effects:")
        for i, nm in enumerate(fx_names):
            logger.info(
                f"  {nm:<12} beta={beta[i]:<12.6g} se={se[i]:<12.6g} p={pval[i]:<12.6g}"
            )
        if len(random_rows) > 0:
            logger.info("Random effects:")
            for nm, vv, pp in random_rows:
                logger.info(
                    f"  {nm:<12} var={vv:<12.6g} PVE={pp:<12.6g}"
                )
            logger.info(
                f"  {'Residual':<12} var={ve:<12.6g} PVE={pve_e:<12.6g}"
            )
        else:
            logger.info("Random effects: None")
        summary_rows.append(
            {
                "trait": trait,
                "used": n_used,
                "total": int(df.shape[0]),
                "e": float(e_eff),
                "r": float(r_eff),
                "hsqr": float(h2) if np.isfinite(h2) else np.nan,
                "ve": float(ve) if np.isfinite(ve) else np.nan,
                "pve_e": float(pve_e) if np.isfinite(pve_e) else np.nan,
                "elapsed_sec": float(time.time() - step_t0),
                "status": "ok",
            }
        )

    out_blup = f"{outprefix}.blup.txt"
    out_summary = f"{outprefix}.reml.summary.tsv"
    blup_out.to_csv(out_blup, sep="\t", index=False)
    pd.DataFrame(summary_rows).to_csv(out_summary, sep="\t", index=False)
    logger.info("=" * 60)
    logger.info(f"BLUP table saved: {out_blup}")
    logger.info(f"Summary table saved: {out_summary}")
    logger.info(f"Total elapsed: {format_elapsed(time.time() - t0)}")
    print_success(f"REML ...Finished [{format_elapsed(time.time() - t0)}]")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print_failure("REML ...Failed")
        print(f"Error: {exc}")
        raise
