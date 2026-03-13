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
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from scipy.stats import t as student_t

from janusx.pyBLUP.blup import BLUP
from ._common.log import setup_logging
from ._common.config_render import emit_cli_configuration
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.pathcheck import ensure_file_exists
from ._common.genoio import strip_default_prefix_suffix
from ._common.status import CliStatus, print_success, print_failure, format_elapsed


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
        # Support numeric slice syntax like "1:3" (inclusive, 1-based).
        if ":" in t:
            left, right = t.split(":", 1)
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
) -> tuple[np.ndarray, list[str]]:
    s = df_sub[term.name]
    if (not term.force_onehot) and _is_numeric_series(s):
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
        return arr, [term.name]

    # Default rule: string/categorical columns are one-hot encoded.
    ss = s.astype("string").fillna("NA").astype(str)
    dummies = pd.get_dummies(
        ss,
        prefix=term.name,
        prefix_sep="=",
        drop_first=(not for_random),
        dtype=float,
    )
    if dummies.shape[1] == 0:
        return np.zeros((len(df_sub), 0), dtype=float), []
    return dummies.to_numpy(dtype=float), [str(c) for c in dummies.columns]


def _gls_fixed_stats_from_blup(
    model: BLUP,
    z_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta = np.asarray(model.beta, dtype=float).reshape(-1)
    n, p = int(model.X.shape[0]), int(model.X.shape[1])

    if model.theta is None or len(z_list) == 0:
        resid = np.asarray(model.residuals, dtype=float).reshape(-1, 1)
        sigma2 = float((resid.T @ resid)[0, 0]) / max(1, n - p)
        xtx = np.asarray(model.X.T @ model.X, dtype=float)
        cov_beta = np.linalg.pinv(xtx) * sigma2
    else:
        theta = np.asarray(model.theta, dtype=float).reshape(-1)
        if theta.size < (len(z_list) + 1):
            raise ValueError("BLUP theta size is inconsistent with random effects.")

        # Reconstruct the same K-list definition used in BLUP._fit()
        k_list: list[np.ndarray] = []
        for i, z in enumerate(z_list):
            q, mean_vec, std_vec = model.onehot_info[i]
            z_std = (
                (np.asarray(z, dtype=float) - np.asarray(mean_vec, dtype=float))
                / np.asarray(std_vec, dtype=float)
                / np.sqrt(float(q))
            )
            k_list.append(np.asarray(z_std @ z_std.T, dtype=float))

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
            "comma list (e.g. 1,3), or numeric range (e.g. 1:3). "
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

    with CliStatus("Loading input table...", enabled=bool(sys.stdout.isatty())) as st:
        try:
            df = pd.read_csv(str(args.file), sep=None, engine="python")
        except Exception:
            st.fail("Loading input table ...Failed")
            raise
        st.complete("Loading input table ...Finished")

    if df.shape[1] < 2:
        raise ValueError("Input file must contain at least 2 columns: sample_id + data columns.")

    sample_col = str(df.columns[0])
    sample_ids = df.iloc[:, 0].astype(str).str.strip()
    if sample_ids.duplicated().any():
        dup = sample_ids[sample_ids.duplicated()].unique().tolist()[:8]
        msg = (
            "Duplicated sample IDs detected in first column; keep all rows as independent records. "
            f"Examples: {', '.join(map(str, dup))}"
        )
        logger.warning(msg)
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
                    ("Sample column", sample_col),
                    ("Samples", int(df.shape[0])),
                ],
            ),
            (
                "Columns",
                [
                    ("Phenotypes", ", ".join(n_cols)),
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
                    ("Log file", log_path),
                ],
            ),
        ],
    )

    blup_out = pd.DataFrame({sample_col: sample_ids.to_numpy(dtype=object)})

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
            continue

        sub = dfx.loc[mask, :].copy()
        y = pd.to_numeric(sub[trait], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)

        x_blocks: list[np.ndarray] = []
        x_names: list[str] = []
        for term in fixed_terms:
            arr, names = _encode_term_matrix(sub, term, for_random=False)
            if arr.shape[1] == 0:
                logger.warning(f"Trait {trait}: fixed term `{term.name}` expanded to 0 columns; skipped.")
                continue
            x_blocks.append(arr)
            x_names.extend(names)
        x_mat = np.concatenate(x_blocks, axis=1) if len(x_blocks) > 0 else None

        z_list: list[np.ndarray] = []
        z_names: list[str] = []
        for term in random_terms:
            arr, _ = _encode_term_matrix(sub, term, for_random=True)
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
        y_hat = (
            model.predict(
                X=x_mat,
                Z=z_list if len(z_list) > 0 else None,
            )
            .reshape(-1)
            .astype(float)
        )

        pred_col = np.full(df.shape[0], np.nan, dtype=float)
        pred_col[np.asarray(mask, dtype=bool)] = y_hat
        blup_out[trait] = pred_col

        # Variance decomposition and heritability
        h2 = 0.0
        random_rows: list[tuple[str, float, float]] = []
        if getattr(model, "var", None) is not None and len(z_names) > 0:
            var_all = np.asarray(model.var, dtype=float).reshape(-1)
            if var_all.size >= (len(z_names) + 1):
                rand_var = var_all[: len(z_names)]
                ve = float(var_all[-1])
                total = float(np.sum(rand_var) + ve)
                if total > 0.0:
                    pve = rand_var / total
                    h2 = float(np.sum(pve))
                    random_rows = [
                        (z_names[i], float(rand_var[i]), float(pve[i])) for i in range(len(z_names))
                    ]

        # Fixed effects stats
        beta, se, pval = _gls_fixed_stats_from_blup(model, z_list)
        fx_names = ["Intercept"] + x_names

        logger.info("-" * 60)
        logger.info(
            f"Trait: {trait} | used={n_used}/{df.shape[0]} | H2={h2:.6f} | elapsed={format_elapsed(time.time()-step_t0)}"
        )
        logger.info("Fixed effects (beta, se, pvalue):")
        for i, nm in enumerate(fx_names):
            logger.info(
                f"  {nm}\tbeta={beta[i]:.6g}\tse={se[i]:.6g}\tp={pval[i]:.6g}"
            )
        if len(random_rows) > 0:
            logger.info("Random effects (var, PVE):")
            for nm, vv, pp in random_rows:
                logger.info(f"  {nm}\tvar={vv:.6g}\tPVE={pp:.6g}")
        else:
            logger.info("Random effects (var, PVE): None")

    out_blup = f"{outprefix}.blup.txt"
    blup_out.to_csv(out_blup, sep="\t", index=False)
    logger.info("=" * 60)
    logger.info(f"BLUP table saved: {out_blup}")
    logger.info(f"Total elapsed: {format_elapsed(time.time() - t0)}")
    print_success(f"REML ...Finished [{format_elapsed(time.time() - t0)}]")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print_failure("REML ...Failed")
        print(f"Error: {exc}")
        raise
