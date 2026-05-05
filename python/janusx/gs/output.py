# -*- coding: utf-8 -*-
"""GS workflow output helpers.

Keep formatting/printing logic out of `workflow.py` so core pipeline code stays focused
on data flow and model execution.
"""

from __future__ import annotations

import logging
import math
import typing

SuccessLogger = typing.Callable[..., None]


def _is_gblup_family(method_key: str) -> bool:
    mk = str(method_key).strip().upper()
    return mk.startswith("GBLUP")


def _model_priority_for_tie(method_key: str) -> int:
    # non-GBLUP first, GBLUP-family last
    if _is_gblup_family(method_key):
        return 0
    return 1


def _best_row_idx_by_fold(
    all_rows: list[tuple[str, int, float, float, float, float, float]],
) -> dict[int, int]:
    # Per-fold best rule:
    # 1) Pearsonr (.3f) descending
    # 2) non-GBLUP preferred over GBLUP on ties
    # 3) R2 (raw) descending
    # 4) row order
    best_row_idx: dict[int, int] = {}
    best_pear3_by_fold: dict[int, float] = {}
    best_prio_by_fold: dict[int, int] = {}
    best_r2_by_fold: dict[int, float] = {}
    for idx, row in enumerate(all_rows):
        m_name, fold_id, pear, _, r2, _, _ = row
        fid = int(fold_id)
        pear3 = float(f"{float(pear):.3f}")
        prio = int(_model_priority_for_tie(str(m_name)))
        r2v = float(r2)
        if fid not in best_row_idx:
            best_pear3_by_fold[fid] = pear3
            best_prio_by_fold[fid] = prio
            best_r2_by_fold[fid] = r2v
            best_row_idx[fid] = int(idx)
            continue
        best_pear3 = float(best_pear3_by_fold[fid])
        best_prio = int(best_prio_by_fold[fid])
        best_r2 = float(best_r2_by_fold[fid])
        better = False
        if pear3 > best_pear3:
            better = True
        elif pear3 == best_pear3:
            if prio > best_prio:
                better = True
            elif prio == best_prio and r2v > best_r2:
                better = True
        if better:
            best_pear3_by_fold[fid] = pear3
            best_prio_by_fold[fid] = prio
            best_r2_by_fold[fid] = r2v
            best_row_idx[fid] = int(idx)
    return best_row_idx


def select_cv_best_model(
    *,
    all_rows: list[tuple[str, int, float, float, float, float, float]],
    method_order: dict[str, int],
) -> str | None:
    idx_by_fold = _best_row_idx_by_fold(all_rows)
    if len(idx_by_fold) == 0:
        return None
    wins: dict[str, int] = {}
    for idx in idx_by_fold.values():
        m = str(all_rows[int(idx)][0])
        wins[m] = int(wins.get(m, 0)) + 1
    if len(wins) == 0:
        return None
    # Max wins; ties -> earlier method order.
    best_method = sorted(
        wins.keys(),
        key=lambda m: (-int(wins.get(m, 0)), int(method_order.get(str(m), 10**6)), str(m)),
    )[0]
    return str(best_method)


def emit_trait_header(
    logger: logging.Logger,
    *,
    trait_name: str,
    train_size: int,
    test_size: int,
    eff_snp: int,
    sep_width: int,
    debug_mode: bool = False,
    gblup_backend_label: str | None = None,
    gblup_backend_n: int | None = None,
    gblup_backend_m: int | None = None,
) -> None:
    logger.info("*" * max(1, int(sep_width)))
    logger.info(f"* Genomic Selection for trait: {trait_name}")
    logger.info(
        f"Train size: {int(train_size)}, Test size: {int(test_size)}, EffSNPs: {int(eff_snp)}"
    )
    if (
        bool(debug_mode)
        and gblup_backend_label is not None
        and gblup_backend_n is not None
        and gblup_backend_m is not None
    ):
        logger.info(
            f"GBLUP backend(auto): {gblup_backend_label} "
            f"(n={int(gblup_backend_n)}, m={int(gblup_backend_m)})"
        )


def emit_method_header(logger: logging.Logger, method_display: str) -> None:
    logger.info(f"** {str(method_display)}")


def emit_method_stage_lines(
    logger: logging.Logger,
    *,
    cv_folds: int | None,
    cv_elapsed_sec: float,
    fit_elapsed_sec: float,
    predict_elapsed_sec: float,
    include_fit: bool,
    include_predict: bool = True,
    terminal_columns: int,
    format_elapsed: typing.Callable[[float], str],
    log_success: SuccessLogger,
) -> None:
    if cv_folds is not None and int(cv_folds) > 0:
        total = int(max(1, int(cv_folds)))
        elapsed_text = format_elapsed(float(cv_elapsed_sec) if math.isfinite(float(cv_elapsed_sec)) else 0.0)
        bar_w = max(
            12,
            min(
                46,
                int(terminal_columns)
                - len("Cross-validation ")
                - len(f" {total}/{total} {elapsed_text}")
                - 2,
            ),
        )
        log_success(
            logger,
            f"Cross-validation {'━' * bar_w} {total}/{total} {elapsed_text}",
            force_color=True,
        )

    if include_fit:
        fit_sec = float(fit_elapsed_sec) if math.isfinite(float(fit_elapsed_sec)) else 0.0
        log_success(logger, f"Fitting ...Finished [{fit_sec:.3f}s]", force_color=True)

    if include_predict:
        pred_sec = float(predict_elapsed_sec) if math.isfinite(float(predict_elapsed_sec)) else 0.0
        log_success(logger, f"Predicting ...Finished [{pred_sec:.3f}s]", force_color=True)


def emit_method_detail_lines(
    logger: logging.Logger,
    *,
    cv_mode: str,
    is_gblup: bool,
    gblup_kernel_label: str | None = None,
    gblup_variance_component: str | None = None,
    gblup_backend_label: str | None = None,
    gblup_pve: float | None = None,
) -> None:
    _ = cv_mode

    if is_gblup:
        logger.info(f"kernel              {str(gblup_kernel_label or 'additive')}")
        logger.info(
            f"variance component  {str(gblup_variance_component or 'REML/FaST')}"
        )
        logger.info(f"backend             {str(gblup_backend_label or 'dense')}")
        pve_val = float("nan")
        try:
            if gblup_pve is not None:
                pve_val = float(gblup_pve)
        except Exception:
            pve_val = float("nan")
        if math.isfinite(pve_val):
            logger.info(f"PVE                 {pve_val:.3f}")
        else:
            logger.info("PVE                 NA")


def emit_cv_fold_table(
    logger: logging.Logger,
    *,
    all_rows: list[tuple[str, int, float, float, float, float, float]],
    method_result_map: dict[str, dict[str, typing.Any]],
    method_display_map: dict[str, str],
) -> None:
    del method_result_map
    logger.info("-" * 60)
    best_row_idx_by_fold = _best_row_idx_by_fold(all_rows)

    row_fmt = (
        "{fold:<4} "
        "{method:<10} "
        "{pear:<8} "
        "{spear:<9} "
        "{r2:<6} "
        "{time:<8} "
        "{best}"
    )
    logger.info(
        row_fmt.format(
            fold="Fold",
            method="Method",
            pear="Pearsonr",
            spear="Spearmanr",
            r2="R2",
            time="time(s)",
            best="Best",
        )
    )
    for idx, row in enumerate(all_rows):
        m_name, fold_id, pear, spear, r2, _pve, t_fold = row
        is_best = int(best_row_idx_by_fold.get(int(fold_id), -1)) == int(idx)
        logger.info(
            row_fmt.format(
                fold=int(fold_id),
                method=method_display_map.get(str(m_name), str(m_name)),
                pear=f"{float(pear):.3f}",
                spear=f"{float(spear):.3f}",
                r2=f"{float(r2):.3f}",
                time=f"{float(t_fold):.3f}",
                best=("*" if is_best else ""),
            )
        )
