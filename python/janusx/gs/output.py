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
    gblup_backend_label: str | None = None,
) -> None:
    _ = cv_mode

    if is_gblup:
        logger.info(f"kernel              {str(gblup_kernel_label or 'additive')}")
        logger.info("variance component  REML/FaST")
        logger.info(f"backend             {str(gblup_backend_label or 'dense')}")


def emit_cv_fold_table(
    logger: logging.Logger,
    *,
    all_rows: list[tuple[str, int, float, float, float, float, float]],
    method_result_map: dict[str, dict[str, typing.Any]],
    method_display_map: dict[str, str],
) -> None:
    logger.info("-" * 60)
    row_fmt = (
        "{fold:<4} "
        "{method:<10} "
        "{pear:<8} "
        "{spear:<9} "
        "{r2:<6} "
        "{h2:<6} "
        "{time:<10}"
    )
    logger.info(
        row_fmt.format(
            fold="Fold",
            method="Method",
            pear="Pearsonr",
            spear="Spearmanr",
            r2="R2",
            h2="h2/PVE",
            time="time(secs)",
        )
    )
    for row in all_rows:
        m_name, fold_id, pear, spear, r2, pve, t_fold = row
        logger.info(
            row_fmt.format(
                fold=int(fold_id),
                method=method_display_map.get(str(m_name), str(m_name)),
                pear=f"{float(pear):.3f}",
                spear=f"{float(spear):.3f}",
                r2=f"{float(r2):.3f}",
                h2=f"{float(pve):.3f}",
                time=f"{float(t_fold):.3f}",
            )
        )
