# -*- coding: utf-8 -*-
"""GS workflow output helpers.

Keep formatting/printing logic out of `workflow.py` so core pipeline code stays focused
on data flow and model execution.
"""

from __future__ import annotations

import logging
import math
import typing

from janusx.script._common.progress import _emit_to_file_handlers, stdout_is_tty
from .progress import format_completion

SuccessLogger = typing.Callable[..., None]
FoldRow = tuple[str, int, float, float, float, float, float, float, float]


def _is_gblup_family(method_key: str) -> bool:
    mk = str(method_key).strip().upper()
    return mk.startswith("GBLUP")


def _model_priority_for_tie(method_key: str) -> int:
    # non-GBLUP first, GBLUP-family last
    if _is_gblup_family(method_key):
        return 0
    return 1


def _unpack_fold_row(row: tuple[typing.Any, ...]) -> FoldRow:
    if len(row) < 7:
        raise ValueError(f"fold row has too few fields: {len(row)}")
    mse = float(row[7]) if len(row) > 7 else float("nan")
    mae = float(row[8]) if len(row) > 8 else float("nan")
    return (
        str(row[0]),
        int(row[1]),
        float(row[2]),
        float(row[3]),
        float(row[4]),
        float(row[5]),
        float(row[6]),
        mse,
        mae,
    )


def _fmt_num(value: float, *, digits: int = 3, na: str = "NA") -> str:
    try:
        v = float(value)
    except Exception:
        return na
    if not math.isfinite(v):
        return na
    return f"{v:.{digits}f}"


def _best_row_idx_by_fold(
    all_rows: list[tuple[typing.Any, ...]],
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
        m_name, fold_id, pear, _, r2, _, _, _, _ = _unpack_fold_row(row)
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
    all_rows: list[tuple[typing.Any, ...]],
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
    tty_style: bool = False,
) -> None:
    if tty_style:
        width = max(40, int(sep_width))
        logger.info("")
        logger.info("=" * width)
        logger.info("GS task")
        logger.info("=" * width)
        logger.info(
            f"* {trait_name} (train={int(train_size)}; test={int(test_size)}; "
            f"EffSNPs={int(eff_snp)})"
        )
        logger.info("-" * width)
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
        return

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


def emit_method_header(
    logger: logging.Logger,
    method_display: str,
    *,
    terminal_logger: logging.Logger | None = None,
) -> None:
    method = str(method_display)
    if terminal_logger is None:
        logger.info(f"** {method}")
        return
    _emit_to_file_handlers(logger, logging.INFO, f"** {method}")
    terminal_logger.info(f"{method}:")


def emit_method_stage_lines(
    logger: logging.Logger,
    *,
    cv_folds: int | None,
    cv_elapsed_sec: float,
    cv_skipped: bool,
    fit_elapsed_sec: float,
    fit_skipped: bool,
    predict_elapsed_sec: float,
    predict_skipped: bool,
    total_elapsed_sec: float | None,
    include_fit: bool,
    include_predict: bool = True,
    terminal_columns: int,
    format_elapsed: typing.Callable[[float], str],
    log_success: SuccessLogger,
    cv_terminal_emitted: bool = False,
    stages_terminal_emitted: bool = False,
) -> None:
    def emit(message: str, *, terminal_message: str | None = None, file_only=False):
        formatted_message = format_completion(message)
        if file_only:
            _emit_to_file_handlers(logger, logging.INFO, formatted_message)
        else:
            log_success(
                logger,
                (
                    terminal_message
                    if stdout_is_tty() and terminal_message is not None
                    else message
                ),
                log_message=formatted_message,
                force_color=stdout_is_tty(),
            )

    if cv_skipped:
        total_sec = float(total_elapsed_sec) if total_elapsed_sec is not None else 0.0
        if not math.isfinite(total_sec):
            total_sec = 0.0
        elapsed_text = format_elapsed(total_sec)
        emit(
            f"CV ...Skipped [{elapsed_text}]",
            terminal_message=f"Cross-validation ...Skipped [{elapsed_text}]",
            file_only=cv_terminal_emitted,
        )
    elif cv_folds is not None and int(cv_folds) > 0:
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
        emit(
            f"CV ...Finished [{total}/{total} folds | {elapsed_text}]",
            terminal_message=f"Cross-validation {'━' * bar_w} {total}/{total} {elapsed_text}",
            file_only=cv_terminal_emitted,
        )

    if include_fit:
        if fit_skipped:
            emit(
                f"Fit ...Skipped [{format_elapsed(0.0)}]",
                terminal_message=f"Fitting ...Skipped [{format_elapsed(0.0)}]",
                file_only=stages_terminal_emitted,
            )
        else:
            fit_sec = float(fit_elapsed_sec) if math.isfinite(float(fit_elapsed_sec)) else 0.0
            emit(
                f"Fit ...Finished [{format_elapsed(fit_sec)}]",
                terminal_message=f"Fitting ...Finished [{fit_sec:.3f}s]",
                file_only=stages_terminal_emitted,
            )

    if include_predict:
        if predict_skipped:
            emit(
                f"Predict ...Skipped [{format_elapsed(0.0)}]",
                terminal_message=f"Predicting ...Skipped [{format_elapsed(0.0)}]",
                file_only=stages_terminal_emitted,
            )
        else:
            pred_sec = float(predict_elapsed_sec) if math.isfinite(float(predict_elapsed_sec)) else 0.0
            emit(
                f"Predict ...Finished [{format_elapsed(pred_sec)}]",
                terminal_message=f"Predicting ...Finished [{pred_sec:.3f}s]",
                file_only=stages_terminal_emitted,
            )


def emit_method_detail_lines(
    logger: logging.Logger,
    *,
    cv_mode: str,
    is_gblup: bool,
    detail_rows: typing.Sequence[tuple[str, str]] | None = None,
    gblup_kernel_label: str | None = None,
    gblup_variance_component: str | None = None,
    gblup_backend_label: str | None = None,
    gblup_pve: float | None = None,
) -> None:
    _ = cv_mode

    rows = list(detail_rows or [])
    if len(rows) > 0:
        key_w = max(8, max(len(str(k)) for k, _ in rows))
        for k, v in rows:
            logger.info(f"{str(k):<{key_w}}  {str(v)}")
        return

    if is_gblup:
        logger.info(f"kernel              {str(gblup_kernel_label or 'additive')}")
        logger.info(f"vc                  {str(gblup_variance_component or 'GRMreml')}")
        pve_val = float("nan")
        try:
            if gblup_pve is not None:
                pve_val = float(gblup_pve)
        except Exception:
            pve_val = float("nan")
        if math.isfinite(pve_val):
            logger.info(f"PVE                 {pve_val:.3f}")


def emit_cv_fold_table(
    logger: logging.Logger,
    *,
    all_rows: list[tuple[typing.Any, ...]],
    method_result_map: dict[str, dict[str, typing.Any]],
    method_display_map: dict[str, str],
    extended_metrics: bool = False,
    section_title: str | None = None,
) -> None:
    del method_result_map
    if section_title is not None and str(section_title).strip() != "":
        logger.info(str(section_title))
    logger.info("-" * (92 if extended_metrics else 60))
    best_row_idx_by_fold = _best_row_idx_by_fold(all_rows)

    method_w = max(
        10,
        min(
            18,
            max((len(str(method_display_map.get(str(row[0]), str(row[0])))) for row in all_rows), default=10),
        ),
    )
    if extended_metrics:
        row_fmt = (
            "{fold:<4} "
            f"{{method:<{method_w}}} "
            "{pear:<8} "
            "{spear:<9} "
            "{r2:<6} "
            "{mse:<10} "
            "{mae:<10} "
            "{pve:<8} "
            "{time:<8} "
            "{best}"
        )
    else:
        row_fmt = (
            "{fold:<4} "
            f"{{method:<{method_w}}} "
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
            mse=("MSE" if extended_metrics else ""),
            mae=("MAE" if extended_metrics else ""),
            pve=("PVE(ph)" if extended_metrics else ""),
            time="time(s)",
            best="Best",
        )
    )
    for idx, row in enumerate(all_rows):
        m_name, fold_id, pear, spear, r2, pve, t_fold, mse, mae = _unpack_fold_row(row)
        is_best = int(best_row_idx_by_fold.get(int(fold_id), -1)) == int(idx)
        row_data = dict(
            fold=int(fold_id),
            method=method_display_map.get(str(m_name), str(m_name)),
            pear=_fmt_num(float(pear), digits=3),
            spear=_fmt_num(float(spear), digits=3),
            r2=_fmt_num(float(r2), digits=3),
            time=_fmt_num(float(t_fold), digits=3),
            best=("*" if is_best else ""),
            mse=_fmt_num(float(mse), digits=6),
            mae=_fmt_num(float(mae), digits=6),
            pve=_fmt_num(float(pve), digits=3),
        )
        logger.info(row_fmt.format(**row_data))
