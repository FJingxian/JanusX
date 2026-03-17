# -*- coding: utf-8 -*-
"""
JanusX: ADAMIXTURE ancestry inference (Rust-kernel backend)

Examples
--------
  jx adamixture -bfile data/geno -k 8 -o out -prefix cohort
  jx adamixture -vcf data/geno.vcf.gz -k 6 -t 16
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from janusx.adamixture import ADAMixtureConfig, train_adamixture
from janusx.gfreader import inspect_genotype_file
from ._common.config_render import emit_cli_configuration
from ._common.genoio import determine_genotype_source_from_args, strip_default_prefix_suffix
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
)
from ._common.status import CliStatus, log_success, print_failure, print_success, stdout_is_tty
from ._common.threads import detect_effective_threads


def _mute_stdout_info_logs(
    logger: logging.Logger,
) -> list[tuple[logging.Handler, int]]:
    muted: list[tuple[logging.Handler, int]] = []
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout:
            muted.append((handler, handler.level))
            handler.setLevel(logging.WARNING)
    return muted


def _restore_handler_levels(muted: list[tuple[logging.Handler, int]]) -> None:
    for handler, level in muted:
        handler.setLevel(level)


class _AdmixtureCliProgress:
    def __init__(self, logger: logging.Logger, *, enabled: bool) -> None:
        self.logger = logger
        self.enabled = bool(enabled)
        self._status: Optional[CliStatus] = None
        self._adam_max_iter = 0

    def _start(self, desc: str) -> None:
        self.close()
        self._status = CliStatus(str(desc), enabled=self.enabled)
        self._status.__enter__()

    def _update_desc(self, desc: str) -> None:
        if self._status is not None:
            self._status.desc = str(desc)

    def _complete(self, message: str) -> None:
        if self._status is not None:
            self._status.complete(str(message))
            self._status = None

    def fail(self, message: str) -> None:
        if self._status is not None:
            self._status.fail(str(message))
            self._status = None

    def close(self) -> None:
        if self._status is not None:
            self._status.__exit__(None, None, None)
            self._status = None

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        e = str(event)
        if e == "data_loaded":
            m = int(payload.get("m", 0))
            n = int(payload.get("n", 0))
            if self.enabled:
                print_success(
                    f"Loaded genotype matrix (SNPs={m}, samples={n})",
                    force_color=True,
                )
                print(f"Data shape: SNPs={m}, samples={n}", flush=True)
            return

        if e == "rsvd_start":
            self._start("RSVD")
            return
        if e == "rsvd_done":
            self._complete("RSVD ...Finished")
            return

        if e == "als_start":
            self._start("ALS initialization")
            return
        if e == "als_done":
            self._complete("ALS initialization ...Finished")
            return

        if e == "adam_start":
            self._adam_max_iter = int(payload.get("max_iter", 0))
            self._start(f"ADAM-EM iter 0/{self._adam_max_iter}")
            return
        if e == "adam_iter":
            it = int(payload.get("iteration", 0))
            ll = float(payload.get("ll", float("nan")))
            lr = float(payload.get("lr", float("nan")))
            if np.isfinite(ll) and np.isfinite(lr):
                self._update_desc(
                    f"ADAM-EM iter {it}/{self._adam_max_iter} "
                    f"ll={ll:.3f} lr={lr:.3e}"
                )
            return
        if e == "adam_done":
            self._complete("ADAM-EM optimization ...Finished")
            return


def _resolve_input(args, logger) -> tuple[str, str, str]:
    gfile, auto_prefix = determine_genotype_source_from_args(args)

    checks = []
    if args.vcf:
        checks.append(ensure_file_exists(logger, gfile, "VCF file"))
        source_label = "VCF"
    elif args.hmp:
        checks.append(ensure_file_exists(logger, gfile, "HMP file"))
        source_label = "HMP"
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, label="FILE genotype input"))
        source_label = "FILE"
    elif args.bfile:
        bprefix = gfile
        if str(gfile).lower().endswith(".bed"):
            bprefix = str(Path(gfile).with_suffix(""))
        checks.append(ensure_plink_prefix_exists(logger, bprefix, label="PLINK prefix"))
        gfile = bprefix
        auto_prefix = os.path.basename(bprefix.rstrip("/\\"))
        source_label = "BFILE"
    else:
        raise ValueError("One genotype input is required: -vcf / -hmp / -file / -bfile.")

    if not ensure_all_true(checks):
        raise FileNotFoundError("Input validation failed.")

    prefix = args.prefix or strip_default_prefix_suffix(auto_prefix)
    return gfile.replace("\\", "/"), source_label, prefix


def _safe_sample_ids(
    genotype_path: str,
    *,
    snps_only: bool,
    maf: float,
    missing_rate: float,
    expected_n: int,
    logger: logging.Logger,
) -> list[str]:
    sample_ids: list[str] = []
    try:
        sample_ids_raw, _ = inspect_genotype_file(
            genotype_path,
            snps_only=bool(snps_only),
            maf=float(maf),
            missing_rate=float(missing_rate),
        )
        sample_ids = [str(x) for x in sample_ids_raw]
    except Exception as ex:
        logger.warning(f"Failed to inspect sample IDs for Q output: {ex}")
        sample_ids = []

    if len(sample_ids) != int(expected_n):
        if len(sample_ids) > 0:
            logger.warning(
                "Sample ID count mismatch for Q output "
                f"(ids={len(sample_ids)}, expected={int(expected_n)}). "
                "Fallback to generated sample IDs."
            )
        sample_ids = [f"S{i+1}" for i in range(int(expected_n))]
    return sample_ids


def _write_matrix_with_row_ids(
    out_path: str,
    row_ids: list[str],
    mat: np.ndarray,
) -> None:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got shape={arr.shape}")
    if len(row_ids) != int(arr.shape[0]):
        raise ValueError(
            f"Row ID length mismatch: ids={len(row_ids)}, rows={int(arr.shape[0])}"
        )
    with open(out_path, "w", encoding="utf-8") as fw:
        for rid, row in zip(row_ids, arr):
            vals = "\t".join(f"{float(v):.8f}" for v in row)
            fw.write(f"{rid}\t{vals}\n")


def _build_parser() -> CliArgumentParser:
    parser = CliArgumentParser(
        prog="jx adamixture",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx adamixture -bfile data/geno -k 8 -o out -prefix cohort",
            "jx adamixture -vcf data/geno.vcf.gz -k 6 -t 16",
        ]),
    )

    req_geno = parser.add_argument_group("Required Genotype Input (Choose one)")
    geno = req_geno.add_mutually_exclusive_group(required=True)
    geno.add_argument("-bfile", "--bfile", type=str, help="PLINK prefix (.bed/.bim/.fam).")
    geno.add_argument("-vcf", "--vcf", type=str, help="VCF/VCF.GZ genotype file.")
    geno.add_argument("-hmp", "--hmp", type=str, help="HMP/HMP.GZ genotype file.")
    geno.add_argument(
        "-file",
        "--file",
        type=str,
        help="Text/NumPy genotype matrix prefix (requires sidecar .id).",
    )
    req_model = parser.add_argument_group("Required Nclusters")
    req_model.add_argument(
        "-k",
        "--k",
        type=int,
        required=True,
        help="Number of ancestry clusters.",
    )
    input_output = parser.add_argument_group("Input/Output Arguments")
    input_output.add_argument(
        "-o",
        "--out",
        type=str,
        default=".",
        help="Output directory (default: current directory).",
    )
    input_output.add_argument("-prefix", "--prefix", type=str, default=None, help="Output prefix.")
    input_output.add_argument(
        "-chunksize",
        "--chunksize",
        type=int,
        default=50000,
        help="Number of SNPs per loading chunk (default: 50000).",
    )
    input_output.add_argument(
        "-snps-only",
        "--snps-only",
        action="store_true",
        default=False,
        help="Input VCF/HMP: keep SNP variants only during loading.",
    )
    input_output.add_argument(
        "-maf",
        "--maf",
        type=float,
        default=0.02,
        help="Minor allele frequency threshold in loading stage (default: 0.02).",
    )
    input_output.add_argument(
        "-geno",
        "--geno",
        type=float,
        default=0.05,
        help="Missing-rate threshold in loading stage (default: 0.05).",
    )
    runtime = parser.add_argument_group("Runtime Arguments")
    runtime.add_argument(
        "-t",
        "--thread",
        dest="thread",
        type=int,
        default=detect_effective_threads(),
        help="Number of CPU threads (default: %(default)s).",
    )
    runtime.add_argument(
        "-threads",
        "--threads",
        dest="thread",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    runtime.add_argument("-seed", "--seed", type=int, default=42, help="Random seed (default: 42).")

    model = parser.add_argument_group("Model/Optimization Arguments")
    model.add_argument(
        "-solver",
        "--solver",
        type=str,
        default="adam-em",
        choices=["auto", "adam", "adam-em"],
        help="Optimization solver (default: adam-em).",
    )
    model.add_argument(
        "-max-iter",
        "--max-iter",
        dest="max_iter",
        type=int,
        default=500,
        help="Maximum optimization iterations (default: 500).",
    )
    model.add_argument(
        "-tol",
        "--tol",
        type=float,
        default=1e-5,
        help="Convergence tolerance (default: 1e-5).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not (0.0 <= float(args.maf) <= 0.5):
        parser.error("-maf must be within [0, 0.5].")
    if not (0.0 <= float(args.geno) <= 1.0):
        parser.error("-geno must be within [0, 1.0].")
    if float(args.tol) <= 0:
        parser.error("-tol/--tol must be > 0.")
    if int(args.max_iter) <= 0:
        parser.error("-max-iter/--max-iter must be a positive integer.")
    if int(args.thread) <= 0:
        parser.error("-t/--thread must be a positive integer.")
    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    thread_capped = False
    resolved_threads = int(args.thread)
    if int(resolved_threads) > int(detected_threads):
        thread_capped = True
        resolved_threads = int(detected_threads)

    outdir = str(args.out).replace("\\", "/").rstrip("/") or "."
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    tmp_logger = logging.getLogger("janusx.adamixture.bootstrap")
    if not tmp_logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        tmp_logger.addHandler(h)
    tmp_logger.setLevel(logging.INFO)
    tmp_logger.propagate = False
    try:
        genotype_path, source_label, auto_prefix = _resolve_input(args, tmp_logger)
    except Exception as e:
        print_failure(str(e), force_color=True)
        raise

    prefix = args.prefix or auto_prefix
    log_path = os.path.join(outdir, f"{prefix}.adamixture.log")
    logger = setup_logging(log_path)
    if thread_capped:
        logger.warning(
            f"Warning: Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(resolved_threads)}."
        )
    logger.info("")
    use_spinner = stdout_is_tty()

    q_out = os.path.join(outdir, f"{prefix}.{int(args.k)}.Q").replace("\\", "/")
    p_out = os.path.join(outdir, f"{prefix}.{int(args.k)}.P").replace("\\", "/")

    emit_cli_configuration(
        logger,
        app_title="JanusX - ADAMIXTURE",
        config_title="ADAMIXTURE CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "Input/Output",
                [
                    ("Genotype", genotype_path),
                    ("Input type", source_label),
                    ("Chunk size", int(args.chunksize)),
                    ("SNPs only", bool(args.snps_only)),
                    ("MAF threshold", float(args.maf)),
                    ("Miss threshold", float(args.geno)),
                    ("Output dir", outdir),
                    ("Prefix", prefix),
                ],
            ),
            (
                "Runtime",
                [
                    ("Threads", f"{int(resolved_threads)} ({detected_threads} available)"),
                    ("Seed", int(args.seed)),
                ],
            ),
            (
                "Model",
                [
                    ("K", int(args.k)),
                    ("Solver", str(args.solver)),
                    ("max_iter", int(args.max_iter)),
                    ("tol", float(args.tol)),
                ],
            ),
        ],
        footer_rows=[
            ("Output prefix", f"{outdir}/{prefix}"),
            ("Q output", q_out),
            ("P output", p_out),
            ("Log file", log_path),
        ],
    )

    cfg = ADAMixtureConfig(
        genotype_path=genotype_path,
        k=int(args.k),
        outdir=outdir,
        prefix=prefix,
        seed=int(args.seed),
        threads=max(1, int(resolved_threads)),
        chunk_size=max(1000, int(args.chunksize)),
        snps_only=bool(args.snps_only),
        maf=float(args.maf),
        geno=float(args.geno),
        solver=str(args.solver),
        max_iter=max(1, int(args.max_iter)),
        tol=float(args.tol),
    )

    progress_ui: Optional[_AdmixtureCliProgress] = None
    callback = None
    muted: list[tuple[logging.Handler, int]] = []
    try:
        if use_spinner:
            progress_ui = _AdmixtureCliProgress(logger, enabled=True)
            callback = progress_ui.on_event
            muted = _mute_stdout_info_logs(logger)
        try:
            p_mat, q_mat, m, n = train_adamixture(cfg, logger, callback=callback)
        finally:
            if progress_ui is not None:
                progress_ui.close()
            _restore_handler_levels(muted)
        q_ids = _safe_sample_ids(
            genotype_path,
            snps_only=bool(args.snps_only),
            maf=float(args.maf),
            missing_rate=float(args.geno),
            expected_n=int(n),
            logger=logger,
        )
        # P matrix is SNP x K (not sample x K), so prepend per-site row IDs.
        p_ids = [f"SNP{i+1}" for i in range(int(m))]
        _write_matrix_with_row_ids(q_out, q_ids, np.asarray(q_mat, dtype=np.float64))
        _write_matrix_with_row_ids(p_out, p_ids, np.asarray(p_mat, dtype=np.float64))
    except Exception as e:
        if progress_ui is not None:
            try:
                progress_ui.fail("ADAMIXTURE ...Failed")
            except Exception:
                pass
        logger.exception(f"ADAMIXTURE failed: {e}")
        print_failure("ADAMIXTURE ...Failed", force_color=True)
        raise

    logger.info(f"Q output: {q_out}")
    logger.info(f"P output: {p_out}")
    wall = float(time.time() - t0)
    now = datetime.now()
    logger.info("")
    log_success(logger, f"Finished. Total wall time: {wall:.2f} seconds")
    logger.info(
        f"{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}"
    )


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
