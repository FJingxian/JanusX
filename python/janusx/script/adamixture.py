# -*- coding: utf-8 -*-
"""
JanusX: ADAMIXTURE ancestry inference (Rust-kernel backend)

Examples
--------
  jx adamixture -bfile data/geno -k 8 -o out -prefix cohort
  jx adamixture -vcf data/geno.vcf.gz -k 6 -threads 16
"""

from __future__ import annotations

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
from ._common.status import CliStatus, print_failure, print_success


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
                print_success(f"Loaded genotype matrix (SNPs={m}, samples={n})")
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


def main() -> None:
    parser = CliArgumentParser(
        prog="jx adamixture",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx adamixture -bfile data/geno -k 8 -o out -prefix cohort",
            "jx adamixture -vcf data/geno.vcf.gz -k 6 -threads 16",
        ]),
    )

    required = parser.add_argument_group("Required Arguments")
    geno = required.add_mutually_exclusive_group(required=True)
    geno.add_argument("-bfile", "--bfile", type=str, help="PLINK prefix (.bed/.bim/.fam).")
    geno.add_argument("-vcf", "--vcf", type=str, help="VCF/VCF.GZ genotype file.")
    geno.add_argument("-hmp", "--hmp", type=str, help="HMP/HMP.GZ genotype file.")
    geno.add_argument(
        "-file",
        "--file",
        type=str,
        help="Text/NumPy genotype matrix prefix (requires sidecar .id).",
    )
    required.add_argument("-k", "--k", type=int, required=True, help="Number of ancestry clusters.")

    optional = parser.add_argument_group("Optional Arguments")
    optional.add_argument(
        "-o",
        "--out",
        type=str,
        default=".",
        help="Output directory (default: current directory).",
    )
    optional.add_argument("-prefix", "--prefix", type=str, default=None, help="Output prefix.")
    optional.add_argument(
        "-threads",
        "--threads",
        type=int,
        default=None,
        help="CPU threads for BLAS/OpenMP backends (default: all available cores).",
    )
    optional.add_argument(
        "-chunksize",
        "--chunksize",
        type=int,
        default=50000,
        help="Number of SNPs per loading chunk (default: 50000).",
    )
    optional.add_argument(
        "-snps-only",
        "--snps-only",
        action="store_true",
        default=False,
        help="Input VCF/HMP: keep SNP variants only during loading.",
    )
    optional.add_argument(
        "-maf",
        "--maf",
        type=float,
        default=0.02,
        help="Minor allele frequency threshold in loading stage (default: 0.02).",
    )
    optional.add_argument(
        "-geno",
        "--geno",
        type=float,
        default=0.05,
        help="Missing-rate threshold in loading stage (default: 0.05).",
    )

    optional.add_argument("-seed", "--seed", type=int, default=42, help="Random seed (default: 42).")
    optional.add_argument("-lr", "--lr", type=float, default=0.005, help="Adam learning rate.")
    optional.add_argument("-min_lr", "--min_lr", type=float, default=1e-6, help="Minimum learning rate.")
    optional.add_argument("-lr_decay", "--lr_decay", type=float, default=0.5, help="Learning rate decay factor.")
    optional.add_argument("-beta1", "--beta1", type=float, default=0.80, help="Adam beta1.")
    optional.add_argument("-beta2", "--beta2", type=float, default=0.88, help="Adam beta2.")
    optional.add_argument("-reg_adam", "--reg_adam", type=float, default=1e-8, help="Adam epsilon.")
    optional.add_argument("-max_iter", "--max_iter", type=int, default=1500, help="Maximum ADAM-EM iterations.")
    optional.add_argument("-check", "--check", type=int, default=5, help="Log-likelihood check interval.")

    optional.add_argument("-max_als", "--max_als", type=int, default=1000, help="Maximum ALS iterations.")
    optional.add_argument("-tole_als", "--tole_als", type=float, default=1e-4, help="ALS convergence tolerance.")
    optional.add_argument("-reg_als", "--reg_als", type=float, default=1e-5, help="ALS regularization.")
    optional.add_argument("-power", "--power", type=int, default=5, help="RSVD power iterations.")
    optional.add_argument("-tole_svd", "--tole_svd", type=float, default=1e-1, help="RSVD convergence tolerance.")

    args = parser.parse_args()
    if not (0.0 <= float(args.maf) <= 0.5):
        parser.error("-maf must be within [0, 0.5].")
    if not (0.0 <= float(args.geno) <= 1.0):
        parser.error("-geno must be within [0, 1.0].")
    if args.threads is not None and int(args.threads) <= 0:
        parser.error("-threads must be a positive integer.")

    resolved_threads = int(args.threads) if args.threads is not None else max(1, int(os.cpu_count() or 1))

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
    logger.info("")
    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())

    q_out = os.path.join(outdir, f"{prefix}.{int(args.k)}.Q").replace("\\", "/")
    p_out = os.path.join(outdir, f"{prefix}.{int(args.k)}.P").replace("\\", "/")

    emit_cli_configuration(
        logger,
        app_title="JanusX - ADAMIXTURE",
        config_title="ADAMIXTURE CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "General",
                [
                    ("Genotype", genotype_path),
                    ("Input type", source_label),
                    ("K", int(args.k)),
                    ("Seed", int(args.seed)),
                    ("Threads", int(resolved_threads)),
                    ("Chunk size", int(args.chunksize)),
                    ("SNPs only", bool(args.snps_only)),
                    ("MAF threshold", float(args.maf)),
                    ("Miss threshold", float(args.geno)),
                ],
            ),
            (
                "Optimization",
                [
                    ("lr", float(args.lr)),
                    ("min_lr", float(args.min_lr)),
                    ("lr_decay", float(args.lr_decay)),
                    ("beta1", float(args.beta1)),
                    ("beta2", float(args.beta2)),
                    ("reg_adam", float(args.reg_adam)),
                    ("max_iter", int(args.max_iter)),
                    ("check", int(args.check)),
                    ("max_als", int(args.max_als)),
                    ("tole_als", float(args.tole_als)),
                    ("reg_als", float(args.reg_als)),
                    ("power", int(args.power)),
                    ("tole_svd", float(args.tole_svd)),
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
        lr=float(args.lr),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        reg_adam=float(args.reg_adam),
        lr_decay=float(args.lr_decay),
        min_lr=float(args.min_lr),
        max_iter=max(1, int(args.max_iter)),
        check=max(1, int(args.check)),
        max_als=max(1, int(args.max_als)),
        tole_als=float(args.tole_als),
        reg_als=float(args.reg_als),
        power=max(1, int(args.power)),
        tole_svd=float(args.tole_svd),
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
        np.savetxt(q_out, np.asarray(q_mat, dtype=np.float64), fmt="%.8f", delimiter=" ")
        np.savetxt(p_out, np.asarray(p_mat, dtype=np.float64), fmt="%.8f", delimiter=" ")
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
    logger.info(f"Finished. Total wall time: {wall:.2f} seconds")
    logger.info(
        f"{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}"
    )


if __name__ == "__main__":
    main()
