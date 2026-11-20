"""Windowed FST command exposed as ``jx fst``."""

from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path

from ._common.cli_args import add_common_out_arg, add_common_prefix_arg, add_common_thread_arg
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.config_render import emit_cli_configuration
from ._common.log import setup_logging
from ._common.outprefix import apply_output_prefix_compat
from ._common.pathcheck import format_path_for_display
from ._common.progress import CliStatus, format_elapsed, log_success
from ._common.threads import detect_effective_threads
from .population_progress import PopulationScanProgress

try:
    from janusx import janusx as jxrs
except Exception:
    jxrs = None


def _normalize_prefix(value: str) -> str:
    raw = str(value).strip()
    if raw.lower().endswith((".bed", ".bim", ".fam")):
        return raw[:-4]
    return raw


def _require_backend() -> None:
    if jxrs is None or not hasattr(jxrs, "fst_bed_window_to_tsv"):
        raise RuntimeError(
            "Rust windowed FST backend is unavailable. Rebuild/install JanusX before running "
            "jx fst."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx fst",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx fst -bfile cohort -p1 pop1.txt -p2 pop2.txt -window 50000 -step 50000 -o results/cohort_fst",
                "jx fst -bfile cohort -within groups.tsv -matrix -window 50000 -step 50000 -o results/cohort_fst",
            ]
        ),
        description=(
            "Windowed population differentiation from PLINK BED. "
            "The default estimator is Weir & Cockerham (wc)."
        ),
    )
    data = parser.add_argument_group("Input")
    data.add_argument("-bfile", "--bfile", required=True, help="PLINK BED/BIM/FAM prefix.")
    data.add_argument("-p1", "--p1", default=None, help="Population 1 sample list (IID or FID IID).")
    data.add_argument("-p2", "--p2", default=None, help="Population 2 sample list (IID or FID IID).")
    data.add_argument(
        "-within",
        "--within",
        default=None,
        help="FID IID GROUP file; required with -matrix for all population pairs.",
    )
    data.add_argument("-chr", "--chr", dest="chrom", default=None, help="Restrict scan to one chromosome.")

    scan = parser.add_argument_group("Window")
    scan.add_argument("-window", "--window", type=int, required=True, help="Window size in bp.")
    scan.add_argument("-step", "--step", type=int, required=True, help="Window step in bp.")
    advanced = parser.add_argument_group("Advanced Options")
    advanced.add_argument(
        "-method",
        "--method",
        choices=("wc", "hudson"),
        default="wc",
        help="Estimator: wc or hudson (default: wc).",
    )
    advanced.add_argument(
        "-matrix",
        "--matrix",
        action="store_true",
        help="With -within, emit every pairwise population comparison.",
    )

    output = parser.add_argument_group("Output / runtime")
    add_common_out_arg(output, default=".", help_profile="current_dir")
    add_common_prefix_arg(output, default=None, help_profile="input_basename")
    add_common_thread_arg(
        output,
        default_threads=max(1, int(detect_effective_threads())),
        help_profile="rust_auto",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _require_backend()
    if int(args.window) <= 0 or int(args.step) <= 0:
        parser.error("-window and -step must be positive")
    if int(args.thread) <= 0:
        parser.error("-t/--thread must be positive")
    if bool(args.matrix):
        if not args.within:
            parser.error("-matrix requires -within groups.tsv")
    elif not args.p1 or not args.p2:
        parser.error("-p1 and -p2 are required unless -matrix is used")

    bfile = _normalize_prefix(args.bfile)
    missing = [
        f"{bfile}{suffix}"
        for suffix in (".bed", ".bim", ".fam")
        if not Path(f"{bfile}{suffix}").is_file()
    ]
    if missing:
        raise FileNotFoundError("incomplete PLINK input prefix; missing: " + ", ".join(missing))
    for sample_file in (args.p1, args.p2, args.within):
        if sample_file and not os.path.isfile(sample_file):
            raise FileNotFoundError(sample_file)

    stem = os.path.basename(os.path.normpath(bfile))
    out_dir, output_prefix, _out_stem = apply_output_prefix_compat(
        args,
        stem,
        fallback_prefix="fst",
    )
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{output_prefix}.windowed.fst"
    log_path = f"{output_prefix}.fst.log"
    logger = setup_logging(log_path)
    emit_cli_configuration(
        logger,
        app_title="JanusX - FST",
        config_title="Windowed FST",
        host=socket.gethostname(),
        sections=[
            (
                "Input",
                [
                    ("BFILE", bfile),
                    ("Population 1", args.p1 or ""),
                    ("Population 2", args.p2 or ""),
                    ("Within", args.within or ""),
                ],
            ),
            (
                "Scan",
                [
                    ("Window", int(args.window)),
                    ("Step", int(args.step)),
                    ("Method", args.method),
                    ("Matrix", bool(args.matrix)),
                    ("Chromosome", args.chrom or "all"),
                ],
            ),
            ("Runtime", [("Threads", int(args.thread)), ("Output", output_path)]),
        ],
    )

    start = time.monotonic()
    progress = PopulationScanProgress(
        description="Computing windowed FST",
        stage_labels=("FST scan",),
        log_unit="site",
    )
    try:
        with CliStatus(
            "Computing windowed FST...",
            enabled=not progress.enabled,
        ) as task:
            try:
                n_rows, n_pairs = jxrs.fst_bed_window_to_tsv(
                    bfile,
                    args.p1 or "",
                    args.p2 or "",
                    args.within or "",
                    output_path,
                    window=int(args.window),
                    step=int(args.step),
                    method=args.method,
                    matrix=bool(args.matrix),
                    chrom=args.chrom,
                    threads=int(args.thread),
                    progress_callback=(progress.callback if progress.enabled else None),
                    progress_every=0,
                )
            except Exception:
                progress.close()
                task.fail("Computing windowed FST ...Failed")
                raise
            progress.finish()
            task.complete("Computing windowed FST ...Finished")
    finally:
        progress.close()
    elapsed = max(0.0, time.monotonic() - start)
    # Import the plotting stack only after the Rust scan completes.  This
    # keeps ``jx fst -h`` and the population scan startup path lightweight.
    from janusx.assoc.workflow_ui import _run_fast_manhattan_from_tsv_with_status

    plot_svg = f"{output_path}.svg"
    _plot_elapsed, plot_paths = _run_fast_manhattan_from_tsv_with_status(
        output_path,
        outsvg=plot_svg,
        score_col="weightedFst",
        score_label=f"{str(args.method).upper()} weighted FST",
        group_cols=("pop1", "pop2") if bool(args.matrix) else (),
    )
    log_success(
        logger,
        f"Windowed FST finished [{format_elapsed(elapsed)}]; "
        f"rows={int(n_rows)}, comparisons={int(n_pairs)}",
    )
    log_success(logger, f"FST table saved: {format_path_for_display(output_path)}")
    for path in plot_paths:
        log_success(logger, f"FST Manhattan plot saved: {format_path_for_display(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
