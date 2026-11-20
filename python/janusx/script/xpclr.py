"""XP-CLR command exposed as ``jx xpclr``."""

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
    if jxrs is None or not hasattr(jxrs, "xpclr_bed_to_tsv"):
        raise RuntimeError(
            "Rust XP-CLR backend is unavailable. Rebuild/install JanusX before running "
            "jx xpclr."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx xpclr",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx xpclr -bfile cohort -p1 pop1.txt -p2 pop2.txt -window 50000 -step 50000 -o results/cohort_xpclr",
                "jx xpclr -bfile cohort -p1 pop1.txt -p2 pop2.txt -map mapfile.snp -ld 0.95",
            ]
        ),
        description=(
            "Windowed XP-CLR selection scan from PLINK BED, following "
            "hardingnj/xpclr v1.1.2."
        ),
    )
    data = parser.add_argument_group("Input")
    data.add_argument("-bfile", "--bfile", required=True, help="PLINK BED/BIM/FAM prefix.")
    data.add_argument("-p1", "--p1", required=True, help="Population 1 sample list (IID or FID IID).")
    data.add_argument("-p2", "--p2", required=True, help="Population 2 sample list (IID or FID IID).")
    data.add_argument(
        "-map",
        "--map",
        dest="map_path",
        default=None,
        help="Optional XP-CLR map file; empty uses physical position * -rrate.",
    )
    data.add_argument("-chr", "--chr", dest="chrom", default=None, help="Restrict scan to one chromosome.")

    scan = parser.add_argument_group("Window")
    scan.add_argument("-window", "--window", type=int, required=True, help="Window size in bp.")
    scan.add_argument("-step", "--step", type=int, required=True, help="Window step in bp.")
    advanced = parser.add_argument_group("Advanced Options")
    advanced.add_argument("-maxsnps", "--maxsnps", type=int, default=600, help="Maximum SNPs per window (default: 600).")
    advanced.add_argument("-minsnps", "--minsnps", type=int, default=10, help="Minimum SNPs per window (default: 10).")
    advanced.add_argument("-ld", "--ld", dest="ld", type=float, default=0.95, help="LD r² pruning threshold (default: 0.95).")
    advanced.add_argument("-rrate", "--rrate", type=float, default=1e-8, help="Physical-to-genetic distance rate (default: 1e-8).")
    advanced.add_argument("-seed", "--seed", type=int, default=42, help="Seed for max-SNP window sampling (default: 42).")

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
    if int(args.maxsnps) <= 0 or int(args.minsnps) < 2:
        parser.error("-maxsnps must be positive and -minsnps must be at least 2")
    if int(args.minsnps) > int(args.maxsnps):
        parser.error("-minsnps cannot exceed -maxsnps")
    if not 0.0 <= float(args.ld) <= 1.0:
        parser.error("-ld must be within [0, 1]")
    if float(args.rrate) < 0.0:
        parser.error("-rrate must be non-negative")
    if int(args.thread) <= 0:
        parser.error("-t/--thread must be positive")

    bfile = _normalize_prefix(args.bfile)
    missing = [
        f"{bfile}{suffix}"
        for suffix in (".bed", ".bim", ".fam")
        if not Path(f"{bfile}{suffix}").is_file()
    ]
    if missing:
        raise FileNotFoundError("incomplete PLINK input prefix; missing: " + ", ".join(missing))
    for path in (args.p1, args.p2, args.map_path):
        if path and not os.path.isfile(path):
            raise FileNotFoundError(path)

    stem = os.path.basename(os.path.normpath(bfile))
    out_dir, output_prefix, _out_stem = apply_output_prefix_compat(
        args,
        stem,
        fallback_prefix="xpclr",
    )
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{output_prefix}.xpclr.tsv"
    log_path = f"{output_prefix}.xpclr.log"
    logger = setup_logging(log_path)
    emit_cli_configuration(
        logger,
        app_title="JanusX - XP-CLR",
        config_title="Windowed XP-CLR",
        host=socket.gethostname(),
        sections=[
            (
                "Input",
                [
                    ("BFILE", bfile),
                    ("Population 1", args.p1),
                    ("Population 2", args.p2),
                    ("Map", args.map_path or "physical-distance fallback"),
                    ("Chromosome", args.chrom or "all"),
                ],
            ),
            (
                "Scan",
                [
                    ("Window", int(args.window)),
                    ("Step", int(args.step)),
                    ("Max SNPs", int(args.maxsnps)),
                    ("Min SNPs", int(args.minsnps)),
                    ("LD cutoff", float(args.ld)),
                    ("Recombination rate", float(args.rrate)),
                ],
            ),
            ("Runtime", [("Threads", int(args.thread)), ("Output", output_path)]),
        ],
    )

    start = time.monotonic()
    progress = PopulationScanProgress(
        description="Computing windowed XP-CLR",
        stages=2,
        stage_labels=("XP-CLR omega", "XP-CLR windows"),
        log_unit="site",
        stage_weights=(0.2, 0.8),
    )
    try:
        with CliStatus(
            "Computing windowed XP-CLR...",
            enabled=not progress.enabled,
        ) as task:
            try:
                n_windows, n_valid = jxrs.xpclr_bed_to_tsv(
                    bfile,
                    args.p1,
                    args.p2,
                    output_path,
                    map_path=args.map_path,
                    chrom=args.chrom,
                    window=int(args.window),
                    step=int(args.step),
                    maxsnps=int(args.maxsnps),
                    minsnps=int(args.minsnps),
                    ld_cutoff=float(args.ld),
                    rrate=float(args.rrate),
                    seed=int(args.seed),
                    threads=int(args.thread),
                    progress_callback=(progress.callback if progress.enabled else None),
                    progress_every=0,
                )
            except Exception:
                progress.close()
                task.fail("Computing windowed XP-CLR ...Failed")
                raise
            progress.finish()
            task.complete("Computing windowed XP-CLR ...Finished")
    finally:
        progress.close()
    elapsed = max(0.0, time.monotonic() - start)
    # Import the plotting stack only after the Rust scan completes.  This
    # keeps ``jx xpclr -h`` and the population scan startup path lightweight.
    from janusx.assoc.workflow_ui import _run_fast_manhattan_from_tsv_with_status

    plot_svg = f"{output_prefix}.xpclr.svg"
    _plot_elapsed, plot_paths = _run_fast_manhattan_from_tsv_with_status(
        output_path,
        outsvg=plot_svg,
        score_col="XPCLR",
        score_label="XP-CLR",
    )
    log_success(
        logger,
        f"Windowed XP-CLR finished [{format_elapsed(elapsed)}]; "
        f"windows={int(n_windows)}, valid={int(n_valid)}",
    )
    log_success(logger, f"XP-CLR table saved: {format_path_for_display(output_path)}")
    for path in plot_paths:
        log_success(logger, f"XP-CLR Manhattan plot saved: {format_path_for_display(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
