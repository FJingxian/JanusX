from __future__ import annotations

import argparse
import logging
import socket
import threading
from pathlib import Path

from janusx import janusx as jxrs

from ._common.config_render import emit_cli_configuration
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.pathcheck import format_kmc_db_pair_for_display, format_path_for_display
from ._common.progress import ProgressAdapter, log_success
from ._common.threads import detect_effective_threads, format_requested_thread_usage

DEFAULT_MEMORY_GB = 8.0


def _normalize_memory_gb(memory_gb: float | int | None) -> float:
    if memory_gb is None:
        return float(DEFAULT_MEMORY_GB)
    gb = float(memory_gb)
    if gb <= 0.0:
        raise ValueError(f"--memory must be > 0 GB, got {memory_gb}")
    return gb


def _memory_gb_to_mb(memory_gb: float | int | None) -> int:
    return max(1, int(round(_normalize_memory_gb(memory_gb) * 1024.0)))


def _reopen_file_handlers_append(logger: logging.Logger) -> None:
    file_specs: list[tuple[str, int, logging.Formatter | None]] = []
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            try:
                handler.flush()
            except Exception:
                pass
            file_specs.append((str(handler.baseFilename), int(handler.level), handler.formatter))
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    for filename, level, formatter in file_specs:
        handler = logging.FileHandler(filename, mode="a", encoding="utf-8")
        handler.setLevel(level)
        if formatter is not None:
            handler.setFormatter(formatter)
        logger.addHandler(handler)


class _KstatsCliProgress:
    _STAGE_LABELS = {
        1: "kstats Stage 1/3",
        2: "kstats Stage 2/3",
        3: "kstats Stage 3/3",
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stage = None
        self._done = 0
        self._total = 0
        self._bar = None
        self._completed_stage = None

    def callback(self, stage: int, done: int, total: int) -> None:
        stage_i = int(stage)
        done_i = max(0, int(done))
        total_i = max(1, int(total))
        with self._lock:
            if (
                self._bar is None
                and self._stage is None
                and self._completed_stage == stage_i
                and done_i >= total_i
            ):
                return
            if self._stage != stage_i or self._bar is None:
                self._close_locked()
                self._stage = stage_i
                self._done = 0
                self._total = total_i
                self._bar = ProgressAdapter(
                    total=total_i,
                    desc=self._stage_desc(stage_i),
                    show_postfix=False,
                    keep_display=False,
                    emit_done=False,
                    force_animate=True,
                )
            elif self._total != total_i:
                self._total = total_i
                self._bar.set_total(total_i)

            if done_i < self._done:
                self._close_locked()
                self._stage = stage_i
                self._done = 0
                self._total = total_i
                self._bar = ProgressAdapter(
                    total=total_i,
                    desc=self._stage_desc(stage_i),
                    show_postfix=False,
                    keep_display=False,
                    emit_done=False,
                    force_animate=True,
                )

            if self._bar is not None and done_i > self._done:
                self._bar.update(done_i - self._done)
            self._done = done_i

            if self._bar is not None and done_i >= total_i:
                self._bar.finish()
                self._bar.close()
                self._bar = None
                self._completed_stage = stage_i
                self._stage = None
                self._done = 0
                self._total = 0

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        self._stage = None
        self._done = 0
        self._total = 0

    @classmethod
    def _stage_desc(cls, stage: int) -> str:
        return cls._STAGE_LABELS.get(int(stage), f"kstats Stage {int(stage)}")


def build_parser():
    parser = CliArgumentParser(
        prog="jx kstats",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx kstats -db sample_A sample_B -pair both -o out",
                "jx kstats -db sample_A -db sample_B -venn -o out -prefix pairAB",
                "jx kstats -db './panel/*.jx*' -venn -o out -prefix panel_venn",
                "jx kstats -db panel.kmc.tsv -pair intersection -sid A B C -o out -t 8 -mem 8",
                "jx kstats -kbin test.kmer/kmerge -compare WY25=WY2_11.jx,WY2_21.jx WY35P=YY3P_11.jx,YY5P_11.jx -o out",
            ]
        ),
        description=(
            "Compute k-mer statistics either from KMC databases or directly from an existing "
            "JanusX kmerge bitmatrix using Rust backends."
        ),
    )

    required = parser.add_argument_group("Required arguments")
    basic = parser.add_argument_group("Basic arguments")
    advanced = parser.add_argument_group("Advanced arguments")

    required.add_argument(
        "-db",
        "--db",
        nargs="+",
        action="append",
        help=(
            "Input KMC databases. Supports KMC prefix, .kmc_pre/.kmc_suf path, "
            "directory/glob auto-discovery, or a one/two-column text file."
        ),
    )
    required.add_argument(
        "-kbin",
        "--kbin",
        type=str,
        default=None,
        help=(
            "Input JanusX kmerge bitmatrix prefix or .meta.json path. "
            "Example: /path/to/kmerge or /path/to/kmerge.meta.json."
        ),
    )

    mode = required.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "-pair",
        "--pair",
        choices=["union", "intersection", "both"],
        default=None,
        help="Write pairwise lower-triangle matrix statistics.",
    )
    mode.add_argument(
        "-venn",
        "--venn",
        action="store_true",
        help=(
            "Write venn-style presence statistics. For 2 samples, outputs the classic "
            "summary row; for >2 samples, outputs one row per observed presence pattern."
        ),
    )
    required.add_argument(
        "-compare",
        "--compare",
        nargs="+",
        default=None,
        help=(
            "Bitmatrix compare groups for -kbin mode. Each item is either NAME=sample1,sample2 "
            "or sample1,sample2. At least 2 groups are required."
        ),
    )

    basic.add_argument(
        "-sid",
        "--sample-id",
        nargs="+",
        default=None,
        help="Optional sample IDs in the same order as -db. Only used in -db mode.",
    )
    basic.add_argument(
        "-o",
        "--out",
        type=str,
        default=".",
        help="Output directory (default: .).",
    )
    basic.add_argument(
        "-prefix",
        "--prefix",
        type=str,
        default="kstats",
        help="Output file prefix (default: kstats).",
    )
    basic.add_argument(
        "-t",
        "--thread",
        type=int,
        default=8,
        help="Number of worker threads for bucket-parallel stages (default: 8).",
    )
    basic.add_argument(
        "-mem",
        "--memory",
        type=float,
        default=DEFAULT_MEMORY_GB,
        help=(
            "Runtime memory budget in GB for bucket and sorted-run stages. "
            "Larger values reduce temp-run fragmentation; this is not a decode-block size "
            "(default: %(default)s)."
        ),
    )
    basic.add_argument(
        "-memory",
        dest="memory",
        type=float,
        help=argparse.SUPPRESS,
    )

    advanced.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Temporary working directory (default: OUT/PREFIX.tmp).",
    )
    advanced.add_argument(
        "--max-run-size",
        type=int,
        default=2048,
        help="Maximum sorted-run size in MB (default: 2048).",
    )
    advanced.add_argument(
        "--bucket-bits",
        type=int,
        default=10,
        help="Number of high bits used for bucketing (default: 10).",
    )
    advanced.add_argument(
        "--batch-size",
        type=int,
        default=65_536,
        help="Per-sample KMC streaming batch size (default: 65536).",
    )
    advanced.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum within-sample KMC count to keep after opening DB (default: 1).",
    )
    advanced.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary bucket/run files after success.",
    )
    advanced.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.memory = _normalize_memory_gb(args.memory)
    except ValueError as exc:
        parser.error(str(exc))
    memory_mb = _memory_gb_to_mb(args.memory)

    db_inputs = [str(item) for group in (args.db or []) for item in group]
    has_db = len(db_inputs) > 0
    has_kbin = args.kbin is not None and str(args.kbin).strip() != ""
    if has_db == has_kbin:
        parser.error("Exactly one of -db/--db or -kbin/--kbin is required.")
    if has_db:
        if args.pair is None and not bool(args.venn):
            parser.error("One of -pair/--pair or -venn/--venn is required in -db mode.")
        if args.compare is not None:
            parser.error("-compare/--compare is only valid in -kbin mode.")
    else:
        if args.sample_id is not None:
            parser.error("-sid/--sample-id is only valid in -db mode.")
        if args.pair is not None or bool(args.venn):
            parser.error("-pair/--pair and -venn/--venn are not used with -kbin; use -compare.")
        if args.compare is None or len(args.compare) < 2:
            parser.error("-kbin mode requires at least 2 -compare group definitions.")
    if args.thread <= 0:
        parser.error("-t/--thread must be > 0.")
    if args.max_run_size <= 0:
        parser.error("--max-run-size must be > 0.")
    if not (1 <= int(args.bucket_bits) <= 20):
        parser.error("--bucket-bits must be within [1, 20].")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0.")
    if args.min_count <= 0:
        parser.error("--min-count must be > 0.")
    detected_threads = int(detect_effective_threads())
    requested_threads = int(args.thread)
    using_threads = min(requested_threads, detected_threads)
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(args.prefix).strip()
    if output_prefix == "":
        parser.error("-prefix/--prefix cannot be empty.")
    log_path = str(out_dir / f"{output_prefix}.kstats.log")
    logger = setup_logging(log_path)

    if requested_threads > detected_threads:
        logger.warning(
            f"Requested threads={requested_threads} exceeds detected available={detected_threads}; using {using_threads}."
        )

    resolved = None
    resolved_n_samples = None
    resolved_sample_ids: list[str] = []
    resolved_prefixes: list[str] = []
    kbin_prefix = None
    compare_specs = None
    if has_db:
        try:
            resolved = jxrs.kmer_resolve_inputs(
                db_inputs=db_inputs,
                sample_ids=None if args.sample_id is None else [str(x).strip() for x in args.sample_id],
            )
        except Exception as exc:
            logger.error(str(exc))
            return 1
        resolved_n_samples = int(resolved["n_samples"])
        resolved_sample_ids = [str(x) for x in resolved["sample_ids"]]
        resolved_prefixes = [str(x) for x in resolved["prefixes"]]
        emit_cli_configuration(
            logger,
            app_title="JanusX kstats",
            config_title="KMC pairwise k-mer statistics",
            host=socket.gethostname(),
            sections=[
                (
                    "Input",
                    [
                        ("Resolved DBs", resolved_n_samples),
                        ("Sample IDs", len(args.sample_id) if args.sample_id is not None else "auto"),
                    ],
                ),
                (
                    "Mode",
                    [
                        ("Operation", f"pair:{args.pair}" if args.pair is not None else "venn"),
                        ("Backend", "bucket-parallel"),
                    ],
                ),
                (
                    "Runtime",
                    [
                        (
                            "Threads",
                            format_requested_thread_usage(
                                requested_threads=requested_threads,
                                using_threads=using_threads,
                                detected_threads=detected_threads,
                            ),
                        ),
                        ("Memory budget GB", float(args.memory)),
                        ("Bucket bits", int(args.bucket_bits)),
                        ("Max run MB", int(args.max_run_size)),
                        ("Batch size", int(args.batch_size)),
                        ("Min count", int(args.min_count)),
                        ("Keep tmp", bool(args.keep_tmp)),
                        ("Force", bool(args.force)),
                        ("Log file", log_path),
                    ],
                ),
            ],
            footer_rows=[("Output dir", str(out_dir)), ("Prefix", output_prefix)],
        )
        logger.info("Resolved KMC DB pairs:")
        for idx, (sample_id, resolved_prefix) in enumerate(
            zip(resolved_sample_ids, resolved_prefixes), start=1
        ):
            kmc_pre, kmc_suf = format_kmc_db_pair_for_display(resolved_prefix)
            logger.info(f"  [{idx}] {sample_id}\t{kmc_pre} | {kmc_suf}")
    else:
        kbin_prefix = str(Path(args.kbin).expanduser().resolve())
        compare_specs = [str(x).strip() for x in args.compare]
        emit_cli_configuration(
            logger,
            app_title="JanusX kstats",
            config_title="Bitmatrix group comparison",
            host=socket.gethostname(),
            sections=[
                (
                    "Input",
                    [
                        ("Kbin prefix", kbin_prefix),
                        ("Compare groups", len(compare_specs)),
                    ],
                ),
                (
                    "Mode",
                    [
                        ("Operation", "compare"),
                        ("Backend", "bitmatrix-parallel"),
                    ],
                ),
                (
                    "Runtime",
                    [
                        (
                            "Threads",
                            format_requested_thread_usage(
                                requested_threads=requested_threads,
                                using_threads=using_threads,
                                detected_threads=detected_threads,
                            ),
                        ),
                        ("Force", bool(args.force)),
                        ("Log file", log_path),
                    ],
                ),
            ],
            footer_rows=[("Output dir", str(out_dir)), ("Prefix", output_prefix)],
        )
        logger.info(f"Input kmerge prefix: {format_path_for_display(kbin_prefix)}")
        logger.info("Compare groups:")
        for idx, spec in enumerate(compare_specs, start=1):
            logger.info(f"  [{idx}] {spec}")
    _reopen_file_handlers_append(logger)

    progress_ui = _KstatsCliProgress()
    try:
        summary = jxrs.kstats_run(
            db_inputs=db_inputs,
            sample_ids=None if args.sample_id is None else [str(x).strip() for x in args.sample_id],
            kbin=kbin_prefix,
            compare_groups=compare_specs,
            out=str(out_dir),
            prefix=output_prefix,
            pair=(None if args.pair is None else str(args.pair)),
            venn=bool(args.venn),
            thread=using_threads,
            memory=memory_mb,
            tmp_dir=(None if args.tmp_dir is None else str(Path(args.tmp_dir).expanduser().resolve())),
            max_run_size=int(args.max_run_size),
            bucket_bits=int(args.bucket_bits),
            batch_size=int(args.batch_size),
            min_count=int(args.min_count),
            keep_tmp=bool(args.keep_tmp),
            force=bool(args.force),
            log_path=log_path,
            progress_callback=progress_ui.callback,
        )
    except Exception as exc:
        progress_ui.close()
        logger.error(str(exc))
        return 1
    finally:
        progress_ui.close()

    backend = str(summary.get("backend", "bucket-parallel"))
    if backend == "bitmatrix-parallel":
        log_success(
            logger,
            (
                f"kstats finished: samples={int(summary['n_samples'])}, groups={int(summary['n_groups'])}, "
                f"k={int(summary['k'])}, scan_mode={summary.get('scan_mode')}, "
                f"matrix_bytes={int(summary['matrix_bytes'])}"
            ),
        )
    else:
        log_success(
            logger,
            f"kstats finished: samples={int(summary['n_samples'])}, k={int(summary['k'])}, matrix_bytes={int(summary['matrix_bytes'])}",
        )
    logger.info("Output files:")
    if summary["pair_intersection"] is not None:
        logger.info(f"  {format_path_for_display(summary['pair_intersection'])}")
    if summary["pair_union"] is not None:
        logger.info(f"  {format_path_for_display(summary['pair_union'])}")
    if summary["venn"] is not None:
        logger.info(f"  {format_path_for_display(summary['venn'])}")
    if summary["compare_groups"] is not None:
        logger.info(f"  {format_path_for_display(summary['compare_groups'])}")
    if summary["compare_patterns"] is not None:
        logger.info(f"  {format_path_for_display(summary['compare_patterns'])}")
    if summary["compare_pairs"] is not None:
        logger.info(f"  {format_path_for_display(summary['compare_pairs'])}")
    logger.info(f"Log file: {format_path_for_display(log_path)}")
    return 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
