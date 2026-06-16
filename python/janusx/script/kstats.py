from __future__ import annotations

import logging
import socket
import threading
from pathlib import Path

from janusx import janusx as jxrs

from ._common.config_render import emit_cli_configuration
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.pathcheck import format_path_for_display
from ._common.progress import ProgressAdapter
from ._common.status import log_success
from ._common.threads import detect_effective_threads, format_requested_thread_usage


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

    def callback(self, stage: int, done: int, total: int) -> None:
        stage_i = int(stage)
        done_i = max(0, int(done))
        total_i = max(1, int(total))
        with self._lock:
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
                "jx kstats -db panel.kmc.tsv -pair intersection -sid A B C -o out -t 8 -memory 4096",
            ]
        ),
        description=(
            "Compute pairwise KMC k-mer statistics from KMC databases using a "
            "bucket-parallel Rust backend without producing merged .bkmer/.bsite outputs."
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
        required=True,
        help=(
            "Input KMC databases. Supports KMC prefix, .kmc_pre/.kmc_suf path, "
            "or a one/two-column text file."
        ),
    )

    mode = required.add_mutually_exclusive_group(required=True)
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
        help="Write a 2-sample venn summary table.",
    )

    basic.add_argument(
        "-sid",
        "--sample-id",
        nargs="+",
        default=None,
        help="Optional sample IDs in the same order as -db.",
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
        "-memory",
        "--memory",
        type=int,
        default=2048,
        help="Total memory budget in MB for stage1 bucket runs (default: 2048).",
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

    db_inputs = [str(item) for group in args.db for item in group]
    if len(db_inputs) == 0:
        parser.error("-db/--db cannot be empty.")
    db_inputs_log = [str(Path(item).expanduser().resolve()) for item in db_inputs]

    if args.thread <= 0:
        parser.error("-t/--thread must be > 0.")
    if args.memory <= 0:
        parser.error("-memory/--memory must be > 0.")
    if args.max_run_size <= 0:
        parser.error("--max-run-size must be > 0.")
    if not (1 <= int(args.bucket_bits) <= 20):
        parser.error("--bucket-bits must be within [1, 20].")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0.")
    if args.min_count <= 0:
        parser.error("--min-count must be > 0.")
    if args.venn and len(db_inputs) != 2:
        parser.error("-venn requires exactly 2 KMC databases.")

    detected_threads = int(detect_effective_threads())
    requested_threads = int(args.thread)
    using_threads = min(requested_threads, detected_threads)
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix).strip()
    if prefix == "":
        parser.error("-prefix/--prefix cannot be empty.")
    log_path = str(out_dir / f"{prefix}.kstats.log")
    logger = setup_logging(log_path)

    if requested_threads > detected_threads:
        logger.warning(
            f"Requested threads={requested_threads} exceeds detected available={detected_threads}; using {using_threads}."
        )

    emit_cli_configuration(
        logger,
        app_title="JanusX kstats",
        config_title="KMC pairwise k-mer statistics",
        host=socket.gethostname(),
        sections=[
            (
                "Input",
                [
                    ("DB inputs", len(db_inputs)),
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
                    ("Memory MB", int(args.memory)),
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
        footer_rows=[("Output dir", str(out_dir)), ("Prefix", prefix)],
    )
    logger.info("Input KMC DB arguments:")
    for idx, path in enumerate(db_inputs_log, start=1):
        logger.info(f"  [{idx}] {path}")
    _reopen_file_handlers_append(logger)

    progress_ui = _KstatsCliProgress()
    try:
        summary = jxrs.kstats_run(
            db_inputs=db_inputs,
            sample_ids=None if args.sample_id is None else [str(x).strip() for x in args.sample_id],
            out=str(out_dir),
            prefix=prefix,
            pair=(None if args.pair is None else str(args.pair)),
            venn=bool(args.venn),
            thread=using_threads,
            memory=int(args.memory),
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
    logger.info(f"Log file: {format_path_for_display(log_path)}")
    return 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
