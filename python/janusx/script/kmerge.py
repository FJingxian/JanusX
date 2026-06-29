from __future__ import annotations

import argparse
import logging
import socket
import threading
from pathlib import Path

from janusx import janusx as jxrs

from ._common.config_render import emit_cli_configuration
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
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


class _KmergeCliProgress:
    _STAGE_LABELS = {
        1: "kmerge Stage 1/3",
        2: "kmerge Stage 2/3",
        3: "kmerge Stage 3/3",
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
        return cls._STAGE_LABELS.get(int(stage), f"kmerge Stage {int(stage)}")


def build_parser():
    parser = CliArgumentParser(
        prog="jx kmerge",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx kmerge -db /data/kmc/S001 /data/kmc/S002 -o out -prefix maize_k31",
                "jx kmerge -db '/data/kmc/*' -o out -prefix maize_k31",
                "jx kmerge -db samples.kmc.tsv -o out -prefix maize_k31 -t 32 -mem 128",
                "jx kmerge -db sampleA.kmc_pre sampleB.kmc_pre -o out -prefix panel --resume",
            ]
        ),
        description=(
            "Merge multi-sample KMC databases into JanusX k-mer bitmatrix output "
            "(.idv + .bkmer + .bsite + .meta.json) and write <prefix>.kmerge.log."
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
            "directory/glob auto-discovery, or a one/two-column text file."
        ),
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
        default="kmerge",
        help="Output file prefix (default: kmerge).",
    )
    basic.add_argument(
        "-t",
        "--thread",
        type=int,
        default=8,
        help="Number of worker threads (default: 8).",
    )
    basic.add_argument(
        "-mem",
        "--memory",
        type=float,
        default=DEFAULT_MEMORY_GB,
        help=(
            "Runtime memory budget in GB for merge and sorted-run stages. "
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
    basic.add_argument(
        "-freq",
        "--freq",
        type=float,
        default=0.02,
        help="Keep k-mers with presence rate between freq and 1-freq (default: 0.02).",
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
        default=1_048_576,
        help="KMC batch size per read call (default: 1048576).",
    )
    advanced.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum within-sample KMC count to keep (default: 1).",
    )
    advanced.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing tmp-dir stage markers.",
    )
    advanced.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary sorted runs and part files after success.",
    )
    advanced.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files and tmp-dir.",
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

    db_inputs = [str(item) for group in args.db for item in group]
    if len(db_inputs) == 0:
        parser.error("-db/--db cannot be empty.")
    if args.thread <= 0:
        parser.error("-t/--thread must be > 0.")
    if args.max_run_size <= 0:
        parser.error("--max-run-size must be > 0.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0.")
    if args.min_count <= 0:
        parser.error("--min-count must be > 0.")
    if not (0.0 <= float(args.freq) <= 0.5):
        parser.error("-freq/--freq must be within [0, 0.5].")
    if not (1 <= int(args.bucket_bits) <= 20):
        parser.error("--bucket-bits must be within [1, 20].")
    detected_threads = int(detect_effective_threads())
    threads = int(args.thread)
    out_dir = str(Path(args.out).expanduser().resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_prefix = str(args.prefix).strip()
    if output_prefix == "":
        parser.error("-prefix/--prefix cannot be empty.")
    log_path = str(Path(out_dir) / f"{output_prefix}.kmerge.log")
    logger = setup_logging(log_path)

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

    if resolved_n_samples == 1 and float(args.freq) > 0.0:
        parser.error(
            "single-sample kmerge with -freq > 0 filters out all present k-mers; "
            "use -freq 0.0 or provide multiple resolved KMC databases."
        )

    if threads > detected_threads:
        logger.warning(
            f"Requested threads={threads} exceeds detected available={detected_threads}; "
            f"using {detected_threads}."
        )
        threads = detected_threads

    emit_cli_configuration(
        logger,
        app_title="JanusX kmerge",
        config_title="Multi-sample KMC merge",
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
                "Filter",
                [
                    ("Min count", int(args.min_count)),
                    ("Freq", float(args.freq)),
                ],
            ),
            (
                "Runtime",
                [
                    (
                        "Threads",
                        format_requested_thread_usage(
                            requested_threads=args.thread,
                            using_threads=threads,
                            detected_threads=detected_threads,
                        ),
                    ),
                    ("Memory budget GB", float(args.memory)),
                    ("Bucket bits", int(args.bucket_bits)),
                    ("Batch size", int(args.batch_size)),
                    ("Keep tmp", bool(args.keep_tmp)),
                    ("Resume", bool(args.resume)),
                    ("Force", bool(args.force)),
                    ("Log file", log_path),
                ],
            ),
        ],
        footer_rows=[("Output dir", out_dir), ("Prefix", output_prefix)],
    )
    logger.info("Resolved KMC DB pairs:")
    for idx, (sample_id, resolved_prefix) in enumerate(
        zip(resolved_sample_ids, resolved_prefixes), start=1
    ):
        kmc_pre, kmc_suf = format_kmc_db_pair_for_display(resolved_prefix)
        logger.info(f"  [{idx}] {sample_id}\t{kmc_pre} | {kmc_suf}")
    _reopen_file_handlers_append(logger)
    progress_ui = _KmergeCliProgress()

    try:
        summary = jxrs.kmerge_run(
            db_inputs=db_inputs,
            sample_ids=None if args.sample_id is None else [str(x).strip() for x in args.sample_id],
            out=out_dir,
            prefix=output_prefix,
            thread=threads,
            memory=memory_mb,
            freq=float(args.freq),
            tmp_dir=None if args.tmp_dir is None else str(Path(args.tmp_dir).expanduser().resolve()),
            max_run_size=int(args.max_run_size),
            bucket_bits=int(args.bucket_bits),
            batch_size=int(args.batch_size),
            min_count=int(args.min_count),
            resume=bool(args.resume),
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
        f"Merged {int(summary['n_kmers'])} k-mers across {int(summary['n_samples'])} samples (k={int(summary['k'])}).",
    )
    logger.info("Output files:")
    logger.info(f"  {format_path_for_display(summary['idv'])}")
    logger.info(f"  {format_path_for_display(summary['bkmer'])}")
    logger.info(f"  {format_path_for_display(summary['bsite'])}")
    logger.info(f"  {format_path_for_display(summary['meta'])}")
    logger.info(f"Log file: {format_path_for_display(log_path)}")
    return 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
