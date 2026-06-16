from __future__ import annotations

import logging
import socket
from pathlib import Path

from janusx import janusx as jxrs

from ._common.config_render import emit_cli_configuration
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.pathcheck import format_path_for_display
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


def build_parser():
    parser = CliArgumentParser(
        prog="jx kmerge",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx kmerge -db /data/kmc/S001 /data/kmc/S002 -o out -prefix maize_k31",
                "jx kmerge -db samples.kmc.tsv -o out -prefix maize_k31 -t 32 -memory 128000",
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
            "or a one/two-column text file."
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
        "-memory",
        "--memory",
        type=int,
        default=2048,
        help="Total memory budget in MB (default: 2048).",
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
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0.")
    if args.min_count <= 0:
        parser.error("--min-count must be > 0.")
    if not (0.0 <= float(args.freq) <= 0.5):
        parser.error("-freq/--freq must be within [0, 0.5].")
    if not (1 <= int(args.bucket_bits) <= 20):
        parser.error("--bucket-bits must be within [1, 20].")
    if len(db_inputs) == 1 and float(args.freq) > 0.0:
        parser.error(
            "single-sample kmerge with -freq > 0 filters out all present k-mers; "
            "use -freq 0.0 or provide multiple samples in one -db group or repeated -db flags."
        )

    detected_threads = int(detect_effective_threads())
    threads = int(args.thread)
    out_dir = str(Path(args.out).expanduser().resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix).strip()
    if prefix == "":
        parser.error("-prefix/--prefix cannot be empty.")
    log_path = str(Path(out_dir) / f"{prefix}.kmerge.log")
    logger = setup_logging(log_path)

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
                    ("DB inputs", len(db_inputs)),
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
                        ("Memory MB", int(args.memory)),
                        ("Bucket bits", int(args.bucket_bits)),
                        ("Batch size", int(args.batch_size)),
                    ("Keep tmp", bool(args.keep_tmp)),
                    ("Resume", bool(args.resume)),
                    ("Force", bool(args.force)),
                    ("Log file", log_path),
                ],
            ),
        ],
        footer_rows=[("Output dir", out_dir), ("Prefix", prefix)],
    )
    logger.info("Input KMC DB arguments:")
    for idx, path in enumerate(db_inputs_log, start=1):
        logger.info(f"  [{idx}] {path}")
    _reopen_file_handlers_append(logger)

    try:
        summary = jxrs.kmerge_run(
            db_inputs=db_inputs,
            sample_ids=None if args.sample_id is None else [str(x).strip() for x in args.sample_id],
            out=out_dir,
            prefix=prefix,
            thread=threads,
            memory=int(args.memory),
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
        )
    except Exception as exc:
        logger.error(str(exc))
        return 1

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
