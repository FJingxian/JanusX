from __future__ import annotations

from pathlib import Path

from janusx import janusx as jxrs

from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.threads import detect_effective_threads


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
            "(.idv + .bkmer + .bsite + .meta.json)."
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
        "-mp",
        "--min-presence",
        type=int,
        default=5,
        help="Keep k-mers present in at least N samples (default: 5).",
    )
    basic.add_argument(
        "-freq",
        "--freq",
        type=float,
        default=0.02,
        help="Keep k-mers with presence rate in [freq, 1-freq] (default: 0.02).",
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

    if args.thread <= 0:
        parser.error("-t/--thread must be > 0.")
    if args.memory <= 0:
        parser.error("-memory/--memory must be > 0.")
    if args.min_presence <= 0:
        parser.error("-mp/--min-presence must be > 0.")
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
    if threads > detected_threads:
        print(
            f"Warning: Requested threads={threads} exceeds detected available={detected_threads}; "
            f"using {detected_threads}.",
            flush=True,
        )
        threads = detected_threads

    out_dir = str(Path(args.out).expanduser().resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix).strip()
    if prefix == "":
        parser.error("-prefix/--prefix cannot be empty.")

    summary = jxrs.kmerge_run(
        db_inputs=db_inputs,
        sample_ids=None if args.sample_id is None else [str(x).strip() for x in args.sample_id],
        out=out_dir,
        prefix=prefix,
        thread=threads,
        memory=int(args.memory),
        min_presence=int(args.min_presence),
        freq=float(args.freq),
        tmp_dir=None if args.tmp_dir is None else str(Path(args.tmp_dir).expanduser().resolve()),
        max_run_size=int(args.max_run_size),
        bucket_bits=int(args.bucket_bits),
        batch_size=int(args.batch_size),
        min_count=int(args.min_count),
        resume=bool(args.resume),
        keep_tmp=bool(args.keep_tmp),
        force=bool(args.force),
    )

    print(
        f"Merged {int(summary['n_kmers'])} k-mers across {int(summary['n_samples'])} samples (k={int(summary['k'])}).",
        flush=True,
    )
    print(
        "Output files:\n"
        f"  {summary['idv']}\n"
        f"  {summary['bkmer']}\n"
        f"  {summary['bsite']}\n"
        f"  {summary['meta']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
