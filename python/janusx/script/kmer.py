from __future__ import annotations

import argparse
import os
from pathlib import Path

from janusx.kmc_bind import run_kmc_count

from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.threads import detect_effective_threads


def _infer_input_type(paths: list[str]) -> str:
    if not paths:
        raise ValueError("No input file provided.")
    low = str(paths[0]).lower()
    if low.endswith((".fq", ".fastq", ".fq.gz", ".fastq.gz")):
        return "fastq"
    if low.endswith((".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz")):
        return "fasta"
    raise ValueError(
        "Cannot infer input type from extension. Please set --input-type fastq/fasta/multiline-fasta."
    )


def _normalize_prefix(path_or_prefix: str) -> str:
    p = str(path_or_prefix).strip()
    if p == "":
        raise ValueError("Output prefix cannot be empty.")
    low = p.lower()
    if low.endswith(".kmc_pre"):
        return p[: -len(".kmc_pre")]
    if low.endswith(".kmc_suf"):
        return p[: -len(".kmc_suf")]
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx kmer",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx kmer -i sample.fastq.gz -o out/sample_k31 -k 31 -t 8",
                "jx kmer -i a_R1.fq.gz a_R2.fq.gz -o out/paired -k 27 --input-type fastq",
                "jx kmer -i sample.fa.gz -o out/sample_fa --input-type fasta --kmc-src /path/to/KMC",
            ]
        ),
        description=(
            "Count k-mers from FASTA/FASTQ via KMC core (pybind C++ backend) and write "
            "KMC database files: <prefix>.kmc_pre and <prefix>.kmc_suf."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="Input FASTQ/FASTA files (one or more).",
    )
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        help="Output KMC prefix (without .kmc_pre/.kmc_suf).",
    )
    parser.add_argument("-k", "--kmer-len", type=int, default=31, help="K-mer length (default: 31).")
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=0,
        help="CPU threads (default: auto-detect).",
    )
    parser.add_argument(
        "-m",
        "--max-ram-gb",
        type=int,
        default=12,
        help="Max RAM in GB for KMC stages (default: 12).",
    )
    parser.add_argument(
        "-ci",
        "--cutoff-min",
        type=int,
        default=2,
        help="Minimal k-mer count cutoff (default: 2).",
    )
    parser.add_argument(
        "-cx",
        "--cutoff-max",
        type=int,
        default=1_000_000_000,
        help="Maximal k-mer count cutoff (default: 1000000000).",
    )
    parser.add_argument(
        "--counter-max",
        type=int,
        default=255,
        help="Maximum stored counter value (default: 255).",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Temporary directory for KMC intermediate files (default: <out>.kmc_tmp).",
    )
    parser.add_argument(
        "--input-type",
        type=str,
        choices=["auto", "fastq", "fasta", "multiline-fasta"],
        default="auto",
        help="Input sequence type (default: auto from extension).",
    )
    parser.add_argument(
        "--no-canonical",
        action="store_true",
        default=False,
        help="Disable canonical k-mer mode.",
    )
    parser.add_argument(
        "--kmc-src",
        type=str,
        default=None,
        help="Path to KMC source tree (contains kmc_core/kmc_runner.h).",
    )
    parser.add_argument(
        "--rebuild-bind",
        action="store_true",
        default=False,
        help="Force rebuild of pybind KMC extension.",
    )
    parser.add_argument(
        "--verbose-build",
        action="store_true",
        default=False,
        help="Print pybind build commands.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_files = [str(Path(x).expanduser()) for x in args.input]
    for p in input_files:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Input file not found: {p}")

    output_prefix = _normalize_prefix(str(Path(args.out).expanduser()))
    out_parent = Path(output_prefix).parent
    if str(out_parent) not in {"", "."}:
        out_parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = str(Path(args.tmp_dir).expanduser()) if args.tmp_dir else f"{output_prefix}.kmc_tmp"
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    if args.input_type == "auto":
        input_type = _infer_input_type(input_files)
    else:
        input_type = str(args.input_type)

    threads = int(args.threads) if int(args.threads) > 0 else int(detect_effective_threads())
    canonical = not bool(args.no_canonical)

    print(
        f"Running KMC count: inputs={len(input_files)}, k={int(args.kmer_len)}, "
        f"threads={threads}, max_ram_gb={int(args.max_ram_gb)}, input_type={input_type}",
        flush=True,
    )

    stats = run_kmc_count(
        input_files=input_files,
        output_prefix=output_prefix,
        tmp_dir=tmp_dir,
        kmer_len=int(args.kmer_len),
        threads=threads,
        max_ram_gb=int(args.max_ram_gb),
        cutoff_min=int(args.cutoff_min),
        cutoff_max=int(args.cutoff_max),
        counter_max=int(args.counter_max),
        canonical=canonical,
        input_type=input_type,
        kmc_src=args.kmc_src,
        rebuild_bind=bool(args.rebuild_bind),
        verbose_build=bool(args.verbose_build),
    )

    print(
        f"KMC finished: stage1={float(stats.get('stage1_time_s', 0.0)):.3f}s, "
        f"stage2={float(stats.get('stage2_time_s', 0.0)):.3f}s, "
        f"unique={int(stats.get('n_unique_kmers', 0))}, total={int(stats.get('n_total_kmers', 0))}"
    )
    print(f"Output files:\n  {output_prefix}.kmc_pre\n  {output_prefix}.kmc_suf")
    return 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
