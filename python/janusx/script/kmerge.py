from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from janusx.kmc_bind import (
    run_kmc_db_info,
    run_kmc_export_bin_multi,
    run_kmc_export_bin_single,
)

from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.progress import ProgressAdapter
from ._common.threads import detect_effective_threads


def _normalize_prefix(path_or_prefix: str) -> str:
    p = str(path_or_prefix).strip()
    if p == "":
        raise ValueError("KMC prefix cannot be empty.")
    low = p.lower()
    if low.endswith(".kmc_pre"):
        return p[: -len(".kmc_pre")]
    if low.endswith(".kmc_suf"):
        return p[: -len(".kmc_suf")]
    return p


def _validate_kmc_prefix(prefix: str) -> None:
    pre = Path(f"{prefix}.kmc_pre")
    suf = Path(f"{prefix}.kmc_suf")
    if not pre.is_file() or not suf.is_file():
        raise FileNotFoundError(
            f"KMC database not complete for prefix '{prefix}'. "
            f"Expected files: {pre} and {suf}."
        )


def _default_sample_id(prefix: str) -> str:
    name = Path(str(prefix)).name
    name = re.sub(r"[^\w.\-]+", "_", str(name)).strip("._-")
    return name if name else "sample"


def _resolve_output(outdir: str, prefix: str) -> str:
    base = str(Path(outdir).expanduser().resolve() / prefix)
    return f"{base}.bin"


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx kmerge",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx kmerge -db s1 s2 s3 -o out -prefix merged_bin",
                "jx kmerge -db sampleA.kmc_pre sampleB -o out -prefix panel",
                "jx kmerge -db sampleA sampleB -o out -prefix merged",
            ]
        ),
        description=(
            "Merge multiple KMC databases into a multi-sample k-mer matrix "
            "and write binary presence/absence output (.bin + .bin.id + .bin.site)."
        ),
    )
    required = parser.add_argument_group("Required arguments")
    optional = parser.add_argument_group("Optional arguments")

    required.add_argument(
        "-db",
        "--db",
        nargs="+",
        required=True,
        help="Input KMC database prefixes (or .kmc_pre/.kmc_suf paths).",
    )
    optional.add_argument(
        "-sid",
        "--sample-id",
        nargs="+",
        default=None,
        help="Sample IDs in the same order as -db (default: derived from prefix name).",
    )
    optional.add_argument(
        "-o",
        "--out",
        type=str,
        default=".",
        help="Output directory for results (default: .).",
    )
    optional.add_argument(
        "-prefix",
        "--prefix",
        type=str,
        default="kmerge",
        help="Prefix for output files (default: kmerge).",
    )
    optional.add_argument(
        "-t",
        "--thread",
        metavar="THREAD",
        type=int,
        default=8,
        help="Number of CPU threads for native kmerge encoding (default: 8).",
    )
    optional.add_argument(
        "--max-kmers",
        type=int,
        default=0,
        help="Debug cap for loaded k-mers per sample (0 means no cap).",
    )
    optional.add_argument(
        "--max-output-sites",
        type=int,
        default=0,
        help=(
            "Safety guard for single-sample exports: abort when total k-mers exceed this "
            "site count (default: disabled). Set <=0 to disable."
        ),
    )
    optional.add_argument(
        "--allow-large-output",
        action="store_true",
        help="Bypass single-sample large-output safety guard.",
    )
    return parser


def _estimate_single_sample_output_bytes(
    *,
    n_sites: int,
    kmer_len: int,
    n_samples: int = 1,
) -> int:
    # Rough order-of-magnitude estimator for direct bin output safety guard.
    if n_sites <= 0:
        return 0
    bin_mat = int(n_sites) * ((int(n_samples) + 7) // 8) + 32  # JXBIN001 header + bit matrix
    site_row = 2 + ((int(kmer_len) + 3) // 4)  # u16 length + packed 2-bit sequence
    site = int(n_sites) * int(site_row) + 24  # JXBSITE1 header
    # ID sidecar is tiny in practice; keep a conservative fixed allowance.
    id_sidecar = 256 + int(n_samples) * 64
    return bin_mat + site + id_sidecar


def _fmt_size(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(max(0, int(n_bytes)))
    u = 0
    while v >= 1024.0 and u < len(units) - 1:
        v /= 1024.0
        u += 1
    return f"{v:.1f}{units[u]}"


def _guard_large_single_sample_output(
    *,
    kmc_prefix: str,
    max_output_sites: int,
    allow_large_output: bool,
    kmc_src: str | None,
    rebuild_bind: bool,
    verbose_build: bool,
) -> None:
    if int(max_output_sites) <= 0 or bool(allow_large_output):
        return
    info = run_kmc_db_info(
        kmc_prefix=str(kmc_prefix),
        kmc_src=kmc_src,
        rebuild_bind=bool(rebuild_bind),
        verbose_build=bool(verbose_build),
    )
    n_sites = int(info.get("total_kmers", 0))
    klen = int(info.get("kmer_length", 31))
    if n_sites <= int(max_output_sites):
        return
    est = _estimate_single_sample_output_bytes(
        n_sites=n_sites,
        kmer_len=klen,
        n_samples=1,
    )
    raise RuntimeError(
        "Refusing large single-sample kmerge output for safety: "
        f"estimated sites={n_sites:,} exceeds --max-output-sites={int(max_output_sites):,}, "
        f"estimated output size~{_fmt_size(est)} (fmt=bin). "
        "Use --max-kmers to downsample, "
        "or pass --allow-large-output if you really want to run this."
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    db_inputs = [str(x) for x in args.db]
    if len(db_inputs) == 0:
        parser.error("No input KMC database provided.")
    max_kmers = max(0, int(args.max_kmers))
    detected_threads = int(detect_effective_threads())
    requested_threads = int(args.thread)
    threads = max(1, requested_threads)
    if threads > detected_threads:
        print(
            f"Warning: Requested threads={threads} exceeds detected available={detected_threads}; "
            f"using {detected_threads}.",
            flush=True,
        )
        threads = detected_threads

    db_prefixes = [_normalize_prefix(x) for x in db_inputs]
    for p in db_prefixes:
        _validate_kmc_prefix(p)

    if args.sample_id is None:
        sample_ids = [_default_sample_id(x) for x in db_prefixes]
    else:
        sample_ids = [str(x).strip() for x in args.sample_id]
        if len(sample_ids) != len(db_prefixes):
            parser.error(
                f"--sample-id count mismatch: got {len(sample_ids)}, expected {len(db_prefixes)}."
            )
        if any(x == "" for x in sample_ids):
            parser.error("--sample-id cannot contain empty value.")

    out_dir = str(Path(args.out).expanduser().resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix).strip()
    if prefix == "":
        parser.error("-prefix/--prefix cannot be empty.")
    out_target = _resolve_output(out_dir, prefix)

    kmc_src = str(os.environ.get("JANUSX_KMC_SRC", "")).strip() or None
    rebuild_bind = str(os.environ.get("JANUSX_KMC_REBUILD", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    verbose_build = str(os.environ.get("JANUSX_BUILD_VERBOSE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    print(
        f"Running KMC merge: samples={len(db_prefixes)}, fmt=bin, "
        f"max_kmers={max_kmers if max_kmers > 0 else 'all'}, threads={threads}",
        flush=True,
    )

    # Single-sample streaming fast path:
    # avoid loading all kmers/counts into Python objects.
    if len(db_prefixes) == 1 and max_kmers == 0:
        sid0 = sample_ids[0]
        pfx0 = db_prefixes[0]
        info0 = run_kmc_db_info(
            kmc_prefix=pfx0,
            kmc_src=kmc_src,
            rebuild_bind=bool(rebuild_bind),
            verbose_build=bool(verbose_build),
        )
        _guard_large_single_sample_output(
            kmc_prefix=pfx0,
            max_output_sites=int(args.max_output_sites),
            allow_large_output=bool(args.allow_large_output),
            kmc_src=kmc_src,
            rebuild_bind=bool(rebuild_bind),
            verbose_build=bool(verbose_build),
        )
        native_total_records = max(0, int(info0.get("total_kmers", 0)))
        merge_total = max(1, native_total_records)
        export_pbar = ProgressAdapter(
            total=merge_total,
            desc=f"Exporting KMC sample (n=1)",
            show_spinner=False,
            show_postfix=False,
            keep_display=True,
            emit_done=False,
            force_animate=True,
        )
        progressed = 0
        export_pbar_closed = False

        def _on_progress_single(processed_records: int, union_sites: int, total_records: int) -> None:
            nonlocal progressed
            target = min(merge_total, max(0, int(processed_records)))
            if target > progressed:
                export_pbar.update(target - progressed)
                progressed = target

        out_prefix = str(Path(out_target).with_suffix(""))
        try:
            export_info = run_kmc_export_bin_single(
                kmc_prefix=pfx0,
                out_prefix=out_prefix,
                sample_id=sid0,
                progress_callback=_on_progress_single,
                progress_every=max(10000, merge_total // 200),
                threads=threads,
                kmc_src=kmc_src,
                rebuild_bind=bool(rebuild_bind),
                verbose_build=bool(verbose_build),
            )
            if progressed < merge_total and native_total_records > 0:
                export_pbar.update(merge_total - progressed)
                progressed = merge_total
            export_pbar.finish()
            export_pbar.close()
            export_pbar_closed = True
            n_sites = int(export_info.get("n_kmers", 0))
            if n_sites <= 0:
                raise RuntimeError(
                    f"No k-mers exported from KMC database: {pfx0}. "
                    "Check upstream kmer counting output."
                )
            print(
                "Output files:\n"
                f"  {str(export_info.get('bin', out_target))}\n"
                f"  {str(export_info.get('id', f'{out_prefix}.bin.id'))}\n"
                f"  {str(export_info.get('site', f'{out_prefix}.bin.site'))}",
                flush=True,
            )
            return 0
        finally:
            if not export_pbar_closed:
                export_pbar.close()

    # Native k-way merge path (mandatory): covers multi-sample and single-sample with --max-kmers.
    out_prefix = str(Path(out_target).with_suffix(""))
    native_total_records = 0
    for prefix_i in db_prefixes:
        info_i = run_kmc_db_info(
            kmc_prefix=prefix_i,
            kmc_src=kmc_src,
            rebuild_bind=bool(rebuild_bind),
            verbose_build=bool(verbose_build),
        )
        n_i = int(info_i.get("total_kmers", 0))
        if max_kmers > 0:
            n_i = min(n_i, max_kmers)
        native_total_records += max(0, n_i)
    merge_total = max(1, int(native_total_records))
    merge_pbar = ProgressAdapter(
        total=merge_total,
        desc=f"Merging KMC samples (n={len(db_prefixes)})",
        show_spinner=False,
        show_postfix=False,
        keep_display=True,
        emit_done=False,
        force_animate=True,
    )
    progressed = 0
    merge_pbar_closed = False
    benchmark_fraction = 0.01
    benchmark_enabled = 3 <= len(db_prefixes) <= 8 and native_total_records > 0
    benchmark_total = 0
    benchmark_pbar = None
    benchmark_progressed = 0
    benchmark_pbar_closed = True
    if benchmark_enabled:
        benchmark_total = max(1, min(int(native_total_records), int(native_total_records * benchmark_fraction)))
        benchmark_pbar = ProgressAdapter(
            total=benchmark_total,
            desc="Benchmarking merge strategy (1% sample)",
            show_spinner=False,
            show_postfix=False,
            keep_display=True,
            emit_done=False,
            force_animate=True,
        )
        benchmark_pbar_closed = False

    def _on_progress(processed_records: int, union_sites: int, total_records: int) -> None:
        nonlocal progressed
        target = min(merge_total, max(0, int(processed_records)))
        if target > progressed:
            merge_pbar.update(target - progressed)
            progressed = target

    def _on_benchmark_progress(processed_records: int, total_records: int, status_code: int) -> None:
        nonlocal benchmark_progressed, benchmark_pbar_closed
        if benchmark_pbar is None:
            return
        local_total = benchmark_total
        target = min(local_total, max(0, int(processed_records)))
        if target > benchmark_progressed:
            benchmark_pbar.update(target - benchmark_progressed)
            benchmark_progressed = target
        code = int(status_code)
        if code > 0 and not benchmark_pbar_closed:
            if benchmark_progressed < local_total:
                benchmark_pbar.update(local_total - benchmark_progressed)
                benchmark_progressed = local_total
            benchmark_pbar.finish()
            benchmark_pbar.close()
            benchmark_pbar_closed = True
            strategy_map = {
                1: "linear_scan",
                2: "loser_tree",
                3: "two_way_direct",
            }
            selected = strategy_map.get(code, "loser_tree")
            print(f"Benchmark selected strategy: {selected}", flush=True)

    try:
        print(
            f"Native k-way merge enabled for {len(db_prefixes)} samples.",
            flush=True,
        )
        export_info = run_kmc_export_bin_multi(
            kmc_prefixes=db_prefixes,
            out_prefix=out_prefix,
            sample_ids=sample_ids,
            max_kmers=max_kmers,
            progress_callback=_on_progress,
            progress_every=max(10000, merge_total // 200),
            benchmark_callback=_on_benchmark_progress if benchmark_enabled else None,
            benchmark_progress_every=max(1000, benchmark_total // 200) if benchmark_enabled else 5000,
            benchmark_fraction=benchmark_fraction if benchmark_enabled else 0.0,
            threads=threads,
            kmc_src=kmc_src,
            rebuild_bind=bool(rebuild_bind),
            verbose_build=bool(verbose_build),
        )
        if benchmark_pbar is not None and not benchmark_pbar_closed:
            if benchmark_progressed < benchmark_total:
                benchmark_pbar.update(benchmark_total - benchmark_progressed)
                benchmark_progressed = benchmark_total
            benchmark_pbar.finish()
            benchmark_pbar.close()
            benchmark_pbar_closed = True
        if progressed < merge_total and native_total_records > 0:
            merge_pbar.update(merge_total - progressed)
            progressed = merge_total
        merge_pbar.finish()
        merge_pbar.close()
        merge_pbar_closed = True
        n_sites = int(export_info.get("n_kmers", 0))
        if n_sites <= 0:
            raise RuntimeError(
                "No non-zero k-mers found across all input KMC databases. "
                "Check upstream `jx kmer` outputs."
            )
        merge_strategy = str(export_info.get("merge_strategy", "")).strip()
        if merge_strategy != "":
            msg = f"Merge strategy selected: {merge_strategy}"
            bench_linear = export_info.get("bench_linear_ns_per_record", None)
            bench_loser_tree = export_info.get("bench_loser_tree_ns_per_record", None)
            if bench_linear is not None or bench_loser_tree is not None:
                parts = []
                if bench_linear is not None:
                    parts.append(f"linear={float(bench_linear):.1f} ns/rec")
                if bench_loser_tree is not None:
                    parts.append(f"loser_tree={float(bench_loser_tree):.1f} ns/rec")
                if parts:
                    msg += " (" + ", ".join(parts) + ")"
            print(msg, flush=True)
        print(
            f"Merged union size: {n_sites} k-mers across {len(sample_ids)} samples.",
            flush=True,
        )
        print(
            "Output files:\n"
            f"  {str(export_info.get('bin', out_target))}\n"
            f"  {str(export_info.get('id', f'{out_prefix}.bin.id'))}\n"
            f"  {str(export_info.get('site', f'{out_prefix}.bin.site'))}",
            flush=True,
        )
        return 0
    finally:
        if not merge_pbar_closed:
            merge_pbar.close()
        if benchmark_pbar is not None and not benchmark_pbar_closed:
            benchmark_pbar.close()


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
