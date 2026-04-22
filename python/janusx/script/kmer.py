from __future__ import annotations

import argparse
import gzip
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

from janusx.kmc_bind import run_kmc_count

from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.status import CliStatus, stdout_is_tty
from ._common.threads import detect_effective_threads


def _infer_input_type(paths: list[str]) -> str:
    if not paths:
        raise ValueError("No input file provided.")
    first = str(paths[0])
    low = first.lower()
    if low.endswith((".fq", ".fastq", ".fq.gz", ".fastq.gz")):
        return "fastq"
    if low.endswith((".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz")):
        return "multiline-fasta"

    opener = gzip.open if low.endswith(".gz") else open
    try:
        with opener(first, "rt", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = str(line).strip()
                if s == "":
                    continue
                if s.startswith(">"):
                    return "multiline-fasta"
                if s.startswith("@"):
                    return "fastq"
                break
    except Exception:
        pass
    raise ValueError("Cannot infer input type automatically from filename/content.")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if raw == "":
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _default_prefix_from_input(paths: list[str]) -> str:
    if not paths:
        return "kmer"
    name = Path(str(paths[0])).name
    lower = name.lower()
    if lower.endswith(".gz"):
        name = name[:-3]
        lower = name.lower()
    for ext in (".fastq", ".fq", ".fasta", ".fa", ".fna"):
        if lower.endswith(ext):
            name = name[: -len(ext)]
            break
    name = re.sub(r"[^\w.\-]+", "_", str(name)).strip("._-")
    return name if name else "kmer"


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


def _memory_limit_kb(limit_mem_gb: float | int | None) -> float:
    if limit_mem_gb is None:
        return math.nan
    try:
        gb = float(limit_mem_gb)
    except Exception:
        return math.nan
    if (not math.isfinite(gb)) or gb <= 0:
        return math.nan
    return float(gb * 1024.0 * 1024.0)


def _proc_tree_rss_kb(root_pid: int) -> float:
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return math.nan
    if proc.returncode != 0:
        return math.nan

    children: dict[int, list[int]] = {}
    rss_kb: dict[int, float] = {}
    for raw in str(proc.stdout or "").splitlines():
        parts = raw.strip().split()
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss = float(parts[2])
        except Exception:
            continue
        rss_kb[pid] = rss
        children.setdefault(ppid, []).append(pid)

    total = 0.0
    stack = [int(root_pid)]
    seen: set[int] = set()
    while len(stack) > 0:
        pid = int(stack.pop())
        if pid in seen:
            continue
        seen.add(pid)
        total += float(rss_kb.get(pid, 0.0))
        stack.extend(children.get(pid, []))
    return float(total)


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(int(proc.pid), signal.SIGKILL)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _resolve_tool_binary(
    *,
    explicit: str | None,
    env_vars: tuple[str, ...],
    path_names: tuple[str, ...],
    label: str,
) -> str:
    candidates: list[str] = []
    if explicit and str(explicit).strip() != "":
        candidates.append(str(explicit).strip())
    for name in env_vars:
        v = str(os.environ.get(name, "")).strip()
        if v != "":
            candidates.append(v)

    for cand in candidates:
        p = Path(cand).expanduser()
        if p.is_file():
            return str(p.resolve())
        hit = shutil.which(cand)
        if hit:
            return str(hit)

    for name in path_names:
        hit = shutil.which(name)
        if hit:
            return str(hit)

    env_text = ", ".join(env_vars)
    path_text = ", ".join(path_names)
    raise RuntimeError(
        f"{label} not found. Please install the required tool and configure executables.\n"
        f"- Set environment variable(s): {env_text}\n"
        f"- Or add executable(s) to PATH: {path_text}\n"
        "Download from the official tool repository, then re-run."
    )


def _default_species_ids_from_input(paths: list[str]) -> list[str]:
    ids: list[str] = []
    used: set[str] = set()
    for idx, raw in enumerate(paths, start=1):
        name = Path(raw).name
        low = name.lower()
        if low.endswith(".gz"):
            name = name[:-3]
            low = name.lower()
        for ext in (".fastq", ".fq", ".fasta", ".fa", ".fna"):
            if low.endswith(ext):
                name = name[: -len(ext)]
                break
        base = re.sub(r"[^\w.\-]+", "_", str(name)).strip("._-")
        if base == "":
            base = f"sample_{idx}"
        sid = base
        dup = 2
        while sid in used:
            sid = f"{base}_{dup}"
            dup += 1
        used.add(sid)
        ids.append(sid)
    return ids


def _run_subprocess_checked(
    *,
    cmd: list[str],
    label: str,
    use_spinner: bool,
    cwd: Path | None = None,
    limit_mem_gb: float | int | None = None,
) -> None:
    limit_kb = _memory_limit_kb(limit_mem_gb)

    def _exec_once() -> subprocess.CompletedProcess[str]:
        if not (math.isfinite(limit_kb) and float(limit_kb) > 0.0):
            return subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        mem_killed = False
        peak_rss_kb = math.nan
        while True:
            rss_now = _proc_tree_rss_kb(int(proc.pid))
            if math.isfinite(rss_now):
                peak_rss_kb = (
                    float(rss_now)
                    if not math.isfinite(peak_rss_kb)
                    else max(float(peak_rss_kb), float(rss_now))
                )
            rc_poll = proc.poll()
            if rc_poll is not None:
                break
            if math.isfinite(rss_now) and float(rss_now) > float(limit_kb):
                mem_killed = True
                _kill_process_group(proc)
                break
            time.sleep(0.2)
        out, err = proc.communicate()
        rc = int(proc.returncode) if proc.returncode is not None else (137 if mem_killed else 1)
        if mem_killed:
            peak_mb = float(peak_rss_kb) / 1024.0 if math.isfinite(peak_rss_kb) else math.nan
            limit_gb = float(limit_mem_gb) if limit_mem_gb is not None else math.nan
            extra = ""
            if math.isfinite(peak_mb):
                extra = f", peak_rss={peak_mb:.1f}MB"
            raise RuntimeError(
                f"{label} failed: memory limit exceeded ({limit_gb:.3f}GB){extra}.\n"
                f"Command: {' '.join(shlex.quote(x) for x in cmd)}"
            )
        return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout=out, stderr=err)

    if use_spinner:
        with CliStatus(f"{label}...", enabled=True, use_process=True) as task:
            try:
                proc = _exec_once()
            except Exception:
                task.fail(f"{label} ...Failed")
                raise
            if proc.returncode != 0:
                task.fail(f"{label} ...Failed")
    else:
        proc = _exec_once()

    if proc.returncode != 0:
        stdout_txt = str(proc.stdout or "").strip()
        stderr_txt = str(proc.stderr or "").strip()
        msg = (
            f"{label} failed (exit={int(proc.returncode)}).\n"
            f"Command: {' '.join(shlex.quote(x) for x in cmd)}"
        )
        if stdout_txt != "":
            msg += f"\nSTDOUT:\n{stdout_txt}"
        if stderr_txt != "":
            msg += f"\nSTDERR:\n{stderr_txt}"
        raise RuntimeError(msg)


def _run_waster_tree_pipeline(
    *,
    input_files: list[str],
    out_dir: Path,
    prefix: str,
    threads: int,
    limit_mem_gb: float | int,
    waster_mode: int,
    waster_sampled: int,
    waster_qcs: int,
    waster_qcn: int,
    waster_pattern: int,
    waster_consensus: int,
    waster_continue_file: str | None,
    waster_bin: str,
    use_spinner: bool,
) -> dict[str, str]:
    work_dir = out_dir / f"{prefix}.waster_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    waster_input_tsv = work_dir / f"{prefix}.waster_input.tsv"
    waster_snp_fa = out_dir / f"{prefix}.waster.snps.fa"
    waster_newick = out_dir / f"{prefix}.waster.nw"
    waster_mode2_patterns = out_dir / f"{prefix}.waster.patterns"

    species_ids = _default_species_ids_from_input(input_files)
    waster_input_tsv.write_text(
        "".join(f"{sid}\t{path}\n" for sid, path in zip(species_ids, input_files)),
        encoding="utf-8",
    )

    print(
        f"Running WASTER workflow: mode={int(waster_mode)}, inputs={len(input_files)}, "
        f"threads={int(threads)}, limit_mem_gb={float(limit_mem_gb):.1f}",
        flush=True,
    )
    print(
        "WASTER command profile: "
        "--mode/--sampled/--qcs/--qcn/--pattern/--consensus/-t (with optional --continue)",
        flush=True,
    )

    def _base_args() -> list[str]:
        cmd = [
            str(waster_bin),
            "--sampled",
            str(int(waster_sampled)),
            "--qcs",
            str(int(waster_qcs)),
            "--qcn",
            str(int(waster_qcn)),
            "--pattern",
            str(int(waster_pattern)),
            "--consensus",
            str(int(waster_consensus)),
            "-t",
            str(int(threads)),
        ]
        if waster_continue_file:
            cmd.extend(["--continue", str(waster_continue_file)])
        return cmd

    if int(waster_mode) == 4:
        # Mode 4 expects SNP alignment input, so first build SNP alignment via mode 3.
        cmd_mode3 = _base_args() + [
            "--mode",
            "3",
            "-i",
            str(waster_input_tsv),
            "-o",
            str(waster_snp_fa),
        ]
        _run_subprocess_checked(
            cmd=cmd_mode3,
            label="Running WASTER mode=3 (build SNP alignment)",
            use_spinner=use_spinner,
            cwd=work_dir,
            limit_mem_gb=limit_mem_gb,
        )
        if not waster_snp_fa.is_file():
            raise RuntimeError(
                "WASTER mode=3 finished but SNP output file was not produced:\n"
                f"  expected: {waster_snp_fa}"
            )

        cmd_mode4 = _base_args() + [
            "--mode",
            "4",
            "-i",
            str(waster_snp_fa),
            "-o",
            str(waster_newick),
        ]
        _run_subprocess_checked(
            cmd=cmd_mode4,
            label="Running WASTER mode=4 (build Newick)",
            use_spinner=use_spinner,
            cwd=work_dir,
            limit_mem_gb=limit_mem_gb,
        )
        if not waster_newick.is_file():
            raise RuntimeError(
                "WASTER mode=4 finished but Newick file was not produced:\n"
                f"  expected: {waster_newick}"
            )
    elif int(waster_mode) == 3:
        cmd_mode3 = _base_args() + [
            "--mode",
            "3",
            "-i",
            str(waster_input_tsv),
            "-o",
            str(waster_snp_fa),
        ]
        _run_subprocess_checked(
            cmd=cmd_mode3,
            label="Running WASTER mode=3 (build SNP alignment)",
            use_spinner=use_spinner,
            cwd=work_dir,
            limit_mem_gb=limit_mem_gb,
        )
        if not waster_snp_fa.is_file():
            raise RuntimeError(
                "WASTER mode=3 finished but SNP output file was not produced:\n"
                f"  expected: {waster_snp_fa}"
            )
    elif int(waster_mode) == 2:
        cmd_mode2 = _base_args() + [
            "--mode",
            "2",
            "-i",
            str(waster_input_tsv),
            "-o",
            str(waster_mode2_patterns),
        ]
        _run_subprocess_checked(
            cmd=cmd_mode2,
            label="Running WASTER mode=2 (build frequent patterns)",
            use_spinner=use_spinner,
            cwd=work_dir,
            limit_mem_gb=limit_mem_gb,
        )
    else:
        cmd_mode1 = _base_args() + [
            "--mode",
            "1",
            "-i",
            str(waster_input_tsv),
            "-o",
            str(waster_newick),
        ]
        _run_subprocess_checked(
            cmd=cmd_mode1,
            label="Running WASTER mode=1 (direct Newick)",
            use_spinner=use_spinner,
            cwd=work_dir,
            limit_mem_gb=limit_mem_gb,
        )
        if not waster_newick.is_file():
            raise RuntimeError(
                "WASTER mode=1 finished but Newick file was not produced:\n"
                f"  expected: {waster_newick}"
            )

    out: dict[str, str] = {
        "waster_input_tsv": str(waster_input_tsv),
        "waster_mode": str(int(waster_mode)),
    }
    if waster_snp_fa.is_file():
        out["waster_snp_fa"] = str(waster_snp_fa)
    if waster_newick.is_file():
        out["waster_newick"] = str(waster_newick)
    if waster_mode2_patterns.is_file():
        out["waster_patterns"] = str(waster_mode2_patterns)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx kmer",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx kmer -fa sample.fastq.gz --count -o out -prefix sample_k19 -k 19 -t 8",
                "jx kmer -fa species1.fa species2.fa --tree -o out -prefix species_tree -t 8 --waster-mode 4",
                "JANUSX_WASTER_BIN=/path/waster jx kmer -fa *.fa --tree -o out",
                "Common params: -fa/-t/-o/-prefix/-k/-limit-mem",
                "Count-only params (--count): -ci/-cx/--counter-max/--tmp-dir",
                "Tree-only params (--tree): --waster-mode/--waster-sampled/--waster-qcs/--waster-qcn/--waster-pattern/--waster-consensus/--waster-continue-file",
            ]
        ),
        description=(
            "K-mer workflow entry with FASTA/FASTQ input. "
            "Use --count for KMC databases (<prefix>.kmc_pre/.kmc_suf), "
            "or --tree for WASTER reference-free Newick."
        ),
    )
    common_group = parser.add_argument_group("Common arguments (all workflows)")
    workflow = parser.add_argument_group("Workflow selector")
    count_group = parser.add_argument_group("Kmer count optional arguments")
    tree_group = parser.add_argument_group("WASTER optional arguments")

    common_group.add_argument(
        "-fa",
        "--fa",
        nargs="+",
        required=True,
        help="Input FASTQ/FASTA files (one or more). Each file is treated as one sample/species.",
    )

    workflow.add_argument(
        "-count",
        "--count",
        action="store_true",
        help="Use KMC counting workflow (default when neither --count nor --tree is set).",
    )
    workflow.add_argument(
        "-tree",
        "--tree",
        action="store_true",
        help="Use WASTER workflow to infer reference-free phylogeny files.",
    )

    common_group.add_argument(
        "-t",
        "--thread",
        metavar="THREAD",
        type=int,
        default=8,
        help="Number of CPU threads (default: 8).",
    )
    common_group.add_argument(
        "-o",
        "--out",
        type=str,
        default=".",
        help="Output directory for results (default: .).",
    )
    common_group.add_argument(
        "-prefix",
        "--prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: inferred from first input).",
    )
    common_group.add_argument(
        "-k",
        "--kmer-len",
        type=int,
        default=19,
        help="K-mer length parameter for KMC count workflow (default: 19).",
    )
    common_group.add_argument(
        "-limit-mem",
        "--limit-mem",
        dest="limit_mem_gb",
        type=int,
        default=8,
        help=(
            "Memory limit in GB (default: 8). "
            "For --count: passed to KMC memory cap. "
            "For --tree: subprocess memory watchdog kills task on limit exceed."
        ),
    )
    common_group.add_argument(
        "-m",
        "--max-ram-gb",
        dest="limit_mem_gb",
        type=int,
        help=argparse.SUPPRESS,
    )

    count_group.add_argument(
        "-ci",
        "--cutoff-min",
        type=int,
        default=2,
        help="Minimal k-mer count cutoff (default: 2).",
    )
    count_group.add_argument(
        "-cx",
        "--cutoff-max",
        type=int,
        default=1_000_000_000,
        help="Maximal k-mer count cutoff (default: 1000000000).",
    )
    count_group.add_argument(
        "--counter-max",
        type=int,
        default=255,
        help="Maximum stored counter value (default: 255).",
    )
    count_group.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Temporary directory for KMC intermediate files (default: <out>.kmc_tmp).",
    )

    tree_group.add_argument(
        "--waster-mode",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help=(
            "WASTER mode (default: 4). "
            "1=whole inference, 2=patterns only, 3=SNPs only, 4=start from SNPs to Newick."
        ),
    )
    tree_group.add_argument(
        "--waster-sampled",
        type=int,
        default=16,
        help="Maximum sampled species used by WASTER (default: 16).",
    )
    tree_group.add_argument(
        "--waster-qcs",
        type=int,
        default=30,
        help="WASTER SNP-base quality threshold for FASTQ inputs (default: 30).",
    )
    tree_group.add_argument(
        "--waster-qcn",
        type=int,
        default=20,
        help="WASTER non-SNP quality threshold for FASTQ inputs (default: 20).",
    )
    tree_group.add_argument(
        "--waster-pattern",
        type=int,
        default=500_000_000,
        help="WASTER number of frequent patterns (default: 500000000).",
    )
    tree_group.add_argument(
        "--waster-consensus",
        type=int,
        default=25_000_000,
        help="WASTER number of consensus profiles (default: 25000000).",
    )
    tree_group.add_argument(
        "--waster-continue-file",
        type=str,
        default=None,
        help="Optional WASTER --continue file path for reusing frequent patterns.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    mode_count = bool(args.count)
    mode_tree = bool(args.tree)
    if mode_count and mode_tree:
        parser.error("--count and --tree are mutually exclusive.")
    if not mode_count and not mode_tree:
        mode_count = True

    if int(args.limit_mem_gb) < 2:
        parser.error("-limit-mem/--limit-mem must be >= 2.")
    if int(args.kmer_len) <= 0:
        parser.error("-k/--kmer-len must be > 0.")

    input_files = [str(Path(x).expanduser()) for x in args.fa]
    for p in input_files:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Input file not found: {p}")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prefix is None:
        prefix = _default_prefix_from_input(input_files)
    else:
        prefix = str(args.prefix).strip()
    if prefix == "":
        raise ValueError("Output prefix cannot be empty.")
    output_prefix = _normalize_prefix(str(out_dir / prefix))
    out_parent = Path(output_prefix).parent
    if str(out_parent) not in {"", "."}:
        out_parent.mkdir(parents=True, exist_ok=True)

    detected_threads = int(detect_effective_threads())
    requested_threads = int(args.thread)
    threads = max(1, int(requested_threads))
    if threads > detected_threads:
        print(
            f"Warning: Requested threads={threads} exceeds detected available={detected_threads}; "
            f"using {detected_threads}.",
            flush=True,
        )
        threads = int(detected_threads)

    use_spinner = bool(stdout_is_tty())

    if mode_count:
        canonical = _env_flag("JANUSX_KMER_CANONICAL", default=True)
        kmc_src = str(os.environ.get("JANUSX_KMC_SRC", "")).strip() or None
        rebuild_bind = _env_flag("JANUSX_KMC_REBUILD", default=False)
        verbose_build = _env_flag("JANUSX_BUILD_VERBOSE", default=False)

        tmp_dir = str(Path(args.tmp_dir).expanduser()) if args.tmp_dir else f"{output_prefix}.kmc_tmp"
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        input_type = _infer_input_type(input_files)

        print(
            f"Running KMC count: inputs={len(input_files)}, k={int(args.kmer_len)}, "
            f"threads={threads}, limit_mem_gb={int(args.limit_mem_gb)}, input_type={input_type}",
            flush=True,
        )

        if use_spinner:
            with CliStatus("Running KMC count...", enabled=True, use_process=True) as task:
                try:
                    stats = run_kmc_count(
                        input_files=input_files,
                        output_prefix=output_prefix,
                        tmp_dir=tmp_dir,
                        kmer_len=int(args.kmer_len),
                        threads=threads,
                        max_ram_gb=int(args.limit_mem_gb),
                        cutoff_min=int(args.cutoff_min),
                        cutoff_max=int(args.cutoff_max),
                        counter_max=int(args.counter_max),
                        canonical=canonical,
                        input_type=input_type,
                        kmc_src=kmc_src,
                        rebuild_bind=bool(rebuild_bind),
                        verbose_build=bool(verbose_build),
                    )
                except Exception:
                    task.fail("Running KMC count ...Failed")
                    raise
        else:
            stats = run_kmc_count(
                input_files=input_files,
                output_prefix=output_prefix,
                tmp_dir=tmp_dir,
                kmer_len=int(args.kmer_len),
                threads=threads,
                max_ram_gb=int(args.limit_mem_gb),
                cutoff_min=int(args.cutoff_min),
                cutoff_max=int(args.cutoff_max),
                counter_max=int(args.counter_max),
                canonical=canonical,
                input_type=input_type,
                kmc_src=kmc_src,
                rebuild_bind=bool(rebuild_bind),
                verbose_build=bool(verbose_build),
            )

        print(
            f"KMC finished: stage1={float(stats.get('stage1_time_s', 0.0)):.3f}s, "
            f"stage2={float(stats.get('stage2_time_s', 0.0)):.3f}s, "
            f"unique={int(stats.get('n_unique_kmers', 0))}, total={int(stats.get('n_total_kmers', 0))}"
        )
        if int(stats.get("n_total_kmers", 0)) == 0:
            print(
                "! Warning: KMC output is empty (total_kmers=0). "
                "Try smaller -k and/or lower -ci (e.g. -ci 1), or provide longer input sequences.",
                flush=True,
            )
        print(f"Output files:\n  {output_prefix}.kmc_pre\n  {output_prefix}.kmc_suf")
        return 0

    if int(args.waster_sampled) <= 0:
        parser.error("--waster-sampled must be > 0.")
    if int(args.waster_pattern) <= 0:
        parser.error("--waster-pattern must be > 0.")
    if int(args.waster_consensus) <= 0:
        parser.error("--waster-consensus must be > 0.")
    if not (0 <= int(args.waster_qcs) <= 93):
        parser.error("--waster-qcs must be in [0, 93].")
    if not (0 <= int(args.waster_qcn) <= 93):
        parser.error("--waster-qcn must be in [0, 93].")

    waster_bin = _resolve_tool_binary(
        explicit=None,
        env_vars=("JANUSX_WASTER_BIN",),
        path_names=("waster", "WASTER"),
        label="WASTER",
    )

    tree_outputs = _run_waster_tree_pipeline(
        input_files=input_files,
        out_dir=out_dir,
        prefix=str(prefix),
        threads=int(threads),
        limit_mem_gb=int(args.limit_mem_gb),
        waster_mode=int(args.waster_mode),
        waster_sampled=int(args.waster_sampled),
        waster_qcs=int(args.waster_qcs),
        waster_qcn=int(args.waster_qcn),
        waster_pattern=int(args.waster_pattern),
        waster_consensus=int(args.waster_consensus),
        waster_continue_file=(
            str(Path(args.waster_continue_file).expanduser())
            if args.waster_continue_file is not None
            else None
        ),
        waster_bin=str(waster_bin),
        use_spinner=use_spinner,
    )
    out_lines = [f"  {tree_outputs['waster_input_tsv']}"]
    if "waster_snp_fa" in tree_outputs:
        out_lines.append(f"  {tree_outputs['waster_snp_fa']}")
    if "waster_patterns" in tree_outputs:
        out_lines.append(f"  {tree_outputs['waster_patterns']}")
    if "waster_newick" in tree_outputs:
        out_lines.append(f"  {tree_outputs['waster_newick']}")
    print(
        f"WASTER tree workflow finished (mode={int(args.waster_mode)}).\n"
        "Output files:\n" + "\n".join(out_lines),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
