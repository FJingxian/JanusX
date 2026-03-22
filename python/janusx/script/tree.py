# -*- coding: utf-8 -*-
"""
JanusX: tree workflow entry (NJ)

Current stage:
1) `jx tree -nj`: genotype/FASTA -> MSA -> Neighbor-Joining tree

Examples
--------
  jx tree -vcf cohort.vcf.gz -o out --prefix cohort_tree -nj
  jx tree -fa aln.fasta -o out --prefix aln_tree -nj
  jx tree -bfile panel -o out --prefix panel_tree --write-phylip -nj
  jx tree -vcf cohort.vcf.gz -o out -nj bionj
  jx tree -vcf cohort.vcf.gz -o out -nj bionj-dist
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from typing import Iterable

import numpy as np

from janusx.gfreader import inspect_genotype_file, load_genotype_chunks
from janusx.janusx import (
    geno_chunk_to_alignment_u8_siteinfo,
    ml_newick_from_alignment_u8,
    nj_newick_from_alignment_u8,
)

from ._common.genoio import determine_genotype_source_from_args
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.pathcheck import (
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
    format_path_for_display,
)
from ._common.status import format_elapsed, log_success


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    old = {k: os.environ.get(k) for k in updates}
    try:
        for k, v in updates.items():
            if v is None:
                continue
            os.environ[k] = str(v)
        yield
    finally:
        for k, old_v in old.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v

def _normalize_out_prefix(path: str) -> str:
    p = str(path).strip()
    low = p.lower()
    for ext in (".nwk", ".newick", ".fasta", ".fa", ".phy", ".phylip"):
        if low.endswith(ext):
            return p[: -len(ext)]
    return p


def _default_prefix_from_genofile(path: str) -> str:
    name = os.path.basename(str(path).strip())
    low = name.lower()
    for ext in (
        ".vcf.gz",
        ".hmp.gz",
        ".vcf",
        ".hmp",
        ".bed",
        ".bim",
        ".fam",
        ".txt",
        ".tsv",
        ".csv",
        ".npy",
    ):
        if low.endswith(ext):
            name = name[: -len(ext)]
            break
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._-")
    return name if name else "tree"


def _resolve_out_prefix(args, gfile: str) -> str:
    out_dir = os.path.abspath(str(args.out).strip() or ".")
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        raise ValueError(f"-o/--out must be an output directory, got file path: {out_dir}")
    os.makedirs(out_dir, mode=0o755, exist_ok=True)

    if args.prefix is not None and str(args.prefix).strip() != "":
        prefix = os.path.basename(str(args.prefix).strip())
        prefix = _normalize_out_prefix(prefix)
    else:
        prefix = _normalize_out_prefix(_default_prefix_from_genofile(gfile))
    if prefix == "":
        prefix = "tree"
    return os.path.join(out_dir, prefix)


def _sanitize_phylip_names(names: Iterable[str]) -> list[str]:
    used: set[str] = set()
    out: list[str] = []
    for raw in names:
        base = re.sub(r"\s+", "_", str(raw).strip())
        base = base.replace("'", "_")
        if base == "":
            base = "sample"
        name = base
        idx = 2
        while name in used:
            name = f"{base}_{idx}"
            idx += 1
        used.add(name)
        out.append(name)
    return out


def _write_fasta(path: str, sample_ids: list[str], seqs: list[bytearray]) -> None:
    with open(path, "w", encoding="utf-8") as fw:
        for sid, seq in zip(sample_ids, seqs):
            fw.write(f">{sid}\n")
            bs = bytes(seq)
            for i in range(0, len(bs), 80):
                fw.write(bs[i : i + 80].decode("ascii"))
                fw.write("\n")


def _write_phylip(path: str, sample_ids: list[str], seqs: list[bytearray]) -> None:
    names = _sanitize_phylip_names(sample_ids)
    n_taxa = len(names)
    n_sites = len(seqs[0]) if seqs else 0
    max_name = max((len(x) for x in names), default=0)
    with open(path, "w", encoding="utf-8") as fw:
        fw.write(f"{n_taxa} {n_sites}\n")
        for name, seq in zip(names, seqs):
            pad = " " * (max_name + 2 - len(name))
            fw.write(f"{name}{pad}{bytes(seq).decode('ascii')}\n")


def _build_alignment_matrix(seqs: list[bytearray]) -> np.ndarray:
    if not seqs:
        raise ValueError("No sequences available for NJ.")
    n_taxa = len(seqs)
    n_sites = len(seqs[0])
    if n_sites == 0:
        raise ValueError("Alignment has zero sites after filtering.")
    aln = np.empty((n_taxa, n_sites), dtype=np.uint8)
    for i, seq in enumerate(seqs):
        if len(seq) != n_sites:
            raise ValueError("Inconsistent sequence lengths while building alignment.")
        aln[i, :] = np.frombuffer(seq, dtype=np.uint8, count=n_sites)
    return aln


def _uniquify_sample_ids(sample_ids: list[str]) -> list[str]:
    used: set[str] = set()
    out: list[str] = []
    for raw in sample_ids:
        base = str(raw).strip() or "sample"
        name = base
        idx = 2
        while name in used:
            name = f"{base}_{idx}"
            idx += 1
        used.add(name)
        out.append(name)
    return out


def _fasta_to_alignment(path: str, logger) -> tuple[list[str], list[bytearray], int, dict[str, float]]:
    t0 = time.time()
    parsed_ids: list[str] = []
    parsed_seqs: list[str] = []
    cur_id: str | None = None
    cur_chunks: list[str] = []

    def _flush_record() -> None:
        nonlocal cur_id, cur_chunks
        if cur_id is None:
            return
        seq = "".join(cur_chunks).replace(" ", "").replace("\t", "").upper().replace("U", "T")
        if seq == "":
            raise ValueError(f"Empty FASTA sequence for sample '{cur_id}' in {path}")
        try:
            seq.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"Non-ASCII sequence for sample '{cur_id}' in {path}") from exc
        parsed_ids.append(cur_id)
        parsed_seqs.append(seq)

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if line == "":
                continue
            if line.startswith(">"):
                _flush_record()
                head = line[1:].strip()
                sid = head.split()[0] if head else f"sample_{len(parsed_ids) + 1}"
                cur_id = sid
                cur_chunks = []
                continue
            if cur_id is None:
                raise ValueError(
                    f"Malformed FASTA: sequence line before header at {path}:{line_no}"
                )
            cur_chunks.append(line)
    _flush_record()

    if len(parsed_ids) < 2:
        raise ValueError(f"Need at least 2 FASTA records for tree inference, got {len(parsed_ids)}.")

    n_sites = len(parsed_seqs[0])
    if n_sites == 0:
        raise ValueError("FASTA alignment has zero sites.")

    for idx, seq in enumerate(parsed_seqs):
        if len(seq) != n_sites:
            raise ValueError(
                f"FASTA alignment length mismatch at sample '{parsed_ids[idx]}': "
                f"got {len(seq)}, expected {n_sites}"
            )

    sample_ids = _uniquify_sample_ids(parsed_ids)
    seqs = [bytearray(seq.encode("ascii")) for seq in parsed_seqs]
    logger.info(f"Alignment ready: samples={len(sample_ids)}, sites={n_sites}")
    return sample_ids, seqs, n_sites, {"read": time.time() - t0, "convert": 0.0}


def _resolve_ml_mode(mode: str | None, asc: bool) -> tuple[str, str, str]:
    m = str(mode if mode is not None else "exact").strip().lower()
    if m not in ("exact", "approx", "compat", "compat-accurate"):
        raise ValueError(f"invalid ml mode: {mode}")
    backend_mode = m
    compat_preset = "none"
    if m == "compat":
        mode_label = "compat [fasttree-compat stage-1]"
        backend_mode = "compat"
        compat_preset = "compat"
    elif m == "compat-accurate":
        mode_label = "compat-accurate [fasttree-compat consistency-first]"
        backend_mode = "compat"
        compat_preset = "compat-accurate"
    else:
        mode_label = f"{m} [rust-jc69-nni-v1]"

    if asc:
        mode_label = f"{mode_label}+ASC(pseudo)"
    return backend_mode, mode_label, compat_preset


def _apply_asc_pseudo_constant_sites(seqs: list[bytearray]) -> tuple[list[bytearray], int]:
    k = int(os.environ.get("JANUSX_ASC_PSEUDO_CONST", "1"))
    if k <= 0:
        return seqs, 0
    tail = (b"A" * k) + (b"C" * k) + (b"G" * k) + (b"T" * k)
    out = [bytearray(s) for s in seqs]
    for s in out:
        s.extend(tail)
    return out, 4 * k


def _validate_inputs(args, logger) -> None:
    if int(args.chunksize) <= 0:
        raise ValueError("-chunksize/--chunksize must be > 0")
    if int(args.thread) < 0:
        raise ValueError("-t/--thread must be >= 0")
    if not (0.0 <= float(args.maf) <= 0.5):
        raise ValueError("-maf must be in [0, 0.5]")
    if not (0.0 <= float(args.geno) <= 1.0):
        raise ValueError("-geno/--geno must be in [0, 1]")
    if not (0.0 <= float(args.het) <= 0.5):
        raise ValueError("-het must be in [0, 0.5]")
    if int(getattr(args, "bootstrap", 0)) < 0:
        raise ValueError("-b/--bootstrap must be >= 0")

    if args.fasta:
        ok = ensure_file_exists(logger, args.fasta, "FASTA alignment")
        if not ok:
            raise FileNotFoundError(args.fasta)
    elif args.vcf:
        ok = ensure_file_exists(logger, args.vcf, "VCF file")
        if not ok:
            raise FileNotFoundError(args.vcf)
    elif args.hmp:
        ok = ensure_file_exists(logger, args.hmp, "HMP file")
        if not ok:
            raise FileNotFoundError(args.hmp)
    elif args.file:
        ok = ensure_file_input_exists(logger, args.file, "FILE genotype input")
        if not ok:
            raise FileNotFoundError(args.file)
    elif args.bfile:
        ok = ensure_plink_prefix_exists(logger, args.bfile, "PLINK prefix")
        if not ok:
            raise FileNotFoundError(args.bfile)


def _genotype_to_alignment(
    args, gfile: str, logger
) -> tuple[list[str], list[bytearray], int, dict[str, float]]:
    sample_ids, _ = inspect_genotype_file(
        gfile,
        snps_only=bool(args.snps_only),
        maf=float(args.maf),
        missing_rate=float(args.geno),
        het=float(args.het),
        delimiter=None,
    )
    n_samples = len(sample_ids)
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for tree inference, got {n_samples}.")

    seqs = [bytearray() for _ in range(n_samples)]
    total_sites = 0
    t_read = 0.0
    t_convert = 0.0

    chunk_iter = iter(
        load_genotype_chunks(
        gfile,
        chunk_size=int(args.chunksize),
        maf=float(args.maf),
        missing_rate=float(args.geno),
        impute=False,
        model="add",
        het=float(args.het),
        snps_only=bool(args.snps_only),
        delimiter=None,
        )
    )

    while True:
        t0 = time.time()
        try:
            geno_chunk, sites = next(chunk_iter)
        except StopIteration:
            break
        t_read += time.time() - t0

        t1 = time.time()
        g = np.asarray(geno_chunk, dtype=np.float32)
        if g.ndim != 2 or g.shape[0] == 0:
            continue
        m, n = int(g.shape[0]), int(g.shape[1])
        if n != n_samples:
            raise ValueError(
                f"Chunk sample size mismatch: got {n}, expected {n_samples}"
            )
        trans = np.asarray(
            geno_chunk_to_alignment_u8_siteinfo(g, sites),
            dtype=np.uint8,
        )
        for j in range(n_samples):
            seqs[j].extend(trans[j].tobytes())
        total_sites += m
        t_convert += time.time() - t1

    if total_sites == 0:
        raise ValueError("No variant sites passed filters; cannot build alignment/tree.")

    logger.info(f"Alignment ready: samples={n_samples}, sites={total_sites}")
    return sample_ids, seqs, total_sites, {"read": t_read, "convert": t_convert}


def _add_tree_input_args(parser: CliArgumentParser, *, required: bool) -> None:
    required_group = parser.add_argument_group("Required arguments")
    g = required_group.add_mutually_exclusive_group(required=required)
    g.add_argument(
        "-vcf", "--vcf", dest="vcf", type=str,
        help="Input genotype file in VCF format (.vcf or .vcf.gz).",
    )
    g.add_argument(
        "-hmp", "--hmp", dest="hmp", type=str,
        help="Input genotype file in HMP format (.hmp or .hmp.gz).",
    )
    g.add_argument(
        "-file", "--file", dest="file", type=str,
        help=(
            "Input genotype numeric matrix (.txt/.tsv/.csv/.npy) or prefix. "
            "Requires sibling prefix.id. Optional site metadata: prefix.site or prefix.bim."
        ),
    )
    g.add_argument(
        "-bfile", "--bfile", dest="bfile", type=str,
        help="Input genotype in PLINK binary format (prefix for .bed, .bim, .fam).",
    )
    g.add_argument(
        "-fa", "--fasta", dest="fasta", type=str,
        help="Input aligned FASTA file (.fa/.fasta).",
    )


def _add_tree_optional_args(
    parser: CliArgumentParser,
    *,
    include_hidden_approx_legacy: bool,
    include_ml_options: bool,
) -> None:
    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "-chunksize", "--chunksize", dest="chunksize", type=int, default=10_000,
        help="SNP chunk size for streaming read (default: %(default)s).",
    )
    optional_group.add_argument(
        "-maf", "--maf", dest="maf", type=float, default=0.02,
        help="Exclude variants with minor allele frequency lower than a threshold (default: %(default)s).",
    )
    optional_group.add_argument(
        "-geno", "--geno", dest="geno", type=float, default=0.05,
        help="Exclude variants with missing call frequencies greater than a threshold (default: %(default)s).",
    )
    optional_group.add_argument(
        "-het", "--het", dest="het", type=float, default=0.02,
        help="Heterozygosity filter threshold (default: %(default)s).",
    )
    optional_group.add_argument(
        "-snps-only", "--snps-only", dest="snps_only", action="store_true", default=False,
        help="Exclude non-SNP variants.",
    )
    if include_hidden_approx_legacy:
        optional_group.add_argument(
            "--approx", dest="approx_legacy", action="store_true", default=False,
            help=argparse.SUPPRESS,
        )
    optional_group.add_argument(
        "-t", "--thread", dest="thread", type=int, default=0,
        help="Thread count for tree inference (0=auto; default: %(default)s).",
    )
    optional_group.add_argument(
        "-o", "--out", dest="out", type=str, default=".",
        help="Output directory for results (default: %(default)s).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", dest="prefix", type=str, default=None,
        help="Output file prefix (default: inferred from genotype file name).",
    )
    optional_group.add_argument(
        "--write-phylip", dest="write_phylip", action="store_true",
        help="Also write relaxed PHYLIP alignment: <out>/<prefix>.phy",
    )
    optional_group.add_argument(
        "--profile", dest="profile", action="store_true",
        help="Report phase timings and save <out>/<prefix>.profile.tsv.",
    )
    if include_ml_options:
        optional_group.add_argument(
            "-asc", "--asc", dest="asc", action="store_true", default=False,
            help="Enable SNP ascertainment-bias pseudo correction for -ml.",
        )
        optional_group.add_argument(
            "-b", "--bootstrap", dest="bootstrap",
            nargs="?", type=int, const=100, default=0, metavar="NITER",
            help=(
                "ML support resamples for internal branch labels "
                "(0=off; '-b' means 100)."
            ),
        )
        optional_group.add_argument(
            "--support", dest="support", type=str, default="bootstrap",
            choices=["bootstrap", "shlike"],
            help=(
                "Support algorithm for -ml when -b>0: "
                "'bootstrap' (global, slower) or 'shlike' (local, faster)."
            ),
        )


def build_nj_parser(prog: str) -> CliArgumentParser:
    parser = CliArgumentParser(
        prog=prog,
        description=(
            "Tree NJ baseline: genotype/FASTA input -> DNA alignment -> "
            "Neighbor-Joining tree (Newick)."
        ),
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx tree -vcf cohort.vcf.gz -o out --prefix cohort_tree -nj",
                "jx tree -fa aln.fasta -o out --prefix aln_tree -nj",
                "jx tree -bfile panel -o out --prefix panel_tree --write-phylip -nj",
                "jx tree -vcf cohort.vcf.gz -o out -nj bionj",
                "jx tree -vcf cohort.vcf.gz -o out -nj approx",
            ]
        ),
    )
    _add_tree_input_args(parser, required=True)
    _add_tree_optional_args(
        parser,
        include_hidden_approx_legacy=True,
        include_ml_options=False,
    )
    return parser


def build_tree_parser() -> CliArgumentParser:
    parser = CliArgumentParser(
        prog="jx tree",
        description=(
            "Tree workflow entrypoint (NJ only).\n"
            "- `-nj` (optional MODE: exact|bionj|bionj-dist|bionj-jc|bionj-binom|bionj-auto|approx): "
            "run Neighbor-Joining workflow."
        ),
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx tree -vcf cohort.vcf.gz -o out --prefix cohort_tree -nj",
                "jx tree -fa aln.fasta -o out --prefix aln_tree -nj",
                "jx tree -vcf cohort.vcf.gz -o out --prefix cohort_tree -nj bionj",
                "jx tree -vcf cohort.vcf.gz -o out --prefix cohort_tree -nj bionj-dist",
                "jx tree -vcf cohort.vcf.gz -o out --prefix cohort_tree -nj approx",
                "jx tree -bfile panel -o out --prefix panel_tree --profile -nj",
            ]
        ),
    )
    _add_tree_input_args(parser, required=False)
    model_group = parser.add_argument_group("Model arguments")
    model_group.add_argument(
        "-nj", "--nj",
        nargs="?",
        const="exact",
        default=None,
        choices=[
            "exact",
            "bionj",
            "bionj-dist",
            "bionj-jc",
            "bionj-binom",
            "bionj-auto",
            "approx",
        ],
        metavar="MODE",
        help=(
            "Run Neighbor-Joining model; MODE: exact (default), approx, "
            "or bionj variants: bionj(dist default), bionj-dist, bionj-jc, "
            "bionj-binom, bionj-auto."
        ),
    )
    _add_tree_optional_args(
        parser,
        include_hidden_approx_legacy=False,
        include_ml_options=False,
    )
    return parser


def _run_nj(args, *, log_tag: str, nj_mode: str = "exact") -> None:
    requested_mode = str(nj_mode if nj_mode is not None else "exact").strip().lower()
    if bool(getattr(args, "approx_legacy", False)):
        requested_mode = "approx"
    bionj_alias_map = {
        "bionj": "bionj",
        "bionj-dist": "bionj-dist",
        "bionj-jc": "bionj-jc",
        "bionj-binom": "bionj-binom",
        "bionj-auto": "bionj-auto",
    }
    allowed_modes = {"exact", "approx", *bionj_alias_map.keys()}
    if requested_mode not in allowed_modes:
        raise ValueError(
            "`-nj` MODE must be one of: exact, approx, "
            "bionj, bionj-dist, bionj-jc, bionj-binom, bionj-auto."
        )

    if args.fasta:
        input_path = str(args.fasta)
        input_kind = "fasta"
    else:
        input_path, _ = determine_genotype_source_from_args(args)
        input_kind = "genotype"

    out_prefix = _resolve_out_prefix(args, input_path)
    logger = setup_logging(f"{out_prefix}.{log_tag}.log")
    _validate_inputs(args, logger)

    if input_kind == "fasta":
        logger.info(f"Input FASTA: {format_path_for_display(input_path)}")
    else:
        logger.info(f"Input genotype: {format_path_for_display(input_path)}")
    if int(args.thread) == 0:
        logger.info("NJ threads: auto")
    else:
        logger.info(f"NJ threads: {int(args.thread)}")
    if requested_mode == "approx":
        logger.info("NJ mode: approximate [rapid-core lowmem]")
    elif requested_mode in bionj_alias_map:
        logger.info("NJ mode: bionj [variance-weighted exact]")
        alias_mode = {
            "bionj-dist": "dist",
            "bionj-jc": "jc",
            "bionj-binom": "binom",
            "bionj-auto": "auto",
        }.get(requested_mode)
        if alias_mode is not None:
            logger.info(f"BIONJ variance mode: {alias_mode} (mode alias)")
        else:
            bionj_var_mode = (
                os.environ.get("JANUSX_BIONJ_VAR_MODE", "dist").strip().lower() or "dist"
            )
            logger.info(f"BIONJ variance mode: {bionj_var_mode} (env/default)")
    else:
        logger.info("NJ mode: exact")

    t0 = time.time()
    if input_kind == "fasta":
        sample_ids, seqs, n_sites, prof_stream = _fasta_to_alignment(input_path, logger)
    else:
        sample_ids, seqs, n_sites, prof_stream = _genotype_to_alignment(args, input_path, logger)

    t_write_fasta = 0.0
    fasta_path = f"{out_prefix}.fasta"
    fasta_in_abs = os.path.abspath(input_path) if input_kind == "fasta" else None
    fasta_out_abs = os.path.abspath(fasta_path)
    if fasta_in_abs is not None and fasta_in_abs == fasta_out_abs:
        logger.info("Input FASTA already matches output FASTA path; skipped rewrite.")
    else:
        t_write_fasta = time.time()
        _write_fasta(fasta_path, sample_ids, seqs)
        t_write_fasta = time.time() - t_write_fasta
        logger.info(f"Wrote FASTA alignment: {format_path_for_display(fasta_path)}")

    t_write_phy = 0.0
    if bool(args.write_phylip):
        t_write_phy = time.time()
        phy_path = f"{out_prefix}.phy"
        _write_phylip(phy_path, sample_ids, seqs)
        t_write_phy = time.time() - t_write_phy
        logger.info(f"Wrote PHYLIP alignment: {format_path_for_display(phy_path)}")

    aln = _build_alignment_matrix(seqs)
    t_nj = time.time()
    env_updates = {}
    nj_approx = "exact"
    if requested_mode == "approx":
        nj_approx = "rapid-core"
        env_updates["JANUSX_RAPID_CORE_MODE"] = "lowmem"
        if bool(args.profile):
            env_updates["JANUSX_RAPID_CORE_PROFILE"] = "1"
    elif requested_mode in bionj_alias_map:
        nj_approx = bionj_alias_map[requested_mode]
        if bool(args.profile):
            env_updates["JANUSX_BIONJ_PROFILE"] = "1"
    with _temporary_env(env_updates):
        newick = nj_newick_from_alignment_u8(
            aln,
            sample_ids,
            min_overlap=1,
            max_taxa=max(2, len(sample_ids)),
            threads=int(args.thread),
            nj_approx=nj_approx,
            top_hits_k=64,
        )
    t_nj = time.time() - t_nj
    t_write_nwk = time.time()
    nwk_path = f"{out_prefix}.nwk"
    with open(nwk_path, "w", encoding="utf-8") as fw:
        fw.write(newick)
        fw.write("\n")
    t_write_nwk = time.time() - t_write_nwk
    logger.info(f"Wrote Newick tree: {format_path_for_display(nwk_path)}")

    elapsed = time.time() - t0
    if bool(args.profile):
        profile = {
            "read": float(prof_stream.get("read", 0.0)),
            "convert": float(prof_stream.get("convert", 0.0)),
            "write_fasta": float(t_write_fasta),
            "write_phylip": float(t_write_phy),
            "nj": float(t_nj),
            "write_newick": float(t_write_nwk),
            "total": float(elapsed),
        }
        logger.info(
            "Profile(s): "
            + ", ".join(f"{k}={v:.4f}" for k, v in profile.items())
        )
        prof_path = f"{out_prefix}.profile.tsv"
        with open(prof_path, "w", encoding="utf-8") as fw:
            fw.write("stage\tseconds\n")
            for k, v in profile.items():
                fw.write(f"{k}\t{v:.8f}\n")
        logger.info(f"Wrote profile: {format_path_for_display(prof_path)}")

    log_success(
        logger,
        f"{log_tag} nj phase finished [samples={len(sample_ids)}, sites={n_sites}, time={format_elapsed(elapsed)}]",
    )


def _run_ml(args, *, log_tag: str) -> None:
    if args.fasta:
        input_path = str(args.fasta)
        input_kind = "fasta"
    else:
        input_path, _ = determine_genotype_source_from_args(args)
        input_kind = "genotype"

    out_prefix = _resolve_out_prefix(args, input_path)
    logger = setup_logging(f"{out_prefix}.{log_tag}.log")
    _validate_inputs(args, logger)

    if input_kind == "fasta":
        logger.info(f"Input FASTA: {format_path_for_display(input_path)}")
    else:
        logger.info(f"Input genotype: {format_path_for_display(input_path)}")
    if int(args.thread) == 0:
        logger.info("ML threads: auto")
    else:
        logger.info(f"ML threads: {int(args.thread)}")

    ml_mode, ml_mode_label, ml_compat_preset = _resolve_ml_mode(
        getattr(args, "ml", "exact"),
        bool(args.asc),
    )
    logger.info(f"ML mode: {ml_mode_label}")
    logger.info(f"ML ASC correction: {'on' if bool(args.asc) else 'off'}")
    bootstrap_niter = int(getattr(args, "bootstrap", 0))
    support_mode = str(getattr(args, "support", "bootstrap")).strip().lower()
    if bootstrap_niter > 0:
        logger.info(
            f"ML support: on ({bootstrap_niter} resamples, mode={support_mode})"
        )
    else:
        logger.info("ML support: off")

    t0 = time.time()
    if input_kind == "fasta":
        sample_ids, seqs, n_sites, prof_stream = _fasta_to_alignment(input_path, logger)
    else:
        sample_ids, seqs, n_sites, prof_stream = _genotype_to_alignment(args, input_path, logger)

    added_asc_sites = 0
    if bool(args.asc):
        seqs, added_asc_sites = _apply_asc_pseudo_constant_sites(seqs)
        n_sites += added_asc_sites
        logger.info(
            f"ASC pseudo correction: appended {added_asc_sites} constant sites "
            f"(A/C/G/T each {added_asc_sites // 4})."
        )

    t_write_fasta = 0.0
    fasta_path = f"{out_prefix}.fasta"
    fasta_in_abs = os.path.abspath(input_path) if input_kind == "fasta" else None
    fasta_out_abs = os.path.abspath(fasta_path)
    if fasta_in_abs is not None and fasta_in_abs == fasta_out_abs:
        logger.info("Input FASTA already matches output FASTA path; skipped rewrite.")
    else:
        t_write_fasta = time.time()
        _write_fasta(fasta_path, sample_ids, seqs)
        t_write_fasta = time.time() - t_write_fasta
        logger.info(f"Wrote FASTA alignment: {format_path_for_display(fasta_path)}")

    t_write_phy = 0.0
    if bool(args.write_phylip):
        t_write_phy = time.time()
        phy_path = f"{out_prefix}.phy"
        _write_phylip(phy_path, sample_ids, seqs)
        t_write_phy = time.time() - t_write_phy
        logger.info(f"Wrote PHYLIP alignment: {format_path_for_display(phy_path)}")

    aln = _build_alignment_matrix(seqs)
    t_ml = time.time()
    ml_env = {}
    if ml_mode == "compat":
        # FastTree-compat stage-1:
        # Use SH-like formula and avoid JanusX-specific stabilization heuristics.
        ml_env["JANUSX_ML_SHLIKE_FASTTREE_FORMULA"] = "1"
        ml_env["JANUSX_ML_SHLIKE_ADAPTIVE"] = "0"
        ml_env["JANUSX_ML_SHLIKE_WEIGHTED"] = "0"
        ml_env["JANUSX_ML_SHLIKE_WINSOR"] = "0"
        ml_env["JANUSX_ML_SHLIKE_RECHECK"] = "0"
        # FastTree-like NNI acceptance: local quartet decision + periodic audits.
        ml_env["JANUSX_ML_COMPAT_STRICT_RECHECK"] = os.environ.get(
            "JANUSX_ML_COMPAT_STRICT_RECHECK", "0"
        )
        ml_env["JANUSX_ML_COMPAT_AUDIT_EVERY"] = os.environ.get(
            "JANUSX_ML_COMPAT_AUDIT_EVERY", "64"
        )
        ml_env["JANUSX_ML_COMPAT_AUDIT_TOL"] = os.environ.get(
            "JANUSX_ML_COMPAT_AUDIT_TOL", "0.1"
        )
        ml_env["JANUSX_ML_COMPAT_ROUND_DROP_TOL"] = os.environ.get(
            "JANUSX_ML_COMPAT_ROUND_DROP_TOL", "0.2"
        )
        # FastTree-like ME pre-phase (lightweight by default), then ML-NNI.
        ml_env["JANUSX_ML_COMPAT_ME_ROUNDS"] = os.environ.get(
            "JANUSX_ML_COMPAT_ME_ROUNDS", "0"
        )
        ml_env["JANUSX_ML_COMPAT_ME_BATCH"] = os.environ.get(
            "JANUSX_ML_COMPAT_ME_BATCH", "4"
        )
        ml_env["JANUSX_ML_COMPAT_ME_PASSES_PER_ROUND"] = os.environ.get(
            "JANUSX_ML_COMPAT_ME_PASSES_PER_ROUND", "1"
        )
        # Optional SPR-chain stage (default off to keep stable baseline).
        ml_env["JANUSX_ML_COMPAT_SPR_ROUNDS"] = os.environ.get(
            "JANUSX_ML_COMPAT_SPR_ROUNDS", "0"
        )
        ml_env["JANUSX_ML_COMPAT_SPR_LEN"] = os.environ.get(
            "JANUSX_ML_COMPAT_SPR_LEN", "2"
        )
        # FastTree-like handling for ambiguous symbols in compat mode:
        # non-ACGT(U) characters are treated as unknown.
        ml_env["JANUSX_ML_COMPAT_STRICT_AMBIG"] = os.environ.get(
            "JANUSX_ML_COMPAT_STRICT_AMBIG", "1"
        )
        if ml_compat_preset == "compat-accurate":
            # Consistency-first CAT preset:
            # round-wise posterior reassignment and soft-rate expectation.
            n_taxa = max(2, int(len(sample_ids)))
            auto_sched_on = str(
                os.environ.get("JANUSX_ML_COMPAT_ENABLE_ME_SPR_AUTO", "0")
            ).strip().lower() in {"1", "true", "yes", "on"}
            if auto_sched_on:
                # Optional FastTree-like ME/SPR schedule for large cohorts.
                # Keep disabled by default to avoid runtime regressions.
                if n_taxa >= 512:
                    me_rounds_auto = max(
                        4, int(math.ceil(1.25 * math.log2(float(n_taxa))))
                    )
                    me_rounds_auto = min(me_rounds_auto, 16)
                    spr_rounds_auto = 1
                    spr_len_auto = 6
                else:
                    me_rounds_auto = 0
                    spr_rounds_auto = 0
                    spr_len_auto = 2
                ml_env["JANUSX_ML_COMPAT_ME_ROUNDS"] = os.environ.get(
                    "JANUSX_ML_COMPAT_ME_ROUNDS", str(me_rounds_auto)
                )
                ml_env["JANUSX_ML_COMPAT_SPR_ROUNDS"] = os.environ.get(
                    "JANUSX_ML_COMPAT_SPR_ROUNDS", str(spr_rounds_auto)
                )
                ml_env["JANUSX_ML_COMPAT_SPR_LEN"] = os.environ.get(
                    "JANUSX_ML_COMPAT_SPR_LEN", str(spr_len_auto)
                )
            compat_cat_update_default = min(4096, int(n_sites))
            ml_env["JANUSX_ML_COMPAT_AUDIT_EVERY"] = os.environ.get(
                "JANUSX_ML_COMPAT_AUDIT_EVERY", "32"
            )
            ml_env["JANUSX_ML_COMPAT_AUDIT_TOL"] = os.environ.get(
                "JANUSX_ML_COMPAT_AUDIT_TOL", "0.05"
            )
            ml_env["JANUSX_ML_COMPAT_ROUND_DROP_TOL"] = os.environ.get(
                "JANUSX_ML_COMPAT_ROUND_DROP_TOL", "0.05"
            )
            ml_env["JANUSX_ML_COMPAT_FULLSCAN_ROUNDS"] = os.environ.get(
                "JANUSX_ML_COMPAT_FULLSCAN_ROUNDS", "1"
            )
            ml_env["JANUSX_ML_COMPAT_ME_FULLSCAN_ROUNDS"] = os.environ.get(
                "JANUSX_ML_COMPAT_ME_FULLSCAN_ROUNDS", "1"
            )
            ml_env["JANUSX_ML_COMPAT_SPR_FULLSCAN_ROUNDS"] = os.environ.get(
                "JANUSX_ML_COMPAT_SPR_FULLSCAN_ROUNDS", "1"
            )
            ml_env["JANUSX_ML_COMPAT_CAT_UPDATE_EVERY"] = os.environ.get(
                "JANUSX_ML_COMPAT_CAT_UPDATE_EVERY", "1"
            )
            ml_env["JANUSX_ML_COMPAT_CAT_UPDATE_SITES"] = os.environ.get(
                "JANUSX_ML_COMPAT_CAT_UPDATE_SITES", str(compat_cat_update_default)
            )
            ml_env["JANUSX_ML_COMPAT_CAT_KEEP_PMIN"] = os.environ.get(
                "JANUSX_ML_COMPAT_CAT_KEEP_PMIN", "0.15"
            )
            ml_env["JANUSX_ML_COMPAT_CAT_PRIOR_WEIGHT"] = os.environ.get(
                "JANUSX_ML_COMPAT_CAT_PRIOR_WEIGHT", "0.2"
            )
            ml_env["JANUSX_ML_COMPAT_CAT_POST_TAU"] = os.environ.get(
                "JANUSX_ML_COMPAT_CAT_POST_TAU", "0.9"
            )
            ml_env["JANUSX_ML_COMPAT_CAT_SOFT_RATE"] = os.environ.get(
                "JANUSX_ML_COMPAT_CAT_SOFT_RATE", "1"
            )
        else:
            # Speed-oriented compat CAT preset: static assignment.
            ml_env["JANUSX_ML_COMPAT_CAT_UPDATE_EVERY"] = os.environ.get(
                "JANUSX_ML_COMPAT_CAT_UPDATE_EVERY", "0"
            )
            ml_env["JANUSX_ML_COMPAT_CAT_KEEP_PMIN"] = os.environ.get(
                "JANUSX_ML_COMPAT_CAT_KEEP_PMIN", "0.55"
            )
            ml_env["JANUSX_ML_COMPAT_CAT_PRIOR_WEIGHT"] = os.environ.get(
                "JANUSX_ML_COMPAT_CAT_PRIOR_WEIGHT", "0.5"
            )
            ml_env["JANUSX_ML_COMPAT_CAT_POST_TAU"] = os.environ.get(
                "JANUSX_ML_COMPAT_CAT_POST_TAU", "1.0"
            )
            ml_env["JANUSX_ML_COMPAT_CAT_SOFT_RATE"] = os.environ.get(
                "JANUSX_ML_COMPAT_CAT_SOFT_RATE", "0"
            )
        compat_cache_cap = int(os.environ.get("JANUSX_ML_COMPAT_CACHE_SITES", "4096"))
        compat_cache_sites = max(256, min(int(n_sites), compat_cache_cap))
        ml_env["JANUSX_ML_SHLIKE_CACHE_SITES"] = str(compat_cache_sites)
        ml_env["JANUSX_ML_SHLIKE_DRAW_SITES"] = str(compat_cache_sites)
        compat_quartet_default = min(2048, int(n_sites))
        compat_quartet_sites = int(
            os.environ.get("JANUSX_ML_COMPAT_QUARTET_SITES", str(compat_quartet_default))
        )
        ml_env["JANUSX_ML_COMPAT_QUARTET_SITES"] = str(
            max(64, min(int(n_sites), compat_quartet_sites))
        )
        compat_me_sites_default = min(768, int(n_sites))
        compat_me_sites = int(
            os.environ.get("JANUSX_ML_COMPAT_ME_SITES", str(compat_me_sites_default))
        )
        ml_env["JANUSX_ML_COMPAT_ME_SITES"] = str(
            max(64, min(int(n_sites), compat_me_sites))
        )
    with _temporary_env(ml_env):
        newick = ml_newick_from_alignment_u8(
            aln,
            sample_ids,
            min_overlap=1,
            max_taxa=max(2, len(sample_ids)),
            threads=int(args.thread),
            ml_mode=ml_mode,
            bootstrap_niter=bootstrap_niter,
            support_mode=support_mode,
        )
    t_ml = time.time() - t_ml
    t_write_nwk = time.time()
    nwk_path = f"{out_prefix}.nwk"
    txt = str(newick).strip()
    if txt == "":
        raise RuntimeError("Rust ML engine produced empty Newick output.")
    with open(nwk_path, "w", encoding="utf-8") as fw:
        fw.write(txt)
        fw.write("\n")
    t_write_nwk = time.time() - t_write_nwk
    logger.info(f"Wrote Newick tree: {format_path_for_display(nwk_path)}")

    elapsed = time.time() - t0
    if bool(args.profile):
        profile = {
            "read": float(prof_stream.get("read", 0.0)),
            "convert": float(prof_stream.get("convert", 0.0)),
            "write_fasta": float(t_write_fasta),
            "write_phylip": float(t_write_phy),
            "ml": float(t_ml),
            "write_newick": float(t_write_nwk),
            "total": float(elapsed),
        }
        logger.info(
            "Profile(s): "
            + ", ".join(f"{k}={v:.4f}" for k, v in profile.items())
        )
        prof_path = f"{out_prefix}.profile.tsv"
        with open(prof_path, "w", encoding="utf-8") as fw:
            fw.write("stage\tseconds\n")
            for k, v in profile.items():
                fw.write(f"{k}\t{v:.8f}\n")
        logger.info(f"Wrote profile: {format_path_for_display(prof_path)}")

    log_success(
        logger,
        f"{log_tag} ml phase finished [samples={len(sample_ids)}, sites={n_sites}, time={format_elapsed(elapsed)}]",
    )


def main(argv: list[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)

    parser = build_tree_parser()
    args = parser.parse_args(argv_list)

    if not any((args.vcf, args.hmp, args.file, args.bfile, args.fasta)):
        parser.error("`-nj` requires one input: -vcf/-hmp/-file/-bfile/-fa.")

    nj_mode = str(args.nj if args.nj is not None else "exact").lower()
    try:
        _run_nj(args, log_tag="tree", nj_mode=nj_mode)
    except Exception as exc:
        parser.error(str(exc))
