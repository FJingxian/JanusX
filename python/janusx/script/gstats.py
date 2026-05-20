# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import socket
import time
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
import pandas as pd

from janusx.gfreader import prepare_cli_input_cache
from ._common.config_render import emit_cli_configuration
from ._common.genoio import determine_genotype_source
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_file_input_site_metadata_exists,
    ensure_plink_prefix_exists,
    format_path_for_display,
)
from ._common.status import CliStatus, format_elapsed, log_success
from ._common.threads import detect_effective_threads

try:
    from janusx import janusx as jxrs
except Exception:
    jxrs = None


@dataclass(frozen=True)
class LdscWindowSpec:
    kind: str
    value: float
    label: str


def _require_rust_backend() -> None:
    if jxrs is None:
        raise RuntimeError(
            "JanusX Rust extension is unavailable. Please rebuild/install the extension first."
        )
    for name in (
        "gstats_bed_site_stats",
        "gstats_bed_joint_stats",
        "gstats_bed_individual_stats",
        "gstats_bed_ldscore",
    ):
        if not hasattr(jxrs, name):
            raise RuntimeError(
                f"Rust extension is missing `{name}`. Please rebuild JanusX so gstats kernels are exported."
            )


def _normalize_plink_prefix(path_or_prefix: str) -> str:
    p = str(path_or_prefix).strip()
    low = p.lower()
    if low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam"):
        return p[:-4]
    return p


def _histogram_edges(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return np.asarray([-0.5, 0.5], dtype=np.float64)
    lo = float(np.min(data))
    hi = float(np.max(data))
    if data.size == 1 or np.isclose(lo, hi):
        pad = max(1e-6, abs(lo) * 0.1 + 1e-6)
        return np.asarray([lo - pad, hi + pad], dtype=np.float64)
    try:
        edges = np.histogram_bin_edges(data, bins="fd")
    except ValueError:
        edges = np.histogram_bin_edges(data, bins="auto")
    if np.asarray(edges).size < 2:
        pad = max(1e-6, (hi - lo) * 0.1)
        return np.asarray([lo - pad, hi + pad], dtype=np.float64)
    return np.asarray(edges, dtype=np.float64)


def _parse_ldsc_window(text: str | None) -> LdscWindowSpec:
    raw = "100kb" if text is None or str(text).strip() == "" else str(text).strip().lower()
    compact = raw.replace(" ", "")
    import re

    m = re.fullmatch(r"([0-9]*\.?[0-9]+)([a-z]*)", compact)
    if m is None:
        raise ValueError(
            f"Invalid -ldsc window: {text!r}. Use forms like 100, 100kb, 0.1mb, 100000b, or 10cm."
        )
    value = float(m.group(1))
    unit = str(m.group(2) or "")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"-ldsc window must be > 0, got {text!r}.")

    if unit == "":
        if not float(value).is_integer():
            raise ValueError(f"SNP-count LD-score window must be an integer, got {text!r}.")
        v = int(round(value))
        return LdscWindowSpec(kind="variants", value=float(v), label=f"{v}snp")
    if unit in {"snp", "snps"}:
        if not float(value).is_integer():
            raise ValueError(f"SNP-count LD-score window must be an integer, got {text!r}.")
        v = int(round(value))
        return LdscWindowSpec(kind="variants", value=float(v), label=f"{v}snp")
    if unit in {"b", "bp"}:
        bp = int(round(value))
        return LdscWindowSpec(kind="bp", value=float(bp), label=f"{bp}b")
    if unit == "kb":
        bp = int(round(value * 1000.0))
        return LdscWindowSpec(kind="bp", value=float(bp), label=compact)
    if unit == "mb":
        bp = int(round(value * 1_000_000.0))
        return LdscWindowSpec(kind="bp", value=float(bp), label=compact)
    if unit == "cm":
        return LdscWindowSpec(kind="cm", value=float(value), label=compact)
    raise ValueError(
        f"Unsupported -ldsc unit in {text!r}. Use forms like 100, 100kb, 0.1mb, 100000b, or 10cm."
    )


def _read_fam_table(prefix: str) -> pd.DataFrame:
    fam_path = f"{prefix}.fam"
    fam = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        engine="python",
        dtype=str,
        keep_default_na=False,
    )
    if fam.shape[1] < 2:
        raise ValueError(f"Malformed FAM file: {fam_path}")
    out = fam.iloc[:, [0, 1]].copy()
    out.columns = ["fid", "iid"]
    out["fid"] = out["fid"].astype(str).str.strip()
    out["iid"] = out["iid"].astype(str).str.strip()
    return out


def _read_bim_table(prefix: str) -> pd.DataFrame:
    bim_path = f"{prefix}.bim"
    bim = pd.read_csv(
        bim_path,
        sep=r"\s+",
        header=None,
        engine="python",
        dtype={0: str},
        keep_default_na=False,
    )
    if bim.shape[1] < 4:
        raise ValueError(f"Malformed BIM file: {bim_path}")
    out = bim.iloc[:, [0, 3]].copy()
    out.columns = ["chr", "pos"]
    out["chr"] = out["chr"].astype(str).str.strip()
    out["pos"] = pd.to_numeric(out["pos"], errors="raise").astype(np.int64, copy=False)
    return out


def _open_text_writer(path: str):
    return open(path, "w", encoding="utf-8", buffering=8 * 1024 * 1024)


def _write_site_tables_from_bim(
    prefix: str,
    *,
    outprefix: str,
    freq: np.ndarray | None = None,
    miss: np.ndarray | None = None,
    het: np.ndarray | None = None,
) -> list[str]:
    freq_arr = None if freq is None else np.asarray(freq, dtype=np.float32).reshape(-1)
    miss_arr = None if miss is None else np.asarray(miss, dtype=np.float32).reshape(-1)
    het_arr = None if het is None else np.asarray(het, dtype=np.float32).reshape(-1)
    expected = next(
        arr.shape[0] for arr in (freq_arr, miss_arr, het_arr) if arr is not None
    )
    for arr in (freq_arr, miss_arr, het_arr):
        if arr is not None and int(arr.shape[0]) != int(expected):
            raise ValueError("site-stat arrays have inconsistent lengths")

    bim_path = f"{prefix}.bim"
    outputs: list[str] = []
    with ExitStack() as stack:
        freq_fh = None
        miss_fh = None
        het_fh = None
        if freq_arr is not None:
            freq_path = f"{outprefix}.freq"
            freq_fh = stack.enter_context(_open_text_writer(freq_path))
            freq_fh.write("chr\tpos\tfreq\n")
            outputs.append(freq_path)
        if miss_arr is not None:
            miss_path = f"{outprefix}.lmiss"
            miss_fh = stack.enter_context(_open_text_writer(miss_path))
            miss_fh.write("chr\tpos\tmiss\n")
            outputs.append(miss_path)
        if het_arr is not None:
            het_path = f"{outprefix}.lhet"
            het_fh = stack.enter_context(_open_text_writer(het_path))
            het_fh.write("chr\tpos\thet\n")
            outputs.append(het_path)

        row_idx = 0
        with open(bim_path, "r", encoding="utf-8", buffering=8 * 1024 * 1024) as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                toks = line.split()
                if len(toks) < 4:
                    raise ValueError(f"Malformed BIM file: {bim_path}:{line_no}")
                if row_idx >= int(expected):
                    raise ValueError(
                        f"BIM/site-stat length mismatch: bim has more rows than stats ({row_idx + 1} > {expected})"
                    )
                chrom = toks[0].strip()
                pos = toks[3]
                if freq_fh is not None:
                    freq_fh.write(f"{chrom}\t{pos}\t{float(freq_arr[row_idx]):.6f}\n")
                if miss_fh is not None:
                    miss_fh.write(f"{chrom}\t{pos}\t{float(miss_arr[row_idx]):.6f}\n")
                if het_fh is not None:
                    het_fh.write(f"{chrom}\t{pos}\t{float(het_arr[row_idx]):.6f}\n")
                row_idx += 1

        if row_idx != int(expected):
            raise ValueError(f"BIM/site-stat length mismatch: bim={row_idx}, stats={expected}")

    return outputs


def _write_individual_tables_from_fam(
    prefix: str,
    *,
    outprefix: str,
    miss: np.ndarray | None = None,
    het: np.ndarray | None = None,
) -> list[str]:
    miss_arr = None if miss is None else np.asarray(miss, dtype=np.float32).reshape(-1)
    het_arr = None if het is None else np.asarray(het, dtype=np.float32).reshape(-1)
    expected = next(arr.shape[0] for arr in (miss_arr, het_arr) if arr is not None)
    for arr in (miss_arr, het_arr):
        if arr is not None and int(arr.shape[0]) != int(expected):
            raise ValueError("individual-stat arrays have inconsistent lengths")

    fam_path = f"{prefix}.fam"
    outputs: list[str] = []
    with ExitStack() as stack:
        miss_fh = None
        het_fh = None
        if miss_arr is not None:
            miss_path = f"{outprefix}.imiss"
            miss_fh = stack.enter_context(_open_text_writer(miss_path))
            miss_fh.write("fid\tiid\tmiss\n")
            outputs.append(miss_path)
        if het_arr is not None:
            het_path = f"{outprefix}.ihet"
            het_fh = stack.enter_context(_open_text_writer(het_path))
            het_fh.write("fid\tiid\thet\n")
            outputs.append(het_path)

        row_idx = 0
        with open(fam_path, "r", encoding="utf-8", buffering=8 * 1024 * 1024) as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                toks = line.split()
                if len(toks) < 2:
                    raise ValueError(f"Malformed FAM file: {fam_path}:{line_no}")
                if row_idx >= int(expected):
                    raise ValueError(
                        f"FAM/individual-stat length mismatch: fam has more rows than stats ({row_idx + 1} > {expected})"
                    )
                fid = toks[0].strip()
                iid = toks[1].strip()
                if miss_fh is not None:
                    miss_fh.write(f"{fid}\t{iid}\t{float(miss_arr[row_idx]):.6f}\n")
                if het_fh is not None:
                    het_fh.write(f"{fid}\t{iid}\t{float(het_arr[row_idx]):.6f}\n")
                row_idx += 1

        if row_idx != int(expected):
            raise ValueError(f"FAM/individual-stat length mismatch: fam={row_idx}, stats={expected}")

    return outputs


def _setup_pdf_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    return plt


def _draw_rate_hist(ax, values: np.ndarray, *, title: str, color: str, xlabel: str) -> None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    bins = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    ax.hist(
        arr,
        bins=bins,
        color=color,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
    )
    if arr.size > 0:
        ax.axvline(float(np.mean(arr)), color="#111827", linestyle="--", linewidth=1.1, alpha=0.9)
        txt = f"n={arr.size:,}\nmean={float(np.mean(arr)):.4f}"
    else:
        txt = "n=0"
    ax.text(
        0.98,
        0.96,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#374151",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", color="#D1D5DB", alpha=0.55, linewidth=0.8)


def _plot_freq_pdf(values: np.ndarray, out_pdf: str) -> None:
    plt = _setup_pdf_matplotlib()
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    edges = _histogram_edges(arr)
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.0))
    fig.patch.set_facecolor("white")
    ax.hist(
        arr,
        bins=edges,
        color="#4C78A8",
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
    )
    if arr.size > 0:
        ax.axvline(float(np.mean(arr)), color="#111827", linestyle="--", linewidth=1.1, alpha=0.9)
        ax.text(
            0.98,
            0.96,
            f"site={arr.size:,}\nmean={float(np.mean(arr)):.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#374151",
        )
    ax.set_title("Minor allele frequency distribution", fontsize=12)
    ax.set_xlabel("MAF")
    ax.set_ylabel("Count")
    ax.grid(axis="y", color="#D1D5DB", alpha=0.55, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_dual_rate_pdf(
    left_values: np.ndarray,
    right_values: np.ndarray,
    *,
    left_title: str,
    right_title: str,
    xlabel: str,
    out_pdf: str,
) -> None:
    plt = _setup_pdf_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.patch.set_facecolor("white")
    _draw_rate_hist(axes[0], left_values, title=left_title, color="#4C78A8", xlabel=xlabel)
    _draw_rate_hist(axes[1], right_values, title=right_title, color="#F58518", xlabel=xlabel)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_ldsc_pdf(df: pd.DataFrame, out_pdf: str, *, window_label: str, n_samples: int) -> None:
    plt = _setup_pdf_matplotlib()
    chr_labels = df["chr"].astype(str).tolist()
    pos = pd.to_numeric(df["pos"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    ldsc = pd.to_numeric(df["ldsc"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(pos) & np.isfinite(ldsc)
    if finite.sum() == 0:
        raise ValueError("No finite LD score values found for plotting.")

    uniq_chr: list[str] = []
    seen: set[str] = set()
    for chrom in chr_labels:
        if chrom not in seen:
            seen.add(chrom)
            uniq_chr.append(chrom)

    offsets: dict[str, float] = {}
    centers: list[tuple[float, str]] = []
    cur = 0.0
    gap = 1_000_000.0
    x = np.zeros((len(df),), dtype=np.float64)
    palette = ("#4C78A8", "#E45756")
    colors = np.empty((len(df),), dtype=object)
    for chr_idx, chrom in enumerate(uniq_chr):
        idx = np.where(df["chr"].astype(str).to_numpy(dtype=object) == chrom)[0]
        if idx.size == 0:
            continue
        chr_pos = pos[idx]
        chr_min = float(np.nanmin(chr_pos))
        chr_max = float(np.nanmax(chr_pos))
        offsets[chrom] = cur - chr_min
        x[idx] = chr_pos + offsets[chrom]
        centers.append(((float(np.nanmin(x[idx])) + float(np.nanmax(x[idx]))) * 0.5, chrom))
        colors[idx] = palette[chr_idx % len(palette)]
        cur = float(np.nanmax(x[idx])) + gap

    fig, ax = plt.subplots(1, 1, figsize=(12.0, 5.2))
    fig.patch.set_facecolor("white")
    ax.scatter(
        x[finite],
        ldsc[finite],
        c=np.asarray(colors[finite], dtype=object),
        s=8,
        linewidths=0.0,
        alpha=0.85,
    )
    ax.axhline(float(np.nanmean(ldsc[finite])), color="#111827", linestyle="--", linewidth=1.1, alpha=0.9)
    ax.set_title(f"LD score Manhattan plot ({window_label}, N={int(n_samples):,})", fontsize=12)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("LD score")
    ax.grid(axis="y", color="#D1D5DB", alpha=0.55, linewidth=0.8)
    if centers:
        ax.set_xticks([c for c, _ in centers], [lab for _, lab in centers])
    ax.text(
        0.99,
        0.97,
        f"site={int(finite.sum()):,}\nmean={float(np.nanmean(ldsc[finite])):.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx gstats",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx gstats -bfile geno_prefix -freq -o outdir -prefix demo",
                "jx gstats -vcf cohort.vcf.gz -miss -het -o outdir -prefix cohort",
                "jx gstats -file geno_prefix -ldsc 100kb -o outdir -prefix panel",
                "jx gstats -bfile geno_prefix -ldsc 100",
            ]
        ),
        description="Genotype basic statistics: MAF / missing / heterozygosity / LD score (BED-cache first).",
    )

    geno = parser.add_argument_group("Genotype input")
    geno.add_argument("-bfile", "--bfile", type=str, help="PLINK prefix (.bed/.bim/.fam).")
    geno.add_argument("-vcf", "--vcf", type=str, help="Input VCF/VCF.GZ; will be cached as BED before statistics.")
    geno.add_argument("-hmp", "--hmp", type=str, help="Input HapMap/HMP(.gz); will be cached as BED before statistics.")
    geno.add_argument(
        "-file",
        "--file",
        type=str,
        help=(
            "Input genotype numeric matrix (.txt/.tsv/.csv/.npy/.bin) or prefix. "
            "Requires sibling prefix.id; BED caching also requires prefix.site or prefix.bim."
        ),
    )

    stats = parser.add_argument_group("Statistics")
    stats.add_argument("-freq", action="store_true", help="Write site MAF table: <prefix>.freq and a PDF histogram.")
    stats.add_argument(
        "-miss",
        action="store_true",
        help="Write missing-rate tables: <prefix>.imiss / <prefix>.lmiss and a PDF distribution plot.",
    )
    stats.add_argument(
        "-het",
        action="store_true",
        help="Write heterozygosity tables: <prefix>.ihet / <prefix>.lhet and a PDF distribution plot.",
    )
    stats.add_argument(
        "-ldsc",
        nargs="?",
        const="100kb",
        default=None,
        metavar="WINDOW",
        help=(
            "Write site LD scores: <prefix>.<N>.<window>.ldsc and a Manhattan PDF. "
            "WINDOW accepts SNP count (e.g. 100), physical distance (100kb/0.1mb/100000b), "
            "or genetic distance (10cm). Default when -ldsc is given without a value: 100kb."
        ),
    )

    out = parser.add_argument_group("Output / runtime")
    out.add_argument("-o", "--out", type=str, default=".", help="Output directory (default: current directory).")
    out.add_argument("-prefix", "--prefix", type=str, default=None, help="Output prefix (default: inferred from input).")
    out.add_argument(
        "-t",
        "--thread",
        "--threads",
        "-threads",
        dest="threads",
        type=int,
        default=max(1, int(detect_effective_threads())),
        help="Thread count for Rust kernels (default: auto-detected).",
    )
    return parser


def main() -> None:
    _require_rust_backend()
    parser = build_parser()
    args = parser.parse_args()

    if not any((args.bfile, args.vcf, args.hmp, args.file)):
        parser.error("one genotype input is required: -bfile / -vcf / -hmp / -file")
    if sum(bool(x) for x in (args.bfile, args.vcf, args.hmp, args.file)) != 1:
        parser.error("provide exactly one genotype input among -bfile / -vcf / -hmp / -file")
    if not any((bool(args.freq), bool(args.miss), bool(args.het), args.ldsc is not None)):
        parser.error("select at least one statistic: -freq / -miss / -het / -ldsc")
    if int(args.threads) <= 0:
        parser.error("--threads must be > 0")

    gfile, auto_prefix = determine_genotype_source(
        vcf=getattr(args, "vcf", None),
        hmp=getattr(args, "hmp", None),
        file=getattr(args, "file", None),
        bfile=getattr(args, "bfile", None),
        prefix=None,
        apply_cache=False,
    )
    if args.prefix is None:
        args.prefix = auto_prefix

    os.makedirs(args.out, exist_ok=True)
    outprefix = os.path.join(str(args.out), str(args.prefix))
    log_path = f"{outprefix}.gstats.log"
    logger = setup_logging(log_path)

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.vcf:
        checks.append(ensure_file_exists(logger, gfile, "Genotype VCF input"))
    elif args.hmp:
        checks.append(ensure_file_exists(logger, gfile, "Genotype HMP input"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
        checks.append(ensure_file_input_site_metadata_exists(logger, gfile, "Genotype FILE site metadata"))
    if not ensure_all_true(checks):
        raise FileNotFoundError(gfile)

    ldsc_spec = _parse_ldsc_window(args.ldsc) if args.ldsc is not None else None

    emit_cli_configuration(
        logger,
        app_title="JanusX gstats",
        config_title="Genotype basic statistics",
        host=socket.gethostname(),
        sections=[
            (
                "Input",
                [
                    ("Source", gfile),
                    ("BFILE", bool(args.bfile)),
                    ("VCF", bool(args.vcf)),
                    ("HMP", bool(args.hmp)),
                    ("FILE", bool(args.file)),
                ],
            ),
            (
                "Statistics",
                [
                    ("MAF (-freq)", bool(args.freq)),
                    ("Missing (-miss)", bool(args.miss)),
                    ("Heterozygosity (-het)", bool(args.het)),
                    ("LD score (-ldsc)", ldsc_spec.label if ldsc_spec is not None else "off"),
                ],
            ),
            (
                "Runtime",
                [
                    ("Threads", int(args.threads)),
                    ("BED-cache for non-bfile", True),
                    ("Log file", log_path),
                ],
            ),
        ],
        footer_rows=[("Output prefix", outprefix)],
    )

    t0 = time.monotonic()
    bed_prefix = str(gfile)
    if not args.bfile:
        cache_msg = "Caching genotype as PLINK BED..."
        delim = "," if (args.file and str(gfile).lower().endswith(".csv")) else None
        with CliStatus(cache_msg, enabled=True) as task:
            bed_prefix = prepare_cli_input_cache(
                str(gfile),
                snps_only=False,
                delimiter=delim,
                prefer_plink_for_txt=True,
            )
            task.complete("Caching genotype as PLINK BED ...Finished")
        bed_prefix = _normalize_plink_prefix(str(bed_prefix))
        log_success(
            logger,
            f"Statistics backend input switched to BED cache: {format_path_for_display(bed_prefix)}",
        )
    else:
        bed_prefix = _normalize_plink_prefix(str(bed_prefix))

    if not ensure_plink_prefix_exists(logger, bed_prefix, "Statistics BED prefix"):
        raise FileNotFoundError(bed_prefix)

    outputs: list[str] = []

    individual_stats = None
    site_stats = None
    if args.freq or args.miss or args.het:
        with CliStatus("Computing genotype statistics...", enabled=True) as task:
            maf_raw, lmiss_raw, lhet_raw, imiss_raw, ihet_raw, n_samples, n_sites = jxrs.gstats_bed_joint_stats(
                str(bed_prefix),
                site_maf=bool(args.freq),
                site_miss=bool(args.miss),
                site_het=bool(args.het),
                individual_miss=bool(args.miss),
                individual_het=bool(args.het),
                threads=int(args.threads),
            )
            task.complete("Computing genotype statistics ...Finished")
        site_stats = {
            "maf": None if maf_raw is None else np.asarray(maf_raw, dtype=np.float32).reshape(-1),
            "miss": None if lmiss_raw is None else np.asarray(lmiss_raw, dtype=np.float32).reshape(-1),
            "het": None if lhet_raw is None else np.asarray(lhet_raw, dtype=np.float32).reshape(-1),
            "n_samples": int(n_samples),
        }
        individual_stats = {
            "miss": None if imiss_raw is None else np.asarray(imiss_raw, dtype=np.float32).reshape(-1),
            "het": None if ihet_raw is None else np.asarray(ihet_raw, dtype=np.float32).reshape(-1),
            "n_sites": int(n_sites),
        }

    if args.freq or args.miss or args.het:
        assert site_stats is not None
        site_outputs = _write_site_tables_from_bim(
            str(bed_prefix),
            outprefix=outprefix,
            freq=site_stats["maf"] if args.freq else None,
            miss=site_stats["miss"] if args.miss else None,
            het=site_stats["het"] if args.het else None,
        )
        outputs.extend(site_outputs)

    if args.miss or args.het:
        assert site_stats is not None and individual_stats is not None
        sample_outputs = _write_individual_tables_from_fam(
            str(bed_prefix),
            outprefix=outprefix,
            miss=individual_stats["miss"] if args.miss else None,
            het=individual_stats["het"] if args.het else None,
        )
        outputs.extend(sample_outputs)

    if args.freq:
        assert site_stats is not None and site_stats["maf"] is not None
        freq_path = f"{outprefix}.freq"
        freq_pdf = f"{outprefix}.freq.pdf"
        _plot_freq_pdf(np.asarray(site_stats["maf"], dtype=np.float64), freq_pdf)
        outputs.append(freq_pdf)
        log_success(logger, f"MAF table saved: {format_path_for_display(freq_path)}")
        log_success(logger, f"MAF PDF saved: {format_path_for_display(freq_pdf)}")

    if args.miss:
        assert site_stats is not None and individual_stats is not None
        miss_pdf = f"{outprefix}.miss.pdf"
        _plot_dual_rate_pdf(
            np.asarray(individual_stats["miss"], dtype=np.float64),
            np.asarray(site_stats["miss"], dtype=np.float64),
            left_title="Individual missing rate",
            right_title="Locus missing rate",
            xlabel="Missing rate",
            out_pdf=miss_pdf,
        )
        outputs.append(miss_pdf)
        imiss_path = f"{outprefix}.imiss"
        lmiss_path = f"{outprefix}.lmiss"
        log_success(logger, f"Individual missing table saved: {format_path_for_display(imiss_path)}")
        log_success(logger, f"Locus missing table saved: {format_path_for_display(lmiss_path)}")
        log_success(logger, f"Missing-rate PDF saved: {format_path_for_display(miss_pdf)}")

    if args.het:
        assert site_stats is not None and individual_stats is not None
        het_pdf = f"{outprefix}.het.pdf"
        _plot_dual_rate_pdf(
            np.asarray(individual_stats["het"], dtype=np.float64),
            np.asarray(site_stats["het"], dtype=np.float64),
            left_title="Individual heterozygosity rate",
            right_title="Locus heterozygosity rate",
            xlabel="Heterozygosity rate",
            out_pdf=het_pdf,
        )
        outputs.append(het_pdf)
        ihet_path = f"{outprefix}.ihet"
        lhet_path = f"{outprefix}.lhet"
        log_success(logger, f"Individual heterozygosity table saved: {format_path_for_display(ihet_path)}")
        log_success(logger, f"Locus heterozygosity table saved: {format_path_for_display(lhet_path)}")
        log_success(logger, f"Heterozygosity PDF saved: {format_path_for_display(het_pdf)}")

    if ldsc_spec is not None:
        bim_df = _read_bim_table(str(bed_prefix))
        with CliStatus("Computing LD scores...", enabled=True) as task:
            m_raw, ld_raw, ld_n_samples = jxrs.gstats_bed_ldscore(
                str(bed_prefix),
                str(ldsc_spec.kind),
                float(ldsc_spec.value),
                threads=int(args.threads),
            )
            task.complete("Computing LD scores ...Finished")
        ld_df = bim_df.copy()
        m_arr = np.asarray(m_raw, dtype=np.int64).reshape(-1)
        ld_arr = np.asarray(ld_raw, dtype=np.float64).reshape(-1)
        if len(ld_df) != int(m_arr.shape[0]) or len(ld_df) != int(ld_arr.shape[0]):
            raise ValueError(
                f"BIM/ldscore length mismatch: bim={len(ld_df)}, M={int(m_arr.shape[0])}, ldsc={int(ld_arr.shape[0])}"
            )
        ld_df["M"] = m_arr
        ld_df["ldsc"] = ld_arr
        ld_path = f"{outprefix}.{int(ld_n_samples)}.{ldsc_spec.label}.ldsc"
        ld_pdf = f"{ld_path}.pdf"
        ld_df.to_csv(ld_path, sep="\t", index=False, float_format="%.6f")
        _plot_ldsc_pdf(ld_df, ld_pdf, window_label=ldsc_spec.label, n_samples=int(ld_n_samples))
        outputs.extend([ld_path, ld_pdf])
        log_success(logger, f"LD-score table saved: {format_path_for_display(ld_path)}")
        log_success(logger, f"LD-score PDF saved: {format_path_for_display(ld_pdf)}")

    elapsed = max(0.0, time.monotonic() - t0)
    log_success(logger, f"gstats finished [{format_elapsed(elapsed)}]")
    if outputs:
        logger.info("Outputs:")
        for path in outputs:
            logger.info(f"  {format_path_for_display(path)}")


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    main()
