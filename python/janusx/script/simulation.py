"""
JanusX Simulation CLI

Rust-first phenotype simulation from an existing genotype input.
Python is kept as the orchestration / CLI layer and optional plotting hook.
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from datetime import datetime
from typing import Any, Callable, Literal, Optional

import numpy as np

from janusx.gfreader import inspect_genotype_file
from janusx.janusx import g2p_simulate

from ._common.config_render import emit_cli_configuration
from ._common.genocache import configure_genotype_cache_from_out
from ._common.genoio import determine_genotype_source_from_args as determine_genotype_source
from ._common.grmio import load_and_align_grm
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
    format_path_for_display,
)
from ._common.progress import ProgressAdapter
from ._common.status import CliStatus, format_elapsed, log_success


def _parse_bimrange(text: str) -> tuple[str, int, int]:
    raw = str(text).strip()
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid bimrange '{raw}'. Expected format like chr1:1000:2000."
        )
    chrom = str(parts[0]).strip()
    start = int(parts[1])
    end = int(parts[2])
    if chrom == "":
        raise ValueError(f"Invalid bimrange '{raw}': chromosome is empty.")
    if end < start:
        raise ValueError(f"Invalid bimrange '{raw}': end < start.")
    return chrom, start, end


def _parse_logic_gate_size(text: str) -> tuple[int, int, int]:
    raw = str(text).strip()
    if raw == "":
        raise ValueError("Empty logic-gate size spec.")
    if "," in raw:
        left, right = raw.split(",", 1)
        k = int(left)
        pair = int(right)
    else:
        k = int(raw)
        pair = 1
    if k <= 0:
        raise ValueError("Logic-gate site count must be > 0.")
    if pair <= 0:
        raise ValueError("Logic-gate count must be > 0.")
    return k, k, pair


def _resolve_distribution(
    args: argparse.Namespace,
) -> tuple[str, float, float, float]:
    if args.gamma_shape is not None:
        shape = float(args.gamma_shape)
        return "gamma", shape, 1.0, 1.0
    return "normal", 1.0, 1.0, 1.0


def _resolve_logic_config(
    args: argparse.Namespace,
) -> tuple[Optional[str], Optional[int], int, int, Optional[int]]:
    if args.logic_gate is not None:
        k_min, k_max, gate_count = _parse_logic_gate_size(str(args.logic_gate[0]))
        logic_mode = str(args.logic_gate[1]).strip().lower()
        return logic_mode, gate_count, k_min, k_max, None
    return None, None, 2, 2, None


def _estimate_simulation_scan_passes(
    *,
    causal_count: int,
    cs_pve: Optional[float],
    bimranges: list[tuple[str, int, int]],
    logic_mode: Optional[str],
    logic_gate_count: Optional[int],
) -> int:
    logic_requested = logic_mode is not None and str(logic_mode).strip() != ""
    base_term_count = (
        (int(logic_gate_count) if logic_gate_count is not None else max(1, int(causal_count)))
        if logic_requested
        else int(causal_count)
    )
    effective_term_count = max(int(base_term_count), len(bimranges))
    causal_pve_target = (
        float(cs_pve)
        if cs_pve is not None
        else (min(0.05 * effective_term_count, 0.95) if effective_term_count > 0 else 0.0)
    )
    needs_causal_scan = effective_term_count > 0 and causal_pve_target > 0.0
    return 1 + int(needs_causal_scan)


def _run_rust_simulation(
    *,
    gfile: str,
    seed: int,
    maf: float,
    missing_rate: float,
    het_threshold: float | None,
    bg_pve: float,
    residual_var: float,
    causal: int,
    cs_pve: Optional[float],
    bimranges: list[tuple[str, int, int]],
    logic_mode: Optional[str],
    logic_gate_count: Optional[int],
    logic_k_min: int,
    logic_k_max: int,
    logic_ld_max: float,
    logic_het_max: float,
    logic_af_min: float,
    logic_af_max: float,
    logic_max_iter: int,
    logic_window_bp: Optional[int],
    logic_effect_model: str,
    background_dist: str,
    gamma_shape: float,
    gamma_scale: float,
    laplace_scale: float,
    outprefix: Optional[str] = None,
    trait_name: Optional[str] = None,
    write_effect_tables: bool = False,
    grm: np.ndarray | None = None,
    snps_only: bool = False,
    progress_callback: Any | None = None,
    progress_total_hint: Optional[int] = None,
    progress_every: int = 10_000,
) -> dict[str, Any]:
    # Keep passing `residual_var` for API compatibility. Rust now derives the
    # effective residual variance as `1 - bg_pve - causal_pve` under the
    # final-variance PVE definition and ignores this input for variance scaling.
    fixed_path = f"{outprefix}.fixed.effects.tsv" if (outprefix and write_effect_tables) else None
    random_path = (
        f"{outprefix}.random.effects.tsv" if (outprefix and write_effect_tables) else None
    )
    return dict(
        g2p_simulate(
            gfile,
            chunk_size=100_000,
            maf_threshold=float(maf),
            max_missing_rate=float(missing_rate),
            het_threshold=None if het_threshold is None else float(het_threshold),
            seed=int(seed),
            residual_var=float(residual_var),
            bg_pve=float(bg_pve),
            background_dist=str(background_dist),
            gamma_shape=float(gamma_shape),
            gamma_scale=float(gamma_scale),
            laplace_scale=float(laplace_scale),
            causal_count=int(max(0, causal)),
            causal_pve=None if cs_pve is None else float(cs_pve),
            bim_ranges=list(bimranges),
            logic_mode=logic_mode,
            logic_gate_count=None if logic_gate_count is None else int(logic_gate_count),
            logic_k_min=int(logic_k_min),
            logic_k_max=int(logic_k_max),
            logic_ld_max=float(logic_ld_max),
            logic_het_max=float(logic_het_max),
            logic_af_min=float(logic_af_min),
            logic_af_max=float(logic_af_max),
            logic_max_iter=int(logic_max_iter),
            logic_window_bp=logic_window_bp,
            logic_effect_model=str(logic_effect_model),
            delimiter=None,
            snps_only=bool(snps_only),
            pheno_prefix=outprefix,
            fixed_effects_path=fixed_path,
            random_effects_path=random_path,
            causal_sites_path=None,
            trait_name=trait_name,
            na_rate=0.1,
            grm=grm,
            progress_callback=progress_callback,
            progress_total_hint=(
                None if progress_total_hint is None else int(max(0, progress_total_hint))
            ),
            progress_every=int(max(1, progress_every)),
        )
    )


def simulate_phenotype_from_genofile(
    gfile: str,
    mode: Literal["single", "garfield"] = "single",
    chunk_size: int = 100_000,
    seed: int = 1,
    maf: float = 0.02,
    missing_rate: float = 0.05,
    het: float | None = None,
    pve: float = 0.5,
    ve: float = 1.0,
    windows: int = 50_000,
    and_k_min: int = 2,
    and_k_max: int = 4,
    and_ld_max: float = 0.2,
    and_het_max: float = 0.05,
    and_af_min: float = 0.02,
    and_af_max: float = 0.98,
    and_target_pve: float = 0.2,
    and_max_iter: int = 100,
    logic_effect_model: Literal["gate", "centered_interaction"] = "gate",
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    logic_mode = "and" if str(mode).lower() == "garfield" else None
    gate_count = 1 if logic_mode is not None else None
    res = _run_rust_simulation(
        gfile=gfile,
        seed=int(seed),
        maf=float(maf),
        missing_rate=float(missing_rate),
        het_threshold=None if het is None else float(het),
        bg_pve=float(pve),
        residual_var=float(ve),
        causal=1,
        cs_pve=float(and_target_pve) if logic_mode is not None else None,
        bimranges=[],
        logic_mode=logic_mode,
        logic_gate_count=gate_count,
        logic_k_min=int(and_k_min),
        logic_k_max=int(and_k_max),
        logic_ld_max=float(and_ld_max),
        logic_het_max=float(and_het_max),
        logic_af_min=float(and_af_min),
        logic_af_max=float(and_af_max),
        logic_max_iter=int(and_max_iter),
        logic_window_bp=int(windows) if logic_mode is not None else None,
        logic_effect_model=str(logic_effect_model),
        background_dist="normal",
        gamma_shape=1.0,
        gamma_scale=1.0,
        laplace_scale=1.0,
        outprefix=None,
        trait_name=None,
        write_effect_tables=False,
        grm=None,
        snps_only=False,
    )
    y = np.asarray(res["phenotype"], dtype=np.float64).reshape(-1, 1)
    outsites = [(str(c), int(s), int(e)) for c, s, e in list(res.get("causal_sites", []))]
    return y, outsites


def write_phenotypes(outprefix: str, sample_ids: np.ndarray, y: np.ndarray, seed: int = 1):
    sample_ids = np.asarray(sample_ids, dtype=str).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)

    pheno3 = np.empty((len(sample_ids), 3), dtype=object)
    pheno3[:, 0] = sample_ids
    pheno3[:, 1] = sample_ids
    pheno3[:, 2] = yv
    np.savetxt(
        f"{outprefix}.pheno",
        pheno3,
        delimiter="\t",
        fmt=["%s", "%s", "%.6f"],
    )

    pheno2 = np.column_stack([sample_ids, yv.astype(object)])
    np.savetxt(
        f"{outprefix}.pheno.txt",
        pheno2,
        delimiter="\t",
        fmt=["%s", "%.6f"],
        header="IID\tPHENO",
        comments="",
    )

    rng = np.random.default_rng(int(seed))
    pheno2_na = pheno2.astype(object, copy=True)
    k = int(round(len(sample_ids) * 0.1))
    if k > 0:
        idx = rng.choice(len(sample_ids), size=k, replace=False)
        pheno2_na[idx, 1] = "NA"
    np.savetxt(
        f"{outprefix}.pheno.NA.txt",
        pheno2_na,
        delimiter="\t",
        fmt=["%s", "%s"],
        header="IID\tPHENO",
        comments="",
    )


def write_sites(outprefix: str, sites: list[tuple[str, int, int]]):
    if not sites:
        return
    arr = np.asarray(sites, dtype=object)
    np.savetxt(f"{outprefix}.causal.sites.tsv", arr, delimiter="\t", fmt=["%s", "%d", "%d"])


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


def _background_effect_label(background_source: str) -> str:
    if str(background_source).strip().lower() == "grm":
        return "breeding values"
    return "background effects"


def _background_effect_axis_label(background_source: str) -> str:
    if str(background_source).strip().lower() == "grm":
        return "Breeding value"
    return "Effect"


def _plot_random_effect_distribution(
    *,
    effects_tsv: str,
    out_pdf: str,
    trait_name: str,
    background_dist: str,
    background_source: str = "marker",
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> None:
    phase_total = 5
    is_grm = str(background_source).strip().lower() == "grm"

    def _report(phase: str, step: int) -> None:
        if progress_callback is not None:
            progress_callback(str(phase), int(step), phase_total)

    if not os.path.exists(effects_tsv):
        raise FileNotFoundError(f"Random effects table not found: {effects_tsv}")

    effects = np.atleast_1d(
        np.loadtxt(
            effects_tsv,
            delimiter="\t",
            skiprows=1,
            usecols=[4],
            dtype=np.float64,
        )
    )
    effects = np.asarray(effects, dtype=np.float64)
    effects = effects[np.isfinite(effects)]
    if effects.size == 0:
        raise ValueError("No finite random effects found for plotting.")
    _report("load-table", 1)

    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    import matplotlib.pyplot as plt
    _report("init-plotting", 2)

    fig, ax_hist = plt.subplots(1, 1, figsize=(7.4, 4.8))
    fig.patch.set_facecolor("white")

    edges = _histogram_edges(effects)
    curve_xs = None
    curve_ys = None
    curve_color = "#C44E52"
    curve_label = None
    if is_grm:
        _report("fit-normal-curve", 2)
        sigma = float(np.std(effects))
        if effects.size >= 2 and sigma > 1e-12:
            mu = float(np.mean(effects))
            curve_xs = np.linspace(float(edges[0]), float(edges[-1]), 256)
            z = (curve_xs - mu) / sigma
            curve_ys = np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))
            curve_label = "normal fit"
        _report("fit-normal-curve", 3)
    else:
        _report("fit-kde-curve", 2)
        try:
            from scipy.stats import gaussian_kde
        except Exception:
            gaussian_kde = None
        if gaussian_kde is not None and effects.size >= 8 and float(np.std(effects)) > 1e-12:
            curve_xs = np.linspace(float(edges[0]), float(edges[-1]), 256)
            kde = gaussian_kde(effects)
            curve_ys = kde(curve_xs)
            curve_label = "kde"
        _report("fit-kde-curve", 3)

    ax_hist.hist(
        effects,
        bins=edges,
        density=True,
        color="#4C78A8",
        alpha=0.80,
        edgecolor="white",
        linewidth=0.8,
    )
    ax_hist.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.85)

    if curve_xs is not None and curve_ys is not None:
        ax_hist.plot(curve_xs, curve_ys, color=curve_color, linewidth=1.8, label=curve_label)

    ax_hist.set_title(f"{trait_name} {_background_effect_label(background_source)}", fontsize=12)
    ax_hist.set_xlabel(_background_effect_axis_label(background_source))
    ax_hist.set_ylabel("Density")
    ax_hist.grid(axis="y", color="#D1D5DB", alpha=0.55, linewidth=0.8)
    ax_hist.text(
        0.98,
        0.97,
        (
            f"normal (grm), entries={effects.size:,}"
            if str(background_source).strip().lower() == "grm"
            else f"{background_dist}, entries={effects.size:,}"
        ),
        transform=ax_hist.transAxes,
        va="top",
        ha="right",
        fontsize=9.2,
        color="#374151",
    )
    if curve_label is not None:
        ax_hist.legend(loc="upper left", frameon=False, fontsize=9.0)
    _report("render-figure", 4)

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    _report("write-pdf", 5)


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx sim",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx sim -bfile geno_prefix -o out -prefix demo",
                "jx sim -vcf geno.vcf.gz -causal 3 -cs-pve 0.15 -o out",
                "jx sim -bfile geno_prefix -logic-gate 3,2 and -bg-pve 0.4 -o out",
                "jx sim -bfile geno_prefix -k panel.grm.npy -o out -prefix demo",
            ]
        ),
        description="JanusX simulation: phenotype from existing genotype (Rust-first)",
    )

    required_group = parser.add_argument_group("Required arguments")
    optional_group = parser.add_argument_group("Optional arguments")
    filter_group = parser.add_argument_group("Genotype filtering")
    pve_group = parser.add_argument_group("Variance / PVE model")
    causal_group = parser.add_argument_group("Causal terms")
    bg_group = parser.add_argument_group("Background effect distribution")

    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument("-vcf", "--vcf", type=str, help="Input genotype file in VCF format (.vcf or .vcf.gz).")
    geno_group.add_argument("-hmp", "--hmp", type=str, help="Input genotype file in HMP format (.hmp or .hmp.gz).")
    geno_group.add_argument("-bfile", "--bfile", type=str, help="Input genotype in PLINK binary format (prefix for .bed, .bim, .fam).")
    geno_group.add_argument(
        "-file",
        "--file",
        type=str,
        help=(
            "Input genotype numeric matrix (.txt/.tsv/.csv/.npy) or prefix. "
            "Requires sibling prefix.id. "
            "Optional site metadata: prefix.site or prefix.bim."
        ),
    )

    optional_group.add_argument("-o", "--out", type=str, default=".", help="Output directory for results (default: .).")
    optional_group.add_argument("-prefix", "--prefix", type=str, default=None, help="Prefix for output files (default: None).")
    optional_group.add_argument(
        "-k",
        "--grm",
        type=str,
        default=None,
        help="Optional precomputed GRM/kernel (.npy or text). If set, background effects are drawn from this kernel instead of marker-wise genotype effects.",
    )
    optional_group.add_argument(
        "-kid",
        "--grm-id",
        type=str,
        default=None,
        help="Optional GRM sample ID file. Default auto-detect: <grm>.id.",
    )
    optional_group.add_argument("--seed", type=int, default=None, help="Random seed. If omitted, use current time.")

    filter_group.add_argument(
        "-chunksize",
        "--chunksize",
        type=int,
        default=100_000,
        help="Compatibility placeholder; simulation core streams in Rust (default: 100,000).",
    )
    filter_group.add_argument("-maf", "--maf", type=float, default=0.02, help="Exclude variants with minor allele frequency lower than a threshold (default: 0.02).")
    filter_group.add_argument("-geno", "--geno", type=float, default=0.05, help="Exclude variants with missing call frequencies greater than a threshold (default: 0.05).")
    filter_group.add_argument(
        "-het",
        "--het",
        type=float,
        default=None,
        help="Optional maximum heterozygosity rate per variant in [0,1]. Disabled by default.",
    )

    pve_group.add_argument(
        "-bg-pve",
        "--bg-pve",
        "-pve",
        "--pve",
        "--polygenic-pve",
        dest="bg_pve",
        type=float,
        default=0.5,
        help=(
            "Background/polygenic variance contribution Var(Mg) in the final phenotype. "
            "Together with --cs-pve, this determines effective residual variance as "
            "1 - bg_pve - cs_pve. Default: %(default)s."
        ),
    )
    pve_group.add_argument(
        "-ve",
        "--ve",
        type=float,
        default=1.0,
        help=(
            "Deprecated compatibility input. Ignored for variance scaling under the "
            "final-variance PVE definition; effective VE is 1 - bg_pve - cs_pve."
        ),
    )
    causal_group.add_argument("-causal", "--causal", type=int, default=1, help="Number of causal SNP terms to sample.")
    causal_group.add_argument(
        "-cs-pve",
        "--cs-pve",
        type=float,
        default=None,
        help=(
            "Overall causal variance contribution Var(Qγ) in the final phenotype. "
            "If omitted, Rust uses Garfield default: min(0.05 * number_of_terms, cap)."
        ),
    )
    causal_group.add_argument(
        "-logic-gate",
        "--logic-gate",
        nargs=2,
        metavar=("SIZE", "MODE"),
        default=None,
        help=(
            "Logic-gate causal terms. SIZE is k or k,count; MODE is and|or|and_or. "
            "Example: -logic-gate 3,2 and."
        ),
    )
    causal_group.add_argument(
        "--logic-effect-model",
        type=str,
        choices=["gate", "centered_interaction"],
        default="gate",
        help=(
            "Logic-term effect model. 'gate' keeps the current mean-centered gate effect; "
            "'centered_interaction' removes member main effects and builds a weak-marginal "
            "interaction benchmark."
        ),
    )
    causal_group.add_argument(
        "-bimrange",
        "--bimrange",
        action="append",
        default=[],
        help="Repeatable causal region: chr:start:end. Can be specified multiple times.",
    )
    # bg_group.add_argument(
    #     "-normal",
    #     "--normal",
    #     action="store_true",
    #     help="Use normal background effects g₀ᵢ ~ N(0,1). This is the default.",
    # )
    bg_group.add_argument(
        "-gamma",
        "--gamma",
        dest="gamma_shape",
        type=float,
        metavar="SHAPE",
        default=None,
        help=(
            "Use signed-gamma background effects with random sign. "
            "SHAPE must be in (0, 1]; SHAPE=1 is Laplace-like. "
            "The Gamma scale θ is fixed to 1. "
            "If omitted, normal background effects g₀ᵢ ~ N(0,1) are used."
        ),
    )
    
    # bg_group.add_argument(
    #     "-laplace",
    #     "--laplace",
    #     dest="laplace_scale",
    #     nargs=1,
    #     metavar=("SCALE",),
    #     default=None,
    #     help="Use Laplace background effects with the given scale.",
    # )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    gfile, prefix = determine_genotype_source(args)
    args.out = os.path.normpath(args.out if args.out is not None else ".")
    outstem = str(args.prefix).strip() if args.prefix is not None else prefix
    outprefix = os.path.join(args.out, outstem)
    os.makedirs(args.out, exist_ok=True, mode=0o755)
    configure_genotype_cache_from_out(args.out)

    log_path = f"{outprefix}.sim.log"
    logger = setup_logging(log_path)

    seed = int(args.seed) if args.seed is not None else int(time.time()) & 0x7FFFFFFF
    bimranges = [_parse_bimrange(x) for x in list(args.bimrange or [])]
    bg_dist, gamma_shape, gamma_scale, laplace_scale = _resolve_distribution(args)
    requested_bg_dist = bg_dist
    grm_forced_normal = bool(args.grm and bg_dist != "normal")
    if args.grm:
        bg_dist = "normal"
        gamma_shape = 1.0
        gamma_scale = 1.0
        laplace_scale = 1.0
    if bg_dist == "gamma":
        if not (0.0 < float(gamma_shape) <= 1.0):
            logger.error("--gamma SHAPE must be in (0, 1].")
            raise SystemExit(1)
    logic_mode, logic_gate_count, logic_k_min, logic_k_max, logic_window_bp = (
        _resolve_logic_config(args)
    )
    cs_pve = float(args.cs_pve) if args.cs_pve is not None else None
    logic_ld_max = 1.0
    logic_het_max = 1.0
    logic_af_min = 0.0
    logic_af_max = 1.0
    logic_max_iter = 256
    logic_effect_model = str(args.logic_effect_model).strip().lower()

    emit_cli_configuration(
        logger,
        app_title="JanusX - Simulation",
        config_title="SIMULATION CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "General",
                [
                    ("Genotype file", gfile),
                    ("MAF threshold", args.maf),
                    ("Missing threshold", args.geno),
                    ("Het threshold", "None" if args.het is None else args.het),
                    ("Background PVE", args.bg_pve),
                    ("Residual input (deprecated)", args.ve),
                    ("Background GRM", args.grm),
                    ("Causal count", args.causal),
                    ("Causal PVE", cs_pve),
                    ("Logic gate", "None" if logic_mode is None else f"{logic_mode} ({logic_k_min},{logic_k_max})"),
                    ("Logic gate count", logic_gate_count),
                    ("Logic window bp", logic_window_bp),
                    ("Logic effect model", logic_effect_model if logic_mode is not None else "None"),
                    ("Background dist", bg_dist),
                    ("SNPs only", False),
                    ("Gamma shape", gamma_shape if bg_dist == "gamma" else "None"),
                    # ("Laplace scale", laplace_scale if bg_dist == "laplace" else "None"),
                    ("BIM ranges", len(bimranges)),
                    ("Seed", seed),
                ],
            ),
        ],
        footer_rows=[("Output prefix", outprefix)],
        line_max_chars=60,
    )

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    if args.grm:
        checks.append(ensure_file_exists(logger, args.grm, "Background GRM"))
    if args.grm_id:
        checks.append(ensure_file_exists(logger, args.grm_id, "Background GRM ID file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    if not (0.0 <= float(args.bg_pve) <= 1.0):
        logger.error("--bg-pve/--pve must be in [0, 1].")
        raise SystemExit(1)
    if args.het is not None and not (0.0 <= float(args.het) <= 1.0):
        logger.error("--het must be in [0, 1].")
        raise SystemExit(1)
    if cs_pve is not None and not (0.0 <= float(cs_pve) <= 1.0):
        logger.error("--cs-pve must be in [0, 1].")
        raise SystemExit(1)
    if cs_pve is not None and float(args.bg_pve) + float(cs_pve) > 1.0:
        logger.error(
            "--bg-pve + --cs-pve must be <= 1 under the final-variance PVE definition. "
            "Got bg_pve=%.6g and cs_pve=%.6g.",
            float(args.bg_pve),
            float(cs_pve),
        )
        raise SystemExit(1)

    t_start = time.time()
    logger.info(f"Simulating phenotype from genotype: {gfile}")
    with CliStatus("Inspecting genotype input...", enabled=True) as task:
        sample_ids, n_sites = inspect_genotype_file(
            gfile,
            snps_only=False,
            maf=float(args.maf),
            missing_rate=float(args.geno),
            het=1.0 if args.het is None else float(args.het),
        )
        task.complete("Inspecting genotype input ...Finished")
    sample_ids = np.asarray(sample_ids, dtype=str)

    aligned_grm = None
    if args.grm:
        with CliStatus("Loading background GRM...", enabled=True) as task:
            aligned_grm, resolved_grm_id = load_and_align_grm(
                str(args.grm),
                sample_ids.tolist(),
                grm_id_path=args.grm_id,
                label="Background GRM",
            )
            task.complete("Loading background GRM ...Finished")
        logger.info(
            "Background GRM mode: using %s%s",
            format_path_for_display(str(args.grm)),
            (
                f" (ID: {format_path_for_display(str(resolved_grm_id))})"
                if resolved_grm_id is not None
                else " (sample order assumed to match genotype input)"
            ),
        )
        if grm_forced_normal:
            logger.warning(
                "Background distribution '%s' is ignored when --grm is provided; JanusX switches to Gaussian sample-level breeding values from the GRM.",
                requested_bg_dist,
            )

    scan_passes = _estimate_simulation_scan_passes(
        causal_count=int(args.causal),
        cs_pve=cs_pve,
        bimranges=bimranges,
        logic_mode=logic_mode,
        logic_gate_count=logic_gate_count,
    )
    progress_total_hint = int(max(0, n_sites))
    progress_total = int(max(1, progress_total_hint * max(1, scan_passes)))
    sim_pbar = ProgressAdapter(
        total=progress_total,
        desc="Simulation work",
        emit_done=False,
        force_animate=True,
    )
    progress_state = {"last_done": 0, "stage": "background"}
    stage_label = {
        "background": "background",
        "causal_additive": "causal-additive",
        "causal_logic": "causal-logic",
        "finalize": "finalize",
    }
    stage_pass = {
        "background": 1,
        "causal_additive": min(2, max(1, scan_passes)),
        "causal_logic": min(2, max(1, scan_passes)),
        "finalize": max(1, scan_passes),
    }

    def _simulation_progress(stage: str, done: int, total: int) -> None:
        stage_key = str(stage)
        total_now = int(total)
        done_now = int(done)
        if stage_key != progress_state["stage"]:
            progress_state["stage"] = stage_key
        if total_now > 0:
            done_now = max(0, min(done_now, total_now))
        delta = done_now - int(progress_state["last_done"])
        if delta > 0:
            sim_pbar.update(delta)
            progress_state["last_done"] = done_now
        if progress_total_hint > 0 and stage_key == "background":
            stage_done_now = min(done_now, progress_total_hint)
            stage_total_now = progress_total_hint
        elif progress_total_hint > 0 and stage_key in ("causal_additive", "causal_logic"):
            stage_done_now = max(0, done_now - progress_total_hint)
            stage_total_now = progress_total_hint
        elif stage_key == "finalize":
            stage_done_now = 1
            stage_total_now = 1
        else:
            stage_done_now = done_now
            stage_total_now = total_now
        pass_idx = int(stage_pass.get(stage_key, max(1, scan_passes)))
        if stage_total_now > 0:
            sim_pbar.set_postfix(
                # stage=stage_label.get(stage_key, stage_key),
                # pass_=f"{pass_idx}/{max(1, scan_passes)}",
                sites=f"{stage_done_now:,}/{stage_total_now:,}",
            )
        else:
            sim_pbar.set_postfix(
                # stage=stage_label.get(stage_key, stage_key),
                # pass_=f"{pass_idx}/{max(1, scan_passes)}",
                sites=f"{stage_done_now:,}",
            )

    sim_start = time.monotonic()
    try:
        res = _run_rust_simulation(
            gfile=gfile,
            seed=seed,
            maf=float(args.maf),
            missing_rate=float(args.geno),
            het_threshold=None if args.het is None else float(args.het),
            bg_pve=float(args.bg_pve),
            residual_var=float(args.ve),
            causal=int(args.causal),
            cs_pve=cs_pve,
            bimranges=bimranges,
            logic_mode=logic_mode,
            logic_gate_count=logic_gate_count,
            logic_k_min=int(logic_k_min),
            logic_k_max=int(logic_k_max),
            logic_ld_max=float(logic_ld_max),
            logic_het_max=float(logic_het_max),
            logic_af_min=float(logic_af_min),
            logic_af_max=float(logic_af_max),
            logic_max_iter=int(logic_max_iter),
            logic_window_bp=logic_window_bp,
            logic_effect_model=logic_effect_model,
            background_dist=bg_dist,
            gamma_shape=float(gamma_shape),
            gamma_scale=float(gamma_scale),
            laplace_scale=float(laplace_scale),
            outprefix=outprefix,
            trait_name=None,
            write_effect_tables=True,
            grm=aligned_grm,
            snps_only=False,
            progress_callback=_simulation_progress,
            progress_total_hint=progress_total_hint,
            progress_every=max(1, min(10_000, progress_total_hint // 200 if progress_total_hint > 0 else 10_000)),
        )
    finally:
        sim_pbar.finish()
        sim_pbar.close()
    sim_elapsed = max(0.0, time.monotonic() - sim_start)
    log_success(logger, f"Simulation ...Finished [{format_elapsed(sim_elapsed)}]")
    random_effects_tsv = f"{outprefix}.random.effects.tsv"
    random_effects_pdf = f"{outprefix}.random.effects.pdf"
    try:
        vis_source = str(res.get("background_source", "marker"))
        vis_label = (
            "GRM breeding values"
            if vis_source.strip().lower() == "grm"
            else _background_effect_label(vis_source)
        )
        vis_total = 5
        vis_pbar = ProgressAdapter(
            total=vis_total,
            desc=f"Visualizing {vis_label}",
            show_remaining=False,
            force_animate=True,
        )
        vis_phase_labels = {
            "load-table": "loading effects",
            "init-plotting": "initializing matplotlib",
            "fit-normal-curve": "fitting normal curve",
            "fit-kde-curve": "estimating density curve",
            "render-figure": "rendering figure",
            "write-pdf": "writing pdf",
        }

        def _visualization_progress(phase: str, done: int, total: int) -> None:
            total_now = max(1, int(total))
            done_now = max(0, min(int(done), total_now))
            vis_pbar.update(done_now - getattr(_visualization_progress, "last_done", 0))
            _visualization_progress.last_done = done_now
            vis_pbar.set_postfix(
                stage=vis_phase_labels.get(str(phase), str(phase)),
                step=f"{done_now}/{total_now}",
            )

        _visualization_progress.last_done = 0  # type: ignore[attr-defined]
        vis_pbar.set_postfix(stage="queued", step=f"0/{vis_total}")
        vis_success = False
        try:
            _plot_random_effect_distribution(
                effects_tsv=random_effects_tsv,
                out_pdf=random_effects_pdf,
                trait_name=str(res.get("trait_name", "PHENO")),
                background_dist=str(bg_dist),
                background_source=str(res.get("background_source", "marker")),
                progress_callback=_visualization_progress,
            )
            vis_success = True
        finally:
            if vis_success:
                vis_pbar.finish()
            vis_pbar.close()
    except Exception as exc:
        logger.warning("Random effect distribution PDF was skipped: %s", exc)

    logger.info(
        "Final-variance PVE runtime: bg_pve=%s, causal_pve=%s, ve=%s.",
        res.get("bg_pve"),
        res.get("causal_pve"),
        res.get("ve"),
    )
    if len(sample_ids) != int(np.asarray(res["phenotype"]).reshape(-1).shape[0]):
        logger.warning("Sample count from inspection differs from Rust phenotype length.")

    logger.info("Outputs:")
    logger.info(f"  {format_path_for_display(f'{outprefix}.pheno')}")
    logger.info(f"  {format_path_for_display(f'{outprefix}.pheno.txt')}")
    logger.info(f"  {format_path_for_display(f'{outprefix}.pheno.NA.txt')}")
    logger.info(f"  {format_path_for_display(f'{outprefix}.fixed.effects.tsv')}")
    logger.info(f"  {format_path_for_display(f'{outprefix}.random.effects.tsv')}")
    if os.path.exists(random_effects_pdf):
        logger.info(f"  {format_path_for_display(random_effects_pdf)}")
    total_elapsed = max(0.0, time.time() - t_start)
    logger.info("Finished, total time: %.2f secs", total_elapsed)
    now = datetime.now()
    logger.info(
        f"{now.year}-{now.month}-{now.day} {now.hour:02d}:{now.minute:02d}:{now.second:02d}"
    )
    return 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
