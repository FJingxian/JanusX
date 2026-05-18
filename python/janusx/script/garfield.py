import argparse
import json
import os
import re
import socket
import time
from typing import Optional

import numpy as np

from janusx.assoc.workflow import load_phenotype
from janusx.gfreader import inspect_genotype_file
from janusx.gtools.reader import readanno
from janusx.script._common.colspec import parse_zero_based_index_specs
from janusx.script._common.config_render import emit_cli_configuration
from janusx.script._common.genoio import determine_genotype_source_from_args as determine_genotype_source
from janusx.script._common.genocache import configure_genotype_cache_from_out
from janusx.script._common.helptext import CliArgumentParser, cli_help_formatter
from janusx.script._common.log import setup_logging
from janusx.script._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_plink_prefix_exists,
    format_path_for_display,
)
from janusx.script._common.progress import ProgressAdapter
from janusx.script._common.status import CliStatus, log_success, print_failure, stdout_is_tty
from janusx.script._common.threads import apply_blas_thread_env, detect_effective_threads


try:
    from janusx.janusx import garfield_logic_search_bed
except Exception as _exc:  # pragma: no cover
    garfield_logic_search_bed = None  # type: ignore[assignment]
    _RUST_IMPORT_ERROR = _exc
else:
    _RUST_IMPORT_ERROR = None


# Python 仅负责 CLI 调度；训练/测试划分、null-model 残差化、ML 候选筛选、
# beam search 与最终导出都由 Rust 端统一完成。


class _GarfieldPhenoLogger:
    """Suppress duplicate phenotype selection line from shared loader."""

    def __init__(self, base_logger):
        self._base = base_logger

    def info(self, message, *args, **kwargs):
        msg = str(message)
        if msg.startswith("Phenotypes to be analyzed: "):
            return
        return self._base.info(message, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)


def _require_rust_backend() -> None:
    if garfield_logic_search_bed is None:
        raise ImportError(
            "janusx Rust extension does not provide the GARFIELD Rust pipeline API. "
            "Please rebuild/reinstall JanusX extension."
        ) from _RUST_IMPORT_ERROR


def _looks_numeric_token(token: object) -> bool:
    s = str(token).strip()
    if s == "":
        return False
    try:
        float(s)
        return True
    except Exception:
        return False


def _split_pheno_line(line: str) -> list[str]:
    if "\t" in line:
        return [x.strip() for x in line.split("\t")]
    if "," in line:
        return [x.strip() for x in line.split(",")]
    return [x.strip() for x in line.split()]


def _read_phenotype_header_names(phenofile: str) -> Optional[list[str]]:
    try:
        with open(phenofile, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = str(raw).rstrip("\r\n")
                if line.strip() == "":
                    continue
                parts = _split_pheno_line(line)
                if len(parts) <= 1:
                    return None
                names = [str(x).strip() for x in parts[1:]]
                if len(names) == 0:
                    return None
                if any((n != "") and (not _looks_numeric_token(n)) for n in names):
                    return names
                return None
    except Exception:
        return None
    return None


def _normalize_trait_names_from_header(pheno, phenofile: str):
    trait_names = [str(c) for c in pheno.columns]
    selected_ncol = pheno.attrs.get("selected_ncol", None)
    if not isinstance(selected_ncol, list) or len(selected_ncol) != len(trait_names):
        return pheno, trait_names

    header_names = _read_phenotype_header_names(phenofile)
    if not header_names:
        return pheno, trait_names

    mapped: list[str] = []
    changed = False
    for idx_obj, current in zip(selected_ncol, trait_names):
        cur = str(current).strip()
        numeric_like = cur.lstrip("+-").isdigit()
        if numeric_like:
            try:
                idx = int(idx_obj)
            except Exception:
                idx = -1
            if 0 <= idx < len(header_names):
                cand = str(header_names[idx]).strip()
                if cand != "":
                    mapped.append(cand)
                    if cand != cur:
                        changed = True
                    continue
        mapped.append(cur)

    if changed:
        pheno = pheno.copy()
        pheno.columns = mapped
        return pheno, mapped
    return pheno, trait_names


def _safe_trait_label(label: object) -> str:
    name = str(label).strip()
    if name == "":
        return "trait"
    return name.replace("/", "_").replace("\\", "_")


def _detect_response_mode(y: np.ndarray) -> tuple[str, np.ndarray, str]:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if y_arr.size == 0:
        raise ValueError("Phenotype vector is empty after sample alignment.")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("Phenotype contains non-finite values after sample alignment.")

    uniq = np.unique(y_arr)
    if uniq.size == 2:
        lo = float(uniq[0])
        hi = float(uniq[1])
        lo_mask = np.isclose(y_arr, lo, rtol=0.0, atol=1e-12)
        hi_mask = np.isclose(y_arr, hi, rtol=0.0, atol=1e-12)
        if np.all(lo_mask | hi_mask):
            y_bin = hi_mask.astype(np.float64)
            note = f", mapped({lo:g}->0,{hi:g}->1)"
            return ("binary", y_bin, note)

    return "continuous", y_arr.astype(np.float64, copy=False), ""


def _read_geneset_lines(path: str) -> list[list[str]]:
    genesets: list[list[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            genes = [g for g in re.split(r"[\s,;]+", line.strip()) if g]
            if genes:
                genesets.append(genes)
    return genesets


def _build_interval_groups(
    genefile: str,
    gff3: str,
    extension: int,
    scan_mode: str,
) -> tuple[list[str], list[list[tuple[str, int, int]]]]:
    genesets = _read_geneset_lines(genefile)
    dfgff3 = readanno(gff3, "ID").iloc[:, :4].set_index(3)
    dfgff3 = dfgff3.loc[~dfgff3.index.duplicated()]

    labels: list[str] = []
    groups: list[list[tuple[str, int, int]]] = []

    def _iv(gene: str) -> Optional[tuple[str, int, int]]:
        if gene not in dfgff3.index:
            return None
        chrom = str(dfgff3.loc[gene, 0])
        start = int(dfgff3.loc[gene, 1]) - int(extension)
        end = int(dfgff3.loc[gene, 2]) + int(extension)
        return (chrom, start, end)

    if scan_mode == "gene":
        for genes in genesets:
            for g in genes:
                iv = _iv(g)
                if iv is None:
                    continue
                labels.append(g)
                groups.append([iv])
    elif scan_mode in {"genepair", "geneset"}:
        for genes in genesets:
            if len(genes) < 2:
                continue
            ivs = [iv for iv in (_iv(g) for g in genes) if iv is not None]
            if len(ivs) < 2:
                continue
            labels.append("|".join(genes))
            groups.append(ivs)
    else:
        raise ValueError(f"unsupported scan_mode: {scan_mode}")

    return labels, groups


def _scan_mode_to_logic_unit_kind(scan_mode: str) -> str:
    mode = str(scan_mode).strip().lower()
    if mode == "window":
        return "window"
    if mode == "gene":
        return "gene"
    if mode in {"genepair", "geneset"}:
        return "geneset"
    raise ValueError(f"unsupported scan_mode: {scan_mode}")


def _derive_fold_from_holdout(val_frac: float) -> int:
    vf = float(val_frac)
    if not (0.0 < vf < 1.0):
        raise ValueError("--val-frac must be within (0, 1).")
    fold = max(2, int(round(1.0 / vf)))
    return fold


def _parse_beam_topk_spec(value: object) -> tuple[int | None, float | None]:
    raw = str(value).strip()
    if raw == "":
        return (None, None)
    try:
        num = float(raw)
    except Exception as exc:  # pragma: no cover - parser surface
        raise ValueError("-beam-topk must be a positive integer or a fraction within (0, 1].") from exc
    if not np.isfinite(num) or num <= 0.0:
        raise ValueError("-beam-topk must be > 0.")
    if num <= 1.0:
        return (None, float(num))
    count = int(round(num))
    if count <= 0:
        raise ValueError("-beam-topk count must be >= 1.")
    return (count, None)


def _write_logic_pseudo_map(path: str, result: dict) -> None:
    unit_kinds = [str(x) for x in result.get("unit_kinds", [])]
    positions = [int(x) for x in result.get("positions", [])]
    exprs = [str(x) for x in result.get("expressions", [])]
    train_scores = [float(x) for x in result.get("train_scores", [])]
    test_scores = [float(x) for x in result.get("test_scores", [])]
    n = min(len(unit_kinds), len(positions), len(exprs), len(train_scores), len(test_scores))
    with open(path, "w", encoding="utf-8") as fw:
        fw.write("chrom\tpos\texpression\ttrain_score\ttest_score\n")
        for i in range(n):
            fw.write(
                f"{unit_kinds[i]}\t{positions[i]}\t{exprs[i]}\t"
                f"{train_scores[i]:.12g}\t{test_scores[i]:.12g}\n"
            )


def _run_scan_with_progress(
    desc: str,
    *,
    use_spinner: bool,
    invoke,
):
    if not bool(use_spinner):
        with CliStatus(f"{desc}...", enabled=False, timeout=0.1):
            return invoke(None)

    progress: Optional[ProgressAdapter] = None
    done_seen = 0
    total_seen = 0

    def _progress_cb(done: int, total: int) -> None:
        nonlocal progress, done_seen, total_seen
        d = max(0, int(done))
        t = max(d, int(total))
        if progress is None and t > 0:
            progress = ProgressAdapter(
                total=t,
                desc=desc,
                show_spinner=True,
                show_postfix=True,
                show_remaining=True,
                emit_done=True,
                force_animate=True,
            )
        if progress is None:
            done_seen = max(done_seen, d)
            total_seen = max(total_seen, t)
            return
        delta = max(0, d - done_seen)
        if delta > 0:
            progress.update(delta)
        progress.set_postfix(progress=f"{d}/{t}")
        done_seen = max(done_seen, d)
        total_seen = max(total_seen, t)

    try:
        out = invoke(_progress_cb)
    except Exception:
        if progress is not None:
            progress.close()
        print_failure(f"{desc} ...Failed", force_color=True)
        raise

    if progress is not None:
        if total_seen > 0:
            if done_seen < total_seen:
                progress.update(total_seen - done_seen)
                done_seen = total_seen
            progress.set_postfix(progress=f"{done_seen}/{total_seen}")
            progress.finish()
        progress.close()
    return out


def main() -> None:
    _require_rust_backend()

    t_start = time.time()
    use_spinner = stdout_is_tty()

    parser = CliArgumentParser(prog="jx garfield", formatter_class=cli_help_formatter())

    required_group = parser.add_argument_group("Required Arguments")
    required_group.add_argument("-bfile", "--bfile", type=str, required=False, help="PLINK prefix (.bed/.bim/.fam).")
    required_group.add_argument("-p", "--pheno", type=str, required=False, help="Phenotype file.")

    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-Window",
        "--Window",
        action="store_true",
        dest="mode_window",
        help="Window scan mode.",
    )
    optional_group.add_argument(
        "-Gene",
        "--Gene",
        type=str,
        default=None,
        dest="gene_mode_file",
        help="Gene scan file (requires -gff/--gff3).",
    )
    optional_group.add_argument(
        "-GeneSet",
        "--GeneSet",
        type=str,
        default=None,
        dest="geneset_mode_file",
        help="Gene-set scan file (requires -gff/--gff3).",
    )
    optional_group.add_argument("-g", "--genefile", type=str, default=None, help=argparse.SUPPRESS)
    optional_group.add_argument("-gff", "--gff3", type=str, default=None, help="GFF3 annotation.")
    optional_group.add_argument(
        "--scan-mode",
        type=str,
        choices=["window", "gene", "genepair", "geneset"],
        default=None,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-n",
        "--n",
        action="extend",
        nargs="+",
        metavar="COL",
        default=None,
        type=str,
        dest="ncol",
        help=(
            "Phenotype column(s), zero-based index (excluding sample ID), comma list (e.g. 0,2), "
            "or numeric range (e.g. 0:2). Repeat this flag for multiple traits."
        ),
    )
    optional_group.add_argument("--ncol", action="extend", nargs="+", metavar="COL", default=None, type=str, dest="ncol", help=argparse.SUPPRESS)
    optional_group.add_argument("-ext", "--extension", type=int, default=100_000, help="Window extension in bp.")
    optional_group.add_argument("-slide", "-step", "--slide", "--step", dest="step", type=int, default=None, help="Window step size (default: extension/2).")
    optional_group.add_argument("-maf", "--maf", type=float, default=0.02, help="Minor allele frequency threshold.")
    optional_group.add_argument("-geno", "--geno", type=float, default=0.05, help="Maximum missing rate threshold.")
    optional_group.add_argument("-het", "--het", type=float, default=0.05, help="Maximum heterozygosity rate threshold; variants with het > threshold are dropped.")
    optional_group.add_argument("-dev", "--dev", action="store_true", help="Use MBIN (HET/DOM/REC) encoding instead of BIN.")
    optional_group.add_argument(
        "-k",
        "--fold",
        type=int,
        default=None,
        help="Optional stratified holdout fold. If omitted, GARFIELD uses the full data without train/test split.",
    )
    optional_group.add_argument(
        "-engine",
        "--engine",
        type=str,
        choices=["RF", "GBDT", "GBDT2"],
        default=None,
        help="Optional ML engine for candidate search. If omitted, skip ML and pass all unit variants directly to beam search.",
    )
    optional_group.add_argument(
        "-topk",
        "--topk",
        type=int,
        default=None,
        help="Top-k ML candidate features per unit. Ignored when -engine is omitted.",
    )
    optional_group.add_argument(
        "-permutation",
        "--permutation",
        "-premutation",
        action="store_true",
        dest="permutation",
        help="Use permutation importance for ML candidate ranking.",
    )
    optional_group.add_argument("-beam-width", "--beam-width", type=int, default=None, help="Beam search width (default: 50).")
    optional_group.add_argument("-layer", "--layer", type=int, default=None, help="Maximum beam-search rule depth (default: 5).")
    optional_group.add_argument(
        "-exh",
        "--exh",
        type=int,
        default=None,
        help="Exact exhaustive depth before beam pruning. Default is auto: `2` when -layer>=2, otherwise `1`.",
    )
    optional_group.add_argument(
        "-logic-gate",
        "--logic-gate",
        type=str,
        default="and_or",
        help="Allowed binary operators in beam search: and, or, or and_or.",
    )
    optional_group.add_argument(
        "--rank-score",
        type=str,
        choices=["interaction_gain", "raw"],
        default="interaction_gain",
        help="Beam ranking score. 'interaction_gain' favors weak-marginal but strong-combination rules; 'raw' keeps the original absolute score.",
    )
    optional_group.add_argument(
        "-beam-topk",
        "--beam-topk",
        type=str,
        default=None,
        help="Final top-k rules after test ranking; accept integer count or fraction in (0,1].",
    )
    optional_group.add_argument("-nsnp", "--nsnp", type=int, default=None, dest="beam_width_compat", help=argparse.SUPPRESS)
    optional_group.add_argument("-m", "--max-pick", type=int, default=None, dest="layer_compat", help=argparse.SUPPRESS)
    optional_group.add_argument(
        "--feature-source",
        type=str,
        choices=["bin", "mbin"],
        default=None,
        dest="feature_source_compat",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument("--top-k-validate", type=int, default=None, dest="beam_topk_compat", help=argparse.SUPPRESS)
    optional_group.add_argument("--val-frac", type=float, default=None, dest="val_frac_compat", help=argparse.SUPPRESS)
    optional_group.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    optional_group.add_argument("-t", "--thread", dest="thread", type=int, default=detect_effective_threads(), help="CPU threads.")
    optional_group.add_argument("--threads", dest="thread", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    optional_group.add_argument("-o", "--out", type=str, default=".", help="Output directory.")
    optional_group.add_argument("-prefix", "--prefix", type=str, default=None, help="Output prefix.")
    optional_group.add_argument(
        "-simbench",
        "--simbench",
        type=str,
        default=None,
        help="Optional simulation benchmark TSV (<prefix>.fixed.effects.tsv). Matching simulation combinations will be appended to GARFIELD rules output using the same residualized-phenotype statistics.",
    )

    args, extras = parser.parse_known_args()

    has_genotype = bool(args.bfile)
    has_pheno = bool(args.pheno)
    if not has_genotype and not has_pheno:
        parser.error("the following arguments are required: -p/--pheno and -bfile/--bfile")
    if not has_genotype:
        parser.error("the following arguments are required: -bfile/--bfile")
    if not has_pheno:
        parser.error("the following arguments are required: -p/--pheno")
    if len(extras) > 0:
        parser.error("unrecognized arguments: " + " ".join(extras))

    if int(args.extension) <= 0:
        parser.error("-ext/--extension must be > 0")
    if args.step is None:
        args.step = max(1, int(args.extension) // 2)
    elif int(args.step) <= 0:
        parser.error("-slide/--slide must be > 0")
    if not (0.0 <= float(args.maf) <= 0.5):
        parser.error("-maf/--maf must be in [0, 0.5]")
    if not (0.0 <= float(args.geno) <= 1.0):
        parser.error("-geno/--geno must be in [0, 1]")
    if not (0.0 <= float(args.het) <= 1.0):
        parser.error("-het/--het must be in [0, 1]")

    explicit_scan_flags = int(bool(args.mode_window)) + int(args.gene_mode_file is not None) + int(args.geneset_mode_file is not None)
    if explicit_scan_flags > 1:
        parser.error("Use only one of -Window, -Gene, or -GeneSet.")
    if args.scan_mode is not None and explicit_scan_flags > 0:
        parser.error("Do not mix hidden --scan-mode with -Window/-Gene/-GeneSet.")
    if args.gene_mode_file is not None and args.genefile is not None:
        parser.error("Do not provide both -Gene and -g/--genefile.")
    if args.geneset_mode_file is not None and args.genefile is not None:
        parser.error("Do not provide both -GeneSet and -g/--genefile.")

    if args.gene_mode_file is not None:
        args.scan_mode = "gene"
        args.genefile = args.gene_mode_file
    elif args.geneset_mode_file is not None:
        args.scan_mode = "geneset"
        args.genefile = args.geneset_mode_file
    elif args.mode_window:
        args.scan_mode = "window"
    elif args.scan_mode is None:
        args.scan_mode = "window"
    elif args.scan_mode == "genepair":
        args.scan_mode = "geneset"

    if args.scan_mode in {"gene", "geneset"} and not args.genefile:
        parser.error(f"scan-mode={args.scan_mode} requires -Gene/-GeneSet or --genefile.")

    args.split_requested = False
    if args.fold is None:
        if args.val_frac_compat is not None:
            try:
                args.fold = _derive_fold_from_holdout(float(args.val_frac_compat))
            except ValueError as e:
                parser.error(str(e))
            args.holdout_frac_requested = float(args.val_frac_compat)
            args.split_requested = True
        else:
            args.holdout_frac_requested = None
    else:
        args.holdout_frac_requested = 1.0 / float(args.fold)
        args.split_requested = True
    if args.fold is not None and int(args.fold) < 2:
        parser.error("-k/--fold must be >= 2")
    args.fold_runtime = int(args.fold) if args.fold is not None else 0

    args.beam_width = (
        int(args.beam_width) if args.beam_width is not None
        else int(args.beam_width_compat) if args.beam_width_compat is not None
        else 50
    )
    if int(args.beam_width) <= 0:
        parser.error("-beam-width must be > 0")

    args.layer = (
        int(args.layer) if args.layer is not None
        else int(args.layer_compat) if args.layer_compat is not None
        else 5
    )
    if int(args.layer) <= 0:
        parser.error("-layer must be > 0")
    if args.exh is None:
        args.exh = 2 if int(args.layer) >= 2 else 1
    if int(args.exh) <= 0:
        parser.error("-exh/--exh must be >= 1")
    args.exh = max(1, int(args.exh))

    if args.engine is None:
        if args.topk is None:
            args.topk = 0
        elif int(args.topk) < 0:
            parser.error("-topk/--topk must be >= 0 when -engine is omitted")
    else:
        if args.topk is None:
            args.topk = min(100, int(args.beam_width))
        if int(args.topk) <= 0:
            parser.error("-topk/--topk must be > 0")

    if args.beam_topk is None:
        if args.beam_topk_compat is not None:
            args.beam_topk = str(int(args.beam_topk_compat))
        else:
            args.beam_topk = "0.1"
    try:
        args.beam_topk_count, args.beam_topk_ratio = _parse_beam_topk_spec(args.beam_topk)
    except ValueError as e:
        parser.error(str(e))

    args.feature_source = (
        str(args.feature_source_compat).lower()
        if args.feature_source_compat is not None
        else ("mbin" if bool(args.dev) else "bin")
    )
    if args.feature_source not in {"bin", "mbin"}:
        parser.error("--feature-source must be one of: bin, mbin")

    if args.engine is not None:
        args.engine = str(args.engine).upper()
    logic_gate_map = {
        "a": "and",
        "and": "and",
        "o": "or",
        "or": "or",
        "ao": "and_or",
        "and_or": "and_or",
        "andor": "and_or",
        "and-or": "and_or",
    }
    logic_gate_key = str(args.logic_gate).strip().lower()
    if logic_gate_key not in logic_gate_map:
        parser.error("-logic-gate must be one of: and, or, and_or")
    args.logic_gate = logic_gate_map[logic_gate_key]

    try:
        args.ncol = parse_zero_based_index_specs(args.ncol, label="-n/--n")
    except ValueError as e:
        parser.error(str(e))

    detected_threads = detect_effective_threads()
    if int(args.thread) <= 0:
        args.thread = int(detected_threads)
    if int(args.thread) > int(detected_threads):
        args.thread = int(detected_threads)

    gfile, prefix = determine_genotype_source(args)
    input_kind = "bfile"

    args.out = os.path.normpath(args.out if args.out is not None else ".")
    outprefix = os.path.join(args.out, prefix)
    os.makedirs(args.out, mode=0o755, exist_ok=True)
    configure_genotype_cache_from_out(args.out)

    log_path = os.path.join(args.out, f"{prefix}.garfield.log")
    logger = setup_logging(log_path)
    apply_blas_thread_env(int(args.thread))

    ml_skipped = args.engine is None
    engine_runtime = "none" if ml_skipped else str(args.engine)
    logic_gate_runtime = args.logic_gate

    checks: list[bool] = []
    checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    checks.append(ensure_file_exists(logger, args.pheno, "Phenotype file"))
    if args.genefile:
        checks.append(ensure_file_exists(logger, args.genefile, "Gene file"))
    if args.gff3:
        checks.append(ensure_file_exists(logger, args.gff3, "GFF3 file"))
    if args.simbench:
        checks.append(ensure_file_exists(logger, args.simbench, "Simulation benchmark TSV"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    feature_source = str(args.feature_source).lower()
    if feature_source not in {"bin", "mbin"}:
        raise ValueError(f"Unsupported feature-source: {args.feature_source}")

    sample_ids, _ = inspect_genotype_file(gfile)

    general_rows = [
        ("Genotype input", gfile),
        ("Input kind", input_kind),
        ("Execution route", "direct Rust BED pipeline"),
        ("Encoding", feature_source),
        ("Phenotype", args.pheno),
        ("Scan mode", args.scan_mode),
        ("Gene file", args.genefile),
        ("GFF3", args.gff3),
        ("MAF", float(args.maf)),
        ("Missing max", float(args.geno)),
        ("Het max", float(args.het)),
        ("Extension", int(args.extension)),
        ("Slide", int(args.step)),
        ("Split", f"{int(args.fold)}-fold stratified holdout" if args.split_requested else "none (full data)"),
        ("Engine", "none (skip ML)" if ml_skipped else args.engine),
        ("Permutation", bool(args.permutation)),
        ("ML top-k", "all variants (skip ML)" if ml_skipped else int(args.topk)),
        ("Beam width", int(args.beam_width)),
        ("Layer", int(args.layer)),
        ("Exhaustive depth", int(args.exh)),
        ("Logic gate", args.logic_gate),
        ("Rank score", str(args.rank_score)),
        ("Beam top-k", str(args.beam_topk)),
        ("Sim bench", args.simbench),
        ("Seed", int(args.seed)),
    ]

    emit_cli_configuration(
        logger,
        app_title="JanusX - GARFIELD (Rust)",
        config_title="GARFIELD CONFIG",
        host=socket.gethostname(),
        sections=[("General", general_rows)],
        footer_rows=[("Threads", f"{int(args.thread)} ({int(detected_threads)} available)"), ("Output prefix", outprefix)],
        line_max_chars=60,
    )

    sample_ids = np.asarray(sample_ids, dtype=str)
    if len(sample_ids) == 0:
        raise ValueError("No sample IDs found in genotype input.")

    with CliStatus("Loading phenotype...", enabled=use_spinner) as task:
        try:
            pheno = load_phenotype(args.pheno, args.ncol, _GarfieldPhenoLogger(logger), id_col=0)
        except Exception:
            task.fail("Loading phenotype ...Failed")
            raise
        task.complete("Loading phenotype ...Finished")

    if pheno.shape[1] == 0:
        raise ValueError("No phenotype columns to analyze.")
    pheno, trait_names = _normalize_trait_names_from_header(pheno, args.pheno)
    logger.info("Phenotype columns: " + ", ".join(trait_names))
    if args.split_requested:
        if ml_skipped:
            logger.info(
                "Rust GARFIELD pipeline: stratified split -> residualize train/test -> "
                "direct beam search on full unit variants -> test ranking."
            )
        else:
            logger.info(
                "Rust GARFIELD pipeline: stratified split -> residualize train/test -> "
                "ML on residualized train phenotype -> beam search -> test ranking."
            )
    else:
        if ml_skipped:
            logger.info(
                "Rust GARFIELD pipeline: no train/test split -> residualize full phenotype -> "
                "direct beam search on full unit variants -> full-data ranking."
            )
        else:
            logger.info(
                "Rust GARFIELD pipeline: no train/test split -> residualize full phenotype -> "
                "ML + beam search on full data -> full-data ranking."
            )

    pheno_all_ids = set(pheno.index.astype(str).to_numpy())
    sample_pool = [sid for sid in sample_ids.tolist() if sid in pheno_all_ids]
    if len(sample_pool) == 0:
        raise ValueError("No overlapping samples between genotype and phenotype.")

    group_labels: list[str] = []
    group_intervals: list[list[tuple[str, int, int]]] = []
    if args.scan_mode in {"gene", "geneset"}:
        if not args.genefile or not args.gff3:
            raise ValueError(f"scan-mode={args.scan_mode} requires both --genefile and --gff3.")
        group_labels, group_intervals = _build_interval_groups(
            args.genefile,
            args.gff3,
            int(args.extension),
            args.scan_mode,
        )
        if len(group_intervals) == 0:
            raise ValueError(f"No valid groups built for scan-mode={args.scan_mode}.")
        logger.info(f"Prepared {len(group_intervals)} interval groups for {args.scan_mode} scan.")

    used_trait_labels: dict[str, int] = {}
    saved = 0
    summary_rows: list[dict[str, object]] = []
    fold_from_val = int(args.fold_runtime)
    implied_val_frac = (1.0 / float(fold_from_val)) if fold_from_val >= 2 else None
    if (
        args.split_requested
        and args.holdout_frac_requested is not None
        and implied_val_frac is not None
        and abs(implied_val_frac - float(args.holdout_frac_requested)) > 0.02
    ):
        logger.warning(
            f"Requested holdout={float(args.holdout_frac_requested):.4g} is approximated by stratified fold={fold_from_val} "
            f"(holdout={implied_val_frac:.4g}) in Rust GARFIELD."
        )

    for trait_idx, trait in enumerate(pheno.columns):
        if trait_idx > 0:
            logger.info("")

        trait_name = str(trait)
        pheno_col = pheno[trait].dropna()
        pheno_ids = set(pheno_col.index.astype(str).to_numpy())
        common_ids = [sid for sid in sample_pool if sid in pheno_ids]
        if len(common_ids) == 0:
            logger.warning(f"No overlapping samples for trait '{trait_name}' after dropna; skipped.")
            continue

        y_raw = pheno_col.loc[common_ids].to_numpy(dtype=float)
        response_mode, y_common, response_note = _detect_response_mode(y_raw)
        logger.info(f"{trait_name} (n={len(common_ids)}, response={response_mode}{response_note})")
        if response_mode != "continuous":
            raise ValueError(
                f"GARFIELD Rust pipeline currently supports continuous phenotypes only; "
                f"trait '{trait_name}' is {response_mode}."
            )
        base_trait = _safe_trait_label(trait_name)
        count = used_trait_labels.get(base_trait, 0) + 1
        used_trait_labels[base_trait] = count
        suffix = base_trait if count == 1 else f"{base_trait}.{count}"
        trait_outprefix = f"{outprefix}.{suffix}"
        trait_seed = int(args.seed) + trait_idx
        scan_desc = {
            "window": f"Scanning windows for '{trait_name}'",
            "gene": f"Scanning genes for '{trait_name}'",
            "genepair": f"Scanning gene pairs for '{trait_name}'",
            "geneset": f"Scanning gene sets for '{trait_name}'",
        }.get(str(args.scan_mode), f"Rust GARFIELD search for '{trait_name}'")
        logic_unit_kind = _scan_mode_to_logic_unit_kind(args.scan_mode)
        rust_groups = group_intervals if args.scan_mode != "window" else None
        rust_group_names = group_labels if args.scan_mode != "window" else None
        trait_logic_prefix = f"{trait_outprefix}.garfield"
        # Rust handles stratified split plus train/test residualization internally
        # before ML candidate search and beam search are executed.
        result = _run_scan_with_progress(
            scan_desc,
            use_spinner=use_spinner,
            invoke=lambda progress_cb: garfield_logic_search_bed(
                gfile,
                np.asarray(y_common, dtype=np.float64),
                sample_ids=list(common_ids),
                unit_kind=logic_unit_kind,
                groups=rust_groups,
                group_names=rust_group_names,
                extension=int(args.extension),
                step=int(args.step),
                bin_mode=feature_source,
                ml_method=str(engine_runtime).lower(),
                ml_importance="permutation" if bool(args.permutation) else "imp",
                ml_top_k=int(args.topk),
                ml_top_frac=0.0,
                permutation_repeats=5,
                permutation_scoring="auto",
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=1,
                min_samples_split=2,
                bootstrap=True,
                feature_subsample=0.0,
                fold=fold_from_val,
                seed=trait_seed,
                max_pick=int(args.layer),
                exhaustive_depth=int(args.exh),
                beam_width=int(args.beam_width),
                logic_gate=str(args.logic_gate).lower(),
                rank_score=str(args.rank_score),
                candidate_keep_ratio=0.10,
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
                het_threshold=float(args.het),
                snps_only=False,
                block_cols=65536,
                threads=int(args.thread),
                low=-5.0,
                high=5.0,
                max_iter=50,
                tol=1e-3,
                add_intercept=True,
                exact_n_max=15000,
                require_lapack=False,
                out_prefix=trait_logic_prefix,
                simbench_path=args.simbench,
                top_rules_per_unit=1,
                max_output_rules=int(args.beam_topk_count or 0),
                max_output_ratio=float(args.beam_topk_ratio or 0.0),
                progress_callback=progress_cb,
                progress_every=0,
            ),
        )

        n_rules = int(result.get("n_rules", 0))
        if n_rules <= 0:
            logger.warning(f"No GARFIELD rules survived Rust search for trait '{trait_name}', skipped.")
            continue

        _write_logic_pseudo_map(f"{trait_logic_prefix}.pseudo", result)
        split_applied = bool(result.get("split_applied", False))
        with open(f"{trait_outprefix}.garfield.run_config.json", "w", encoding="utf-8") as fw:
            json.dump(
                {
                    "route": "rust-bed",
                    "split_applied": split_applied,
                    "scan_mode": args.scan_mode,
                    "unit_kind": logic_unit_kind,
                    "response": response_mode,
                    "engine": args.engine,
                    "engine_runtime": engine_runtime,
                    "ml_skipped": ml_skipped,
                    "permutation": bool(args.permutation),
                    "layer": int(args.layer),
                    "exhaustive_depth": int(args.exh),
                    "beam_width": int(args.beam_width),
                    "rank_score": str(args.rank_score),
                    "ml_top_k": (None if ml_skipped else int(args.topk)),
                    "extension": int(args.extension),
                    "step": int(args.step),
                    "logic_gate_requested": args.logic_gate,
                    "logic_gate_runtime": logic_gate_runtime,
                    "beam_topk": str(args.beam_topk),
                    "beam_topk_count_runtime": int(result.get("n_rules", 0)),
                    "beam_topk_ratio_runtime": (
                        float(args.beam_topk_ratio) if args.beam_topk_ratio is not None else None
                    ),
                    "val_frac_requested": (
                        float(args.holdout_frac_requested)
                        if args.holdout_frac_requested is not None
                        else None
                    ),
                    "fold_effective": int(fold_from_val) if split_applied else None,
                    "holdout_frac_effective": (
                        float(implied_val_frac) if split_applied and implied_val_frac is not None else None
                    ),
                    "ranking_dataset": "test" if split_applied else "full",
                    "feature_source": feature_source,
                    "maf": float(args.maf),
                    "geno": float(args.geno),
                    "het": float(args.het),
                    "simbench_path": args.simbench,
                    "simbench_rows": int(result.get("n_simbench", 0)),
                    "seed": trait_seed,
                },
                fw,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(
            f"GARFIELD null-model PVE for '{trait_name}': "
            f"train={float(result.get('train_pve', float('nan'))):.4g}, "
            f"test={float(result.get('test_pve', float('nan'))):.4g}"
        )
        if args.simbench:
            logger.info(
                f"GARFIELD simbench rows appended for '{trait_name}': "
                f"{int(result.get('n_simbench', 0))}"
            )
        rank_scores = (
            [float(x) for x in result.get("test_scores", [])]
            if split_applied
            else [float(x) for x in result.get("train_scores", [])]
        )
        best_holdout = rank_scores[0] if len(rank_scores) > 0 else float("nan")
        summary_rows.append(
            {
                "trait": trait_name,
                "n_samples": len(common_ids),
                "n_train": int(result.get("n_train", 0)),
                "n_holdout": int(result.get("n_test", 0)) if split_applied else 0,
                "holdout_kind": "test" if split_applied else "full",
                "best_holdout_score": best_holdout,
                "route": "rust-bed",
            }
        )
        log_success(
            logger,
            f"Saved GARFIELD output for trait '{trait_name}': "
            f"{format_path_for_display(trait_logic_prefix)}",
        )
        saved += 1

    if saved == 0:
        raise ValueError("No GARFIELD outputs were generated for the selected phenotype columns.")

    summary_path = f"{outprefix}.garfield.summary.tsv"
    with open(summary_path, "w", encoding="utf-8") as fw:
        fw.write("trait\tn_samples\tn_train\tn_holdout\tholdout_kind\tbest_holdout_score\troute\n")
        for row in summary_rows:
            fw.write(
                f"{row['trait']}\t{int(row['n_samples'])}\t{int(row['n_train'])}\t"
                f"{int(row['n_holdout'])}\t{row['holdout_kind']}\t"
                f"{float(row['best_holdout_score']):.12g}\t{row['route']}\n"
            )

    lt = time.localtime()
    log_success(
        logger,
        f"\nFinished GARFIELD. Total wall time: {round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}",
    )


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    main()
