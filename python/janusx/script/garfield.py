import argparse
import json
import os
import re
import socket
import time
from typing import Optional

import numpy as np

from janusx.assoc.workflow import (
    _inspect_genotype_with_status,
    _load_covariates_for_models,
    _load_phenotype_with_status,
)
from janusx.gtools.reader import readanno
from janusx.script._common.colspec import parse_zero_based_index_specs
from janusx.script._common.config_render import emit_cli_configuration
from janusx.script._common.genoio import determine_genotype_source_from_args as determine_genotype_source
from janusx.script._common.genocache import configure_genotype_cache_from_out
from janusx.script._common.grmio import load_and_align_grm
from janusx.script._common.helptext import CliArgumentParser, cli_help_formatter
from janusx.script._common.log import setup_logging
from janusx.script._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_plink_prefix_exists,
    format_path_for_display,
)
from janusx.script._common.progress import ProgressAdapter, build_rich_progress, rich_progress_available
from janusx.script._common.status import CliStatus, log_success, print_failure, stdout_is_tty, success_symbol
from janusx.script._common.threads import apply_outer_thread_cap, detect_effective_threads
from janusx.assoc.workflow_ui import _emit_plain_info_line


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


def _describe_rank_schedule(rank_score_runtime: str) -> str:
    mode = str(rank_score_runtime).strip().lower()
    if mode == "raw":
        return "all raw"
    m = re.fullmatch(r"gain_from_layer:(\d+)", mode)
    if m is not None:
        gain_start = int(m.group(1))
        if gain_start <= 1:
            return "gain from layer 1 (all gain)"
        return f"raw through layer {gain_start - 1}, gain from layer {gain_start}"
    return mode

def _emit_garfield_summary_to_log(logger, summary_rows: list[dict[str, object]]) -> None:
    if len(summary_rows) == 0:
        return
    logger.info("")
    logger.info("GARFIELD summary")
    logger.info("trait\tn_samples\tbest_score\troute")
    for row in summary_rows:
        logger.info(
            f"{row['trait']}\t{int(row['n_samples'])}\t"
            f"{float(row['best_score']):.12g}\t{row['route']}"
        )


def _load_json_if_exists(path: Optional[str]):
    if path is None:
        return None
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _remove_file_if_exists(path: Optional[str]) -> None:
    if path is None:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        return


def _split_structure_prior_payload(payload):
    if not isinstance(payload, dict):
        return (None, payload)
    prior_payload = payload.get("prior")
    posterior_payload = dict(payload)
    posterior_payload.pop("prior", None)
    return (prior_payload, posterior_payload)


class _GarfieldStageProgress:
    _HIDDEN_STAGES = {"null_prep", "structure_prep"}

    def __init__(self, *, scan_desc: str, enabled: bool) -> None:
        self.scan_desc = str(scan_desc)
        self.enabled = bool(enabled)
        self._rich = None
        self._tasks: dict[str, int] = {}
        self._current_stage: str | None = None
        self._adapter: ProgressAdapter | None = None
        self._adapter_done = 0
        if self.enabled and rich_progress_available():
            self._rich = build_rich_progress(
                show_spinner=True,
                show_bar=True,
                show_percentage=True,
                show_elapsed=True,
                show_remaining=True,
                field_templates=["{task.fields[postfix]}"],
                finished_text=f"[green]{success_symbol()}[/green]",
                transient=False,
            )
            if self._rich is not None:
                self._rich.start()

    def _label(self, stage: str) -> str:
        if stage == "null_prep":
            return "Preparing Bg Noise"
        if stage == "null_penalty":
            return "Estimating Bg Noise"
        if stage == "structure_prep":
            return "Preparing Priors"
        if stage == "structure_prior":
            return "Learning Priors"
        return self.scan_desc

    def update(self, stage: str, done: int, total: int, meta: object | None = None) -> None:
        stage_key = str(stage).strip().lower()
        if stage_key in self._HIDDEN_STAGES:
            return
        done_i = max(0, int(done))
        total_i = max(done_i, int(total))
        postfix = "" if meta in {None, ""} else str(meta)
        label = self._label(stage_key)

        if self._rich is not None:
            task_id = self._tasks.get(stage_key)
            if task_id is None:
                task_id = self._rich.add_task(
                    label,
                    total=max(1, total_i),
                    completed=min(done_i, max(1, total_i)),
                    postfix=postfix,
                )
                self._tasks[stage_key] = task_id
            else:
                self._rich.update(
                    task_id,
                    description=label,
                    total=max(1, total_i),
                    completed=min(done_i, max(1, total_i)),
                    postfix=postfix,
                )
            return

        if not self.enabled:
            return
        if self._current_stage != stage_key or self._adapter is None:
            if self._adapter is not None:
                self._adapter.finish()
                self._adapter.close()
            self._adapter = ProgressAdapter(
                total=max(1, total_i),
                desc=label,
                show_spinner=True,
                show_postfix=True,
                show_remaining=True,
                emit_done=True,
                force_animate=True,
            )
            self._current_stage = stage_key
            self._adapter_done = 0
        else:
            self._adapter.set_total(max(1, total_i))

        delta = max(0, done_i - self._adapter_done)
        if delta > 0:
            self._adapter.update(delta)
        if postfix != "":
            self._adapter.set_postfix(progress=f"{done_i}/{total_i}", detail=postfix)
        else:
            self._adapter.set_postfix(progress=f"{done_i}/{total_i}")
        self._adapter_done = done_i

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.finish()
            self._adapter.close()
            self._adapter = None
        if self._rich is not None:
            self._rich.stop()
            self._rich = None


def _run_scan_with_progress(
    desc: str,
    *,
    use_spinner: bool,
    invoke,
):
    if not bool(use_spinner):
        with CliStatus(f"{desc}...", enabled=False, timeout=0.1):
            return invoke(None)

    stage_progress = _GarfieldStageProgress(scan_desc=desc, enabled=bool(use_spinner))

    def _progress_cb(*event) -> None:
        if len(event) == 4:
            stage, done, total, meta = event
            stage_progress.update(str(stage), int(done), int(total), meta)
            return
        if len(event) == 2:
            done, total = event
            stage_progress.update("scan", int(done), int(total), None)
            return
        raise ValueError(f"unexpected GARFIELD progress event: {event!r}")

    try:
        out = invoke(_progress_cb)
    except Exception:
        stage_progress.close()
        print_failure(f"{desc} ...Failed", force_color=True)
        raise

    stage_progress.close()
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
    optional_group.add_argument("-step", "--step", dest="step", type=int, default=None, help="Window step size (default: extension/2).")
    optional_group.add_argument("-maf", "--maf", type=float, default=0.02, help="Minor allele frequency threshold.")
    optional_group.add_argument(
        "-geno",
        "--geno",
        type=float,
        default=0.05,
        help="Maximum pure-line missing rate threshold; heterozygotes (1) and true missing (NA) are both counted as missing.",
    )
    optional_group.add_argument(
        "-dev",
        "--dev",
        action="store_true",
        help="Deprecated compatibility flag. Under pure-line GARFIELD it aliases BIN encoding.",
    )
    optional_group.add_argument(
        "-k",
        "--grm",
        type=str,
        default=None,
        help="Optional precomputed GRM/kernel (.npy or text). If set, GARFIELD residualization uses this matrix on the aligned sample set instead of rebuilding GRM from BED.",
    )
    optional_group.add_argument(
        "-c",
        "--cov",
        action="append",
        type=str,
        default=None,
        dest="cov_inputs",
        help=(
            "Additional covariate input (repeatable). Each -c accepts either: "
            "(1) covariate file path, or "
            "    covariate file format: first column sample ID, remaining columns numeric covariates; or "
            "(2) single-site token chr:pos / chr:start:end (start must equal end, "
            "supports full-width colon). "
            "Examples: -c cov.tsv -c 1:1000 -c 1:1000:1000."
        ),
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
        "-width",
        "--width",
        type=int,
        default=None,
        help="Unified width controlling both ML top-k and beam width (default: 100).",
    )
    optional_group.add_argument(
        "-permutation",
        "--permutation",
        action="store_true",
        dest="permutation",
        help=(
            "Enable rule-level permutation null calibration on representative scan units "
            "(up to 32 units; if fewer are available, use all of them)."
        ),
    )
    optional_group.add_argument(
        "-no-clean",
        "--no-clean",
        action="store_true",
        dest="no_clean",
        help="Disable structured beam pruning and fall back to the legacy unfiltered fixed-width search.",
    )
    optional_group.add_argument("--prior-not", type=float, default=None, help=argparse.SUPPRESS)
    optional_group.add_argument("-layer", "--layer", type=int, default=None, help="Maximum beam-search rule depth (default: 4).")
    optional_group.add_argument(
        "-logic-gate",
        "--logic-gate",
        type=str,
        default="and_or",
        help="Allowed binary operators in beam search: and, or, or and_or.",
    )
    optional_group.add_argument(
        "-bimrange",
        "--bimrange",
        type=str,
        action="append",
        default=None,
        help=(
            "Restrict only the final scan stage to one or more genomic bp intervals. "
            "Repeat the flag or use comma-separated items, e.g. "
            "--bimrange 10:110800000-111200000,10:112000000-112200000. "
            "Background-noise calibration remains genome-wide."
        ),
    )
    optional_group.add_argument("-m", "--max-pick", type=int, default=None, dest="layer_compat", help=argparse.SUPPRESS)
    optional_group.add_argument(
        "--feature-source",
        type=str,
        choices=["bin", "mbin"],
        default=None,
        dest="feature_source_compat",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument("--seed", type=int, default=42, help="Random seed.")
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
        parser.error("-step/--step must be > 0")
    if not (0.0 <= float(args.maf) <= 0.5):
        parser.error("-maf/--maf must be in [0, 0.5]")
    if not (0.0 <= float(args.geno) <= 1.0):
        parser.error("-geno/--geno must be in [0, 1]")
    if (
        args.grm is not None
        and str(args.grm).strip().isdigit()
        and not os.path.exists(str(args.grm).strip())
    ):
        parser.error("-k/--grm now expects a GRM path.")

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

    args.width = int(args.width) if args.width is not None else 100
    if int(args.width) <= 0:
        parser.error("-width/--width must be > 0")
    args.beam_width = int(args.width)

    args.layer = (
        int(args.layer) if args.layer is not None
        else int(args.layer_compat) if args.layer_compat is not None
        else 4
    )
    if int(args.layer) <= 0:
        parser.error("-layer must be > 0")
    args.exhaustive_depth_runtime = 2 if int(args.layer) >= 2 else 1
    if args.prior_not is not None:
        try:
            if not np.isfinite(float(args.prior_not)):
                parser.error("--prior-not must be finite when provided")
        except Exception:
            parser.error("--prior-not must be finite when provided")

    args.topk = int(args.width) if args.engine is not None else 0
    args.top_rules_runtime = 1
    args.max_output_rules_runtime = 0
    args.max_output_ratio_runtime = 0.0

    args.feature_source = (
        str(args.feature_source_compat).lower()
        if args.feature_source_compat is not None
        else ("mbin" if bool(args.dev) else "bin")
    )
    if args.feature_source not in {"bin", "mbin"}:
        parser.error("--feature-source must be one of: bin, mbin")
    args.feature_source_requested = str(args.feature_source)
    if args.feature_source == "mbin":
        args.feature_source = "bin"

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
    args.rank_score = "gain_from_layer:2"
    args.rank_schedule_source = "fixed-combination-gain"
    args.gain_start_layer_runtime = 2

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
    apply_outer_thread_cap(int(args.thread))
    if args.feature_source_requested == "mbin":
        logger.warning(
            "MBIN encoding is deprecated for GARFIELD pure-line mode; using BIN instead "
            "(heterozygotes and NA are both treated as missing before binary decoding)."
        )

    ml_skipped = args.engine is None
    engine_runtime = "none" if ml_skipped else str(args.engine)
    logic_gate_runtime = args.logic_gate
    rank_score_runtime = str(args.rank_score)
    rank_schedule_source = str(args.rank_schedule_source)
    gain_start_layer_runtime = (
        None if args.gain_start_layer_runtime is None else int(args.gain_start_layer_runtime)
    )
    rank_schedule_runtime = _describe_rank_schedule(rank_score_runtime)

    checks: list[bool] = []
    checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    checks.append(ensure_file_exists(logger, args.pheno, "Phenotype file"))
    if args.grm:
        checks.append(ensure_file_exists(logger, args.grm, "GARFIELD GRM"))
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

    general_rows = [
        ("Genotype input", gfile),
        ("Input kind", input_kind),
        ("Execution route", "direct Rust BED pipeline"),
        ("Encoding", feature_source),
        ("Residualization GRM", args.grm if args.grm else "auto from genotype"),
        ("Covariates", None if not args.cov_inputs else ",".join(str(x) for x in args.cov_inputs)),
        ("Phenotype", args.pheno),
        ("Scan mode", args.scan_mode),
        ("Gene file", args.genefile),
        ("GFF3", args.gff3),
        ("MAF", float(args.maf)),
        ("Missing max (1+NA)", float(args.geno)),
        ("Extension", int(args.extension)),
        ("Step", int(args.step)),
        ("Bimrange", None if not args.bimrange else ",".join(str(x) for x in args.bimrange)),
        ("Split", "none (full data)"),
        ("Engine", "none (skip ML)" if ml_skipped else args.engine),
        ("Permutation", bool(args.permutation)),
        ("Structured pruning", not bool(args.no_clean)),
        ("NOT control", "null penalty only"),
        ("Width", int(args.width)),
        ("Rules/unit", int(args.top_rules_runtime)),
        ("Layer", int(args.layer)),
        ("Pair seed depth", int(args.exhaustive_depth_runtime)),
        ("Logic gate", args.logic_gate),
        ("Rule ranking", rank_schedule_runtime),
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
    # logger.info(
    #     "Rule-ranking resolution: logic_gate=%s, source=%s -> %s",
    #     logic_gate_runtime,
    #     rank_schedule_source,
    #     rank_schedule_runtime,
    # )

    pheno = _load_phenotype_with_status(
        args.pheno,
        args.ncol,
        _GarfieldPhenoLogger(logger),
        id_col=0,
        use_spinner=use_spinner,
    )

    sample_ids, _n_snps = _inspect_genotype_with_status(
        gfile,
        logger,
        use_spinner=use_spinner,
        snps_only=False,
        maf_threshold=float(args.maf),
        max_missing_rate=float(args.geno),
        het_threshold=0.0,
    )
    sample_ids = np.asarray(sample_ids, dtype=str)
    if len(sample_ids) == 0:
        raise ValueError("No sample IDs found in genotype input.")
    sample_index_map = {sid: i for i, sid in enumerate(sample_ids.tolist())}

    aligned_grm = None
    resolved_grm_id = None
    if args.grm:
        grm_src = os.path.basename(str(args.grm))
        with CliStatus(f"Loading GRM from {grm_src}...", enabled=use_spinner) as task:
            try:
                aligned_grm, resolved_grm_id = load_and_align_grm(
                    str(args.grm),
                    sample_ids.tolist(),
                    grm_id_path=None,
                    label="GARFIELD GRM",
                )
            except Exception:
                task.fail(f"Loading GRM from {grm_src} ...Failed")
                raise
            task.complete(f"Loading GRM from {grm_src} (n={aligned_grm.shape[0]})")

    cov_all, cov_ids = _load_covariates_for_models(
        cov_inputs=args.cov_inputs,
        genofile=gfile,
        sample_ids=sample_ids,
        chunk_size=65536,
        logger=logger,
        context="streaming",
        use_spinner=use_spinner,
        snps_only=False,
    )
    if cov_all is not None:
        cov_all = np.asarray(cov_all, dtype=np.float64, order="C")
    if cov_ids is not None:
        cov_ids = np.asarray(cov_ids, dtype=str)

    if pheno.shape[1] == 0:
        raise ValueError("No phenotype columns to analyze.")
    pheno, trait_names = _normalize_trait_names_from_header(pheno, args.pheno)
    geno_ids = sample_ids.astype(str)
    pheno_ids_all = pheno.index.astype(str).to_numpy()
    common = set(geno_ids) & set(pheno_ids_all)
    if aligned_grm is not None:
        common &= set(geno_ids)
    if cov_ids is not None:
        common &= set(cov_ids.astype(str))
    sample_pool = [sid for sid in geno_ids.tolist() if sid in common]
    if len(sample_pool) == 0:
        raise ValueError("No overlapping samples across genotype/phenotype/GRM/cov.")

    grm_n: int | str = "NA" if aligned_grm is None else int(aligned_grm.shape[0])
    cov_n: int | str = "NA" if cov_ids is None else int(len(cov_ids))
    split_line = "-"*60
    _emit_plain_info_line(
        logger,
        (
            f"geno={len(geno_ids)}, pheno={len(pheno_ids_all)}, "
            f"grm={grm_n}, q=NA, cov={cov_n} -> {len(sample_pool)}"
            f"\n{split_line}"
        ),
        use_spinner=use_spinner,
    )

    cov_index = (
        None
        if cov_ids is None
        else {sid: i for i, sid in enumerate(cov_ids.astype(str).tolist())}
    )

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
    garfield_manifest_traits: list[dict[str, object]] = []

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
        trait_grm = None
        if aligned_grm is not None:
            common_positions = np.asarray(
                [sample_index_map[sid] for sid in common_ids],
                dtype=np.intp,
            )
            trait_grm = np.asarray(
                aligned_grm[np.ix_(common_positions, common_positions)],
                dtype=np.float64,
            )
        trait_cov = None
        if cov_all is not None and cov_index is not None:
            cov_take = np.asarray([cov_index[sid] for sid in common_ids], dtype=np.intp)
            trait_cov = np.asarray(cov_all[cov_take, :], dtype=np.float64, order="C")
        scan_desc = {
            "window": "Scan Windows",
            "gene": "Scan Genes",
            "genepair": "Scan Gene Pairs",
            "geneset": "Scan Gene Sets",
        }.get(str(args.scan_mode), f"Rust GARFIELD search for '{trait_name}'")
        logic_unit_kind = _scan_mode_to_logic_unit_kind(args.scan_mode)
        rust_groups = group_intervals if args.scan_mode != "window" else None
        rust_group_names = group_labels if args.scan_mode != "window" else None
        trait_logic_prefix = f"{trait_outprefix}.garfield"
        # Rust handles full-data residualization before ML candidate search
        # and beam search are executed.
        result = _run_scan_with_progress(
            scan_desc,
            use_spinner=use_spinner,
            invoke=lambda progress_cb: garfield_logic_search_bed(
                gfile,
                np.asarray(y_common, dtype=np.float64),
                grm=trait_grm,
                x_cov=trait_cov,
                sample_ids=list(common_ids),
                unit_kind=logic_unit_kind,
                groups=rust_groups,
                group_names=rust_group_names,
                extension=int(args.extension),
                step=int(args.step),
                scan_bimranges=args.bimrange,
                bin_mode=feature_source,
                ml_method=str(engine_runtime).lower(),
                ml_importance="imp",
                ml_top_k=int(args.topk),
                ml_top_frac=0.0,
                permutation_repeats=5,
                permutation_scoring="auto",
                n_estimators=100,
                max_depth=int(args.layer) + 1,
                min_samples_leaf=1,
                min_samples_split=2,
                bootstrap=True,
                feature_subsample=0.0,
                fold=0,
                seed=trait_seed,
                max_pick=int(args.layer),
                exhaustive_depth=int(args.exhaustive_depth_runtime),
                beam_width=int(args.beam_width),
                logic_gate=str(args.logic_gate).lower(),
                rank_score=str(args.rank_score),
                maf_threshold=float(args.maf),
                max_missing_rate=float(args.geno),
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
                top_rules_per_unit=int(args.top_rules_runtime),
                max_output_rules=int(args.max_output_rules_runtime),
                max_output_ratio=float(args.max_output_ratio_runtime),
                rule_permutation=bool(args.permutation),
                prior_len=None,
                no_clean=bool(args.no_clean),
                progress_callback=progress_cb,
                progress_every=0,
            ),
        )

        pseudo_path = f"{trait_logic_prefix}.pseudo"
        posterior_tsv_path = f"{trait_logic_prefix}.posterior.tsv"
        posterior_json_path = result.get("posterior_json")
        run_config_path = f"{trait_outprefix}.garfield.run_config.json"
        skipped_messages = result.get("skipped_messages") or []
        if len(skipped_messages) > 0:
            logger.warning(
                f"GARFIELD skipped {len(skipped_messages)} unit(s) for trait '{trait_name}'."
            )
            for msg in skipped_messages:
                logger.warning(str(msg))
        n_rules = int(result.get("n_rules", 0))
        if n_rules <= 0:
            _remove_file_if_exists(pseudo_path)
            _remove_file_if_exists(posterior_tsv_path)
            _remove_file_if_exists(run_config_path)
            _remove_file_if_exists(posterior_json_path)
            logger.warning(f"No GARFIELD rules survived Rust search for trait '{trait_name}', skipped.")
            continue

        _remove_file_if_exists(pseudo_path)
        _remove_file_if_exists(posterior_tsv_path)
        split_applied = bool(result.get("split_applied", False))
        prior_payload, posterior_payload = _split_structure_prior_payload(
            _load_json_if_exists(posterior_json_path)
        )
        _remove_file_if_exists(run_config_path)
        _remove_file_if_exists(posterior_json_path)
        trait_manifest = {
            "trait": trait_name,
            "trait_output_prefix": trait_outprefix,
            "garfield_prefix": trait_logic_prefix,
            "route": "rust-bed",
            "split_applied": split_applied,
            "scan_mode": args.scan_mode,
            "unit_kind": logic_unit_kind,
            "response": response_mode,
            "engine": args.engine,
            "engine_runtime": engine_runtime,
            "ml_skipped": ml_skipped,
            "permutation": bool(args.permutation),
            "rule_permutation_active": bool(result.get("rule_permutation_active", False)),
            "null_chunk_bp": int(result.get("null_chunk_bp", 0)),
            "null_chunk_min_snps": int(result.get("null_chunk_min_snps", 0)),
            "null_chunk_target": int(result.get("null_chunk_target", 0)),
            "null_chunk_valid_total": int(result.get("null_chunk_valid_total", 0)),
            "null_chunk_selected": int(result.get("null_chunk_selected", 0)),
            "representative_units_target": int(result.get("representative_units_target", 0)),
            "representative_units_used": int(result.get("representative_units_used", 0)),
            "permutation_null_repeats": int(result.get("permutation_null_repeats", 0)),
            "permutation_bootstrap_repeats": int(
                result.get("permutation_bootstrap_repeats", 0)
            ),
            "no_clean_requested": bool(args.no_clean),
            "structured_pruning": not bool(args.no_clean),
            "width": int(args.width),
            "top_rules_per_unit": int(args.top_rules_runtime),
            "layer": int(args.layer),
            "pair_seed_depth": int(args.exhaustive_depth_runtime),
            "beam_width": int(args.beam_width),
            "not_control": "null_penalty_only",
            "prior_not_ignored": (
                None if args.prior_not is None else float(args.prior_not)
            ),
            "rank_schedule_source": rank_schedule_source,
            "gain_start_layer_runtime": gain_start_layer_runtime,
            "rank_schedule_runtime": rank_schedule_runtime,
            "rank_score": rank_score_runtime,
            "rank_score_runtime": rank_score_runtime,
            "ml_top_k": (None if ml_skipped else int(args.topk)),
            "extension": int(args.extension),
            "step": int(args.step),
            "bimrange": (list(args.bimrange) if args.bimrange else None),
            "logic_gate_requested": args.logic_gate,
            "logic_gate_runtime": logic_gate_runtime,
            "ranking_dataset": "full",
            "feature_source": feature_source,
            "grm_path": args.grm,
            "grm_id_path": resolved_grm_id,
            "maf": float(args.maf),
            "geno": float(args.geno),
            "pure_line_missing_rule": "heterozygote_or_na",
            "simbench_path": args.simbench,
            "simbench_rows": int(result.get("n_simbench", 0)),
            "seed": trait_seed,
            "n_samples": len(common_ids),
            "n_train": int(result.get("n_train", 0)),
            "n_test": int(result.get("n_test", 0)),
            "n_rules": int(result.get("n_rules", 0)),
            "units_total": int(result.get("units_total", 0)),
            "units_scanned": int(result.get("units_scanned", 0)),
            "train_pve": float(result.get("train_pve", float("nan"))),
            "test_pve": float(result.get("test_pve", float("nan"))),
            "train_sigma_g2": float(result.get("train_sigma_g2", float("nan"))),
            "train_sigma_e2": float(result.get("train_sigma_e2", float("nan"))),
            "test_sigma_g2": float(result.get("test_sigma_g2", float("nan"))),
            "test_sigma_e2": float(result.get("test_sigma_e2", float("nan"))),
            "outputs": {
                "rules_tsv": result.get("rules_tsv"),
            },
            "prior": prior_payload,
            "posterior": posterior_payload,
        }
        garfield_manifest_traits.append(trait_manifest)
        logger.info(f"GARFIELD null-model PVE for '{trait_name}': full={float(result.get('train_pve', float('nan'))):.4g}")
        if args.simbench:
            logger.info(
                f"GARFIELD simbench rows appended for '{trait_name}': "
                f"{int(result.get('n_simbench', 0))}"
            )
        rank_scores = [
            float(x)
            for x in (
                result.get("scores")
                or result.get("test_scores")
                or result.get("train_scores")
                or []
            )
        ]
        best_score = rank_scores[0] if len(rank_scores) > 0 else float("nan")
        summary_rows.append(
            {
                "trait": trait_name,
                "n_samples": len(common_ids),
                "best_score": best_score,
                "route": "rust-bed",
            }
        )
        log_success(
            logger,
            f"Saved GARFIELD output for trait: "
            f"\n\t{format_path_for_display(trait_logic_prefix)}",
        )
        saved += 1

    if saved == 0:
        raise ValueError("No GARFIELD outputs were generated for the selected phenotype columns.")

    aggregate_json_path = f"{outprefix}.garfield.json"
    with open(aggregate_json_path, "w", encoding="utf-8") as fw:
        json.dump(
            {
                "format": "janusx.garfield.summary.v1",
                "log_file": log_path,
                "output_prefix": outprefix,
                "genotype_input": gfile,
                "input_kind": input_kind,
                "execution_route": "direct Rust BED pipeline",
                "encoding": feature_source,
                "phenotype_file": args.pheno,
                "grm_path": args.grm,
                "grm_id_path": resolved_grm_id,
                "scan_mode": args.scan_mode,
                "logic_gate_runtime": logic_gate_runtime,
                "rank_score_runtime": rank_score_runtime,
                "rank_schedule_runtime": rank_schedule_runtime,
                "rank_schedule_source": rank_schedule_source,
                "permutation": bool(args.permutation),
                "null_chunk_bp": int(args.extension) * 2,
                "null_chunk_target": 150 if bool(args.permutation) else 0,
                "null_chunk_min_snps": 50 if bool(args.permutation) else 0,
                "bimrange": (list(args.bimrange) if args.bimrange else None),
                "top_rules_per_unit": int(args.top_rules_runtime),
                "pair_seed_depth": int(args.exhaustive_depth_runtime),
                "not_control": "null_penalty_only",
                "prior_not_ignored": (
                    None if args.prior_not is None else float(args.prior_not)
                ),
                "thread": int(args.thread),
                "seed": int(args.seed),
                "summary_rows": summary_rows,
                "traits": garfield_manifest_traits,
            },
            fw,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"GARFIELD JSON: {format_path_for_display(aggregate_json_path)}")
    _emit_garfield_summary_to_log(logger, summary_rows)

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
