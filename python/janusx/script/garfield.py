import argparse
import hashlib
import os
import socket
import time
from pathlib import Path
from typing import Optional

import numpy as np

from janusx.assoc.workflow import load_phenotype
from janusx.gfreader import SiteInfo, inspect_genotype_file
from janusx.gfreader.gfreader import save_genotype_streaming
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
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
    format_path_for_display,
)
from janusx.script._common.status import CliStatus, log_success, stdout_is_tty
from janusx.script._common.threads import apply_blas_thread_env, detect_effective_threads


try:
    from janusx.janusx import (
        garfield_eval_rule_bin,
        garfield_prepare_input_bin,
        garfield_scan_groups_bin,
        garfield_scan_windows_bin,
        garfield_subset_bin_samples,
        load_site_info,
    )
except Exception as _exc:  # pragma: no cover
    garfield_eval_rule_bin = None  # type: ignore[assignment]
    garfield_prepare_input_bin = None  # type: ignore[assignment]
    garfield_scan_groups_bin = None  # type: ignore[assignment]
    garfield_scan_windows_bin = None  # type: ignore[assignment]
    garfield_subset_bin_samples = None  # type: ignore[assignment]
    load_site_info = None  # type: ignore[assignment]
    _RUST_IMPORT_ERROR = _exc
else:
    _RUST_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Legacy GARFIELD implementation note
# -----------------------------------------------------------------------------
# 旧版 GARFIELD（Python 内做窗口/基因扫描、候选筛选与逻辑回归）已经下线。
# 该版本改为：Python 只做调度，Rust 完成解码/扫描/评分/规则验证。
# 主要闭环：输入 -> 编码(BIN) -> beam(train) -> 验证(val) -> 导出(pseudo)


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
    missing = (
        garfield_eval_rule_bin is None
        or garfield_prepare_input_bin is None
        or garfield_scan_groups_bin is None
        or garfield_scan_windows_bin is None
        or garfield_subset_bin_samples is None
        or load_site_info is None
    )
    if missing:
        raise ImportError(
            "janusx Rust extension does not provide new GARFIELD APIs. "
            "Please rebuild/reinstall JanusX extension."
        ) from _RUST_IMPORT_ERROR


def _resolve_bin_path(path_or_prefix: str) -> Optional[str]:
    p = str(Path(path_or_prefix).expanduser())
    if p.lower().endswith(".bin") and Path(p).is_file():
        return p
    cand = f"{p}.bin"
    if Path(cand).is_file():
        return cand
    return None


def _infer_input_kind(args) -> str:
    if args.bfile:
        return "bfile"
    if args.vcf:
        return "vcf"
    if args.hmp:
        return "hmp"
    return "file"


def _safe_cache_stem(text: str) -> str:
    stem = str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem.strip("._") or "geno"


def _input_signature_paths(input_kind: str, gfile: str) -> list[Path]:
    if input_kind == "bfile":
        p = str(gfile).rstrip("/")
        prefix = p[:-4] if p.lower().endswith((".bed", ".bim", ".fam")) else p
        return [Path(f"{prefix}.bed"), Path(f"{prefix}.bim"), Path(f"{prefix}.fam")]

    p = Path(str(gfile))
    if p.exists():
        return [p]

    # FILE prefix style: try known sidecars
    known = [
        Path(f"{gfile}.txt"),
        Path(f"{gfile}.tsv"),
        Path(f"{gfile}.csv"),
        Path(f"{gfile}.npy"),
        Path(f"{gfile}.bin"),
        Path(f"{gfile}.id"),
        Path(f"{gfile}.fam"),
        Path(f"{gfile}.site"),
        Path(f"{gfile}.site.tsv"),
        Path(f"{gfile}.site.txt"),
        Path(f"{gfile}.site.csv"),
        Path(f"{gfile}.bim"),
    ]
    return [x for x in known if x.exists()]


def _build_conversion_cache_key(
    input_kind: str,
    gfile: str,
    *,
    threads: int,
    maf: float,
    geno: float,
    impute: bool,
    het: float,
) -> str:
    h = hashlib.sha1()
    h.update(f"kind={input_kind}\n".encode("utf-8"))
    h.update(f"src={Path(str(gfile)).expanduser().resolve() if Path(str(gfile)).exists() else str(gfile)}\n".encode("utf-8"))
    h.update(f"threads={int(threads)}\nmaf={float(maf):.8g}\ngeno={float(geno):.8g}\nimpute={int(bool(impute))}\nhet={float(het):.8g}\n".encode("utf-8"))
    for p in _input_signature_paths(input_kind, gfile):
        try:
            st = p.stat()
        except OSError:
            continue
        h.update(f"{str(p)}|{int(st.st_size)}|{int(st.st_mtime_ns)}\n".encode("utf-8"))
    return h.hexdigest()[:12]


def _prepare_input_bins(
    *,
    input_kind: str,
    gfile: str,
    prefix: str,
    out_dir: str,
    threads: int,
    logger,
    use_spinner: bool,
    maf: float = 0.0,
    geno: float = 1.0,
    impute: bool = False,
    het: float = 0.05,
) -> tuple[str, Optional[str]]:
    # Explicit .bin input: reuse as primary BIN, skip MBIN synthesis from binary input.
    gfile_norm = str(Path(str(gfile)).expanduser())
    existing_bin = _resolve_bin_path(gfile_norm)
    explicit_bin_input = gfile_norm.lower().endswith(".bin") and Path(gfile_norm).is_file()
    if explicit_bin_input and existing_bin is not None and input_kind == "file":
        logger.info(f"Input already BIN: {existing_bin}")
        return existing_bin, None

    cache_dir = Path(out_dir) / ".garfield_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _build_conversion_cache_key(
        input_kind,
        gfile,
        threads=int(threads),
        maf=float(maf),
        geno=float(geno),
        impute=bool(impute),
        het=float(het),
    )
    stem = _safe_cache_stem(prefix)
    cache_base = cache_dir / f"{stem}.{cache_key}"
    bin_path = str(cache_base) + ".bin02.bin"
    mbin_path = str(cache_base) + ".mbin.bin"

    def _sidecars_ready(path: str) -> bool:
        p = Path(path)
        pre = p.with_suffix("") if p.suffix.lower() == ".bin" else p
        return Path(str(pre) + ".bin.id").is_file() and Path(str(pre) + ".bin.site").is_file()

    need_bin = not (Path(bin_path).is_file() and _sidecars_ready(bin_path))
    need_mbin = not (Path(mbin_path).is_file() and _sidecars_ready(mbin_path))

    if need_bin:
        with CliStatus("Preparing BIN cache (bin02)...", enabled=use_spinner) as task:
            try:
                garfield_prepare_input_bin(
                    gfile,
                    bin_path,
                    input_kind=input_kind,
                    mode="bin",
                    threads=int(threads),
                    maf=float(maf),
                    geno=float(geno),
                    impute=bool(impute),
                    het=float(het),
                )
            except Exception:
                task.fail("Preparing BIN cache (bin02) ...Failed")
                raise
            task.complete("Preparing BIN cache (bin02) ...Finished")

    if need_mbin:
        with CliStatus("Preparing MBIN cache (dom/rec/het)...", enabled=use_spinner) as task:
            try:
                garfield_prepare_input_bin(
                    gfile,
                    mbin_path,
                    input_kind=input_kind,
                    mode="mbin",
                    threads=int(threads),
                    maf=float(maf),
                    geno=float(geno),
                    impute=bool(impute),
                    het=float(het),
                )
            except Exception:
                task.fail("Preparing MBIN cache (dom/rec/het) ...Failed")
                raise
            task.complete("Preparing MBIN cache (dom/rec/het) ...Finished")

    return bin_path, mbin_path


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
            genes = [g for g in line.strip().split() if g]
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
    elif scan_mode == "genepair":
        for genes in genesets:
            if len(genes) < 2:
                continue
            g1, g2 = genes[0], genes[1]
            iv1, iv2 = _iv(g1), _iv(g2)
            if iv1 is None or iv2 is None:
                continue
            labels.append(f"{g1}|{g2}")
            groups.append([iv1, iv2])
    else:
        raise ValueError(f"unsupported scan_mode: {scan_mode}")

    return labels, groups


def _split_train_val(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 4:
        raise ValueError("Need at least 4 samples for train/val split.")
    vf = float(val_frac)
    if not (0.0 < vf < 0.8):
        raise ValueError("--val-frac must be in (0, 0.8).")
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_val = int(max(1, round(n * vf)))
    n_val = min(n - 1, n_val)
    val_idx = np.sort(idx[:n_val])
    train_idx = np.sort(idx[n_val:])
    return train_idx, val_idx


def _write_sample_split(out_prefix: str, sample_ids: list[str], train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    split_dir = Path(f"{out_prefix}.split")
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train.samples.txt"
    val_path = split_dir / "val.samples.txt"
    with open(train_path, "w", encoding="utf-8") as fw:
        for i in train_idx.tolist():
            fw.write(f"{sample_ids[int(i)]}\n")
    with open(val_path, "w", encoding="utf-8") as fw:
        for i in val_idx.tolist():
            fw.write(f"{sample_ids[int(i)]}\n")


def _write_xcombine_results(
    outprefix: str,
    sample_ids: list[str],
    records: list[dict],
    *,
    chrom: str = "pseudo",
) -> None:
    map_path = f"{outprefix}.pseudo"
    n_samples = len(sample_ids)

    def _iter_chunks():
        pos = 1
        with open(map_path, "w", encoding="utf-8") as f:
            f.write("chrom\tpos\texpression\ttrain_score\tval_score\n")
            for rec in records:
                xcombine = np.asarray(rec["xcombine"], dtype=np.int8).reshape(-1)
                if xcombine.size != n_samples:
                    raise ValueError("xcombine length does not match sample_ids")
                expr = str(rec.get("expression", ""))
                tr = float(rec.get("train_score", float("nan")))
                va = float(rec.get("val_score", float("nan")))
                f.write(f"{chrom}\t{pos}\t{expr}\t{tr:.12g}\t{va:.12g}\n")
                yield xcombine.reshape(1, -1), [SiteInfo(str(chrom), int(pos), "A", "T")]
                pos += 1

    save_genotype_streaming(outprefix, sample_ids, _iter_chunks())


def _literal_lookup(bin_path: str, n_rows: int) -> dict[int, str]:
    sites = list(load_site_info(str(bin_path), None))
    out: dict[int, str] = {}
    for i, s in enumerate(sites[:n_rows]):
        chrom = str(getattr(s, "chrom", "N"))
        pos = int(getattr(s, "pos", i + 1))
        ref = str(getattr(s, "ref_allele", "N"))
        alt = str(getattr(s, "alt_allele", "N"))
        out[i] = f"{chrom}_{pos}[{ref}>{alt}]"
    return out


def _expression_from_indices(indices: list[int], lut: dict[int, str]) -> str:
    if not indices:
        return "1"
    return " & ".join(lut.get(int(i), f"IDX{int(i)}") for i in indices)


def _write_rules_tsv(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fw:
        fw.write(
            "rank\tscan_mode\tgroup\tn_candidates\tselected_indices\texpression\ttrain_score\tval_score\ttrain_support\tval_support\n"
        )
        for i, r in enumerate(rows, start=1):
            fw.write(
                f"{i}\t{r['scan_mode']}\t{r['group']}\t{int(r['n_candidates'])}\t"
                f"{','.join(map(str, r['selected']))}\t{r['expression']}\t"
                f"{float(r['train_score']):.12g}\t{float(r['val_score']):.12g}\t"
                f"{int(r['train_support'])}\t{int(r['val_support'])}\n"
            )


def main() -> None:
    _require_rust_backend()

    t_start = time.time()
    use_spinner = stdout_is_tty()

    parser = CliArgumentParser(prog="jx garfield", formatter_class=cli_help_formatter())

    required_group = parser.add_argument_group("Required Arguments")
    geno_group = required_group.add_mutually_exclusive_group(required=False)
    geno_group.add_argument("-bfile", "--bfile", type=str, help="PLINK prefix (.bed/.bim/.fam).")
    geno_group.add_argument("-vcf", "--vcf", type=str, help="VCF/VCF.GZ input.")
    geno_group.add_argument("-hmp", "--hmp", type=str, help="HMP/HMP.GZ input.")
    geno_group.add_argument("-file", "--file", type=str, help="Numeric matrix prefix or path.")
    required_group.add_argument("-p", "--pheno", type=str, required=False, help="Phenotype file.")

    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument("-g", "--genefile", type=str, default=None, help="Gene list/set file.")
    optional_group.add_argument("-gff", "--gff3", type=str, default=None, help="GFF3 annotation.")
    optional_group.add_argument(
        "--scan-mode",
        type=str,
        choices=["window", "gene", "genepair"],
        default="window",
        help="Scan strategy (default: window).",
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
    optional_group.add_argument("-step", "--step", type=int, default=None, help="Window step size (default: extension/2).")
    optional_group.add_argument("-ext", "--extension", type=int, default=50_000, help="Window extension in bp.")
    optional_group.add_argument("-nsnp", "--nsnp", type=int, default=5, help="Beam width / top SNP candidates.")
    optional_group.add_argument("-m", "--max-pick", type=int, default=3, help="Maximum literals in beam rule.")
    optional_group.add_argument(
        "--feature-source",
        type=str,
        choices=["bin", "mbin"],
        default="mbin",
        help="Feature cache for beam search (default: mbin).",
    )
    optional_group.add_argument("--top-k-validate", type=int, default=10, help="Top train candidates for validation.")
    optional_group.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction (default: 0.2).")
    optional_group.add_argument("-het", "--het", type=float, default=0.05, help="Max heterozygosity rate for BIN mode filtering (default: 0.05).")
    optional_group.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    optional_group.add_argument("-t", "--thread", dest="thread", type=int, default=detect_effective_threads(), help="CPU threads.")
    optional_group.add_argument("--threads", dest="thread", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    optional_group.add_argument("-o", "--out", type=str, default=".", help="Output directory.")
    optional_group.add_argument("-prefix", "--prefix", type=str, default=None, help="Output prefix.")

    args, extras = parser.parse_known_args()

    has_genotype = bool(args.vcf or args.hmp or args.file or args.bfile)
    has_pheno = bool(args.pheno)
    if (not has_pheno) and (not has_genotype):
        parser.error(
            "the following arguments are required: -p/--pheno & (-vcf VCF | -hmp HMP | -file FILE | -bfile BFILE)"
        )
    if not has_pheno:
        parser.error("the following arguments are required: -p/--pheno")
    if not has_genotype:
        parser.error("the following arguments are required: (-vcf VCF | -hmp HMP | -file FILE | -bfile BFILE)")
    if len(extras) > 0:
        parser.error("unrecognized arguments: " + " ".join(extras))

    if int(args.extension) <= 0:
        parser.error("-ext/--extension must be > 0")
    if args.step is None:
        args.step = max(1, int(args.extension) // 2)
    elif int(args.step) <= 0:
        parser.error("-step/--step must be > 0")
    if int(args.nsnp) <= 0:
        parser.error("-nsnp/--nsnp must be > 0")
    if int(args.max_pick) <= 0:
        parser.error("-m/--max-pick must be > 0")
    if int(args.top_k_validate) <= 0:
        parser.error("--top-k-validate must be > 0")
    if not (0.0 <= float(args.het) <= 1.0):
        parser.error("-het/--het must be in [0, 1]")

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
    input_kind = _infer_input_kind(args)

    args.out = os.path.normpath(args.out if args.out is not None else ".")
    outprefix = os.path.join(args.out, prefix)
    os.makedirs(args.out, mode=0o755, exist_ok=True)
    configure_genotype_cache_from_out(args.out)

    log_path = os.path.join(args.out, f"{prefix}.garfield.log")
    logger = setup_logging(log_path)
    apply_blas_thread_env(int(args.thread))

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype input"))
    checks.append(ensure_file_exists(logger, args.pheno, "Phenotype file"))
    if args.genefile:
        checks.append(ensure_file_exists(logger, args.genefile, "Gene file"))
    if args.gff3:
        checks.append(ensure_file_exists(logger, args.gff3, "GFF3 file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    bin_path, mbin_path = _prepare_input_bins(
        input_kind=input_kind,
        gfile=gfile,
        prefix=prefix,
        out_dir=args.out,
        threads=int(args.thread),
        logger=logger,
        use_spinner=use_spinner,
        het=float(args.het),
    )
    scan_bin_path = bin_path
    if str(args.feature_source).lower() == "mbin":
        if mbin_path is not None:
            scan_bin_path = mbin_path
        else:
            logger.warning(
                "feature-source=mbin requested, but current input is already BIN; fallback to bin."
            )
    elif str(args.feature_source).lower() == "bin":
        scan_bin_path = bin_path
    else:
        raise ValueError(f"Unsupported feature-source: {args.feature_source}")

    emit_cli_configuration(
        logger,
        app_title="JanusX - GARFIELD (Rust-first)",
        config_title="GARFIELD CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "General",
                [
                    ("Genotype input", gfile),
                    ("Input kind", input_kind),
                    ("BIN path", bin_path),
                    ("MBIN path", mbin_path if mbin_path else "(skip; input already BIN)"),
                    ("Feature source", str(args.feature_source).lower()),
                    ("Scan BIN path", scan_bin_path),
                    ("Phenotype", args.pheno),
                    ("Scan mode", args.scan_mode),
                    ("Gene file", args.genefile),
                    ("GFF3", args.gff3),
                    ("Extension", int(args.extension)),
                    ("Step", int(args.step)),
                    ("Beam width", int(args.nsnp)),
                    ("Max pick", int(args.max_pick)),
                    ("Top-K validate", int(args.top_k_validate)),
                    ("Val frac", float(args.val_frac)),
                    ("BIN het max", float(args.het)),
                    ("Seed", int(args.seed)),
                ],
            )
        ],
        footer_rows=[("Threads", f"{int(args.thread)} ({int(detected_threads)} available)"), ("Output prefix", outprefix)],
        line_max_chars=60,
    )

    sample_ids, n_rows = inspect_genotype_file(scan_bin_path)
    sample_ids = np.asarray(sample_ids, dtype=str)
    if len(sample_ids) == 0:
        raise ValueError("No sample IDs found in BIN input.")

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

    full_id_to_idx = {sid: i for i, sid in enumerate(sample_ids.tolist())}
    pheno_all_ids = set(pheno.index.astype(str).to_numpy())
    sample_pool = [sid for sid in sample_ids.tolist() if sid in pheno_all_ids]
    if len(sample_pool) == 0:
        raise ValueError("No overlapping samples between genotype and phenotype.")

    group_labels: list[str] = []
    group_intervals: list[list[tuple[str, int, int]]] = []
    if args.scan_mode in {"gene", "genepair"}:
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

    lut = _literal_lookup(scan_bin_path, int(n_rows))

    used_trait_labels: dict[str, int] = {}
    saved = 0

    tmp_dir = Path(args.out) / ".garfield_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[tuple[str, int, int, int, float]] = []

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

        train_idx, val_idx = _split_train_val(len(common_ids), float(args.val_frac), int(args.seed) + trait_idx)

        common_global_idx = [full_id_to_idx[sid] for sid in common_ids]
        common_bin = str(tmp_dir / f"{prefix}.{_safe_trait_label(trait_name)}.common.bin")
        train_bin = str(tmp_dir / f"{prefix}.{_safe_trait_label(trait_name)}.train.bin")
        val_bin = str(tmp_dir / f"{prefix}.{_safe_trait_label(trait_name)}.val.bin")

        with CliStatus(f"Preparing BIN subsets for '{trait_name}'...", enabled=use_spinner) as task:
            try:
                garfield_subset_bin_samples(scan_bin_path, common_bin, common_global_idx)
                garfield_subset_bin_samples(common_bin, train_bin, train_idx.astype(np.int64).tolist())
                garfield_subset_bin_samples(common_bin, val_bin, val_idx.astype(np.int64).tolist())
            except Exception:
                task.fail(f"Preparing BIN subsets for '{trait_name}' ...Failed")
                raise
            task.complete(f"Preparing BIN subsets for '{trait_name}' ...Finished")

        y_train = np.asarray(y_common[train_idx], dtype=np.float64)
        y_val = np.asarray(y_common[val_idx], dtype=np.float64)

        candidates: list[dict] = []

        with CliStatus(f"Train beam search for '{trait_name}'...", enabled=use_spinner, timeout=0.1) as task:
            try:
                if args.scan_mode == "window":
                    w_results = garfield_scan_windows_bin(
                        train_bin,
                        y_train,
                        response=response_mode,
                        max_pick=int(args.max_pick),
                        beam_width=int(args.nsnp),
                        extension=int(args.extension),
                        step=int(args.step),
                        enforce_feature_exclusion=True,
                    )
                    for wi, chrom, bp_start, bp_end, n_cand, sc, sel_idx in w_results:
                        candidates.append(
                            {
                                "scan_mode": "window",
                                "group": f"{chrom}:{int(bp_start)}-{int(bp_end)}",
                                "group_id": int(wi),
                                "n_candidates": int(n_cand),
                                "train_score": float(sc),
                                "selected": [int(x) for x in sel_idx],
                            }
                        )
                else:
                    g_results = garfield_scan_groups_bin(
                        train_bin,
                        y_train,
                        group_intervals,
                        response=response_mode,
                        max_pick=int(args.max_pick),
                        beam_width=int(args.nsnp),
                        enforce_feature_exclusion=True,
                    )
                    for gi, n_cand, sc, sel_idx in g_results:
                        label = group_labels[int(gi)] if int(gi) < len(group_labels) else f"group_{int(gi)}"
                        candidates.append(
                            {
                                "scan_mode": args.scan_mode,
                                "group": label,
                                "group_id": int(gi),
                                "n_candidates": int(n_cand),
                                "train_score": float(sc),
                                "selected": [int(x) for x in sel_idx],
                            }
                        )
            except Exception:
                task.fail(f"Train beam search for '{trait_name}' ...Failed")
                raise
            task.complete(f"Train beam search for '{trait_name}' ...Finished")

        candidates = [c for c in candidates if len(c["selected"]) > 0 and np.isfinite(c["train_score"]) ]
        if len(candidates) == 0:
            logger.warning(f"No valid train candidates for trait '{trait_name}', skipped.")
            continue

        candidates.sort(key=lambda x: (-float(x["train_score"]), len(x["selected"]), x["group"]))
        top_cands = candidates[: int(args.top_k_validate)]

        final_rows: list[dict] = []
        for c in top_cands:
            selected = [int(x) for x in c["selected"]]
            tr_score, tr_support, _ = garfield_eval_rule_bin(
                train_bin,
                y_train,
                selected,
                response=response_mode,
            )
            va_score, va_support, _ = garfield_eval_rule_bin(
                val_bin,
                y_val,
                selected,
                response=response_mode,
            )
            _full_score, _full_support, xcombine = garfield_eval_rule_bin(
                common_bin,
                np.asarray(y_common, dtype=np.float64),
                selected,
                response=response_mode,
            )
            if isinstance(xcombine, (bytes, bytearray, memoryview)):
                xcombine_arr = np.frombuffer(xcombine, dtype=np.uint8).astype(np.int8, copy=False)
            else:
                xcombine_arr = np.asarray(xcombine, dtype=np.int8)

            expr = _expression_from_indices(selected, lut)
            final_rows.append(
                {
                    "scan_mode": c["scan_mode"],
                    "group": c["group"],
                    "n_candidates": int(c["n_candidates"]),
                    "selected": selected,
                    "expression": expr,
                    "train_score": float(tr_score),
                    "val_score": float(va_score),
                    "train_support": int(tr_support),
                    "val_support": int(va_support),
                    "xcombine": xcombine_arr,
                }
            )

        final_rows.sort(key=lambda x: (-float(x["val_score"]), -float(x["train_score"]), len(x["selected"])))

        base_trait = _safe_trait_label(trait_name)
        count = used_trait_labels.get(base_trait, 0) + 1
        used_trait_labels[base_trait] = count
        suffix = base_trait if count == 1 else f"{base_trait}.{count}"
        trait_outprefix = f"{outprefix}.{suffix}"

        _write_sample_split(trait_outprefix, list(common_ids), train_idx, val_idx)

        with open(f"{trait_outprefix}.garfield.run_config.json", "w", encoding="utf-8") as fw:
            fw.write(
                "{\n"
                f"  \"scan_mode\": \"{args.scan_mode}\",\n"
                f"  \"response\": \"{response_mode}\",\n"
                f"  \"max_pick\": {int(args.max_pick)},\n"
                f"  \"beam_width\": {int(args.nsnp)},\n"
                f"  \"extension\": {int(args.extension)},\n"
                f"  \"step\": {int(args.step)},\n"
                f"  \"top_k_validate\": {int(args.top_k_validate)},\n"
                f"  \"val_frac\": {float(args.val_frac):.6g},\n"
                f"  \"seed\": {int(args.seed) + trait_idx}\n"
                "}\n"
            )

        _write_rules_tsv(f"{trait_outprefix}.garfield.rules.tsv", final_rows)
        _write_xcombine_results(f"{trait_outprefix}.garfield", list(common_ids), final_rows)

        best_val = float(final_rows[0]["val_score"]) if len(final_rows) > 0 else float("nan")
        summary_rows.append((trait_name, len(common_ids), len(train_idx), len(val_idx), best_val))

        log_success(
            logger,
            f"Saved GARFIELD output for trait '{trait_name}': "
            f"{format_path_for_display(f'{trait_outprefix}.garfield')}",
        )
        saved += 1

    if saved == 0:
        raise ValueError("No GARFIELD outputs were generated for the selected phenotype columns.")

    summary_path = f"{outprefix}.garfield.summary.tsv"
    with open(summary_path, "w", encoding="utf-8") as fw:
        fw.write("trait\tn_samples\tn_train\tn_val\tbest_val_score\n")
        for t, n, nt, nv, b in summary_rows:
            fw.write(f"{t}\t{int(n)}\t{int(nt)}\t{int(nv)}\t{float(b):.12g}\n")

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
