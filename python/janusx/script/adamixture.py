"""JanusX wrapper for the upstream ADAMIXTURE population analysis CLI.

``jx adamixture`` owns the JanusX command line contract and output naming.
Fitting, genotype decoding, optimisation and CV are delegated to the installed
``adamixture`` executable so there is one implementation of the model.  Marker
coordinates are referenced from the BED ``.bim`` or PGEN ``.pvar`` used by the
run; no per-K ``.P.site`` copies are emitted.
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import os
import re
import shlex
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from janusx.adamixture.external_cli import (
    ExternalAdmixtureError,
    resolve_executable,
    run as run_external_adamixture,
)
from janusx.gfreader import (
    inspect_genotype_file,
    load_genotype_chunks,
    prepare_cli_input_cache,
    save_genotype_streaming,
)
from ._common.cli_args import (
    add_common_genotype_source_args,
    add_common_out_arg,
    add_common_prefix_arg,
    add_common_thread_arg,
)
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.config_render import emit_cli_configuration
from ._common.genoio import (
    determine_genotype_source_from_args,
    genotype_load_status_done,
    genotype_load_status_fail,
    genotype_load_status_open,
)
from ._common.log import setup_logging
from ._common.outprefix import apply_output_prefix_compat
from ._common.pathcheck import (
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
    format_path_for_display,
)
from ._common.progress import (
    CliStatus,
    ProgressBarAdapter,
    _emit_to_file_handlers,
    format_elapsed,
    log_success,
    print_failure,
    print_success,
    stdout_is_tty,
)
from ._common.threads import detect_effective_threads


_DEFAULT_CV_FOLDS = 5


def _log_detail(logger: logging.Logger, message: str, *args: object) -> None:
    """Write diagnostic details to the run log without terminal noise."""

    try:
        rendered = str(message) % args if args else str(message)
    except Exception:
        rendered = " ".join([str(message), *(str(arg) for arg in args)])
    if _emit_to_file_handlers(logger, logging.INFO, rendered):
        return
    # Keep a useful fallback when logging was intentionally disabled or a
    # filesystem log could not be opened.
    logger.info(str(message), *args)


_ADAMIXTURE_KEY_LINE_PATTERNS = (
    r"\b(?:iteration|iter(?:ation)?|epoch|round)\b",
    r"priming iteration",
    r"final log[- ]likelihood",
    r"\b(?:converged|convergence)\b",
    r"cross[- ]validation",
    r"\bcv(?: index| score| error)?\b",
    r"\bcompleted in\b",
    r"total elapsed time",
    r"q and p matrices saved",
    r"\b(?:warning|error|failed)\b",
)


def _adamixture_key_line(line: str) -> str | None:
    """Keep useful upstream progress without copying its verbose banner."""

    text = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", str(line)).strip()
    if not text:
        return None
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in _ADAMIXTURE_KEY_LINE_PATTERNS):
        return text
    return None


def _adamixture_invocation_command(argv: Optional[list[str]] = None) -> str:
    """Render the public launcher command consistently in the run log."""

    tokens = [str(value) for value in (sys.argv[1:] if argv is None else argv)]
    return shlex.join(["jx", "adamixture", *tokens])


def _parse_k_spec(raw_spec: str) -> list[int]:
    """Parse a single K, an inclusive range, or a comma-separated list."""

    raw = str(raw_spec or "").strip()
    if not raw:
        raise ValueError("Empty -k/--k specification.")
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.replace(":", "..").split("..")
        try:
            numbers = [int(x) for x in parts if x != ""]
        except ValueError as exc:
            raise ValueError(f"Invalid K specification: {raw_spec!r}.") from exc
        if len(numbers) == 1:
            values.append(numbers[0])
        elif len(numbers) in (2, 3):
            start, stop = numbers[:2]
            step = numbers[2] if len(numbers) == 3 else 1
            if step == 0:
                raise ValueError("K range step cannot be zero.")
            if (stop - start) * step < 0:
                raise ValueError(f"K range has an incompatible step: {token!r}.")
            values.extend(range(start, stop + (1 if step > 0 else -1), step))
        else:
            raise ValueError(f"Invalid K specification: {token!r}.")
    values = sorted(set(int(x) for x in values))
    if not values or any(x <= 0 for x in values):
        raise ValueError("K values must be positive integers.")
    return values


def _parse_cv_spec(raw: Optional[str]) -> tuple[int, str]:
    if raw is None:
        return 0, "off"
    text = str(raw).strip()
    folds = _DEFAULT_CV_FOLDS if text == "" else int(text)
    if folds == 0:
        return 0, "off"
    if folds < 2:
        raise ValueError("-cv/--cv must be >= 2 when enabled.")
    return int(folds), f"{int(folds)}-fold"


def _resolve_input(args: argparse.Namespace, logger: logging.Logger) -> tuple[str, str, str]:
    genotype_path, auto_prefix = determine_genotype_source_from_args(args)
    if args.vcf:
        if not ensure_file_exists(logger, genotype_path, "VCF file"):
            raise FileNotFoundError(genotype_path)
        source = "VCF"
    elif args.hmp:
        if not ensure_file_exists(logger, genotype_path, "HMP file"):
            raise FileNotFoundError(genotype_path)
        source = "HMP"
    elif args.file:
        if not ensure_file_input_exists(logger, genotype_path, label="FILE genotype input"):
            raise FileNotFoundError(genotype_path)
        source = "FILE"
    elif args.bfile or args.pfile:
        if not ensure_plink_prefix_exists(logger, genotype_path, label="PLINK prefix"):
            raise FileNotFoundError(genotype_path)
        source = "PFILE" if args.pfile else "BFILE"
    else:
        raise ValueError("One genotype input is required.")
    return str(genotype_path), source, str(auto_prefix)


def _read_fam_ids(prefix: str) -> list[str]:
    result: list[str] = []
    with open(f"{prefix}.fam", "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.strip().split()
            if fields:
                result.append(fields[1] if len(fields) > 1 else fields[0])
    return result


def _read_psam_ids(prefix: str) -> list[str]:
    path = Path(f"{prefix}.psam")
    if not path.is_file():
        return []
    result: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            result.append(fields[1] if len(fields) > 1 else fields[0])
    return result


def _count_pvar_sites(prefix: str) -> int:
    candidates = [Path(f"{prefix}.pvar"), Path(f"{prefix}.pvar.zst"), Path(f"{prefix}.pvar.gz")]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return 0
    if path.suffix == ".zst":
        try:
            import zstandard as zstd

            with path.open("rb") as raw:
                reader = zstd.ZstdDecompressor().stream_reader(raw)
                with io.TextIOWrapper(reader, encoding="utf-8", errors="replace") as text:
                    return sum(
                        1
                        for line in text
                        if line.strip() and not line.startswith("#")
                    )
        except Exception:
            return 0
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if line.strip() and not line.startswith("#"))
    else:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def _read_vcf_metadata(path: str) -> tuple[list[str], int]:
    ids: list[str] = []
    n_sites = 0
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                fields = line.rstrip("\r\n").split("\t")
                ids = [str(x) for x in fields[9:]]
                continue
            if not line or line.startswith("#"):
                continue
            if len(line.rstrip("\r\n").split("\t")) >= 5:
                n_sites += 1
    return ids, n_sites


def _count_bim_sites(prefix: str) -> int:
    with open(f"{prefix}.bim", "r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def _inspect_sample_and_site_count(
    genotype_path: str,
    source: str,
) -> tuple[list[str], int]:
    """Read sample IDs and a site count without materialising all marker rows."""

    if source == "BFILE":
        prefix = str(genotype_path)
        if prefix.lower().endswith(".bed"):
            prefix = prefix[:-4]
        return _read_fam_ids(prefix), _count_bim_sites(prefix)
    if source == "PFILE":
        prefix = str(genotype_path)
        if prefix.lower().endswith(".pgen"):
            prefix = prefix[:-5]
        return _read_psam_ids(prefix), _count_pvar_sites(prefix)
    if source == "VCF":
        return _read_vcf_metadata(genotype_path)
    ids, n_snps = inspect_genotype_file(
        genotype_path, snps_only=False, maf=0.0, missing_rate=1.0
    )
    return [str(x) for x in ids], int(n_snps)


def _normalize_sample_metadata(
    ids: list[str],
    *,
    expected_n: int,
    logger: logging.Logger,
) -> list[str]:
    ids = list(ids)
    if len(ids) != int(expected_n):
        logger.warning(
            "Sample ID count differs from ADAMIXTURE Q rows; using generated IDs "
            f"(metadata={len(ids)}, Q={int(expected_n)})."
        )
        ids = [f"S{i + 1}" for i in range(int(expected_n))]
    return ids


def _inspect_admixture_input(
    genotype_path: str,
    source: str,
) -> tuple[list[str], int]:
    """Read sample IDs and a marker count for the opening CLI status line."""

    source_upper = str(source).upper()
    if source_upper in {"BFILE", "PFILE", "VCF"}:
        ids, n_snps = _inspect_sample_and_site_count(genotype_path, source_upper)
        # PVAR.ZST and some VCF readers may not expose metadata when an
        # optional decompressor is unavailable.  Fall back to the common
        # inspector for counts/IDs.
        if not ids or n_snps <= 0:
            try:
                fallback_ids, fallback_n_snps = inspect_genotype_file(
                    genotype_path,
                    snps_only=False,
                    maf=0.0,
                    missing_rate=1.0,
                )
                if not ids:
                    ids = [str(x) for x in fallback_ids]
                if n_snps <= 0:
                    n_snps = int(fallback_n_snps)
            except Exception:
                pass
        return ids, int(n_snps)

    ids, n_snps = inspect_genotype_file(
        genotype_path,
        snps_only=False,
        maf=0.0,
        missing_rate=1.0,
    )
    return [str(x) for x in ids], int(n_snps)


def _write_q(path: str, ids: list[str], matrix: np.ndarray) -> None:
    arr = np.asarray(matrix, dtype=np.float64)
    with open(path, "w", encoding="utf-8") as handle:
        for sample_id, row in zip(ids, arr):
            values = "\t".join(f"{float(x):.8f}" for x in row)
            handle.write(f"{sample_id}\t{values}\n")


def _write_p(path: str, matrix: np.ndarray) -> None:
    np.save(path, np.asarray(matrix, dtype=np.float32))


def _read_tag_values(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    path = Path(str(raw)).expanduser()
    if path.is_file():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _render_plot(
    q: np.ndarray,
    ids: list[str],
    path: str,
    tags: list[str],
    logger: logging.Logger,
) -> Optional[str]:
    try:
        from janusx.bioplotkit.popstructure import plot_admixture_structure

        plot_admixture_structure(
            np.asarray(q, dtype=np.float64),
            sample_ids=list(ids),
            out_path=str(path),
            tag_samples=list(tags),
            show_xticks=False,
            dpi=300,
        )
        return str(path)
    except Exception as exc:
        logger.warning(f"Failed to render JanusX structure plot: {exc}")
        return None


def _prepare_adamixture_input(
    source: str,
    genotype_path: str,
    threads: int,
    temp_dir: str | None = None,
) -> tuple[str, str]:
    """Resolve one reusable BED/PGEN input for the external ADAMIXTURE run.

    VCF/HMP and matrix inputs are converted through the shared JanusX cache
    layer once per command.  Existing BED/PGEN inputs are passed through so
    their BIM/PVAR remains the authoritative marker metadata.  Cache conversion
    is intentionally persistent and reusable instead of being repeated for
    every K value.  Legacy JanusX PGEN files using storage mode 0x01 are
    converted to a temporary BED fileset because the upstream reader rejects
    that mode. Native PGEN files are passed through directly.
    """

    source_upper = str(source).upper()
    if source_upper == "BFILE":
        return genotype_path, source_upper
    if source_upper == "PFILE":
        pgen_path = str(genotype_path)
        if not pgen_path.lower().endswith(".pgen"):
            pgen_path = f"{pgen_path}.pgen"
        try:
            with open(pgen_path, "rb") as handle:
                header = handle.read(3)
        except OSError as exc:
            raise RuntimeError(f"Cannot read PGEN header: {pgen_path}: {exc}") from exc
        if len(header) != 3 or header[:2] != b"\x6c\x1b":
            raise RuntimeError(f"Invalid PGEN header: {pgen_path}")
        if header[2] != 0x01:
            return genotype_path, source_upper
        if not temp_dir:
            raise RuntimeError(
                "A temporary directory is required for legacy mode-0x01 PFILE conversion."
            )
        prefix = str(Path(temp_dir) / "input")
        try:
            psam_prefix = str(genotype_path)
            if psam_prefix.lower().endswith(".pgen"):
                psam_prefix = psam_prefix[:-5]
            sample_ids = _read_psam_ids(psam_prefix)
            if not sample_ids:
                raise RuntimeError(f"PFILE has no readable sample IDs: {genotype_path}")
            save_genotype_streaming(
                prefix,
                sample_ids,
                load_genotype_chunks(
                    genotype_path,
                    chunk_size=8192,
                    maf=0.0,
                    missing_rate=1.0,
                    impute=False,
                    model="add",
                    snps_only=False,
                    force_kind="pfile",
                ),
                fmt="plink",
                # Keep this compatibility conversion quiet; the main CLI
                # reports the real genotype load and K progress below.
                total_snps=None,
                desc="Preparing ADAMIXTURE genotype",
            )
        except Exception as exc:
            raise RuntimeError(f"PFILE to BED conversion failed: {exc}") from exc
        return prefix + ".bed", "BFILE"
    if source_upper not in {"VCF", "HMP", "FILE"}:
        return genotype_path, source_upper

    cache_prefix = prepare_cli_input_cache(
        genotype_path,
        snps_only=False,
        prefer_plink_for_txt=True,
        force_kind=source_upper.lower(),
        threads=max(1, int(threads)),
    )
    cache_prefix = str(cache_prefix)
    bed_path = cache_prefix if cache_prefix.lower().endswith(".bed") else f"{cache_prefix}.bed"
    if not Path(bed_path).is_file():
        raise RuntimeError(
            "ADAMIXTURE genotype cache was not created: "
            f"{bed_path}"
        )
    return bed_path, "BFILE"


def _marker_metadata_path(genotype_path: str, source: str) -> str | None:
    """Return the BIM/PVAR used by the external run, if it is available."""

    raw = str(genotype_path)
    source_upper = str(source).upper()
    if source_upper == "BFILE":
        prefix = raw[:-4] if raw.lower().endswith(".bed") else raw
        candidate = Path(f"{prefix}.bim")
        return str(candidate) if candidate.is_file() else None
    if source_upper == "PFILE":
        prefix = raw[:-5] if raw.lower().endswith(".pgen") else raw
        for suffix in (".pvar", ".pvar.zst", ".pvar.gz"):
            candidate = Path(f"{prefix}{suffix}")
            if candidate.is_file():
                return str(candidate)
    return None


def _build_parser() -> CliArgumentParser:
    parser = CliArgumentParser(
        prog="jx adamixture",
        formatter_class=cli_help_formatter(),
        epilog=(
            minimal_help_epilog([
                "jx adamixture -bfile data/geno -k 8 -o out/cohort",
                "jx adamixture -pfile data/geno -k 1..10 -t 16",
            ])
            + "\n  https://github.com/AI-sandbox/ADAMIXTURE"
        ),
        description="Population structure inference delegated to the ADAMIXTURE CLI.",
    )
    required = parser.add_argument_group("Input and model")
    genotype = required.add_mutually_exclusive_group(required=True)
    add_common_genotype_source_args(genotype, include_file=True, help_profile="adamixture")
    required.add_argument("-k", "--k", required=True, help="K value, range, or comma-separated list.")

    output = parser.add_argument_group("Output")
    add_common_out_arg(output, default=".", help_profile="current_dir")
    add_common_prefix_arg(output, default=None, help_profile="simple")
    output.add_argument("-tag", "--tag", default=None, help="Sample labels file or comma-separated sample IDs.")
    output.add_argument("--no-plot", action="store_true", help="Skip JanusX structure plot rendering.")

    runtime = parser.add_argument_group("ADAMIXTURE runtime")
    add_common_thread_arg(runtime, default_threads=max(1, int(detect_effective_threads())), dest="thread")
    runtime.add_argument("-seed", "--seed", type=int, default=42)
    runtime.add_argument("--algorithm", choices=("brqn", "adamem"), default="adamem")
    runtime.add_argument("--device", choices=("cpu", "mps", "gpu"), default="cpu")
    runtime.add_argument("--n-inits", dest="n_inits", type=int, default=1)
    runtime.add_argument("-max-iter", "--max-iter", dest="max_iter", type=int, default=500)
    runtime.add_argument("-tol", "--tol", type=float, default=1e-5)
    runtime.add_argument(
        "-cv", "--cv", nargs="?", const=str(_DEFAULT_CV_FOLDS), default=None, metavar="N",
        help="Delegate N-fold cross-validation to ADAMIXTURE (default 5 when omitted).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if int(args.thread) <= 0:
        parser.error("-t/--thread must be positive")
    if int(args.max_iter) <= 0:
        parser.error("-max-iter/--max-iter must be positive")
    if float(args.tol) <= 0:
        parser.error("-tol/--tol must be positive")
    if int(args.n_inits) <= 0:
        parser.error("--n-inits must be positive")
    try:
        k_values = _parse_k_spec(args.k)
        cv_folds, cv_label = _parse_cv_spec(args.cv)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    bootstrap = logging.getLogger("janusx.adamixture.bootstrap")
    bootstrap.handlers.clear()
    bootstrap.addHandler(logging.NullHandler())
    bootstrap.setLevel(logging.INFO)
    bootstrap.propagate = False
    try:
        executable = resolve_executable()
        genotype_path, source, auto_prefix = _resolve_input(args, bootstrap)
    except (ExternalAdmixtureError, OSError, ValueError) as exc:
        print_failure(str(exc), force_color=True)
        raise

    outdir, outprefix, prefix = apply_output_prefix_compat(
        args, auto_prefix, argv=sys.argv[1:], fallback_prefix="adamixture"
    )
    os.makedirs(outdir, exist_ok=True)
    tags = _read_tag_values(args.tag)
    total_started = time.time()
    rows: list[dict[str, Any]] = []
    log_path = f"{outprefix}.adamixture.log"
    logger = setup_logging(log_path)
    _log_detail(logger, "=" * 60)
    _log_detail(logger, "JanusX - ADAMIXTURE")
    _log_detail(logger, "=" * 60)
    _log_detail(logger, "[ Command ]")
    _log_detail(logger, "  %s", _adamixture_invocation_command())
    _log_detail(logger, "")

    genotype_src = format_path_for_display(str(genotype_path))

    emit_cli_configuration(
        logger,
        app_title="JanusX - ADAMIXTURE",
        config_title="ADAMIXTURE CONFIG",
        host=socket.gethostname(),
        sections=[
            ("Input", [("Genotype", genotype_path), ("Input type", source), ("K", args.k)]),
            ("ADAMIXTURE", [
                ("Executable", executable),
                ("Threads", int(args.thread)),
                ("Seed", int(args.seed)),
                ("Algorithm", str(args.algorithm)),
                ("Device", str(args.device)),
                ("Initialisations", int(args.n_inits)),
                ("Max iter", int(args.max_iter)),
                ("Tol", float(args.tol)),
                ("CV", cv_label),
            ]),
        ],
        footer_rows=[("Output prefix", outprefix)],
    )

    inspect_t0 = time.monotonic()
    with CliStatus(
        genotype_load_status_open(genotype_src),
        enabled=stdout_is_tty(),
        # Metadata inspection is Python-side and does not need the separate
        # process spinner used by long native GRM calls.
        use_process=False,
    ) as load_status:
        try:
            input_ids, input_n_snps = _inspect_admixture_input(
                genotype_path,
                source,
            )
        except Exception:
            load_status.fail(genotype_load_status_fail(genotype_src))
            raise
        load_status.complete(
            genotype_load_status_done(
                genotype_src,
                n_samples=len(input_ids),
                n_snps=input_n_snps,
            ),
        )
    input_metadata_elapsed = max(0.0, time.monotonic() - inspect_t0)
    _log_detail(
        logger,
        "Input metadata: n=%d, nSNP=%d, elapsed=%.3fs",
        len(input_ids),
        int(input_n_snps),
        input_metadata_elapsed,
    )

    # Normalize VCF/HMP/matrix inputs once before the K loop. ADAMIXTURE then
    # reads the same BED/PGEN payload for every K. Native PGEN is passed
    # through directly; only legacy mode-0x01 PGEN needs a BED fallback.
    input_temp_dir = (
        tempfile.TemporaryDirectory(prefix=f".{prefix}.input.", dir=outdir)
        if source == "PFILE"
        else None
    )
    try:
        run_path, run_source = _prepare_adamixture_input(
            source,
            genotype_path,
            int(args.thread),
            input_temp_dir.name if input_temp_dir is not None else None,
        )
    except Exception:
        if input_temp_dir is not None:
            input_temp_dir.cleanup()
        raise
    # Keep the user's persistent PVAR as the marker reference. For legacy
    # mode-0x01 PGEN, the external reader consumes the temporary BED payload,
    # but marker coordinates still come from the original PVAR.
    marker_path = (
        _marker_metadata_path(genotype_path, source)
        if source == "PFILE"
        else _marker_metadata_path(run_path, run_source)
    )
    _log_detail(logger, "ADAMIXTURE genotype input: %s (%s)", run_path, run_source)

    multi_k_tty = bool(len(k_values) > 1 and stdout_is_tty())
    single_k_tty = bool(len(k_values) == 1 and stdout_is_tty())
    k_progress = ProgressBarAdapter(
        total=len(k_values),
        desc=("ADAMIXTURE" if single_k_tty else "ADAMIXTURE K scan"),
        show_spinner=True,
        show_postfix=False,
        keep_display=False,
        show_remaining=True,
        emit_done=not single_k_tty,
        force_animate=not single_k_tty,
        log_unit="K",
        show_bar=not single_k_tty,
        show_percentage=not single_k_tty,
    )

    try:
        for index, k in enumerate(k_values, start=1):
            k_i = int(k)
            _log_detail(logger, "ADAMIXTURE executable: %s", executable)
            _log_detail(logger, "Input: %s (%s)", genotype_path, source)
            _log_detail(logger, "K=%d, threads=%d, seed=%d", k_i, int(args.thread), int(args.seed))
            _log_detail(
                logger,
                "algorithm=%s, device=%s, n_inits=%d",
                args.algorithm,
                args.device,
                int(args.n_inits),
            )
            _log_detail(
                logger,
                "max_iter=%d, tol=%.6g, CV=%s",
                int(args.max_iter),
                float(args.tol),
                cv_label,
            )
            k_progress.set_desc(f"ADAMIXTURE K={k_i} ({index}/{len(k_values)})")

            def _stream_key_line(line: str) -> None:
                key_line = _adamixture_key_line(line)
                if key_line is not None:
                    _log_detail(logger, "K=%d | %s", k_i, key_line)
                    if len(k_values) == 1 and stdout_is_tty():
                        k_progress.emit_line(f"K={k_i} | {key_line}")

            q_mat, p_mat, elapsed = run_external_adamixture(
                genotype_path=run_path,
                outdir=outdir,
                prefix=prefix,
                k=k_i,
                threads=int(args.thread),
                seed=int(args.seed),
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                algorithm=str(args.algorithm),
                device=str(args.device),
                n_inits=int(args.n_inits),
                cv_folds=int(cv_folds),
                stream_callback=_stream_key_line,
            )
            _log_detail(logger, "K=%d | external run wall time: %.3fs", k_i, float(elapsed))
            k_progress.set_desc(f"Writing ADAMIXTURE K={k_i} outputs")
            ids = _normalize_sample_metadata(
                input_ids,
                expected_n=int(q_mat.shape[0]),
                logger=logger,
            )
            q_out = f"{outprefix}.{k_i}.Q.txt"
            p_out = f"{outprefix}.{k_i}.P.npy"
            _write_q(q_out, ids, q_mat)
            _write_p(p_out, p_mat)
            plot_out = None
            if not bool(args.no_plot):
                k_progress.set_desc(f"Plotting ADAMIXTURE K={k_i}")
                plot_out = _render_plot(
                    q_mat, ids, f"{outprefix}.{k_i}.adamixture.svg", tags, logger
                )
            _log_detail(logger, "Q output: %s", format_path_for_display(q_out))
            _log_detail(logger, "P output: %s", format_path_for_display(p_out))
            if plot_out:
                _log_detail(
                    logger,
                    "Structure plot: %s",
                    format_path_for_display(plot_out),
                )
            completion = (
                f"ADAMIXTURE K={k_i} ...Finished "
                f"[{format_elapsed(float(elapsed))}]"
            )
            _emit_to_file_handlers(logger, logging.INFO, completion)
            if multi_k_tty:
                # Keep a single live row for a K sweep.  The next K will
                # replace this description in place; no completion line
                # is printed for every K.
                k_progress.update(1)
                k_progress.set_desc(f"✔︎ {completion}")
            elif stdout_is_tty():
                # Single-K TTY runs show the filtered iteration details;
                # the final generic completion line is emitted below.
                k_progress.update(1)
            else:
                k_progress.emit_line(completion, success=True)
            rows.append({
                "k": k_i,
                "q": q_out,
                "p": p_out,
                "plot": plot_out,
                "log": str(getattr(logger, "_janusx_log_path", log_path)),
                "wall": elapsed,
            })
            if not multi_k_tty and not stdout_is_tty():
                k_progress.update(1)
    except Exception as exc:
        k_progress.close(show_done=False)
        if input_temp_dir is not None:
            input_temp_dir.cleanup()
        print_failure(f"ADAMIXTURE failed: {exc}", force_color=True)
        raise
    else:
        # Close the aggregate live bar silently; the terminal receives one
        # generic completion line after the saved-output report below.
        k_progress.finish()
        k_progress.close(show_done=False)

    total_elapsed = max(0.0, time.time() - total_started)
    final_status = f"ADAMIXTURE ...Finished [{format_elapsed(total_elapsed)}]"
    _log_detail(logger, final_status)
    if multi_k_tty:
        print_success(final_status, force_color=True)

    _log_detail(logger, "ADAMIXTURE outputs:")
    for row in rows:
        _log_detail(
            logger,
            "K=%d: Q=%s P=%s",
            row["k"],
            row["q"],
            row["p"],
        )
    saved_lines = ["Saved ADAMIXTURE outputs:"]
    if marker_path is not None:
        saved_lines.insert(
            0,
            f"Marker metadata: {format_path_for_display(marker_path)}",
        )
    for row in rows:
        saved_lines.append(f"  K={int(row['k'])}:")
        saved_lines.append(f"    {format_path_for_display(row['q'])}")
        saved_lines.append(f"    {format_path_for_display(row['p'])}")
        if row.get("plot"):
            saved_lines.append(f"    {format_path_for_display(row['plot'])}")
    log_success(
        logger,
        "\n".join(saved_lines),
    )
    endinfo = (
        f"\nTotal wall time: "
        f"{round(total_elapsed, 2)} seconds\n"
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    )
    log_success(logger, endinfo)
    if input_temp_dir is not None:
        input_temp_dir.cleanup()
    return 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    main()
