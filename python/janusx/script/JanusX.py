#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
    jx <module> [options]

Options:
    -h, --help             Show this help message
    -v, --version          Show version/build information
    Note: launcher-only flags are not supported in jxpy:
          -update/-clean/-list/-upgrade/-uninstall
          Please use launcher command: `jx ...`

Modules:
    Genome-wide Association Studies (GWAS):
    grm           Build genomic relationship matrix
    pca           Principal component analysis for population structure
    gwas          Run genome-wide association analysis
    postgwas      Post-process GWAS results and downstream plots

    Genomic Selection (GS):
    gs            Genomic prediction and model-based selection
    reml          Estimate heritability/effect components by REML-BLUP

    Genetic Association by Random Forest and InterpretivE Logic Decisions (GARFIELD):
    garfield      Random-forest based marker-trait association
    postgarfield  Summarize and visualize GARFIELD outputs

    Bulk Segregation Analysis (BSA):
    postbsa       Post-process and visualize BSA results

    Pipeline and utility:
    fastq2vcf     Variant-calling pipeline from FASTQ to VCF
    fastq2count   RNA-seq counting pipeline from FASTQ to count matrix
    tree          Tree workflow entry (`-nj` Neighbor-Joining / `-ml` Rust ML v1)
    adamixture    ADAMIXTURE ancestry inference
    hybrid        Build pairwise hybrid genotype matrix from parent lists
    gformat       Convert genotype files across plink/vcf/txt/npy
    gmerge        Merge genotype/variant tables
    webui         Start JanusX web UI (postgwas first)

    Benchmark:
    sim           Quick simulation workflow
    simulation    Extended simulation and benchmarking workflow
'''
import sys
import subprocess
import importlib
import os
import textwrap
from datetime import date
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from janusx._optional_deps import format_missing_dependency_for_module
from ._common.interrupt import install_interrupt_handlers, force_exit
try:
    v = version("janusx")
except PackageNotFoundError:
    v = "0.0.0"

__BUILD_DATE_FALLBACK__ = "2026-03-28"


def _build_date() -> str:
    """
    Resolve build date from latest git commit date.
    Fallback order:
      1) release-tag-updated fallback date in source
      2) local current date
    """
    repo_root = Path(__file__).resolve().parents[3]
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "log", "-1", "--date=format:%Y-%m-%d", "--format=%cd"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    if __BUILD_DATE_FALLBACK__:
        return __BUILD_DATE_FALLBACK__
    return date.today().isoformat()

__logo__ = r'''
       _                      __   __
      | |                     \ \ / /
      | | __ _ _ __  _   _ ___ \ V / 
  _   | |/ _` | '_ \| | | / __| > <  
 | |__| | (_| | | | | |_| \__ \/ . \ 
  \____/ \__,_|_| |_|\__,_|___/_/ \_\ Tools for GWAS and GS
  ---------------------------------------------------------
'''
__version__ = (
    f"JanusX v{v} by Jingxian FU, Yazhouwan National Laboratory\n"
    "Please report issues to <fujingxian@yzwlab.cn>\n"
    f"Build date: {_build_date()}\n"
)

_MODULE_NAMES = [
    "gwas", "postgwas", "postgarfield", "postbsa",
    "garfield", "grm", "pca", "gs", "reml",
    "sim", "simulation", "adamixture", "tree", "gformat", "gmerge", "hybrid", "webui",
    "fastq2vcf", "fastq2count",
]
_LAUNCHER_ONLY_FLAGS = {
    "-update", "--update",
    "-clean", "--clean",
    "-list", "--list",
    "-upgrade", "--upgrade",
    "-uninstall", "--uninstall",
}
_LAUNCHER_ONLY_MODULES = set()

def _load_script_module(name: str):
    return importlib.import_module(f"janusx.script.{name}")


def _supports_color() -> bool:
    """Match launcher behavior: color only when stdout is a terminal."""
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _style_green(text: str) -> str:
    if _supports_color():
        return f"\033[32m{text}\033[0m"
    return text


def _style_yellow(text: str) -> str:
    if _supports_color():
        return f"\033[33m{text}\033[0m"
    return text


def _style_blue(text: str) -> str:
    if _supports_color():
        return f"\033[34m{text}\033[0m"
    return text


def _style_orange(text: str) -> str:
    if _supports_color():
        return f"\033[38;5;208m{text}\033[0m"
    return text


def _style_white(text: str) -> str:
    if _supports_color():
        return f"\033[37m{text}\033[0m"
    return text


def _help_line_width() -> int:
    cols = 100
    val = str(os.environ.get("COLUMNS", "")).strip()
    if val.isdigit():
        try:
            parsed = int(val)
            if parsed > 0:
                cols = parsed
        except Exception:
            pass
    return max(48, min(160, cols - 2))


def _wrap_help_text(text: str, width: int) -> list[str]:
    if width <= 0:
        return [text]
    wrapped = textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped if wrapped else [""]


def _print_help_entry(indent: int, key: str, desc: str, key_width: int, total_width: int) -> None:
    lead = " " * indent
    min_desc_width = 20
    key_padded = f"{key:<{key_width}}"
    min_total = indent + key_width + 2 + min_desc_width
    if total_width <= min_total:
        print(f"{lead}{_style_green(key)}")
        for line in _wrap_help_text(desc, max(min_desc_width, total_width - indent - 2)):
            print(f"{lead}  {_style_white(line)}")
        return

    desc_width = total_width - indent - key_width - 2
    wrapped = _wrap_help_text(desc, desc_width)
    if not wrapped:
        print(f"{lead}{_style_green(key_padded)}")
        return

    print(f"{lead}{_style_green(key_padded)}  {_style_white(wrapped[0])}")
    pad = " " * key_width
    for line in wrapped[1:]:
        print(f"{lead}{pad}  {_style_white(line)}")


def _print_cli_help() -> None:
    width = _help_line_width()
    print(__logo__)
    print(_style_orange("Usage:"))
    print(f"  {_style_green('jx [flags]')}")
    print(f"  {_style_green('jx [module] -h')}")
    print()

    print(_style_orange("Flags:"))
    _print_help_entry(2, "-h, --help", "Show this help message", 16, width)
    _print_help_entry(2, "-v, --version", "Show version/build information", 16, width)
    print()
    print(f"  {_style_blue('Launcher-only flags (use `jx`):')}")
    _print_help_entry(
        4,
        "-update/-clean/-list/-upgrade/-uninstall",
        "Not available in `jxpy`; please use launcher command `jx ...`.",
        40,
        width,
    )
    print()

    print(_style_orange("Modules:"))
    print(f"  {_style_blue('Genome-wide Association Studies (GWAS):')}")
    _print_help_entry(4, "grm", "Build genomic relationship matrix", 12, width)
    _print_help_entry(4, "pca", "Principal component analysis for population structure", 12, width)
    _print_help_entry(4, "gwas", "Run genome-wide association analysis", 12, width)
    _print_help_entry(4, "postgwas", "Post-process GWAS results and downstream plots", 12, width)
    print()

    print(f"  {_style_blue('Genomic Selection (GS):')}")
    _print_help_entry(4, "gs", "Genomic prediction and model-based selection", 12, width)
    _print_help_entry(4, "reml", "Estimate heritability/effect components by REML-BLUP", 12, width)
    print()

    print(
        f"  {_style_blue('Genetic Association by Random Forest and InterpretivE Logic Decisions (GARFIELD):')}"
    )
    _print_help_entry(4, "garfield", "Random-forest based marker-trait association", 12, width)
    _print_help_entry(4, "postgarfield", "Summarize and visualize GARFIELD outputs", 12, width)
    print()

    print(f"  {_style_blue('Bulk Segregation Analysis (BSA):')}")
    _print_help_entry(4, "postbsa", "Post-process and visualize BSA results", 12, width)
    print()

    print(f"  {_style_blue('Pipeline and utility:')}")
    _print_help_entry(4, "fastq2vcf", "Variant-calling pipeline from FASTQ to VCF", 12, width)
    _print_help_entry(4, "fastq2count", "RNA-seq counting pipeline from FASTQ to count matrix", 12, width)
    _print_help_entry(4, "tree", "Tree workflow entry (`-nj` Neighbor-Joining / `-ml` Rust ML v1)", 12, width)
    _print_help_entry(4, "adamixture", "ADAMIXTURE ancestry inference", 12, width)
    _print_help_entry(4, "hybrid", "Build pairwise hybrid genotype matrix from parent lists", 12, width)
    _print_help_entry(4, "gformat", "Convert genotype files across plink/vcf/txt/npy", 12, width)
    _print_help_entry(4, "gmerge", "Merge genotype/variant tables", 12, width)
    _print_help_entry(4, "webui", "Start JanusX web UI (postgwas first)", 12, width)
    print()

    print(f"  {_style_blue('Benchmark:')}")
    _print_help_entry(4, "sim", "Quick simulation workflow", 12, width)
    _print_help_entry(4, "simulation", "Extended simulation and benchmarking workflow", 12, width)


def _print_help() -> None:
    _print_cli_help()


def main():
    install_interrupt_handlers()
    if str(os.environ.get("JANUSX_ENTRYPOINT", "")).strip() == "":
        prog = Path(sys.argv[0]).name.lower() if len(sys.argv) > 0 else ""
        if prog.startswith("jxpy"):
            os.environ["JANUSX_ENTRYPOINT"] = "jxpy"
    if len(sys.argv) > 1:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            _print_help()
        elif sys.argv[1] == '-v' or sys.argv[1] == '--version':
            print(__logo__)
            print(__version__)
        elif sys.argv[1] in _LAUNCHER_ONLY_FLAGS:
            print(__logo__)
            launcher_cmd = "jx " + " ".join(sys.argv[1:])
            print(_style_yellow(f"Error: `{sys.argv[1]}` is launcher-only and not available in `jxpy`."))
            print(_style_yellow(f"Please use launcher command: `{launcher_cmd}`"))
        else:
            module_name = sys.argv[1]
            if module_name in _LAUNCHER_ONLY_MODULES:
                print(__logo__)
                launcher_cmd = "jx " + " ".join(sys.argv[1:])
                print(_style_yellow(f"Error: module `{module_name}` is launcher-only and not available in `jxpy`."))
                print(_style_yellow(f"Please use launcher command: `{launcher_cmd}`"))
                return
            if module_name in _MODULE_NAMES:
                # Keep argparse usage as "jx <module> ..."
                sys.argv[0] = f"jx {module_name}"
                del sys.argv[1]
                try:
                    result = _load_script_module(module_name).main()  # Process of target module
                    if isinstance(result, int):
                        raise SystemExit(result)
                except ModuleNotFoundError as exc:
                    msg = format_missing_dependency_for_module(
                        exc.name,
                        f"Missing optional dependency required by `jx {module_name}`.",
                        original_error=exc,
                    )
                    if msg is None:
                        raise
                    print(__logo__)
                    print(f"Error: {msg}")
                    raise SystemExit(1)
                except KeyboardInterrupt:
                    force_exit(130, "Interrupted by user (Ctrl+C).")
            elif module_name not in _MODULE_NAMES:
                print(_style_yellow(f"Unknown module: {sys.argv[1]}"))
                _print_help()
    else:
        _print_help()

if __name__ == "__main__":
    install_interrupt_handlers()
    main()
