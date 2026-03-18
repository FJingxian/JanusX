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
    fastq2vcf     Variant-calling pipeline from FASTQ to VCF (launcher-only)
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
from datetime import date
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from ._common.interrupt import install_interrupt_handlers, force_exit
try:
    v = version("janusx")
except PackageNotFoundError:
    v = "0.0.0"

__BUILD_DATE_FALLBACK__ = "2026-03-04"


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
    "sim", "simulation", "adamixture", "gformat", "gmerge", "hybrid", "webui",
]
_LAUNCHER_ONLY_FLAGS = {
    "-update", "--update",
    "-clean", "--clean",
    "-list", "--list",
    "-upgrade", "--upgrade",
    "-uninstall", "--uninstall",
}
_LAUNCHER_ONLY_MODULES = {
    "fastq2vcf",
    "fastq2count",
}

def _load_script_module(name: str):
    return importlib.import_module(f"janusx.script.{name}")

def _print_help() -> None:
    print(__logo__)
    if __doc__:
        print(__doc__.strip())


def main():
    install_interrupt_handlers()
    if len(sys.argv) > 1:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            _print_help()
        elif sys.argv[1] == '-v' or sys.argv[1] == '--version':
            print(__logo__)
            print(__version__)
        elif sys.argv[1] in _LAUNCHER_ONLY_FLAGS:
            print(__logo__)
            launcher_cmd = "jx " + " ".join(sys.argv[1:])
            print(f"Error: `{sys.argv[1]}` is launcher-only and not available in `jxpy`.")
            print(f"Please use launcher command: `{launcher_cmd}`")
        else:
            module_name = sys.argv[1]
            if module_name in _LAUNCHER_ONLY_MODULES:
                print(__logo__)
                launcher_cmd = "jx " + " ".join(sys.argv[1:])
                print(f"Error: module `{module_name}` is launcher-only and not available in `jxpy`.")
                print(f"Please use launcher command: `{launcher_cmd}`")
                return
            if module_name in _MODULE_NAMES:
                # Keep argparse usage as "jx <module> ..."
                sys.argv[0] = f"jx {module_name}"
                del sys.argv[1]
                try:
                    _load_script_module(module_name).main()  # Process of target module
                except KeyboardInterrupt:
                    force_exit(130, "Interrupted by user (Ctrl+C).")
            elif module_name not in _MODULE_NAMES:
                print(f"Unknown module: {sys.argv[1]}")
                _print_help()
    else:
        _print_help()

if __name__ == "__main__":
    install_interrupt_handlers()
    main()
