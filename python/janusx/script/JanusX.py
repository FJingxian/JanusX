#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
    jx <module> [options]

Options:
    -h, --help             Show this help message
    -v, --version          Show version/build information
    -update, --update      Update JanusX: `jx --update [latest] [--verbose]`

Modules:
    Genome-wide Association Studies (GWAS):
    grm           Build genomic relationship matrix
    pca           Principal component analysis for population structure
    gwas          Run genome-wide association analysis
    postgwas      Post-process GWAS results and downstream plots

    Genomic Selection (GS):
    gs            Genomic prediction and model-based selection

    GARFIELD:
    garfield      Random-forest based marker-trait association
    postgarfield  Summarize and visualize GARFIELD outputs

    Bulk Segregation Analysis (BSA):
    postbsa       Post-process and visualize BSA results

    Pipeline and utility:
    fastq2vcf     Variant-calling pipeline from FASTQ to VCF
    gmerge        Merge genotype/variant tables

    Benchmark:
    sim           Quick simulation workflow
    simulation    Extended simulation and benchmarking workflow
'''
import sys
import subprocess
from datetime import date
from pathlib import Path
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*ChainedAssignmentError.*"
)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.backends.backend_pdf # pdf support
import matplotlib.backends.backend_svg # svg support
from janusx.script import gwas, gs, postgwas, postgarfield, postbsa, garfield, grm, pca, sim, gmerge, fastq2vcf, simulation, update
from importlib.metadata import version, PackageNotFoundError
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

def _print_help() -> None:
    print(__logo__)
    if __doc__:
        print(__doc__.strip())


def main():
    module = dict(zip(
        ['gwas','postgwas','postgarfield','postbsa','garfield','grm','pca','gs','sim','simulation','gmerge','fastq2vcf'],
        [gwas,postgwas,postgarfield,postbsa,garfield,grm,pca,gs,sim,simulation,gmerge,fastq2vcf],
    ))
    if len(sys.argv)>1:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            _print_help()
        elif sys.argv[1] == '-v' or sys.argv[1] == '--version':
            print(__logo__)
            print(__version__)
        elif sys.argv[1] == '-update' or sys.argv[1] == '--update':
            print(__logo__)
            sys.argv[0] = "jx --update"
            del sys.argv[1]
            update.main()
        else:
            module_name = sys.argv[1]
            if sys.argv[1] in module.keys():
                # Keep argparse usage as "jx <module> ..."
                sys.argv[0] = f"jx {module_name}"
                del sys.argv[1]
                module[module_name].main() # Process of Target Module
            elif sys.argv[1] not in module.keys():
                print(f"Unknown module: {sys.argv[1]}")
                _print_help()
    else:
        _print_help()

if __name__ == "__main__":
    main()
