#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
    jx <module> [options]

Options:
    -h, --help             Show this help message
    -v, --version          Show version/build information
    -update, --update      Update JanusX: `jx --update [latest] [--verbose]`
    -load, --load          List/load files: `jx --load` or `jx --load <type> <name> <file>`
    -clean, --clean        Clear GWAS history DB, or `jx --clean <load_id>` to remove one loaded file record

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
from datetime import date
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from ._common.gwas_history import remove_loaded_file_by_id
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
    "garfield", "grm", "pca", "gs",
    "sim", "simulation", "gformat", "gmerge", "fastq2vcf", "hybrid", "webui", "loadanno",
]


def _load_script_module(name: str):
    return importlib.import_module(f"janusx.script.{name}")

def _print_help() -> None:
    print(__logo__)
    if __doc__:
        print(__doc__.strip())


def _resolve_runtime_home() -> Path:
    env_home = os.environ.get("JX_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()
    p1 = (Path.home() / "JanusX" / ".janusx").resolve()
    p2 = (Path.home() / ".janusx").resolve()
    return p1 if p1.exists() else p2


def _clean_history_db() -> None:
    target = _resolve_runtime_home() / "janusx_tasks.db"
    removed = []
    if target.exists():
        try:
            target.unlink()
            removed.append(str(target))
        except Exception as e:
            print(f"Error: failed to remove {target}: {e}")
            return
    if len(removed) == 0:
        print("GWAS history DB already clean.")
    else:
        print("GWAS history DB cleaned.")
        for p in removed:
            print(f"  removed {p}")
    print("A new history DB will be created automatically on next `jx gwas`.")


def _clean_loaded_by_id(load_id: str) -> None:
    out = remove_loaded_file_by_id(load_id)
    print(f"Removed load id: {out.get('id','')}")
    print(f"  type: {out.get('type','')}  name: {out.get('name','')}")
    for p in out.get("removed_paths", []):
        print(f"  removed {p}")
    for p in out.get("failed_paths", []):
        print(f"  failed {p}")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            _print_help()
        elif sys.argv[1] == '-v' or sys.argv[1] == '--version':
            print(__logo__)
            print(__version__)
        elif sys.argv[1] == '-update' or sys.argv[1] == '--update':
            print(__logo__)
            sys.argv[0] = "jx --update"
            del sys.argv[1]
            _load_script_module("update").main()
        elif sys.argv[1] == '-clean' or sys.argv[1] == '--clean':
            print(__logo__)
            clean_args = sys.argv[2:]
            if len(clean_args) == 0:
                _clean_history_db()
            elif len(clean_args) == 1:
                _clean_loaded_by_id(clean_args[0])
            else:
                print("Usage: jx --clean [<load_id>]")
        elif sys.argv[1] == '-load' or sys.argv[1] == '--load':
            print(__logo__)
            sys.argv[0] = "jx --load"
            del sys.argv[1]
            _load_script_module("loadanno").main()
        else:
            module_name = sys.argv[1]
            if module_name in _MODULE_NAMES:
                # Keep argparse usage as "jx <module> ..."
                sys.argv[0] = f"jx {module_name}"
                del sys.argv[1]
                _load_script_module(module_name).main()  # Process of target module
            elif module_name not in _MODULE_NAMES:
                print(f"Unknown module: {sys.argv[1]}")
                _print_help()
    else:
        _print_help()

if __name__ == "__main__":
    main()
