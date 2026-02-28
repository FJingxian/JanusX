#!/user/bin/env python
# -*- coding: utf-8 -*-
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

__BUILD_DATE_FALLBACK__ = "2026-02-27"


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
'''
_banner_line = "*" * 60
__version__ = (
    f"{_banner_line}\n"
    f">JanusX v{v} by Jingxian FU, Yazhouwan National Laboratory\n"
    "Please report issues to <fujingxian@yzwlab.cn>\n"
    f"Build date: {_build_date()}\n"
    f"{_banner_line}"
)

def main():
    module = dict(zip(
        ['gwas','postgwas','postgarfield','postbsa','garfield','grm','pca','gs','sim','simulation','gmerge','fastq2vcf','update'],
        [gwas,postgwas,postgarfield,postbsa,garfield,grm,pca,gs,sim,simulation,gmerge,fastq2vcf,update],
    ))
    if len(sys.argv)>1:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            print(__logo__)
            print("Usage: jx <module> [options]")
            print(f"Available modules: {' '.join(module.keys())}")
            print("Shortcut: jx -update / --update  (same as: jx update)")
        elif sys.argv[1] == '-v' or sys.argv[1] == '--version':
            print(__logo__)
            print(__version__)
        elif sys.argv[1] == '-update' or sys.argv[1] == '--update':
            print(__logo__)
            # Keep argparse usage as "jx --update ..."
            sys.argv[0] = "jx update"
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
                print(f"Usage: {sys.argv[0]} <module> [options]")
                print(f"Available modules: {' '.join(module.keys())}")
    else:
        print(f"Usage: {sys.argv[0]} <module> [options]")
        print(f"Available modules: {' '.join(module.keys())}")
        print("Shortcut: jx -update / --update  (same as: jx update)")

if __name__ == "__main__":
    main()
