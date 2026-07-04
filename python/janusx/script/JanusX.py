#!/user/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

'''
Usage:
    jx <module> [options]

Options:
    -h, --help             Show this help message
    -v, --version          Show version/build information

Modules:
    Genome-wide Association Studies (GWAS):
    grm           Build genomic relationship matrix
    pca           Principal component analysis for population structure
    gstats        Genotype basic statistics and LD score
    gwas          Run genome-wide association analysis
    postgwas      Post-process GWAS results and downstream plots

    Genomic Selection (GS):
    gs            Genomic prediction and model-based selection
    reml          Estimate broad/narrow heritability and BLUE by REML
    postgs        Summarize and visualize GS results

    Pipeline and utility:
    garfield      Random-forest based marker-trait association
    postbsa       Post-process and visualize BSA results
    kmer          K-mer counting workflow via KMC
    kmerge        Merge multi-sample KMC databases into genotype matrix
    kstats        Compute pairwise KMC k-mer statistics
    view          View .bkmer/.bsite as plain text for pipes
    tree          Tree workflow entry (`-nj` Neighbor-Joining / `-ml` FastTree ML)
    treeplot      Visualize Newick/GRM trees with toytree
    fastpop       FastPop ancestry inference
    hybrid        Build pairwise hybrid genotype matrix from parent lists
    gformat       Convert genotype files across plink/vcf/txt/npy
    gmerge        Merge genotype/variant tables
'''
import sys
import subprocess
import importlib
import os
import re
import difflib
import textwrap
from datetime import date
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from janusx._optional_deps import format_missing_dependency_for_module
from ._common.interrupt import install_interrupt_handlers, force_exit
from ._common.threads import apply_outer_thread_cap, detect_effective_threads
try:
    v = version("janusx")
except PackageNotFoundError:
    v = "0.0.0"

_CONSOLE = Console()
_CLI_HELP_MAX_WIDTH = 100
_CLI_HELP_KEY_WIDTH = 14
_CLI_HELP_GAP = 2

__BUILD_DATE_FALLBACK__ = "2026-07-05"


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
_BUILD_DATE = _build_date()
_HELP_BUILD_DATE = __BUILD_DATE_FALLBACK__ or _BUILD_DATE
__version__ = (
    f"JanusX v{v} by Jingxian FU, Huazhong Agricultural University\n"
    "Please report issues to <fujingxian@webmail.hzau.edu.cn>\n"
    f"Build date: {_BUILD_DATE}\n"
)
_CLI_FLAGS = [
    ("-h, --help", "Show this help message"),
    ("-v, --version", "Show version/build information"),
]
_CLI_MODULE_SECTIONS = [
    (
        "Genome-wide Association Studies (GWAS)",
        [
            ("grm", "Build genomic relationship matrix"),
            ("pca", "Principal component analysis for population structure"),
            ("gwas", "Run genome-wide association analysis"),
            ("postgwas", "Post-process GWAS results and downstream plots"),
        ],
    ),
    (
        "Genomic Selection (GS)",
        [
            ("gs", "Genomic prediction and model-based selection"),
            ("reml", "Estimate broad/narrow heritability and BLUE by REML"),
            ("postgs", "Summarize and visualize GS results"),
        ],
    ),
    (
        "Pipeline and Utility",
        [
            ("garfield", "Random-forest based marker-trait association"),
            ("postbsa", "Post-process and visualize BSA results"),
            ("kmer", "K-mer counting workflow via KMC"),
            ("kmerge", "Merge multi-sample KMC databases into genotype matrix"),
            ("kstats", "Compute pairwise KMC k-mer statistics"),
            ("view", "View .bkmer/.bsite as plain text for shell pipes"),
            ("tree", "Tree workflow entry (`-nj` Neighbor-Joining / `-ml` FastTree ML)"),
            ("treeplot", "Visualize Newick/GRM trees with toytree"),
            ("fastpop", "FastPop ancestry inference"),
            ("hybrid", "Build pairwise hybrid genotype matrix from parent lists"),
            ("gformat", "Convert genotype files across plink/vcf/txt/npy"),
            ("gmerge", "Merge genotype/variant tables"),
        ],
    ),
]

_MODULE_NAMES = [
    "gwas", "postgwas", "postgarfield", "postbsa",
    "garfield", "grm", "pca", "gstats", "gs", "reml", "postgs",
    "sim", "simulation", "benchmark", "gblupbench", "bayesbench", "garfieldbench", "fastpop", "adamixture", "tree", "gformat", "gmerge", "hybrid", "webui",
    "fastq2vcf", "fastq2count", "kmer", "kmerge", "kstats", "view", "treeplot", "refcheck",
]
_SCRIPT_MODULE_ALIASES = {
    # Keep CLI surface stable (`jx gwas` / `jx gs`) while routing heavy
    # implementations to workflow modules.
    "gwas": "janusx.assoc.workflow",
    "gs": "janusx.gs.workflow",
}
_LAUNCHER_ONLY_FLAGS = {
    "-update", "--update",
    "-clean", "--clean",
    "-list", "--list",
    "-upgrade", "--upgrade",
    "-uninstall", "--uninstall",
}
_LAUNCHER_ONLY_MODULES = set()

_THREAD_FLAG_ALIASES = ("-t", "--thread", "--threads", "-threads")


def _parse_cli_thread_value(argv_tail: list[str]) -> int | None:
    """
    Best-effort parse for common thread flags before heavy module imports.

    Supported forms:
      - `-t 16`
      - `--thread 16`
      - `--threads=16`
      - `-threads=16`
    """
    if len(argv_tail) == 0:
        return None

    for i, tok in enumerate(argv_tail):
        t = str(tok).strip()
        if t == "":
            continue

        # key=value forms
        for alias in _THREAD_FLAG_ALIASES:
            prefix = f"{alias}="
            if t.startswith(prefix):
                raw = t[len(prefix):].strip()
                try:
                    v = int(raw)
                except Exception:
                    return None
                return v if v > 0 else None

        # separated forms: --thread 16
        if t in _THREAD_FLAG_ALIASES:
            if i + 1 >= len(argv_tail):
                return None
            raw = str(argv_tail[i + 1]).strip()
            # Accept plain positive integer only for early env config.
            if re.fullmatch(r"[+]?\d+", raw) is None:
                return None
            v = int(raw)
            return v if v > 0 else None
    return None


def _apply_early_thread_env_from_cli(module_name: str, argv_tail: list[str]) -> None:
    """
    Pre-apply thread env vars before loading target submodule.

    This helps BLAS backends (especially Accelerate on macOS) pick up user
    thread limits before NumPy/SciPy are imported in submodule top-level code.
    """
    _ = module_name  # reserved for module-specific overrides if needed later
    t = _parse_cli_thread_value(argv_tail)
    if t is None:
        # Important for macOS Accelerate: set thread env before NumPy/SciPy import
        # even when user does not pass -t explicitly.
        try:
            t = int(detect_effective_threads())
        except Exception:
            t = 1
    _ = apply_outer_thread_cap(max(1, int(t)), set_jx_threads=True, set_max_keys=True)

def _load_script_module(name: str):
    target = _SCRIPT_MODULE_ALIASES.get(str(name), str(name))
    if str(target).startswith("janusx."):
        return importlib.import_module(str(target))
    return importlib.import_module(f"janusx.script.{target}")


def _suggest_module_names(name: str, limit: int = 1) -> list[str]:
    query = str(name).strip().lower()
    if query == "":
        return []

    candidates = sorted(set(_MODULE_NAMES))
    lowered = {candidate.lower(): candidate for candidate in candidates}
    matches = difflib.get_close_matches(query, list(lowered.keys()), n=max(1, int(limit)), cutoff=0.5)
    return [lowered[match] for match in matches]


def _print_unknown_module_error(module_name: str) -> None:
    suggestions = _suggest_module_names(module_name)
    _CONSOLE.print(f"[bold red]Error:[/bold red] '{module_name}' is not a recognized JanusX module.")
    if len(suggestions) == 1:
        _CONSOLE.print("The most similar module is:")
        _CONSOLE.print(f"    [green]{suggestions[0]}[/green]")


def _render_logo() -> Panel:
    logo = Text(__logo__.strip("\n"), style="bold green")
    return Panel.fit(logo, border_style="green", padding=(0, 1))


def _help_panel_width() -> int:
    try:
        width = int(_CONSOLE.size.width)
    except Exception:
        width = _CLI_HELP_MAX_WIDTH
    if width <= 0:
        return _CLI_HELP_MAX_WIDTH
    return min(width, _CLI_HELP_MAX_WIDTH)


def _help_content_width() -> int:
    return max(20, _help_panel_width() - 4)


def _render_help_header() -> Panel:
    title = Text(f"JanusX (v{v}  Build date: {_HELP_BUILD_DATE})", style="bold green", justify="center")
    return Panel(
        title,
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 1),
        width=_help_panel_width(),
    )


def _render_help_text(
    rows: list[tuple[str, str]],
    *,
    key_style: str = "bold green",
    desc_style: str = "white",
    first_key_style: str | None = None,
    first_desc_style: str | None = None,
) -> Text:
    text = Text()
    content_width = _help_content_width()
    key_width = int(_CLI_HELP_KEY_WIDTH)
    gap = " " * int(_CLI_HELP_GAP)
    desc_width = max(10, content_width - key_width - len(gap))
    for idx, (key_raw, desc_raw) in enumerate(rows):
        key = str(key_raw)
        desc = str(desc_raw)
        key_row_style = first_key_style if idx == 0 and first_key_style is not None else key_style
        desc_row_style = first_desc_style if idx == 0 and first_desc_style is not None else desc_style
        wrapped = textwrap.wrap(
            desc,
            width=desc_width,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        for line_idx, line in enumerate(wrapped):
            line_key = key if line_idx == 0 else ""
            text.append(f"{line_key:<{key_width}}", style=key_row_style if line_idx == 0 else "")
            text.append(gap)
            text.append(line, style=desc_row_style)
            if not (idx == len(rows) - 1 and line_idx == len(wrapped) - 1):
                text.append("\n")
    return text


def _render_module_section(title: str, rows: list[tuple[str, str]]) -> Panel:
    return Panel(
        _render_help_text([(str(key), str(desc)) for key, desc in rows]),
        title=f"[bold blue]{title}[/bold blue]",
        title_align="left",
        border_style="blue",
        box=box.ROUNDED,
        padding=(0, 1),
        width=_help_panel_width(),
    )


def _print_cli_help() -> None:
    parts: list[object] = [
        _render_help_header(),
        Panel(
            Group(
                _render_help_text(
                    [
                        ("Usage", "jx [flags]"),
                        ("", "jx [module] -h"),
                    ],
                    first_key_style="bold orange3",
                    first_desc_style="bold green",
                    desc_style="bold green",
                ),
                Text(""),
                _render_help_text([(str(key), str(desc)) for key, desc in _CLI_FLAGS]),
            ),
            title="[bold orange3]CLI[/bold orange3]",
            title_align="left",
            border_style="orange3",
            box=box.ROUNDED,
            padding=(0, 1),
            width=_help_panel_width(),
        ),
    ]
    for title, rows in _CLI_MODULE_SECTIONS:
        parts.append(_render_module_section(title, rows))
    _CONSOLE.print(Group(*parts))


def _print_help() -> None:
    _print_cli_help()


def main():
    install_interrupt_handlers()
    if sys.version_info < (3, 10):
        _CONSOLE.print(
            "[bold red]Error:[/bold red] "
            f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported. "
            "JanusX requires Python >= 3.10."
        )
        raise SystemExit(1)
    if str(os.environ.get("JANUSX_ENTRYPOINT", "")).strip() == "":
        prog = Path(sys.argv[0]).name.lower() if len(sys.argv) > 0 else ""
        if prog.startswith("jxpy"):
            os.environ["JANUSX_ENTRYPOINT"] = "jxpy"
    if len(sys.argv) > 1:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            _print_help()
        elif sys.argv[1] == '-v' or sys.argv[1] == '--version':
            _CONSOLE.print(_render_logo())
            _CONSOLE.print(__version__.rstrip())
        elif sys.argv[1] in _LAUNCHER_ONLY_FLAGS:
            _CONSOLE.print(_render_logo())
            launcher_cmd = "jx " + " ".join(sys.argv[1:])
            _CONSOLE.print(
                f"[bold red]Error:[/bold red] `{sys.argv[1]}` is launcher-only and not available in `jxpy`."
            )
            _CONSOLE.print(f"[yellow]Please use launcher command:[/yellow] `{launcher_cmd}`")
        else:
            module_name = sys.argv[1]
            if module_name in _LAUNCHER_ONLY_MODULES:
                _CONSOLE.print(_render_logo())
                launcher_cmd = "jx " + " ".join(sys.argv[1:])
                _CONSOLE.print(
                    f"[bold red]Error:[/bold red] module `{module_name}` is launcher-only and not available in `jxpy`."
                )
                _CONSOLE.print(f"[yellow]Please use launcher command:[/yellow] `{launcher_cmd}`")
                return
            if module_name in _MODULE_NAMES:
                _apply_early_thread_env_from_cli(module_name, sys.argv[2:])
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
                    _CONSOLE.print(_render_logo())
                    _CONSOLE.print(f"[bold red]Error:[/bold red] {msg}")
                    raise SystemExit(1)
                except KeyboardInterrupt:
                    force_exit(130, "Interrupted by user (Ctrl+C).")
            elif module_name not in _MODULE_NAMES:
                _print_unknown_module_error(module_name)
    else:
        _print_help()

if __name__ == "__main__":
    install_interrupt_handlers()
    main()
