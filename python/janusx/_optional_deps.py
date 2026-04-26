from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Sequence


@dataclass(frozen=True)
class OptionalDependencySpec:
    packages: tuple[str, ...]
    extra: str | None = None


_OPTIONAL_DEPENDENCIES: dict[str, OptionalDependencySpec] = {
    "sklearn": OptionalDependencySpec(packages=("scikit-learn",), extra="ml"),
    "xgboost": OptionalDependencySpec(packages=("xgboost",), extra="ml"),
    "dynamictreecut": OptionalDependencySpec(packages=("dynamicTreeCut",), extra="wgcna"),
    "statsmodels": OptionalDependencySpec(packages=("statsmodels",), extra="stats"),
    "pypardiso": OptionalDependencySpec(packages=("pypardiso", "mkl"), extra="sparse"),
    "mkl": OptionalDependencySpec(packages=("mkl", "pypardiso"), extra="sparse"),
    "toytree": OptionalDependencySpec(packages=("toytree", "toyplot"), extra="treeplot"),
    "toyplot": OptionalDependencySpec(packages=("toyplot", "toytree"), extra="treeplot"),
}


def _normalize_package_list(packages: Sequence[str]) -> list[str]:
    out: list[str] = []
    for item in packages:
        val = str(item).strip()
        if val == "":
            continue
        out.append(val)
    return out


def _top_module_name(module_name: str | None) -> str:
    raw = str(module_name or "").strip()
    if raw == "":
        return ""
    return raw.split(".", 1)[0].strip().lower()


def resolve_optional_dependency(module_name: str | None) -> OptionalDependencySpec | None:
    return _OPTIONAL_DEPENDENCIES.get(_top_module_name(module_name))


def _python_executable_hint() -> str:
    exe = str(sys.executable or "").strip()
    if exe == "":
        return "python"
    if " " in exe:
        return f'"{exe}"'
    return exe


def format_install_hint(
    packages: Sequence[str],
    *,
    extra: str | None = None,
    entrypoint: str | None = None,
) -> str:
    package_list = _normalize_package_list(packages)
    if len(package_list) == 0:
        return ""

    pkg_joined = " ".join(package_list)
    pip_cmd = f"python -m pip install {pkg_joined}"
    pip_extra_cmd = f'python -m pip install "janusx[{extra}]"' if extra else None
    runtime_pip_cmd = f"{_python_executable_hint()} -m pip install {pkg_joined}"
    runtime_extra_cmd = (
        f'{_python_executable_hint()} -m pip install "janusx[{extra}]"'
        if extra
        else None
    )
    mode = str(entrypoint or "").strip().lower()
    if mode == "":
        mode = detect_entrypoint()

    if mode == "jx":
        if extra:
            return (
                f"Install in launcher runtime with `{runtime_pip_cmd}` "
                f"(or `{runtime_extra_cmd}`)."
            )
        return f"Install in launcher runtime with `{runtime_pip_cmd}`."

    if mode == "jxpy":
        if extra:
            return f"Install with `{pip_cmd}` (or `{pip_extra_cmd}`)."
        return f"Install with `{pip_cmd}`."

    if extra:
        return f"Install with `{pip_cmd}` (or `{pip_extra_cmd}`)."
    return f"Install with `{pip_cmd}`."


def detect_entrypoint() -> str:
    explicit = str(os.environ.get("JANUSX_ENTRYPOINT", "")).strip().lower()
    if explicit in {"jx", "jxpy"}:
        return explicit

    argv0 = str(sys.argv[0] if len(sys.argv) > 0 else "").strip().lower()
    prog = os.path.basename(argv0)
    if prog.startswith("jxpy"):
        return "jxpy"
    if prog == "jx":
        return "jx"
    return "unknown"


def format_missing_dependency_message(
    requirement: str,
    *,
    packages: Sequence[str],
    extra: str | None = None,
    original_error: Exception | None = None,
    entrypoint: str | None = None,
) -> str:
    req = str(requirement).strip()
    if req == "":
        req = "Missing optional dependency."

    hint = format_install_hint(packages, extra=extra, entrypoint=entrypoint)
    if hint == "":
        hint = "Please install the required package(s)."

    if original_error is None:
        return f"{req} {hint}"
    return f"{req} {hint} Original import error: {original_error}"


def format_missing_dependency_for_module(
    module_name: str | None,
    requirement: str,
    *,
    original_error: Exception | None = None,
    entrypoint: str | None = None,
) -> str | None:
    spec = resolve_optional_dependency(module_name)
    if spec is None:
        return None
    return format_missing_dependency_message(
        requirement,
        packages=spec.packages,
        extra=spec.extra,
        original_error=original_error,
        entrypoint=entrypoint,
    )
