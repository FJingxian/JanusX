from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, requires as distribution_requires
import os
import re
import sys
from typing import Any, Sequence


@dataclass(frozen=True)
class OptionalDependencySpec:
    packages: tuple[str, ...]
    extra: str | None = None


_OPTIONAL_DEPENDENCIES: dict[str, OptionalDependencySpec] = {
    "sklearn": OptionalDependencySpec(packages=("scikit-learn",), extra="ml"),
    "xgboost": OptionalDependencySpec(packages=("xgboost",), extra="ml"),
    "statsmodels": OptionalDependencySpec(packages=("statsmodels",), extra="stats"),
}


def optional_requirement_for_extra(
    package_name: str,
    extra: str,
    *,
    fallback: str,
) -> str:
    """Return the installed distribution's pinned requirement for an extra."""
    wanted_package = re.sub(r"[-_.]+", "-", str(package_name).strip().lower())
    wanted_extra = str(extra).strip().lower()
    try:
        requirements = distribution_requires("janusx") or ()
    except PackageNotFoundError:
        requirements = ()
    for raw in requirements:
        payload, separator, marker = str(raw).partition(";")
        match = re.match(r"\s*([A-Za-z0-9_.-]+)", payload)
        if match is None:
            continue
        normalized = re.sub(r"[-_.]+", "-", match.group(1).lower())
        if normalized != wanted_package:
            continue
        extra_matches = bool(
            separator
            and re.search(
                rf"""\bextra\s*==\s*['"]{re.escape(wanted_extra)}['"]""",
                marker,
            )
        )
        if not extra_matches:
            continue
        requirement = payload.strip()
        if requirement:
            return requirement
    return str(fallback).strip()


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


def classify_import_failure(exc: Exception, module_name: str) -> str:
    """Classify an optional import failure without hiding runtime breakage."""
    if isinstance(exc, ModuleNotFoundError):
        missing = str(getattr(exc, "name", None) or "").strip().lower()
        expected = _top_module_name(module_name)
        if missing == expected:
            return "missing"
    return "broken"


def load_optional_native_symbols(
    module: Any,
    names: Sequence[str],
) -> dict[str, Any]:
    """Load extension symbols independently so one missing symbol is isolated."""
    return {name: getattr(module, name, None) for name in names}


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
