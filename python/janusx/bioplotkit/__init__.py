from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

__name__ = "bioplotkit"
__version__ = "1.0"

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "PCSHOW": ("janusx.bioplotkit.pcshow", "PCSHOW"),
    "GWASPLOT": ("janusx.bioplotkit.manhanden", "GWASPLOT"),
    "apply_integer_yticks": ("janusx.bioplotkit.manhanden", "apply_integer_yticks"),
    "LDblock": ("janusx.bioplotkit.LDBlock", "LDblock"),
    "gsplot": ("janusx.bioplotkit.gsplot", None),
    "color_set": ("janusx.bioplotkit.sci_set", "color_set"),
    "marker_set": ("janusx.bioplotkit.sci_set", "marker_set"),
}

__all__ = sorted(_LAZY_EXPORTS.keys())


def _load_export(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(str(name))
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = spec
    mod = import_module(mod_name)
    if attr_name is None:
        value: Any = mod
    else:
        value = getattr(mod, attr_name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    return _load_export(name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
