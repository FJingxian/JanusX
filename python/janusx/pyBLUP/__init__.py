from __future__ import annotations

"""
pyBLUP public API.

Keep this surface explicit while avoiding eager imports of heavy optional
dependencies such as SciPy. This matters on macOS because early SciPy/OpenBLAS
imports can pull a foreign OpenMP runtime into the process before JanusX gets a
chance to select its bundled OpenBLAS eigendecomposition backend.
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "QK": ("janusx.pyBLUP.QK2", "QK"),
    "GRM": ("janusx.pyBLUP.QK2", "GRM"),
    "assoc": ("janusx.pyBLUP.assoc", None),
    "FEM": ("janusx.pyBLUP.assoc", "FEM"),
    "lmm_reml": ("janusx.pyBLUP.assoc", "lmm_reml"),
    "lmm_reml_null": ("janusx.pyBLUP.assoc", "lmm_reml_null"),
    "fastlmm_reml_null": ("janusx.pyBLUP.assoc", "fastlmm_reml_null"),
    "fastlmm_reml": ("janusx.pyBLUP.assoc", "fastlmm_reml"),
    "fastlmm_assoc_chunk": ("janusx.pyBLUP.assoc", "fastlmm_assoc_chunk"),
    "ml_loglike_null": ("janusx.pyBLUP.assoc", "ml_loglike_null"),
    "lmm_assoc_fixed": ("janusx.pyBLUP.assoc", "lmm_assoc_fixed"),
    "fvlmm_assoc_fixed": ("janusx.pyBLUP.assoc", "fvlmm_assoc_fixed"),
    "fvlmm_assoc_prepare_cache": ("janusx.pyBLUP.assoc", "fvlmm_assoc_prepare_cache"),
    "fvlmm_assoc_fixed_with_cache": ("janusx.pyBLUP.assoc", "fvlmm_assoc_fixed_with_cache"),
    "LMM": ("janusx.pyBLUP.assoc", "LMM"),
    "FastLMM": ("janusx.pyBLUP.assoc", "FastLMM"),
    "FvLMM": ("janusx.pyBLUP.assoc", "FvLMM"),
    "LM": ("janusx.pyBLUP.assoc", "LM"),
    "farmcpu": ("janusx.pyBLUP.assoc", "farmcpu"),
    "stream_grm": ("janusx.pyBLUP.stream_grm", None),
    "StreamingGrmStats": ("janusx.pyBLUP.stream_grm", "StreamingGrmStats"),
    "auto_stream_grm_chunk_size": ("janusx.pyBLUP.stream_grm", "auto_stream_grm_chunk_size"),
    "build_streaming_grm_from_chunks": ("janusx.pyBLUP.stream_grm", "build_streaming_grm_from_chunks"),
    "kfold": ("janusx.pyBLUP.kfold", "kfold"),
    "mlm": ("janusx.pyBLUP.mlm", None),
    "BLUP": ("janusx.pyBLUP.mlm", "BLUP"),
    "ml": ("janusx.pyBLUP.ml", None),
    "MLGS": ("janusx.pyBLUP.ml", "MLGS"),
    "bayes": ("janusx.pyBLUP.bayes", None),
    "blup": ("janusx.pyBLUP.blup", None),
}

__all__ = sorted(_LAZY_EXPORTS.keys())


def _load_export(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(str(name))
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = spec
    mod = import_module(mod_name)
    value: Any
    if attr_name is None:
        value = mod
    else:
        value = getattr(mod, attr_name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    return _load_export(name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
