"""
pyBLUP public API.

Keep this surface explicit to avoid wildcard pollution and accidental
shadowing by duplicated helper names across modules.

Note:
`janusx.pyBLUP` is maintained as a compatibility layer.
For new code, prefer the thin wrappers under:
  - `janusx.assoc` (GWAS/association)
  - `janusx.gs` (genomic selection)
"""

from .QK2 import QK, GRM
from .assoc import (
    FEM,
    lmm_reml,
    lmm_reml_null,
    fastlmm_reml_null,
    fastlmm_reml,
    fastlmm_assoc_chunk,
    ml_loglike_null,
    lmm_assoc_fixed,
    LMM,
    FastLMM,
    LM,
    farmcpu,
)
from .stream_grm import (
    StreamingGrmStats,
    auto_stream_grm_chunk_size,
    build_streaming_grm_from_chunks,
)
from .kfold import kfold
from .mlm import BLUP as BLUP
from .ml import MLGS

__all__ = [
    "QK",
    "GRM",
    "FEM",
    "lmm_reml",
    "lmm_reml_null",
    "fastlmm_reml_null",
    "fastlmm_reml",
    "fastlmm_assoc_chunk",
    "ml_loglike_null",
    "lmm_assoc_fixed",
    "LMM",
    "FastLMM",
    "LM",
    "farmcpu",
    "StreamingGrmStats",
    "auto_stream_grm_chunk_size",
    "build_streaming_grm_from_chunks",
    "kfold",
    "BLUP",
    "MLGS",
]
