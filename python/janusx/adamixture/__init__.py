from __future__ import annotations

from .core import (
    ADAMixtureConfig,
    evaluate_adamixture_cverror as _evaluate_adamixture_cverror,
    rsvd_streaming,
    train_adamixture,
)


FastPopConfig = ADAMixtureConfig
evaluate_adamixture_cverror = _evaluate_adamixture_cverror
# Backward-compatible typo alias kept because older JanusX code imported it.
evaluate_admixture_cverror = _evaluate_adamixture_cverror


def train_fastpop(*args, **kwargs):
    return train_adamixture(*args, **kwargs)


def evaluate_fastpop_cverror(*args, **kwargs):
    return _evaluate_adamixture_cverror(*args, **kwargs)


__all__ = [
    "ADAMixtureConfig",
    "FastPopConfig",
    "train_adamixture",
    "train_fastpop",
    "rsvd_streaming",
    "evaluate_adamixture_cverror",
    "evaluate_admixture_cverror",
    "evaluate_fastpop_cverror",
]
