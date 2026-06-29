from __future__ import annotations

from janusx.adamixture import (
    ADAMixtureConfig,
    evaluate_adamixture_cverror,
    rsvd_streaming,
    train_adamixture,
)


FastPopConfig = ADAMixtureConfig


def train_fastpop(*args, **kwargs):
    return train_adamixture(*args, **kwargs)


def evaluate_fastpop_cverror(*args, **kwargs):
    return evaluate_adamixture_cverror(*args, **kwargs)


__all__ = [
    "FastPopConfig",
    "train_fastpop",
    "rsvd_streaming",
    "evaluate_fastpop_cverror",
]
