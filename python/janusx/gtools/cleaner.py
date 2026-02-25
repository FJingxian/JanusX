"""
General-purpose text/label cleaning helpers.
"""

import re
from typing import Union

import numpy as np
import pandas as pd


def natural_tokens(text: str) -> list[tuple[int, Union[int, str]]]:
    """
    Tokenize text into comparable natural-sort chunks.

    Example
    -------
    "chr12a" -> [(1, "chr"), (0, 12), (1, "a")]
    """
    tokens: list[tuple[int, Union[int, str]]] = []
    for part in re.split(r"(\d+)", text):
        if not part:
            continue
        if part.isdigit():
            tokens.append((0, int(part)))
        else:
            tokens.append((1, part.lower()))
    return tokens


def chrom_sort_key(label: object) -> tuple[int, object]:
    """
    Natural ordering for chromosome labels:
      1) numeric chromosomes (1,2,...)
      2) X/Y/M/MT
      3) other contigs by natural sort
      4) missing values last

    Returns
    -------
    tuple[int, object]
        Comparable key for stable chromosome ordering.
    """
    if pd.isna(label):
        return (3, "")

    if isinstance(label, (int, np.integer)):
        return (0, int(label))

    if isinstance(label, (float, np.floating)) and float(label).is_integer():
        return (0, int(label))

    text = str(label).strip()
    no_chr_prefix = text[3:] if text.lower().startswith("chr") else text
    upper = no_chr_prefix.upper()

    if no_chr_prefix.isdigit():
        return (0, int(no_chr_prefix))

    special_chr = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    if upper in special_chr:
        return (1, special_chr[upper])

    return (2, natural_tokens(no_chr_prefix))


# Backward-compatible aliases (prefer public names without leading underscore).
_natural_tokens = natural_tokens
_chrom_sort_key = chrom_sort_key
