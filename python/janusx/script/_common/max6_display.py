from __future__ import annotations

import math


MAX6_LD_TAG_LOADING_DESC = "Loading Max6 LD tags on the fixed sample pool"
MAX6_BONFERRONI_LOADING_DESC = "Preparing the unique-pair Bonferroni family"


def pwald_bonferroni_reference(
    n_unique_pairs: object,
    alpha: float = 0.05,
) -> dict[str, float | int] | None:
    """Return a one-df, one-test-per-pair Bonferroni reference for pwald."""
    try:
        n_pairs = int(n_unique_pairs)
    except (TypeError, ValueError, OverflowError):
        return None
    if n_pairs <= 0:
        return None

    alpha_value = float(alpha)
    if not (math.isfinite(alpha_value) and 0.0 < alpha_value < 1.0):
        raise ValueError("Bonferroni alpha must be finite and between 0 and 1")

    p_cut = alpha_value / n_pairs
    return {
        "n_unique_pairs": n_pairs,
        "alpha": alpha_value,
        "p_cut": p_cut,
        "minus_log10_p_cut": -math.log10(p_cut),
    }


def format_max6_count(value: object) -> str:
    """Keep large Max6 counts compact in CLI status lines."""
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return "NA"
    if abs(count) < 1_000_000:
        return str(count)

    mantissa, exponent = f"{count:.4e}".split("e", 1)
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"
