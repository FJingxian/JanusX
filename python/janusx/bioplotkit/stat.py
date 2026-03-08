from __future__ import annotations

from typing import Dict, Hashable, List, Sequence, Union, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def _normalize_input(
    data: Union[pd.DataFrame, List[Sequence[float]]],
    value_col: Optional[str] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Normalize input into a DataFrame with columns: ['group', 'value'].
    """
    if isinstance(data, pd.DataFrame):
        if value_col is None or group_col is None:
            if data.shape[1] != 2:
                raise ValueError("For DataFrame input, provide value_col and group_col, or use exactly 2 columns.")
            value_col = data.columns[0]
            group_col = data.columns[1]

        df = data[[value_col, group_col]].copy()
        df.columns = ["value", "group"]
        df = df.dropna()
        return df

    if isinstance(data, list):
        rows = []
        for i, arr in enumerate(data):
            arr = np.asarray(arr, dtype=float)
            arr = arr[~np.isnan(arr)]
            for v in arr:
                rows.append((i, v))
        return pd.DataFrame(rows, columns=["group", "value"])

    raise TypeError("Input must be a pandas DataFrame or a list of group-value lists.")


def _tukey_significance_matrix(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Return a symmetric boolean matrix:
    sig.loc[g1, g2] == True means g1 and g2 are significantly different.
    """
    groups = list(pd.unique(df["group"]))
    sig = pd.DataFrame(False, index=groups, columns=groups, dtype=bool)

    tukey = pairwise_tukeyhsd(
        endog=df["value"].values,
        groups=df["group"].values,
        alpha=alpha,
    )

    # statsmodels summary rows:
    # group1, group2, meandiff, p-adj, lower, upper, reject
    summary_data = tukey.summary().data[1:]
    for row in summary_data:
        g1, g2, *_rest, reject = row
        sig.loc[g1, g2] = bool(reject)
        sig.loc[g2, g1] = bool(reject)

    return sig


def _compact_letter_display(
    sig_matrix: pd.DataFrame,
    means: pd.Series,
) -> Dict[object, str]:
    """
    Generate compact letter display from a significance matrix.

    Rule:
    - Groups that are NOT significantly different can share letters.
    - Groups that ARE significantly different cannot share any letter.
    """
    groups = list(means.sort_values(ascending=False).index)

    letter_sets: List[set] = []
    group_letters: Dict[object, set] = {g: set() for g in groups}

    for g in groups:
        placed = False

        for i, members in enumerate(letter_sets):
            # Can join this letter group only if g is NOT significantly different
            # from all existing members in that letter group.
            if all(not sig_matrix.loc[g, m] for m in members):
                members.add(g)
                group_letters[g].add(i)
                placed = True

        if not placed:
            # create a new letter group
            letter_sets.append({g})
            group_letters[g].add(len(letter_sets) - 1)

    # A refinement pass:
    # try to add groups into existing letter groups if allowed,
    # so intermediate groups can have labels like "ab"
    for i, members in enumerate(letter_sets):
        for g in groups:
            if g in members:
                continue
            if all(not sig_matrix.loc[g, m] for m in members):
                members.add(g)
                group_letters[g].add(i)

    letters = "abcdefghijklmnopqrstuvwxyz"
    if len(letter_sets) > len(letters):
        raise ValueError("Too many letter groups (>26). Extend the letter set manually.")

    out = {}
    for g in groups:
        idxs = sorted(group_letters[g])
        out[g] = "".join(letters[i] for i in idxs)

    return out


def multiple_comparison_letters(
    data: Union[pd.DataFrame, List[Sequence[float]]],
    value_col: Optional[str] = None,
    group_col: Optional[str] = None,
    alpha: float = 0.05,
    omnibus: str = "anova",
    posthoc: str = "tukey",
    check_omnibus: bool = False,
) -> Dict[object, str]:
    """
    Perform multiple comparison and return compact letter display.

    Parameters
    ----------
    data
        Either:
        1) pandas DataFrame with two columns: value and group
        2) list of lists: [[group1_values], [group2_values], ...]
    value_col
        Name of numeric column for DataFrame input.
    group_col
        Name of group column for DataFrame input.
    alpha
        Significance threshold.
    omnibus
        Currently supports:
        - 'anova'
        - 'kruskal'
    posthoc
        Currently supports:
        - 'tukey'
    check_omnibus
        If True, first run omnibus test; if not significant, all groups get 'a'.

    Returns
    -------
    Dict[object, str]
        Compact letter display, e.g.
        {'A': 'a', 'B': 'ab', 'C': 'b'}
        or
        {0: 'a', 1: 'ab', 2: 'b'}
    """
    df = _normalize_input(data, value_col=value_col, group_col=group_col)

    if df.empty:
        raise ValueError("No valid data after removing missing values.")

    groups = list(pd.unique(df["group"]))
    grouped = [df.loc[df["group"] == g, "value"].values for g in groups]

    if len(groups) < 2:
        return {groups[0]: "a"} if groups else {}

    if check_omnibus:
        if omnibus == "anova":
            stat, p = stats.f_oneway(*grouped)
        elif omnibus == "kruskal":
            stat, p = stats.kruskal(*grouped)
        else:
            raise ValueError("Unsupported omnibus test. Use 'anova' or 'kruskal'.")

        if p >= alpha:
            return {g: "a" for g in groups}

    if posthoc != "tukey":
        raise ValueError("Currently only posthoc='tukey' is implemented.")

    sig = _tukey_significance_matrix(df, alpha=alpha)
    means = df.groupby("group")["value"].mean()

    return _compact_letter_display(sig, means)


def multiple_comparison_groupby(
    data: pd.DataFrame,
    value_col: Hashable,
    group_cols: Union[Hashable, Sequence[Hashable]],
    alpha: float = 0.05,
    omnibus: str = "anova",
    posthoc: str = "tukey",
    check_omnibus: bool = False,
) -> pd.DataFrame:
    """
    Multiple comparison for groupby-like inputs.

    Parameters
    ----------
    data
        Source DataFrame that contains one numeric value column and one/multiple group columns.
    value_col
        Numeric phenotype column.
    group_cols
        A single group column or a sequence of group columns.

    Returns
    -------
    pd.DataFrame
        Index = group key (single value or tuple),
        Columns = ['mean', 'n', 'letters'].
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")

    if isinstance(group_cols, (str, int, float, tuple)):
        group_cols = [group_cols]
    else:
        group_cols = list(group_cols)

    if len(group_cols) == 0:
        raise ValueError("`group_cols` cannot be empty.")

    required_cols = [value_col, *group_cols]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in input DataFrame: {missing}")

    df = data[required_cols].copy().dropna()
    if df.empty:
        raise ValueError("No valid rows after dropping NA values.")

    if len(group_cols) == 1:
        df["_group_key"] = df[group_cols[0]]
    else:
        df["_group_key"] = list(df[group_cols].itertuples(index=False, name=None))

    # Special case: exactly two groups -> fallback to Welch's t-test.
    unique_groups = list(pd.unique(df["_group_key"]))
    if len(unique_groups) == 2:
        g1, g2 = unique_groups
        x1 = df.loc[df["_group_key"] == g1, value_col].astype(float).values
        x2 = df.loc[df["_group_key"] == g2, value_col].astype(float).values
        _stat, pvalue = stats.ttest_ind(x1, x2, equal_var=False, nan_policy="omit")

        means = df.groupby("_group_key")[value_col].mean()
        counts = df.groupby("_group_key")[value_col].size()
        out = pd.DataFrame({"mean": means, "n": counts})

        if np.isnan(pvalue) or pvalue >= alpha:
            letters = {g1: "a", g2: "a"}
        else:
            ranked = means.sort_values(ascending=False).index.tolist()
            letters = {ranked[0]: "a", ranked[1]: "b"}

        out["letters"] = [letters.get(g, "") for g in out.index]
        out = out.sort_values("mean", ascending=False)
        out.attrs["pvalue"] = float(pvalue) if np.isfinite(pvalue) else np.nan
        return out

    # statsmodels Tukey expects 1D group labels; tuple-like keys can be
    # expanded into 2D arrays internally. Factorize to stable integer labels.
    group_labels, group_uniques = pd.factorize(df["_group_key"], sort=False)
    mc_df = pd.DataFrame({"value": df[value_col].values, "group": group_labels})
    letters_by_label = multiple_comparison_letters(
        mc_df,
        value_col="value",
        group_col="group",
        alpha=alpha,
        omnibus=omnibus,
        posthoc=posthoc,
        check_omnibus=check_omnibus,
    )
    letters = {
        group_uniques[int(k)]: v
        for k, v in letters_by_label.items()
        if int(k) < len(group_uniques)
    }

    means = df.groupby("_group_key")[value_col].mean()
    counts = df.groupby("_group_key")[value_col].size()

    out = pd.DataFrame({"mean": means, "n": counts})
    out["letters"] = [letters.get(g, "") for g in out.index]
    out = out.sort_values("mean", ascending=False)
    return out
