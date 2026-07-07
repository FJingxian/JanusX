from __future__ import annotations

import numpy as np


def _normalize_n_samples(x: object) -> int:
    if isinstance(x, (int, np.integer)):
        n = int(x)
    else:
        arr = np.asarray(x)
        if arr.ndim <= 0:
            raise ValueError("KFold input must be an integer or array-like with length.")
        n = int(arr.shape[0])
    if n < 2:
        raise ValueError(f"n must be >=2, got {n}")
    return n


def _balanced_fold_sizes(n_samples: int, n_splits: int) -> np.ndarray:
    n = int(n_samples)
    k = int(n_splits)
    if k < 2:
        raise ValueError(f"n_splits must be >=2, got {k}")
    if k > n:
        raise ValueError(f"n_splits must be <= n_samples, got n_splits={k}, n_samples={n}")
    fold_sizes = np.full(k, n // k, dtype=np.int64)
    fold_sizes[: (n % k)] += 1
    return fold_sizes


class KFold:
    """
    Lightweight sklearn-style KFold splitter.

    Notes
    -----
    - `split(...)` yields `(train_idx, test_idx)` like sklearn.
    - `shuffle=True` uses a NumPy Generator permutation.
    - `random_state=None` keeps stochastic behavior; otherwise it is reproducible.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = None if random_state is None else int(random_state)
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >=2, got {n_splits}")
        if (not self.shuffle) and (self.random_state is not None):
            raise ValueError(
                "Setting random_state has no effect when shuffle is False."
            )
        if (self.random_state is not None) and (self.random_state < 0):
            raise ValueError(f"random_state must be >=0, got {self.random_state}")

    def split(
        self,
        X: object,
        y: object = None,
        groups: object = None,
    ):
        del y, groups
        n_samples = _normalize_n_samples(X)
        indices = np.arange(n_samples, dtype=np.int64)
        if self.shuffle:
            rng = (
                np.random.default_rng()
                if self.random_state is None
                else np.random.default_rng(int(self.random_state))
            )
            indices = np.asarray(rng.permutation(indices), dtype=np.int64)

        fold_sizes = _balanced_fold_sizes(n_samples, self.n_splits)
        current = 0
        all_idx = np.arange(n_samples, dtype=np.int64)
        for fold_size in fold_sizes:
            start, stop = current, current + int(fold_size)
            test_idx = np.asarray(indices[start:stop], dtype=np.int64)
            current = stop
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[test_idx] = True
            train_idx = np.asarray(all_idx[~test_mask], dtype=np.int64)
            yield train_idx, test_idx


def kfold(
    n: int,
    k: int = 5,
    *,
    seed: int | None = 520,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate k-fold train/test splits in legacy JanusX order.

    Parameters
    ----------
    n : int
        Sample size.
    k : int
        Number of folds.
    seed : int | None
        Random seed used when `shuffle=True`.
    shuffle : bool
        Whether to shuffle samples before splitting.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Fold list in `(test_idx, train_idx)` order for historical callers.
    """
    splitter = KFold(
        n_splits=int(k),
        shuffle=bool(shuffle),
        random_state=seed,
    )
    return [
        (
            np.asarray(test_idx, dtype=np.int64),
            np.asarray(train_idx, dtype=np.int64),
        )
        for train_idx, test_idx in splitter.split(int(n))
    ]


__all__ = ["KFold", "kfold"]
