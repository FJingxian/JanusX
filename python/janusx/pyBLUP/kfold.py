import numpy as np
def kfold(n:int,k:int=5,seed:int=520):
    """
    Generate k-fold train/test splits.

    Parameters
    ----------
    n : int
        Sample size.
    k : int
        Number of folds.
    seed : int
        Random seed.
    """
    if int(n) < 2:
        raise ValueError(f"n must be >=2, got {n}")
    if int(k) < 2:
        raise ValueError(f"k must be >=2, got {k}")
    if int(k) > int(n):
        raise ValueError(f"k must be <= n, got k={k}, n={n}")

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n, dtype=int)
    shuffled = rng.permutation(all_idx)

    # Balanced fold sizes: first (n % k) folds get one extra sample.
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: (n % k)] += 1

    out: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for fs in fold_sizes:
        end = start + int(fs)
        testrow = shuffled[start:end]
        start = end
        test_mask = np.zeros(n, dtype=bool)
        test_mask[testrow] = True
        trainrow = all_idx[~test_mask]
        out.append((testrow, trainrow))
    return out
