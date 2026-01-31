import numpy as np
from numpy.typing import NDArray
from typing import Union

def _testX(
    X: Union[NDArray[np.floating], None],
    n: int,
    add_intercept: bool = True,
) -> Union[NDArray[np.float32], None]:
    """
    Ensure X is (n, p) and optionally add an intercept column.
    If any column of X is (approximately) all ones, treat it as intercept and do NOT add another.

    Parameters
    ----------
    X : np.ndarray | None
        Covariates without intercept.
    n : int
        Sample count.
    add_intercept : bool
        Whether to append an intercept if missing.
    """
    if X is None:
        return np.ones((n, 1), dtype=np.float32) if add_intercept else None

    X = np.asarray(X)
    if X.ndim == 1:
        if X.shape[0] != n:
            raise ValueError(f"X length {X.shape[0]} != n {n}")
        X = X.reshape(n, 1)
    elif X.ndim == 2:
        if X.shape[0] != n:
            raise ValueError(f"X.shape[0] {X.shape[0]} != n {n}")
    else:
        raise ValueError("X must be 1D or 2D.")

    X = X.astype(np.float32, copy=False)

    ones = np.ones((n,), dtype=X.dtype)

    # 任意一列是截距(全1) => 不添加
    has_intercept = np.any([np.allclose(X[:, j], ones, rtol=0.0, atol=1e-8) for j in range(X.shape[1])])

    if add_intercept and not has_intercept:
        return np.column_stack([X, ones.reshape(n, 1)])
    return X
    
def _testY(y: NDArray[np.floating]) -> NDArray[np.float32]:
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim == 2:
        if y.shape[1] != 1:
            raise ValueError(f"Y must be (n,) or (n,1), got shape={y.shape}.")
    else:
        raise ValueError(f"Y must be 1D or 2D, got ndim={y.ndim}.")

    if y.size == 0:
        raise ValueError("Y is empty.")
    if np.any(~np.isfinite(y)):
        raise ValueError("Y contains NaN or Inf.")

    return y.astype(np.float32, copy=False)
