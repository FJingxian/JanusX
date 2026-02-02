"""Logic regression (Rust backend) Python interface.

This module wraps the PyO3-exported function ``janusx.janusx.logregfit``.

Quick start:
    import numpy as np
    from janusx.garfield.logreg import logreg_fit

    x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=np.uint8)
    y = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)

    model = logreg_fit(x, y, response="binary", select="anneal", ntrees=1, nleaves=4)
    print(model.expression, model.score)

Notes:
- x must be a binary matrix (0/1). Shape: (n_samples, n_features)
- For response="binary", y must be 0/1; for response="continuous", y is float.
- seps is optional: a list of float covariate vectors (each length n_samples).
- weights is optional: a length n_samples non-negative vector.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

try:
    import numpy as _np  # optional
except Exception:  # pragma: no cover
    _np = None

_backend_error = None
try:
    from janusx.janusx import logregfit as _logregfit
except Exception as _exc:  # pragma: no cover
    _backend_error = _exc
    _logregfit = None
else:
    _backend_error = None

__all__ = ["backend_available", "logregfit", "logreg_fit"]


def backend_available() -> bool:
    """Return True if the Rust backend (janusx.janusx) is importable."""
    return _logregfit is not None


def _require_backend() -> None:
    if _logregfit is None:
        raise ImportError(
            "janusx.janusx (Rust extension) is not available. "
            "Build/install the extension before calling logregfit."
        ) from _backend_error


def _tolist(x: Any) -> Any:
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def _to_u8_matrix(x: Any) -> List[List[int]]:
    mat = _tolist(x)
    if _np is not None and isinstance(mat, _np.ndarray):
        mat = mat.tolist()
    if not isinstance(mat, list):
        raise TypeError("x must be a 2D list or numpy array")
    return mat


def _to_f64_vector(y: Any) -> List[float]:
    vec = _tolist(y)
    if _np is not None and isinstance(vec, _np.ndarray):
        vec = vec.tolist()
    if not isinstance(vec, list):
        raise TypeError("y must be a 1D list or numpy array")
    return vec


def _to_f64_matrix(seps: Optional[Sequence[Any]]) -> Optional[List[List[float]]]:
    if seps is None:
        return None
    out: List[List[float]] = []
    for col in seps:
        out.append(_to_f64_vector(col))
    return out


def logregfit(
    x: Any,
    y: Any,
    *,
    response: str = "binary",
    score: str = "loglik",
    ntrees: int = 1,
    nleaves: int = 8,
    treesize: int = 8,
    select: str = "anneal",
    anneal_start: float = 0.0,
    anneal_end: float = 0.0,
    anneal_iter: int = 0,
    anneal_earlyout: int = 0,
    anneal_update: int = 0,
    minmass: int = 0,
    penalty: float = 0.0,
    seed: int = 42,
    allow_or: bool = True,
    weights: Optional[Any] = None,
    seps: Optional[Sequence[Any]] = None,
):
    """Fit a logic regression model via the Rust backend.

    Parameters
    - x: binary matrix (list-of-lists or numpy array) of shape (n_samples, n_features)
    - y: response vector (list or numpy array) length n_samples
    - response: "binary" or "continuous"
    - score: "loglik" or "mse"
    - ntrees, nleaves, treesize: tree structure parameters
    - select: "anneal", "greedy", or "fast"
    - anneal_*: annealing controls (if select="anneal")
    - minmass: minimum support per literal (binary only)
    - penalty: leaf penalty term
    - seed: RNG seed
    - allow_or: allow OR operators in the tree
    - weights: optional sample weights
    - seps: optional covariates (list of float vectors)

    Returns
    - PyLogicRegModel with fields: expression, score, xcombine, betas, trace
    """
    _require_backend()
    x_list = _to_u8_matrix(x)
    y_list = _to_f64_vector(y)
    w_list = _to_f64_vector(weights) if weights is not None else None
    seps_list = _to_f64_matrix(seps)
    return _logregfit(
        x_list,
        y_list,
        response=response,
        score=score,
        ntrees=ntrees,
        nleaves=nleaves,
        treesize=treesize,
        select=select,
        anneal_start=anneal_start,
        anneal_end=anneal_end,
        anneal_iter=anneal_iter,
        anneal_earlyout=anneal_earlyout,
        anneal_update=anneal_update,
        minmass=minmass,
        penalty=penalty,
        seed=seed,
        allow_or=allow_or,
        weights=w_list,
        seps=seps_list,
    )


def logreg_fit(*args: Any, **kwargs: Any):
    """Alias for logregfit (PEP8-friendly name)."""
    return logregfit(*args, **kwargs)
