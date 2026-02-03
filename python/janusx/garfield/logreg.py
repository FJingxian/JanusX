"""AND/NOT logic regression (Rust backend) Python interface.

This module wraps the PyO3-exported function ``janusx.janusx.fit_best_and_not``.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Sequence

import numpy as np

try:
    import numpy as _np  # optional
except Exception:  # pragma: no cover
    _np = None

_backend_error = None
try:
    from janusx.janusx import fit_best_and_not as _fit_best_and_not
except Exception as _exc:  # pragma: no cover
    _backend_error = _exc
    _fit_best_and_not = None
else:
    _backend_error = None


def backend_available() -> bool:
    """Return True if the Rust backend (janusx.janusx) is importable."""
    return _fit_best_and_not is not None


def _require_backend() -> None:
    if _fit_best_and_not is None:
        raise ImportError(
            "janusx.janusx (Rust extension) is not available. "
            "Build/install the extension before calling fit_best_and_not."
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


def logreg(
    x: Any,
    y: Any,
    *,
    response: Literal['binary','continuous'] = "binary",
    max_literals: int = 0,
    allow_empty: bool = True,
    weights: Optional[Any] = None,
    tags: Optional[Sequence[str]] = None,
):
    """Fit the best AND/NOT conjunction.

    Parameters
    - x: binary matrix (list-of-lists or numpy array) shape (n_samples, n_features)
    - y: response vector length n_samples
    - response: "binary" or "continuous"
    - max_literals: maximum number of literals in the conjunction (0 = no limit)
    - allow_empty: allow empty conjunction (always true)
    - weights: optional non-negative sample weights length n_samples

    Returns
    - dict with keys: indices, expression, xcombine, score

    Notes
    - For response="binary", y must be 0/1 and score uses log-likelihood.
    - For response="continuous", score uses mean squared error.
    - xcombine is the predicted conjunction output (0/1) for each sample.
    - tags can be used to render expression with custom feature names. If omitted,
      default tags are X0..X{p-1} based on the number of features.
    """
    _require_backend()
    x_list = _to_u8_matrix(x)
    y_list = _to_f64_vector(y)
    w_list = _to_f64_vector(weights) if weights is not None else None
    if response == 'binary':
        score = 'loglik'
    elif response == 'continuous':
        score = 'mse'
    else:
        raise ValueError(f'{response} is not continuous or binary.')
    result = _fit_best_and_not(
        x_list,
        y_list,
        response=response,
        score=score,
        max_literals=max_literals,
        allow_empty=allow_empty,
        weights=w_list,
    )
    if tags is None:
        if not x_list or not x_list[0]:
            tags = []
        else:
            tags = [f"X{i}" for i in range(len(x_list[0]))]
    else:
        if len(tags) == 0:
            raise ValueError("tags must not be empty when provided")

    literals = result["literals"]
    parts = []
    for idx, negated in literals:
        if idx >= len(tags):
            raise ValueError("tags length must cover all feature indices")
        name = str(tags[idx])
        parts.append(f"!{name}" if negated else name)
    expression = " & ".join(parts) if parts else "1"
    return {
        "indices": result["indices"],
        "expression": expression,
        "xcombine": np.array([int(v) for v in result["xcombine"]]),
        "score": float(result["score"]),
    }
