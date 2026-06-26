"""
Public typing aliases for ``janusx.assoc``.

These aliases intentionally follow a sklearn-style contract:

- public APIs accept broad ``array-like`` inputs,
- pandas objects are accepted when label alignment is useful,
- runtime validation is still responsible for enforcing shape and finiteness,
- internal kernels normalize numeric data to contiguous floating arrays.
"""

from __future__ import annotations

import os
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

try:
    from scipy.sparse import spmatrix as SparseMatrix
except Exception:  # pragma: no cover - scipy is a project dependency
    SparseMatrix = Any


PathLikeStr: TypeAlias = str | os.PathLike[str]
ArrayLike: TypeAlias = npt.ArrayLike
VectorLike: TypeAlias = pd.Series | pd.DataFrame | ArrayLike
MatrixLike: TypeAlias = pd.DataFrame | ArrayLike
ResponseLike: TypeAlias = VectorLike
CovariateLike: TypeAlias = pd.Series | MatrixLike
DenseKinshipLike: TypeAlias = MatrixLike | PathLikeStr
SparseKinshipLike: TypeAlias = SparseMatrix | PathLikeStr
KinshipLike: TypeAlias = DenseKinshipLike | SparseKinshipLike
AssocMatrixLike: TypeAlias = pd.Series | MatrixLike

AssocLayout: TypeAlias = Literal["sample_major", "snp_major"]
AssocModelName: TypeAlias = Literal["glm", "lm", "lmm", "fastlmm", "fvlmm", "splmm", "farmcpu"]

Float64Vector: TypeAlias = npt.NDArray[np.float64]
Float64Matrix: TypeAlias = npt.NDArray[np.float64]
Float32Matrix: TypeAlias = npt.NDArray[np.float32]
Int64Vector: TypeAlias = npt.NDArray[np.int64]


__all__ = [
    "ArrayLike",
    "AssocLayout",
    "AssocMatrixLike",
    "AssocModelName",
    "CovariateLike",
    "DenseKinshipLike",
    "Float32Matrix",
    "Float64Matrix",
    "Float64Vector",
    "Int64Vector",
    "KinshipLike",
    "MatrixLike",
    "PathLikeStr",
    "ResponseLike",
    "SparseKinshipLike",
    "VectorLike",
]
