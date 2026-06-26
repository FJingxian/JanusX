from .config import AssociationConfig
from ._typing import (
    ArrayLike,
    AssocLayout,
    AssocMatrixLike,
    AssocModelName,
    CovariateLike,
    DenseKinshipLike,
    Float32Matrix,
    Float64Matrix,
    Float64Vector,
    Int64Vector,
    KinshipLike,
    MatrixLike,
    PathLikeStr,
    ResponseLike,
    SparseKinshipLike,
    VectorLike,
)
from .api import ASSOC
from .lm import LinearModel
from .result import AssociationResult
from .runner import (
    prewarm_gwas_runner,
    run_gwas_args,
    run_gwas_argv,
    run_gwas_cli,
    run_gwas_config,
)

AssociationModel = LinearModel


def run_assoc_api_smoke() -> None:
    from .smoke import run_assoc_api_smoke as _impl

    _impl()

__all__ = [
    "ArrayLike",
    "AssociationConfig",
    "AssociationResult",
    "AssociationModel",
    "ASSOC",
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
    "LinearModel",
    "MatrixLike",
    "PathLikeStr",
    "prewarm_gwas_runner",
    "ResponseLike",
    "run_assoc_api_smoke",
    "run_gwas_args",
    "run_gwas_argv",
    "run_gwas_cli",
    "run_gwas_config",
    "SparseKinshipLike",
    "VectorLike",
]
