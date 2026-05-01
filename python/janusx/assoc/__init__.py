from .config import AssociationConfig
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

__all__ = [
    "AssociationConfig",
    "AssociationResult",
    "AssociationModel",
    "LinearModel",
    "prewarm_gwas_runner",
    "run_gwas_args",
    "run_gwas_argv",
    "run_gwas_cli",
    "run_gwas_config",
]
