from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

BLUP_METHOD = "BLUP"
BLUP_SMALL_N = 15_000
BLUP_SMALL_M = 15_000
BLUP_FORCE_ENV = "GS_BLUP"
BlupForceMode = Literal["gblup", "exact", "pcg"]


@dataclass(frozen=True)
class BlupDispatch:
    requested_method: str
    effective_method: Literal["GBLUP", "rrBLUP"]
    rrblup_solver: Literal["exact", "pcg"] | None
    rrblup_exact_backend: Literal["snp"] | None
    route_key: str
    route_label: str
    n_samples: int
    n_markers: int
    threshold_n: int
    threshold_m: int

    @property
    def requires_resident_packed(self) -> bool:
        return False


def is_blup_method(method: str) -> bool:
    return str(method).strip().upper() == BLUP_METHOD


def resolve_blup_force_mode(raw: str | None = None) -> BlupForceMode | None:
    token_raw = os.environ.get(BLUP_FORCE_ENV) if raw is None else raw
    if token_raw is None:
        return None
    token = str(token_raw).strip()
    if token == "":
        return None
    if token == "0":
        return "gblup"
    if token == "1":
        return "exact"
    if token == "2":
        return "pcg"
    raise ValueError(
        f"Invalid {BLUP_FORCE_ENV}={token!r}; expected 0 (GBLUP), 1 (rrBLUP exact), or 2 (rrBLUP PCG)."
    )


def describe_blup_dispatch_policy(
    *,
    threshold_n: int = BLUP_SMALL_N,
    threshold_m: int = BLUP_SMALL_M,
) -> str:
    force_mode = resolve_blup_force_mode()
    if force_mode == "gblup":
        return f"forced by {BLUP_FORCE_ENV}=0 -> GBLUP"
    if force_mode == "exact":
        return f"forced by {BLUP_FORCE_ENV}=1 -> exact rrBLUP(snp spectral REML)"
    if force_mode == "pcg":
        return f"forced by {BLUP_FORCE_ENV}=2 -> PCG rrBLUP"
    return (
        f"auto dispatch: n<={int(threshold_n)} -> GBLUP; "
        f"n>{int(threshold_n)} and m<={int(threshold_m)} -> exact rrBLUP(snp); "
        f"n>{int(threshold_n)} and m>{int(threshold_m)} -> PCG rrBLUP"
    )


def resolve_blup_dispatch(
    n_samples: int,
    n_markers: int,
    *,
    threshold_n: int = BLUP_SMALL_N,
    threshold_m: int = BLUP_SMALL_M,
) -> BlupDispatch:
    n = int(max(0, int(n_samples)))
    m = int(max(0, int(n_markers)))
    n_cut = int(max(1, int(threshold_n)))
    m_cut = int(max(1, int(threshold_m)))
    force_mode = resolve_blup_force_mode()
    if force_mode == "gblup":
        return BlupDispatch(
            requested_method=BLUP_METHOD,
            effective_method="GBLUP",
            rrblup_solver=None,
            rrblup_exact_backend=None,
            route_key="gblup_forced_env0",
            route_label=f"GBLUP (forced by {BLUP_FORCE_ENV}=0)",
            n_samples=n,
            n_markers=m,
            threshold_n=n_cut,
            threshold_m=m_cut,
        )
    if force_mode == "exact":
        return BlupDispatch(
            requested_method=BLUP_METHOD,
            effective_method="rrBLUP",
            rrblup_solver="exact",
            rrblup_exact_backend="snp",
            route_key="exact_rrblup_snp_forced_env1",
            route_label=f"rrBLUP exact spectral REML (forced by {BLUP_FORCE_ENV}=1)",
            n_samples=n,
            n_markers=m,
            threshold_n=n_cut,
            threshold_m=m_cut,
        )
    if force_mode == "pcg":
        return BlupDispatch(
            requested_method=BLUP_METHOD,
            effective_method="rrBLUP",
            rrblup_solver="pcg",
            rrblup_exact_backend=None,
            route_key="pcg_rrblup_he_forced_env2",
            route_label=f"rrBLUP PCG (forced by {BLUP_FORCE_ENV}=2)",
            n_samples=n,
            n_markers=m,
            threshold_n=n_cut,
            threshold_m=m_cut,
        )
    if n <= n_cut:
        return BlupDispatch(
            requested_method=BLUP_METHOD,
            effective_method="GBLUP",
            rrblup_solver=None,
            rrblup_exact_backend=None,
            route_key="gblup_n_le_threshold",
            route_label="GBLUP",
            n_samples=n,
            n_markers=m,
            threshold_n=n_cut,
            threshold_m=m_cut,
        )

    if m <= m_cut:
        return BlupDispatch(
            requested_method=BLUP_METHOD,
            effective_method="rrBLUP",
            rrblup_solver="exact",
            rrblup_exact_backend="snp",
            route_key="exact_rrblup_snp_n_gt_threshold_m_le_threshold",
            route_label="rrBLUP exact spectral REML",
            n_samples=n,
            n_markers=m,
            threshold_n=n_cut,
            threshold_m=m_cut,
        )

    return BlupDispatch(
        requested_method=BLUP_METHOD,
        effective_method="rrBLUP",
        rrblup_solver="pcg",
        rrblup_exact_backend=None,
        route_key="pcg_rrblup_he_large",
        route_label="rrBLUP PCG",
        n_samples=n,
        n_markers=m,
        threshold_n=n_cut,
        threshold_m=m_cut,
    )
