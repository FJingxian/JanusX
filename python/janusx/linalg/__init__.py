from __future__ import annotations

from typing import Any


def eigh(
    a: Any,
    *,
    threads: int = 0,
    driver: str | None = "auto",
    jobz: str = "V",
    require_lapack: bool = False,
    inplace: bool = False,
    return_meta: bool = False,
):
    """
    Rust-backed symmetric eigendecomposition.

    API intentionally mirrors NumPy style:
        eigvals, eigvecs = janusx.linalg.eigh(A)

    Parameters
    ----------
    a:
        Square matrix-like input. Internally coerced to float64.
    threads:
        BLAS thread hint (0 = keep runtime default).
    driver:
        LAPACK driver token (e.g., "auto", "dsyevd", "dsyevr").
    jobz:
        "V" for values+vectors, "N" for values only.
    require_lapack:
        If True, raise when non-LAPACK backend is used.
    inplace:
        If True, may overwrite a contiguous working matrix to reduce copies.
    return_meta:
        If True, return full native tuple with backend/thread/timing metadata.
        Else return `(eigvals, eigvecs)` only.
    """
    import numpy as np
    from .. import janusx as _jx

    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"eigh expects a square 2D matrix, got shape={arr.shape}")

    t = int(max(0, int(threads)))
    kw = dict(
        threads=t,
        driver=driver,
        jobz=jobz,
        require_lapack=bool(require_lapack),
    )

    ret = None
    if bool(inplace) and hasattr(_jx, "rust_eigh_from_array_f64_inplace"):
        # Contiguous working buffer for in-place LAPACK path.
        work = np.array(arr, dtype=np.float64, order="C", copy=False)
        try:
            ret = _jx.rust_eigh_from_array_f64_inplace(work, **kw)
        except Exception:
            # Fallback to non-inplace path for compatibility.
            ret = None

    if ret is None:
        ret = _jx.rust_eigh_from_array_f64(arr, **kw)
    if bool(return_meta):
        return ret
    return ret[0], ret[1]

