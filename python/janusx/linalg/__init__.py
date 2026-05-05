from __future__ import annotations

from typing import Any

__all__ = ["eigh"]


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
    Compute eigenvalues and eigenvectors of a real symmetric matrix.

    This function is backed by JanusX native Rust/LAPACK kernels and is designed
    to be API-compatible with NumPy/SciPy-style usage:
    ``eigvals, eigvecs = jx.linalg.eigh(A)``.

    Input is coerced to ``float64`` internally. If ``inplace=True``, JanusX may
    overwrite a contiguous working buffer to reduce extra copies.

    Parameters
    ----------
    a : array_like, shape (n, n)
        Real symmetric matrix. Non-``float64`` inputs are converted to
        ``float64``.
    threads : int, default=0
        BLAS thread hint.
        ``0`` means keep the current runtime thread setting.
    driver : {"auto", "dsyevd", "dsyevr"}, optional
        Preferred LAPACK eigensolver driver. ``"auto"`` uses JanusX default.
    jobz : {"V", "N"}, default="V"
        ``"V"`` computes eigenvalues and eigenvectors.
        ``"N"`` computes eigenvalues only.
    require_lapack : bool, default=False
        If ``True``, raise an error when LAPACK backend is not used.
    inplace : bool, default=False
        If ``True``, try the in-place native path to reduce copy overhead.
        This may mutate the working matrix buffer.
    return_meta : bool, default=False
        If ``False`` (default), return NumPy-like output:
        ``(eigvals, eigvecs)``.
        If ``True``, return full native diagnostics tuple:
        ``(eigvals, eigvecs, blas_backend, evd_backend, n,``
        ``threads_before, threads_in_stage, threads_after, lapack_used, elapsed_seconds)``.

    Returns
    -------
    eigvals : ndarray of shape (n,)
        Eigenvalues in ascending order.
    eigvecs : ndarray of shape (n, n) or None
        Column ``eigvecs[:, i]`` is the normalized eigenvector for
        ``eigvals[i]``. Returns ``None`` when ``jobz="N"``.
    or tuple
        When ``return_meta=True``, returns the full metadata tuple described
        above.

    Raises
    ------
    ValueError
        If input is not a square 2D matrix.
    RuntimeError
        If native eigendecomposition fails.

    Notes
    -----
    - For best performance, pass a contiguous ``float64`` matrix.
    - ``inplace=True`` typically reduces extra copy overhead in large problems.
    - Numerical behavior follows LAPACK backend semantics.

    Examples
    --------
    Basic decomposition
    >>> import numpy as np
    >>> import janusx as jx
    >>> A = np.array([[2.0, 1.0], [1.0, 3.0]])
    >>> w, v = jx.linalg.eigh(A)
    >>> np.allclose(A @ v, v * w)
    True

    Eigenvalues only
    >>> w, v = jx.linalg.eigh(A, jobz="N")
    >>> v is None
    True

    With backend/timing diagnostics
    >>> ret = jx.linalg.eigh(A, return_meta=True)
    >>> len(ret)
    10
    >>> ret[3]  # evd backend label, e.g. 'lapack_dsyevd'
    'lapack_dsyevd'
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
