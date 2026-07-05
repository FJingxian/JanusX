from __future__ import annotations

import concurrent.futures as cf
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import sparse as sp
except Exception:  # pragma: no cover - scipy is expected in normal installs
    sp = None

from janusx import janusx as _jxrs
from janusx.pyBLUP.assoc import FvLMM, LM, LMM
from janusx.script._common.grmio import (
    load_grm_matrix,
    read_id_file,
    resolve_grm_id_path,
)

from ._typing import (
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
    PathLikeStr,
    ResponseLike,
)
from .workflow_model_packed import (
    _splmm_normalize_sparse_grm_path,
    _splmm_sparse_null_fit,
)

try:
    from janusx.janusx import spgrm_dense_f32_to_jxgrm as _spgrm_dense_f32_to_jxgrm
except Exception:  # pragma: no cover - extension should normally provide this
    _spgrm_dense_f32_to_jxgrm = None

try:
    from janusx.janusx import spgrm_dense_npy_to_jxgrm as _spgrm_dense_npy_to_jxgrm
except Exception:  # pragma: no cover
    _spgrm_dense_npy_to_jxgrm = None

try:
    from janusx.janusx import splmm_assoc_pcg_dense_f32 as _splmm_assoc_pcg_dense_f32
except Exception:  # pragma: no cover
    _splmm_assoc_pcg_dense_f32 = None


_MODEL_ALIASES = {
    "glm": "lm",
    "lm": "lm",
    "lmm": "lmm",
    "fvlmm": "fvlmm",
    "splmm": "splmm",
    "farmcpu": "farmcpu",
}


def _null_fit_record(
    *,
    model: str,
    route: str,
    backend: str,
    converged: bool,
    lbd: float | None = None,
    log10_lbd: float | None = None,
    pve: float | None = None,
    ml0: float | None = None,
    reml0: float | None = None,
    va: float | None = None,
    ve: float | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": str(model),
        "route": str(route),
        "backend": str(backend),
        "converged": bool(converged),
        "lambda": None if lbd is None or not np.isfinite(lbd) else float(lbd),
        "log10_lambda": None if log10_lbd is None or not np.isfinite(log10_lbd) else float(log10_lbd),
        "pve": None if pve is None or not np.isfinite(pve) else float(pve),
        "ml0": None if ml0 is None or not np.isfinite(ml0) else float(ml0),
        "reml0": None if reml0 is None or not np.isfinite(reml0) else float(reml0),
        "va": None if va is None or not np.isfinite(va) else float(va),
        "ve": None if ve is None or not np.isfinite(ve) else float(ve),
    }
    if extras:
        out.update(dict(extras))
    return out


def _dense_backend_null_fit(route: str, backend_model: Any) -> dict[str, Any]:
    lbd = getattr(backend_model, "lbd_null", None)
    pve = getattr(backend_model, "pve", None)
    ml0 = getattr(backend_model, "ML0", None)
    reml0 = getattr(backend_model, "LL0", None)
    log10_lbd: float | None = None
    try:
        if lbd is not None and np.isfinite(float(lbd)) and float(lbd) > 0.0:
            log10_lbd = float(np.log10(float(lbd)))
    except Exception:
        log10_lbd = None
    return _null_fit_record(
        model=str(route),
        route=str(route),
        backend=type(backend_model).__name__,
        converged=True,
        lbd=None if lbd is None else float(lbd),
        log10_lbd=log10_lbd,
        pve=None if pve is None else float(pve),
        ml0=None if ml0 is None else float(ml0),
        reml0=None if reml0 is None else float(reml0),
    )


def _normalize_model_name(model: str) -> str:
    key = str(model).strip().lower()
    if key == "":
        key = "lmm"
    out = _MODEL_ALIASES.get(key)
    if out is None:
        raise ValueError(
            "Unsupported ASSOC model. Expected one of: "
            "glm/lm, lmm, fvlmm, splmm, farmcpu."
        )
    return out


def _ensure_unique_index(index: pd.Index, *, label: str) -> None:
    if not index.is_unique:
        raise ValueError(f"{label} index must be unique for automatic alignment.")


def _sample_ids_to_strings(index: pd.Index | None) -> list[str] | None:
    if index is None:
        return None
    return [str(x) for x in index.tolist()]


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="replace")
    return value


def _ordered_labels(index: pd.Index | None) -> list[Any] | None:
    if index is None:
        return None
    return [_to_python_scalar(v) for v in index.tolist()]


def _index_covers_all(container: pd.Index, target: pd.Index) -> bool:
    seen: dict[Any, bool] = {}
    for raw in container.tolist():
        seen[_to_python_scalar(raw)] = True
    for raw in target.tolist():
        if _to_python_scalar(raw) not in seen:
            return False
    return True


def _normalize_positive_int(value: Any, *, default: int) -> int:
    try:
        iv = int(value)
    except Exception:
        return int(default)
    return int(default) if iv <= 0 else iv


def _clone_backend_model_for_assoc(model: Any) -> Any:
    if isinstance(model, LM):
        x_cov = None
        if int(model.X.shape[1]) > 1:
            x_cov = np.ascontiguousarray(np.asarray(model.X[:, 1:], dtype=np.float64), dtype=np.float64)
        return LM(np.asarray(model.y, dtype=np.float64).reshape(-1), X=x_cov)
    if isinstance(model, (LMM, FvLMM)):
        return type(model).from_lmm(model)
    raise TypeError(f"Unsupported backend model for clone: {type(model).__name__}")


def _coerce_response(y: ResponseLike) -> tuple[Float64Vector, pd.Index | None]:
    if isinstance(y, pd.Series):
        _ensure_unique_index(y.index, label="phenotype")
        arr = np.asarray(y.to_numpy(dtype=np.float64, copy=False), dtype=np.float64).reshape(-1)
        return np.ascontiguousarray(arr, dtype=np.float64), y.index.copy()
    if isinstance(y, pd.DataFrame):
        if int(y.shape[1]) != 1:
            raise ValueError("Phenotype DataFrame must contain exactly one column.")
        _ensure_unique_index(y.index, label="phenotype")
        arr = np.asarray(y.iloc[:, 0].to_numpy(dtype=np.float64, copy=False), dtype=np.float64).reshape(-1)
        return np.ascontiguousarray(arr, dtype=np.float64), y.index.copy()
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 2:
        if int(arr.shape[1]) != 1:
            raise ValueError("Phenotype array must be 1D or a single-column 2D array.")
        arr = arr.reshape(-1)
    elif arr.ndim != 1:
        raise ValueError("Phenotype input must be 1D.")
    return np.ascontiguousarray(arr, dtype=np.float64), None


def _coerce_covariates(
    X: CovariateLike | None,
    *,
    sample_index: pd.Index | None,
    n_samples: int,
) -> Float64Matrix | None:
    if X is None:
        return None
    if isinstance(X, pd.Series):
        _ensure_unique_index(X.index, label="covariate")
        X = X.to_frame()
    if isinstance(X, pd.DataFrame):
        _ensure_unique_index(X.index, label="covariate")
        X_use = X.loc[_ordered_labels(sample_index)] if sample_index is not None else X
        arr = np.asarray(X_use.to_numpy(dtype=np.float64, copy=False), dtype=np.float64)
    else:
        arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Covariates must be a 2D matrix.")
    if int(arr.shape[0]) != int(n_samples):
        raise ValueError(
            f"Covariate row count mismatch: got {arr.shape[0]}, expected {n_samples}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Covariates contain NaN/Inf values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _positions_from_ids(
    target_ids: list[str],
    source_ids: list[str],
    *,
    label: str,
) -> Int64Vector:
    pos: dict[str, int] = {}
    for idx, raw in enumerate(source_ids):
        key = str(raw)
        if key in pos:
            raise ValueError(f"{label} IDs must be unique.")
        pos[key] = idx
    missing = [sid for sid in target_ids if sid not in pos]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{label} is missing {len(missing)} sample IDs: {preview}")
    return np.ascontiguousarray([pos[sid] for sid in target_ids], dtype=np.int64)


def _load_dense_kinship_from_path(
    path: PathLikeStr,
    *,
    sample_index: pd.Index | None,
    n_samples: int,
) -> Float64Matrix:
    raw = str(path)
    low = raw.lower()
    arr: np.ndarray
    if low.endswith(".npy"):
        arr = np.load(raw, mmap_mode="r")
    else:
        arr = np.ascontiguousarray(load_grm_matrix(raw), dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[0]) != int(arr.shape[1]):
        raise ValueError(f"Dense kinship must be square, got shape={arr.shape}.")
    id_path = resolve_grm_id_path(raw, None)
    if id_path is not None and sample_index is not None:
        target_ids = _sample_ids_to_strings(sample_index) or []
        source_ids = [str(x) for x in read_id_file(id_path)]
        if len(source_ids) != int(arr.shape[0]):
            raise ValueError(
                f"GRM ID file length mismatch: len(id)={len(source_ids)}, matrix_n={arr.shape[0]}."
            )
        if target_ids != source_ids:
            take = _positions_from_ids(target_ids, source_ids, label="GRM")
            arr = np.ascontiguousarray(np.asarray(arr[np.ix_(take, take)], dtype=np.float64), dtype=np.float64)
    elif int(arr.shape[0]) != int(n_samples):
        raise ValueError(
            "Dense kinship size does not match phenotype. "
            "Provide a matching GRM or a sibling `.id` file for automatic reordering."
        )
    return arr


def _coerce_dense_kinship(
    k: DenseKinshipLike,
    *,
    sample_index: pd.Index | None,
    n_samples: int,
) -> Float64Matrix:
    path = _maybe_path(k)
    if path is not None:
        return _load_dense_kinship_from_path(path, sample_index=sample_index, n_samples=n_samples)
    if isinstance(k, pd.DataFrame):
        _ensure_unique_index(k.index, label="kinship")
        _ensure_unique_index(k.columns, label="kinship")
        if sample_index is not None:
            labels = _ordered_labels(sample_index)
            k_use = k.loc[labels, labels]
        else:
            k_use = k
        arr = np.asarray(k_use.to_numpy(dtype=np.float64, copy=False), dtype=np.float64)
    else:
        arr = np.asarray(k)
    if arr.ndim != 2 or int(arr.shape[0]) != int(arr.shape[1]):
        raise ValueError(f"Kinship must be a square matrix, got shape={arr.shape}.")
    if int(arr.shape[0]) != int(n_samples):
        raise ValueError(
            f"Kinship size mismatch: got {arr.shape}, expected ({n_samples}, {n_samples})."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Kinship contains NaN/Inf values.")
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float64), dtype=np.float64)


def _maybe_path(value: Any) -> str | None:
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    return None


def _sparse_grm_n_samples(path: str) -> int:
    if not hasattr(_jxrs, "splmm_sparse_grm_diag_stats"):
        raise RuntimeError("Rust extension missing splmm_sparse_grm_diag_stats.")
    _mean_diag, _min_diag, _max_diag, n_samples, _nnz = _jxrs.splmm_sparse_grm_diag_stats(
        str(path),
        None,
    )
    return int(n_samples)


def _write_id_sidecar(grm_path: str, sample_index: pd.Index | None) -> None:
    if sample_index is None:
        return
    with open(f"{grm_path}.id", "w", encoding="utf-8") as fh:
        for raw in sample_index.tolist():
            fh.write(f"{raw}\n")


def _write_sparse_grm_from_scipy(matrix: Any, out_path: str) -> str:
    if sp is None or not sp.issparse(matrix):
        raise TypeError("Expected a scipy sparse matrix.")
    mat = matrix.tocsc(copy=False)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Sparse kinship must be square, got shape={mat.shape}.")
    if not np.all(np.isfinite(mat.data)):
        raise ValueError("Sparse kinship contains NaN/Inf values.")
    mat = ((mat + mat.T) * 0.5).tocsc()
    mat.sum_duplicates()
    n = int(mat.shape[0])
    col_ptr = np.zeros(n + 1, dtype=np.uint64)
    row_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    for col in range(n):
        start = int(mat.indptr[col])
        end = int(mat.indptr[col + 1])
        rows = np.asarray(mat.indices[start:end], dtype=np.int64)
        vals = np.asarray(mat.data[start:end], dtype=np.float64)
        keep = rows >= col
        rows = rows[keep]
        vals = vals[keep]
        if rows.size > 0:
            order = np.argsort(rows, kind="mergesort")
            rows = rows[order]
            vals = vals[order]
        diag_pos = int(np.searchsorted(rows, col))
        if diag_pos >= int(rows.size) or int(rows[diag_pos]) != int(col):
            rows = np.insert(rows, diag_pos, int(col))
            vals = np.insert(vals, diag_pos, 0.0)
        row_chunks.append(np.asarray(rows, dtype=np.uint32))
        value_chunks.append(np.asarray(vals, dtype=np.float64))
        col_ptr[col + 1] = col_ptr[col] + int(rows.size)
    row_idx = (
        np.asarray(np.concatenate(row_chunks), dtype=np.uint32)
        if row_chunks
        else np.zeros((0,), dtype=np.uint32)
    )
    values = (
        np.asarray(np.concatenate(value_chunks), dtype=np.float64)
        if value_chunks
        else np.zeros((0,), dtype=np.float64)
    )
    nnz = int(values.shape[0])
    row_bytes = int(row_idx.nbytes)
    pad = (8 - (row_bytes % 8)) % 8
    with open(out_path, "wb") as fh:
        fh.write(np.asarray([n], dtype="<u8").tobytes(order="C"))
        fh.write(np.asarray([nnz], dtype="<u8").tobytes(order="C"))
        fh.write(np.asarray(col_ptr, dtype="<u8").tobytes(order="C"))
        fh.write(np.asarray(row_idx, dtype="<u4").tobytes(order="C"))
        if pad:
            fh.write(b"\x00" * pad)
        fh.write(np.asarray(values, dtype="<f8").tobytes(order="C"))
    return out_path


def _coerce_assoc_matrix_numpy(
    G: np.ndarray,
    *,
    n_samples: int,
    layout: str | None,
) -> tuple[Float32Matrix, pd.Index]:
    arr = np.asarray(G)
    if arr.ndim == 1:
        if int(arr.shape[0]) != int(n_samples):
            raise ValueError(
                f"1D association input length mismatch: got {arr.shape[0]}, expected {n_samples}."
            )
        out = np.ascontiguousarray(arr.reshape(1, -1), dtype=np.float32)
        return out, pd.Index(["snp0"])
    if arr.ndim != 2:
        raise ValueError("Association input G must be 1D or 2D.")
    r, c = int(arr.shape[0]), int(arr.shape[1])
    if layout is None:
        if r == n_samples and c != n_samples:
            out = np.ascontiguousarray(np.asarray(arr, dtype=np.float32).T, dtype=np.float32)
            return out, pd.Index([f"snp{i}" for i in range(c)])
        if c == n_samples and r != n_samples:
            out = np.ascontiguousarray(np.asarray(arr, dtype=np.float32), dtype=np.float32)
            return out, pd.Index([f"snp{i}" for i in range(r)])
        raise ValueError(
            "Cannot infer G layout. Pass sample-major (n x m), SNP-major (m x n), "
            "or specify `layout='sample_major'` / `layout='snp_major'`."
        )
    layout = str(layout).strip().lower()
    if layout == "sample_major":
        if r != n_samples:
            raise ValueError(f"Expected sample-major G with {n_samples} rows, got {arr.shape}.")
        out = np.ascontiguousarray(np.asarray(arr, dtype=np.float32).T, dtype=np.float32)
        return out, pd.Index([f"snp{i}" for i in range(c)])
    if layout == "snp_major":
        if c != n_samples:
            raise ValueError(f"Expected SNP-major G with {n_samples} columns, got {arr.shape}.")
        out = np.ascontiguousarray(np.asarray(arr, dtype=np.float32), dtype=np.float32)
        return out, pd.Index([f"snp{i}" for i in range(r)])
    raise ValueError("layout must be one of: None, 'sample_major', 'snp_major'.")


def _coerce_assoc_matrix_pandas(
    G: pd.Series | pd.DataFrame,
    *,
    sample_index: pd.Index | None,
    n_samples: int,
    layout: str | None,
) -> tuple[Float32Matrix, pd.Index]:
    if isinstance(G, pd.Series):
        if sample_index is not None:
            _ensure_unique_index(G.index, label="association input")
            vec = G.loc[_ordered_labels(sample_index)]
            name = G.name if G.name is not None else "snp0"
            out = np.ascontiguousarray(vec.to_numpy(dtype=np.float32, copy=False).reshape(1, -1), dtype=np.float32)
            return out, pd.Index([name])
        return _coerce_assoc_matrix_numpy(G.to_numpy(), n_samples=n_samples, layout=layout)

    _ensure_unique_index(G.index, label="association input")
    _ensure_unique_index(G.columns, label="association input")
    if layout is not None:
        layout = str(layout).strip().lower()
        if layout == "sample_major":
            if sample_index is not None:
                g_use = G.loc[_ordered_labels(sample_index)]
            elif int(G.shape[0]) != int(n_samples):
                raise ValueError(f"Expected sample-major G with {n_samples} rows, got {G.shape}.")
            else:
                g_use = G
            names = pd.Index(g_use.columns.copy())
            out = np.ascontiguousarray(g_use.to_numpy(dtype=np.float32, copy=False).T, dtype=np.float32)
            return out, names
        if layout == "snp_major":
            if sample_index is not None:
                g_use = G.loc[:, _ordered_labels(sample_index)]
            elif int(G.shape[1]) != int(n_samples):
                raise ValueError(f"Expected SNP-major G with {n_samples} columns, got {G.shape}.")
            else:
                g_use = G
            names = pd.Index(g_use.index.copy())
            out = np.ascontiguousarray(g_use.to_numpy(dtype=np.float32, copy=False), dtype=np.float32)
            return out, names
        raise ValueError("layout must be one of: None, 'sample_major', 'snp_major'.")

    if sample_index is None:
        return _coerce_assoc_matrix_numpy(G.to_numpy(), n_samples=n_samples, layout=None)

    rows_cover = _index_covers_all(G.index, sample_index)
    cols_cover = _index_covers_all(G.columns, sample_index)
    if rows_cover and not cols_cover:
        g_use = G.loc[_ordered_labels(sample_index)]
        names = pd.Index(g_use.columns.copy())
        out = np.ascontiguousarray(g_use.to_numpy(dtype=np.float32, copy=False).T, dtype=np.float32)
        return out, names
    if cols_cover and not rows_cover:
        g_use = G.loc[:, _ordered_labels(sample_index)]
        names = pd.Index(g_use.index.copy())
        out = np.ascontiguousarray(g_use.to_numpy(dtype=np.float32, copy=False), dtype=np.float32)
        return out, names
    raise ValueError(
        "Cannot infer pandas G layout from sample index. "
        "Specify `layout='sample_major'` or `layout='snp_major'`."
    )


class ASSOC:
    """
    Lightweight sklearn-style association wrapper.

    Supported first-version routes:
      - `glm` / `lm`
      - `lmm`
      - `fvlmm`
      - `splmm` exact

    Notes
    -----
    - `assoc(G)` treats `G` as the already-prepared design matrix and does not
      re-center or re-standardize it.
    - `farmcpu` is intentionally not implemented in this first in-memory API.
    - Fitted state is exposed via `route_`, `null_fit_`, and `fit_result_`.
    """

    def __init__(
        self,
        model: AssocModelName | str = "lmm",
        model_args: dict[str, Any] | None = None,
        *,
        threads: int | None = None,
        chunk_size: int | None = None,
    ):
        self.model_name = _normalize_model_name(model)
        self.model_args = {} if model_args is None else dict(model_args)
        self._threads_override = threads
        self._chunk_size_override = chunk_size
        self.requested_model_ = self.model_name
        self.effective_model_: str | None = None
        self.route_: str | None = None
        self.backend_model_: Any = None
        self.sample_index_: pd.Index | None = None
        self.sample_ids_str_: list[str] | None = None
        self.n_samples_: int | None = None
        self.y_: np.ndarray | None = None
        self.X_: np.ndarray | None = None
        self.null_fit_: dict[str, Any] | None = None
        self.fit_result_: dict[str, Any] | None = None
        self.fitted_: bool = False
        self.sparse_grm_path_: str | None = None
        self.sparse_sample_idx_: np.ndarray | None = None
        self._tmpdir_obj: tempfile.TemporaryDirectory[str] | None = None
        self._tmpdir_path: Path | None = None
        self.result_: pd.DataFrame | None = None

    @property
    def threads_(self) -> int:
        raw = self._threads_override
        if raw is None:
            raw = self.model_args.get("threads", 1)
        return _normalize_positive_int(raw, default=1)

    @property
    def block_rows_(self) -> int:
        raw = self.model_args.get("block_rows", 0)
        try:
            val = int(raw)
        except Exception:
            val = 0
        return max(0, val)

    @property
    def chunk_size_(self) -> int:
        raw = self._chunk_size_override
        if raw is None:
            raw = self.model_args.get("chunk_size", self.model_args.get("chunksize", 0))
        try:
            val = int(raw)
        except Exception:
            val = 0
        return max(0, val)

    @property
    def sparse_cutoff_(self) -> float:
        raw = self.model_args.get("sparse_cutoff", self.model_args.get("threshold", 0.05))
        val = float(raw)
        if not np.isfinite(val):
            raise ValueError(
                f"sparse_cutoff must be finite; negative disables off-diagonal thresholding, got {raw!r}"
            )
        return float(val)

    @property
    def sparse_abs_threshold_(self) -> bool:
        return bool(self.model_args.get("abs_threshold", False))

    @property
    def sparse_null_objective_(self) -> str:
        raw = self.model_args.get("sparse_null_objective", self.model_args.get("spk_mode", "fastgwa"))
        mode = str(raw).strip().lower()
        if mode not in {"raw", "fastgwa"}:
            raise ValueError(
                f"sparse_null_objective must be 'raw' or 'fastgwa', got {raw!r}"
            )
        return mode

    @staticmethod
    def toy_data(seed: int = 42) -> dict[str, Any]:
        """
        Embedded toy dataset for quick API checks.

        Returns
        -------
        dict
            Keys:
              `y`        : pandas Series
              `X`        : pandas DataFrame
              `G`        : pandas DataFrame, sample-major
              `K`        : dense numpy kinship
              `K_sparse` : scipy sparse kinship when SciPy sparse is available
        """
        rng = np.random.default_rng(int(seed))
        n = 8
        m = 5
        ids = pd.Index([f"id{i}" for i in range(n)], name="sample")
        y = pd.Series(rng.normal(size=n), index=ids, name="trait")
        X = pd.DataFrame(
            {
                "cov1": rng.normal(size=n),
                "cov2": np.linspace(-1.0, 1.0, n, dtype=np.float64),
            },
            index=ids,
        )
        G = pd.DataFrame(
            rng.normal(size=(n, m)),
            index=ids,
            columns=[f"snp{i}" for i in range(m)],
        )
        K = np.eye(n, dtype=np.float64)
        K[0, 1] = K[1, 0] = 0.15
        K[2, 3] = K[3, 2] = 0.08
        K[4, 5] = K[5, 4] = 0.05
        out: dict[str, Any] = {"y": y, "X": X, "G": G, "K": K}
        out["K_sparse"] = None if sp is None else sp.csc_matrix(K)
        return out

    def _ensure_tmpdir(self) -> Path:
        if self._tmpdir_obj is None:
            self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="jx_assoc_api_")
            self._tmpdir_path = Path(self._tmpdir_obj.name)
        assert self._tmpdir_path is not None
        return self._tmpdir_path

    def _finalize_fit_result(self, *, kinship_kind: str) -> None:
        backend_name = None
        if self.backend_model_ is not None:
            backend_name = type(self.backend_model_).__name__
        elif self.null_fit_ is not None:
            backend_name = self.null_fit_.get("backend")
        self.fit_result_ = {
            "requested_model": str(self.requested_model_),
            "effective_model": None if self.effective_model_ is None else str(self.effective_model_),
            "route": None if self.route_ is None else str(self.route_),
            "backend": backend_name,
            "n_samples": None if self.n_samples_ is None else int(self.n_samples_),
            "n_covariates": 0 if self.X_ is None else int(self.X_.shape[1]),
            "kinship_kind": str(kinship_kind),
            "threads": int(self.threads_),
            "chunk_size": int(self.chunk_size_),
            "has_sample_index": bool(self.sample_index_ is not None),
            "sparse_grm_path": self.sparse_grm_path_,
            "sparse_null_objective": (self.sparse_null_objective_ if self.route_ == "splmm" else None),
            "used_sparse_subset": bool(
                self.sparse_sample_idx_ is not None and int(self.sparse_sample_idx_.shape[0]) > 0
            ),
            "null_fit": None if self.null_fit_ is None else dict(self.null_fit_),
        }
        self.fitted_ = True

    def _resolve_route(self, k: Any) -> str:
        if self.model_name == "farmcpu":
            return "farmcpu"
        if self.model_name == "lm":
            return "lm"
        if self.model_name == "splmm":
            return "splmm"
        if sp is not None and sp.issparse(k):
            return "splmm"
        path = _maybe_path(k)
        if path is not None:
            low = str(path).lower()
            if low.endswith(".spgrm") or low.endswith(".jxgrm"):
                return "splmm"
        if self.model_name == "fvlmm" or bool(self.model_args.get("fv", False)):
            return "fvlmm"
        return "lmm"

    def _prepare_sparse_grm_from_path(
        self,
        path: str,
        *,
        sample_index: pd.Index | None,
        n_samples: int,
    ) -> tuple[str, np.ndarray | None]:
        sparse_path = _splmm_normalize_sparse_grm_path(path)
        id_path = resolve_grm_id_path(sparse_path, None)
        if sample_index is None:
            return sparse_path, None
        target_ids = _sample_ids_to_strings(sample_index) or []
        if id_path is not None:
            source_ids = [str(x) for x in read_id_file(id_path)]
            if target_ids == source_ids:
                return sparse_path, None
            return sparse_path, _positions_from_ids(target_ids, source_ids, label="sparse GRM")
        sparse_n = _sparse_grm_n_samples(sparse_path)
        if int(sparse_n) != int(n_samples):
            raise ValueError(
                "Sparse GRM path does not have a sibling `.id` file, so ASSOC cannot "
                f"subset/reorder it automatically (sparse_n={sparse_n}, y_n={n_samples})."
            )
        return sparse_path, None

    def _build_sparse_grm_from_dense_path(
        self,
        dense_path: str,
        *,
        sample_index: pd.Index | None,
        n_samples: int,
    ) -> tuple[str, np.ndarray | None]:
        low = str(dense_path).lower()
        can_use_npy_direct = bool(
            low.endswith(".npy")
            and _spgrm_dense_npy_to_jxgrm is not None
            and sample_index is None
        )
        out_prefix = str(self._ensure_tmpdir() / "dense_kinship")
        if can_use_npy_direct:
            sparse_path, _n, _nnz = _spgrm_dense_npy_to_jxgrm(
                str(dense_path),
                out_prefix,
                threshold=float(self.sparse_cutoff_),
                abs_threshold=bool(self.sparse_abs_threshold_),
            )
            _write_id_sidecar(str(sparse_path), sample_index)
            return str(sparse_path), None
        dense = _load_dense_kinship_from_path(
            str(dense_path),
            sample_index=sample_index,
            n_samples=n_samples,
        )
        return self._build_sparse_grm_from_dense_array(dense, sample_index=sample_index)

    def _build_sparse_grm_from_dense_array(
        self,
        dense: np.ndarray,
        *,
        sample_index: pd.Index | None,
    ) -> tuple[str, np.ndarray | None]:
        if _spgrm_dense_f32_to_jxgrm is None:
            raise RuntimeError("Rust extension missing spgrm_dense_f32_to_jxgrm.")
        arr = np.asarray(dense, dtype=np.float32)
        if arr.ndim != 2 or int(arr.shape[0]) != int(arr.shape[1]):
            raise ValueError(f"Dense kinship must be square, got shape={arr.shape}.")
        out_prefix = str(self._ensure_tmpdir() / "dense_kinship")
        sparse_path, _n, _nnz = _spgrm_dense_f32_to_jxgrm(
            np.ascontiguousarray(arr, dtype=np.float32),
            out_prefix,
            threshold=float(self.sparse_cutoff_),
            abs_threshold=bool(self.sparse_abs_threshold_),
        )
        _write_id_sidecar(str(sparse_path), sample_index)
        return str(sparse_path), None

    def _build_sparse_grm_from_scipy(
        self,
        matrix: Any,
        *,
        sample_index: pd.Index | None,
        n_samples: int,
    ) -> tuple[str, np.ndarray | None]:
        if sp is None or not sp.issparse(matrix):
            raise TypeError("Expected a scipy sparse kinship matrix.")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Sparse kinship must be square, got shape={matrix.shape}.")
        if int(matrix.shape[0]) != int(n_samples):
            raise ValueError(
                f"Sparse kinship size mismatch: got {matrix.shape}, expected ({n_samples}, {n_samples})."
            )
        out_path = str(self._ensure_tmpdir() / "dense_kinship.spgrm")
        sparse_path = _write_sparse_grm_from_scipy(matrix, out_path)
        _write_id_sidecar(str(sparse_path), sample_index)
        return str(sparse_path), None

    def _prepare_sparse_kinship(
        self,
        k: Any,
        *,
        sample_index: pd.Index | None,
        n_samples: int,
    ) -> tuple[str, np.ndarray | None]:
        path = _maybe_path(k)
        if path is not None:
            low = str(path).lower()
            if low.endswith(".spgrm") or low.endswith(".jxgrm"):
                return self._prepare_sparse_grm_from_path(path, sample_index=sample_index, n_samples=n_samples)
            return self._build_sparse_grm_from_dense_path(path, sample_index=sample_index, n_samples=n_samples)
        if sp is not None and sp.issparse(k):
            return self._build_sparse_grm_from_scipy(k, sample_index=sample_index, n_samples=n_samples)
        dense = _coerce_dense_kinship(k, sample_index=sample_index, n_samples=n_samples)
        return self._build_sparse_grm_from_dense_array(dense, sample_index=sample_index)

    def _resolve_assoc_parallelism(
        self,
        n_rows: int,
        *,
        threads: int | None,
        chunk_size: int | None,
    ) -> tuple[int, int, int]:
        assoc_threads = self.threads_ if threads is None else _normalize_positive_int(threads, default=1)
        resolved_chunk_size = self.chunk_size_ if chunk_size is None else max(0, int(chunk_size))
        if assoc_threads <= 1:
            return 1, max(0, resolved_chunk_size), 1
        if resolved_chunk_size <= 0:
            resolved_chunk_size = max(1, (int(n_rows) + int(assoc_threads) - 1) // int(assoc_threads))
        if int(n_rows) <= int(resolved_chunk_size):
            return 1, int(resolved_chunk_size), int(assoc_threads)
        outer_threads = min(int(assoc_threads), (int(n_rows) + int(resolved_chunk_size) - 1) // int(resolved_chunk_size))
        if outer_threads <= 1:
            return 1, int(resolved_chunk_size), int(assoc_threads)
        return int(outer_threads), int(resolved_chunk_size), 1

    def _assoc_backend_dense(
        self,
        g_snp_major: np.ndarray,
        *,
        threads: int,
        chunk_size: int,
    ) -> np.ndarray:
        if self.backend_model_ is None:
            raise RuntimeError("Backend model is not initialized.")
        n_rows = int(g_snp_major.shape[0])
        outer_threads, resolved_chunk_size, inner_threads = self._resolve_assoc_parallelism(
            n_rows,
            threads=threads,
            chunk_size=chunk_size,
        )
        if outer_threads <= 1:
            arr = np.asarray(
                self.backend_model_.gwas(
                    np.ascontiguousarray(g_snp_major, dtype=np.float32),
                    threads=int(inner_threads),
                ),
                dtype=np.float64,
            )
            if int(arr.ndim) != 2 or int(arr.shape[1]) < 3:
                raise RuntimeError(f"Unexpected association output shape: {arr.shape}")
            return arr[:, :3]

        chunk_specs = [
            (start, min(start + resolved_chunk_size, n_rows))
            for start in range(0, n_rows, resolved_chunk_size)
        ]
        out = np.empty((n_rows, 3), dtype=np.float64)

        def _run_chunk(start: int, stop: int) -> tuple[int, np.ndarray]:
            backend = _clone_backend_model_for_assoc(self.backend_model_)
            chunk = np.ascontiguousarray(g_snp_major[start:stop], dtype=np.float32)
            arr = np.asarray(backend.gwas(chunk, threads=1), dtype=np.float64)
            if int(arr.ndim) != 2 or int(arr.shape[1]) < 3:
                raise RuntimeError(f"Unexpected chunk association output shape: {arr.shape}")
            return start, arr[:, :3]

        with cf.ThreadPoolExecutor(max_workers=int(outer_threads), thread_name_prefix="jx-assoc") as ex:
            futures = [ex.submit(_run_chunk, start, stop) for start, stop in chunk_specs]
            for fut in futures:
                start, arr = fut.result()
                stop = start + int(arr.shape[0])
                out[start:stop, :] = arr
        return out

    def _assoc_backend_splmm(
        self,
        g_snp_major: np.ndarray,
        *,
        threads: int,
        chunk_size: int,
    ) -> np.ndarray:
        if _splmm_assoc_pcg_dense_f32 is None:
            raise RuntimeError("Rust extension missing splmm_assoc_pcg_dense_f32.")
        if self.null_fit_ is None or self.sparse_grm_path_ is None:
            raise RuntimeError("SparseLMM state is incomplete; fit must finish successfully first.")
        n_rows = int(g_snp_major.shape[0])
        outer_threads, resolved_chunk_size, inner_threads = self._resolve_assoc_parallelism(
            n_rows,
            threads=threads,
            chunk_size=chunk_size,
        )
        if outer_threads <= 1:
            return np.asarray(
                _splmm_assoc_pcg_dense_f32(
                    np.ascontiguousarray(g_snp_major, dtype=np.float32),
                    self.y_,
                    float(self.null_fit_["lambda"]),
                    str(self.sparse_grm_path_),
                    x_cov=self.X_,
                    sparse_sample_indices=self.sparse_sample_idx_,
                    threads=int(inner_threads),
                    block_rows=self.block_rows_,
                ),
                dtype=np.float64,
            )

        chunk_specs = [
            (start, min(start + resolved_chunk_size, n_rows))
            for start in range(0, n_rows, resolved_chunk_size)
        ]
        out = np.empty((n_rows, 3), dtype=np.float64)

        def _run_chunk(start: int, stop: int) -> tuple[int, np.ndarray]:
            chunk = np.ascontiguousarray(g_snp_major[start:stop], dtype=np.float32)
            arr = np.asarray(
                _splmm_assoc_pcg_dense_f32(
                    chunk,
                    self.y_,
                    float(self.null_fit_["lambda"]),
                    str(self.sparse_grm_path_),
                    x_cov=self.X_,
                    sparse_sample_indices=self.sparse_sample_idx_,
                    threads=1,
                    block_rows=self.block_rows_,
                ),
                dtype=np.float64,
            )
            return start, arr

        with cf.ThreadPoolExecutor(max_workers=int(outer_threads), thread_name_prefix="jx-splmm") as ex:
            futures = [ex.submit(_run_chunk, start, stop) for start, stop in chunk_specs]
            for fut in futures:
                start, arr = fut.result()
                stop = start + int(arr.shape[0])
                out[start:stop, :] = arr
        return out

    def fit(
        self,
        y: ResponseLike,
        X: CovariateLike | None = None,
        k: KinshipLike | None = None,
    ) -> "ASSOC":
        """
        Fit the null association model.

        Parameters
        ----------
        y : ResponseLike
            Phenotype vector. Accepts a 1D numpy-like object, a pandas
            ``Series``, or a single-column pandas ``DataFrame``. Numeric values
            are coerced to contiguous ``np.float64``.
        X : CovariateLike, default=None
            Covariate design matrix. Accepts a 1D/2D numpy-like object, pandas
            ``Series``, or pandas ``DataFrame``. Numeric values are coerced to
            contiguous ``np.float64``. When pandas objects are provided,
            ``ASSOC`` aligns by sample index.
        k : KinshipLike, default=None
            Kinship / GRM input. Accepts a dense numpy-like square matrix,
            pandas ``DataFrame``, scipy sparse matrix, or a path to a dense or
            sparse GRM on disk. Required for all non-``lm`` routes.

        Returns
        -------
        ASSOC
            Fitted estimator.

        Notes
        -----
        Public input aliases are exported from ``janusx.assoc``:
        ``ResponseLike``, ``CovariateLike``, and ``KinshipLike``.
        """
        y_arr, sample_index = _coerce_response(y)
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("Phenotype contains NaN/Inf values.")
        x_cov = _coerce_covariates(X, sample_index=sample_index, n_samples=int(y_arr.shape[0]))
        route = self._resolve_route(k)

        self.sample_index_ = sample_index
        self.sample_ids_str_ = _sample_ids_to_strings(sample_index)
        self.n_samples_ = int(y_arr.shape[0])
        self.y_ = y_arr
        self.X_ = x_cov
        self.route_ = route
        self.effective_model_ = route
        self.result_ = None

        if route == "farmcpu":
            raise NotImplementedError(
                "ASSOC first version does not support in-memory FarmCPU yet. "
                "Use janusx.assoc.LinearModel or the CLI workflow for FarmCPU."
            )

        if route == "lm":
            self.backend_model_ = LM(y_arr, X=x_cov)
            self.null_fit_ = _null_fit_record(
                model="lm",
                route="lm",
                backend=type(self.backend_model_).__name__,
                converged=True,
            )
            self.sparse_grm_path_ = None
            self.sparse_sample_idx_ = None
            self._finalize_fit_result(kinship_kind="none")
            return self

        if k is None:
            raise ValueError(f"Model `{route}` requires a kinship / GRM input `k`.")

        if route == "splmm":
            sparse_path, sparse_sample_idx = self._prepare_sparse_kinship(
                k,
                sample_index=sample_index,
                n_samples=self.n_samples_,
            )
            null_fit = _splmm_sparse_null_fit(
                jxgrm_path=str(sparse_path),
                sample_idx=sparse_sample_idx,
                y_vec=y_arr,
                x_cov=x_cov,
                progress_callback=None,
                residualized_approx=False,
                objective_mode=self.sparse_null_objective_,
                threads=self.threads_,
            )
            lbd = float(null_fit.get("lambda", float("nan")))
            if not (np.isfinite(lbd) and lbd >= 0.0):
                raise RuntimeError(f"SparseLMM exact null fit returned invalid lambda: {lbd}")
            self.backend_model_ = None
            self.null_fit_ = _null_fit_record(
                model="splmm",
                route="splmm",
                backend=str(null_fit.get("backend", "sparse_cholesky")),
                converged=bool(null_fit.get("converged", True)),
                lbd=float(null_fit["lambda"]) if "lambda" in null_fit else None,
                log10_lbd=float(null_fit["log10_lambda"]) if "log10_lambda" in null_fit else None,
                pve=float(null_fit["pve"]) if "pve" in null_fit else None,
                ml0=float(null_fit["ml"]) if "ml" in null_fit else None,
                reml0=float(null_fit["reml"]) if "reml" in null_fit else None,
                va=float(null_fit["sigma_g2"]) if "sigma_g2" in null_fit else None,
                ve=float(null_fit["sigma_e2"]) if "sigma_e2" in null_fit else None,
                extras=dict(null_fit),
            )
            self.sparse_grm_path_ = str(sparse_path)
            self.sparse_sample_idx_ = (
                None
                if sparse_sample_idx is None
                else np.ascontiguousarray(np.asarray(sparse_sample_idx, dtype=np.int64).reshape(-1), dtype=np.int64)
            )
            self._finalize_fit_result(kinship_kind="sparse")
            return self

        kinship = _coerce_dense_kinship(k, sample_index=sample_index, n_samples=self.n_samples_)
        if route == "fvlmm":
            self.backend_model_ = FvLMM(y_arr, x_cov, kinship)
        else:
            self.backend_model_ = LMM(y_arr, x_cov, kinship)
        self.null_fit_ = _dense_backend_null_fit(route, self.backend_model_)
        self.sparse_grm_path_ = None
        self.sparse_sample_idx_ = None
        self._finalize_fit_result(kinship_kind="dense")
        return self

    def assoc(
        self,
        G: AssocMatrixLike,
        layout: AssocLayout | str | None = None,
        *,
        threads: int | None = None,
        chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """
        Run association scans on a pre-aligned design matrix.

        Parameters
        ----------
        G : AssocMatrixLike
            Variant matrix to scan. Accepts numpy-like input, pandas
            ``Series``, or pandas ``DataFrame``. ``G`` is treated as already
            prepared numeric data and is not centered or standardized again.
            Numeric values are coerced to contiguous ``np.float32`` for the
            scan kernel.
        layout : {'sample_major', 'snp_major'}, default=None
            Layout hint for 2D ``G``. If omitted, numpy-like inputs are inferred
            from shape when possible, and pandas inputs are inferred from sample
            labels when available.
        threads : int, default=None
            Scan-time thread override. If chunked parallelism is used, outer
            Python workers split SNP chunks and inner backend calls fall back to
            single-thread execution to avoid oversubscription.
        chunk_size : int, default=None
            Optional SNP chunk size for outer parallel scanning.

        Returns
        -------
        pandas.DataFrame
            Association statistics indexed by feature name with columns
            ``beta``, ``se``, and ``pwald``.

        Notes
        -----
        Public input aliases are exported from ``janusx.assoc``:
        ``AssocMatrixLike`` and ``AssocLayout``.
        """
        if self.y_ is None or self.n_samples_ is None or self.route_ is None:
            raise RuntimeError("Call `fit(y, X, k)` before `assoc(G)`.")
        if isinstance(G, (pd.Series, pd.DataFrame)):
            g_snp_major, feature_names = _coerce_assoc_matrix_pandas(
                G,
                sample_index=self.sample_index_,
                n_samples=self.n_samples_,
                layout=layout,
            )
        else:
            g_snp_major, feature_names = _coerce_assoc_matrix_numpy(
                np.asarray(G),
                n_samples=self.n_samples_,
                layout=layout,
            )
        if not np.all(np.isfinite(g_snp_major)):
            raise ValueError("Association input G contains NaN/Inf values.")

        if self.route_ == "splmm":
            arr = self._assoc_backend_splmm(
                g_snp_major,
                threads=self.threads_ if threads is None else int(threads),
                chunk_size=self.chunk_size_ if chunk_size is None else int(chunk_size),
            )
        else:
            arr = self._assoc_backend_dense(
                g_snp_major,
                threads=self.threads_ if threads is None else int(threads),
                chunk_size=self.chunk_size_ if chunk_size is None else int(chunk_size),
            )

        out = pd.DataFrame(
            {
                "beta": arr[:, 0],
                "se": arr[:, 1],
                "pwald": arr[:, 2],
            },
            index=feature_names,
        )
        self.result_ = out
        return out


__all__ = ["ASSOC"]
