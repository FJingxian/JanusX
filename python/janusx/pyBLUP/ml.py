from __future__ import annotations

import os
import typing
from contextlib import ExitStack, nullcontext
from typing import Any, Optional

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional dependency path
    threadpool_limits = None  # type: ignore[assignment]

try:
    from xgboost import XGBRegressor

    _HAS_XGBOOST = True
    _XGBOOST_IMPORT_ERROR: Exception | None = None
except Exception as _xgb_exc:  # pragma: no cover - optional dependency
    XGBRegressor = None  # type: ignore[assignment]
    _HAS_XGBOOST = False
    _XGBOOST_IMPORT_ERROR = _xgb_exc


ScoreName = typing.Literal["pearson", "r2", "neg_rmse"]
MethodName = typing.Literal["rf", "et", "gbdt", "xgb", "svm", "enet"]
FeatureAxisName = typing.Literal["auto", "marker_by_sample", "sample_by_marker"]
SearchSchemeName = typing.Literal["legacy", "multicenter"]


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 2:
        return 0.0
    if (not np.isfinite(a).all()) or (not np.isfinite(b).all()):
        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]
    if a.size < 2:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    if a.size != b.size or a.size == 0:
        return float("-inf")
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    ss_res = float(np.sum((a - b) ** 2))
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    if a.size != b.size or a.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _score_value(y_true: np.ndarray, y_pred: np.ndarray, scoring: ScoreName) -> float:
    if scoring == "pearson":
        return _safe_pearson(y_true, y_pred)
    if scoring == "r2":
        return _safe_r2(y_true, y_pred)
    if scoring == "neg_rmse":
        return -_rmse(y_true, y_pred)
    raise ValueError(f"Unsupported scoring method: {scoring}")


def _as_1d_y(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Phenotype y is empty.")
    if not np.isfinite(arr).any():
        raise ValueError("Phenotype y contains no finite value.")
    return arr


def _prepare_covariates(cov: np.ndarray | None, n_samples: int) -> np.ndarray | None:
    if cov is None:
        return None
    arr = np.asarray(cov, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"cov must be 2D. got shape={arr.shape}")
    if arr.shape[0] != n_samples:
        raise ValueError(
            f"Row number of cov is not equal to sample size. cov.shape={arr.shape}, n={n_samples}"
        )
    return arr


def _resolve_marker_matrix(
    M: np.ndarray,
    y_len: int,
    feature_axis: FeatureAxisName = "auto",
) -> tuple[np.ndarray, str]:
    arr = np.asarray(M, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"M must be 2D. got shape={arr.shape}")

    if feature_axis == "marker_by_sample":
        if arr.shape[1] != y_len:
            raise ValueError(
                f"M is declared as marker_by_sample but M.shape[1]={arr.shape[1]} != len(y)={y_len}"
            )
        return arr.T.copy(), "marker_by_sample"

    if feature_axis == "sample_by_marker":
        if arr.shape[0] != y_len:
            raise ValueError(
                f"M is declared as sample_by_marker but M.shape[0]={arr.shape[0]} != len(y)={y_len}"
            )
        return arr.copy(), "sample_by_marker"

    if arr.shape[0] == y_len and arr.shape[1] != y_len:
        return arr.copy(), "sample_by_marker"
    if arr.shape[1] == y_len and arr.shape[0] != y_len:
        return arr.T.copy(), "marker_by_sample"
    if arr.shape[0] == y_len and arr.shape[1] == y_len:
        return arr.copy(), "sample_by_marker"
    raise ValueError(
        f"Cannot infer marker/sample axis from M.shape={arr.shape} and len(y)={y_len}. "
        "Please specify feature_axis explicitly."
    )


def _impute_matrix_with_means(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(X, dtype=np.float32)
    means = np.nanmean(mat, axis=0)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32, copy=False)
    if np.isnan(mat).any():
        idx = np.where(np.isnan(mat))
        mat = mat.copy()
        mat[idx] = means[idx[1]]
    return mat, means


def _apply_column_means(X: np.ndarray, means: np.ndarray) -> np.ndarray:
    mat = np.asarray(X, dtype=np.float32)
    if mat.shape[1] != means.shape[0]:
        raise ValueError(
            f"Feature number mismatch during prediction. X.shape={mat.shape}, means.shape={means.shape}"
        )
    if np.isnan(mat).any():
        idx = np.where(np.isnan(mat))
        mat = mat.copy()
        mat[idx] = means[idx[1]]
    return mat


def _unique_sorted(vals: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[Any] = set()
    for v in vals:
        key = v
        if isinstance(v, float):
            key = round(v, 8)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


class MLGS:
    """
    Machine-learning genomic selection with SNP-specific search spaces.

    Notes
    -----
    This class is designed for genomic selection with:
    - many markers (typically p >> n),
    - continuous phenotype,
    - marker matrix coded as 0/1/2, optionally with NaN.

    The tuner uses a two-stage search:
    1. a coarse random search over a compact, GS-oriented parameter space;
    2. a fine search around the best coarse setting.

    Pearson correlation is used by default because ranking accuracy is usually
    more informative than plain R^2 in breeding-style prediction tasks.
    """

    def __init__(
        self,
        y: np.ndarray,
        M: np.ndarray,
        cov: np.ndarray | None = None,
        method: MethodName = "rf",
        seed: Optional[int] = None,
        cv: int = 3,
        scoring: ScoreName = "pearson",
        feature_axis: FeatureAxisName = "auto",
        n_jobs: Optional[int] = None,
        coarse_iter: Optional[int] = None,
        fine_iter: Optional[int] = None,
        search_scheme: SearchSchemeName = "multicenter",
        coarse_top_k: int = 3,
        confirm_top_k: int = 3,
        confirm_repeats: int = 2,
        fit_on_init: bool = True,
        verbose: bool = False,
    ):
        self.method = str(method)
        self.seed = 0 if seed is None else int(seed)
        self.cv = max(2, int(cv))
        self.scoring = scoring
        self.feature_axis = feature_axis
        self.n_jobs = max(1, int(os.cpu_count() or 1)) if n_jobs is None else max(1, int(n_jobs))
        self.verbose = bool(verbose)
        self.parallel_mode_ = self._parallel_mode()
        self.search_scheme = typing.cast(SearchSchemeName, str(search_scheme))
        default_top_k = 2 if self.method == "svm" else 3
        self.coarse_top_k = max(1, int(coarse_top_k))
        self.confirm_top_k = max(1, int(confirm_top_k))
        if coarse_top_k == 3 and self.method == "svm":
            self.coarse_top_k = default_top_k
        if confirm_top_k == 3 and self.method == "svm":
            self.confirm_top_k = default_top_k
        self.confirm_repeats = max(1, int(confirm_repeats))

        self.y = _as_1d_y(y)
        self.X_marker, self.feature_axis_ = _resolve_marker_matrix(M, self.y.size, feature_axis)
        self.cov = _prepare_covariates(cov, self.y.size)
        self.marker_count_ = int(self.X_marker.shape[1])
        self.sample_count_ = int(self.X_marker.shape[0])
        self.cov_count_ = 0 if self.cov is None else int(self.cov.shape[1])

        self.X_marker, self.marker_means_ = _impute_matrix_with_means(self.X_marker)
        if self.cov is not None:
            self.cov, self.cov_means_ = _impute_matrix_with_means(self.cov.astype(np.float32, copy=False))
        else:
            self.cov_means_ = None

        self.X = self._merge_features(self.X_marker, self.cov)
        self.coarse_iter = (
            int(coarse_iter)
            if coarse_iter is not None
            else self._default_search_iter(stage="coarse")
        )
        self.fine_iter = (
            int(fine_iter)
            if fine_iter is not None
            else self._default_search_iter(stage="fine")
        )

        self.model: Any = None
        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.best_metrics_: dict[str, float] | None = None
        self.best_n_estimators_: int | None = None
        self.cv_results_: list[dict[str, Any]] = []

        if fit_on_init:
            self.fit()

    def _parallel_mode(self) -> typing.Literal["model", "search"]:
        if self.method in {"svm", "enet"}:
            return "search"
        return "model"

    def _model_n_jobs(self) -> int:
        if self.parallel_mode_ == "model":
            return self.n_jobs
        return 1

    def _thread_context(self):
        use_joblib_backend = self.parallel_mode_ == "model" and self.method in {"rf", "et"}
        use_threadpool_limit = self.parallel_mode_ == "model" and threadpool_limits is not None
        if (not use_joblib_backend) and (not use_threadpool_limit):
            return nullcontext()

        stack = ExitStack()
        if use_joblib_backend:
            stack.enter_context(parallel_backend("threading", n_jobs=self._model_n_jobs()))
        if use_threadpool_limit:
            stack.enter_context(threadpool_limits(limits=self._model_n_jobs()))
        return stack

    @staticmethod
    def _result_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row.get("score", float("-inf"))),
            float(row.get("pearson", float("-inf"))),
            float(row.get("r2", float("-inf"))),
            -float(row.get("rmse", float("inf"))),
        )

    @staticmethod
    def _param_signature(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        norm: list[tuple[str, Any]] = []
        for k, v in sorted(params.items()):
            if isinstance(v, float):
                norm.append((k, round(v, 10)))
            else:
                norm.append((k, v))
        return tuple(norm)

    def _dedup_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, Any], ...]] = set()
        for params in candidates:
            sig = self._param_signature(params)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(dict(params))
        return out

    def _top_result_rows(self, rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        uniq: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, Any], ...]] = set()
        for row in sorted(rows, key=self._result_sort_key, reverse=True):
            sig = self._param_signature(dict(row["params"]))
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append(row)
            if len(uniq) >= max(1, int(k)):
                break
        return uniq

    def _default_search_iter(self, stage: typing.Literal["coarse", "fine"]) -> int:
        n = self.sample_count_
        p = self.marker_count_
        large_n = n >= 1500
        medium_n = n >= 600
        huge_p = p >= 100_000
        large_p = p >= 20_000

        if self.method == "rf":
            if stage == "coarse":
                if large_n or huge_p:
                    return 6
                if medium_n or large_p:
                    return 8
                return 10
            if large_n or huge_p:
                return 3
            if medium_n or large_p:
                return 4
            return 6
        if self.method == "et":
            if stage == "coarse":
                if large_n or huge_p:
                    return 6
                if medium_n or large_p:
                    return 8
                return 10
            if large_n or huge_p:
                return 3
            if medium_n or large_p:
                return 4
            return 6
        if self.method == "gbdt":
            if stage == "coarse":
                if large_n or huge_p:
                    return 8
                if medium_n or large_p:
                    return 10
                return 12
            if large_n or huge_p:
                return 4
            if medium_n or large_p:
                return 6
            return 8
        if self.method == "xgb":
            if stage == "coarse":
                if large_n or huge_p:
                    return 8
                if medium_n or large_p:
                    return 10
                return 12
            if large_n or huge_p:
                return 4
            if medium_n or large_p:
                return 5
            return 6
        if self.method == "svm":
            if stage == "coarse":
                if large_n or huge_p:
                    return 4
                if medium_n or large_p:
                    return 6
                return 8
            if large_n or huge_p:
                return 2
            if medium_n or large_p:
                return 3
            return 4
        if self.method == "enet":
            if stage == "coarse":
                if large_n or huge_p:
                    return 6
                if medium_n or large_p:
                    return 8
                return 10
            if large_n or huge_p:
                return 3
            if medium_n or large_p:
                return 4
            return 6
        return 8 if stage == "coarse" else 4

    @staticmethod
    def _merge_features(X_marker: np.ndarray, cov: np.ndarray | None) -> np.ndarray:
        if cov is None:
            return np.asarray(X_marker, dtype=np.float32, order="C")
        return np.concatenate(
            [
                np.asarray(X_marker, dtype=np.float32, order="C"),
                np.asarray(cov, dtype=np.float32, order="C"),
            ],
            axis=1,
        )

    def _build_estimator(self, params: dict[str, Any]) -> Any:
        if self.method == "rf":
            base = dict(
                random_state=self.seed,
                n_jobs=self._model_n_jobs(),
                bootstrap=True,
            )
            base.update(params)
            return RandomForestRegressor(**base)

        if self.method == "et":
            base = dict(
                random_state=self.seed,
                n_jobs=self._model_n_jobs(),
                bootstrap=False,
            )
            base.update(params)
            return ExtraTreesRegressor(**base)

        if self.method == "gbdt":
            base = dict(
                random_state=self.seed,
                loss="squared_error",
                early_stopping=False,
            )
            base.update(params)
            return HistGradientBoostingRegressor(**base)

        if self.method == "xgb":
            if not _HAS_XGBOOST or XGBRegressor is None:
                raise ImportError(
                    "XGBoost is unavailable for method='xgb'. "
                    f"Original import error: {_XGBOOST_IMPORT_ERROR}"
                )
            base = dict(
                objective="reg:squarederror",
                booster="gbtree",
                tree_method="hist",
                random_state=self.seed,
                n_jobs=self._model_n_jobs(),
                verbosity=0,
                n_estimators=2000,
            )
            base.update(params)
            return XGBRegressor(**base)

        if self.method == "svm":
            est = Pipeline(
                [
                    ("scale", StandardScaler(with_mean=True, with_std=True)),
                    ("svr", SVR(kernel="rbf", cache_size=1024)),
                ]
            )
            if params:
                est.set_params(**params)
            return est

        if self.method == "enet":
            est = Pipeline(
                [
                    ("scale", StandardScaler(with_mean=True, with_std=True)),
                    (
                        "enet",
                        ElasticNet(
                            random_state=self.seed,
                            max_iter=5000,
                            selection="random",
                        ),
                    ),
                ]
            )
            if params:
                est.set_params(**params)
            return est

        raise ValueError(f"Unsupported MLGS method: {self.method}")

    def _coarse_space(self) -> dict[str, list[Any]]:
        n = self.sample_count_
        p = self.marker_count_

        if self.method == "rf":
            frac_min = max(0.005, min(0.05, 32.0 / max(1.0, float(p))))
            max_features = _unique_sorted(
                ["sqrt", round(frac_min, 4), 0.01, 0.03, 0.05, 0.10, 0.20]
            )
            leaf_vals = _unique_sorted(
                [
                    1,
                    2,
                    max(3, n // 100),
                    max(5, n // 50),
                    max(10, n // 25),
                ]
            )
            depth_vals = [None, 8, 16, 24]
            return {
                "n_estimators": [300, 600, 900],
                "max_features": max_features,
                "min_samples_leaf": leaf_vals,
                "max_depth": depth_vals,
            }

        if self.method == "et":
            frac_min = max(0.005, min(0.05, 32.0 / max(1.0, float(p))))
            max_features = _unique_sorted(
                ["sqrt", round(frac_min, 4), 0.01, 0.03, 0.05, 0.10, 0.20]
            )
            leaf_vals = _unique_sorted(
                [
                    1,
                    2,
                    max(3, n // 120),
                    max(5, n // 60),
                    max(10, n // 30),
                ]
            )
            return {
                "n_estimators": [300, 600, 900],
                "max_features": max_features,
                "min_samples_leaf": leaf_vals,
                "max_depth": [None, 8, 16, 24],
            }

        if self.method == "gbdt":
            leaf_vals = _unique_sorted([15, 31, 63, 127])
            min_leaf = _unique_sorted([5, 10, max(20, n // 50), max(50, n // 20)])
            return {
                "learning_rate": [0.03, 0.05, 0.10],
                "max_iter": [200, 400, 800],
                "max_leaf_nodes": leaf_vals,
                "min_samples_leaf": min_leaf,
                "l2_regularization": [0.0, 0.01, 0.1, 1.0],
                "max_depth": [None, 4, 8],
            }

        if self.method == "xgb":
            frac_min = max(0.02, min(0.20, 64.0 / max(1.0, float(p))))
            colsample = _unique_sorted(
                [round(frac_min, 4), 0.05, 0.10, 0.20, 0.40, 0.70]
            )
            return {
                "learning_rate": [0.03, 0.05, 0.10],
                "max_depth": [2, 4, 6],
                "subsample": [0.5, 0.7, 0.9, 1.0],
                "colsample_bytree": colsample,
                "min_child_weight": [1, 3, 5, 10],
                "reg_lambda": [1.0, 3.0, 5.0, 10.0],
            }

        if self.method == "svm":
            gamma_base = max(1.0 / max(32.0, float(p)), 1e-5)
            return {
                "svr__C": [0.5, 1.0, 2.0, 4.0, 8.0],
                "svr__epsilon": [0.01, 0.05, 0.10, 0.20],
                "svr__gamma": _unique_sorted(
                    [
                        "scale",
                        "auto",
                        round(gamma_base, 6),
                        round(max(gamma_base / 4.0, 1e-6), 6),
                    ]
                ),
            }

        if self.method == "enet":
            return {
                "enet__alpha": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0],
                "enet__l1_ratio": [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
                "enet__tol": [1e-4, 5e-4, 1e-3],
            }

        raise ValueError(f"Unsupported MLGS method: {self.method}")

    def _fine_space(self, best: dict[str, Any]) -> dict[str, list[Any]]:
        n = self.sample_count_
        p = self.marker_count_

        if self.method == "rf":
            best_depth = best.get("max_depth", None)
            best_leaf = int(best.get("min_samples_leaf", 1))
            best_n = int(best.get("n_estimators", 300))
            best_feat = best.get("max_features", "sqrt")
            if isinstance(best_feat, float):
                feat_vals = _unique_sorted(
                    [
                        max(0.005, round(best_feat / 2.0, 4)),
                        round(best_feat, 4),
                        min(0.5, round(best_feat * 2.0, 4)),
                        "sqrt",
                    ]
                )
            else:
                feat_vals = _unique_sorted(["sqrt", 0.01, 0.03, 0.05, 0.10, min(0.2, 256.0 / max(1.0, float(p)))])
            depth_vals: list[Any]
            if best_depth is None:
                depth_vals = [None, 12, 20, 28]
            else:
                depth_vals = _unique_sorted(
                    [None, max(4, best_depth // 2), best_depth, best_depth * 2]
                )
            return {
                "n_estimators": _unique_sorted([max(200, best_n - 200), best_n, best_n + 200, best_n + 400]),
                "max_features": feat_vals,
                "min_samples_leaf": _unique_sorted([1, max(1, best_leaf // 2), best_leaf, best_leaf * 2, max(10, n // 25)]),
                "max_depth": depth_vals,
            }

        if self.method == "et":
            best_depth = best.get("max_depth", None)
            best_leaf = int(best.get("min_samples_leaf", 1))
            best_n = int(best.get("n_estimators", 300))
            best_feat = best.get("max_features", "sqrt")
            if isinstance(best_feat, float):
                feat_vals = _unique_sorted(
                    [
                        max(0.005, round(best_feat / 2.0, 4)),
                        round(best_feat, 4),
                        min(0.5, round(best_feat * 2.0, 4)),
                        "sqrt",
                    ]
                )
            else:
                feat_vals = _unique_sorted(
                    ["sqrt", 0.01, 0.03, 0.05, 0.10, min(0.2, 256.0 / max(1.0, float(p)))]
                )
            if best_depth is None:
                depth_vals = [None, 12, 20, 28]
            else:
                depth_vals = _unique_sorted([None, max(4, best_depth // 2), best_depth, best_depth * 2])
            return {
                "n_estimators": _unique_sorted([max(200, best_n - 200), best_n, best_n + 200, best_n + 400]),
                "max_features": feat_vals,
                "min_samples_leaf": _unique_sorted([1, max(1, best_leaf // 2), best_leaf, best_leaf * 2, max(10, n // 25)]),
                "max_depth": depth_vals,
            }

        if self.method == "gbdt":
            best_iter = int(best.get("max_iter", 400))
            best_leaf_nodes = int(best.get("max_leaf_nodes", 31))
            best_min_leaf = int(best.get("min_samples_leaf", 20))
            best_lr = float(best.get("learning_rate", 0.05))
            best_depth = best.get("max_depth", None)
            if best_depth is None:
                depth_vals = [None, 4, 8]
            else:
                depth_vals = _unique_sorted([None, max(2, best_depth - 2), best_depth, best_depth + 2])
            return {
                "learning_rate": _unique_sorted([0.02, round(best_lr, 4), min(0.2, round(best_lr * 1.5, 4))]),
                "max_iter": _unique_sorted([max(100, best_iter // 2), best_iter, best_iter + 200]),
                "max_leaf_nodes": _unique_sorted([max(15, best_leaf_nodes // 2), best_leaf_nodes, best_leaf_nodes * 2]),
                "min_samples_leaf": _unique_sorted([max(2, best_min_leaf // 2), best_min_leaf, best_min_leaf * 2, max(20, n // 40)]),
                "l2_regularization": _unique_sorted([0.0, float(best.get("l2_regularization", 0.0)), 0.1, 1.0]),
                "max_depth": depth_vals,
            }

        if self.method == "xgb":
            best_lr = float(best.get("learning_rate", 0.05))
            best_depth = int(best.get("max_depth", 4))
            best_sub = float(best.get("subsample", 0.8))
            best_col = float(best.get("colsample_bytree", 0.2))
            best_mc = int(best.get("min_child_weight", 3))
            best_l2 = float(best.get("reg_lambda", 1.0))
            min_col = max(0.02, 32.0 / max(1.0, float(p)))
            return {
                "learning_rate": _unique_sorted([0.02, round(best_lr, 4), min(0.2, round(best_lr * 1.5, 4))]),
                "max_depth": _unique_sorted([max(2, best_depth - 1), best_depth, best_depth + 1]),
                "subsample": _unique_sorted([max(0.4, round(best_sub - 0.2, 3)), round(best_sub, 3), min(1.0, round(best_sub + 0.2, 3))]),
                "colsample_bytree": _unique_sorted([max(min_col, round(best_col / 2.0, 3)), round(best_col, 3), min(0.8, round(best_col * 1.5, 3))]),
                "min_child_weight": _unique_sorted([max(1, best_mc // 2), best_mc, best_mc * 2]),
                "reg_lambda": _unique_sorted([max(0.5, best_l2 / 2.0), best_l2, best_l2 * 2.0]),
            }

        if self.method == "svm":
            best_c = float(best.get("svr__C", 1.0))
            best_eps = float(best.get("svr__epsilon", 0.1))
            best_gamma = best.get("svr__gamma", "scale")
            gamma_vals: list[Any]
            if isinstance(best_gamma, str):
                gamma_vals = _unique_sorted(
                    [
                        best_gamma,
                        "scale",
                        "auto",
                        round(max(1.0 / max(64.0, float(p)), 1e-6), 6),
                    ]
                )
            else:
                gamma_f = float(best_gamma)
                gamma_vals = _unique_sorted(
                    [
                        "scale",
                        round(max(gamma_f / 4.0, 1e-6), 6),
                        round(gamma_f, 6),
                        round(max(gamma_f * 4.0, 1e-6), 6),
                    ]
                )
            return {
                "svr__C": _unique_sorted(
                    [
                        max(0.25, round(best_c / 2.0, 4)),
                        round(best_c, 4),
                        round(best_c * 2.0, 4),
                        round(best_c * 4.0, 4),
                    ]
                ),
                "svr__epsilon": _unique_sorted(
                    [
                        max(0.001, round(best_eps / 2.0, 4)),
                        round(best_eps, 4),
                        round(min(0.5, best_eps * 2.0), 4),
                    ]
                ),
                "svr__gamma": gamma_vals,
            }

        if self.method == "enet":
            best_alpha = float(best.get("enet__alpha", 0.01))
            best_l1 = float(best.get("enet__l1_ratio", 0.5))
            best_tol = float(best.get("enet__tol", 1e-4))
            return {
                "enet__alpha": _unique_sorted(
                    [
                        max(1e-5, round(best_alpha / 5.0, 6)),
                        max(1e-5, round(best_alpha / 2.0, 6)),
                        round(best_alpha, 6),
                        round(best_alpha * 2.0, 6),
                        round(best_alpha * 5.0, 6),
                    ]
                ),
                "enet__l1_ratio": _unique_sorted(
                    [
                        max(0.01, round(best_l1 - 0.20, 3)),
                        max(0.01, round(best_l1 - 0.10, 3)),
                        round(best_l1, 3),
                        min(0.99, round(best_l1 + 0.10, 3)),
                        min(0.99, round(best_l1 + 0.20, 3)),
                    ]
                ),
                "enet__tol": _unique_sorted(
                    [1e-4, 5e-4, 1e-3, round(best_tol, 6)]
                ),
            }

        raise ValueError(f"Unsupported MLGS method: {self.method}")

    def _parameter_candidates(
        self,
        space: dict[str, list[Any]],
        n_iter: int,
        stage: str,
    ) -> list[dict[str, Any]]:
        if n_iter <= 0:
            return []
        rng = np.random.RandomState(self.seed + (17 if stage == "fine" else 0))
        candidates = list(ParameterSampler(space, n_iter=n_iter, random_state=rng))
        return candidates

    def _cv_splitter(self, random_state: int | None = None) -> KFold:
        n_splits = min(self.cv, self.sample_count_)
        if n_splits < 2:
            raise ValueError("At least 2 folds are required for MLGS cross-validation.")
        return KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.seed if random_state is None else int(random_state),
        )

    def _evaluate_candidate_once(
        self,
        params: dict[str, Any],
        stage: str,
        splitter_seed: int | None = None,
    ) -> dict[str, Any]:
        splitter = self._cv_splitter(random_state=splitter_seed)
        scores: list[float] = []
        pearsons: list[float] = []
        r2s: list[float] = []
        rmses: list[float] = []
        xgb_iters: list[int] = []

        for fold_id, (tr, va) in enumerate(splitter.split(self.X), start=1):
            est = self._build_estimator(params)
            X_tr = self.X[tr]
            y_tr = self.y[tr]
            X_va = self.X[va]
            y_va = self.y[va]

            with self._thread_context():
                if self.method == "xgb":
                    est.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_va, y_va)],
                        verbose=False,
                    )
                    best_iter = getattr(est, "best_iteration", None)
                    if best_iter is not None:
                        xgb_iters.append(int(best_iter) + 1)
                else:
                    est.fit(X_tr, y_tr)

                pred = np.asarray(est.predict(X_va), dtype=float).reshape(-1)
            pear = _safe_pearson(y_va, pred)
            r2 = _safe_r2(y_va, pred)
            rmse = _rmse(y_va, pred)
            score = _score_value(y_va, pred, self.scoring)

            scores.append(score)
            pearsons.append(pear)
            r2s.append(r2)
            rmses.append(rmse)

        result = {
            "stage": stage,
            "params": dict(params),
            "score": float(np.mean(scores)) if scores else float("-inf"),
            "pearson": float(np.mean(pearsons)) if pearsons else 0.0,
            "r2": float(np.mean(r2s)) if r2s else float("-inf"),
            "rmse": float(np.mean(rmses)) if rmses else float("inf"),
        }
        if xgb_iters:
            result["n_estimators"] = int(np.median(xgb_iters))
        return result

    def _evaluate_candidate(self, params: dict[str, Any], stage: str) -> dict[str, Any]:
        return self._evaluate_candidate_once(params, stage=stage, splitter_seed=self.seed)

    def _evaluate_confirm_candidate(self, params: dict[str, Any]) -> dict[str, Any]:
        rows = [
            self._evaluate_candidate_once(
                params,
                stage="confirm",
                splitter_seed=self.seed + 1009 + rep * 97,
            )
            for rep in range(self.confirm_repeats)
        ]
        result = {
            "stage": "confirm",
            "params": dict(params),
            "score": float(np.mean([x["score"] for x in rows])),
            "pearson": float(np.mean([x["pearson"] for x in rows])),
            "r2": float(np.mean([x["r2"] for x in rows])),
            "rmse": float(np.mean([x["rmse"] for x in rows])),
            "confirm_repeats": int(self.confirm_repeats),
        }
        xgb_iters = [int(x["n_estimators"]) for x in rows if "n_estimators" in x]
        if xgb_iters:
            result["n_estimators"] = int(np.median(xgb_iters))
        return result

    def _evaluate_candidates(
        self,
        candidates: list[dict[str, Any]],
        stage: str,
        evaluator: typing.Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        eval_fn = evaluator
        if eval_fn is None:
            eval_fn = lambda params: self._evaluate_candidate(params, stage=stage)
        if self.parallel_mode_ == "search" and self.n_jobs > 1 and len(candidates) > 1:
            return Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(eval_fn)(params)
                for params in candidates
            )
        return [eval_fn(params) for params in candidates]

    def fit(self) -> "MLGS":
        coarse_space = self._coarse_space()
        coarse_candidates = self._parameter_candidates(coarse_space, self.coarse_iter, "coarse")
        if not coarse_candidates:
            raise RuntimeError("No coarse-search candidate was generated.")

        results = self._evaluate_candidates(coarse_candidates, stage="coarse")
        if self.search_scheme == "legacy":
            best_coarse = max(results, key=self._result_sort_key)
            fine_space = self._fine_space(best_coarse["params"])
            fine_candidates = self._parameter_candidates(fine_space, self.fine_iter, "fine")
            fine_results = self._evaluate_candidates(fine_candidates, stage="fine")
            if fine_results:
                results.extend(fine_results)
                best_row = max(fine_results + [best_coarse], key=self._result_sort_key)
            else:
                best_row = best_coarse
        else:
            coarse_centers = self._top_result_rows(results, self.coarse_top_k)
            fine_candidates_all: list[dict[str, Any]] = []
            for center in coarse_centers:
                fine_space = self._fine_space(center["params"])
                fine_candidates_all.extend(
                    self._parameter_candidates(fine_space, self.fine_iter, "fine")
                )
            fine_candidates = self._dedup_candidates(fine_candidates_all)
            fine_results = self._evaluate_candidates(fine_candidates, stage="fine")
            if fine_results:
                results.extend(fine_results)
            confirm_centers = self._top_result_rows(results, self.confirm_top_k)
            confirm_results = self._evaluate_candidates(
                [dict(row["params"]) for row in confirm_centers],
                stage="confirm",
                evaluator=self._evaluate_confirm_candidate,
            )
            if confirm_results:
                results.extend(confirm_results)
                best_row = max(confirm_results, key=self._result_sort_key)
            elif fine_results:
                best_row = max(fine_results + coarse_centers, key=self._result_sort_key)
            else:
                best_row = max(coarse_centers, key=self._result_sort_key)

        self.cv_results_ = results
        self.best_params_ = dict(best_row["params"])
        self.best_score_ = float(best_row["score"])
        self.best_metrics_ = {
            "pearson": float(best_row["pearson"]),
            "r2": float(best_row["r2"]),
            "rmse": float(best_row["rmse"]),
        }
        self.best_n_estimators_ = (
            int(best_row["n_estimators"])
            if "n_estimators" in best_row and best_row["n_estimators"] is not None
            else None
        )

        final_params = dict(self.best_params_)
        if self.method == "xgb" and self.best_n_estimators_ is not None:
            final_params["n_estimators"] = int(max(50, self.best_n_estimators_))

        self.model = self._build_estimator(final_params)
        with self._thread_context():
            self.model.fit(self.X, self.y)
        return self

    def _prepare_predict_marker(self, M: np.ndarray) -> np.ndarray:
        arr = np.asarray(M, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"M must be 2D. got shape={arr.shape}")
        if arr.shape[1] == self.marker_count_:
            X = arr
        elif arr.shape[0] == self.marker_count_:
            X = arr.T
        else:
            raise ValueError(
                f"Marker number mismatch for prediction. "
                f"expected marker dimension={self.marker_count_}, got shape={arr.shape}"
            )
        return _apply_column_means(X, self.marker_means_)

    def _prepare_predict_cov(self, cov: np.ndarray | None, n_samples: int) -> np.ndarray | None:
        if self.cov_count_ == 0:
            if cov is not None:
                raise ValueError("Model was trained without covariates, but cov was provided for prediction.")
            return None
        if cov is None:
            raise ValueError("Model was trained with covariates. cov is required for prediction.")
        arr = _prepare_covariates(cov, n_samples)
        if arr is None:
            raise ValueError("cov is required for prediction.")
        if arr.shape[1] != self.cov_count_:
            raise ValueError(
                f"Covariate column mismatch for prediction. expected={self.cov_count_}, got={arr.shape[1]}"
            )
        if self.cov_means_ is None:
            return np.asarray(arr, dtype=np.float32)
        return _apply_column_means(np.asarray(arr, dtype=np.float32), self.cov_means_)

    def predict(self, M: np.ndarray, cov: np.ndarray | None = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MLGS model is not fitted. Call fit() first.")
        X_marker = self._prepare_predict_marker(M)
        X_cov = self._prepare_predict_cov(cov, X_marker.shape[0])
        X = self._merge_features(X_marker, X_cov)
        with self._thread_context():
            pred = np.asarray(self.model.predict(X), dtype=float).reshape(-1, 1)
        return pred

    def score(
        self,
        y_true: np.ndarray,
        M: np.ndarray | None = None,
        cov: np.ndarray | None = None,
        y_pred: np.ndarray | None = None,
        scoring: ScoreName | None = None,
    ) -> float:
        metric = self.scoring if scoring is None else scoring
        y_arr = _as_1d_y(y_true)
        if y_pred is None:
            if M is None:
                raise ValueError("M is required when y_pred is not provided.")
            pred = self.predict(M, cov=cov).reshape(-1)
        else:
            pred = np.asarray(y_pred, dtype=float).reshape(-1)
        return _score_value(y_arr, pred, metric)

    def search_summary(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.cv_results_]

    def fit_predict(self, M: np.ndarray | None = None, cov: np.ndarray | None = None) -> np.ndarray:
        if self.model is None:
            self.fit()
        if M is None:
            with self._thread_context():
                return np.asarray(self.model.predict(self.X), dtype=float).reshape(-1, 1)
        return self.predict(M, cov=cov)


__all__ = ["MLGS"]
