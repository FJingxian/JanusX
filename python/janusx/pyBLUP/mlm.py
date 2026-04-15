from typing import Union, Literal, Any, Optional
import os
import time
import numpy as np
from scipy.optimize import minimize_scalar
try:
    from scipy.linalg.blas import dgemm as _blas_dgemm
except Exception:
    _blas_dgemm = None
try:
    from scipy.linalg.blas import dsyrk as _blas_dsyrk
except Exception:
    _blas_dsyrk = None
try:
    from scipy.linalg import eigh as _scipy_eigh
except Exception:
    _scipy_eigh = None
from .QK2 import GRM

try:
    from janusx.janusx import (
        bed_packed_row_flip_mask as _bed_packed_row_flip_mask,
        bed_packed_decode_rows_f32 as _bed_packed_decode_rows_f32,
    )
except Exception:
    _bed_packed_row_flip_mask = None
    _bed_packed_decode_rows_f32 = None

try:
    from janusx.janusx import bed_packed_decode_stats_f64 as _bed_packed_decode_stats_f64
except Exception:
    _bed_packed_decode_stats_f64 = None

try:
    from janusx.janusx import bed_packed_fit_stats_f64 as _bed_packed_fit_stats_f64
except Exception:
    _bed_packed_fit_stats_f64 = None


def _env_truthy(name: str, default: str = "0") -> bool:
    v = str(os.getenv(name, default)).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _env_positive_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        v = int(raw)
    except Exception:
        return int(default)
    return max(1, v)


def _env_positive_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        v = float(raw)
    except Exception:
        return float(default)
    return max(float(np.finfo(np.float64).eps), v)


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    raw = str(os.getenv(name, default)).strip().lower()
    return raw if raw in choices else str(default).strip().lower()


def _detect_memory_limit_bytes() -> Optional[int]:
    cgroup_paths = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    for path in cgroup_paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read().strip()
        except Exception:
            continue
        if raw == "" or raw.lower() == "max":
            continue
        try:
            val = int(raw)
        except Exception:
            continue
        # Skip sentinel-style "unlimited" values.
        if val > 0 and val < (1 << 60):
            return val
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = pages * page_size
        if total > 0:
            return total
    except Exception:
        return None
    return None


def _coerce_bed_packed_ctx(M: Any) -> Optional[dict[str, Any]]:
    ctx_in: Optional[dict[str, Any]] = None
    if isinstance(M, dict):
        if "packed" in M and "maf" in M and "n_samples" in M:
            ctx_in = M
    elif isinstance(M, (tuple, list)) and len(M) >= 5:
        packed, miss, maf, _denom, n_samples = M[:5]
        ctx_in = {
            "packed": packed,
            "missing_rate": miss,
            "maf": maf,
            "n_samples": n_samples,
        }
    if ctx_in is None:
        return None
    packed = np.ascontiguousarray(np.asarray(ctx_in["packed"], dtype=np.uint8))
    if packed.ndim != 2:
        raise ValueError("packed must be 2D with shape (m, ceil(n_samples/4)).")
    maf = np.ascontiguousarray(np.asarray(ctx_in["maf"], dtype=np.float32).reshape(-1))
    if maf.shape[0] != packed.shape[0]:
        raise ValueError(f"maf length mismatch: got {maf.shape[0]}, expected {packed.shape[0]}")
    n_samples = int(ctx_in["n_samples"])
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0 in packed context.")
    exp_bps = (n_samples + 3) // 4
    if packed.shape[1] != exp_bps:
        raise ValueError(
            f"packed bytes_per_snp mismatch: got {packed.shape[1]}, expected {exp_bps}"
        )
    row_flip = ctx_in.get("row_flip", None)
    if row_flip is None:
        if _bed_packed_row_flip_mask is None:
            raise RuntimeError(
                "Rust extension missing bed_packed_row_flip_mask. Rebuild janusx extension."
            )
        row_flip = np.asarray(
            _bed_packed_row_flip_mask(packed, int(n_samples)),
            dtype=np.bool_,
        )
    row_flip = np.ascontiguousarray(np.asarray(row_flip, dtype=np.bool_).reshape(-1))
    if row_flip.shape[0] != packed.shape[0]:
        raise ValueError(
            f"row_flip length mismatch: got {row_flip.shape[0]}, expected {packed.shape[0]}"
        )
    ctx_out = {
        "packed": packed,
        "maf": maf,
        "n_samples": n_samples,
        "row_flip": row_flip,
    }
    if "missing_rate" in ctx_in:
        ctx_out["missing_rate"] = np.ascontiguousarray(
            np.asarray(ctx_in["missing_rate"], dtype=np.float32).reshape(-1)
        )
    if isinstance(M, dict):
        M.update(ctx_out)
        return M
    return ctx_out


def _normalize_sample_indices(
    sample_indices: Optional[np.ndarray],
    n_samples: int,
    n_target: int,
) -> np.ndarray:
    if sample_indices is None:
        if int(n_samples) != int(n_target):
            raise ValueError(
                f"Packed n_samples={n_samples} does not match target n={n_target}. "
                "Provide sample_indices to align samples."
            )
        return np.arange(int(n_samples), dtype=np.int64)
    sidx = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    if sidx.size != int(n_target):
        raise ValueError(
            f"sample_indices length mismatch: got {sidx.size}, expected {n_target}"
        )
    if np.any(sidx < 0) or np.any(sidx >= int(n_samples)):
        raise ValueError("sample_indices has out-of-range values.")
    return np.ascontiguousarray(sidx, dtype=np.int64)


def _decode_packed_rows_f32(
    packed_ctx: dict[str, Any],
    row_idx: np.ndarray,
    sample_indices: Optional[np.ndarray],
) -> np.ndarray:
    if _bed_packed_decode_rows_f32 is None:
        raise RuntimeError(
            "Rust extension missing bed_packed_decode_rows_f32. Rebuild janusx extension."
        )
    ridx = np.ascontiguousarray(np.asarray(row_idx, dtype=np.int64).reshape(-1))
    decoded = _bed_packed_decode_rows_f32(
        packed_ctx["packed"],
        int(packed_ctx["n_samples"]),
        ridx,
        packed_ctx["row_flip"],
        packed_ctx["maf"],
        sample_indices,
    )
    return np.ascontiguousarray(np.asarray(decoded, dtype=np.float32), dtype=np.float32)


def _gram_rankk_update(
    gram: np.ndarray,
    blk64: np.ndarray,
    *,
    lowmem: bool,
) -> np.ndarray:
    if (not lowmem) or (_blas_dgemm is None):
        if lowmem and (_blas_dsyrk is not None):
            return _blas_dsyrk(
                alpha=1.0,
                a=blk64,
                beta=1.0,
                c=gram,
                lower=1,
                trans=0,
                overwrite_c=1,
            )
        gram += blk64 @ blk64.T
        return gram
    if _blas_dsyrk is not None:
        return _blas_dsyrk(
            alpha=1.0,
            a=blk64,
            beta=1.0,
            c=gram,
            lower=1,
            trans=0,
            overwrite_c=1,
        )
    return _blas_dgemm(
        alpha=1.0,
        a=blk64,
        b=blk64,
        beta=1.0,
        c=gram,
        trans_b=1,
        overwrite_c=1,
    )


def _symmetric_eigh_full(
    a: np.ndarray,
    *,
    overwrite_a: bool,
    lowmem: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if _scipy_eigh is None:
        return np.linalg.eigh(a)
    if not lowmem:
        return _scipy_eigh(
            a,
            lower=True,
            overwrite_a=overwrite_a,
            check_finite=False,
            driver="evd",
        )
    return _scipy_eigh(
        a,
        lower=True,
        overwrite_a=overwrite_a,
        check_finite=False,
        driver="evr",
    )


def _estimate_fast_gram_peak_bytes(
    *,
    m: int,
    block_cols: int,
    resident_bytes: int,
) -> int:
    gram_bytes = int(m) * int(m) * np.dtype(np.float64).itemsize
    blk_f32_bytes = int(m) * int(block_cols) * np.dtype(np.float32).itemsize
    blk_f64_bytes = int(m) * int(block_cols) * np.dtype(np.float64).itemsize
    # Conservative estimate for the legacy fast path:
    # resident payload + gram itself + one full m x m temporary product +
    # eigendecomposition workspace/eigenvector footprint (~2x gram).
    return int(resident_bytes) + blk_f32_bytes + blk_f64_bytes + (4 * gram_bytes)


def _choose_gram_strategy(
    *,
    m: int,
    block_cols: int,
    resident_bytes: int,
) -> tuple[str, Optional[int], int]:
    mode = _env_choice("JX_MLM_GRAM_MODE", "auto", {"auto", "fast", "lowmem"})
    mem_limit = _detect_memory_limit_bytes()
    est_fast_peak = _estimate_fast_gram_peak_bytes(
        m=int(m),
        block_cols=int(block_cols),
        resident_bytes=int(resident_bytes),
    )
    if mode == "fast":
        return "fast", mem_limit, est_fast_peak
    if mode == "lowmem":
        return "lowmem", mem_limit, est_fast_peak
    frac = _env_positive_float("JX_MLM_GRAM_FAST_MEM_FRAC", 0.45)
    if mem_limit is not None and mem_limit > 0:
        budget = int(mem_limit * frac)
        return ("fast" if est_fast_peak <= budget else "lowmem"), mem_limit, est_fast_peak
    fallback_m = _env_positive_int("JX_MLM_GRAM_FAST_MAX_M", 20_000)
    return ("fast" if int(m) <= fallback_m else "lowmem"), mem_limit, est_fast_peak


class BLUP:
    def __init__(
        self,
        y: np.ndarray,
        M: Any,
        cov: Union[np.ndarray, None] = None,
        Z: Union[np.ndarray, None] = None,
        kinship: Literal[None, 1] = None,
        log: bool = False,
        sample_indices: Union[np.ndarray, None] = None,
    ):
        """
        Fast solution of the mixed linear model via Brent's method.

        Parameters
        ----------
        y : np.ndarray
            Phenotype vector of shape (n, 1).
        M : np.ndarray
            Marker matrix of shape (m, n) with genotypes coded as 0/1/2.
        cov : np.ndarray, optional
            Fixed-effect design matrix of shape (n, p).
        Z : np.ndarray, optional
            Random-effect design matrix of shape (n, q).
        kinship : {None, 1}
            Kinship specification; None disables kinship.
        """
        self.log = log
        self._debug_stage = _env_truthy("JX_MLM_DEBUG_STAGE", "0")
        self._reml_calls = 0
        t_init = time.time()
        self.y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        self._packed_ctx = _coerce_bed_packed_ctx(M)
        self._packed_sample_indices = None
        self._packed_all_rows = None
        self._packed_sample_chunk = _env_positive_int("JX_MLM_PACKED_SAMPLE_CHUNK", 8192)
        self.M = None
        if cov is not None:
            cov = np.asarray(cov, dtype=np.float64)
            if cov.ndim == 1:
                cov = cov.reshape(-1, 1)
            if cov.shape[0] != self.y.shape[0]:
                raise ValueError(
                    f"cov rows must match len(y). Got cov.shape={cov.shape}, len(y)={self.y.shape[0]}"
                )
        # Represent default random-effect design as implicit identity.
        z_is_identity = Z is None
        if not z_is_identity:
            Z = np.asarray(Z, dtype=np.float64)
            if Z.ndim != 2:
                raise ValueError("Z must be a 2D array.")
            if Z.shape[0] != self.y.shape[0] or Z.shape[1] != self.y.shape[0]:
                raise ValueError(
                    f"Z must be square with shape (n, n). Got Z.shape={Z.shape}, n={self.y.shape[0]}"
                )
        self.X = (
            np.concatenate([np.ones((self.y.shape[0], 1)), cov], axis=1)
            if cov is not None
            else np.ones((self.y.shape[0], 1))
        )  # Design matrix of 1st vector
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.kinship = kinship  # control method to calculate kinship matrix
        self._m_mean = None
        self._m_var_sum = None
        self._pred_chunk_size = 32_768
        self._alpha_sum = None
        self._m_alpha = None
        self._mean_sq = None
        self._mean_malpha = None
        self._g_eps = 1e-6
        self._implicit_kinship_fast = False
        self._fast_v = None
        self._fast_inv_s = None
        self._fast_svals = None
        self._fast_projected = False
        if self._packed_ctx is not None:
            m_packed = int(self._packed_ctx["packed"].shape[0])
            if m_packed < self.n:
                self._fit_with_packed(
                    Z=Z,
                    z_is_identity=z_is_identity,
                    sample_indices=sample_indices,
                    t_init=t_init,
                )
                return
            # Fallback: decode selected samples to dense and reuse the generic path.
            sidx_dense = _normalize_sample_indices(
                sample_indices,
                int(self._packed_ctx["n_samples"]),
                int(self.n),
            )
            all_rows_dense = np.arange(m_packed, dtype=np.int64)
            self.M = _decode_packed_rows_f32(self._packed_ctx, all_rows_dense, sidx_dense)
            self._packed_ctx = None
            self._packed_sample_indices = None
            self._packed_all_rows = None

        # Keep marker matrix in float32 by default to avoid a full float64 copy.
        if self.M is None:
            self.M = np.asarray(M)
        else:
            self.M = np.asarray(self.M)
        if not np.issubdtype(self.M.dtype, np.floating):
            self.M = self.M.astype(np.float32, copy=False)
        elif self.M.dtype != np.float32:
            self.M = self.M.astype(np.float32, copy=False)
        if self.M.ndim != 2:
            raise ValueError("M must be a 2D array with shape (m, n).")
        if self.M.shape[1] != self.y.shape[0]:
            raise ValueError(
                f"M must be (m, n) with n=len(y). Got M.shape={self.M.shape}, len(y)={self.y.shape[0]}"
            )

        usefastlmm = self.M.shape[0] < self.n  # use FaST-LMM if num of snp less than num of samples
        if self._debug_stage:
            print(
                f"[MLM-DEBUG] fit_start n={self.n} m={self.M.shape[0]} "
                f"dtype_M={self.M.dtype} kinship={self.kinship} usefast={usefastlmm}",
                flush=True,
            )
        if self.kinship is not None:
            t_k = time.time()
            self._m_mean = np.mean(self.M, axis=1, dtype=np.float32, keepdims=True)
            m_var = np.var(self.M, axis=1, dtype=np.float32, keepdims=True)
            self._m_var_sum = float(np.sum(m_var, dtype=np.float32))
            if self._m_var_sum <= 0.0:
                self._m_var_sum = 1e-12
            if usefastlmm and z_is_identity:
                # Low-rank kinship path:
                # avoid explicit n x n G and recover u/alpha from SVD spectrum.
                self._implicit_kinship_fast = True
                self.G = None
                self.Z = None
            else:
                self.G = GRM(self.M, log=self.log).astype(np.float32, copy=False)
                self.G += np.float32(self._g_eps) * np.eye(self.n, dtype=np.float32)  # Add regular item
                self.Z = np.eye(self.n, dtype=np.float64) if z_is_identity else Z
            if self._debug_stage:
                print(
                    f"[MLM-DEBUG] kinship_prep_done implicit_fast={self._implicit_kinship_fast} "
                    f"elapsed={time.time() - t_k:.3f}s",
                    flush=True,
                )
        else:
            # For rrBLUP (kinship=None), random-effect covariance is identity.
            # Keep it implicit to avoid allocating a dense m x m eye matrix.
            self.G = None
            self.Z = self.M.T if z_is_identity else Z @ self.M.T
        # Simplify inverse matrix
        self.Dh = None
        if usefastlmm and z_is_identity:
            # Memory-lean FaST projection:
            # do eigendecomposition on m x m Gram (M M^T) instead of SVD on n x m.
            # This avoids materializing U of size n x m for large n.
            t_svd = time.time()
            gram = self.M @ self.M.T
            gram_mode, mem_limit, est_fast_peak = _choose_gram_strategy(
                m=int(self.M.shape[0]),
                block_cols=0,
                resident_bytes=int(self.M.nbytes),
            )
            if self._debug_stage:
                print(
                    f"[MLM-DEBUG] fast_gram_start shape={gram.shape} dtype={gram.dtype}",
                    flush=True,
                )
                print(
                    f"[MLM-DEBUG] fast_gram_mode mode={gram_mode} "
                    f"est_fast_peak_gib={est_fast_peak / 1024**3:.3f} "
                    f"mem_limit_gib={((mem_limit / 1024**3) if mem_limit is not None else float('nan')):.3f}",
                    flush=True,
                )
            eigvals, eigvec = _symmetric_eigh_full(
                gram,
                overwrite_a=True,
                lowmem=(gram_mode == "lowmem"),
            )
            max_eval = float(eigvals[-1]) if eigvals.size > 0 else 0.0
            tol = np.finfo(np.float64).eps * max(1.0, max_eval) * max(1, self.M.shape[0])
            keep_start = int(np.searchsorted(eigvals, tol, side="right"))
            if keep_start >= eigvals.shape[0]:
                raise RuntimeError("Numerically singular marker matrix in FaST mode (all eigenvalues dropped).")
            eigvals = eigvals[keep_start:]
            eigvec = eigvec[:, keep_start:]
            del gram
            svals = np.sqrt(eigvals.astype(np.float64, copy=False))
            inv_s = 1.0 / svals
            mx = self.M @ self.X
            my = self.M @ self.y
            self.X = (eigvec.T @ mx) * inv_s[:, None]
            self.y = (eigvec.T @ my) * inv_s[:, None]
            self.S = svals ** 2
            self._fast_v = eigvec
            self._fast_inv_s = inv_s
            self._fast_svals = svals
            self._fast_projected = True
            if self.kinship is None:
                self.Z = None
            if self._debug_stage:
                print(
                    f"[MLM-DEBUG] fast_gram_done rank={svals.shape[0]} elapsed={time.time() - t_svd:.3f}s",
                    flush=True,
                )
        elif usefastlmm:
            t_svd = time.time()
            if self.kinship is None:
                # kinship-free model uses ZM^T as random-effect design.
                fast_input = self.Z
            elif self._implicit_kinship_fast:
                # For default Z=I, avoid materializing an n x n identity matrix.
                fast_input = self.M.T
            else:
                fast_input = self.M.T if z_is_identity else (Z @ self.M.T)
            if self._debug_stage:
                print(
                    f"[MLM-DEBUG] svd_start shape={fast_input.shape} dtype={fast_input.dtype}",
                    flush=True,
                )
            u, svals, vh = np.linalg.svd(fast_input, full_matrices=False)
            self.Dh = u.T
            self.S = svals.astype(np.float64, copy=False) ** 2
            if self._debug_stage:
                print(
                    f"[MLM-DEBUG] svd_done rank={svals.shape[0]} elapsed={time.time() - t_svd:.3f}s",
                    flush=True,
                )
        else:
            t_eig = time.time()
            if self.kinship is not None and z_is_identity:
                self.S, eigvec = np.linalg.eigh(self.G)
            elif self.kinship is None:
                self.S, eigvec = np.linalg.eigh(self.Z @ self.Z.T)
            else:
                self.S, eigvec = np.linalg.eigh(self.Z @ self.G @ self.Z.T)
            self.Dh = eigvec.T
            if self._debug_stage:
                print(
                    f"[MLM-DEBUG] eigh_done rank={self.S.shape[0]} elapsed={time.time() - t_eig:.3f}s",
                    flush=True,
                )
        if not self._fast_projected:
            assert self.Dh is not None
            self.X = self.Dh @ self.X
            self.y = self.Dh @ self.y
            if usefastlmm and self.kinship is None:
                # Dh @ (ZM^T) = diag(s) @ Vh from SVD: avoids one extra large GEMM.
                self.Z = svals[:, None] * vh
            elif self.kinship is not None and z_is_identity:
                self.Z = self.Dh
            else:
                self.Z = self.Dh @ self.Z
        t_opt = time.time()
        if self._debug_stage:
            print("[MLM-DEBUG] reml_opt_start", flush=True)
        result = minimize_scalar(lambda lbd: -self._REML(10**(lbd)), bounds=(-6, 6), method='bounded')  # minimize REML
        lbd = 10 ** result.x
        self._REML(lbd)
        if self._debug_stage:
            print(
                f"[MLM-DEBUG] reml_opt_done calls={self._reml_calls} "
                f"lambda={lbd:.6g} elapsed={time.time() - t_opt:.3f}s",
                flush=True,
            )
        if self.kinship is not None and self._implicit_kinship_fast:
            # In FaST low-rank mode:
            #   alpha = U * diag(1/(s^2+lambda)) * r_rot = Dh.T @ (V_inv * r)
            #   u     = U * diag(s^2 + eps) * alpha_rot
            vr = (self.V_inv.ravel() * self.r.ravel()).reshape(-1, 1)
            spectral = (self.S + self._g_eps).reshape(-1, 1)
            if self._fast_v is not None and self._fast_inv_s is not None:
                proj_alpha = self._fast_v @ (self._fast_inv_s[:, None] * vr)
                proj_u = self._fast_v @ (self._fast_inv_s[:, None] * spectral * vr)
                self.alpha = self.M.T @ proj_alpha
                self.u = self.M.T @ proj_u
            else:
                self.alpha = self.Dh.T @ vr
                self.u = self.Dh.T @ (spectral * vr)
        else:
            rhs = (self.V_inv.ravel() * self.r.ravel()).reshape(-1, 1)
            if self.kinship is None and self._fast_v is not None and self._fast_svals is not None:
                self.u = self._fast_v @ (self._fast_svals[:, None] * rhs)
            elif self.kinship is None:
                self.u = self.Z.T @ rhs
            else:
                self.u = self.G @ self.Z.T @ rhs
            self.alpha = np.linalg.solve(self.G, self.u) if self.kinship is not None else None
        if self.kinship is not None and self.alpha is not None:
            # Cache compact terms for prediction:
            # cross(M_new, M_train) @ alpha without materializing cross kernel.
            self._alpha_sum = float(np.sum(self.alpha, dtype=np.float64))
            self._m_alpha = self.M @ self.alpha
            self._mean_sq = float((self._m_mean.T @ self._m_mean)[0, 0]) if self._m_mean is not None else None
            self._mean_malpha = (
                float((self._m_mean.T @ self._m_alpha)[0, 0]) if self._m_mean is not None else None
            )
        sigma_g2 = self.rTV_invr / (self.n - self.p)
        sigma_e2 = lbd * sigma_g2
        if self.kinship is None:
            g = self.M.T @ self.u
            var_g = float(np.var(g, ddof=1))
        else:
            mean_s = float(np.mean(self.S))
            var_g = sigma_g2 * mean_s
        self.pve = var_g / (var_g + sigma_e2)
        if self._debug_stage:
            print(
                f"[MLM-DEBUG] fit_done pve={self.pve:.6f} elapsed={time.time() - t_init:.3f}s",
                flush=True,
            )

    def _fit_with_packed(
        self,
        *,
        Z: Union[np.ndarray, None],
        z_is_identity: bool,
        sample_indices: Union[np.ndarray, None],
        t_init: float,
    ) -> None:
        if self._packed_ctx is None:
            raise RuntimeError("Internal error: packed context is missing.")
        if not z_is_identity or Z is not None:
            raise ValueError("Packed BLUP currently only supports default random-effect design Z=I.")
        m = int(self._packed_ctx["packed"].shape[0])
        self._packed_sample_indices = _normalize_sample_indices(
            sample_indices,
            int(self._packed_ctx["n_samples"]),
            int(self.n),
        )
        self._packed_all_rows = np.arange(m, dtype=np.int64)
        usefastlmm = m < self.n
        if not usefastlmm:
            raise ValueError(
                f"Packed BLUP requires m < n for low-rank FaST mode. Got m={m}, n={self.n}."
            )
        if self._debug_stage:
            print(
                f"[MLM-DEBUG] fit_start n={self.n} m={m} dtype_M=packed kinship={self.kinship} usefast={usefastlmm}",
                flush=True,
            )
        t_k = time.time()
        sum_rows = np.zeros((m, 1), dtype=np.float64)
        sq_sum_rows = np.zeros((m, 1), dtype=np.float64)
        mx = np.zeros((m, self.p), dtype=np.float64)
        my = np.zeros((m, 1), dtype=np.float64)
        n_train = int(self._packed_sample_indices.shape[0])
        step = max(1, int(self._packed_sample_chunk))
        gram_mode, mem_limit, est_fast_peak = _choose_gram_strategy(
            m=m,
            block_cols=min(step, n_train),
            resident_bytes=int(self._packed_ctx["packed"].nbytes),
        )
        gram = np.zeros(
            (m, m),
            dtype=np.float64,
            order=("F" if gram_mode == "lowmem" else "C"),
        )
        packed_agg_mode = _env_choice(
            "JX_MLM_PACKED_AGG_MODE",
            "auto",
            {"auto", "python", "rust", "rust_decode", "rust_full"},
        )
        if packed_agg_mode == "rust":
            packed_agg_mode = "rust_decode"
        use_rust_packed_decode = bool(
            (gram_mode == "fast")
            and (_bed_packed_decode_stats_f64 is not None)
            and (packed_agg_mode in {"auto", "rust_decode"})
        )
        use_rust_packed_full = bool(
            (gram_mode == "fast")
            and (_bed_packed_fit_stats_f64 is not None)
            and (packed_agg_mode == "rust_full")
        )
        if packed_agg_mode == "rust_decode" and _bed_packed_decode_stats_f64 is None:
            raise RuntimeError(
                "JX_MLM_PACKED_AGG_MODE=rust_decode requires Rust extension function bed_packed_decode_stats_f64."
            )
        if packed_agg_mode == "rust_full" and _bed_packed_fit_stats_f64 is None:
            raise RuntimeError(
                "JX_MLM_PACKED_AGG_MODE=rust_full requires Rust extension function bed_packed_fit_stats_f64."
            )
        if self._debug_stage:
            print(
                f"[MLM-DEBUG] fast_gram_mode mode={gram_mode} "
                f"est_fast_peak_gib={est_fast_peak / 1024**3:.3f} "
                f"mem_limit_gib={((mem_limit / 1024**3) if mem_limit is not None else float('nan')):.3f}",
                flush=True,
            )
            print(
                f"[MLM-DEBUG] packed_agg mode={('rust_full' if use_rust_packed_full else ('rust_decode' if use_rust_packed_decode else 'python'))}",
                flush=True,
            )
        if use_rust_packed_full:
            sum_rows_raw, sq_sum_rows_raw, mx_raw, my_raw, gram_raw = _bed_packed_fit_stats_f64(
                self._packed_ctx["packed"],
                int(self._packed_ctx["n_samples"]),
                self._packed_ctx["row_flip"],
                self._packed_ctx["maf"],
                self._packed_sample_indices,
                self.X,
                self.y,
                sample_block=step,
                threads=0,
            )
            sum_rows = np.asarray(sum_rows_raw, dtype=np.float64).reshape(m, 1)
            sq_sum_rows = np.asarray(sq_sum_rows_raw, dtype=np.float64).reshape(m, 1)
            mx = np.asarray(mx_raw, dtype=np.float64)
            my = np.asarray(my_raw, dtype=np.float64)
            gram = np.asarray(gram_raw, dtype=np.float64, order="C")
        elif use_rust_packed_decode:
            for st in range(0, n_train, step):
                ed = min(st + step, n_train)
                sidx_blk = np.ascontiguousarray(self._packed_sample_indices[st:ed], dtype=np.int64)
                blk64_raw, sum_rows_raw, sq_sum_rows_raw = _bed_packed_decode_stats_f64(
                    self._packed_ctx["packed"],
                    int(self._packed_ctx["n_samples"]),
                    self._packed_ctx["row_flip"],
                    self._packed_ctx["maf"],
                    sidx_blk,
                    threads=0,
                )
                blk64 = np.asarray(blk64_raw, dtype=np.float64)
                sum_rows += np.asarray(sum_rows_raw, dtype=np.float64).reshape(m, 1)
                sq_sum_rows += np.asarray(sq_sum_rows_raw, dtype=np.float64).reshape(m, 1)
                mx += blk64 @ self.X[st:ed, :]
                my += blk64 @ self.y[st:ed, :]
                gram = _gram_rankk_update(gram, blk64, lowmem=False)
        else:
            for st in range(0, n_train, step):
                ed = min(st + step, n_train)
                sidx_blk = np.ascontiguousarray(self._packed_sample_indices[st:ed], dtype=np.int64)
                blk = _decode_packed_rows_f32(self._packed_ctx, self._packed_all_rows, sidx_blk)
                blk64 = (
                    np.asfortranarray(blk, dtype=np.float64)
                    if gram_mode == "lowmem"
                    else np.asarray(blk, dtype=np.float64)
                )
                sum_rows += np.sum(blk64, axis=1, keepdims=True)
                sq_sum_rows += np.sum(blk64 * blk64, axis=1, keepdims=True)
                mx += blk64 @ self.X[st:ed, :]
                my += blk64 @ self.y[st:ed, :]
                gram = _gram_rankk_update(
                    gram,
                    blk64,
                    lowmem=(gram_mode == "lowmem"),
                )

        self._m_mean = sum_rows / float(self.n)
        m_var = sq_sum_rows / float(self.n) - self._m_mean * self._m_mean
        m_var = np.maximum(m_var, 0.0)
        self._m_var_sum = float(np.sum(m_var, dtype=np.float64))
        if self._m_var_sum <= 0.0:
            self._m_var_sum = 1e-12
        self._implicit_kinship_fast = bool(self.kinship is not None)
        self.G = None
        self.Z = None
        if self._debug_stage:
            print(
                f"[MLM-DEBUG] kinship_prep_done implicit_fast={self._implicit_kinship_fast} "
                f"elapsed={time.time() - t_k:.3f}s",
                flush=True,
            )

        t_svd = time.time()
        if self._debug_stage:
            print(
                f"[MLM-DEBUG] fast_gram_start shape={gram.shape} dtype={gram.dtype}",
                flush=True,
            )
        eigvals, eigvec = _symmetric_eigh_full(
            gram,
            overwrite_a=True,
            lowmem=(gram_mode == "lowmem"),
        )
        max_eval = float(eigvals[-1]) if eigvals.size > 0 else 0.0
        tol = np.finfo(np.float64).eps * max(1.0, max_eval) * max(1, m)
        keep_start = int(np.searchsorted(eigvals, tol, side="right"))
        if keep_start >= eigvals.shape[0]:
            raise RuntimeError("Numerically singular marker matrix in packed FaST mode.")
        eigvals = eigvals[keep_start:]
        eigvec = eigvec[:, keep_start:]
        del gram
        svals = np.sqrt(eigvals.astype(np.float64, copy=False))
        inv_s = 1.0 / svals
        self.S = svals ** 2
        self._fast_v = eigvec
        self._fast_inv_s = inv_s
        self._fast_svals = svals
        self._fast_projected = True
        self.Dh = None
        self.X = (eigvec.T @ mx) * inv_s[:, None]
        self.y = (eigvec.T @ my) * inv_s[:, None]
        if self._debug_stage:
            print(
                f"[MLM-DEBUG] fast_gram_done rank={svals.shape[0]} elapsed={time.time() - t_svd:.3f}s",
                flush=True,
            )

        t_opt = time.time()
        if self._debug_stage:
            print("[MLM-DEBUG] reml_opt_start", flush=True)
        result = minimize_scalar(
            lambda lbd: -self._REML(10 ** (lbd)),
            bounds=(-6, 6),
            method="bounded",
        )
        lbd = 10 ** result.x
        self._REML(lbd)
        if self._debug_stage:
            print(
                f"[MLM-DEBUG] reml_opt_done calls={self._reml_calls} "
                f"lambda={lbd:.6g} elapsed={time.time() - t_opt:.3f}s",
                flush=True,
            )

        rhs = (self.V_inv.ravel() * self.r.ravel()).reshape(-1, 1)
        if self.kinship is not None:
            spectral = (self.S + self._g_eps).reshape(-1, 1)
            q_alpha = self._fast_v @ (self._fast_inv_s[:, None] * rhs)
            q_u = self._fast_v @ (self._fast_inv_s[:, None] * spectral * rhs)
            self.alpha = np.empty((self.n, 1), dtype=np.float64)
            self.u = np.empty((self.n, 1), dtype=np.float64)
            for st in range(0, n_train, step):
                ed = min(st + step, n_train)
                sidx_blk = np.ascontiguousarray(self._packed_sample_indices[st:ed], dtype=np.int64)
                blk = _decode_packed_rows_f32(self._packed_ctx, self._packed_all_rows, sidx_blk)
                blk64 = np.asarray(blk, dtype=np.float64)
                self.alpha[st:ed] = blk64.T @ q_alpha
                self.u[st:ed] = blk64.T @ q_u
            self._alpha_sum = float(np.sum(self.alpha, dtype=np.float64))
            self._m_alpha = self._fast_v @ (self._fast_svals[:, None] * rhs)
            self._mean_sq = float((self._m_mean.T @ self._m_mean)[0, 0])
            self._mean_malpha = float((self._m_mean.T @ self._m_alpha)[0, 0])
            var_g = float((self.rTV_invr / (self.n - self.p)) * np.mean(self.S))
        else:
            self.u = self._fast_v @ (self._fast_svals[:, None] * rhs)
            self.alpha = None
            g = np.empty((self.n, 1), dtype=np.float64)
            for st in range(0, n_train, step):
                ed = min(st + step, n_train)
                sidx_blk = np.ascontiguousarray(self._packed_sample_indices[st:ed], dtype=np.int64)
                blk = _decode_packed_rows_f32(self._packed_ctx, self._packed_all_rows, sidx_blk)
                blk64 = np.asarray(blk, dtype=np.float64)
                g[st:ed] = blk64.T @ self.u
            var_g = float(np.var(g, ddof=1))

        sigma_g2 = self.rTV_invr / (self.n - self.p)
        sigma_e2 = lbd * sigma_g2
        self.pve = var_g / (var_g + sigma_e2)
        if self._debug_stage:
            print(
                f"[MLM-DEBUG] fit_done pve={self.pve:.6f} elapsed={time.time() - t_init:.3f}s",
                flush=True,
            )

    def _cross_grm_times_alpha_packed(
        self,
        packed_ctx: dict[str, Any],
        sample_indices: np.ndarray,
    ) -> np.ndarray:
        if (
            self._m_mean is None
            or self._m_var_sum is None
            or self._m_alpha is None
            or self._alpha_sum is None
            or self._mean_sq is None
            or self._mean_malpha is None
        ):
            raise RuntimeError("cross GRM compact terms are not initialized.")
        m_expect = int(self._m_mean.shape[0])
        if int(packed_ctx["packed"].shape[0]) != m_expect:
            raise ValueError(
                f"Packed SNP count mismatch: got {packed_ctx['packed'].shape[0]}, expected {m_expect}"
            )
        n_new = int(sample_indices.shape[0])
        out = np.empty((n_new, 1), dtype=np.float64)
        step = max(1, int(self._packed_sample_chunk))
        all_rows = np.arange(m_expect, dtype=np.int64)
        if self._debug_stage:
            t_cross = time.time()
            print(
                f"[MLM-DEBUG] cross_grm_times_alpha_start n_new={n_new} chunk={step}",
                flush=True,
            )
        for st in range(0, n_new, step):
            ed = min(st + step, n_new)
            sidx_blk = np.ascontiguousarray(sample_indices[st:ed], dtype=np.int64)
            blk = _decode_packed_rows_f32(packed_ctx, all_rows, sidx_blk)
            blk64 = np.asarray(blk, dtype=np.float64)
            term1 = blk64.T @ self._m_alpha
            term2 = blk64.T @ self._m_mean
            out[st:ed] = (
                term1
                - term2 * self._alpha_sum
                - self._mean_malpha
                + self._mean_sq * self._alpha_sum
            ) / self._m_var_sum
        if self._debug_stage:
            print(
                f"[MLM-DEBUG] cross_grm_times_alpha_done elapsed={time.time() - t_cross:.3f}s",
                flush=True,
            )
        return out

    def _predict_marker_effect_packed(
        self,
        packed_ctx: dict[str, Any],
        sample_indices: np.ndarray,
    ) -> np.ndarray:
        if self.u is None:
            raise RuntimeError("Marker effects are not initialized.")
        m_expect = int(self.u.shape[0])
        if int(packed_ctx["packed"].shape[0]) != m_expect:
            raise ValueError(
                f"Packed SNP count mismatch: got {packed_ctx['packed'].shape[0]}, expected {m_expect}"
            )
        n_new = int(sample_indices.shape[0])
        out = np.empty((n_new, 1), dtype=np.float64)
        step = max(1, int(self._packed_sample_chunk))
        all_rows = np.arange(m_expect, dtype=np.int64)
        for st in range(0, n_new, step):
            ed = min(st + step, n_new)
            sidx_blk = np.ascontiguousarray(sample_indices[st:ed], dtype=np.int64)
            blk = _decode_packed_rows_f32(packed_ctx, all_rows, sidx_blk)
            out[st:ed] = np.asarray(blk, dtype=np.float64).T @ self.u
        return out

    def _REML(self, lbd: float):
        """Restricted maximum likelihood (REML) for the null model."""
        self._reml_calls += 1
        n, p_cov = self.X.shape
        p = p_cov
        V = self.S + lbd
        V_inv = 1 / V
        X_cov_snp = self.X
        XTV_invX = V_inv * X_cov_snp.T @ X_cov_snp
        XTV_invy = V_inv * X_cov_snp.T @ self.y
        beta = np.linalg.solve(XTV_invX, XTV_invy)
        r = self.y - X_cov_snp @ beta
        rTV_invr = (V_inv * r.T @ r)[0, 0]
        log_detV = np.sum(np.log(V))
        _, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
        total_log = (n - p) * np.log(rTV_invr) + log_detV + log_detXTV_invX  # log items
        c = (n - p) * (np.log(n - p) - 1 - np.log(2 * np.pi)) / 2  # Constant
        self.V_inv = V_inv
        self.rTV_invr = rTV_invr
        self.r = r
        self.beta = beta
        return c - total_log / 2

    def _cross_grm(self, M_new: np.ndarray) -> np.ndarray:
        if self._m_mean is None or self._m_var_sum is None:
            raise RuntimeError("cross GRM requested before kinship statistics are initialized.")
        # Use train-set mean/variance for test-set kernel to avoid leakage and
        # keep predict() consistent with the fitted GRM.
        cross = M_new.T @ self.M
        cross -= M_new.T @ self._m_mean
        cross -= self._m_mean.T @ self.M
        cross += float((self._m_mean.T @ self._m_mean)[0, 0])
        return cross / self._m_var_sum

    def _cross_grm_times_alpha(self, M_new: np.ndarray) -> np.ndarray:
        """
        Compute cross_GRM(M_new, M_train) @ alpha in chunks without building
        the full n_new x n_train cross kernel matrix.
        """
        if (
            self._m_mean is None
            or self._m_var_sum is None
            or self._m_alpha is None
            or self._alpha_sum is None
            or self._mean_sq is None
            or self._mean_malpha is None
        ):
            raise RuntimeError("cross GRM compact terms are not initialized.")
        n_new = M_new.shape[1]
        out = np.empty((n_new, 1), dtype=np.float64)
        step = max(1, int(self._pred_chunk_size))
        if self._debug_stage:
            t_cross = time.time()
            print(
                f"[MLM-DEBUG] cross_grm_times_alpha_start n_new={n_new} chunk={step}",
                flush=True,
            )
        # Formula:
        # cross @ alpha =
        # [M_new^T(M alpha) - (M_new^T m_mean) * sum(alpha)
        #  - (m_mean^T M alpha) + (m_mean^T m_mean) * sum(alpha)] / var_sum
        for st in range(0, n_new, step):
            ed = min(st + step, n_new)
            blk = M_new[:, st:ed]
            term1 = blk.T @ self._m_alpha
            term2 = blk.T @ self._m_mean
            out[st:ed] = (
                term1
                - term2 * self._alpha_sum
                - self._mean_malpha
                + self._mean_sq * self._alpha_sum
            ) / self._m_var_sum
        if self._debug_stage:
            print(
                f"[MLM-DEBUG] cross_grm_times_alpha_done elapsed={time.time() - t_cross:.3f}s",
                flush=True,
            )
        return out

    def predict(
        self,
        M: Any,
        cov: Union[np.ndarray, None] = None,
        sample_indices: Union[np.ndarray, None] = None,
    ):
        t_pred = time.time()
        packed_ctx = _coerce_bed_packed_ctx(M)
        if packed_ctx is not None:
            n_target = (
                int(cov.shape[0])
                if cov is not None
                else (
                    int(np.asarray(sample_indices, dtype=np.int64).reshape(-1).shape[0])
                    if sample_indices is not None
                    else int(packed_ctx["n_samples"])
                )
            )
            sidx = _normalize_sample_indices(
                sample_indices,
                int(packed_ctx["n_samples"]),
                n_target,
            )
            if cov is not None:
                cov = np.asarray(cov, dtype=np.float64)
                if cov.ndim == 1:
                    cov = cov.reshape(-1, 1)
                if cov.shape[0] != sidx.shape[0]:
                    raise ValueError(
                        f"cov rows must match selected sample count. Got cov.shape={cov.shape}, n={sidx.shape[0]}"
                    )
            X = (
                np.concatenate([np.ones((sidx.shape[0], 1)), cov], axis=1)
                if cov is not None
                else np.ones((sidx.shape[0], 1))
            )
            # Keep packed prediction chunked even when the model itself was fitted
            # from a dense training matrix. This avoids materializing the full
            # packed test set as a temporary dense matrix during prediction.
            if self.kinship is not None:
                rand_eff = self._cross_grm_times_alpha_packed(packed_ctx, sidx)
                out = X @ self.beta + rand_eff
            else:
                out = X @ self.beta + self._predict_marker_effect_packed(packed_ctx, sidx)
            if self._debug_stage:
                print(
                    f"[MLM-DEBUG] predict_done n_new={sidx.shape[0]} elapsed={time.time() - t_pred:.3f}s",
                    flush=True,
                )
            return out

        M = np.asarray(M, dtype=np.float32)
        if M.ndim != 2:
            raise ValueError("M must be a 2D array with shape (m, n_new).")
        if self.M is None:
            raise ValueError("Model was fitted with packed genotypes; provide packed payload for prediction.")
        if M.shape[0] != self.M.shape[0]:
            raise ValueError(
                f"M must have the same SNP rows as training data. Expected {self.M.shape[0]}, got {M.shape[0]}"
            )
        if cov is not None:
            cov = np.asarray(cov, dtype=np.float64)
            if cov.ndim == 1:
                cov = cov.reshape(-1, 1)
            if cov.shape[0] != M.shape[1]:
                raise ValueError(
                    f"cov rows must match M.shape[1]. Got cov.shape={cov.shape}, M.shape={M.shape}"
                )
        X = (
            np.concatenate([np.ones((M.shape[1], 1)), cov], axis=1)
            if cov is not None
            else np.ones((M.shape[1], 1))
        )
        if self.kinship is not None:
            rand_eff = self._cross_grm_times_alpha(M)
            out = X @ self.beta + rand_eff
        else:
            out = X @ self.beta + M.T @ self.u
        if self._debug_stage:
            print(
                f"[MLM-DEBUG] predict_done n_new={M.shape[1]} elapsed={time.time() - t_pred:.3f}s",
                flush=True,
            )
        return out
