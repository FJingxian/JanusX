use pyo3::prelude::*;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
use std::ffi::OsString;
#[cfg(target_os = "macos")]
use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) type CblasInt = std::os::raw::c_int;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_COL_MAJOR: CblasInt = 102;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_ROW_MAJOR: CblasInt = 101;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_NO_TRANS: CblasInt = 111;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_TRANS: CblasInt = 112;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_UPPER: CblasInt = 121;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_LOWER: CblasInt = 122;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_LEFT: CblasInt = 141;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_DIAG_NON_UNIT: CblasInt = 131;

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SgemmBackend {
    Accelerate,
    OpenBlas,
    Blas,
    Rust,
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
static SGEMM_BACKEND: OnceLock<SgemmBackend> = OnceLock::new();

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const HAS_OPENBLAS_BACKEND: bool = cfg!(any(all(feature = "blas-openblas", jx_openblas_available)));
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const HAS_ACCELERATE_BACKEND: bool = cfg!(all(
    target_os = "macos",
    not(all(feature = "blas-openblas", jx_openblas_available))
));
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const HAS_BLAS_BACKEND: bool = cfg!(all(
    target_os = "linux",
    jx_blas_available,
    not(all(feature = "blas-openblas", jx_openblas_available))
));

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn default_sgemm_backend() -> SgemmBackend {
    #[cfg(target_os = "macos")]
    {
        // macOS policy:
        // - If Accelerate backend is compiled in, use it by default for BLAS
        //   (GRM/GEMM/SYRK-heavy paths).
        // - Otherwise, fall back to OpenBLAS when this wheel/build is
        //   OpenBLAS-only.
        if HAS_ACCELERATE_BACKEND {
            return SgemmBackend::Accelerate;
        }
        if HAS_OPENBLAS_BACKEND {
            return SgemmBackend::OpenBlas;
        }
        return SgemmBackend::Rust;
    }
    #[cfg(target_os = "windows")]
    {
        if HAS_OPENBLAS_BACKEND {
            return SgemmBackend::OpenBlas;
        }
        return SgemmBackend::Rust;
    }
    #[cfg(target_os = "linux")]
    {
        if HAS_OPENBLAS_BACKEND {
            return SgemmBackend::OpenBlas;
        }
        if HAS_BLAS_BACKEND {
            return SgemmBackend::Blas;
        }
        return SgemmBackend::Rust;
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn resolve_sgemm_backend() -> SgemmBackend {
    let default_backend = default_sgemm_backend();
    let req = std::env::var("JX_RUST_BLAS_BACKEND")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "auto".to_string());
    match req.as_str() {
        "" | "auto" => default_backend,
        "openblas" => {
            if HAS_OPENBLAS_BACKEND {
                SgemmBackend::OpenBlas
            } else {
                default_backend
            }
        }
        "accelerate" => {
            if HAS_ACCELERATE_BACKEND {
                SgemmBackend::Accelerate
            } else {
                default_backend
            }
        }
        "blas" => {
            if HAS_BLAS_BACKEND {
                SgemmBackend::Blas
            } else {
                default_backend
            }
        }
        "rust" => SgemmBackend::Rust,
        _ => default_backend,
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn selected_sgemm_backend() -> SgemmBackend {
    *SGEMM_BACKEND.get_or_init(resolve_sgemm_backend)
}

#[cfg(all(
    target_os = "macos",
    not(all(feature = "blas-openblas", jx_openblas_available))
))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_accelerate(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    #[allow(dead_code)]
    fn cblas_dgemm_accelerate(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dtrsm"]
    fn cblas_dtrsm_accelerate(
        order: CblasInt,
        side: CblasInt,
        uplo: CblasInt,
        transa: CblasInt,
        diag: CblasInt,
        m: CblasInt,
        n: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *mut f64,
        ldb: CblasInt,
    );
    #[link_name = "cblas_ddot"]
    fn cblas_ddot_accelerate(
        n: CblasInt,
        x: *const f64,
        incx: CblasInt,
        y: *const f64,
        incy: CblasInt,
    ) -> f64;
    #[link_name = "cblas_daxpy"]
    fn cblas_daxpy_accelerate(
        n: CblasInt,
        alpha: f64,
        x: *const f64,
        incx: CblasInt,
        y: *mut f64,
        incy: CblasInt,
    );
    #[link_name = "cblas_ssyrk"]
    fn cblas_ssyrk_accelerate(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dsyrk"]
    fn cblas_dsyrk_accelerate(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd_accelerate(
        jobz: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        w: *mut f64,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    #[link_name = "dsyevr_"]
    fn lapack_dsyevr_accelerate(
        jobz: *const std::os::raw::c_char,
        range: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        vl: *const f64,
        vu: *const f64,
        il: *const CblasInt,
        iu: *const CblasInt,
        abstol: *const f64,
        m: *mut CblasInt,
        w: *mut f64,
        z: *mut f64,
        ldz: *const CblasInt,
        isuppz: *mut CblasInt,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
}

#[cfg(all(
    target_os = "linux",
    jx_blas_available,
    not(all(feature = "blas-openblas", jx_openblas_available))
))]
#[link(name = "blas")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_blas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_blas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dtrsm"]
    fn cblas_dtrsm_blas(
        order: CblasInt,
        side: CblasInt,
        uplo: CblasInt,
        transa: CblasInt,
        diag: CblasInt,
        m: CblasInt,
        n: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *mut f64,
        ldb: CblasInt,
    );
    #[link_name = "cblas_ddot"]
    fn cblas_ddot_blas(
        n: CblasInt,
        x: *const f64,
        incx: CblasInt,
        y: *const f64,
        incy: CblasInt,
    ) -> f64;
    #[link_name = "cblas_daxpy"]
    fn cblas_daxpy_blas(
        n: CblasInt,
        alpha: f64,
        x: *const f64,
        incx: CblasInt,
        y: *mut f64,
        incy: CblasInt,
    );
    #[link_name = "cblas_ssyrk"]
    fn cblas_ssyrk_blas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dsyrk"]
    fn cblas_dsyrk_blas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
}

#[cfg(all(
    target_os = "macos",
    feature = "blas-openblas",
    jx_openblas_available,
    not(jx_openblas_link_openblas0)
))]
#[link(name = "openblas")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dtrsm"]
    fn cblas_dtrsm_openblas(
        order: CblasInt,
        side: CblasInt,
        uplo: CblasInt,
        transa: CblasInt,
        diag: CblasInt,
        m: CblasInt,
        n: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *mut f64,
        ldb: CblasInt,
    );
    #[link_name = "cblas_ddot"]
    fn cblas_ddot_openblas(
        n: CblasInt,
        x: *const f64,
        incx: CblasInt,
        y: *const f64,
        incy: CblasInt,
    ) -> f64;
    #[link_name = "cblas_daxpy"]
    fn cblas_daxpy_openblas(
        n: CblasInt,
        alpha: f64,
        x: *const f64,
        incx: CblasInt,
        y: *mut f64,
        incy: CblasInt,
    );
    #[link_name = "cblas_ssyrk"]
    fn cblas_ssyrk_openblas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dsyrk"]
    fn cblas_dsyrk_openblas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[cfg(jx_openblas_lapack_available)]
    #[cfg(jx_openblas_lapack_available)]
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd_openblas(
        jobz: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        w: *mut f64,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    #[cfg(jx_openblas_lapack_available)]
    #[cfg(jx_openblas_lapack_available)]
    #[link_name = "dsyevr_"]
    fn lapack_dsyevr_openblas(
        jobz: *const std::os::raw::c_char,
        range: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        vl: *const f64,
        vu: *const f64,
        il: *const CblasInt,
        iu: *const CblasInt,
        abstol: *const f64,
        m: *mut CblasInt,
        w: *mut f64,
        z: *mut f64,
        ldz: *const CblasInt,
        isuppz: *mut CblasInt,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    fn openblas_set_num_threads(num_threads: CblasInt);
    fn openblas_get_num_threads() -> CblasInt;
}

#[cfg(all(
    target_os = "macos",
    feature = "blas-openblas",
    jx_openblas_available,
    jx_openblas_link_openblas0
))]
#[link(name = "openblas.0")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dtrsm"]
    fn cblas_dtrsm_openblas(
        order: CblasInt,
        side: CblasInt,
        uplo: CblasInt,
        transa: CblasInt,
        diag: CblasInt,
        m: CblasInt,
        n: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *mut f64,
        ldb: CblasInt,
    );
    #[link_name = "cblas_ddot"]
    fn cblas_ddot_openblas(
        n: CblasInt,
        x: *const f64,
        incx: CblasInt,
        y: *const f64,
        incy: CblasInt,
    ) -> f64;
    #[link_name = "cblas_daxpy"]
    fn cblas_daxpy_openblas(
        n: CblasInt,
        alpha: f64,
        x: *const f64,
        incx: CblasInt,
        y: *mut f64,
        incy: CblasInt,
    );
    #[link_name = "cblas_ssyrk"]
    fn cblas_ssyrk_openblas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dsyrk"]
    fn cblas_dsyrk_openblas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[cfg(jx_openblas_lapack_available)]
    #[cfg(jx_openblas_lapack_available)]
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd_openblas(
        jobz: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        w: *mut f64,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    #[cfg(jx_openblas_lapack_available)]
    #[cfg(jx_openblas_lapack_available)]
    #[link_name = "dsyevr_"]
    fn lapack_dsyevr_openblas(
        jobz: *const std::os::raw::c_char,
        range: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        vl: *const f64,
        vu: *const f64,
        il: *const CblasInt,
        iu: *const CblasInt,
        abstol: *const f64,
        m: *mut CblasInt,
        w: *mut f64,
        z: *mut f64,
        ldz: *const CblasInt,
        isuppz: *mut CblasInt,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    fn openblas_set_num_threads(num_threads: CblasInt);
    fn openblas_get_num_threads() -> CblasInt;
}

#[cfg(all(
    target_os = "linux",
    feature = "blas-openblas",
    jx_openblas_available,
    not(jx_openblas_link_openblas0)
))]
#[link(name = "openblas")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dtrsm"]
    fn cblas_dtrsm_openblas(
        order: CblasInt,
        side: CblasInt,
        uplo: CblasInt,
        transa: CblasInt,
        diag: CblasInt,
        m: CblasInt,
        n: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *mut f64,
        ldb: CblasInt,
    );
    #[link_name = "cblas_ddot"]
    fn cblas_ddot_openblas(
        n: CblasInt,
        x: *const f64,
        incx: CblasInt,
        y: *const f64,
        incy: CblasInt,
    ) -> f64;
    #[link_name = "cblas_daxpy"]
    fn cblas_daxpy_openblas(
        n: CblasInt,
        alpha: f64,
        x: *const f64,
        incx: CblasInt,
        y: *mut f64,
        incy: CblasInt,
    );
    #[link_name = "cblas_ssyrk"]
    fn cblas_ssyrk_openblas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dsyrk"]
    fn cblas_dsyrk_openblas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[cfg(jx_openblas_lapack_available)]
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd_openblas(
        jobz: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        w: *mut f64,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    #[cfg(jx_openblas_lapack_available)]
    #[link_name = "dsyevr_"]
    fn lapack_dsyevr_openblas(
        jobz: *const std::os::raw::c_char,
        range: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        vl: *const f64,
        vu: *const f64,
        il: *const CblasInt,
        iu: *const CblasInt,
        abstol: *const f64,
        m: *mut CblasInt,
        w: *mut f64,
        z: *mut f64,
        ldz: *const CblasInt,
        isuppz: *mut CblasInt,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    fn openblas_set_num_threads(num_threads: CblasInt);
    fn openblas_get_num_threads() -> CblasInt;
}

#[cfg(all(
    target_os = "linux",
    feature = "blas-openblas",
    jx_openblas_available,
    jx_openblas_link_openblas0
))]
#[link(name = ":libopenblas.so.0")]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dtrsm"]
    fn cblas_dtrsm_openblas(
        order: CblasInt,
        side: CblasInt,
        uplo: CblasInt,
        transa: CblasInt,
        diag: CblasInt,
        m: CblasInt,
        n: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *mut f64,
        ldb: CblasInt,
    );
    #[link_name = "cblas_ddot"]
    fn cblas_ddot_openblas(
        n: CblasInt,
        x: *const f64,
        incx: CblasInt,
        y: *const f64,
        incy: CblasInt,
    ) -> f64;
    #[link_name = "cblas_daxpy"]
    fn cblas_daxpy_openblas(
        n: CblasInt,
        alpha: f64,
        x: *const f64,
        incx: CblasInt,
        y: *mut f64,
        incy: CblasInt,
    );
    #[link_name = "cblas_ssyrk"]
    fn cblas_ssyrk_openblas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dsyrk"]
    fn cblas_dsyrk_openblas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[cfg(jx_openblas_lapack_available)]
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd_openblas(
        jobz: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        w: *mut f64,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    #[cfg(jx_openblas_lapack_available)]
    #[link_name = "dsyevr_"]
    fn lapack_dsyevr_openblas(
        jobz: *const std::os::raw::c_char,
        range: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        vl: *const f64,
        vu: *const f64,
        il: *const CblasInt,
        iu: *const CblasInt,
        abstol: *const f64,
        m: *mut CblasInt,
        w: *mut f64,
        z: *mut f64,
        ldz: *const CblasInt,
        isuppz: *mut CblasInt,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    fn openblas_set_num_threads(num_threads: CblasInt);
    fn openblas_get_num_threads() -> CblasInt;
}

#[cfg(all(
    target_os = "windows",
    feature = "blas-openblas",
    jx_openblas_available,
    not(jx_openblas_link_openblas_plain)
))]
#[cfg_attr(jx_openblas_static_link, link(name = "libopenblas", kind = "static"))]
#[cfg_attr(not(jx_openblas_static_link), link(name = "libopenblas"))]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dtrsm"]
    fn cblas_dtrsm_openblas(
        order: CblasInt,
        side: CblasInt,
        uplo: CblasInt,
        transa: CblasInt,
        diag: CblasInt,
        m: CblasInt,
        n: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *mut f64,
        ldb: CblasInt,
    );
    #[link_name = "cblas_ddot"]
    fn cblas_ddot_openblas(
        n: CblasInt,
        x: *const f64,
        incx: CblasInt,
        y: *const f64,
        incy: CblasInt,
    ) -> f64;
    #[link_name = "cblas_daxpy"]
    fn cblas_daxpy_openblas(
        n: CblasInt,
        alpha: f64,
        x: *const f64,
        incx: CblasInt,
        y: *mut f64,
        incy: CblasInt,
    );
    #[link_name = "cblas_ssyrk"]
    fn cblas_ssyrk_openblas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dsyrk"]
    fn cblas_dsyrk_openblas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[cfg(jx_openblas_lapack_available)]
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd_openblas(
        jobz: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        w: *mut f64,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    #[cfg(jx_openblas_lapack_available)]
    #[link_name = "dsyevr_"]
    fn lapack_dsyevr_openblas(
        jobz: *const std::os::raw::c_char,
        range: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        vl: *const f64,
        vu: *const f64,
        il: *const CblasInt,
        iu: *const CblasInt,
        abstol: *const f64,
        m: *mut CblasInt,
        w: *mut f64,
        z: *mut f64,
        ldz: *const CblasInt,
        isuppz: *mut CblasInt,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    fn openblas_set_num_threads(num_threads: CblasInt);
    fn openblas_get_num_threads() -> CblasInt;
}

#[cfg(all(
    target_os = "windows",
    feature = "blas-openblas",
    jx_openblas_available,
    jx_openblas_link_openblas_plain
))]
#[cfg_attr(jx_openblas_static_link, link(name = "openblas", kind = "static"))]
#[cfg_attr(not(jx_openblas_static_link), link(name = "openblas"))]
unsafe extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        b: *const f32,
        ldb: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dgemm"]
    fn cblas_dgemm_openblas(
        order: CblasInt,
        transa: CblasInt,
        transb: CblasInt,
        m: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *const f64,
        ldb: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dtrsm"]
    fn cblas_dtrsm_openblas(
        order: CblasInt,
        side: CblasInt,
        uplo: CblasInt,
        transa: CblasInt,
        diag: CblasInt,
        m: CblasInt,
        n: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        b: *mut f64,
        ldb: CblasInt,
    );
    #[link_name = "cblas_ddot"]
    fn cblas_ddot_openblas(
        n: CblasInt,
        x: *const f64,
        incx: CblasInt,
        y: *const f64,
        incy: CblasInt,
    ) -> f64;
    #[link_name = "cblas_daxpy"]
    fn cblas_daxpy_openblas(
        n: CblasInt,
        alpha: f64,
        x: *const f64,
        incx: CblasInt,
        y: *mut f64,
        incy: CblasInt,
    );
    #[link_name = "cblas_ssyrk"]
    fn cblas_ssyrk_openblas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f32,
        a: *const f32,
        lda: CblasInt,
        beta: f32,
        c: *mut f32,
        ldc: CblasInt,
    );
    #[link_name = "cblas_dsyrk"]
    fn cblas_dsyrk_openblas(
        order: CblasInt,
        uplo: CblasInt,
        trans: CblasInt,
        n: CblasInt,
        k: CblasInt,
        alpha: f64,
        a: *const f64,
        lda: CblasInt,
        beta: f64,
        c: *mut f64,
        ldc: CblasInt,
    );
    #[cfg(jx_openblas_lapack_available)]
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd_openblas(
        jobz: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        w: *mut f64,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    #[cfg(jx_openblas_lapack_available)]
    #[link_name = "dsyevr_"]
    fn lapack_dsyevr_openblas(
        jobz: *const std::os::raw::c_char,
        range: *const std::os::raw::c_char,
        uplo: *const std::os::raw::c_char,
        n: *const CblasInt,
        a: *mut f64,
        lda: *const CblasInt,
        vl: *const f64,
        vu: *const f64,
        il: *const CblasInt,
        iu: *const CblasInt,
        abstol: *const f64,
        m: *mut CblasInt,
        w: *mut f64,
        z: *mut f64,
        ldz: *const CblasInt,
        isuppz: *mut CblasInt,
        work: *mut f64,
        lwork: *const CblasInt,
        iwork: *mut CblasInt,
        liwork: *const CblasInt,
        info: *mut CblasInt,
    );
    fn openblas_set_num_threads(num_threads: CblasInt);
    fn openblas_get_num_threads() -> CblasInt;
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_sgemm_rust(
    order: CblasInt,
    transa: CblasInt,
    transb: CblasInt,
    m: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f32,
    a: *const f32,
    lda: CblasInt,
    b: *const f32,
    ldb: CblasInt,
    beta: f32,
    c: *mut f32,
    ldc: CblasInt,
) {
    assert_eq!(
        order, CBLAS_COL_MAJOR,
        "Rust SGEMM fallback expects column-major order"
    );
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let (lda, ldb, ldc) = (lda as usize, ldb as usize, ldc as usize);

    let a_cols = if transa == CBLAS_NO_TRANS { k } else { m };
    let b_cols = if transb == CBLAS_NO_TRANS { n } else { k };
    let a_slice = std::slice::from_raw_parts(a, lda.saturating_mul(a_cols));
    let b_slice = std::slice::from_raw_parts(b, ldb.saturating_mul(b_cols));
    let c_slice = std::slice::from_raw_parts_mut(c, ldc.saturating_mul(n));

    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0_f32;
            for p in 0..k {
                let av = if transa == CBLAS_NO_TRANS {
                    a_slice[row + p * lda]
                } else {
                    a_slice[p + row * lda]
                };
                let bv = if transb == CBLAS_NO_TRANS {
                    b_slice[p + col * ldb]
                } else {
                    b_slice[col + p * ldb]
                };
                acc += av * bv;
            }
            let idx = row + col * ldc;
            c_slice[idx] = alpha * acc + beta * c_slice[idx];
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn cblas_sgemm_dispatch(
    order: CblasInt,
    transa: CblasInt,
    transb: CblasInt,
    m: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f32,
    a: *const f32,
    lda: CblasInt,
    b: *const f32,
    ldb: CblasInt,
    beta: f32,
    c: *mut f32,
    ldc: CblasInt,
) {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(all(
                target_os = "macos",
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_sgemm_accelerate(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(all(feature = "blas-openblas", jx_openblas_available))]
            {
                cblas_sgemm_openblas(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::Blas => {
            #[cfg(all(
                target_os = "linux",
                jx_blas_available,
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_sgemm_blas(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::Rust => {
            cblas_sgemm_rust(
                order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            );
            return;
        }
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_sgemm_accelerate(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "windows"),
        feature = "blas-openblas",
        jx_openblas_available
    ))]
    {
        cblas_sgemm_openblas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        target_os = "linux",
        jx_blas_available,
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_sgemm_blas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        target_os = "linux",
        not(jx_blas_available),
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_sgemm_rust(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        target_os = "windows",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_sgemm_rust(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
#[allow(dead_code)]
fn checked_dgemm_matrix_span(
    order: CblasInt,
    rows: usize,
    columns: usize,
    leading_dimension: usize,
) -> Option<usize> {
    if rows == 0 || columns == 0 {
        return Some(0);
    }
    let minimum_leading_dimension = if order == CBLAS_ROW_MAJOR {
        columns
    } else if order == CBLAS_COL_MAJOR {
        rows
    } else {
        return None;
    };
    if leading_dimension < minimum_leading_dimension {
        return None;
    }
    let span = if order == CBLAS_ROW_MAJOR {
        (rows - 1)
            .checked_mul(leading_dimension)
            .and_then(|offset| offset.checked_add(columns))
    } else {
        (columns - 1)
            .checked_mul(leading_dimension)
            .and_then(|offset| offset.checked_add(rows))
    };
    span
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
#[allow(dead_code)]
unsafe fn cblas_dgemm_rust(
    order: CblasInt,
    transa: CblasInt,
    transb: CblasInt,
    m: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f64,
    a: *const f64,
    lda: CblasInt,
    b: *const f64,
    ldb: CblasInt,
    beta: f64,
    c: *mut f64,
    ldc: CblasInt,
) {
    assert!(
        m >= 0 && n >= 0 && k >= 0,
        "Rust DGEMM dimensions must be nonnegative"
    );
    assert!(
        lda >= 0 && ldb >= 0 && ldc >= 0,
        "Rust DGEMM leading dimensions must be nonnegative"
    );
    assert!(
        order == CBLAS_ROW_MAJOR || order == CBLAS_COL_MAJOR,
        "Rust DGEMM received an unsupported matrix order"
    );

    // CBLAS treats a zero-sized output as a no-op. Return before creating any
    // slices so null A/B/C pointers are harmless for this supported contract.
    if m == 0 || n == 0 {
        return;
    }

    let (m, n, k) = (m as usize, n as usize, k as usize);
    let (lda, ldb, ldc) = (lda as usize, ldb as usize, ldc as usize);

    let c_span =
        checked_dgemm_matrix_span(order, m, n, ldc).expect("Rust DGEMM C matrix span overflow");
    let c_slice = std::slice::from_raw_parts_mut(c, c_span);

    // BLAS specifies that A and B are not referenced when alpha is zero or K
    // is zero. In particular, beta=0 must also avoid reading C so NaN inputs
    // do not leak into the result.
    if alpha == 0.0 || k == 0 {
        for col in 0..n {
            for row in 0..m {
                let idx = if order == CBLAS_ROW_MAJOR {
                    row * ldc + col
                } else {
                    row + col * ldc
                };
                c_slice[idx] = if beta == 0.0 {
                    0.0
                } else {
                    beta * c_slice[idx]
                };
            }
        }
        return;
    }

    let (a_rows, a_columns) = if transa == CBLAS_NO_TRANS {
        (m, k)
    } else {
        (k, m)
    };
    let (b_rows, b_columns) = if transb == CBLAS_NO_TRANS {
        (k, n)
    } else {
        (n, k)
    };
    let a_span = checked_dgemm_matrix_span(order, a_rows, a_columns, lda)
        .expect("Rust DGEMM A matrix span overflow");
    let b_span = checked_dgemm_matrix_span(order, b_rows, b_columns, ldb)
        .expect("Rust DGEMM B matrix span overflow");
    let a_slice = std::slice::from_raw_parts(a, a_span);
    let b_slice = std::slice::from_raw_parts(b, b_span);

    if order == CBLAS_ROW_MAJOR {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0_f64;
                for p in 0..k {
                    let av = if transa == CBLAS_NO_TRANS {
                        a_slice[row * lda + p]
                    } else {
                        a_slice[p * lda + row]
                    };
                    let bv = if transb == CBLAS_NO_TRANS {
                        b_slice[p * ldb + col]
                    } else {
                        b_slice[col * ldb + p]
                    };
                    acc += av * bv;
                }
                let idx = row * ldc + col;
                c_slice[idx] = if beta == 0.0 {
                    alpha * acc
                } else {
                    alpha * acc + beta * c_slice[idx]
                };
            }
        }
        return;
    }

    assert_eq!(
        order, CBLAS_COL_MAJOR,
        "Rust DGEMM fallback expects row- or column-major order"
    );
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0_f64;
            for p in 0..k {
                let av = if transa == CBLAS_NO_TRANS {
                    a_slice[row + p * lda]
                } else {
                    a_slice[p + row * lda]
                };
                let bv = if transb == CBLAS_NO_TRANS {
                    b_slice[p + col * ldb]
                } else {
                    b_slice[col + p * ldb]
                };
                acc += av * bv;
            }
            let idx = row + col * ldc;
            c_slice[idx] = if beta == 0.0 {
                alpha * acc
            } else {
                alpha * acc + beta * c_slice[idx]
            };
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
#[allow(dead_code)]
pub(crate) unsafe fn cblas_dgemm_dispatch(
    order: CblasInt,
    transa: CblasInt,
    transb: CblasInt,
    m: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f64,
    a: *const f64,
    lda: CblasInt,
    b: *const f64,
    ldb: CblasInt,
    beta: f64,
    c: *mut f64,
    ldc: CblasInt,
) {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(all(
                target_os = "macos",
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_dgemm_accelerate(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(all(feature = "blas-openblas", jx_openblas_available))]
            {
                cblas_dgemm_openblas(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::Blas => {
            #[cfg(all(
                target_os = "linux",
                jx_blas_available,
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_dgemm_blas(
                    order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                );
                return;
            }
        }
        SgemmBackend::Rust => {
            cblas_dgemm_rust(
                order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            );
            return;
        }
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_dgemm_accelerate(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "windows"),
        feature = "blas-openblas",
        jx_openblas_available
    ))]
    {
        cblas_dgemm_openblas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        target_os = "linux",
        jx_blas_available,
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_dgemm_blas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        target_os = "linux",
        not(jx_blas_available),
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_dgemm_rust(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(all(
        target_os = "windows",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_dgemm_rust(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_dtrsm_rust(
    order: CblasInt,
    side: CblasInt,
    uplo: CblasInt,
    transa: CblasInt,
    diag: CblasInt,
    m: CblasInt,
    n: CblasInt,
    alpha: f64,
    a: *const f64,
    lda: CblasInt,
    b: *mut f64,
    ldb: CblasInt,
) {
    assert_eq!(
        side, CBLAS_LEFT,
        "Rust DTRSM fallback currently supports left-side solves only"
    );
    assert_eq!(
        uplo, CBLAS_LOWER,
        "Rust DTRSM fallback currently supports lower triangles only"
    );
    assert_eq!(
        diag, CBLAS_DIAG_NON_UNIT,
        "Rust DTRSM fallback expects non-unit diagonals"
    );
    assert!(order == CBLAS_ROW_MAJOR || order == CBLAS_COL_MAJOR);
    assert!(m >= 0 && n >= 0 && lda >= m);
    let rows = m as usize;
    let columns = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    assert!(if order == CBLAS_ROW_MAJOR {
        ldb >= columns
    } else {
        ldb >= rows
    });
    let a_slice = std::slice::from_raw_parts(a, rows.saturating_mul(lda));
    let b_slice = std::slice::from_raw_parts_mut(b, rows.saturating_mul(ldb));
    assert!(transa == CBLAS_NO_TRANS || transa == CBLAS_TRANS);
    if order == CBLAS_ROW_MAJOR {
        if transa == CBLAS_NO_TRANS {
            for column in 0..columns {
                for row in 0..rows {
                    let mut value = alpha * b_slice[row * ldb + column];
                    for pivot in 0..row {
                        value -= a_slice[row * lda + pivot] * b_slice[pivot * ldb + column];
                    }
                    b_slice[row * ldb + column] = value / a_slice[row * lda + row];
                }
            }
        } else {
            for column in 0..columns {
                for row in (0..rows).rev() {
                    let mut value = alpha * b_slice[row * ldb + column];
                    for pivot in (row + 1)..rows {
                        value -= a_slice[pivot * lda + row] * b_slice[pivot * ldb + column];
                    }
                    b_slice[row * ldb + column] = value / a_slice[row * lda + row];
                }
            }
        }
    } else if transa == CBLAS_NO_TRANS {
        for column in 0..columns {
            for row in 0..rows {
                let mut value = alpha * b_slice[row + column * ldb];
                for pivot in 0..row {
                    value -= a_slice[row + pivot * lda] * b_slice[pivot + column * ldb];
                }
                b_slice[row + column * ldb] = value / a_slice[row + row * lda];
            }
        }
    } else {
        for column in 0..columns {
            for row in (0..rows).rev() {
                let mut value = alpha * b_slice[row + column * ldb];
                for pivot in (row + 1)..rows {
                    value -= a_slice[pivot + row * lda] * b_slice[pivot + column * ldb];
                }
                b_slice[row + column * ldb] = value / a_slice[row + row * lda];
            }
        }
    }
}

/// Dispatch a row/column-major double-precision triangular solve to the
/// selected CBLAS backend.  The Rust fallback intentionally implements only
/// the lower, left-side, non-unit case used by the GRM interaction scanner;
/// unsupported combinations fail loudly instead of silently changing the
/// solve semantics.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
#[allow(dead_code)]
pub(crate) unsafe fn cblas_dtrsm_dispatch(
    order: CblasInt,
    side: CblasInt,
    uplo: CblasInt,
    transa: CblasInt,
    diag: CblasInt,
    m: CblasInt,
    n: CblasInt,
    alpha: f64,
    a: *const f64,
    lda: CblasInt,
    b: *mut f64,
    ldb: CblasInt,
) {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(all(
                target_os = "macos",
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_dtrsm_accelerate(
                    order, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb,
                );
                return;
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(all(feature = "blas-openblas", jx_openblas_available))]
            {
                cblas_dtrsm_openblas(order, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
                return;
            }
        }
        SgemmBackend::Blas => {
            #[cfg(all(
                target_os = "linux",
                jx_blas_available,
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_dtrsm_blas(order, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
                return;
            }
        }
        SgemmBackend::Rust => {
            cblas_dtrsm_rust(order, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
            return;
        }
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_dtrsm_accelerate(order, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        return;
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "windows"),
        feature = "blas-openblas",
        jx_openblas_available
    ))]
    {
        cblas_dtrsm_openblas(order, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        return;
    }
    #[cfg(all(
        target_os = "linux",
        jx_blas_available,
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_dtrsm_blas(order, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        return;
    }
    #[cfg(any(
        target_os = "windows",
        all(target_os = "linux", not(jx_blas_available))
    ))]
    {
        cblas_dtrsm_rust(order, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_ddot_rust(
    n: CblasInt,
    x: *const f64,
    incx: CblasInt,
    y: *const f64,
    incy: CblasInt,
) -> f64 {
    if n <= 0 {
        return 0.0_f64;
    }
    assert!(
        incx > 0 && incy > 0,
        "Rust DDOT fallback expects positive increments"
    );
    let n_usize = n as usize;
    let incx_usize = incx as usize;
    let incy_usize = incy as usize;
    if incx_usize == 1 && incy_usize == 1 {
        let xs = std::slice::from_raw_parts(x, n_usize);
        let ys = std::slice::from_raw_parts(y, n_usize);
        xs.iter()
            .zip(ys.iter())
            .map(|(a, b)| (*a) * (*b))
            .sum::<f64>()
    } else {
        let xs = std::slice::from_raw_parts(x, n_usize.saturating_mul(incx_usize));
        let ys = std::slice::from_raw_parts(y, n_usize.saturating_mul(incy_usize));
        let mut acc = 0.0_f64;
        for i in 0..n_usize {
            acc += xs[i * incx_usize] * ys[i * incy_usize];
        }
        acc
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn cblas_ddot_dispatch(
    n: CblasInt,
    x: *const f64,
    incx: CblasInt,
    y: *const f64,
    incy: CblasInt,
) -> f64 {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(all(
                target_os = "macos",
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                return cblas_ddot_accelerate(n, x, incx, y, incy);
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(all(feature = "blas-openblas", jx_openblas_available))]
            {
                return cblas_ddot_openblas(n, x, incx, y, incy);
            }
        }
        SgemmBackend::Blas => {
            #[cfg(all(
                target_os = "linux",
                jx_blas_available,
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                return cblas_ddot_blas(n, x, incx, y, incy);
            }
        }
        SgemmBackend::Rust => {
            return cblas_ddot_rust(n, x, incx, y, incy);
        }
    };

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        return cblas_ddot_accelerate(n, x, incx, y, incy);
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "windows"),
        feature = "blas-openblas",
        jx_openblas_available
    ))]
    {
        return cblas_ddot_openblas(n, x, incx, y, incy);
    }
    #[cfg(all(
        target_os = "linux",
        jx_blas_available,
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        return cblas_ddot_blas(n, x, incx, y, incy);
    }
    #[cfg(all(
        target_os = "linux",
        not(jx_blas_available),
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        return cblas_ddot_rust(n, x, incx, y, incy);
    }
    #[cfg(all(
        target_os = "windows",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        return cblas_ddot_rust(n, x, incx, y, incy);
    }

    #[cfg(all(
        target_os = "macos",
        all(feature = "blas-openblas", jx_openblas_available)
    ))]
    {
        // In OpenBLAS-only macOS builds, all compile-time fallback blocks above
        // are intentionally unavailable; keep a concrete return path.
        return cblas_ddot_rust(n, x, incx, y, incy);
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_daxpy_rust(
    n: CblasInt,
    alpha: f64,
    x: *const f64,
    incx: CblasInt,
    y: *mut f64,
    incy: CblasInt,
) {
    if n <= 0 || alpha == 0.0_f64 {
        return;
    }
    assert!(
        incx > 0 && incy > 0,
        "Rust DAXPY fallback expects positive increments"
    );
    let n_usize = n as usize;
    let incx_usize = incx as usize;
    let incy_usize = incy as usize;
    if incx_usize == 1 && incy_usize == 1 {
        let xs = std::slice::from_raw_parts(x, n_usize);
        let ys = std::slice::from_raw_parts_mut(y, n_usize);
        for i in 0..n_usize {
            ys[i] = alpha.mul_add(xs[i], ys[i]);
        }
    } else {
        let xs = std::slice::from_raw_parts(x, n_usize.saturating_mul(incx_usize));
        let ys = std::slice::from_raw_parts_mut(y, n_usize.saturating_mul(incy_usize));
        for i in 0..n_usize {
            let yi = i * incy_usize;
            ys[yi] = alpha.mul_add(xs[i * incx_usize], ys[yi]);
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn cblas_daxpy_dispatch(
    n: CblasInt,
    alpha: f64,
    x: *const f64,
    incx: CblasInt,
    y: *mut f64,
    incy: CblasInt,
) {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(all(
                target_os = "macos",
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_daxpy_accelerate(n, alpha, x, incx, y, incy);
                return;
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(all(feature = "blas-openblas", jx_openblas_available))]
            {
                cblas_daxpy_openblas(n, alpha, x, incx, y, incy);
                return;
            }
        }
        SgemmBackend::Blas => {
            #[cfg(all(
                target_os = "linux",
                jx_blas_available,
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_daxpy_blas(n, alpha, x, incx, y, incy);
                return;
            }
        }
        SgemmBackend::Rust => {
            cblas_daxpy_rust(n, alpha, x, incx, y, incy);
            return;
        }
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_daxpy_accelerate(n, alpha, x, incx, y, incy);
        return;
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "windows"),
        feature = "blas-openblas",
        jx_openblas_available
    ))]
    {
        cblas_daxpy_openblas(n, alpha, x, incx, y, incy);
        return;
    }
    #[cfg(all(
        target_os = "linux",
        jx_blas_available,
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_daxpy_blas(n, alpha, x, incx, y, incy);
        return;
    }
    #[cfg(all(
        target_os = "linux",
        not(jx_blas_available),
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_daxpy_rust(n, alpha, x, incx, y, incy);
        return;
    }
    #[cfg(all(
        target_os = "windows",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_daxpy_rust(n, alpha, x, incx, y, incy);
        return;
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_ssyrk_rust(
    order: CblasInt,
    uplo: CblasInt,
    trans: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f32,
    a: *const f32,
    lda: CblasInt,
    beta: f32,
    c: *mut f32,
    ldc: CblasInt,
) {
    assert_eq!(
        order, CBLAS_COL_MAJOR,
        "Rust SSYRK fallback expects column-major order"
    );
    let (n, k) = (n as usize, k as usize);
    let (lda, ldc) = (lda as usize, ldc as usize);
    let a_cols = if trans == CBLAS_NO_TRANS { k } else { n };
    let a_slice = std::slice::from_raw_parts(a, lda.saturating_mul(a_cols));
    let c_slice = std::slice::from_raw_parts_mut(c, ldc.saturating_mul(n));

    for col in 0..n {
        let row_start = if uplo == CBLAS_UPPER { 0 } else { col };
        let row_end = if uplo == CBLAS_UPPER { col + 1 } else { n };
        for row in row_start..row_end {
            let mut acc = 0.0_f32;
            for p in 0..k {
                let av = if trans == CBLAS_NO_TRANS {
                    a_slice[row + p * lda]
                } else {
                    a_slice[p + row * lda]
                };
                let bv = if trans == CBLAS_NO_TRANS {
                    a_slice[col + p * lda]
                } else {
                    a_slice[p + col * lda]
                };
                acc += av * bv;
            }
            let idx = row + col * ldc;
            c_slice[idx] = alpha * acc + beta * c_slice[idx];
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn cblas_ssyrk_dispatch(
    order: CblasInt,
    uplo: CblasInt,
    trans: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f32,
    a: *const f32,
    lda: CblasInt,
    beta: f32,
    c: *mut f32,
    ldc: CblasInt,
) {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(all(
                target_os = "macos",
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_ssyrk_accelerate(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
                return;
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(all(feature = "blas-openblas", jx_openblas_available))]
            {
                cblas_ssyrk_openblas(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
                return;
            }
        }
        SgemmBackend::Blas => {
            #[cfg(all(
                target_os = "linux",
                jx_blas_available,
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_ssyrk_blas(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
                return;
            }
        }
        SgemmBackend::Rust => {
            cblas_ssyrk_rust(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
            return;
        }
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_ssyrk_accelerate(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "windows"),
        feature = "blas-openblas",
        jx_openblas_available
    ))]
    {
        cblas_ssyrk_openblas(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
    #[cfg(all(
        target_os = "linux",
        jx_blas_available,
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_ssyrk_blas(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
    #[cfg(all(
        target_os = "linux",
        not(jx_blas_available),
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_ssyrk_rust(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
    #[cfg(all(
        target_os = "windows",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_ssyrk_rust(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
unsafe fn cblas_dsyrk_rust(
    order: CblasInt,
    uplo: CblasInt,
    trans: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f64,
    a: *const f64,
    lda: CblasInt,
    beta: f64,
    c: *mut f64,
    ldc: CblasInt,
) {
    assert_eq!(
        order, CBLAS_COL_MAJOR,
        "Rust DSYRK fallback expects column-major order"
    );
    let (n, k) = (n as usize, k as usize);
    let (lda, ldc) = (lda as usize, ldc as usize);
    let a_cols = if trans == CBLAS_NO_TRANS { k } else { n };
    let a_slice = std::slice::from_raw_parts(a, lda.saturating_mul(a_cols));
    let c_slice = std::slice::from_raw_parts_mut(c, ldc.saturating_mul(n));

    for col in 0..n {
        let row_start = if uplo == CBLAS_UPPER { 0 } else { col };
        let row_end = if uplo == CBLAS_UPPER { col + 1 } else { n };
        for row in row_start..row_end {
            let mut acc = 0.0_f64;
            for p in 0..k {
                let av = if trans == CBLAS_NO_TRANS {
                    a_slice[row + p * lda]
                } else {
                    a_slice[p + row * lda]
                };
                let bv = if trans == CBLAS_NO_TRANS {
                    a_slice[col + p * lda]
                } else {
                    a_slice[p + col * lda]
                };
                acc += av * bv;
            }
            let idx = row + col * ldc;
            c_slice[idx] = alpha * acc + beta * c_slice[idx];
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn cblas_dsyrk_dispatch(
    order: CblasInt,
    uplo: CblasInt,
    trans: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: f64,
    a: *const f64,
    lda: CblasInt,
    beta: f64,
    c: *mut f64,
    ldc: CblasInt,
) {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(all(
                target_os = "macos",
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_dsyrk_accelerate(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
                return;
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(all(feature = "blas-openblas", jx_openblas_available))]
            {
                cblas_dsyrk_openblas(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
                return;
            }
        }
        SgemmBackend::Blas => {
            #[cfg(all(
                target_os = "linux",
                jx_blas_available,
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                cblas_dsyrk_blas(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
                return;
            }
        }
        SgemmBackend::Rust => {
            cblas_dsyrk_rust(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
            return;
        }
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_dsyrk_accelerate(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "windows"),
        feature = "blas-openblas",
        jx_openblas_available
    ))]
    {
        cblas_dsyrk_openblas(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
    #[cfg(all(
        target_os = "linux",
        jx_blas_available,
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_dsyrk_blas(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
    #[cfg(all(
        target_os = "linux",
        not(jx_blas_available),
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_dsyrk_rust(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
    #[cfg(all(
        target_os = "windows",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        cblas_dsyrk_rust(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn lapack_dsyevd_dispatch(
    jobz: *const std::os::raw::c_char,
    uplo: *const std::os::raw::c_char,
    n: *const CblasInt,
    a: *mut f64,
    lda: *const CblasInt,
    w: *mut f64,
    work: *mut f64,
    lwork: *const CblasInt,
    iwork: *mut CblasInt,
    liwork: *const CblasInt,
    info: *mut CblasInt,
) -> Result<(), &'static str> {
    #[cfg(target_os = "macos")]
    {
        if should_try_openblas_lapack_on_macos() {
            if let Some(ob) = openblas_lapack_dyn() {
                let (prev_threads, prev_omp_threads) =
                    openblas_lapack_dyn_threads_guard(ob, preferred_openblas_thread_cap());
                (ob.dsyevd)(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
                openblas_lapack_dyn_threads_restore(ob, prev_threads, prev_omp_threads);
                return Ok(());
            }
            if matches!(mac_eigh_lapack_pref(), MacEighLapackPref::OpenBlas) {
                return Err("macOS OpenBLAS LAPACK requested but dynamic OpenBLAS loader failed");
            }
        }
    }

    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(all(
                target_os = "macos",
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                lapack_dsyevd_accelerate(
                    jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info,
                );
                return Ok(());
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(all(
                any(
                    target_os = "macos",
                    target_os = "linux",
                    all(target_os = "windows", jx_openblas_lapack_available)
                ),
                feature = "blas-openblas",
                jx_openblas_available,
                jx_openblas_lapack_available
            ))]
            {
                lapack_dsyevd_openblas(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
                return Ok(());
            }
        }
        SgemmBackend::Blas | SgemmBackend::Rust => {}
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        lapack_dsyevd_accelerate(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
        return Ok(());
    }
    #[cfg(all(
        any(
            target_os = "linux",
            all(target_os = "windows", jx_openblas_lapack_available)
        ),
        feature = "blas-openblas",
        jx_openblas_available,
        jx_openblas_lapack_available
    ))]
    {
        lapack_dsyevd_openblas(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
        return Ok(());
    }
    #[cfg(all(
        target_os = "windows",
        not(all(
            feature = "blas-openblas",
            jx_openblas_available,
            jx_openblas_lapack_available
        ))
    ))]
    {
        let _ = (jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
        return Err("lapack_dsyevd unavailable on this Windows build");
    }
    #[allow(unreachable_code)]
    Err("lapack_dsyevd backend unavailable")
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) unsafe fn lapack_dsyevr_dispatch(
    jobz: *const std::os::raw::c_char,
    range: *const std::os::raw::c_char,
    uplo: *const std::os::raw::c_char,
    n: *const CblasInt,
    a: *mut f64,
    lda: *const CblasInt,
    vl: *const f64,
    vu: *const f64,
    il: *const CblasInt,
    iu: *const CblasInt,
    abstol: *const f64,
    m: *mut CblasInt,
    w: *mut f64,
    z: *mut f64,
    ldz: *const CblasInt,
    isuppz: *mut CblasInt,
    work: *mut f64,
    lwork: *const CblasInt,
    iwork: *mut CblasInt,
    liwork: *const CblasInt,
    info: *mut CblasInt,
) -> Result<(), &'static str> {
    #[cfg(target_os = "macos")]
    {
        if should_try_openblas_lapack_on_macos() {
            if let Some(ob) = openblas_lapack_dyn() {
                let (prev_threads, prev_omp_threads) =
                    openblas_lapack_dyn_threads_guard(ob, preferred_openblas_thread_cap());
                (ob.dsyevr)(
                    jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz,
                    work, lwork, iwork, liwork, info,
                );
                openblas_lapack_dyn_threads_restore(ob, prev_threads, prev_omp_threads);
                return Ok(());
            }
            if matches!(mac_eigh_lapack_pref(), MacEighLapackPref::OpenBlas) {
                return Err("macOS OpenBLAS LAPACK requested but dynamic OpenBLAS loader failed");
            }
        }
    }

    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => {
            #[cfg(all(
                target_os = "macos",
                not(all(feature = "blas-openblas", jx_openblas_available))
            ))]
            {
                lapack_dsyevr_accelerate(
                    jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz,
                    work, lwork, iwork, liwork, info,
                );
                return Ok(());
            }
        }
        SgemmBackend::OpenBlas => {
            #[cfg(all(
                any(
                    target_os = "macos",
                    target_os = "linux",
                    all(target_os = "windows", jx_openblas_lapack_available)
                ),
                feature = "blas-openblas",
                jx_openblas_available,
                jx_openblas_lapack_available
            ))]
            {
                lapack_dsyevr_openblas(
                    jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz,
                    work, lwork, iwork, liwork, info,
                );
                return Ok(());
            }
        }
        SgemmBackend::Blas | SgemmBackend::Rust => {}
    }

    #[cfg(all(
        target_os = "macos",
        not(all(feature = "blas-openblas", jx_openblas_available))
    ))]
    {
        lapack_dsyevr_accelerate(
            jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work,
            lwork, iwork, liwork, info,
        );
        return Ok(());
    }
    #[cfg(all(
        any(
            target_os = "linux",
            all(target_os = "windows", jx_openblas_lapack_available)
        ),
        feature = "blas-openblas",
        jx_openblas_available,
        jx_openblas_lapack_available
    ))]
    {
        lapack_dsyevr_openblas(
            jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work,
            lwork, iwork, liwork, info,
        );
        return Ok(());
    }
    #[cfg(all(
        target_os = "windows",
        not(all(
            feature = "blas-openblas",
            jx_openblas_available,
            jx_openblas_lapack_available
        ))
    ))]
    {
        let _ = (
            jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work,
            lwork, iwork, liwork, info,
        );
        return Err("lapack_dsyevr unavailable on this Windows build");
    }
    #[allow(unreachable_code)]
    Err("lapack_dsyevr backend unavailable")
}

#[pyfunction]
pub fn rust_sgemm_backend() -> String {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        rust_sgemm_backend_tag().to_string()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "unsupported".to_string()
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) fn rust_sgemm_backend_tag() -> &'static str {
    match selected_sgemm_backend() {
        SgemmBackend::Accelerate => "accelerate",
        SgemmBackend::OpenBlas => "openblas",
        SgemmBackend::Blas => "blas",
        SgemmBackend::Rust => "rust",
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) fn rust_sgemm_prefers_rayon_rowmajor_f32_kernel() -> bool {
    if let Ok(raw) = std::env::var("JX_ROWMAJOR_F32_KERNEL") {
        let norm = raw.trim().to_ascii_lowercase();
        match norm.as_str() {
            "rayon" | "parallel" | "custom" => return true,
            "blas" | "gemm" | "serial" => return false,
            _ => {}
        }
    }
    #[cfg(target_os = "windows")]
    {
        return match selected_sgemm_backend() {
            // Windows benchmarks now favor the custom Rayon row-major kernel
            // as the default path for the current HE/PCG access pattern.
            SgemmBackend::OpenBlas => true,
            SgemmBackend::Accelerate => false,
            SgemmBackend::Blas => true,
            SgemmBackend::Rust => true,
        };
    }

    #[cfg(not(target_os = "windows"))]
    {
        return match selected_sgemm_backend() {
            // Accelerate-backed dense BLAS is still the safer default on macOS
            // until HE/PCG kernels are tuned more aggressively there.
            SgemmBackend::Accelerate => false,
            // Linux/OpenBLAS HE/PCG benchmarks consistently favor the custom
            // Rayon row-major kernel for the current access pattern.
            SgemmBackend::OpenBlas => true,
            SgemmBackend::Blas => false,
            SgemmBackend::Rust => true,
        };
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
#[inline]
pub(crate) fn rust_sgemm_prefers_rayon_rowmajor_f32_kernel() -> bool {
    true
}

#[pyfunction]
pub fn rust_eigh_lapack_backend() -> String {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        rust_eigh_lapack_backend_tag().to_string()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "unsupported".to_string()
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) fn rust_eigh_lapack_backend_tag() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        if should_try_openblas_lapack_on_macos() && openblas_lapack_dyn().is_some() {
            return "openblas_dyn";
        }
        return match selected_sgemm_backend() {
            SgemmBackend::OpenBlas => "openblas",
            SgemmBackend::Accelerate => "accelerate",
            SgemmBackend::Blas => "blas",
            SgemmBackend::Rust => "rust",
        };
    }
    #[cfg(target_os = "linux")]
    {
        return match selected_sgemm_backend() {
            SgemmBackend::OpenBlas => "openblas",
            SgemmBackend::Blas => "blas",
            SgemmBackend::Rust => "rust",
            SgemmBackend::Accelerate => "accelerate",
        };
    }
    #[cfg(target_os = "windows")]
    {
        return match selected_sgemm_backend() {
            SgemmBackend::OpenBlas => "openblas",
            SgemmBackend::Rust => "rust",
            SgemmBackend::Accelerate => "accelerate",
            SgemmBackend::Blas => "blas",
        };
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn apply_blas_thread_env_hints(threads: usize) {
    let t = threads.max(1).to_string();
    for key in [
        "OMP_NUM_THREADS",
        "OMP_MAX_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OPENBLAS_MAX_THREADS",
        "MKL_NUM_THREADS",
        "MKL_MAX_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "JX_MLM_BLAS_THREADS",
    ] {
        std::env::set_var(key, &t);
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
type SavedEnvVar = (&'static str, Option<OsString>);

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const BLAS_THREAD_ENV_HINT_KEYS: [&str; 10] = [
    "OMP_NUM_THREADS",
    "OMP_MAX_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_MAX_THREADS",
    "MKL_NUM_THREADS",
    "MKL_MAX_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "JX_MLM_BLAS_THREADS",
];

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn capture_blas_thread_env_hints() -> Vec<SavedEnvVar> {
    BLAS_THREAD_ENV_HINT_KEYS
        .iter()
        .map(|&key| (key, std::env::var_os(key)))
        .collect()
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
fn restore_blas_thread_env_hints(saved: &[SavedEnvVar]) {
    for (key, val) in saved.iter() {
        if let Some(v) = val {
            std::env::set_var(key, v);
        } else {
            std::env::remove_var(key);
        }
    }
}

#[cfg(target_os = "macos")]
#[inline]
fn accelerate_blas_set_threading_mode(multithreaded: bool) -> Option<bool> {
    type BlasSetThreadingFn = unsafe extern "C" fn(std::os::raw::c_uint) -> std::os::raw::c_int;
    const BLAS_THREADING_MULTI_THREADED: std::os::raw::c_uint = 0;
    const BLAS_THREADING_SINGLE_THREADED: std::os::raw::c_uint = 1;
    let sym = b"BLASSetThreading\0";
    unsafe {
        let fp = libc::dlsym(
            libc::RTLD_DEFAULT,
            sym.as_ptr() as *const std::os::raw::c_char,
        );
        if fp.is_null() {
            return None;
        }
        let f: BlasSetThreadingFn = std::mem::transmute(fp);
        let mode = if multithreaded {
            BLAS_THREADING_MULTI_THREADED
        } else {
            BLAS_THREADING_SINGLE_THREADED
        };
        Some(f(mode) == 0)
    }
}

#[cfg(not(target_os = "macos"))]
#[inline]
fn accelerate_blas_set_threading_mode(_multithreaded: bool) -> Option<bool> {
    None
}

#[cfg(target_os = "macos")]
#[inline]
fn accelerate_blas_get_threading_mode() -> Option<isize> {
    type BlasGetThreadingFn = unsafe extern "C" fn() -> std::os::raw::c_uint;
    // See Accelerate vecLib thread_api.h:
    // 0 => BLAS_THREADING_MULTI_THREADED, 1 => BLAS_THREADING_SINGLE_THREADED.
    const BLAS_THREADING_SINGLE_THREADED: std::os::raw::c_uint = 1;
    let sym = b"BLASGetThreading\0";
    unsafe {
        let fp = libc::dlsym(
            libc::RTLD_DEFAULT,
            sym.as_ptr() as *const std::os::raw::c_char,
        );
        if fp.is_null() {
            return None;
        }
        let f: BlasGetThreadingFn = std::mem::transmute(fp);
        let mode = f();
        if mode == BLAS_THREADING_SINGLE_THREADED {
            Some(1_isize)
        } else {
            // For multi-threaded mode, thread count is decided internally by
            // Accelerate and capped by VECLIB_MAXIMUM_THREADS when set.
            std::env::var("VECLIB_MAXIMUM_THREADS")
                .ok()
                .and_then(|s| s.trim().parse::<isize>().ok())
                .filter(|v| *v > 0)
                .or(Some(-1))
        }
    }
}

#[cfg(not(target_os = "macos"))]
#[inline]
fn accelerate_blas_get_threading_mode() -> Option<isize> {
    None
}

#[cfg(target_os = "macos")]
type OpenBlasLapackDsyevdFn = unsafe extern "C" fn(
    jobz: *const std::os::raw::c_char,
    uplo: *const std::os::raw::c_char,
    n: *const CblasInt,
    a: *mut f64,
    lda: *const CblasInt,
    w: *mut f64,
    work: *mut f64,
    lwork: *const CblasInt,
    iwork: *mut CblasInt,
    liwork: *const CblasInt,
    info: *mut CblasInt,
);

#[cfg(target_os = "macos")]
type OpenBlasLapackDsyevrFn = unsafe extern "C" fn(
    jobz: *const std::os::raw::c_char,
    range: *const std::os::raw::c_char,
    uplo: *const std::os::raw::c_char,
    n: *const CblasInt,
    a: *mut f64,
    lda: *const CblasInt,
    vl: *const f64,
    vu: *const f64,
    il: *const CblasInt,
    iu: *const CblasInt,
    abstol: *const f64,
    m: *mut CblasInt,
    w: *mut f64,
    z: *mut f64,
    ldz: *const CblasInt,
    isuppz: *mut CblasInt,
    work: *mut f64,
    lwork: *const CblasInt,
    iwork: *mut CblasInt,
    liwork: *const CblasInt,
    info: *mut CblasInt,
);

#[cfg(target_os = "macos")]
type OpenBlasSetThreadsFn = unsafe extern "C" fn(CblasInt);

#[cfg(target_os = "macos")]
type OpenBlasGetThreadsFn = unsafe extern "C" fn() -> CblasInt;

#[cfg(target_os = "macos")]
type OmpSetNumThreadsFn = unsafe extern "C" fn(std::os::raw::c_int);

#[cfg(target_os = "macos")]
type OmpGetMaxThreadsFn = unsafe extern "C" fn() -> std::os::raw::c_int;

#[cfg(target_os = "macos")]
struct OpenBlasLapackDyn {
    #[allow(dead_code)]
    handle: usize,
    dsyevd: OpenBlasLapackDsyevdFn,
    dsyevr: OpenBlasLapackDsyevrFn,
    set_threads: Option<OpenBlasSetThreadsFn>,
    get_threads: Option<OpenBlasGetThreadsFn>,
    omp_set_threads: Option<OmpSetNumThreadsFn>,
    omp_get_max_threads: Option<OmpGetMaxThreadsFn>,
}

#[cfg(target_os = "macos")]
static OPENBLAS_LAPACK_DYN: OnceLock<Option<OpenBlasLapackDyn>> = OnceLock::new();
#[cfg(target_os = "macos")]
static EIGH_OPENBLAS_DYN_THREAD_HINT: AtomicUsize = AtomicUsize::new(0);

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn _dyld_image_count() -> u32;
    fn _dyld_get_image_name(image_index: u32) -> *const std::os::raw::c_char;
}

#[cfg(target_os = "macos")]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MacEighLapackPref {
    Auto,
    Accelerate,
    OpenBlas,
}

#[cfg(target_os = "macos")]
#[inline]
fn mac_eigh_lapack_pref() -> MacEighLapackPref {
    let raw = std::env::var("JX_RUST_EIGH_LAPACK_BACKEND")
        .ok()
        .or_else(|| std::env::var("JX_RUST_LAPACK_BACKEND").ok())
        .unwrap_or_else(|| "auto".to_string())
        .trim()
        .to_ascii_lowercase();
    match raw.as_str() {
        "accelerate" | "veclib" => MacEighLapackPref::Accelerate,
        "openblas" => MacEighLapackPref::OpenBlas,
        _ => MacEighLapackPref::Auto,
    }
}

#[cfg(target_os = "macos")]
#[inline]
fn should_try_openblas_lapack_on_macos() -> bool {
    // A later optional import can introduce another libomp even after the
    // initial backend prewarm. Recheck before using the dynamic LAPACK path:
    // mixed OpenMP worker/barrier state can segfault inside either runtime.
    if macos_has_multiple_openmp_runtimes_loaded() {
        return false;
    }
    match mac_eigh_lapack_pref() {
        MacEighLapackPref::OpenBlas => true,
        MacEighLapackPref::Accelerate => false,
        // Auto policy on macOS:
        // - Reuse an already-loaded OpenBLAS runtime when one is present
        //   (for example NumPy/Conda already loaded it for this process).
        // - Otherwise prefer bundled wheel-local OpenBLAS LAPACK when safe.
        // - If a foreign OpenMP runtime is already loaded but no OpenBLAS is
        //   available to reuse, fall back to Accelerate to avoid mixed-
        //   runtime crashes.
        MacEighLapackPref::Auto => {
            if macos_has_loaded_openblas_runtime() {
                return true;
            }
            !macos_has_foreign_openmp_runtime_loaded()
        }
    }
}

#[cfg(target_os = "macos")]
#[inline]
fn preferred_openblas_thread_cap() -> usize {
    let hinted = EIGH_OPENBLAS_DYN_THREAD_HINT.load(Ordering::SeqCst);
    if hinted > 0 {
        return hinted;
    }
    for key in [
        "JX_MLM_BLAS_THREADS",
        "OPENBLAS_NUM_THREADS",
        "JX_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ] {
        if let Ok(raw) = std::env::var(key) {
            if let Ok(v) = raw.trim().parse::<usize>() {
                if v > 0 {
                    return v;
                }
            }
        }
    }
    std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
}

#[cfg(target_os = "macos")]
#[inline]
fn push_unique_candidate(out: &mut Vec<String>, path: String) {
    if path.is_empty() {
        return;
    }
    if !out.iter().any(|v| v == &path) {
        out.push(path);
    }
}

#[cfg(target_os = "macos")]
#[inline]
fn push_openblas_candidates_in_dir(out: &mut Vec<String>, dir: &Path) {
    if !(dir.exists() && dir.is_dir()) {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut preferred = Vec::<String>::new();
    let mut others = Vec::<String>::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !(path.exists() && path.is_file()) {
            continue;
        }
        let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        let lname = name.to_ascii_lowercase();
        if !lname.starts_with("libopenblas") || !lname.ends_with(".dylib") {
            continue;
        }
        let path_s = path.to_string_lossy().to_string();
        if lname == "libopenblas.0.dylib" || lname == "libopenblas.dylib" {
            preferred.push(path_s);
        } else {
            others.push(path_s);
        }
    }
    preferred.sort_unstable();
    others.sort_unstable();
    for p in preferred.into_iter().chain(others.into_iter()) {
        push_unique_candidate(out, p);
    }
}

#[cfg(target_os = "macos")]
fn extension_module_dir() -> Option<PathBuf> {
    unsafe {
        let mut info: libc::Dl_info = std::mem::zeroed();
        let sym = rust_sgemm_backend as *const () as *const libc::c_void;
        if libc::dladdr(sym, &mut info as *mut libc::Dl_info) == 0 || info.dli_fname.is_null() {
            return None;
        }
        let c_path = std::ffi::CStr::from_ptr(info.dli_fname);
        let path = PathBuf::from(c_path.to_string_lossy().to_string());
        path.parent().map(|p| p.to_path_buf())
    }
}

#[cfg(target_os = "macos")]
fn macos_openblas_bundle_dirs() -> Vec<PathBuf> {
    let mut out = Vec::<PathBuf>::new();
    if let Some(mod_dir) = extension_module_dir() {
        for rel in [".dylibs", ".libs", ""] {
            let probe = if rel.is_empty() {
                mod_dir.clone()
            } else {
                mod_dir.join(rel)
            };
            if probe.exists() && probe.is_dir() && !out.iter().any(|p| p == &probe) {
                out.push(probe);
            }
        }
        if let Some(parent) = mod_dir.parent() {
            let sib = parent.join("janusx.libs");
            if sib.exists() && sib.is_dir() && !out.iter().any(|p| p == &sib) {
                out.push(sib);
            }
        }
    }
    out
}

#[cfg(target_os = "macos")]
#[inline]
fn macos_is_openmp_runtime_name(name: &str) -> bool {
    let low = name.to_ascii_lowercase();
    low.starts_with("libomp") || low.starts_with("libgomp") || low.starts_with("libiomp")
}

#[cfg(target_os = "macos")]
fn macos_has_multiple_openmp_runtimes_loaded() -> bool {
    let mut seen = Vec::<PathBuf>::new();
    for path in macos_loaded_image_paths() {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !macos_is_openmp_runtime_name(name) {
            continue;
        }
        let canonical = std::fs::canonicalize(&path).unwrap_or(path);
        if !seen.contains(&canonical) {
            seen.push(canonical);
            if seen.len() > 1 {
                return true;
            }
        }
    }
    false
}

#[cfg(target_os = "macos")]
#[inline]
fn macos_is_openblas_runtime_name(name: &str) -> bool {
    let low = name.to_ascii_lowercase();
    low.ends_with(".dylib") && low.contains("openblas")
}

#[cfg(target_os = "macos")]
fn macos_loaded_image_paths() -> Vec<PathBuf> {
    let mut out = Vec::<PathBuf>::new();
    let nimg = unsafe { _dyld_image_count() };
    for i in 0..nimg {
        let p = unsafe { _dyld_get_image_name(i) };
        if p.is_null() {
            continue;
        }
        let raw = unsafe { std::ffi::CStr::from_ptr(p) };
        let path = PathBuf::from(raw.to_string_lossy().to_string());
        if !out.iter().any(|v| v == &path) {
            out.push(path);
        }
    }
    out
}

#[cfg(target_os = "macos")]
fn macos_loaded_openblas_candidates() -> Vec<String> {
    let mut out = Vec::<String>::new();
    for path in macos_loaded_image_paths().into_iter() {
        let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        if !macos_is_openblas_runtime_name(name) {
            continue;
        }
        let path_s = path.to_string_lossy().to_string();
        if !out.iter().any(|v| v == &path_s) {
            out.push(path_s);
        }
    }
    out
}

#[cfg(target_os = "macos")]
#[inline]
fn macos_has_loaded_openblas_runtime() -> bool {
    !macos_loaded_openblas_candidates().is_empty()
}

#[cfg(target_os = "macos")]
fn macos_has_foreign_openmp_runtime_loaded() -> bool {
    let bundle_dirs = macos_openblas_bundle_dirs();
    if bundle_dirs.is_empty() {
        return false;
    }

    for path in macos_loaded_image_paths().into_iter() {
        let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        if !macos_is_openmp_runtime_name(name) {
            continue;
        }
        if !bundle_dirs.iter().any(|root| path.starts_with(root)) {
            return true;
        }
    }
    false
}

#[cfg(target_os = "macos")]
fn openblas_lapack_candidates(pref: MacEighLapackPref) -> Vec<String> {
    let mut out = Vec::<String>::new();

    // 0) Prefer already-loaded OpenBLAS first in auto mode. This reuses the
    //    process-local runtime and avoids dlopen() introducing a second
    //    OpenMP runtime when NumPy/Conda has already loaded one.
    if matches!(pref, MacEighLapackPref::Auto) {
        for cand in macos_loaded_openblas_candidates() {
            push_unique_candidate(&mut out, cand);
        }
    }

    // 1) Explicit user-provided locations (file or directory).
    for key in ["JX_OPENBLAS_LIB_PATH", "OPENBLAS_LIB_PATH"] {
        if let Ok(v) = std::env::var(key) {
            let s = v.trim();
            if !s.is_empty() {
                let p = Path::new(s);
                if p.exists() && p.is_dir() {
                    push_openblas_candidates_in_dir(&mut out, p);
                } else {
                    push_unique_candidate(&mut out, s.to_string());
                }
            }
        }
    }

    // Explicit OpenBLAS mode should still consider already-loaded runtimes
    // after honoring user-provided paths.
    if matches!(pref, MacEighLapackPref::OpenBlas) {
        for cand in macos_loaded_openblas_candidates() {
            push_unique_candidate(&mut out, cand);
        }
    }

    // 2) Wheel-local candidates:
    //    janusx/.dylibs, janusx/.libs, and sibling janusx.libs.
    if let Some(mod_dir) = extension_module_dir() {
        for rel in [".dylibs", ".libs", ""] {
            let probe_dir = if rel.is_empty() {
                mod_dir.clone()
            } else {
                mod_dir.join(rel)
            };
            push_openblas_candidates_in_dir(&mut out, &probe_dir);
        }
        if let Some(parent) = mod_dir.parent() {
            push_openblas_candidates_in_dir(&mut out, &parent.join("janusx.libs"));
        }
    }

    // 3) Host-system libraries require an explicit OpenBLAS backend request.
    // Auto must not pull a second Conda/Homebrew OpenMP stack into an isolated
    // Python environment. Without a reusable/bundled runtime use Accelerate.
    if matches!(pref, MacEighLapackPref::OpenBlas) {
        if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
            let base = Path::new(&prefix).join("lib");
            for leaf in [
                "libopenblas.dylib",
                "libopenblas.0.dylib",
                "libopenblasp-r0.3.32.dylib",
                "libopenblas_armv8p-r0.3.32.dylib",
                "libopenblas_vortexp-r0.3.32.dylib",
            ] {
                let p = base.join(leaf);
                if p.exists() {
                    push_unique_candidate(&mut out, p.to_string_lossy().to_string());
                }
            }
        }
        for p in [
            "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
            "/opt/homebrew/opt/openblas/lib/libopenblas.0.dylib",
            "/usr/local/opt/openblas/lib/libopenblas.dylib",
            "/usr/local/opt/openblas/lib/libopenblas.0.dylib",
            "libopenblas.dylib",
            "libopenblas.0.dylib",
        ] {
            push_unique_candidate(&mut out, p.to_string());
        }
    }
    out
}

#[cfg(target_os = "macos")]
unsafe fn openblas_lapack_dyn_load_once() -> Option<OpenBlasLapackDyn> {
    let pref = mac_eigh_lapack_pref();
    let mut handle: *mut libc::c_void = std::ptr::null_mut();
    for cand in openblas_lapack_candidates(pref).into_iter() {
        let Ok(cpath) = std::ffi::CString::new(cand.as_bytes()) else {
            continue;
        };
        let h = libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if !h.is_null() {
            handle = h;
            break;
        }
    }
    if handle.is_null() {
        return None;
    }

    let d_dsyevd = libc::dlsym(handle, b"dsyevd_\0".as_ptr() as *const std::os::raw::c_char);
    let d_dsyevr = libc::dlsym(handle, b"dsyevr_\0".as_ptr() as *const std::os::raw::c_char);
    if d_dsyevd.is_null() || d_dsyevr.is_null() {
        let _ = libc::dlclose(handle);
        return None;
    }

    let d_set = libc::dlsym(
        handle,
        b"openblas_set_num_threads\0".as_ptr() as *const std::os::raw::c_char,
    );
    let d_set_alt = libc::dlsym(
        handle,
        b"openblas_set_num_threads_\0".as_ptr() as *const std::os::raw::c_char,
    );
    let d_get = libc::dlsym(
        handle,
        b"openblas_get_num_threads\0".as_ptr() as *const std::os::raw::c_char,
    );
    let d_get_alt = libc::dlsym(
        handle,
        b"openblas_get_num_threads_\0".as_ptr() as *const std::os::raw::c_char,
    );
    let d_omp_set = libc::dlsym(
        handle,
        b"omp_set_num_threads\0".as_ptr() as *const std::os::raw::c_char,
    );
    let d_omp_get_max = libc::dlsym(
        handle,
        b"omp_get_max_threads\0".as_ptr() as *const std::os::raw::c_char,
    );
    let set_threads = if !d_set.is_null() {
        Some(std::mem::transmute::<*mut libc::c_void, OpenBlasSetThreadsFn>(d_set))
    } else if !d_set_alt.is_null() {
        Some(std::mem::transmute::<*mut libc::c_void, OpenBlasSetThreadsFn>(d_set_alt))
    } else {
        None
    };
    let get_threads = if !d_get.is_null() {
        Some(std::mem::transmute::<*mut libc::c_void, OpenBlasGetThreadsFn>(d_get))
    } else if !d_get_alt.is_null() {
        Some(std::mem::transmute::<*mut libc::c_void, OpenBlasGetThreadsFn>(d_get_alt))
    } else {
        None
    };
    let omp_set_threads = if !d_omp_set.is_null() {
        Some(std::mem::transmute::<*mut libc::c_void, OmpSetNumThreadsFn>(d_omp_set))
    } else {
        None
    };
    let omp_get_max_threads = if !d_omp_get_max.is_null() {
        Some(std::mem::transmute::<*mut libc::c_void, OmpGetMaxThreadsFn>(d_omp_get_max))
    } else {
        None
    };

    Some(OpenBlasLapackDyn {
        handle: handle as usize,
        dsyevd: std::mem::transmute::<*mut libc::c_void, OpenBlasLapackDsyevdFn>(d_dsyevd),
        dsyevr: std::mem::transmute::<*mut libc::c_void, OpenBlasLapackDsyevrFn>(d_dsyevr),
        set_threads,
        get_threads,
        omp_set_threads,
        omp_get_max_threads,
    })
}

#[cfg(target_os = "macos")]
#[inline]
fn openblas_lapack_dyn() -> Option<&'static OpenBlasLapackDyn> {
    OPENBLAS_LAPACK_DYN
        .get_or_init(|| unsafe { openblas_lapack_dyn_load_once() })
        .as_ref()
}

#[cfg(target_os = "macos")]
#[inline]
pub(crate) fn prewarm_eigh_openblas_dyn_runtime() {
    if should_try_openblas_lapack_on_macos() {
        let _ = openblas_lapack_dyn();
    }
}

#[cfg(target_os = "macos")]
#[inline]
pub(crate) fn macos_eigh_uses_openblas_dyn() -> bool {
    should_try_openblas_lapack_on_macos() && openblas_lapack_dyn().is_some()
}

#[cfg(target_os = "macos")]
#[inline]
fn macos_openblas_dyn_current_threads() -> isize {
    let Some(ob) = openblas_lapack_dyn() else {
        return -1;
    };
    if let Some(getter) = ob.get_threads {
        let v = unsafe { getter() };
        if v > 0 {
            return v as isize;
        }
    }
    if let Some(getter) = ob.omp_get_max_threads {
        let v = unsafe { getter() };
        if v > 0 {
            return v as isize;
        }
    }
    -1
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
pub(crate) fn with_eigh_thread_stage<T, F>(threads: usize, f: F) -> (isize, isize, T, isize)
where
    F: FnOnce() -> T,
{
    #[cfg(target_os = "macos")]
    {
        prewarm_eigh_openblas_dyn_runtime();
        if macos_eigh_uses_openblas_dyn() {
            let before = macos_openblas_dyn_current_threads();
            let new_hint = if threads > 0 { threads.max(1) } else { 0 };
            let prev_hint = EIGH_OPENBLAS_DYN_THREAD_HINT.swap(new_hint, Ordering::SeqCst);
            let in_stage = if threads > 0 {
                threads as isize
            } else {
                before
            };
            let out = f();
            EIGH_OPENBLAS_DYN_THREAD_HINT.store(prev_hint, Ordering::SeqCst);
            let after = macos_openblas_dyn_current_threads();
            return (before, in_stage, out, after);
        }
    }

    let before = rust_blas_get_num_threads();
    let (in_stage, out, after) = {
        let _guard = BlasThreadGuard::enter(threads);
        let in_stage = rust_blas_get_num_threads();
        let out = f();
        let after = rust_blas_get_num_threads();
        (in_stage, out, after)
    };
    (before, in_stage, out, after)
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
#[inline]
pub(crate) fn with_eigh_thread_stage<T, F>(_threads: usize, f: F) -> (isize, isize, T, isize)
where
    F: FnOnce() -> T,
{
    let out = f();
    (-1, -1, out, -1)
}

#[cfg(target_os = "macos")]
#[inline]
unsafe fn openblas_lapack_dyn_threads_guard(
    ob: &OpenBlasLapackDyn,
    target_threads: usize,
) -> (Option<usize>, Option<usize>) {
    let prev = ob.get_threads.and_then(|getter| {
        let v = getter();
        if v > 0 {
            Some(v as usize)
        } else {
            None
        }
    });
    let prev_omp = ob.omp_get_max_threads.and_then(|getter| {
        let v = getter();
        if v > 0 {
            Some(v as usize)
        } else {
            None
        }
    });
    if let Some(setter) = ob.omp_set_threads {
        setter(target_threads.max(1).min(i32::MAX as usize) as std::os::raw::c_int);
    }
    if let Some(setter) = ob.set_threads {
        setter(target_threads.max(1).min(i32::MAX as usize) as CblasInt);
    }
    (prev, prev_omp)
}

#[cfg(target_os = "macos")]
#[inline]
unsafe fn openblas_lapack_dyn_threads_restore(
    ob: &OpenBlasLapackDyn,
    prev_threads: Option<usize>,
    prev_omp_threads: Option<usize>,
) {
    if let (Some(prev), Some(setter)) = (prev_omp_threads, ob.omp_set_threads) {
        setter(prev.min(i32::MAX as usize) as std::os::raw::c_int);
    }
    if let (Some(prev), Some(setter)) = (prev_threads, ob.set_threads) {
        setter(prev.min(i32::MAX as usize) as CblasInt);
    }
}

#[cfg(all(feature = "blas-openblas", jx_openblas_available))]
#[inline]
fn rust_openblas_set_threads_impl(threads: usize) -> bool {
    unsafe {
        let t = (threads.max(1)).min(i32::MAX as usize) as CblasInt;
        openblas_set_num_threads(t);
    }
    true
}

#[cfg(not(all(feature = "blas-openblas", jx_openblas_available)))]
#[inline]
fn rust_openblas_set_threads_impl(_threads: usize) -> bool {
    false
}

#[cfg(all(feature = "blas-openblas", jx_openblas_available))]
#[inline]
fn rust_openblas_get_threads_impl() -> Option<usize> {
    unsafe {
        let v = openblas_get_num_threads();
        if v > 0 {
            return Some(v as usize);
        }
    }
    None
}

#[cfg(not(all(feature = "blas-openblas", jx_openblas_available)))]
#[inline]
fn rust_openblas_get_threads_impl() -> Option<usize> {
    None
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) struct OpenBlasThreadGuard {
    prev_threads: Option<usize>,
    active: bool,
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
impl OpenBlasThreadGuard {
    #[inline]
    pub(crate) fn enter(target_threads: usize) -> Self {
        let prev = rust_openblas_get_threads_impl();
        let mut active = false;
        if target_threads > 0 && rust_openblas_set_threads_impl(target_threads) {
            active = true;
        }
        Self {
            prev_threads: prev,
            active,
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
impl Drop for OpenBlasThreadGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Some(prev) = self.prev_threads {
            let _ = rust_openblas_set_threads_impl(prev);
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
enum BlasThreadState {
    OpenBlas { prev_threads: Option<usize> },
    Accelerate { prev_mode: Option<isize> },
    Other,
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) struct BlasThreadGuard {
    saved_env: Vec<SavedEnvVar>,
    state: BlasThreadState,
    active: bool,
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
impl BlasThreadGuard {
    #[inline]
    pub(crate) fn enter(target_threads: usize) -> Self {
        let saved_env = capture_blas_thread_env_hints();
        if target_threads == 0 {
            return Self {
                saved_env,
                state: BlasThreadState::Other,
                active: false,
            };
        }

        let backend = selected_sgemm_backend();
        let state = match backend {
            SgemmBackend::OpenBlas => BlasThreadState::OpenBlas {
                prev_threads: rust_openblas_get_threads_impl(),
            },
            SgemmBackend::Accelerate => BlasThreadState::Accelerate {
                prev_mode: accelerate_blas_get_threading_mode(),
            },
            _ => BlasThreadState::Other,
        };

        let t = target_threads.max(1);
        apply_blas_thread_env_hints(t);
        match backend {
            SgemmBackend::OpenBlas => {
                let _ = rust_openblas_set_threads_impl(t);
            }
            SgemmBackend::Accelerate => {
                let _ = accelerate_blas_set_threading_mode(t > 1);
            }
            _ => {}
        }

        Self {
            saved_env,
            state,
            active: true,
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
impl Drop for BlasThreadGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        restore_blas_thread_env_hints(&self.saved_env);
        match self.state {
            BlasThreadState::OpenBlas { prev_threads } => {
                if let Some(prev) = prev_threads {
                    let _ = rust_openblas_set_threads_impl(prev);
                }
            }
            BlasThreadState::Accelerate { prev_mode } => match prev_mode {
                Some(1) => {
                    let _ = accelerate_blas_set_threading_mode(false);
                }
                Some(_) => {
                    let _ = accelerate_blas_set_threading_mode(true);
                }
                None => {}
            },
            BlasThreadState::Other => {}
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub(crate) struct OpenBlasThreadGuard;

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
impl OpenBlasThreadGuard {
    #[inline]
    pub(crate) fn enter(_target_threads: usize) -> Self {
        Self
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub(crate) struct BlasThreadGuard;

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
impl BlasThreadGuard {
    #[inline]
    pub(crate) fn enter(_target_threads: usize) -> Self {
        Self
    }
}

#[pyfunction]
#[pyo3(signature = (threads))]
pub fn rust_blas_set_num_threads(threads: usize) -> bool {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        let t = threads.max(1);
        apply_blas_thread_env_hints(t);
        let backend = selected_sgemm_backend();
        if backend == SgemmBackend::OpenBlas {
            return rust_openblas_set_threads_impl(t);
        }
        if backend == SgemmBackend::Accelerate {
            if let Some(ok) = accelerate_blas_set_threading_mode(t > 1) {
                return ok;
            }
            // API unavailable on older macOS; env hints remain the fallback.
            return true;
        }
        return false;
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = threads;
        false
    }
}

#[pyfunction]
pub fn rust_blas_get_num_threads() -> isize {
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        let backend = selected_sgemm_backend();
        if backend == SgemmBackend::OpenBlas {
            return rust_openblas_get_threads_impl()
                .map(|v| v as isize)
                .unwrap_or(-1);
        }
        if backend == SgemmBackend::Accelerate {
            if let Some(v) = accelerate_blas_get_threading_mode() {
                return v;
            }
            return std::env::var("VECLIB_MAXIMUM_THREADS")
                .ok()
                .and_then(|s| s.trim().parse::<isize>().ok())
                .filter(|x| *x > 0)
                .unwrap_or(-1);
        }
        return -1;
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        -1
    }
}

#[cfg(all(
    test,
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod tests {
    use super::*;

    #[cfg(all(target_os = "linux", feature = "blas-openblas"))]
    #[test]
    fn linux_openblas_feature_does_not_use_rust_fallback() {
        assert_ne!(rust_sgemm_backend_tag(), "rust");
    }

    fn matrix_storage(matrix: &[Vec<f64>], order: CblasInt) -> Vec<f64> {
        let rows = matrix.len();
        let columns = matrix[0].len();
        if order == CBLAS_ROW_MAJOR {
            matrix.iter().flat_map(|row| row.iter().copied()).collect()
        } else {
            (0..columns)
                .flat_map(|column| (0..rows).map(move |row| matrix[row][column]))
                .collect()
        }
    }

    fn padded_matrix_storage(
        matrix: &[Vec<f64>],
        order: CblasInt,
        leading_dimension: usize,
    ) -> Vec<f64> {
        let rows = matrix.len();
        let columns = matrix[0].len();
        let span = checked_dgemm_matrix_span(order, rows, columns, leading_dimension).unwrap();
        let mut storage = vec![f64::NAN; span];
        for row in 0..rows {
            for column in 0..columns {
                let index = if order == CBLAS_ROW_MAJOR {
                    row * leading_dimension + column
                } else {
                    row + column * leading_dimension
                };
                storage[index] = matrix[row][column];
            }
        }
        storage
    }

    #[test]
    fn checked_dgemm_matrix_span_stops_at_final_accessed_element() {
        assert_eq!(checked_dgemm_matrix_span(CBLAS_ROW_MAJOR, 2, 3, 5), Some(8));
        assert_eq!(
            checked_dgemm_matrix_span(CBLAS_COL_MAJOR, 2, 3, 5),
            Some(12)
        );
        assert_eq!(checked_dgemm_matrix_span(CBLAS_ROW_MAJOR, 2, 3, 2), None);
        assert_eq!(checked_dgemm_matrix_span(CBLAS_COL_MAJOR, 2, 3, 1), None);
        assert_eq!(checked_dgemm_matrix_span(999, 2, 3, 5), None);
        assert_eq!(checked_dgemm_matrix_span(CBLAS_ROW_MAJOR, 0, 3, 5), Some(0));
        assert_eq!(checked_dgemm_matrix_span(CBLAS_COL_MAJOR, 2, 0, 5), Some(0));
    }

    #[test]
    fn rust_dgemm_fallback_supports_padded_leading_dimensions() {
        let m = 2;
        let n = 3;
        let k = 2;
        let logical_a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let logical_b = vec![vec![5.0, 6.0, 7.0], vec![8.0, 9.0, 10.0]];
        let initial_c = vec![vec![0.5, 1.0, 1.5], vec![2.0, 2.5, 3.0]];

        for &order in &[CBLAS_ROW_MAJOR, CBLAS_COL_MAJOR] {
            for &transa in &[CBLAS_NO_TRANS, CBLAS_TRANS] {
                for &transb in &[CBLAS_NO_TRANS, CBLAS_TRANS] {
                    let physical_a = if transa == CBLAS_NO_TRANS {
                        logical_a.clone()
                    } else {
                        (0..k)
                            .map(|row| (0..m).map(|column| logical_a[column][row]).collect())
                            .collect()
                    };
                    let physical_b = if transb == CBLAS_NO_TRANS {
                        logical_b.clone()
                    } else {
                        (0..n)
                            .map(|row| (0..k).map(|column| logical_b[column][row]).collect())
                            .collect()
                    };
                    let lda = 5;
                    let ldb = 6;
                    let ldc = 7;
                    // The backing allocations intentionally end at the last element
                    // addressed by DGEMM, not at ld * physical_rows/columns.
                    let a = padded_matrix_storage(&physical_a, order, lda);
                    let b = padded_matrix_storage(&physical_b, order, ldb);
                    let mut c = padded_matrix_storage(&initial_c, order, ldc);

                    unsafe {
                        cblas_dgemm_rust(
                            order,
                            transa,
                            transb,
                            m as CblasInt,
                            n as CblasInt,
                            k as CblasInt,
                            1.5,
                            a.as_ptr(),
                            lda as CblasInt,
                            b.as_ptr(),
                            ldb as CblasInt,
                            -0.25,
                            c.as_mut_ptr(),
                            ldc as CblasInt,
                        );
                    }

                    for row in 0..m {
                        for column in 0..n {
                            let product = (0..k)
                                .map(|p| logical_a[row][p] * logical_b[p][column])
                                .sum::<f64>();
                            let index = if order == CBLAS_ROW_MAJOR {
                                row * ldc + column
                            } else {
                                row + column * ldc
                            };
                            let expected = 1.5 * product - 0.25 * initial_c[row][column];
                            assert_eq!(c[index], expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rust_dgemm_fallback_alpha_zero_skips_nan_operands_and_honors_beta() {
        let m = 2;
        let n = 2;
        let k = 3;
        let initial_c = vec![vec![1.5, f64::NAN], vec![-2.0, 4.25]];

        for &order in &[CBLAS_ROW_MAJOR, CBLAS_COL_MAJOR] {
            for &transa in &[CBLAS_NO_TRANS, CBLAS_TRANS] {
                for &transb in &[CBLAS_NO_TRANS, CBLAS_TRANS] {
                    let physical_a = if transa == CBLAS_NO_TRANS {
                        vec![vec![f64::NAN; k]; m]
                    } else {
                        vec![vec![f64::NAN; m]; k]
                    };
                    let physical_b = if transb == CBLAS_NO_TRANS {
                        vec![vec![f64::NAN; n]; k]
                    } else {
                        vec![vec![f64::NAN; k]; n]
                    };
                    let lda = 5;
                    let ldb = 6;
                    let ldc = 7;
                    let a = padded_matrix_storage(&physical_a, order, lda);
                    let b = padded_matrix_storage(&physical_b, order, ldb);
                    let mut c = padded_matrix_storage(&initial_c, order, ldc);

                    unsafe {
                        cblas_dgemm_rust(
                            order,
                            transa,
                            transb,
                            m as CblasInt,
                            n as CblasInt,
                            k as CblasInt,
                            0.0,
                            a.as_ptr(),
                            lda as CblasInt,
                            b.as_ptr(),
                            ldb as CblasInt,
                            2.0,
                            c.as_mut_ptr(),
                            ldc as CblasInt,
                        );
                    }

                    let expected = [[3.0, f64::NAN], [-4.0, 8.5]];
                    for row in 0..m {
                        for column in 0..n {
                            let index = if order == CBLAS_ROW_MAJOR {
                                row * ldc + column
                            } else {
                                row + column * ldc
                            };
                            if expected[row][column].is_nan() {
                                assert!(c[index].is_nan());
                            } else {
                                assert_eq!(c[index], expected[row][column]);
                            }
                        }
                    }

                    let mut c =
                        vec![f64::NAN; checked_dgemm_matrix_span(order, m, n, ldc).unwrap()];
                    unsafe {
                        cblas_dgemm_rust(
                            order,
                            transa,
                            transb,
                            m as CblasInt,
                            n as CblasInt,
                            k as CblasInt,
                            0.0,
                            std::ptr::null(),
                            lda as CblasInt,
                            std::ptr::null(),
                            ldb as CblasInt,
                            0.0,
                            c.as_mut_ptr(),
                            ldc as CblasInt,
                        );
                    }
                    for row in 0..m {
                        for column in 0..n {
                            let index = if order == CBLAS_ROW_MAJOR {
                                row * ldc + column
                            } else {
                                row + column * ldc
                            };
                            assert_eq!(c[index], 0.0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rust_dgemm_fallback_k_zero_skips_operands_and_scales_c() {
        let m = 2;
        let n = 3;
        let ldc = 7;
        let initial_c = vec![vec![1.5, f64::NAN, -2.0], vec![4.25, 0.0, 3.0]];

        for &order in &[CBLAS_ROW_MAJOR, CBLAS_COL_MAJOR] {
            let mut c = padded_matrix_storage(&initial_c, order, ldc);
            unsafe {
                cblas_dgemm_rust(
                    order,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    m as CblasInt,
                    n as CblasInt,
                    0,
                    f64::NAN,
                    std::ptr::null(),
                    1,
                    std::ptr::null(),
                    1,
                    -0.5,
                    c.as_mut_ptr(),
                    ldc as CblasInt,
                );
            }

            let expected = [[-0.75, f64::NAN, 1.0], [-2.125, -0.0, -1.5]];
            for row in 0..m {
                for column in 0..n {
                    let index = if order == CBLAS_ROW_MAJOR {
                        row * ldc + column
                    } else {
                        row + column * ldc
                    };
                    if expected[row][column].is_nan() {
                        assert!(c[index].is_nan());
                    } else {
                        assert_eq!(c[index], expected[row][column]);
                    }
                }
            }
        }
    }

    #[test]
    fn rust_dgemm_fallback_zero_dimensions_do_not_dereference_inputs() {
        unsafe {
            cblas_dgemm_rust(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                0,
                2,
                3,
                1.0,
                std::ptr::null(),
                3,
                std::ptr::null(),
                2,
                1.0,
                std::ptr::null_mut(),
                2,
            );
            cblas_dgemm_rust(
                CBLAS_COL_MAJOR,
                CBLAS_TRANS,
                CBLAS_TRANS,
                2,
                0,
                3,
                f64::NAN,
                std::ptr::null(),
                3,
                std::ptr::null(),
                2,
                f64::NAN,
                std::ptr::null_mut(),
                2,
            );
        }
    }

    #[test]
    fn rust_dgemm_fallback_supports_cblas_layouts_transposes_and_scalars() {
        let m = 2;
        let n = 3;
        let k = 2;
        let logical_a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let logical_b = vec![vec![5.0, 6.0, 7.0], vec![8.0, 9.0, 10.0]];
        let initial_c = vec![vec![0.5, 1.0, 1.5], vec![2.0, 2.5, 3.0]];
        let expected = vec![vec![41.75, 47.5, 53.25], vec![93.0, 106.75, 120.5]];

        for &order in &[CBLAS_ROW_MAJOR, CBLAS_COL_MAJOR] {
            for &transa in &[CBLAS_NO_TRANS, CBLAS_TRANS] {
                for &transb in &[CBLAS_NO_TRANS, CBLAS_TRANS] {
                    let physical_a = if transa == CBLAS_NO_TRANS {
                        logical_a.clone()
                    } else {
                        (0..k)
                            .map(|row| (0..m).map(|column| logical_a[column][row]).collect())
                            .collect()
                    };
                    let physical_b = if transb == CBLAS_NO_TRANS {
                        logical_b.clone()
                    } else {
                        (0..n)
                            .map(|row| (0..k).map(|column| logical_b[column][row]).collect())
                            .collect()
                    };
                    let a = matrix_storage(&physical_a, order);
                    let b = matrix_storage(&physical_b, order);
                    let mut c = matrix_storage(&initial_c, order);
                    let lda = if order == CBLAS_ROW_MAJOR {
                        physical_a[0].len()
                    } else {
                        physical_a.len()
                    } as CblasInt;
                    let ldb = if order == CBLAS_ROW_MAJOR {
                        physical_b[0].len()
                    } else {
                        physical_b.len()
                    } as CblasInt;
                    let ldc = if order == CBLAS_ROW_MAJOR { n } else { m } as CblasInt;

                    unsafe {
                        cblas_dgemm_rust(
                            order,
                            transa,
                            transb,
                            m as CblasInt,
                            n as CblasInt,
                            k as CblasInt,
                            2.0,
                            a.as_ptr(),
                            lda,
                            b.as_ptr(),
                            ldb,
                            -0.5,
                            c.as_mut_ptr(),
                            ldc,
                        );
                    }

                    for row in 0..m {
                        for column in 0..n {
                            let actual = if order == CBLAS_ROW_MAJOR {
                                c[row * ldc as usize + column]
                            } else {
                                c[row + column * ldc as usize]
                            };
                            assert!(
                                (actual - expected[row][column]).abs() < 1.0e-12,
                                "order={order}, transa={transa}, transb={transb}, ({row}, {column}): actual={actual:.16e}, expected={:.16e}",
                                expected[row][column]
                            );
                        }
                    }
                }
            }
        }

        let expected_product = vec![vec![21.0, 24.0, 27.0], vec![47.0, 54.0, 61.0]];
        for &order in &[CBLAS_ROW_MAJOR, CBLAS_COL_MAJOR] {
            let a = matrix_storage(&logical_a, order);
            let b = matrix_storage(&logical_b, order);
            let mut c = vec![f64::NAN; m * n];
            let lda = if order == CBLAS_ROW_MAJOR { k } else { m } as CblasInt;
            let ldb = if order == CBLAS_ROW_MAJOR { n } else { k } as CblasInt;
            let ldc = if order == CBLAS_ROW_MAJOR { n } else { m } as CblasInt;

            unsafe {
                cblas_dgemm_rust(
                    order,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    m as CblasInt,
                    n as CblasInt,
                    k as CblasInt,
                    1.0,
                    a.as_ptr(),
                    lda,
                    b.as_ptr(),
                    ldb,
                    0.0,
                    c.as_mut_ptr(),
                    ldc,
                );
            }

            for row in 0..m {
                for column in 0..n {
                    let actual = if order == CBLAS_ROW_MAJOR {
                        c[row * ldc as usize + column]
                    } else {
                        c[row + column * ldc as usize]
                    };
                    assert_eq!(actual, expected_product[row][column]);
                }
            }
        }
    }
}
