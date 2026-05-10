use pyo3::prelude::*;
#[cfg(target_os = "macos")]
use std::path::Path;
use std::sync::OnceLock;

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) type CblasInt = std::os::raw::c_int;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_COL_MAJOR: CblasInt = 102;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_NO_TRANS: CblasInt = 111;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_TRANS: CblasInt = 112;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub(crate) const CBLAS_UPPER: CblasInt = 121;

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
        if HAS_OPENBLAS_BACKEND {
            return SgemmBackend::OpenBlas;
        }
        return SgemmBackend::Accelerate;
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
            #[cfg(any(
                target_os = "windows",
                all(feature = "blas-openblas", jx_openblas_available)
            ))]
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
    #[cfg(all(target_os = "linux", feature = "blas-openblas", jx_openblas_available))]
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
    #[cfg(target_os = "windows")]
    {
        cblas_sgemm_openblas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
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
    assert_eq!(
        order, CBLAS_COL_MAJOR,
        "Rust DGEMM fallback expects column-major order"
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
            c_slice[idx] = alpha * acc + beta * c_slice[idx];
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
#[inline]
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
            #[cfg(any(
                target_os = "windows",
                all(feature = "blas-openblas", jx_openblas_available)
            ))]
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
    #[cfg(all(target_os = "linux", feature = "blas-openblas", jx_openblas_available))]
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
    #[cfg(target_os = "windows")]
    {
        cblas_dgemm_openblas(
            order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
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
            #[cfg(any(
                target_os = "windows",
                all(feature = "blas-openblas", jx_openblas_available)
            ))]
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
    #[cfg(all(target_os = "linux", feature = "blas-openblas", jx_openblas_available))]
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
    #[cfg(target_os = "windows")]
    {
        cblas_ssyrk_openblas(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
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
            #[cfg(any(
                target_os = "windows",
                all(feature = "blas-openblas", jx_openblas_available)
            ))]
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
    #[cfg(all(target_os = "linux", feature = "blas-openblas", jx_openblas_available))]
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
    #[cfg(target_os = "windows")]
    {
        cblas_dsyrk_openblas(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
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
                if let Some(setter) = ob.set_threads {
                    setter(preferred_openblas_thread_cap().min(i32::MAX as usize) as CblasInt);
                }
                (ob.dsyevd)(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
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
                jx_openblas_available
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
        jx_openblas_available
    ))]
    {
        lapack_dsyevd_openblas(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
        return Ok(());
    }
    #[cfg(target_os = "windows")]
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
                if let Some(setter) = ob.set_threads {
                    setter(preferred_openblas_thread_cap().min(i32::MAX as usize) as CblasInt);
                }
                (ob.dsyevr)(
                    jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz,
                    work, lwork, iwork, liwork, info,
                );
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
                jx_openblas_available
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
        jx_openblas_available
    ))]
    {
        lapack_dsyevr_openblas(
            jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work,
            lwork, iwork, liwork, info,
        );
        return Ok(());
    }
    #[cfg(target_os = "windows")]
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
        match selected_sgemm_backend() {
            SgemmBackend::Accelerate => "accelerate".to_string(),
            SgemmBackend::OpenBlas => "openblas".to_string(),
            SgemmBackend::Blas => "blas".to_string(),
            SgemmBackend::Rust => "rust".to_string(),
        }
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
struct OpenBlasLapackDyn {
    #[allow(dead_code)]
    handle: usize,
    dsyevd: OpenBlasLapackDsyevdFn,
    dsyevr: OpenBlasLapackDsyevrFn,
    set_threads: Option<OpenBlasSetThreadsFn>,
}

#[cfg(target_os = "macos")]
static OPENBLAS_LAPACK_DYN: OnceLock<Option<OpenBlasLapackDyn>> = OnceLock::new();

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
    match mac_eigh_lapack_pref() {
        MacEighLapackPref::OpenBlas => true,
        MacEighLapackPref::Accelerate => false,
        MacEighLapackPref::Auto => matches!(selected_sgemm_backend(), SgemmBackend::Accelerate),
    }
}

#[cfg(target_os = "macos")]
#[inline]
fn preferred_openblas_thread_cap() -> usize {
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
fn openblas_lapack_candidates() -> Vec<String> {
    let mut out = Vec::<String>::new();
    for key in ["JX_OPENBLAS_LIB_PATH", "OPENBLAS_LIB_PATH"] {
        if let Ok(v) = std::env::var(key) {
            let s = v.trim();
            if !s.is_empty() {
                push_unique_candidate(&mut out, s.to_string());
            }
        }
    }
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
        "/usr/local/opt/openblas/lib/libopenblas.dylib",
        "libopenblas.dylib",
        "libopenblas.0.dylib",
    ] {
        push_unique_candidate(&mut out, p.to_string());
    }
    out
}

#[cfg(target_os = "macos")]
unsafe fn openblas_lapack_dyn_load_once() -> Option<OpenBlasLapackDyn> {
    let mut handle: *mut libc::c_void = std::ptr::null_mut();
    for cand in openblas_lapack_candidates().into_iter() {
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
    let set_threads = if !d_set.is_null() {
        Some(std::mem::transmute::<*mut libc::c_void, OpenBlasSetThreadsFn>(d_set))
    } else if !d_set_alt.is_null() {
        Some(std::mem::transmute::<*mut libc::c_void, OpenBlasSetThreadsFn>(d_set_alt))
    } else {
        None
    };

    Some(OpenBlasLapackDyn {
        handle: handle as usize,
        dsyevd: std::mem::transmute::<*mut libc::c_void, OpenBlasLapackDsyevdFn>(d_dsyevd),
        dsyevr: std::mem::transmute::<*mut libc::c_void, OpenBlasLapackDsyevrFn>(d_dsyevr),
        set_threads,
    })
}

#[cfg(target_os = "macos")]
#[inline]
fn openblas_lapack_dyn() -> Option<&'static OpenBlasLapackDyn> {
    OPENBLAS_LAPACK_DYN
        .get_or_init(|| unsafe { openblas_lapack_dyn_load_once() })
        .as_ref()
}

#[cfg(any(
    target_os = "windows",
    all(feature = "blas-openblas", jx_openblas_available)
))]
#[inline]
fn rust_openblas_set_threads_impl(threads: usize) -> bool {
    unsafe {
        let t = (threads.max(1)).min(i32::MAX as usize) as CblasInt;
        openblas_set_num_threads(t);
    }
    true
}

#[cfg(not(any(
    target_os = "windows",
    all(feature = "blas-openblas", jx_openblas_available)
)))]
#[inline]
fn rust_openblas_set_threads_impl(_threads: usize) -> bool {
    false
}

#[cfg(any(
    target_os = "windows",
    all(feature = "blas-openblas", jx_openblas_available)
))]
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

#[cfg(not(any(
    target_os = "windows",
    all(feature = "blas-openblas", jx_openblas_available)
)))]
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

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub(crate) struct OpenBlasThreadGuard;

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
impl OpenBlasThreadGuard {
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
